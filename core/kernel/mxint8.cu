// kernel/mxint8.cu  —  plain-MXINT8 baseline kernels.
//
// The matched-optimization baseline for the SSNF kernels: identical structure
// (thread mapping, FP32 accumulate, on-the-fly activation quant, online
// softmax), differing ONLY in the weight/KV read. SSNF does the sub-byte unpack
// (ssnf::unpack_ssnf_weight_elem); MXINT8 reads the int8 mantissa directly. So the
// SSNF-vs-MXINT8 latency delta isolates exactly the unpack overhead vs the
// fewer-bytes-read benefit — the decode-memory-bound question.
//
// Layout (out-innermost SoA, from ssnf_lib.pack.pack_weight_mxint8):
//   scale_exp [nb, OUT] int8           flat: blk*OUT + o
//   qweight   [nb, 32, OUT] int8       flat: (blk*32 + k)*OUT + o
// KV (pack_kv_mxint8): scale_exp [H,nb,L], qweight [H,nb,32,L].
//
// STATUS: GPU-UNVALIDATED until built; logic verified on CPU
// (tests/test_emulation.py MXINT8 cases). Same correctness-first level as the
// SSNF kernels (no tensor cores / split-K yet) so the comparison is fair.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <math.h>
#include "core/ssnf_utils.cuh"

constexpr int BLOCK = 32;                  // one E8M0 scaling block
#include <ATen/cuda/CUDAContext.h>

// Stage-0 activation quant launcher (defined in wa_gemm.cu) — shared so the MXINT8
// W+A path quantizes the activation identically to SSNF (matched comparison).

// ==== COLUMN-MAJOR WIDE-LOAD decode (matched to the SSNF wide-load kernels) =========
//   qweight_cm [nb,OUT,32]: a column's 32 int8 are CONTIGUOUS -> wide-load 32 bytes
//   (2x int4) into registers per block instead of 32 strided 1-byte reads. The narrow
//   path left the kernel latency-bound (ncu: SM/DRAM/L1 all ~35%); the wide load cuts
//   the per-block load count 32->2 and reaches DRAM. Bit-identical (same int8 codes).
__device__ __forceinline__ void mx_load_col32(const int8_t* __restrict__ qcm, long base, int8_t (&wbuf)[32]) {
    *reinterpret_cast<int4*>(wbuf)      = __ldg(reinterpret_cast<const int4*>(qcm + base * 32));
    *reinterpret_cast<int4*>(wbuf + 16) = __ldg(reinterpret_cast<const int4*>(qcm + base * 32 + 16));
}


// dequant MXINT8 weight -> bf16 [K,OUT] (coalesced) for prefill dequant-once + cuBLAS.
__global__ void mxint8_dequant_bf16_kernel(
        const int8_t* __restrict__ scale_exp, const int8_t* __restrict__ qweight_cm,
        __nv_bfloat16* __restrict__ Wbf, int OUT, int K, int NB) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int blk = blockIdx.y;
    if (o >= OUT) return;
    const float scale = ssnf::e8m0_to_scale(scale_exp[blk * OUT + o]);
    int8_t wbuf[32]; mx_load_col32(qweight_cm, (long)blk * OUT + o, wbuf);
    #pragma unroll
    for (int k = 0; k < BLOCK; ++k)
        Wbf[((long)(blk * BLOCK + k)) * OUT + o] = __float2bfloat16((float)wbuf[k] * scale);
}


inline int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

// ---- KV WRITE (MXINT8 baseline), matched to kv_write_kernel ------------------
//   Same thread-per-token structure; quantize 32 head_dim to int8 directly (no
//   bit-pack), store to qweight [H,nb,L,32].
__global__ void mxint8_kv_write_kernel(
        const __nv_bfloat16* __restrict__ X,    // [H, L, D]
        int8_t* __restrict__ scale_exp,          // [H, nb, L]
        int8_t* __restrict__ qweight,            // [H, nb, L, 32]
        int H, int L, int D, int NB) {
    const int h = blockIdx.x;
    const int blk = blockIdx.z;                           // grid-z = nb (matched occupancy fix)
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= L) return;
    const long xb = ((long)h * L + j) * D + (long)blk * BLOCK;
    float amax = 1e-30f;
    #pragma unroll
    for (int k = 0; k < BLOCK; ++k) amax = fmaxf(amax, fabsf(__bfloat162float(X[xb + k])));
    const int ea = ssnf::e8m0_exp_from_amax(amax);
    const float s = exp2f((float)ea);
    const long tok = (long)(h * NB + blk) * L + j;
    #pragma unroll
    for (int k = 0; k < BLOCK; ++k) {
        int q = (int)rintf(__bfloat162float(X[xb + k]) / s);
        qweight[tok * BLOCK + k] = (int8_t)max(-127, min(127, q));
    }
    scale_exp[tok] = (int8_t)ea;
}


// ==== COLUMN-MAJOR WIDE-LOAD hosts (qweight_cm [nb,OUT,32]) — matched to SSNF wide ====
torch::Tensor mxint8_dequant_bf16_cuda(
        torch::Tensor scale_exp, torch::Tensor qweight_cm, int64_t OUT, int64_t K, int64_t NB) {
    auto W = torch::empty({K, OUT}, torch::dtype(torch::kBFloat16).device(scale_exp.device()));
    const int threads = 256, blocks = (int)((OUT + threads - 1) / threads);
    mxint8_dequant_bf16_kernel<<<dim3(blocks, (int)NB), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        scale_exp.data_ptr<int8_t>(), qweight_cm.data_ptr<int8_t>(),
        reinterpret_cast<__nv_bfloat16*>(W.data_ptr<at::BFloat16>()), (int)OUT, (int)K, (int)NB);
    return W;
}


// Defined in wa_gemm.cu ("FAIR MXINT8 twin" of the SSNF skinny host): same adaptive
// NT + split-K + combine dispatch, only the weight format differs. Forward-declared
// here so the small-M dispatch below can route to it (matched to the SSNF path).
torch::Tensor mxint8_gemm_fused_skinny_cuda(
        torch::Tensor X, torch::Tensor scale_exp, torch::Tensor qweight_cm,
        int64_t M, int64_t OUT, int64_t K, int64_t NB);


// MXINT8 KV write launcher -> (scale_exp [H,nb,L], qweight [H,nb,L,32]).
std::vector<torch::Tensor> mxint8_kv_write_cuda(
        torch::Tensor X, int64_t H, int64_t L, int64_t D, int64_t NB) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    auto i8 = X.options().dtype(torch::kInt8);
    auto scale_exp = torch::empty({H, NB, L}, i8);
    auto qweight   = torch::empty({H, NB, L, BLOCK}, i8);
    const int TPB = 128;                                  // matched: grid-z=nb occupancy fix
    mxint8_kv_write_kernel<<<dim3((int)H, ((int)L + TPB - 1) / TPB, (int)NB), TPB, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
        scale_exp.data_ptr<int8_t>(), qweight.data_ptr<int8_t>(), (int)H, (int)L, (int)D, (int)NB);
    return {scale_exp, qweight};
}
