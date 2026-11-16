// kernel/mxfp.cu  —  native MXFP8-E4M3 / MXFP6-E2M3 KV-decode kernels.
//
// The native-FP baselines for the KV channel, matched-optimization to the MXINT8
// KV kernel (kernel/mxint8.cu :: mxint8_kv_split_kernel): IDENTICAL flash-decode
// structure (thread-per-key Pass 1, thread-per-head-dim Pass 2, split-K, online
// softmax, per-32 E8M0 block scale). The ONLY difference is the element read:
//   MXINT8 : (float) int8                 (1 byte, no unpack)
//   MXFP8  : E4M3 hw-cvt of a uint8       (1 byte, fp8 decode)      -> 8.25 b/elem
//   MXFP6  : E2M3 from a 6-bit code       (4-into-3 byte unpack)    -> 6.25 b/elem
// So the MXINT8-vs-MXFP8 delta isolates the fp8 decode, and MXFP6 additionally
// pays the sub-byte (6-bit) unpack — the honest decode-ALU cost of a native FP KV
// cache. Weight stays BF16 (KV channel); the RPS-bench weight path is bf16 cuBLAS.
//
// Layout (mirror of pack_kv_mxint8, token-major, block/kd innermost):
//   MXFP8 : scale_exp int8[H,nb,L]      qcodes uint8[H,nb,L,32]   (E4M3 byte/elem)
//   MXFP6 : scale_exp int8[H,nb,L]      qcodes uint8[H,nb,L,24]   (32x6bit = 24 B/block)

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include "core/ssnf_utils.cuh"
#include <ATen/cuda/CUDAContext.h>

namespace {
constexpr int BLOCK = 32;
constexpr int KV_CHUNK = 128;   // keys per chunk (must match kv_attention.cu / mxint8.cu)

// ---- element decode primitives ---------------------------------------------

inline int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }
// ---- MXFP8-E4M3 KV WRITE (encode) — mirror of mxint8_kv_write_kernel (kernel/mxint8.cu).
//   Same thread-per-token structure; per-32 block E8M0 scale sized to the E4M3 element max
//   (E_MAX=8: e4m3 largest normal = 1.75*2^8 = 448), element = hw float->E4M3. qcodes byte-
//   identical stride to MXINT8 (32 B/block) so the slot layout / blob surgery are shared.
__device__ __forceinline__ int e8m0_exp_from_amax_fp8(float amax) {
    int e = (int)floorf(log2f(fmaxf(amax, 1e-30f))) - 8;   // e4m3 emax = 8
    return max(-127, min(127, e));
}

__device__ __forceinline__ uint8_t f_to_e4m3(float x) {
    return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
__global__ void mxfp8_kv_write_kernel(
        const __nv_bfloat16* __restrict__ X,    // [H, L, D]
        int8_t* __restrict__ scale_exp,          // [H, nb, L]
        uint8_t* __restrict__ qcodes,            // [H, nb, L, 32] (E4M3 bytes)
        int H, int L, int D, int NB) {
    const int h = blockIdx.x;
    const int blk = blockIdx.z;                           // grid-z = nb (occupancy fix)
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= L) return;
    const long xb = ((long)h * L + j) * D + (long)blk * BLOCK;
    float amax = 1e-30f;
    #pragma unroll
    for (int k = 0; k < BLOCK; ++k) amax = fmaxf(amax, fabsf(__bfloat162float(X[xb + k])));
    const int ea = e8m0_exp_from_amax_fp8(amax);
    const float s = exp2f((float)ea);
    const long tok = (long)(h * NB + blk) * L + j;
    #pragma unroll
    for (int k = 0; k < BLOCK; ++k)
        qcodes[tok * BLOCK + k] = f_to_e4m3(__bfloat162float(X[xb + k]) / s);
    scale_exp[tok] = (int8_t)ea;
}


// ---- MXFP6-E2M3 KV WRITE (encode) — 6-bit sub-byte packing (32x6bit = 24 B/block).
//   E2M3: 1 sign | 2 exp | 3 mant, bias=1, max=7.5 -> E8M0 E_MAX=2. Round-to-nearest float->e2m3
//   (no HW instruction), saturate at 7.5. Pack 4 codes per 3-byte little-endian word (matches
//   e2m3_word decode). qcodes stride = 24 B/block.
__device__ __forceinline__ int e8m0_exp_from_amax_e2m3(float amax) {
    int e = (int)floorf(log2f(fmaxf(amax, 1e-30f))) - 2;   // e2m3 emax = 2 (max normal 1.875*2^2 = 7.5)
    return max(-127, min(127, e));
}

__device__ __forceinline__ uint8_t f_to_e2m3(float x) {   // -> 6-bit code sign(1)|exp(2)|mant(3)
    const int sign = signbit(x) ? 0x20 : 0;
    float a = fabsf(x);
    if (a >= 7.5f) return (uint8_t)(sign | 0x1F);          // e=3,m=7 = 7.5 (saturate)
    int e, m;
    if (a < 1.0f) {                                        // subnormal region [0,1): value = m/8
        m = (int)rintf(a * 8.0f);                          // 0..8
        if (m < 8) return (uint8_t)(sign | m);             // e=0
        e = 1; m = 0;                                      // rounded up to 1.0
    } else {                                               // normal: value = 2^(e-1)*(1+m/8)
        e = (int)floorf(log2f(a)) + 1;                     // 1..3
        if (e > 3) e = 3;
        const float base = exp2f((float)(e - 1)), step = exp2f((float)(e - 4));
        m = (int)rintf((a - base) / step);                 // 0..8
        if (m >= 8) { e += 1; m = 0; if (e > 3) { e = 3; m = 7; } }
    }
    return (uint8_t)(sign | (e << 3) | m);
}
__global__ void mxfp6_kv_write_kernel(
        const __nv_bfloat16* __restrict__ X,    // [H, L, D]
        int8_t* __restrict__ scale_exp,          // [H, nb, L]
        uint8_t* __restrict__ qcodes,            // [H, nb, L, 24] (32x6bit packed)
        int H, int L, int D, int NB) {
    const int h = blockIdx.x;
    const int blk = blockIdx.z;
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= L) return;
    const long xb = ((long)h * L + j) * D + (long)blk * BLOCK;
    float amax = 1e-30f;
    #pragma unroll
    for (int k = 0; k < BLOCK; ++k) amax = fmaxf(amax, fabsf(__bfloat162float(X[xb + k])));
    const int ea = e8m0_exp_from_amax_e2m3(amax);
    const float s = exp2f((float)ea);
    const long tok = (long)(h * NB + blk) * L + j;
    uint8_t* out = qcodes + tok * 24;
    #pragma unroll
    for (int g = 0; g < BLOCK / 4; ++g) {                  // 8 groups of 4 codes -> 3 bytes each
        unsigned word = 0;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int code = f_to_e2m3(__bfloat162float(X[xb + g * 4 + w]) / s);
            word |= (unsigned)(code & 0x3F) << (6 * w);
        }
        out[g * 3 + 0] = (uint8_t)(word & 0xff);
        out[g * 3 + 1] = (uint8_t)((word >> 8) & 0xff);
        out[g * 3 + 2] = (uint8_t)((word >> 16) & 0xff);
    }
    scale_exp[tok] = (int8_t)ea;
}


// ---- FUSED append (quantize+pack+scatter in one kernel) — mirror of kv_append_paged_fused_kernel
//   but for the MX 8.25b slot [scale(1) | 32 e4m3 bytes] per block. Writes straight into the paged
//   cache slot (via slot_mapping), byte-IDENTICAL to mxfp8_kv_write + _planes_to_blob + scatter.
__global__ void mxfp8_kv_append_fused_kernel(
        const __nv_bfloat16* __restrict__ key,     // [N, Hk*D]
        const __nv_bfloat16* __restrict__ value,   // [N, Hk*D]
        uint8_t* __restrict__ cache,               // [nslots, Hk, S] contiguous
        const int64_t* __restrict__ slot_mapping,  // [N]
        int N, int Hk, int D, int NB, int per_head, int S) {
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long)N * Hk * NB) return;
    const int blk = (int)(tid % NB);
    const int h   = (int)((tid / NB) % Hk);
    const int n   = (int)(tid / ((long)NB * Hk));
    const long slot = slot_mapping[n];
    if (slot < 0) return;                                       // padded/invalid slot
    const long xb = (long)n * (Hk * D) + (long)h * D + (long)blk * BLOCK;
    uint8_t* dst = cache + ((long)slot * Hk + h) * S + (long)blk * (1 + BLOCK);
    { // K
        float amax = 1e-30f;
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) amax = fmaxf(amax, fabsf(__bfloat162float(key[xb + k])));
        const int ea = e8m0_exp_from_amax_fp8(amax);
        const float s = exp2f((float)ea);
        dst[0] = (uint8_t)(int8_t)ea;
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) dst[1 + k] = f_to_e4m3(__bfloat162float(key[xb + k]) / s);
    }
    { // V (second per_head)
        uint8_t* vd = dst + per_head;
        float amax = 1e-30f;
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) amax = fmaxf(amax, fabsf(__bfloat162float(value[xb + k])));
        const int ea = e8m0_exp_from_amax_fp8(amax);
        const float s = exp2f((float)ea);
        vd[0] = (uint8_t)(int8_t)ea;
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) vd[1 + k] = f_to_e4m3(__bfloat162float(value[xb + k]) / s);
    }
}


// ---- FUSED append for MXINT8 slot [scale(1) | 32 int8 bytes] per block (same slot as MXFP8,
//   int8 codec: rint(x/s) clamped +-127). Byte-IDENTICAL to mxint8_kv_write + blob + scatter.
__global__ void mxint8_kv_append_fused_kernel(
        const __nv_bfloat16* __restrict__ key,
        const __nv_bfloat16* __restrict__ value,
        uint8_t* __restrict__ cache,
        const int64_t* __restrict__ slot_mapping,
        int N, int Hk, int D, int NB, int per_head, int S) {
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long)N * Hk * NB) return;
    const int blk = (int)(tid % NB);
    const int h   = (int)((tid / NB) % Hk);
    const int n   = (int)(tid / ((long)NB * Hk));
    const long slot = slot_mapping[n];
    if (slot < 0) return;
    const long xb = (long)n * (Hk * D) + (long)h * D + (long)blk * BLOCK;
    uint8_t* dst = cache + ((long)slot * Hk + h) * S + (long)blk * (1 + BLOCK);
    for (int pv = 0; pv < 2; ++pv) {                            // pv=0 K, pv=1 V
        const __nv_bfloat16* src = pv ? value : key;
        uint8_t* out = dst + (pv ? per_head : 0);
        float amax = 1e-30f;
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) amax = fmaxf(amax, fabsf(__bfloat162float(src[xb + k])));
        const int ea = ssnf::e8m0_exp_from_amax(amax);
        const float s = exp2f((float)ea);
        out[0] = (uint8_t)(int8_t)ea;
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) {
            int q = (int)rintf(__bfloat162float(src[xb + k]) / s);
            out[1 + k] = (uint8_t)(int8_t)max(-127, min(127, q));
        }
    }
}


// ---- FUSED append for MXFP6-E2M3 slot [scale(1) | 24 packed 6-bit bytes] per block.
__global__ void mxfp6_kv_append_fused_kernel(
        const __nv_bfloat16* __restrict__ key,
        const __nv_bfloat16* __restrict__ value,
        uint8_t* __restrict__ cache,
        const int64_t* __restrict__ slot_mapping,
        int N, int Hk, int D, int NB, int per_head, int S) {
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long)N * Hk * NB) return;
    const int blk = (int)(tid % NB);
    const int h   = (int)((tid / NB) % Hk);
    const int n   = (int)(tid / ((long)NB * Hk));
    const long slot = slot_mapping[n];
    if (slot < 0) return;
    const long xb = (long)n * (Hk * D) + (long)h * D + (long)blk * BLOCK;
    uint8_t* dst = cache + ((long)slot * Hk + h) * S + (long)blk * (1 + 24);
    for (int pv = 0; pv < 2; ++pv) {                             // pv=0 K, pv=1 V
        const __nv_bfloat16* src = pv ? value : key;
        uint8_t* out = dst + (pv ? per_head : 0);
        float amax = 1e-30f;
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) amax = fmaxf(amax, fabsf(__bfloat162float(src[xb + k])));
        const int ea = e8m0_exp_from_amax_e2m3(amax);
        const float s = exp2f((float)ea);
        out[0] = (uint8_t)(int8_t)ea;
        #pragma unroll
        for (int g = 0; g < BLOCK / 4; ++g) {                   // 8 groups of 4 codes -> 3 bytes
            unsigned word = 0;
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                const int code = f_to_e2m3(__bfloat162float(src[xb + g * 4 + w]) / s);
                word |= (unsigned)(code & 0x3F) << (6 * w);
            }
            out[1 + g * 3 + 0] = (uint8_t)(word & 0xff);
            out[1 + g * 3 + 1] = (uint8_t)((word >> 8) & 0xff);
            out[1 + g * 3 + 2] = (uint8_t)((word >> 16) & 0xff);
        }
    }
}

} // namespace

// ---- MXFP8 KV write launcher (mirror of mxint8_kv_write_cuda). qcodes int8 tensor holds the
//   raw E4M3 bytes (bit-reinterpret) so _planes_to_blob_mxint8 (int8 cat) is reused unchanged.
std::vector<torch::Tensor> mxfp8_kv_write_cuda(
        torch::Tensor X, int64_t H, int64_t L, int64_t D, int64_t NB) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    auto i8 = X.options().dtype(torch::kInt8);
    auto scale_exp = torch::empty({H, NB, L}, i8);
    auto qcodes    = torch::empty({H, NB, L, BLOCK}, i8);
    const int TPB = 128;
    mxfp8_kv_write_kernel<<<dim3((int)H, ((int)L + TPB - 1) / TPB, (int)NB), TPB, 0,
                            at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
        scale_exp.data_ptr<int8_t>(),
        reinterpret_cast<uint8_t*>(qcodes.data_ptr<int8_t>()), (int)H, (int)L, (int)D, (int)NB);
    return {scale_exp, qcodes};
}


// ---- MXFP6 KV write launcher. qcodes int8 tensor [H,NB,L,24] holds packed 6-bit e2m3 codes.
std::vector<torch::Tensor> mxfp6_kv_write_cuda(
        torch::Tensor X, int64_t H, int64_t L, int64_t D, int64_t NB) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    auto i8 = X.options().dtype(torch::kInt8);
    auto scale_exp = torch::empty({H, NB, L}, i8);
    auto qcodes    = torch::empty({H, NB, L, 24}, i8);      // 32x6bit packed
    const int TPB = 128;
    mxfp6_kv_write_kernel<<<dim3((int)H, ((int)L + TPB - 1) / TPB, (int)NB), TPB, 0,
                            at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
        scale_exp.data_ptr<int8_t>(),
        reinterpret_cast<uint8_t*>(qcodes.data_ptr<int8_t>()), (int)H, (int)L, (int)D, (int)NB);
    return {scale_exp, qcodes};
}


// ---- FUSED append launchers (deploy write path, matches SSNF's kv_append_paged_fused).
void mxfp8_kv_append_fused_cuda(
        torch::Tensor key, torch::Tensor value, torch::Tensor kv_cache, torch::Tensor slot_mapping,
        int64_t N, int64_t Hk, int64_t D, int64_t NB, int64_t per_head, int64_t S) {
    TORCH_CHECK(key.is_cuda() && key.scalar_type() == torch::kBFloat16, "key must be CUDA bf16");
    TORCH_CHECK(value.scalar_type() == torch::kBFloat16, "value must be bf16");
    TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8, "kv_cache must be uint8");
    TORCH_CHECK(slot_mapping.scalar_type() == torch::kLong, "slot_mapping must be int64");
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous() && kv_cache.is_contiguous(),
                "key/value/kv_cache must be contiguous");
    if (N <= 0) return;
    const long total = N * Hk * NB;
    const int TPB = 128;
    mxfp8_kv_append_fused_kernel<<<(int)((total + TPB - 1) / TPB), TPB, 0,
                                   at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value.data_ptr<at::BFloat16>()),
        kv_cache.data_ptr<uint8_t>(), slot_mapping.data_ptr<int64_t>(),
        (int)N, (int)Hk, (int)D, (int)NB, (int)per_head, (int)S);
}

void mxint8_kv_append_fused_cuda(
        torch::Tensor key, torch::Tensor value, torch::Tensor kv_cache, torch::Tensor slot_mapping,
        int64_t N, int64_t Hk, int64_t D, int64_t NB, int64_t per_head, int64_t S) {
    TORCH_CHECK(key.is_cuda() && key.scalar_type() == torch::kBFloat16, "key must be CUDA bf16");
    TORCH_CHECK(value.scalar_type() == torch::kBFloat16, "value must be bf16");
    TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8, "kv_cache must be uint8");
    TORCH_CHECK(slot_mapping.scalar_type() == torch::kLong, "slot_mapping must be int64");
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous() && kv_cache.is_contiguous(),
                "key/value/kv_cache must be contiguous");
    if (N <= 0) return;
    const long total = N * Hk * NB;
    const int TPB = 128;
    mxint8_kv_append_fused_kernel<<<(int)((total + TPB - 1) / TPB), TPB, 0,
                                    at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value.data_ptr<at::BFloat16>()),
        kv_cache.data_ptr<uint8_t>(), slot_mapping.data_ptr<int64_t>(),
        (int)N, (int)Hk, (int)D, (int)NB, (int)per_head, (int)S);
}

void mxfp6_kv_append_fused_cuda(
        torch::Tensor key, torch::Tensor value, torch::Tensor kv_cache, torch::Tensor slot_mapping,
        int64_t N, int64_t Hk, int64_t D, int64_t NB, int64_t per_head, int64_t S) {
    TORCH_CHECK(key.is_cuda() && key.scalar_type() == torch::kBFloat16, "key must be CUDA bf16");
    TORCH_CHECK(value.scalar_type() == torch::kBFloat16, "value must be bf16");
    TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8, "kv_cache must be uint8");
    TORCH_CHECK(slot_mapping.scalar_type() == torch::kLong, "slot_mapping must be int64");
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous() && kv_cache.is_contiguous(),
                "key/value/kv_cache must be contiguous");
    if (N <= 0) return;
    const long total = N * Hk * NB;
    const int TPB = 128;
    mxfp6_kv_append_fused_kernel<<<(int)((total + TPB - 1) / TPB), TPB, 0,
                                   at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value.data_ptr<at::BFloat16>()),
        kv_cache.data_ptr<uint8_t>(), slot_mapping.data_ptr<int64_t>(),
        (int)N, (int)Hk, (int)D, (int)NB, (int)per_head, (int)S);
}
