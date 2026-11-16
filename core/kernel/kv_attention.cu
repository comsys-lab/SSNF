// kernel/kv_attention.cu  —  [pure CUDA]  KV-cache flash-decode attention
//
// Decode-step attention with K/V stored SSNF-signed and dequantized on the fly
// inside the FlashAttention load path. Separating dequant into a standalone pass would double KV
// traffic, so it is fused here.
//
// STATUS: dequant reuses the certified ssnf::unpack_ssnf_kv_elem, so the math
// matches ssnf_lib.reference.kv_attention; certify with
// tests/test_kv.py::test_kv_decode_attention_vs_oracle.
//
// OCCUPANCY: SPLIT-KV / FLASH-DECODING. The original kernel launched one block
// per head (H=8 blocks) -> that leaves >90% of the SMs idle on the target GPU,
// a hard launch-config ceiling no per-SM tuning can lift. We now split the key
// axis into tiles of KV_TILE keys: grid = (H, S) with S = ceil(Lk/KV_TILE), so
// the block count grows 8 -> 8*S and fills the machine. Each (h,s) block runs
// online softmax over only its key tile and writes a PARTIAL (acc, m, l); a
// second combine kernel merges the S partials per head with the standard
// online-softmax rescale (m_g = max_s m_s; weight each partial by exp(m_s-m_g)).
// This is the memory-parallelism fix, not the compute fix.
//
// Scope of these kernels: Q·K^T is a per-key block reduction over head_dim, and
// P·V is a per-thread scalar accumulate (the deployed decode path lives in
// kv_paged.cu; these are the append + exploration kernels).
//   * Single decode step (Lq=1): the query is the newest token, so it attends
//     to ALL Lk cached keys -> no causal mask needed (matches the oracle with
//     Lq=1). Multi-step / chunked decode adds masking.
//
// Layout: K/V planes are TOKEN-MAJOR [H, nb, L, UB|SB] (BYTES innermost; Stage
// 4a) so a warp's 32 head_dim reads at a fixed key coalesce. One block per
// (head, key-tile); one thread per head_dim element e (blockDim padded to pow2).

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <cstdlib>
#include <math.h>
#include <type_traits>
#include "core/ssnf_utils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <mma.h>

namespace {
using namespace nvcuda;

constexpr int BLOCK = 32;
constexpr int WSK = BLOCK + 8;   // shared K-tile width (+pad) — matches wa_gemm.cu
constexpr int KV_CHUNK = 128;   // keys processed per chunk (bounds shared mem)
constexpr int CP_CHUNK = 64;    // keys per cp.async double-buffer chunk (smaller smem)

// cp.async-copy n bytes global->shared in 4-byte chunks (UB,SB multiples of 4 keep
// the offsets 4-aligned); a <4-byte tail (small shared fields) is copied directly.

inline int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

// split the P·V Lk-loop so base_blocks*S ~= mult*#SM fills the machine; cap at nLk
// (each split needs >=1 32-key chunk). SSNF_PV_SPLIT_MULT (env, default 4).
inline int pv_split_count(int base_blocks, int nLk) {
    static int sm = -1;
    if (sm < 0) cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    int mult = 4;
    if (const char* e = getenv("SSNF_PV_SPLIT_MULT")) { int m = atoi(e); if (m > 0) mult = m; }
    int S = (mult * sm + base_blocks - 1) / base_blocks;
    if (S > nLk) S = nLk;
    if (S < 1) S = 1;
    return S;
}

// ---- KV WRITE (prefill): pack_kv as a CUDA kernel ---------------------------
//   bf16 X [H,L,D] (post-projection+RoPE) -> token-major SSNF planes that
//   kv_decode_* reads back unchanged. THREAD-PER-TOKEN (the write mirror of the
//   Phase-18 read): thread (h,j) loops the nb head_dim blocks; for a fixed
//   (h,blk) consecutive threads = consecutive tokens write UB-contiguous bytes
//   -> coalesced store (u4: 512 B/warp). Blocks = H*ceil(L/TPB), occupancy free
//   (no split needed). decompose_ssnf_block + dense LSB bit-pack (ssnf_utils). One
//   plane set (K or V) per call.
// thread-per-token, one head_dim BLOCK per block-grid-z. Lifting nb into the grid
// (vs an in-thread blk loop) is the occupancy fix for GQA: at H=8 the old grid was
// only H*ceil(L/256)=32 blocks -> <0.5 block/SM, so the kernel ran
// ~10x off its (BW-light) ceiling. grid (H, ceil(L/TPB), nb) -> H*ceil(L/128)*nb
// (~224) blocks fills the machine. Consecutive threads = consecutive tokens at a
// fixed (h,blk) -> UB-contiguous coalesced store (the Phase-18 write mirror, kept).
__global__ void kv_write_kernel(
        const __nv_bfloat16* __restrict__ X,    // [H, L, D]
        int8_t*  __restrict__ scale_exp,         // [H, nb, L]
        uint8_t* __restrict__ upper,             // [H, nb, L, UB]
        uint8_t* __restrict__ shared,            // [H, nb, L, SB]
        int H, int L, int D, int NB, int l_b, int g_size, int UB, int SB, int lightms) {
    const int h = blockIdx.x;
    const int blk = blockIdx.z;                            // one head_dim block per grid-z
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= L) return;
    const int ng = 32 / g_size, wbits = 8 - l_b;
    float x[32];
    const long xb = ((long)h * L + j) * D + (long)blk * 32;
    #pragma unroll
    for (int k = 0; k < 32; ++k) x[k] = __bfloat162float(X[xb + k]);
    int h_word[32], l_g_word[16];
    const int ea = lightms ? ssnf::decompose_lightms_block(x, l_b, g_size, h_word, l_g_word)
                           : ssnf::decompose_ssnf_block(x, l_b, g_size, h_word, l_g_word);
    const long tok = (long)(h * NB + blk) * L + j;
    uint32_t ureg[8] = {0u,0u,0u,0u,0u,0u,0u,0u};         // <=24 packed upper bytes
    ssnf::pack_codes_lsb(h_word, 32, wbits, (uint8_t*)ureg, UB);
    uint32_t* dU = reinterpret_cast<uint32_t*>(upper + tok * UB);  // UB,tok*UB 4-aligned
    #pragma unroll
    for (int w = 0; w < 8; ++w) if (w < (UB >> 2)) dU[w] = ureg[w];
    uint8_t sbuf[8];
    ssnf::pack_codes_lsb(l_g_word, ng, l_b, sbuf, SB);
    for (int bi = 0; bi < SB; ++bi) shared[tok * SB + bi] = sbuf[bi];
    scale_exp[tok] = (int8_t)ea;
}


// ---- KV APPEND, PAGED-BLOB, FUSED (deploy decode) ----------------------------
//   ONE kernel replacing do_kv_cache_update's ~14-op python sequence (permute+
//   contiguous(k,v), kv_write x2, planes_to_blob x2, cat, scatter). Reads k/v
//   [N, Hk*D] DIRECTLY (no permute/copy), quantizes+packs each 32-block, and writes
//   the interleaved [K per_head | V per_head] blob straight into the cache slot
//   cache[(slot*Hk + h)*S + ...] (S = 2*per_head). Byte-IDENTICAL to the old path
//   (same decompose_ssnf_block/pack_codes_lsb; blob layout per_head = nb*[scale|upper|
//   shared], K then V). thread = (n,h,blk). NO rotation (deploy default path).
__global__ void kv_append_paged_fused_kernel(
        const __nv_bfloat16* __restrict__ key,     // [N, Hk*D]
        const __nv_bfloat16* __restrict__ value,   // [N, Hk*D]
        uint8_t* __restrict__ cache,               // [nslots, Hk, S] contiguous
        const int64_t* __restrict__ slot_mapping,  // [N]
        int N, int Hk, int D, int NB, int l_b, int g_size, int UB, int SB,
        int per_head, int S, int lightms,
        const float* __restrict__ HK, int radius) {     // HK: [Hk*D] K-importance (nullptr=off), K only
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long)N * Hk * NB) return;
    const int blk = (int)(tid % NB);
    const int h   = (int)((tid / NB) % Hk);
    const int n   = (int)(tid / ((long)NB * Hk));
    const long slot = slot_mapping[n];
    if (slot < 0) return;                                   // padded/invalid slot
    const int ng = 32 / g_size, wbits = 8 - l_b;
    const long xb = (long)n * (Hk * D) + (long)h * D + (long)blk * 32;
    uint8_t* dst = cache + ((long)slot * Hk + h) * S + (long)blk * (1 + UB + SB);
    int h_word[32], l_g_word[16];
    uint8_t ubuf[32], sbuf[8];
    // --- K (first per_head): importance-weighted shared ---
    {
        float x[32];
        #pragma unroll
        for (int k = 0; k < 32; ++k) x[k] = __bfloat162float(key[xb + k]);
        const float* Hblk = HK ? HK + (long)h * D + (long)blk * 32 : nullptr;
        const int ea = lightms                            // lightms -> INT domain (INT-hmean if Hblk)
            ? ssnf::decompose_lightms_block(x, l_b, g_size, h_word, l_g_word, Hblk)
            : ssnf::decompose_ssnf_block(x, l_b, g_size, h_word, l_g_word, Hblk, radius);   // FP (hmean/trunc)
        ssnf::pack_codes_lsb(h_word, 32, wbits, ubuf, UB);
        ssnf::pack_codes_lsb(l_g_word, ng, l_b, sbuf, SB);
        dst[0] = (uint8_t)(int8_t)ea;
        for (int bi = 0; bi < UB; ++bi) dst[1 + bi] = ubuf[bi];
        for (int bi = 0; bi < SB; ++bi) dst[1 + UB + bi] = sbuf[bi];
    }
    // --- V (second per_head) ---
    {
        float x[32];
        #pragma unroll
        for (int k = 0; k < 32; ++k) x[k] = __bfloat162float(value[xb + k]);
        const int ea = lightms ? ssnf::decompose_lightms_block(x, l_b, g_size, h_word, l_g_word)
                               : ssnf::decompose_ssnf_block(x, l_b, g_size, h_word, l_g_word);
        ssnf::pack_codes_lsb(h_word, 32, wbits, ubuf, UB);
        ssnf::pack_codes_lsb(l_g_word, ng, l_b, sbuf, SB);
        uint8_t* vd = dst + per_head;
        vd[0] = (uint8_t)(int8_t)ea;
        for (int bi = 0; bi < UB; ++bi) vd[1 + bi] = ubuf[bi];
        for (int bi = 0; bi < SB; ++bi) vd[1 + UB + bi] = sbuf[bi];
    }
}


// ASYM K/V fused append: K written with (ku,kgs)+importance, V with (vu,vgs) plain. K plane =
// first per_head_K bytes of the slot, V plane = next per_head_V. Mirror of kv_append_paged_fused
// slot S = per_head_K + per_head_V.
__global__ void kv_append_paged_fused_asym_kernel(
        const __nv_bfloat16* __restrict__ key, const __nv_bfloat16* __restrict__ value,
        uint8_t* __restrict__ cache, const int64_t* __restrict__ slot_mapping,
        int N, int Hk, int D, int NB, int ku, int kgs, int KUB, int KSB,
        int vu, int vgs, int VUB, int VSB, int per_head_K, int S, int lightms,
        const float* __restrict__ HK, int radius, int k4p1) {
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long)N * Hk * NB) return;
    const int blk = (int)(tid % NB);
    const int h   = (int)((tid / NB) % Hk);
    const int n   = (int)(tid / ((long)NB * Hk));
    const long slot = slot_mapping[n];
    if (slot < 0) return;
    const long xb = (long)n * (Hk * D) + (long)h * D + (long)blk * 32;
    const long slotbase = ((long)slot * Hk + h) * S;
    int h_word[32], l_g_word[16];
    uint8_t ubuf[32], sbuf[8];
    // --- K (per_head_K, first): importance-weighted (ku,kgs) ---
    {
        const int kng = 32 / kgs, kwbits = 8 - ku;
        uint8_t* dst = cache + slotbase + (long)blk * (1 + KUB + KSB);
        float x[32];
        #pragma unroll
        for (int k = 0; k < 32; ++k) x[k] = __bfloat162float(key[xb + k]);
        const float* Hblk = HK ? HK + (long)h * D + (long)blk * 32 : nullptr;
        const int ea = lightms
            ? ssnf::decompose_lightms_block(x, ku, kgs, h_word, l_g_word, Hblk)
            : ssnf::decompose_ssnf_block(x, ku, kgs, h_word, l_g_word, Hblk, radius);
        if (k4p1) {
            // 4+1 SPLIT K plane (same codes/bytes, byte-aligned rearrangement):
            //   [nibble(low4)x32 = 16B x NB | highbit x32 = 4B x NB | scale x NB | shared x NB]
            // Decode consumes directly from the prefetch buffer (no staging).
            uint8_t* base = cache + slotbase;
            uint8_t* nb4 = base + (long)blk * 16;
            #pragma unroll
            for (int i = 0; i < 16; ++i)
                nb4[i] = (uint8_t)((h_word[2 * i] & 0xF) | ((h_word[2 * i + 1] & 0xF) << 4));
            uint8_t* hb = base + (long)NB * 16 + (long)blk * 4;
            #pragma unroll
            for (int by = 0; by < 4; ++by) {
                int v = 0;
                #pragma unroll
                for (int bi = 0; bi < 8; ++bi) v |= ((h_word[by * 8 + bi] >> 4) & 1) << bi;
                hb[by] = (uint8_t)v;
            }
            ssnf::pack_codes_lsb(l_g_word, kng, ku, sbuf, KSB);
            base[(long)NB * 20 + blk] = (uint8_t)(int8_t)ea;
            base[(long)NB * 21 + blk] = sbuf[0];                 // KSB==1 (kgs==32)
        } else {
            ssnf::pack_codes_lsb(h_word, 32, kwbits, ubuf, KUB);
            ssnf::pack_codes_lsb(l_g_word, kng, ku, sbuf, KSB);
            dst[0] = (uint8_t)(int8_t)ea;
            for (int bi = 0; bi < KUB; ++bi) dst[1 + bi] = ubuf[bi];
            for (int bi = 0; bi < KSB; ++bi) dst[1 + KUB + bi] = sbuf[bi];
        }
    }
    // --- V (per_head_K offset): plain (vu,vgs) ---
    {
        const int vng = 32 / vgs, vwbits = 8 - vu;
        uint8_t* vd = cache + slotbase + per_head_K + (long)blk * (1 + VUB + VSB);
        float x[32];
        #pragma unroll
        for (int k = 0; k < 32; ++k) x[k] = __bfloat162float(value[xb + k]);
        const int ea = lightms ? ssnf::decompose_lightms_block(x, vu, vgs, h_word, l_g_word)
                               : ssnf::decompose_ssnf_block(x, vu, vgs, h_word, l_g_word);
        ssnf::pack_codes_lsb(h_word, 32, vwbits, ubuf, VUB);
        ssnf::pack_codes_lsb(l_g_word, vng, vu, sbuf, VSB);
        vd[0] = (uint8_t)(int8_t)ea;
        for (int bi = 0; bi < VUB; ++bi) vd[1 + bi] = ubuf[bi];
        for (int bi = 0; bi < VSB; ++bi) vd[1 + VUB + bi] = sbuf[bi];
    }
}

} // namespace

// KV write launcher: bf16 X [H,L,D] -> (scale_exp [H,nb,L], upper [H,nb,L,UB],
// shared [H,nb,L,SB]) — the certified token-major planes. UB/SB from l_b,g_size.
std::vector<torch::Tensor> kv_write_cuda(
        torch::Tensor X, int64_t H, int64_t L, int64_t D, int64_t NB, int64_t l_b, int64_t g_size) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    const int wbits = 8 - (int)l_b;
    const int UB = BLOCK * wbits / 8;
    const int SB = ((BLOCK / (int)g_size) * (int)l_b + 7) / 8;
    auto i8 = X.options().dtype(torch::kInt8);
    auto u8 = X.options().dtype(torch::kUInt8);
    auto scale_exp = torch::empty({H, NB, L}, i8);
    auto upper     = torch::empty({H, NB, L, UB}, u8);
    auto shared    = torch::empty({H, NB, L, SB}, u8);
    const int TPB = 128;                                  // smaller block + grid-z=nb -> more blocks
    const char* lm = getenv("SSNF_LIGHTMS"); const int lightms = (lm && atoi(lm) == 1) ? 1 : 0;
    kv_write_kernel<<<dim3((int)H, ((int)L + TPB - 1) / TPB, (int)NB), TPB, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
        scale_exp.data_ptr<int8_t>(), upper.data_ptr<uint8_t>(), shared.data_ptr<uint8_t>(),
        (int)H, (int)L, (int)D, (int)NB, (int)l_b, (int)g_size, UB, SB, lightms);
    return {scale_exp, upper, shared};
}


// FUSED paged-blob append (deploy decode): key/value [N, Hk*D] bf16 -> quantized blob written
// straight into cache slots. Replaces do_kv_cache_update's ~14-op sequence with ONE launch.
void kv_append_paged_fused_cuda(
        torch::Tensor key, torch::Tensor value, torch::Tensor kv_cache, torch::Tensor slot_mapping,
        int64_t N, int64_t Hk, int64_t D, int64_t NB, int64_t l_b, int64_t g_size,
        int64_t per_head, int64_t S, torch::Tensor H_k, int64_t radius) {
    TORCH_CHECK(key.is_cuda() && key.scalar_type() == torch::kBFloat16, "key must be CUDA bf16");
    TORCH_CHECK(value.scalar_type() == torch::kBFloat16, "value must be bf16");
    TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8, "kv_cache must be uint8");
    TORCH_CHECK(slot_mapping.scalar_type() == torch::kLong, "slot_mapping must be int64");
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous() && kv_cache.is_contiguous(),
                "key/value/kv_cache must be contiguous");
    if (N <= 0) return;
    const int wbits = 8 - (int)l_b;
    const int UB = BLOCK * wbits / 8;
    const int SB = ((BLOCK / (int)g_size) * (int)l_b + 7) / 8;
    const char* lm = getenv("SSNF_LIGHTMS"); const int lightms = (lm && atoi(lm) == 1) ? 1 : 0;
    // H_k: [Hk*D] fp32 K-importance (empty tensor = off; K only, V stays plain-mean)
    const float* HK = nullptr;
    if (H_k.defined() && H_k.numel() > 0) {
        TORCH_CHECK(H_k.is_cuda() && H_k.scalar_type() == torch::kFloat32 && H_k.is_contiguous(),
                    "H_k must be CUDA fp32 contiguous");
        TORCH_CHECK(H_k.numel() == Hk * D, "H_k must have Hk*D elements");
        HK = H_k.data_ptr<float>();
    }
    const long total = N * Hk * NB;
    const int TPB = 128;
    kv_append_paged_fused_kernel<<<(int)((total + TPB - 1) / TPB), TPB, 0,
                                   at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value.data_ptr<at::BFloat16>()),
        kv_cache.data_ptr<uint8_t>(), slot_mapping.data_ptr<int64_t>(),
        (int)N, (int)Hk, (int)D, (int)NB, (int)l_b, (int)g_size, UB, SB, (int)per_head, (int)S, lightms,
        HK, (int)radius);
}


// ASYM K/V fused append launcher: K (ku,kgs)+importance, V (vu,vgs) plain. slot S = per_head_K+per_head_V.
void kv_append_paged_fused_asym_cuda(
        torch::Tensor key, torch::Tensor value, torch::Tensor kv_cache, torch::Tensor slot_mapping,
        int64_t N, int64_t Hk, int64_t D, int64_t NB, int64_t ku, int64_t kgs, int64_t vu, int64_t vgs,
        int64_t per_head_K, int64_t S, torch::Tensor H_k, int64_t radius) {
    TORCH_CHECK(key.is_cuda() && key.scalar_type() == torch::kBFloat16, "key must be CUDA bf16");
    TORCH_CHECK(value.scalar_type() == torch::kBFloat16, "value must be bf16");
    TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8, "kv_cache must be uint8");
    TORCH_CHECK(slot_mapping.scalar_type() == torch::kLong, "slot_mapping must be int64");
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous() && kv_cache.is_contiguous(),
                "key/value/kv_cache must be contiguous");
    if (N <= 0) return;
    const int KUB = BLOCK * (8 - (int)ku) / 8, KSB = ((BLOCK / (int)kgs) * (int)ku + 7) / 8;
    const int VUB = BLOCK * (8 - (int)vu) / 8, VSB = ((BLOCK / (int)vgs) * (int)vu + 7) / 8;
    const char* lm = getenv("SSNF_LIGHTMS"); const int lightms = (lm && atoi(lm) == 1) ? 1 : 0;
    const float* HK = nullptr;
    if (H_k.defined() && H_k.numel() > 0) {
        TORCH_CHECK(H_k.is_cuda() && H_k.scalar_type() == torch::kFloat32 && H_k.is_contiguous() &&
                    H_k.numel() == Hk * D, "H_k must be CUDA fp32 contiguous [Hk*D]");
        HK = H_k.data_ptr<float>();
    }
    // 4+1 split K plane (deploy asym config only; must match the decode dispatch conditions).
    int k4p1 = 0;                            // OPT-IN (SSNF_KV_K4P1=1); must match the decode dispatch gate
    if (const char* e = getenv("SSNF_KV_K4P1"))
        k4p1 = (ku == 3 && kgs == 32 && D == 128 && (S % 16 == 0)) && (atoi(e) != 0);
    const long total = N * Hk * NB;
    const int TPB = 128;
    kv_append_paged_fused_asym_kernel<<<(int)((total + TPB - 1) / TPB), TPB, 0,
                                   at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value.data_ptr<at::BFloat16>()),
        kv_cache.data_ptr<uint8_t>(), slot_mapping.data_ptr<int64_t>(),
        (int)N, (int)Hk, (int)D, (int)NB, (int)ku, (int)kgs, KUB, KSB, (int)vu, (int)vgs, VUB, VSB,
        (int)per_head_K, (int)S, lightms, HK, (int)radius, k4p1);
}
