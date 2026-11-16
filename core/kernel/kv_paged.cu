// Paged SSNF decode-attention kernel for the vLLM backend.
// Reads K/V directly from vLLM's paged cache via block_table — no Python gather.
//
// Cache layout (uint8): [num_blocks, block_size, Hkv, slot_bytes]
//   slot_bytes = 2 * per_head ; per_head = NB*(1+UB+SB)
//   per token-head: [ K: {scale(1)|upper(UB)|shared(SB)} x NB | V: same ]
// One CUDA block per (request b, KV head hk); D threads (thread d owns head_dim d).
// GQA-fused: the block processes all G = H/Hkv query heads sharing hk, so each K/V
// element is dequantized from the paged cache ONCE and reused across the group
// (was G x redundant with a per-(b,query-head) grid). Online-softmax flash decode.
//
// KERNEL MAP (the DEFAULT deployed decode path is marked *):
//   SSNF asym   : v8_asym_cpa -> +kstage -> +tcqk(*)      [SSNF_KV_ASYM set, e.g. K u3/gs16 + V
//                                                          u4/gs32; SSNF_KV_KSTAGE / SSNF_KV_TCQK
//                                                          gates isolate each optimization]
//   SSNF sym    : kv_decode_paged / v8 / v8_cpa /          [CONFIG=ssnf_kv without SSNF_KV_ASYM:
//                 v8_kstage_cpa / v8_tcqk_cpa               K==V config, same gates]
//   baselines   : mxint8 / mxfp8 / mxfp6 {plain, cpa}      [their own defaults]
//   fairness    : mxint8/mxfp6 _tcqk_cpa                   [SSNF_MXI_TCQK / SSNF_MX6_TCQK, default
//                                                          OFF — measured SLOWER than their own
//                                                          defaults (0.64-0.74x / 0.54-0.57x);
//                                                          kept as evidence that the SSNF-specific
//                                                          optimizations do not transfer]
//   experiment  : v8_asym_k4p1_cpa                         [SSNF_KV_K4P1, no reported number]
//   combine     : kv_paged_combine                         [split-K reduction, all paths]
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <torch/extension.h>
#include "ssnf_utils.cuh"

// MXFP8-E4M3 element decode (hw fp8 cvt); mirror of kernel/mxfp.cu::e4m3_to_f. Used by the
// mxfp8 paged-decode kernels below (the fp8 twin of the int8 mxint8 paged kernels).
static __device__ __forceinline__ float e4m3_to_f(uint8_t b) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3)));
}
// MXFP6-E2M3 element decode; mirror of kernel/mxfp.cu::e2m3_val/e2m3_word. 6-bit code sign|exp2|mant3,
// bias=1. Codes packed 4-per-3-byte-word (offsets 0/6/12/18). mxfp6_elem: read element kd from a
// 24-byte packed block (blkstride=25: 1 scale byte + 24 code bytes).
static __device__ __forceinline__ float e2m3_val(int code) {
    const int e = (code >> 3) & 3, m = code & 7;
    const int mant = m + ((e != 0) << 3);
    const float v = (float)mant * exp2f((float)(max(e, 1) - 4));
    return __int_as_float(__float_as_int(v) ^ ((code & 0x20) << 26));   // sign bit5 -> bit31
}
static __device__ __forceinline__ float mxfp6_elem(const uint8_t* code24, int kd) {
    const int g = kd >> 2, w = kd & 3;
    const uint8_t* p = code24 + g * 3;
    const unsigned word = (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16);
    return e2m3_val((int)((word >> (6 * w)) & 0x3Fu));
}

static constexpr int MXB = 32;   // MX block size (matches ssnf_utils)
static constexpr int KV_CHUNK = 32;        // cp.async staging chunk in TOKENS (MX formats;
// unrelated to the 32-element scaling block): measured best for all three
// The staging chunk is per format, measured at each format's own decode operating point.
// SSNF slots are 160 B, so a 64-key chunk still fits the 48 KB block budget and is faster;
// the MX slots (264/200 B) are too large at 64 -- the double buffer alone collapses
// occupancy -- so they keep 32.
static constexpr int KV_CHUNK_SSNF = 64;   // SSNF symmetric + asym kernels

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// GQA-fused warp-per-key chunked flash decode. One CUDA block per (request b, KV head hk)
// processes ALL G = H/Hkv query heads that share hk, so each K/V element is read from the
// paged cache ONCE and reused across the group (was G× redundant with a per-(b,h) grid).
// Pass1: a warp dequantizes one key's K once, accumulates G dot products via __shfl;
// Pass2: thread d dequantizes V[d] once, accumulates into all G heads. KV global traffic /G.
// G is a compile-time template param so the per-head register arrays size exactly to G and
// the g-loops fully unroll (a runtime G would spill these arrays to local memory).
template<int G>
__global__ void kv_decode_paged_kernel(
        const __nv_bfloat16* __restrict__ q,     // [B,H,D]
        const uint8_t* __restrict__ cache,       // [num_blocks, block_size, Hkv, slot_bytes]
        const int* __restrict__ block_table,     // [B, max_blocks]
        const int* __restrict__ seq_lens,        // [B]
        __nv_bfloat16* __restrict__ out,         // [B,H,D]
        int B, int H, int Hkv, int D, int NB, int l_b, int g_size, int UB, int SB,
        int block_size, int slot_bytes, int max_blocks, int per_head, float sm_scale) {
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int h0 = hk * G;                          // first query head of the group
    const int L = seq_lens[b];
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int blkstride = 1 + UB + SB;

    extern __shared__ float sm[];                   // [G*D] q  +  [G*KV_CHUNK_SSNF] scores
    float* q_sh = sm; float* sc = sm + G * D;
    for (int i = tid; i < G * D; i += blockDim.x)   // load all G heads' q
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();

    const int dblk = tid / MXB, dkd = tid % MXB;    // thread tid owns head_dim d=tid (Pass2)
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }

    for (int cs = 0; cs < L; cs += KV_CHUNK_SSNF) {
        const int nC = min(KV_CHUNK_SSNF, L - cs);
        // ---- Pass 1: one warp per key, dequant K once, G __shfl reduces ----
        for (int kk = warpId; kk < nC; kk += nWarps) {
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            const long sbase = ((long)phys * Hkv + hk) * slot_bytes;
            float part[G];
            #pragma unroll
            for (int g = 0; g < G; ++g) part[g] = 0.0f;
            for (int d = lane; d < D; d += 32) {
                const int blk = d / MXB, kd = d % MXB;
                const long kblk = sbase + (long)blk * blkstride;
                const float ksc = ssnf::e8m0_to_scale((int8_t)cache[kblk]);
                const int code = ssnf::unpack_ssnf_kv_elem_bfe(cache, cache, kblk + 1, kblk + 1 + UB,
                                                       0, kd, l_b, g_size, UB, SB);
                const float kval = ksc * (float)code;
                #pragma unroll
                for (int g = 0; g < G; ++g) part[g] += q_sh[g * D + d] * kval;
            }
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                float r = warp_reduce_sum(part[g]);
                if (lane == 0) sc[g * KV_CHUNK_SSNF + kk] = r * sm_scale;
            }
        }
        __syncthreads();
        // ---- per-head chunk max + online-softmax rescale ----
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * KV_CHUNK_SSNF + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        // ---- Pass 2: thread tid dequants V[d=tid] (from shared) once, into all G heads ----
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        for (int kk = 0; kk < nC; ++kk) {
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            const long vblk = ((long)phys * Hkv + hk) * slot_bytes + per_head
                              + (long)dblk * blkstride;
            const float vsc = ssnf::e8m0_to_scale((int8_t)cache[vblk]);
            const int vcode = ssnf::unpack_ssnf_kv_elem_bfe(cache, cache, vblk + 1, vblk + 1 + UB,
                                                    0, dkd, l_b, g_size, UB, SB);
            const float vval = vsc * (float)vcode;
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = __expf(sc[g * KV_CHUNK_SSNF + kk] - m_new[g]);
                lsum[g] += p;
                a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();                            // protect sc[] before next chunk
    }
    #pragma unroll
    for (int g = 0; g < G; ++g)
        out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
}

// ===== int8-staged-V twin of kv_decode_paged_kernel (v8/vt lever, ported from the
// dense kv_decode_wide_kernel). Pass-1 (K-dot) is byte-identical to the base kernel;
// Pass-2 (P·V) is the only change: instead of unpacking each sub-byte V element with a
// bfe per (key,d) in the accumulation loop, a coalesced staging phase reconstructs every
// V code of the chunk to a plain int8 (up*2^l_b+sh, fits int8 for u2/u3 since |up|<=q_max)
// stored TRANSPOSED [blk][kd][kk] (CH=chunk+4 pad -> conflict-free int32 reads). Pass-2
// then reads 4 contiguous int8/int32 with no funnel — exactly like the MXINT8 twin, so
// SSNF's V byte-advantage converts to time (memory: ssnf-vs-mxint8-kv-decode-state).
// Compile-time LB_/GSIZE_ (stream_block_uspec + the `if constexpr(LB_>0)` staging guard) are
// Combine kernel: merge the S split-K partials per (request, query head) with the standard
// flash rescale (m_g = max_s m_s; weight each partial by exp(m_s - m_g)). One block per
// (query head h, request b); thread e owns head_dim e. Format-agnostic (fp32 partials).
__global__ void kv_paged_combine_kernel(
        const float* __restrict__ part_o,        // [B,H,S,D] unnormalized acc
        const float* __restrict__ part_m,        // [B,H,S]   partial max
        const float* __restrict__ part_l,        // [B,H,S]   partial denom
        __nv_bfloat16* __restrict__ out,         // [B,H,D]
        int H, int D, int S) {
    const int h = blockIdx.x, e = threadIdx.x;
    if (e >= D) return;
    const long b = blockIdx.y;
    part_o += b * (long)H * S * D; part_m += b * (long)H * S; part_l += b * (long)H * S;
    out += b * (long)H * D;
    float m_g = -INFINITY;
    for (int s = 0; s < S; ++s) m_g = fmaxf(m_g, part_m[h * S + s]);
    float l = 0.0f, acc = 0.0f;
    for (int s = 0; s < S; ++s) {
        const float w = __expf(part_m[h * S + s] - m_g);   // empty/lesser tile -> ~0
        l   += part_l[h * S + s] * w;
        acc += part_o[((long)h * S + s) * D + e] * w;
    }
    out[h * D + e] = __float2bfloat16(l > 0.0f ? acc / l : 0.0f);
}

// REQUIRED: a runtime guard compiles dead u4 code that blows registers and regresses.
// SPLIT-K: grid.z = S splits the L key axis into disjoint tiles so the block count grows
// (B*Hkv -> B*Hkv*S) and fills the machine at LOW batch / long context (the under-occupied,
// latency-bound regime; ncu B128 was already saturated so kv_split_count returns S=1 there).
// S==1 keeps the original direct-to-out path (byte-identical, no partials/combine).
template<int G, int LB_, int GSIZE_>
__global__ void kv_decode_paged_v8_kernel(
        const __nv_bfloat16* __restrict__ q,     // [B,H,D]
        const uint8_t* __restrict__ cache,       // [num_blocks, block_size, Hkv, slot_bytes]
        const int* __restrict__ block_table,     // [B, max_blocks]
        const int* __restrict__ seq_lens,        // [B]
        __nv_bfloat16* __restrict__ out,         // [B,H,D]  (S==1 direct write)
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,  // S>1
        int B, int H, int Hkv, int D, int NB, int l_b, int g_size, int UB, int SB,
        int block_size, int slot_bytes, int max_blocks, int per_head, float sm_scale, int S) {
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int s = blockIdx.z;
    const int h0 = hk * G;
    const int L = seq_lens[b];
    const int key_tile = (L + S - 1) / S;           // keys per split (last split may be short/empty)
    const int j0 = s * key_tile, j1 = min(j0 + key_tile, L);
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int blkstride = 1 + UB + SB;
    const int CH = KV_CHUNK_SSNF + 4;                    // padded transposed chunk width (CH/4 odd)

    extern __shared__ float sm[];                   // [G*D] q | [G*KV_CHUNK_SSNF] scores |
    float* q_sh   = sm;                             // [NB*KV_CHUNK_SSNF] V scales | [NB*MXB*CH] int8 V
    float* sc     = sm + G * D;
    float* vsc_sh = sc + G * KV_CHUNK_SSNF;              // staged per-(blk,key) V E8M0 scale
    int8_t* v8buf = (int8_t*)(vsc_sh + NB * KV_CHUNK_SSNF);  // staged int8 V, [blk][kd][kk]
    for (int i = tid; i < G * D; i += blockDim.x)
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();

    const int dblk = tid / MXB, dkd = tid % MXB;    // thread tid owns head_dim d=tid (Pass2)
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }

    for (int cs = j0; cs < j1; cs += KV_CHUNK_SSNF) {
        const int nC = min(KV_CHUNK_SSNF, j1 - cs);
        // ---- Pass 1: one warp per key, dequant K once, G __shfl reduces (UNCHANGED) ----
        for (int kk = warpId; kk < nC; kk += nWarps) {
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            const long sbase = ((long)phys * Hkv + hk) * slot_bytes;
            float part[G];
            #pragma unroll
            for (int g = 0; g < G; ++g) part[g] = 0.0f;
            for (int d = lane; d < D; d += 32) {
                const int blk = d / MXB, kd = d % MXB;
                const long kblk = sbase + (long)blk * blkstride;
                const float ksc = ssnf::e8m0_to_scale((int8_t)cache[kblk]);
                const int code = ssnf::unpack_ssnf_kv_elem_bfe(cache, cache, kblk + 1, kblk + 1 + UB,
                                                       0, kd, l_b, g_size, UB, SB);
                const float kval = ksc * (float)code;
                #pragma unroll
                for (int g = 0; g < G; ++g) part[g] += q_sh[g * D + d] * kval;
            }
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                float r = warp_reduce_sum(part[g]);
                if (lane == 0) sc[g * KV_CHUNK_SSNF + kk] = r * sm_scale;
            }
        }
        __syncthreads();
        // ---- per-head chunk max + online-softmax rescale ----
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * KV_CHUNK_SSNF + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        // Convert scores -> probabilities ONCE (cooperative), not 128-d x G redundantly in
        // Pass-2. Scores sc[g][kk] are d-independent, m_new[g] identical across d-threads, so
        // the per-thread expf was pure redundancy (the dense kernel's cooperative-softmax lever).
        __syncthreads();                            // all threads read raw sc for m_new before overwrite
        for (int i = tid; i < G * nC; i += blockDim.x) {
            const int g = i / nC, kk = i % nC;
            sc[g * KV_CHUNK_SSNF + kk] = __expf(sc[g * KV_CHUNK_SSNF + kk] - m_new[g]);
        }
        // ---- V int8-staging: one thread per key streams all NB blocks -> int8 transposed ----
        // The paged slot packs [scale(1)|upper(UB)|shared(SB)] so the upper bytes start at an
        // ODD offset (cache+vblk+1) -> not 4-aligned, and stream_block_uspec does uint32 reads.
        // Copy this block's UBc+SBc bytes into 4-aligned local buffers first, then stream (base 0).
        constexpr int WBITS = 8 - LB_;
        constexpr int UBc = MXB * WBITS / 8;                 // upper bytes per 32-block
        constexpr int SBc = ((MXB / GSIZE_) * LB_ + 7) / 8;      // shared bytes per 32-block
        for (int kk = tid; kk < nC; kk += blockDim.x) {
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            const long vbase = ((long)phys * Hkv + hk) * slot_bytes + per_head;
            #pragma unroll 1
            for (int blk = 0; blk < NB; ++blk) {
                const long vblk = vbase + (long)blk * blkstride;
                vsc_sh[blk * KV_CHUNK_SSNF + kk] = ssnf::e8m0_to_scale((int8_t)cache[vblk]);
                alignas(4) uint8_t ub[UBc], shb[SBc];
                #pragma unroll
                for (int i = 0; i < UBc; ++i) ub[i]  = cache[vblk + 1 + i];
                #pragma unroll
                for (int i = 0; i < SBc; ++i) shb[i] = cache[vblk + 1 + UB + i];
                ssnf::stream_block_uspec<LB_, GSIZE_>(ub, shb, 0,
                    [&](int kd, int w) { v8buf[(long)blk * MXB * CH + kd * CH + kk] = (int8_t)w; });
            }
        }
        __syncthreads();
        // ---- Pass 2: thread tid owns d=tid; read staged int8 V (contiguous, vectorized) ----
        const int8_t* vrow = v8buf + (long)dblk * MXB * CH + (long)dkd * CH;
        const float* vscb = vsc_sh + dblk * KV_CHUNK_SSNF;
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        int kk = 0;
        for (; kk + 3 < nC; kk += 4) {
            const int32_t w4 = *reinterpret_cast<const int32_t*>(vrow + kk);
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float vval = (float)((int8_t)((w4 >> (8 * t)) & 0xff)) * vscb[kk + t];
                #pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float p = sc[g * KV_CHUNK_SSNF + kk + t];   // already a probability
                    lsum[g] += p;
                    a[g] += p * vval;
                }
            }
        }
        for (; kk < nC; ++kk) {
            const float vval = (float)vrow[kk] * vscb[kk];
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = sc[g * KV_CHUNK_SSNF + kk];           // already a probability
                lsum[g] += p;
                a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();                            // protect sc[]/staged V before next chunk
    }
    if (S == 1) {                                   // no split: normalize + write out directly
        #pragma unroll
        for (int g = 0; g < G; ++g)
            out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
    } else {                                        // split-K: write UNnormalized partials for combine
        #pragma unroll
        for (int g = 0; g < G; ++g)
            part_o[(((long)b * H + (h0 + g)) * S + s) * D + tid] = acc[g];
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                part_m[((long)b * H + (h0 + g)) * S + s] = m_i[g];
                part_l[((long)b * H + (h0 + g)) * S + s] = l_i[g];
            }
        }
    }
}

// ===== ASYMMETRIC K/V twin of kv_decode_paged_v8_kernel. K and V may use
// DIFFERENT (l_b,g_size): K uses RUNTIME (ku,kgs,KUB,KSB) in Pass-1 (already runtime via
// unpack_ssnf_kv_elem_bfe); the template (LB_,GSIZE_) is V's config for the compile-time V int8-staging.
// K plane = first per_head_K bytes of the slot, V plane = next per_head_V. Byte-identical math to v8
// when symmetric. Does NOT touch the shipped symmetric v8/cpa/base kernels (separate dispatch). =====
template<int G, int LB_, int GSIZE_>
__global__ void kv_decode_paged_v8_asym_kernel(
        const __nv_bfloat16* __restrict__ q,
        const uint8_t* __restrict__ cache,
        const int* __restrict__ block_table,
        const int* __restrict__ seq_lens,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int B, int H, int Hkv, int D, int NB, int ku, int kgs, int KUB, int KSB, int VUB, int VSB,
        int block_size, int slot_bytes, int max_blocks, int per_head_K, float sm_scale, int S) {
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int s = blockIdx.z;
    const int h0 = hk * G;
    const int L = seq_lens[b];
    const int key_tile = (L + S - 1) / S;
    const int j0 = s * key_tile, j1 = min(j0 + key_tile, L);
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int Kblkstride = 1 + KUB + KSB;                 // K plane block stride (Pass-1)
    const int Vblkstride = 1 + VUB + VSB;                 // V plane block stride (Pass-2)
    const int CH = KV_CHUNK_SSNF + 4;

    extern __shared__ float sm[];
    float* q_sh   = sm;
    float* sc     = sm + G * D;
    float* vsc_sh = sc + G * KV_CHUNK_SSNF;
    int8_t* v8buf = (int8_t*)(vsc_sh + NB * KV_CHUNK_SSNF);
    for (int i = tid; i < G * D; i += blockDim.x)
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();

    const int dblk = tid / MXB, dkd = tid % MXB;
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }

    for (int cs = j0; cs < j1; cs += KV_CHUNK_SSNF) {
        const int nC = min(KV_CHUNK_SSNF, j1 - cs);
        // ---- Pass 1: K uses RUNTIME (ku,kgs,KUB,KSB), K plane stride Kblkstride ----
        for (int kk = warpId; kk < nC; kk += nWarps) {
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            const long sbase = ((long)phys * Hkv + hk) * slot_bytes;
            float part[G];
            #pragma unroll
            for (int g = 0; g < G; ++g) part[g] = 0.0f;
            for (int d = lane; d < D; d += 32) {
                const int blk = d / MXB, kd = d % MXB;
                const long kblk = sbase + (long)blk * Kblkstride;
                const float ksc = ssnf::e8m0_to_scale((int8_t)cache[kblk]);
                const int code = ssnf::unpack_ssnf_kv_elem_bfe(cache, cache, kblk + 1, kblk + 1 + KUB,
                                                       0, kd, ku, kgs, KUB, KSB);
                const float kval = ksc * (float)code;
                #pragma unroll
                for (int g = 0; g < G; ++g) part[g] += q_sh[g * D + d] * kval;
            }
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                float r = warp_reduce_sum(part[g]);
                if (lane == 0) sc[g * KV_CHUNK_SSNF + kk] = r * sm_scale;
            }
        }
        __syncthreads();
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * KV_CHUNK_SSNF + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        __syncthreads();
        for (int i = tid; i < G * nC; i += blockDim.x) {
            const int g = i / nC, kk = i % nC;
            sc[g * KV_CHUNK_SSNF + kk] = __expf(sc[g * KV_CHUNK_SSNF + kk] - m_new[g]);
        }
        // ---- V int8-staging: template (LB_,GSIZE_) = V config; V plane at per_head_K, stride Vblkstride, VUB ----
        constexpr int WBITS = 8 - LB_;
        constexpr int UBc = MXB * WBITS / 8;
        constexpr int SBc = ((MXB / GSIZE_) * LB_ + 7) / 8;
        for (int kk = tid; kk < nC; kk += blockDim.x) {
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            const long vbase = ((long)phys * Hkv + hk) * slot_bytes + per_head_K;
            #pragma unroll 1
            for (int blk = 0; blk < NB; ++blk) {
                const long vblk = vbase + (long)blk * Vblkstride;
                vsc_sh[blk * KV_CHUNK_SSNF + kk] = ssnf::e8m0_to_scale((int8_t)cache[vblk]);
                alignas(4) uint8_t ub[UBc], shb[SBc];
                #pragma unroll
                for (int i = 0; i < UBc; ++i) ub[i]  = cache[vblk + 1 + i];
                #pragma unroll
                for (int i = 0; i < SBc; ++i) shb[i] = cache[vblk + 1 + VUB + i];
                ssnf::stream_block_uspec<LB_, GSIZE_>(ub, shb, 0,
                    [&](int kd, int w) { v8buf[(long)blk * MXB * CH + kd * CH + kk] = (int8_t)w; });
            }
        }
        __syncthreads();
        const int8_t* vrow = v8buf + (long)dblk * MXB * CH + (long)dkd * CH;
        const float* vscb = vsc_sh + dblk * KV_CHUNK_SSNF;
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        int kk = 0;
        for (; kk + 3 < nC; kk += 4) {
            const int32_t w4 = *reinterpret_cast<const int32_t*>(vrow + kk);
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float vval = (float)((int8_t)((w4 >> (8 * t)) & 0xff)) * vscb[kk + t];
                #pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float p = sc[g * KV_CHUNK_SSNF + kk + t];
                    lsum[g] += p;
                    a[g] += p * vval;
                }
            }
        }
        for (; kk < nC; ++kk) {
            const float vval = (float)vrow[kk] * vscb[kk];
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = sc[g * KV_CHUNK_SSNF + kk];
                lsum[g] += p;
                a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();
    }
    if (S == 1) {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
    } else {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            part_o[(((long)b * H + (h0 + g)) * S + s) * D + tid] = acc[g];
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                part_m[((long)b * H + (h0 + g)) * S + s] = m_i[g];
                part_l[((long)b * H + (h0 + g)) * S + s] = l_i[g];
            }
        }
    }
}

// ===== cp.async double-buffered twin of kv_decode_paged_v8_kernel =====
// ncu (B128 L1024): the #1 stall is long_scoreboard = 5.7 cyc = GLOBAL-load latency (the scattered
// per-key paged-cache reads in Pass-1 K + V-staging), with only 0.88 eligible warps to hide it.
// This kernel prefetches the NEXT chunk's raw KV slots into shared via cp.async while computing the
// current chunk, so the compute reads from shared (prefetched) and the DRAM latency overlaps. Viable
// because occupancy is REGISTER-capped with large shared slack (Block Limit: reg 8 < shared 11), so
// the +2*KV_CHUNK_SSNF*slot_bytes double-buffer does not lower occupancy. Requires slot_bytes%16==0 (host
// gate) for 16-byte cp.async. Bit-identical math to the v8 kernel (same reads, just from shared).
template<int G, int LB_, int GSIZE_>
__global__ void kv_decode_paged_v8_cpa_kernel(
        const __nv_bfloat16* __restrict__ q,
        const uint8_t* __restrict__ cache,
        const int* __restrict__ block_table,
        const int* __restrict__ seq_lens,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int B, int H, int Hkv, int D, int NB, int l_b, int g_size, int UB, int SB,
        int block_size, int slot_bytes, int max_blocks, int per_head, float sm_scale, int S) {
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int s = blockIdx.z;
    const int h0 = hk * G;
    const int L = seq_lens[b];
    const int key_tile = (L + S - 1) / S;
    const int j0 = s * key_tile, j1 = min(j0 + key_tile, L);
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int blkstride = 1 + UB + SB;
    const int CH = KV_CHUNK_SSNF + 4;

    extern __shared__ float sm[];
    float* q_sh   = sm;
    float* sc     = sm + G * D;
    float* vsc_sh = sc + G * KV_CHUNK_SSNF;
    int8_t* v8buf = (int8_t*)(vsc_sh + NB * KV_CHUNK_SSNF);
    uint8_t* rawbuf = (uint8_t*)(v8buf + NB * MXB * CH);  // [2][KV_CHUNK_SSNF*slot_bytes], 16-aligned
    for (int i = tid; i < G * D; i += blockDim.x)
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();

    const int dblk = tid / MXB, dkd = tid % MXB;
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }

    // Hoist this lane's q-slice into registers (Pass-1 re-reads it per key, a large share of
    // the shared-load LSU bottleneck), so cutting shared-load INSTRUCTIONS wins even at the
    // register/occupancy cost. D<=128 fully cached.
    constexpr int MAXNB = 4;
    const bool qhoist = (D <= MAXNB * 32);
    float qreg[MAXNB * G];
    if (qhoist) {
        #pragma unroll
        for (int di = 0; di < MAXNB; ++di) {
            const int d = di * 32 + lane;
            #pragma unroll
            for (int g = 0; g < G; ++g) qreg[di * G + g] = (d < D) ? q_sh[g * D + d] : 0.0f;
        }
    }

    constexpr int WBITS = 8 - LB_;
    constexpr int UBc = MXB * WBITS / 8;
    constexpr int SBc = ((MXB / GSIZE_) * LB_ + 7) / 8;
    const int U16 = slot_bytes >> 4;                     // 16B cp.async units per slot (slot%16==0)
    const long BUFSZ = (long)KV_CHUNK_SSNF * slot_bytes;
    // async-copy chunk [cs, cs+nC) raw KV slots (scattered per-key) into rb (shared)
    auto prefetch = [&](int cs, int nC, uint8_t* rb) {
        for (int idx = tid; idx < nC * U16; idx += blockDim.x) {
            const int kk = idx / U16, u16 = idx % U16;
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            __pipeline_memcpy_async(rb + (long)kk * slot_bytes + (long)u16 * 16,
                                    cache + ((long)phys * Hkv + hk) * slot_bytes + (long)u16 * 16, 16);
        }
        __pipeline_commit();
    };

    const int nchunks = (j1 > j0) ? (j1 - j0 + KV_CHUNK_SSNF - 1) / KV_CHUNK_SSNF : 0;
    if (nchunks > 0) prefetch(j0, min(KV_CHUNK_SSNF, j1 - j0), rawbuf);
    for (int ci = 0; ci < nchunks; ++ci) {
        const int cs = j0 + ci * KV_CHUNK_SSNF;
        const int nC = min(KV_CHUNK_SSNF, j1 - cs);
        const uint8_t* rb = rawbuf + (long)(ci & 1) * BUFSZ;
        const bool hasNext = (ci + 1 < nchunks);
        if (hasNext) prefetch(cs + KV_CHUNK_SSNF, min(KV_CHUNK_SSNF, j1 - cs - KV_CHUNK_SSNF),
                              rawbuf + (long)((ci + 1) & 1) * BUFSZ);
        __pipeline_wait_prior(hasNext ? 1 : 0);         // chunk ci's slots now resident in rb
        __syncthreads();

        // ---- Pass 1: K-dot, K read from prefetched shared rb (rb-relative slot base = kk*slot_bytes) ----
        for (int kk = warpId; kk < nC; kk += nWarps) {
            const long sbase = (long)kk * slot_bytes;
            float part[G];
            #pragma unroll
            for (int g = 0; g < G; ++g) part[g] = 0.0f;
            if (qhoist) {                            // D<=128: q from registers (di,g compile-time)
                #pragma unroll
                for (int di = 0; di < MAXNB; ++di) {
                    const int d = di * 32 + lane;
                    if (d >= D) continue;
                    const int blk = d / MXB, kd = d % MXB;
                    const long kblk = sbase + (long)blk * blkstride;
                    const float ksc = ssnf::e8m0_to_scale((int8_t)rb[kblk]);
                    const int code = ssnf::unpack_ssnf_kv_elem_bfe(rb, rb, kblk + 1, kblk + 1 + UB,
                                                           0, kd, l_b, g_size, UB, SB);
                    const float kval = ksc * (float)code;
                    #pragma unroll
                    for (int g = 0; g < G; ++g) part[g] += qreg[di * G + g] * kval;
                }
            } else {
                for (int d = lane; d < D; d += 32) {
                    const int blk = d / MXB, kd = d % MXB;
                    const long kblk = sbase + (long)blk * blkstride;
                    const float ksc = ssnf::e8m0_to_scale((int8_t)rb[kblk]);
                    const int code = ssnf::unpack_ssnf_kv_elem_bfe(rb, rb, kblk + 1, kblk + 1 + UB,
                                                           0, kd, l_b, g_size, UB, SB);
                    const float kval = ksc * (float)code;
                    #pragma unroll
                    for (int g = 0; g < G; ++g) part[g] += q_sh[g * D + d] * kval;
                }
            }
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                float r = warp_reduce_sum(part[g]);
                if (lane == 0) sc[g * KV_CHUNK_SSNF + kk] = r * sm_scale;
            }
        }
        __syncthreads();
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * KV_CHUNK_SSNF + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        __syncthreads();
        for (int i = tid; i < G * nC; i += blockDim.x) {
            const int g = i / nC, kk = i % nC;
            sc[g * KV_CHUNK_SSNF + kk] = __expf(sc[g * KV_CHUNK_SSNF + kk] - m_new[g]);
        }
        // ---- V int8-staging from prefetched shared rb ----
        for (int kk = tid; kk < nC; kk += blockDim.x) {
            const long vbase = (long)kk * slot_bytes + per_head;
            #pragma unroll 1
            for (int blk = 0; blk < NB; ++blk) {
                const long vblk = vbase + (long)blk * blkstride;
                vsc_sh[blk * KV_CHUNK_SSNF + kk] = ssnf::e8m0_to_scale((int8_t)rb[vblk]);
                alignas(4) uint8_t ub[UBc], shb[SBc];
                #pragma unroll
                for (int i = 0; i < UBc; ++i) ub[i]  = rb[vblk + 1 + i];
                #pragma unroll
                for (int i = 0; i < SBc; ++i) shb[i] = rb[vblk + 1 + UB + i];
                ssnf::stream_block_uspec<LB_, GSIZE_>(ub, shb, 0,
                    [&](int kd, int w) { v8buf[(long)blk * MXB * CH + kd * CH + kk] = (int8_t)w; });
            }
        }
        __syncthreads();
        // ---- Pass 2: P·V from staged int8 (same as v8) ----
        const int8_t* vrow = v8buf + (long)dblk * MXB * CH + (long)dkd * CH;
        const float* vscb = vsc_sh + dblk * KV_CHUNK_SSNF;
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        int kk = 0;
        for (; kk + 3 < nC; kk += 4) {
            const int32_t w4 = *reinterpret_cast<const int32_t*>(vrow + kk);
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float vval = (float)((int8_t)((w4 >> (8 * t)) & 0xff)) * vscb[kk + t];
                #pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float p = sc[g * KV_CHUNK_SSNF + kk + t];
                    lsum[g] += p; a[g] += p * vval;
                }
            }
        }
        for (; kk < nC; ++kk) {
            const float vval = (float)vrow[kk] * vscb[kk];
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = sc[g * KV_CHUNK_SSNF + kk];
                lsum[g] += p; a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();
    }
    if (S == 1) {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
    } else {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            part_o[(((long)b * H + (h0 + g)) * S + s) * D + tid] = acc[g];
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                part_m[((long)b * H + (h0 + g)) * S + s] = m_i[g];
                part_l[((long)b * H + (h0 + g)) * S + s] = l_i[g];
            }
        }
    }
}

// ===== K-STAGED SYMMETRIC cpa twin : same
// surgery as the asym kstage kernel, but K==V config so the K staging is the SAME compile-time
// stream_block_uspec<LB_,GSIZE_> as V — generalizes to every dispatched (l_b,g_size), no hardcoding.
// Pass-1 becomes: stage K -> int8 shared k8[kk][D+4] (pad = bank-conflict-free), then QK reads
// ONE int32 per lane per key. Faster than the runtime unpack; dispatch needs D==128. =====
template<int G, int LB_, int GSIZE_>
__global__ void kv_decode_paged_v8_kstage_cpa_kernel(
        const __nv_bfloat16* __restrict__ q,
        const uint8_t* __restrict__ cache,
        const int* __restrict__ block_table,
        const int* __restrict__ seq_lens,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int B, int H, int Hkv, int D, int NB, int l_b, int g_size, int UB, int SB,
        int block_size, int slot_bytes, int max_blocks, int per_head, float sm_scale, int S) {
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int s = blockIdx.z;
    const int h0 = hk * G;
    const int L = seq_lens[b];
    const int key_tile = (L + S - 1) / S;
    const int j0 = s * key_tile, j1 = min(j0 + key_tile, L);
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int blkstride = 1 + UB + SB;
    const int CH = KV_CHUNK_SSNF + 4;

    extern __shared__ float sm[];
    float* q_sh   = sm;
    float* sc     = sm + G * D;
    float* vsc_sh = sc + G * KV_CHUNK_SSNF;
    int8_t* v8buf = (int8_t*)(vsc_sh + NB * KV_CHUNK_SSNF);
    uint8_t* rawbuf = (uint8_t*)(v8buf + NB * MXB * CH);
    const int KROW = D + 4;                                             // +4 pad (bank-conflict-free)
    int8_t* k8 = (int8_t*)(rawbuf + 2 * (long)KV_CHUNK_SSNF * slot_bytes);   // [KV_CHUNK_SSNF][KROW]
    float* ksc_sh = (float*)(k8 + KV_CHUNK_SSNF * KROW);                     // [NB][KV_CHUNK_SSNF]
    for (int i = tid; i < G * D; i += blockDim.x)
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();

    const int dblk = tid / MXB, dkd = tid % MXB;
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }

    float q4reg[4 * G];                       // q for d = lane*4 + t (matches the int32 K read)
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int d = lane * 4 + t;
        #pragma unroll
        for (int g = 0; g < G; ++g) q4reg[t * G + g] = (d < D) ? q_sh[g * D + d] : 0.0f;
    }

    constexpr int WBITS = 8 - LB_;
    constexpr int UBc = MXB * WBITS / 8;
    constexpr int SBc = ((MXB / GSIZE_) * LB_ + 7) / 8;
    const int U16 = slot_bytes >> 4;
    const long BUFSZ = (long)KV_CHUNK_SSNF * slot_bytes;
    auto prefetch = [&](int cs, int nC, uint8_t* rb) {
        for (int idx = tid; idx < nC * U16; idx += blockDim.x) {
            const int kk = idx / U16, u16 = idx % U16;
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            __pipeline_memcpy_async(rb + (long)kk * slot_bytes + (long)u16 * 16,
                                    cache + ((long)phys * Hkv + hk) * slot_bytes + (long)u16 * 16, 16);
        }
        __pipeline_commit();
    };

    const int nchunks = (j1 > j0) ? (j1 - j0 + KV_CHUNK_SSNF - 1) / KV_CHUNK_SSNF : 0;
    if (nchunks > 0) prefetch(j0, min(KV_CHUNK_SSNF, j1 - j0), rawbuf);
    for (int ci = 0; ci < nchunks; ++ci) {
        const int cs = j0 + ci * KV_CHUNK_SSNF;
        const int nC = min(KV_CHUNK_SSNF, j1 - cs);
        const uint8_t* rb = rawbuf + (long)(ci & 1) * BUFSZ;
        const bool hasNext = (ci + 1 < nchunks);
        if (hasNext) prefetch(cs + KV_CHUNK_SSNF, min(KV_CHUNK_SSNF, j1 - cs - KV_CHUNK_SSNF),
                              rawbuf + (long)((ci + 1) & 1) * BUFSZ);
        __pipeline_wait_prior(hasNext ? 1 : 0);
        __syncthreads();

        // Pass 1a: K staging — same compile-time stream unpack as V, (kk,blk) parallel
        for (int idx = tid; idx < nC * NB; idx += blockDim.x) {
            const int kk = idx / NB, blk = idx % NB;
            const long kblk = (long)kk * slot_bytes + (long)blk * blkstride;
            ksc_sh[blk * KV_CHUNK_SSNF + kk] = ssnf::e8m0_to_scale((int8_t)rb[kblk]);
            alignas(4) uint8_t ub[UBc], shb[SBc];
            #pragma unroll
            for (int i = 0; i < UBc; ++i) ub[i]  = rb[kblk + 1 + i];
            #pragma unroll
            for (int i = 0; i < SBc; ++i) shb[i] = rb[kblk + 1 + UB + i];
            ssnf::stream_block_uspec<LB_, GSIZE_>(ub, shb, 0,
                [&](int kd, int w) { k8[kk * KROW + blk * MXB + kd] = (int8_t)w; });
        }
        __syncthreads();
        // Pass 1b: QK — warp-per-key, ONE int32 LDS per lane covers its 4 channels
        for (int kk = warpId; kk < nC; kk += nWarps) {
            float part[G];
            #pragma unroll
            for (int g = 0; g < G; ++g) part[g] = 0.0f;
            const int32_t w4 = *reinterpret_cast<const int32_t*>(k8 + kk * KROW + lane * 4);
            const float ksc = ksc_sh[(lane >> 3) * KV_CHUNK_SSNF + kk];   // block = (lane*4)/32
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float kval = (float)((int8_t)((w4 >> (8 * t)) & 0xff)) * ksc;
                #pragma unroll
                for (int g = 0; g < G; ++g) part[g] += q4reg[t * G + g] * kval;
            }
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                float r = warp_reduce_sum(part[g]);
                if (lane == 0) sc[g * KV_CHUNK_SSNF + kk] = r * sm_scale;
            }
        }
        __syncthreads();
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * KV_CHUNK_SSNF + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        __syncthreads();
        for (int i = tid; i < G * nC; i += blockDim.x) {
            const int g = i / nC, kk = i % nC;
            sc[g * KV_CHUNK_SSNF + kk] = __expf(sc[g * KV_CHUNK_SSNF + kk] - m_new[g]);
        }
        for (int kk = tid; kk < nC; kk += blockDim.x) {   // V staging (unchanged)
            const long vbase = (long)kk * slot_bytes + per_head;
            #pragma unroll 1
            for (int blk = 0; blk < NB; ++blk) {
                const long vblk = vbase + (long)blk * blkstride;
                vsc_sh[blk * KV_CHUNK_SSNF + kk] = ssnf::e8m0_to_scale((int8_t)rb[vblk]);
                alignas(4) uint8_t ub[UBc], shb[SBc];
                #pragma unroll
                for (int i = 0; i < UBc; ++i) ub[i]  = rb[vblk + 1 + i];
                #pragma unroll
                for (int i = 0; i < SBc; ++i) shb[i] = rb[vblk + 1 + UB + i];
                ssnf::stream_block_uspec<LB_, GSIZE_>(ub, shb, 0,
                    [&](int kd, int w) { v8buf[(long)blk * MXB * CH + kd * CH + kk] = (int8_t)w; });
            }
        }
        __syncthreads();
        const int8_t* vrow = v8buf + (long)dblk * MXB * CH + (long)dkd * CH;
        const float* vscb = vsc_sh + dblk * KV_CHUNK_SSNF;
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        int kk = 0;
        for (; kk + 3 < nC; kk += 4) {
            const int32_t w4 = *reinterpret_cast<const int32_t*>(vrow + kk);
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float vval = (float)((int8_t)((w4 >> (8 * t)) & 0xff)) * vscb[kk + t];
                #pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float p = sc[g * KV_CHUNK_SSNF + kk + t];
                    lsum[g] += p; a[g] += p * vval;
                }
            }
        }
        for (; kk < nC; ++kk) {
            const float vval = (float)vrow[kk] * vscb[kk];
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = sc[g * KV_CHUNK_SSNF + kk];
                lsum[g] += p; a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();
    }
    if (S == 1) {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
    } else {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            part_o[(((long)b * H + (h0 + g)) * S + s) * D + tid] = acc[g];
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                part_m[((long)b * H + (h0 + g)) * S + s] = m_i[g];
                part_l[((long)b * H + (h0 + g)) * S + s] = l_i[g];
            }
        }
    }
}

// ===== TC-QK SYMMETRIC cpa twin: kstage staging (compile-time
// stream_block_uspec<LB_,GSIZE_>, all (l_b,g_size)) + QK via m16n8k16 HMMA â mma absorbs the CUDA-core
// FMA + warp-reduce; B-frag consumes the staged int8 so the path is format-independent.
// Faster than the FMA path at every deployed shape. Needs D==128 & G==4. =====
template<int G, int LB_, int GSIZE_>
__global__ void kv_decode_paged_v8_tcqk_cpa_kernel(
        const __nv_bfloat16* __restrict__ q,
        const uint8_t* __restrict__ cache,
        const int* __restrict__ block_table,
        const int* __restrict__ seq_lens,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int B, int H, int Hkv, int D, int NB, int l_b, int g_size, int UB, int SB,
        int block_size, int slot_bytes, int max_blocks, int per_head, float sm_scale, int S) {
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int s = blockIdx.z;
    const int h0 = hk * G;
    const int L = seq_lens[b];
    const int key_tile = (L + S - 1) / S;
    const int j0 = s * key_tile, j1 = min(j0 + key_tile, L);
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int blkstride = 1 + UB + SB;
    const int CH = KV_CHUNK_SSNF + 4;

    extern __shared__ float sm[];
    float* q_sh   = sm;
    float* sc     = sm + G * D;
    float* vsc_sh = sc + G * KV_CHUNK_SSNF;
    int8_t* v8buf = (int8_t*)(vsc_sh + NB * KV_CHUNK_SSNF);
    uint8_t* rawbuf = (uint8_t*)(v8buf + NB * MXB * CH);
    const int KROW = D + 4;                                             // +4 pad (bank-conflict-free)
    int8_t* k8 = (int8_t*)(rawbuf + 2 * (long)KV_CHUNK_SSNF * slot_bytes);   // [KV_CHUNK_SSNF][KROW]
    float* ksc_sh = (float*)(k8 + KV_CHUNK_SSNF * KROW);                     // [NB][KV_CHUNK_SSNF]
    constexpr int QP = 136;
    __nv_bfloat16* qtile = (__nv_bfloat16*)(ksc_sh + NB * KV_CHUNK_SSNF);    // [16][QP] zero-padded q
    for (int i = tid; i < G * D; i += blockDim.x)
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();

    const int dblk = tid / MXB, dkd = tid % MXB;
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }

    for (int i = tid; i < 16 * QP; i += blockDim.x) {                  // q bf16 tile for mma A
        const int r = i / QP, cc = i % QP;
        qtile[i] = (r < G && cc < D) ? __float2bfloat16(q_sh[r * D + cc]) : __float2bfloat16(0.0f);
    }
    __syncthreads();

    constexpr int WBITS = 8 - LB_;
    constexpr int UBc = MXB * WBITS / 8;
    constexpr int SBc = ((MXB / GSIZE_) * LB_ + 7) / 8;
    const int U16 = slot_bytes >> 4;
    const long BUFSZ = (long)KV_CHUNK_SSNF * slot_bytes;
    auto prefetch = [&](int cs, int nC, uint8_t* rb) {
        for (int idx = tid; idx < nC * U16; idx += blockDim.x) {
            const int kk = idx / U16, u16 = idx % U16;
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            __pipeline_memcpy_async(rb + (long)kk * slot_bytes + (long)u16 * 16,
                                    cache + ((long)phys * Hkv + hk) * slot_bytes + (long)u16 * 16, 16);
        }
        __pipeline_commit();
    };

    const int nchunks = (j1 > j0) ? (j1 - j0 + KV_CHUNK_SSNF - 1) / KV_CHUNK_SSNF : 0;
    if (nchunks > 0) prefetch(j0, min(KV_CHUNK_SSNF, j1 - j0), rawbuf);
    for (int ci = 0; ci < nchunks; ++ci) {
        const int cs = j0 + ci * KV_CHUNK_SSNF;
        const int nC = min(KV_CHUNK_SSNF, j1 - cs);
        const uint8_t* rb = rawbuf + (long)(ci & 1) * BUFSZ;
        const bool hasNext = (ci + 1 < nchunks);
        if (hasNext) prefetch(cs + KV_CHUNK_SSNF, min(KV_CHUNK_SSNF, j1 - cs - KV_CHUNK_SSNF),
                              rawbuf + (long)((ci + 1) & 1) * BUFSZ);
        __pipeline_wait_prior(hasNext ? 1 : 0);
        __syncthreads();

        // Pass 1a: K staging — same compile-time stream unpack as V, (kk,blk) parallel
        for (int idx = tid; idx < nC * NB; idx += blockDim.x) {
            const int kk = idx / NB, blk = idx % NB;
            const long kblk = (long)kk * slot_bytes + (long)blk * blkstride;
            ksc_sh[blk * KV_CHUNK_SSNF + kk] = ssnf::e8m0_to_scale((int8_t)rb[kblk]);
            alignas(4) uint8_t ub[UBc], shb[SBc];
            #pragma unroll
            for (int i = 0; i < UBc; ++i) ub[i]  = rb[kblk + 1 + i];
            #pragma unroll
            for (int i = 0; i < SBc; ++i) shb[i] = rb[kblk + 1 + UB + i];
            ssnf::stream_block_uspec<LB_, GSIZE_>(ub, shb, 0,
                [&](int kd, int w) { k8[kk * KROW + blk * MXB + kd] = (int8_t)w; });
        }
        __syncthreads();
        // Pass 1b: QK via HMMA â warp w owns key-group w; mma absorbs FMA + warp-reduce
        {
            // warp-stride loop: when the staging chunk is larger than nWarps*8 keys (e.g. the
            // 64-key SSNF chunk with 4 warps), each warp owns SEVERAL 8-key groups; a single
            // ng=warpId would cover only the first nWarps*8 keys of the chunk.
            for (int ng = warpId; ng * 8 < nC; ng += nWarps) {
                float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
                const int bn = ng * 8 + (lane >> 2);
                const bool kvld = (bn < nC);
                #pragma unroll
                for (int s5 = 0; s5 < 8; ++s5) {
                    uint32_t a0, a1, a2, a3;
                    {
                        const int ar = lane >> 2, ac = s5 * 16 + (lane & 3) * 2;
                        __nv_bfloat162 h;
                        h.x = qtile[ar * QP + ac];       h.y = qtile[ar * QP + ac + 1];       a0 = *reinterpret_cast<uint32_t*>(&h);
                        h.x = qtile[(ar + 8) * QP + ac]; h.y = qtile[(ar + 8) * QP + ac + 1]; a1 = *reinterpret_cast<uint32_t*>(&h);
                        h.x = qtile[ar * QP + ac + 8];   h.y = qtile[ar * QP + ac + 9];       a2 = *reinterpret_cast<uint32_t*>(&h);
                        h.x = qtile[(ar + 8) * QP + ac + 8]; h.y = qtile[(ar + 8) * QP + ac + 9]; a3 = *reinterpret_cast<uint32_t*>(&h);
                    }
                    uint32_t b0 = 0, b1 = 0;
                    if (kvld) {
                        const int8_t* krow = k8 + bn * KROW + s5 * 16 + (lane & 3) * 2;
                        const float ks0 = ksc_sh[((s5 * 16 + (lane & 3) * 2) >> 5) * KV_CHUNK_SSNF + bn];
                        const float ks1 = ksc_sh[((s5 * 16 + (lane & 3) * 2 + 8) >> 5) * KV_CHUNK_SSNF + bn];
                        __nv_bfloat162 lo = __floats2bfloat162_rn((float)krow[0] * ks0, (float)krow[1] * ks0);
                        __nv_bfloat162 hi = __floats2bfloat162_rn((float)krow[8] * ks1, (float)krow[9] * ks1);
                        b0 = *reinterpret_cast<uint32_t*>(&lo);
                        b1 = *reinterpret_cast<uint32_t*>(&hi);
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
                const int crow = lane >> 2, ccol0 = ng * 8 + (lane & 3) * 2;
                if (crow < G) {
                    if (ccol0 < nC)     sc[crow * KV_CHUNK_SSNF + ccol0]     = c0 * sm_scale;
                    if (ccol0 + 1 < nC) sc[crow * KV_CHUNK_SSNF + ccol0 + 1] = c1 * sm_scale;
                }
            }
        }
        __syncthreads();
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * KV_CHUNK_SSNF + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        __syncthreads();
        for (int i = tid; i < G * nC; i += blockDim.x) {
            const int g = i / nC, kk = i % nC;
            sc[g * KV_CHUNK_SSNF + kk] = __expf(sc[g * KV_CHUNK_SSNF + kk] - m_new[g]);
        }
        for (int kk = tid; kk < nC; kk += blockDim.x) {   // V staging (unchanged)
            const long vbase = (long)kk * slot_bytes + per_head;
            #pragma unroll 1
            for (int blk = 0; blk < NB; ++blk) {
                const long vblk = vbase + (long)blk * blkstride;
                vsc_sh[blk * KV_CHUNK_SSNF + kk] = ssnf::e8m0_to_scale((int8_t)rb[vblk]);
                alignas(4) uint8_t ub[UBc], shb[SBc];
                #pragma unroll
                for (int i = 0; i < UBc; ++i) ub[i]  = rb[vblk + 1 + i];
                #pragma unroll
                for (int i = 0; i < SBc; ++i) shb[i] = rb[vblk + 1 + UB + i];
                ssnf::stream_block_uspec<LB_, GSIZE_>(ub, shb, 0,
                    [&](int kd, int w) { v8buf[(long)blk * MXB * CH + kd * CH + kk] = (int8_t)w; });
            }
        }
        __syncthreads();
        const int8_t* vrow = v8buf + (long)dblk * MXB * CH + (long)dkd * CH;
        const float* vscb = vsc_sh + dblk * KV_CHUNK_SSNF;
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        int kk = 0;
        for (; kk + 3 < nC; kk += 4) {
            const int32_t w4 = *reinterpret_cast<const int32_t*>(vrow + kk);
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float vval = (float)((int8_t)((w4 >> (8 * t)) & 0xff)) * vscb[kk + t];
                #pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float p = sc[g * KV_CHUNK_SSNF + kk + t];
                    lsum[g] += p; a[g] += p * vval;
                }
            }
        }
        for (; kk < nC; ++kk) {
            const float vval = (float)vrow[kk] * vscb[kk];
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = sc[g * KV_CHUNK_SSNF + kk];
                lsum[g] += p; a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();
    }
    if (S == 1) {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
    } else {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            part_o[(((long)b * H + (h0 + g)) * S + s) * D + tid] = acc[g];
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                part_m[((long)b * H + (h0 + g)) * S + s] = m_i[g];
                part_l[((long)b * H + (h0 + g)) * S + s] = l_i[g];
            }
        }
    }
}

// ===== cp.async double-buffered ASYM K/V twin (K runtime ku/kgs, V template (LB_,GSIZE_)=V config).
// prefetch copies the whole slot (K plane + V plane) into rb; separate K/V blkstride + V offset. =====
template<int G, int LB_, int GSIZE_>
__global__ void kv_decode_paged_v8_asym_cpa_kernel(
        const __nv_bfloat16* __restrict__ q,
        const uint8_t* __restrict__ cache,
        const int* __restrict__ block_table,
        const int* __restrict__ seq_lens,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int B, int H, int Hkv, int D, int NB, int ku, int kgs, int KUB, int KSB, int VUB, int VSB,
        int block_size, int slot_bytes, int max_blocks, int per_head_K, float sm_scale, int S) {
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int s = blockIdx.z;
    const int h0 = hk * G;
    const int L = seq_lens[b];
    const int key_tile = (L + S - 1) / S;
    const int j0 = s * key_tile, j1 = min(j0 + key_tile, L);
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int Kblkstride = 1 + KUB + KSB;
    const int Vblkstride = 1 + VUB + VSB;
    const int CH = KV_CHUNK_SSNF + 4;

    extern __shared__ float sm[];
    float* q_sh   = sm;
    float* sc     = sm + G * D;
    float* vsc_sh = sc + G * KV_CHUNK_SSNF;
    int8_t* v8buf = (int8_t*)(vsc_sh + NB * KV_CHUNK_SSNF);
    uint8_t* rawbuf = (uint8_t*)(v8buf + NB * MXB * CH);
    for (int i = tid; i < G * D; i += blockDim.x)
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();

    const int dblk = tid / MXB, dkd = tid % MXB;
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }

    constexpr int MAXNB = 4;
    const bool qhoist = (D <= MAXNB * 32);
    float qreg[MAXNB * G];
    if (qhoist) {
        #pragma unroll
        for (int di = 0; di < MAXNB; ++di) {
            const int d = di * 32 + lane;
            #pragma unroll
            for (int g = 0; g < G; ++g) qreg[di * G + g] = (d < D) ? q_sh[g * D + d] : 0.0f;
        }
    }

    constexpr int WBITS = 8 - LB_;
    constexpr int UBc = MXB * WBITS / 8;
    constexpr int SBc = ((MXB / GSIZE_) * LB_ + 7) / 8;
    const int U16 = slot_bytes >> 4;
    const long BUFSZ = (long)KV_CHUNK_SSNF * slot_bytes;
    auto prefetch = [&](int cs, int nC, uint8_t* rb) {
        for (int idx = tid; idx < nC * U16; idx += blockDim.x) {
            const int kk = idx / U16, u16 = idx % U16;
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            __pipeline_memcpy_async(rb + (long)kk * slot_bytes + (long)u16 * 16,
                                    cache + ((long)phys * Hkv + hk) * slot_bytes + (long)u16 * 16, 16);
        }
        __pipeline_commit();
    };

    const int nchunks = (j1 > j0) ? (j1 - j0 + KV_CHUNK_SSNF - 1) / KV_CHUNK_SSNF : 0;
    if (nchunks > 0) prefetch(j0, min(KV_CHUNK_SSNF, j1 - j0), rawbuf);
    for (int ci = 0; ci < nchunks; ++ci) {
        const int cs = j0 + ci * KV_CHUNK_SSNF;
        const int nC = min(KV_CHUNK_SSNF, j1 - cs);
        const uint8_t* rb = rawbuf + (long)(ci & 1) * BUFSZ;
        const bool hasNext = (ci + 1 < nchunks);
        if (hasNext) prefetch(cs + KV_CHUNK_SSNF, min(KV_CHUNK_SSNF, j1 - cs - KV_CHUNK_SSNF),
                              rawbuf + (long)((ci + 1) & 1) * BUFSZ);
        __pipeline_wait_prior(hasNext ? 1 : 0);
        __syncthreads();

        for (int kk = warpId; kk < nC; kk += nWarps) {   // Pass 1: K runtime ku/kgs, Kblkstride
            const long sbase = (long)kk * slot_bytes;
            float part[G];
            #pragma unroll
            for (int g = 0; g < G; ++g) part[g] = 0.0f;
            if (qhoist) {
                #pragma unroll
                for (int di = 0; di < MAXNB; ++di) {
                    const int d = di * 32 + lane;
                    if (d >= D) continue;
                    const int blk = d / MXB, kd = d % MXB;
                    const long kblk = sbase + (long)blk * Kblkstride;
                    const float ksc = ssnf::e8m0_to_scale((int8_t)rb[kblk]);
                    const int code = ssnf::unpack_ssnf_kv_elem_bfe(rb, rb, kblk + 1, kblk + 1 + KUB,
                                                           0, kd, ku, kgs, KUB, KSB);
                    const float kval = ksc * (float)code;
                    #pragma unroll
                    for (int g = 0; g < G; ++g) part[g] += qreg[di * G + g] * kval;
                }
            } else {
                for (int d = lane; d < D; d += 32) {
                    const int blk = d / MXB, kd = d % MXB;
                    const long kblk = sbase + (long)blk * Kblkstride;
                    const float ksc = ssnf::e8m0_to_scale((int8_t)rb[kblk]);
                    const int code = ssnf::unpack_ssnf_kv_elem_bfe(rb, rb, kblk + 1, kblk + 1 + KUB,
                                                           0, kd, ku, kgs, KUB, KSB);
                    const float kval = ksc * (float)code;
                    #pragma unroll
                    for (int g = 0; g < G; ++g) part[g] += q_sh[g * D + d] * kval;
                }
            }
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                float r = warp_reduce_sum(part[g]);
                if (lane == 0) sc[g * KV_CHUNK_SSNF + kk] = r * sm_scale;
            }
        }
        __syncthreads();
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * KV_CHUNK_SSNF + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        __syncthreads();
        for (int i = tid; i < G * nC; i += blockDim.x) {
            const int g = i / nC, kk = i % nC;
            sc[g * KV_CHUNK_SSNF + kk] = __expf(sc[g * KV_CHUNK_SSNF + kk] - m_new[g]);
        }
        for (int kk = tid; kk < nC; kk += blockDim.x) {   // V staging: V plane per_head_K, Vblkstride, VUB
            const long vbase = (long)kk * slot_bytes + per_head_K;
            #pragma unroll 1
            for (int blk = 0; blk < NB; ++blk) {
                const long vblk = vbase + (long)blk * Vblkstride;
                vsc_sh[blk * KV_CHUNK_SSNF + kk] = ssnf::e8m0_to_scale((int8_t)rb[vblk]);
                alignas(4) uint8_t ub[UBc], shb[SBc];
                #pragma unroll
                for (int i = 0; i < UBc; ++i) ub[i]  = rb[vblk + 1 + i];
                #pragma unroll
                for (int i = 0; i < SBc; ++i) shb[i] = rb[vblk + 1 + VUB + i];
                ssnf::stream_block_uspec<LB_, GSIZE_>(ub, shb, 0,
                    [&](int kd, int w) { v8buf[(long)blk * MXB * CH + kd * CH + kk] = (int8_t)w; });
            }
        }
        __syncthreads();
        const int8_t* vrow = v8buf + (long)dblk * MXB * CH + (long)dkd * CH;
        const float* vscb = vsc_sh + dblk * KV_CHUNK_SSNF;
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        int kk = 0;
        for (; kk + 3 < nC; kk += 4) {
            const int32_t w4 = *reinterpret_cast<const int32_t*>(vrow + kk);
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float vval = (float)((int8_t)((w4 >> (8 * t)) & 0xff)) * vscb[kk + t];
                #pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float p = sc[g * KV_CHUNK_SSNF + kk + t];
                    lsum[g] += p; a[g] += p * vval;
                }
            }
        }
        for (; kk < nC; ++kk) {
            const float vval = (float)vrow[kk] * vscb[kk];
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = sc[g * KV_CHUNK_SSNF + kk];
                lsum[g] += p; a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();
    }
    if (S == 1) {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
    } else {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            part_o[(((long)b * H + (h0 + g)) * S + s) * D + tid] = acc[g];
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                part_m[((long)b * H + (h0 + g)) * S + s] = m_i[g];
                part_l[((long)b * H + (h0 + g)) * S + s] = l_i[g];
            }
        }
    }
}

// q [B,H,D] bf16, cache uint8 [nblk,bs,Hkv,slot], block_table int32 [B,max_blocks],
// seq_lens int32 [B] -> out [B,H,D] bf16.
torch::Tensor kv_decode_paged_cuda(
        torch::Tensor q, torch::Tensor cache, torch::Tensor block_table,
        torch::Tensor seq_lens, int64_t B, int64_t H, int64_t Hkv, int64_t D,
        int64_t NB, int64_t l_b, int64_t g_size, int64_t UB, int64_t SB,
        int64_t block_size, int64_t slot_bytes, int64_t per_head) {
    auto out = torch::empty({B, H, D},
                 torch::dtype(torch::kBFloat16).device(q.device()));
    const int max_blocks = block_table.size(1);
    const float sm_scale = 1.0f / sqrtf((float)D);
    const int G = H / Hkv;
    TORCH_CHECK(H == G * Hkv, "kv_decode_paged: H must be a multiple of Hkv");
    const int threads = D;
    const size_t base_shmem = (size_t)(G * D + G * KV_CHUNK_SSNF) * sizeof(float);
    const int CH = KV_CHUNK_SSNF + 4;
    const size_t v8_shmem = base_shmem
        + (size_t)NB * KV_CHUNK_SSNF * sizeof(float)    // staged V E8M0 scales
        + (size_t)NB * MXB * CH;                   // staged int8 V [blk][kd][kk]
    // int8-staged-V fast path. compile-time (LB_,GSIZE_) is required by stream_block_uspec + the
    // constexpr staging guard. u4 int8-stages too: word=up*16+sh in [-120,119] (|up|<=7) fits int8;
    // stream_block_uspec<4,.> handles the nibble unpack (WBITS=4, no straddle). Extended to u4 and
    // gs32 for the nibble/asym-V frontier. Others fall to the runtime base kernel.
    const bool v8ok =
        (l_b == 2 && (g_size == 8 || g_size == 16)) ||
        (l_b == 3 && (g_size == 8 || g_size == 16 || g_size == 32)) ||
        (l_b == 4 && (g_size == 8 || g_size == 16 || g_size == 32));
    // SPLIT-K: fill the machine when the base grid (B*Hkv blocks) under-occupies (low batch /
    // long ctx). L_hint = the block-table capacity (safe upper bound; per-request L handled in
    // the kernel). S=1 -> original direct path (no partials/combine). v8 path only. Default
    // mult=8 (not the dense mult=3): this paged kernel is latency-bound (ncu) on 70-SM Blackwell,
    // so it wants more splits; S still collapses to 1 once the base grid is full (>= mult*SM
    // blocks, ~B>=70 here) so high batch is a correct no-op. SSNF_KV_SPLIT_MULT overrides.
    const int L_hint = max_blocks * (int)block_size;
    int S = 1;   // S = split-K factor (key-splits), NOT the slot-byte S of kv_layout.py
    if (v8ok) {
        int mult = 8;
        if (const char* e = getenv("SSNF_KV_SPLIT_MULT")) { int m = atoi(e); if (m > 0) mult = m; }
        static int nsm = -1;
        if (nsm < 0) cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
        const int base = (int)(B * Hkv);
        S = ((long)mult * nsm + base - 1) / base;
        const int maxS = (L_hint + 31) / 32;      // keep >= 32 keys/tile
        if (S > maxS) S = maxS;
        if (S < 1) S = 1;
        // NOTE: sweeps confirm a non-monotonic split optimum — B=1 short-ctx over-splits, but
        // every global floor/cap that fixes it perturbs neighboring shapes, and most of the surface
        // is within run-to-run noise. Default (mult=8, 32-key floor) kept as the compromise.
    }
    const dim3 grid(B, Hkv, S);                   // one block per (request, kv head, split); fuses G q-heads
    torch::Tensor part_o, part_m, part_l;
    float *po = nullptr, *pm = nullptr, *pl = nullptr;
    if (S > 1) {
        auto fopt = torch::dtype(torch::kFloat32).device(q.device());
        part_o = torch::empty({B, H, S, D}, fopt);
        part_m = torch::empty({B, H, S}, fopt);
        part_l = torch::empty({B, H, S}, fopt);
        po = (float*)part_o.data_ptr(); pm = (float*)part_m.data_ptr(); pl = (float*)part_l.data_ptr();
    }
    // cp.async double-buffer prefetch: hides the #1 stall (long_scoreboard=global-load latency).
    // Adds a 2*KV_CHUNK_SSNF*slot_bytes shared double-buffer; needs slot_bytes%16==0 for 16B cp.async.
    // Occupancy-safe (register-capped, shared slack). SSNF_KV_CPA=0 disables (A/B).
    const size_t cpa_shmem = v8_shmem + (size_t)2 * KV_CHUNK_SSNF * slot_bytes;
    bool use_cpa = v8ok && (slot_bytes % 16 == 0);
    if (const char* e = getenv("SSNF_KV_CPA")) use_cpa = use_cpa && (atoi(e) != 0);
    // K-staged pass-1: any v8 (l_b,g_size), needs D==128 for the one-int32-per-lane QK read.
    // SSNF_KV_KSTAGE=0 disables.
    bool kstage = use_cpa && D == 128;
    if (const char* e = getenv("SSNF_KV_KSTAGE")) kstage = kstage && (atoi(e) != 0);
    const size_t kstage_shmem = cpa_shmem + (size_t)KV_CHUNK_SSNF * (D + 4) + (size_t)NB * KV_CHUNK_SSNF * sizeof(float);
    // TC-QK (DEFAULT ON, SSNF_KV_TCQK=0 disables): HMMA QK on the staged int8 (format-independent). G==4 only.
    bool tcqk = kstage && (G == 4);
    if (const char* e = getenv("SSNF_KV_TCQK")) tcqk = tcqk && (atoi(e) != 0);
    const size_t tcqk_shmem = kstage_shmem + (size_t)16 * 136 * 2;
    auto launch_v8 = [&](auto Gc, auto Uc, auto GSc) {
        constexpr int Gv = decltype(Gc)::value, LBv = decltype(Uc)::value, GSIZEv = decltype(GSc)::value;
        if (tcqk)
            kv_decode_paged_v8_tcqk_cpa_kernel<Gv, LBv, GSIZEv><<<grid, threads, tcqk_shmem>>>(
                (const __nv_bfloat16*)q.data_ptr(), (const uint8_t*)cache.data_ptr(),
                (const int*)block_table.data_ptr(), (const int*)seq_lens.data_ptr(),
                (__nv_bfloat16*)out.data_ptr(), po, pm, pl,
                B, H, Hkv, D, NB, l_b, g_size, UB, SB, block_size, slot_bytes, max_blocks,
                per_head, sm_scale, S);
        else if (kstage)
            kv_decode_paged_v8_kstage_cpa_kernel<Gv, LBv, GSIZEv><<<grid, threads, kstage_shmem>>>(
                (const __nv_bfloat16*)q.data_ptr(), (const uint8_t*)cache.data_ptr(),
                (const int*)block_table.data_ptr(), (const int*)seq_lens.data_ptr(),
                (__nv_bfloat16*)out.data_ptr(), po, pm, pl,
                B, H, Hkv, D, NB, l_b, g_size, UB, SB, block_size, slot_bytes, max_blocks,
                per_head, sm_scale, S);
        else if (use_cpa)
            kv_decode_paged_v8_cpa_kernel<Gv, LBv, GSIZEv><<<grid, threads, cpa_shmem>>>(
                (const __nv_bfloat16*)q.data_ptr(), (const uint8_t*)cache.data_ptr(),
                (const int*)block_table.data_ptr(), (const int*)seq_lens.data_ptr(),
                (__nv_bfloat16*)out.data_ptr(), po, pm, pl,
                B, H, Hkv, D, NB, l_b, g_size, UB, SB, block_size, slot_bytes, max_blocks,
                per_head, sm_scale, S);
        else
            kv_decode_paged_v8_kernel<Gv, LBv, GSIZEv><<<grid, threads, v8_shmem>>>(
                (const __nv_bfloat16*)q.data_ptr(), (const uint8_t*)cache.data_ptr(),
                (const int*)block_table.data_ptr(), (const int*)seq_lens.data_ptr(),
                (__nv_bfloat16*)out.data_ptr(), po, pm, pl,
                B, H, Hkv, D, NB, l_b, g_size, UB, SB, block_size, slot_bytes, max_blocks,
                per_head, sm_scale, S);
    };
    auto launch = [&](auto Gc) {
        constexpr int Gv = decltype(Gc)::value;
        kv_decode_paged_kernel<Gv><<<dim3(B, Hkv), threads, base_shmem>>>(
            (const __nv_bfloat16*)q.data_ptr(), (const uint8_t*)cache.data_ptr(),
            (const int*)block_table.data_ptr(), (const int*)seq_lens.data_ptr(),
            (__nv_bfloat16*)out.data_ptr(),
            B, H, Hkv, D, NB, l_b, g_size, UB, SB, block_size, slot_bytes, max_blocks,
            per_head, sm_scale);
    };
    #define V8CASE(GV,UU,GG) launch_v8(GV, std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{})
    #define SSNF_PAGED_V8(GV) do {                          \
        if      (l_b == 2 && g_size == 8)  V8CASE(GV, 2, 8);    \
        else if (l_b == 2 && g_size == 16) V8CASE(GV, 2, 16);   \
        else if (l_b == 3 && g_size == 8)  V8CASE(GV, 3, 8);    \
        else if (l_b == 3 && g_size == 16) V8CASE(GV, 3, 16);   \
        else if (l_b == 3 && g_size == 32) V8CASE(GV, 3, 32);   \
        else if (l_b == 4 && g_size == 8)  V8CASE(GV, 4, 8);    \
        else if (l_b == 4 && g_size == 16) V8CASE(GV, 4, 16);   \
        else                         V8CASE(GV, 4, 32);   \
    } while (0)
    if (v8ok) {
        switch (G) {
            case 1: SSNF_PAGED_V8((std::integral_constant<int,1>{})); break;
            case 2: SSNF_PAGED_V8((std::integral_constant<int,2>{})); break;
            case 4: SSNF_PAGED_V8((std::integral_constant<int,4>{})); break;
            case 8: SSNF_PAGED_V8((std::integral_constant<int,8>{})); break;
            default: TORCH_CHECK(false, "kv_decode_paged: unsupported GQA group G=", G, " (have 1,2,4,8)");
        }
    } else switch (G) {                           // compile-time G: unrolled, no local-mem spill
        case 1: launch(std::integral_constant<int,1>{}); break;
        case 2: launch(std::integral_constant<int,2>{}); break;
        case 4: launch(std::integral_constant<int,4>{}); break;
        case 8: launch(std::integral_constant<int,8>{}); break;
        default: TORCH_CHECK(false, "kv_decode_paged: unsupported GQA group G=", G, " (have 1,2,4,8)");
    }
    #undef SSNF_PAGED_V8
    if (S > 1)                                    // merge the S partials -> out
        kv_paged_combine_kernel<<<dim3(H, B), threads>>>(po, pm, pl,
            (__nv_bfloat16*)out.data_ptr(), H, D, S);
    return out;
}

// ASYM K/V decode launcher: K (ku,kgs,KUB,KSB) + V (vu,vgs,VUB,VSB) different configs; template on
// ===== K-STAGED asym cpa twin: pass-1's runtime per-element bfe unpack dominated kernel time
// (instruction-volume bound). Stage K like V instead: compile-time stream unpack into int8 shared
// k8[kk][D+4] (+4 pad = bank-conflict-free writes AND reads), then QK reads ONE int32 per lane per
// key (D=128 = 32 lanes x 4ch). Output differs only by summation order.
// K fixed (u3,gs32) = deploy asym config; launcher guards dispatch. SSNF_KV_KSTAGE=0 disables. =====
template<int G, int LB_, int GSIZE_, int KGS_T = 32>
__global__ void kv_decode_paged_v8_asym_kstage_cpa_kernel(
        const __nv_bfloat16* __restrict__ q,
        const uint8_t* __restrict__ cache,
        const int* __restrict__ block_table,
        const int* __restrict__ seq_lens,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int B, int H, int Hkv, int D, int NB, int ku, int kgs, int KUB, int KSB, int VUB, int VSB,
        int block_size, int slot_bytes, int max_blocks, int per_head_K, float sm_scale, int S) {
    constexpr int KU_ = 3, KGS_ = KGS_T;               // compile-time K u3, group size 16 or 32
    constexpr int KUBc = MXB * (8 - KU_) / 8;          // 20
    constexpr int KSBc = ((MXB / KGS_) * KU_ + 7) / 8; // 1
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int s = blockIdx.z;
    const int h0 = hk * G;
    const int L = seq_lens[b];
    const int key_tile = (L + S - 1) / S;
    const int j0 = s * key_tile, j1 = min(j0 + key_tile, L);
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int Kblkstride = 1 + KUB + KSB;
    const int Vblkstride = 1 + VUB + VSB;
    const int CH = KV_CHUNK_SSNF + 4;

    extern __shared__ float sm[];
    float* q_sh   = sm;
    float* sc     = sm + G * D;
    float* vsc_sh = sc + G * KV_CHUNK_SSNF;
    int8_t* v8buf = (int8_t*)(vsc_sh + NB * KV_CHUNK_SSNF);
    uint8_t* rawbuf = (uint8_t*)(v8buf + NB * MXB * CH);
    const int KROW = D + 4;                                             // +4 pad (bank-conflict-free)
    int8_t* k8 = (int8_t*)(rawbuf + 2 * (long)KV_CHUNK_SSNF * slot_bytes);   // [KV_CHUNK_SSNF][KROW]
    float* ksc_sh = (float*)(k8 + KV_CHUNK_SSNF * KROW);                     // [NB][KV_CHUNK_SSNF]
    for (int i = tid; i < G * D; i += blockDim.x)
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();

    const int dblk = tid / MXB, dkd = tid % MXB;
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }

    float q4reg[4 * G];                       // q for d = lane*4 + t (matches the int32 K read)
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int d = lane * 4 + t;
        #pragma unroll
        for (int g = 0; g < G; ++g) q4reg[t * G + g] = (d < D) ? q_sh[g * D + d] : 0.0f;
    }

    constexpr int WBITS = 8 - LB_;
    constexpr int UBc = MXB * WBITS / 8;
    constexpr int SBc = ((MXB / GSIZE_) * LB_ + 7) / 8;
    const int U16 = slot_bytes >> 4;
    const long BUFSZ = (long)KV_CHUNK_SSNF * slot_bytes;
    auto prefetch = [&](int cs, int nC, uint8_t* rb) {
        for (int idx = tid; idx < nC * U16; idx += blockDim.x) {
            const int kk = idx / U16, u16 = idx % U16;
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            __pipeline_memcpy_async(rb + (long)kk * slot_bytes + (long)u16 * 16,
                                    cache + ((long)phys * Hkv + hk) * slot_bytes + (long)u16 * 16, 16);
        }
        __pipeline_commit();
    };

    const int nchunks = (j1 > j0) ? (j1 - j0 + KV_CHUNK_SSNF - 1) / KV_CHUNK_SSNF : 0;
    if (nchunks > 0) prefetch(j0, min(KV_CHUNK_SSNF, j1 - j0), rawbuf);
    for (int ci = 0; ci < nchunks; ++ci) {
        const int cs = j0 + ci * KV_CHUNK_SSNF;
        const int nC = min(KV_CHUNK_SSNF, j1 - cs);
        const uint8_t* rb = rawbuf + (long)(ci & 1) * BUFSZ;
        const bool hasNext = (ci + 1 < nchunks);
        if (hasNext) prefetch(cs + KV_CHUNK_SSNF, min(KV_CHUNK_SSNF, j1 - cs - KV_CHUNK_SSNF),
                              rawbuf + (long)((ci + 1) & 1) * BUFSZ);
        __pipeline_wait_prior(hasNext ? 1 : 0);
        __syncthreads();

        // Pass 1a: K staging — compile-time stream unpack (same mechanism as V), (kk,blk) parallel
        for (int idx = tid; idx < nC * NB; idx += blockDim.x) {
            const int kk = idx / NB, blk = idx % NB;
            const long kblk = (long)kk * slot_bytes + (long)blk * Kblkstride;
            ksc_sh[blk * KV_CHUNK_SSNF + kk] = ssnf::e8m0_to_scale((int8_t)rb[kblk]);
            alignas(4) uint8_t ub[KUBc], shb[KSBc];
            #pragma unroll
            for (int i = 0; i < KUBc; ++i) ub[i]  = rb[kblk + 1 + i];
            #pragma unroll
            for (int i = 0; i < KSBc; ++i) shb[i] = rb[kblk + 1 + KUB + i];
            ssnf::stream_block_uspec<KU_, KGS_>(ub, shb, 0,
                [&](int kd, int w) { k8[kk * KROW + blk * MXB + kd] = (int8_t)w; });
        }
        __syncthreads();
        // Pass 1b: QK — warp-per-key, ONE int32 LDS per lane covers its 4 channels
        for (int kk = warpId; kk < nC; kk += nWarps) {
            float part[G];
            #pragma unroll
            for (int g = 0; g < G; ++g) part[g] = 0.0f;
            const int32_t w4 = *reinterpret_cast<const int32_t*>(k8 + kk * KROW + lane * 4);
            const float ksc = ksc_sh[(lane >> 3) * KV_CHUNK_SSNF + kk];   // block = (lane*4)/32
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float kval = (float)((int8_t)((w4 >> (8 * t)) & 0xff)) * ksc;
                #pragma unroll
                for (int g = 0; g < G; ++g) part[g] += q4reg[t * G + g] * kval;
            }
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                float r = warp_reduce_sum(part[g]);
                if (lane == 0) sc[g * KV_CHUNK_SSNF + kk] = r * sm_scale;
            }
        }
        __syncthreads();
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * KV_CHUNK_SSNF + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        __syncthreads();
        for (int i = tid; i < G * nC; i += blockDim.x) {
            const int g = i / nC, kk = i % nC;
            sc[g * KV_CHUNK_SSNF + kk] = __expf(sc[g * KV_CHUNK_SSNF + kk] - m_new[g]);
        }
        for (int kk = tid; kk < nC; kk += blockDim.x) {   // V staging (unchanged)
            const long vbase = (long)kk * slot_bytes + per_head_K;
            #pragma unroll 1
            for (int blk = 0; blk < NB; ++blk) {
                const long vblk = vbase + (long)blk * Vblkstride;
                vsc_sh[blk * KV_CHUNK_SSNF + kk] = ssnf::e8m0_to_scale((int8_t)rb[vblk]);
                alignas(4) uint8_t ub[UBc], shb[SBc];
                #pragma unroll
                for (int i = 0; i < UBc; ++i) ub[i]  = rb[vblk + 1 + i];
                #pragma unroll
                for (int i = 0; i < SBc; ++i) shb[i] = rb[vblk + 1 + VUB + i];
                ssnf::stream_block_uspec<LB_, GSIZE_>(ub, shb, 0,
                    [&](int kd, int w) { v8buf[(long)blk * MXB * CH + kd * CH + kk] = (int8_t)w; });
            }
        }
        __syncthreads();
        const int8_t* vrow = v8buf + (long)dblk * MXB * CH + (long)dkd * CH;
        const float* vscb = vsc_sh + dblk * KV_CHUNK_SSNF;
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        int kk = 0;
        for (; kk + 3 < nC; kk += 4) {
            const int32_t w4 = *reinterpret_cast<const int32_t*>(vrow + kk);
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float vval = (float)((int8_t)((w4 >> (8 * t)) & 0xff)) * vscb[kk + t];
                #pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float p = sc[g * KV_CHUNK_SSNF + kk + t];
                    lsum[g] += p; a[g] += p * vval;
                }
            }
        }
        for (; kk < nC; ++kk) {
            const float vval = (float)vrow[kk] * vscb[kk];
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = sc[g * KV_CHUNK_SSNF + kk];
                lsum[g] += p; a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();
    }
    if (S == 1) {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
    } else {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            part_o[(((long)b * H + (h0 + g)) * S + s) * D + tid] = acc[g];
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                part_m[((long)b * H + (h0 + g)) * S + s] = m_i[g];
                part_l[((long)b * H + (h0 + g)) * S + s] = l_i[g];
            }
        }
    }
}

// ===== TC-QK asym cpa twin : K int8-staging
// (kstage) kept, QK dot replaced by ldmatrix-less manual-fragment HMMA: q as zero-padded bf16 tile
// [16][136] in shared, warp w = key-group w (8 keys), 8x mma.m16n8k16 accumulate over k-slices
// (B-frag = staged int8 x ksc, 2xLDS16+cvt per lane), C-frag extraction -> sc. FMA+warp-reduce
// absorbed into mma; the score difference vs the FMA path is bf16 rounding only. =====
template<int G, int LB_, int GSIZE_, int KGS_T = 32>
__global__ void kv_decode_paged_v8_asym_tcqk_cpa_kernel(
        const __nv_bfloat16* __restrict__ q,
        const uint8_t* __restrict__ cache,
        const int* __restrict__ block_table,
        const int* __restrict__ seq_lens,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int B, int H, int Hkv, int D, int NB, int ku, int kgs, int KUB, int KSB, int VUB, int VSB,
        int block_size, int slot_bytes, int max_blocks, int per_head_K, float sm_scale, int S) {
    constexpr int KU_ = 3, KGS_ = KGS_T;
    constexpr int KUBc = MXB * (8 - KU_) / 8;
    constexpr int KSBc = ((MXB / KGS_) * KU_ + 7) / 8;
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int s = blockIdx.z;
    const int h0 = hk * G;
    const int L = seq_lens[b];
    const int key_tile = (L + S - 1) / S;
    const int j0 = s * key_tile, j1 = min(j0 + key_tile, L);
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int Kblkstride = 1 + KUB + KSB;
    const int Vblkstride = 1 + VUB + VSB;
    const int CH = KV_CHUNK_SSNF + 4;

    extern __shared__ float sm[];
    float* q_sh   = sm;
    float* sc     = sm + G * D;
    float* vsc_sh = sc + G * KV_CHUNK_SSNF;
    int8_t* v8buf = (int8_t*)(vsc_sh + NB * KV_CHUNK_SSNF);
    uint8_t* rawbuf = (uint8_t*)(v8buf + NB * MXB * CH);
    const int KROW = D + 4;
    int8_t* k8 = (int8_t*)(rawbuf + 2 * (long)KV_CHUNK_SSNF * slot_bytes);
    float* ksc_sh = (float*)(k8 + KV_CHUNK_SSNF * KROW);
    constexpr int QP = 136;
    __nv_bfloat16* qtile = (__nv_bfloat16*)(ksc_sh + NB * KV_CHUNK_SSNF);
    for (int i = tid; i < G * D; i += blockDim.x)
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();
    for (int i = tid; i < 16 * QP; i += blockDim.x) {
        const int r = i / QP, cc = i % QP;
        qtile[i] = (r < G && cc < D) ? __float2bfloat16(q_sh[r * D + cc]) : __float2bfloat16(0.0f);
    }
    __syncthreads();

    const int dblk = tid / MXB, dkd = tid % MXB;
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }

    constexpr int WBITS = 8 - LB_;
    constexpr int UBc = MXB * WBITS / 8;
    constexpr int SBc = ((MXB / GSIZE_) * LB_ + 7) / 8;
    const int U16 = slot_bytes >> 4;
    const long BUFSZ = (long)KV_CHUNK_SSNF * slot_bytes;
    auto prefetch = [&](int cs, int nC, uint8_t* rb) {
        for (int idx = tid; idx < nC * U16; idx += blockDim.x) {
            const int kk = idx / U16, u16 = idx % U16;
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            __pipeline_memcpy_async(rb + (long)kk * slot_bytes + (long)u16 * 16,
                                    cache + ((long)phys * Hkv + hk) * slot_bytes + (long)u16 * 16, 16);
        }
        __pipeline_commit();
    };

    const int nchunks = (j1 > j0) ? (j1 - j0 + KV_CHUNK_SSNF - 1) / KV_CHUNK_SSNF : 0;
    if (nchunks > 0) prefetch(j0, min(KV_CHUNK_SSNF, j1 - j0), rawbuf);
    for (int ci = 0; ci < nchunks; ++ci) {
        const int cs = j0 + ci * KV_CHUNK_SSNF;
        const int nC = min(KV_CHUNK_SSNF, j1 - cs);
        const uint8_t* rb = rawbuf + (long)(ci & 1) * BUFSZ;
        const bool hasNext = (ci + 1 < nchunks);
        if (hasNext) prefetch(cs + KV_CHUNK_SSNF, min(KV_CHUNK_SSNF, j1 - cs - KV_CHUNK_SSNF),
                              rawbuf + (long)((ci + 1) & 1) * BUFSZ);
        __pipeline_wait_prior(hasNext ? 1 : 0);
        __syncthreads();

        // Pass 1a: K staging (unchanged from kstage)
        for (int idx = tid; idx < nC * NB; idx += blockDim.x) {
            const int kk = idx / NB, blk = idx % NB;
            const long kblk = (long)kk * slot_bytes + (long)blk * Kblkstride;
            ksc_sh[blk * KV_CHUNK_SSNF + kk] = ssnf::e8m0_to_scale((int8_t)rb[kblk]);
            alignas(4) uint8_t ub[KUBc], shb[KSBc];
            #pragma unroll
            for (int i = 0; i < KUBc; ++i) ub[i]  = rb[kblk + 1 + i];
            #pragma unroll
            for (int i = 0; i < KSBc; ++i) shb[i] = rb[kblk + 1 + KUB + i];
            ssnf::stream_block_uspec<KU_, KGS_>(ub, shb, 0,
                [&](int kd, int w) { k8[kk * KROW + blk * MXB + kd] = (int8_t)w; });
        }
        __syncthreads();
        // Pass 1b: QK via HMMA — warp w owns key-group w; mma absorbs FMA + warp-reduce
        {
            // warp-stride loop: when the staging chunk is larger than nWarps*8 keys (e.g. the
            // 64-key SSNF chunk with 4 warps), each warp owns SEVERAL 8-key groups; a single
            // ng=warpId would cover only the first nWarps*8 keys of the chunk.
            for (int ng = warpId; ng * 8 < nC; ng += nWarps) {
                float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
                const int bn = ng * 8 + (lane >> 2);
                const bool kvld = (bn < nC);
                #pragma unroll
                for (int s5 = 0; s5 < 8; ++s5) {
                    uint32_t a0, a1, a2, a3;
                    {
                        const int ar = lane >> 2, ac = s5 * 16 + (lane & 3) * 2;
                        __nv_bfloat162 h;
                        h.x = qtile[ar * QP + ac];       h.y = qtile[ar * QP + ac + 1];       a0 = *reinterpret_cast<uint32_t*>(&h);
                        h.x = qtile[(ar + 8) * QP + ac]; h.y = qtile[(ar + 8) * QP + ac + 1]; a1 = *reinterpret_cast<uint32_t*>(&h);
                        h.x = qtile[ar * QP + ac + 8];   h.y = qtile[ar * QP + ac + 9];       a2 = *reinterpret_cast<uint32_t*>(&h);
                        h.x = qtile[(ar + 8) * QP + ac + 8]; h.y = qtile[(ar + 8) * QP + ac + 9]; a3 = *reinterpret_cast<uint32_t*>(&h);
                    }
                    uint32_t b0 = 0, b1 = 0;
                    if (kvld) {
                        const int8_t* krow = k8 + bn * KROW + s5 * 16 + (lane & 3) * 2;
                        const float ks0 = ksc_sh[((s5 * 16 + (lane & 3) * 2) >> 5) * KV_CHUNK_SSNF + bn];
                        const float ks1 = ksc_sh[((s5 * 16 + (lane & 3) * 2 + 8) >> 5) * KV_CHUNK_SSNF + bn];
                        __nv_bfloat162 lo = __floats2bfloat162_rn((float)krow[0] * ks0, (float)krow[1] * ks0);
                        __nv_bfloat162 hi = __floats2bfloat162_rn((float)krow[8] * ks1, (float)krow[9] * ks1);
                        b0 = *reinterpret_cast<uint32_t*>(&lo);
                        b1 = *reinterpret_cast<uint32_t*>(&hi);
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
                const int crow = lane >> 2, ccol0 = ng * 8 + (lane & 3) * 2;
                if (crow < G) {
                    if (ccol0 < nC)     sc[crow * KV_CHUNK_SSNF + ccol0]     = c0 * sm_scale;
                    if (ccol0 + 1 < nC) sc[crow * KV_CHUNK_SSNF + ccol0 + 1] = c1 * sm_scale;
                }
            }
        }
        __syncthreads();
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * KV_CHUNK_SSNF + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        __syncthreads();
        for (int i = tid; i < G * nC; i += blockDim.x) {
            const int g = i / nC, kk = i % nC;
            sc[g * KV_CHUNK_SSNF + kk] = __expf(sc[g * KV_CHUNK_SSNF + kk] - m_new[g]);
        }
        for (int kk = tid; kk < nC; kk += blockDim.x) {
            const long vbase = (long)kk * slot_bytes + per_head_K;
            #pragma unroll 1
            for (int blk = 0; blk < NB; ++blk) {
                const long vblk = vbase + (long)blk * Vblkstride;
                vsc_sh[blk * KV_CHUNK_SSNF + kk] = ssnf::e8m0_to_scale((int8_t)rb[vblk]);
                alignas(4) uint8_t ub[UBc], shb[SBc];
                #pragma unroll
                for (int i = 0; i < UBc; ++i) ub[i]  = rb[vblk + 1 + i];
                #pragma unroll
                for (int i = 0; i < SBc; ++i) shb[i] = rb[vblk + 1 + VUB + i];
                ssnf::stream_block_uspec<LB_, GSIZE_>(ub, shb, 0,
                    [&](int kd, int w) { v8buf[(long)blk * MXB * CH + kd * CH + kk] = (int8_t)w; });
            }
        }
        __syncthreads();
        const int8_t* vrow = v8buf + (long)dblk * MXB * CH + (long)dkd * CH;
        const float* vscb = vsc_sh + dblk * KV_CHUNK_SSNF;
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        int kk = 0;
        for (; kk + 3 < nC; kk += 4) {
            const int32_t w4 = *reinterpret_cast<const int32_t*>(vrow + kk);
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float vval = (float)((int8_t)((w4 >> (8 * t)) & 0xff)) * vscb[kk + t];
                #pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float p = sc[g * KV_CHUNK_SSNF + kk + t];
                    lsum[g] += p; a[g] += p * vval;
                }
            }
        }
        for (; kk < nC; ++kk) {
            const float vval = (float)vrow[kk] * vscb[kk];
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = sc[g * KV_CHUNK_SSNF + kk];
                lsum[g] += p; a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();
    }
    if (S == 1) {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
    } else {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            part_o[(((long)b * H + (h0 + g)) * S + s) * D + tid] = acc[g];
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                part_m[((long)b * H + (h0 + g)) * S + s] = m_i[g];
                part_l[((long)b * H + (h0 + g)) * S + s] = l_i[g];
            }
        }
    }
}

// ===== 4+1 SPLIT asym cpa twin : K plane stored as
// [nib 4x16 | highbit 4x4 | scale x4 | shared x4] (same codes/bytes, byte-aligned; written by the
// k4p1 append). Pass-1 consumes K DIRECTLY from the prefetched rb (no staging, no extra barrier):
// per key per lane 2x LDS.32 + folded dequant; per-warp running max feeds the softmax scan (32->nW
// LDS). Bit-exact vs the kstage kernel. =====
template<int G, int LB_, int GSIZE_>
__global__ void kv_decode_paged_v8_asym_k4p1_cpa_kernel(
        const __nv_bfloat16* __restrict__ q,
        const uint8_t* __restrict__ cache,
        const int* __restrict__ block_table,
        const int* __restrict__ seq_lens,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int B, int H, int Hkv, int D, int NB, int ku, int kgs, int KUB, int KSB, int VUB, int VSB,
        int block_size, int slot_bytes, int max_blocks, int per_head_K, float sm_scale, int S) {
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int s = blockIdx.z;
    const int h0 = hk * G;
    const int L = seq_lens[b];
    const int key_tile = (L + S - 1) / S;
    const int j0 = s * key_tile, j1 = min(j0 + key_tile, L);
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int Vblkstride = 1 + VUB + VSB;
    const int CH = KV_CHUNK_SSNF + 4;

    extern __shared__ float sm[];
    float* q_sh   = sm;
    float* sc     = sm + G * D;
    float* vsc_sh = sc + G * KV_CHUNK_SSNF;
    int8_t* v8buf = (int8_t*)(vsc_sh + NB * KV_CHUNK_SSNF);
    uint8_t* rawbuf = (uint8_t*)(v8buf + NB * MXB * CH);
    for (int i = tid; i < G * D; i += blockDim.x)
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();

    const int dblk = tid / MXB, dkd = tid % MXB;
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }

    float q4reg[4 * G];                       // q for d = lane*4 + t (matches the int32 K reads)
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int d = lane * 4 + t;
        #pragma unroll
        for (int g = 0; g < G; ++g) q4reg[t * G + g] = (d < D) ? q_sh[g * D + d] : 0.0f;
    }

    constexpr int WBITS = 8 - LB_;
    constexpr int UBc = MXB * WBITS / 8;
    constexpr int SBc = ((MXB / GSIZE_) * LB_ + 7) / 8;
    const int U16 = slot_bytes >> 4;
    const long BUFSZ = (long)KV_CHUNK_SSNF * slot_bytes;
    float* wmax = (float*)(rawbuf + 2 * BUFSZ);                        // [G][16] per-warp maxes
    auto prefetch = [&](int cs, int nC, uint8_t* rb) {
        for (int idx = tid; idx < nC * U16; idx += blockDim.x) {
            const int kk = idx / U16, u16 = idx % U16;
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            __pipeline_memcpy_async(rb + (long)kk * slot_bytes + (long)u16 * 16,
                                    cache + ((long)phys * Hkv + hk) * slot_bytes + (long)u16 * 16, 16);
        }
        __pipeline_commit();
    };

    const int nchunks = (j1 > j0) ? (j1 - j0 + KV_CHUNK_SSNF - 1) / KV_CHUNK_SSNF : 0;
    if (nchunks > 0) prefetch(j0, min(KV_CHUNK_SSNF, j1 - j0), rawbuf);
    for (int ci = 0; ci < nchunks; ++ci) {
        const int cs = j0 + ci * KV_CHUNK_SSNF;
        const int nC = min(KV_CHUNK_SSNF, j1 - cs);
        const uint8_t* rb = rawbuf + (long)(ci & 1) * BUFSZ;
        const bool hasNext = (ci + 1 < nchunks);
        if (hasNext) prefetch(cs + KV_CHUNK_SSNF, min(KV_CHUNK_SSNF, j1 - cs - KV_CHUNK_SSNF),
                              rawbuf + (long)((ci + 1) & 1) * BUFSZ);
        __pipeline_wait_prior(hasNext ? 1 : 0);
        __syncthreads();

        // Pass 1: 4+1 direct QK (folded dequant + per-warp running max)
        {
            float wm[G];
            #pragma unroll
            for (int g = 0; g < G; ++g) wm[g] = -INFINITY;
            for (int kk = warpId; kk < nC; kk += nWarps) {
                const long kb = (long)kk * slot_bytes;
                const int bb = lane >> 3, c = (lane & 7) * 4;
                const uint32_t nw = *reinterpret_cast<const uint32_t*>(rb + kb + bb * 16 + ((c >> 3) << 2));
                const uint32_t bw = *reinterpret_cast<const uint32_t*>(rb + kb + 64 + (bb << 2));
                const float ksc = ssnf::e8m0_to_scale((int8_t)rb[kb + 80 + bb]);
                const int shc = ((int)((rb[kb + 84 + bb] & 7u) << 29)) >> 29;
                const float k8s = ksc * 8.0f, shs = (float)shc * ksc;
                float part[G];
                #pragma unroll
                for (int g = 0; g < G; ++g) part[g] = 0.0f;
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const int nib = (nw >> (((c & 4) + t) << 2)) & 0xF;
                    const int hb = (bw >> (c + t)) & 1;
                    const int up5 = ((nib | (hb << 4)) ^ 16) - 16;
                    const float kval = (float)up5 * k8s + shs;
                    #pragma unroll
                    for (int g = 0; g < G; ++g) part[g] += q4reg[t * G + g] * kval;
                }
                #pragma unroll
                for (int g = 0; g < G; ++g) {
                    float r = warp_reduce_sum(part[g]);
                    if (lane == 0) { r *= sm_scale; sc[g * KV_CHUNK_SSNF + kk] = r; wm[g] = fmaxf(wm[g], r); }
                }
            }
            if (lane == 0) {
                #pragma unroll
                for (int g = 0; g < G; ++g) wmax[g * 16 + warpId] = wm[g];
            }
        }
        __syncthreads();
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int w = 0; w < nWarps; ++w) m_chunk = fmaxf(m_chunk, wmax[g * 16 + w]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        __syncthreads();
        for (int i = tid; i < G * nC; i += blockDim.x) {
            const int g = i / nC, kk = i % nC;
            sc[g * KV_CHUNK_SSNF + kk] = __expf(sc[g * KV_CHUNK_SSNF + kk] - m_new[g]);
        }
        for (int kk = tid; kk < nC; kk += blockDim.x) {   // V staging (unchanged)
            const long vbase = (long)kk * slot_bytes + per_head_K;
            #pragma unroll 1
            for (int blk = 0; blk < NB; ++blk) {
                const long vblk = vbase + (long)blk * Vblkstride;
                vsc_sh[blk * KV_CHUNK_SSNF + kk] = ssnf::e8m0_to_scale((int8_t)rb[vblk]);
                alignas(4) uint8_t ub[UBc], shb[SBc];
                #pragma unroll
                for (int i = 0; i < UBc; ++i) ub[i]  = rb[vblk + 1 + i];
                #pragma unroll
                for (int i = 0; i < SBc; ++i) shb[i] = rb[vblk + 1 + VUB + i];
                ssnf::stream_block_uspec<LB_, GSIZE_>(ub, shb, 0,
                    [&](int kd, int w) { v8buf[(long)blk * MXB * CH + kd * CH + kk] = (int8_t)w; });
            }
        }
        __syncthreads();
        const int8_t* vrow = v8buf + (long)dblk * MXB * CH + (long)dkd * CH;
        const float* vscb = vsc_sh + dblk * KV_CHUNK_SSNF;
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        int kk = 0;
        for (; kk + 3 < nC; kk += 4) {
            const int32_t w4 = *reinterpret_cast<const int32_t*>(vrow + kk);
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float vval = (float)((int8_t)((w4 >> (8 * t)) & 0xff)) * vscb[kk + t];
                #pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float p = sc[g * KV_CHUNK_SSNF + kk + t];
                    lsum[g] += p; a[g] += p * vval;
                }
            }
        }
        for (; kk < nC; ++kk) {
            const float vval = (float)vrow[kk] * vscb[kk];
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = sc[g * KV_CHUNK_SSNF + kk];
                lsum[g] += p; a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();
    }
    if (S == 1) {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
    } else {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            part_o[(((long)b * H + (h0 + g)) * S + s) * D + tid] = acc[g];
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                part_m[((long)b * H + (h0 + g)) * S + s] = m_i[g];
                part_l[((long)b * H + (h0 + g)) * S + s] = l_i[g];
            }
        }
    }
}

// V's (vu,vgs), K runtime. Non-cpa asym kernel (int8-staged V + split-K). Separate from the symmetric
// kv_decode_paged (shipped path untouched). slot_bytes = per_head_K + per_head_V.
torch::Tensor kv_decode_paged_asym_cuda(
        torch::Tensor q, torch::Tensor cache, torch::Tensor block_table, torch::Tensor seq_lens,
        int64_t B, int64_t H, int64_t Hkv, int64_t D, int64_t NB,
        int64_t ku, int64_t kgs, int64_t KUB, int64_t KSB,
        int64_t vu, int64_t vgs, int64_t VUB, int64_t VSB,
        int64_t block_size, int64_t slot_bytes, int64_t per_head_K) {
    auto out = torch::empty({B, H, D}, torch::dtype(torch::kBFloat16).device(q.device()));
    const int max_blocks = block_table.size(1);
    const float sm_scale = 1.0f / sqrtf((float)D);
    const int G = H / Hkv;
    TORCH_CHECK(H == G * Hkv, "kv_decode_paged_asym: H must be a multiple of Hkv");
    const int threads = D;
    const size_t base_shmem = (size_t)(G * D + G * KV_CHUNK_SSNF) * sizeof(float);
    const int CH = KV_CHUNK_SSNF + 4;
    const size_t v8_shmem = base_shmem + (size_t)NB * KV_CHUNK_SSNF * sizeof(float) + (size_t)NB * MXB * CH;
    int S = 1;
    { int mult = 8; if (const char* e = getenv("SSNF_KV_SPLIT_MULT")) { int m = atoi(e); if (m > 0) mult = m; }
      static int nsm = -1; if (nsm < 0) cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
      const int base = (int)(B * Hkv); S = ((long)mult * nsm + base - 1) / base;
      const int maxS = (max_blocks * (int)block_size + 31) / 32; if (S > maxS) S = maxS; if (S < 1) S = 1; }
    const dim3 grid(B, Hkv, S);
    torch::Tensor part_o, part_m, part_l; float *po = nullptr, *pm = nullptr, *pl = nullptr;
    if (S > 1) { auto fopt = torch::dtype(torch::kFloat32).device(q.device());
        part_o = torch::empty({B, H, S, D}, fopt); part_m = torch::empty({B, H, S}, fopt); part_l = torch::empty({B, H, S}, fopt);
        po = (float*)part_o.data_ptr(); pm = (float*)part_m.data_ptr(); pl = (float*)part_l.data_ptr(); }
    const size_t cpa_shmem = v8_shmem + (size_t)2 * KV_CHUNK_SSNF * slot_bytes;   // + double-buffer
    bool use_cpa = (slot_bytes % 16 == 0);
    if (const char* e = getenv("SSNF_KV_CPA")) use_cpa = use_cpa && (atoi(e) != 0);
    // K-staged pass-1: K u3, group size 16 or 32 (kernel templated on KGS_T; the K
    // shared field is 1 byte for both). SSNF_KV_KSTAGE=0 disables.
    bool kstage = use_cpa && ku == 3 && (kgs == 32 || kgs == 16) && D == 128;
    if (const char* e = getenv("SSNF_KV_KSTAGE")) kstage = kstage && (atoi(e) != 0);
    const size_t kstage_shmem = cpa_shmem + (size_t)KV_CHUNK_SSNF * (D + 4) + (size_t)NB * KV_CHUNK_SSNF * sizeof(float);
    // 4+1 split K plane: cache written by the k4p1 append (same gate there).
    // SSNF_KV_K4P1=0 disables BOTH sides -> old layout + kstage kernel. kgs32 only (not generalized).
    bool k4p1 = false;                       // OPT-IN (SSNF_KV_K4P1=1): an integrated A/B measured a
    // tie at the deployed shapes -- the kstage build is already past the instruction-bound regime
    // where the 4+1 cuts helped the prototype.
    if (const char* e = getenv("SSNF_KV_K4P1")) k4p1 = kstage && kgs == 32 && (atoi(e) != 0);
    const size_t k4p1_shmem = cpa_shmem + 64 * sizeof(float);            // + wmax[G][16]
    // TC-QK (DEFAULT ON, SSNF_KV_TCQK=0 disables): HMMA QK on the kstage substrate; mma absorbs
    // the CUDA-core FMA+warp-reduce. A deployed A/B measured it at least as fast at every shape;
    // the score difference is bf16 rounding only. Storage unchanged.
    bool tcqk = kstage && (G == 4);
    if (const char* e = getenv("SSNF_KV_TCQK")) tcqk = tcqk && (atoi(e) != 0);
    const size_t tcqk_shmem = kstage_shmem + (size_t)16 * 136 * 2;        // + q bf16 tile
    #define ASYM_KARGS \
        (const __nv_bfloat16*)q.data_ptr(), (const uint8_t*)cache.data_ptr(), \
        (const int*)block_table.data_ptr(), (const int*)seq_lens.data_ptr(), \
        (__nv_bfloat16*)out.data_ptr(), po, pm, pl, \
        B, H, Hkv, D, NB, (int)ku, (int)kgs, (int)KUB, (int)KSB, (int)VUB, (int)VSB, \
        block_size, slot_bytes, max_blocks, (int)per_head_K, sm_scale, S
    auto launch = [&](auto Gc, auto Uc, auto GSc) {
        constexpr int Gv = decltype(Gc)::value, LBv = decltype(Uc)::value, GSIZEv = decltype(GSc)::value;
        if (tcqk) {   // K int8-staging + HMMA QK; dispatch on K group size (16 or 32)
            if (kgs == 16)
                kv_decode_paged_v8_asym_tcqk_cpa_kernel<Gv, LBv, GSIZEv, 16><<<grid, threads, tcqk_shmem>>>(ASYM_KARGS);
            else
                kv_decode_paged_v8_asym_tcqk_cpa_kernel<Gv, LBv, GSIZEv, 32><<<grid, threads, tcqk_shmem>>>(ASYM_KARGS);
        }
        else if (k4p1)
            kv_decode_paged_v8_asym_k4p1_cpa_kernel<Gv, LBv, GSIZEv><<<grid, threads, k4p1_shmem>>>(ASYM_KARGS);
        else if (kstage) {
            if (kgs == 16)
                kv_decode_paged_v8_asym_kstage_cpa_kernel<Gv, LBv, GSIZEv, 16><<<grid, threads, kstage_shmem>>>(ASYM_KARGS);
            else
                kv_decode_paged_v8_asym_kstage_cpa_kernel<Gv, LBv, GSIZEv, 32><<<grid, threads, kstage_shmem>>>(ASYM_KARGS);
        }
        else if (use_cpa)
            kv_decode_paged_v8_asym_cpa_kernel<Gv, LBv, GSIZEv><<<grid, threads, cpa_shmem>>>(ASYM_KARGS);
        else
            kv_decode_paged_v8_asym_kernel<Gv, LBv, GSIZEv><<<grid, threads, v8_shmem>>>(ASYM_KARGS); };
    #undef ASYM_KARGS
    #define ASYM_V8(GV) do {                                                                             \
        if      (vu == 2 && vgs == 8)  launch(GV, std::integral_constant<int,2>{}, std::integral_constant<int,8>{});  \
        else if (vu == 2 && vgs == 16) launch(GV, std::integral_constant<int,2>{}, std::integral_constant<int,16>{}); \
        else if (vu == 3 && vgs == 8)  launch(GV, std::integral_constant<int,3>{}, std::integral_constant<int,8>{});  \
        else if (vu == 3 && vgs == 16) launch(GV, std::integral_constant<int,3>{}, std::integral_constant<int,16>{}); \
        else if (vu == 3 && vgs == 32) launch(GV, std::integral_constant<int,3>{}, std::integral_constant<int,32>{}); \
        else if (vu == 4 && vgs == 8)  launch(GV, std::integral_constant<int,4>{}, std::integral_constant<int,8>{});  \
        else if (vu == 4 && vgs == 16) launch(GV, std::integral_constant<int,4>{}, std::integral_constant<int,16>{}); \
        else                           launch(GV, std::integral_constant<int,4>{}, std::integral_constant<int,32>{}); \
    } while (0)
    switch (G) {
        case 1: ASYM_V8((std::integral_constant<int,1>{})); break;
        case 2: ASYM_V8((std::integral_constant<int,2>{})); break;
        case 4: ASYM_V8((std::integral_constant<int,4>{})); break;
        case 8: ASYM_V8((std::integral_constant<int,8>{})); break;
        default: TORCH_CHECK(false, "kv_decode_paged_asym: unsupported GQA group G=", G);
    }
    #undef ASYM_V8
    if (S > 1)
        kv_paged_combine_kernel<<<dim3(H, B), threads>>>(po, pm, pl,
            (__nv_bfloat16*)out.data_ptr(), H, D, S);
    return out;
}

// ===== FAIR MXINT8 twin of kv_decode_paged (same GQA-fused flash-decode vehicle) =====
// MXINT8 paged cache slot: per token-head = [ K: {scale(1)|int8 code x 32} x NB | V: same ],
// blkstride = 1+32, per_head = NB*33. Unpack is a plain int8 read x per-block E8M0 scale
// (vs SSNF's sub-byte unpack_ssnf_kv_elem_bfe) — the ONLY difference from kv_decode_paged_kernel.
// SPLIT-K twin (grid.z=S), fair mirror of the SSNF v8 split so low-batch comparisons stay
// same-vehicle. S==1 keeps the original direct-to-out path.
template<int G>
__global__ void mxint8_kv_decode_paged_kernel(
        const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ cache,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int B, int H, int Hkv, int D, int NB,
        int block_size, int slot_bytes, int max_blocks, int per_head, float sm_scale, int S) {
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int s = blockIdx.z;
    const int h0 = hk * G;
    const int L = seq_lens[b];
    const int key_tile = (L + S - 1) / S;
    const int j0 = s * key_tile, j1 = min(j0 + key_tile, L);
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int blkstride = 1 + MXB;                  // scale(1) + 32 int8 codes
    extern __shared__ float sm[];
    float* q_sh = sm; float* sc = sm + G * D;
    for (int i = tid; i < G * D; i += blockDim.x)
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();
    const int dblk = tid / MXB, dkd = tid % MXB;
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }
    for (int cs = j0; cs < j1; cs += KV_CHUNK_SSNF) {
        const int nC = min(KV_CHUNK_SSNF, j1 - cs);
        for (int kk = warpId; kk < nC; kk += nWarps) {
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            const long sbase = ((long)phys * Hkv + hk) * slot_bytes;
            float part[G];
            #pragma unroll
            for (int g = 0; g < G; ++g) part[g] = 0.0f;
            for (int d = lane; d < D; d += 32) {
                const int blk = d / MXB, kd = d % MXB;
                const long kblk = sbase + (long)blk * blkstride;
                const float ksc = ssnf::e8m0_to_scale((int8_t)cache[kblk]);
                const float kval = ksc * (float)((int8_t)cache[kblk + 1 + kd]);
                #pragma unroll
                for (int g = 0; g < G; ++g) part[g] += q_sh[g * D + d] * kval;
            }
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                float r = warp_reduce_sum(part[g]);
                if (lane == 0) sc[g * KV_CHUNK_SSNF + kk] = r * sm_scale;
            }
        }
        __syncthreads();
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * KV_CHUNK_SSNF + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        for (int kk = 0; kk < nC; ++kk) {
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            const long vblk = ((long)phys * Hkv + hk) * slot_bytes + per_head
                              + (long)dblk * blkstride;
            const float vsc = ssnf::e8m0_to_scale((int8_t)cache[vblk]);
            const float vval = vsc * (float)((int8_t)cache[vblk + 1 + dkd]);
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = __expf(sc[g * KV_CHUNK_SSNF + kk] - m_new[g]);
                lsum[g] += p;
                a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();
    }
    if (S == 1) {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
    } else {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            part_o[(((long)b * H + (h0 + g)) * S + s) * D + tid] = acc[g];
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                part_m[((long)b * H + (h0 + g)) * S + s] = m_i[g];
                part_l[((long)b * H + (h0 + g)) * S + s] = l_i[g];
            }
        }
    }
}

// ===== MXINT8 TC-QK cpa twin : the K-staging + HMMA-QK path ported from the SSNF asym
// kernel so the baseline runs the SAME kernel feature set (fairness: measured 5.8-6.4% of SSNF throughput coming from kstage+TC-QK, which only
// SSNF had). Only the staging step differs: MXINT8 K/V codes ARE int8, so the unpack is a copy.
// Everything downstream (bf16 q-tile, mma.m16n8k16, softmax, PV) is byte-identical to the SSNF twin.
template<int G>
__global__ void mxint8_kv_decode_paged_tcqk_cpa_kernel(
        const __nv_bfloat16* __restrict__ q,
        const uint8_t* __restrict__ cache,
        const int* __restrict__ block_table,
        const int* __restrict__ seq_lens,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int B, int H, int Hkv, int D, int NB,
        int block_size, int slot_bytes, int max_blocks, int per_head, float sm_scale, int S) {
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int s = blockIdx.z;
    const int h0 = hk * G;
    const int L = seq_lens[b];
    const int key_tile = (L + S - 1) / S;
    const int j0 = s * key_tile, j1 = min(j0 + key_tile, L);
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int Kblkstride = 1 + MXB;          // E8M0 scale + 32 int8 codes
    const int Vblkstride = 1 + MXB;
    const int CH = KV_CHUNK + 4;

    extern __shared__ float sm[];
    float* q_sh   = sm;
    float* sc     = sm + G * D;
    float* vsc_sh = sc + G * KV_CHUNK;
    int8_t* v8buf = (int8_t*)(vsc_sh + NB * KV_CHUNK);
    uint8_t* rawbuf = (uint8_t*)(v8buf + NB * MXB * CH);
    const int KROW = D + 4;
    int8_t* k8 = (int8_t*)(rawbuf + 2 * (long)KV_CHUNK * slot_bytes);
    float* ksc_sh = (float*)(k8 + KV_CHUNK * KROW);
    constexpr int QP = 136;
    __nv_bfloat16* qtile = (__nv_bfloat16*)(ksc_sh + NB * KV_CHUNK);
    for (int i = tid; i < G * D; i += blockDim.x)
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();
    for (int i = tid; i < 16 * QP; i += blockDim.x) {
        const int r = i / QP, cc = i % QP;
        qtile[i] = (r < G && cc < D) ? __float2bfloat16(q_sh[r * D + cc]) : __float2bfloat16(0.0f);
    }
    __syncthreads();

    const int dblk = tid / MXB, dkd = tid % MXB;
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }

    const int cpu = (slot_bytes % 16 == 0) ? 16 : ((slot_bytes % 8 == 0) ? 8 : 4);
    const int NU = slot_bytes / cpu;
    const long BUFSZ = (long)KV_CHUNK * slot_bytes;
    auto prefetch = [&](int cs, int nC, uint8_t* rb) {
        for (int idx = tid; idx < nC * NU; idx += blockDim.x) {
            const int kk = idx / NU, uu = idx % NU;
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            __pipeline_memcpy_async(rb + (long)kk * slot_bytes + (long)uu * cpu,
                                    cache + ((long)phys * Hkv + hk) * slot_bytes + (long)uu * cpu, cpu);
        }
        __pipeline_commit();
    };

    const int nchunks = (j1 > j0) ? (j1 - j0 + KV_CHUNK - 1) / KV_CHUNK : 0;
    if (nchunks > 0) prefetch(j0, min(KV_CHUNK, j1 - j0), rawbuf);
    for (int ci = 0; ci < nchunks; ++ci) {
        const int cs = j0 + ci * KV_CHUNK;
        const int nC = min(KV_CHUNK, j1 - cs);
        const uint8_t* rb = rawbuf + (long)(ci & 1) * BUFSZ;
        const bool hasNext = (ci + 1 < nchunks);
        if (hasNext) prefetch(cs + KV_CHUNK, min(KV_CHUNK, j1 - cs - KV_CHUNK),
                              rawbuf + (long)((ci + 1) & 1) * BUFSZ);
        __pipeline_wait_prior(hasNext ? 1 : 0);
        __syncthreads();

        // Pass 1a: K staging (unchanged from kstage)
        for (int idx = tid; idx < nC * NB; idx += blockDim.x) {
            const int kk = idx / NB, blk = idx % NB;
            const long kblk = (long)kk * slot_bytes + (long)blk * Kblkstride;
            ksc_sh[blk * KV_CHUNK + kk] = ssnf::e8m0_to_scale((int8_t)rb[kblk]);
            #pragma unroll
            for (int kd = 0; kd < MXB; ++kd)
                k8[kk * KROW + blk * MXB + kd] = (int8_t)rb[kblk + 1 + kd];
        }
        __syncthreads();
        // Pass 1b: QK via HMMA — warp w owns key-group w; mma absorbs FMA + warp-reduce
        {
            // warp-stride loop: when the staging chunk is larger than nWarps*8 keys (e.g. the
            // 64-key SSNF chunk with 4 warps), each warp owns SEVERAL 8-key groups; a single
            // ng=warpId would cover only the first nWarps*8 keys of the chunk.
            for (int ng = warpId; ng * 8 < nC; ng += nWarps) {
                float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
                const int bn = ng * 8 + (lane >> 2);
                const bool kvld = (bn < nC);
                #pragma unroll
                for (int s5 = 0; s5 < 8; ++s5) {
                    uint32_t a0, a1, a2, a3;
                    {
                        const int ar = lane >> 2, ac = s5 * 16 + (lane & 3) * 2;
                        __nv_bfloat162 h;
                        h.x = qtile[ar * QP + ac];       h.y = qtile[ar * QP + ac + 1];       a0 = *reinterpret_cast<uint32_t*>(&h);
                        h.x = qtile[(ar + 8) * QP + ac]; h.y = qtile[(ar + 8) * QP + ac + 1]; a1 = *reinterpret_cast<uint32_t*>(&h);
                        h.x = qtile[ar * QP + ac + 8];   h.y = qtile[ar * QP + ac + 9];       a2 = *reinterpret_cast<uint32_t*>(&h);
                        h.x = qtile[(ar + 8) * QP + ac + 8]; h.y = qtile[(ar + 8) * QP + ac + 9]; a3 = *reinterpret_cast<uint32_t*>(&h);
                    }
                    uint32_t b0 = 0, b1 = 0;
                    if (kvld) {
                        const int8_t* krow = k8 + bn * KROW + s5 * 16 + (lane & 3) * 2;
                        const float ks0 = ksc_sh[((s5 * 16 + (lane & 3) * 2) >> 5) * KV_CHUNK + bn];
                        const float ks1 = ksc_sh[((s5 * 16 + (lane & 3) * 2 + 8) >> 5) * KV_CHUNK + bn];
                        __nv_bfloat162 lo = __floats2bfloat162_rn((float)krow[0] * ks0, (float)krow[1] * ks0);
                        __nv_bfloat162 hi = __floats2bfloat162_rn((float)krow[8] * ks1, (float)krow[9] * ks1);
                        b0 = *reinterpret_cast<uint32_t*>(&lo);
                        b1 = *reinterpret_cast<uint32_t*>(&hi);
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
                const int crow = lane >> 2, ccol0 = ng * 8 + (lane & 3) * 2;
                if (crow < G) {
                    if (ccol0 < nC)     sc[crow * KV_CHUNK + ccol0]     = c0 * sm_scale;
                    if (ccol0 + 1 < nC) sc[crow * KV_CHUNK + ccol0 + 1] = c1 * sm_scale;
                }
            }
        }
        __syncthreads();
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * KV_CHUNK + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        __syncthreads();
        for (int i = tid; i < G * nC; i += blockDim.x) {
            const int g = i / nC, kk = i % nC;
            sc[g * KV_CHUNK + kk] = __expf(sc[g * KV_CHUNK + kk] - m_new[g]);
        }
        for (int kk = tid; kk < nC; kk += blockDim.x) {
            const long vbase = (long)kk * slot_bytes + per_head;
            #pragma unroll 1
            for (int blk = 0; blk < NB; ++blk) {
                const long vblk = vbase + (long)blk * Vblkstride;
                vsc_sh[blk * KV_CHUNK + kk] = ssnf::e8m0_to_scale((int8_t)rb[vblk]);
                #pragma unroll
                for (int kd = 0; kd < MXB; ++kd)
                    v8buf[(long)blk * MXB * CH + kd * CH + kk] = (int8_t)rb[vblk + 1 + kd];
            }
        }
        __syncthreads();
        const int8_t* vrow = v8buf + (long)dblk * MXB * CH + (long)dkd * CH;
        const float* vscb = vsc_sh + dblk * KV_CHUNK;
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        int kk = 0;
        for (; kk + 3 < nC; kk += 4) {
            const int32_t w4 = *reinterpret_cast<const int32_t*>(vrow + kk);
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float vval = (float)((int8_t)((w4 >> (8 * t)) & 0xff)) * vscb[kk + t];
                #pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float p = sc[g * KV_CHUNK + kk + t];
                    lsum[g] += p; a[g] += p * vval;
                }
            }
        }
        for (; kk < nC; ++kk) {
            const float vval = (float)vrow[kk] * vscb[kk];
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = sc[g * KV_CHUNK + kk];
                lsum[g] += p; a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();
    }
    if (S == 1) {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
    } else {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            part_o[(((long)b * H + (h0 + g)) * S + s) * D + tid] = acc[g];
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                part_m[((long)b * H + (h0 + g)) * S + s] = m_i[g];
                part_l[((long)b * H + (h0 + g)) * S + s] = l_i[g];
            }
        }
    }
}



// ===== MXFP6 TC-QK cpa twin : the same K-staging + HMMA-QK port, this time for the
// format whose unpack IS expensive (6-bit E2M3, the slowest decode of the four). The MXINT8 port was
// slower because its codes are already int8 -- staging was pure overhead. Here the decode cost is
// real, so staging it once per chunk should pay. Staged values are bf16 with the block scale folded
// in (E2M3 decodes to a float, not an int8 code), so the B-fragment loads bf16 pairs directly.
// ===== MXINT8 TC-QK cpa twin : the K-staging + HMMA-QK path ported from the SSNF asym
// kernel so the baseline runs the SAME kernel feature set (fairness: measured 5.8-6.4% of SSNF throughput coming from kstage+TC-QK, which only
// SSNF had). Only the staging step differs: MXINT8 K/V codes ARE int8, so the unpack is a copy.
// Everything downstream (bf16 q-tile, mma.m16n8k16, softmax, PV) is byte-identical to the SSNF twin.
template<int G>
__global__ void mxfp6_kv_decode_paged_tcqk_cpa_kernel(
        const __nv_bfloat16* __restrict__ q,
        const uint8_t* __restrict__ cache,
        const int* __restrict__ block_table,
        const int* __restrict__ seq_lens,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int B, int H, int Hkv, int D, int NB,
        int block_size, int slot_bytes, int max_blocks, int per_head, float sm_scale, int S) {
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int s = blockIdx.z;
    const int h0 = hk * G;
    const int L = seq_lens[b];
    const int key_tile = (L + S - 1) / S;
    const int j0 = s * key_tile, j1 = min(j0 + key_tile, L);
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int Kblkstride = 1 + (MXB * 6) / 8;   // E8M0 scale + 32 six-bit codes = 25 B
    const int Vblkstride = 1 + (MXB * 6) / 8;
    const int CH = KV_CHUNK + 4;

    extern __shared__ float sm[];
    float* q_sh   = sm;
    float* sc     = sm + G * D;
    float* vsc_sh = sc + G * KV_CHUNK;
    __nv_bfloat16* vbf = (__nv_bfloat16*)(vsc_sh + NB * KV_CHUNK);
    uint8_t* rawbuf = (uint8_t*)(vbf + NB * MXB * CH);
    const int KROW = D + 4;
    __nv_bfloat16* kbf = (__nv_bfloat16*)(rawbuf + 2 * (long)KV_CHUNK * slot_bytes);
    float* ksc_sh = (float*)(kbf + KV_CHUNK * KROW);
    constexpr int QP = 136;
    __nv_bfloat16* qtile = (__nv_bfloat16*)(ksc_sh + NB * KV_CHUNK);
    for (int i = tid; i < G * D; i += blockDim.x)
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();
    for (int i = tid; i < 16 * QP; i += blockDim.x) {
        const int r = i / QP, cc = i % QP;
        qtile[i] = (r < G && cc < D) ? __float2bfloat16(q_sh[r * D + cc]) : __float2bfloat16(0.0f);
    }
    __syncthreads();

    const int dblk = tid / MXB, dkd = tid % MXB;
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }

    const int cpu = (slot_bytes % 16 == 0) ? 16 : ((slot_bytes % 8 == 0) ? 8 : 4);
    const int NU = slot_bytes / cpu;
    const long BUFSZ = (long)KV_CHUNK * slot_bytes;
    auto prefetch = [&](int cs, int nC, uint8_t* rb) {
        for (int idx = tid; idx < nC * NU; idx += blockDim.x) {
            const int kk = idx / NU, uu = idx % NU;
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            __pipeline_memcpy_async(rb + (long)kk * slot_bytes + (long)uu * cpu,
                                    cache + ((long)phys * Hkv + hk) * slot_bytes + (long)uu * cpu, cpu);
        }
        __pipeline_commit();
    };

    const int nchunks = (j1 > j0) ? (j1 - j0 + KV_CHUNK - 1) / KV_CHUNK : 0;
    if (nchunks > 0) prefetch(j0, min(KV_CHUNK, j1 - j0), rawbuf);
    for (int ci = 0; ci < nchunks; ++ci) {
        const int cs = j0 + ci * KV_CHUNK;
        const int nC = min(KV_CHUNK, j1 - cs);
        const uint8_t* rb = rawbuf + (long)(ci & 1) * BUFSZ;
        const bool hasNext = (ci + 1 < nchunks);
        if (hasNext) prefetch(cs + KV_CHUNK, min(KV_CHUNK, j1 - cs - KV_CHUNK),
                              rawbuf + (long)((ci + 1) & 1) * BUFSZ);
        __pipeline_wait_prior(hasNext ? 1 : 0);
        __syncthreads();

        // Pass 1a: K staging (unchanged from kstage)
        for (int idx = tid; idx < nC * NB; idx += blockDim.x) {
            const int kk = idx / NB, blk = idx % NB;
            const long kblk = (long)kk * slot_bytes + (long)blk * Kblkstride;
            ksc_sh[blk * KV_CHUNK + kk] = ssnf::e8m0_to_scale((int8_t)rb[kblk]);
            const float ksc_ = ksc_sh[blk * KV_CHUNK + kk];
            #pragma unroll
            for (int kd = 0; kd < MXB; ++kd)
                kbf[kk * KROW + blk * MXB + kd] =
                    __float2bfloat16(ksc_ * mxfp6_elem(rb + kblk + 1, kd));
        }
        __syncthreads();
        // Pass 1b: QK via HMMA — warp w owns key-group w; mma absorbs FMA + warp-reduce
        {
            // warp-stride loop: when the staging chunk is larger than nWarps*8 keys (e.g. the
            // 64-key SSNF chunk with 4 warps), each warp owns SEVERAL 8-key groups; a single
            // ng=warpId would cover only the first nWarps*8 keys of the chunk.
            for (int ng = warpId; ng * 8 < nC; ng += nWarps) {
                float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
                const int bn = ng * 8 + (lane >> 2);
                const bool kvld = (bn < nC);
                #pragma unroll
                for (int s5 = 0; s5 < 8; ++s5) {
                    uint32_t a0, a1, a2, a3;
                    {
                        const int ar = lane >> 2, ac = s5 * 16 + (lane & 3) * 2;
                        __nv_bfloat162 h;
                        h.x = qtile[ar * QP + ac];       h.y = qtile[ar * QP + ac + 1];       a0 = *reinterpret_cast<uint32_t*>(&h);
                        h.x = qtile[(ar + 8) * QP + ac]; h.y = qtile[(ar + 8) * QP + ac + 1]; a1 = *reinterpret_cast<uint32_t*>(&h);
                        h.x = qtile[ar * QP + ac + 8];   h.y = qtile[ar * QP + ac + 9];       a2 = *reinterpret_cast<uint32_t*>(&h);
                        h.x = qtile[(ar + 8) * QP + ac + 8]; h.y = qtile[(ar + 8) * QP + ac + 9]; a3 = *reinterpret_cast<uint32_t*>(&h);
                    }
                    uint32_t b0 = 0, b1 = 0;
                    if (kvld) {
                        const __nv_bfloat16* krow = kbf + bn * KROW + s5 * 16 + (lane & 3) * 2;
                        __nv_bfloat162 lo = __halves2bfloat162(krow[0], krow[1]);
                        __nv_bfloat162 hi = __halves2bfloat162(krow[8], krow[9]);
                        b0 = *reinterpret_cast<uint32_t*>(&lo);
                        b1 = *reinterpret_cast<uint32_t*>(&hi);
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
                const int crow = lane >> 2, ccol0 = ng * 8 + (lane & 3) * 2;
                if (crow < G) {
                    if (ccol0 < nC)     sc[crow * KV_CHUNK + ccol0]     = c0 * sm_scale;
                    if (ccol0 + 1 < nC) sc[crow * KV_CHUNK + ccol0 + 1] = c1 * sm_scale;
                }
            }
        }
        __syncthreads();
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * KV_CHUNK + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        __syncthreads();
        for (int i = tid; i < G * nC; i += blockDim.x) {
            const int g = i / nC, kk = i % nC;
            sc[g * KV_CHUNK + kk] = __expf(sc[g * KV_CHUNK + kk] - m_new[g]);
        }
        for (int kk = tid; kk < nC; kk += blockDim.x) {
            const long vbase = (long)kk * slot_bytes + per_head;
            #pragma unroll 1
            for (int blk = 0; blk < NB; ++blk) {
                const long vblk = vbase + (long)blk * Vblkstride;
                vsc_sh[blk * KV_CHUNK + kk] = ssnf::e8m0_to_scale((int8_t)rb[vblk]);
                const float vsc_ = vsc_sh[blk * KV_CHUNK + kk];
                #pragma unroll
                for (int kd = 0; kd < MXB; ++kd)
                    vbf[(long)blk * MXB * CH + kd * CH + kk] =
                        __float2bfloat16(vsc_ * mxfp6_elem(rb + vblk + 1, kd));
            }
        }
        __syncthreads();
        const __nv_bfloat16* vrow = vbf + (long)dblk * MXB * CH + (long)dkd * CH;
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        int kk = 0;
        for (; kk < nC; ++kk) {
            const float vval = __bfloat162float(vrow[kk]);
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = sc[g * KV_CHUNK + kk];
                lsum[g] += p; a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();
    }
    if (S == 1) {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
    } else {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            part_o[(((long)b * H + (h0 + g)) * S + s) * D + tid] = acc[g];
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                part_m[((long)b * H + (h0 + g)) * S + s] = m_i[g];
                part_l[((long)b * H + (h0 + g)) * S + s] = l_i[g];
            }
        }
    }
}





// cp.async + cooperative-softmax twin of mxint8_kv_decode_paged_kernel — the FAIR mirror of the
// SSNF v8 cpa kernel (same vehicle: prefetch raw slots to shared, read K/V from shared, softmax once).
// mxint8 slot_bytes = 2*nb*33 = 264 (nb=4) is %8 not %16, so cp.async uses a runtime 8B unit.
template<int G>
__global__ void mxint8_kv_decode_paged_cpa_kernel(
        const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ cache,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int B, int H, int Hkv, int D, int NB,
        int block_size, int slot_bytes, int max_blocks, int per_head, float sm_scale, int S) {
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int s = blockIdx.z;
    const int h0 = hk * G;
    const int L = seq_lens[b];
    const int key_tile = (L + S - 1) / S;
    const int j0 = s * key_tile, j1 = min(j0 + key_tile, L);
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int blkstride = 1 + MXB;
    extern __shared__ float sm[];
    float* q_sh = sm; float* sc = sm + G * D;
    uint8_t* rawbuf = (uint8_t*)(sc + G * KV_CHUNK);         // [2][KV_CHUNK*slot_bytes]
    for (int i = tid; i < G * D; i += blockDim.x)
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();
    const int dblk = tid / MXB, dkd = tid % MXB;
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }

    constexpr int MAXNB = 4;                             // q-hoist: cache lane's q-slice (see SSNF cpa)
    const bool qhoist = (D <= MAXNB * 32);
    float qreg[MAXNB * G];
    if (qhoist) {
        #pragma unroll
        for (int di = 0; di < MAXNB; ++di) {
            const int d = di * 32 + lane;
            #pragma unroll
            for (int g = 0; g < G; ++g) qreg[di * G + g] = (d < D) ? q_sh[g * D + d] : 0.0f;
        }
    }
    const int cpu = (slot_bytes % 16 == 0) ? 16 : ((slot_bytes % 8 == 0) ? 8 : 4);
    const int NU = slot_bytes / cpu;
    const long BUFSZ = (long)KV_CHUNK * slot_bytes;
    auto prefetch = [&](int cs, int nC, uint8_t* rb) {
        for (int idx = tid; idx < nC * NU; idx += blockDim.x) {
            const int kk = idx / NU, uu = idx % NU;
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            __pipeline_memcpy_async(rb + (long)kk * slot_bytes + (long)uu * cpu,
                                    cache + ((long)phys * Hkv + hk) * slot_bytes + (long)uu * cpu, cpu);
        }
        __pipeline_commit();
    };

    const int nchunks = (j1 > j0) ? (j1 - j0 + KV_CHUNK - 1) / KV_CHUNK : 0;
    if (nchunks > 0) prefetch(j0, min(KV_CHUNK, j1 - j0), rawbuf);
    for (int ci = 0; ci < nchunks; ++ci) {
        const int cs = j0 + ci * KV_CHUNK;
        const int nC = min(KV_CHUNK, j1 - cs);
        const uint8_t* rb = rawbuf + (long)(ci & 1) * BUFSZ;
        const bool hasNext = (ci + 1 < nchunks);
        if (hasNext) prefetch(cs + KV_CHUNK, min(KV_CHUNK, j1 - cs - KV_CHUNK),
                              rawbuf + (long)((ci + 1) & 1) * BUFSZ);
        __pipeline_wait_prior(hasNext ? 1 : 0);
        __syncthreads();
        for (int kk = warpId; kk < nC; kk += nWarps) {
            const long sbase = (long)kk * slot_bytes;
            float part[G];
            #pragma unroll
            for (int g = 0; g < G; ++g) part[g] = 0.0f;
            if (qhoist) {
                #pragma unroll
                for (int di = 0; di < MAXNB; ++di) {
                    const int d = di * 32 + lane;
                    if (d >= D) continue;
                    const int blk = d / MXB, kd = d % MXB;
                    const long kblk = sbase + (long)blk * blkstride;
                    const float ksc = ssnf::e8m0_to_scale((int8_t)rb[kblk]);
                    const float kval = ksc * (float)((int8_t)rb[kblk + 1 + kd]);
                    #pragma unroll
                    for (int g = 0; g < G; ++g) part[g] += qreg[di * G + g] * kval;
                }
            } else {
                for (int d = lane; d < D; d += 32) {
                    const int blk = d / MXB, kd = d % MXB;
                    const long kblk = sbase + (long)blk * blkstride;
                    const float ksc = ssnf::e8m0_to_scale((int8_t)rb[kblk]);
                    const float kval = ksc * (float)((int8_t)rb[kblk + 1 + kd]);
                    #pragma unroll
                    for (int g = 0; g < G; ++g) part[g] += q_sh[g * D + d] * kval;
                }
            }
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                float r = warp_reduce_sum(part[g]);
                if (lane == 0) sc[g * KV_CHUNK + kk] = r * sm_scale;
            }
        }
        __syncthreads();
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * KV_CHUNK + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        __syncthreads();                            // cooperative softmax (match the SSNF v8 vehicle)
        for (int i = tid; i < G * nC; i += blockDim.x) {
            const int g = i / nC, kk = i % nC;
            sc[g * KV_CHUNK + kk] = __expf(sc[g * KV_CHUNK + kk] - m_new[g]);
        }
        __syncthreads();
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        for (int kk = 0; kk < nC; ++kk) {
            const long vblk = (long)kk * slot_bytes + per_head + (long)dblk * blkstride;
            const float vval = ssnf::e8m0_to_scale((int8_t)rb[vblk]) * (float)((int8_t)rb[vblk + 1 + dkd]);
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = sc[g * KV_CHUNK + kk];
                lsum[g] += p; a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();
    }
    if (S == 1) {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
    } else {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            part_o[(((long)b * H + (h0 + g)) * S + s) * D + tid] = acc[g];
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                part_m[((long)b * H + (h0 + g)) * S + s] = m_i[g];
                part_l[((long)b * H + (h0 + g)) * S + s] = l_i[g];
            }
        }
    }
}

torch::Tensor mxint8_kv_decode_paged_cuda(
        torch::Tensor q, torch::Tensor cache, torch::Tensor block_table,
        torch::Tensor seq_lens, int64_t B, int64_t H, int64_t Hkv, int64_t D,
        int64_t NB, int64_t block_size, int64_t slot_bytes, int64_t per_head) {
    auto out = torch::empty({B, H, D}, torch::dtype(torch::kBFloat16).device(q.device()));
    const int max_blocks = block_table.size(1);
    const float sm_scale = 1.0f / sqrtf((float)D);
    const int G = H / Hkv;
    TORCH_CHECK(H == G * Hkv, "mxint8_kv_decode_paged: H must be a multiple of Hkv");
    const int threads = D;
    const size_t shmem = (size_t)(G * D + G * KV_CHUNK) * sizeof(float);
    // SPLIT-K (fair mirror of SSNF v8): same default mult=8 target so low-batch numbers are
    // same-vehicle. S collapses to 1 when the base grid fills the machine.
    const int L_hint = max_blocks * (int)block_size;
    int S = 1;
    {
        int mult = 8;
        if (const char* e = getenv("SSNF_KV_SPLIT_MULT")) { int m = atoi(e); if (m > 0) mult = m; }
        static int nsm = -1;
        if (nsm < 0) cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
        const int base = (int)(B * Hkv);
        S = ((long)mult * nsm + base - 1) / base;
        const int maxS = (L_hint + 31) / 32;
        if (S > maxS) S = maxS;
        if (S < 1) S = 1;
    }
    const dim3 grid(B, Hkv, S);
    torch::Tensor part_o, part_m, part_l;
    float *po = nullptr, *pm = nullptr, *pl = nullptr;
    if (S > 1) {
        auto fopt = torch::dtype(torch::kFloat32).device(q.device());
        part_o = torch::empty({B, H, S, D}, fopt);
        part_m = torch::empty({B, H, S}, fopt);
        part_l = torch::empty({B, H, S}, fopt);
        po = (float*)part_o.data_ptr(); pm = (float*)part_m.data_ptr(); pl = (float*)part_l.data_ptr();
    }
    // cp.async fair mirror (default on; SSNF_KV_CPA=0 disables). slot_bytes%4==0 always -> works.
    const size_t cpa_shmem = shmem + (size_t)2 * KV_CHUNK * slot_bytes;
    bool use_cpa = (slot_bytes % 4 == 0);
    if (const char* e = getenv("SSNF_KV_CPA")) use_cpa = use_cpa && (atoi(e) != 0);
    // K-staging + TC-QK, ported from the SSNF asym twin so both formats run the same feature set
    // (fairness). Same gates as SSNF: needs D==128 and G==4; SSNF_KV_TCQK=0 / SSNF_KV_KSTAGE=0
    // fall back to the plain cp.async kernel. Shared layout mirrors the SSNF tcqk kernel exactly.
    const int CHt = KV_CHUNK + 4;
    const size_t tcqk_shmem = cpa_shmem
        + (size_t)NB * KV_CHUNK * sizeof(float)          // vsc_sh
        + (size_t)NB * MXB * CHt                         // v8buf
        + (size_t)KV_CHUNK * (D + 4)                     // k8
        + (size_t)NB * KV_CHUNK * sizeof(float)          // ksc_sh
        + (size_t)16 * 136 * 2;                          // q bf16 tile
    // MEASURED (eval_serving/verify_mxint8_tcqk.py): the port is 26-36% SLOWER than
    // the plain cp.async kernel at every MXINT8 decode batch (B=67/35/18/9 -> 0.72/0.74/0.64/0.65x).
    // Staging only pays off when the unpack is expensive (SSNF reconstructs u3 upper+shared); MXINT8
    // codes ARE int8, so staging is pure overhead and the extra shared memory costs occupancy.
    // Kept opt-in (SSNF_MXI_TCQK=1) as the evidence for the fairness argument; DEFAULT OFF so the
    // baseline keeps its fastest kernel.
    bool mxi_tcqk = false;
    if (const char* e = getenv("SSNF_MXI_TCQK"))
        mxi_tcqk = (atoi(e) != 0) && use_cpa && D == 128 && G == 4;
    auto launch = [&](auto Gc) {
        constexpr int Gv = decltype(Gc)::value;
        if (mxi_tcqk)
            mxint8_kv_decode_paged_tcqk_cpa_kernel<Gv><<<grid, threads, tcqk_shmem>>>(
                (const __nv_bfloat16*)q.data_ptr(), (const uint8_t*)cache.data_ptr(),
                (const int*)block_table.data_ptr(), (const int*)seq_lens.data_ptr(),
                (__nv_bfloat16*)out.data_ptr(), po, pm, pl,
                B, H, Hkv, D, NB, block_size, slot_bytes, max_blocks, per_head, sm_scale, S);
        else if (use_cpa)
            mxint8_kv_decode_paged_cpa_kernel<Gv><<<grid, threads, cpa_shmem>>>(
                (const __nv_bfloat16*)q.data_ptr(), (const uint8_t*)cache.data_ptr(),
                (const int*)block_table.data_ptr(), (const int*)seq_lens.data_ptr(),
                (__nv_bfloat16*)out.data_ptr(), po, pm, pl,
                B, H, Hkv, D, NB, block_size, slot_bytes, max_blocks, per_head, sm_scale, S);
        else
            mxint8_kv_decode_paged_kernel<Gv><<<grid, threads, shmem>>>(
                (const __nv_bfloat16*)q.data_ptr(), (const uint8_t*)cache.data_ptr(),
                (const int*)block_table.data_ptr(), (const int*)seq_lens.data_ptr(),
                (__nv_bfloat16*)out.data_ptr(), po, pm, pl,
                B, H, Hkv, D, NB, block_size, slot_bytes, max_blocks, per_head, sm_scale, S);
    };
    switch (G) {
        case 1: launch(std::integral_constant<int,1>{}); break;
        case 2: launch(std::integral_constant<int,2>{}); break;
        case 4: launch(std::integral_constant<int,4>{}); break;
        case 8: launch(std::integral_constant<int,8>{}); break;
        default: TORCH_CHECK(false, "mxint8_kv_decode_paged: unsupported GQA group G=", G);
    }
    if (S > 1)
        kv_paged_combine_kernel<<<dim3(H, B), threads>>>(po, pm, pl,
            (__nv_bfloat16*)out.data_ptr(), H, D, S);
    return out;
}

// ============================================================================
// MXFP8-E4M3 paged decode — byte-for-byte clone of the mxint8 paged kernels above
// (same slot layout: 1 B E8M0 scale + 32 codes/block, same split-K + cp.async vehicle).
// The ONLY change is the element read: (int8_t) -> e4m3_to_f (hw fp8 cvt). The E8M0
// scale byte handling is identical (the write stores a fp8-sized exponent). This makes
// MXFP8-vs-MXINT8 an "element codec only" comparison at equal optimization.
// ============================================================================
template<int G>
__global__ void mxfp8_kv_decode_paged_kernel(
        const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ cache,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int B, int H, int Hkv, int D, int NB,
        int block_size, int slot_bytes, int max_blocks, int per_head, float sm_scale, int S) {
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int s = blockIdx.z;
    const int h0 = hk * G;
    const int L = seq_lens[b];
    const int key_tile = (L + S - 1) / S;
    const int j0 = s * key_tile, j1 = min(j0 + key_tile, L);
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int blkstride = 1 + MXB;                  // scale(1) + 32 e4m3 codes
    extern __shared__ float sm[];
    float* q_sh = sm; float* sc = sm + G * D;
    for (int i = tid; i < G * D; i += blockDim.x)
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();
    const int dblk = tid / MXB, dkd = tid % MXB;
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }
    for (int cs = j0; cs < j1; cs += KV_CHUNK) {
        const int nC = min(KV_CHUNK, j1 - cs);
        for (int kk = warpId; kk < nC; kk += nWarps) {
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            const long sbase = ((long)phys * Hkv + hk) * slot_bytes;
            float part[G];
            #pragma unroll
            for (int g = 0; g < G; ++g) part[g] = 0.0f;
            for (int d = lane; d < D; d += 32) {
                const int blk = d / MXB, kd = d % MXB;
                const long kblk = sbase + (long)blk * blkstride;
                const float ksc = ssnf::e8m0_to_scale((int8_t)cache[kblk]);
                const float kval = ksc * e4m3_to_f(cache[kblk + 1 + kd]);
                #pragma unroll
                for (int g = 0; g < G; ++g) part[g] += q_sh[g * D + d] * kval;
            }
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                float r = warp_reduce_sum(part[g]);
                if (lane == 0) sc[g * KV_CHUNK + kk] = r * sm_scale;
            }
        }
        __syncthreads();
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * KV_CHUNK + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        for (int kk = 0; kk < nC; ++kk) {
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            const long vblk = ((long)phys * Hkv + hk) * slot_bytes + per_head
                              + (long)dblk * blkstride;
            const float vsc = ssnf::e8m0_to_scale((int8_t)cache[vblk]);
            const float vval = vsc * e4m3_to_f(cache[vblk + 1 + dkd]);
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = __expf(sc[g * KV_CHUNK + kk] - m_new[g]);
                lsum[g] += p;
                a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();
    }
    if (S == 1) {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
    } else {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            part_o[(((long)b * H + (h0 + g)) * S + s) * D + tid] = acc[g];
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                part_m[((long)b * H + (h0 + g)) * S + s] = m_i[g];
                part_l[((long)b * H + (h0 + g)) * S + s] = l_i[g];
            }
        }
    }
}

template<int G>
__global__ void mxfp8_kv_decode_paged_cpa_kernel(
        const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ cache,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int B, int H, int Hkv, int D, int NB,
        int block_size, int slot_bytes, int max_blocks, int per_head, float sm_scale, int S) {
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int s = blockIdx.z;
    const int h0 = hk * G;
    const int L = seq_lens[b];
    const int key_tile = (L + S - 1) / S;
    const int j0 = s * key_tile, j1 = min(j0 + key_tile, L);
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int blkstride = 1 + MXB;
    extern __shared__ float sm[];
    float* q_sh = sm; float* sc = sm + G * D;
    uint8_t* rawbuf = (uint8_t*)(sc + G * KV_CHUNK);         // [2][KV_CHUNK*slot_bytes]
    for (int i = tid; i < G * D; i += blockDim.x)
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();
    const int dblk = tid / MXB, dkd = tid % MXB;
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }

    constexpr int MAXNB = 4;                             // q-hoist: cache lane's q-slice
    const bool qhoist = (D <= MAXNB * 32);
    float qreg[MAXNB * G];
    if (qhoist) {
        #pragma unroll
        for (int di = 0; di < MAXNB; ++di) {
            const int d = di * 32 + lane;
            #pragma unroll
            for (int g = 0; g < G; ++g) qreg[di * G + g] = (d < D) ? q_sh[g * D + d] : 0.0f;
        }
    }
    const int cpu = (slot_bytes % 16 == 0) ? 16 : ((slot_bytes % 8 == 0) ? 8 : 4);
    const int NU = slot_bytes / cpu;
    const long BUFSZ = (long)KV_CHUNK * slot_bytes;
    auto prefetch = [&](int cs, int nC, uint8_t* rb) {
        for (int idx = tid; idx < nC * NU; idx += blockDim.x) {
            const int kk = idx / NU, uu = idx % NU;
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            __pipeline_memcpy_async(rb + (long)kk * slot_bytes + (long)uu * cpu,
                                    cache + ((long)phys * Hkv + hk) * slot_bytes + (long)uu * cpu, cpu);
        }
        __pipeline_commit();
    };

    const int nchunks = (j1 > j0) ? (j1 - j0 + KV_CHUNK - 1) / KV_CHUNK : 0;
    if (nchunks > 0) prefetch(j0, min(KV_CHUNK, j1 - j0), rawbuf);
    for (int ci = 0; ci < nchunks; ++ci) {
        const int cs = j0 + ci * KV_CHUNK;
        const int nC = min(KV_CHUNK, j1 - cs);
        const uint8_t* rb = rawbuf + (long)(ci & 1) * BUFSZ;
        const bool hasNext = (ci + 1 < nchunks);
        if (hasNext) prefetch(cs + KV_CHUNK, min(KV_CHUNK, j1 - cs - KV_CHUNK),
                              rawbuf + (long)((ci + 1) & 1) * BUFSZ);
        __pipeline_wait_prior(hasNext ? 1 : 0);
        __syncthreads();
        for (int kk = warpId; kk < nC; kk += nWarps) {
            const long sbase = (long)kk * slot_bytes;
            float part[G];
            #pragma unroll
            for (int g = 0; g < G; ++g) part[g] = 0.0f;
            if (qhoist) {
                #pragma unroll
                for (int di = 0; di < MAXNB; ++di) {
                    const int d = di * 32 + lane;
                    if (d >= D) continue;
                    const int blk = d / MXB, kd = d % MXB;
                    const long kblk = sbase + (long)blk * blkstride;
                    const float ksc = ssnf::e8m0_to_scale((int8_t)rb[kblk]);
                    const float kval = ksc * e4m3_to_f(rb[kblk + 1 + kd]);
                    #pragma unroll
                    for (int g = 0; g < G; ++g) part[g] += qreg[di * G + g] * kval;
                }
            } else {
                for (int d = lane; d < D; d += 32) {
                    const int blk = d / MXB, kd = d % MXB;
                    const long kblk = sbase + (long)blk * blkstride;
                    const float ksc = ssnf::e8m0_to_scale((int8_t)rb[kblk]);
                    const float kval = ksc * e4m3_to_f(rb[kblk + 1 + kd]);
                    #pragma unroll
                    for (int g = 0; g < G; ++g) part[g] += q_sh[g * D + d] * kval;
                }
            }
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                float r = warp_reduce_sum(part[g]);
                if (lane == 0) sc[g * KV_CHUNK + kk] = r * sm_scale;
            }
        }
        __syncthreads();
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * KV_CHUNK + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        __syncthreads();                            // cooperative softmax
        for (int i = tid; i < G * nC; i += blockDim.x) {
            const int g = i / nC, kk = i % nC;
            sc[g * KV_CHUNK + kk] = __expf(sc[g * KV_CHUNK + kk] - m_new[g]);
        }
        __syncthreads();
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        for (int kk = 0; kk < nC; ++kk) {
            const long vblk = (long)kk * slot_bytes + per_head + (long)dblk * blkstride;
            const float vval = ssnf::e8m0_to_scale((int8_t)rb[vblk]) * e4m3_to_f(rb[vblk + 1 + dkd]);
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = sc[g * KV_CHUNK + kk];
                lsum[g] += p; a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();
    }
    if (S == 1) {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
    } else {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            part_o[(((long)b * H + (h0 + g)) * S + s) * D + tid] = acc[g];
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                part_m[((long)b * H + (h0 + g)) * S + s] = m_i[g];
                part_l[((long)b * H + (h0 + g)) * S + s] = l_i[g];
            }
        }
    }
}

torch::Tensor mxfp8_kv_decode_paged_cuda(
        torch::Tensor q, torch::Tensor cache, torch::Tensor block_table,
        torch::Tensor seq_lens, int64_t B, int64_t H, int64_t Hkv, int64_t D,
        int64_t NB, int64_t block_size, int64_t slot_bytes, int64_t per_head) {
    auto out = torch::empty({B, H, D}, torch::dtype(torch::kBFloat16).device(q.device()));
    const int max_blocks = block_table.size(1);
    const float sm_scale = 1.0f / sqrtf((float)D);
    const int G = H / Hkv;
    TORCH_CHECK(H == G * Hkv, "mxfp8_kv_decode_paged: H must be a multiple of Hkv");
    const int threads = D;
    const size_t shmem = (size_t)(G * D + G * KV_CHUNK) * sizeof(float);
    const int L_hint = max_blocks * (int)block_size;
    int S = 1;
    {
        int mult = 8;
        if (const char* e = getenv("SSNF_KV_SPLIT_MULT")) { int m = atoi(e); if (m > 0) mult = m; }
        static int nsm = -1;
        if (nsm < 0) cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
        const int base = (int)(B * Hkv);
        S = ((long)mult * nsm + base - 1) / base;
        const int maxS = (L_hint + 31) / 32;
        if (S > maxS) S = maxS;
        if (S < 1) S = 1;
    }
    const dim3 grid(B, Hkv, S);
    torch::Tensor part_o, part_m, part_l;
    float *po = nullptr, *pm = nullptr, *pl = nullptr;
    if (S > 1) {
        auto fopt = torch::dtype(torch::kFloat32).device(q.device());
        part_o = torch::empty({B, H, S, D}, fopt);
        part_m = torch::empty({B, H, S}, fopt);
        part_l = torch::empty({B, H, S}, fopt);
        po = (float*)part_o.data_ptr(); pm = (float*)part_m.data_ptr(); pl = (float*)part_l.data_ptr();
    }
    const size_t cpa_shmem = shmem + (size_t)2 * KV_CHUNK * slot_bytes;
    bool use_cpa = (slot_bytes % 4 == 0);
    if (const char* e = getenv("SSNF_KV_CPA")) use_cpa = use_cpa && (atoi(e) != 0);
    auto launch = [&](auto Gc) {
        constexpr int Gv = decltype(Gc)::value;
        if (use_cpa)
            mxfp8_kv_decode_paged_cpa_kernel<Gv><<<grid, threads, cpa_shmem>>>(
                (const __nv_bfloat16*)q.data_ptr(), (const uint8_t*)cache.data_ptr(),
                (const int*)block_table.data_ptr(), (const int*)seq_lens.data_ptr(),
                (__nv_bfloat16*)out.data_ptr(), po, pm, pl,
                B, H, Hkv, D, NB, block_size, slot_bytes, max_blocks, per_head, sm_scale, S);
        else
            mxfp8_kv_decode_paged_kernel<Gv><<<grid, threads, shmem>>>(
                (const __nv_bfloat16*)q.data_ptr(), (const uint8_t*)cache.data_ptr(),
                (const int*)block_table.data_ptr(), (const int*)seq_lens.data_ptr(),
                (__nv_bfloat16*)out.data_ptr(), po, pm, pl,
                B, H, Hkv, D, NB, block_size, slot_bytes, max_blocks, per_head, sm_scale, S);
    };
    switch (G) {
        case 1: launch(std::integral_constant<int,1>{}); break;
        case 2: launch(std::integral_constant<int,2>{}); break;
        case 4: launch(std::integral_constant<int,4>{}); break;
        case 8: launch(std::integral_constant<int,8>{}); break;
        default: TORCH_CHECK(false, "mxfp8_kv_decode_paged: unsupported GQA group G=", G);
    }
    if (S > 1)
        kv_paged_combine_kernel<<<dim3(H, B), threads>>>(po, pm, pl,
            (__nv_bfloat16*)out.data_ptr(), H, D, S);
    return out;
}

// ============================================================================
// MXFP6-E2M3 paged decode — clone of the mxfp8 paged kernels with 6-bit sub-byte codes.
// slot block = 1 B E8M0 scale + 24 B packed codes (blkstride=25); element read = mxfp6_elem
// (unpack a 6-bit e2m3 code) instead of a single byte. Everything else (split-K, cp.async,
// softmax) identical. 6.25 b/elem.
// ============================================================================
template<int G>
__global__ void mxfp6_kv_decode_paged_kernel(
        const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ cache,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int B, int H, int Hkv, int D, int NB,
        int block_size, int slot_bytes, int max_blocks, int per_head, float sm_scale, int S) {
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int s = blockIdx.z;
    const int h0 = hk * G;
    const int L = seq_lens[b];
    const int key_tile = (L + S - 1) / S;
    const int j0 = s * key_tile, j1 = min(j0 + key_tile, L);
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int blkstride = 1 + 24;                   // scale(1) + 24 packed code bytes
    extern __shared__ float sm[];
    float* q_sh = sm; float* sc = sm + G * D;
    for (int i = tid; i < G * D; i += blockDim.x)
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();
    const int dblk = tid / MXB, dkd = tid % MXB;
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }
    for (int cs = j0; cs < j1; cs += KV_CHUNK) {
        const int nC = min(KV_CHUNK, j1 - cs);
        for (int kk = warpId; kk < nC; kk += nWarps) {
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            const long sbase = ((long)phys * Hkv + hk) * slot_bytes;
            float part[G];
            #pragma unroll
            for (int g = 0; g < G; ++g) part[g] = 0.0f;
            for (int d = lane; d < D; d += 32) {
                const int blk = d / MXB, kd = d % MXB;
                const long kblk = sbase + (long)blk * blkstride;
                const float ksc = ssnf::e8m0_to_scale((int8_t)cache[kblk]);
                const float kval = ksc * mxfp6_elem(cache + kblk + 1, kd);
                #pragma unroll
                for (int g = 0; g < G; ++g) part[g] += q_sh[g * D + d] * kval;
            }
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                float r = warp_reduce_sum(part[g]);
                if (lane == 0) sc[g * KV_CHUNK + kk] = r * sm_scale;
            }
        }
        __syncthreads();
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * KV_CHUNK + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        for (int kk = 0; kk < nC; ++kk) {
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            const long vblk = ((long)phys * Hkv + hk) * slot_bytes + per_head
                              + (long)dblk * blkstride;
            const float vsc = ssnf::e8m0_to_scale((int8_t)cache[vblk]);
            const float vval = vsc * mxfp6_elem(cache + vblk + 1, dkd);
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = __expf(sc[g * KV_CHUNK + kk] - m_new[g]);
                lsum[g] += p;
                a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();
    }
    if (S == 1) {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
    } else {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            part_o[(((long)b * H + (h0 + g)) * S + s) * D + tid] = acc[g];
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                part_m[((long)b * H + (h0 + g)) * S + s] = m_i[g];
                part_l[((long)b * H + (h0 + g)) * S + s] = l_i[g];
            }
        }
    }
}

template<int G>
__global__ void mxfp6_kv_decode_paged_cpa_kernel(
        const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ cache,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int B, int H, int Hkv, int D, int NB,
        int block_size, int slot_bytes, int max_blocks, int per_head, float sm_scale, int S) {
    const int b = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int s = blockIdx.z;
    const int h0 = hk * G;
    const int L = seq_lens[b];
    const int key_tile = (L + S - 1) / S;
    const int j0 = s * key_tile, j1 = min(j0 + key_tile, L);
    const int nWarps = blockDim.x >> 5, warpId = tid >> 5, lane = tid & 31;
    const int blkstride = 1 + 24;
    extern __shared__ float sm[];
    float* q_sh = sm; float* sc = sm + G * D;
    uint8_t* rawbuf = (uint8_t*)(sc + G * KV_CHUNK);         // [2][KV_CHUNK*slot_bytes]
    for (int i = tid; i < G * D; i += blockDim.x)
        q_sh[i] = __bfloat162float(q[((long)b * H + h0) * D + i]);
    __syncthreads();
    const int dblk = tid / MXB, dkd = tid % MXB;
    float m_i[G], l_i[G], acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }

    constexpr int MAXNB = 4;
    const bool qhoist = (D <= MAXNB * 32);
    float qreg[MAXNB * G];
    if (qhoist) {
        #pragma unroll
        for (int di = 0; di < MAXNB; ++di) {
            const int d = di * 32 + lane;
            #pragma unroll
            for (int g = 0; g < G; ++g) qreg[di * G + g] = (d < D) ? q_sh[g * D + d] : 0.0f;
        }
    }
    const int cpu = (slot_bytes % 16 == 0) ? 16 : ((slot_bytes % 8 == 0) ? 8 : 4);
    const int NU = slot_bytes / cpu;
    const long BUFSZ = (long)KV_CHUNK * slot_bytes;
    auto prefetch = [&](int cs, int nC, uint8_t* rb) {
        for (int idx = tid; idx < nC * NU; idx += blockDim.x) {
            const int kk = idx / NU, uu = idx % NU;
            const int j = cs + kk;
            const int phys = block_table[(long)b * max_blocks + j / block_size] * block_size
                             + (j % block_size);
            __pipeline_memcpy_async(rb + (long)kk * slot_bytes + (long)uu * cpu,
                                    cache + ((long)phys * Hkv + hk) * slot_bytes + (long)uu * cpu, cpu);
        }
        __pipeline_commit();
    };

    const int nchunks = (j1 > j0) ? (j1 - j0 + KV_CHUNK - 1) / KV_CHUNK : 0;
    if (nchunks > 0) prefetch(j0, min(KV_CHUNK, j1 - j0), rawbuf);
    for (int ci = 0; ci < nchunks; ++ci) {
        const int cs = j0 + ci * KV_CHUNK;
        const int nC = min(KV_CHUNK, j1 - cs);
        const uint8_t* rb = rawbuf + (long)(ci & 1) * BUFSZ;
        const bool hasNext = (ci + 1 < nchunks);
        if (hasNext) prefetch(cs + KV_CHUNK, min(KV_CHUNK, j1 - cs - KV_CHUNK),
                              rawbuf + (long)((ci + 1) & 1) * BUFSZ);
        __pipeline_wait_prior(hasNext ? 1 : 0);
        __syncthreads();
        for (int kk = warpId; kk < nC; kk += nWarps) {
            const long sbase = (long)kk * slot_bytes;
            float part[G];
            #pragma unroll
            for (int g = 0; g < G; ++g) part[g] = 0.0f;
            if (qhoist) {
                #pragma unroll
                for (int di = 0; di < MAXNB; ++di) {
                    const int d = di * 32 + lane;
                    if (d >= D) continue;
                    const int blk = d / MXB, kd = d % MXB;
                    const long kblk = sbase + (long)blk * blkstride;
                    const float ksc = ssnf::e8m0_to_scale((int8_t)rb[kblk]);
                    const float kval = ksc * mxfp6_elem(rb + kblk + 1, kd);
                    #pragma unroll
                    for (int g = 0; g < G; ++g) part[g] += qreg[di * G + g] * kval;
                }
            } else {
                for (int d = lane; d < D; d += 32) {
                    const int blk = d / MXB, kd = d % MXB;
                    const long kblk = sbase + (long)blk * blkstride;
                    const float ksc = ssnf::e8m0_to_scale((int8_t)rb[kblk]);
                    const float kval = ksc * mxfp6_elem(rb + kblk + 1, kd);
                    #pragma unroll
                    for (int g = 0; g < G; ++g) part[g] += q_sh[g * D + d] * kval;
                }
            }
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                float r = warp_reduce_sum(part[g]);
                if (lane == 0) sc[g * KV_CHUNK + kk] = r * sm_scale;
            }
        }
        __syncthreads();
        float m_new[G], alpha[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * KV_CHUNK + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            alpha[g] = __expf(m_i[g] - m_new[g]);
        }
        __syncthreads();
        for (int i = tid; i < G * nC; i += blockDim.x) {
            const int g = i / nC, kk = i % nC;
            sc[g * KV_CHUNK + kk] = __expf(sc[g * KV_CHUNK + kk] - m_new[g]);
        }
        __syncthreads();
        float a[G], lsum[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) { a[g] = 0.0f; lsum[g] = 0.0f; }
        for (int kk = 0; kk < nC; ++kk) {
            const long vblk = (long)kk * slot_bytes + per_head + (long)dblk * blkstride;
            const float vval = ssnf::e8m0_to_scale((int8_t)rb[vblk]) * mxfp6_elem(rb + vblk + 1, dkd);
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = sc[g * KV_CHUNK + kk];
                lsum[g] += p; a[g] += p * vval;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            l_i[g] = l_i[g] * alpha[g] + lsum[g];
            acc[g] = acc[g] * alpha[g] + a[g];
            m_i[g] = m_new[g];
        }
        __syncthreads();
    }
    if (S == 1) {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            out[((long)b * H + (h0 + g)) * D + tid] = __float2bfloat16(l_i[g] > 0.0f ? acc[g] / l_i[g] : 0.0f);
    } else {
        #pragma unroll
        for (int g = 0; g < G; ++g)
            part_o[(((long)b * H + (h0 + g)) * S + s) * D + tid] = acc[g];
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                part_m[((long)b * H + (h0 + g)) * S + s] = m_i[g];
                part_l[((long)b * H + (h0 + g)) * S + s] = l_i[g];
            }
        }
    }
}

torch::Tensor mxfp6_kv_decode_paged_cuda(
        torch::Tensor q, torch::Tensor cache, torch::Tensor block_table,
        torch::Tensor seq_lens, int64_t B, int64_t H, int64_t Hkv, int64_t D,
        int64_t NB, int64_t block_size, int64_t slot_bytes, int64_t per_head) {
    auto out = torch::empty({B, H, D}, torch::dtype(torch::kBFloat16).device(q.device()));
    const int max_blocks = block_table.size(1);
    const float sm_scale = 1.0f / sqrtf((float)D);
    const int G = H / Hkv;
    TORCH_CHECK(H == G * Hkv, "mxfp6_kv_decode_paged: H must be a multiple of Hkv");
    const int threads = D;
    const size_t shmem = (size_t)(G * D + G * KV_CHUNK) * sizeof(float);
    const int L_hint = max_blocks * (int)block_size;
    int S = 1;
    {
        int mult = 8;
        if (const char* e = getenv("SSNF_KV_SPLIT_MULT")) { int m = atoi(e); if (m > 0) mult = m; }
        static int nsm = -1;
        if (nsm < 0) cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
        const int base = (int)(B * Hkv);
        S = ((long)mult * nsm + base - 1) / base;
        const int maxS = (L_hint + 31) / 32;
        if (S > maxS) S = maxS;
        if (S < 1) S = 1;
    }
    const dim3 grid(B, Hkv, S);
    torch::Tensor part_o, part_m, part_l;
    float *po = nullptr, *pm = nullptr, *pl = nullptr;
    if (S > 1) {
        auto fopt = torch::dtype(torch::kFloat32).device(q.device());
        part_o = torch::empty({B, H, S, D}, fopt);
        part_m = torch::empty({B, H, S}, fopt);
        part_l = torch::empty({B, H, S}, fopt);
        po = (float*)part_o.data_ptr(); pm = (float*)part_m.data_ptr(); pl = (float*)part_l.data_ptr();
    }
    const size_t cpa_shmem = shmem + (size_t)2 * KV_CHUNK * slot_bytes;
    bool use_cpa = (slot_bytes % 4 == 0);
    if (const char* e = getenv("SSNF_KV_CPA")) use_cpa = use_cpa && (atoi(e) != 0);
    // TC-QK port (, opt-in SSNF_MX6_TCQK=1): staging pays off only if the unpack is expensive;
    // for MXFP6 (6-bit E2M3) it might, unlike MXINT8. Same shared layout as the SSNF twin but with bf16
    // staging buffers, so the budget is base + rawbuf + kbf(2B) + vbf(2B) + q tile.
    const int CH6 = KV_CHUNK + 4;
    const size_t tcqk6_shmem = cpa_shmem
        + (size_t)NB * KV_CHUNK * sizeof(float)              // vsc_sh
        + (size_t)NB * MXB * CH6 * sizeof(__nv_bfloat16)     // vbf
        + (size_t)KV_CHUNK * (D + 4) * sizeof(__nv_bfloat16) // kbf
        + (size_t)NB * KV_CHUNK * sizeof(float)              // ksc_sh
        + (size_t)16 * 136 * 2;                              // q bf16 tile
    bool mx6_tcqk = false;
    if (const char* e = getenv("SSNF_MX6_TCQK"))
        mx6_tcqk = (atoi(e) != 0) && use_cpa && D == 128 && G == 4;
    auto launch = [&](auto Gc) {
        constexpr int Gv = decltype(Gc)::value;
        if (mx6_tcqk)
            mxfp6_kv_decode_paged_tcqk_cpa_kernel<Gv><<<grid, threads, tcqk6_shmem>>>(
                (const __nv_bfloat16*)q.data_ptr(), (const uint8_t*)cache.data_ptr(),
                (const int*)block_table.data_ptr(), (const int*)seq_lens.data_ptr(),
                (__nv_bfloat16*)out.data_ptr(), po, pm, pl,
                B, H, Hkv, D, NB, block_size, slot_bytes, max_blocks, per_head, sm_scale, S);
        else if (use_cpa)
            mxfp6_kv_decode_paged_cpa_kernel<Gv><<<grid, threads, cpa_shmem>>>(
                (const __nv_bfloat16*)q.data_ptr(), (const uint8_t*)cache.data_ptr(),
                (const int*)block_table.data_ptr(), (const int*)seq_lens.data_ptr(),
                (__nv_bfloat16*)out.data_ptr(), po, pm, pl,
                B, H, Hkv, D, NB, block_size, slot_bytes, max_blocks, per_head, sm_scale, S);
        else
            mxfp6_kv_decode_paged_kernel<Gv><<<grid, threads, shmem>>>(
                (const __nv_bfloat16*)q.data_ptr(), (const uint8_t*)cache.data_ptr(),
                (const int*)block_table.data_ptr(), (const int*)seq_lens.data_ptr(),
                (__nv_bfloat16*)out.data_ptr(), po, pm, pl,
                B, H, Hkv, D, NB, block_size, slot_bytes, max_blocks, per_head, sm_scale, S);
    };
    switch (G) {
        case 1: launch(std::integral_constant<int,1>{}); break;
        case 2: launch(std::integral_constant<int,2>{}); break;
        case 4: launch(std::integral_constant<int,4>{}); break;
        case 8: launch(std::integral_constant<int,8>{}); break;
        default: TORCH_CHECK(false, "mxfp6_kv_decode_paged: unsupported GQA group G=", G);
    }
    if (S > 1)
        kv_paged_combine_kernel<<<dim3(H, B), threads>>>(po, pm, pl,
            (__nv_bfloat16*)out.data_ptr(), H, D, S);
    return out;
}
