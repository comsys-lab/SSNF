// kernel/wa_gemm.cu  —  W+A GEMM  +  W-only prefill GEMM
//
// Design intent: route these two GEMM scopes to CUTLASS for tensor-core throughput.
//   * W+A GEMM    : weight unpacked to INT8, activation quantized to MXINT8 on
//                   the fly, INT8 IMMA mainloop, (scale_w*scale_a) epilogue.
//   * Prefill GEMM: weight unpacked + dequantized to BF16, BF16 tensor-core
//                   mainloop with FP32 accumulation.
//
// STATUS: GPU-UNVALIDATED CORRECTNESS BASELINES (plain CUDA, NOT yet CUTLASS).
// Like kv_attention, these reuse the certified ssnf::unpack_ssnf_weight_elem
// and match the oracle (ssnf_lib.reference.wa_matmul / wonly_matmul), so the
// pytest gates can certify the *math + wiring* end-to-end today. They are NOT
// the optimized path — one thread per output element, FP32 accumulate, no
// tensor cores.
//
// A CUTLASS mixed-input mainloop is a possible alternative implementation; the
// kernels below are the ones the reported numbers use.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <cuda_pipeline.h>
#include <math.h>
#include "core/ssnf_utils.cuh"

using namespace nvcuda;                    // wmma fragments (tensor-core prefill GEMM)
#include <ATen/cuda/CUDAContext.h>

namespace {

constexpr int BLOCK = 32;
constexpr int WSK = BLOCK + 8;             // shared K width (+pad to dodge bank conflicts)

// MXFP8-E4M3 element decode (hw fp8 cvt); used by the mxfp8 weight-GEMM twins below (same
// bf16-dequant vehicle as MXINT8, only the element codec int8 -> e4m3 differs).
__device__ __forceinline__ float e4m3_to_f(uint8_t b) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3)));
}

// MXFP6-E2M3 element decode; mirror of kernel/mxfp.cu::e2m3_val. mxfp6_elem: read element kd from a
// 24-byte packed block (4 codes per 3-byte LE word). Used by the mxfp6 weight-GEMM twins.
__device__ __forceinline__ float e2m3_val_w(int code) {
    const int e = (code >> 3) & 3, m = code & 7;
    const int mant = m + ((e != 0) << 3);
    const float v = (float)mant * exp2f((float)(max(e, 1) - 4));
    return __int_as_float(__float_as_int(v) ^ ((code & 0x20) << 26));
}

__device__ __forceinline__ float mxfp6_elem_w(const uint8_t* code24, int kd) {
    const int g = kd >> 2, w = kd & 3;
    const uint8_t* p = code24 + g * 3;
    const unsigned word = (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16);
    return e2m3_val_w((int)((word >> (6 * w)) & 0x3Fu));
}


// ---- W-only prefill WMMA, SOFTWARE-PIPELINED (cp.async-style overlap) --------
//   The plain WMMA stage is stage(unpack ALL) -> sync -> matmul -> sync, so the
//   per-element unpack ALU and the tensor-core MMA run SERIALLY -> staging-bound
//   (P22: the fast matmul exposes the u2/u3 unpack). Double-buffer As/Bs and, in
//   each K-step, issue the CURRENT tile's MMA first, then unpack the NEXT tile
//   into the other buffer: the unpack (ALU + its global byte loads) overlaps the
//   MMA (tensor-core pipe) -> the unpack hides behind the matmul (the lever P22
//   flagged). Matched mxint8_gemm_wmma_pipe has the same shape (int8 stage), so
//   the comparison stays fair. Opt-in via SSNF_TILE_CFG=11.
// Stream-dequant one weight column to a bf16 row (for the pipelined WMMA stage):
// gather the column's UB upper + SB shared bytes once, rolling-bit-buffer emit 32
// codes (shared advanced per group). Cheaper ALU than per-element straddle; uses
// 64 of 128 threads, but here the stage OVERLAPS the MMA, so its lower width is
// hidden — the win is the smaller total work. Same dense bytes -> bit-exact.
template<bool CM>
__device__ __forceinline__ void dequant_col_stream_bf16(
        const uint8_t* __restrict__ upper, const uint8_t* __restrict__ shared,
        int blk, int o, int OUT, int l_b, int g_size, int UB, int SB,
        float scale, __nv_bfloat16* dst) {
    const int wbits = 8 - l_b;
    const uint32_t umask = (1u << wbits) - 1u, usign = 1u << (wbits - 1);
    const uint32_t smask = (1u << l_b) - 1u, ssign = 1u << (l_b - 1);
    const int gsmask = g_size - 1;
    uint32_t ureg[6] = {0u,0u,0u,0u,0u,0u};
    uint32_t sreg[3] = {0u,0u,0u};
    if constexpr (CM) {
        // column-major [nb,OUT,UB]: the column's UB bytes are CONTIGUOUS -> wide-load
        // (adjacent columns UB apart -> the warp coalesces), vs the row-major *OUT stride.
        const long base = (long)blk * OUT + o;
        const uint32_t* src = reinterpret_cast<const uint32_t*>(upper + base * UB);
        #pragma unroll
        for (int i = 0; i < 6; ++i) if (i < (UB >> 2)) ureg[i] = src[i];
        #pragma unroll
        for (int bi = 0; bi < 8; ++bi)
            if (bi < SB) sreg[bi >> 2] |= (uint32_t)shared[base * SB + bi] << (8 * (bi & 3));
    } else {
        #pragma unroll
        for (int bi = 0; bi < 24; ++bi)
            if (bi < UB) ureg[bi >> 2] |= (uint32_t)upper[((long)blk*UB + bi)*OUT + o] << (8*(bi&3));
        #pragma unroll
        for (int bi = 0; bi < 8; ++bi)
            if (bi < SB) sreg[bi >> 2] |= (uint32_t)shared[((long)blk*SB + bi)*OUT + o] << (8*(bi&3));
    }
    uint64_t ubuf = 0; int unb = 0, uwi = 0;
    uint64_t sbuf = 0; int snb = 0, swi = 0; int l_g_code = 0;
    #pragma unroll
    for (int k = 0; k < BLOCK; ++k) {
        if (unb < wbits) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
        const int h_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
        ubuf >>= wbits; unb -= wbits;
        if ((k & gsmask) == 0) {
            if (snb < l_b) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
            l_g_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign;
            sbuf >>= l_b; snb -= l_b;
        }
        dst[k] = __float2bfloat16((float)(h_code * (1 << l_b) + l_g_code) * scale);
    }
}


// MXFP8-SSNF (E3M4) column dequant: SAME byte gather as dequant_col_stream_bf16, but each
// upper field decodes as an FP8 E3M4 element (sign/exp/mantissa + ldexpf). Only the
// per-element reconstruction differs -> isolates the fused-GEMM ALU cost of the format.
template<bool CM>
__device__ __forceinline__ void dequant_col_stream_fp8(
        const uint8_t* __restrict__ upper, const uint8_t* __restrict__ shared,
        int blk, int o, int OUT, int l_b, int g_size, int UB, int SB,
        float scale, __nv_bfloat16* dst) {
    constexpr int EB = 3, MB = 4, BIAS = 3, EMIN = -2;
    const int wbits = 8 - l_b, mbup = MB - l_b;
    const uint32_t umask = (1u << wbits) - 1u;
    const uint32_t expmask = (1u << EB) - 1u, upmask = (1u << mbup) - 1u, lead = 1u << mbup;
    const uint32_t smask = (1u << l_b) - 1u, ssign = 1u << (l_b - 1);
    const int gsmask = g_size - 1;
    uint32_t ureg[6] = {0u,0u,0u,0u,0u,0u};
    uint32_t sreg[3] = {0u,0u,0u};
    if constexpr (CM) {
        const long base = (long)blk * OUT + o;
        const uint32_t* src = reinterpret_cast<const uint32_t*>(upper + base * UB);
        #pragma unroll
        for (int i = 0; i < 6; ++i) if (i < (UB >> 2)) ureg[i] = src[i];
        #pragma unroll
        for (int bi = 0; bi < 8; ++bi)
            if (bi < SB) sreg[bi >> 2] |= (uint32_t)shared[base * SB + bi] << (8 * (bi & 3));
    } else {
        #pragma unroll
        for (int bi = 0; bi < 24; ++bi)
            if (bi < UB) ureg[bi >> 2] |= (uint32_t)upper[((long)blk*UB + bi)*OUT + o] << (8*(bi&3));
        #pragma unroll
        for (int bi = 0; bi < 8; ++bi)
            if (bi < SB) sreg[bi >> 2] |= (uint32_t)shared[((long)blk*SB + bi)*OUT + o] << (8*(bi&3));
    }
    uint64_t ubuf = 0; int unb = 0, uwi = 0;
    uint64_t sbuf = 0; int snb = 0, swi = 0; int l_g_code = 0;
    #pragma unroll
    for (int k = 0; k < BLOCK; ++k) {
        if (unb < wbits) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
        const uint32_t field = (uint32_t)ubuf & umask;
        ubuf >>= wbits; unb -= wbits;
        const uint32_t sgn = field >> (wbits - 1);
        const uint32_t expf = (field >> mbup) & expmask;
        const int m_up = (int)((expf ? lead : 0u) | (field & upmask));
        const int ee = expf ? (int)expf - BIAS : EMIN;
        if ((k & gsmask) == 0) {
            if (snb < l_b) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
            l_g_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign;
            sbuf >>= l_b; snb -= l_b;
        }
        int mag = m_up << l_b;
        if (sgn) mag = -mag;
        dst[k] = __float2bfloat16(ldexpf((float)(mag + l_g_code), ee - MB) * scale);
    }
}


template<bool CM, bool FP8 = false>
__global__ void wonly_gemm_wmma_pipe(
        const __nv_bfloat16* __restrict__ X, const int8_t* __restrict__ scale_exp,
        const uint8_t* __restrict__ upper, const uint8_t* __restrict__ shared,
        __nv_bfloat16* __restrict__ Y,
        int M, int OUT, int K, int NB, int l_b, int g_size, int UB, int SB) {
    __shared__ __nv_bfloat16 As[2][64][WSK];
    __shared__ __nv_bfloat16 Bs[2][64][WSK];
    __shared__ float tmp[4][16][16];
    const int m0 = blockIdx.y * 64, o0 = blockIdx.x * 64;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wM = warp >> 1, wN = warp & 1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    // stage block `b` (As=X, Bs=dequant W) into shared buffer `buf`
    auto stage = [&](int b, int buf) {
        for (int idx = tid; idx < 64 * BLOCK; idx += 128) {
            const int m = idx / BLOCK, k = idx % BLOCK;
            As[buf][m][k] = (m0 + m < M) ? X[(long)(m0 + m) * K + b * BLOCK + k]
                                         : __float2bfloat16(0.0f);
        }
        for (int oc = tid; oc < 64; oc += 128) {     // column-streaming unpack (hidden by MMA)
            const int o = o0 + oc;
            if (o < OUT) {
                const float sc = ssnf::e8m0_to_scale(scale_exp[b * OUT + o]);
                if constexpr (FP8)
                    dequant_col_stream_fp8<CM>(upper, shared, b, o, OUT, l_b, g_size, UB, SB, sc, &Bs[buf][oc][0]);
                else
                    dequant_col_stream_bf16<CM>(upper, shared, b, o, OUT, l_b, g_size, UB, SB, sc, &Bs[buf][oc][0]);
            } else {
                #pragma unroll
                for (int k = 0; k < BLOCK; ++k) Bs[buf][oc][k] = __float2bfloat16(0.0f);
            }
        }
    };

    stage(0, 0);
    __syncthreads();
    for (int blk = 0; blk < NB; ++blk) {
        const int cur = blk & 1, nxt = cur ^ 1;
        #pragma unroll
        for (int wk = 0; wk < BLOCK / 16; ++wk) {        // CURRENT tile MMA (tensor core)
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &As[cur][wM*32 + i*16][wk*16], WSK);
            #pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &Bs[cur][wN*32 + j*16][wk*16], WSK);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        if (blk + 1 < NB) stage(blk + 1, nxt);           // NEXT tile unpack — overlaps the MMA
        __syncthreads();
    }
    float* wt = &tmp[warp][0][0];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(wt, acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            const int mb = m0 + wM*32 + i*16, ob = o0 + wN*32 + j*16;
            for (int e = lane; e < 256; e += 32) {
                const int m = mb + e / 16, o = ob + e % 16;
                if (m < M && o < OUT) Y[(long)m * OUT + o] = __float2bfloat16(wt[e]);
            }
            __syncwarp();
        }
}


// Tile-config dispatch. Swept optimum (OUT=4096) is M-ADAPTIVE: a
// 128x128 tile reuses best but yields too few blocks (OUT/128=32 < the SM count) until
// M fills the second grid dim, so small M prefers 64x64 (more blocks). Crossover
// ~M=256. SSNF_TILE_CFG (env) forces a specific config (for sweeps).
//   cfg 1 = 64x64 r4x4 ;  cfg 5 = 128x128 r8x8
static inline int tile_cfg(int M) {
    if (const char* e = getenv("SSNF_TILE_CFG")) { int c = atoi(e); if (c >= 0) return c; }
    return (M >= 256) ? 5 : 1;
}

// ===== FUSED SKINNY-M quantized tensor-core GEMM (B>=16 decode) =====
//   Y[M,OUT] = X[M,K] @ dequant(W)^T, M small (16-32), memory-bound on the 11MB
//   quantized weight. Reads quantized W (NOT the 34MB bf16), dequants per K-block in
//   shared, mma.sync. MT=16 (skinny) x NT=64, 4 warps (1 N-subtile each). grid
//   (OUT/64, ceil(M/16), splitK) for occupancy; partials summed by gemm_skinny_combine.
//   Current stage: production unpack_ssnf_weight_elem (correctness first; bfe/cp.async later).
//   MT = ceil(M/16) (1 or 2): ONE block handles ALL M rows (weight read once/splitK).
//   Fast dequant via stream_block_dense_bfe (column-major planes, single bfe/code).
template<int LB_, int GSIZE_, int MT, bool RA = false, int NT = 64>
__global__ void wonly_gemm_fused_skinny(
        const __nv_bfloat16* __restrict__ X, const int8_t* __restrict__ scale_exp,
        const uint8_t* __restrict__ upper_cm, const uint8_t* __restrict__ shared_cm,
        float* __restrict__ partial, int M, int OUT, int K, int NB, int splitK) {
    constexpr int NS = NT / 64;                      // N-subtiles per warp (NT=64->1, 128->2)
    __shared__ __nv_bfloat16 As[16 * MT][WSK];
    __shared__ __nv_bfloat16 Bs[NT][WSK];
    __shared__ float tmp[4][16][16];
    const int o0 = blockIdx.x * NT, sp = blockIdx.y;
    const int per = (NB + splitK - 1) / splitK, b0 = sp * per, b1 = min(b0 + per, NB);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MT][NS];
    #pragma unroll
    for (int mi = 0; mi < MT; ++mi)
        #pragma unroll
        for (int ns = 0; ns < NS; ++ns) wmma::fill_fragment(acc[mi][ns], 0.0f);
    for (int blk = b0; blk < b1; ++blk) {
        for (int idx = tid; idx < 16 * MT * BLOCK; idx += 128) {
            const int m = idx / BLOCK, k = idx % BLOCK;
            As[m][k] = (m < M) ? X[(long)m * K + blk * BLOCK + k] : __float2bfloat16(0.0f);
        }
        if (tid < NT) {                              // NT threads dequant NT columns (full util at NT=128)
            const int o = o0 + tid;
            if (o < OUT) {
                const float sc = ssnf::e8m0_to_scale(scale_exp[blk * OUT + o]);
                auto emit = [&](int k, int word) { Bs[tid][k] = __float2bfloat16((float)word * sc); };
                if constexpr (RA) ssnf::stream_block_ra<LB_, GSIZE_>(upper_cm, shared_cm, (long)blk * OUT + o, emit);
                else              ssnf::stream_block_dense_bfe<LB_, GSIZE_>(upper_cm, shared_cm, (long)blk * OUT + o, emit);
            } else {
                #pragma unroll
                for (int k = 0; k < BLOCK; ++k) Bs[tid][k] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int wk = 0; wk < BLOCK / 16; ++wk) {
            #pragma unroll
            for (int ns = 0; ns < NS; ++ns) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
                wmma::load_matrix_sync(b, &Bs[(warp * NS + ns) * 16][wk * 16], WSK);
                #pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
                    wmma::load_matrix_sync(a, &As[mi * 16][wk * 16], WSK);
                    wmma::mma_sync(acc[mi][ns], a, b, acc[mi][ns]);
                }
            }
        }
        __syncthreads();
    }
    float* wt = &tmp[warp][0][0];
    #pragma unroll
    for (int ns = 0; ns < NS; ++ns) {
        const int ob = o0 + (warp * NS + ns) * 16;
        #pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            wmma::store_matrix_sync(wt, acc[mi][ns], 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int m = mi * 16 + e / 16, o = ob + e % 16;
                if (m < M && o < OUT) partial[((long)sp * M + m) * OUT + o] = wt[e];
            }
            __syncwarp();
        }
    }
}


// FAIR MXINT8 twin of the fused skinny GEMM. Same vehicle (read quantized weight
// direct, dequant per K-block to bf16 in shared, WMMA), so the SSNF-vs-MXINT8 decode
// comparison differs ONLY in the weight format: MXINT8 reads 17MB int8 (qweight_cm,
// 32 contiguous int8 per (blk,o)) vs SSNF's 11MB sub-byte planes. No sub-byte unpack
// here (byte-aligned) -> if anything MXINT8 has the cheaper dequant; the only thing
// left to separate them is bytes-moved = the mantissa-sharing advantage. Adaptive NT.
template<int MT, int NT = 64>
__global__ void mxint8_gemm_fused_skinny(
        const __nv_bfloat16* __restrict__ X, const int8_t* __restrict__ scale_exp,
        const int8_t* __restrict__ qweight_cm,
        float* __restrict__ partial, int M, int OUT, int K, int NB, int splitK) {
    constexpr int NS = NT / 64;
    __shared__ __nv_bfloat16 As[16 * MT][WSK];
    __shared__ __nv_bfloat16 Bs[NT][WSK];
    __shared__ float tmp[4][16][16];
    const int o0 = blockIdx.x * NT, sp = blockIdx.y;
    const int per = (NB + splitK - 1) / splitK, b0 = sp * per, b1 = min(b0 + per, NB);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MT][NS];
    #pragma unroll
    for (int mi = 0; mi < MT; ++mi)
        #pragma unroll
        for (int ns = 0; ns < NS; ++ns) wmma::fill_fragment(acc[mi][ns], 0.0f);
    for (int blk = b0; blk < b1; ++blk) {
        for (int idx = tid; idx < 16 * MT * BLOCK; idx += 128) {
            const int m = idx / BLOCK, k = idx % BLOCK;
            As[m][k] = (m < M) ? X[(long)m * K + blk * BLOCK + k] : __float2bfloat16(0.0f);
        }
        if (tid < NT) {
            const int o = o0 + tid;
            if (o < OUT) {
                const float sc = ssnf::e8m0_to_scale(scale_exp[blk * OUT + o]);
                const int8_t* col = qweight_cm + ((long)blk * OUT + o) * BLOCK;   // 32 contiguous int8
                #pragma unroll
                for (int k = 0; k < BLOCK; ++k) Bs[tid][k] = __float2bfloat16((float)col[k] * sc);
            } else {
                #pragma unroll
                for (int k = 0; k < BLOCK; ++k) Bs[tid][k] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int wk = 0; wk < BLOCK / 16; ++wk) {
            #pragma unroll
            for (int ns = 0; ns < NS; ++ns) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
                wmma::load_matrix_sync(b, &Bs[(warp * NS + ns) * 16][wk * 16], WSK);
                #pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
                    wmma::load_matrix_sync(a, &As[mi * 16][wk * 16], WSK);
                    wmma::mma_sync(acc[mi][ns], a, b, acc[mi][ns]);
                }
            }
        }
        __syncthreads();
    }
    float* wt = &tmp[warp][0][0];
    #pragma unroll
    for (int ns = 0; ns < NS; ++ns) {
        const int ob = o0 + (warp * NS + ns) * 16;
        #pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            wmma::store_matrix_sync(wt, acc[mi][ns], 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int m = mi * 16 + e / 16, o = ob + e % 16;
                if (m < M && o < OUT) partial[((long)sp * M + m) * OUT + o] = wt[e];
            }
            __syncwarp();
        }
    }
}


// MXFP8-E4M3 skinny fused TC-GEMM — byte-for-byte clone of mxint8_gemm_fused_skinny with
// the element read (float)int8 -> e4m3_to_f (hw fp8 cvt). Dequant to bf16 into shared,
// then bf16 wmma (the fair W-only vehicle; e4m3 element does NOT ride fp8 tensor cores here).
template<int MT, int NT>
__global__ void mxfp8_gemm_fused_skinny(
        const __nv_bfloat16* __restrict__ X, const int8_t* __restrict__ scale_exp,
        const int8_t* __restrict__ qweight_cm,
        float* __restrict__ partial, int M, int OUT, int K, int NB, int splitK) {
    constexpr int NS = NT / 64;
    __shared__ __nv_bfloat16 As[16 * MT][WSK];
    __shared__ __nv_bfloat16 Bs[NT][WSK];
    __shared__ float tmp[4][16][16];
    const int o0 = blockIdx.x * NT, sp = blockIdx.y;
    const int per = (NB + splitK - 1) / splitK, b0 = sp * per, b1 = min(b0 + per, NB);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MT][NS];
    #pragma unroll
    for (int mi = 0; mi < MT; ++mi)
        #pragma unroll
        for (int ns = 0; ns < NS; ++ns) wmma::fill_fragment(acc[mi][ns], 0.0f);
    for (int blk = b0; blk < b1; ++blk) {
        for (int idx = tid; idx < 16 * MT * BLOCK; idx += 128) {
            const int m = idx / BLOCK, k = idx % BLOCK;
            As[m][k] = (m < M) ? X[(long)m * K + blk * BLOCK + k] : __float2bfloat16(0.0f);
        }
        if (tid < NT) {
            const int o = o0 + tid;
            if (o < OUT) {
                const float sc = ssnf::e8m0_to_scale(scale_exp[blk * OUT + o]);
                const int8_t* col = qweight_cm + ((long)blk * OUT + o) * BLOCK;   // 32 e4m3 bytes
                #pragma unroll
                for (int k = 0; k < BLOCK; ++k) Bs[tid][k] = __float2bfloat16(e4m3_to_f((uint8_t)col[k]) * sc);
            } else {
                #pragma unroll
                for (int k = 0; k < BLOCK; ++k) Bs[tid][k] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int wk = 0; wk < BLOCK / 16; ++wk) {
            #pragma unroll
            for (int ns = 0; ns < NS; ++ns) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
                wmma::load_matrix_sync(b, &Bs[(warp * NS + ns) * 16][wk * 16], WSK);
                #pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
                    wmma::load_matrix_sync(a, &As[mi * 16][wk * 16], WSK);
                    wmma::mma_sync(acc[mi][ns], a, b, acc[mi][ns]);
                }
            }
        }
        __syncthreads();
    }
    float* wt = &tmp[warp][0][0];
    #pragma unroll
    for (int ns = 0; ns < NS; ++ns) {
        const int ob = o0 + (warp * NS + ns) * 16;
        #pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            wmma::store_matrix_sync(wt, acc[mi][ns], 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int m = mi * 16 + e / 16, o = ob + e % 16;
                if (m < M && o < OUT) partial[((long)sp * M + m) * OUT + o] = wt[e];
            }
            __syncwarp();
        }
    }
}


// MXFP8 large-M dequant -> bf16 [K,OUT] (matmul-ready), clone of mxint8_dequant_bf16_kernel.
__global__ void mxfp8_dequant_bf16_kernel(
        const int8_t* __restrict__ scale_exp, const int8_t* __restrict__ qweight_cm,
        __nv_bfloat16* __restrict__ Wbf, int OUT, int K, int NB) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int blk = blockIdx.y;
    if (o >= OUT) return;
    const float scale = ssnf::e8m0_to_scale(scale_exp[blk * OUT + o]);
    const int8_t* col = qweight_cm + ((long)blk * OUT + o) * BLOCK;
    #pragma unroll
    for (int k = 0; k < BLOCK; ++k)
        Wbf[((long)(blk * BLOCK + k)) * OUT + o] = __float2bfloat16(e4m3_to_f((uint8_t)col[k]) * scale);
}


// MXFP6-E2M3 skinny fused TC-GEMM — clone of mxfp8 with 6-bit unpack (qweight_cm stride = 24 B/block,
// element via mxfp6_elem_w). bf16-dequant into shared, then bf16 wmma (W-only, activation bf16).
template<int MT, int NT>
__global__ void mxfp6_gemm_fused_skinny(
        const __nv_bfloat16* __restrict__ X, const int8_t* __restrict__ scale_exp,
        const int8_t* __restrict__ qweight_cm,
        float* __restrict__ partial, int M, int OUT, int K, int NB, int splitK) {
    constexpr int NS = NT / 64;
    __shared__ __nv_bfloat16 As[16 * MT][WSK];
    __shared__ __nv_bfloat16 Bs[NT][WSK];
    __shared__ float tmp[4][16][16];
    const int o0 = blockIdx.x * NT, sp = blockIdx.y;
    const int per = (NB + splitK - 1) / splitK, b0 = sp * per, b1 = min(b0 + per, NB);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MT][NS];
    #pragma unroll
    for (int mi = 0; mi < MT; ++mi)
        #pragma unroll
        for (int ns = 0; ns < NS; ++ns) wmma::fill_fragment(acc[mi][ns], 0.0f);
    for (int blk = b0; blk < b1; ++blk) {
        for (int idx = tid; idx < 16 * MT * BLOCK; idx += 128) {
            const int m = idx / BLOCK, k = idx % BLOCK;
            As[m][k] = (m < M) ? X[(long)m * K + blk * BLOCK + k] : __float2bfloat16(0.0f);
        }
        if (tid < NT) {
            const int o = o0 + tid;
            if (o < OUT) {
                const float sc = ssnf::e8m0_to_scale(scale_exp[blk * OUT + o]);
                const uint8_t* col = (const uint8_t*)qweight_cm + ((long)blk * OUT + o) * 24;  // 24 packed bytes
                #pragma unroll
                for (int k = 0; k < BLOCK; ++k) Bs[tid][k] = __float2bfloat16(mxfp6_elem_w(col, k) * sc);
            } else {
                #pragma unroll
                for (int k = 0; k < BLOCK; ++k) Bs[tid][k] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int wk = 0; wk < BLOCK / 16; ++wk) {
            #pragma unroll
            for (int ns = 0; ns < NS; ++ns) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
                wmma::load_matrix_sync(b, &Bs[(warp * NS + ns) * 16][wk * 16], WSK);
                #pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
                    wmma::load_matrix_sync(a, &As[mi * 16][wk * 16], WSK);
                    wmma::mma_sync(acc[mi][ns], a, b, acc[mi][ns]);
                }
            }
        }
        __syncthreads();
    }
    float* wt = &tmp[warp][0][0];
    #pragma unroll
    for (int ns = 0; ns < NS; ++ns) {
        const int ob = o0 + (warp * NS + ns) * 16;
        #pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            wmma::store_matrix_sync(wt, acc[mi][ns], 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int m = mi * 16 + e / 16, o = ob + e % 16;
                if (m < M && o < OUT) partial[((long)sp * M + m) * OUT + o] = wt[e];
            }
            __syncwarp();
        }
    }
}


// MXFP6 large-M dequant -> bf16 [K,OUT] (matmul-ready), 6-bit unpack.
__global__ void mxfp6_dequant_bf16_kernel(
        const int8_t* __restrict__ scale_exp, const int8_t* __restrict__ qweight_cm,
        __nv_bfloat16* __restrict__ Wbf, int OUT, int K, int NB) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int blk = blockIdx.y;
    if (o >= OUT) return;
    const float scale = ssnf::e8m0_to_scale(scale_exp[blk * OUT + o]);
    const uint8_t* col = (const uint8_t*)qweight_cm + ((long)blk * OUT + o) * 24;
    #pragma unroll
    for (int k = 0; k < BLOCK; ++k)
        Wbf[((long)(blk * BLOCK + k)) * OUT + o] = __float2bfloat16(mxfp6_elem_w(col, k) * scale);
}


// cp.async PIPELINED variant (Marlin core): prefetch the next K-block's quantized
// weight (Wq/Ws) to shared via async copy, overlapping the current block's dequant+mma
// -> hides the weight load latency (ncu: long_scoreboard 59% + barrier 21%).
template<int LB_, int GSIZE_, int MT>
__global__ void wonly_gemm_fused_skinny_pipe(
        const __nv_bfloat16* __restrict__ X, const int8_t* __restrict__ scale_exp,
        const uint8_t* __restrict__ upper_cm, const uint8_t* __restrict__ shared_cm,
        float* __restrict__ partial, int M, int OUT, int K, int NB, int splitK) {
    constexpr int WBITS = 8 - LB_, UBc = BLOCK * WBITS / 8;
    constexpr int SBc = ((BLOCK / GSIZE_) * LB_ + 7) / 8;
    constexpr int WQB = 64 * UBc, WSB = 64 * SBc;
    __shared__ uint8_t Wq[2][WQB];
    __shared__ uint8_t Ws[2][((WSB + 15) / 16) * 16];
    __shared__ __nv_bfloat16 As[16 * MT][WSK];
    __shared__ __nv_bfloat16 Bs[64][WSK];
    __shared__ float tmp[4][16][16];
    const int o0 = blockIdx.x * 64, sp = blockIdx.y;
    const int per = (NB + splitK - 1) / splitK, b0 = sp * per, b1 = min(b0 + per, NB);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MT];
    #pragma unroll
    for (int mi = 0; mi < MT; ++mi) wmma::fill_fragment(acc[mi], 0.0f);
    auto prefetch = [&](int blk, int buf) {
        const uint8_t* us = upper_cm + (long)(blk * OUT + o0) * UBc;
        const uint8_t* ss = shared_cm + (long)(blk * OUT + o0) * SBc;
        // per-block weight (64 cols * UBc) is 16-byte aligned -> 16-byte cp.async (Marlin alignment lesson)
        for (int i = tid * 16; i < WQB; i += 128 * 16) __pipeline_memcpy_async(&Wq[buf][i], us + i, 16);
        for (int i = tid * 16; i < WSB; i += 128 * 16) __pipeline_memcpy_async(&Ws[buf][i], ss + i, 16);
        __pipeline_commit();
    };
    const int nblk = b1 - b0;
    if (nblk > 0) prefetch(b0, 0);
    for (int i = 0; i < nblk; ++i) {
        const int blk = b0 + i, buf = i & 1;
        if (i + 1 < nblk) prefetch(blk + 1, (i + 1) & 1);
        __pipeline_wait_prior(i + 1 < nblk ? 1 : 0);
        __syncthreads();
        for (int idx = tid; idx < 16 * MT * BLOCK; idx += 128) {
            const int m = idx / BLOCK, k = idx % BLOCK;
            As[m][k] = (m < M) ? X[(long)m * K + blk * BLOCK + k] : __float2bfloat16(0.0f);
        }
        if (tid < 64) {
            const int o = o0 + tid;
            const float sc = (o < OUT) ? ssnf::e8m0_to_scale(scale_exp[blk * OUT + o]) : 0.0f;
            ssnf::stream_block_dense_bfe<LB_, GSIZE_>(Wq[buf], Ws[buf], (long)tid,
                [&](int k, int word) { Bs[tid][k] = __float2bfloat16((float)word * sc); });
        }
        __syncthreads();
        #pragma unroll
        for (int wk = 0; wk < BLOCK / 16; ++wk) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::load_matrix_sync(b, &Bs[warp * 16][wk * 16], WSK);
            #pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
                wmma::load_matrix_sync(a, &As[mi * 16][wk * 16], WSK);
                wmma::mma_sync(acc[mi], a, b, acc[mi]);
            }
        }
        __syncthreads();
    }
    const int ob = o0 + warp * 16;
    float* wt = &tmp[warp][0][0];
    #pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        wmma::store_matrix_sync(wt, acc[mi], 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            const int m = mi * 16 + e / 16, o = ob + e % 16;
            if (m < M && o < OUT) partial[((long)sp * M + m) * OUT + o] = wt[e];
        }
        __syncwarp();
    }
}


__global__ void gemm_skinny_combine(const float* __restrict__ partial,
        __nv_bfloat16* __restrict__ Y, int M, int OUT, int splitK) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)M * OUT) return;
    float s = 0.0f;
    for (int sp = 0; sp < splitK; ++sp) s += partial[(long)sp * M * OUT + idx];
    Y[idx] = __float2bfloat16(s);
}


} // namespace

// ---- host launchers (signatures match ssnf_lib.ops / the pybind schema) -------
static inline void gemm_dims(int64_t l_b, int64_t g_size, int& UB, int& SB) {
    const int wbits = 8 - (int)l_b;
    UB = BLOCK * wbits / 8;
    SB = ((BLOCK / (int)g_size) * (int)l_b + 7) / 8;
}

// Defined below; forward-declared so the small-M dispatch in wonly_gemm_tc_cuda
// can route to it (the 16-row skinny tile + auto split-K fills the GPU at small M).
torch::Tensor wonly_gemm_fused_skinny_cuda(
        torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
        int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t l_b, int64_t g_size);


// TENSOR-CORE W-only prefill GEMM: BF16 WMMA (software-pipelined) with the weight
// dequant wide-loading the COLUMN-MAJOR planes (coalesced, overlaps the MMA). Combines
// tensor cores + coalesced unpack -> the fastest W-only prefill path. bf16 operands
// (fp32 accumulate); slight precision vs the fp32 scalar path, inherent to WMMA.
// --- Dequant SSNF weight planes -> dense bf16 [OUT,K], so large-M (prefill) GEMM can
//   run on cuBLAS (torch.matmul) instead of the fused wmma_pipe (~30x slower per nsys;
//   it dominated E2E at 73%). Column-major planes: scale_exp[nb,OUT], upper_cm[nb,OUT,UB],
//   shared_cm[nb,OUT,SB]. Element (o,k): blk=k/32, kd=k%32, col=blk*OUT+o. Reconstruction
//   ((up<<l_b)+sh)*scale is bit-identical to dequant_col_stream_bf16 used by wmma_pipe. ----
// B: block-cooperative dequant. A block owns TILE consecutive columns (col=blk*OUT+o). It
// COALESCED-loads the tile's packed upper/shared bytes into shared (adjacent columns are
// contiguous in the dense col-major plane), then each (lc,kd) thread bfe-extracts its element
// FROM SHARED (no global strided per-elem reads) and COALESCED-writes W. exp2i scale. BIT-EXACT
// to the per-element kernel (same dense planes / same up*2^l_b+sh / 2^int scale). u3/gs16 & u2/gs8.
template<int LB_, int GSIZE_, int TILE>
__global__ void wonly_dequant_bf16_bc_kernel(
        const int8_t* __restrict__ scale_exp, const uint8_t* __restrict__ upper_cm,
        const uint8_t* __restrict__ shared_cm, __nv_bfloat16* __restrict__ W, int OUT, int K) {
    constexpr int WBITS = 8 - LB_;
    constexpr int UBc = 32 * WBITS / 8;                 // upper bytes/column
    constexpr int SBc = ((32 / GSIZE_) * LB_ + 7) / 8;      // shared bytes/column
    const int NB = K >> 5;
    const long ncol = (long)NB * OUT;
    const long c0 = (long)blockIdx.x * TILE;
    if (c0 >= ncol) return;
    const int ncols = (int)min((long)TILE, ncol - c0);
    __shared__ uint8_t up_s[TILE * UBc];
    __shared__ uint8_t sh_s[TILE * SBc];
    __shared__ float   sc_s[TILE];
    const int tid = threadIdx.x;                         // blockDim.x == TILE*32
    for (int i = tid; i < ncols * UBc; i += blockDim.x) up_s[i] = upper_cm[c0 * UBc + i];
    for (int i = tid; i < ncols * SBc; i += blockDim.x) sh_s[i] = shared_cm[c0 * SBc + i];
    if (tid < ncols) sc_s[tid] = ssnf::exp2i(scale_exp[c0 + tid]);
    __syncthreads();
    const int lc = tid >> 5, kd = tid & 31;
    if (lc < ncols) {
        const long col = c0 + lc;
        const int blk = (int)(col / OUT), o = (int)(col % OUT);
        const int gb = kd * WBITS, bi = gb >> 3, bo = gb & 7;
        uint32_t cu = (uint32_t)up_s[lc * UBc + bi] >> bo;
        if (bo + WBITS > 8 && bi + 1 < UBc) cu |= (uint32_t)up_s[lc * UBc + bi + 1] << (8 - bo);
        const int up = ssnf::sign_extend(cu & ((1u << WBITS) - 1u), WBITS);
        const int g = kd / GSIZE_;
        const int sgb = g * LB_, sbi = sgb >> 3, sbo = sgb & 7;
        uint32_t cs = (uint32_t)sh_s[lc * SBc + sbi] >> sbo;
        if (sbo + LB_ > 8 && sbi + 1 < SBc) cs |= (uint32_t)sh_s[lc * SBc + sbi + 1] << (8 - sbo);
        const int sh = ssnf::sign_extend(cs & ((1u << LB_) - 1u), LB_);
        const int word = up * (1 << LB_) + sh;
        W[(long)o * K + (long)blk * 32 + kd] = __float2bfloat16((float)word * sc_s[lc]);
    }
}


__global__ void wonly_dequant_bf16_kernel(
        const int8_t* __restrict__ scale_exp, const uint8_t* __restrict__ upper_cm,
        const uint8_t* __restrict__ shared_cm, __nv_bfloat16* __restrict__ W,
        int OUT, int K, int l_b, int g_size, int UB, int SB, bool legacy) {
    const long total = (long)OUT * K;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x) {
        const int o = (int)(idx / K), k = (int)(idx % K);
        const int blk = k >> 5, kd = k & 31;                       // BLOCK == 32
        const long col = (long)blk * OUT + o;
        const float sc = legacy ? ssnf::e8m0_to_scale(scale_exp[col]) : ssnf::exp2i(scale_exp[col]);
        const int code = ssnf::unpack_ssnf_kv_elem_bfe(upper_cm, shared_cm, col * (long)UB,
                                                   col * (long)SB, 0, kd, l_b, g_size, UB, SB);
        W[(long)o * K + k] = __float2bfloat16((float)code * sc);
    }
}


torch::Tensor wonly_dequant_bf16_cuda(
        torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
        int64_t OUT, int64_t K, int64_t NB, int64_t l_b, int64_t g_size) {
    int UB, SB; gemm_dims(l_b, g_size, UB, SB);
    auto W = torch::empty({OUT, K}, torch::dtype(torch::kBFloat16).device(upper_cm.device()));
    const long total = (long)OUT * K;
    const int threads = 256;
    const bool legacy = (getenv("SSNF_DQ_LEGACY") && atoi(getenv("SSNF_DQ_LEGACY")) != 0);  // per-elem ref
    auto* sc = scale_exp.data_ptr<int8_t>(); auto* up = upper_cm.data_ptr<uint8_t>();
    auto* sh = shared_cm.data_ptr<uint8_t>();
    auto* w = reinterpret_cast<__nv_bfloat16*>(W.data_ptr<at::BFloat16>());
    auto st = at::cuda::getCurrentCUDAStream();
    // block-cooperative path — MEASURED SLOWER (0.81x) than the per-element kernel: the latter's
    // column byte-reads are L1-cached broadcasts (32 lanes/col = 1 fetch), so staging to shared only
    // adds sync; the cost is the irreducible sub-byte unpack ALU. Kept behind SSNF_DQ_BC for the record.
    const bool use_bc = (getenv("SSNF_DQ_BC") && atoi(getenv("SSNF_DQ_BC")) != 0);
    if (use_bc && ((l_b == 3 && g_size == 16) || (l_b == 2 && g_size == 8))) {
        constexpr int TILE = 8;
        const long ncol = total / 32;
        const int blocks = (int)((ncol + TILE - 1) / TILE);
        if (l_b == 3) wonly_dequant_bf16_bc_kernel<3, 16, TILE><<<blocks, TILE * 32, 0, st>>>(sc, up, sh, w, (int)OUT, (int)K);
        else        wonly_dequant_bf16_bc_kernel<2, 8, TILE><<<blocks, TILE * 32, 0, st>>>(sc, up, sh, w, (int)OUT, (int)K);
        return W;
    }
    const long nb_ = (total + threads - 1) / threads;
    const int blocks = (int)(nb_ < 65535 ? nb_ : 65535);
    wonly_dequant_bf16_kernel<<<blocks, threads, 0, st>>>(sc, up, sh, w, (int)OUT, (int)K, (int)l_b, (int)g_size, UB, SB, legacy);
    return W;
}


torch::Tensor wonly_gemm_tc_cuda(
        torch::Tensor X, torch::Tensor scale_exp,
        torch::Tensor upper_cm, torch::Tensor shared_cm,
        int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t l_b, int64_t g_size) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    // Small-M (<=64): the 64x64 WMMA tile wastes most of its rows and the grid is
    // too small to fill the GPU (~8-11x slower, bandwidth-starved). Route to the
    // skinny 16-row + split-K kernel. Only u3/gs16 & u2/gs8 are specialized; other
    // (l_b,g_size) fall through. Override with SSNF_NO_SKINNY=1.
    if (M <= 256 && ((l_b == 3 && g_size == 16) || (l_b == 2 && g_size == 8))
            && !(getenv("SSNF_NO_SKINNY") && atoi(getenv("SSNF_NO_SKINNY")) != 0)) {
        return wonly_gemm_fused_skinny_cuda(X, scale_exp, upper_cm, shared_cm, M, OUT, K, NB, l_b, g_size);
    }
    int UB, SB; gemm_dims(l_b, g_size, UB, SB);
    auto Y = torch::empty({M, OUT}, X.options());
    wonly_gemm_wmma_pipe<true><<<dim3(((int)OUT + 63) / 64, ((int)M + 63) / 64), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
        scale_exp.data_ptr<int8_t>(), upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, (int)K, (int)NB, (int)l_b, (int)g_size, UB, SB);
    return Y;
}


// FUSED SKINNY-M quantized tensor-core GEMM (B>=16 decode). Reads quantized W (row-major
// upper/shared fields), dequants in shared, mma.sync. splitK for occupancy + combine.
torch::Tensor wonly_gemm_fused_skinny_cuda(
        torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
        int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t l_b, int64_t g_size) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    // MT = #16-row tiles a block handles. Round UP to a supported instantiation:
    // the kernel guards `m < M`, so over-provisioning rows is correct (extra rows
    // are zero). Set {1,2,3,4,6,8,12,16} covers M up to 256 (16*16).
    const int MTneed = ((int)M + 15) / 16;
    const int MT = MTneed <= 4 ? MTneed : MTneed <= 6 ? 6 : MTneed <= 8 ? 8 : MTneed <= 12 ? 12 : 16;
    // NT=128 (NS=2) doubles accumulator frags/warp -> register pressure scales with
    // MT*NS. Safe only for short tiles; force NT=64 once MT>4 (tall tiles) to avoid spills.
    int NT = (MT <= 4 && (int)M >= 32) ? 128 : 64;
    if (const char* e = getenv("SSNF_FUSED_NT")) { int v = atoi(e); NT = (v == 128) ? 128 : (v == 64 ? 64 : NT); }
    const int gx = ((int)OUT + NT - 1) / NT;
    int splitK = ssnf::gemv_splitk_count(gx, (int)NB, 7);   // ~8x SM blocks (swept optimum)
    if (const char* e = getenv("SSNF_FUSED_SPLITK")) { int v = atoi(e); if (v > 0) splitK = v; }
    auto partial = torch::empty({(int64_t)splitK, M, OUT}, X.options().dtype(torch::kFloat32));
    auto Y = torch::empty({M, OUT}, X.options());
    const bool pipe = getenv("SSNF_FUSED_PIPE") && atoi(getenv("SSNF_FUSED_PIPE")) != 0;
    const bool ra = getenv("SSNF_FUSED_RA") && atoi(getenv("SSNF_FUSED_RA")) != 0;  // upper_cm carries the ra plane
    auto L = [&](auto Ut, auto Gt, auto MTt) {
        auto args = std::make_tuple(reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
            scale_exp.data_ptr<int8_t>(), upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
            partial.data_ptr<float>(), (int)M, (int)OUT, (int)K, (int)NB, splitK);
        auto go = [&](auto K_) { K_<<<dim3(gx, splitK), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                std::get<0>(args), std::get<1>(args), std::get<2>(args), std::get<3>(args), std::get<4>(args),
                std::get<5>(args), std::get<6>(args), std::get<7>(args), std::get<8>(args), std::get<9>(args)); };
        if (pipe)         go(wonly_gemm_fused_skinny_pipe<decltype(Ut)::value, decltype(Gt)::value, decltype(MTt)::value>);
        else if (NT==128) go(wonly_gemm_fused_skinny<decltype(Ut)::value, decltype(Gt)::value, decltype(MTt)::value, false, 128>);
        else if (ra)      go(wonly_gemm_fused_skinny<decltype(Ut)::value, decltype(Gt)::value, decltype(MTt)::value, true>);
        else              go(wonly_gemm_fused_skinny<decltype(Ut)::value, decltype(Gt)::value, decltype(MTt)::value, false>); };
#define FSC(UU,GG,MM) case MM: L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,MM>{}); break;
#define FSMT(UU,GG) switch (MT) { \
        FSC(UU,GG,1) FSC(UU,GG,2) FSC(UU,GG,3) FSC(UU,GG,4) \
        FSC(UU,GG,6) FSC(UU,GG,8) FSC(UU,GG,12) \
        default: L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,16>{}); break; }
    if      ((int)l_b==3 && (int)g_size==16) { FSMT(3,16); }
    else if ((int)l_b==2 && (int)g_size==8)  { FSMT(2,8); }
    else TORCH_CHECK(false, "fused skinny: only u3/gs16, u2/gs8");
#undef FSMT
#undef FSC
    const int tot = (int)(M * OUT), thr = 256;
    gemm_skinny_combine<<<(tot + thr - 1) / thr, thr, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, splitK);
    return Y;
}


// FAIR MXINT8 twin host. Same dispatch (adaptive NT, splitK, combine) as the SSNF
// fused host -> only the weight format/bytes differ between the two measurements.
torch::Tensor mxint8_gemm_fused_skinny_cuda(
        torch::Tensor X, torch::Tensor scale_exp, torch::Tensor qweight_cm,
        int64_t M, int64_t OUT, int64_t K, int64_t NB) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    // Round MT up to a supported instantiation (kernel guards m<M); {1,2,3,4,6,8,12,16}
    // covers M up to 256. Force NT=64 for tall tiles (MT>4) to bound register pressure.
    const int MTneed = ((int)M + 15) / 16;
    const int MT = MTneed <= 4 ? MTneed : MTneed <= 6 ? 6 : MTneed <= 8 ? 8 : MTneed <= 12 ? 12 : 16;
    int NT = (MT <= 4 && (int)M >= 32) ? 128 : 64;
    if (const char* e = getenv("SSNF_FUSED_NT")) { int v = atoi(e); NT = (v == 128) ? 128 : (v == 64 ? 64 : NT); }
    const int gx = ((int)OUT + NT - 1) / NT;
    int splitK = ssnf::gemv_splitk_count(gx, (int)NB, 7);
    if (const char* e = getenv("SSNF_FUSED_SPLITK")) { int v = atoi(e); if (v > 0) splitK = v; }
    auto partial = torch::empty({(int64_t)splitK, M, OUT}, X.options().dtype(torch::kFloat32));
    auto Y = torch::empty({M, OUT}, X.options());
    auto L = [&](auto MTt) {
        auto go = [&](auto K_) { K_<<<dim3(gx, splitK), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), scale_exp.data_ptr<int8_t>(),
            qweight_cm.data_ptr<int8_t>(), partial.data_ptr<float>(),
            (int)M, (int)OUT, (int)K, (int)NB, splitK); };
        if (NT==128) go(mxint8_gemm_fused_skinny<decltype(MTt)::value, 128>);
        else         go(mxint8_gemm_fused_skinny<decltype(MTt)::value, 64>); };
#define MXC(MM) case MM: L(std::integral_constant<int,MM>{}); break;
    switch (MT) {
        MXC(1) MXC(2) MXC(3) MXC(4) MXC(6) MXC(8) MXC(12)
        default: L(std::integral_constant<int,16>{}); break; }
#undef MXC
    const int tot = (int)(M * OUT), thr = 256;
    gemm_skinny_combine<<<(tot + thr - 1) / thr, thr, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, splitK);
    return Y;
}


// MXFP8 skinny fused GEMM launcher — identical routing to mxint8_gemm_fused_skinny_cuda.
torch::Tensor mxfp8_gemm_fused_skinny_cuda(
        torch::Tensor X, torch::Tensor scale_exp, torch::Tensor qweight_cm,
        int64_t M, int64_t OUT, int64_t K, int64_t NB) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    const int MTneed = ((int)M + 15) / 16;
    const int MT = MTneed <= 4 ? MTneed : MTneed <= 6 ? 6 : MTneed <= 8 ? 8 : MTneed <= 12 ? 12 : 16;
    int NT = (MT <= 4 && (int)M >= 32) ? 128 : 64;
    if (const char* e = getenv("SSNF_FUSED_NT")) { int v = atoi(e); NT = (v == 128) ? 128 : (v == 64 ? 64 : NT); }
    const int gx = ((int)OUT + NT - 1) / NT;
    int splitK = ssnf::gemv_splitk_count(gx, (int)NB, 7);
    if (const char* e = getenv("SSNF_FUSED_SPLITK")) { int v = atoi(e); if (v > 0) splitK = v; }
    auto partial = torch::empty({(int64_t)splitK, M, OUT}, X.options().dtype(torch::kFloat32));
    auto Y = torch::empty({M, OUT}, X.options());
    auto L = [&](auto MTt) {
        auto go = [&](auto K_) { K_<<<dim3(gx, splitK), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), scale_exp.data_ptr<int8_t>(),
            qweight_cm.data_ptr<int8_t>(), partial.data_ptr<float>(),
            (int)M, (int)OUT, (int)K, (int)NB, splitK); };
        if (NT==128) go(mxfp8_gemm_fused_skinny<decltype(MTt)::value, 128>);
        else         go(mxfp8_gemm_fused_skinny<decltype(MTt)::value, 64>); };
#define MXF(MM) case MM: L(std::integral_constant<int,MM>{}); break;
    switch (MT) {
        MXF(1) MXF(2) MXF(3) MXF(4) MXF(6) MXF(8) MXF(12)
        default: L(std::integral_constant<int,16>{}); break; }
#undef MXF
    const int tot = (int)(M * OUT), thr = 256;
    gemm_skinny_combine<<<(tot + thr - 1) / thr, thr, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, splitK);
    return Y;
}


torch::Tensor mxfp8_dequant_bf16_cuda(
        torch::Tensor scale_exp, torch::Tensor qweight_cm, int64_t OUT, int64_t K, int64_t NB) {
    auto W = torch::empty({K, OUT}, torch::dtype(torch::kBFloat16).device(scale_exp.device()));
    const int threads = 256, blocks = (int)((OUT + threads - 1) / threads);
    mxfp8_dequant_bf16_kernel<<<dim3(blocks, (int)NB), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        scale_exp.data_ptr<int8_t>(), qweight_cm.data_ptr<int8_t>(),
        reinterpret_cast<__nv_bfloat16*>(W.data_ptr<at::BFloat16>()), (int)OUT, (int)K, (int)NB);
    return W;
}


torch::Tensor mxfp6_gemm_fused_skinny_cuda(
        torch::Tensor X, torch::Tensor scale_exp, torch::Tensor qweight_cm,
        int64_t M, int64_t OUT, int64_t K, int64_t NB) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    const int MTneed = ((int)M + 15) / 16;
    const int MT = MTneed <= 4 ? MTneed : MTneed <= 6 ? 6 : MTneed <= 8 ? 8 : MTneed <= 12 ? 12 : 16;
    int NT = (MT <= 4 && (int)M >= 32) ? 128 : 64;
    if (const char* e = getenv("SSNF_FUSED_NT")) { int v = atoi(e); NT = (v == 128) ? 128 : (v == 64 ? 64 : NT); }
    const int gx = ((int)OUT + NT - 1) / NT;
    int splitK = ssnf::gemv_splitk_count(gx, (int)NB, 7);
    if (const char* e = getenv("SSNF_FUSED_SPLITK")) { int v = atoi(e); if (v > 0) splitK = v; }
    auto partial = torch::empty({(int64_t)splitK, M, OUT}, X.options().dtype(torch::kFloat32));
    auto Y = torch::empty({M, OUT}, X.options());
    auto L = [&](auto MTt) {
        auto go = [&](auto K_) { K_<<<dim3(gx, splitK), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), scale_exp.data_ptr<int8_t>(),
            qweight_cm.data_ptr<int8_t>(), partial.data_ptr<float>(),
            (int)M, (int)OUT, (int)K, (int)NB, splitK); };
        if (NT==128) go(mxfp6_gemm_fused_skinny<decltype(MTt)::value, 128>);
        else         go(mxfp6_gemm_fused_skinny<decltype(MTt)::value, 64>); };
#define MXS(MM) case MM: L(std::integral_constant<int,MM>{}); break;
    switch (MT) {
        MXS(1) MXS(2) MXS(3) MXS(4) MXS(6) MXS(8) MXS(12)
        default: L(std::integral_constant<int,16>{}); break; }
#undef MXS
    const int tot = (int)(M * OUT), thr = 256;
    gemm_skinny_combine<<<(tot + thr - 1) / thr, thr, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, splitK);
    return Y;
}


torch::Tensor mxfp6_dequant_bf16_cuda(
        torch::Tensor scale_exp, torch::Tensor qweight_cm, int64_t OUT, int64_t K, int64_t NB) {
    auto W = torch::empty({K, OUT}, torch::dtype(torch::kBFloat16).device(scale_exp.device()));
    const int threads = 256, blocks = (int)((OUT + threads - 1) / threads);
    mxfp6_dequant_bf16_kernel<<<dim3(blocks, (int)NB), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        scale_exp.data_ptr<int8_t>(), qweight_cm.data_ptr<int8_t>(),
        reinterpret_cast<__nv_bfloat16*>(W.data_ptr<at::BFloat16>()), (int)OUT, (int)K, (int)NB);
    return W;
}
