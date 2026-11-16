// kernel/core/ssnf_utils.cuh
//
// Shared device-side primitives for the SSNF-signed kernels. The doc's
// centerpiece: "extract `upper` and `shared` from the bitstream and synthesize
// an INT8 word" is needed identically by W-only GEMV, W+A GEMM and KV
// attention, so it lives here as __device__ __forceinline__ and is #included by
// every .cu. Edit the unpack in ONE place.
//
// =============================================================================
//  LAYOUT THIS HEADER DECODES  (the CURRENT certified pack — ssnf_lib/pack.py)
// =============================================================================
//  Out-innermost SoA, dense LSB-first. Per 32-element block `blk`, column `o`:
//    upper field  [nb, UB, OUT]  UB = 32*(8-l_b)/8       flat: (blk*UB+bi)*OUT+o
//    shared field [nb, SB, OUT]  SB = ceil((32/g_size)*l_b/8) flat: (blk*SB+sbi)*OUT+o
//    scale_exp    [nb, OUT] int8                        flat: blk*OUT+o
//  Within-block element k:  upper code at bit k*(8-l_b);  shared code (group
//  g=k/g_size) at bit g*l_b. Codes are dense and CAN straddle a byte boundary, and
//  the two bytes of a straddling code are OUT elements apart in memory.
//  unpack_ssnf_weight_elem() handles the straddle with a 2-byte load + shift/or.
//  This is bit-exact vs ssnf_lib.pack.dequant_weight (certified via the pytest
//  roundtrip gate); the CUDA kernels inherit that correctness for free.
//
// =============================================================================
//  bfe.s32 / 128-bit loads  (the doc's optimization — NOT enabled here)
// =============================================================================
//  Single-op `bfe.s32` extraction and int4/float4 vectorized loads require the
//  register-aligned packing the doc proposes (pad/swizzle so N whole codes fall
//  inside one 32-bit word, byte-contiguous). The dense layout above is NOT
//  byte-contiguous per code, so bfe cannot replace the shift/or path until the
//  packing is changed. When that lands:
//    (1) re-lay-out ssnf_lib.pack to register-aligned codes,
//    (2) RE-CERTIFY the new layout through the pack<->unpack roundtrip test,
//    (3) flip SSNF_USE_BFE and implement the bfe variant below.
//  bfe_s32() is provided now so the eventual switch is mechanical.

#pragma once
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifndef SSNF_USE_BFE
#define SSNF_USE_BFE 0   // 0 = portable shift/mask (current layout); 1 = bfe.s32 (target layout)
#endif

namespace ssnf {

// Notation follows the paper: `l_b` = shared field bitwidth, `g_size` = mantissa group size;
// element word = h_code*2^l_b + l_g_code (paper h*2^l_b + l_g), scaled by the E8M0 block scale.

constexpr int E_MAX = 6;   // MXINT8 (elem_bitwidth=8): E_max = n - 2 = 6

// --- HOST: choose # of key-splits so grid blocks ~ mult * #SM (occupancy variant 1)
//   The decode attention launches H*S blocks; with H tiny (8) a single split
//   leaves most SMs idle on the target GPU. Derive S from the live SM count so
//   the grid lands at ~mult x #SM (Little's-law in-flight target).
//   SSNF_KV_SPLIT_MULT (env, default 24) sweeps the multiplier; MIN_TILE keeps a
//   tiny Lk from over-splitting into trivial tiles. Returns S (#tiles per head).
//   Default raised 3->24: the sub-byte decode kernels are latency-bound (ncu:
//   DRAM ~8-19%, No-Eligible 48-88%, Waves/SM 0.4-0.7), so mult=3 (target ~0.5
//   wave) left the machine starved. mult=24 fills several waves. Measured on
//   70-SM Blackwell (H=32,Hkv=8,Lk=4680,u2/gs8), MXINT8 / SSNF decode us:
//     B=1 : 147->85 (1.72x) / 67->65    B=16: 1726->1460 (1.18x) / 1046->957 (1.09x)
//     B=32: 3452->2956 (1.17x) / ~flat  B>=64: base grid already full -> S=1 (no-op).
//   MXINT8 (latency-bound mxint8_kv_split) keeps gaining to 24; SSNF (already
//   near-saturated wide kernel) is flat 12-24 and still a net win vs the old
//   mult=3. Both formats share this path, so the fix applies to both alike; the
//   over-split cliff is beyond ~mult=48 (SSNF B=1 regresses there). NOTE: the
//   paged deploy path (kv_paged.cu) has its OWN split (base=B*Hkv, mult=8) and
//   is unchanged here.
inline int kv_split_count(long Lk, int H) {
    static int sm = -1;
    if (sm < 0) cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    int mult = 24;
    if (const char* e = getenv("SSNF_KV_SPLIT_MULT")) { int m = atoi(e); if (m > 0) mult = m; }
    constexpr int MIN_TILE = 32;                       // floor on keys/tile
    int S    = (int)(((long)mult * sm + H - 1) / H);   // splits to hit mult*#SM blocks
    int maxS = (int)((Lk + MIN_TILE - 1) / MIN_TILE);  // cap so tile >= MIN_TILE
    if (S > maxS) S = maxS;
    if (S < 1)    S = 1;
    return S;
}

// --- HOST: GEMV split-K factor. The W-only GEMV launches base_blocks =
//   ceil(OUT/128) blocks (one thread per output column); for OUT=4096 that is 32
//   blocks << the SM count. Split the K-reduction into `splitK` partial sums so the
//   grid becomes base_blocks*splitK ~ mult*#SM and fills the machine; a cheap 2nd
//   kernel sums the partials (no atomics). Capped at NB (each split needs >=1
//   32-block). SSNF_GEMV_SPLITK_MULT (env, default 3) sweeps the multiplier.
inline int gemv_splitk_count(int base_blocks, int NB, int default_mult = 3) {
    static int sm = -1;
    if (sm < 0) cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    int mult = default_mult;
    if (const char* e = getenv("SSNF_GEMV_SPLITK_MULT")) { int m = atoi(e); if (m > 0) mult = m; }
    int sp = (int)(((long)mult * sm + base_blocks - 1) / base_blocks);
    if (sp > NB) sp = NB;
    if (sp < 1)  sp = 1;
    return sp;
}

// --- E8M0 shared scale: stored exponent (int8) -> 2^exp ----------------------
__device__ __forceinline__ float e8m0_to_scale(int8_t exp) {
    return exp2f(static_cast<float>(exp));
}

// --- 2^n for a SMALL integer n (build the power-of-2 float bits directly) -----
//   Bit-exact to ldexpf/exp2f for n in [-126,127] (2^n is an exact float), but a
//   single IADD+SHL+MOV instead of the ldexpf library routine. Used to replace the
//   per-element FP8 reconstruction scale 2^(ee-MB) (n = ee-MB in [-6,0] for E3M4).
__device__ __forceinline__ float exp2i(int n) {
    return __int_as_float((127 + n) << 23);
}

// --- sign-extend the low `bits` of v (v already masked to those bits) --------
__device__ __forceinline__ int sign_extend(uint32_t v, int bits) {
    const int s = 1 << (bits - 1);
    return (static_cast<int>(v) ^ s) - s;
}

// --- PTX bit-field-extract (sign-extended); for the TARGET register-aligned
//     layout. `pos` = start bit, `len` = width. Single instruction (~4 cyc). ---
__device__ __forceinline__ int bfe_s32(int32_t word, int pos, int len) {
    int d;
    asm("bfe.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(word), "r"(pos), "r"(len));
    return d;
}

// --- extract one dense-packed signed code that may straddle two bytes --------
//   plane[(blk*BYTES + byteIdx)*OUT + o] is the byte stream; the code of width
//   `width` starts at global bit `bit0` within this block's BYTES bytes.
__device__ __forceinline__ int extract_code(const uint8_t* plane, int base_byte,
                                            int OUT, int o, int bit0, int width,
                                            int n_bytes) {
    const int bi  = bit0 >> 3;
    const int off = bit0 & 7;
    uint32_t b0 = plane[(base_byte + bi) * OUT + o];
    uint32_t code = b0 >> off;
    if (off + width > 8 && (bi + 1) < n_bytes) {
        uint32_t b1 = plane[(base_byte + bi + 1) * OUT + o];
        code |= b1 << (8 - off);
    }
    code &= (1u << width) - 1u;
    return sign_extend(code, width);
}

// --- reconstruct the INT8 MXINT8 word for element k of block blk, column o ---
//   word = upper*2^l_b + shared_expanded  (a valid MXINT8 integer; the W+A path
//   feeds this straight to IMMA, the W-only path casts to bf16 and scales).
__device__ __forceinline__ int unpack_ssnf_weight_elem(
        const uint8_t* upper, const uint8_t* shared,
        int blk, int o, int k, int OUT,
        int l_b, int g_size, int UB, int SB) {
    const int wbits = 8 - l_b;
#if SSNF_USE_BFE
    // A register-aligned packing would allow replacing the two extract_code() calls with a
    // 32-bit vectorized load + two bfe_s32() ops; not valid against the current dense layout,
    // so this path is guarded off by default.
    static_assert(SSNF_USE_BFE == 0, "bfe path needs the register-aligned pack; see header");
    return 0;
#else
    const int h_code = extract_code(upper, blk * UB, OUT, o, k * wbits, wbits, UB);
    const int g       = k >> (__ffs(g_size) - 1);   // g_size is a power of 2: avoid runtime divide
    const int l_g_code = extract_code(shared, blk * SB, OUT, o, g * l_b, l_b, SB);
    return h_code * (1 << l_b) + l_g_code;
#endif
}

// --- (l_b,g_size)-SPECIALIZED streaming block unpack (l_b<4), shared by the wide-load GEMV
//   and the KV-decode K read. One 32-element block whose UBc bytes are CONTIGUOUS at
//   `upper_cm + base*UBc` (column-major weight: base=blk*OUT+o; token-major KV:
//   base=(hk*NB+blk)*Lcap+key); per_elem(k, word) called for each reconstructed word.
//   With LB_/GSIZE_ compile-time, WBITS/UBc/SBc and the masks/shifts are constants and the
//   SEPARATED variant: yields (kd, h_code, l_g_code) separately (not the combined word) so the
//   caller can do separated-scale (sepsc) accumulation: dot = scale*(2^l_b*Σq·up + Σ_g qg·sh).
//   Same rolling-buffer unpack as stream_block_uspec; l_g_code is the current group's code.
template<int LB_, int GSIZE_, typename F>
__device__ __forceinline__ void stream_block_uspec_sep(
        const uint8_t* __restrict__ upper_cm, const uint8_t* __restrict__ shared_cm,
        long base, F per_elem) {
    constexpr int MXB = 32;
    constexpr int Up = LB_ > 0 ? LB_ : 1, Gp = GSIZE_ > 0 ? GSIZE_ : 1;   // guards: this template is also
    constexpr int UBc = MXB * (8 - Up) / 8, SBc = ((MXB / Gp) * Up + 7) / 8;  // INSTANTIATED (dead) for
    constexpr int NW = UBc / 4, NSW = (SBc + 3) / 4 > 0 ? (SBc + 3) / 4 : 1;  // LB_=-1 (u4) -> avoid <<(-)
    constexpr uint32_t umask = (1u << (8 - Up)) - 1u, usign = 1u << (8 - Up - 1);
    constexpr uint32_t smask = (1u << Up) - 1u, ssign = 1u << (Up - 1);
    constexpr int gsmask = Gp - 1, WBITS = 8 - Up;
    uint32_t sreg[NSW] = {0u};
    #pragma unroll
    for (int i = 0; i < SBc; ++i) sreg[i >> 2] |= (uint32_t)shared_cm[base * SBc + i] << (8 * (i & 3));
    const uint32_t* src = reinterpret_cast<const uint32_t*>(upper_cm + base * UBc);
    uint32_t ureg[NW];
    #pragma unroll
    for (int i = 0; i < NW; ++i) ureg[i] = src[i];
    uint64_t ubuf = 0; int unb = 0, uwi = 0;
    uint64_t sbuf = 0; int snb = 0, swi = 0; int l_g_code = 0;
    #pragma unroll
    for (int k = 0; k < MXB; ++k) {
        if (unb < WBITS) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
        const int h_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
        ubuf >>= WBITS; unb -= WBITS;
        if ((k & gsmask) == 0) {
            if (snb < LB_) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
            l_g_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign;
            sbuf >>= LB_; snb -= LB_;
        }
        per_elem(k, h_code, l_g_code);
    }
}

//   k-loop unroll makes the rolling-buffer schedule static -> ureg register-resident.
//   Bit-exact to the runtime streaming unpack. (kernel_ver3.md / compile_time_optimization.md.)
template<int LB_, int GSIZE_, typename F>
__device__ __forceinline__ void stream_block_uspec(
        const uint8_t* __restrict__ upper_cm, const uint8_t* __restrict__ shared_cm,
        long base, F per_elem) {
    constexpr int MXB = 32;
    constexpr int WBITS = 8 - LB_;
    constexpr int UBc = MXB * WBITS / 8;
    constexpr int SBc = ((MXB / GSIZE_) * LB_ + 7) / 8;
    constexpr int NW  = UBc / 4;
    constexpr int NSW = (SBc + 3) / 4 > 0 ? (SBc + 3) / 4 : 1;
    constexpr uint32_t umask = (1u << WBITS) - 1u, usign = 1u << (WBITS - 1);
    constexpr uint32_t smask = (1u << LB_) - 1u, ssign = 1u << (LB_ - 1);
    constexpr int gsmask = GSIZE_ - 1;
    uint32_t sreg[NSW] = {0u};
    #pragma unroll
    for (int i = 0; i < SBc; ++i) sreg[i >> 2] |= (uint32_t)shared_cm[base * SBc + i] << (8 * (i & 3));
    const uint32_t* src = reinterpret_cast<const uint32_t*>(upper_cm + base * UBc);
    uint32_t ureg[NW];
    #pragma unroll
    for (int i = 0; i < NW; ++i) ureg[i] = src[i];
    uint64_t ubuf = 0; int unb = 0, uwi = 0;
    uint64_t sbuf = 0; int snb = 0, swi = 0; int l_g_code = 0;
    #pragma unroll
    for (int k = 0; k < MXB; ++k) {
        if (unb < WBITS) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
        const int h_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
        ubuf >>= WBITS; unb -= WBITS;
        if ((k & gsmask) == 0) {
            if (snb < LB_) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
            l_g_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign;
            sbuf >>= LB_; snb -= LB_;
        }
        per_elem(k, h_code * (1 << LB_) + l_g_code);
    }
}

// --- SSNF-unsigned streaming unpack (ssnf_lib.pack.pack_weight_unsigned). Same dense
//   plane layout as stream_block_uspec, but the shared code is UNSIGNED (floor-upper
//   design) so it needs NO sign-extend and the int8 word is a pure bit-concat
//   (h_code<<LB_)|sh: the shared bits slot into the upper word's zeroed low LB_ bits.
//   Saves the shared xor/sub sign-extend; reconstruction is OR not signed-add.
template<int LB_, int GSIZE_, typename F>
__device__ __forceinline__ void stream_block_uspec_unsigned(
        const uint8_t* __restrict__ upper_cm, const uint8_t* __restrict__ shared_cm,
        long base, F per_elem) {
    constexpr int MXB = 32;
    constexpr int WBITS = 8 - LB_;
    constexpr int UBc = MXB * WBITS / 8;
    constexpr int SBc = ((MXB / GSIZE_) * LB_ + 7) / 8;
    constexpr int NW  = UBc / 4;
    constexpr int NSW = (SBc + 3) / 4 > 0 ? (SBc + 3) / 4 : 1;
    constexpr uint32_t umask = (1u << WBITS) - 1u, usign = 1u << (WBITS - 1);
    constexpr uint32_t smask = (1u << LB_) - 1u;
    constexpr int gsmask = GSIZE_ - 1;
    uint32_t sreg[NSW] = {0u};
    #pragma unroll
    for (int i = 0; i < SBc; ++i) sreg[i >> 2] |= (uint32_t)shared_cm[base * SBc + i] << (8 * (i & 3));
    const uint32_t* src = reinterpret_cast<const uint32_t*>(upper_cm + base * UBc);
    uint32_t ureg[NW];
    #pragma unroll
    for (int i = 0; i < NW; ++i) ureg[i] = src[i];
    uint64_t ubuf = 0; int unb = 0, uwi = 0;
    uint64_t sbuf = 0; int snb = 0, swi = 0; int l_g_code = 0;
    #pragma unroll
    for (int k = 0; k < MXB; ++k) {
        if (unb < WBITS) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
        const int h_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
        ubuf >>= WBITS; unb -= WBITS;
        if ((k & gsmask) == 0) {
            if (snb < LB_) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
            l_g_code = (int)((uint32_t)sbuf & smask);     // UNSIGNED: no sign-extend
            sbuf >>= LB_; snb -= LB_;
        }
        per_elem(k, (h_code << LB_) | l_g_code);           // OR-concat (low LB_ bits of h_code<<LB_ are 0)
    }
}

// --- MXFP8-SSNF (E3M4) streaming unpack. SAME dense plane layout as stream_block_uspec
//   (upper field width 8-LB_, shared LB_-bit signed), but each upper field is an FP8 E3M4
//   element [sign:1|exp:3|upmant:(MB-LB_)] -> yields the reconstructed FP8 VALUE (float, before
//   the block scale): val = sign * 2^(ee-MB) * (m_up*2^LB_ + sh). Bit-exact to ssnf_mxfp8(e3m4)
//   The ONLY difference vs the INT path is per-element ALU
//   (exp/mantissa split + ldexpf); memory traffic is identical -> isolates fused decode ALU.
// block_e (default 0): FOLD the E8M0 block exponent into the per-element FP scale, so the caller
//   drops a separate `* e8m0_to_scale(exp)` (an exp2f MUFU + a mul per element). block_e=0 keeps
//   the pre-block-scale value (weight dequant path, unchanged/bit-exact).
template<int LB_, int GSIZE_, typename F>
__device__ __forceinline__ void stream_block_uspec_fp8_e3m4(
        const uint8_t* __restrict__ upper_cm, const uint8_t* __restrict__ shared_cm,
        long base, F per_elem, int block_e = 0) {
    constexpr int MXB = 32, EB = 3, MB = 4, BIAS = 3, EMIN = -2;
    constexpr int WBITS = 8 - LB_, MBUP = MB - LB_;
    constexpr int UBc = MXB * WBITS / 8;
    constexpr int SBc = ((MXB / GSIZE_) * LB_ + 7) / 8;
    constexpr int NW  = UBc / 4;
    constexpr int NSW = (SBc + 3) / 4 > 0 ? (SBc + 3) / 4 : 1;
    constexpr uint32_t umask = (1u << WBITS) - 1u;
    constexpr uint32_t expmask = (1u << EB) - 1u, upmask = (1u << MBUP) - 1u, lead = 1u << MBUP;
    constexpr uint32_t smask = (1u << LB_) - 1u, ssign = 1u << (LB_ - 1);
    constexpr int gsmask = GSIZE_ - 1;
    uint32_t sreg[NSW] = {0u};
    #pragma unroll
    for (int i = 0; i < SBc; ++i) sreg[i >> 2] |= (uint32_t)shared_cm[base * SBc + i] << (8 * (i & 3));
    const uint32_t* src = reinterpret_cast<const uint32_t*>(upper_cm + base * UBc);
    uint32_t ureg[NW];
    #pragma unroll
    for (int i = 0; i < NW; ++i) ureg[i] = src[i];
    uint64_t ubuf = 0; int unb = 0, uwi = 0;
    uint64_t sbuf = 0; int snb = 0, swi = 0; int l_g_code = 0;
    #pragma unroll
    for (int k = 0; k < MXB; ++k) {
        if (unb < WBITS) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
        const uint32_t field = (uint32_t)ubuf & umask;
        ubuf >>= WBITS; unb -= WBITS;
        const uint32_t sgn = field >> (WBITS - 1);
        const uint32_t expf = (field >> MBUP) & expmask;
        const int m_up = (int)((expf ? lead : 0u) | (field & upmask));
        const int ee = expf ? (int)expf - BIAS : EMIN;
        if ((k & gsmask) == 0) {
            if (snb < LB_) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
            l_g_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign;
            sbuf >>= LB_; snb -= LB_;
        }
        int mag = m_up << LB_;
        if (sgn) mag = -mag;
        per_elem(k, (float)(mag + l_g_code) * exp2i(ee - MB + block_e));  // folded block scale (block_e=0: pre-scale)
    }
}

// --- DENSE + SELECTIVE-bfe unpack: reads the EXISTING dense upper_cm plane (pack_weight,
//   0.66x bytes, NO repack) but extracts each code with a SINGLE bfe.s32 when it does not
//   cross a 32-bit register boundary (compile-time known), and funnelshift+bfe only for the
//   few straddlers. Keeps dense bytes AND ~1 op/code. Bit-exact to stream_block_uspec.
template<int LB_, int GSIZE_, typename F>
__device__ __forceinline__ void stream_block_dense_bfe(
        const uint8_t* __restrict__ upper_cm, const uint8_t* __restrict__ shared_cm,
        long base, F per_elem) {
    constexpr int MXB = 32;
    constexpr int WBITS = 8 - LB_;
    constexpr int UBc = MXB * WBITS / 8;
    constexpr int NW  = UBc / 4;
    constexpr int SBc = ((MXB / GSIZE_) * LB_ + 7) / 8;
    constexpr int gs_shift = (GSIZE_>=2?1:0)+(GSIZE_>=4?1:0)+(GSIZE_>=8?1:0)+(GSIZE_>=16?1:0)+(GSIZE_>=32?1:0);
    const uint32_t* src = reinterpret_cast<const uint32_t*>(upper_cm + base * UBc);
    uint32_t ureg[NW];
    #pragma unroll
    for (int i = 0; i < NW; ++i) ureg[i] = src[i];
    uint32_t sreg = 0u;
    #pragma unroll
    for (int i = 0; i < SBc; ++i) sreg |= (uint32_t)shared_cm[base * SBc + i] << (8 * i);
    #pragma unroll
    for (int k = 0; k < MXB; ++k) {
        const int gb = k * WBITS, wi = gb >> 5, bo = gb & 31;
        int up;
        if (bo + WBITS <= 32) {                                  // no straddle -> 1 bfe
            up = bfe_s32((int)ureg[wi], bo, WBITS);
        } else {                                                 // straddle -> funnel window + bfe
            const uint32_t hi = (wi + 1 < NW) ? ureg[wi + 1] : 0u;
            up = bfe_s32((int)__funnelshift_r(ureg[wi], hi, bo), 0, WBITS);
        }
        const int g  = k >> gs_shift;
        const int sh = bfe_s32((int)sreg, g * LB_, LB_);
        per_elem(k, up * (1 << LB_) + sh);
    }
}

// --- REGISTER-ALIGNED unpack (ssnf_lib.pack.pack_weight_ra). Upper codes padded so each
//   (8-LB_)-bit code lies wholly inside one 32-bit word (CPW codes/word, NW words) ->
//   ONE bfe.s32 per code (HW sign-extend), no straddle/rolling-buffer/mask/sign-extend.
//   shared dense in 1 word (n_group*LB_<=8), also bfe. Reconstructs the SAME signed int8
//   word as stream_block_uspec. base = blk*OUT+o (column-major); a column's NW words are
//   contiguous at upper_ra_cm + base*NW*4.
template<int LB_, int GSIZE_, typename F>
__device__ __forceinline__ void stream_block_ra(
        const uint8_t* __restrict__ upper_ra_cm, const uint8_t* __restrict__ shared_cm,
        long base, F per_elem) {
    constexpr int MXB = 32;
    constexpr int WBITS = 8 - LB_;
    constexpr int CPW = 32 / WBITS;
    constexpr int NW  = (MXB + CPW - 1) / CPW;
    constexpr int SBc = ((MXB / GSIZE_) * LB_ + 7) / 8;
    constexpr int gs_shift = (GSIZE_>=2?1:0)+(GSIZE_>=4?1:0)+(GSIZE_>=8?1:0)+(GSIZE_>=16?1:0)+(GSIZE_>=32?1:0);
    const uint32_t* src = reinterpret_cast<const uint32_t*>(upper_ra_cm + base * (NW * 4));
    uint32_t ureg[NW];
    #pragma unroll
    for (int i = 0; i < NW; ++i) ureg[i] = src[i];
    uint32_t sreg = 0u;
    #pragma unroll
    for (int i = 0; i < SBc; ++i) sreg |= (uint32_t)shared_cm[base * SBc + i] << (8 * i);
    #pragma unroll
    for (int k = 0; k < MXB; ++k) {
        const int up = bfe_s32((int)ureg[k / CPW], (k % CPW) * WBITS, WBITS);   // 1 op, signed
        const int g  = k >> gs_shift;
        const int sh = bfe_s32((int)sreg, g * LB_, LB_);                          // 1 op, signed
        per_elem(k, up * (1 << LB_) + sh);
    }
}

// --- NIBBLE-ALIGNED re-layout streaming unpack (ssnf_lib.pack.pack_weight_relayout).
//   The (8-l_b)-bit per-element upper code is split into a high nibble (signed 4-bit,
//   16B plane, bfe-extractable with HW sign-extend, no straddle) + a small dense
//   low_un ((4-l_b)-bit) plane. Reconstructs the SAME int8 word as stream_block_uspec:
//     w = hi4*16 + low_un*2^l_b + shared.   hi4_cm: 16B/block; lowun_cm: ceil((4-l_b)*32/8)
//   B/block (u3:4, u2:8); shared_cm: SBc B/block. Bit-exact to pack_weight_relayout.
template<int LB_, int GSIZE_, typename F>
__device__ __forceinline__ void stream_block_relayout(
        const uint8_t* __restrict__ hi4_cm, const uint8_t* __restrict__ lowun_cm,
        const uint8_t* __restrict__ shared_cm, long base, F per_elem) {
    constexpr int MXB = 32;
    constexpr int LU = 4 - LB_;                          // low_un bits/elem (u3:1 u2:2)
    constexpr int LUB = (LU * MXB + 7) / 8;             // u3:4 u2:8 bytes
    constexpr int NLU = (LUB + 3) / 4;
    constexpr int SBc = ((MXB / GSIZE_) * LB_ + 7) / 8;
    constexpr int gs_shift = (GSIZE_>=2?1:0)+(GSIZE_>=4?1:0)+(GSIZE_>=8?1:0)+(GSIZE_>=16?1:0)+(GSIZE_>=32?1:0);
    constexpr uint32_t lumask = (1u << LU) - 1u;
    constexpr uint32_t smask = (1u << LB_) - 1u, ssign = 1u << (LB_ - 1);
    const uint32_t* hp = reinterpret_cast<const uint32_t*>(hi4_cm + base * 16);
    uint32_t hw[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) hw[i] = hp[i];          // 16B = 32 signed nibbles
    uint32_t lw[NLU] = {0u};
    #pragma unroll
    for (int i = 0; i < LUB; ++i) lw[i >> 2] |= (uint32_t)lowun_cm[base * LUB + i] << (8 * (i & 3));
    uint32_t sreg = 0u;
    #pragma unroll
    for (int i = 0; i < SBc; ++i) sreg |= (uint32_t)shared_cm[base * SBc + i] << (8 * (i & 3));
    #pragma unroll
    for (int k = 0; k < MXB; ++k) {
        const int hi = (int)((hw[k >> 3] >> ((k & 7) * 4)) & 0xF) - 8;    // biased nibble -> signed
        const int bit = k * LU;                                          // (4-l_b)-bit field, no straddle
        const int lu = (int)((lw[bit >> 5] >> (bit & 31)) & lumask);     // unsigned
        const int g = k >> gs_shift;
        const int sh = (int)(((sreg >> (g * LB_)) & smask) ^ ssign) - (int)ssign;
        per_elem(k, hi * 16 + (lu << LB_) + sh);
    }
}

// --- KV variant: TOKEN-MAJOR planes [H, nb, L, BYTES] (BYTES innermost; Stage
//     4a). base_u/base_h fold in the per-head + per-block offset (= same value as
//     the old [.,BYTES,L] base since UB*L == L*UB); a block's BYTES bytes for a
//     fixed `key` are CONTIGUOUS (stride 1), so the per-element address is
//     base + key*BYTES + byteIdx and the straddle byte is the adjacent one. This
//     makes a warp's 32 head_dim reads at a fixed key coalesce. Same straddle
//     handling, same codes. (Stage 4b's word-aligned bfe variant was tried and
//     reverted — slower; the kernel is load-latency/MLP-bound, not extraction-
//     instruction-bound..) ------------------------------
__device__ __forceinline__ int unpack_ssnf_kv_elem(
        const uint8_t* upper, const uint8_t* shared,
        long base_u, long base_h, int L, int key, int k,
        int l_b, int g_size, int UB, int SB) {
    const int wbits = 8 - l_b;
    const int g = k >> (__ffs(g_size) - 1);          // g_size is a power of 2: avoid runtime divide
    const long ku = base_u + (long)key * UB;     // this key's contiguous upper bytes
    const long kh = base_h + (long)key * SB;     // this key's contiguous shared bytes
    // upper
    const int ub0 = (k * wbits) >> 3, uoff = (k * wbits) & 7;
    uint32_t code = (uint32_t)upper[ku + ub0] >> uoff;
    if (uoff + wbits > 8 && (ub0 + 1) < UB)
        code |= (uint32_t)upper[ku + ub0 + 1] << (8 - uoff);
    code &= (1u << wbits) - 1u;
    const int h_code = sign_extend(code, wbits);
    // shared
    const int sb0 = (g * l_b) >> 3, soff = (g * l_b) & 7;
    uint32_t sc = (uint32_t)shared[kh + sb0] >> soff;
    if (soff + l_b > 8 && (sb0 + 1) < SB)
        sc |= (uint32_t)shared[kh + sb0 + 1] << (8 - soff);
    sc &= (1u << l_b) - 1u;
    const int l_g_code = sign_extend(sc, l_b);
    return h_code * (1 << l_b) + l_g_code;
}

// --- KV variant, l_b==4 FAST PATH: upper(4b) and shared(4b) are both nibble-
//   aligned, so NO straddle ever happens. Replaces the general path's
//   conditional 2-byte load + variable mask + xor-sign-extend with a single
//   byte load + one bfe.s32 per plane (sign-extended in HW). u4 is the
//   fewest-bytes config (16B upper/block) and the KV benchmark default; the
//   general unpack_ssnf_kv_elem stays for l_b in {2,3}. Bit-exact: up*16+sh with
//   both codes 4-bit two's-complement == the general reconstruction at l_b=4. ----
__device__ __forceinline__ int unpack_ssnf_kv_elem_u4(
        const uint8_t* upper, const uint8_t* shared,
        long base_u, long base_h, int key, int k,
        int g_size, int UB, int SB) {
    const long ku = base_u + (long)key * UB;
    const long kh = base_h + (long)key * SB;
    const int h_code = bfe_s32((int)upper[ku + (k >> 1)], (k & 1) * 4, 4);
    const int g       = k >> (__ffs(g_size) - 1);
    const int l_g_code = bfe_s32((int)shared[kh + (g >> 1)], (g & 1) * 4, 4);
    return h_code * 16 + l_g_code;
}

// --- KV variant, general l_b (handles straddle) but extract via bfe.s32 instead of
//   the manual shift/mask/xor-sign-extend. Same loads + same straddle guard as
//   unpack_ssnf_kv_elem (safe: never reads past the plane), but the per-element field
//   decode is one HW instruction per plane. Bit-exact to unpack_ssnf_kv_elem. ----
__device__ __forceinline__ int unpack_ssnf_kv_elem_bfe(
        const uint8_t* upper, const uint8_t* shared,
        long base_u, long base_h, int key, int k,
        int l_b, int g_size, int UB, int SB) {
    const int wbits = 8 - l_b;
    const int g = k >> (__ffs(g_size) - 1);
    const long ku = base_u + (long)key * UB;
    const long kh = base_h + (long)key * SB;
    const int ub0 = (k * wbits) >> 3, uoff = (k * wbits) & 7;
    uint32_t uw = (uint32_t)upper[ku + ub0];
    if (uoff + wbits > 8 && (ub0 + 1) < UB) uw |= (uint32_t)upper[ku + ub0 + 1] << 8;
    const int h_code = bfe_s32((int)uw, uoff, wbits);
    const int sb0 = (g * l_b) >> 3, soff = (g * l_b) & 7;
    uint32_t sw = (uint32_t)shared[kh + sb0];
    if (soff + l_b > 8 && (sb0 + 1) < SB) sw |= (uint32_t)shared[kh + sb0 + 1] << 8;
    const int l_g_code = bfe_s32((int)sw, soff, l_b);
    return (h_code << l_b) + l_g_code;
}

// NOTE (body micro-opt, REFUTED twice): wide-loading a block's plane words into registers to
//   replace the per-element scattered shared byte-reads — both v1 (per-lane load-all) AND v2
//   (warp-cooperative coalesced load + __shfl distribute) were bit-exact but ~2.6x SLOWER. Root cause
//   (ncu): holding w[8] pushes the kernel to 93 reg/thread -> 36.7% occupancy, REGISTER-LIMITED. The
//   paged body is OCCUPANCY/latency-bound (Eligible warps 1.14), NOT shared-bandwidth-bound — the
//   11.67% shared efficiency is real but non-binding. The register-LIGHT per-element unpack_ssnf_kv_elem_bfe
//   is optimal precisely because it minimizes registers -> max occupancy. Don't spend registers here.

// --- KV variant, NO shared residual (rotated low-bit path). Under block-Had-32 the shared
//   residual is iso-bit worthless (measured), so we drop it: the int8 word is
//   just h_code << l_b (l_b trailing zeros). Reads ONLY the upper field -> no shared load, no group
//   index, no SB. Bit-exact to unpack_ssnf_kv_elem with l_g_code==0; equals plain MXINT(8-l_b) staged
//   to int8. Same upper/scale planes as the SSNF pack -> storage format unchanged (shared absent).
__device__ __forceinline__ int unpack_ssnf_kv_elem_nores(
        const uint8_t* upper, long base_u, int key, int k, int l_b, int UB) {
    const int wbits = 8 - l_b;
    const long ku = base_u + (long)key * UB;
    const int ub0 = (k * wbits) >> 3, uoff = (k * wbits) & 7;
    uint32_t uw = (uint32_t)upper[ku + ub0];
    if (uoff + wbits > 8 && (ub0 + 1) < UB) uw |= (uint32_t)upper[ku + ub0 + 1] << 8;
    const int h_code = bfe_s32((int)uw, uoff, wbits);
    return h_code << l_b;
}

// --- KV variant, MXFP8-SSNF (E3M4): single-element FP8 VALUE (pre-block-scale).
//   FP8 analog of unpack_ssnf_kv_elem (token-major planes [.,L,BYTES]). Extracts the
//   (8-l_b)-bit FP8 E3M4 field [sign|exp:3|upmant:(4-l_b)] for element k and the l_b-bit
//   SIGNED group-shared code, returns the reconstructed FP8 value
//     sign * 2^(ee-MB) * (m_up*2^l_b + l_g_code)
//   (the caller multiplies the E8M0 block scale). Bit-exact to
//   stream_block_uspec_fp8_e3m4 / ssnf_mxfp8(e3m4); same straddle handling as the
//   INT unpack_ssnf_kv_elem (the ONLY change is the per-element FP decode). ---------
__device__ __forceinline__ float unpack_ssnf_kv_elem_fp8(
        const uint8_t* upper, const uint8_t* shared,
        long base_u, long base_h, int key, int k,
        int l_b, int g_size, int UB, int SB, int block_e = 0) {   // block_e: fold E8M0 scale (drops caller exp2f+mul)
    constexpr int EB = 3, MB = 4, BIAS = 3, EMIN = -2;   // E3M4
    const int wbits = 8 - l_b, mbup = MB - l_b;
    const int g = k >> (__ffs(g_size) - 1);
    const long ku = base_u + (long)key * UB;
    const long kh = base_h + (long)key * SB;
    // upper FP8 field (may straddle two bytes)
    const int ub0 = (k * wbits) >> 3, uoff = (k * wbits) & 7;
    uint32_t code = (uint32_t)upper[ku + ub0] >> uoff;
    if (uoff + wbits > 8 && (ub0 + 1) < UB)
        code |= (uint32_t)upper[ku + ub0 + 1] << (8 - uoff);
    code &= (1u << wbits) - 1u;
    const uint32_t sgn  = code >> (wbits - 1);
    const uint32_t expf = (code >> mbup) & ((1u << EB) - 1u);
    const uint32_t upm  = code & ((1u << mbup) - 1u);
    const int m_up = (int)((expf ? (1u << mbup) : 0u) | upm);   // implicit leading 1 for normals
    const int ee   = expf ? (int)expf - BIAS : EMIN;
    // shared SIGNED l_b-bit code for group g (may straddle)
    const int sb0 = (g * l_b) >> 3, soff = (g * l_b) & 7;
    uint32_t sc = (uint32_t)shared[kh + sb0] >> soff;
    if (soff + l_b > 8 && (sb0 + 1) < SB)
        sc |= (uint32_t)shared[kh + sb0 + 1] << (8 - soff);
    sc &= (1u << l_b) - 1u;
    const int l_g_code = sign_extend(sc, l_b);
    int mag = m_up << l_b;
    if (sgn) mag = -mag;
    return (float)(mag + l_g_code) * exp2i(ee - MB + block_e);   // folded block scale (block_e=0: pre-scale)
}

// =============================================================================
//  PACK / QUANTIZE primitives (device counterpart of ssnf_lib.pack.decompose) —
//  shared by the KV write/append kernels (bit-pack tail) and the W+A activation
//  pre-pass (int8-word combine tail). New numerics: none — bit-exact to
//  pack.decompose / reference.quant_act(share=True).
// =============================================================================

// E8M0 exponent for a block's max-abs:  clamp(floor(log2 amax) - E_MAX, +-127).
__device__ __forceinline__ int e8m0_exp_from_amax(float amax) {
    int e = (int)floorf(log2f(fmaxf(amax, 1e-30f))) - E_MAX;
    return max(-127, min(127, e));
}

// Decompose one 32-element block (thread holds x[32]) -> (8-l_b)-bit h_word[32] +
// l_b-bit l_g_word[32/g_size] (one per g_size-group) + E8M0 exponent (return). Identical to
// pack.decompose: h_word = clip(round(x/(sa*2^l_b))); l_g_word from the group-mean
// residual. g_size is a power of two. h_word/l_g_word buffers sized 32 / 32/g_size.
//  H (optional, 32 per-channel importances for this block) turns the shared group-mean into a
//  WEIGHTED mean (H=nullptr -> uniform -> bit-exact legacy). radius>0 = trunc search: seed at the
//  rounded weighted mean, try l_g_word+{-radius..radius}, RE-ROUND h_word per candidate, keep
//  argmin_g Sum_k H_k(recon-x)^2 (radius=0 == weighted mean == hmean). Matches ssnf_lib.pack.decompose.
__device__ __forceinline__ int decompose_ssnf_block(
        const float* __restrict__ x, int l_b, int g_size, int* h_word, int* l_g_word,
        const float* __restrict__ H = nullptr, int radius = 0) {
    float amax = 1e-30f;
    #pragma unroll
    for (int k = 0; k < 32; ++k) amax = fmaxf(amax, fabsf(x[k]));
    const int ea = e8m0_exp_from_amax(amax);
    const float sa = exp2f((float)ea), s_unshared = sa * exp2f((float)l_b);
    const int qmax = (1 << (7 - l_b)) - 1, smin = -(1 << (l_b - 1)), smax = (1 << (l_b - 1)) - 1;
    const int ng = 32 / g_size;
    for (int g = 0; g < ng; ++g) {
        float num = 0.0f, den = 0.0f;                          // weighted-mean seed
        for (int kk = 0; kk < g_size; ++kk) {
            const int k = g * g_size + kk;
            h_word[k] = max(-qmax, min(qmax, (int)rintf(x[k] / s_unshared)));
            const float w = H ? H[k] : 1.0f;
            num += w * (x[k] - (float)h_word[k] * s_unshared);
            den += w;
        }
        int l_g_seed = max(smin, min(smax, (int)rintf(num / den / sa)));
        if (radius == 0) { l_g_word[g] = l_g_seed; continue; }
        int best_l_g_word = l_g_seed; float best_err = 1e30f;              // trunc: search sh, re-round upper
        for (int d = -radius; d <= radius; ++d) {
            const int sh = max(smin, min(smax, l_g_seed + d));
            float err = 0.0f;
            for (int kk = 0; kk < g_size; ++kk) {
                const int k = g * g_size + kk;
                const int up = max(-qmax, min(qmax, (int)rintf((x[k] - (float)sh * sa) / s_unshared)));
                const float rec = (float)up * s_unshared + (float)sh * sa, e = rec - x[k];
                err += (H ? H[k] : 1.0f) * e * e;
            }
            if (err < best_err) { best_err = err; best_l_g_word = sh; }
        }
        for (int kk = 0; kk < g_size; ++kk) {                       // finalize upper at best_l_g_word
            const int k = g * g_size + kk;
            h_word[k] = max(-qmax, min(qmax, (int)rintf((x[k] - (float)best_l_g_word * sa) / s_unshared)));
        }
        l_g_word[g] = best_l_g_word;
    }
    return ea;
}


// light-SSNF decompose: same stored format as decompose_ssnf_block, but INTEGER-friendly.
// One FP round per element (q8 = round(x/sa)); everything after is integer — the
// per-element FP residual (x - h_word*s_unshared) and the FP group-mean are replaced
// by integer subtract + integer round-to-nearest mean. h_word = round(q8 / 2^l_b) and
// shared = round(mean(q8 - h_word*2^l_b)) in pure int. (Validated ~= decompose_ssnf_block
// in QSNR/PPL; a double-rounding variant of SSNF.)
//   H (optional, 32 per-channel importances): shared = round(H-WEIGHTED mean of the integer low
//   bits) = INT-hmean. H=nullptr -> uniform int round-mean -> bit-exact legacy.
//   Cheaper than FP hmean: no per-element FP residual (x - up*s_unshared); the int low-bit res is
//   what gets stored anyway. Matches ssnf_lib.pack.decompose_lightms.
__device__ __forceinline__ int decompose_lightms_block(
        const float* __restrict__ x, int l_b, int g_size, int* h_word, int* l_g_word,
        const float* __restrict__ H = nullptr) {
    float amax = 1e-30f;
    #pragma unroll
    for (int k = 0; k < 32; ++k) amax = fmaxf(amax, fabsf(x[k]));
    const int ea = e8m0_exp_from_amax(amax);
    const float inv_sa = exp2f(-(float)ea);
    const int qmax = (1 << (7 - l_b)) - 1, smin = -(1 << (l_b - 1)), smax = (1 << (l_b - 1)) - 1;
    const int half_u = 1 << (l_b - 1), lg = __ffs(g_size) - 1, half_g = g_size >> 1;
    int q8[32];
    #pragma unroll
    for (int k = 0; k < 32; ++k) q8[k] = (int)rintf(x[k] * inv_sa);   // only FP round / elem
    const int ng = 32 / g_size;
    for (int g = 0; g < ng; ++g) {
        int ssum = 0; float num = 0.0f, den = 0.0f;
        for (int kk = 0; kk < g_size; ++kk) {
            const int k = g * g_size + kk;
            const int q = q8[k];
            const int sgn = q >> 31;                     // 0 (q>=0) or -1 (q<0)
            const int mag = (((q ^ sgn) - sgn) + half_u) >> l_b;   // (|q|+half_u)>>l_b
            int up = (mag ^ sgn) - sgn;                  // branchless round-half-away (==branchy legacy)
            up = max(-qmax, min(qmax, up));
            h_word[k] = up;
            const int res = max(smin, min(smax, q - (up << l_b)));
            if (H) { const float w = H[k]; num += w * (float)res; den += w; }   // weighted (warp-uniform)
            else ssum += res;                            // unweighted int round-mean
        }
        const int m = H ? (int)rintf(num / den)                                 // weighted round-mean
                        : ((ssum >= 0) ? ((ssum + half_g) >> lg) : -(((-ssum) + half_g) >> lg));
        l_g_word[g] = max(smin, min(smax, m));
    }
    return ea;
}

// Dense LSB bit-pack tail (inverse of extract_code): n codes of `width` bits,
// LSB-first, into a contiguous `nbytes`-byte buffer. A code straddles at most two
// bytes (width<=7). Used by the KV write/append kernels to produce the certified
// token-major planes that kv_decode_* reads back.
__device__ __forceinline__ void pack_codes_lsb(
        const int* __restrict__ codes, int n, int width, uint8_t* __restrict__ buf, int nbytes) {
    for (int i = 0; i < nbytes; ++i) buf[i] = 0;
    for (int i = 0; i < n; ++i) {
        const uint32_t code = (uint32_t)codes[i] & ((1u << width) - 1u);
        const int bit = i * width, by = bit >> 3, off = bit & 7;
        buf[by] |= (uint8_t)(code << off);
        if (off + width > 8 && by + 1 < nbytes) buf[by + 1] |= (uint8_t)(code >> (8 - off));
    }
}

}  // namespace ssnf

// =============================================================================
//  Micro-profiling: clock64() spans, compiled out unless requested.
//  Build with  -DENABLE_PROFILING  and pass a global cycle buffer to the kernel.
//  WARNING: inserting clock64() perturbs the scheduler/register allocation —
//  use only to direction-find, never as the reported number (use cuda.Event +
//  ncu for that). See README "Profiling".
// =============================================================================
#ifdef ENABLE_PROFILING
  #define SSNF_PROF_DECL  unsigned long long _prof_t0 = 0ull, _prof_dt = 0ull;
  #define SSNF_PROF_START() do { _prof_t0 = clock64(); } while (0)
  #define SSNF_PROF_STOP(buf, idx) do { _prof_dt = clock64() - _prof_t0; (buf)[(idx)] = _prof_dt; } while (0)
#else
  #define SSNF_PROF_DECL
  #define SSNF_PROF_START() do {} while (0)
  #define SSNF_PROF_STOP(buf, idx) do {} while (0)
#endif
