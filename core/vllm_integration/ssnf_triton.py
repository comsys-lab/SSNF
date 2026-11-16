"""SSNF 5.44b fused mixed-input GEMM in Triton — reads the dense sub-byte planes
(scale_exp[NB,OUT], upper_cm[NB,OUT,UB=20], shared_cm[NB,OUT,SB=1]) and unpacks in the
mainloop (NO bf16 materialize). u3/gs16 specialized. y = x @ W where
W[k,o] = (h_word(5b,signed) * 2^u + l_g_word_{k//16}(3b,signed)) * 2^scale_exp[k//32,o].
"""
import os
from collections.abc import Callable

import torch, triton, triton.language as tl

# u3/gs16 constants (constexpr so the @jit kernel can read them)
WBITS = tl.constexpr(5)      # 8 - u
UBLK = tl.constexpr(20)      # UB bytes/column  (32*5/8)
GS = tl.constexpr(16)
U = tl.constexpr(3)
UBLK_PY = 20


def _rw_configs() -> list[triton.Config]:
    # curated compute-tiles + method-3 MEMORY-BOUND tiles (small BN, high occupancy: warps=8, stages=2)
    # so SSNF's fewer bytes (20 vs mxint8 32) convert to time at small/medium M.
    specs = [(64, 64, 4, 3), (64, 128, 4, 3), (128, 64, 4, 3), (128, 64, 8, 3),
             (128, 128, 4, 3), (128, 128, 8, 4), (256, 64, 4, 3), (256, 64, 8, 3),
             (256, 64, 8, 4), (256, 128, 8, 4), (128, 64, 4, 4), (64, 64, 4, 2),
             (128, 32, 4, 3), (256, 32, 8, 3),
             # --- method 3: memory-bound-friendly (small BN, warps=8/occupancy, stages=2) ---
             (64, 32, 8, 2), (64, 32, 8, 3), (128, 32, 8, 3), (64, 64, 8, 2),
             (64, 64, 8, 3), (32, 64, 4, 3), (32, 128, 4, 3), (256, 32, 8, 4),
             # --- num_warps=2 (register-light -> higher occupancy): fixes the medium-M valley.
             # ncu showed the rw kernel is register-limited; dropping to 2 warps/block frees
             # registers -> more blocks/SM, which wins at small and medium M. ---
             (64, 128, 2, 3), (64, 64, 2, 3), (64, 128, 2, 2), (64, 64, 2, 2),
             (32, 128, 2, 3), (128, 64, 2, 3), (64, 256, 2, 3), (32, 64, 2, 3)]
    # NOTE: even smaller nw2 tiles do not help the attention projections — the lost compute
    # efficiency offsets the occupancy gain; split-K is what helps those shapes.
    return [triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn}, num_warps=w, num_stages=s) for (bm, bn, w, s) in specs]


def _mbucket(M: int) -> int:
    """Power-of-two autotune bucket of M (capped at 512); the kernel body ignores it, so the
    compiled binary and the output are unchanged."""
    m = int(M)
    if m <= 16:
        return 16
    if m >= 512:
        return 512
    return 1 << (m - 1).bit_length()


@triton.autotune(configs=_rw_configs(), key=['OUT', 'K'])   # method-1 twin: byte-extract (no where-chain)
@triton.jit
def _ssnf_gemm_kernel(
        x_ptr, sc_ptr, up_ptr, sh_ptr, y_ptr,
        M, OUT, K, NB,
        stride_xm, stride_xk,
        stride_scb, stride_sco,      # scale_exp [NB, OUT]
        stride_upc,                  # upper_cm flat col stride (=UB), byte stride=1
        stride_shc,                  # shared_cm flat col stride (=SB)
        stride_ym, stride_yo,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)          # [BM]
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)          # [BN]
    rk = tl.arange(0, 32)                                 # one SSNF block
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = rm < M
    mask_n = rn < OUT
    # precompute per-k bit indices (constants over blk)
    k2 = rk[:, None]                                      # [32,1]
    gb = k2 * WBITS
    bi = gb // 8
    bo = gb % 8
    bip = tl.minimum(bi + 1, UBLK - 1)
    g = k2 // GS                                          # [32,1] group idx
    for blk in range(0, NB):
        # A tile [BM,32]
        a_ptr = x_ptr + rm[:, None] * stride_xm + (blk * 32 + rk[None, :]) * stride_xk
        a = tl.load(a_ptr, mask=mask_m[:, None], other=0.0).to(tl.bfloat16)      # [BM,32]
        # column index col = blk*OUT + o  (o = rn)
        col = blk * OUT + rn                                                     # [BN]
        # scale [BN]
        se = tl.load(sc_ptr + blk * stride_scb + rn * stride_sco, mask=mask_n, other=0).to(tl.int32)
        scale = tl.exp2(se.to(tl.float32))                                       # [BN]
        # upper: [32,BN] gather. ba/bb are bytes at col*UB + bi / bi+1
        base = col[None, :] * stride_upc                                         # [1,BN]
        ba = tl.load(up_ptr + base + bi, mask=mask_n[None, :], other=0).to(tl.int32)
        bb = tl.load(up_ptr + base + bip, mask=mask_n[None, :], other=0).to(tl.int32)
        code = ((ba >> bo) | (bb << (8 - bo))) & 0x1F                            # [32,BN]
        up = (code ^ 16) - 16                                                    # sign-extend 5b
        # shared: 1 byte/col; extract 3b at g*3
        shb = tl.load(sh_ptr + col * stride_shc, mask=mask_n, other=0).to(tl.int32)   # [BN]
        l_g_code = (shb[None, :] >> (g * U)) & 0x7                                # [32,BN]
        sh = (l_g_code ^ 4) - 4                                                   # sign-extend 3b
        word = up * (1 << U) + sh                                               # [32,BN] int
        b = word.to(tl.float32) * scale[None, :]                                # [32,BN]
        acc += tl.dot(a, b.to(tl.bfloat16), out_dtype=tl.float32)
    y_ptr2 = y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yo
    tl.store(y_ptr2, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(configs=_rw_configs(), key=['OUT', 'K', 'MB'])   # MB = power-of-two M bucket
# NOTE: with a ('OUT','K')-only key the config chosen during warmup was reused at every decode M,
# which is slow for the small decode batches. Bucketing M into the key lets each band tune itself.
# Output is bit-identical (the body does not depend on MB).
@triton.jit
def _ssnf_gemm_rw_kernel(
        x_ptr, sc_ptr, up32_ptr, sh_ptr, y_ptr,
        M, OUT, K, NB, MB,
        stride_xm, stride_xk,
        stride_scb, stride_sco,
        stride_shc,
        stride_ym, stride_yo,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        GS_P: tl.constexpr = 16, SB: tl.constexpr = 1):
    # register-word unpack: load the 5 uint32 words per column ONCE, extract all 32 codes
    # via compile-time bit positions (no per-(k,o) byte gather). u3; group size GS_P (16 or 8).
    # The upper field is group-size-independent; only the shared field differs (SB bytes/col:
    # gs16 -> 1 byte / 2 groups, gs8 -> 2 bytes / 4 groups, LSB-first per _pack_codes_lsb).
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = rm < M; mask_n = rn < OUT
    k2 = rk[:, None]                       # [32,1]
    gb = k2 * WBITS
    wi = gb // 32; bo = gb % 32            # word index / bit offset (uint32 stream)
    strad = (bo + WBITS) > 32
    sh_amt = tl.where(strad, 32 - bo, 1)
    g = k2 // GS_P
    for blk in range(0, NB):
        a_ptr = x_ptr + rm[:, None] * stride_xm + (blk * 32 + rk[None, :]) * stride_xk
        a = tl.load(a_ptr, mask=mask_m[:, None], other=0.0).to(tl.bfloat16)
        col = blk * OUT + rn                                   # [BN]
        se = tl.load(sc_ptr + blk * stride_scb + rn * stride_sco, mask=mask_n, other=0).to(tl.int32)
        scale = tl.exp2(se.to(tl.float32))                     # [BN]
        b5 = col * 5
        u0 = tl.load(up32_ptr + b5 + 0, mask=mask_n, other=0).to(tl.uint32)
        u1 = tl.load(up32_ptr + b5 + 1, mask=mask_n, other=0).to(tl.uint32)
        u2 = tl.load(up32_ptr + b5 + 2, mask=mask_n, other=0).to(tl.uint32)
        u3 = tl.load(up32_ptr + b5 + 3, mask=mask_n, other=0).to(tl.uint32)
        u4 = tl.load(up32_ptr + b5 + 4, mask=mask_n, other=0).to(tl.uint32)
        u0 = u0[None, :]; u1 = u1[None, :]; u2 = u2[None, :]; u3 = u3[None, :]; u4 = u4[None, :]  # [1,BN]
        w  = tl.where(wi == 0, u0, tl.where(wi == 1, u1, tl.where(wi == 2, u2, tl.where(wi == 3, u3, u4))))
        wn = tl.where(wi == 0, u1, tl.where(wi == 1, u2, tl.where(wi == 2, u3, u4)))   # wi<4 when straddling
        hi = tl.where(strad, wn << sh_amt.to(tl.uint32), tl.zeros_like(w))
        code = ((w >> bo.to(tl.uint32)) | hi) & 0x1F           # [32,BN] uint32
        up = (code.to(tl.int32) ^ 16) - 16
        shb = tl.load(sh_ptr + col * stride_shc, mask=mask_n, other=0).to(tl.int32)    # [BN] byte 0
        if SB == 2:
            sh1 = tl.load(sh_ptr + col * stride_shc + 1, mask=mask_n, other=0).to(tl.int32)
            shb = shb | (sh1 << 8)                                                     # LSB-first 16b
        l_g_code = (shb[None, :] >> (g * U)) & 0x7
        sh = (l_g_code ^ 4) - 4
        word = up * (1 << U) + sh
        b = word.to(tl.float32) * scale[None, :]
        acc += tl.dot(a, b.to(tl.bfloat16), out_dtype=tl.float32)
    y_ptr2 = y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yo
    tl.store(y_ptr2, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def _gk_configs() -> list[triton.Config]:
    # gather kernel: adds BLOCK_K (32/64/128 = 1/2/4 SSNF blocks per K-step) to the search.
    specs = []
    for bm, bn, w, s in [(64, 64, 4, 3), (128, 64, 4, 3), (128, 64, 8, 3), (128, 128, 4, 3),
                         (128, 128, 8, 4), (256, 64, 8, 3), (256, 64, 8, 4), (256, 128, 8, 4)]:
        for bk in (32, 64, 128):
            specs.append(triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk}, num_warps=w, num_stages=s))
    return specs


@triton.autotune(configs=_gk_configs(), key=['OUT', 'K'])
@triton.jit
def _ssnf_gemm_gk_kernel(
        x_ptr, sc_ptr, up32_ptr, sh_ptr, y_ptr, M, OUT, K, NB,
        stride_xm, stride_xk, stride_scb, stride_sco, stride_shc, stride_ym, stride_yo,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # BLOCK_K = 32*NB_SUB SSNF blocks per K-step; register-WORD load but [BK,BN] gather over rows
    # spanning NB_SUB blocks (each row's block = blk_base + rk//32). Bigger tl.dot K.
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = rm < M; mask_n = rn < OUT
    k2 = rk[:, None]                                   # [BK,1]
    sub = k2 // 32; kk = k2 % 32
    gb = kk * WBITS; wi = gb // 32; bo = gb % 32
    strad = (bo + WBITS) > 32; sh_amt = tl.where(strad, 32 - bo, 1)
    g = kk // GS
    NKS = K // BLOCK_K
    for ks in range(0, NKS):
        blk = ks * (BLOCK_K // 32) + sub                # [BK,1] block index per row
        a_ptr = x_ptr + rm[:, None] * stride_xm + (ks * BLOCK_K + rk[None, :]) * stride_xk
        a = tl.load(a_ptr, mask=mask_m[:, None], other=0.0).to(tl.bfloat16)   # [BM,BK]
        col = blk * OUT + rn[None, :]                    # [BK,BN]
        se = tl.load(sc_ptr + blk * stride_scb + rn[None, :] * stride_sco, mask=mask_n[None, :], other=0).to(tl.int32)
        scale = tl.exp2(se.to(tl.float32))               # [BK,BN]
        b5 = col * 5
        w = tl.load(up32_ptr + b5 + wi, mask=mask_n[None, :], other=0).to(tl.uint32)
        wn = tl.load(up32_ptr + b5 + tl.minimum(wi + 1, 4), mask=mask_n[None, :], other=0).to(tl.uint32)
        hi = tl.where(strad, wn << sh_amt.to(tl.uint32), tl.zeros_like(w))
        code = ((w >> bo.to(tl.uint32)) | hi) & 0x1F
        up = (code.to(tl.int32) ^ 16) - 16
        shb = tl.load(sh_ptr + col * stride_shc, mask=mask_n[None, :], other=0).to(tl.int32)
        l_g_code = (shb >> (g * U)) & 0x7
        sh = (l_g_code ^ 4) - 4
        word = up * (1 << U) + sh
        b = word.to(tl.float32) * scale
        acc += tl.dot(a, b.to(tl.bfloat16), out_dtype=tl.float32)
    y2 = y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yo
    tl.store(y2, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def ssnf_gemm_gk(x: torch.Tensor, scale_exp: torch.Tensor, upper_cm: torch.Tensor,
                 shared_cm: torch.Tensor, M: int, OUT: int, K: int, NB: int) -> torch.Tensor:
    """Fused GEMM via the BLOCK_K-gather kernel; y = x @ W in bf16."""
    y = torch.empty((M, OUT), dtype=torch.bfloat16, device=x.device)
    up32 = upper_cm.view(torch.int32).reshape(-1); sh = shared_cm.reshape(-1)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(OUT, META['BLOCK_N']))
    _ssnf_gemm_gk_kernel[grid](x, scale_exp, up32, sh, y, M, OUT, K, NB,
        x.stride(0), x.stride(1), scale_exp.stride(0), scale_exp.stride(1), 1,
        y.stride(0), y.stride(1))
    return y


# ---- FAIR MXINT8 twin: SAME Triton vehicle (tiling/autotune) as SSNF, int8 unpack (trivial) ----
@triton.autotune(configs=_rw_configs(), key=['OUT', 'K', 'MB'])   # MB: see _ssnf_gemm_rw_kernel
@triton.jit
def _mxint8_gemm_kernel(
        x_ptr, sc_ptr, qw_ptr, y_ptr, M, OUT, K, NB, MB,
        stride_xm, stride_xk, stride_scb, stride_sco, stride_ym, stride_yo,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = rm < M; mask_n = rn < OUT
    for blk in range(0, NB):
        a_ptr = x_ptr + rm[:, None] * stride_xm + (blk * 32 + rk[None, :]) * stride_xk
        a = tl.load(a_ptr, mask=mask_m[:, None], other=0.0).to(tl.bfloat16)
        col = blk * OUT + rn                                   # [BN]
        se = tl.load(sc_ptr + blk * stride_scb + rn * stride_sco, mask=mask_n, other=0).to(tl.int32)
        scale = tl.exp2(se.to(tl.float32))                     # [BN]
        # qweight_cm[blk,o,kk] flat = col*32 + kk  (int8, no bit-unpack)
        code = tl.load(qw_ptr + col[None, :] * 32 + rk[:, None], mask=mask_n[None, :], other=0).to(tl.int32)  # [32,BN]
        b = code.to(tl.float32) * scale[None, :]
        acc += tl.dot(a, b.to(tl.bfloat16), out_dtype=tl.float32)
    y2 = y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yo
    tl.store(y2, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def mxint8_gemm(x: torch.Tensor, scale_exp: torch.Tensor, qweight_cm: torch.Tensor,
                M: int, OUT: int, K: int, NB: int) -> torch.Tensor:
    """Fair MXINT8 twin of ssnf_gemm (same Triton vehicle, trivial int8 unpack)."""
    y = torch.empty((M, OUT), dtype=torch.bfloat16, device=x.device)
    qw = qweight_cm.reshape(-1)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(OUT, META['BLOCK_N']))
    _mxint8_gemm_kernel[grid](x, scale_exp, qw, y, M, OUT, K, NB, _mbucket(M),
        x.stride(0), x.stride(1), scale_exp.stride(0), scale_exp.stride(1),
        y.stride(0), y.stride(1))
    return y


def ssnf_gemm(x: torch.Tensor, scale_exp: torch.Tensor, upper_cm: torch.Tensor,
              shared_cm: torch.Tensor, M: int, OUT: int, K: int, NB: int, BM: int = 64,
              BN: int = 64, rw: bool = True, g_size: int = 16) -> torch.Tensor:
    # rw (register-word) is the DEFAULT (proven E2E, no regression). byte-extract (rw=False) is
    # ~5% faster op-level at M>=256 but did NOT improve E2E -> kept available, not default.
    if rw:  # register-word + autotuned
        y = torch.empty((M, OUT), dtype=torch.bfloat16, device=x.device)
        up32 = upper_cm.view(torch.int32).reshape(-1)          # [NB*OUT*5]
        sh = shared_cm.reshape(-1)
        SB = int(shared_cm.shape[-1])                          # 1 (gs16) or 2 (gs8), = stride per col
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(OUT, META['BLOCK_N']))
        _ssnf_gemm_rw_kernel[grid](
            x, scale_exp, up32, sh, y, M, OUT, K, NB, _mbucket(M),
            x.stride(0), x.stride(1), scale_exp.stride(0), scale_exp.stride(1), SB,
            y.stride(0), y.stride(1), GS_P=g_size, SB=SB)
        return y
    return _ssnf_gemm(x, scale_exp, upper_cm, shared_cm, M, OUT, K, NB, BM, BN)


# =============================================================================
#  NIBBLE kernel — u4/gs4 (attn q/k/v/o under SSNF_PROFILE=mixed). 4-bit upper packs
#  8 codes per uint32 with NO byte straddle, so the u3 straddle branch (strad /
#  sh_amt / wn / hi second where-chain) DISAPPEARS — unpack is one 4-way word
#  select + shift + mask. Tests whether removing that ALU makes attn u4 beat u3.
#  shared: gs4 -> 8 groups * 4b = one uint32/column.
# =============================================================================
@triton.autotune(configs=_rw_configs(), key=['OUT', 'K'])
@triton.jit
def _ssnf_gemm_nibble_kernel(
        x_ptr, sc_ptr, up32_ptr, sh_ptr, y_ptr,
        M, OUT, K, NB,
        stride_xm, stride_xk, stride_scb, stride_sco, stride_shc,
        stride_ym, stride_yo,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = rm < M; mask_n = rn < OUT
    k2 = rk[:, None]                       # [32,1]
    wi = k2 // 8                           # word index (8 nibbles / uint32) — NO straddle
    bo = (k2 % 8) * 4                      # bit offset within the word
    g = k2 // 4                            # gs4 group index (8 groups/block)
    for blk in range(0, NB):
        a_ptr = x_ptr + rm[:, None] * stride_xm + (blk * 32 + rk[None, :]) * stride_xk
        a = tl.load(a_ptr, mask=mask_m[:, None], other=0.0).to(tl.bfloat16)
        col = blk * OUT + rn                                   # [BN]
        se = tl.load(sc_ptr + blk * stride_scb + rn * stride_sco, mask=mask_n, other=0).to(tl.int32)
        scale = tl.exp2(se.to(tl.float32))                     # [BN]
        b4 = col * 4
        u0 = tl.load(up32_ptr + b4 + 0, mask=mask_n, other=0).to(tl.uint32)[None, :]
        u1 = tl.load(up32_ptr + b4 + 1, mask=mask_n, other=0).to(tl.uint32)[None, :]
        u2 = tl.load(up32_ptr + b4 + 2, mask=mask_n, other=0).to(tl.uint32)[None, :]
        u3 = tl.load(up32_ptr + b4 + 3, mask=mask_n, other=0).to(tl.uint32)[None, :]
        w = tl.where(wi == 0, u0, tl.where(wi == 1, u1, tl.where(wi == 2, u2, u3)))  # [32,BN]
        code = (w >> bo.to(tl.uint32)) & 0xF                   # nibble, no straddle merge
        up = (code.to(tl.int32) ^ 8) - 8                       # sign-extend 4b
        shb = tl.load(sh_ptr + col * stride_shc, mask=mask_n, other=0).to(tl.int32)[None, :]
        l_g_code = (shb >> (g * 4)) & 0xF
        sh = (l_g_code ^ 8) - 8                                 # sign-extend 4b
        word = up * 16 + sh
        b = word.to(tl.float32) * scale[None, :]
        acc += tl.dot(a, b.to(tl.bfloat16), out_dtype=tl.float32)
    y_ptr2 = y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yo
    tl.store(y_ptr2, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def ssnf_gemm_nibble(x: torch.Tensor, scale_exp: torch.Tensor, upper_cm: torch.Tensor,
                     shared_cm: torch.Tensor, M: int, OUT: int, K: int, NB: int) -> torch.Tensor:
    """u4/gs4 fused GEMM. upper_cm uint8 [NB,OUT,16]->int32 [NB,OUT,4]; shared_cm uint8
    [NB,OUT,4]->int32 [NB,OUT,1]. y = x @ W, W[k,o]=(up4*16+sh4)*2^scale_exp[k//32,o]."""
    y = torch.empty((M, OUT), dtype=torch.bfloat16, device=x.device)
    up32 = upper_cm.view(torch.int32).reshape(-1)          # [NB*OUT*4]
    sh32 = shared_cm.view(torch.int32).reshape(-1)         # [NB*OUT]
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(OUT, META['BLOCK_N']))
    _ssnf_gemm_nibble_kernel[grid](
        x, scale_exp, up32, sh32, y, M, OUT, K, NB,
        x.stride(0), x.stride(1), scale_exp.stride(0), scale_exp.stride(1), 1,
        y.stride(0), y.stride(1))
    return y


# SPLIT-K (v2, partials-based — no atomic): for block-starved shapes (o_proj 4096^2, down_proj)
# the (M/BM)x(OUT/BN) grid gives <1 wave/SM. Split K across grid.z -> more CTAs -> fill SMs. Each
# pid_k writes its OWN partials slice [pid_k,m,o] (no atomic, autotune-safe); torch sums over split.
# (v1 atomic version was buggy under autotune — this avoids it.)
@triton.autotune(configs=_rw_configs(), key=['OUT', 'K', 'MB'])   # MB: see _ssnf_gemm_rw_kernel
@triton.jit
def _ssnf_gemm_rw_splitk_kernel(
        x_ptr, sc_ptr, up32_ptr, sh_ptr, p_ptr, M, OUT, K, NB, MB,
        stride_xm, stride_xk, stride_scb, stride_sco, stride_shc, stride_pk, stride_pm, stride_po,
        SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        GS_P: tl.constexpr = 16, SB: tl.constexpr = 1):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1); pid_k = tl.program_id(2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = rm < M; mask_n = rn < OUT
    k2 = rk[:, None]; gb = k2 * WBITS; wi = gb // 32; bo = gb % 32
    strad = (bo + WBITS) > 32; sh_amt = tl.where(strad, 32 - bo, 1); g = k2 // GS_P
    blocks_per = tl.cdiv(NB, SPLIT_K)
    b0 = pid_k * blocks_per; b1 = tl.minimum(b0 + blocks_per, NB)
    for blk in range(b0, b1):
        a = tl.load(x_ptr + rm[:, None] * stride_xm + (blk * 32 + rk[None, :]) * stride_xk,
                    mask=mask_m[:, None], other=0.0).to(tl.bfloat16)
        col = blk * OUT + rn
        se = tl.load(sc_ptr + blk * stride_scb + rn * stride_sco, mask=mask_n, other=0).to(tl.int32)
        scale = tl.exp2(se.to(tl.float32))
        b5 = col * 5
        u0 = tl.load(up32_ptr + b5 + 0, mask=mask_n, other=0).to(tl.uint32)[None, :]
        u1 = tl.load(up32_ptr + b5 + 1, mask=mask_n, other=0).to(tl.uint32)[None, :]
        u2 = tl.load(up32_ptr + b5 + 2, mask=mask_n, other=0).to(tl.uint32)[None, :]
        u3 = tl.load(up32_ptr + b5 + 3, mask=mask_n, other=0).to(tl.uint32)[None, :]
        u4 = tl.load(up32_ptr + b5 + 4, mask=mask_n, other=0).to(tl.uint32)[None, :]
        w  = tl.where(wi == 0, u0, tl.where(wi == 1, u1, tl.where(wi == 2, u2, tl.where(wi == 3, u3, u4))))
        wn = tl.where(wi == 0, u1, tl.where(wi == 1, u2, tl.where(wi == 2, u3, u4)))
        hi = tl.where(strad, wn << sh_amt.to(tl.uint32), tl.zeros_like(w))
        code = ((w >> bo.to(tl.uint32)) | hi) & 0x1F
        up = (code.to(tl.int32) ^ 16) - 16
        shb = tl.load(sh_ptr + col * stride_shc, mask=mask_n, other=0).to(tl.int32)
        if SB == 2:
            sh1 = tl.load(sh_ptr + col * stride_shc + 1, mask=mask_n, other=0).to(tl.int32)
            shb = shb | (sh1 << 8)
        l_g_code = (shb[None, :] >> (g * U)) & 0x7
        sh = (l_g_code ^ 4) - 4
        word = up * (1 << U) + sh
        acc += tl.dot(a, (word.to(tl.float32) * scale[None, :]).to(tl.bfloat16), out_dtype=tl.float32)
    p = p_ptr + pid_k * stride_pk + rm[:, None] * stride_pm + rn[None, :] * stride_po
    tl.store(p, acc, mask=mask_m[:, None] & mask_n[None, :])


def ssnf_gemm_sk(x: torch.Tensor, scale_exp: torch.Tensor, upper_cm: torch.Tensor,
                 shared_cm: torch.Tensor, M: int, OUT: int, K: int, NB: int,
                 SPLIT_K: int = 4, g_size: int = 16) -> torch.Tensor:
    """Split-K fused GEMM (partials over grid.z, torch sum); y = x @ W in bf16."""
    part = torch.empty((SPLIT_K, M, OUT), dtype=torch.float32, device=x.device)
    up32 = upper_cm.view(torch.int32).reshape(-1)
    sh = shared_cm.reshape(-1)
    SB = int(shared_cm.shape[-1])
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(OUT, META['BLOCK_N']), SPLIT_K)
    _ssnf_gemm_rw_splitk_kernel[grid](
        x, scale_exp, up32, sh, part, M, OUT, K, NB, _mbucket(M),
        x.stride(0), x.stride(1), scale_exp.stride(0), scale_exp.stride(1), SB,
        part.stride(0), part.stride(1), part.stride(2), SPLIT_K=SPLIT_K, GS_P=g_size, SB=SB)
    return part.sum(0).to(torch.bfloat16)


_SPLIT_CACHE = {}      # (OUT, K, M-bucket, gs) -> measured best SPLIT_K
# Persisted so the load-time measurement runs once per machine, not once per engine boot
# (measuring 4 candidates x 6 buckets x 4 shapes costs ~4 min, mostly Triton compiles).
# SSNF_SPLIT_CACHE=<path> overrides; SSNF_NO_SPLIT_TUNE=1 skips tuning entirely (formula fallback).
_SPLIT_CACHE_PATH = os.environ.get(
    "SSNF_SPLIT_CACHE", os.path.expanduser("~/.cache/ssnf/split_cache.json"))


def _split_cache_load() -> None:
    if _SPLIT_CACHE:
        return
    try:
        import json
        with open(_SPLIT_CACHE_PATH) as f:
            for k, v in json.load(f).items():
                _SPLIT_CACHE[tuple(int(t) for t in k.split(","))] = int(v)
    except Exception:
        pass


def _split_cache_store() -> None:
    try:
        import json
        os.makedirs(os.path.dirname(_SPLIT_CACHE_PATH), exist_ok=True)
        with open(_SPLIT_CACHE_PATH, "w") as f:
            json.dump({",".join(str(t) for t in k): v for k, v in _SPLIT_CACHE.items()}, f)
    except Exception:
        pass


def _formula_split(M: int, OUT: int) -> int:
    """Original shape-based guess. Kept as the fallback and as the starting point."""
    import math
    ctas = math.ceil(M / 64) * math.ceil(OUT / 64)
    return max(1, min(4, round(512 / max(ctas, 1))))


def _measure_split(x: torch.Tensor, scale_exp: torch.Tensor, upper_cm: torch.Tensor,
                   shared_cm: torch.Tensor, M: int, OUT: int, K: int, NB: int,
                   g_size: int) -> int:
    """Time SPLIT_K in {1,2,3,4} once per (shape, M bucket) and cache the winner (the surface is
    non-monotonic, so it is measured, not modelled; quant.py warms every M bucket at load)."""
    import statistics
    cands = (1, 2, 3, 4)

    def make(c: int) -> Callable[[], torch.Tensor]:
        if c == 1:
            return lambda: ssnf_gemm(x, scale_exp, upper_cm, shared_cm, M, OUT, K, NB, g_size=g_size)
        return lambda: ssnf_gemm_sk(x, scale_exp, upper_cm, shared_cm, M, OUT, K, NB, SPLIT_K=c, g_size=g_size)

    calls, samples = {}, {c: [] for c in cands}
    for c in cands:                                   # compile + warm every candidate first
        try:
            f = make(c)
            for _ in range(3):
                f()
            calls[c] = f
        except Exception:
            pass
    if not calls:
        return _formula_split(M, OUT)
    torch.cuda.synchronize()
    for _ in range(3):                                # interleaved rounds -> clock drift cannot pick
        for c, f in calls.items():                    # the measured winner
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for _ in range(20):
                f()
            e1.record(); torch.cuda.synchronize()
            samples[c].append(e0.elapsed_time(e1))
    med = {c: statistics.median(v) for c, v in samples.items() if v}
    return min(med, key=med.get)


def ssnf_gemm_best(x: torch.Tensor, scale_exp: torch.Tensor, upper_cm: torch.Tensor,
                   shared_cm: torch.Tensor, M: int, OUT: int, K: int, NB: int,
                   g_size: int = 16) -> torch.Tensor:
    """Dispatch to the SPLIT_K that measures fastest for this (OUT, K, M bucket, gs), cached
    via _measure_split; SSNF_NO_SPLIT_TUNE=1 restores the per-call formula."""
    if os.getenv("SSNF_NO_SPLIT_TUNE", "0") != "0":
        # exact previous behaviour: the formula is evaluated per call with the true M
        # (NOT cached per bucket, which would round M up to the bucket and change the choice)
        split = _formula_split(M, OUT)
    else:
        key = (OUT, K, _mbucket(M), g_size)
        split = _SPLIT_CACHE.get(key)
        if split is None:
            _split_cache_load()
            split = _SPLIT_CACHE.get(key)
        if split is None:
            split = _measure_split(x, scale_exp, upper_cm, shared_cm, M, OUT, K, NB, g_size)
            _SPLIT_CACHE[key] = split
            _split_cache_store()
    if split <= 1:
        return ssnf_gemm(x, scale_exp, upper_cm, shared_cm, M, OUT, K, NB, g_size=g_size)
    return ssnf_gemm_sk(x, scale_exp, upper_cm, shared_cm, M, OUT, K, NB, SPLIT_K=split, g_size=g_size)


def _ssnf_gemm(x: torch.Tensor, scale_exp: torch.Tensor, upper_cm: torch.Tensor,
               shared_cm: torch.Tensor, M: int, OUT: int, K: int, NB: int,
               BM: int = 64, BN: int = 64) -> torch.Tensor:
    y = torch.empty((M, OUT), dtype=torch.bfloat16, device=x.device)
    up = upper_cm.reshape(-1)     # flat, col stride = UB
    sh = shared_cm.reshape(-1)    # flat, col stride = SB
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(OUT, META['BLOCK_N']))
    _ssnf_gemm_kernel[grid](
        x, scale_exp, up, sh, y,
        M, OUT, K, NB,
        x.stride(0), x.stride(1),
        scale_exp.stride(0), scale_exp.stride(1),
        UBLK_PY, 1,
        y.stride(0), y.stride(1))
    return y


if __name__ == "__main__":
    import numpy as np, sys
    from ssnf_vllm.pack_gpu import pack_weight_ssnf_gpu
    import ssnf_cuda  # for reference dequant
    dev = "cuda"
    # correctness vs the CUDA dense dequant reference
    for (OUT, K, M) in [(64, 128, 32), (256, 512, 64), (4096, 4096, 128), (4096, 14336, 256)]:
        Wf = torch.randn(OUT, K, device=dev) * 0.02
        ps = pack_weight_ssnf_gpu(Wf, 3, 16); nb = ps["nb"]
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.5
        Wdq = torch.ops.ssnf.wonly_dequant_bf16(ps["scale_exp"], ps["upper_cm"], ps["shared_cm"], OUT, K, nb, 3, 16)
        ref = x @ Wdq.t()
        y = ssnf_gemm(x, ps["scale_exp"], ps["upper_cm"], ps["shared_cm"], M, OUT, K, nb)
        yrw = ssnf_gemm(x, ps["scale_exp"], ps["upper_cm"], ps["shared_cm"], M, OUT, K, nb, rw=True)
        rel = float((y.float() - ref.float()).norm() / (ref.float().norm() + 1e-9))
        relrw = float((yrw.float() - ref.float()).norm() / (ref.float().norm() + 1e-9))
        print(f"OUT={OUT} K={K} M={M}: v1 rel={rel:.3e} {'OK' if rel<2e-2 else 'FAIL'}   rw rel={relrw:.3e} {'OK' if relrw<2e-2 else 'FAIL'}")


# --- diagnostic: same as rw but SKIP the bit-extraction (code = u0 & 0x1F) to isolate unpack cost
@triton.autotune(configs=_rw_configs(), key=['OUT', 'K'])
@triton.jit
def _ssnf_gemm_noext_kernel(x_ptr, sc_ptr, up32_ptr, sh_ptr, y_ptr, M, OUT, K, NB,
        stride_xm, stride_xk, stride_scb, stride_sco, stride_shc, stride_ym, stride_yo,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m*BLOCK_M+tl.arange(0,BLOCK_M); rn = pid_n*BLOCK_N+tl.arange(0,BLOCK_N); rk = tl.arange(0,32)
    acc = tl.zeros((BLOCK_M,BLOCK_N),dtype=tl.float32); mask_m=rm<M; mask_n=rn<OUT
    for blk in range(0,NB):
        a=tl.load(x_ptr+rm[:,None]*stride_xm+(blk*32+rk[None,:])*stride_xk,mask=mask_m[:,None],other=0.0).to(tl.bfloat16)
        col=blk*OUT+rn
        se=tl.load(sc_ptr+blk*stride_scb+rn*stride_sco,mask=mask_n,other=0).to(tl.int32); scale=tl.exp2(se.to(tl.float32))
        b5=col*5
        u0=tl.load(up32_ptr+b5+0,mask=mask_n,other=0).to(tl.uint32)[None,:]
        code=(u0 + rk[:,None].to(tl.uint32)) & 0x1F              # trivial: no where/straddle/shared
        b=code.to(tl.float32)*scale[None,:]
        acc+=tl.dot(a,b.to(tl.bfloat16),out_dtype=tl.float32)
    tl.store(y_ptr+rm[:,None]*stride_ym+rn[None,:]*stride_yo, acc.to(tl.bfloat16), mask=mask_m[:,None]&mask_n[None,:])

def ssnf_gemm_noext(x: torch.Tensor, scale_exp: torch.Tensor, upper_cm: torch.Tensor,
                    shared_cm: torch.Tensor, M: int, OUT: int, K: int, NB: int) -> torch.Tensor:
    """Diagnostic GEMM that skips the bit-extraction (isolates unpack cost); NOT numerically valid."""
    y=torch.empty((M,OUT),dtype=torch.bfloat16,device=x.device); up32=upper_cm.view(torch.int32).reshape(-1); sh=shared_cm.reshape(-1)
    grid=lambda META:(triton.cdiv(M,META['BLOCK_M']),triton.cdiv(OUT,META['BLOCK_N']))
    _ssnf_gemm_noext_kernel[grid](x,scale_exp,up32,sh,y,M,OUT,K,NB,x.stride(0),x.stride(1),scale_exp.stride(0),scale_exp.stride(1),1,y.stride(0),y.stride(1))
    return y


# --- user's idea: compile-time UNROLL (no where-chain, constant shifts, no runtime straddle test).
#     Assemble 32 per-code [BN] rows into [BN,32] via a bit-reversed tl.join tree, then trans -> [32,BN].
def _brev5(x: int) -> int: return int('{:05b}'.format(x)[::-1], 2)
_BREV5 = [_brev5(p) for p in range(32)]   # input[p] = code_{brev5(p)} -> join-tree output natural order

@triton.jit
def _ssnf_unroll_kernel(x_ptr, sc_ptr, up32_ptr, sh_ptr, y_ptr, M, OUT, K, NB,
        stride_xm, stride_xk, stride_scb, stride_sco, stride_shc, stride_ym, stride_yo,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m*BLOCK_M+tl.arange(0,BLOCK_M); rn = pid_n*BLOCK_N+tl.arange(0,BLOCK_N); rk = tl.arange(0,32)
    acc = tl.zeros((BLOCK_M,BLOCK_N),dtype=tl.float32); mask_m=rm<M; mask_n=rn<OUT
    for blk in range(0,NB):
        a=tl.load(x_ptr+rm[:,None]*stride_xm+(blk*32+rk[None,:])*stride_xk,mask=mask_m[:,None],other=0.0).to(tl.bfloat16)
        col=blk*OUT+rn
        se=tl.load(sc_ptr+blk*stride_scb+rn*stride_sco,mask=mask_n,other=0).to(tl.int32); scale=tl.exp2(se.to(tl.float32))
        b5=col*5
        u0=tl.load(up32_ptr+b5+0,mask=mask_n,other=0).to(tl.uint32)
        u1=tl.load(up32_ptr+b5+1,mask=mask_n,other=0).to(tl.uint32)
        u2=tl.load(up32_ptr+b5+2,mask=mask_n,other=0).to(tl.uint32)
        u3=tl.load(up32_ptr+b5+3,mask=mask_n,other=0).to(tl.uint32)
        u4=tl.load(up32_ptr+b5+4,mask=mask_n,other=0).to(tl.uint32)
        shb=tl.load(sh_ptr+col*stride_shc,mask=mask_n,other=0).to(tl.int32)   # [BN]
        words = (u0, u1, u2, u3, u4)
        rows = []
        for k in range(32):                                        # UNROLLED: wi/bo/straddle all python consts
            gb=k*5; wi=gb//32; bo=gb%32
            uw = u0 if wi==0 else (u1 if wi==1 else (u2 if wi==2 else (u3 if wi==3 else u4)))
            if bo+5>32:
                un = u1 if wi==0 else (u2 if wi==1 else (u3 if wi==2 else u4))
                upb = (uw>>bo) | (un<<(32-bo))
            else:
                upb = uw>>bo
            up=((upb & 0x1F).to(tl.int32) ^ 16) - 16
            g=k//16; sh=(((shb>>(g*3)) & 0x7) ^ 4) - 4
            rows.append((up*8+sh).to(tl.float32)*scale)            # [BN]
        # assemble [BN,32] via bit-reversed balanced join tree, then trans -> [32,BN]
        t = []
        for p in range(32): t.append(rows[_BREV5[p]])
        while len(t) > 1:
            nt = []
            for i in range(len(t)//2): nt.append(tl.reshape(tl.join(t[2*i], t[2*i+1]), (BLOCK_N, -1)))
            t = nt
        b = tl.trans(t[0])                                         # [32,BN]
        acc+=tl.dot(a,b.to(tl.bfloat16),out_dtype=tl.float32)
    tl.store(y_ptr+rm[:,None]*stride_ym+rn[None,:]*stride_yo, acc.to(tl.bfloat16), mask=mask_m[:,None]&mask_n[None,:])

def ssnf_gemm_unroll(x: torch.Tensor, scale_exp: torch.Tensor, upper_cm: torch.Tensor,
                     shared_cm: torch.Tensor, M: int, OUT: int, K: int, NB: int,
                     BM: int = 64, BN: int = 64) -> torch.Tensor:
    """Fused GEMM via the compile-time-unrolled join-tree kernel; y = x @ W in bf16."""
    y=torch.empty((M,OUT),dtype=torch.bfloat16,device=x.device); up32=upper_cm.view(torch.int32).reshape(-1); sh=shared_cm.reshape(-1)
    _ssnf_unroll_kernel[(triton.cdiv(M,BM),triton.cdiv(OUT,BN))](x,scale_exp,up32,sh,y,M,OUT,K,NB,
        x.stride(0),x.stride(1),scale_exp.stride(0),scale_exp.stride(1),1,y.stride(0),y.stride(1),BLOCK_M=BM,BLOCK_N=BN)
    return y


# --- register-aligned repack (offline): 32 5-bit upper codes -> 6 uint32 (6 codes/word, NO straddle).
#     +4 bytes/col (20->24, 5.44b->~6.5b). Kernel then extracts each code with 1 shift+mask, no wn/straddle.
def pack_upper_ra(upper_cm: torch.Tensor) -> torch.Tensor:   # [NB,OUT,20] uint8 -> [NB,OUT,6] int32
    NB, OUT, _ = upper_cm.shape
    l_b = (upper_cm.view(torch.int32).reshape(NB, OUT, 5).to(torch.int64)) & 0xffffffff  # 5 words unsigned
    codes = torch.zeros(NB, OUT, 32, dtype=torch.int64, device=upper_cm.device)
    for k in range(32):
        gb = k * 5; wi = gb // 32; bo = gb % 32
        w = l_b[..., wi] >> bo
        if bo + 5 > 32:
            w = w | (l_b[..., wi + 1] << (32 - bo))
        codes[..., k] = w & 0x1F
    ra = torch.zeros(NB, OUT, 6, dtype=torch.int64, device=upper_cm.device)
    for k in range(32):
        ra[..., k // 6] = ra[..., k // 6] | (codes[..., k] << ((k % 6) * 5))
    return (ra & 0xffffffff).to(torch.int32)


@triton.autotune(configs=_rw_configs(), key=['OUT', 'K'])
@triton.jit
def _ssnf_ra_kernel(x_ptr, sc_ptr, ra_ptr, sh_ptr, y_ptr, M, OUT, K, NB,
        stride_xm, stride_xk, stride_scb, stride_sco, stride_shc, stride_ym, stride_yo,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        GS_P: tl.constexpr = 16, SB: tl.constexpr = 1):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m*BLOCK_M+tl.arange(0,BLOCK_M); rn = pid_n*BLOCK_N+tl.arange(0,BLOCK_N); rk = tl.arange(0,32)
    acc = tl.zeros((BLOCK_M,BLOCK_N),dtype=tl.float32); mask_m=rm<M; mask_n=rn<OUT
    k2 = rk[:,None]; wi = k2//6; bit = (k2%6)*5; g = k2//GS_P     # NO straddle (register-aligned)
    for blk in range(0,NB):
        a=tl.load(x_ptr+rm[:,None]*stride_xm+(blk*32+rk[None,:])*stride_xk,mask=mask_m[:,None],other=0.0).to(tl.bfloat16)
        col=blk*OUT+rn
        se=tl.load(sc_ptr+blk*stride_scb+rn*stride_sco,mask=mask_n,other=0).to(tl.int32); scale=tl.exp2(se.to(tl.float32))
        b6=col*6
        r0=tl.load(ra_ptr+b6+0,mask=mask_n,other=0).to(tl.uint32)[None,:]
        r1=tl.load(ra_ptr+b6+1,mask=mask_n,other=0).to(tl.uint32)[None,:]
        r2=tl.load(ra_ptr+b6+2,mask=mask_n,other=0).to(tl.uint32)[None,:]
        r3=tl.load(ra_ptr+b6+3,mask=mask_n,other=0).to(tl.uint32)[None,:]
        r4=tl.load(ra_ptr+b6+4,mask=mask_n,other=0).to(tl.uint32)[None,:]
        r5=tl.load(ra_ptr+b6+5,mask=mask_n,other=0).to(tl.uint32)[None,:]
        w=tl.where(wi==0,r0,tl.where(wi==1,r1,tl.where(wi==2,r2,tl.where(wi==3,r3,tl.where(wi==4,r4,r5)))))
        code=(w >> bit.to(tl.uint32)) & 0x1F                     # single shift, no straddle
        up=(code.to(tl.int32) ^ 16) - 16
        shb=tl.load(sh_ptr+col*stride_shc,mask=mask_n,other=0).to(tl.int32)
        if SB == 2:
            sh1=tl.load(sh_ptr+col*stride_shc+1,mask=mask_n,other=0).to(tl.int32)
            shb=shb | (sh1 << 8)
        sh=(((shb[None,:] >> (g*U)) & 0x7) ^ 4) - 4
        b=((up*8+sh).to(tl.float32))*scale[None,:]
        acc+=tl.dot(a,b.to(tl.bfloat16),out_dtype=tl.float32)
    tl.store(y_ptr+rm[:,None]*stride_ym+rn[None,:]*stride_yo, acc.to(tl.bfloat16), mask=mask_m[:,None]&mask_n[None,:])

def ssnf_gemm_ra(x: torch.Tensor, scale_exp: torch.Tensor, upper_ra: torch.Tensor,
                 shared_cm: torch.Tensor, M: int, OUT: int, K: int, NB: int,
                 g_size: int = 16) -> torch.Tensor:
    """Fused GEMM on the register-aligned upper plane (no straddle); y = x @ W in bf16."""
    y=torch.empty((M,OUT),dtype=torch.bfloat16,device=x.device); ra=upper_ra.reshape(-1); sh=shared_cm.reshape(-1)
    SB=int(shared_cm.shape[-1])
    grid=lambda META:(triton.cdiv(M,META['BLOCK_M']),triton.cdiv(OUT,META['BLOCK_N']))
    _ssnf_ra_kernel[grid](x,scale_exp,ra,sh,y,M,OUT,K,NB,x.stride(0),x.stride(1),scale_exp.stride(0),scale_exp.stride(1),SB,y.stride(0),y.stride(1),GS_P=g_size,SB=SB)
    return y


# --- W+A int8: activation int8 (from quant_act) x SSNF weight (unpack->int8) via int8 IMMA (tl.dot int8->int32)
#     + per-K-block scale (sa_row x scale_w_col) in fp32 epilogue. Tests if activation-quant -> int8 tensorcore
#     beats weight-only (bf16 activation). weight u3/gs16 (rw unpack).
@triton.autotune(configs=_rw_configs(), key=['OUT', 'K'])
@triton.jit
def _wa_int8_kernel(qx_ptr, sa_ptr, sc_ptr, up32_ptr, sh_ptr, y_ptr, M, OUT, K, NB,
        stride_qxm, stride_qxk, stride_sam, stride_sab, stride_scb, stride_sco, stride_shc,
        stride_ym, stride_yo, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m=tl.program_id(0); pid_n=tl.program_id(1)
    rm=pid_m*BLOCK_M+tl.arange(0,BLOCK_M); rn=pid_n*BLOCK_N+tl.arange(0,BLOCK_N); rk=tl.arange(0,32)
    acc=tl.zeros((BLOCK_M,BLOCK_N),dtype=tl.float32); mask_m=rm<M; mask_n=rn<OUT
    k2=rk[:,None]; gb=k2*WBITS; wi=gb//32; bo=gb%32; strad=(bo+WBITS)>32; sh_amt=tl.where(strad,32-bo,1); g=k2//GS
    for blk in range(0,NB):
        a=tl.load(qx_ptr+rm[:,None]*stride_qxm+(blk*32+rk[None,:])*stride_qxk,mask=mask_m[:,None],other=0).to(tl.int8)  # [BM,32]
        sa=tl.exp2(tl.load(sa_ptr+rm*stride_sam+blk*stride_sab,mask=mask_m,other=0).to(tl.float32))                    # [BM]
        col=blk*OUT+rn
        scw=tl.exp2(tl.load(sc_ptr+blk*stride_scb+rn*stride_sco,mask=mask_n,other=0).to(tl.float32))                  # [BN]
        b5=col*5
        u0=tl.load(up32_ptr+b5+0,mask=mask_n,other=0).to(tl.uint32)[None,:]; u1=tl.load(up32_ptr+b5+1,mask=mask_n,other=0).to(tl.uint32)[None,:]
        u2=tl.load(up32_ptr+b5+2,mask=mask_n,other=0).to(tl.uint32)[None,:]; u3=tl.load(up32_ptr+b5+3,mask=mask_n,other=0).to(tl.uint32)[None,:]
        u4=tl.load(up32_ptr+b5+4,mask=mask_n,other=0).to(tl.uint32)[None,:]
        w=tl.where(wi==0,u0,tl.where(wi==1,u1,tl.where(wi==2,u2,tl.where(wi==3,u3,u4))))
        wn=tl.where(wi==0,u1,tl.where(wi==1,u2,tl.where(wi==2,u3,u4)))
        hi=tl.where(strad,wn<<sh_amt.to(tl.uint32),tl.zeros_like(w))
        code=((w>>bo.to(tl.uint32))|hi)&0x1F; up=(code.to(tl.int32)^16)-16
        shb=tl.load(sh_ptr+col*stride_shc,mask=mask_n,other=0).to(tl.int32)
        sh=(((shb[None,:]>>(g*U))&0x7)^4)-4
        word=(up*(1<<U)+sh)                                          # int, SSNF weight int8 word
        word_i8=tl.minimum(tl.maximum(word,-128),127).to(tl.int8)   # [32,BN]
        acc32=tl.dot(a, word_i8, out_dtype=tl.int32)                # int8 IMMA -> int32 [BM,BN]
        acc+=acc32.to(tl.float32)*sa[:,None]*scw[None,:]            # per-block scale (row x col)
    tl.store(y_ptr+rm[:,None]*stride_ym+rn[None,:]*stride_yo, acc.to(tl.bfloat16), mask=mask_m[:,None]&mask_n[None,:])

def wa_int8_gemm(qx: torch.Tensor, sa: torch.Tensor, scale_exp: torch.Tensor,
                 upper_cm: torch.Tensor, shared_cm: torch.Tensor, M: int, OUT: int,
                 K: int, NB: int) -> torch.Tensor:
    """W+A GEMM: int8 activation x SSNF weight via int8 IMMA with per-K-block scales."""
    y=torch.empty((M,OUT),dtype=torch.bfloat16,device=qx.device); up32=upper_cm.view(torch.int32).reshape(-1); sh=shared_cm.reshape(-1)
    grid=lambda META:(triton.cdiv(M,META['BLOCK_M']),triton.cdiv(OUT,META['BLOCK_N']))
    _wa_int8_kernel[grid](qx,sa,scale_exp,up32,sh,y,M,OUT,K,NB,qx.stride(0),qx.stride(1),sa.stride(0),sa.stride(1),
        scale_exp.stride(0),scale_exp.stride(1),1,y.stride(0),y.stride(1))
    return y
