# ssnf_lib/reference.py — REFERENCE ONLY, not on the deployed path; no reported number uses it.
# The verification ORACLE: pure NumPy ground-truth results every CUDA/CUTLASS kernel in kernel/
# must reproduce (wonly/wa matmuls, quant_act, kv_attention*, plus _ssnf_signed_ref as an
# independent cross-check). Layering: reference.py -> pack.py (one direction; no cycle).

import math
import numpy as np

from ssnf_lib.pack import (
    BLOCK, E_MAX,
    _e8m0_scale, decompose,
    dequant_weight, weight_int8, dequant_weight_nores,
    dequant_weight_mxint8, weight_int8_mxint8,
)


# =============================================================================
#  MATMUL / ATTENTION ORACLES  (certified transcription targets for the kernels)
# =============================================================================
def wonly_matmul(p: dict, X: np.ndarray) -> np.ndarray:
    """W-only: X [M,K] -> Y [M,OUT].  Y = X @ dequant(W)^T.  (M=1 == GEMV.)"""
    W_dq = dequant_weight(p)                                       # [OUT, K]
    return np.asarray(X, np.float64) @ W_dq.T


def quant_act(X: np.ndarray, l_b: int | None = None, g_size: int | None = None,
              share: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Runtime activation quant. X [M,K] -> (qX int [M,nb,32], scale_a [M,nb]).
    share=False -> MXINT8 (kernel default); share=True -> SSNF-s (accuracy match)."""
    M, K = X.shape
    nb = K // BLOCK
    xb = np.asarray(X, np.float64).reshape(M, nb, BLOCK)
    if not share:
        s = _e8m0_scale(np.abs(xb).max(axis=2, keepdims=True))     # [M,nb,1]
        q = np.clip(np.round(xb / s), -127, 127).astype(np.int64)
        return q, s[..., 0]
    s2, qu, rs = decompose(xb.reshape(M * nb, BLOCK), l_b, g_size)
    r_exp = np.repeat(rs, g_size, axis=1)
    qfull = (qu * (1 << l_b) + r_exp).reshape(M, nb, BLOCK)
    return qfull, s2.reshape(M, nb)


def wa_matmul(p: dict, X: np.ndarray, share_act: bool = False) -> np.ndarray:
    """W+A: X [M,K] -> Y [M,OUT].  Per block: (scale_a*scale_w) * int8-dot."""
    qW, sW = weight_int8(p)                                        # [OUT,nb,32],[OUT,nb]
    qX, sX = quant_act(X, p["l_b"], p["g_size"], share=share_act)        # [M,nb,32],[M,nb]
    intdot = np.einsum("mbk,obk->mbo", qX.astype(np.int64), qW.astype(np.int64))
    return np.einsum("mbo,mb,ob->mo", intdot.astype(np.float64), sX, sW)


def kv_attention(Q: np.ndarray, pK: dict, pV: dict, causal: bool = True) -> np.ndarray:
    """Fused-dequant attention mirror. Q [H,Lq,D]; pK,pV from pack_kv.
    scores = Q @ Kdq^T / sqrt(D) -> (causal) softmax -> @ Vdq.  Returns [H,Lq,D]."""
    H, Lq, D = Q.shape
    out = np.zeros((H, Lq, D), dtype=np.float64)
    for h in range(H):
        Kdq = dequant_weight(pK["_per"][h])                        # [Lk, D]
        Vdq = dequant_weight(pV["_per"][h])                        # [Lk, D]
        Lk = Kdq.shape[0]
        scores = (np.asarray(Q[h], np.float64) @ Kdq.T) / math.sqrt(D)   # [Lq,Lk]
        if causal:                                                  # query t sees keys <= t
            m = np.triu(np.ones((Lq, Lk)), k=1 + (Lk - Lq))
            scores = np.where(m > 0, -np.inf, scores)
        scores = scores - scores.max(axis=1, keepdims=True)
        pmat = np.exp(scores)
        pmat /= pmat.sum(axis=1, keepdims=True)
        out[h] = pmat @ Vdq
    return out


# =============================================================================
#  MXINT8 BASELINE ORACLES  (same math as above, weight in plain MXINT8)
# =============================================================================
def wonly_matmul_mxint8(p: dict, X: np.ndarray) -> np.ndarray:
    """W-only MXINT8: X [M,K] -> Y [M,OUT] = X @ dequant_mxint8(W)^T."""
    return np.asarray(X, np.float64) @ dequant_weight_mxint8(p).T


def wa_matmul_mxint8(p: dict, X: np.ndarray) -> np.ndarray:
    """W+A MXINT8: int8 weight (no sharing) x MXINT8 activation, per-block dot."""
    qW, sW = weight_int8_mxint8(p)                                # [OUT,nb,32],[OUT,nb]
    qX, sX = quant_act(X, share=False)                            # [M,nb,32],[M,nb]
    intdot = np.einsum("mbk,obk->mbo", qX.astype(np.int64), qW.astype(np.int64))
    return np.einsum("mbo,mb,ob->mo", intdot.astype(np.float64), sX, sW)


def kv_attention_nores(Q: np.ndarray, pK: dict, pV: dict, causal: bool = True) -> np.ndarray:
    """Attention with K/V in the no-residual format (pack_kv_nores); mirror of kv_attention."""
    H, Lq, D = Q.shape
    out = np.zeros((H, Lq, D), dtype=np.float64)
    for h in range(H):
        Kdq = dequant_weight_nores(pK["_per"][h])
        Vdq = dequant_weight_nores(pV["_per"][h])
        Lk = Kdq.shape[0]
        scores = (np.asarray(Q[h], np.float64) @ Kdq.T) / math.sqrt(D)
        if causal:
            m = np.triu(np.ones((Lq, Lk)), k=1 + (Lk - Lq))
            scores = np.where(m > 0, -np.inf, scores)
        scores = scores - scores.max(axis=1, keepdims=True)
        pmat = np.exp(scores)
        pmat /= pmat.sum(axis=1, keepdims=True)
        out[h] = pmat @ Vdq
    return out


def kv_attention_mxint8(Q: np.ndarray, pK: dict, pV: dict, causal: bool = True) -> np.ndarray:
    """Attention with K/V stored in plain MXINT8 (mirror of kv_attention)."""
    H, Lq, D = Q.shape
    out = np.zeros((H, Lq, D), dtype=np.float64)
    for h in range(H):
        Kdq = dequant_weight_mxint8(pK["_per"][h])
        Vdq = dequant_weight_mxint8(pV["_per"][h])
        Lk = Kdq.shape[0]
        scores = (np.asarray(Q[h], np.float64) @ Kdq.T) / math.sqrt(D)
        if causal:
            m = np.triu(np.ones((Lq, Lk)), k=1 + (Lk - Lq))
            scores = np.where(m > 0, -np.inf, scores)
        scores = scores - scores.max(axis=1, keepdims=True)
        pmat = np.exp(scores)
        pmat /= pmat.sum(axis=1, keepdims=True)
        out[h] = pmat @ Vdq
    return out


def _ssnf_signed_ref(x_blocked: np.ndarray, l_b: int, g_size: int) -> np.ndarray:
    """Independent port of SSNF_signed (cross-checks decompose+reconstruct)."""
    xf = np.asarray(x_blocked, np.float64)
    B = xf.shape[0]
    amax = np.maximum(np.abs(xf).max(1, keepdims=True), 1e-30)
    s_base = 2.0 ** (np.floor(np.log2(amax)) - E_MAX)
    s_unshared = s_base * (1 << l_b)
    qmax = (1 << (7 - l_b)) - 1
    qun = np.clip(np.round(xf / s_unshared), -qmax, qmax)
    xun = qun * s_unshared
    res = xf - xun
    n_group = BLOCK // g_size
    ravg = res.reshape(B, n_group, g_size).mean(2)
    smin, smax = -(1 << (l_b - 1)), (1 << (l_b - 1)) - 1
    l_g_word = np.clip(np.round(ravg / s_base), smin, smax)
    return xun + np.repeat(l_g_word, g_size, 1) * s_base
