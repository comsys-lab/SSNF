"""Baseline block-scaled quantizers used by the comparison drivers (F7 table, zero-shot, QSNR).
`mxfp_quant` = plain OCP MXFP (per-32 E8M0 floor scale + FP(eb,mb) element, no sharing);
`ssnf_mxint8`/`ssnf_mxint8_efb` = the INT-element SSNF reference encoders. Numerics ported
verbatim from the original comparison codec so re-runs match the shipped F7 logs bit-for-bit.
"""
from __future__ import annotations

import torch

BLOCK = 32


def _fmt_params(eb: int, mb: int) -> tuple[int, int, float]:
    """Return (emin, maxexp, maxval) of the FP(eb, mb) format."""
    bias = 2 ** (eb - 1) - 1
    emin = 1 - bias
    maxexp = (2 ** eb - 1) - bias
    maxval = (2 - 2.0 ** (-mb)) * (2.0 ** maxexp)
    return emin, maxexp, maxval


def _fp_round(y: torch.Tensor, e: torch.Tensor, step: torch.Tensor, maxval: float) -> torch.Tensor:
    """Round y to the FP grid at per-element exponent e (quantum=step), clamp to +-maxval."""
    q = torch.round(y / step) * step
    return q.clamp(-maxval, maxval)


def mxfp_quant(x: torch.Tensor, eb: int, mb: int, scale_log2_offset: float = 0.0,
               block: int = BLOCK) -> torch.Tensor:
    """Plain OCP MXFP: per-`block` E8M0 floor scale aligned to maxexp + E{eb}M{mb} element."""
    emin, maxexp, maxval = _fmt_params(eb, mb)
    xf = x.reshape(-1, block).to(torch.float32)
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    s_base = torch.exp2(torch.floor(torch.log2(absmax)) - float(maxexp) + scale_log2_offset)
    y = xf / s_base
    e = torch.floor(torch.log2(y.abs().clamp(min=1e-30))).clamp(min=float(emin), max=float(maxexp))
    upper = _fp_round(y, e, torch.exp2(e - float(mb)), maxval)
    return (upper * s_base).reshape(x.shape)


def ssnf_mxint8(x: torch.Tensor, l_b: int, mg: int) -> torch.Tensor:
    """Reference deployed MXINT8-SSNF: INT8 element, upper (8-l_b)-bit + shared l_b-bit over mg, E8M0 block scale."""
    xf = x.reshape(-1, BLOCK).to(torch.float32)
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    if l_b == 0:                                              # plain MXINT8 (no sharing)
        s = torch.exp2(torch.ceil(torch.log2(absmax / 127.0)))
        return (torch.round(xf / s).clamp(-127, 127) * s).reshape(x.shape)
    q_max = (1 << (7 - l_b)) - 1
    s_base = torch.exp2(torch.floor(torch.log2(absmax / 64.0)))
    s_un = s_base * float(1 << l_b)
    q_un = torch.round(xf / s_un).clamp(-q_max, q_max)
    res = xf - q_un * s_un
    s_min, s_max = -(1 << (l_b - 1)), (1 << (l_b - 1)) - 1
    res_avg = res.reshape(res.shape[0], -1, mg).mean(-1, keepdim=True).expand(-1, -1, mg).reshape(res.shape)
    shared = torch.round(res_avg / s_base).clamp(s_min, s_max)
    return (q_un * s_un + shared * s_base).reshape(x.shape)


def ssnf_mxint8_efb(x: torch.Tensor, l_b: int, mg: int, efb_iters: int = 2) -> torch.Tensor:
    """MXINT8-SSNF with error-feedback coordinate descent (re-rounds the upper code after `shared` is fixed; efb_iters=0 reduces to ssnf_mxint8, same stored format)."""
    xf = x.reshape(-1, BLOCK).to(torch.float32)
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    q_max = (1 << (7 - l_b)) - 1
    s_base = torch.exp2(torch.floor(torch.log2(absmax / 64.0)))
    s_un = s_base * float(1 << l_b)
    s_min, s_max = -(1 << (l_b - 1)), (1 << (l_b - 1)) - 1
    q_un = torch.round(xf / s_un).clamp(-q_max, q_max)
    shared = None
    for it in range(max(1, efb_iters + 1)):
        res = xf - q_un * s_un
        shared = torch.round(res.reshape(res.shape[0], -1, mg).mean(-1, keepdim=True)
                             .expand(-1, -1, mg).reshape(res.shape) / s_base).clamp(s_min, s_max)
        if it == efb_iters:
            break
        q_un = torch.round((xf - shared * s_base) / s_un).clamp(-q_max, q_max)   # efb: re-round upper
    return (q_un * s_un + shared * s_base).reshape(x.shape)


def bits_mxint8(l_b: int, mg: int) -> float:
    """Effective bits/element of MXINT8-SSNF: (8-l_b) upper + l_b/mg shared + E8M0(8)/block."""
    return (8 - l_b) + l_b / mg + 8.0 / BLOCK
