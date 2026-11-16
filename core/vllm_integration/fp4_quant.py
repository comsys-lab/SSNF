"""Self-contained MXFP4 / NVFP4 fake-quant (torch only, no CUDA) for vLLM accuracy foils.

MXFP4-E2M1: per-32 E8M0 shared scale + E2M1 element (4.25 b/elem); NVFP4: per-16 E4M3
micro-scale + per-tensor FP32 + E2M1 element (4.5 b/elem). Quantize->dequantize to bf16.
Verified SQNR-identical to the accuracy-side codecs (comparison.mxfp(2,1) / nvfp4).
"""
import torch


def _round_fp(y: torch.Tensor, eb: int, mb: int) -> torch.Tensor:
    """round y (already scaled) to the E{eb}M{mb} grid, clamp to +-maxval."""
    bias = 2 ** (eb - 1) - 1
    emin, maxexp = 1 - bias, (2 ** eb - 1) - bias
    maxval = (2 - 2.0 ** (-mb)) * (2.0 ** maxexp)
    e = torch.floor(torch.log2(y.abs().clamp(min=1e-30))).clamp(min=float(emin), max=float(maxexp))
    step = torch.exp2(e - float(mb))
    return (torch.round(y / step) * step).clamp(-maxval, maxval)


def mxfp4(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """MXFP4-E2M1: per-`block` E8M0 shared scale + E2M1 element. Matches comparison.mxfp(x,2,1)."""
    shp = x.shape
    xf = x.reshape(-1, block).float()
    am = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    X = torch.exp2(torch.floor(torch.log2(am)) - 2.0)          # E8M0 shared scale (E2M1 emax = 2)
    return (_round_fp(xf / X, 2, 1) * X).reshape(shp).to(x.dtype)


def nvfp4(x: torch.Tensor) -> torch.Tensor:
    """NVFP4: E2M1 element + per-16 E4M3 micro-scale + per-tensor FP32 scale."""
    shp = x.shape
    xf = x.reshape(-1, 16).float()
    s_tensor = (xf.abs().amax().clamp(min=1e-30) / (448.0 * 6.0)).clamp(min=1e-30)   # per-tensor FP32
    bamax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    micro = _round_fp(bamax / 6.0 / s_tensor, 4, 3).clamp(min=1e-30)                 # per-16 E4M3
    eff = micro * s_tensor
    return (_round_fp(xf / eff, 2, 1) * eff).reshape(shp).to(x.dtype)


QUANTIZERS = {"mxfp4": mxfp4, "nvfp4": nvfp4}
