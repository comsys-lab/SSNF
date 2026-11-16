# Fake-quant the KV cache (per-32 along head_dim) via forward_pre_hooks on vLLM Attention modules.
# Notation: `l_b` = shared field bitwidth, `g_size` = mantissa group size.
# FULL-quant: K/V quantized for EVERY attention (prefill AND decode) — same treatment as the PPL path.
# Formats: mxfp4/nvfp4/mxfp8/mxfp6/mxint8; ssnf/ssnf_kv (K H-weighted + V uniform). bf16 store, no CUDA.
import os
from typing import TYPE_CHECKING

import torch
from ssnf_vllm.fp4_quant import mxfp4, nvfp4, _round_fp

if TYPE_CHECKING:
    from vllm import LLM


def _mxint8(x: torch.Tensor, n: int = 8) -> torch.Tensor:
    """OCP MXINT-n: per-32 E8M0 floor scale 2^(floor(log2 amax)-(n-2)) + signed n-bit element
    (unified on the OCP floor scale; the older ceil scale disagreed with measure.py::mxint_n
    in 1.25% of blocks, QSNR gap 0.046 dB at n=8)."""
    xf = x.reshape(-1, 32).float()
    am = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    qmax = (1 << (n - 1)) - 1
    s = torch.exp2(torch.floor(torch.log2(am)) - float(n - 2))
    return (torch.round(xf / s).clamp(-qmax, qmax) * s).reshape(x.shape).to(x.dtype)


def _ssnf_hw(x: torch.Tensor, l_b: int, g_size: int,
             Hfull: torch.Tensor | None = None) -> torch.Tensor:
    """INT-SSNF (E8M0 + upper + H-weighted shared group-mean, radius=0). Mirrors eval_accuracy/wkv_v2_ppl.ssnf_hw."""
    shp = x.shape
    xf = x.reshape(-1, 32).float(); B = xf.shape[0]
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    s_base = torch.exp2(torch.floor(torch.log2(absmax)) - 6.0)
    s_unshared = s_base * float(1 << l_b)
    q_max = (1 << (7 - l_b)) - 1
    s_min, s_max = -(1 << (l_b - 1)), (1 << (l_b - 1)) - 1
    n_group = 32 // g_size
    Hb = torch.ones_like(xf) if Hfull is None else Hfull.reshape(-1, 32).float()
    h_word = torch.round(xf / s_unshared).clamp(-q_max, q_max)
    residual = (xf - h_word * s_unshared).reshape(B, n_group, g_size)
    Hg = Hb.reshape(B, n_group, g_size)
    res_avg = (Hg * residual).sum(2) / Hg.sum(2).clamp(min=1e-30)
    l_g_word = torch.round(res_avg / s_base).clamp(s_min, s_max)
    return (h_word * s_unshared + l_g_word.repeat_interleave(g_size, dim=1) * s_base).reshape(shp).to(x.dtype)


def _mxfp(x: torch.Tensor, eb: int, mb: int, block: int = 32) -> torch.Tensor:
    """per-`block` E8M0 shared scale + E{eb}M{mb} element (MXFP8=E4M3, MXFP6=E2M3). Mirrors fp4_quant.mxfp4."""
    shp = x.shape
    xf = x.reshape(-1, block).float()
    am = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    maxexp = (2 ** eb - 1) - (2 ** (eb - 1) - 1)                # E8M0 aligns block amax to the format emax
    X = torch.exp2(torch.floor(torch.log2(am)) - float(maxexp))
    return (_round_fp(xf / X, eb, mb) * X).reshape(shp).to(x.dtype)


# mxfp8/mxfp6 keep their historical names for the E4M3/E2M3 variants used in the paper; the
# *_e5m2 / *_e3m2 entries are the OCP alternates at the same element width (not in the paper tables).
_SIMPLE = {"mxfp4": mxfp4, "nvfp4": nvfp4, "mxint8": _mxint8,
           "mxfp8": lambda x: _mxfp(x, 4, 3), "mxfp6": lambda x: _mxfp(x, 2, 3),
           "mxfp8_e5m2": lambda x: _mxfp(x, 5, 2), "mxfp6_e3m2": lambda x: _mxfp(x, 3, 2)}

# SSNF mantissa-group sizes for the KV planes. Defaults reproduce the earlier campaign
# (K u3/gs32 = 5.34b, V u4/gs32 = 4.38b); the paper's deployed config is K gs16 (5.44b), so the
# task runs pass SSNF_K_GS=16. Defaults are left alone so old results stay reproducible.
_K_GS = int(os.environ.get("SSNF_K_GS", "32"))
_V_GS = int(os.environ.get("SSNF_V_GS", "32"))
# shared-field bitwidth (paper l_b). Deployed: K 3, V 4. Exposed so the configuration-search grid can be
# swept on the task benchmarks too, not just on PPL.
_K_LB = int(os.environ.get("SSNF_K_LB", "3"))
_V_LB = int(os.environ.get("SSNF_V_LB", "4"))


def _install(model: torch.nn.Module, fmt: str, hk: torch.Tensor | None = None) -> int:
    """Register the KV fake-quant forward_pre_hook on every Attention module; returns the count."""
    from vllm.model_executor.layers.attention.attention import Attention
    attns = [m for m in model.modules() if isinstance(m, Attention)]

    if fmt in _SIMPLE:
        q = _SIMPLE[fmt]

        def hook(module: torch.nn.Module, args: tuple) -> tuple:
            query, key, value = args[0], args[1], args[2]
            if key is not None and key.numel() % 32 == 0:
                key, value = q(key.float()).to(key.dtype), q(value.float()).to(value.dtype)
            return (query, key, value) + tuple(args[3:])
        for m in attns:
            m.register_forward_pre_hook(hook)
        return len(attns)

    if fmt in ("ssnf", "ssnf_kv"):                     # K = act-weighted u3/gs32, V = uniform u4/gs32
        assert hk is not None and hk.shape[0] == len(attns), f"H_K layers {hk.shape} != attn {len(attns)}"
        for l, m in enumerate(attns):
            Hl = hk[l].to(model.device if hasattr(model, "device") else "cuda")   # [Hkv, D]

            def mk(Hl: torch.Tensor):
                def hook(module: torch.nn.Module, args: tuple) -> tuple:
                    query, key, value = args[0], args[1], args[2]
                    T = key.shape[0]; Hkv, D = Hl.shape
                    Hf = Hl.reshape(1, Hkv, D).expand(T, Hkv, D)
                    k = _ssnf_hw(key.float().reshape(T, Hkv, D), _K_LB, _K_GS, Hf).reshape(key.shape).to(key.dtype)
                    v = _ssnf_hw(value.float().reshape(T, Hkv, D), _V_LB, _V_GS, None).reshape(value.shape).to(value.dtype)
                    return (query, k, v) + tuple(args[3:])
                return hook
            m.register_forward_pre_hook(mk(Hl))
        return len(attns)

    raise ValueError(f"unknown KV fake-quant fmt {fmt}")


def install(llm: "LLM", fmt: str, hk_path: str | None = None) -> int:
    """Install full-quant KV hooks in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0). ssnf needs hk_path=H_*.pt."""
    hk = None
    if fmt in ("ssnf", "ssnf_kv"):
        hk = torch.load(hk_path, map_location="cuda")["H_K"].float()      # [L, Hkv, D]
    return llm.apply_model(lambda model: _install(model, fmt, hk))


def _install_w(model: torch.nn.Module, fmt: str) -> int:
    """Quantize every transformer linear weight in place, leaving the tensor in BF16."""
    from vllm.model_executor.layers.linear import LinearBase
    fn = _SIMPLE[fmt]
    n = 0
    with torch.no_grad():
        for _, m in model.named_modules():
            if isinstance(m, LinearBase) and getattr(m, "weight", None) is not None:
                m.weight.data.copy_(fn(m.weight.data.float()).to(m.weight.dtype))
                n += 1
    return n


def install_weight(llm: "LLM", fmt: str) -> int:
    """Weight-side fake-quant for element codecs the packed CUDA path does not implement
    (packers hardcode E4M3/E2M3, so E5M2/E3M2 ride this numerically-equivalent bf16 path;
    verify the equivalence on E4M3/E2M3 first). lm_head stays BF16 — the paper's quant scope."""
    return llm.apply_model(lambda model: _install_w(model, fmt))
