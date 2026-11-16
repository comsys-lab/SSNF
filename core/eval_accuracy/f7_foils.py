"""F7 comparison-format PPL (MXFP8/MXFP6/MXFP4/NVFP4), W+K+V all quantized, in the SAME 20-window harness as measure.py.
MXFP = OCP floor scale (comparison.mxfp); NVFP4 = 16-element E2M1 + FP8(E4M3) per-block scale + tensor-level FP32 scale.
Run: SSNF_MODEL=<id> CUDA_VISIBLE_DEVICES=<g> python eval_accuracy/f7_foils.py
"""
from __future__ import annotations
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling modules (wkv_v2_ppl, comparison)
from wkv_v2_ppl import ppl, _S, LINEAR_KEYS, MW
from comparison import mxfp   # native MXFP (OCP floor scale)

DEV = "cuda"


def _round_fp(y: torch.Tensor, eb: int, mb: int) -> torch.Tensor:
    """Round to the FP(eb, mb) grid (per-element exponent) with saturation."""
    bias = 2 ** (eb - 1) - 1; emin = 1 - bias; maxexp = (2 ** eb - 1) - bias
    maxval = (2 - 2.0 ** (-mb)) * (2.0 ** maxexp)
    e = torch.floor(torch.log2(y.abs().clamp(min=1e-30))).clamp(min=float(emin), max=float(maxexp))
    step = torch.exp2(e - float(mb))
    return (torch.round(y / step) * step).clamp(-maxval, maxval)


def _nvfp4(x: torch.Tensor) -> torch.Tensor:   # NVFP4: 16-elem block, E2M1 element, FP8(E4M3) block scale, FP32 tensor scale
    shp = x.shape; xf = x.reshape(-1, 16).float()
    st = (xf.abs().amax().clamp(min=1e-30) / (448.0 * 6.0)).clamp(min=1e-30)
    mi = _round_fp(xf.abs().amax(-1, keepdim=True).clamp(min=1e-30) / 6.0 / st, 4, 3).clamp(min=1e-30)
    eff = mi * st
    return (_round_fp(xf / eff, 2, 1) * eff).reshape(shp)


# E5M2 / E3M2 are the OCP MX alternates at the SAME element width as E4M3 / E2M3 (iso-BPE 8.25 /
# 6.25): they trade mantissa bits for exponent range. Measuring them substantiates the paper
# claim that the chosen variants "allocate more element bits to the mantissa than alternative
# formats of the same width". Not in the paper tables.
QFN = {
    "MXFP8-E4M3": lambda x: mxfp(x, 4, 3),
    "MXFP8-E5M2": lambda x: mxfp(x, 5, 2),
    "MXFP6-E2M3": lambda x: mxfp(x, 2, 3),
    "MXFP6-E3M2": lambda x: mxfp(x, 3, 2),
    "MXFP4-E2M1": lambda x: mxfp(x, 2, 1),
    "NVFP4":      _nvfp4,
}
BPE = {"MXFP8-E4M3": 8.25, "MXFP8-E5M2": 8.25, "MXFP6-E2M3": 6.25, "MXFP6-E3M2": 6.25,
       "MXFP4-E2M1": 4.25, "NVFP4": 4.50}
RUN = os.environ.get("F7_FORMATS", ",".join(QFN)).split(",")


def main() -> None:
    """Measure BF16 and per-format W+K+V PPL for every format in F7_FORMATS."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    MODEL = os.environ["SSNF_MODEL"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()

    _real = F.scaled_dot_product_attention
    def _kv(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fn = QFN[_S["mode"]]; return fn(k.float()).to(k.dtype), fn(v.float()).to(v.dtype)
    def _patch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *a, **kw) -> torch.Tensor:
        if _S["on"]: k, v = _kv(k, v)
        return _real(q, k, v, *a, **kw)
    F.scaled_dot_product_attention = _patch

    try: _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception: _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in _wt["text"] if t.strip()), return_tensors="pt").input_ids

    def is_t(n: str, m: torch.nn.Module) -> bool: return isinstance(m, torch.nn.Linear) and any(p in n for p in LINEAR_KEYS)
    targets = [(n, m) for n, m in model.named_modules() if is_t(n, m)]
    master = {m: m.weight.detach().to("cpu", copy=True) for _, m in targets}
    def restore() -> None:
        for _, m in targets: m.weight.data.copy_(master[m].to(DEV))

    _S["on"] = False; bf = ppl(model, ids)
    print(f"\n# F7 foils — {MODEL} | wikitext-2 {min(MW, (ids.size(1) - 2048) // 1024 + 1)}w 2048/1024 sdpa | W+K+V", flush=True)
    print(f"BF16                 {bf:.4f}", flush=True)
    for name in RUN:
        fn = QFN[name]
        for _, m in targets: m.weight.data.copy_(fn(master[m].to(DEV)).to(m.weight.dtype))
        _S["mode"] = name; _S["on"] = True; p = ppl(model, ids); _S["on"] = False; restore()
        print(f"{name:12s} ({BPE[name]:.2f}b)  {p:.4f} ({p-bf:+.4f}, {(p/bf-1)*100:+.2f}%)", flush=True)


if __name__ == "__main__":
    main()
