"""Verify the FP8 element-saturation convention shared by every format.
The shared `_round_fp` formula treats every encoding as normal, which overstates the E4M3 maximum
(480 vs OCP 448) and the E5M2 maximum (114688 vs OCP 57344); this script (1) checks a corrected
quantizer against torch's native float8 casts exactly and (2) measures the W+K+V PPL difference.
Run: SSNF_MODEL=<id> PPL_WINDOWS=20 CUDA_VISIBLE_DEVICES=0 python eval_accuracy/fp8_saturation_check.py
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling modules (wkv_v2_ppl)
os.environ.setdefault("PPL_WINDOWS", "20")

import wkv_v2_ppl as W  # noqa: E402

DEV = "cuda"

# (exponent bits, mantissa bits) -> (max normal exponent, max representable magnitude)
FP_SPEC: dict[tuple[int, int], tuple[int, float]] = {
    (4, 3): (8, 448.0),        # OCP FP8 E4M3  (torch.float8_e4m3fn: no inf, S.1111.111 = NaN)
    (5, 2): (15, 57344.0),     # OCP FP8 E5M2  (IEEE-style: exponent 31 is inf/NaN)
    (2, 3): (2, 7.5),          # OCP MXFP6 E2M3 — all encodings normal
    (3, 2): (4, 28.0),         # OCP MXFP6 E3M2 — all encodings normal
    (2, 1): (2, 6.0),          # OCP MXFP4 E2M1 — all encodings normal
}


def round_fp(y: torch.Tensor, eb: int, mb: int) -> torch.Tensor:
    """Round to the E{eb}M{mb} grid using the format's true representable range."""
    emax, maxval = FP_SPEC[(eb, mb)]
    emin = 1 - (2 ** (eb - 1) - 1)
    e = torch.floor(torch.log2(y.abs().clamp(min=1e-30))).clamp(min=float(emin), max=float(emax))
    step = torch.exp2(e - float(mb))
    return (torch.round(y / step) * step).clamp(-maxval, maxval)


def round_fp_legacy(y: torch.Tensor, eb: int, mb: int) -> torch.Tensor:
    """The shared repo formula (every encoding treated as normal), kept for the A/B."""
    bias = 2 ** (eb - 1) - 1
    emin = 1 - bias
    maxexp = (2 ** eb - 1) - bias
    maxval = (2 - 2.0 ** (-mb)) * (2.0 ** maxexp)
    e = torch.floor(torch.log2(y.abs().clamp(min=1e-30))).clamp(min=float(emin), max=float(maxexp))
    step = torch.exp2(e - float(mb))
    return (torch.round(y / step) * step).clamp(-maxval, maxval)


def mxfp(x: torch.Tensor, eb: int, mb: int,
         rounder: Callable[[torch.Tensor, int, int], torch.Tensor]) -> torch.Tensor:
    """Per-32 E8M0 block scale aligned to the format's max exponent + E{eb}M{mb} element."""
    emax = FP_SPEC[(eb, mb)][0] if rounder is round_fp else (2 ** eb - 1) - (2 ** (eb - 1) - 1)
    shp = x.shape
    xf = x.reshape(-1, 32).float()
    am = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    X = torch.exp2(torch.floor(torch.log2(am)) - float(emax))
    return (rounder(xf / X, eb, mb) * X).reshape(shp).to(x.dtype)


def nvfp4(x: torch.Tensor,
          rounder: Callable[[torch.Tensor, int, int], torch.Tensor]) -> torch.Tensor:
    """NVFP4: 16-element E2M1 block + per-block FP8(E4M3) scale + tensor-level FP32 scale."""
    shp = x.shape
    xf = x.reshape(-1, 16).float()
    st = (xf.abs().amax().clamp(min=1e-30) / (448.0 * 6.0)).clamp(min=1e-30)
    mi = rounder(xf.abs().amax(-1, keepdim=True).clamp(min=1e-30) / 6.0 / st, 4, 3).clamp(min=1e-30)
    eff = mi * st
    return (rounder(xf / eff, 2, 1) * eff).reshape(shp).to(x.dtype)


def verify() -> bool:
    """PASS iff round_fp reproduces torch's native float8 casts with zero error (normals only)."""
    torch.manual_seed(0)
    ok = True
    for (eb, mb), dt in (((4, 3), torch.float8_e4m3fn), ((5, 2), torch.float8_e5m2)):
        # cover normals, subnormals and the saturation region
        y = torch.cat([torch.randn(200000, device=DEV) * 100,
                       torch.linspace(-2 * FP_SPEC[(eb, mb)][1], 2 * FP_SPEC[(eb, mb)][1],
                                      200000, device=DEV)])
        ref = y.to(dt).float().clamp(-FP_SPEC[(eb, mb)][1], FP_SPEC[(eb, mb)][1])
        ref = torch.where(torch.isnan(ref), torch.sign(y) * FP_SPEC[(eb, mb)][1], ref)
        got = round_fp(y, eb, mb)
        sub = y.abs() >= 2.0 ** (1 - (2 ** (eb - 1) - 1))       # skip subnormals (not used by MX)
        d = (got - ref)[sub].abs().max().item()
        leg = (round_fp_legacy(y, eb, mb) - ref)[sub].abs().max().item()
        print(f"  E{eb}M{mb}: corrected vs torch maxdiff {d:.3e} | legacy vs torch maxdiff {leg:.3e}")
        ok &= d == 0.0
    print("  verify:", "PASS" if ok else "FAIL")
    return ok


def main() -> None:
    """Run the codec verification, then the legacy-vs-corrected W+K+V PPL A/B."""
    print("# 1. element codec vs torch native float8")
    if not verify():
        sys.exit(1)

    if os.environ.get("VERIFY_ONLY"):
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    MODEL = os.environ.get("SSNF_MODEL", "NousResearch/Meta-Llama-3.1-8B")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()

    st: dict[str, object] = {"on": False, "fn": None}
    _real = F.scaled_dot_product_attention

    def _patch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *a, **kw) -> torch.Tensor:
        if st["on"]:
            k, v = st["fn"](k.float()).to(k.dtype), st["fn"](v.float()).to(v.dtype)
        return _real(q, k, v, *a, **kw)
    F.scaled_dot_product_attention = _patch

    try:
        wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()), return_tensors="pt").input_ids

    def is_t(n: str, m: torch.nn.Module) -> bool:
        return isinstance(m, torch.nn.Linear) and any(p in n for p in W.LINEAR_KEYS)
    targets = [(n, m) for n, m in model.named_modules() if is_t(n, m)]
    master = {m: m.weight.detach().to("cpu", copy=True) for _, m in targets}

    st["on"] = False
    bf = W.ppl(model, ids, None)
    print(f"\n# 2. W+K+V PPL — {MODEL}, {W.MW}w  |  BF16 = {bf:.4f}")
    for eb, mb, name in ((4, 3, "MXFP8-E4M3"), (5, 2, "MXFP8-E5M2"),
                         (2, 3, "MXFP6-E2M3"), (3, 2, "MXFP6-E3M2")):
        for rounder, lbl in ((round_fp_legacy, "legacy"), (round_fp, "corrected")):
            fn = (lambda x, _e=eb, _m=mb, _r=rounder: mxfp(x, _e, _m, _r))
            for _, m in targets:
                m.weight.data.copy_(fn(master[m].to(DEV)).to(m.weight.dtype))
            st["fn"] = fn
            st["on"] = True
            p = W.ppl(model, ids, None)
            st["on"] = False
            for _, m in targets:
                m.weight.data.copy_(master[m].to(DEV))
            print(f"  {name:12s} {lbl:9s}  PPL {p:.4f} ({(p/bf-1)*100:+.2f}%)", flush=True)


if __name__ == "__main__":
    main()
