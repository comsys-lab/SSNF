"""Comparison — SSNF vs native MX formats (MXINT8/6/4, MXFP8/6/4, SSNF-INT8/FP variants) on a common bits-per-element axis: weight QSNR and weight-only PPL (see comparison.md).
QSNR only:   CUDA_VISIBLE_DEVICES=1 python eval_accuracy/comparison.py --qsnr
QSNR + PPL:  HF_HUB_OFFLINE=1 SSNF_MODEL=NousResearch/Meta-Llama-3.1-8B \
             CUDA_VISIBLE_DEVICES=1 python eval_accuracy/comparison.py
"""
from __future__ import annotations
import os, sys, glob, torch
from typing import Callable
from safetensors import safe_open
from mxfp_baseline import mxfp_quant, ssnf_mxint8_efb, bits_mxint8, BLOCK

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def qsnr(x: torch.Tensor, xq: torch.Tensor) -> float:
    """Reconstruction QSNR of xq against x, in dB."""
    e = (x - xq).double().pow(2).mean()
    return 10.0 * torch.log10(x.double().pow(2).mean() / e.clamp(min=1e-45)).item()


def mxint_n(x: torch.Tensor, n: int) -> torch.Tensor:  # native MXINT-n: per-block E8M0 + signed n-bit
    xf = x.reshape(-1, BLOCK).float(); am = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    qmax = (1 << (n - 1)) - 1
    s = torch.exp2(torch.ceil(torch.log2(am / qmax)))
    return (torch.round(xf / s).clamp(-qmax, qmax) * s).reshape(x.shape)


def mxfp(x: torch.Tensor, eb: int, mb: int) -> torch.Tensor:  # native OCP MXFP (floor scale)
    return mxfp_quant(x, eb, mb)


def bits_int_native(n: int) -> float: return n + 8.0 / BLOCK
def bits_fp_native(eb: int, mb: int) -> float: return 1 + eb + mb + 8.0 / BLOCK


# Dense sweep knobs (user request): u in {1..4}, group size gs(mg) in {2,4,8,16,32}.
SWEEP_U = (1, 2, 3, 4)
SWEEP_MG = (2, 4, 8, 16, 32)


# REGISTRY: (group, name, bits, fn, ppl?)
def reg() -> list[tuple[str, str, float, Callable[[torch.Tensor], torch.Tensor], bool]]:
    """Build the (group, name, bits, fn, ppl?) format registry."""
    R = []
    add = lambda *a: R.append(a)
    # ---- native anchors (Pareto reference points) ----
    add("native-INT", "MXINT8",     bits_int_native(8),  lambda x: mxint_n(x, 8), True)
    add("native-INT", "MXINT6",     bits_int_native(6),  lambda x: mxint_n(x, 6), True)
    add("native-INT", "MXINT4",     bits_int_native(4),  lambda x: mxint_n(x, 4), True)
    add("native-FP",  "MXFP8-E4M3", bits_fp_native(4, 3), lambda x: mxfp(x, 4, 3), True)
    add("native-FP",  "MXFP6-E2M3", bits_fp_native(2, 3), lambda x: mxfp(x, 2, 3), True)
    add("native-FP",  "MXFP4-E2M1", bits_fp_native(2, 1), lambda x: mxfp(x, 2, 1), True)
    # ---- dense u x mg sweep per base ----
    #   kind int8 -> ssnf_mxint8_efb (the deployed INT-element encoder family).
    DENSE = [("SSNF-INT8",  "INT8", "int8",  None, None, True)]
    for grp, tag, kind, eb, mb, ppl in DENSE:
        for l_b in SWEEP_U:
            for mg in SWEEP_MG:
                b = bits_mxint8(l_b, mg); fn = (lambda x, l_b=l_b, mg=mg: ssnf_mxint8_efb(x, l_b, mg, 2))
                add(grp, f"{tag} lb{l_b}/mg{mg}", b, fn, ppl)
    return R


REGISTRY = reg()


def load_layers() -> dict[str, torch.Tensor]:
    """Load two representative layer-0 weights of SSNF_MODEL from the local HF cache."""
    # Standalone weight-QSNR helper only (run_qsnr); the mxfp() encoder the drivers import does not
    # touch this. Reads the first checkpoint shard of SSNF_MODEL from the local HF cache.
    model = os.environ.get("SSNF_MODEL", "NousResearch/Meta-Llama-3.1-8B")
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    f = sorted(glob.glob(os.path.join(hub, "models--" + model.replace("/", "--"),
                         "snapshots", "*", "model-00001-of-*.safetensors")))[0]
    out = {}
    with safe_open(f, framework="pt") as fh:
        for k in fh.keys():
            if k.endswith("layers.0.self_attn.q_proj.weight"): out["q_proj"] = fh.get_tensor(k)
            if k.endswith("layers.0.mlp.down_proj.weight"):     out["down_proj"] = fh.get_tensor(k)
    return out


def run_qsnr() -> None:
    """Print reconstruction QSNR of every registry format on real weights."""
    layers = load_layers()
    print("# Comparison — calibration-free reconstruction QSNR (real Llama-3.1-8B weights)\n")
    for ln, W in layers.items():
        W = W.to(DEV).float()
        print(f"## {ln}  shape={tuple(W.shape)}")
        print(f"   {'group':>11} {'format':>16} {'bits':>5} {'QSNR':>7}")
        for grp, nm, b, fn, _ in sorted(REGISTRY, key=lambda r: (r[0], -r[2])):
            print(f"   {grp:>11} {nm:>16} {b:>5.2f} {qsnr(W, fn(W).float()):>7.2f}")
        print()


def run_ppl() -> None:
    """Measure weight-only wikitext-2 PPL for the PPL-flagged registry formats."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    MODEL = os.environ.get("SSNF_MODEL", "NousResearch/Meta-Llama-3.1-8B")
    MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 20
    LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    def is_target(n: str, m: torch.nn.Module) -> bool: return isinstance(m, torch.nn.Linear) and any(p in n for p in LINEAR_KEYS)

    @torch.no_grad()
    def ppl(model: torch.nn.Module, ids: torch.Tensor) -> float:
        seq = ids.size(1); nll, ntok, prev, n = 0.0, 0, 0, 0
        for b in range(0, seq, STRIDE):
            e = min(b + MAXLEN, seq); trg = e - prev
            inp = ids[:, b:e].to(DEV); tgt = inp.clone(); tgt[:, :-trg] = -100
            nll += model(inp, labels=tgt).loss.double().item() * trg; ntok += trg; prev = e; n += 1
            if n >= MAX_WINDOWS or e == seq: break
        return torch.exp(torch.tensor(nll / ntok)).item()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(DEV).eval()
    try: _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception: _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in _wt["text"] if t.strip()), return_tensors="pt").input_ids
    targets = [(n, m) for n, m in model.named_modules() if is_target(n, m)]
    master = {m: m.weight.detach().to("cpu", copy=True) for _, m in targets}
    def restore() -> None:
        for _, m in targets: m.weight.data.copy_(master[m].to(DEV))
    bf = ppl(model, ids)
    print(f"\n# Weight-only PPL — wikitext-2 ({ids.size(1):,} tok) | BF16 = {bf:.4f} | {MAX_WINDOWS}w | gate <=3%\n", flush=True)
    print(f"   {'group':>11} {'format':>16} {'bits':>5} {'PPL':>9} {'dPPL%':>8}  gate", flush=True)
    only = os.environ.get("SSNF_PPL_GROUP")                # restrict PPL to one group (forces it on)
    for grp, nm, b, fn, doppl in sorted(REGISTRY, key=lambda r: (-r[2], r[0])):
        if only:
            if grp != only: continue
        elif not doppl: continue
        restore()
        for _, m in targets: m.weight.data.copy_(fn(master[m].to(DEV)).to(m.weight.dtype))
        p = ppl(model, ids); restore()
        d = (p / bf - 1) * 100
        print(f"   {grp:>11} {nm:>16} {b:>5.2f} {p:>9.4f} {d:>+7.2f}%  {'PASS' if d <= 3 else ''}", flush=True)


if __name__ == "__main__":
    run_qsnr()
    if "--qsnr" not in sys.argv:
        run_ppl()
