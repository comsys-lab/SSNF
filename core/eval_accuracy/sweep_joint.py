"""Joint W+KV SSNF configurations — full-window WikiText-2 PPL of whole (W, K, V) configs (configuration search, paper Section 4).
Same encoders and fixed settings as sweep_ubgs.py: W search radius = 1 << l_b, KV radius 0, no E8M0 scale-offset search, calibration = wikitext-2-raw train 128 x 2048.
Run: SSNF_MODEL=... SSNF_HK=... SSNF_W_LB=3 SSNF_W_GS=8 SSNF_K_LB=3 SSNF_K_GS=16 \
     SSNF_V_LB=4 SSNF_V_GS=32 SSNF_JOINT_OUT=out.csv python eval_accuracy/sweep_joint.py
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Callable

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling modules (wkv_v2_ppl, sweep_ubgs)

os.environ.setdefault("PPL_WINDOWS", "100000")
os.environ.setdefault("SSNF_W_SCALE_OFFS", "0")

import wkv_v2_ppl as W  # noqa: E402
from sweep_ubgs import bpe, bpe_aligned  # noqa: E402

DEV = "cuda"
_P = {"on": False}


def main() -> None:
    """Measure BF16 / KV-only / W+KV PPL for one (W, K, V) SSNF configuration and write a CSV row."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    MODEL = os.environ["SSNF_MODEL"]
    wlb, wgs = int(os.environ["SSNF_W_LB"]), int(os.environ["SSNF_W_GS"])
    klb, kgs = int(os.environ["SSNF_K_LB"]), int(os.environ["SSNF_K_GS"])
    vlb, vgs = int(os.environ["SSNF_V_LB"]), int(os.environ["SSNF_V_GS"])
    out = os.environ["SSNF_JOINT_OUT"]
    NCAL = int(os.environ.get("SSNF_CALIB_W", "128"))
    CSPLIT = os.environ.get("SSNF_CALIB_SPLIT", "train")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    d = torch.load(os.environ["SSNF_HK"], map_location=DEV)
    W._S["HK"] = d["H_K"].to(torch.float32).to(DEV)
    W._S["L"], W._S["Hkv"] = W._S["HK"].shape[0], W._S["HK"].shape[1]

    _real = F.scaled_dot_product_attention

    def _patch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *a, **kw) -> torch.Tensor:
        if _P["on"]:
            layer = W._S["i"] % W._S["L"]
            W._S["i"] += 1
            B, nh, Sq, D = k.shape
            hidx = torch.arange(nh, device=k.device) // (nh // W._S["Hkv"])
            Hf = W._S["HK"][layer][hidx].reshape(1, nh, 1, D).expand(B, nh, Sq, D)
            k = W.ssnf_hw(k.float(), klb, kgs, Hf, radius=0).to(k.dtype)
            v = W.ssnf_hw(v.float(), vlb, vgs, None, radius=0).to(v.dtype)
        return _real(q, k, v, *a, **kw)
    F.scaled_dot_product_attention = _patch

    def _load(split: str) -> torch.Tensor:
        try:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        except Exception:
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        return tok("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt").input_ids

    ids = _load("test")

    def is_t(n: str, m: torch.nn.Module) -> bool:
        return isinstance(m, torch.nn.Linear) and any(p in n for p in W.LINEAR_KEYS)
    targets = [(n, m) for n, m in model.named_modules() if is_t(n, m)]
    master = {m: m.weight.detach().to("cpu", copy=True) for _, m in targets}

    cal = _load(CSPLIT)
    acc = {m: [torch.zeros(m.weight.shape[1], device=DEV, dtype=torch.float64), 0] for _, m in targets}

    def mk(m: torch.nn.Linear) -> Callable:
        def h(mod: torch.nn.Module, inp: tuple) -> None:
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            acc[m][0] += x.pow(2).sum(0).double()
            acc[m][1] += x.shape[0]
        return h
    hs = [m.register_forward_pre_hook(mk(m)) for _, m in targets]
    nseg = min(NCAL, cal.size(1) // W.MAXLEN)
    with torch.no_grad():
        for i in range(nseg):
            model(cal[:, i * W.MAXLEN:(i + 1) * W.MAXLEN].to(DEV))
    for h in hs:
        h.remove()
    Hw = {m: (s / max(c, 1)).float() for m, (s, c) in acc.items()}
    print(f"# weight calib: {CSPLIT}, {nseg}x{W.MAXLEN} tok", flush=True)

    offs = tuple(int(x) for x in os.environ["SSNF_W_SCALE_OFFS"].split(","))
    _P["on"] = False
    t0 = time.time()
    bf = W.ppl(model, ids, None)
    print(f"# {MODEL} | BF16 {bf:.4f} ({time.time()-t0:.0f}s), window cap {W.MW}", flush=True)

    # KV-only, then W+KV — the split the paper reports
    _P["on"] = True
    t0 = time.time()
    p_kv = W.ppl(model, ids, None)
    _P["on"] = False
    for _, m in targets:
        Wt = master[m].to(DEV)
        OUT, Kin = Wt.shape
        Hf = Hw[m].reshape(1, Kin).expand(OUT, Kin)
        m.weight.data.copy_(W.wq_scale(Wt, wlb, wgs, Hf, radius=(1 << wlb), offs=offs).to(m.weight.dtype))
    _P["on"] = True
    p_wkv = W.ppl(model, ids, None)
    _P["on"] = False
    for _, m in targets:
        m.weight.data.copy_(master[m].to(DEV))

    wb, kb, vb = bpe(wlb, wgs), bpe(klb, kgs), bpe(vlb, vgs)
    wa, ka, va = bpe_aligned(wlb, wgs), bpe_aligned(klb, kgs), bpe_aligned(vlb, vgs)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("model,w_lb,w_gs,k_lb,k_gs,v_lb,v_gs,w_bpe,kv_bpe,w_bpe_aligned,kv_bpe_aligned,"
                "bf16,ppl_kv_only,d_kv_pct,ppl_wkv,d_wkv_pct,secs\n")
        f.write(f"{MODEL},{wlb},{wgs},{klb},{kgs},{vlb},{vgs},{wb:.4f},{(kb+vb)/2:.4f},"
                f"{wa:.4f},{(ka+va)/2:.4f},{bf:.6f},{p_kv:.6f},{(p_kv/bf-1)*100:.4f},"
                f"{p_wkv:.6f},{(p_wkv/bf-1)*100:.4f},{time.time()-t0:.1f}\n")
    print(f"W {wb:.2f}b (aligned {wa:.2f}) / KV {(kb+vb)/2:.2f}b (aligned {(ka+va)/2:.2f})  "
          f"KV-only {p_kv:.4f} ({(p_kv/bf-1)*100:+.2f}%)  W+KV {p_wkv:.4f} ({(p_wkv/bf-1)*100:+.2f}%)",
          flush=True)


if __name__ == "__main__":
    main()
