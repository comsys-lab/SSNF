"""Calibrate per-layer importance H = E[x^2] (per-input-channel) for the SSNF greedy+H weight encoder.

Saves {"<layer_idx>.<role>": tensor[K]} keyed like quant.py's _hkey (roles: qkv/o/gu/down).
Usage (ACCURACY environment): python -m ssnf_vllm.calibrate_h --model <hf_id_or_path> --out ssnf_H.pt
Then serve with SSNF_HFILE=ssnf_H.pt (and SSNF_PROFILE=mixed for the nibble config).
"""
import argparse
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def _hkey(name: str) -> str:
    m = re.search(r"layers\.(\d+)\.", name)
    li = m.group(1) if m else "x"
    if any(k in name for k in ("q_proj", "k_proj", "v_proj", "qkv_proj")):
        role = "qkv"
    elif "o_proj" in name:
        role = "o"
    elif any(k in name for k in ("gate_proj", "up_proj", "gate_up_proj")):
        role = "gu"
    elif "down_proj" in name:
        role = "down"
    else:
        role = "other"
    return f"{li}.{role}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    # Paper §Methodology: "128 sequences of 2,048 tokens from the WikiText-2 training split".
    # (An earlier revision collected a single 4,096-token forward from the TEST split -- i.e. the
    # split PPL is evaluated on, and 1/64 of the stated sample count.)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--split", default="train")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device).eval()

    try:
        wt = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.split)
    except Exception:
        wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=args.split)
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()), return_tensors="pt").input_ids

    KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    tgt = [(n, m) for n, m in model.named_modules()
           if isinstance(m, torch.nn.Linear) and any(k in n for k in KEYS)]
    acc = {n: (torch.zeros(m.weight.shape[1], device=args.device, dtype=torch.float64), [0]) for n, m in tgt}

    def mkhook(n: str):
        def h(mod: torch.nn.Module, inp: tuple) -> None:
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            acc[n][0].add_((x * x).sum(0).double())
            acc[n][1][0] += x.shape[0]
        return h

    hs = [m.register_forward_pre_hook(mkhook(n)) for n, m in tgt]
    nseg = min(args.nsamples, ids.shape[1] // args.seqlen)
    print(f"# calib: {args.split} split, {nseg} x {args.seqlen} tok", flush=True)
    with torch.no_grad():
        for i in range(nseg):
            model(ids[:, i * args.seqlen:(i + 1) * args.seqlen].to(args.device))
    for h in hs:
        h.remove()

    # Reduce to _hkey (q/k/v collapse to one 'qkv' key; identical input so any is fine).
    out = {}
    for n, _ in tgt:
        s, c = acc[n]
        out[_hkey(n)] = (s / max(c[0], 1)).float().cpu()      # [K] E[x^2] per input channel
    torch.save(out, args.out)
    print(f"saved {len(out)} H vectors -> {args.out}  (calibrated on {acc[tgt[0][0]][1][0]} tokens)")


if __name__ == "__main__":
    main()
