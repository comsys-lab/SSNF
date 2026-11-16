"""Zero-shot accuracy under W+KV quantization, log-likelihood scoring (lm-eval conventions, 0-shot) on the HF fake-quant path.
Tasks: PIQA, ARC-Easy/Challenge, HellaSwag, WinoGrande, BoolQ, OpenBookQA, MMLU; acc = argmax summed log-prob, acc_norm = per-character normalized (WinoGrande: partial evaluation, acc is the reported number).
CONFIG in {bf16, mxint8, mxfp8_e4m3, mxfp8_e5m2, mxfp6_e2m3, mxfp6_e3m2, mxfp4, nvfp4, ssnf}, all W+KV.
Run: CONFIG=ssnf SSNF_MODEL=NousResearch/Meta-Llama-3.1-8B SSNF_HK=calib/H_K_metallama318b.pt CUDA_VISIBLE_DEVICES=0 python eval_accuracy/eval_zeroshot.py
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling modules (wkv_v2_ppl, comparison, f7_foils)

from wkv_v2_ppl import ssnf_hw, wq_scale, LINEAR_KEYS  # noqa: E402  same encoders as the PPL table
from comparison import mxfp                            # noqa: E402  OCP MXFP (floor scale)
from f7_foils import _nvfp4                            # noqa: E402

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DEV = "cuda"
TASKS = [t for t in os.environ.get(
    "ZS_TASKS", "piqa,arc_easy,arc_challenge,hellaswag,winogrande").split(",") if t]
NSAMP = int(os.environ.get("NSAMP", "0"))         # 0 -> full split
BATCH = int(os.environ.get("ZS_BATCH", "16"))
MAXTOK = int(os.environ.get("ZS_MAXTOK", "8192"))   # padded tokens per batch
MODEL = os.environ.get("SSNF_MODEL", "NousResearch/Meta-Llama-3.1-8B")


def mxint_n(x: torch.Tensor, n: int) -> torch.Tensor:
    """OCP MXINT-n: per-32 block, E8M0 = 2^(floor(log2 amax) - (n-2)), signed n-bit element."""
    shp = x.shape
    xf = x.reshape(-1, 32).float()
    am = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    qmax = (1 << (n - 1)) - 1
    s = torch.exp2(torch.floor(torch.log2(am)) - float(n - 2))
    return (torch.round(xf / s).clamp(-qmax, qmax) * s).reshape(shp).to(x.dtype)


# ---------------------------------------------------------------------------
# formats
# ---------------------------------------------------------------------------
_S = {"on": False, "cfg": None, "HK": None, "L": 0, "Hkv": 0, "i": 0}

# SSNF deployed configuration (paper): W l_b=3/g=8 (5.63b), K l_b=3/g=16 (5.44b), V l_b=4/g=32 (4.38b)
SSNF_W = (3, 8)
SSNF_K = (3, 16)
SSNF_V = (4, 32)

ELEM = {                                   # element codec applied to both weights and KV
    "mxint8":     lambda x: mxint_n(x, 8),
    "mxfp8_e4m3": lambda x: mxfp(x, 4, 3),
    "mxfp8_e5m2": lambda x: mxfp(x, 5, 2),
    "mxfp6_e2m3": lambda x: mxfp(x, 2, 3),
    "mxfp6_e3m2": lambda x: mxfp(x, 3, 2),
    "mxfp4":      lambda x: mxfp(x, 2, 1),
    "nvfp4":      _nvfp4,
}
BPE = {"bf16": "16 / 16", "mxint8": "8.25 / 8.25", "mxfp8_e4m3": "8.25 / 8.25",
       "mxfp8_e5m2": "8.25 / 8.25", "mxfp6_e2m3": "6.25 / 6.25", "mxfp6_e3m2": "6.25 / 6.25",
       "mxfp4": "4.25 / 4.25", "nvfp4": "4.50 / 4.50", "ssnf": "5.63 / 4.91"}


def _kv_quant(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize K and V with the active CONFIG's codec."""
    cfg = _S["cfg"]
    if cfg == "ssnf":
        layer = _S["i"] % _S["L"]
        _S["i"] += 1
        B, nh, Sq, D = k.shape
        hidx = torch.arange(nh, device=k.device) // (nh // _S["Hkv"])
        Hf = _S["HK"][layer][hidx].reshape(1, nh, 1, D).expand(B, nh, Sq, D)
        return (ssnf_hw(k.float(), *SSNF_K, Hf, radius=0).to(k.dtype),
                ssnf_hw(v.float(), *SSNF_V, None, radius=0).to(v.dtype))
    fn = ELEM[cfg]
    return fn(k.float()).to(k.dtype), fn(v.float()).to(v.dtype)


# ---------------------------------------------------------------------------
# task construction -- (context, [continuations], gold) per document
# ---------------------------------------------------------------------------
def _hs_clean(t: str) -> str:
    """Clean a HellaSwag text fragment (lm-eval preprocessing)."""
    t = t.strip().replace(" [title]", ". ")
    t = re.sub(r"\[.*?\]", "", t)
    return t.replace("  ", " ")


def build(task: str) -> list[tuple[str, list[str], int] | tuple[list[str], str, int]]:
    """Build (context, [continuations], gold) documents for one task (WinoGrande: ([contexts], continuation, gold))."""
    from datasets import load_dataset
    if task == "piqa":
        ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
        return [(f"Question: {d['goal']}\nAnswer:", [f" {d['sol1']}", f" {d['sol2']}"], int(d["label"]))
                for d in ds]
    if task in ("arc_easy", "arc_challenge"):
        cfg = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        ds = load_dataset("allenai/ai2_arc", cfg, split="test")
        out = []
        for d in ds:
            labels, texts = d["choices"]["label"], d["choices"]["text"]
            if d["answerKey"] not in labels:
                continue
            out.append((f"Question: {d['question']}\nAnswer:", [f" {t}" for t in texts],
                        labels.index(d["answerKey"])))
        return out
    if task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        out = []
        for d in ds:
            ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
            q = _hs_clean(d["activity_label"] + ": " + ctx)
            out.append((q, [" " + _hs_clean(e) for e in d["endings"]], int(d["label"])))
        return out
    if task == "boolq":
        ds = load_dataset("google/boolq", split="validation")
        return [(f"{d['passage']}\nQuestion: {d['question']}?\nAnswer:", [" no", " yes"],
                 int(d["answer"])) for d in ds]
    if task == "openbookqa":
        ds = load_dataset("allenai/openbookqa", "main", split="test")
        out = []
        for d in ds:
            labels, texts = d["choices"]["label"], d["choices"]["text"]
            if d["answerKey"] not in labels:
                continue
            out.append((d["question_stem"], [f" {t}" for t in texts], labels.index(d["answerKey"])))
        return out
    if task == "mmlu":
        # lm-eval 0-shot format: per-subject header, lettered options, single-letter continuation
        ds = load_dataset("cais/mmlu", "all", split="test")
        out = []
        for d in ds:
            subj = d["subject"].replace("_", " ")
            head = f"The following are multiple choice questions (with answers) about {subj}.\n\n"
            body = "\n".join(f"{k}. {c}" for k, c in zip("ABCD", d["choices"]))
            out.append((f"{head}{d['question']}\n{body}\nAnswer:",
                        [" A", " B", " C", " D"], int(d["answer"])))
        return out
    if task == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation",
                          trust_remote_code=True)
        out = []
        for d in ds:
            cut = d["sentence"].index("_")
            tail = d["sentence"][cut + 1:]
            # partial evaluation: the two candidates differ only in the CONTEXT, and the scored
            # continuation (the text after the blank) is identical for both.
            out.append(([d["sentence"][:cut] + d["option1"], d["sentence"][:cut] + d["option2"]],
                        tail, int(d["answer"]) - 1))
        return out
    raise SystemExit(f"unknown task {task}")


# ---------------------------------------------------------------------------
# log-likelihood scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def _score(model: torch.nn.Module, tok: "PreTrainedTokenizerBase", pairs: list[tuple[str, str]]) -> list[float]:
    """pairs = [(context_str, continuation_str)] -> [sum log p(continuation | context)]."""
    out = []
    # Tokenize everything first so batches can be formed on a TOKEN budget rather than a fixed
    # count: BoolQ passages are ~10x longer than a PIQA goal, and a fixed batch of those overflows
    # 24 GB once the logits tensor is materialized.
    enc_all = []
    for ctx, cont in pairs:
        c = tok(ctx, add_special_tokens=False).input_ids
        k = tok(cont, add_special_tokens=False).input_ids
        if not k:                                         # empty continuation cannot be scored
            k = [tok.eos_token_id]
        enc_all.append((c, k))

    batches, cur, cur_max = [], [], 0
    for e in enc_all:
        m = max(cur_max, len(e[0]) + len(e[1]))
        if cur and (len(cur) + 1 > BATCH or (len(cur) + 1) * m > MAXTOK):
            batches.append(cur)
            cur, cur_max = [e], len(e[0]) + len(e[1])
        else:
            cur.append(e)
            cur_max = m
    if cur:
        batches.append(cur)

    for enc in batches:
        L = max(len(c) + len(k) for c, k in enc)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        # RIGHT padding: with causal attention, tokens after the sequence cannot influence the
        # logits of earlier positions, and every real token keeps its natural position index. Left
        # padding would need both an attention mask and shifted position ids to be equivalent.
        ids = torch.full((len(enc), L), pad, dtype=torch.long)
        mask = torch.zeros((len(enc), L), dtype=torch.long)
        for j, (c, k) in enumerate(enc):
            s = c + k
            ids[j, :len(s)] = torch.tensor(s)
            mask[j, :len(s)] = 1
        ids, mask = ids.to(DEV), mask.to(DEV)
        _S["i"] = 0
        logits = model(ids, attention_mask=mask).logits
        for j, (c, k) in enumerate(enc):
            nc, n = len(c), len(k)
            # continuation occupies positions nc .. nc+n-1; each is predicted from the one before.
            # Slice BEFORE the float()/log_softmax: casting the whole [B, L, vocab] tensor to fp32
            # is what overflows memory on long-context tasks. log_softmax is per position, so
            # slicing first is numerically identical.
            lp = logits[j, nc - 1:nc + n - 1, :].float().log_softmax(-1)
            tgt = ids[j, nc:nc + n]
            out.append(lp.gather(-1, tgt.unsqueeze(-1)).sum().item())
        del logits
    return out


def run_task(model: torch.nn.Module, tok: "PreTrainedTokenizerBase", task: str) -> dict[str, int | float]:
    """Score one task and return n / acc / acc_norm with standard errors."""
    docs = build(task)
    if NSAMP:
        docs = docs[:NSAMP]
    pairs, index = [], []
    for d in docs:
        if task == "winogrande":
            ctxs, cont, gold = d
            index.append((len(pairs), len(ctxs), gold, [len(cont)] * len(ctxs)))
            pairs += [(c, cont) for c in ctxs]
        else:
            ctx, conts, gold = d
            index.append((len(pairs), len(conts), gold, [len(c) for c in conts]))
            pairs += [(ctx, c) for c in conts]
    lps = _score(model, tok, pairs)
    ok = okn = 0
    for start, n, gold, lens in index:
        s = lps[start:start + n]
        if int(max(range(n), key=lambda z: s[z])) == gold:
            ok += 1
        sn = [s[z] / max(lens[z], 1) for z in range(n)]
        if int(max(range(n), key=lambda z: sn[z])) == gold:
            okn += 1
    N = len(index)
    acc, acc_norm = ok / N, okn / N
    return dict(n=N, acc=100 * acc, acc_se=100 * math.sqrt(acc * (1 - acc) / N),
                acc_norm=100 * acc_norm, acc_norm_se=100 * math.sqrt(acc_norm * (1 - acc_norm) / N))


def main() -> None:
    """Quantize weights per CONFIG, install the KV hook, and run the task suite."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = os.environ.get("CONFIG", "bf16")
    assert cfg in BPE, f"unknown CONFIG={cfg} (pick one of {list(BPE)})"
    _S["cfg"] = cfg

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()

    _real = F.scaled_dot_product_attention

    def _patch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *a, **kw) -> torch.Tensor:
        if _S["on"]:
            k, v = _kv_quant(k, v)
        return _real(q, k, v, *a, **kw)
    F.scaled_dot_product_attention = _patch

    def is_t(n: str, m: torch.nn.Module) -> bool:
        return isinstance(m, torch.nn.Linear) and any(p in n for p in LINEAR_KEYS)
    targets = [(n, m) for n, m in model.named_modules() if is_t(n, m)]

    if cfg != "bf16":
        if cfg == "ssnf":
            d = torch.load(os.environ["SSNF_HK"], map_location=DEV)
            _S["HK"] = d["H_K"].to(torch.float32).to(DEV)
            _S["L"], _S["Hkv"] = _S["HK"].shape[0], _S["HK"].shape[1]
            Hw = _calib_weight_importance(model, tok, targets)
            offs = tuple(int(x) for x in os.environ.get("SSNF_W_SCALE_OFFS", "0").split(","))
            lb, g_size = SSNF_W
            for _, m in targets:
                Wt = m.weight.detach().float()
                OUT, Kin = Wt.shape
                Hf = Hw[m].reshape(1, Kin).expand(OUT, Kin)
                m.weight.data.copy_(
                    wq_scale(Wt, lb, g_size, Hf, radius=(1 << lb), offs=offs).to(m.weight.dtype))
        else:
            fn = ELEM[cfg]
            for _, m in targets:
                m.weight.data.copy_(fn(m.weight.detach().float()).to(m.weight.dtype))
        _S["on"] = True

    print(f"# zero-shot — {MODEL} | CONFIG={cfg} (W/KV bpe {BPE[cfg]}) | tasks={','.join(TASKS)}",
          flush=True)
    res, t0 = {}, time.time()
    for t in TASKS:
        s = time.time()
        r = run_task(model, tok, t)
        res[t] = r
        print(f"  {t:14s} n={r['n']:5d}  acc {r['acc']:.2f} ± {r['acc_se']:.2f}   "
              f"acc_norm {r['acc_norm']:.2f} ± {r['acc_norm_se']:.2f}   [{time.time()-s:.0f}s]",
              flush=True)
    accs = [res[t]["acc"] for t in TASKS]
    avg = sum(accs) / len(accs)
    se = math.sqrt(sum(res[t]["acc_se"] ** 2 for t in TASKS)) / len(TASKS)
    print(f"  {'AVERAGE':14s}          acc {avg:.2f} ± {se:.2f}   [{time.time()-t0:.0f}s total]",
          flush=True)
    out = dict(cfg=cfg, model=MODEL, bpe=BPE[cfg], per_task=res, avg=avg, avg_se=se,
               elapsed_s=round(time.time() - t0, 1))
    if os.environ.get("CP_OUT"):
        with open(os.environ["CP_OUT"], "w") as f:
            json.dump(out, f, indent=1)


def _calib_weight_importance(model: torch.nn.Module, tok: "PreTrainedTokenizerBase",
                             targets: list[tuple[str, torch.nn.Linear]]) -> dict[torch.nn.Linear, torch.Tensor]:
    """E[x^2] per input channel from wikitext-2-raw TRAIN, 128 x 2048-token segments (paper §Methodology)."""
    from datasets import load_dataset
    n_seg = int(os.environ.get("SSNF_CALIB_W", "128"))
    seqlen = int(os.environ.get("PPL_MAXLEN", "2048"))
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except Exception:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt").input_ids
    acc = {m: [torch.zeros(m.weight.shape[1], device=DEV, dtype=torch.float64), 0] for _, m in targets}

    def mk(m: torch.nn.Linear) -> Callable:
        def h(mod: torch.nn.Module, inp: tuple) -> None:
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            acc[m][0] += x.pow(2).sum(0).double()
            acc[m][1] += x.shape[0]
        return h
    hs = [m.register_forward_pre_hook(mk(m)) for _, m in targets]
    n = min(n_seg, ids.size(1) // seqlen)
    with torch.no_grad():
        for i in range(n):
            model(ids[:, i * seqlen:(i + 1) * seqlen].to(DEV))
    for h in hs:
        h.remove()
    print(f"# weight calib: train split, {n}x{seqlen} tok", flush=True)
    return {m: (s / max(c, 1)).float() for m, (s, c) in acc.items()}


if __name__ == "__main__":
    main()
