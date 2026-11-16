"""GSM8K accuracy eval (in-process vLLM) across KV/weight formats. Protocol matches lm-eval `gsm8k`
default: 5-shot CoT (train[:5], calculator <<...>> stripped), greedy, strict '#### N' + flexible
last-number scoring. One CONFIG per process; CONFIG ∈ {bf16, fp8, mxint8_kv, ssnf_kv, ssnf, *_fq};
N=0 → full 1319 test; CP_OUT=<json>. Run:
  CONFIG=bf16 N=200 CUDA_VISIBLE_DEVICES=0 CP_OUT=/tmp/g_bf16.json python eval_serving/eval_gsm8k.py  (SERVING environment)
"""
from __future__ import annotations

import json, os, re, time
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import ssnf_vllm  # noqa: F401 — registers ssnf/mxint8 quant + SSNF KV backend
from vllm import LLM, SamplingParams
from datasets import Dataset, load_dataset

MODEL = os.environ.get("SSNF_MODEL", "NousResearch/Meta-Llama-3.1-8B")
N = int(os.environ.get("N", "0"))          # 0 → full test set (1319)
NSHOT = int(os.environ.get("NSHOT", "5"))
MAXTOK = int(os.environ.get("MAXTOK", "256"))

_CALC = re.compile(r"<<[^>]*>>")           # strip calculator annotations from few-shot
_GOLD = re.compile(r"####\s*(-?[0-9][0-9,]*\.?[0-9]*)")
_STRICT = re.compile(r"####\s*(-?[0-9][0-9,]*\.?[0-9]*)")
_NUM = re.compile(r"-?[0-9][0-9,]*\.?[0-9]*")   # flexible: any number


def norm(s: str | None) -> float | None:
    """Normalize a numeric string for comparison: drop $ , spaces, trailing .0 / dot."""
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def build_fewshot(train: Dataset) -> str:
    """Build the NSHOT few-shot preamble from GSM8K train examples."""
    shots = []
    for ex in train.select(range(NSHOT)):
        ans = _CALC.sub("", ex["answer"]).strip()      # keep reasoning + '#### N', drop <<>>
        shots.append(f"Question: {ex['question']}\nAnswer: {ans}")
    return "\n\n".join(shots)


def make_engine(cfg: str) -> LLM:
    """Build the vLLM engine for CONFIG cfg, installing KV/weight fake-quant hooks where needed."""
    kw = dict(model=MODEL, dtype="bfloat16", enforce_eager=True,
              max_model_len=2048, gpu_memory_utilization=0.85)
    if cfg == "ssnf":
        kw.update(quantization="ssnf", kv_cache_dtype="ssnf_u3gs16")   # W+KV
    elif cfg == "ssnf_kv":
        kw.update(kv_cache_dtype="ssnf_u3gs16")                        # KV-only (deploy reco)
    elif cfg == "mxint8_kv":
        kw.update(kv_cache_dtype="mxint8_kv")                         # fair int8 baseline
    elif cfg == "fp8":
        kw.update(kv_cache_dtype="fp8")
    elif cfg in ("mxfp4", "nvfp4"):
        kw.update(quantization=cfg)                                   # W fake-quant; KV via hook (native bf16)
    elif cfg == "ssnf_fq":
        kw.update(quantization="ssnf")                               # W+KV: deployed ssnf weight + ssnf KV hook
    elif cfg == "mxint8_fq":
        kw.update(quantization="mxint8")                             # W+KV: mxint8 weight + mxint8 KV hook
    elif cfg == "mxfp8_fq":
        kw.update(quantization="mxfp8w")                             # W+KV: MXFP8-E4M3 weight + E4M3 KV hook
    elif cfg == "mxfp6_fq":
        kw.update(quantization="mxfp6w")                             # W+KV: MXFP6-E2M3 weight + E2M3 KV hook
    elif cfg == "ssnf_kv_fq":
        pass                                                          # KV-only (optional; W=bf16)
    elif cfg in ("mxfp8_e5m2_fq", "mxfp6_e3m2_fq", "mxfp8_fqw"):
        # W+KV via fake-quant. E5M2/E3M2 have no packed CUDA path at all; mxfp8_fqw is E4M3 routed
        # the same way on purpose, because the packed E4M3 path saturates at the true format maximum
        # (448) while the fake-quant codec used everywhere else saturates at 480. That makes the
        # packed weights ~1.2-1.5 dB worse, so comparing packed-E4M3 against fake-quant-E5M2 would
        # be comparing two conventions. mxfp8_fqw puts E4M3 on the same convention as its alternate.
        # (E2M3 needs no such variant: its packed and fake-quant codecs agree bit-exactly.)
        pass
    elif cfg == "bf16":
        pass                                                          # baseline, no quant
    else:
        raise SystemExit(f"unknown CONFIG={cfg}")
    llm = LLM(**kw)
    _FQ = {"mxfp4": "mxfp4", "nvfp4": "nvfp4", "mxint8_fq": "mxint8",
           "mxfp8_fq": "mxfp8", "mxfp6_fq": "mxfp6",
           "mxfp8_e5m2_fq": "mxfp8_e5m2", "mxfp6_e3m2_fq": "mxfp6_e3m2",
           "mxfp8_fqw": "mxfp8",
           "ssnf_kv_fq": "ssnf_kv", "ssnf_fq": "ssnf"}                # full-quant KV hook (prefill+decode)
    # configs whose WEIGHTS also go through fake-quant (no packed path for the OCP alternates)
    _WFQ = {"mxfp8_e5m2_fq": "mxfp8_e5m2", "mxfp6_e3m2_fq": "mxfp6_e3m2",
            "mxfp8_fqw": "mxfp8"}
    if cfg in _FQ:
        from ssnf_vllm.kv_fakequant import install
        hk = os.environ.get("SSNF_HK") if _FQ[cfg] in ("ssnf", "ssnf_kv") else None
        print(f"  [{cfg}] KV fake-quant hooks:", install(llm, _FQ[cfg], hk), flush=True)
    wfq = os.environ.get("WFQ") or _WFQ.get(cfg)      # WFQ=<fmt> forces it for equivalence checks
    if wfq:
        from ssnf_vllm.kv_fakequant import install_weight
        print(f"  [{cfg}] weight fake-quant ({wfq}) layers:", install_weight(llm, wfq), flush=True)
    return llm


if __name__ == "__main__":
    cfg = os.environ.get("CONFIG", "bf16")
    ds = load_dataset("openai/gsm8k", "main")
    test = ds["test"] if N == 0 else ds["test"].select(range(min(N, len(ds["test"]))))
    preamble = build_fewshot(ds["train"])
    prompts = [f"{preamble}\n\nQuestion: {q}\nAnswer:" for q in test["question"]]
    golds = [norm(_GOLD.search(a).group(1)) for a in test["answer"]]

    t0 = time.time()
    llm = make_engine(cfg)
    load_s = time.time() - t0
    sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK, stop=["\nQuestion:", "\n\n"])

    t = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t

    strict_ok = flex_ok = 0
    fails = []
    for i, o in enumerate(outs):
        gen = o.outputs[0].text
        g = golds[i]
        sm = _STRICT.search(gen)
        strict = norm(sm.group(1)) if sm else None
        nums = _NUM.findall(gen)
        flex = norm(nums[-1]) if nums else None
        s_hit = (g is not None and strict is not None and abs(strict - g) < 1e-4)
        f_hit = (g is not None and flex is not None and abs(flex - g) < 1e-4)
        strict_ok += s_hit
        flex_ok += f_hit
        if not f_hit and len(fails) < 5:
            fails.append({"q": test["question"][i][:70], "gold": g,
                          "flex": flex, "gen_tail": gen[-120:].replace("\n", " ")})

    n = len(outs)
    # binomial standard error of a proportion: sqrt(p(1-p)/n) — reliability comes from n, not reruns
    def binom_se(k: int) -> float:
        """Binomial standard error of a proportion k/n."""
        p = k / n
        return (p * (1 - p) / n) ** 0.5
    res = {"cfg": cfg, "model": MODEL, "nshot": NSHOT, "n": n,
           "strict_acc": round(strict_ok / n, 4), "strict_se": round(binom_se(strict_ok), 4),
           "flex_acc": round(flex_ok / n, 4), "flex_se": round(binom_se(flex_ok), 4),
           "load_s": round(load_s, 1), "gen_s": round(dt, 1)}
    print(json.dumps(res, indent=2))
    print(f"  strict {res['strict_acc']:.4f} ± {res['strict_se']:.4f}  "
          f"flex {res['flex_acc']:.4f} ± {res['flex_se']:.4f}  (n={n})")
    for f in fails:
        print("  FAIL", f)
    if os.environ.get("CP_OUT"):
        json.dump(res, open(os.environ["CP_OUT"], "w"))
