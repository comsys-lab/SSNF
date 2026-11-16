"""LongBench accuracy eval (in-process vLLM): official templates / gen lengths / metrics, middle-
truncation; 12 tasks = 8 KIVI tasks + 4 gap-fillers. KV path = fake-quant hooks on prefill AND decode
— STRICTER than the deployed serving path, scores are a lower bound; do not relax it. Env: CONFIG ∈
{bf16, fp8, mxint8_kv, ssnf_kv, ssnf, *_fq}, TASKS=<comma list>, NSAMP=N cap, LB_MAXLEN, LB_CHAT, CP_OUT.
Run: CONFIG=bf16 CUDA_VISIBLE_DEVICES=0 CP_OUT=/tmp/lb_bf16.json python eval_serving/eval_longbench.py  (SERVING environment)
"""
from __future__ import annotations

import json, math, os, re, statistics, string, time, zipfile
from collections import Counter
from difflib import SequenceMatcher
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import ssnf_vllm  # noqa: F401 — registers ssnf/mxint8 quant + SSNF KV backend
from typing import TYPE_CHECKING, Any
from vllm import LLM, SamplingParams
from huggingface_hub import hf_hub_download
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

MODEL = os.environ.get("SSNF_MODEL", "NousResearch/Meta-Llama-3.1-8B")
# LongBench truncation length (KIVI: 4096 for most models, 8192 for Mistral)
MAXLEN = int(os.environ.get("LB_MAXLEN", "8192" if "mistral" in MODEL.lower() else "4096"))
NSAMP = int(os.environ.get("NSAMP", "0"))          # 0 → full task set
# 12-task set = the 8 KIVI tasks + 4 added to close the category gaps (multi-doc QA 0/3 and synthetic
# 0/2 under the KIVI subset; those two categories are exactly where KV compression hurts most, and
# KIVI covered them with NIAH instead).
TASKS = ["qasper", "qmsum", "multi_news", "trec", "triviaqa", "samsum", "lcc", "repobench-p",
         "hotpotqa", "2wikimqa", "musique", "passage_retrieval_en"]
# instruct/chat models need the chat template on the generative tasks (official LongBench build_chat);
# the few-shot and code tasks stay in raw in-context format. Auto-on if the model id says instruct/chat.
_uc = os.environ.get("LB_CHAT", "auto")
USE_CHAT = (("instruct" in MODEL.lower()) or ("chat" in MODEL.lower())) if _uc == "auto" else (_uc == "1")
NO_CHAT_TASKS = {"trec", "triviaqa", "samsum", "lcc", "repobench-p"}

# --- official LongBench per-task prompt templates (dataset2prompt) ---
DATASET2PROMPT = {
    "qasper": ("You are given a scientific article and a question. Answer the question as concisely as "
               "you can, using a single phrase or sentence if possible. If the question cannot be answered "
               "based on the information in the article, write \"unanswerable\". If the question is a yes/no "
               "question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
               "Article: {context}\n\n Answer the question based on the above article as concisely as you "
               "can, using a single phrase or sentence if possible. If the question cannot be answered based "
               "on the information in the article, write \"unanswerable\". If the question is a yes/no "
               "question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
               "Question: {input}\n\nAnswer:"),
    "qmsum": ("You are given a meeting transcript and a query containing a question or instruction. Answer "
              "the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based "
              "on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:"),
    "multi_news": ("You are given several news passages. Write a one-page summary of all news. \n\nNews:\n"
                   "{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"),
    "trec": ("Please determine the type of the question below. Here are some examples of questions.\n\n"
             "{context}\n{input}"),
    "triviaqa": ("Answer the question based on the given passage. Only give me the answer and do not output "
                 "any other words. The following are some examples.\n\n{context}\n\n{input}"),
    "samsum": ("Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
               "{context}\n\n{input}"),
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
    # --- multi-document QA (all three share one official template) ---
    "hotpotqa": ("Answer the question based on the given passages. Only give me the answer and do not "
                 "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
                 "question based on the given passages. Only give me the answer and do not output any "
                 "other words.\n\nQuestion: {input}\nAnswer:"),
    "2wikimqa": ("Answer the question based on the given passages. Only give me the answer and do not "
                 "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
                 "question based on the given passages. Only give me the answer and do not output any "
                 "other words.\n\nQuestion: {input}\nAnswer:"),
    "musique": ("Answer the question based on the given passages. Only give me the answer and do not "
                "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
                "question based on the given passages. Only give me the answer and do not output any "
                "other words.\n\nQuestion: {input}\nAnswer:"),
    # --- synthetic retrieval ---
    "passage_retrieval_en": ("Here are 30 paragraphs from Wikipedia, along with an abstract. Please "
                             "determine which paragraph the abstract is from.\n\n{context}\n\nThe "
                             "following is an abstract.\n\n{input}\n\nPlease enter the number of the "
                             "paragraph that the abstract is from. The answer format must be like "
                             "\"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: "),
}
# official per-task max generation length (dataset2maxlen)
DATASET2MAXLEN = {"qasper": 128, "qmsum": 512, "multi_news": 512, "trec": 64,
                  "triviaqa": 32, "samsum": 128, "lcc": 64, "repobench-p": 64,
                  "hotpotqa": 32, "2wikimqa": 32, "musique": 32, "passage_retrieval_en": 32}


# --- official LongBench metrics ---
def normalize_answer(s: str) -> str:
    """Official LongBench answer normalization: lowercase, strip punctuation/articles/extra spaces."""
    def remove_articles(t: str) -> str: return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t: str) -> str: return " ".join(t.split())
    def remove_punc(t: str) -> str: return "".join(ch for ch in t if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1(pred_tokens: list[str], gt_tokens: list[str]) -> float:
    """Token-level F1 between prediction and ground-truth token lists."""
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **kw: Any) -> float:
    """Official LongBench QA metric: normalized token F1."""
    p, g = normalize_answer(prediction).split(), normalize_answer(ground_truth).split()
    if not p or not g:
        return 0.0
    return _f1(p, g)


def classification_score(prediction: str, ground_truth: str, all_classes: list[str] | None = None, **kw: Any) -> float:
    """Official LongBench classification metric: credit split over matched class labels."""
    em = [c for c in (all_classes or []) if c in prediction]
    for m in list(em):
        if m in ground_truth and m != ground_truth:
            em.remove(m)
    return 1.0 / len(em) if (em and ground_truth in em) else 0.0


def code_sim_score(prediction: str, ground_truth: str, **kw: Any) -> float:
    """Official LongBench code metric: edit similarity of the first code line vs ground truth."""
    pred = ""
    for line in prediction.lstrip("\n").split("\n"):
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            pred = line
            break
    return SequenceMatcher(None, pred, ground_truth).ratio()


try:
    from rouge import Rouge
    _rouge = Rouge()

    def rouge_score(prediction: str, ground_truth: str, **kw: Any) -> float:
        """ROUGE-L F1 via the `rouge` package (exact LongBench parity)."""
        if not prediction.strip() or not ground_truth.strip():
            return 0.0
        try:
            return _rouge.get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"]
        except Exception:
            return 0.0
    _ROUGE_IMPL = "rouge-pkg"
except ImportError:
    def _lcs(a: list[str], b: list[str]) -> int:
        """Length of the longest common subsequence of two token lists."""
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
        return dp[len(a)][len(b)]

    def rouge_score(prediction: str, ground_truth: str, **kw: Any) -> float:
        """ROUGE-L F1 fallback computed from the LCS (no `rouge` package installed)."""
        p, g = prediction.split(), ground_truth.split()
        if not p or not g:
            return 0.0
        lcs = _lcs(p, g)
        if lcs == 0:
            return 0.0
        prec, rec = lcs / len(p), lcs / len(g)
        return 2 * prec * rec / (prec + rec)
    _ROUGE_IMPL = "lcs-fallback"

def retrieval_score(prediction: str, ground_truth: str, **kw: Any) -> float:
    """Official LongBench passage_retrieval metric: fraction of emitted numbers equal to the gold paragraph id."""
    m = re.findall(r"Paragraph (\d+)", ground_truth)
    if not m:
        return 0.0
    gold = m[0]
    nums = re.findall(r"\d+", prediction)
    if not nums:
        return 0.0
    return float(sum(1 for n in nums if n == gold) / len(nums))


DATASET2METRIC = {"qasper": qa_f1_score, "qmsum": rouge_score, "multi_news": rouge_score,
                  "trec": classification_score, "triviaqa": qa_f1_score, "samsum": rouge_score,
                  "lcc": code_sim_score, "repobench-p": code_sim_score,
                  "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score, "musique": qa_f1_score,
                  "passage_retrieval_en": retrieval_score}
FIRST_LINE = {"trec", "triviaqa", "samsum"}       # official: take first line of the prediction


def make_engine(cfg: str) -> LLM:
    """Build the vLLM engine for CONFIG cfg, installing KV/weight fake-quant hooks where needed."""
    kw = dict(model=MODEL, dtype="bfloat16", enforce_eager=True,
              max_model_len=MAXLEN + 640, gpu_memory_utilization=0.9)
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
        pass                                                          # KV-only (optional)
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


def truncate_middle(tok: PreTrainedTokenizerBase, prompt: str, max_length: int) -> str:
    """LongBench middle-truncation: keep the first and last half of the tokenized prompt."""
    ids = tok.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_length:
        return prompt
    half = max_length // 2
    return tok.decode(ids[:half], skip_special_tokens=True) + tok.decode(ids[-half:], skip_special_tokens=True)


_ZIP = {"path": None}


def load_task(task: str, nsamp: int = 0) -> list[dict[str, Any]]:
    """Load one LongBench task straight from THUDM/LongBench data.zip (datasets>=3.0 dropped the script loader)."""
    if _ZIP["path"] is None:
        _ZIP["path"] = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    rows = []
    with zipfile.ZipFile(_ZIP["path"]).open(f"data/{task}.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
            if nsamp and len(rows) >= nsamp:
                break
    return rows


if __name__ == "__main__":
    cfg = os.environ.get("CONFIG", "bf16")
    tasks = os.environ.get("TASKS", ",".join(TASKS)).split(",")
    t0 = time.time()
    llm = make_engine(cfg)
    load_s = time.time() - t0
    tok = llm.get_tokenizer()

    per_task, per_task_se, counts = {}, {}, {}
    t = time.time()
    for task in tasks:
        data = load_task(task, NSAMP)
        tmpl, gen_len, metric = DATASET2PROMPT[task], DATASET2MAXLEN[task], DATASET2METRIC[task]
        prompts, metas = [], []
        for ex in data:
            p = truncate_middle(tok, tmpl.format(input=ex["input"], context=ex["context"]), MAXLEN)
            prompts.append(p)
            metas.append((ex["answers"], ex.get("all_classes")))
        sp = SamplingParams(temperature=0.0, max_tokens=gen_len)
        if USE_CHAT and task not in NO_CHAT_TASKS:      # wrap generative tasks in the model chat template
            if getattr(tok, "chat_template", None):
                outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp, use_tqdm=False)
            else:                                        # tokenizer has no chat template (Llama-2-chat) →
                outs = llm.generate([f"[INST]{p}[/INST]" for p in prompts], sp)  # official LongBench [INST] wrap
        else:
            outs = llm.generate(prompts, sp)

        scores = []
        for o, (answers, all_classes) in zip(outs, metas):
            pred = o.outputs[0].text
            if task in FIRST_LINE:
                pred = pred.lstrip("\n").split("\n")[0]
            scores.append(max((metric(pred, gt, all_classes=all_classes) for gt in answers), default=0.0))
        # score in [0,100]; standard error of the mean = std/sqrt(n) (continuous metric, not binomial)
        se = 100 * statistics.stdev(scores) / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
        per_task[task] = round(100 * sum(scores) / len(scores), 2)
        per_task_se[task] = round(se, 2)
        counts[task] = len(scores)
        print(f"  {task:14s} {per_task[task]:6.2f} ± {per_task_se[task]:.2f}  (n={len(scores)})", flush=True)

    dt = time.time() - t
    avg = round(sum(per_task.values()) / len(per_task), 2) if per_task else 0.0
    # SE of the unweighted 8-task average, treating tasks as independent: sqrt(sum se_i^2)/k
    avg_se = round(sum(s * s for s in per_task_se.values()) ** 0.5 / len(per_task_se), 2) if per_task_se else 0.0
    res = {"cfg": cfg, "model": MODEL, "maxlen": MAXLEN, "chat": USE_CHAT, "rouge_impl": _ROUGE_IMPL,
           "per_task": per_task, "per_task_se": per_task_se, "avg": avg, "avg_se": avg_se,
           "counts": counts, "load_s": round(load_s, 1), "eval_s": round(dt, 1)}
    print(json.dumps({"cfg": cfg, "avg": avg, "avg_se": avg_se, "per_task": per_task}, indent=2))
    if os.environ.get("CP_OUT"):
        json.dump(res, open(os.environ["CP_OUT"], "w"), indent=2)
