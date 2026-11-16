"""Emit the campaign job files, one "TAG<TAB>shell command" line per job, each guarded on its own
output file so the queue is safe to re-run. Phases: zeroshot, zeroshot2, gsm8k, longbench, consistency, t4_joint.
Interpreters: PY_ACC (ACCURACY env) / PY_SRV (SERVING env), both default "python" (README §2). Fixed
env per job: HF_HUB_OFFLINE=1, SSNF_W_SCALE_OFFS=0, SSNF_KV_RADIUS=0, calibration = wikitext-2 train
128x2048, SSNF deployed config W l_b3/g8, K l_b3/g16, V l_b4/g32. Run: python gen_jobs.py <phase>.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMP = os.path.join(REPO, "campaign")
RAW = os.path.join(CAMP, "raw")            # per-job JSON results land here
CALIB = os.path.join(REPO, "calib")        # H_K_* (KV encoder) and H_w_* (weight encoder) files
VENV = os.environ.get("PY_ACC", "python")  # ACCURACY environment
VLLM = os.environ.get("PY_SRV", "python")  # SERVING environment

from experiment_config import (BASE, CHAT, COMMON, SSNF_KV, FORMATS, GSM8K_REDO,  # noqa: E402
                               CONSISTENCY_GSM8K, CONSISTENCY_LB)


def guard(out: str, cmd: str) -> str:
    """Wrap cmd so it exits early when the output file already exists and is non-empty."""
    return f'[ -s "{out}" ] && exit 0; {cmd}'


def zeroshot() -> list[str]:
    """Zero-shot 5-task jobs: 9 formats x 3 base models (HF path, ACCURACY env)."""
    out = []
    for tag, (model, hk, hw) in BASE.items():
        for _, zs in FORMATS:
            f = f"{RAW}/zeroshot__{tag}__{zs}.json"
            cmd = (f'cd "{REPO}" && '
                   f"{COMMON} CONFIG={zs} SSNF_MODEL={model} SSNF_HK=\"{CALIB}/{hk}\" "
                   f"SSNF_CALIB_W=128 CP_OUT=\"{f}\" {VENV} eval_accuracy/eval_zeroshot.py")
            out.append(f"zs_{tag}_{zs}\t{guard(f, cmd)}")
    return out


def zeroshot2() -> list[str]:
    """BoolQ + OpenBookQA + MMLU jobs completing the 8-task set (separate phase/output; the summary merges the two files)."""
    out = []
    for tag, (model, hk, hw) in BASE.items():
        for _, zs in FORMATS:
            f = f"{RAW}/zeroshot2__{tag}__{zs}.json"
            cmd = (f'cd "{REPO}" && '
                   f"{COMMON} ZS_TASKS=boolq,openbookqa,mmlu CONFIG={zs} "
                   f"SSNF_MODEL={model} SSNF_HK=\"{CALIB}/{hk}\" SSNF_CALIB_W=128 CP_OUT=\"{f}\" "
                   f"{VENV} eval_accuracy/eval_zeroshot.py")
            out.append(f"zs2_{tag}_{zs}\t{guard(f, cmd)}")
    return out


def gsm8k() -> list[str]:
    """GSM8K jobs: the 4 changed formats x 3 base models (vLLM path, SERVING env)."""
    out = []
    for tag, (model, hk, hw) in BASE.items():
        for cfg in GSM8K_REDO:
            f = f"{RAW}/gsm8k__{tag}__{cfg}.json"
            extra = f"{SSNF_KV} SSNF_PROFILE=gs8 SSNF_HFILE=\"{CALIB}/{hw}\" " if cfg == "ssnf_fq" else ""
            cmd = (f'cd "{REPO}" && '
                   f"{COMMON} PYTHONPATH=\"{REPO}\" VLLM_ENABLE_V1_MULTIPROCESSING=0 "
                   f"{extra}CONFIG={cfg} SSNF_MODEL={model} SSNF_HK=\"{CALIB}/{hk}\" N=0 "
                   f"CP_OUT=\"{f}\" {VLLM} eval_serving/eval_gsm8k.py")
            out.append(f"gsm_{tag}_{cfg}\t{guard(f, cmd)}")
    return out


def longbench() -> list[str]:
    """LongBench 12-task jobs: 9 formats x 3 instruct/chat models (vLLM path, SERVING env)."""
    out = []
    for tag, (model, hk, hw, maxlen) in CHAT.items():
        for cfg, _ in FORMATS:
            f = f"{RAW}/longbench12__{tag}__{cfg}.json"
            extra = f"{SSNF_KV} SSNF_PROFILE=gs8 SSNF_HFILE=\"{CALIB}/{hw}\" " if cfg == "ssnf_fq" else ""
            cmd = (f'cd "{REPO}" && '
                   f"{COMMON} PYTHONPATH=\"{REPO}\" VLLM_ENABLE_V1_MULTIPROCESSING=0 "
                   f"{extra}CONFIG={cfg} SSNF_MODEL={model} SSNF_HK=\"{CALIB}/{hk}\" LB_MAXLEN={maxlen} "
                   f"NSAMP=0 CP_OUT=\"{f}\" {VLLM} eval_serving/eval_longbench.py")
            out.append(f"lb_{tag}_{cfg}\t{guard(f, cmd)}")
    return out


def consistency() -> list[str]:
    """Emit the E4M3 element-codec convention pair (experiment_config.CONSISTENCY_GSM8K)."""
    out = []
    for tag, (model, hk, hw) in BASE.items():                       # GSM8K: both conventions
        for cfg in CONSISTENCY_GSM8K:
            f = f"{RAW}/gsm8k__{tag}__{cfg}.json"
            cmd = (f'cd "{REPO}" && '
                   f"{COMMON} PYTHONPATH=\"{REPO}\" VLLM_ENABLE_V1_MULTIPROCESSING=0 "
                   f"CONFIG={cfg} SSNF_MODEL={model} SSNF_HK=\"{CALIB}/{hk}\" N=0 "
                   f"CP_OUT=\"{f}\" {VLLM} eval_serving/eval_gsm8k.py")
            out.append(f"cons_gsm_{tag}_{cfg}\t{guard(f, cmd)}")
    for tag, (model, hk, hw, maxlen) in CHAT.items():               # LongBench: packed comes from B3
        f = f"{RAW}/longbench12__{tag}__{CONSISTENCY_LB}.json"
        cmd = (f'cd "{REPO}" && '
               f"{COMMON} PYTHONPATH=\"{REPO}\" VLLM_ENABLE_V1_MULTIPROCESSING=0 "
               f"CONFIG={CONSISTENCY_LB} SSNF_MODEL={model} SSNF_HK=\"{CALIB}/{hk}\" LB_MAXLEN={maxlen} "
               f"NSAMP=0 CP_OUT=\"{f}\" {VLLM} eval_serving/eval_longbench.py")
        out.append(f"cons_lb_{tag}_{CONSISTENCY_LB}\t{guard(f, cmd)}")
    return out


def t4_joint() -> list[str]:
    """Joint W+KV PPL jobs: the deployed config, the key-heavy alternative, and the two lower probes."""
    cfgs = [("deployed", 3, 8, 3, 16, 4, 32), ("e1prime", 3, 16, 3, 4, 4, 32),
            ("kprobe", 3, 8, 3, 32, 4, 32), ("vprobe", 3, 8, 3, 16, 4, 16)]
    out = []
    for tag, (model, hk, hw) in BASE.items():
        for name, wlb, wgs, klb, kgs, vlb, vgs in cfgs:
            f = f"{REPO}/eval_accuracy/logs/joint_{name}_{tag}.csv"
            cmd = (f'cd "{REPO}" && '
                   f"{COMMON} SSNF_MODEL={model} SSNF_HK=\"{CALIB}/{hk}\" "
                   f"SSNF_JOINT=1 SSNF_W_LB={wlb} SSNF_W_GS={wgs} SSNF_K_LB={klb} SSNF_K_GS={kgs} "
                   f"SSNF_V_LB={vlb} SSNF_V_GS={vgs} SSNF_CALIB_SPLIT=train SSNF_CALIB_W=128 "
                   f"SSNF_JOINT_OUT=\"{f}\" {VENV} eval_accuracy/sweep_joint.py")
            out.append(f"joint_{name}_{tag}\t{guard(f, cmd)}")
    return out


if __name__ == "__main__":
    which = sys.argv[1]
    fn = {"zeroshot": zeroshot, "zeroshot2": zeroshot2, "gsm8k": gsm8k, "longbench": longbench,
          "consistency": consistency, "t4_joint": t4_joint}[which]
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(os.path.join(REPO, "eval_accuracy", "logs"), exist_ok=True)
    path = f"{CAMP}/jobs_{which}.txt"
    with open(path, "w") as f:
        f.write("\n".join(fn()) + "\n")
    print(f"{path}: {len(fn())} jobs")
