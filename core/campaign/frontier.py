"""Pick the T3 frontier points from the T1 sweep and emit LongBench jobs for them.
In: eval_accuracy/logs/{k,v}_<model>.csv. Out: campaign/jobs_t3.txt (+ results to campaign/raw/).
Frontier = per plane/model, non-dominated on the DEPLOYED byte-aligned cost (packer rounds the shared
field up to a byte per block): lowest PPL at each aligned bpe, then the best FRONTIER_TOP inside the
band FRONTIER_LO..FRONTIER_HI.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMP = os.path.join(REPO, "campaign")
SWEEP = os.path.join(REPO, "eval_accuracy", "logs")
RAW = os.path.join(CAMP, "raw")
HW = os.path.join(REPO, "calib")
HK = os.path.join(REPO, "calib")
VLLM = os.environ.get("PY_SRV", "python")  # SERVING-environment python

BAND = (float(os.environ.get("FRONTIER_LO", "4.0")), float(os.environ.get("FRONTIER_HI", "6.5")))
TOP = int(os.environ.get("FRONTIER_TOP", "3"))

# LongBench runs on the instruct/chat models; map them to the base tag the sweep used.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_config import CHAT as _CHAT  # noqa: E402
CHAT = {base: (tag, *rest) for base, (tag, *rest) in zip(
    ("llama31", "mistral", "llama2"),
    ((t, *v) for t, v in _CHAT.items()))}
DEPLOY = {"k": (3, 16), "v": (4, 32)}          # the other KV plane stays at its deployed setting


def frontier(plane: str, model: str) -> list[dict[str, str]]:
    """Return the frontier sweep rows for one plane/model, sorted by aligned bits-per-element."""
    path = f"{SWEEP}/{plane}_{model}.csv"
    if not os.path.exists(path):
        return []
    rows = [r for r in csv.DictReader(open(path))]
    if not rows:
        return []
    best = {}
    for r in rows:
        a = float(r["bpe_aligned"])
        if not BAND[0] <= a <= BAND[1]:
            continue
        d = float(r["delta_pct"])
        if a not in best or d < float(best[a]["delta_pct"]):
            best[a] = r
    ranked = sorted(best.values(), key=lambda r: float(r["delta_pct"]))[:TOP]
    return sorted(ranked, key=lambda r: float(r["bpe_aligned"]))


def main() -> None:
    """Write campaign/jobs_t3.txt with one LongBench job per frontier point."""
    os.makedirs(RAW, exist_ok=True)
    jobs = []
    for base, (tag, model, hk, hw, maxlen) in CHAT.items():
        for plane in ("k", "v"):
            for r in frontier(plane, base):
                lb, g_size = int(r["lb"]), int(r["gs"])
                klb, kgs = (lb, g_size) if plane == "k" else DEPLOY["k"]
                vlb, vgs = (lb, g_size) if plane == "v" else DEPLOY["v"]
                name = f"{plane}lb{lb}gs{g_size}"
                out = f"{RAW}/longbench12_frontier__{tag}__{name}.json"
                cmd = (f"[ -s {out} ] && exit 0; cd {REPO} && HF_HUB_OFFLINE=1 PYTHONPATH={REPO} "
                       f"VLLM_ENABLE_V1_MULTIPROCESSING=0 SSNF_W_SCALE_OFFS=0 SSNF_KV_RADIUS=0 "
                       f"SSNF_K_LB={klb} SSNF_K_GS={kgs} SSNF_V_LB={vlb} SSNF_V_GS={vgs} "
                       f"SSNF_PROFILE=gs8 SSNF_HFILE={HW}/{hw} CONFIG=ssnf_fq SSNF_MODEL={model} "
                       f"SSNF_HK={HK}/{hk} LB_MAXLEN={maxlen} NSAMP=0 CP_OUT={out} "
                       f"{VLLM} eval_serving/eval_longbench.py")
                jobs.append(f"t3_{tag}_{name}\t{cmd}")
                print(f"  frontier {base}/{plane}: l_b={lb} g={g_size} "
                      f"aligned {r['bpe_aligned']}b  dPPL {r['delta_pct']}%")
    path = f"{CAMP}/jobs_t3.txt"
    with open(path, "w") as f:
        f.write("\n".join(jobs) + ("\n" if jobs else ""))
    print(f"{path}: {len(jobs)} jobs (band {BAND[0]}-{BAND[1]} aligned bpe, top {TOP} per plane)")


if __name__ == "__main__":
    main()
