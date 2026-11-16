"""Campaign status: what has landed, what is still missing. Safe to run at any time.
In: eval_accuracy/logs/*.csv (sweeps) + campaign/raw/*.json (task results); when a directory has
no runs of your own it falls back to the shipped logs_subset/ copies, so the accuracy tables
reproduce without a GPU (same pattern as parse.py / make_tables.py). Out: stdout table."""
from __future__ import annotations

import csv
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _with_fallback(primary: str, shipped: str, pattern: str) -> str:
    """Return primary if it holds any file matching pattern, else the shipped logs_subset dir."""
    if glob.glob(os.path.join(primary, pattern)):
        return primary
    return shipped


SWEEP = _with_fallback(os.path.join(REPO, "eval_accuracy", "logs"),
                       os.path.join(REPO, "logs_subset", "sweep"), "*.csv")
RAW = _with_fallback(os.path.join(REPO, "campaign", "raw"),
                     os.path.join(REPO, "logs_subset", "task"), "*.json")
GRID = 7 * 6          # l_b 1..7 x g_size {1,2,4,8,16,32}


def line(s: str = "") -> None:
    """Print one output line."""
    print(s)


def main() -> None:
    """Print the campaign status table (T1/T4 sweeps + B1/B2/B3/T3 task results)."""
    line("=" * 78)
    line("T1 — per-plane (l_b x g_size) PPL sweep, full window")
    tot = done = 0
    for plane in ("w", "k", "v"):
        for model in ("llama31", "mistral", "llama2"):
            f = f"{SWEEP}/{plane}_{model}.csv"
            n = (sum(1 for _ in open(f)) - 1) if os.path.exists(f) else 0
            tot += GRID
            done += n
            bar = "#" * int(24 * n / GRID) + "." * (24 - int(24 * n / GRID))
            line(f"  {plane}-{model:8s} [{bar}] {n:2d}/{GRID}")
    line(f"  total {done}/{tot} ({100*done/tot:.0f}%)")

    line()
    line("T4 — joint W+KV configurations")
    for f in sorted(glob.glob(f"{SWEEP}/joint_*.csv") + glob.glob(f"{SWEEP}/validate_joint.csv")):
        r = list(csv.DictReader(open(f)))
        if r:
            r = r[0]
            line(f"  {os.path.basename(f)[6:-4]:22s} W {r['w_bpe']}b / KV {r['kv_bpe']}b  "
                 f"KV-only {float(r['d_kv_pct']):+.2f}%  W+KV {float(r['d_wkv_pct']):+.2f}%")

    for label, pat, key in (("B1 — zero-shot 5-task", "zeroshot__*.json", "avg"),
                            ("B2 — GSM8K", "gsm8k__*.json", "strict_acc"),
                            ("B3 — LongBench 12-task", "longbench12__*.json", "avg"),
                            ("T3 — LongBench frontier", "longbench12_frontier__*.json", "avg")):
        fs = sorted(glob.glob(f"{RAW}/{pat}"))
        line()
        line(f"{label}  ({len(fs)} results)")
        for f in fs:
            try:
                d = json.load(open(f))
            except Exception:
                line(f"  {os.path.basename(f)}: UNREADABLE")
                continue
            v = d.get(key)
            if v is None:
                for k in ("avg", "strict_acc", "acc"):
                    if k in d:
                        v = d[k]
                        break
            se = d.get("avg_se", d.get("strict_se", ""))
            se = f" ± {se:.2f}" if isinstance(se, (int, float)) else ""
            line(f"  {os.path.basename(f)[:-5]:46s} {v}{se}")
    line("=" * 78)


if __name__ == "__main__":
    main()
