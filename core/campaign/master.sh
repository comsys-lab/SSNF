#!/usr/bin/env bash
# Accuracy campaign — runs every phase back to back, unattended.
#
# Phase order is chosen so that nothing waits on a result it does not need:
#   1  T1   per-plane (l_b x g_size) PPL sweep, full window   (experiments/accuracy-sweep/run.sh,
#           launched separately)
#   2  B1   zero-shot 5-task, 9 formats x 3 base models
#   3  B2   GSM8K, the 4 changed formats x 3 base models
#   4  B3   LongBench 12-task, 9 formats x 3 instruct/chat models
#   5  T4   joint W+KV configurations x 3 models
#   6  T3   LongBench on the T1 frontier points  (generated after T1 lands)
#
# Every phase uses runq.sh, which hands jobs to whichever GPU is idle — the box may be shared, so
# the campaign expands and contracts with whatever capacity is free instead of reserving all four.
# All phases are resume-safe: finished outputs are skipped, so re-running master.sh continues.
#
# Interpreters: export PY_ACC (ACCURACY-env python) and PY_SRV (SERVING-env python); both default
# to "python". gen_jobs.py bakes them into the job commands.
set -u
CAMP=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$CAMP/.." && pwd)
PY=${PY_ACC:-python}
mkdir -p "$CAMP/logs"
MASTER=$CAMP/logs/campaign.log
say() { echo "$(date '+%m-%d %H:%M:%S') [master] $*" | tee -a "$MASTER"; }

say "campaign start"

# ---- 1. wait for the T1 sweep (launched by experiments/accuracy-sweep/run.sh) ----
# The later phases do not consume T1 output, so they start immediately and share the queue with
# whatever T1 jobs are still running -- runq.sh only ever takes an idle GPU. Only T3 waits for T1.
if [ "${WAIT_T1:-0}" = "1" ]; then
  say "phase T1: waiting for the per-plane PPL sweep to finish"
  while pgrep -f "sweep_ubgs.py" > /dev/null; do sleep 120; done
  say "phase T1: done"
else
  say "phase T1: running in parallel (later phases do not depend on it)"
fi
for f in "$REPO"/eval_accuracy/logs/{w,k,v}_*.csv; do
  [ -f "$f" ] && say "  $(basename "$f"): $(($(wc -l < "$f") - 1)) rows"
done

# ---- 2-5. the phases that do not depend on T1 output ----
for ph in zeroshot zeroshot2 gsm8k longbench consistency t4_joint; do
  say "phase $ph: generating jobs"
  "$PY" "$CAMP/gen_jobs.py" "$ph" | tee -a "$MASTER"
  bash "$CAMP/runq.sh" "$CAMP/jobs_$ph.txt" "$ph"
  say "phase $ph: complete"
done

# ---- 6. T3: LongBench on the frontier points that T1 defines ----
say "phase T3: waiting for T1 to finish before computing the frontier"
while pgrep -f "sweep_ubgs.py" > /dev/null; do sleep 120; done
say "phase T3: computing frontier from the T1 sweep"
"$PY" "$CAMP/frontier.py" | tee -a "$MASTER"
if [ -s "$CAMP/jobs_t3.txt" ]; then
  bash "$CAMP/runq.sh" "$CAMP/jobs_t3.txt" t3
  say "phase T3: complete"
else
  say "phase T3: no frontier jobs emitted — check frontier.py output"
fi

say "===== campaign complete ====="
"$PY" "$CAMP/summarize.py" | tee -a "$MASTER"
