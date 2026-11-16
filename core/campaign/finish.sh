#!/usr/bin/env bash
# Final sweep-up after (or alongside) master.sh: re-queues every job file so transient failures
# (GPU races, OOM from a shared box) are recovered — the per-job output guards skip finished work,
# so re-queuing is idempotent. Optionally restrict GPUs via $CAMP/allowed_gpus (one index per line).
set -u
CAMP=$(cd "$(dirname "$0")" && pwd)
MASTER=$CAMP/logs/campaign.log
mkdir -p "$CAMP/logs"
say() { echo "$(date '+%m-%d %H:%M:%S') [finish] $*" | tee -a "$MASTER"; }

if pgrep -f "campaign/master.sh" > /dev/null; then
  say "waiting for master.sh to finish before the retry sweep"
  while pgrep -f "campaign/master.sh" > /dev/null; do sleep 120; done
fi
for jf in t3 consistency zeroshot2 gsm8k longbench zeroshot t4_joint; do
  [ -s "$CAMP/jobs_$jf.txt" ] || continue
  say "re-queue $jf"
  bash "$CAMP/runq.sh" "$CAMP/jobs_$jf.txt" "retry_$jf"
done
say "===== retry sweep complete ====="
"${PY_ACC:-python}" "$CAMP/summarize.py" | tee -a "$MASTER"
