#!/usr/bin/env bash
# Generic GPU job queue for the accuracy campaign.
#
#   bash runq.sh <jobfile> <phase-name>
#
# Each line of <jobfile> is "TAG<TAB>COMMAND". The dispatcher polls nvidia-smi and hands the next
# pending job to whichever GPU is idle, so it shares the box with other users instead of reserving
# it. CUDA_VISIBLE_DEVICES is exported for the child; everything else the job needs must be in the
# command line itself.
#
# Idempotent: a job whose output marker already exists should exit immediately (each generator emits
# commands that check for their own output first), so re-running the queue resumes rather than redoes.
set -u
CAMP=$(cd "$(dirname "$0")" && pwd)
JOBFILE=$1
PHASE=$2
LOGD=$CAMP/logs/$PHASE
mkdir -p "$LOGD"
MASTER=$CAMP/logs/campaign.log

FREE_MIB=${FREE_MIB:-2000}
POLL=${POLL:-60}

say() { echo "$(date '+%m-%d %H:%M:%S') [$PHASE] $*" | tee -a "$MASTER"; }

mapfile -t JOBS < <(grep -v '^\s*$' "$JOBFILE")
say "queue start: ${#JOBS[@]} jobs"

declare -A PID_ON TAG_ON
next=0
while :; do
  for g in "${!PID_ON[@]}"; do
    if ! kill -0 "${PID_ON[$g]}" 2>/dev/null; then
      wait "${PID_ON[$g]}" 2>/dev/null; rc=$?
      say "GPU$g finished ${TAG_ON[$g]} (rc=$rc)"
      unset "PID_ON[$g]" "TAG_ON[$g]"
    fi
  done

  if [ "$next" -lt "${#JOBS[@]}" ]; then
    while read -r g used; do
      g=${g%,}; used=${used%,}
      [ "$next" -lt "${#JOBS[@]}" ] || break
      [ -n "${PID_ON[$g]:-}" ] && continue
      # optional GPU allowlist: if $CAMP/allowed_gpus exists, dispatch only to the listed indices.
      if [ -f "$CAMP/allowed_gpus" ]; then
        grep -qwE "$g" "$CAMP/allowed_gpus" || continue
      fi
      [ "$used" -lt "$FREE_MIB" ] || continue
      tag=$(printf '%s' "${JOBS[$next]}" | cut -f1)
      cmd=$(printf '%s' "${JOBS[$next]}" | cut -f2-)
      say "GPU$g start $tag"
      CUDA_VISIBLE_DEVICES=$g nohup bash -c "$cmd" > "$LOGD/$tag.log" 2>&1 &
      PID_ON[$g]=$!; TAG_ON[$g]=$tag
      next=$((next + 1))
      sleep 25                      # let it claim memory before probing the next GPU
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  fi

  [ "$next" -ge "${#JOBS[@]}" ] && [ "${#PID_ON[@]}" -eq 0 ] && break
  sleep "$POLL"
done
say "queue complete (${#JOBS[@]} jobs)"
