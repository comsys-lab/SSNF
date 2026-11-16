#!/bin/bash
# Serving throughput: each format is measured at the largest batch it can hold (saturation).
# One model replica per GPU; every GPU stays loaded so the measurement reflects the deployed
# data-parallel setting. Results append to results/raw_g<gpu>.txt; parse.py turns them into
# the normalized table and the figure data.
#
#   bash run.sh                 # full sweep, ROUNDS repetitions
#   ROUNDS=1 bash run.sh        # quick single pass (~41 min on 4 GPUs)
#   FORMATS="bf16 ssnf_wkv" bash run.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
source "$HERE/config.sh"
export PYTHONPATH=${PYTHONPATH:-$ROOT}
mkdir -p "$HERE/results"
read -ra GPU_ARR <<< "$GPUS"; read -ra PAIR_ARR <<< "$PAIRS"; read -ra FMT_ARR <<< "$FORMATS"

one_run() {  # gpu, format, context, round
  local g=$1 cfg=$2 IN=$3 r=$4 N
  for p in "${PAIR_ARR[@]}"; do [ "${p%%:*}" = "$IN" ] && N=${p##*:}; done
  local EX=""; [ "$cfg" = "ssnf_wkv" ] && EX="SSNF_KV_ASYM=$SSNF_KV_ASYM SSNF_PROFILE=$SSNF_PROFILE"
  echo "### cfg=$cfg ctx=$IN gpu=$g round=$r" >> "$HERE/results/raw_g${g}.txt"
  env $EX CUDA_VISIBLE_DEVICES=$g CONFIG=$cfg N=$N IN=$IN OUT=$OUT_TOKENS MAXSEQ=$MAXSEQ \
      GPU_UTIL=$GPU_UTIL MODEL=$MODEL python "$ROOT/eval_serving/serving_throughput.py" \
      2> "$HERE/results/log_${cfg}_${IN}_r${r}.err" | grep "^THRUPUT" >> "$HERE/results/raw_g${g}.txt" \
      || echo "FAIL $cfg $IN r$r" >> "$HERE/results/raw_g${g}.txt"
}

# Work list: (format, context) pairs, longest context first so the GPUs finish together.
WORK=(); for p in "${PAIR_ARR[@]}"; do for f in "${FMT_ARR[@]}"; do WORK+=("$f:${p%%:*}"); done; done
IFS=$'\n' WORK=($(printf '%s\n' "${WORK[@]}" | sort -t: -k2 -rn)); unset IFS

echo "===== throughput sweep: ${#WORK[@]} runs x $ROUNDS rounds on ${#GPU_ARR[@]} GPUs ====="; date
for r in $(seq 1 "$ROUNDS"); do
  echo "--- round $r ---"; date
  i=0
  for g in "${GPU_ARR[@]}"; do
    (  for ((j=i; j<${#WORK[@]}; j+=${#GPU_ARR[@]})); do
         one_run "$g" "${WORK[j]%%:*}" "${WORK[j]##*:}" "$r"
       done ) &
    i=$((i+1))
  done
  wait
done
echo "===== done ====="; date
echo "Now run:  python parse.py"
