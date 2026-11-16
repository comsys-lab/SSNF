#!/bin/bash
# Maximum batch size: boot the engine once per format and read the KV-cache pool it reserves.
# Deterministic (pure memory accounting), so one run per format is enough.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
source "$HERE/config.sh"
export PYTHONPATH=${PYTHONPATH:-$ROOT}
mkdir -p "$HERE/results"; OUT="$HERE/results/capacity.txt"; : > "$OUT"
for cfg in $FORMATS; do
  EX=""; [ "$cfg" = "ssnf_wkv" ] && EX="SSNF_KV_ASYM=$SSNF_KV_ASYM SSNF_PROFILE=$SSNF_PROFILE"
  env $EX CUDA_VISIBLE_DEVICES=$GPU CONFIG=$cfg IN=$IN_TOKENS OUT=$OUT_TOKENS MODEL=$MODEL \
      python "$ROOT/eval_serving/serving_capacity.py" 2> "$HERE/results/log_${cfg}.err" \
      | grep "^CAP" | tee -a "$OUT" || echo "FAIL $cfg" | tee -a "$OUT"
done
echo; echo "pool tokens = num_gpu_blocks x block_size; max batch at a context = pool / (ctx + 256)"
