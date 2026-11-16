#!/usr/bin/env bash
# WikiText-2 PPL with weights and KV caches quantized (SSNF-W, SSNF-K, SSNF-V).
#   ssnf_f00  without W and K                      ssnf_f10  W: activation-weighted weight search
#   ssnf_f01  K: query-weighted key residuals      ssnf_f11  W and K
# Usage: run.sh [llama31|mistral]
set -euo pipefail
cd "$(dirname "$0")/../.."                        # extended/accuracy
export PYTHONPATH=$PWD
data=${SSNF_DATA_DIR:-data}
model=${1:-llama31}
python -m engine.ppl --model "$model" ${SSNF_MODEL_DIR:+--model-id "$SSNF_MODEL_DIR"} \
    --calib-dir ../../core/calib --corpus-text "$data/wikitext/test.txt" \
    --methods bf16,ssnf_f00,ssnf_f10,ssnf_f01,ssnf_f11 --max-windows 0 \
    --max-seconds 14400 --threads 4 --out "experiments/calibration_statistics/output/$model"
