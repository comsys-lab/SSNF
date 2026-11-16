#!/usr/bin/env bash
# Tensor-wise MXFP4/MXFP6 mixed precision within the SSNF storage (W 5.75, K 5.5, V 4.5 bits/value).
# 1) tensorwise.py: on 128 sequences of the WikiText-2 training split, assigns MXFP6 to the weight tensors
#    and per-layer K/V caches with the largest activation-weighted reconstruction error reduction per byte,
#    and MXFP4 to the remaining tensors, within the SSNF storage. Writes tensor_map.json.
# 2) verified.py: WikiText-2 test PPL with tensor_map.json; reconstructs the quantized values on the CPU
#    as a second check.
# 3) tensorwise_tasks.py: GSM8K (5-shot, 1,319 examples) with tensor_map.json.
# Usage: run.sh [llama31|mistral]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/../.."                                  # extended/accuracy
export PYTHONPATH=$PWD:$here
data=${SSNF_DATA_DIR:-data}
model=${1:-llama31}
out=$here/output/$model
common=(--method mx_tensor --model "$model" ${SSNF_MODEL_DIR:+--model-id "$SSNF_MODEL_DIR"} --calib-dir ../../core/calib
        --corpus-text "$data/wikitext/test.txt" --train-text "$data/wikitext/train.txt"
        --threads 4 --weight-chunk-blocks 131072)

python -m tensorwise "${common[@]}" --map-calib-windows 128 --max-windows 16 --max-seconds 10800 --out "$out/map"
export TENSOR_MAP_PATH=$out/map/tensor_map.json
python -m verified "${common[@]}" --max-windows 0 --max-seconds 7200 --out "$out/ppl"
if [ "$model" = llama31 ]; then
    python -m tensorwise_tasks --selected-task gsm "${common[@]}" --task-batch-size 8 --task-max-new-tokens 256 \
        --task-nshot 5 --task-max-samples 0 --max-seconds 21600 --out "$out/gsm8k"
fi
