#!/usr/bin/env bash
# IFEval (541 prompts, deterministic decoding, up to 1,280 output tokens) with weights and KV caches quantized.
# Requires ../../third_party/fetch.sh (IFEval sources and NLTK data).
# Usage: run.sh [llama31i|mistrali]
set -euo pipefail
cd "$(dirname "$0")/../.."                        # extended/accuracy
export PYTHONPATH=$PWD
calib=../../core/calib
case ${1:-llama31i} in
  llama31i) model=(--model llama31i --model-id "${SSNF_MODEL_DIR:-meta-llama/Llama-3.1-8B-Instruct}") ;;
  mistrali) model=(--model mistrali --model-id "${SSNF_MODEL_DIR:-mistralai/Mistral-7B-Instruct-v0.3}"
                   --hw $calib/H_w_mistrali.pt --hk $calib/H_K_mistral7binstructv03.pt) ;;
  *) echo "usage: run.sh [llama31i|mistrali]" >&2; exit 2 ;;
esac
python -m tasks.adapters --adapter-task ifeval --adapter-limit 0 --adapter-ifeval-max-new 1280 \
    "${model[@]}" --calib-dir $calib --methods bf16,ssnf_f11,mxfp4 \
    --max-seconds 86400 --threads 4 --out "experiments/ifeval/output/${1:-llama31i}"
