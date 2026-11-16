#!/usr/bin/env bash
# Llama-2-13B WikiText-2 PPL: BF16 and SSNF (W and K), with BF16 weights on two GPUs.
# SSNF-W, SSNF-K, SSNF-V, and the quantization methods are the same as for the 7B and 8B models.
set -euo pipefail
cd "$(dirname "$0")/../.."                        # extended/accuracy
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
data=${SSNF_DATA_DIR:-data}
python -m engine.ppl_sharded_matched --model llama213 --model-id "${SSNF_MODEL_DIR:-meta-llama/Llama-2-13b-hf}" \
    --hw calib/H_w_llama213.pt --hk calib/H_K_llama213.pt \
    --corpus-text "$data/wikitext/test.txt" --train-text "$data/wikitext/train.txt" --map-calib-windows 32 \
    --methods bf16,ssnf_f11 --max-windows 0 --dtype bfloat16 \
    --max-seconds 20000 --threads 4 --out experiments/llama2_13b/output
