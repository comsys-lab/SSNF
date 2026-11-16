#!/usr/bin/env bash
# Weight-only quantization of Llama-3.1-8B, WikiText-2 PPL: SSNF and AMS-Quant (AMS-Quant paper, arXiv:2510.16045).
# Reported results: ssnf_3_16_h (5-bit signed upper, 3-bit lower/16) and ams6_g3_mse (AMS-Quant FP5.33).
set -euo pipefail
cd "$(dirname "$0")/../.."                        # extended/accuracy
export PYTHONPATH=$PWD
data=${SSNF_DATA_DIR:-data}
python -m engine.ppl_ams_fair --model llama31 --model-id "${SSNF_MODEL_DIR:-NousResearch/Meta-Llama-3.1-8B}" \
    --calib-dir ../../core/calib --corpus-text "$data/wikitext/test.txt" \
    --methods bf16,ams6_g3_mse,ssnf_3_32_h,ssnf_3_16_h,ssnf_3_8_h,ssnf_4_4_h --stages 2,16,0 \
    --max-seconds 14400 --threads 4 --out experiments/ams_quant/output
