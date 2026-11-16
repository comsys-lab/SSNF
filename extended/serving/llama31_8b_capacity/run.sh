#!/usr/bin/env bash
# Llama-3.1-8B maximum batch size and decode throughput on one 24 GB GPU
# (BF16, MXFP6, SSNF; 2K/16K inputs, 256 output tokens; three runs, each with a different format sequence).
# The SSNF decode step time at 2K and batch size 8 (8 / 305.36 tokens/s = 26.20 ms) is the reference
# of the overhead estimate in ../kv_quant_overhead.
# Requires the serving environment (core/README.md, §2) and the SSNF CUDA kernels built by ../prepare.py build.
set -euo pipefail
cd "$(dirname "$0")"
: "${SSNF_GPU_UUID:?set SSNF_GPU_UUID to the target GPU (nvidia-smi -L)}"
# Smoke runs (128 input and 8 output tokens) that the pilot phase requires as prerequisites.
for pair in bf16:smoke_bf16_a02 mxfp6w_wkv:smoke_mxfp6_a01 ssnf_wkv:smoke_ssnf_a01; do
    fmt=${pair%%:*}; name=${pair#*:}
    [ -f "results/$name/summary.json" ] || python run.py --gpu-uuid "$SSNF_GPU_UUID" --format "$fmt" --stage smoke --out "results/$name"
done
for phase in pilots fixed capacity; do
    python campaign.py "$phase"
done
