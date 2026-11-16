#!/usr/bin/env bash
# Llama-2-13B serving on one 24 GB GPU: SSNF and MXFP6 (weights and KV caches), 2K and 16K inputs,
# 256 output tokens. For each format, bundle.py finds the maximum batch size that completes all decode
# steps in GPU memory, then measures batch size 1 and the maximum batch size (three runs each).
# preload_patch.py keeps the BF16 weights in CPU memory; vLLM moves one layer at a time to the GPU, where
# the SSNF and MXFP6 quantization of core/ quantizes the layer. Inference runs on the GPU, and Triton
# autotuning runs only for the measured batch sizes.
# Requires: SSNF CUDA kernels built by ../prepare.py build, SSNF_MODEL_DIR = local Llama-2-13b-hf directory.
set -euo pipefail
cd "$(dirname "$0")"
: "${SSNF_GPU_UUID:?set SSNF_GPU_UUID to the target GPU (nvidia-smi -L)}"
: "${SSNF_MODEL_DIR:?set SSNF_MODEL_DIR to a local Llama-2-13b-hf directory}"
for ctx in 2048 16384; do
  for fmt in ssnf_wkv mxfp6w_wkv; do
    smoke="output/smoke_${ctx}_${fmt}"                # prerequisite of the measurement (128 input, 8 output tokens)
    [ -f "$smoke/summary.json" ] || python bundle_run.py --gpu-uuid "$SSNF_GPU_UUID" --format "$fmt" --stage smoke \
        --mode fixed --input-tokens 128 --output-tokens 8 --batches 1 --trial-timeout 900 --total-timeout 900 --out "$smoke"
    python bundle_run.py --gpu-uuid "$SSNF_GPU_UUID" --format "$fmt" --stage pilot --mode fixed \
        --input-tokens "$ctx" --output-tokens 256 --batches 1 --repeats 1 --warmups 1 \
        --trial-timeout 900 --total-timeout 3600 --after "$smoke" --out "output/measure_${ctx}_${fmt}"
  done
done
