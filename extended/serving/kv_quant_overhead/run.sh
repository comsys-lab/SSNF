#!/usr/bin/env bash
# On-the-fly KV cache quantization overhead in decode attention (Llama-3.1-8B, layers 0/16/31,
# 2K/16K inputs, batch size 1/4/8). Each case measures attention time with and without quantization of
# the new token into the KV cache, for BF16, MXFP6, and SSNF, using the Q/K/V values of Llama-3.1-8B.
# Requires: SSNF CUDA kernels built by ../prepare.py build; WikiText-2 training split (../../accuracy/prepare_wikitext.py).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
: "${SSNF_GPU_UUID:?set SSNF_GPU_UUID to the target GPU (nvidia-smi -L)}"
export SSNF_DATA_DIR=${SSNF_DATA_DIR:-$here/../../accuracy/data}
python capture_layers.py --model L31       # 2K operands   -> operands/
python capture_layers16k.py --model L31    # 16K operands  -> operands16k/
python kv_actual.py                        # timings       -> kv_actual_full/results.json
