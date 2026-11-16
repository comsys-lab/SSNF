#!/bin/bash
# Shared configuration for the capacity experiment (paper Figure: maximum batch size).
# Edit here; run.sh reads these.

MODEL=${MODEL:-NousResearch/Meta-Llama-3.1-8B}
FORMATS=${FORMATS:-"bf16 mxint8_wkv mxfp8w_wkv mxfp6w_wkv ssnf_wkv"}
IN_TOKENS=${IN_TOKENS:-2048}
OUT_TOKENS=${OUT_TOKENS:-256}
GPU=${GPU:-0}

# SSNF configuration: W l_b3/g8 (5.63 b) + K l_b3/g16 (5.44 b) + V l_b4/g32 (4.38 b)
export SSNF_KV_ASYM=${SSNF_KV_ASYM:-"3,16,4,32"}
export SSNF_PROFILE=${SSNF_PROFILE:-gs8}
export VLLM_ATTENTION_BACKEND=CUSTOM
export VLLM_ENABLE_V1_MULTIPROCESSING=0
