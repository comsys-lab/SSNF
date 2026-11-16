#!/bin/bash
# Shared configuration for the throughput experiment (paper Figure: serving throughput).
# Edit here; run.sh reads these.

MODEL=${MODEL:-NousResearch/Meta-Llama-3.1-8B}

# Formats to compare. bf16 is the baseline; the other four are the quantized formats.
FORMATS=${FORMATS:-"bf16 mxint8_wkv mxfp8w_wkv mxfp6w_wkv ssnf_wkv"}

# Context length : number of offered requests. N is chosen so the engine stays saturated
# (N >> the batch any format can hold) while bounding the runtime.
PAIRS=${PAIRS:-"2048:256 4096:192 8192:128 16384:96"}

OUT_TOKENS=${OUT_TOKENS:-256}      # generated tokens per request
MAXSEQ=${MAXSEQ:-512}              # max_num_seqs; KV memory, not this cap, must bind
GPU_UTIL=${GPU_UTIL:-0.85}
ROUNDS=${ROUNDS:-5}                # repetitions per (format, context); the paper reports 5
GPUS=${GPUS:-"0 1 2 3"}            # one model replica per GPU (data parallel)

# SSNF configuration: W u3/gs8 (5.63 b) + K u3/gs16 (5.44 b) + V u4/gs32 (4.38 b)
export SSNF_KV_ASYM=${SSNF_KV_ASYM:-"3,16,4,32"}
export SSNF_PROFILE=${SSNF_PROFILE:-gs8}
# KV decode split-K multiplier. 32 is the measured optimum for every quantized format.
export SSNF_KV_SPLIT_MULT=${SSNF_KV_SPLIT_MULT:-32}
# Both formats run their fastest weight-GEMM path: SSNF_TRITON gates the fused Triton mixed-input
# GEMM for SSNF (quant.py) AND its fair MXINT8 twin (quant_mxint8.py). Explicit so the measured
# condition is pinned, not inherited from module defaults.
export SSNF_TRITON=${SSNF_TRITON:-1}
export VLLM_ATTENTION_BACKEND=CUSTOM
export VLLM_ENABLE_V1_MULTIPROCESSING=0
