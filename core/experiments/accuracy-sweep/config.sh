#!/bin/bash
# Shared configuration for the per-plane (l_b x g_size) sweep. Edit here; run.sh reads these.

MODELS=${MODELS:-"llama31 mistral llama2"}
PLANES=${PLANES:-"w k v"}
LB_GRID=${LB_GRID:-"1,2,3,4,5,6,7"}
GS_GRID=${GS_GRID:-"1,2,4,8,16,32"}

# Short tag -> HF model id and calibration file (calib/, see README §3.4).
declare -A HFID=( [llama31]=NousResearch/Meta-Llama-3.1-8B
                  [mistral]=mistralai/Mistral-7B-v0.3
                  [llama2]=NousResearch/Llama-2-7b-hf )
declare -A HKF=(  [llama31]=H_K_metallama318b.pt
                  [mistral]=H_K_mistral7bv03.pt
                  [llama2]=H_K_llama27bhf.pt )

# Fixed protocol: no encoder search (deployed), calibration = wikitext-2 train 128x2048.
export SSNF_W_SCALE_OFFS=0 SSNF_KV_RADIUS=0 SSNF_CALIB_SPLIT=train SSNF_CALIB_W=128 HF_HUB_OFFLINE=1
