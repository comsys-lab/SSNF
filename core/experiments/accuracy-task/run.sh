#!/usr/bin/env bash
# Zero-shot / GSM8K / LongBench through the campaign queue.
# Zero-shot runs in the ACCURACY environment; GSM8K and LongBench need the SERVING environment with
# the extension built.
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0

# PY_ACC / PY_SRV select the ACCURACY- and SERVING-environment interpreters for the generated
# jobs (README §2); both default to "python".
for STAGE in zeroshot zeroshot2 gsm8k longbench consistency t4_joint; do
  python "$REPO/campaign/gen_jobs.py" "$STAGE"
  bash   "$REPO/campaign/runq.sh" "$REPO/campaign/jobs_${STAGE}.txt" "$STAGE"
done
