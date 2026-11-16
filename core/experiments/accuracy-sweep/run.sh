#!/usr/bin/env bash
# Per-plane sweep over the (l_b, g_size) grid.  Requires the ACCURACY environment; no build.
# All settings live in config.sh.
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
export PYTHONPATH="$REPO"

# sweep_ubgs.py always writes eval_accuracy/logs/<plane>_<tag>.csv (resume-safe per config).
OUT="$REPO/eval_accuracy/logs"; mkdir -p "$OUT"

for MODEL in $MODELS; do
  for PLANE in $PLANES; do
    [ -f "$OUT/${PLANE}_${MODEL}.csv" ] && { echo "skip ${PLANE}_${MODEL} (done)"; continue; }
    SSNF_SWEEP_PLANE=$PLANE \
    SSNF_SWEEP_LB=$LB_GRID SSNF_SWEEP_GS=$GS_GRID \
    SSNF_MODEL="${HFID[$MODEL]}" SSNF_HK="$REPO/calib/${HKF[$MODEL]}" SSNF_TAG=$MODEL \
    python "$REPO/eval_accuracy/sweep_ubgs.py" 2>&1 | tee "$OUT/${PLANE}_${MODEL}.log"
  done
done
