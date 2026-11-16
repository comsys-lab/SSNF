#!/usr/bin/env bash
# Llama-2-7B WikiText-2 PPL with KV cache quantization: BF16, SSNF, KIVI-2/4, and KVQuant-3/4
# (original code of KIVI and KVQuant; KVQuant keeps 1% FP16 outliers).
# Usage: run.sh WORK_ROOT
#   Prepare first:  ../../../third_party/fetch.sh WORK_ROOT
#                   python -m venv WORK_ROOT/envs/{baseline,kivi,kvquant}, then pip install -r envs/<name>.txt
#                   build KIVI's CUDA/Triton kernels in WORK_ROOT/sources/KIVI (see its README)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "${1:?usage: run.sh WORK_ROOT}" && pwd)
cd "$here/../.."                                  # extended/accuracy
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$root/logs"

"$root/envs/baseline/bin/python" -m engine.kv_official_prepare --root "$root" --action inputs
"$root/envs/baseline/bin/python" -m unittest unit_tests.test_external_ppl -v
"$root/envs/kivi/bin/python" -m engine.kv_official_validate --root "$root"
CUDA_VISIBLE_DEVICES=0,1 "$root/envs/kivi/bin/python" -m engine.ppl_kivi_sharded --root "$root" --validate-sharding
if [ ! -f "$root/artifacts/quantizers4.pickle" ]; then
  "$root/envs/kvquant/bin/python" -m engine.kv_official_prepare --root "$root" --action fisher
  "$root/envs/kvquant/bin/python" -m engine.kv_official_parallel --root "$root" --action collect --activations "$root/activations"
  "$root/envs/kvquant/bin/python" -m engine.kv_official_parallel --root "$root" --action fit --activations "$root/activations" --workers 8
fi
"$root/envs/baseline/bin/python" -m engine.kv_official_run --root "$root" --methods bf16,ssnf,kivi2,kivi4,kvquant3,kvquant4
"$root/envs/baseline/bin/python" -m engine.kv_official_report --root "$root"
