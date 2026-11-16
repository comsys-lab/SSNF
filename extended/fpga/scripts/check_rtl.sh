#!/usr/bin/env bash
set -euo pipefail
ssnf_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ssnf_out=${1:?output directory}
ssnf_vectors=${2:-"$ssnf_root/results/hardware_vectors_static"}
ssnf_ivl=("${IVERILOG:-iverilog}")
ssnf_vvp=("${VVP:-vvp}")
mkdir "$ssnf_out"
mkdir -p "$ssnf_out/source_snapshot/rtl" "$ssnf_out/source_snapshot/tb"
cp "$ssnf_root"/rtl/*.sv "$ssnf_out/source_snapshot/rtl/"
cp "$ssnf_root"/tb/*.sv "$ssnf_out/source_snapshot/tb/"
for ssnf_case in source dot32 bench2 bench16 static0 static1 static2; do
 ssnf_top="tb_$ssnf_case"
 ssnf_params=()
 ssnf_vec=source
 if [[ "$ssnf_case" == dot32 ]]; then ssnf_vec=vectors; fi
 if [[ "$ssnf_case" == static* ]]; then
  ssnf_top=tb_bench;ssnf_vec=controller
  ssnf_params=(-Ptb_bench.TEST_REPLICAS=2 -Ptb_bench.TEST_FORMAT="${ssnf_case#static}")
 fi
 if [[ "$ssnf_case" == bench* ]]; then
  ssnf_top=tb_bench;ssnf_vec=controller
  ssnf_params=(-Ptb_bench.TEST_REPLICAS="${ssnf_case#bench}")
 fi
 ssnf_sources=()
 for ssnf_src in fp_helpers.sv int_sum_tree.sv ssnf_dot32.sv mxint8_dot32.sv bf16_mul.sv fp32_add.sv bf16_dot32.sv input_source.sv ssnf_bench.sv; do ssnf_sources+=("$ssnf_out/source_snapshot/rtl/$ssnf_src"); done
 "${ssnf_ivl[@]}" -g2012 -Wall -s "$ssnf_top" "${ssnf_params[@]}" -o "$ssnf_out/$ssnf_case.vvp" "${ssnf_sources[@]}" "$ssnf_out/source_snapshot/tb/$ssnf_top.sv" >"$ssnf_out/$ssnf_case.compile.log" 2>&1
 timeout 180s "${ssnf_vvp[@]}" "$ssnf_out/$ssnf_case.vvp" "+VECTORS=$ssnf_vectors/$ssnf_vec.txt" | tee "$ssnf_out/$ssnf_case.log"
done
python3 - "$ssnf_root" "$ssnf_out" <<'PY'
from pathlib import Path
import hashlib,json,sys
root,out=map(Path,sys.argv[1:])
files=[p for d in ['rtl','tb'] for p in (root/d).glob('*.sv')]
assert all(p.read_bytes()==(out/'source_snapshot'/p.relative_to(root)).read_bytes() for p in files),'RTL changed during verification'
(out/'verification.json').write_text(json.dumps({'status':'rtl_simulation_passed','hardware_measured':False,'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'logs':{n:(out/(n+'.log')).read_text() for n in ['source','dot32','bench2','bench16','static0','static1','static2']}},indent=2))
PY
