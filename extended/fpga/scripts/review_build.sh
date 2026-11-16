#!/usr/bin/env bash
# Read-only post-build review, no board load.
set -eo pipefail
ssnf_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ssnf_build=$1
ssnf_format=$2
ssnf_checkpoint=$(python3 - "$ssnf_build" <<'PY'
from pathlib import Path
import sys
paths=list((Path(sys.argv[1])/'link_temp/link/vivado/vpl/prj/prj.runs/impl_1').glob('*_routed.dcp'))
assert len(paths)==1,paths
print(paths[0])
PY
)
python3 "$ssnf_root/scripts/write_manifest.py" "$ssnf_root" "$ssnf_build/review_source_manifest.json"
vivado -mode batch -nojournal -log "$ssnf_build/review.log" -source "$ssnf_root/scripts/review_route.tcl" -tclargs "$ssnf_checkpoint" "$ssnf_build/review"
python3 "$ssnf_root/scripts/validate_build.py" "$ssnf_build" --replicas 16 --format-id "$ssnf_format"
python3 "$ssnf_root/scripts/collect_resources.py" "$ssnf_build/review/utilization_routed.rpt" "$ssnf_build/review/resources.json" --format-id "$ssnf_format" --replicas 16
