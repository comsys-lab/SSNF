#!/usr/bin/env bash
# Usage: build.sh BUILD_DIR FPGA_PART PLATFORM_XPFM REPLICAS CLOCK_MHZ FORMAT
# Source the server-specific env.sh before invoking. No board operations here.
set -eo pipefail
ssnf_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (( $# != 6 )); then
    printf 'Usage: build.sh BUILD_DIR FPGA_PART PLATFORM_XPFM REPLICAS CLOCK_MHZ FORMAT\n' >&2
    exit 2
fi
ssnf_build=$1
ssnf_part=$2
ssnf_platform=$3
ssnf_replicas=${4:-16}
ssnf_clock=$5
ssnf_format=$6
[[ "$ssnf_format" =~ ^[0-2]$ ]] || exit 2
[[ "$ssnf_replicas" =~ ^([1-9]|1[0-6])$ ]] || { printf 'Replicas must be 1..16\n' >&2; exit 2; }
[[ "$ssnf_clock" =~ ^[0-9]+$ ]] || exit 2
for ssnf_tool in vivado v++ g++; do command -v "$ssnf_tool" >/dev/null || exit 1; done
[[ -f "$ssnf_platform" ]] || { printf 'Development platform .xpfm must be verified first\n' >&2; exit 1; }
[[ -n ${SSNF_RTL_GATE:-} ]] || { printf 'SSNF_RTL_GATE must name the successful RTL verification record\n' >&2; exit 1; }
python3 "$ssnf_root/scripts/check_source_gate.py" "$ssnf_root" "$SSNF_RTL_GATE"
[[ ! -e "$ssnf_build" ]] || { printf 'Use a new build directory; refusing to overwrite a build\n' >&2; exit 1; }
mkdir -p "$ssnf_build"
ssnf_build=$(cd -- "$ssnf_build" && pwd)
python3 "$ssnf_root/scripts/write_manifest.py" "$ssnf_root" "$ssnf_build/source_manifest.json"
python3 - "$ssnf_build" "$ssnf_replicas" "$ssnf_clock" "$ssnf_format" "$ssnf_platform" <<'PYCFG'
from pathlib import Path
import sys,json
p,n,m,f,platform=sys.argv[1:]
(Path(p)/'actual_config.json').write_text(json.dumps({'replicas':int(n),'target_clock_mhz':int(m),'format_id':int(f),'platform':platform},indent=2))
PYCFG
df -h "$ssnf_build" > "$ssnf_build/disk_before.txt"
vivado -mode batch -nojournal -log "$ssnf_build/package.log" -source "$ssnf_root/scripts/package_rtl.tcl" \
    -tclargs "$ssnf_root" "$ssnf_build" "$ssnf_part" "$ssnf_replicas" "$ssnf_format"
ssnf_cache=${SSNF_IP_CACHE:-"$ssnf_root/build/ip_cache"}
mkdir -p "$ssnf_cache"
v++ --link --target hw --platform "$ssnf_platform" --kernel_frequency "$ssnf_clock" --remote_ip_cache "$ssnf_cache" \
    --temp_dir "$ssnf_build/link_temp" --log_dir "$ssnf_build/link_logs" \
    --report_dir "$ssnf_build/link_reports" --save-temps \
    -o "$ssnf_build/ssnf_bench.xclbin" "$ssnf_build/ssnf_bench.xo" 2>&1 | tee "$ssnf_build/link.log"
ssnf_xrt_root=${XILINX_XRT:-/opt/xilinx/xrt}
g++ -std=c++17 -O2 -Wall -Wextra -I "$ssnf_xrt_root/include" \
    "$ssnf_root/host/run_bench.cpp" -L "$ssnf_xrt_root/lib" -Wl,-rpath,"$ssnf_xrt_root/lib" \
    -lxrt_coreutil -pthread -o "$ssnf_build/run_bench"
sha256sum "$ssnf_build/ssnf_bench.xclbin" > "$ssnf_build/image.sha256"
printf 'Build commands completed. Review implementation timing and DRC before loading.\n'
