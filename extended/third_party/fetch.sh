#!/usr/bin/env bash
# Fetch third-party baselines at the pinned commits and apply the compatibility patches.
# Usage: fetch.sh WORK_ROOT
#   WORK_ROOT/sources/{KIVI,KVQuant,omniserve}   upstream repositories (patched)
#   ../accuracy/tasks/vendor/                    Google IFEval sources, Python deps, NLTK data
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work=${1:?usage: fetch.sh WORK_ROOT}
mkdir -p "$work/sources"
src=$(cd -- "$work/sources" && pwd)

clone() {  # name url commit
    [ -d "$src/$1/.git" ] || git clone --quiet "$2" "$src/$1"
    git -C "$src/$1" checkout --quiet "$3"
    test "$(git -C "$src/$1" rev-parse HEAD)" = "$3"
}
clone KIVI      https://github.com/jy-yuan/KIVI.git          876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
clone KVQuant   https://github.com/SqueezeAILab/KVQuant.git  57a238357f0ffe50084670fcd5781c9848f80ea2
clone omniserve https://github.com/mit-han-lab/omniserve.git 02b2925aa6fa3b92b06316a1524b7f38922cd9c8

patch -d "$src" -p1 --forward < "$here/patches/kivi_kvquant.patch"
git -C "$src/omniserve" apply "$here/patches/omniserve.patch"

# Google IFEval: pinned files with SHA-256 checks, plus its Python dependencies and NLTK data.
vendor="$here/../accuracy/tasks/vendor"
mkdir -p "$vendor"
cp "$here/ifeval_sources.json" "$vendor/ifeval_sources.json"
python3 - "$here/ifeval_sources.json" "$vendor" <<'PY'
import hashlib, json, sys, urllib.request
from pathlib import Path
spec, vendor = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
for f in spec['files']:
    data = urllib.request.urlopen(f['url']).read()
    if hashlib.sha256(data).hexdigest() != f['sha256']:
        raise SystemExit('sha256 mismatch: ' + f['path'])
    out = vendor / f['path']
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
PY
python3 -m pip install --quiet --target "$vendor/python_deps" \
    absl-py==2.5.1 immutabledict==4.3.1 langdetect==1.0.9 nltk==3.9.2 rouge==1.0.1
python3 -c "import sys; sys.path.insert(0, '$vendor/python_deps'); import nltk; \
nltk.download('punkt', '$vendor/nltk_data', quiet=True); nltk.download('punkt_tab', '$vendor/nltk_data', quiet=True)"
echo "fetched into $src and $vendor"
