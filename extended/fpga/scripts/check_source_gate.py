#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
root,gate=map(Path,sys.argv[1:])
data=json.loads(gate.read_text())
if data.get('status')!='rtl_simulation_passed':raise SystemExit('RTL verification did not pass')
for name,digest in data['source_sha256'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise SystemExit('Source changed since RTL verification: '+name)
print('RTL source hashes match successful verification.')
