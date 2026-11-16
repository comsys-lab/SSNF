#!/usr/bin/env python3
import hashlib,json,subprocess,sys
from pathlib import Path
root,output=map(Path,sys.argv[1:])
files=[root/'config.json',root/'reference.py',root/'source_reference.py',root/'measurement_plan.json']
for directory,pattern in [('rtl','*.sv'),('tb','*.sv'),('host','*.cpp'),('scripts','*.py'),('scripts','*.sh'),('scripts','*.tcl')]:
    files.extend(sorted((root/directory).glob(pattern)))
revision=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],text=True,capture_output=True)
data={'git_base_revision':revision.stdout.strip(),'files':{
    str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
output.write_text(json.dumps(data,indent=2))
