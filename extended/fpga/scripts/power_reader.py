#!/usr/bin/env python3
"""Read actual XRT board sensors through xbutil, preserving the exact JSON report."""
import json
import subprocess
import time
from pathlib import Path
from analyze import board_power


def sample(device,output,index):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    raw=output/f'sensor_{index:06d}.json'
    if raw.exists():raise FileExistsError(raw)
    before=time.monotonic_ns()
    result=subprocess.run(['xbutil','examine','--device',device,'--report','electrical','thermal',
                           '--format','JSON','--output',str(raw)],text=True,capture_output=True,timeout=15)
    after=time.monotonic_ns()
    if result.returncode:
        (output/f'sensor_{index:06d}.stderr').write_text(result.stderr+result.stdout)
        raise RuntimeError('xbutil sensor read failed; inspect sensor stderr')
    report=json.loads(raw.read_text())
    devices=[d for d in report['devices'] if d.get('device_id')==device]
    if len(devices)!=1:raise ValueError('Sensor report device BDF mismatch')
    d=devices[0]
    row={'action':'telemetry','before_ns':before,'after_ns':after,
         'electrical':d['electrical'],'thermal':d.get('thermals',[]),
         'device':device,'raw_report':str(raw)}
    board_power(row)
    if str(d['electrical'].get('power_consumption_warning','false')).lower()=='true':
        raise RuntimeError('Card reports power warning')
    return row


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--device',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=40)
    p.add_argument('--period',type=float,default=.25)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'samples.jsonl').open('x') as log:
        for n in range(args.samples):
            before=time.monotonic()
            row=sample(args.device,args.output/'raw',n)
            log.write(json.dumps(row)+'\n');log.flush()
            remaining=args.period-(time.monotonic()-before)
            if remaining>0:time.sleep(remaining)
