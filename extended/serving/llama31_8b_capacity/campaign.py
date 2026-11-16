#!/usr/bin/env python3
"""Finite checked phase; each child is bounded and uses the shared GPU lock."""
import argparse
import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from protocol import save

HERE=Path(__file__).resolve().parent
PY=Path(sys.executable)
GPU=__import__('os').environ.get('SSNF_GPU_UUID','')
FMTS=['bf16','mxfp6w_wkv','ssnf_wkv']
LABELS={'bf16':'bf16','mxfp6w_wkv':'mxfp6','ssnf_wkv':'ssnf'}
SMOKES={'bf16':'smoke_bf16_a02','mxfp6w_wkv':'smoke_mxfp6_a01','ssnf_wkv':'smoke_ssnf_a01'}


def pilot_name(ctx,fmt): return f'pilot_{ctx//1024}k_{LABELS[fmt]}_a01'


def launch(name,fmt,stage,ctx,batches,after,mode='fixed',cycles=1,minimum=0,warmups=1):
    out=HERE/'results'/name
    if out.exists():
        if (out/'summary.json').exists() and json.loads((out/'summary.json').read_text()).get('run_valid'):
            print('REUSE',name,flush=True);return
        raise RuntimeError(f'Existing incomplete/failed run must be inspected before retry: {out}')
    cmd=[str(PY),str(HERE/'run.py'),'--gpu-uuid',GPU,'--format',fmt,'--stage',stage,'--mode',mode,
         '--input-tokens',str(ctx),'--output-tokens','256','--batches',batches,'--repeats','1',
         '--cycles',str(cycles),'--minimum-block-seconds',str(minimum),'--warmups',str(warmups),
         '--trial-timeout','600' if mode=='capacity' else '180','--total-timeout','1800',
         '--after',str(HERE/'results'/after),'--out',str(out)]
    with (HERE/'campaign_commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),name=name,command=cmd))+'\n')
    print('START',name,flush=True)
    subprocess.run(cmd,check=True,cwd=HERE)
    d=json.loads((out/'summary.json').read_text())
    if not d['run_valid']: raise RuntimeError(f'Run did not validate: {name}')
    if mode=='fixed' and stage=='measure':
        stability=[]
        for b in sorted({t['batch'] for t in d['trials']}):
            values=[json.loads((out/f'b{b}_warmup{w}'/'summary.json').read_text())['wall_s'] for w in range(warmups)]
            relative=abs(values[-1]-values[-2])/max(values[-1],values[-2]) if len(values)>=2 else None
            stability.append(dict(batch=b,warmup_wall_s=values,last_two_relative_difference=relative,
                                  passed=relative is not None and relative<=.15))
        save(out/'warmup_stability.json',stability)
        if not all(x['passed'] for x in stability): raise RuntimeError(f'Warmup stability needs investigation: {name}')
    print('PASS',name,[(t['batch'],t.get('fixed_batch_valid',t.get('capacity_verified'))) for t in d['trials']],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilots','fixed','capacity'])
    a=p.parse_args()
    if a.phase=='pilots':
        for ctx in [2048,16384]:
            for fmt in FMTS:
                launch(pilot_name(ctx,fmt),fmt,'pilot',ctx,'1,4,8' if ctx==2048 else '1',SMOKES[fmt])
        return
    cycles={}
    pilots={}
    for ctx in [2048,16384]:
        pilots[ctx]=[json.loads((HERE/'results'/pilot_name(ctx,f)/'summary.json').read_text()) for f in FMTS]
        if not all(p['run_valid'] for p in pilots[ctx]): raise RuntimeError('Missing valid pilot')
        fastest=min(t['all_batch_decode_wall_s'] for p in pilots[ctx] for t in p['trials'])
        cycles[ctx]=math.ceil(12/fastest)
        if cycles[ctx]>8: raise RuntimeError('Short pilot needs extra planning')
    freeze=HERE/'fixed_frozen.json'
    frozen=dict(cycles=cycles,minimum_seconds=10,rounds=3,warmups=3,
                orders=[FMTS,FMTS[1:]+FMTS[:1],FMTS[2:]+FMTS[:2]])
    if freeze.exists() and json.loads(freeze.read_text())!=json.loads(json.dumps(frozen)):
        raise RuntimeError('Frozen settings differ')
    if not freeze.exists(): save(freeze,frozen)
    if a.phase=='fixed':
        for r,order in enumerate(frozen['orders'],1):
            for ctx in [2048,16384]:
                for fmt in order:
                    launch(f'fixed_{ctx//1024}k_{LABELS[fmt]}_r{r}_a01',fmt,'measure',ctx,
                           '1,4,8' if ctx==2048 else '1',pilot_name(ctx,fmt),cycles=cycles[ctx],minimum=10,warmups=3)
    else:
        for ctx in [16384,2048]:
            for fmt in FMTS:
                high_pilot=f'capacity_pilot_{ctx//1024}k_{LABELS[fmt]}_a01'
                launch(high_pilot,fmt,'pilot',ctx,
                       'B0' if ctx==16384 and fmt=='bf16' else 'B0-1',
                       pilot_name(ctx,fmt),mode='capacity',warmups=0)
                evidence=json.loads((HERE/'results'/high_pilot/'summary.json').read_text())
                if not all(t['capacity_verified'] for t in evidence['trials']):
                    raise RuntimeError(f'High-batch pilot did not verify full-generation residency: {high_pilot}')
                launch(f'capacity_{ctx//1024}k_{LABELS[fmt]}_a01',fmt,'measure',ctx,
                       'B0-1,B0,B0+1',high_pilot,mode='capacity',warmups=0)


if __name__=='__main__': main()
