#!/usr/bin/env python3
"""One bounded fixed-batch/capacity run. Does not overwrite or touch other jobs."""
from __future__ import annotations
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

from protocol import save, sha

HERE = Path(__file__).resolve().parent
ENERGY = HERE.parent
sys.path.insert(0, str(ENERGY))
from common import fixed_environment, source_hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--format', choices=['bf16','mxfp6w_wkv','ssnf_wkv'], required=True)
    p.add_argument('--stage', choices=['smoke','pilot','measure'], default='smoke')
    p.add_argument('--mode', choices=['fixed','capacity'], default='fixed')
    p.add_argument('--input-tokens', type=int, default=128)
    p.add_argument('--output-tokens', type=int, default=8)
    p.add_argument('--batches', default='1', help='Comma-separated B values or capacity offsets B0-1,B0,B0+1')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--cycles', type=int, default=1, help='Complete trials per timing block; not independent repetitions')
    p.add_argument('--minimum-block-seconds', type=float, default=0)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--trial-timeout', type=float, default=180)
    p.add_argument('--total-timeout', type=float, default=900)
    p.add_argument('--after', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.input_tokens < 1 or a.output_tokens < 2 or a.repeats < 1 or a.cycles < 1 or a.warmups < 0:
        p.error('Invalid workload')
    cfg = vars(a).copy()
    cfg['out'] = str(a.out.resolve())
    cfg['after'] = str(a.after.resolve()) if a.after else None
    if a.dry_run:
        print(json.dumps(cfg, indent=2)); return 0
    if a.stage != 'smoke':
        if not a.after:
            p.error('pilot/measure requires a successful prerequisite')
        prev = json.loads((a.after/'summary.json').read_text())
        if not prev.get('run_valid') or prev['format'] != a.format or prev['gpu_uuid'] != a.gpu_uuid:
            p.error('Prerequisite did not pass with same format/GPU')
        if a.stage == 'measure' and (prev['stage'] != 'pilot' or prev['input_tokens'] != a.input_tokens or prev['output_tokens'] != a.output_tokens):
            p.error('measurement requires same-context/output pilot')
    a.out.mkdir(parents=True, exist_ok=False)
    cfg['source_hashes'] = source_hashes()
    cfg['driver_hashes'] = {f.name:sha(f) for f in HERE.glob('*.py')}
    cfg['created_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(a.out/'config.json',cfg)
    (a.out/'source_snapshot').mkdir()
    for path in HERE.glob('*.py'):
        shutil.copy2(path,a.out/'source_snapshot'/path.name)
    env = os.environ.copy()
    for k in list(env):
        if k.startswith('SSNF_'): del env[k]
    env.update(fixed_environment(a.gpu_uuid))
    env['SSNF_HFILE'] = str(HERE.parents[1]/'accuracy/calib/H_w_llama213.pt')
    env['SSNF_KV_IMPORTANCE'] = str(HERE.parents[1]/'accuracy/calib/H_K_llama213.pt')
    # Split-choice cache is session-local; core artifacts stay intact.
    env['SSNF_SPLIT_CACHE'] = str(a.out.resolve()/'split_cache.json')
    env['D1_GUARD_LOG'] = str(a.out.resolve()/'gpu_activity_guard.json')
    env['PATH'] = str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    started=time.monotonic()
    lock=(HERE.parent/(a.gpu_uuid+'.lock')).open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
    with (a.out/'worker.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,str(HERE/'bundle_worker.py'),str(a.out.resolve())], env=env,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(a.out/'launcher.json',dict(pid=proc.pid,started_unix=time.time()))
        try: code=proc.wait(timeout=a.total_timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            save(a.out/'status.json',dict(state='timeout_or_interrupted',partial_results=True));return 2
    save(a.out/'launcher_final.json',dict(exit_code=code,elapsed_s=time.monotonic()-started))
    fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
    lock.close()
    print(f'{a.out}: exit={code}',flush=True)
    return code


if __name__=='__main__': raise SystemExit(main())
