"""Staged execution with durable commands, receipts, watchdogs and pilot budgets."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .ppl import atomic_json
from .kv_official_report import report


def run(root, label, env, module, arguments, timeout, idle=300):
    # Codebook fitting mostly uses CPU; model evaluations must not overlap.
    is_evaluation = module.endswith(('.ppl_external', '.ppl_kivi_sharded'))
    gpu_index = '0,2' if module.endswith('.ppl_kivi_sharded') else '1' if is_evaluation and 'kvquant4' in arguments else '0'
    if is_evaluation:
        lock_root = root.parent if root.name == 'calibration_smoke' else root
        with ExitStack() as stack:
            print(json.dumps(dict(event='waiting_for_evaluation_slot', label=label)), flush=True)
            for device in sorted(gpu_index.split(','), key=int):
                lock_name = 'evaluation.lock' if device == '0' else f'evaluation_gpu{device}.lock'
                lock = stack.enter_context((lock_root/lock_name).open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX)
            return execute(root, label, env, module, arguments, timeout, idle, gpu_index)
    return execute(root, label, env, module, arguments, timeout, idle, gpu_index)


def execute(root, label, env, module, arguments, timeout, idle=300, gpu_index='0'):
    command = [str(root/'envs'/env/'bin/python'), '-u', '-m', module, '--root', str(root), *map(str, arguments)]
    log = root/'logs'/f'{label}.log'
    receipt = root/'logs'/f'{label}.receipt.json'
    started = time.time()
    atomic_json(receipt, dict(state='running', command=command, started=started, timeout=timeout, idle_timeout=idle,
        cuda_visible_devices=gpu_index))
    with (root/'commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(label=label, command=command, started=started))+'\n')
    env_vars = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_index, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with log.open('a') as f:
        process = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, env=env_vars, start_new_session=True)
        reason = None
        while process.poll() is None:
            now = time.time()
            if now-started > timeout:
                reason = 'wall timeout'
            elif now-max(log.stat().st_mtime, started) > idle:
                reason = 'no progress timeout'
            if reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(2)
    result = dict(state='complete' if process.returncode == 0 and not reason else 'failed', command=command,
        started=started, seconds=time.time()-started, returncode=process.returncode, reason=reason,
        cuda_visible_devices=gpu_index)
    atomic_json(receipt, result)
    print(json.dumps(dict(label=label, **result)), flush=True)
    if result['state'] != 'complete':
        raise RuntimeError(f'{label} failed; see {log}')
    return result


def eval_job(root, method, stage, output, timeout):
    env = 'kivi' if method.startswith('kivi') else 'kvquant' if method.startswith('kvquant') else 'baseline'
    batch = 16 if method == 'kivi2' else 12 if method == 'kivi4' else 1
    label = f'{output}_{method}_{stage}'
    module = 'engine.ppl_kivi_sharded' if method == 'kivi4' else 'engine.ppl_external'
    run(root, label, env, module,
        ['--method', method, '--stage', stage, '--output', output, '--batch-size', batch, '--max-seconds', timeout-15], timeout)
    summary = json.loads((root/output/method/'summary.json').read_text())
    assert summary['complete'] and summary['stage'] == stage
    return summary


def cohort(root, method):
    completed = root/'results'/method/'summary.json'
    if completed.exists() and json.loads(completed.read_text()).get('full_test'):
        eval_job(root, method, 'full', 'results', 180)
        report(root)
        return
    smoke = eval_job(root, method, 'smoke', 'smoke_final', 180)
    assert smoke['target_count'] == 255
    pilot = eval_job(root, method, 'pilot', 'results', 600)
    repeat = eval_job(root, method, 'pilot', 'repeat_pilot', 600)
    original = [r for s in (root/'results'/method/'windows.jsonl').read_text().splitlines()
                if (r := json.loads(s))['window_id'] < 2]
    repeated = [r for s in (root/'repeat_pilot'/method/'windows.jsonl').read_text().splitlines()
                if (r := json.loads(s))['window_id'] < 2]
    assert len(original) == len(repeated) == 2
    for a, b in zip(original, repeated):
        assert a['window_id'] == b['window_id'] and a['input_sha256'] == b['input_sha256']
        assert a['target_count'] == b['target_count'] and abs(a['nll']-b['nll'])/a['target_count'] < 1e-6
    # Padded batches run at the actual full-stage size, even for the two-window pilot.
    group_seconds = max(w['elapsed_group_s'] for w in original)
    batch = original[0]['actual_batch_size']
    groups = (pilot['corpus_windows']+batch-1)//batch
    estimate = group_seconds * groups
    timeout = max(180, estimate*1.8+120)
    atomic_json(root/f'pilot_{method}.json', dict(smoke=smoke, pilot=pilot, repeat=repeat,
        mean_nll_repeat_tolerance=1e-6, full_groups=groups, estimated_full_seconds=estimate,
        timeout_seconds=timeout, batch=batch, complete=True))
    print(json.dumps(dict(event='pilot_passed', method=method, full_seconds_estimate=estimate, timeout=timeout)), flush=True)
    eval_job(root, method, 'full', 'results', timeout)
    report(root)


def calibration(root):
    diagnostic = root/'calibration_smoke'
    diagnostic.mkdir(exist_ok=True)
    for name in ('inputs', 'sources', 'envs'):
        p = diagnostic/name
        if not p.exists():
            p.symlink_to(root/name, target_is_directory=True)
    # Calibration diagnostics have their own logs and artifacts.
    (diagnostic/'logs').mkdir(exist_ok=True)
    scratch_smoke = '/tmp/ssnf_kvquant_fisher_smoke_20261029'
    run(diagnostic, 'fisher_smoke', 'kvquant', 'engine.kv_official_prepare',
        ['--action', 'fisher', '--scratch', scratch_smoke, '--samples', 1, '--seqlen', 256], 180)
    run(diagnostic, 'quantizer_smoke3', 'kvquant', 'engine.kv_official_prepare',
        ['--action', 'quantize', '--scratch', scratch_smoke, '--samples', 1, '--seqlen', 256, '--bits', 3], 900)
    smoke = eval_job(diagnostic, 'kvquant3', 'smoke', 'smoke_final', 180)
    assert smoke['target_count'] == 255
    scratch_pilot = '/tmp/ssnf_kvquant_fisher_pilot_20261029'
    pilot = run(root, 'fisher_pilot', 'kvquant', 'engine.kv_official_prepare',
        ['--action', 'fisher', '--scratch', scratch_pilot, '--samples', 2, '--seqlen', 2048], 600)
    scratch = '/tmp/ssnf_kvquant_fisher_20261029'
    run(root, 'fisher_full', 'kvquant', 'engine.kv_official_prepare',
        ['--action', 'fisher', '--scratch', scratch], max(600, pilot['seconds']*8*1.8), idle=300)
    quantizers(root)


def quantizers(root):
    scratch = '/tmp/ssnf_kvquant_fisher_20261029'
    for bits in (3, 4):
        run(root, f'quantizer{bits}', 'kvquant', 'engine.kv_official_prepare',
            ['--action', 'quantize', '--scratch', scratch, '--bits', bits], 14400, idle=900)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--methods', default='bf16,ssnf,kivi2,kivi4,kvquant3,kvquant4')
    p.add_argument('--calibrate', action='store_true')
    p.add_argument('--quantizers-only', action='store_true', help='Resume codebooks from the completed full Fisher artifact.')
    a = p.parse_args()
    root = Path(a.root).resolve()
    if a.calibrate:
        calibration(root)
    elif a.quantizers_only:
        quantizers(root)
    for method in a.methods.split(','):
        cohort(root, method)


if __name__ == '__main__':
    main()
