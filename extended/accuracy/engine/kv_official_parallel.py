"""Run independent official KVQuant codebook fits in CPU worker processes.

The official layerwise calibration forward is reused unchanged. Activation
collection is moved losslessly to CPU, then each original SimQuant.quantize call
runs in an independent process with the same data, Fisher, parameters and seed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time

import torch

from .kv_official_prepare import LazyFisher
from .ppl import atomic_json, load_torch, sha_file


def collect(root, scratch, samples, seqlen):
    from types import SimpleNamespace
    from transformers import AutoModelForCausalLM
    inputs = json.loads((root/'inputs/manifest.json').read_text())
    protocol = dict(inputs=inputs['calibration'], model=inputs['model'], samples=samples, seqlen=seqlen,
        collector_sha256=sha_file(Path(__file__)), dtype='float16', collection='official layerwise forward; exact FP16 output copy to CPU')
    scratch.mkdir(parents=True, exist_ok=True)
    if (scratch/'manifest.json').exists():
        previous = json.loads((scratch/'manifest.json').read_text())
        assert previous['protocol'] == protocol
        if previous['complete']:
            for name, h in previous['files'].items():
                assert sha_file(scratch/name) == h
            return
    atomic_json(scratch/'manifest.json', dict(protocol=protocol, complete=False))
    sys.path.insert(0, str(root/'sources/KVQuant/quant'))
    spec = importlib.util.spec_from_file_location('official_calibration', root/'sources/KVQuant/quant/llama_simquant.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    official.args = SimpleNamespace(nsamples=samples)
    model = AutoModelForCausalLM.from_pretrained(inputs['model']['snapshot'], torch_dtype=torch.float16,
        attn_implementation='sdpa', local_files_only=True).eval()
    model.seqlen = seqlen
    names = {id(m): n+'.weight' for n, m in model.named_modules() if n.endswith(('self_attn.k_proj', 'self_attn.v_proj'))}
    add_batch = official.SimQuant.add_batch
    def cpu_add_batch(self, inp, out):
        return add_batch(self, inp, out.detach().cpu())
    files = {}
    def capture(self, **kwargs):
        name = names[id(self.layer)]
        value = self.out
        assert value.shape == (samples*seqlen, 4096) and value.dtype == torch.float16
        p = scratch/(name+'.pt')
        torch.save(value, p)
        files[p.name] = sha_file(p)
        print(json.dumps(dict(event='activation', name=name, shape=list(value.shape))), flush=True)
        return None
    official.SimQuant.add_batch = cpu_add_batch
    official.SimQuant.quantize = capture
    ids = load_torch(root/'inputs/train16.pt')['ids'][:samples, :seqlen]
    loader = [(x[None], x[None]) for x in ids]
    official.llama_calibration(model, loader, torch.device('cuda:0'), ['k_proj'], ['v_proj'], 3,
        include_sparse=True, sparsity_threshold=.99, nuq=True, fisher=None,
        norm=False, cap_outliers=-1, first_few_fp16=-1)
    assert len(files) == 64
    atomic_json(scratch/'manifest.json', dict(protocol=protocol, complete=True, files=files))


def fit_one(task):
    root, activations, fisher, output, samples, bits, name = task
    torch.set_num_threads(4)
    sys.path.insert(0, str(Path(root)/'sources/KVQuant/quant'))
    from kvquant.simquant_module_quantizer import SimQuant
    started = time.monotonic()
    q = SimQuant.__new__(SimQuant)
    q.out = load_torch(Path(activations)/(name+'.pt'))
    q.bits, q.perchannel, q.qchannel, q.nsamples = bits, True, 0 if '.k_proj.' in name else -1, samples
    q.rows = q.columns = 4096
    h = LazyFisher(Path(fisher), samples)[name]
    assert h.shape == (1, q.out.shape[0], q.out.shape[1])
    assert torch.isfinite(h).all()
    value = q.quantize(include_sparse=True, sparsity_threshold=.99, nuq=True,
        fisher=h, norm=False, cap_outliers=False, first_few_fp16=-1)
    target = Path(output)/f'{bits}_{name}.pickle'
    with target.with_suffix('.tmp').open('wb') as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    target.with_suffix('.tmp').replace(target)
    return dict(bits=bits, name=name, seconds=time.monotonic()-started, sha256=sha_file(target))


def fit(root, activations, fisher, output, workers, bits, limit):
    am = json.loads((activations/'manifest.json').read_text())
    fm = json.loads((fisher/'manifest.json').read_text())
    assert am['complete'] and fm['complete']
    assert am['protocol']['inputs'] == fm['calibration']
    assert am['protocol']['samples'] == fm['samples'] and am['protocol']['seqlen'] == fm['seqlen']
    source = root/'sources/KVQuant/quant/kvquant/simquant_module_quantizer.py'
    protocol = dict(activations=am, fisher=fm, source_sha256=sha_file(source), driver_sha256=sha_file(Path(__file__)),
        cpu_threads_per_worker=4, sparse=.01, nuq=True, norm=False, first_few_fp16=-1,
        environment={n:os.getenv(n) for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    output.mkdir(parents=True, exist_ok=True)
    lock = output/'protocol.json'
    if lock.exists():
        assert json.loads(lock.read_text()) == protocol
    atomic_json(lock, protocol)
    names = [n[:-3] for n in sorted(am['files'])]
    tasks = [(str(root),str(activations),str(fisher),str(output),fm['samples'],bit,name)
             for bit in bits for name in names if not (output/f'{bit}_{name}.pickle').exists()]
    if limit:
        tasks = tasks[:limit]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(fit_one, tasks):
            with (output/'completed.jsonl').open('a') as f:
                f.write(json.dumps(result)+'\n')
            print(json.dumps(dict(event='quantizer', **result)), flush=True)
    for bit in bits:
        paths = [output/f'{bit}_{name}.pickle' for name in names]
        if not all(p.exists() for p in paths):
            continue
        values = {}
        for name, p in zip(names, paths):
            with p.open('rb') as f:
                values[name.removesuffix('.weight')] = pickle.load(f)
        target = root/f'artifacts/quantizers{bit}.pickle'
        target.parent.mkdir(exist_ok=True)
        with target.open('wb') as f:
            pickle.dump(values, f, protocol=pickle.HIGHEST_PROTOCOL)
        atomic_json(root/f'artifacts/quantizers{bit}.json', dict(protocol=protocol, modules=sorted(values), sha256=sha_file(target)))
    if all((root/f'artifacts/quantizers{bit}.pickle').exists() for bit in (3,4)):
        atomic_json(root/'artifacts/calibration.json', dict(protocol=protocol, uses_test=False,
            evaluation_dtype='float16', execution='independent official quantizer calls in CPU processes'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--action', required=True, choices=('collect','fit'))
    p.add_argument('--activations', required=True)
    p.add_argument('--fisher', default='/tmp/ssnf_kvquant_fisher_20261029')
    p.add_argument('--output', default=None)
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--seqlen', type=int, default=2048)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--bits', default='3,4')
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    root = Path(a.root).resolve()
    if a.action == 'collect':
        collect(root, Path(a.activations), a.samples, a.seqlen)
    else:
        fit(root, Path(a.activations), Path(a.fisher), Path(a.output) if a.output else root/'artifacts/parallel_parts',
            a.workers, list(map(int,a.bits.split(','))), a.limit)
