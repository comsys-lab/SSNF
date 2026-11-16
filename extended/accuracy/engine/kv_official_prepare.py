"""Prepare immutable inputs, activation Fisher, and official KVQuant codebooks.

Fisher follows KVQuant gradients/run-fisher.py: BF16 activation gradient squared,
one full causal-LM loss per example, then concatenation along the token dimension.
Frozen weights avoid unused parameter gradients; output hooks replace LinearAct.
Only K/V Fisher tensors are saved, losslessly in their original BF16 precision.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import random
import sys
import time
from types import SimpleNamespace

import torch

from .ppl import atomic_json, digest, load_torch, sha_file


def prepare_inputs(root):
    from transformers import AutoTokenizer
    repo = Path.cwd()
    snapshot = Path.home()/'.cache/huggingface/hub/models--NousResearch--Llama-2-7b-hf/snapshots/8efe6c9b93655b934e27bd9981e3ec13e55aee9d'
    data = Path(__import__('os').environ.get('SSNF_DATA_DIR','data'))/'wikitext'
    out = root/'inputs'
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    test = tok((data/'test.txt').read_text(), return_tensors='pt').input_ids.reshape(-1).long()
    torch.save(dict(ids=test), out/'test.pt')
    # Official KVQuant calibration joins ALL raw train rows, including empty rows.
    import pyarrow as pa
    source = json.loads((data/'manifest.json').read_text())['train']['source_arrow']
    with pa.memory_map(source) as f:
        rows = pa.ipc.open_stream(f).read_all()['text'].to_pylist()
    # Keep a slow-tokenizer check: the official calibration loader uses use_fast=False.
    slow = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=False)
    train = slow('\n\n'.join(rows), return_tensors='pt').input_ids.reshape(-1).long()
    rng = random.Random(0)
    starts = [rng.randint(0, len(train)-2048-1) for _ in range(16)]
    calibration = torch.stack([train[i:i+2048] for i in starts])
    torch.save(dict(ids=calibration, starts=starts), out/'train16.pt')
    hk = repo/'core/calib/H_K_llama27bhf.pt'
    h = load_torch(hk)
    model_files = {p.name: dict(sha256=sha_file(p), bytes=p.stat().st_size) for p in sorted(snapshot.iterdir()) if p.is_file()}
    meta = dict(model=dict(id='NousResearch/Llama-2-7b-hf', revision=snapshot.name, snapshot=str(snapshot), files=model_files),
        test=dict(source=str(data/'test.txt'), source_sha256=sha_file(data/'test.txt'), file_sha256=sha_file(out/'test.pt'),
                  tokens=len(test), token_sha256=digest(test.tolist()), join='nonempty rows with two newlines; existing engine convention',
                  tokenizer='AutoTokenizer fast, default BOS once per corpus'),
        calibration=dict(split='WikiText-2 raw train', source_arrow=source, source_sha256=sha_file(source),
            join='all raw rows with two newlines, official KVQuant convention', tokenizer='AutoTokenizer use_fast=False',
            seed=0, samples=16, seqlen=2048, starts=starts, tokens_sha256=digest(calibration.tolist()),
            file_sha256=sha_file(out/'train16.pt'), uses_test=False),
        ssnf=dict(path=str(hk), sha256=sha_file(hk), meta=h['meta'],
            provenance='Core calibration artifact: WikiText-2 train first 128 x 2048 tokens; metadata names model and token count; no embedded dataset hash',
            source_code_sha256={p.name:sha_file(p) for p in (repo/'core').rglob('calib_importance.py')},
            uses_test=False))
    atomic_json(out/'manifest.json', meta)
    print(json.dumps(dict(test_tokens=len(test), train_shape=list(calibration.shape))), flush=True)


def fisher(root, scratch, samples, seqlen):
    from transformers import AutoModelForCausalLM
    inputs = json.loads((root/'inputs/manifest.json').read_text())
    path = root/'inputs/train16.pt'
    assert sha_file(path) == inputs['calibration']['file_sha256']
    ids = load_torch(path)['ids'][:samples, :seqlen]
    scratch.mkdir(parents=True, exist_ok=True)
    protocol = dict(calibration=inputs['calibration'], samples=samples, seqlen=seqlen,
        model=inputs['model'], source_sha256=sha_file(Path(__file__)), dtype='bfloat16',
        operation='(activation_gradient ** 2).float(), stored losslessly as BF16',
        official_source='KVQuant gradients/run-fisher.py and LinearAct in modeling_llama.py',
        adaptation='frozen parameter weights; equivalent activation output gradient hooks; PyTorch SDPA')
    ph = digest(protocol)
    if (scratch/'manifest.json').exists():
        prior = json.loads((scratch/'manifest.json').read_text())
        if prior['protocol_hash'] != ph:
            raise ValueError('Fisher resume protocol mismatch')
    atomic_json(scratch/'manifest.json', dict(**protocol, protocol_hash=ph, complete=False))
    model = AutoModelForCausalLM.from_pretrained(inputs['model']['snapshot'], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    model.requires_grad_(False)
    handles = []
    gradients = {}
    names = []
    for i, layer in enumerate(model.model.layers):
        for plane in ('k', 'v'):
            name = f'model.layers.{i}.self_attn.{plane}_proj.weight'
            names.append(name)
            module = getattr(layer.self_attn, plane+'_proj')
            def forward(module, inp, output, name=name):
                def backward(gradient):
                    squared = gradient.square()
                    if not torch.isfinite(squared).all():
                        raise ValueError('non-finite activation Fisher')
                    gradients[name] = squared.detach().cpu()
                output.register_hook(backward)
            handles.append(module.register_forward_hook(forward))
    def enable_input_grad(module, inp, out):
        out.requires_grad_(True)
    handles.append(model.model.embed_tokens.register_forward_hook(enable_input_grad))
    records = []
    for index, example in enumerate(ids):
        receipt = scratch/f'sample{index:02d}.json'
        if receipt.exists():
            old = json.loads(receipt.read_text())
            assert old['protocol_hash'] == ph
            for f, h in old['files'].items():
                assert sha_file(scratch/f) == h
            records.append(old)
            continue
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        gradients.clear()
        x = example[None].cuda()
        output = model(x, labels=x, use_cache=False)
        output.loss.backward()
        if set(gradients) != set(names) or not all(g.shape == (1, seqlen, 4096) for g in gradients.values()):
            raise ValueError('Fisher module coverage or shape mismatch')
        files = {}
        nonzero = {}
        for name, tensor in gradients.items():
            p = scratch/f'{name}.sample{index:02d}.pt'
            torch.save(tensor, p)
            files[p.name] = sha_file(p)
            nonzero[name] = int(torch.count_nonzero(tensor))
        if any(n == 0 for n in nonzero.values()):
            raise ValueError('entire Fisher tensor is zero')
        record = dict(index=index, protocol_hash=ph, loss=output.loss.item(), files=files, nonzero=nonzero,
            elapsed_s=time.monotonic()-t0, peak_cuda_bytes=torch.cuda.max_memory_allocated())
        atomic_json(receipt, record)
        records.append(record)
        print(json.dumps(dict(event='fisher', index=index, seconds=record['elapsed_s'], peak_cuda_bytes=record['peak_cuda_bytes'])), flush=True)
        del output, x
    for h in handles:
        h.remove()
    atomic_json(scratch/'manifest.json', dict(**protocol, protocol_hash=ph, complete=True,
        sample_records=[r['index'] for r in records], storage=str(scratch),
        files=digest([r['files'] for r in records])))


class LazyFisher:
    def __init__(self, scratch, samples):
        self.scratch, self.samples = scratch, samples

    def __getitem__(self, key):
        return torch.cat([load_torch(self.scratch/f'{key}.sample{i:02d}.pt') for i in range(self.samples)], dim=1).float()


def quantize(root, scratch, bits, samples, seqlen):
    from transformers import AutoModelForCausalLM
    fm = json.loads((scratch/'manifest.json').read_text())
    assert fm['complete'] and fm['samples'] == samples and fm['seqlen'] == seqlen
    inputs = json.loads((root/'inputs/manifest.json').read_text())
    assert fm['calibration'] == inputs['calibration']
    sys.path.insert(0, str(root/'sources/KVQuant/quant'))
    spec = importlib.util.spec_from_file_location('official_llama_simquant', root/'sources/KVQuant/quant/llama_simquant.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    official.args = SimpleNamespace(nsamples=samples)
    ids = load_torch(root/'inputs/train16.pt')['ids'][:samples, :seqlen]
    loader = [(x[None], x[None]) for x in ids]
    model = AutoModelForCausalLM.from_pretrained(inputs['model']['snapshot'], local_files_only=True,
        torch_dtype=torch.float16, attn_implementation='sdpa').eval()
    model.seqlen = seqlen
    artifact = root/'artifacts'
    artifact.mkdir(exist_ok=True)
    # The compatibility copy checkpoints completed projection quantizers here.
    official.CHECKPOINT_DIR = artifact/f'quantizers{bits}_parts'
    official.CHECKPOINT_DIR.mkdir(exist_ok=True)
    protocol = dict(fisher=fm, calibration=inputs['calibration'], bits=bits, sparse=.01, nuq=True,
        source_sha256=sha_file(root/'sources/KVQuant/quant/llama_simquant.py'),
        quantizer_source_sha256=sha_file(root/'sources/KVQuant/quant/kvquant/simquant_module_quantizer.py'))
    lock = official.CHECKPOINT_DIR/'protocol.json'
    if lock.exists() and json.loads(lock.read_text()) != protocol:
        raise ValueError('quantizer checkpoint protocol mismatch')
    atomic_json(lock, protocol)
    t0 = time.monotonic()
    quantizers = official.llama_calibration(model, loader, torch.device('cuda:0'), ['k_proj'], ['v_proj'], bits,
        include_sparse=True, sparsity_threshold=.99, nuq=True, fisher=LazyFisher(scratch, samples),
        norm=False, cap_outliers=-1, first_few_fp16=-1)
    assert len(quantizers) == 64
    target = artifact/f'quantizers{bits}.pickle'
    with target.with_suffix('.tmp').open('wb') as f:
        pickle.dump(quantizers, f, protocol=pickle.HIGHEST_PROTOCOL)
    target.with_suffix('.tmp').replace(target)
    atomic_json(artifact/f'quantizers{bits}.json', dict(protocol=protocol, seconds=time.monotonic()-t0,
        modules=sorted(quantizers), sha256=sha_file(target)))
    # Shared record is independent of bit width and does not change after eval starts.
    atomic_json(artifact/'calibration.json', dict(fisher=fm, inputs=inputs['calibration'],
        evaluation_dtype='float16', norm=False, attention_sink=False, uses_test=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--action', choices=('inputs', 'fisher', 'quantize'), required=True)
    p.add_argument('--scratch', default='/tmp/ssnf_kvquant_fisher_20261029')
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--seqlen', type=int, default=2048)
    p.add_argument('--bits', type=int, choices=(3, 4), default=3)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    root = Path(a.root).resolve()
    if a.action == 'inputs':
        prepare_inputs(root)
    elif a.action == 'fisher':
        fisher(root, Path(a.scratch), a.samples, a.seqlen)
    else:
        quantize(root, Path(a.scratch), a.bits, a.samples, a.seqlen)


if __name__ == '__main__':
    main()
