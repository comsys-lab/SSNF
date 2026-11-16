"""Frozen-input PPL adapters for the official KIVI and KVQuant implementations.

Run from the repository root with ``python -m engine.ppl_external``.
Each window resets cache and positions. KIVI consumes one ground-truth token per
step, including the unscored prefix, so every scored token reads the real cache.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import torch
import torch.nn.functional as F

from . import codecs
from .ppl import atomic_json, canonical_windows, digest, load_torch, sha_file

sys.path.insert(0,str(Path(__file__).resolve().parent))
METHODS = ('bf16', 'ssnf', 'ssnf_wkv', 'qserve4', 'qserve4g128', 'fp16')
VERSION = 'official-qserve-w4a8kv4-ppl-v1'


def environment():
    packages = {}
    for name in ('torch', 'transformers', 'tokenizers', 'triton', 'numpy', 'scikit-learn', 'accelerate'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                packages=packages, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                capability=torch.cuda.get_device_capability(), visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'))


def source_hashes(root, method):
    paths = [Path(__file__), Path(__file__).with_name('codecs.py'), Path(__file__).with_name('ppl.py')]
    if method.startswith('kivi'):
        paths += sorted((root/'sources/KIVI').rglob('*.py'))
        paths += sorted((root/'sources/KIVI/quant/csrc').glob('*'))
        paths += sorted((root/'sources/KIVI/quant').glob('*.so'))
    if method.startswith('kvquant'):
        paths += sorted((root/'sources/KVQuant/quant/kvquant').glob('*.py'))
        paths += [root/f'artifacts/quantizers{method[-1]}.pickle', root/'artifacts/calibration.json']
    if method.startswith('qserve'): paths += [Path(__file__).with_name(n) for n in ('backend.py','native_model.py')]
    if method.startswith('qserve'):
        paths += sorted((root/'sources/omniserve/omniserve').rglob('*.py'))
        paths += [p for p in (root/'sources/omniserve/kernels/csrc').rglob('*') if p.suffix in ('.cpp','.cu','.h','.hpp')]
        paths += sorted((root/'build').rglob('*.so'))
        paths += [root/'artifacts/checkpoint_audit.json']
        tag='Llama-2-7B-QServe' if method=='qserve4' else 'Llama-2-7B-QServe-g128'
        paths += [root/'checkpoints'/tag/'pytorch_model.bin']
    if method=='ssnf_wkv': paths += [root/'artifacts/ssnf_weight_importance.json']
    return {str(p.resolve()): sha_file(p) for p in paths if p.is_file()}


class Adapter:
    def __init__(self, root, method):
        from transformers import AutoModelForCausalLM, LlamaConfig
        self.root, self.method = root, method
        self.calls = Counter()
        self.handles = []
        self.real_sdpa = F.scaled_dot_product_attention
        self.inputs = json.loads((root/'inputs/manifest.json').read_text())
        self.dtype = torch.bfloat16 if method in ('bf16', 'ssnf', 'ssnf_wkv') else torch.float16
        model_path = self.inputs['model']['snapshot']
        if method.startswith('qserve'):
            from .native_model import NativeAdapter
            self.native=NativeAdapter(-1 if method=='qserve4' else 128)
            self.model=self.native.model
            self.layers=list(self.model.model.layers)
            return
        if method.startswith('kivi'):
            sys.path.insert(0, str(root/'sources/KIVI'))
            sys.path.insert(0, str(root/'sources/KIVI/quant'))
            from models.llama_kivi import LlamaForCausalLM_KIVI
            config = LlamaConfig.from_pretrained(model_path, local_files_only=True)
            config.k_bits = config.v_bits = int(method[-1])
            config.group_size, config.residual_length, config.use_flash = 32, 128, False
            self.model = LlamaForCausalLM_KIVI.from_pretrained(
                model_path, config=config, torch_dtype=self.dtype, local_files_only=True,
                attn_implementation='eager').cuda().eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=self.dtype, local_files_only=True,
                attn_implementation='sdpa').cuda().eval()
        self.layers = list(self.model.model.layers)
        if method=='ssnf_wkv':
            record=json.loads((root/'artifacts/ssnf_weight_importance.json').read_text())
            path=Path(record['path']);assert sha_file(path)==record['sha256']
            hw=load_torch(path)
            targets=[(n,m) for n,m in self.model.named_modules() if isinstance(m,torch.nn.Linear) and n.endswith(('.q_proj','.k_proj','.v_proj','.o_proj','.gate_proj','.up_proj','.down_proj'))]
            assert len(targets)==224
            with torch.no_grad():
                for name,module in targets:
                    li=name.split('.')[2];role=name.rsplit('.',1)[-1]
                    keyrole='qkv' if role in ('q_proj','k_proj','v_proj') else 'gu' if role in ('gate_proj','up_proj') else 'o' if role=='o_proj' else 'down'
                    h=hw[li+'.'+keyrole].float().cuda().reshape(-1,32)
                    blocks=module.weight.view(-1,32)
                    for start in range(0,len(blocks),32768):
                        end=min(start+32768,len(blocks))
                        importance=h[torch.arange(start,end,device='cuda')%h.shape[0]]
                        blocks[start:end].copy_(codecs.ssnf(blocks[start:end].float(),3,8,importance,search=True).to(self.dtype))
            self.weight_linears=224
        if method in ('ssnf','ssnf_wkv'):
            hk_path = Path(self.inputs['ssnf']['path'])
            if sha_file(hk_path) != self.inputs['ssnf']['sha256']:
                raise ValueError('SSNF importance artifact changed')
            self.hk = load_torch(hk_path)['H_K'].cuda().float()
            assert self.hk.shape == (len(self.layers), 32, 128)
            for i, layer in enumerate(self.layers):
                def before(module, inp, index=i):
                    self.layer = index
                self.handles.append(layer.self_attn.register_forward_pre_hook(before))
            F.scaled_dot_product_attention = self.sdpa
        if method.startswith('kvquant'):
            import pickle
            sys.path.insert(0, str(root/'sources/KVQuant/quant'))
            from kvquant.simquant_module_quantizer import make_quant_sim, QuantLinearSim
            with (root/f'artifacts/quantizers{method[-1]}.pickle').open('rb') as f:
                quantizers = pickle.load(f)
            expected = {f'model.layers.{i}.self_attn.{p}_proj' for i in range(len(self.layers)) for p in ('k', 'v')}
            if set(quantizers) != expected:
                raise ValueError('quantizer layer coverage mismatch')
            for plane in ('k', 'v'):
                selected = {n: q for n, q in quantizers.items() if n.endswith(f'{plane}_proj')}
                make_quant_sim(self.model, selected, int(method[-1]), perchannel=plane == 'k',
                    include_sparse=True, sparsity_threshold=.99, dynamicquantization=plane == 'v',
                    nuq=True, nf_nuq=False, norm=False, cap_outliers=-1, first_few_fp16=-1, clamp=False)
            replacements = {n: m for n, m in self.model.named_modules() if isinstance(m, QuantLinearSim)}
            assert set(replacements) == expected
            for name, module in replacements.items():
                def called(module, inp, out, name=name):
                    self.calls[name] += 1
                self.handles.append(module.register_forward_hook(called))

    def sdpa(self, query, key, value, *args, **kwargs):
        self.calls[self.layer] += 1
        key = codecs.ssnf(key, 3, 16, self.hk[self.layer][None, :, None, :])
        value = codecs.ssnf(value, 4, 32)
        return self.real_sdpa(query, key, value, *args, **kwargs)

    def close(self):
        if hasattr(self,'native'): self.native.close()
        F.scaled_dot_product_attention = self.real_sdpa
        for h in self.handles:
            h.remove()

    @torch.inference_mode()
    def score(self, inputs, target_starts, heartbeat=None):
        if self.method.startswith('qserve'):
            return self.native.score(inputs,target_starts,heartbeat)
        self.calls.clear()
        if self.method.startswith('kivi'):
            return teacher_forced(self.model, inputs, target_starts, heartbeat)
        outputs = self.model(inputs, use_cache=False, position_ids=torch.arange(inputs.shape[1], device=inputs.device)[None])
        nll = F.cross_entropy(outputs.logits[:, :-1].float().transpose(1, 2), inputs[:, 1:], reduction='none').double()
        totals = [nll[i, start-1:].sum().item() for i, start in enumerate(target_starts)]
        expected = len(self.layers) if self.method in ('ssnf','ssnf_wkv') else 2*len(self.layers) if self.method.startswith('kvquant') else 0
        if expected and (len(self.calls) != expected or any(c != 1 for c in self.calls.values())):
            raise ValueError(f'quantization coverage mismatch: {self.calls}')
        return totals, dict(quantized_module_calls=len(self.calls),weight_linears=getattr(self,'weight_linears',0))


@torch.inference_mode()
def teacher_forced(model, inputs, target_starts, heartbeat=None):
    """The logit at local t predicts target t+1; never feed a future token."""
    batch, length = inputs.shape
    cache = None
    totals = torch.zeros(batch, dtype=torch.float64, device=inputs.device)
    compressed_reads = 0
    cache_updates = 0
    for t in range(length-1):
        if cache is not None:
            assert len(cache) == len(model.model.layers)
            assert all(c[-1] == t for c in cache)
            if all(c[0] is not None and c[4] is not None for c in cache):
                compressed_reads += 1
        out = model(inputs[:, t:t+1], past_key_values=cache, use_cache=True,
                    position_ids=torch.full((batch, 1), t, device=inputs.device, dtype=torch.long))
        cache = out.past_key_values
        assert all(c[-1] == t+1 for c in cache)
        cache_updates += 1
        losses = F.cross_entropy(out.logits[:, -1].float(), inputs[:, t+1], reduction='none').double()
        mask = torch.tensor([t+1 >= start for start in target_starts], device=inputs.device)
        totals += losses * mask
        if heartbeat and ((t+1) % 64 == 0 or t == length-2):
            heartbeat(t+1)
    if length >= 256 and compressed_reads == 0:
        raise ValueError('KIVI did not read compressed K and V in every layer')
    assert cache_updates == length-1
    return totals.tolist(), dict(cache_updates=cache_updates, compressed_cache_reads_all_layers=compressed_reads,
                                final_cache_length=cache[0][-1], reset_cache_at_window_start=True)


def evaluate(args):
    root = Path(args.root).resolve()
    method = args.method
    out = root/args.output/method
    out.mkdir(parents=True, exist_ok=True)
    inputs_meta = json.loads((root/'inputs/manifest.json').read_text())
    data_path = root/'inputs/test.pt'
    assert sha_file(data_path) == inputs_meta['test']['file_sha256']
    ids = load_torch(data_path)['ids']
    if args.stage == 'smoke':
        ids = ids[:256]
    windows = canonical_windows(len(ids), 2048, 1024)
    selected = windows[:1] if args.stage == 'smoke' else windows[:2] if args.stage == 'pilot' else windows
    env = environment()
    meta = dict(version=VERSION, method=method, inputs=inputs_meta, token_sha256=digest(ids.tolist()),
        context=2048, stride=1024, cache_reset='each window', position_ids='local, start at zero each window',
        weight_activation_dtype='bfloat16' if method in ('bf16', 'ssnf', 'ssnf_wkv') else 'W4A8, FP16 interfaces' if method.startswith('qserve') else 'float16',
        kv_dtype='packed INT4 + FP16 scale/zero' if method.startswith('qserve') else 'bfloat16' if method == 'bf16' else 'float16' if method == 'fp16' else 'quantized reconstruction',
        official_revision='02b2925aa6fa3b92b06316a1524b7f38922cd9c8' if method.startswith('qserve') else '876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6' if method.startswith('kivi') else
                          '57a238357f0ffe50084670fcd5781c9848f80ea2' if method.startswith('kvquant') else None,
        settings=dict(k_lb=3,k_gs=16,v_lb=4,v_gs=32,weighted_k=True,weight_scope=method=='ssnf_wkv',w_lb=3 if method=='ssnf_wkv' else None,w_gs=8 if method=='ssnf_wkv' else None,weighted_w=method=='ssnf_wkv',search_w=method=='ssnf_wkv') if method in ('ssnf','ssnf_wkv') else
                 dict(weight_bits=4,activation_bits=8,kv_bits=4,weight_group=-1 if method=='qserve4' else 128,kv_axis='dynamic per-token per-head asymmetric',pages=64,teacher_forcing=True,sparse_attention=False,fp16_outlier_preservation=False,logit_projection_batch_size=1) if method.startswith('qserve') else
                 dict(k_bits=int(method[-1]), v_bits=int(method[-1]), group_size=32, residual_length=128,
                      prefill='official eager attention; assertion-only compatibility patch', teacher_forcing=True) if method.startswith('kivi') else
                 dict(bits=int(method[-1]), nuq=True, sparse=.01, key='per-channel, pre-RoPE', value='per-token',
                      norm=False, first_few_fp16=-1) if method.startswith('kvquant') else {},
        sources=source_hashes(root, method), environment=env, windows=windows, batch_size=args.batch_size,
        batch_padding='repeat last independent window; score real windows once')
    # Stage and requested extent may grow; input, code, arithmetic and batch cannot change on resume.
    protocol = digest(meta)
    meta['protocol_hash'] = protocol
    if (out/'meta.json').exists() and json.loads((out/'meta.json').read_text())['protocol_hash'] != protocol:
        raise ValueError('resume refused: different input, code, environment, or settings')
    atomic_json(out/'meta.json', meta)
    rows = {}
    log = out/'windows.jsonl'
    if log.exists():
        for line in log.read_text().splitlines():
            row = json.loads(line)
            if row['protocol_hash'] != protocol or row['window_id'] in rows:
                raise ValueError('mixed protocol or duplicate window')
            w = windows[row['window_id']]
            if any(row[k] != w[k] for k in w) or not math.isfinite(row['nll']) or row['nll'] <= 0:
                raise ValueError('invalid resumed window')
            rows[row['window_id']] = row
    todo = [w for w in selected if w['window_id'] not in rows]
    started = time.monotonic()
    adapter = Adapter(root, method) if todo else None
    try:
        while todo:
            first = todo.pop(0)
            group = [first]
            while todo and len(group) < args.batch_size and todo[0]['end']-todo[0]['start'] == first['end']-first['start']:
                group.append(todo.pop(0))
            if time.monotonic()-started > args.max_seconds:
                raise TimeoutError('job time limit')
            padded = group + [group[-1]] * (args.batch_size-len(group))
            batch = torch.stack([ids[w['start']:w['end']] for w in padded]).cuda()
            target_starts = [w['target_start']-w['start'] for w in padded]
            t0 = time.monotonic()
            def heartbeat(tokens):
                if time.monotonic()-started > args.max_seconds:
                    raise TimeoutError('job time limit during teacher forcing')
                state = dict(method=method, state='evaluating', window_ids=[w['window_id'] for w in group],
                             input_tokens_completed=tokens, completed_windows=len(rows), updated=time.time())
                atomic_json(out/'status.json', state)
                print(json.dumps(state), flush=True)
            totals, audit = adapter.score(batch, target_starts, heartbeat)
            for duplicate in totals[len(group):]:
                if not math.isclose(duplicate, totals[len(group)-1], rel_tol=1e-7, abs_tol=1e-6):
                    raise ValueError('independent duplicate batch rows disagree')
            torch.cuda.synchronize()
            elapsed = time.monotonic()-t0
            for w, nll in zip(group, totals):
                count = w['end']-w['target_start']
                if not math.isfinite(nll) or nll <= 0 or count != w['target_count']:
                    raise ValueError('invalid NLL or target count')
                row = dict(**w, nll=nll, protocol_hash=protocol,
                    input_sha256=digest(ids[w['start']:w['end']].tolist()), elapsed_group_s=elapsed,
                    group_size=len(group), actual_batch_size=len(padded),
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(), audit=audit)
                with log.open('a') as f:
                    f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())
                rows[w['window_id']] = row
            print(json.dumps(dict(event='windows', method=method, completed=len(rows), requested=len(selected), seconds=elapsed)), flush=True)
            atomic_json(out/'status.json', dict(state='evaluating', completed_windows=len(rows), updated=time.time()))
    finally:
        if adapter:
            adapter.close()
    chosen = [rows[w['window_id']] for w in selected if w['window_id'] in rows]
    count = sum(w['target_count'] for w in chosen)
    nll = sum(w['nll'] for w in chosen)
    complete = len(chosen) == len(selected)
    if complete and args.stage == 'full' and count != len(ids)-1:
        raise ValueError('missing or duplicate corpus targets')
    summary = dict(method=method, stage=args.stage, complete=complete, full_test=complete and args.stage == 'full',
        completed_windows=len(chosen), corpus_windows=len(windows), target_count=count, nll=nll,
        ppl=math.exp(nll/count) if count else None, elapsed_this_invocation_s=time.monotonic()-started,
        protocol_hash=protocol, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    atomic_json(out/'summary.json', summary)
    atomic_json(out/'status.json', summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--stage', choices=('smoke', 'pilot', 'full'), required=True)
    p.add_argument('--output', default='results')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--max-seconds', type=float, default=3600)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        evaluate(args)
    except Exception as e:
        out = Path(args.root)/args.output/args.method
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out/'failure.json', dict(error=repr(e), time=time.time(), command=sys.argv))
        raise


if __name__ == '__main__':
    main()
