#!/usr/bin/env python3
"""Transactional, resumable CPU calibration of core-defined SSNF calibration statistics.

Only q_proj/o_proj/gate_proj/down_proj inputs are collected: q/k/v share
one input and gate/up share one input. H_K is post-RoPE E[q**2], grouped
by the query heads that share each KV head. No evaluation split is read.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import time

import torch
import torch.nn.functional as F

STOP = False
VERSION = 'ssnf-calibration-v1'
CORE_SOURCE_SHA256 = {'calibrate_h.py': 'db2e58137a753dcb5eab2fdfb24f7540e165f207bf4b537d35ab980ef8f5670a', 'calib_importance.py': 'dc873a8196b5df87a66adf2ce4314cea4369ba6aab0cf3d0c31445ce614c38f4'}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


COMPATIBLE_OPERATIONAL_PREDECESSOR = '4453f8d162567fea8033084b2cc6f59747642360f8119e3a6a4418e718d2c4a9'

def load_checkpoint(path):
    # This is our own archived checkpoint; allow only its TorchVersion metadata type.
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        saved = torch.load(path, map_location='cpu', weights_only=True)
    if 'protocol' in saved:
        saved['protocol'] = json.loads(json.dumps(saved['protocol']))
    return saved

def atomic_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n'); temp.replace(p)


def atomic_torch(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    torch.save(obj, temp); temp.replace(p)


class Collector:
    """One active chunk has temporary sums; only a checked complete chunk commits."""
    def __init__(self, model, saved=None):
        cfg = model.config
        self.layers = list(model.model.layers)
        self.L = cfg.num_hidden_layers; self.Hq = cfg.num_attention_heads
        self.Hkv = cfg.num_key_value_heads
        self.D = getattr(cfg, 'head_dim', cfg.hidden_size // self.Hq)
        assert self.Hq % self.Hkv == 0
        self.G = self.Hq // self.Hkv
        self.modules = {}
        for i, layer in enumerate(self.layers):
            self.modules[f'{i}.qkv'] = layer.self_attn.q_proj
            self.modules[f'{i}.o'] = layer.self_attn.o_proj
            self.modules[f'{i}.gu'] = layer.mlp.gate_proj
            self.modules[f'{i}.down'] = layer.mlp.down_proj
        assert len(self.modules) == 4 * self.L
        self.sums_w = {k: torch.zeros(m.in_features, dtype=torch.float64) for k, m in self.modules.items()}
        self.counts_w = {k: 0 for k in self.modules}
        self.sums_q = torch.zeros(self.L, self.Hq, self.D, dtype=torch.float64)
        self.counts_q = [0] * self.L
        self.committed = 0
        if saved is not None:
            self.sums_w = saved['sums_w']; self.counts_w = saved['counts_w']
            self.sums_q = saved['sums_q']; self.counts_q = saved['counts_q']
            self.committed = saved['completed_chunks']
            assert set(self.sums_w) == set(self.modules)
        self.active = False; self.layer = None; self.handles = []
        self.real_sdpa = F.scaled_dot_product_attention
        for i, layer in enumerate(self.layers):
            def enter(_m, _a, index=i): self.layer = index
            def leave(_m, _a, _out): self.layer = None
            self.handles += [layer.register_forward_pre_hook(enter), layer.register_forward_hook(leave)]
        for key, module in self.modules.items():
            def hook(_m, inp, name=key):
                if not self.active: return
                assert self.layer == int(name.split('.')[0])
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                self.delta_w[name] = self.delta_w.get(name, 0) + x.square().sum(0).double().cpu()
                self.delta_wc[name] = self.delta_wc.get(name, 0) + x.shape[0]
                self.calls_w[name] = self.calls_w.get(name, 0) + 1
            self.handles.append(module.register_forward_pre_hook(hook))
        F.scaled_dot_product_attention = self.sdpa

    def sdpa(self, q, k, v, *args, **kwargs):
        if self.active:
            i = self.layer
            assert i is not None and q.ndim == 4 and q.shape[1] == self.Hq and q.shape[-1] == self.D
            self.delta_q[i] += q.detach().float().square().sum(dim=(0, 2)).double().cpu()
            self.delta_qc[i] += q.shape[0] * q.shape[2]
            self.calls_q[i] += 1
        return self.real_sdpa(q, k, v, *args, **kwargs)

    def begin(self):
        assert not self.active
        self.delta_w = {}; self.delta_wc = {}; self.calls_w = {}
        self.delta_q = torch.zeros_like(self.sums_q)
        self.delta_qc = [0] * self.L; self.calls_q = [0] * self.L
        self.active = True

    def commit(self, expected_tokens):
        self.active = False
        assert set(self.delta_w) == set(self.modules)
        assert all(self.delta_wc[k] == expected_tokens and self.calls_w[k] == 1 for k in self.modules)
        assert all(c == expected_tokens for c in self.delta_qc)
        assert all(c == 1 for c in self.calls_q)
        assert torch.isfinite(self.delta_q).all()
        for key in self.modules:
            assert torch.isfinite(self.delta_w[key]).all()
            self.sums_w[key] += self.delta_w[key]; self.counts_w[key] += self.delta_wc[key]
        self.sums_q += self.delta_q
        self.counts_q = [a+b for a, b in zip(self.counts_q, self.delta_qc)]
        self.committed += 1

    def abort(self):
        self.active = False

    def checkpoint(self):
        return {'completed_chunks': self.committed, 'sums_w': self.sums_w,
                'counts_w': self.counts_w, 'sums_q': self.sums_q, 'counts_q': self.counts_q}

    def means(self):
        assert self.committed > 0 and all(c > 0 for c in self.counts_q)
        hw = {k: (v/self.counts_w[k]).float() for k, v in self.sums_w.items()}
        eq = self.sums_q / torch.tensor(self.counts_q, dtype=torch.float64)[:, None, None]
        hk = eq.reshape(self.L, self.Hkv, self.G, self.D).mean(2).float()
        return hw, hk

    def close(self):
        F.scaled_dot_product_attention = self.real_sdpa
        for handle in self.handles: handle.remove()


def selftest():
    """Independent stored-input means, GQA reference, and transactional resume."""
    import tempfile
    import numpy as np
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.set_num_threads(2); torch.manual_seed(130213)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, attention_dropout=0)
    cfg._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(cfg).float().eval()
    chunks = [torch.randint(0, 64, (1, 16)), torch.randint(0, 64, (1, 16))]
    stored = {}; queries = {0: [], 1: []}; current = {'layer': None}; handles = []
    real = F.scaled_dot_product_attention
    for i, layer in enumerate(model.model.layers):
        def enter(_m, _a, index=i): current['layer'] = index
        handles.append(layer.register_forward_pre_hook(enter))
        for module_name in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
            module = layer.get_submodule(module_name); key = f'{i}.{module_name}'
            def capture(_m, inp, name=key): stored.setdefault(name, []).append(inp[0].detach().clone().numpy())
            handles.append(module.register_forward_pre_hook(capture))
    def reference_sdpa(q, k, v, *args, **kwargs):
        queries[current['layer']].append(q.detach().clone().numpy())
        return real(q, k, v, *args, **kwargs)
    F.scaled_dot_product_attention = reference_sdpa
    with torch.no_grad():
        for ids in chunks: model.model(ids, use_cache=False)
    F.scaled_dot_product_attention = real
    for h in handles: h.remove()
    reference_w = {}
    roles = {'qkv': 'self_attn.q_proj', 'o': 'self_attn.o_proj', 'gu': 'mlp.gate_proj', 'down': 'mlp.down_proj'}
    for i in range(2):
        for role, name in roles.items():
            x = np.concatenate([a.reshape(-1, a.shape[-1]) for a in stored[f'{i}.{name}']], axis=0).astype(np.float64)
            reference_w[f'{i}.{role}'] = (x*x).mean(0)
        for other in ['self_attn.k_proj', 'self_attn.v_proj']:
            assert all(np.array_equal(a,b) for a,b in zip(stored[f'{i}.self_attn.q_proj'], stored[f'{i}.{other}']))
        assert all(np.array_equal(a,b) for a,b in zip(stored[f'{i}.mlp.gate_proj'], stored[f'{i}.mlp.up_proj']))
    reference_k = []
    for i in range(2):
        q = np.concatenate([x.transpose(0,2,1,3).reshape(-1,4,8) for x in queries[i]], axis=0).astype(np.float64)
        reference_k.append((q*q).mean(0).reshape(2,2,8).mean(1))
    collector = Collector(model)
    with torch.no_grad():
        collector.begin(); model.model(chunks[0], use_cache=False); collector.commit(16)
    with tempfile.TemporaryDirectory() as tmp:
        filename = Path(tmp)/'checkpoint.pt'; atomic_torch(filename, {**collector.checkpoint(), 'protocol': {'torch_version': torch.__version__}})
        saved = load_checkpoint(filename)
        assert type(saved['protocol']['torch_version']) is str
    collector.close(); collector = Collector(model, saved)
    before = collector.checkpoint()['sums_q'].clone()
    collector.begin()
    with torch.no_grad(): model.model(chunks[1], use_cache=False)
    collector.abort()
    assert torch.equal(before, collector.sums_q) and collector.committed == 1
    with torch.no_grad():
        collector.begin(); model.model(chunks[1], use_cache=False); collector.commit(16)
    hw, hk = collector.means()
    for key, expected in reference_w.items(): np.testing.assert_allclose(hw[key].numpy(), expected, rtol=3e-6, atol=1e-8)
    np.testing.assert_allclose(hk.numpy(), np.stack(reference_k), rtol=3e-6, atol=1e-8)
    assert collector.committed == 2 and set(collector.counts_w.values()) == {32} and collector.counts_q == [32,32]
    collector.close()
    print(json.dumps({'status':'pass','test':'independent_numpy_means_GQA_role_dedup_checkpoint_resume_aborted_chunk','layers':2,'committed_chunks':2,'tokens_per_layer':32}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--model'); ap.add_argument('--model-manifest'); ap.add_argument('--train-text')
    ap.add_argument('--out'); ap.add_argument('--nsamples', type=int, default=128)
    ap.add_argument('--seqlen', type=int, default=2048); ap.add_argument('--threads', type=int, default=32)
    ap.add_argument('--max-seconds', type=float, default=7200); ap.add_argument('--end-unix', type=float, default=0)
    args = ap.parse_args()
    if args.self_test: selftest(); return
    if not all([args.model,args.model_manifest,args.train_text,args.out]): ap.error('model, model-manifest, train-text and out required')
    started = time.time(); torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    modelpath = Path(args.model); proof = json.loads(Path(args.model_manifest).read_text())
    assert proof['state'] == 'complete'
    for f in proof['files']:
        assert f['state']=='complete' and (modelpath/f['name']).stat().st_size==f['size']
        if f.get('sha256'): assert f['actual_sha256']==f['sha256']
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    textpath = Path(args.train_text); ids = tok(textpath.read_text(), return_tensors='pt').input_ids.reshape(-1).long()
    assert args.nsamples > 0 and ids.numel() >= args.nsamples*args.seqlen
    tokenizer_files = {p.name:sha_file(p) for p in modelpath.iterdir() if p.is_file() and (p.name.startswith('tokenizer') or p.name=='special_tokens_map.json')}
    protocol = {'version':VERSION,'core_definitions_sha256':CORE_SOURCE_SHA256,'code_sha256':sha_file(__file__),'model_repo':proof['repo'],'model_revision':proof['revision'],
                'model_manifest_sha256':sha_file(args.model_manifest),'weight_sha256':{f['name']:f['actual_sha256'] for f in proof['files'] if f['name'].endswith('.safetensors')},
                'config_sha256':sha_file(modelpath/'config.json'),'tokenizer_files':tokenizer_files,
                'train_text_sha256':sha_file(textpath),'train_token_sha256':hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                'source':'WikiText-2 raw train, nonempty rows joined by two newlines; tokenizer defaults once per corpus',
                'seqlen':args.seqlen,'chunk_policy':'prefix, nonoverlapping; chunk i is [i*seqlen,(i+1)*seqlen)',
                'dtype':'float32','accumulator_dtype':'float64; per-chunk square/sum float32 as core',
                'H_W':'per-linear input E[x^2], 4 unique roles qkv/o/gu/down per layer',
                'H_K':'SDPA post-RoPE E[q^2], explicit layer mapping, GQA group mean',
                'torch_version':str(torch.__version__),'transformers_version':__import__('transformers').__version__}
    protocol_hash=digest(protocol); checkpoint=out/'checkpoint.pt';saved=None
    if checkpoint.exists():
        saved=load_checkpoint(checkpoint)
        assert saved['protocol_hash']==digest(saved['protocol']),'Corrupt checkpoint protocol digest'
        if saved['protocol_hash']!=protocol_hash:
            old_protocol=saved['protocol']; candidate=dict(protocol)
            candidate['code_sha256']=old_protocol['code_sha256']
            assert old_protocol['code_sha256']==COMPATIBLE_OPERATIONAL_PREDECESSOR and candidate==old_protocol, 'Cannot resume a different scientific calibration protocol'
            protocol=old_protocol;protocol_hash=saved['protocol_hash']
        assert saved['completed_chunks']<=args.nsamples,'Requested count smaller than committed count'
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).cpu().eval()
    model.config.use_cache=False;collector=Collector(model,saved)
    times=list(saved.get('chunk_timings',[])) if saved else []
    resumed_from_chunks=collector.committed
    prefix_checkpoint_sha256=sha_file(checkpoint) if saved else None
    operational_code_sha256=sha_file(__file__)
    try:
        with torch.no_grad():
            while collector.committed<args.nsamples:
                elapsed=time.time()-started
                if STOP or elapsed>=args.max_seconds or (args.end_unix and time.time()>=args.end_unix):break
                if times and args.max_seconds-elapsed<max(times[-3:])*1.2:break
                i=collector.committed;sample=ids[i*args.seqlen:(i+1)*args.seqlen].unsqueeze(0);tick=time.time();collector.begin()
                try:model.model(sample,use_cache=False);collector.commit(args.seqlen)
                except BaseException:collector.abort();raise
                seconds=time.time()-tick;times.append(seconds)
                record={**collector.checkpoint(),'protocol_hash':protocol_hash,'protocol':protocol,'chunk_timings':times,'operational_code_sha256':operational_code_sha256}
                atomic_torch(checkpoint,record)
                atomic_json(out/'status.json',{'status':'calibrating','completed_chunks':collector.committed,'requested_chunks':args.nsamples,'last_chunk_seconds':seconds,'protocol_hash':protocol_hash,'updated_unix':time.time()})
                print(json.dumps({'event':'chunk','completed':collector.committed,'requested':args.nsamples,'seconds':seconds}),flush=True)
        assert collector.committed>0,'No complete calibration chunks'
        hw,hk=collector.means();ntokens=collector.committed*args.seqlen
        meta={**protocol,'protocol_hash':protocol_hash,'nchunk':collector.committed,'maxlen':args.seqlen,'ntok':ntokens,
              'L':collector.L,'Hkv':collector.Hkv,'D':collector.D,'G':collector.G,
              'counts_w':collector.counts_w,'counts_q':collector.counts_q,
              'source_chunks':[{'index':i,'start':i*args.seqlen,'end':(i+1)*args.seqlen} for i in range(collector.committed)],
              'operational_code_sha256':operational_code_sha256,'resumed_from_chunks':resumed_from_chunks,'prefix_checkpoint_sha256':prefix_checkpoint_sha256,'hostname':socket.gethostname(),'completed_unix':time.time(),'chunk_timings':times}
        assert set(collector.counts_w.values())=={ntokens} and set(collector.counts_q)=={ntokens}
        assert all(torch.isfinite(v).all() and (v>=0).all() for v in hw.values()) and torch.isfinite(hk).all() and (hk>=0).all()
        atomic_torch(out/'H_w_llama213.pt',hw);atomic_json(out/'H_w_llama213.provenance.json',meta)
        atomic_torch(out/'H_K_llama213.pt',{'H_K':hk,'meta':meta})
        complete=collector.committed==args.nsamples
        summary={'status':'complete' if complete else 'partial','completed_chunks':collector.committed,'requested_chunks':args.nsamples,
                 'tokens_per_layer':ntokens,'elapsed_seconds':time.time()-started,'protocol_hash':protocol_hash,
                 'H_w_sha256':sha_file(out/'H_w_llama213.pt'),'H_K_sha256':sha_file(out/'H_K_llama213.pt'),
                 'chunk_timings':times,'projected_128_chunk_seconds':sum(times)/len(times)*128 if times else None,
                 'scope':'calibration only; no test/validation inputs; no reused 7B statistics','operational_code_sha256':operational_code_sha256,'resumed_from_chunks':resumed_from_chunks,'prefix_checkpoint_sha256':prefix_checkpoint_sha256}
        atomic_json(out/'summary.json',summary);atomic_json(out/'status.json',summary)
        stage=out/'stages'/str(collector.committed);stage.mkdir(parents=True,exist_ok=True)
        import shutil
        for name in ['summary.json','H_w_llama213.pt','H_K_llama213.pt','H_w_llama213.provenance.json']:
            shutil.copy2(out/name,stage/name)
        print(json.dumps(summary),flush=True)
    finally:collector.close()


def stop(_sig,_frame):
    global STOP
    STOP=True


if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);main()
