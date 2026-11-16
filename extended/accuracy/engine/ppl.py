"""Resumable, exact-target-count HF perplexity evaluation."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

try:
    from . import codecs
except ImportError:
    import codecs as _stdlib_codecs
    import importlib.util
    spec = importlib.util.spec_from_file_location('ssnf_codecs', Path(__file__).with_name('codecs.py'))
    codecs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codecs)

VERSION = 'ppl-engine-v1'
ROLES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
MODELS = {
    'llama31': ('NousResearch/Meta-Llama-3.1-8B', 'H_w_llama31.pt', 'H_K_metallama318b.pt'),
    'mistral': ('mistralai/Mistral-7B-v0.3', 'H_w_mistral.pt', 'H_K_mistral7bv03.pt'),
    'llama2': ('NousResearch/Llama-2-7b-hf', 'H_w_llama2.pt', 'H_K_llama27bhf.pt'),
    'llama31i': ('NousResearch/Meta-Llama-3.1-8B-Instruct', 'H_w_llama31i.pt', 'H_K_metallama318binstruct.pt'),
}
SCOPE = {
    'bf16': ('', False, False),
    'ssnf_f00': ('wkv', False, False), 'ssnf_f10': ('wkv', True, False),
    'ssnf_f01': ('wkv', False, True), 'ssnf_f11': ('wkv', True, True),
    'ssnf_w_mse': ('w', False, False), 'ssnf_w_weighted': ('w', True, False),
    'ssnf_k_unweighted': ('k', False, False), 'ssnf_k_weighted': ('k', False, True),
    'ssnf_kv_unweighted': ('kv', False, False), 'ssnf_kv_weighted': ('kv', False, True),
    'mx_p': ('wkv', True, True), 'int_p': ('wkv', True, True),
    'mxfp4': ('wkv', False, False), 'mxfp6': ('wkv', False, False),
    'int5': ('wkv', False, False),
}
STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def load_torch(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def canonical_windows(seq, maxlen=2048, stride=1024):
    rows = []
    prev = 0
    for b in range(0, seq, stride):
        e = min(b + maxlen, seq)
        # Explicit global target positions. First input position has no causal prediction.
        start_target = max(prev, b + 1)
        if e > start_target:
            rows.append({'window_id': len(rows), 'start': b, 'end': e,
                         'target_start': start_target, 'target_count': e - start_target})
        prev = e
        if e == seq:
            break
    return rows


def spread_order(n):
    """Nested deterministic farthest-interval order; a prefix covers the full corpus."""
    if n <= 1:
        return list(range(n))
    result = [0, n - 1]
    heap = [(-(n - 1), 0, n - 1)]
    while heap:
        _, lo, hi = heapq.heappop(heap)
        if hi - lo <= 1:
            continue
        mid = (lo + hi) // 2
        result.append(mid)
        heapq.heappush(heap, (-(mid - lo), lo, mid))
        heapq.heappush(heap, (-(hi - mid), mid, hi))
    assert len(result) == n and len(set(result)) == n
    return result


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='llama31')
    p.add_argument('--model-id', default=None)
    p.add_argument('--revision', default=None)
    p.add_argument('--method', choices=list(SCOPE), default='bf16')
    p.add_argument('--methods', default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', choices=['bfloat16', 'float32', 'float16'], default='bfloat16')
    p.add_argument('--out', required=True)
    p.add_argument('--core-root', default=None)
    p.add_argument('--calib-dir', default=None)
    p.add_argument('--hw', default=None)
    p.add_argument('--hk', default=None)
    p.add_argument('--data-path', default=None, help='Pretokenized torch file with ids, or JSON text list/text.')
    p.add_argument('--corpus-text', default=None, help='Portable already-joined raw UTF-8 evaluation corpus.')
    p.add_argument('--train-text', default=None, help='Portable WikiText-2 train corpus, only for static map calibration.')
    p.add_argument('--dataset', default='wikitext')
    p.add_argument('--dataset-config', default='wikitext-2-raw-v1')
    p.add_argument('--dataset-split', default='test')
    p.add_argument('--maxlen', type=int, default=2048)
    p.add_argument('--stride', type=int, default=1024)
    p.add_argument('--max-windows', type=int, default=32, help='0 means all canonical windows.')
    p.add_argument('--window-policy', choices=['spread', 'prefix'], default='spread')
    p.add_argument('--max-seconds', type=float, default=3600)
    p.add_argument('--end-unix', type=float, default=0)
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--offline', action='store_true')
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--map-calib-windows', type=int, default=8)
    p.add_argument('--weight-chunk-blocks', type=int, default=32768)
    p.add_argument('--w-lb', type=int, default=3)
    p.add_argument('--w-gs', type=int, default=8)
    p.add_argument('--k-lb', type=int, default=3)
    p.add_argument('--k-gs', type=int, default=16)
    p.add_argument('--v-lb', type=int, default=4)
    p.add_argument('--v-gs', type=int, default=32)
    p.add_argument('--budget-w', type=float, default=5.75)
    p.add_argument('--budget-k', type=float, default=5.5)
    p.add_argument('--budget-v', type=float, default=4.5)
    p.add_argument('--map-policy', choices=['importance', 'mse', 'fixed'], default='importance')
    return p.parse_args()


class Engine:
    def __init__(self, args, started):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.a, self.started = args, started
        self.method, self.scope, self.w_weighted, self.k_weighted = 'bf16', '', False, False
        self.active = False
        self.layer = None
        self.calls = {}
        self.kv_maps = {}
        self.storage = {}
        self.enc_details = []
        self.collect = None
        self.map_losses = {}
        self.device = torch.device(args.device)
        self.dtype = getattr(torch, args.dtype)
        model_id = args.model_id or MODELS.get(args.model, (args.model, None, None))[0]
        self.model_id = model_id
        kw = {'local_files_only': args.offline, 'trust_remote_code': args.trust_remote_code}
        if args.revision:
            kw['revision'] = args.revision
        self.tok = AutoTokenizer.from_pretrained(model_id, **kw)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=self.dtype, attn_implementation='sdpa', **kw).to(self.device).eval()
        self.model.config.use_cache = False
        self.targets = [(n, m) for n, m in self.model.named_modules()
                        if isinstance(m, torch.nn.Linear) and any(n.endswith('.' + r) for r in ROLES)]
        self.layers = list(self.model.model.layers)
        expected = len(self.layers) * 7
        if len(self.targets) != expected:
            raise RuntimeError(f'linear scope mismatch {len(self.targets)} != {expected}')
        self.master = {n: m.weight.detach().to('cpu', copy=True) for n, m in self.targets}
        self.hw, self.hk, self.calib_provenance = {}, None, {}
        calib = Path(args.calib_dir) if args.calib_dir else (Path(args.core_root) / 'calib' if args.core_root else None)
        default_files = MODELS.get(args.model, (None, None, None))[1:]
        hw = Path(args.hw) if args.hw else (calib / default_files[0] if calib and default_files[0] else None)
        hk = Path(args.hk) if args.hk else (calib / default_files[1] if calib and default_files[1] else None)
        if hw and hw.exists():
            self.hw = load_torch(hw)
            self.calib_provenance['hw'] = {'path': str(hw), 'sha256': sha_file(hw),
                'basis': 'core/README.md: WikiText-2 train first 128 nonoverlapping 2048-token segments; H_w has no embedded provenance.'}
        if hk and hk.exists():
            data = load_torch(hk)
            self.hk = data['H_K'].float().to(self.device)
            self.calib_provenance['hk'] = {'path': str(hk), 'sha256': sha_file(hk), 'meta': data.get('meta', {}),
                'basis': 'Core calibration file; inspect embedded metadata and README together.'}
            if self.hk.shape[0] != len(self.layers):
                raise RuntimeError('H_K layer count mismatch')
        self.real_sdpa = F.scaled_dot_product_attention
        self.handles = []
        for i, layer in enumerate(self.layers):
            def before(module, inp, layer_id=i):
                self.layer = layer_id
            self.handles.append(layer.self_attn.register_forward_pre_hook(before))
        F.scaled_dot_product_attention = self.sdpa
        self.ids, self.dataset_meta = self.load_ids(args.dataset_split, args.corpus_text or args.data_path)
        self.windows = canonical_windows(self.ids.numel(), args.maxlen, args.stride)
        self.order = spread_order(len(self.windows)) if args.window_policy == 'spread' else list(range(len(self.windows)))
        self.selected = self.order if args.max_windows == 0 else self.order[:args.max_windows]
        if not self.selected:
            raise RuntimeError('empty evaluation window set')

    def load_ids(self, split, path=None):
        if path:
            path = Path(path)
            if path.suffix in ('.pt', '.pth'):
                obj = load_torch(path)
                ids = obj['ids'] if isinstance(obj, dict) else obj
                meta = {'path': str(path), 'sha256': sha_file(path)}
            else:
                if path.suffix == '.txt':
                    text = path.read_text()
                else:
                    obj = json.loads(path.read_text())
                    text = obj['text'] if isinstance(obj, dict) else '\n\n'.join(str(x) for x in obj)
                ids = self.tok(text, return_tensors='pt').input_ids
                meta = {'path': str(path), 'sha256': sha_file(path)}
        else:
            from datasets import load_dataset
            try:
                ds = load_dataset(self.a.dataset, self.a.dataset_config, split=split)
            except Exception:
                if self.a.dataset != 'wikitext':
                    raise
                ds = load_dataset('Salesforce/wikitext', self.a.dataset_config, split=split)
            text = '\n\n'.join(t for t in ds['text'] if t.strip())
            ids = self.tok(text, return_tensors='pt').input_ids
            meta = {'dataset': self.a.dataset, 'config': self.a.dataset_config, 'split': split,
                    'fingerprint': getattr(ds, '_fingerprint', None), 'join': 'nonempty rows with two newlines'}
        ids = ids.reshape(-1).long().cpu()
        meta.update(token_count=ids.numel(), token_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                    special_tokens='tokenizer default once per corpus, not once per window')
        return ids, meta

    def elapsed(self):
        return time.time() - self.started

    def should_stop(self):
        return STOP or self.elapsed() >= self.a.max_seconds or (self.a.end_unix and time.time() >= self.a.end_unix)

    def h_w(self, name, device):
        parts = name.split('.')
        li = parts[parts.index('layers') + 1]
        role = name.rsplit('.', 1)[-1]
        keyrole = 'qkv' if role in ('q_proj', 'k_proj', 'v_proj') else ('gu' if role in ('gate_proj', 'up_proj') else ('o' if role == 'o_proj' else 'down'))
        key = f'{li}.{keyrole}'
        if key not in self.hw:
            raise RuntimeError(f'weighted W needs H_w[{key}], available keys {list(self.hw)[:4]}')
        return self.hw[key].float().to(device)

    def h_k(self, x, layer):
        if self.hk is None:
            raise RuntimeError('weighted K requires --hk or named --calib-dir file')
        nh = x.shape[1]
        h = self.hk[layer]
        hkv = h.shape[0]
        if nh % hkv:
            raise RuntimeError(f'GQA shape mismatch {nh}/{hkv}')
        idx = torch.arange(nh, device=x.device) // (nh // hkv)
        return h[idx].reshape(1, nh, 1, x.shape[-1])

    def plain_codec(self, x, kind, bits):
        return codecs.mxfp(x, 2, bits - 3) if kind == 'mx' else codecs.mxint(x, bits)

    def sdpa(self, query, key, value, *args, **kwargs):
        layer = self.layer
        if self.active or self.collect:
            if layer is None:
                raise RuntimeError('SDPA called without explicit layer mapping')
            self.calls[layer] = self.calls.get(layer, 0) + 1
        if self.collect:
            self.collect_kv(layer, key, value)
        if self.active:
            if 'k' in self.scope:
                key = self.quant_kv(key, 'k', layer)
            if 'v' in self.scope:
                value = self.quant_kv(value, 'v', layer)
        return self.real_sdpa(query, key, value, *args, **kwargs)

    def quant_kv(self, x, plane, layer):
        a = self.a
        if self.method.startswith('ssnf'):
            h = self.h_k(x, layer) if plane == 'k' and self.k_weighted else None
            return codecs.ssnf(x, a.k_lb if plane == 'k' else a.v_lb,
                               a.k_gs if plane == 'k' else a.v_gs, h)
        if self.method in ('mxfp4', 'mxfp6'):
            return self.plain_codec(x, 'mx', 4 if self.method == 'mxfp4' else 6)
        if self.method == 'int5':
            return codecs.mxint(x, 5)
        kind = 'mx' if self.method == 'mx_p' else 'int'
        low, high = self.mix_bits(kind, plane)
        qlow, qhigh = self.plain_codec(x, kind, low), self.plain_codec(x, kind, high)
        mask = self.kv_maps[(kind, plane)][layer].to(x.device)
        nh, hkv = x.shape[1], mask.shape[0]
        idx = torch.arange(nh, device=x.device) // (nh // hkv)
        mask = mask[idx].repeat_interleave(32, -1).reshape(1, nh, 1, x.shape[-1])
        return torch.where(mask, qhigh, qlow)

    def mix_bits(self, kind, plane):
        if kind == 'mx':
            return 4, 6
        budget = getattr(self.a, 'budget_' + plane)
        return (4, 5) if budget < 5.25 else (5, 6)

    def collect_kv(self, layer, key, value):
        kind = self.collect
        for plane, x in [('k', key), ('v', value)]:
            hkv = self.model.config.num_key_value_heads
            if x.shape[1] != hkv:
                x = x[:, ::x.shape[1] // hkv]
            h = self.h_k(x, layer) if plane == 'k' and self.a.map_policy == 'importance' else None
            low, high = self.mix_bits(kind, plane)
            a, b = self.plain_codec(x, kind, low), self.plain_codec(x, kind, high)
            error = (a.float() - x.float()).square() - (b.float() - x.float()).square()
            if h is not None:
                error *= h
            gain = error.reshape(x.shape[0], hkv, x.shape[2], -1, 32).sum((0, 2, 4))
            k = (kind, plane, layer)
            if k not in self.map_losses:
                self.map_losses[k] = [gain, x.shape[0] * x.shape[2]]
            else:
                self.map_losses[k][0] += gain
                self.map_losses[k][1] += x.shape[0] * x.shape[2]

    @torch.no_grad()
    def build_kv_maps(self, kind):
        if (kind, 'k') in self.kv_maps:
            return
        ids, meta = self.load_ids('train', self.a.train_text)
        n = min(self.a.map_calib_windows, ids.numel() // self.a.maxlen)
        if n != self.a.map_calib_windows or n < 1:
            raise RuntimeError('not enough complete train calibration segments')
        self.active = False
        self.collect = kind
        try:
            for i in range(n):
                if self.should_stop():
                    raise TimeoutError('time budget exhausted during KV-map calibration')
                self.model(ids[i*self.a.maxlen:(i+1)*self.a.maxlen][None].to(self.device), use_cache=False)
        finally:
            self.collect = None
        for plane in ('k', 'v'):
            losses = [self.map_losses[(kind, plane, l)] for l in range(len(self.layers))]
            gain = torch.stack([v / c for v, c in losses]).cpu()
            low, high = self.mix_bits(kind, plane)
            count = codecs.binary_high_count(gain.numel(), low, high, getattr(self.a, 'budget_' + plane), charge_map=False)
            self.kv_maps[(kind, plane)] = codecs.choose_high_map(gain, count, self.a.map_policy)
            self.storage[plane] = codecs.binary_encoded_bytes(gain.numel(), low, high, count, static_map=True)
            self.storage[plane]['payload_bpe'] = self.storage[plane]['payload_bytes'] * 8 / (gain.numel() * 32)
        self.map_calibration = {'dataset': meta, 'segments': n, 'segment_tokens': self.a.maxlen,
                                'importance_file': self.calib_provenance.get('hk'), 'uses_test': False}
        print(json.dumps({'event': 'kv_map_ready', 'kind': kind, 'train_segments': n}), flush=True)

    @torch.no_grad()
    def prepare(self, method):
        self.active = False
        self.method = method
        self.scope, self.w_weighted, self.k_weighted = SCOPE[method]
        for name, module in self.targets:
            module.weight.copy_(self.master[name].to(self.device))
        self.storage, self.enc_details = {}, []
        if method in ('mx_p', 'int_p'):
            self.build_kv_maps('mx' if method == 'mx_p' else 'int')
        if 'w' in self.scope:
            for name, module in self.targets:
                if self.should_stop():
                    raise TimeoutError('time budget exhausted during weight encoding')
                source = self.master[name]
                shape = source.shape
                blocks = source.numel() // 32
                # Index channel importance per chunk; expanding then reshaping a full
                # weight-shaped view would materialize hundreds of MB repeatedly.
                h = self.h_w(name, self.device).reshape(-1, 32) if self.w_weighted else None
                def chunk_h(start, end):
                    if h is None:
                        return None
                    return h[torch.arange(start, end, device=self.device) % h.shape[0]]
                if method in ('mx_p', 'int_p'):
                    kind = 'mx' if method == 'mx_p' else 'int'
                    low, high = self.mix_bits(kind, 'w')
                    gains = []
                    for start in range(0, blocks, self.a.weight_chunk_blocks):
                        end = min(blocks, start + self.a.weight_chunk_blocks)
                        x = source.reshape(-1, 32)[start:end].to(self.device).float()
                        hh = chunk_h(start, end)
                        ql, qh = self.plain_codec(x, kind, low), self.plain_codec(x, kind, high)
                        gain = (ql - x).square() - (qh - x).square()
                        if hh is not None:
                            gain *= hh
                        gains.append(gain.sum(-1).cpu())
                    gain = torch.cat(gains)
                    count = codecs.binary_high_count(blocks, low, high, self.a.budget_w)
                    mask = codecs.choose_high_map(gain, count, self.a.map_policy)
                    sizes = codecs.binary_encoded_bytes(blocks, low, high, count)
                    self.enc_details.append({'tensor': name, 'elements': source.numel(), 'high_count': count,
                                             'policy': 'per-tensor fixed byte quota, gain-ranked blocks', **sizes})
                for start in range(0, blocks, self.a.weight_chunk_blocks):
                    end = min(blocks, start + self.a.weight_chunk_blocks)
                    x = source.reshape(-1, 32)[start:end].to(self.device).float()
                    hh = chunk_h(start, end)
                    if method.startswith('ssnf'):
                        q = codecs.ssnf(x, self.a.w_lb, self.a.w_gs, hh, search=True)
                    elif method in ('mx_p', 'int_p'):
                        q = torch.where(mask[start:end].to(self.device)[:, None],
                                        self.plain_codec(x, kind, high), self.plain_codec(x, kind, low))
                    elif method == 'int5':
                        q = codecs.mxint(x, 5)
                    else:
                        q = self.plain_codec(x, 'mx', 4 if method == 'mxfp4' else 6)
                    module.weight.reshape(-1, 32)[start:end].copy_(q.to(self.dtype))
        self.active = True

    def protocol(self, method):
        a = self.a
        details = {'engine_version': VERSION, 'engine_sha256': sha_file(Path(__file__)),
            'codec_sha256': sha_file(Path(codecs.__file__)), 'model': self.model_id,
            'revision': a.revision, 'config_commit_hash': getattr(self.model.config, '_commit_hash', None),
            'method': method, 'scope': SCOPE[method][0], 'geometry': {'w': [a.w_lb,a.w_gs], 'k': [a.k_lb,a.k_gs], 'v': [a.v_lb,a.v_gs]},
            'dtype': a.dtype, 'device': str(self.device), 'backend': 'HF SDPA; use_cache=False',
            'torch_version': torch.__version__, 'data': self.dataset_meta,
            'maxlen': a.maxlen, 'stride': a.stride, 'window_policy': a.window_policy,
            'full_window_order': self.order, 'canonical_windows': self.windows,
            'calibration': self.calib_provenance, 'map_calib_windows': a.map_calib_windows,
            'budget_bpe': {'w':a.budget_w,'k':a.budget_k,'v':a.budget_v}, 'map_policy': a.map_policy,
            'linear_count': len(self.targets), 'hostname': socket.gethostname(),
            'device_name': torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'CPU'}
        comparable = {k: details[k] for k in ['engine_version','engine_sha256','model','revision','config_commit_hash','dtype','backend',
            'torch_version','maxlen','stride','window_policy','full_window_order','canonical_windows']}
        comparable['device_type'] = self.device.type
        comparable['device_name'] = details['device_name']
        comparable['token_sha256'] = self.dataset_meta['token_sha256']
        details['comparison_hash'] = digest(comparable)
        # Host identity is audit information; device/dtype/backend stay protocol-defining.
        return details, digest({k:v for k,v in details.items() if k!='hostname'})

    @torch.no_grad()
    def evaluate(self, method):
        out = Path(self.a.out) / method
        out.mkdir(parents=True, exist_ok=True)
        meta, protocol_hash = self.protocol(method)
        meta.update(protocol_hash=protocol_hash, corpus_windows=len(self.windows), requested_windows=len(self.selected), selected_window_ids=self.selected,
                    evaluation_extent='full' if len(self.selected)==len(self.windows) else 'prespecified_corpus_spread_subset')
        existing_meta = out/'meta.json'
        if existing_meta.exists() and json.loads(existing_meta.read_text()).get('protocol_hash') != protocol_hash:
            raise RuntimeError('refusing to resume different protocol in same result directory')
        atomic_json(existing_meta, meta)
        rows = {}
        log = out/'windows.jsonl'
        if log.exists():
            for line in log.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    raise RuntimeError('truncated JSONL found: repair explicitly before resuming')
                if r.get('protocol_hash') != protocol_hash:
                    raise RuntimeError('mixed protocol JSONL')
                rows[r['window_id']] = r
        todo = [i for i in self.selected if i not in rows]
        encoding_start = time.time()
        if todo:
            atomic_json(out/'status.json', {'state':'encoding','method':method,'protocol_hash':protocol_hash,'updated':time.time()})
            self.prepare(method)
        encoding_s = time.time() - encoding_start
        atomic_json(out/'encoding.json', {'method':method,'seconds':encoding_s,'storage':self.storage,
                    'weight_tensors':self.enc_details,'map_calibration':getattr(self,'map_calibration',None)})
        elapsed_samples = []
        for index in todo:
            remaining = self.a.max_seconds - self.elapsed()
            if self.should_stop() or (elapsed_samples and remaining < max(elapsed_samples[-3:]) * 1.2):
                break
            w = self.windows[index]
            inputs = self.ids[w['start']:w['end']][None].to(self.device)
            local_target = w['target_start'] - w['start']
            self.calls = {}
            t0 = time.time()
            logits = self.model(inputs, use_cache=False).logits
            # Target at input position j is predicted by logits at j-1.
            scores = logits[:, local_target-1:-1, :].float()
            targets = inputs[:, local_target:]
            nll = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), targets.reshape(-1), reduction='none').double().sum().item()
            count = targets.numel()
            dt = time.time()-t0
            del logits,scores,targets,inputs
            if count != w['target_count'] or not math.isfinite(nll):
                raise RuntimeError(f'invalid metric nll={nll}, actual targets={count}, expected={w["target_count"]}')
            if self.scope and (len(self.calls) != len(self.layers) or any(v!=1 for v in self.calls.values())):
                raise RuntimeError(f'KV hook/layer call mismatch: {self.calls}')
            row = {'record':'window','protocol_hash':protocol_hash,'model':self.model_id,'method':method,**w,
                   'nll':nll,'target_count':count,'elapsed_s':dt,'completed_unix':time.time(),
                   'input_sha256':hashlib.sha256(self.ids[w['start']:w['end']].numpy().tobytes()).hexdigest()}
            with log.open('a') as f:
                f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())
            rows[index] = row
            elapsed_samples.append(dt)
            atomic_json(out/'status.json',{'state':'evaluating','method':method,'completed_windows':sum(i in rows for i in self.selected),
                'requested_windows':len(self.selected),'last_window_seconds':dt,'updated':time.time(),'protocol_hash':protocol_hash})
            print(json.dumps({'event':'window', 'method':method, 'window_id':index,'seconds':round(dt,3),
                  'done':sum(i in rows for i in self.selected),'requested':len(self.selected)}),flush=True)
        complete = all(i in rows for i in self.selected)
        chosen = [rows[i] for i in self.selected if i in rows]
        nll, count = sum(r['nll'] for r in chosen),sum(r['target_count'] for r in chosen)
        summary = {'status':'complete' if complete else 'partial','model':self.model_id,'method':method,'protocol_hash':protocol_hash,
                   'ppl':math.exp(nll/count) if count and nll/count<700 else None, 'total_nll':nll,'target_count':count,
                   'completed_windows':len(chosen),'requested_windows':len(self.selected),'corpus_windows':len(self.windows),
                   'window_ids':[r['window_id'] for r in chosen], 'evaluation_extent':meta['evaluation_extent'],
                   'elapsed_s':self.elapsed(),'encoding_s':encoding_s,'device':str(self.device),'dtype':self.a.dtype,
                   'backend':meta['backend'],'completed_unix':time.time()}
        atomic_json(out/'summary.json',summary)
        atomic_json(out/'status.json',summary)
        print(json.dumps({'event':'summary',**summary}),flush=True)
        return complete

    def close(self):
        F.scaled_dot_product_attention = self.real_sdpa
        for h in self.handles:
            h.remove()


def main():
    a = parse()
    started = time.time()
    Path(a.out).mkdir(parents=True,exist_ok=True)
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.offline:
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['HF_DATASETS_OFFLINE']='1'
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,request_stop)
    methods = a.methods.split(',') if a.methods else [a.method]
    if any(m not in SCOPE for m in methods):
        raise ValueError('unknown method')
    engine = None
    try:
        engine = Engine(a,started)
        for method in methods:
            if engine.should_stop():
                break
            if not engine.evaluate(method):
                break
    except Exception as e:
        atomic_json(Path(a.out)/'error.json',{'error_type':type(e).__name__,'error':str(e),'elapsed_s':time.time()-started,
                    'method':getattr(engine,'method',a.method),'time':time.time()})
        raise
    finally:
        if engine:
            engine.close()


if __name__=='__main__':
    main()
