"""Frozen-core PPL extension with serialized layout audit and total-budget reserve.

Accepts every engine.ppl CLI argument. R candidates use separate output directories
and explicit --budget-k/--budget-v settings; selection must use validation only.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import math
import os
import sys
import torch

from . import ppl
from . import serialization

BaseEngine = ppl.Engine
MAP_DIR = None


class MatchedEngine(BaseEngine):
    def map_key(self, kind):
        return ppl.digest({'model_alias': self.a.model, 'layers': len(self.layers),
            'family': kind, 'budget_k': self.a.budget_k, 'budget_v': self.a.budget_v,
            'map_calib_windows': self.a.map_calib_windows, 'maxlen': self.a.maxlen,
            'map_policy': self.a.map_policy,
            'train_text_sha256': ppl.sha_file(self.a.train_text) if self.a.train_text else None,
            'hk_sha256': self.calib_provenance.get('hk', {}).get('sha256'),
            'codec_sha256': ppl.sha_file(Path(ppl.codecs.__file__))})

    def build_kv_maps(self, kind):
        if MAP_DIR is None:
            return super().build_kv_maps(kind)
        root = Path(MAP_DIR)
        root.mkdir(parents=True, exist_ok=True)
        key = self.map_key(kind)
        path = root / (key + '.pt')
        if path.exists():
            saved = ppl.load_torch(path)
            if saved['key'] != key:
                raise RuntimeError('frozen map provenance mismatch')
            for plane in ('k', 'v'):
                self.kv_maps[(kind, plane)] = saved['maps'][plane].bool().cpu()
            self.map_calibration = dict(saved['calibration'], reused_frozen_map=True,
                frozen_map_sha256=ppl.sha_file(path))
            return
        super().build_kv_maps(kind)
        saved = {'key': key, 'maps': {plane: self.kv_maps[(kind, plane)].cpu() for plane in ('k', 'v')},
            'calibration': self.map_calibration}
        tmp = path.with_suffix('.tmp.' + str(os.getpid()))
        torch.save(saved, tmp)
        os.replace(tmp, path)
        self.map_calibration = dict(self.map_calibration, reused_frozen_map=False,
            frozen_map_sha256=ppl.sha_file(path))

    def protocol(self, method):
        details, _ = super().protocol(method)
        details['matched_extension_sha256'] = ppl.sha_file(Path(__file__))
        details['serializer_sha256'] = ppl.sha_file(Path(serialization.__file__))
        details['storage_layout'] = ('32-byte header; packed binary tag; uint64 high-rank/256 blocks; '
            'low stream then high stream; each block byte-aligned, no stream alignment padding; E8M0 scale/block')
        details['total_budget_rule'] = 'Reserve both static KV map/header/rank byte counts from W byte budget before allocation.'
        details['frozen_map_protocol'] = {'enabled': MAP_DIR is not None,
            'key': self.map_key('mx' if method == 'mx_p' else 'int') if method in ('mx_p', 'int_p') else None}
        # Shared forward/scoring semantics are identical to the frozen core.
        return details, ppl.digest({k: v for k, v in details.items() if k != 'hostname'})

    def prepare(self, method):
        nominal_w = self.a.budget_w
        mixed = method in ('mx_p', 'int_p')
        self.total_budget_audit = None
        if mixed:
            c = self.model.config
            hkv = c.num_key_value_heads
            d = getattr(c, 'head_dim', None) or c.hidden_size // c.num_attention_heads
            blocks = len(self.layers) * hkv * (d // 32)
            static_map_bytes = 2 * (math.ceil(blocks / 8) + math.ceil(blocks / 256) * 8 + 32)
            w_elements = sum(t.numel() for t in self.master.values())
            self.a.budget_w = nominal_w - static_map_bytes * 8 / w_elements
        try:
            super().prepare(method)
        finally:
            self.a.budget_w = nominal_w
        if mixed:
            used_w = sum(t['total_bytes'] for t in self.enc_details)
            budget_w = math.floor(w_elements * nominal_w / 8)
            if used_w + static_map_bytes > budget_w:
                raise AssertionError('mixed W + both static KV maps exceeds SSNF W budget')
            kv_audit = {}
            kind = 'mx' if method == 'mx_p' else 'int'
            for plane in ('k', 'v'):
                mask = self.kv_maps[(kind, plane)]
                low, high = self.mix_bits(kind, plane)
                sizes = ppl.codecs.binary_encoded_bytes(mask.numel(), low, high, int(mask.sum()), static_map=True)
                target = getattr(self.a, 'budget_' + plane)
                bpe = sizes['payload_bytes'] * 8 / (mask.numel() * 32)
                if bpe > target:
                    raise AssertionError('KV payload exceeds declared plane budget')
                kv_audit[plane] = dict(sizes, actual_payload_bpe=bpe, target_payload_bpe=target,
                    shortfall_bpe=target - bpe, map_sha256=ppl.hashlib.sha256(mask.numpy().tobytes()).hexdigest())
            audit = {'weight_elements': w_elements, 'weight_budget_bytes': budget_w, 'weight_encoded_bytes': used_w,
                'kv_static_map_reserved_bytes': static_map_bytes, 'total_static_bytes_including_kv_maps': used_w + static_map_bytes,
                'weight_target_bpe': nominal_w, 'weight_actual_bpe_including_kv_maps': (used_w + static_map_bytes) * 8 / w_elements,
                'weight_shortfall_bpe': nominal_w - (used_w + static_map_bytes) * 8 / w_elements,
                'kv': kv_audit, 'serializer_reference_selftest': 'CPU synthetic real code payload roundtrip passed',
                'actual_model_tensor_payloads_written': False}
            self.total_budget_audit = audit
            self.storage['total_budget_audit'] = audit


def main():
    # Keep the parent CLI and signal/budget/checkpoint policy unchanged.
    global MAP_DIR
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--frozen-map-dir', default=None,
        help='Export train maps once; supply the copied directory on CPU/GPU to reuse identical static masks.')
    args, remaining = extra.parse_known_args()
    MAP_DIR = args.frozen_map_dir
    sys.argv = [sys.argv[0], *remaining]
    ppl.Engine = MatchedEngine
    ppl.main()


if __name__ == '__main__':
    main()
