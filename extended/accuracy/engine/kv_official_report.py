"""Aggregate only complete, aligned whole-corpus external-KV evaluations."""
import argparse
import csv
import json
import math
from pathlib import Path

from .ppl import atomic_json, canonical_windows, digest


def report(root):
    inputs = json.loads((root/'inputs/manifest.json').read_text())
    expected = canonical_windows(inputs['test']['tokens'])
    table = []
    raw = {}
    for method in ('bf16', 'ssnf', 'kivi2', 'kivi4', 'kvquant3', 'kvquant4'):
        folder = root/'results'/method
        row = dict(method=method, weight_activation_dtype='BF16' if method in ('bf16', 'ssnf') else 'FP16',
                   status='not_completed', windows=0, targets=0, ppl='', delta_bf16='', increase_pct='')
        if (folder/'summary.json').exists():
            summary = json.loads((folder/'summary.json').read_text())
            meta = json.loads((folder/'meta.json').read_text())
            rows = [json.loads(s) for s in (folder/'windows.jsonl').read_text().splitlines()]
            row.update(windows=len(rows), targets=sum(r['target_count'] for r in rows))
            if summary['full_test'] and summary['complete']:
                assert meta['inputs'] == inputs
                assert len(rows) == len(expected) and len({r['window_id'] for r in rows}) == len(rows)
                rows.sort(key=lambda r:r['window_id'])
                for r, w in zip(rows, expected):
                    assert all(r[k] == w[k] for k in w)
                    assert r['protocol_hash'] == summary['protocol_hash'] == meta['protocol_hash']
                    assert math.isfinite(r['nll']) and r['nll'] > 0
                count = sum(r['target_count'] for r in rows)
                nll = sum(r['nll'] for r in rows)
                ppl = math.exp(nll/count)
                assert count == inputs['test']['tokens']-1 == summary['target_count']
                assert math.isclose(ppl, summary['ppl'], rel_tol=1e-12)
                row.update(status='full_test_complete', ppl=ppl)
                raw[method] = rows
        if (folder/'failure.json').exists() and row['status'] != 'full_test_complete':
            row['status'] = 'failed_or_incomplete'
        table.append(row)
    if 'bf16' in raw:
        base = table[0]['ppl']
        for row in table:
            if row['status'] == 'full_test_complete':
                row.update(delta_bf16=row['ppl']-base, increase_pct=100*(row['ppl']/base-1))
                for b, r in zip(raw['bf16'], raw[row['method']]):
                    assert b['input_sha256'] == r['input_sha256']
    with (root/'comparison.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    atomic_json(root/'comparison.json', dict(rows=table, complete=all(r['status']=='full_test_complete' for r in table),
        inputs_sha256=digest(inputs), precision_note='KIVI and KVQuant use FP16 weights and activations; deltas to BF16 include this arithmetic difference.',
        implementation_note='KIVI official packed decode with official eager prefill (assertion compatibility patch); KVQuant official simulated quantization using SDPA attention.',
        scope_note='SSNF is KV-only reconstruction; no weight quantization or serving-performance claims.'))
    print(json.dumps(table, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    report(Path(p.parse_args().root))
