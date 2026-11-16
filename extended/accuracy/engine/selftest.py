"""Small CPU checks for codec/code-path invariants, without model evaluation."""
import argparse
from pathlib import Path
import sys
import torch
from . import codecs
from .ppl import canonical_windows, spread_order


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--core-root', required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(a.core_root) / 'eval_accuracy'))
    from wkv_v2_ppl import ssnf_hw, wq_scale
    torch.manual_seed(1328)
    x = torch.randn(23, 128) * 7
    h = torch.rand_like(x) + .01
    for lb, gs in [(3, 8), (3, 16), (4, 32)]:
        for weighted in (False, True):
            hh = h if weighted else None
            assert torch.equal(codecs.ssnf(x, lb, gs, hh), ssnf_hw(x, lb, gs, hh))
            assert torch.equal(codecs.ssnf(x, lb, gs, hh, search=True),
                               wq_scale(x, lb, gs, hh, radius=1 << lb, offs=(0,)))
    for n in [2, 10, 2048, 2049, 4000, 10000, 256789]:
        rows = canonical_windows(n)
        assert sum(w['target_count'] for w in rows) == n - 1
        targets = [i for w in rows for i in range(w['target_start'], w['end'])]
        assert targets == list(range(1, n))
        order = spread_order(len(rows))
        assert sorted(order) == list(range(len(rows)))
        # Selecting any prefix preserves the originally declared target positions.
        assert [rows[i] for i in order[:2]] == [rows[i] for i in order[:4]][:2]
    for f in [lambda z: codecs.mxint(z, 4), lambda z: codecs.mxint(z, 5),
              lambda z: codecs.mxint(z, 6), lambda z: codecs.mxfp(z, 2, 1),
              lambda z: codecs.mxfp(z, 2, 3)]:
        for z in (x, torch.zeros_like(x)):
            q = f(z)
            assert torch.isfinite(q).all() and torch.equal(f(q), q)
    for n in [128, 32768]:
        for lo, hi, budget in [(4, 6, 5.75), (5, 6, 5.5), (4, 5, 4.5)]:
            count = codecs.binary_high_count(n, lo, hi, budget)
            assert codecs.binary_encoded_bytes(n, lo, hi, count)['total_bytes'] * 8 / (n * 32) <= budget
    print('PASS: core SSNF equivalence; causal target coverage; nested subsets; baseline finite/idempotent; encoded byte quotas')


if __name__ == '__main__':
    main()
