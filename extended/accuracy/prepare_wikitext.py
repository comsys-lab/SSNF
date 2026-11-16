#!/usr/bin/env python3
"""Export WikiText-2 (raw) splits as the plain-text corpora read by the extended drivers.

Rows are joined as non-empty rows separated by a blank line. The SHA-256 of each file
must match the corpora used for the reported results.
Output: $SSNF_DATA_DIR/wikitext/{train,validation,test}.txt (default: ./data).
"""
import hashlib
import os
from pathlib import Path

from datasets import load_dataset

EXPECTED = {
    'train': 'fb23ad9643a34514eec5cb85ec2a6f49d1a33e6a3d5077dff5a403e1d18f5047',
    'validation': 'f3714de012180492efbcf4f0380322389a27f735ddb63e8c016a6a6285f50408',
    'test': 'c5b5caea5bd655cb221545a484f2f0f59d35092a17a66840d7b9513d0b99687d',
}

out = Path(os.environ.get('SSNF_DATA_DIR', 'data')) / 'wikitext'
out.mkdir(parents=True, exist_ok=True)
for split, digest in EXPECTED.items():
    rows = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split=split)['text']
    text = '\n\n'.join(r for r in rows if r)
    actual = hashlib.sha256(text.encode()).hexdigest()
    if actual != digest:
        raise SystemExit(f'{split}: sha256 {actual} differs from the reported corpus {digest}')
    (out / f'{split}.txt').write_text(text, encoding='utf-8')
    print(split, len(text.encode()), 'bytes', actual)
