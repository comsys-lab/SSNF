"""CPU reference byte streams for binary precision maps and numerical payloads.

This compact reference uses byte-aligned blocks, without stream alignment padding:
32-byte header; one bit/block map; uint64 high-rank per 256 blocks; low stream;
high stream. Each block contains 32 signed codes and one E8M0 exponent byte.
The reference is for a storage comparison, not a serving kernel layout.
"""
from __future__ import annotations

import struct
import numpy as np
import torch
from . import codecs

MAGIC = b'ACCMIX01'
HEADER = struct.Struct('<8sQBBHQI')


def _pack_codes(codes, bits):
    c = np.asarray(codes, dtype=np.uint8).reshape(-1, 32)
    shifts = np.arange(bits, dtype=np.uint8)
    binary = ((c[..., None] >> shifts) & 1).reshape(len(c), 32 * bits)
    return np.packbits(binary, axis=1, bitorder='little')


def _unpack_codes(data, bits):
    binary = np.unpackbits(data, axis=1, bitorder='little').reshape(len(data), 32, bits)
    return (binary.astype(np.uint16) * (1 << np.arange(bits))).sum(-1).astype(np.uint8)


def positive_codebook(bits):
    mb, bias = bits - 3, 1
    values = []
    for i in range(1 << (bits - 1)):
        exp, mant = i >> mb, i & ((1 << mb) - 1)
        values.append((mant / (1 << mb)) * (2 ** (1 - bias)) if exp == 0
                      else (1 + mant / (1 << mb)) * 2 ** (exp - bias))
    return np.asarray(values, dtype=np.float32)


def encode_blocks(x, bits, family):
    """Encode a CPU tensor into per-block code bytes followed by E8M0 byte."""
    x = x.reshape(-1, 32).float().cpu()
    e = torch.floor(torch.log2(x.abs().amax(-1).clamp_min(1e-30))) - ((bits - 2) if family == 'int' else 2)
    if not ((e >= -127) & (e <= 127)).all():
        raise ValueError('reference E8M0 exponent outside finite 0..254')
    scale = torch.exp2(e)[:, None]
    q = codecs.mxint(x, bits) if family == 'int' else codecs.mxfp(x, 2, bits - 3)
    unit = (q / scale).numpy()
    if family == 'int':
        c = unit.astype(np.int16) & ((1 << bits) - 1)
    else:
        book = positive_codebook(bits)
        mag = np.searchsorted(book, np.abs(unit)).astype(np.uint8)
        if not np.array_equal(book[mag], np.abs(unit)):
            raise AssertionError('MX value does not belong to the advertised finite codebook')
        c = mag | ((unit < 0).astype(np.uint8) << (bits - 1))
    payload = _pack_codes(c, bits)
    exponent = (e.numpy().astype(np.int16) + 127).astype(np.uint8)[:, None]
    return np.concatenate([payload, exponent], axis=1)


def decode_blocks(raw, bits, family):
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1, 4 * bits + 1)
    c = _unpack_codes(raw[:, :-1], bits)
    if family == 'int':
        unit = np.where(c >= (1 << (bits - 1)), c.astype(np.int16) - (1 << bits), c)
    else:
        unit = positive_codebook(bits)[c & ((1 << (bits - 1)) - 1)]
        unit = np.where(c & (1 << (bits - 1)), -unit, unit)
    scale = np.exp2(raw[:, -1].astype(np.int16) - 127).astype(np.float32)
    return torch.from_numpy((unit * scale[:, None]).astype(np.float32))


def pack_mixed(x, mask, low, high, family):
    x, mask = x.reshape(-1, 32).cpu(), mask.reshape(-1).cpu().bool()
    n, count = len(x), int(mask.sum())
    header = HEADER.pack(MAGIC, n, low, high, 0, count, 256)
    tags = np.packbits(mask.numpy(), bitorder='little').tobytes()
    ranks = np.asarray([int(mask[:i].sum()) for i in range(0, n, 256)], dtype='<u8').tobytes()
    low_payload = encode_blocks(x[~mask], low, family).tobytes()
    high_payload = encode_blocks(x[mask], high, family).tobytes()
    blob = header + tags + ranks + low_payload + high_payload
    expected = codecs.binary_encoded_bytes(n, low, high, count)['total_bytes']
    assert len(blob) == expected
    return blob


def unpack_mixed(blob, family):
    magic, n, low, high, _, count, interval = HEADER.unpack(blob[:32])
    if magic != MAGIC or interval != 256:
        raise ValueError('unsupported precision-map byte stream')
    tags_end = 32 + (n + 7) // 8
    mask = np.unpackbits(np.frombuffer(blob[32:tags_end], dtype=np.uint8), bitorder='little')[:n].astype(bool)
    ranks_end = tags_end + ((n + 255) // 256) * 8
    ranks = np.frombuffer(blob[tags_end:ranks_end], dtype='<u8')
    assert all(ranks[j] == mask[:i].sum() for j, i in enumerate(range(0, n, 256)))
    low_end = ranks_end + (n - count) * (4 * low + 1)
    a = decode_blocks(np.frombuffer(blob[ranks_end:low_end], dtype=np.uint8), low, family)
    b = decode_blocks(np.frombuffer(blob[low_end:], dtype=np.uint8), high, family)
    result = torch.empty(n, 32)
    result[~torch.from_numpy(mask)] = a
    result[torch.from_numpy(mask)] = b
    assert int(mask.sum()) == count and len(blob) == codecs.binary_encoded_bytes(n, low, high, count)['total_bytes']
    return result


def selftest():
    torch.manual_seed(1962)
    x = torch.randn(513, 32)
    x[0] = 0
    x[1] = torch.linspace(-6, 6, 32)
    for family, low, high, budget in [('mx', 4, 6, 5.75), ('int', 5, 6, 5.5), ('int', 4, 5, 4.5)]:
        count = codecs.binary_high_count(len(x), low, high, budget)
        mask = codecs.choose_high_map(torch.arange(len(x)), count)
        blob = pack_mixed(x, mask, low, high, family)
        result = unpack_mixed(blob, family)
        ref = torch.where(mask[:, None],
            codecs.mxint(x, high) if family == 'int' else codecs.mxfp(x, 2, high - 3),
            codecs.mxint(x, low) if family == 'int' else codecs.mxfp(x, 2, low - 3))
        assert torch.equal(result, ref)
        assert len(blob) * 8 / x.numel() <= budget
    print('PASS: real numerical MXFP/INT code pack/unpack; E8M0; tags/ranks/header; exact bytes and budgets')


if __name__ == '__main__':
    selftest()
