"""Explicit fake-quant codecs; all rounding, scales and candidate sets are fixed."""
from __future__ import annotations

import math
import torch

BLOCK = 32


def ssnf(x, lb, gs, importance=None, search=False, scale_offsets=(0,)):
    shape, dtype = x.shape, x.dtype
    z = x.reshape(-1, BLOCK).float()
    if z.numel() == 0:
        return x.clone()
    if BLOCK % gs or not 1 <= lb <= 7:
        raise ValueError((lb, gs))
    h = torch.ones_like(z) if importance is None else torch.broadcast_to(importance, shape).reshape(-1, BLOCK).float()
    e = torch.floor(torch.log2(z.abs().amax(-1, keepdim=True).clamp_min(1e-30))) - 6
    qmax = (1 << (7 - lb)) - 1
    lo, hi = -(1 << (lb - 1)), (1 << (lb - 1)) - 1
    ng = BLOCK // gs
    best, best_block = None, None
    for offset in scale_offsets:
        scale = torch.exp2(e + offset)
        upper_scale = scale * (1 << lb)
        upper = torch.round(z / upper_scale).clamp(-qmax, qmax)
        if not search:
            residual = (z - upper * upper_scale).reshape(-1, ng, gs)
            hg = h.reshape(-1, ng, gs)
            shared = torch.round((residual * hg).sum(-1) / hg.sum(-1).clamp_min(1e-30) / scale)
            shared = shared.clamp(lo, hi)
            recon = upper * upper_scale + shared.repeat_interleave(gs, -1) * scale
        else:
            # Ascending unique shared codes. A strict '<' fixes ties identically.
            best_group = torch.full((len(z), ng), float('inf'), device=z.device)
            recon = torch.zeros_like(z)
            for code in range(lo, hi + 1):
                shifted = code * scale
                candidate_upper = torch.round((z - shifted) / upper_scale).clamp(-qmax, qmax)
                candidate = candidate_upper * upper_scale + shifted
                error = (h * (candidate - z).square()).reshape(-1, ng, gs).sum(-1)
                take = error < best_group
                recon = torch.where(take.repeat_interleave(gs, -1), candidate, recon)
                best_group = torch.where(take, error, best_group)
        block_loss = (h * (recon - z).square()).sum(-1, keepdim=True)
        if best is None:
            best, best_block = recon, block_loss
        else:
            take = block_loss < best_block
            best = torch.where(take, recon, best)
            best_block = torch.where(take, block_loss, best_block)
    return best.reshape(shape).to(dtype)


def mxint(x, bits, scale_offsets=(0,), importance=None):
    """Symmetric code range; per-32 floor-log2 scale, including saturation."""
    shape, dtype = x.shape, x.dtype
    z = x.reshape(-1, BLOCK).float()
    e = torch.floor(torch.log2(z.abs().amax(-1, keepdim=True).clamp_min(1e-30))) - (bits - 2)
    qmax = (1 << (bits - 1)) - 1
    h = torch.ones_like(z) if importance is None else torch.broadcast_to(importance, shape).reshape(-1, BLOCK).float()
    best = loss = None
    for off in scale_offsets:
        scale = torch.exp2(e + off)
        value = torch.round(z / scale).clamp(-qmax, qmax) * scale
        error = (h * (value - z).square()).sum(-1, keepdim=True)
        if best is None:
            best, loss = value, error
        else:
            take = error < loss
            best, loss = torch.where(take, value, best), torch.where(take, error, loss)
    return best.reshape(shape).to(dtype)


def mxfp(x, eb, mb, scale_offsets=(0,), importance=None):
    """Finite MXFP codebook, nearest-even with subnormals and signed saturation."""
    shape, dtype = x.shape, x.dtype
    z = x.reshape(-1, BLOCK).float()
    bias = (1 << (eb - 1)) - 1
    emin, emax = 1 - bias, (1 << eb) - 1 - bias
    maximum = (2 - 2 ** (-mb)) * 2 ** emax
    exponent = torch.floor(torch.log2(z.abs().amax(-1, keepdim=True).clamp_min(1e-30))) - emax
    h = torch.ones_like(z) if importance is None else torch.broadcast_to(importance, shape).reshape(-1, BLOCK).float()
    best = loss = None
    for off in scale_offsets:
        scale = torch.exp2(exponent + off)
        scaled = z / scale
        element_e = torch.floor(torch.log2(scaled.abs().clamp_min(1e-30))).clamp(emin, emax)
        step = torch.exp2(element_e - mb)
        value = (torch.round(scaled / step) * step).clamp(-maximum, maximum) * scale
        error = (h * (value - z).square()).sum(-1, keepdim=True)
        if best is None:
            best, loss = value, error
        else:
            take = error < loss
            best, loss = torch.where(take, value, best), torch.where(take, error, loss)
    return best.reshape(shape).to(dtype)


def nominal_ssnf(lb, gs):
    return 8 - lb + lb / gs + .25


def encoded_ssnf(lb, gs):
    return ((8 - lb) * 4 + math.ceil((32 // gs) * lb / 8) + 1) / 4


def binary_encoded_bytes(blocks, lo_bits, hi_bits, high_count, *, static_map=False):
    """Actually serializable payload+scale+tag/rank/header length, not runtime GPU memory."""
    payload = ((blocks - high_count) * (4 * lo_bits + 1) + high_count * (4 * hi_bits + 1))
    tags = math.ceil(blocks / 8)
    rank = math.ceil(blocks / 256) * 8
    header = 32
    return {'payload_bytes': payload, 'tag_bytes': tags, 'rank_bytes': rank,
            'header_bytes': header, 'total_bytes': payload + tags + rank + header,
            'map_static': static_map}


def binary_high_count(blocks, low_bits, high_bits, budget_bpe, *, charge_map=True):
    base = blocks * (4 * low_bits + 1)
    overhead = (math.ceil(blocks / 8) + math.ceil(blocks / 256) * 8 + 32) if charge_map else 0
    budget_bytes = math.floor(blocks * 4 * budget_bpe)
    count = (budget_bytes - base - overhead) // (4 * (high_bits - low_bits))
    if count < 0:
        raise ValueError('budget smaller than low-precision payload plus metadata')
    return min(blocks, int(count))


def choose_high_map(gain, count, policy='importance'):
    """Stable block tie order. Fixed policy is a deterministic density pattern."""
    flat = gain.reshape(-1)
    n = flat.numel()
    mask = torch.zeros(n, dtype=torch.bool)
    if count <= 0:
        return mask.reshape(gain.shape)
    if count >= n:
        return torch.ones_like(mask).reshape(gain.shape)
    if policy == 'fixed':
        i = torch.arange(n, dtype=torch.int64)
        mask = ((i + 1) * count // n) > (i * count // n)
    else:
        order = torch.argsort(flat.float().cpu(), descending=True, stable=True)
        mask[order[:count]] = True
    return mask.reshape(gain.shape)
