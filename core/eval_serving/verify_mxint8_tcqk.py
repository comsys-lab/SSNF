"""Verify the MXINT8 K-staging + TC-QK port: (1) correctness, SSNF_KV_TCQK on vs the existing cp.async
kernel (expect rel ~6.7e-6, bf16 score rounding -- not bit-exact); (2) per-call speed at the real MXINT8
decode operating points B = pool/(ctx+256): 2k -> 67, 4k -> 35, 8k -> 18, 16k -> 9; (3) SSNF asym-path
regression (must be unchanged by this rebuild). Run in the SERVING environment."""
from __future__ import annotations

import os, numpy as np, torch
from collections.abc import Callable
from typing import Any
import ssnf_cuda  # noqa: F401
from ssnf_lib.pack import pack_kv
from ssnf_vllm.kv_layout import kv_slot_size, planes_to_blob

dev = "cuda"; rng = np.random.default_rng(0); torch.manual_seed(0)
BS = 16; QH, KVH, D = 32, 8, 128
POINTS = [(67, 2048), (35, 4096), (18, 8192), (9, 16384)]        # MXINT8 decode batches
SSNF_POINTS = [(134, 2048), (18, 16384)]                          # SSNF regression check

def mxint8_cache(B: int, CTX: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Pack K and V as plain MXINT8 blocks: [E8M0 scale][32 int8 codes] per 32-element block."""
    nb = D // 32
    def pack(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:     # X: [KVH, CTX, D] float32
        blocks = X.reshape(KVH, CTX, nb, 32)
        amax = np.abs(blocks).max(-1, keepdims=True)
        e = np.where(amax > 0, np.floor(np.log2(np.maximum(amax, 1e-30))).astype(np.int32) - 6, 0)
        scale = np.power(2.0, e)
        q = np.clip(np.rint(blocks / scale), -127, 127).astype(np.int8)
        return e.astype(np.int8).squeeze(-1), q                   # [KVH,CTX,nb], [KVH,CTX,nb,32]
    eK, qK = pack(rng.standard_normal((KVH, CTX, D)).astype(np.float32))
    eV, qV = pack(rng.standard_normal((KVH, CTX, D)).astype(np.float32))
    per_head = nb * (1 + 32)
    slot = np.zeros((CTX, KVH, 2 * per_head), np.uint8)
    for pl, (e, q) in enumerate(((eK, qK), (eV, qV))):
        for blk in range(nb):
            base = pl * per_head + blk * 33
            slot[:, :, base] = e[:, :, blk].T.view(np.uint8)
            slot[:, :, base + 1:base + 33] = q[:, :, blk, :].transpose(1, 0, 2).view(np.uint8)
    nb2 = (CTX + BS - 1) // BS
    flat = np.zeros((nb2 * BS, KVH, 2 * per_head), np.uint8); flat[:CTX] = slot
    cache = torch.from_numpy(flat.reshape(nb2, BS, KVH, 2 * per_head)).cuda().repeat(B, 1, 1, 1)
    bt = torch.arange(nb2, dtype=torch.int32).view(1, nb2).repeat(B, 1).cuda()
    for i in range(B):
        bt[i] += i * nb2
    seq = torch.full((B,), CTX, dtype=torch.int32).cuda()
    return cache, bt, seq, nb, per_head

def ssnf_cache(B: int, CTX: int) -> dict[str, Any]:
    """Build a packed SSNF paged KV cache (K u3/gs16, V u4/gs32) for the regression check."""
    K0 = rng.standard_normal((KVH, CTX, D)).astype(np.float32)
    V0 = rng.standard_normal((KVH, CTX, D)).astype(np.float32)
    pK, pV = pack_kv(K0, 3, 16), pack_kv(V0, 4, 32)
    _, nb, KUB, KSB = kv_slot_size(pK); phK = nb * (1 + KUB + KSB)
    _, _, VUB, VSB = kv_slot_size(pV); phV = nb * (1 + VUB + VSB)
    slot = np.concatenate([planes_to_blob(pK), planes_to_blob(pV)], axis=-1)
    nb2 = (CTX + BS - 1) // BS
    flat = np.zeros((nb2 * BS, KVH, phK + phV), np.uint8); flat[:CTX] = slot
    cache = torch.from_numpy(flat.reshape(nb2, BS, KVH, phK + phV)).cuda().repeat(B, 1, 1, 1)
    bt = torch.arange(nb2, dtype=torch.int32).view(1, nb2).repeat(B, 1).cuda()
    for i in range(B):
        bt[i] += i * nb2
    seq = torch.full((B,), CTX, dtype=torch.int32).cuda()
    return dict(cache=cache, bt=bt, seq=seq, nb=nb, KUB=KUB, KSB=KSB, VUB=VUB, VSB=VSB,
                slot=phK + phV, phK=phK)

def timeit(fn: Callable[[], object], it: int = 100, warm: int = 20) -> float:
    """Mean per-call time of fn in microseconds (CUDA-event timed)."""
    for _ in range(warm): fn()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(it): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / it * 1000

print("== MXINT8: ported TC-QK (ON) vs existing cp.async kernel (OFF) ==")
for B, CTX in POINTS:
    cache, bt, seq, nb, per_head = mxint8_cache(B, CTX)
    q = torch.randn(B, QH, D, device=dev, dtype=torch.bfloat16)
    call = lambda: torch.ops.ssnf.mxint8_kv_decode_paged(q, cache, bt, seq, B, QH, KVH, D, nb,
                                                         BS, 2 * per_head, per_head)
    os.environ["SSNF_KV_TCQK"] = "0"; off = call().float(); t_off = timeit(call)
    os.environ.pop("SSNF_KV_TCQK");   on  = call().float(); t_on  = timeit(call)
    rel = ((on - off).abs().max() / off.abs().max()).item()
    print(f"  B={B:4d} ctx={CTX:6d} | OFF {t_off:8.1f}us  ON {t_on:8.1f}us  ({t_off/t_on:5.3f}x)  rel_diff={rel:.2e}", flush=True)
    del cache, q; torch.cuda.empty_cache()

print("\n== SSNF regression (asym path must be unchanged by the rebuild) ==")
for B, CTX in SSNF_POINTS:
    c = ssnf_cache(B, CTX)
    q = torch.randn(B, QH, D, device=dev, dtype=torch.bfloat16)
    f = lambda: torch.ops.ssnf.kv_decode_paged_asym(q, c['cache'], c['bt'], c['seq'], B, QH, KVH, D,
                                                    c['nb'], 3, 16, c['KUB'], c['KSB'], 4, 32,
                                                    c['VUB'], c['VSB'], BS, c['slot'], c['phK'])
    print(f"  B={B:4d} ctx={CTX:6d} | {timeit(f):8.1f}us", flush=True)
    del c, q; torch.cuda.empty_cache()
