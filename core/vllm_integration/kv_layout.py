"""SSNF paged-KV single-tensor block layout: blob[num_tokens, H, S] uint8 (token-major),
S = nb*(1+UB+SB); per (tok, head) = [ scale(nb) | upper(nb*UB) | shared(nb*SB) ].
Provides kv_slot_size / planes_to_blob (write path) / blob_to_planes (read path).
The round-trip + kernel-equality is the correctness gate for the layout.
"""
import numpy as np


def kv_slot_size(pk: dict) -> tuple[int, int, int, int]:
    """pack_kv dict -> (S, nb, UB, SB), S = slot bytes per token per head."""
    nb, UB, SB = int(pk["nb"]), int(pk["UB"]), int(pk["SB"])
    S = nb * (1 + UB + SB)
    return S, nb, UB, SB


def planes_to_blob(pk: dict) -> np.ndarray:
    """pack_kv dict -> uint8 blob [L, H, S] (token-major, bytes contiguous per token-head)."""
    H, L = int(pk["H"]), int(pk["L"])
    S, nb, UB, SB = kv_slot_size(pk)
    se = pk["scale_exp"].astype(np.uint8)          # [H,nb,L]
    up = pk["upper"]                                # [H,nb,L,UB]
    sh = pk["shared"]                               # [H,nb,L,SB]
    blob = np.empty((L, H, S), dtype=np.uint8)
    for b in range(nb):
        off = b * (1 + UB + SB)
        blob[:, :, off] = se[:, b, :].T                                 # scale  [L,H]
        blob[:, :, off + 1:off + 1 + UB] = up[:, b, :, :].transpose(1, 0, 2)   # upper [L,H,UB]
        blob[:, :, off + 1 + UB:off + 1 + UB + SB] = sh[:, b, :, :].transpose(1, 0, 2)
    return blob


def blob_to_planes(blob: np.ndarray, H: int, L: int, D: int, nb: int, UB: int, SB: int,
                   l_b: int, g_size: int) -> dict:
    """Inverse: uint8 blob [L,H,S] -> pack_kv-style dict the kernel indexes."""
    se = np.empty((H, nb, L), dtype=np.int8)
    up = np.empty((H, nb, L, UB), dtype=np.uint8)
    sh = np.empty((H, nb, L, SB), dtype=np.uint8)
    for b in range(nb):
        off = b * (1 + UB + SB)
        se[:, b, :] = blob[:, :, off].T.view(np.int8)
        up[:, b, :, :] = blob[:, :, off + 1:off + 1 + UB].transpose(1, 0, 2)
        sh[:, b, :, :] = blob[:, :, off + 1 + UB:off + 1 + UB + SB].transpose(1, 0, 2)
    return dict(scale_exp=np.ascontiguousarray(se), upper=np.ascontiguousarray(up),
                shared=np.ascontiguousarray(sh), H=H, L=L, D=D, nb=nb, l_b=l_b, g_size=g_size, UB=UB, SB=SB)
