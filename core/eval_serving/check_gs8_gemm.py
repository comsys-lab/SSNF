"""Correctness: generalized Triton SSNF weight-GEMM (gs8 + gs16) vs the C++ dequant reference.
Reference = torch.ops.ssnf.wonly_dequant_bf16(...,u,gs) @ matmul (the M>=thr branch, ground truth).
PASS = rel err ~2.6e-3 (bf16 accumulate). If it FAILs with much larger error, rebuild the
extension against the installed torch/CUDA before suspecting the kernels.
"""
from __future__ import annotations

import os, sys
from typing import Any
import torch
import ssnf_vllm  # registers torch.ops.ssnf
from ssnf_vllm.pack_gpu import pack_weight_ssnf_gpu
from ssnf_vllm.ssnf_triton import ssnf_gemm, ssnf_gemm_sk, ssnf_gemm_best, ssnf_gemm_ra, pack_upper_ra

torch.manual_seed(0)
DEV = "cuda"
U = 3

def ref_y(x: torch.Tensor, p: dict[str, Any], OUT: int, K: int, nb: int, g_size: int) -> torch.Tensor:
    """Ground-truth y = x @ W.t() using the C++ dequant reference."""
    W = torch.ops.ssnf.wonly_dequant_bf16(p["scale_exp"], p["upper_cm"], p["shared_cm"], OUT, K, nb, U, g_size)
    return x @ W.t()

def relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative Frobenius error of a vs reference b."""
    return (a.float() - b.float()).norm().item() / b.float().norm().clamp(min=1e-9).norm().item()

for g_size in (16, 8):
    OUT, K = 512, 4096
    W = torch.randn(OUT, K, dtype=torch.bfloat16, device=DEV) * 0.05
    p = pack_weight_ssnf_gpu(W, U, g_size)
    nb = p["nb"]; SB = p["shared_cm"].shape[-1]
    print(f"\n===== gs={g_size}  (SB={SB} bytes/col, n_group={p['n_group']}) =====")
    ra = pack_upper_ra(p["upper_cm"])
    for M in (64, 128, 256, 512):
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV) * 0.1
        yref = ref_y(x, p, OUT, K, nb, g_size)
        y_rw  = ssnf_gemm(x, p["scale_exp"], p["upper_cm"], p["shared_cm"], M, OUT, K, nb, g_size=g_size)
        y_sk  = ssnf_gemm_sk(x, p["scale_exp"], p["upper_cm"], p["shared_cm"], M, OUT, K, nb, SPLIT_K=4, g_size=g_size)
        y_bst = ssnf_gemm_best(x, p["scale_exp"], p["upper_cm"], p["shared_cm"], M, OUT, K, nb, g_size=g_size)
        y_ra  = ssnf_gemm_ra(x, p["scale_exp"], ra, p["shared_cm"], M, OUT, K, nb, g_size=g_size)
        print(f"  M={M:4d}  rw={relerr(y_rw,yref):.2e}  sk={relerr(y_sk,yref):.2e}  "
              f"best={relerr(y_bst,yref):.2e}  ra={relerr(y_ra,yref):.2e}")
print("\nOK (rel err ~1e-3 = bf16 rounding is PASS; ~1e-1+ = WRONG)")
