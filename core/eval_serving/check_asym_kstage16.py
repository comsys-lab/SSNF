"""Correctness of the kgs16 asym K-staging / TC-QK decode kernels vs the known-good cpa-only kernel.
Same packed cache (K u3/gs16, V u4/gs32); compares kv_decode_paged_asym under 3 dispatch paths:
  cpa-only (SSNF_KV_KSTAGE=0), kstage-only (SSNF_KV_TCQK=0), tcqk (default). rel err ~1e-5 = PASS.
On FAIL with much larger error, rebuild the extension (python setup.py build_ext --inplace) first."""
from __future__ import annotations

import os, numpy as np, torch
from typing import Any
import ssnf_cuda  # noqa
from ssnf_lib.pack import pack_kv
from ssnf_vllm.kv_layout import kv_slot_size, planes_to_blob
dev="cuda"; rng=np.random.default_rng(0); torch.manual_seed(0)
B, CTX, BS = 8, 2048, 16
QH, KVH, D = 32, 8, 128

def make_cache(KU: int, KGS: int, VU: int, VGS: int) -> dict[str, Any]:
    """Build a packed paged KV cache (B copies) plus block table / seq lens for the given K/V formats."""
    K0=rng.standard_normal((KVH,CTX,D)).astype(np.float32); V0=rng.standard_normal((KVH,CTX,D)).astype(np.float32)
    pK,pV=pack_kv(K0,KU,KGS),pack_kv(V0,VU,VGS)
    _,nb,KUB,KSB=kv_slot_size(pK); phK=nb*(1+KUB+KSB)
    _,_,VUB,VSB=kv_slot_size(pV); phV=nb*(1+VUB+VSB)
    slot=np.concatenate([planes_to_blob(pK),planes_to_blob(pV)],axis=-1)
    nblocks=(CTX+BS-1)//BS
    flat=np.zeros((nblocks*BS,KVH,phK+phV),np.uint8); flat[:CTX]=slot
    cache=torch.from_numpy(flat.reshape(nblocks,BS,KVH,phK+phV)).cuda().repeat(B,1,1,1)
    bt=torch.arange(nblocks,dtype=torch.int32).view(1,nblocks).repeat(B,1).cuda()
    for i in range(B): bt[i]+=i*nblocks
    seq=torch.full((B,),CTX,dtype=torch.int32).cuda()
    return dict(cache=cache,bt=bt,seq=seq,nb=nb,KUB=KUB,KSB=KSB,VUB=VUB,VSB=VSB,phK=phK,phV=phV,KU=KU,KGS=KGS,VU=VU,VGS=VGS)

q1=torch.randn(B,QH,D,device=dev,dtype=torch.bfloat16)
def run(c: dict[str, Any]) -> torch.Tensor:
    """Run one asym paged-decode call on cache dict c under the current env dispatch."""
    return torch.ops.ssnf.kv_decode_paged_asym(q1,c['cache'],c['bt'],c['seq'],B,QH,KVH,D,c['nb'],
        c['KU'],c['KGS'],c['KUB'],c['KSB'],c['VU'],c['VGS'],c['VUB'],c['VSB'],BS,c['phK']+c['phV'],c['phK'])
def relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative Frobenius error of a vs reference b."""
    return (a.float()-b.float()).norm().item()/b.float().norm().item()

for tag,KGS in [("K gs16 slot160", 16), ("K gs32 slot168 (regression)", 32)]:
    c=make_cache(3,KGS,4,32)
    os.environ["SSNF_KV_KSTAGE"]="0"; os.environ["SSNF_KV_TCQK"]="0"; cpa=run(c); torch.cuda.synchronize()
    os.environ["SSNF_KV_KSTAGE"]="1"; os.environ["SSNF_KV_TCQK"]="0"; kst=run(c); torch.cuda.synchronize()
    os.environ["SSNF_KV_KSTAGE"]="1"; os.environ["SSNF_KV_TCQK"]="1"; tcq=run(c); torch.cuda.synchronize()
    print(f"{tag}: kstage-vs-cpa={relerr(kst,cpa):.2e}  tcqk-vs-cpa={relerr(tcq,cpa):.2e}  "
          f"{'PASS' if max(relerr(kst,cpa),relerr(tcq,cpa))<1e-3 else 'FAIL'}")
