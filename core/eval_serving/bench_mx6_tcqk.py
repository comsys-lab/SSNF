"""MXFP6 TC-QK port: ON (SSNF_MX6_TCQK=1) vs the deployed cp.async kernel, at MXFP6's own decode
operating points (2K -> B=106, 16K -> B=14). Paired, same process, interleaved rounds."""
from __future__ import annotations

import os, numpy as np, torch, statistics as st
from collections.abc import Callable
import ssnf_cuda  # noqa
rng=np.random.default_rng(0); torch.manual_seed(0)
BS=16; QH,KVH,D=32,8,128
def mk(B: int, CTX: int) -> tuple[Callable[[], torch.Tensor], int]:
    """Build a synthetic MXFP6 paged cache; return (decode-call closure, total cache bytes)."""
    nb=D//32; ph=nb*25
    slot=np.random.default_rng(1).integers(0,255,(CTX,KVH,2*ph),dtype=np.uint8)
    for pl in range(2):
        for blk in range(nb): slot[:,:,pl*ph+blk*25]=0     # E8M0 exponent byte 0 -> scale 2^0
    nb2=(CTX+BS-1)//BS; flat=np.zeros((nb2*BS,KVH,2*ph),np.uint8); flat[:CTX]=slot
    cache=torch.from_numpy(flat.reshape(nb2,BS,KVH,2*ph)).cuda().repeat(B,1,1,1)
    bt=torch.arange(nb2,dtype=torch.int32).view(1,nb2).repeat(B,1).cuda()
    for i in range(B): bt[i]+=i*nb2
    seq=torch.full((B,),CTX,dtype=torch.int32).cuda()
    q=torch.randn(B,QH,D,device="cuda",dtype=torch.bfloat16)
    return (lambda: torch.ops.ssnf.mxfp6_kv_decode_paged(q,cache,bt,seq,B,QH,KVH,D,nb,BS,2*ph,ph)), B*CTX*KVH*2*ph
def timeit(fn: Callable[[], object], it: int = 120, warm: int = 20) -> float:
    """Mean per-call time of fn in microseconds (CUDA-event timed)."""
    for _ in range(warm): fn()
    torch.cuda.synchronize(); e0,e1=torch.cuda.Event(True),torch.cuda.Event(True)
    e0.record()
    for _ in range(it): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1)/it*1000
for B,CTX in ((106,2048),(14,16384)):
    f,byt=mk(B,CTX)
    os.environ.pop("SSNF_MX6_TCQK",None); ref=f().float()
    os.environ["SSNF_MX6_TCQK"]="1";      on_=f().float()
    rel=((on_-ref).abs().max()/ref.abs().max()).item()
    s={"off":[],"on":[]}
    for r in range(3):
        os.environ.pop("SSNF_MX6_TCQK",None); s["off"].append(timeit(f))
        os.environ["SSNF_MX6_TCQK"]="1";      s["on"].append(timeit(f))
    off,on=st.median(s["off"]),st.median(s["on"])
    print(f"B={B:4d} ctx={CTX:6d} | OFF {off:8.1f}us ({byt/off/1e3:5.0f} GB/s)  ON {on:8.1f}us ({byt/on/1e3:5.0f} GB/s)"
          f"  -> {off/on:5.3f}x   rel_diff={rel:.1e}", flush=True)
    os.environ.pop("SSNF_MX6_TCQK",None); torch.cuda.empty_cache()
