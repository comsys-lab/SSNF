"""CPU-only packed append oracle; no CUDA extension or GPU packer calls."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np


def oracle(key, value, fmt, importance, core):
    """Return [N,H,slot] bytes for BF16-rounded CPU float32 inputs."""
    path=Path(core)/'quant/ssnf_lib/pack.py'
    spec=importlib.util.spec_from_file_location('s4_cpu_pack',path)
    pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)
    n,h,d=key.shape;nb=d//32
    planes=[]
    for index,x in enumerate((key,value)):
        blocks=np.asarray(x,np.float64).reshape(-1,32)
        if fmt=='ssnf':
            lb,gs=(3,16) if index==0 else (4,32)
            H=np.broadcast_to(importance,(n,h,d)).reshape(-1,32) if index==0 else None
            scale,upper,shared=pack.decompose(blocks,lb,gs,H=H,radius=0)
            exponent=np.rint(np.log2(scale)).astype(np.int8).view(np.uint8)
            codes=np.concatenate((exponent,pack._pack_codes_lsb(upper,8-lb),
                                  pack._pack_codes_lsb(shared,lb)),axis=-1)
        elif fmt=='mxfp6':
            # CUDA f_to_e2m3 uses round-to-nearest-even. The legacy CPU
            # pack_kv_mxfp6 uses tie-to-lower and is not the deployed oracle.
            exp=np.clip(np.floor(np.log2(np.maximum(np.abs(blocks).max(-1,keepdims=True),1e-30)))-2,-127,127)
            scaled=blocks/np.exp2(exp)
            grid=pack._e2m3_pos_grid()
            a=np.abs(scaled);hi=np.clip(np.searchsorted(grid,a),0,31);lo=np.maximum(hi-1,0)
            dl=np.abs(a-grid[lo]);dh=np.abs(a-grid[hi])
            lower=(dl<dh)|((dl==dh)&((lo&1)==0))
            mag=np.where(lower,lo,hi).astype(np.int64)
            words=mag|(np.signbit(scaled).astype(np.int64)<<5)
            codes=np.concatenate((exp.astype(np.int8).view(np.uint8),pack._pack_codes_lsb(words,6)),axis=-1)
        else:
            raise ValueError(fmt)
        planes.append(codes.reshape(n,h,-1))
    return np.concatenate(planes,axis=-1)
