"""Isolated two-RTX3090 BF16 evaluator retaining the frozen core scorer/codec.

The original engine.ppl and engine.ppl_matched modules remain byte identical.
This wrapper changes only placement, cross-device H_K indexing, and protocol
provenance. It requires every model parameter and SDPA activation to be BF16.
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from engine import ppl, ppl_matched

BASE=ppl.Engine
ORIGINAL_LOAD=AutoModelForCausalLM.from_pretrained
ORIGINAL_HK=BASE.h_k
ORIGINAL_SDPA=BASE.sdpa
ORIGINAL_PROTOCOL=BASE.protocol


def sharded_load(*args,**kwargs):
    kwargs.update(device_map='balanced', max_memory={0:'20GiB',1:'20GiB'},low_cpu_mem_usage=True)
    model=ORIGINAL_LOAD(*args,**kwargs)
    devices={str(p.device) for p in model.parameters()}
    dtypes={str(p.dtype) for p in model.parameters() if p.is_floating_point()}
    if devices!={'cuda:0','cuda:1'} or dtypes!={'torch.bfloat16'}:
        raise RuntimeError(f'invalid sharded model placement/dtype: {devices}, {dtypes}')
    print(json.dumps({'event':'sharded_model_loaded','parameter_devices':sorted(devices),
                      'floating_parameter_dtypes':sorted(dtypes),
                      'device_map':model.hf_device_map}),flush=True)
    # The unchanged Engine constructor calls .to(cuda:0). Dispatch already
    # placed all modules; moving the whole model would invalidate sharding.
    model.to=lambda *ignored,**also_ignored:model
    return model


def device_local_hk(self,x,layer):
    if self.hk is None:raise RuntimeError('missing H_K calibration')
    h=self.hk[layer].to(x.device)
    nh=x.shape[1];hkv=h.shape[0]
    if nh%hkv:raise RuntimeError(f'GQA shape mismatch {nh}/{hkv}')
    idx=torch.arange(nh,device=x.device)//(nh//hkv)
    return h[idx].reshape(1,nh,1,x.shape[-1])


def checked_sdpa(self,query,key,value,*args,**kwargs):
    if {query.dtype,key.dtype,value.dtype}!={torch.bfloat16}:
        raise RuntimeError(f'non-BF16 attention activations: {query.dtype}, {key.dtype}, {value.dtype}')
    return ORIGINAL_SDPA(self,query,key,value,*args,**kwargs)


def sharded_protocol(self,method):
    details,_=ORIGINAL_PROTOCOL(self,method)
    details['backend']='HF SDPA; use_cache=False; accelerate balanced CUDA0+CUDA1'
    details['sharded_runtime_sha256']=ppl.sha_file(Path(__file__))
    details['device_map']=self.model.hf_device_map
    details['parameter_dtype_check']='all floating parameters bfloat16 on CUDA0/1'
    details['attention_dtype_check']='every SDPA query/key/value bfloat16'
    details['comparison_hash']=ppl.digest({'base_comparison_hash':details['comparison_hash'],
                                           'sharded_runtime_sha256':details['sharded_runtime_sha256'],
                                           'backend':details['backend'],'device_map':details['device_map']})
    return details,ppl.digest({k:v for k,v in details.items() if k!='hostname'})


def main():
    AutoModelForCausalLM.from_pretrained=sharded_load
    BASE.h_k=device_local_hk
    BASE.sdpa=checked_sdpa
    BASE.protocol=sharded_protocol
    ppl_matched.main()


if __name__=='__main__':main()
