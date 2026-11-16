"""Numerically validate native W4A8 packing/GEMM, RMSNorm and paged KV4 reads."""
import json,sys,time
from pathlib import Path
import torch
from backend import R,load_backend
from native_model import allocate_cache
if __name__=='__main__':
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
 backend=load_backend();sys.path.insert(0,str(R/'sources/omniserve'))
 from omniserve.modeling.layers.quantized_linear.w4a8_linear import W4A8OF16LinearDynamicInputScale
 from omniserve.modeling.layers.ctx_update_kv import ApplyBiasRopeUpdateKVCacheWrapper
 from omniserve.modeling.layers.decoding_attention import DecodingAttentionWrapper
 from types import SimpleNamespace
 report=[]
 for group in [-1,128]:
  signed=torch.randint(-8,8,(128,128),device='cuda')
  scale=1/32 if group==-1 else 1/16
  w=(signed.float()*(scale if group==-1 else 2*scale)).half()
  linear=torch.nn.Linear(128,128,bias=False,device='cuda',dtype=torch.float16);linear.weight.data.copy_(w)
  q=W4A8OF16LinearDynamicInputScale.from_linear(linear,4,group,s1_scale=torch.full((128,),scale,device='cuda',dtype=torch.float16),s2_scale=None if group==-1 else torch.full((128,1),2,device='cuda',dtype=torch.int8),zeros=torch.full((128,1),8,device='cuda',dtype=torch.int8))
  for batch in [1,16,32]:
   xi=torch.randint(-32,33,(batch,128),device='cuda',dtype=torch.int8);xi[:,0]=127
   x=xi.half()/16;quant=torch.empty_like(xi);scales=torch.empty(batch,device='cuda',dtype=torch.float16);sums=torch.empty_like(scales)
   backend.fused_kernels.invoke_quant_fuse_sum(quant,x,sums,scales)
   assert torch.equal(quant,xi);assert (scales==1/16).all()
   out=torch.full((batch,128),float('nan'),device='cuda',dtype=torch.float16)
   q(quant,scales,sums,out);torch.cuda.synchronize()
   expected=(xi.float()@w.float().T/16).half()
   torch.testing.assert_close(out,expected,rtol=1e-3,atol=.003)
   report.append(dict(operation='W4A8',group=group,batch=batch,max_abs_error=(out-expected).abs().max().item()))
 x=torch.randn(4,4096,device='cuda',dtype=torch.float16);weight=torch.rand(4096,device='cuda',dtype=torch.float16)+.5
 out=torch.empty_like(x);backend.layernorm_ops.rms_norm(out,x,weight,1e-5,False)
 expected=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-5)*weight.float()).half()
 torch.testing.assert_close(out,expected,rtol=.003,atol=.004)
 report.append(dict(operation='RMSNorm',max_abs_error=(out-expected).abs().max().item()))
 # Uniform attention with exactly representable per-head 4-bit value grids.
 # The mean changes at every token, detecting omitted, duplicated or future values.
 batch=2;length=257;caches,tables,pages,pagebytes=allocate_cache(batch,length,layers=1)
 meta=SimpleNamespace(padding_offsets=torch.zeros(batch,dtype=torch.int32,device='cuda'),cu_seqlens=torch.arange(batch+1,dtype=torch.int32,device='cuda'),retrieval_context_lens=torch.ones(batch,dtype=torch.int32,device='cuda'),streaming_context_lens=None,max_seq_len=1,retrieval_block_tables=tables,streaming_block_tables=[None])
 flags=torch.ones(32,dtype=torch.int32,device='cuda');rank=torch.arange(32,dtype=torch.int32,device='cuda')
 ctx=ApplyBiasRopeUpdateKVCacheWrapper(0,32,32,64,128,10000,None,4096,True,'fine_grained',dict(INT4_ENABLED=True,ZEROS_ENABLED=True),True)
 dec=DecodingAttentionWrapper(0,False,128,None,4096,64,128,10000,None,True,'fine_grained',dict(INT4_ENABLED=True,ZEROS_ENABLED=True),True,0,16,0,4096,1)
 maximum=0
 for t in range(length):
  qkv=torch.zeros((batch,12288),device='cuda',dtype=torch.float16)
  q,k,v=qkv.split([4096]*3,dim=-1);q=q.reshape(batch,32,128);k=k.reshape(batch,32,128);v=v.reshape(batch,32,128)
  grid=(torch.arange(128,device='cuda')%16).half()/16
  v.copy_(grid[None,None]+torch.arange(32,device='cuda').half()[None,:,None]/16+t/4+torch.arange(batch,device='cuda').half()[:,None,None]*4)
  k.copy_(grid[None,None]+.125)
  meta.max_seq_len=t+1;meta.retrieval_context_lens.fill_(t+1)
  if t==0:
   ctx(qkv,meta,flags,rank,0,0,0,0,32,0,None);actual=v.clone()
  else:
   actual,_=dec(q,k,v,meta,flags,rank,0,0,0,0,32,0,None,None)
  if t in [0,1,2,63,64,65,127,128,129,255,256]:
   expected=(grid[None,None]+torch.arange(32,device='cuda').half()[None,:,None]/16+t/8+torch.arange(batch,device='cuda').half()[:,None,None]*4).expand_as(actual)
   torch.cuda.synchronize();err=(actual-expected).abs().max().item();maximum=max(maximum,err)
   print(json.dumps(dict(event='cache_check',token=t+1,max_abs_error=err,mean_error=(actual.float()-expected.float()).mean().item(),actual_first=actual[0,0,:16].tolist(),expected_first=expected[0,0,:16].tolist())),flush=True)
   torch.testing.assert_close(actual,expected,rtol=.002,atol=.01)
   report.append(dict(operation='native KV4 uniform causal attention',cache_length=t+1,max_abs_error=err))
 assert all(torch.count_nonzero(x).item()>0 for pair in caches for x in pair)
 result=dict(passed=True,measurements=report,cache_page_bytes=pagebytes,uniform_grid='All V values and FP16 scale/zero exactly representable: scale=1/16, head offset=h/16, token offset=t/4. Original larger-offset diagnostic produced expected FP16 scale/zero rounding quantization error and is retained in logs.')
 (R/'artifacts/operation_smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
