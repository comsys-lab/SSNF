import sys,os,json,time,inspect,argparse,math,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(ROOT/'kernel_profile'));import bench as b
b.GPU=os.environ.get('SSNF_GPU_UUID','');original_inventory=b.inventory
b.inventory=lambda:dict(original_inventory(),scope='Actual Llama3.1 layer0/16/31 train BF16 K/V, query at final position; B4/B8 repeat B1',input_seed=None)
b.OPERANDS=HERE/'operands';b.OPERANDS16=HERE/'operands16k'
src=inspect.getsource(b.kv_case)
src=src.replace("kval = torch.randn(batch, context, hk, d, generator=gen, device='cuda', dtype=torch.bfloat16)","captured=torch.load((OPERANDS16 if context==16384 else OPERANDS)/'L31'/f'layer{CURRENT_LAYER}_qkv.pt',weights_only=True)\n    kval=captured['k'].permute(0,2,1,3).repeat(batch,1,1,1).contiguous().cuda()")
src=src.replace("vval = torch.randn(batch, context, hk, d, generator=gen, device='cuda', dtype=torch.bfloat16)","vval=captured['v'].permute(0,2,1,3).repeat(batch,1,1,1).contiguous().cuda()")
src=src.replace("q = torch.randn(batch, hq, d, generator=gen, device='cuda', dtype=torch.bfloat16)","q=captured['q'][:,:,-1,:].repeat(batch,1,1).contiguous().cuda()")
src=src.replace("funcs, refs, info = {}, {}, {}","funcs, refs, info = {}, {}, {}\n    appends={}\n    lastk=kval[:,-1].contiguous();lastv=vval[:,-1].contiguous();lastslots=torch.arange(batch,device='cuda',dtype=torch.int64)*context+context-1")
src=src.replace("out = torch.empty_like(q)","appends[fmt]=lambda kc=kc,vc=vc,scale=scale:reshape_and_cache_flash(lastk,lastv,kc,vc,lastslots,'auto',scale,scale)\n            out = torch.empty_like(q)")
src=src.replace("SimpleNamespace(layer_name='model.layers.0.self_attn.attn')","SimpleNamespace(layer_name=f'model.layers.{CURRENT_LAYER}.self_attn.attn')")
src=src.replace("funcs[fmt]=lambda impl=impl,cache=cache:","appends[fmt]=lambda impl=impl,cache=cache:impl.do_kv_cache_update(SimpleNamespace(layer_name=f'model.layers.{CURRENT_LAYER}.self_attn.attn'),lastk,lastv,cache,lastslots)\n            funcs[fmt]=lambda impl=impl,cache=cache:")
src=src.replace("    return funcs,refs,dict(kind='kv'","    if INCLUDE_APPEND:\n        funcs={f:(lambda update=appends[f],decode=funcs[f]:(update(),decode())[1])for f in formats}\n    return funcs,refs,dict(kind='kv'")
src=src.replace('synthetic BF16 normal; not model activations','actual post-RoPE train operands; B4/8 repeat one captured request').replace('layer-0 query importance for SSNF','selected-layer query importance for SSNF');exec(src,b.__dict__)
ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--profile',action='store_true');ap.add_argument('--context',type=int,default=16384);ap.add_argument('--batch',type=int,default=8);a=ap.parse_args();out=HERE/('kv_actual_smoke'if a.smoke else f'kv_actual_profile_{a.context}_{a.batch}'if a.profile else'kv_actual_full');out.mkdir(exist_ok=False);torch,nv,handle=b.environment(out);rows=[]
def timing(fn):
 for _ in range(4):fn()
 def sample(n):
  torch.cuda.synchronize();s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);t=time.monotonic();s.record()
  for _ in range(n):fn()
  e.record();e.synchronize();return s.elapsed_time(e)/1000,time.monotonic()-t
 est=sample(20)[0]/20;target=.03 if a.smoke else .4;n=max(20,math.ceil(target/est));sec,wall=sample(n);calls=n
 while sec<target*.9:
  more,w=sample(n);sec+=more;wall+=w;calls+=n
 return dict(cuda_s=sec,wall_s=wall,calls=calls,us_call=sec/calls*1e6,cuda_graph=False,stream='default; core paged kernels launch on default stream')
for layer in ([0]if a.smoke else[16]if a.profile else[0,16,31]):
 b.CURRENT_LAYER=layer
 for context in ([2048]if a.smoke else[a.context]if a.profile else[2048,16384]):
  for batch in ([1]if a.smoke else[a.batch]if a.profile else[1,4,8]):
   for include_append in [False,True]:
    b.INCLUDE_APPEND=include_append;t=time.monotonic();funcs,refs,meta=b.kv_case(context,batch,torch,b.FORMATS);checks={f:b.correctness(funcs[f],refs[f],meta['reference_tolerance'],torch)for f in b.FORMATS};prep=time.monotonic()-t
    source=(b.OPERANDS16 if context==16384 else b.OPERANDS)/'L31'/f'layer{layer}_qkv.pt';meta.update(operand_path=str(source),operand_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),layer=layer,append_included=include_append)
    if a.profile:
     for fmt,fn in funcs.items():
      for _ in range(5):fn()
      torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();fn();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
    else:
     for rnd in range(1 if a.smoke else 3):
      order=b.FORMATS[rnd:]+b.FORMATS[:rnd]
      for fmt in order:
       tm=timing(funcs[fmt]);row=dict(layer=layer,context=context,batch=batch,append_included=include_append,format=fmt,round=rnd,timing=tm,checks=checks,metadata=meta,preparation_wall_s=prep);rows.append(row);b.write(out/'partial.json',dict(rows=rows));print({k:row[k]for k in ['layer','context','batch','append_included','format','round']},tm,flush=True)
    del funcs,refs;torch.cuda.empty_cache()
b.write(out/'results.json',dict(complete=True,rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='CUDA Graph disabled for every method; actual train BF16 post-RoPE operands; fixed context cache; optional last-token re-encode into same cache before full QK/softmax/PV; no full-model scheduling or growing context; B4/B8 replicate one request'))
