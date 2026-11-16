"""Evaluate frozen matrix/layer MXFP map; serialize codes selected on the GPU."""
import hashlib,json,os,struct
from pathlib import Path
import numpy as np,torch
from engine import ppl,codecs,serialization
ppl.SCOPE['mx_tensor']=('wkv',True,True)
def probe_gpu(x,q,bits):
 z=x.detach().reshape(-1,32).float(); y=q.detach().reshape(-1,32).float();ex=torch.floor(torch.log2(z.abs().amax(-1).clamp_min(1e-30)))-2
 assert bool(((ex>=-127)&(ex<=127)).all()); scale=torch.exp2(ex)[:,None];u=y/scale
 book=torch.tensor(serialization.positive_codebook(bits),device=z.device);idx=torch.searchsorted(book,u.abs().contiguous());assert bool((idx<len(book)).all()) and bool((book[idx]==u.abs()).all())
 codes=(idx.to(torch.int32)|(torch.signbit(u).to(torch.int32)<<(bits-1))).cpu().numpy();sc=(ex.to(torch.int16).cpu().numpy()+127).astype(np.uint8)[:,None];raw=np.concatenate([serialization._pack_codes(codes,bits),sc],axis=1)
 dec=serialization.decode_blocks(raw,bits,'mx');assert torch.equal(dec,y.cpu());return struct.pack('<8sQBBBB12x',b'CTRLv001',len(z),1,bits,32,0)+raw.tobytes()
Base=ppl.Engine
class VerifiedEngine(Base):
 def protocol(self,method):
  d,_=super().protocol(method);d.update(tensor_map_path=os.environ['TENSOR_MAP_PATH'],tensor_map_sha256=ppl.sha_file(os.environ['TENSOR_MAP_PATH']),tensor_selection='frozen calibration selection; one format per matrix or layer K/V',scale_offsets=[0],probes='GPU selected codes and independent CPU decode');return d,ppl.digest({k:v for k,v in d.items() if k!='hostname'})
 @torch.no_grad()
 def prepare(self,method):
  if getattr(self,'verified_prepared',False):return
  self.active=False;self.method=method;self.scope='wkv';self.w_weighted=True;self.k_weighted=True;self.enc_details=[];self.probed=set()
  frozen=json.loads(Path(os.environ['TENSOR_MAP_PATH']).read_text());self.storage=frozen['storage'];self.map_calibration=frozen['calibration'];assert self.map_calibration['segments']==128 and not self.map_calibration['test_used']
  self.tensor_kv={k:self.storage[k]['format_by_layer'] for k in 'kv'};formats=self.storage['w']['formats'];assert set(formats)==set(n for n,_ in self.targets)
  self.artifact_dir=Path(self.a.out)/method/'byte_probes';self.artifact_dir.mkdir(parents=True,exist_ok=True)
  for name,module in self.targets:
   bits=formats[name];source=self.master[name];flat=source.reshape(-1,32)
   for b in range(0,len(flat),self.a.weight_chunk_blocks):
    if self.should_stop():raise TimeoutError('weight preparation time limit')
    end=min(len(flat),b+self.a.weight_chunk_blocks);x=flat[b:end].to(self.device).float();q=codecs.mxfp(x,2,bits-3)
    if b==0:
     elems=source.shape[1]*2;blob=probe_gpu(x[:elems//32],q[:elems//32],bits);(self.artifact_dir/(name+'.bin')).write_bytes(blob)
    module.weight.reshape(-1,32)[b:end].copy_(q.to(self.dtype))
   self.enc_details.append({'tensor':name,'format':'MXFP'+str(bits),'elements':source.numel(),'gpu_code_probe':True})
  self.active=True;self.verified_prepared=True
 def quant_kv(self,x,plane,layer):
  bits=self.tensor_kv[plane][layer];q=codecs.mxfp(x,2,bits-3);key=(plane,layer)
  if key not in self.probed:
   blob=probe_gpu(x[0,0,:1,:64].float(),q[0,0,:1,:64].float(),bits);(self.artifact_dir/f'{plane}_layer{layer}.bin').write_bytes(blob);self.probed.add(key)
  return q
ppl.Engine=VerifiedEngine
if __name__=='__main__':ppl.main()
