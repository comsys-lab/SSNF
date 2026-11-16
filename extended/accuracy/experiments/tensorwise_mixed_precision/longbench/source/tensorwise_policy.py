"""Provisional fixed-budget matrix/layer MXFP PPL, fixed train calibration only."""
import json,math,os,time
from pathlib import Path
import torch
from engine import ppl,codecs
ppl.SCOPE['mx_tensor']=('wkv',True,True)
Base=ppl.Engine
class TensorEngine(Base):
 def protocol(self,method):
  d,_=super().protocol(method);d.update(tensor_allocation='one FP4/FP6 per weight matrix, per layer K and V',scale_offsets=[0],selection='training weighted reconstruction gain per added byte; deterministic greedy',selection_test_used=False);return d,ppl.digest({k:v for k,v in d.items() if k!='hostname'})
 @torch.no_grad()
 def prepare(self,method):
  if getattr(self,'tensor_prepared',False):return
  self.active=False;self.method=method;self.scope='wkv';self.w_weighted=True;self.k_weighted=True;self.enc_details=[];self.storage={}
  train,meta=self.load_ids('train',self.a.train_text);n=self.a.map_calib_windows;assert len(train)>=n*self.a.maxlen
  self.collect='mx'
  try:
   for i in range(n):
    if self.should_stop():raise TimeoutError('calibration time limit')
    self.model(train[i*self.a.maxlen:(i+1)*self.a.maxlen][None].to(self.device),use_cache=False)
    print(json.dumps({'event':'kv_calibration','done':i+1,'requested':n}),flush=True)
  finally:self.collect=None
  layers=len(self.layers);hkv=self.model.config.num_key_value_heads; dim=self.model.config.hidden_size//self.model.config.num_attention_heads
  self.tensor_kv={};kv_details={}
  for plane,budget in [('k',5.5),('v',4.5)]:
   el_layer=2048*hkv*dim;cap=layers*el_layer*budget//8+layers*32;mapbytes=(layers+7)//8;base=layers*el_layer*4.25/8+layers*32+mapbytes;extra=el_layer*2/8;count=max(0,min(layers,int((cap-base)//extra)))
   gains=[float((self.map_losses[('mx',plane,l)][0]/self.map_losses[('mx',plane,l)][1]).sum()) for l in range(layers)]
   selected=set(sorted(range(layers),key=lambda i:(-gains[i],i))[:count]);self.tensor_kv[plane]=[6 if i in selected else 4 for i in range(layers)]
   used=int(base+count*extra);assert used<=cap;kv_details[plane]={'cap_bytes':int(cap),'actual_bytes':used,'slack_bytes':int(cap-used),'format_by_layer':self.tensor_kv[plane],'gain_per_layer':gains,'reference_tokens':2048,'map_bytes':mapbytes}
  gains=[];total=sum(m.weight.numel() for _,m in self.targets);headers=32*len(self.targets);maps=(len(self.targets)+7)//8;cap=int(total*5.75/8)+headers;base=int(total*4.25/8)+headers+maps
  for j,(name,module) in enumerate(self.targets):
   source=self.master[name];flat=source.reshape(-1,32);h=self.h_w(name,self.device).reshape(-1,32);gain=0.
   for b in range(0,len(flat),self.a.weight_chunk_blocks):
    end=min(len(flat),b+self.a.weight_chunk_blocks);x=flat[b:end].to(self.device).float();hh=h[torch.arange(b,end,device=self.device)%len(h)];ql=codecs.mxfp(x,2,1);qh=codecs.mxfp(x,2,3);gain+=float((((ql-x).square()-(qh-x).square())*hh).sum(dtype=torch.float64))
   extra=source.numel()//4;gains.append({'name':name,'elements':source.numel(),'gain':gain,'extra_bytes':extra,'gain_per_byte':gain/extra});print(json.dumps({'event':'weight_gain','done':j+1,'total':len(self.targets)}),flush=True)
  available=cap-base;chosen=set()
  for item in sorted(gains,key=lambda z:(-z['gain_per_byte'],z['name'])):
   if item['gain']>0 and item['extra_bytes']<=available:available-=item['extra_bytes'];chosen.add(item['name'])
  for name,module in self.targets:
   bits=6 if name in chosen else 4;source=self.master[name];flat=source.reshape(-1,32)
   for b in range(0,len(flat),self.a.weight_chunk_blocks):
    end=min(len(flat),b+self.a.weight_chunk_blocks);x=flat[b:end].to(self.device).float();module.weight.reshape(-1,32)[b:end].copy_(codecs.mxfp(x,2,bits-3).to(self.dtype))
   self.enc_details.append({'tensor':name,'format':'MXFP'+str(bits),'elements':source.numel(),'payload_bytes':source.numel()*(bits+.25)/8,'header_bytes':32})
  self.storage={'w':{'cap_bytes':cap,'actual_bytes':cap-available,'slack_bytes':available,'map_bytes':maps,'formats':{n:(6 if n in chosen else 4) for n,_ in self.targets},'gain':gains},**kv_details}
  self.map_calibration={'train':meta,'segments':n,'segment_tokens':self.a.maxlen,'test_used':False,'weight_importance':self.calib_provenance.get('hw'),'query_importance':self.calib_provenance.get('hk')}
  out=Path(self.a.out);out.mkdir(parents=True,exist_ok=True);(out/'tensor_map.json').write_text(json.dumps({'storage':self.storage,'calibration':self.map_calibration},indent=2));self.active=True;self.tensor_prepared=True
 def quant_kv(self,x,plane,layer):return codecs.mxfp(x,2,self.tensor_kv[plane][layer]-3)
ppl.Engine=TensorEngine
if __name__=='__main__':ppl.main()
