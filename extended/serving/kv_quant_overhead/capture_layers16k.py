import os,sys,json,time,hashlib,argparse
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES','3');os.environ['HF_HUB_OFFLINE']='1';os.environ['OMP_NUM_THREADS']='4'
import torch,torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;torch.set_num_threads(4)
a=argparse.ArgumentParser();a.add_argument('--model',choices=['L31','L27'],required=True);a=a.parse_args()
name={'L31':'NousResearch--Meta-Llama-3.1-8B','L27':'NousResearch--Llama-2-7b-hf'}[a.model];modelpath=next((Path(os.environ.get('HF_HUB_CACHE',Path.home()/'.cache/huggingface/hub'))/('models--'+name)/'snapshots').iterdir());out=HERE/'operands16k'/a.model;out.mkdir(parents=True,exist_ok=True)
train=Path(os.environ.get('SSNF_DATA_DIR','data'))/'wikitext/train.txt';text=train.read_text();tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True);ids=tok(text,return_tensors='pt',add_special_tokens=False)['input_ids'][0][:16384];torch.save(ids,out/'tokens.pt')
model=AutoModelForCausalLM.from_pretrained(modelpath,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval();layers=model.model.layers;selected=[0,len(layers)//2,len(layers)-1];current={};manifest=dict(model=str(modelpath),train=str(train),train_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),token_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),selection='first 16384 train tokens; no test or performance selection',layers=selected,files={})
orig=F.scaled_dot_product_attention
for i,l in enumerate(layers):
 def layerpre(mod,args,index=i):current['layer']=index
 l.register_forward_pre_hook(layerpre)
 if i in selected:
  for role,mod in [('qkv',l.self_attn.q_proj),('o',l.self_attn.o_proj),('gu',l.mlp.gate_proj),('down',l.mlp.down_proj)]:
   def pre(mod,args,index=i,r=role):
    p=out/f'layer{index}_x_{r}.pt';torch.save(args[0].detach().reshape(-1,args[0].shape[-1])[:128].cpu().contiguous(),p)
   mod.register_forward_pre_hook(pre)
def tapped(q,k,v,*args,**kwargs):
 i=current['layer']
 if i in selected:
  p=out/f'layer{i}_qkv.pt';torch.save(dict(q=q.detach().cpu(),k=k.detach().cpu(),v=v.detach().cpu()),p)
 return orig(q,k,v,*args,**kwargs)
F.scaled_dot_product_attention=tapped
with torch.no_grad():
 y=model.model(ids[None].cuda(),use_cache=False).last_hidden_state;assert torch.isfinite(y).all()
F.scaled_dot_product_attention=orig
for p in sorted(out.glob('*.pt')):
 obj=torch.load(p,weights_only=True);values=obj.values()if isinstance(obj,dict)else[obj]
 assert all(torch.isfinite(v).all()for v in values);manifest['files'][p.name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),shapes=[list(v.shape)for v in values])
assert len([p for p in out.glob('layer*_qkv.pt') if '_x_' not in p.name])==3
manifest['complete']=True;(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
