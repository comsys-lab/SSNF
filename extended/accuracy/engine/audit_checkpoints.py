"""Audit official packed checkpoints and their unquantized transformed tensors."""
from pathlib import Path
import hashlib,json,math,time
import torch
from safetensors import safe_open
R=Path(__file__).resolve().parents[1]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
if __name__=='__main__':
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 inp=read(R/'inputs/manifest.json');base=Path(inp['model']['snapshot']);index=read(base/'model.safetensors.index.json')['weight_map']
 def tensor(key):
  with safe_open(base/index[key],framework='pt',device='cpu') as f:return f.get_tensor(key).float().cuda()
 E=tensor('model.embed_tokens.weight');H=tensor('lm_head.weight');N=tensor('model.norm.weight');H=H*N[None]
 output=[]
 for tag,auditname in [('Llama-2-7B-QServe','qserve_checkpoint_audit.json'),('Llama-2-7B-QServe-g128','qserve_g128_checkpoint_audit.json')]:
  t=time.monotonic();folder=R/'checkpoints'/tag;path=folder/'pytorch_model.bin';original=read(R/'checkpoints'/auditname)
  assert sha(path)==original['file_sha256'];v=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
  assert len(v)==original['tensor_count']
  cfg=read(folder/'config.json');bcfg=read(base/'config.json')
  for k in ['hidden_size','intermediate_size','num_attention_heads','num_hidden_layers','vocab_size','rms_norm_eps']:
   assert cfg[k]==bcfg[k],k
  assert sha(folder/'tokenizer.model')==sha(base/'tokenizer.model')
  for k,x in v.items():
   if 'norm' in k:assert (x==1).all(),k
  Q=v['model.embed_tokens.weight'].float().cuda();qhead=v['lm_head.weight'].float().cuda()
  # Infer the common rotation from one fixed vocabulary subset. Validate on held-out rows
  # and on the independently folded language-model head. This never changes the checkpoint.
  fit=8192
  rotation=torch.linalg.solve(E[:fit].T@E[:fit],E[:fit].T@Q[:fit])
  embedding_relative_rmse=((E[fit:]@rotation-Q[fit:]).square().mean()/Q[fit:].square().mean()).sqrt().item()
  head_relative_rmse=((H@rotation-qhead).square().mean()/qhead.square().mean()).sqrt().item()
  orthogonality_relative_rmse=((rotation.T@rotation-torch.eye(4096,device='cuda')).square().mean()*4096).sqrt().item()
  report=dict(tag=tag,checkpoint_sha256=original['file_sha256'],revision=original['revision'],tensor_count=len(v),tensor_bytes=sum(x.numel()*x.element_size() for x in v.values()),
   model_config_matches=True,tokenizer_matches=True,norms_folded_to_ones=True,
   original_base=inp['model'],rotation_fit_rows=fit,heldout_embedding_relative_rmse=embedding_relative_rmse,
   folded_lm_head_relative_rmse=head_relative_rmse,orthogonality_relative_rmse=orthogonality_relative_rmse,seconds=time.monotonic()-t)
  print(json.dumps(report),flush=True)
  assert embedding_relative_rmse<1e-3 and head_relative_rmse<1e-3 and orthogonality_relative_rmse<1e-3,report
  output.append(report)
  del v,Q,qhead,rotation;torch.cuda.empty_cache()
 (R/'artifacts/checkpoint_audit.json').write_text(json.dumps(dict(passed=True,checkpoints=output),indent=2)+'\n')
