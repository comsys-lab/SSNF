"""Audit paired whole-test raw NLL and publish direct, scope-labeled comparisons."""
from pathlib import Path
import csv,hashlib,json,math
R=Path(__file__).resolve().parent
B=R.parent/'20261029_kv_official'
D=R.parent/'20261029_kvquant_outlier_ablation'

def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def raw(p):return sorted([json.loads(s) for s in p.read_text().splitlines()],key=lambda x:x['window_id'])

if __name__=='__main__':
 assert read(R/'execution_complete.json')['complete']
 for name in ['checkpoint_audit','operation_smoke','baseline_parity']:
  assert read(R/'artifacts'/f'{name}.json')['passed']
 cases=[('bf16','BF16 baseline','unquantized',B/'results/bf16'),
        ('fp16','FP16 baseline','unquantized',R/'results/fp16'),
        ('ssnf_wkv','SSNF weight+KV','weight+KV',R/'results/ssnf_wkv'),
        ('qserve4','QServe W4A8KV4','weight+activation+KV',R/'results/qserve4'),
        ('qserve4g128','QServe W4A8KV4 g128','weight+activation+KV',R/'results/qserve4g128'),
        ('ssnf','SSNF KV-only','KV-only',B/'results/ssnf'),
        ('kivi2','KIVI-2','KV-only',B/'results/kivi2'),
        ('kivi4','KIVI-4','KV-only',B/'results/kivi4'),
        ('kvquant4','KVQuant-4, no outlier preservation','KV-only',D/'dense/results/kvquant4')]
 entries=[];reference=None;reference_meta=None;files={};metas={}
 for key,label,scope,folder in cases:
  s,m=read(folder/'summary.json'),read(folder/'meta.json');a=raw(folder/'windows.jsonl')
  assert s['complete'] and s['full_test'] and s['completed_windows']==330
  assert len(a)==330 and [x['window_id'] for x in a]==list(range(330))
  assert sum(x['target_count'] for x in a)==s['target_count']==338534
  nll=math.fsum(x['nll'] for x in a);ppl=math.exp(nll/338534)
  assert math.isclose(ppl,s['ppl'],rel_tol=1e-12)
  last=1
  for x,w in zip(a,m['windows']):
   assert all(x[k]==v for k,v in w.items())
   assert x['protocol_hash']==s['protocol_hash']==m['protocol_hash']
   assert x['target_start']==last and x['target_count']==x['end']-x['target_start'];last=x['end']
   assert math.isfinite(x['nll']) and x['nll']>0
   if key.startswith('qserve'):
    audit=x['audit'];assert audit['native_kv4_layers']==32 and audit['compressed_cache_reads_all_layers']==x['end']-x['start']-2
    assert audit['logit_projection_batch_size']==1 and audit['reset_cache_at_window_start']
   elif key=='ssnf_wkv':assert x['audit']['weight_linears']==224 and x['audit']['quantized_module_calls']==32
  assert last==338535
  if reference is None:reference,reference_meta=a,m
  for field in ['inputs','token_sha256','context','stride','cache_reset','position_ids','windows']:assert m[field]==reference_meta[field],(key,field)
  for x,y in zip(reference,a):assert x['input_sha256']==y['input_sha256']
  if key=='kvquant4':assert m['settings']['sparse']==0
  for name in ['meta.json','summary.json','windows.jsonl']:files[str(folder/name)]=sha(folder/name)
  if folder.is_relative_to(R):
   for path,wanted in m['sources'].items():assert sha(Path(path))==wanted,path
  metas[key]=m
  entries.append(dict(key=key,method=label,quantization_scope=scope,weight_activation_dtype=m['weight_activation_dtype'],
   ppl=s['ppl'],nll=s['nll'],targets=338534,windows=330,batch=m['batch_size'],raw_results=str(folder),
   ppl_increase_vs_bf16_pct=0.,ppl_increase_vs_fp16_pct=0.))
 bf16=entries[0]['ppl'];fp16=entries[1]['ppl'];ssnf=entries[2]['ppl']
 for x in entries:
  x['ppl_increase_vs_bf16_pct']=100*(x['ppl']/bf16-1)
  x['ppl_increase_vs_fp16_pct']=100*(x['ppl']/fp16-1)
 for name,table in [('comparison',entries),('comparison_weight_kv',entries[:5])]:
  with (R/(name+'.csv')).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 direct=[]
 for x in entries[3:5]:
  direct.append(dict(method=x['method'],ppl=x['ppl'],reference='new common FP16 baseline',ppl_increase_vs_fp16_pct=x['ppl_increase_vs_fp16_pct'],
   ssnf_wkv_ppl=ssnf,ssnf_ppl_lower_by=x['ppl']-ssnf,ssnf_ppl_lower_pct=100*(1-ssnf/x['ppl']),
   native_reference_rate_gap_pp=x['ppl_increase_vs_fp16_pct']-entries[2]['ppl_increase_vs_bf16_pct'],
   note='Rate gap uses FP16 for QServe and BF16 for SSNF; both baseline values included. Distinct precision/bit budgets remain.'))
 result=dict(complete=True,model='Llama-2-7B',dataset='WikiText-2 raw test',context=2048,stride=1024,targets=338534,windows=330,
  rows=entries,direct_weight_kv=direct,
  interpretation='Directly measured aligned evaluation inputs, target masks and positions. QServe uses official packed INT4 weights, native INT8 activations and packed per-head INT4 KV; SSNF uses BF16 arithmetic and its weight+KV representation. Not matched storage/activation precision. KV-only methods explicitly labeled.',
  implementation='Official QServe checkpoint, weight loading and native CUDA W4A8/KV4. One-token prompt attention is exactly V; later tokens use native paged decode. The unquantized FP16 LM head runs per row to preserve single-sequence arithmetic. No test calibration or parameter tuning in this run.')
 save(R/'comparison.json',result)
 save(R/'final_audit.json',dict(passed=True,complete=True,aligned_windows=330,targets=338534,paired_inputs_identical=True,no_missing_or_duplicate_targets=True,
  native_w4a8_kv4_all_layers=True,ssnf_weight_kv_224_linears=True,nll_ppl_recomputed=True,files=files,
  native_source_manifest=metas['qserve4']['sources']))
 lines=['Direct QServe W4A8KV4 PPL comparison','Same frozen checkpoint family, tokenizer and 338535 test tokens as the KIVI/KVQuant cohort.',
 'Context 2048, stride 1024, 330 independent windows, local positions, all 338534 targets scored once including the final short window.',
 'QServe official source revision 02b2925; per-channel and group-128 published checkpoints, SHA checked; underlying model audited through config/tokenizer and common orthogonal transforms of embedding and folded LM head.',
 'Actual native CUDA W4A8 linear kernels, INT8 activations and per-head asymmetric packed INT4 KV with FP16 scale/zero; dense attention, no LServe sparsity. One ground-truth token per step reads native KV4 cache.',
 'The first one-token prompt attention is exactly V, avoiding FlashAttention dependency. Output LM-head FP16 GEMV runs per independent row after proving batched transformer hidden states identical. Single/batch-32 NLL matches exactly in smoke. No custom quantization substitute.',
 'CUDA 13 compatibility: official ENABLE_BF16 macro retained; removed CUB Max replaced with cuda::maximum; optional serving/sparse imports excluded. The inference quantization algorithms are unchanged. See compatibility.patch and actual build.ninja flags.',
 'SSNF weights and KV both quantized with core calibration statistics; W lower3/group8 with search, K lower3/group16, V lower4/group32. BF16 arithmetic. Calib provenance recorded; old importance has no embedded dataset hash.',
 'QServe public checkpoint calibration data is not recollected here. No test-based refitting or configuration selection. Both published weight granularities were fixed before evaluation.',
 'Method-specific bit widths and activation precision differ; no matched-storage or serving throughput claim. Older KV-only SSNF is separately labeled.',
 'Batch sizes: QServe transformer 32 and LM-head 1; SSNF/FP16 1. Dedicated GPUs 0/2 for QServe, 1 SSNF, 3 FP16. Runtime is execution management, not a performance benchmark.',
 'Raw NLL, input hashes, protocol/source hashes and actual cache/module coverage are in results/<method>/{windows.jsonl,meta.json,summary.json}. Baseline and KV-only reference rows point to preserved original runs.',
 'Operation smoke includes exact native W4A8 integer checks, RMSNorm reference and independent 257-token sequences crossing 64/128/256-token cache boundaries. Initial large-offset synthetic values exposed expected FP16 scale/zero rounding and were replaced with exactly representable grids for the uniform-mean check; diagnostics retained.',
 'All methods passed smoke, identical two-window repeat NLL and full coverage audit. Existing BF16 first two windows replay with exact NLL parity. Pilot budgets, commands, watchdog receipts and failures are preserved under artifacts/, commands.jsonl and logs/.',
 'PPL and native-reference deltas:']
 lines += [f"{x['method']}: PPL {x['ppl']:.12f}; BF16-relative {x['ppl_increase_vs_bf16_pct']:+.8f}%; FP16-relative {x['ppl_increase_vs_fp16_pct']:+.8f}%" for x in entries]
 (R/'implementation.txt').write_text('\n'.join(lines)+'\n')
 print(json.dumps(result,indent=2))
