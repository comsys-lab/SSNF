"""Validated stages and per-GPU watchdogs for the direct QServe comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,math,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;PROJECT=R.parents[3]
sys.path.insert(0,str(PROJECT))
from engine.kv_official_run import execute
os.environ['PYTHONPATH']=str(R)

def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def evaluate(method,stage,output,batch,gpu,timeout):
 execute(R,f'{method}_{output}_{stage}','qserve','engine.ppl_qserve',
  ['--method',method,'--stage',stage,'--output',output,'--batch-size',batch,'--max-seconds',timeout-15],timeout,idle=180,gpu_index=str(gpu))
 s=read(R/output/method/'summary.json');assert s['complete'] and s['stage']==stage
 return s

def valid_smoke(method,output):
 s=read(R/output/method/'summary.json');a=rows(R/output/method/'windows.jsonl')
 assert s['complete'] and s['target_count']==255 and len(a)==1 and math.isfinite(s['ppl'])
 if method.startswith('qserve'):
  assert a[0]['audit']['native_kv4_layers']==32 and a[0]['audit']['compressed_cache_reads_all_layers']==254
 elif method=='ssnf_wkv':assert a[0]['audit']['weight_linears']==224 and a[0]['audit']['quantized_module_calls']==32
 return s

def cohort(method,batch,gpu):
 complete=R/'results'/method/'summary.json'
 if complete.exists() and read(complete).get('full_test'):
  m=read(complete.with_name('meta.json'));a=rows(complete.with_name('windows.jsonl'))
  assert len(a)==330 and sum(x['target_count'] for x in a)==338534
  for p,h in m['sources'].items():assert sha(Path(p))==h
  return dict(method=method,state='already_complete',ppl=read(complete)['ppl'])
 if method.startswith('qserve'):
  smoke=valid_smoke(method,'smoke_rowhead')
  if method=='qserve4g128':evaluate(method,'smoke','smoke_single_final',1,gpu,300)
  single=read(R/('smoke' if method=='qserve4' else 'smoke_single_final')/method/'summary.json')
  delta=abs(single['nll']-smoke['nll'])/255
  assert delta<1e-6,(method,'single/batched mismatch',delta)
  save(R/'artifacts'/f'{method}_batch_parity.json',dict(passed=True,single=single,batch32=smoke,mean_nll_absolute_difference=delta))
 else:
  smoke=evaluate(method,'smoke','smoke_final',batch,gpu,300);valid_smoke(method,'smoke_final')
 pilot=evaluate(method,'pilot','results',batch,gpu,900)
 repeat=evaluate(method,'pilot','repeat_pilot',batch,gpu,900)
 a=rows(R/'results'/method/'windows.jsonl');b=rows(R/'repeat_pilot'/method/'windows.jsonl')
 assert len(a)==len(b)==2
 diffs=[]
 for x,y in zip(a,b):
  assert x['input_sha256']==y['input_sha256'] and x['target_count']==y['target_count']
  delta=abs(x['nll']-y['nll'])/x['target_count'];assert delta<1e-6,(method,delta);diffs.append(delta)
 metadata=read(R/'results'/method/'meta.json')
 counts={}
 for w in metadata['windows']:counts[w['end']-w['start']]=counts.get(w['end']-w['start'],0)+1
 groups=sum(math.ceil(v/batch) for v in counts.values())
 estimate=max(x['elapsed_group_s'] for x in a)*groups
 timeout=max(300,2*estimate+180)
 save(R/'artifacts'/f'{method}_pilot.json',dict(passed=True,smoke=smoke,pilot=pilot,repeat=repeat,mean_nll_repeat_absolute_difference=diffs,batch=batch,full_groups=groups,estimated_full_seconds=estimate,timeout_seconds=timeout))
 print(json.dumps(dict(event='pilot_passed',method=method,estimated_full_seconds=estimate)),flush=True)
 full=evaluate(method,'full','results',batch,gpu,timeout)
 assert full['full_test'] and full['completed_windows']==330 and full['target_count']==338534
 return dict(method=method,state='complete',ppl=full['ppl'])

def reference_parity():
 evaluate('bf16','pilot','parity',1,3,180)
 base=R.parent/'20261029_kv_official/results/bf16';a=rows(base/'windows.jsonl')[:2];b=rows(R/'parity/bf16/windows.jsonl')
 for x,y in zip(a,b):
  assert x['input_sha256']==y['input_sha256'] and x['target_count']==y['target_count'] and abs(x['nll']-y['nll'])/x['target_count']<1e-6
 save(R/'artifacts/baseline_parity.json',dict(passed=True,reference=str(base),mean_nll_differences=[abs(x['nll']-y['nll'])/x['target_count'] for x,y in zip(a,b)]))
 return cohort('fp16',1,3)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--methods',default='qserve4,qserve4g128,ssnf_wkv,fp16');a=p.parse_args()
 assert read(R/'artifacts/operation_smoke.json')['passed'] and read(R/'artifacts/checkpoint_audit.json')['passed']
 alljobs={'qserve4':lambda:cohort('qserve4',32,0),'qserve4g128':lambda:cohort('qserve4g128',32,2),'ssnf_wkv':lambda:cohort('ssnf_wkv',1,1),'fp16':reference_parity}
 jobs={m:alljobs[m] for m in a.methods.split(',')};finished=[]
 with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
  for result in pool.map(lambda fn:fn(),jobs.values()):
   finished.append(result);print(json.dumps(result),flush=True)
 save(R/'execution_complete.json',dict(complete=True,methods=finished,time=time.time()))
