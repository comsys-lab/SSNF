"""One frozen tensor-wise MXFP condition, with gated 4/25/96 evaluation."""
from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, importlib.util, json, math, os, signal, sys, time, zipfile
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[3]
sys.path.insert(0,str(RUN/'source'))
import regex
import numpy as np
import torch
from engine import ppl, codecs, serialization
from tasks import adapters, generate
spec=importlib.util.spec_from_file_location('frozen_tensor_policy',RUN/'source/tensorwise_policy.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,x): ppl.atomic_json(Path(p),x)
def event(name,**kw):
    print(json.dumps({'event':name,'unix':time.time(),**kw},ensure_ascii=False),flush=True)
def require(ok,message):
    if not ok: raise RuntimeError(message)

def real_field_probe(x,actual,bits,path):
    """Pack GPU-selected numerical codes/scales; CPU decoder never reselects codes."""
    x=x.reshape(-1,32).float()
    exponent=torch.floor(torch.log2(x.abs().amax(-1).clamp_min(1e-30)))-2
    require(bool(((exponent>=-127)&(exponent<=127)).all()),'E8M0 range')
    # Actual quantizer output is already selected. Derive its exact finite code.
    unit=(actual.reshape(-1,32).float()/torch.exp2(exponent)[:,None]).cpu().numpy()
    book=serialization.positive_codebook(bits)
    mag=np.searchsorted(book,np.abs(unit))
    require(bool((mag<len(book)).all()),'MXFP codebook range')
    require(np.array_equal(book[mag],np.abs(unit)),'actual value outside MXFP codebook')
    codes=mag.astype(np.uint8)|((unit<0).astype(np.uint8)<<(bits-1))
    payload=serialization._pack_codes(codes,bits)
    scales=(exponent.cpu().numpy().astype(np.int16)+127).astype(np.uint8)[:,None]
    raw=np.concatenate([payload,scales],axis=1)
    decoded=serialization.decode_blocks(raw,bits,'mx')
    require(torch.equal(decoded.to(actual.dtype),actual.reshape(-1,32).cpu()),'independent actual-field decode')
    blob=raw.tobytes();Path(path).write_bytes(blob)
    return {'elements':int(x.numel()),'bytes':len(blob),'bits':bits,
            'sha256':hashlib.sha256(blob).hexdigest(),'independent_decode_exact':True}

class AuditedTensorEngine(policy.TensorEngine):
    def __init__(self,args,started):
        self.probes=[];self.seen_kv=set();self.ready_for_probes=False
        super().__init__(args,started)
        require(len(self.targets)==224 and len(self.layers)==32,'model scope')
        require(sum(x.numel() for x in self.master.values())==6979321856,'weight elements')
        require(self.model.config.num_key_value_heads==8,'KV heads')
    @torch.no_grad()
    def prepare(self,method):
        frozen=RUN/'results/frozen_tensor_map.json'
        if frozen.exists():
            saved=read(frozen)
            require(saved['protocol_hash']==self.run_protocol_hash,'resume map protocol mismatch')
            self.storage=saved['storage'];self.map_calibration=saved['calibration']
            self.tensor_kv={p:self.storage[p]['format_by_layer'] for p in ['k','v']}
            self.method='mx_tensor';self.scope='wkv';self.w_weighted=True;self.k_weighted=True
            for index,(name,module) in enumerate(self.targets):
                flat=self.master[name].reshape(-1,32)
                bits=self.storage['w']['formats'][name]
                for start in range(0,len(flat),self.a.weight_chunk_blocks):
                    if self.should_stop():raise TimeoutError('resume preparation time limit')
                    end=min(len(flat),start+self.a.weight_chunk_blocks)
                    module.weight.reshape(-1,32)[start:end].copy_(codecs.mxfp(flat[start:end].to(self.device).float(),2,bits-3).to(self.dtype))
                event('resume_weight',done=index+1,total=224)
            self.active=True;self.tensor_prepared=True
        else:
            super().prepare(method)
            proof=read(BASE/'calibration/proof.json')
            require(self.map_calibration['train']['token_sha256']==proof['token_sha256'],'train token hash')
            require(self.map_calibration['train']['sha256']==proof['train_sha256'],'train text hash')
            write(frozen,{'protocol_hash':self.run_protocol_hash,'storage':self.storage,'calibration':self.map_calibration})
        self.frozen_map_hash=sha(frozen)
        cost={}
        weight=sum(int(module.weight.numel()*(self.storage['w']['formats'][name]+.25)/8)+32 for name,module in self.targets)+28
        require(weight==self.storage['w']['actual_bytes'] and weight<=5016394752,'weight accounting')
        for context in [2048,16384]:
            planes={'w':weight}
            for p,hi,expected2,expected16 in [('k',19,45614084,364905476),('v',3,37225476,297796612)]:
                formats=self.tensor_kv[p]
                require(len(formats)==32 and formats.count(6)==hi and set(formats)<={4,6},'tensor KV count')
                planes[p]=sum(int(context*8*128*(bits+.25)/8)+32 for bits in formats)+4
                require(planes[p]==(expected2 if context==2048 else expected16),'KV byte formula')
            cap={'w':5016394752,'k':46138368 if context==2048 else 369099776,'v':37749760 if context==2048 else 301990912}
            require(all(planes[p]<=cap[p] for p in planes),'plane cap')
            cost[str(context)]={'planes':planes,'total':sum(planes.values()),'cap':sum(cap.values()),'slack':sum(cap.values())-sum(planes.values())}
        self.accounting=cost
        for index,(name,module) in enumerate(self.targets):
            x=self.master[name].reshape(-1,32)[:4].to(self.device).float()
            actual=module.weight.detach().reshape(-1,32)[:4]
            bits=self.storage['w']['formats'][name]
            record=real_field_probe(x,actual,bits,RUN/f'probes/w_{index:03d}.bin')
            self.probes.append({'plane':'w','tensor':name,**record})
        self.ready_for_probes=True
        write(RUN/'results/encoding_audit.json',{'accounting':cost,'frozen_map_sha256':self.frozen_map_hash,'probes':self.probes,'full_packed_checkpoint_written':False})
        event('prepared',seconds=self.elapsed(),weight_probes=len(self.probes),accounting=cost)
    def quant_kv(self,x,plane,layer):
        q=super().quant_kv(x,plane,layer)
        key=(plane,layer)
        if self.ready_for_probes and key not in self.seen_kv:
            source=x[:1,:1,:1,:].float();actual=q[:1,:1,:1,:]
            r=real_field_probe(source,actual,self.tensor_kv[plane][layer],RUN/f'probes/{plane}_{layer:02d}.bin')
            self.probes.append({'plane':plane,'layer':layer,**r});self.seen_kv.add(key)
        return q

def validate_rows(rows,baseline,allowed,protocol_hash,map_hash):
    require(set(rows)<=allowed,'unexpected task/sample')
    for key,r in rows.items():
        require(r['protocol_hash']==protocol_hash and r['frozen_map_sha256']==map_hash,'row provenance')
        ref=baseline[key]
        for f in ['document_sha256','input_token_ids_sha256','prompt_tokens','answers','all_classes']:
            require(r[f]==ref[f],f'baseline input mismatch: {key} {f}')
        require(math.isfinite(r['score']) and 0<=r['score']<=1,'score range')

def summary(stage,keys,rows,engine,protocol_hash):
    require(keys<=set(rows),'stage missing examples')
    require(len(engine.probes)==288 and len(engine.seen_kv)==64,'288 actual probes')
    means={t:100*sum(rows[k]['score'] for k in keys if k[0]==t)/sum(k[0]==t for k in keys) for t in TASKS}
    result={'status':'complete','stage':stage,'valid_unique_examples':len(keys),'all_completed_examples':len(rows),
            'tasks':means,'macro_score':sum(means.values())/12,'validation_only':True,'protocol_hash':protocol_hash,
            'frozen_map_sha256':engine.frozen_map_hash,'accounting':engine.accounting,'probes_passed':len(engine.probes),
            'total_seconds':engine.elapsed(),'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'updated_unix':time.time()}
    write(RUN/f'results/{stage}_summary.json',result)
    write(RUN/'results/probe_audit.json',engine.probes)
    event('stage_complete',**result)
    return result

def final_audit(rows,baseline,engine,protocol_hash,helper):
    # Re-read saved outputs and independently recompute all per-example metrics.
    reread={}
    for line in (RUN/'results/samples.jsonl').read_text().splitlines():
        r=json.loads(line);k=(r['task'],r['sample_id']);require(k not in reread,'duplicate saved sample');reread[k]=r
    require(set(reread)==ALL,'96 exact IDs')
    validate_rows(reread,baseline,ALL,protocol_hash,engine.frozen_map_hash)
    with zipfile.ZipFile(DATASET) as z:
        for task in TASKS:
            examples=[json.loads(x) for x in z.read(f'data/{task}.jsonl').decode().splitlines() if x.strip()]
            for sid in IDS[task]:
                r=reread[(task,sid)];ex=examples[sid]
                rescored=adapters.longbench_score(helper,task,ex,r['text'])
                require(abs(rescored['score']-r['score'])<=1e-8,'independent score mismatch')
                require(engine.tok.decode(r['raw_generated_token_ids'],skip_special_tokens=True)==r['text'],'saved text/token mismatch')
                require(len(r['raw_generated_token_ids'])<=helper.DATASET2MAXLEN[task],'output limit')
                require(set(r['cache_audit']['cache_update_calls'])==set(map(str,range(32))),'saved cache layers')
                require(all(v==r['generation_steps'] for v in r['cache_audit']['cache_update_calls'].values()),'saved cache updates')
    result=summary('validation',ALL,reread,engine,protocol_hash)
    rng=np.random.default_rng(CFG['bootstrap_seed']);draws=np.zeros(CFG['bootstrap_draws']);taskdiff={}
    for task in TASKS:
        delta=np.array([100*(baseline[(task,s)]['score']-reread[(task,s)]['score']) for s in IDS[task]])
        taskdiff[task]=float(delta.mean());draws+=delta[rng.integers(0,8,size=(CFG['bootstrap_draws'],8))].mean(1)/12
    audit={'passed':True,'examples':96,'tasks':12,'independent_rescore_mismatches':0,'probes':288,
           'macro_score':result['macro_score'],'SSNF_minus_tensor_points':sum(taskdiff.values())/12,
           'SSNF_minus_tensor_CI95':np.quantile(draws,[.025,.975]).tolist(),'per_task_SSNF_minus_tensor':taskdiff,
           'bootstrap_seed':CFG['bootstrap_seed'],'bootstrap_draws':CFG['bootstrap_draws'],'accounting':engine.accounting,
           'samples_sha256':sha(RUN/'results/samples.jsonl'),'frozen_map_sha256':engine.frozen_map_hash,
           'protocol_hash':protocol_hash,'updated_unix':time.time()}
    write(RUN/'verification.json',audit)
    write(RUN/'status.json',{'status':'complete','verification_passed':True,**result})
    report=['# Tensor-wise MXFP4/MXFP6 LongBench validation','',
            'Llama-3.1-8B-Instruct; same fixed 96 examples and 16K input cap. Internal validation, not held-out test.','',
            '| Method | Macro score | Examples |','| --- | --- | --- |',
            '| BF16 | 52.13 | 96 |',f'| SSNF Balanced | {"<span style=\"color:#d32f2f\">51.87</span>" if audit["SSNF_minus_tensor_points"]<0 else "51.87"} | 96 |',
            f'| Tensor-wise MXFP4/MXFP6 | {result["macro_score"]:.2f} | 96 |','| Block-wise MXFP4/MXFP6 Balanced | 49.56 | 96 |','',
            f'SSNF minus tensor-wise Mixed: {audit["SSNF_minus_tensor_points"]:+.2f} points; stratified paired 95% CI [{audit["SSNF_minus_tensor_CI95"][0]:+.2f}, {audit["SSNF_minus_tensor_CI95"][1]:+.2f}].','',
            'All 96 inputs match the stored BF16/SSNF baseline. Independent re-scoring and 288 actual field probes passed.','',
            '[Verification](verification.json), [protocol](protocol.json), [frozen map](results/frozen_tensor_map.json), [outputs](results/samples.jsonl).']
    (RUN/'RESULTS.md').write_text('\n'.join(report)+'\n')
    event('final_verified',**audit)

def main():
    started=time.time();torch.set_num_threads(4);torch.manual_seed(20261031)
    require(str(torch.__version__)=='2.12.0+cu132','torch version')
    require(__import__('transformers').__version__=='5.12.1','transformers version')
    require(torch.cuda.get_device_name()=='NVIDIA RTX PRO 4000 Blackwell','GPU type')
    require(torch.cuda.mem_get_info()[0]>20*1024**3,'GPU free memory')
    for rel,digest in CFG['source_sha256'].items():require(sha(RUN/rel)==digest,'source hash '+rel)
    require(sha(BASE/'manifest.json')==CFG['reference_manifest_sha256'],'baseline manifest')
    for name,digest in CFG['calibration_sha256'].items():require(sha(BASE/'calibration'/name)==digest,'calibration hash')
    require(sha(DATASET)==BM['dataset_sha256'],'dataset ZIP')
    require(sha(BASE/'validation_ids.json')==BM['validation_ids_sha256'],'validation IDs')
    require(sha(Path(BM['model'])/'config.json')==BM['model_config_sha256'],'checkpoint config')
    for method in ['bf16','ssnf_balanced']:
        protocol=read(BASE/f'results/{method}/protocol.json')
        require(protocol['dtype']=='bfloat16' and protocol['device']==torch.cuda.get_device_name(),'baseline environment')
        require(protocol['scorer_sha256']==sha(RUN/'source/longbench_scorer.py'),'baseline scorer hash')
    baseline={}
    for r in map(json.loads,(BASE/'results/ssnf_balanced/samples.jsonl').read_text().splitlines()):
        key=(r['task'],r['sample_id']);require(key not in baseline,'baseline duplicate');baseline[key]=r
    require(set(baseline)==ALL,'baseline IDs')
    command=[sys.argv[0],'--model','llama31i','--model-id',BM['model'],'--revision',BM['revision'],'--method','mx_tensor',
             '--device','cuda','--dtype','bfloat16','--maxlen','2048','--map-calib-windows','128','--map-policy','importance',
             '--out',str(RUN/'results'),'--hw',str(BASE/'calibration/H_w.pt'),'--hk',str(BASE/'calibration/H_K.pt'),
             '--corpus-text',str(Path(os.environ.get('SSNF_DATA_DIR','data'))/'wikitext/validation.txt'),'--dataset-split','validation',
             '--train-text',str(Path(os.environ.get('SSNF_DATA_DIR','data'))/'wikitext/train.txt'),'--max-windows','1',
             '--max-seconds','7200','--end-unix',str(started+7200),'--threads','4','--weight-chunk-blocks','32768','--offline']
    saved=sys.argv;sys.argv=command
    try:args=ppl.parse()
    finally:sys.argv=saved
    protocol={'manifest_sha256':sha(RUN/'manifest.json'),'source_sha256':CFG['source_sha256'],'worker_sha256':sha(__file__),
              'model':BM['model'],'revision':BM['revision'],'dtype':'bfloat16','device':torch.cuda.get_device_name(),
              'torch':str(torch.__version__),'transformers':__import__('transformers').__version__,'ids':IDS,'input_cap':16384,
              'calibration_windows':128,'calibration_length':2048,'generation':'greedy, batch1, checkpoint EOS, no stop strings',
              'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,'TF32_cudnn':torch.backends.cudnn.allow_tf32,'seed':20261031,
              'source_rule':'one FP4/FP6 per complete matrix or per complete layer/plane; train-only selection'}
    protocol['model_files']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in Path(BM['model']).iterdir() if p.is_file() and p.suffix in ['.json','.safetensors']}
    ph=ppl.digest(protocol);protocol['protocol_hash']=ph
    if (RUN/'protocol.json').exists():require(read(RUN/'protocol.json')==protocol,'resume protocol changed')
    else:write(RUN/'protocol.json',protocol)
    event('preflight_passed',protocol_hash=ph,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    engine=AuditedTensorEngine(args,started);engine.run_protocol_hash=ph
    try:
        helper=adapters.load_longbench_helpers(RUN/'source/longbench_scorer.py')
        engine.prepare('mx_tensor')
        rows={}
        if (RUN/'results/samples.jsonl').exists():
            for line in (RUN/'results/samples.jsonl').read_text().splitlines():
                r=json.loads(line);key=(r['task'],r['sample_id']);require(key not in rows,'duplicate resume row');rows[key]=r
        validate_rows(rows,baseline,ALL,ph,engine.frozen_map_hash)
        with zipfile.ZipFile(DATASET) as z:data={t:[json.loads(x) for x in z.read(f'data/{t}.jsonl').decode().splitlines() if x.strip()] for t in TASKS}
        smoke=[tuple(x) for x in CFG['smoke']]
        pilot=[(t,s) for t in TASKS for s in IDS[t][:2]]
        stages=[('smoke',smoke),('pilot',list(dict.fromkeys(pilot+smoke))),('validation',[(t,s) for t in TASKS for s in IDS[t]])]
        for stage,ordered in stages:
            write(RUN/'status.json',{'status':'running','stage':stage,'completed':len(rows),'requested':96,'updated_unix':time.time()})
            for task,sid in ordered:
                if (task,sid) in rows:continue
                if engine.should_stop():raise TimeoutError('overall time limit')
                ex=data[task][sid];prompt,add,audit=adapters.longbench_prompt(engine.tok,helper,task,ex,16384,True)
                ref=baseline[(task,sid)]
                require(ppl.digest(ex)==ref['document_sha256'] and audit['input_token_ids_sha256']==ref['input_token_ids_sha256'],'frozen prompt mismatch')
                options=SimpleNamespace(task_max_new_tokens=helper.DATASET2MAXLEN[task],task_stop_check=1,task_stops=(),task_add_special_tokens=add)
                verify=(task,sid) in [('qasper',50),('triviaqa',101)]
                tick=time.time();output,cache=generate.generate_batch(engine,[prompt],options,verify=verify)
                if verify:require(cache['cache_vs_recompute_prefill']['top1_agreement']==1,'prefill top1 differs')
                score=adapters.longbench_score(helper,task,ex,output[0]['text'])
                r={'candidate':'mx_tensor','protocol_hash':ph,'frozen_map_sha256':engine.frozen_map_hash,'split':'internal_validation',
                   'task':task,'sample_id':sid,'document_sha256':ppl.digest(ex),'input_token_ids_sha256':audit['input_token_ids_sha256'],
                   'prompt_audit':audit,'gold':ex['answers'],**output[0],**score,'cache_audit':cache,
                   'elapsed_s':time.time()-tick,'completed_unix':time.time()}
                validate_rows({(task,sid):r},baseline,ALL,ph,engine.frozen_map_hash)
                with (RUN/'results/samples.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                rows[(task,sid)]=r
                write(RUN/'status.json',{'status':'running','stage':stage,'completed':len(rows),'requested':96,'updated_unix':time.time()})
                event('sample',stage=stage,task=task,sample_id=sid,completed=len(rows),seconds=r['elapsed_s'],input_tokens=r['prompt_tokens'])
            if stage=='smoke':
                require(len(rows)>=4 and len(engine.probes)==288,'smoke probes/count')
                write(RUN/'results/smoke_summary.json',{'status':'complete','examples':4,'probes':288,'longest_input_tokens':16382,'elapsed_s':engine.elapsed()})
                event('smoke_passed',examples=4,probes=288,elapsed_s=engine.elapsed())
            elif stage=='pilot':
                summary('pilot',set(ordered),rows,engine,ph)
                measured=sum(r['elapsed_s'] for r in rows.values())
                eta=(96-len(rows))*measured/len(rows)*1.5+60
                require(engine.elapsed()+eta<7200,'pilot ETA exceeds budget')
                event('pilot_passed',examples=25,remaining_estimated_seconds=eta)
        final_audit(rows,baseline,engine,ph,adapters.load_longbench_helpers(RUN/'source/longbench_scorer.py'))
    finally:engine.close()

CFG=read(RUN/'manifest.json');BASE=Path(CFG['baseline_dir']);BM=read(RUN/'baseline_manifest.json')
TASKS=BM['tasks'];IDS=read(RUN/'baseline_validation_ids.json');ALL={(t,s) for t in TASKS for s in IDS[t]};DATASET=Path(BM['dataset'])
if __name__=='__main__':
    signal.signal(signal.SIGTERM,ppl.request_stop);signal.signal(signal.SIGINT,ppl.request_stop)
    try:main()
    except BaseException as exc:
        event('failure',error=repr(exc));write(RUN/'failure.json',{'error':repr(exc),'unix':time.time()})
        write(RUN/'status.json',{'status':'failed','error':repr(exc),'updated_unix':time.time()});raise
