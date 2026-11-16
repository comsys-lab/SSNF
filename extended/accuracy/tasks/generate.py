"""Resumable GSM8K generation with the same weight/KV codecs as engine.ppl.

Engine CLI flags are shared. Extra flags are prefixed --task-. Run as
python -m tasks.generate --out ... --model llama31 --method bf16 [...].
"""
from __future__ import annotations
import argparse
from collections import Counter
import inspect
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

import torch
from transformers.cache_utils import DynamicCache
from engine import ppl
from engine import ppl_matched

VERSION='gsm8k-cache-v3'
CALC=re.compile(r'<<[^>]*>>')
STRICT=re.compile(r'####\s*(-?[0-9][0-9,]*\.?[0-9]*)')
NUM=re.compile(r'-?[0-9][0-9,]*\.?[0-9]*')
STOPS=('\nQuestion:','\n\n')


def norm(s):
    if s is None:return None
    try:return float(s.replace(',','').replace('$','').strip())
    except ValueError:return None


def score(text,gold):
    strict=STRICT.search(text);nums=NUM.findall(text)
    s=norm(strict.group(1)) if strict else None;f=norm(nums[-1]) if nums else None
    return {'strict_prediction':s,'flex_prediction':f,
            'strict_correct':gold is not None and s is not None and abs(s-gold)<1e-4,
            'flex_correct':gold is not None and f is not None and abs(f-gold)<1e-4}


def truncate(text,stops=STOPS):
    cuts=[text.find(stop) for stop in stops if stop in text]
    return text[:min(cuts)] if cuts else text


def last_logit_kwargs(model):
    parameters=inspect.signature(type(model).forward).parameters
    if 'logits_to_keep' in parameters:return {'logits_to_keep':1}
    if 'num_logits_to_keep' in parameters:return {'num_logits_to_keep':1}
    return {} # Explicitly recorded fallback for older/unsupported model APIs.


class QuantizedDynamicCache(DynamicCache):
    """Quantize only new RoPE-applied KV before storing the dequantized values.

    Block scales/groups are token-local along the channel dimension, so cached
    old tokens need not be quantized again. HF stores the dequantized tensors;
    this evaluates numerical accuracy, not packed-cache performance or memory.
    """
    def __init__(self,engine):
        super().__init__(config=engine.model.config)
        self.engine=engine;self.update_calls=Counter();self.new_elements=Counter()

    def update(self,key_states,value_states,layer_idx,*args,**kwargs):
        self.update_calls[layer_idx]+=1
        if 'k' in self.engine.scope:
            key_states=self.engine.quant_kv(key_states,'k',layer_idx)
        if 'v' in self.engine.scope:
            value_states=self.engine.quant_kv(value_states,'v',layer_idx)
        self.new_elements['k']+=key_states.numel();self.new_elements['v']+=value_states.numel()
        return super().update(key_states,value_states,layer_idx,*args,**kwargs)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def dataset(a):
    if a.task_train and a.task_test:
        train=read_jsonl(a.task_train);test=read_jsonl(a.task_test)
        source={'train_path':str(a.task_train),'train_sha256':ppl.sha_file(a.task_train),
                'test_path':str(a.task_test),'test_sha256':ppl.sha_file(a.task_test)}
    else:
        from datasets import load_dataset
        ds=load_dataset('openai/gsm8k','main')
        train=list(ds['train']);test=list(ds['test'])
        source={'dataset':'openai/gsm8k','config':'main','train_fingerprint':ds['train']._fingerprint,
                'test_fingerprint':ds['test']._fingerprint}
    if len(train)<a.task_nshot or not test:raise ValueError('empty/insufficient task data')
    shots=[f"Question: {ex['question']}\nAnswer: {CALC.sub('',ex['answer']).strip()}" for ex in train[:a.task_nshot]]
    preamble='\n\n'.join(shots)
    prompts=[f"{preamble}\n\nQuestion: {ex['question']}\nAnswer:" for ex in test]
    gold=[]
    for ex in test:
        m=STRICT.search(ex['answer'])
        if not m:raise ValueError('gold answer has no #### numeric target')
        gold.append(norm(m.group(1)))
    return test,prompts,gold,source


@torch.inference_mode()
def generate_batch(engine,prompts,a,verify=False):
    tok=engine.tok
    tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token_id=tok.eos_token_id
    stops=getattr(a,'task_stops',STOPS)
    enc=tok(prompts,padding=True,return_tensors='pt',add_special_tokens=getattr(a,'task_add_special_tokens',True))
    ids=enc['input_ids'].to(engine.device);mask=enc['attention_mask'].to(engine.device)
    max_context=getattr(engine.model.config,'max_position_embeddings',None)
    if max_context and ids.shape[1]+a.task_max_new_tokens>max_context:
        raise ValueError(f'prompt+generation exceeds model context: {ids.shape[1]}+{a.task_max_new_tokens}>{max_context}')
    positions=(mask.cumsum(-1)-1).masked_fill(mask==0,1)
    logit_kwargs=last_logit_kwargs(engine.model)
    check=None
    if verify:
        engine.active=True;engine.calls={}
        check=engine.model(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False,**logit_kwargs).logits[:,-1].float()
    engine.active=False # Cache.update applies KV codec once; never double-quantize in SDPA.
    cache=QuantizedDynamicCache(engine)
    eos=engine.model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos]) if eos is not None else set()
    generated=[[] for _ in prompts];finished=[False]*len(prompts);reasons=['max_new_tokens']*len(prompts)
    step_ids=ids;steps=0;parity=None
    for step in range(a.task_max_new_tokens):
        if engine.should_stop():raise TimeoutError('time budget exhausted in generation')
        positions=(mask.cumsum(-1)-1).masked_fill(mask==0,1)
        if step:positions=positions[:,-1:]
        result=engine.model(input_ids=step_ids,attention_mask=mask,position_ids=positions,past_key_values=cache,use_cache=True,**logit_kwargs)
        if result.past_key_values is not cache:raise RuntimeError('model replaced explicit quantizing cache')
        logits=result.logits[:,-1].float()
        if not torch.isfinite(logits).all():raise RuntimeError('nonfinite generation logits')
        if check is not None:
            delta=(logits-check).abs();parity={'max_abs_logits':delta.max().item(),
                'mean_abs_logits':delta.mean().item(),'top1_agreement':(logits.argmax(-1)==check.argmax(-1)).float().mean().item()}
            torch.testing.assert_close(logits,check,rtol=.01,atol=.125)
            check=None
        next_ids=logits.argmax(-1);steps+=1
        for j,token in enumerate(next_ids.tolist()):
            if finished[j]:continue
            generated[j].append(token)
            if token in eos:finished[j]=True;reasons[j]='eos'
        if (step+1)%a.task_stop_check==0 or all(finished) or step+1==a.task_max_new_tokens:
            texts=tok.batch_decode(generated,skip_special_tokens=True)
            for j,text in enumerate(texts):
                if any(s in text for s in stops):finished[j]=True;reasons[j]='stop_string'
        if all(finished):break
        step_ids=next_ids[:,None]
        mask=torch.cat([mask,torch.ones((len(prompts),1),device=mask.device,dtype=mask.dtype)],dim=1)
    if set(cache.update_calls)!=set(range(len(engine.layers))) or any(n!=steps for n in cache.update_calls.values()):
        raise RuntimeError(f'cache hook/layer mismatch: {dict(cache.update_calls)}, steps {steps}')
    texts=tok.batch_decode(generated,skip_special_tokens=True)
    return [{'text':truncate(text,stops),'raw_generated_token_ids':tokens,'stop_reason':reason,
             'prompt_tokens':int(maskrow.sum()),'generation_steps':steps} for text,tokens,reason,maskrow in
            zip(texts,generated,reasons,enc['attention_mask'])],{'cache_update_calls':dict(cache.update_calls),
            'new_kv_elements':dict(cache.new_elements),'cache_vs_recompute_prefill':parity,
            'logit_selection_kwargs':logit_kwargs,'logit_selection_fallback_full_tensor':not bool(logit_kwargs)}


def load_existing(log,protocol_hash):
    rows={}
    if not log.exists():return rows
    text=log.read_text()
    lines=text.split('\n')
    for number,line in enumerate(lines):
        if not line.strip():continue
        try:r=json.loads(line)
        except ValueError:
            if number==len(lines)-1 and not text.endswith('\n'):
                # Preserve interrupted tail before rewriting only its valid prefix.
                log.with_name(log.name+f'.interrupted.{time.time_ns()}').write_text(line)
                log.write_text('\n'.join(lines[:-1])+'\n');break
            raise
        if r.get('protocol_hash')!=protocol_hash:raise RuntimeError('mixed task protocol')
        i=r['sample_id']
        if i in rows and rows[i]!=r:raise RuntimeError('conflicting duplicate task sample')
        rows[i]=r
    return rows


def summary(rows,selected,total,meta,elapsed,encoding):
    chosen=[rows[i] for i in selected if i in rows];n=len(chosen)
    result={'status':'complete' if n==len(selected) else 'partial','model':meta['model'],'method':meta['method'],
        'protocol_hash':meta['protocol_hash'],'comparison_hash':meta['comparison_hash'],
        'completed_samples':n,'requested_samples':len(selected),'test_samples':total,
        'full_test':n==total,'sample_ids':[r['sample_id'] for r in chosen],
        'encoding_s':encoding,'elapsed_s':elapsed,'updated_unix':time.time()}
    for metric in ('strict','flex'):
        k=sum(r[metric+'_correct'] for r in chosen);p=k/n if n else None
        result[metric+'_accuracy']=p;result[metric+'_correct']=k
        result[metric+'_binomial_se']=math.sqrt(p*(1-p)/n) if n else None
    return result


def evaluate(engine,a,method,test,prompts,gold,source):
    root=Path(engine.a.out)/method;root.mkdir(parents=True,exist_ok=True)
    base,_=engine.protocol(method)
    protocol={'task_engine_version':VERSION,'task_engine_sha256':ppl.sha_file(Path(__file__)),
        'ppl_engine_sha256':base['engine_sha256'],
        'matched_extension_sha256':base['matched_extension_sha256'],
        'serializer_sha256':base['serializer_sha256'],
        'storage_layout':base['storage_layout'],'total_budget_rule':base['total_budget_rule'],
        'model':engine.model_id,'revision':base.get('revision'),'config_commit_hash':base.get('config_commit_hash'),
        'dtype':engine.a.dtype,'backend':'HF SDPA + DynamicCache.update fake quant','device_type':engine.device.type,
        'device_name':torch.cuda.get_device_name(engine.device) if engine.device.type=='cuda' else 'CPU',
        'torch_version':torch.__version__,'data':source,'prompts_sha256':ppl.digest(prompts),
        'test_order':ppl.spread_order(len(test)),'nshot':a.task_nshot,'generation':{'greedy':True,
            'max_new_tokens':a.task_max_new_tokens,'stop_strings':STOPS,'stop_check_interval':a.task_stop_check,
            'logit_selection_kwargs':last_logit_kwargs(engine.model),
            'add_special_tokens':True,'padding_side':'left','batch_size':a.task_batch_size},
        'codec_sha256':base['codec_sha256'],'geometry':base['geometry'],'calibration':base['calibration'],
        'map_policy':engine.a.map_policy,'map_calib_windows':engine.a.map_calib_windows,'budget_bpe':base['budget_bpe']}
    common={k:v for k,v in protocol.items() if k not in ('calibration',)}
    comparison_hash=ppl.digest(common)
    protocol.update(method=method,scope=ppl.SCOPE[method][0])
    protocol_hash=ppl.digest(protocol)
    order=protocol['test_order'];selected=order if a.task_max_samples==0 else order[:a.task_max_samples]
    meta={**protocol,'protocol_hash':protocol_hash,'comparison_hash':comparison_hash,'requested_sample_ids':selected,
          'frozen_map_protocol':base['frozen_map_protocol'],
          'evaluation_extent':'full_test' if len(selected)==len(test) else 'prespecified_test_spread_subset',
          'numerical_evaluation_only':'Dequantized fake-quant cache; not a packed-cache performance/memory result.'}
    existing=root/'meta.json'
    if existing.exists():
        old=json.loads(existing.read_text())
        if old.get('protocol_hash')!=protocol_hash:
            raise RuntimeError('refusing task resume with changed method/data/scorer/settings')
        for key in ('storage','encoding_details','map_calibration','total_budget_audit'):
            if key in old:meta[key]=old[key]
    ppl.atomic_json(existing,meta)
    log=root/'samples.jsonl';rows=load_existing(log,protocol_hash)
    missing=[i for i in selected if i not in rows]
    encoding=0
    if missing:
        before=time.time();engine.prepare(method);encoding=time.time()-before
        meta.update(storage=engine.storage,encoding_details=engine.enc_details,
                    map_calibration=getattr(engine,'map_calibration',None),
                    total_budget_audit=getattr(engine,'total_budget_audit',None))
        ppl.atomic_json(existing,meta)
        engine.model.config.use_cache=True
    for start in range(0,len(missing),a.task_batch_size):
        if engine.should_stop():break
        batch=missing[start:start+a.task_batch_size];begin=time.time()
        result,audit=generate_batch(engine,[prompts[i] for i in batch],a,verify=a.task_verify_cache and start==0)
        duration=time.time()-begin
        for i,res in zip(batch,result):
            row={'record':'sample','sample_id':i,'protocol_hash':protocol_hash,'model':engine.model_id,'method':method,
                'prompt_sha256':ppl.digest(prompts[i]),'gold':gold[i],**res,**score(res['text'],gold[i]),
                'batch_elapsed_s':duration,'batch_size':len(batch),'completed_unix':time.time()}
            with log.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
            rows[i]=row
        report=summary(rows,selected,len(test),meta,engine.elapsed(),encoding)
        ppl.atomic_json(root/'summary.json',report);ppl.atomic_json(root/'last_batch_audit.json',audit)
        print(json.dumps({'event':'task_batch','method':method,'completed':report['completed_samples'],
            'requested':len(selected),'batch_seconds':duration,'strict_accuracy':report['strict_accuracy']}),flush=True)
    report=summary(rows,selected,len(test),meta,engine.elapsed(),encoding)
    ppl.atomic_json(root/'summary.json',report)
    return report


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--task-max-samples',type=int,default=64,help='0 = full test.')
    p.add_argument('--task-batch-size',type=int,default=4)
    p.add_argument('--task-max-new-tokens',type=int,default=256)
    p.add_argument('--task-nshot',type=int,default=5)
    p.add_argument('--task-stop-check',type=int,default=4)
    p.add_argument('--task-test',type=Path);p.add_argument('--task-train',type=Path)
    p.add_argument('--task-verify-cache',action='store_true')
    p.add_argument('--frozen-map-dir',default=None)
    taskargs,remaining=p.parse_known_args();sys.argv=[sys.argv[0],*remaining]
    args=ppl.parse()
    if min(taskargs.task_batch_size,taskargs.task_max_new_tokens,taskargs.task_stop_check)<1:raise ValueError('positive task sizes required')
    if taskargs.task_max_samples<0:raise ValueError('task max samples must be >=0')
    signal.signal(signal.SIGTERM,ppl.request_stop);signal.signal(signal.SIGINT,ppl.request_stop)
    started=time.time();test,prompts,gold,source=dataset(taskargs)
    ppl_matched.MAP_DIR=taskargs.frozen_map_dir
    engine=ppl_matched.MatchedEngine(args,started)
    try:
        for method in (args.methods.split(',') if args.methods else [args.method]):
            result=evaluate(engine,taskargs,method,test,prompts,gold,source)
            print(json.dumps({'event':'task_summary',**result}),flush=True)
            if engine.should_stop():break
    finally:engine.close()


if __name__=='__main__':main()
