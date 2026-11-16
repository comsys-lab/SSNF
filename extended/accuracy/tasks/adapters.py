"""LongBench and official IFEval adapters for the validated cached HF generator.

Shared engine flags plus --adapter-task longbench|ifeval. No model runs occur at
import time. Dependencies and the pinned official IFEval source are task-local.
"""
from __future__ import annotations
import __future__
import argparse
import ast
from collections import Counter
from difflib import SequenceMatcher
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import signal
import statistics
import string
import sys
import time
from types import SimpleNamespace
from typing import Any
import zipfile

VENDOR=Path(__file__).parent/'vendor'
sys.path.insert(0,str(VENDOR/'python_deps'))
sys.path.insert(0,str(VENDOR))
import torch
from engine import ppl,ppl_matched
from tasks import generate

VERSION='longbench-ifeval-adapter-v3'


def load_longbench_helpers(path):
    """Reuse core templates/scorers without importing vLLM/model code."""
    source=Path(path).read_text();tree=ast.parse(source)
    constants={'TASKS','NO_CHAT_TASKS','DATASET2PROMPT','DATASET2MAXLEN','DATASET2METRIC','FIRST_LINE'}
    functions={'normalize_answer','_f1','qa_f1_score','classification_score','code_sim_score','retrieval_score','truncate_middle'}
    nodes=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in node.targets):nodes.append(node)
        elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in functions:nodes.append(node)
        elif isinstance(node,ast.Try) and 'from rouge import Rouge' in ast.get_source_segment(source,node):nodes.append(node)
    env=dict(Counter=Counter,SequenceMatcher=SequenceMatcher,re=re,string=string,math=math,statistics=statistics,Any=Any)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),env)
    if env.get('_ROUGE_IMPL')!='rouge-pkg':raise RuntimeError('LongBench requires core rouge-pkg convention, not fallback')
    if not constants.issubset(env):raise RuntimeError('core LongBench helper extraction incomplete')
    return SimpleNamespace(**env)


def ifeval_setup():
    import nltk
    nltk.data.path.insert(0,str(VENDOR/'nltk_data'))
    from langdetect import DetectorFactory
    DetectorFactory.seed=0
    from instruction_following_eval import evaluation_lib
    return evaluation_lib


def ifeval_score(lib,example,response):
    inp=lib.InputExample(key=example['key'],instruction_id_list=example['instruction_id_list'],
        prompt=example['prompt'],kwargs=example['kwargs'])
    seed=20261025+int(example['key'])
    random.seed(seed);strict=lib.test_instruction_following_strict(inp,{inp.prompt:response})
    random.seed(seed);loose=lib.test_instruction_following_loose(inp,{inp.prompt:response})
    return {'strict_prompt_correct':strict.follow_all_instructions,'loose_prompt_correct':loose.follow_all_instructions,
        'strict_instruction_correct':strict.follow_instruction_list,'loose_instruction_correct':loose.follow_instruction_list,
        'instruction_ids':example['instruction_id_list']}


def longbench_prompt(tok,helper,task,example,cap,chat):
    raw=helper.DATASET2PROMPT[task].format(input=example['input'],context=example['context'])
    before_ids=tok.encode(raw,add_special_tokens=False)
    wrapped=chat and task not in helper.NO_CHAT_TASKS
    add_special=not(wrapped and bool(tok.chat_template))
    def render(text):
        if wrapped:
            return tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True) if tok.chat_template else f'[INST]{text}[/INST]'
        return text
    untrimmed=render(raw);untrimmed_ids=tok.encode(untrimmed,add_special_tokens=add_special)
    fixed_overhead=len(tok.encode(render(''),add_special_tokens=add_special))
    kept=len(before_ids);prompt=untrimmed;ids=untrimmed_ids
    if len(ids)>cap:
        # Reserve the wrapper/BOS overhead before trimming the original prompt.
        # Retokenization at the concatenation boundary can add tokens, so verify
        # and shrink deterministically without consulting any model output.
        kept=min(len(before_ids)-1,cap-fixed_overhead-2)
        while kept>1:
            left=(kept+1)//2;right=kept//2
            text=tok.decode(before_ids[:left],skip_special_tokens=True)+tok.decode(before_ids[-right:],skip_special_tokens=True)
            prompt=render(text);ids=tok.encode(prompt,add_special_tokens=add_special)
            if len(ids)<=cap:break
            kept-=max(1,len(ids)-cap)
        if len(ids)>cap:raise ValueError('input cap is too short for chat wrapper and a nonempty middle-truncated prompt')
    audit={'untruncated_prompt_tokens':len(before_ids),'requested_middle_truncation_cap':cap,
        'untruncated_rendered_prompt_tokens':len(untrimmed_ids),'final_input_token_cap':cap,
        'fixed_wrapper_special_token_overhead':fixed_overhead,'kept_original_prompt_tokens':kept,
        'truncated_before_chat':kept<len(before_ids),'chat_wrapped':wrapped,'add_special_tokens':add_special,
        'rendered_prompt_tokens':len(ids),'rendered_prompt_sha256':ppl.digest(prompt),'input_token_ids_sha256':ppl.digest(ids),
        'untruncated_prompt_sha256':ppl.digest(raw)}
    return prompt,add_special,audit


def longbench_score(helper,task,example,text):
    pred=text.lstrip('\n').split('\n')[0] if task in helper.FIRST_LINE else text
    scores=[helper.DATASET2METRIC[task](pred,gt,all_classes=example.get('all_classes')) for gt in example['answers']]
    return {'score':max(scores,default=0.0),'scored_prediction':pred,'answers':example['answers'],'all_classes':example.get('all_classes')}


def load_data(a):
    if a.adapter_task=='ifeval':
        lib=ifeval_setup();path=a.adapter_ifeval_data or VENDOR/'instruction_following_eval/data/input_data.jsonl'
        rows=generate.read_jsonl(path)
        return {'ifeval':rows},{'ifeval_data_sha256':ppl.sha_file(path),'ifeval_sources':json.loads((VENDOR/'ifeval_sources.json').read_text())},lib
    helper=load_longbench_helpers(a.adapter_longbench_source)
    if a.adapter_data_zip:
        path=Path(a.adapter_data_zip)
    else:
        from huggingface_hub import hf_hub_download
        path=Path(hf_hub_download('THUDM/LongBench','data.zip',repo_type='dataset',local_files_only=a.adapter_offline_data))
    tasks=a.adapter_longbench_tasks.split(',') if a.adapter_longbench_tasks else helper.TASKS
    rows={}
    with zipfile.ZipFile(path) as z:
        for task in tasks:
            if task not in helper.TASKS:raise ValueError('unsupported LongBench task: '+task)
            rows[task]=[json.loads(line) for line in z.read(f'data/{task}.jsonl').decode().split('\n') if line.strip()]
    source={'longbench_zip_path':str(path),'longbench_zip_sha256':ppl.sha_file(path),
        'core_prompt_scorer_path':str(a.adapter_longbench_source),
        'core_prompt_scorer_sha256':ppl.sha_file(a.adapter_longbench_source),'rouge_implementation':'rouge-pkg',
        'code_similarity_implementation':'core SequenceMatcher convention'}
    return rows,source,helper


def make_protocol(engine,a,method,task,rows,source,helper):
    base,_=engine.protocol(method)
    chat=(a.adapter_chat=='on' or (a.adapter_chat=='auto' and any(x in engine.model_id.lower() for x in ('instruct','chat'))))
    if a.adapter_task=='ifeval':
        if not engine.tok.chat_template:raise RuntimeError('IFEval requires an instruction model with a recorded chat template')
        chat=True
    gen=a.adapter_ifeval_max_new if a.adapter_task=='ifeval' else helper.DATASET2MAXLEN[task]
    common={'adapter_version':VERSION,'adapter_sha256':ppl.sha_file(Path(__file__)),
        'generator_sha256':ppl.sha_file(Path(generate.__file__)),'ppl_engine_sha256':base['engine_sha256'],
        'matched_extension_sha256':base['matched_extension_sha256'],'serializer_sha256':base['serializer_sha256'],
        'codec_sha256':base['codec_sha256'],'model':engine.model_id,'revision':base.get('revision'),
        'config_commit_hash':base.get('config_commit_hash'),'dtype':engine.a.dtype,'backend':'HF SDPA + quantized DynamicCache.update',
        'device_type':engine.device.type,'device_name':torch.cuda.get_device_name(engine.device) if engine.device.type=='cuda' else 'CPU',
        'torch_version':torch.__version__,'task':task,'benchmark':a.adapter_task,'data':source,
        'raw_task_data_sha256':ppl.digest(rows),'test_order':ppl.spread_order(len(rows)),
        'generation':{'greedy':True,'max_new_tokens':gen,'stop_strings':[],'batch_size':a.adapter_batch_size,
            'logit_selection_kwargs':generate.last_logit_kwargs(engine.model),
            'chat':chat,'chat_template_sha256':ppl.digest(engine.tok.chat_template),
            'final_input_token_cap':a.adapter_context_limit if a.adapter_task=='longbench' else None,
            'truncation':'reserve wrapper/BOS overhead; middle-trim original raw tokens; render+retokenize and shrink until final input <= cap' if a.adapter_task=='longbench' else 'none'},
        'geometry':base['geometry'],'budget_bpe':base['budget_bpe'],'map_policy':engine.a.map_policy,
        'map_calib_windows':engine.a.map_calib_windows,'total_budget_rule':base['total_budget_rule'],
        'storage_layout':base['storage_layout'],'scorer_rng':'Python 20261025+official key per example; langdetect seed0',
        'dependency_versions':{name:(__import__('nltk').__version__ if name=='nltk' else importlib.metadata.version(name)) for name in ('rouge','nltk','langdetect','immutabledict','absl-py')}}
    comparison=ppl.digest(common)
    protocol={**common,'method':method,'scope':ppl.SCOPE[method][0],'calibration':base['calibration'],
        'frozen_map_protocol':base['frozen_map_protocol']}
    return {**protocol,'comparison_hash':comparison,'protocol_hash':ppl.digest(protocol)},chat,gen


def summarize(rows,selected,total,meta,elapsed):
    chosen=[rows[i] for i in selected if i in rows];n=len(chosen)
    result={'status':'complete' if n==len(selected) else 'partial','task':meta['task'],'benchmark':meta['benchmark'],
        'model':meta['model'],'method':meta['method'],'protocol_hash':meta['protocol_hash'],'comparison_hash':meta['comparison_hash'],
        'completed_samples':n,'requested_samples':len(selected),'test_samples':total,'full_test':n==total,
        'elapsed_s':elapsed,'sample_ids':[r['sample_id'] for r in chosen],'updated_unix':time.time()}
    if meta['benchmark']=='longbench':result['score']=100*sum(r['score'] for r in chosen)/n if n else None
    else:
        for metric in ('strict','loose'):
            result[metric+'_prompt_accuracy']=sum(r[metric+'_prompt_correct'] for r in chosen)/n if n else None
            flags=[ok for r in chosen for ok in r[metric+'_instruction_correct']]
            result[metric+'_instruction_accuracy']=sum(flags)/len(flags) if flags else None
            result['instruction_count']=len(flags)
    return result


def evaluate(engine,a,method,all_data,source,helper):
    plans=[]
    for task,data in all_data.items():
        root=Path(engine.a.out)/method/task;root.mkdir(parents=True,exist_ok=True)
        meta,chat,gen=make_protocol(engine,a,method,task,data,source,helper)
        order=meta['test_order'];selected=order if a.adapter_limit==0 else order[:a.adapter_limit]
        meta.update(requested_sample_ids=selected,evaluation_extent='full_test' if len(selected)==len(data) else 'prespecified_spread_subset')
        old=json.loads((root/'meta.json').read_text()) if (root/'meta.json').exists() else {}
        if old and old.get('protocol_hash')!=meta['protocol_hash']:raise RuntimeError('refusing changed adapter protocol resume')
        for key in ('storage','encoding_details','map_calibration','total_budget_audit'):
            if key in old:meta[key]=old[key]
        ppl.atomic_json(root/'meta.json',meta)
        rows=generate.load_existing(root/'samples.jsonl',meta['protocol_hash'])
        plans.append((task,data,root,meta,chat,gen,selected,rows))
    prepared=any(any(i not in rows for i in selected) for _,_,_,_,_,_,selected,rows in plans)
    if prepared:
        engine.prepare(method);engine.model.config.use_cache=True
    summaries=[]
    for task,data,root,meta,chat,gen,selected,rows in plans:
        if prepared:
            meta.update(storage=engine.storage,encoding_details=engine.enc_details,
                map_calibration=getattr(engine,'map_calibration',None),total_budget_audit=getattr(engine,'total_budget_audit',None))
        ppl.atomic_json(root/'meta.json',meta)
        missing=[i for i in selected if i not in rows]
        for start in range(0,len(missing),a.adapter_batch_size):
            if engine.should_stop():break
            batch=missing[start:start+a.adapter_batch_size];prompts=[];audits=[];special=[]
            for i in batch:
                ex=data[i]
                if a.adapter_task=='longbench':prompt,add,audit=longbench_prompt(engine.tok,helper,task,ex,a.adapter_context_limit,chat)
                else:
                    prompt=engine.tok.apply_chat_template([{'role':'user','content':ex['prompt']}],tokenize=False,add_generation_prompt=True)
                    add=False;ids=engine.tok.encode(prompt,add_special_tokens=False)
                    audit={'rendered_prompt_sha256':ppl.digest(prompt),'input_token_ids_sha256':ppl.digest(ids),
                        'rendered_prompt_tokens':len(ids),'chat_wrapped':True,'add_special_tokens':False,'truncated_before_chat':False}
                prompts.append(prompt);audits.append(audit);special.append(add)
            if len(set(special))!=1:raise RuntimeError('a task batch mixed tokenization conventions')
            options=SimpleNamespace(task_max_new_tokens=gen,task_stop_check=1,task_stops=(),task_add_special_tokens=special[0])
            before=time.time();outs,cacheaudit=generate.generate_batch(engine,prompts,options,verify=a.adapter_verify_cache and start==0)
            duration=time.time()-before
            for i,prompt,audit,out in zip(batch,prompts,audits,outs):
                ex=data[i]
                scores=longbench_score(helper,task,ex,out['text']) if a.adapter_task=='longbench' else ifeval_score(helper,ex,out['text'])
                row={'record':'sample','sample_id':i,'original_key':ex.get('key',ex.get('_id',i)),
                    'task':task,'benchmark':a.adapter_task,'model':engine.model_id,'method':method,
                    'protocol_hash':meta['protocol_hash'],'document_sha256':ppl.digest(ex),'prompt_sha256':ppl.digest(prompt),
                    'raw_gold':ex['answers'] if a.adapter_task=='longbench' else {'instruction_id_list':ex['instruction_id_list'],'kwargs':ex['kwargs']},
                    **audit,**out,**scores,'batch_elapsed_s':duration,'batch_size':len(batch),'completed_unix':time.time()}
                log=root/'samples.jsonl'
                with log.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();__import__('os').fsync(f.fileno())
                rows[i]=row
            current=summarize(rows,selected,len(data),meta,engine.elapsed())
            ppl.atomic_json(root/'summary.json',current);ppl.atomic_json(root/'last_batch_audit.json',cacheaudit)
            print(json.dumps({'event':'adapter_batch','method':method,'task':task,'completed':len(rows),
                'requested':len(selected),'batch_seconds':duration}),flush=True)
        current=summarize(rows,selected,len(data),meta,engine.elapsed());ppl.atomic_json(root/'summary.json',current);summaries.append(current)
        if engine.should_stop():break
    suite={'method':method,'model':engine.model_id,'benchmark':a.adapter_task,'tasks':summaries,'updated_unix':time.time()}
    suite['complete_requested_suite']=len(summaries)==len(plans) and all(s['status']=='complete' for s in summaries)
    if a.adapter_task=='longbench':suite['macro_score']=statistics.mean(s['score'] for s in summaries) if suite['complete_requested_suite'] else None
    ppl.atomic_json(Path(engine.a.out)/method/'suite_summary.json',suite)
    return suite


def main():
    extra=argparse.ArgumentParser(add_help=False)
    extra.add_argument('--adapter-task',required=True,choices=['longbench','ifeval'])
    extra.add_argument('--adapter-limit',type=int,default=16,help='Per-task fixed nested spread sample count; 0 = all.')
    extra.add_argument('--adapter-batch-size',type=int,default=1)
    extra.add_argument('--adapter-context-limit',type=int,default=4096)
    extra.add_argument('--adapter-longbench-tasks',default=None)
    extra.add_argument('--adapter-longbench-source',type=Path,default=Path(__file__).resolve().parents[3]/'core/eval_serving/eval_longbench.py')
    extra.add_argument('--adapter-data-zip',default=None)
    extra.add_argument('--adapter-offline-data',action='store_true')
    extra.add_argument('--adapter-chat',choices=['auto','on','off'],default='auto')
    extra.add_argument('--adapter-ifeval-data',type=Path,default=None)
    extra.add_argument('--adapter-ifeval-max-new',type=int,default=1280)
    extra.add_argument('--adapter-verify-cache',action='store_true')
    extra.add_argument('--frozen-map-dir',default=None)
    a,rest=extra.parse_known_args();sys.argv=[sys.argv[0],*rest];args=ppl.parse()
    if a.adapter_limit<0 or min(a.adapter_batch_size,a.adapter_context_limit,a.adapter_ifeval_max_new)<1:raise ValueError('invalid adapter workload')
    signal.signal(signal.SIGTERM,ppl.request_stop);signal.signal(signal.SIGINT,ppl.request_stop)
    started=time.time();data,source,helper=load_data(a)
    ppl_matched.MAP_DIR=a.frozen_map_dir;engine=ppl_matched.MatchedEngine(args,started)
    try:
        for method in args.methods.split(',') if args.methods else [args.method]:
            result=evaluate(engine,a,method,data,source,helper)
            print(json.dumps({'event':'adapter_summary',**result}),flush=True)
            if engine.should_stop():break
    finally:engine.close()


if __name__=='__main__':main()
