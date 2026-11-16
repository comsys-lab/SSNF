"""Isolated core-kernel generation with auditable per-step batch/residency."""
from __future__ import annotations
import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from protocol import PrefillBarrier, page_capacity, save, sha, validate_trial

HERE = Path(__file__).resolve().parent
ENERGY = HERE.parent
sys.path.insert(0,str(ENERGY))
from common import BUILD, MODEL, REVISION, CORE, digest, source_hashes
from telemetry import Telemetry


class CapacityAdmissionBlocked(RuntimeError):
    pass


def run_trial(llm, cfg, batch, directory, tag, warmup=False):
    import torch
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    directory.mkdir()
    engine=llm.llm_engine
    scheduler=engine.engine_core.engine_core.scheduler
    base=llm.get_tokenizer().encode('The old lighthouse keeper recorded every ship and storm across forty long years by the sea. ')
    ids=(base*(cfg['input_tokens']//len(base)+2))[:cfg['input_tokens']]
    params=SamplingParams(temperature=0,ignore_eos=True,max_tokens=cfg['output_tokens'],
                          output_kind=RequestOutputKind.CUMULATIVE)
    rows=[dict(request_id=f'{tag}-{i}',output_tokens=0,finished=False,first_t=None,last_t=None) for i in range(batch)]
    by_id={r['request_id']:r for r in rows}
    internal_to_external={}
    barrier=PrefillBarrier(scheduler,[])
    original_schedule=scheduler.schedule
    original_preempt=scheduler._preempt_request
    steps=[]
    active=[None]
    preempt_events=[]
    stall_steps=0

    def preempt(req,stamp):
        preempt_events.append(dict(request_id=req.request_id,num_computed_tokens=req.num_computed_tokens,
                                   num_output_tokens=req.num_output_tokens,t=stamp))
        return original_preempt(req,stamp)

    def schedule():
        nonlocal stall_steps
        released=barrier.before_schedule() if barrier else False
        before={rid:dict(computed=r.num_computed_tokens,prompt=r.num_prompt_tokens,
                         output=r.num_output_tokens,status=str(r.status)) for rid,r in scheduler.requests.items() if rid in internal_to_external}
        before_events=len(preempt_events)
        out=original_schedule()
        internal_scheduled=dict(out.num_scheduled_tokens)
        scheduled={internal_to_external[rid]:n for rid,n in internal_scheduled.items()}
        decode=[internal_to_external[rid] for rid in internal_scheduled if rid in before and before[rid]['computed']>=cfg['input_tokens']]
        chosen=[rid for rid in internal_scheduled if rid in before]
        blocks={internal_to_external[rid]:[len(g) for g in scheduler.kv_cache_manager.get_block_ids(rid)]
                for rid in internal_to_external if rid in scheduler.requests and scheduler.requests[rid] in scheduler.running}
        row=dict(step_index=len(steps),t=time.perf_counter(),scheduled_ids=list(scheduled),
                 scheduled_tokens=scheduled,decode_ids=decode,
                 internal_scheduled_ids=list(internal_scheduled),
                 computed_before={internal_to_external[rid]:before[rid]['computed'] for rid in chosen},
                 output_before={internal_to_external[rid]:before[rid]['output'] for rid in chosen},
                 min_computed_before=min((before[rid]['computed'] for rid in chosen),default=-1),
                 max_computed_before=max((before[rid]['computed'] for rid in chosen),default=-1),
                 running_ids=[internal_to_external.get(r.request_id,r.request_id) for r in scheduler.running],
                 waiting_ids=[internal_to_external.get(r.request_id,r.request_id) for r in scheduler.waiting],
                 block_counts=blocks,free_blocks=scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
                 preempted=preempt_events[before_events:],barrier_released=released,
                 input_tokens=cfg['input_tokens'],wall_s=None)
        steps.append(row);active[0]=row
        stall_steps=stall_steps+1 if not scheduled else 0
        if stall_steps>2:
            if (cfg['mode']=='capacity' and row['waiting_ids'] and
                    row['free_blocks'] < math.ceil(cfg['input_tokens']/scheduler.cache_config.block_size)):
                raise CapacityAdmissionBlocked('Insufficient free KV pages to admit all prefills while completed prefills remain resident')
            raise RuntimeError('No schedulable tokens for three steps; barrier cannot admit all requests')
        return out

    torch.cuda.synchronize()
    start=time.perf_counter()
    scheduler.schedule=schedule
    scheduler._preempt_request=preempt
    issue=None
    try:
        for row in rows:
            row['offered_t']=start
            row['enqueued_t']=time.perf_counter()
            internal=engine.add_request(row['request_id'],TokensPrompt(prompt_token_ids=ids),params.clone())
            internal_to_external[internal]=row['request_id']
            row['internal_request_id']=internal
        barrier.request_ids=set(internal_to_external)
        while engine.has_unfinished_requests():
            if time.perf_counter()-start > cfg['trial_timeout']:
                raise TimeoutError('Trial time budget exceeded')
            t=time.perf_counter()
            try:
                outs=engine.step()
            finally:
                now=time.perf_counter()
                if active[0] is not None: active[0]['wall_s']=now-t
            if active[0] is None:
                raise RuntimeError('No scheduler trace for engine step')
            active[0]['wall_s']=now-t
            for out in outs:
                row=by_id[out.request_id]
                output=list(out.outputs[0].token_ids)
                if len(output)<row['output_tokens']:
                    raise RuntimeError('Output count decreased')
                if output and row['first_t'] is None: row['first_t']=now
                row['output_tokens']=len(output)
                if out.finished:
                    row.update(finished=True,last_t=now,output_sha256=digest(output),output_token_ids=output)
            if len(steps)%32==0:
                save(directory/'progress.json',dict(steps=len(steps),elapsed_s=now-start,
                     completed=sum(r['finished'] for r in rows),output_tokens=sum(r['output_tokens'] for r in rows),
                     preemptions=len(preempt_events)))
        torch.cuda.synchronize()
    except CapacityAdmissionBlocked as exc:
        issue=repr(exc)
        engine.abort_request(list(by_id))
        torch.cuda.synchronize()
    except Exception as exc:
        issue=repr(exc)
        raise
    finally:
        end=time.perf_counter()
        scheduler.schedule=original_schedule;scheduler._preempt_request=original_preempt
        if barrier: barrier.restore()
        for name,records in [('requests',rows),('scheduler',steps)]:
            with (directory/f'{name}.jsonl').open('w') as f:
                for row in records: f.write(json.dumps(row,allow_nan=False)+'\n')
        save(directory/'bounds.json',dict(start=start,end=end,wall_s=end-start,error=issue))
        save(directory/'workload.json',dict(batch=batch,prompt_sha256=digest(ids),input_tokens=len(ids),
                 output_tokens=cfg['output_tokens'],warmup=warmup,mode=cfg['mode'],tag=tag))
    check=validate_trial(rows,steps,batch,cfg['output_tokens'],cfg['mode'])
    check['admission_blocked']=issue is not None and issue.startswith('CapacityAdmissionBlocked')
    if check['admission_blocked']:
        check['valid_capacity_failure_observation']=True
    check.update(batch=batch,input_tokens=cfg['input_tokens'],output_tokens=cfg['output_tokens'],
                 mode=cfg['mode'],warmup=warmup,wall_s=end-start,total_output_tokens=sum(r['output_tokens'] for r in rows),
                 throughput=sum(r['output_tokens'] for r in rows)/(end-start),
                 prefilling_and_first_output_s=sum(s['wall_s'] for s in steps if not s['barrier_released']),
                 trace_steps=len(steps),prompt_sha256=digest(ids),directory=str(directory))
    save(directory/'summary.json',check)
    return check


def main(directory):
    cfg=json.loads((directory/'config.json').read_text())
    if cfg['source_hashes']!=source_hashes(): raise RuntimeError('Sources changed after launch')
    for name,value in cfg['driver_hashes'].items():
        if sha(HERE/name)!=value: raise RuntimeError('Driver changed after launch')
    build=json.loads((BUILD/'manifest.json').read_text())
    if sha(build['binary'])!=build['binary_sha256']: raise RuntimeError('Extension hash mismatch')
    meter=Telemetry(cfg['gpu_uuid']);meter.assert_exclusive()
    sys.path[:0]=[str(BUILD/'lib'),str(CORE)]
    import torch
    import vllm
    from vllm.platforms.interface import Platform
    old=Platform.device_id_to_physical_device_id.__func__
    def physical(cls,device):
        ids=os.environ.get(cls.device_control_env_var,'').split(',')
        if ids and ids[device].startswith('GPU-'):
            return meter.nv.nvmlDeviceGetIndex(meter.nv.nvmlDeviceGetHandleByUUID(ids[device]))
        return old(cls,device)
    Platform.device_id_to_physical_device_id=classmethod(physical)
    import ssnf_cuda
    import ssnf_vllm
    if torch.cuda.device_count()!=1 or Path(ssnf_cuda.__file__).resolve()!=Path(build['binary']).resolve():
        raise RuntimeError('Wrong GPU isolation/extension')
    if not vllm.__version__.startswith('0.23.') or torch.__version__!=build['torch']:
        raise RuntimeError('Runtime mismatch')
    from vllm import LLM
    kw=dict(model=MODEL,revision=REVISION,tokenizer_revision=REVISION,dtype='bfloat16',enforce_eager=True,
            max_model_len=cfg['input_tokens']+cfg['output_tokens']+16,gpu_memory_utilization=.85,
            tensor_parallel_size=1,max_num_seqs=512,enable_chunked_prefill=True,enable_prefix_caching=False,seed=20261025)
    if cfg['format']=='ssnf_wkv': kw.update(quantization='ssnf',kv_cache_dtype='ssnf_u3gs16')
    elif cfg['format']=='mxfp6w_wkv': kw.update(quantization='mxfp6w',kv_cache_dtype='mxfp6_kv')
    save(directory/'status.json',dict(state='model_loading',t=time.time()))
    llm=LLM(**kw)
    scheduler=llm.llm_engine.engine_core.engine_core.scheduler
    cc=llm.llm_engine.vllm_config.cache_config
    caps=page_capacity(cc.num_gpu_blocks,cc.block_size,cfg['input_tokens'],cfg['output_tokens'])
    vroot=Path(importlib.util.find_spec('vllm').submodule_search_locations[0])
    metadata=dict(device=meter.metadata(),torch=torch.__version__,vllm=vllm.__version__,
                  engine_args=kw,scheduler_config=repr(llm.llm_engine.vllm_config.scheduler_config),
                  cache_config=repr(cc),scheduler_type=type(scheduler).__name__,capacity=caps,
                  extension_sha256=build['binary_sha256'],scheduler_sha256=sha(vroot/'v1/core/sched/scheduler.py'),
                  diagnostic_policy='prefill barrier using next_decode_eligible_step; original scheduler budgets/allocator retained',
                  new_gpu_energy_claim=False)
    save(directory/'metadata.json',metadata)
    candidates=[]
    for item in cfg['batches'].split(','):
        if item.startswith('B0'):
            offset=int(item[2:] or '0');value=caps['conservative_batch']+offset
        else: value=int(item)
        if value>0 and value not in candidates: candidates.append(value)
    save(directory/'resolved_workloads.json',dict(batches=candidates,repeats=cfg['repeats'],warmups=cfg['warmups']))
    trials=[];blocks=[]
    for batch in candidates:
        for w in range(cfg['warmups']):
            save(directory/'status.json',dict(state='warmup',batch=batch,repeat=w))
            warm=run_trial(llm,cfg,batch,directory/f'b{batch}_warmup{w}',f'b{batch}-w{w}',True)
            if not warm['complete_generation']: raise RuntimeError('Warmup did not complete')
        if cfg['format']=='ssnf_wkv' and (not ssnf_vllm.quant._H_CACHE or ssnf_vllm.kv_backend._HK['t'] is None or not ssnf_vllm.quant._TRITON_OK):
            raise RuntimeError('SSNF importance/Triton not active')
        for r in range(cfg['repeats']):
            meter.assert_exclusive()
            save(directory/'status.json',dict(state='measuring',batch=batch,repeat=r))
            block=[]
            for cycle in range(cfg['cycles']):
                row=run_trial(llm,cfg,batch,directory/f'b{batch}_r{r}_c{cycle}',f'b{batch}-r{r}-c{cycle}')
                row.update(repeat=r,cycle=cycle)
                trials.append(row);block.append(row)
                save(directory/'trials.json',trials)
                if cfg['mode']=='fixed' and not row['fixed_batch_valid']:
                    raise RuntimeError('Fixed batch validation failed; do not expand')
            seconds=sum(t['all_batch_decode_wall_s'] for t in block)
            tokens=sum(t['all_batch_decode_tokens'] for t in block)
            blocks.append(dict(batch=batch,repeat=r,cycles=cfg['cycles'],decode_wall_s=seconds,
                               decode_tokens=tokens,decode_tok_s=tokens/seconds if seconds else None,
                               minimum_duration_passed=seconds>=cfg['minimum_block_seconds']))
            save(directory/'blocks.json',blocks)
    # Capacity nonsuccess is a valid observation if full generation completed.
    valid=all((t['complete_generation'] and not t['issues']) or t.get('valid_capacity_failure_observation') for t in trials)
    if cfg['mode']=='fixed': valid=valid and all(t['fixed_batch_valid'] for t in trials)
    if cfg['mode']=='fixed': valid=valid and all(b['minimum_duration_passed'] for b in blocks)
    summary=dict(run_valid=valid,stage=cfg['stage'],mode=cfg['mode'],format=cfg['format'],gpu_uuid=cfg['gpu_uuid'],
                 input_tokens=cfg['input_tokens'],output_tokens=cfg['output_tokens'],capacity=caps,trials=trials,blocks=blocks,
                 created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(directory/'summary.json',summary)
    save(directory/'status.json',dict(state='complete' if valid else 'validation_failed'))
    return 0 if valid else 2


if __name__=='__main__':
    directory=Path(sys.argv[1])
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(TimeoutError('Controller stopped worker')))
    try: raise SystemExit(main(directory))
    except Exception as exc:
        traceback.print_exc();save(directory/'status.json',dict(state='failed',error=repr(exc),partial_results=True));raise SystemExit(1)
