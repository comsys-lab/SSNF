"""Fixed AMS/SSNF W-only comparison, with optional AMS-byte-ceiling SSNF mix.

No menu is selected from test outcomes. New method labels preserve geometry,
objective and budget contract independently. Original engine/codecs stay frozen.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import sys
import time
import numpy as np
import torch
from . import ppl,codecs,ssnf_mixed,ams_codec
from .ppl_ams import AMSEngine,METHODS as AMS_METHODS

CoreEngine=ppl.Engine
FIXED={f'ssnf_{lb}_{gs}_{objective}':(lb,gs,objective=='h')
       for lb,gs in ((4,4),(3,32),(3,16),(3,8)) for objective in ('mse','h')}
MATCHED={'ssnf_ams_ceiling_mse':False,'ssnf_ams_ceiling_h':True}
MINIMUM=('ssnf_3_32_mse','ssnf_3_32_h','ssnf_3_16_mse','ssnf_3_16_h')
for name,(_,_,weighted) in FIXED.items():ppl.SCOPE[name]=('w',weighted,False)
for name,weighted in MATCHED.items():ppl.SCOPE[name]=('w',weighted,False)
LOW,HIGH=(4,4),(3,16)
UNIFORM_HEADER=struct.Struct('<8sQBBBBIII')
UNIFORM_MAGIC=b'SSNFUNI1'

def pack_uniform(x,geometry,importance=None):
    raw=ssnf_mixed.encode_geometry(x,geometry,importance,search=True)
    return UNIFORM_HEADER.pack(UNIFORM_MAGIC,len(raw),*geometry,0,0,0,0,1)+raw.tobytes()

def unpack_uniform(blob):
    if len(blob)<32:raise ValueError('short native SSNF header')
    magic,n,lb,gs,r0,r1,r2,r3,flag=UNIFORM_HEADER.unpack(blob[:32])
    if magic!=UNIFORM_MAGIC or (r0,r1,r2,r3,flag)!=(0,0,0,0,1):raise ValueError('invalid native SSNF header')
    if len(blob)!=32+n*ssnf_mixed.block_bytes((lb,gs)):raise ValueError('native SSNF byte mismatch')
    return ssnf_mixed.decode_geometry(np.frombuffer(blob[32:],np.uint8),(lb,gs))

def matched_quota(rows,cols):
    blocks=rows*cols//32
    if rows*cols%32:raise ValueError('complete32-element blocks required')
    ceiling=ams_codec.sizes(rows,cols,6,3)['total_bytes']
    base=ssnf_mixed.sizes(blocks,LOW,HIGH,0)['total_bytes']
    delta=ssnf_mixed.block_bytes(HIGH)-ssnf_mixed.block_bytes(LOW)
    if delta<=0 or ceiling<base:raise ValueError('AMS ceiling below lowest SSNF mixed layout')
    high=min(blocks,(ceiling-base)//delta);cost=ssnf_mixed.sizes(blocks,LOW,HIGH,high)
    return high,ceiling,cost

def ceiling_map(gain,quota):
    # The two geometries are not nested: a more expensive geometry need not
    # improve every block. The contract is <=ceiling, so never buy negative or
    # zero gain just to fill bytes. Remaining slack is reported explicitly.
    count=min(quota,int((gain>0).sum()))
    return codecs.choose_high_map(gain,count,'importance'),count

class FairEngine(AMSEngine):
    def protocol(self,method):
        if method in FIXED or method in MATCHED:details,_=CoreEngine.protocol(self,method)
        else:details,_=super().protocol(method)
        details['AMS_fair_extension']={'version':'fixed-W-only-v1','driver_sha256':ppl.sha_file(__file__),
            'SSNF_serializer_sha256':ppl.sha_file(ssnf_mixed.__file__),'AMS_codec_sha256':ppl.sha_file(ams_codec.__file__),
            'selection':'fixed reported menu; no test-driven geometry/objective choice','scope':'seven linear projections W-only; BF16 KV/embedding/head/norm unchanged',
            'MSE':'no H_W in codec or allocation','weighted':'same frozen legacy H_W in codec and allocation; provenance disclosed',
            'native_uniform_storage':'32-byte versioned tensor header plus actual byte-rounded SSNF fields and scale; no tag/rank',
            'matched_storage':'same32-byte mixed header, packed1bit tags, uint64 rank per256 blocks, mixed geometry payloads',
            'matched_quota':'per-tensor AMS FP6g3 actual-size ceiling including FP32 row scales/tails/header; at-most quota, positive reconstruction gains only; stricter than unrestricted global allocation',
            'matched_low_geometry':list(LOW),'matched_high_geometry':list(HIGH),'no_KV_precision_maps':True}
        details['AMS_fair_extension']['prior_reference_manifest_sha256']=ppl.sha_file(self.a.prior_reference_manifest) if method in references(self.a) else None
        if method in FIXED:
            lb,gs,weighted=FIXED[method]
            details['geometry']={'w':[lb,gs],'k':'unquantized','v':'unquantized'}
            details['AMS_fair_extension']['objective']='H_W weighted reconstruction' if weighted else 'MSE'
            details['AMS_fair_extension']['encoded_block_W_bpe']=codecs.encoded_ssnf(lb,gs)
        elif method in MATCHED:
            details['geometry']={'w':{'low':list(LOW),'high':list(HIGH)},'k':'unquantized','v':'unquantized'}
            details['AMS_fair_extension']['objective']='H_W weighted codec and allocation' if MATCHED[method] else 'unweighted codec and allocation'
        details['AMS_fair_extension']['legacy_budget_fields_are_not_fixed_row_measured_cost']=True
        return details,ppl.digest({k:v for k,v in details.items() if k!='hostname'})

    def _root(self,method):
        root=Path(self.a.out)/method/'artifacts';root.mkdir(parents=True,exist_ok=True);return root

    def _importance(self,name,weighted,start,end):
        if not weighted:return None
        h=self.h_w(name,self.device).reshape(-1,32)
        return h[torch.arange(start,end,device=self.device)%h.shape[0]]

    @torch.no_grad()
    def _fixed(self,method):
        lb,gs,weighted=FIXED[method];saved=(self.a.w_lb,self.a.w_gs)
        try:
            self.a.w_lb,self.a.w_gs=lb,gs;CoreEngine.prepare(self,method)
        finally:self.a.w_lb,self.a.w_gs=saved
        details=[];total=0;elements=0
        for name,module in self.targets:
            source=self.master[name];rows,cols=source.shape;n=source.numel()//32
            probe=source[:min(2,rows)].to(self.device).float()
            h=self.h_w(name,self.device)[None,:].expand_as(probe) if weighted else None
            blob=pack_uniform(probe,(lb,gs),h);decoded=unpack_uniform(blob).reshape_as(probe)
            if not torch.equal(decoded.to(self.dtype),module.weight[:len(probe)].detach().cpu()):raise AssertionError('native SSNF source-row codec mismatch')
            path=self._root(method)/(name+'.native.bin');path.write_bytes(blob)
            payload=n*ssnf_mixed.block_bytes((lb,gs));cost=payload+32;total+=cost;elements+=source.numel()
            details.append({'tensor':name,'rows':rows,'cols':cols,'elements':source.numel(),'geometry':[lb,gs],
                'payload_bytes':payload,'header_bytes':32,'tag_bytes':0,'rank_bytes':0,'total_bytes':cost,
                'source_probe':{'rows':len(probe),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'path':str(path),'roundtrip_matches_forward_weight':True}})
        self.enc_details=details;self.storage['fair_W_encoded_audit']={'elements':elements,'total_bytes':total,'actual_bpe':8*total/elements,
            'codec_objective':'H_W' if weighted else 'MSE','full_checkpoint_written':False,'source_rows_actually_serialized_per_tensor':2,
            'note':'Complete arbitrary-tensor native serializer exists; full model length audited from actual shapes, actual2row probes stored.'}

    @torch.no_grad()
    def _matched(self,method):
        weighted=MATCHED[method];self.active=False;self.method=method;self.scope='w';self.w_weighted=weighted;self.k_weighted=False
        self.storage={};self.enc_details=[];total=ceiling_total=elements=0
        for name,module in self.targets:
            if self.should_stop():raise TimeoutError('AMS-ceiling SSNF preparation budget exhausted')
            source=self.master[name];rows,cols=source.shape;blocks=source.numel()//32
            quota,ceiling,_=matched_quota(rows,cols);gains=[]
            for start in range(0,blocks,self.a.weight_chunk_blocks):
                end=min(blocks,start+self.a.weight_chunk_blocks);x=source.reshape(-1,32)[start:end].to(self.device).float()
                h=self._importance(name,weighted,start,end)
                low=codecs.ssnf(x,*LOW,h,search=True);high=codecs.ssnf(x,*HIGH,h,search=True)
                diff=(low-x).square()-(high-x).square()
                if h is not None:diff*=h
                gains.append(diff.sum(-1).cpu())
            mask,count=ceiling_map(torch.cat(gains),quota);cost=ssnf_mixed.sizes(blocks,LOW,HIGH,count)
            for start in range(0,blocks,self.a.weight_chunk_blocks):
                end=min(blocks,start+self.a.weight_chunk_blocks);x=source.reshape(-1,32)[start:end].to(self.device).float()
                h=self._importance(name,weighted,start,end)
                q=torch.where(mask[start:end].to(self.device)[:,None],codecs.ssnf(x,*HIGH,h,search=True),codecs.ssnf(x,*LOW,h,search=True))
                module.weight.reshape(-1,32)[start:end].copy_(q.to(self.dtype))
            probe=source[:min(2,rows)].to(self.device).float();h=self.h_w(name,self.device)[None,:].expand_as(probe) if weighted else None
            prefix=mask[:probe.numel()//32];blob=ssnf_mixed.pack(probe,prefix,LOW,HIGH,h,search=True);decoded,returned,_=ssnf_mixed.unpack(blob)
            if not torch.equal(returned,prefix) or not torch.equal(decoded.reshape_as(probe).to(self.dtype),module.weight[:len(probe)].detach().cpu()):raise AssertionError('AMS-ceiling mixed source-row mismatch')
            root=self._root(method);path=root/(name+'.mixed.bin');path.write_bytes(blob);tag=np.packbits(mask.numpy(),bitorder='little').tobytes();tagpath=root/(name+'.tags.bin');tagpath.write_bytes(tag)
            self.enc_details.append({'tensor':name,'rows':rows,'cols':cols,'elements':source.numel(),'high_count':count,'max_byte_feasible_high_count':quota,'low_geometry':list(LOW),'high_geometry':list(HIGH),
                'AMS_ceiling_bytes':ceiling,'slack_bytes':ceiling-cost['total_bytes'],'map_sha256':hashlib.sha256(mask.numpy().tobytes()).hexdigest(),**cost,
                'tag_artifact':str(tagpath),'tag_artifact_sha256':hashlib.sha256(tag).hexdigest(),
                'source_probe':{'rows':len(probe),'bytes':len(blob),'path':str(path),'sha256':hashlib.sha256(blob).hexdigest(),'roundtrip_matches_forward_weight':True}})
            total+=cost['total_bytes'];ceiling_total+=ceiling;elements+=source.numel()
        if total>ceiling_total:raise AssertionError('SSNF exceeds actual AMS FP6g3 total ceiling')
        self.storage['fair_W_encoded_audit']={'elements':elements,'total_bytes':total,'actual_bpe':8*total/elements,
            'AMS_ceiling_bytes':ceiling_total,'AMS_ceiling_bpe':8*ceiling_total/elements,'slack_bytes':ceiling_total-total,
            'allocation':'per-tensor at-most byte quota, stable positive train-reconstruction-gain ranking','codec_objective':'H_W' if weighted else 'MSE',
            'full_checkpoint_written':False,'source_rows_actually_serialized_per_tensor':2,'KV_static_map_bytes':0}
        self.active=True

    @torch.no_grad()
    def prepare(self,method):
        if getattr(self,'_fair_prepared',None)==method:
            self.storage['fair_in_process_preparation_reused']=True
            self.storage['fair_original_preparation_artifact']=getattr(self,'_fair_preparation_artifact',None)
            return
        started=time.time()
        if method in FIXED:self._fixed(method)
        elif method in MATCHED:self._matched(method)
        else:super().prepare(method)
        self._fair_prepared=method;self.storage['fair_in_process_preparation_reused']=False
        self._fair_preparation_artifact=str(Path(self.a.out)/method/'fair_preparation.json')
        ppl.atomic_json(Path(self._fair_preparation_artifact),{'status':'complete','method':method,'encoding_seconds':time.time()-started,
            'scope':'W-only','storage':self.storage,'weight_tensor_count':len(self.targets),'driver_sha256':ppl.sha_file(__file__)})

    def evaluate(self,method):
        # Avoid overwriting valid preparation evidence on a checkpoint-only
        # prefix replay. Numerical protocol and every requested target checked.
        out=Path(self.a.out)/method;meta=out/'meta.json';raw=out/'windows.jsonl';_,protocol_hash=self.protocol(method)
        if meta.exists() and raw.exists() and json.loads(meta.read_text()).get('protocol_hash')==protocol_hash:
            rows={r['window_id']:r for r in [json.loads(line) for line in raw.open() if line.strip()]}
            complete=True
            for index in self.selected:
                row=rows.get(index);expected=self.windows[index]
                input_sha=hashlib.sha256(self.ids[expected['start']:expected['end']].numpy().tobytes()).hexdigest()
                if row is None or row.get('protocol_hash')!=protocol_hash or row.get('input_sha256')!=input_sha or any(row[k]!=expected[k] for k in ('start','end','target_start','target_count')) or not math.isfinite(row['nll']):complete=False;break
            summary=out/'summary.json'
            previous=json.loads(summary.read_text()) if summary.exists() else {}
            if complete and previous.get('status')=='complete' and previous.get('completed_windows',0)==len(self.selected):
                print(json.dumps({'event':'fair_existing_prefix_reused','method':method,'windows':len(self.selected)}),flush=True);return True
        saved_encoding=(out/'encoding.json').read_bytes() if (out/'encoding.json').exists() else None
        saved_raw_sha=ppl.sha_file(raw) if raw.exists() else None
        result=super().evaluate(method)
        if saved_encoding is not None and saved_raw_sha is not None and raw.exists() and ppl.sha_file(raw)==saved_raw_sha:
            # Existing samples can be re-summarized without running prepare.
            # Preserve their original encoding evidence in that no-op case.
            (out/'encoding.json').write_bytes(saved_encoding)
        return result

def parse():
    extra=argparse.ArgumentParser(add_help=False);extra.add_argument('--stages',default='2,16');extra.add_argument('--selftest',action='store_true');extra.add_argument('--verify-only',action='store_true');extra.add_argument('--prior-reference-manifest',type=Path)
    x,rest=extra.parse_known_args()
    if x.selftest:return x
    saved=sys.argv
    try:sys.argv=[saved[0],*rest];a=ppl.parse()
    finally:sys.argv=saved
    a.stages=[0 if s in ('0','full') else int(s) for s in x.stages.split(',')]
    if not a.stages or a.stages[0]!=2 or any(s<0 for s in a.stages) or any(b<=a for a,b in zip(a.stages,a.stages[1:]) if b!=0) or (0 in a.stages[:-1]):raise ValueError('stages must start2, increase, and optional full must be last')
    if 0 in a.stages and 16 not in a.stages[:-1]:raise ValueError('full evaluation requires prior16-window pilot')
    a.methods=a.methods or ','.join(MINIMUM);a.max_windows=2;a.selftest=x.selftest;a.verify_only=x.verify_only;a.prior_reference_manifest=x.prior_reference_manifest
    if any(m not in FIXED and m not in MATCHED and m not in AMS_METHODS and m!='bf16' for m in a.methods.split(',')):raise ValueError('unknown fixed fair method')
    return a

def references(a):
    if not getattr(a,'prior_reference_manifest',None):return {}
    data=json.loads(Path(a.prior_reference_manifest).read_text());return data.get('methods',data)

def read_rows(path):
    data=[json.loads(line) for line in Path(path).open() if line.strip()];indexed={x['window_id']:x for x in data}
    if len(indexed)!=len(data):raise ValueError('duplicate prior/reference window IDs')
    return indexed

def replay_gate(reference,current,selected):
    old=json.loads((reference/'meta.json').read_text());new=json.loads((current/'meta.json').read_text())
    for key in ('comparison_hash','engine_sha256','codec_sha256','model','revision','dtype','backend','torch_version'):
        if old[key]!=new[key]:raise AssertionError('prior numeric method/cohort mismatch '+key)
    if old['scope']!=new['scope'] or new['scope']!='w':raise AssertionError('prior must be W-only')
    for key in ('hw','hk'):
        if old['calibration'][key]['sha256']!=new['calibration'][key]['sha256']:raise AssertionError('prior importance SHA mismatch')
    checks={'immutable_model_snapshot_identity':old['model'],'same_original_core_and_codec_SHA':True,'same_importance_SHA':True,
        'old_quantized_full_weight_hash_available':False,'old_raw_not_copied_or_rehashed':True}
    if 'ams_paper_core' in old:
        if old['ams_paper_core']!=new.get('ams_paper_core'):raise AssertionError('AMS paper-core numerical definition/code differs')
        oldenc=json.loads((reference/'encoding.json').read_text());newenc=json.loads((current/'encoding.json').read_text())
        aa={x['tensor']:x for x in oldenc['weight_tensors']};bb={x['tensor']:x for x in newenc['weight_tensors']}
        if aa.keys()!=bb.keys():raise AssertionError('AMS weight inventory differs')
        for name in aa:
            for key in ('rows','input_columns','elements','total_bytes'):
                if aa[name][key]!=bb[name][key]:raise AssertionError('AMS tensor geometry/bytes differs')
            if aa[name]['pack_reference']['payload_sha256']!=bb[name]['pack_reference']['payload_sha256']:raise AssertionError('AMS actual source-row reconstructed-weight payload differs')
        checks.update(actual_source_row_weight_payloads_equal=True,source_row_probed_tensors=len(aa))
    else:
        if old['geometry']['w']!=new['geometry']['w']:raise AssertionError('fixed SSNF W geometry differs')
        if new['method'] not in FIXED or old['method']!=('ssnf_w_weighted' if FIXED[new['method']][2] else 'ssnf_w_mse'):raise AssertionError('fixed SSNF objective differs from legacy reference')
        checks.update(fixed_SSNF_same_core_prepare_and_geometry=True,
            weight_evidence_limit='Old native SSNF row did not store full quantized-weight hash. Same immutable model/core/codec/H/geometry plus actual new payload probes and fresh NLL replay are used; do not claim an unavailable historical full-weight hash.')
    aa=read_rows(reference/'windows.jsonl');bb=read_rows(current/'windows.jsonl');errors=[]
    for index in selected:
        if index not in aa or index not in bb:raise AssertionError('prior replay missing selected ID')
        for key in ('input_sha256','start','end','target_start','target_count'):
            if aa[index][key]!=bb[index][key]:raise AssertionError('prior replay input/target mismatch')
        errors.append(abs(aa[index]['nll']-bb[index]['nll'])/bb[index]['target_count'])
    if max(errors,default=0)>1e-5:raise AssertionError('prior replay mean NLL exceeds1e-5')
    return {'status':'passed','reference':str(reference),'reference_meta_sha256':ppl.sha_file(reference/'meta.json'),
        'reference_raw_sha256':ppl.sha_file(reference/'windows.jsonl'),'old_protocol_hash':old['protocol_hash'],
        'new_replay_protocol_hash':new['protocol_hash'],'window_ids':selected,'max_abs_mean_NLL_difference':max(errors,default=0),'tolerance':1e-5,'weight_and_math_evidence':checks}

def continue_reference(engine,a,base,method,reference):
    reference=Path(reference);oldmeta=json.loads((reference/'meta.json').read_text());old=read_rows(reference/'windows.jsonl')
    manifest=json.loads(Path(a.prior_reference_manifest).read_text())
    for filename,expected in manifest.get('source_evidence',{}).get(method,{}).items():
        if ppl.sha_file(reference/filename)!=expected:raise ValueError('immutable prior artifact SHA changed: '+filename)
    if base.resolve()==reference.parent.resolve() or base.resolve().is_relative_to(reference.parent.resolve()):raise ValueError('new output must not modify old raw result tree')
    old_ids=set(old)
    if not old_ids or any(i not in engine.order for i in old_ids):raise ValueError('prior IDs outside canonical corpus')
    if old_ids!=set(engine.order[:len(old_ids)]):raise ValueError('prior must be the fixed nested corpus-spread prefix')
    replay_ids=[i for i in engine.order if i in old_ids]
    if len(replay_ids)<16:raise ValueError('prior continuation requires at least16 replayable windows')
    a.out=str(base/'replay')
    for stage in a.stages:
        if stage==0:break
        if stage>len(replay_ids):raise ValueError('replay stage larger than prior extent')
        engine.selected=replay_ids[:stage]
        if engine.should_stop() or not engine.evaluate(method):raise TimeoutError('prior replay stage incomplete')
        proof=replay_gate(reference,Path(a.out)/method,engine.selected)
        ppl.atomic_json(Path(a.out)/method/f'replay_{stage}_gate.json',proof)
    if a.stages[-1]!=0:return
    if not (Path(a.out)/method/'replay_16_gate.json').exists():raise ValueError('unseen continuation requires16-window replay gate')
    unseen=[i for i in engine.order if i not in old_ids]
    if not unseen:raise ValueError('prior already covers full corpus; reuse it instead')
    a.out=str(base/'unseen');engine.selected=unseen
    if engine.should_stop() or not engine.evaluate(method):raise TimeoutError('new unseen complement incomplete')
    newpath=Path(a.out)/method;newmeta=json.loads((newpath/'meta.json').read_text());new=read_rows(newpath/'windows.jsonl')
    if set(new)!=set(unseen) or old_ids&set(new):raise AssertionError('continuation duplicated/missed IDs')
    combined={**old,**new}
    if set(combined)!=set(engine.order):raise AssertionError('union not full canonical corpus')
    count=sum(x['target_count'] for x in combined.values());nll=sum(x['nll'] for x in combined.values())
    if count!=engine.ids.numel()-1 or not math.isfinite(nll):raise AssertionError('full union token denominator invalid')
    ppl.atomic_json(base/'unions'/f'{method}.json',{'status':'complete_registered_union','method':method,'reference':str(reference),
        'old_protocol_hash':oldmeta['protocol_hash'],'new_protocol_hash':newmeta['protocol_hash'],'old_windows':len(old),'new_unseen_windows':len(new),
        'old_ids':sorted(old),'new_ids':sorted(new),'corpus_windows':len(engine.order),'target_count':count,'total_nll':nll,'ppl':math.exp(nll/count),
        'reference_raw_sha256':ppl.sha_file(reference/'windows.jsonl'),'new_raw_sha256':ppl.sha_file(newpath/'windows.jsonl'),
        'replay_gate':str(base/'replay'/method/'replay_16_gate.json'),
        'interpretation':'Full ID union of immutable old and independently registered new protocols after numeric-method replay; not a single-protocol rerun or rewritten old raw.'})

def verify(a):
    checked=[];refs=references(a)
    for method in a.methods.split(','):
        if method in refs:
            replay=Path(a.out)/'replay'/method;stage=max(s for s in a.stages if s>0)
            proof=json.loads((replay/f'replay_{stage}_gate.json').read_text());assert proof['status']=='passed' and proof['max_abs_mean_NLL_difference']<=1e-5
            if a.stages[-1]==0:
                union=json.loads((Path(a.out)/'unions'/f'{method}.json').read_text());assert union['status']=='complete_registered_union'
                assert not set(union['old_ids'])&set(union['new_ids']) and len(union['old_ids'])+len(union['new_ids'])==union['corpus_windows']
                assert ppl.sha_file(Path(refs[method])/'windows.jsonl')==union['reference_raw_sha256']
                assert ppl.sha_file(Path(a.out)/'unseen'/method/'windows.jsonl')==union['new_raw_sha256']
                checked.append({'method':method,'extent':'registered_old_new_full_union','windows':union['corpus_windows'],'targets':union['target_count']})
            else:checked.append({'method':method,'extent':'fresh_prior_replay_only','windows':stage})
            continue
        path=Path(a.out)/method;meta=json.loads((path/'meta.json').read_text());summary=json.loads((path/'summary.json').read_text());rows=[json.loads(line) for line in (path/'windows.jsonl').open() if line.strip()];indexed={x['window_id']:x for x in rows}
        expected=meta['corpus_windows'] if a.stages[-1]==0 else min(a.stages[-1],meta['corpus_windows'])
        wanted=meta['full_window_order'][:expected]
        assert len(indexed)==len(rows) and set(wanted)<=indexed.keys() and summary['status']=='complete'
        assert summary['completed_windows']==summary['requested_windows']==expected
        count=sum(indexed[i]['target_count'] for i in wanted);nll=sum(indexed[i]['nll'] for i in wanted)
        assert count==summary['target_count'] and math.isclose(nll,summary['total_nll'],rel_tol=1e-11) and math.isfinite(nll)
        assert math.isclose(math.exp(nll/count),summary['ppl'],rel_tol=1e-11)
        if a.stages[-1]==0:assert count==meta['data']['token_count']-1
        assert meta['scope'] in ('w','') and meta['AMS_fair_extension']['no_KV_precision_maps']
        if method in MATCHED:
            enc=json.loads((path/'encoding.json').read_text());audit=enc['storage']['fair_W_encoded_audit'];assert audit['total_bytes']<=audit['AMS_ceiling_bytes']
        checked.append({'method':method,'windows':expected,'targets':count})
    print(json.dumps({'status':'verified','rows':checked,'W_only':True}))

def selftest():
    import tempfile
    from types import SimpleNamespace
    torch.set_num_threads(1);torch.manual_seed(318)
    x=torch.randn(3,4096).to(torch.bfloat16).float();h=torch.logspace(-2,2,4096)[None].expand_as(x)
    for geometry in ((4,4),(3,32),(3,16),(3,8)):
        for importance in (None,h):
            blob=pack_uniform(x,geometry,importance);q=unpack_uniform(blob).reshape_as(x)
            assert torch.equal(q,codecs.ssnf(x,*geometry,importance,search=True))
            assert len(blob)==32+x.numel()//32*ssnf_mixed.block_bytes(geometry)
    count,ceiling,cost=matched_quota(*x.shape);assert 0<count<x.numel()//32 and cost['total_bytes']<=ceiling
    for importance in (None,h):
        ql=codecs.ssnf(x,*LOW,importance,search=True);qh=codecs.ssnf(x,*HIGH,importance,search=True)
        gain=((ql-x).square()-(qh-x).square())*(importance if importance is not None else 1)
        mask,used=ceiling_map(gain.reshape(-1,32).sum(-1),count);blob=ssnf_mixed.pack(x,mask,LOW,HIGH,importance,True);q,m,_=ssnf_mixed.unpack(blob)
        assert torch.equal(m,mask) and torch.equal(q,torch.where(mask[:,None],qh.reshape(-1,32),ql.reshape(-1,32))) and len(blob)==ssnf_mixed.sizes(x.numel()//32,LOW,HIGH,used)['total_bytes']<=ceiling
    zero,count_zero=ceiling_map(torch.tensor([0.,-1.,0.]),2);assert count_zero==0 and not zero.any()
    # Exercise actual preparation dispatch and frozen internal-H semantics on
    # a small weight holder; no model forward or pretrained model is created.
    with tempfile.TemporaryDirectory(prefix='ams_fair_prepare_') as directory:
        module=torch.nn.Linear(4096,3,bias=False).to(torch.bfloat16);module.weight.data.copy_(x.to(torch.bfloat16))
        engine=FairEngine.__new__(FairEngine);engine.device=torch.device('cpu');engine.dtype=torch.bfloat16
        name='model.layers.0.self_attn.q_proj';engine.targets=[(name,module)];engine.master={name:x.to(torch.bfloat16)}
        engine.hw={'0.qkv':h[0]};engine.model=module;engine.should_stop=lambda:False
        engine.a=SimpleNamespace(out=directory,w_lb=3,w_gs=8,weight_chunk_blocks=64,dtype='bfloat16')
        for method in [*FIXED,*MATCHED,'parent_fp6']:
            (Path(directory)/method).mkdir(exist_ok=True);engine.prepare(method)
            assert torch.isfinite(module.weight).all()
            if method in FIXED:
                lb,gs,weighted=FIXED[method];expected=codecs.ssnf(x,lb,gs,h if weighted else None,search=True).to(torch.bfloat16)
                assert torch.equal(module.weight,expected)
            before=module.weight.detach().clone();engine.prepare(method)
            assert torch.equal(module.weight,before) and engine.storage['fair_in_process_preparation_reused']
    print(json.dumps({'status':'passed','CPU_only':True,'GPU_initialized':torch.cuda.is_initialized(),'fixed4_geometry_x2objective_roundtrip':True,
        'AMS_byte_ceiling_mixed_MSE_H_roundtrip':True,'same_byte_cost_for_3_32_and3_16':ssnf_mixed.block_bytes((3,32))==ssnf_mixed.block_bytes((3,16)),
        'actual_fixed_and_mixed_prepare_plus_parent_dispatch':True,'preparation_reuse':True,
        'synthetic_matched_high_count':count,'synthetic_SSNF_bytes':cost['total_bytes'],'synthetic_AMS_ceiling':ceiling,'no_model_forward':True}))

def main():
    a=parse()
    if a.selftest:selftest();return
    if a.verify_only:verify(a);return
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    if a.threads:torch.set_num_threads(a.threads)
    if a.offline:os.environ['HF_HUB_OFFLINE']=os.environ['HF_DATASETS_OFFLINE']='1'
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,ppl.request_stop)
    engine=None;start=time.time()
    try:
        engine=FairEngine(a,start)
        for method in a.methods.split(','):
            a.out=str(Path(a.out))
            if method in references(a):
                (root/'unions').mkdir(parents=True,exist_ok=True);continue_reference(engine,a,root,method,references(a)[method]);a.out=str(root);continue
            a.out=str(root)
            for stage in a.stages:
                engine.selected=engine.order[:] if stage==0 else engine.order[:stage]
                if engine.should_stop() or not engine.evaluate(method):raise TimeoutError('fair W-only stage incomplete')
                ppl.atomic_json(Path(a.out)/method/f'stage_{stage}_gate.json',{'status':'passed','method':method,'window_ids':engine.selected,'requested_windows':len(engine.selected),'full_corpus':stage==0})
    except Exception as e:ppl.atomic_json(Path(a.out)/'error.json',{'error_type':type(e).__name__,'error':str(e),'elapsed_seconds':time.time()-start});raise
    finally:
        if engine:engine.close()
    a.out=str(root)
    verify(a)

if __name__=='__main__':main()
