"""Six W-only AMS paper-core rows on the frozen HF PPL evaluator.

This is our explicitly documented reconstruction, not official AMS software or
an exact reproduction of its published OpenCompass scores/CUDA measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys
import time

import torch

from . import ams_codec,ams_math_reference,ppl

METHODS={'parent_fp6':(6,1,False),'parent_fp5':(5,1,False),
    'ams6_g3_mse':(6,3,False),'ams6_g3_h':(6,3,True),
    'ams5_g4_mse':(5,4,False),'ams5_g4_h':(5,4,True)}
for name,(_,_,weighted) in METHODS.items():ppl.SCOPE[name]=('w',weighted,False)
PACK_CHECK_ROWS=2
BIAS=1


class AMSEngine(ppl.Engine):
    def protocol(self,method):
        if method=='bf16':return super().protocol(method)
        if method not in METHODS:raise ValueError(method)
        bits,group,weighted=METHODS[method]
        details,_=super().protocol(method)
        details['geometry']={'w':{'parent_format':f'E2M{bits-3}','parent_bits':bits,'shared_lsb_group':group},
            'k':'unquantized','v':'unquantized'}
        details['ams_paper_core']={'official_implementation':False,'exact_author_result_reproduction':False,
            'source':'https://arxiv.org/html/2510.16045v1','source_version':'v1 2025-10-16',
            'scope':'seven linear projections W-only; embedding/output-head/norm and all KV remain unchanged',
            'parent_bits':bits,'group_size':group,'weighted_shared_objective':weighted,
            'finite_codebook':f'signed E2M{bits-3}; all exponent patterns finite; subnormals retained; no Inf/NaN',
            'bias':BIAS,'bias_convention':'A constant codebook bias shift cancels with amax/maxfinite scaling; bias1 chosen consistently.',
            'channel_scale':'one FP32 max(abs(W[row,:]))/maxfinite per output row; all-zero row scale1',
            'scale_axis_source':'AMS says channel-wise; output-row axis is disclosed, supported by official TC-FPx scale[output-row] API.',
            'scale_axis_supporting_source':'https://github.com/usyd-fsalab/fp6_llm/blob/12e83379f16a4ee1494be00db6956aab56baf620/fp6_llm/csrc/utils/weight_dequant.h#L43',
            'supporting_source_role':'TC-FPx baseline/framework convention, not official AMS code.',
            'parent_rtn':'nearest-even finite parent code, preserving sign including negative zero; parent upper fields stay fixed',
            'shared_search':'only bit0/bit1, physical FP32 reconstruction MSE or H_w SSE; exact shared-loss ties choose0',
            'tail_loss':'last row-local short group includes real weights only; dummy weights have zero objective weight',
            'storage':'32-byte tensor header; 4-byte scale per row; g3 uint16 words with zero unused private fields; g4 contiguous private4 and shared1 bitstreams',
            'forward_dtype':self.a.dtype,'objective_before_forward_dtype_cast':True,
            'pack_reference_rows_per_tensor':PACK_CHECK_ROWS,
            'published_score_equivalence_claimed':False,'published_cuda_performance_claimed':False,
            'adapter_sha256':ppl.sha_file(Path(__file__)),'codec_sha256':ppl.sha_file(Path(ams_codec.__file__)),
            'independent_fraction_reference_sha256':ppl.sha_file(Path(ams_math_reference.__file__))}
        return details,ppl.digest({k:v for k,v in details.items() if k!='hostname'})

    @torch.no_grad()
    def prepare(self,method):
        if method=='bf16':return super().prepare(method)
        if method not in METHODS:raise ValueError(method)
        bits,group,weighted=METHODS[method]
        self.active=False;self.method=method;self.scope='w';self.w_weighted=weighted;self.k_weighted=False
        self.storage,self.enc_details={},[]
        totals={'elements':0,'payload_bytes':0,'channel_scale_bytes':0,'header_bytes':0,'total_bytes':0,
            'parent_midpoint_ties':0,'shared_objective_ties':0,'zero_channels':0,'physical_zero_padded_private_slots':0,
            'precast_unweighted_sse':0.,'precast_objective_sse':0.,'forward_dtype_unweighted_sse':0.,'forward_dtype_objective_sse':0.}
        for name,module in self.targets:
            if self.should_stop():raise TimeoutError('AMS encoding time budget exhausted')
            source=self.master[name];rows,cols=source.shape
            h=self.h_w(name,self.device)[None,:] if weighted else None
            chunk_rows=max(1,self.a.weight_chunk_blocks*32//cols)
            layout=ams_codec.sizes(rows,cols,bits,group)
            diag={'parent_midpoint_ties':0,'shared_objective_ties':0,'zero_channels':0}
            loss=torch.zeros(4,dtype=torch.float64,device=self.device);pack_audit=None
            for start in range(0,rows,chunk_rows):
                if self.should_stop():raise TimeoutError('AMS chunk encoding time budget exhausted')
                end=min(rows,start+chunk_rows);x=source[start:end].to(self.device).float()
                result=ams_codec.encode(x,bits,group,h,bias=BIAS)
                q=result['reconstructed'];forward=q.to(self.dtype)
                before=(q-x).square();after=(forward.float()-x).square()
                loss+=torch.stack([before.sum(dtype=torch.float64),(before*(h if h is not None else 1)).sum(dtype=torch.float64),
                    after.sum(dtype=torch.float64),(after*(h if h is not None else 1)).sum(dtype=torch.float64)])
                for key in diag:diag[key]+=result['diagnostics'][key]
                if pack_audit is None and PACK_CHECK_ROWS:
                    n=min(PACK_CHECK_ROWS,len(x));small=dict(result)
                    for key in ('codes','scales','reconstructed','parent_codes'):small[key]=result[key][:n]
                    small['shared']=result['shared'][:n] if result['shared'] is not None else None
                    blob=ams_codec.pack(small);restored=ams_codec.unpack(blob)
                    if not torch.equal(restored['codes'],small['codes'].cpu()) or not torch.equal(restored['reconstructed'],small['reconstructed'].cpu()):
                        raise RuntimeError('AMS actual source-row numerical serialization roundtrip mismatch')
                    pack_audit={'actual_source_rows':n,'actual_bytes':len(blob),'payload_sha256':hashlib.sha256(blob).hexdigest(),
                        'float32_reconstruction_bitwise_equal':True,'codes_equal':True,'whole_tensor_payload_written':False}
                module.weight[start:end].copy_(forward)
            values=loss.cpu().tolist()
            tensor={'tensor':name,'rows':rows,'input_columns':cols,'elements':source.numel(),**layout,**diag,
                'real_tail_weights_per_row':cols%group if group>1 else 0,
                'parent_upper_fields_fixed':True,'pack_reference':pack_audit,
                **dict(zip(['precast_unweighted_sse','precast_objective_sse','forward_dtype_unweighted_sse','forward_dtype_objective_sse'],values))}
            self.enc_details.append(tensor)
            totals['elements']+=source.numel()
            for key in ('payload_bytes','channel_scale_bytes','header_bytes','total_bytes','physical_zero_padded_private_slots'):totals[key]+=layout[key]
            for key in diag:totals[key]+=diag[key]
            for key,value in zip(['precast_unweighted_sse','precast_objective_sse','forward_dtype_unweighted_sse','forward_dtype_objective_sse'],values):totals[key]+=value
        # The common unchanged parameter-data cost is stated separately from
        # this new serialized linear-tensor layout and its own exact headers.
        selected={id(module.weight) for _,module in self.targets}
        fixed_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters() if id(p) not in selected)
        totals.update(quantized_linear_encoded_bpe=totals['total_bytes']*8/totals['elements'],
            quantized_linear_nominal_bpe=bits if group==1 else bits-1+1/group,
            unchanged_parameter_data_bytes=fixed_bytes,
            full_model_tensor_data_plus_linear_headers_bytes=totals['total_bytes']+fixed_bytes,
            full_checkpoint_container_size_claimed=False,whole_model_packed_files_written=False,
            objective='H_w physical reconstruction SSE' if weighted else 'physical reconstruction MSE/SSE',
            encoding_scale_dtype='float32',forward_dtype=self.a.dtype,
            reference_note='Numerical serializer is implemented for full arbitrary tensors; source-row pack checks validate its application without writing redundant full-model artifacts.')
        self.storage['ams_paper_core_encoded_audit']=totals
        self.active=True


def selftest():
    global PACK_CHECK_ROWS
    from types import SimpleNamespace
    ams_codec.selftest();torch.manual_seed(624)
    module=torch.nn.Linear(17,23,bias=False).to(torch.bfloat16)
    e=AMSEngine.__new__(AMSEngine);e.device=torch.device('cpu');e.dtype=torch.bfloat16
    name='model.layers.0.self_attn.q_proj';e.targets=[(name,module)]
    e.master={name:module.weight.detach().clone()};e.hw={'0.qkv':torch.linspace(.01,2,17)}
    e.a=SimpleNamespace(weight_chunk_blocks=4,dtype='bfloat16');e.model=module;e.should_stop=lambda:False
    for method in METHODS:
        e.prepare(method)
        audit=e.storage['ams_paper_core_encoded_audit']
        assert torch.isfinite(module.weight).all() and len(e.enc_details)==1
        assert audit['total_bytes']==ams_codec.sizes(23,17,*METHODS[method][:2])['total_bytes']
        assert e.enc_details[0]['pack_reference']['float32_reconstruction_bitwise_equal']
        first=module.weight.detach().clone();e.prepare(method)
        assert torch.equal(module.weight,first),'preparation must restore the same original weights independently'
    print('PASS six W-only AMS/parent preparation paths, row chunk boundaries, H alignment, BF16 reconstruction, byte audit and actual-row serializer checks')


def main():
    global PACK_CHECK_ROWS,BIAS
    extra=argparse.ArgumentParser(add_help=False)
    extra.add_argument('--ams-pack-check-rows',type=int,default=2)
    extra.add_argument('--ams-bias',type=int,default=1)
    extra.add_argument('--ams-selftest',action='store_true')
    args,rest=extra.parse_known_args();PACK_CHECK_ROWS=args.ams_pack_check_rows;BIAS=args.ams_bias
    if PACK_CHECK_ROWS<0 or BIAS not in (1,2,3):raise ValueError('invalid AMS numeric/reference controls')
    if args.ams_selftest:selftest();return
    sys.argv=[sys.argv[0],*rest];ppl.Engine=AMSEngine;ppl.main()


if __name__=='__main__':main()
