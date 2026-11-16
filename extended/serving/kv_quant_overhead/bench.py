"""Isolated core-path microbenchmarks. No GPU work occurs for --inventory.

Timing is same-input repeated invocation after warmup. This is not an E2E
serving measurement. GPU execution requires the parent campaign's handoff.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SUB = REPO / "core"
ENERGY = REPO / "extended/serving"
SNAPSHOT = Path(os.environ.get('SSNF_MODEL_DIR', 'models/Meta-Llama-3.1-8B'))
GPU = os.environ.get('SSNF_GPU_UUID', '')
FORMATS = ('bf16', 'mxfp6', 'ssnf')
WEIGHTS = {
    'qkv': ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight'],
    'o': ['self_attn.o_proj.weight'],
    'gu': ['mlp.gate_proj.weight', 'mlp.up_proj.weight'],
    'down': ['mlp.down_proj.weight'],
}
SHAPES = {'qkv': (6144, 4096), 'o': (4096, 4096), 'gu': (28672, 4096), 'down': (4096, 14336)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def inventory():
    files = [SUB / 'vllm_integration/quant.py', SUB / 'vllm_integration/quant_mxfp6.py',
             SUB / 'vllm_integration/ssnf_triton.py', SUB / 'vllm_integration/kv_backend.py',
             SUB / 'vllm_integration/paged_prefill.py', SUB / 'quant/pack_gpu.py',
             SUB / 'quant/ssnf_lib/pack.py', SUB / 'kernel/kv_attention.cu', SUB / 'kernel/mxfp.cu',
             HERE / 'append_oracle.py', HERE / 'runtime_policy.json', HERE / 'split_cache_frozen.json',
             Path(__import__('triton').__file__).parent / 'runtime/autotuner.py',
             SUB / 'calib/H_w_llama31.pt', SUB / 'calib/H_K_metallama318b.pt',
             SNAPSHOT / 'config.json', SNAPSHOT / 'model.safetensors.index.json', Path(__file__)]
    build = json.loads((ENERGY / 'build/manifest.json').read_text())
    return dict(model_snapshot=str(SNAPSHOT), shapes=SHAPES, weights=WEIGHTS,
                input_seed=20261025, gpu_uuid=GPU, extension=build['binary'],
                extension_sha256=build['binary_sha256'],
                source_hashes={str(p): sha(p) for p in files},
                ncu=shutil.which('ncu'), nsys=shutil.which('nsys'),
                scope='actual layer-0 weights; synthetic activations and KV; repeated-input microbenchmark')


def environment(out):
    # Every worker gets the same frozen split-K choices; tile autotuning remains active.
    policy=json.loads((HERE/'runtime_policy.json').read_text())
    source=HERE/policy['original_cache']; frozen=HERE/policy['frozen_cache']
    if any(sha(p)!=policy['sha256'] for p in (source,frozen)):
        raise RuntimeError('Frozen dispatch source hash changed')
    cache=out/'split_cache.json'
    shutil.copyfile(frozen,cache)
    os.environ.update(CUDA_VISIBLE_DEVICES=GPU, SSNF_PROFILE='gs8', SSNF_TRITON='1',
                      SSNF_KV_ASYM='3,16,4,32', SSNF_KV_SPLIT_MULT='32',
                      SSNF_APPEND_FUSED='1', SSNF_W_SCALE_OFFS='0', SSNF_KV_RADIUS='0',
                      SSNF_RADIUS='0', SSNF_KV_WSHARE_RADIUS='0', SSNF_LIGHTMS='0', SSNF_KV_K4P1='0',
                      SSNF_HFILE=str(SUB / 'calib/H_w_llama31.pt'),
                      SSNF_KV_IMPORTANCE=str(SUB / 'calib/H_K_metallama318b.pt'),
                      SSNF_SPLIT_CACHE=str(cache), HF_HUB_OFFLINE='1',
                      HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    for key in ('SSNF_RA', 'SSNF_NO_SPLITK', 'SSNF_NO_SPLIT_TUNE', 'SSNF_MX6_TCQK',
                'SSNF_KV_TCQK', 'SSNF_KV_KSTAGE'):
        os.environ.pop(key, None)
    sys.path[:0] = [str(ENERGY / 'build/lib'), str(SUB)]
    import pynvml as nv
    nv.nvmlInit()
    h = nv.nvmlDeviceGetHandleByUUID(GPU)
    procs = nv.nvmlDeviceGetComputeRunningProcesses(h)
    if any(p.pid != os.getpid() for p in procs):
        raise RuntimeError(f'GPU not exclusive: {[p.pid for p in procs]}')
    import torch
    import vllm
    from vllm.platforms.interface import Platform
    original = Platform.device_id_to_physical_device_id.__func__
    def physical(cls, device_id):
        visible = os.environ.get(cls.device_control_env_var, '').split(',')
        if visible[device_id].startswith('GPU-'):
            return nv.nvmlDeviceGetIndex(nv.nvmlDeviceGetHandleByUUID(visible[device_id]))
        return original(cls, device_id)
    Platform.device_id_to_physical_device_id = classmethod(physical)
    import ssnf_cuda
    import ssnf_vllm
    meta = inventory()
    if sha(ssnf_cuda.__file__) != meta['extension_sha256']:
        raise RuntimeError('Wrong core extension')
    if torch.cuda.device_count() != 1 or 'RTX PRO 4000' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Wrong GPU')
    torch.set_num_threads(4)
    torch.manual_seed(20261025)
    meta.update(worker_pid=os.getpid(),metadata_monotonic_s=time.monotonic(),
                torch=torch.__version__, vllm=vllm.__version__, device=torch.cuda.get_device_name(0),
                driver=nv.nvmlSystemGetDriverVersion(), power_limit_w=nv.nvmlDeviceGetPowerManagementLimit(h)/1000,
                split_cache_start_sha256=sha(cache) if cache.exists() else None,
                split_cache_policy=policy,split_cache_run_path=str(cache),
                triton_cache_autotuning_env=os.environ.get('TRITON_CACHE_AUTOTUNING'),
                dispatch_scope='split-K choices fixed; Triton tile autotuning remains process-local and is captured per case',
                clock_policy='unchanged', cache_policy='repeated same operands after warmup; no forced L2 eviction')
    write(out / 'metadata.json', meta)
    return torch, nv, h


def load_weight(role, torch):
    from safetensors import safe_open
    index = json.loads((SNAPSHOT / 'model.safetensors.index.json').read_text())['weight_map']
    tensors, sources = [], []
    for suffix in WEIGHTS[role]:
        key = 'model.layers.0.' + suffix
        with safe_open(SNAPSHOT / index[key], framework='pt', device='cpu') as f:
            tensors.append(f.get_tensor(key))
        sources.append(dict(tensor=key, shard=index[key]))
    w = torch.cat(tensors, 0).to(device='cuda', dtype=torch.bfloat16)
    if tuple(w.shape) != SHAPES[role]:
        raise RuntimeError(f'Shape mismatch {w.shape}')
    return w, sources


def weight_case(role, m, torch, formats):
    from ssnf_vllm.quant import SSNFConfig, SSNFLinearMethod
    from ssnf_vllm.quant_mxfp6 import MXFP6Config, MXFP6LinearMethod
    w, sources = load_weight(role, torch)
    n, k = w.shape
    gen = torch.Generator(device='cuda').manual_seed(20261025 + m)
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16, generator=gen)
    funcs, refs, info = {}, {}, {}
    for fmt in formats:
        if fmt == 'bf16':
            funcs[fmt] = lambda: torch.nn.functional.linear(x, w)
            refs[fmt] = lambda: torch.nn.functional.linear(x, w)
            info[fmt] = dict(route='BF16 F.linear/cuBLAS', stored_bytes=w.numel()*2,
                             global_bf16_materialization=False)
            continue
        layer = torch.nn.Module()
        layer.register_parameter('weight', torch.nn.Parameter(w.clone(), requires_grad=False))
        if fmt == 'ssnf':
            prefix = {'qkv':'self_attn.qkv_proj', 'o':'self_attn.o_proj', 'gu':'mlp.gate_up_proj', 'down':'mlp.down_proj'}[role]
            method = SSNFLinearMethod(SSNFConfig(), f'model.layers.0.{prefix}')
        else:
            method = MXFP6LinearMethod(MXFP6Config())
        method.process_weights_after_loading(layer)
        funcs[fmt] = lambda method=method, layer=layer: method.apply(layer, x)
        if fmt == 'ssnf':
            from ssnf_vllm import quant
            if not quant._TRITON_OK:
                raise RuntimeError('Core SSNF Triton import/warmup failed; do not silently benchmark fallback')
            wdq = torch.ops.ssnf.wonly_dequant_bf16(layer.ssnf_s, layer.ssnf_hc, layer.ssnf_lgc,
                                                  n, k, k//32, 3, 8)
            refs[fmt] = lambda wdq=wdq: torch.nn.functional.linear(x, wdq)
            tensors = [layer.ssnf_s, layer.ssnf_hc, layer.ssnf_lgc]
            info[fmt] = dict(route='core Triton fused reconstruction and BF16 dot',
                             stored_bytes=sum(t.numel()*t.element_size() for t in tensors),
                             global_bf16_materialization=False,
                             frozen_split_k=json.loads((HERE/'split_cache_frozen.json').read_text())[f'{n},{k},{16 if m<=16 else 1<<(m-1).bit_length()},8'],
                             note='split-K scratch/reduction remains included; warmup and packing excluded')
        else:
            wdqt = torch.ops.ssnf.mxfp6_dequant_bf16(layer.mx6_s, layer.mx6_qw, n, k, k//32)
            refs[fmt] = lambda wdqt=wdqt: x @ wdqt
            deq = m >= (96 if max(n,k)>=8192 else 128)
            info[fmt] = dict(route='dequant + BF16 matmul' if deq else 'fused skinny BF16 reconstruction/GEMM',
                             stored_bytes=layer.mx6_s.numel()+layer.mx6_qw.numel(),
                             global_bf16_materialization=deq, intermediate_bf16_bytes=n*k*2 if deq else 0)
    return funcs, refs, dict(kind='weight', role=role, m=m, n=n, k=k, flops=2*m*n*k,
                             sources=sources, format_info=info, reference_tolerance=0.005)


def kv_case(context, batch, torch, formats):
    from ssnf_vllm.kv_backend import SSNFAttentionImpl
    from ssnf_vllm.paged_prefill import dequant_past_asym, dequant_past_mxfp6
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    from vllm._custom_ops import reshape_and_cache_flash
    bs, hq, hk, d = 16, 32, 8, 128
    pages = math.ceil(context/bs)
    gen = torch.Generator(device='cuda').manual_seed(20261025)
    kval = torch.randn(batch, context, hk, d, generator=gen, device='cuda', dtype=torch.bfloat16)
    vval = torch.randn(batch, context, hk, d, generator=gen, device='cuda', dtype=torch.bfloat16)
    q = torch.randn(batch, hq, d, generator=gen, device='cuda', dtype=torch.bfloat16)
    bt = torch.arange(batch*pages, device='cuda', dtype=torch.int32).view(batch,pages)
    sl = torch.full((batch,), context, device='cuda', dtype=torch.int32)
    cuq = torch.arange(batch+1, device='cuda', dtype=torch.int32)
    slots = torch.arange(batch*context, device='cuda', dtype=torch.int64)
    funcs, refs, info = {}, {}, {}
    def ref_attention(k, v):
        # fp32 math reference, decoding all keys available to one query.
        kk = k.repeat_interleave(hq//hk, dim=1).float()
        vv = v.repeat_interleave(hq//hk, dim=1).float()
        scores = torch.einsum('bhd,bhld->bhl', q.float(), kk)/math.sqrt(d)
        return torch.einsum('bhl,bhld->bhd', scores.softmax(-1), vv)
    for fmt in formats:
        if fmt == 'bf16':
            cache=torch.empty(batch*pages,2,bs,hk,d,device='cuda',dtype=torch.bfloat16)
            kc,vc=cache.unbind(1)
            scale=torch.ones((),device='cuda',dtype=torch.float32)
            reshape_and_cache_flash(kval.reshape(-1,hk,d),vval.reshape(-1,hk,d),kc,vc,slots,'auto',scale,scale)
            out = torch.empty_like(q)
            funcs[fmt] = lambda kc=kc,vc=vc,out=out: flash_attn_varlen_func(
                q=q,k=kc,v=vc,out=out,cu_seqlens_q=cuq,max_seqlen_q=1,
                seqused_k=sl,max_seqlen_k=context,softmax_scale=1/math.sqrt(d),
                causal=True,block_table=bt,fa_version=2)
            refs[fmt] = lambda: ref_attention(kval.permute(0,2,1,3),vval.permute(0,2,1,3))
            info[fmt] = dict(route='vLLM FlashAttention2 paged decode',stored_bytes=2*kval.numel()*2,
                            cache_layout='[num_blocks,2,16,8,128]; strided key/value views from unbind(1)',
                            cache_strides=list(cache.stride()),key_strides=list(kc.stride()),value_strides=list(vc.stride()))
        else:
            impl=SSNFAttentionImpl(hq,d,1/math.sqrt(d),num_kv_heads=hk,
                                   kv_cache_dtype='ssnf_u3gs16' if fmt=='ssnf' else 'mxfp6_kv')
            slot = impl.slot if fmt=='ssnf' else 2*impl.per_head
            cache=torch.zeros(batch*pages,bs,hk,slot,device='cuda',dtype=torch.uint8)
            impl.do_kv_cache_update(SimpleNamespace(layer_name='model.layers.0.self_attn.attn'),
                                   kval.reshape(-1,hk,d),vval.reshape(-1,hk,d),cache,slots)
            funcs[fmt]=lambda impl=impl,cache=cache: impl._paged(q,cache,bt,sl,batch,hq,hk,d)
            rec=[]
            for b in range(batch):
                if fmt=='ssnf':
                    rec.append(dequant_past_asym(cache,bt[b],context,4,impl.KUB,impl.KSB,
                                                impl.VUB,impl.VSB,impl.phK,3,16,4,32,hk,d))
                else:
                    rec.append(dequant_past_mxfp6(cache,bt[b],context,4,impl.per_head,hk,d))
            kr=torch.stack([r[0] for r in rec]); vr=torch.stack([r[1] for r in rec])
            refs[fmt]=lambda kr=kr,vr=vr: ref_attention(kr,vr)
            info[fmt]=dict(route='core asymmetric paged decode' if fmt=='ssnf' else 'core MXFP6 paged decode',
                           stored_bytes=cache.numel(), encoding='core fused append; layer-0 query importance for SSNF')
    return funcs,refs,dict(kind='kv',context=context,batch=batch,hq=hq,hk=hk,d=d,
                           flops=4*batch*hq*context*d,format_info=info,
                           reference_tolerance=0.01,inputs='synthetic BF16 normal; not model activations')


def append_case(context, batch, torch, formats):
    from ssnf_vllm.kv_backend import SSNFAttentionImpl
    from vllm._custom_ops import reshape_and_cache_flash
    from append_oracle import oracle
    if batch!=1 or context%16: raise ValueError('Append protocol requires B1 and page-aligned context')
    bs,hk,d=16,8,128;pages=context//bs
    gen=torch.Generator(device='cpu').manual_seed(20261025)
    key_cpu=torch.randn(batch,hk,d,generator=gen,dtype=torch.bfloat16)
    value_cpu=torch.randn(batch,hk,d,generator=gen,dtype=torch.bfloat16)
    key=key_cpu.cuda();value=value_cpu.cuda()
    slots=torch.tensor([context-1],device='cuda',dtype=torch.int64)
    layer=SimpleNamespace(layer_name='model.layers.0.self_attn.attn')
    importance=torch.load(SUB/'calib/H_K_metallama318b.pt',map_location='cpu')['H_K'][0].float().numpy()
    funcs,refs,info={},{},{}
    for fmt in formats:
        if fmt=='bf16':
            cache=torch.zeros(pages,2,bs,hk,d,device='cuda',dtype=torch.bfloat16)
            kc,vc=cache.unbind(1)
            scale=torch.ones((),device='cuda',dtype=torch.float32)
            def call(cache=cache,kc=kc,vc=vc,scale=scale):
                reshape_and_cache_flash(key,value,kc,vc,slots,'auto',scale,scale)
                return cache
            expected=torch.zeros_like(cache,device='cpu')
            expected[-1,0,-1]=key_cpu[0];expected[-1,1,-1]=value_cpu[0]
            bytes_written=2*hk*d*2
        else:
            impl=SSNFAttentionImpl(32,d,1/math.sqrt(d),num_kv_heads=hk,
                                  kv_cache_dtype='ssnf_u3gs16' if fmt=='ssnf' else 'mxfp6_kv')
            slot=impl.slot if fmt=='ssnf' else 2*impl.per_head
            cache=torch.zeros(pages,bs,hk,slot,device='cuda',dtype=torch.uint8)
            def call(cache=cache,impl=impl):
                impl.do_kv_cache_update(layer,key,value,cache,slots)
                return cache
            expected=torch.zeros_like(cache,device='cpu')
            blob=oracle(key_cpu.float().numpy(),value_cpu.float().numpy(),fmt,importance,SUB)
            expected[-1,-1]=torch.from_numpy(blob[0])
            bytes_written=hk*slot
        funcs[fmt]=call
        refs[fmt]=lambda expected=expected: expected.cuda()
        info[fmt]=dict(route='vLLM BF16 reshape_and_cache_flash' if fmt=='bf16' else 'core fused quantize+pack+scatter',
                       stored_bytes=cache.numel()*cache.element_size(),input_bytes=2*hk*d*2,
                       bytes_written=bytes_written,slot_index=context-1,
                       correctness='exact whole-cache comparison to independent CPU expected cache; all untouched slots must remain zero')
    return funcs,refs,dict(kind='append',context=context,batch=batch,hk=hk,d=d,
                          format_info=info,reference_tolerance=0,
                          inputs='synthetic BF16 normal; one token; actual layer-0 K importance',
                          scope='warm repeated overwrite of same final slot; excludes projection/RoPE, allocation and decode; not growing-context or prefill cost')


def correctness(fn, ref, tolerance, torch):
    y=fn(); r=ref(); torch.cuda.synchronize()
    if tuple(y.shape)!=tuple(r.shape) or not torch.isfinite(y).all() or not torch.isfinite(r).all():
        raise RuntimeError('Invalid output shape or nonfinite values')
    error=float((y.float()-r.float()).norm()/r.float().norm().clamp(min=1e-20))
    maximum=float((y.float()-r.float()).abs().max())
    if error>tolerance:
        raise RuntimeError(f'Reconstructed-value reference mismatch: {error} > {tolerance}')
    answer=dict(relative_frobenius=error,max_abs=maximum,tolerance=tolerance,passed=True)
    if tolerance==0:
        answer.update(exact_equal=bool(torch.equal(y,r)),mismatched_elements=int((y!=r).sum()),
                      checked_elements=y.numel(),checked_bytes=y.numel()*y.element_size())
        if not answer['exact_equal']: raise RuntimeError('Exact cache comparison failed')
    return answer


def reference_case(name, torch):
    """Empirical throughput references, never advertised as hardware ceilings."""
    if name == 'bandwidth':
        import ctypes
        rt=ctypes.CDLL('/usr/local/cuda-13.2/lib64/libcudart.so')
        value=ctypes.c_int()
        rc=rt.cudaDeviceGetAttribute(ctypes.byref(value),38,0)
        if rc != 0 or value.value<=0: raise RuntimeError(f'Cannot verify L2 size: CUDA status {rc}, size {value.value}')
        size_bytes=512*1024*1024
        if size_bytes<4*value.value: raise RuntimeError('Streaming operand is not sufficiently larger than L2')
        src=torch.ones(size_bytes//4,device='cuda',dtype=torch.float32)
        dst=torch.empty_like(src)
        fn=lambda: torch.add(src,1.0,out=dst)
        ref=lambda: torch.full_like(dst,2.0)
        meta=dict(kind='reference',role=name,dtype='float32',elements=src.numel(),
                  source_bytes=size_bytes,destination_bytes=size_bytes,
                  logical_traffic_bytes=2*size_bytes,l2_bytes=value.value,
                  flops=src.numel(),reference_tolerance=1e-7,
                  note='empirical streaming read+write bandwidth; logical bytes cross-checked against NCU traffic')
    elif name == 'dense':
        n=8192
        gen=torch.Generator(device='cuda').manual_seed(20261025)
        a=torch.randn(n,n,generator=gen,device='cuda',dtype=torch.bfloat16)/math.sqrt(n)
        b=torch.randn(n,n,generator=gen,device='cuda',dtype=torch.bfloat16)
        out=torch.empty(n,n,device='cuda',dtype=torch.bfloat16)
        fn=lambda: torch.mm(a,b,out=out)
        torch.backends.cuda.matmul.allow_tf32=False
        ref=lambda: torch.mm(a.float(),b.float())
        meta=dict(kind='reference',role=name,dtype='bfloat16',m=n,n=n,k=n,
                  flops=2*n*n*n,reference_tolerance=0.005,
                  note='empirical dense BF16 GEMM throughput, not a theoretical or guaranteed ceiling; FP32 reference disables TF32')
    else:
        raise ValueError('Unknown reference case')
    return {'reference':fn},{'reference':ref},meta


def timed(fn, torch, seconds):
    def sample(n):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start=time.monotonic(); a.record()
        for _ in range(n): fn()
        b.record(); b.synchronize()
        end=time.monotonic()
        return a.elapsed_time(b),end-start,start,end
    ms,_,_,_=sample(20)
    n=max(20,min(200000,math.ceil(seconds*1000/max(ms/20,0.001))))
    ms,wall,start,end=sample(n)
    # Actual sufficient duration is validated, not silently assumed from pilot.
    if ms<seconds*1000*.90:
        n=math.ceil(n*seconds*1000/max(ms,.001)*1.10)
        ms,wall,start,end=sample(n)
    if ms<seconds*1000*.90:
        raise RuntimeError('Timing interval remained too short')
    return dict(iterations=n,cuda_ms=ms,wall_s=wall,us_per_call=ms*1000/n,
                target_seconds=seconds,minimum_seconds=0.9*seconds,
                worker_pid=os.getpid(),start_monotonic_s=start,end_monotonic_s=end)


def tile_snapshot(meta, formats):
    """Read active Triton autotuner state without launching or changing kernels."""
    if meta['kind']!='weight' or 'ssnf' not in formats:
        return dict(state='not_applicable')
    from ssnf_vllm import ssnf_triton as module
    def primitive(value):
        if value is None or isinstance(value,(str,int,float,bool)): return value
        if isinstance(value,(list,tuple)): return [primitive(x) for x in value]
        if isinstance(value,dict): return {str(k):primitive(v) for k,v in value.items()}
        return str(value)
    def config(c):
        if c is None:return None
        return {k:primitive(getattr(c,k,None)) for k in
                ('kwargs','num_warps','num_stages','num_ctas','maxnreg')}
    mb=16 if meta['m']<=16 else min(512,1<<(meta['m']-1).bit_length())
    prefix=[meta['n'],meta['k'],mb]
    split=meta['format_info']['ssnf']['frozen_split_k']
    active='_ssnf_gemm_rw_splitk_kernel' if split>1 else '_ssnf_gemm_rw_kernel'
    states={}
    for name in ('_ssnf_gemm_rw_kernel','_ssnf_gemm_rw_splitk_kernel'):
        obj=getattr(module,name,None)
        if obj is None:
            states[name]=dict(state='unavailable');continue
        entries=[dict(key=primitive(k),config=config(v)) for k,v in getattr(obj,'cache',{}).items()]
        matches=[e for e in entries if e['key'][:3]==prefix]
        states[name]=dict(state='available',key_names=primitive(getattr(obj,'keys',None)),
                          cache_results=primitive(getattr(obj,'cache_results',None)),
                          best_config=config(getattr(obj,'best_config',None)),cache_entries=entries,
                          matching_case_entries=matches)
    selected=states[active].get('matching_case_entries',[])
    return dict(state='available' if selected else 'selection_unavailable',worker_pid=os.getpid(),
                captured_monotonic_s=time.monotonic(),active_kernel=active,shape_key_prefix=prefix,
                split_k=split,selected_entries=selected,kernels=states,
                scope='Observed process-local tile selection; split-K frozen independently; no extra tuning or kernel launch')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--inventory',action='store_true')
    ap.add_argument('--kind',choices=['weight','kv','append','reference'],default='weight')
    ap.add_argument('--roles',default='qkv,o,gu,down')
    ap.add_argument('--ms',default='1,16,128')
    ap.add_argument('--contexts',default='2048,16384')
    ap.add_argument('--batch',type=int,default=1)
    ap.add_argument('--formats',default=','.join(FORMATS))
    ap.add_argument('--seconds',type=float,default=None)
    ap.add_argument('--rounds',type=int,default=3)
    ap.add_argument('--profile',action='store_true')
    ap.add_argument('--optional',action='store_true',help='Predeclared time-gated M64/M256 extension; keep separate from core')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.seconds is None: args.seconds=10.0 if args.kind=='reference' else 3.0
    if args.optional and (args.kind!='weight' or args.profile or not set(map(int,args.ms.split(','))) <= {64,256}):
        raise ValueError('Optional extension is timing-only weight M64/M256')
    if args.inventory:
        write(args.out,inventory()); return
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists(): raise RuntimeError('Refusing to overwrite completed run')
    write(args.out/'status.json',dict(state='starting',args={k:str(v) for k,v in vars(args).items()}))
    started=time.monotonic()
    write(args.out/'worker_window.json',dict(worker_pid=os.getpid(),start_monotonic_s=started,state='running'))
    try:
        torch,nv,h=environment(args.out)
        formats=('reference',) if args.kind=='reference' else tuple(args.formats.split(','))
        if not all(f in (*FORMATS,'reference') for f in formats): raise ValueError('Unknown format')
        if args.kind=='weight':
            cases=[('weight',role,int(m)) for role in args.roles.split(',') for m in args.ms.split(',')]
        elif args.kind in ('kv','append'):
            cases=[(args.kind,int(c),args.batch) for c in args.contexts.split(',')]
        else:
            cases=[('reference',role,0) for role in args.roles.split(',')]
        results=[]
        for kind,a,b in cases:
            case_start=time.monotonic()
            if kind=='weight': funcs,refs,meta=weight_case(a,b,torch,formats)
            elif kind=='kv': funcs,refs,meta=kv_case(a,b,torch,formats)
            elif kind=='append': funcs,refs,meta=append_case(a,b,torch,formats)
            else: funcs,refs,meta=reference_case(a,torch)
            meta['experiment_tier']='optional_m64_m256' if args.optional else 'core'
            checks={f:correctness(funcs[f],refs[f],meta['reference_tolerance'],torch) for f in formats}
            for f in formats:
                for _ in range(10): funcs[f]()
            torch.cuda.synchronize()
            samples={f:[] for f in formats}
            tiles_before=tile_snapshot(meta,formats)
            phase_start=time.monotonic()
            profile_window=None
            if args.profile:
                if len(cases)!=1 or len(formats)!=1: raise ValueError('Profile exactly one case/format')
                torch.cuda.profiler.start()
                torch.cuda.nvtx.range_push('serving_s4_measured_op')
                funcs[formats[0]]()
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
                profile_window=dict(worker_pid=os.getpid(),start_monotonic_s=phase_start,end_monotonic_s=time.monotonic())
            else:
                for r in range(args.rounds):
                    for f in formats[r%len(formats):]+formats[:r%len(formats)]:
                        row=timed(funcs[f],torch,args.seconds)
                        row.update(round=r+1,temp_c=nv.nvmlDeviceGetTemperature(h,0),
                                   graphics_clock_mhz=nv.nvmlDeviceGetClockInfo(h,0))
                        samples[f].append(row)
                        print(json.dumps(dict(event='timed',case=meta,format=f,sample=row)),flush=True)
            phase_end=time.monotonic()
            tiles_after=tile_snapshot(meta,formats)
            for f in formats:
                results.append(dict(case=meta,format=f,correctness=checks[f],samples=samples[f],
                                    autotuner_before=tiles_before if f=='ssnf' else None,
                                    autotuner_after=tiles_after if f=='ssnf' else None,
                                    case_setup_s=phase_start-case_start,case_elapsed_s=time.monotonic()-case_start,
                                    post_timing_correctness=correctness(funcs[f],refs[f],0,torch) if kind=='append' else None,
                                    worker_pid=os.getpid(),measurement_start_monotonic_s=phase_start,
                                    measurement_end_monotonic_s=phase_end,profile_window=profile_window,
                                    median_us=statistics.median(s['us_per_call'] for s in samples[f]) if samples[f] else None))
            write(args.out/'partial.json',results)
            del funcs,refs
            torch.cuda.empty_cache()
        policy=json.loads((HERE/'runtime_policy.json').read_text())
        dispatch_hashes={str(p):sha(p) for p in (HERE/policy['original_cache'],HERE/policy['frozen_cache'],args.out/'split_cache.json')}
        write(args.out/'dispatch_validation.json',dict(expected_sha256=policy['sha256'],hashes=dispatch_hashes,
                                                      passed=all(v==policy['sha256'] for v in dispatch_hashes.values())))
        if any(v!=policy['sha256'] for v in dispatch_hashes.values()):
            raise RuntimeError('Split dispatch changed during worker execution')
        write(args.out/'results.json',results)
        write(args.out/'status.json',dict(state='complete',elapsed_s=time.monotonic()-started,
                                         profiled=args.profile,rows=len(results)))
    except Exception as exc:
        write(args.out/'status.json',dict(state='failed',error=repr(exc),elapsed_s=time.monotonic()-started))
        raise
    finally:
        write(args.out/'worker_window.json',dict(worker_pid=os.getpid(),start_monotonic_s=started,
                                               end_monotonic_s=time.monotonic(),state='finished'))


if __name__=='__main__': main()
