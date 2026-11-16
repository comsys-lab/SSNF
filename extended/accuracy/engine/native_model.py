"""Teacher-forced access to the official packed W4A8 and paged KV4 model."""
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import json,sys,time
import torch
import torch.nn.functional as F
from backend import R, load_backend


def configure_imports():
    load_backend()
    sys.path.insert(0,str(R/'sources/omniserve'))


def allocate_cache(batch,length,layers=32):
    # Matches official CacheEngine: 64 tokens x 32 heads x 128 channels / 2,
    # then FP16 scale and minimum per head and token, separately for K and V.
    pages=(length+63)//64
    page_bytes=64*32*128//2+64*32*4
    caches=[];tables=[]
    offsets=torch.arange(batch*pages,dtype=torch.int64,device='cuda').reshape(batch,pages)*page_bytes
    for _ in range(layers):
        k=torch.zeros((batch*pages,page_bytes),dtype=torch.int8,device='cuda')
        v=torch.zeros_like(k)
        tables.append(torch.stack((offsets+k.data_ptr(),offsets+v.data_ptr()),dim=1).contiguous())
        caches.append((k,v))
    return caches,tables,pages,page_bytes


class DenseConfig:
    sparse_kv_cache_enabled=lambda self:False
    get_dec_sub_chunk_per_block=lambda self:4
    get_sparse_decode_mode=lambda self:0
    get_dec_dynamic_sparse_token_budget=lambda self:0
    get_dec_selector_update_interval=lambda self:1


class NativeAdapter:
    def __init__(self,group_size):
        configure_imports()
        from transformers import LlamaConfig
        from omniserve.modeling.models.llama_w4a8_unpad import LlamaForCausalLM
        from omniserve.sampling_params import SamplingParams
        from omniserve.utils.quant_config import QServeQuantConfig
        config=LlamaConfig.from_pretrained(R/'checkpoints'/('Llama-2-7B-QServe' if group_size==-1 else 'Llama-2-7B-QServe-g128'),local_files_only=True)
        assert config.hidden_size==4096 and config.num_hidden_layers==32
        mc=SimpleNamespace(sp_attn_config=DenseConfig(),kv_quant_granularity='fine_grained',multiblock_switch=4096,chunk_prefill_size=2048)
        previous=torch.get_default_dtype();torch.set_default_dtype(torch.float16)
        try:
            with torch.device('cuda'):
                self.model=LlamaForCausalLM(config,mc,group_size,SamplingParams(temperature=0),quant_config=QServeQuantConfig(weight_bits=4),kv_cache_config=dict(INT4_ENABLED=True,ZEROS_ENABLED=True))
        finally:torch.set_default_dtype(previous)
        tag='Llama-2-7B-QServe' if group_size==-1 else 'Llama-2-7B-QServe-g128'
        self.model.load_weights(str(R/'checkpoints'/tag))
        self.model.eval().requires_grad_(False)
        self.calls=Counter();self.handles=[]
        for i,layer in enumerate(self.model.model.layers):
            attn=layer.self_attn
            attn.register_buffer('retrieval_head_flags',torch.ones(32,dtype=torch.int32,device='cuda'))
            attn.register_buffer('head_rank_table',torch.arange(32,dtype=torch.int32,device='cuda'))
            attn.head_mask_type=attn.streaming_info=None
            attn.num_retrieval_kv_heads=32;attn.num_streaming_kv_heads=0
            attn.sink_size=attn.local_size=attn.sink_blocks=attn.local_blocks=0
            for label,module in [('qkv',attn.qkv_proj),('o',attn.o_proj),('gate_up',layer.mlp.gate_up_proj),('down',layer.mlp.down_proj),('attention',attn.decoding_attention_wrapper)]:
                def hook(mod,inp,out,key=(i,label)):self.calls[key]+=1
                self.handles.append(module.register_forward_hook(hook))
            assert attn.kv_cache_config==dict(INT4_ENABLED=True,ZEROS_ENABLED=True)
            assert attn.decoding_attention_wrapper.forward.__name__=='forward_pure_dense'
        self.weight_tensor_bytes=sum(x.numel()*x.element_size() for x in self.model.state_dict().values())

    @torch.inference_mode()
    def score(self,inputs,target_starts,heartbeat=None):
        from omniserve.utils.input_metadata import InputMetadata
        batch,length=inputs.shape
        caches,tables,pages,page_bytes=allocate_cache(batch,length-1)
        metadata=InputMetadata(is_prompt=True,padding_offsets=torch.zeros(batch,dtype=torch.int32,device='cuda'),cu_seqlens=torch.arange(batch+1,dtype=torch.int32,device='cuda'),retrieval_context_lens=torch.ones(batch,dtype=torch.int32,device='cuda'),streaming_context_lens=None,max_seq_len=1,retrieval_block_tables=tables,streaming_block_tables=[None]*32,max_retrieval_block_table_len=pages,max_streaming_block_table_len=0,kv_scales=None,kv_cache_dtype=torch.int8,batched_seq_len=batch,model=self.model)
        self.calls.clear();totals=torch.zeros(batch,dtype=torch.float64,device='cuda')
        for t in range(length-1):
            metadata.is_prompt=t==0
            metadata.max_seq_len=t+1
            metadata.retrieval_context_lens.fill_(t+1)
            if t:metadata.padding_offsets=metadata.cu_seqlens=None
            hidden=self.model.model(inputs[:,t],metadata)
            # Keep the unquantized FP16 head at batch one, matching the single-sequence
            # scorer. Native INT8 layers and packed KV4 remain batched and unchanged.
            logits=torch.cat([self.model.lm_head(hidden[i:i+1]) for i in range(batch)],dim=0)
            assert logits.shape==(batch,32000)
            losses=F.cross_entropy(logits.float(),inputs[:,t+1],reduction='none').double()
            totals+=losses*torch.tensor([t+1>=start for start in target_starts],device='cuda')
            if heartbeat and ((t+1)%64==0 or t==length-2):heartbeat(t+1)
        for i in range(32):
            for label in ('qkv','o','gate_up','down'):assert self.calls[(i,label)]==length-1
            assert self.calls[(i,'attention')]==length-2
        assert all(torch.count_nonzero(k).item()>0 and torch.count_nonzero(v).item()>0 for k,v in caches)
        audit=dict(cache_updates=length-1,compressed_cache_reads_all_layers=length-2,final_cache_length=length-1,
            reset_cache_at_window_start=True,native_w4a8_calls_per_layer=4*(length-1),native_kv4_layers=32,logit_projection_batch_size=1,
            cache_dtype='int8 container, two INT4 values/byte plus FP16 scale and minimum',cache_page_bytes=page_bytes,
            cache_pages_per_sequence=pages,cache_allocated_bytes=sum(x.numel()*x.element_size() for pair in caches for x in pair),weight_tensor_bytes=self.weight_tensor_bytes)
        return totals.tolist(),audit

    def close(self):
        for h in self.handles:h.remove()
