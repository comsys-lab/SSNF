"""Metric A capacity, TP-parametrized. pool = num_gpu_blocks*block_size; max_conc = pool/(IN+OUT).
Env: CONFIG, IN, OUT, MODEL, TP (tensor_parallel_size)."""
import os
if os.environ.get("TP","1")=="1":
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"
import ssnf_vllm  # noqa
from vllm import LLM, SamplingParams
CFG=os.environ.get("CONFIG","bf16")
IN=int(os.environ.get("IN","2048")); OUT=int(os.environ.get("OUT","256"))
TP=int(os.environ.get("TP","1"))
MODEL=os.environ.get("MODEL","NousResearch/Meta-Llama-3.1-8B")
kw=dict(model=MODEL, dtype="bfloat16", enforce_eager=True, max_model_len=IN+OUT+16,
        gpu_memory_utilization=0.85, tensor_parallel_size=TP,
        enable_prefix_caching=False, enable_chunked_prefill=True)
if CFG=="mxint8_wkv": kw.update(quantization="mxint8", kv_cache_dtype="mxint8_kv")
elif CFG=="mxfp8w_wkv": kw.update(quantization="mxfp8w", kv_cache_dtype="mxfp8_kv")
elif CFG=="mxfp6w_wkv": kw.update(quantization="mxfp6w", kv_cache_dtype="mxfp6_kv")
elif CFG=="ssnf_wkv": kw.update(quantization="ssnf", kv_cache_dtype="ssnf_u3gs16")
llm=LLM(**kw)
cc=llm.llm_engine.vllm_config.cache_config
pool=cc.num_gpu_blocks*cc.block_size; conc=pool/(IN+OUT)
llm.generate(["hi"], SamplingParams(max_tokens=1))
print(f"CAP cfg={CFG} TP={TP} IN={IN} OUT={OUT} | num_gpu_blocks={cc.num_gpu_blocks} "
      f"block_size={cc.block_size} pool_tokens={pool} max_concurrent={conc:.2f}", flush=True)
