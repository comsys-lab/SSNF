"""Unfixed-batch offline throughput: issue N >> capacity requests, let vLLM's continuous-batching
scheduler pack as many as fit (bounded by KV pool = num_gpu_blocks). Measures whether the KV-footprint
benefit converts to aggregate tok/s. max_num_seqs set high so KV-memory (capacity) is the binding
constraint, not the seq cap."""
import os, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import ssnf_vllm  # noqa
from vllm import LLM, SamplingParams, TokensPrompt
CFG = os.environ.get("CONFIG", "bf16")
N = int(os.environ.get("N", "256")); IN = int(os.environ.get("IN", "1024")); OUT = int(os.environ.get("OUT", "512"))
MAXSEQ = int(os.environ.get("MAXSEQ", "512"))
UTIL = float(os.environ.get("GPU_UTIL", "0.85"))
kw = dict(model=os.environ.get("MODEL", "NousResearch/Meta-Llama-3.1-8B"), dtype="bfloat16", enforce_eager=True,
          max_model_len=IN + OUT + 16, gpu_memory_utilization=UTIL, tensor_parallel_size=1,
          max_num_seqs=MAXSEQ, enable_prefix_caching=False,
          enable_chunked_prefill=True)  # vLLM V1 default; passed explicitly to match serving_capacity.py
if CFG == "mxint8_kv": kw.update(kv_cache_dtype="mxint8_kv")
elif CFG == "ssnf_kv": kw.update(kv_cache_dtype="ssnf_u3gs16")
elif CFG == "fp8": kw.update(kv_cache_dtype="fp8")
elif CFG == "mxfp6_kv": kw.update(kv_cache_dtype="mxfp6_kv")
elif CFG == "mxfp8_kv": kw.update(kv_cache_dtype="mxfp8_kv")
elif CFG == "ssnf_wkv": kw.update(quantization="ssnf", kv_cache_dtype="ssnf_u3gs16")
elif CFG == "mxint8_wkv": kw.update(quantization="mxint8", kv_cache_dtype="mxint8_kv")
elif CFG == "mxfp8w_wkv": kw.update(quantization="mxfp8w", kv_cache_dtype="mxfp8_kv")
elif CFG == "mxfp6w_wkv": kw.update(quantization="mxfp6w", kv_cache_dtype="mxfp6_kv")
llm = LLM(**kw)
cc = llm.llm_engine.vllm_config.cache_config
pool = cc.num_gpu_blocks * cc.block_size
kv_conc = pool // (IN + OUT)                       # how many IN+OUT reqs fit in KV concurrently
tok = llm.get_tokenizer()
base = tok.encode("The old lighthouse keeper recorded every ship and storm across forty long years by the sea. ")
ids = (base * ((IN // len(base)) + 2))[:IN]
prompts = [TokensPrompt(prompt_token_ids=ids) for _ in range(N)]
sp = SamplingParams(temperature=0.0, max_tokens=OUT, ignore_eos=True)
llm.generate(prompts[:8], SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))  # warmup
t0 = time.perf_counter()
outs = llm.generate(prompts, sp)
wall = time.perf_counter() - t0
gen = sum(len(o.outputs[0].token_ids) for o in outs)
print(f"THRUPUT cfg={CFG} N={N} IN={IN} OUT={OUT} maxseq={MAXSEQ} | pool_tok={pool} "
      f"kv_fit={kv_conc} (min(maxseq,kv_fit)={min(MAXSEQ,kv_conc)}) | wall={wall:.2f}s "
      f"gen_tok={gen} tok_s={gen/wall:.1f} req_s={N/wall:.3f}", flush=True)
