# Serving throughput (paper Section 4, LLM Serving Throughput)

> Requires the SERVING environment with the CUDA extension built **and the vLLM patches
> applied** (top-level README §2.1) — without them the engine fails at boot on `cache_dtype`.

Each format serves the same workload and is measured at **the largest batch it can hold**, so the
memory saved by a narrower format turns into concurrency. One model replica runs on each GPU
(data parallel); all GPUs stay loaded, which is the deployed setting the paper describes.

## Run

```bash
bash run.sh                    # 5 repetitions (paper setting), ~3.4 h on 4 GPUs
ROUNDS=1 bash run.sh           # single pass, ~41 min
FORMATS="bf16 ssnf_wkv" bash run.sh
python parse.py                # table + normalized ratios
python parse.py --tex fig.tex  # also emit the figure
```

Settings live in `config.sh` (model, formats, context/request pairs, output length, GPUs, rounds,
and the SSNF format configuration). Raw output lands in `results/raw_g<gpu>.txt`; re-running appends,
so a crashed sweep can simply be repeated and `parse.py` will use every recorded run.

## What is measured

| Item | Value |
|---|---|
| Contexts / offered requests | 2K:256, 4K:192, 8K:128, 16K:96 |
| Output tokens | 256 per request |
| Batch | not fixed — the KV-cache pool binds, so each format fills to its own limit |
| Engine | vLLM 0.23 (V1), eager, `gpu_memory_utilization=0.85`, chunked prefill on, prefix caching off |
| Repetitions | 5 per (format, context) |

`N` is set well above any format's capacity so the engine stays saturated; a fixed batch would
under-count the formats that can hold more requests.

## Expected result

Normalized to BF16, SSNF reaches roughly 1.59x at 2K and 2.27x at 16K, with the relative standard
deviation across repetitions below 1.3% for every cell. Our own raw logs ship in
`logs_subset/serving/r5_all.txt`; when `results/` has no runs of your own, `python parse.py` falls
back to them, so the reported table reproduces without spending GPU time.

## Notes

- Throughput is a per-replica number. Replicas do not communicate during inference, so an aggregate
  deployment scales linearly (measured within 1-2% of the replica count).
- Run on otherwise idle GPUs. If another process holds memory, the engine cannot reserve its
  KV-cache pool at `gpu_memory_utilization=0.85` and the run aborts at start-up.
