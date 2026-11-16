# Maximum batch size (paper Section 4, Maximum Batch Size)

> Requires the SERVING environment with the CUDA extension built **and the vLLM patches
> applied** (top-level README §2.1) — without them the engine fails at boot on `cache_dtype`.

The maximum batch size follows from how many KV-cache tokens fit in GPU memory once the model
weights are resident. vLLM reports this at start-up as `num_gpu_blocks`, so the measurement is
one engine boot per format and is **deterministic** — repeating it gives the same integer.

## Run

```bash
bash run.sh            # ~15 min: five engine boots on one GPU
GPU=1 bash run.sh
```

## Expected result (24 GB GPU, Llama-3.1-8B)

| Format | num_gpu_blocks | pool tokens | x BF16 |
|---|---|---|---|
| BF16 | 2,015 | 32,240 | 1.00 |
| MXINT8 | 9,664 | 154,624 | 4.80 |
| MXFP8 | 9,676 | 154,816 | 4.80 |
| MXFP6 | 15,308 | 244,928 | 7.60 |
| SSNF | 19,347 | 309,552 | **9.60** |

Maximum batch at a context = pool tokens / (context + 256 output tokens). The numbers reproduce
exactly on an otherwise idle 24 GB GPU; if another process holds memory the pool shrinks, because
the engine reserves a fixed fraction of the *free* memory.
