# The exact settings behind every reported number

One place to look up what a reported number was measured with. Every variable here changes the
result; a missing one silently measures something else (see README §3.2).

## Deployed SSNF configuration

| Plane | shared bits `l_b` | mantissa group `g_size` | unshared bits `8 - l_b` | nominal bits/element |
|---|---|---|---|---|
| Weight | 3 | 8 | 5 | 5.625 |
| Key | 3 | 16 | 5 | 5.4375 |
| Value | 4 | 32 | 4 | 4.375 |

Scaling block = 32 elements, E8M0 block scale. KV average = 4.906 bits/element.
The reconstruction word is 8-bit integer, so `l_b + (8 - l_b) = 8` on every plane.

## Encoder settings

| Setting | Value | Note |
|---|---|---|
| `SSNF_W_SCALE_OFFS` | `0` | no E8M0 scale-offset search. The search is a per-model trade, not a uniform gain |
| `SSNF_KV_RADIUS` | `0` | no shared-field search; the deployed path does not perform one |
| `SSNF_PROFILE` | `gs8` | weight bit configuration |
| `SSNF_HFILE` | `calib/H_w_<model>.pt` | weight importance `E[x^2]`. Without it the search runs unweighted, which is not Algorithm 2 |
| `SSNF_HK` | `calib/H_K_<model>.pt` | key importance `E[q^2]` |
| (rotation) | not implemented in this package | the paper uses no rotation |

## Calibration

| Item | Value |
|---|---|
| Dataset | `wikitext`, `wikitext-2-raw-v1`, **train** split |
| Segments | first 128 non-overlapping windows of 2,048 tokens |
| `SSNF_CALIB_SPLIT` / `SSNF_CALIB_W` | `train` / `128` |

Evaluation uses the **test** split, so calibration and evaluation data are disjoint.

## Accuracy measurement

| Item | Value |
|---|---|
| Perplexity | WikiText-2 test, 2,048-token windows at stride 1,024, **all** windows (281 / 322 / 329 for Llama-3.1-8B / Mistral-7B / Llama-2-7B) |
| Sweep grid | `l_b` in {1,…,7} × `g_size` in {1,2,4,8,16,32}, same full-window protocol |
| Zero-shot | 8 tasks, full sets, log-likelihood scoring, base models |
| GSM8K | 5-shot, greedy (`temperature=0`), strict match, all 1,319 |
| LongBench | 12 tasks, task-native metrics, instruct/chat models, per-model max length 4096 / 8192 / 3456 |
| Runs | one deterministic pass; no seed |

## Serving measurement

| Item | Value |
|---|---|
| Engine | vLLM 0.23.0, V1, eager execution |
| `gpu_memory_utilization` | 0.85 |
| Chunked prefill / prefix caching | on / off |
| Parallelism | one replica per GPU, `TP=1`, four GPUs |
| `VLLM_ATTENTION_BACKEND` | `CUSTOM` |
| `SSNF_KV_SPLIT_MULT` | 32 |
| `SSNF_KV_ASYM` | `3,16,4,32` |
| Request count `N` | 256 / 192 / 128 / 96 at context 2K / 4K / 8K / 16K |
| Output length | 256 tokens |
| `MAXSEQ` | 512 |
| Runs | repeated rounds, averaged; spread below 1.3% |

## Baseline formats compared

BF16, MXINT8, MXFP8-E4M3, MXFP8-E5M2, MXFP6-E2M3, MXFP6-E3M2, MXFP4-E2M1, NVFP4, SSNF.
E5M2 and E3M2 are iso-bit foils for E4M3 and E2M3 (8.25 and 6.25 bits/element respectively).

All formats use one conversion convention: block scale `2^(floor(log2 amax) - emax_elem)` with
elements saturating at the format maximum.
