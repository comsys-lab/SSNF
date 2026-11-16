# SSNF Storage Costs

The nominal storage cost of SSNF is 5.63 bits per weight element and an average of 4.91 bits per KV cache element. The packed storage cost, which rounds the shared fields of each 32-element block up to a byte, is 5.75 bits per weight element and 5.00 bits per KV cache element. Relative to MXFP6 (6.25 bits per value), the packed storage cost reduces the memory traffic of a single read by 8% for weights and by 20% for KV caches. These values follow from the storage costs, not from kernel results. The SSNF configurations follow [`core/configs/deployed.md`](../../core/configs/deployed.md).

## Nominal and Packed Storage Costs

The nominal storage cost is the average bitwidth, which comprises the per-value bits and the amortized shared and scaling-factor bits. The packed storage cost additionally rounds the shared fields of each 32-element block up to a byte, as in the packed SSNF representation.

| Configuration | Per-value field | Shared field | Scale | Nominal storage cost (bits/value) | Packed bits per 32 values | Packed storage cost (bits/value) |
|---|---|---|---|---:|---|---:|
| SSNF-W | 5-bit signed upper | 3-bit lower/8 | E8M0/32 | 5 + 3/8 + 8/32 = 5.625 | 160 + 12→16 + 8 = 184 | 5.75 |
| SSNF-K | 5-bit signed upper | 3-bit lower/16 | E8M0/32 | 5 + 3/16 + 8/32 = 5.4375 | 160 + 6→8 + 8 = 176 | 5.50 |
| SSNF-V | 4-bit signed upper | 4-bit lower/32 | E8M0/32 | 4 + 4/32 + 8/32 = 4.375 | 128 + 4→8 + 8 = 144 | 4.50 |
| KV average | | | | 4.90625 | | 5.00 |

In SSNF-W, the 3-bit shared field of each group of eight values adds only 0.375 bits per value to the nominal storage cost.

## Memory Traffic Relative to MXFP6

With one E8M0 scale per 32 values, MXFP6 requires 6.25 bits per value. If a kernel reads every packed byte of a tensor a single time, the memory traffic follows the bits per value:

- Weights: 1 − 5.75 / 6.25 = 8% less memory traffic.
- KV caches: 1 − 5.00 / 6.25 = 20% less memory traffic.

## KV Cache Writes during Decoding

During decoding, SSNF quantizes and appends each newly generated key and value, while attention reads the entire KV cache. For Llama-3.1-8B (8 KV heads, head dimension 128) at batch size 8 with 16K tokens:

- Writes per layer and generated token: 8 requests × 2 (K, V) × 8 × 128 values × 5 bits = 10.24 KB.
- Reads per layer and generated token: 8 requests × 16,384 tokens × 2 × 8 × 128 values × 5 bits = 167.77 MB.
- The writes account for 0.006% of the bytes that attention reads.

[`../serving/kv_quant_overhead`](../serving/kv_quant_overhead) measures the corresponding quantization time.

## Storage Costs of KIVI, KVQuant, and QServe

The storage cost in the KIVI, KVQuant, and QServe comparison is the sum of linear-layer weight bytes and KV cache bytes for one 2,048-token Llama-2-7B sequence, relative to the same sum in BF16 (14,025,752,576 bytes). Embeddings and output heads remain in BF16 for all methods, and the sum excludes them together with tensor headers and PagedAttention block sizes.

| Method | Weights (bytes) | KV cache | Storage cost relative to BF16 |
|---|---:|---|---:|
| SSNF | 4,654,628,864 (5.75 bits per value) | 335,544,320 bytes (K 5.50, V 4.50 bits per value) | 0.3558 |
| QServe, per-channel | 3,243,442,176 (INT4 + FP16 scale and zero point) | 285,212,672 bytes (4 bits + FP16 scale and zero point per head and token) | 0.2516 |
| KIVI-4 | 12,952,010,752 (FP16) | 5.685 bits per value | 0.9506 |
| KVQuant-4 | 12,952,010,752 (FP16) | not in the sum | > 0.9234 (weights only) |
