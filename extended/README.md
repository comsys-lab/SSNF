# Additional Experiments

This directory contains additional experiments that use SSNF quantization, the SSNF CUDA kernels, and the calibration statistics in [`../core`](../core) without modification. The experiments compare SSNF with AMS-Quant, tensor-wise MXFP4/MXFP6 mixed precision, KIVI, KVQuant, and QServe, evaluate each calibration statistic, and extend the accuracy and serving evaluation to Llama-2-13B and IFEval. The experiments also measure the on-the-fly KV cache quantization overhead and the energy of an SSNF dot-product unit on an Alveo U250 FPGA.

- [`accuracy/engine/`](accuracy/engine): PPL evaluation in PyTorch, AMS-Quant quantization, and the KIVI, KVQuant, and QServe evaluation.
- [`accuracy/tasks/`](accuracy/tasks): GSM8K, zero-shot, LongBench, and IFEval evaluation.
- [`accuracy/experiments/`](accuracy/experiments): `run.sh` and `results/` for each experiment.
- [`accuracy/calib/`](accuracy/calib): calibration statistics for Llama-2-13B; the other models use `core/calib`.
- [`serving/`](serving): serving experiments with vLLM and the SSNF CUDA kernels of `core/`; `prepare.py` builds the kernels.
- [`fpga/`](fpga): FPGA implementation of the SSNF, MXINT8, and BF16 dot-product units on the Alveo U250 and their energy measurement.
- [`third_party/`](third_party): `fetch.sh`, which obtains the original KIVI, KVQuant, QServe, and IFEval code, and the modifications applied to that code.
- [`notes/`](notes): SSNF storage costs and memory traffic.

## Setup

```bash
pip install -r ../core/env/requirements_accuracy.txt      # accuracy experiments
cd accuracy && python prepare_wikitext.py && cd ..        # WikiText-2 text (SHA-256 checked)
bash third_party/fetch.sh work                            # only for IFEval and KIVI/KVQuant/QServe
python serving/prepare.py build                           # serving environment: SSNF CUDA kernels -> serving/build
```

The serving experiments use the serving environment of `core/` ([`core/README.md`](../core/README.md), §2) and the SSNF CUDA kernels that `serving/prepare.py build` compiles from `core/` into `serving/build/`. `SSNF_MODEL_DIR` sets a local model directory, and `SSNF_GPU_UUID` sets the GPU for the serving experiments.

## Experiments and Results

SSNF uses SSNF-W, SSNF-K, and SSNF-V ([`../README.md`](../README.md)); the AMS-Quant comparison uses a 3-bit lower/16 weight configuration. PPL uses the WikiText-2 test set with 2,048-token sequences and a 1,024-token stride.

### 1. AMS-Quant ([`accuracy/experiments/ams_quant`](accuracy/experiments/ams_quant))

If only the weights of Llama-3.1-8B are quantized, SSNF achieves a lower PPL increase than AMS-Quant FP5.33, while SSNF uses 2.96% more bits per weight element. AMS-Quant selects shared bits by weight reconstruction error, whereas SSNF selects shared bits by activation-weighted reconstruction error because weight errors across input channels contribute differently to linear-layer outputs. `engine/ams_codec.py` implements AMS-Quant as described in its paper.

| Method | Bits/value | PPL | PPL increase over BF16 |
|---|---:|---:|---:|
| BF16 | 16 | 5.587 | – |
| SSNF (5-bit signed upper, 3-bit lower/16) | 5.50 | 5.681 | 1.67% |
| AMS-Quant FP5.33 | 5.34 | 5.727 | 2.49% |

### 2. Tensor-Wise Mixed Precision ([`accuracy/experiments/tensorwise_mixed_precision`](accuracy/experiments/tensorwise_mixed_precision))

SSNF achieves lower PPL and higher GSM8K and LongBench scores than tensor-wise MXFP4/MXFP6 mixed precision. The baseline assigns MXFP4 or MXFP6 to each weight tensor and to the KV cache of each layer by activation-weighted reconstruction error on the WikiText-2 training split, within the storage cost of SSNF. Both methods use 5.75 bits per weight element, while SSNF and the baseline use 5.00 and 4.94 bits per KV cache element, respectively.

| Method | PPL Llama-3.1-8B | PPL Mistral-7B | GSM8K Llama-3.1-8B (%) | LongBench Llama-3.1-8B-Instruct (score) |
|---|---:|---:|---:|---:|
| BF16 | 5.59 | 4.83 | 51.78 | 52.13 |
| SSNF | 5.75 | 4.88 | 47.99 | 51.87 |
| Mixed precision | 6.04 | 4.96 | 38.97 | 51.31 |

GSM8K uses the 1,319-example test set with 5-shot evaluation. LongBench uses 96 validation examples, 8 from each of 12 tasks, with inputs up to 16K tokens; the paired-bootstrap 95% interval of the SSNF gain over the baseline is [−2.84, +4.33]. `tensorwise_mixed_precision/longbench/worker.py` is the LongBench evaluation of the baseline; it requires the files of its original evaluation and does not run on its own.

### 3. Calibration Statistics ([`accuracy/experiments/calibration_statistics`](accuracy/experiments/calibration_statistics))

Each calibration statistic reduces the PPL increase. W denotes the activation-weighted offline search for weights, which weights each squared error by the mean-squared activation of the corresponding input channel. K denotes the weighted mean of key residuals, which weights each residual by the mean-squared activation of the corresponding query channel. Without W, the weight search uses unweighted squared errors; without K, keys use the arithmetic mean of the residuals.

| Calibration statistics | PPL increase over BF16, Llama-3.1-8B | PPL increase over BF16, Mistral-7B |
|---|---:|---:|
| None | 4.55% | 1.56% |
| W | 3.32% | 1.10% |
| K | 4.07% | 1.45% |
| W and K | 2.91% | 0.97% |

### 4. Llama-2-13B ([`accuracy/experiments/llama2_13b`](accuracy/experiments/llama2_13b), [`serving/llama2_13b_capacity`](serving/llama2_13b_capacity))

SSNF extends the accuracy and serving benefits to a 13B model. On Llama-2-13B, WikiText-2 PPL increases by 1.01% over BF16, from 4.4296 to 4.4744, with the calibration statistics in [`accuracy/calib`](accuracy/calib). With 16K input tokens on a 24 GB RTX PRO 4000, SSNF supports 2x the maximum batch size of MXFP6 (2 versus 1) and achieves 1.79x the decode throughput of MXFP6 at the maximum batch size of each format (26.72 versus 14.96 tokens/s). The 1.79x ratio includes the effect of the larger batch size; at batch size 1, SSNF achieves 19.19 tokens/s, 1.28x the decode throughput of MXFP6.

### 5. IFEval ([`accuracy/experiments/ifeval`](accuracy/experiments/ifeval))

SSNF achieves higher IFEval accuracy than MXFP4 on both instruction-tuned models. IFEval uses 541 prompts with deterministic decoding and up to 1,280 output tokens. Values are prompt-level strict accuracy (%).

| Format | Llama-3.1-8B-Instruct | Mistral-7B-Instruct-v0.3 |
|---|---:|---:|
| BF16 | 75.60 | 48.80 |
| SSNF | 73.75 | 47.13 |
| MXFP4 | 67.28 | 43.25 |

The other downstream tasks, namely eight zero-shot tasks including MMLU, 5-shot GSM8K, and 12 LongBench tasks, use the evaluation in `core/` (`core/eval_accuracy/eval_zeroshot.py`, `core/eval_serving/eval_gsm8k.py`, `core/eval_serving/eval_longbench.py`; [`core/README.md`](../core/README.md), §3).

### 6. KIVI, KVQuant, and QServe ([`accuracy/experiments/kv_cache_baselines`](accuracy/experiments/kv_cache_baselines), [`accuracy/experiments/qserve`](accuracy/experiments/qserve))

SSNF has a lower storage cost than KIVI and KVQuant and increases PPL less than QServe, with each method quantizing different tensors. PPL uses Llama-2-7B on WikiText-2, and KIVI, KVQuant, and QServe use their original code ([`third_party`](third_party)). The storage cost is the sum of weight and KV cache bytes relative to BF16 ([`notes/storage_costs.md`](notes/storage_costs.md)). KIVI and KVQuant quantize only KV caches and keep FP16 weights; QServe also quantizes activations.

| Method | Quantized tensors | Storage cost relative to BF16 | PPL increase over BF16 |
|---|---|---:|---:|
| SSNF | W, KV | 0.36 | 1.22% |
| KIVI-4 | KV | 0.95 | 0.04% |
| KVQuant-4 (1% FP16 outliers) | KV | > 0.92 | 0.32% |
| QServe W4A8KV4 (per-channel) | W, A, KV | 0.25 | 6.14% |

W/A/KV denote quantized weights/activations/KV caches. Without FP16 outliers, KVQuant-4 increases PPL by 1.06%. `qserve/run.py` is the QServe evaluation and requires the files of its original evaluation; `engine/ppl_qserve.py`, `engine/backend.py`, and `engine/native_model.py` contain the PPL evaluation.

### 7. On-the-Fly KV Cache Quantization Overhead ([`serving/kv_quant_overhead`](serving/kv_quant_overhead))

On-the-fly KV cache quantization accounts for a small fraction of decoding time. SSNF and MXFP6 both compute a scale for each 32-element block, convert values, and write the packed fields. SSNF also averages the quantization residuals of each mantissa group and rounds the means into shared fields. At batch size 8 on Llama-3.1-8B, quantization accounts for 8.12% and 1.71% of SSNF attention time for 2K and 16K inputs, respectively, averaged over layers 0, 16, and 31. Across 32 layers at 2K, the additional quantization time of SSNF over MXFP6 accounts for 0.56% of the SSNF decode step time at batch size 8 (26.20 ms, [`serving/llama31_8b_capacity`](serving/llama31_8b_capacity)).

### 8. Energy ([`fpga`](fpga))

An SSNF dot-product unit on an Alveo U250 FPGA exhibits 91.86% less energy than the BF16 dot-product unit and 10.70% less energy than the MXINT8 dot-product unit, at the unit level. [`fpga/README.md`](fpga/README.md) describes the FPGA implementation of the dot-product units and the energy measurement.

### 9. SSNF Storage Costs ([`notes/storage_costs.md`](notes/storage_costs.md))

The nominal storage cost is 5.625 bits per weight element and 4.90625 bits per KV cache element; the packed storage cost is 5.75 and 5.00 bits, respectively. Relative to MXFP6 (6.25 bits per value), the packed storage cost reduces the memory traffic of a single read by 8% for weights and by 20% for KV caches.

## Results

- Each `results/` directory contains the reported values; `raw_summary_sha256` gives the SHA-256 of each original output.
- Source and protocol hashes inside the results refer to the original code. The code here differs from it only in paths, names, and comments, so those hashes do not match the files in this repository.
- The PPL values of mixed precision and the IFEval values of MXFP4 come from outputs that are not included; `results/summary.json` notes each value.
