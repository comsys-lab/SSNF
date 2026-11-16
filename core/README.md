# SSNF Code and Data

This directory reproduces the accuracy and serving results of the paper. It contains the SSNF
quantization and dequantization kernels, the vLLM integration, the quantizers, and every evaluation
driver used to produce the tables and figures.

**The accuracy results do not require compiling anything.** Perplexity, the configuration search, and
the zero-shot tasks quantize and dequantize in PyTorch: `pip install` and one GPU are
enough. Only the two task benchmarks that go through the serving path (GSM8K, LongBench) and the
throughput/capacity measurements need the CUDA extension. The two tiers are described in §2 and §3.

---

## 1. Layout

```
core/
├── README.md                  this file
├── LICENSE                    Apache-2.0 (copyright holder anonymized)
├── setup.py                   builds the CUDA extension (sm_120 gencode)
├── env/
│   ├── requirements_accuracy.txt   pinned environment for tier 1 (PPL / configuration search / zero-shot)
│   ├── requirements_serving.txt    pinned environment for tier 2 (vLLM tasks, throughput)
│   └── vllm_patches/ + apply_vllm_patches.py   six patched vLLM 0.23 files (packed-KV dtypes, §2.1)
├── kernel/                    CUDA sources; all six are needed to link
│   ├── kv_paged.cu            paged KV decode attention (SSNF symmetric/asymmetric + MX baselines)
│   ├── kv_attention.cu        fused KV append — Algorithm 3 at serving time
│   ├── wa_gemm.cu             SSNF weight path: wonly_dequant_bf16, wonly_gemm_tc
│   ├── mxint8.cu, mxfp.cu
│   ├── core/ssnf_utils.cuh      E8M0 scale conversion, upper/shared field unpacking
│   └── pybind.cpp             registers torch.ops.ssnf.*
├── quant/
│   ├── ssnf_lib/{pack,reference}.py       reference packer / numpy oracle
│   ├── pack_gpu.py                        on-GPU packer used by the serving path
│   ├── calibrate_h.py                     mean-squared input activation E[x^2]  -> H_w
│   └── calib_importance.py                mean-squared query activation E[q^2]  -> H_K
├── vllm_integration/          --quantization ssnf / mxint8 / mxfp8w / mxfp6w, KV backend, slot layout
├── ssnf_vllm/, ssnf_lib/      import aliases: the sources import these package names;
│                              each __init__.py redirects to vllm_integration/ and quant/ssnf_lib/
├── eval_accuracy/             tier 1 drivers (§3.1)
├── eval_serving/              tier 2 drivers (§3.3)
├── experiments/               one directory per experiment: README + run script + config
├── campaign/                  job generation + GPU queue + status summary (§4)
├── calib/                     calibration statistics, so the encoders can run as reported (§3.4)
├── configs/                   the exact settings behind every reported number
└── logs_subset/               representative raw outputs (§6)
```

## 2. Environment

Two Python environments, because the accuracy stack and the serving stack pin different PyTorch
versions. Keep them separate — the compiled extension links against the serving stack's PyTorch and
will not import under the accuracy stack.

| | Tier 1 — accuracy | Tier 2 — serving / tasks |
|---|---|---|
| Python | 3.12.3 | 3.12.3 |
| PyTorch | 2.12.0+cu132 | 2.11.0+cu130 |
| transformers | 5.12.1 | 5.12.1 |
| datasets | 3.6.0 | 5.0.0 |
| numpy | 1.26.4 | 2.3.5 |
| vLLM | — | 0.23.0 (V1 engine) |
| triton / flashinfer | — | 3.6.0 / 0.6.12 |
| install | `pip install -r env/requirements_accuracy.txt` | `pip install -r env/requirements_serving.txt` |

Hardware used for every reported number: **NVIDIA RTX PRO 4000 Blackwell (sm_120), 24 GB, driver
590.48.01**, four cards, one job per card (no tensor parallelism). OS Ubuntu 24.04.4 LTS,
kernel 6.8.0-136. CUDA runtime 13.0, compiler nvcc 13.2 (V13.2.78). Accuracy jobs are deterministic
and need one card; the batch-size and throughput figures use all four.

The 24 GB memory capacity limits this directory to 7–8B models because a BF16 baseline for a 13B model
does not fit on one card. [`../extended`](../extended) evaluates Llama-2-13B with BF16 weights on two GPUs.

### 2.1 Building the CUDA extension (tier 2 only)

```console
# from the SERVING environment — the .so links against its PyTorch
python env/apply_vllm_patches.py            # one-time: whitelists the packed KV dtypes in vLLM
python setup.py build_ext --inplace
```

The first command copies six small patched vLLM 0.23 files (shipped in `env/vllm_patches/`, each
original kept as `.orig`) into the installed vLLM: it adds the packed cache dtypes
(`ssnf_u3gs16`, `mxint8_kv`, `mxfp8_kv`, `mxfp6_kv`) to the static `kv_cache_dtype` allowlist and
uint8 storage map, routes them to the CUSTOM attention backend, and teaches the KV-pool
page-size accounting the packed slot sizes (reads `SSNF_KV_ASYM`). Without it every serving run
fails at engine boot with a `cache_dtype` validation error. Tier 1 does not need it. The files in
`env/vllm_patches/` are modified vLLM 0.23 sources and remain under vLLM's Apache-2.0 license.

The build compiles all six `kernel/` sources into one extension with
`-O3 -gencode arch=compute_120,code=sm_120 --use_fast_math`, and takes about ten minutes. For a
different GPU, change the `arch`/`code` pair in `setup.py`; note that the TC-QK path uses
`mma.sync.m16n8k16` (BF16 HMMA).

### 2.2 Quick check — confirm the build before running anything

```console
python eval_serving/check_gs8_gemm.py        # weight GEMM vs reference reconstruction (rel err ~2.6e-3)
python eval_serving/check_asym_kstage16.py   # agreement between KV decode paths
```

## 3. Reproducing the results

The selected SSNF configuration for weights and KV caches appears under three `CONFIG` names because
three drivers use three execution paths of the same configuration:

| `CONFIG` | Drivers | Path |
|---|---|---|
| `ssnf_wkv` | `serving_throughput.py`, `serving_capacity.py` | packed weights + packed KV (the real kernels) |
| `ssnf_fq` | `eval_gsm8k.py`, `eval_longbench.py`, campaign | packed weights + KV caches quantized and dequantized in PyTorch |
| `ssnf` | `eval_zeroshot.py` | weights and KV caches quantized and dequantized in PyTorch (no extension) |

The same pattern holds for every format — one name per execution path:

| Format | vLLM registration | serving `CONFIG` | task `CONFIG` | zero-shot `CONFIG` |
|---|---|---|---|---|
| SSNF | `ssnf` | `ssnf_wkv` | `ssnf_fq` | `ssnf` |
| MXINT8 | `mxint8` | `mxint8_wkv` | `mxint8_fq` | `mxint8` |
| MXFP8 (E4M3) | `mxfp8w` | `mxfp8w_wkv` | `mxfp8_fq` | `mxfp8_e4m3` |
| MXFP6 (E2M3) | `mxfp6w` | `mxfp6w_wkv` | `mxfp6_fq` | `mxfp6_e2m3` |

Suffixes: `_wkv` = packed weights + packed KV (real kernels); `_kv` = KV-only variants;
`_fq` = packed weights + KV caches quantized and dequantized in PyTorch; no suffix on the zero-shot
side = weights and KV caches quantized and dequantized in PyTorch. `mxfp8_fqw` is E4M3 with weights
also quantized and dequantized in PyTorch — the packed E4M3 path saturates at the format maximum (448)
while the shared PyTorch codec saturates at 480, so the `consistency` campaign phase measures both
conventions.

### 3.1 Tier 1 — no compilation

All three quantize and dequantize in PyTorch with the same two encoder functions
(`eval_accuracy/wkv_v2_ppl.py`: `ssnf_hw` for keys and values, `wq_scale` for weights), so the
numbers come from one code path.

```console
# Examples use Llama-3.1-8B; §3.4 maps the other checkpoints to their calib/ files.
# WikiText-2 perplexity, configuration search for one tensor type (paper Section 4)
SSNF_SWEEP_PLANE=k SSNF_SWEEP_LB=1,2,3,4,5,6,7 SSNF_SWEEP_GS=1,2,4,8,16,32 \
SSNF_MODEL=NousResearch/Meta-Llama-3.1-8B SSNF_HK=calib/H_K_metallama318b.pt \
python eval_accuracy/sweep_ubgs.py

# a whole configuration for weights and KV caches, including the selected one
SSNF_W_LB=3 SSNF_W_GS=8 SSNF_K_LB=3 SSNF_K_GS=16 SSNF_V_LB=4 SSNF_V_GS=32 \
SSNF_MODEL=NousResearch/Meta-Llama-3.1-8B SSNF_HK=calib/H_K_metallama318b.pt SSNF_JOINT_OUT=joint.csv \
python eval_accuracy/sweep_joint.py

# zero-shot tasks (PIQA, ARC-e, ARC-c, HellaSwag, WinoGrande, BoolQ, OpenBookQA, MMLU)
CONFIG=ssnf SSNF_MODEL=NousResearch/Meta-Llama-3.1-8B SSNF_HK=calib/H_K_metallama318b.pt CP_OUT=zs.json \
python eval_accuracy/eval_zeroshot.py

# encoder unit test: the quantizer must match a direct transcription of Algorithm 1 + 3
SSNF_SWEEP_SELFTEST=1 SSNF_SWEEP_PLANE=k python eval_accuracy/sweep_ubgs.py

# arithmetic check: the dot-product decomposition of paper Figure 2 against a direct product
python eval_accuracy/verify_dot.py

# FP8 saturation check: the shared element codec vs torch's native float8 casts,
# then the PPL of both saturation conventions (VERIFY_ONLY=1 for codec only)
SSNF_MODEL=NousResearch/Meta-Llama-3.1-8B python eval_accuracy/fp8_saturation_check.py
```

The paper's perplexity table is produced by `eval_accuracy/{measure.py, f7_foils.py}`, then
typeset by `make_tables.py`. The driver logs behind the published tables are shipped in
`logs_subset/table/`, so `python eval_accuracy/make_tables.py` reproduces the tables without a GPU;
re-running the drivers overrides the shipped logs via `eval_accuracy/logs/`.

### 3.2 Settings that change the numbers

Pass these explicitly. Several driver defaults differ from the paper configuration, and a missing
variable silently measures something else.

| Variable | Value | If omitted |
|---|---|---|
| `SSNF_W_SCALE_OFFS` | `0` | drivers default to `0` (selected configuration, no E8M0 scale-offset search); pass it anyway so a script that predates the default cannot switch the weight encoder |
| `SSNF_KV_RADIUS` | `0` | drivers default to `0` (selected configuration, no shared-field search); same reasoning |
| `SSNF_CALIB_SPLIT` / `SSNF_CALIB_W` | `train` / `128` | calibration falls back to one window of the *evaluation* split |
| `SSNF_K_LB,K_GS,V_LB,V_GS` | `3,16,4,32` | drivers default to the selected KV cache configuration; pass it explicitly in campaign scripts |
| `SSNF_PROFILE` / `SSNF_HFILE` | `gs8` / `calib/H_w_*.pt` | without the calibration statistics file the weight search uses unweighted squared errors, which is not Algorithm 2 |
| `HF_HUB_OFFLINE` | `1` | datasets or checkpoints may change mid-campaign |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `0` | the in-process quantization hooks are never installed |
| `PYTHONPATH` | package root | `import ssnf_vllm` / `ssnf_vllm` / `ssnf_lib` fails |

### 3.3 Tier 2 — needs the built extension

```console
CONFIG=ssnf_fq SSNF_MODEL=NousResearch/Meta-Llama-3.1-8B SSNF_HK=calib/H_K_metallama318b.pt \
SSNF_PROFILE=gs8 SSNF_HFILE=calib/H_w_llama31.pt \
SSNF_K_LB=3 SSNF_K_GS=16 SSNF_V_LB=4 SSNF_V_GS=32 N=0 CP_OUT=gsm8k.json \
python eval_serving/eval_gsm8k.py            # GSM8K, 5-shot, greedy, full 1319

CONFIG=ssnf_fq ... LB_MAXLEN=4096 NSAMP=0 CP_OUT=lb.json \
python eval_serving/eval_longbench.py        # LongBench, 12 tasks, full sets
```

Serving measurements (both figures) run under vLLM 0.23 V1, eager execution,
`gpu_memory_utilization=0.85`, chunked prefill on, prefix caching off:

```console
export PYTHONPATH=<package root> VLLM_ATTENTION_BACKEND=CUSTOM
export SSNF_KV_SPLIT_MULT=32                       # KV decode split-K multiple
export SSNF_KV_ASYM="3,16,4,32" SSNF_PROFILE=gs8   # SSNF: W u3/gs8, K u3/gs16, V u4/gs32

# throughput — replace CONFIG with bf16 / mxint8_wkv / mxfp8w_wkv / mxfp6w_wkv / ssnf_wkv
CONFIG=ssnf_wkv N=256 IN=2048 OUT=256 MAXSEQ=512 GPU_UTIL=0.85 python eval_serving/serving_throughput.py
# maximum batch size
CONFIG=ssnf_wkv IN=2048 OUT=256 python eval_serving/serving_capacity.py
```

`N` is 256 / 192 / 128 / 96 for context 2K / 4K / 8K / 16K.

### 3.4 Calibration statistics

The encoders need two calibration statistics per model: `H_K`, the mean-squared activation of each
query channel `E[q^2]`, and `H_w`, the mean-squared activation of each input channel `E[x^2]`. **Both are shipped in `calib/`**, so the encoders reproduce exactly what
the paper reports and can be checked independently of the calibration step. They can also be
regenerated in under a minute per model:

```console
SSNF_MODEL=NousResearch/Meta-Llama-3.1-8B CALIB_NSAMPLES=128 CALIB_MAXLEN=2048 \
python quant/calib_importance.py            # env-driven; writes calib/H_K_<tag>.pt
python quant/calibrate_h.py --model <model> --out H_w.pt --split train --nsamples 128 --seqlen 2048
```

Both draw **128 non-overlapping 2,048-token segments from the WikiText-2-raw training split** —
the same calibration set the paper describes. Perplexity is measured on the *test* split, so
calibration and evaluation data are disjoint. These files are statistics derived from a public
dataset, not a new dataset.

## 4. Running everything

`campaign/` reproduces the full accuracy campaign unattended. It dispatches each job to whichever
GPU is idle, so it shares a machine rather than reserving it, and every job checks for its own
output first — an interrupted run resumes instead of restarting.

```console
python campaign/gen_jobs.py zeroshot        # emit a job list (phases: zeroshot zeroshot2 gsm8k
bash   campaign/runq.sh campaign/jobs_zeroshot.txt zeroshot   #  longbench consistency t4_joint)
bash   campaign/finish.sh                   # idempotent retry sweep over every job file
python campaign/summarize.py                # progress and results table
```

`campaign/allowed_gpus` (optional, one GPU index per line) restricts which GPUs the queue uses on
a shared box.

`experiments/` holds the same work split per experiment, each with its own README, run script, and
configuration:

| Directory | Paper | Command | Time |
|---|---|---|---|
| `experiments/accuracy-sweep/` | Section 4 (configuration search) | `bash run.sh` | ~14 GPU-h for the full grid |
| `experiments/accuracy-task/` | zero-shot, GSM8K, and LongBench evaluation | `bash run.sh` | ~30 GPU-h |
| `experiments/serving-capacity/` | maximum batch size figure | `bash run.sh` | ~15 min, deterministic |
| `experiments/serving-throughput/` | throughput figure | `bash run.sh` then `python parse.py` | 41 min per round, 5 rounds ~3.4 h on 4 GPUs |

## 5. Datasets

| Use | Dataset | Split / size |
|---|---|---|
| Perplexity | `wikitext`, `wikitext-2-raw-v1` | **test**, 2,048 tokens at stride 1,024, all windows (281 / 322 / 329 for the three models) |
| Calibration | same | **train**, first 128 × 2,048 tokens |
| Zero-shot | `ybisk/piqa` (val 1,838); `allenai/ai2_arc` ARC-Easy (test 2,376) and ARC-Challenge (test 1,172); `Rowan/hellaswag` (val 10,042); `allenai/winogrande` `winogrande_xl` (val 1,267); `google/boolq` (val 3,270); `allenai/openbookqa` `main` (test 500); `cais/mmlu` `all` (test 14,042) | full, no subsampling |
| GSM8K | `openai/gsm8k` `main` | test, all 1,319, 5-shot |
| LongBench | `THUDM/LongBench` | 12 tasks, 3,200 examples |

Checkpoints: `NousResearch/Meta-Llama-3.1-8B`, `mistralai/Mistral-7B-v0.3`,
`NousResearch/Llama-2-7b-hf`, and the Instruct/chat variants of the same three for LongBench.

LongBench ships one `data.zip`; `datasets>=3.0` dropped its script loader, so the driver reads
`data/{task}.jsonl` straight out of the archive.

## 6. Reading the results correctly

- **Two storage costs.** The configuration search records both the nominal storage cost and the packed
  storage cost, which rounds the shared fields of each block up to a byte. They differ, and
  configurations with a lower nominal storage cost can have the same packed storage cost. The paper's
  format table reports the nominal storage cost.
- **Element conversion.** Every format uses one convention: block scale
  `2^(floor(log2 amax) - emax_elem)`, elements saturating at the format maximum. How often a block
  maximum saturates therefore depends on the format's headroom — MXINT8 1.1%, SSNF 5.5%,
  MXFP6-E2M3 9.3%, MXFP8-E4M3 19.3%.
- **The accuracy evaluation is stricter than serving.** At serving time the self-attention of a prefill
  chunk runs on the raw BF16 keys and values, and quantization applies only on the write to the
  cache. Every evaluation here quantizes *every* attention input, prompt self-attention included, so
  the reported accuracy is a lower bound on the serving path.
- **Not directly comparable to KIVI-style numbers.** Those methods keep the most recent tokens in
  full precision; everything here is quantized.
- **Runs.** Every accuracy measurement is a single deterministic pass — perplexity is one forward
  pass, the zero-shot tasks score log-likelihoods, and the generative benchmarks decode greedily.
  There is no seed, and repetition would return identical numbers; the reported intervals are
  sampling error (binomial, or the standard error of the per-task mean). Only the throughput figure
  averages repeated runs.
- **Scoring.** The driver in `eval_accuracy/eval_zeroshot.py` scores the zero-shot tasks rather than an
  external harness. The BF16 scores of this driver can be compared with the published reference values
  for the same checkpoints.

`logs_subset/` holds representative raw outputs: the configuration search CSVs, the zero-shot result
JSONs, the raw throughput logs the figures are parsed from, and (`logs_subset/table/`) the driver
logs behind the three perplexity tables. The last two regenerate the paper artifacts without a GPU:
`python experiments/serving-throughput/parse.py` for the throughput table and
`python eval_accuracy/make_tables.py` for the perplexity tables.

## 7. Environment variables that select kernel paths

The `SSNF_` prefix is the package namespace, not "the SSNF format only" — several variables
control the baseline kernels.

| Variable | Default | Purpose |
|---|---|---|
| `SSNF_KV_SPLIT_MULT` | 8 (measurements use 32) | KV decode split-K multiple |
| `SSNF_KV_KSTAGE`, `SSNF_KV_TCQK` | on | SSNF KV decode K-staging / TC-QK |
| `SSNF_MXI_TCQK`, `SSNF_MX6_TCQK` | **off** | TC-QK ported onto the MXINT8 / MXFP6 baselines. Fairness experiments only; both are measurably slower (MXINT8 0.64–0.74x, MXFP6 0.54–0.57x of their own defaults), and no reported number uses them. Reproduce with `eval_serving/verify_mxint8_tcqk.py` and `eval_serving/bench_mx6_tcqk.py` |
| `SSNF_PROFILE` | uniform | weight configuration (`gs8` is the selected one) |
| `SSNF_KV_ASYM` | — | `"ku,kgs,vu,vgs"` KV configuration. Asymmetric = different (l_b, g_size) for K and V — the selected configuration |
| `SSNF_NO_SPLIT_TUNE` | 0 | use the formula instead of a measured split-K for the weight GEMM |
| `SSNF_TRITON` | 1 | Triton fused GEMM path |
| `SSNF_APPEND_FUSED` | 1 | fused single-kernel KV append; `0` falls back to the unfused op sequence (A/B gate) |
| `SSNF_HK` / `SSNF_KV_IMPORTANCE` | — | **the same `calib/H_K_*.pt` file, two consumers**: `SSNF_HK` feeds the accuracy drivers, `SSNF_KV_IMPORTANCE` feeds the serving KV encoder. Setting only `SSNF_HK` for a serving run is silently ignored |
| `SSNF_KV_WSHARE_RADIUS` | 0 | serving KV shared-field search radius (selected: 0 = weighted mean, no search) |
| `SSNF_RADIUS` | 0 (= full `1<<l_b` search) | serving weight-encoder search radius; the serving packer uses the full range |
| `SSNF_SPLIT_CACHE` | `~/.cache/ssnf/split_cache.json` | measured split-K choice cache for the weight GEMM |
| `SSNF_KV_CPA`, `SSNF_KV_KSTAGE` | on | A/B gates for the cp.async and K-staging KV decode techniques (`check_asym_kstage16.py` toggles them) |
| `SSNF_NO_SKINNY`, `SSNF_WA_FOLD` | off | MXINT8 A/B fallbacks |
| `SSNF_KV_K4P1`, `SSNF_RA`, `SSNF_FUSED_RA`, `SSNF_KV_SEPSC`, `SSNF_NORES_SEPSC`, `SSNF_KV_WIDE` | off | exploratory variants kept for completeness — **not on any reported path** (4+1 key field split, register-aligned 6.5 b weights, separate-scale / wide-load KV reads) |

## 8. Paper sections to code

| Paper | Content | Files |
|---|---|---|
| Section 3, Figure 1 (representation) | E8M0 block scale + per-group shared field | `quant/ssnf_lib/pack.py`, `quant/pack_gpu.py` |
| Section 3, Figure 2 (arithmetic) | integer reconstruction `(h·2^l + l_g)·S`, dot-product decomposition | `kernel/core/ssnf_utils.cuh` |
| Algorithm 2 (weight quantization) | activation-weighted exhaustive search | `eval_accuracy/wkv_v2_ppl.py::wq_scale`, `quant/calibrate_h.py` |
| Algorithm 3 (KV cache quantization) | weighted mean with the mean-squared query activation (keys), arithmetic mean (values) | `eval_accuracy/wkv_v2_ppl.py::ssnf_hw`, `kernel/kv_attention.cu` |
| Section 4 (accuracy) | perplexity, configuration search, zero-shot, GSM8K, LongBench | `eval_accuracy/`, `eval_serving/` |
| Section 4 (batch size) | KV slot accounting | `vllm_integration/kv_layout.py`, `eval_serving/serving_capacity.py` |
| Section 4 (throughput) | decode attention + weight GEMM | `kernel/kv_paged.cu`, `vllm_integration/ssnf_triton.py` |

The serving path uses `pack_gpu.py`, `wa_gemm.cu`, `kv_attention.cu`, `kv_paged.cu`, and
`ssnf_triton.py`. Every registered kernel op is called by a shipped driver or check; the only
non-default kernel code left is behind the documented gates of §7 (symmetric configuration,
A/B toggles, the fairness ports, and the exploratory flags marked "not on any reported path")
plus the alternative encoders in `ssnf_lib/pack.py`.

### Notation to code

| Paper | Meaning | Python | Kernels | Env vars | Configuration search CSV |
|---|---|---|---|---|---|
| `l_b` | shared field bitwidth | `l_b` (`lb` in the configuration search CSV) | `int l_b`, template `LBv`; per tensor type `ku`/`vu` | `SSNF_{W,K,V}_LB`, `SSNF_SWEEP_LB`; fields 1/3 of `SSNF_KV_ASYM` | `lb` |
| `g_size` | mantissa group size | `g_size` | `int g_size`, `GSIZEv`; `kgs`/`vgs` | `SSNF_{W,K,V}_GS`, `SSNF_SWEEP_GS` | `gs` |
| `h_b` | unshared bitwidth = 8 − l_b | derived: `8 - l_b` (`wbits` in pack_gpu) | derived | — | — |
| `blk` | scaling block size (32) | `BLOCK` | literal 32 | — | — |
| — | number of scaling blocks | `nb` | `NB` | — | — |
| `h` | unshared (upper) word | `h_word` | `h_code`; packed byte array `upper`, `UB` bytes | — | — |
| `l_g` | shared word | `l_g_word` | `l_g_code`; packed byte array `shared`, `SB` bytes | — | — |
| `S` (scale) | E8M0 block scale 2^e | `s_base` (step of l_g), `s_unshared` (= s_base·2^l_b, step of h) | `e8m0_to_scale`; stored as `scale_exp` (int8) | — | — |
| `E[x²]` | mean-squared input activation | `H` | — | `SSNF_HFILE` → `calib/H_w_*.pt` | — |
| `E[q²]` | mean-squared query activation | `H_K` | — | `SSNF_HK` (accuracy) / `SSNF_KV_IMPORTANCE` (serving) → `calib/H_K_*.pt` | — |
| — | encoder search radius | `radius` | — | `SSNF_RADIUS` (W, serving), `SSNF_KV_RADIUS` (KV, accuracy), `SSNF_KV_WSHARE_RADIUS` (KV, serving) | — |
| bits/value | h_b + l_b/g_size + 8/blk | `bpe()` / `bpe_aligned()` | — | — | `bpe_nominal`, `bpe_aligned` |

## 9. Limitations

- **Tensor parallelism is not supported.** With `TP>1` the SSNF KV decode path hits a device-side
  assertion on large batches. Every reported number uses one replica per GPU (`TP=1`).
- **Only sm_120 was verified.** The kernels are written against Blackwell; other architectures
  compile only after changing the gencode pair and were not measured.
- **This directory evaluates 7–8B models on 24 GB cards** (see §2); [`../extended`](../extended) evaluates Llama-2-13B.
- **The fast KV decode paths (K-staging / TC-QK) assume `head_dim == 128`** — true for all three
  reported models; other head sizes fall back to the slower generic kernel.
- **Run-to-run variation** of the throughput measurement stays below 1.3%; accuracy measurements are
  deterministic.

## 10. License and external references

`LICENSE` — Apache License 2.0; the copyright holder is anonymized. This directory contains
no pointer to any repository or dataset controlled by the authors. The only URLs anywhere in the
directory are (a) the Apache license text inside `LICENSE` and (b) upstream vLLM/PyTorch source
comments preserved verbatim in the patched files under `env/vllm_patches/` (third-party issue
trackers and documentation, unmodified so the files stay diffable against vLLM 0.23).

One component is part of this package rather than an external dependency, so it does not appear in
the requirement files: `ssnf_cuda`, the CUDA extension built by `setup.py` (§2.1). No vLLM plugin is
needed — every driver runs with `VLLM_ENABLE_V1_MULTIPROCESSING=0`, so the quantization methods and
the KV backend register in-process via `import ssnf_vllm`.

The `+cu130` PyTorch build pinned in `env/requirements_serving.txt` comes from the official PyTorch
CUDA 13.0 wheel index. Install it the way your site normally installs that build; the exact version
is pinned in the file.
