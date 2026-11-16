# Downstream task accuracy

Three benchmarks over the same nine formats and three model families.

| Benchmark | Tier | Driver | Protocol |
|---|---|---|---|
| zero-shot, 8 tasks | 1 — no build | `eval_accuracy/eval_zeroshot.py` | log-likelihood scoring, base models, full sets |
| GSM8K | 2 — needs the extension | `eval_serving/eval_gsm8k.py` | 5-shot, greedy, all 1,319 |
| LongBench, 12 tasks | 2 — needs the extension | `eval_serving/eval_longbench.py` | task-native metrics, instruct/chat models, 3,200 examples |

The zero-shot suite needs no build because it is pure fake quantization. GSM8K and LongBench go
through the serving path, where SSNF weights are really packed, so they need the compiled extension.

## Run

Settings (models, formats, fixed env) live in `../../campaign/experiment_config.py`; this
experiment has no `config.sh` of its own. `run.sh` dispatches every phase (zero-shot, zero-shot2,
GSM8K, LongBench, the E4M3 consistency pair, joint W+KV) through the campaign queue.

```console
bash run.sh
python ../../campaign/summarize.py
```

Every run is deterministic — greedy decoding, no seed. The intervals reported in the paper are
sampling error, not run-to-run variation.

The scorer is ours, not an external harness: its BF16 scores can be compared with the
published reference values for the same checkpoints.
