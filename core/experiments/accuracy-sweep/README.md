# Configuration search (paper Section 4)

Grid: shared bitwidth `l_b` in {1..7} x mantissa group size `g_size` in {1,2,4,8,16,32} = 42 points.
`g_size=1` is included so that "every group size dividing 32" holds literally.

Two modes:

- **per-plane** (`sweep_ubgs.py`) — quantize one plane (W, K or V), keep the other two in BF16.
  3 models x 3 planes x 42 points = 378 measurements.
- **joint** (`sweep_joint.py`) — a whole W+KV configuration, including the deployed one.

No compilation needed: both are pure-PyTorch fake quantization.

## Run

Settings (models, planes, grids, calibration mapping) live in `config.sh`.

```console
bash run.sh                       # the whole grid, one plane/model per GPU
python ../../campaign/summarize.py
```

Each measurement is a single deterministic pass over all WikiText-2 test windows (2,048 tokens,
stride 1,024). Roughly 14 GPU-hours for the full per-plane grid.

## Output

`eval_accuracy/logs/{plane}_{model}.csv`, one row per grid point, recording perplexity, the nominal
bits/element, and the byte-aligned deployed cost. Our own run is in `../../logs_subset/sweep/`.
