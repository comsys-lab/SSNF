# SSNF Dot-Product Unit

An SSNF dot-product unit on an Alveo U250 FPGA exhibits 91.86% less energy than the BF16 dot-product unit and 10.70% less energy than the MXINT8 dot-product unit, at the unit level. This directory provides the SystemVerilog design of the SSNF, MXINT8, and BF16 dot-product units, the host code, and the scripts that build the FPGA images and measure the energy of each unit.

## Dot-Product Units

Each unit computes one 32-element dot product per cycle.

| Unit | Format | Arithmetic |
|---|---|---|
| `ssnf_dot32.sv` | SSNF-W: 5-bit signed upper, 3-bit lower/8, E8M0/32 | 5x5-bit unshared–unshared partial products for each value (B in Figure 2 of [`../../README.md`](../../README.md)), 3x8-bit unshared–shared and 3x3-bit shared–shared partial products for each mantissa group (C, D, and E), integer accumulation, and a final FP32 conversion that applies the sum of the two E8M0 exponents (A) |
| `mxint8_dot32.sv` | MXINT8: INT8, E8M0/32 | 8x8-bit integer products and integer accumulation |
| `bf16_dot32.sv` | BF16: E8M7 | 8x8-bit products (`bf16_mul.sv`) and FP32 adders (`fp32_add.sv`) |

`ssnf_bench.sv` instantiates up to 16 copies of the unit of one format, each with its own `input_source.sv`, which generates new input values in every cycle. `reference.py` and `source_reference.py` reproduce the outputs of the units and of `input_source.sv` in Python.

## Energy Measurement

The reported energy corresponds to the energy that the active units consume for each dot product, excluding the energy that the FPGA consumes while the units are inactive. Inactive units keep their inputs fixed and their clock enable off within the same FPGA image, while the input generation remains active. Dividing the power increase (J/s) by the number of dot products per second gives this energy per dot product (J/dot product).

- **Platform.** Alveo U250 (`xcu250-figd2104-2L-e`, `xilinx_u250_gen3x16_xdma_4_1_202210_1`), Vitis/Vivado 2023.1, and XRT. Each format uses its own FPGA image, and 8 units run at 100 MHz.
- **Inputs.** All three formats use the same input values, which each format represents without quantization error. These input values do not represent the distributions of model weights or activations.
- **Procedure.** Each measurement reads the FPGA power reported by XRT with the units inactive, active, and inactive; each interval has 5s of warm-up and 45s of measurement. Five fixed seeds give five measurements per format, with a different format sequence for each seed.
- **Energy per dot product.** E = (P_active − mean(P_inactive,before, P_inactive,after)) / (dot products per second).

| Format | Energy (nJ/dot product), mean ± s.d. over 5 seeds | Power increase (W) |
|---|---:|---:|
| SSNF | 0.6730 ± 0.0159 | 0.5384 ± 0.0127 |
| MXINT8 | 0.7536 ± 0.0283 | 0.6029 ± 0.0226 |
| BF16 | 8.2705 ± 0.0524 | 6.6163 ± 0.0419 |

Averaged over the five seeds, SSNF uses 7.598 nJ/dot product less energy than BF16 (95% t-interval [7.527, 7.668]) and 0.081 nJ/dot product less energy than MXINT8 (95% t-interval [0.033, 0.128]). [`results/triplets.csv`](results/triplets.csv) contains the values of each seed, [`results/windows.csv`](results/windows.csv) the power of each interval, and [`results/audit.json`](results/audit.json) the consistency checks of these results. These values are unit-level FPGA results, not GPU serving energy or ASIC energy.

## Reproduction

```bash
# 1. RTL simulation against the Python reference (Icarus Verilog >= 12)
python3 scripts/generate_tests.py vectors --registered
bash scripts/check_rtl.sh sim vectors              # writes sim/verification.json

# 2. Build one static image per format (0 = SSNF, 1 = MXINT8, 2 = BF16)
source env.sh                                       # Vitis 2023.1 and XRT settings
export SSNF_RTL_GATE=$PWD/sim/verification.json
bash scripts/build.sh build_ssnf xcu250-figd2104-2L-e \
    /opt/xilinx/platforms/xilinx_u250_gen3x16_xdma_4_1_202210_1/xilinx_u250_gen3x16_xdma_4_1_202210_1.xpfm 16 100 0
bash scripts/review_build.sh build_ssnf 0          # timing, DRC, and resource review of the routed image

# 3. Power measurement with the units inactive, active, and inactive, then aggregation
python3 scripts/measure_paired.py --help
python3 scripts/aggregate_paired.py <measurement root>
```

`measure_paired.py` runs one format and one seed set per call, in three stages (`smoke`, `pilot`, `measure`); it takes the host program, the FPGA image, the test vectors, and the build validation of that image. `aggregate_paired.py` reads the five `measure` runs of each format and computes the mean, standard deviation, and 95% t-interval of the energy per dot product of each format.

The FPGA images contain 16 SSNF units, 16 MXINT8 units, and 8 BF16 units; 8 units run in each measurement (`config.json`).
