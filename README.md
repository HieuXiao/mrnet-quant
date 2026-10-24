# mrnet-quant

Code for the paper *Resource-Efficient Knee MRI Inference on Raspberry Pi 5: Accuracy–Latency–Energy Trade-offs for an MRNet-Style Model* (submitted to ICIIT 2027).

An MRNet-style anterior cruciate ligament (ACL) tear classifier (AlexNet features per slice, channel-wise max over slices, logistic fusion of three planes) is trained in FP32, exported to ONNX, converted to FP16 and post-training INT8 variants, and measured exam by exam on a Raspberry Pi 5 CPU: AUC, decision agreement with the FP32 model, latency, peak memory and board energy from the PMIC.

## Contents
```
mrnetq/          preprocessing, model, training, ONNX export, quantization, exam-level inference (NumPy + ONNX Runtime)
tools/           Pi measurement queue, PMIC power logger, task list, tables and figure, text analyses
configs/         split_manifest.json (internal train/validation split), pi_tasks.json (all Pi runs of the paper)
scripts/         end-to-end reproduction steps
```

## Data
The Stanford MRNet v1.0 dataset is available from the Stanford AIMI shared datasets under a research use agreement; every user has to request access individually. Place it as
```
data/MRNet-v1.0/{train,valid}/{sagittal,coronal,axial}/NNNN.npy
data/MRNet-v1.0/{train,valid}-acl.csv
```
or set `MRNETQ_DATA`. Neither the data, images derived from it, nor trained weights are distributed here.

The official training split (1,130 exams) is divided into an internal training part (961 exams) and an internal validation part (169 exams), stratified by the ACL label (`configs/split_manifest.json`). All choices are made on the internal validation part. The official validation split (120 exams) is the test set and is run once per configuration.

## Installation
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # workstation: training, export, quantization, tables
pip install -r requirements-pi.txt       # Raspberry Pi: inference only
```
Versions are those used in the paper (Python 3.12 on the workstation, Python 3.13 on Raspberry Pi OS / Debian 13).

## Reproduction
| Step | Command | Output |
|---|---|---|
| 1. Train (3 planes x 3 seeds x 2 backbones) | `scripts/train_all.sh` | `runs/train/` |
| 2. Export and quantize | `scripts/export_quantize.sh` | `runs/models/<backbone>_seed<k>/<variant>/` |
| 3. Measure on the Pi | `scripts/pi_setup.sh <ssh host>` | `~/mrnetq_pi/results/`, `logs/pmic.csv` |
| 4. Same models on x86 | `scripts/run_x86.sh` | `runs/x86/` |
| 5. Tables, figure, text numbers | `scripts/make_tables.sh <ssh host>` | `tables/` |

`configs/pi_tasks.json` lists the 108 Pi runs (`python tools/make_tasks.py` regenerates it). The queue can be stopped and restarted at any time; finished runs are skipped. Measurement protocol: a run starts below 60 °C; latency runs pause before any exam when the SoC is above 68 °C until it cools to 60 °C, then run one unmeasured exam; runs whose clock dropped below 2.3 GHz are repeated. Energy per exam is the mean PMIC power during the exam times its duration.

Single steps:
```bash
python -m mrnetq.train --plane sagittal --seed 0                       # one plane, one seed
python -m mrnetq.export --seed 0                                        # ONNX + head.json + fusion.json
python -m mrnetq.quantize --bundle runs/models/alexnet_seed0 --variants int8_minmax_pc
python -m mrnetq.infer --bundle runs/models/alexnet_seed0 --variant int8_minmax_pc --split test --k 16 --out runs/demo
```

## Notation
| Paper | Code (`--variant`) | Name in experiment logs |
|---|---|---|
| FP32 reference | `fp32` | P1 |
| FP16 | `fp16` | Q1 |
| INT8 MinMax (per-channel / per-tensor) | `int8_minmax_pc` / `int8_minmax_pt` | Q2a / Q2d |
| INT8 Percentile 99.99 | `int8_percentile_pc` | Q2b |
| INT8 Entropy (KL, 2048 bins) | `int8_entropy2048_pc` | Q2c |
| INT8 MinMax, max-slice calibration | `int8_maxslice_pc` | Q5, alias `int8_aggaware_pc` |
| INT8, conv1 in FP32 | `int8_conv1fp32_pc` | Q3, alias `int8_mix_c1_pc` |
| INT8 with reduced-range weights | `int8_minmax_pc_rr` | |
| K slices per plane (all, 24, 16, 8) | `--k 0/24/16/8` | S_all, S24, S16, S8 |
| Changed decisions (flips) | `pred` differs from the FP32 run of the same seed | |

Entropy calibration: `quantize_static` in ONNX Runtime 1.31 uses 128 histogram bins, which gives the same model as MinMax for this network; `int8_entropy2048_pc` computes the ranges with 2,048 bins and passes them as tensor overrides.

x86: ONNX Runtime's U8S8 kernels on AVX2/AVX-512 CPUs without VNNI can saturate 16-bit intermediate sums (see the ONNX Runtime quantization documentation); `int8_minmax_pc_rr` avoids this.

## License
MIT (see `LICENSE`). The MRNet data remain subject to the Stanford AIMI research use agreement.
