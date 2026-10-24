#!/usr/bin/env bash
# Same INT8 models on an x86 CPU (quality only).
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  for v in fp32 int8_minmax_pc int8_minmax_pc_rr; do
    python -m mrnetq.infer --bundle "runs/models/alexnet_seed${seed}" --variant "$v" --split test --k 0 \
      --threads 4 --repeats 1 --warmup 5 --cool-above 999 --out "runs/x86/alexnet_s${seed}_test_${v}_k0"
  done
done
