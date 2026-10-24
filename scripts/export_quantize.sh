#!/usr/bin/env bash
# Export every seed to ONNX and build all FP16/INT8 variants.
set -euo pipefail
cd "$(dirname "$0")/.."
for backbone in alexnet mobilenet_v3_small; do
  for seed in 0 1 2; do
    python -m mrnetq.export --backbone "$backbone" --seed "$seed"
    python -m mrnetq.quantize --bundle "runs/models/${backbone}_seed${seed}"
  done
done
