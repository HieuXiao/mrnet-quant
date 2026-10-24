#!/usr/bin/env bash
# Three planes x three seeds for both backbones (about 4 h per backbone on an RTX 3060).
set -euo pipefail
cd "$(dirname "$0")/.."
for backbone in alexnet mobilenet_v3_small; do
  for seed in 0 1 2; do
    for plane in sagittal coronal axial; do
      python -m mrnetq.train --backbone "$backbone" --plane "$plane" --seed "$seed"
    done
  done
done
