#!/usr/bin/env bash
# Pull the Pi results and rebuild the tables, Figure 1 and the numbers in the text.
# Usage: scripts/make_tables.sh <ssh host> [remote dir, default mrnetq_pi]
set -euo pipefail
cd "$(dirname "$0")/.."
host=$1; dir=${2:-mrnetq_pi}
mkdir -p runs/pi
rsync -a "$host:$dir/results/" runs/pi/results/
rsync -a "$host:$dir/logs/" runs/pi/logs/
python tools/make_tables.py --pi runs/pi/results --logs runs/pi/logs --x86 runs/x86 --out tables
for what in h4 threshold ece overhead; do python tools/analyze.py "$what" --pi runs/pi/results; done
