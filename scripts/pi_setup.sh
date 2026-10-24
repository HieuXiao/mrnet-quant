#!/usr/bin/env bash
# Copy code, models and data to the Pi and start the measurement queue.
# Usage: scripts/pi_setup.sh <ssh host> [remote dir, default mrnetq_pi]
set -euo pipefail
cd "$(dirname "$0")/.."
host=$1; dir=${2:-mrnetq_pi}
ssh "$host" "mkdir -p $dir/models $dir/data"
rsync -a --exclude runs --exclude data --exclude .git ./ "$host:$dir/code/"
rsync -a --exclude _pre runs/models/ "$host:$dir/models/"
rsync -a data/MRNet-v1.0 "$host:$dir/data/"
ssh "$host" "cd $dir && python3 -m venv venv && venv/bin/pip install -q -r code/requirements-pi.txt"
ssh "$host" "cd $dir/code && nohup setsid ../venv/bin/python tools/pi_queue.py --home ~/$dir --python ~/$dir/venv/bin/python > ../queue.out 2>&1 &"
echo "queue started; progress: ssh $host tail -f $dir/logs/queue_status.txt"
