#!/usr/bin/env bash
set -euo pipefail
campaign=$1
data=$2
backbone=$3
source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
mkdir -p "$campaign"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
export PYTHONPATH="$source_root"
export MPLCONFIGDIR=/tmp/m6-last-two-matplotlib
cd "$source_root"
finish() {
  rc=$?
  printf 'exit_code=%s\nfinished_utc=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$campaign/launcher_result.txt"
}
trap finish EXIT
python train/downstream/experiments/momentum_unfreeze/run.py \
  --campaign-root "$campaign" --data-root "$data" --backbone "$backbone" \
  > "$campaign/launch.log" 2>&1
