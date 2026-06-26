#!/bin/bash
# Runs the full k=1..5 sweep on both real and straight-track data (10 runs).
# Each run logs to sweep_logs/<config>.log. Run inside the container, ideally
# inside tmux so it survives disconnection.
#
# Usage:
#   bash run_sweep.sh

set -e  # stop if any run errors out (remove this line to continue past failures)

YAML=./scripts/configs/mamba_pretrain.yaml
ROOT=/workspace/PP_collision/checkpoints
LOGDIR=/workspace/PP_collision/sweep_logs

mkdir -p "$LOGDIR"

CONFIGS=(
  sweep_real_k1
  sweep_real_k2
  sweep_real_k3
  sweep_real_k4
  sweep_real_k5
  sweep_straight_k1
  sweep_straight_k2
  sweep_straight_k3
  sweep_straight_k4
  sweep_straight_k5
)

echo "Starting sweep of ${#CONFIGS[@]} runs at $(date)"

for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "============================================================"
  echo "RUN: $cfg   (started $(date +%H:%M:%S))"
  echo "============================================================"
  python -m train.pretrain.nppmamba.train_multi_gpu_mamba1 \
      --yaml_config="$YAML" \
      --config="$cfg" \
      --run_num="sweep" \
      --root_dir="$ROOT" \
      2>&1 | tee "$LOGDIR/${cfg}.log"
  echo "FINISHED: $cfg   ($(date +%H:%M:%S))"
done

echo ""
echo "============================================================"
echo "SWEEP COMPLETE at $(date)"
echo "Logs in $LOGDIR/"
echo "Final val losses:"
for cfg in "${CONFIGS[@]}"; do
  last=$(grep "Val loss" "$LOGDIR/${cfg}.log" | tail -1)
  echo "  $cfg : $last"
done