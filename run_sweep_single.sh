#!/bin/bash
# Single-track k=1..5 sweep. Run inside container, ideally in tmux.
set -e
YAML=./scripts/configs/mamba_pretrain.yaml
ROOT=/workspace/PP_collision/checkpoints
LOGDIR=/workspace/PP_collision/sweep_logs
mkdir -p "$LOGDIR"
CONFIGS=(sweep_single_k1 sweep_single_k2 sweep_single_k3 sweep_single_k4 sweep_single_k5)
echo "Starting single-track sweep at $(date)"
for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "============================================================"
  echo "RUN: $cfg  (started $(date +%H:%M:%S))"
  echo "============================================================"
  python -m train.pretrain.nppmamba.train_multi_gpu_mamba1 \
      --yaml_config="$YAML" --config="$cfg" --run_num="sweep" --root_dir="$ROOT" \
      2>&1 | tee "$LOGDIR/${cfg}.log"
  echo "FINISHED: $cfg ($(date +%H:%M:%S))"
done
echo ""
echo "SINGLE-TRACK SWEEP COMPLETE at $(date)"
for cfg in "${CONFIGS[@]}"; do
  echo "  $cfg : $(grep 'Val loss' "$LOGDIR/${cfg}.log" | tail -1)"
done