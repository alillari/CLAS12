#!/bin/bash
# Regression width scan m1..m6 on real 3-particle data, k=1.
# Depth fixed (12), only embed_dim varies. Run inside container, INSIDE tmux.
# No 'set -e': a single failed run won't kill the rest of the sweep.
YAML=./scripts/configs/mamba_pretrain.yaml
ROOT=/workspace/PP_collision/checkpoints
LOGDIR=/workspace/PP_collision/sweep_logs
mkdir -p "$LOGDIR"

CONFIGS=(scale_m1_reg scale_m2_reg scale_m3_reg scale_m4_reg scale_m5_reg scale_m6_reg)

echo "=== SCALING SWEEP START $(date) ==="
for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "============================================================"
  echo "RUN: $cfg   (started $(date '+%Y-%m-%d %H:%M:%S'))"
  echo "============================================================"
  python -m train.pretrain.nppmamba.train_multi_gpu_mamba1 \
      --yaml_config="$YAML" --config="$cfg" --run_num="sweep" --root_dir="$ROOT" \
      2>&1 | tee "$LOGDIR/${cfg}.log"
  echo "FINISHED: $cfg   ($(date '+%H:%M:%S'))"
done

echo ""
echo "=== SCALING SWEEP COMPLETE $(date) ==="
echo "Final val losses:"
for cfg in "${CONFIGS[@]}"; do
  echo "  $cfg : $(grep 'Val loss' "$LOGDIR/${cfg}.log" | tail -1)"
done