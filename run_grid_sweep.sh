#!/bin/bash
# Generalized scaling grid: width x depth x data-fraction.
# Reads real event counts from the data (no hardcoding). Generates a SELF-CONTAINED
# yaml (copies the *default anchor from the main yaml + adds one block per cell) and
# runs the trainer against THAT file -- your main mamba_pretrain.yaml is never touched.
#
# Edit the three arrays below. Run inside container, in tmux.
# CURRENT: all widths, all depths, 100% data.
# Later (with more data): FRACTIONS=(25 50 100)

WIDTHS=(64 128 256 512 1024 1536)
DEPTHS=(12)
FRACTIONS=(1 2.4 11.6 20 47.6 100)

DATA_ROOT=/workspace/PP_collision/data/mmap_v4
MAIN_YAML=./scripts/configs/mamba_pretrain.yaml
GEN_YAML=./scripts/configs/_grid_generated.yaml
ROOT=/workspace/PP_collision/checkpoints
LOGDIR=/workspace/PP_collision/sweep_logs
SMOKE=0
[ "$1" == "--smoke" ] && SMOKE=1 && echo "*** SMOKE MODE: 10 steps per cell ***"
RUN_NUM=$([ "$SMOKE" == "1" ] && echo "smoke" || echo "sweep")
mkdir -p "$LOGDIR"

# --- copy the *default anchor block out of the main yaml into the generated file ---
# grabs from the line containing 'default: &default' up to (but not including) the
# next top-level key (a line starting with a non-space, non-'#').
awk '
  /^[A-Za-z_].*&default/ {cap=1}
  cap==1 {print}
  cap==1 && NR>1 && /^[A-Za-z_]/ && $0 !~ /&default/ {}
' "$MAIN_YAML" > /tmp/_default_probe.txt

# More robust: copy from the &default anchor line until the next line that starts a
# new top-level mapping key (non-indented, ends with ':' and isn't the anchor itself).
python3 - "$MAIN_YAML" "$GEN_YAML" <<'PYEOF'
import sys, re
main_yaml, gen_yaml = sys.argv[1], sys.argv[2]
lines = open(main_yaml).read().splitlines()
out = []
capturing = False
for i, ln in enumerate(lines):
    if not capturing and re.search(r'&default', ln):
        capturing = True
        out.append(ln); continue
    if capturing:
        # stop at next non-indented top-level key that's a new block (not the anchor)
        if ln and not ln[0].isspace() and not ln.lstrip().startswith('#') and '&default' not in ln:
            break
        out.append(ln)
with open(gen_yaml, 'w') as f:
    f.write('\n'.join(out) + '\n')
print(f"[gen] copied *default anchor ({len(out)} lines) into {gen_yaml}")
PYEOF

# --- append one block per (fraction, width, depth) with real event counts ---
CONFIGS=()
STEPS=100000
[ "$SMOKE" == "1" ] && STEPS=10
for frac in "${FRACTIONS[@]}"; do
  fdec=$(python3 -c "print($frac/100.0)")
  nevents=$(python3 dev_scripts/count_events.py "$DATA_ROOT" "$fdec")
  for w in "${WIDTHS[@]}"; do
    for d in "${DEPTHS[@]}"; do
      name="scale_w${w}_d${d}_n${nevents}"
      CONFIGS+=("$name")
      cat >> "$GEN_YAML" <<BLOCK

${name}:
  <<: *default
  data_root: ${DATA_ROOT}
  reader_type: ragged_npy
  stat_dir:  /workspace/PP_collision/data/stats
  checkpoint_dir: /workspace/PP_collision/checkpoints
  klen: 1
  embed_method: pos_only
  voxelize: false
  space_filling_order: true
  rep_aaai: false
  nexttoken: false
  ablate_loss_scale: false
  data_fraction: ${fdec}
  num_layers_backbone: ${d}
  d_state: 16
  d_conv: 4
  expand: 2
  batch_size: 128
  local_batch_size: 8
  valid_batch_size: 32
  local_valid_batch_size: 2
  warmup_steps: $([ "$SMOKE" == "1" ] && echo 2 || echo 10000)
  total_steps: ${STEPS}
  max_lr: 0.0002
  min_lr: 0.00002
  dropout: 0.1
  n_eval_steps: $([ "$SMOKE" == "1" ] && echo 5 || echo 1000)
  max_val_batches: 1000
  use_wandb: true
  save_version: ${name}
BLOCK
    done
  done
done

echo "Generated ${#CONFIGS[@]} configs -> $GEN_YAML"

# --- run them, against the generated self-contained yaml ---
echo "=== GRID SWEEP START $(date) ==="
for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "=== RUN: $cfg ($(date '+%m-%d %H:%M:%S')) ==="
  python -m train.pretrain.nppmamba.train_multi_gpu_mamba1 \
      --yaml_config="$GEN_YAML" --config="$cfg" --run_num="$RUN_NUM" --root_dir="$ROOT" \
      2>&1 | tee "$LOGDIR/${cfg}.log"
done
echo ""
echo "=== COMPLETE $(date) ==="
for cfg in "${CONFIGS[@]}"; do
  echo "  $cfg : $(grep 'Val loss' "$LOGDIR/${cfg}.log" | tail -1)"
done