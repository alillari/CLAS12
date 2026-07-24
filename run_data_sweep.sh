#!/bin/bash
# Generalized scaling grid: width x depth x data-fraction.
# Reads real event counts from the data (no hardcoding). Generates a SELF-CONTAINED
# yaml (copies the *default anchor from the main yaml + adds one block per cell) and
# runs the trainer against THAT file -- your main mamba_pretrain.yaml is never touched.
#
# Edit the three arrays below. Run inside container, in tmux.
# CURRENT: FIGURE 5(b) style -- m3 (w256) fixed, data fraction swept.
# Fractions match the paper's convention (1, 2.4, 11.6, 20, 47.6, 100 %).
# Step count is held constant across fractions, as the paper does, so smaller
# fractions simply cycle their subset more times.
# Hyperparameters are the tuned base recipe from the 100-trial w256 Optuna
# search; use_mup:true makes the trainer scale the matrix-group LR by
# (width/256), so the same base max_lr is correct at every width.

WIDTHS=(256)
DEPTHS=(12)
FRACTIONS=(11.6)

# --- SWEEP TAG: everything for this sweep lands in ONE descriptive folder ---
# Convention: Descriptive_Name_YYYY-MM-DD  (ISO date sorts chronologically).
# Change this for every new sweep so results never mix together.
# Tip: use $(date +%F) to auto-date, but a fixed string is safer if you may
# resume/re-run the sweep on a later day and want it in the same folder.
SWEEP_TAG=Data_Fraction_Test_1_2026-07-22

DATA_ROOT=/workspace/PP_collision/data/mmap_v4
MAIN_YAML=./scripts/configs/mamba_pretrain.yaml
SMOKE=0
[ "$1" == "--smoke" ] && SMOKE=1 && echo "*** SMOKE MODE: 10 steps per cell ***"
RUN_NUM=$([ "$SMOKE" == "1" ] && echo "smoke" || echo "sweep")

# smoke runs go to their own tag so they never touch real sweep results
[ "$SMOKE" == "1" ] && SWEEP_TAG="${SWEEP_TAG}_SMOKE"

# descriptive sweep folder lives inside checkpoints/
SWEEP_DIR=/workspace/PP_collision/checkpoints/${SWEEP_TAG}
ROOT=${SWEEP_DIR}
LOGDIR=${SWEEP_DIR}/logs
GEN_YAML=${SWEEP_DIR}/_grid_generated.yaml
mkdir -p "$ROOT" "$LOGDIR"

echo "=== SWEEP: ${SWEEP_TAG} ==="
echo "    all output -> ${SWEEP_DIR}"

# --- copy the *default anchor block out of the main yaml into the generated file ---
# grabs from the line containing 'default: &default' up to (but not including) the
# next top-level key (a line starting with a non-space, non-'#').
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
  checkpoint_dir: ${ROOT}
  klen: 1
  embed_method: pos_only
  voxelize: false
  space_filling_order: true
  rep_aaai: false
  nexttoken: false
  ablate_loss_scale: false
  data_fraction: ${fdec}
  embed_dim: ${w}
  num_layers_backbone: ${d}
  d_state: 16
  d_conv: 4
  expand: 2
  use_mup: true
  mup_base_width: 256
  batch_size: 128
  local_batch_size: 128
  valid_batch_size: 32
  local_valid_batch_size: 2
  warmup_steps: $([ "$SMOKE" == "1" ] && echo 2 || echo 10000)
  total_steps: ${STEPS}
  max_lr: 0.005654656285806099
  min_lr: 0.0005654656285806099
  dropout: 0.0903784154249514
  weight_decay: 0.0842571876326289
  grad_clip_value: 1.8025125913594795
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
SUMMARY="${SWEEP_DIR}/RESULTS.txt"
{
  echo "sweep: ${SWEEP_TAG}"
  echo "finished: $(date)"
  echo "widths: ${WIDTHS[*]}   depths: ${DEPTHS[*]}   fractions: ${FRACTIONS[*]}"
  echo ""
  for cfg in "${CONFIGS[@]}"; do
    echo "  $cfg : $(grep 'Val loss' "$LOGDIR/${cfg}.log" | tail -1)"
  done
} | tee "$SUMMARY"
echo ""
echo "Summary written to $SUMMARY"