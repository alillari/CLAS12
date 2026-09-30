#!/usr/bin/env bash
set -euo pipefail

factor_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
factor_repo=$(cd -- "$factor_dir/../../../.." && pwd)
factor_root=${CLAS12_FACTOR_ROOT:?Set CLAS12_FACTOR_ROOT to the shared experiment output directory}
factor_kind=${CLAS12_FACTOR_KIND:?Set CLAS12_FACTOR_KIND to pretrained or adapteronly}
factor_python=${CLAS12_FACTOR_PYTHON:-python}
factor_backbone=/home/alessio/ML-work/pretrained-FMs/campaign_4/scale_w1536_d12_n39553933/ckpt_best.tar
factor_data=/home/alessio/ML-work/HIPO_processing/multi_particle_storage/big-data/mmap_canonical_loose_truthseg_event_v6_02
export PYTHONPATH="$factor_repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$factor_root/mpl-cache"
mkdir -p "$factor_root"
exec > >(tee -a "$factor_root/launcher.log") 2>&1
trap 'factor_exit=$?; echo "$factor_exit" > "$factor_root/exit_code.txt"' EXIT
git -C "$factor_repo" rev-parse HEAD > "$factor_root/source_commit.txt"
git -C "$factor_repo" status --porcelain > "$factor_root/source_status.txt"

if [[ ! -f "$factor_root/training_track_count.txt" ]]; then
  echo cohort > "$factor_root/stage.txt"
  "$factor_python" "$factor_repo/train/downstream/experiments/m6_context/prepare_long.py" --data-root "$factor_data" --output "$factor_root"
fi
factor_tracks=$(cat "$factor_root/training_track_count.txt")
"$factor_python" "$factor_dir/render.py" --output "$factor_root" --tracks "$factor_tracks"
if [[ ! -f "$factor_root/target_stats.json" ]]; then
  echo statistics > "$factor_root/stage.txt"
  "$factor_python" "$factor_repo/train/downstream/scripts/compute_regression_target_stats.py" \
    --data-root "$factor_data" --split pretrain --task mom \
    --adapter-sample-mode event_segment --segment-min-clusters 12 \
    --low-thr 1 --high-thr 100 --limit-size "$factor_tracks" \
    --output "$factor_root/target_stats.json"
fi
if [[ "$factor_kind" == pretrained ]]; then test -f "$factor_backbone"; fi
for factor_mode in track gather membership; do
  factor_arm="${factor_kind}_${factor_mode}"
  echo "training_${factor_arm}" > "$factor_root/stage.txt"
  factor_flags=()
  if [[ "$factor_kind" == pretrained ]]; then factor_flags=(--usepretrain --pretrained_ckpt "$factor_backbone"); fi
  "$factor_python" "$factor_repo/train/downstream/train_track_regression.py" \
    --yaml_config "$factor_root/experiment.yaml" --config "$factor_arm" \
    "${factor_flags[@]}" --eventnumber "$factor_tracks" \
    --train_batch_size 128 --seed 11 --run_num 11 \
    --root_dir "$factor_root/$factor_arm" \
    --checkpoint_dir "$factor_root/$factor_arm/checkpoints" \
    --resolved_config_path "$factor_root/$factor_arm/resolved_config.json" \
    --artifact_summary "$factor_root/$factor_arm/artifacts.json"
  echo "evaluating_${factor_arm}" > "$factor_root/stage.txt"
  "$factor_python" "$factor_dir/prepare_eval.py" "$factor_root" "$factor_arm"
  "$factor_python" "$factor_repo/train/downstream/eval/evaluate_track_regression.py" \
    --analysis-config "$factor_root/$factor_arm/analysis.yaml"
done
echo complete > "$factor_root/stage.txt"
