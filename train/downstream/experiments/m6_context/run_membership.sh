#!/usr/bin/env bash
set -euo pipefail

context_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
context_repo=$(cd -- "$context_dir/../../../.." && pwd)
context_python=${CLAS12_CONTEXT_PYTHON:-python}
context_checkpoint=${CLAS12_CONTEXT_BACKBONE:-/home/alessio/ML-work/pretrained-FMs/campaign_4/scale_w1536_d12_n39553933/ckpt_best.tar}
context_runs=${CLAS12_CONTEXT_RUNS:-/home/alessio/ML-work/result_deep_storage/experiments/m6_membership_$(date +%Y%m%d-%H%M%S)}
export CLAS12_CONTEXT_DATA_ROOT=${CLAS12_CONTEXT_DATA_ROOT:-/home/alessio/ML-work/HIPO_processing/multi_particle_storage/big-data/mmap_canonical_loose_truthseg_event_v6_02}
export CLAS12_CONTEXT_STATS="$context_runs/target_stats.json"
export PYTHONPATH="$context_repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$context_runs/mpl-cache"
mkdir -p "$context_runs"
exec > >(tee -a "$context_runs/launcher.log") 2>&1
trap 'context_exit=$?; echo "$context_exit" > "$context_runs/exit_code.txt"' EXIT
git -C "$context_repo" rev-parse HEAD > "$context_runs/source_commit.txt"
git -C "$context_repo" status --porcelain > "$context_runs/source_status.txt"
echo preflight > "$context_runs/stage.txt"

"$context_python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable in the experiment container')
from mamba_ssm import Mamba
from train.downstream.model import MambaTrackRegressionHead
backbone = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 12, 16, device='cuda', requires_grad=True)
backbone(x).sum().backward()
head = MambaTrackRegressionHead(input_dim=1536, num_layers=1,
    num_output_dim=4, num_feature_layers=12, num_embedder_layers=0,
    pooling='attention', embed_method='pos_only', pe_method='nerf',
    track_membership_channel=True).cuda()
points = torch.randn(2, 12, 3, device='cuda')
features = torch.randn(12, 2, 12, 1536, device='cuda')
head(points, features, pretrain=True,
     track_membership=(torch.arange(12, device='cuda')[None, :] < 6).expand(2, -1),
     padding_mask=torch.ones(2, 12, device='cuda', dtype=torch.bool))['pred_regression'].sum().backward()
assert head.input_proj[1].in_features == 1537
assert head.input_proj[1].weight.grad[:, -1].abs().sum() > 0
torch.cuda.synchronize()
print('PASS: Mamba1 and membership-conditioned momentum adapter CUDA forward/backward')
print('GPU:', torch.cuda.get_device_name(), 'torch:', torch.__version__)
PY

test -f "$context_checkpoint"
echo cohort > "$context_runs/stage.txt"
"$context_python" "$context_dir/prepare_long.py" --data-root "$CLAS12_CONTEXT_DATA_ROOT" --output "$context_runs"
"$context_python" - "$context_runs/experiment.yaml" <<'PYCONFIG'
import sys
from ruamel.yaml import YAML
yaml = YAML()
with open(sys.argv[1]) as stream:
    configs = yaml.load(stream)
config = dict(configs['event_context'])
config['adapter_sample_mode'] = 'event_segment_membership'
with open(sys.argv[1], 'w') as stream:
    yaml.dump({'membership_context': config}, stream)
PYCONFIG
context_tracks=$(cat "$context_runs/training_track_count.txt")
echo statistics > "$context_runs/stage.txt"
"$context_python" "$context_repo/train/downstream/scripts/compute_regression_target_stats.py" \
  --data-root "$CLAS12_CONTEXT_DATA_ROOT" --split pretrain --task p_phi_theta \
  --adapter-sample-mode event_segment --segment-min-clusters 12 \
  --low-thr 1 --high-thr 100 --limit-size "$context_tracks" --output "$CLAS12_CONTEXT_STATS"

for context_arm in membership; do
  echo "training_$context_arm" > "$context_runs/stage.txt"
  "$context_python" "$context_repo/train/downstream/train_track_regression.py" \
    --yaml_config "$context_runs/experiment.yaml" --config "${context_arm}_context" \
    --usepretrain --pretrained_ckpt "$context_checkpoint" \
    --eventnumber "$context_tracks" --train_batch_size 32 --seed 42 \
    --root_dir "$context_runs/$context_arm" \
    --checkpoint_dir "$context_runs/$context_arm/checkpoints" \
    --resolved_config_path "$context_runs/$context_arm/resolved_config.json" \
    --artifact_summary "$context_runs/$context_arm/artifacts.json"
done

"$context_python" - "$context_runs" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {}
for arm in ('membership',):
    record = json.loads((root / arm / 'artifacts.json').read_text())
    summary[arm] = {key: record.get(key) for key in (
        'adapter_sample_mode', 'eventnumber', 'seed', 'best_step',
        'checkpoint', 'checkpoint_selection_status', 'selected_physics_metrics')}
(root / 'validation_comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
print('Validation comparison; these tracks were used for checkpoint selection.')
PY
for context_arm in membership; do
  echo "evaluating_$context_arm" > "$context_runs/stage.txt"
  "$context_python" "$context_dir/prepare_long_eval.py" "$context_runs" "$context_arm" "$context_checkpoint"
  "$context_python" "$context_repo/train/downstream/eval/evaluate_track_regression.py" \
    --analysis-config "$context_runs/$context_arm/analysis.yaml"
done
echo complete > "$context_runs/stage.txt"
