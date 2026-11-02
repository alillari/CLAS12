# Train and inspect a CLAS12 adapter

This workflow prepares target statistics, trains a track-finding or momentum
adapter, evaluates the checkpoint and identifies the plots to inspect. Complete
[environment/data setup](../SETUP.md) first. Technical explanations live in
[Training options](downstream/TRAINING_OPTIONS.md),
[Target statistics](downstream/TARGET_STATISTICS.md) and
[Evaluation and plots](downstream/eval/README.md). For Mike's Chinchilla sweep
and backbone pretraining, use [Backbone recipes](BACKBONE_RECIPES.md).
For one complete pretraining → both-adapters run, follow
[the worked example](WORKED_EXAMPLE.md).

## 1. Choose a task and prepare the run configuration

Run from the repository root with the training environment active:

```bash
export CLAS12_DATA_ROOT=/absolute/path/to/mmap_canonical_loose_truthseg_event_v6_02
export CLAS12_ARTIFACT_ROOT=/absolute/path/to/result_deep_storage
export TRAIN_ROOT="$CLAS12_ARTIFACT_ROOT/downstream_log/adapter_study"
export MODEL_YAML="$TRAIN_ROOT/config/model.yaml"
export CLAS12_MOMENTUM_STATS="$TRAIN_ROOT/config/momentum_stats_train50k.json"
mkdir -p "$TRAIN_ROOT/config"
cp scripts/configs/clas12_adapter_recipes.yaml "$MODEL_YAML"
```

Use a new `TRAIN_ROOT` for each study. Edit the copied YAML when changing a
recipe. It provides these starting profiles:

| Choice | Track finding | Momentum |
| --- | --- | --- |
| Profile | `clas12_track_finding_50k` | `clas12_momentum_50k` |
| Training samples | 50k events | 50k truth tracks |
| Train / validation batch | 32 / 32 | 128 / 128 |
| Validation samples | 5k events | 10k tracks |
| Head | Width-256 attention decoder, 16 queries, zero Mamba layers | Width-256, one Mamba2 refinement layer, attention pooling |
| Objective / selection | Class/Dice/focal loss / option-2 signal ARI | Standardized component MAE / validation loss |

Both profiles use 30k updates, validation every 1k updates, AdamW weight decay
1e-4, norm clipping 0.1, dropout 0.1, seed 42 and LR 1e-5 → 3e-4 → approximately
1e-5 with 1k warmup and one 30k-update cosine cycle.

The default commands below are **AdapterOnly** fits. To use frozen pretrained
features, set `CLAS12_PRETRAINED_CKPT` to an absolute checkpoint path and add
`--usepretrain --pretrained_ckpt "$CLAS12_PRETRAINED_CKPT"` to the training
command. Match the YAML backbone width, depth, family, block settings, `klen`
and embedding to that checkpoint before starting. The supplied profiles match
width-512/depth-12/state-16/k1 Mamba1. Keep the task's head settings when changing
the backbone; see [backbone/head options](downstream/TRAINING_OPTIONS.md#backbone-and-trainable-head).

## 2. Generate momentum target statistics

Skip this step for track finding. Momentum training standardizes each target
component using its **training-set** mean and standard deviation. This gives
components a comparable numerical scale and defines their relative weighting in
the loss. The file is also needed to construct and decode the regression head.
Coordinate normalization is handled separately by the loader.

Generate one stats file for the selected training cohort and reuse it across
models trained on those same labels:

```bash
python train/downstream/scripts/compute_regression_target_stats.py \
  --data-root "$CLAS12_DATA_ROOT" --split pretrain --task mom \
  --adapter-sample-mode event_segment --segment-target-source mctrue \
  --segment-min-clusters 12 --low-thr 1 --high-thr 100 --limit-size 50000 \
  --output "$CLAS12_MOMENTUM_STATS"

python -m json.tool "$CLAS12_MOMENTUM_STATS"
```

Check `task: mom`, the ordered `mc_entrance_px/py/pz` columns,
`selected_events: 50000` (this field counts tracks), `count: [50000,50000,50000]`,
and finite means/positive standard deviations. Match the YAML's filters and
training limit. Changing the label budget or selected tracks requires new stats;
keep validation/evaluation labels out of this calculation. For a custom/random
cohort, use its exact track IDs instead of the prefix scan. See
[Target statistics](downstream/TARGET_STATISTICS.md) for the calculation, all
command options, zero-variance behavior and checkpoint handling.

## 3. Train the selected task

Track finding:

```bash
python train/downstream/train_track_finding.py \
  --yaml_config "$MODEL_YAML" --config clas12_track_finding_50k \
  --run_num finding_seed42 --eventnumber 50000 --train_batch_size 32 --seed 42 \
  --checkpoint_dir "$TRAIN_ROOT/finding/checkpoints" \
  --log_file_name training.log --checkpoint_file_name adapter.pth \
  --resolved_config_path "$TRAIN_ROOT/finding/resolved.json" \
  --artifact_summary "$TRAIN_ROOT/finding/artifacts.json"
```

Momentum:

```bash
python train/downstream/train_track_regression.py \
  --yaml_config "$MODEL_YAML" --config clas12_momentum_50k \
  --run_num momentum_seed42 --eventnumber 50000 --train_batch_size 128 --seed 42 \
  --checkpoint_dir "$TRAIN_ROOT/momentum/checkpoints" \
  --log_file_name training.log --checkpoint_file_name adapter.pth \
  --resolved_config_path "$TRAIN_ROOT/momentum/resolved.json" \
  --artifact_summary "$TRAIN_ROOT/momentum/artifacts.json"
```

Pass the batch explicitly: the CLI default is 32 and overrides the YAML.
`--eventnumber` counts accepted tracks for momentum. For a smaller label budget,
change it and regenerate the corresponding stats; retain the independent
validation budget. `max_optimizer_steps` sets the update cap; the YAML's
`total_steps` alone does not enable step-controlled adapter training.

After completion, inspect `artifacts.json` for `final_step`, selected checkpoint
and metrics, and retain `resolved.json`, the copied model YAML, stats JSON and
matching backbone weights. These commands save `checkpoints/adapter.pth` and
`checkpoints/training.log` under the selected task directory.

## 4. Evaluate the saved checkpoint

The model YAML's `data_root_test` chooses the evaluated population. For output
behavior checks, evaluating the validation population is useful; label it as
validation. For a final test, use a separately reserved population and keep all
tracks from one source event together. The `*_evaluation` profiles select a
separate test root without changing architecture or target normalization.

Momentum evaluation needs `aux_target_test` and `metadata.json` with the declared
auxiliary layout for its CVT entrance comparison. Track-finding evaluation can
compare COATJAVA when `coatjava_seg_pred_test` is available. Training alone does
not require all these evaluation sidecars.

Analysis uses a YAML configuration and the corresponding evaluator. Campaign
runners render this file automatically. For a standalone run, copy the named
templates and choose the evaluation data root:

```bash
export CLAS12_EVAL_DATA_ROOT=/absolute/path/to/evaluation_product
cp train/downstream/eval/track_finding_analysis_frozen_example.yaml "$TRAIN_ROOT/config/finding_analysis.yaml"
cp train/downstream/eval/momentum_analysis_frozen_example.yaml "$TRAIN_ROOT/config/momentum_analysis.yaml"
```

The templates select frozen-backbone mode. For the **AdapterOnly** commands in
section 3, edit these existing fields in each copied analysis YAML:

```yaml
use_pretrained_backbone: false
pretrained_checkpoint: null
```

For frozen-backbone fits, keep `use_pretrained_backbone: true` and set
`CLAS12_PRETRAINED_CKPT` to the same checkpoint used in training. All run paths
come from `TRAIN_ROOT` and `MODEL_YAML`; no Python configuration-generation script
is needed. The templates use the copied YAML's `*_evaluation` profiles, which
retain the training model/filter/normalizer settings and set `data_root_test`
from `CLAS12_EVAL_DATA_ROOT`. If inspecting validation behavior, set that variable
to the validation product and label the results accordingly.

Keep finding's threshold/matching cuts consistent with training. Its template
has `evaluate_coatjava: false`; enable it when the sidecar is available. For a
modified recipe, change the model profile and analysis fields to match it.

```bash
python train/downstream/eval/evaluate_track_finding.py \
  --analysis-config "$TRAIN_ROOT/config/finding_analysis.yaml"

python train/downstream/eval/evaluate_track_regression.py \
  --analysis-config "$TRAIN_ROOT/config/momentum_analysis.yaml"
```

Run only the evaluator for the task you trained. `--max-samples` can reduce the
evaluation limit for a quick behavior check. Check actual sample counts and
finite-prediction/fit statuses in the resulting `summary.json` and CSV files.

## 5. Inspect output behavior

For track finding, let the plotter read the history path from the training
receipt so it selects the correct invocation:

```bash
python train/downstream/campaign/plot_track_finding_validation.py \
  --artifact-summary "$TRAIN_ROOT/finding/artifacts.json" \
  --output-dir "$TRAIN_ROOT/finding/plots/validation"
```

This writes `metrics_vs_step.png/.pdf`, `background_vs_signal_ari.png/.pdf` and
`trend_summary.json`. Check signal ARI together with purity, efficiency, fake
rate and background contamination. A better ARI can accompany a worse operating
point on another metric. Final finding evaluation writes per-event, per-track
and binned CSVs; it does not automatically create per-run PNG figures.

Momentum evaluation generates plots under `momentum/evaluation/plots/`:

| Inspect | What to check |
| --- | --- |
| `ml/training_curves.png` | Training/validation loss separation and where the best checkpoint was selected. The current plot uses epoch coordinates even for step-controlled logs. |
| `ml/ml_error_bars_components.png`, `ml/ml_error_bars_kinematics.png` | Component/direction errors, bias and tails beyond the scalar training loss. |
| `ml/absolute_error_cdf.png` | Broad improvement versus a small high-error tail. |
| `physics_2d/comparison_residual_*.png` | Momentum/angular dependencies, systematic offsets and nonuniform behavior. |
| `momentum_residuals.png` | Shape and asymmetry of `(p_pred-p_true)/p_true`; inspect tails outside the displayed range in the tables. |
| `delta_p_over_p_fits/`, `delta_theta_fits/` | Individual-bin fit status, occupancy, tails and Gaussian-core description. Inspect these before relying on fitted resolution summaries. |

[Evaluation and plots](downstream/eval/README.md) explains units, populations,
plot ranges, fit controls and the difference between core width and full error.
For multiple fits, use [campaign tools](downstream/campaign/README.md) to collate
runs and produce label-budget/backbone comparisons.
