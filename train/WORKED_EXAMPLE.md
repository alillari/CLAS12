# One backbone, two frozen-backbone adapters

This example pretrains one center-only Mamba1 backbone, then uses the same
`ckpt_best.tar` for track finding and momentum regression. It includes target
statistics, separate final evaluation and plots. Run each section from the
repository root in the environment prepared by [SETUP](../SETUP.md).

| Stage | Configuration and budget |
| --- | --- |
| Backbone | `scale_w512_d12_chinchilla`: width 512, depth 12, state 16, k=1; actual batch 64, 200k updates |
| Track finding | Frozen backbone; width-256 attention decoder, 16 queries, zero Mamba refinement layers; 50k events, batch 32, 30k updates |
| Momentum | Same frozen backbone; width-256, one Mamba2 refinement layer, attention pooling; 50k truth tracks, batch 128, 30k updates |

The input contracts already match between the supplied recipes. The momentum
head predicts entrance `(px,py,pz)` with standardized MAE. The backbone is frozen
in both adapter fits; the momentum refinement block is trainable.

For an edgexpert launch, use [Remote jobs](REMOTE_RUNS.md) to map these paths
inside the container and preserve source, logs and artifacts across disconnects.

## 1. Set paths and select data populations

Use the following prepared CLAS12 event products:

| Root | Required contents and role |
| --- | --- |
| `CLAS12_PRETRAIN_DATA` | `features` and `seg_target` trees for both `pretrain` and `test`. Its test trees select the backbone checkpoint. |
| `CLAS12_DATA_ROOT` | Adapter training `features_pretrain`, `seg_target_pretrain`, `reg_target_pretrain`; can share the pretraining product's training events. |
| `CLAS12_VALIDATION_DATA_ROOT` | Adapter checkpoint-selection `features_test`, `seg_target_test`, `reg_target_test`. |
| `CLAS12_EVAL_DATA_ROOT` | Final evaluation test trees plus `aux_target_test` and `metadata.json` declaring `aux_target_layout` for momentum/CVT comparison. |

Keep backbone-validation, adapter-validation and final-evaluation source events
disjoint, including every segment from each event. These variables select
existing products; they do not split a dataset. Use the same coordinate/target
conventions and calibration across them. Finding's optional COATJAVA comparison
additionally requires `coatjava_seg_pred_test` in the final product.

```bash
export CLAS12_PRETRAIN_DATA=/absolute/path/to/pretraining_product
export CLAS12_DATA_ROOT="$CLAS12_PRETRAIN_DATA"
export CLAS12_VALIDATION_DATA_ROOT=/absolute/path/to/adapter_validation_product
export CLAS12_EVAL_DATA_ROOT=/absolute/path/to/final_evaluation_product
export CLAS12_ARTIFACT_ROOT=/absolute/path/to/result_deep_storage

export EXAMPLE_ROOT="$CLAS12_ARTIFACT_ROOT/backbone_to_adapters_w512"
export TRAIN_ROOT="$EXAMPLE_ROOT/adapters"
export MODEL_YAML="$TRAIN_ROOT/config/model.yaml"
export CLAS12_MOMENTUM_STATS="$TRAIN_ROOT/config/momentum_stats_train50k.json"

mkdir -p "$EXAMPLE_ROOT/config" "$TRAIN_ROOT/config"
cp scripts/configs/mamba_chinchilla_sweep.yaml "$EXAMPLE_ROOT/config/pretraining.yaml"
cp scripts/configs/clas12_adapter_recipes.yaml "$MODEL_YAML"
```

Use a new `EXAMPLE_ROOT` for a new run. Keep these environment variables set for
all commands; the copied configurations resolve input and output paths from them.

## 2. Pretrain the backbone

The unweighted recipe runs directly on the prepared data:

```bash
python -m train.pretrain \
  --yaml_config "$EXAMPLE_ROOT/config/pretraining.yaml" \
  --config scale_w512_d12_chinchilla --run_num seed42 \
  --root_dir "$EXAMPLE_ROOT/pretrain"
```

The recipe uses 1–50 clusters/event, position-only NeRF embedding, calibrated12
band-Hilbert ordering and outward-band coordinate targets. `seed42` is the run
name; the imported pretraining dataset initializes Torch's seed to 42. See
[Backbone recipes](BACKBONE_RECIPES.md) for optimizer details and seed behavior.

After pretraining returns successfully, select its best validation checkpoint:

```bash
export CLAS12_PRETRAINED_CKPT="$EXAMPLE_ROOT/pretrain/scale_w512_d12_chinchilla/seed42/training_checkpoints/ckpt_best.tar"
test -s "$CLAS12_PRETRAINED_CKPT"

python train/pretrain/plot_pretraining_history.py \
  --log "$EXAMPLE_ROOT/pretrain/scale_w512_d12_chinchilla/seed42/config_scale_w512_d12_chinchilla_run_seed42.csv" \
  --output-dir "$EXAMPLE_ROOT/plots/pretraining"
```

`pretraining_history.png/.pdf` shows raw logged loss and LR versus updates, with
the best validation observation marked. Logged training losses are sampled
batch losses; validation loss averages the squared errors of eligible target coordinates.
The LR is the logged schedule value; rank-based matrix scaling is specified
in the recipe. Do not interpret it as every parameter group's applied LR.

## 3. Generate momentum target statistics

Use only the first 50k accepted **training tracks**, with the same filters used
by the momentum profile. Track finding needs no target-standardization file.

```bash
python train/downstream/scripts/compute_regression_target_stats.py \
  --data-root "$CLAS12_DATA_ROOT" --split pretrain --task mom \
  --adapter-sample-mode event_segment --segment-target-source mctrue \
  --segment-min-clusters 12 --low-thr 1 --high-thr 100 --limit-size 50000 \
  --output "$CLAS12_MOMENTUM_STATS"

python -m json.tool "$CLAS12_MOMENTUM_STATS"
```

Expect three `mc_entrance_p*` columns, `selected_events: 50000` and per-column
`count: [50000,50000,50000]`, with finite means and positive standard deviations.
The `selected_events` field counts tracks here. This JSON defines component
standardization in the loss and is stored with the run; it is separate from
the pretraining pickle files. See [Target statistics](downstream/TARGET_STATISTICS.md).

## 4. Train both adapters on that checkpoint

The copied YAML supplies matching width-512/depth-12/state-16/k1 Mamba1 backbone
settings. Both commands use exactly `CLAS12_PRETRAINED_CKPT`; the heads retain
their independent task settings. Validation reads `CLAS12_VALIDATION_DATA_ROOT`.

```bash
python train/downstream/train_track_finding.py \
  --yaml_config "$MODEL_YAML" --config clas12_track_finding_50k \
  --usepretrain --pretrained_ckpt "$CLAS12_PRETRAINED_CKPT" \
  --run_num finding_seed42 --eventnumber 50000 --train_batch_size 32 --seed 42 \
  --checkpoint_dir "$TRAIN_ROOT/finding/checkpoints" \
  --log_file_name training.log --checkpoint_file_name adapter.pth \
  --resolved_config_path "$TRAIN_ROOT/finding/resolved.json" \
  --artifact_summary "$TRAIN_ROOT/finding/artifacts.json"

python train/downstream/train_track_regression.py \
  --yaml_config "$MODEL_YAML" --config clas12_momentum_50k \
  --usepretrain --pretrained_ckpt "$CLAS12_PRETRAINED_CKPT" \
  --run_num momentum_seed42 --eventnumber 50000 --train_batch_size 128 --seed 42 \
  --checkpoint_dir "$TRAIN_ROOT/momentum/checkpoints" \
  --log_file_name training.log --checkpoint_file_name adapter.pth \
  --resolved_config_path "$TRAIN_ROOT/momentum/resolved.json" \
  --artifact_summary "$TRAIN_ROOT/momentum/artifacts.json"
```

Each task writes `checkpoints/adapter.pth`, `checkpoints/training.log`,
`resolved.json` and `artifacts.json`. Check the actual final step (30,000),
selected checkpoint and validation counts in the receipt/log. Finding selects
option-2 signal ARI with a loss tie-break; momentum selects minimum validation
loss. These receipts also record the backbone path used by each fit.

## 5. Evaluate on the reserved final population

Copy the named analysis templates. They already select frozen-backbone mode,
the matching task profiles, checkpoint/log locations, assignment threshold 0.2
for finding, and entrance-frame momentum evaluation:

```bash
cp train/downstream/eval/track_finding_analysis_frozen_example.yaml "$TRAIN_ROOT/config/finding_analysis.yaml"
cp train/downstream/eval/momentum_analysis_frozen_example.yaml "$TRAIN_ROOT/config/momentum_analysis.yaml"

python train/downstream/eval/evaluate_track_finding.py \
  --analysis-config "$TRAIN_ROOT/config/finding_analysis.yaml"

python train/downstream/eval/evaluate_track_regression.py \
  --analysis-config "$TRAIN_ROOT/config/momentum_analysis.yaml"
```

The analysis files select `*_evaluation` model profiles, which inherit their
training architecture, filters and momentum stats while reading test data from
`CLAS12_EVAL_DATA_ROOT`. The finding limit is 10k events; momentum is 50k tracks.
Momentum evaluation initializes the shared trainer and opens the original
training trees as well as the evaluation test trees. Keep `CLAS12_DATA_ROOT`
available; change only the evaluation test root, not the training root.
Check actual accepted counts, missing baseline populations and fit statuses.
Finding's template sets `evaluate_coatjava: false`; change it to true when the
final product has the segmentation sidecar and that comparison is desired.

Analysis is the evaluator plus a YAML configuration. Campaign runners create
equivalent per-run analysis YAMLs automatically; standalone runs can copy/edit
these templates. No configuration-generation Python loop is required.

## 6. Plot and inspect behavior

Find the correct training invocation directly through its artifact receipt:

```bash
python train/downstream/campaign/plot_track_finding_validation.py \
  --artifact-summary "$TRAIN_ROOT/finding/artifacts.json" \
  --output-dir "$TRAIN_ROOT/finding/plots/validation"
```

| Output | What to inspect |
| --- | --- |
| `finding/plots/validation/metrics_vs_step.png/.pdf` | ARI, purity, efficiency, fake/split/merge and background behavior across validation steps. |
| `finding/plots/validation/background_vs_signal_ari.png/.pdf` | Whether improved signal ARI accompanies degraded purity or background rejection. |
| `finding/plots/validation/trend_summary.json` | Checkpoints and adjacent validation intervals associated with metric trade-offs. |
| `finding/evaluation/per_event_metrics.csv`, `per_truth_track_metrics.csv`, `binned_metrics.csv` | Final event/track failure patterns and kinematic dependence. Final finding evaluation writes tables; the figures above describe validation. |
| `momentum/evaluation/plots/ml/` | Automatically generated loss curves, component/kinematic errors and absolute-error CDFs. |
| `momentum/evaluation/plots/physics_2d/` | Kinematic residual dependencies and same-track adapter/CVT error comparisons. |
| `momentum/evaluation/plots/delta_p_over_p_fits/`, `delta_theta_fits/` | Per-bin occupancy, trimming, fit status and Gaussian-core description. |

Retain all model/analysis YAMLs, statistics, logs, receipts and checkpoints under
`EXAMPLE_ROOT`. Keep the environment variables/paths with the run so saved YAMLs
can be resolved again. The [evaluation reference](downstream/eval/README.md)
explains units, full-distribution metrics, display limits and fitted widths.
This is an executable recipe; it is not a report of a training run performed
while writing the example.

## 7. Expand the checked single runs into campaigns

Continue with **[From single runs to campaigns](downstream/campaign/GETTING_STARTED.md)**.
It reuses this backbone and the same named adapter profiles for a small
AdapterOnly/frozen-feature comparison, then explains label/backbone/seed sweeps,
queue restart, existing-checkpoint reuse and comparison plots. The campaign
runner does not automatically import the two adapters trained above.

For optimizer search after these runs, follow
[Tune a working momentum adapter](downstream/tuning/GETTING_STARTED.md). It
explains trial/seed budgets, diagnostics and selected-recipe campaign handoff.
