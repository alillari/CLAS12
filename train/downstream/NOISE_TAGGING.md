# Noise tagging

CLAS12 noise supervision uses the shared
[minimum-four-cluster eligibility policy](../../docs/evaluation/primary_mc_min4_v1.md).
Clusters belonging to an MC segment with fewer than four owned clusters in the
complete event become noise-class examples (`1`); exactly four clusters remain
eligible signal. This is applied before serialization/chunking in both HDF5 and
legacy ragged adapter loaders. Physical primary/MC provenance remains unchanged.

Page 4 of the [metrics cheat sheet](../../docs/evaluation/metrics_cheat_sheet_v1.pdf)
defines the confusion matrix, retention, rejection, precision, accuracy, ROC AUC,
and average precision, with equations and interpretation alongside the finding
metrics. Noise rejection is a fraction, not a reciprocal rejection factor.

## HDF5 AdapterOnly benchmark

This branch integrates noise tagging with the published HDF5 cluster product.
The shared data support is a separate ancestor commit; the other adapter
integrations are independent sibling branches.

```bash
python -m train.downstream.benchmark_hdf5_noise \
  --data-root /absolute/path/to/cvt-clusters-h5 \
  --output-dir /absolute/path/to/new/noise-run
```

Defaults: 5,000 accepted training samples, 3,000 updates, 1,000 validation
samples and 2,000 test samples, seed 42. Cohorts use accepted prefixes. The
runner checks disjoint source-event identities, reloads the validation-selected
checkpoint and writes configurations, cohort IDs, predictions and `summary.json`.
For a smoke test add `--train-size 32 --steps 2 --validation-size 16 --test-size 16`.
The configuration is `hdf5_noise_5k` in `scripts/configs/clas12_adapter_recipes.yaml`.

Samples are whole events. The trainer forces event sampling to retain background.
Labels use `primary_segment < 0` (including contested clusters) as noise. The
COATJAVA baseline is explicitly its unassigned clusters, not a dedicated noise
prediction. Checkpoints are selected by validation loss. Evaluation reports the
default threshold and thresholds selected on validation for 95%, 99%, and
COATJAVA-matched signal retention. Test labels never select thresholds.
The ordinary `eval_point_classification.py` CLI accepts the explicit downstream
checkpoint and optional pretrained backbone. It calibrates on validation and
evaluates the HDF5 test split with those thresholds held fixed.

## Class-balanced training

Use `--class-balance inverse_frequency` for the matched 5k/3k control, or
`--config hdf5_noise_50k_balanced` for 50,000 training events and 30,000 updates
with 5,000 validation and 10,000 test events. The larger recipe uses batch 32,
AdamW with peak learning rate 3e-4, weight decay 1e-4, 1,000 warmup updates,
cosine decay to 1e-5, gradient norm clipping at 1.0, and validation every 1,000
updates. It retains the pilot's 759,310-parameter attention head and seed 42.
Both runs are AdapterOnly: no pretrained backbone features enter the head.
The point trainer currently fixes gradient clipping at 1.0 and uses head dropout
0.0; inherited generic clipping/dropout options do not control this head.

For each class c, the weight is N/(2*N_c), counted from accepted training
events. This gives equal total weight to signal and noise; it does not force
equal prediction counts. Padding is excluded. These weights are fixed during
validation and recorded in the resolved config and checkpoint. Validation loss
is pooled over clusters; per-class losses, signal retention, and noise rejection
are also logged in `checkpoints/training.log.jsonl`.

`validation_metrics.json` records calibration; `test_metrics.json` records fixed
operating points and a threshold curve. Compressed score files and per-event
prediction records allow subsequent analysis without rerunning inference. Class
weighting changes score calibration, so interpret scores through these measured
tradeoffs. For diagnosis of a previous run use `--checkpoint /path/to/checkpoint`
to restore its saved training recipe and cohort size; add `--evaluate-training`
to compare training and validation score separation.

## 2026-11-05 benchmark

These recorded runs used `hilbert_angular_grid: legacy_v1`. New fits inherit
main's `unit_v2` default; set `legacy_v1` explicitly in a copied recipe to
reproduce the historical setup. Campaign collation keeps the grids in separate
model series.

The matched 5k-event/3k-update control (same cohort, model, seed, and schedule)
increased test signal retention at threshold 0.5 from 33.70% to 87.82% after
balancing. At validation-selected 95% retention thresholds, test noise rejection
rose from 51.27% to 68.19%. Signal ROC AUC rose from 0.87796 to 0.91063.
Both checkpoints were selected at update 3,000.

The 50k-event/30k-update balanced run selected update 30,000. On 10,000 test
events (1,347,583 clusters), the results were:

| Operating point | Signal retained | Noise rejected |
| --- | ---: | ---: |
| Default threshold 0.5 | 95.77% | 94.27% |
| Validation target 95% retention | 95.22% | 94.76% |
| Validation target 99% retention | 98.92% | 83.20% |
| Validation target COATJAVA retention | 91.65% | 96.50% |
| COATJAVA unassigned baseline | 91.51% | 97.34% |

At approximately matched signal retention, the adapter's noise rejection is
0.83 percentage points lower than COATJAVA (paired test-event bootstrap 95%
interval: 0.75 to 0.91 points lower). This is a single-seed result; the interval
does not include variation from retraining. The larger run changes both data
volume and update budget, so their individual effects are not isolated. The
additional 8,000 test events beyond the exploratory pilot gave 95.82% signal
retention and 94.30% noise rejection at threshold 0.5.

a02c job IDs: `hdf5-noise-reweight-control-20261105-v1`,
`hdf5-noise-balanced-50k-30k-20261105-v1`, and
`hdf5-noise-benchmark-audit-20261105-v1`. Outputs are under
`/home/alessio/ML-work/result_deep_storage/edge_mcp/jobs/<job-id>/output`.
The immutable training source SHA256 is
`58f16d9d7cfb411c71f391fc9eee4ef4e5b556371971d7792c483748536e46fb`.
The audit checks exact control cohorts, common evaluation prefixes, checkpoint
selection, restored class weights, and completed update budgets.

## Metrics and label-count plotting

Noise evaluation reports accuracy and precision/recall for **both** classes:
signal is label 0, noise is label 1. Signal recall is the existing signal-retention
metric; noise recall is noise rejection. Precision answers how many clusters
predicted as that class actually belong to it. Accuracy uses all valid clusters;
balanced accuracy averages the two class recalls. Macro precision and recall
also average the two classes. Metrics pool cluster counts across events and do
not apply the training loss weights. Zero-denominator metrics are `null`.

These fields appear in benchmark summaries, each calibrated operating point,
and validation history. For HDF5 noise tagging, the ordinary
`eval_point_classification.py` CLI writes the same complete report as the benchmark.
Legacy non-HDF5 inference retains its `<log>.json` metric report.

For future learning curves, copy `scripts/configs/noise_tagging_campaign.example.yaml`
and list each run's result directory. A completed directory must contain
`summary.json`, `resolved_config.json`, and `cohorts.json` from either the noise benchmark or ordinary HDF5 noise evaluator.
The labeled-event count comes from the actual accepted training cohort. Model IDs
distinguish model series; repeat a model ID across label budgets and seeds.

```bash
python -m train.downstream.campaign.collate_noise_tagging_campaign \
  --manifest /absolute/path/to/noise_campaign.yaml \
  --output-dir /absolute/path/to/campaign/summary

python -m train.downstream.campaign.plot_noise_tagging_campaign \
  --run-table /absolute/path/to/campaign/summary/run_table.json \
  --output-dir /absolute/path/to/campaign/plots \
  --operating-point default
```

The plotter also accepts `--manifest` directly. It writes a six-metric overview,
individual metric plots, and 1x3/3x1 accuracy–precision–recall presentation panels
for signal and noise, in PNG and PDF. X is labeled training events on a log scale;
y is the classification metric. COATJAVA's unassigned-cluster baseline is dashed.
Use `--operating-point retain_95pct`, `retain_99pct`, or
`match_coatjava_retention` for the other validation-calibrated views.

Different update budgets, recorded optimization schedules, data contracts, or
validation/test event identities produce separate comparison directories. The
JSON table retains the full comparison metadata. Different models, input settings,
backbone checkpoint fingerprints, and class-balance policies remain distinct curves within a comparison. Repeated
seeds use equal-weight seed means and sample-standard-deviation bars; duplicate
model/budget/seed records are rejected. If any seed's metric is undefined, that
budget is a plot gap. COATJAVA counts must agree on identical evaluation cohorts;
the baseline is never averaged across different populations. Unfinished runs are
listed as skipped. These commands only read results and generate reports; they
do not launch training or alter checkpoints.


## Ordinary training and evaluation

The point-classification CLI honors the YAML task, label count, batch size, seed
and embedder depth unless explicitly overridden. `--run_num` names a run; use
`--seed` to change initialization. Give each fit its own YAML `checkpoint_dir`;
`--root_dir` only redirects runtime logs. The public CLI starts fresh training.

```bash
export CLAS12_DATA_ROOT=/absolute/path/to/cvt-clusters-h5
export CLAS12_ARTIFACT_ROOT=/absolute/path/to/new/noise-training

python -m train.downstream.train_point_classification \
  --yaml_config scripts/configs/clas12_adapter_recipes.yaml \
  --config hdf5_noise_50k_balanced --seed 42 \
  --root_dir "$CLAS12_ARTIFACT_ROOT/runtime"

python -m train.downstream.eval_point_classification \
  --yaml_config scripts/configs/clas12_adapter_recipes.yaml \
  --config hdf5_noise_50k_balanced \
  --checkpoint "$CLAS12_ARTIFACT_ROOT/downstream_log/hdf5_noise_50k_balanced/checkpoints/hdf5_noise_50k_balanced_nid_0_checkpoint.pth" \
  --root_dir /absolute/path/to/new/noise-evaluation \
  --eventnumber 10000 --eval_batch_size 32
```

For frozen features, pass `--pretrained_ckpt /absolute/path/to/backbone.tar` to
**both** commands and use a YAML architecture matching that backbone. The training
flag implies frozen mode; `--usepretrain` is an optional assertion that a backbone
was supplied. Evaluation requires the explicit path even if training saved one.
The backbone has gradients disabled and stays in evaluation mode. For historical
backbone checkpoints without a grid marker, set `hilbert_angular_grid: legacy_v1`
in the training YAML; the trainer rejects a mismatched grid.

New checkpoints store the input mode, SHA256 of the frozen backbone, resolved
training configuration, Hilbert grid, data contract, and training-cohort digest. Loading rejects
mode changes, different backbone bytes, incompatible head state, a changed training
product, or a different reconstructed cohort. Relocating identical backbone bytes
is supported. Historical AdapterOnly checkpoints with explicit saved mode are
supported; historical frozen heads without a fingerprint are rejected because the
required backbone cannot be verified. Older HDF5 checkpoints lack a cohort digest;
the report marks that provenance limitation.

HDF5 evaluation restores architecture, preprocessing, training seed and cohort
selection from the adapter checkpoint, including `legacy_v1` for historical
checkpoints without a grid marker. YAML supplies current data paths and worker
count; `--eval_batch_size`, `--eventnumber` (test events) and optional
`--validation_events` control evaluation only. `resolved_config.json/.yaml` retain
training settings; `evaluation_config.json` records evaluation overrides.
`cohorts.json`, split score/prediction/metric files and `summary.json` feed the
campaign tools directly. Completed evaluations are not overwritten.

The training directory also receives `training_run.json` with completed updates
and a checkpoint digest. Keep it alongside the checkpoint: evaluation reports its
actual completed budget when the digest matches. Without a matching receipt,
completed updates are unknown and campaign grouping uses the saved requested
budget; the selected checkpoint step is always reported separately.

Noise tagging supports **one process on one GPU (or CPU)**. Multi-process launches
fail before distributed/CUDA initialization. Adapter resume, distributed training,
and full-backbone fine-tuning are outside this implementation.

For a bounded real-data integration check with a width-512/depth-12 Mamba1
checkpoint matching the supplied recipe:

```bash
python -m train.downstream.tests.smoke_noise_workflow \
  --data-root "$CLAS12_DATA_ROOT" \
  --backbone /absolute/path/to/pretrained/ckpt_best.tar \
  --output-dir /absolute/path/to/new/noise-smoke
```

This runs both ordinary CLI workflows on 32 training, 16 validation and 18 test
events, for two updates each. It checks frozen weights/gradients, reloads, partial
batches, provenance and campaign collation. AdapterOnly uses the current default
grid; frozen mode follows the supplied backbone's grid metadata (historical
unmarked backbones use `legacy_v1`). It makes no performance claim and does
not generate a learning curve.

## Workflow validation (2026-11-05)

After integrating current `main`, including its versioned Hilbert grid:

- `hdf5-noise-merge-review-suite-20261105-v1`: downstream tests plus the HDF5
  pretraining and Hilbert-grid suites ran 261 tests: 255 passed, six skipped,
  no failures or errors. Five skips require CUDA and one requires an optional
  real-data pretraining fixture. The earlier pretraining-fixture error is fixed
  on current `main`.
- `hdf5-noise-merge-cli-smoke-20261105-v1`: both ordinary CLI modes passed GPU
  training, reload, validation calibration, COATJAVA export and campaign
  collation. AdapterOnly used `unit_v2`; frozen features used the historical
  backbone's `legacy_v1`. Frozen backbone tensors were unchanged and had no
  gradients. Both runs retained the final partial test batch and training config.
- `hdf5-noise-merge-legacy-regression-20261105-v1`: both evaluation entrypoints
  restored the existing balanced 50k/30k checkpoint's `legacy_v1` grid and
  reproduced its original scores exactly on 32 events / 4,411 clusters with
  matched batch size 32. Its saved label/update budgets were retained.

These are integration checks. The performance evidence remains the earlier
balanced AdapterOnly benchmark above.
