# Track-Regression Campaign Runner

This directory contains local campaign tools for training CLAS12 track-regression
adapters on a folder of pretrained Mamba backbones and then running the physics
evaluation for each adapter.

For one-output `p`, `theta`, and `phi` ablations, see
[Single-target regression](../SINGLE_TARGET_REGRESSION.md) for loss definitions,
config presets, matched statistics generation, and evaluation behavior.

The runner is intentionally local-first. It does not submit SLURM jobs. It runs
one training/evaluation job at a time and sets `CUDA_VISIBLE_DEVICES` for each
subprocess.

## Expected Backbone Layout

The default pretrained root is:

```text
/home/alessio/ML-work/pretrained-FMs/campaign_1/
```

The manifest builder expects one subdirectory per pretrained run:

```text
campaign_1/
  scale_w1536_d12_n5483352/
    <one checkpoint file, often with .tar extension>
  scale_w128_d12_n54834/
    <one checkpoint file>
```

The `.tar` files are treated as PyTorch checkpoint files, not as archives.

For the current campaign, metadata is parsed from the directory name:

- `w1536` -> `base_dim: 1536`, `embed_dim: 1536`
- `d12` -> `num_layers_backbone: 12`
- `n5483352` -> `pretrain_events: 5483352`

If a backbone directory later contains `metadata.yaml` or `metadata.json`, that
sidecar is preferred over filename-derived metadata.

## Build A Manifest

From the repository root:

```bash
python train/downstream/campaign/build_track_regression_manifest.py \
  --checkpoint-root /home/alessio/ML-work/pretrained-FMs/campaign_1 \
  --campaign-name campaign_1_track_regression \
  --artifact-root /home/alessio/ML-work/result_deep_storage
```

This writes:

```text
/home/alessio/ML-work/result_deep_storage/campaigns/campaign_1_track_regression/manifest.yaml
```

Useful smaller test manifest settings:

```bash
python train/downstream/campaign/build_track_regression_manifest.py \
  --checkpoint-root /home/alessio/ML-work/pretrained-FMs/campaign_1 \
  --campaign-name campaign_1_track_regression_smoke \
  --artifact-root /home/alessio/ML-work/result_deep_storage \
  --eventnumber 100 \
  --train-batch-size 8 \
  --max-samples 100
```

Build a matrix with four labeled-data amounts for every backbone:

```bash
python train/downstream/campaign/build_track_regression_manifest.py \
  --checkpoint-root /home/alessio/ML-work/pretrained-FMs/campaign_1 \
  --campaign-name campaign_1_track_regression_label_sweep \
  --artifact-root /home/alessio/ML-work/result_deep_storage \
  --eventnumber 100,1000,10000,70000
```

With multiple `--eventnumber` values, each backbone is expanded into one
adapter run per labeled-data amount. For example:

```text
adapteronly_label100
adapteronly_label1000
adapteronly_label10000
adapteronly_label70000
scale_w1536_d12_n5483352_label100
scale_w1536_d12_n5483352_label1000
scale_w1536_d12_n5483352_label10000
scale_w1536_d12_n5483352_label70000
```

Adapter-only baselines are included by default, one per labeled-data amount.
They use `scripts/configs/mamba_clas12_track_regression_adapteronly.yaml` and
omit `--usepretrain`.

## Build An Adapter-Only Campaign With No Backbones

To run only adapter-only baselines over a broad labeled-data range, use an empty
checkpoint root and pass `--allow-empty`. The checkpoint root must exist, but it
does not need to contain any pretrained backbone directories.

From the repository root:

```bash
mkdir -p /tmp/no_pretrained_backbones

python train/downstream/campaign/build_track_regression_manifest.py \
  --checkpoint-root /tmp/no_pretrained_backbones \
  --campaign-name adapteronly_label_sweep \
  --artifact-root /home/alessio/ML-work/result_deep_storage \
  --eventnumber 100,1000,10000,50000,100000,200000 \
  --allow-empty
```

This writes an adapter-only manifest:

```text
/home/alessio/ML-work/result_deep_storage/campaigns/adapteronly_label_sweep/manifest.yaml
```

The generated runs have IDs like:

```text
adapteronly_label100
adapteronly_label1000
adapteronly_label10000
adapteronly_label50000
adapteronly_label100000
adapteronly_label200000
```

Run the campaign normally:

```bash
python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest /home/alessio/ML-work/result_deep_storage/campaigns/adapteronly_label_sweep/manifest.yaml \
  --cuda-device 0
```

Adapter-only runs set `use_pretrained_backbone: false`, use
`scripts/configs/mamba_clas12_track_regression_adapteronly.yaml`, and the
training command omits `--usepretrain`. No pretrained checkpoint is loaded or
validated for these rows.

## Build An Optuna Seed-Ablation Manifest

To retest a broad sample of completed Optuna trials across several training
seeds:

```bash
python train/downstream/tuning/build_track_regression_seed_ablation.py \
  --storage sqlite:////path/to/study.db \
  --study-name clas12_adapteronly_tuning \
  --ablation-name clas12_adapteronly_seed_ablation \
  --output-root /home/alessio/ML-work/result_deep_storage \
  --seeds 11,17,23,31,43 \
  --top-k 5 \
  --quantile-k 10 \
  --random-k 0
```

The builder uses completed trials only. It selects the top trials by objective
value, evenly spaced quantile trials across the full completed-trial ranking,
and optional random extra trials controlled by `--sample-seed`. Each selected
trial is expanded once per seed with run IDs like `trial_000123_seed_17`.

Run the generated manifest with the normal campaign runner:

```bash
python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest /home/alessio/ML-work/result_deep_storage/campaigns/clas12_adapteronly_seed_ablation/manifest.yaml \
  --cuda-device 0
```

For seed-ablation campaigns, collation also writes:

```text
summary/seed_ablation_runs.csv
summary/seed_ablation_trials.csv
```

To build a manifest without adapter-only baselines:

```bash
python train/downstream/campaign/build_track_regression_manifest.py \
  --checkpoint-root /home/alessio/ML-work/pretrained-FMs/campaign_1 \
  --campaign-name campaign_1_track_regression_label_sweep \
  --artifact-root /home/alessio/ML-work/result_deep_storage \
  --eventnumber 100,1000,10000,70000 \
  --no-adapter-only
```

## Use The Best Optuna Fine-Tuning Recipe

To compare pretrained backbones against the adapter-only baseline using the best
fine-tuning hyperparameters from an AdapterOnly Optuna study:

```bash
python train/downstream/campaign/build_track_regression_manifest.py \
  --checkpoint-root /home/alessio/ML-work/pretrained-FMs/campaign_1 \
  --campaign-name campaign_1_track_regression_best_optuna_recipe \
  --artifact-root /home/alessio/ML-work/result_deep_storage \
  --eventnumber 50000 \
  --optuna-best-trial \
  --optuna-storage sqlite:////home/alessio/ML-work/result_deep_storage/optuna/adapteronly_rangefind_50k.db \
  --optuna-study-name adapteronly_rangefind_50k
```

This is the recommended way to reuse a good tuning run for a normal campaign
when you want one shared fine-tuning recipe across adapter-only and pretrained
backbone rows. The manifest builder loads `study.best_trial` from the Optuna
storage, records its provenance under `source_optuna`, and writes the selected
hyperparameters into campaign-level `training_overrides`.

The tuned scalar recipe and the recorded execution contract are imported:

- `max_lr`
- `min_lr_ratio`
- derived `min_lr`
- `warmup_fraction`
- `adapter_weight_decay`
- `grad_clip_value`
- `dropout`

The execution contract also ports `max_optimizer_steps`, validation and
early-stopping settings, `max_val_batches`, warmup, and the cosine schedule.
The schedule records either an equal-cycle `n_cycles` policy or an explicit
`scheduler_first_cycle_steps` length (needed for a final partial cycle such as
7,000 steps in a 20,000-step budget). Training label count, batch size, data
paths, and checkpoint selection remain explicit campaign choices and are
recorded alongside the Optuna provenance.

Explicit `--training-override KEY=VALUE` arguments take precedence over the
imported recipe, so you can import the best recipe and still override one value:

```bash
python train/downstream/campaign/build_track_regression_manifest.py \
  --checkpoint-root /home/alessio/ML-work/pretrained-FMs/campaign_1 \
  --campaign-name campaign_1_track_regression_best_optuna_recipe_dropout12 \
  --artifact-root /home/alessio/ML-work/result_deep_storage \
  --eventnumber 50000 \
  --optuna-best-trial \
  --optuna-storage sqlite:////home/alessio/ML-work/result_deep_storage/optuna/adapteronly_rangefind_50k.db \
  --optuna-study-name adapteronly_rangefind_50k \
  --training-override dropout=0.12
```

Use a tuning study whose fixed settings match the campaign you want to run, or
intentionally decide which differences are acceptable. In particular, changing
the labeled-data amount, batch size, training-step budget, validation cadence, or
dataset split can make the "best" trial less directly transferable.

## Training-Time Controls

Training parameters are inherited from
`scripts/configs/mamba_clas12_track_regression_pretrained.yaml`, but the
manifest builder can override them for the whole campaign.

Common time-control options:

- `--max-epochs`
- `--early-stopping-patience`
- `--early-stopping-warmup-steps`
- `--max-train-batches`
- `--max-val-batches`
- `--training-override KEY=VALUE`

For optimizer and regularization settings, prefer `--optuna-best-trial` when
the values should come from a completed tuning run. Use manual
`--training-override KEY=VALUE` for one-off edits or for parameters that were
not part of the Optuna search space.

`--max-train-batches` and `--max-val-batches` cap the number of batches used per
epoch. They are useful for quick exploratory sweeps because they bound the time
spent in each epoch regardless of the labeled-data amount.

Example quick exploratory manifest:

```bash
python train/downstream/campaign/build_track_regression_manifest.py \
  --checkpoint-root /home/alessio/ML-work/pretrained-FMs/campaign_1 \
  --campaign-name campaign_1_track_regression_quick \
  --artifact-root /home/alessio/ML-work/result_deep_storage \
  --eventnumber 100,1000,10000,70000 \
  --max-epochs 8 \
  --early-stopping-patience 2 \
  --early-stopping-warmup-steps 1 \
  --max-train-batches 50 \
  --max-val-batches 100 \
  --max-samples 1000
```

Example fuller manifest:

```bash
python train/downstream/campaign/build_track_regression_manifest.py \
  --checkpoint-root /home/alessio/ML-work/pretrained-FMs/campaign_1 \
  --campaign-name campaign_1_track_regression_full \
  --artifact-root /home/alessio/ML-work/result_deep_storage \
  --eventnumber 100,1000,10000,70000 \
  --max-epochs 300 \
  --early-stopping-patience 20 \
  --early-stopping-warmup-steps 50 \
  --max-samples 500000
```

`500000` is also the default when `--max-samples` is omitted. The evaluator
uses this value for both the test-view cap and the number of predictions, so it
does not inherit the usually smaller training `limit_size`. The final partial
batch is retained. Use a smaller explicit value for smoke or exploratory runs.

Any additional model YAML key can be overridden:

```bash
python train/downstream/campaign/build_track_regression_manifest.py \
  --checkpoint-root /home/alessio/ML-work/pretrained-FMs/campaign_1 \
  --campaign-name campaign_1_track_regression_custom \
  --artifact-root /home/alessio/ML-work/result_deep_storage \
  --eventnumber 100,1000,10000,70000 \
  --training-override max_lr=0.0001 \
  --training-override warmup_steps=50 \
  --training-override total_steps=2000
```

## Dry Run

Check the selected runs and commands without writing per-run configs or running
training:

```bash
python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest /home/alessio/ML-work/result_deep_storage/campaigns/campaign_1_track_regression/manifest.yaml \
  --cuda-device 0 \
  --dry-run
```

Run a dry-run for a single backbone:

```bash
python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest /home/alessio/ML-work/result_deep_storage/campaigns/campaign_1_track_regression/manifest.yaml \
  --only scale_w1536_d12_n5483352 \
  --cuda-device 0 \
  --dry-run
```

## Run The Campaign

Run all manifest entries sequentially on local GPU 0:

```bash
python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest /home/alessio/ML-work/result_deep_storage/campaigns/campaign_1_track_regression/manifest.yaml \
  --cuda-device 0
```

Run one entry:

```bash
python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest /home/alessio/ML-work/result_deep_storage/campaigns/campaign_1_track_regression/manifest.yaml \
  --only scale_w1536_d12_n5483352 \
  --cuda-device 0
```

Run the first `N` entries in manifest order:

```bash
python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest /home/alessio/ML-work/result_deep_storage/campaigns/campaign_1_track_regression/manifest.yaml \
  --limit 3 \
  --cuda-device 0
```

Train without evaluating:

```bash
python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest /home/alessio/ML-work/result_deep_storage/campaigns/campaign_1_track_regression/manifest.yaml \
  --skip-eval \
  --cuda-device 0
```

## Check Progress

Print a campaign progress table:

```bash
python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest /home/alessio/ML-work/result_deep_storage/campaigns/campaign_1_track_regression/manifest.yaml \
  --status
```

The status table reports the resumable stage for each run and whether the
expected adapter checkpoint and evaluation summary exist.

Watch the active training log for a run:

```bash
tail -f /home/alessio/ML-work/result_deep_storage/campaigns/campaign_1_track_regression/logs/<run_id>.train.stdout.log
```

Watch the active evaluation log for a run:

```bash
tail -f /home/alessio/ML-work/result_deep_storage/campaigns/campaign_1_track_regression/logs/<run_id>.eval.stdout.log
```

## Outputs

Campaign outputs are written outside the repo:

```text
<artifact_root>/campaigns/<campaign_name>/
  manifest.yaml
  status.yaml
  logs/
    <run_id>.train.stdout.log
    <run_id>.eval.stdout.log
  runs/
    <run_id>/
      config/
        model.yaml
        analysis.yaml
      train/
        artifacts.json
      checkpoints/
        <run_id>.log
        <run_id>_adapter_checkpoint.pth
      evaluation/
        summary.json
        ml_metrics.csv
        binned_metrics.csv
        delta_p_over_p_fits.csv
        delta_theta_fits.csv
        campaign_headline_metrics.jsonl
        plots/
          delta_p_over_p_fits/
          delta_theta_fits/
  summary/
    campaign_headline_metrics.jsonl
    run_table.csv
    delta_p_over_p_fits.csv
    delta_theta_fits.csv
```

`status.yaml` records resumable stage state for each run. The main statuses are:

- `pending`
- `running_train`
- `train_done`
- `running_eval`
- `eval_done`
- `failed`

## Resume And Force Options

The runner is resumable. If a run has `train_done` or `eval_done` and the
expected output exists, that stage is skipped.

Force retraining:

```bash
python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest <manifest.yaml> \
  --only <run_id> \
  --force-train \
  --cuda-device 0
```

Force reevaluation:

```bash
python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest <manifest.yaml> \
  --only <run_id> \
  --force-eval \
  --cuda-device 0
```

Rebuild only campaign-level summary files from existing evaluations:

```bash
python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest <manifest.yaml> \
  --collate-only
```

## Plot Campaign Scaling

After evaluation, collate campaign-level metrics:

```bash
python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest /home/alessio/ML-work/result_deep_storage/campaigns/campaign_1_track_regression_label_sweep/manifest.yaml \
  --collate-only
```

Then make scaling plots:

```bash
python train/downstream/campaign/plot_track_regression_campaign.py \
  --campaign-dir /home/alessio/ML-work/result_deep_storage/campaigns/campaign_1_track_regression_label_sweep
```

The plotting script has five plot-suite choices:

```bash
--plot-suite standard              # existing MAE/RMSE/R2 scaling plots
--plot-suite momentum-resolution   # fitted delta-p/p momentum-resolution plots
--plot-suite presentation          # slide-ready physical error, resolution and bias panels
--plot-suite presentation-ml       # only slide-ready MAE/RMSE/R2 metrics
--plot-suite all                   # all suites, including presentation exports
```

### Slide-ready momentum figures

The presentation suite uses the track-finding figures' cream/navy/rust palette,
with vector PDF and 300-dpi PNG exports, horizontal and vertical combined panels,
and standalone panels under `presentation/individual/`. It reads stored summaries
only: it does not retrain, load checkpoints, rerun evaluation, or refit residuals.

`presentation` also includes a `presentation/ml_metrics/` figure set. Use
`--plot-suite presentation-ml` to generate only this set:

- `regression_metrics_*_{1x3,3x1}` compares mean component MAE, pooled component
  RMSE and mean component R² across label budgets, with COATJAVA as the reference.
- `{mae,rmse,r2}_components_*` shows px, py and pz separately, with corresponding
  standalone panels under `ml_metrics/individual/`.
- `rmse_kinematic_*` adds p/theta/wrapped-phi RMSE panels. These quantities have
  no stored R² in the campaign summaries; their R² is not inferred or fabricated.
- `individual/r2_cartesian_10k_plus_*` explicitly zooms the high-label regime;
  the main R² figures still show all budgets and preserve negative values.

The aggregate MAE and R² are arithmetic means of the three Cartesian component
values. Pooled RMSE is `sqrt((RMSE_px² + RMSE_py² + RMSE_pz²) / 3)`, requiring
equal component sample counts. It is not the arithmetic mean of component RMSEs
or the per-track 3D vector-error RMSE. CSV exports and a separate provenance file
record all values and definitions. Existing `standard` plots are unchanged.

```bash
PY=/home/alessio/miniconda3/envs/fm4npp/bin/python
export MPLCONFIGDIR=/tmp/matplotlib-cache
CAMPAIGN_DIR=/path/to/completed/campaign
"$PY" train/downstream/campaign/plot_track_regression_campaign.py \
  --campaign-dir "$CAMPAIGN_DIR" \
  --plot-suite presentation
```

Use `--output-dir /path/to/local/plots` when reading a mounted, read-only campaign.
The suite writes into `<output-dir>/presentation/`. Optional
`--presentation-labels 100000` selects a common training-label budget for the
resolution and bias figures; by default it uses the largest budget available for
every evaluated model, without choosing by performance.

- `momentum_scaling_*_{1x3,3x1}`: physical MAE in momentum, theta and wrapped phi
  versus labeled events, with logarithmic axes for positive errors. Single-budget
  campaigns instead produce categorical `momentum_errors_*` comparisons.
- `momentum_resolution_*_{1x2,2x1}` and `momentum_bias_*_{1x2,2x1}`: fitted width
  and mean of delta-p/p (percent) and delta-theta (degrees), versus true momentum.
  A campaign with only one fit table produces single-panel outputs.
- `all_models`, `focus`, `other_pretrained_lines` and
  `other_pretrained_envelope` variants are produced where applicable. Focus uses
  adapter-only, the largest available pretrained backbone, and COATJAVA.
  The envelope is a min/max range across other backbones, **not an uncertainty
  interval**. Fixed-budget categorical comparisons do not draw envelopes.
- `model_labels.csv`, `selected_runs.csv`, `*_plot_data.csv`, and
  `provenance.json` retain source identities, sample counts and source checksums.
  `m1`, `m2`, etc. are assigned by width/depth/source ID within the campaign;
  consult the mapping before comparing labels across campaigns.

COATJAVA denotes the stored `cvt` method (`CVT::Tracks`). Identical repeated
baseline results are shown once; differing baseline values/counts cause an error
rather than being averaged. Duplicate model/budget runs also require an explicit
subset. Physical quantities are taken from `ml_error / kinematic`, never averaged
across mixed-unit native losses. Only `fit_status=ok` bins enter resolution/bias
curves; failed and sparse bins leave gaps. Error bars are stored fit-parameter
errors, not residual widths or seed uncertainties. Missing fit tables are reported
and skipped, while physical MAE figures remain available.

For campaigns evaluated before `delta_p_over_p_fits.csv` or
`delta_theta_fits.csv` existed, rerun evaluation before making the corresponding
resolution plots:

```bash
python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest /home/alessio/ML-work/result_deep_storage/campaigns/campaign_1_track_regression_label_sweep/manifest.yaml \
  --force-eval \
  --cuda-device 0
```

The plotting script reads:

```text
<campaign_dir>/summary/campaign_headline_metrics.jsonl
<campaign_dir>/summary/delta_p_over_p_fits.csv
<campaign_dir>/summary/delta_theta_fits.csv
<campaign_dir>/manifest.yaml
```

and writes:

```text
<campaign_dir>/summary/plots/
  plot_data.csv
  best_by_slice.csv
  mae_mean_vs_labeled_events_by_width.png
  mae_mean_vs_backbone_params_by_labeled_events.png
  mae_mean_vs_pretrain_events_by_labeled_events.png
  rmse_mean_vs_labeled_events_by_width.png
  rmse_mean_vs_backbone_params_by_labeled_events.png
  rmse_mean_vs_pretrain_events_by_labeled_events.png
  r2_mean_vs_labeled_events_by_width.png
  r2_mean_vs_backbone_params_by_labeled_events.png
  r2_mean_vs_pretrain_events_by_labeled_events.png
  mae_<component>_vs_*.png
  rmse_<component>_vs_*.png
  r2_<component>_vs_*.png
  momentum_resolution/
    presentation_sigma_delta_p_over_p.png
    presentation_sigma_delta_theta.png
    delta_p_over_p_conventional.png
    delta_p_over_p_conventional_mean.png
    delta_p_over_p_conventional_sigma.png
    delta_theta_conventional.png
    delta_theta_conventional_mean.png
    delta_theta_conventional_sigma.png
    delta_p_over_p_adapter_only.png
    delta_p_over_p_adapter_only_mean.png
    delta_p_over_p_adapter_only_sigma.png
    delta_theta_adapter_only.png
    delta_theta_adapter_only_mean.png
    delta_theta_adapter_only_sigma.png
    delta_p_over_p_pretrained_adapter.png
    delta_p_over_p_pretrained_adapter_mean.png
    delta_p_over_p_pretrained_adapter_sigma.png
    delta_theta_pretrained_adapter.png
    delta_theta_pretrained_adapter_mean.png
    delta_theta_pretrained_adapter_sigma.png
    delta_p_over_p_all.png
    delta_p_over_p_all_mean.png
    delta_p_over_p_all_sigma.png
    delta_theta_all.png
    delta_theta_all_mean.png
    delta_theta_all_sigma.png
    delta_p_over_p_conventional_adapter_only.png
    delta_p_over_p_conventional_adapter_only_mean.png
    delta_p_over_p_conventional_adapter_only_sigma.png
    delta_theta_conventional_adapter_only.png
    delta_theta_conventional_adapter_only_mean.png
    delta_theta_conventional_adapter_only_sigma.png
    delta_p_over_p_conventional_pretrained_adapter.png
    delta_p_over_p_conventional_pretrained_adapter_mean.png
    delta_p_over_p_conventional_pretrained_adapter_sigma.png
    delta_theta_conventional_pretrained_adapter.png
    delta_theta_conventional_pretrained_adapter_mean.png
    delta_theta_conventional_pretrained_adapter_sigma.png
    runs/<run_id>/
      delta_p_over_p.png
      delta_p_over_p_mean.png
      delta_p_over_p_sigma.png
      delta_theta.png
      delta_theta_mean.png
      delta_theta_sigma.png
```

The three main campaign axes are:

- backbone model size: `backbone_n_params`, computed by instantiating the same
  `Mamba1GPT` backbone used by training and counting trainable parameters;
- backbone pretraining data size: parsed from the `n<int>` token;
- adapter labeled-data size: parsed from the `_label<int>` suffix or read from
  the manifest.

The script avoids 3D plots. Instead it makes trace families and small multiples,
so each figure varies two axes while holding the third axis in panels or traces.
If the model class cannot be imported in a lightweight environment, the plots
fall back to using `embed_dim` instead of `backbone_n_params`.

The campaign plots intentionally use only adapter regression metrics. COATJAVA
or CVT reconstruction comparison rows are ignored here. The metrics are:

- MAE for `px`, `py`, and `pz`;
- RMSE for `px`, `py`, and `pz`;
- R2 for `px`, `py`, and `pz`;
- mean MAE averaged across `px`, `py`, and `pz`;
- mean RMSE averaged across `px`, `py`, and `pz`;
- mean R2 averaged across `px`, `py`, and `pz`.

Momentum-resolution plots use Gaussian fits to `(p_reco - p_true) / p_true` in
configured true-momentum bins. The plotted mean is the fitted bias and the
plotted sigma is the fitted standard deviation, shown as a percentage. Sparse
bins are skipped during evaluation and recorded in `delta_p_over_p_fits.csv`
with a `fit_status`.

Polar-angle resolution plots use the same fit machinery and true-momentum bins
for `theta_reco - theta_true`. The fitted mean and sigma are shown in degrees
and recorded in `delta_theta_fits.csv`.

Every run also writes one histogram-and-fit diagnostic per method and momentum
bin beneath `evaluation/plots/delta_p_over_p_fits/` and
`evaluation/plots/delta_theta_fits/`. These include skipped and failed fits as
well as successful ones; campaign-level summary plots continue to consume the
CSV fit values without changing their selection or aggregation.

The default delta-p/p binning is limited to the region with useful
coverage:

```yaml
delta_p_over_p_bins_gev: [0.25, 0.5, ..., 3.0]
delta_p_over_p_min_bin_entries: 200
delta_p_over_p_min_populated_histogram_bins: 8
delta_p_over_p_histogram_bins: 40
delta_p_over_p_fit_quantile: 0.98
delta_theta_bins_gev: [0.25, 0.5, ..., 3.0]
delta_theta_min_bin_entries: 200
delta_theta_min_populated_histogram_bins: 8
delta_theta_histogram_bins: 40
delta_theta_fit_quantile: 0.98
```

The minimum-entry and populated-histogram-bin cuts are deliberately stricter
than the broad binned metrics because narrow momentum slices need enough
statistics for stable Gaussian fits.

The presentation plot
`momentum_resolution/presentation_sigma_delta_p_over_p.png` shows only fitted
sigma versus `p` for the largest AdapterOnly run when present, compared with the
matching conventional CVT reconstruction. Its legend reports the selected
adapter's labeled training-track count, for example `AdapterOnly, 50k tracks`.
`momentum_resolution/presentation_sigma_delta_theta.png` uses the same selected
run and baseline, with `σ(Δθ)` in degrees.

## Troubleshooting

If manifest building fails with a parse error, check that the backbone directory
name contains `w<int>`, `d<int>`, and `n<int>` tokens.

If training fails immediately with a checkpoint mismatch, check the rendered
per-run `config/model.yaml`. The parsed `embed_dim` and `num_layers_backbone`
must match the pretrained backbone.

If matplotlib or fontconfig cache warnings appear, they are usually harmless.
The runner sets `MPLCONFIGDIR` under the campaign directory for subprocesses.

If a run fails, inspect:

```text
<campaign_dir>/logs/<run_id>.train.stdout.log
<campaign_dir>/logs/<run_id>.eval.stdout.log
```

## Single-target campaigns with macro checkpoint selection

Select `clas12_track_regression_adapteronly_p_only`, `_theta_only`, or `_phi_only`
with `--adapter-only-model-config`; pretrained equivalents use
`--pretrained-model-config clas12_track_regression_pretrained_<target>_only`.
Generate the matching statistics files on the training machine first.
Use a distinct campaign name for each target when generating separate manifests.

Validation now defaults to a fixed **50,000 accepted tracks/event-segments**, with
`valid_batch_size: 128`, independent of both the training label budget and training
batch size. It still reads `data_root_test`; a separate validation folder/split is
deferred. `--max-samples` controls subsequent evaluation, not training validation.

```bash
/home/alessio/miniconda3/envs/fm4npp/bin/python \
  train/downstream/campaign/build_track_regression_manifest.py \
  --adapter-only-only \
  --campaign-name p_only_macro \
  --artifact-root /path/to/artifacts \
  --adapter-only-model-config clas12_track_regression_adapteronly_p_only \
  --eventnumber 1000,10000,100000 \
  --validation-samples 50000 \
  --validation-batch-size 128 \
  --training-override 'physics_checkpoint.tail_threshold=0.10' \
  --training-override 'physics_checkpoint.guardrails={B_macro: null, B_worst: null, T_macro: null}'
```

`--validation-samples` sets `limit_test_size`; `--validation-batch-size` sets
`valid_batch_size`. Both can also be set in the base model YAML or manifest's
`training_overrides`. `limit_test_data: false` explicitly requests the entire
accepted test dataset. Defaults use `max_val_batches: null`; an explicit batch
cap that would truncate the requested validation sample is rejected. Set the
sample limit itself when a smaller validation budget is intended.

Before the first optimizer step, training reads the validation truth once and
writes `validation_support.json` alongside its checkpoint history. This reports
requested/actual counts, every truth-bin occupancy, invalid/out-of-range truth,
and a hash of the ordered truth sample. If too few bins pass the configured
occupancy requirement, training stops with the report path. Increase validation
statistics or explicitly revise the truth range/binning and minimum valid bins;
there is no automatic relaxation. Subsequent validation must match this truth
sample. The extra pass does not run model inference or consume the training RNG.
The existing partial-final-batch behavior is preserved.

### Nested overrides

`--training-override` accepts YAML scalars, lists and mappings, plus dotted paths.
Quote the whole expression to protect it from shell interpretation. For example:

```text
--training-override 'physics_checkpoint.bin_edges=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0]'
--training-override 'physics_checkpoint.guardrails.B_macro=null'
--training-override 'physics_checkpoint={min_bin_entries: 200, require_configured_guardrails: false}'
```

Mappings merge recursively; lists and scalars replace the previous value. CLI
entries apply in order. Rendering merges base YAML, campaign overrides, then
per-run overrides, so changing one guardrail preserves the other limits and bin
settings. In a manually edited manifest, use actual nested YAML mappings rather
than literal dotted keys. Unknown physics settings and invalid bin definitions
fail during rendering. Omitted guardrails retain their inherited values; use
`null` to disable a particular limit explicitly.

### Campaign reports and plots

After training/evaluation, normal collation (or `--collate-only`) writes:

- `run_table.csv`: target, selection status, selected step/epoch/loss, validation
  and evaluation macro metrics, valid-bin counts and scalar inclusive metrics.
- `physics_checkpoint_summary.md`: compact table including runs without an
  acceptable checkpoint.
- `physics_checkpoint_metrics.csv`: separate `validation_selected` and
  `evaluation` rows, with units, full bin edges, policy ID and truth-sample hash.
- `physics_campaign_summary.json`: full validation histories, occupancy reports
  and evaluation diagnostics, including per-bin metrics.
- `physics_checkpoint_history.jsonl`: every checkpoint from every run, with its
  validation selection/Pareto flags.

Each run's `<checkpoint_stem>_physics_report.json` points to its current history
and occupancy report. Collation can follow these pointers on relocated mounts,
and prefers them over older artifact summaries after a rerun. Missing legacy
reports remain unavailable; rejected checkpoints are not used to fill selected
metric columns. Runs that fail preflight can still be inspected with
`--collate-only`.

```bash
/home/alessio/miniconda3/envs/fm4npp/bin/python \
  train/downstream/campaign/run_track_regression_campaign.py \
  --manifest /path/to/campaign/manifest.yaml --collate-only

/home/alessio/miniconda3/envs/fm4npp/bin/python \
  train/downstream/campaign/plot_track_regression_campaign.py \
  --campaign-dir /path/to/campaign --plot-suite physics
```

The `physics` suite writes PNG/PDF panels for width, macro bias, worst bias and
tails against training sample budget, with selection status shown. Validation
selection and evaluation use separate panels. Different targets, metric policies,
truth samples or valid-bin sets are not combined into one curve. `plot_index.json`
records the runs, policy, cohort, bin support and statuses behind each plot.
`standard` and `all` include these macro plots too; `all` skips the joint-vector
presentation/fitted-resolution suites for scalar-only campaigns. Mixed-target
standard plots are separated by task to avoid averaging log-momentum and angles.

These evaluation diagnostics still use the existing test folder and may overlap
selection samples; they are not an independent final test benchmark. Evaluation
metrics never drive checkpoint reselection. Hyperparameters can be specified
manually through the existing overrides; Optuna recipe import is unchanged.
