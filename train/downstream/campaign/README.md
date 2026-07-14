# Track-Regression Campaign Runner

This directory contains local campaign tools for training CLAS12 track-regression
adapters on a folder of pretrained Mamba backbones and then running the physics
evaluation for each adapter.

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
scale_w1536_d12_n5483352_label100
scale_w1536_d12_n5483352_label1000
scale_w1536_d12_n5483352_label10000
scale_w1536_d12_n5483352_label70000
```

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
  --max-samples 10000
```

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
        campaign_headline_metrics.jsonl
        plots/
  summary/
    campaign_headline_metrics.jsonl
    run_table.csv
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

The plotting script reads:

```text
<campaign_dir>/summary/campaign_headline_metrics.jsonl
<campaign_dir>/manifest.yaml
```

and writes:

```text
<campaign_dir>/summary/plots/
  plot_data.csv
  best_by_slice.csv
  rmse_mean_vs_labeled_events_by_width.png
  rmse_mean_vs_backbone_params_by_labeled_events.png
  rmse_mean_vs_pretrain_events_by_labeled_events.png
  r2_mean_vs_labeled_events_by_width.png
  r2_mean_vs_backbone_params_by_labeled_events.png
  r2_mean_vs_pretrain_events_by_labeled_events.png
  rmse_<component>_vs_*.png
  r2_<component>_vs_*.png
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

- RMSE for `px`, `py`, and `pz`;
- R2 for `px`, `py`, and `pz`;
- mean RMSE averaged across `px`, `py`, and `pz`;
- mean R2 averaged across `px`, `py`, and `pz`.

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
