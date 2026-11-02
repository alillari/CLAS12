# Move from single adapter runs to campaigns

Start here after a successful [single adapter run](../../TRAINING_RECIPES.md)
or the [backbone-to-adapters example](../../WORKED_EXAMPLE.md). A campaign runs
the same public training and evaluation commands for a list of configurations,
keeps their outputs separate, records progress and builds comparison tables.

The runners execute one training/evaluation subprocess at a time on the chosen
GPU. They do not pretrain backbones or search hyperparameters automatically.
Keep the successful single-run recipe as the starting point. For optimizer
search first, follow [the tuning walkthrough](../tuning/GETTING_STARTED.md);
it shows how to import a selected recipe into these runners.

## What the files mean

| File | Purpose |
| --- | --- |
| Base model YAML | The working adapter recipe: architecture, filters, optimizer and validation settings. Select its named profile explicitly. |
| `manifest.yaml` | The run list: backbone checkpoints, labeled sample counts, output paths and shared/per-run overrides. Building it does not train anything. |
| `runs/<run_id>/config/model.yaml` | The runner's rendered recipe for one fit. |
| `runs/<run_id>/config/analysis.yaml` | Automatically rendered evaluator configuration using that fit's checkpoint/backbone. No analysis-generation script is needed. |
| `status.yaml` | Stage bookkeeping used to skip completed work and report failures. It is not an optimizer checkpoint. |
| `summary/` | Collated results and comparison plots. |

Normal execution is **build manifest → dry-run → train/evaluate → collate → plot**.
This walkthrough starts with one Mamba1 width-512/depth-12/state-16/k1 backbone,
one 50k label budget, and an AdapterOnly baseline for each task: two fits per
campaign. These are new fits; existing standalone adapter weights are not
automatically imported or continued.

## 1. Reuse the working inputs and register the backbone

Keep the environment from the worked example: `CLAS12_ARTIFACT_ROOT`,
`CLAS12_DATA_ROOT`, `CLAS12_VALIDATION_DATA_ROOT`, `MODEL_YAML`,
`CLAS12_PRETRAINED_CKPT` and `CLAS12_MOMENTUM_STATS`. `MODEL_YAML` is the copied
adapter recipe with profiles `clas12_track_finding_50k` and
`clas12_momentum_50k`; carry any intentional changes from the successful run.

The builders discover **backbones**, not trained adapters. They expect one
directory per backbone with `w<width>` and `d<depth>` tokens in its name and one
checkpoint-like file directly inside. Register the worked example's checkpoint
without copying or changing its weights:

```bash
export CAMPAIGN_CHECKPOINT_ROOT="$CLAS12_ARTIFACT_ROOT/campaign_inputs/one_backbone"
mkdir -p "$CAMPAIGN_CHECKPOINT_ROOT/scale_w512_d12"
ln -s "$CLAS12_PRETRAINED_CKPT" "$CAMPAIGN_CHECKPOINT_ROOT/scale_w512_d12/ckpt_best.tar"

export FINDING_CAMPAIGN="$CLAS12_ARTIFACT_ROOT/campaigns/finding_w512_50k"
export MOMENTUM_CAMPAIGN="$CLAS12_ARTIFACT_ROOT/campaigns/momentum_w512_50k"
```

The symlink is a one-time setup; leave an existing correct link in place.
Optional `n<events>` in the directory name records the number of pretraining
events, not optimizer updates. Without it, the name parser records zero/unknown.
Neither a directory name nor a metadata sidecar validates checkpoint tensors;
the chosen family/embedding/block settings must still match the weights.

Reuse the **50k training-track** momentum stats from the single run. Both
AdapterOnly and frozen-feature rows use the same target normalizer. See
[Target statistics](../TARGET_STATISTICS.md) if the data/filter/budget changed.

### Additional finding preflight inputs

The finding campaign's dataset preflight requires `features`, `seg_target` and
`coatjava_seg_pred` for **both pretrain and test splits under `CLAS12_DATA_ROOT`**.
It checks event/hit alignment and a minimum multitrack-event fraction (default
0.8). This preflight checks that root's test trees even when training validation
uses `CLAS12_VALIDATION_DATA_ROOT`. It does not establish event disjointness.

The campaign may therefore require COATJAVA sidecars that a standalone fit did
not need. Its analysis template also enables COATJAVA evaluation, so the actual
validation/evaluation root needs `coatjava_seg_pred_test`. If baseline evaluation
is intentionally disabled, set `analysis_overrides.evaluate_coatjava: false` in
the finding manifest; the preflight's sidecar requirement still applies. Choose
`--min-multitrack-fraction` based on the intended dataset population rather than
lowering it merely to pass a failed check.

## 2. Build one manifest per task

Both commands reuse the successful 30k-update profiles. The explicit width/depth
overrides keep the AdapterOnly input embedding at the same 512-wide input as
the frozen-feature head; builders otherwise give their baseline rows older
128-wide metadata. This shared override is appropriate for this **single-width**
example, not a mixed-width backbone sweep.

Track finding:

```bash
python train/downstream/campaign/build_track_finding_manifest.py \
  --checkpoint-root "$CAMPAIGN_CHECKPOINT_ROOT" \
  --campaign-name finding_w512_50k --artifact-root "$CLAS12_ARTIFACT_ROOT" \
  --base-model-yaml "$MODEL_YAML" --base-model-config clas12_track_finding_50k \
  --base-analysis-yaml train/downstream/eval/track_finding_analysis_adapteronly.yaml \
  --eventnumber 50000 --train-batch-size 32 \
  --validation-events 5000 --final-evaluation-events 10000 \
  --seed 42 --assignment-threshold 0.2 \
  --training-override embed_dim=512 --training-override num_layers_backbone=12
```

Momentum:

```bash
python train/downstream/campaign/build_track_regression_manifest.py \
  --checkpoint-root "$CAMPAIGN_CHECKPOINT_ROOT" \
  --campaign-name momentum_w512_50k --artifact-root "$CLAS12_ARTIFACT_ROOT" \
  --base-model-yaml "$MODEL_YAML" --pretrained-model-config clas12_momentum_50k \
  --adapter-only-model-yaml "$MODEL_YAML" --adapter-only-model-config clas12_momentum_50k \
  --base-analysis-yaml train/downstream/eval/track_regression_analysis_adapteronly.yaml \
  --eventnumber 50000 --train-batch-size 128 \
  --validation-samples 10000 --validation-batch-size 128 --max-samples 50000 \
  --training-override seed=42 \
  --training-override embed_dim=512 --training-override num_layers_backbone=12
```

Each builder writes `<campaign_dir>/manifest.yaml`. AdapterOnly rows are
included by default; `--no-adapter-only` omits them. Inspect the `runs` list,
checkpoint paths, base profiles and `training_overrides` before launch.

| Task | Generated run IDs in this example |
| --- | --- |
| Finding | `adapteronly_w128_d12_label50000`, `scale_w512_d12_label50000` |
| Momentum | `adapteronly_label50000`, `scale_w512_d12` |

The finding baseline's `w128` is part of the builder's run label; the explicit
override above makes its actual input width 512. Always inspect rendered
parameters rather than inferring architecture from the run ID.

Overrides apply after the base recipe; per-run `training_overrides` apply after
campaign-level overrides. Do not edit global defaults to change one run.

## 3. Inspect the commands before starting

```bash
python train/downstream/campaign/run_track_finding_campaign.py \
  --manifest "$FINDING_CAMPAIGN/manifest.yaml" --dry-run

python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest "$MOMENTUM_CAMPAIGN/manifest.yaml" --dry-run
```

Dry-run prints the planned commands and paths and checks selected pretrained
checkpoint files exist. It does not render configs, load weights, check data or
exercise CUDA. Confirm batch size, label count, backbone flags and output paths.

Finding additionally offers a CPU dataset preflight without training:

```bash
python train/downstream/campaign/run_track_finding_campaign.py \
  --manifest "$FINDING_CAMPAIGN/manifest.yaml" --preflight-only
```

This writes `summary/data_preflight.csv`. Review the sampled counts, alignment
and multitrack fractions before spending GPU time.

## 4. Run the campaigns and inspect progress

Run the two commands sequentially if they share GPU 0:

```bash
python train/downstream/campaign/run_track_finding_campaign.py \
  --manifest "$FINDING_CAMPAIGN/manifest.yaml" --cuda-device 0

python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest "$MOMENTUM_CAMPAIGN/manifest.yaml" --cuda-device 0
```

For each row the runner renders model/analysis YAMLs, trains, evaluates and
updates `status.yaml`. Normal execution collates results after the selected
queue succeeds. A failed subprocess stops the queue; inspect its stdout log
before restarting. Finding also performs a CUDA preflight for each subprocess.

```bash
python train/downstream/campaign/run_track_finding_campaign.py \
  --manifest "$FINDING_CAMPAIGN/manifest.yaml" --status

python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest "$MOMENTUM_CAMPAIGN/manifest.yaml" --status
```

For a detached Docker launch, monitoring and result retrieval, follow
[Remote edgexpert jobs](../../REMOTE_RUNS.md).

The runners do not detach themselves. Use the terminal/session or job mechanism
normally used for long training runs. Changing `--cuda-device` selects the GPU
exposed to the subprocess; it does not make rows run concurrently.

## 5. Read the outputs and generate comparisons

```text
campaign_dir/
  manifest.yaml
  status.yaml
  logs/<run_id>.train.stdout.log
  logs/<run_id>.eval.stdout.log
  runs/<run_id>/
    config/model.yaml
    config/analysis.yaml
    train/artifacts.json
    checkpoints/<run_id>.log
    checkpoints/<run_id>_adapter_checkpoint.pth
    evaluation/summary.json
    evaluation/...                 # per-task metrics/predictions/plots
  summary/run_table.csv
  summary/campaign_headline_metrics.jsonl
```

Finding may record attempt-specific log paths; use the current path in
`status.yaml` after retries. Start with `summary/run_table.csv`, then inspect
each run's receipt and evaluator summary for actual steps, counts and selection.
Momentum evaluator plots are generated per run. Compare completed runs with:

```bash
python train/downstream/campaign/plot_track_finding_campaign.py \
  --campaign-dir "$FINDING_CAMPAIGN"

python train/downstream/campaign/plot_track_regression_campaign.py \
  --campaign-dir "$MOMENTUM_CAMPAIGN" --plot-suite all
```

Campaign figures are written under `summary/plots/` unless an output directory
is supplied. Finding produces model/label-budget, individual-metric and
presentation panels; momentum's `all` suite includes supported metric and
resolution panels. With only one backbone and one budget, there is no scaling
curve to infer. Use the [evaluation reference](../eval/README.md) to interpret
per-run residual/fit plots and [the campaign reference](README.md) for the full
suite/output options.

### Validation versus a final test

These runners render one model YAML per row and use its `data_root_test` for
both checkpoint-selection validation and automatic evaluation. The example
therefore evaluates prefixes of the configured validation product. A larger
`--max-samples` or `--final-evaluation-events` limit does **not** make that an
independent test; `CLAS12_EVAL_DATA_ROOT` is not automatically selected here.

Treat these automatic results as validation diagnostics. For an independent
final result, evaluate a reserved population with an explicit evaluation model
profile, as in [the worked example](../../WORKED_EXAMPLE.md#5-evaluate-on-the-reserved-final-population),
matching each trained row's architecture, stats, weights and operating point.
Keep those outputs separate. Do not present the default campaign comparisons
as held-out test performance.

## 6. Restart, select rows or repeat evaluation

Rerun the same runner command with the **same manifest** to continue a queue.
Completed training/evaluation stages are skipped when their expected files and
status records are present. An interrupted fit is not automatically resumed
from its optimizer state: the public training CLI starts a fresh fit when that
stage is rerun. Reusing a campaign directory after changing a recipe does not
invalidate completed statuses automatically; use a new campaign name for a
changed scientific recipe/population/seed.

After a regression evaluation failure, use `--skip-train --force-eval` when
retrying inference on its saved adapter. A plain restart can repeat training
because the current status is `failed`. Finding explicitly preserves a saved
training checkpoint after an evaluation-stage failure.

| Goal | Control |
| --- | --- |
| Run a particular row | `--only RUN_ID`; repeat it to select several rows. |
| Process the first row | `--limit 1`; useful after inspecting the manifest order. |
| Train without evaluation | `--skip-eval`; rerun normally later. |
| Recompute evaluations of completed rows | `--force-eval`; check training status/checkpoints first. Regression can additionally use `--skip-train` to prohibit training. |
| Start fresh training for selected rows | `--force-train`; this is not a resume flag. |
| Rebuild tables from existing evaluator outputs | `--collate-only`; no training or inference. |

For example, rebuild summaries and refresh plots without rerunning models:

```bash
python train/downstream/campaign/run_track_finding_campaign.py \
  --manifest "$FINDING_CAMPAIGN/manifest.yaml" --collate-only
python train/downstream/campaign/plot_track_finding_campaign.py \
  --campaign-dir "$FINDING_CAMPAIGN"

python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest "$MOMENTUM_CAMPAIGN/manifest.yaml" --collate-only
python train/downstream/campaign/plot_track_regression_campaign.py \
  --campaign-dir "$MOMENTUM_CAMPAIGN" --plot-suite all
```

### What about the adapters already trained as single runs?

Keep using their standalone evaluators/analysis files if the goal is to inspect
those existing weights. Building a campaign does not discover or import them.
Regression supports deliberate reuse: point a row's `adapter_checkpoint` and
`training_log` at those files, match its model/normalizer/population metadata,
and run `--skip-train --force-eval`. Finding has no `--skip-train` switch; use
its standalone evaluator for evaluation-only work outside an established
completed campaign. Copying a checkpoint or changing a status field alone does
not establish a compatible recipe or a valid completed training stage.

## 7. Expand only after the small campaign is checked

- **More backbones:** add one appropriately named input directory per checkpoint,
  with matching family/embedding/block metadata. Remove the global width=512
  override for a mixed-width sweep and set baseline width deliberately. The
  finding builder parses names; the regression builder also reads metadata
  sidecars. Neither automatically infers every architectural option from weights.
- **More label budgets:** both builders accept `--eventnumber 1000,10000,50000`.
  Momentum needs stats from each corresponding training-track cohort; add each
  existing row's `training_overrides.regression_target_stats` path explicitly.
  Reusing the 50k stats for all budgets changes the intended low-label recipe.
- **More seeds:** use a new campaign name and change finding's `--seed` or
  regression's `--training-override seed=...`. Neither builder creates a seed
  grid from a list automatically.
- **Other architecture/optimizer choices:** transfer a validated recipe via the
  base profile and explicit overrides. A campaign automates repetition; it does
  not make unequal head capacities, different normalizers or different samples
  comparable by itself.

For a label sweep, generate each stats JSON with the existing
`compute_regression_target_stats.py` command and that budget, then add this
mapping to each relevant **existing** momentum row in the manifest:

```yaml
training_overrides:
  regression_target_stats: /absolute/path/to/stats_for_this_rows_training_tracks.json
```

The same cohort/budget's AdapterOnly and frozen rows should share that file.
Keep the validation budget separate from the training-label budget.
