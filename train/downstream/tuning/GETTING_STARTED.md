# Tune a working momentum adapter

Start after the [single momentum run](../MOMENTUM.md) or
[worked example](../../WORKED_EXAMPLE.md) succeeds. This walkthrough uses the
same 50k training tracks, target statistics, 10k validation tracks and width-512
AdapterOnly architecture. It searches optimizer settings; it does not pretrain
backbones, tune track finding, or search architecture. A frozen-backbone
campaign can reuse the selected recipe later, but that transfer needs evaluation.

## 1. Understand the experiment

A **study** is a collection of trials with a common dataset, architecture,
training budget and objective. A **trial** proposes one optimizer recipe.
Each **training seed** starts a fresh adapter fit with that recipe. With three
seeds, one trial costs three fits. The worker runs these sequentially on one GPU;
`--n-jobs` currently supports only 1.

Each fit selects its checkpoint by minimum finite validation loss. The trial
objective is the mean of those per-seed minima by default; smaller is better.
For standardized MAE this is error in standardized target space, not GeV.
`--seed` controls Optuna's sampler; `--trial-seeds` controls training randomness.
Final evaluation data must stay outside this selection process.

| Optuna varies | Current range | What it controls |
| --- | --- | --- |
| `max_lr` | 0.0004–0.0025, log scale | Peak learning rate. |
| `min_lr_ratio` | 0.0005–0.08, log scale | Initial/floor LR is `max_lr * min_lr_ratio`. |
| `warmup_fraction` | 0.10–0.35 | Fraction of each cosine cycle used for warmup. |
| `adapter_weight_decay` | 0.001–0.5, log scale | AdamW regularization. |
| `grad_clip_value` | 0.5–8, log scale | Gradient **norm** clipping threshold. |
| `dropout` | 0.02–0.20 | Adapter dropout probability. |

These ranges are defined in
[track_regression_search_space.py](track_regression_search_space.py).
They exclude several standard recipe values: peak LR 0.0003, decay 0.0001,
clip norm 0.1 and warmup fraction 1000/30000. The worker does **not** enqueue
that baseline automatically. Retain an independent baseline with matched data,
seeds, budget and selection. Edit the search-space function and create a new
study if the intended range should include those values. A search winner is
only the best completed candidate under this study's protocol.

## 2. Freeze the working recipe and paths

Run from the repository root in the training environment with `optuna`
installed. Keep `CLAS12_DATA_ROOT`, `CLAS12_VALIDATION_DATA_ROOT`,
`CLAS12_MOMENTUM_STATS`, `CLAS12_ARTIFACT_ROOT` and `MODEL_YAML` from the single
run. Stats must describe the same 50k training-track cohort and filters; see
[target statistics](../TARGET_STATISTICS.md) if changing that cohort.

```bash
export TUNING_ROOT="$CLAS12_ARTIFACT_ROOT/optuna"
export TUNING_MODEL_YAML="$TUNING_ROOT/config/momentum.yaml"
mkdir -p "$TUNING_ROOT/config"
cp "$MODEL_YAML" "$TUNING_MODEL_YAML"
export STUDY=momentum_50k_30k_updates
export STORAGE="sqlite:///$TUNING_ROOT/$STUDY.db"
```

Use an absolute artifact root: SQLite then receives four slashes before the
absolute path. Keep this copied YAML and the resolved data/stats paths stable
for the life of the study. Do not recopy an edited recipe over it when resuming.
The selected `clas12_momentum_50k` profile fixes task `mom`, standardized MAE,
10k validation tracks at batch 128, filters and architecture.

## 3. Check the worker with one short fit

```bash
python train/downstream/tuning/run_track_regression_optuna.py \
  --storage "sqlite:///$TUNING_ROOT/momentum_smoke.db" \
  --study-name momentum_smoke --output-root "$TUNING_ROOT" \
  --yaml-config "$TUNING_MODEL_YAML" --config clas12_momentum_50k \
  --n-trials 1 --trial-seeds 42 --seed 42 \
  --eventnumber 50000 --train-batch-size 128 \
  --max-optimizer-steps 200 --scheduler-mode cosine_restarts --n-cycles 1 \
  --val-interval-steps 100 --early-stopping-min-steps 200 \
  --max-val-batches 80 --num-data-workers 4 --cuda-device 0
```

This uses the same labels/stats but only 200 optimizer updates. It checks the
training path, not convergence. The 80-batch cap permits all 10k validation
tracks at batch 128. A lower cap cannot silently shrink that fixed validation
sample: reduce `limit_test_size` in a separate copied profile if needed.
Use a separate smoke database/study so short fits cannot win the full study.

## 4. Run the search and resume it

```bash
python train/downstream/tuning/run_track_regression_optuna.py \
  --storage "$STORAGE" --study-name "$STUDY" --output-root "$TUNING_ROOT" \
  --yaml-config "$TUNING_MODEL_YAML" --config clas12_momentum_50k \
  --n-trials 20 --trial-seeds 11,17,23 --seed 42 --seed-objective mean \
  --eventnumber 50000 --train-batch-size 128 \
  --max-optimizer-steps 30000 --scheduler-mode cosine_restarts --n-cycles 1 \
  --val-interval-steps 1000 --early-stopping-min-steps 30000 \
  --early-stopping-patience 8 --max-val-batches 80 \
  --num-data-workers 4 --cuda-device 0
```

This requests 20 trials × 3 seeds = 60 sequential fits, up to 1.8 million
optimizer updates plus validation. Minimum early-stopping step equals the
budget so each successful fit gets the full budget. One cycle warms up, then
cosine-decays to the floor without another cycle inside those 30k updates.

To add five trials, rerun this command with `--n-trials 5`. That means five
**additional** trials, not five total. The database retains trial state; keep
its per-seed artifacts too. Interrupted fits do not resume their optimizer
checkpoint automatically. The study contract rejects changes to fixed inputs,
YAML contents, stats, budget, seed protocol and schedule. Use a new study for
changed conditions, including changed search ranges. There is no terminal
session detachment built into the worker.

## 5. Inspect the study and its plots

```bash
python train/downstream/tuning/analyze_track_regression_optuna.py \
  --storage "$STORAGE" --study-name "$STUDY" \
  --study-dir "$TUNING_ROOT/$STUDY" \
  --output-dir "$TUNING_ROOT/$STUDY/analysis/optuna_diagnostics"

python -m json.tool "$TUNING_ROOT/$STUDY/analysis/optuna_diagnostics/study_summary.json"
```

The analyzer reads existing trials and logs; it does not train or evaluate
adapters. Start with these outputs:

| Output | What to check |
| --- | --- |
| `study_summary.json`, `completed_trials.csv` | Best trial/parameters/objective, completed versus failed trials and recorded seed results. |
| `optimization_history.png`, `objective_distribution.png` | Whether additional trials improve the objective and how broad the outcomes are. |
| `aggregate_validation_loss_by_step.png/.csv` | Per-step seed aggregate, then trial population summaries, where logs are available. |
| `parameter_importance.csv`, parameter-slice plots | Which settings correlate with validation loss inside this search. |
| Parallel-coordinate / pairwise-interaction plots | Promising combinations; these use the top 20% of completed trials by default. |
| `checkpoint_rank_stability*` | Whether early checkpoint rankings predict later rankings. |
| `generated_outputs.json` | The actual generated-file list; some diagnostics need more completed trials/history. |

Per-step curves combine contemporaneous losses using the chosen seed objective
and only steps present for every seed. Final trial scores instead combine each
seed's best loss, whose selected steps may differ; these need not equal a curve
minimum. Missing seed logs suppress that trial's curve rather than reduce its
seed population.

Read individual `trial_result.json` and `seed_*/train/artifacts.json` before
accepting a low score. Check seed spread, counts, best step, failures and
whether training still improves at the budget boundary. Importance and
interaction plots describe this sampled study; they are not causal evidence.

## 6. Confirm candidates on fresh seeds

The seed-ablation builder selects completed recipes and produces an ordinary
regression campaign manifest. This example selects up to three best recipes
and trains each on seeds 31,43,59: up to nine fresh AdapterOnly fits.

```bash
python train/downstream/tuning/build_track_regression_seed_ablation.py \
  --storage "$STORAGE" --study-name "$STUDY" \
  --ablation-name momentum_tuning_confirm --output-root "$CLAS12_ARTIFACT_ROOT" \
  --yaml-config "$TUNING_MODEL_YAML" --config clas12_momentum_50k \
  --top-k 3 --quantile-k 0 --random-k 0 --seeds 31,43,59 \
  --eventnumber 50000 --train-batch-size 128 --max-optimizer-steps 30000 \
  --n-cycles 1 --val-interval-steps 1000 --early-stopping-min-steps 30000 \
  --max-val-batches 80 --max-samples 50000

export CONFIRM_MANIFEST="$CLAS12_ARTIFACT_ROOT/campaigns/momentum_tuning_confirm/manifest.yaml"
python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest "$CONFIRM_MANIFEST" --dry-run
python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest "$CONFIRM_MANIFEST" --cuda-device 0
python train/downstream/campaign/plot_track_regression_campaign.py \
  --campaign-dir "$CLAS12_ARTIFACT_ROOT/campaigns/momentum_tuning_confirm" --plot-suite all
```

The builder recovers source resolved settings when available; those can override
its fallback budget defaults. Inspect manifest overrides and rendered YAMLs.
`--quantile-k` adds recipes across the completed ranking, `--random-k` adds a
random subset, and `--sample-seed` controls that selection. Overlapping recipe
selections are deduplicated before expanding training seeds.

Automatic campaign evaluation uses the campaign's validation root. It remains
selection diagnostics; reserve a separate final cohort for the explicit
[evaluation workflow](../../WORKED_EXAMPLE.md#5-evaluate-on-the-reserved-final-population).

## 7. Reuse the best recipe in a new campaign

To import only the best completed trial's optimizer/schedule recipe into a
new single-seed AdapterOnly campaign:

```bash
python train/downstream/campaign/build_track_regression_manifest.py \
  --adapter-only-only --campaign-name momentum_tuned_seed31 \
  --artifact-root "$CLAS12_ARTIFACT_ROOT" \
  --base-model-yaml "$TUNING_MODEL_YAML" --pretrained-model-config clas12_momentum_50k \
  --adapter-only-model-yaml "$TUNING_MODEL_YAML" --adapter-only-model-config clas12_momentum_50k \
  --eventnumber 50000 --train-batch-size 128 \
  --validation-samples 10000 --validation-batch-size 128 --max-samples 50000 \
  --training-override seed=31 \
  --training-override embed_dim=512 --training-override num_layers_backbone=12 \
  --optuna-best-trial --optuna-storage "$STORAGE" --optuna-study-name "$STUDY"
```

Then dry-run, execute and plot using the campaign commands above with the new
campaign path. Importing a recipe does **not** import trained adapter weights.
The manifest builder carries tuned scalars and the recorded update/schedule/
validation contract; label count, batch size, data and architecture must still
be chosen consistently. For a frozen-backbone comparison, use the
[campaign walkthrough](../campaign/GETTING_STARTED.md) and add these three
Optuna flags to its momentum builder. Check backbone compatibility and evaluate
recipe transfer rather than assuming an AdapterOnly optimum transfers.

For schedule alternatives, pruning, W&B and output layout, see the
[tuning reference](README.md).
