# AdapterOnly Optuna Tuning

This directory contains the Optuna worker used to tune the CLAS12 AdapterOnly
track-regression baseline. The tuner calls the shared one-configuration training
entrypoint in `train/downstream/track_regression_experiment.py`; it does not
replace the trainer or the campaign manifest tools.

## What Optuna Tunes

The current search space is defined in `track_regression_search_space.py`.

Currently sampled:

- `max_lr`
- `min_lr_ratio`, with `min_lr = max_lr * min_lr_ratio`
- `warmup_fraction`
- `adapter_weight_decay`
- `grad_clip_value`
- `dropout`

Keep these fixed for a given study:

- labeled-data amount, via `--eventnumber`
- batch size, via `--train-batch-size`
- training budget, via `--max-optimizer-steps`
- validation cadence and budget
- dataset paths and split
- AdapterOnly mode
- optimizer and scheduler family
- architecture choices

Use a new study name/database when changing the search space or fixed training
budget.

## Smoke Test

Run a tiny plumbing test before long studies:

```bash
mkdir -p /home/alessio/ML-work/result_deep_storage/optuna

python train/downstream/tuning/run_track_regression_optuna.py \
  --storage sqlite:////home/alessio/ML-work/result_deep_storage/optuna/adapteronly_smoke.db \
  --study-name adapteronly_smoke \
  --output-root /home/alessio/ML-work/result_deep_storage/optuna \
  --n-trials 3 \
  --cuda-device 0 \
  --eventnumber 1000 \
  --train-batch-size 16 \
  --max-optimizer-steps 200 \
  --val-interval-steps 100 \
  --early-stopping-min-steps 200 \
  --max-val-batches 50 \
  --num-data-workers 4
```

## Range-Finding Run

Example 50k-step range-finding study:

```bash
python train/downstream/tuning/run_track_regression_optuna.py \
  --storage sqlite:////home/alessio/ML-work/result_deep_storage/optuna/adapteronly_rangefind_50k.db \
  --study-name adapteronly_rangefind_50k \
  --output-root /home/alessio/ML-work/result_deep_storage/optuna \
  --n-trials 50 \
  --cuda-device 0 \
  --eventnumber 50000 \
  --train-batch-size 128 \
  --max-optimizer-steps 50000 \
  --val-interval-steps 5000 \
  --early-stopping-min-steps 20000 \
  --early-stopping-patience 4 \
  --max-val-batches 1000 \
  --num-data-workers 24
```

`scheduler_first_cycle_steps` is set automatically to `max_optimizer_steps` by
the launcher, so each trial uses one warmup plus cosine cycle over the full
trial budget.

## W&B Logging

W&B is optional and off by default. Online mode creates one W&B run per Optuna
trial:

```bash
python train/downstream/tuning/run_track_regression_optuna.py \
  ... \
  --wandb-mode online \
  --wandb-project clas12-adapteronly
```

Offline mode writes local W&B run files that can be synced later:

```bash
python train/downstream/tuning/run_track_regression_optuna.py \
  ... \
  --wandb-mode offline \
  --wandb-project clas12-adapteronly

wandb sync /home/alessio/ML-work/result_deep_storage/optuna/<study_name>/trial_*/wandb/offline-run-*
```

Logged metrics include:

- `train/loss`
- `val/loss`
- `lr`
- `best/val_loss`
- `best/step`
- `step`
- `epoch`

The code logs configs and scalar metrics only. It does not upload raw data or
checkpoints as W&B artifacts.

## Outputs

For study `<study_name>`, outputs are written under:

```text
<output_root>/<study_name>/
```

Each trial has its own directory:

```text
trial_000123/
  config/
    model.yaml
    resolved_config.json
  checkpoints/
    trial_000123.log
    trial_000123_adapter_checkpoint.pth
  train/
    artifacts.json
  trial_result.json
  wandb/                  # only when W&B is enabled
```

The Optuna SQLite database is the path passed through `--storage`.

## Resuming A Study

To continue a stopped study, rerun the same command with the same:

- `--storage`
- `--study-name`
- `--output-root`

`--n-trials` means "run this many additional trials in this invocation." It does
not mean "bring the study up to this total number of trials."

Interrupted half-finished trials are not resumed from their partial checkpoint.
The worker continues by adding new trials to the existing study.

## Inspecting Results

Basic Optuna summary:

```bash
python - <<'PY'
import optuna

storage = "sqlite:////home/alessio/ML-work/result_deep_storage/optuna/adapteronly_rangefind_50k.db"
study = optuna.load_study("adapteronly_rangefind_50k", storage=storage)

print("Trials:", len(study.trials))
print("Best value:", study.best_value)
print("Best trial:", study.best_trial.number)
print("Best params:")
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")

df = study.trials_dataframe()
cols = [
    "number",
    "state",
    "value",
    "params_max_lr",
    "params_min_lr_ratio",
    "params_warmup_fraction",
    "params_adapter_weight_decay",
    "params_grad_clip_value",
    "params_dropout",
    "user_attrs_best_step",
]
print(df[cols].sort_values("value").head(10).to_string(index=False))
PY
```

If `params_min_lr_ratio` is missing in an older study, compute it as:

```text
min_lr_ratio = params_min_lr / params_max_lr
```

## Choosing The Next Search Space

Use completed trials only. Pick a "good set" from either the top 20-30 percent
or all trials within a tolerance of the best validation loss.

For log-scale parameters (`max_lr`, `min_lr_ratio`, `adapter_weight_decay`,
`grad_clip_value`), inspect ranges in log space. If a parameter performs well
across several orders of magnitude and shows no interaction, fix it to a simple
value in the next study rather than spending search budget on it.

If many good trials have `best_step == max_optimizer_steps`, the training budget
is probably truncating learning. Run a longer, narrower study or train the top
configs directly for a larger final budget.

