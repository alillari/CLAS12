# Momentum hyperparameter tuning reference

**Start with [Tune a working momentum adapter](GETTING_STARTED.md)** for the
sequence from a successful fit through a smoke study, full search, plots,
fresh-seed confirmation and campaign handoff. Ordinary fitting and statistics
are covered in the [momentum guide](../MOMENTUM.md).

## Tool boundaries

| Tool | Input → result |
| --- | --- |
| [run_track_regression_optuna.py](run_track_regression_optuna.py) | Named model YAML + fixed protocol → Optuna trials and fresh AdapterOnly fits. |
| [track_regression_search_space.py](track_regression_search_space.py) | Defines sampled ranges and derived LR values. Edit this function to change ranges; there is no per-parameter range CLI. |
| [analyze_track_regression_optuna.py](analyze_track_regression_optuna.py) | Existing database + fit logs → tables and diagnostic plots. |
| [build_track_regression_seed_ablation.py](build_track_regression_seed_ablation.py) | Completed recipes → candidate-by-seed campaign manifest. |
| [build_track_regression_manifest.py](../campaign/build_track_regression_manifest.py) | `--optuna-best-trial` imports the best completed recipe into a new ordinary campaign. |

The worker calls the shared `track_regression_experiment.py` training path with
`usepretrain=False`. Architecture, task, loss, dataset/filter contract and label
budget come from the selected YAML plus explicit CLI overrides. There is no
maintained Optuna worker here for finding or backbone pretraining. All current
studies minimize best finite validation loss; see
[checkpoint selection](../PHYSICS_CHECKPOINTS.md).

## Fixed controls and precedence

Base YAML values resolve environment variables and parameter references through
`YParams`, as in normal training. Fixed CLI settings override the base recipe;
sampled settings override corresponding optimizer values. Each seed receives
its own seed/output paths and a saved resolved configuration.

| Control | Meaning |
| --- | --- |
| `--yaml-config`, `--config` | Model file and named profile; both should be explicit when using portable onboarding recipes. |
| `--data-root`, `--data-root-test`, `--stat-dir`, `--regression-target-stats` | Optional overrides of dataset/normalization paths. |
| `--eventnumber` | Accepted training tracks, default 50k. Generate stats for the same training cohort. |
| `--train-batch-size` | Training tracks/batch; default 32, so pass 128 for the onboarding recipe. |
| `--max-optimizer-steps` | Maximum updates per fit, default 3000. |
| `--val-interval-steps`, `--max-val-batches` | Validation cadence and safety cap. The cap must accommodate the YAML's fixed validation sample. |
| `--early-stopping-min-steps`, `--early-stopping-patience` | Earliest stopping point and number of non-improving validation checks. Minimum steps must not exceed the budget. |
| `--max-train-batches` | Optional per-epoch training batch cap; changes data exposure. |
| `--num-data-workers`, `--cuda-device` | Loader workers and GPU selection. |
| `--sampler tpe` or `random`, `--seed` | Suggestion algorithm and its RNG seed. |
| `--trial-seeds` | Comma-separated fresh-fit seeds; default `11,17,23`. |
| `--seed-objective` | `mean` (default), `median`, or `mean_plus_std` of per-seed best validation losses. The last uses population standard deviation. |
| `--n-trials` | Additional trials in this invocation, default 3. |
| `--enable-pruning` | Allows intermediate-loss pruning for a **single** training seed only; incompatible with multi-seed aggregation. |

The [walkthrough's range table](GETTING_STARTED.md#1-understand-the-experiment)
lists the exact sampled intervals and explains why these are distinct from the
standard adapter defaults. Hold fixed the training labels, target stats, batch,
validation population/cadence, update budget, seeds and objective when comparing
trials. A different task/normalizer changes the meaning of the objective.

## Scheduler choices

`--scheduler-mode cosine_restarts` is the default. Set either `--n-cycles N`
(equal cycles in the full budget) or `--scheduler-first-cycle-steps S` (explicit
cycle length); these are mutually exclusive. A 20k budget with cycle length
7000 has two resets and a final partial cycle. `warmup_steps` is the sampled
fraction times the cycle length, truncated to an integer and at least 1.

`--scheduler-mode cosine_hold` instead warms up and decays once to `min_lr`,
then holds that floor through the remaining updates. It samples
`scheduler_anneal_steps` using `--anneal-steps-min`, `--anneal-steps-max` and
`--anneal-steps-step`. For example, at a 20k budget, min 5000, max 16000 and
step 1000 explore earlier decay endpoints. Endpoints must fit the budget.
Here `warmup_fraction` spans 0.10–0.40 and applies to the sampled annealing
duration. Do not pass either cycle control in hold mode.

Tune LR settings jointly with the schedule. The objective remains the best
validation observation anywhere in training, even if the last observation is
worse. Inspect per-seed curves and endpoint behavior before increasing budgets.

## Study state and artifacts

```text
<output-root>/<study-name>/
  trial_000000/
    seed_11/
      config/model.yaml
      config/resolved_config.json
      checkpoints/trial_000000_seed_11_adapter_checkpoint.pth
      train/artifacts.json
      wandb/                         # optional local W&B directory
    seed_17/ ...
    seed_23/ ...
    trial_result.json
  analysis/optuna_diagnostics/       # produced by the analyzer
```

The database is stored at `--storage`, independently of these fit artifacts.
Preserve both. Resume with the same storage, study name, output root, YAML and
fixed protocol. The study fingerprints YAML contents, resolved paths, stats,
label/batch/update/validation settings, schedule and seed protocol. Contract
changes require a new study. Also use a new study after editing the search
ranges. A partially finished fit is not optimizer-resumed when adding trials.

## Analysis controls

Pass `--study-dir` when fit artifacts are not beside the SQLite database and
`--output-dir` for an explicit diagnostics destination. Missing histories or
few completed trials can suppress particular plots; inspect
`generated_outputs.json` and any diagnostic error receipt.

Parallel-coordinate and interaction plots default to the top 20% of completed
trials. Choose `--parallel-filter all`, `top-fraction`, `absolute`, or `relative`,
then set `--top-fraction`, `--absolute-threshold`, or `--relative-pct` respectively.
`--interaction-params` chooses parameters explicitly; otherwise importance
ranking determines them. `--checkpoint-fractions` and
`--checkpoint-tolerance-fraction` control comparisons of early and final ranks.

Use completed trials to choose narrower ranges. Inspect log-scale parameters
in log space and retain interactions. If good fits still improve near the
budget limit, test longer training under a new protocol. Confirm candidates on
fresh seeds and score a separate final cohort after selection; trial importance
and an Optuna winner alone do not establish generalization.

## Optional W&B logging

W&B is disabled by default. Add `--wandb-mode offline` or `online` and optionally
`--wandb-project`, `--wandb-entity`, `--wandb-dir` to the worker. This requires
`wandb` installed; online mode also needs the usual account configuration.
There is **one run per trial/seed**, grouped by study. Without `--wandb-dir`,
local files go beneath that seed's `wandb/` directory (W&B may add another
`wandb/` level inside it). Sync the actual `offline-run-*` directory reported
by W&B with its `wandb sync` CLI.

Logs include training/validation loss, LR, best loss/step, update and epoch.
The integration sends configs and scalar metrics; it does not upload raw data
or checkpoints as W&B artifacts. SQLite and local receipts remain the inputs
for the analyzer and recipe import.
