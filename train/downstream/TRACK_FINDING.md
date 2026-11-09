# CLAS12 track finding

The track-finding adapter assigns clusters in a complete event to signal-track
instances or background. Start with [environment/data setup](../../SETUP.md),
then follow the task-specific route below. Technical settings are defined in
[Training options](TRAINING_OPTIONS.md#track-finding-head-and-objective).

## Inputs and starting recipe

- Use aligned `features_pretrain/test`, `seg_target_pretrain/test` and
  `reg_target_pretrain/test` event products. The shared training/evaluation path
  requests regression sidecars for physical diagnostics even though the finding
  loss uses segmentation. Features are raw three-column xyz cluster centers.
- Signal segment IDs define instances; `-1` denotes background. With
  `track_target_mode: signal_only`, background is negative in each signal mask
  and is not an additional query target.
- Samples and `--eventnumber` count **events**. The starting profile accepts
  1–512 clusters/event, uses 16 queries and a width-256 attention decoder,
  two prototype-attention refinement layers and zero Mamba input/refinement layers.
- `clas12_track_finding_50k` in
  [the adapter recipes](../../scripts/configs/clas12_adapter_recipes.yaml) uses
  50k training events, batch 32, 30k optimizer updates, validation every 1k updates
  on 5k events, and seed 42. Its class/no-object/Dice/focal weights are
  0.5 / 0.1 / 1 / 30.

The preset's frozen-backbone contract is Mamba1 width 512/depth 12/state 16/k1,
position-only NeRF embedding and calibrated12 band-Hilbert serialization.
Other checkpoints require their matching configuration. For AdapterOnly,
omit `--usepretrain`; for frozen features, supply both `--usepretrain` and
`--pretrained_ckpt`. Track finding needs **no target-standardization JSON**.

Run the trainer with one process and select the GPU with `CUDA_VISIBLE_DEVICES`.
Frozen-feature evaluation requires the original backbone weights. New adapter
checkpoints record the backbone file's SHA-256 digest, input mode, and decoder
settings. Older checkpoints use their saved training parameters to validate
the input mode and decoder, and lack a backbone digest.

## Run and evaluate

1. [Set paths and copy the model recipe](../TRAINING_RECIPES.md#1-choose-a-task-and-prepare-the-run-configuration).
   Set `CLAS12_VALIDATION_DATA_ROOT` if validation uses a separate event product.
2. [Run the finding command](../TRAINING_RECIPES.md#track-finding).
   Pass `--train_batch_size 32`; the CLI overrides YAML batch settings and also
   sets the validation batch. `max_optimizer_steps` controls the update budget.
3. [Copy/edit the named analysis YAML and evaluate](../TRAINING_RECIPES.md#4-evaluate-the-saved-checkpoint).
   Its `*_evaluation` profile reads `CLAS12_EVAL_DATA_ROOT`. Set pretrained mode
   to match the fit and use the exact same frozen checkpoint when applicable.
4. [Plot the training history from its artifact receipt](../TRAINING_RECIPES.md#5-inspect-output-behavior),
   then inspect final evaluator tables and the metric population counts.

Checkpoint selection uses **assignment-option-2 ARI** in the configured
`validation_ari_mode`, with a validation-loss tie-break. The preset uses signal
ARI. Selection averages eligible events across the full validation cohort.
No-object assignments count as background for both options. A run that reaches
its update or epoch cap validates the last update and must produce a finite
selected checkpoint to succeed. Option 2 combines mask probability with query-class confidence and excludes
no-object queries. Its point-assignment threshold is 0.2; IoU, purity and
efficiency matching cuts are each 0.5. Matching maximizes the number of pairs that pass all three
cuts, then their total IoU. Keep those definitions fixed when comparing
runs. Changing the threshold changes the efficiency/background operating point.

## Inspect behavior

| Output | Use |
| --- | --- |
| Training `artifacts.json` | Selected checkpoint, actual final step and `validation_history_dir`. |
| `metrics_vs_step.png/.pdf` | ARI together with purity, efficiency, fake/split/merge and background metrics. |
| `background_vs_signal_ari.png/.pdf`, `trend_summary.json` | Metric trade-offs across validations and corresponding snapshots. |
| Evaluation `summary.json`, `per_event_metrics.csv` | Overall behavior, accepted population and failing events. |
| `per_track_matches.csv`, `per_truth_track_metrics.csv`, `binned_metrics.csv` | Matching quality, lost tracks and kinematic dependence. |
| Optional `per_point_predictions.csv` | Detailed cluster assignments; enable `save_per_point_predictions` when needed. |

Final finding evaluation writes tables; validation plots are produced by the
separate history plotter. Neither substitutes for event/track inspection.
See [Evaluation and plots](eval/README.md) for metric definitions and controls.

`evaluate_coatjava: true` additionally requires the appropriate test segmentation
sidecar. The standalone template defaults it to false. The **campaign preflight**
has stricter sidecar requirements for both splits; read
[the campaign walkthrough](campaign/GETTING_STARTED.md#additional-finding-preflight-inputs)
before moving to campaigns.

## Next steps

Use [From single runs to campaigns](campaign/GETTING_STARTED.md) to repeat the
validated recipe across backbones, label counts or seeds. For both tasks on the
same backbone, use [the worked example](../WORKED_EXAMPLE.md). Reserve source
events explicitly for a final test: sample limits alone do not create a split.

## HDF5 AdapterOnly benchmark

This branch integrates track finding with the published HDF5 cluster product.
The shared data support is a separate ancestor commit; the other adapter
integrations are independent sibling branches.

```bash
python -m train.downstream.benchmark_hdf5_finding \
  --data-root /absolute/path/to/cvt-clusters-h5 \
  --output-dir /absolute/path/to/new/finding-run
```

Defaults: 5,000 accepted training samples, 3,000 updates, 1,000 validation
samples and 2,000 test samples, seed 42. Cohorts use accepted prefixes. The
runner checks disjoint source-event identities, reloads the validation-selected
checkpoint and writes configurations, cohort IDs, predictions and `summary.json`.
For a smoke test add `--train-size 32 --steps 2 --validation-size 16 --test-size 16`.
The configuration is `hdf5_finding_5k` in `scripts/configs/clas12_adapter_recipes.yaml`.

Samples are whole events. `primary_segment == -1`, including contested clusters,
remains background. The model, Hungarian targets and losses are unchanged.
Both methods use identical truth/masks; COATJAVA groups come from first-pass
`reco_segment`. Checkpoint selection uses validation signal ARI with a loss
tie-break; inference remains option 2 at threshold 0.2. Standalone evaluation
uses `test` by default and requires a target momentum scale of 1 for GeV/c.
Campaign analysis rendering selects this scale from the effective HDF5 reader,
including manifest and run training overrides. Explicit analysis overrides take
precedence; incompatible HDF5 momentum units are rejected by evaluation.
