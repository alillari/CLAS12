# CLAS12 momentum regression

The momentum adapter predicts a track's Cartesian entrance momentum
`(px,py,pz)` from its cluster sequence. The standard recipe uses supplied
MC-truth track membership; it does not automatically consume predictions from
the track-finding adapter. Begin with [environment/data setup](../../SETUP.md).

## Inputs and starting recipe

- Use aligned `features_pretrain/test`, `seg_target_pretrain/test` and
  `reg_target_pretrain/test` event products, with real momentum targets.
  Features are raw three-column xyz cluster centers.
- `adapter_sample_mode: event_segment` creates one sample per accepted truth
  segment. The standard filters accept **12–100 clusters/track**; this is a
  minimum of 12, not an exactly-12 requirement.
- `--eventnumber` counts **accepted tracks**, not source events. Multiple tracks
  from one source event must remain together when separating data populations.
- `clas12_momentum_50k` in
  [the adapter recipes](../../scripts/configs/clas12_adapter_recipes.yaml) uses
  50k training tracks, train/validation batch 128, 30k updates, validation every
  1k updates on 10k tracks, and seed 42.
- The starting head has width 256, one Mamba2 refinement layer with state 64,
  convolution 4 and expansion 2, zero input-embedder layers and attention pooling.
  The loss is standardized component MAE; checkpoint selection is minimum finite
  validation loss.

For AdapterOnly, omit `--usepretrain`. Frozen-feature fits require that flag and
`--pretrained_ckpt`, with architecture/embedding/klen matching the saved backbone.
The preset matches Mamba1 width 512/depth 12/state 16/k1. The head's Mamba2 family
is independent of that backbone choice. Mamba-family head width/count remain
fixed by the ordinary trainer; Transformer adapters expose additional controls.
See [Training options](TRAINING_OPTIONS.md#regression-target-and-head).

## Run and evaluate

1. [Set paths and copy the model recipe](../TRAINING_RECIPES.md#1-choose-a-task-and-prepare-the-run-configuration).
2. [Generate target statistics](../TRAINING_RECIPES.md#2-generate-momentum-target-statistics)
   from the exact accepted **training-track** population. Set
   `CLAS12_MOMENTUM_STATS` to that JSON. Check columns, count, mean/std and filters.
3. [Run the momentum command](../TRAINING_RECIPES.md#momentum).
   Explicitly pass `--train_batch_size 128`: the CLI defaults to 32 and overrides
   the YAML training batch. `train_momentum_regression.py` is a compatibility
   wrapper; `train_track_regression.py` is the shared public CLI.
4. [Copy/edit the analysis YAML and evaluate](../TRAINING_RECIPES.md#4-evaluate-the-saved-checkpoint).
   Set pretrained mode to match the fit, keep the matching stats and backbone,
   and select the intended final product with `CLAS12_EVAL_DATA_ROOT`.
5. [Inspect the generated plots](../TRAINING_RECIPES.md#5-inspect-output-behavior)
   and the evaluator's actual population/fit counts.

The stats JSON changes the target scale and relative component weighting in the
loss. Recompute it when task, selected tracks, filters or label budget change;
reuse it across models fitted on the same labels. Keep validation/evaluation
labels out of that calculation. Read [Target statistics](TARGET_STATISTICS.md)
for the formulas, sampling policy and checkpoint-buffer behavior.

## Inspect behavior

Momentum evaluation requires `aux_target_test` and `metadata.json` with the
declared auxiliary layout for its conventional entrance comparison. Current
MC momentum targets are in MeV and the auxiliary reconstruction momentum is
in GeV; the analysis scales are 0.001 and 1.0 respectively. Evaluation uses
entrance truth/output directly. The CVT reference combines `CVT::Tracks` p/theta
with the matched `CVT::Trajectory` entrance direction's phi.

| Output | Use |
| --- | --- |
| Training `artifacts.json`, log and adapter checkpoint | Actual updates, minimum-loss selection and the run's saved model/normalizer. |
| `summary.json`, `ml_metrics.csv`, `binned_metrics.csv` | Component/magnitude/direction errors, bias, tails and differential populations. |
| `plots/ml/training_curves.png` | Loss evolution; the current plot uses epoch coordinates, so consult the original step log for update timing. |
| `plots/ml/ml_error_bars_*.png`, `absolute_error_cdf.png` | Physical error metrics and tail behavior beyond the scalar loss. These bars are not seed uncertainties. |
| `plots/physics_2d/` | Kinematic dependencies and same-track adapter/CVT error comparisons. |
| `plots/delta_p_over_p_fits/`, `plots/delta_theta_fits/` and fit CSVs | Per-bin occupancy, trimming, fit status and Gaussian-core descriptions. |

The evaluator generates these momentum figures automatically. A narrow fitted
core does not imply small bias or tails; inspect them together. See
[Evaluation and plots](eval/README.md) for residual definitions, units, ranges
and fit controls, and [checkpoint policy](PHYSICS_CHECKPOINTS.md) for selection.

## Next steps

Use [From single runs to campaigns](campaign/GETTING_STARTED.md) to repeat a
working recipe and collate comparisons. [Optuna tuning](tuning/README.md) is a
separate optional search workflow; [probes](probes/README.md) diagnose frozen
representations and layer mixing. For the complete backbone plus both-adapters
sequence, use [the worked example](../WORKED_EXAMPLE.md).
