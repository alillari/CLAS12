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

### Adapter dropout

Set `downstream_dropout` explicitly to keep head regularization independent of
the pretrained backbone's `dropout`. It takes precedence in both the training
and evaluation head factories; omitting it preserves the regression trainer's
legacy fallback to `dropout` (or zero). New checkpoints record the dropout
settings and reject changes on training resume. Legacy regression checkpoints
used their saved `dropout` for the head even if `downstream_dropout` was present;
resume those with the original effective value. The resume-specific dropout
check is skipped for inference, which disables dropout; ordinary architecture
checks still apply.

### Layer-mixture learning rate

`adapter_feature_mixing_lr_ratio` optionally multiplies the learning rate of
the pretrained layer-mixture logits. Its default is `1.0`, preserving the
historical optimizer layout. For example, `100.0` lets the mixture adapt faster
while keeping the rest of the adapter on its configured schedule. The ratio
applies to both step and epoch schedules, is recorded in the checkpoint, and
must match when resuming training. Inference uses the saved weights directly.

### Coordinate-assisted pretrained readout

Set `pretrained_readout: coordinate` in the model YAML to retain the raw position
embedding alongside the frozen backbone features. The head applies separate
LayerNorm/linear projections to the raw embedding and the learned average of
backbone layers, adds the projected values, and passes them through the same
refinement and pooling. The feature projection starts at zero, so the initial
function and all common weights match AdapterOnly with the same adapter seed.
Its projection receives gradients immediately; the layer-mixing weights begin
learning once that projection moves away from zero.

The default `features` retains the existing pretrained feature-only readout.
Use the same readout setting for training, resume and evaluation; checkpoints
record it and reject mismatches or missing projection weights. In AdapterOnly
mode the feature branch is unused. With width-512 inputs and a width-256 head,
the coordinate option adds 132,352 parameters. It changes the architecture, so
include a matched raw-only capacity control when attributing gains to pretraining.

### Reproducible backbone encoding shapes

`backbone_feature_kind` explicitly selects the Mamba backbone's adapter inputs:
`accumulated_states` (default) uses each state after residual addition;
`residual_updates` captures each block output before addition. The latter requires
a backbone exposing `mamba_layers`. These policies have the same tensor shapes
but different meanings. Train and evaluate with the same policy; it is recorded
inside `backbone_encoding` and mismatches are rejected. Existing checkpoints
without this field default to accumulated states. If the original training code
or run records confirm that an unmarked adapter used residual updates, add both
settings to the model YAML used to evaluate it or resume compatible training:

```yaml
backbone_feature_kind: residual_updates
legacy_backbone_feature_kind: residual_updates
```

The legacy setting supplies only the missing feature policy. The trainer still
checks any saved padding and precision controls, and any explicit checkpoint
feature policy takes precedence. Evaluation leaves the checkpoint file intact.
Checkpoints written after a resume record the policy explicitly and no longer
require the legacy setting.

For explicit numerical qualification of Mamba3 checkpoints, use:

```yaml
backbone_sequence_padding: 128
backbone_batch_padding: 128
regression_matmul_precision: highest
```

The backbone sees fixed dimensions: shorter inputs receive the padding sentinel,
and larger evaluation batches are split into batches of the configured size.
The adapter receives only the original samples/tokens and their original masks.
Sequences exceeding the configured length are rejected. The `highest` option
disables TF32 and BF16 reduced-precision reductions during regression training
and evaluation, including for the adapter. The previous PyTorch settings are
restored after the operation. It does not change Mamba3's internal BF16
implementation or promise universal numerical invariance.

Defaults are zero padding controls and `default` precision, preserving the
existing forward shapes and process settings. Training checkpoints and artifact
summaries record these options; resume and evaluation require matching settings.
Use the same model YAML for both. A fixed shape avoids the observed dependence
on input/batch dimensions in the qualified Mike checkpoints, at additional
computation cost; check numerical parity when using other architectures.

## Run and evaluate

Run the momentum trainer with one process. To use a particular GPU, select
it with `CUDA_VISIBLE_DEVICES` before starting the training CLI.

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

### Mamba2 adapter initialization

Fresh momentum heads preserve the Mamba2 constructor's `dt_bias`, `A_log` and
`D`. The previous generic initializer overwrote all three, including setting
`dt_bias` to zero and changing the constructor's positive time-step range.
All other initialization rules and random-number consumption are unchanged.

Set `adapter_ssm_initialization: legacy` in the model YAML to reproduce the old
fresh-head initialization. The default is `constructor`. Checkpoints and training
artifact summaries record the initialization mode; unmarked historical
checkpoints are identified as `legacy` when loaded. Loading trained weights
does not reinitialize them, and checkpoint tensor shapes are unchanged.
This corrects an initialization overwrite; momentum gains still require a
matched comparison for each backbone and label budget.

Set `adapter_initialization_seed` to seed a fresh adapter immediately before
construction. This controls which random draws initialize the head when
comparing different backbone constructors with the same adapter recipe. The
training RNG state is restored before the first update.

## Inspect behavior

Legacy mmap evaluation requires `aux_target_test` and `metadata.json` with the
declared auxiliary layout for its conventional entrance comparison. In that
product, MC momentum targets are in MeV and auxiliary reconstruction momentum
is in GeV; the analysis scales are 0.001 and 1.0 respectively.

The `cvt-clusters-h5-v1` product stores momentum targets in GeV/c and supplies
the conventional reference through its HDF5 reader. Its evaluator uses scale 1
and requires saved GeV/c checkpoint evidence; the legacy mmap scale does not
apply. Evaluation uses entrance truth/output directly. The CVT reference
combines `CVT::Tracks` p/theta with the matched `CVT::Trajectory` entrance
direction's phi.

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
working recipe and collate comparisons. [Optuna tuning](tuning/GETTING_STARTED.md) is a
separate optional search workflow; [probes](probes/README.md) diagnose frozen
representations and layer mixing. For the complete backbone plus both-adapters
sequence, use [the worked example](../WORKED_EXAMPLE.md).

## HDF5 AdapterOnly benchmark

This branch integrates momentum regression with the published HDF5 cluster product.
The shared data support is a separate ancestor commit; the other adapter
integrations are independent sibling branches.

```bash
python -m train.downstream.benchmark_hdf5_momentum \
  --data-root /absolute/path/to/cvt-clusters-h5 \
  --output-dir /absolute/path/to/new/momentum-run
```

Defaults: 5,000 accepted training samples, 3,000 updates, 1,000 validation
samples and 2,000 test samples, seed 42. Cohorts use accepted prefixes. The
runner checks disjoint source-event identities, reloads the validation-selected
checkpoint and writes configurations, cohort IDs, predictions and `summary.json`.
For a smoke test add `--train-size 32 --steps 2 --validation-size 16 --test-size 16`.
The configuration is `hdf5_momentum_5k` in `scripts/configs/clas12_adapter_recipes.yaml`.

Samples are truth segments with at least 12 clusters. Training-only statistics
join primary particle IDs explicitly and use GeV/c; legacy MeV statistics or a
different dataset plan are rejected. Checkpoints use validation loss. COATJAVA
matches require a unique valid owner and exactly one first-pass candidate;
missing, ambiguous and invalid-entrance cases are recorded separately. The
reference uses same-pass p, theta and entrance phi rather than the DOCA vector.
The adapter uses truth cluster groups, so the paired momentum result is not an
end-to-end reconstructed-track comparison. All accepted truth tracks are also
reported for the adapter. Standalone evaluation defaults to the `test` split.

HDF5 evaluation infers particle class and electric charge from the selected truth
track's signed PDG ID using `fm4npp.particle_metadata.metadata_from_pdg`. It uses
valid PID fields on that truth segment, including when the input is an impure
COATJAVA group. Conflicting valid PIDs are rejected. Unknown IDs retain an unknown
charge and are counted separately, rather than becoming positive particles.
The helper covers the usual leptons/light hadrons and PDG nuclear codes; it is
not a complete particle database. Evaluation class indices remain pion=0,
kaon=1, proton=2, electron=3 and other=4, distinct from training classifier indices.
It can also be used directly: `python -m fm4npp.particle_metadata 211 -211 11 -11`.

Prediction rows retain dataset plan, split, source ID and original event ordinal.
The historical scalar residual plotter accepts these HDF5 identities as well as
legacy mmap metadata, and rejects cross-plan or cross-split joins. The compatibility
`eval_momentum_regression.py` CLI also defaults to held-out HDF5 test data; pass
`--evaluation-split validation` explicitly to reproduce a validation result.

Both evaluation CLIs validate checkpoint momentum units before loading weights.
HDF5 requires a checkpoint whose saved data contract or physics settings explicitly
declare GeV/c, with no conflicting unit declarations. MeV checkpoints and checkpoints
with unknown units are rejected, even when the current YAML points to GeV statistics.
Evaluate legacy MeV checkpoints with their original mmap configuration; changing the
stats file does not convert their saved normalizer or log-momentum output units.
