# Momentum checkpoint policy

New momentum training selects the checkpoint with the lowest finite validation
loss. The baseline is center-only inputs, Cartesian `mom` targets, standardized
component MAE, attention pooling and the existing one-layer Mamba2 head.

```yaml
task: mom
input_representation: center_only
regression_loss: mae
pooling: attention
checkpoint_selection: validation_loss
```

Only improvements in validation loss replace the saved checkpoint. The early
stopping `min_delta` controls patience; smaller improvements still save better
weights. Final validation runs when an epoch cap precedes the next validation
interval. A run with no finite validation loss fails instead of advertising an
old checkpoint as a new result. Nonfinite predictions on valid targets fail
validation instead of disappearing through the regression-loss mask.
Optuna uses best validation loss and retains its
existing repeated-seed aggregation.

Validation defaults to 50,000 accepted samples with batch size 128, independently
of the training label budget. The final partial batch is retained. An explicit
batch cap cannot truncate the requested sample limit. This still uses the
configured test root as validation; subsequent evaluation is not automatically
an independent final test set.

## Physics diagnostics and historical results

Evaluation continues to report truth-binned central width `(q84-q16)/2`, median
bias and tail fraction. The diagnostic `physics_checkpoint` bin/units settings
remain readable, including metadata from historical checkpoints. These metrics
do not select new training checkpoints. Diagnostic selection/Pareto flags are
unavailable for new runs because training no longer writes an all-checkpoint
ledger. Campaign collation retains historical ledgers when present and keeps
their validation metrics separate from evaluation metrics.

Width-based training selection is retired. With unset bias/tail guardrails it
can reward narrow, systematically wrong predictions; joint-head momentum width
also says nothing about angular accuracy. Old configurations with
`checkpoint_selection: physics` are rejected for training. Historical center-only
and auxiliary-input model YAML remains usable for inference; the evaluator
disables its training policy and uses the standard task loss for diagnostics.

The offline `select_physics_checkpoint.py` script and `CheckpointSummary` reader
remain for inspecting/reselecting saved historical validation ledgers. They
require the original candidate weight files. Their availability does not make
width-only ranking a supported new training policy. Never reselect using the
evaluation population or compare widths without inspecting bias and tails.

The endpoint-geometry embedding, COATJAVA-reference resolution losses and
single-target training presets are retired. Existing position-plus-auxiliary
backbone contracts and historical scalar/coordinate target decoders remain.
The complete pre-cleanup implementation is preserved in Git at `b566b54`.
Endpoint-geometry checkpoints require that archived implementation.
