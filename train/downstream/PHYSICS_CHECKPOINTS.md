# Physics checkpoint selection

Momentum/angle runs now default to `checkpoint_selection: physics`. Training loss
and validation loss remain diagnostic quantities. Validation saves **every**
evaluated checkpoint, then publishes the selected weights at the usual checkpoint
filename. Early stopping tracks changes in the physics selection, with the existing
patience/warmup controls. No acceptable checkpoint means no published checkpoint:
standalone training exits with an explicit error **after** writing the reports.
There is no fallback to minimum loss or to a rejected candidate.

Without configured guardrails, the default is **provisional selection by width**.
Reports explicitly say `selected_provisional_guardrails_disabled`. Partially
configured limits are enforced and labeled `selected_provisional_partial_guardrails`.
Set `require_configured_guardrails: true` to require all three limits.
COATJAVA does not set any limit or influence selection.

## Configuration

Set these keys in the model YAML configuration (or its inherited defaults):

```yaml
checkpoint_selection: physics
physics_checkpoint:
  residual: p_relative
  bin_quantity: p
  bin_edges: [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
  momentum_scale_to_gev: 0.001  # native momentum targets are MeV
  tail_threshold: 0.10         # strict abs(residual) > threshold
  min_bin_entries: 200
  min_valid_bins: null         # default: ALL configured bins
  guardrails:
    B_macro: null              # maximum macro absolute median residual
    B_worst: null              # maximum absolute median of any valid bin
    T_macro: null              # maximum macro tail fraction
  require_configured_guardrails: false
  width_tie_atol: 1.0e-6
  width_tie_rtol: 0.0
```

The example shows the momentum defaults. Choose the studied truth range **before
training**, using validation support; sparse studies may need wider bins or an
explicit smaller `min_valid_bins`. Out-of-range and invalid-truth counts are
reported. Inclusive diagnostics also include finite out-of-range residuals.
Never adapt edges using checkpoint predictions. Bins are `[low, high)` except the
last bin includes its upper edge.

If `residual`, `bin_quantity`, `bin_edges`, and `tail_threshold` are omitted:

| Task | Residual | Truth bins | Tail threshold |
|---|---|---|---|
| `mom`, `momentum`, `p`, `p_phi_theta`, `pt_phi_eta` | `(p_pred-p_true)/p_true` | 0.25 to 3 GeV, 0.25 GeV spacing | 0.10 (fraction) |
| `theta` | `theta_pred-theta_true` | 6 equal bins from 0 to pi | 0.05 rad |
| `phi` | `atan2(sin(phi_pred-phi_true),cos(phi_pred-phi_true))` | 12 equal bins from -pi to pi | 0.05 rad |

All angles and residuals refer to the entrance, with no transport to DOCA.
For a single-target angular adapter, `bin_quantity: p` is supported using the
underlying raw MC truth momentum. Joint adapters can select on `theta` or
`phi_wrapped` instead of `p_relative`. `p_absolute` is also supported, in GeV.
The `p` head's log output is decoded to physical momentum for resolution;
its training loss remains Huber on the log difference. These metrics do not
change any training objective.

## Statistics and selection

Quantiles use NumPy linear interpolation: each valid bin reports `q16`, `q50`,
`q84`, `w68=(q84-q16)/2`, `abs_median_bias=abs(q50)`, and `tail_fraction`.
A bin needs at least `min_bin_entries` validation tracks. Every valid bin has
weight 1 in `W_macro`, `B_macro`, and `T_macro`; `B_worst` is the maximum absolute
median. Bin occupancy is truth-derived. Nonfinite predictions invalidate the
checkpoint; they cannot silently remove difficult tracks/bins from ranking.
Nonpositive predicted physical p is also invalid.

Selection rejects nonfinite summaries/loss, insufficient valid bins and any
configured bias/tail violation. Among remaining checkpoints it finds the minimum
`W_macro`. Widths within `max(width_tie_atol, width_tie_rtol * abs(minimum))` of
that minimum are tied, then ordered by `B_macro`, `B_worst`, `T_macro`, width,
and finally earliest step/epoch. There is no weighted score.

The exact four-objective Pareto frontier includes all summary-valid checkpoints,
including ones that fail guardrails. Equal metric vectors both lie on the frontier.
Invalid summaries are excluded. Validation truth values and order are hashed and
must remain identical; compared checkpoints must have identical valid-bin support.
Validation defaults to 50,000 accepted samples with batch size 128, independently
of the training sample/batch sizes. Set `limit_test_size` and `valid_batch_size`
to override these defaults. `max_val_batches` defaults to null; an explicit cap
that would truncate the requested sample is rejected. Before training, a truth-only
pass writes `validation_support.json` and rejects insufficient bin occupancy.
The validation loader retains the final partial batch. Single-process validation is currently required;
DDP physics selection raises an explicit error rather than using per-rank quantiles.

## Artifacts

Each execution creates a unique `<checkpoint_stem>_physics_*/` directory beside
the usual checkpoint file, containing:

- `step_*_epoch_*.pth`: all validation checkpoints, including rejected ones.
- `checkpoint_summary.json`: resolved configuration, every per-bin and inclusive
  statistic, step/epoch/loss, each guardrail's enabled/pass status, eligibility,
  rejection reasons, Pareto/selection flags, and best candidates if none passed.
- `checkpoint_summary.csv` and `.md`: compact checkpoint tables.
- `checkpoint_metrics_by_step.png` and `.pdf`: loss, four physics quantities,
  and valid-bin count versus optimizer step; selected point and guard limits marked.
- `pareto_frontier.json` and `checkpoint_pareto.png`/`.pdf`: frontier and pairwise
  views of the four-dimensional tradeoff.

The run's `*_artifacts.json` links the summary and records selection status,
selected metrics, and the separate diagnostic minimum-loss step/epoch.
Existing checkpoint aliases are archived into `previous_selected.pth` before a
new run selects anything. An epoch cap before the first validation interval now
triggers final validation. Resumed training starts a new comparison history, since
the old validation sample may not be reproducible; prior run reports are retained.
This costs more disk space than saving only the best weights (optimizer and
scheduler states are retained for each checkpoint).

Both evaluation entrypoints additionally write `physics_checkpoint_summary.json`
using the checkpoint's saved metric configuration where available. These are
**evaluation diagnostics only** and never select weights using the evaluation
sample. `selected`/`on_pareto_frontier` come from the original validation ledger;
they are null when that ledger is unavailable. Existing inclusive physics/ML
reports remain. Old checkpoints can be evaluated with the current YAML's metric
configuration, but their missing historical validation summaries cannot be recovered
from a single selected weight file.

Optuna physics-mode runs optimize the selected `W_macro` (with the configured seed
aggregation), report eligible widths for pruning, and prune runs with no acceptable
checkpoint. Diagnostic seed losses remain recorded separately. The study contract
includes the resolved physics policy, so old loss-based studies cannot be silently
mixed with new physics-based trials. To reproduce the legacy selection and Optuna
objective explicitly set `checkpoint_selection: validation_loss`; legacy training
does not produce an all-checkpoint physics ledger.

## Reselect without retraining

Supply measured limits to the saved validation summaries. Omitted limits retain
the original settings; binning, residuals, sample and occupancy remain fixed.
The output directory must be new. The source checkpoints/reports stay unchanged.
The following limits are examples of CLI syntax, **not recommended physics cuts**:

```bash
/home/alessio/miniconda3/envs/fm4npp/bin/python \
  train/downstream/scripts/select_physics_checkpoint.py \
  --summary /path/to/run_physics_id/checkpoint_summary.json \
  --output-dir /path/to/new_selection \
  --b-macro 0.01 --b-worst 0.02 --t-macro 0.10
```

This saves revised reports, plots, and `selected_checkpoint.pth`. If none passes,
it saves the reports/best candidates, exits nonzero and creates no selected file.
Original checkpoint files must remain available to copy newly selected weights.
