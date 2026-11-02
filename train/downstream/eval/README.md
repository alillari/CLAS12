# Downstream evaluation

Task entry points: [track finding](../TRACK_FINDING.md) and
[momentum regression](../MOMENTUM.md). This page defines their evaluator outputs
and diagnostic controls.

For runnable training/evaluation commands, start with
[Training workflow](../../TRAINING_RECIPES.md). Evaluation must use the saved
run's model configuration and matching backbone checkpoint. Reusing the
checkpoint-selection validation population is not an independent final test.

For standalone frozen-feature fits, use the named
[finding](track_finding_analysis_frozen_example.yaml) and
[momentum](momentum_analysis_frozen_example.yaml) analysis templates. The
[worked example](../../WORKED_EXAMPLE.md) supplies all paths and commands.
Campaign runners already render per-run analysis YAMLs automatically; analysis
is configured through YAML and run through the evaluator, without an inline
configuration-generation script.

## Choose the diagnostic

| Question | Tool/output | Interpretation |
| --- | --- | --- |
| Is tracking improving through training? | `plot_track_finding_validation.py`: `metrics_vs_step.png/.pdf` | Compare ARI with global purity/efficiency, fake/split/merge rates and background metrics on a fixed validation population. |
| Does better signal ARI trade away background performance? | Same tool: `background_vs_signal_ari.png/.pdf`, `trend_summary.json` | The report identifies adjacent validations with rising signal ARI and worsening purity/rejection/fakes; use the recorded checkpoint paths to inspect those states. |
| Which tracks/events fail? | Finding evaluator: `per_event_metrics.csv`, `per_track_matches.csv`, `per_truth_track_metrics.csv`, optional `per_point_predictions.csv` | Localize low efficiency, fake tracks, split/merge cases and cluster assignments. The per-run evaluator writes tables, not automatic event-display PNGs. |
| Does momentum training generalize across validations? | `plots/ml/training_curves.png`, `training_history.csv` | Inspect validation minima and train/validation separation. The current plot/parser uses epoch coordinates, so multiple step validations can share an x-coordinate; inspect the original `Step` log column for update timing. |
| Are physical errors acceptable beyond the loss value? | `plots/ml/ml_error_bars_components.png`, `ml_error_bars_kinematics.png`, `ml_metrics.csv` | Compare MAE/RMSE/upper-tail error/bias. These are metric bars, not seed uncertainties or confidence intervals. |
| Is improvement broad or confined to the core? | `plots/ml/absolute_error_cdf.png` and tail metrics in `summary.json` | Compare error quantiles and tails together. A narrower fitted core can coexist with worse tail behavior. |
| Is there a kinematic bias or acceptance dependence? | `plots/physics_2d/comparison_residual_*.png`, `binned_metrics.csv` | Residuals versus true kinematics expose offsets, trends and nonuniform spread; color is track count on a log scale. |
| How do individual track errors compare with CVT? | `plots/physics_2d/direct_error_adapter_vs_cvt_*.png` | Below the equal-error diagonal favors the adapter on that same track; use common valid populations. |
| Is a fitted resolution summary credible? | `plots/delta_p_over_p_fits/`, `plots/delta_theta_fits/` and matching fit CSVs | Inspect occupancy, trimming, fit status and tails before quoting fitted mean/sigma. |

Momentum evaluation generates its plot suite automatically. The
`training_curves.png` requires a readable, parseable training log. Finding
training histories are plotted separately; multi-run finding plots require
campaign collation:

```bash
python train/downstream/campaign/plot_track_finding_validation.py \
  --artifact-summary /path/to/finding/artifacts.json --output-dir /path/to/plots

python train/downstream/campaign/plot_track_finding_campaign.py \
  --campaign-dir /path/to/collated/campaign

python train/downstream/campaign/plot_track_regression_campaign.py \
  --campaign-dir /path/to/collated/campaign --plot-suite all
```

## Analysis configuration and plot controls

Training YAML controls the model/data contract; analysis YAML controls inference
population limits, assignment/metric choices, units and plot definitions.
Use the run's saved model profile and normalizer. Both evaluators accept
`--analysis-config`, `--checkpoint`, `--output-dir` and `--max-samples`.
The momentum evaluator additionally accepts model/profile/log/backbone overrides;
finding takes those fields from the analysis YAML.

| Analysis field / CLI control | Effect and user decision |
| --- | --- |
| `model_yaml`, `model_config`, `checkpoint` | Model profile and adapter weights. Architecture and target normalization must match training. |
| `use_pretrained_backbone`, `pretrained_checkpoint` | Enable and locate the matching frozen backbone; false for AdapterOnly. |
| `training_log`, `run_name`, `analysis_tag` | Training-history source and result labels. Set explicit log paths for loss plots. |
| `output_dir` | Evaluation destination; separate distinct cohorts/definitions to keep results identifiable. |
| `batch_size`, `num_workers` | Inference memory/throughput controls, independent of training batch. |
| `max_samples` | Accepted sample cap: events for finding, tracks for momentum. Finding also accepts `final_evaluation_events`, which takes precedence. Check actual counts in the output. |
| `evaluate_coatjava` | Finding baseline evaluation; true requires its segmentation sidecar. Set false for adapter-only diagnostics without that sidecar. |
| `save_per_point_predictions` / `--save-per-point-predictions` | Finding point assignments; off by default because this can produce a large table. |
| `assignment_option`, `assignment_threshold` | Finding assignment policy and rejection threshold. Use the same operating point across comparisons. |
| `match_iou_threshold`, `match_min_purity`, `match_min_efficiency` | Finding truth-match metric cuts. Changing them changes the reported task definition. |
| `momentum_bins_gev`, `pt_bins_gev`, `theta_bins_deg`, `pid_labels` | Differential result bins/labels; fields depend on the task. Check occupancy and compare with common binning. |
| `target_momentum_scale_to_gev`, `auxiliary_momentum_scale_to_gev` | Momentum unit conversions. Current MC targets use 0.001 and auxiliary momenta use 1.0. Finding uses the target scale for momentum-binned diagnostics. |
| `physics_histogram_bins`, `physics_plot_quantile` | Momentum 2D histogram granularity and robust display limits, default example 100 / 0.995. Display clipping does not redefine the CSV error metrics. |
| `delta_p_over_p_bins_gev`, `delta_theta_bins_gev` | True-momentum bins for relative-momentum/angular core fits. |
| `delta_*_min_bin_entries` | Minimum tracks before fitting a bin; default 200. Sparse bins receive explicit status instead of an accepted Gaussian fit. |
| `delta_*_histogram_bins`, `delta_*_min_populated_histogram_bins` | Fit histogram granularity and occupied-bin requirement; defaults 40 / 8. |
| `delta_*_fit_quantile` | Central residual fraction retained for the Gaussian fit; default 0.98. This estimates a trimmed core, not the full residual distribution. |
| `relative_error_tolerances` | Thresholds for reported fractions within a relative momentum error, e.g. 1%, 5%, 10%. |
| `include_sample_metadata`, `sample_metadata_jsonl` | Momentum metadata join; null path uses the test root's `samples_test.jsonl`. Inspect identity/join diagnostics. |
| `charge_source`, `fallback_charge` | Momentum differential labels; charge does not rotate or transport predictions. |

For the `delta_*` rows, the concrete prefixes are `delta_p_over_p` and
`delta_theta`. The overall width68 statistic is `(q84-q16)/2`; fitted sigma comes
from a Gaussian fit after quantile trimming. Neither quantity establishes small
bias or small tails. Fit-covariance errors are uncertainties of that fit, not
variation over independently trained models. The momentum residual histogram
shows a fixed -50% to +50% window; consult the full-distribution tables for tails
outside it. Angular residuals use the evaluator's wrapped azimuth convention.

Finding threshold sweeps use `--assignment-thresholds` (including 0.0), an
explicit output directory, hash-partition calibration controls and
`--min-efficiency-retention`. The tool selects on its calibration partition and
reports the held-out partition separately. Inspect `threshold_metrics.csv`,
`threshold_sweep.json` and `selected_heldout_summary.json`; selecting an operating
point from final test performance would reuse the test labels.

## Track-finding background policy

Track finding supports `track_target_mode: signal_only`. Background clusters
are negative members of every signal-track mask, never a separate query target.
The query-level no-object class and the point assignment threshold remain active.
Both `validation_ari_mode: signal` and `validation_ari_mode: inclusive` are supported;
inclusive ARI does not require a background query.

Training log columns `ARI_signal_option1` and `ARI_signal_option2` compare
assignment policies using signal ARI. With `validation_ari_mode: inclusive`,
the columns are `ARI_inclusive_option1` and `ARI_inclusive_option2` instead.
Option 1 uses mask probability; option 2 uses mask probability times query
class confidence and excludes no-object queries. Checkpoint selection uses
option 2 in the configured ARI mode. The callback key `val/ari` is that selected
score; explicit `val/ari_<mode>_option1` and `val/ari_<mode>_option2` keys report
both assignment policies. Older optimizer-step logs duplicated option 2 into
the option-1 column; this correction applies to new logs only.

The retired `unified_noise_instance` training mode and
`truth_joint_hungarian_qualified` evaluation mode raise errors when requested
through YAML, campaign overrides, CLI options, or adapter checkpoint metadata.
Explicit unsupported values are never silently converted to defaults. Old
configurations declaring `noise_attribution_mode: native` remain accepted; new
configurations can omit that field. Checkpoints without mode metadata retain
the historical signal-only/native default.

Evaluation and threshold sweeps now score only native predictions, without
truth-assisted query relabeling. New outputs use `metric_view: native` and stop
writing noise-attribution reports or duplicate `*_native.csv` files. The standard
CSV filenames remain. Summary JSON retains `native_metrics`, `native_baselines`,
and `native_comparisons` as aliases for existing campaign plots. Existing
checkpoints and campaign artifacts are not rewritten by this cleanup.

## Track-finding validation histories and later checkpoint selection

Every validation now aggregates the adapter's computed metrics for both
assignment options: signal/inclusive ARI, track purity/efficiency, matched
IoU/purity/efficiency, fake/miss/split/merge rates, background rejection,
contamination, and signal loss to background. Global track purity/efficiency
use pooled matched/predicted/truth counts. Other rates retain the evaluator's
event-average definitions, omitting undefined values. Counts and event-averaged
loss components are recorded too. No extra model forward pass is required.

The existing ARI-based selection, loss tie-break, and early stopping rule stay
unchanged. In particular, legacy selection averages batch-level ARI means;
the new full-validation ARIs average individual eligible events. These can
differ for an uneven final batch. History records keep `selection_score` and
`selection_aggregation` separate from the full-validation metrics.

Each training invocation creates a fresh directory beside its training log:

```text
<log-stem>_validation/run_<unique-id>/
  metadata.json
  metrics.jsonl
  metrics.csv
  checkpoints/validation_000001_step_000001000_epoch_000000.pth
  ...
```

The directory is printed and included as `validation_history_dir` in the
training artifact summary. CSV/JSONL rows include the step, epoch, learning
rate, selection result, fixed threshold, metric matching cuts, both options'
metrics, and the corresponding snapshot path. New invocations do not replace
previous histories. In distributed runs the history is written by rank zero
for its validation loader; it does not introduce cross-rank metric reduction.

`save_validation_checkpoints: true` is the default. Every validation retains
an immutable snapshot, including non-improving and early-stopping validations.
Snapshots contain adapter weights, any active LoRA weights, configuration,
and validation metrics. They omit optimizer/scheduler states and the frozen
backbone. They load through the existing evaluator with `--checkpoint`; they
are not training-resume checkpoints. The normal best checkpoint remains full.
Keep the matching backbone checkpoint and rendered analysis/model YAMLs.

To disable snapshots while retaining every metric record, set
`save_validation_checkpoints: false`, or pass `--no-save-validation-checkpoints`
when building a campaign. Snapshots consume disk space proportional to the
number of validation checks.

Plot one history without retraining or reevaluating:

```bash
PY=/home/alessio/miniconda3/envs/fm4npp/bin/python
export MPLCONFIGDIR=/tmp/matplotlib-cache
HISTORY_DIR=/path/to/checkpoints/run-name_validation/run_unique-id
"$PY" train/downstream/campaign/plot_track_finding_validation.py \
  --history "$HISTORY_DIR"
```

This writes PNG/PDF panels of metrics versus optimizer step and purity,
background rejection, and fake rate versus signal ARI. `trend_summary.json`
identifies the best snapshot for each metric and adjacent validation intervals
where signal ARI rises while purity/background performance deteriorates.
`--after-step 10000` restricts the analysis to later training;
`--min-delta 0.001` sets the absolute change required to flag a trade-off.
`--assignment-option 1` plots the membership-only diagnostic instead.
The tool rejects mixed histories, sample-count changes, and changes in
assignment threshold or metric matching cuts.
`--artifact-summary /path/to/artifacts.json` is an alternative to `--history`;
it reads `validation_history_dir` from the training receipt. A relative recorded
path is resolved against the receipt's directory. The options are mutually
exclusive.

For later evaluation of a snapshot:

```bash
"$PY" train/downstream/eval/evaluate_track_finding.py \
  --analysis-config /path/to/run/config/analysis.yaml \
  --checkpoint /path/to/history/checkpoints/validation_000010_step_000010000_epoch_000003.pth \
  --output-dir /path/to/separate/snapshot_evaluation
```

The plots describe a validation trajectory, not proof of causality or an
independent test result. Keep training/data/threshold fixed when comparing
selection rules, and use a genuinely disjoint sample for final confirmation.
The current selection rule can stop training early; set an adequate step/epoch
budget and patience if the goal is to observe later degradation. Historical
ARI-only logs cannot reconstruct discarded metrics or overwritten checkpoints.
Momentum-binned and COATJAVA comparisons still come from the final evaluator.

## Track-regression evaluation

This evaluator initializes the shared trainer and opens both the model YAML
training root (`data_root`) and evaluation test root (`data_root_test`). Keep
the original training trees accessible, even for checkpoint-only evaluation.
Use a separate final `data_root_test`; retain the training root and saved stats.

From the repository root, run the standalone adapter-only campaign example
(edit its paths to match an existing run first):

```bash
python train/downstream/eval/evaluate_track_regression.py \
  --analysis-config train/downstream/eval/track_regression_analysis_adapteronly.yaml
```

For normal campaign execution, use the per-run
`<campaign_dir>/runs/<run_id>/config/analysis.yaml` generated by
`train/downstream/campaign/run_track_regression_campaign.py`. Those rendered
configs carry the exact run checkpoint, model YAML, training log, output
directory, and pretrained-backbone flag for the campaign row.

The YAML controls the model configuration, checkpoint, unit conversions,
sample count, binning, and output location. Command-line overrides are
available for the checkpoint, output directory, and sample count.

The checked-in YAMLs use the current campaign directory structure. Set
`CLAS12_ARTIFACT_ROOT` or `CLAS12_CAMPAIGN_NAME` to point the standalone
examples at a different campaign:

```bash
export CLAS12_ARTIFACT_ROOT=/path/to/result_deep_storage
export CLAS12_CAMPAIGN_NAME=campaign_3_cluster_track_regression
```

The track-regression YAMLs derive paths from that root:

- campaign: `{artifact_root}/campaigns/<campaign-name>`
- run config: `<campaign-dir>/runs/<run-id>/config/analysis.yaml`
- checkpoint: `<campaign-dir>/runs/<run-id>/checkpoints/<run-id>_adapter_checkpoint.pth`
- evaluation: `<campaign-dir>/runs/<run-id>/evaluation`

Pretrained standalone evaluation also needs `CLAS12_PRETRAIN_ROOT` when the
pretrained checkpoint root differs from the checked-in example.

After per-run evaluations finish, campaign collation writes:

- `<campaign-dir>/summary/campaign_headline_metrics.jsonl`
- `<campaign-dir>/summary/run_table.csv`
- `<campaign-dir>/summary/delta_p_over_p_fits.csv` when per-run fit files exist
- `<campaign-dir>/summary/delta_theta_fits.csv` when per-run fit files exist

The analysis writes:

- `predictions.csv.gz`: one row per evaluated track, including entrance adapter
  output, innermost-hit MC truth, the `cvt_entrance_*` benchmark, explicitly
  labelled `legacy_*` auxiliary vectors (diagnostic only), PID, charge,
  event/track identifiers, source file, and hit count;
- `summary.json`: global component, momentum, transverse-momentum, direction,
  resolution, tail, ML-regression, training-history, and data-consistency
  metrics;
- `ml_metrics.csv`: final evaluation-set MAE, RMSE, median absolute error,
  95th-percentile absolute error, and bias for Adapter and the CVT entrance
  benchmark;
- `ml_metrics_summary.json`: the same ML metrics in nested JSON form;
- `campaign_headline_metrics.jsonl`: flat, campaign-friendly headline rows
  with run metadata, RMSE/MAE/bias/tail metrics, and component R² values;
- `training_history.csv`: parsed epoch, train-loss, validation-loss, and
  epoch-time values when the checkpoint training log is available;
- `binned_metrics.csv`: momentum, polar-angle, and PID differential metrics;
- `delta_p_over_p_fits.csv` and `delta_theta_fits.csv`: Gaussian residual fits
  in configured true-momentum bins for detailed momentum and polar-angle
  resolution plots;
- `plots/`: residual and resolution plots.

`plots/ml/` contains ML-oriented diagnostics:

- `training_curves.png`: train and validation loss versus epoch, inferred from
  the checkpoint sibling log unless `training_log` is set explicitly in the
  YAML;
- `ml_error_bars_components.png`: component-space `px`, `py`, `pz` error
  metrics;
- `ml_error_bars_kinematics.png`: derived `p`, `pT`, `theta`, wrapped `phi`,
  and `eta` error metrics;
- `absolute_error_cdf.png`: cumulative absolute-error distributions, which
  make the median and tail behavior directly comparable across methods.

`plots/physics_2d/` contains side-by-side Adapter and CVT entrance benchmark 2D residual
figures for spherical momentum `p`, cylindrical transverse momentum `pT`,
`theta`, wrapped `phi`, and pseudorapidity `eta`. Both methods use identical
axes and color normalization. For every reconstructed quantity, its residual
is plotted against all five true kinematic variables, so cross-dependencies
such as momentum resolution versus `theta` or `eta` are visible.

The same directory contains direct per-track error comparisons. Their x-axis
is the absolute CVT error, their y-axis is the absolute Adapter error, and the
diagonal means equal performance. Tracks below the diagonal favor the Adapter;
the annotated percentage reports how often that occurs.

`plots/` also contains detailed one-dimensional fit summaries:

- `delta_p_over_p_vs_true_p.png`, `delta_p_over_p_mean_vs_true_p.png`, and
  `delta_p_over_p_sigma_vs_true_p.png` fit `(p_reco - p_true) / p_true` in
  true-momentum bins and display the mean/sigma as percentages;
- `delta_theta_vs_true_p.png`, `delta_theta_mean_vs_true_p.png`, and
  `delta_theta_sigma_vs_true_p.png` fit `theta_reco - theta_true` in the same
  style of true-momentum bins and display the mean/sigma in degrees.

`plots/delta_p_over_p_fits/` and `plots/delta_theta_fits/` retain one diagnostic
PNG for every fitted method and true-momentum bin. Each image shows the full
finite-residual distribution on a log-count scale, the quantile boundaries,
the exact trimmed histogram supplied to `curve_fit`, its Gaussian overlay, and
the saved fit status, sample counts, mean, sigma, and covariance errors. Sparse
or failed fits are also written so missing summary points remain auditable.

The current CLAS12 dataset stores MC entrance momentum in MeV and auxiliary
reconstruction momentum in GeV. Those conversions are explicit in the YAML.
CUDA is required by the installed Mamba/causal-convolution forward kernels.

All momentum evaluation now uses `MC::True` and raw adapter output at the
innermost matched CVT hit. There is no DOCA transport or charge-dependent
rotation. The stored conventional benchmark uses `cvt_benchmark_p` and
`cvt_benchmark_theta` from `CVT::Tracks`, and `cvttraj_entrance_phi` from
`atan2(cy,cx)` of the selected same-track `CVT::Trajectory` entrance row.
The evaluator reads these fields by name from `metadata.json:aux_target_layout`
and constructs the comparison vector using that entrance phi. It does not use
the trajectory bank's diagnostic `phi` field or `CVT::Tracks.phi0`.

This is explicitly a p/theta/entrance-phi benchmark, not a claim that all three
values come from `CVT::Trajectory`. The current 20-column product stores the
full trajectory momentum only in separate diagnostics. Missing entrance phi
produces a missing CVT benchmark, with no DOCA fallback. Row selection tests
only the three entrance benchmark columns, so missing legacy values cannot
select a different comparison row. Momentum scaling applies to p only;
theta and phi remain radians when constructing the vector.

The `cvt` method key is retained for campaign compatibility. Legacy
`CVTRec::Tracks` and `REC::Particle` vectors have no entrance-phi counterpart
in this product and are excluded from all performance metrics. Their stored
vectors, and the original CVT vector, are exported only under `legacy_*` names.
`MC::Particle` remains generator-level diagnostic information. Charge is used
only for differential labels; a fallback charge never changes a prediction.

### Evaluation contract

Momentum summaries and fit tables carry
`evaluation_contract: clas12_momentum_entrance_v1` and
`comparison_truth: mctrue_inner_hit`. The evaluator uses entrance predictions
and truth directly. Analysis settings requesting DOCA/swingback are converted
to this contract with a notice. Collation requires matching contracts.

To refresh evaluations without retraining, then regenerate campaign plots:

```bash
python train/downstream/campaign/run_track_regression_campaign.py \
  --manifest /path/to/campaign/manifest.yaml --skip-train --force-eval
python train/downstream/campaign/plot_track_regression_campaign.py \
  --campaign-dir /path/to/campaign --plot-suite all
```
