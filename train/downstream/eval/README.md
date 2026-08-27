# Track-regression evaluation

Run the baseline analysis from any working directory:

```bash
python train/downstream/eval/evaluate_track_regression.py \
  --analysis-config train/downstream/eval/track_regression_analysis_adapteronly.yaml
```

The YAML controls the model configuration, checkpoint, unit conversions,
sample count, binning, and output location. Command-line overrides are
available for the checkpoint, output directory, and sample count.

Training and evaluation configs use `CLAS12_ARTIFACT_ROOT` as the external
storage root. Set it once to move checkpoints, logs, inference outputs, and
plots together:

```bash
export CLAS12_ARTIFACT_ROOT=/path/to/storage/july-8th
```

The track-regression YAMLs derive paths from that root:

- training: `{artifact_root}/downstream_log/<run-name>/checkpoints`
- evaluation: `{artifact_root}/downstream_log/<run-name>/evaluation_output/<analysis-tag>`

If `CLAS12_ARTIFACT_ROOT` is unset, the configs keep using the current default
`/home/alessio/ML-work/result_deep_storage/july-8th`.

The analysis writes:

- `predictions.csv.gz`: one row per evaluated track, including raw adapter
  output, DOCA-space adapter output, raw innermost-hit truth, DOCA-space
  comparison truth, CVT/CVTRec/reconstructed-particle baselines, PID, charge,
  event/track identifiers, source file, and hit count;
- `summary.json`: global component, momentum, transverse-momentum, direction,
  resolution, tail, ML-regression, training-history, and data-consistency
  metrics;
- `ml_metrics.csv`: final evaluation-set MAE, RMSE, median absolute error,
  95th-percentile absolute error, and bias for Adapter, `CVT::Tracks`, and
  `CVTRec::Tracks`;
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

`plots/physics_2d/` contains side-by-side Adapter and `CVT::Tracks` 2D residual
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

The current CLAS12 dataset stores MC entrance momentum in MeV and auxiliary
reconstruction momentum in GeV. Those conversions are explicit in the YAML.
CUDA is required by the installed Mamba/causal-convolution forward kernels.

The training target is still `MC::True` at the innermost CVT point. For
evaluation, both `MC::True` and the raw Adapter output are swung back in
transverse direction to the DOCA/vertex frame before comparison with
`CVT::Tracks` and `CVTRec::Tracks`. The swingback keeps `pT` and `pz` fixed,
changes only wrapped phi, and uses the charge policy and field/radius
parameters declared in the YAML. Current datasets fall back to positive charge
when metadata is not available. `CVT::Tracks` is the fair first-fit baseline.
`CVTRec::Tracks` is retained only as a PID-corrected reference. The stored
`REC::Particle` values are audited but excluded from performance plots because
they do not currently match the corresponding `CVTRec::Tracks` rows as
expected. `MC::Particle` is generator-level and is not used as the
post-energy-loss target.
