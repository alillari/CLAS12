# Track-regression evaluation

Run the baseline analysis from any working directory:

```bash
python train/downstream/eval/evaluate_track_regression.py \
  --analysis-config train/downstream/eval/track_regression_analysis.yaml
```

The YAML controls the model configuration, checkpoint, unit conversions,
sample count, binning, and output location. Command-line overrides are
available for the checkpoint, output directory, and sample count.

The analysis writes:

- `predictions.csv.gz`: one row per evaluated track, including adapter output,
  truth, CVT/CVTRec/reconstructed-particle baselines, PID, charge, event/track
  identifiers, source file, and hit count;
- `summary.json`: global component, momentum, transverse-momentum, direction,
  resolution, and tail metrics;
- `binned_metrics.csv`: momentum, polar-angle, and PID differential metrics;
- `plots/`: residual and resolution plots.

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

The current CLAS12 dataset stores MC entrance momentum in MeV and auxiliary
reconstruction momentum in GeV. Those conversions are explicit in the YAML.
CUDA is required by the installed Mamba/causal-convolution forward kernels.

The comparison target is `MC::True` at the innermost CVT point. `CVT::Tracks`
is the fair first-fit baseline. `CVTRec::Tracks` is retained only as a
PID-corrected reference. The stored `REC::Particle` values are audited but
excluded from performance plots because they do not currently match the
corresponding `CVTRec::Tracks` rows as expected. `MC::Particle` is
generator-level and is not used as the post-energy-loss target.
