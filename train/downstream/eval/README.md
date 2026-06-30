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

The current CLAS12 dataset stores MC entrance momentum in MeV and auxiliary
reconstruction momentum in GeV. Those conversions are explicit in the YAML.
CUDA is required by the installed Mamba/causal-convolution forward kernels.
