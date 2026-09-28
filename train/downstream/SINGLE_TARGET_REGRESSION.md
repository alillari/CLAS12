# Single-target momentum-adapter ablation

Set `task: p`, `task: theta`, or `task: phi`, with `regression_loss: huber`.
Each task constructs a **one-output head** and selects only its own truth label.
The existing `mom`, `pt_phi_eta`, and `p_phi_theta` tasks retain their objectives.

| Task | Head output / training target | Loss, Huber delta = 1 |
| --- | --- | --- |
| `p` | `log(p)` in the dataset's momentum unit | `Huber(log(p_pred) - log(p_true))` |
| `theta` | polar angle in radians | `Huber(theta_pred - theta_true)` |
| `phi` | azimuth in radians | `Huber(atan2(sin(phi_pred - phi_true), cos(phi_pred - phi_true)))` |

For `p`, decode the output with `p_pred = exp(output)`. This parameterization
keeps momentum positive and avoids taking the log of an unconstrained momentum
prediction. Changing momentum units adds a constant to both log targets and
predictions; the loss residual is unchanged. There is no division by target
standard deviation or by a conventional-reconstruction resolution. The loaded
normalization is identity for all three tasks, even if the stats file reports
nonzero means or nonunit standard deviations. `theta` is not wrapped. `phi`
has one scalar output, rather than the joint modes' cos/sin pair.

Zero momentum is invalid for `p` and `theta`; zero transverse momentum is
invalid for `phi`. Nonfinite labels are masked. Repeated phi labels within a
selected segment use a circular mean. Checkpoint task metadata distinguishes
the three one-output heads, preventing cross-task loading.

## Presets and training

Both `scripts/configs/mamba_clas12_track_regression_adapteronly.yaml` and
`scripts/configs/mamba_clas12_track_regression_pretrained.yaml` provide:

```text
clas12_track_regression_adapteronly_{p,theta,phi}_only
clas12_track_regression_pretrained_{p,theta,phi}_only
```

They inherit their respective base settings, including the v6_02 dataset and
12-cluster minimum. For another data product or geometry configuration, copy
the relevant existing config and change `task`, `regression_loss`, and
`regression_target_stats`. Match the dataset, segment filters, splits, seed,
optimizer, and training budget across the ablation runs.

Generate task-specific statistics from the same training split and filters.
The `p` stats column is `mc_entrance_log_p`; raw-p or Cartesian stats cannot
substitute for it. From the repository root, for the checked-in presets:

```bash
PY=/home/alessio/miniconda3/envs/fm4npp/bin/python
DATA_ROOT=/home/alessio/mnt/edgexpert-a02c/ML-work/HIPO_processing/multi_particle_storage/big-data/mmap_canonical_loose_truthseg_event_v6_02
for TARGET in p theta phi; do
  "$PY" train/downstream/scripts/compute_regression_target_stats.py \
    --data-root "$DATA_ROOT" --split pretrain --task "$TARGET" \
    --adapter-sample-mode event_segment --segment-target-source mctrue \
    --segment-min-clusters 12
done

"$PY" train/downstream/train_track_regression.py \
  --yaml_config scripts/configs/mamba_clas12_track_regression_adapteronly.yaml \
  --config clas12_track_regression_adapteronly_p_only \
  --eventnumber 50000 --train_batch_size 128 --run_num 0
```

Replace `p_only` with `theta_only` or `phi_only` for the other targets.
For a frozen pretrained backbone, select the matching pretrained YAML/preset
and supply `--usepretrain --pretrained_ckpt /path/to/backbone-checkpoint`.

The campaign manifest builder accepts these presets through
`--adapter-only-model-config` and `--pretrained-model-config`. Use a distinct
campaign directory per target. If applying the axis through training overrides,
set all three of `task`, `regression_loss=huber`, and `regression_target_stats`.
An inherited MAE/MSE setting is rejected for these tasks to prevent accidentally
training a different objective.

## Evaluation

Use `train/downstream/eval/evaluate_track_regression.py` with the same model
YAML, preset, and trained adapter checkpoint. For example:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache "$PY" train/downstream/eval/evaluate_track_regression.py \
  --model-yaml scripts/configs/mamba_clas12_track_regression_adapteronly.yaml \
  --model-config clas12_track_regression_adapteronly_p_only \
  --checkpoint /path/to/adapter-checkpoint.pth \
  --output-dir /path/to/p-only-evaluation
```

Single-target evaluation writes predictions, `summary.json`, ML metric tables,
campaign headline rows, and native/physical error plots. It evaluates the
selected quantity at the innermost matched CVT hit. It does not reconstruct a
full momentum vector, perform a DOCA swingback, or produce conventional-track
comparisons or full-vector resolution plots. All regression evaluation now
uses the entrance-only contract, including when loading a legacy analysis
YAML. Summary metadata records `evaluation_contract`, the effective comparison
truth, and `swingback_enabled: false`.

Momentum is decoded and reported in GeV using the analysis configuration's
`target_momentum_scale_to_gev`. Native momentum metrics use log residuals;
theta and wrapped phi errors use radians. `summary.json` also records the
sample-averaged Huber objective. Compare each single-target run against the
matching quantity in joint-target evaluations; losses across different tasks
have different meanings.

Focused CPU verification:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache "$PY" -m unittest \
  train.downstream.tests.test_single_target_regression \
  train.downstream.tests.test_regression_targets \
  train.downstream.tests.test_event_segment_dataset \
  train.downstream.tests.test_evaluate_track_regression_segment_identity
```

Actual Mamba forward/training and checkpoint evaluation require CUDA.
