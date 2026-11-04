# Regression target statistics

The regression trainer requires a JSON file supplied through
`regression_target_stats`. It specifies the target columns and their training-set
means and standard deviations. The [training workflow](../TRAINING_RECIPES.md)
contains the generation command; this reference explains its meaning and options.

## What the file controls

For Cartesian momentum (`task: mom`), the head predicts standardized entrance
momentum components. For component `j`, the training target is

```text
z_j = (p_j - mean_j) / std_j
p_j = z_j * std_j + mean_j              # decoding into the stored target unit
```

The trainer applies the selected loss to standardized components. With MAE,
this weights a physical component error by `1 / std_j`; with MSE the weight is
`1 / std_j**2`. Changing the statistics changes the optimization problem and the
scale of validation loss. Use physical evaluation errors when comparing runs
whose normalizers differ.

The head stores `mean`, `std` and the constant-component mask as checkpoint
buffers. A component with `std <= 1e-8` is treated as constant: its standardized
target is zero and its decoded prediction is the mean. Inspect unexpected zero
standard deviations rather than assuming the data have useful variation.

This file standardizes **regression targets**. Input xyz-to-eta/phi/r conversion,
coordinate bounds and band-Hilbert geometry are separate loader/calibration
operations. Track-finding training does not need a regression-target stats file.
Backbone pretraining's coordinate MSE does not use this target normalizer either.

## How the generator samples and weights targets

`scripts/compute_regression_target_stats.py` scans source events in order and
accepted non-background segment labels in sorted order. It selects the first
accepted tracks up to `--limit-size`, matching the ordinary event-segment
loader's ordering with the same filters. Selected tracks consume the label
budget even when their momentum targets are partially or entirely missing.

Each track contributes **one target vector**, formed by averaging finite copies
of the target stored on its clusters. Long tracks do not receive more statistical
weight than short tracks. Statistics use population variance (`ddof=0`) across
these track vectors. For COATJAVA candidates, the target copies come from the
candidate's dominant matched truth segment after the purity requirement.

Each component uses the selected tracks with at least one finite copy of
that component. The JSON records the number of selected tracks in
`selected_events` and the contributing track counts in `count`. The helper
fails if a component has no finite observations in that cohort. New statistics
files use version 3. Regenerate older statistics for future runs if the training
cohort contains missing labels, since older generators could include later
tracks. For custom or random cohorts, compute statistics from the exact
selected track IDs.

## Command options

From the repository root:

```bash
python train/downstream/scripts/compute_regression_target_stats.py \
  --data-root "$CLAS12_DATA_ROOT" --split pretrain --task mom \
  --adapter-sample-mode event_segment --segment-target-source mctrue \
  --segment-min-clusters 12 --low-thr 1 --high-thr 100 --limit-size 50000 \
  --output "$CLAS12_MOMENTUM_STATS"
```

| Option | Effect and choice |
| --- | --- |
| `--data-root` | Root containing aligned `features_<split>`, `seg_target_<split>` and `reg_target_<split>` RaggedMmap trees. Required. |
| `--split` | Source split; default `pretrain`. Use the training split to avoid label leakage from validation/evaluation. |
| `--task` | Target transform/column selection. Use `mom` for Cartesian momentum; match the training YAML. Required for event-segment statistics. |
| `--adapter-sample-mode` | `event_segment`; one accepted segment per sample. |
| `--segment-target-source` | `mctrue` or `seg_target` reads truth segment labels; `coatjava` groups reconstruction candidates and requires `coatjava_seg_pred_<split>`. Match training. |
| `--segment-min-clusters` | Minimum segment size; default 12. |
| `--segment-exact-clusters` | Select exactly `segment_min_clusters` instead of at least that many; off by default. |
| `--low-thr`, `--high-thr` | Inclusive lower/upper cluster-count bounds on each segment. Defaults 1 and 100; the segment minimum also applies. |
| `--segment-min-truth-purity` | Dominant truth fraction required for a COATJAVA candidate; default 1.0. No additional effect for truth-defined segments. |
| `--limit-size` | Maximum accepted **tracks**, despite the output's `selected_events` field name. Omit to use all accepted tracks; match the labeled training budget when preparing a fixed-budget run. |
| `--output` | Destination JSON; parents are created. Set explicitly so its cohort/budget is identifiable. Without it, the helper uses `<data-root parent>/stats/` and a task/filter-derived filename. |
| `--chunk-size` | Accepted by the CLI, but does not control the event-segment scan. It is not a sampling or batch-size setting. |

For `pt_phi_eta` and `p_phi_theta`, phi is encoded as cosine/sine components;
the loader sets their effective mean/std to 0/1. Other components use the file's
statistics. Vertex tasks use their corresponding target columns. See
[training options](TRAINING_OPTIONS.md#regression-target-and-head) for task choices.

## Inspect the result before training

The JSON contains `task`, ordered `columns`, `filters`, `data_root`, `split`,
`selected_events`, per-column `count`, `mean` and `std`. For a valid 50k momentum
cohort, expect three columns and 50,000 finite target vectors:

```bash
python - <<'PY'
import json, math, os
with open(os.environ["CLAS12_MOMENTUM_STATS"]) as stream:
    stats = json.load(stream)
assert stats["task"] == "mom"
assert stats["split"] == "pretrain"
assert stats["columns"] == ["mc_entrance_px", "mc_entrance_py", "mc_entrance_pz"]
assert stats["selected_events"] == 50000
assert stats["count"] == [50000, 50000, 50000]
assert all(math.isfinite(x) for x in stats["mean"])
assert all(math.isfinite(x) and x > 1e-8 for x in stats["std"])
print("Target statistics:", stats["mean"], stats["std"])
PY
```

Adjust the expected count for your training budget. The loader validates column
compatibility and vector lengths; it does **not** automatically enforce the
recorded split, filters or cohort identity. That consistency is the run author's
responsibility. Statistics remain in the dataset's target units; unit conversion
to GeV belongs to evaluation, not this command.

## Reuse, regeneration and checkpoint handling

Reuse one stats file across AdapterOnly and frozen-backbone runs when their
task, labels, selected training tracks and filters agree. Recompute when any of
those change, including the label budget. Do not recompute from validation or
evaluation data. Keep validation size independent of the training label budget.

For evaluation, retain the training model YAML, its referenced stats file and
the adapter checkpoint together. The evaluator constructs the head using the
configured file and then loads the checkpoint's normalizer buffers. Replacing
the JSON does not renormalize existing trained weights. A matching column layout
alone does not establish that a file belongs to that checkpoint.

Implementation: [generator](scripts/compute_regression_target_stats.py),
[stats loading and target transforms](regression_utils.py), and
[normalizer buffers](model.py).
