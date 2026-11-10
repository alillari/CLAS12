# Saved-prediction calibration and event diagnostics

These CPU tools operate on complete events and fixed model checkpoints. They
support arbitrary model names and cohort sizes. They require NumPy, SciPy and
scikit-learn; plotting also uses Matplotlib. Neither tool imports the trainer or
runs inference.

## Select an assignment threshold

```bash
python -m train.downstream.eval.calibrate_track_finding \
  --validation /path/to/validation_predictions.npz \
  --metadata /path/to/validation_metadata.npz \
  --test /path/to/test_predictions.npz \
  --test-metadata /path/to/test_metadata.npz \
  --truth-policy primary_mc_min4_v1 \
  --objective dice \
  --output-dir /path/to/new_dice_calibration
```

Omit the metadata options when the prediction archives already contain the
required metadata. Omit both test options for calibration without held-out
reporting. Run once per model; no model tiers or training sizes are assumed.

The two supported objectives are:

- `dice`: maximize `mc_track_dice_mean`, including zero contributions from missed
  eligible truth tracks. Ties within `1e-12` prefer higher pooled completeness,
  then purity, then the lower threshold.
- `quality-efficiency`: maximize pooled track efficiency with both per-pair
  purity and completeness cuts. For example, add
  `--objective quality-efficiency --min-purity 0.9 --min-completeness 0.9`.
  Matching is recomputed at every threshold; all eligible truth tracks remain
  in the denominator. The default IoU cut is 0.5. Ties prefer Dice within
  `1e-12`, completeness, then the lower threshold. The cuts constrain each
  accepted pair rather than global hit averages.

The default search scans 0 through 1 in increments of 0.01, then scans 21 points
around the current best at steps 0.001 and 0.0001. Supply `--thresholds 0.2 0.3
0.4 0.5` to replace the initial grid; `--refine-steps` with no values disables
refinement. Results are the **best scanned** operating points, without a claim
of continuous global optimality. Pure hit-purity maximization is deliberately
excluded because it can favor almost-empty predictions.

The raw gate is strictly `signal & (scores > threshold) & (pred >= 0)`.
Inactive/no-object candidates may have negative scores; they stay excluded.
Cached contingency-table results are checked against the shared evaluator at
the selected point. Regression tests cover other thresholds, sparse IDs,
matching ties, empty events and asymmetric quality cuts.

Outputs in a **new** directory are `validation_scan.json`, `selection.json` and
`summary.json`. The validation choice and its receipt are persisted before test
predictions are opened; its hash is checked after reporting. When test data are
supplied, validation and test must carry physical identities and must be disjoint.
If held-out validation fails, the selection remains available and no successful
summary is written. Existing outputs are never overwritten.

## Archive contract

Every NPZ uses one-dimensional flat arrays plus complete-event offsets:

| Field | Contract |
| --- | --- |
| `truth` | Integer MC track IDs; background/non-target = -1 |
| `pred` | Integer raw query winners, or explicitly declared final assignments |
| `offsets` | Integer array starting at 0, monotone, ending at number of points |
| `scores`, `signal` | Required for raw candidates: finite scores and boolean native gate |
| `source_id`, `original_ordinal` | One integer per event; together form its unique physical identity |
| `point_ids` | Unique original cluster indices within each event |
| `provenance` | Optional shared `ClusterProvenance` integer codes |
| `xyz` | Optional finite `(n_points, 3)` saved cluster centers in cm |
| `endpoints` | Optional finite `(n_points, 6)` measurement-support endpoints in cm |
| `detector`, `layer` | Optional aligned integer arrays; CLAS12 detector 1 = BMT, 2 = BST |

An aligned metadata NPZ must include exactly matching `truth` and `offsets`.
Duplicate fields must agree; metadata cannot override contradictory archive
contents. The existing `verified_point_identity.npz` study sidecar can therefore
be supplied directly. The tool applies an explicit `primary_mc_min4_v1` projection
itself; it never trusts an unversioned precomputed `eligible_truth` field.
`as_saved` is the default policy. Projection counts support in each full event;
cropped/padded arrays are invalid inputs. If original point IDs are supplied for
projection, they must be a full permutation of `0..n_event_points-1`.

For archived HDF5 scalar provenance, `background_only` with optional `ownership`,
`origin_noise` and `purity` can replace `provenance` codes. The non-target label
alone never establishes overlay origin. Without provenance, overlay counts are
null and the overlay-failure category is unavailable.

No spatial join is performed. To export HDF5 geometry, look up each event by
`source_event_index`, verify `source_id` and `original_ordinal`, and reorder
`clusters.xyz`, `clusters.endpoints`, `clusters.detector` and `clusters.layer`
using the archived original `point_ids`. Save them under the names above. Preserve
the original truth, full-event offsets and physical identities in that sidecar.

## Compare models and select failure examples

Create a JSON manifest; relative paths resolve beside this file:

```json
{
  "truth_policy": "primary_mc_min4_v1",
  "baseline": {
    "name": "COATJAVA",
    "path": "reference.npz",
    "kind": "final",
    "prediction_key": "coatjava",
    "metadata": "reference_metadata.npz"
  },
  "models": [
    {
      "name": "My adapter",
      "path": "test_predictions.npz",
      "metadata": "test_metadata.npz",
      "kind": "raw",
      "assignment_threshold": 0.3399
    }
  ]
}
```

```bash
python -m train.downstream.eval.compare_track_finding_events \
  --manifest /path/to/comparison.json --focus-model 'My adapter' \
  --max-examples 4 --min-purity 0.9 --min-completeness 0.9 \
  --output-dir /path/to/new_event_comparison
```

Add any number of named models. Every input must carry physical event identities
and original point IDs; the tool aligns different event orders and point
serializations, then verifies truth and any shared provenance/geometry fields.
Supply geometry and provenance in the baseline archive or its metadata sidecar.
Every spec explicitly declares `kind: raw` with a threshold, or `kind: final`
without one. This avoids accidentally interpreting ungated query winners as
final tracks. `prediction_key` defaults to `pred`.

The baseline can be any reconstruction. It must pass the configured quality
cuts on a track while the focus model fails. Categories are `overlay`,
`fragmentation`, `mixing`, `miss` and `incomplete`; restrict them with
`--categories`. Fragmentation requires at least two predictions containing at
least two target clusters each. Mixing counts attached clusters from other
eligible MC tracks. A miss means no Dice association, which can coexist with
some target clusters being assigned to a prediction associated with another
truth track. Incompleteness includes clusters rejected or placed elsewhere.
Categories can overlap.

Candidates rank by baseline-minus-model track Dice; category order is cycled
to show distinct failure types first, with duplicate event/track pairs removed.
`comparison.json` records counts and selected examples, all input hashes,
thresholds, policies and full-cohort summaries. Examples are deliberately
selected illustrations, **not an unbiased sample or a frequency estimate**.

Plots compare baseline and focus model in transverse and longitudinal views,
with assignment matrices and associated-prediction composition for every model.
Geometry-free inputs produce assignment audits. Saved centers and measurement
supports are not fitted trajectories or measured 3D crossings. BMT C supports
use their arcs when detector metadata is available. Missing geometry does not
prevent metric comparison. Use `--no-plots` for JSON-only diagnostics.

Paired-event bootstrap intervals report the pooled quality-efficiency difference
between the baseline and each model. All clusters/tracks of an event move
together, preserving track weighting. `--bootstrap-resamples` and `--seed`
control the calculation. Replicates with no eligible truth tracks are omitted
and counted explicitly. Intervals cover test-event sampling, not training seeds.

## Runnable synthetic example

From the repository root, choose a new scratch directory:

```bash
python -m train.downstream.eval.examples.make_finding_analysis_example \
  --output-dir /tmp/finding-analysis-example

python -m train.downstream.eval.calibrate_track_finding \
  --validation /tmp/finding-analysis-example/validation.npz \
  --test /tmp/finding-analysis-example/test.npz \
  --objective dice --thresholds 0.2 0.5 0.7 1 --refine-steps \
  --output-dir /tmp/finding-analysis-example/dice

python -m train.downstream.eval.calibrate_track_finding \
  --validation /tmp/finding-analysis-example/validation.npz \
  --test /tmp/finding-analysis-example/test.npz \
  --objective quality-efficiency --thresholds 0.2 0.5 0.7 1 --refine-steps \
  --output-dir /tmp/finding-analysis-example/quality

python -m train.downstream.eval.compare_track_finding_events \
  --manifest /tmp/finding-analysis-example/comparison.json \
  --focus-model 'Example adapter' --max-examples 1 \
  --output-dir /tmp/finding-analysis-example/events
```

The synthetic example selects 0.2 for Dice and 0.7 for joint 90% quality efficiency.
Its selected event shows overlay pickup at 0.2. It has no detector geometry and
is only an execution/metric example.

Run CPU regressions with:

```bash
python -m unittest train.downstream.tests.test_finding_analysis_tools \
  train.downstream.tests.test_mc_reconstruction_metrics -v
```
