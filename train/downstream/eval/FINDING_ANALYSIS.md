# Saved-prediction calibration and event diagnostics

These CPU tools operate on complete events and fixed model checkpoints. They
support arbitrary model names and cohort sizes. They require NumPy, SciPy and
scikit-learn; plotting also uses Matplotlib. Neither tool imports the trainer or
runs inference.

## Export predictions and select an assignment threshold

For an HDF5 checkpoint, set `evaluation_split: validation` in the analysis YAML
and run the ordinary evaluator once. Export the test split with a separate YAML
and destination if you want the calibrator to report both splits from archives.
Use the same model profile, checkpoint and decoder settings for both exports.

```bash
python -m train.downstream.eval.evaluate_track_finding \
  --analysis-config /path/to/validation_analysis.yaml \
  --output-dir /path/to/validation_eval --save-raw-predictions
```

The evaluator writes `raw_predictions.npz` beside its ordinary results. The
archive captures winners and scores before the configured assignment cut. The
export requires whole HDF5 events and carries the original cluster indices
through serialization. Use `--save-query-scores` in place of
`--save-raw-predictions` to retain all binary option-2 candidates for a query
confidence scan.

```bash
python -m train.downstream.eval.calibrate_track_finding \
  --validation /path/to/validation_eval/raw_predictions.npz \
  --objective dice \
  --output-dir /path/to/new_dice_calibration
```

Add `--test /path/to/test_eval/raw_predictions.npz` for held-out reporting.
Older archives can supply aligned sidecars via `--metadata` and
`--test-metadata`. Run once per checkpoint. For the ordinary held-out evaluator,
copy the selected `threshold` into `analysis.assignment_threshold` and set
`analysis.evaluation_split: test`.

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

### Compare a fixed query cutoff with a joint scan

Use archives from `--save-query-scores` for both arms. Run two calibrations with
the same assignment grid, assignment refinement steps, match cuts and optional
F1 floor. For the first, pass `--fixed-query-confidence 0.5`. For the second,
replace that option with a query grid such as
`--query-confidence-thresholds 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0`.
Each joint refinement stage searches both axes around the current best point.
Specify `--query-refine-steps` to set one step per assignment refinement stage.
The default query steps are ten times the assignment steps, capped at 1.
The query cut is strict. The cluster goes to the highest-scoring eligible
query only if its product score is strictly above the assignment cutoff.

`--min-matched-track-f1` restricts validation selection to scanned points
meeting an absolute floor in [0, 1]. Matched-track F1 is
`2 * n_qualified_tracks / (n_true_tracks + n_pred_tracks)` and uses the
specified IoU, purity and completeness cuts. To use the historical 0.5
matching cuts, pass `--min-iou 0.5 --min-purity 0.5 --min-completeness 0.5`.
Both selected validation and fixed test reports include this F1 alongside
mean Dice and pooled hit purity/completeness. The floor belongs to validation
selection. If every scanned point misses it, calibration fails.

Use separate output directories for the two scans and supply the same
held-out archive to both. The evaluator accepts the selected
`query_confidence_threshold` in the analysis YAML with assignment option 2
and a binary track/no-object head. Its existing query gate applies when the
field is omitted.

Outputs in a **new** directory are `validation_scan.json`, `selection.json` and
`summary.json`. The validation choice and its receipt are persisted before test
predictions are opened; its hash is checked after reporting. When test data are
supplied, validation and test must carry physical identities and must be disjoint.
Evaluator-produced archive pairs also verify the split, adapter and backbone
digests, assignment settings and truth policy.
If held-out validation fails, the selection remains available and no successful
summary is written. Existing outputs are never overwritten.

## Archive contract

An NPZ uses flattened cluster arrays plus complete-event offsets:

| Field | Contract |
| --- | --- |
| `truth` | Integer MC track IDs; background/non-target = -1 |
| `pred` | Integer raw query winners, or explicitly declared final assignments |
| `offsets` | Integer array starting at 0, monotone, ending at number of points |
| `scores`, `signal` | Required for raw candidates: finite scores and boolean native gate |
| `source_id`, `original_ordinal` | One integer per event; together form its unique physical identity |
| `point_ids` | Unique original cluster indices within each event |
| `query_confidence`, `candidate_scores` | Optional binary option-2 probabilities shaped `(n_events, n_queries)` and product scores shaped `(n_points, n_queries)` |
| `evaluation_contract` | Optional scalar JSON with split, checkpoint digests, assignment settings and eligibility policy |
| `provenance` | Optional shared `ClusterProvenance` integer codes |
| `xyz` | Optional finite `(n_points, 3)` saved cluster centers in cm |
| `endpoints` | Optional finite `(n_points, 6)` measurement-support endpoints in cm |
| `detector`, `layer` | Optional aligned integer arrays; CLAS12 detector 1 = BMT, 2 = BST |

Labels and original cluster indices must fit signed `int64`; original indices
must be nonnegative. Unsigned NPZ/HDF5 labels within that range are normalized
before thresholding so that rejected points retain the signed -1 sentinel.
Every raw provenance field must span all points, and geometry must be real-valued.

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
`selection_config` preserves the requested categories, example limit and minimum
truth-cluster count, including when no example qualifies.
`quality_summaries` contains each method's pooled efficiency at the configured
cuts. The shared evaluator's `summaries` retain their default accepted-match
configuration; the legacy `track_efficiency_global` field is not a 90%/90% metric.

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
Count arrays must contain nonnegative integer values no larger than `2**53-1`;
unsigned arrays are converted before calculating signed differences. Existing
output directories are rejected before input processing.

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
  train.downstream.tests.test_finding_analysis_edge_cases \
  train.downstream.tests.test_mc_reconstruction_metrics -v
```
