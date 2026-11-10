# Primary MC eligibility for track finding and noise tagging

Policy version: `primary_mc_min4_v1`. Effective for adapter executions using this
patch. This enforces the already agreed reconstructability cut downstream of the
skimmer; it does not change the meaning of physical primary-particle provenance.

For each **complete event**, count the clusters assigned to each nonnegative MC
segment ID under the existing ownership rule. If the count is less than four,
replace that segment's downstream label with `-1`. Keep tracks with exactly four
clusters or more, preserving their IDs. Never remove the cluster tokens.

| Track size in the full event | Finding target | Noise-tagging target |
| --- | --- | --- |
| 1, 2 or 3 owned clusters | `-1`, non-target/background | `1`, noise class |
| 4 or more owned clusters | Existing segment ID | Existing signal label (`0`) |
| Already non-target | Unchanged | Existing noise policy |

The size is **not** computed after sorting, cropping, chunking, batching or padding.
A four-cluster track remains eligible when a training chunk contains only two
of its clusters. A track ID reused in another event contributes nothing to the
current event's count. All-short-track events remain present with no finding
targets and all newly excluded members labelled as noise.

## Implementation and scope

The shared implementation is
[`eligible_primary_labels`](../../fm4npp/datasets/track_eligibility.py).
The track-finding and noise-tagging trainer entry points enable it automatically
for CLAS12 position-only layouts (`xyz`/`etaphr`). Train, validation and test
loaders propagate the same setting. The track-finding experiment entry point
records it in resolved configuration before launch.

- HDF5: `CVTDownstreamSource(..., apply_primary_track_cut=True)` projects
  `seg_target` from `clusters.primary_segment`, then derives `noise_target` from
  that projected segmentation.
- Legacy ragged data: lazy label views apply the same cut to full-event
  segmentation. The noise view changes newly excluded clusters to class `1`
  and preserves other existing noise labels. No on-disk arrays are rewritten.
- Noise class-balance counts read the projected label view, so any automatic
  inverse-frequency weights use the new supervision consistently.
- Track-finding campaign preflight counts eligible tracks under this policy.

The generic dataset/source defaults retain their old projection for callers such
as momentum regression and PID. Direct loader users must explicitly pass
`apply_primary_track_cut=True`; the two adapter trainers do this automatically.
Original four-column TPC tasks are not changed. PID labels, regression targets,
COATJAVA predictions, physical owner IDs, and provenance are unchanged.

The loss formulas, model architectures and checkpoint-selection rules are
unchanged. **Training gradients do change through the corrected target labels.**
For finding, excluded clusters are negative members of every eligible-track mask;
they do not create a new background-track target. For noise tagging, they train
the noise class. Existing event-length cuts are unchanged; an event-count cap
combined with the existing maximum-track cut can select a different cohort
because fewer tracks now count as targets. Record the resulting cohort identity.

## Provenance and evaluation

The HDF5 data contract records:

```json
{
  "track_eligibility_policy": "primary_mc_min4_v1",
  "primary_track_min_clusters": 4,
  "noise_policy": "primary_unassigned_or_lt4_is_noise_v2"
}
```

Short-track clusters still have their original MC provenance. They are not
verified overlay. They reduce eligible-track Dice/purity if they contaminate an
associated prediction; independent predictions consisting only of these
non-target clusters remain neutral to the primary Dice score.

The shared metric API continues to score supplied truth labels without silently
changing them. To rescore a saved full-event prediction archive under this policy:

```bash
python -m train.downstream.eval.rescore_track_finding \
  --input /path/to/predictions.npz --prediction-key pred coatjava \
  --truth-policy primary_mc_min4_v1 \
  --output /path/to/new_summary.json
```

The default `--truth-policy as_saved` preserves historical behavior. The explicit
option changes only truth labels in memory, records the policy, source hash and
number of relabelled clusters, and leaves predictions and the archive unchanged.
With an HDF5 provenance join, stored labels must exactly equal either the original
full-event labels or their projection under the explicitly selected policy.
Physical event and original cluster identities are still verified.

Compare checkpoints on the same truth policy, event cohort and prediction
assignment settings. A score computed with the cut is not directly comparable
to an older score using all raw primary labels.

## Existing jobs and checkpoints

Running managed jobs retain their frozen source snapshots and original targets;
this local patch does not stop, restart or modify them. Use the patched source
for subsequent runs. Existing predictions can be evaluated under the new policy
without retraining; training under the corrected supervision requires a new run
or an explicitly recorded continuation.

The noise checkpoint loader retains its strict data-contract check. An older
checkpoint's original contract does not silently match the new contract; do not
rewrite its saved metadata to hide that difference. Historical execution should
use its original source/configuration, with separately labelled rescoring when
needed. No automatic checkpoint migration is included in this patch.

## Validation

`test_primary_track_eligibility.py` covers the 1/2/3/4/5-cluster boundary, sparse
and event-local IDs, idempotence, both adapter entry points, HDF5 and ragged
storage, serialization, chunking, padding, all-short events, class-balance
counts, all three loader splits, provenance preservation and explicit rescoring.
The broader finding, noise, metric and momentum-sample regression suites check
compatibility. The versioned metric equations remain in
[`mc_reconstruction_v1.pdf`](mc_reconstruction_v1.pdf).
