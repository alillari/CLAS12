"""Score saved, final cluster assignments with mc_reconstruction_v1 (CPU only).

No inference, threshold tuning or checkpoint selection is done. Optional explicit
truth-policy projection uses the same full-event cut as the downstream adapters.
See docs/evaluation/mc_reconstruction_v1.tex for the NPZ and provenance contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
import sklearn

from train.downstream import track_finding_metrics as metrics
from fm4npp.datasets import track_eligibility


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rescore_archive(path, prediction_keys=("pred",), *, events_per_batch=64,
                    hdf5_root=None, split=None, plan_id=None, event_sink=None,
                    truth_policy="as_saved"):
    """Return summaries plus a receipt; optionally stream event rows to a callback.

    With HDF5, require and verify both physical event identity and original point
    indices. Never infer a provenance alignment from coordinates or truth labels.
    """
    if events_per_batch < 1:
        raise ValueError("events_per_batch must be positive")
    if truth_policy not in {"as_saved", track_eligibility.PRIMARY_TRACK_POLICY}:
        raise ValueError("Unknown truth_policy")
    apply_cut = truth_policy == track_eligibility.PRIMARY_TRACK_POLICY
    relabelled_points = 0
    keys = tuple(prediction_keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("prediction_keys must be nonempty and unique")
    if bool(hdf5_root) != bool(split) or bool(hdf5_root) != bool(plan_id):
        raise ValueError("HDF5 provenance requires root, split and plan_id together")
    dataset = None
    totals = {key: metrics.MCReconstructionAccumulator() for key in keys}
    try:
        with np.load(path, allow_pickle=False) as archive:
            offsets = np.asarray(archive["offsets"])
            truth = metrics._point_array(archive["truth"], "truth", integer=True)
            if (offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) == 0
                    or offsets[0] != 0 or offsets[-1] != len(truth)
                    or np.any(offsets[1:] < offsets[:-1])):
                raise ValueError("Invalid event offsets")
            predictions = {key: metrics._point_array(archive[key], key, len(truth), integer=True) for key in keys}
            raw_names = ("background_only", "ownership", "origin_noise", "purity")
            sidecars = {k: metrics._point_array(archive[k], k, len(truth))
                        for k in ("provenance", "point_ids", *raw_names) if k in archive}
            if "provenance" in sidecars and any(k in sidecars for k in raw_names):
                raise ValueError("Supply provenance codes or raw HDF5 sidecars, not both")
            if any(k in sidecars for k in raw_names[1:]) and "background_only" not in sidecars:
                raise ValueError("Raw provenance sidecars require background_only")
            event_count = len(offsets) - 1
            identity_names = ("source_event_index", "source_id", "original_ordinal")
            identities = {k: metrics._point_array(archive[k], k, event_count, integer=True)
                          for k in identity_names if k in archive}
            if "source_event_index" in identities:
                indices = identities["source_event_index"]
                if np.any(indices < 0) or len(np.unique(indices)) != event_count:
                    raise ValueError("Repeated or negative source_event_index")
            if "source_id" in identities and "original_ordinal" in identities:
                physical_keys = set(zip(identities["source_id"], identities["original_ordinal"]))
                if len(physical_keys) != event_count:
                    raise ValueError("Duplicate physical events")
            if hdf5_root:
                if any(k not in identities for k in identity_names) or "point_ids" not in sidecars:
                    raise ValueError("HDF5 joins require event identities and original point_ids")
                if any(k in sidecars for k in ("provenance", *raw_names)):
                    raise ValueError("Choose archived provenance or an HDF5 join, not both")
                from fm4npp.datasets.cvt_hdf5 import CVTClusterEvents
                dataset = CVTClusterEvents(hdf5_root, split)
                if dataset.plan_id != plan_id:
                    raise ValueError("HDF5 plan_id mismatch")
            for start in range(0, event_count, events_per_batch):
                batch = {key: metrics.MCReconstructionAccumulator() for key in keys}
                for e in range(start, min(start + events_per_batch, event_count)):
                    sl = slice(int(offsets[e]), int(offsets[e + 1]))
                    event_truth = truth[sl]
                    point_ids = sidecars["point_ids"][sl] if "point_ids" in sidecars else None
                    provenance = sidecars["provenance"][sl] if "provenance" in sidecars else None
                    raw = {k: sidecars[k][sl] for k in raw_names if k in sidecars}
                    if dataset is not None:
                        fields = dataset.read(int(identities["source_event_index"][e]), [
                            "events.source_id", "events.original_ordinal", "clusters.primary_segment",
                            *("clusters." + k for k in raw_names)])
                        if any(int(fields["events." + k]) != int(identities[k][e])
                               for k in ("source_id", "original_ordinal")):
                            raise ValueError(f"HDF5 physical event identity mismatch at event {e}")
                        original_truth = fields["clusters.primary_segment"]
                        if (point_ids.dtype.kind not in "iu" or
                                not np.array_equal(np.sort(point_ids), np.arange(len(original_truth)))):
                            raise ValueError(f"Expected a full-event original-point permutation at event {e}")
                        original_truth = original_truth[point_ids]
                        projected_truth = (track_eligibility.eligible_primary_labels(original_truth)
                                           if apply_cut else original_truth)
                        if not (np.array_equal(original_truth, event_truth) or
                                np.array_equal(projected_truth, event_truth)):
                            raise ValueError(f"HDF5 truth label mismatch at event {e} under {truth_policy}")
                        raw = {k: fields["clusters." + k][point_ids] for k in raw_names}
                    if apply_cut:
                        projected = track_eligibility.eligible_primary_labels(event_truth)
                        relabelled_points += int(np.count_nonzero(projected != event_truth))
                        event_truth = projected
                    if raw:
                        provenance = metrics.cluster_provenance_from_hdf5(**raw)
                    for key in keys:
                        row = metrics.event_mc_reconstruction_metrics(
                            event_truth, predictions[key][sl], provenance=provenance, point_ids=point_ids)
                        batch[key].add(row)
                        if event_sink is not None:
                            event_sink({"event_index": e, "method": key,
                                        **{k: int(v[e]) for k, v in identities.items()}, **row})
                for key in keys:
                    totals[key].merge(batch[key])
    finally:
        if dataset is not None:
            dataset.close()
    return {
        "mc_reconstruction_version": metrics.MC_RECONSTRUCTION_VERSION,
        "primary_evaluation_metric": "mc_track_dice_mean",
        "checkpoint_selection_modified": False,
        "metrics": {key: value.summary() for key, value in totals.items()},
        "receipt": {
            "input": str(Path(path).resolve()), "input_sha256": _sha256(path),
            "metrics_source_sha256": _sha256(metrics.__file__),
            "scorer_source_sha256": _sha256(__file__),
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "sklearn_version": sklearn.__version__, "events_per_batch": events_per_batch,
            "hdf5_root": str(Path(hdf5_root).resolve()) if hdf5_root else None,
            "split": split, "plan_id": plan_id,
            "provenance_source": ("hdf5" if hdf5_root else
                "archive" if any(k in sidecars for k in ("provenance", *raw_names)) else "unavailable"),
            "full_event_membership_verified": bool(hdf5_root),
            "prediction_keys": list(keys),
            "truth_policy": truth_policy, "n_relabelled_truth_points": relabelled_points,
            "eligibility_source_sha256": _sha256(track_eligibility.__file__) if apply_cut else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-key", nargs="+", default=["pred"])
    parser.add_argument("--events-per-batch", type=int, default=64)
    parser.add_argument("--truth-policy", choices=["as_saved", track_eligibility.PRIMARY_TRACK_POLICY],
                        default="as_saved", help="Explicitly project full-event truth labels; default preserves historical labels")
    parser.add_argument("--hdf5-root", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--plan-id")
    parser.add_argument("--events-output", type=Path, help="Optional per-event JSONL including both match lists")
    args = parser.parse_args()
    outputs = [args.output] + ([args.events_output] if args.events_output else [])
    if len({p.resolve() for p in outputs}) != len(outputs) or any(p.exists() for p in outputs):
        parser.error("Output paths must be distinct and new; historical outputs are never overwritten")
    stream = args.events_output.open("x") if args.events_output else None
    try:
        result = rescore_archive(args.input, args.prediction_key,
            events_per_batch=args.events_per_batch, hdf5_root=args.hdf5_root,
            split=args.split, plan_id=args.plan_id,
            truth_policy=args.truth_policy,
            event_sink=(lambda row: stream.write(json.dumps(row, allow_nan=False) + "\n")) if stream else None)
        with args.output.open("x") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    finally:
        if stream:
            stream.close()


if __name__ == "__main__":
    main()
