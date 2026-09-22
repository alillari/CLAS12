#!/usr/bin/env python3
"""Build physical loss scales from the aligned v7 COATJAVA auxiliary target.

For the canonical v7 event product, aux_target columns 16--18 are:
  * CVT::Tracks momentum magnitude in GeV,
  * CVT::Tracks theta in rad,
  * CVT::Trajectory phi at the innermost CVT trajectory surface in rad.

The final quantity is intentionally not CVT::Tracks phi0.  It is the trajectory
direction nearest the detector entrance and is therefore the appropriate
conventional reference for the MC-entrance phi regression target.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from fm4npp.datasets.dataset import RaggedMmap
from regression_utils import transform_regression_target_numpy


AUX_CVT_BENCHMARK_P = 16
AUX_CVT_BENCHMARK_THETA = 17
AUX_CVTTRAJ_ENTRANCE_PHI = 18


def wrapped_delta(prediction, truth):
    return np.arctan2(np.sin(prediction - truth), np.cos(prediction - truth))


def central_width_68(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, math.nan
    q16, q84 = np.quantile(values, [0.16, 0.84])
    return float(0.5 * (q84 - q16)), float(q16), float(q84)


def residual_summary(values, unit):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    width, q16, q84 = central_width_68(values)
    return {
        "unit": unit,
        "count": int(len(values)),
        "central_width_68": width,
        "q16": q16,
        "q84": q84,
        "median": float(np.median(values)) if len(values) else math.nan,
        "mean": float(np.mean(values)) if len(values) else math.nan,
        "mean_absolute": float(np.mean(np.abs(values))) if len(values) else math.nan,
    }


def selected_segments(labels, min_clusters, exact_clusters):
    result = []
    for label in sorted(int(value) for value in np.unique(labels) if int(value) != -1):
        n_points = int(np.sum(labels == label))
        if exact_clusters and n_points != min_clusters:
            continue
        if not exact_clusters and n_points < min_clusters:
            continue
        result.append(label)
    return result


def finite_segment_target(reg_values):
    transformed = transform_regression_target_numpy(reg_values, "p_phi_theta")
    if transformed.ndim != 2 or transformed.shape[-1] != 4:
        raise ValueError(f"Unexpected transformed target shape {transformed.shape}")
    finite = np.isfinite(transformed)
    if not np.all(finite.any(axis=0)):
        return None
    return np.where(finite, transformed, 0.0).sum(axis=0) / finite.sum(axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--segment-min-clusters", type=int, default=12)
    parser.add_argument("--segment-exact-clusters", action="store_true")
    parser.add_argument(
        "--target-momentum-scale-to-gev", type=float, default=1.0e-3,
        help="Conversion from stored MC entrance momentum to GeV (v7 is MeV).",
    )
    parser.add_argument(
        "--limit-size", type=int, default=1_000_000,
        help="Maximum accepted truth segments. Use 0 to scan every accepted segment.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Defaults to <data-root parent>/stats/regression_loss_reference_cvttraj_...json",
    )
    return parser.parse_args()


def compute_reference_stats(
    data_root,
    split,
    min_clusters,
    exact_clusters,
    target_momentum_scale_to_gev,
    limit_size,
):
    data_root = Path(data_root)
    reg = RaggedMmap(data_root / f"reg_target_{split}")
    seg = RaggedMmap(data_root / f"seg_target_{split}")
    aux = RaggedMmap(data_root / f"aux_target_{split}")
    if not (len(reg) == len(seg) == len(aux)):
        raise ValueError(
            "reg_target, seg_target, and aux_target must have equal event counts: "
            f"{len(reg)}, {len(seg)}, {len(aux)}"
        )
    if target_momentum_scale_to_gev <= 0.0:
        raise ValueError("target_momentum_scale_to_gev must be positive")
    if limit_size is not None and limit_size < 0:
        raise ValueError("limit_size must be nonnegative")

    residuals = {name: [] for name in ("p_absolute", "p_relative", "theta", "phi")}
    diagnostics = {
        "events_scanned": 0,
        "candidate_segments": 0,
        "accepted_segments": 0,
        "rejected_missing_or_nonfinite_target": 0,
        "rejected_aux_shape": 0,
        "rejected_missing_or_nonfinite_reference": 0,
        "aux_rows_with_nonconstant_reference": 0,
    }
    limit_reached = False
    for event_index in range(len(reg)):
        labels = np.asarray(seg[event_index])
        reg_values = np.asarray(reg[event_index], dtype=np.float64)
        aux_values = np.asarray(aux[event_index], dtype=np.float64)
        diagnostics["events_scanned"] += 1
        if (
            reg_values.ndim != 2
            or aux_values.ndim != 2
            or len(labels) != len(reg_values)
            or len(labels) != len(aux_values)
        ):
            raise ValueError(
                f"Event {event_index} has incompatible target shapes: "
                f"seg={labels.shape}, reg={reg_values.shape}, aux={aux_values.shape}"
            )
        if aux_values.shape[1] <= AUX_CVTTRAJ_ENTRANCE_PHI:
            diagnostics["rejected_aux_shape"] += 1
            continue

        for label in selected_segments(labels, min_clusters, exact_clusters):
            diagnostics["candidate_segments"] += 1
            mask = labels == label
            truth = finite_segment_target(reg_values[mask])
            if truth is None or not np.all(np.isfinite(truth)):
                diagnostics["rejected_missing_or_nonfinite_target"] += 1
                continue
            truth_p_gev = truth[0] * target_momentum_scale_to_gev
            truth_phi = math.atan2(truth[2], truth[1])
            truth_theta = truth[3]
            if not math.isfinite(truth_p_gev) or truth_p_gev <= 0.0:
                diagnostics["rejected_missing_or_nonfinite_target"] += 1
                continue

            reference_rows = aux_values[mask][:, [
                AUX_CVT_BENCHMARK_P,
                AUX_CVT_BENCHMARK_THETA,
                AUX_CVTTRAJ_ENTRANCE_PHI,
            ]]
            if not np.all(np.isfinite(reference_rows)):
                diagnostics["rejected_missing_or_nonfinite_reference"] += 1
                continue
            reference = np.median(reference_rows, axis=0)
            if not np.allclose(reference_rows, reference, rtol=1.0e-6, atol=1.0e-6):
                diagnostics["aux_rows_with_nonconstant_reference"] += 1
            cvt_p, cvt_theta, cvttraj_phi = reference
            residuals["p_absolute"].append(cvt_p - truth_p_gev)
            residuals["p_relative"].append((cvt_p - truth_p_gev) / truth_p_gev)
            residuals["theta"].append(cvt_theta - truth_theta)
            residuals["phi"].append(wrapped_delta(cvttraj_phi, truth_phi))
            diagnostics["accepted_segments"] += 1
            if limit_size and diagnostics["accepted_segments"] >= limit_size:
                limit_reached = True
                break
        if limit_reached:
            break

    if not diagnostics["accepted_segments"]:
        raise ValueError("No finite matched CVT/CVT::Trajectory reference segments were found")
    return {
        "schema": "clas12_regression_loss_reference_v1",
        "version": 1,
        "task": "p_phi_theta",
        "reference_method": "CVT::Tracks p/theta plus CVT::Trajectory entrance phi",
        "data_root": str(data_root.resolve()),
        "split": split,
        "target_momentum_scale_to_gev": float(target_momentum_scale_to_gev),
        "selection": {
            "adapter_sample_mode": "event_segment",
            "segment_target_source": "mctrue",
            "segment_min_clusters": int(min_clusters),
            "segment_exact_clusters": bool(exact_clusters),
            "limit_size": None if not limit_size else int(limit_size),
            "selection_order": "first accepted segments in deterministic RaggedMmap order",
        },
        "aux_target_layout": {
            "cvt_benchmark_p": AUX_CVT_BENCHMARK_P,
            "cvt_benchmark_theta": AUX_CVT_BENCHMARK_THETA,
            "cvttraj_entrance_phi": AUX_CVTTRAJ_ENTRANCE_PHI,
        },
        "residual_definitions": {
            "p_absolute_gev": "CVT::Tracks p - MC entrance p",
            "p_relative": "(CVT::Tracks p - MC entrance p) / MC entrance p",
            "theta_rad": "CVT::Tracks theta - MC entrance theta",
            "phi_rad_wrapped": "wrap(CVT::Trajectory entrance phi - MC entrance phi)",
        },
        "residuals": {
            "p_absolute_gev": residual_summary(residuals["p_absolute"], "GeV"),
            "p_relative": residual_summary(residuals["p_relative"], "fraction"),
            "theta_rad": residual_summary(residuals["theta"], "rad"),
            "phi_rad_wrapped": residual_summary(residuals["phi"], "rad"),
        },
        "diagnostics": diagnostics,
    }


def main():
    args = parse_args()
    if args.output is None:
        exact = "_exact" if args.segment_exact_clusters else ""
        args.output = (
            args.data_root.parent / "stats" /
            "regression_loss_reference_cvttraj_p_phi_theta_event_segment_"
            f"mctrue_min{args.segment_min_clusters}{exact}.json"
        )
    stats = compute_reference_stats(
        args.data_root,
        args.split,
        args.segment_min_clusters,
        args.segment_exact_clusters,
        args.target_momentum_scale_to_gev,
        args.limit_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
