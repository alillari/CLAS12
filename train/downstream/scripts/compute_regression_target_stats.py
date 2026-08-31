#!/usr/bin/env python3
"""Compute event-weighted regression-target statistics from a training split."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from fm4npp.datasets.dataset import RaggedMmap
from regression_utils import (
    REGRESSION_TARGET_COLUMNS,
    regression_target_columns,
    transform_regression_target_numpy,
)

REGRESSION_TARGET_COLUMNS_V6 = REGRESSION_TARGET_COLUMNS + (
    "mc_entrance_phi",
    "mc_entrance_kappa",
    "mc_entrance_tan_lambda",
)


def update_running_stats(count, mean, m2, values):
    """Merge finite rows from a two-dimensional array into Welford state."""
    for column in range(values.shape[1]):
        finite = values[:, column][np.isfinite(values[:, column])]
        if finite.size == 0:
            continue
        batch_count = np.int64(finite.size)
        batch_mean = finite.mean(dtype=np.float64)
        batch_m2 = np.square(finite - batch_mean).sum(dtype=np.float64)
        delta = batch_mean - mean[column]
        total = count[column] + batch_count
        mean[column] += delta * batch_count / total
        m2[column] += batch_m2 + delta * delta * count[column] * batch_count / total
        count[column] = total


def _regression_width(reg):
    if len(reg) == 0:
        raise ValueError("reg_target is empty")
    shape = np.asarray(reg[0]).shape
    if len(shape) != 2:
        raise ValueError(f"Expected two-dimensional reg_target rows, found shape {shape}")
    return int(shape[1])


def _raw_columns_for_width(width):
    if width == len(REGRESSION_TARGET_COLUMNS):
        return list(REGRESSION_TARGET_COLUMNS)
    if width == len(REGRESSION_TARGET_COLUMNS_V6):
        return list(REGRESSION_TARGET_COLUMNS_V6)
    raise ValueError(f"Unsupported reg_target width {width}")


def _finite_segment_target(values, task):
    if task is not None:
        values = transform_regression_target_numpy(values, task)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    if np.any(counts == 0):
        return None
    return np.where(finite, values, 0.0).sum(axis=0) / counts


def _selected_event_segments(seg, min_clusters, exact_clusters, ignore_label=-1):
    out = []
    for label in sorted(int(x) for x in np.unique(seg) if int(x) != int(ignore_label)):
        n_points = int(np.sum(seg == label))
        if exact_clusters:
            if n_points != min_clusters:
                continue
        elif n_points < min_clusters:
            continue
        out.append(label)
    return out


def compute_stats(data_root, split, low_thr, high_thr, limit_size, chunk_size, task=None,
                  adapter_sample_mode="track_legacy", segment_target_source="mctrue",
                  segment_min_clusters=12, segment_exact_clusters=False):
    reg = RaggedMmap(data_root / f"reg_target_{split}")
    features = RaggedMmap(data_root / f"features_{split}")
    if len(reg) != len(features):
        raise ValueError(f"reg_target has {len(reg)} events but features has {len(features)}")

    n_raw_columns = _regression_width(reg)
    raw_columns = _raw_columns_for_width(n_raw_columns)
    output_columns = (
        raw_columns
        if task is None
        else list(regression_target_columns(task))
    )
    n_columns = len(output_columns)
    count = np.zeros(n_columns, dtype=np.int64)
    mean = np.zeros(n_columns, dtype=np.float64)
    m2 = np.zeros(n_columns, dtype=np.float64)
    adapter_sample_mode = str(adapter_sample_mode)

    if adapter_sample_mode == "track_legacy":
        reg_sizes = reg.ends - reg.starts
        feature_sizes = features.ends - features.starts
        if np.any(reg_sizes % n_raw_columns):
            raise ValueError(f"Regression arrays do not have the expected {n_raw_columns}-column layout")
        if np.any(feature_sizes % 3):
            raise ValueError("Feature arrays do not have the expected xyz layout")

        reg_hits = reg_sizes // n_raw_columns
        feature_hits = feature_sizes // 3
        if not np.array_equal(reg_hits, feature_hits):
            raise ValueError("Regression-target and feature hit counts differ")

        selected = np.flatnonzero((feature_hits >= low_thr) & (feature_hits <= high_thr))
        if limit_size is not None:
            selected = selected[:limit_size]
        if selected.size == 0:
            raise ValueError("No events passed the configured hit-count filters")
        offsets = np.arange(n_raw_columns, dtype=np.int64)

        for start in range(0, selected.size, chunk_size):
            indices = selected[start:start + chunk_size]
            first_values = reg.memmap[reg.starts[indices, None] + offsets]
            values = np.asarray(first_values, dtype=np.float64)
            if task is not None:
                values = transform_regression_target_numpy(values, task)
            update_running_stats(count, mean, m2, values)
        selected_count = int(selected.size)
    elif adapter_sample_mode == "event_segment":
        if task is None:
            raise ValueError("event_segment statistics require --task")
        if segment_target_source not in {"mctrue", "seg_target"}:
            raise ValueError("event_segment statistics currently support segment_target_source=mctrue/seg_target")
        seg = RaggedMmap(data_root / f"seg_target_{split}")
        if len(seg) != len(features):
            raise ValueError(f"seg_target has {len(seg)} events but features has {len(features)}")
        selected_count = 0
        for event_idx in range(len(features)):
            labels = np.asarray(seg[event_idx])
            values = np.asarray(reg[event_idx], dtype=np.float64)
            feature_values = np.asarray(features[event_idx])
            if values.ndim != 2 or values.shape[1] != n_raw_columns:
                raise ValueError(
                    f"reg_target event {event_idx} has shape {values.shape}, "
                    f"expected (*, {n_raw_columns})"
                )
            if feature_values.ndim != 2 or feature_values.shape[1] != 3:
                raise ValueError(
                    f"features event {event_idx} has shape {feature_values.shape}, expected (*, 3)"
                )
            if values.shape[0] != feature_values.shape[0] or labels.shape[0] != feature_values.shape[0]:
                raise ValueError(f"features/reg_target/seg_target hit counts differ for event {event_idx}")
            for label in _selected_event_segments(
                labels,
                min_clusters=int(segment_min_clusters),
                exact_clusters=bool(segment_exact_clusters),
            ):
                mask = labels == int(label)
                n_points = int(np.sum(mask))
                if n_points < low_thr or n_points > high_thr:
                    continue
                target = _finite_segment_target(values[mask], task)
                if target is None:
                    continue
                update_running_stats(count, mean, m2, target.reshape(1, -1))
                selected_count += 1
                if limit_size is not None and selected_count == int(limit_size):
                    break
            if limit_size is not None and selected_count == int(limit_size):
                break
        if selected_count == 0:
            raise ValueError("No event segments passed the configured filters")
    else:
        raise ValueError("adapter_sample_mode must be 'track_legacy' or 'event_segment'")

    variance = m2 / np.maximum(count, 1)
    return {
        "version": 2 if task is not None else 1,
        "data_root": str(data_root.resolve()),
        "split": split,
        "task": task or "raw",
        "columns": output_columns,
        "filters": {
            "low_thr": low_thr,
            "high_thr": high_thr,
            "adapter_sample_mode": adapter_sample_mode,
            "segment_target_source": segment_target_source,
            "segment_min_clusters": int(segment_min_clusters),
            "segment_exact_clusters": bool(segment_exact_clusters),
        },
        "selected_events": selected_count,
        "count": count.tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--low-thr", type=int, default=1)
    parser.add_argument("--high-thr", type=int, default=100)
    parser.add_argument("--limit-size", type=int)
    parser.add_argument("--chunk-size", type=int, default=250000)
    parser.add_argument("--adapter-sample-mode", choices=("track_legacy", "event_segment"), default="track_legacy")
    parser.add_argument("--segment-target-source", default="mctrue")
    parser.add_argument("--segment-min-clusters", type=int, default=12)
    parser.add_argument("--segment-exact-clusters", action="store_true")
    parser.add_argument(
        "--task",
        help=(
            "Optional regression task to compute target-specific stats, e.g. "
            "mom or pt_phi_eta. Omit to preserve the legacy seven-column stats file."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (defaults to <data-root parent>/stats/regression_target_stats.json)",
    )
    args = parser.parse_args()
    if args.output is None:
        filename = (
            "regression_target_stats.json"
            if args.task is None
            else f"regression_target_stats_{args.task}.json"
        )
        args.output = args.data_root.parent / "stats" / filename

    stats = compute_stats(
        args.data_root,
        args.split,
        args.low_thr,
        args.high_thr,
        args.limit_size,
        args.chunk_size,
        args.task,
        adapter_sample_mode=args.adapter_sample_mode,
        segment_target_source=args.segment_target_source,
        segment_min_clusters=args.segment_min_clusters,
        segment_exact_clusters=args.segment_exact_clusters,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        json.dump(stats, stream, indent=2)
        stream.write("\n")
    print(json.dumps(stats, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
