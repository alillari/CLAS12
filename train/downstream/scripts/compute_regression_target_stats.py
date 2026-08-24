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


def compute_stats(data_root, split, low_thr, high_thr, limit_size, chunk_size, task=None):
    reg = RaggedMmap(data_root / f"reg_target_{split}")
    features = RaggedMmap(data_root / f"features_{split}")
    if len(reg) != len(features):
        raise ValueError(f"reg_target has {len(reg)} events but features has {len(features)}")

    n_raw_columns = len(REGRESSION_TARGET_COLUMNS)
    output_columns = (
        list(REGRESSION_TARGET_COLUMNS)
        if task is None
        else list(regression_target_columns(task))
    )
    n_columns = len(output_columns)
    reg_sizes = reg.ends - reg.starts
    feature_sizes = features.ends - features.starts
    if np.any(reg_sizes % n_raw_columns):
        raise ValueError("Regression arrays do not have the expected seven-column layout")
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

    count = np.zeros(n_columns, dtype=np.int64)
    mean = np.zeros(n_columns, dtype=np.float64)
    m2 = np.zeros(n_columns, dtype=np.float64)
    offsets = np.arange(n_raw_columns, dtype=np.int64)

    for start in range(0, selected.size, chunk_size):
        indices = selected[start:start + chunk_size]
        first_values = reg.memmap[reg.starts[indices, None] + offsets]
        values = np.asarray(first_values, dtype=np.float64)
        if task is not None:
            values = transform_regression_target_numpy(values, task)
        update_running_stats(count, mean, m2, values)

    variance = m2 / np.maximum(count, 1)
    return {
        "version": 2 if task is not None else 1,
        "data_root": str(data_root.resolve()),
        "split": split,
        "task": task or "raw",
        "columns": output_columns,
        "filters": {"low_thr": low_thr, "high_thr": high_thr},
        "selected_events": int(selected.size),
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
    parser.add_argument(
        "--task",
        help=(
            "Optional regression task to compute target-specific stats, e.g. "
            "mom or p_phi_eta. Omit to preserve the legacy seven-column stats file."
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
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        json.dump(stats, stream, indent=2)
        stream.write("\n")
    print(json.dumps(stats, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
