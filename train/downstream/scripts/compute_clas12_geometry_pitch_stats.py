#!/usr/bin/env python3
"""Compute train-only global pitch normalization for v7 CLAS12 geometry tokens."""

import argparse
import json
from pathlib import Path

import numpy as np
from mmap_ninja import RaggedMmap


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=4_000_000)
    parser.add_argument("--progress-every-chunks", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.chunk_rows < 1:
        raise ValueError("--chunk-rows must be positive")
    sidecar_name = "cluster_geometry_context_target_pretrain"
    sidecar = RaggedMmap(args.data_root / sidecar_name)
    values = np.asarray(sidecar.memmap)
    if values.dtype != np.float32 or values.size % 11:
        raise RuntimeError(
            f"Expected fixed float32 geometry rows of width 11, got dtype={values.dtype}, size={values.size}"
        )
    rows = values.reshape(-1, 11)
    count = 0
    total = 0.0
    total_sq = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    for chunk_index, start in enumerate(range(0, len(rows), args.chunk_rows), start=1):
        pitch = rows[start:start + args.chunk_rows, 10].astype(np.float64, copy=False)
        if not np.isfinite(pitch).all() or np.any(pitch <= 0.0):
            raise RuntimeError(f"Non-finite or non-positive pitch in rows starting at {start}")
        count += pitch.size
        total += float(pitch.sum())
        total_sq += float(np.square(pitch).sum())
        minimum = min(minimum, float(pitch.min()))
        maximum = max(maximum, float(pitch.max()))
        if args.progress_every_chunks and chunk_index % args.progress_every_chunks == 0:
            print(f"processed {min(start + len(pitch), len(rows))}/{len(rows)} pitch rows", flush=True)
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    result = {
        "schema": "clas12_geometry_pitch_stats_v1",
        "data_root": str(args.data_root.resolve()),
        "split": "pretrain",
        "sidecar": sidecar_name,
        "geometry_layout": [
            "x1", "y1", "z1", "x2", "y2", "z2", "sx", "sy", "sz",
            "physical_length_cm", "pitch_cm",
        ],
        "units": "cm",
        "count": count,
        "mean_cm": mean,
        "std_cm": float(np.sqrt(variance)),
        "min_cm": minimum,
        "max_cm": maximum,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
