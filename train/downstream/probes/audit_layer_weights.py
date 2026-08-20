#!/usr/bin/env python3
"""Audit trained adapter layer-mixture weights from downstream checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="Adapter checkpoint files or directories to search recursively",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for audit outputs")
    parser.add_argument(
        "--pattern",
        default="*.pth",
        help="Checkpoint filename glob used when a path is a directory",
    )
    return parser.parse_args()


def discover_checkpoints(paths: list[str], pattern: str) -> list[Path]:
    checkpoints: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            checkpoints.append(path)
        elif path.is_dir():
            checkpoints.extend(sorted(path.rglob(pattern)))
        else:
            raise FileNotFoundError(path)
    return sorted(dict.fromkeys(checkpoints))


def checkpoint_metadata(checkpoint: dict) -> dict:
    params = checkpoint.get("params") or {}
    return {
        "epoch": checkpoint.get("epoch"),
        "best_loss": checkpoint.get("best_loss"),
        "best_step": checkpoint.get("best_step"),
        "global_step": checkpoint.get("global_step"),
        "pretrained_ckpt": params.get("pretrained_ckpt"),
        "embed_dim": params.get("embed_dim"),
        "num_layers_backbone": params.get("num_layers_backbone"),
        "mambaversion": params.get("mambaversion"),
        "eventnumber": params.get("limit_size"),
        "model_version": params.get("model_version"),
    }


def audit_checkpoint(path: Path) -> tuple[list[dict], dict] | None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    if state is None:
        return None
    weights = None
    for key, value in state.items():
        if key.replace("module.", "") == "weighted_avg_weights":
            weights = value.detach().cpu().float().reshape(-1)
            break
    if weights is None:
        return None

    probs = torch.softmax(weights, dim=0)
    entropy = float(-(probs * torch.log(probs.clamp_min(1.0e-12))).sum().item())
    effective_layers = float(torch.exp(torch.tensor(entropy)).item())
    top_layer = int(torch.argmax(probs).item())
    meta = checkpoint_metadata(checkpoint)
    summary = {
        "checkpoint": str(path),
        "num_layers": int(probs.numel()),
        "top_layer": top_layer,
        "top_weight": float(probs[top_layer].item()),
        "entropy": entropy,
        "effective_layers": effective_layers,
        **meta,
    }
    rows = []
    for layer_idx, (raw, prob) in enumerate(zip(weights.tolist(), probs.tolist())):
        rows.append(
            {
                "checkpoint": str(path),
                "layer": layer_idx,
                "raw_weight": raw,
                "softmax_weight": prob,
                **summary,
            }
        )
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(path: Path, summaries: list[dict], rows: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - plotting dependency is optional
        print(f"Skipping heatmap generation: {exc}")
        return

    checkpoints = [summary["checkpoint"] for summary in summaries]
    max_layers = max(int(summary["num_layers"]) for summary in summaries)
    matrix = np.full((len(checkpoints), max_layers), np.nan)
    row_index = {checkpoint: idx for idx, checkpoint in enumerate(checkpoints)}
    for row in rows:
        matrix[row_index[row["checkpoint"]], int(row["layer"])] = float(row["softmax_weight"])

    height = max(3.0, min(18.0, 0.28 * len(checkpoints) + 2.0))
    fig, ax = plt.subplots(figsize=(10, height), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Backbone layer")
    ax.set_ylabel("Checkpoint")
    ax.set_yticks(range(len(checkpoints)))
    ax.set_yticklabels([Path(item).name for item in checkpoints], fontsize=7)
    fig.colorbar(im, ax=ax, label="softmax(weighted_avg_weights)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def json_safe(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = []
    skipped = []
    for checkpoint in discover_checkpoints(args.paths, args.pattern):
        result = audit_checkpoint(checkpoint)
        if result is None:
            skipped.append(str(checkpoint))
            continue
        rows, summary = result
        all_rows.extend(rows)
        summaries.append(summary)

    if not summaries:
        raise RuntimeError("No checkpoints with weighted_avg_weights were found")

    write_csv(output_dir / "layer_weight_audit.csv", all_rows)
    with (output_dir / "layer_weight_audit.json").open("w") as stream:
        json.dump(
            json_safe({"summaries": summaries, "skipped": skipped}),
            stream,
            indent=2,
            allow_nan=False,
        )
    plot_heatmap(output_dir / "layer_weight_heatmap.png", summaries, all_rows)


if __name__ == "__main__":
    main()
