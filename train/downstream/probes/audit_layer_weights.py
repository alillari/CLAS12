#!/usr/bin/env python3
"""Audit trained adapter layer-mixture weights from downstream checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Adapter checkpoint files or directories to search recursively",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for audit outputs")
    parser.add_argument(
        "--campaign_dir",
        help="Audit manifest-selected checkpoints; exclude unused adapter-only mixers",
    )
    parser.add_argument(
        "--completed_only", action="store_true",
        help="In campaign mode, select runs with completed training and evaluation artifacts",
    )
    parser.add_argument(
        "--pattern",
        default="*.pth",
        help="Checkpoint filename glob used when a path is a directory",
    )
    args = parser.parse_args()
    if bool(args.paths) == bool(args.campaign_dir):
        parser.error("Supply checkpoint paths OR --campaign_dir")
    if args.completed_only and not args.campaign_dir:
        parser.error("--completed_only requires --campaign_dir")
    return args


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
    params = {**params.get("params", {}), **params}
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
        "best_ari": checkpoint.get("best_ARI", checkpoint.get("best_ari")),
        "track_target_mode": checkpoint.get("track_target_mode", params.get("track_target_mode")),
        "validation_ari_mode": checkpoint.get("validation_ari_mode", params.get("validation_ari_mode")),
        "assignment_option": params.get("assignment_option"),
        "assignment_threshold": params.get("assignment_threshold"),
    }


def audit_checkpoint(path: Path, context: dict | None = None) -> tuple[list[dict], dict] | None:
    # Mapping modern checkpoints avoids reading optimizer tensors over mounted storage.
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False, mmap=zipfile.is_zipfile(path)
    )
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
    if weights.numel() == 0:
        return None
    if not torch.isfinite(weights).all():
        raise ValueError(f"Nonfinite layer weights: {path}")

    meta = checkpoint_metadata(checkpoint)
    for key in ("num_layers_backbone", "embed_dim"):
        if context and context.get(key) is not None and meta.get(key) is not None:
            if int(context[key]) != int(meta[key]):
                raise ValueError(f"Manifest/checkpoint {key} mismatch: {path}")
    meta.update(context or {})
    depth = meta.get("num_layers_backbone")
    if depth is not None and int(depth) != weights.numel():
        raise ValueError(f"Expected {depth} layer weights, found {weights.numel()}: {path}")
    projection = next(
        (v for k, v in state.items() if k.removeprefix("module.") == "input_proj.1.weight"),
        None,
    )
    if projection is not None and meta.get("embed_dim") is not None:
        if projection.shape[1] != int(meta["embed_dim"]):
            raise ValueError(f"Projection/backbone width mismatch: {path}")

    probs = torch.softmax(weights, dim=0)
    entropy = float(-(probs * torch.log(probs.clamp_min(1.0e-12))).sum().item())
    effective_layers = float(torch.exp(torch.tensor(entropy)).item())
    top_layer = int(torch.argmax(probs).item())
    summary = {
        "checkpoint": str(path),
        "num_layers": int(probs.numel()),
        "top_layer": top_layer,
        "top_layer_1based": top_layer + 1,
        "top_weight": float(probs[top_layer].item()),
        "entropy": entropy,
        "effective_layers": effective_layers,
        "projection_input_dim": int(projection.shape[1]) if projection is not None else None,
        "projection_output_dim": int(projection.shape[0]) if projection is not None else None,
        **meta,
    }
    rows = []
    for layer_idx, (raw, prob) in enumerate(zip(weights.tolist(), probs.tolist())):
        rows.append(
            {
                "checkpoint": str(path),
                "layer": layer_idx,
                "layer_1based": layer_idx + 1,
                "raw_weight": raw,
                "softmax_weight": prob,
                **summary,
            }
        )
    return rows, summary


def campaign_checkpoints(
    campaign_dir: Path, *, completed_only: bool = False
) -> tuple[list[tuple[Path, dict]], list[dict]]:
    """Select exactly the manifest checkpoint per run, rebasing a mounted campaign."""
    import yaml

    manifest = yaml.safe_load((campaign_dir / "manifest.yaml").read_text())
    original_root = Path(manifest["campaign_dir"])

    def local_path(raw: str) -> Path:
        path = Path(raw)
        if path.is_absolute():
            try:
                path = path.relative_to(original_root)
            except ValueError as exc:
                raise ValueError(f"Artifact is outside manifest campaign_dir: {raw}") from exc
        if ".." in path.parts:
            raise ValueError(f"Artifact escapes campaign directory: {raw}")
        return campaign_dir / path

    selected, skipped = [], []
    run_ids = set()
    for run in manifest["runs"]:
        run_id = run["run_id"]
        if run_id in run_ids:
            raise ValueError(f"Duplicate run_id: {run_id}")
        run_ids.add(run_id)
        usepretrain = run.get("use_pretrained_backbone")
        if not isinstance(usepretrain, bool):
            raise ValueError(f"Missing/invalid use_pretrained_backbone: {run_id}")
        if not usepretrain:
            skipped.append({"run_id": run_id, "reason": "adapter_only_mixer_unused"})
            continue
        path = local_path(run["adapter_checkpoint"])
        artifact_path = local_path(run["artifact_summary"]) if run.get("artifact_summary") else None
        if completed_only:
            if artifact_path is None or not artifact_path.is_file():
                skipped.append({"run_id": run_id, "reason": "training_not_complete"})
                continue
            evaluation_path = local_path(run["evaluation_dir"]) / "summary.json"
            if not evaluation_path.is_file():
                skipped.append({"run_id": run_id, "reason": "evaluation_not_complete"})
                continue
        if not path.is_file():
            raise FileNotFoundError(path)
        if artifact_path and artifact_path.is_file():
            artifact = json.loads(artifact_path.read_text())
            if artifact.get("usepretrain") is not True:
                raise ValueError(f"Training artifact contradicts pretrained manifest: {run_id}")
            if completed_only:
                if not isinstance(artifact.get("final_step"), int) or artifact["final_step"] <= 0:
                    raise ValueError(f"Training artifact has no final_step: {run_id}")
                if local_path(artifact["checkpoint"]) != path:
                    raise ValueError(f"Training artifact checkpoint differs from manifest: {run_id}")
        context = {key: run.get(key) for key in (
            "run_id", "backbone_run_id", "embed_dim", "num_layers_backbone",
            "pretrain_events", "labeled_events", "use_pretrained_backbone",
        )}
        context.update({
            "campaign": manifest.get("campaign_name", campaign_dir.name),
            "pretrained_ckpt": run.get("pretrained_checkpoint"),
            "eventnumber": run.get("eventnumber", run.get("labeled_events")),
        })
        if completed_only:
            context["final_step"] = artifact["final_step"]
        selected.append((path, context))
    selected.sort(key=lambda item: (
        int(item[1].get("embed_dim") or 0), int(item[1].get("eventnumber") or 0),
        item[1]["run_id"],
    ))
    return selected, skipped


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
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
    ax.set_xlabel("Backbone layer (1-based)")
    ax.set_xticks(range(max_layers), range(1, max_layers + 1))
    ax.set_ylabel("Checkpoint")
    ax.set_yticks(range(len(checkpoints)))
    ax.set_yticklabels([
        f"w{s['embed_dim']} | {int(s['eventnumber']):,} labels"
        if s.get("run_id") and s.get("embed_dim") and s.get("eventnumber")
        else Path(s["checkpoint"]).name for s in summaries
    ], fontsize=8)
    ax.set_title("Learned backbone layer mixture")
    fig.colorbar(im, ax=ax, label="softmax(weighted_avg_weights)")
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
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
    if args.campaign_dir:
        selected, skipped = campaign_checkpoints(
            Path(args.campaign_dir).expanduser().resolve(), completed_only=args.completed_only
        )
    else:
        selected = [(path, {}) for path in discover_checkpoints(args.paths, args.pattern)]
    for index, (checkpoint, context) in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {context.get('run_id', checkpoint.name)}", flush=True)
        result = audit_checkpoint(checkpoint, context)
        if result is None:
            if args.campaign_dir:
                raise ValueError(f"Pretrained adapter has no nonempty layer mixer: {checkpoint}")
            skipped.append(str(checkpoint))
            continue
        rows, summary = result
        all_rows.extend(rows)
        summaries.append(summary)

    if not summaries:
        raise RuntimeError("No checkpoints with weighted_avg_weights were found")

    write_csv(output_dir / "layer_weight_audit.csv", all_rows)
    write_csv(output_dir / "layer_weight_summary.csv", summaries)
    with (output_dir / "layer_weight_audit.json").open("w") as stream:
        json.dump(
            json_safe({"summaries": summaries, "skipped": skipped,
                       "campaign_dir": args.campaign_dir,
                       "completed_only": args.completed_only,
                       "layer_indexing": "layer/top_layer are zero-based; *_1based and plots are one-based"}),
            stream,
            indent=2,
            allow_nan=False,
        )
    plot_heatmap(output_dir / "layer_weight_heatmap.png", summaries, all_rows)
    print(f"Audited {len(summaries)} checkpoints; skipped {len(skipped)}. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
