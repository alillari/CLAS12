#!/usr/bin/env python3
"""Build a manifest for CLAS12 track-finding adapter campaigns."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from campaign_util import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_CHECKPOINT_ROOT,
    campaign_dir,
    discover_checkpoint,
    parse_run_name,
    run_paths,
    utc_now,
    write_yaml,
)


DEFAULT_CAMPAIGN_NAME = "campaign_1_track_finding"
DEFAULT_MODEL_YAML = "scripts/configs/mamba_clas12_track_finding_adapteronly.yaml"
DEFAULT_ANALYSIS_YAML = "train/downstream/eval/track_finding_analysis_adapteronly.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument(
        "--checkpoint-run",
        action="append",
        default=[],
        help="Include only this pretrained run directory name. Can be repeated.",
    )
    parser.add_argument("--campaign-name", default=DEFAULT_CAMPAIGN_NAME)
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--base-model-yaml", default=DEFAULT_MODEL_YAML)
    parser.add_argument("--base-analysis-yaml", default=DEFAULT_ANALYSIS_YAML)
    parser.add_argument("--data-root", help="RaggedMmap event dataset root for both train and test unless overridden.")
    parser.add_argument("--data-root-train", help="RaggedMmap event dataset root for training.")
    parser.add_argument("--data-root-test", help="RaggedMmap event dataset root for evaluation.")
    parser.add_argument("--stat-dir", help="Stats/calibration directory passed to the rendered model YAML.")
    parser.add_argument("--eventnumber", default="50000")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--preflight-max-events", type=int, help="Maximum events sampled by campaign preflight.")
    parser.add_argument("--min-multitrack-fraction", type=float, help="Minimum preflight fraction of events with more than one signal track.")
    parser.add_argument("--max-epochs", type=int, help="Override downstream max_epochs.")
    parser.add_argument("--max-optimizer-steps", type=int, help="Enable step-oriented training and set the optimizer-step budget.")
    parser.add_argument("--val-interval-steps", type=int, help="Validation/checkpoint cadence in optimizer steps.")
    parser.add_argument("--scheduler-first-cycle-steps", type=int, help="First-cycle length for the adapter scheduler.")
    parser.add_argument("--warmup-steps", type=int, help="Warmup steps for the adapter scheduler.")
    parser.add_argument("--early-stopping-min-steps", type=int, help="Minimum optimizer steps before early stopping/pruning.")
    parser.add_argument("--early-stopping-patience", type=int, help="Validation checks without improvement before early stopping.")
    parser.add_argument("--early-stopping-min-delta", type=float, help="Minimum ARI/loss improvement for checkpoint/early stopping.")
    parser.add_argument("--max-train-batches", type=int, help="Maximum train batches per epoch.")
    parser.add_argument("--max-val-batches", type=int, help="Maximum validation batches per validation pass.")
    parser.add_argument("--num-data-workers", type=int, help="DataLoader worker count.")
    parser.add_argument("--training-override", action="append", default=[], metavar="KEY=VALUE", help="Additional rendered model YAML override. Can be repeated.")
    parser.add_argument("--manifest")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--no-adapter-only", action="store_true")
    return parser.parse_args()


def parse_eventnumbers(value: str) -> list[int]:
    out = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out or [50000]


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_training_overrides(args: argparse.Namespace) -> dict[str, Any]:
    direct = {
        "data_root": args.data_root,
        "data_root_train": args.data_root_train or args.data_root,
        "data_root_test": args.data_root_test or args.data_root,
        "stat_dir": args.stat_dir,
        "preflight_max_events": args.preflight_max_events,
        "min_multitrack_fraction": args.min_multitrack_fraction,
        "max_epochs": args.max_epochs,
        "max_optimizer_steps": args.max_optimizer_steps,
        "val_interval_steps": args.val_interval_steps,
        "scheduler_first_cycle_steps": args.scheduler_first_cycle_steps,
        "warmup_steps": args.warmup_steps,
        "early_stopping_min_steps": args.early_stopping_min_steps,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "num_data_workers": args.num_data_workers,
    }
    overrides = {key: value for key, value in direct.items() if value is not None}
    for item in args.training_override:
        if "=" not in item:
            raise ValueError(f"--training-override must be KEY=VALUE; got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--training-override has an empty key: {item!r}")
        overrides[key] = parse_scalar(value.strip())
    return overrides


def adapter_row(base_dir: Path, eventnumber: int, train_batch_size: int, max_samples: int) -> dict:
    run_id = f"adapteronly_label{eventnumber}"
    row = {
        "run_id": run_id,
        "backbone_run_id": "adapteronly",
        "source_dir": None,
        "pretrained_checkpoint": None,
        "use_pretrained_backbone": False,
        "base_model_yaml": DEFAULT_MODEL_YAML,
        "base_model_config": "clas12_track_finding_adapteronly",
        "model_family": "adapteronly",
        "base_dim": 128,
        "embed_dim": 128,
        "num_layers_backbone": 6,
        "pretrain_events": 0,
        "eventnumber": int(eventnumber),
        "labeled_events": int(eventnumber),
        "train_batch_size": int(train_batch_size),
        "max_samples": int(max_samples),
        "status": "pending",
        "model_config": f"clas12_track_finding_adapteronly_{run_id}",
    }
    row.update({key: str(value) for key, value in run_paths(base_dir, run_id).items()})
    return row


def pretrained_row(source_dir: Path, checkpoint: Path, base_dir: Path, eventnumber: int, train_batch_size: int, max_samples: int) -> dict:
    metadata = parse_run_name(source_dir.name)
    run_id = f"{source_dir.name}_label{eventnumber}"
    row = {
        "run_id": run_id,
        "backbone_run_id": source_dir.name,
        "source_dir": str(source_dir.resolve()),
        "pretrained_checkpoint": str(checkpoint.resolve()),
        "use_pretrained_backbone": True,
        "base_model_yaml": DEFAULT_MODEL_YAML,
        "base_model_config": "clas12_track_finding_adapteronly",
        "model_family": "mamba1",
        "base_dim": metadata["base_dim"],
        "embed_dim": metadata["embed_dim"],
        "num_layers_backbone": metadata["num_layers_backbone"],
        "pretrain_events": metadata["pretrain_events"],
        "eventnumber": int(eventnumber),
        "labeled_events": int(eventnumber),
        "train_batch_size": int(train_batch_size),
        "max_samples": int(max_samples),
        "status": "pending",
        "model_config": f"clas12_track_finding_pretrained_{run_id}",
    }
    row.update({key: str(value) for key, value in run_paths(base_dir, run_id).items()})
    return row


def main() -> None:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    base_dir = campaign_dir(artifact_root, args.campaign_name)
    manifest_path = Path(args.manifest).resolve() if args.manifest else base_dir / "manifest.yaml"
    eventnumbers = parse_eventnumbers(args.eventnumber)
    training_overrides = parse_training_overrides(args)

    runs = []
    if not args.no_adapter_only:
        for eventnumber in eventnumbers:
            row = adapter_row(base_dir, eventnumber, args.train_batch_size, args.max_samples)
            row["base_model_yaml"] = args.base_model_yaml
            runs.append(row)

    if checkpoint_root.is_dir():
        source_dirs = sorted(path for path in checkpoint_root.iterdir() if path.is_dir())
        if args.checkpoint_run:
            requested = set(args.checkpoint_run)
            available = {path.name for path in source_dirs}
            missing = sorted(requested - available)
            if missing:
                raise FileNotFoundError(
                    "Requested pretrained run(s) not found under "
                    f"{checkpoint_root}: {', '.join(missing)}"
                )
            source_dirs = [path for path in source_dirs if path.name in requested]
        for source_dir in source_dirs:
            checkpoint = discover_checkpoint(source_dir)
            for eventnumber in eventnumbers:
                row = pretrained_row(
                    source_dir,
                    checkpoint,
                    base_dir,
                    eventnumber,
                    args.train_batch_size,
                    args.max_samples,
                )
                row["base_model_yaml"] = args.base_model_yaml
                runs.append(row)
    elif not args.allow_empty:
        raise FileNotFoundError(f"Checkpoint root does not exist: {checkpoint_root}")

    if not runs and not args.allow_empty:
        raise FileNotFoundError("No track-finding campaign rows were generated")

    manifest = {
        "campaign_name": args.campaign_name,
        "campaign_dir": str(base_dir),
        "artifact_root": str(artifact_root),
        "checkpoint_root": str(checkpoint_root),
        "base_model_yaml": args.base_model_yaml,
        "base_analysis_yaml": args.base_analysis_yaml,
        "created_at": utc_now(),
        "defaults": {
            "eventnumbers": eventnumbers,
            "train_batch_size": int(args.train_batch_size),
            "max_samples": int(args.max_samples),
        },
        "training_overrides": training_overrides,
        "runs": runs,
    }
    write_yaml(manifest_path, manifest)
    print(f"Wrote {len(runs)} runs to {manifest_path}")


if __name__ == "__main__":
    main()
