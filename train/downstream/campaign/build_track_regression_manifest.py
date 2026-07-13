#!/usr/bin/env python3
"""Build a manifest for CLAS12 track-regression adapter campaigns."""

from __future__ import annotations

import argparse
from pathlib import Path

from campaign_util import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_BASE_ANALYSIS_YAML,
    DEFAULT_BASE_MODEL_YAML,
    DEFAULT_CAMPAIGN_NAME,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_EVENTNUMBER,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_TRAIN_BATCH_SIZE,
    build_run_row,
    campaign_dir,
    discover_checkpoint,
    utc_now,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        default=str(DEFAULT_CHECKPOINT_ROOT),
        help="Directory containing one subdirectory per pretrained backbone.",
    )
    parser.add_argument(
        "--campaign-name",
        default=DEFAULT_CAMPAIGN_NAME,
        help="Campaign name used under <artifact-root>/campaigns/.",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help="External root where campaign outputs are written.",
    )
    parser.add_argument(
        "--base-model-yaml",
        default=str(DEFAULT_BASE_MODEL_YAML),
        help="Base downstream pretrained track-regression YAML.",
    )
    parser.add_argument(
        "--base-analysis-yaml",
        default=str(DEFAULT_BASE_ANALYSIS_YAML),
        help="Base track-regression analysis YAML.",
    )
    parser.add_argument(
        "--eventnumber",
        action="append",
        default=None,
        help=(
            "Number of labeled events for adapter training. May be repeated or "
            "comma-separated, e.g. --eventnumber 100,1000,10000,70000."
        ),
    )
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--max-epochs", type=int, help="Override downstream max_epochs for rendered model YAMLs.")
    parser.add_argument("--early-stopping-patience", type=int, help="Override early_stopping_patience.")
    parser.add_argument("--early-stopping-warmup-steps", type=int, help="Override early_stopping_warmup_steps.")
    parser.add_argument("--max-train-batches", type=int, help="Maximum train batches per epoch.")
    parser.add_argument("--max-val-batches", type=int, help="Maximum validation batches per epoch.")
    parser.add_argument(
        "--training-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional rendered model YAML override. Can be repeated.",
    )
    parser.add_argument(
        "--manifest",
        help="Optional explicit manifest output path. Defaults to campaign_dir/manifest.yaml.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write an empty manifest if no run directories are found.",
    )
    return parser.parse_args()


def parse_eventnumbers(values: list[str] | None) -> list[int]:
    if not values:
        return [DEFAULT_EVENTNUMBER]
    parsed = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                parsed.append(int(item))
    if not parsed:
        raise ValueError("At least one --eventnumber value is required")
    seen = set()
    unique = []
    for value in parsed:
        if value <= 0:
            raise ValueError(f"--eventnumber values must be positive; got {value}")
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def parse_scalar(value: str):
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


def parse_training_overrides(args: argparse.Namespace) -> dict:
    overrides = {}
    direct = {
        "max_epochs": args.max_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_warmup_steps": args.early_stopping_warmup_steps,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
    }
    overrides.update({key: value for key, value in direct.items() if value is not None})
    for item in args.training_override:
        if "=" not in item:
            raise ValueError(f"--training-override must be KEY=VALUE; got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--training-override has an empty key: {item!r}")
        overrides[key] = parse_scalar(value.strip())
    return overrides


def main() -> None:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    base_dir = campaign_dir(artifact_root, args.campaign_name)
    manifest_path = Path(args.manifest).resolve() if args.manifest else base_dir / "manifest.yaml"
    eventnumbers = parse_eventnumbers(args.eventnumber)
    training_overrides = parse_training_overrides(args)

    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {checkpoint_root}")

    source_dirs = sorted(path for path in checkpoint_root.iterdir() if path.is_dir())
    if not source_dirs and not args.allow_empty:
        raise FileNotFoundError(f"No pretrained run directories found in {checkpoint_root}")

    runs = []
    errors = []
    for source_dir in source_dirs:
        try:
            checkpoint = discover_checkpoint(source_dir)
            for eventnumber in eventnumbers:
                run_id = (
                    source_dir.name
                    if len(eventnumbers) == 1
                    else f"{source_dir.name}_label{eventnumber}"
                )
                runs.append(
                    build_run_row(
                        source_dir=source_dir,
                        checkpoint=checkpoint,
                        base_dir=base_dir,
                        eventnumber=eventnumber,
                        train_batch_size=args.train_batch_size,
                        max_samples=args.max_samples,
                        run_id=run_id,
                    )
                )
        except Exception as exc:
            errors.append(f"{source_dir.name}: {exc}")

    if errors:
        raise RuntimeError(
            "Failed to build manifest for one or more pretrained directories:\n"
            + "\n".join(f"  - {error}" for error in errors)
        )

    manifest = {
        "campaign_name": args.campaign_name,
        "campaign_dir": str(base_dir),
        "artifact_root": str(artifact_root),
        "checkpoint_root": str(checkpoint_root),
        "base_model_yaml": args.base_model_yaml,
        "base_analysis_yaml": args.base_analysis_yaml,
        "created_at": utc_now(),
        "defaults": {
            "eventnumbers": [int(value) for value in eventnumbers],
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
