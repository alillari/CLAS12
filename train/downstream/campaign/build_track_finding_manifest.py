#!/usr/bin/env python3
"""Build a manifest for CLAS12 track-finding adapter campaigns."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument("--campaign-name", default=DEFAULT_CAMPAIGN_NAME)
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--base-model-yaml", default=DEFAULT_MODEL_YAML)
    parser.add_argument("--base-analysis-yaml", default=DEFAULT_ANALYSIS_YAML)
    parser.add_argument("--eventnumber", default="50000")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=10000)
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

    runs = []
    if not args.no_adapter_only:
        for eventnumber in eventnumbers:
            row = adapter_row(base_dir, eventnumber, args.train_batch_size, args.max_samples)
            row["base_model_yaml"] = args.base_model_yaml
            runs.append(row)

    if checkpoint_root.is_dir():
        for source_dir in sorted(path for path in checkpoint_root.iterdir() if path.is_dir()):
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
        "training_overrides": {},
        "runs": runs,
    }
    write_yaml(manifest_path, manifest)
    print(f"Wrote {len(runs)} runs to {manifest_path}")


if __name__ == "__main__":
    main()

