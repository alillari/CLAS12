#!/usr/bin/env python3
"""CLI wrapper for downstream event-level track-finding training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from .track_finding_experiment import (
        TrackFindingExperimentConfig,
        train_experiment,
    )
except ImportError:
    from track_finding_experiment import (
        TrackFindingExperimentConfig,
        train_experiment,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml_config", default="", type=str, help="Path to YAML config file")
    parser.add_argument("--config", default="", type=str, help="Model config name")
    parser.add_argument("--run_num", default="0", type=str, help="Sub run number")
    parser.add_argument("--root_dir", default="./downstream_log/", type=str, help="Root dir to store results")
    parser.add_argument("--global_log_dir", default="globallogs", type=str, help="Global dir to store logging only")
    parser.add_argument("--eventnumber", default=50000, type=int, help="Downstream training event number")
    parser.add_argument("--usepretrain", action="store_true", help="Use pretrained backbone")
    parser.add_argument("--no-pretrain", dest="usepretrain", action="store_false", help="Disable pretrained backbone")
    parser.set_defaults(usepretrain=False)
    parser.add_argument("--train_batch_size", default=32, type=int, help="Train batch size")
    parser.add_argument("--pretrained_ckpt", default=None, type=str, help="Required pretrained checkpoint path with --usepretrain.")
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="Explicit checkpoint/log directory.")
    parser.add_argument("--log_file_name", default=None, type=str, help="Deterministic training log filename.")
    parser.add_argument("--checkpoint_file_name", default=None, type=str, help="Deterministic checkpoint filename.")
    parser.add_argument("--artifact_summary", default=None, type=str, help="JSON path for training metadata.")
    parser.add_argument("--resolved_config_path", default=None, type=str, help="Optional JSON path for resolved params.")
    parser.add_argument("--seed", default=None, type=int, help="Optional random seed for this training run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_experiment(
        TrackFindingExperimentConfig(
            yaml_config=args.yaml_config,
            config=args.config,
            run_num=args.run_num,
            root_dir=args.root_dir,
            global_log_dir=args.global_log_dir,
            eventnumber=args.eventnumber,
            usepretrain=args.usepretrain,
            train_batch_size=args.train_batch_size,
            pretrained_ckpt=args.pretrained_ckpt,
            checkpoint_dir=args.checkpoint_dir,
            log_file_name=args.log_file_name,
            checkpoint_file_name=args.checkpoint_file_name,
            artifact_summary=args.artifact_summary,
            resolved_config_path=args.resolved_config_path,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
