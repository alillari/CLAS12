#!/usr/bin/env python3
"""Run Optuna tuning for AdapterOnly CLAS12 track regression."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from train.downstream.tuning.track_regression_search_space import (  # noqa: E402
    suggest_adapteronly_optimizer_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", required=True, help="Optuna storage URL, e.g. sqlite:////path/study.db")
    parser.add_argument("--study-name", required=True, help="Resumable Optuna study name")
    parser.add_argument("--output-root", required=True, help="Root for per-trial output directories")
    parser.add_argument(
        "--yaml-config",
        default="scripts/configs/mamba_clas12_track_regression_adapteronly.yaml",
        help="Base AdapterOnly YAML config",
    )
    parser.add_argument(
        "--config",
        default="clas12_track_regression_adapteronly",
        help="Base config name inside --yaml-config",
    )
    parser.add_argument("--data-root", help="Override data_root and data_root_train")
    parser.add_argument("--data-root-test", help="Override data_root_test")
    parser.add_argument("--stat-dir", help="Override stat_dir")
    parser.add_argument("--regression-target-stats", help="Override regression_target_stats")
    parser.add_argument("--eventnumber", type=int, default=50000)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-optimizer-steps", type=int, default=3000)
    parser.add_argument("--val-interval-steps", type=int, default=500)
    parser.add_argument("--early-stopping-min-steps", type=int, default=1000)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int, default=2001)
    parser.add_argument("--num-data-workers", type=int)
    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    parser.add_argument("--enable-pruning", action="store_true")
    return parser.parse_args()


def yaml_parser() -> YAML:
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 4096
    return yaml


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return yaml_parser().load(stream) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        yaml_parser().dump(data, stream)
    tmp.replace(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
    tmp.replace(path)


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def load_base_params(yaml_config: Path, config_name: str) -> dict[str, Any]:
    data = read_yaml(yaml_config)
    if config_name not in data:
        raise KeyError(f"{yaml_config} does not contain config {config_name!r}")
    return dict(data[config_name])


def fixed_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "max_optimizer_steps": int(args.max_optimizer_steps),
        "scheduler_first_cycle_steps": int(args.max_optimizer_steps),
        "val_interval_steps": int(args.val_interval_steps),
        "early_stopping_min_steps": int(args.early_stopping_min_steps),
        "early_stopping_patience": int(args.early_stopping_patience),
        "max_val_batches": int(args.max_val_batches),
        "limit_data": True,
        "limit_size": int(args.eventnumber),
    }
    if args.max_train_batches is not None:
        overrides["max_train_batches"] = int(args.max_train_batches)
    if args.num_data_workers is not None:
        overrides["num_data_workers"] = int(args.num_data_workers)
    if args.data_root:
        data_root = str(Path(args.data_root).resolve())
        overrides["data_root"] = data_root
        overrides["data_root_train"] = data_root
    if args.data_root_test:
        overrides["data_root_test"] = str(Path(args.data_root_test).resolve())
    if args.stat_dir:
        overrides["stat_dir"] = str(Path(args.stat_dir).resolve())
    if args.regression_target_stats:
        overrides["regression_target_stats"] = str(Path(args.regression_target_stats).resolve())
    return overrides


def make_sampler(args: argparse.Namespace):
    import optuna

    if args.sampler == "random":
        return optuna.samplers.RandomSampler(seed=args.seed)
    return optuna.samplers.TPESampler(seed=args.seed)


def make_pruner(args: argparse.Namespace):
    import optuna

    if not args.enable_pruning:
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=args.early_stopping_min_steps,
        interval_steps=args.val_interval_steps,
    )


def set_trial_attrs(trial: Any, attrs: dict[str, Any]) -> None:
    for key, value in attrs.items():
        trial.set_user_attr(key, value)


def objective_factory(args: argparse.Namespace):
    base_yaml = Path(args.yaml_config).resolve()
    output_root = Path(args.output_root).resolve()
    study_dir = output_root / args.study_name
    base_params = load_base_params(base_yaml, args.config)
    fixed = fixed_overrides(args)

    def objective(trial: Any) -> float:
        from train.downstream.track_regression_experiment import (
            TrackRegressionExperimentConfig,
            train_experiment,
        )

        trial_name = f"trial_{trial.number:06d}"
        trial_dir = study_dir / trial_name
        config_dir = trial_dir / "config"
        checkpoint_dir = trial_dir / "checkpoints"
        train_dir = trial_dir / "train"

        suggested = suggest_adapteronly_optimizer_params(trial)
        params = dict(base_params)
        params.update(fixed)
        params.update(suggested)
        params.update({
            "artifact_root": str(study_dir),
            "downstream_dir": str(trial_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "model_version": f"{args.study_name}_{trial_name}",
        })

        trial_config_name = f"{args.config}_{trial_name}"
        trial_yaml = config_dir / "model.yaml"
        resolved_config = config_dir / "resolved_config.json"
        artifact_summary = train_dir / "artifacts.json"
        write_yaml(trial_yaml, {trial_config_name: params})

        set_trial_attrs(trial, {
            "study_name": args.study_name,
            "trial_dir": str(trial_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "hostname": socket.gethostname(),
            "git_commit": git_commit(),
            "data_root": params.get("data_root"),
            "data_root_test": params.get("data_root_test"),
        })

        result = train_experiment(
            TrackRegressionExperimentConfig(
                yaml_config=str(trial_yaml),
                config=trial_config_name,
                run_num=trial_name,
                root_dir=str(train_dir),
                global_log_dir=str(study_dir / "global_logs"),
                eventnumber=args.eventnumber,
                usepretrain=False,
                train_batch_size=args.train_batch_size,
                checkpoint_dir=str(checkpoint_dir),
                log_file_name=f"{trial_name}.log",
                checkpoint_file_name=f"{trial_name}_adapter_checkpoint.pth",
                artifact_summary=str(artifact_summary),
                resolved_config_path=str(resolved_config),
            ),
            optuna_trial=trial if args.enable_pruning else None,
        )

        set_trial_attrs(trial, {
            "best_step": result.get("best_step"),
            "best_epoch": result.get("best_epoch"),
            "checkpoint_path": result.get("checkpoint_path"),
            "log_file": result.get("log_file"),
            "artifact_summary": result.get("artifact_summary"),
        })
        write_json(trial_dir / "trial_result.json", result)
        return float(result["best_val_loss"])

    return objective


def main() -> None:
    args = parse_args()
    if args.n_jobs != 1:
        raise ValueError(
            "This worker supports --n-jobs 1 only. For parallel studies, run one "
            "worker process per machine against the same Optuna storage/study."
        )
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    import optuna

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        sampler=make_sampler(args),
        pruner=make_pruner(args),
    )
    study.optimize(
        objective_factory(args),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        gc_after_trial=True,
    )

    print(f"Study: {args.study_name}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
