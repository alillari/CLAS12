#!/usr/bin/env python3
"""Build a seed-ablation campaign manifest from a completed Optuna study."""

from __future__ import annotations

import argparse
import json
import random
import socket
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN_DIR = REPO_ROOT / "train" / "downstream" / "campaign"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CAMPAIGN_DIR))

from campaign_util import (  # noqa: E402
    DEFAULT_ADAPTER_ONLY_MODEL_YAML,
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_BASE_ANALYSIS_YAML,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_TRAIN_BATCH_SIZE,
    campaign_dir,
    load_base_model_config,
    run_paths,
    utc_now,
    write_yaml,
)
from train.downstream.tuning.run_track_regression_optuna import (  # noqa: E402
    fixed_overrides,
    git_commit,
)


DEFAULT_SEEDS = (11, 17, 23, 31, 43)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", required=True, help="Optuna storage URL, e.g. sqlite:////path/study.db")
    parser.add_argument("--study-name", required=True, help="Existing Optuna study name")
    parser.add_argument("--ablation-name", required=True, help="Campaign name for seed-ablation outputs")
    parser.add_argument("--output-root", default=str(DEFAULT_ARTIFACT_ROOT), help="Root where campaigns are written")
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
    parser.add_argument(
        "--base-analysis-yaml",
        default=str(DEFAULT_BASE_ANALYSIS_YAML),
        help="Base track-regression analysis YAML",
    )
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--quantile-k", type=int, default=10)
    parser.add_argument("--random-k", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=12345)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--manifest", help="Optional explicit manifest output path")

    parser.add_argument("--data-root", help="Override data_root and data_root_train")
    parser.add_argument("--data-root-test", help="Override data_root_test")
    parser.add_argument("--stat-dir", help="Override stat_dir")
    parser.add_argument("--regression-target-stats", help="Override regression_target_stats")
    parser.add_argument("--eventnumber", type=int, default=50000)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--max-optimizer-steps", type=int, default=3000)
    parser.add_argument("--val-interval-steps", type=int, default=500)
    parser.add_argument("--early-stopping-min-steps", type=int, default=1000)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int, default=2001)
    parser.add_argument("--num-data-workers", type=int)
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    seeds = []
    for item in value.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    seen = set()
    unique = []
    for seed in seeds:
        if seed not in seen:
            unique.append(seed)
            seen.add(seed)
    return unique


def complete_trials(study: Any) -> list[Any]:
    import optuna

    trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    reverse = study.direction == optuna.study.StudyDirection.MAXIMIZE
    return sorted(trials, key=lambda trial: float(trial.value), reverse=reverse)


def quantile_indices(n_trials: int, n_quantiles: int) -> list[int]:
    if n_trials <= 0 or n_quantiles <= 0:
        return []
    if n_quantiles == 1:
        return [0]
    count = min(n_trials, n_quantiles)
    return sorted({round(index * (n_trials - 1) / (count - 1)) for index in range(count)})


def select_trials(
    sorted_trials: list[Any],
    top_k: int,
    quantile_k: int,
    random_k: int,
    sample_seed: int,
) -> list[tuple[Any, str]]:
    selected: dict[int, tuple[Any, list[str]]] = {}

    def add(trial: Any, reason: str) -> None:
        if trial.number not in selected:
            selected[trial.number] = (trial, [])
        selected[trial.number][1].append(reason)

    for trial in sorted_trials[:max(0, top_k)]:
        add(trial, "top")
    for index in quantile_indices(len(sorted_trials), quantile_k):
        add(sorted_trials[index], "quantile")

    remaining = [trial for trial in sorted_trials if trial.number not in selected]
    if random_k > 0 and remaining:
        rng = random.Random(sample_seed)
        for trial in rng.sample(remaining, k=min(random_k, len(remaining))):
            add(trial, "random")

    return [(trial, "+".join(reasons)) for trial, reasons in selected.values()]


def trial_training_overrides(trial: Any) -> dict[str, Any]:
    overrides = dict(trial.params)
    if "max_lr" in overrides and "min_lr_ratio" in overrides:
        overrides["min_lr"] = float(overrides["max_lr"]) * float(overrides["min_lr_ratio"])
    return overrides


RUN_SPECIFIC_PARAM_KEYS = {
    "artifact_root",
    "downstream_dir",
    "checkpoint_dir",
    "model_version",
    "experiment_dir",
    "checkpoint_path",
    "training_log_path",
    "trained_checkpoint_path",
    "seed",
}


def source_resolved_config_path(trial: Any) -> Path | None:
    trial_dir = trial.user_attrs.get("trial_dir")
    if not trial_dir:
        return None
    path = Path(trial_dir) / "config" / "resolved_config.json"
    return path if path.is_file() else None


def source_training_overrides(trial: Any) -> tuple[dict[str, Any], str | None]:
    resolved_path = source_resolved_config_path(trial)
    if resolved_path is None:
        return trial_training_overrides(trial), None
    with resolved_path.open() as stream:
        resolved = json.load(stream)
    for key in RUN_SPECIFIC_PARAM_KEYS:
        resolved.pop(key, None)
    return resolved, str(resolved_path.resolve())


def build_run_row(
    base_dir: Path,
    base_params: dict[str, Any],
    args: argparse.Namespace,
    trial: Any,
    seed: int,
    selection_reason: str,
) -> dict[str, Any]:
    trial_name = f"trial_{trial.number:06d}"
    run_id = f"{trial_name}_seed_{seed}"
    paths = run_paths(base_dir, run_id)
    training_overrides, source_config_path = source_training_overrides(trial)
    params = dict(base_params)
    params.update(fixed_overrides(args))
    params.update(training_overrides)
    eventnumber = int(params.get("limit_size", args.eventnumber))
    train_batch_size = int(params.get("batch_size", args.train_batch_size))

    row = {
        "run_id": run_id,
        "backbone_run_id": "adapteronly",
        "source_dir": None,
        "pretrained_checkpoint": None,
        "use_pretrained_backbone": False,
        "base_model_yaml": str(Path(args.yaml_config)),
        "base_model_config": args.config,
        "model_family": "adapteronly",
        "base_dim": int(params.get("base_dim", 128)),
        "embed_dim": int(params.get("embed_dim", params.get("base_dim", 128))),
        "num_layers_backbone": int(params.get("num_layers_backbone", 0)),
        "pretrain_events": 0,
        "eventnumber": eventnumber,
        "labeled_events": eventnumber,
        "train_batch_size": train_batch_size,
        "max_samples": int(args.max_samples),
        "status": "pending",
        "model_config": f"{args.config}_{run_id}",
        "seed": int(seed),
        "training_overrides": training_overrides,
        "source_study_name": args.study_name,
        "source_trial_number": int(trial.number),
        "source_trial_value": float(trial.value),
        "source_trial_dir": trial.user_attrs.get("trial_dir"),
        "source_resolved_config": source_config_path,
        "source_trial_checkpoint": trial.user_attrs.get("checkpoint_path"),
        "selection_reason": selection_reason,
    }
    row.update({key: str(value) for key, value in paths.items()})
    return row


def main() -> None:
    args = parse_args()
    if args.top_k < 0 or args.quantile_k < 0 or args.random_k < 0:
        raise ValueError("--top-k, --quantile-k, and --random-k must be non-negative")

    import optuna

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    sorted_trials = complete_trials(study)
    if not sorted_trials:
        raise ValueError(f"Study {args.study_name!r} has no completed trials")

    selected = select_trials(
        sorted_trials,
        top_k=args.top_k,
        quantile_k=args.quantile_k,
        random_k=args.random_k,
        sample_seed=args.sample_seed,
    )
    seeds = parse_seeds(args.seeds)

    artifact_root = Path(args.output_root).resolve()
    base_dir = campaign_dir(artifact_root, args.ablation_name)
    manifest_path = Path(args.manifest).resolve() if args.manifest else base_dir / "manifest.yaml"
    base_yaml = Path(args.yaml_config).resolve()
    base_params = load_base_model_config(base_yaml, args.config)

    runs = [
        build_run_row(base_dir, base_params, args, trial, seed, reason)
        for trial, reason in selected
        for seed in seeds
    ]
    manifest = {
        "campaign_name": args.ablation_name,
        "campaign_dir": str(base_dir),
        "artifact_root": str(artifact_root),
        "base_model_yaml": str(DEFAULT_ADAPTER_ONLY_MODEL_YAML),
        "adapter_only_model_yaml": str(Path(args.yaml_config)),
        "base_analysis_yaml": str(Path(args.base_analysis_yaml)),
        "created_at": utc_now(),
        "campaign_type": "optuna_seed_ablation",
        "source": {
            "storage": args.storage,
            "study_name": args.study_name,
            "direction": study.direction.name,
            "completed_trials": len(sorted_trials),
            "selected_trials": len(selected),
            "seeds": seeds,
            "top_k": int(args.top_k),
            "quantile_k": int(args.quantile_k),
            "random_k": int(args.random_k),
            "sample_seed": int(args.sample_seed),
            "hostname": socket.gethostname(),
            "git_commit": git_commit(),
        },
        "defaults": {
            "eventnumbers": [int(args.eventnumber)],
            "train_batch_size": int(args.train_batch_size),
            "max_samples": int(args.max_samples),
        },
        "training_overrides": fixed_overrides(args),
        "runs": runs,
    }
    write_yaml(manifest_path, manifest)
    print(
        f"Wrote {len(runs)} seed-ablation runs from {len(selected)} "
        f"Optuna trials to {manifest_path}"
    )


if __name__ == "__main__":
    main()
