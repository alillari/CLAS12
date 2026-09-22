#!/usr/bin/env python3
"""Build a manifest for CLAS12 track-regression adapter campaigns."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from campaign_util import (
    DEFAULT_ADAPTER_ONLY_MODEL_YAML,
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_BASE_ANALYSIS_YAML,
    DEFAULT_BASE_MODEL_YAML,
    DEFAULT_CAMPAIGN_NAME,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_EVENTNUMBER,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_TRAIN_BATCH_SIZE,
    build_adapter_only_run_row,
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
        "--pretrained-model-config",
        default="clas12_track_regression_pretrained",
        help=(
            "Named configuration to render for every pretrained-backbone row "
            "from --base-model-yaml."
        ),
    )
    parser.add_argument(
        "--pretrain-events",
        type=int,
        help=(
            "Pretraining-event count recorded for checkpoint directories whose "
            "names do not include an n<events> token."
        ),
    )
    parser.add_argument(
        "--adapter-only-model-yaml",
        default=str(DEFAULT_ADAPTER_ONLY_MODEL_YAML),
        help="Base downstream adapter-only track-regression YAML.",
    )
    parser.add_argument(
        "--adapter-only-model-config",
        default="clas12_track_regression_adapteronly",
        help="Named configuration to render from --adapter-only-model-yaml.",
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
        "--optuna-storage",
        help="Optuna storage URL for importing a best fine-tuning recipe.",
    )
    parser.add_argument(
        "--optuna-study-name",
        help="Optuna study name for importing a best fine-tuning recipe.",
    )
    parser.add_argument(
        "--optuna-best-trial",
        action="store_true",
        help="Apply the best completed Optuna trial's tuned fine-tuning recipe to the campaign.",
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
    parser.add_argument(
        "--no-adapter-only",
        action="store_true",
        help="Do not add adapter-only baseline rows.",
    )
    parser.add_argument(
        "--adapter-only-only",
        action="store_true",
        help="Build only adapter-only rows; do not inspect or add pretrained-backbone rows.",
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


TUNED_OPTUNA_PARAM_KEYS = (
    "max_lr",
    "min_lr_ratio",
    "warmup_fraction",
    "adapter_weight_decay",
    "grad_clip_value",
    "dropout",
)

EXECUTION_CONTRACT_KEYS = (
    "max_optimizer_steps",
    "val_interval_steps",
    "early_stopping_min_steps",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "max_val_batches",
    "warmup_steps",
)


def validate_optuna_args(args: argparse.Namespace) -> None:
    provided = [args.optuna_storage, args.optuna_study_name]
    if args.optuna_best_trial:
        missing = []
        if not args.optuna_storage:
            missing.append("--optuna-storage")
        if not args.optuna_study_name:
            missing.append("--optuna-study-name")
        if missing:
            raise ValueError(
                "--optuna-best-trial requires " + " and ".join(missing)
            )
    elif any(provided):
        raise ValueError(
            "--optuna-storage and --optuna-study-name are only used with "
            "--optuna-best-trial"
        )


def best_trial_recipe(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not args.optuna_best_trial:
        return {}, None

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna best-trial import requested, but optuna is not installed."
        ) from exc

    study = optuna.load_study(
        study_name=args.optuna_study_name,
        storage=args.optuna_storage,
    )
    trial = study.best_trial
    if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
        raise ValueError(
            f"Study {args.optuna_study_name!r} best trial is not complete "
            "or has no objective value."
        )

    contract = trial.user_attrs.get("execution_contract")
    if not isinstance(contract, dict):
        raise ValueError(
            f"Best trial {trial.number} has no execution_contract provenance. "
            "Legacy studies cannot be faithfully imported; rerun tuning or set "
            "all schedule overrides explicitly."
        )
    scheduler_mode = str(contract.get("scheduler_mode", "cosine_restarts"))
    if scheduler_mode not in ("cosine_restarts", "cosine_hold"):
        raise ValueError(f"Unsupported Optuna scheduler_mode: {scheduler_mode!r}")
    tuned_keys = TUNED_OPTUNA_PARAM_KEYS + (
        ("scheduler_anneal_steps",) if scheduler_mode == "cosine_hold" else ()
    )
    missing = [key for key in tuned_keys if key not in trial.params]
    if missing:
        raise ValueError(
            f"Best trial {trial.number} is missing tuned parameter(s): "
            + ", ".join(missing)
        )
    recipe = {key: trial.params[key] for key in tuned_keys}
    recipe["min_lr"] = float(recipe["max_lr"]) * float(recipe["min_lr_ratio"])
    missing_contract = [key for key in EXECUTION_CONTRACT_KEYS if key not in contract]
    if scheduler_mode == "cosine_restarts" and (
        "n_cycles" not in contract and "scheduler_first_cycle_steps" not in contract
    ):
        missing_contract.append("n_cycles or scheduler_first_cycle_steps")
    if scheduler_mode == "cosine_hold" and "scheduler_anneal_steps" not in contract:
        missing_contract.append("scheduler_anneal_steps")
    if missing_contract:
        raise ValueError(
            f"Best trial {trial.number} execution contract is missing " + ", ".join(missing_contract)
        )
    max_steps = int(contract["max_optimizer_steps"])
    if max_steps <= 0:
        raise ValueError(f"Invalid Optuna execution contract for trial {trial.number}: {contract!r}")
    if scheduler_mode == "cosine_restarts":
        n_cycles = contract.get("n_cycles")
        source_cycle_steps = int(contract.get("scheduler_first_cycle_steps", 0))
        if source_cycle_steps <= 0:
            raise ValueError(f"Invalid Optuna execution contract for trial {trial.number}: {contract!r}")
        if n_cycles is not None:
            n_cycles = int(n_cycles)
            if n_cycles <= 0 or max_steps % n_cycles or source_cycle_steps != max_steps // n_cycles:
                raise ValueError(f"Invalid Optuna execution contract for trial {trial.number}: {contract!r}")
        recipe["scheduler_first_cycle_steps"] = source_cycle_steps
    else:
        anneal_steps = int(recipe["scheduler_anneal_steps"])
        if anneal_steps <= 0 or anneal_steps > max_steps:
            raise ValueError(f"Invalid Optuna execution contract for trial {trial.number}: {contract!r}")
        if int(contract["scheduler_anneal_steps"]) != anneal_steps:
            raise ValueError(
                f"Optuna trial {trial.number} scheduler_anneal_steps disagrees with "
                "its execution contract"
            )
        recipe["scheduler_mode"] = "cosine_hold"
    recipe.update({key: contract[key] for key in EXECUTION_CONTRACT_KEYS})
    source = {
        "storage": args.optuna_storage,
        "study_name": args.optuna_study_name,
        "best_trial_number": int(trial.number),
        "best_trial_value": float(trial.value),
        "params": dict(recipe),
        "execution_contract": dict(contract),
        "training_seeds": trial.user_attrs.get("trial_seeds"),
        "seed_objective": trial.user_attrs.get("seed_objective"),
        "seed_mean": trial.user_attrs.get("seed_mean"),
        "seed_std": trial.user_attrs.get("seed_std"),
    }
    return recipe, source


def main() -> None:
    args = parse_args()
    validate_optuna_args(args)
    if args.adapter_only_only and args.no_adapter_only:
        raise ValueError("--adapter-only-only cannot be combined with --no-adapter-only")
    checkpoint_root = Path(args.checkpoint_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    base_dir = campaign_dir(artifact_root, args.campaign_name)
    manifest_path = Path(args.manifest).resolve() if args.manifest else base_dir / "manifest.yaml"
    eventnumbers = parse_eventnumbers(args.eventnumber)
    optuna_overrides, source_optuna = best_trial_recipe(args)
    training_overrides = dict(optuna_overrides)
    manual_overrides = parse_training_overrides(args)
    training_overrides.update(manual_overrides)
    imported_scheduler_mode = str(
        (source_optuna or {}).get("execution_contract", {}).get(
            "scheduler_mode", "cosine_restarts"
        )
    )
    if (
        source_optuna is not None
        and imported_scheduler_mode == "cosine_restarts"
        and "scheduler_first_cycle_steps" not in manual_overrides
    ):
        n_cycles = source_optuna["execution_contract"].get("n_cycles")
        max_steps = int(training_overrides["max_optimizer_steps"])
        if n_cycles is not None and max_steps % int(n_cycles):
            raise ValueError(
                "Campaign max_optimizer_steps must be divisible by the imported "
                f"Optuna n_cycles={n_cycles}; got {max_steps}. Override "
                "scheduler_first_cycle_steps explicitly to use a non-integral policy."
            )
        if n_cycles is not None:
            training_overrides["scheduler_first_cycle_steps"] = max_steps // int(n_cycles)
        else:
            source_cycle_steps = int(
                source_optuna["execution_contract"]["scheduler_first_cycle_steps"]
            )
            if source_cycle_steps > max_steps:
                raise ValueError(
                    "Imported Optuna scheduler_first_cycle_steps exceeds the campaign "
                    f"max_optimizer_steps: {source_cycle_steps} > {max_steps}. Override "
                    "scheduler_first_cycle_steps explicitly or use a compatible budget."
                )
            training_overrides["scheduler_first_cycle_steps"] = source_cycle_steps
    if source_optuna is not None and "warmup_fraction" in training_overrides:
        warmup_reference_key = (
            "scheduler_anneal_steps"
            if str(training_overrides.get("scheduler_mode", "cosine_restarts")) == "cosine_hold"
            else "scheduler_first_cycle_steps"
        )
        training_overrides["warmup_steps"] = max(
            1,
            int(
                float(training_overrides["warmup_fraction"])
                * int(training_overrides[warmup_reference_key])
            ),
        )
    if "scheduler_first_cycle_steps" in training_overrides:
        cycle_steps = int(training_overrides["scheduler_first_cycle_steps"])
        if cycle_steps <= 0:
            raise ValueError("scheduler_first_cycle_steps must be positive")
        if "max_optimizer_steps" in training_overrides:
            max_steps = int(training_overrides["max_optimizer_steps"])
            if max_steps <= 0 or cycle_steps > max_steps:
                raise ValueError(
                    "scheduler_first_cycle_steps must be no larger than a positive "
                    f"max_optimizer_steps; got {cycle_steps} and {max_steps}"
                )
    if str(training_overrides.get("scheduler_mode", "cosine_restarts")) == "cosine_hold":
        anneal_steps = int(training_overrides.get("scheduler_anneal_steps", 0))
        max_steps = int(training_overrides.get("max_optimizer_steps", 0))
        if anneal_steps <= 0 or max_steps <= 0 or anneal_steps > max_steps:
            raise ValueError(
                "scheduler_anneal_steps must be in [1, max_optimizer_steps] "
                "for scheduler_mode=cosine_hold"
            )

    if args.adapter_only_only:
        source_dirs = []
    else:
        if not checkpoint_root.is_dir():
            raise FileNotFoundError(f"Checkpoint root does not exist: {checkpoint_root}")
        source_dirs = sorted(path for path in checkpoint_root.iterdir() if path.is_dir())
        if not source_dirs and not args.allow_empty:
            raise FileNotFoundError(f"No pretrained run directories found in {checkpoint_root}")

    runs = []
    if not args.no_adapter_only:
        for eventnumber in eventnumbers:
            row = build_adapter_only_run_row(
                base_dir=base_dir,
                eventnumber=eventnumber,
                train_batch_size=args.train_batch_size,
                max_samples=args.max_samples,
            )
            row["base_model_yaml"] = args.adapter_only_model_yaml
            row["base_model_config"] = args.adapter_only_model_config
            runs.append(row)

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
                        base_model_yaml=args.base_model_yaml,
                        base_model_config=args.pretrained_model_config,
                        pretrain_events=args.pretrain_events,
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
        "pretrained_model_config": args.pretrained_model_config,
        "adapter_only_model_yaml": args.adapter_only_model_yaml,
        "adapter_only_model_config": args.adapter_only_model_config,
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
    if source_optuna is not None:
        manifest["campaign_type"] = "optuna_best_recipe_backbone_comparison"
        manifest["source_optuna"] = source_optuna
    write_yaml(manifest_path, manifest)
    print(f"Wrote {len(runs)} runs to {manifest_path}")


if __name__ == "__main__":
    main()
