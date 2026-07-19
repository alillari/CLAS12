"""Optuna search spaces for downstream track regression."""

from __future__ import annotations

from typing import Any


def suggest_adapteronly_optimizer_params(trial: Any) -> dict[str, float | int]:
    """First-stage AdapterOnly optimizer/schedule search space."""

    max_lr = trial.suggest_float("max_lr", 6e-4, 2e-3, log=True)
    min_lr_ratio = trial.suggest_float("min_lr_ratio", .001, 5e-2, log=True)
    warmup_fraction = trial.suggest_float("warmup_fraction", 0.2, 0.26)

    return {
        "max_lr": max_lr,
        "min_lr_ratio": min_lr_ratio,
        "min_lr": max_lr * min_lr_ratio,
        "warmup_fraction": warmup_fraction,
        "adapter_weight_decay": trial.suggest_float(
            "adapter_weight_decay",
            2e-3,
            5e-1,
            log=True,
        ),
        "grad_clip_value": trial.suggest_float(
            "grad_clip_value",
            1.0,
            5.0,
            log=True,
        ),
        "dropout": trial.suggest_float("dropout", 0.06, 0.15),
    }
