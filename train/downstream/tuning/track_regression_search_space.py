"""Optuna search spaces for downstream track regression."""

from __future__ import annotations

from typing import Any


def suggest_adapteronly_optimizer_params(
    trial: Any,
    *,
    scheduler_mode: str = "cosine_restarts",
    anneal_steps_min: int | None = None,
    anneal_steps_max: int | None = None,
    anneal_steps_step: int | None = None,
) -> dict[str, float | int]:
    """AdapterOnly optimizer search space, optionally including a hold schedule."""

    max_lr = trial.suggest_float("max_lr", 4e-4, 2.5e-3, log=True)
    min_lr_ratio = trial.suggest_float("min_lr_ratio", 5e-4, 8e-2, log=True)
    # Existing cycle studies concentrated near the old 0.35 ceiling.  Permit a
    # slightly longer warmup only for the new no-restart schedule.
    warmup_upper = 0.40 if scheduler_mode == "cosine_hold" else 0.35
    warmup_fraction = trial.suggest_float("warmup_fraction", 0.10, warmup_upper)

    params: dict[str, float | int] = {
        "max_lr": max_lr,
        "min_lr_ratio": min_lr_ratio,
        "min_lr": max_lr * min_lr_ratio,
        "warmup_fraction": warmup_fraction,
        "adapter_weight_decay": trial.suggest_float(
            "adapter_weight_decay",
            1e-3,
            5e-1,
            log=True,
        ),
        "grad_clip_value": trial.suggest_float(
            "grad_clip_value",
            0.5,
            8.0,
            log=True,
        ),
        "dropout": trial.suggest_float("dropout", 0.02, 0.20),
    }
    if scheduler_mode == "cosine_hold":
        if (
            anneal_steps_min is None
            or anneal_steps_max is None
            or anneal_steps_step is None
        ):
            raise ValueError("cosine_hold requires an anneal-steps search range")
        params["scheduler_anneal_steps"] = trial.suggest_int(
            "scheduler_anneal_steps",
            int(anneal_steps_min),
            int(anneal_steps_max),
            step=int(anneal_steps_step),
        )
    elif scheduler_mode != "cosine_restarts":
        raise ValueError(f"Unsupported scheduler_mode: {scheduler_mode!r}")
    return params
