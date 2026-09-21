"""Small learning-rate schedulers used by downstream training."""

from __future__ import annotations

import math
from typing import Any


class CosineAnnealingWarmupThenHold:
    """Warm up, cosine-anneal once, then hold the minimum learning rate.

    ``CosineAnnealingWarmupRestarts`` repeats its cycle after
    ``first_cycle_steps``.  For short annealing phases this makes the learning
    rate jump back up even when the desired policy is low-LR refinement.  This
    scheduler deliberately has no restart: after ``anneal_steps`` every
    parameter group remains at ``min_lr``.

    The first optimizer update uses ``min_lr``.  Calling :meth:`step` after an
    update advances the schedule, matching the step-order convention used by
    the existing downstream trainer.
    """

    def __init__(
        self,
        optimizer: Any,
        *,
        anneal_steps: int,
        max_lr: float,
        min_lr: float,
        warmup_steps: int = 0,
    ) -> None:
        if anneal_steps <= 0:
            raise ValueError("anneal_steps must be positive")
        if warmup_steps < 0 or warmup_steps >= anneal_steps:
            raise ValueError("warmup_steps must be in [0, anneal_steps)")
        if max_lr < min_lr:
            raise ValueError("max_lr must be greater than or equal to min_lr")

        self.optimizer = optimizer
        self.anneal_steps = int(anneal_steps)
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(warmup_steps)
        self.step_count = 0
        self._last_lr = [self.min_lr for _ in self.optimizer.param_groups]
        self._set_lr(self.min_lr)

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)
        self._last_lr = [float(lr) for _ in self.optimizer.param_groups]

    def _lr_at_step(self, step: int) -> float:
        if self.warmup_steps and step <= self.warmup_steps:
            return self.min_lr + (self.max_lr - self.min_lr) * step / self.warmup_steps
        if step <= self.anneal_steps:
            progress = (step - self.warmup_steps) / (self.anneal_steps - self.warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine
        return self.min_lr

    def step(self) -> None:
        self.step_count += 1
        self._set_lr(self._lr_at_step(self.step_count))

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def state_dict(self) -> dict[str, int | float]:
        return {
            "anneal_steps": self.anneal_steps,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict: dict[str, int | float]) -> None:
        expected = {
            "anneal_steps": self.anneal_steps,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
        }
        for key, value in expected.items():
            if state_dict.get(key) != value:
                raise ValueError(
                    f"Cannot load scheduler state with different {key}: "
                    f"expected {value!r}, got {state_dict.get(key)!r}"
                )
        self.step_count = int(state_dict["step_count"])
        self._set_lr(self._lr_at_step(self.step_count))
