from types import SimpleNamespace

import unittest
import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from train.downstream.lr_schedulers import CosineAnnealingWarmupThenHold
from train.downstream.regression_optimization import (
    apply_regression_lr_ratios, feature_mixing_lr_ratio,
    regression_optimizer_parameters, validate_regression_optimizer_resume,
)


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weighted_avg_weights = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
        self.projection = torch.nn.Linear(3, 1, bias=False, dtype=torch.float64)


def check_mixture_multiplier(schedule):
    model = TinyHead()
    params = SimpleNamespace(adapter_feature_mixing_lr_ratio=100.0)
    optimizer = torch.optim.AdamW(regression_optimizer_parameters(model, params),
                                  lr=.001, weight_decay=0, eps=0)
    scheduler = (CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=6,
                    max_lr=.001, min_lr=.00001, warmup_steps=1)
                 if schedule == "restarts" else
                 CosineAnnealingWarmupThenHold(optimizer, anneal_steps=6,
                    max_lr=.001, min_lr=.00001, warmup_steps=1))
    for step in range(9):
        optimizer.zero_grad()
        (model.weighted_avg_weights.sum() + model.projection.weight.sum()).backward()
        old_mix = model.weighted_avg_weights.detach().clone()
        old_projection = model.projection.weight.detach().clone()
        base = optimizer.param_groups[0]["lr"]
        apply_regression_lr_ratios(optimizer)
        apply_regression_lr_ratios(optimizer)  # Multiple batches in one epoch.
        optimizer.step()
        torch.testing.assert_close(old_mix - model.weighted_avg_weights,
                                   torch.full_like(old_mix, 100 * base))
        torch.testing.assert_close(old_projection - model.projection.weight,
                                   torch.full_like(old_projection, base))
        scheduler.step()
        if step == 3:
            state = optimizer.state_dict()
            resumed = torch.optim.AdamW(regression_optimizer_parameters(model, params), lr=.001)
            resumed.load_state_dict(state)
            optimizer = resumed
            scheduler.optimizer = resumed
    validate_regression_optimizer_resume({'adapter_feature_mixing_lr_ratio': 100}, params)
    try:
        validate_regression_optimizer_resume({}, params)
    except ValueError as exc:
        assert "different adapter_feature_mixing_lr_ratio" in str(exc)
    else:
        raise AssertionError("Mismatched resume was accepted")


class RegressionOptimizationTest(unittest.TestCase):
    def test_restarts_and_hold_preserve_actual_update_ratios_after_resume(self):
        for schedule in ('restarts', 'hold'):
            with self.subTest(schedule=schedule):
                check_mixture_multiplier(schedule)

    def test_default_optimizer_layout_and_legacy_checkpoint_remain_compatible(self):
        model = TinyHead()
        params = SimpleNamespace()
        original = torch.optim.AdamW(model.parameters())
        controlled = torch.optim.AdamW(regression_optimizer_parameters(model, params))
        self.assertEqual(original.state_dict(), controlled.state_dict())
        apply_regression_lr_ratios(controlled)
        self.assertEqual(original.state_dict(), controlled.state_dict())
        validate_regression_optimizer_resume({}, params)

    def test_reject_invalid_mixture_rate(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite and positive"):
                feature_mixing_lr_ratio(SimpleNamespace(adapter_feature_mixing_lr_ratio=value))
