import sys
import unittest
from pathlib import Path

import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from train.downstream.lr_schedulers import CosineAnnealingWarmupThenHold


class CosineAnnealingWarmupThenHoldTest(unittest.TestCase):
    def test_reaches_and_holds_the_minimum_without_a_restart(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = CosineAnnealingWarmupThenHold(
            optimizer,
            anneal_steps=10,
            max_lr=1.0,
            min_lr=0.1,
            warmup_steps=2,
        )
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)

        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.55)
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0)

        for _ in range(8):
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)

        for _ in range(10):
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)

    def test_rejects_a_warmup_that_consumes_the_annealing_phase(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        with self.assertRaisesRegex(ValueError, "warmup_steps"):
            CosineAnnealingWarmupThenHold(
                optimizer,
                anneal_steps=10,
                max_lr=1.0,
                min_lr=0.1,
                warmup_steps=10,
            )

    def test_matches_the_existing_scheduler_through_the_first_cycle(self):
        hold_parameter = torch.nn.Parameter(torch.tensor(1.0))
        restart_parameter = torch.nn.Parameter(torch.tensor(1.0))
        hold_optimizer = torch.optim.SGD([hold_parameter], lr=1.0)
        restart_optimizer = torch.optim.SGD([restart_parameter], lr=1.0)
        hold = CosineAnnealingWarmupThenHold(
            hold_optimizer,
            anneal_steps=10,
            max_lr=1.0,
            min_lr=0.1,
            warmup_steps=2,
        )
        restart = CosineAnnealingWarmupRestarts(
            restart_optimizer,
            first_cycle_steps=10,
            max_lr=1.0,
            min_lr=0.1,
            warmup_steps=2,
        )
        self.assertAlmostEqual(
            hold_optimizer.param_groups[0]["lr"],
            restart_optimizer.param_groups[0]["lr"],
        )
        for _ in range(10):
            hold.step()
            restart.step()
            self.assertAlmostEqual(
                hold_optimizer.param_groups[0]["lr"],
                restart_optimizer.param_groups[0]["lr"],
            )

        hold.step()
        restart.step()
        self.assertAlmostEqual(hold_optimizer.param_groups[0]["lr"], 0.1)
        self.assertGreater(restart_optimizer.param_groups[0]["lr"], 0.1)


if __name__ == "__main__":
    unittest.main()
