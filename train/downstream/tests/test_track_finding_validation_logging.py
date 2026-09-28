"""Regression coverage for distinct assignment-policy ARIs in step logs."""

import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

# The entrypoint installs the downstream imports used by the trainer.
from train.downstream.track_finding_experiment import DownstreamTrainer


class TrackFindingValidationLoggingTest(unittest.TestCase):
    def test_step_log_preserves_both_policies_and_selects_option2(self):
        for mode in ("signal", "inclusive"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                trainer = DownstreamTrainer.__new__(DownstreamTrainer)
                trainer.params = SimpleNamespace(
                    max_optimizer_steps=1, val_interval_steps=1,
                    early_stopping_min_steps=0, max_epochs=1, validation_ari_mode=mode,
                )
                trainer.startEpoch = trainer.global_step = trainer.iters = 0
                trainer.use_lora = False
                trainer.model = torch.nn.Identity()
                trainer.down_model = torch.nn.Linear(1, 1)
                trainer.down_optimizer = torch.optim.SGD(trainer.down_model.parameters(), lr=.01)
                trainer.down_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.down_optimizer, lambda _: 1.)
                trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
                trainer.grad_clip_value = 1.
                trainer.train_data_loader = [torch.ones(1, 1)]
                trainer.down_results = {}
                trainer._compute_batch_loss = lambda batch, **kwargs: (
                    trainer.down_model(batch).square().mean(), {}, {},
                )

                def validate(**kwargs):
                    trainer.down_results["ARI"].extend([.1, .3])
                    trainer.down_results["ARI_2"].extend([.6, .8])
                    return .5

                trainer.validate_end_to_end_one_epoch = validate
                trainer.best_step = None
                trainer.best_ARI = 0.
                trainer.best_loss = float("inf")
                trainer.min_delta = 1e-4
                trainer.patience = 10
                trainer.stagnation_counter = 0
                trainer._save_checkpoint = Mock()
                callback = Mock()
                trial = Mock()
                trial.should_prune.return_value = False
                path = Path(directory) / "training.log"
                path.write_text(
                    "Step\tEpoch\tTrain_Loss\tVal_Loss\t"
                    f"ARI_{mode}_option1\tARI_{mode}_option2\tLR\tTime\n"
                )
                trainer._train_by_optimizer_step(
                    pretrain=False, log_file_path=path, checkpoint_file_name="unused.pth",
                    optuna_trial=trial, metrics_callback=callback,
                )
                with path.open() as stream:
                    rows = list(csv.DictReader(stream, delimiter="\t"))
                self.assertEqual(len(rows), 1)
                self.assertAlmostEqual(float(rows[0][f"ARI_{mode}_option1"]), .2)
                self.assertAlmostEqual(float(rows[0][f"ARI_{mode}_option2"]), .7)
                metrics = callback.call_args.args[0]
                self.assertAlmostEqual(metrics[f"val/ari_{mode}_option1"], .2)
                self.assertAlmostEqual(metrics[f"val/ari_{mode}_option2"], .7)
                self.assertAlmostEqual(metrics["val/ari"], .7)
                self.assertAlmostEqual(trainer.best_ARI, .7)
                self.assertAlmostEqual(trial.report.call_args.args[0], .7)
                trainer._save_checkpoint.assert_called_once()


if __name__ == "__main__":
    unittest.main()
