"""Validation aggregation, immutable snapshot, and trend-analysis integration."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from train.downstream.track_finding_experiment import DownstreamTrainer
from train.downstream.loss import PointHungarianMatcher
from train.downstream.track_finding_metrics import EventMetricAccumulator
from train.downstream.campaign.plot_track_finding_validation import read_history, trend_summary, plot_history


class FixtureHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.calls = 0
        self.classes = torch.tensor([[[.05, .95], [.05, .95]],
                                     [[.05, .95], [.05, .95]],
                                     [[.05, .95], [.95, .05]]])
        self.masks = torch.tensor([
            [[.95, .05], [.95, .05], [.05, .95], [.05, .95], [.05, .05], [.05, .05]],
            [[.95, .05], [.95, .05], [.05, .95], [.05, .95], [.05, .05], [.05, .05]],
            [[.95, .05], [.95, .05], [.95, .05], [.95, .05], [.05, .05], [.05, .05]],
        ])

    def forward(self, points, **kwargs):
        self.calls += 1
        event_ids = points[:, 0, 0].long()
        return {"class_probs": self.classes[event_ids], "mask_probs": self.masks[event_ids]}


def fixture_trainer(directory):
    trainer = DownstreamTrainer.__new__(DownstreamTrainer)
    trainer.params = SimpleNamespace(
        max_validation_events=3, max_val_batches=None, assignment_threshold=.2,
        validation_ari_mode="signal", track_target_mode="signal_only",
        checkpoint_dir=str(directory), training_log_path=str(Path(directory) / "training.log"),
        save_validation_checkpoints=True,
    )
    trainer.device = "cpu"
    trainer.use_lora = trainer.use_amp = trainer.log_to_screen = False
    trainer.model = torch.nn.Identity()
    trainer.down_model = FixtureHead()
    trainer.down_optimizer = torch.optim.SGD(trainer.down_model.parameters(), lr=.01)
    trainer.down_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.down_optimizer, lambda _: 1.)
    trainer.matcher = PointHungarianMatcher(cost_class=.5, cost_dice=1, cost_focal=30)
    trainer.loss_matched_ce_weight = .5
    trainer.loss_unmatched_ce_weight = .1
    trainer.loss_dice_weight = 1
    trainer.loss_focal_weight = 30
    trainer.best_loss, trainer.best_ARI, trainer.best_step = float("inf"), 0., None
    trainer.global_step = 100
    trainer.min_delta, trainer.early_stopping_min_steps, trainer.stagnation_counter = 1e-4, 0, 0
    trainer.down_results = {key: [] for key in (
        "val", "ARI", "ARI_2", "loss_matched_ce", "loss_unmatched_ce", "loss_dice", "loss_focal",
    )}
    labels = torch.tensor([[0, 0, 1, 1, -1, -100], [0, 0, -1, -1, -100, -100],
                           [0, 0, 1, 1, -1, -100], [0, 0, 1, 1, -1, -100]])
    points = torch.zeros(4, 6, 3)
    points[:, :, 0] = torch.arange(4).reshape(-1, 1)
    points[labels == -100] = -100
    trainer.val_data_loader = [{"points": points[start:start+2], "target": labels[start:start+2]}
                               for start in (0, 2)]
    return trainer


class ValidationHistoryTest(unittest.TestCase):
    def test_validation_aggregates_events_once_and_pools_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = fixture_trainer(directory)
            loss = trainer.validate_end_to_end_one_epoch()
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(trainer.down_model.calls, 2)
            summary = trainer._last_validation_metrics[2]
            self.assertEqual(summary["n_events"], 3)  # truncated final batch
            self.assertEqual(summary["n_pred_tracks"], 5)
            self.assertEqual(summary["n_matched_tracks"], 4)
            self.assertAlmostEqual(summary["ari_signal"], 2/3)
            self.assertAlmostEqual(summary["track_purity"], 5/6)
            self.assertAlmostEqual(summary["track_purity_global"], 4/5)
            self.assertAlmostEqual(summary["background_rejection"], 2/3)
            # Preserve the historical selection rule separately from event means.
            self.assertAlmostEqual(np.mean(trainer.down_results["ARI_2"]), .5)
            self.assertIn("val/option2/ari_with_background", trainer._validation_metric_callback())

    def test_every_validation_is_retained_and_snapshot_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = fixture_trainer(directory)
            loss = trainer.validate_end_to_end_one_epoch()
            trainer._record_validation_result(loss, .5, "best.pth", epoch=0, step=100)
            first_snapshot_weights = trainer.down_model.weight.detach().clone()
            trainer.down_model.weight.data.fill_(2.)
            trainer.global_step = 200
            # Worse validation still gets its own snapshot and history record.
            trainer._record_validation_result(loss+1, .4, "best.pth", epoch=0, step=200)
            history = Path(trainer.params.validation_history_dir)
            rows = read_history(history)
            self.assertEqual(len(rows), 2)
            self.assertEqual([r["selected_best"] for r in rows], [True, False])
            self.assertNotEqual(rows[0]["checkpoint"], rows[1]["checkpoint"])
            old = torch.load(rows[0]["checkpoint"], weights_only=False)
            new = torch.load(rows[1]["checkpoint"], weights_only=False)
            best = torch.load(Path(directory) / "best.pth", weights_only=False)
            torch.testing.assert_close(old["model_state_dict"]["weight"], first_snapshot_weights)
            self.assertEqual(new["model_state_dict"]["weight"].item(), 2.)
            self.assertNotIn("optimizer_state_dict", new)
            self.assertIn("optimizer_state_dict", best)
            self.assertEqual(old["validation_record"]["step"], 100)
            trainer.load_checkpoint(rows[0]["checkpoint"], inference=True)
            torch.testing.assert_close(trainer.down_model.weight, first_snapshot_weights)
            with self.assertRaisesRegex(ValueError, "inference/selection"):
                trainer.load_checkpoint(rows[0]["checkpoint"], inference=False)
            self.assertAlmostEqual(trainer.best_ARI, .5)
            self.assertEqual(len(read_history(history / "metrics.csv")), 2)
            # A rerun using the same log path creates a separate history.
            trainer._validation_history = None
            trainer._persist_validation(0, 100, loss, .5, True)
            self.assertNotEqual(trainer.params.validation_history_dir, str(history))
            self.assertTrue(Path(rows[0]["checkpoint"]).is_file())

    def test_snapshot_opt_out_keeps_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = fixture_trainer(directory)
            trainer.params.save_validation_checkpoints = False
            loss = trainer.validate_end_to_end_one_epoch()
            trainer._persist_validation(0, 100, loss, .5, False)
            history = Path(trainer.params.validation_history_dir)
            self.assertIsNone(read_history(history)[0]["checkpoint"])
            self.assertFalse((history / "checkpoints").exists())

    def test_early_stopping_validation_is_saved_before_return(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = fixture_trainer(directory)
            trainer.params.max_optimizer_steps = 3
            trainer.params.max_epochs = 1
            trainer.params.val_interval_steps = 1
            trainer.params.early_stopping_min_steps = 0
            trainer.startEpoch = trainer.global_step = trainer.iters = 0
            trainer.patience = 1
            trainer.grad_clip_value = 1.
            trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
            trainer.train_data_loader = [None, None, None]
            trainer._compute_batch_loss = lambda *args, **kwargs: (
                trainer.down_model.weight.square(), {}, {},
            )
            trainer._train_by_optimizer_step(
                pretrain=False, log_file_path=trainer.params.training_log_path,
                checkpoint_file_name="best.pth",
            )
            rows = read_history(trainer.params.validation_history_dir)
            self.assertEqual([row["step"] for row in rows], [1, 2])
            self.assertEqual([row["selected_best"] for row in rows], [True, False])
            self.assertTrue(all(Path(row["checkpoint"]).is_file() for row in rows))

    def test_tradeoff_analysis_and_plot_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for index, (ari, purity, rejection) in enumerate(((.8, .95, .8), (.9, .9, .7), (.95, .85, .6)), 1):
                row = {"validation_index": index, "step": index*100, "epoch": 0,
                       "assignment_threshold": .2, "selected_best": True,
                       "checkpoint": f"step{index}.pth", "option2_ari_signal": ari,
                       "option2_ari_with_background": ari-.05,
                       "option2_track_purity_global": purity, "option2_background_rejection": rejection,
                       "option2_fake_rate": 1-purity}
                rows.append(row)
            result = trend_summary(rows)
            self.assertEqual(len(result["ari_rising_degradation_intervals"]), 2)
            self.assertEqual(result["best_by_metric"]["track_purity_global"]["step"], 100)
            self.assertEqual(result["best_by_metric"]["ari_signal"]["step"], 300)
            plot_history(rows, directory)
            for name in ("metrics_vs_step", "background_vs_signal_ari"):
                for suffix in ("png", "pdf"):
                    self.assertGreater((Path(directory) / f"{name}.{suffix}").stat().st_size, 1000)
            path = Path(directory) / "metrics.jsonl"
            rows[1]["assignment_threshold"] = .5
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, "differing assignment_threshold"):
                read_history(path)


if __name__ == "__main__":
    unittest.main()
