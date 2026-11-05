"""Validation aggregation, immutable snapshot, and trend-analysis integration."""

import os
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from train.downstream.track_finding_experiment import DownstreamTrainer
from train.downstream.loss import PointHungarianMatcher
from train.downstream.track_finding_metrics import EventMetricAccumulator
from train.downstream.campaign.plot_track_finding_validation import read_history, trend_summary, plot_history
from train.downstream.campaign.plot_track_finding_validation import main as plot_cli


class FixtureHead(torch.nn.Module):
    softmax_mask = False
    do_masked_attn = True
    num_heads = 4
    embed_method = "add"
    pe_method = "nerf"

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
    trainer.use_pretrained_backbone = False
    trainer.backbone_sha256 = None
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
    def test_plot_cli_uses_history_recorded_in_training_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = fixture_trainer(root)
            loss = trainer.validate_end_to_end_one_epoch()
            trainer._record_validation_result(loss, .5, "best.pth", epoch=0, step=100)
            history = Path(trainer.params.validation_history_dir)
            receipt = root / "artifacts.json"
            receipt.write_text(json.dumps({"validation_history_dir": str(history.relative_to(root))}))
            output = root / "receipt_plots"
            with patch("sys.argv", ["plot", "--artifact-summary", str(receipt),
                                    "--output-dir", str(output)]):
                plot_cli()
            for name in ("metrics_vs_step.png", "background_vs_signal_ari.pdf", "trend_summary.json"):
                self.assertGreater((output / name).stat().st_size, 0)

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
            self.assertAlmostEqual(trainer._selection_ari(2), 2/3)
            self.assertAlmostEqual(loss, trainer._last_validation_loss_event_mean)
            self.assertIn("val/option2/ari_with_background", trainer._validation_metric_callback())

    def test_selection_is_batch_invariant_and_restores_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = fixture_trainer(directory)
            points = torch.cat([b["points"] for b in trainer.val_data_loader])[:3]
            labels = torch.cat([b["target"] for b in trainer.val_data_loader])[:3]
            losses = []
            trainer.model.eval()
            trainer.down_model.train()
            for size in (1, 2, 3):
                trainer.val_data_loader = [dict(points=x, target=y)
                                           for x, y in zip(points.split(size), labels.split(size))]
                losses.append(trainer.validate_end_to_end_one_epoch())
                self.assertAlmostEqual(trainer._selection_ari(2), 2/3)
                self.assertTrue(trainer.down_model.training)
                self.assertFalse(trainer.model.training)
            np.testing.assert_allclose(losses, losses[0], rtol=1e-6)
            trainer.model.train()
            trainer.down_model.eval()
            trainer.val_data_loader = [dict(points=points, target=torch.full_like(labels, -1))]
            with self.assertRaisesRegex(RuntimeError, "no events eligible"):
                trainer.validate_end_to_end_one_epoch()
            self.assertTrue(trainer.model.training)
            self.assertFalse(trainer.down_model.training)

    def test_option_one_validation_excludes_no_object_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = fixture_trainer(directory)
            trainer.down_model.classes[0, 1] = torch.tensor([.95, .05])
            first = trainer.val_data_loader[0]
            trainer.val_data_loader = [{k: v[:1] for k, v in first.items()}]
            trainer.validate_end_to_end_one_epoch()
            metrics = trainer._last_validation_metrics[1]
            self.assertEqual(metrics["n_matched_tracks"], 1)
            self.assertAlmostEqual(metrics["signal_loss_to_background"], .5)

    def test_epoch_cap_validates_last_update_and_saves_only_a_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = fixture_trainer(directory)
            trainer.params.max_optimizer_steps = 10
            trainer.params.max_epochs = 1
            trainer.params.val_interval_steps = 4
            trainer.startEpoch = trainer.global_step = trainer.iters = 0
            trainer.last_validation_step = None
            trainer.patience = 3
            trainer.grad_clip_value = 1.
            trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
            trainer.train_data_loader = [None, None]
            trainer._compute_batch_loss = lambda *args, **kwargs: (trainer.down_model.weight.square(), {}, {})
            best = Path(directory, "selected.pth")
            best.write_bytes(b"stale")
            trainer.params.trained_checkpoint_path = None
            with self.assertRaisesRegex(RuntimeError, "no finite validation checkpoint"):
                trainer._require_selected_checkpoint()
            trainer._train_by_optimizer_step(False, trainer.params.training_log_path, best.name)
            trainer._require_selected_checkpoint()
            self.assertEqual(trainer.params.trained_checkpoint_path, str(best))
            record, = read_history(trainer.params.validation_history_dir)
            self.assertEqual(record['selection_aggregation'], 'mean_of_eligible_events')
            self.assertEqual(record['step'], 2)
            self.assertEqual(torch.load(best, weights_only=False)['best_step'], 2)

    def test_failed_checkpoint_write_preserves_destination_and_published_path(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = fixture_trainer(directory)
            trainer.params.trained_checkpoint_path = None
            best = Path(directory, 'best.pth')
            best.write_bytes(b'prior run')
            with patch('torch.save', side_effect=OSError('write failed')):
                with self.assertRaisesRegex(OSError, 'write failed'):
                    trainer._save_checkpoint(best.name, 0, True, .5)
            self.assertIsNone(trainer.params.trained_checkpoint_path)
            self.assertEqual(best.read_bytes(), b'prior run')
            self.assertEqual(list(best.parent.iterdir()), [best])

    def test_multirank_guard_precedes_device_setup_and_file_creation(self):
        from train.downstream.track_finding_experiment import train_experiment

        with patch.dict(os.environ, WORLD_SIZE='2'), patch('torch.distributed.init_process_group') as init:
            for launch in (lambda: train_experiment(None), lambda: DownstreamTrainer(None, None)):
                with self.assertRaisesRegex(RuntimeError, "Track finding supports one process"):
                    launch()
            init.assert_not_called()
        with patch.dict(os.environ, WORLD_SIZE='1'), patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.get_world_size', return_value=2):
            with self.assertRaisesRegex(RuntimeError, "Track finding supports one process"):
                train_experiment(None)

    def test_checkpoint_guards_mode_decoder_and_backbone_including_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = fixture_trainer(directory)
            trainer.use_pretrained_backbone = True
            trainer.params.pretrained_ckpt = '/old/mount/base.pth'
            trainer.backbone_sha256 = 'trained-digest'
            best = Path(directory, 'adapter.pth')
            trainer._save_checkpoint(best.name, 0, True, .5)
            trainer.load_checkpoint(best, inference=True, use_pretrained_backbone=True)
            with self.assertRaisesRegex(ValueError, 'input mode'):
                trainer.load_checkpoint(best, inference=True, use_pretrained_backbone=False)
            for key, changed in [('do_masked_attn', False), ('softmax_mask', True), ('num_heads', 8)]:
                with patch.object(trainer.down_model, key, changed):
                    with self.assertRaisesRegex(ValueError, 'decoder mismatch'):
                        trainer.load_checkpoint(best, inference=True, use_pretrained_backbone=True)
            trainer.backbone_sha256 = 'different-digest'
            with self.assertRaisesRegex(ValueError, 'different pretrained backbone weights'):
                trainer.load_checkpoint(best, inference=True, use_pretrained_backbone=True)
            trainer.backbone_sha256 = None
            with self.assertRaisesRegex(ValueError, 'loaded pretrained backbone'):
                trainer.load_checkpoint(best, inference=True, use_pretrained_backbone=True)
            old = torch.load(best, weights_only=False)
            del old['forward_config']
            old['params'] = {'params': {'pretrained_ckpt': None, 'softmax_mask': False}}
            torch.save(old, best)
            trainer.load_checkpoint(best, inference=True, use_pretrained_backbone=False)
            with self.assertRaisesRegex(ValueError, 'input mode'):
                trainer.load_checkpoint(best, inference=True, use_pretrained_backbone=True)
            old['params'] = {}
            torch.save(old, best)
            with self.assertRaisesRegex(ValueError, 'legacy'):
                trainer.load_checkpoint(best, inference=True, use_pretrained_backbone=False)

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
            training_modes = []
            def train_batch(*args, **kwargs):
                training_modes.append(trainer.down_model.training)
                return trainer.down_model.weight.square(), {}, {}
            trainer._compute_batch_loss = train_batch
            trainer._train_by_optimizer_step(
                pretrain=False, log_file_path=trainer.params.training_log_path,
                checkpoint_file_name="best.pth",
            )
            self.assertEqual(training_modes, [True, True])
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
