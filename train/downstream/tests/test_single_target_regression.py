import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from ruamel.yaml import YAML

from train.downstream.loss import masked_regression_loss
from train.downstream.regression_utils import (
    load_regression_target_stats,
    regression_angular_indices,
    regression_output_dim,
    regression_target_columns,
    resolve_regression_loss,
    single_target_to_physical_numpy,
    target_to_cartesian_numpy,
    transform_regression_target_numpy,
    transform_regression_target_torch,
)
from train.downstream.eval.evaluate_track_regression import write_single_target_evaluation
from train.downstream.model import RegressionTargetNormalizer, MambaTrackRegressionHead
from train.downstream.track_regression_trainer import DownstreamTrainer
from train.downstream.scripts.compute_regression_target_stats import _finite_segment_target, compute_stats
from mmap_ninja import RaggedMmap


class SingleTargetRegressionTest(unittest.TestCase):
    def loss(self, pred, truth, task, valid=None):
        if valid is None:
            valid = torch.ones_like(truth, dtype=torch.bool)
        return masked_regression_loss(
            {"pred": pred}, {"target": truth, "target_valid": valid},
            option=resolve_regression_loss(task),
            angular_indices=regression_angular_indices(task), target_std=[1.0],
        )["loss"]

    def test_transforms_match_physics_and_preserve_batch_hit_axes(self):
        reg = np.array([[[3.0, 4.0, -12.0], [-3.0, -4.0, 12.0]]])
        expected = {
            "p": np.log(np.linalg.norm(reg, axis=-1)),
            "theta": np.arctan2(np.hypot(reg[..., 0], reg[..., 1]), reg[..., 2]),
            "phi": np.arctan2(reg[..., 1], reg[..., 0]),
        }
        for task in expected:
            with self.subTest(task=task):
                target = transform_regression_target_numpy(reg, task)
                self.assertEqual(target.shape, (1, 2, 1))
                np.testing.assert_allclose(target[..., 0], expected[task])
                np.testing.assert_allclose(
                    transform_regression_target_torch(torch.tensor(reg), task), target
                )
                self.assertEqual(regression_output_dim(task), 1)
                with self.assertRaises(ValueError):
                    target_to_cartesian_numpy(target, task)

    def test_undefined_targets_are_masked(self):
        reg = np.array([[0., 0., 0.], [0., 0., 2.], [np.nan, 1., 2.]])
        for task, invalid in (("p", [True, False, True]),
                              ("theta", [True, False, True]),
                              ("phi", [True, True, True])):
            for target in (transform_regression_target_numpy(reg, task),
                           transform_regression_target_torch(torch.tensor(reg), task).numpy()):
                self.assertEqual(np.isnan(target[:, 0]).tolist(), invalid)

    def test_log_p_huber_is_unit_invariant_and_has_correct_gradient(self):
        truth_p = torch.tensor([[10.], [100.]])
        residual = torch.tensor([[0.4], [-2.0]])
        for unit_scale in (1., 1000.):
            truth = (truth_p * unit_scale).log()
            pred = (truth + residual).detach().requires_grad_()
            loss = self.loss(pred, truth, "p")
            self.assertAlmostEqual(loss.item(), (0.08 + 1.5) / 2, places=6)
            loss.backward()
            torch.testing.assert_close(pred.grad, torch.tensor([[0.2], [-0.5]]))
            decoded = single_target_to_physical_numpy(pred.detach().numpy(), "p")
            np.testing.assert_allclose(decoded, np.exp(pred.detach().numpy()), rtol=1e-6)
            self.assertTrue((decoded > 0).all())

    def test_phi_huber_wraps_and_is_periodic_with_finite_gradients(self):
        for turns in (0, -2, 3):
            pred = torch.tensor([[math.pi - .01 + turns * 2 * math.pi]],
                                dtype=torch.float64, requires_grad=True)
            truth = torch.tensor([[-math.pi + .01]], dtype=torch.float64)
            loss = self.loss(pred, truth, "phi")
            self.assertAlmostEqual(loss.item(), .5 * .02**2, places=10)
            loss.backward()
            self.assertAlmostEqual(pred.grad.item(), -.02, places=10)

    def test_theta_uses_unwrapped_radian_huber(self):
        pred = torch.tensor([[2.5]], requires_grad=True)
        loss = self.loss(pred, torch.tensor([[.5]]), "theta")
        self.assertEqual(loss.item(), 1.5)
        loss.backward()
        self.assertEqual(pred.grad.item(), 1.)

    def test_masked_nan_angles_do_not_poison_loss_or_gradients(self):
        pred = torch.tensor([[.2], [float("nan")]], requires_grad=True)
        truth = torch.tensor([[0.], [float("nan")]])
        loss = self.loss(pred, truth, "phi")
        self.assertAlmostEqual(loss.item(), .02, places=6)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([[.2], [0.]]))
        pred = torch.tensor([[float("nan")]], requires_grad=True)
        loss = self.loss(pred, pred.detach(), "phi")
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertEqual(pred.grad.item(), 0.)

    def test_single_target_stats_keep_physical_loss_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.json"
            for task in ("p", "theta", "phi"):
                path.write_text(json.dumps({"columns": regression_target_columns(task),
                                            "mean": [10.], "std": [.01]}))
                stats = load_regression_target_stats(path, task)
                self.assertEqual(stats["mean"], [0.])
                self.assertEqual(stats["std"], [1.])
                self.assertEqual(stats["phi_pairs"], [])
                self.assertEqual(stats["angular_indices"], [0] if task == "phi" else [])
                with self.assertRaises(ValueError):
                    resolve_regression_loss(task, "mae")
            path.write_text(json.dumps({"columns": ["mc_entrance_p"], "mean": [1.], "std": [1.]}))
            with self.assertRaises(ValueError):
                load_regression_target_stats(path, "p")
        self.assertEqual(resolve_regression_loss("mom"), "mse")
        self.assertEqual(resolve_regression_loss("p_phi_theta", "mae"), "mae")

    def test_trainer_targets_use_segment_mask_and_circular_phi_mean(self):
        trainer = DownstreamTrainer.__new__(DownstreamTrainer)
        trainer.down_model = SimpleNamespace(target_normalizer=RegressionTargetNormalizer(1))
        phi = torch.tensor([math.pi - .01, -math.pi + .01, 0.])
        reg = torch.stack((phi.cos(), phi.sin(), torch.zeros_like(phi)), dim=-1)[None]
        mask = torch.ones((1, 3), dtype=torch.bool)
        segment = torch.tensor([[True, True, False]])
        for task in ("p", "theta", "phi"):
            trainer.params = SimpleNamespace(task=task)
            trainer.regression_target_stats = {"task": task}
            target = trainer.build_regression_targets(reg, mask, segment)
            self.assertEqual(target["target"].shape, (1, 1))
            self.assertTrue(target["target_valid"].all())
            expected = _finite_segment_target(reg[0, :2].numpy(), task)
            np.testing.assert_allclose(target["target"].numpy()[0], expected, atol=1e-7)
            if task == "phi":
                self.assertAlmostEqual(abs(target["target"].item()), math.pi, places=6)
            empty = trainer.build_regression_targets(reg, mask, torch.zeros_like(segment))
            self.assertFalse(empty["target_valid"].any())

    def test_stats_generator_accepts_each_single_target_on_ragged_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = np.tile([3., 4., 12., 0., 0., 0., 13.], (12, 1))
            RaggedMmap.from_lists(root / "features_pretrain", [np.ones((12, 3))])
            RaggedMmap.from_lists(root / "reg_target_pretrain", [raw])
            RaggedMmap.from_lists(root / "seg_target_pretrain", [np.zeros(12, dtype=np.int64)])
            for task in ("p", "theta", "phi"):
                stats = compute_stats(root, "pretrain", 1, 100, None, 10, task=task)
                self.assertEqual(stats["columns"], list(regression_target_columns(task)))
                np.testing.assert_allclose(stats["mean"], transform_regression_target_numpy(raw, task)[0])

    def test_presets_construct_single_output_heads(self):
        root = Path(__file__).resolve().parents[3]
        for family in ("adapteronly", "pretrained"):
            configs = YAML(typ="safe").load(root / "scripts" / "configs" /
                                             f"mamba_clas12_track_regression_{family}.yaml")
            for task in ("p", "theta", "phi"):
                config = configs[f"clas12_track_regression_{family}_{task}_only"]
                self.assertEqual(config["task"], task)
                self.assertEqual(resolve_regression_loss(task, config["regression_loss"]), "huber")
                head = MambaTrackRegressionHead(
                    input_dim=128, num_layers=0, num_embedder_layers=0,
                    num_output_dim=regression_output_dim(task),
                    embed_method="pos_only", pooling="attention", target_mean=[0.], target_std=[1.],
                )
                self.assertEqual(head.out_mlp(torch.zeros((2, 256))).shape, (2, 1))

    def test_evaluation_exports_only_learned_quantity(self):
        with tempfile.TemporaryDirectory() as tmp:
            for task in ("p", "theta", "phi"):
                output = Path(tmp) / task
                output.mkdir()
                config = {"output_dir": output, "target_momentum_scale_to_gev": .001,
                          "checkpoint": output / "model.pth", "model_config": task}
                truth = np.array([[math.pi - .01]]) if task == "phi" else np.array([[1.]])
                pred = np.array([[-math.pi + .01]]) if task == "phi" else truth + .2
                with patch("train.downstream.eval.evaluate_track_regression.make_training_curve_plot"), \
                     patch("train.downstream.eval.evaluate_track_regression.make_ml_error_bar_plot"):
                    summary = write_single_target_evaluation(
                        config, task, [{"true_native": truth.item(), "adapter_native": pred.item()}], truth, pred
                    )
                expected_loss = .0002 if task == "phi" else .02
                self.assertAlmostEqual(summary["huber_loss"], expected_loss, places=8)
                self.assertEqual(summary["n_valid_targets"], 1)
                self.assertFalse(summary["swingback_enabled"])
                self.assertEqual(summary["comparison_truth"], "mctrue_inner_hit")
                rows = [json.loads(line) for line in (output / "campaign_headline_metrics.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(row["record_type"] == "ml_error" for row in rows))
                self.assertTrue((output / "predictions.csv.gz").exists())
                if task == "p":
                    metrics = summary["methods"]["adapter"]["kinematic"]["p_gev"]
                    self.assertAlmostEqual(metrics["mae"], (math.exp(1.2) - math.exp(1.)) * .001)
                if task == "phi":
                    self.assertAlmostEqual(summary["methods"]["adapter"]["native_target"]["phi"]["mae"], .02)


if __name__ == "__main__":
    unittest.main()
