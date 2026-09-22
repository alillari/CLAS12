import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

DOWNSTREAM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOWNSTREAM_DIR))

from loss import masked_regression_loss
from scripts.compute_regression_loss_reference_stats import central_width_68
from regression_utils import (
    load_regression_loss_reference_stats,
    REGRESSION_TARGET_COLUMNS,
    load_regression_target_stats,
    project_phi_pair_numpy,
    regression_phi_pairs,
    target_to_cartesian_numpy,
    transform_regression_target_numpy,
    transform_regression_target_torch,
)


class RegressionTargetTransformTest(unittest.TestCase):
    def test_pt_phi_eta_round_trip_numpy(self):
        reg = np.array([
            [300.0, 400.0, 1200.0, 0.0, 0.0, 0.0, 1300.0],
            [-500.0, 250.0, -700.0, 0.0, 0.0, 0.0, 900.0],
        ])
        target = transform_regression_target_numpy(reg, "pt_phi_eta")
        np.testing.assert_allclose(target[:, 0], np.hypot(reg[:, 0], reg[:, 1]))
        np.testing.assert_allclose(target[:, 1], np.cos(np.arctan2(reg[:, 1], reg[:, 0])))
        np.testing.assert_allclose(target[:, 2], np.sin(np.arctan2(reg[:, 1], reg[:, 0])))
        self.assertEqual(target.shape[-1], 4)
        cart = target_to_cartesian_numpy(target, "pt_phi_eta")
        np.testing.assert_allclose(cart, reg[:, :3], rtol=1.0e-6, atol=1.0e-6)

    def test_pt_phi_eta_torch_matches_numpy(self):
        reg = np.array([[300.0, 400.0, 1200.0, 0.0, 0.0, 0.0, 1300.0]])
        expected = transform_regression_target_numpy(reg, "pt_phi_eta")
        actual = transform_regression_target_torch(
            torch.as_tensor(reg, dtype=torch.float32), "pt_phi_eta"
        ).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)

    def test_pt_phi_eta_cartesian_uses_projected_direction(self):
        target = np.array([[500.0, 30.0, 40.0, 1.2]])
        cart = target_to_cartesian_numpy(target, "pt_phi_eta")
        np.testing.assert_allclose(cart[0, 0], 300.0, rtol=1.0e-6)
        np.testing.assert_allclose(cart[0, 1], 400.0, rtol=1.0e-6)

    def test_p_phi_theta_round_trip_numpy(self):
        reg = np.array([
            [300.0, 400.0, 1200.0, 0.0, 0.0, 0.0, 1300.0],
            [-500.0, 250.0, -700.0, 0.0, 0.0, 0.0, 900.0],
        ])
        target = transform_regression_target_numpy(reg, "p_phi_theta")
        expected_p = np.linalg.norm(reg[:, :3], axis=1)
        expected_theta = np.arctan2(np.hypot(reg[:, 0], reg[:, 1]), reg[:, 2])
        np.testing.assert_allclose(target[:, 0], expected_p)
        np.testing.assert_allclose(target[:, 1], np.cos(np.arctan2(reg[:, 1], reg[:, 0])))
        np.testing.assert_allclose(target[:, 2], np.sin(np.arctan2(reg[:, 1], reg[:, 0])))
        np.testing.assert_allclose(target[:, 3], expected_theta)
        self.assertEqual(target.shape[-1], 4)
        cart = target_to_cartesian_numpy(target, "p_phi_theta")
        np.testing.assert_allclose(cart, reg[:, :3], rtol=1.0e-6, atol=1.0e-6)

    def test_p_phi_theta_torch_matches_numpy(self):
        reg = np.array([[300.0, 400.0, 1200.0, 0.0, 0.0, 0.0, 1300.0]])
        expected = transform_regression_target_numpy(reg, "p_phi_theta")
        actual = transform_regression_target_torch(
            torch.as_tensor(reg, dtype=torch.float32), "p_phi_theta"
        ).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)

    def test_wrapped_phi_loss_crosses_boundary(self):
        pred = torch.tensor([[0.0, math.pi - 0.01, 0.0]])
        truth = torch.tensor([[0.0, -math.pi + 0.01, 0.0]])
        loss = masked_regression_loss(
            {"pred": pred},
            {"target": truth, "target_valid": torch.ones_like(truth, dtype=torch.bool)},
            option="mae",
            angular_indices=(1,),
            target_std=[1.0, 1.0, 1.0],
        )["loss"]
        self.assertLess(float(loss), 0.01)

    def test_pt_phi_eta_rejects_raw_cartesian_stats(self):
        payload = {
            "version": 1,
            "columns": list(REGRESSION_TARGET_COLUMNS),
            "mean": [0.0] * len(REGRESSION_TARGET_COLUMNS),
            "std": [1.0] * len(REGRESSION_TARGET_COLUMNS),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_regression_target_stats(path, "pt_phi_eta")

    def test_p_phi_theta_rejects_raw_cartesian_stats(self):
        payload = {
            "version": 1,
            "columns": list(REGRESSION_TARGET_COLUMNS),
            "mean": [0.0] * len(REGRESSION_TARGET_COLUMNS),
            "std": [1.0] * len(REGRESSION_TARGET_COLUMNS),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_regression_target_stats(path, "p_phi_theta")

    def test_pt_phi_eta_rejects_legacy_scalar_phi_stats(self):
        payload = {
            "version": 2,
            "columns": ["mc_entrance_pt", "mc_entrance_phi", "mc_entrance_eta"],
            "mean": [1.0, 0.0, 0.0],
            "std": [0.5, 1.0, 1.0],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_regression_target_stats(path, "pt_phi_eta")

    def test_cosphi_sinphi_stats_are_not_standardized(self):
        payload = {
            "version": 2,
            "columns": [
                "mc_entrance_pt",
                "mc_entrance_cosphi",
                "mc_entrance_sinphi",
                "mc_entrance_eta",
            ],
            "mean": [2.0, 0.1, -0.2, 0.3],
            "std": [0.4, 0.7, 0.8, 1.2],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"
            path.write_text(json.dumps(payload))
            stats = load_regression_target_stats(path, "pt_phi_eta")
        self.assertEqual(
            stats["columns"],
            [
                "mc_entrance_pt",
                "mc_entrance_cosphi",
                "mc_entrance_sinphi",
                "mc_entrance_eta",
            ],
        )
        self.assertEqual(stats["mean"], [2.0, 0.0, 0.0, 0.3])
        self.assertEqual(stats["std"], [0.4, 1.0, 1.0, 1.2])
        self.assertEqual(stats["angular_indices"], [])
        self.assertEqual(stats["phi_pairs"], [[1, 2]])

    def test_project_phi_pair_numpy_normalizes_direction(self):
        cosphi, sinphi = project_phi_pair_numpy(np.array([3.0]), np.array([4.0]))
        np.testing.assert_allclose(cosphi, [0.6], rtol=1.0e-6)
        np.testing.assert_allclose(sinphi, [0.8], rtol=1.0e-6)

    def test_phi_pair_loss_ignores_prediction_radius(self):
        pred = torch.tensor([[0.0, 30.0, 40.0, 0.0]])
        truth = torch.tensor([[0.0, 0.6, 0.8, 0.0]])
        loss = masked_regression_loss(
            {"pred": pred},
            {"target": truth, "target_valid": torch.ones_like(truth, dtype=torch.bool)},
            option="mse",
            phi_pairs=regression_phi_pairs("pt_phi_eta"),
        )["loss"]
        self.assertLess(float(loss), 1.0e-6)

    def test_phi_pair_loss_opposite_direction(self):
        pred = torch.tensor([[0.0, -0.6, -0.8, 0.0]])
        truth = torch.tensor([[0.0, 0.6, 0.8, 0.0]])
        loss = masked_regression_loss(
            {"pred": pred},
            {"target": truth, "target_valid": torch.ones_like(truth, dtype=torch.bool)},
            option="mse",
            phi_pairs=regression_phi_pairs("pt_phi_eta"),
        )["loss"]
        self.assertAlmostEqual(float(loss), 2.0 / 3.0, places=6)

    def test_phi_pair_loss_crosses_boundary(self):
        pred_phi = math.pi - 0.01
        truth_phi = -math.pi + 0.01
        pred = torch.tensor([[0.0, math.cos(pred_phi), math.sin(pred_phi), 0.0]])
        truth = torch.tensor([[0.0, math.cos(truth_phi), math.sin(truth_phi), 0.0]])
        loss = masked_regression_loss(
            {"pred": pred},
            {"target": truth, "target_valid": torch.ones_like(truth, dtype=torch.bool)},
            option="mse",
            phi_pairs=regression_phi_pairs("pt_phi_eta"),
        )["loss"]
        self.assertLess(float(loss), 1.0e-4)

    def test_physical_resolution_l1_uses_physical_scales_and_wrapped_phi(self):
        # p and theta are normalized targets.  Their physical residuals are
        # recovered with target_std before division by the reference scales.
        pred = torch.tensor([[1.2, math.cos(math.pi - 0.01), math.sin(math.pi - 0.01), 2.3]])
        truth = torch.tensor([[1.0, math.cos(-math.pi + 0.01), math.sin(-math.pi + 0.01), 2.0]])
        scales = {
            "p_scale_gev": 0.1,
            "theta_scale_rad": 0.2,
            "phi_scale_rad": 0.02,
            "target_momentum_scale_to_gev": 1.0e-3,
        }
        loss = masked_regression_loss(
            {"pred": pred},
            {"target": truth, "target_valid": torch.ones_like(truth, dtype=torch.bool)},
            option="physical_resolution_l1",
            target_std=[100.0, 1.0, 1.0, 0.5],
            phi_pairs=regression_phi_pairs("p_phi_theta"),
            physical_scales=scales,
        )["loss"]
        # p: 0.2 normalized * 100 MeV * 1e-3 / 0.1 = 0.2
        # theta: 0.3 normalized * 0.5 / 0.2 = 0.75
        # phi: 0.02 rad / 0.02 = 1.0
        self.assertAlmostEqual(float(loss), (0.2 + 0.75 + 1.0) / 3.0, places=5)

    def test_physical_resolution_relative_huber_uses_true_p_and_wrapped_phi(self):
        # The normalized true p is 2, but the physical true p is
        # (2 * 100 + 800) MeV = 1 GeV.  A 50 MeV prediction error is 5%.
        pred = torch.tensor([[2.5, math.cos(math.pi - 0.01), math.sin(math.pi - 0.01), 2.3]])
        truth = torch.tensor([[2.0, math.cos(-math.pi + 0.01), math.sin(-math.pi + 0.01), 2.0]])
        scales = {
            "p_scale_relative": 0.04,
            "theta_scale_rad": 0.2,
            "phi_scale_rad": 0.02,
            "target_momentum_scale_to_gev": 1.0e-3,
        }
        loss = masked_regression_loss(
            {"pred": pred},
            {"target": truth, "target_valid": torch.ones_like(truth, dtype=torch.bool)},
            option="physical_resolution_relative_huber",
            target_mean=[800.0, 0.0, 0.0, 0.0],
            target_std=[100.0, 1.0, 1.0, 0.5],
            phi_pairs=regression_phi_pairs("p_phi_theta"),
            physical_scales=scales,
        )["loss"]
        # Huber with delta=1 on scaled residuals: p=1.25, theta=.75, phi=1.
        self.assertAlmostEqual(float(loss), (0.75 + 0.28125 + 0.5) / 3.0, places=5)

    def test_physical_resolution_relative_huber_requires_target_mean(self):
        target = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "requires target_mean"):
            masked_regression_loss(
                {"pred": target},
                {"target": target, "target_valid": torch.ones_like(target, dtype=torch.bool)},
                option="physical_resolution_relative_huber",
                target_std=[1.0] * 4,
                phi_pairs=regression_phi_pairs("p_phi_theta"),
                physical_scales={"p_scale_relative": 0.04, "theta_scale_rad": 0.01,
                                 "phi_scale_rad": 0.02, "target_momentum_scale_to_gev": 0.001},
            )

    def test_physical_resolution_relative_huber_divides_by_each_true_momentum(self):
        truth = torch.tensor([[1.0, 1.0, 0.0, 1.0], [2.0, 1.0, 0.0, 1.0]])
        pred = truth.clone()
        pred[:, 0] += 0.1  # Same absolute 0.1 GeV error for both tracks.
        valid = torch.zeros_like(truth, dtype=torch.bool)
        valid[:, 0] = True
        loss = masked_regression_loss(
            {"pred": pred}, {"target": truth, "target_valid": valid},
            option="physical_resolution_relative_huber",
            target_mean=[0.0] * 4, target_std=[1000.0, 1.0, 1.0, 1.0],
            phi_pairs=regression_phi_pairs("p_phi_theta"),
            physical_scales={"p_scale_relative": 0.1, "theta_scale_rad": 0.01,
                             "phi_scale_rad": 0.02, "target_momentum_scale_to_gev": 0.001},
        )["loss"]
        # Relative residuals are 10% and 5%, hence scaled residuals 1 and .5.
        self.assertAlmostEqual(float(loss), (0.5 + 0.125) / 2.0, places=5)

    def test_reference_width_is_offset_invariant(self):
        residuals = np.array([-0.08, -0.05, -0.01, 0.02, 0.09])
        width, q16, q84 = central_width_68(residuals)
        shifted_width, shifted_q16, shifted_q84 = central_width_68(residuals + 0.25)
        self.assertAlmostEqual(width, shifted_width)
        self.assertAlmostEqual(shifted_q16 - q16, 0.25)
        self.assertAlmostEqual(shifted_q84 - q84, 0.25)

    def test_load_physical_resolution_reference_stats(self):
        payload = {
            "schema": "clas12_regression_loss_reference_v1",
            "task": "p_phi_theta",
            "target_momentum_scale_to_gev": 1.0e-3,
            "residuals": {
                "p_absolute_gev": {"unit": "GeV", "central_width_68": 0.04},
                "p_relative": {"unit": "fraction", "central_width_68": 0.06},
                "theta_rad": {"unit": "rad", "central_width_68": 0.01},
                "phi_rad_wrapped": {"unit": "rad", "central_width_68": 0.02},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "loss_reference.json"
            path.write_text(json.dumps(payload))
            loaded = load_regression_loss_reference_stats(path, "p_phi_theta")
            relative = load_regression_loss_reference_stats(
                path, "p_phi_theta", momentum_residual="relative"
            )
        self.assertEqual(loaded["p_scale_gev"], 0.04)
        self.assertEqual(relative["p_scale_relative"], 0.06)
        self.assertNotIn("p_scale_gev", relative)
        self.assertEqual(loaded["theta_scale_rad"], 0.01)
        self.assertEqual(loaded["phi_scale_rad"], 0.02)


if __name__ == "__main__":
    unittest.main()
