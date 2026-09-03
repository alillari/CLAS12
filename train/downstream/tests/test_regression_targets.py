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
from regression_utils import (
    REGRESSION_TARGET_COLUMNS,
    load_regression_target_stats,
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
        cart = target_to_cartesian_numpy(target, "pt_phi_eta")
        np.testing.assert_allclose(cart, reg[:, :3], rtol=1.0e-6, atol=1.0e-6)

    def test_pt_phi_eta_torch_matches_numpy(self):
        reg = np.array([[300.0, 400.0, 1200.0, 0.0, 0.0, 0.0, 1300.0]])
        expected = transform_regression_target_numpy(reg, "pt_phi_eta")
        actual = transform_regression_target_torch(
            torch.as_tensor(reg, dtype=torch.float32), "pt_phi_eta"
        ).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)

    def test_p_phi_theta_round_trip_numpy(self):
        reg = np.array([
            [300.0, 400.0, 1200.0, 0.0, 0.0, 0.0, 1300.0],
            [-500.0, 250.0, -700.0, 0.0, 0.0, 0.0, 900.0],
        ])
        target = transform_regression_target_numpy(reg, "p_phi_theta")
        expected_p = np.linalg.norm(reg[:, :3], axis=1)
        expected_theta = np.arctan2(np.hypot(reg[:, 0], reg[:, 1]), reg[:, 2])
        np.testing.assert_allclose(target[:, 0], expected_p)
        np.testing.assert_allclose(target[:, 2], expected_theta)
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


if __name__ == "__main__":
    unittest.main()
