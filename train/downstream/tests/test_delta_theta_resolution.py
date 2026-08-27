import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

DOWNSTREAM_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = DOWNSTREAM_DIR / "eval"
CAMPAIGN_DIR = DOWNSTREAM_DIR / "campaign"
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(CAMPAIGN_DIR))

from evaluate_track_regression import (  # noqa: E402
    calculate_delta_p_over_p_fit_rows,
    calculate_delta_theta_fit_rows,
)
from plot_track_regression_campaign import make_theta_resolution_plots  # noqa: E402


def vector_from_p_theta(p_gev, theta_deg):
    theta_rad = np.radians(theta_deg)
    return np.column_stack((
        p_gev * np.sin(theta_rad),
        np.zeros_like(p_gev),
        p_gev * np.cos(theta_rad),
    ))


class DeltaThetaResolutionTest(unittest.TestCase):
    def test_delta_theta_fit_uses_degree_residuals(self):
        rng = np.random.default_rng(12345)
        true_p = np.full(400, 1.0)
        true_theta = np.linspace(45.0, 75.0, len(true_p))
        residual = rng.normal(loc=2.0, scale=0.3, size=len(true_p))
        truth = vector_from_p_theta(true_p, true_theta)
        prediction = vector_from_p_theta(true_p, true_theta + residual)
        config = {
            "delta_theta_min_bin_entries": 5,
            "delta_theta_histogram_bins": 10,
            "delta_theta_min_populated_histogram_bins": 1,
            "delta_theta_fit_quantile": 1.0,
        }

        rows = calculate_delta_theta_fit_rows(
            truth,
            {"adapter": prediction, "cvt": prediction},
            [0.5, 1.5],
            config,
        )

        adapter = next(row for row in rows if row["method"] == "adapter")
        self.assertEqual(adapter["n"], len(true_p))
        self.assertIn(adapter["fit_status"], {"ok", "moment_fallback_fit_failed"})
        self.assertAlmostEqual(adapter["fit_mean"], float(np.mean(residual)), delta=0.1)
        self.assertAlmostEqual(adapter["fit_sigma"], float(np.std(residual, ddof=1)), delta=0.1)

    def test_delta_theta_zero_width_reports_status(self):
        true_p = np.full(8, 1.0)
        true_theta = np.linspace(50.0, 70.0, len(true_p))
        truth = vector_from_p_theta(true_p, true_theta)
        prediction = vector_from_p_theta(true_p, true_theta + 3.0)
        config = {
            "delta_theta_min_bin_entries": 5,
            "delta_theta_histogram_bins": 10,
            "delta_theta_min_populated_histogram_bins": 1,
            "delta_theta_fit_quantile": 1.0,
        }

        rows = calculate_delta_theta_fit_rows(
            truth,
            {"adapter": prediction, "cvt": prediction},
            [0.5, 1.5],
            config,
        )

        adapter = next(row for row in rows if row["method"] == "adapter")
        self.assertEqual(adapter["fit_status"], "zero_width")
        self.assertAlmostEqual(adapter["fit_mean"], 3.0)
        self.assertIsNone(adapter["fit_sigma"])

    def test_sparse_delta_theta_bin_is_skipped(self):
        true_p = np.full(4, 1.0)
        true_theta = np.linspace(50.0, 70.0, len(true_p))
        truth = vector_from_p_theta(true_p, true_theta)
        prediction = vector_from_p_theta(true_p, true_theta + 1.0)
        config = {
            "delta_theta_min_bin_entries": 5,
            "delta_theta_histogram_bins": 10,
            "delta_theta_min_populated_histogram_bins": 1,
            "delta_theta_fit_quantile": 1.0,
        }

        rows = calculate_delta_theta_fit_rows(
            truth,
            {"adapter": prediction, "cvt": prediction},
            [0.5, 1.5],
            config,
        )

        adapter = next(row for row in rows if row["method"] == "adapter")
        self.assertEqual(adapter["fit_status"], "skipped_sparse")
        self.assertIsNone(adapter["fit_mean"])
        self.assertIsNone(adapter["fit_sigma"])

    def test_delta_p_over_p_schema_remains_stable(self):
        true_p = np.full(8, 1.0)
        true_theta = np.linspace(50.0, 70.0, len(true_p))
        truth = vector_from_p_theta(true_p, true_theta)
        prediction = vector_from_p_theta(true_p * 1.1, true_theta)
        config = {
            "delta_p_over_p_min_bin_entries": 5,
            "delta_p_over_p_histogram_bins": 10,
            "delta_p_over_p_min_populated_histogram_bins": 1,
            "delta_p_over_p_fit_quantile": 1.0,
        }

        rows = calculate_delta_p_over_p_fit_rows(
            truth,
            {"adapter": prediction, "cvt": prediction},
            [0.5, 1.5],
            config,
        )

        expected_fields = [
            "group", "bin", "bin_low_gev", "bin_high_gev", "bin_center_gev",
            "method", "n", "fit_mean", "fit_sigma", "fit_mean_error",
            "fit_sigma_error", "fit_status",
        ]
        self.assertEqual(list(rows[0]), expected_fields)
        adapter = next(row for row in rows if row["method"] == "adapter")
        self.assertAlmostEqual(adapter["fit_mean"], 0.1)

    def test_campaign_theta_plot_smoke(self):
        rows = []
        for method, family in (("adapter", "adapter_only"), ("cvt", "conventional")):
            for center in (0.625, 0.875):
                rows.append({
                    "run_id": "adapteronly_label100",
                    "use_pretrained_backbone": "false",
                    "model_family": "adapteronly" if method == "adapter" else "mamba1",
                    "embed_dim": "128",
                    "pretrain_events": "0",
                    "labeled_events": "100",
                    "method": method,
                    "bin_center_gev": str(center),
                    "bin_low_gev": str(center - 0.125),
                    "bin_high_gev": str(center + 0.125),
                    "n": "250",
                    "fit_mean": "0.1",
                    "fit_sigma": "1.2" if family == "adapter_only" else "1.5",
                    "fit_status": "ok",
                })
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            make_theta_resolution_plots(rows, output_dir)
            plot_dir = output_dir / "momentum_resolution"
            self.assertTrue((plot_dir / "presentation_sigma_delta_theta.png").exists())
            self.assertTrue((plot_dir / "delta_theta_all_sigma.png").exists())


if __name__ == "__main__":
    unittest.main()
