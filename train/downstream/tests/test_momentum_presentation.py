"""Protect the scientific contracts of presentation-only exports."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "campaign"))
from momentum_presentation import (
    FIT_METRICS, fit_panel, headline_panel, make_axes, model_catalog, prepare_fits,
    prepare_headlines, select_runs, unique_baseline,
)
import matplotlib.pyplot as plt
from momentum_ml_presentation import prepare_component_rows


class MomentumPresentationTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "a": dict(backbone_run_id="adapteronly", model_family="adapteronly", embed_dim=128, labeled_events=100),
            "b": dict(backbone_run_id="wide", model_family="mamba1", embed_dim=1536, labeled_events=100),
            "c": dict(backbone_run_id="adapteronly", model_family="adapteronly", embed_dim=128, labeled_events=1000),
        }
        self.catalog, self.runs, self.ordered = model_catalog(self.manifest, self.manifest)

    def test_budget_is_common_not_independently_maximized(self):
        budget, runs = select_runs(self.runs)
        self.assertEqual(budget, 100)
        self.assertEqual(runs, {"a", "b"})
        with self.assertRaisesRegex(ValueError, "not available for every model"):
            select_runs(self.runs, 1000)

    def test_repeated_baseline_not_averaged_and_differences_rejected(self):
        row = {"mae": .2, "n": 100}
        self.assertEqual(unique_baseline([row, dict(row)], ("mae", "n")), row)
        for changed in ({"mae": .3, "n": 100}, {"mae": .2, "n": 50}):
            with self.assertRaisesRegex(ValueError, "differ across runs"):
                unique_baseline([row, changed], ("mae", "n"))

    def test_no_mixed_unit_native_loss_used_for_physical_errors(self):
        raw = []
        for variable, unit in (("p_gev", "GeV"), ("theta_deg", "deg"), ("phi_deg", "deg")):
            raw.append(dict(run_name="a", record_type="ml_error", space="kinematic", method="adapter",
                            variable=variable, unit=unit, mae=.1, n=100, comparison_truth="truth"))
        raw.append(dict(run_name="a", record_type="ml_error", space="native_target", method="adapter",
                        variable="theta", unit="rad", mae=123))
        rows = prepare_headlines(raw, {"a": self.runs["a"]})
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["mae"] == .1 for r in rows))
        raw[1]["unit"] = "rad"
        with self.assertRaisesRegex(ValueError, "Unexpected physical unit"):
            prepare_headlines(raw, {"a": self.runs["a"]})

    def test_fit_units_errorbars_and_failed_bin_gap(self):
        raw = []
        for i, status in enumerate(("ok", "moment_fallback_fit_failed", "ok")):
            raw.append(dict(run_id="a", method="adapter", bin_low_gev=i + .25,
                            bin_high_gev=i + .5, bin_center_gev=i + .375,
                            fit_mean=.01, fit_sigma=.02, fit_mean_error=.001,
                            fit_sigma_error=.002, n=200, fit_status=status))
        rows = prepare_fits(raw, {"a"}, self.runs, "delta_p_over_p")
        self.assertEqual([r["plotted"] for r in rows], [True, False, True])
        for metric, expected in ((FIT_METRICS[0], 2.), (FIT_METRICS[1], .02)):
            fig, axes = make_axes(1)
            fit_panel(axes[0], rows, metric, self.catalog, self.ordered, "all_models", "fit_sigma")
            line = axes[0].lines[0]
            self.assertAlmostEqual(line.get_ydata()[0], expected)
            self.assertTrue(np.isnan(line.get_ydata()[1]))
            # Error bar length uses fit uncertainty, never the residual width.
            segment = axes[0].collections[0].get_segments()[0]
            self.assertAlmostEqual(segment[1, 1] - segment[0, 1], .004 * metric[4])
            plt.close(fig)

    def component_fixture(self):
        return [dict(run_name="a", record_type="ml_error", space="component", method="adapter",
                     variable=variable, unit="GeV", n=100, comparison_truth="truth",
                     mae=mae, rmse=rmse, r2=r2)
                for variable, mae, rmse, r2 in (("px_gev", .1, 1., -2.),
                                                ("py_gev", .2, 2., .5),
                                                ("pz_gev", .4, 4., .9))]

    def test_cartesian_aggregation_pools_squared_errors(self):
        components, aggregate = prepare_component_rows(self.component_fixture(), {"a": self.runs["a"]})
        self.assertEqual(len(components), 3)
        row = aggregate[0]
        self.assertAlmostEqual(row["mae"], .7 / 3)
        self.assertAlmostEqual(row["rmse"], np.sqrt(21 / 3))
        self.assertNotAlmostEqual(row["rmse"], 7 / 3)
        self.assertAlmostEqual(row["r2"], -.6 / 3)
        self.assertEqual(row["n"], 100)  # track count is not tripled

    def test_cartesian_aggregation_rejects_missing_r2_or_unequal_counts(self):
        for key, value, message in (("r2", None, "Missing finite"), ("n", 50, "Unequal component counts")):
            rows = self.component_fixture()
            rows[0][key] = value
            with self.assertRaisesRegex(ValueError, message):
                prepare_component_rows(rows, {"a": self.runs["a"]})

    def test_r2_axis_preserves_negative_values(self):
        _, rows = prepare_component_rows(self.component_fixture(), {"a": self.runs["a"]})
        for scaling in (False, True):
            fig, axes = make_axes(1)
            headline_panel(axes[0], rows, ("cartesian", "R2", "R2"), self.catalog,
                           self.ordered, "all_models", scaling, field="r2")
            self.assertEqual(axes[0].get_yscale(), "linear")
            self.assertLess(axes[0].get_ylim()[0], rows[0]["r2"])
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
