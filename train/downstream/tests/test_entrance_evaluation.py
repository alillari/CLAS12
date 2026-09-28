import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from train.downstream.evaluation_contract import (
    EVALUATION_CONTRACT, configure_entrance_evaluation, is_entrance_evaluation,
)
from train.downstream.eval.evaluate_track_regression import (
    AUX_LAYOUT, CVT_ENTRANCE_COLUMNS, aux_row_for_sample, cvt_entrance_vector,
    entrance_comparison_arrays, load_aux_layout, calculate_ml_metrics,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "campaign"))
from campaign_util import collate_summary, render_analysis_yaml, read_yaml
import run_track_regression_campaign as runner
from plot_track_regression_campaign import read_jsonl, read_csv_rows


LAYOUT = list(AUX_LAYOUT) + list(CVT_ENTRANCE_COLUMNS) + ["cvt_doca_phi0"]


class EntranceEvaluationTest(unittest.TestCase):
    def row(self, phi=math.pi - .01):
        row = np.full(len(LAYOUT), np.nan)
        row[16:19] = [2., .8, phi]
        return row

    def test_cvt_conversion_ignores_all_doca_inputs_and_scales_only_p(self):
        row = self.row()
        expected = np.array([2*np.sin(.8)*np.cos(row[18]),
                             2*np.sin(.8)*np.sin(row[18]), 2*np.cos(.8)])
        np.testing.assert_allclose(cvt_entrance_vector(row, LAYOUT), expected)
        for phi0 in (0., -2., np.nan, np.inf):
            row[:16] = np.arange(16) * 100.
            row[19] = phi0
            np.testing.assert_allclose(cvt_entrance_vector(row, LAYOUT), expected)
        row[16] *= 1000
        np.testing.assert_allclose(cvt_entrance_vector(row, LAYOUT, .001), expected)

    def test_layout_names_control_column_selection(self):
        row = self.row()
        order = list(reversed(range(len(LAYOUT))))
        np.testing.assert_allclose(
            cvt_entrance_vector(row[order], [LAYOUT[i] for i in order]),
            cvt_entrance_vector(row, LAYOUT),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"aux_target_layout": LAYOUT}))
            self.assertEqual(load_aux_layout(root), LAYOUT)
            metadata.write_text(json.dumps({"aux_target_layout": list(AUX_LAYOUT)}))
            with self.assertRaisesRegex(ValueError, "no DOCA fallback"):
                load_aux_layout(root)
        with self.assertRaisesRegex(ValueError, "metadata width"):
            cvt_entrance_vector(row[:16], LAYOUT)

    def test_missing_entrance_phi_never_uses_valid_legacy_vector(self):
        row = self.row()
        row[:16] = 1.
        row[18] = np.nan
        row[19] = 1.2
        self.assertTrue(np.isnan(cvt_entrance_vector(row, LAYOUT)).all())

    def test_row_selection_ignores_legacy_validity_and_keeps_truth_segment(self):
        first = self.row(phi=.3)
        second = self.row(phi=1.2)
        second[:16] = 1.
        second[19] = 0.
        class Dataset:
            memmap_seg_target = [np.array([2, 2, 3])]
        rows = np.stack([first, second, self.row(phi=2.)])
        selected = aux_row_for_sample(
            Dataset(), [rows], 0, 7, truth_segment_label=2,
            valid_columns=[16, 17, 18],
        )
        self.assertAlmostEqual(selected[18], .3)
        self.assertTrue(np.isnan(selected[19]))

    def test_metrics_use_entrance_phi_and_raw_adapter_without_charge(self):
        truth = cvt_entrance_vector(self.row(phi=math.pi-.01), LAYOUT)
        cvt = cvt_entrance_vector(self.row(phi=-math.pi+.01), LAYOUT)
        record = {}
        for prefix, vector in (("true", truth), ("adapter", truth), ("cvt_entrance", cvt)):
            record.update({f"{prefix}_{axis}_gev": value for axis, value in zip(("px", "py", "pz"), vector)})
        record.update(legacy_cvt_px_gev=1e6, legacy_cvtrec_py_gev=-1e6, charge=0)
        actual_truth, predictions = entrance_comparison_arrays([record])
        self.assertEqual(set(predictions), {"adapter", "cvt"})
        np.testing.assert_array_equal(actual_truth[0], truth)
        rows, nested = calculate_ml_metrics(actual_truth, predictions)
        self.assertAlmostEqual(nested["cvt"]["phi_deg"]["mae"], np.degrees(.02))
        self.assertEqual(nested["adapter"]["phi_deg"]["mae"], 0.)
        self.assertEqual(nested["adapter"]["px_gev"]["n"], 1)
        self.assertFalse(any(row["method"] == "cvtrec" for row in rows))
        # Legacy vectors, phi0 and charge cannot influence any metric input.
        record.update(legacy_cvt_px_gev=np.nan, legacy_cvtrec_py_gev=42., charge=-1)
        unchanged_truth, unchanged_predictions = entrance_comparison_arrays([record])
        np.testing.assert_array_equal(unchanged_truth, actual_truth)
        for method in predictions:
            np.testing.assert_array_equal(unchanged_predictions[method], predictions[method])

    def test_old_configs_migrate_and_new_defaults_are_entrance(self):
        legacy = dict(comparison_truth="mctrue_swingback_doca", swingback_enabled=True,
                      swingback_r_hit_cm=6.5, swingback_magnetic_field_t=5,
                      swingback_polarity=1, write_unswung_diagnostics=True)
        with patch("warnings.warn") as notice:
            configure_entrance_evaluation(legacy)
        self.assertIn("retired", notice.call_args.args[0])
        self.assertTrue(is_entrance_evaluation(legacy))
        self.assertNotIn("swingback_r_hit_cm", legacy)
        self.assertFalse(legacy["swingback_enabled"])
        self.assertTrue(is_entrance_evaluation(configure_entrance_evaluation({})))
        self.assertFalse(is_entrance_evaluation({"comparison_truth": "mctrue_inner_hit"}))

    def test_rendering_migrates_old_campaign_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.yaml"
            base.write_text("analysis:\n  comparison_truth: mctrue_swingback_doca\n  swingback_enabled: true\n")
            manifest = dict(base_analysis_yaml=str(base), artifact_root=tmp, campaign_name="test")
            run = dict(run_id="one", model_yaml=str(root/"model.yaml"), model_config="one",
                       adapter_checkpoint=str(root/"adapter.pth"), training_log=str(root/"train.log"),
                       evaluation_dir=str(root/"evaluation"), max_samples=10,
                       analysis_yaml=str(root/"analysis.yaml"), use_pretrained_backbone=False)
            with patch("warnings.warn") as notice:
                render_analysis_yaml(manifest, run)
            notice.assert_called_once()
            self.assertTrue(is_entrance_evaluation(read_yaml(root/"analysis.yaml")["analysis"]))

    def test_collation_rejects_legacy_before_touching_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            (evaluation/"summary.json").write_text(json.dumps({"comparison_truth":"mctrue_swingback_doca"}))
            summary = root / "summary"
            summary.mkdir()
            output = summary / "campaign_headline_metrics.jsonl"
            output.write_text("keep me\n")
            with self.assertRaisesRegex(ValueError, "Reevaluate"):
                collate_summary(dict(campaign_dir=tmp, runs=[dict(evaluation_dir=str(evaluation))]))
            self.assertEqual(output.read_text(), "keep me\n")

    def test_plot_readers_reject_legacy_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "headlines.jsonl"
            path.write_text(json.dumps({"comparison_truth":"mctrue_swingback_doca"})+'\n')
            with self.assertRaisesRegex(ValueError, "Reevaluate"):
                read_jsonl(path)
            path.write_text(json.dumps(configure_entrance_evaluation({}))+'\n')
            self.assertEqual(len(read_jsonl(path)), 1)
            path = root / "fits.csv"
            path.write_text("method,fit_sigma\ncvt,0.1\n")
            with self.assertRaisesRegex(ValueError, "Reevaluate"):
                read_csv_rows(path)
            path.write_text(f"evaluation_contract,comparison_truth,method,fit_sigma\n{EVALUATION_CONTRACT},mctrue_inner_hit,cvt,0.1\n")
            self.assertEqual(len(read_csv_rows(path)), 1)

    def test_runner_skips_only_current_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.json"
            run = dict(run_id="one", evaluation_dir=tmp, eval_stdout=str(root/"eval.log"))
            args = SimpleNamespace(skip_eval=False, force_eval=False, cuda_device="0")
            status = {"runs":{"one":{"status":"eval_done"}}}
            with patch.object(runner, "load_status", return_value=status), \
                 patch.object(runner, "update_status"), \
                 patch.object(runner, "command_env", side_effect=RuntimeError("evaluation reached")):
                summary.write_text(json.dumps(configure_entrance_evaluation({})))
                runner.eval_if_needed(args, {}, root/"status.yaml", status, run)
                summary.write_text(json.dumps({"comparison_truth":"mctrue_swingback_doca"}))
                with self.assertRaisesRegex(RuntimeError, "evaluation reached"):
                    runner.eval_if_needed(args, {}, root/"status.yaml", status, run)

    def test_evaluation_only_mode_never_trains_with_missing_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "adapter.pth"
            run = dict(run_id="one", adapter_checkpoint=str(checkpoint))
            args = SimpleNamespace(skip_train=True)
            with self.assertRaisesRegex(FileNotFoundError, "--skip-train"):
                runner.train_if_needed(args, {}, root/"status.yaml", {}, run)
            checkpoint.touch()
            with patch.object(runner, "run_logged_command") as train:
                runner.train_if_needed(args, {}, root/"status.yaml", {}, run)
            train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
