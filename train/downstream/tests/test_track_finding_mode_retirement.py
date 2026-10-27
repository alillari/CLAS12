"""Integration checks for mode rejection and native evaluation outputs."""

import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from train.downstream.eval import evaluate_track_finding as evaluator

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "campaign"), *sys.path]):
    from train.downstream.campaign import build_track_finding_manifest as builder
    from train.downstream.campaign import run_track_finding_campaign as runner


class TrackFindingRetirementTest(unittest.TestCase):
    def test_manifest_cli_and_generic_overrides_reject_retired_modes(self):
        with patch.object(sys, "argv", ["build", "--track-target-mode", "unified_noise_instance"]):
            with patch("sys.stderr"), self.assertRaises(SystemExit) as caught:
                builder.parse_args()
            self.assertEqual(caught.exception.code, 2)
        for override in ("track_target_mode=unified_noise_instance",
                         "noise_attribution_mode=truth_joint_hungarian_qualified"):
            # A later override must not conceal an earlier retired-mode request.
            with patch.object(sys, "argv", ["build", "--training-override", override,
                                           "--training-override", "track_target_mode=signal_only"]):
                args = builder.parse_args()
            with self.assertRaisesRegex(ValueError, "removed"):
                builder.parse_training_overrides(args)

    def test_rendering_rejects_old_modes_before_writing(self):
        manifest = {"base_model_yaml": "unused", "base_analysis_yaml": "unused",
                    "training_overrides": {"track_target_mode": "unified_noise_instance"},
                    "analysis_overrides": {"noise_attribution_mode": "truth_joint_hungarian_qualified"}}
        with patch.object(runner, "load_base_model_config", return_value={}), \
             patch.object(runner, "write_yaml") as write:
            with self.assertRaisesRegex(ValueError, "removed"):
                runner.render_model_yaml(manifest, {})
            write.assert_not_called()
        with patch.object(runner, "read_yaml", return_value={"analysis": {}}), \
             patch.object(runner, "write_yaml") as write:
            with self.assertRaisesRegex(ValueError, "removed"):
                runner.render_analysis_yaml(manifest, {})
            write.assert_not_called()

    def test_old_manifest_fails_before_preflight_or_launch(self):
        for scope in ("manifest", "run"):
            manifest = {"campaign_dir": "/unused", "runs": [{"run_id": "test"}]}
            target = manifest if scope == "manifest" else manifest["runs"][0]
            target["analysis_overrides"] = {
                "noise_attribution_mode": "truth_joint_hungarian_qualified",
            }
            args = SimpleNamespace(manifest="unused.yaml", only=None, limit=None)
            with self.subTest(scope=scope), \
                 patch.object(runner, "parse_args", return_value=args), \
                 patch.object(runner, "read_yaml", return_value=manifest), \
                 patch.object(runner, "normalize_manifest_paths", side_effect=lambda data: data), \
                 patch.object(runner, "preflight_dataset") as preflight, \
                 patch.object(runner, "train_if_needed") as train:
                with self.assertRaisesRegex(ValueError, "removed"):
                    runner.main()
                preflight.assert_not_called()
                train.assert_not_called()

    def test_native_evaluation_and_threshold_sweep_outputs(self):
        labels = torch.tensor([[0, 0, 1, 1, -1, -100]]).repeat(20, 1)
        points = torch.zeros(20, 6, 3)
        points[:, -1] = -100
        batch = {"points": points, "labels": labels, "coatjava_seg_pred": labels}
        masks = torch.tensor([[[.95, .02], [.95, .02], [.02, .95], [.02, .95],
                               [.1, .1], [.1, .1]]]).repeat(20, 1, 1)
        classes = torch.tensor([[[.05, .95], [.05, .95]]]).repeat(20, 1, 1)

        class FakeTrainer:
            def __init__(self, params, args):
                self.params = params
                self.device = "cpu"
                self.val_data_loader = [batch]
                self.down_model = lambda *a, **kw: {"class_probs": classes, "mask_probs": masks}
                self.output_dir = Path(args.root_dir).parent

            def launch(self):
                self.output_dir.mkdir(parents=True, exist_ok=True)

            def cleanup(self):
                pass

            def _unpack_batch(self, batch):
                return batch["points"], batch["labels"], None, None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.yaml"
            model.write_text("model:\n  track_target_mode: signal_only\n  validation_ari_mode: inclusive\n")
            for sweep in (False, True):
                with self.subTest(sweep=sweep):
                    out = root / ("sweep" if sweep else "evaluation")
                    analysis = {"model_yaml": str(model), "model_config": "model",
                                "checkpoint": "unused.pth", "output_dir": str(out),
                                "max_samples": 20, "assignment_option": 2,
                                "assignment_threshold": .2, "evaluate_coatjava": True}
                    config = root / "analysis.json"
                    config.write_text(json.dumps({"analysis": analysis}))
                    argv = ["eval", "--analysis-config", str(config), "--output-dir", str(out)]
                    if sweep:
                        argv += ["--assignment-thresholds", "0", ".2", "--calibration-modulus", "2"]
                    else:
                        argv += ["--save-per-point-predictions"]
                    with patch.object(sys, "argv", argv), \
                         patch.object(evaluator, "DownstreamTrainer", FakeTrainer), \
                         patch.object(evaluator, "load_checkpoint"):
                        evaluator.main()
                    result = json.loads((out / ("selected_heldout_summary.json" if sweep else "summary.json")).read_text())
                    self.assertEqual(result["metrics"]["ari_signal"], 1.0)
                    self.assertEqual(result["metrics"]["background_rejection"], 1.0)
                    self.assertEqual(result["metrics"], result["native_metrics"])
                    self.assertEqual(result["baselines"], result["native_baselines"])
                    self.assertNotIn("noise_attribution", result)
                    self.assertFalse((out / "per_event_noise_attribution.csv").exists())
                    csv_path = out / ("threshold_metrics.csv" if sweep else "per_event_metrics.csv")
                    with csv_path.open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual({row["metric_view"] for row in rows}, {"native"})
                    self.assertEqual(len(rows), 8 if sweep else 40)
                    if sweep:
                        self.assertEqual(result["selected_threshold"], .2)


if __name__ == "__main__":
    unittest.main()
