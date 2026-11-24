"""Integration checks for mode rejection and native evaluation outputs."""

import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from train.downstream.eval import evaluate_track_finding as evaluator
from train.downstream.eval.finding_prediction_archive import load_archive
from fm4npp.datasets.track_eligibility import primary_track_eligibility_contract

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
        labels = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1, -100]]).repeat(20, 1)
        points = torch.zeros(20, 10, 3)
        points[:, -1] = -100
        batch = {"points": points, "labels": labels, "coatjava_seg_pred": labels}
        masks = torch.tensor([[[.95, .02]] * 4 + [[.02, .95]] * 4 +
                              [[.1, .1], [.1, .1]]]).repeat(20, 1, 1)
        classes = torch.tensor([[[.05, .95], [.05, .95]]]).repeat(20, 1, 1)

        class FakeLoader(list):
            dataset = SimpleNamespace(track_eligibility_contract=primary_track_eligibility_contract())

        class FakeTrainer:
            def __init__(self, params, args):
                self.params = params
                # The effective dataset contract must win over stale saved settings.
                self.params.track_eligibility_contract = {"policy": "stale_checkpoint"}
                self.device = "cpu"
                self.val_data_loader = FakeLoader([batch])
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
                    self.assertEqual(result['track_eligibility_contract'], primary_track_eligibility_contract())
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
                        sweep_result = json.loads((out / 'threshold_sweep.json').read_text())
                        self.assertEqual(sweep_result['track_eligibility_contract'], primary_track_eligibility_contract())
                        self.assertEqual({r['track_eligibility_min_clusters'] for r in rows}, {'4'})
                    else:
                        headlines = [json.loads(line) for line in
                                     (out / 'campaign_headline_metrics.jsonl').read_text().splitlines()]
                        self.assertTrue(headlines)
                        for headline in headlines:
                            self.assertEqual(headline['track_eligibility_contract'], primary_track_eligibility_contract())

    def test_exported_cutoffs_match_direct_evaluation(self):
        points = torch.zeros(1, 7, 3)
        points[:, -1] = -100
        batch = dict(points=points, labels=torch.tensor([[0, 0, 0, 0, -1, -1, -100]]),
                     point_ids=torch.tensor([[2, 0, 1, 3, 5, 4, -1]]),
                     source_id=torch.tensor([17]), original_ordinal=torch.tensor([21]),
                     source_event_index=torch.tensor([0]))
        probabilities = torch.tensor([[[.25, .75], [.625, .375]]])
        masks = torch.tensor([[[.5, 1.], [.25, 1.], [.75, 1.], [.125, .75],
                               [.125, .25], [.25, .25], [0., 0.]]])

        class FakeLoader(list):
            dataset = SimpleNamespace(split="validation",
                                      track_eligibility_contract=primary_track_eligibility_contract())

        class FakeTrainer:
            def __init__(self, params, args):
                self.params = params
                self.device = "cpu"
                self.val_data_loader = FakeLoader([batch])
                self.val_data_loader.dataset.split = params.validation_split
                self.down_model = lambda *a, **kw: {"class_probs": probabilities, "mask_probs": masks}

            def launch(self):
                pass

            def cleanup(self):
                pass

            def _unpack_batch(self, batch):
                return batch["points"], batch["labels"], None, None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "head.pth"
            checkpoint.write_bytes(b"head fixture")
            model = root / "model.yaml"
            model.write_text("model:\n  reader_type: hdf5\n")
            config = root / "analysis.json"
            analysis = dict(model_yaml=str(model), model_config="model", checkpoint=str(checkpoint),
                            max_samples=1, assignment_option=2, evaluate_coatjava=False)

            def run(name, flags, **settings):
                output = root / name
                config.write_text(json.dumps({"analysis": dict(analysis, output_dir=str(output), **settings)}))
                with patch.object(sys, "argv", ["eval", "--analysis-config", str(config), *flags]), \
                     patch.object(evaluator, "DownstreamTrainer", FakeTrainer), \
                     patch.object(evaluator, "load_checkpoint"):
                    evaluator.main()
                return output

            validation = run("validation", ["--save-query-scores"], evaluation_split="validation",
                             assignment_threshold=.5)
            archive = load_archive(validation / "raw_predictions.npz", require_scores=True)
            self.assertEqual(archive.physical_keys(), [(17, 21)])
            self.assertEqual(archive.receipt["evaluation_contract"]["split"], "validation")
            np.testing.assert_array_equal(archive.arrays["offsets"], [0, 6])
            np.testing.assert_array_equal(archive.arrays["point_ids"], [2, 0, 1, 3, 5, 4])
            np.testing.assert_array_equal(archive.final_predictions(.2, .25), [0, 1, 0, 1, -1, -1])
            np.testing.assert_array_equal(archive.final_predictions(.2, .5), archive.final_predictions(.2))
            for index, (query, threshold) in enumerate(((None, .2), (.25, .2), (.25, .375), (1., .2))):
                with self.subTest(query=query, threshold=threshold):
                    out = run(f"test-{index}", ["--save-per-point-predictions"], assignment_threshold=threshold,
                              query_confidence_threshold=query)
                    with (out / "per_point_predictions.csv").open() as stream:
                        direct = [int(row["pred_label"]) for row in csv.DictReader(stream)]
                    np.testing.assert_array_equal(archive.final_predictions(threshold, query), direct)


if __name__ == "__main__":
    unittest.main()
