import unittest
import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from fm4npp.datasets.dataset import MyCollator

from train.downstream.loss import PointHungarianMatcher, compute_point_loss
from train.downstream.track_finding_metrics import (
    MatchConfig,
    compare_metric_summaries,
    event_track_metrics,
    summarize_event_metrics,
    track_momentum_by_label,
)
from train.downstream.track_finding_contract import validate_track_finding_modes
from train.downstream.track_finding_experiment import TrackFindingExperimentConfig, resolve_params
from train.downstream.track_finding_trainer import DownstreamTrainer
from train.downstream.track_finding_targets import (
    SIGNAL_ONLY,
    build_track_instance_targets,
    validation_ari_metric,
)
from train.downstream.eval.evaluate_track_finding import (
    EventMetricAccumulator,
    evaluate_event_method,
    read_analysis,
    parse_args as evaluation_args,
    apply_assignment_threshold,
    select_threshold,
    threshold_partition,
)


class TrackFindingMetricsTest(unittest.TestCase):
    @staticmethod
    def _dict_sample(n_points, coatjava_labels):
        return {
            "points": torch.zeros(n_points, 3),
            "target": torch.zeros(n_points, dtype=torch.long),
            "knearest_points": torch.zeros(n_points, 3),
            "reg_target": torch.zeros(n_points, 7),
            "pid_target": torch.zeros(n_points, dtype=torch.long),
            "noise_target": torch.zeros(n_points, dtype=torch.long),
            "coatjava_seg_pred": torch.as_tensor(coatjava_labels, dtype=torch.long),
        }

    def test_collator_preserves_and_pads_coatjava_sidecar(self):
        batch = MyCollator()([
            self._dict_sample(2, [0, -1]),
            self._dict_sample(3, [1, 1, -1]),
        ])
        self.assertEqual(tuple(batch["coatjava_seg_pred"].shape), (2, 3))
        self.assertEqual(batch["coatjava_seg_pred"][0].tolist(), [0, -1, -100])

    def test_background_label_is_excluded_from_primary_ari(self):
        truth = np.array([1, 1, 2, 2, -1, -1])
        pred = np.array([7, 7, 8, 8, 9, 9])
        pred_signal = np.array([True, True, True, True, True, True])
        row = event_track_metrics(truth, pred, pred_signal_mask=pred_signal)
        self.assertEqual(row["n_true_tracks"], 2)
        self.assertEqual(row["n_background_points"], 2)
        self.assertAlmostEqual(row["ari_signal"], 1.0)
        self.assertLess(row["background_rejection"], 1.0)

    def test_retired_modes_are_rejected_in_configs_and_checkpoint_metadata(self):
        for key, value in (("track_target_mode", "unified_noise_instance"),
                           ("noise_attribution_mode", "truth_joint_hungarian_qualified")):
            for config in ({key: value}, {"params": {key: value}},
                           {"params": {"params": {key: value}}}):
                with self.subTest(config=config), self.assertRaisesRegex(ValueError, "removed"):
                    validate_track_finding_modes(config)
        validate_track_finding_modes({})
        validate_track_finding_modes({"track_target_mode": "signal_only", "noise_attribution_mode": "native"})
        labels = torch.tensor([[-1, 0]])
        with self.assertRaisesRegex(ValueError, "removed"):
            build_track_instance_targets(labels, torch.ones_like(labels, dtype=torch.bool),
                                         mode="unified_noise_instance")

    def test_retired_checkpoint_is_rejected_before_loading_weights(self):
        trainer = DownstreamTrainer.__new__(DownstreamTrainer)
        trainer.device = "cpu"
        for checkpoint in ({"track_target_mode": "unified_noise_instance"},
                           {"params": {"params": {"track_target_mode": "unified_noise_instance"}}}):
            with patch("torch.load", return_value=checkpoint):
                with self.assertRaisesRegex(ValueError, "checkpoint.*removed"):
                    trainer.load_checkpoint("retired.pth", inference=True)
                with self.assertRaisesRegex(ValueError, "checkpoint.*removed"):
                    trainer.restore_checkpoint("retired.pth")

    def test_retired_training_and_analysis_yaml_fail_early(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("model:\n  track_target_mode: unified_noise_instance\n")
            with self.assertRaisesRegex(ValueError, "removed"):
                resolve_params(TrackFindingExperimentConfig(yaml_config=str(path), config="model"))
            with self.assertRaisesRegex(ValueError, "removed"):
                DownstreamTrainer(SimpleNamespace(track_target_mode="unified_noise_instance"), None)
            path.write_text("analysis:\n  noise_attribution_mode: truth_joint_hungarian_qualified\n")
            with self.assertRaisesRegex(ValueError, "removed"):
                read_analysis(path)

    def test_evaluation_cli_rejects_retired_attribution(self):
        with patch.object(sys, "argv", ["eval", "--analysis-config", "unused.yaml",
                                      "--noise-attribution-mode", "truth_joint_hungarian_qualified"]):
            with patch("sys.stderr"), self.assertRaises(SystemExit) as caught:
                evaluation_args()
            self.assertEqual(caught.exception.code, 2)

    def test_background_and_padding_produce_no_targets(self):
        labels = torch.tensor([[-1, -1, -100], [-100, -100, -100]])
        valid = torch.tensor([[True, True, False], [False, False, False]])
        targets, _inverse = build_track_instance_targets(
            labels,
            valid,
            mode=SIGNAL_ONLY,
        )
        self.assertEqual(tuple(targets[0]["masks"].shape), (0, 3))
        self.assertEqual(targets[0]["masks"].tolist(), [])
        self.assertEqual(tuple(targets[1]["masks"].shape), (0, 3))

    def test_signal_target_has_a_finite_hungarian_loss(self):
        labels = torch.tensor([[-1, -1, 0, 0, -100]])
        valid = torch.tensor([[True, True, True, True, False]])
        targets, _inverse = build_track_instance_targets(
            labels,
            valid,
            mode=SIGNAL_ONLY,
        )
        outputs = {
            "pred_probs": torch.tensor([[
                [0.05, 0.95],
                [0.05, 0.95],
                [0.95, 0.05],
            ]]),
            "pred_masks": torch.tensor([[
                [0.95, 0.95, 0.05, 0.05, 0.50],
                [0.05, 0.05, 0.95, 0.95, 0.50],
                [0.05, 0.05, 0.05, 0.05, 0.50],
            ]]),
        }
        matcher = PointHungarianMatcher(cost_class=1, cost_dice=1, cost_focal=20)
        indices = matcher(outputs, targets, valid)
        self.assertEqual(indices[0][0].numel(), 1)
        losses = compute_point_loss(outputs, targets, valid, matcher)
        for value in losses.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_signal_only_targets_exclude_background(self):
        labels = torch.tensor([[-1, -1, 0, 0, -100]])
        valid = torch.tensor([[True, True, True, True, False]])
        targets, _inverse = build_track_instance_targets(labels, valid, mode=SIGNAL_ONLY)
        self.assertEqual(targets[0]["labels"].tolist(), [1])
        self.assertEqual(targets[0]["masks"].tolist(), [[0.0, 0.0, 1.0, 1.0, 0.0]])

    def test_inclusive_validation_ari_includes_noise_but_excludes_padding(self):
        truth = np.array([-1, -1, 0, 0, -100])
        pred = np.array([7, 7, 4, 4, 999])
        valid = np.array([True, True, True, True, False])
        row = event_track_metrics(truth, pred, valid_mask=valid)
        self.assertEqual(validation_ari_metric("inclusive"), "ari_with_background")
        self.assertEqual(validation_ari_metric("signal"), "ari_signal")
        self.assertAlmostEqual(row[validation_ari_metric("inclusive")], 1.0)
        # A distinct noise query outside the true signal support must not
        # penalize the one-track signal-ARI edge case.
        self.assertAlmostEqual(row[validation_ari_metric("signal")], 1.0)
        with self.assertRaises(ValueError):
            validation_ari_metric("unknown")

    def test_evaluation_never_relabels_a_query_using_truth(self):
        truth = np.array([-1, -1, 0, 0])
        pred = np.array([6, 6, 2, 2])
        signal = np.ones_like(truth, dtype=bool)
        evaluated = evaluate_event_method(
            "adapter", 0, 0, 0, truth, pred, signal, signal, None, MatchConfig(), 0.001,
        )
        np.testing.assert_array_equal(evaluated["pred"], pred)
        np.testing.assert_array_equal(evaluated["signal"], signal)
        row = evaluated["rows"][0]
        self.assertEqual(row["n_pred_tracks"], 2)
        self.assertEqual(row["background_rejection"], 0.0)
        self.assertEqual(row["ari_signal"], 1.0)
        self.assertEqual(row["metric_view"], "native")

    def test_matching_reports_efficiency_and_purity(self):
        truth = np.array([1, 1, 1, 2, 2, 2, -1])
        pred = np.array([0, 0, 0, 1, 1, 1, -1])
        row = event_track_metrics(
            truth,
            pred,
            config=MatchConfig(iou_threshold=0.5, min_purity=0.5, min_efficiency=0.5),
        )
        self.assertEqual(row["n_matched_tracks"], 2)
        self.assertAlmostEqual(row["track_efficiency"], 1.0)
        self.assertAlmostEqual(row["track_purity"], 1.0)
        self.assertEqual(row["fake_rate"], 0.0)

    def test_split_and_merge_are_flagged(self):
        truth = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        pred = np.array([0, 0, 1, 1, 1, 1, 1, 1])
        row = event_track_metrics(truth, pred)
        self.assertGreater(row["split_rate"], 0.0)
        self.assertGreater(row["merge_rate"], 0.0)

    def test_summary_aggregates_counts(self):
        rows = [
            event_track_metrics(np.array([1, 1, 2, 2]), np.array([0, 0, 1, 1])),
            event_track_metrics(np.array([1, 1, -1]), np.array([0, 0, -1])),
        ]
        summary = summarize_event_metrics(rows)
        self.assertEqual(summary["n_events"], 2)
        self.assertEqual(summary["n_true_tracks"], 3)
        self.assertEqual(summary["n_matched_tracks"], 3)

    def test_track_momentum_by_label_uses_signal_tracks(self):
        truth = np.array([1, 1, -1, 2])
        reg = np.array([
            [1000.0, 0.0, 0.0],
            [1000.0, 0.0, 0.0],
            [9999.0, 0.0, 0.0],
            [0.0, 2000.0, 0.0],
        ])
        values = track_momentum_by_label(truth, reg, momentum_scale=0.001)
        self.assertAlmostEqual(values[1]["p_gev"], 1.0)
        self.assertAlmostEqual(values[2]["pt_gev"], 2.0)
        self.assertNotIn(-1, values)

    def test_comparison_is_candidate_minus_baseline(self):
        deltas = compare_metric_summaries(
            {"ari_signal": 0.8, "fake_rate": 0.1},
            {"ari_signal": 0.7, "fake_rate": 0.2},
        )
        self.assertAlmostEqual(deltas["ari_signal"], 0.1)
        self.assertAlmostEqual(deltas["fake_rate"], -0.1)
        self.assertIsNone(deltas["track_efficiency_global"])

    def test_threshold_application_marks_low_score_points_unassigned(self):
        inferred = {
            "assignments": torch.tensor([[4, 7, -1]]),
            "classes": torch.tensor([[1, 1, 0]]),
            "scores": torch.tensor([[0.01, 0.20, -1.0]]),
        }
        assignments, classes = apply_assignment_threshold(inferred, 0.05)
        self.assertEqual(assignments.tolist(), [[-1, 7, -1]])
        self.assertEqual(classes.tolist(), [[0, 1, 0]])

    def test_streaming_sweep_summary_matches_list_summary(self):
        rows = [
            event_track_metrics(np.array([1, 1, -1]), np.array([0, 0, -1])),
            event_track_metrics(np.array([1, -1, -1]), np.array([0, 1, -1])),
        ]
        accumulator = EventMetricAccumulator()
        for row in rows:
            accumulator.add(row)
        expected = summarize_event_metrics(rows)
        actual = accumulator.summary()
        for key, value in expected.items():
            if value is None:
                self.assertIsNone(actual[key])
            else:
                self.assertAlmostEqual(actual[key], value)

    def test_threshold_selection_uses_calibration_constraint_only(self):
        summaries = {
            0.0: {"ari_with_background": 0.70, "track_efficiency_global": 0.96},
            0.05: {"ari_with_background": 0.80, "track_efficiency_global": 0.951},
            0.10: {"ari_with_background": 0.90, "track_efficiency_global": 0.94},
        }
        selected, floor = select_threshold(summaries, min_efficiency_retention=0.99)
        self.assertAlmostEqual(floor, 0.9504)
        self.assertEqual(selected, 0.05)
        self.assertEqual(threshold_partition(42, 10, 0), threshold_partition(42, 10, 0))


if __name__ == "__main__":
    unittest.main()
