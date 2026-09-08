import unittest

import numpy as np
import torch

from fm4npp.datasets.dataset import MyCollator

from train.downstream.track_finding_metrics import (
    MatchConfig,
    compare_metric_summaries,
    event_track_metrics,
    summarize_event_metrics,
    track_momentum_by_label,
)
from train.downstream.eval.evaluate_track_finding import (
    EventMetricAccumulator,
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
