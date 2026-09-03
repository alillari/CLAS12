import unittest

import numpy as np

from train.downstream.track_finding_metrics import (
    MatchConfig,
    event_track_metrics,
    summarize_event_metrics,
    track_momentum_by_label,
)


class TrackFindingMetricsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

