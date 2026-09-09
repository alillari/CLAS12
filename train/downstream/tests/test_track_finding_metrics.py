import unittest

import numpy as np
import torch

from fm4npp.datasets.dataset import MyCollator

from train.downstream.loss import PointHungarianMatcher, compute_point_loss
from train.downstream.track_finding_metrics import (
    MatchConfig,
    NOISE_ATTRIBUTION_NATIVE,
    NOISE_ATTRIBUTION_TRUTH_JOINT_HUNGARIAN_QUALIFIED,
    canonicalize_noise_query,
    compare_metric_summaries,
    event_track_metrics,
    summarize_event_metrics,
    track_momentum_by_label,
)
from train.downstream.track_finding_targets import (
    SIGNAL_ONLY,
    UNIFIED_NOISE_INSTANCE,
    build_track_instance_targets,
    validation_ari_metric,
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

    def test_unified_noise_instance_is_an_object_target_without_padding(self):
        labels = torch.tensor([[-1, -1, 0, 0, -100]])
        valid = torch.tensor([[True, True, True, True, False]])
        targets, inverse = build_track_instance_targets(
            labels,
            valid,
            mode=UNIFIED_NOISE_INSTANCE,
        )
        self.assertEqual(targets[0]["labels"].tolist(), [1, 1])
        self.assertEqual(targets[0]["masks"].tolist(), [
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
        ])
        self.assertEqual(inverse[0].numel(), 4)

    def test_all_noise_is_one_target_and_padding_is_never_a_target(self):
        labels = torch.tensor([[-1, -1, -100], [-100, -100, -100]])
        valid = torch.tensor([[True, True, False], [False, False, False]])
        targets, _inverse = build_track_instance_targets(
            labels,
            valid,
            mode=UNIFIED_NOISE_INSTANCE,
        )
        self.assertEqual(tuple(targets[0]["masks"].shape), (1, 3))
        self.assertEqual(targets[0]["masks"].tolist(), [[1.0, 1.0, 0.0]])
        self.assertEqual(tuple(targets[1]["masks"].shape), (0, 3))

    def test_unified_noise_target_has_a_finite_hungarian_loss(self):
        labels = torch.tensor([[-1, -1, 0, 0, -100]])
        valid = torch.tensor([[True, True, True, True, False]])
        targets, _inverse = build_track_instance_targets(
            labels,
            valid,
            mode=UNIFIED_NOISE_INSTANCE,
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
        self.assertEqual(indices[0][0].numel(), 2)
        losses = compute_point_loss(outputs, targets, valid, matcher)
        for value in losses.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_signal_only_mode_remains_explicit_compatibility_behavior(self):
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

    def test_qualified_noise_query_is_canonicalized_without_hiding_signal(self):
        truth = np.array([-1, -1, 0, 0, 1, 1])
        pred = np.array([9, 9, 3, 3, 4, 4])
        signal = np.ones_like(truth, dtype=bool)
        canonical_pred, canonical_signal, attribution = canonicalize_noise_query(
            truth,
            pred,
            pred_signal_mask=signal,
            mode=NOISE_ATTRIBUTION_TRUTH_JOINT_HUNGARIAN_QUALIFIED,
        )
        self.assertTrue(attribution["noise_match_qualified"])
        self.assertEqual(attribution["noise_pred_id"], 9)
        self.assertEqual(canonical_pred.tolist(), [-1, -1, 3, 3, 4, 4])
        self.assertEqual(canonical_signal.tolist(), [False, False, True, True, True, True])
        native = event_track_metrics(truth, pred, pred_signal_mask=signal)
        canonical = event_track_metrics(
            truth, canonical_pred, pred_signal_mask=canonical_signal,
        )
        self.assertEqual(native["n_pred_tracks"], 3)
        self.assertEqual(canonical["n_pred_tracks"], 2)
        self.assertEqual(canonical["n_matched_tracks"], 2)
        self.assertNotIn(9, {match["pred_id"] for match in canonical["matches"]})
        self.assertAlmostEqual(canonical["background_rejection"], 1.0)
        self.assertAlmostEqual(canonical["signal_loss_to_background"], 0.0)

    def test_noise_query_signal_leakage_becomes_signal_loss(self):
        truth = np.array([-1, -1, 0, 0])
        pred = np.array([8, 8, 8, 7])
        signal = np.ones_like(truth, dtype=bool)
        canonical_pred, canonical_signal, attribution = canonicalize_noise_query(
            truth,
            pred,
            pred_signal_mask=signal,
            mode=NOISE_ATTRIBUTION_TRUTH_JOINT_HUNGARIAN_QUALIFIED,
        )
        self.assertTrue(attribution["noise_match_qualified"])
        canonical = event_track_metrics(
            truth, canonical_pred, pred_signal_mask=canonical_signal,
        )
        self.assertAlmostEqual(canonical["signal_loss_to_background"], 0.5)
        self.assertAlmostEqual(canonical["background_rejection"], 1.0)

    def test_ambiguous_noise_candidate_is_not_reclassified(self):
        truth = np.array([-1, -1, 0, 0, 0])
        pred = np.array([5, 5, 5, 5, 5])
        signal = np.ones_like(truth, dtype=bool)
        canonical_pred, canonical_signal, attribution = canonicalize_noise_query(
            truth,
            pred,
            pred_signal_mask=signal,
            mode=NOISE_ATTRIBUTION_TRUTH_JOINT_HUNGARIAN_QUALIFIED,
        )
        self.assertFalse(attribution["noise_match_qualified"])
        self.assertEqual(attribution["noise_attribution_status"], "no_joint_noise_candidate")
        self.assertEqual(canonical_pred.tolist(), pred.tolist())
        self.assertEqual(canonical_signal.tolist(), signal.tolist())

    def test_joint_noise_candidate_below_threshold_is_rejected(self):
        truth = np.array([-1, -1] + [0] * 10)
        pred = np.array([5, 5] + [5] * 3 + [6] * 7)
        signal = np.ones_like(truth, dtype=bool)
        canonical_pred, canonical_signal, attribution = canonicalize_noise_query(
            truth,
            pred,
            pred_signal_mask=signal,
            mode=NOISE_ATTRIBUTION_TRUTH_JOINT_HUNGARIAN_QUALIFIED,
        )
        self.assertEqual(attribution["noise_pred_id"], 5)
        self.assertFalse(attribution["noise_match_qualified"])
        self.assertEqual(attribution["noise_attribution_status"], "rejected_threshold")
        self.assertEqual(canonical_pred.tolist(), pred.tolist())
        self.assertEqual(canonical_signal.tolist(), signal.tolist())

    def test_native_noise_attribution_leaves_prediction_untouched(self):
        truth = np.array([-1, -1, 0, 0])
        pred = np.array([6, 6, 2, 2])
        signal = np.ones_like(truth, dtype=bool)
        canonical_pred, canonical_signal, attribution = canonicalize_noise_query(
            truth, pred, pred_signal_mask=signal, mode=NOISE_ATTRIBUTION_NATIVE,
        )
        self.assertEqual(canonical_pred.tolist(), pred.tolist())
        self.assertEqual(canonical_signal.tolist(), signal.tolist())
        self.assertEqual(attribution["noise_attribution_status"], "native")

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
