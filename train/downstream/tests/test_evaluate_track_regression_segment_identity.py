import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train.downstream.eval.evaluate_track_regression import (
    matched_resolution_comparison,
    calculate_metrics,
    build_campaign_headline_rows,
    attach_sample_metadata,
    aux_row_for_sample,
    resolve_charge,
    resolve_evaluation_sample_identity,
)


class EvaluationSampleIdentityTest(unittest.TestCase):
    def test_legacy_idxlist_identity_is_preserved(self):
        class Dataset:
            idxlist = [11, 12]

        real_index, segment_label = resolve_evaluation_sample_identity(
            {}, Dataset(), dataset_index=1, local_index=0
        )
        self.assertEqual(real_index, 12)
        self.assertIsNone(segment_label)

    def test_batch_segment_identity_wins_over_tuple_idxlist(self):
        class Dataset:
            idxlist = [(99, 7)]

        batch = {
            "source_event_index": torch.as_tensor([4]),
            "segment_label": torch.as_tensor([2]),
        }
        real_index, segment_label = resolve_evaluation_sample_identity(
            batch, Dataset(), dataset_index=0, local_index=0
        )
        self.assertEqual(real_index, 4)
        self.assertEqual(segment_label, 2)

    def test_tuple_idxlist_segment_identity_fallback(self):
        class Dataset:
            idxlist = [(4, 2)]

        real_index, segment_label = resolve_evaluation_sample_identity(
            {}, Dataset(), dataset_index=0, local_index=0
        )
        self.assertEqual(real_index, 4)
        self.assertEqual(segment_label, 2)

    def test_aux_row_is_filtered_to_segment(self):
        class Aux:
            def __getitem__(self, index):
                if index != 0:
                    raise AssertionError(index)
                return np.asarray([
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                ])

        class Dataset:
            def _segment_source_for_filtering(self):
                return [np.asarray([0, 1, 1])]

        row = aux_row_for_sample(Dataset(), Aux(), real_index=0, segment_label=1)
        np.testing.assert_allclose(row, [2.0, 20.0])

    def test_coatjava_aux_row_uses_resolved_truth_segment(self):
        class Aux:
            def __getitem__(self, index):
                return np.asarray([[1.0], [20.0], [30.0]])

        class Dataset:
            memmap_seg_target = [np.asarray([0, 1, 1])]

            def _segment_source_for_filtering(self):
                return [np.asarray([7, 7, 7])]

        row = aux_row_for_sample(
            Dataset(), Aux(), real_index=0, segment_label=7, truth_segment_label=1
        )
        np.testing.assert_allclose(row, [20.0])

    def test_sample_metadata_attaches_to_duplicate_event_records(self):
        records = [
            {"real_index": 0, "segment_label": 0},
            {"real_index": 0, "segment_label": 1},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples_test.jsonl"
            path.write_text(
                json.dumps({
                    "source_file": "input.hipo",
                    "event": 123,
                    "trkid": 5,
                    "truth_tid": 9,
                    "pid": 211,
                }) + "\n"
            )
            attach_sample_metadata(records, path)

        self.assertEqual(records[0]["source_file"], "input.hipo")
        self.assertEqual(records[1]["source_file"], "input.hipo")
        self.assertEqual(records[0]["event"], 123)
        self.assertEqual(records[1]["truth_track_id"], 9)

    def test_missing_metadata_pid_uses_charge_fallback(self):
        records = [{"real_index": 0, "segment_label": 0}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples_test.jsonl"
            path.write_text(json.dumps({"source_file": "event.hipo", "event": 7}) + "\n")
            attach_sample_metadata(records, path)

        self.assertEqual(records[0]["pdg"], "")
        self.assertEqual(records[0]["charge"], "")
        charges, summary = resolve_charge(records, {"charge_source": "metadata_or_positive"})
        self.assertEqual(charges.tolist(), [1])
        self.assertEqual(summary["records_missing_metadata_charge"], 1)

    def test_resolution_ratio_compares_only_shared_finite_residuals(self):
        # Rows 2 and 3 are valid for only one method. Row 4 has zero truth
        # momentum and row 5 has missing truth.
        truth = np.zeros((6, 3))
        truth[:, 0] = [1, 1, 1, 1, 0, np.nan]
        adapter = np.zeros_like(truth)
        cvt = np.zeros_like(truth)
        adapter[:, 0] = [.9, 1.1, 12, np.nan, 1, 1]
        cvt[:, 0] = [.8, 1.2, np.nan, 7, 1, 1]
        predictions = {"adapter": adapter, "cvt": cvt}
        comparison = matched_resolution_comparison(truth, predictions)
        self.assertEqual(comparison["adapter_cvt_matched_tracks"], 2)
        self.assertAlmostEqual(comparison["adapter_to_cvt_resolution_ratio"], .5)
        summary = {"methods": {k: calculate_metrics(truth, p, [.1]) for k, p in predictions.items()},
                   **comparison}
        self.assertEqual([m["n"] for m in summary["methods"].values()], [4, 4])
        row = next(row for row in build_campaign_headline_rows({}, summary, [])
                   if row["record_type"] == "comparison_summary")
        self.assertEqual(row["adapter_cvt_matched_tracks"], 2)
        self.assertAlmostEqual(row["adapter_to_cvt_resolution_ratio"], .5)

    def test_undefined_resolution_ratio_still_reports_matched_count(self):
        truth = np.ones((3, 3))
        for cvt, expected_count in [(np.full_like(truth, np.nan), 0), (truth, 3)]:
            comparison = matched_resolution_comparison(truth, {"adapter": truth, "cvt": cvt})
            self.assertIsNone(comparison["adapter_to_cvt_resolution_ratio"])
            self.assertEqual(comparison["adapter_cvt_matched_tracks"], expected_count)


if __name__ == "__main__":
    unittest.main()
