import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train.downstream.eval.evaluate_track_regression import (
    adapter_to_cvt_resolution_ratio,
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

    def test_missing_cvt_momentum_metrics_have_no_ratio(self):
        methods = {
            "adapter": {"momentum": {"relative_resolution_68": 0.12}},
            "cvt": {"n": 0},
        }
        self.assertIsNone(adapter_to_cvt_resolution_ratio(methods))

    def test_available_cvt_momentum_metrics_compute_ratio(self):
        methods = {
            "adapter": {"momentum": {"relative_resolution_68": 0.12}},
            "cvt": {"momentum": {"relative_resolution_68": 0.24}},
        }
        self.assertEqual(adapter_to_cvt_resolution_ratio(methods), 0.5)


if __name__ == "__main__":
    unittest.main()
