import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from mmap_ninja import RaggedMmap

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fm4npp.datasets.dataset import (
    EventSegmentTPCBatchDataset,
    MyCollator,
    TPCBatchDataset,
    resolve_adapter_sample_mode,
)
from train.downstream.scripts.compute_regression_target_stats import compute_stats


def write_ragged(root, name, arrays):
    RaggedMmap.from_lists(root / name, arrays)


class EventSegmentDatasetTest(unittest.TestCase):
    def test_adapter_sample_mode_registry(self):
        class Params:
            adapter_sample_mode = "track_legacy"

        mode, dataset_cls, kwargs = resolve_adapter_sample_mode(Params())
        self.assertEqual(mode, "track_legacy")
        self.assertIs(dataset_cls, TPCBatchDataset)
        self.assertEqual(kwargs, {})

        Params.adapter_sample_mode = "event_segment"
        Params.segment_min_clusters = 6
        Params.segment_exact_clusters = True
        mode, dataset_cls, kwargs = resolve_adapter_sample_mode(Params())
        self.assertEqual(mode, "event_segment")
        self.assertIs(dataset_cls, EventSegmentTPCBatchDataset)
        self.assertEqual(kwargs["segment_min_clusters"], 6)
        self.assertTrue(kwargs["segment_exact_clusters"])

        Params.adapter_sample_mode = "missing_mode"
        with self.assertRaisesRegex(ValueError, "Unknown adapter_sample_mode"):
            resolve_adapter_sample_mode(Params())

    def test_event_is_split_into_segment_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            features = np.array(
                [
                    [6.0, 0.0, 0.0],
                    [7.0, 0.0, 0.0],
                    [8.0, 0.0, 0.0],
                    [9.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
            seg = np.array([-1, 0, 0, 1, 1], dtype=np.int64)
            reg = np.array(
                [
                    [np.nan] * 10,
                    [1.0, 2.0, 3.0, 0.0, 0.0, -2.0, 4.0, np.nan, np.nan, np.nan],
                    [1.0, 2.0, 3.0, 0.0, 0.0, -2.0, 4.0, np.nan, np.nan, np.nan],
                    [10.0, 20.0, 30.0, 0.0, 0.0, -3.0, 40.0, np.nan, np.nan, np.nan],
                    [10.0, 20.0, 30.0, 0.0, 0.0, -3.0, 40.0, np.nan, np.nan, np.nan],
                ],
                dtype=np.float32,
            )
            pid = np.array([-1, 0, 0, 2, 2], dtype=np.int64)
            noise = np.array([1, 0, 0, 0, 0], dtype=np.int64)

            write_ragged(root, "features_pretrain", [features])
            write_ragged(root, "seg_target_pretrain", [seg])
            write_ragged(root, "reg_target_pretrain", [reg])
            write_ragged(root, "pid_target_pretrain", [pid])
            write_ragged(root, "noise_target_pretrain", [noise])

            dataset = EventSegmentTPCBatchDataset(
                data_root=str(root),
                split="pretrain",
                train=True,
                return_dict=True,
                normalize=False,
                serialization="radius",
                num_pred_points=1,
                segment_min_clusters=2,
                segment_exact_clusters=True,
                require_reg_target=True,
                require_pid_target=True,
                require_noise_target=True,
            )

            self.assertEqual(len(dataset), 2)
            first = dataset[0]
            second = dataset[1]

            self.assertEqual(first["source_event_index"], 0)
            self.assertEqual(first["segment_label"], 0)
            self.assertEqual(second["segment_label"], 1)
            self.assertEqual(first["points"].shape[0], 2)
            self.assertEqual(second["points"].shape[0], 2)
            self.assertTrue(first["target_segment_mask"].all().item())
            self.assertTrue(np.all(first["target"].numpy() == 0))
            self.assertTrue(np.all(first["noise_target"].numpy() == 0))
            np.testing.assert_allclose(first["reg_target"].numpy()[:, :3], [[1.0, 2.0, 3.0]] * 2)
            np.testing.assert_allclose(second["reg_target"].numpy()[:, :3], [[10.0, 20.0, 30.0]] * 2)

            stats = compute_stats(
                root,
                "pretrain",
                low_thr=1,
                high_thr=100,
                limit_size=None,
                chunk_size=10,
                task="mom",
                adapter_sample_mode="event_segment",
                segment_min_clusters=2,
                segment_exact_clusters=True,
            )
            self.assertEqual(stats["selected_events"], 2)
            np.testing.assert_allclose(stats["mean"], [5.5, 11.0, 16.5])
            np.testing.assert_allclose(stats["std"], [4.5, 9.0, 13.5])

    def test_collator_pads_segment_metadata(self):
        batch = [
            {
                "points": np_to_tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
                "target": np_to_tensor([0, 0], dtype=np.int64),
                "knearest_points": np_to_tensor([[0.0, 0.0, 2.0], [-100.0, -100.0, -100.0]]),
                "reg_target": np_to_tensor([[1.0] * 10, [1.0] * 10]),
                "pid_target": np_to_tensor([0, 0], dtype=np.int64),
                "noise_target": np_to_tensor([0, 0], dtype=np.int64),
                "target_segment_mask": np_to_tensor([True, True], dtype=np.bool_),
                "source_event_index": 4,
                "segment_label": 2,
            },
            {
                "points": np_to_tensor([[0.0, 0.0, 3.0]]),
                "target": np_to_tensor([0], dtype=np.int64),
                "knearest_points": np_to_tensor([[-100.0, -100.0, -100.0]]),
                "reg_target": np_to_tensor([[2.0] * 10]),
                "pid_target": np_to_tensor([1], dtype=np.int64),
                "noise_target": np_to_tensor([0], dtype=np.int64),
                "target_segment_mask": np_to_tensor([True], dtype=np.bool_),
                "source_event_index": 5,
                "segment_label": 3,
            },
        ]
        out = MyCollator().collate_dict(batch)
        self.assertEqual(tuple(out["target_segment_mask"].shape), (2, 2))
        self.assertFalse(out["target_segment_mask"][1, 1].item())
        self.assertEqual(out["source_event_index"].tolist(), [4, 5])
        self.assertEqual(out["segment_label"].tolist(), [2, 3])


def np_to_tensor(values, dtype=np.float32):
    import torch

    return torch.as_tensor(np.asarray(values, dtype=dtype))


if __name__ == "__main__":
    unittest.main()
