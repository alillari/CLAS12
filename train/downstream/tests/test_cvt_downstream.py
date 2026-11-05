"""Named-field joins, downstream sample semantics and legacy parity."""
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from mmap_ninja import RaggedMmap
from torch.utils.data import DataLoader

from fm4npp.datasets.cvt_downstream import CVTDownstreamSource, downstream_split
from fm4npp.datasets.dataset import TPCBatchDataset, EventSegmentTPCBatchDataset, MyCollator
from train.pretrain.tests.test_cvt_hdf5 import publish_fixture


def event(ordinal=0, clones=False):
    # Segment labels, particle IDs, native track IDs and rows intentionally differ.
    n = 8
    c = {
        "xyz": np.column_stack((np.arange(n) + 7., np.zeros(n), np.arange(n))).astype("f4"),
        "primary_segment": np.array([3, 3, 3, -1, 8, 8, -1, -1]),
        "primary_tid": np.array([42, 42, 42, -1, 7, 7, -1, -1]),
        "ownership": np.array([1, 1, 1, 0, 1, 1, 2, 3]),
        "reco_segment": np.array([9, 9, 9, 9, 2, 2, -1, -1]),
        "native_track_id": np.array([25, 25, 25, 25, 11, 11, -1, -1]),
        "primary_pid": np.array([211, 211, 211, 0, -211, -211, 0, 0]),
        "primary_pid_valid": np.array([1, 1, 1, 0, 1, 1, 0, 0], dtype=bool),
        "endpoints": np.arange(n * 6).reshape(n, 6).astype("f4"),
        "direction": np.ones((n, 3), dtype="f4"), "length": np.ones(n), "pitch": np.ones(n),
    }
    c.update({k: np.arange(n) for k in ("detector", "layer", "sector", "cluster_id", "seed_strip")})
    p = {"tid": np.array([7, 42]), "observed_momentum": np.array([[.2, .3, .4], [1., 2., 3.]]),
         "observed_valid": np.ones((2, 3), dtype=bool), "vertex": np.zeros((2, 3)),
         "generated_energy": np.array([.6, 4.])}
    r = {"native": np.array([(11,), (25,)], dtype=[("ID", "i4")]),
         "purity_tid": np.array([42 if clones else 7, 42]),
         "purity_valid": np.ones(2, dtype=bool), "unique_owner": np.ones(2, dtype=bool),
         "purity": np.array([1., .75]), "benchmark_p": np.array([.54, 3.74]),
         "benchmark_theta": np.array([.7, .65]), "entrance_phi": np.array([.8, 1.1])}
    result = {"events/source_id": 17, "events/original_ordinal": ordinal}
    for domain, fields in (("clusters", c), ("particles", p), ("reco/cvt_tracks", r)):
        result.update({domain + "/" + k: v for k, v in fields.items()})
    return result


def dataset(root, cls=TPCBatchDataset, **kwargs):
    return cls(root, split="train", reader_type="hdf5", normalize=False,
               voxelize=False, return_dict=True, compute_neighbors=False,
               return_coatjava_seg_pred=True, **kwargs)


class DownstreamHDF5Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        publish_fixture(self.root, {s: [[event(i)]] for i, s in enumerate(("train", "validation", "test"))})

    def tearDown(self):
        self.tmp.cleanup()

    def test_explicit_joins_and_binary_policy(self):
        source = CVTDownstreamSource(self.root, "train")
        reg = source.point_field(0, "reg_target")
        np.testing.assert_array_equal(reg[0, :3], [1, 2, 3])
        np.testing.assert_allclose(reg[4, :3], [.2, .3, .4])
        self.assertTrue(np.isnan(reg[[3, 6, 7]]).all())
        np.testing.assert_array_equal(source.point_field(0, "noise_target"), [0, 0, 0, 1, 0, 0, 1, 1])
        self.assertEqual(pickle.loads(pickle.dumps(source)).identity(0), {"source_id": 17, "original_ordinal": 0})
        source.close()



    def test_invalid_owner_join_fails_instead_of_using_a_row_number(self):
        row = event()
        row["particles/tid"] = np.array([7, 99])
        with tempfile.TemporaryDirectory() as tmp:
            publish_fixture(Path(tmp), {"test": [[row]]})
            source = CVTDownstreamSource(tmp, "test")
            with self.assertRaisesRegex(ValueError, "absent from particles"):
                source.point_field(0, "reg_target")
            source.close()

    def test_full_event_and_segment_samples(self):
        full = dataset(self.root)
        item = full[0]
        self.assertEqual(len(item["noise_target"]), 8)
        self.assertEqual(int(item["noise_target"].sum()), 3)
        segments = dataset(self.root, EventSegmentTPCBatchDataset, segment_min_clusters=2)
        self.assertEqual(len(segments), 2)
        self.assertEqual(int(segments[0]["noise_target"].sum()), 0)
        for workers in (0, 2):
            batch = next(iter(DataLoader(full, batch_size=1, num_workers=workers, collate_fn=MyCollator())))
            self.assertEqual(batch["source_id"].tolist(), [17])
            self.assertEqual(batch["original_ordinal"].tolist(), [0])

    def test_same_tensor_contract_as_legacy_including_geometry(self):
        source = CVTDownstreamSource(self.root, "train")
        with tempfile.TemporaryDirectory() as tmp:
            for stem in source.fields:
                RaggedMmap.from_lists(Path(tmp) / f"{stem}_train", [source.point_field(0, stem)])
            options = dict(split="train", normalize=False, voxelize=False, return_dict=True,
                           return_coatjava_seg_pred=True, input_representation="clas12_pos_plus_aux_v1",
                           compute_neighbors=False)
            legacy = TPCBatchDataset(tmp, **options)[0]
            hdf5 = TPCBatchDataset(self.root, reader_type="hdf5", **options)[0]
            for key, value in legacy.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, hdf5[key], equal_nan=True)
        source.close()



    def test_default_splits(self):
        from types import SimpleNamespace
        self.assertEqual(downstream_split(SimpleNamespace(reader_type="hdf5"), "validation"), "validation")
        self.assertEqual(downstream_split(SimpleNamespace(), "validation"), "test")


if __name__ == "__main__":
    unittest.main()
