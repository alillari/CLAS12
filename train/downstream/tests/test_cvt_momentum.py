from train.downstream.scripts.compute_regression_target_stats import compute_stats
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import torch
from fm4npp.datasets.cvt_downstream import CVTDownstreamSource
from train.downstream.tests.test_cvt_downstream import event, publish_fixture
from train.downstream.tests import test_cvt_downstream as fixtures


class MomentumHDF5Test(unittest.TestCase):
    setUp = fixtures.DownstreamHDF5Test.setUp
    tearDown = fixtures.DownstreamHDF5Test.tearDown

    def test_ambiguous_baselines_are_not_arbitrarily_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            publish_fixture(Path(tmp), {"test": [[event(clones=True)]]})
            source = CVTDownstreamSource(tmp, "test")
            self.assertEqual(source.benchmark(0, 3)["status"], "ambiguous")
            self.assertTrue(np.isnan(source.benchmark(0, 3)["values"]).all())
            self.assertEqual(source.benchmark(0, 3, 9)["status"], "matched")
            source.close()

    def test_invalid_components_and_entrance_do_not_become_targets(self):
        row = event()
        row["particles/observed_valid"][1, 0] = False
        row["reco/cvt_tracks/entrance_phi"][1] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            publish_fixture(Path(tmp), {"test": [[row]]})
            source = CVTDownstreamSource(tmp, "test")
            target = source.point_field(0, "reg_target")
            self.assertTrue(np.isnan(target[:3, 0]).all())
            np.testing.assert_array_equal(target[0, 1:3], [2., 3.])
            self.assertEqual(source.benchmark(0, 3)["status"], "invalid_entrance")
            source.close()

    def test_stats_are_track_weighted_and_training_only(self):
        stats = compute_stats(self.root, "train", 1, 100, 2, 10, task="mom",
                              segment_min_clusters=2, reader_type="hdf5")
        np.testing.assert_allclose(stats["mean"], [.6, 1.15, 1.7])
        self.assertEqual(stats["data_contract"]["momentum_unit"], "GeV/c")
        with self.assertRaisesRegex(ValueError, "training split"):
            compute_stats(self.root, "test", 1, 100, 2, 10, task="mom", reader_type="hdf5")

    def test_native_track_ids_are_distinct_from_rows(self):
        source = CVTDownstreamSource(self.root, "train")
        self.assertEqual(source.benchmark(0, 3)["track_row"], 1)
        self.assertEqual(source.benchmark(0, 3, 9)["status"], "matched")
        self.assertEqual(source.benchmark(0, 3, 2)["status"], "missing")
        source.close()
