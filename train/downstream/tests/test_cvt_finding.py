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


class FindingHDF5Test(unittest.TestCase):
    setUp = fixtures.DownstreamHDF5Test.setUp
    tearDown = fixtures.DownstreamHDF5Test.tearDown

    def test_campaign_preflight_reads_all_hdf5_splits(self):
        from train.downstream.campaign.run_track_finding_campaign import preflight_dataset
        report = preflight_dataset(self.root, max_events=1, min_multitrack_fraction=0.)
        self.assertTrue(report["passed"])
        self.assertEqual(set(report["splits"]), {"train", "validation", "test"})

    def test_unassigned_clusters_remain_background_for_every_track(self):
        from train.downstream.track_finding_targets import build_track_instance_targets
        source = CVTDownstreamSource(self.root, "train")
        labels = torch.as_tensor(source.point_field(0, "seg_target")).unsqueeze(0)
        targets, _ = build_track_instance_targets(labels, torch.ones_like(labels, dtype=torch.bool), mode="signal_only")
        self.assertEqual(tuple(targets[0]["masks"].shape), (2, 8))
        self.assertEqual(float(targets[0]["masks"][:, [3, 6, 7]].sum()), 0.)
        source.close()
