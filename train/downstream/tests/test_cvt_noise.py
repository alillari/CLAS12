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


class NoiseHDF5Test(unittest.TestCase):
    setUp = fixtures.DownstreamHDF5Test.setUp
    tearDown = fixtures.DownstreamHDF5Test.tearDown

    def test_update_budget_counts_optimizer_updates_only(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from train.downstream.point_classification_trainer import DownstreamTrainer
        trainer = DownstreamTrainer.__new__(DownstreamTrainer)
        trainer.params = SimpleNamespace(max_optimizer_steps=7, val_interval_steps=3, max_epochs=10)
        trainer.train_data_loader = [0, 1]
        trainer.startEpoch = 0
        trainer.best_loss = float("inf")
        trainer.down_results = {}
        trainer.model = Mock()
        trainer.down_model = Mock()
        trainer.down_scheduler = Mock()
        trainer._train_one_batch = Mock(return_value=.5)
        trainer.validate_end_to_end_one_epoch = Mock(side_effect=[.4, .3, .2])
        trainer._save_checkpoint = Mock()
        trainer._train_by_optimizer_step(False, "checkpoint", str(self.root / "training.log"))
        self.assertEqual(trainer.global_step, 7)
        self.assertEqual(trainer._train_one_batch.call_count, 7)
        self.assertEqual(trainer.down_scheduler.step.call_count, 7)
        self.assertEqual(trainer.validate_end_to_end_one_epoch.call_count, 3)
        self.assertEqual(trainer._save_checkpoint.call_count, 3)
