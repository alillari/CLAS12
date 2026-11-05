"""Backbone and adapter loading must reject mismatched CLAS12 serialization."""
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.downstream.point_classification_trainer import DownstreamTrainer as Classifier
from train.downstream.track_finding_trainer import DownstreamTrainer as Finder
from train.downstream.track_regression_trainer import DownstreamTrainer as Regressor


class HilbertCheckpointTest(unittest.TestCase):
    def test_backbone_load_uses_matching_grid_and_restores_historical_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'backbone.pt'
            state = torch.nn.Linear(3, 3).state_dict()
            for cls in (Classifier, Finder, Regressor):
                trainer = cls.__new__(cls)
                trainer.params = SimpleNamespace()
                trainer.device = 'cpu'
                trainer.world_rank = 1
                trainer.model = torch.nn.Linear(3, 3)
                for grid in ('unit_v2', 'legacy_v1'):
                    trainer.params.hilbert_angular_grid = grid
                    torch.save({'hilbert_angular_grid': grid, 'model_state': state}, path)
                    trainer.restore_checkpoint(path, load_optimizer_state=False)
                    torch.testing.assert_close(trainer.model.weight, state['weight'])
                torch.save({'model_state': state}, path)
                trainer.restore_checkpoint(path, load_optimizer_state=False)
                trainer.params.hilbert_angular_grid = 'unit_v2'
                with self.assertRaisesRegex(ValueError, 'Checkpoint Hilbert grid'):
                    trainer.restore_checkpoint(path, load_optimizer_state=False)

    def test_adapter_load_rejects_cross_grid_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'adapter.pt'
            for cls in (Classifier, Finder, Regressor):
                trainer = cls.__new__(cls)
                trainer.params = SimpleNamespace(hilbert_angular_grid='unit_v2')
                trainer.device = 'cpu'
                for metadata in ({}, {'hilbert_angular_grid': 'legacy_v1'}):
                    torch.save(metadata, path)
                    with self.assertRaisesRegex(ValueError, 'Checkpoint Hilbert grid'):
                        trainer.load_checkpoint(path, inference=True)
