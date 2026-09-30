import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from mmap_ninja import RaggedMmap

from fm4npp.datasets.dataset import (
    EventContextSegmentTPCBatchDataset, EventSegmentTPCBatchDataset,
    MyCollator, TPCBatchDataset, resolve_adapter_sample_mode,
)
from train.downstream.event_context import (
    backbone_features, validate_event_context_config, validate_checkpoint_sample_mode,
)
from train.downstream.scripts.compute_regression_target_stats import compute_stats


class CausalBackbone(torch.nn.Module):
    """Context-sensitive CPU model: exposes alignment mistakes without CUDA."""
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.last_shape = None

    def forward(self, points, return_z=False):
        self.last_shape = points.shape
        x = points.masked_fill(points == -100, 0) * self.scale
        return None, [x, x.cumsum(1)], x


class EventContextTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.features = [
            np.array([[20, 0, 1], [8, 1, 2], [10, 0, 3], [8, 1, 2],
                      [15, 1, 1], [7, 0, 0], [22, 0, 2]], dtype=np.float32),
            np.array([[17, 0, 2], [8, 1, 1], [10, 0, 0], [12, 1, 1],
                      [14, 0, 2], [16, 0, 1]], dtype=np.float32),
        ]
        self.labels = [np.array([0, -1, 1, 0, 1, -1, 1]), np.array([0, 0, 1, 1, 0, -1])]
        regression = []
        for labels in self.labels:
            values = np.zeros((len(labels), 7), dtype=np.float32)
            values[:, :3] = np.where(labels[:, None] == 0, [1, 2, 3], [4, 5, 6])
            values[labels == -1] = np.nan
            regression.append(values)
        for stem, arrays in [('features', self.features), ('seg_target', self.labels),
                             ('reg_target', regression)]:
            RaggedMmap.from_lists(self.root / f'{stem}_pretrain', arrays)
        self.kwargs = dict(data_root=str(self.root), split='pretrain',
                           return_dict=True, require_reg_target=True, num_pred_points=1,
                           segment_min_clusters=2, high_thr=3, voxelize=False,
                           limit_data=True, limit_size=3)

    def views(self, **overrides):
        kwargs = dict(self.kwargs, **overrides)
        return EventSegmentTPCBatchDataset(**kwargs), EventContextSegmentTPCBatchDataset(**kwargs)

    def test_identical_track_cohort_targets_order_and_full_event_context(self):
        baseline, context = self.views()
        self.assertEqual(context.idxlist, baseline.idxlist)
        self.assertEqual(len(context), 3)  # a partial last event still counts tracks
        full = TPCBatchDataset(data_root=str(self.root), split='pretrain',
            return_dict=True, num_pred_points=1, voxelize=False, high_thr=100)
        for index in range(len(context)):
            a, b = baseline[index], context[index]
            for key in a:
                if isinstance(a[key], torch.Tensor):
                    torch.testing.assert_close(a[key], b[key], equal_nan=True)
                else:
                    self.assertEqual(a[key], b[key])
            event_index = b['source_event_index']
            torch.testing.assert_close(b['backbone_points'], full[event_index]['points'])
            torch.testing.assert_close(b['backbone_points'][b['backbone_token_index']], b['points'], rtol=0, atol=0)
            self.assertEqual(len(b['backbone_points']), len(self.features[event_index]))

    def test_dedup_gather_padding_and_context_effect(self):
        baseline, context = self.views()
        batch = MyCollator()([context[i] for i in range(3)])
        self.assertEqual(batch['backbone_points'].shape, (2, 7, 3))
        self.assertEqual(batch['backbone_event_index'].tolist(), [0, 0, 1])
        model = CausalBackbone().eval()
        with torch.no_grad():
            actual = backbone_features(model, batch['points'], batch, 'event_segment_context')
            self.assertEqual(model.last_shape, torch.Size([2, 7, 3]))
            _, full_layers, _ = model(batch['backbone_points'], return_z=True)
            for i in range(3):
                n = len(context[i]['points'])
                for layer in range(2):
                    expected = full_layers[layer][batch['backbone_event_index'][i],
                                                  batch['backbone_token_index'][i, :n]]
                    torch.testing.assert_close(actual[layer, i, :n], expected)
            isolated_batch = MyCollator()([baseline[i] for i in range(3)])
            isolated = backbone_features(model, isolated_batch['points'], isolated_batch)
        valid = batch['points'][..., 0] != -100
        torch.testing.assert_close(actual[0][valid], isolated[0][valid])
        self.assertGreater((actual[1][valid] - isolated[1][valid]).abs().max().item(), .1)
        self.assertTrue((actual[:, ~valid] == 0).all())
        self.assertFalse(actual.requires_grad)

    def test_shuffled_track_batch_keeps_identity(self):
        _, context = self.views()
        batch = MyCollator()([context[i] for i in [2, 0, 1]])
        self.assertEqual(batch['backbone_event_index'].tolist(), [0, 1, 1])
        self.assertEqual(batch['source_event_index'].tolist(), [1, 0, 0])
        with torch.no_grad():
            backbone_features(CausalBackbone().eval(), batch['points'], batch, 'event_segment_context')

    def test_coatjava_gathers_candidate_but_targets_dominant_truth(self):
        candidates = [np.array([7, -1, 7, 7, 8, -1, 8]), self.labels[1]]
        RaggedMmap.from_lists(self.root / 'coatjava_seg_pred_pretrain', candidates)
        baseline, context = self.views(segment_target_source='coatjava', segment_min_truth_purity=.6)
        self.assertEqual(context.idxlist, baseline.idxlist)
        a, b = baseline[0], context[0]
        self.assertEqual(b['segment_label'], 7)
        self.assertEqual(b['truth_segment_label'], 0)
        self.assertEqual(int(b['target_segment_mask'].sum()), 2)
        torch.testing.assert_close(a['target_segment_mask'], b['target_segment_mask'])
        torch.testing.assert_close(b['backbone_points'][b['backbone_token_index']], b['points'])

    def test_training_step_and_validation_use_event_features(self):
        from train.downstream.eval.evaluate_track_regression import DownstreamTrainer
        from train.downstream.model import RegressionTargetNormalizer

        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(3, 3)
                self.target_normalizer = RegressionTargetNormalizer(3, mean=[0]*3, std=[1]*3)

            def forward(self, points, feature=None, padding_mask=None, **kwargs):
                x = feature.mean(0)
                mask = padding_mask[..., None]
                pooled = (x * mask).sum(1) / mask.sum(1)
                return {'pred_regression': self.projection(pooled)}

        _, context = self.views()
        batch = MyCollator()([context[i] for i in range(3)])
        trainer = DownstreamTrainer.__new__(DownstreamTrainer)
        trainer.params = SimpleNamespace(adapter_sample_mode='event_segment_context',
                                        task='mom', max_val_batches=None)
        trainer.device = 'cpu'
        trainer.model = CausalBackbone().eval()
        trainer.down_model = Head()
        trainer.down_optimizer = torch.optim.SGD(trainer.down_model.parameters(), lr=.01)
        trainer.regression_target_stats = dict(task='mom', angular_indices=[], mean=[0]*3, std=[1]*3)
        trainer.regression_loss = 'mae'
        trainer.regression_loss_reference = None
        trainer.grad_clip_value = 1.
        trainer.physics_config = None
        trainer.log_to_screen = False
        trainer.down_results = {'val': []}
        trainer.val_data_loader = [batch]
        before = trainer.down_model.projection.weight.detach().clone()
        loss = trainer._train_one_batch(batch, pretrain=True)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, trainer.down_model.projection.weight))
        self.assertIsNone(trainer.model.scale.grad)
        self.assertEqual(trainer.model.last_shape, torch.Size([2, 7, 3]))
        self.assertTrue(np.isfinite(trainer.validate_end_to_end_one_epoch(pretrain=True)))
        self.assertEqual(trainer.model.last_shape, torch.Size([2, 7, 3]))

    def test_misalignment_and_wrong_mode_fail(self):
        _, context = self.views()
        batch = MyCollator()([context[0]])
        model = CausalBackbone().eval()
        with self.assertRaisesRegex(ValueError, 'sample_mode'):
            backbone_features(model, batch['points'], batch)
        batch['backbone_token_index'][0, 0] = (batch['backbone_token_index'][0, 0] + 1) % 7
        with self.assertRaisesRegex(ValueError, 'reproduce'):
            backbone_features(model, batch['points'], batch, 'event_segment_context')

    def test_existing_target_statistics_are_identical(self):
        kwargs = dict(low_thr=1, high_thr=3, limit_size=3, chunk_size=10,
                      task='mom', segment_min_clusters=2)
        a = compute_stats(self.root, 'pretrain', adapter_sample_mode='event_segment', **kwargs)
        b = compute_stats(self.root, 'pretrain', adapter_sample_mode='event_segment_context', **kwargs)
        for key in ('mean', 'std', 'count', 'selected_events'):
            np.testing.assert_equal(a[key], b[key])

    def test_registry_and_unsupported_configs(self):
        mode, cls, kwargs = resolve_adapter_sample_mode(SimpleNamespace(
            adapter_sample_mode='event_segment_context', segment_min_clusters=2))
        self.assertIs(cls, EventContextSegmentTPCBatchDataset)
        self.assertEqual(kwargs['segment_min_clusters'], 2)
        for override in [dict(chunk_training=True), dict(return_dict=False),
                         dict(input_representation='clas12_geometry_v1')]:
            with self.assertRaises(ValueError):
                cls(**dict(self.kwargs, **override))
        with self.assertRaisesRegex(ValueError, 'pretrained_ckpt'):
            validate_event_context_config(SimpleNamespace(adapter_sample_mode=mode))
        validate_event_context_config(SimpleNamespace(adapter_sample_mode=mode, pretrained_ckpt='m6.tar'))

    def test_checkpoint_context_must_match(self):
        validate_checkpoint_sample_mode({}, 'event_segment')
        validate_checkpoint_sample_mode({'adapter_sample_mode': 'event_segment_context'}, 'event_segment_context')
        with self.assertRaisesRegex(ValueError, 'context differs'):
            validate_checkpoint_sample_mode({}, 'event_segment_context')
        with self.assertRaisesRegex(ValueError, 'context differs'):
            validate_checkpoint_sample_mode({'adapter_sample_mode': 'event_segment_context'}, 'event_segment')

    def test_matched_m6_configs_only_change_context(self):
        path = Path(__file__).resolve().parents[3] / 'scripts/configs/mamba_clas12_track_regression_pretrained.yaml'
        configs = yaml.safe_load(path.read_text())
        track = configs['clas12_track_regression_m6_track_context']
        event = configs['clas12_track_regression_m6_event_context']
        self.assertEqual(track['embed_dim'], 1536)
        self.assertEqual({k for k in track if track[k] != event[k]}, {'adapter_sample_mode'})


if __name__ == '__main__':
    unittest.main()
