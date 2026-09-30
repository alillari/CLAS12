"""Contracts for full-event, track-membership-conditioned momentum inputs."""
import unittest
from types import SimpleNamespace
import torch

from train.downstream.tests import test_momentum_event_context as fixtures
CausalBackbone = fixtures.CausalBackbone
from train.downstream.event_context import (
    EVENT_MEMBERSHIP_MODE, adapter_sequence, backbone_features,
    validate_checkpoint_sample_mode, validate_event_context_config,
)
from train.downstream.model import MambaTrackRegressionHead
from fm4npp.datasets.dataset import MyCollator, resolve_adapter_sample_mode


class MembershipTest(unittest.TestCase):
    setUp = fixtures.EventContextTest.setUp
    views = fixtures.EventContextTest.views

    def inputs(self):
        _, dataset = self.views()
        batch = MyCollator()([dataset[i] for i in (2, 0, 1)])
        model = CausalBackbone().eval()
        with torch.no_grad():
            features = backbone_features(model, batch['points'], batch, EVENT_MEMBERSHIP_MODE)
        points, kwargs = adapter_sequence(batch['points'], batch, EVENT_MEMBERSHIP_MODE)
        return model, batch, features, points, kwargs

    def head(self):
        # Actual projection, layer mixing, normalization, pooling and output head;
        # CUDA-only sequence layers are omitted for this focused CPU test.
        torch.manual_seed(42)
        return MambaTrackRegressionHead(input_dim=3, embed_dim=16, num_layers=0,
            num_embedder_layers=0, num_feature_layers=2, num_output_dim=3,
            pooling='attention', embed_method='pos_only', return_embedding=True,
            track_membership_channel=True)

    def test_full_events_and_membership_align_with_row_mapping(self):
        model, batch, features, points, kwargs = self.inputs()
        self.assertEqual(tuple(features.shape), (2, 3, 7, 3))
        self.assertEqual(tuple(model.last_shape), (2, 7, 3))  # deduplicated
        self.assertEqual(kwargs['padding_mask'].sum(1).tolist(), [6, 7, 7])
        self.assertEqual(kwargs['track_membership'].sum(1).tolist(), [3, 2, 3])
        torch.testing.assert_close(features[:, 1], features[:, 2])  # same event
        self.assertFalse(torch.equal(kwargs['track_membership'][1], kwargs['track_membership'][2]))
        for i in range(3):
            indices = batch['backbone_token_index'][i][batch['points'][i, :, 0] != -100]
            self.assertTrue(kwargs['track_membership'][i, indices].all())
            self.assertEqual(int(kwargs['track_membership'][i].sum()), len(indices))
        # Other tracks AND noise remain in adapter input and pooling support.
        self.assertTrue((kwargs['padding_mask'] & ~kwargs['track_membership']).any())
        self.assertFalse(kwargs['track_membership'][~kwargs['padding_mask']].any())

    def test_projection_membership_gradient_and_context_influence(self):
        model, batch, features, points, kwargs = self.inputs()
        head = self.head()
        self.assertEqual(head.input_proj[1].in_features, 4)
        output = head(points, features, pretrain=True, **kwargs)
        torch.testing.assert_close(output['embedding_pre_projection'][..., -1],
                                   kwargs['track_membership'].float())
        self.assertFalse(torch.allclose(output['pred_regression'][1], output['pred_regression'][2]))
        output['pred_regression'].square().sum().backward()
        self.assertGreater(head.input_proj[1].weight.grad[:, -1].abs().sum().item(), 0)
        self.assertIsNone(model.scale.grad)
        changed = features.clone()
        nonmember = kwargs['padding_mask'] & ~kwargs['track_membership']
        changed[..., 0] += nonmember.unsqueeze(0) * 10
        other = head(points, changed, pretrain=True, **kwargs)['pred_regression']
        self.assertFalse(torch.allclose(output['pred_regression'], other))
        padded = features.clone()
        padded[:, ~kwargs['padding_mask']] = 999
        actual = head(points, padded, pretrain=True, **kwargs)['pred_regression']
        torch.testing.assert_close(actual, output['pred_regression'])

    def test_adapter_only_membership_uses_full_event_and_membership(self):
        _, batch, _, points, kwargs = self.inputs()
        head = self.head()
        output = head(points, feature=None, pretrain=False, **kwargs)
        self.assertEqual(head.input_proj[1].in_features, 4)
        torch.testing.assert_close(output['embedding_pre_projection'][..., -1],
                                   kwargs['track_membership'].float())
        self.assertFalse(torch.allclose(output['pred_regression'][1], output['pred_regression'][2]))
        output['pred_regression'].square().sum().backward()
        self.assertGreater(head.input_proj[1].weight.grad[:, -1].abs().sum().item(), 0)
        # Gather mode without a backbone receives exactly the baseline input.
        gather_points, gather_kwargs = adapter_sequence(
            batch['points'], batch, 'event_segment_context')
        track_points, track_kwargs = adapter_sequence(
            batch['points'], batch, 'event_segment')
        torch.testing.assert_close(gather_points, track_points)
        torch.testing.assert_close(gather_kwargs['padding_mask'], track_kwargs['padding_mask'])

    def test_membership_validation_and_checkpoint_guards(self):
        _, _, features, points, kwargs = self.inputs()
        head = self.head()
        with self.assertRaisesRegex(ValueError, 'membership'):
            head(points, features, pretrain=True, padding_mask=kwargs['padding_mask'])
        bad = dict(kwargs, track_membership=torch.full_like(kwargs['track_membership'], 0.5, dtype=torch.float))
        with self.assertRaisesRegex(ValueError, 'binary'):
            head(points, features, pretrain=True, **bad)
        for saved in ('event_segment', 'event_segment_context'):
            with self.assertRaisesRegex(ValueError, 'differs'):
                validate_checkpoint_sample_mode({'adapter_sample_mode': saved}, EVENT_MEMBERSHIP_MODE)
        validate_checkpoint_sample_mode({'adapter_sample_mode': EVENT_MEMBERSHIP_MODE}, EVENT_MEMBERSHIP_MODE)
        validate_event_context_config(SimpleNamespace(adapter_sample_mode=EVENT_MEMBERSHIP_MODE))
        mode, cls, _ = resolve_adapter_sample_mode(SimpleNamespace(adapter_sample_mode=EVENT_MEMBERSHIP_MODE))
        self.assertEqual(mode, EVENT_MEMBERSHIP_MODE)

    def test_preset_and_evaluator_construct_widened_head(self):
        from pathlib import Path
        import yaml
        from train.downstream.eval.evaluate_track_regression import build_head
        root = Path(__file__).resolve().parents[3]
        configs = yaml.safe_load((root / 'scripts/configs/mamba_clas12_track_regression_pretrained.yaml').read_text())
        config = configs['clas12_track_regression_m6_event_membership']
        self.assertEqual(config['adapter_sample_mode'], EVENT_MEMBERSHIP_MODE)
        self.assertEqual(config['embed_dim'], 1536)
        params = SimpleNamespace(**dict(config, num_output_classes=4, num_embedder_layers=0))
        trainer = SimpleNamespace(params=params, device='cpu',
            regression_target_stats=dict(mean=[0]*4, std=[1]*4))
        head = build_head(trainer)
        self.assertEqual(head.input_proj[1].in_features, 1537)
        self.assertEqual(head.input_proj[1].out_features, 256)

    def test_training_and_validation_keep_track_targets(self):
        from train.downstream.track_regression_trainer import DownstreamTrainer
        model, batch, _, _, _ = self.inputs()
        trainer = DownstreamTrainer.__new__(DownstreamTrainer)
        trainer.params = SimpleNamespace(adapter_sample_mode=EVENT_MEMBERSHIP_MODE, task='mom', max_val_batches=None)
        trainer.device = 'cpu'
        trainer.model = model
        trainer.down_model = self.head()
        trainer.down_optimizer = torch.optim.SGD(trainer.down_model.parameters(), lr=.01)
        trainer.regression_target_stats = dict(task='mom', angular_indices=[], mean=[0]*3, std=[1]*3)
        trainer.regression_loss = 'mae'
        trainer.regression_loss_reference = None
        trainer.grad_clip_value = 1.
        trainer.physics_config = None
        trainer.log_to_screen = False
        trainer.down_results = {'val': []}
        trainer.val_data_loader = [batch]
        before = trainer.down_model.input_proj[1].weight.detach().clone()
        loss = trainer._train_one_batch(batch, pretrain=True)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, trainer.down_model.input_proj[1].weight))
        self.assertTrue(torch.isfinite(torch.tensor(trainer.validate_end_to_end_one_epoch(pretrain=True))))
        self.assertIsNone(model.scale.grad)


if __name__ == '__main__':
    unittest.main()
