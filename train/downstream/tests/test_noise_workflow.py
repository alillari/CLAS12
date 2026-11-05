"""Noise mode safety, CLI reproducibility and shared evaluation contracts."""
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from fm4npp.utils import YParams
from train.downstream.noise_checkpoint import (
    checkpoint_config, require_single_process, seed_everything, training_contract, validate_noise_checkpoint,
    restore_evaluation_config, validate_noise_data,
)
from train.downstream.train_point_classification import parse_args, configure_training

ROOT = Path(__file__).resolve().parents[3]


class NoiseWorkflowTest(unittest.TestCase):
    def test_cli_preserves_yaml_task_and_synchronizes_overrides(self):
        args = parse_args(['--yaml_config', 'unused', '--config', 'noise'])
        params = YParams(str(ROOT / 'scripts/configs/clas12_adapter_recipes.yaml'), 'hdf5_noise_50k_balanced')
        configure_training(params, args)
        self.assertEqual((params.task, params.limit_size, params.batch_size, params.seed), ('nid', 50000, 32, 42))
        args.task, args.eventnumber, args.train_batch_size, args.seed = 'nid', 128, 16, 71
        configure_training(params, args)
        for key, value in dict(task='nid', num_output_classes=2, limit_size=128, batch_size=16, seed=71).items():
            self.assertEqual(getattr(params, key), value)
            self.assertEqual(params.params[key], value)

    def test_seed_controls_model_numpy_and_python_initialization(self):
        def sample():
            return torch.nn.Linear(3, 2).weight.detach().clone(), np.random.rand(), random.random()
        seed_everything(71)
        first = sample()
        seed_everything(71)
        second = sample()
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertEqual(first[1:], second[1:])

    def test_multirank_fails_before_cuda_or_distributed_initialization(self):
        from train.downstream.point_classification_trainer import DownstreamTrainer
        with patch.dict(os.environ, WORLD_SIZE='2'), patch('torch.distributed.init_process_group') as init, \
                patch('torch.cuda.set_device') as device:
            with self.assertRaisesRegex(ValueError, 'single-process'):
                DownstreamTrainer(SimpleNamespace(task='nid'), SimpleNamespace())
            init.assert_not_called()
            device.assert_not_called()
            require_single_process('pid')
        with patch.dict(os.environ, WORLD_SIZE='1'), patch('torch.distributed.is_initialized', return_value=True), \
                patch('torch.distributed.get_world_size', return_value=2):
            with self.assertRaisesRegex(ValueError, 'single-process'):
                require_single_process('nid')

    def test_checkpoint_identity_and_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, c = (root / name for name in ('original', 'renamed', 'different'))
            a.write_bytes(b'pretrained weights'); b.write_bytes(a.read_bytes()); c.write_bytes(b'other weights')
            contract = lambda path: training_contract(SimpleNamespace(pretrained_ckpt=path))
            saved = dict(noise_training_contract=contract(a))
            validate_noise_checkpoint(saved, contract(b))
            with self.assertRaisesRegex(ValueError, 'different pretrained backbone'):
                validate_noise_checkpoint(saved, contract(c))
            with self.assertRaisesRegex(ValueError, 'mode mismatch'):
                validate_noise_checkpoint(saved, contract(None))
            with self.assertRaisesRegex(ValueError, 'mode mismatch'):
                validate_noise_checkpoint(dict(noise_training_contract=contract(None)), contract(a))

    def test_checkpoint_cohort_and_data_product_mismatch_are_rejected(self):
        saved = dict(training_config=dict(data_contract={'plan_id': 'original'}), training_cohort_digest='abc')
        validate_noise_data(saved, {'plan_id': 'original'}, 'abc')
        with self.assertRaisesRegex(ValueError, 'training cohort'):
            validate_noise_data(saved, {'plan_id': 'original'}, 'different')
        with self.assertRaisesRegex(ValueError, 'data contract'):
            validate_noise_data(saved, {'plan_id': 'changed'}, 'abc')

    def test_evaluation_restores_training_recipe_but_allows_relocation(self):
        params = YParams(str(ROOT / 'scripts/configs/clas12_adapter_recipes.yaml'), 'hdf5_noise_50k_balanced')
        saved = dict(training_config=dict(params.params, batch_size=8, seed=71, limit_size=32,
                                         num_embedder_layers=2, data_root='/old/data'))
        params['data_root'] = '/new/data'
        restore_evaluation_config(params, saved, validation_size=16)
        self.assertEqual((params.limit_size, params.seed, params.num_embedder_layers), (32, 71, 2))
        self.assertEqual((params.data_root, params.limit_test_size), ('/new/data', 16))
        self.assertEqual(saved['training_config']['limit_test_size'], 5000)
        with self.assertRaisesRegex(ValueError, 'HDF5 noise-tagging'):
            restore_evaluation_config(params, {'training_config': dict(task='pid')})

    def test_legacy_adapteronly_support_and_unverifiable_frozen_rejection(self):
        old = dict(params=dict(params=dict(task='pid', pretrained_ckpt='stale'), task='nid', pretrained_ckpt=None))
        self.assertEqual(checkpoint_config(old)['task'], 'nid')
        validate_noise_checkpoint(old, dict(mode='adapteronly'))
        old['params']['pretrained_ckpt'] = 'old backbone'
        with self.assertRaisesRegex(ValueError, 'no backbone fingerprint'):
            validate_noise_checkpoint(old, dict(mode='frozen'))
        with self.assertRaisesRegex(ValueError, 'no verifiable training mode'):
            validate_noise_checkpoint({}, dict(mode='adapteronly'))

    def test_shared_evaluator_uses_frozen_features_and_ignores_padding(self):
        from train.downstream.noise_evaluation import evaluate
        class Head(torch.nn.Module):
            def forward(self, points, feature=None, pretrain=False, padding_mask=None):
                self.feature = feature
                self.pretrain = pretrain
                return dict(pred_logits=points[..., :2])
        class Loader(list):
            dataset = SimpleNamespace(split='validation')
        batch = dict(points=torch.tensor([[[2., 0., 0.], [0., 2., 0.], [-100., -100., -100.]]]),
                     noise_target=torch.tensor([[0, 1, -100]]), coatjava_seg_pred=torch.tensor([[4, -1, -100]]),
                     source_id=torch.tensor([1]), original_ordinal=torch.tensor([2]), source_event_index=torch.tensor([3]))
        backbone = Mock(return_value=(None, [torch.zeros(1, 3, 2)], None))
        trainer = SimpleNamespace(device='cpu', model=backbone, down_model=Head(),
                                  noise_training_contract=dict(mode='frozen'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation = evaluate(trainer, Loader([batch]), root, 'validation')
            self.assertTrue(trainer.down_model.pretrain)
            self.assertEqual(tuple(trainer.down_model.feature.shape), (1, 1, 3, 2))
            np.testing.assert_array_equal(np.load(root / 'validation_scores.npz')['labels'], [0, 1])
            loader = Loader([batch]); loader.dataset = SimpleNamespace(split='test')
            test = evaluate(trainer, loader, root, thresholds=validation['calibration'])
            for name, value in test['operating_points'].items():
                self.assertEqual(value['threshold'], validation['calibration'][name]['threshold'])
            trainer.noise_training_contract['mode'] = 'adapteronly'
            backbone.reset_mock()
            evaluate(trainer, loader, root)
            backbone.assert_not_called()
            self.assertIsNone(trainer.down_model.feature)

    def test_frozen_campaign_series_include_backbone_identity(self):
        from train.downstream.tests.test_noise_campaign import write_run, manifest
        from train.downstream.campaign.collate_noise_tagging_campaign import collate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [write_run(root, 'a'), write_run(root, 'b')]
            for i, run in enumerate(runs):
                path = root / run['results_dir'] / 'summary.json'
                summary = json.loads(path.read_text())
                summary.update(training_mode='frozen', backbone_checkpoint_sha256=str(i) * 64)
                path.write_text(json.dumps(summary))
            report = collate(manifest(root, runs))
            self.assertEqual(len({row['series_id'] for row in report['rows']}), 2)


if __name__ == '__main__':
    unittest.main()
