import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

# The evaluator establishes the repo's legacy downstream import paths.
from train.downstream.eval.evaluate_track_regression import DownstreamTrainer
from train.downstream.model import RegressionTargetNormalizer
from train.downstream.physics_checkpoints import CheckpointSummary, resolve_config, summarize


class Params(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.))
        self.target_normalizer = RegressionTargetNormalizer(1, mean=[0.], std=[1.])

    def forward(self, points, **kwargs):
        return {'pred_regression': points[:, 0, :1] + self.offset}


class PhysicsTrainingTest(unittest.TestCase):
    def trainer(self, directory, guards=None):
        trainer = DownstreamTrainer.__new__(DownstreamTrainer)
        trainer.params = Params(checkpoint_dir=directory, task='p')
        trainer.physics_config = resolve_config('p', dict(bin_edges=[0, 1, 2], min_bin_entries=2,
                                                        guardrails=guards or {}))
        trainer.checkpoint_selection = 'physics'
        trainer.device = 'cpu'
        trainer.down_model = Head()
        trainer.model = torch.nn.Identity()
        trainer.down_optimizer = torch.optim.SGD(trainer.down_model.parameters(), lr=.1)
        trainer.down_scheduler = torch.optim.lr_scheduler.StepLR(trainer.down_optimizer, step_size=1)
        trainer.regression_target_stats = dict(task='p', columns=['log_p'], path='synthetic.json',
                                              angular_indices=[], std=[1.], mean=[0.])
        trainer.regression_loss = 'huber'
        trainer.regression_loss_reference = None
        trainer.best_loss = np.inf
        trainer.min_delta = 1e-4
        trainer.best_step = trainer.best_epoch = trainer.best_loss_step = trainer.best_loss_epoch = None
        trainer.global_step = 0
        trainer.stagnation_counter = 0
        trainer.early_stopping_min_steps = 0
        trainer.log_to_screen = False
        trainer.down_results = {'val':[]}
        trainer._initialize_physics_history('selected.pth')
        return trainer

    def validation(self, trainer, step, loss, width, bias=0):
        trainer.global_step = step
        with torch.no_grad():
            trainer.down_model.offset.fill_(step)
        trainer.last_validation_physics = summarize([.5,.5,1.5,1.5],
            [bias-width,bias+width,bias-width,bias+width], trainer.physics_config)
        trainer._record_validation_result(loss, 'selected.pth', 0, step)

    @patch.object(CheckpointSummary, 'plot')
    def test_saved_weights_follow_physics_not_loss(self, _):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.trainer(tmp)
            self.validation(trainer, 1, 2., .02)
            self.validation(trainer, 2, 1., .04)
            self.assertEqual(trainer.best_step, 1)
            self.assertEqual(trainer.best_loss_step, 2)
            self.assertEqual(trainer.best_loss, 1.)
            checkpoint = torch.load(Path(tmp, 'selected.pth'), weights_only=False)
            self.assertEqual(checkpoint['model_state_dict']['offset'].item(), 1)
            self.assertEqual(len(list(trainer.physics_history.output_dir.glob('step_*.pth'))), 2)
            summary = json.loads(Path(trainer.params.physics_checkpoint_summary).read_text())
            self.assertEqual([r['selected'] for r in summary['checkpoints']], [True,False])

    @patch.object(CheckpointSummary, 'plot')
    def test_no_guardrail_pass_archives_stale_alias_and_publishes_none(self, _):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'selected.pth').write_text('stale')
            trainer = self.trainer(tmp, dict(B_macro=.001, B_worst=.002, T_macro=.1))
            self.validation(trainer, 1, .01, .001, bias=.03)
            self.assertFalse(Path(tmp, 'selected.pth').exists())
            self.assertTrue((trainer.physics_history.output_dir / 'previous_selected.pth').exists())
            self.assertIsNone(trainer.params.trained_checkpoint_path)
            self.assertEqual(trainer.params.checkpoint_selection_status, 'no_checkpoint_passed')

    def test_validation_restores_training_and_uses_masked_raw_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.trainer(tmp)
            trainer.down_model.train()
            trainer.model.eval()
            trainer._representation_input = lambda grouped, _: grouped
            trainer._geometry_context_kwargs = lambda *_: {}
            p = torch.tensor([500.,500.,1500.,1500.])
            # Second hit is padding and deliberately has a conflicting label.
            reg = torch.zeros(4, 2, 3); reg[:,0,0] = p; reg[:,1,0] = 9000
            points = torch.full((4,2,3), -100.); points[:,0,0] = p.log()
            trainer.val_data_loader = [dict(points=points, reg_target=reg)]
            trainer.regression_loss = 'log_p_huber'
            # Use the canonical resolver to avoid duplicating the loss name contract.
            from train.downstream.regression_utils import resolve_regression_loss
            trainer.regression_loss = resolve_regression_loss('p')
            first = trainer.validate_end_to_end_one_epoch()
            self.assertEqual(first, 0.)
            self.assertTrue(trainer.down_model.training)
            self.assertFalse(trainer.model.training)
            self.assertEqual([r['n_truth'] for r in trainer.last_validation_physics['per_bin']], [2,2])
            support = trainer.last_validation_physics['truth_support_hash']
            with torch.no_grad():
                trainer.down_model.offset.fill_(.02)
            second = trainer.validate_end_to_end_one_epoch()
            self.assertGreater(second, first)
            self.assertEqual(len(trainer.down_results['val']), 1)
            self.assertEqual(support, trainer.last_validation_physics['truth_support_hash'])
            self.assertAlmostEqual(trainer.last_validation_physics['B_macro'], np.expm1(.02), places=6)

    @patch.object(CheckpointSummary, 'plot')
    def test_final_validation_when_epoch_cap_precedes_interval(self, _):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.trainer(tmp)
            trainer.global_step = 2; trainer.epoch = 1; trainer.last_validation_step = None
            trainer.last_validation_physics = summarize([.5,.5,1.5,1.5], [0,0,0,0], trainer.physics_config)
            with patch.object(trainer, 'validate_end_to_end_one_epoch', return_value=.2) as validate:
                trainer._finish_checkpoint_selection(False, 'selected.pth')
                trainer._finish_checkpoint_selection(False, 'selected.pth')
            self.assertEqual(validate.call_count, 1)
            self.assertEqual(trainer.best_step, 2)


if __name__ == '__main__':
    unittest.main()
