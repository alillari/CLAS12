import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from train.downstream.eval.evaluate_track_regression import DownstreamTrainer
from train.downstream.model import RegressionTargetNormalizer
from train.downstream.physics_checkpoints import resolve_config, selection_mode
from train.downstream.regression_utils import validate_regression_training


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
        self.target_normalizer = RegressionTargetNormalizer(3)

    def forward(self, points, **kwargs):
        return {'pred_regression': points[:, 0, :3] + self.offset}


class ValidationLossTrainingTest(unittest.TestCase):
    def trainer(self, directory):
        trainer = DownstreamTrainer.__new__(DownstreamTrainer)
        trainer.params = Params(checkpoint_dir=directory, task='mom', trained_checkpoint_path=None)
        trainer.physics_config = resolve_config('mom')
        trainer.checkpoint_selection = 'validation_loss'
        trainer.device = 'cpu'
        trainer.down_model = Head()
        trainer.model = torch.nn.Identity()
        trainer.down_optimizer = torch.optim.SGD(trainer.down_model.parameters(), lr=.1)
        trainer.down_scheduler = torch.optim.lr_scheduler.StepLR(trainer.down_optimizer, step_size=1)
        trainer.regression_target_stats = dict(task='mom', columns=['px', 'py', 'pz'], path='synthetic.json',
                                              angular_indices=[], phi_pairs=[], std=[1.]*3, mean=[0.]*3)
        trainer.regression_loss = 'mae'
        trainer.best_loss = np.inf
        trainer.min_delta = 1e-4
        trainer.best_step = trainer.best_epoch = trainer.best_loss_step = trainer.best_loss_epoch = None
        trainer.global_step = 0
        trainer.last_validation_step = None
        trainer.stagnation_counter = 0
        trainer.early_stopping_min_steps = 0
        trainer.log_to_screen = False
        trainer.down_results = {'val': []}
        return trainer

    def validation(self, trainer, step, loss):
        trainer.global_step = step
        with torch.no_grad():
            trainer.down_model.offset.fill_(step)
        trainer._record_validation_result(loss, 'selected.pth', 0, step)

    def test_saved_weights_follow_minimum_loss_and_only_one_file_is_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.trainer(tmp)
            self.validation(trainer, 1, .2)
            self.validation(trainer, 2, .1)
            self.validation(trainer, 3, .15)
            checkpoint = torch.load(Path(tmp, 'selected.pth'), weights_only=False)
            self.assertEqual(checkpoint['model_state_dict']['offset'].item(), 2)
            self.assertEqual(trainer.best_step, 2)
            self.assertEqual(trainer.best_loss, .1)
            self.assertEqual(checkpoint['checkpoint_selection'], 'validation_loss')
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ['selected.pth'])

    def test_min_delta_controls_patience_without_discarding_better_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.trainer(tmp)
            self.validation(trainer, 1, .2)
            self.validation(trainer, 2, .2 - 1e-5)
            self.assertEqual(trainer.best_step, 2)
            self.assertEqual(trainer.stagnation_counter, 1)

    def test_nonfinite_loss_does_not_overwrite_selected_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.trainer(tmp)
            self.validation(trainer, 1, .2)
            for step, loss in enumerate((float('nan'), float('inf')), 2):
                self.validation(trainer, step, loss)
            checkpoint = torch.load(Path(tmp, 'selected.pth'), weights_only=False)
            self.assertEqual(checkpoint['global_step'], 1)
            self.assertEqual(trainer.best_step, 1)

    def test_small_improvements_accumulate_for_patience(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.trainer(tmp)
            self.validation(trainer, 1, .2)
            self.validation(trainer, 2, .2 - 6e-5)
            self.assertEqual(trainer.stagnation_counter, 1)
            self.validation(trainer, 3, .2 - 12e-5)
            self.assertEqual(trainer.stagnation_counter, 0)
            self.assertEqual(trainer.best_step, 3)

    def test_optuna_pruning_reports_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.trainer(tmp)
            class Trial:
                reports = []
                def report(self, value, step):
                    self.reports.append((value, step))
                def should_prune(self):
                    return False
            trial = Trial()
            trainer._record_validation_result(.2, 'selected.pth', 0, 5, optuna_trial=trial)
            self.assertEqual(trial.reports, [(.2, 5)])

    def test_validation_masks_padding_and_restores_model_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.trainer(tmp)
            trainer.down_model.train()
            trainer.model.eval()
            trainer._representation_input = lambda grouped, _: grouped
            reg = torch.zeros(4, 2, 3)
            reg[:, 0, 0] = 500
            reg[:, 1, 0] = 9000
            points = torch.full((4, 2, 3), -100.)
            points[:, 0, :] = reg[:, 0, :]
            trainer.val_data_loader = [dict(points=points, reg_target=reg)]
            self.assertEqual(trainer.validate_end_to_end_one_epoch(), 0.)
            self.assertTrue(trainer.down_model.training)
            self.assertFalse(trainer.model.training)
            with torch.no_grad():
                trainer.down_model.offset.fill_(.02)
            self.assertGreater(trainer.validate_end_to_end_one_epoch(), 0.)
            self.assertEqual(len(trainer.down_results['val']), 1)

    def test_final_validation_when_epoch_cap_precedes_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.trainer(tmp)
            trainer.global_step = 2
            trainer.epoch = 1
            with patch.object(trainer, 'validate_end_to_end_one_epoch', return_value=.2) as validate:
                trainer._finish_validation(False, 'selected.pth')
                trainer._finish_validation(False, 'selected.pth')
            self.assertEqual(validate.call_count, 1)
            self.assertEqual(trainer.best_step, 2)

    def test_nonfinite_predictions_cannot_be_masked_into_a_finite_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.trainer(tmp)
            trainer._representation_input = lambda grouped, _: grouped
            trainer.val_data_loader = [dict(points=torch.ones(2, 1, 3), reg_target=torch.ones(2, 1, 3))]
            with torch.no_grad():
                trainer.down_model.offset.fill_(float('nan'))
            with self.assertRaisesRegex(RuntimeError, 'nonfinite predictions'):
                trainer.validate_end_to_end_one_epoch()
            self.assertIsNone(trainer.params.trained_checkpoint_path)

    def test_no_finite_loss_fails_even_if_old_checkpoint_file_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.trainer(tmp)
            Path(tmp, 'selected.pth').write_text('previous run')
            trainer.global_step = 2
            with patch.object(trainer, 'validate_end_to_end_one_epoch', return_value=float('nan')):
                with self.assertRaisesRegex(RuntimeError, 'no finite validation-loss checkpoint'):
                    trainer._finish_validation(False, 'selected.pth')
            self.assertIsNone(trainer.params.trained_checkpoint_path)

    def test_old_width_selected_weights_allow_inference_but_not_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.trainer(tmp)
            self.validation(trainer, 1, .2)
            path = Path(tmp, 'selected.pth')
            checkpoint = torch.load(path, weights_only=False)
            checkpoint['checkpoint_selection'] = 'physics'
            torch.save(checkpoint, path)
            trainer.load_checkpoint(path, inference=True)
            self.assertEqual(trainer.loaded_checkpoint_metadata['checkpoint_selection'], 'physics')
            with self.assertRaisesRegex(ValueError, 'Cannot resume width-selected training'):
                trainer.load_checkpoint(path, inference=False)

    def test_defaults_and_retired_recipes(self):
        validate_regression_training({'task': 'mom'})
        self.assertEqual(selection_mode({'task': 'mom'}), 'validation_loss')
        for overrides in ({'task': 'p'}, {'task': 'theta'}, {'task': 'phi'},
                          {'input_representation': 'clas12_geometry_v1'},
                          {'regression_loss': 'physical_resolution_l1'},
                          {'regression_loss': 'physical_resolution_relative_huber'},
                          {'checkpoint_selection': 'physics'}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_regression_training({'task': 'mom', **overrides})


if __name__ == '__main__':
    unittest.main()
