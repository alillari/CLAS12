import ast
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ruamel.yaml import YAML

from dev_scripts.chinchilla_restart import run
from dev_scripts.chinchilla_restart.policy import parameter_groups, scale_matrix_rates
from fm4npp.hilbert import assign_clas12_layer

ROOT = Path(__file__).parent


class RecipeTests(unittest.TestCase):
    def test_recovered_schedule_and_optimizer_restore(self):
        model = torch.nn.Module()
        model.vector = torch.nn.Parameter(torch.ones(2))
        model.matrix = torch.nn.Parameter(torch.ones(2, 2))
        model.ssm_vector = torch.nn.Parameter(torch.ones(2))
        model.ssm_vector._no_weight_decay = True
        model.ssm_matrix = torch.nn.Parameter(torch.ones(2, 2))
        model.ssm_matrix._no_weight_decay = True
        recipe = json.loads((ROOT / 'reference/width512_resolved.json').read_text())

        def construct():
            opt = torch.optim.AdamW(parameter_groups(model, recipe['weight_decay']), lr=recipe['min_lr'], betas=(.9, .95))
            sched = run.shared.CosineAnnealingWarmupRestarts(opt, first_cycle_steps=200000,
                         warmup_steps=20000, max_lr=recipe['max_lr'], min_lr=recipe['min_lr'])
            return opt, sched

        opt, sched = construct()
        self.assertEqual(len(opt.param_groups), 4)
        self.assertEqual({g['weight_decay'] for g in opt.param_groups}, {0., recipe['weight_decay']})
        self.assertTrue(all(g['lr'] == recipe['min_lr'] for g in opt.param_groups))
        # Actual historical CSV samples, not just the recipe's nominal values.
        expected = {1: .000989268821978768, 10000: .005438531181851391,
                    20000: .009888238512457074, 100000: .0062112147509622495,
                    197000: .0009949220204946206, 200000: .0009888238512457074}
        for step, lr in expected.items():
            sched.step(epoch=step)
            scale_matrix_rates(opt, 2.)
            for group in opt.param_groups:
                self.assertAlmostEqual(group['lr'], lr / (2 if group['matrix'] else 1), places=13)
        sched.step(epoch=197000)
        scale_matrix_rates(opt, 2.)
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        opt.step()
        saved_opt, saved_sched = copy.deepcopy(opt.state_dict()), copy.deepcopy(sched.state_dict())
        clone_opt, clone_sched = construct()
        clone_opt.load_state_dict(saved_opt)
        clone_sched.load_state_dict(saved_sched)
        sched.step(); scale_matrix_rates(opt, 2.)
        clone_sched.step(); scale_matrix_rates(clone_opt, 2.)
        self.assertEqual([g['lr'] for g in opt.param_groups], [g['lr'] for g in clone_opt.param_groups])
        self.assertEqual([g['matrix'] for g in clone_opt.param_groups], [False, False, True, True])

    def test_selection_matches_first_eligible_policy_and_50_cluster_filter(self):
        lengths = np.array([8, 51, 50, 49, 100, 2])
        with TemporaryDirectory() as temporary:
            feature = Path(temporary) / 'data.ninja'
            feature.write_bytes(b'fixture')
            class FakeRagged:
                starts = np.r_[0, np.cumsum(lengths[:-1] * 3)]
                ends = np.cumsum(lengths * 3)
                memmap = SimpleNamespace(filename=str(feature))
                def __len__(self): return len(lengths)
                def __getitem__(self, i): return np.zeros((lengths[i], 3), dtype=np.float32)
            dataset = run.CohortDataset.__new__(run.CohortDataset)
            dataset.memmap_feature = FakeRagged()
            dataset.memmap_seg_target = [None] * len(lengths)
            dataset.memmap_reg_target = [None] * len(lengths)
            dataset.selection_params = SimpleNamespace(minimum_clusters=1, maximum_clusters=50)
            dataset.selection_count, dataset.selection_seed, dataset.selection_split = 3, 42, 'pretrain'
            with patch.object(run, 'OUT', Path(temporary)):
                dataset.filter_data()
            np.testing.assert_array_equal(dataset.idxlist, [0, 2, 3])
            self.assertEqual(dataset.audit['eligible_events'], 4)

    def test_k1_targets_match_recovered_historical_function(self):
        tree = ast.parse((ROOT / 'reference/dataset_original.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'knn_later_indices_batch')
        namespace = {'torch': torch, 'assign_clas12_layer': assign_clas12_layer}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<historical-target>', 'exec'), namespace)
        generator = torch.Generator().manual_seed(42)
        for size in (8, 20, 50):
            points = torch.rand((size, 3), generator=generator)
            points = points[points[:, 2].argsort()]
            actual, _ = run.outward_band_targets(points, 1)
            expected = namespace['knn_later_indices_batch'](points[None], 1)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_recipe_preserves_original_training_hyperparameters(self):
        yaml = YAML(typ='safe')
        original = json.loads((ROOT / 'reference/width512_resolved.json').read_text())
        replacements = yaml.load((ROOT / 'recipe.yaml').read_text())
        fields = ['embed_dim', 'num_layers_backbone', 'd_state', 'expand', 'klen', 'dropout',
                  'use_mup', 'mup_base_width', 'local_batch_size', 'local_valid_batch_size',
                  'warmup_steps', 'total_steps', 'max_lr', 'min_lr', 'weight_decay',
                  'grad_clip_value', 'n_eval_steps', 'max_val_batches']
        for recipe in replacements.values():
            for field in fields:
                self.assertEqual(recipe[field], original[field], field)
            for field in ['seed', 'validation_seed', 'train_events', 'validation_events',
                          'minimum_clusters', 'maximum_clusters', 'clas12_band_config', 'target_rule']:
                self.assertIn(field, recipe)
        self.assertEqual(json.loads((ROOT / 'reference/calibration_original.json').read_text()),
                         json.loads(Path('dev_scripts/clas12_band_calibration.json').read_text()))


if __name__ == '__main__':
    unittest.main()
