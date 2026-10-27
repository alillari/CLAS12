import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from train.downstream.physics_checkpoints import (
    CheckpointSummary, assess, rank_checkpoints, resolve_config, summarize,
    summarize_native, write_evaluation_summary,
)


class PhysicsCheckpointTest(unittest.TestCase):
    def config(self, **kwargs):
        return resolve_config('mom', dict(bin_edges=[0, 1, 2], min_bin_entries=2, **kwargs))

    def row(self, step, metrics, **kwargs):
        row = summarize([.5, .5, 1.5, 1.5], [-.1, .1, -.1, .1], self.config())
        row.update(step=step, epoch=0, validation_loss=1 / step, checkpoint=f'{step}.pth',
                   **dict(zip(('W_macro', 'B_macro', 'B_worst', 'T_macro'), metrics)))
        row.update(kwargs)
        return row

    def test_macro_weights_bins_equally_and_retains_inclusive(self):
        cfg = self.config()
        bins = [.5]*100 + [1.5]*4
        residuals = [-.01, .01]*50 + [-.2, -.1, .1, .2]
        result = summarize(bins, residuals, cfg)
        expected = np.quantile(residuals[-4:], [.16, .84])
        self.assertAlmostEqual(result['W_macro'], (.01 + (expected[1]-expected[0])/2)/2)
        self.assertAlmostEqual(result['T_macro'], .25)
        self.assertAlmostEqual(result['inclusive']['tail_fraction'], 2/104)
        self.assertEqual(result['n_valid_bins'], 2)

    def test_edges_sparse_bins_and_nonfinite_predictions(self):
        cfg = self.config(min_valid_bins=1)
        row = summarize([0, .5, 1, 2, 3], [0, 0, .1, np.nan, 0], cfg)
        self.assertEqual([r['n_truth'] for r in row['per_bin']], [2, 2])
        self.assertEqual(row['per_bin'][1]['invalid_reason'], 'nonfinite_predictions')
        self.assertEqual(row['n_outside_bins'], 1)
        self.assertFalse(assess(row, cfg)['eligible'])
        sparse = summarize([.5, .5, 1.5], [0, 0, .3], cfg)
        self.assertIsNone(sparse['per_bin'][1]['w68'])
        self.assertTrue(assess(sparse, cfg)['eligible'])
        self.assertFalse(assess(sparse, self.config())['eligible'])

    def test_momentum_log_decode_and_bins_are_truth_only(self):
        cfg = self.config()
        truth = np.log([[500], [500], [1500], [1500]])
        pred = np.log([[1250], [1250], [3750], [3750]])
        row = summarize_native(pred, truth, 'p', cfg)
        self.assertEqual([r['n_truth'] for r in row['per_bin']], [2, 2])
        self.assertAlmostEqual(row['B_macro'], 1.5)
        self.assertAlmostEqual(row['T_macro'], 1)
        self.assertAlmostEqual(row['W_macro'], 0)

    def test_angles_and_angle_binning_by_raw_truth_p(self):
        cfg = resolve_config('phi', dict(bin_quantity='p', bin_edges=[0, 1], min_bin_entries=2))
        row = summarize_native([[np.pi-.01]]*2, [[-np.pi+.01]]*2, 'phi', cfg,
                               truth_xyz=[[500, 0, 0]]*2)
        self.assertAlmostEqual(row['B_macro'], .02)
        theta = resolve_config('theta', dict(bin_edges=[0, 3.2], min_bin_entries=2))
        row = summarize_native([[3.]]*2, [[.1]]*2, 'theta', theta)
        self.assertAlmostEqual(row['B_macro'], 2.9)

    def test_guards_never_fallback_and_pareto_includes_tradeoffs(self):
        cfg = self.config(guardrails=dict(B_macro=.02, B_worst=.03, T_macro=.1))
        rows = [self.row(1, (.02, .01, .02, .05)), self.row(2, (.01, .03, .04, .1)),
                self.row(3, (.03, .02, .03, .08))]
        self.assertEqual(rank_checkpoints(rows, cfg)['step'], 1)
        self.assertEqual([r['on_pareto_frontier'] for r in rows], [True, True, False])
        self.assertIsNone(rank_checkpoints(rows[1:2], cfg))
        self.assertFalse(rows[1]['selected'])
        self.assertIn('B_macro', rows[1]['rejection_reasons'])
        for metric, value in [('B_worst', .04), ('T_macro', .2)]:
            bad = self.row(4, (.001, .001, .001, .001), **{metric:value})
            self.assertIsNone(rank_checkpoints([bad], cfg))

    def test_width_tie_hierarchy_and_provisional_selection(self):
        cfg = self.config(width_tie_atol=.001)
        rows = [self.row(1, (.02, .02, .02, .03)), self.row(2, (.0205, .01, .02, .03)),
                self.row(3, (.0208, .01, .015, .03)), self.row(4, (.0209, .01, .015, .02)),
                self.row(5, (.022, 0, 0, 0))]
        self.assertEqual(rank_checkpoints(rows, cfg)['step'], 4)
        strict = self.config(require_configured_guardrails=True)
        self.assertIsNone(rank_checkpoints(rows, strict))

    def test_invalid_loss_and_changed_cohort(self):
        cfg = self.config()
        a = self.row(1, (.02, .01, .02, .05), validation_loss=np.nan)
        self.assertIsNone(rank_checkpoints([a], cfg))
        b = self.row(2, (.02, .01, .02, .05), truth_support_hash='different')
        with self.assertRaisesRegex(ValueError, 'truth sample'):
            rank_checkpoints([a, b], cfg)
        good = self.row(3, (.02, .01, .02, .05))
        changed = copy.deepcopy(good); changed['valid_bin_indices'] = [0]
        with self.assertRaisesRegex(ValueError, 'valid-bin'):
            rank_checkpoints([good, changed], cfg)

    def test_json_roundtrip_plots_and_diagnostic_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config()
            history = CheckpointSummary(cfg, tmp)
            summary = summarize([.5, .5, 1.5, 1.5], [-.01, .01, -.02, .02], cfg)
            history.add(summary, step=1, epoch=0, validation_loss=1, checkpoint='one.pth')
            history.add(summary, step=2, epoch=0, validation_loss=np.nan, checkpoint='two.pth')
            payload = history.write()
            self.assertEqual(payload['selection_status'], 'selected_provisional_guardrails_disabled')
            parsed = json.loads(Path(tmp, 'checkpoint_summary.json').read_text())
            self.assertIsNone(parsed['checkpoints'][1]['validation_loss'])
            restored = CheckpointSummary(cfg, tmp, parsed['checkpoints'])
            restored.write(plots=False)
            self.assertFalse(restored.records[1]['eligible'])
            self.assertTrue(Path(tmp, 'checkpoint_metrics_by_step.pdf').is_file())
            self.assertTrue(Path(tmp, 'checkpoint_pareto.png').is_file())
            evaluation = write_evaluation_summary(tmp, [[500,0,0]]*2, [[500,0,0]]*2, 'mom',
                self.config(min_valid_bins=1), metadata={'global_step':1, 'epoch':0,
                'physics_checkpoint_summary_path':str(Path(tmp, 'checkpoint_summary.json'))})
            self.assertTrue(evaluation['checkpoint']['selected'])
            self.assertEqual(evaluation['purpose'], 'evaluation_diagnostics_only')

    def test_nonfinite_predictions_and_extreme_values_write_strict_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config()
            history = CheckpointSummary(cfg, tmp)
            bad = summarize([.5,.5,1.5,1.5], [np.nan]*4, cfg)
            history.add(bad, step=1, epoch=0, validation_loss=0., checkpoint='bad.pth')
            self.assertEqual(history.write(plots=False)['selection_status'], 'no_checkpoint_passed')
            huge = summarize([.5,.5,1.5,1.5], [-1e308,1e308,-1e308,1e308], cfg)
            self.assertTrue(np.isfinite(huge['W_macro']))
            self.assertAlmostEqual(huge['W_macro']/1e308, .68)
            history.add(huge, step=2, epoch=0, validation_loss=1., checkpoint='huge.pth')
            history.write(plots=False)

    def test_joint_decoders_and_nonpositive_predicted_magnitude(self):
        cfg = self.config(min_valid_bins=1)
        for task, truth in [('mom', [[300,400,0]]*2),
                            ('p_phi_theta', [[500,1,0,np.pi/2]]*2),
                            ('pt_phi_eta', [[500,1,0,0]]*2)]:
            with self.subTest(task=task):
                result = summarize_native(truth, truth, task, cfg)
                self.assertTrue(assess(result, cfg)['eligible'])
                self.assertEqual(result['W_macro'], 0)
        result = summarize_native([[-500,1,0,1]]*2, [[500,1,0,1]]*2, 'p_phi_theta', cfg)
        self.assertEqual(result['n_invalid_predictions'], 2)
        self.assertFalse(assess(result, cfg)['eligible'])

    def test_config_validation(self):
        for opts in (dict(min_valid_bins=0), dict(min_bin_entries=1), dict(tail_threshold=0),
                     dict(bin_edges=[1,1]), dict(guardrails={'T_macro':1.1}), dict(guardrails={'typo':1})):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                resolve_config('mom', opts)
        with self.assertRaises(ValueError):
            resolve_config('phi', dict(residual='p_relative'))


if __name__ == '__main__':
    unittest.main()
