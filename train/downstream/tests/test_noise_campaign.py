"""Scientific aggregation contracts and a CPU-only plotting smoke test."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from train.downstream.campaign.collate_noise_tagging_campaign import collate, write_tables
from train.downstream.campaign.plot_noise_tagging_campaign import curve_points, plot_report


def write_run(root, name, labels=2, steps=3000, seed=42, test_start=20):
    directory = root / name
    directory.mkdir()
    cohorts = {split: [dict(source_id=0, original_ordinal=i) for i in ids]
               for split, ids in dict(train=range(labels), validation=range(10, 12),
                                      test=range(test_start, test_start + 3)).items()}
    config = dict(seed=seed, max_optimizer_steps=steps, local_batch_size=32, max_lr=.0003,
                  min_lr=.00001, warmup_steps=100, val_interval_steps=500,
                  input_layout='xyz', embed_dim=512)
    summary = dict(task='noise', sample_unit='event', test_split='test', training_mode='adapteronly',
                   train_samples=labels, validation_samples=2, test_samples=3, seed=seed,
                   optimizer_steps=steps, selected_checkpoint_step=steps, adapter_parameters=1000,
                   class_weights=[2., .5], data_contract=dict(plan_id='fixture', noise_policy='primary'),
                   adapter=dict(threshold=.5, confusion_true_rows_pred_columns=[[30, 10], [20, 140]]),
                   coatjava_unassigned=dict(confusion_true_rows_pred_columns=[[35, 5], [10, 150]]),
                   threshold_source='validation', operating_points={
                       'retain_95pct': dict(threshold=.8, confusion_true_rows_pred_columns=[[38, 2], [40, 120]])})
    for filename, data in [('summary.json', summary), ('resolved_config.json', config), ('cohorts.json', cohorts)]:
        (directory / filename).write_text(json.dumps(data))
    return dict(run_id=name, results_dir=name)


def manifest(root, runs):
    path = root / 'manifest.json'
    path.write_text(json.dumps(dict(campaign_name='Synthetic plotting fixture', runs=runs)))
    return path


class NoiseCampaignTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_matching_runs_share_curve_and_export_explicit_metrics(self):
        runs = [write_run(self.root, 'small'), write_run(self.root, 'large', labels=4)]
        report = collate(manifest(self.root, runs))
        defaults = [row for row in report['rows'] if row['operating_point'] == 'default']
        self.assertEqual(len({row['comparison_id'] for row in defaults}), 1)
        self.assertEqual(len({row['series_id'] for row in defaults}), 1)
        self.assertEqual([row['labeled_events'] for row in defaults], [2, 4])
        self.assertEqual(defaults[0]['metrics']['accuracy'], .85)
        write_tables(report, self.root / 'tables')
        header = (self.root / 'tables/run_table.csv').read_text().splitlines()[0]
        for key in ('accuracy', 'signal_precision', 'signal_recall', 'noise_precision', 'noise_recall', 'coatjava_accuracy'):
            self.assertIn(key, header)

    def test_different_budgets_and_same_sized_different_test_cohorts_are_separate(self):
        runs = [write_run(self.root, 'base'), write_run(self.root, 'updates', steps=30000),
                write_run(self.root, 'cohort', test_start=30)]
        report = collate(manifest(self.root, runs))
        self.assertEqual(len({row['comparison_id'] for row in report['rows']}), 3)

    def test_hilbert_grids_are_separate_series_and_unmarked_runs_are_legacy(self):
        runs = [write_run(self.root, 'historical'), write_run(self.root, 'legacy', labels=4),
                write_run(self.root, 'current')]
        for name, grid in [('legacy', 'legacy_v1'), ('current', 'unit_v2')]:
            path = self.root / name / 'resolved_config.json'
            config = json.loads(path.read_text())
            config['hilbert_angular_grid'] = grid
            path.write_text(json.dumps(config))
        report = collate(manifest(self.root, runs))
        ids = {row['run_id']: row['series_id'] for row in report['rows']}
        self.assertEqual(ids['historical'], ids['legacy'])
        self.assertNotEqual(ids['historical'], ids['current'])

    def test_inconsistent_coatjava_baseline_is_rejected(self):
        runs = [write_run(self.root, 'small'), write_run(self.root, 'large', labels=4)]
        path = self.root / 'large/summary.json'
        data = json.loads(path.read_text())
        data['coatjava_unassigned']['confusion_true_rows_pred_columns'] = [[34, 6], [10, 150]]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'COATJAVA baseline differs'):
            collate(manifest(self.root, runs))

    def test_threshold_calibration_must_be_validation_only(self):
        run = write_run(self.root, 'bad')
        path = self.root / 'bad/summary.json'
        data = json.loads(path.read_text())
        data['threshold_source'] = 'test'
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'validation-selected'):
            collate(manifest(self.root, [run]))

    def test_duplicate_seed_is_not_treated_as_independent_repeat(self):
        runs = [write_run(self.root, 'first'), write_run(self.root, 'duplicate')]
        with self.assertRaisesRegex(ValueError, 'Duplicate model/budget/seed'):
            collate(manifest(self.root, runs))

    def test_seed_aggregation_and_undefined_metric_gap(self):
        rows = [dict(labeled_events=100, metrics=dict(accuracy=.8)),
                dict(labeled_events=100, metrics=dict(accuracy=1.)),
                dict(labeled_events=1000, metrics=dict(accuracy=None))]
        points = curve_points(rows, 'accuracy')
        self.assertAlmostEqual(points[0][1], .9)
        self.assertAlmostEqual(points[0][2], np.sqrt(.02))
        self.assertEqual(points[0][3], 2)
        self.assertTrue(np.isnan(points[1][1]))
        rows.append(dict(labeled_events=1000, metrics=dict(accuracy=.9)))
        self.assertTrue(np.isnan(curve_points(rows, 'accuracy')[1][1]))

    def test_unfinished_runs_are_reported_without_inventing_results(self):
        run = write_run(self.root, 'complete')
        report = collate(manifest(self.root, [run, dict(run_id='pending', results_dir='pending')]))
        self.assertEqual(report['skipped_runs'], ['pending'])

    def test_plot_smoke_writes_all_layouts_without_training(self):
        runs = [write_run(self.root, 'small'), write_run(self.root, 'large', labels=4)]
        report = collate(manifest(self.root, runs))
        paths = plot_report(report, self.root / 'plots', formats=['png'])
        self.assertEqual(len(paths), 11)
        for path in paths:
            self.assertGreater(Path(path).stat().st_size, 1000)
        self.assertTrue(any('signal_presentation_1x3' in path for path in paths))
        self.assertTrue(any('noise_presentation_3x1' in path for path in paths))


if __name__ == '__main__':
    unittest.main()
