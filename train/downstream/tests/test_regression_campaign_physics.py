import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'train/downstream/campaign'))
sys.path.insert(0, str(ROOT / 'train/downstream'))
from campaign_util import (build_adapter_only_run_row, render_model_yaml, render_analysis_yaml,
                           collate_summary, read_yaml)
from build_track_regression_manifest import parse_training_overrides
from config_overrides import deep_merge
from physics_reporting import collect_run, make_physics_campaign_plots
from train.downstream.physics_checkpoints import CheckpointSummary, resolve_config, summarize, write_evaluation_summary
from train.downstream.validation_config import configure_validation
from train.downstream.track_regression_experiment import TrackRegressionExperimentConfig, resolve_params
from train.downstream.evaluation_contract import EVALUATION_CONTRACT
import plot_track_regression_campaign as plotter


class CampaignPhysicsTest(unittest.TestCase):
    def args(self, overrides):
        return SimpleNamespace(max_epochs=None, early_stopping_patience=None, early_stopping_warmup_steps=None,
                               max_train_batches=None, max_val_batches=None, training_override=overrides)

    def manifest(self, directory):
        return dict(campaign_dir=str(directory), artifact_root=str(directory), campaign_name='synthetic',
                    base_model_yaml=str(ROOT/'scripts/configs/mamba_clas12_track_regression_adapteronly.yaml'),
                    base_analysis_yaml=str(ROOT/'train/downstream/eval/track_regression_analysis_adapteronly.yaml'), runs=[])

    def test_nested_overrides_are_typed_and_merge_in_command_order(self):
        values = parse_training_overrides(self.args([
            'physics_checkpoint.guardrails.B_macro=0.01',
            'physics_checkpoint={guardrails: {B_worst: 0.02}, min_bin_entries: 250}',
            'physics_checkpoint.guardrails.B_macro=null',
            'physics_checkpoint.bin_edges=[0.25, 0.5, 1.0, 2.0]',
            'physics_checkpoint.require_configured_guardrails=true',
            'max_val_batches=None',
        ]))
        cfg = values['physics_checkpoint']
        self.assertIsNone(cfg['guardrails']['B_macro'])
        self.assertEqual(cfg['guardrails']['B_worst'], .02)
        self.assertEqual(cfg['bin_edges'], [.25,.5,1.,2.])
        self.assertIs(cfg['require_configured_guardrails'], True)
        self.assertIsNone(values['max_val_batches'])
        with self.assertRaisesRegex(ValueError, 'non-mapping'):
            parse_training_overrides(self.args(['physics_checkpoint=7','physics_checkpoint.guardrails.B_macro=0.01']))
        with self.assertRaises(ValueError):
            parse_training_overrides(self.args(['physics_checkpoint..guardrails=1']))
        with self.assertRaises(ValueError):
            parse_training_overrides(self.args(['physics_checkpoint.bin_edges=[1,2']))
        base = dict(physics_checkpoint=dict(guardrails=dict(B_macro=.03, T_macro=.1)))
        merged = deep_merge(base, dict(physics_checkpoint=dict(guardrails=dict(B_macro=.01))))
        self.assertEqual(merged['physics_checkpoint']['guardrails'], dict(B_macro=.01,T_macro=.1))
        self.assertEqual(base['physics_checkpoint']['guardrails']['B_macro'], .03)

    def test_rendered_heads_keep_validation_fixed_and_preserve_other_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.manifest(Path(tmp))
            manifest['training_overrides'] = parse_training_overrides(self.args([
                'physics_checkpoint.guardrails.B_macro=0.01', 'physics_checkpoint.guardrails.T_macro=0.1']))
            for family in ('adapteronly', 'pretrained'):
                for task in ('p','theta','phi'):
                    for budget, batch in ((1000,32),(100000,64)):
                        with self.subTest(family=family, task=task, budget=budget):
                            row = build_adapter_only_run_row(Path(tmp)/family/task, budget, batch, 5000)
                            row['base_model_yaml'] = str(ROOT/f'scripts/configs/mamba_clas12_track_regression_{family}.yaml')
                            row['base_model_config'] = f'clas12_track_regression_{family}_{task}_only'
                            row['training_overrides'] = {'physics_checkpoint': {'guardrails': {'B_worst':.02}}}
                            if family == 'pretrained':
                                row.update(model_family='mamba1', num_layers_backbone=6)
                            render_model_yaml(manifest,row); render_analysis_yaml(manifest,row)
                            params = resolve_params(TrackRegressionExperimentConfig(yaml_config=row['model_yaml'],
                                config=row['model_config'], eventnumber=budget, train_batch_size=batch))
                            self.assertEqual(params.task, task)
                            self.assertEqual(params.limit_size, budget)
                            self.assertEqual(params.params['limit_size'], budget)
                            self.assertEqual(params.batch_size, batch)
                            self.assertEqual(params.params['batch_size'], batch)
                            self.assertEqual(params.limit_test_size, 50000)
                            self.assertEqual(params.valid_batch_size, 128)
                            self.assertIsNone(params.max_val_batches)
                            self.assertEqual(params.physics_checkpoint['min_bin_entries'],200)
                            self.assertEqual(params.physics_checkpoint['guardrails'], dict(B_macro=.01,B_worst=.02,T_macro=.1))
                            self.assertEqual(params.physics_checkpoint['bin_quantity'],task)
                            self.assertFalse(params.drop_last_test)
            row['training_overrides'] = {'physics_checkpoint': {'typo':1}}
            with self.assertRaisesRegex(ValueError,'Unknown physics_checkpoint'):
                render_model_yaml(manifest,row)

    def test_batch_cap_cannot_silently_shrink_validation_sample(self):
        with self.assertRaisesRegex(ValueError,'truncate'):
            configure_validation(dict(limit_test_size=50000, valid_batch_size=32, max_val_batches=500))
        params = configure_validation(dict(limit_test_size=1000, valid_batch_size=128, max_val_batches=8))
        self.assertEqual(params['limit_test_size'],1000)
        for key, value in [('limit_test_size',0),('valid_batch_size',0),('max_val_batches',0),('limit_test_data','false')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                configure_validation({key:value})

    def create_run(self, directory, budget, task='p', passed=True):
        row = build_adapter_only_run_row(directory/task, budget,32,5000)
        root = Path(row['checkpoint_dir']); root.mkdir(parents=True)
        evaluation_dir=Path(row['evaluation_dir']); evaluation_dir.mkdir(parents=True)
        cfg = resolve_config(task, dict(bin_edges=[0,1,2], min_bin_entries=2,
                                       guardrails={} if passed else dict(B_macro=0)))
        ledger = CheckpointSummary(cfg, root/'ledger')
        summary = summarize([.5,.5,1.5,1.5],[-.01,.03,-.01,.03],cfg)
        ledger.add(summary,step=10,epoch=1,validation_loss=.4,checkpoint=root/'a.pth')
        ledger.add(summarize([.5,.5,1.5,1.5],[-.02,.04,-.02,.04],cfg),step=20,epoch=2,validation_loss=.2,checkpoint=root/'b.pth')
        payload=ledger.write(plots=False)
        # A relocated mounted path is resolved against the current run directory.
        pointer=Path(row['adapter_checkpoint']).with_name(Path(row['adapter_checkpoint']).stem+'_physics_report.json')
        pointer.write_text(json.dumps(dict(task=task, checkpoint_summary='/old/machine/ledger/checkpoint_summary.json',
                                          selection_status=payload['selection_status'])))
        contract=dict(evaluation_contract=EVALUATION_CONTRACT,comparison_truth='mctrue_inner_hit',swingback_enabled=False)
        if passed:
            truth = [[500,0,0]]*2+[[1500,0,0]]*2
            if task == 'p':
                import numpy as np
                truth=np.log([[500]]*2+[[1500]]*2)
            else:
                truth=[[.5],[.5],[1.5],[1.5]]
            write_evaluation_summary(evaluation_dir,truth,truth,task,cfg, metadata=dict(global_step=10,epoch=1))
            (evaluation_dir/'summary.json').write_text(json.dumps(dict(**contract,training_target_task=task,
                methods={'adapter':{'kinematic':{task+'_rad':{'mae':.1,'rmse':.2,'n':4}}}},training_history={})))
            (evaluation_dir/'campaign_headline_metrics.jsonl').write_text('')
        return row

    def test_collation_preserves_selected_validation_and_evaluation_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp); manifest=self.manifest(directory)
            row=self.create_run(directory,1000)
            rejected=self.create_run(directory,100000,passed=False)
            manifest['runs']=[row,rejected]
            collate_summary(manifest)
            with (directory/'summary/run_table.csv').open() as stream:
                table=list(csv.DictReader(stream))
            self.assertEqual(table[0]['selected_step'],'10')
            self.assertEqual(table[0]['selected_validation_loss'],'0.4')
            self.assertEqual(table[0]['training_target_task'],'p')
            self.assertGreater(float(table[0]['validation_W_macro']),0)
            self.assertEqual(float(table[0]['evaluation_W_macro']),0)
            self.assertEqual(table[1]['checkpoint_selection_status'],'no_checkpoint_passed')
            self.assertEqual(table[1]['validation_W_macro'],'')
            history=[json.loads(line) for line in (directory/'summary/physics_checkpoint_history.jsonl').read_text().splitlines()]
            self.assertEqual(len(history),4)
            self.assertIn('per_bin',history[0]['checkpoint'])
            text=(directory/'summary/physics_checkpoint_summary.md').read_text()
            self.assertIn('no_checkpoint_passed',text)
            self.assertIn('selected_provisional_guardrails_disabled',text)

    def test_latest_preflight_failure_does_not_reuse_previous_selected_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            row=self.create_run(Path(tmp),1000)
            root=Path(row['checkpoint_dir']); empty=CheckpointSummary(resolve_config('p'),root/'new_ledger')
            empty.write(plots=False)
            pointer=Path(row['adapter_checkpoint']).with_name(Path(row['adapter_checkpoint']).stem+'_physics_report.json')
            pointer.write_text(json.dumps(dict(task='p',checkpoint_summary=str(root/'new_ledger/checkpoint_summary.json'),
                                              selection_status='insufficient_validation_bins')))
            report=collect_run(row)
            self.assertEqual(report['table']['checkpoint_selection_status'],'insufficient_validation_bins')
            self.assertIsNone(report['table']['selected_step'])
            self.assertIsNone(report['table']['validation_W_macro'])

    def test_scalar_all_plot_suite_skips_joint_figures_and_separates_metric_cohorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);manifest=self.manifest(directory)
            manifest['runs']=[self.create_run(directory,1000,task='p'),self.create_run(directory,10000,task='phi')]
            # Give mixed tasks unique identifiers, as a combined manifest requires.
            for row in manifest['runs']:
                row['run_id']=Path(row['run_dir']).parents[1].name+'_'+row['run_id']
            collate_summary(manifest)
            index=make_physics_campaign_plots(directory/'summary/physics_checkpoint_metrics.csv',directory/'plots')
            self.assertEqual({item['task'] for item in index},{'p','phi'})
            self.assertEqual({item['source'] for item in index},{'validation_selected','evaluation'})
            self.assertTrue(all((directory/'plots'/f"{item['plot']}.pdf").is_file() for item in index))
            args=SimpleNamespace(campaign_dir=str(directory),manifest=None,headline_jsonl=None,output_dir=None,
                                 plot_suite='all',presentation_labels=None)
            with patch.object(plotter,'parse_args',return_value=args), \
                 patch('momentum_presentation.make_presentation') as joint, \
                 patch('momentum_ml_presentation.make_ml_presentation') as cartesian:
                plotter.main()
            joint.assert_not_called();cartesian.assert_not_called()


if __name__ == '__main__':
    unittest.main()
