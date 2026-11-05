"""Bounded real-data CLI smoke; requires HDF5 data and a matching pretrained backbone.

This checks execution and provenance, not model quality. No learning-curve plots.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from ruamel.yaml import YAML

from fm4npp.utils import YParams
from train.downstream.hdf5_benchmark_utils import clean, write_json
from train.downstream.noise_checkpoint import file_sha256
from train.downstream.point_classification_trainer import DownstreamTrainer
from train.downstream.train_point_classification import main as train_main
from train.downstream.eval_point_classification import main as eval_main
from train.downstream.campaign.collate_noise_tagging_campaign import collate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--backbone', required=True, help='Width-512, depth-12 Mamba1 checkpoint')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    os.environ['CLAS12_DATA_ROOT'] = args.data_root
    os.environ['CLAS12_ARTIFACT_ROOT'] = str(output)
    recipe = Path(__file__).resolve().parents[3] / 'scripts/configs/clas12_adapter_recipes.yaml'
    original_train = DownstreamTrainer.train
    audit = {}

    def checked_train(trainer, **kwargs):
        before = {key: value.detach().cpu().clone() for key, value in trainer.model.state_dict().items()}
        original_train(trainer, **kwargs)
        assert trainer.global_step == 2
        assert all(not p.requires_grad and p.grad is None for p in trainer.model.parameters())
        assert all(torch.equal(before[key], value.detach().cpu()) for key, value in trainer.model.state_dict().items())
        mode = trainer.noise_training_contract['mode']
        audit[mode] = dict(backbone_unchanged=True, updates=trainer.global_step,
                           selected_checkpoint=trainer.best_checkpoint_path)

    runs = []
    for mode in ('adapteronly', 'frozen'):
        params = YParams(str(recipe), 'hdf5_noise_50k_balanced')
        overrides = dict(limit_size=32, limit_test_size=16, max_validation_events=16,
                         batch_size=8, valid_batch_size=7, num_data_workers=0,
                         max_optimizer_steps=2, total_steps=2, first_cycle_steps=2,
                         scheduler_first_cycle_steps=2, warmup_steps=0, val_interval_steps=1,
                         checkpoint_dir=str(output / mode / 'checkpoints'), log_file_name='training.log', seed=71)
        params.update_params(overrides)
        config_path = output / f'{mode}.yaml'
        with config_path.open('w') as stream:
            YAML().dump({'smoke': clean(params.params)}, stream)
        backbone_args = ['--pretrained_ckpt', args.backbone] if mode == 'frozen' else []
        cli = ['--yaml_config', str(config_path), '--config', 'smoke', '--root_dir', str(output / mode / 'runtime')]
        with patch.object(DownstreamTrainer, 'train', checked_train):
            train_main(cli + backbone_args)
        checkpoint = output / mode / 'checkpoints/training_checkpoint.pth'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['training_config']['task'] == 'nid'
        assert saved['training_config']['seed'] == 71
        assert saved['noise_training_contract']['mode'] == mode
        assert saved['training_cohort_digest']
        evaluation = output / mode / 'evaluation'
        evaluation_cli = ['eval_point_classification', '--yaml_config', str(config_path), '--config', 'smoke',
                          '--root_dir', str(evaluation), '--checkpoint', str(checkpoint),
                          '--eventnumber', '18', '--eval_batch_size', '7'] + backbone_args
        with patch.object(sys, 'argv', evaluation_cli):
            eval_main()
        summary = json.loads((evaluation / 'summary.json').read_text())
        config = json.loads((evaluation / 'resolved_config.json').read_text())
        assert (summary['train_samples'], summary['validation_samples'], summary['test_samples']) == (32, 16, 18)
        assert summary['training_cohort_verified']
        assert summary['optimizer_steps'] == 2
        assert config['batch_size'] == 8 and config['seed'] == 71  # Inference uses batch 7.
        assert summary['training_mode'] == mode
        if mode == 'frozen':
            assert summary['backbone_checkpoint_sha256'] == file_sha256(args.backbone)
        audit[mode].update(export_valid=True, partial_test_batch=True, training_config_preserved=True)
        runs.append(dict(run_id=mode, results_dir=str(evaluation.resolve())))
        gc.collect()
        torch.cuda.empty_cache()
    manifest = output / 'manifest.json'
    write_json(manifest, dict(campaign_name='CLI smoke', runs=runs))
    report = collate(manifest)
    assert {row['model_id'] for row in report['rows']} == {'adapteronly', 'frozen'}
    write_json(output / 'smoke.json', dict(checks=audit, campaign_rows=len(report['rows']), performance_claim=False))


if __name__ == '__main__':
    main()
