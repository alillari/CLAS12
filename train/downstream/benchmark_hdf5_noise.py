"""Bounded HDF5 noise AdapterOnly benchmark against COATJAVA."""
from train.downstream.hdf5_benchmark_utils import clean, write_json, cohort, event_keys
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import numpy as np
import torch
from ruamel.yaml import YAML
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm4npp.utils import YParams
from fm4npp.datasets.dataset import get_val_loader
from train.downstream.noise_metrics import (
    at_threshold, calibrate_thresholds, score_summary, training_class_counts,
)

@torch.no_grad()
def evaluate(trainer, loader, output, split='test', thresholds=None):
    trainer.down_model.eval()
    records = []
    labels_all, scores_all, coat_all, offsets = [], [], [], [0]
    for batch in loader:
        points = batch['points'].to(trainer.device)
        mask = points[..., 0] != -100
        pred = trainer.down_model(points, feature=None, padding_mask=mask)
        scores = pred['pred_logits'].softmax(-1)[..., 1].cpu().numpy()
        for i in range(len(points)):
            identity = {key: int(batch[key][i]) for key in ('source_id', 'original_ordinal', 'source_event_index')}
            valid = mask[i].cpu().numpy()
            y = batch['noise_target'][i].numpy()[valid]
            predictions = {
                'adapter': (scores[i][valid] > .5).astype(int),
                'coatjava_unassigned': (batch['coatjava_seg_pred'][i].numpy()[valid] < 0).astype(int)
            }
            row = dict(identity, n_clusters=len(y))
            for name, labels in predictions.items():
                counts = np.bincount(2 * y + labels, minlength=4).reshape(2, 2)
                row[name] = counts
            records.append(row)
            labels_all.append(y)
            scores_all.append(scores[i][valid])
            coat_all.append(predictions['coatjava_unassigned'])
            offsets.append(offsets[-1] + len(y))
    labels = np.concatenate(labels_all).astype(np.int8)
    scores = np.concatenate(scores_all)
    coat = np.concatenate(coat_all).astype(np.int8)
    summary = score_summary(labels, scores, coat)
    if split == 'validation':
        assert loader.dataset.split == 'validation', 'Thresholds require the validation split'
        summary['calibration'] = calibrate_thresholds(labels, scores, coat)
    if thresholds is not None:
        summary['threshold_source'] = 'validation'
        summary['operating_points'] = {
            name: dict(target_signal_retention=point['target_signal_retention'],
                       **at_threshold(labels, scores, point['threshold']))
            for name, point in thresholds.items()
        }
    summary['threshold_curve'] = [at_threshold(labels, scores, t) for t in np.linspace(0, 1, 101)]
    np.savez_compressed(output / f'{split}_scores.npz', labels=labels, noise_score=scores,
                        coatjava_noise=coat, event_offsets=np.asarray(offsets))
    write_json(output / f'{split}_predictions.json', records)
    write_json(output / f'{split}_metrics.json', summary)
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', default='hdf5_noise_5k')
    parser.add_argument('--train-size', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--validation-size', type=int)
    parser.add_argument('--test-size', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--class-balance', choices=['none', 'inverse_frequency'])
    parser.add_argument('--checkpoint', type=Path, help='Evaluate an existing checkpoint without training')
    parser.add_argument('--evaluate-training', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['CLAS12_DATA_ROOT'] = str(args.data_root.resolve())
    os.environ['CLAS12_ARTIFACT_ROOT'] = str(output)
    config = args.config
    params = YParams(str(ROOT / 'scripts/configs/clas12_adapter_recipes.yaml'), config)
    train_size = args.train_size if args.train_size is not None else params.limit_size
    validation_size = args.validation_size if args.validation_size is not None else params.limit_test_size
    test_size = args.test_size if args.test_size is not None else getattr(params, 'test_size', 2000)
    steps = args.steps if args.steps is not None else params.max_optimizer_steps
    overrides = dict(
        limit_size=train_size,
        limit_test_size=validation_size,
        max_validation_events=validation_size,
        max_optimizer_steps=steps,
        total_steps=steps,
        first_cycle_steps=steps,
        scheduler_first_cycle_steps=steps,
        warmup_steps=min(params.warmup_steps, steps // 10),
        val_interval_steps=min(params.val_interval_steps, steps),
        num_data_workers=args.workers,
        pretrained_ckpt=None,
        log_file_name='training.log',
        checkpoint_file_name='training_checkpoint.pth',
        checkpoint_dir=str(output / 'checkpoints'),
        require_reg_target=False
    )
    for key, value in overrides.items():
        params[key] = value
    if args.class_balance is not None:
        params['noise_class_balance'] = args.class_balance
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    from train.downstream.point_classification_trainer import DownstreamTrainer
    trainer_args = SimpleNamespace(
        root_dir=str(output / 'runtime'),
        global_log_dir='logs',
        config=config,
        run_num='0'
    )
    trainer = DownstreamTrainer(params, trainer_args)
    trainer.launch()
    cohorts = {
        'train': cohort(trainer.train_data_loader.dataset),
        'validation': cohort(trainer.val_data_loader.dataset)
    }
    assert not event_keys(cohorts['train']) & event_keys(cohorts['validation']), 'Train/validation event overlap'
    assert len(cohorts['train']) == train_size and len(cohorts['validation']) == validation_size
    counts = getattr(params, 'noise_class_counts', None)
    if counts is None:
        counts = training_class_counts(trainer.train_data_loader.dataset).tolist()
    write_json(output / 'resolved_config.json', params.params)
    with (output / 'resolved_config.yaml').open('w') as stream:
        YAML().dump({config: clean(params.params)}, stream)
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.resolve()
        trainer.build_head()
    else:
        trainer.train(pretrain=False)
        assert trainer.global_step == steps, 'Incomplete update budget'
        checkpoint = Path(params.checkpoint_dir) / 'training_checkpoint.pth'
    trainer.load_checkpoint(str(checkpoint), inference=True)
    validation = evaluate(trainer, trainer.val_data_loader, output, split='validation')
    if args.evaluate_training:
        evaluate(trainer, trainer.train_data_loader, output, split='training')
    params['limit_test_size'] = test_size
    loader = get_val_loader(params, False)
    cohorts['test'] = cohort(loader.dataset)
    assert len(cohorts['test']) == test_size
    assert not event_keys(cohorts['test']) & (event_keys(cohorts['train']) | event_keys(cohorts['validation'])), 'Test event overlap'
    write_json(output / 'cohorts.json', cohorts)
    summary = evaluate(trainer, loader, output, thresholds=validation['calibration'])
    summary.pop('threshold_curve')
    summary.update(
        task='noise',
        optimizer_steps=getattr(trainer, 'global_step', None),
        selected_checkpoint_step=trainer.loaded_checkpoint_step,
        evaluated_existing_checkpoint=args.checkpoint is not None,
        training_class_counts=counts,
        class_weights=None if trainer.class_weights is None else trainer.class_weights.cpu().tolist(),
        seed=params.seed,
        train_samples=len(cohorts['train']),
        validation_samples=len(cohorts['validation']),
        test_samples=len(cohorts['test']),
        data_contract=params.data_contract,
        checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        sample_unit='event',
        validation_selection='loss',
        test_split='test',
        adapter_parameters=sum((p.numel() for p in trainer.down_model.parameters()))
    )
    write_json(output / 'summary.json', summary)
    print(json.dumps(clean(summary), indent=2), flush=True)
    trainer.cleanup()
if __name__ == '__main__':
    main()
