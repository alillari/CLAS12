"""Bounded HDF5 finding AdapterOnly benchmark against COATJAVA."""
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
from train.downstream.track_finding_metrics import MatchConfig, event_track_metrics, summarize_event_metrics
from train.downstream.loss import assign_points_to_masks

@torch.no_grad()
def evaluate(trainer, loader, output):
    trainer.down_model.eval()
    event_rows = {name: [] for name in ('adapter', 'coatjava')}
    records = []
    for batch in loader:
        points = batch['points'].to(trainer.device)
        mask = points[..., 0] != -100
        pred = trainer.down_model(points, feature=None, padding_mask=mask)
        assignment = assign_points_to_masks(
            {'pred_probs': pred['class_probs'], 'pred_masks': pred['mask_probs'].permute(0, 2, 1)},
            option=trainer.params.assignment_option,
            threshold=trainer.params.assignment_threshold
        )
        for i in range(len(points)):
            identity = {key: int(batch[key][i]) for key in ('source_id', 'original_ordinal', 'source_event_index')}
            valid = mask[i].cpu().numpy()
            y = batch['target'][i].numpy()
            for name in event_rows:
                labels = assignment['assignments'][i].cpu().numpy().copy() if name == 'adapter' else batch['coatjava_seg_pred'][i].numpy()
                signal = assignment['classes'][i].cpu().numpy() != 0 if name == 'adapter' else labels >= 0
                labels[~signal] = -1
                row = event_track_metrics(
                    y,
                    labels,
                    valid_mask=valid,
                    pred_signal_mask=signal & valid,
                    config=MatchConfig()
                )
                row.update(identity)
                event_rows[name].append(row)
    summary = {name: summarize_event_metrics(rows) for name, rows in event_rows.items()}
    records = event_rows
    write_json(output / 'test_predictions.json', records)
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--train-size', type=int, default=5000)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--validation-size', type=int, default=1000)
    parser.add_argument('--test-size', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['CLAS12_DATA_ROOT'] = str(args.data_root.resolve())
    os.environ['CLAS12_ARTIFACT_ROOT'] = str(output)
    config = 'hdf5_finding_5k'
    params = YParams(str(ROOT / 'scripts/configs/clas12_adapter_recipes.yaml'), config)
    overrides = dict(
        limit_size=args.train_size,
        limit_test_size=args.validation_size,
        max_validation_events=args.validation_size,
        max_optimizer_steps=args.steps,
        total_steps=args.steps,
        first_cycle_steps=args.steps,
        scheduler_first_cycle_steps=args.steps,
        warmup_steps=min(100, args.steps // 10),
        val_interval_steps=min(500, args.steps),
        num_data_workers=args.workers,
        pretrained_ckpt=None,
        log_file_name='training.log',
        checkpoint_file_name='training_checkpoint.pth',
        checkpoint_dir=str(output / 'checkpoints'),
        require_reg_target=False
    )
    for key, value in overrides.items():
        params[key] = value
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    from train.downstream.track_finding_trainer import DownstreamTrainer
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
    assert len(cohorts['train']) == args.train_size and len(cohorts['validation']) == args.validation_size
    write_json(output / 'resolved_config.json', params.params)
    with (output / 'resolved_config.yaml').open('w') as stream:
        YAML().dump({config: clean(params.params)}, stream)
    trainer.train(pretrain=False)
    assert trainer.global_step == args.steps, 'Incomplete update budget'
    checkpoint = Path(params.checkpoint_dir) / 'training_checkpoint.pth'
    trainer.load_checkpoint(str(checkpoint), inference=True)
    params['limit_test_size'] = args.test_size
    loader = get_val_loader(params, False)
    cohorts['test'] = cohort(loader.dataset)
    assert len(cohorts['test']) == args.test_size
    assert not event_keys(cohorts['test']) & (event_keys(cohorts['train']) | event_keys(cohorts['validation'])), 'Test event overlap'
    write_json(output / 'cohorts.json', cohorts)
    summary = evaluate(trainer, loader, output)
    summary.update(
        task='finding',
        optimizer_steps=trainer.global_step,
        seed=params.seed,
        train_samples=len(cohorts['train']),
        validation_samples=len(cohorts['validation']),
        test_samples=len(cohorts['test']),
        data_contract=params.data_contract,
        checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        sample_unit='event',
        validation_selection=f'{trainer.checkpoint_selection_metric}_then_loss',
        test_split='test',
        adapter_parameters=sum((p.numel() for p in trainer.down_model.parameters()))
    )
    write_json(output / 'summary.json', summary)
    print(json.dumps(clean(summary), indent=2), flush=True)
    trainer.cleanup()
if __name__ == '__main__':
    main()
