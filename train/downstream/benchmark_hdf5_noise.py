"""Bounded HDF5 noise AdapterOnly benchmark against COATJAVA."""
from train.downstream.hdf5_benchmark_utils import clean
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from fm4npp.utils import YParams
from train.downstream.noise_checkpoint import restore_evaluation_config
from train.downstream.noise_evaluation import export_noise_evaluation


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
    if args.checkpoint is not None:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        restore_evaluation_config(params, saved, args.validation_size)
        params['pretrained_ckpt'] = None
        params['noise_class_balance'] = 'none'
    from train.downstream.point_classification_trainer import DownstreamTrainer
    trainer_args = SimpleNamespace(
        root_dir=str(output / 'runtime'),
        global_log_dir='logs',
        config=config,
        run_num='0'
    )
    trainer = DownstreamTrainer(params, trainer_args)
    try:
        trainer.launch()
        if args.checkpoint is not None:
            checkpoint = args.checkpoint.resolve()
            trainer.build_head()
        else:
            trainer.train(pretrain=False)
            assert trainer.global_step == steps, 'Incomplete update budget'
            checkpoint = Path(params.checkpoint_dir) / 'training_checkpoint.pth'
        summary = export_noise_evaluation(
            trainer, output, checkpoint, test_size, evaluate_training=args.evaluate_training,
            optimizer_steps=None if args.checkpoint else trainer.global_step,
            evaluated_existing_checkpoint=args.checkpoint is not None)
        print(json.dumps(clean(summary), indent=2), flush=True)
    finally:
        trainer.cleanup()

if __name__ == '__main__':
    main()
