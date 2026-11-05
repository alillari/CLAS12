#!/usr/bin/env python3
"""Train a point-classification adapter using the YAML recipe and explicit overrides."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm4npp.utils import YParams
from fm4npp.artifact_paths import artifact_path
from train.downstream.noise_checkpoint import require_single_process, file_sha256


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml_config', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run_num', default='0')
    parser.add_argument('--root_dir', default=artifact_path('downstream_log'))
    parser.add_argument('--global_log_dir', default='globallogs')
    parser.add_argument('--eventnumber', type=int, help='Override training sample count')
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--usepretrain', action='store_true', help='Require a pretrained backbone')
    parser.add_argument('--pretrained_ckpt', help='Backbone checkpoint; implies frozen-backbone training')
    parser.add_argument('--task', choices=['pid', 'nid'])
    return parser.parse_args(argv)


def configure_training(params, args):
    params['task'] = args.task or getattr(params, 'task', 'pid')
    require_single_process(params.task)
    params['num_output_classes'] = 2 if params.task == 'nid' else 5
    for arg, key in [('eventnumber', 'limit_size'), ('train_batch_size', 'batch_size'), ('seed', 'seed')]:
        value = getattr(args, arg)
        if value is not None:
            params[key] = value
    if args.eventnumber is not None:
        params['limit_data'] = True
    params['seed'] = getattr(params, 'seed', 42)
    params['pretrained_ckpt'] = args.pretrained_ckpt or getattr(params, 'pretrained_ckpt', None)
    if args.usepretrain and not params.pretrained_ckpt:
        raise ValueError('--usepretrain requires --pretrained_ckpt or pretrained_ckpt in YAML')
    params['continue_from_best'] = False  # Public CLI always starts a fresh adapter.
    if not getattr(params, 'log_file_name', None):
        params['log_file_name'] = f'{args.config}_{params.task}_{args.run_num}.log'
    return params


def main(argv=None):
    args = parse_args(argv)
    params = configure_training(YParams(os.path.abspath(args.yaml_config), args.config), args)
    from train.downstream.point_classification_trainer import DownstreamTrainer
    trainer = DownstreamTrainer(params, args)
    try:
        trainer.launch()
        trainer.train(pretrain=bool(params.pretrained_ckpt), train_from_checkpoint=False)
        if params.task == 'nid':
            from train.downstream.hdf5_benchmark_utils import write_json
            write_json(Path(params.checkpoint_dir) / 'training_run.json', dict(
                checkpoint=trainer.best_checkpoint_path,
                checkpoint_sha256=file_sha256(trainer.best_checkpoint_path),
                optimizer_steps=getattr(trainer, 'global_step', None),
                training_contract=trainer.noise_training_contract,
                training_config=params.params,
            ))
    finally:
        trainer.cleanup()


if __name__ == '__main__':
    main()
