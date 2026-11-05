"""Noise adapter execution and checkpoint contracts."""
import hashlib
import os
from pathlib import Path
import random

import numpy as np
import torch


def require_single_process(task):
    if task == 'nid' and (int(os.environ.get('WORLD_SIZE', '1')) > 1 or
                          (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)):
        raise ValueError('Noise tagging currently supports single-process, single-GPU execution only')


def seed_everything(seed):
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def checkpoint_config(checkpoint):
    if 'training_config' in checkpoint:
        return dict(checkpoint['training_config'])
    raw = checkpoint.get('params', {})
    # Historical checkpoints saved vars(YParams), including a stale nested copy.
    return {**raw.get('params', {}), **{key: value for key, value in raw.items()
                                      if key != 'params' and not key.startswith('_')}}


def restore_evaluation_config(params, checkpoint, validation_size=None):
    """Use saved training settings, allowing data relocation and evaluation workers."""
    config = checkpoint_config(checkpoint)
    if config.get('task') != 'nid' or config.get('reader_type') != 'hdf5':
        raise ValueError('Expected an HDF5 noise-tagging checkpoint')
    locations = {key: params.params[key] for key in
                 ('data_root', 'data_root_train', 'data_root_test', 'stat_dir', 'num_data_workers')
                 if key in params.params}
    params.update_params(config)
    params.update_params(locations)
    params['validation_split'] = 'validation'
    if validation_size is not None:
        if validation_size < 1:
            raise ValueError('Validation event count must be positive')
        params['limit_test_size'] = validation_size
    params['return_coatjava_seg_pred'] = True
    params['return_coatjava_seg_pred_test'] = True


def validate_noise_data(checkpoint, data_contract, cohort_digest):
    config = checkpoint_config(checkpoint)
    if config.get('data_contract') != data_contract:
        raise ValueError('Noise checkpoint data contract does not match the configured data')
    saved = checkpoint.get('training_cohort_digest')
    if saved is not None and saved != cohort_digest:
        raise ValueError('Noise checkpoint training cohort does not match the configured data')


def training_contract(params):
    path = getattr(params, 'pretrained_ckpt', None)
    return dict(version=1, task='nid', mode='frozen' if path else 'adapteronly',
                backbone_sha256=file_sha256(path) if path else None)


def validate_noise_checkpoint(checkpoint, current):
    saved = checkpoint.get('noise_training_contract')
    if saved is None:
        config = checkpoint_config(checkpoint)
        if config.get('task') != 'nid' or 'pretrained_ckpt' not in config:
            raise ValueError('Noise checkpoint has no verifiable training mode')
        saved = dict(task='nid', mode='frozen' if config['pretrained_ckpt'] else 'adapteronly')
    if saved.get('task') != 'nid' or saved.get('mode') != current['mode']:
        raise ValueError(f"Noise checkpoint training mode mismatch: saved={saved.get('mode')}, current={current['mode']}")
    if current['mode'] == 'frozen':
        if not saved.get('backbone_sha256'):
            raise ValueError('Legacy frozen noise checkpoint has no backbone fingerprint; identity cannot be verified')
        if saved['backbone_sha256'] != current['backbone_sha256']:
            raise ValueError('Noise checkpoint requires a different pretrained backbone checkpoint')
