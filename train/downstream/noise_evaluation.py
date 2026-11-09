"""Shared validation calibration and held-out export for noise adapters."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from ruamel.yaml import YAML

from fm4npp.datasets.dataset import get_val_loader
from train.downstream.input_representation import representation_input
from train.downstream.hdf5_benchmark_utils import clean, write_json, cohort, event_keys
from train.downstream.noise_checkpoint import checkpoint_config, file_sha256
from train.downstream.noise_metrics import at_threshold, calibrate_thresholds, score_summary


def cohort_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@torch.no_grad()
def evaluate(trainer, loader, output, split='test', thresholds=None):
    trainer.down_model.eval()
    trainer.model.eval()
    records = []
    labels_all, scores_all, coat_all, offsets = [], [], [], [0]
    for batch in loader:
        points = batch['points'].to(trainer.device)
        mask = points[..., 0] != -100
        model_input = representation_input(points, batch, getattr(trainer, 'params', None))
        if trainer.noise_training_contract['mode'] == 'frozen':
            _, embeddings, _ = trainer.model(model_input, return_z=True)
            pred = trainer.down_model(model_input, torch.stack(embeddings), pretrain=True, padding_mask=mask)
        else:
            pred = trainer.down_model(model_input, feature=None, padding_mask=mask)
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
    if not labels_all:
        raise ValueError(f'No accepted events in {split}')
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


def export_noise_evaluation(trainer, output, checkpoint, test_size, *, evaluate_training=False,
                            optimizer_steps=None, evaluated_existing_checkpoint=True):
    """Reload the head, calibrate on validation, and export fixed-threshold test metrics.

    Training provenance comes from the checkpoint, never inference CLI overrides.
    New checkpoints also verify the reconstructed training cohort at load time.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise FileExistsError(f'Evaluation already exists: {output}')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint_config(saved)
    checkpoint_sha256 = file_sha256(checkpoint)
    receipt = Path(checkpoint).parent / 'training_run.json'
    if optimizer_steps is None and receipt.exists():
        completed = json.loads(receipt.read_text())
        if completed.get('checkpoint_sha256') == checkpoint_sha256:
            optimizer_steps = completed.get('optimizer_steps')
    counts = saved.get('class_counts', config.get('noise_class_counts'))
    trainer.load_checkpoint(str(checkpoint), inference=True)
    cohorts = dict(train=cohort(trainer.train_data_loader.dataset),
                   validation=cohort(trainer.val_data_loader.dataset))
    if config.get('limit_data') and len(cohorts['train']) != config['limit_size']:
        raise ValueError('Reconstructed training cohort differs from the saved training budget')
    test_params = deepcopy(trainer.params)
    test_params['data_root'] = trainer.params.data_root_test
    test_params['limit_test_data'] = True
    test_params['limit_test_size'] = int(test_size)
    test_params['test_split'] = 'test'
    loader = get_val_loader(test_params, False)
    cohorts['test'] = cohort(loader.dataset)
    if trainer.val_data_loader.dataset.split != 'validation' or loader.dataset.split != 'test':
        raise ValueError('Noise calibration requires validation and held-out test splits')
    if not all(cohorts.values()):
        raise ValueError('Train, validation and test cohorts must all be nonempty')
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if event_keys(cohorts[left]) & event_keys(cohorts[right]):
            raise ValueError(f'{left}/{right} event overlap')
    if len(cohorts['test']) != test_size:
        raise ValueError('Requested test cohort was not fully accepted')
    write_json(output / 'resolved_config.json', config)
    with (output / 'resolved_config.yaml').open('w') as stream:
        YAML().dump({trainer.config: clean(config)}, stream)
    write_json(output / 'cohorts.json', cohorts)
    write_json(output / 'evaluation_config.json', dict(
        runtime=trainer.params.params, test_runtime=test_params.params))
    validation = evaluate(trainer, trainer.val_data_loader, output, split='validation')
    if evaluate_training:
        # An evaluation loader must not shuffle or drop training events.
        from torch.utils.data import DataLoader
        train_loader = DataLoader(trainer.train_data_loader.dataset,
                                  batch_size=trainer.params.local_valid_batch_size,
                                  collate_fn=trainer.train_data_loader.collate_fn,
                                  num_workers=trainer.params.num_data_workers, shuffle=False)
        evaluate(trainer, train_loader, output, split='training')
    summary = evaluate(trainer, loader, output, thresholds=validation['calibration'])
    summary.pop('threshold_curve')
    summary.update(
        task='noise', training_mode=trainer.noise_training_contract['mode'],
        backbone_checkpoint_sha256=trainer.noise_training_contract['backbone_sha256'],
        optimizer_steps=optimizer_steps,
        selected_checkpoint_step=trainer.loaded_checkpoint_step,
        evaluated_existing_checkpoint=evaluated_existing_checkpoint,
        training_class_counts=counts,
        class_weights=None if trainer.class_weights is None else trainer.class_weights.cpu().tolist(),
        seed=config['seed'], train_samples=len(cohorts['train']),
        validation_samples=len(cohorts['validation']), test_samples=len(cohorts['test']),
        data_contract=trainer.params.data_contract, checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=checkpoint_sha256, sample_unit='event',
        validation_selection='loss', test_split='test',
        adapter_parameters=sum(p.numel() for p in trainer.down_model.parameters()),
        training_cohort_verified=saved.get('training_cohort_digest') is not None,
    )
    write_json(output / 'summary.json', summary)
    return summary
