"""Isolated no-Mamba downstream campaign; repository defaults remain unchanged."""
import argparse
import copy
import csv
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'train/downstream'))
HERE = Path(__file__).resolve().parent
DATA = '/home/alessio/ML-work/HIPO_processing/multi_particle_storage/big-data/mmap_canonical_loose_truthseg_event_v6_02'
COHORTS = Path('/cohorts')
OUT = Path('/results')
# The original campaign snapshots retain zero. A separate adapter-only ablation
# opts into one historical Mamba2 refinement block in its own child process.
MOMENTUM_REFINEMENT_LAYERS = 0
QUEUES = {
    'a02c': [('finding', 'mamba1'), ('momentum', 'mamba3'), ('momentum', 'mamba1'), ('finding', 'mamba3')],
    'a518': [('finding', 'adapteronly'), ('momentum', 'mamba2'), ('finding', 'mamba2'), ('momentum', 'adapteronly')],
}
CHECKPOINTS = {
    v: f'/pretrained/{v}/training/{v}_chinchilla_w512_d12_k1/seed42/training_checkpoints/ckpt_best.tar'
    for v in ('mamba1', 'mamba2', 'mamba3')
}


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def yaml_write(path, value):
    with Path(path).open('w') as f: YAML().dump(value, f)


def model_digest(model):
    h = hashlib.sha256()
    for name, t in model.state_dict().items():
        h.update(name.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def recipe(task, backbone, run_dir, pilot):
    filename = ('mamba_clas12_track_finding_adapteronly.yaml' if task == 'finding'
                else 'mamba_clas12_track_regression_adapteronly.yaml')
    base = YAML(typ='safe').load((ROOT / 'scripts/configs' / filename).read_text())['default']
    params = dict(base)
    version = 'mamba1' if backbone == 'adapteronly' else backbone
    source = YAML(typ='safe').load((ROOT / 'dev_scripts/backbone_pretraining_comparison/recipe.yaml').read_text())[version + '_chinchilla_w512_d12_k1']
    for key in ('embed_dim', 'num_layers_backbone', 'd_state', 'd_conv', 'expand', 'klen',
                'embed_method', 'pe_method', 'headdim', 'ngroups', 'chunk_size',
                'rope_fraction', 'is_mimo', 'is_outproj_norm', 'use_mem_eff_path'):
        if key in source: params[key] = source[key]
    train_n = 256 if pilot else 50000
    val_n = 64 if pilot else (5000 if task == 'finding' else 10000)
    steps = 5 if pilot else 30000
    params.update(data_root=DATA, data_root_train=DATA, data_root_test=DATA,
        stat_dir=str(COHORTS), regression_target_stats=str(COHORTS / 'momentum_stats.json'),
        artifact_root=str(run_dir), downstream_dir=str(run_dir / 'train'), checkpoint_dir=str(run_dir / 'checkpoints'),
        backbone_version=version, mambaversion=version, adapter_version='mamba2',
        adapter_embed_dim=256, num_adapter_layers=MOMENTUM_REFINEMENT_LAYERS if task == 'momentum' else 0, num_embedder_layers=0,
        campaign_no_mamba_head=not (task == 'momentum' and MOMENTUM_REFINEMENT_LAYERS), coordinate_bypass=False, input_representation='center_only',
        batch_size=32 if task == 'finding' else 128, local_batch_size=32 if task == 'finding' else 128,
        valid_batch_size=32 if task == 'finding' else 128, local_valid_batch_size=32 if task == 'finding' else 128,
        limit_data=True, limit_size=train_n, limit_test_data=True, limit_test_size=val_n,
        max_validation_events=val_n, drop_last_test=False, max_val_batches=None,
        max_train_batches=None, max_epochs=10000, max_optimizer_steps=steps, total_steps=30000,
        val_interval_steps=steps if pilot else 1000, first_cycle_steps=30000,
        scheduler_first_cycle_steps=30000, scheduler_mode='cosine_restarts',
        warmup_steps=1000, max_lr=3e-4, min_lr=1e-5, adapter_weight_decay=1e-4,
        grad_clip_value=0.1, early_stopping_min_steps=30001, early_stopping_patience=100000,
        early_stopping_warmup_steps=30001, early_stopping_min_delta=0.0,
        num_data_workers=0 if pilot else 4, log_to_screen=True, use_amp=False, seed=42,
        dropout=0.1, downstream_dropout=0.1, serialization='clas12_band_hilbert',
        clas12_band_config='calibrated12', input_layout='xyz', space_filling_order=False,
        low_thr=1, high_thr=512 if task == 'finding' else 100,
        return_dict=True, return_knn_target=False, return_reg_test=True,
        return_coatjava_seg_pred=False, require_coatjava_seg_pred=False,
        checkpoint_selection='validation_loss', save_validation_checkpoints=False,
        model_version=task + '_' + backbone, num_prototypes=16, max_gt_classes=16 if task == 'finding' else 150,
        max_tracks=17 if task == 'finding' else 150,
        track_target_mode='signal_only', validation_ari_mode='signal',
        assignment_option=2, assignment_threshold=0.2, num_self_attn_layers=2,
        task='mom', pooling='attention', regression_loss='mae')
    return params


def install_runtime(task, stage, pilot):
    """Scope the cohort selection and zero-layer head to this child process."""
    import fm4npp.datasets.dataset as dataset_module
    role = 'evaluation' if stage == 'eval' else 'validation'
    def cached_filter(self, low_thr=-1, high_thr=1e10, max_tracks=150):
        selected_role = 'train' if self.train else role
        with np.load(COHORTS / f'{task}_{selected_role}.npz') as data:
            keys, lengths = data['keys'], data['lengths']
        if pilot:
            n = 256 if self.train else 64
            keys, lengths = keys[:n], lengths[:n]
        self.idxlist = [tuple(int(v) for v in row) for row in keys] if keys.ndim == 2 else [int(v) for v in keys]
        self.seqlens = lengths.tolist()
        self.tooshort, self.toolong, self.toomanytracks = [], [], []
        self.longest, self.shortest = int(lengths.max()), int(lengths.min())
        self.campaign_keys_sha256 = hashlib.sha256(keys.tobytes()).hexdigest()
        print('FIXED_COHORT', task, selected_role, len(keys), self.campaign_keys_sha256, flush=True)
    dataset_module.TPCBatchDataset.filter_data = cached_filter
    dataset_module.EventSegmentTPCBatchDataset.filter_data = cached_filter

    # The momentum trainer and evaluator hard-code num_layers=1. Override their
    # constructor only inside this campaign; assert the actual module graph.
    import model as head_module
    base = head_module.MambaTrackRegressionHead
    class NoMambaRegressionHead(base):
        def __init__(self, *args, **kwargs):
            kwargs.update(num_layers=0, num_embedder_layers=0)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(42)
                super().__init__(*args, **kwargs)
            assert MOMENTUM_REFINEMENT_LAYERS in (0, 1)
            if MOMENTUM_REFINEMENT_LAYERS:
                from fm4npp.models.mamba_config import build_mamba_block
                # Add the block after constructing the shared head so inserting
                # it cannot shift initialization of the MLP or coordinate path.
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(43)
                    block = torch.nn.Sequential(head_module.RMSNorm(self.embed_dim),
                        build_mamba_block('mamba2', d_model=self.embed_dim, d_state=64, d_conv=4, expand=2))
                self.mamba_layers.append(block)
            assert len(self.mamba_layers) == MOMENTUM_REFINEMENT_LAYERS and not self.mamba_embedder_layers
            assert sum(hasattr(m, '_mamba_config') for m in self.modules()) == MOMENTUM_REFINEMENT_LAYERS
    head_module.MambaTrackRegressionHead = NoMambaRegressionHead
    import trackinghead
    tracking_base = trackinghead.MambaAttentionHead
    class NoMambaTrackingHead(tracking_base):
        def __init__(self, *args, **kwargs):
            kwargs.update(num_layers=0, num_embedder_layers=0)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(42)
                super().__init__(*args, **kwargs)
            assert not self.mamba_layers and not self.mamba_embedder_layers
            assert not any(hasattr(m, '_mamba_config') for m in self.modules())
    trackinghead.MambaAttentionHead = NoMambaTrackingHead
    # Stable checkpoint metadata even though the common constructor name is Mamba*.
    trainer_module = importlib.import_module('track_finding_trainer' if task == 'finding' else 'track_regression_trainer')
    if task == 'momentum': trainer_module.MambaTrackRegressionHead = NoMambaRegressionHead
    else: trainer_module.MambaAttentionHead = NoMambaTrackingHead
    return trainer_module


def train(task, backbone, run_dir, pilot):
    module = install_runtime(task, 'train', pilot)
    params = recipe(task, backbone, run_dir, pilot)
    config_path = run_dir / 'model.yaml'; yaml_write(config_path, {'run': params})
    expected_steps = params['max_optimizer_steps']
    base = module.DownstreamTrainer
    audit = {}

    class AuditedTrainer(base):
        def launch(self):
            super().launch()
            self.model.requires_grad_(False); self.model.eval()
            self.backbone_digest = model_digest(self.model)
            self.audit_updates = 0
            self.audit_started = time.monotonic()
            audit.update(backbone_initial_sha256=self.backbone_digest,
                train_samples=len(self.train_data_loader.dataset), validation_samples=len(self.val_data_loader.dataset),
                train_keys_sha256=self.train_data_loader.dataset.campaign_keys_sha256,
                validation_keys_sha256=self.val_data_loader.dataset.campaign_keys_sha256)

        def ensure_audit(self):
            if getattr(self, 'audit_installed', False): return
            expected_blocks = MOMENTUM_REFINEMENT_LAYERS if task == 'momentum' else 0
            assert sum(hasattr(m, '_mamba_config') for m in self.down_model.modules()) == expected_blocks
            assert len(self.down_model.mamba_layers) == expected_blocks and not self.down_model.mamba_embedder_layers
            shared = hashlib.sha256()
            for name, tensor in self.down_model.state_dict().items():
                if name.startswith('mamba_layers.'): continue
                shared.update(name.encode()); shared.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            audit.update(head_mamba_blocks=expected_blocks, head_parameters=sum(p.numel() for p in self.down_model.parameters()),
                shared_initial_head_sha256=shared.hexdigest(),
                head_trainable_parameters=sum(p.numel() for p in self.down_model.parameters() if p.requires_grad),
                initial_head_sha256=model_digest(self.down_model))
            def before(opt, args, kwargs):
                grads = [p.grad for group in opt.param_groups for p in group['params'] if p.grad is not None]
                assert grads and torch.stack([torch.isfinite(g).all() for g in grads]).all().item(), 'Nonfinite or missing head gradients'
                assert all(p.grad is None for p in self.model.parameters()), 'Frozen backbone received gradients'
                if self.audit_updates == 0:
                    assert any(torch.count_nonzero(g).item() for g in grads), 'All head gradients zero'
            def after(opt, args, kwargs):
                self.audit_updates += 1
                if self.audit_updates <= 3 or self.audit_updates % 100 == 0:
                    write(run_dir / 'progress.json', {'task': task, 'backbone': backbone,
                        'step': self.audit_updates, 'total_steps': expected_steps,
                        'elapsed_seconds': time.monotonic() - self.audit_started})
            self.down_optimizer.register_step_pre_hook(before)
            self.down_optimizer.register_step_post_hook(after)
            self.audit_installed = True

        def _compute_batch_loss(self, *args, **kwargs):
            self.ensure_audit()
            result = super()._compute_batch_loss(*args, **kwargs)
            assert torch.isfinite(result[0]).item(), 'Nonfinite track-finding loss'
            return result

        def _train_one_batch(self, *args, **kwargs):
            self.ensure_audit()
            value = super()._train_one_batch(*args, **kwargs)
            assert math.isfinite(value), 'Nonfinite momentum loss'
            return value

        def train(self, *args, **kwargs):
            super().train(*args, **kwargs)
            assert self.global_step == self.audit_updates == expected_steps, (self.global_step, self.audit_updates)
            assert model_digest(self.model) == self.backbone_digest, 'Frozen backbone changed'
            checkpoint_path = Path(self.params.checkpoint_dir) / self.params.checkpoint_file_name
            checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{self.device}' if isinstance(self.device, int) else self.device, weights_only=False)
            self.down_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.down_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.down_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.down_model.eval()
            clone = copy.deepcopy(self.down_model)
            clone.load_state_dict(checkpoint['model_state_dict'], strict=True)
            batch = next(iter(self.val_data_loader))
            points = batch['points'].to(self.device)
            mask = points[..., 0] != -100
            with torch.no_grad():
                features = torch.stack(self.model(points, return_z=True)[1]) if backbone != 'adapteronly' else None
                a = self.down_model(points, features, pretrain=features is not None, padding_mask=mask)
                b = clone(points, features, pretrain=features is not None, padding_mask=mask)
                for key in ('class_probs', 'mask_probs') if task == 'finding' else ('pred_regression',):
                    assert torch.isfinite(a[key]).all().item()
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            audit.update(complete=True, finite_gradient_updates=self.audit_updates, final_step=self.global_step,
                elapsed_seconds=time.monotonic()-self.audit_started, backbone_unchanged=True,
                strict_head_reload=True, prediction_equivalence=True, optimizer_scheduler_restore=True,
                best_loss=float(self.best_loss), best_step=self.best_step, best_ari=float(self.best_ARI) if task == 'finding' else None,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), checkpoint=str(checkpoint_path))
            write(run_dir / 'training_complete.json', audit)

    module.DownstreamTrainer = AuditedTrainer
    experiment = importlib.import_module('train.downstream.track_finding_experiment' if task == 'finding' else 'train.downstream.track_regression_experiment')
    experiment.DownstreamTrainer = AuditedTrainer
    cls = experiment.TrackFindingExperimentConfig if task == 'finding' else experiment.TrackRegressionExperimentConfig
    cfg = cls(yaml_config=str(config_path), config='run', run_num=task+'_'+backbone,
        root_dir=str(run_dir / 'train'), eventnumber=params['limit_size'],
        usepretrain=backbone!='adapteronly', train_batch_size=params['batch_size'],
        pretrained_ckpt=CHECKPOINTS.get(backbone), checkpoint_dir=str(run_dir / 'checkpoints'),
        log_file_name='training.log', checkpoint_file_name='best.pth',
        artifact_summary=str(run_dir / 'artifacts.json'), resolved_config_path=str(run_dir / 'resolved.json'), seed=42)
    experiment.train_experiment(cfg)


def evaluate(task, backbone, run_dir, pilot):
    module = install_runtime(task, 'eval', pilot)
    base = module.DownstreamTrainer
    class StrictEvaluationTrainer(base):
        def launch(self):
            super().launch(); self.model.requires_grad_(False); self.model.eval()
        def load_checkpoint(self, path, inference=False):
            saved = torch.load(path, map_location=f'cuda:{self.device}' if isinstance(self.device, int) else self.device, weights_only=False)
            expected_blocks = MOMENTUM_REFINEMENT_LAYERS if task == 'momentum' else 0
            assert sum(hasattr(m, '_mamba_config') for m in self.down_model.modules()) == expected_blocks
            self.down_model.load_state_dict(saved['model_state_dict'], strict=True)
            return super().load_checkpoint(path, inference=inference)
    module.DownstreamTrainer = StrictEvaluationTrainer
    base_name = 'track_finding_analysis_adapteronly.yaml' if task == 'finding' else 'track_regression_analysis_adapteronly.yaml'
    analysis = YAML(typ='safe').load((ROOT / 'train/downstream/eval' / base_name).read_text())['analysis']
    n = 64 if pilot else (10000 if task == 'finding' else 50000)
    analysis.update(model_yaml=str(run_dir / 'model.yaml'), model_config='run',
        checkpoint=str(run_dir / 'checkpoints/best.pth'), training_log=str(run_dir / 'checkpoints/training.log'),
        output_dir=str(run_dir / 'evaluation'), run_name=task+'_'+backbone, run_num=task+'_'+backbone,
        campaign_name='backbone_downstream_no_mamba_20261029', analysis_tag=task+'_'+backbone,
        use_pretrained_backbone=backbone!='adapteronly', pretrained_checkpoint=CHECKPOINTS.get(backbone),
        max_samples=n, final_evaluation_events=n, batch_size=32 if task=='finding' else 128,
        num_workers=0 if pilot else 4, assignment_option=2, assignment_threshold=0.2,
        evaluate_coatjava=True, include_sample_metadata=True)
    yaml_write(run_dir / 'analysis.yaml', {'analysis': analysis})
    evaluator = importlib.import_module('train.downstream.eval.evaluate_track_finding' if task == 'finding' else 'train.downstream.eval.evaluate_track_regression')
    evaluator.DownstreamTrainer = StrictEvaluationTrainer
    sys.argv = [sys.argv[0], '--analysis-config', str(run_dir / 'analysis.yaml')]
    evaluator.main()
    summary = json.loads((run_dir / 'evaluation/summary.json').read_text())
    audit = {'complete': True, 'task': task, 'backbone': backbone,
        'expected_samples': n, 'cohort_role': 'evaluation', 'source_event_start': 100000}
    with np.load(COHORTS / f'{task}_evaluation.npz') as data:
        keys = data['keys'][:n]
    audit['evaluation_keys_sha256'] = hashlib.sha256(keys.tobytes()).hexdigest()
    if task == 'finding':
        assert summary['metrics']['n_events'] == n
        audit['actual_samples'] = summary['metrics']['n_events']
        audit['metrics'] = summary['metrics']
    else:
        with gzip.open(run_dir / 'evaluation/predictions.csv.gz', 'rt') as f:
            records = list(csv.DictReader(f))
        assert len(records) == n
        observed = np.asarray([(int(r['source_event_index']), int(r['segment_label'])) for r in records], dtype=np.int64)
        assert np.array_equal(observed, keys), 'Evaluation changed cohort identity/order'
        truth = np.asarray([[float(r['true_'+c+'_gev']) for c in ('px','py','pz')] for r in records], dtype=np.float64)
        assert np.isfinite(truth).all()
        audit.update(actual_samples=len(records), truth_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),
            adapter_valid_samples=summary['methods']['adapter']['n'])
    write(run_dir / 'evaluation_complete.json', {'complete': True, 'task': task, 'backbone': backbone,
        **audit})


def worker(host, pilot):
    OUT.mkdir(exist_ok=True)
    assert (COHORTS / 'data_ready.json').exists()
    if not pilot:
        gate = json.loads(Path('/pilot/pilot_complete.json').read_text())
        assert gate['complete'] and len(gate['runs']) == 4
    status = {'host': host, 'pilot': pilot, 'complete': False, 'runs': [], 'active': None}
    for task, backbone in QUEUES[host]:
        name = task+'_'+backbone
        run_dir = OUT / 'runs' / name; run_dir.mkdir(parents=True, exist_ok=True)
        entry = {'name': name, 'task': task, 'backbone': backbone, 'status': 'running'}
        status['runs'].append(entry); status['active'] = name; write(OUT/'status.json', status)
        for stage in ('train', 'eval'):
            entry['stage'] = stage; write(OUT/'status.json', status)
            cmd = [sys.executable, str(Path(__file__).resolve()), '--stage', stage, '--task', task,
                   '--backbone', backbone, '--run-dir', str(run_dir)] + (['--pilot'] if pilot else [])
            with (run_dir / (stage+'.stdout.log')).open('w') as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            entry[stage+'_exit_code'] = result.returncode
            if result.returncode:
                entry['status'] = 'failed'; write(OUT/'status.json', status)
                raise RuntimeError(f'{name} {stage} failed; see {run_dir}/{stage}.stdout.log')
        entry['status'] = 'complete'
        entry['training'] = json.loads((run_dir/'training_complete.json').read_text())
        entry['evaluation'] = json.loads((run_dir/'evaluation_complete.json').read_text())
        write(OUT/'status.json', status)
    status.update(complete=True, active=None)
    write(OUT/'status.json', status)
    write(OUT/('pilot_complete.json' if pilot else 'completed.json'), status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', choices=QUEUES)
    parser.add_argument('--stage', choices=('train', 'eval'))
    parser.add_argument('--task', choices=('finding', 'momentum'))
    parser.add_argument('--backbone', choices=('adapteronly', 'mamba1', 'mamba2', 'mamba3'))
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.host: worker(args.host, args.pilot)
    elif args.stage == 'train': train(args.task, args.backbone, args.run_dir, args.pilot)
    else: evaluate(args.task, args.backbone, args.run_dir, args.pilot)


if __name__ == '__main__': main()
