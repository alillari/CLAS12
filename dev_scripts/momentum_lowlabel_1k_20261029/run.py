"""Eight matched 1k-label momentum fits, each with a real-data CUDA pilot."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev_scripts.backbone_downstream_20261029 import run as common

HERE = Path(__file__).resolve().parent
OUT = Path('/results')
ARMS = [(v, layers) for v in ('adapteronly', 'mamba1', 'mamba2', 'mamba3') for layers in (0, 1)]
REFERENCE = json.loads((HERE / 'reference.json').read_text())


def read(path): return json.loads(Path(path).read_text())


def configure(layers):
    common.MOMENTUM_REFINEMENT_LAYERS = layers
    original_recipe = common.recipe
    def recipe(task, backbone, run_dir, pilot):
        p = original_recipe(task, backbone, run_dir, pilot)
        p.update(limit_size=256 if pilot else 1000, num_data_workers=0,
                 model_version=f'momentum_1k_{backbone}_m2ref{layers}')
        return p
    common.recipe = recipe
    original_yaml = common.yaml_write
    def yaml_write(path, data):
        if 'analysis' in data:
            data['analysis'].update(campaign_name='momentum_lowlabel_1k_20261029',
                analysis_tag=f'momentum_1k_m2ref{layers}')
        return original_yaml(path, data)
    common.yaml_write = yaml_write
    flags = {'fp32_head_forward_calls': 0, 'refinement_nonzero_gradient': False}
    original_install = common.install_runtime
    def install(task, stage, pilot):
        module = original_install(task, stage, pilot)
        original_parameters = module.legacy_initialization_parameters
        def parameters(model):
            yield from original_parameters(model)
            if not (hasattr(model, 'weighted_avg_weights') and hasattr(model, 'out_mlp')): return
            # Replace legacy random initialization with identical pre-update
            # weights. This bundle already has 1k-only normalization buffers.
            state = common.torch.load(f'/cohorts/initial_head_{layers}.pth', map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            ready = read('/cohorts/data_ready.json')
            assert common.model_digest(model) == ready['initial_heads'][str(layers)]['initial_head_sha256']
            assert all(p.dtype == common.torch.float32 for p in model.parameters())
            def fp32(_module, args):
                assert args[0].dtype == common.torch.float32
                flags['fp32_head_forward_calls'] += 1
            model.input_proj.register_forward_pre_hook(fp32)
            def gradient(g):
                if not flags['refinement_nonzero_gradient']:
                    assert common.torch.isfinite(g).all().item()
                    flags['refinement_nonzero_gradient'] = bool(common.torch.count_nonzero(g).item())
                return g
            for block in model.mamba_layers:
                block.register_forward_pre_hook(fp32)
                for p in block.parameters():
                    if p.requires_grad: p.register_hook(gradient)
        module.legacy_initialization_parameters = parameters
        return module
    common.install_runtime = install
    return flags


def verify(path, backbone, layers, pilot):
    t, e, cfg = (read(path / f) for f in ('training_complete.json', 'evaluation_complete.json', 'resolved.json'))
    ready = read('/cohorts/data_ready.json')
    assert t['complete'] and e['complete']
    assert t['finite_gradient_updates'] == t['final_step'] == (5 if pilot else 30000)
    assert t['train_samples'] == (256 if pilot else 1000)
    assert t['validation_samples'] == (64 if pilot else 10000)
    assert t['head_mamba_blocks'] == layers and not cfg['num_embedder_layers']
    assert t['head_parameters'] == (664744 if layers else 232720)
    assert t['head_trainable_parameters'] == (632488 if layers else 200464)
    for k in ('strict_head_reload', 'prediction_equivalence', 'optimizer_scheduler_restore', 'backbone_unchanged', 'head_fp32_execution'):
        assert t[k], k
    assert t['initial_head_sha256'] == ready['initial_heads'][str(layers)]['initial_head_sha256']
    assert t['shared_initial_head_sha256'] == ready['initial_heads'][str(layers)]['shared_initial_head_sha256']
    if pilot:
        import hashlib
        for role, field, count in [('train', 'train_keys_sha256', 256), ('validation', 'validation_keys_sha256', 64)]:
            with common.np.load(f'/cohorts/momentum_{role}.npz') as cohort:
                assert t[field] == hashlib.sha256(cohort['keys'][:count].tobytes()).hexdigest()
    if layers: assert t['refinement_nonzero_gradient']
    if backbone != 'adapteronly':
        assert t['backbone_initial_sha256'] == REFERENCE['backbone_training'][backbone]['backbone_initial_sha256']
    assert e['actual_samples'] == e['adapter_valid_samples'] == (64 if pilot else 50000)
    if not pilot:
        assert t['train_keys_sha256'] == ready['train_keys_sha256']
        assert t['validation_keys_sha256'] == ready['validation_keys_sha256']
        for key in ('evaluation_keys_sha256', 'truth_sha256'):
            assert e[key] == REFERENCE['evaluation'][key], key
        for key in ('batch_size', 'valid_batch_size', 'max_optimizer_steps', 'max_lr', 'min_lr', 'warmup_steps',
                    'scheduler_first_cycle_steps', 'adapter_weight_decay', 'grad_clip_value', 'dropout',
                    'regression_loss', 'pooling', 'embed_dim', 'adapter_embed_dim', 'seed', 'use_amp', 'adapter_version', 'coordinate_bypass'):
            assert cfg[key] == REFERENCE['baseline_config'][key], key
    return {'name': path.name, 'backbone': backbone, 'refinement_layers': layers, 'status': 'complete',
        'training': t, 'evaluation': e}


def worker():
    ready = read('/cohorts/data_ready.json')
    assert ready['complete'] and ready['training_tracks'] == 1000
    from dev_scripts.momentum_lowlabel_1k_20261029.prepare import file_sha
    imported = read(HERE / 'checkpoint_expected.json')
    m2 = Path(common.CHECKPOINTS['mamba2'])
    while not m2.exists():
        common.write(OUT / 'status.json', {'host': 'a02c', 'complete': False,
            'phase': 'waiting_for_checkpoint_transfer', 'active': None,
            'expected_checkpoint_sha256': imported['sha256'], 'runs': []})
        time.sleep(15)
    assert m2.stat().st_size == imported['bytes']
    assert file_sha(m2) == imported['sha256']
    imported.update(complete=True, verified_before_pilot=True)
    ready['checkpoints']['mamba2'] = {'sha256': imported['sha256'], 'bytes': m2.stat().st_size, 'verified_before_pilot': True}
    common.write(OUT / 'checkpoint_import.json', imported)
    status = {'host': 'a02c', 'training_tracks': 1000, 'complete': False, 'phase': 'pilot', 'active': None, 'runs': []}
    common.write(OUT / 'data_ready.json', ready)
    for pilot in (True, False):
        entries = []
        status['phase'] = 'pilot' if pilot else 'production'
        if not pilot:
            gate = read(OUT / 'pilot_complete.json')
            assert gate['complete'] and len(gate['runs']) == 8
        for backbone, layers in ARMS:
            name = f'momentum_{backbone}_m2ref{layers}'
            path = OUT / ('pilot' if pilot else 'runs') / name
            path.mkdir(parents=True, exist_ok=True)
            status.update(active=name, stage='train')
            common.write(OUT / 'status.json', status)
            for stage in ('train', 'eval'):
                status['stage'] = stage; common.write(OUT / 'status.json', status)
                cmd = [sys.executable, str(Path(__file__).resolve()), '--stage', stage,
                    '--backbone', backbone, '--layers', str(layers), '--run-dir', str(path)]
                if pilot: cmd.append('--pilot')
                with (path / (stage + '.stdout.log')).open('w') as log:
                    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
                if r.returncode:
                    status.update(failed=True, exit_code=r.returncode)
                    common.write(OUT / 'status.json', status)
                    raise RuntimeError(f'{name} {stage} failed in {status["phase"]}; inspect its stdout')
            entry = verify(path, backbone, layers, pilot)
            entries.append(entry)
            status['runs'] = entries
            common.write(OUT / 'status.json', status)
        if pilot:
            for field in ('evaluation_keys_sha256', 'truth_sha256'):
                assert len({r['evaluation'][field] for r in entries}) == 1, field
            common.write(OUT / 'pilot_complete.json', {'complete': True, 'runs': entries})
            status['runs'] = []
    status.update(complete=True, phase='complete', active=None, stage=None)
    common.write(OUT / 'status.json', status)
    common.write(OUT / 'completed.json', status)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('train', 'eval'))
    p.add_argument('--backbone', choices=('adapteronly', 'mamba1', 'mamba2', 'mamba3'))
    p.add_argument('--layers', type=int, choices=(0, 1))
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    common.torch.set_num_threads(4)
    if args.stage:
        flags = configure(args.layers)
        if args.stage == 'train':
            common.train('momentum', args.backbone, args.run_dir, args.pilot)
            assert flags['fp32_head_forward_calls'] and (not args.layers or flags['refinement_nonzero_gradient'])
            path = args.run_dir / 'training_complete.json'
            t = read(path); t.update(**flags, head_fp32_execution=True, normalization_training_tracks=1000)
            common.write(path, t)
        else: common.evaluate('momentum', args.backbone, args.run_dir, args.pilot)
    else: worker()


if __name__ == '__main__': main()
