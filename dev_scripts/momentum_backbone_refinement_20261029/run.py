"""Matched frozen-backbone momentum fits with the existing Mamba2 head."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev_scripts.backbone_downstream_20261029 import run as common

HERE = Path(__file__).resolve().parent
REFERENCE = json.loads((HERE / 'reference.json').read_text())
OUT = Path('/results')
QUEUES = {'a02c': ['mamba1', 'mamba3'], 'a518': ['mamba2']}
RECIPE_KEYS = ('batch_size', 'valid_batch_size', 'limit_size', 'limit_test_size', 'seed',
    'max_optimizer_steps', 'max_lr', 'min_lr', 'warmup_steps', 'scheduler_first_cycle_steps',
    'adapter_weight_decay', 'grad_clip_value', 'dropout', 'regression_loss', 'pooling',
    'embed_dim', 'adapter_embed_dim', 'num_adapter_layers', 'num_embedder_layers', 'use_amp',
    'adapter_version', 'coordinate_bypass')


class InitialStateCaptured(Exception):
    pass


def install_initialization(capture=False):
    """Leave core code intact; pin the complete head after its legacy initializer."""
    original_install = common.install_runtime
    flags = {'head_fp32_forward_calls': 0, 'refinement_nonzero_gradient': False}

    def install(task, stage, pilot):
        module = original_install(task, stage, pilot)
        original_parameters = module.legacy_initialization_parameters

        def parameters(model):
            yield from original_parameters(model)
            if not (hasattr(model, 'weighted_avg_weights') and hasattr(model, 'out_mlp')):
                return
            if capture:
                digest = common.model_digest(model)
                assert digest == REFERENCE['control_training']['initial_head_sha256'], digest
                common.torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                  OUT / 'initial_head.pth')
                common.write(OUT / 'initial_head.json', {'complete': True, 'optimizer_updates': 0,
                    'initial_head_sha256': digest, 'source': REFERENCE['control_job'],
                    'method': 'Exact reconstruction before the first update; checked against completed control audit'})
                raise InitialStateCaptured()
            state = common.torch.load('/initial/initial_head.pth', map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            assert common.model_digest(model) == REFERENCE['control_training']['initial_head_sha256']
            assert all(p.dtype == common.torch.float32 for p in model.parameters())

            def check_fp32(_module, args):
                assert args[0].dtype == common.torch.float32, args[0].dtype
                flags['head_fp32_forward_calls'] += 1

            def check_gradient(gradient):
                if not flags['refinement_nonzero_gradient']:
                    assert common.torch.isfinite(gradient).all().item()
                    flags['refinement_nonzero_gradient'] = bool(common.torch.count_nonzero(gradient).item())
                return gradient

            for block in model.mamba_layers:
                block.register_forward_pre_hook(check_fp32)
                for p in block.parameters():
                    if p.requires_grad: p.register_hook(check_gradient)

        module.legacy_initialization_parameters = parameters
        return module

    common.install_runtime = install
    return flags


def verify(run_dir, version, pilot):
    training = json.loads((run_dir / 'training_complete.json').read_text())
    evaluation = json.loads((run_dir / 'evaluation_complete.json').read_text())
    config = json.loads((run_dir / 'resolved.json').read_text())
    target = REFERENCE['pilot_training'] if pilot else REFERENCE['control_training']
    assert training['complete'] and evaluation['complete']
    assert training['final_step'] == training['finite_gradient_updates'] == (5 if pilot else 30000)
    assert training['head_mamba_blocks'] == 1
    assert training['head_parameters'] == 664744 and training['head_trainable_parameters'] == 632488
    assert training['initial_head_sha256'] == target['initial_head_sha256']
    assert training['shared_initial_head_sha256'] == target['shared_initial_head_sha256']
    for key in ('train_keys_sha256', 'validation_keys_sha256', 'train_samples', 'validation_samples'):
        assert training[key] == target[key], key
    for key in ('backbone_unchanged', 'strict_head_reload', 'prediction_equivalence', 'optimizer_scheduler_restore',
                'refinement_nonzero_gradient', 'head_fp32_execution'):
        assert training[key], key
    assert training['backbone_initial_sha256'] == REFERENCE['backbone_training'][version]['backbone_initial_sha256']
    assert evaluation['actual_samples'] == evaluation['adapter_valid_samples'] == (64 if pilot else 50000)
    if not pilot:
        for key in ('evaluation_keys_sha256', 'truth_sha256'):
            assert evaluation[key] == REFERENCE['control_evaluation'][key], key
        for key in RECIPE_KEYS:
            assert config[key] == REFERENCE['control_config'][key], (key, config[key])
    return {'name': 'momentum_' + version, 'backbone': version, 'status': 'complete',
        'training': training, 'evaluation': evaluation}


def worker(host, pilot):
    assert (Path('/cohorts') / 'data_ready.json').exists()
    assert json.loads(Path('/initial/initial_head.json').read_text())['complete']
    if not pilot:
        gate = json.loads(Path('/pilot/pilot_complete.json').read_text())
        assert gate['complete'] and {r['backbone'] for r in gate['runs']} == set(QUEUES[host])
    status = {'host': host, 'pilot': pilot, 'complete': False, 'active': None, 'runs': []}
    for version in QUEUES[host]:
        name = 'momentum_' + version
        run_dir = OUT / 'runs' / name
        run_dir.mkdir(parents=True, exist_ok=True)
        status.update(active=name, stage='train')
        common.write(OUT / 'status.json', status)
        for stage in ('train', 'eval'):
            status['stage'] = stage
            common.write(OUT / 'status.json', status)
            command = [sys.executable, str(Path(__file__).resolve()), '--stage', stage,
                       '--backbone', version, '--run-dir', str(run_dir)]
            if pilot: command.append('--pilot')
            with (run_dir / (stage + '.stdout.log')).open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                status.update(stage=stage, failed=True, exit_code=result.returncode)
                common.write(OUT / 'status.json', status)
                raise RuntimeError(f'{name} {stage} failed; inspect its stdout log')
        status['runs'].append(verify(run_dir, version, pilot))
        common.write(OUT / 'status.json', status)
    status.update(complete=True, active=None, stage=None)
    common.write(OUT / 'status.json', status)
    common.write(OUT / ('pilot_complete.json' if pilot else 'completed.json'), status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', choices=QUEUES)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--capture-initial', action='store_true')
    parser.add_argument('--stage', choices=('train', 'eval'))
    parser.add_argument('--backbone', choices=('mamba1', 'mamba2', 'mamba3'))
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    common.MOMENTUM_REFINEMENT_LAYERS = 1
    common.torch.set_num_threads(4)
    if args.capture_initial:
        scratch = OUT / 'reconstruction'
        scratch.mkdir(parents=True, exist_ok=True)
        install_initialization(capture=True)
        try: common.train('momentum', 'adapteronly', scratch, True)
        except InitialStateCaptured: pass
        assert (OUT / 'initial_head.json').exists()
    elif args.stage:
        flags = install_initialization()
        if args.stage == 'train':
            common.train('momentum', args.backbone, args.run_dir, args.pilot)
            assert flags['head_fp32_forward_calls'] > 0 and flags['refinement_nonzero_gradient']
            path = args.run_dir / 'training_complete.json'
            audit = json.loads(path.read_text())
            audit.update(**flags, head_fp32_execution=True, matched_complete_initial_head=True)
            common.write(path, audit)
        else: common.evaluate('momentum', args.backbone, args.run_dir, args.pilot)
    else: worker(args.host, args.pilot)


if __name__ == '__main__': main()
