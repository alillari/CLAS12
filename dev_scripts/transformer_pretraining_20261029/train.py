"""Matched Transformer pretraining, reusing the completed Mamba training loop."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from ruamel.yaml import YAML

from dev_scripts.backbone_pretraining_comparison import train as reference
from dev_scripts.backbone_pretraining_comparison.flops import describe
from dev_scripts.chinchilla_restart.policy import group_audit
from fm4npp.models.mamba_config import architecture_metadata
from fm4npp.utils import YParams

HERE = Path(__file__).resolve().parent


def digest(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        value.update(name.encode()); value.update(str(array.dtype).encode())
        value.update(str(array.shape).encode()); value.update(array.tobytes())
    return value.hexdigest()


def source_digest():
    root = HERE.parents[1]
    files = [*HERE.glob('*.py'), HERE / 'recipe.yaml',
             root / 'dev_scripts/chinchilla_restart/trainer.py',
             root / 'dev_scripts/chinchilla_restart/policy.py',
             root / 'dev_scripts/backbone_pretraining_comparison/train.py',
             root / 'dev_scripts/backbone_pretraining_comparison/targets.py',
             root / 'fm4npp/models/transformer.py', root / 'fm4npp/models/mamba_config.py',
             root / 'fm4npp/models/embed.py', root / 'fm4npp/datasets/dataset_pretrain.py']
    return hashlib.sha256(''.join(str(p.relative_to(root)) + reference.sha(p)
                                  for p in sorted(files)).encode()).hexdigest()


def model_contract(model, dataset):
    # Fork RNG state so these checks do not change the paired dropout streams.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        x = dataset[0][0][None].cuda()
        model.eval()
        with torch.no_grad():
            expected = model(x)
            cut = max(1, x.shape[1] // 2)
            changed = x.clone(); changed[:, cut:] = .123
            torch.testing.assert_close(model(changed)[:, :cut], expected[:, :cut], rtol=1e-5, atol=1e-6)
            padded = torch.nn.functional.pad(x, (0, 0, 0, 3), value=-100.)
            torch.testing.assert_close(model(padded)[:, :x.shape[1]], expected, rtol=1e-5, atol=1e-6)
            assert torch.count_nonzero(model(padded)[:, x.shape[1]:]) == 0
        model.train()
    return {'causal_prefix': 'passed', 'padding_invariance': 'passed'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoding', choices=['nope', 'rope'], required=True)
    parser.add_argument('--phase', choices=['pilot', 'train'], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    config = f'transformer_{args.encoding}_w512_d12_k1'
    reference.OUT, reference.RECIPE = out, HERE / 'recipe.yaml'
    if args.phase == 'train':
        gate = json.loads(Path('/pilot/pilot_complete.json').read_text())
        assert gate['source_digest'] == source_digest(), 'Pilot/source mismatch'
        assert gate['passed'] is True
    params = YParams(str(reference.RECIPE), config)
    params.stat_dir = str(out / 'runtime_stats'); Path(params.stat_dir).mkdir(exist_ok=True)
    for name, values in [('loss_bin_pp.pkl', np.array([0., 1e9])), ('loss_weight_pp.pkl', np.array([1.]))]:
        with (Path(params.stat_dir) / name).open('wb') as stream: pickle.dump(values, stream)
    if args.phase == 'pilot':
        params.total_steps = 2000
        params.warmup_steps = 200
        # Same complete cohorts; compressed schedule reaches the production peak.
    random.seed(params.seed); np.random.seed(params.seed)
    torch.manual_seed(params.seed); torch.cuda.manual_seed_all(params.seed)
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    assert not torch.is_autocast_enabled()
    reference.shared.get_data_loader = reference.make_loaders
    resolved = {k: v for k, v in vars(params).items() if not k.startswith('_') and k != 'params'}
    with (out / 'resolved_recipe.yaml').open('w') as f: YAML().dump({config: resolved}, f)
    start = time.monotonic()
    trainer = reference.AuditedTrainer(params, SimpleNamespace(root_dir=str(out / 'training'), config=config, run_num='seed42'))
    trainer.audit_started = time.monotonic()
    assert trainer.iters == 0, 'Each arm starts fresh'
    blocks = architecture_metadata(trainer.model)['blocks']
    assert len(blocks) == 12 and all(c['version'] == 'transformer' and c['causal'] for c in blocks.values())
    assert all(p.dtype == torch.float32 for p in trainer.model.parameters())
    assert sum(p.numel() for p in trainer.model.parameters()) == 19339651
    assert not trainer.model.embedder.embed.projection.requires_grad
    assert reference.sha(HERE.parent / 'backbone_pretraining_comparison/targets.py') == json.loads((HERE / 'mamba_reference.json').read_text())['models'][0]['completed']['targets_sha256']
    refs = json.loads((HERE / 'mamba_reference.json').read_text())['models']
    cohorts = json.loads((out / 'cohorts.json').read_text())
    for ref in refs:
        for split in ['training', 'validation']:
            assert cohorts[split]['indices_sha256'] == ref['cohorts'][split]['indices_sha256'], (ref['version'], split)
    ckpt = torch.load('/reference_mamba2/training/mamba2_chinchilla_w512_d12_k1/seed42/training_checkpoints/ckpt_best.tar', map_location='cpu', weights_only=False)
    projection = trainer.model.embedder.embed.projection
    torch.testing.assert_close(projection.cpu(), ckpt['model_state']['embedder.embed.projection'], rtol=0, atol=0)
    del ckpt; gc.collect()
    cost = describe(trainer.model, 'transformer')
    initial = digest(trainer.model.state_dict())
    audit = dict(phase=args.phase, encoding=args.encoding, source_digest=source_digest(),
                 initial_state_sha256=initial,
                 embedding_sha256=digest({'projection': projection}),
                 sampler_generator_sha256=digest({'generator': trainer.train_data_loader.generator.get_state()}),
                 target_audit=reference.target_audit(trainer.train_data_loader.dataset),
                 model_contract=model_contract(trainer.model, trainer.train_data_loader.dataset),
                 architecture=architecture_metadata(trainer.model), parameter_count=cost['parameters'],
                 trainable_parameter_count=cost['trainable_parameters'], compute=cost,
                 precision='FP32 parameters, attention, gradients and loss; no autocast',
                 train_events=params.train_events, validation_events=params.validation_events,
                 effective_batch_size=params.local_batch_size, effective_validation_events=2000,
                 total_steps=params.total_steps, seed=params.seed,
                 optimizer={'name':'AdamW', 'betas':[.9,.95], 'width_ratio':trainer.mup_matrix_ratio,
                            'groups':group_audit(trainer.model, trainer.optimizer)},
                 torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                 validation_boundary='Same checkpoint-selection cohort as Mamba; not independent final evaluation.',
                 resume_boundary='Fresh run. Existing epoch-level resume does not preserve exact sampler cursor.')
    reference.write_json(out / 'started.json', audit)
    trainer.launch()
    assert trainer.iters == params.total_steps == trainer.audited_steps
    audit.update(final_step=trainer.iters, finite_gradient_updates=trainer.audited_steps,
                 best_validation_mse=float(trainer.best_loss), elapsed_seconds=time.monotonic()-start,
                 checkpoint_reload=reference.reload_check(trainer),
                 peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    reference.write_json(out / 'completed.json', audit)
    print('ARM_COMPLETE ' + json.dumps({k:audit[k] for k in ['encoding','final_step','best_validation_mse','elapsed_seconds']}), flush=True)


if __name__ == '__main__': main()
