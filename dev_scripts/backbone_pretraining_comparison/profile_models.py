"""Same-host native training-core benchmark and partial profiler coverage audit."""

import gc
import json
from pathlib import Path
import statistics
import time

import torch
from ruamel.yaml import YAML

from fm4npp.models.mamba_config import build_backbone
from dev_scripts.backbone_pretraining_comparison.flops import describe
from dev_scripts.chinchilla_restart.policy import parameter_groups


def main():
    torch.set_num_threads(4)
    recipes = YAML(typ='safe').load(Path(__file__).with_name('recipe.yaml').read_text())
    output = {}
    for version in ['mamba1', 'mamba2', 'mamba3']:
        torch.manual_seed(42)
        params = recipes[version + '_chinchilla_w512_d12_k1']
        model = build_backbone(params).cuda().train()
        if version == 'mamba2':
            assert all(m.use_mem_eff_path for m in model.modules() if hasattr(m, '_mamba_config'))
        cost = describe(model, version)
        opt = torch.optim.AdamW(parameter_groups(model, params['weight_decay']), lr=0., betas=(.9, .95))
        x = torch.rand(64, 50, 3, device='cuda')
        target = torch.rand(64, 50, 3, device='cuda')

        def update():
            opt.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = (prediction - target).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), params['grad_clip_value'])
            opt.step()
            return loss

        for _ in range(10): update()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(30):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall = time.perf_counter()
            begin.record(); loss = update(); end.record()
            torch.cuda.synchronize()
            assert torch.isfinite(loss)
            samples.append({'gpu_ms': begin.elapsed_time(end), 'wall_ms': (time.perf_counter() - wall) * 1000})
        peak = torch.cuda.max_memory_allocated()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                   with_flops=True, record_shapes=True) as prof:
            update()
            torch.cuda.synchronize()
        events = prof.key_averages()
        covered = sum(event.flops or 0 for event in events)
        cost['benchmark'] = {'batch': 64, 'length': 50, 'warmup_updates': 10, 'measured_updates': 30,
                             'median_gpu_ms': statistics.median(s['gpu_ms'] for s in samples),
                             'median_wall_ms': statistics.median(s['wall_ms'] for s in samples),
                             'peak_allocated_bytes': peak, 'samples': samples,
                             'scope': 'forward + MSE + backward + value clip + AdamW; lr=0 keeps weights fixed; excludes data/validation/audit',
                             'initialization': 'fresh fixed-seed model; performance benchmark, not a loss evaluation'}
        cost['profiler'] = {'reported_flops': covered,
                            'expected_dense_arithmetic_flops': cost['dense_training_flops_per_padded_token'] * 64 * 50,
                            'warning': 'Profiler counts only supported operators; fused/custom kernels may be absent. Do not treat this as total model FLOPs.',
                            'counted_ops': [{'name': e.key, 'flops': e.flops, 'calls': e.count} for e in events if e.flops],
                            'custom_ops': [{'name': e.key, 'calls': e.count} for e in events if any(s in e.key.lower() for s in ['mamba', 'scan', 'conv1d', 'triton'])]}
        output[version] = cost
        print('PROFILED', version, cost['benchmark']['median_gpu_ms'], cost['estimated_training_flops_per_padded_token'], flush=True)
        del model, opt, x, target, loss, events, prof
        gc.collect(); torch.cuda.empty_cache()
    result = {'host': __import__('socket').gethostname(), 'gpu': torch.cuda.get_device_name(0),
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'models': output}
    Path('/results/compute_profile.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__': main()
