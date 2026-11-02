"""Counterbalanced model/precision benchmark, preserving native training runs."""

import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
from ruamel.yaml import YAML

from fm4npp.models.mamba_config import build_backbone
from dev_scripts.chinchilla_restart.policy import parameter_groups


CONDITIONS = ['mamba2_fp32', 'mamba2_bf16_blocks', 'mamba3_bf16_blocks']


def telemetry():
    argv = ['nvidia-smi', '--query-gpu=uuid,name,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw',
            '--format=csv,noheader,nounits']
    result = subprocess.run(argv, capture_output=True, text=True)
    return {'values': result.stdout.strip(), 'error': result.stderr.strip() if result.returncode else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-for')
    args = parser.parse_args()
    if args.wait_for:
        root = Path(args.wait_for)
        while not (root / 'runner_result.json').exists():
            Path('/results/phase.json').write_text(json.dumps({'phase': 'waiting_for_training_to_exit'}) + '\n')
            time.sleep(30)
        # The supervisor writes its result at the end of child/worker shutdown.
        time.sleep(5)
        # Do not benchmark against another task, or immediately after a hot run.
        while True:
            processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                       capture_output=True, text=True)
            temperature = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                                        capture_output=True, text=True)
            idle = processes.returncode == 0 and not processes.stdout.strip()
            cool = temperature.returncode == 0 and all(float(v) <= 50 for v in temperature.stdout.splitlines())
            if idle and cool: break
            Path('/results/phase.json').write_text(json.dumps({'phase': 'waiting_for_idle_cool_gpu',
                 'gpu_processes': processes.stdout.strip(), 'temperature': temperature.stdout.strip()}) + '\n')
            time.sleep(30)
    torch.set_num_threads(4)
    recipes = YAML(typ='safe').load(Path(__file__).with_name('recipe.yaml').read_text())
    rounds = [CONDITIONS, list(reversed(CONDITIONS)), [CONDITIONS[1], CONDITIONS[0], CONDITIONS[2]]]
    report = {'hostname': __import__('socket').gethostname(), 'gpu': torch.cuda.get_device_name(0),
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'batch': 64, 'length': 50,
              'warmup_updates': 10, 'measured_updates_per_round': 30, 'rounds': [],
              'scope': 'fresh fixed-seed models, shared inputs, zero-LR AdamW; forward/MSE/backward/value-clip/update; no data loading or validation',
              'bf16_boundary': 'SSM blocks only, FP32 parameters and caller outputs; identical boundary for Mamba2 and Mamba3',
              'limitations': ['Microbenchmark is not convergence/accuracy evidence.',
                              'Host temperature and clocks may drift; orders are counterbalanced.',
                              'Mamba3 FP32 kernel execution is unsupported by this wrapper.']}
    for round_index, order in enumerate(rounds):
        for condition in order:
            version = 'mamba2' if condition.startswith('mamba2') else 'mamba3'
            params = recipes[version + '_chinchilla_w512_d12_k1']
            torch.manual_seed(42)
            model = build_backbone(params).cuda().train()
            blocks = [m for m in model.modules() if hasattr(m, '_mamba_config')]
            assert len(blocks) == 12
            if version == 'mamba2': assert all(m.use_mem_eff_path for m in blocks)
            if condition == 'mamba2_bf16_blocks':
                for block in blocks:
                    original = block.forward
                    def wrapped(u, *a, _forward=original, **kw):
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            result = _forward(u, *a, **kw)
                        return result.to(dtype=u.dtype)
                    block.forward = wrapped
            # Use identical input/target values, independent of model RNG use.
            generator = torch.Generator(device='cuda').manual_seed(123)
            x = torch.rand((64, 50, 3), generator=generator, device='cuda')
            target = torch.rand((64, 50, 3), generator=generator, device='cuda')
            opt = torch.optim.AdamW(parameter_groups(model, params['weight_decay']), lr=0., betas=(.9, .95))
            seen = []
            handle = blocks[0].in_proj.register_forward_hook(lambda m, a, out: seen.append(str(out.dtype)))

            def update():
                opt.zero_grad(set_to_none=True)
                loss = (model(x) - target).square().mean()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), params['grad_clip_value'])
                opt.step()
                return loss

            loss = update(); handle.remove()
            expected = 'torch.float32' if condition == 'mamba2_fp32' else 'torch.bfloat16'
            assert seen and seen[0] == expected, (condition, seen)
            assert all(p.dtype == torch.float32 for p in model.parameters())
            assert all(p.grad is None or p.grad.dtype == torch.float32 for p in model.parameters())
            assert torch.isfinite(loss)
            for _ in range(9): update()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            before = telemetry()
            samples = []
            for _ in range(30):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall = time.perf_counter()
                start.record(); loss = update(); end.record()
                torch.cuda.synchronize()
                samples.append({'gpu_ms': start.elapsed_time(end), 'wall_ms': (time.perf_counter() - wall) * 1000})
            assert torch.isfinite(loss)
            record = {'round': round_index, 'condition': condition, 'observed_in_proj_dtype': seen[0],
                      'median_gpu_ms': statistics.median(s['gpu_ms'] for s in samples),
                      'median_wall_ms': statistics.median(s['wall_ms'] for s in samples),
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'telemetry_before': before, 'telemetry_after': telemetry(), 'samples': samples}
            if round_index == 0:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                    update(); torch.cuda.synchronize()
                events = list(prof.key_averages())
                def gpu_time(event):
                    return getattr(event, 'self_device_time_total', getattr(event, 'self_cuda_time_total', 0))
                record['top_gpu_ops'] = [{'name': e.key, 'self_device_us': gpu_time(e), 'calls': e.count}
                                         for e in sorted(events, key=gpu_time, reverse=True)[:15]]
                del prof, events
            report['rounds'].append(record)
            Path('/results/precision_benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
            print('BENCHMARK', condition, round_index, record['median_gpu_ms'], flush=True)
            del model, opt, blocks, x, target, loss
            if condition == 'mamba2_bf16_blocks': del original, wrapped
            gc.collect(); torch.cuda.empty_cache()
    report['summary'] = {condition: {'median_gpu_ms': statistics.median(r['median_gpu_ms'] for r in report['rounds'] if r['condition'] == condition),
                                    'round_gpu_ms': [r['median_gpu_ms'] for r in report['rounds'] if r['condition'] == condition]}
                         for condition in CONDITIONS}
    report['complete'] = True
    Path('/results/precision_benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
    Path('/results/precision_complete.json').write_text(json.dumps({'complete': True, 'summary': report['summary']}, indent=2) + '\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__': main()
