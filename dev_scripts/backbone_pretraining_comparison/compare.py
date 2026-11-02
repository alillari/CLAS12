"""Recorded validation-learning curves versus steps and estimated training compute."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
from mmap_ninja import RaggedMmap

from dev_scripts.backbone_pretraining_comparison.tokens import replay


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def curve_from_log(path):
    return [{'step': int(step), 'validation_mse': float(loss)}
            for step, loss in re.findall(r'Step\s*=\s*(\d+), Val loss\s*=\s*([0-9.eE+-]+)', Path(path).read_text())]


def render(models, output, partial=False, loglog=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'mamba1': '#3b6fb6', 'mamba2': '#d98a29', 'mamba3': '#35805c'}
    for axis, filename, xlabel in [
        ('step', 'validation_loss_vs_steps', 'Optimizer updates'),
        ('estimated_training_flops', 'validation_loss_vs_compute', 'Estimated training compute (PFLOP)'),
        ('dense_training_flops', 'validation_loss_vs_dense_compute', 'Dense-projection training arithmetic (PFLOP)')]:
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        for model in sorted(models, key=lambda m: m['version']):
            points = model['curve']
            x = np.array([p[axis] for p in points], dtype=float)
            if axis != 'step': x /= 1e15
            y = np.array([p['validation_mse'] for p in points])
            label = model['version'].replace('mamba', 'Mamba ')
            if not model['complete']: label += ' (running)'
            ax.plot(x, y, label=label, color=colors[model['version']], linewidth=1.6)
            best = int(np.argmin(y))
            ax.scatter(x[best], y[best], color=colors[model['version']], marker='*', s=70, zorder=3)
        ax.set(xlabel=xlabel, ylabel='Checkpoint validation MSE', yscale='log',
               xscale='log' if loglog else 'linear',
               title='Matched k=1 pretraining' + (' — interim' if partial else '') + (' — log–log' if loglog else ''))
        ax.grid(alpha=.2, which='both'); ax.legend(frameon=False)
        fig.text(.5, -.012, 'Raw validation loss at each checkpoint; no cumulative averaging. Single seed.', ha='center', fontsize=9)
        if loglog: filename += '_loglog'
        fig.savefig(output / (filename + '.png'), dpi=180, bbox_inches='tight')
        fig.savefig(output / (filename + '.pdf'), bbox_inches='tight')
        plt.close(fig)


def compare(m1_root, data_root, output, partial=False):
    output.mkdir(parents=True, exist_ok=True)
    models = json.loads(Path(__file__).with_name('completed_models.json').read_text())
    profile = json.loads(Path(__file__).with_name('compute_profile.json').read_text())
    m1_cohorts = json.loads((m1_root / 'cohorts.json').read_text())
    complete = (m1_root / 'completed.json').exists()
    m1_result = json.loads((m1_root / ('completed.json' if complete else 'started.json')).read_text())
    models.append({'version': 'mamba1', 'host': 'a02c', 'job_id': 'mamba1-chinchilla-w512-d12-k1-20261028',
                   'completed': m1_result, 'cohorts': m1_cohorts,
                   'curve': curve_from_log(m1_root / 'training/mamba1_chinchilla_w512_d12_k1/seed42/train.log')})
    for model in models:
        for split in ['training', 'validation']:
            assert model['cohorts'][split]['indices_sha256'] == m1_cohorts[split]['indices_sha256'], 'Cohort mismatch'
        assert model['completed']['total_steps'] == 200000
        assert model['completed']['effective_batch_size'] == 64
        assert model['completed']['architecture']['blocks']
        assert all(v['d_state'] == 16 for v in model['completed']['architecture']['blocks'].values())
        assert model['curve'], model['version']
        model['complete'] = model['version'] != 'mamba1' or complete
    if not partial:
        assert all(m['complete'] for m in models)
        assert all(m['completed']['final_step'] == 200000 and
                   m['completed']['finite_gradient_updates'] == 200000 and
                   'passed' in m['completed']['checkpoint_reload'] for m in models)
    features = RaggedMmap(str(data_root / 'features_pretrain'))
    lengths = (np.asarray(features.ends) - np.asarray(features.starts)) // 3
    selected = np.flatnonzero((lengths >= 1) & (lengths <= 50))[:10000000]
    # Verify selection bytes using NumPy's actual .npy format, as in training.
    import io
    stream = io.BytesIO(); np.save(stream, selected)
    assert hashlib.sha256(stream.getvalue()).hexdigest() == m1_cohorts['training']['indices_sha256']
    padded, valid = replay(lengths[selected], 64, 200000)
    cumulative_padded, cumulative_valid = np.cumsum(padded), np.cumsum(valid)
    common_rows = []
    for model in models:
        version = model['version']; cost = profile['models'][version]
        assert cost['parameters'] == model['completed']['parameter_count']
        for point in model['curve']:
            n = point['step'] - 1
            point['padded_tokens'] = int(cumulative_padded[n])
            point['valid_tokens'] = int(cumulative_valid[n])
            point['estimated_training_flops'] = point['padded_tokens'] * cost['estimated_training_flops_per_padded_token']
            point['dense_training_flops'] = point['padded_tokens'] * cost['dense_training_flops_per_padded_token']
            common_rows.append({'version': version, **point})
        best = min(model['curve'], key=lambda p: p['validation_mse'])
        model['summary'] = {'parameters': cost['parameters'], 'final_step': model['curve'][-1]['step'],
                            'best_validation_mse': best['validation_mse'], 'best_step': best['step'],
                            'final_validation_mse': model['curve'][-1]['validation_mse'],
                            'estimated_training_pflop': model['curve'][-1]['estimated_training_flops'] / 1e15,
                            'dense_training_pflop': model['curve'][-1]['dense_training_flops'] / 1e15,
                            'recorded_elapsed_seconds': model['completed'].get('elapsed_seconds'),
                            'same_host_training_core_benchmark': cost['benchmark'],
                            'first_threshold_crossings': {str(threshold): next((p for p in model['curve'] if p['validation_mse'] <= threshold), None)
                                                         for threshold in [.001, .0008, .0006, .0005]}}
    budget = min(m['curve'][-1]['estimated_training_flops'] for m in models)
    report = {'complete': all(m['complete'] for m in models), 'models': models,
              'compute_profile': profile, 'common_compute_budget_flops': budget,
              'at_common_compute_budget': {m['version']: max((p for p in m['curve'] if p['estimated_training_flops'] <= budget), key=lambda p: p['step']) for m in models},
              'training_event_presentations': 200000 * 64,
              'total_padded_training_tokens': int(cumulative_padded[-1]), 'total_valid_training_tokens': int(cumulative_valid[-1]),
              'limitations': ['Single seed; matched ported recipe, not family-specific tuning.',
                              'Validation selects checkpoints; no downstream or independent holdout evaluation is included.',
                              'FLOPs are documented algorithmic estimates, not measured executed GPU operations.',
                              'Precision, native initialization, and architecture-specific kernels differ.',
                              'Original campaign wall times were measured on two hosts; the core benchmark uses one host.']}
    with (output / 'learning_curves.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(common_rows[0])); writer.writeheader(); writer.writerows(common_rows)
    save(output / 'summary.json', report)
    render(models, output, partial=partial)
    lines = ['# Matched pretraining comparison', '', '| Backbone | Best validation MSE | Best step | Final validation MSE | Estimated training PFLOP |',
             '| --- | ---: | ---: | ---: | ---: |']
    for m in sorted(models, key=lambda m:m['version']):
        s=m['summary']; lines.append(f"| {m['version']} | {s['best_validation_mse']:.8f} | {s['best_step']} | {s['final_validation_mse']:.8f} | {s['estimated_training_pflop']:.3f} |")
    lines.extend(['', 'These are raw checkpoint validation losses, not sums or cumulative averages.', '', *['- '+x for x in report['limitations']]])
    (output / 'report.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mamba1-output', default='/mamba1')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', default='/results/comparison')
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    root, output = Path(args.mamba1_output), Path(args.output)
    if args.wait:
        while not (root / 'runner_result.json').exists():
            progress = json.loads((root / 'progress.json').read_text()) if (root / 'progress.json').exists() else None
            save(output.parent / 'phase.json', {'phase': 'waiting_for_matched_mamba1', 'progress': progress})
            time.sleep(30)
        result = json.loads((root / 'runner_result.json').read_text())
        if result['exit_code'] != 0: raise RuntimeError('Mamba1 training failed: '+json.dumps(result))
    save(output.parent / 'phase.json', {'phase': 'building_pretraining_comparison'})
    report = compare(root, Path(args.data_root), output, partial=args.partial)
    if report['complete']:
        save(output.parent / 'comparison_complete.json', {'complete': True, 'models': [m['version'] for m in report['models']],
             'summary_path': str(output / 'summary.json'), 'scope': 'pretraining learning curves, documented compute estimates, runtime/memory; no downstream'})
    print(json.dumps({m['version']:m['summary'] for m in report['models']}, default=str), flush=True)


if __name__ == '__main__': main()
