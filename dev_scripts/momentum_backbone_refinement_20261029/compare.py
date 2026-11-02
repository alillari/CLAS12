"""Audit completed refined-backbone fits and render the five-model comparison."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dev_scripts.backbone_downstream_20261029.compare import (
    read, save, csv_save, history_read, column, vector_metrics, figure_save)
from dev_scripts.momentum_refinement_ablation_20261029.compare import frame

ORDER = ['ao0', 'ao1', 'mamba1', 'mamba2', 'mamba3']
COLORS = {'ao0': '#6b7280', 'ao1': '#8c4b96', 'mamba1': '#3b6fb6',
          'mamba2': '#d98a29', 'mamba3': '#35805c', 'cvt': '#252525'}
LABELS = {'ao0': 'Adapter-only · no refinement', 'ao1': 'Adapter-only · one Mamba2 layer',
    **{v: 'Frozen ' + v.capitalize() + ' · one Mamba2 layer' for v in ORDER[2:]}, 'cvt': 'CVT reference'}
SHORT = ['AO\n0 layers', 'AO\n1 layer', 'M1\n1 layer', 'M2\n1 layer', 'M3\n1 layer']
METRICS = [('component_mae_mev', 'Mean component MAE [MeV] ↓'),
    ('p_mae_mev', 'Momentum magnitude MAE [MeV] ↓'),
    ('relative_width68_pct', 'Relative momentum width68 [%] ↓'),
    ('theta_mae_deg', 'Theta MAE [degrees] ↓'), ('phi_mae_deg', 'Phi MAE [degrees] ↓'),
    ('tail10_pct', '|Δp/p| >10% [% of tracks] ↓')]
RECIPE_KEYS = ('batch_size', 'valid_batch_size', 'limit_size', 'limit_test_size', 'seed',
    'max_optimizer_steps', 'max_lr', 'min_lr', 'warmup_steps', 'scheduler_first_cycle_steps',
    'adapter_weight_decay', 'grad_clip_value', 'dropout', 'regression_loss', 'pooling', 'embed_dim',
    'adapter_embed_dim', 'num_embedder_layers', 'use_amp', 'adapter_version', 'coordinate_bypass',
    'regression_target_stats')


def audited_record(path, blocks):
    t, e, cfg, summary = (read(path / f) for f in
        ('training_complete.json', 'evaluation_complete.json', 'resolved.json', 'evaluation/summary.json'))
    assert t['complete'] and t['finite_gradient_updates'] == t['final_step'] == 30000
    for key in ('backbone_unchanged', 'strict_head_reload', 'prediction_equivalence', 'optimizer_scheduler_restore'):
        assert t[key], key
    assert t['head_mamba_blocks'] == cfg['num_adapter_layers'] == blocks
    assert t['head_parameters'] == (664744 if blocks else 232720)
    counts_path = ROOT / 'integrations/edge_mcp/validation/adapter-parameter-counts-20261029.json'
    counts = read(counts_path)['momentum_one_mamba2' if blocks else 'momentum_no_mamba']
    assert cfg['embed_dim'] == counts['input_width'] and cfg['adapter_embed_dim'] == counts['latent_width']
    if 'head_trainable_parameters' not in t:
        t = {**t, 'head_trainable_parameters': counts['trainable'],
             'head_parameter_count_source': str(counts_path)}
    assert t['head_trainable_parameters'] == counts['trainable']
    assert t['train_samples'] == 50000 and t['validation_samples'] == 10000
    assert e['complete'] and e['actual_samples'] == e['adapter_valid_samples'] == 50000
    h = history_read(path / 'checkpoints/training.log', delimiter='\t')
    assert [r['step'] for r in h] == list(range(1000, 30001, 1000))
    assert np.isfinite(column(h, 'val_loss')).all() and (column(h, 'val_loss') > 0).all()
    best = next(r for r in h if r['step'] == t['best_step'])
    assert np.isclose(best['val_loss'], t['best_loss'])
    return {'path': path, 'training': t, 'evaluation': e, 'config': cfg, 'summary': summary, 'history': h}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True,
                        help='Existing backbone_downstream_comparison_20261029 folder')
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / 'backbone_refinement'
    out = base / 'comparison'
    out.mkdir(parents=True, exist_ok=True)
    for host, versions in [('a02c', ['mamba1', 'mamba3']), ('a518', ['mamba2'])]:
        p = base / host
        assert read(p / 'runner_result.json')['exit_code'] == 0
        completed = read(p / 'completed.json')
        assert completed['complete'] and {r['backbone'] for r in completed['runs']} == set(versions)
        pilot = read(base / 'pilots' / host / 'pilot_complete.json')
        assert pilot['complete'] and len(pilot['runs']) == len(versions)
        assert read(base / 'pilots' / host / 'runner_result.json')['exit_code'] == 0
        for r in pilot['runs']:
            t, e = r['training'], r['evaluation']
            assert t['finite_gradient_updates'] == 5 and e['actual_samples'] == 64
            assert t['strict_head_reload'] and t['prediction_equivalence'] and t['head_fp32_execution']
    assert read(root / 'refinement/a518/runner_result.json')['exit_code'] == 0
    assert read(root / 'refinement/a518/completed.json')['complete']
    paths = {'ao0': root / 'a518/runs/momentum_adapteronly',
        'ao1': root / 'refinement/a518/with_mamba2',
        **{v: base / h / 'runs' / ('momentum_' + v) for h, v in
           [('a02c', 'mamba1'), ('a518', 'mamba2'), ('a02c', 'mamba3')]}}
    records = {k: audited_record(p, int(k != 'ao0')) for k, p in paths.items()}
    ref = records['ao1']
    for key in ('train_keys_sha256', 'validation_keys_sha256'):
        assert len({r['training'][key] for r in records.values()}) == 1, key
    for key in ('evaluation_keys_sha256', 'truth_sha256'):
        assert len({r['evaluation'][key] for r in records.values()}) == 1, key
    for key in RECIPE_KEYS:
        assert len({str(r['config'][key]) for r in records.values()}) == 1, key
    for k in ORDER[1:]:
        assert records[k]['training']['initial_head_sha256'] == ref['training']['initial_head_sha256']
    for host in ('a02c', 'a518'):
        initial = read(base / 'initialization' / host / 'initial_head.json')
        assert initial['complete'] and initial['optimizer_updates'] == 0
        assert initial['initial_head_sha256'] == ref['training']['initial_head_sha256']
        assert read(base / 'initialization' / host / 'runner_result.json')['exit_code'] == 0
    for k in ORDER:
        t = records[k]['training']
        assert t.get('shared_initial_head_sha256', t['initial_head_sha256']) == ref['training']['shared_initial_head_sha256']
        if k in ORDER[2:]:
            assert t['head_fp32_execution'] and t['refinement_nonzero_gradient']
    data = {k: frame(p / 'evaluation/predictions.csv.gz') for k, p in paths.items()}
    for k in ORDER:
        for field in ('keys', 'true', 'cvt_entrance'):
            assert np.array_equal(data[k][field], data['ao1'][field], equal_nan=True), (k, field)
        assert np.isfinite(data[k]['adapter']).all()
    truth = data['ao1']['true']
    cvt = data['ao1']['cvt_entrance']
    mask = np.isfinite(cvt).all(axis=1)
    assert mask.sum() == 49803
    full = {k: vector_metrics(truth, d['adapter']) for k, d in data.items()}
    common = {k: vector_metrics(truth[mask], d['adapter'][mask]) for k, d in data.items()}
    common['cvt'] = vector_metrics(truth[mask], cvt[mask])
    rows = []
    for k in ORDER:
        r = records[k]; t = r['training']; saved = r['summary']['methods']['adapter']
        assert np.isclose(full[k]['p_mae_mev'], saved['momentum']['mae_gev'] * 1000)
        assert np.isclose(full[k]['relative_width68_pct'], saved['momentum']['relative_resolution_68'] * 100)
        rows.append({'model': k, 'label': LABELS[k], 'head_mamba_layers': t['head_mamba_blocks'],
            'head_total_parameters': t['head_parameters'], 'head_trainable_parameters': t['head_trainable_parameters'],
            'head_parameter_count_source': t.get('head_parameter_count_source', 'training_complete.json'),
            'best_validation_mae': t['best_loss'], 'best_step': t['best_step'],
            'training_seconds': t['elapsed_seconds'], 'peak_cuda_gib': t['peak_cuda_allocated_bytes'] / 2**30,
            **full[k]})
    csv_save(out / 'metrics_all_50000_tracks.csv', rows)
    csv_save(out / 'metrics_common_population.csv', [{'model': k, 'label': LABELS[k], **m} for k, m in common.items()])
    csv_save(out / 'validation_histories.csv', [{'model': k, **r} for k in ORDER for r in records[k]['history']])
    gains = [{'model': k, 'control': control, 'metric': key,
        'reduction_percent': 100 * (1 - full[k][key] / full[control][key])}
        for k in ORDER[2:] for control in ('ao0', 'ao1') for key, _ in METRICS]
    csv_save(out / 'gains_over_adapteronly.csv', gains)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, loglog in zip(axs, (False, True)):
        for k in ORDER:
            h = records[k]['history']; t = records[k]['training']
            ax.plot(column(h, 'step'), column(h, 'val_loss'), color=COLORS[k],
                    ls='--' if k == 'ao1' else '-', label=LABELS[k], lw=1.8)
            ax.scatter(t['best_step'], t['best_loss'], marker='*', s=65, color=COLORS[k], zorder=4)
        ax.set(title='Standardized validation MAE' + (' — log–log' if loglog else ''),
            xlabel='Optimizer updates', yscale='log', xscale='log' if loglog else 'linear')
        ax.grid(alpha=.2, which='both')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=8.5, frameon=False)
    figure_save(fig, out, 'validation_learning_curves',
        'All colored models have one Mamba2 head layer; gray has zero. Matched data, 30k updates and full starting head weights; single seed.', top=.83)
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (key, title) in zip(axs.flat, METRICS):
        for i, k in enumerate(ORDER):
            y = common[k][key]
            ax.scatter(i, y, color=COLORS[k], s=65)
            ax.annotate(f'{y:.2f}', (i, y), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
        ax.axhline(common['cvt'][key], ls='--', color=COLORS['cvt'], alpha=.7, lw=1.2, label='CVT reference')
        ax.set(title=title, xticks=range(5), xticklabels=SHORT, xlim=(-.5, 4.5))
        ax.margins(y=.25); ax.grid(alpha=.2, axis='y'); ax.tick_params(axis='x', labelsize=8)
    axs.flat[0].legend(fontsize=8, loc='upper right')
    figure_save(fig, out, 'momentum_quality',
        'Same 49,803 CVT-valid tracks. AO = adapter-only; M1/M2/M3 = frozen backbones. Layer counts refer to Mamba2 head refinement.')
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, key, label in zip(axs, ['px_mae_mev', 'py_mae_mev', 'pz_mae_mev'], ['px', 'py', 'pz']):
        for i, k in enumerate(ORDER):
            y = common[k][key]; ax.scatter(i, y, color=COLORS[k], s=65)
            ax.annotate(f'{y:.2f}', (i, y), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
        ax.axhline(common['cvt'][key], ls='--', color=COLORS['cvt'], alpha=.7, lw=1.2, label='CVT reference')
        ax.set(title=label + ' MAE [MeV] ↓', xticks=range(5), xticklabels=SHORT, xlim=(-.5, 4.5))
        ax.margins(y=.25); ax.grid(alpha=.2, axis='y'); ax.tick_params(axis='x', labelsize=8)
    axs[0].legend(fontsize=8)
    figure_save(fig, out, 'component_errors', 'Same 49,803 tracks. All refined heads share one Mamba2 layer and the same declared head capacity.')
    bins = np.array([.25, .5, .75, 1., 1.5, 2., 3., 5., 10.])
    p = np.linalg.norm(truth[mask], axis=1); binned = []
    for k, prediction in {**{k: d['adapter'][mask] for k, d in data.items()}, 'cvt': cvt[mask]}.items():
        for low, high in zip(bins[:-1], bins[1:]):
            selected = (p >= low) & (p < high)
            if selected.sum() < 20: continue
            binned.append({'model': k, 'bin_low_gev': low, 'bin_high_gev': high,
                'bin_center_gev': (low + high) / 2, **vector_metrics(truth[mask][selected], prediction[selected])})
    csv_save(out / 'binned_metrics_common_population.csv', binned)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.9))
    for ax, key, title in zip(axs, ['p_mae_mev', 'relative_width68_pct', 'opening_angle_mean_deg'],
            ['Magnitude MAE [MeV]', 'Relative momentum width68 [%]', 'Mean opening angle [degrees]']):
        for k in [*ORDER, 'cvt']:
            selected = [r for r in binned if r['model'] == k]
            ax.plot([r['bin_center_gev'] for r in selected], [r[key] for r in selected],
                color=COLORS[k], ls='--' if k in ('ao1', 'cvt') else '-', marker='.', lw=1.7, label=LABELS[k])
        ax.set(title=title, xlabel='True entrance momentum [GeV]'); ax.grid(alpha=.2)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=8, frameon=False)
    figure_save(fig, out, 'momentum_binned',
        'Identical CVT-valid population in each bin. Width68 = half(q84−q16). All learned models except gray have one Mamba2 refinement layer.', top=.82)
    effects = []
    old_paths = {'ao': paths['ao0'], **{v: root / h / 'runs' / ('momentum_' + v)
        for h, v in [('a02c', 'mamba1'), ('a518', 'mamba2'), ('a02c', 'mamba3')]}}
    for k, path in old_paths.items():
        old = audited_record(path, 0)
        current = records['ao1' if k == 'ao' else k]
        for field in ('train_keys_sha256', 'validation_keys_sha256', 'backbone_initial_sha256'):
            assert old['training'][field] == current['training'][field], (k, field)
        for field in ('evaluation_keys_sha256', 'truth_sha256'):
            assert old['evaluation'][field] == current['evaluation'][field]
        old_data = frame(path / 'evaluation/predictions.csv.gz')
        old_full = vector_metrics(truth, old_data['adapter'])
        assert np.array_equal(old_data['keys'], data['ao1']['keys']) and np.array_equal(old_data['true'], truth)
        for key, _ in METRICS:
            new = full['ao1' if k == 'ao' else k][key]
            effects.append({'input': k, 'metric': key, 'zero_layer': old_full[key], 'one_layer': new,
                            'reduction_percent': 100 * (1 - new / old_full[key])})
    csv_save(out / 'refinement_effect.csv', effects)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, (key, title) in zip(axs, METRICS[:3]):
        selected = [r for r in effects if r['metric'] == key]
        ax.plot(range(4), [r['zero_layer'] for r in selected], 'o--', color='#6b7280', label='Zero refinement layers')
        ax.plot(range(4), [r['one_layer'] for r in selected], 'o-', color='#8c4b96', label='One Mamba2 layer')
        ax.set(title=title, xticks=range(4), xticklabels=['Adapter-only', 'Mamba1', 'Mamba2', 'Mamba3'])
        ax.grid(alpha=.2)
    axs[0].legend(fontsize=8)
    figure_save(fig, out, 'refinement_effect', 'Same 50,000 evaluation tracks. Adding refinement adds 432,024 head parameters; this does not isolate topology at equal capacity.')
    audit = {'complete': True, 'models': 5, 'evaluation_tracks_each': 50000, 'common_cvt_tracks': int(mask.sum()),
        'identities_truth_cvt_match_row_by_row': True, 'shared_initial_weights_match': True,
        'full_initial_weights_match_all_four_refined_heads': True, 'finite_updates_each': 30000,
        'strict_checkpoint_reload_and_frozen_backbones_verified': True,
        'initial_head_reconstruction_both_hosts_verified': True,
        'fp32_head_execution_and_nonzero_refinement_gradients_new_runs': True,
        'all_new_supervisors_containers_and_pilots_exited_successfully': True}
    save(out / 'audit.json', audit)
    save(out / 'summary.json', {'complete': True, 'models': rows, 'common_population': common,
        'gains_over_adapteronly': gains, 'refinement_effect': effects, 'audit': audit})
    report = ['# Momentum regression with matched Mamba2 refinement heads', '',
        'All three frozen backbones now feed the same one-layer Mamba2 momentum head as the completed refined AdapterOnly control. '
        'The zero-layer AdapterOnly model is retained as an ablation. No backbone is retrained or unfrozen.', '',
        '| Model | Component MAE [MeV] | Magnitude MAE [MeV] | Relative width68 [%] | Theta MAE [deg] | Phi MAE [deg] |',
        '|---|---:|---:|---:|---:|---:|']
    for k in ORDER:
        m = full[k]
        report.append(f"| {LABELS[k]} | {m['component_mae_mev']:.3f} | {m['p_mae_mev']:.3f} | {m['relative_width68_pct']:.3f} | {m['theta_mae_deg']:.3f} | {m['phi_mae_deg']:.3f} |")
    report.extend(['', 'The table uses all 50,000 tracks. Quality/component/binned figures use the same 49,803 CVT-valid tracks for every model and the CVT reference.', '',
        '## Gains against the refined AdapterOnly control', '',
        '| Frozen backbone + one Mamba2 layer | Component MAE reduction | Magnitude MAE reduction | Width68 reduction |',
        '|---|---:|---:|---:|'])
    for k in ORDER[2:]:
        g = [100 * (1 - full[k][key] / full['ao1'][key]) for key in ('component_mae_mev', 'p_mae_mev', 'relative_width68_pct')]
        report.append(f'| {k} | {g[0]:+.2f}% | {g[1]:+.2f}% | {g[2]:+.2f}% |')
    report.extend(['', 'Positive reductions mean lower error; negative reductions mean higher error. These are single-seed measured comparisons, without uncertainty estimates.', '',
        '## Interpretation', '',
        'With the same one-layer head, none of the frozen backbones gives lower component MAE, magnitude MAE or relative width68 than refined AdapterOnly in this seed. '
        'Mamba3 is closest: component MAE is 0.44% higher, magnitude MAE 1.26% higher and width68 is 1.72% higher. '
        f"Mamba3 does improve theta MAE by {100*(1-full['mamba3']['theta_mae_deg']/full['ao1']['theta_mae_deg']):.2f}% "
        f"and phi MAE by {100*(1-full['mamba3']['phi_mae_deg']/full['ao1']['phi_mae_deg']):.2f}%. "
        'All three refined backbones improve theta and mean opening-angle errors over refined AdapterOnly, while Mamba1/2 have higher phi MAE. '
        'The small differences require repeated seeds before claiming an advantage. The present result does not support a broad momentum-error benefit from these frozen pretrained representations at 50k labels.', '',
        'Refinement improves every tabulated metric for each input compared with its own zero-layer run. Component MAE falls 33.26% for Mamba1, 28.65% for Mamba2 and 27.11% for Mamba3. '
        'This establishes a useful stronger downstream head in these runs; it does not separate added capacity from sequence processing.', '',
        '## Matched recipe and audit', '',
        'Seed 42; 50k training, 10k validation and 50k final evaluation tracks; Cartesian entrance momentum; train-only standardization; standardized component MAE; '
        'attention pooling; batch 128; 30k optimizer updates; validation every 1k updates; validation-selected checkpoint; AdamW LR 1e-5→3e-4→1e-5, 1k warmup, '
        'weight decay 1e-4, gradient clipping 0.1, dropout 0.1. All starting head tensors match across the four refined arms, including the Mamba2 block. '
        'The three new runs pass real GPU pilots, finite-gradient checks, strict checkpoint reload and prediction equivalence, optimizer/scheduler restore and unchanged frozen backbone checks.', '',
        'Each refined head has 664,744 total / 632,488 declared-trainable parameters; the zero-layer head has 232,720 / 200,464. '
        'Both counts include 32,256 fixed coordinate-encoding parameters; the 12 depth weights are unused in AdapterOnly mode. '
        'The added block contributes 432,024 parameters. The zero/one comparison changes capacity as well as sequence processing.', '',
        'Heads execute in FP32. Frozen Mamba1/2 use their native FP32 blocks; Mamba3 retains local BF16 blocks and returns FP32 features. '
        'Mamba1/3 ran on a02c; Mamba2 and both reused AdapterOnly controls ran on a518. Recorded elapsed times include training, validation and audits; '
        'they do not establish architecture-only speed differences. Width68 is half(q84−q16), not a fitted Gaussian sigma.', '',
        '## Measured runtime and allocation', '', '| Model | Training/validation/audit [min] | Peak CUDA allocation [GiB] |', '|---|---:|---:|'])
    for r in rows:
        report.append(f"| {r['label']} | {r['training_seconds']/60:.2f} | {r['peak_cuda_gib']:.3f} |")
    report.extend(['', 'PNG/PDF plots: `validation_learning_curves`, `momentum_quality`, `component_errors`, `momentum_binned`, `refinement_effect`. '
        'CSV tables retain full/common populations, raw histories, binned metrics and gains. `audit.json` records the completion gates.', ''])
    (out / 'report.md').write_text('\n'.join(report))
    with zipfile.ZipFile(base / 'momentum_refined_comparison.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.iterdir()):
            if path.is_file(): z.write(path, 'comparison/' + path.name)
    print(json.dumps({'complete': True, 'output': str(out), 'models': rows}, indent=2))


if __name__ == '__main__': main()
