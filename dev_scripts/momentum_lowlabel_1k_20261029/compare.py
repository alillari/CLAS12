"""Audit eight 1k-label momentum fits; compare heads and the retained 50k fits."""
import argparse
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

VERSIONS = ['adapteronly', 'mamba1', 'mamba2', 'mamba3']
LABELS = {'adapteronly': 'Adapter-only', 'mamba1': 'Frozen Mamba1', 'mamba2': 'Frozen Mamba2',
          'mamba3': 'Frozen Mamba3', 'cvt': 'CVT reference'}
COLORS = {'adapteronly': '#6b7280', 'mamba1': '#3b6fb6', 'mamba2': '#d98a29', 'mamba3': '#35805c', 'cvt': '#252525'}
METRICS = [('component_mae_mev', 'Mean component MAE [MeV] ↓'), ('p_mae_mev', 'Magnitude MAE [MeV] ↓'),
    ('relative_width68_pct', 'Relative momentum width68 [%] ↓'), ('theta_mae_deg', 'Theta MAE [degrees] ↓'),
    ('phi_mae_deg', 'Phi MAE [degrees] ↓'), ('tail10_pct', '|Δp/p| >10% [% of tracks] ↓')]


def key(v, layers): return f'{v}_m2ref{layers}'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--baseline', type=Path, default=ROOT / 'research_reports/backbone_downstream_comparison_20261029')
    args = p.parse_args(); root = args.root.resolve(); base = args.baseline.resolve()
    remote = root / 'a02c'; out = root / 'comparison'; out.mkdir(parents=True, exist_ok=True)
    assert read(remote / 'runner_result.json')['exit_code'] == 0
    assert read(remote / 'completed.json')['complete'] and len(read(remote / 'completed.json')['runs']) == 8
    pilots = read(remote / 'pilot_complete.json')
    assert pilots['complete'] and len(pilots['runs']) == 8
    for arm in pilots['runs']:
        t, e = arm['training'], arm['evaluation']
        assert t['finite_gradient_updates'] == 5 and e['actual_samples'] == 64
        assert t['strict_head_reload'] and t['prediction_equivalence'] and t['head_fp32_execution']
    ready = read(remote / 'data_ready.json')
    reference = read(Path(__file__).with_name('reference.json'))
    assert ready['complete'] and ready['training_tracks'] == 1000
    assert ready['previous_selected_data_content_revalidated'] and not ready['scientific_source_changes_since_50k']
    records = {}; data = {}; full = {}; common = {}; rows = []
    for v in VERSIONS:
        for layers in (0, 1):
            k = key(v, layers); path = remote / 'runs' / ('momentum_' + k)
            t, e, cfg, summary = (read(path / f) for f in ('training_complete.json', 'evaluation_complete.json', 'resolved.json', 'evaluation/summary.json'))
            assert t['complete'] and e['complete']
            assert t['final_step'] == t['finite_gradient_updates'] == 30000
            assert t['train_samples'] == 1000 and t['validation_samples'] == 10000
            assert t['head_mamba_blocks'] == cfg['num_adapter_layers'] == layers
            assert t['normalization_training_tracks'] == 1000
            for flag in ('backbone_unchanged', 'strict_head_reload', 'prediction_equivalence', 'optimizer_scheduler_restore', 'head_fp32_execution'):
                assert t[flag], (k, flag)
            if layers: assert t['refinement_nonzero_gradient']
            assert t['train_keys_sha256'] == ready['train_keys_sha256']
            assert t['validation_keys_sha256'] == ready['validation_keys_sha256']
            for field in ('initial_head_sha256', 'shared_initial_head_sha256'):
                assert t[field] == ready['initial_heads'][str(layers)][field]
            assert e['actual_samples'] == e['adapter_valid_samples'] == 50000
            assert e['evaluation_keys_sha256'] == ready['evaluation_keys_sha256']
            assert t['head_parameters'] == (664744 if layers else 232720)
            assert t['head_trainable_parameters'] == (632488 if layers else 200464)
            if v != 'adapteronly':
                assert t['backbone_initial_sha256'] == reference['backbone_training'][v]['backbone_initial_sha256']
            assert e['truth_sha256'] == reference['evaluation']['truth_sha256']
            assert cfg['use_amp'] is False and cfg['num_embedder_layers'] == 0 and cfg['coordinate_bypass'] is False
            h = history_read(path / 'checkpoints/training.log', delimiter='\t')
            assert [r['step'] for r in h] == list(range(1000, 30001, 1000))
            assert np.isfinite(column(h, 'val_loss')).all() and (column(h, 'val_loss') > 0).all()
            assert np.isclose(next(r['val_loss'] for r in h if r['step'] == t['best_step']), t['best_loss'])
            records[k] = {'training': t, 'evaluation': e, 'config': cfg, 'summary': summary, 'history': h}
            data[k] = frame(path / 'evaluation/predictions.csv.gz')
    for field in ('batch_size', 'max_optimizer_steps', 'seed', 'max_lr', 'min_lr', 'warmup_steps', 'scheduler_first_cycle_steps',
            'adapter_weight_decay', 'grad_clip_value', 'dropout', 'regression_loss', 'pooling', 'embed_dim', 'adapter_embed_dim', 'regression_target_stats'):
        assert len({str(r['config'][field]) for r in records.values()}) == 1, field
    ref = data[key('adapteronly', 0)]
    for k, d in data.items():
        for field in ('keys', 'true', 'cvt_entrance'):
            assert np.array_equal(d[field], ref[field], equal_nan=True), (k, field)
        assert np.isfinite(d['adapter']).all()
    old_ref = frame(base / 'a518/runs/momentum_adapteronly/evaluation/predictions.csv.gz')
    for field in ('keys', 'true', 'cvt_entrance'):
        assert np.array_equal(ref[field], old_ref[field], equal_nan=True), field
    truth, cvt = ref['true'], ref['cvt_entrance']; mask = np.isfinite(cvt).all(axis=1)
    assert mask.sum() == 49803
    cvt_metrics = vector_metrics(truth[mask], cvt[mask])
    for v in VERSIONS:
        for layers in (0, 1):
            k = key(v, layers); r = records[k]; t = r['training']
            full[k] = vector_metrics(truth, data[k]['adapter'])
            common[k] = vector_metrics(truth[mask], data[k]['adapter'][mask])
            s = r['summary']['methods']['adapter']
            assert np.isclose(full[k]['p_mae_mev'], 1000 * s['momentum']['mae_gev'])
            assert np.isclose(full[k]['relative_width68_pct'], 100 * s['momentum']['relative_resolution_68'])
            rows.append({'model': v, 'refinement_layers': layers, 'training_tracks': 1000,
                'head_total_parameters': t['head_parameters'], 'head_trainable_parameters': t['head_trainable_parameters'],
                'best_validation_mae': t['best_loss'], 'best_step': t['best_step'],
                'training_seconds': t['elapsed_seconds'], 'peak_cuda_gib': t['peak_cuda_allocated_bytes'] / 2**30, **full[k]})
    csv_save(out / 'metrics_all_50000_tracks.csv', rows)
    csv_save(out / 'metrics_common_population.csv', [{'model': k, **m} for k, m in common.items()] + [{'model': 'cvt', **cvt_metrics}])
    csv_save(out / 'validation_histories.csv', [{'model': k, **r} for k in records for r in records[k]['history']])
    gains = [{'backbone': v, 'refinement_layers': layers, 'metric': field,
        'reduction_percent_vs_matched_adapteronly': 100 * (1 - full[key(v, layers)][field] / full[key('adapteronly', layers)][field])}
        for layers in (0, 1) for v in VERSIONS[1:] for field, _ in METRICS]
    csv_save(out / 'gains_over_adapteronly.csv', gains)
    effects = [{'input': v, 'metric': field, 'zero_layer': full[key(v, 0)][field], 'one_layer': full[key(v, 1)][field],
        'reduction_percent': 100 * (1 - full[key(v, 1)][field] / full[key(v, 0)][field])}
        for v in VERSIONS for field, _ in METRICS]
    csv_save(out / 'refinement_effect.csv', effects)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.8))
    for layers in (0, 1):
        for col, loglog in enumerate((False, True)):
            ax = axs[layers, col]
            for v in VERSIONS:
                k = key(v, layers); h = records[k]['history']; t = records[k]['training']
                ax.plot(column(h, 'step'), column(h, 'val_loss'), label=LABELS[v], color=COLORS[v], lw=1.8)
                ax.scatter(t['best_step'], t['best_loss'], marker='*', s=65, color=COLORS[v], zorder=4)
            ax.set(title=f'{layers} Mamba2 head layers: validation MAE' + (' — log–log' if loglog else ''),
                xlabel='Optimizer updates', yscale='log', xscale='log' if loglog else 'linear')
            ax.grid(alpha=.2, which='both')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=9, frameon=False)
    figure_save(fig, out, 'validation_learning_curves', 'Raw standardized validation MAE; 1k training / 10k validation tracks, seed42, 30k updates. Stars mark selected checkpoints.', top=.94)

    def paired_quality(name, specs, size):
        fig, axs = plt.subplots(2 if len(specs) == 6 else 1, 3, figsize=size)
        for ax, (field, title) in zip(np.asarray(axs).flat, specs):
            for layers, color, marker in [(0, '#6b7280', 'o'), (1, '#8c4b96', 's')]:
                values = [common[key(v, layers)][field] for v in VERSIONS]
                ax.plot(range(4), values, marker=marker, color=color, ls='--' if layers == 0 else '-',
                    label=f'{layers} Mamba2 refinement layers')
            ax.axhline(cvt_metrics[field], color=COLORS['cvt'], ls=':', label='CVT reference')
            ax.set(title=title, xticks=range(4), xticklabels=['AO', 'Mamba1', 'Mamba2', 'Mamba3']); ax.grid(alpha=.2)
        handles, labels = np.asarray(axs).flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=9, frameon=False)
        figure_save(fig, out, name, 'Same 49,803 CVT-valid tracks. AO = adapter-only. Refinement adds 432,024 head parameters. Single seed.', top=.91 if len(specs) == 6 else .84)
    paired_quality('momentum_quality', METRICS, (13, 7.8))
    paired_quality('component_errors', [(f'{v}_mae_mev', f'{v} MAE [MeV] ↓') for v in ('px', 'py', 'pz')], (13, 4.6))
    bins = np.array([.25, .5, .75, 1., 1.5, 2., 3., 5., 10.]); ptrue = np.linalg.norm(truth[mask], axis=1)
    binned = []
    for k, prediction in {**{k: d['adapter'][mask] for k, d in data.items()}, 'cvt': cvt[mask]}.items():
        for low, high in zip(bins[:-1], bins[1:]):
            selected = (ptrue >= low) & (ptrue < high)
            if selected.sum() < 20: continue
            binned.append({'model': k, 'bin_low_gev': low, 'bin_high_gev': high, 'bin_center_gev': (low + high) / 2,
                **vector_metrics(truth[mask][selected], prediction[selected])})
    csv_save(out / 'binned_metrics_common_population.csv', binned)
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    for layers in (0, 1):
        for ax, field, title in zip(axs[layers], ['p_mae_mev', 'relative_width68_pct', 'opening_angle_mean_deg'],
                ['Magnitude MAE [MeV]', 'Relative momentum width68 [%]', 'Mean opening angle [degrees]']):
            for v in [*VERSIONS, 'cvt']:
                k = 'cvt' if v == 'cvt' else key(v, layers)
                selected = [r for r in binned if r['model'] == k]
                ax.plot([r['bin_center_gev'] for r in selected], [r[field] for r in selected], color=COLORS[v],
                    marker='.', ls='--' if v == 'cvt' else '-', label=LABELS[v])
            ax.set(title=f'{layers} head layers: ' + title, xlabel='True entrance momentum [GeV]'); ax.grid(alpha=.2)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=9, frameon=False)
    figure_save(fig, out, 'momentum_binned', 'Identical CVT-valid populations in each bin; 1k training tracks. Width68 = half(q84−q16), not a Gaussian fit.', top=.94)
    old_zero = {r['model']: r for r in read(base / 'comparison/summary.json')['runs'] if r['task'] == 'momentum'}
    old_one = {r['model']: r for r in read(base / 'backbone_refinement/comparison/summary.json')['models']}
    old = {key(v, layers): (old_zero[v] if not layers else old_one['ao1' if v == 'adapteronly' else v])
        for v in VERSIONS for layers in (0, 1)}
    budget = [{'model': v, 'refinement_layers': layers, 'metric': field,
        'training_1k': full[key(v, layers)][field], 'training_50k': old[key(v, layers)][field],
        'error_ratio_1k_over_50k': full[key(v, layers)][field] / old[key(v, layers)][field]}
        for layers in (0, 1) for v in VERSIONS for field, _ in METRICS]
    csv_save(out / 'training_label_budget_comparison.csv', budget)
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.8))
    for layers in (0, 1):
        for ax, (field, title) in zip(axs[layers], METRICS[:3]):
            for n, color, marker in [(1000, '#b24a48', 'o'), (50000, '#6b7280', 's')]:
                values = [full[key(v, layers)][field] if n == 1000 else old[key(v, layers)][field] for v in VERSIONS]
                ax.plot(range(4), values, marker=marker, color=color, label=f'{n:,} training tracks')
            ax.set(title=f'{layers} head layers: ' + title, xticks=range(4), xticklabels=['AO', 'Mamba1', 'Mamba2', 'Mamba3']); ax.grid(alpha=.2)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=9, frameon=False)
    figure_save(fig, out, 'training_label_budget_comparison', 'Same 50,000 evaluation tracks and 30k-update budget. Target statistics are recalculated per training subset; compare physical errors.', top=.94)
    audit = {'complete': True, 'runs': 8, 'training_tracks': 1000, 'training_source_events': ready['training_source_events'],
        'validation_tracks': 10000, 'evaluation_tracks': 50000, 'common_cvt_tracks': int(mask.sum()),
        'exact_updates_each': 30000, 'cohorts_truth_cvt_and_initial_weights_verified': True,
        'label_budget': '1k training labels plus the retained 10k validation labels for checkpoint selection',
        'normalization': '1k training tracks only; 50k-label normalization buffers replaced before training',
        'precision': 'FP32 heads; native frozen Mamba1/2 FP32 and Mamba3 local BF16', 'single_seed': True}
    save(out / 'audit.json', audit)
    save(out / 'summary.json', {'complete': True, 'models': rows, 'metrics_all_tracks': full, 'metrics_common_population': common,
        'gains_over_adapteronly': gains, 'refinement_effect': effects, 'training_label_budget_comparison': budget, 'audit': audit})
    report = ['# Momentum at 1,000 labeled training tracks', '',
        'Eight fresh fits on a02c: AdapterOnly and frozen Mamba1/Mamba2/Mamba3, each with zero or one Mamba2 refinement layer. '
        'All eight GPU pilots and the final training/evaluation audits passed. The same seeded 1k subset and shared starting weights were used.', '',
        'At 1k training tracks, every frozen pretrained backbone substantially outperforms AdapterOnly with the same refinement count. '
        'With one refinement layer, magnitude MAE is 50.23–56.12% lower and relative width68 is 56.38–63.20% lower than refined AdapterOnly. '
        'This contrasts with the near tie at 50k training tracks: the measured benefit depends strongly on the supervised training-label budget in this recipe. '
        'Mamba3 has the lowest errors among the eight fits on the reported aggregate metrics, subject to the single-seed and native-precision caveats below.', '',
        '| Input | Mamba2 head layers | Component MAE [MeV] | Magnitude MAE [MeV] | Relative width68 [%] | Theta MAE [deg] | Phi MAE [deg] |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        report.append(f"| {LABELS[r['model']]} | {r['refinement_layers']} | {r['component_mae_mev']:.3f} | {r['p_mae_mev']:.3f} | {r['relative_width68_pct']:.3f} | {r['theta_mae_deg']:.3f} | {r['phi_mae_deg']:.3f} |")
    report.extend(['', 'Table: all 50,000 evaluation tracks. Quality/component/binned plots: the identical 49,803 CVT-valid tracks for every model/reference.', '',
        '## Gains against AdapterOnly with the same refinement count', '',
        '| Frozen backbone | Head layers | Component MAE reduction | Magnitude MAE reduction | Width68 reduction |', '|---|---:|---:|---:|---:|'])
    for layers in (0, 1):
        for v in VERSIONS[1:]:
            g = [100 * (1 - full[key(v, layers)][f] / full[key('adapteronly', layers)][f]) for f, _ in METRICS[:3]]
            report.append(f'| {v} | {layers} | {g[0]:+.2f}% | {g[1]:+.2f}% | {g[2]:+.2f}% |')
    report.extend(['', 'Positive reductions mean lower error. This single seed and single 1k subset do not establish statistical equivalence or a reliable seed/subset-averaged advantage.', '',
        '## Refinement effects and dependence on training labels', '',
        'Adding refinement reduces component MAE by 16.36% for AdapterOnly and 2.25–3.19% for frozen backbones, and improves theta/phi MAE for every input. '
        'Its effect on momentum magnitude is mixed: AdapterOnly improves 4.59%, Mamba1 worsens 4.01%, Mamba2 improves 0.26%, and Mamba3 improves 1.92%. '
        'AdapterOnly width68 worsens from 23.825% to 26.587%, and its >10% relative-error tail grows from 64.154% to 67.802%. '
        'Mamba3 width68 improves from 10.448% to 9.783%, with its tail falling from 34.062% to 31.126%. '
        'These checkpoints were selected by standardized Cartesian component MAE; that objective need not improve every derived physical metric.', '',
        'At 50k training tracks, refined AdapterOnly and refined Mamba3 had component MAE 27.736 and 27.858 MeV and magnitude MAE 49.806 and 50.432 MeV. '
        'At 1k they have component MAE 176.235 and 82.427 MeV and magnitude MAE 321.578 and 141.116 MeV. '
        'The magnitude-error increase from 50k to 1k is therefore 6.46 times for refined AdapterOnly versus 2.80 times for refined Mamba3. '
        'Pretraining helps substantially at low training-label count, but does not recover 50k-label performance. '
        'All 1k fits also remain worse than the CVT reference on the common-population aggregate metrics. '
        'The momentum-bin plots show the pretrained advantage across all displayed bins, rather than only in the overall average.', '',
        'This comparison measures frozen pretrained representations versus the established AdapterOnly input path. '
        'Without a matched frozen-random-backbone control it does not isolate learned pretraining weights from every effect of adding a frozen backbone. '
        'It also does not measure fine-tuning, annotation-budget scaling exponents, or seed/subset uncertainty.', '',
        '## Recorded training resources', '',
        '| Input | Head layers | Best step | Training hours | Peak allocated CUDA GiB |',
        '|---|---:|---:|---:|---:|'])
    for r in rows:
        report.append(f"| {LABELS[r['model']]} | {r['refinement_layers']} | {r['best_step']:,} | {r['training_seconds']/3600:.3f} | {r['peak_cuda_gib']:.3f} |")
    report.extend(['',
        'The original zero-worker data path repeatedly recomputes deterministic Hilbert/kNN preprocessing. '
        'A separate CPU-only diagnostic identified this input bottleneck; the active fits were preserved. '
        'Runtimes are end-to-end training records including validation/audits; CUDA allocation is not total host or device memory. '
        'These values should not be used as architecture-only speed comparisons.', '',
        '## Recipe and provenance', '',
        f"The training subset has 1,000 tracks from {ready['training_source_events']} source events, sampled without replacement using numpy.default_rng(42) from the original 50k training cohort. "
        'The 10k validation and 50k evaluation cohorts are retained exactly. Their source events are disjoint. Thus 1k refers to training labels, not a strict 1k-total-annotation budget: '
        '10k validation labels are still used for selection. Final evaluation is independent of checkpoint selection, conditional on the existing dataset split.', '',
        'Only the selected 1k tracks determine momentum target means/stds. Starting target-normalization buffers were replaced accordingly; other initial head tensors match the original campaign. '
        'Native evaluation truth, identities and CVT vectors match the prior 50k-label experiment row by row. Standardized validation curves are only compared within the 1k experiment; '
        'cross-label-budget plots use physical errors because target scales differ.', '',
        'Seed42; 30k finite optimizer updates; batch128 (last batch retained); validation every1k updates; best-validation-loss checkpoint; AdamW LR1e-5→3e-4→1e-5, '
        '1k warmup, cosine schedule, weight decay1e-4, norm clipping0.1, dropout0.1; Cartesian entrance momentum targets; standardized component MAE; attention pooling; '
        'width512 input /256 adapter. Zero DataLoader workers were used. No other training hyperparameter was tuned. '
        'Both input-embedder Mamba counts remain zero. One-layer heads use Mamba2 state64/conv4/expand2 for every input family.', '',
        'Heads have 232,720 total/200,464 declared-trainable parameters without refinement and 664,744/632,488 with refinement, a difference of 432,024. '
        'Both include 32,256 fixed coordinate-encoding parameters; 12 depth weights are unused in AdapterOnly mode. Capacity and sequence processing change together in the refinement ablation. '
        'Backbones stay frozen and unchanged. All fits ran on a02c using an immutable verified source/image; primary checkouts and original checkpoints were preserved. '
        'Mamba2 was copied from a518 with source/local/destination SHA256 verification. Deleted v7 data and compiler caches were not required; the retained v6 cohort content was freshly fingerprinted. '
        'Heads are FP32; Mamba3 retains native local BF16 blocks while Mamba1/2 are FP32. Runtimes include validation/audit and are not architecture-only speed evidence.', '',
        'Raw histories, all/common-population metrics, gains, refinement effects, momentum bins, runtimes/peak CUDA allocation and physical 1k-versus-50k comparisons are saved as JSON/CSV. '
        'Five figures are exported as PNG/PDF. These are momentum fits on truth-associated track segments with supplied membership, not end-to-end track reconstruction.', ''])
    (out / 'report.md').write_text('\n'.join(report))
    with zipfile.ZipFile(root / 'momentum_1k_comparison.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            if f.is_file(): z.write(f, 'comparison/' + f.name)
    print({'complete': True, 'output': str(out), 'runs': 8})


if __name__ == '__main__': main()
