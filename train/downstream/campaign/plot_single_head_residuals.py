#!/usr/bin/env python3
"""Inspect cached p/theta predictions without inference or checkpoint reselection.

Older scalar evaluation files contain only the learned truth quantity. Join the
two evaluations by event AND segment identity to obtain truth p for theta plots.
Never substitute a predicted momentum or silently take a cohort intersection.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


IDENTITY = ('source_file', 'event', 'source_event_index', 'segment_label',
            'truth_segment_label', 'adapter_sample_mode')
CONTRACT = 'clas12_momentum_entrance_v1'


def load_evaluation(directory, task):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    if (summary.get('evaluation_contract') != CONTRACT or
            summary.get('training_target_task') != task or summary.get('swingback_enabled') is not False):
        raise ValueError(f'{directory}: expected entrance-only {task} evaluation')
    physics = json.loads((directory / 'physics_checkpoint_summary.json').read_text())
    variable = 'p_gev' if task == 'p' else 'theta_rad'
    path = directory / 'predictions.csv.gz'
    keys, truth, prediction, hits = [], [], [], []
    with gzip.open(path, 'rt') as stream:
        reader = csv.DictReader(stream)
        required = {*IDENTITY, 'n_hits', f'true_{variable}', f'adapter_{variable}'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing columns {required - set(reader.fieldnames or [])}')
        for row in reader:
            key = tuple(row[k] for k in IDENTITY)
            if any(not value for value in key):
                raise ValueError(f'{path}: incomplete event/segment identity {key}')
            keys.append(key)
            truth.append(float(row[f'true_{variable}']))
            prediction.append(float(row[f'adapter_{variable}']))
            hits.append(int(row['n_hits']))
    if len(keys) != len(set(keys)):
        raise ValueError(f'{path}: duplicate event/segment identities')
    if not keys or len(keys) != summary['n_records']:
        raise ValueError(f'{path}: empty or incomplete predictions')
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(directory=str(directory.resolve()), summary=summary, physics=physics,
                keys=keys, truth=np.array(truth), prediction=np.array(prediction),
                hits=np.array(hits), sha256=digest.hexdigest())


def align_reference(primary, reference):
    lookup = {key: i for i, key in enumerate(reference['keys'])}
    if (len(lookup) != len(reference['keys']) or
            len(set(primary['keys'])) != len(primary['keys'])):
        raise ValueError('Duplicate event/segment identities')
    if set(primary['keys']) != set(reference['keys']):
        raise ValueError('Evaluations have different cohorts; refusing a partial truth join')
    order = np.array([lookup[key] for key in primary['keys']], dtype=int)
    if not np.array_equal(primary['hits'], reference['hits'][order]):
        raise ValueError('Matched tracks disagree on hit counts')
    return order


def statistics(values, threshold):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return dict(n=len(values), n_finite=0)
    q = np.quantile(finite, [.025, .16, .5, .84, .975])
    return dict(n=len(values), n_finite=len(finite), q025=q[0], q16=q[1],
                median=q[2], q84=q[3], q975=q[4], w68=(q[3]-q[1])/2,
                w95=(q[4]-q[0])/2, rms=float(np.sqrt(np.mean(finite**2))),
                minimum=float(finite.min()), maximum=float(finite.max()),
                tail_fraction=float(np.mean(abs(finite) > threshold)))


def binned_statistics(x, residual, edges, threshold, min_entries):
    rows = []
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (x >= low) & ((x <= high) if i == len(edges)-2 else (x < high))
        stats = statistics(residual[mask], threshold)
        rows.append(dict(low=float(low), high=float(high), **stats,
                         valid=stats['n'] >= min_entries and stats['n'] == stats['n_finite']))
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, destination):
    for suffix in ('png', 'pdf'):
        fig.savefig(destination.with_suffix('.' + suffix), dpi=180)
    plt.close(fig)


def curve(rows, key):
    x = np.array([(r['low'] + r['high'])/2 for r in rows])
    y = np.array([r.get(key, np.nan) if r['valid'] else np.nan for r in rows])
    return x, y


def residual_maps(data, x, edges, xlabel, tables, output):
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), layout='constrained')
    for ax, task in zip(axes, ('p', 'theta')):
        d = data[task]
        residual, limit = d['residual'], d['core_limit']
        finite = np.isfinite(x) & np.isfinite(residual)
        in_x = finite & (x >= edges[0]) & (x <= edges[-1])
        shown = in_x & (abs(residual) <= limit)
        _, _, _, mesh = ax.hist2d(x[shown], residual[shown],
            bins=[np.linspace(edges[0], edges[-1], 111), np.linspace(-limit, limit, 151)],
            norm=LogNorm(), cmap='viridis', cmin=1)
        fig.colorbar(mesh, ax=ax, label='Tracks per cell (log scale)')
        for key, color, style, label in [('q16', 'white', '--', '16th / 84th percentiles'),
                                        ('q84', 'white', '--', None),
                                        ('median', '#ff7035', '-', 'Median')]:
            ax.plot(*curve(tables[task], key), color=color, ls=style, lw=2, label=label)
        ax.axhline(0, color='black', lw=1)
        omitted = int((in_x & ~shown).sum())
        outside = int((finite & ~in_x).sum())
        ax.set(title=f"{d['label']} | {len(residual):,} tracks | {omitted:,} outside vertical view; {outside:,} outside x range",
               xlabel=xlabel, ylabel=d['ylabel'], xlim=(edges[0], edges[-1]), ylim=(-limit, limit))
        ax.legend(loc='upper right', facecolor='#354456', labelcolor='white', fontsize=9)
    fig.suptitle('Residuals: zero is unbiased; narrow bands mean a narrow core', fontsize=14)
    save_figure(fig, output)


def metric_curves(data, p, edges, tables, output):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex='col', layout='constrained')
    for col, task in enumerate(('p', 'theta')):
        d, rows = data[task], tables[task]
        axes[0, col].plot(*curve(rows, 'median'), 'o-', color='#b84424')
        axes[0, col].axhline(0, color='black', lw=1)
        axes[0, col].set(title=d['label'], ylabel=f"Median residual [{d['unit']}]")
        for key, label in [('w68', 'Central 68% half-width'), ('w95', 'Central 95% half-width')]:
            axes[1, col].plot(*curve(rows, key), 'o-', label=label)
        axes[1, col].set(ylabel=f"Half-width [{d['unit']}]", ylim=(0, None))
        axes[1, col].legend(fontsize=9)
        for threshold in d['thresholds']:
            tail_rows = binned_statistics(p, d['residual'], edges, threshold, d['min_entries'])
            x, y = curve(tail_rows, 'tail_fraction')
            axes[2, col].plot(x, 100*y, 'o-', label=f"|residual| > {threshold:.3g}{d['unit']}")
        axes[2, col].set(ylabel='Tracks beyond threshold [%]', xlabel='True momentum [GeV]', ylim=(0, None))
        axes[2, col].legend(fontsize=9)
        for ax in axes[:, col]:
            ax.grid(alpha=.25)
    fig.suptitle('Bias, core width and tails versus true momentum (all residuals retained)')
    save_figure(fig, output)


def tail_plots(data, p, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    for col, task in enumerate(('p', 'theta')):
        d = data[task]
        r = d['residual']
        finite = np.isfinite(p) & np.isfinite(r)
        ax = axes[0, col]
        ax.scatter(p[finite], r[finite], s=2, alpha=.2, rasterized=True)
        ax.set_yscale('symlog', linthresh=d['thresholds'][0])
        ax.axhline(0, color='black', lw=1)
        ax.set(title=d['label'] + ': full residual range', xlabel='True momentum [GeV]', ylabel=d['ylabel'])
        ax.text(.02, .97, 'Linear near zero; logarithmic tails', va='top', transform=ax.transAxes, fontsize=9)
        absolute = np.sort(abs(r[np.isfinite(r)]))
        # Inclusive survival function P(|residual| >= x); preserve the last track.
        unique, first = np.unique(absolute, return_index=True)
        survival = 100*(len(absolute)-first)/len(absolute)
        axes[1, col].step(unique, survival, where='post')
        axes[1, col].set(xscale='log', yscale='log', xlabel=f"Absolute residual [{d['unit']}]",
                         ylabel='Tracks at or beyond this error [%]')
        for t in d['thresholds']:
            axes[1, col].axvline(t, color='gray', ls=':', alpha=.7)
        axes[1, col].grid(alpha=.25, which='both')
    fig.suptitle('Rare failures: no residual trimming or Gaussian fit')
    save_figure(fig, output)


def checkpoint_review(data, output):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex='col', layout='constrained')
    candidates = {}
    for col, task in enumerate(('p', 'theta')):
        d = data[task]
        path = Path(d['directory']).parents[2] / 'summary/physics_checkpoint_history.jsonl'
        if not path.is_file():
            raise FileNotFoundError(f'{path}: collate the campaign first')
        entries = [json.loads(line)['checkpoint'] for line in path.read_text().splitlines() if line.strip()]
        eligible = [row for row in entries if row['eligible']]
        selected = next(row for row in entries if row['selected'])
        choices = [('Selected by width', selected),
                   ('Lowest validation loss', min(eligible, key=lambda r: r['validation_loss'])),
                   ('Final checkpoint', max(entries, key=lambda r: r['step']))]
        candidates[task] = []
        for label, row in choices:
            candidates[task].append(dict(label=label, **{key: row[key] for key in
                ('step', 'validation_loss', 'W_macro', 'B_macro', 'B_worst', 'T_macro')}))
            bins = [b for b in row['per_bin'] if b['valid']]
            xscale = 1 if task == 'p' else 180/np.pi
            x = [(b['low']+b['high'])/2*xscale for b in bins]
            for ax, key, scale in zip(axes[:, col], ('q50', 'w68', 'tail_fraction'), (d['scale'], d['scale'], 100)):
                ax.plot(x, [b[key]*scale for b in bins], 'o-', label=f"{label}: {row['step']:,}")
        axes[0, col].set(title=d['label'], ylabel=f"Median residual [{d['unit']}]")
        axes[0, col].axhline(0, color='black', lw=1)
        axes[0, col].legend(fontsize=8)
        axes[1, col].set(ylabel=f"Central 68% half-width [{d['unit']}]", ylim=(0, None))
        axes[2, col].set(ylabel=f"Tails beyond {d['thresholds'][0]:.3g}{d['unit']} [%]", ylim=(0, None),
                         xlabel='True momentum [GeV]' if task == 'p' else 'True theta [degrees]')
        for ax in axes[:, col]:
            ax.grid(alpha=.25)
    fig.suptitle('Validation checkpoint tradeoffs in the original selection bins')
    save_figure(fig, output)
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p-evaluation', required=True, type=Path)
    parser.add_argument('--theta-evaluation', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--min-bin-entries', type=int, default=200)
    args = parser.parse_args()
    if args.min_bin_entries < 2:
        parser.error('--min-bin-entries must be at least 2')
    data = {task: load_evaluation(path, task) for task, path in
            [('p', args.p_evaluation), ('theta', args.theta_evaluation)]}
    order = align_reference(data['p'], data['theta'])
    for key in ('truth', 'prediction', 'hits'):
        data['theta'][key] = data['theta'][key][order]
    data['theta']['keys'] = [data['theta']['keys'][i] for i in order]
    p, theta = data['p']['truth'], np.rad2deg(data['theta']['truth'])
    if not np.isfinite(p).all() or np.any(p <= 0) or not np.isfinite(theta).all():
        raise ValueError('Nonfinite or nonpositive momentum / nonfinite angular truth')
    data['p'].update(label='Momentum magnitude head', unit='%', scale=100,
                     ylabel=r'$(p_{pred}-p_{true})/p_{true}$ [%]', core_limit=20)
    data['theta'].update(label='Theta head', unit='deg', scale=180/np.pi,
                         ylabel=r'$\theta_{pred}-\theta_{true}$ [degrees]', core_limit=2)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    p_edges = np.arange(.25, 3.01, .25)
    theta_edges = np.arange(30., 150.1, 5.)
    tables, angular_tables, report = {}, {}, dict(n_matched=len(p), identity_columns=IDENTITY, heads={})
    for task, d in data.items():
        residual = d['prediction'] - d['truth']
        if task == 'p':
            residual = residual / d['truth']
        d['residual'] = residual*d['scale']
        d['min_entries'] = args.min_bin_entries
        threshold = d['physics']['config']['tail_threshold']*d['scale']
        d['thresholds'] = sorted(set([threshold] + ([20., 50.] if task == 'p' else [5., 10.])))
        tables[task] = binned_statistics(p, d['residual'], p_edges, threshold, args.min_bin_entries)
        angular_tables[task] = binned_statistics(theta, d['residual'], theta_edges, threshold, args.min_bin_entries)
        write_csv(output / f'{task}_by_true_momentum.csv', tables[task])
        write_csv(output / f'{task}_by_true_theta.csv', angular_tables[task])
        tails = {str(t): int((abs(d['residual']) > t).sum()) for t in d['thresholds']}
        report['heads'][task] = dict(source=d['directory'], predictions_sha256=d['sha256'],
            inclusive=statistics(d['residual'], threshold), display_unit=d['unit'],
            tail_counts=tails, tail_threshold=threshold, physics_summary=d['physics'],
            by_true_momentum=tables[task], by_true_theta=angular_tables[task])
        worst = np.argsort(np.where(np.isfinite(d['residual']), abs(d['residual']), np.inf))[-30:][::-1]
        write_csv(output / f'{task}_largest_errors.csv', [dict(zip(IDENTITY, d['keys'][i]),
            true_p_gev=p[i], true_theta_deg=theta[i], residual=d['residual'][i],
            residual_unit=d['unit'], n_hits=int(d['hits'][i])) for i in worst])
    residual_maps(data, p, p_edges, 'True momentum [GeV]', tables, output / 'residuals_vs_true_momentum')
    residual_maps(data, theta, theta_edges, 'True theta [degrees]', angular_tables, output / 'residuals_vs_true_theta')
    metric_curves(data, p, p_edges, tables, output / 'bias_width_tails_vs_true_momentum')
    tail_plots(data, p, output / 'full_residual_tails')
    report['checkpoint_candidates'] = checkpoint_review(data, output / 'checkpoint_tradeoffs')
    (output / 'diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Wrote residual diagnostics for {len(p):,} matched tracks to {output.resolve()}')


if __name__ == '__main__':
    main()
