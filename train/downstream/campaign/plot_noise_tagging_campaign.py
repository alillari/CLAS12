"""Plot noise-classification metrics against labeled training-event count.

Uses the track-finding campaign's overview, individual, and presentation layouts,
with binary cluster metrics and no grouping or truth-matching machinery.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from train.downstream.campaign.collate_noise_tagging_campaign import (
    OPERATING_POINTS, collate, read_json, validate_rows, write_tables,
)

# Match the track-finding presentation palette without importing its evaluator.
PAPER, NAVY, RUST = '#F7F6F2', '#18344A', '#B85C3B'
TEXT, MUTED, SOFT = '#202428', '#6B7177', '#E8E6E1'
COLORS = (NAVY, RUST, '#438D80', '#9272A6', '#AD9143', '#668BB2')
METRICS = (('accuracy', 'Accuracy'), ('signal_precision', 'Signal precision'),
           ('signal_recall', 'Signal recall'), ('balanced_accuracy', 'Balanced accuracy'),
           ('noise_precision', 'Noise precision'), ('noise_recall', 'Noise recall'))
POLICY_LABELS = dict(default='Threshold 0.5', retain_95pct='Validation target: 95% signal recall',
                     retain_99pct='Validation target: 99% signal recall',
                     match_coatjava_retention='Validation target: COATJAVA signal recall')


def curve_points(rows, key):
    """Equal-weight seed means, after pooling clusters within each evaluation."""
    by_budget = defaultdict(list)
    for row in rows:
        by_budget[row['labeled_events']].append(row['metrics'][key])
    points = []
    for budget, values in sorted(by_budget.items()):
        valid = [value for value in values if value is not None]
        # An undefined metric remains a gap, never a zero or an invented value.
        all_defined = len(valid) == len(values)
        mean = float(np.mean(valid)) if all_defined else np.nan
        std = float(np.std(valid, ddof=1)) if all_defined and len(valid) > 1 else 0.
        points.append((budget, mean, std, len(valid)))
    return points


def series_catalog(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['series_id']].append(row)
    names = {}
    for key, group in groups.items():
        labels = {row['model_label'] for row in group}
        if len(labels) != 1:
            raise ValueError('A model series has inconsistent display labels')
        suffix = 'balanced' if group[0]['class_balance'] == 'inverse_frequency' else 'unweighted'
        names[key] = f"{group[0]['model_label']} · {suffix}"
    # Different architectures with the same display label must remain distinct.
    for key, name in list(names.items()):
        if list(names.values()).count(name) > 1:
            for duplicate in [k for k, value in names.items() if value == name]:
                names[duplicate] += f' [{duplicate[:6]}]'
    return groups, names


def draw_metric(ax, rows, key, title):
    groups, names = series_catalog(rows)
    for index, series in enumerate(sorted(groups, key=lambda key: names[key])):
        points = curve_points(groups[series], key)
        x, y, std, _ = np.array(points).T
        ax.errorbar(x, y, yerr=std, marker='o', ms=5, lw=2,
                    capsize=3, color=COLORS[index % len(COLORS)], label=names[series])
    baseline = rows[0]['coatjava'][key]
    if baseline is not None:
        ax.axhline(baseline, color=TEXT, linestyle='--', lw=1.6, label='COATJAVA unassigned')
    budgets = sorted({row['labeled_events'] for row in rows})
    ax.set_xscale('log')
    ax.set_xticks(budgets)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlim(budgets[0] / 1.3, budgets[-1] * 1.3)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.))
    ax.set(title=title, xlabel='Labeled training events', ylabel=title)
    ax.set_facecolor(PAPER)
    ax.grid(axis='y', color=SOFT, lw=.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(colors=TEXT, labelsize=9)


def render_group(rows, output, formats):
    output.mkdir(parents=True, exist_ok=True)
    comparison = rows[0]['comparison']
    title = (f"{POLICY_LABELS[rows[0]['operating_point']]} · {rows[0]['optimizer_steps']:,} updates\n"
             f"{comparison['validation_events']:,} validation / {comparison['test_events']:,} test events")
    footer = ('Cluster metrics; mean ±1 SD across repeated seeds.\n'
              'Undefined class metrics are shown as gaps.')
    outputs = []

    def save(metrics, layout, name):
        nrow, ncol = layout
        size = (14, 8.5) if layout == (2, 3) else (7, 11) if layout == (3, 1) else (15, 5) if ncol == 3 else (8, 5.5)
        fig, axes = plt.subplots(nrow, ncol, figsize=size, squeeze=False, constrained_layout=True)
        fig.patch.set_facecolor(PAPER)
        for ax, (key, label) in zip(axes.flat, metrics):
            draw_metric(ax, rows, key, label)
        fig.suptitle(title, color=TEXT, fontsize=12)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center',
                   ncol=min(3 if size[0] >= 12 else 1, len(labels)),
                   frameon=False, fontsize=9, title=footer, title_fontsize=8)
        for extension in formats:
            path = output / f'{name}.{extension}'
            fig.savefig(path, dpi=180, facecolor=PAPER)
            outputs.append(str(path))
        plt.close(fig)

    save(METRICS, (2, 3), 'overview_vs_labeled_events')
    for key, label in METRICS:
        save([(key, label)], (1, 1), f'{key}_vs_labeled_events')
    for label in ('signal', 'noise'):
        metrics = [('accuracy', 'Accuracy'), (f'{label}_precision', f'{label.title()} precision'),
                   (f'{label}_recall', f'{label.title()} recall')]
        for layout in ((1, 3), (3, 1)):
            save(metrics, layout, f'{label}_presentation_{layout[0]}x{layout[1]}')
    return outputs


def plot_report(report, output, operating_point='default', formats=('png', 'pdf')):
    validate_rows(report['rows'])
    groups = defaultdict(list)
    for row in report['rows']:
        if row['operating_point'] == operating_point:
            groups[row['comparison_id']].append(row)
    if not groups:
        raise ValueError(f'No completed evaluations for operating point {operating_point}')
    outputs = []
    for comparison_id, rows in sorted(groups.items()):
        outputs.extend(render_group(rows, Path(output) / operating_point / comparison_id[:12], formats))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest', type=Path)
    source.add_argument('--run-table', type=Path, help='Previously collated run_table.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--operating-point', choices=OPERATING_POINTS, default='default')
    parser.add_argument('--formats', nargs='+', choices=('png', 'pdf'), default=['png', 'pdf'])
    args = parser.parse_args()
    report = collate(args.manifest) if args.manifest else read_json(args.run_table)
    write_tables(report, args.output_dir)
    paths = plot_report(report, args.output_dir, args.operating_point, args.formats)
    print(f'Wrote {len(paths)} plots to {args.output_dir}')


if __name__ == '__main__':
    main()
