"""Measurement-aware plots for selected paired track-finding diagnostics."""
from __future__ import annotations

import numpy as np
import textwrap

BLUE = '#147ba6'
PURPLE = '#8755a3'
ORANGE = '#df7d20'
RED = '#c84748'
PINK = '#bf72a7'
GREY = '#c5ccd3'


def measurement_support(event, index):
    """Saved CLAS12 BST/BMT support; never a fitted or synthetic trajectory."""
    a, b = event['endpoints'][index, :3], event['endpoints'][index, 3:]
    detector = event.get('detector')
    if (detector is not None and detector[index] == 1 and abs(a[2] - b[2]) < 1e-5
            and np.linalg.norm(a[:2] - b[:2]) > 1):
        phi = np.unwrap(np.arctan2([a[1], b[1]], [a[0], b[0]]))
        angles = np.linspace(*phi, 70)
        r = (np.linalg.norm(a[:2]) + np.linalg.norm(b[:2])) / 2
        return np.column_stack([r * np.cos(angles), r * np.sin(angles), np.full(70, a[2])])
    return np.linspace(a, b, 40)


def _geometry(ax, event, stat, tid, view):
    xyz = event['xyz']; pred = event['pred']; target = event['truth'] == tid
    chosen = pred == stat['pred_id'] if stat['pred_id'] is not None else np.zeros(len(pred), bool)
    correct = target & chosen
    fragment = target & (pred >= 0) & ~chosen
    rejected = target & (pred < 0)
    overlay = chosen & ~target & (event.get('provenance', np.zeros(len(pred))) == 2)
    other_mc = chosen & ~target & (event['truth'] >= 0)
    other = chosen & ~target & ~overlay & ~other_mc
    x, y = (xyz[:, 0], xyz[:, 1]) if view == 'xy' else (xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))
    if 'endpoints' in event:
        for i in np.flatnonzero(target | chosen):
            pts = measurement_support(event, i)
            sx, sy = (pts[:, 0], pts[:, 1]) if view == 'xy' else (pts[:, 2], np.hypot(pts[:, 0], pts[:, 1]))
            color = BLUE if target[i] else ORANGE if overlay[i] else RED if other_mc[i] else PINK
            ax.plot(sx, sy, color=color, alpha=.22, lw=.9, zorder=1)
    context = ~target & ~chosen
    ax.scatter(x[context], y[context], s=9, c=GREY, alpha=.5, zorder=2)
    for mask, color, marker in ((correct, BLUE, 'o'), (fragment, PURPLE, 'D'),
                                (overlay, ORANGE, '^'), (other_mc, RED, 'v'), (other, PINK, 's')):
        ax.scatter(x[mask], y[mask], s=40, c=color, marker=marker, edgecolors='white', lw=.5, zorder=4)
    ax.scatter(x[rejected], y[rejected], s=48, facecolors='white', edgecolors=BLUE, lw=1.3, zorder=5)
    if view == 'xy':
        ax.set(xlabel='x [cm]', ylabel='y [cm]'); ax.set_aspect('equal', adjustable='box')
    else:
        ax.set(xlabel='z [cm]', ylabel='Radius [cm]')
    # Center-based limits are shared between methods and do not imply 3D crossings.
    margin = max(1., float(np.ptp(x)) * .06)
    ax.set_xlim(float(x.min()) - margin, float(x.max()) + margin)
    margin = max(1., float(np.ptp(y)) * .06)
    ax.set_ylim(float(y.min()) - margin, float(y.max()) + margin)
    ax.grid(alpha=.13)


def plot_comparison(events, selected, baseline, focus_model, output_stem):
    """Write PNG/SVG; include every named method in the assignment audit panels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from pathlib import Path

    paths = [Path(str(output_stem) + '.' + ext) for ext in ('png', 'svg')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Event plot outputs must be new')
    tid = selected['true_id']; methods = list(events); reference = events[baseline]
    truth = reference['truth']; ids = np.flatnonzero(truth == tid)
    ids = ids[np.argsort(reference['point_ids'][ids], kind='stable')]
    geometry = 'xyz' in reference
    rows = 3 if geometry else 1
    height = (10.5 if geometry else 6.2) + max(0, len(methods) - 5) * .28
    display_names = [textwrap.fill(name, width=18) for name in methods]
    long_names = max(len(name) for name in methods) > 12
    left = .17 if long_names else .075
    stats = selected['methods']
    titles = {'overlay': 'Verified overlay clusters enter the prediction',
              'fragmentation': 'One MC track is fragmented across predictions',
              'mixing': 'Clusters from different MC tracks are mixed',
              'miss': 'The Dice association misses this MC track',
              'incomplete': 'The reconstructed track loses true clusters'}
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'}):
        fig = plt.figure(figsize=(13.5, height))
        gs = fig.add_gridspec(rows, 2, left=left, right=.975, bottom=.20 if geometry else .32, top=.77,
                             hspace=.5, wspace=.45 if long_names else .25, height_ratios=[1, 1, .8] if geometry else [1])
        fig.text(.075, .955, titles[selected['category']], fontsize=18, weight='semibold')
        ident = selected['physical_identity']
        fig.text(.075, .918, f"Source {ident['source_id']} · ordinal {ident['original_ordinal']} · MC track {tid} · {len(ids)} true clusters", fontsize=11)
        for j, name in enumerate((baseline, focus_model)):
            stat = stats[name]
            fig.text(.075 + j * .50, .865, textwrap.shorten(name, width=48, placeholder='…'), fontsize=13, weight='semibold')
            purity = f"{stat['purity']:.1%}" if stat['purity'] is not None else 'undefined'
            fig.text(.075 + j * .50, .828, f"Purity {purity} · completeness {stat['completeness']:.1%} · Dice {stat['dice']:.3f}", fontsize=10)
            if geometry:
                for i, view in enumerate(('xy', 'zr')):
                    _geometry(fig.add_subplot(gs[i, j]), events[name], stat, tid, view)
        ax = fig.add_subplot(gs[-1, 0]); matrix = np.ones((len(methods), len(ids), 4))
        for row, name in enumerate(methods):
            pred = events[name]['pred']; matched = stats[name]['pred_id']
            for col, i in enumerate(ids):
                p = int(pred[i]); color = BLUE if p >= 0 and p == matched else PURPLE if p >= 0 else '#ebeff3'
                matrix[row, col] = matplotlib.colors.to_rgba(color)
                ax.text(col, row, str(p) if p >= 0 else '–', ha='center', va='center',
                        fontsize=max(5, min(9, 110 / len(ids))), color='white' if p >= 0 else '#687888')
        ax.imshow(matrix, aspect='auto', interpolation='nearest')
        ax.set(yticks=range(len(methods)), yticklabels=display_names, xticks=range(len(ids)),
               xticklabels=[str(int(reference['point_ids'][i])) for i in ids],
               xlabel='Original cluster ID; cell = prediction ID, – = rejected')
        ax.tick_params(axis='x', labelsize=7)
        ax.set_xticks(np.arange(-.5, len(ids), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(methods), 1), minor=True)
        ax.grid(which='minor', color='white', lw=1.5); ax.tick_params(which='minor', length=0)
        ax.set_title('Assignment of target clusters', loc='left', fontsize=11)
        ax = fig.add_subplot(gs[-1, 1]); left = np.zeros(len(methods))
        for key, color in (('intersection', BLUE), ('overlay', ORANGE), ('other_mc', RED), ('other_nontarget', PINK)):
            values = np.array([stats[name][key] or 0 for name in methods])
            ax.barh(range(len(methods)), values, left=left, color=color, height=.65)
            for i, (v, l) in enumerate(zip(values, left)):
                if v:
                    ax.text(l + v / 2, i, str(v), color='white', va='center', ha='center', fontsize=9)
            left += values
        ax.set(yticks=range(len(methods)), yticklabels=display_names, ylim=(len(methods) - .5, -.5),
               xlabel=f'Associated-prediction clusters | dotted = {len(ids)} true')
        ax.set_title('Correct clusters and attached contamination', loc='left', fontsize=11)
        ax.axvline(len(ids), color='.4', ls=':', lw=1); ax.set_xlim(0, max(len(ids), left.max()) + 1)
        ax.xaxis.get_major_locator().set_params(integer=True)
        handles = [Line2D([], [], marker=m, ls='', color=c, label=l) for m,c,l in (
            ('o', BLUE, 'Correct target'), ('D', PURPLE, 'Target in another prediction'),
            ('^', ORANGE, 'Attached verified overlay'), ('v', RED, 'Attached other MC'),
            ('s', PINK, 'Attached other non-target / unknown'))]
        handles.append(Line2D([], [], marker='o', ls='', mfc='white', mec=BLUE, label='Rejected target'))
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .075 if geometry else .105), ncol=3, frameon=False, fontsize=9)
        text = ('Saved cluster centers and measurement supports; no fitted trajectories. Centers are not measured 3D crossings.'
                if geometry else 'Cluster-assignment audit; no geometry was supplied.')
        fig.text(.075, .05, text, fontsize=9, color='#526579')
        fig.text(.075, .027, 'Selected illustrative quality failure; not a typical-event claim. Thresholds are fixed inputs from prior calibration.', fontsize=9, color='#526579')
        try:
            for p in paths:
                fig.savefig(p, dpi=160)
        finally:
            plt.close(fig)
