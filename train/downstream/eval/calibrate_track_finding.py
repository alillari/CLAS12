"""Select a Dice or quality-efficiency assignment threshold on validation only."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from train.downstream import track_finding_metrics as metrics
from train.downstream.eval.finding_prediction_archive import load_archive, probability, sha256, write_json
from fm4npp.datasets.track_eligibility import PRIMARY_TRACK_POLICY


@dataclass
class EventCache:
    scores: np.ndarray
    points: np.ndarray
    pred_code: np.ndarray
    truth_code: np.ndarray
    truth_sizes: np.ndarray
    n_pred: int


def cache_event(event):
    truth, pred = event['truth'], event['pred']
    tids = sorted(np.unique(truth[truth >= 0]), key=lambda t: int(event['point_ids'][truth == t].min()))
    truth_code = np.full(len(truth), -1, dtype=np.int64)
    sizes = []
    for i, t in enumerate(tids):
        mask = truth == t
        truth_code[mask] = i
        sizes.append(int(mask.sum()))
    gate = event['signal'] & (pred >= 0)
    pids = np.unique(pred[gate])
    return EventCache(event['scores'][gate], event['point_ids'][gate],
                      np.searchsorted(pids, pred[gate]), truth_code[gate],
                      np.asarray(sizes, dtype=np.int64), len(pids))


def score_cached(cache, threshold, config):
    """Exact contingency-table counterparts of the shared two independent matches."""
    nt = len(cache.truth_sizes)
    if nt == 0 or cache.n_pred == 0:
        return 0., 0, 0, 0
    keep = cache.scores > threshold
    pc, tc = cache.pred_code[keep], cache.truth_code[keep]
    good = tc >= 0
    sizes = np.bincount(pc, minlength=cache.n_pred)
    inter = np.bincount(pc[good] * nt + tc[good], minlength=cache.n_pred * nt).reshape(cache.n_pred, nt)
    alive = np.flatnonzero(sizes > 0)
    if not len(alive):
        return 0., 0, 0, 0
    overlap_all = inter[alive]
    psizes = sizes[alive, None]
    union = psizes + cache.truth_sizes[None, :] - overlap_all
    iou = overlap_all / union
    eligible = ((iou >= config.iou_threshold) & (overlap_all / psizes >= config.min_purity)
                & (overlap_all / cache.truth_sizes[None, :] >= config.min_efficiency))
    weights = np.where(eligible, min(len(alive), nt) + 1 + iou, 0.)
    a, b = linear_sum_assignment(-weights)
    qualified = int(eligible[a, b].sum())
    if not good.any():
        return 0., 0, 0, qualified
    pmin = np.full(cache.n_pred, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(pmin, pc, cache.points[keep])
    active = np.flatnonzero(inter.sum(axis=1) > 0)
    active = active[np.argsort(pmin[active], kind='stable')]
    overlap = inter[active]
    dice = 2. * overlap / (sizes[active, None] + cache.truth_sizes[None, :])
    js, its = linear_sum_assignment(-dice)
    positive = overlap[js, its] > 0
    js, its = js[positive], its[positive]
    return float(dice[js, its].sum()), int(overlap[js, its].sum()), int(sizes[active[js]].sum()), qualified


def summarize_scan(caches, threshold, config):
    nt = sum(len(c.truth_sizes) for c in caches)
    ns = sum(int(c.truth_sizes.sum()) for c in caches)
    if not nt:
        raise ValueError('Calibration requires at least one eligible truth track')
    totals = [score_cached(c, threshold, config) for c in caches]
    dice = sum(x[0] for x in totals); correct = sum(x[1] for x in totals)
    associated = sum(x[2] for x in totals); qualified = sum(x[3] for x in totals)
    return dict(threshold=float(threshold), mc_track_dice_mean=dice / nt,
                mc_hit_completeness_global=correct / ns,
                mc_hit_purity_global=correct / associated if associated else None,
                quality_efficiency=qualified / nt, n_qualified_tracks=qualified,
                n_true_tracks=nt, n_mc_correct_points=correct, n_mc_associated_pred_points=associated)


def select_operating_point(rows, objective):
    if objective not in ('dice', 'quality-efficiency'):
        raise ValueError('Objective must be dice or quality-efficiency')
    rows = list(rows)
    if not rows:
        raise ValueError('Threshold scan is empty')
    if objective == 'quality-efficiency':
        best = max(x['n_qualified_tracks'] for x in rows)
        rows = [x for x in rows if x['n_qualified_tracks'] == best]
    best_dice = max(x['mc_track_dice_mean'] for x in rows)
    rows = [x for x in rows if best_dice - x['mc_track_dice_mean'] <= 1e-12]
    return max(rows, key=lambda x: (x['n_mc_correct_points'],
               (x['mc_hit_purity_global'] or 0.) if objective == 'dice' else 0., -x['threshold']))


def score_full(archive, threshold, config):
    total = metrics.MCReconstructionAccumulator(); qualified = 0
    final = archive.final_predictions(threshold)
    for e in range(archive.n_events):
        event = archive.event(e); pred = final[archive.event_slice(e)]
        total.add(metrics.event_mc_reconstruction_metrics(event['truth'], pred,
            provenance=event.get('provenance'), point_ids=event['point_ids']))
        qualified += metrics.event_track_metrics(event['truth'], pred, config=config)['n_matched_tracks']
    result = total.summary()
    nt = result['n_true_tracks']
    result.update(n_qualified_tracks=qualified, quality_efficiency=qualified / nt if nt else None)
    den = result['n_signal_points'] + result['n_mc_associated_pred_points']
    result['mc_hit_f1_micro'] = 2 * result['n_mc_correct_points'] / den if den else None
    return result


def calibrate(validation, output_dir, *, metadata=None, test=None, test_metadata=None,
              truth_policy='as_saved', objective='dice', min_purity=.9, min_completeness=.9,
              min_iou=.5, thresholds=None, refine_steps=(.001, .0001)):
    config = metrics.MatchConfig(min_purity=probability(min_purity, 'min_purity'),
        min_efficiency=probability(min_completeness, 'min_completeness'),
        iou_threshold=probability(min_iou, 'min_iou'))
    if objective not in ('dice', 'quality-efficiency'):
        raise ValueError('Unknown calibration objective')
    if test_metadata and not test:
        raise ValueError('test_metadata requires a test archive')
    grid = sorted({probability(t, 'threshold') for t in (thresholds if thresholds is not None else np.linspace(0, 1, 101))})
    if not grid:
        raise ValueError('Threshold scan is empty')
    steps = [float(step) for step in refine_steps]
    if any(not np.isfinite(step) or step <= 0 or step > 1 for step in steps):
        raise ValueError('Refinement steps must be finite and in (0, 1]')
    archive = load_archive(validation, metadata=metadata, truth_policy=truth_policy, require_scores=True)
    if test:
        archive.physical_keys()  # Fail early when held-out separation cannot be verified.
    caches = [cache_event(archive.event(e)) for e in range(archive.n_events)]
    rows = {t: summarize_scan(caches, t, config) for t in grid}
    for step in steps:
        center = select_operating_point(rows.values(), objective)['threshold']
        for t in sorted({round(center + i * step, 12) for i in range(-10, 11) if 0 <= center + i * step <= 1}):
            if t not in rows:
                rows[t] = summarize_scan(caches, t, config)
    selected = select_operating_point(rows.values(), objective)
    full = score_full(archive, selected['threshold'], config)
    for key in ('mc_track_dice_mean', 'mc_hit_completeness_global', 'mc_hit_purity_global', 'quality_efficiency'):
        a, b = selected[key], full[key]
        if not ((a is None and b is None) or (a is not None and b is not None and np.isclose(a, b, rtol=1e-12, atol=1e-12))):
            raise RuntimeError(f'Cached/shared evaluator parity failure: {key}')
    result = dict(objective=objective, selected=selected, quality_config=dict(vars(config)),
        validation_receipt=archive.receipt, validation_metrics=full,
        search=dict(scope='best scanned threshold', thresholds=sorted(rows), refine_steps=steps),
        truth_policy=truth_policy, selection_split='validation', checkpoint_selection_modified=False,
        tool_source_sha256=sha256(__file__))
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'validation_scan.json', dict(rows=[rows[t] for t in sorted(rows)]))
    write_json(output / 'selection.json', result)
    selection_hash = sha256(output / 'selection.json')
    # No test archive content is opened until the validation choice is persisted.
    heldout = None
    if test:
        heldout_archive = load_archive(test, metadata=test_metadata, truth_policy=truth_policy, require_scores=True)
        if set(archive.physical_keys()) & set(heldout_archive.physical_keys()):
            raise ValueError('Validation and test physical events overlap')
        heldout = dict(receipt=heldout_archive.receipt,
                       metrics=score_full(heldout_archive, selected['threshold'], config))
    if sha256(output / 'selection.json') != selection_hash:
        raise RuntimeError('Selection changed during held-out reporting')
    summary = dict(selection=result, selection_sha256=selection_hash, selected_before_test_reporting=True, heldout=heldout)
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', required=True, type=Path)
    parser.add_argument('--metadata', type=Path, help='Aligned validation metadata NPZ')
    parser.add_argument('--test', type=Path)
    parser.add_argument('--test-metadata', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--objective', choices=['dice', 'quality-efficiency'], default='dice')
    parser.add_argument('--truth-policy', choices=['as_saved', PRIMARY_TRACK_POLICY], default='as_saved')
    parser.add_argument('--min-purity', type=float, default=.9)
    parser.add_argument('--min-completeness', type=float, default=.9)
    parser.add_argument('--min-iou', type=float, default=.5)
    parser.add_argument('--thresholds', nargs='+', type=float)
    parser.add_argument('--refine-steps', nargs='*', type=float, default=[.001, .0001])
    args = parser.parse_args()
    summary = calibrate(**vars(args))
    print(f"Selected threshold {summary['selection']['selected']['threshold']:.7g}; wrote {args.output_dir}")


if __name__ == '__main__':
    main()
