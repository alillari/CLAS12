"""Select track-finding assignment and optional query-confidence cutoffs on validation."""
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
    keep = cache.scores > threshold
    pc, tc = cache.pred_code[keep], cache.truth_code[keep]
    sizes = np.bincount(pc, minlength=cache.n_pred)
    alive = np.flatnonzero(sizes > 0)
    n_pred = len(alive)
    if not nt or not n_pred:
        return 0., 0, 0, 0, n_pred
    good = tc >= 0
    inter = np.bincount(pc[good] * nt + tc[good], minlength=cache.n_pred * nt).reshape(cache.n_pred, nt)
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
        return 0., 0, 0, qualified, n_pred
    pmin = np.full(cache.n_pred, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(pmin, pc, cache.points[keep])
    active = np.flatnonzero(inter.sum(axis=1) > 0)
    active = active[np.argsort(pmin[active], kind='stable')]
    overlap = inter[active]
    dice = 2. * overlap / (sizes[active, None] + cache.truth_sizes[None, :])
    js, its = linear_sum_assignment(-dice)
    positive = overlap[js, its] > 0
    js, its = js[positive], its[positive]
    return (float(dice[js, its].sum()), int(overlap[js, its].sum()),
            int(sizes[active[js]].sum()), qualified, n_pred)


def summarize_scan(caches, threshold, config):
    nt = sum(len(c.truth_sizes) for c in caches)
    ns = sum(int(c.truth_sizes.sum()) for c in caches)
    if not nt:
        raise ValueError('Calibration requires at least one eligible truth track')
    totals = [score_cached(c, threshold, config) for c in caches]
    dice = sum(x[0] for x in totals); correct = sum(x[1] for x in totals)
    associated = sum(x[2] for x in totals); qualified = sum(x[3] for x in totals)
    n_pred = sum(x[4] for x in totals)
    return dict(threshold=float(threshold), mc_track_dice_mean=dice / nt,
                mc_hit_completeness_global=correct / ns,
                mc_hit_purity_global=correct / associated if associated else None,
                quality_efficiency=qualified / nt, n_qualified_tracks=qualified,
                matched_track_f1=2 * qualified / (nt + n_pred), n_pred_tracks=n_pred,
                n_true_tracks=nt, n_mc_correct_points=correct, n_mc_associated_pred_points=associated)


def select_operating_point(rows, objective, min_matched_track_f1=None):
    if objective not in ('dice', 'quality-efficiency'):
        raise ValueError('Objective must be dice or quality-efficiency')
    rows = list(rows)
    if not rows:
        raise ValueError('Threshold scan is empty')
    if min_matched_track_f1 is not None:
        rows = [row for row in rows if row['matched_track_f1'] >= min_matched_track_f1]
        if not rows:
            raise ValueError('No scanned operating point passed the minimum matched-track F1')
    if objective == 'quality-efficiency':
        best = max(x['n_qualified_tracks'] for x in rows)
        rows = [x for x in rows if x['n_qualified_tracks'] == best]
    best_dice = max(x['mc_track_dice_mean'] for x in rows)
    rows = [x for x in rows if best_dice - x['mc_track_dice_mean'] <= 1e-12]
    return max(rows, key=lambda x: (x['n_mc_correct_points'],
               (x['mc_hit_purity_global'] or 0.) if objective == 'dice' else 0., -x['threshold'],
               -x.get('query_confidence_threshold', 0.)))


def score_full(archive, threshold, config, query_threshold=None):
    total = metrics.MCReconstructionAccumulator(); qualified = 0
    final = archive.final_predictions(threshold, query_threshold)
    for e in range(archive.n_events):
        event = archive.event(e); pred = final[archive.event_slice(e)]
        total.add(metrics.event_mc_reconstruction_metrics(event['truth'], pred,
            provenance=event.get('provenance'), point_ids=event['point_ids']))
        qualified += metrics.event_track_metrics(event['truth'], pred, config=config)['n_matched_tracks']
    result = total.summary()
    nt = result['n_true_tracks']
    track_denominator = nt + result['n_pred_tracks']
    result.update(n_qualified_tracks=qualified, quality_efficiency=qualified / nt if nt else None,
                  matched_track_f1=2 * qualified / track_denominator if track_denominator else None)
    den = result['n_signal_points'] + result['n_mc_associated_pred_points']
    result['mc_hit_f1_micro'] = 2 * result['n_mc_correct_points'] / den if den else None
    return result


def scan(archive, rows, query_grid, assignment_grid, config):
    for query_threshold in query_grid:
        pending = [t for t in assignment_grid if (query_threshold, t) not in rows]
        if not pending:
            continue
        if query_threshold is None:
            caches = [cache_event(archive.event(e)) for e in range(archive.n_events)]
        else:
            caches = [cache_event(archive.event_with_query_cut(e, query_threshold))
                      for e in range(archive.n_events)]
        for threshold in pending:
            row = summarize_scan(caches, threshold, config)
            if query_threshold is not None:
                row['query_confidence_threshold'] = query_threshold
            rows[query_threshold, threshold] = row


def refinement_grid(center, step):
    return sorted({round(center + i * step, 12)
                   for i in range(-10, 11) if 0 <= center + i * step <= 1})


def calibrate(validation, output_dir, *, metadata=None, test=None, test_metadata=None,
              truth_policy='as_saved', objective='dice', min_purity=.9, min_completeness=.9,
              min_iou=.5, thresholds=None, refine_steps=(.001, .0001),
              fixed_query_confidence=None, query_confidence_thresholds=None, query_refine_steps=None,
              min_matched_track_f1=None):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Calibration output directory must be new: {output}')
    config = metrics.MatchConfig(min_purity=probability(min_purity, 'min_purity'),
        min_efficiency=probability(min_completeness, 'min_completeness'),
        iou_threshold=probability(min_iou, 'min_iou'))
    if objective not in ('dice', 'quality-efficiency'):
        raise ValueError('Unknown calibration objective')
    if test_metadata and not test:
        raise ValueError('test_metadata requires a test archive')
    grid = sorted({probability(t, 'threshold') for t in (thresholds if thresholds is not None else (i / 100 for i in range(101)))})
    if not grid:
        raise ValueError('Threshold scan is empty')
    steps = [float(step) for step in refine_steps]
    if any(not np.isfinite(step) or step <= 0 or step > 1 for step in steps):
        raise ValueError('Refinement steps must be finite and in (0, 1]')
    if min_matched_track_f1 is not None:
        min_matched_track_f1 = probability(min_matched_track_f1, 'min_matched_track_f1')
    joint = query_confidence_thresholds is not None
    if fixed_query_confidence is not None:
        if joint:
            raise ValueError('Choose either a fixed query-confidence cutoff or a query grid')
        fixed_query_confidence = probability(fixed_query_confidence, 'fixed_query_confidence')
    query_grid = (sorted({probability(q, 'query-confidence threshold') for q in query_confidence_thresholds})
                  if joint else [fixed_query_confidence])
    if not query_grid:
        raise ValueError('Query-confidence grid is empty')
    if query_refine_steps is not None and (not joint or len(query_refine_steps) != len(steps)):
        raise ValueError('Supply one query refinement step per assignment refinement step for a joint scan')
    query_steps = []
    if joint:
        query_steps = ([min(1., 10 * step) for step in steps] if query_refine_steps is None
                       else [float(step) for step in query_refine_steps])
    if any(not np.isfinite(step) or not 0 < step <= 1 for step in query_steps):
        raise ValueError('Query refinement steps must be finite and in (0, 1]')
    archive = load_archive(validation, metadata=metadata, truth_policy=truth_policy, require_scores=True)
    contract = archive.receipt.get('evaluation_contract')
    if contract and contract['split'] != 'validation':
        raise ValueError('Threshold selection requires a validation archive')
    if query_grid != [None] and 'query_confidence' not in archive.arrays:
        raise ValueError('A query-confidence cutoff requires an archive exported with --save-query-scores')
    if test:
        archive.physical_keys()  # Fail early when held-out separation cannot be verified.
    rows = {}
    scan(archive, rows, query_grid, grid, config)
    for i, step in enumerate(steps):
        selectable = [row for row in rows.values()
                      if min_matched_track_f1 is None or row['matched_track_f1'] >= min_matched_track_f1]
        if not selectable:
            # Explore near the most feasible point before declaring the floor unattainable.
            best_f1 = max(row['matched_track_f1'] for row in rows.values())
            selectable = [row for row in rows.values() if row['matched_track_f1'] == best_f1]
        center = select_operating_point(selectable, objective)
        queries = (refinement_grid(center['query_confidence_threshold'], query_steps[i])
                   if joint else query_grid)
        scan(archive, rows, queries, refinement_grid(center['threshold'], step), config)
    selected = select_operating_point(rows.values(), objective, min_matched_track_f1)
    query_threshold = selected.get('query_confidence_threshold')
    full = score_full(archive, selected['threshold'], config, query_threshold)
    for key in ('mc_track_dice_mean', 'mc_hit_completeness_global', 'mc_hit_purity_global',
                'quality_efficiency', 'matched_track_f1', 'n_pred_tracks'):
        a, b = selected[key], full[key]
        if not ((a is None and b is None) or (a is not None and b is not None and np.isclose(a, b, rtol=1e-12, atol=1e-12))):
            raise RuntimeError(f'Cached/shared evaluator parity failure: {key}')
    result = dict(objective=objective, selected=selected, quality_config=dict(vars(config)),
        min_matched_track_f1=min_matched_track_f1,
        validation_receipt=archive.receipt, validation_metrics=full,
        search=dict(scope='best scanned operating point', thresholds=sorted({t for _, t in rows}),
                    refine_steps=steps, query_refine_steps=query_steps,
                    query_confidence_thresholds=sorted({q for q, _ in rows if q is not None})),
        truth_policy=truth_policy, selection_split='validation', checkpoint_selection_modified=False,
        tool_source_sha256=sha256(__file__))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'validation_scan.json', dict(rows=[rows[t] for t in sorted(rows)]))
    write_json(output / 'selection.json', result)
    selection_hash = sha256(output / 'selection.json')
    # No test archive content is opened until the validation choice is persisted.
    heldout = None
    if test:
        heldout_archive = load_archive(test, metadata=test_metadata, truth_policy=truth_policy, require_scores=True)
        if set(archive.physical_keys()) & set(heldout_archive.physical_keys()):
            raise ValueError('Validation and test physical events overlap')
        heldout_contract = heldout_archive.receipt.get('evaluation_contract')
        if heldout_contract and heldout_contract['split'] != 'test':
            raise ValueError('Held-out reporting requires a test archive')
        if contract and heldout_contract:
            if contract != {**heldout_contract, 'split': 'validation'}:
                raise ValueError('Validation and test archive evaluation contracts differ')
        heldout = dict(receipt=heldout_archive.receipt,
                       metrics=score_full(heldout_archive, selected['threshold'], config, query_threshold))
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
    query = parser.add_mutually_exclusive_group()
    query.add_argument('--fixed-query-confidence', type=float)
    query.add_argument('--query-confidence-thresholds', nargs='+', type=float)
    parser.add_argument('--query-refine-steps', nargs='*', type=float)
    parser.add_argument('--min-matched-track-f1', type=float)
    args = parser.parse_args()
    summary = calibrate(**vars(args))
    selected = summary['selection']['selected']
    cutoffs = f"assignment threshold {selected['threshold']:.7g}"
    if 'query_confidence_threshold' in selected:
        cutoffs += f", query confidence {selected['query_confidence_threshold']:.7g}"
    print(f"Selected {cutoffs}. Wrote {args.output_dir}")


if __name__ == '__main__':
    main()
