"""Find and plot paired track-finding failures from arbitrary named predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train.downstream import track_finding_metrics as metrics
from train.downstream.eval.finding_prediction_archive import (
    PredictionArchive, align_to, load_archive, probability, sha256, write_json,
)
from fm4npp.datasets.track_eligibility import PRIMARY_TRACK_POLICY

CATEGORIES = ('overlay', 'fragmentation', 'mixing', 'miss', 'incomplete')


def select_examples(candidates, maximum):
    """Cycle categories, advancing past duplicates before choosing each example."""
    queues = {category: iter(rows) for category, rows in candidates.items()}
    selected = []; used = set()
    while len(selected) < maximum:
        progress = False
        for category, queue in queues.items():
            row = next((r for r in queue if (r['event_index'], r['true_id']) not in used), None)
            if row is not None:
                selected.append(dict(category=category, **row))
                used.add((row['event_index'], row['true_id'])); progress = True
                if len(selected) == maximum:
                    break
        if not progress:
            break
    return selected


def track_diagnostics(truth, pred, *, provenance=None, point_ids=None, config=None):
    """Associate once by Dice; determine quality eligibility with a fresh matcher."""
    config = config or metrics.MatchConfig(min_purity=.9, min_efficiency=.9)
    row = metrics.event_mc_reconstruction_metrics(truth, pred, provenance=provenance, point_ids=point_ids)
    accepted = metrics.event_track_metrics(truth, pred, config=config)
    qualified = {m['true_id']: m['pred_id'] for m in accepted['matches']}
    matches = {m['true_id']: m for m in row['dice_matches']}
    tracks = []
    for tid in sorted(np.unique(truth[truth >= 0]).tolist()):
        true = truth == tid
        match = matches.get(tid)
        pid = match['pred_id'] if match else None
        selected = pred == pid if pid is not None else np.zeros(len(pred), dtype=bool)
        inter = int((true & selected).sum()); size = int(selected.sum()); nt = int(true.sum())
        overlay = int((selected & ~true & (provenance == metrics.ClusterProvenance.OVERLAY_ONLY)).sum()) if provenance is not None else None
        other_mc = int((selected & (truth >= 0) & ~true).sum())
        overlaps = [dict(pred_id=int(p), intersection=int((true & (pred == p)).sum()), pred_size=int((pred == p).sum()))
                    for p in np.unique(pred[true & (pred >= 0)])]
        tracks.append(dict(true_id=tid, true_size=nt, pred_id=pid, intersection=inter, pred_size=size,
            purity=inter / size if size else None, completeness=inter / nt, dice=match['dice'] if match else 0.,
            qualified_pred_id=qualified.get(tid), overlay=overlay, other_mc=other_mc,
            other_nontarget=size - inter - other_mc - (overlay or 0),
            rejected=int((true & (pred < 0)).sum()), overlaps=overlaps))
    return dict(n_true_tracks=row['n_true_tracks'], n_qualified_tracks=accepted['n_matched_tracks'],
        dice_sum=row['mc_track_dice_sum'], tracks=tracks), row


def failure_categories(reference, model, config):
    if reference['qualified_pred_id'] is None or model['qualified_pred_id'] is not None:
        return []
    result = []
    if model['overlay'] and (model['purity'] or 0.) < config.min_purity:
        result.append('overlay')
    if sum(x['intersection'] >= 2 for x in model['overlaps']) >= 2:
        result.append('fragmentation')
    if model['other_mc'] > 0:
        result.append('mixing')
    if model['intersection'] == 0:
        result.append('miss')
    if model['completeness'] < config.min_efficiency:
        result.append('incomplete')
    return result


def paired_efficiency_interval(reference_counts, model_counts, truth_counts, *, resamples=1000, seed=42):
    """Paired complete-event bootstrap of pooled quality-efficiency differences."""
    a, b, n = [np.asarray(x) for x in (reference_counts, model_counts, truth_counts)]
    if (a.ndim != 1 or a.shape != b.shape or a.shape != n.shape or not len(n)
            or not np.isfinite(np.concatenate([a, b, n])).all()
            or np.any((a < 0) | (b < 0) | (n < 0) | (a > n) | (b > n))
            or n.sum() <= 0 or resamples < 1):
        raise ValueError('Invalid paired event counts or bootstrap resamples')
    rng = np.random.default_rng(seed); values = []
    for _ in range(resamples):
        ix = rng.integers(len(n), size=len(n)); den = n[ix].sum()
        if den:
            values.append(float((a[ix] - b[ix]).sum() / den))
    return dict(reference_minus_model=float((a-b).sum() / n.sum()),
                ci95=np.quantile(values, [.025, .975]).tolist() if values else None,
                resamples=resamples, valid_resamples=len(values), seed=seed,
                uncertainty='paired test-event sampling; training-seed variation excluded')


def _spec_archive(spec, root, truth_policy):
    if spec.get('kind') not in ('raw', 'final'):
        raise ValueError('Each prediction spec must explicitly declare kind: raw or final')
    if not isinstance(spec.get('name'), str) or not spec['name'].strip():
        raise ValueError('Each prediction requires a nonempty name')
    raw = spec['kind'] == 'raw'
    if raw != (spec.get('assignment_threshold') is not None):
        raise ValueError('Raw predictions require assignment_threshold; final predictions must omit it')
    archive = load_archive(root / spec['path'], metadata=root / spec['metadata'] if spec.get('metadata') else None,
        prediction_key=spec.get('prediction_key', 'pred'), truth_policy=truth_policy, require_scores=raw)
    pred = archive.final_predictions(spec.get('assignment_threshold'))
    archive.arrays['pred'] = pred
    archive.receipt.update(name=spec['name'], kind=spec['kind'], assignment_threshold=spec.get('assignment_threshold'))
    return archive


def compare_events(manifest, output_dir, *, focus_model, min_purity=.9, min_completeness=.9,
                   min_iou=.5, categories=CATEGORIES, max_examples=3, min_truth_clusters=4,
                   bootstrap_resamples=1000, seed=42, plots=True):
    if max_examples < 1 or min_truth_clusters < 1 or bootstrap_resamples < 1:
        raise ValueError('Example, truth-cluster and bootstrap counts must be positive')
    categories = tuple(categories)
    if not categories or any(c not in CATEGORIES for c in categories):
        raise ValueError('Unknown or empty failure categories')
    config = metrics.MatchConfig(min_purity=probability(min_purity, 'min_purity'),
        min_efficiency=probability(min_completeness, 'min_completeness'), iou_threshold=probability(min_iou, 'min_iou'))
    path = Path(manifest); plan = json.loads(path.read_text()); root = path.resolve().parent
    truth_policy = plan.get('truth_policy', 'as_saved')
    if truth_policy not in ('as_saved', PRIMARY_TRACK_POLICY):
        raise ValueError('Unknown truth policy')
    specs = [plan['baseline'], *plan['models']]
    names = [spec['name'] for spec in specs]
    if len(names) < 2 or len(set(names)) != len(names) or focus_model not in names[1:]:
        raise ValueError('Names must be unique and focus_model must name a model')
    baseline = _spec_archive(specs[0], root, truth_policy)
    baseline.physical_keys()
    if not baseline.receipt['original_point_ids_supplied']:
        raise ValueError('Paired comparisons require original point_ids')
    archives = {names[0]: baseline}
    for spec in specs[1:]:
        other = _spec_archive(spec, root, truth_policy); ix = align_to(baseline, other)
        archives[spec['name']] = PredictionArchive({**baseline.arrays, 'pred': other.arrays['pred'][ix]}, other.receipt)
    totals = {n: metrics.MCReconstructionAccumulator() for n in names}
    diagnostics = {n: [] for n in names}; candidates = {c: [] for c in categories}
    for e in range(baseline.n_events):
        for name, archive in archives.items():
            event = archive.event(e)
            detail, row = track_diagnostics(event['truth'], event['pred'], provenance=event.get('provenance'), point_ids=event['point_ids'], config=config)
            diagnostics[name].append(detail); totals[name].add(row)
        ref = {t['true_id']: t for t in diagnostics[names[0]][e]['tracks']}
        for model in diagnostics[focus_model][e]['tracks']:
            if model['true_size'] < min_truth_clusters:
                continue
            reference = ref[model['true_id']]
            for category in failure_categories(reference, model, config):
                if category in candidates:
                    candidates[category].append(dict(event_index=e, true_id=model['true_id'],
                        dice_advantage=reference['dice'] - model['dice']))
    for rows in candidates.values():
        rows.sort(key=lambda r: (-r['dice_advantage'], r['event_index'], r['true_id']))
    selected = select_examples(candidates, max_examples)
    nt = np.array([d['n_true_tracks'] for d in diagnostics[names[0]]])
    ref_counts = np.array([d['n_qualified_tracks'] for d in diagnostics[names[0]]])
    paired = {}
    for name in names[1:]:
        counts = [d['n_qualified_tracks'] for d in diagnostics[name]]
        paired[name] = paired_efficiency_interval(ref_counts, counts, nt, resamples=bootstrap_resamples, seed=seed) if nt.sum() else None
    keys = baseline.physical_keys()
    for row in selected:
        e, tid = row['event_index'], row['true_id']; row['physical_identity'] = dict(source_id=keys[e][0], original_ordinal=keys[e][1])
        row['methods'] = {name: next(t for t in diagnostics[name][e]['tracks'] if t['true_id'] == tid) for name in names}
    result = dict(baseline=names[0], focus_model=focus_model, truth_policy=truth_policy,
        quality_config=dict(vars(config)), inputs={n: a.receipt for n,a in archives.items()},
        manifest_sha256=sha256(path), tool_source_sha256=sha256(__file__),
        summaries={n: a.summary() for n,a in totals.items()}, paired_quality_efficiency=paired,
        candidate_counts={c: len(v) for c,v in candidates.items()}, selected=selected,
        selection='Illustrative quality failures; selected examples are not an unbiased sample')
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'comparison.json', result)
    if plots:
        from train.downstream.eval.plot_finding_events import plot_comparison
        for i, row in enumerate(selected):
            e = row['event_index']
            plot_comparison({n:a.event(e) for n,a in archives.items()}, row, names[0], focus_model,
                            out / f'example_{i+1:02d}_{row["category"]}')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--focus-model', required=True)
    p.add_argument('--min-purity', type=float, default=.9)
    p.add_argument('--min-completeness', type=float, default=.9)
    p.add_argument('--min-iou', type=float, default=.5)
    p.add_argument('--categories', nargs='+', choices=CATEGORIES, default=list(CATEGORIES))
    p.add_argument('--max-examples', type=int, default=3)
    p.add_argument('--min-truth-clusters', type=int, default=4)
    p.add_argument('--bootstrap-resamples', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-plots', action='store_true')
    args = vars(p.parse_args()); args['plots'] = not args.pop('no_plots')
    result = compare_events(**args)
    print(f"Wrote {len(result['selected'])} selected comparisons to {args['output_dir']}")


if __name__ == '__main__':
    main()
