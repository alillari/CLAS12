"""Collate whole-event noise benchmarks for label-count learning curves."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from ruamel.yaml import YAML

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from train.downstream.noise_metrics import METRIC_DEFINITIONS, confusion_metrics

METRICS = ("accuracy", "signal_precision", "signal_recall", "noise_precision",
           "noise_recall", "balanced_accuracy", "macro_precision", "macro_recall")
OPERATING_POINTS = ("default", "retain_95pct", "retain_99pct", "match_coatjava_retention")
SCHEDULE_FIELDS = ("local_batch_size", "max_lr", "min_lr", "warmup_steps",
                   "max_optimizer_steps", "val_interval_steps", "scheduler_mode",
                   "adapter_weight_decay", "grad_clip_value")
MODEL_FIELDS = ("embed_dim", "num_layers_backbone", "backbone_version", "adapter_version",
                "num_embedder_layers", "num_adapter_layers", "use_attention_head",
                "input_layout", "input_representation", "serialization", "clas12_band_config",
                "embed_method", "pe_method", "dropout", "downstream_dropout")


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def cohort_ids(records):
    ids = sorted((int(row['source_id']), int(row['original_ordinal'])) for row in records)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source-event identities in cohort")
    return ids


def load_run(run, manifest_dir):
    directory = (manifest_dir / run['results_dir']).resolve()
    summary = read_json(directory / 'summary.json')
    config = read_json(directory / 'resolved_config.json')
    cohorts = read_json(directory / 'cohorts.json')
    if summary.get('task') != 'noise' or summary.get('sample_unit') != 'event':
        raise ValueError(f"{run['run_id']}: expected a whole-event noise benchmark")
    if summary.get('test_split') != 'test':
        raise ValueError("Campaign performance must use the test split")
    identities = {split: cohort_ids(cohorts[split]) for split in ('train', 'validation', 'test')}
    for split, ids in identities.items():
        if len(ids) != summary[f'{split}_samples'] or not ids:
            raise ValueError(f"{run['run_id']}: {split} event count does not match cohort")
    sets = {split: set(ids) for split, ids in identities.items()}
    if sets['train'] & sets['validation'] or sets['train'] & sets['test'] or sets['validation'] & sets['test']:
        raise ValueError("Training, validation, and test cohorts overlap")

    schedule = {key: config.get(key) for key in SCHEDULE_FIELDS}
    steps = summary.get('optimizer_steps')
    # Re-evaluation summaries do not report executed updates; the saved recipe
    # supplies the original budget, not the selected checkpoint's step.
    if steps is None:
        steps = config.get('max_optimizer_steps')
    if not steps or summary.get('seed') is None or summary['seed'] != config.get('seed'):
        raise ValueError("Missing update budget or inconsistent training seed")
    comparison = dict(
        data_contract=summary['data_contract'], schedule=schedule, optimizer_steps=int(steps),
        validation_cohort_sha256=digest(identities['validation']),
        test_cohort_sha256=digest(identities['test']),
        validation_events=len(identities['validation']), test_events=len(identities['test']),
        checkpoint_selection=summary.get('validation_selection'),
    )
    balance = 'inverse_frequency' if summary.get('class_weights') is not None else 'none'
    model_id = run.get('model_id', summary.get('training_mode'))
    if model_id is None:
        if config.get('pretrained_ckpt'):
            raise ValueError("Pretrained evaluations require a model_id in the manifest")
        model_id = 'adapteronly'
    model_label = run.get('model_label', 'Adapter only' if model_id == 'adapteronly' else model_id)
    model = dict(model_id=model_id, class_balance=balance, adapter_parameters=summary['adapter_parameters'],
                 backbone_checkpoint_sha256=summary.get('backbone_checkpoint_sha256'),
                 options={key: config.get(key) for key in MODEL_FIELDS})
    common = dict(
        run_id=run['run_id'], results_dir=str(directory), model_id=model_id, model_label=model_label,
        series_id=digest(model), model=model, class_balance=balance,
        labeled_events=len(identities['train']), seed=int(summary['seed']),
        optimizer_steps=int(steps), selected_checkpoint_step=summary.get('selected_checkpoint_step'),
        checkpoint_sha256=summary.get('checkpoint_sha256'),
        train_cohort_sha256=digest(identities['train']),
        comparison_id=digest(comparison), comparison=comparison,
    )
    baseline = confusion_metrics(summary['coatjava_unassigned']['confusion_true_rows_pred_columns'])
    points = dict(default=summary['adapter'], **summary.get('operating_points', {}))
    rows = []
    for policy, point in points.items():
        if policy != 'default' and summary.get('threshold_source') != 'validation':
            raise ValueError("Calibrated operating points require validation-selected thresholds")
        metrics = confusion_metrics(point['confusion_true_rows_pred_columns'])
        if metrics['n_clusters'] != baseline['n_clusters']:
            raise ValueError("Adapter and COATJAVA cluster populations differ")
        rows.append(dict(common, operating_point=policy, threshold=point.get('threshold'),
                         metrics=metrics, coatjava=baseline))
    return rows


def collate(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = YAML(typ='safe').load(manifest_path.read_text())
    rows, skipped, seen = [], [], set()
    for run in manifest['runs']:
        run_id = run['run_id']
        if run_id in seen:
            raise ValueError(f"Duplicate run_id: {run_id}")
        seen.add(run_id)
        directory = (manifest_path.parent / run['results_dir']).resolve()
        if not (directory / 'summary.json').is_file():
            skipped.append(run_id)
            continue
        rows.extend(load_run(run, manifest_path.parent))
    if not rows:
        raise ValueError("No completed noise evaluations to collate")
    validate_rows(rows)
    return dict(schema_version=1, campaign_name=manifest.get('campaign_name', manifest_path.stem),
                metric_definitions=METRIC_DEFINITIONS, manifest=str(manifest_path),
                skipped_runs=skipped, rows=rows)


def validate_rows(rows):
    if not rows:
        raise ValueError("No completed noise evaluations to plot")
    baselines, points = {}, set()
    for row in rows:
        key = row['comparison_id']
        baseline = row['coatjava']['confusion_true_rows_pred_columns']
        if key in baselines and baselines[key] != baseline:
            raise ValueError("COATJAVA baseline differs on an identical evaluation cohort")
        baselines[key] = baseline
        point = (key, row['series_id'], row['operating_point'], row['labeled_events'], row['seed'])
        if point in points:
            raise ValueError("Duplicate model/budget/seed point; give distinct models distinct model_id values")
        points.add(point)


def write_tables(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run_table.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    flat = []
    for row in report['rows']:
        item = {key: value for key, value in row.items() if key not in ('metrics', 'coatjava', 'model', 'comparison')}
        item.update({key: row['metrics'][key] for key in METRICS})
        item.update({f'coatjava_{key}': row['coatjava'][key] for key in METRICS})
        item['test_clusters'] = row['metrics']['n_clusters']
        flat.append(item)
    with (output / 'run_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = collate(args.manifest)
    write_tables(report, args.output_dir)
    print(f"Wrote {len(report['rows'])} metric rows; skipped {len(report['skipped_runs'])} unfinished runs")


if __name__ == '__main__':
    main()
