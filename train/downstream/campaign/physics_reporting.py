"""Campaign collation of validation selection and evaluation diagnostics.

These sources stay separate: evaluation metrics never reselect checkpoints.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from physics_checkpoints import SCHEMA, METRICS, atomic_json


def read_json(path):
    return json.loads(Path(path).read_text())


def report_path(value, run):
    if not value:
        return None
    path = Path(value)
    # Absolute provenance may refer to the training machine rather than this mount.
    directory = Path(run.get('checkpoint_dir', Path(run['adapter_checkpoint']).parent))
    relocated = directory / path.parent.name / path.name
    for candidate in (relocated, path):
        if candidate.is_file():
            return candidate
    return None


def collect_run(run):
    checkpoint = Path(run['adapter_checkpoint'])
    pointer = checkpoint.with_name(checkpoint.stem + '_physics_report.json')
    artifact_path = Path(run.get('artifact_summary', Path(run['evaluation_dir']).parent / 'train' / 'artifacts.json'))
    artifact = read_json(artifact_path) if artifact_path.is_file() else {}
    current = read_json(pointer) if pointer.is_file() else {}
    ledger_value = current.get('checkpoint_summary') if current else artifact.get('physics_checkpoint_summary')
    ledger_path = report_path(ledger_value, run)
    ledger = read_json(ledger_path) if ledger_path else None
    if ledger and ledger.get('schema') != SCHEMA:
        raise ValueError(f'Unsupported physics checkpoint schema: {ledger_path}')
    support_value = current.get('validation_support') if current else artifact.get('validation_support')
    support_path = report_path(support_value, run)
    support = read_json(support_path) if support_path else None
    evaluation_path = Path(run['evaluation_dir']) / 'physics_checkpoint_summary.json'
    evaluation = read_json(evaluation_path) if evaluation_path.is_file() else None
    if evaluation and (evaluation.get('schema') != SCHEMA or evaluation.get('purpose') != 'evaluation_diagnostics_only'):
        raise ValueError(f'Unsupported evaluation physics summary: {evaluation_path}')
    summary_path = Path(run['evaluation_dir']) / 'summary.json'
    summary = read_json(summary_path) if summary_path.is_file() else {}
    task = current.get('task') or artifact.get('task') or summary.get('training_target_task')
    status = (ledger['selection_status'] if ledger and ledger.get('checkpoints') else
              current.get('selection_status') or artifact.get('checkpoint_selection_status') or 'unavailable')
    selected = [r for r in (ledger or {}).get('checkpoints', []) if r.get('selected')]
    if len(selected) > 1:
        raise ValueError(f'Multiple selected checkpoints in {ledger_path}')
    selected = selected[0] if selected else None
    table = dict(training_target_task=task, checkpoint_selection_status=status,
                 selected_step=selected.get('step') if selected else None,
                 selected_epoch=selected.get('epoch') if selected else None,
                 selected_validation_loss=selected.get('validation_loss') if selected else None,
                 physics_checkpoint_summary=str(ledger_path) if ledger_path else None,
                 validation_support=str(support_path) if support_path else None,
                 validation_n_samples=(support or {}).get('n_samples'),
                 validation_occupancy_passed=(support or {}).get('passed'),
                 evaluation_step=(evaluation or {}).get('checkpoint', {}).get('step'),
                 evaluation_epoch=(evaluation or {}).get('checkpoint', {}).get('epoch'))
    for key in (*METRICS, 'n_valid_bins', 'on_pareto_frontier', 'passes_guardrails', 'guardrails_configured'):
        table['validation_' + key] = selected.get(key) if selected else None
        table['evaluation_' + key] = (evaluation or {}).get('checkpoint', {}).get(key)
    if support and not selected:
        table['validation_n_valid_bins'] = support['n_valid_bins']
    if selected:
        table['validation_n_samples'] = selected['n_samples']
    # Scalar heads' existing inclusive metrics remain useful alongside macros.
    adapter = summary.get('methods', {}).get('adapter', {})
    for variable, metrics in adapter.get('kinematic', {}).items():
        for key in ('n', 'mae', 'rmse', 'median_absolute_error', 'p95_absolute_error', 'bias', 'r2'):
            if key in metrics:
                table[f'adapter_{variable}_{key}'] = metrics[key]
    meta = {key: run.get(key) for key in ('run_id', 'backbone_run_id', 'model_family', 'embed_dim', 'seed')}
    meta.update(task=task, labeled_events=run.get('labeled_events', run.get('eventnumber')),
                selection_status=status)
    rows = []
    for source, row, config in (
        ('validation_selected', selected, (ledger or {}).get('config')),
        ('evaluation', (evaluation or {}).get('checkpoint'), (evaluation or {}).get('config')),
    ):
        if row is None or config is None:
            continue
        contract = json.dumps(config, sort_keys=True)
        rows.append({**meta, 'source': source, 'residual': config['residual'],
            'residual_unit': config['residual_unit'], 'bin_quantity': config['bin_quantity'],
            'bin_unit': config['bin_unit'], 'bin_edges': json.dumps(config['bin_edges']),
            'config_id': hashlib.sha256(contract.encode()).hexdigest()[:12],
            'truth_support_hash': row.get('truth_support_hash'),
            'valid_bin_indices': json.dumps(row['valid_bin_indices']),
            **{key: row.get(key) for key in ('step','epoch','validation_loss',*METRICS,
                'n_samples','n_valid_bins','eligible','passes_guardrails','guardrails_configured',
                'on_pareto_frontier','selected')},
            **{'limit_' + key: config['guardrails'].get(key) for key in ('B_macro','B_worst','T_macro')}})
    return dict(metadata=meta, table=table, metric_rows=rows, validation=ledger,
                validation_support=support, evaluation=evaluation)


def write_csv(path, rows, empty_fields=()):
    fields = sorted({key for row in rows for key in row}) or list(empty_fields)
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def write_campaign_reports(directory, reports):
    directory = Path(directory)
    rows = [row for report in reports for row in report['metric_rows']]
    write_csv(directory / 'physics_checkpoint_metrics.csv', rows, ('run_id','source',*METRICS))
    atomic_json(directory / 'physics_campaign_summary.json', {'schema': SCHEMA, 'runs': reports})
    with (directory / 'physics_checkpoint_history.jsonl').open('w') as stream:
        for report in reports:
            for checkpoint in (report['validation'] or {}).get('checkpoints', []):
                stream.write(json.dumps(dict(metadata=report['metadata'], config=report['validation']['config'],
                                             checkpoint=checkpoint), allow_nan=False) + '\n')
    lines = ['# Campaign physics checkpoint summary', '',
             'Validation selection and evaluation diagnostics are reported separately. Evaluation does not select weights.', '',
             '| Run | Task | Selection status | Selected step | Val W | Val B | Val worst B | Val tails | Val bins | Eval W |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def number(value):
        return '—' if value is None else f'{value:.6g}'
    for report in reports:
        row = report['table']
        lines.append('| ' + ' | '.join([str(report['metadata']['run_id']), str(row['training_target_task']),
            row['checkpoint_selection_status'], number(row['selected_step']),
            *[number(row['validation_'+key]) for key in (*METRICS,'n_valid_bins')], number(row['evaluation_W_macro'])]) + ' |')
    lines += ['', 'Units and fixed bin definitions are in physics_checkpoint_metrics.csv and physics_campaign_summary.json.',
              'Null values mean unavailable or no checkpoint selected; they are never replaced by the best rejected candidate.']
    (directory / 'physics_checkpoint_summary.md').write_text('\n'.join(lines) + '\n')


def make_physics_campaign_plots(csv_path, output_dir):
    """Separate panels by task, metric policy, truth cohort, and valid bins."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    with Path(csv_path).open() as stream:
        rows = list(csv.DictReader(stream))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k, '') for k in ('source','task','config_id','truth_support_hash','valid_bin_indices'))
        groups[key].append(row)
    index = []
    for key, group in sorted(groups.items()):
        source, task, config_id, cohort, _ = key
        if not any(r.get('W_macro') for r in group):
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        series = defaultdict(list)
        for row in group:
            series[(row.get('backbone_run_id') or row['run_id'], row.get('seed', ''))].append(row)
        for ax, metric in zip(axes.flat, METRICS):
            for (backbone, seed), series_rows in sorted(series.items()):
                ordered = sorted(series_rows, key=lambda r: float(r.get('labeled_events') or 0))
                x = [float(r.get('labeled_events') or 0) for r in ordered]
                y = [float(r[metric]) if r.get(metric) else np.nan for r in ordered]
                ax.plot(x, y, 'o-', label=backbone + (f' seed {seed}' if seed else ''))
                for xx, yy, row in zip(x, y, ordered):
                    if row.get('eligible') == 'False':
                        ax.scatter(xx, yy, color='red', marker='x', zorder=5)
            limit = group[0].get('limit_' + metric)
            if limit:
                ax.axhline(float(limit), color='red', ls='--', lw=1)
            if all(float(r.get('labeled_events') or 0) > 0 for r in group):
                ax.set_xscale('log')
            ax.set(xlabel='Training labeled sample budget',
                   ylabel=f"{metric} [{('fraction' if metric == 'T_macro' else group[0]['residual_unit'])}]")
            if metric == 'T_macro':
                ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        statuses = ', '.join(sorted({r['selection_status'] for r in group}))
        fig.suptitle(f"{task}: {source}; {group[0]['residual']}\nSelection: {statuses}\nPolicy {config_id}, cohort {cohort[:8]}; red x = diagnostic failure", fontsize=10)
        identity = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:10]
        stem = f'{task}_{source}_{identity}'
        for suffix in ('png','pdf'):
            fig.savefig(output_dir / f'{stem}.{suffix}', dpi=150)
        plt.close(fig)
        index.append(dict(task=task, source=source, config_id=config_id, truth_support_hash=cohort,
                          valid_bin_indices=group[0]['valid_bin_indices'], plot=stem,
                          runs=[r['run_id'] for r in group],
                          selection_statuses={r['run_id']:r['selection_status'] for r in group}))
    atomic_json(output_dir / 'plot_index.json', index)
    return index
