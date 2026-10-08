from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


DEFAULT_TRACK_METRICS = [
    'val_loss',
    'val_perplexity',
    'val_utilization',
    'val_eeg_source_perplexity',
    'val_fnirs_source_perplexity',
    'val_eeg_observation_perplexity',
    'val_fnirs_observation_perplexity',
]


DEFAULT_VISUALIZATION_TITLES = {
    'best_val_loss_ranking': 'Best Val Loss Ranking',
    'gate1_health_overview': 'Gate1 Health Overview',
    'stability_overview': 'Training Stability Overview',
    'trajectory_patterns': 'TensorBoard-Style Trajectory Patterns',
    'branch_perplexity_trajectories': 'Branch Perplexity Trajectories',
}


PATTERN_LABELS = {
    'steady_improvement': 'Steady Improvement',
    'observation_starved': 'Observation-Starved',
    'delayed_observation_activation': 'Delayed Observation Activation',
    'persistent_balance_pressure': 'Persistent Balance Pressure',
    'late_instability': 'Late Instability',
}


PATTERN_DESCRIPTIONS = {
    'steady_improvement': 'Runs in this group keep improving while balance pressure decays and observation code usage stays engaged.',
    'observation_starved': 'Runs in this group never build enough observation-code diversity, so later gains mostly come from reconstruction tweaks rather than healthier token usage.',
    'delayed_observation_activation': 'Runs in this group only unlock observation code usage late in training, so potentially useful codebook behavior arrives after much of the optimization budget is already spent.',
    'persistent_balance_pressure': 'Runs in this group keep paying a high codebook-balance penalty late into training, which suggests the optimizer is still reallocating codes instead of converging cleanly.',
    'late_instability': 'Runs in this group find a good checkpoint but cannot hold it, with visible late-stage rebounds or oscillation in the tracked metrics.',
}


EFFECTIVENESS_LABELS = {
    'baseline_reference': 'Baseline Reference',
    'effective': 'Effective',
    'mostly_effective': 'Mostly Effective',
    'mixed_tradeoff': 'Mixed Trade-Off',
    'health_only_gain': 'Health-Only Gain',
    'limited_or_negative': 'Limited Or Negative',
    'unscored': 'Unscored',
}


def resolve_run_dirs(
    runs_root: str | Path,
    run_names: Sequence[str] | None = None,
    patterns: Sequence[str] | None = None,
) -> list[Path]:
    runs_root = Path(runs_root)
    resolved: list[Path] = []
    seen: set[Path] = set()

    def is_run_dir(path: Path) -> bool:
        return path.is_dir() and (path / 'metrics.json').exists()

    def is_archived(path: Path) -> bool:
        try:
            return 'archive' in path.relative_to(runs_root).parts
        except ValueError:
            return False

    def add_path(path: Path, *, skip_archive: bool = False) -> None:
        real_path = path.resolve()
        if real_path in seen or not is_run_dir(path):
            return
        if skip_archive and is_archived(path):
            return
        seen.add(real_path)
        resolved.append(real_path)

    for run_name in run_names or []:
        candidate = Path(run_name)
        if candidate.is_dir():
            add_path(candidate)
            continue
        candidate = runs_root / run_name
        if candidate.is_dir():
            add_path(candidate)

    for pattern in patterns or []:
        pattern_matches = list(runs_root.glob(pattern))
        if '/' not in pattern and '\\' not in pattern:
            pattern_matches.extend(runs_root.rglob(pattern))
        for candidate in sorted(set(pattern_matches)):
            add_path(candidate)

    if not resolved:
        for metrics_path in sorted(runs_root.rglob('metrics.json')):
            add_path(metrics_path.parent, skip_archive=True)

    return resolved


def build_run_summary(
    run_dir: str | Path,
    split: str = 'test',
    track_metrics: Sequence[str] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    metrics_payload = _load_json(run_dir / 'metrics.json')
    final_summary = _load_json_optional(run_dir / 'final_summary.json') or {}
    split_payload = _load_json_optional(run_dir / 'analysis' / f'split_{split}.json') or {}
    config = _load_yaml_optional(run_dir / 'config.yaml') or {}

    epochs = metrics_payload.get('epochs', [])
    final_metrics = metrics_payload.get('final_metrics', {})
    best_epoch = _safe_int(final_metrics.get('best_epoch'))
    last_epoch_record = epochs[-1] if epochs else {}
    best_epoch_record = _find_epoch_record(epochs, best_epoch) or last_epoch_record
    track_metrics = list(track_metrics or DEFAULT_TRACK_METRICS)

    summary: dict[str, Any] = {
        'run_name': run_dir.name,
        'run_dir': str(run_dir),
        'started_at': metrics_payload.get('started_at'),
        'completed_at': metrics_payload.get('completed_at'),
        'epoch_count': len(epochs),
        'best_epoch': best_epoch,
        'best_monitor': _safe_float(final_metrics.get('best_monitor')),
        'best_val_loss': _first_present(
            _safe_float(final_metrics.get('val_loss')),
            _safe_float(_extract_metric(best_epoch_record, 'val_loss')),
        ),
        'last_val_loss': _safe_float(_extract_metric(last_epoch_record, 'val_loss')),
        'promotion_verdict': final_summary.get('promotion_verdict'),
        'gate1_status': _nested_get(final_summary, 'gate_verdicts', 'gate1'),
        'gate2_status': _nested_get(final_summary, 'gate_verdicts', 'gate2'),
        'gate3_status': _nested_get(final_summary, 'gate_verdicts', 'gate3'),
        'gate4_status': _nested_get(final_summary, 'gate_verdicts', 'gate4'),
        'coupling_lag_policy': final_summary.get('coupling_lag_policy'),
        'experiment_name': _nested_get(config, 'experiment', 'name'),
        'experiment_description': _nested_get(config, 'experiment', 'description'),
        'learning_rate': _safe_float(_nested_get(config, 'training', 'learning_rate')),
        'source_codebook_size': _safe_int(_nested_get(config, 'model', 'source', 'codebook_size')),
        'eeg_observation_codebook_size': _safe_int(_nested_get(config, 'model', 'eeg_observation', 'codebook_size')),
        'fnirs_observation_codebook_size': _safe_int(_nested_get(config, 'model', 'fnirs_observation', 'codebook_size')),
        'quantizer_beta': _safe_float(_nested_get(config, 'model', 'quantizer', 'beta')),
        'quantizer_decay': _safe_float(_nested_get(config, 'model', 'quantizer', 'decay')),
        'codebook_balance_weight': _safe_float(_nested_get(config, 'loss', 'codebook', 'balance_weight')),
        'coupling_weight': _safe_float(_nested_get(config, 'loss', 'coupling', 'weight')),
    }
    summary['observation_codebook_sizes'] = _format_observation_sizes(summary)
    if summary['best_val_loss'] is not None and summary['last_val_loss'] is not None:
        summary['best_to_last_val_loss_gap'] = summary['last_val_loss'] - summary['best_val_loss']
    else:
        summary['best_to_last_val_loss_gap'] = None

    for metric_name in track_metrics:
        summary[f'best_{metric_name}'] = _first_present(
            _safe_float(final_metrics.get(metric_name)),
            _safe_float(_extract_metric(best_epoch_record, metric_name)),
        )
        summary[f'last_{metric_name}'] = _safe_float(_extract_metric(last_epoch_record, metric_name))

    summary.update(_summarize_val_utilization(epochs))
    summary.update(_summarize_training_dynamics(epochs, summary))

    gates = split_payload.get('gates', split_payload)
    gate1 = gates.get('gate1', {}) if isinstance(gates, dict) else {}
    summary.update(_summarize_gate1(gate1))
    summary['trajectory_pattern'] = _classify_trajectory_pattern(summary)
    return summary


def collect_run_summaries(
    run_dirs: Iterable[str | Path],
    split: str = 'test',
    track_metrics: Sequence[str] | None = None,
    baseline: str | None = None,
) -> list[dict[str, Any]]:
    rows = [build_run_summary(run_dir, split=split, track_metrics=track_metrics) for run_dir in run_dirs]
    baseline_row = None
    if baseline:
        baseline_row = _attach_baseline_deltas(rows, baseline)
    _attach_run_labels(rows, baseline_row)
    return rows


def sort_run_summaries(
    rows: Sequence[dict[str, Any]],
    sort_by: str = 'best_val_loss',
    descending: bool = False,
) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[int, Any]:
        value = row.get(sort_by)
        return (value is None, value)

    return sorted(rows, key=sort_key, reverse=descending)


def render_markdown_table(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
) -> str:
    headers = list(columns)
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
    ]
    for row in rows:
        values = [_format_table_value(row.get(column)) for column in headers]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def prepare_report_directory(
    report_root: str | Path,
    run_dirs: Sequence[str | Path],
    report_name: str | None = None,
) -> Path:
    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    if report_name:
        base_name = _slugify(report_name)
    else:
        run_names = [Path(run_dir).name for run_dir in run_dirs]
        if len(run_names) == 1:
            base_name = _slugify(run_names[0])
        elif len(run_names) <= 3:
            base_name = _slugify('_vs_'.join(run_names))
        else:
            base_name = _slugify(f'{len(run_names)}_runs_comparison')

    if not base_name:
        base_name = 'run_comparison'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f'{timestamp}_{base_name}'
    report_dir = report_root / stem
    counter = 1
    while report_dir.exists():
        report_dir = report_root / f'{stem}_{counter:02d}'
        counter += 1
    report_dir.mkdir(parents=True, exist_ok=False)
    return report_dir


def write_report_bundle(
    report_dir: str | Path,
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
    metadata: dict[str, Any] | None = None,
    include_visualizations: bool = True,
) -> dict[str, Any]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / 'summary.csv'
    json_path = report_dir / 'summary.json'
    analysis_path = report_dir / 'analysis.json'
    markdown_path = report_dir / 'report.md'
    metadata_path = report_dir / 'metadata.json'

    write_csv_report(csv_path, rows)
    write_json_report(json_path, rows)

    analysis_payload = build_comparison_analysis(rows, baseline=_safe_str((metadata or {}).get('baseline')))
    analysis_path.write_text(json.dumps(analysis_payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    figure_paths: dict[str, Path] = {}
    if include_visualizations and HAS_MATPLOTLIB:
        figure_paths = generate_visualizations(rows, report_dir / 'figures')

    metadata_payload = dict(metadata or {})
    metadata_payload.update(
        {
            'generated_at': datetime.now().isoformat(),
            'report_dir': str(report_dir),
            'run_count': len(rows),
            'columns': list(columns),
            'artifacts': {
                'summary_csv': str(csv_path),
                'summary_json': str(json_path),
                'analysis_json': str(analysis_path),
                'report_markdown': str(markdown_path),
                'figures': {name: str(path) for name, path in figure_paths.items()},
            },
        }
    )
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    markdown_path.write_text(
        build_markdown_report(rows, columns, metadata_payload, figure_paths, analysis_payload),
        encoding='utf-8',
    )

    return {
        'report_dir': report_dir,
        'summary_csv': csv_path,
        'summary_json': json_path,
        'analysis_json': analysis_path,
        'report_markdown': markdown_path,
        'metadata_json': metadata_path,
        'figures': figure_paths,
    }


def generate_visualizations(
    rows: Sequence[dict[str, Any]],
    figures_dir: str | Path,
) -> dict[str, Path]:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: dict[str, Path] = {}
    best_loss_path = figures_dir / 'best_val_loss_ranking.png'
    gate1_path = figures_dir / 'gate1_health_overview.png'
    stability_path = figures_dir / 'stability_overview.png'
    trajectory_path = figures_dir / 'trajectory_patterns.png'
    branch_path = figures_dir / 'branch_perplexity_trajectories.png'

    history_cache = _build_history_cache(
        rows,
        [
            'val_loss',
            'codebook_balance_loss',
            'eeg_amp_loss',
            'eeg_observation_perplexity',
            'eeg_source_perplexity',
            'fnirs_source_perplexity',
            'fnirs_observation_perplexity',
        ],
    )

    _save_figure(_plot_best_val_loss_ranking(rows), best_loss_path)
    figure_paths['best_val_loss_ranking'] = best_loss_path

    _save_figure(_plot_gate1_health_overview(rows), gate1_path)
    figure_paths['gate1_health_overview'] = gate1_path

    _save_figure(_plot_stability_overview(rows), stability_path)
    figure_paths['stability_overview'] = stability_path

    _save_figure(_plot_trajectory_patterns(rows, history_cache), trajectory_path)
    figure_paths['trajectory_patterns'] = trajectory_path

    _save_figure(_plot_branch_perplexity_trajectories(rows, history_cache), branch_path)
    figure_paths['branch_perplexity_trajectories'] = branch_path
    return figure_paths


def build_markdown_report(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
    metadata: dict[str, Any],
    figure_paths: dict[str, Path] | None = None,
    analysis: dict[str, Any] | None = None,
) -> str:
    figure_paths = figure_paths or {}
    lines = [
        '# Run Comparison Report',
        '',
        f"- Generated at: {metadata.get('generated_at', '')}",
        f"- Compared runs: {len(rows)}",
        f"- Split: {metadata.get('split', '')}",
        f"- Baseline: {metadata.get('baseline', '')}",
        f"- Sort by: {metadata.get('sort_by', '')}",
        '',
        '## Summary Table',
        '',
        render_markdown_table(rows, columns),
    ]

    if analysis:
        lines.extend(_render_analysis_section(analysis))

    if figure_paths:
        report_dir = Path(metadata.get('report_dir', '.'))
        lines.extend(['', '## Visualizations', ''])
        for figure_key, figure_path in figure_paths.items():
            title = DEFAULT_VISUALIZATION_TITLES.get(figure_key, figure_key.replace('_', ' ').title())
            relative_path = Path(figure_path).relative_to(report_dir)
            lines.extend([
                f'### {title}',
                '',
                f'![{title}]({relative_path.as_posix()})',
                '',
            ])

    return '\n'.join(lines).rstrip() + '\n'


def write_csv_report(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _collect_fieldnames(rows)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json_report(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_markdown_report(
    path: str | Path,
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_table(rows, columns) + '\n', encoding='utf-8')


def _plot_best_val_loss_ranking(rows: Sequence[dict[str, Any]]):
    labels = [str(row.get('run_name', 'unknown')) for row in rows]
    values = [_safe_float(row.get('best_val_loss')) or 0.0 for row in rows]
    baseline_run = next((row.get('baseline_run') for row in rows if row.get('baseline_run')), None)
    colors = ['#4C956C' if label == baseline_run else '#2A6F97' for label in labels]

    figure_height = max(4.5, 0.7 * len(rows) + 1.5)
    figure, axis = plt.subplots(figsize=(11, figure_height))
    y_positions = list(range(len(labels)))
    axis.barh(y_positions, values, color=colors)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlabel('Best val_loss')
    axis.set_title('Best Validation Loss by Run')
    axis.grid(axis='x', linestyle='--', alpha=0.25)
    for y_pos, value in zip(y_positions, values):
        axis.text(value, y_pos, f' {value:.4f}', va='center', ha='left', fontsize=9)
    figure.tight_layout()
    return figure


def _plot_gate1_health_overview(rows: Sequence[dict[str, Any]]):
    labels = [str(row.get('run_name', 'unknown')) for row in rows]
    min_active = [_safe_float(row.get('gate1_min_active_ratio')) or 0.0 for row in rows]
    min_perplexity = [_safe_float(row.get('gate1_min_perplexity_ratio')) or 0.0 for row in rows]
    max_top5 = [_safe_float(row.get('gate1_max_top_5_coverage')) or 0.0 for row in rows]

    figure_height = max(4.5, 0.7 * len(rows) + 1.5)
    figure, axes = plt.subplots(1, 3, figsize=(16, figure_height), sharey=True)
    metrics = [
        ('Min active ratio', min_active, 0.5, '#4C956C'),
        ('Min perplexity ratio', min_perplexity, 0.3, '#3D5A80'),
        ('Max top-5 coverage', max_top5, 0.5, '#BC4749'),
    ]
    y_positions = list(range(len(labels)))

    for axis, (title, values, threshold, color) in zip(axes, metrics):
        axis.barh(y_positions, values, color=color)
        axis.axvline(threshold, color='#222222', linestyle='--', linewidth=1)
        axis.set_title(title)
        axis.grid(axis='x', linestyle='--', alpha=0.25)
        for y_pos, value in zip(y_positions, values):
            axis.text(value, y_pos, f' {value:.3f}', va='center', ha='left', fontsize=8)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    figure.suptitle('Gate1 Health Comparison', fontsize=14)
    figure.tight_layout()
    return figure


def _plot_stability_overview(rows: Sequence[dict[str, Any]]):
    labels = [str(row.get('run_name', 'unknown')) for row in rows]
    low_epoch_counts = [_safe_float(row.get('val_utilization_low_epochs')) or 0.0 for row in rows]
    switch_counts = [_safe_float(row.get('val_utilization_switches')) or 0.0 for row in rows]
    best_last_gaps = [_safe_float(row.get('best_to_last_val_loss_gap')) or 0.0 for row in rows]

    x_positions = list(range(len(labels)))
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    metrics = [
        ('Low-utilization epochs', low_epoch_counts, '#8E7DBE'),
        ('Utilization regime switches', switch_counts, '#F4A259'),
        ('Last-best val_loss gap', best_last_gaps, '#2A9D8F'),
    ]

    for axis, (title, values, color) in zip(axes, metrics):
        axis.bar(x_positions, values, color=color)
        axis.set_title(title)
        axis.grid(axis='y', linestyle='--', alpha=0.25)
        for x_pos, value in zip(x_positions, values):
            axis.text(x_pos, value, f'{value:.2f}', va='bottom', ha='center', fontsize=8)

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(labels, rotation=30, ha='right')
    figure.suptitle('Training Stability Summary', fontsize=14)
    figure.tight_layout()
    return figure


def _plot_trajectory_patterns(
    rows: Sequence[dict[str, Any]],
    history_cache: dict[str, dict[str, Any]],
):
    metrics = [
        ('val_loss', 'Val Loss'),
        ('codebook_balance_loss', 'Train Codebook Balance Loss'),
        ('eeg_amp_loss', 'Train EEG Amplitude Loss'),
        ('eeg_observation_perplexity', 'Train EEG Observation Perplexity'),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
    figure.suptitle('TensorBoard-Style Trajectory Patterns', fontsize=15)
    styles = _build_run_plot_styles(rows)

    for axis, (metric_name, title) in zip(axes.flat, metrics):
        _plot_metric_overlay(axis, rows, history_cache, metric_name, styles)
        axis.set_title(title)
        axis.set_xlabel('Epoch')
        axis.grid(alpha=0.25, linestyle='--')

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc='upper center', ncol=min(3, len(labels)), frameon=False)
        figure.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        figure.tight_layout()
    return figure


def _plot_branch_perplexity_trajectories(
    rows: Sequence[dict[str, Any]],
    history_cache: dict[str, dict[str, Any]],
):
    metrics = [
        ('eeg_source_perplexity', 'Train EEG Source Perplexity'),
        ('fnirs_source_perplexity', 'Train fNIRS Source Perplexity'),
        ('eeg_observation_perplexity', 'Train EEG Observation Perplexity'),
        ('fnirs_observation_perplexity', 'Train fNIRS Observation Perplexity'),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
    figure.suptitle('Branch Perplexity Trajectories', fontsize=15)
    styles = _build_run_plot_styles(rows)

    for axis, (metric_name, title) in zip(axes.flat, metrics):
        _plot_metric_overlay(axis, rows, history_cache, metric_name, styles)
        axis.set_title(title)
        axis.set_xlabel('Epoch')
        axis.grid(alpha=0.25, linestyle='--')

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc='upper center', ncol=min(3, len(labels)), frameon=False)
        figure.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        figure.tight_layout()
    return figure


def _save_figure(figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(figure)


def _slugify(value: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9]+', '_', value).strip('_').lower()
    return slug[:80]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _load_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _load_yaml_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def build_comparison_analysis(
    rows: Sequence[dict[str, Any]],
    baseline: str | None = None,
) -> dict[str, Any]:
    baseline_row = _find_baseline_row(rows, baseline) if baseline else None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_safe_str(row.get('trajectory_pattern')) or 'steady_improvement', []).append(row)

    pattern_groups = []
    for pattern_key, group_rows in sorted(grouped.items(), key=lambda item: (len(item[1]) * -1, item[0])):
        pattern_groups.append(
            {
                'pattern_key': pattern_key,
                'pattern_label': PATTERN_LABELS.get(pattern_key, pattern_key.replace('_', ' ').title()),
                'runs': [str(group_row.get('run_name')) for group_row in group_rows],
                'summary': _summarize_pattern_group(pattern_key, group_rows),
            }
        )

    run_findings = [_build_run_finding(row, baseline_row) for row in rows]
    return {
        'baseline_run': baseline_row.get('run_name') if baseline_row else None,
        'pattern_groups': pattern_groups,
        'run_findings': run_findings,
    }


def _find_epoch_record(epochs: Sequence[dict[str, Any]], best_epoch: int | None) -> dict[str, Any] | None:
    if best_epoch is None:
        return None
    for epoch_record in epochs:
        if _safe_int(epoch_record.get('epoch')) == best_epoch:
            return epoch_record
    return None


def _extract_metric(epoch_record: dict[str, Any], metric_name: str) -> Any:
    if not epoch_record:
        return None
    if metric_name in epoch_record:
        return epoch_record.get(metric_name)
    metrics = epoch_record.get('metrics', {})
    if metric_name in metrics:
        return metrics.get(metric_name)
    loss_breakdown = epoch_record.get('loss_breakdown', {})
    if metric_name in loss_breakdown:
        return loss_breakdown.get(metric_name)
    return None


def _extract_metric_values(epochs: Sequence[dict[str, Any]], metric_name: str) -> list[float]:
    values: list[float] = []
    for epoch_record in epochs:
        value = _safe_float(_extract_metric(epoch_record, metric_name))
        if value is not None:
            values.append(value)
    return values


def _extract_metric_series(epochs: Sequence[dict[str, Any]], metric_name: str) -> tuple[list[int], list[float]]:
    epoch_numbers: list[int] = []
    values: list[float] = []
    for index, epoch_record in enumerate(epochs, start=1):
        value = _safe_float(_extract_metric(epoch_record, metric_name))
        if value is None:
            continue
        epoch_numbers.append(_safe_int(epoch_record.get('epoch')) or index)
        values.append(value)
    return epoch_numbers, values


def _summarize_training_dynamics(
    epochs: Sequence[dict[str, Any]],
    row: dict[str, Any],
) -> dict[str, Any]:
    val_loss_stats = _summarize_series(_extract_metric_values(epochs, 'val_loss'))
    balance_stats = _summarize_series(_extract_metric_values(epochs, 'codebook_balance_loss'))
    eeg_amp_stats = _summarize_series(_extract_metric_values(epochs, 'eeg_amp_loss'))
    eeg_obs_stats = _summarize_series(_extract_metric_values(epochs, 'eeg_observation_perplexity'))
    fnirs_obs_stats = _summarize_series(_extract_metric_values(epochs, 'fnirs_observation_perplexity'))
    source_overlap_stats = _summarize_series(_extract_metric_values(epochs, 'source_code_overlap'))

    eeg_obs_codebook_size = _safe_float(row.get('eeg_observation_codebook_size'))
    fnirs_obs_codebook_size = _safe_float(row.get('fnirs_observation_codebook_size'))
    eeg_activation_epoch = _first_epoch_at_or_above(
        _extract_metric_values(epochs, 'eeg_observation_perplexity'),
        0.2 * eeg_obs_codebook_size if eeg_obs_codebook_size else None,
    )
    fnirs_activation_epoch = _first_epoch_at_or_above(
        _extract_metric_values(epochs, 'fnirs_observation_perplexity'),
        0.2 * fnirs_obs_codebook_size if fnirs_obs_codebook_size else None,
    )
    observation_activation_epoch = _max_present(eeg_activation_epoch, fnirs_activation_epoch)
    epoch_count = max(_safe_int(row.get('epoch_count')) or 0, 1)

    eeg_obs_tail_ratio = _ratio(eeg_obs_stats['tail_mean'], eeg_obs_codebook_size)
    fnirs_obs_tail_ratio = _ratio(fnirs_obs_stats['tail_mean'], fnirs_obs_codebook_size)
    eeg_obs_head_ratio = _ratio(eeg_obs_stats['head_mean'], eeg_obs_codebook_size)
    fnirs_obs_head_ratio = _ratio(fnirs_obs_stats['head_mean'], fnirs_obs_codebook_size)

    return {
        'val_loss_reduction': _subtract(val_loss_stats['head_mean'], val_loss_stats['tail_mean']),
        'val_loss_volatility': val_loss_stats['volatility'],
        'train_balance_tail_mean': balance_stats['tail_mean'],
        'train_balance_volatility': balance_stats['volatility'],
        'train_eeg_amp_tail_mean': eeg_amp_stats['tail_mean'],
        'train_eeg_amp_reduction': _subtract(eeg_amp_stats['head_mean'], eeg_amp_stats['tail_mean']),
        'train_eeg_observation_head_ratio': eeg_obs_head_ratio,
        'train_eeg_observation_tail_ratio': eeg_obs_tail_ratio,
        'train_fnirs_observation_head_ratio': fnirs_obs_head_ratio,
        'train_fnirs_observation_tail_ratio': fnirs_obs_tail_ratio,
        'train_observation_head_ratio': _average_present([eeg_obs_head_ratio, fnirs_obs_head_ratio]),
        'train_observation_tail_ratio': _average_present([eeg_obs_tail_ratio, fnirs_obs_tail_ratio]),
        'observation_activation_epoch': observation_activation_epoch,
        'observation_activation_ratio': _ratio(observation_activation_epoch, epoch_count),
        'train_source_overlap_tail_mean': source_overlap_stats['tail_mean'],
        'train_source_overlap_volatility': source_overlap_stats['volatility'],
    }


def _summarize_val_utilization(epochs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = [
        _safe_float(_extract_metric(epoch_record, 'val_utilization'))
        for epoch_record in epochs
    ]
    values = [value for value in values if value is not None]
    if not values:
        return {
            'val_utilization_min': None,
            'val_utilization_max': None,
            'val_utilization_low_epochs': None,
            'val_utilization_switches': None,
        }
    switches = 0
    for previous, current in zip(values, values[1:]):
        if (previous < 0.5) != (current < 0.5):
            switches += 1
    return {
        'val_utilization_min': min(values),
        'val_utilization_max': max(values),
        'val_utilization_low_epochs': sum(value < 0.5 for value in values),
        'val_utilization_switches': switches,
    }


def _summarize_gate1(gate1: dict[str, Any]) -> dict[str, Any]:
    codebooks = _nested_get(gate1, 'metrics', 'codebooks') or {}
    if not isinstance(codebooks, dict) or not codebooks:
        return {
            'gate1_failed_codebooks': None,
            'gate1_min_active_ratio': None,
            'gate1_min_active_codebook': None,
            'gate1_min_perplexity_ratio': None,
            'gate1_min_perplexity_codebook': None,
            'gate1_max_top_5_coverage': None,
            'gate1_max_top_5_codebook': None,
            'gate1_max_dead_code_ratio': None,
            'gate1_max_dead_codebook': None,
            'gate1_primary_bottleneck': None,
        }

    failed_codebooks: list[str] = []
    min_active_ratio: tuple[float, str] | None = None
    min_perplexity_ratio: tuple[float, str] | None = None
    max_top_5: tuple[float, str] | None = None
    max_dead_ratio: tuple[float, str] | None = None
    worst_margin: tuple[float, str] | None = None

    for codebook_name, metrics in codebooks.items():
        codebook_size = max(_safe_int(metrics.get('codebook_size')) or 1, 1)
        active_ratio = _safe_float(metrics.get('active_code_ratio')) or 0.0
        perplexity = _safe_float(metrics.get('perplexity')) or 0.0
        top_5_coverage = _safe_float(metrics.get('top_5_coverage')) or 0.0
        dead_code_count = _safe_float(metrics.get('dead_code_count')) or 0.0
        dead_ratio = dead_code_count / codebook_size
        perplexity_ratio = perplexity / codebook_size

        if not bool(metrics.get('passes_thresholds')):
            failed_codebooks.append(codebook_name)

        min_active_ratio = _pick_min_pair(min_active_ratio, (active_ratio, codebook_name))
        min_perplexity_ratio = _pick_min_pair(min_perplexity_ratio, (perplexity_ratio, codebook_name))
        max_top_5 = _pick_max_pair(max_top_5, (top_5_coverage, codebook_name))
        max_dead_ratio = _pick_max_pair(max_dead_ratio, (dead_ratio, codebook_name))

        margin_candidates = [
            (active_ratio - 0.5, f'{codebook_name}.active_code_ratio={active_ratio:.4f}'),
            (perplexity_ratio - 0.3, f'{codebook_name}.perplexity_ratio={perplexity_ratio:.4f}'),
            (0.5 - top_5_coverage, f'{codebook_name}.top_5_coverage={top_5_coverage:.4f}'),
            (0.3 - dead_ratio, f'{codebook_name}.dead_code_ratio={dead_ratio:.4f}'),
        ]
        for candidate in margin_candidates:
            if worst_margin is None or candidate[0] < worst_margin[0]:
                worst_margin = candidate

    return {
        'gate1_failed_codebooks': ','.join(failed_codebooks),
        'gate1_min_active_ratio': min_active_ratio[0] if min_active_ratio else None,
        'gate1_min_active_codebook': min_active_ratio[1] if min_active_ratio else None,
        'gate1_min_perplexity_ratio': min_perplexity_ratio[0] if min_perplexity_ratio else None,
        'gate1_min_perplexity_codebook': min_perplexity_ratio[1] if min_perplexity_ratio else None,
        'gate1_max_top_5_coverage': max_top_5[0] if max_top_5 else None,
        'gate1_max_top_5_codebook': max_top_5[1] if max_top_5 else None,
        'gate1_max_dead_code_ratio': max_dead_ratio[0] if max_dead_ratio else None,
        'gate1_max_dead_codebook': max_dead_ratio[1] if max_dead_ratio else None,
        'gate1_primary_bottleneck': worst_margin[1] if worst_margin else None,
    }


def _attach_baseline_deltas(rows: Sequence[dict[str, Any]], baseline: str) -> dict[str, Any]:
    baseline_row = None
    for row in rows:
        if row.get('run_name') == baseline or row.get('experiment_name') == baseline or row.get('run_dir') == baseline:
            baseline_row = row
            break
    if baseline_row is None:
        available = ', '.join(sorted(str(row.get('run_name')) for row in rows))
        raise ValueError(f'Baseline run not found: {baseline}. Available runs: {available}')

    delta_fields = [
        'best_val_loss',
        'gate1_min_active_ratio',
        'gate1_min_perplexity_ratio',
        'gate1_max_top_5_coverage',
        'gate1_max_dead_code_ratio',
        'val_utilization_low_epochs',
        'val_utilization_switches',
    ]
    for row in rows:
        row['baseline_run'] = baseline_row.get('run_name')
        for field in delta_fields:
            row[f'delta_{field}'] = _delta(row.get(field), baseline_row.get(field))
    return baseline_row


def _attach_run_labels(
    rows: Sequence[dict[str, Any]],
    baseline_row: dict[str, Any] | None,
) -> None:
    for row in rows:
        row['trajectory_pattern'] = row.get('trajectory_pattern') or _classify_trajectory_pattern(row)
        row['improvement_effectiveness'] = _assess_effectiveness(row, baseline_row)


def _assess_effectiveness(
    row: dict[str, Any],
    baseline_row: dict[str, Any] | None,
) -> str:
    if baseline_row is None:
        return 'unscored'
    if row.get('run_name') == baseline_row.get('run_name'):
        return 'baseline_reference'

    loss_gain = -(_safe_float(row.get('delta_best_val_loss')) or 0.0)
    health_terms = [
        _normalize_change(row.get('delta_gate1_min_active_ratio'), positive_is_better=True),
        _normalize_change(row.get('delta_gate1_min_perplexity_ratio'), positive_is_better=True),
        _normalize_change(row.get('delta_gate1_max_top_5_coverage'), positive_is_better=False),
        _normalize_change(row.get('delta_gate1_max_dead_code_ratio'), positive_is_better=False),
    ]
    health_terms = [term for term in health_terms if term is not None]
    health_gain = _average_present(health_terms) or 0.0

    if loss_gain >= 0.05 and health_gain >= 0.25:
        return 'effective'
    if loss_gain >= 0.05 and health_gain >= -0.10:
        return 'mostly_effective'
    if loss_gain >= 0.05:
        return 'mixed_tradeoff'
    if health_gain >= 0.25:
        return 'health_only_gain'
    return 'limited_or_negative'


def _classify_trajectory_pattern(row: dict[str, Any]) -> str:
    observation_tail_ratio = _safe_float(row.get('train_observation_tail_ratio'))
    observation_activation_ratio = _safe_float(row.get('observation_activation_ratio'))
    balance_tail_mean = _safe_float(row.get('train_balance_tail_mean'))
    balance_volatility = _safe_float(row.get('train_balance_volatility'))
    late_gap = _safe_float(row.get('best_to_last_val_loss_gap'))
    val_loss_volatility = _safe_float(row.get('val_loss_volatility'))

    if observation_tail_ratio is not None and observation_tail_ratio < 0.15:
        return 'observation_starved'
    if observation_activation_ratio is not None and observation_activation_ratio > 0.60:
        return 'delayed_observation_activation'
    if (balance_tail_mean is not None and balance_tail_mean > 0.26) or (balance_volatility is not None and balance_volatility > 0.04):
        return 'persistent_balance_pressure'
    if (late_gap is not None and late_gap > 0.05) and (val_loss_volatility is not None and val_loss_volatility > 0.025):
        return 'late_instability'
    return 'steady_improvement'


def _pick_min_pair(current: tuple[float, str] | None, candidate: tuple[float, str]) -> tuple[float, str]:
    if current is None or candidate[0] < current[0]:
        return candidate
    return current


def _pick_max_pair(current: tuple[float, str] | None, candidate: tuple[float, str]) -> tuple[float, str]:
    if current is None or candidate[0] > current[0]:
        return candidate
    return current


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _delta(current: Any, baseline: Any) -> float | None:
    current_value = _safe_float(current)
    baseline_value = _safe_float(baseline)
    if current_value is None or baseline_value is None:
        return None
    return current_value - baseline_value


def _safe_float(value: Any) -> float | None:
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None or value == '':
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _format_observation_sizes(row: dict[str, Any]) -> str | None:
    eeg_size = row.get('eeg_observation_codebook_size')
    fnirs_size = row.get('fnirs_observation_codebook_size')
    if eeg_size is None and fnirs_size is None:
        return None
    return f'{eeg_size}/{fnirs_size}'


def _collect_fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    return fieldnames


def _format_table_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def _summarize_series(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {
            'start': None,
            'end': None,
            'head_mean': None,
            'tail_mean': None,
            'min': None,
            'max': None,
            'volatility': None,
        }

    window = max(1, len(values) // 5)
    head_values = list(values[:window])
    tail_values = list(values[-window:])
    diffs = [abs(current - previous) for previous, current in zip(values, values[1:])]
    return {
        'start': values[0],
        'end': values[-1],
        'head_mean': _average_present(head_values),
        'tail_mean': _average_present(tail_values),
        'min': min(values),
        'max': max(values),
        'volatility': _average_present(diffs),
    }


def _average_present(values: Sequence[float | None]) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _ratio(value: float | int | None, denominator: float | int | None) -> float | None:
    if value is None or denominator in (None, 0):
        return None
    return float(value) / float(denominator)


def _first_epoch_at_or_above(values: Sequence[float], threshold: float | None) -> int | None:
    if threshold is None:
        return None
    for index, value in enumerate(values, start=1):
        if value >= threshold:
            return index
    return None


def _max_present(*values: int | None) -> int | None:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return max(numeric_values)


def _normalize_change(value: Any, positive_is_better: bool) -> float | None:
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return None
    if not positive_is_better:
        numeric_value = -numeric_value
    return numeric_value / 0.05


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _find_baseline_row(
    rows: Sequence[dict[str, Any]],
    baseline: str | None,
) -> dict[str, Any] | None:
    if baseline is None:
        return None
    for row in rows:
        if row.get('run_name') == baseline or row.get('experiment_name') == baseline or row.get('run_dir') == baseline:
            return row
    return None


def _summarize_pattern_group(
    pattern_key: str,
    rows: Sequence[dict[str, Any]],
) -> str:
    observation_tail_ratio = _average_present([_safe_float(row.get('train_observation_tail_ratio')) for row in rows])
    balance_tail_mean = _average_present([_safe_float(row.get('train_balance_tail_mean')) for row in rows])
    description = PATTERN_DESCRIPTIONS.get(pattern_key, '')
    metrics_summary = []
    if observation_tail_ratio is not None:
        metrics_summary.append(f'avg observation tail ratio {observation_tail_ratio:.3f}')
    if balance_tail_mean is not None:
        metrics_summary.append(f'avg late balance loss {balance_tail_mean:.3f}')
    if metrics_summary:
        return f"{description} Key signal: {', '.join(metrics_summary)}."
    return description


def _build_run_finding(
    row: dict[str, Any],
    baseline_row: dict[str, Any] | None,
) -> dict[str, Any]:
    pattern_key = _safe_str(row.get('trajectory_pattern')) or 'steady_improvement'
    effectiveness_key = _safe_str(row.get('improvement_effectiveness')) or 'unscored'
    return {
        'run_name': row.get('run_name'),
        'pattern_key': pattern_key,
        'pattern_label': PATTERN_LABELS.get(pattern_key, pattern_key.replace('_', ' ').title()),
        'effectiveness_key': effectiveness_key,
        'effectiveness_label': EFFECTIVENESS_LABELS.get(effectiveness_key, effectiveness_key.replace('_', ' ').title()),
        'config_changes': _collect_config_changes(row, baseline_row),
        'hypothesis': _infer_change_hypothesis(row, baseline_row),
        'evidence': _collect_evidence_points(row, baseline_row),
    }


def _collect_config_changes(
    row: dict[str, Any],
    baseline_row: dict[str, Any] | None,
) -> list[str]:
    if baseline_row is None or row.get('run_name') == baseline_row.get('run_name'):
        return []

    config_fields = [
        ('learning_rate', 'learning rate'),
        ('source_codebook_size', 'source codebook size'),
        ('codebook_balance_weight', 'codebook balance weight'),
        ('coupling_weight', 'coupling weight'),
        ('quantizer_beta', 'quantizer beta'),
        ('quantizer_decay', 'quantizer decay'),
    ]
    changes: list[str] = []
    for field, label in config_fields:
        current = row.get(field)
        baseline_value = baseline_row.get(field)
        if current == baseline_value:
            continue
        if _safe_float(current) is not None and _safe_float(baseline_value) is not None:
            changes.append(f'{label}: {_safe_float(baseline_value):.4f} -> {_safe_float(current):.4f}')
        else:
            changes.append(f'{label}: {baseline_value} -> {current}')
    return changes


def _infer_change_hypothesis(
    row: dict[str, Any],
    baseline_row: dict[str, Any] | None,
) -> str:
    reasons: list[str] = []
    pattern_key = _safe_str(row.get('trajectory_pattern')) or 'steady_improvement'
    pattern_reason = _pattern_primary_cause(pattern_key, row)
    if pattern_reason:
        reasons.append(pattern_reason)

    if baseline_row is not None and row.get('run_name') != baseline_row.get('run_name'):
        current_lr = _safe_float(row.get('learning_rate'))
        baseline_lr = _safe_float(baseline_row.get('learning_rate'))
        current_volatility = _safe_float(row.get('val_loss_volatility'))
        baseline_volatility = _safe_float(baseline_row.get('val_loss_volatility'))
        if current_lr is not None and baseline_lr is not None and current_lr != baseline_lr and current_volatility is not None and baseline_volatility is not None:
            if current_volatility < baseline_volatility:
                reasons.append('learning-rate change reduced late val_loss oscillation')
            elif current_volatility > baseline_volatility:
                reasons.append('learning-rate change increased late val_loss oscillation')

        current_balance_tail = _safe_float(row.get('train_balance_tail_mean'))
        baseline_balance_tail = _safe_float(baseline_row.get('train_balance_tail_mean'))
        current_balance_weight = _safe_float(row.get('codebook_balance_weight'))
        baseline_balance_weight = _safe_float(baseline_row.get('codebook_balance_weight'))
        if current_balance_tail is not None and baseline_balance_tail is not None and current_balance_weight != baseline_balance_weight:
            if current_balance_tail < baseline_balance_tail:
                reasons.append('balance-weight change reduced persistent codebook rebalancing pressure')
            elif current_balance_tail > baseline_balance_tail:
                reasons.append('balance-weight change left more late-stage balance pressure')

        current_observation_tail = _safe_float(row.get('train_observation_tail_ratio'))
        baseline_observation_tail = _safe_float(baseline_row.get('train_observation_tail_ratio'))
        if row.get('source_codebook_size') != baseline_row.get('source_codebook_size') and current_observation_tail is not None and baseline_observation_tail is not None:
            if current_observation_tail > baseline_observation_tail:
                reasons.append('source codebook change improved downstream observation code usage')
            elif current_observation_tail < baseline_observation_tail:
                reasons.append('source codebook change suppressed downstream observation code usage')

    if not reasons:
        reasons.append(PATTERN_DESCRIPTIONS.get(pattern_key, 'The run shows a distinct optimization pattern.'))

    unique_reasons: list[str] = []
    for reason in reasons:
        if reason not in unique_reasons:
            unique_reasons.append(reason)
    return '; '.join(unique_reasons[:2])


def _pattern_primary_cause(
    pattern_key: str,
    row: dict[str, Any],
) -> str | None:
    if pattern_key == 'observation_starved':
        observation_tail_ratio = _safe_float(row.get('train_observation_tail_ratio'))
        if observation_tail_ratio is not None:
            return f'observation codebooks stay under-activated late in training, with tail perplexity ratio only {observation_tail_ratio:.3f}'
        return 'observation codebooks stay under-activated late in training'

    if pattern_key == 'delayed_observation_activation':
        activation_epoch = _safe_int(row.get('observation_activation_epoch'))
        epoch_count = _safe_int(row.get('epoch_count'))
        if activation_epoch is not None and epoch_count is not None:
            return f'observation codebooks only become meaningfully active around epoch {activation_epoch}/{epoch_count}'
        return 'observation codebooks activate too late to influence most of training'

    if pattern_key == 'persistent_balance_pressure':
        balance_tail_mean = _safe_float(row.get('train_balance_tail_mean'))
        if balance_tail_mean is not None:
            return f'codebook balance loss stays elevated late in training at about {balance_tail_mean:.3f}'
        return 'codebook balance loss stays elevated late in training'

    if pattern_key == 'late_instability':
        late_gap = _safe_float(row.get('best_to_last_val_loss_gap'))
        if late_gap is not None:
            return f'the run finds a good checkpoint but gives back {late_gap:.3f} val_loss by the final epoch'
        return 'the run finds a good checkpoint but cannot hold it through late training'

    return None


def _collect_evidence_points(
    row: dict[str, Any],
    baseline_row: dict[str, Any] | None,
) -> list[str]:
    evidence: list[str] = []
    if baseline_row is not None and row.get('run_name') != baseline_row.get('run_name'):
        best_val_delta = _safe_float(row.get('delta_best_val_loss'))
        if best_val_delta is not None:
            direction = 'improved' if best_val_delta < 0 else 'worsened'
            evidence.append(f'best val_loss {direction} by {abs(best_val_delta):.4f}')

        active_delta = _safe_float(row.get('delta_gate1_min_active_ratio'))
        if active_delta is not None:
            evidence.append(f'min active ratio delta {active_delta:+.4f}')

        top5_delta = _safe_float(row.get('delta_gate1_max_top_5_coverage'))
        if top5_delta is not None:
            evidence.append(f'max top-5 coverage delta {top5_delta:+.4f}')

        dead_delta = _safe_float(row.get('delta_gate1_max_dead_code_ratio'))
        if dead_delta is not None:
            evidence.append(f'max dead-code ratio delta {dead_delta:+.4f}')

    observation_activation_epoch = _safe_int(row.get('observation_activation_epoch'))
    epoch_count = _safe_int(row.get('epoch_count'))
    observation_tail_ratio = _safe_float(row.get('train_observation_tail_ratio'))
    if observation_activation_epoch is not None and epoch_count is not None:
        evidence.append(f'observation activation reached the 0.2 codebook-ratio threshold at epoch {observation_activation_epoch}/{epoch_count}')
    if observation_tail_ratio is not None:
        evidence.append(f'late observation perplexity ratio is {observation_tail_ratio:.3f}')

    balance_tail_mean = _safe_float(row.get('train_balance_tail_mean'))
    if balance_tail_mean is not None:
        evidence.append(f'late train codebook balance loss averages {balance_tail_mean:.3f}')

    eeg_amp_reduction = _safe_float(row.get('train_eeg_amp_reduction'))
    if eeg_amp_reduction is not None:
        evidence.append(f'train EEG amplitude loss improved by {eeg_amp_reduction:.3f} from early to late epochs')

    return evidence[:5]


def _render_analysis_section(analysis: dict[str, Any]) -> list[str]:
    lines = ['', '## Pattern Analysis', '']

    pattern_groups = analysis.get('pattern_groups', [])
    if pattern_groups:
        lines.extend(['### Grouped Runs', ''])
        for group in pattern_groups:
            lines.append(f"- {group.get('pattern_label')}: {', '.join(group.get('runs', []))}")
            lines.append(f"  {group.get('summary')}")
        lines.append('')

    run_findings = analysis.get('run_findings', [])
    if run_findings:
        lines.extend(['### Run Findings', ''])
        for finding in run_findings:
            lines.append(f"#### {finding.get('run_name')}")
            lines.append('')
            lines.append(f"- Pattern: {finding.get('pattern_label')}")
            lines.append(f"- Effectiveness: {finding.get('effectiveness_label')}")
            if finding.get('config_changes'):
                lines.append(f"- Config changes: {'; '.join(finding.get('config_changes', []))}")
            lines.append(f"- Hypothesis: {finding.get('hypothesis')}")
            lines.append('- Evidence:')
            for evidence in finding.get('evidence', []):
                lines.append(f'  - {evidence}')
            lines.append('')

    return lines


def _build_history_cache(
    rows: Sequence[dict[str, Any]],
    metric_names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    history_cache: dict[str, dict[str, Any]] = {}
    unique_metrics = list(dict.fromkeys(metric_names))
    for row in rows:
        run_name = _safe_str(row.get('run_name')) or 'unknown'
        run_dir = Path(_safe_str(row.get('run_dir')) or '')
        metrics_payload = _load_json(run_dir / 'metrics.json')
        epochs = metrics_payload.get('epochs', [])
        history_cache[run_name] = {
            'series': {
                metric_name: _extract_metric_series(epochs, metric_name)
                for metric_name in unique_metrics
            },
        }
    return history_cache


def _build_run_plot_styles(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    styles: dict[str, dict[str, Any]] = {}
    baseline_run = next((_safe_str(row.get('baseline_run')) for row in rows if row.get('baseline_run')), None)
    color_map = plt.get_cmap('tab10')
    for index, row in enumerate(rows):
        run_name = _safe_str(row.get('run_name')) or f'run_{index}'
        if run_name == baseline_run:
            styles[run_name] = {'color': '#111111', 'linewidth': 2.8, 'alpha': 0.95}
        else:
            styles[run_name] = {'color': color_map(index % 10), 'linewidth': 1.8, 'alpha': 0.9}
    return styles


def _plot_metric_overlay(
    axis,
    rows: Sequence[dict[str, Any]],
    history_cache: dict[str, dict[str, Any]],
    metric_name: str,
    styles: dict[str, dict[str, Any]],
) -> None:
    has_data = False
    for row in rows:
        run_name = _safe_str(row.get('run_name')) or 'unknown'
        epoch_numbers, values = history_cache.get(run_name, {}).get('series', {}).get(metric_name, ([], []))
        if not values:
            continue
        has_data = True
        axis.plot(epoch_numbers, values, label=run_name, **styles.get(run_name, {}))

    if not has_data:
        axis.text(0.5, 0.5, 'No data', transform=axis.transAxes, ha='center', va='center', fontsize=11)


__all__ = [
    'DEFAULT_TRACK_METRICS',
    'build_markdown_report',
    'build_comparison_analysis',
    'build_run_summary',
    'collect_run_summaries',
    'generate_visualizations',
    'prepare_report_directory',
    'render_markdown_table',
    'resolve_run_dirs',
    'sort_run_summaries',
    'write_report_bundle',
    'write_csv_report',
    'write_json_report',
    'write_markdown_report',
]
