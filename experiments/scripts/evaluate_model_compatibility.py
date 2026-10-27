#!/usr/bin/env python3
"""Native public Hb census; data-free checks and retained descriptive summaries.

The metadata index supplies identities/events only. Measured values always enter
through the unified native Hb reader. Run under the project's durable supervisor
for a census; each terminal record is independently resumable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
from src.data.unified_physiology import load_cached_fnirs_record, CACHED_FNIRS_INPUT_SCHEMA
from src.inference.model_compatibility import weak_hb_profiles

DEFAULT_CONFIG = ROOT / 'experiments/configs/physiology_semantic_tokenizer/model_compatibility_cached_v2.yaml'
SCHEMA = 'model_compatibility_census_cached_v2'
IDENTITY = ['dataset_id', 'subject_id', 'record_id', 'join_key', 'record_token']
ROW_FIELDS = IDENTITY + ['length_s', 'window_id', 'start_s', 'stop_s', 'start_index',
    'stop_index', 'samples', 'pair_index', 'pair', 'status', 'nominal_D', 'best_D',
    'best_tau', 'improvement', 'denominator', 'centered_rms', 'kernel_count']


def _now():
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def load_config(path=DEFAULT_CONFIG):
    config = yaml.safe_load(Path(path).read_text())
    if config.get('schema') != 'model_compatibility_cached_v2':
        raise ValueError('expected cached v2 contract; native v1 replay requires its retained source snapshot')
    census, metric, summary = config['census'], config['metric'], config['summary']
    if census['reader'] != 'src.data.unified_physiology.load_cached_fnirs_record':
        raise ValueError('unified cached Hb reader is required')
    if census['temporal_processing'] != CACHED_FNIRS_INPUT_SCHEMA or census['eeg_access'] != 'none':
        raise ValueError('this census consumes bandpassed cache measurements and reads no EEG')
    if any(float(x) <= 0 for x in census['lengths_s']) or len(set(census['lengths_s'])) != len(census['lengths_s']):
        raise ValueError('lengths_s must be unique positive durations')
    tau = tau_grid(config)
    if metric['nominal_tau'] not in tau or not 0 < metric['rcond'] < 1:
        raise ValueError('nominal tau must be in the fixed profile grid')
    if summary['bootstrap_replicates'] < 1 or summary['ecdf_points'] < 3:
        raise ValueError('invalid summary resolution')
    if summary['metrics'] != ['nominal_D', 'best_D', 'best_tau', 'improvement']:
        raise ValueError('unexpected summary metric contract')
    return config


def tau_grid(config):
    metric = config['metric']
    lo, hi, step = (float(metric[x]) for x in ('tau_min', 'tau_max', 'tau_step'))
    if not np.isfinite([lo, hi, step]).all() or lo <= 0 or step <= 0 or hi < lo:
        raise ValueError('invalid tau grid')
    grid = np.arange(lo, hi + step / 2, step)
    if not np.isclose(grid[-1], hi):
        raise ValueError('tau endpoints must be exact grid nodes')
    return grid


def metric_kwargs(config):
    metric = config['metric']
    return dict(taus=tau_grid(config), nominal_tau=float(metric['nominal_tau']),
        half_widths=tuple(metric['half_widths_s']), center_spacing=float(metric['center_spacing_s']),
        rho=float(metric['rho']), xi=metric['xi'], zeta=metric['zeta'], rcond=float(metric['rcond']))


def record_identity(record):
    return dict(dataset_id=record.dataset_id, subject_id=record.canonical_subject_id,
        record_id=record.record_id, join_key=record.join_key,
        record_token=hashlib.sha256(record.join_key.encode()).hexdigest()[:16])


def alignment_boundaries(events):
    boundaries = set()
    for event in events:
        support = event.get('metadata', {}).get('alignment_support_ms', {}).get('fnirs', [])
        boundaries.update(float(x) / 1000 for x in support if x is not None and np.isfinite(float(x)))
    return sorted(boundaries)


def enumerate_windows(duration_s, lengths_s, boundaries=()):
    """Half-open nonoverlapping native-time windows; drop tails without padding."""
    windows = []
    for length in lengths_s:
        length = float(length)
        for index in range(max(0, int(np.floor((float(duration_s) + 1e-8) / length)))):
            start, stop = index * length, (index + 1) * length
            windows.append(dict(length_s=length, window_id=index, start_s=start, stop_s=stop,
                status='excluded_alignment_boundary' if any(start < b < stop for b in boundaries) else 'planned'))
    return windows


def plan_inventory(index, config):
    """Use metadata only, including explicit estimates before native clocks load."""
    census = config['census']
    excluded = set(census['excluded_join_keys'])
    records = [r for r in index.records if r.dataset_id in census['datasets'] and r.join_key not in excluded]
    records.sort(key=lambda r: r.join_key)
    if len({r.join_key for r in records}) != len(records):
        raise ValueError('duplicate canonical records require an explicit branch contract')
    subjects = {(r.dataset_id, r.canonical_subject_id) for r in records}
    if len(records) != census['expected_records'] or len(subjects) != census['expected_subjects']:
        raise ValueError(f'fixed public inventory differs: {len(records)} records, {len(subjects)} dataset-native subjects')
    inventory = []
    for record in records:
        row = record_identity(record)
        count, channels = record.manifest['array_shapes']['fnirs']
        if channels % 2:
            raise ValueError('registered Hb channel count must be even')
        duration = count / record.sample_rate_hz
        windows = enumerate_windows(duration, census['lengths_s'], alignment_boundaries(index.events_by_join_key.get(record.join_key, [])))
        row.update(signal_branch=record.signal_branch, pair_count=channels // 2,
            cache_duration_estimate_s=duration,
            estimated_windows_by_length={str(int(length)): sum(w['length_s'] == length for w in windows) for length in census['lengths_s']},
            estimated_pair_windows=len(windows) * channels // 2,
            source_files=record.manifest.get('source_files', []),
            native_contract=record.manifest.get('native_contract', {}))
        inventory.append(row)
    plan = dict(schema=SCHEMA, created_at=_now(), records=len(records), subjects=len(subjects),
        datasets=dict(Counter(r.dataset_id for r in records)),
        excluded_join_keys=sorted(excluded), lengths_s=census['lengths_s'],
        count_note='Record/subject identities exact; cache-duration window counts are estimates until cached arrays and support are read.',
        estimated_pair_windows=sum(r['estimated_pair_windows'] for r in inventory),
        stopping_rule=config['stopping_rule'])
    return plan, inventory, records


def prepare_output(output, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'resolved_config.yaml'
    if path.exists():
        if yaml.safe_load(path.read_text()) != config:
            raise ValueError('existing run has a different resolved contract; choose a fresh versioned output')
    else:
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return output


def make_plan(output, config, index):
    plan, inventory, records = plan_inventory(index, config)
    path = output / 'record_inventory.jsonl'
    if path.exists():
        previous = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if previous != _jsonable(inventory):
            raise ValueError('metadata inventory changed; retained plan is immutable')
    else:
        path.write_text(''.join(json.dumps(_jsonable(row), ensure_ascii=False) + '\n' for row in inventory))
        write_json(output / 'plan.json', plan)
    return plan, inventory, records


def _paths(output, record):
    base = Path(output) / 'records' / record_identity(record)['record_token']
    return base.with_suffix('.csv.gz'), base.with_suffix('.meta.json'), base.with_suffix('.profiles.npz')


def replay_window(record, window, *, config, project_root=ROOT):
    """Replay a cached-v2 window; never relabel a retained native-v1 curve."""
    if config.get('schema') != 'model_compatibility_cached_v2':
        raise ValueError('Historical native curve replay requires the retained v1 source snapshot')
    native = load_cached_fnirs_record(Path(project_root), record)
    times = np.asarray(native['time_s'], float)
    left, right = np.searchsorted(times, [float(window['start_s']), float(window['stop_s'])])
    if 'start_index' in window and (left != int(window['start_index']) or right != int(window['stop_index'])):
        raise ValueError('cache clock differs from retained window identity')
    return dict(values=np.asarray(native['values'])[left:right], time_s=times[left:right],
        channel_names=native['channel_names'], provenance=native['provenance'])


def evaluate_record(record, events, config, output, *, project_root=ROOT, reader=None, metric_fn=None):
    """One record is the resumable unit; failures never disappear from the census."""
    started = time.monotonic()
    csv_path, meta_path, profile_path = _paths(output, record)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    identity = record_identity(record)
    meta = dict(schema=SCHEMA, **identity, execution='failed', started_at=_now())
    try:
        if config['census']['temporal_processing'] != CACHED_FNIRS_INPUT_SCHEMA:
            raise ValueError('Retired native census cannot produce new measurement evidence')
        native = (load_cached_fnirs_record if reader is None else reader)(Path(project_root), record)
        hb, times = np.asarray(native['values']), np.asarray(native['time_s'], float)
        if hb.ndim != 3 or hb.shape[2] != 2 or times.shape != (len(hb),) or len(hb) < 3:
            raise ValueError('native Hb must be [T,pair,2] with matching T>=3 clock')
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or times[0] < -1e-8:
            raise ValueError('native clock must be finite increasing nonnegative seconds')
        rate = float(native['sample_rate_hz'])
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError('native sample rate must be positive finite')
        native_dt = 1 / rate
        duration = float(times[-1] + native_dt)
        windows = enumerate_windows(duration, config['census']['lengths_s'], alignment_boundaries(events))
        names = native['channel_names']
        if len(names) != hb.shape[1] * 2:
            raise ValueError('native Hb pair identities do not match tensor')
        rows, profiles, failures = [], {}, []
        for length in config['census']['lengths_s']:
            selected = [w for w in windows if w['length_s'] == length]
            label = str(int(length)) + 's'
            profiles['profiles_' + label] = np.full((len(selected), hb.shape[1], len(tau_grid(config))), np.nan)
            profiles['window_start_s_' + label] = np.asarray([w['start_s'] for w in selected])
            profiles['window_stop_s_' + label] = np.asarray([w['stop_s'] for w in selected])
        for window in windows:
            left, right = np.searchsorted(times, [window['start_s'], window['stop_s']])
            values = hb[left:right]
            result = None
            status = window['status']
            if status == 'planned':
                clock_tolerance = max(1e-7, native_dt*1e-5)
                if (len(values) < 3
                        or times[left] > window['start_s'] + 1.5*native_dt
                        or times[right-1] + native_dt < window['stop_s'] - clock_tolerance
                        or np.any(np.diff(times[left:right]) > 1.5*native_dt)):
                    status = 'missing_time_support'
                else:
                    try:
                        result = (weak_hb_profiles if metric_fn is None else metric_fn)(values, times=times[left:right], **metric_kwargs(config))
                    except Exception as error:
                        status = 'numerical_failure'
                        failures.append(dict(length_s=window['length_s'], window_id=window['window_id'],
                            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
            if result is not None:
                profiles['profiles_' + str(int(window['length_s'])) + 's'][window['window_id']] = result['profiles']
            for pair in range(hb.shape[1]):
                row = dict(**identity, **{k: window[k] for k in ('length_s', 'window_id', 'start_s', 'stop_s')},
                    start_index=int(left), stop_index=int(right), samples=len(values), pair_index=pair,
                    pair=str(names[2 * pair]).removesuffix('_HbO'),
                    status=str(result['status'][pair]) if result is not None else status)
                for name in ('nominal_D', 'best_D', 'best_tau', 'denominator'):
                    row[name] = float(result[name][pair]) if result is not None else np.nan
                row['improvement'] = row['nominal_D'] - row['best_D']
                row['centered_rms'] = float(np.sqrt(np.mean((values[:, pair] - values[:, pair].mean(axis=0))**2))) if len(values) and np.isfinite(values[:, pair]).all() else np.nan
                row['kernel_count'] = int(result['kernel_count']) if result is not None else 0
                rows.append(row)
        frame = pd.DataFrame(rows, columns=ROW_FIELDS)
        temporary = csv_path.with_name(csv_path.name + '.partial')
        frame.to_csv(temporary, index=False, compression='gzip')
        temporary.replace(csv_path)
        profiles.update(taus=tau_grid(config), pair_names=np.asarray(names[::2], dtype=str))
        temporary = profile_path.with_name(profile_path.name + '.partial')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **profiles)
        temporary.replace(profile_path)
        meta.update(execution='completed', rows=len(rows), windows=len(windows), pairs=hb.shape[1],
            duration_s=duration, sample_rate_hz=float(native['sample_rate_hz']),
            measurement_provenance=native['provenance'], source_temporal_processing=CACHED_FNIRS_INPUT_SCHEMA,
            boundary_s=alignment_boundaries(events), status_counts=dict(Counter(frame.status)),
            windows_by_length={str(int(length)): sum(w['length_s'] == length for w in windows) for length in config['census']['lengths_s']},
            tail_s_by_length={str(int(length)): duration % float(length) for length in config['census']['lengths_s']},
            numerical_window_failures=failures, csv_path=str(csv_path), profiles_path=str(profile_path))
    except Exception as error:
        meta.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    meta.update(finished_at=_now(), elapsed_s=time.monotonic() - started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    write_json(meta_path, meta)
    return meta


def _record_job(job):
    return evaluate_record(*job)


def execute_records(records, events_by_key, config, output, *, workers=1, resume=False):
    """Bound submitted work to twice the process count; no unbounded futures."""
    if workers < 1:
        raise ValueError('workers must be positive')
    output = Path(output)
    terminal, jobs = [], []
    for record in records:
        _, meta_path, _ = _paths(output, record)
        if meta_path.exists():
            if not resume:
                raise ValueError('record evidence already exists; use --resume or a fresh output')
            meta = json.loads(meta_path.read_text())
            if meta['join_key'] != record.join_key or meta['execution'] not in ('completed', 'failed'):
                raise ValueError('invalid retained terminal record')
            terminal.append(meta)
        else:
            jobs.append((record, events_by_key.get(record.join_key, []), config, output))
    started = time.monotonic()
    def progress():
        done = len(terminal)
        failed = sum(m['execution'] == 'failed' for m in terminal)
        elapsed = time.monotonic() - started
        produced = done - (len(records) - len(jobs))
        eta = elapsed / produced * (len(records) - done) if produced else None
        state = dict(schema=SCHEMA, updated_at=_now(), records_planned=len(records),
            records_terminal=done, records_completed=done-failed, records_failed=failed,
            workers=workers, max_in_flight=2*workers, elapsed_s=elapsed, estimated_remaining_s=eta)
        write_json(output / 'progress.json', state)
        print(json.dumps(state), flush=True)
    progress()
    if workers == 1:
        for job in jobs:
            terminal.append(_record_job(job))
            progress()
    else:
        iterator = iter(jobs)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {}
            def submit():
                for _ in range(2*workers-len(pending)):
                    job = next(iterator, None)
                    if job is None:
                        break
                    pending[pool.submit(_record_job, job)] = job
            submit()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    job = pending.pop(future)
                    try:
                        terminal.append(future.result())
                    except Exception as error:
                        record = job[0]
                        meta = dict(schema=SCHEMA, **record_identity(record), execution='failed',
                            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
                        write_json(_paths(output, record)[1], meta)
                        terminal.append(meta)
                    progress()
                submit()
    terminal.sort(key=lambda m: m['join_key'])
    write_json(output / 'execution.json', dict(schema=SCHEMA, finished_at=_now(),
        execution='completed_with_failures' if any(m['execution'] == 'failed' for m in terminal) else 'completed',
        records_planned=len(records), records_terminal=len(terminal),
        records_failed=sum(m['execution'] == 'failed' for m in terminal), records=terminal))
    return terminal


def weighted_quantile(values, weights, probabilities):
    """Exact inverse weighted empirical CDF, with no interpolation."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(values, kind='stable')
    x, w = values[order], weights[order]
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('weighted quantile requires finite values and positive total weight')
    cumulative = np.cumsum(w)
    cumulative /= cumulative[-1]
    indices = np.minimum(np.searchsorted(cumulative, probabilities, side='left'), len(x)-1)
    return x[indices]


def _record_distribution(group, metrics):
    good = group.loc[group.status == 'ok'].dropna(subset=metrics).copy()
    if not len(good):
        return None
    counts = good.groupby('window_id').size()
    weights = 1 / (len(counts) * good.window_id.map(counts).to_numpy(float))
    weights /= weights.sum()
    return good[metrics].to_numpy(float), weights, len(counts)


def summarize(output, config):
    """Read one record at a time; retain only valid scores and identity weights.

    Quantiles are exact weighted ECDF inverses. The plotting ECDF is compressed
    by observed ranks, without using that grid for any table value. Subject
    bootstrap means and medians are explicitly separate estimands.
    """
    output = Path(output)
    inventory = [json.loads(line) for line in (output/'record_inventory.jsonl').read_text().splitlines() if line.strip()]
    metas = []
    for row in inventory:
        path = output/'records'/f"{row['record_token']}.meta.json"
        if not path.exists():
            raise ValueError('summarize requires every planned record to be terminal')
        metas.append(json.loads(path.read_text()))
    completed = [m for m in metas if m['execution'] == 'completed']
    metrics = config['summary']['metrics']
    subject_records = defaultdict(list)
    record_rows, statuses = [], []
    status_totals = Counter()
    for meta in completed:
        frame = pd.read_csv(meta['csv_path'])
        for length, group in frame.groupby('length_s'):
            for status, count in group.status.value_counts().items():
                statuses.append(dict(**{k: meta[k] for k in IDENTITY}, length_s=float(length), status=status, pair_windows=int(count)))
                status_totals[(meta['dataset_id'],float(length),status)] += int(count)
            distribution = _record_distribution(group, metrics)
            if distribution is not None:
                subject_records[(meta['dataset_id'],meta['subject_id'],float(length))].append(distribution)
            for column, metric in enumerate(metrics):
                values, weights, nwin = distribution if distribution is not None else (np.empty((0,len(metrics))), np.empty(0), 0)
                quantiles = weighted_quantile(values[:,column],weights,[.5,.9]) if len(values) else [np.nan,np.nan]
                record_rows.append(dict(**{k: meta[k] for k in IDENTITY}, length_s=float(length), metric=metric,
                    planned_pair_windows=len(group), valid_pair_windows=len(values),
                    planned_windows=group.window_id.nunique(), valid_windows=nwin,
                    mean=float(weights@values[:,column]) if len(values) else np.nan,
                    median=quantiles[0],p90=quantiles[1]))
    planned = Counter((r['dataset_id'],r['subject_id']) for r in inventory)
    completed_counts = Counter((m['dataset_id'],m['subject_id']) for m in completed)
    subject_data, subject_rows = {}, []
    for (dataset,subject), nrecords in sorted(planned.items()):
        for length in config['census']['lengths_s']:
            records = subject_records.get((dataset,subject,float(length)),[])
            if records:
                values = np.concatenate([r[0] for r in records])
                weights = np.concatenate([r[1]/len(records) for r in records])
                subject_data[(dataset,subject,float(length))] = (values,weights)
            else:
                values,weights = np.empty((0,len(metrics))),np.empty(0)
            for column,metric in enumerate(metrics):
                quantiles = weighted_quantile(values[:,column],weights,[.5,.9]) if len(values) else [np.nan,np.nan]
                subject_rows.append(dict(dataset_id=dataset,subject_id=subject,length_s=float(length),metric=metric,
                    planned_records=nrecords,completed_records=completed_counts[(dataset,subject)],valid_records=len(records),
                    valid_pair_windows=len(values),mean=float(weights@values[:,column]) if len(values) else np.nan,
                    median=quantiles[0],p90=quantiles[1]))
    rng = np.random.default_rng(config['seed'])
    summary_rows, ecdf_rows = [], []
    alpha = (1-float(config['summary']['confidence']))/2
    datasets = sorted({r['dataset_id'] for r in inventory})+['all_dataset_subjects']
    for dataset in datasets:
        select = inventory if dataset == 'all_dataset_subjects' else [r for r in inventory if r['dataset_id']==dataset]
        selected_metas = metas if dataset == 'all_dataset_subjects' else [m for m in metas if m['dataset_id']==dataset]
        for length in config['census']['lengths_s']:
            subjects = [data for (ds,_,support),data in sorted(subject_data.items())
                if support==length and (dataset=='all_dataset_subjects' or ds==dataset)]
            counts = Counter()
            for (native_dataset,support,status),count in status_totals.items():
                if support==length and (dataset=='all_dataset_subjects' or native_dataset==dataset):
                    counts[status] += count
            denominators = dict(planned_pair_windows=sum(counts.values()),
                excluded_pair_windows=sum(n for name,n in counts.items() if name.startswith('excluded_')),
                missing_pair_windows=counts['missing_support']+counts['missing_time_support'],
                insufficient_energy_pair_windows=counts['insufficient_energy'],
                numerical_failure_pair_windows=counts['numerical_failure'])
            if not subjects:
                for metric in metrics:
                    summary_rows.append(dict(dataset_id=dataset,length_s=float(length),metric=metric,
                        planned_records=len(select),completed_records=sum(m['execution']=='completed' for m in selected_metas),
                        failed_records=sum(m['execution']=='failed' for m in selected_metas),
                        planned_subjects=len({(r['dataset_id'],r['subject_id']) for r in select}),
                        valid_subjects=0,valid_pair_windows=0,mean=np.nan,**denominators,
                        **{name:np.nan for name in ['raw_median','raw_p90','subject_median',
                            'mean_ci_low','mean_ci_high','subject_median_ci_low','subject_median_ci_high']
                            +[f'q{int(round(p*100)):02d}' for p in config['summary']['quantiles']]}))
                continue
            values = np.concatenate([data[0] for data in subjects])
            weights = np.concatenate([data[1]/len(subjects) for data in subjects])
            weights /= weights.sum()
            for column,metric in enumerate(metrics):
                x = values[:,column]
                subject_means = np.array([data[1]@data[0][:,column] for data in subjects])
                subject_medians = np.array([weighted_quantile(data[0][:,column],data[1],[.5])[0] for data in subjects])
                quantiles = weighted_quantile(x,weights,config['summary']['quantiles'])
                row = dict(dataset_id=dataset,length_s=float(length),metric=metric,
                    planned_records=len(select),completed_records=sum(m['execution']=='completed' for m in selected_metas),
                    failed_records=sum(m['execution']=='failed' for m in selected_metas),
                    planned_subjects=len({(r['dataset_id'],r['subject_id']) for r in select}),
                    valid_subjects=len(subjects),valid_pair_windows=len(x),mean=float(weights@x),
                    **denominators,
                    raw_median=float(np.median(x)),raw_p90=float(np.quantile(x,.9,method='inverted_cdf')),
                    subject_median=float(np.median(subject_medians)))
                for probability,value in zip(config['summary']['quantiles'],quantiles):
                    row[f'q{int(round(probability*100)):02d}']=float(value)
                draws = rng.integers(0,len(subjects),(config['summary']['bootstrap_replicates'],len(subjects)))
                row['mean_ci_low'],row['mean_ci_high']=np.quantile(subject_means[draws].mean(axis=1),[alpha,1-alpha])
                row['subject_median_ci_low'],row['subject_median_ci_high']=np.quantile(np.median(subject_medians[draws],axis=1),[alpha,1-alpha])
                summary_rows.append(row)
                order=np.argsort(x,kind='stable');ordered=x[order]
                cumulative=np.cumsum(weights[order]);cumulative/=cumulative[-1]
                sample=np.unique(np.linspace(0,len(x)-1,min(len(x),config['summary']['ecdf_points']),dtype=int))
                sampled=np.unique(ordered[sample]);right=np.searchsorted(ordered,sampled,side='right')
                ecdf_rows.extend(dict(dataset_id=dataset,length_s=float(length),metric=metric,value=float(value),
                    raw_ecdf=float(rank/len(x)),subject_equal_ecdf=float(cumulative[rank-1]))
                    for value,rank in zip(sampled,right))
    pd.DataFrame(record_rows,columns=IDENTITY+['length_s','metric','planned_pair_windows',
        'valid_pair_windows','planned_windows','valid_windows','mean','median','p90']).to_csv(output/'record_summary.csv',index=False)
    pd.DataFrame(subject_rows).to_csv(output/'subject_summary.csv',index=False)
    pd.DataFrame(statuses,columns=IDENTITY+['length_s','status','pair_windows']).to_csv(output/'status_counts.csv',index=False)
    pd.DataFrame(summary_rows).to_csv(output/'summary.csv',index=False)
    pd.DataFrame(ecdf_rows,columns=['dataset_id','length_s','metric','value','raw_ecdf',
        'subject_equal_ecdf']).to_csv(output/'distribution_ecdf.csv.gz',index=False,compression='gzip')
    counts=Counter()
    for meta in completed:
        counts.update(meta['status_counts'])
    summary=dict(schema=SCHEMA,execution='completed',created_at=_now(),
        planned_records=len(inventory),completed_records=len(completed),failed_records=len(metas)-len(completed),
        planned_subjects=len(planned),total_planned_pair_windows=sum(m.get('rows',0) for m in completed),
        unknown_native_window_denominator_records=[m['join_key'] for m in metas if m['execution']=='failed'],
        status_counts=dict(counts),hierarchy=config['summary']['hierarchy'],
        quantile_rule='exact inverse weighted ECDF; table values do not use plotting compression',
        ecdf_plot_compression='at most ecdf_points observed ranks per dataset/length/metric; CDF values exact at retained nodes',
        bootstrap_unit=config['summary']['bootstrap_unit'],bootstrap_replicates=config['summary']['bootstrap_replicates'],
        bootstrap_estimands=['mean of subject means','median of subject medians'],confidence=config['summary']['confidence'])
    write_json(output/'summary.json',summary)
    return summary


def synthetic_hb(sample_rate_hz, *, initial=None):
    """Exact small-signal rest-Jacobian response with independent initial state."""
    from src.inference import t3a_balloon_robust_ssm as core
    time = np.arange(0,30,1/float(sample_rate_hz))
    parameters = core.BalloonParameters()
    jacobian = core.balloon_rhs_jacobian(np.zeros(6),parameters)
    observation = core.observation_jacobian(np.zeros(6),parameters)
    a,b = jacobian[1:,1:],jacobian[1:,0]
    state,drive = np.zeros((len(time),5)),np.zeros(len(time))
    for frequency,amplitude,phase in ((1/30,.014,.2),(1/12,.009,1.1),(1/7.5,.006,-.4)):
        oscillation=amplitude*np.exp(1j*(2*np.pi*frequency*time+phase))
        state+=np.real(oscillation[:,None]*np.linalg.solve(2j*np.pi*frequency*np.eye(5)-a,b))
        drive+=oscillation.real
    if initial is not None:
        eigenvalues,eigenvectors=np.linalg.eig(a)
        weights=np.linalg.solve(eigenvectors,np.asarray(initial))
        state+=np.real((np.exp(time[:,None]*eigenvalues)*weights)@eigenvectors.T)
    values=state@observation[:,1:].T+drive[:,None]*observation[:,0]
    return time,values[:,1:]


def synthetic_checks(config):
    """No cache/index/source access: exercise tensors, exclusions and invariants."""
    times = np.arange(300)/10
    y = np.stack([np.sin(times/3), .3*np.cos(times/4)], axis=-1)[:,None,:]
    result = weak_hb_profiles(y, times=times, **metric_kwargs(config))
    scaled = weak_hb_profiles(7*y+[11.,-9.], times=times, **metric_kwargs(config))
    missing = y.copy(); missing[20,0,1] = np.nan
    absent = weak_hb_profiles(missing, times=times, **metric_kwargs(config))
    constant = weak_hb_profiles(np.ones_like(y), times=times, **metric_kwargs(config))
    assert result['profiles'].shape == (1,len(tau_grid(config)))
    assert result['status'][0] == 'ok'
    assert np.allclose(result['profiles'], scaled['profiles'],atol=1e-10)
    assert absent['status'][0] == 'missing_support' and np.isnan(absent['nominal_D'][0])
    assert constant['status'][0] == 'insufficient_energy'
    assert result['best_D'][0] <= result['nominal_D'][0]+1e-12
    windows = enumerate_windows(185,[30,90],[45,90])
    assert len(windows)==8 and sum(w['status'].startswith('excluded') for w in windows)==2
    cases=[]
    for rate in (10.,47.62):
        t,base=synthetic_hb(rate)
        _,initial=synthetic_hb(rate,initial=[.01,.02,.03,.13,.03])
        assert initial[0,0]*initial[0,1] > 0
        altered=base.copy();altered[:,1]+=.03*np.sin(2*np.pi*t/6.3+.7)
        variants={'legal_small_signal':base,'legal_same_phase_initial':initial,
            'common_scale_offset':7*base+[11.,-9.],
            'relative_HbR_gain_2':base*[1.,2.], 'independent_HbR_component':altered}
        for label,signal in variants.items():
            score=weak_hb_profiles(signal,times=t,**metric_kwargs(config))
            cases.append(dict(case=label,sample_rate_hz=rate,seed=None,noise_relative_rms=0,
                nominal_D=float(score['nominal_D'][0]),best_D=float(score['best_D'][0]),
                best_tau=float(score['best_tau'][0]),status=str(score['status'][0]),
                initial_HbO=float(signal[0,0]),initial_HbR=float(signal[0,1])))
        signal_rms=float(np.sqrt(np.mean((base-base.mean(axis=0))**2)))
        for seed in (3,4,5,6):
            rng=np.random.default_rng(seed)
            noise=rng.normal(size=base.shape)
            noise/=np.sqrt(np.mean(noise**2))
            for level in (0.,.03,.1):
                score=weak_hb_profiles(base+noise*level*signal_rms,times=t,**metric_kwargs(config))
                cases.append(dict(case='additive_independent_noise',sample_rate_hz=rate,seed=seed,
                    noise_relative_rms=level,nominal_D=float(score['nominal_D'][0]),
                    best_D=float(score['best_D'][0]),best_tau=float(score['best_tau'][0]),status=str(score['status'][0])))
    convergence=[]
    for rate in (10.,20.,47.62):
        t,signal=synthetic_hb(rate,initial=[.01,.02,.03,.13,.01])
        score=weak_hb_profiles(signal,times=t,**metric_kwargs(config))
        convergence.append(dict(sample_rate_hz=rate,nominal_D=float(score['nominal_D'][0])))
    assert convergence[2]['nominal_D'] < convergence[1]['nominal_D'] < convergence[0]['nominal_D']
    return dict(schema=SCHEMA,execution='passed',data_access='none',
        profile_shape=list(result['profiles'].shape),nominal_D=float(result['nominal_D'][0]),
        best_D=float(result['best_D'][0]),status_missing=str(absent['status'][0]),
        status_constant=str(constant['status'][0]),cases=cases,
        quadrature_convergence=convergence,
        truth='Exact linearized rest-Jacobian frequency response plus legal homogeneous initial state; perturbations explicitly synthetic.',
        noise_rule='One fixed Gaussian realization per seed and rate, RMS normalized, reused at 0/.03/.1 of joint centered Hb signal RMS.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['plan','synthetic','pilot','census','summarize','panels'],required=True)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--panel-stage',choices=['plan','linear','nonlinear','space','all',
        'pilot-linear','pilot-nonlinear','pilot-space'],default='all')
    args=parser.parse_args(argv)
    config=load_config(args.config)
    if args.stage == 'panels':
        raise ValueError('The cached census cannot reuse native-parent fitted panels. Replay retained panels with their frozen v1 source; new panels require newly fitted cached parents.')
    output=prepare_output(args.output,config)
    if args.stage=='synthetic':
        result=synthetic_checks(config)
        write_json(output/'synthetic_checks.json',result)
    elif args.stage=='summarize':
        result=summarize(output,config)
    else:
        index=CleanPhysiologyCacheIndex(ROOT/config['census']['metadata_cache'])
        plan,_,records=make_plan(output,config,index)
        if args.stage=='plan':
            result=plan
        else:
            if args.stage=='pilot':
                selected=[]
                for dataset in sorted(config['census']['datasets']):
                    selected.extend([r for r in records if r.dataset_id==dataset][:config['census']['pilot_records_per_dataset']])
                records=selected
                output=output/'pilot'
            terminal=execute_records(records,index.events_by_join_key,config,output,workers=args.workers,resume=args.resume)
            result=dict(execution='completed',records=len(terminal),failed=sum(m['execution']=='failed' for m in terminal))
    print(json.dumps(_jsonable(result),ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
