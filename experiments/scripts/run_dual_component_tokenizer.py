#!/usr/bin/env python3
"""One resumable controller for the public dual-component tokenizer suite.

The data owner performs all read-boundary checks. This entry owns the immutable
resolved contract, bounded record/teacher work, training checkpoints and task
outcomes. A source snapshot may run here while ``--project-root`` still points
at the original public data registry. No dataset is opened by check/synthetic.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
import traceback

for _variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '1'
SOURCE = Path(__file__).resolve().parents[2]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
import numpy as np
import yaml

DEFAULT_CONFIG = SOURCE / 'experiments/configs/physiology_semantic_tokenizer/dual_component_tokenizer_v1.yaml'
STAGES = ('check', 'plan', 'synthetic', 'prepare', 'teacher', 'pack', 'benchmark',
          'train-one', 'masked-one', 'sensitivity', 'analyze', 'masked', 'all', 'refine', 'refine-one',
          'voltage', 'voltage-one')
VARIANTS = ('continuous_student', 'hard_student', 'component_compression', 'smooth_hard', 'ordinary_hard')
TERMINAL = ('completed', 'failed', 'unavailable')


def serial(value):
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)): return [serial(v) for v in value]
    if isinstance(value, np.generic): return serial(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def identity(value):
    return hashlib.sha256(json.dumps(serial(value), sort_keys=True).encode()).hexdigest()[:24]


def now(): return datetime.now(timezone.utc).isoformat()


def read_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if isinstance(cfg, dict) and cfg.get('schema') == 'dual_component_voltage_v1':
        extra=cfg;cfg=yaml.safe_load((SOURCE/extra['base_contract']).read_text())
        cfg.update(schema=extra['schema'],experiment_id=extra['experiment_id'],voltage=extra['voltage'],claim_limits=extra['claim_limits'])
        cfg['resources'].update(extra['resources'])
        v=cfg['voltage']
        if (v['selection_splits']!=['train','validation'] or v['final_split']!='test'
                or v['eeg']['sample_rate_hz']!=200 or v['hb']['sample_rate_hz']!=10
                or v['eeg']['hop_samples']!=20 or v['hb']['hop_samples']!=20
                or v['steps']<1 or v['codebook_size']!=128):
            raise ValueError('frozen voltage tensor/split/finite-budget contract required')
    if isinstance(cfg, dict) and cfg.get('schema') == 'dual_component_refinement_v1':
        base = yaml.safe_load((SOURCE / cfg['base_contract']).read_text())
        base.update(schema=cfg['schema'], experiment_id=cfg['experiment_id'],
            refinement=cfg['refinement'], claim_limits=cfg['claim_limits'])
        base['resources'].update(cfg['resources'])
        cfg = base
        if cfg['refinement']['allowed_splits'] != ['train', 'validation']:
            raise ValueError('refinement cannot select on the held-out test subjects')
    if not isinstance(cfg, dict) or not cfg.get('schema') or not cfg.get('experiment_id'):
        raise ValueError('versioned schema and experiment identity required')
    if sorted(cfg.get('variants', VARIANTS)) != sorted(VARIANTS):
        raise ValueError('all five declared tokenizer comparisons are required')
    training = cfg['training']
    if min(int(training[k]) for k in ('steps', 'batch_size', 'checkpoint_every')) < 1:
        raise ValueError('positive fixed training and checkpoint budgets required')
    if float(training['learning_rate']) <= 0:
        raise ValueError('positive learning rate required')
    resources = cfg.get('resources', {})
    if int(resources.get('numerical_threads', 1)) != 1:
        raise ValueError('one numerical-library thread per CPU worker is required')
    if int(resources.get('max_in_flight_per_worker', 2)) not in (1, 2):
        raise ValueError('bounded CPU task submission required')
    if cfg['data'].get('protected_data') != 'forbidden':
        raise ValueError('this public suite cannot read protected data')
    return cfg


def seeds(cfg): return list(cfg['training'].get('seeds', cfg.get('seeds', (17, 29, 43))))


def freeze_contract(out, cfg, project_root):
    """Never resume under a different contract, source entry or data root."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    path = out / 'resolved_config.json'
    resolved = dict(config=serial(cfg), project_root=str(Path(project_root).resolve()),
                    source_root=str(SOURCE.resolve()))
    if path.exists():
        if read_json(path) != resolved:
            raise ValueError('resume contract drift: use the original source snapshot, config and project root')
    else:
        write_json(path, resolved)
    return resolved


@contextmanager
def owner_lock(out, name='controller', *, blocking=False):
    """Nonblocking POSIX owner lock; a crashed owner releases it automatically."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    handle = (out / (name + '.lock')).open('a')
    try:
        try: fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc: raise RuntimeError(f'another owner is active: {out / name}') from exc
        yield
    finally:
        handle.close()


def task_path(out, stage, key):
    return Path(out) / 'tasks' / stage / identity(key)


def task_begin(out, stage, key, retry_failed=False):
    base = task_path(out, stage, key)
    path = base / 'status.json'
    previous = read_json(path) if path.exists() else {}
    if previous.get('status') == 'completed': return None
    if previous.get('status') in ('failed', 'unavailable') and not retry_failed: return None
    attempt = int(previous.get('attempt', 0)) + 1
    if previous:
        # Retain an interrupted attempt as well as numerical/preparation failures.
        retained = dict(previous)
        if retained.get('status') == 'running': retained['status'] = 'interrupted'
        write_json(base / f'attempt_{attempt - 1:04d}.json', retained)
    status = dict(key=key, stage=stage, status='running', attempt=attempt, pid=os.getpid(), started=now())
    write_json(path, status)
    return status


def task_finish(out, stage, key, status, **details):
    path = task_path(out, stage, key) / 'status.json'
    record = read_json(path) if path.exists() else dict(key=key, stage=stage, attempt=1)
    record.update(status=status, finished=now(), **details)
    write_json(path, record)
    return record


def task_status(out, stage, key):
    path = task_path(out, stage, key) / 'status.json'
    return read_json(path) if path.exists() else dict(status='pending', key=key)


def array_paths(out, stage, row):
    return Path(out) / stage / (identity(row['id']) + '.npz')


def read_arrays(out, stage, row, keys=None):
    if task_status(out, stage, row['id'])['status'] != 'completed':
        raise FileNotFoundError(f'{stage} unavailable for {row["id"]}')
    with np.load(array_paths(out, stage, row), allow_pickle=False) as archive:
        return {key: archive[key] for key in (archive.files if keys is None else keys)}


def _bounded_jobs(worker, payloads, workers, inflight=2, progress=None):
    """Bound submissions and return every failure without dropping a denominator."""
    iterator = iter(payloads); results = []
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        pending = {}
        def submit():
            try: payload = next(iterator)
            except StopIteration: return False
            pending[pool.submit(worker, payload)] = payload
            return True
        for _ in range(workers * inflight):
            if not submit(): break
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                payload = pending.pop(future)
                try: result = future.result()
                except Exception:
                    # Worker-level failures are retained by the caller using its
                    # original payload identity, rather than disappearing.
                    result = dict(status='failed', payload=payload, error=traceback.format_exc())
                results.append(result)
                if progress is not None: progress(result, len(results))
                submit()
    return results


def plan(project_root, cfg, out):
    from src.data.dual_component_tokenizer import metadata_inventory
    manifest = metadata_inventory(project_root, cfg)
    path = Path(out) / 'inventory.json'
    if path.exists() and read_json(path) != serial(manifest):
        raise ValueError('planned metadata changed; frozen inventory cannot be overwritten')
    write_json(path, manifest)
    return dict(status='completed', planned_windows=len(manifest['rows']),
                planned_trials=manifest['planned_trials'], measured_arrays_read=0)


def _prepare_record(payload):
    from src.data.dual_component_tokenizer import prepare_cached_record
    project_root, cfg, out, rows, retry = payload
    pending = [r for r in rows if task_begin(out, 'prepared', r['id'], retry) is not None]
    if not pending: return dict(status='completed', skipped=len(rows), prepared=0)
    completed = set(); counts = Counter()
    try:
        # Group by native join_key, so EEG and optical files open only once.
        for row, arrays, error in prepare_cached_record(project_root, cfg, pending):
            if row['id'] not in {r['id'] for r in pending} or row['id'] in completed:
                raise ValueError('preparation yielded an unplanned or duplicated row')
            if arrays is None or error is not None:
                task_finish(out, 'prepared', row['id'], 'failed', error=error)
                completed.add(row['id'])
                counts['failed'] += 1
            else:
                write_npz(array_paths(out, 'prepared', row), **arrays)
                task_finish(out, 'prepared', row['id'], 'completed')
                completed.add(row['id'])
                counts['prepared'] += 1
    except Exception:
        error = traceback.format_exc()
        for row in pending:
            if row['id'] not in completed:
                task_finish(out, 'prepared', row['id'], 'failed', error=error)
                completed.add(row['id']); counts['failed'] += 1
    for row in pending:
        if row['id'] not in completed:
            task_finish(out, 'prepared', row['id'], 'failed', error='record producer did not yield planned row')
            counts['failed'] += 1
    return dict(status='completed', record=rows[0]['join_key'], **counts)


def stage_prepare(project_root, cfg, out, workers, retry=False):
    # A standalone/resumed prepare rechecks current events and geometry before
    # the first native signal read, even when plan was completed previously.
    plan(project_root, cfg, out)
    rows = read_json(Path(out) / 'inventory.json')['rows']
    grouped = defaultdict(list)
    for row in rows: grouped[row['join_key']].append(row)
    start = time.monotonic()
    def progress(result, count):
        write_json(Path(out) / 'prepare_progress.json', dict(status='running',
            completed_records=count, planned_records=len(grouped), elapsed_seconds=time.monotonic() - start,
            last=result))
    result = _bounded_jobs(_prepare_record,
        ((str(project_root), cfg, str(out), group, retry) for _, group in sorted(grouped.items())),
        workers, cfg.get('resources', {}).get('max_in_flight_per_worker', 2), progress)
    for record in result:
        if 'payload' in record:
            for row in record['payload'][3]:
                if task_status(out, 'prepared', row['id'])['status'] not in TERMINAL:
                    task_finish(out, 'prepared', row['id'], 'failed', error=record['error'])
    statuses = Counter(task_status(out, 'prepared', r['id'])['status'] for r in rows)
    summary = dict(status='completed', counts=dict(statuses), planned_windows=len(rows),
                   records=len(grouped), elapsed_seconds=time.monotonic() - start)
    if statuses.get('pending') or statuses.get('running'):
        summary['status'] = 'failed'
    write_json(Path(out) / 'prepare_summary.json', summary)
    return summary


def _teacher_task(payload):
    from src.data.dual_component_tokenizer import produce_teacher
    cfg, tak_cfg, out, row, coordinate, retry = payload
    if task_begin(out, 'teacher', row['id'], retry) is None:
        return task_status(out, 'teacher', row['id'])
    start = time.monotonic()
    try:
        arrays = read_arrays(out, 'prepared', row)
        result, numerical = produce_teacher(arrays, coordinate, tak_cfg,
            arm_name=cfg.get('teacher', {}).get('arm', 'tak_fixed'))
        accepted = result is not None and bool(numerical.get('accepted'))
        if result is not None:
            # Even a failed convergence/integration check retains its numerical
            # candidate; pack reads only completed task statuses.
            write_npz(array_paths(out, 'teacher', row), **result)
        return task_finish(out, 'teacher', row['id'], 'completed' if accepted else 'failed',
            numerical=numerical, elapsed_seconds=time.monotonic() - start,
            peak_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    except FileNotFoundError as exc:
        return task_finish(out, 'teacher', row['id'], 'unavailable', error=str(exc),
            elapsed_seconds=time.monotonic() - start)
    except Exception:
        return task_finish(out, 'teacher', row['id'], 'failed', error=traceback.format_exc(),
            elapsed_seconds=time.monotonic() - start)


def stage_teacher(project_root, cfg, out, workers, retry=False):
    from src.data.dual_component_tokenizer import fit_training_coordinates
    rows = read_json(Path(out) / 'inventory.json')['rows']
    statuses = [task_status(out, 'prepared', r['id'])['status'] for r in rows]
    if any(s not in TERMINAL for s in statuses):
        raise ValueError('prepare must have a terminal outcome for every planned window')
    available = [r for r, s in zip(rows, statuses) if s == 'completed']
    path = Path(out) / 'coordinates.json'
    if path.exists(): coordinates = read_json(path)
    else:
        coordinates = fit_training_coordinates(available, lambda row: read_arrays(out, 'prepared', row))
        write_json(path, coordinates)
    teacher_path = Path(cfg['teacher'].get('config', cfg['teacher'].get('contract')))
    if not teacher_path.is_absolute(): teacher_path = SOURCE / teacher_path
    tak_cfg = yaml.safe_load(teacher_path.read_text())
    frozen = Path(out) / 'teacher_config.json'
    if frozen.exists() and read_json(frozen) != serial(tak_cfg):
        raise ValueError('teacher contract changed across resume')
    write_json(frozen, tak_cfg)
    workers = benchmark_cpu_teacher(cfg, tak_cfg, out, rows, coordinates, workers, retry)
    started = time.monotonic()
    def progress(result, count):
        elapsed = time.monotonic() - started
        write_json(Path(out) / 'teacher_progress.json', dict(status='running', completed_tasks=count,
            planned_tasks=len(rows), elapsed_seconds=elapsed,
            estimated_remaining_seconds=(len(rows) - count) * elapsed / max(count, 1), last=result))
    # A missing prepared row still receives its own terminal teacher outcome.
    payloads = ((cfg, tak_cfg, str(out), row, coordinates.get(row['coordinate']), retry) for row in rows)
    results = _bounded_jobs(_teacher_task, payloads, workers,
        cfg.get('resources', {}).get('max_in_flight_per_worker', 2), progress)
    for result in results:
        if 'payload' in result:
            row = result['payload'][3]
            task_finish(out, 'teacher', row['id'], 'failed', error=result['error'])
    counts = Counter(task_status(out, 'teacher', r['id'])['status'] for r in rows)
    summary = dict(status='completed' if all(s in TERMINAL for s in counts) else 'failed',
        counts=dict(counts), planned_windows=len(rows), elapsed_seconds=time.monotonic() - started)
    write_json(Path(out) / 'teacher_summary.json', summary)
    return summary


def benchmark_cpu_teacher(cfg, tak_cfg, out, rows, coordinates, maximum_workers, retry=False):
    """Calibrate on real pending tasks, which the full stage then reuses.

    Different metadata-order samples and process/Numba cold starts make this an
    engineering scheduling estimate. No extra fit or duplicate target is added.
    """
    path = Path(out) / 'cpu_benchmark.json'
    prior = read_json(path) if path.exists() else {}
    scores = list(prior.get('scores', []))
    if prior.get('status') == 'completed':
        eligible = [r for r in scores if r['workers'] <= maximum_workers]
        return max(eligible, key=lambda r: r['terminal_tasks_per_second'])['workers'] if eligible else maximum_workers
    candidates = sorted({min(n, maximum_workers) for n in (8, 16, 32)})
    for workers in candidates:
        if any(r['workers'] == workers for r in scores): continue
        available = [r for r in rows if task_status(out, 'prepared', r['id'])['status'] == 'completed'
            and task_status(out, 'teacher', r['id'])['status'] not in TERMINAL][:workers * 2]
        if not available: break
        started = time.monotonic()
        results = _bounded_jobs(_teacher_task,
            ((cfg, tak_cfg, str(out), row, coordinates[row['coordinate']], retry) for row in available), workers)
        elapsed = time.monotonic() - started
        for result in results:
            if 'payload' in result:
                task_finish(out, 'teacher', result['payload'][3]['id'], 'failed', error=result['error'])
        counts = Counter(task_status(out, 'teacher', r['id'])['status'] for r in available)
        record = dict(workers=workers, attempted_tasks=len(available), counts=dict(counts),
            elapsed_seconds=elapsed, terminal_tasks_per_second=sum(counts[s] for s in TERMINAL) / max(elapsed, 1e-9),
            maximum_worker_peak_resident_bytes=max((r.get('peak_resident_bytes', 0) for r in results), default=0),
            metadata_ids=[r['id'] for r in available])
        scores.append(record)
        write_json(path, dict(status='running', scores=scores, maximum_workers=maximum_workers,
            interpretation='rough engineering estimate; different prespecified metadata-order tasks; includes process/Numba cold starts',
            extra_teacher_fits=0, all_tasks_reused_by_full_stage=True))
    chosen = max(scores, key=lambda r: r['terminal_tasks_per_second'])['workers'] if scores else maximum_workers
    write_json(path, dict(status='completed', scores=scores, chosen_workers=chosen,
        maximum_workers=maximum_workers, available_affinity_cpus=len(os.sched_getaffinity(0)),
        reason='highest observed terminal-task throughput within explicitly bounded CPU budget',
        interpretation='rough engineering estimate; different metadata-order tasks; includes process/Numba cold starts',
        extra_teacher_fits=0, all_tasks_reused_by_full_stage=True))
    return chosen


def stage_pack(cfg, out):
    from src.data.dual_component_tokenizer import pack_training_cache
    rows = read_json(Path(out) / 'inventory.json')['rows']
    if any(task_status(out, 'teacher', r['id'])['status'] not in TERMINAL for r in rows):
        raise ValueError('every planned teacher task requires a retained terminal outcome before packing')
    manifest = Path(out) / 'cache' / 'manifest.json'
    if manifest.exists(): return read_json(manifest)
    return pack_training_cache(cfg, rows, read_json(Path(out) / 'coordinates.json'),
        lambda row: read_arrays(out, 'prepared', row), lambda row: read_arrays(out, 'teacher', row),
        Path(out) / 'cache')


def load_cache(out):
    cache = Path(out) / 'cache'; manifest = read_json(cache / 'manifest.json')
    names = ('eeg_input', 'hb_input', 'eeg_u', 'eeg_s', 'eeg_e', 'hb_u', 'hb_s', 'hb_e')
    arrays = {name: np.load(cache / manifest.get('arrays', {}).get(name, name + '.npy'), mmap_mode='r')
              for name in names}
    if manifest.get('schema') == 'dual_component_refinement_cache_v1':
        for name in ('eeg_spectral_input', 'patch_metadata'):
            if name in manifest.get('arrays', {}):
                arrays[name] = np.load(cache / manifest['arrays'][name], mmap_mode='r', allow_pickle=False)
    if any(len(a) != len(manifest['rows']) for a in arrays.values()):
        raise ValueError('packed arrays and metadata population differ')
    for modality in ('eeg', 'hb'):
        # This test acts at the consumer boundary and does not access raw data.
        for index in range(0, len(manifest['rows']), 256):
            section = slice(index, index + 256)
            if not np.allclose(arrays[modality + '_u'][section],
                arrays[modality + '_s'][section] + arrays[modality + '_e'][section], atol=2e-6, rtol=2e-6):
                raise ValueError('packed teacher sum identity failed')
    return manifest, arrays


def _smooth_arrays(cfg, out, manifest):
    from src.analysis.dual_component_tokenizer import fit_smooth_targets
    cache = Path(out) / 'cache'
    with owner_lock(cache, 'smoothing', blocking=True):
        path = cache / 'smoothing.json'
        result = read_json(path) if path.exists() else fit_smooth_targets(cfg, manifest, cache)
    result = result.get('arrays', result)
    loaded = {}
    for modality in ('eeg', 'hb'):
        name = result.get(modality + '_s', result.get('smooth_s_' + modality))
        if name is None: name = 'smooth_s_' + modality + '.npy'
        path = Path(name)
        if not path.is_absolute(): path = Path(out) / 'cache' / path
        loaded[modality] = np.load(path, mmap_mode='r')
    return loaded


def _alternative_task(payload):
    from src.data.dual_component_tokenizer import produce_alternative_teacher
    cfg, tak_cfg, out, row, coordinate, retry = payload
    stage = 'sensitivity_teacher'
    if task_begin(out, stage, row['id'], retry) is None:
        return task_status(out, stage, row['id'])
    try:
        arrays, reference = read_arrays(out, 'prepared', row), read_arrays(out, 'teacher', row)
        alternative, evidence = produce_alternative_teacher(arrays, coordinate, tak_cfg,
            reference, cfg['sensitivity'])
        accepted = alternative is not None and bool(evidence.get('accepted'))
        if alternative is not None: write_npz(array_paths(out, stage, row), **alternative)
        return task_finish(out, stage, row['id'], 'completed' if accepted else 'unavailable', numerical=evidence)
    except FileNotFoundError as exc:
        return task_finish(out, stage, row['id'], 'unavailable', error=str(exc))
    except Exception:
        return task_finish(out, stage, row['id'], 'failed', error=traceback.format_exc())


def stage_sensitivity(cfg, out, workers, device='cpu', retry=False):
    """Prespecified training identities; no replacement when a witness fails."""
    settings = cfg.get('sensitivity', {})
    if not settings.get('enabled', False): return dict(status='disabled')
    manifest, arrays = load_cache(out)
    inventory = read_json(Path(out) / 'inventory.json')['rows']
    trials = defaultdict(set)
    for row in inventory:
        if row['split'] == 'train': trials[row['dataset']].add(row['trial_id'])
    chosen = {trial for dataset in sorted(trials)
        for trial in sorted(trials[dataset])[:int(settings['training_trials_per_dataset'])]}
    selected = [row for row in inventory if row['trial_id'] in chosen and row['split'] == 'train']
    folder = Path(out) / 'sensitivity'; folder.mkdir(parents=True, exist_ok=True)
    selection = dict(rows=selected, selected_trials=sorted(chosen),
        selection='lexicographic training trial identities before array availability; no replacement')
    selection_path = folder / 'selection.json'
    if selection_path.exists() and read_json(selection_path) != serial(selection):
        raise ValueError('sensitivity selection changed across resume')
    write_json(selection_path, selection)
    coordinates = read_json(Path(out) / 'coordinates.json')
    tak_cfg = read_json(Path(out) / 'teacher_config.json')
    results = _bounded_jobs(_alternative_task,
        ((cfg, tak_cfg, str(out), row, coordinates.get(row['coordinate']), retry) for row in selected), workers)
    for result in results:
        if 'payload' in result:
            task_finish(out, 'sensitivity_teacher', result['payload'][3]['id'], 'failed', error=result['error'])
    grouped = defaultdict(list)
    for row in selected: grouped[row['trial_id']].append(row)
    valid_trials = {trial for trial, group in grouped.items() if len(group) == 3
        and all(task_status(out, 'sensitivity_teacher', r['id'])['status'] == 'completed' for r in group)}
    index_by_id = {row['id']: i for i, row in enumerate(manifest['rows'])}
    rows = [r for r in selected if r['trial_id'] in valid_trials and r['id'] in index_by_id]
    failed_trials = sorted(chosen - valid_trials)
    paired_manifest = dict(manifest, rows=rows, planned_trials=len(chosen), retained_trials=len(valid_trials),
        planned_windows=len(selected), retained_windows=len(rows), failed_trials=failed_trials,
        sensitivity='same-model feasible whole-trajectory restart; train-only small paired vocabulary diagnostic')
    write_json(folder / 'manifest.json', paired_manifest)
    if not rows:
        result = dict(status='unavailable', planned_trials=len(chosen), retained_trials=0,
            failed_trials=failed_trials, reason='no complete three-region legal alternative witness; no replacement')
        write_json(folder / 'summary.json', result); return result
    indices = [index_by_id[r['id']] for r in rows]
    paired_cfg = deepcopy(cfg); paired_cfg['training']['steps'] = int(settings['paired_steps'])
    seed = int(settings.get('seed', 17)); variant = settings.get('variant', 'hard_student')
    summaries = {}
    for arm in ('reference', 'alternative'):
        arm_root = folder / arm; cache = arm_root / 'cache'; cache.mkdir(parents=True, exist_ok=True)
        for name, value in arrays.items():
            data = np.asarray(value[indices])
            if arm == 'alternative' and name not in ('eeg_input', 'hb_input'):
                modality, component = name.split('_')
                data = []
                for row in rows:
                    teacher = read_arrays(out, 'sensitivity_teacher', row)
                    scale = np.asarray(manifest['observation_training_sd'][row['coordinate']])
                    source = teacher[dict(u='u', s='physical', e='remaining')[component]][300:500] / scale
                    source = source[:, :1] if modality == 'eeg' else source[:, 1:]
                    data.append(source.reshape(10, -1))
                data = np.asarray(data, dtype='float32')
            target = cache / (name + '.npy')
            if target.exists():
                if not np.array_equal(np.load(target, mmap_mode='r'), data):
                    raise ValueError('paired sensitivity cache changed across resume')
            else:
                temporary = target.with_name(target.stem + '.tmp.npy')
                np.save(temporary, data); temporary.replace(target)
        write_json(cache / 'manifest.json', paired_manifest)
        summaries[arm] = train_one(paired_cfg, arm_root, variant, seed, device,
            job_output=arm_root / f'seed_{seed}')
    result = dict(status='completed', planned_trials=len(chosen), retained_trials=len(valid_trials),
        failed_trials=failed_trials, arms=summaries, seed=seed, steps=settings['paired_steps'],
        interpretation='small training-identity sensitivity diagnostic; no independent-subject evaluation')
    write_json(folder / 'summary.json', result)
    return result


def _batch(arrays, indices, device, physical=None):
    import torch
    inputs, observed, components, masks = {}, {}, {}, {}
    for modality in ('eeg', 'hb'):
        def tensor(array): return torch.as_tensor(np.asarray(array[indices]).copy(), device=device)
        inputs[modality] = tensor(arrays[modality + '_input'])
        observed[modality] = tensor(arrays[modality + '_u'])
        components[modality] = tensor(arrays[modality + '_s'] if physical is None else physical[modality])
        masks[modality] = torch.isfinite(inputs[modality])
    return inputs, observed, components, masks


def _forward(model, inputs, observed, physical, masks):
    if model.variant == 'component_compression':
        return model.forward_components(physical, {m: observed[m] - physical[m] for m in physical})
    return model(inputs, masks)


def _torch_save(path, state):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(state, temporary); temporary.replace(path)


def _checkpoint(model, optimizer, step, cfg, variant, seed, history):
    import torch
    device = next(model.parameters()).device
    return dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), step=step,
        config=cfg, variant=variant, seed=seed, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
        cuda_rng_device=str(device) if device.type == 'cuda' else None,
        history=history)


def train_one(cfg, out, variant, seed, device='cpu', *, steps=None, job_output=None, export=True,
              gpu_memory_bytes=None):
    """A fixed-budget model job with exact model/optimizer/sampler RNG resume.

    ``steps`` overrides are restricted by the CLI to a separate pilot output.
    Calling this function with a lower budget is useful for interruption tests.
    """
    import torch
    from src.tokenizers.dual_component_tokenizer import DualComponentTokenizer
    if variant not in VARIANTS: raise ValueError('unknown tokenizer variant')
    job = Path(job_output) if job_output else Path(out) / 'jobs' / variant / f'seed_{seed}'
    job.mkdir(parents=True, exist_ok=True)
    with owner_lock(job, 'training'):
        manifest, arrays = load_cache(out)
        train_indices = np.asarray([i for i, row in enumerate(manifest['rows']) if row['split'] == 'train'])
        if not len(train_indices): raise ValueError('training-subject examples required')
        datasets = sorted({manifest['rows'][i]['dataset'] for i in train_indices})
        by_dataset = [np.asarray([i for i in train_indices if manifest['rows'][i]['dataset'] == dataset])
                      for dataset in datasets]
        training = cfg['training']; training_steps = int(training['steps'] if steps is None else steps)
        torch.set_num_threads(int(cfg.get('resources', {}).get('torch_threads', 1)))
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if device.startswith('cuda'):
            torch.cuda.set_device(device)
            torch.cuda.manual_seed(seed)
            memory_budget_bytes = gpu_memory_bytes if gpu_memory_bytes is not None else int(
                cfg.get('resources', {}).get('max_new_gpu_memory_mib', 4096)) * 1024 ** 2
            # Reserve room for the CUDA context/runtime outside the allocator.
            allocator_budget_bytes = max(memory_budget_bytes - 512 * 1024 ** 2, 64 * 1024 ** 2)
            total_memory = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(allocator_budget_bytes / total_memory, 1.), device)
            torch.cuda.reset_peak_memory_stats(device)
        model = DualComponentTokenizer(cfg['model'], variant).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
            weight_decay=float(training.get('weight_decay', 0)))
        checkpoint_path = job / 'checkpoint.pt'; step = 0; history = []
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if state['config'] != cfg or state['variant'] != variant or state['seed'] != seed:
                raise ValueError('training checkpoint contract drift')
            model.load_state_dict(state['model_state']); optimizer.load_state_dict(state['optimizer_state'])
            step, history = state['step'], state.get('history', [])
            random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'].cpu())
            if state.get('cuda_rng') is not None and device.startswith('cuda'):
                torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)
        if step > training_steps: raise ValueError('checkpoint is beyond requested fixed step budget')
        physical_arrays = _smooth_arrays(cfg, out, manifest) if variant == 'smooth_hard' else None
        started = time.monotonic(); first_step = step; last = {}
        model.train()
        while step < training_steps:
            selected_datasets = np.random.randint(len(by_dataset), size=int(training['batch_size']))
            indices = np.asarray([np.random.choice(by_dataset[index]) for index in selected_datasets])
            inputs, observed, physical, masks = _batch(arrays, indices, device, physical_arrays)
            output = _forward(model, inputs, observed, physical, masks)
            terms = model.objective(output, observed, physical,
                lambda_O=float(training.get('lambda_O', 1)), beta=float(training.get('beta', 1)))
            loss = terms['total']
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite training objective')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get('gradient_clip', 5)))
            if not torch.isfinite(norm): raise FloatingPointError('nonfinite training gradient')
            optimizer.step(); step += 1
            last = {name: float(terms[name].detach()) for name in ('total', 'S', 'O', 'sum', 'vq')}
            if step % int(training['checkpoint_every']) == 0 or step == training_steps:
                elapsed = time.monotonic() - started
                if device.startswith('cuda'): torch.cuda.synchronize(device)
                rate = (step - first_step) / max(elapsed, 1e-9)
                progress = dict(status='running', variant=variant, seed=seed, step=step, budget=training_steps,
                    elapsed_seconds=elapsed, steps_per_second=rate,
                    examples_per_second=rate * training['batch_size'],
                    estimated_remaining_seconds=(training_steps - step) / max(rate, 1e-9), loss=last,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else 0,
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.startswith('cuda') else 0)
                history.append(dict(step=step, **last))
                _torch_save(checkpoint_path, _checkpoint(model, optimizer, step, cfg, variant, seed, history))
                write_json(job / 'progress.json', progress)
        if device.startswith('cuda'): torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        metrics = export_predictions(cfg, out, job, model, manifest, arrays, device, physical_arrays) if export else {}
        result = dict(status='completed', variant=variant, seed=seed, step=step, budget=training_steps,
            train_examples=len(train_indices), total_examples=len(manifest['rows']),
            elapsed_seconds=elapsed, trained_steps=step - first_step,
            steps_per_second=(step - first_step) / max(elapsed, 1e-9),
            examples_per_second=(step - first_step) * training['batch_size'] / max(elapsed, 1e-9),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.startswith('cuda') else 0,
            final_loss=last or (history[-1] if history else {}), metrics=metrics,
            target_coordinates='observation divided by frozen training SD; S and O share the same scale',
            hard_ids_exported=variant != 'continuous_student')
        write_json(job / 'summary.json', result)
        return result


def export_predictions(cfg, out, job, model, manifest, arrays, device, physical_arrays=None):
    import torch
    count = len(manifest['rows']); batch_size = cfg['training'].get('eval_batch_size', 128)
    exported = dict(row_index=np.arange(count, dtype=np.int64))
    for modality in ('eeg', 'hb'):
        for name in ('S', 'O', 'sum'):
            exported[modality + '_' + name] = np.empty_like(arrays[modality + '_u'])
        if model.hard:
            for name in ('ids_S', 'ids_O'):
                exported[modality + '_' + name] = np.empty(arrays[modality + '_u'].shape[:-1], dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            indices = np.arange(start, min(start + batch_size, count))
            inputs, observed, physical, masks = _batch(arrays, indices, device, physical_arrays)
            output = _forward(model, inputs, observed, physical, masks)
            for modality in ('eeg', 'hb'):
                for name in ('S', 'O', 'sum') + (('ids_S', 'ids_O') if model.hard else ()):
                    exported[modality + '_' + name][indices] = output[modality][name].cpu().numpy()
        if model.hard:
            codewords = {m + '_' + c: v.cpu().numpy()
                for m, components in model.decoded_codewords().items() for c, v in components.items()}
            write_npz(Path(job) / 'codewords.npz', **codewords)
    write_npz(Path(job) / 'predictions.npz', **exported)
    metrics = {}
    for split in sorted({r['split'] for r in manifest['rows']}):
        indices = [i for i, row in enumerate(manifest['rows']) if row['split'] == split]
        metrics[split] = dict(examples=len(indices), modalities={})
        for modality in ('eeg', 'hb'):
            u, s = arrays[modality + '_u'][indices], arrays[modality + '_s'][indices]
            if physical_arrays is not None: s = np.asarray(physical_arrays[modality][indices])
            truth = dict(S=s, O=u - s, sum=u)
            metrics[split]['modalities'][modality] = {
                name: float(np.mean((exported[modality + '_' + name][indices] - target) ** 2))
                for name, target in truth.items()}
    return metrics


def _spawn_job(cfg_path, out, project_root, variant, seed, device, parent_pid, *, pilot=None, steps=None,
               gpu_memory_bytes=None):
    command = [sys.executable, str(Path(__file__).resolve()), '--config', str(cfg_path),
        '--output', str(out), '--project-root', str(project_root), '--stage', 'train-one',
        '--variant', variant, '--seed', str(seed), '--device', device, '--parent-pid', str(parent_pid)]
    job = Path(pilot) if pilot else Path(out) / 'jobs' / variant / f'seed_{seed}'
    job.mkdir(parents=True, exist_ok=True)
    if pilot: command += ['--pilot-output', str(pilot), '--steps', str(steps), '--no-export']
    if gpu_memory_bytes: command += ['--gpu-memory-bytes', str(gpu_memory_bytes)]
    handle = (job / 'process.log').open('a')
    process = subprocess.Popen(command, cwd=SOURCE, stdout=handle, stderr=subprocess.STDOUT,
        env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8'))
    return process, handle, job


def train_suite(cfg, cfg_path, out, project_root, devices, gpu_workers, retry=False):
    """Bound independent jobs per device; subprocess exit is never success evidence."""
    _smooth_arrays(cfg, out, load_cache(out)[0])
    pending = [(v, s) for v in VARIANTS for s in seeds(cfg)]
    slots = [device for device in devices for _ in range(gpu_workers)]
    running = []; results = []
    while pending or running:
        while pending and len(running) < len(slots):
            variant, seed = pending.pop(0); key = [variant, seed]
            if task_begin(out, 'training', key, retry) is None:
                results.append(task_status(out, 'training', key)); continue
            used = Counter(item[3] for item in running)
            device = next(d for d in slots if used[d] < slots.count(d))
            process, handle, job = _spawn_job(cfg_path, out, project_root, variant, seed, device, os.getpid(),
                gpu_memory_bytes=int(cfg.get('resources', {}).get('max_new_gpu_memory_mib', 4096)) * 1024 ** 2 // slots.count(device))
            running.append((process, handle, key, device, job))
        if not running: continue
        time.sleep(.2)
        for item in list(running):
            process, handle, key, device, job = item
            code = process.poll()
            if code is None: continue
            handle.close(); running.remove(item)
            summary = read_json(job / 'summary.json') if (job / 'summary.json').exists() else {}
            success = code == 0 and summary.get('status') == 'completed' and summary.get('step') == cfg['training']['steps']
            results.append(task_finish(out, 'training', key, 'completed' if success else 'failed',
                returncode=code, device=device, summary=summary,
                log=str(job / 'process.log')))
            write_json(Path(out) / 'training_progress.json', dict(status='running',
                completed_jobs=len(results), total_jobs=len(VARIANTS) * len(seeds(cfg)),
                active=[dict(key=x[2], device=x[3], pid=x[0].pid) for x in running]))
    summary = dict(status='completed' if all(x['status'] == 'completed' for x in results) else 'failed',
        jobs=results, planned_jobs=len(VARIANTS) * len(seeds(cfg)))
    write_json(Path(out) / 'training_summary.json', summary)
    return summary


def benchmark(cfg, cfg_path, out, project_root, device, steps=30):
    """Measure one and two concurrent workers on identical fresh pilot jobs."""
    def completed_at_requested_steps(row):
        return (row.get('status') == 'completed' and row.get('returncode') == 0
                and row.get('step') == steps and row.get('budget') == steps)

    scores = []
    for concurrency in (1, 2):
        folder = Path(out) / 'benchmark' / f'workers_{concurrency}'
        if (folder / 'summary.json').exists():
            retained = read_json(folder / 'summary.json')
            if (retained.get('steps_per_worker') != steps
                    or not all(completed_at_requested_steps(row) for row in retained.get('workers', []))):
                raise ValueError('retained benchmark violates the requested step budget; use a fresh benchmark')
            scores.append(retained); continue
        started = time.monotonic(); processes = []
        for worker in range(concurrency):
            job = folder / f'worker_{worker}'
            processes.append(_spawn_job(cfg_path, out, project_root, 'hard_student', 1700 + worker,
                device, os.getpid(), pilot=job, steps=steps,
                gpu_memory_bytes=int(cfg.get('resources', {}).get('max_new_gpu_memory_mib', 4096)) * 1024 ** 2 // concurrency))
        rows = []
        for process, handle, job in processes:
            code = process.wait(); handle.close()
            row = read_json(job / 'summary.json') if (job / 'summary.json').exists() else {}
            rows.append(dict(returncode=code, job=job, **row))
        elapsed = time.monotonic() - started
        result = dict(concurrency=concurrency, device=device, steps_per_worker=steps,
            elapsed_seconds=elapsed, workers=rows,
            aggregate_examples_per_second=sum(r.get('trained_steps', 0) for r in rows) * cfg['training']['batch_size'] / elapsed,
            aggregate_peak_reserved_bytes=sum(r.get('peak_cuda_reserved_bytes', 0) for r in rows),
            status='completed' if all(completed_at_requested_steps(r) for r in rows) else 'failed',
            includes='process startup, cache checks and training; no prediction export')
        write_json(folder / 'summary.json', result); scores.append(result)
    successful = [r for r in scores if r['status'] == 'completed'
        and r['aggregate_peak_reserved_bytes'] <= cfg.get('resources', {}).get('gpu_memory_budget_bytes',
            cfg.get('resources', {}).get('max_new_gpu_memory_mib', 4096) * 1024 ** 2)]
    best = max(successful, key=lambda r: r['aggregate_examples_per_second']) if successful else None
    result = dict(status='completed' if best else 'failed', scores=scores,
        recommended_gpu_workers_per_device=None if best is None else best['concurrency'],
        memory_measure='sum of worker peak CUDA reserved memory; concurrent bound, not exact simultaneous peak')
    write_json(Path(out) / 'benchmark.json', result)
    return result


def masked_one(cfg, out, route, seed, device, gpu_memory_bytes=None):
    from src.analysis.dual_component_tokenizer import run_masked_pilot
    job = Path(out) / 'masked' / route / f'seed_{seed}'
    with owner_lock(job, 'masked'):
        effective = deepcopy(cfg)
        total = gpu_memory_bytes or int(cfg.get('resources', {}).get('max_new_gpu_memory_mib', 4096)) * 1024 ** 2
        effective.setdefault('resources', {})['max_new_gpu_memory_mib'] = max(total / 1024 ** 2 - 512, 64)
        return run_masked_pilot(effective, out, device=device, seed=seed, route=route)


def masked_suite(cfg, cfg_path, out, project_root, devices, gpu_workers, retry=False):
    if not cfg.get('masked', {}).get('enabled', False): return dict(status='disabled')
    settings = cfg['masked']
    pending = [(route, seed) for route in settings.get('routes', ('hard_student', 'smooth_hard', 'ordinary_hard'))
               for seed in settings.get('seeds', seeds(cfg))]
    slots = [device for device in devices for _ in range(gpu_workers)]
    running = []; results = []; planned = len(pending)
    while pending or running:
        while pending and len(running) < len(slots):
            route, seed = pending.pop(0); key = [route, seed]
            if task_begin(out, 'masked', key, retry) is None:
                results.append(task_status(out, 'masked', key)); continue
            used = Counter(item[3] for item in running)
            device = next(d for d in slots if used[d] < slots.count(d))
            job = Path(out) / 'masked' / route / f'seed_{seed}'; job.mkdir(parents=True, exist_ok=True)
            memory = int(cfg.get('resources', {}).get('max_new_gpu_memory_mib', 4096)) * 1024 ** 2 // slots.count(device)
            command = [sys.executable, str(Path(__file__).resolve()), '--config', str(cfg_path),
                '--output', str(out), '--project-root', str(project_root), '--stage', 'masked-one',
                '--variant', route, '--seed', str(seed), '--device', device,
                '--parent-pid', str(os.getpid()), '--gpu-memory-bytes', str(memory)]
            handle = (job / 'process.log').open('a')
            process = subprocess.Popen(command, cwd=SOURCE, stdout=handle, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8'))
            running.append((process, handle, key, device, job))
        if not running: continue
        time.sleep(.2)
        for item in list(running):
            process, handle, key, device, job = item
            code = process.poll()
            if code is None: continue
            handle.close(); running.remove(item)
            summary = read_json(job / 'summary.json') if (job / 'summary.json').exists() else {}
            success = code == 0 and summary.get('status') == 'completed'
            results.append(task_finish(out, 'masked', key, 'completed' if success else 'failed',
                returncode=code, device=device, summary=summary, log=str(job / 'process.log')))
            write_json(Path(out) / 'masked_progress.json', dict(status='running', completed_jobs=len(results),
                planned_jobs=planned, active=[dict(key=x[2], device=x[3], pid=x[0].pid) for x in running]))
    result = dict(status='completed' if all(r['status'] == 'completed' for r in results) else 'failed',
        planned_jobs=planned, jobs=results)
    write_json(Path(out) / 'masked_summary.json', result)
    return result


def synthetic(cfg, *, numerical_teacher=True):
    """No filesystem data reads: verify all branches and checkpoint serialization."""
    import torch
    from src.tokenizers.dual_component_tokenizer import DualComponentTokenizer
    torch.set_num_threads(1); torch.manual_seed(17)
    model_cfg = dict(cfg['model'], width=8, latent_dim=4, codebook_size=4, kmeans_iters=2)
    inputs = {m: torch.randn(2, 10, d) for m, d in model_cfg['input_dims'].items()}
    observed = {m: torch.randn(2, 10, d) for m, d in model_cfg['target_dims'].items()}
    physical = {m: value * .6 for m, value in observed.items()}
    results = []
    for variant in VARIANTS:
        model = DualComponentTokenizer(model_cfg, variant)
        output = _forward(model, inputs, observed, physical, {m: torch.ones_like(x, dtype=torch.bool) for m, x in inputs.items()})
        terms = model.objective(output, observed, physical); terms['total'].backward()
        for modality in observed:
            if not torch.equal(output[modality]['sum'], output[modality]['S'] + output[modality]['O']):
                raise AssertionError('student sum identity failed')
        if model.hard:
            model.eval()
            with torch.no_grad():
                output = _forward(model, inputs, observed, physical,
                    {m: torch.ones_like(x, dtype=torch.bool) for m, x in inputs.items()})
                decoded = model.decode_ids({m: {c: output[m]['ids_' + c] for c in ('S', 'O')} for m in observed})
                if any(not torch.allclose(decoded[m]['sum'], output[m]['sum'], atol=1e-6) for m in observed):
                    raise AssertionError('integer-only decoding differs from hard forward')
        results.append(dict(variant=variant, loss=float(terms['total'].detach()), hard=model.hard))
    result = dict(status='completed', variants=results, measured_arrays_read=0)
    if numerical_teacher:
        from src.data.dual_component_tokenizer import synthetic_teacher_check
        path = Path(cfg['teacher'].get('config', cfg['teacher'].get('contract')))
        if not path.is_absolute(): path = SOURCE / path
        result['teacher'] = synthetic_teacher_check(yaml.safe_load(path.read_text()))
        if result['teacher']['status'] != 'completed':
            raise ValueError('synthetic Tak teacher/integration verification failed')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--project-root', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage', choices=STAGES, default='check')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--devices', nargs='+')
    parser.add_argument('--gpu-workers', type=int)
    parser.add_argument('--variant', choices=VARIANTS, default='hard_student')
    parser.add_argument('--candidate', help='named candidate from the refinement contract')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--pilot-output', type=Path)
    parser.add_argument('--no-export', action='store_true')
    parser.add_argument('--parent-pid', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--gpu-memory-bytes', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--evaluation-split', choices=('development','test'),default='development',help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv); cfg = read_config(args.config)
    if cfg['schema']=='dual_component_voltage_v1' and args.stage not in ('check','synthetic','voltage','voltage-one'):
        raise ValueError('voltage targets require their own versioned execution stages')
    if cfg['schema'] == 'dual_component_refinement_v1' and args.stage not in ('check', 'refine', 'refine-one'):
        raise ValueError('refinement uses its finite train/validation-only stages')
    if args.stage == 'check':
        if cfg['schema']=='dual_component_voltage_v1':
            print(json.dumps(dict(status='voltage_contract_valid',experiment_id=cfg['experiment_id'],
                stages=['check','synthetic','voltage','voltage-one'],candidates=cfg['voltage']['candidates'],
                selection_splits=cfg['voltage']['selection_splits'],final_split=cfg['voltage']['final_split'],
                ssms_refitted=0,protected_data='forbidden',measured_arrays_read=0)));return
        print(json.dumps(dict(status='software_contract_valid', stages=STAGES, variants=VARIANTS,
            seeds=seeds(cfg), protected_data='forbidden', measured_arrays_read=0))); return
    if args.stage == 'synthetic':
        result = voltage_synthetic(cfg) if cfg['schema']=='dual_component_voltage_v1' else synthetic(cfg)
        if args.output: write_json(args.output / 'synthetic.json', result)
        print(json.dumps(serial(result))); return
    if args.output is None: raise ValueError('--output required for a retained run')
    out, root = args.output.resolve(), args.project_root.resolve()
    if args.steps is not None and (args.stage not in ('train-one','voltage-one') or args.pilot_output is None):
        raise ValueError('step overrides require train-one and a separate --pilot-output')
    if args.pilot_output and args.pilot_output.resolve() == out:
        raise ValueError('pilot output must differ from the formal run root')
    workers = args.workers or int(cfg.get('resources', {}).get('cpu_workers', 1))
    if workers < 1 or workers > len(os.sched_getaffinity(0)):
        raise ValueError('CPU workers must fit the available CPU affinity')
    devices = args.devices or cfg.get('resources', {}).get('devices', ['cpu'])
    gpu_workers = args.gpu_workers or int(cfg.get('resources', {}).get('gpu_workers_per_device', 1))
    if gpu_workers not in (1, 2): raise ValueError('one or two measured GPU workers per device required')
    if args.parent_pid is not None:
        if args.stage not in ('train-one', 'masked-one', 'refine-one','voltage-one') or os.getppid() != args.parent_pid:
            raise ValueError('worker must be a direct child of its live controller')
        freeze_contract(out, cfg, root)
        if args.stage=='voltage-one':
            result=(voltage_final(cfg,out,args.candidate,args.seed,args.device,args.gpu_memory_bytes)
                if args.evaluation_split=='test' else voltage_train(cfg,out,args.candidate,args.seed,args.device,args.gpu_memory_bytes,
                    steps=args.steps,job_override=args.pilot_output,export=not args.no_export))
        elif args.stage == 'refine-one':
            result = refine_one(cfg, out, args.candidate, args.seed, args.device, args.gpu_memory_bytes)
        elif args.stage == 'masked-one':
            result = masked_one(cfg, out, args.variant, args.seed, args.device, args.gpu_memory_bytes)
        else:
            result = train_one(cfg, out, args.variant, args.seed, args.device,
                steps=args.steps, job_output=args.pilot_output, export=not args.no_export,
                gpu_memory_bytes=args.gpu_memory_bytes)
        print(json.dumps(serial(result))); return
    with owner_lock(out):
        freeze_contract(out, cfg, root)
        selected = ('synthetic', 'plan', 'prepare', 'teacher', 'pack', 'benchmark', 'train',
                    'sensitivity', 'analyze', 'masked') if args.stage == 'all' else (args.stage,)
        result = None
        for stage in selected:
            write_json(out / 'controller.json', dict(status='running', stage=stage, pid=os.getpid(), updated=now(),
                command=sys.argv, workers=workers, devices=devices, gpu_workers=gpu_workers))
            if stage=='voltage': result=run_voltage(cfg,args.config.resolve(),out,root,devices,gpu_workers,workers)
            elif stage=='voltage-one':result=voltage_train(cfg,out,args.candidate,args.seed,args.device,args.gpu_memory_bytes)
            elif stage == 'synthetic':
                path = out / 'synthetic.json'
                result = read_json(path) if path.exists() else synthetic(cfg)
                write_json(path, result)
            elif stage == 'refine': result = run_refinement(cfg, args.config.resolve(), out, root, devices, gpu_workers)
            elif stage == 'refine-one': result = refine_one(cfg, out, args.candidate, args.seed, args.device, args.gpu_memory_bytes)
            elif stage == 'plan': result = plan(root, cfg, out)
            elif stage == 'prepare': result = stage_prepare(root, cfg, out, workers, args.retry_failed)
            elif stage == 'teacher': result = stage_teacher(root, cfg, out, workers, args.retry_failed)
            elif stage == 'pack': result = stage_pack(cfg, out)
            elif stage == 'benchmark':
                result = benchmark(cfg, args.config.resolve(), out, root, devices[0], int(cfg.get('resources', {}).get('benchmark_steps', 30)))
                if args.stage == 'all' and result.get('recommended_gpu_workers_per_device'):
                    gpu_workers = result['recommended_gpu_workers_per_device']
            elif stage == 'train': result = train_suite(cfg, args.config.resolve(), out, root, devices, gpu_workers, args.retry_failed)
            elif stage == 'train-one': result = train_one(cfg, out, args.variant, args.seed, args.device,
                steps=args.steps, job_output=args.pilot_output, export=not args.no_export,
                gpu_memory_bytes=args.gpu_memory_bytes)
            elif stage == 'masked-one': result = masked_one(cfg, out, args.variant, args.seed,
                args.device, args.gpu_memory_bytes)
            elif stage == 'sensitivity': result = stage_sensitivity(cfg, out, workers, devices[0], args.retry_failed)
            elif stage == 'analyze':
                from src.analysis.dual_component_tokenizer import analyze_suite
                result = analyze_suite(cfg, out)
            elif stage == 'masked':
                result = masked_suite(cfg, args.config.resolve(), out, root, devices, gpu_workers, args.retry_failed)
                if args.stage == 'all':
                    from src.analysis.dual_component_tokenizer import analyze_suite
                    result = analyze_suite(cfg, out)
            if isinstance(result, dict) and result.get('status') == 'failed':
                raise RuntimeError(f'{stage} failed; retained task outcomes must be reviewed')
        write_json(out / 'controller.json', dict(status='completed', stage=args.stage, pid=os.getpid(), updated=now()))
        print(json.dumps(serial(result)))


def prepare_refinement(cfg, out, root):
    """Reuse only parent train/validation rows after exact public-lineage checks."""
    settings = cfg['refinement']; parent = (Path(root) / settings['parent_run']).resolve()
    admitted = (Path(root) / 'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261028_v3').resolve()
    if parent != admitted or settings['allowed_splits'] != ['train', 'validation']:
        raise ValueError('exact completed public parent and train/validation-only refinement required')
    contract = read_json(parent / 'resolved_config.json')['config']
    if contract['data']['protected_data'] != 'forbidden' or read_json(parent / 'controller.json')['status'] != 'completed':
        raise ValueError('completed public parent required before array reads')
    destination = Path(out) / 'cache'; destination.mkdir(exist_ok=True)
    if (destination / 'manifest.json').exists(): return read_json(destination / 'manifest.json')
    source = parent / 'cache'; manifest = read_json(source / 'manifest.json')
    if manifest['schema'] != 'dual_component_training_cache_v1' or manifest['target_units'] != 'training_observation_SD':
        raise ValueError('unaltered full Tak component/residual training coordinates required')
    selected = np.array([i for i, row in enumerate(manifest['rows']) if row['split'] in settings['allowed_splits']])
    rows = [manifest['rows'][i] for i in selected]
    planned_rows = [r for r in read_json(parent / 'inventory.json')['rows'] if r['split'] in settings['allowed_splits']]
    relevant_trials = {r['trial_id'] for r in planned_rows}
    for name, shape in manifest['shapes'].items():
        if name not in ('eeg_input', 'hb_input', 'eeg_u', 'eeg_s', 'eeg_e', 'hb_u', 'hb_s', 'hb_e'):
            raise ValueError('unregistered refinement cache field')
        values = np.load(source / manifest['arrays'][name], mmap_mode='r', allow_pickle=False)
        target = np.lib.format.open_memmap(destination / (name + '.tmp.npy'), mode='w+',
            shape=(len(rows), *shape[1:]), dtype='float32')
        for start in range(0, len(rows), 256): target[start:start + 256] = values[selected[start:start + 256]]
        target.flush(); del target
        (destination / (name + '.tmp.npy')).replace(destination / (name + '.npy'))
    from src.data.functional_tokenizer import identity_digest
    result = dict(manifest, rows=rows, shapes={k:[len(rows), *v[1:]] for k, v in manifest['shapes'].items()},
        schema='dual_component_refinement_cache_v1', input_parent=str(parent), parent_indices=selected.tolist(),
        selection_splits=settings['allowed_splits'], test_arrays_selected=False,
        parent_retained_windows=manifest['retained_windows'], retained_windows=len(rows),
        retained_trials=len(rows) // 3, planned_windows=len(planned_rows), planned_trials=len(relevant_trials),
        failed_trials=[x for x in manifest['failed_trials'] if x in relevant_trials],
        identity=identity_digest(rows), source_identity=manifest['identity'])
    result['independent_subjects'] = {ds:{split:len({r['subject'] for r in rows if r['dataset']==ds and r['split']==split})
        for split in ('train','validation')} for ds in sorted({r['dataset'] for r in rows})}
    if any(c['input_mode'] == 'spectral' for c in settings['candidates']):
        from src.data.ssm_prepared import EEG_PROXY_INPUT_SCHEMA
        values = np.lib.format.open_memmap(destination / 'eeg_spectral_input.npy', mode='w+', dtype='float32', shape=(len(rows),10,620))
        u = np.load(destination / 'eeg_u.npy', mmap_mode='r')
        coordinates = sorted({r['coordinate'] for r in rows})
        metadata = np.lib.format.open_memmap(destination / 'patch_metadata.npy', mode='w+', dtype='float32',
            shape=(len(rows),10,20*(len(coordinates)+1)))
        for i, row in enumerate(rows):
            item = read_arrays(parent, 'prepared', row, keys=('features','identity','eeg_proxy_input','eeg_proxy_rate_hz'))
            if str(item['identity']) != row['id'] or str(item['eeg_proxy_input']) != EEG_PROXY_INPUT_SCHEMA or float(item['eeg_proxy_rate_hz']) != 10.:
                raise ValueError('exact native-channel 10Hz EEG feature lineage required')
            features = np.asarray(item['features'][300:500], dtype='float32')
            if features.shape != (200,30): raise ValueError('six channels by five bands required')
            values[i] = np.column_stack((u[i].reshape(200),features)).reshape(10,620)
            meta = np.zeros((200,len(coordinates)+1), dtype='float32')
            meta[:,coordinates.index(row['coordinate'])] = 1.
            meta[:,-1] = np.arange(200)/199. - .5
            metadata[i] = meta.reshape(10,-1)
        fitted = {}
        for coordinate in coordinates:
            training = np.array([r['coordinate']==coordinate and r['split']=='train' for r in rows])
            all_rows = np.flatnonzero([r['coordinate']==coordinate for r in rows])
            band = np.asarray(values[training]).reshape(-1,31)[:,1:]
            mean = band.mean(0); scale = np.maximum(band.std(0),1e-5)
            fitted[coordinate] = dict(mean=mean,scale=scale,training_rows=int(training.sum()))
            for start in range(0,len(all_rows),128):
                ix=all_rows[start:start+128];x=np.asarray(values[ix]).reshape(-1,20,31)
                x[...,1:] = (x[...,1:]-mean)/scale
                values[ix] = x.reshape(len(ix),10,620)
        values.flush();metadata.flush();del values,metadata
        result['arrays'].update(eeg_spectral_input='eeg_spectral_input.npy',patch_metadata='patch_metadata.npy')
        result['shapes'].update(eeg_spectral_input=[len(rows),10,620],patch_metadata=[len(rows),10,20*(len(coordinates)+1)])
        result['spectral_frontend'] = dict(schema='dual_component_spectral_frontend_v1',
            channels='own EEG proxy plus original six EEG channels x five log-power bands',
            fitting=fitted, metadata_coordinates=coordinates,
            metadata='dataset/region and position within existing 20s window; no task, condition or subject ID',
            source='registered 10Hz prepared EEG features only; no Hb values or teacher outputs read for EEG input')
    write_json(destination / 'manifest.json', result)
    return result


def refinement_inputs(arrays, indices, candidate, device):
    import torch
    from src.tokenizers.dual_component_tokenizer import local_patch_context
    inputs, masks = {}, {}
    for modality in ('eeg', 'hb'):
        key = ('eeg_spectral_input' if modality == 'eeg' and candidate['input_mode'] == 'spectral'
            else modality + ('_input' if candidate['input_mode'] == 'native' else '_u'))
        value = torch.as_tensor(np.asarray(arrays[key][indices]).copy(), device=device)
        if candidate.get('include_metadata'):
            meta = torch.as_tensor(np.asarray(arrays['patch_metadata'][indices]).copy(), device=device)
            value = torch.cat((value.reshape(*value.shape[:2],20,-1),meta.reshape(*meta.shape[:2],20,-1)), -1).flatten(-2)
        inputs[modality], masks[modality] = local_patch_context(value, candidate['context_radius'])
    return inputs, masks


def refinement_predictions(model, arrays, candidate, device, batch_size=256):
    import torch
    total = len(arrays['eeg_u']); predictions = {}; hard_ids = {}
    for modality in ('eeg', 'hb'):
        for component in ('S', 'O', 'sum'):
            predictions[modality + '_' + component] = np.empty_like(arrays[modality + '_u'])
        if model.hard:
            for component in ('S', 'O'):
                hard_ids[modality + '_' + component] = np.empty((total, 10, 1), dtype='int16')
    model.eval()
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            ix = np.arange(start, min(start + batch_size, total))
            inputs, masks = refinement_inputs(arrays, ix, candidate, device)
            result = model(inputs, masks)
            for modality in ('eeg', 'hb'):
                for component in ('S', 'O', 'sum'):
                    predictions[modality + '_' + component][ix] = result[modality][component].cpu().numpy()
                if model.hard:
                    for component in ('S', 'O'):
                        hard_ids[modality + '_' + component][ix, :, 0] = result[modality]['ids_' + component].cpu().numpy()
        words = {m + '_' + c:v.cpu().numpy()[None] for m, branches in model.decoded_codewords().items()
            for c, v in branches.items()} if model.hard else {}
    return predictions, hard_ids, words


def save_refinement_evaluation(cfg, out, destination, name, predictions, arrays, manifest,
                               hard_ids=None, words=None, bits=None, eligible=True):
    from src.analysis.dual_component_tokenizer import (
        refinement_fidelity, refinement_code_health, refinement_decision, _csv,
    )
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    rows = manifest['rows']; subjects, metrics = refinement_fidelity(predictions, arrays, rows, name)
    _csv(destination / 'subject_metrics.csv', subjects)
    health = []
    if hard_ids:
        for modality in ('eeg', 'hb'):
            for component in ('S', 'O'):
                key = modality + '_' + component
                health.extend(refinement_code_health(hard_ids[key], words[key], rows, modality, component,
                    cfg['refinement']['codebook_size']))
        write_npz(destination / 'hard_ids.npz', **hard_ids)
        write_npz(destination / 'codewords.npz', **words)
    selected = np.array([i for i, row in enumerate(rows) if row['split'] == 'validation'])
    write_npz(destination / 'validation_predictions.npz', row_index=selected,
        **{key:value[selected] for key, value in predictions.items()})
    result = dict(status='completed', name=name, metrics=metrics, codebook_health=health,
        nominal_bits_per_modality_patch=bits, continuous_sidecars=False,
        deployable_student=eligible, test_used=False)
    baseline = Path(out) / 'baseline_hard' / 'summary.json'
    if baseline.exists():
        result['decision'] = refinement_decision(metrics, read_json(baseline)['metrics'], health,
            cfg['refinement']['health'])
    write_json(destination / 'summary.json', result)
    return result


def refinement_baseline(cfg, out, root, device):
    import torch
    from src.tokenizers.dual_component_tokenizer import DualComponentTokenizer
    destination = Path(out) / 'baseline_hard'
    if (destination / 'summary.json').exists(): return read_json(destination / 'summary.json')
    manifest, arrays = load_cache(out); parent = Path(root) / cfg['refinement']['parent_run']
    state = torch.load(parent / 'jobs/hard_student/seed_17/checkpoint.pt', map_location=device, weights_only=False)
    model = DualComponentTokenizer(state['config']['model'], 'hard_student').to(device)
    model.load_state_dict(state['model_state'])
    candidate = dict(input_mode='native', context_radius=0)
    prediction, ids, words = refinement_predictions(model, arrays, candidate, device)
    return save_refinement_evaluation(cfg, out, destination, 'parent_hard_seed17', prediction, arrays,
        manifest, ids, words, bits=14)


def refine_one(cfg, out, name, seed, device='cpu', memory_bytes=None):
    """Fixed-exposure candidate, then train-only waveform VQ and validation."""
    import torch
    from src.tokenizers.dual_component_tokenizer import DualComponentTokenizer, WaveformDictionary
    settings = cfg['refinement']
    candidate = next((x for x in settings['candidates'] if x['name'] == name), None)
    if candidate is None: raise ValueError('unregistered finite refinement candidate')
    job = Path(out) / 'refinement' / name / f'seed_{seed}'; job.mkdir(parents=True, exist_ok=True)
    if (job / 'summary.json').exists(): return read_json(job / 'summary.json')
    with owner_lock(job, 'candidate'):
        torch.set_num_threads(cfg['resources']['torch_threads'])
        if device.startswith('cuda'):
            torch.cuda.set_device(device)
            cap = memory_bytes or cfg['resources']['max_new_gpu_memory_mib'] * 1024 ** 2
            torch.cuda.set_per_process_memory_fraction(max(cap - 512 * 1024 ** 2, 64 * 1024 ** 2) /
                torch.cuda.get_device_properties(device).total_memory, device)
            torch.cuda.reset_peak_memory_stats(device)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        manifest, arrays = load_cache(out); rows = manifest['rows']
        if any(r['split'] not in ('train', 'validation') for r in rows):
            raise ValueError('test rows cannot enter refinement')
        train = np.array([r['split'] == 'train' for r in rows])
        groups = [np.flatnonzero(train & np.array([r['dataset'] == ds for r in rows]))
                  for ds in sorted({r['dataset'] for r in rows})]
        oracle = candidate['training_mode'] == 'oracle'; started = time.monotonic()
        scales, derivative_scales = {}, {}
        for modality, channels in (('eeg', 1), ('hb', 2)):
            for component, suffix in (('S', 's'), ('O', 'e')):
                truth = np.asarray(arrays[modality + '_' + suffix][train])
                scales[modality + '_' + component] = max(float(np.var(truth)), .01)
                delta = np.diff(truth.reshape(-1, 20, channels), axis=1)
                derivative_scales[modality + '_' + component] = max(float(np.mean(delta ** 2)), 1e-6)
        if oracle:
            predictions = {m + '_' + c:np.asarray(arrays[m + '_' + key]) for m in ('eeg', 'hb')
                for c, key in (('S', 's'), ('O', 'e'), ('sum', 'u'))}
            model = None; learned_ids = {}; learned_words = {}
        else:
            model_cfg = deepcopy(cfg['model'])
            factor = 1 + 2 * candidate['context_radius']
            model_cfg['input_dims'] = {}
            for m in ('eeg','hb'):
                key = ('eeg_spectral_input' if m=='eeg' and candidate['input_mode']=='spectral'
                    else m + ('_input' if candidate['input_mode']=='native' else '_u'))
                dimension=arrays[key].shape[-1]
                if candidate.get('include_metadata'):dimension += arrays['patch_metadata'].shape[-1]
                model_cfg['input_dims'][m] = dimension * factor
            model_cfg.update(width=candidate.get('width',model_cfg['width']),encoder_kind=candidate.get('encoder_kind','mlp'),
                input_channels={m:d//factor//20 for m,d in model_cfg['input_dims'].items()})
            model_cfg.update(revive_dead_codes=candidate.get('revive_dead_codes', False), ema_decay=.95,
                dead_code_threshold=1., revival_warmup_steps=100, revival_interval=25,
                revival_stop_after_steps=max(100, candidate['steps'] - 250),
                max_revivals_per_event=8, revival_strategy='diverse_farthest', revival_count_prior='uniform_batch')
            model = DualComponentTokenizer(model_cfg, candidate['training_mode']).to(device)
            checkpoint = job / 'checkpoint.pt'; start = 0; history = []
            optimizer = torch.optim.AdamW(model.parameters(), lr=candidate['learning_rate'],
                weight_decay=candidate.get('weight_decay', .0001))
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location=device, weights_only=False)
                if saved['candidate'] != candidate or saved['seed'] != seed: raise ValueError('refinement checkpoint drift')
                model.load_state_dict(saved['model_state']); optimizer.load_state_dict(saved['optimizer_state'])
                start = saved['step']; history = saved['history']; np.random.set_state(saved['numpy_rng'])
                torch.set_rng_state(saved['torch_rng'].cpu())
                if saved.get('cuda_rng') is not None and device.startswith('cuda'):
                    torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), device)
            elif candidate['initial'] == 'parent_continuous':
                parent = Path(manifest['input_parent'])
                saved = torch.load(parent / 'jobs/continuous_student' / f'seed_{seed}' / 'checkpoint.pt', map_location=device, weights_only=False)
                model.load_state_dict(saved['model_state'])
            elif candidate['initial'] == 'refinement_continuous':
                parent = Path(manifest['input_parent']).parent / settings['initial_refinement_run']
                initial_seed = candidate.get('initial_seed', seed)
                saved = torch.load(parent / 'refinement' / candidate['initial_candidate'] / f'seed_{initial_seed}' / 'checkpoint.pt',
                    map_location=device, weights_only=False)
                model.load_state_dict(saved['model_state'])
            step_budget = int(candidate['steps'])
            if start > step_budget: raise ValueError('refinement checkpoint exceeds fixed budget')
            model.train()
            for step in range(start, step_budget):
                group_ids = np.random.randint(len(groups), size=settings['batch_size'])
                ix = np.array([np.random.choice(groups[g]) for g in group_ids])
                inputs, masks = refinement_inputs(arrays, ix, candidate, device)
                output = model(inputs, masks); loss = torch.zeros((), device=device)
                terms = {}
                for modality, channels in (('eeg', 1), ('hb', 2)):
                    for component, suffix in (('S', 's'), ('O', 'e')):
                        key = modality + '_' + component
                        truth = torch.as_tensor(np.asarray(arrays[modality + '_' + suffix][ix]).copy(), device=device)
                        estimate = output[modality][component]
                        level = (estimate - truth).square().mean()
                        normalizer = scales[key] if candidate['balance_components'] else 1.
                        loss = loss + level / normalizer
                        if candidate['derivative_weight']:
                            error = (estimate - truth).reshape(-1, 20, channels)
                            loss = loss + candidate['derivative_weight'] * error.diff(dim=1).square().mean() / derivative_scales[key]
                        terms[key] = float(level.detach())
                    loss = loss + output[modality]['commitment']
                loss = loss / 2
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite refinement loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
                if (step + 1) % settings['checkpoint_every'] == 0 or step + 1 == step_budget:
                    history.append(dict(step=step + 1, loss=float(loss.detach()), **terms))
                    _torch_save(checkpoint, dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                        candidate=candidate, model_config=model_cfg, seed=seed, step=step + 1, history=history,
                        numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None))
                    write_json(job / 'progress.json', dict(status='training', step=step + 1, budget=step_budget,
                        elapsed_seconds=time.monotonic() - started, last_loss=history[-1]))
            predictions, learned_ids, learned_words = refinement_predictions(model, arrays, candidate, device,
                settings['evaluation_batch_size'])
        results = []
        if model is not None and model.hard:
            results.append(save_refinement_evaluation(cfg, out, job / 'hard', name, predictions, arrays,
                manifest, learned_ids, learned_words, bits=2 * np.log2(settings['codebook_size'])))
        else:
            if not oracle:
                save_refinement_evaluation(cfg, out, job / 'continuous', name + '_continuous', predictions, arrays, manifest)
            books, ids = {}, {}
            for modality, channels in (('eeg', 1), ('hb', 2)):
                for component in ('S', 'O'):
                    key = modality + '_' + component; value = predictions[key]
                    book = WaveformDictionary(settings['codebook_size'], max(settings['waveform_vq_levels']), channels,
                        candidate['derivative_weight'], seed, settings['codebook_fit_patches'], settings['codebook_fit_iterations'])
                    book.fit(value[train].reshape(-1, value.shape[-1]))
                    books[key] = book; ids[key] = book.encode(value)
            write_npz(job / 'waveform_dictionaries.npz', **{key + '_' + k:v for key, book in books.items() for k, v in book.state().items()})
            for levels in settings['waveform_vq_levels']:
                hard_ids = {k:v[..., :levels] for k, v in ids.items()}
                words = {k:b.codewords[:levels] for k, b in books.items()}
                hard = {k:books[k].decode(v) for k, v in hard_ids.items()}
                for modality in ('eeg', 'hb'): hard[modality + '_sum'] = hard[modality + '_S'] + hard[modality + '_O']
                results.append(save_refinement_evaluation(cfg, out, job / f'vq{levels}', name + f'_vq{levels}',
                    hard, arrays, manifest, hard_ids, words, bits=2 * np.log2(settings['codebook_size']) * levels, eligible=not oracle))
        result = dict(status='completed', candidate=candidate, seed=seed, elapsed_seconds=time.monotonic() - started,
            nominal_observation_support_seconds=2 * (1 + 2 * candidate['context_radius']),
            input_scope='same-modality processed coordinates; extended-context tokens require overlap-closed downstream masks',
            variants=[dict(name=r['name'], bits=r['nominal_bits_per_modality_patch'], decision=r.get('decision'),
                deployable_student=r['deployable_student']) for r in results], test_used=False)
        if model is not None:
            progress = read_json(job / 'progress.json')
            result['training'] = dict(completed_steps=step_budget, resumed_from_step=start,
                elapsed_seconds=progress['elapsed_seconds'],
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else 0,
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.startswith('cuda') else 0)
        if candidate['initial'] in ('parent_continuous','refinement_continuous'):
            result['initialization'] = dict(source=candidate['initial'], weights_only=True,
                optimizer='fresh for declared additional steps', prior_step_count=saved['step'])
        write_json(job / 'summary.json', result)
        return result


def refinement_jobs(cfg, cfg_path, out, root, devices, gpu_workers, pending, phase):
    settings = cfg['refinement']
    pending = list(pending); planned = len(pending)
    results = []; running = []
    slots = [d for d in devices for _ in range(gpu_workers)]
    while pending or running:
        while pending and len(running) < len(slots):
            name, seed = pending.pop(0); job = Path(out) / 'refinement' / name / f'seed_{seed}'
            if (job / 'summary.json').exists(): results.append(read_json(job / 'summary.json')); continue
            used = Counter(x[3] for x in running); device = next(d for d in slots if used[d] < slots.count(d))
            job.mkdir(parents=True, exist_ok=True)
            memory = cfg['resources']['max_new_gpu_memory_mib'] * 1024 ** 2 // slots.count(device)
            command = [sys.executable, str(Path(__file__).resolve()), '--config', str(cfg_path), '--project-root', str(root),
                '--output', str(out), '--stage', 'refine-one', '--candidate', name, '--seed', str(seed), '--device', device,
                '--gpu-memory-bytes', str(memory), '--parent-pid', str(os.getpid())]
            log = (job / 'process.log').open('a'); process = subprocess.Popen(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT)
            running.append((process, log, (name, seed), device, job))
        if not running: continue
        time.sleep(.2)
        for item in list(running):
            process, log, key, device, job = item
            code = process.poll()
            if code is None: continue
            log.close(); running.remove(item)
            record = read_json(job / 'summary.json') if (job / 'summary.json').exists() else dict(status='failed', candidate=key[0], seed=key[1], log=str(job / 'process.log'))
            record['returncode'] = code; results.append(record)
            if code != 0: record['status'] = 'failed'
            write_json(Path(out) / 'refinement_progress.json', dict(phase=phase, completed=len(results), planned=planned,
                active=[dict(candidate=x[2], device=x[3], pid=x[0].pid) for x in running], last=record))
    return results


def run_refinement(cfg, cfg_path, out, root, devices, gpu_workers):
    prepare_refinement(cfg, out, root)
    import torch
    torch.set_num_threads(cfg['resources']['torch_threads'])
    refinement_baseline(cfg, out, root, 'cpu')
    settings = cfg['refinement']
    results = refinement_jobs(cfg, cfg_path, out, root, devices, gpu_workers,
        [(c['name'], settings['seed']) for c in settings['candidates']], 'screen')
    variants = []
    for result in results:
        if result.get('status') != 'completed': continue
        for row in result['variants']:
            if row['deployable_student']: variants.append(dict(row, candidate=result['candidate']['name'], seed=result['seed']))
    def rank(row):
        decision = row['decision']; values = decision['fidelity']
        mean_ratio = np.mean([v['NMSE'] / v['baseline_NMSE'] for v in values])
        if decision['health_pass'] and decision['fidelity_pass']: return (0, row['bits'], mean_ratio)
        return (1, not decision['health_pass'], mean_ratio, row['bits'])
    ranked = sorted(variants, key=rank)
    confirmation = []
    replicate_diagnostic = settings.get('seed_replication', 'only_if_all_thresholds_pass') == 'selected_even_if_diagnostic'
    if ranked and (replicate_diagnostic or
            (ranked[0]['decision']['health_pass'] and ranked[0]['decision']['fidelity_pass'])):
        confirmation = refinement_jobs(cfg, cfg_path, out, root, devices, gpu_workers,
            [(ranked[0]['candidate'], seed) for seed in settings['confirmation_seeds']], 'seed_confirmation')
    summary = dict(status='completed' if all(r.get('status') == 'completed' for r in results + confirmation) else 'failed',
        candidates=results, ranking=ranked, selected=ranked[0] if ranked else None,
        selection_split='validation', test_used=False, confirmation=confirmation,
        seed_replication_scope='optimization stability on the same validation subjects, not independent data confirmation',
        interpretation='Health and waveform fidelity must both pass; oracle compression cannot be selected')
    summary['requirements_met'] = bool(confirmation and len(confirmation) == len(settings['confirmation_seeds']) and
        all(r.get('status') == 'completed' and any(v['name'] == ranked[0]['name'] and v['decision']['health_pass']
            and v['decision']['fidelity_pass'] for v in r['variants']) for r in confirmation))
    write_json(Path(out) / 'refinement_summary.json', summary)
    from src.analysis.dual_component_tokenizer import render_refinement
    figures = render_refinement(cfg, out, summary['selected'])
    write_json(Path(out) / 'figure_index.json', dict(files=figures,
        selection='first identity and baseline-median/P90 error, different validation subjects; no candidate-based selection'))
    return dict(status=summary['status'], completed_candidates=len(results), selected=summary['selected'])


def voltage_synthetic(cfg):
    from src.data.dual_component_tokenizer import voltage_proxy_correction, BANDS
    from src.inference.observation_baselines import eeg_band_power
    from src.tokenizers.dual_component_tokenizer import waveform_windows,overlap_decode
    import torch
    rng=np.random.default_rng(17);t=np.arange(18000)/200
    x=np.column_stack([(1+.15*np.sin(.2*t+i))*np.sin(2*np.pi*(8+i)*t)+.2*rng.normal(size=len(t)) for i in range(3)])
    features=np.log(eeg_band_power(x,bands=BANDS,sample_rate=200,target_rate=10)).reshape(900,-1)
    features-=features[:50].mean(0);a=np.array([np.linspace(-.2,.8,15),np.sin(np.arange(15))])*.02
    u=features@a.T;target=u+.004*np.sin(np.arange(900)[:,None]*.1)*np.array([1.,.7])
    zero=voltage_proxy_correction(x,a,target,strength=0.)
    corrected=voltage_proxy_correction(x,a,target)
    if not np.array_equal(zero['corrected'],x):raise AssertionError('zero correction changed voltage')
    if corrected['diagnostic']['after']>=corrected['diagnostic']['before']:raise AssertionError('synthetic reprojection did not improve')
    y=torch.tensor(np.sin(np.arange(200)[None,:,None]*.03),dtype=torch.float64)
    windows,_=waveform_windows(y,hop=20,span=40)
    recovered=overlap_decode(windows,hop=20,channels=1,length=200)
    error=float((recovered-y).abs().max())
    if error>1e-12:raise AssertionError('overlap identity mismatch')
    return dict(status='completed',measured_arrays_read=0,correction=corrected['diagnostic'],
        zero_correction_max_abs=0.,voltage_closure_max_abs=float(np.max(np.abs(corrected['corrected']+corrected['remaining']-x))),
        overlap_identity_max_abs=error)


def _voltage_record(payload):
    from src.data.dual_component_tokenizer import prepare_voltage_record
    root,cfg,out,parent,rows=payload
    pending=[]
    for trial in sorted({r['trial_id'] for r in rows}):
        if task_begin(out,'voltage_prepared',trial) is not None:
            pending.extend(r for r in rows if r['trial_id']==trial)
    if not pending:return dict(status='completed',skipped=True)
    done=set();counts=Counter();started=time.monotonic()
    try:
        co=read_json(Path(parent)/'coordinates.json')
        read_teacher=lambda row:read_arrays(parent,'teacher',row)
        v=cfg['voltage']['correction']
        for trial,arrays,meta in prepare_voltage_record(root,cfg,pending,co,read_teacher,
                strength=v['strength'],max_log_gain=v['maximum_log_amplitude_gain'],backtracking=v['backtracking']):
            done.add(trial)
            if arrays is None:task_finish(out,'voltage_prepared',trial,'failed',**meta);counts['failed']+=1
            else:
                write_npz(array_paths(out,'voltage_prepared',dict(id=trial)),**arrays)
                task_finish(out,'voltage_prepared',trial,'completed',metadata=meta);counts['completed']+=1
    except Exception:
        error=traceback.format_exc()
        for trial in {r['trial_id'] for r in pending}-done:
            task_finish(out,'voltage_prepared',trial,'failed',error=error);counts['failed']+=1
    return dict(status='completed',record=rows[0]['join_key'],elapsed_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**counts)


def prepare_voltage(cfg,out,root,workers,*,final=False):
    from src.data.dual_component_tokenizer import metadata_inventory
    v=cfg['voltage'];parent=(Path(root)/v['parent_run']).resolve()
    admitted=(Path(root)/'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261028_v3').resolve()
    if parent!=admitted or read_json(parent/'controller.json')['status']!='completed':
        raise ValueError('exact completed frozen public teacher parent required')
    if read_json(parent/'resolved_config.json')['config']['data']['protected_data']!='forbidden':
        raise ValueError('protected source is not admitted')
    if final and not (Path(out)/'voltage_selection.json').exists():
        raise ValueError('freeze validation choice before accessing public test signals')
    inventory=metadata_inventory(root,cfg)
    parent_manifest=read_json(parent/'cache/manifest.json')
    retained={r['trial_id'] for r in parent_manifest['rows']}
    splits=['test'] if final else ['train','validation']
    planned=[r for r in inventory['rows'] if r['split'] in splits]
    rows=[r for r in planned if r['trial_id'] in retained]
    plan=dict(schema='dual_component_voltage_plan_v1',parent=str(parent),rows=planned,
        planned_trials=len({r['trial_id'] for r in planned}),parent_available_trials=len({r['trial_id'] for r in rows}),
        parent_failed_trials=sorted({r['trial_id'] for r in planned}-retained),splits=splits,
        ssms_refitted=0,protected_data='forbidden')
    suffix='test' if final else 'development'
    write_json(Path(out)/f'voltage_{suffix}_plan.json',plan)
    grouped=defaultdict(list)
    for row in rows:
        if task_status(out,'voltage_prepared',row['trial_id'])['status'] not in TERMINAL:
            grouped[row['join_key']].append(row)
    payloads=[(str(root),cfg,str(out),str(parent),g) for _,g in sorted(grouped.items())]
    results=[]
    def progress(record,number):
        write_json(Path(out)/f'voltage_{suffix}_progress.json',dict(completed_records=len(results)+number,
            planned_records=len(payloads),last=record,updated=now()))
    if payloads and not final:
        remaining=list(payloads);bench=[]
        for count in cfg['resources']['cpu_benchmark_workers']:
            selected=remaining[:count];remaining=remaining[count:]
            if not selected:break
            started=time.monotonic();batch=_bounded_jobs(_voltage_record,selected,count,progress=progress)
            elapsed=time.monotonic()-started;results.extend(batch)
            bench.append(dict(workers=count,seconds=elapsed,completed_trials=sum(r.get('completed',0) for r in batch),
                trials_per_second=sum(r.get('completed',0) for r in batch)/elapsed,
                peak_worker_rss_kib=max(r.get('peak_rss_kib',0) for r in batch)))
            write_json(Path(out)/'voltage_cpu_benchmark.json',dict(measurements=bench,
                scope='distinct planned record batches; preparation reused, no extra SSM fits',
                limitation='record sizes and native-load overhead differ; throughput is operational, not a controlled hardware comparison'))
        if bench:
            workers=max(bench,key=lambda r:r['trials_per_second'])['workers']
            write_json(Path(out)/'voltage_cpu_choice.json',dict(workers=workers,reason='highest measured completed-trial throughput within memory/CPU bound'))
        if remaining:results.extend(_bounded_jobs(_voltage_record,remaining,workers,progress=progress))
    elif payloads:results=_bounded_jobs(_voltage_record,payloads,workers,progress=progress)
    records=[task_status(out,'voltage_prepared',t) for t in sorted({r['trial_id'] for r in rows})]
    counts=Counter(r['status'] for r in records)
    summary=dict(status='completed',planned_trials=plan['planned_trials'],parent_failures=plan['parent_failed_trials'],
        correction_outcomes=dict(counts),failed=[dict(trial=r['key'],error=r.get('error')) for r in records if r['status']!='completed'],
        workers=workers,records=results,ssms_refitted=0)
    write_json(Path(out)/f'voltage_{suffix}_prepare_summary.json',summary)
    return pack_voltage(cfg,out,parent,plan,final=final)


def pack_voltage(cfg,out,parent,plan,*,final=False):
    folder=Path(out)/('voltage_test_cache' if final else 'voltage_cache');folder.mkdir(exist_ok=True)
    if (folder/'manifest.json').exists():return read_json(folder/'manifest.json')
    trials=sorted({r['trial_id'] for r in plan['rows']});successful=[];rows={'eeg':[],'hb':[]}
    scale_path=Path(out)/'voltage_scales.json';totals=defaultdict(lambda:[0.,0])
    for trial in trials:
        record=task_status(out,'voltage_prepared',trial)
        if record['status']!='completed':continue
        meta=record['metadata'];group=meta['rows'];base=group[0];successful.append((trial,meta))
        for channel in meta['eeg_channels']:
            rows['eeg'].append(dict(dataset=base['dataset'],subject=base['subject'],record=base['record'],split=base['split'],
                trial_id=trial,id=trial+'__electrode_'+channel,channel=channel,coordinate=base['dataset']+'|'+channel))
        rows['hb'].extend(dict(dataset=r['dataset'],subject=r['subject'],record=r['record'],split=r['split'],
            trial_id=trial,id=r['id'],channel=r['anchor']['hb_channel'],coordinate=r['coordinate']) for r in group)
        if not final and base['split']=='train':
            with np.load(array_paths(out,'voltage_prepared',dict(id=trial)),allow_pickle=False) as z:
                for channel,value in zip(meta['eeg_channels'],z['eeg_u']):
                    key=base['dataset']+'|'+channel;totals[key][0]+=float(np.sum(np.asarray(value,float)**2));totals[key][1]+=value.size
    if not successful:raise ValueError('no admitted voltage targets; preserve failure evidence')
    if final:scales=read_json(scale_path)
    else:
        old=read_json(Path(parent)/'cache/manifest.json')['observation_training_sd']
        scales=dict(eeg={k:max(float(np.sqrt(s/n)),1e-8) for k,(s,n) in totals.items()},
            hb={k:v[1:] for k,v in old.items()},fit='training-only raw EEG RMS without centering; unchanged parent Hb SD')
        write_json(scale_path,scales)
    arrays={};coordinates={}
    for modality,length,channels in [('eeg',4000,1),('hb',200,2)]:
        coordinates[modality]=sorted(scales[modality])
        for row in rows[modality]:
            if row['coordinate'] not in scales[modality]:raise ValueError('test/validation coordinate lacks training scale')
            row['coordinate_index']=coordinates[modality].index(row['coordinate'])
        for key in ['u','s','o']:
            arrays[modality+'_'+key]=np.lib.format.open_memmap(folder/f'{modality}_{key}.npy',mode='w+',dtype='float32',
                shape=(len(rows[modality]),length,channels))
    position={'eeg':0,'hb':0};moments=defaultdict(lambda:np.zeros(4));derivatives=defaultdict(lambda:np.zeros(2))
    for trial,meta in successful:
        with np.load(array_paths(out,'voltage_prepared',dict(id=trial)),allow_pickle=False) as z:
            for modality in ['eeg','hb']:
                n=len(meta['eeg_channels']) if modality=='eeg' else 3
                group=rows[modality][position[modality]:position[modality]+n]
                scale=np.asarray([scales[modality][r['coordinate']] for r in group])
                scale=scale.reshape(n,1,-1)
                for key in ['u','s','o']:
                    value=np.asarray(z[modality+'_'+key],float)
                    if modality=='eeg':value=value[...,None]
                    value=(value/scale).astype('float32')
                    arrays[modality+'_'+key][position[modality]:position[modality]+n]=value
                    if group[0]['split']=='train':
                        v64=np.asarray(value,float);moments[modality+'_'+key]+=np.array([v64.sum(),np.square(v64).sum(),v64.size,0.])
                        d=np.diff(v64,axis=1);derivatives[modality+'_'+key]+=np.array([np.square(d).sum(),d.size])
                position[modality]+=n
    for value in arrays.values():value.flush()
    statistics={k:dict(variance=max(float(sq/n-(s/n)**2),1e-4),mean=float(s/n),
        derivative_mse=max(float(derivatives[k][0]/derivatives[k][1]),1e-6)) for k,(s,sq,n,_) in moments.items()}
    manifest=dict(schema='dual_component_voltage_cache_v1',rows=rows,coordinates=coordinates,
        arrays={k:k+'.npy' for k in arrays},shapes={k:list(x.shape) for k,x in arrays.items()},
        planned_trials=plan['planned_trials'],retained_trials=len(successful),statistics=statistics,
        scale_source=str(scale_path),final=final,target='conditional voltage correction; fixed Tak Hb S/O',
        source_parent=str(parent),protected_data='forbidden')
    if final:manifest['statistics']=read_json(Path(out)/'voltage_cache/manifest.json')['statistics']
    write_json(folder/'manifest.json',manifest);return manifest


def load_voltage_cache(out,*,final=False):
    folder=Path(out)/('voltage_test_cache' if final else 'voltage_cache')
    manifest=read_json(folder/'manifest.json')
    if manifest['schema']!='dual_component_voltage_cache_v1' or manifest['protected_data']!='forbidden':
        raise ValueError('versioned public voltage cache required')
    arrays={k:np.load(folder/v,mmap_mode='r',allow_pickle=False) for k,v in manifest['arrays'].items()}
    return manifest,arrays


def _voltage_batch(manifest,arrays,modality,indices,candidate,cfg,device,positions=None,*,load_targets=True):
    import torch
    from src.tokenizers.dual_component_tokenizer import waveform_windows
    settings=cfg['voltage'][modality];hop=settings['hop_samples'];span=candidate['span_samples']
    indices=np.asarray(indices);rows=manifest['rows'][modality]
    length=arrays[modality+'_u'].shape[1];channels=arrays[modality+'_u'].shape[2]
    coordinates=np.array([rows[i]['coordinate_index'] for i in indices])
    metadata=np.eye(len(manifest['coordinates'][modality]),dtype='float32')[coordinates]
    def selected(key,width):
        if positions is None:
            value=torch.as_tensor(np.asarray(arrays[key][indices]).copy(),device=device)
            return waveform_windows(value,hop=hop,span=width)
        time_index=np.asarray(positions)[:,None]*hop-(width-hop)//2+np.arange(width)[None]
        valid=(time_index>=0)&(time_index<length);take=np.clip(time_index,0,length-1)
        value=np.asarray(arrays[key][indices[:,None],take,:]).copy()
        mask=np.broadcast_to(valid[:,:,None],value.shape).copy()
        value[~mask]=0
        return torch.as_tensor(value.reshape(len(indices),1,-1),device=device),torch.as_tensor(mask.reshape(len(indices),1,-1),device=device)
    inputs,masks=selected(modality+'_u',settings['context_samples'])
    meta=torch.as_tensor(metadata,device=device)[:,None].expand(-1,inputs.shape[1],-1)
    inputs=torch.cat((inputs,meta),-1);masks=torch.cat((masks,torch.ones_like(meta,dtype=torch.bool)),-1)
    target={};target_mask={}
    if load_targets:
        for component,key in [('S','s'),('O','o')]:target[component],target_mask[component]=selected(modality+'_'+key,span)
    return inputs,masks,target,target_mask


def _voltage_predict(model,manifest,arrays,modality,indices,candidate,cfg,device):
    import torch
    from src.tokenizers.dual_component_tokenizer import overlap_decode
    with torch.inference_mode():
        inputs,masks,_,_=_voltage_batch(manifest,arrays,modality,indices,candidate,cfg,device,load_targets=False)
        result=model({modality:inputs},{modality:masks})[modality]
        words={c:result[c].cpu().numpy() for c in ['S','O']}
    setting=cfg['voltage'][modality];channels=arrays[modality+'_u'].shape[2];length=arrays[modality+'_u'].shape[1]
    full={c:overlap_decode(value,hop=setting['hop_samples'],channels=channels,length=length) for c,value in words.items()}
    full['sum']=full['S']+full['O']
    return words,full


def _voltage_codebooks(path):
    from src.tokenizers.dual_component_tokenizer import WaveformDictionary
    with np.load(path,allow_pickle=False) as archive:
        result={}
        for key in ['S','O','ordinary']:
            result[key]=WaveformDictionary.from_state({n:archive[key+'_'+n] for n in
                ['codewords','value_scale','delta_scale','channels','derivative_weight','dimension']})
    return result


def voltage_train(cfg,out,name,seed,device='cpu',memory_bytes=None,*,steps=None,job_override=None,export=True):
    import torch
    from src.tokenizers.dual_component_tokenizer import DualComponentTokenizer,WaveformDictionary,overlap_decode,waveform_windows
    settings=cfg['voltage'];candidate=next(c for c in settings['candidates'] if c['name']==name)
    modality=candidate['modality'];spec=settings[modality]
    job=Path(job_override) if job_override else Path(out)/'voltage_jobs'/name/f'seed_{seed}'
    job.mkdir(parents=True,exist_ok=True)
    if (job/'summary.json').exists():return read_json(job/'summary.json')
    with owner_lock(job,'training'):
        torch.set_num_threads(cfg['resources']['torch_threads']);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        if device.startswith('cuda'):
            torch.cuda.set_device(device);cap=memory_bytes or cfg['resources']['max_new_gpu_memory_mib']*2**20
            torch.cuda.set_per_process_memory_fraction(max(cap-512*2**20,64*2**20)/torch.cuda.get_device_properties(device).total_memory,device)
            torch.cuda.reset_peak_memory_stats(device)
        manifest,arrays=load_voltage_cache(out);rows=manifest['rows'][modality]
        if any(r['split'] not in ('train','validation') for r in rows):raise ValueError('test row entered voltage training')
        training=np.array([r['split']=='train' for r in rows]);train_indices=np.flatnonzero(training)
        groups=[np.array([i for i,r in enumerate(rows) if r['split']=='train' and r['dataset']==d]) for d in sorted({r['dataset'] for r in rows})]
        if any(len(g)==0 for g in groups):raise ValueError('missing dataset training support')
        channels=arrays[modality+'_u'].shape[2];length=arrays[modality+'_u'].shape[1]
        model_cfg=dict(input_dims={modality:spec['context_samples']*channels+len(manifest['coordinates'][modality])},
            target_dims={modality:candidate['span_samples']*channels},width=settings['width'],latent_dim=settings['latent_dim'],
            codebook_size=settings['codebook_size'])
        model=DualComponentTokenizer(model_cfg,'continuous_student').to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
        checkpoint=job/'checkpoint.pt';start=0;history=[];prior_elapsed=0.;budget=int(steps or settings['steps'])
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location=device,weights_only=False)
            if saved['candidate']!=candidate or saved['seed']!=seed or saved['budget']!=budget:raise ValueError('voltage checkpoint contract drift')
            model.load_state_dict(saved['model_state']);optimizer.load_state_dict(saved['optimizer_state'])
            start=saved['step'];history=saved['history'];np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu())
            prior_elapsed=saved['elapsed_seconds']
            if device.startswith('cuda') and saved.get('cuda_rng') is not None:torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),device)
        if start>budget:raise ValueError('checkpoint exceeded frozen step budget')
        started=time.monotonic();model.train();statistics=manifest['statistics']
        for step in range(start,budget):
            batch=spec['batch_size'];group_ids=np.random.randint(len(groups),size=batch);ix=np.empty(batch,dtype=int)
            for g,members in enumerate(groups):
                take=group_ids==g;ix[take]=members[np.random.randint(len(members),size=int(take.sum()))]
            positions=np.random.randint(length//spec['hop_samples'],size=batch) if modality=='eeg' else None
            inputs,masks,targets,target_masks=_voltage_batch(manifest,arrays,modality,ix,candidate,cfg,device,positions)
            result=model({modality:inputs},{modality:masks})[modality];loss=inputs.sum()*0.;terms={}
            for component,key in [('S','s'),('O','o')]:
                mask=target_masks[component];difference=result[component]-targets[component]
                mse=(difference.square()*mask).sum()/mask.sum().clamp_min(1)
                variance=statistics[modality+'_'+key]['variance'];loss=loss+mse/variance
                terms[component]=float(mse.detach())
                if modality=='hb' and candidate['boundary_weight']:
                    full=overlap_decode(result[component],hop=spec['hop_samples'],channels=channels,length=length)
                    truth=torch.as_tensor(np.asarray(arrays[modality+'_'+key][ix]).copy(),device=device)
                    error=full.diff(dim=1)-truth.diff(dim=1)
                    seam=error[:,spec['hop_samples']-1::spec['hop_samples']]
                    loss=loss+candidate['boundary_weight']*seam.square().mean()/statistics[modality+'_'+key]['derivative_mse']
                if candidate['overlap_weight']:
                    blocks=result[component].reshape(batch,-1,candidate['span_samples'],channels)
                    overlap=blocks[:,:-1,spec['hop_samples']:]-blocks[:,1:,:-spec['hop_samples']]
                    loss=loss+candidate['overlap_weight']*overlap.square().mean()/variance
            loss=loss/2
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite voltage training objective')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            if (step+1)%settings['checkpoint_every']==0 or step+1==budget:
                elapsed=prior_elapsed+time.monotonic()-started;history.append(dict(step=step+1,loss=float(loss.detach()),**terms))
                _torch_save(checkpoint,dict(candidate=candidate,model_config=model_cfg,seed=seed,step=step+1,budget=budget,
                    model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),history=history,elapsed_seconds=elapsed,
                    numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None))
                write_json(job/'progress.json',dict(status='training',step=step+1,budget=budget,elapsed_seconds=elapsed,last=history[-1]))
        model.eval();training_elapsed=prior_elapsed+time.monotonic()-started
        if export:
            book_path=job/'dictionaries.npz'
            if not book_path.exists():
                rng=np.random.default_rng(seed);count=settings['codebook_fit_patches']
                fit_indices=rng.choice(train_indices,size=min(len(train_indices),max(256,int(np.ceil(count/(length//spec['hop_samples'])))*2)),replace=False)
                samples={'S':[],'O':[],'ordinary':[]}
                for begin in range(0,len(fit_indices),32):
                    selection=fit_indices[begin:begin+32]
                    words,_=_voltage_predict(model,manifest,arrays,modality,selection,candidate,cfg,device)
                    raw=torch.as_tensor(np.asarray(arrays[modality+'_u'][selection]).copy())
                    raw_words,valid=waveform_windows(raw,hop=spec['hop_samples'],span=candidate['span_samples'])
                    take=valid.all(-1).numpy()
                    for c in ['S','O']:samples[c].append(words[c][take])
                    samples['ordinary'].append(raw_words.numpy()[take])
                books={};levels=max(spec['codebook_levels'])
                for c in ['S','O','ordinary']:
                    book=WaveformDictionary(settings['codebook_size'],levels*(2 if c=='ordinary' else 1),channels,0.,seed,
                        settings['codebook_fit_patches'],settings['codebook_iterations'])
                    books[c]=book.fit(np.concatenate(samples[c]))
                write_npz(book_path,**{c+'_'+k:v for c,book in books.items() for k,v in book.state().items()})
            books=_voltage_codebooks(book_path)
            evaluation=voltage_evaluate(cfg,out,job,model,candidate,manifest,arrays,books,device)
        else:evaluation=None
        result=dict(status='completed',candidate=candidate,seed=seed,step=budget,budget=budget,
            training_elapsed_seconds=training_elapsed,elapsed_seconds=time.monotonic()-started+prior_elapsed,
            steps_per_second=(budget-start)/max(time.monotonic()-started,1e-9) if not export else budget/training_elapsed,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.startswith('cuda') else 0,
            evaluation=evaluation,test_used=False)
        write_json(job/'summary.json',result);return result


def voltage_evaluate(cfg,out,job,model,candidate,manifest,arrays,books,device,*,final=False):
    import torch
    from scipy.ndimage import gaussian_filter1d
    from src.tokenizers.dual_component_tokenizer import overlap_decode,waveform_windows
    from src.analysis.dual_component_tokenizer import voltage_case_metrics,voltage_aggregate,_csv
    modality=candidate['modality'];spec=cfg['voltage'][modality];rows=manifest['rows'][modality]
    destination=Path(job)/('test_evaluation' if final else 'evaluation');destination.mkdir(exist_ok=True)
    if (destination/'summary.json').exists():return read_json(destination/'summary.json')
    n=len(rows);channels=arrays[modality+'_u'].shape[2];length=arrays[modality+'_u'].shape[1]
    levels=spec['codebook_levels'];variants=['continuous','identity']+[f'vq{l}' for l in levels]+[f'ordinary_vq{2*l}' for l in levels]
    if modality=='hb' and candidate['name']=='hb_blocks':variants.append('vq2_smoothed')
    signals=['joint'] if channels==1 else ['joint','HbO','HbR']
    metrics={name:np.empty((n,3,len(signals),5)) for name in variants}
    split_array=np.array([r['split'] for r in rows]);dataset_array=np.array([r['dataset'] for r in rows])
    subject_keys=sorted({(r['dataset'],r['subject']) for r in rows});subject_id=np.array([subject_keys.index((r['dataset'],r['subject'])) for r in rows])
    counts={};seen={};size=cfg['voltage']['codebook_size'];level_count=max(levels)
    for split in np.unique(split_array):
        for dataset in ['pooled',*np.unique(dataset_array)]:
            for c in ['S','O']:
                for level in range(level_count):counts[(split,dataset,c,level)]=np.zeros(size,dtype='int64')
        for c in ['S','O']:
            for level in range(level_count):seen[(split,c,level)]=np.zeros((len(subject_keys),size),dtype=bool)
    for begin in range(0,n,32):
        ix=np.arange(begin,min(begin+32,n));words,full=_voltage_predict(model,manifest,arrays,modality,ix,candidate,cfg,device)
        targets={c:np.asarray(arrays[modality+'_'+key][ix]) for c,key in [('S','s'),('O','o'),('sum','u')]}
        metrics['continuous'][ix],_=voltage_case_metrics(full,targets,hop=spec['hop_samples'])
        metrics['identity'][ix],_=voltage_case_metrics(dict(S=targets['sum'],O=np.zeros_like(targets['sum']),sum=targets['sum']),targets,hop=spec['hop_samples'])
        ids={c:books[c].encode(words[c]) for c in ['S','O']}
        for split in np.unique(split_array[ix]):
            for c in ['S','O']:
                for level in range(level_count):
                    subset=split_array[ix]==split
                    seen[(split,c,level)][subject_id[ix][subset,None],ids[c][subset,:,level]]=True
                    for dataset in ['pooled',*np.unique(dataset_array[ix])]:
                        take=subset if dataset=='pooled' else subset&(dataset_array[ix]==dataset)
                        counts[(split,dataset,c,level)]+=np.bincount(ids[c][take,:,level].ravel(),minlength=size)
        raw_windows,_=waveform_windows(torch.as_tensor(targets['sum']),hop=spec['hop_samples'],span=candidate['span_samples'])
        ordinary_ids=books['ordinary'].encode(raw_windows.numpy())
        for level in levels:
            decoded={c:overlap_decode(books[c].decode(ids[c][...,:level]),hop=spec['hop_samples'],channels=channels,length=length) for c in ['S','O']}
            decoded['sum']=decoded['S']+decoded['O'];metrics[f'vq{level}'][ix],_=voltage_case_metrics(decoded,targets,hop=spec['hop_samples'])
            ordinary=overlap_decode(books['ordinary'].decode(ordinary_ids[...,:2*level]),hop=spec['hop_samples'],channels=channels,length=length)
            metrics[f'ordinary_vq{2*level}'][ix],_=voltage_case_metrics(dict(sum=ordinary),targets,hop=spec['hop_samples'])
            if 'vq2_smoothed' in metrics:
                smoothed={c:gaussian_filter1d(decoded[c],sigma=1.,axis=1,mode='nearest',truncate=4.) for c in ['S','O']}
                smoothed['sum']=smoothed['S']+smoothed['O'];metrics['vq2_smoothed'][ix],_=voltage_case_metrics(smoothed,targets,hop=spec['hop_samples'])
        if begin%2048==0:write_json(destination/'progress.json',dict(completed_cases=int(ix[-1]+1),planned_cases=n,updated=now()))
    health=[];threshold=cfg['voltage']['codebook_health']
    for (split,dataset,component,level),count in counts.items():
        if not count.sum():continue
        probability=count/count.sum();nonzero=probability>0;perplexity=float(np.exp(-np.sum(probability[nonzero]*np.log(probability[nonzero]))))
        active=float(np.mean(nonzero));required=threshold['train_active_fraction'] if split=='train' else threshold['validation_active_fraction']
        health.append(dict(split=split,dataset=dataset,component=component,level=level,size=size,active_codes=int(nonzero.sum()),
            active_fraction=active,perplexity=perplexity,max_probability=float(probability.max()),
            code_counts=count.tolist(),code_subject_counts=seen[(split,component,level)].sum(0).tolist() if dataset=='pooled' else None,
            passed=bool(active>=required and perplexity/size>=threshold['effective_fraction'] and probability.max()<=threshold['maximum_code_probability'])))
    reports={}
    for variant,value in metrics.items():
        subjects,summary=voltage_aggregate(value,rows,modality=modality,variant=variant,signals=signals)
        _csv(destination/f'{variant}_subjects.csv',subjects)
        lev=int(variant.replace('vq','')) if variant.startswith('vq') and variant[2:].isdigit() else None
        reports[variant]=dict(metrics=summary,health_pass=all(h['passed'] for h in health if h['dataset']=='pooled' and h['level']<lev) if lev else None,
            bits_per_patch=2*lev*np.log2(size) if lev else None,
            bits_per_second=2*lev*np.log2(size)*spec['sample_rate_hz']/spec['hop_samples'] if lev else None)
    write_npz(destination/'case_metrics.npz',**metrics)
    report=dict(status='completed',modality=modality,candidate=candidate,variants=reports,codebook_health=health,
        cases=n,signals=signals,final=final,aggregation='case time mean -> equal cases within record -> equal records within subject -> equal subjects within dataset',
        units='frozen training raw EEG RMS squared or inherited Hb observation SD squared',
        boundary='squared error of signed adjacent-sample differences at token hop boundaries',
        continuous_metadata='dataset/channel training scale only; no sample-specific decoder inputs')
    write_json(destination/'summary.json',report);return report


def voltage_final(cfg,out,name,seed,device,memory_bytes=None):
    import torch
    from src.tokenizers.dual_component_tokenizer import DualComponentTokenizer
    selection_path=Path(out)/'voltage_selection.json'
    if not selection_path.exists():raise ValueError('validation selection must exist before test cache access')
    selected=read_json(selection_path)
    if name not in {'eeg_voltage','hb_blocks',selected['hb_candidate']} or seed not in [cfg['voltage']['seed'],*cfg['voltage']['confirmation_seeds']]:
        raise ValueError('unselected candidate cannot access public test arrays')
    job=Path(out)/'voltage_jobs'/name/f'seed_{seed}'
    if (job/'test_result.json').exists():return read_json(job/'test_result.json')
    training=read_json(job/'summary.json')
    if training['status']!='completed' or training['step']!=cfg['voltage']['steps']:raise ValueError('completed fixed-budget checkpoint required')
    torch.set_num_threads(cfg['resources']['torch_threads'])
    if device.startswith('cuda'):
        torch.cuda.set_device(device);cap=memory_bytes or cfg['resources']['max_new_gpu_memory_mib']*2**20
        torch.cuda.set_per_process_memory_fraction(max(cap-512*2**20,64*2**20)/torch.cuda.get_device_properties(device).total_memory,device)
    saved=torch.load(job/'checkpoint.pt',map_location=device,weights_only=False)
    model=DualComponentTokenizer(saved['model_config'],'continuous_student').to(device);model.load_state_dict(saved['model_state']);model.eval()
    manifest,arrays=load_voltage_cache(out,final=True)
    if any(r['split']!='test' for rows in manifest['rows'].values() for r in rows):raise ValueError('final cache split mismatch')
    result=voltage_evaluate(cfg,out,job,model,training['candidate'],manifest,arrays,_voltage_codebooks(job/'dictionaries.npz'),device,final=True)
    result=dict(status='completed',candidate=training['candidate'],seed=seed,evaluation=result)
    write_json(job/'test_result.json',result);return result


def _launch_voltage(cfg,cfg_path,out,root,name,seed,device,workers,*,pilot=None,steps=None,final=False):
    job=Path(pilot) if pilot else Path(out)/'voltage_jobs'/name/f'seed_{seed}';job.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(Path(__file__).resolve()),'--config',str(cfg_path),'--project-root',str(root),
        '--output',str(out),'--stage','voltage-one','--candidate',name,'--seed',str(seed),'--device',device,
        '--gpu-memory-bytes',str(cfg['resources']['max_new_gpu_memory_mib']*2**20//workers),'--parent-pid',str(os.getpid())]
    if pilot:command.extend(['--pilot-output',str(pilot),'--steps',str(steps),'--no-export'])
    if final:command.extend(['--evaluation-split','test'])
    handle=(job/('test_process.log' if final else 'process.log')).open('a')
    return subprocess.Popen(command,cwd=SOURCE,stdout=handle,stderr=subprocess.STDOUT),handle,job


def voltage_jobs(cfg,cfg_path,out,root,devices,gpu_workers,pending,*,phase):
    pending=list(pending);planned=len(pending);results=[];running=[];slots=[d for d in devices for _ in range(gpu_workers)]
    final=phase=='test';filename='test_result.json' if final else 'summary.json'
    while pending or running:
        while pending and len(running)<len(slots):
            name,seed=pending.pop(0);job=Path(out)/'voltage_jobs'/name/f'seed_{seed}'
            if (job/filename).exists():results.append(read_json(job/filename));continue
            usage=Counter(x[3] for x in running);device=next(d for d in slots if usage[d]<slots.count(d))
            process,handle,job=_launch_voltage(cfg,cfg_path,out,root,name,seed,device,slots.count(device),final=final)
            running.append((process,handle,(name,seed),device,job))
        if not running:continue
        time.sleep(.3)
        for item in list(running):
            process,handle,key,device,job=item;code=process.poll()
            if code is None:continue
            handle.close();running.remove(item)
            result=read_json(job/filename) if (job/filename).exists() else dict(status='failed',candidate=key[0],seed=key[1],log=str(job/'process.log'))
            if code!=0:result=dict(result,status='failed',returncode=code)
            results.append(result)
            write_json(Path(out)/'voltage_training_progress.json',dict(phase=phase,completed=len(results),planned=planned,
                active=[dict(candidate=x[2][0],seed=x[2][1],device=x[3],pid=x[0].pid) for x in running],last=result.get('status'),updated=now()))
    write_json(Path(out)/f'voltage_{phase}_jobs.json',dict(status='completed' if all(x['status']=='completed' for x in results) else 'failed',jobs=results))
    if any(x['status']!='completed' for x in results):raise RuntimeError(f'voltage {phase} job failed; inspect retained process logs')
    return results


def voltage_benchmark(cfg,cfg_path,out,root,devices):
    path=Path(out)/'voltage_gpu_benchmark.json'
    if path.exists():return read_json(path)['workers_per_device']
    benchmarks=[]
    for count in [1,2]:
        processes=[];started=time.monotonic()
        for i,device in enumerate(devices):
            for w in range(count):
                pilot=Path(out)/'voltage_benchmark'/f'workers_{count}'/f'device_{i}_worker_{w}'
                processes.append(_launch_voltage(cfg,cfg_path,out,root,'eeg_voltage',1700+i*2+w,device,count,pilot=pilot,steps=100))
        records=[]
        for process,handle,job in processes:
            code=process.wait();handle.close();record=read_json(job/'summary.json') if (job/'summary.json').exists() else {}
            if code or record.get('step')!=100 or record.get('budget')!=100:
                raise RuntimeError('voltage fixed-100-step GPU benchmark failed')
            records.append(record)
        elapsed=time.monotonic()-started
        benchmarks.append(dict(workers_per_device=count,seconds=elapsed,aggregate_steps_per_second=100*len(records)/elapsed,
            jobs=records,peak_reserved_total_bytes=sum(r['peak_cuda_reserved_bytes'] for r in records)))
    selected=max(benchmarks,key=lambda x:x['aggregate_steps_per_second'])['workers_per_device']
    write_json(path,dict(status='completed',workers_per_device=selected,measurements=benchmarks,
        reason='highest observed aggregate throughput; shared per-device 4GiB allocation bound with reserved CUDA context'))
    return selected


def select_voltage(cfg,results):
    hb=[r for r in results if r['candidate']['modality']=='hb'];base=next(r for r in hb if r['candidate']['name']=='hb_blocks')
    def metrics(result,variant='vq2'):return {(r['dataset'],r['component'],r['signal'],r['metric']):r['value'] for r in result['evaluation']['variants'][variant]['metrics']
        if r['split']=='validation' and r['metric'] in ('NMSE','boundary_error_NMSE')}
    baseline=metrics(base);candidates=[]
    pairs=[(r,'vq2') for r in hb]
    if cfg['voltage'].get('decoder_synthesis_sigma_samples')==1.:
        pairs.append((base,'vq2_smoothed'))
    for result,variant in pairs:
        current=metrics(result,variant);comparisons=[dict(dataset=k[0],component=k[1],signal=k[2],metric=k[3],baseline=b,value=current[k],
            ratio=current[k]/b if b else None) for k,b in baseline.items()]
        good=all(x['ratio'] is not None and x['ratio']<=1.05 for x in comparisons if x['metric']=='NMSE')
        boundary=float(np.mean([x['ratio'] for x in comparisons if x['metric']=='boundary_error_NMSE']))
        candidates.append(dict(candidate=result['candidate']['name'],variant=variant,fidelity_noninferiority=good,boundary_ratio=boundary,comparisons=comparisons,
            codebook_health_pass=result['evaluation']['variants']['vq2']['health_pass']))
    eligible=[r for r in candidates if r['fidelity_noninferiority'] and r['codebook_health_pass']]
    choice=min(eligible,key=lambda r:r['boundary_ratio']) if eligible else next(r for r in candidates if r['candidate']=='hb_blocks')
    return dict(hb_candidate=choice['candidate'],hb_variant=choice['variant'],eeg_candidate='eeg_voltage',eeg_levels=4,hb_levels=2,
        boundary_goal_met=bool(choice['fidelity_noninferiority'] and choice['boundary_ratio']<=1-cfg['voltage']['minimum_boundary_improvement']),
        candidates=candidates,selection_split='validation',test_used=False,selected_at=now(),
        interpretation='noninferiority screen is an engineering threshold, not a statistical noninferiority test')


def run_voltage(cfg,cfg_path,out,root,devices,gpu_workers,workers):
    if cfg['voltage'].get('reuse_run'):reuse_voltage(cfg,out,root)
    synthetic_path=Path(out)/'voltage_synthetic.json'
    if not synthetic_path.exists():write_json(synthetic_path,voltage_synthetic(cfg))
    write_json(Path(out)/'voltage_stage.json',dict(stage='development_targets',updated=now()))
    prepare_voltage(cfg,out,root,workers)
    write_json(Path(out)/'voltage_stage.json',dict(stage='gpu_benchmark',updated=now()))
    gpu_workers=voltage_benchmark(cfg,cfg_path,out,root,devices)
    v=cfg['voltage'];write_json(Path(out)/'voltage_stage.json',dict(stage='screen_training',updated=now()))
    screen=voltage_jobs(cfg,cfg_path,out,root,devices,gpu_workers,[(c['name'],v['seed']) for c in v['candidates']],phase='screen')
    selection_path=Path(out)/'voltage_selection.json'
    if not selection_path.exists():write_json(selection_path,select_voltage(cfg,screen))
    selection=read_json(selection_path);candidates=sorted({'eeg_voltage','hb_blocks',selection['hb_candidate']})
    write_json(Path(out)/'voltage_stage.json',dict(stage='seed_replication',updated=now()))
    voltage_jobs(cfg,cfg_path,out,root,devices,gpu_workers,[(c,s) for c in candidates for s in v['confirmation_seeds']],phase='replication')
    write_json(Path(out)/'voltage_stage.json',dict(stage='frozen_public_test_targets',updated=now()))
    prepare_voltage(cfg,out,root,workers,final=True)
    write_json(Path(out)/'voltage_stage.json',dict(stage='frozen_public_test',updated=now()))
    final=voltage_jobs(cfg,cfg_path,out,root,devices,gpu_workers,[(c,s) for c in candidates for s in [v['seed'],*v['confirmation_seeds']]],phase='test')
    summary=summarize_voltage(cfg,out,selection,final)
    render_voltage(cfg,out,selection,device=devices[0])
    write_json(Path(out)/'voltage_stage.json',dict(stage='completed',updated=now()))
    return summary


def reuse_voltage(cfg,out,root):
    """Versioned decoder-only consumer; completed numerical artifacts stay immutable."""
    import shutil
    record_path=Path(out)/'voltage_input_reuse.json'
    if record_path.exists():return read_json(record_path)
    parent=(Path(root)/cfg['voltage']['reuse_run']).resolve()
    admitted=(Path(root)/'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261029_voltage_v1').resolve()
    if parent!=admitted:raise ValueError('exact completed voltage producer required')
    while read_json(parent/'controller.json').get('status')!='completed':
        unit=read_json(parent/'launch.json')['unit']
        state=subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','--value'],text=True).strip()
        if state not in ('active','activating'):raise RuntimeError('voltage producer is not completed or running')
        write_json(Path(out)/'voltage_stage.json',dict(stage='waiting_for_completed_voltage_producer',parent=str(parent),updated=now()))
        time.sleep(10)
    source_cfg=read_json(parent/'resolved_config.json')['config'];candidate=deepcopy(cfg['voltage'])
    for key in ['reuse_run','decoder_synthesis_sigma_samples','decoder_synthesis']:candidate.pop(key,None)
    if candidate!=source_cfg['voltage']:raise ValueError('decoder-only reuse changed training or data contract')
    names=['voltage_prepared','tasks','voltage_cache','voltage_test_cache','voltage_jobs','voltage_scales.json',
        'voltage_development_plan.json','voltage_test_plan.json','voltage_development_prepare_summary.json',
        'voltage_test_prepare_summary.json','voltage_cpu_benchmark.json','voltage_cpu_choice.json','voltage_gpu_benchmark.json',
        'voltage_synthetic.json']
    def link(src,dst):
        if Path(dst).exists():return dst
        os.link(src,dst);return dst
    for name in names:
        source=parent/name;target=Path(out)/name
        if source.is_dir():shutil.copytree(source,target,copy_function=link,dirs_exist_ok=True)
        elif source.exists() and not target.exists():shutil.copy2(source,target)
    result=dict(schema='dual_voltage_decoder_reuse_v1',parent=str(parent),source_config=str(parent/'resolved_config.json'),
        training_and_data_contract_equal=True,reused=names,new_training_steps=0,new_ssm_fits=0,
        storage='hardlinks to completed producer arrays and checkpoint/evaluation artifacts; metadata copied where rewritten',
        selection_source=str(Path(out)/'voltage_selection.json'))
    write_json(record_path,result);return result


def summarize_voltage(cfg,out,selection,final):
    import csv
    from src.analysis.dual_component_tokenizer import _csv,paired_subject_bootstrap
    grouped=defaultdict(list)
    for result in final:
        name=result['candidate']['name'];modality=result['candidate']['modality'];seed=result['seed']
        folder=Path(out)/'voltage_jobs'/name/f'seed_{seed}'/'test_evaluation'
        main='vq4' if modality=='eeg' else 'vq2'
        names={main:('selected' if name in ('eeg_voltage',selection['hb_candidate']) else 'hb_blocks'),
            'continuous':name+'_continuous','identity':modality+'_identity',
            ('ordinary_vq8' if modality=='eeg' else 'ordinary_vq4'):name+'_ordinary'}
        if name=='hb_blocks':
            names[main]='hb_blocks'
            names['vq2_smoothed']=('selected' if selection.get('hb_variant')=='vq2_smoothed' else 'hb_blocks_smoothed')
        for variant,label in names.items():
            file=folder/f'{variant}_subjects.csv'
            if not file.exists():continue
            for row in csv.DictReader(file.open()):
                key=(row['dataset'],row['subject'],modality,row['component'],row['signal'],row['metric'],label)
                if row['value']:grouped[key].append((seed,float(row['value'])))
    subjects=[]
    for key,values in grouped.items():
        # Identity baseline may have been evaluated under several Hb candidates.
        by_seed=defaultdict(list)
        for seed,value in values:by_seed[seed].append(value)
        subjects.append(dict(zip(('dataset','subject','modality','component','signal','metric','model'),key),
            value=float(np.mean([np.mean(x) for x in by_seed.values()])),seeds=len(by_seed)))
    if selection['hb_candidate']=='hb_blocks' and selection.get('hb_variant','vq2')=='vq2':
        subjects.extend(dict(r,model='selected') for r in subjects.copy() if r['modality']=='hb' and r['model']=='hb_blocks')
    _csv(Path(out)/'voltage_analysis_subjects.csv',subjects)
    grouped_summary=defaultdict(list)
    for row in subjects:grouped_summary[tuple(row[k] for k in ('dataset','modality','component','signal','metric','model'))].append(row)
    metrics=[dict(zip(('dataset','modality','component','signal','metric','model'),key),
        value=float(np.mean([r['value'] for r in group])),subjects=len(group)) for key,group in grouped_summary.items()]
    paired=[]
    for dataset in sorted({r['dataset'] for r in subjects}):
        for component in ['S','O','sum']:
            for signal in ['joint','HbO','HbR']:
                for metric in ['NMSE','boundary_error_NMSE']:
                    subset=[r for r in subjects if (r['dataset'],r['modality'],r['component'],r['signal'],r['metric'])==(dataset,'hb',component,signal,metric)]
                    first=[r for r in subset if r['model']=='selected'];second=[r for r in subset if r['model']=='hb_blocks']
                    paired.append(dict(dataset=dataset,component=component,signal=signal,metric=metric,
                        **paired_subject_bootstrap(first,second,value='value',samples=2000,seed=17)))
    dev=read_json(Path(out)/'voltage_development_prepare_summary.json');test=read_json(Path(out)/'voltage_test_prepare_summary.json')
    summary=dict(status='completed',experiment_id=cfg['experiment_id'],selection=selection,metrics=metrics,
        hb_paired_subject_comparisons=paired,development_targets=dev,test_targets=test,
        ssms_refitted=0,seeds=[cfg['voltage']['seed'],*cfg['voltage']['confirmation_seeds']],
        interpretation='exploratory conditional voltage targets and fixed-Tak Hb reconstruction; no fresh test or physiological truth claim',
        inference='paired whole-subject percentile intervals; seeds averaged within subject; no multiplicity-adjusted significance claims')
    write_json(Path(out)/'voltage_summary.json',summary);return summary


def render_voltage(cfg,out,selection,device='cpu'):
    import torch
    from src.tokenizers.dual_component_tokenizer import DualComponentTokenizer,synthesize_component
    from src.analysis.dual_component_tokenizer import render_voltage_examples
    manifest,arrays=load_voltage_cache(out,final=True);scales=read_json(Path(out)/'voltage_scales.json')
    seed=cfg['voltage']['seed'];models={};books={}
    for name in {'eeg_voltage','hb_blocks',selection['hb_candidate']}:
        folder=Path(out)/'voltage_jobs'/name/f'seed_{seed}'
        saved=torch.load(folder/'checkpoint.pt',map_location=device,weights_only=False)
        model=DualComponentTokenizer(saved['model_config'],'continuous_student').to(device)
        model.load_state_dict(saved['model_state']);models[name]=(model.eval(),saved['candidate']);books[name]=_voltage_codebooks(folder/'dictionaries.npz')
    with np.load(Path(out)/'voltage_jobs/hb_blocks'/f'seed_{seed}'/'test_evaluation/case_metrics.npz') as z:baseline_error=z['vq2'][:,2,0,0]
    examples=[];hb_rows=manifest['rows']['hb']
    for dataset in sorted({r['dataset'] for r in hb_rows}):
        group=[i for i,r in enumerate(hb_rows) if r['dataset']==dataset];used=set()
        for label,quantile in [('first_identity',None),('median_baseline_error',.5),('p90_baseline_error',.9)]:
            eligible=[i for i in group if hb_rows[i]['subject'] not in used]
            if not eligible:continue
            i=eligible[0] if quantile is None else min(eligible,key=lambda j:abs(baseline_error[j]-np.quantile(baseline_error[group],quantile)))
            used.add(hb_rows[i]['subject'])
            for modality,index in [('hb',i),('eeg',next(j for j,r in enumerate(manifest['rows']['eeg']) if r['trial_id']==hb_rows[i]['trial_id']))]:
                row=manifest['rows'][modality][index];spec=cfg['voltage'][modality];channels=arrays[modality+'_u'].shape[2]
                true={c:np.asarray(arrays[modality+'_'+key][index]) for c,key in [('S','s'),('O','o'),('sum','u')]}
                predictions={}
                for role,name in [('selected','eeg_voltage' if modality=='eeg' else selection['hb_candidate']),('baseline','eeg_voltage' if modality=='eeg' else 'hb_blocks')]:
                    model,candidate=models[name];word,continuous=_voltage_predict(model,manifest,arrays,modality,[index],candidate,cfg,device)
                    levels=4 if modality=='eeg' else 2
                    if role=='baseline' and modality=='eeg':levels=2
                    sigma=cfg['voltage'].get('decoder_synthesis_sigma_samples',0.) if modality=='hb' and role=='selected' and selection.get('hb_variant')=='vq2_smoothed' else 0.
                    prediction={c:synthesize_component(books[name][c].decode(books[name][c].encode(word[c],levels=levels)),
                        hop=spec['hop_samples'],channels=channels,length=true['sum'].shape[0],smoothing_sigma=sigma)[0] for c in ['S','O']}
                    prediction['sum']=prediction['S']+prediction['O'];predictions[role]=prediction
                    if role=='selected':predictions['continuous']={c:v[0] for c,v in continuous.items()}
                factor=scales['eeg'][row['coordinate']] if modality=='eeg' else 1.
                examples.append(dict(row=row,modality=modality,selection=label,targets={k:v*factor for k,v in true.items()},
                    predictions={role:{k:v*factor for k,v in d.items()} for role,d in predictions.items()},
                    sample_rate_hz=spec['sample_rate_hz'],hop=spec['hop_samples'],candidate=(selection['hb_candidate']+' + fixed 100ms synthesis' if selection.get('hb_variant')=='vq2_smoothed' else selection['hb_candidate']) if modality=='hb' else 'eeg_voltage',
                    baseline='2-level/branch' if modality=='eeg' else 'independent 2s blocks',
                    selected_rate='4-level/branch; 560 bit/s/electrode' if modality=='eeg' else '2-level/branch; 28 bit/2s',
                    units='uV' if modality=='eeg' else 'Frozen training SD'))
    return render_voltage_examples(out,examples,read_json(Path(out)/'voltage_summary.json'))


if __name__ == '__main__':
    main()
