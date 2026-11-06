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
          'voltage', 'voltage-one', 'voltage-token-atlas', 'eeg-o', 'eeg-o-one',
          'voltage-teacher-plan', 'voltage-teacher-pilot', 'voltage-teacher', 'voltage-teacher-report',
          'separation-plan', 'separation-pilot', 'separation', 'separation-report',
          'power-cap-plan', 'power-cap', 'power-cap-report', 'power-cap-method-report',
          'latent-health-plan', 'latent-health-prepare', 'latent-health-benchmark', 'latent-health-one', 'latent-health')
VARIANTS = ('continuous_student', 'hard_student', 'component_compression', 'smooth_hard', 'ordinary_hard')
TERMINAL = ('completed', 'failed', 'unavailable')


def serial(value):
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, np.ndarray): return serial(value.tolist())
    if isinstance(value, (tuple, list)): return [serial(v) for v in value]
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
    if isinstance(cfg,dict) and cfg.get('schema')=='dual_component_voltage_power_cap_posterior_v1':
        extra=cfg;cfg=read_config(SOURCE/extra['base_contract'])
        cfg.update(schema=extra['schema'],experiment_id=extra['experiment_id'],posterior=extra['posterior'])
        cfg['resources'].update(extra['resources']);s=cfg['posterior']
        expected=[dict(case_id='simultaneous_eeg_nirs__VP002__cnt_nback__0026',injection_channel='O1'),
                  dict(case_id='simultaneous_eeg_nirs__VP005__cnt_nback__0000',injection_channel='O2')]
        if (s['cases']!=expected or s['allowed_splits']!=['train','validation']
                or s['test_access']!='forbidden' or s['protected_data']!='forbidden'
                or s['regions']!=['left_motor','prefrontal','occipital'] or s['channels']!=17
                or s['occipital_direction']!=[0,-1,0] or s['occipital_hb_channel']!='POOz'
                or s['occipital_eeg_channels']!=['POz','O1','O2','Pz','P3','P4']
                or s['training']!='original_six_first_training_windows_only'
                or s['injection']!='identical_retained_waveform_and_absolute_amplitude_at_named_posterior_electrode'
                or s['affected_roi']!=2 or s['cap']!=1. or s['score_seconds']!=[30,50]
                or s['initial_solver_budget']!=cfg['examples']['initial_solver_budget']
                or s['continuation_solver_budget']!=cfg['examples']['continuation_solver_budget']):
            raise ValueError('fixed two-case posterior scope, geometry and training-only reference required')
    if isinstance(cfg,dict) and cfg.get('schema')=='dual_component_voltage_power_cap_examples_v1':
        extra=cfg;cfg=read_config(SOURCE/extra['base_contract'])
        cfg.update(schema=extra['schema'],experiment_id=extra['experiment_id'],examples=extra['examples'])
        cfg['resources'].update(extra['resources']);s=cfg['examples']
        if (s['subjects']!=['VP005','VP014','VP017','VP019'] or s['allowed_split']!='validation'
                or s['test_access']!='forbidden' or s['protected_data']!='forbidden'
                or s['selection']!='first_retained_validation_slot_per_other_subject_before_array_reads'
                or s['injection']!='identical_retained_Fp1_waveform_and_absolute_amplitude'
                or s['training']!='reuse_frozen_reference_and_coordinates_no_refit'
                or s['variants']!=['baseline','ocular'] or s['cap']!=1. or s['score_seconds']!=[30,50]
                or cfg['power_cap']['physical']['observation_weights']!=[.1,10.,10.]
                or s['initial_solver_budget']!={'max_iterations':600,'max_evaluations_per_trial':1600}
                or s['continuation_solver_budget']!={'max_iterations':1500,'max_evaluations_per_trial':4000}):
            raise ValueError('fixed four-subject Hb-weighted examples contract required')
    if isinstance(cfg,dict) and cfg.get('schema')=='dual_component_voltage_power_cap_v1':
        extra=cfg;cfg=read_config(SOURCE/extra['base_contract'])
        cfg.update(schema=extra['schema'],experiment_id=extra['experiment_id'],
            power_cap=extra['power_cap'],claim_limits=extra['claim_limits'])
        cfg['resources'].update(extra['resources']);s=cfg['power_cap']
        if (s['allowed_splits']!=['train','validation'] or s['test_access']!='forbidden'
                or s['protected_data']!='forbidden' or s['channels']!=16
                or s['sample_rate_hz']!=200 or s['seconds']!=90
                or s['case_id']!='simultaneous_eeg_nirs__VP002__cnt_nback__0026'
                or s['reference']['cap_primary']!=1. or s['reference']['cap_sensitivity']!=[2.,4.]
                or s['reference']['shrinkage']!=.1
                or s['local']['window_samples']!=200 or s['local']['hop_samples']!=20
                or s['physical']['arm']!='tak_fixed' or s['physical']['condition']!='joint_full'
                or s['interventions']['neural']!=['neural10','neural35','neural55']):
            raise ValueError('fixed one-window power-cap diagnostic and training-only reference required')
        weights=np.asarray(s['physical'].get('observation_weights',[1.,1.,1.]),float)
        if weights.shape!=(3,) or not np.isfinite(weights).all() or np.any(weights<=0):
            raise ValueError('positive named EEG/HbO/HbR objective weights required')
        budget=s['physical'].get('solver_budget',{})
        if set(budget)-{'max_iterations','max_evaluations_per_trial'} or any(
                not isinstance(v,int) or v<1 for v in budget.values()):
            raise ValueError('only positive iteration/evaluation budget overrides are supported')
    if isinstance(cfg,dict) and cfg.get('schema')=='dual_component_voltage_separation_v1':
        extra=cfg;cfg=read_config(SOURCE/extra['base_contract'])
        cfg.update(schema=extra['schema'],experiment_id=extra['experiment_id'],
            separation=extra['separation'],claim_limits=extra['claim_limits'])
        cfg['resources'].update(extra['resources']);s=cfg['separation']
        if (s['allowed_splits']!=['train','validation'] or s['test_access']!='forbidden'
                or s['protected_data']!='forbidden' or s['channels']!=16 or s['sample_rate_hz']!=200
                or s['ranks']!=[4,8,12] or s['innovation_fractions']!=[0.,.25,.5,.75,1.]
                or s['predictor_lags_samples']!=[1,2,3,4] or s['codebook']['size']!=128
                or s['codebook']['S_levels']!=4 or s['codebook']['O_levels']!=4):
            raise ValueError('finite native-voltage two-scale separation contract required')
    if isinstance(cfg, dict) and cfg.get('schema') == 'dual_component_latent_health_v1':
        extra = cfg
        cfg = read_config(SOURCE / extra['base_contract'])
        cfg.update(schema=extra['schema'], experiment_id=extra['experiment_id'],
                   latent_health=extra['latent_health'], claim_limits=extra['claim_limits'])
        cfg['resources'].update(extra['resources'])
        s = cfg['latent_health']
        if (s['parent_run'] != 'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261028_v3'
                or s['allowed_splits'] != ['train', 'validation'] or s['test_access'] != 'forbidden'
                or s['measured_modality'] != 'hb' or s['parent_steps'] != [250, 550]
                or s['event_support_s'] != [-5, 25] or s['seconds'] != 30 or s['sample_rate_hz'] != 10
                or s['patch_seconds'] != 2 or s['patches'] != 15 or s['channels'] != 2 or s['input_dim'] != 40
                or s['cross_modality'] or s['cross_branch_decoder'] or s['continuous_sidecars']
                or s['codebook_sizes'] != [128, 512] or s['candidates'] != ['continuous', 'vq128', 'vq512']
                or s['seeds'] != [17, 29, 43] or s['dropout'] != 0.
                or s['warmup_steps'] != 1000 or s['continuation_steps'] != 2000
                or s['revival_stop_after_steps'] > s['continuation_steps'] - 500):
            raise ValueError('fixed global latent health tensor, development boundary and finite matrix required')
    if isinstance(cfg, dict) and cfg.get('schema') == 'dual_component_voltage_teacher_v1':
        extra = cfg
        cfg = read_config(SOURCE / extra['base_contract'])
        cfg.update(schema=extra['schema'], experiment_id=extra['experiment_id'],
                   voltage_teacher=extra['voltage_teacher'], claim_limits=extra['claim_limits'])
        cfg['resources'].update(extra['resources'])
        s = cfg['voltage_teacher']
        if (s['allowed_splits'] != ['train', 'validation'] or s['test_access'] != 'forbidden'
                or s['stft'] != dict(samples=200, hop=50, window='hann', boundary='zeros', full_spectrum=True)
                or s['arms'] != ['raw', 'legacy_target', 'eog_regression', 'wiener', 'physics_wiener', 'shifted_physics']
                or s['amplitudes_training_rms'] != [.5, 1., 2.]):
            raise ValueError('fixed voltage teacher transform, development scope and finite arms required')
    if isinstance(cfg, dict) and cfg.get('schema') == 'dual_component_eeg_o_v1':
        extra = cfg
        cfg = read_config(SOURCE / extra['base_contract'])
        cfg.update(schema=extra['schema'], experiment_id=extra['experiment_id'],
                   eeg_o=extra['eeg_o'], claim_limits=extra['claim_limits'])
        cfg['resources'].update(extra['resources'])
        spec = cfg['eeg_o']
        names=(['proxy_short','proxy_full'] if spec.get('architecture')=='projection_gain'
               else ['single_short','single_long','joint_long'])
        if (spec['allowed_splits'] != ['train', 'validation'] or spec['test_access'] != 'forbidden'
                or spec['sample_rate_hz'] != 200 or spec['hop_samples'] != 20
                or spec['channels'] != 16 or spec['length_samples'] != 4000
                or spec['codebook_size'] != 128 or spec['codebook_levels'] != 4
                or spec['steps'] < 1 or spec['batch_trials'] < 1
                or spec['bits_per_O_patch'] != 28 or spec['bits_per_O_second_per_electrode'] != 280
                or [c['name'] for c in spec['candidates']] != names
                or any(c['context_bins'] < 1 or c['context_bins'] % 2 != 1 for c in spec['candidates'])):
            raise ValueError('finite EEG-O development-only tensor, candidate and bitrate contract required')
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
    parser.add_argument('--analysis-output', type=Path, help='Fresh direct-child retained-token analysis export')
    parser.add_argument('--report-output', type=Path, help='Fresh figure collection for the retained-token atlas')
    parser.add_argument('--posterior-run', type=Path, help='Completed posterior evidence for the voltage-method report')
    parser.add_argument('--token-source-analysis', type=Path, help='Reuse complete token assignments for shape-diverse display examples, without inference')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv); cfg = read_config(args.config)
    if cfg['schema']=='dual_component_voltage_power_cap_posterior_v1':
        if args.stage not in ('check','synthetic','power-cap-plan','power-cap','power-cap-report'):
            raise ValueError('use the bounded posterior power-cap stages')
        if args.stage=='check':
            print(json.dumps(dict(status='posterior_contract_valid',measured_arrays_read=0)));return
        if args.output is None:raise ValueError('--output required')
        if args.stage=='power-cap-report':
            from src.analysis.dual_component_tokenizer import render_voltage_power_cap_examples
            if args.report_output is None:raise ValueError('--report-output must be fresh')
            result=render_voltage_power_cap_examples(args.output,args.report_output)
        else:
            with owner_lock(args.output):
                freeze_contract(args.output,cfg,args.project_root)
                result=run_voltage_power_cap_posterior(cfg,args.output,args.project_root,
                    args.workers or cfg['resources']['cpu_workers'],args.stage)
        print(json.dumps(serial(result)));return
    if cfg['schema']=='dual_component_voltage_power_cap_examples_v1':
        if args.stage not in ('check','synthetic','power-cap-plan','power-cap','power-cap-report','power-cap-method-report'):
            raise ValueError('use the bounded examples power-cap stages')
        if args.stage=='check':
            print(json.dumps(dict(status='examples_contract_valid',measured_arrays_read=0)));return
        if args.output is None:raise ValueError('--output required')
        if args.stage=='power-cap-method-report':
            from src.analysis.dual_component_tokenizer import render_voltage_power_cap_method_report
            if args.report_output is None or args.posterior_run is None:
                raise ValueError('--report-output and --posterior-run required')
            result=render_voltage_power_cap_method_report(args.output,args.posterior_run,args.report_output,
                workers=args.workers or 2)
        elif args.stage=='power-cap-report':
            from src.analysis.dual_component_tokenizer import render_voltage_power_cap_examples
            if args.report_output is None:raise ValueError('--report-output must be fresh')
            result=render_voltage_power_cap_examples(args.output,args.report_output)
        else:
            with owner_lock(args.output):
                freeze_contract(args.output,cfg,args.project_root)
                result=run_voltage_power_cap_examples(cfg,args.output,args.project_root,
                    args.workers or cfg['resources']['cpu_workers'],args.stage)
        print(json.dumps(serial(result)));return
    if cfg['schema']=='dual_component_voltage_power_cap_v1':
        if args.stage not in ('check','synthetic','power-cap-plan','power-cap'):
            raise ValueError('use the finite power-cap diagnostic stages')
        if args.stage=='check':
            print(json.dumps(dict(status='power_cap_contract_valid',measured_arrays_read=0,
                experiment_id=cfg['experiment_id'],test_access='forbidden')));return
        if args.output is None:raise ValueError('--output required')
        with owner_lock(args.output):
            freeze_contract(args.output,cfg,args.project_root)
            result=run_voltage_power_cap_probe(cfg,args.output,args.project_root,
                args.workers or cfg['resources']['cpu_workers'],args.stage)
        print(json.dumps(serial(result)));return
    if cfg['schema']=='dual_component_voltage_separation_v1':
        if args.stage not in ('check','synthetic','separation-plan','separation-pilot','separation','separation-report'):
            raise ValueError('use the versioned separation stages')
        if args.stage=='check':
            print(json.dumps(dict(status='separation_contract_valid',measured_arrays_read=0,
                experiment_id=cfg['experiment_id'],test_access='forbidden')));return
        if args.output is None:raise ValueError('--output required')
        if args.stage=='separation-report':
            from src.analysis.dual_component_tokenizer import render_voltage_separation_report
            result=render_voltage_separation_report(args.output,args.analysis_output)
        else:
            with owner_lock(args.output):
                freeze_contract(args.output,cfg,args.project_root)
                result=run_voltage_separation(cfg,args.output,args.project_root,
                    args.workers or cfg['resources']['cpu_workers'],args.stage)
        print(json.dumps(serial(result)));return
    if cfg['schema'] == 'dual_component_latent_health_v1':
        result = latent_health_dispatch(args, cfg)
        print(json.dumps(serial(result)))
        return
    if cfg['schema'] == 'dual_component_voltage_teacher_v1':
        if args.stage not in ('check', 'synthetic', 'voltage-teacher-plan', 'voltage-teacher-pilot',
                              'voltage-teacher', 'voltage-teacher-report'):
            raise ValueError('target-feasibility contract cannot launch tokenizer training')
        if args.stage == 'check':
            print(json.dumps(dict(status='voltage_teacher_contract_valid', measured_arrays_read=0,
                experiment_id=cfg['experiment_id'], protected_data='forbidden', test_access='forbidden')))
            return
        if args.output is None:
            raise ValueError('--output required for retained feasibility evidence')
        if args.stage == 'voltage-teacher-report':
            from src.analysis.dual_component_tokenizer import render_voltage_teacher_report
            result = render_voltage_teacher_report(args.output, args.analysis_output)
        elif args.stage == 'synthetic':
            result = voltage_teacher_synthetic(cfg, args.output)
        else:
            with owner_lock(args.output):
                freeze_contract(args.output, cfg, args.project_root)
                result = run_voltage_teacher(cfg, args.output, args.project_root,
                    args.workers or cfg['resources']['cpu_workers'], args.stage)
        print(json.dumps(serial(result)))
        return
    if cfg['schema']=='dual_component_eeg_o_v1' and args.stage not in ('check','synthetic','eeg-o','eeg-o-one'):
        raise ValueError('EEG-O refinement uses only its development stages')
    if cfg['schema']=='dual_component_eeg_o_v1' and args.evaluation_split!='development':
        raise ValueError('EEG-O refinement forbids test evaluation')
    if cfg['schema']=='dual_component_voltage_v1' and args.stage not in ('check','synthetic','voltage','voltage-one','voltage-token-atlas'):
        raise ValueError('voltage targets require their own versioned execution stages')
    if cfg['schema'] == 'dual_component_refinement_v1' and args.stage not in ('check', 'refine', 'refine-one'):
        raise ValueError('refinement uses its finite train/validation-only stages')
    if args.stage == 'check':
        if cfg['schema']=='dual_component_eeg_o_v1':
            print(json.dumps(dict(status='eeg_o_contract_valid', experiment_id=cfg['experiment_id'],
                allowed_splits=cfg['eeg_o']['allowed_splits'], test_access='forbidden',
                candidates=cfg['eeg_o']['candidates'], measured_arrays_read=0)));return
        if cfg['schema']=='dual_component_voltage_v1':
            print(json.dumps(dict(status='voltage_contract_valid',experiment_id=cfg['experiment_id'],
                stages=['check','synthetic','voltage','voltage-one'],candidates=cfg['voltage']['candidates'],
                selection_splits=cfg['voltage']['selection_splits'],final_split=cfg['voltage']['final_split'],
                ssms_refitted=0,protected_data='forbidden',measured_arrays_read=0)));return
        print(json.dumps(dict(status='software_contract_valid', stages=STAGES, variants=VARIANTS,
            seeds=seeds(cfg), protected_data='forbidden', measured_arrays_read=0))); return
    if args.stage == 'synthetic':
        result = (eeg_o_synthetic(cfg) if cfg['schema']=='dual_component_eeg_o_v1' else
                  voltage_synthetic(cfg) if cfg['schema']=='dual_component_voltage_v1' else synthetic(cfg))
        if args.output: write_json(args.output / 'synthetic.json', result)
        print(json.dumps(serial(result))); return
    if args.output is None: raise ValueError('--output required for a retained run')
    out, root = args.output.resolve(), args.project_root.resolve()
    if args.stage=='voltage-token-atlas':
        if args.analysis_output is None or args.report_output is None:raise ValueError('Explicit analysis and report destinations required')
        result=(voltage_token_shape_examples(cfg,out,args.token_source_analysis.resolve(),args.analysis_output.resolve(),args.report_output.resolve())
            if args.token_source_analysis else voltage_token_atlas(cfg,out,args.analysis_output.resolve(),args.report_output.resolve(),args.devices or ['cpu']))
        print(json.dumps(serial(result)));return
    if args.steps is not None and (args.stage not in ('train-one','voltage-one','eeg-o-one') or args.pilot_output is None):
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
        if args.stage not in ('train-one', 'masked-one', 'refine-one','voltage-one','eeg-o-one') or os.getppid() != args.parent_pid:
            raise ValueError('worker must be a direct child of its live controller')
        freeze_contract(out, cfg, root)
        if args.stage=='eeg-o-one':
            result=eeg_o_train(cfg,out,args.candidate,args.seed,args.device,args.gpu_memory_bytes,
                steps=args.steps,job_override=args.pilot_output,export=not args.no_export)
        elif args.stage=='voltage-one':
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
            if stage=='eeg-o': result=run_eeg_o(cfg,args.config.resolve(),out,root,devices,workers)
            elif stage=='eeg-o-one': result=eeg_o_train(cfg,out,args.candidate,args.seed,args.device,args.gpu_memory_bytes)
            elif stage=='voltage': result=run_voltage(cfg,args.config.resolve(),out,root,devices,gpu_workers,workers)
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
    residual=cfg['schema']=='dual_component_eeg_o_v1'
    if residual and final:raise ValueError('EEG-O development cannot launch test evaluation')
    job=Path(pilot) if pilot else Path(out)/('eeg_o_jobs' if residual else 'voltage_jobs')/name/f'seed_{seed}';job.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(Path(__file__).resolve()),'--config',str(cfg_path),'--project-root',str(root),
        '--output',str(out),'--stage','eeg-o-one' if residual else 'voltage-one','--candidate',name,'--seed',str(seed),'--device',device,
        '--gpu-memory-bytes',str(cfg['resources']['max_new_gpu_memory_mib']*2**20//workers),'--parent-pid',str(os.getpid())]
    if pilot:command.extend(['--pilot-output',str(pilot),'--steps',str(steps),'--no-export'])
    if final:command.extend(['--evaluation-split','test'])
    handle=(job/('test_process.log' if final else 'process.log')).open('a')
    return subprocess.Popen(command,cwd=SOURCE,stdout=handle,stderr=subprocess.STDOUT),handle,job


def voltage_jobs(cfg,cfg_path,out,root,devices,gpu_workers,pending,*,phase):
    pending=list(pending);planned=len(pending);results=[];running=[];slots=[d for d in devices for _ in range(gpu_workers)]
    prefix='eeg_o' if cfg['schema']=='dual_component_eeg_o_v1' else 'voltage'
    final=phase=='test';filename='test_result.json' if final else 'summary.json'
    while pending or running:
        while pending and len(running)<len(slots):
            name,seed=pending.pop(0);job=Path(out)/(prefix+'_jobs')/name/f'seed_{seed}'
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
            write_json(Path(out)/(prefix+'_training_progress.json'),dict(phase=phase,completed=len(results),planned=planned,
                active=[dict(candidate=x[2][0],seed=x[2][1],device=x[3],pid=x[0].pid) for x in running],last=result.get('status'),updated=now()))
    write_json(Path(out)/f'{prefix}_{phase}_jobs.json',dict(status='completed' if all(x['status']=='completed' for x in results) else 'failed',jobs=results))
    if any(x['status']!='completed' for x in results):raise RuntimeError(f'voltage {phase} job failed; inspect retained process logs')
    return results


def voltage_benchmark(cfg,cfg_path,out,root,devices):
    residual=cfg['schema']=='dual_component_eeg_o_v1'
    prefix='eeg_o' if residual else 'voltage'
    path=Path(out)/(prefix+'_gpu_benchmark.json')
    if path.exists():return read_json(path)['workers_per_device']
    benchmarks=[]
    for count in [1,2]:
        processes=[];started=time.monotonic()
        for i,device in enumerate(devices):
            for w in range(count):
                pilot=Path(out)/(prefix+'_benchmark')/f'workers_{count}'/f'device_{i}_worker_{w}'
                name=cfg['eeg_o']['candidates'][-1]['name'] if residual else 'eeg_voltage'
                processes.append(_launch_voltage(cfg,cfg_path,out,root,name,1700+i*2+w,device,count,pilot=pilot,steps=100))
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


def voltage_token_atlas_manifest(cfg,out):
    """Check the retained public boundary before opening observation arrays."""
    if cfg.get('schema')!='dual_component_voltage_v1' or cfg['data']['protected_data']!='forbidden':
        raise ValueError('Public voltage contract required')
    saved=read_json(out/'resolved_config.json')['config']
    if saved['voltage']!=cfg['voltage'] or saved['data']!=cfg['data']:raise ValueError('Retained voltage contract mismatch')
    if read_json(out/'voltage_summary.json').get('status')!='completed':raise ValueError('Completed tokenizer evaluation required')
    selection=read_json(out/'voltage_selection.json')
    if selection.get('selection_split')!='validation' or selection.get('test_used') is not False:raise ValueError('Frozen validation selection required')
    manifest=read_json(out/'voltage_test_cache/manifest.json')
    if manifest.get('schema')!='dual_component_voltage_cache_v1' or manifest.get('protected_data')!='forbidden' or manifest.get('final') is not True:
        raise ValueError('Previously evaluated public test cache required')
    for modality in ['eeg','hb']:
        rows=manifest['rows'][modality]
        if not rows or any(r['split']!='test' or r['dataset'] not in cfg['data']['datasets'] for r in rows):raise ValueError('Unexpected token-atlas population')
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate token-atlas case identity')
    return manifest,selection


def _voltage_token_export_examples(book,ids,raw,population_rows,spec,selected,folder):
    from src.analysis.dual_component_tokenizer import _csv
    details=[]
    for order,item in enumerate(selected):
        code=np.array(item['ids'],dtype='int16');case,position=np.where(np.all(ids==code,axis=-1))
        if len(case)!=item['occurrences']:raise ValueError('All-member token count mismatch')
        sample=position[:,None]*spec['hop_samples']+np.arange(spec['hop_samples'])[None]
        original=np.asarray(raw[case[:,None],sample,:]);decoded=book.decode(code).reshape(spec['hop_samples'],raw.shape[2])
        direct=sum(book.codewords[level][int(value)] for level,value in enumerate(code)).reshape(decoded.shape)
        if not np.array_equal(decoded,direct):raise ValueError('Independent integer-only prototype decoding mismatch')
        filename=f'example_{order+1:02d}.npz';write_npz(folder/filename,original=original,prototype=decoded,case_indices=case,patch_indices=position,ids=code)
        memberships=[dict(case_index=int(i),patch_index=int(j),token_start_s=float(j*spec['hop_samples']/spec['sample_rate_hz']),**population_rows[i]) for i,j in zip(case,position)]
        _csv(folder/f'example_{order+1:02d}_members.csv',memberships)
        dataset_counts=Counter(population_rows[i]['dataset'] for i in case);subjects={(population_rows[i]['dataset'],population_rows[i]['subject']) for i in case}
        details.append(dict(item,file=filename,subjects=len(subjects),dataset_counts=dict(dataset_counts),
            rows=int(len(set(case))),coordinates=len({population_rows[i]['coordinate'] for i in case}),
            raw_minus_S_RMS=float(np.sqrt(np.mean((original-decoded)**2))),
            display_all_members=True,independent_decoder_max_difference=0.))
    return details


def _voltage_token_atlas_assign(cfg,out,export,manifest,selection,modality,device):
    import torch
    from src.tokenizers.dual_component_tokenizer import DualComponentTokenizer
    folder=export/modality;folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():return read_json(folder/'summary.json')
    name=selection['eeg_candidate' if modality=='eeg' else 'hb_candidate'];seed=cfg['voltage']['seed']
    job=out/'voltage_jobs'/name/f'seed_{seed}';saved=torch.load(job/'checkpoint.pt',map_location='cpu',weights_only=False)
    if saved['step']!=saved['budget'] or read_json(job/'test_evaluation/summary.json').get('status')!='completed':raise ValueError('Completed selected model required')
    candidate=saved['candidate'];spec=cfg['voltage'][modality];levels=int(selection[modality+'_levels'])
    if candidate['span_samples']!=spec['hop_samples']:raise ValueError('Single fixed prototype requires the selected nonoverlap support')
    if device.startswith('cuda'):
        torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(3*2**30/torch.cuda.get_device_properties(device).total_memory,device)
        torch.cuda.reset_peak_memory_stats(device)
    model=DualComponentTokenizer(saved['model_config'],'continuous_student').to(device);model.load_state_dict(saved['model_state']);model.eval()
    book=_voltage_codebooks(job/'dictionaries.npz')['S']
    if book.levels<levels:raise ValueError('Complete selected S-code tuple required')
    raw=np.load(out/'voltage_test_cache'/manifest['arrays'][modality+'_u'],mmap_mode='r',allow_pickle=False)
    shape=tuple(manifest['shapes'][modality+'_u'])
    if raw.shape!=shape or shape[0]!=len(manifest['rows'][modality]):raise ValueError('Observation identity/shape mismatch')
    arrays={modality+'_u':raw};patches=shape[1]//spec['hop_samples'];dimensions=spec['hop_samples']*shape[2]
    if shape[1]%spec['hop_samples'] or book.dimension!=dimensions:raise ValueError('Full nonpadded patch support required')
    def infer(indices,check=False):
        with torch.inference_mode():
            inputs,masks,_,_=_voltage_batch(manifest,arrays,modality,indices,candidate,cfg,device,load_targets=False)
            words=model.branches[modality]['S'](inputs,masks)['waveform']
            if check:
                complete=model({modality:inputs},{modality:masks})[modality]['S']
                if not torch.equal(words,complete):raise ValueError('Independent S branch changed frozen full-model output')
            return book.encode(words.cpu().numpy(),levels=levels)
    benchmarks=[]
    for batch in [32,128]:
        indices=np.arange(min(batch,shape[0]));started=time.monotonic();result=infer(indices,check=batch==32)
        benchmarks.append(dict(batch_cases=len(indices),seconds=time.monotonic()-started,patches=result.shape[0]*patches))
    batch=max(benchmarks,key=lambda b:b['patches']/b['seconds'])['batch_cases']
    path=folder/'s_ids.npy';progress_path=folder/'progress.json';begin=0
    if path.exists():
        ids=np.load(path,mmap_mode='r+');progress=read_json(progress_path);begin=int(progress['completed_cases'])
        if ids.shape!=(shape[0],patches,levels) or progress['source_checkpoint']!=str(job/'checkpoint.pt'):raise ValueError('Assignment resume identity mismatch')
    else:
        ids=np.lib.format.open_memmap(path,mode='w+',dtype='int16',shape=(shape[0],patches,levels));ids[:]=-1;ids.flush()
        write_json(progress_path,dict(completed_cases=0,planned_cases=shape[0],source_checkpoint=str(job/'checkpoint.pt')))
    started=time.monotonic()
    for offset in range(begin,shape[0],batch):
        stop=min(offset+batch,shape[0]);ids[offset:stop]=infer(np.arange(offset,stop));ids.flush()
        write_json(progress_path,dict(completed_cases=stop,planned_cases=shape[0],source_checkpoint=str(job/'checkpoint.pt'),
            elapsed_s=time.monotonic()-started,patches_per_second=(stop-begin)*patches/max(time.monotonic()-started,1e-9),batch_cases=batch))
    if np.any(ids<0) or np.any(ids>=book.size):raise ValueError('Incomplete token assignment')
    from src.analysis.dual_component_tokenizer import token_tuple_catalog,select_token_tuple_examples
    catalog=token_tuple_catalog(ids,manifest['rows'][modality],size=book.size)
    write_npz(folder/'catalog.npz',**catalog)
    selected=select_token_tuple_examples(catalog,book,levels=levels,count=3)
    details=_voltage_token_export_examples(book,ids,raw,manifest['rows'][modality],spec,selected,folder)
    summary=dict(status='completed',modality=modality,source_checkpoint=str(job/'checkpoint.pt'),source_dictionary=str(job/'dictionaries.npz'),
        assignment_source=str(out/'voltage_test_cache/manifest.json'),cases=shape[0],patches_per_case=patches,total_patches=int(shape[0]*patches),
        levels=levels,codebook_size=book.size,unique_tuples=len(catalog['keys']),eligible_tuples=int(((catalog['counts']>=20)&(catalog['subject_counts']>=2)).sum()),
        selected=details,selection='Among 25 most frequent complete tuples occurring >=20 times in >=2 subjects: most frequent, then two greedy farthest decoded waveforms by RMS distance (amplitude and shape); no closeness-to-input selection',
        benchmarks=benchmarks,batch_cases=batch,device=device,seconds=time.monotonic()-started,
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.startswith('cuda') else 0,
        scope='all previously evaluated public-test cases; normalized raw observations u; no per-patch centering/scaling; complete S tuple; before cross-token Hb synthesis',
        hop_samples=spec['hop_samples'],sample_rate_hz=spec['sample_rate_hz'],channels=shape[2],new_fits=0)
    write_json(folder/'summary.json',summary);return summary


def voltage_token_atlas(cfg,out,export,report,devices):
    import shutil
    import torch
    from concurrent.futures import ThreadPoolExecutor
    if export.parent!=out or export==out or report.is_relative_to(out):raise ValueError('Direct-child analysis and separate report destinations required')
    if not os.environ.get('INVOCATION_ID'):raise ValueError('Durable supervisor required for complete token assignment replay')
    manifest,selection=voltage_token_atlas_manifest(cfg,out)
    export.mkdir(exist_ok=True);torch.set_num_threads(2)
    with owner_lock(export):
        provenance=dict(schema='voltage_S_token_correspondence_v1',source_run=str(out),seed=cfg['voltage']['seed'],
            modalities=['eeg','hb'],scope='all retained public-test cases',source_cache=str(out/'voltage_test_cache/manifest.json'),
            output_support=dict(eeg='100ms, 20 samples, 4 IDs',hb='2s, 20 HbO/HbR sample pairs, 2 IDs'),
            label='S decoder prototype, before Hb cross-token synthesis; original means observed input u, not S teacher target',
            protected_reads=0,raw_dataset_reads=0,new_fits=0)
        if (export/'source.json').exists() and read_json(export/'source.json')!=provenance:raise ValueError('Frozen analysis provenance changed')
        write_json(export/'source.json',provenance)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={m:pool.submit(_voltage_token_atlas_assign,cfg,out,export,manifest,selection,m,devices[i%len(devices)]) for i,m in enumerate(['eeg','hb'])}
            results={m:f.result() for m,f in futures.items()}
        summary=dict(status='completed',**provenance,results=results);write_json(export/'summary.json',summary)
        shutil.copy2(Path(__file__),export/'runner_source.py');shutil.copy2(SOURCE/'src/analysis/dual_component_tokenizer.py',export/'analysis_source.py')
        from src.analysis.dual_component_tokenizer import render_token_correspondence
        render_token_correspondence(export,report)
        return dict(status='completed',analysis=str(export),report=str(report),total_patches={m:r['total_patches'] for m,r in results.items()})


def voltage_token_shape_examples(cfg,out,source,export,report):
    """Display-only selection from immutable complete assignments; no encoder replay."""
    import shutil
    from src.analysis.dual_component_tokenizer import select_token_tuple_examples,render_token_correspondence
    manifest,selection=voltage_token_atlas_manifest(cfg,out);parent=read_json(source/'summary.json')
    if parent['status']!='completed' or Path(parent['source_run'])!=out or export.parent!=source or export.exists():raise ValueError('Completed assignment source and fresh child display analysis required')
    launch=read_json(source/'launch.json');command=launch['command'];frequency_report=command[command.index('--report-output')+1]
    export.mkdir();results={}
    for modality in ['eeg','hb']:
        folder=export/modality;folder.mkdir();old=parent['results'][modality]
        ids=np.load(source/modality/'s_ids.npy',mmap_mode='r',allow_pickle=False)
        raw=np.load(out/'voltage_test_cache'/manifest['arrays'][modality+'_u'],mmap_mode='r',allow_pickle=False)
        if ids.shape[:2]!=(len(raw),old['patches_per_case']) or ids.shape[2]!=old['levels']:raise ValueError('Complete source assignment shape required')
        with np.load(source/modality/'catalog.npz',allow_pickle=False) as a:catalog={k:a[k] for k in a.files}
        book=_voltage_codebooks(Path(old['source_dictionary']))['S'];selected=select_token_tuple_examples(catalog,book,levels=old['levels'],count=3,mode='shape')
        details=_voltage_token_export_examples(book,ids,raw,manifest['rows'][modality],cfg['voltage'][modality],selected,folder)
        result={k:old[k] for k in ['modality','cases','patches_per_case','total_patches','levels','codebook_size','unique_tuples','eligible_tuples','hop_samples','sample_rate_hz','channels','scope','source_checkpoint','source_dictionary']}
        result.update(status='completed',selected=details,assignment_summary=str(source/modality/'summary.json'),
            selection='Among complete tuples with >=20 members and >=2 subjects, take 25 largest time-centered decoded RMS prototypes; first largest RMS then two greedy farthest centered waveforms; display original levels unchanged; no raw-fit selection',new_fits=0,new_encoder_replays=0)
        write_json(folder/'summary.json',result);results[modality]=result
    summary={k:v for k,v in parent.items() if k!='results'};summary.update(results=results,selection_view='shape',assignment_source=str(source),frequency_report=frequency_report,new_encoder_replays=0)
    write_json(export/'summary.json',summary);shutil.copy2(Path(__file__),export/'runner_source.py');shutil.copy2(SOURCE/'src/analysis/dual_component_tokenizer.py',export/'analysis_source.py')
    render_token_correspondence(export,report);return dict(status='completed',analysis=str(export),report=str(report),new_encoder_replays=0)


def _eeg_o_parent(cfg, out, root=None):
    if root is None:
        root = Path(read_json(Path(out) / 'resolved_config.json')['project_root'])
    parent = (Path(root) / cfg['eeg_o']['parent_run']).resolve()
    if read_json(parent / 'voltage_summary.json').get('status') != 'completed':
        raise ValueError('completed retained voltage parent required')
    return parent


def _eeg_o_feature_worker(payload):
    from src.data.dual_component_tokenizer import eeg_residual_band_features
    source, destination, lo, hi = payload
    values = np.load(source, mmap_mode='r', allow_pickle=False)
    bands, features = eeg_residual_band_features(values[lo:hi, :, 0])
    if destination:
        for name, value in [('bands', bands), ('log_power', features)]:
            target = np.lib.format.open_memmap(Path(destination) / (name + '.npy'), mode='r+')
            target[lo:hi] = value
            target.flush()
    return dict(lo=lo, hi=hi, checksum=float(features.mean()),
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def eeg_o_prepare(cfg, out, root, workers):
    from src.data.dual_component_tokenizer import load_eeg_residual_development
    out = Path(out); parent = _eeg_o_parent(cfg, out, root)
    manifest, arrays, trials = load_eeg_residual_development(parent)
    folder = out / 'eeg_o_features'; folder.mkdir(exist_ok=True)
    path = folder / 'manifest.json'
    if path.exists():
        result = read_json(path)
        if result['source_parent'] != str(parent) or result['source_cases'] != len(manifest['rows']['eeg']):
            raise ValueError('residual feature source drift')
        return result
    if cfg['eeg_o'].get('feature_reuse_run'):
        reuse=(Path(root)/cfg['eeg_o']['feature_reuse_run']).resolve()
        if read_json(reuse/'eeg_o_summary.json')['status']!='completed':raise ValueError('completed residual source required')
        old_settings=read_json(reuse/'resolved_config.json')['config']['eeg_o']
        for key in ['parent_run','overfit_patches','overfit_steps','codebook_size','codebook_levels','codebook_fit_patches','codebook_iterations']:
            if cfg['eeg_o'][key]!=old_settings[key]:raise ValueError('residual diagnostic reuse contract drift')
        result=read_json(reuse/'eeg_o_features/manifest.json')
        if result['source_parent']!=str(parent) or result['trials']!=trials:raise ValueError('reuse identity mismatch')
        for name in ['bands.npy','log_power.npy']:
            destination=folder/name;original=reuse/'eeg_o_features'/name
            if destination.exists():
                if destination.resolve()!=original.resolve():raise ValueError('feature reuse path drift')
            else:destination.symlink_to(original)
        result.update(feature_reuse_run=str(reuse))
        if cfg['eeg_o'].get('architecture')=='projection_gain':
            result.update(eeg_o_prepare_projection(parent,folder,manifest,trials))
        write_json(path,result)
        write_json(out/'eeg_o_feature_progress.json',dict(status='completed',source_cases=len(manifest['rows']['eeg']),
            reused_carriers=str(reuse),new_fields=['observed_EEG_proxy','fixed_training_projection'],updated=now()))
        return result
    raw_path = parent / 'voltage_cache/eeg_u.npy'
    block = cfg['resources']['feature_block_cases']; n = len(arrays['eeg_u'])
    benchmark_path = out / 'eeg_o_cpu_benchmark.json'
    if benchmark_path.exists():
        selected_workers = read_json(benchmark_path)['workers']
    else:
        measurements = []
        for count in sorted({min(c, workers) for c in cfg['resources']['feature_benchmark_workers']}):
            if count > len(os.sched_getaffinity(0)): continue
            payloads = [(str(raw_path), None, lo, min(lo + block, min(n, 2048)))
                        for lo in range(0, min(n, 2048), block)]
            started = time.monotonic()
            results = _bounded_jobs(_eeg_o_feature_worker, payloads, count)
            if any('hi' not in r for r in results): raise RuntimeError('residual feature benchmark failed')
            elapsed = time.monotonic() - started
            measurements.append(dict(workers=count, seconds=elapsed, cases_per_second=min(n, 2048)/elapsed,
                                     peak_worker_rss_kib=max(r['peak_rss_kib'] for r in results)))
        selected_workers = max(measurements, key=lambda r: r['cases_per_second'])['workers']
        write_json(benchmark_path, dict(workers=selected_workers, measurements=measurements,
            reason='highest observed bounded feature throughput, including worker startup, on the same source rows'))
    for name, shape in [('bands', (n, 4000, 5)), ('log_power', (n, 200, 5))]:
        array_path = folder / (name + '.npy')
        if not array_path.exists():
            value = np.lib.format.open_memmap(array_path, mode='w+', dtype='float32', shape=shape);value.flush()
        elif np.load(array_path, mmap_mode='r', allow_pickle=False).shape != shape:
            raise ValueError('incomplete feature array shape mismatch')
    progress_path = out / 'eeg_o_feature_progress.json'
    done = set(read_json(progress_path).get('completed_blocks', [])) if progress_path.exists() else set()
    payloads = [(str(raw_path), str(folder), lo, min(lo + block, n)) for lo in range(0, n, block) if lo not in done]
    def progress(result, completed):
        if 'lo' not in result:
            write_json(progress_path, dict(status='failed', completed_blocks=sorted(done), failure=result))
            raise RuntimeError('residual feature worker failed; retained partial cache')
        done.add(result['lo'])
        if completed % 16 == 0 or completed == len(payloads):
            write_json(progress_path, dict(status='preparing', completed_blocks=sorted(done),
                completed_cases=sum(min(block, n-lo) for lo in done), planned_cases=n, workers=selected_workers, updated=now()))
    _bounded_jobs(_eeg_o_feature_worker, payloads, selected_workers, progress=progress)
    features = np.load(folder / 'log_power.npy', mmap_mode='r', allow_pickle=False)
    rows = manifest['rows']['eeg']; coordinates = manifest['coordinates']['eeg']
    means, scales = [], []
    for coordinate in coordinates:
        ix = [i for i, row in enumerate(rows) if row['split']=='train' and row['coordinate']==coordinate]
        if not ix: raise ValueError('coordinate lacks training support')
        value = np.asarray(features[ix], dtype=float)
        means.append(value.mean((0,1)).tolist());scales.append(np.maximum(value.std((0,1)), 1e-3).tolist())
    result = dict(schema='eeg_o_band_features_v1', source_parent=str(parent), source_cases=n,
        source_manifest=str(parent / 'voltage_cache/manifest.json'), trials=trials,
        coordinates=coordinates, feature_mean=means, feature_scale=scales,
        scale_fit='training subjects only, per dataset/electrode/band',
        raw_support='full retained 20s offline EEG for zero-phase filtering',
        planned_trials=manifest.get('planned_trials'), retained_trials=manifest.get('retained_trials'),
        source_failures_preserved=True, test_used=False, feature_workers=selected_workers)
    write_json(path, result)
    write_json(progress_path, dict(status='completed', completed_blocks=sorted(done),
        completed_cases=n, planned_cases=n, workers=selected_workers, updated=now()))
    return result


def eeg_o_prepare_projection(parent, folder, manifest, trials):
    from src.data.dual_component_tokenizer import load_retained_eeg_projection
    datasets=sorted({r['dataset'] for r in trials});maps={};values=[];dataset_ids=[]
    rows=manifest['rows']['eeg']
    for trial in trials:
        status=task_status(parent,'voltage_prepared',trial['trial_id'])
        if status.get('status')!='completed':raise ValueError('unavailable retained observed proxy')
        expected=[rows[i] for i in trial['indices']]
        observed,mapping=load_retained_eeg_projection(array_paths(parent,'voltage_prepared',dict(id=trial['trial_id'])),status['metadata'],expected)
        dataset=trial['dataset']
        if dataset in maps and not np.allclose(maps[dataset],mapping,rtol=0,atol=1e-7):
            raise ValueError('projection must remain the same frozen training transform within a dataset')
        maps[dataset]=mapping;values.append(observed);dataset_ids.append(datasets.index(dataset))
    values=np.asarray(values,dtype='float32');ids=np.asarray(dataset_ids,dtype='int64')
    means=[];scales=[]
    for dataset in datasets:
        ix=[i for i,r in enumerate(trials) if r['dataset']==dataset and r['split']=='train']
        training=np.asarray(values[ix],float)
        means.append(training.mean((0,1)).tolist());scales.append(np.maximum(training.std((0,1)),1e-3).tolist())
    np.save(Path(folder)/'observed_proxy.npy',values);np.save(Path(folder)/'proxy_dataset_ids.npy',ids)
    return dict(proxy_mean=means,proxy_scale=scales,gain_map=[maps[d].tolist() for d in datasets],proxy_datasets=datasets,
        proxy_source_fields=['original_proxy','projection'],
        raw_support='observed EEG proxy inherits original 90s EEG filtering and [-30,-25]s reference; central20s proxy/gain input; no Hb or teacher trajectory input')


def _eeg_o_data(cfg, out):
    from src.data.dual_component_tokenizer import load_eeg_residual_development
    parent = _eeg_o_parent(cfg, out)
    manifest, arrays, trials = load_eeg_residual_development(parent)
    features = read_json(Path(out) / 'eeg_o_features/manifest.json')
    if features['source_parent'] != str(parent) or features['trials'] != trials:
        raise ValueError('residual feature/target identity mismatch')
    arrays.update({key: np.load(Path(out) / 'eeg_o_features' / (key + '.npy'), mmap_mode='r', allow_pickle=False)
                   for key in ('bands', 'log_power')})
    if cfg['eeg_o'].get('architecture')=='projection_gain':
        arrays.update({key:np.load(Path(out)/'eeg_o_features'/(key+'.npy'),mmap_mode='r',allow_pickle=False)
                       for key in ('observed_proxy','proxy_dataset_ids')})
    return parent, manifest, arrays, features


def _eeg_o_batch(arrays, manifest, trial_indices, trials, device):
    import torch
    cases = np.asarray([trials[i]['indices'] for i in trial_indices])
    coordinates = np.asarray([[manifest['rows']['eeg'][i]['coordinate_index'] for i in group] for group in cases])
    carriers = torch.as_tensor(np.asarray(arrays['bands'][cases]), device=device)
    features = (None if 'observed_proxy' in arrays else
                torch.as_tensor(np.asarray(arrays['log_power'][cases]), device=device))
    return cases, carriers, features, torch.as_tensor(coordinates, dtype=torch.long, device=device)


def _eeg_o_sample_groups(trials):
    groups = defaultdict(lambda: defaultdict(list))
    for i, trial in enumerate(trials):
        if trial['split']=='train':groups[trial['dataset']][trial['subject']].append(i)
    return {d: list(subjects.values()) for d, subjects in sorted(groups.items())}


def _eeg_o_sample(groups, count, rng):
    datasets = list(groups); chosen = []
    for _ in range(count):
        subjects = groups[datasets[int(rng.integers(len(datasets)))]]
        trials = subjects[int(rng.integers(len(subjects)))]
        chosen.append(trials[int(rng.integers(len(trials)))])
    return chosen


def _eeg_o_prediction(model, carriers, power, coordinates, arrays, trial_indices, device):
    import torch
    from src.tokenizers.dual_component_tokenizer import ProjectionGainResidualPredictor
    if isinstance(model,ProjectionGainResidualPredictor):
        proxy=torch.as_tensor(np.asarray(arrays['observed_proxy'][trial_indices]),device=device)
        dataset_ids=torch.as_tensor(np.asarray(arrays['proxy_dataset_ids'][trial_indices]),device=device,dtype=torch.long)
        return model(carriers,proxy,dataset_ids)['O']
    return model(carriers,power,coordinates)['O']


def eeg_o_train(cfg, out, name, seed, device='cpu', memory_bytes=None, *, steps=None, job_override=None, export=True):
    import torch
    from src.tokenizers.dual_component_tokenizer import BandGainResidualPredictor, ProjectionGainResidualPredictor, WaveformDictionary
    settings = cfg['eeg_o'];candidate = next(c for c in settings['candidates'] if c['name']==name)
    job = Path(job_override) if job_override else Path(out) / 'eeg_o_jobs' / name / f'seed_{seed}'
    job.mkdir(parents=True, exist_ok=True)
    if (job/'summary.json').exists():return read_json(job/'summary.json')
    with owner_lock(job, 'training'):
        torch.set_num_threads(cfg['resources']['torch_threads']);torch.manual_seed(seed)
        if device.startswith('cuda'):
            torch.cuda.set_device(device)
            cap=memory_bytes or cfg['resources']['max_new_gpu_memory_mib']*2**20
            torch.cuda.set_per_process_memory_fraction(max(cap-512*2**20,64*2**20)/torch.cuda.get_device_properties(device).total_memory,device)
            torch.cuda.reset_peak_memory_stats(device)
        parent, manifest, arrays, features = _eeg_o_data(cfg, out)
        trials = features['trials'];groups = _eeg_o_sample_groups(trials);rng=np.random.default_rng(seed)
        model_config = dict(channels=settings['channels'], width=settings['width'], context_bins=candidate['context_bins'],
            max_log_gain=settings['maximum_log_gain'], hop=settings['hop_samples'])
        if settings.get('architecture')=='projection_gain':
            model_config.update(proxy_mean=features['proxy_mean'],proxy_scale=features['proxy_scale'],gain_map=features['gain_map'])
            model=ProjectionGainResidualPredictor(**model_config).to(device)
        else:
            model_config.update(feature_mean=features['feature_mean'],feature_scale=features['feature_scale'],joint=candidate['joint'])
            model=BandGainResidualPredictor(**model_config).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
        budget=int(steps or settings['steps']);checkpoint=job/'checkpoint.pt';start=0;history=[];prior_elapsed=0.
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location=device,weights_only=False)
            if saved['candidate']!=candidate or saved['seed']!=seed or saved['budget']!=budget or saved['model_config']!=model_config:
                raise ValueError('EEG-O checkpoint contract drift')
            model.load_state_dict(saved['model_state']);optimizer.load_state_dict(saved['optimizer_state'])
            start=saved['step'];history=saved['history'];rng.bit_generator.state=saved['rng_state']
            prior_elapsed=saved['elapsed_seconds']
        if start>budget:raise ValueError('checkpoint exceeded fixed EEG-O budget')
        variance=manifest['statistics']['eeg_o']['variance'];delta_scale=manifest['statistics']['eeg_o']['derivative_mse']
        started=time.monotonic();model.train()
        for step in range(start,budget):
            selected=_eeg_o_sample(groups,settings['batch_trials'],rng)
            cases,carriers,power,coordinates=_eeg_o_batch(arrays,manifest,selected,trials,device)
            target=torch.as_tensor(np.asarray(arrays['eeg_o'][cases])[...,0],device=device)
            prediction=_eeg_o_prediction(model,carriers,power,coordinates,arrays,selected,device);error=prediction-target
            mse=error.square().mean();derivative=error.diff(dim=-1).square().mean()
            loss=mse/variance+settings['derivative_weight']*derivative/delta_scale
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite EEG-O objective')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            if (step+1)%settings['checkpoint_every']==0 or step+1==budget:
                elapsed=prior_elapsed+time.monotonic()-started
                history.append(dict(step=step+1,loss=float(loss.detach()),MSE=float(mse.detach()),derivative_error=float(derivative.detach())))
                _torch_save(checkpoint,dict(candidate=candidate,model_config=model_config,seed=seed,step=step+1,budget=budget,
                    model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),history=history,
                    elapsed_seconds=elapsed,rng_state=rng.bit_generator.state))
                write_json(job/'progress.json',dict(status='training',step=step+1,budget=budget,elapsed_seconds=elapsed,last=history[-1]))
        training_elapsed=prior_elapsed+time.monotonic()-started;model.eval();evaluation=None
        if export:
            dictionary_path=job/'dictionary.npz'
            if dictionary_path.exists():
                with np.load(dictionary_path,allow_pickle=False) as z:book=WaveformDictionary.from_state(dict(z))
            else:
                fit_rng=np.random.default_rng(seed+101)
                fit_trials=_eeg_o_sample(groups,max(32,2*len(features['coordinates'])),fit_rng)
                samples=[]
                with torch.inference_mode():
                    for begin in range(0,len(fit_trials),settings['batch_trials']):
                        cases,carriers,power,coordinates=_eeg_o_batch(arrays,manifest,fit_trials[begin:begin+settings['batch_trials']],trials,device)
                        predicted=_eeg_o_prediction(model,carriers,power,coordinates,arrays,fit_trials[begin:begin+settings['batch_trials']],device).cpu().numpy()
                        samples.append(predicted.reshape(-1,settings['hop_samples']))
                book=WaveformDictionary(settings['codebook_size'],settings['codebook_levels'],1,0.,seed,
                    settings['codebook_fit_patches'],settings['codebook_iterations']).fit(np.concatenate(samples))
                write_npz(dictionary_path,**book.state())
            evaluation=eeg_o_evaluate(cfg,out,job,model,book,seed,device)
        result=dict(status='completed',candidate=candidate,seed=seed,step=budget,budget=budget,
            model_parameters=sum(p.numel() for p in model.parameters()),training_elapsed_seconds=training_elapsed,
            elapsed_seconds=time.monotonic()-started+prior_elapsed,steps_per_second=budget/max(training_elapsed,1e-9),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.startswith('cuda') else 0,
            evaluation=evaluation,test_used=False)
        write_json(job/'summary.json',result)
        write_json(job/'progress.json',dict(status='completed',step=budget,budget=budget,elapsed_seconds=result['elapsed_seconds']))
        return result


def eeg_o_synthetic(cfg):
    import torch
    from src.tokenizers.dual_component_tokenizer import BandGainResidualPredictor, ProjectionGainResidualPredictor, WaveformDictionary
    torch.set_num_threads(1);torch.manual_seed(17)
    t=np.arange(800)/200
    carriers=np.stack([np.sin(2*np.pi*f*t) for f in (2,6,10,20,35)],-1).astype('float32')[None,None]
    power=np.log(np.maximum((carriers.reshape(1,1,40,20,5)**2).mean(3),1e-12))
    projection=cfg['eeg_o'].get('architecture')=='projection_gain'
    if projection:
        mapping=np.zeros((1,5,3),dtype='float32');mapping[0,0,0]=1
        model=ProjectionGainResidualPredictor([[0.]*3],[[1.]*3],mapping,channels=1,width=8,context_bins=19)
        features=torch.zeros(1,40,3);coordinates=torch.zeros(1,dtype=torch.long)
    else:
        model=BandGainResidualPredictor([[0.]*5],[[1.]*5],channels=1,width=8,context_bins=19)
        features=torch.tensor(power);coordinates=torch.zeros((1,1),dtype=torch.long)
    x=torch.tensor(carriers)
    zero=model(x,features,coordinates)['O']
    if torch.count_nonzero(zero):raise AssertionError('zero gain must be exact zero residual')
    coefficient=.1
    with torch.no_grad():
        model.network[-1].bias[0]=float(2*np.log(1-coefficient) if projection else np.arctanh(np.log(1-coefficient)/.7))
    result=model(x,features,coordinates);expected=coefficient*x[...,0]
    error=float(torch.max(abs(result['O']-expected)).detach())
    result['O'].square().mean().backward()
    if error>1e-6 or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise AssertionError('band-gain synthetic identity or gradient failed')
    rng=np.random.default_rng(17);waves=(rng.normal(size=(256,1))*np.sin(np.arange(20)[None]*.1)).astype('float32')
    book=WaveformDictionary(16,2,max_iter=10).fit(waves)
    restored=WaveformDictionary.from_state(book.state());ids=book.encode(waves)
    if not np.array_equal(book.decode(ids),restored.decode(ids)):raise AssertionError('integer reload mismatch')
    return dict(status='completed',experiment_id=cfg['experiment_id'],measured_arrays_read=0,
        analytical_max_abs_error=error,zero_gain_identity=True,finite_gradients=True,integer_reload_exact=True)


def eeg_o_overfit(cfg, out, device):
    import torch
    from src.tokenizers.dual_component_tokenizer import DualComponentTokenizer
    path=Path(out)/'overfit.json'
    if path.exists():return read_json(path)
    parent,manifest,arrays,features=_eeg_o_data(cfg,out);spec=cfg['eeg_o'];seed=spec['seed']
    torch.set_num_threads(cfg['resources']['torch_threads']);torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction((cfg['resources']['max_new_gpu_memory_mib']-512)*2**20/torch.cuda.get_device_properties(device).total_memory,device)
    rng=np.random.default_rng(seed);indices=np.array([i for i,r in enumerate(manifest['rows']['eeg']) if r['split']=='train'])
    indices=rng.choice(indices,size=spec['overfit_patches'],replace=False);positions=rng.integers(0,200,len(indices))
    candidate=next(c for c in cfg['voltage']['candidates'] if c['name']=='eeg_voltage')
    inputs,masks,targets,target_masks=_voltage_batch(manifest,arrays,'eeg',indices,candidate,cfg,device,positions)
    config=dict(input_dims={'eeg':inputs.shape[-1]},target_dims={'eeg':20},width=128,latent_dim=64,codebook_size=128)
    branch=DualComponentTokenizer(config,'continuous_student').branches['eeg']['O'].to(device)
    optimizer=torch.optim.AdamW(branch.parameters(),lr=.001,weight_decay=0.)
    truth=targets['O'];mask=target_masks['O'];zero=float((truth.square()*mask).sum().cpu()/mask.sum().cpu())
    history=[];started=time.monotonic()
    for step in range(spec['overfit_steps']):
        prediction=branch(inputs,masks)['waveform'];mse=((prediction-truth).square()*mask).sum()/mask.sum()
        loss=mse/manifest['statistics']['eeg_o']['variance']
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if (step+1)%250==0 or step+1==spec['overfit_steps']:
            history.append(dict(step=step+1,MSE=float(mse.detach()),zero_prediction_skill=1-float(mse.detach())/zero))
            write_json(Path(out)/'overfit_progress.json',dict(step=step+1,budget=spec['overfit_steps'],last=history[-1]))
    with torch.inference_mode():pred=branch(inputs,masks)['waveform'].cpu().numpy().ravel()
    target=truth.cpu().numpy().ravel();mse=float(np.mean((pred-target)**2))
    result=dict(status='completed',model='same-size original O MLP, fixed training patches only',
        patches=len(indices),steps=spec['overfit_steps'],MSE=mse,zero_MSE=zero,zero_prediction_skill=1-mse/zero,
        correlation=float(np.corrcoef(pred,target)[0,1]),RMS_ratio=float(np.sqrt(np.mean(pred**2)/np.mean(target**2))),
        elapsed_seconds=time.monotonic()-started,history=history,
        samples=[dict(id=manifest['rows']['eeg'][i]['id'],patch=int(p)) for i,p in zip(indices,positions)],
        interpretation='training capacity check only; no validation or deployable improvement claim')
    write_json(path,result)
    del branch,optimizer,inputs,masks,targets,truth
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return result


def eeg_o_oracle(cfg, out):
    from src.tokenizers.dual_component_tokenizer import WaveformDictionary
    from src.analysis.dual_component_tokenizer import eeg_residual_case_metrics,eeg_residual_aggregate,_csv
    folder=Path(out)/'oracle';folder.mkdir(exist_ok=True);path=folder/'summary.json'
    if path.exists():return read_json(path)
    parent,manifest,arrays,features=_eeg_o_data(cfg,out);settings=cfg['eeg_o'];rng=np.random.default_rng(settings['seed'])
    rows=manifest['rows']['eeg'];train=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    fit=rng.choice(train,size=min(512,len(train)),replace=False)
    samples=np.asarray(arrays['eeg_o'][fit]).reshape(-1,20)
    book=WaveformDictionary(settings['codebook_size'],settings['codebook_levels'],1,0.,settings['seed'],
        settings['codebook_fit_patches'],settings['codebook_iterations']).fit(samples)
    write_npz(folder/'dictionary.npz',**book.state())
    validation=[i for i,r in enumerate(rows) if r['split']=='validation'];collected=defaultdict(list)
    for start in range(0,len(validation),128):
        indices=validation[start:start+128];truth=np.asarray(arrays['eeg_o'][indices])[:,:,0]
        ids=book.encode(truth.reshape(len(truth),200,20));pred=book.decode(ids).reshape(truth.shape)
        for k,v in eeg_residual_case_metrics(pred,truth).items():collected[k].append(v)
    values={k:np.concatenate(v) for k,v in collected.items()}
    subjects,metrics=eeg_residual_aggregate(values,[rows[i] for i in validation]);_csv(folder/'subjects.csv',subjects)
    result=dict(status='completed',diagnostic_only=True,oracle_reads_O_target=True,metrics=metrics,
        bits_per_O_second_per_electrode=280,training_cases=len(fit),validation_cases=len(validation),test_used=False,
        interpretation='dictionary fitted only to training O targets; direct target encoding is a capacity reference, never a deployable student')
    write_json(path,result);return result


def eeg_o_evaluate(cfg, out, job, model, book, seed, device):
    import torch
    from src.tokenizers.dual_component_tokenizer import DualComponentTokenizer
    from src.analysis.dual_component_tokenizer import eeg_residual_case_metrics,eeg_residual_aggregate,_csv
    path=Path(job)/'evaluation.json'
    if path.exists():return read_json(path)
    parent,manifest,arrays,features=_eeg_o_data(cfg,out);settings=cfg['eeg_o'];trials=features['trials']
    selected=[i for i,r in enumerate(trials) if r['split']=='validation']
    baseline_folder=parent/'voltage_jobs/eeg_voltage'/f'seed_{seed}'
    saved=torch.load(baseline_folder/'checkpoint.pt',map_location=device,weights_only=False)
    baseline=DualComponentTokenizer(saved['model_config'],'continuous_student').to(device)
    baseline.load_state_dict(saved['model_state']);baseline.eval();baseline_book=_voltage_codebooks(baseline_folder/'dictionaries.npz')['S']
    values={name:defaultdict(list) for name in ('continuous','hard')};case_order=[]
    counts=np.zeros((book.levels,book.size),dtype='int64');replay_error=0.
    for begin in range(0,len(selected),settings['batch_trials']):
        with torch.inference_mode():
            indices,carriers,power,coordinates=_eeg_o_batch(arrays,manifest,selected[begin:begin+settings['batch_trials']],trials,device)
            prediction=_eeg_o_prediction(model,carriers,power,coordinates,arrays,selected[begin:begin+settings['batch_trials']],device).cpu().numpy().reshape(-1,4000)
            ix=indices.ravel();truth=np.asarray(arrays['eeg_o'][ix])[:,:,0]
            inputs,masks,_,_=_voltage_batch(manifest,arrays,'eeg',ix,saved['candidate'],cfg,device,load_targets=False)
            sw=baseline.branches['eeg']['S'](inputs,masks)['waveform'].cpu().numpy()
        frozen_s=baseline_book.decode(baseline_book.encode(sw)).reshape(-1,4000)
        ids=book.encode(prediction.reshape(-1,200,20));hard=book.decode(ids).reshape(-1,4000)
        # Decoder state round-trip is checked on the complete batch, with no inputs.
        from src.tokenizers.dual_component_tokenizer import WaveformDictionary
        independent=WaveformDictionary.from_state(book.state()).decode(ids).reshape(hard.shape)
        replay_error=max(replay_error,float(np.max(abs(independent-hard))))
        for level in range(book.levels):counts[level]+=np.bincount(ids[...,level].ravel(),minlength=book.size)
        for name,pred in [('continuous',prediction),('hard',hard)]:
            metrics=eeg_residual_case_metrics(pred,truth)
            metrics['frozen_S_plus_O_NMSE']=np.mean((frozen_s+pred-np.asarray(arrays['eeg_u'][ix])[:,:,0])**2,axis=1)
            for key,value in metrics.items():values[name][key].append(value)
        case_order.extend(ix.tolist())
        write_json(Path(job)/'evaluation_progress.json',dict(completed_trials=min(begin+settings['batch_trials'],len(selected)),
            planned_trials=len(selected),updated=now()))
    reports={};all_subjects=[];metrics_to_save={}
    for name,terms in values.items():
        terms={k:np.concatenate(v) for k,v in terms.items()}
        subjects,metrics=eeg_residual_aggregate(terms,[manifest['rows']['eeg'][i] for i in case_order])
        all_subjects.extend(dict(r,variant=name) for r in subjects);reports[name]=metrics
        metrics_to_save.update({name+'_'+k:v for k,v in terms.items()})
    _csv(Path(job)/'subjects.csv',all_subjects);write_npz(Path(job)/'case_metrics.npz',row_index=np.asarray(case_order),**metrics_to_save)
    health=[]
    for level,count in enumerate(counts):
        p=count/count.sum();positive=p>0
        health.append(dict(level=level,active_codes=int(positive.sum()),size=book.size,
            perplexity=float(np.exp(-np.sum(p[positive]*np.log(p[positive])))),max_probability=float(p.max()),code_counts=count.tolist()))
    old=read_json(baseline_folder/'evaluation/summary.json')['variants']['vq4']['metrics']
    result=dict(status='completed',metrics=reports,codebook_health=health,validation_cases=len(case_order),
        bits_per_O_second_per_electrode=280,integer_reload_max_abs=replay_error,
        baseline=[r for r in old if r['split']=='validation' and r['signal']=='joint' and r['metric'] in ('NMSE','centered_correlation')],
        S_branch='frozen original voltage S dictionary with the matching seed',
        input_support=features['raw_support'],test_used=False)
    if replay_error!=0:raise AssertionError('independent O integer decoder mismatch')
    write_json(path,result);return result


def select_eeg_o(results):
    ranked=[]
    for result in results:
        if result['status']!='completed':continue
        rows=result['evaluation']['metrics']['hard']
        if not rows or any(r['split']!='validation' for r in rows):raise ValueError('EEG-O selection requires validation only')
        ratios=[r['NMSE']/r['zero_NMSE'] for r in rows]
        ranked.append(dict(candidate=result['candidate']['name'],worst_ratio=max(ratios),mean_ratio=float(np.mean(ratios))))
    if not ranked:raise ValueError('no completed residual candidates')
    winner=min(ranked,key=lambda r:(r['worst_ratio'],r['mean_ratio'],r['candidate']))
    return dict(candidate=winner['candidate'],ranked=sorted(ranked,key=lambda r:r['worst_ratio']),
                selection_split='validation',test_used=False,selected_at=now())


def run_eeg_o(cfg, cfg_path, out, root, devices, workers):
    settings=cfg['eeg_o'];out=Path(out)
    synthetic_path=out/'synthetic.json'
    if not synthetic_path.exists():write_json(synthetic_path,eeg_o_synthetic(cfg))
    write_json(out/'eeg_o_stage.json',dict(stage='features',updated=now()))
    prepare=eeg_o_prepare(cfg,out,root,workers)
    write_json(out/'eeg_o_stage.json',dict(stage='benchmark',updated=now()))
    concurrency=voltage_benchmark(cfg,cfg_path,out,root,devices)
    # CPU dictionary reference and GPU memorization check are independent.
    write_json(out/'eeg_o_stage.json',dict(stage='diagnostics',updated=now()))
    if settings.get('feature_reuse_run'):
        reuse=(Path(root)/settings['feature_reuse_run']).resolve()
        for name in ['overfit.json','oracle']:
            destination=out/name
            if not destination.exists():destination.symlink_to(reuse/name,target_is_directory=name=='oracle')
        overfit=read_json(out/'overfit.json');oracle=read_json(out/'oracle/summary.json')
    else:
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            future=pool.submit(eeg_o_oracle,cfg,out)
            overfit=eeg_o_overfit(cfg,out,devices[0]);oracle=future.result()
    write_json(out/'eeg_o_diagnostics.json',dict(status='completed',overfit=overfit,oracle=oracle))
    write_json(out/'eeg_o_stage.json',dict(stage='screen',updated=now()))
    screen=voltage_jobs(cfg,cfg_path,out,root,devices,concurrency,
        [(c['name'],settings['seed']) for c in settings['candidates']],phase='screen')
    selection_path=out/'eeg_o_selection.json'
    if not selection_path.exists():write_json(selection_path,select_eeg_o(screen))
    selection=read_json(selection_path)
    write_json(out/'eeg_o_stage.json',dict(stage='replication',updated=now()))
    repetitions=voltage_jobs(cfg,cfg_path,out,root,devices,concurrency,
        [(selection['candidate'],seed) for seed in settings['confirmation_seeds']],phase='replication')
    selected=[r for r in screen if r['candidate']['name']==selection['candidate']]+repetitions
    summary=dict(status='completed',experiment_id=cfg['experiment_id'],selection=selection,
        selected_results=selected,diagnostics=dict(overfit='overfit.json',oracle='oracle/summary.json'),
        planned_trials=prepare['planned_trials'],retained_trials=prepare['retained_trials'],
        source_parent=prepare['source_parent'],ssms_refitted=0,test_used=False,
        interpretation='exploratory fixed-target EEG O recovery; unchanged validation subjects across optimization seeds')
    write_json(out/'eeg_o_summary.json',summary)
    write_json(out/'eeg_o_stage.json',dict(stage='completed',updated=now()))
    return summary


def render_eeg_o_examples(cfg, out, destination):
    """Frozen-model validation examples selected by target amplitude, not gain."""
    import torch
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from src.tokenizers.dual_component_tokenizer import BandGainResidualPredictor,ProjectionGainResidualPredictor,WaveformDictionary,DualComponentTokenizer
    from src.analysis.dual_component_tokenizer import eeg_residual_case_metrics
    out,destination=Path(out),Path(destination);destination.mkdir(parents=True,exist_ok=True)
    result=read_json(out/'eeg_o_summary.json')
    if result['status']!='completed':raise ValueError('completed residual run required')
    parent,manifest,arrays,features=_eeg_o_data(cfg,out);seed=cfg['eeg_o']['seed'];device='cpu'
    torch.set_num_threads(2)
    name=result['selection']['candidate'];job=out/'eeg_o_jobs'/name/f'seed_{seed}'
    saved=torch.load(job/'checkpoint.pt',map_location=device,weights_only=False)
    cls=ProjectionGainResidualPredictor if 'proxy_mean' in saved['model_config'] else BandGainResidualPredictor
    model=cls(**saved['model_config']);model.load_state_dict(saved['model_state']);model.eval()
    with np.load(job/'dictionary.npz',allow_pickle=False) as z:book=WaveformDictionary.from_state(dict(z))
    old_folder=parent/'voltage_jobs/eeg_voltage'/f'seed_{seed}'
    old_saved=torch.load(old_folder/'checkpoint.pt',map_location=device,weights_only=False)
    old_model=DualComponentTokenizer(old_saved['model_config'],'continuous_student')
    old_model.load_state_dict(old_saved['model_state']);old_model.eval();old_book=_voltage_codebooks(old_folder/'dictionaries.npz')['O']
    with np.load(old_folder/'evaluation/case_metrics.npz',allow_pickle=False) as z:energy=z['identity'][:,1,0,0].copy()
    scales=read_json(parent/'voltage_scales.json')['eeg'];rows=manifest['rows']['eeg'];trials=features['trials']
    trial_lookup={r['trial_id']:i for i,r in enumerate(trials)};examples=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        population=[i for i,r in enumerate(rows) if r['dataset']==dataset and r['split']=='validation'];used=set()
        for label,quantile in [('p10',.1),('median',.5),('p90',.9)]:
            threshold=float(np.quantile(np.sqrt(energy[population]),quantile))
            eligible=[i for i in population if rows[i]['subject'] not in used]
            index=min(eligible,key=lambda i:abs(np.sqrt(energy[i])-threshold));row=rows[index];used.add(row['subject'])
            trial_index=trial_lookup[row['trial_id']];within=trials[trial_index]['indices'].index(index)
            with torch.inference_mode():
                cases,carriers,power,coordinates=_eeg_o_batch(arrays,manifest,[trial_index],trials,device)
                continuous=_eeg_o_prediction(model,carriers,power,coordinates,arrays,[trial_index],device).cpu().numpy()[0,within]
                inputs,masks,_,_=_voltage_batch(manifest,arrays,'eeg',[index],old_saved['candidate'],cfg,device,load_targets=False)
                old_words=old_model.branches['eeg']['O'](inputs,masks)['waveform'].cpu().numpy()
            original=old_book.decode(old_book.encode(old_words)).reshape(1,4000)
            hard=book.decode(book.encode(continuous.reshape(1,200,20))).reshape(1,4000)
            truth=np.asarray(arrays['eeg_o'][index])[:,0][None]
            old_metrics=eeg_residual_case_metrics(original,truth);new_metrics=eeg_residual_case_metrics(hard,truth)
            def info(metrics):
                return dict(NMSE=float(metrics['NMSE'][0]),correlation=float(metrics['centered_correlation'][0]),
                    RMS_ratio=float(np.sqrt(metrics['prediction_power'][0]/metrics['zero_NMSE'][0])),
                    zero_prediction_skill=1-float(metrics['NMSE'][0]/metrics['zero_NMSE'][0]))
            old_info,new_info=info(old_metrics),info(new_metrics);factor=float(scales[row['coordinate']]);t=np.arange(4000)/200
            fig,axes=plt.subplots(2,1,figsize=(14,7.5));fig.subplots_adjust(left=.07,right=.985,bottom=.22,top=.79,hspace=.42)
            for ax,lo,hi in [(axes[0],0,4000),(axes[1],1600,2400)]:
                ax.plot(t[lo:hi],truth[0,lo:hi]*factor,color='black',lw=1.6,label='Fixed O target')
                ax.plot(t[lo:hi],original[0,lo:hi]*factor,color='#888888',ls='--',lw=1.2,label='Original hard O')
                ax.plot(t[lo:hi],continuous[lo:hi]*factor,color='#d55e00',ls=':',lw=1.2,label='New before quantization')
                ax.plot(t[lo:hi],hard[0,lo:hi]*factor,color='#0072b2',lw=1.3,label='New hard O')
                ax.set_ylabel('Voltage correction residual (µV)');ax.set_xlabel('Event time (s)');ax.grid(alpha=.2)
            axes[1].set_ylim(axes[0].get_ylim())
            axes[0].set_title('Full 20 s scored support; amplitude scale is specific to this example')
            axes[1].set_title('Fixed 8–12 s zoom; shared amplitude scale')
            handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.86),ncol=4,frameon=False)
            title='Single-Trial' if dataset=='eeg_fnirs_single_trial' else 'Simultaneous'
            fig.suptitle(f'{title} | {row["subject"]} / {row["record"]} / {row["trial_id"].split("__")[-1]} / {row["channel"]}\n'
                f'{label} target-RMS example | validation | seed 17 | O bitrate 280 bit/s/electrode',fontsize=13,y=.985)
            fig.text(.07,.105,f'Full 20 s old → new: correlation {old_info["correlation"]:.3f} → {new_info["correlation"]:.3f}; '
                f'RMS ratio {old_info["RMS_ratio"]:.3f} → {new_info["RMS_ratio"]:.3f}; '
                f'MSE / zero-MSE {1-old_info["zero_prediction_skill"]:.3f} → {1-new_info["zero_prediction_skill"]:.3f}.',fontsize=10)
            support='90 s EEG reference/filter preprocessing' if cfg['eeg_o'].get('architecture')=='projection_gain' else 'full 20 s EEG filtering'
            fig.text(.07,.065,f'Old input: single-electrode 2 s. New input: {support}. No Hb or teacher trajectories enter the encoder.',fontsize=10)
            fig.text(.07,.025,f'Selection: target O RMS near {100*quantile:g}th percentile of {len(population):,} validation electrode-cases; distinct subjects; no candidate-error selection.',fontsize=10)
            filename=f'{dataset}_{label}_O.png';fig.savefig(destination/filename,dpi=250);plt.close(fig)
            write_npz(destination/f'{dataset}_{label}_O.npz',time_s=t,target_uV=truth[0]*factor,
                      old_hard_uV=original[0]*factor,new_continuous_uV=continuous*factor,new_hard_uV=hard[0]*factor)
            examples.append(dict(row=row,selection=label,target_RMS_quantile=quantile,candidate_cases=len(population),
                old=old_info,new=new_info,figure=filename))
    index=dict(source_run=str(out),source_selection=str(out/'eeg_o_selection.json'),examples=examples,
        selected_seed=seed,selection_rule='validation target RMS P10/P50/P90 in frozen training raw-EEG units; distinct subjects',
        target='fixed conditional voltage correction, not physiological noise truth',input_support=features['raw_support'])
    write_json(destination/'figure_index.json',index);return index


def _voltage_teacher_read(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _voltage_teacher_preparation(payload):
    from src.data.dual_component_tokenizer import prepare_voltage_teacher_record
    cfg, out, root, entries = payload
    out = Path(out); parent = Path(root)/cfg['voltage_teacher']['parent_run']
    started = time.monotonic(); results = []
    pending = [e for e in entries if not (out/'prepared'/f"{e['id']}.json").exists()]
    if not pending:
        return dict(status='reused', subject=entries[0]['subject'])
    try:
        co = read_json(parent/'coordinates.json')
        def read_parent(stage, row):
            source = Path(root)/cfg['voltage_teacher']['legacy_voltage_run'] if stage=='voltage_prepared' else parent
            return read_arrays(source, stage, row)
        for entry, arrays, meta in prepare_voltage_teacher_record(root, cfg, pending, co, read_parent):
            write_npz(out/'prepared'/f"{entry['id']}.npz", **arrays)
            write_json(out/'prepared'/f"{entry['id']}.json", dict(status='completed', **meta))
            results.append(entry['id'])
    except Exception:
        error = traceback.format_exc()
        for entry in pending:
            if entry['id'] not in results:
                write_json(out/'prepared'/f"{entry['id']}.json", dict(status='failed', entry=entry, error=error))
    return dict(subject=entries[0]['subject'], completed=len(results), elapsed_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _voltage_teacher_fit(payload):
    from src.data.dual_component_tokenizer import produce_teacher
    cfg, out, entry, roi, calibration = payload
    out = Path(out); key = entry['id']+f'__{roi}' + ('__Hb_only' if calibration else '')
    destination = out/'physical'/key
    if destination.with_suffix('.json').exists():
        return read_json(destination.with_suffix('.json'))
    started = time.monotonic()
    try:
        base = _voltage_teacher_read(out/'prepared'/f"{entry['id']}.npz")
        features = _voltage_teacher_read(out/'features'/f"{entry['id']}.npz")
        co = read_json(out/'coordinates.json')[entry['rows'][roi]['coordinate']]
        indices = base['roi_indices'][roi]
        from src.data.ssm_prepared import EEG_PROXY_INPUT_SCHEMA
        from src.data.unified_physiology import CACHED_FNIRS_INPUT_SCHEMA
        a = dict(features=features['ell'].reshape(900, -1, 5)[:, indices].reshape(900, 30),
            eeg=base['x'][6000:10000, indices].astype('float32'), hb=base['hb'][:, roi],
            native_time=base['hb_native_time'][:, roi], time_s=np.arange(900)*.1,
            identity=np.array(entry['rows'][roi]['id']), eeg_proxy_input=np.array(EEG_PROXY_INPUT_SCHEMA),
            eeg_proxy_rate_hz=np.array(10.), fnirs_input=np.array(CACHED_FNIRS_INPUT_SCHEMA))
        tak = read_json(out/'teacher_config.json')
        arrays, status = produce_teacher(a, co, tak, eeg_visible=not calibration)
        if arrays is not None:
            write_npz(destination.with_suffix('.npz'), **arrays)
        report = dict(key=key, entry_id=entry['id'], subject=entry['subject'], split=entry['split'], roi=roi,
            calibration=calibration, status='completed' if status.get('accepted') else 'failed',
            fit=status, elapsed_seconds=time.monotonic()-started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        report = dict(key=key, entry_id=entry['id'], subject=entry['subject'], split=entry['split'], roi=roi,
            calibration=calibration, status='failed', error=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started)
    write_json(destination.with_suffix('.json'), report)
    return report


def _voltage_teacher_evaluation(payload):
    from src.data.dual_component_tokenizer import (voltage_wiener_target, voltage_eog_reference,
        voltage_teacher_intervention, paired_voltage_targets, voltage_proxy_correction, voltage_stft)
    from src.analysis.dual_component_tokenizer import voltage_teacher_natural_metrics, voltage_teacher_delta_metrics
    cfg, out, entry, donor_id = payload
    out = Path(out); spec = cfg['voltage_teacher']; folder = out/'evaluation'/entry['id']
    if (folder/'metrics.json').exists():
        return read_json(folder/'metrics.json')
    started = time.monotonic()
    try:
        base = _voltage_teacher_read(out/'prepared'/f"{entry['id']}.npz")
        model = _voltage_teacher_read(out/'noise_model.npz')
        co = read_json(out/'coordinates.json'); confidence = read_json(out/'physical_variance.json')
        x, eog = base['x'], base['eog']; c = x.shape[1]
        projection = np.zeros((3, c*5)); physical = []; hb_fit = []; failures = []
        features = _voltage_teacher_read(out/'features'/f"{entry['id']}.npz")['ell']
        variance = []
        for roi, row in enumerate(entry['rows']):
            cc = co[row['coordinate']]; loadings = np.asarray(cc['pc']).reshape(6, 5)*cc['eeg_factor']
            for j, channel in enumerate(base['roi_indices'][roi]):
                projection[roi, channel*5:(channel+1)*5] = loadings[j]
            path = out/'physical'/f"{entry['id']}__{roi}"
            status = read_json(path.with_suffix('.json'))
            if status['status'] == 'completed':
                fit = _voltage_teacher_read(path.with_suffix('.npz'))
                physical.append(fit['physical'][:, 0]); hb_fit.append(fit['physical'][:, 1:]/cc['hb_factor'])
                variance.append(confidence[row['coordinate']]['variance'])
            else:
                # Explicitly recorded abstention; it is not a successful physical estimate.
                physical.append(features @ projection[roi]); hb_fit.append(np.full((900, 2), np.nan))
                variance.append(1e12); failures.append(roi)
        physical = np.stack(physical, axis=1); variance = np.asarray(variance)
        def estimate(value, reference, *, legacy=None):
            w = voltage_wiener_target(value, reference, model, spec)
            p = voltage_wiener_target(value, reference, model, spec, projection=projection,
                physical=physical, physical_variance=variance)
            shifted = voltage_wiener_target(value, reference, model, spec, projection=projection,
                physical=np.roll(physical, 450, axis=0), physical_variance=variance)
            old = (voltage_proxy_correction(value, base['legacy_projection'], base['legacy_physical'])
                   if legacy is None else dict(corrected=legacy, diagnostic={'status': 'retained_replay'}))
            return dict(raw=value.copy(), legacy_target=old['corrected'],
                eog_regression=value-voltage_eog_reference(reference, spec) @ model['transfer'],
                wiener=w['s'], physics_wiener=p['s'], shifted_physics=shifted['s']), dict(
                    clipped_fraction=p['clipped_fraction'], power_floor_fraction=p['power_floor_fraction'],
                    closure_max_abs=p['closure_max_abs'], legacy_status=old['diagnostic']['status'])
        baseline, diagnostics = estimate(x, eog, legacy=base['legacy_s'])
        _, frames, ze = voltage_stft(voltage_eog_reference(eog, spec), spec)
        activity = np.sum(abs(ze)**2, axis=(1, 2))
        low = np.interp(np.arange(len(x))/200, frames, activity) <= model['low_eog_threshold']
        common = {k: entry[k] for k in ('id', 'subject', 'split', 'slot')}
        natural = [dict(common, method=method, **voltage_teacher_natural_metrics(
            x, value, eog, model['voltage_scale'], low)) for method, value in baseline.items()]
        write_npz(folder/'base.npz', x=x.astype('float32'), eog=eog.astype('float32'),
            **{k: v.astype('float32') for k, v in baseline.items() if k != 'raw'},
            physical=physical, proxy=features @ projection.T, hb=base['hb'],
            hb_fit=np.stack(hb_fit, axis=1), legacy_proxy=base['legacy_proxy'],
            legacy_physical=base['legacy_physical'], projection=projection, physical_variance=variance,
            low_eog=low, channels=base['channels'], voltage_scale=model['voltage_scale'])
        perturbations, paired_checks = [], []
        if entry['split'] == 'validation':
            donor = _voltage_teacher_read(out/'prepared'/f'{donor_id}.npz')['eog']
            for kind in spec['perturbations']:
                for amplitude in spec['amplitudes_training_rms']:
                    n, reference = voltage_teacher_intervention(x, eog, donor, model, spec, kind, amplitude, spec['seed'])
                    estimates, diag = estimate(x+n, reference)
                    for method, value in estimates.items():
                        metric = voltage_teacher_delta_metrics(baseline[method], value, n, kind)
                        perturbations.append(dict(common, method=method, kind=kind, amplitude=amplitude,
                            **metric, legacy_status=diag['legacy_status']))
                    if kind != 'neural_alpha':
                        pair = paired_voltage_targets(x, baseline['physics_wiener'], n)
                        paired_checks.append(dict(kind=kind, amplitude=amplitude,
                            S_change_max_abs=float(np.max(abs(pair['s']-baseline['physics_wiener']))),
                            O_change_error_max_abs=float(np.max(abs(pair['o']-(x-baseline['physics_wiener'])-n))),
                            closure_max_abs=float(np.max(abs(pair['x']-pair['s']-pair['o'])))))
                    if amplitude == 1.:
                        sl = slice(6000, 10000)
                        arrays = dict(x=(x+n)[sl].astype('float32'), n=n[sl].astype('float32'),
                            **{k: v[sl].astype('float32') for k, v in estimates.items() if k != 'raw'})
                        if kind != 'neural_alpha':
                            arrays.update(paired_s=pair['s'][sl].astype('float32'), paired_o=pair['o'][sl].astype('float32'))
                        write_npz(folder/f'{kind}.npz', **arrays)
        report = dict(status='completed', entry=entry, natural=natural, perturbations=perturbations,
            paired_checks=paired_checks, physics_failed_rois=failures, diagnostics=diagnostics,
            elapsed_seconds=time.monotonic()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        report = dict(status='failed', entry=entry, error=traceback.format_exc(), elapsed_seconds=time.monotonic()-started)
    write_json(folder/'metrics.json', report)
    return report


def voltage_teacher_synthetic(cfg, out):
    from src.data.dual_component_tokenizer import (fit_voltage_noise_model, voltage_wiener_target,
        voltage_proxy_correction, paired_voltage_targets, eeg_band_power, BANDS)
    out = Path(out); path = out/'synthetic.json'; spec = cfg['voltage_teacher']
    if path.exists():
        return read_json(path)
    started = time.monotonic(); t = np.arange(18000)/200
    def generate(seed):
        rng = np.random.default_rng(seed); c = 6
        clean = np.zeros((len(t), c))
        for j in range(c):
            for frequency, amp in ((3, 2), (6, 3), (10, 7), (19, 3), (36, 1)):
                envelope = 1+.35*np.sin(2*np.pi*.04*t+rng.uniform(0, 6))
                clean[:, j] += amp*envelope*np.sin(2*np.pi*(frequency+rng.uniform(-.2,.2))*t+rng.uniform(0,6))
        eog = np.zeros((len(t), 2))
        for j in range(2):
            for onset in rng.uniform(5, 85, 20):
                eog[:, j] += rng.uniform(25, 90)*np.exp(-.5*((t-onset)/rng.uniform(.10, .30))**2)
        transfer = np.array([[.7,.6,.45,.15,.1,.05],[.2,-.2,.1,-.1,.05,-.05]])
        return dict(x=clean+eog@transfer, eog=eog, clean=clean, split='train')
    training = [generate(seed) for seed in spec['synthetic_train_seeds']]
    model = fit_voltage_noise_model(training, spec)
    a = np.zeros((2, 30)); a[0, 2::5] = 1/6; a[1, 0::5] = 1/6
    metrics, maximum_closure = [], 0.
    for seed in spec['synthetic_eval_seeds']:
        sample = generate(seed); x, clean, eog = sample['x'], sample['clean'], sample['eog']
        powers = np.log(eeg_band_power(clean, bands=BANDS, sample_rate=200, target_rate=10)).reshape(900,30)
        powers -= powers[:50].mean(0); prior = powers @ a.T
        no = voltage_wiener_target(x, eog, model, spec)
        phys = voltage_wiener_target(x, eog, model, spec, projection=a, physical=prior,
            physical_variance=np.ones(2)*.05)
        wrong = voltage_wiener_target(x, eog, model, spec, projection=a, physical=np.roll(prior,450,axis=0)+2.,
            physical_variance=np.ones(2)*.05)
        legacy = voltage_proxy_correction(x,a,prior)
        outputs = dict(raw=x, legacy_target=legacy['corrected'], wiener=no['s'], physics_oracle=phys['s'],
            wrong_physics=wrong['s'])
        sl = slice(6000,10000); denominator = np.mean(clean[sl]**2)
        for name, s in outputs.items():
            metrics.append(dict(seed=seed, method=name, clean_NMSE=float(np.mean((s[sl]-clean[sl])**2)/denominator),
                correlation=float(np.corrcoef(s[sl].ravel(),clean[sl].ravel())[0,1])))
        pair = paired_voltage_targets(x, phys['s'], x-clean)
        maximum_closure = max(maximum_closure, float(np.max(abs(pair['s']+pair['o']-pair['x']))))
        if seed == spec['synthetic_eval_seeds'][0]:
            write_npz(out/'synthetic_example.npz', x=x, clean=clean, eog=eog, **{k:v for k,v in outputs.items() if k!='raw'})
    report = dict(schema='voltage_teacher_synthetic_v1', status='completed', metrics=metrics,
        training_seeds=spec['synthetic_train_seeds'], evaluation_seeds=spec['synthetic_eval_seeds'],
        maximum_pair_closure=maximum_closure, physical_prior=spec['synthetic_physical_prior'],
        elapsed_seconds=time.monotonic()-started)
    write_json(path,report)
    return report


def run_voltage_teacher(cfg, out, root, workers, stage):
    from src.data.dual_component_tokenizer import (voltage_teacher_plan, fit_voltage_noise_model,
        voltage_power_evidence)
    from src.data.ssm_prepared import fit_feature_coordinate, EEG_PROXY_INPUT_SCHEMA
    from src.data.unified_physiology import CACHED_FNIRS_INPUT_SCHEMA
    from src.analysis.dual_component_tokenizer import summarize_voltage_teacher
    out, root = Path(out), Path(root); spec=cfg['voltage_teacher']
    expected_parent = 'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261028_v3'
    if spec['parent_run'] != expected_parent:
        raise ValueError('exact retained public parent required')
    if spec['legacy_voltage_run'] != expected_parent.replace('20261028_v3','20261029_voltage_v1'):
        raise ValueError('exact retained legacy voltage label run required')
    retained = read_json(root/spec['parent_run']/'inventory.json')
    panel = voltage_teacher_plan(cfg,retained)
    if (out/'plan.json').exists() and read_json(out/'plan.json') != panel:
        raise ValueError('panel drift; preserve the original run')
    write_json(out/'plan.json',panel)
    if stage == 'voltage-teacher-plan':
        return dict(status='planned', windows=len(panel['rows']), planned=panel['planned_windows'], measured_arrays_read=0)
    write_json(out/'controller.json',dict(status='running',stage='synthetic',pid=os.getpid(),updated=now()))
    voltage_teacher_synthetic(cfg,out)
    groups=defaultdict(list)
    for entry in panel['rows']:groups[entry['subject']].append(entry)
    write_json(out/'controller.json',dict(status='running',stage='preparing',pid=os.getpid(),updated=now(),workers=workers))
    prepared = _bounded_jobs(_voltage_teacher_preparation,
        [(cfg,str(out),str(root),group) for group in groups.values()],workers,inflight=1)
    write_json(out/'preparation.json',prepared)
    available = [e for e in panel['rows'] if read_json(out/'prepared'/f"{e['id']}.json")['status']=='completed']
    if not (out/'noise_model.npz').exists():
        training=[]
        for entry in available:
            if entry['split']=='train':
                training.append(dict(_voltage_teacher_read(out/'prepared'/f"{entry['id']}.npz"),split='train'))
        model=fit_voltage_noise_model(training,spec)
        write_npz(out/'noise_model.npz',**model)
    model=_voltage_teacher_read(out/'noise_model.npz')
    if not (out/'noise_calibration.json').exists():
        write_json(out/'noise_calibration.json',dict(training_ids=[e['id'] for e in available if e['split']=='train'],
            low_eog_threshold=model['low_eog_threshold'], low_eog_training_frames=model['low_eog_training_frames'],
            transfer=model['transfer'], noise_model='EOG-derived covariance, conditional mean zero'))
    for entry in available:
        path=out/'features'/f"{entry['id']}.npz"
        if not path.exists():
            a=_voltage_teacher_read(out/'prepared'/f"{entry['id']}.npz")
            evidence=voltage_power_evidence(a['x'],a['eog'],model,spec)
            write_npz(path,ell=evidence['ell'],power=evidence['power'],reference=evidence['reference'])
    if not (out/'coordinates.json').exists():
        chunks=defaultdict(list); ids=defaultdict(list)
        for entry in available:
            if entry['split']!='train':continue
            a=_voltage_teacher_read(out/'prepared'/f"{entry['id']}.npz")
            features=_voltage_teacher_read(out/'features'/f"{entry['id']}.npz")['ell'].reshape(900,-1,5)
            for i,row in enumerate(entry['rows']):
                chunks[row['coordinate']].append((features[:,a['roi_indices'][i]].reshape(900,30),a['hb'][:,i]))
                ids[row['coordinate']].append(row['id'])
        coordinates={}
        for key,values in chunks.items():
            coordinates[key]=dict(fit_feature_coordinate(values),training_ids=ids[key],
                eeg_proxy_input=EEG_PROXY_INPUT_SCHEMA,fnirs_input=CACHED_FNIRS_INPUT_SCHEMA,
                feature_generation='noise_subtracted_bandpower_v1_new_coordinate_not_legacy_PCA')
        write_json(out/'coordinates.json',coordinates)
    tak=yaml.safe_load((SOURCE/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_10hz_v2.yaml').read_text())
    write_json(out/'teacher_config.json',tak)
    jobs=[(cfg,str(out),entry,roi,False) for entry in available for roi in range(3)]
    jobs += [(cfg,str(out),entry,roi,True) for entry in available if entry['split']=='train' and entry['slot']==0 for roi in range(3)]
    if stage=='voltage-teacher-pilot':
        jobs=jobs[:2]
    def progress(result, count):
        write_json(out/'controller.json',dict(status='running',stage='physical',pid=os.getpid(),updated=now(),
            planned_fits=len(jobs),terminal_fits=count,workers=workers,last_fit=result.get('key')))
    write_json(out/'controller.json',dict(status='running',stage='physical',pid=os.getpid(),updated=now(),planned_fits=len(jobs),workers=workers))
    results=_bounded_jobs(_voltage_teacher_fit,jobs,min(workers,len(jobs)),inflight=1,progress=progress)
    write_json(out/('pilot.json' if stage=='voltage-teacher-pilot' else 'physical_fits.json'),results)
    if stage=='voltage-teacher-pilot':
        write_json(out/'controller.json',dict(status='pilot_completed',updated=now()))
        return dict(status='pilot_completed', fits=results)
    coordinates=read_json(out/'coordinates.json'); confidence={}
    for key,co in coordinates.items():
        calibrated=[r for r in results if r.get('calibration') and r['status']=='completed'
            and next(e for e in available if e['id']==r['entry_id'])['rows'][r['roi']]['coordinate']==key]
        errors=[]
        for r in calibrated:
            a=_voltage_teacher_read(out/'physical'/f"{r['key']}.npz")
            errors.append(float(np.mean((a['physical'][50:-1,0]-a['u'][50:-1,0])**2)))
        variance=max(float(np.mean(errors)),float(co['sd'][0])**2) if errors else 1e12
        confidence[key]=dict(variance=variance, Hb_only_calibration_errors=errors,
            calibration_subjects=[r['subject'] for r in calibrated], planned_subjects=len(spec['train_subjects']),
            trustworthy=False, interpretation='conservative_training_cross_modality_error_floor_not_posterior_uncertainty')
    write_json(out/'physical_variance.json',confidence)
    validation=sorted([e for e in available if e['split']=='validation' and e['slot']==1],key=lambda e:e['subject'])
    donor={e['subject']:validation[(i+1)%len(validation)]['id'] for i,e in enumerate(validation)}
    write_json(out/'controller.json',dict(status='running',stage='interventions',pid=os.getpid(),updated=now(),workers=workers))
    evaluations=_bounded_jobs(_voltage_teacher_evaluation,
        [(cfg,str(out),e,donor.get(e['subject'])) for e in available],workers,inflight=1)
    report=summarize_voltage_teacher(out,panel,evaluations,results,cfg)
    write_json(out/'summary.json',report)
    write_json(out/'controller.json',dict(status='completed',updated=now(),summary='summary.json'))
    return report


def _latent_health_trial(payload):
    from src.data.dual_component_tokenizer import latent_health_hb_patch
    parent, trial, scales, spec = payload
    started = time.monotonic()
    try:
        # Split is checked before opening even the retained teacher file.
        if any(r['split'] not in spec['allowed_splits'] for r in trial['rows']):
            raise ValueError('development-only Hb read boundary')
        samples = []
        for row in trial['rows']:
            arrays = read_arrays(parent, 'teacher', row,
                keys=('identity', 'teacher_arm', 'u', 'physical', 'remaining'))
            samples.append(latent_health_hb_patch(row, arrays, scales[row['coordinate']], spec))
        return dict(status='completed', trial_id=trial['trial_id'], rows=trial['rows'],
                    arrays={k: np.stack([r[k] for r in samples]) for k in ('u', 'S', 'O')},
                    retained_bytes=sum(array_paths(parent, 'teacher', r).stat().st_size for r in trial['rows']),
                    elapsed_seconds=time.monotonic() - started)
    except Exception:
        return dict(status='failed', trial_id=trial['trial_id'], error=traceback.format_exc(),
                    elapsed_seconds=time.monotonic() - started)


def prepare_latent_health(cfg, out, root, workers):
    from src.data.dual_component_tokenizer import latent_health_plan
    out, root = Path(out), Path(root)
    spec = cfg['latent_health']; parent = root / spec['parent_run']
    plan_path = out / 'plan.json'
    if not plan_path.exists():
        write_json(plan_path, latent_health_plan(parent, spec))
    plan = read_json(plan_path)
    if (plan['parent'] != str(parent.resolve()) or plan['allowed_splits'] != ['train', 'validation']
            or plan['test_access'] != 'forbidden' or plan['protected_data'] != 'forbidden'):
        raise ValueError('frozen public development plan required before retained array reads')
    destination = out / 'cache'; destination.mkdir(exist_ok=True)
    if (destination / 'manifest.json').exists():
        return read_json(destination / 'manifest.json')
    payloads = [(str(parent), t, plan['observation_training_sd'], spec) for t in plan['trials']]
    timings = []
    for count in cfg['resources']['prepare_benchmark_workers']:
        start = time.monotonic()
        sample = _bounded_jobs(_latent_health_trial, payloads[:64], count)
        seconds = time.monotonic() - start
        timings.append(dict(workers=count, seconds=seconds, trials_per_second=len(sample) / seconds,
                            retained_MiB_per_second=sum(r.get('retained_bytes', 0) for r in sample) / seconds / 1024 ** 2,
                            failures=sum(r['status'] != 'completed' for r in sample)))
    selected_workers = min(workers, max(timings, key=lambda r: r['trials_per_second'])['workers'])
    write_json(out / 'prepare_resources.json', dict(benchmarks=timings, selected_workers=selected_workers,
        reason='highest measured retained-file throughput within the CPU ceiling; competing Tak and downstream jobs retained'))
    def progress(result, completed):
        if completed % 100 == 0 or completed == len(payloads):
            write_json(out / 'prepare_progress.json', dict(completed=completed, planned=len(payloads),
                last_trial=result.get('trial_id'), last_status=result['status'], updated=now()))
    started = time.monotonic()
    results = _bounded_jobs(_latent_health_trial, payloads, selected_workers, progress=progress)
    bytrial = {r['trial_id']: r for r in results if 'trial_id' in r}
    failures = list(plan['failures']); complete = []
    for trial in plan['trials']:
        result = bytrial.get(trial['trial_id'], dict(status='failed', trial_id=trial['trial_id'], error='worker did not return identity'))
        if result['status'] == 'completed':
            complete.append(result)
        else:
            failures.append({k: v for k, v in result.items() if k != 'arrays'})
    if not complete:
        write_json(out / 'prepare_failures.json', failures)
        raise ValueError('no complete three-location Hb trials; failures retained')
    rows = [row for result in complete for row in result['rows']]
    values = {k: np.concatenate([r['arrays'][k] for r in complete]) for k in ('u', 'S', 'O')}
    for key, value in values.items():
        temporary = destination / f'{key}.tmp.npy'
        np.save(temporary, value, allow_pickle=False)
        temporary.replace(destination / f'{key}.npy')
    train = np.array([r['split'] == 'train' for r in rows])
    balance = {c: np.maximum(values[c][train].reshape(-1, 2).var(0, dtype=np.float64),
                             spec['minimum_target_variance']).tolist() for c in ('S', 'O')}
    result = dict(schema='dual_component_latent_health_cache_v1', status='completed', rows=rows,
        arrays={k: f'{k}.npy' for k in values}, shapes={k: list(v.shape) for k, v in values.items()},
        planned_trials=plan['planned_trials'], retained_trials=len(complete),
        planned_windows=plan['planned_windows'], retained_windows=len(rows), failures=failures,
        parent_identity=plan['parent_identity'], plan_identity=plan['identity'], parent=str(parent.resolve()),
        target_variance=balance, variance_fit='training subjects only, separate component and color',
        observation_training_sd=plan['observation_training_sd'],
        normalization='parent training observation SD; no new per-window centering or scaling',
        event_support_s=[-5, 25], parent_reference_event_s=[-30, -25],
        test_arrays_read=0, ssms_refitted=0, eeg_arrays_used_as_inputs=0,
        elapsed_seconds=time.monotonic() - started,
        subjects={ds: {sp: len({r['subject'] for r in rows if r['dataset'] == ds and r['split'] == sp})
                       for sp in ('train', 'validation')} for ds in sorted({r['dataset'] for r in rows})})
    write_json(destination / 'manifest.json', result)
    return result


def load_latent_health_cache(out):
    folder = Path(out) / 'cache'; manifest = read_json(folder / 'manifest.json')
    if (manifest['schema'] != 'dual_component_latent_health_cache_v1' or manifest['test_arrays_read'] != 0
            or any(r['split'] not in ('train', 'validation') for r in manifest['rows'])):
        raise ValueError('development-only latent health cache required before arrays')
    arrays = {k: np.load(folder / manifest['arrays'][k], mmap_mode='r', allow_pickle=False) for k in ('u', 'S', 'O')}
    if any(v.shape != (len(manifest['rows']), 15, 40) or v.dtype != np.float32 for v in arrays.values()):
        raise ValueError('latent health cache tensor/dtype mismatch')
    return manifest, arrays


def _latent_health_groups(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        if row['split'] == 'train':
            groups[row['dataset']][row['subject']].append(i)
    return [[np.array(indices) for _, indices in sorted(subjects.items())] for _, subjects in sorted(groups.items())]


def _latent_health_sample(rng, groups, count):
    indices = []
    for _ in range(count):
        subjects = groups[int(rng.integers(len(groups)))]; choices = subjects[int(rng.integers(len(subjects)))]
        indices.append(int(rng.choice(choices)))
    return np.array(indices)


def evaluate_latent_health(cfg, model, arrays, manifest, job, device):
    import torch
    from src.analysis.dual_component_tokenizer import latent_hb_fidelity, latent_codebook_health, _csv
    spec = cfg['latent_health']; rows = manifest['rows']; count = len(rows)
    predictions = {c: np.empty((count, 15, 40), dtype='float32') for c in ('S', 'O', 'sum')}
    ids = {c: np.empty((count, 15), dtype='int16') for c in ('S', 'O')} if model.hard else {}
    latents = {c: np.empty((count, 15, spec['latent_dim']), dtype='float32') for c in ('S', 'O')}
    maximum_decode_difference = 0.; model.eval()
    before = {c: b['quantizer'].codebook.clone() for c, b in model.branches.items()}
    with torch.inference_mode():
        for start in range(0, count, spec['evaluation_batch_size']):
            stop = min(start + spec['evaluation_batch_size'], count)
            x = torch.as_tensor(np.array(arrays['u'][start:stop]), device=device)
            result = model(x)
            for c in predictions:
                predictions[c][start:stop] = result[c].cpu().numpy()
            for c in latents:
                latents[c][start:stop] = result['latent'][c].cpu().numpy()
            if model.hard:
                replay = model.decode_ids(result['ids'])
                for c in ('S', 'O'):
                    ids[c][start:stop] = result['ids'][c].cpu().numpy()
                    maximum_decode_difference = max(maximum_decode_difference, float((replay[c] - result[c]).abs().max()))
    if maximum_decode_difference > 2e-5:
        raise AssertionError('hard forward does not reproduce integer-only decoding')
    if any(not torch.equal(before[c], b['quantizer'].codebook) for c, b in model.branches.items()):
        raise AssertionError('evaluation changed the frozen codebook')
    fidelity = latent_hb_fidelity(predictions, arrays, rows)
    _csv(Path(job) / 'subject_metrics.csv', fidelity.pop('subjects'))
    validation = np.flatnonzero([r['split'] == 'validation' for r in rows])
    write_npz(Path(job) / 'validation_predictions.npz', row_index=validation,
              **{k: v[validation] for k, v in predictions.items()})
    health, geometry = {}, []
    for c, values in latents.items():
        for split in ('train', 'validation'):
            selected = np.array([r['split'] == split for r in rows])
            x = np.asarray(values[selected], dtype=float).reshape(-1, spec['latent_dim'])
            covariance = x.T @ x / len(x) - np.outer(x.mean(0), x.mean(0))
            eigen = np.linalg.eigvalsh(covariance).clip(0)
            geometry.append(dict(branch=c, split=split, participation_rank=float(eigen.sum() ** 2 /
                max(float(np.square(eigen).sum()), 1e-30)), dimension=spec['latent_dim']))
        if model.hard:
            q = model.branches[c]['quantizer']
            h = latent_codebook_health(ids[c], q.codebook.cpu().numpy(), rows, spec['health'])
            _csv(Path(job) / f'{c}_code_coverage.csv', h.pop('coverage'))
            h.update(total_revivals=int(q.revival_count), ema_updates=int(q.update_count))
            health[c] = h
    context = []
    if model.hard:
        write_npz(Path(job) / 'hard_ids.npz', **ids)
        write_npz(Path(job) / 'codebooks.npz', **{c: b['quantizer'].codebook.cpu().numpy() for c, b in model.branches.items()})
        for dataset in sorted({r['dataset'] for r in rows}):
            choices = {}
            for i in validation:
                r = rows[i]
                if r['dataset'] == dataset:
                    choices.setdefault(r['subject'], i)
            selected = np.array(list(choices.values()))
            if len(selected) < 2:
                continue
            donors = np.roll(selected, 1)
            x = np.array(arrays['u'][selected]); changed = np.array(arrays['u'][donors]); changed[:, 7] = x[:, 7]
            with torch.inference_mode():
                result = model(torch.as_tensor(changed, device=device))
            for c in ('S', 'O'):
                context.append(dict(dataset=dataset, branch=c, examples=len(selected),
                    central_patch=7, central_event_seconds=[9, 11],
                    ID_change_fraction=float(np.mean(result['ids'][c][:, 7].cpu().numpy() != ids[c][selected, 7])),
                    central_reconstruction_change_NMSE=float(np.mean(np.square(
                        result[c][:, 7].cpu().numpy() - predictions[c][selected, 7]))),
                    source_ids=[rows[i]['id'] for i in selected], donor_ids=[rows[i]['id'] for i in donors],
                    meaning='one first validation window per subject; other-subject same-dataset context, central observation fixed; diagnostic only'))
    return dict(fidelity=fidelity, health=health, latent_geometry=geometry, context_replacement=context,
        integer_decode_max_abs_difference=maximum_decode_difference, test_used=False,
        health_pass=all(h['passed'] for h in health.values()) if health else None)


def train_latent_health(cfg, out, candidate, seed, device, *, memory_bytes=None, pilot_output=None, steps=None):
    import torch
    from src.tokenizers.dual_component_tokenizer import GlobalLatentComponentTokenizer
    spec = cfg['latent_health']; out = Path(out)
    if candidate not in ['warmup', *spec['candidates']] or seed not in spec['seeds']:
        raise ValueError('unregistered latent health job')
    pilot = pilot_output is not None
    if (steps is not None and not pilot) or (pilot and (steps is None or not 1 <= steps <= cfg['resources']['benchmark_steps'])):
        raise ValueError('step overrides are bounded resource pilots only')
    job = Path(pilot_output) if pilot else out / 'latent_health' / candidate / f'seed_{seed}'
    if pilot and not job.resolve().is_relative_to((out / 'benchmark').resolve()):
        raise ValueError('resource pilot artifacts must remain under this run benchmark directory')
    job.mkdir(parents=True, exist_ok=True)
    if (job / 'summary.json').exists():
        return read_json(job / 'summary.json')
    with owner_lock(job, 'training'):
        try:
            torch.set_num_threads(cfg['resources']['torch_threads'])
            random.seed(seed); torch.manual_seed(seed)
            if device.startswith('cuda'):
                torch.cuda.set_device(device)
                budget_bytes = memory_bytes or cfg['resources']['max_new_gpu_memory_mib'] * 1024 ** 2
                torch.cuda.set_per_process_memory_fraction(budget_bytes / torch.cuda.get_device_properties(device).total_memory, device)
                torch.cuda.reset_peak_memory_stats(device)
            manifest, arrays = load_latent_health_cache(out)
            groups = _latent_health_groups(manifest['rows'])
            if len(groups) != 2 or any(not group for group in groups):
                raise ValueError('both registered training datasets required')
            hard = candidate.startswith('vq'); size = int(candidate[2:]) if hard else 128
            model = GlobalLatentComponentTokenizer(spec, codebook_size=size, hard=hard).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
            rng = np.random.default_rng(seed + (0 if candidate == 'warmup' else 1000))
            budget = steps if pilot else spec['warmup_steps' if candidate == 'warmup' else 'continuation_steps']
            signature = dict(schema=cfg['schema'], candidate=candidate, seed=seed, budget=budget,
                             spec=spec, plan_identity=manifest['plan_identity'], pilot=pilot)
            checkpoint = job / 'checkpoint.pt'; history = []; start = 0
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location=device, weights_only=False)
                if saved['signature'] != signature:
                    raise ValueError('latent health resume signature changed')
                model.load_state_dict(saved['model_state']); optimizer.load_state_dict(saved['optimizer_state'])
                start, history = int(saved['step']), saved['history']
                rng.bit_generator.state = saved['numpy_rng']; torch.set_rng_state(saved['torch_rng'].cpu())
                if device.startswith('cuda') and saved.get('cuda_rng') is not None:
                    torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), device)
            elif candidate != 'warmup' and not pilot:
                warmup = out / 'latent_health/warmup' / f'seed_{seed}'
                if not (warmup / 'summary.json').exists() or read_json(warmup / 'summary.json')['status'] != 'completed':
                    raise ValueError('same-seed completed continuous warmup required')
                saved = torch.load(warmup / 'checkpoint.pt', map_location=device, weights_only=False)
                if saved['step'] != spec['warmup_steps'] or saved['signature']['plan_identity'] != manifest['plan_identity']:
                    raise ValueError('warmup exposure or data identity mismatch')
                weights = {k: v for k, v in saved['model_state'].items() if '.quantizer.' not in k}
                loaded = model.load_state_dict(weights, strict=False)
                if loaded.unexpected_keys or any('.quantizer.' not in k for k in loaded.missing_keys):
                    raise ValueError('warmup architecture mismatch')
            if start > budget:
                raise ValueError('checkpoint exceeds fixed step budget')
            if hard and not bool(model.branches['S']['quantizer'].initialized):
                initialization = _latent_health_sample(np.random.default_rng(seed + 9000), groups, spec['initialization_rows'])
                if any(manifest['rows'][i]['split'] != 'train' for i in initialization):
                    raise AssertionError('non-training initialization row')
                model.initialize_codebooks(torch.as_tensor(np.array(arrays['u'][initialization]), device=device))
                write_json(job / 'initialization.json', dict(row_indices=initialization, split='train',
                    patches=len(initialization) * spec['patches'], targets_used=False))
            variances = {c: torch.as_tensor(manifest['target_variance'][c], dtype=torch.float32, device=device)
                         for c in ('S', 'O')}
            started = time.monotonic(); model.train()
            for step in range(start, budget):
                indices = _latent_health_sample(rng, groups, spec['batch_size'])
                x = torch.as_tensor(np.array(arrays['u'][indices]), device=device)
                result = model(x)
                loss = result['commitment']; parts = {}
                for c in ('S', 'O'):
                    target = torch.as_tensor(np.array(arrays[c][indices]), device=device)
                    error = (result[c] - target).reshape(-1, 2).square()
                    term = (error / variances[c]).mean()
                    loss = loss + term / 2
                    parts[c] = float(error.mean().detach())
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('nonfinite latent health loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip'])
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError('nonfinite latent health gradient')
                optimizer.step()
                if (step + 1) % spec['checkpoint_every'] == 0 or step + 1 == budget:
                    item = dict(step=step + 1, loss=float(loss.detach()), waveform_MSE=parts,
                        elapsed_seconds=time.monotonic() - started,
                        batch_health={c: {k: float(v.detach()) for k, v in h.items()}
                                      for c, h in result['health'].items()})
                    history.append(item)
                    _torch_save(checkpoint, dict(signature=signature, model_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict(), step=step + 1, history=history,
                        numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None))
                    write_json(job / 'progress.json', dict(status='training', candidate=candidate, seed=seed,
                        step=step + 1, budget=budget, updated=now(), last=item))
            elapsed = time.monotonic() - started
            result = dict(status='completed', candidate=candidate, seed=seed, completed_steps=budget,
                total_exposure_steps=budget + (spec['warmup_steps'] if candidate != 'warmup' and not pilot else 0),
                resumed_from_step=start, elapsed_training_seconds=elapsed,
                steps_per_second=(budget - start) / max(elapsed, 1e-9),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else 0,
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.startswith('cuda') else 0,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                nominal_bits_per_branch_patch=float(np.log2(size)) if hard else None,
                phase='resource_pilot' if pilot else candidate, test_used=False,
                source_checkpoint=str(checkpoint), history=history)
            if candidate != 'warmup' and not pilot:
                result['evaluation'] = evaluate_latent_health(cfg, model, arrays, manifest, job, device)
            write_json(job / 'summary.json', result)
            write_json(job / 'progress.json', dict(status='completed', step=budget, budget=budget, updated=now()))
            return result
        except Exception:
            write_json(job / 'failure.json', dict(status='failed', candidate=candidate, seed=seed,
                updated=now(), error=traceback.format_exc(), checkpoint_preserved=(job / 'checkpoint.pt').exists()))
            raise


def latent_health_jobs(cfg, cfg_path, out, root, devices, workers, jobs, *, benchmark_name=None):
    """Bounded child jobs under the single durable suite controller."""
    pending = list(jobs); total = len(pending); active = []; results = []
    slots = [device for device in devices for _ in range(workers)]
    started = time.monotonic()
    while pending or active:
        while pending and len(active) < len(slots):
            candidate, seed = pending.pop(0)
            job = (Path(out) / 'benchmark' / benchmark_name / f'seed_{seed}' if benchmark_name
                   else Path(out) / 'latent_health' / candidate / f'seed_{seed}')
            if (job / 'summary.json').exists():
                results.append(read_json(job / 'summary.json')); continue
            if (job / 'failure.json').exists():
                results.append(read_json(job / 'failure.json')); continue
            used = Counter(item['device'] for item in active)
            device = next(d for d in slots if used[d] < slots.count(d))
            job.mkdir(parents=True, exist_ok=True)
            memory = cfg['resources']['max_new_gpu_memory_mib'] * 1024 ** 2 // slots.count(device)
            command = [sys.executable, str(Path(__file__).resolve()), '--config', str(cfg_path),
                '--project-root', str(root), '--output', str(out), '--stage', 'latent-health-one',
                '--candidate', candidate, '--seed', str(seed), '--device', device,
                '--gpu-memory-bytes', str(memory), '--parent-pid', str(os.getpid())]
            if benchmark_name:
                command += ['--pilot-output', str(job), '--steps', str(cfg['resources']['benchmark_steps'])]
            log = (job / 'process.log').open('a')
            process = subprocess.Popen(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT)
            write_json(job / 'launch.json', dict(command=command, working_directory=str(SOURCE),
                pid=process.pid, controller_pid=os.getpid(), device=device, memory_bytes=memory, launched_at=now()))
            active.append(dict(process=process, log=log, candidate=candidate, seed=seed, device=device, job=job))
        if not active:
            continue
        time.sleep(.25)
        for item in list(active):
            code = item['process'].poll()
            if code is None:
                continue
            item['log'].close(); active.remove(item); job = item['job']
            record = (read_json(job / 'summary.json') if code == 0 and (job / 'summary.json').exists()
                      else read_json(job / 'failure.json') if (job / 'failure.json').exists()
                      else dict(status='failed', candidate=item['candidate'], seed=item['seed'],
                                returncode=code, log=str(job / 'process.log')))
            results.append(record)
        write_json(Path(out) / 'training_progress.json', dict(phase=benchmark_name or 'training',
            completed=len(results), planned=total, elapsed_seconds=time.monotonic() - started,
            active=[dict(candidate=x['candidate'], seed=x['seed'], device=x['device'], pid=x['process'].pid) for x in active],
            updated=now()))
    return dict(results=results, wall_seconds=time.monotonic() - started)


def benchmark_latent_health(cfg, cfg_path, out, root, devices):
    path = Path(out) / 'benchmark.json'
    if path.exists():
        return read_json(path)
    trials = []
    for workers in (1, 2):
        report = latent_health_jobs(cfg, cfg_path, out, root, devices[:1], workers,
            [('vq512', seed) for seed in cfg['latent_health']['seeds'][:workers]], benchmark_name=f'workers_{workers}')
        if any(r['status'] != 'completed' for r in report['results']):
            trials.append(dict(workers=workers, status='failed', jobs=report['results'])); continue
        trials.append(dict(workers=workers, status='completed', wall_seconds=report['wall_seconds'],
            aggregate_steps_per_second=sum(r['completed_steps'] for r in report['results']) / report['wall_seconds'],
            steady_steps_per_second=sum(r['steps_per_second'] for r in report['results']),
            maximum_worker_reserved_bytes=max(r['peak_cuda_reserved_bytes'] for r in report['results'])))
    complete = [r for r in trials if r['status'] == 'completed']
    if not complete:
        write_json(path, dict(status='failed', trials=trials))
        raise RuntimeError('both supervised GPU resource pilots failed')
    choice = max(complete, key=lambda r: r['aggregate_steps_per_second'])
    result = dict(status='completed', trials=trials, gpu_workers_per_device=choice['workers'], devices=devices[:1],
        reason='highest measured aggregate throughput within the memory cap; CUDA1 reserved for competing jobs',
        estimated_training_seconds=(3 * cfg['latent_health']['warmup_steps'] + 9 * cfg['latent_health']['continuation_steps']) /
            max(choice['steady_steps_per_second'], 1e-9), measured_at=now())
    write_json(path, result)
    return result


def summarize_latent_health(cfg, out, results):
    spec = cfg['latent_health']; manifest, _ = load_latent_health_cache(out)
    complete = [r for r in results if r['status'] == 'completed' and 'evaluation' in r]
    comparisons = []
    for seed in spec['seeds']:
        baseline = next((r for r in complete if r['candidate'] == 'continuous' and r['seed'] == seed), None)
        if baseline is None:
            continue
        lookup = {tuple(m[k] for k in ('dataset', 'split', 'branch', 'signal', 'metric')): m
                  for m in baseline['evaluation']['fidelity']['metrics']}
        for result in complete:
            if result['seed'] != seed or result['candidate'] == 'continuous':
                continue
            for metric in result['evaluation']['fidelity']['metrics']:
                if metric['metric'] != 'NMSE' or metric['split'] != 'validation':
                    continue
                key = tuple(metric[k] for k in ('dataset', 'split', 'branch', 'signal', 'metric'))
                b = lookup[key]
                comparisons.append(dict(candidate=result['candidate'], seed=seed,
                    dataset=metric['dataset'], branch=metric['branch'], signal=metric['signal'],
                    continuous_NMSE=b['value'], hard_NMSE=metric['value'],
                    hard_minus_continuous=metric['value'] - b['value'], subjects=metric['subjects']))
    result = dict(status='completed' if len(complete) == 9 else 'completed_with_failures',
        experiment_id=cfg['experiment_id'], measured_modality='hb', eeg_measured=False,
        planned_jobs=9, completed_jobs=len(complete),
        failures=[{k: r.get(k) for k in ('candidate', 'seed', 'status', 'error', 'log')} for r in results if r['status'] != 'completed'],
        planned_trials=manifest['planned_trials'], retained_trials=manifest['retained_trials'],
        planned_windows=manifest['planned_windows'], retained_windows=manifest['retained_windows'],
        data_failures=manifest['failures'], subjects=manifest['subjects'],
        results=[{k: r[k] for k in ('candidate', 'seed', 'total_exposure_steps', 'nominal_bits_per_branch_patch', 'evaluation')} for r in complete],
        quantization_comparisons=comparisons, test_used=False, ssms_refitted=0, selection='none',
        interpretation='engineering health and hard-decoding evidence only; new EEG supervision remains independent',
        completed_at=now())
    write_json(Path(out) / 'summary.json', result)
    return result


def synthetic_latent_health(cfg):
    import torch
    from src.tokenizers.dual_component_tokenizer import GlobalLatentComponentTokenizer
    torch.set_num_threads(2); torch.manual_seed(117)
    cases = []
    for modality, dimension in [('hb', 40), ('eeg_interface', 400)]:
        spec = dict(cfg['latent_health'], input_dim=dimension, width=16, latent_dim=8, heads=2,
                    encoder_layers=1, decoder_layers=1, kmeans_iters=3)
        model = GlobalLatentComponentTokenizer(spec, codebook_size=8, hard=True)
        t = torch.linspace(0, 30, 15 * dimension)
        phases = torch.arange(4.)[:, None]
        source = torch.sin(t[None] * .7 + phases).reshape(4, 15, dimension)
        residual = .2 * torch.cos(t[None] * 1.3 + phases).reshape(4, 15, dimension)
        x = source + residual
        model.initialize_codebooks(x)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        for _ in range(8):
            model.train(); result = model(x)
            loss = (result['S'] - source).square().mean() + (result['O'] - residual).square().mean() + result['commitment']
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode():
            result = model(x); replay = model.decode_ids(result['ids'])
        difference = max(float((replay[c] - result[c]).abs().max()) for c in ('S', 'O'))
        if not torch.isfinite(loss) or difference > 2e-5:
            raise AssertionError('synthetic global hard-token decoding failed')
        cases.append(dict(modality=modality, input_shape=list(x.shape),
            ID_shape=list(result['ids']['S'].shape), output_shape=list(replay['S'].shape),
            steps=8, finite_loss=float(loss.detach()), integer_decode_max_abs_difference=difference))
    return dict(status='passed', cases=cases, measured_arrays_read=0,
                interpretation='software interface and gradient/replay evidence, not measured EEG health')


def latent_health_dispatch(args, cfg):
    from src.data.dual_component_tokenizer import latent_health_plan
    allowed = ('check', 'synthetic', 'latent-health-plan', 'latent-health-prepare',
               'latent-health-benchmark', 'latent-health-one', 'latent-health')
    if args.stage not in allowed or args.evaluation_split != 'development':
        raise ValueError('latent health cannot launch legacy stages or test evaluation')
    if args.stage == 'check':
        return dict(status='latent_health_contract_valid', experiment_id=cfg['experiment_id'],
                    measured_modality='hb', measured_arrays_read=0, test_access='forbidden')
    if args.output is None:
        raise ValueError('explicit fresh latent health run output required')
    out, root = args.output.resolve(), args.project_root.resolve()
    if args.stage == 'synthetic':
        result = synthetic_latent_health(cfg); write_json(out / 'synthetic.json', result); return result
    if args.stage == 'latent-health-one':
        if args.parent_pid is not None and os.getppid() != args.parent_pid:
            raise ValueError('worker must be a direct child of its suite controller')
        freeze_contract(out, cfg, root)
        result = train_latent_health(cfg, out, args.candidate, args.seed, args.device,
            memory_bytes=args.gpu_memory_bytes, pilot_output=args.pilot_output, steps=args.steps)
        return {k: result[k] for k in ('status', 'candidate', 'seed', 'completed_steps', 'total_exposure_steps')}
    if args.steps is not None or args.pilot_output is not None:
        raise ValueError('resource overrides only apply to bounded child pilots')
    devices = args.devices or cfg['resources']['devices']
    if devices != ['cuda:0']:
        raise ValueError('this resource contract reserves CUDA1 for existing jobs')
    with owner_lock(out):
        freeze_contract(out, cfg, root)
        try:
            plan = latent_health_plan(root / cfg['latent_health']['parent_run'], cfg['latent_health'])
            if (out / 'plan.json').exists() and read_json(out / 'plan.json') != serial(plan):
                raise ValueError('retained parent or development plan changed')
            write_json(out / 'plan.json', plan)
            if args.stage == 'latent-health-plan':
                return dict(status='planned', planned_trials=plan['planned_trials'], planned_windows=plan['planned_windows'], measured_arrays_read=0)
            if not (out / 'synthetic.json').exists() or read_json(out / 'synthetic.json')['status'] != 'passed':
                raise ValueError('synthetic software checks must pass before measured cache access')
            write_json(out / 'controller.json', dict(status='running', stage='prepare', pid=os.getpid(), updated=now()))
            prepared = prepare_latent_health(cfg, out, root, args.workers or cfg['resources']['cpu_workers'])
            if args.stage == 'latent-health-prepare':
                write_json(out / 'controller.json', dict(status='prepared', updated=now()))
                return dict(status='prepared', retained_trials=prepared['retained_trials'], retained_windows=prepared['retained_windows'])
            write_json(out / 'controller.json', dict(status='running', stage='benchmark', pid=os.getpid(), updated=now()))
            benchmark = benchmark_latent_health(cfg, args.config.resolve(), out, root, devices)
            if benchmark['status'] != 'completed':
                raise RuntimeError('resource benchmark failed; retained evidence requires review')
            if args.stage == 'latent-health-benchmark':
                write_json(out / 'controller.json', dict(status='benchmarked', updated=now()))
                return benchmark
            workers = benchmark['gpu_workers_per_device']; seeds = cfg['latent_health']['seeds']
            write_json(out / 'controller.json', dict(status='running', stage='warmup', pid=os.getpid(),
                devices=devices, workers=workers, updated=now(), estimated_training_seconds=benchmark['estimated_training_seconds']))
            warmup = latent_health_jobs(cfg, args.config.resolve(), out, root, devices, workers,
                [('warmup', seed) for seed in seeds])
            if any(r['status'] != 'completed' for r in warmup['results']):
                raise RuntimeError('continuous warmup failed; downstream jobs not substituted')
            write_json(out / 'controller.json', dict(status='running', stage='training', pid=os.getpid(),
                devices=devices, workers=workers, updated=now(), planned_jobs=9))
            trained = latent_health_jobs(cfg, args.config.resolve(), out, root, devices, workers,
                [(candidate, seed) for candidate in cfg['latent_health']['candidates'] for seed in seeds])
            summary = summarize_latent_health(cfg, out, trained['results'])
            write_json(out / 'controller.json', dict(status=summary['status'], stage='latent-health',
                pid=os.getpid(), updated=now(), planned_jobs=9, completed_jobs=summary['completed_jobs']))
            return dict(status=summary['status'], completed_jobs=summary['completed_jobs'], summary=str(out / 'summary.json'))
        except Exception:
            write_json(out / 'controller.json', dict(status='failed', stage=args.stage, pid=os.getpid(),
                updated=now(), error=traceback.format_exc()))
            raise


def _separation_prepare(payload):
    from src.data.dual_component_tokenizer import read_voltage_separation_input,separation_covariances
    cfg,out,plan,entry=payload;out=Path(out);path=out/'features'/f"{entry['id']}.npz"
    if path.exists():return dict(id=entry['id'],status='reused')
    started=time.monotonic()
    try:
        arrays,metadata=read_voltage_separation_input(plan,entry)
        cov=separation_covariances(arrays['x'],cfg['separation'])
        write_npz(path,**cov,hb=arrays['hb'],native_time=arrays['hb_native_time'],roi_indices=arrays['roi_indices'],
            channels=arrays['channels'],voltage_rms=np.sqrt(np.mean(arrays['x']**2,axis=0)))
        result=dict(id=entry['id'],status='completed',metadata=metadata,elapsed_seconds=time.monotonic()-started)
    except Exception:result=dict(id=entry['id'],status='failed',error=traceback.format_exc())
    write_json(out/'features'/f"{entry['id']}.json",result);return result


def _separation_state_fit(payload):
    from src.data.dual_component_tokenizer import produce_teacher,read_voltage_separation_input
    from src.data.ssm_prepared import EEG_PROXY_INPUT_SCHEMA
    from src.data.unified_physiology import CACHED_FNIRS_INPUT_SCHEMA
    cfg,out,plan,entry,roi=payload;out=Path(out);key=entry['id']+f'__{roi}';path=out/'physical'/key
    if path.with_suffix('.json').exists():return read_json(path.with_suffix('.json'))
    started=time.monotonic()
    try:
        base,_=read_voltage_separation_input(plan,entry);features=_voltage_teacher_read(out/'features'/f"{entry['id']}.npz")
        indices=base['roi_indices'][roi];co=read_json(out/'coordinates.json')[entry['rows'][roi]['coordinate']]
        arrays=dict(features=features['features'][:,indices].reshape(900,30),hb=base['hb'][:,roi],
            eeg=base['x'][6000:10000,indices].astype('float32'),native_time=base['hb_native_time'][:,roi],
            time_s=np.arange(900)*.1,identity=np.array(entry['rows'][roi]['id']),
            eeg_proxy_input=np.array(EEG_PROXY_INPUT_SCHEMA),eeg_proxy_rate_hz=np.array(10.),fnirs_input=np.array(CACHED_FNIRS_INPUT_SCHEMA))
        result,status=produce_teacher(arrays,co,read_json(out/'teacher_config.json'),eeg_visible=False)
        if result is not None:
            if result['visible'][:,0].any():raise AssertionError('EEG entered the Hb-only state fit')
            write_npz(path.with_suffix('.npz'),**result)
        report=dict(id=key,entry_id=entry['id'],roi=roi,phase=entry['phase'],subject=entry['subject'],
            status='completed' if status.get('accepted') else 'failed',fit=status,
            EEG_observation_rows_used=0,elapsed_seconds=time.monotonic()-started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:report=dict(id=key,entry_id=entry['id'],roi=roi,phase=entry['phase'],status='failed',error=traceback.format_exc())
    write_json(path.with_suffix('.json'),report);return report


def _separation_fir_features(hb,lags):
    n=len(hb);t=np.arange(n)
    return np.concatenate([hb[np.clip(t+round(lag*10),0,n-1)] for lag in lags],axis=1)


def _separation_guidance(cfg,out,plan):
    from src.data.dual_component_tokenizer import separation_state_epochs,phase_shuffled_states
    spec=cfg['separation'];out=Path(out);co=read_json(out/'coordinates.json')
    entries=plan['rows'];prepared={e['id']:_voltage_teacher_read(out/'features'/f"{e['id']}.npz") for e in entries}
    raw={};design={};physical={}
    for entry in entries:
        a=prepared[entry['id']];u=[];x=[];p=[]
        for roi,row in enumerate(entry['rows']):
            cc=co[row['coordinate']]
            u.append(a['features'][:,a['roi_indices'][roi]].reshape(900,30)@np.asarray(cc['pc'])*cc['eeg_factor'])
            x.append(_separation_fir_features(a['hb'][:,roi]*cc['hb_factor'],spec['empirical_lags_s']))
            path=out/'physical'/f"{entry['id']}__{roi}"
            status=read_json(path.with_suffix('.json'))
            p.append(_voltage_teacher_read(path.with_suffix('.npz'))['physical'][:,0] if status['status']=='completed' else np.full(900,np.nan))
        raw[entry['id']]=np.stack(u,axis=1);design[entry['id']]=x;physical[entry['id']]=np.stack(p,axis=1)
    fit=[e for e in entries if e['phase']=='fit'];models={};predictions={e['id']:np.zeros((900,3)) for e in entries};records=[]
    def train(group,roi,alpha):
        x=np.concatenate([design[e['id']][roi][150:750] for e in group]);y=np.concatenate([raw[e['id']][150:750,roi] for e in group])
        mean=x.mean(0);scale=np.maximum(x.std(0),1e-8);x=(x-mean)/scale;ym=y.mean()
        beta=np.linalg.solve(x.T@x/len(x)+alpha*np.eye(x.shape[1]),x.T@(y-ym)/len(x))
        return dict(mean=mean,scale=scale,beta=beta,intercept=ym)
    def predict(model,x):return (x-model['mean'])/model['scale']@model['beta']+model['intercept']
    for roi in range(3):
        scores=[]
        for alpha in spec['empirical_ridges']:
            errors=[]
            for held in fit:
                model=train([e for e in fit if e['subject']!=held['subject']],roi,alpha)
                y=predict(model,design[held['id']][roi]);errors.append(np.mean((y[150:750]-raw[held['id']][150:750,roi])**2))
            scores.append(dict(ridge=alpha,mean_subject_MSE=float(np.mean(errors))))
        alpha=min(scores,key=lambda r:r['mean_subject_MSE'])['ridge'];full=train(fit,roi,alpha)
        models[str(roi)]=dict(full,ridge=alpha,selection=scores)
        for entry in entries:
            model=train([e for e in fit if e['subject']!=entry['subject']],roi,alpha) if entry['phase']=='fit' else full
            predictions[entry['id']][:,roi]=predict(model,design[entry['id']][roi])
    write_json(out/'empirical_guidance.json',dict(models=models,fit_ids=[e['id'] for e in fit],
        first_window_predictions='leave_subject_out',selection='training_fit_subject_cross_validation',eeg_eval_inputs=False))
    for index,entry in enumerate(entries):
        key=entry['id'];sham=phase_shuffled_states(physical[key],spec['seed']+index)
        arrays={name:separation_state_epochs(value,spec) for name,value in
            [('physical',physical[key]),('empirical',predictions[key]),('eeg',raw[key]),('sham',sham)]}
        write_npz(out/'guidance'/f'{key}.npz',**arrays,raw_proxy=raw[key],physical_proxy=physical[key],empirical_proxy=predictions[key])


def _separation_model_fit(payload):
    from src.data.dual_component_tokenizer import (read_voltage_separation_input,fit_state_guided_subspace,
        fit_neighbor_predictor,apply_voltage_separation,separation_probe)
    cfg,out,plan,method=payload;out=Path(out);spec=cfg['separation'];path=out/'models'/f'{method}.json'
    if path.exists():return read_json(path)
    started=time.monotonic();fit=[];select=[];voltages=[];channels=None
    for entry in plan['rows']:
        if entry['phase']=='validation':continue
        cov=_voltage_teacher_read(out/'features'/f"{entry['id']}.npz")
        states=_voltage_teacher_read(out/'guidance'/f"{entry['id']}.npz")
        item=dict(phase=entry['phase'],cov=cov,states=states)
        (fit if entry['phase']=='fit' else select).append(item)
        if entry['phase']=='fit':
            base,_=read_voltage_separation_input(plan,entry);voltages.append(base['x']);channels=list(base['channels'])
    grid=[];failures=[]
    for rank in spec['ranks']:
        try:
            model=fit_state_guided_subspace(fit,select,rank,method,spec)
            model['predictor']=fit_neighbor_predictor(voltages,model,spec)
            write_npz(out/'models'/f'{method}_r{rank}.npz',**{k:v for k,v in model.items() if k not in ('selection','condition_number','idempotence_error','operator_norm')})
            write_json(out/'models'/f'{method}_r{rank}.json',{k:model[k] for k in ('selection','condition_number','idempotence_error','operator_norm')})
            coefficients=[]
            for seed in spec['probe_calibration_seeds']:
                for kind in ('ocular','fine','neural10','neural35','neural55'):
                    n=separation_probe(channels,seed,kind,spec);result=apply_voltage_separation(n,model,spec,innovation_fraction=1.)
                    target=n if kind.startswith('neural') else 0.
                    intercept=(result['s']+result['o_temporal']-target)[6000:10000]
                    slope=result['o_temporal'][6000:10000];denom=float(np.mean(n[6000:10000]**2))
                    coefficients.append((kind,float(np.mean(intercept**2)/denom),float(np.mean(intercept*slope)/denom),float(np.mean(slope**2)/denom)))
            for eta in spec['innovation_fractions']:
                scores={kind:float(np.mean([a-2*eta*b+eta*eta*c for k,a,b,c in coefficients if k==kind]))
                    for kind in ('ocular','fine','neural10','neural35','neural55')}
                score=(scores['ocular']+scores['fine']+2*sum(scores[k] for k in scores if k.startswith('neural')))/8
                grid.append(dict(method=method,rank=rank,innovation_fraction=eta,selection_score=score,**scores))
        except Exception:failures.append(dict(rank=rank,error=traceback.format_exc()))
    if grid:
        selected=min(grid,key=lambda r:(r['selection_score'],r['rank'],r['innovation_fraction']))
        report=dict(method=method,status='completed',selected=selected,grid=grid,failures=failures,elapsed_seconds=time.monotonic()-started)
    else:report=dict(method=method,status='failed',grid=[],failures=failures)
    write_json(path,report);return report


def _separation_load_models(out,cfg):
    spec=cfg['separation'];out=Path(out);models={};metadata={}
    for method in spec['guidance']:
        record=read_json(out/'models'/f'{method}.json')
        if record['status']!='completed':continue
        for name,choice in [(method,record['selected']),(method+'_matched',spec['fixed_comparison'])]:
            rank=choice['rank'];path=out/'models'/f'{method}_r{rank}.npz'
            if not path.exists():continue
            model=_voltage_teacher_read(path);model['innovation_fraction']=choice['innovation_fraction']
            models[name]=model;metadata[name]=dict(method=method,rank=rank,innovation_fraction=choice['innovation_fraction'])
    if 'physical' in models:
        models['physical_spatial']=dict(models['physical'],innovation_fraction=0.)
        metadata['physical_spatial']=dict(metadata['physical'],innovation_fraction=0.)
    return models,metadata


def _separation_synthetic(payload):
    from src.data.dual_component_tokenizer import (separation_synthetic_record,separation_covariances,
        separation_state_epochs,phase_shuffled_states,fit_state_guided_subspace,
        fit_neighbor_predictor,apply_voltage_separation)
    cfg,out,scenario=payload;out=Path(out);spec=cfg['separation'];folder=out/'synthetic'/scenario
    if (folder/'summary.json').exists():return read_json(folder/'summary.json')
    started=time.monotonic();fit=[];select=[];voltages=[];records=[];models={};failures=[]
    for slot,phase,items in [(0,'fit',fit),(1,'select',select)]:
        for seed in spec['synthetic_fit_seeds']:
            a=separation_synthetic_record(seed,slot,scenario)
            states=dict(physical=separation_state_epochs(a['state'],spec),
                sham=separation_state_epochs(phase_shuffled_states(a['state'],seed+slot*1000),spec))
            items.append(dict(phase=phase,cov=separation_covariances(a['x'],spec),states=states))
            if slot==0:voltages.append(a['x'])
    for method in ('physical','pca','sham'):
        try:
            model=fit_state_guided_subspace(fit,select,8,method,spec)
            model['predictor']=fit_neighbor_predictor(voltages,model,spec)
            for eta,suffix in [(0.,'_spatial'),(.5,'_temporal')]:
                models[method+suffix]=dict(model,innovation_fraction=eta)
        except Exception:failures.append(dict(method=method,error=traceback.format_exc()))
    for seed in spec['synthetic_eval_seeds']:
        a=separation_synthetic_record(seed,2,scenario);sl=slice(6000,10000)
        outputs={'raw':a['x'],'known_subspace':a['x']@a['true_neural_basis']@a['true_neural_basis'].T}
        outputs.update({name:apply_voltage_separation(a['x'],model,spec)['s'] for name,model in models.items()})
        for name,s in outputs.items():
            if name=='raw':parts={k:a[k] for k in ('clean','major','fine')}
            elif name=='known_subspace':parts={k:a[k]@a['true_neural_basis']@a['true_neural_basis'].T for k in ('clean','major','fine')}
            else:parts={k:apply_voltage_separation(a[k],models[name],spec)['s'] for k in ('clean','major','fine')}
            records.append(dict(scenario=scenario,seed=seed,method=name,
                clean_NMSE=float(np.mean((s[sl]-a['clean'][sl])**2)/np.mean(a['clean'][sl]**2)),
                major_S_NMSE=float(np.mean(parts['major'][sl]**2)/np.mean(a['major'][sl]**2)),
                fine_S_NMSE=float(np.mean(parts['fine'][sl]**2)/np.mean(a['fine'][sl]**2)),
                neural_loss_NMSE=float(np.mean((parts['clean'][sl]-a['clean'][sl])**2)/np.mean(a['clean'][sl]**2))))
        if seed==spec['synthetic_eval_seeds'][0]:
            write_npz(folder/'example.npz',**{k:a[k][sl] for k in ('x','clean','major','fine')},
                channels=a['channels'],**{k:v[sl] for k,v in outputs.items() if k!='raw'})
    report=dict(status='completed' if not failures else 'completed_with_failures',scenario=scenario,metrics=records,
        failures=failures,fit_seeds=spec['synthetic_fit_seeds'],evaluation_seeds=spec['synthetic_eval_seeds'],
        guidance='known_generator_state_not_estimated_Tak_state',rank=8,innovation_fraction=.5,
        elapsed_seconds=time.monotonic()-started)
    write_json(folder/'summary.json',report);return report


def _separation_words(value,scale):
    value=np.asarray(value,float)/np.asarray(scale,float)
    return value.reshape(-1,20,value.shape[1]).transpose(0,2,1).reshape(-1,20)


def _separation_hard(value,scale,book,levels):
    ids=book.encode(_separation_words(value,scale),levels=levels)
    decoded=book.decode(ids).reshape(-1,value.shape[1],20).transpose(0,2,1).reshape(value.shape)
    return decoded*np.asarray(scale)[None],ids.reshape(-1,value.shape[1],levels)


def _separation_book_fit(payload):
    from src.data.dual_component_tokenizer import read_voltage_separation_input,apply_voltage_separation
    from src.tokenizers.dual_component_tokenizer import WaveformDictionary
    cfg,out,plan,method,branch=payload;out=Path(out);spec=cfg['separation'];settings=spec['codebook']
    key=method+'_'+branch;path=out/'codebooks'/key
    if path.with_suffix('.json').exists():return read_json(path.with_suffix('.json'))
    started=time.monotonic()
    try:
        models,_=_separation_load_models(out,cfg);scale=np.asarray(read_json(out/'scales.json')['voltage_scale'])
        samples=[];fit_ids=[]
        for entry in plan['rows']:
            if entry['phase']!='fit':continue
            a,_=read_voltage_separation_input(plan,entry)
            value=a['x'] if method=='raw' else apply_voltage_separation(a['x'],models[method],spec)[branch.lower()]
            samples.append(_separation_words(value,scale));fit_ids.append(entry['id'])
        levels=(8 if method=='raw' or (method=='physical' and branch=='O') else 4)
        book=WaveformDictionary(settings['size'],levels,1,settings['derivative_weight'],
            settings['seed'],settings['fit_patches'],settings['iterations']).fit(np.concatenate(samples))
        write_npz(path.with_suffix('.npz'),**book.state())
        report=dict(status='completed',method=method,branch=branch,levels=levels,fit_ids=fit_ids,
            input_patches=sum(len(v) for v in samples),fit_patches=settings['fit_patches'],
            bits_per_second_per_electrode=70*levels,elapsed_seconds=time.monotonic()-started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:report=dict(status='failed',method=method,branch=branch,error=traceback.format_exc())
    write_json(path.with_suffix('.json'),report);return report


def _separation_evaluate(payload):
    from src.data.dual_component_tokenizer import read_voltage_separation_input,apply_voltage_separation,separation_probe
    from src.tokenizers.dual_component_tokenizer import WaveformDictionary
    from src.analysis.dual_component_tokenizer import separation_wave_metrics,separation_change_metrics,separation_decode_metrics
    cfg,out,plan,entry,donor_entry=payload;out=Path(out);spec=cfg['separation'];folder=out/'evaluation'/entry['id']
    if (folder/'metrics.json').exists():return read_json(folder/'metrics.json')
    started=time.monotonic()
    try:
        a,_=read_voltage_separation_input(plan,entry,include_eog=True)
        donor,_=read_voltage_separation_input(plan,donor_entry,include_eog=True)
        scale=np.asarray(read_json(out/'scales.json')['voltage_scale']);rms=float(np.sqrt(np.mean(scale**2)))
        models,metadata=_separation_load_models(out,cfg);x=a['x'];sl=slice(6000,10000)
        base={name:apply_voltage_separation(x,model,spec) for name,model in models.items()}
        base['raw']=dict(s=x,o=np.zeros_like(x));natural=[];changes=[];decoded=[];hard_changes=[]
        books={};hard_base={};stored={k:a[k] for k in ('x','eog','channels','legacy_s')}
        for name,result in base.items():
            natural.append(dict(subject=entry['subject'],id=entry['id'],method=name,
                **separation_wave_metrics(x,result['s'],result['o'],scale,a['eog'])))
            if name!='raw':stored[name]=result['s'].astype('float32')
        if 'physical' in base:
            stored.update(physical_o_spatial=base['physical']['o_spatial'].astype('float32'),
                physical_o_temporal=base['physical']['o_temporal'].astype('float32'))
        for name in spec['codebook']['candidates']:
            if name not in base:continue
            for branch in (['raw'] if name=='raw' else ['S','O']):
                key=name+'_'+branch;path=out/'codebooks'/key
                if not path.with_suffix('.json').exists() or read_json(path.with_suffix('.json'))['status']!='completed':continue
                book=WaveformDictionary.from_state(_voltage_teacher_read(path.with_suffix('.npz')));books[key]=book
                value=x if name=='raw' else base[name][branch.lower()]
                for levels in ([8] if name=='raw' else [4,8] if name=='physical' and branch=='O' else [4]):
                    pred,ids=_separation_hard(value[sl],scale,book,levels);hard_base[key+str(levels)]=pred
                    decoded.append(dict(subject=entry['subject'],id=entry['id'],method=name,branch=branch,levels=levels,
                        **separation_decode_metrics(value[sl],pred,scale)))
                    stored[f'hard_{name}_{branch}{levels}']=pred.astype('float32');stored[f'ids_{name}_{branch}{levels}']=ids
        write_npz(folder/'base.npz',**stored)
        # The encoder is frozen and linear before quantization. Every changed input is nevertheless re-evaluated.
        for kind in spec['measured_perturbations']:
            seed=spec['probe_evaluation_seed']+int(entry['subject'].removeprefix('VP'))*100+entry['slot']
            if kind=='combined':
                n0=separation_probe(a['channels'],seed,'ocular',spec,rms=1.,donor_eog=donor['eog'])
                n0+=separation_probe(a['channels'],seed+7,'fine',spec,rms=1.)
                n0/=np.sqrt(np.mean(n0[sl]**2))
            else:n0=separation_probe(a['channels'],seed,kind,spec,donor_eog=donor['eog'] if kind=='ocular' else None)
            for amplitude in spec['probe_amplitudes_training_rms']:
                n=n0*amplitude*rms;altered={name:apply_voltage_separation(x+n,model,spec) for name,model in models.items()}
                altered['raw']=dict(s=x+n,o=np.zeros_like(x));arrays=dict(x=(x+n)[sl].astype('float32'),n=n[sl].astype('float32'))
                for name,result in altered.items():
                    changes.append(dict(subject=entry['subject'],id=entry['id'],method=name,kind=kind,amplitude=amplitude,
                        **separation_change_metrics(base[name]['s'],result['s'],n,kind)))
                    if name!='raw':arrays[name]=result['s'][sl].astype('float32')
                for name in spec['codebook']['candidates']:
                    if name=='raw' or name not in altered or name+'_S' not in books or name+'_O' not in books:continue
                    hs,ids_s=_separation_hard(altered[name]['s'][sl],scale,books[name+'_S'],4)
                    for levels in ([4,8] if name=='physical' else [4]):
                        ho,ids_o=_separation_hard(altered[name]['o'][sl],scale,books[name+'_O'],levels)
                        label=name if levels==4 else name+'_O8'
                        # Pad only the scoring interface; padding never enters an encoder or filter.
                        pad=lambda v:np.pad(v,((6000,8000),(0,0)))
                        hard_changes.append(dict(subject=entry['subject'],id=entry['id'],method=label,kind=kind,amplitude=amplitude,
                            **separation_change_metrics(pad(hard_base[name+'_S4']),pad(hs),n,kind,
                                base_o=pad(hard_base[name+'_O'+str(levels)]),changed_o=pad(ho))))
                        arrays[f'hard_{label}_s']=hs.astype('float32');arrays[f'hard_{label}_o']=ho.astype('float32')
                        arrays[f'ids_{label}_s']=ids_s;arrays[f'ids_{label}_o']=ids_o
                if amplitude==.25:write_npz(folder/f'{kind}.npz',**arrays)
        report=dict(status='completed',entry=entry,donor_id=donor_entry['id'],natural=natural,changes=changes,
            decoded=decoded,hard_changes=hard_changes,elapsed_seconds=time.monotonic()-started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:report=dict(status='failed',entry=entry,error=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
    write_json(folder/'metrics.json',report);return report


def run_voltage_separation(cfg,out,root,workers,stage):
    from src.data.dual_component_tokenizer import voltage_separation_plan
    from src.data.ssm_prepared import fit_feature_coordinate,EEG_PROXY_INPUT_SCHEMA
    from src.data.unified_physiology import CACHED_FNIRS_INPUT_SCHEMA
    from src.analysis.dual_component_tokenizer import summarize_voltage_separation
    out,root=Path(out),Path(root);spec=cfg['separation']
    def progress(stage,total):
        def update(result,count):
            write_json(out/'controller.json',dict(status='running',stage=stage,pid=os.getpid(),updated=now(),
                workers=workers,planned=total,terminal=count,last_status=result.get('status')))
        update({},0);return update
    if stage!='separation-plan':
        jobs=[(cfg,str(out),s) for s in spec['synthetic_scenarios']]
        synthetic=_bounded_jobs(_separation_synthetic,jobs,min(workers,len(jobs)),inflight=1,
            progress=progress('synthetic',len(jobs)))
        write_json(out/'synthetic.json',synthetic)
        if any(not r.get('metrics') for r in synthetic):raise RuntimeError('synthetic execution failed; measured arrays remain unread')
        if stage=='synthetic':
            write_json(out/'controller.json',dict(status='completed',stage='synthetic',updated=now()))
            return dict(status='completed',scenarios=len(synthetic),measured_arrays_read=0)
    plan=voltage_separation_plan(root,cfg)
    if (out/'plan.json').exists() and read_json(out/'plan.json')!=plan:raise ValueError('immutable panel drift')
    write_json(out/'plan.json',plan)
    if stage=='separation-plan':return dict(status='planned',windows=len(plan['rows']),measured_arrays_read=0)
    prepared=_bounded_jobs(_separation_prepare,[(cfg,str(out),plan,e) for e in plan['rows']],workers,
        inflight=1,progress=progress('prepare',len(plan['rows'])))
    write_json(out/'preparation.json',prepared)
    if any(r['status']=='failed' for r in prepared):raise RuntimeError('incomplete fixed panel preparation; failures retained')
    if not (out/'coordinates.json').exists():
        chunks=defaultdict(list);ids=defaultdict(list);scales=[]
        for entry in plan['rows']:
            if entry['phase']!='fit':continue
            a=_voltage_teacher_read(out/'features'/f"{entry['id']}.npz");scales.append(a['voltage_rms']**2)
            for roi,row in enumerate(entry['rows']):
                chunks[row['coordinate']].append((a['features'][:,a['roi_indices'][roi]].reshape(900,30),a['hb'][:,roi]))
                ids[row['coordinate']].append(row['id'])
        coordinates={k:dict(fit_feature_coordinate(v),training_ids=ids[k],eeg_proxy_input=EEG_PROXY_INPUT_SCHEMA,
            fnirs_input=CACHED_FNIRS_INPUT_SCHEMA,feature_generation='unmodified_native_EEG_bandpower_new_training_coordinate') for k,v in chunks.items()}
        write_json(out/'coordinates.json',coordinates)
        write_json(out/'scales.json',dict(voltage_scale=np.sqrt(np.mean(scales,axis=0)),fit_ids=plan['fit_ids'],
            type='whole_90second_original_voltage_RMS_per_channel_no_centering'))
    teacher=yaml.safe_load((SOURCE/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_10hz_v2.yaml').read_text())
    write_json(out/'teacher_config.json',teacher)
    jobs=[(cfg,str(out),plan,e,roi) for e in plan['rows'] for roi in range(3)]
    if stage=='separation-pilot':jobs=jobs[:2]
    physical=_bounded_jobs(_separation_state_fit,jobs,min(workers,len(jobs)),inflight=1,progress=progress('physical',len(jobs)))
    write_json(out/('pilot.json' if stage=='separation-pilot' else 'physical_fits.json'),physical)
    if stage=='separation-pilot':
        write_json(out/'controller.json',dict(status='pilot_completed',updated=now()))
        return dict(status='pilot_completed',fits=physical)
    progress('guidance',1);_separation_guidance(cfg,out,plan)
    jobs=[(cfg,str(out),plan,m) for m in spec['guidance']]
    models=_bounded_jobs(_separation_model_fit,jobs,min(workers,len(jobs)),inflight=1,progress=progress('subspaces',len(jobs)))
    loaded,metadata=_separation_load_models(out,cfg);write_json(out/'model_selection.json',models)
    write_json(out/'selected_models.json',metadata)
    jobs=[(cfg,str(out),plan,m,b) for m in spec['codebook']['candidates'] if m=='raw' or m in loaded
        for b in (['raw'] if m=='raw' else ['S','O'])]
    books=_bounded_jobs(_separation_book_fit,jobs,min(workers,len(jobs)),inflight=1,progress=progress('codebooks',len(jobs)))
    write_json(out/'codebook_fits.json',books)
    donors=sorted([e for e in plan['rows'] if e['phase']=='validation' and e['slot']==1],key=lambda e:e['subject'])
    donor={e['subject']:donors[(i+1)%len(donors)] for i,e in enumerate(donors)}
    validation=[e for e in plan['rows'] if e['phase']=='validation']
    jobs=[(cfg,str(out),plan,e,donor[e['subject']]) for e in validation]
    evaluations=_bounded_jobs(_separation_evaluate,jobs,min(workers,len(jobs)),inflight=1,progress=progress('evaluation',len(jobs)))
    write_json(out/'evaluation_outcomes.json',[{k:v for k,v in r.items() if k not in ('natural','changes','decoded','hard_changes')} for r in evaluations])
    if not any(r['status']=='completed' for r in evaluations):raise RuntimeError('all validation windows failed; full outcomes retained')
    report=summarize_voltage_separation(out,plan,evaluations,physical,models,books,cfg)
    write_json(out/'summary.json',report)
    write_json(out/'controller.json',dict(status='completed',updated=now(),summary='summary.json'))
    return report


def voltage_power_cap_plan(cfg, root):
    from src.data.dual_component_tokenizer import voltage_separation_plan
    spec=cfg['power_cap'];root=Path(root);parent=root/spec['parent_run']
    original=read_json(parent/'resolved_config.json')['config']
    if (original['schema']!='dual_component_voltage_separation_v1'
            or original['separation']!=cfg['separation']
            or read_json(parent/'summary.json')['status']!='completed'):
        raise ValueError('the exact completed public separation parent is required')
    panel=voltage_separation_plan(root,cfg)
    if panel!=read_json(parent/'plan.json'):raise ValueError('parent panel identity drift')
    cases=[e for e in panel['rows'] if e['id']==spec['case_id'] and e['phase']=='validation']
    if len(cases)!=1:raise ValueError('one named validation case required before signal reads')
    pair=read_json(parent/spec['paired_input']/'result.json')
    joint=read_json(parent/spec['joint_fit']/'resolved_config.json')
    if (pair['case']['id']!=spec['case_id'] or joint['case_id']!=spec['case_id']
            or read_json(parent/spec['joint_fit']/'verification.json')['status']!='passed'
            or joint['visible_components']!=['EEG_log_bandpower_PCA_proxy','HbO','HbR']):
        raise ValueError('exact paired ocular input and verified joint fits required')
    training=[e for e in panel['rows'] if e['phase']=='fit']
    if len(training)!=6:raise ValueError('six frozen first training windows required')
    continuation=None
    if spec['physical'].get('same_objective_continuation'):
        continuation=root/spec['physical']['same_objective_continuation']
        prior=read_json(continuation/'resolved_config.json')['config']['power_cap']
        current=deepcopy(spec);prior=deepcopy(prior)
        for value in (current,prior):
            value['physical'].pop('solver_budget',None)
            value['physical'].pop('same_objective_continuation',None)
        if current!=prior or read_json(continuation/'controller.json')['status']!='completed':
            raise ValueError('same-objective continuation requires a terminal identical scientific contract')
    return dict(schema='voltage_power_cap_one_window_plan_v1',parent=str(parent.resolve()),
        panel=panel,case=cases[0],training=training,paired_input=str((parent/spec['paired_input']).resolve()),
        joint_fit=str((parent/spec['joint_fit']).resolve()),test_access='forbidden',protected_data='forbidden',
        observation_weights=spec['physical'].get('observation_weights',[1.,1.,1.]),
        solver_budget=spec['physical'].get('solver_budget',{}),
        continuation_source=str(continuation.resolve()) if continuation else None)


def _power_cap_fit(payload):
    from src.data.dual_component_tokenizer import produce_teacher
    from src.data.ssm_prepared import EEG_PROXY_INPUT_SCHEMA
    from src.data.unified_physiology import CACHED_FNIRS_INPUT_SCHEMA
    out_string,variant,roi=payload;out=Path(out_string);key=f'{variant}__{roi}'
    path=out/'physical'/key
    if path.with_suffix('.json').exists():return read_json(path.with_suffix('.json'))
    start=time.monotonic();plan=read_json(out/'plan.json');parent=Path(plan['parent'])
    weights=plan.get('observation_weights',[1.,1.,1.])
    reuse=roi==1 and variant in ('baseline','ocular') and weights==[1.,1.,1.]
    try:
        previous=None
        if plan.get('continuation_source'):
            previous_path=Path(plan['continuation_source'])/'physical'/(key+'.json')
            previous=read_json(previous_path)
            if previous['key']!=key or previous['observation_weights']!=weights:
                raise ValueError('continuation task identity or objective weight changed')
            if previous['status']=='completed':
                report=dict(previous,reused=True,continuation_source=str(previous_path))
                write_json(path.with_suffix('.json'),report);return report
        if reuse:
            arm='without_added_ocular' if variant=='baseline' else 'with_added_ocular'
            source=Path(plan['joint_fit'])/(arm+'.npz');status=read_json(source.with_suffix('.json'))
            if status['status']!='completed' or status['visible_observation_counts']!=[899,899,899]:
                raise ValueError('retained joint prefrontal fit is incomplete')
            fit=status['fit'];array_path=str(source)
        else:
            a=_voltage_teacher_read(out/'inputs.npz');indices=a['roi_indices'][roi]
            row=plan['case']['rows'][roi];co=read_json(out/'coordinates.json')[row['coordinate']]
            x=a['x'] if variant=='baseline' else a['x']+a['n_'+variant]
            prepared=dict(features=a['features_'+variant][:,indices].reshape(900,30),
                hb=a['hb'][:,roi],eeg=x[6000:10000,indices].astype('float32'),
                native_time=a['hb_native_time'][:,roi],time_s=np.arange(900)*.1,
                identity=np.array(row['id']),eeg_proxy_input=np.array(EEG_PROXY_INPUT_SCHEMA),
                eeg_proxy_rate_hz=np.array(10.),fnirs_input=np.array(CACHED_FNIRS_INPUT_SCHEMA))
            options=dict(eeg_visible=True)
            if weights!=[1.,1.,1.]:options['observation_weights']=weights
            if previous and previous.get('array_path') and Path(previous['array_path']).exists():
                saved=_voltage_teacher_read(previous['array_path'])
                expected=np.column_stack((prepared['features']@np.asarray(co['pc'])*co['eeg_factor'],prepared['hb']*co['hb_factor']))
                if not np.array_equal(expected,saved['u']):raise ValueError('continuation observations changed')
                options['warm_start']=dict(driver=saved['driver'],initial_state=saved['initial_state'])
            teacher=read_json(parent/'teacher_config.json')
            if plan.get('solver_budget'):teacher['solver'].update(plan['solver_budget'])
            result,fit=produce_teacher(prepared,co,teacher,**options)
            array_path=str(path.with_suffix('.npz'))
            if result is not None:
                if result['visible'].sum(0).tolist()!=[899,899,899]:raise AssertionError('joint visibility changed')
                write_npz(array_path,**result)
        report=dict(key=key,variant=variant,roi=roi,status='completed' if fit['accepted'] else 'failed',
            fit=fit,array_path=array_path,reused=reuse,observation_weights=weights,
            solver_budget=plan.get('solver_budget',{}),continuation_source=plan.get('continuation_source'),
            elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:report=dict(key=key,variant=variant,roi=roi,status='failed',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
    write_json(path.with_suffix('.json'),report);return report


def _power_cap_target(payload):
    from src.data.dual_component_tokenizer import voltage_power_cap_target,BANDS
    out_string,method,variant,cap=payload;out=Path(out_string)
    key=f'{method}__{variant}__cap{cap:g}';path=out/'targets'/key
    if path.with_suffix('.json').exists():return read_json(path.with_suffix('.json'))
    start=time.monotonic()
    try:
        a=_voltage_teacher_read(out/'inputs.npz');model=_voltage_teacher_read(out/'reference.npz')
        x=a['x'] if variant=='baseline' else a['x']+a['n_'+variant]
        proxy=None
        if method!='fixed_statistics':
            source='baseline' if method=='physical_frozen_reference' else variant
            states=read_json(out/'proxy_status.json')
            if states[source]['status']!='completed':raise ValueError('joint reference failed: '+source)
            with np.load(out/'proxies.npz',allow_pickle=False) as z:proxy=z[source]
        result=voltage_power_cap_target(x,model,proxy,cap=cap)
        final=result['final_reanalysis_ratio'];support=(result['frame_time_s']>=30)&(result['frame_time_s']<50)
        from src.inference.observation_baselines import eeg_band_power
        power=eeg_band_power(result['s'],bands=BANDS,sample_rate=200,target_rate=10)
        log=np.log(np.maximum(power,1e-20));log-=log[:50].mean(axis=0)
        measured_proxy=log.reshape(900,-1)@model['projection'].T
        save={k:v for k,v in result.items() if k not in ('s','o')}
        save.update(s_score=result['s'][6000:10000],o_score=result['o'][6000:10000],final_proxy=measured_proxy)
        write_npz(path.with_suffix('.npz'),**save)
        original=x[6000:10000];ss=result['s'][6000:10000]
        centred=original-original.mean(0);diff=(ss-original)-(ss-original).mean(0)
        report=dict(key=key,status='completed',method=method,variant=variant,cap=cap,
            local_max_ratio=float(result['local_output_ratio'].max()),
            final_score_max_ratio_by_band=final[support].max(0),
            final_score_violation_fraction_by_band=np.mean(final[support]>1+1e-6,axis=0),
            local_clipped_direction_fraction=np.mean(result['clipped_directions'][support]/x.shape[1],axis=0),
            baseline_or_input_AC_distortion=float(np.linalg.norm(diff)/np.linalg.norm(centred)),
            baseline_or_input_offset_change_RMS_uV=float(np.sqrt(np.mean((ss-original).mean(0)**2))),
            closure_max_abs=float(result['closure_max_abs']),elapsed_seconds=time.monotonic()-start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:report=dict(key=key,status='failed',method=method,variant=variant,cap=cap,
        error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
    write_json(path.with_suffix('.json'),report);return report


def run_voltage_power_cap_probe(cfg,out,root,workers,stage):
    from src.data.dual_component_tokenizer import (
        project_voltage_second_moment,read_voltage_separation_input,separation_covariances,
        fit_voltage_power_cap_reference,apply_voltage_separation,separation_probe,voltage_power_cap_target,
    )
    out,root=Path(out),Path(root);spec=cfg['power_cap']
    x=np.array([[.1,-.1,.1,-.1]]);n=np.array([[.05,.05,-.05,-.05]])
    low,_=project_voltage_second_moment(x,np.eye(1),1.)
    altered,_=project_voltage_second_moment(x+n,np.eye(1),1.)
    high,check=project_voltage_second_moment(x*100,np.eye(1),1.)
    synthetic=dict(status='passed',local_bound=float(check['max_output_eigenvalue']),
        inactive_cap_artifact_leakage=float(np.linalg.norm(altered-low)/np.linalg.norm(n)),
        saturated_RMS=float(np.sqrt(np.mean(high*high))),measured_arrays_read=0)
    if abs(synthetic['inactive_cap_artifact_leakage']-1)>1e-12 or synthetic['local_bound']>1+1e-12:
        raise AssertionError('analytic counterexample or bounded projection failed')
    if not (out/'synthetic.json').exists():
        started=time.monotonic();rng=np.random.default_rng(39)
        channels=cfg.get('posterior',{}).get('channels',16)
        demo=rng.normal(size=(18000,channels))*.2
        demo[:,0]+=8*np.exp(-.5*((np.arange(18000)/200-40)/.3)**2)
        a=np.eye(3,channels*5)
        model=dict(projection=a,mean=np.zeros(channels*5),lift=a.T,baseline_log_power=np.zeros(channels*5),
            correlation=np.tile(np.eye(channels),(6,1,1)),complement_power=np.ones(channels))
        target=voltage_power_cap_target(demo,model,np.zeros((900,3)),cap=1.)
        synthetic.update(channels=channels,full_channel_target_seconds=time.monotonic()-started,
            full_channel_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            closure_max_abs=float(target['closure_max_abs']),
            full_local_bound_ratio=float(target['local_output_ratio'].max()),
            final_reanalysis_measured=True)
        del demo,target
    else:
        synthetic=read_json(out/'synthetic.json')
    write_json(out/'synthetic.json',synthetic)
    weights=spec['physical'].get('observation_weights',[1.,1.,1.])
    if weights!=[1.,1.,1.] and not (out/'weighted_synthetic.json').exists():
        from src.data.dual_component_tokenizer import synthetic_teacher_check
        teacher=yaml.safe_load((SOURCE/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_10hz_v2.yaml').read_text())
        write_json(out/'controller.json',dict(status='running',stage='weighted_synthetic',pid=os.getpid(),updated=now()))
        weighted=synthetic_teacher_check(teacher,observation_weights=weights)
        write_json(out/'weighted_synthetic.json',weighted)
    elif weights!=[1.,1.,1.] and read_json(out/'weighted_synthetic.json')['status']!='completed':
        raise ValueError('weighted synthetic numerical verification is incomplete')
    if stage=='synthetic':return synthetic
    plan=voltage_power_cap_plan(cfg,root);parent=Path(plan['parent'])
    if (out/'plan.json').exists() and read_json(out/'plan.json')!=plan:raise ValueError('one-window plan drift')
    write_json(out/'plan.json',plan)
    if stage=='power-cap-plan':return dict(status='planned',training_windows=6,validation_windows=1,measured_arrays_read=0)
    def progress(label,total):
        def update(result,count):
            write_json(out/'controller.json',dict(status='running',stage=label,planned=total,terminal=count,
                workers=workers,pid=os.getpid(),updated=now(),last_status=result.get('status')))
        update({},0);return update
    if not (out/'reference.npz').exists():
        progress('prepare_and_train_statistics',1)
        base,_=read_voltage_separation_input(plan['panel'],plan['case'])
        case=plan['case'];coordinates=read_json(parent/'coordinates.json')
        a=np.zeros((3,80))
        for roi,row in enumerate(case['rows']):
            co=coordinates[row['coordinate']];load=np.asarray(co['pc']).reshape(6,5)*co['eeg_factor']
            for j,c in enumerate(base['roi_indices'][roi]):a[roi,c*5:(c+1)*5]=load[j]
        training=[]
        for entry in plan['training']:
            v,_=read_voltage_separation_input(plan['panel'],entry)
            if not np.array_equal(v['channels'],base['channels']):raise ValueError('native channel identities differ')
            training.append(dict(split=entry['split'],x=v['x']))
        model=fit_voltage_power_cap_reference(training,a,shrinkage=spec['reference']['shrinkage'])
        with np.load(Path(plan['paired_input'])/'state_proxy_check_v1/diagnostic_inputs.npz',allow_pickle=False) as z:
            if not np.array_equal(z['Fp1_before'],base['x'][:,list(base['channels']).index('Fp1')]):
                raise ValueError('retained single-channel input changed')
            known=z['added_ocular'].copy()
        injection=np.zeros_like(base['x']);injection[:,list(base['channels']).index('Fp1')]=known
        inputs={k:base[k] for k in ('x','hb','hb_native_time','roi_indices','channels')};inputs['n_ocular']=injection
        amplitude=float(np.sqrt(np.mean(injection[6000:10000]**2)))
        for kind in spec['interventions']['neural']:
            inputs['n_'+kind]=separation_probe(list(base['channels']),spec['interventions']['neural_seed'],kind,
                cfg['separation'],rms=amplitude)
        for variant in ['baseline','ocular',*spec['interventions']['neural']]:
            value=base['x'] if variant=='baseline' else base['x']+inputs['n_'+variant]
            inputs['features_'+variant]=separation_covariances(value,cfg['separation'])['features']
        original=_voltage_teacher_read(parent/'features'/f"{case['id']}.npz")
        if not np.array_equal(inputs['features_baseline'],original['features']):raise ValueError('original proxy features changed')
        delta=(inputs['features_ocular']-inputs['features_baseline']).reshape(900,80)
        parallel=delta@np.linalg.pinv(a)@a;perpendicular=delta-parallel;sl=slice(300,500)
        norm=float(np.sum(delta[sl]**2))
        write_json(out/'proxy_nullspace.json',dict(projection_rank=int(np.linalg.matrix_rank(a)),feature_dimensions=80,
            parallel_squared_norm_fraction=float(np.sum(parallel[sl]**2)/norm),
            null_squared_norm_fraction=float(np.sum(perpendicular[sl]**2)/norm),
            orthogonality_error=float(abs(np.sum(parallel[sl]*perpendicular[sl]))),
            scope='Euclidean squared norm of referenced log-power changes, not voltage energy'))
        write_json(out/'coordinates.json',coordinates);write_npz(out/'inputs.npz',**inputs)
        write_npz(out/'reference.npz',**model)
    variants=['baseline','ocular',*spec['interventions']['neural']]
    jobs=[(str(out),v,r) for v in variants for r in range(3) if v!='ocular' or r==1]
    fits=_bounded_jobs(_power_cap_fit,jobs,min(workers,len(jobs)),inflight=1,progress=progress('joint_fits',len(jobs)))
    indexed={(r['variant'],r['roi']):r for r in fits};proxy={};statuses={}
    for variant in variants:
        records=[indexed[('baseline' if variant=='ocular' and roi!=1 else variant,roi)] for roi in range(3)]
        status='completed' if all(r['status']=='completed' for r in records) else 'failed'
        statuses[variant]=dict(status=status,records=[r['key'] for r in records])
        if status=='completed':proxy[variant]=np.stack([_voltage_teacher_read(r['array_path'])['physical'][:,0] for r in records],axis=1)
    write_json(out/'physical_fits.json',fits);write_json(out/'proxy_status.json',statuses);write_npz(out/'proxies.npz',**proxy)
    inputs=_voltage_teacher_read(out/'inputs.npz')
    previous=read_json(Path(plan['paired_input'])/'result.json')['model']
    weights=Path(previous['weights'])
    if weights.resolve().parent!=parent.resolve()/'models':raise ValueError('old target weights outside the declared parent')
    old=_voltage_teacher_read(weights);old['innovation_fraction']=previous['innovation_fraction']
    for variant in variants:
        value=inputs['x'] if variant=='baseline' else inputs['x']+inputs['n_'+variant]
        target=apply_voltage_separation(value,old,cfg['separation'])
        write_npz(out/'targets'/f'current_target__{variant}.npz',s_score=target['s'][6000:10000],o_score=target['o'][6000:10000])
    caps=[spec['reference']['cap_primary'],*spec['reference']['cap_sensitivity']]
    jobs=[]
    for cap in caps:
        for variant in variants:
            methods=['fixed_statistics','physical_refitted_reference'] if variant=='baseline' else spec['conditions'][1:]
            jobs.extend((str(out),m,variant,cap) for m in methods)
    results=_bounded_jobs(_power_cap_target,jobs,min(workers,len(jobs)),inflight=1,progress=progress('voltage_targets',len(jobs)))
    write_json(out/'target_outcomes.json',results)
    from src.analysis.dual_component_tokenizer import summarize_voltage_power_cap
    summary=summarize_voltage_power_cap(out,cfg,plan,results,fits)
    write_json(out/'summary.json',summary)
    write_json(out/'controller.json',dict(status='completed',stage='complete',updated=now(),summary='summary.json'))
    return dict(status='completed',summary=str(out/'summary.json'),target_tasks=len(results))



def run_voltage_power_cap_examples(cfg,out,root,workers,stage):
    """Four fixed public examples; retain both first and continuation outcomes."""
    from src.data.dual_component_tokenizer import (
        voltage_separation_plan,voltage_power_cap_example_cases,
        read_voltage_separation_input,separation_covariances,
    )
    out,root=Path(out),Path(root);spec=cfg['examples']
    # Full-size synthetic uses the identical numerical owner, with no measured read.
    if stage=='synthetic':
        return run_voltage_power_cap_probe(cfg,out,root,workers,'synthetic')
    panel=voltage_separation_plan(root,cfg)
    source=root/spec['reference_run']
    source_plan=read_json(source/'plan.json');source_cfg=read_json(source/'resolved_config.json')['config']
    source_summary=read_json(source/'summary.json')
    if (source_plan['panel']!=panel or source_cfg['power_cap']!=cfg['power_cap']
            or source_summary['status']!='completed'
            or source_summary['counts']['fit_failures'] or source_summary['counts']['target_failures']
            or read_json(source/'verification.json')['status']!='passed'
            or read_json(source/'controller.json')['status']!='completed'):
        raise ValueError('exact verified completed reference setting required before arrays')
    cases=voltage_power_cap_example_cases(panel,source_plan['case']['id'],spec['subjects'])
    parent=Path(source_plan['parent'])
    plan=dict(schema='voltage_power_cap_examples_plan_v1',reference_run=str(source.resolve()),
        parent=str(parent),panel=panel,cases=cases,original_case=source_plan['case'],
        selection=spec['selection'],independent_subjects=5,new_subjects=4,
        test_access='forbidden',protected_data='forbidden',measured_arrays_read=0,
        injection='identical retained waveform in Fp1; donor VP005 second retained window',
        donor_overlap='VP005 example uses a different window from the same subject as the fixed donor',
        stopping=spec['stopping'],selection_use=spec['selection_use'])
    if (out/'plan.json').exists() and read_json(out/'plan.json')!=plan:
        raise ValueError('examples plan changed')
    write_json(out/'plan.json',plan)
    if stage=='power-cap-plan':
        return dict(status='planned',new_windows=4,original_windows=1,measured_arrays_read=0)
    if not (out/'synthetic.json').exists() or not (out/'weighted_synthetic.json').exists():
        run_voltage_power_cap_probe(cfg,out,root,workers,'synthetic')
    if (read_json(out/'synthetic.json')['status']!='passed'
            or read_json(out/'weighted_synthetic.json')['status']!='completed'):
        raise ValueError('synthetic verification required before measured arrays')
    def progress(label,total):
        def update(result,count):
            write_json(out/'controller.json',dict(status='running',stage=label,planned=total,terminal=count,
                workers=workers,pid=os.getpid(),updated=now(),last_status=result.get('status')))
        update({},0);return update
    reference=_voltage_teacher_read(source/'reference.npz')
    source_inputs=_voltage_teacher_read(source/'inputs.npz')
    coordinates=read_json(source/'coordinates.json');prepared=[]
    for case in cases:
        folder=out/'cases'/case['subject'];status=folder/'prepare.json'
        if status.exists():
            prepared.append(read_json(status));continue
        start=time.monotonic()
        try:
            base,metadata=read_voltage_separation_input(panel,case)
            for key in ('channels','roi_indices'):
                if not np.array_equal(base[key],source_inputs[key]):
                    raise ValueError('example channel/ROI mapping differs from frozen reference')
            inputs={k:base[k] for k in ('x','hb','hb_native_time','roi_indices','channels')}
            inputs['n_ocular']=source_inputs['n_ocular'].copy()
            fp=list(base['channels']).index('Fp1')
            if np.any(inputs['n_ocular'][:,np.arange(16)!=fp]):
                raise ValueError('retained perturbation must be Fp1 only')
            for variant in spec['variants']:
                x=base['x'] if variant=='baseline' else base['x']+inputs['n_ocular']
                inputs['features_'+variant]=separation_covariances(x,cfg['separation'])['features']
            retained=_voltage_teacher_read(parent/'features'/f"{case['id']}.npz")
            if not np.array_equal(inputs['features_baseline'],retained['features']):
                raise ValueError('retained baseline feature replay differs')
            case_plan=dict(parent=str(parent),case=case,observation_weights=[.1,10.,10.],
                solver_budget=spec['initial_solver_budget'],continuation_source=None,
                test_access='forbidden',protected_data='forbidden')
            write_json(folder/'plan.json',case_plan);write_json(folder/'coordinates.json',coordinates)
            write_npz(folder/'inputs.npz',**inputs);write_npz(folder/'reference.npz',**reference)
            record=dict(subject=case['subject'],case_id=case['id'],status='completed',
                case_root=str(folder.resolve()),metadata=metadata,seconds=time.monotonic()-start)
        except Exception:
            record=dict(subject=case['subject'],case_id=case['id'],status='failed',
                error=traceback.format_exc(),seconds=time.monotonic()-start)
        write_json(status,record);prepared.append(record)
    write_json(out/'preparation.json',prepared)
    valid=[r for r in prepared if r['status']=='completed']
    jobs=[(r['case_root'],variant,roi) for r in valid
        for variant,roi in [('baseline',0),('baseline',1),('baseline',2),('ocular',1)]]
    first=_bounded_jobs(_power_cap_fit,jobs,min(workers,max(1,len(jobs))),inflight=1,
        progress=progress('initial_joint_fits',len(jobs))) if jobs else []
    # Task records are read by identity because completed jobs return asynchronously.
    continuation=[]
    for record in valid:
        folder=Path(record['case_root']);retry=folder/'continuation_v1'
        failed=[]
        for variant,roi in [('baseline',0),('baseline',1),('baseline',2),('ocular',1)]:
            fit=read_json(folder/'physical'/f'{variant}__{roi}.json')
            if fit['status']!='completed':failed.append((variant,roi))
        if failed:
            new_plan=dict(read_json(folder/'plan.json'),continuation_source=str(folder),
                solver_budget=spec['continuation_solver_budget'])
            write_json(retry/'plan.json',new_plan);write_json(retry/'coordinates.json',coordinates)
            # A local continuation shares the immutable inputs through a hard link.
            destination=retry/'inputs.npz'
            if not destination.exists():os.link(folder/'inputs.npz',destination)
            continuation.extend((str(retry),variant,roi) for variant,roi in failed)
    second=_bounded_jobs(_power_cap_fit,continuation,min(workers,max(1,len(continuation))),inflight=1,
        progress=progress('same_objective_continuation',len(continuation))) if continuation else []
    fit_records=[]
    for record in valid:
        folder=Path(record['case_root']);indexed={};attempts=[]
        for variant,roi in [('baseline',0),('baseline',1),('baseline',2),('ocular',1)]:
            name=f'{variant}__{roi}.json'
            first_path=folder/'physical'/name;fit=read_json(first_path)
            attempts.append(dict(fit,attempt=1,record_path=str(first_path)))
            second_path=folder/'continuation_v1/physical'/name
            if second_path.exists():
                fit=read_json(second_path);attempts.append(dict(fit,attempt=2,record_path=str(second_path)))
            indexed[(variant,roi)]=fit
            fit_records.append(dict(fit,subject=record['subject'],case_id=record['case_id']))
        write_json(folder/'fit_attempts.json',attempts)
        write_json(folder/'physical_fits.json',list(indexed.values()))
        proxies={};states={}
        for variant in spec['variants']:
            fits=[indexed[('baseline' if variant=='ocular' and roi!=1 else variant,roi)] for roi in range(3)]
            accepted=all(fit['status']=='completed' for fit in fits)
            states[variant]=dict(status='completed' if accepted else 'failed',records=[f['key'] for f in fits])
            if accepted:
                proxies[variant]=np.stack([_voltage_teacher_read(fit['array_path'])['physical'][:,0] for fit in fits],axis=1)
        write_json(folder/'proxy_status.json',states);write_npz(folder/'proxies.npz',**proxies)
    write_json(out/'physical_fits.json',fit_records)
    jobs=[(r['case_root'],'physical_refitted_reference',variant,1.) for r in valid for variant in spec['variants']]
    targets=_bounded_jobs(_power_cap_target,jobs,min(workers,max(1,len(jobs))),inflight=1,
        progress=progress('voltage_targets',len(jobs))) if jobs else []
    # Include the exact original example, without refitting or replacing its evidence.
    entries=[dict(case=source_plan['case'],run=str(source),reused=True)]
    entries.extend(dict(case=case,run=str(out/'cases'/case['subject']),reused=False) for case in cases)
    from src.analysis.dual_component_tokenizer import summarize_voltage_power_cap_examples
    summary=summarize_voltage_power_cap_examples(entries,fit_records)
    summary.update(experiment_id=cfg['experiment_id'],plan='plan.json',
        preparation=prepared,initial_fit_outcomes=first,continuation_fit_outcomes=second,
        new_target_outcomes=targets,observation_weights=[.1,10.,10.],primary_cap=1.)
    write_json(out/'summary.json',summary)
    write_json(out/'controller.json',dict(status='completed',stage='complete',updated=now(),summary='summary.json'))
    return dict(status='completed',summary=str(out/'summary.json'),counts=summary['counts'])



def _power_cap_posterior_prepare(payload):
    from src.data.dual_component_tokenizer import prepare_voltage_power_cap_posterior
    cfg,root,out,entry=payload;out=Path(out);folder=out/'prepared'/entry['id']
    status=folder.with_suffix('.json')
    if status.exists():return read_json(status)
    started=time.monotonic()
    try:
        plan=read_json(out/'plan.json')
        arrays,metadata=prepare_voltage_power_cap_posterior(root,cfg,plan,entry)
        write_npz(folder.with_suffix('.npz'),**arrays)
        result=dict(status='completed',case_id=entry['id'],phase=entry['phase'],metadata=metadata,
            array_path=str(folder.with_suffix('.npz')),elapsed_seconds=time.monotonic()-started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        result=dict(status='failed',case_id=entry['id'],phase=entry['phase'],error=traceback.format_exc())
    write_json(status,result);return result


def run_voltage_power_cap_posterior(cfg,out,root,workers,stage):
    from src.data.dual_component_tokenizer import (
        voltage_power_cap_posterior_plan,fit_voltage_power_cap_reference,separation_covariances,
    )
    from src.data.ssm_prepared import fit_feature_coordinate,EEG_PROXY_INPUT_SCHEMA
    from src.data.unified_physiology import CACHED_FNIRS_INPUT_SCHEMA
    out,root=Path(out),Path(root);spec=cfg['posterior']
    if stage=='synthetic':return run_voltage_power_cap_probe(cfg,out,root,workers,'synthetic')
    plan=voltage_power_cap_posterior_plan(root,cfg)
    if (out/'plan.json').exists() and read_json(out/'plan.json')!=plan:raise ValueError('posterior plan drift')
    write_json(out/'plan.json',plan)
    if stage=='power-cap-plan':
        return dict(status='planned',training_windows=6,validation_windows=2,channels=17,measured_arrays_read=0)
    if not (out/'synthetic.json').exists() or not (out/'weighted_synthetic.json').exists():
        run_voltage_power_cap_probe(cfg,out,root,workers,'synthetic')
    if (read_json(out/'synthetic.json')['status']!='passed'
            or read_json(out/'weighted_synthetic.json')['status']!='completed'):
        raise ValueError('posterior numerical synthetic checks required before signals')
    def progress(label,total):
        def update(result,count):
            write_json(out/'controller.json',dict(status='running',stage=label,planned=total,terminal=count,
                workers=workers,pid=os.getpid(),updated=now(),last_status=result.get('status')))
        update({},0);return update
    prepared=_bounded_jobs(_power_cap_posterior_prepare,
        [(cfg,str(root),str(out),entry) for entry in plan['rows']],workers,inflight=1,
        progress=progress('prepare_posterior_inputs',8))
    write_json(out/'preparation.json',prepared)
    if any(r['status']!='completed' for r in prepared):
        raise RuntimeError('posterior preparation failure retained; no changed panel or imputation')
    source=Path(plan['reference_run']);source_inputs=_voltage_teacher_read(source/'inputs.npz')
    known=source_inputs['n_ocular'][:,list(source_inputs['channels']).index('Fp1')]
    original_coordinates=read_json(source/'coordinates.json')
    coordinates={row['coordinate']:original_coordinates[row['coordinate']]
        for row in plan['training'][0]['rows'][:2]}
    training=[];chunks=[];ids=[]
    for entry in plan['training']:
        a=_voltage_teacher_read(out/'prepared'/(entry['id']+'.npz'))
        training.append(dict(split='train',x=a['x']))
        chunks.append((a['features_baseline'][:,a['roi_indices'][2]].reshape(900,30),a['hb'][:,2]))
        ids.append(entry['rows'][2]['id'])
    key='simultaneous_eeg_nirs__occipital'
    coordinates[key]=dict(fit_feature_coordinate(chunks),training_ids=ids,eeg_proxy_input=EEG_PROXY_INPUT_SCHEMA,
        fnirs_input=CACHED_FNIRS_INPUT_SCHEMA,feature_generation='unmodified_native_EEG_bandpower_new_training_coordinate')
    write_json(out/'coordinates.json',coordinates)
    first=_voltage_teacher_read(out/'prepared'/(plan['training'][0]['id']+'.npz'))
    projection=np.zeros((3,85))
    for roi,row in enumerate(plan['training'][0]['rows']):
        co=coordinates[row['coordinate']];loading=np.asarray(co['pc']).reshape(6,5)*co['eeg_factor']
        for j,c in enumerate(first['roi_indices'][roi]):projection[roi,c*5:(c+1)*5]=loading[j]
    reference=fit_voltage_power_cap_reference(training,projection)
    write_npz(out/'reference.npz',**reference)
    jobs=[];entries=[]
    for case in plan['cases']:
        folder=out/'cases'/case['subject'];a=_voltage_teacher_read(out/'prepared'/(case['id']+'.npz'))
        if not np.array_equal(a['channels'],first['channels']):raise ValueError('posterior channel order changed')
        n=np.zeros_like(a['x']);n[:,list(a['channels']).index(case['injection_channel'])]=known
        a['n_ocular']=n;a['features_ocular']=separation_covariances(a['x']+n,cfg['separation'])['features']
        write_npz(folder/'inputs.npz',**a);write_npz(folder/'reference.npz',**reference)
        write_json(folder/'coordinates.json',coordinates)
        case_plan=dict(parent=plan['parent'],case=case,observation_weights=[.1,10.,10.],
            solver_budget=spec['initial_solver_budget'],continuation_source=None,
            test_access='forbidden',protected_data='forbidden')
        write_json(folder/'plan.json',case_plan)
        prior=Path(case['original_physical_run'])
        old_fits=read_json(prior/'physical_fits.json')
        old_co=read_json(prior/'coordinates.json')
        for roi in (0,1):
            indices=a['roi_indices'][roi];row=case['rows'][roi];co=coordinates[row['coordinate']]
            if co!=old_co[row['coordinate']]:raise ValueError('unchanged ROI coordinate changed')
            if not np.array_equal(a['features_baseline'][:,indices],a['features_ocular'][:,indices]):
                raise ValueError('posterior injection reached an allegedly unaffected ROI')
            fit=next(r for r in old_fits if r['variant']=='baseline' and r['roi']==roi)
            expected=np.column_stack((a['features_baseline'][:,indices].reshape(900,30)@np.asarray(co['pc'])*co['eeg_factor'],
                a['hb'][:,roi]*co['hb_factor']))
            old=_voltage_teacher_read(fit['array_path'])
            if (fit['status']!='completed' or fit['observation_weights']!=[.1,10.,10.]
                    or not np.array_equal(old['u'],expected)):
                raise ValueError('baseline reuse requires accepted same-weight exact observations')
            write_json(folder/'physical'/f'baseline__{roi}.json',dict(fit,reused=True,
                reuse_source=str(prior/'physical_fits.json'),same_observations_exact=True))
        jobs.extend((str(folder),variant,2) for variant in ('baseline','ocular'))
        entries.append(dict(case=case,run=str(folder),reused=False,display_channel=case['injection_channel'],metric_roi=2))
    initial=_bounded_jobs(_power_cap_fit,jobs,min(workers,4),inflight=1,
        progress=progress('posterior_joint_fits',4))
    retry_jobs=[]
    for entry in entries:
        folder=Path(entry['run']);retry=folder/'continuation_v1'
        failed=[v for v in ('baseline','ocular') if read_json(folder/'physical'/f'{v}__2.json')['status']!='completed']
        if failed:
            write_json(retry/'plan.json',dict(read_json(folder/'plan.json'),continuation_source=str(folder),
                solver_budget=spec['continuation_solver_budget']))
            write_json(retry/'coordinates.json',coordinates)
            if not (retry/'inputs.npz').exists():os.link(folder/'inputs.npz',retry/'inputs.npz')
            retry_jobs.extend((str(retry),v,2) for v in failed)
    continued=_bounded_jobs(_power_cap_fit,retry_jobs,min(workers,max(1,len(retry_jobs))),inflight=1,
        progress=progress('same_objective_continuation',len(retry_jobs))) if retry_jobs else []
    new_fits=[];target_jobs=[]
    for entry in entries:
        folder=Path(entry['run']);physical=[read_json(folder/'physical'/f'baseline__{roi}.json') for roi in (0,1)]
        attempts=[]
        for variant in ('baseline','ocular'):
            path=folder/'physical'/f'{variant}__2.json';fit=read_json(path)
            attempts.append(dict(fit,attempt=1))
            alternative=folder/'continuation_v1/physical'/path.name
            if alternative.exists():fit=read_json(alternative);attempts.append(dict(fit,attempt=2))
            physical.append(fit);new_fits.append(dict(fit,subject=entry['case']['subject']))
        write_json(folder/'fit_attempts.json',attempts);write_json(folder/'physical_fits.json',physical)
        proxies={};states={}
        for variant in ('baseline','ocular'):
            records=[physical[0],physical[1],next(r for r in physical if r['roi']==2 and r['variant']==variant)]
            accepted=all(r['status']=='completed' for r in records)
            states[variant]=dict(status='completed' if accepted else 'failed',records=[r['key'] for r in records])
            if accepted:proxies[variant]=np.stack([_voltage_teacher_read(r['array_path'])['physical'][:,0] for r in records],axis=1)
        write_json(folder/'proxy_status.json',states);write_npz(folder/'proxies.npz',**proxies)
        target_jobs.extend((str(folder),'physical_refitted_reference',v,1.) for v in ('baseline','ocular'))
    targets=_bounded_jobs(_power_cap_target,target_jobs,min(workers,4),inflight=1,
        progress=progress('posterior_voltage_targets',4))
    from src.analysis.dual_component_tokenizer import summarize_voltage_power_cap_examples
    summary=summarize_voltage_power_cap_examples(entries,new_fits,planned_new_fits=4)
    summary.update(schema='voltage_power_cap_posterior_results_v1',experiment_id=cfg['experiment_id'],
        preparation=prepared,initial_fit_outcomes=initial,continuation_fit_outcomes=continued,
        new_target_outcomes=targets,observation_weights=[.1,10.,10.],primary_cap=1.,
        reused_unchanged_ROI_fits=4,regions=spec['regions'],channels=17,label_start='F',
        interpretation=spec['claim'],posterior_training_ids=ids)
    write_json(out/'physical_fits.json',new_fits);write_json(out/'summary.json',summary)
    write_json(out/'controller.json',dict(status='completed',stage='complete',updated=now(),summary='summary.json'))
    return dict(status='completed',counts=summary['counts'],summary=str(out/'summary.json'))


if __name__ == '__main__':
    main()
