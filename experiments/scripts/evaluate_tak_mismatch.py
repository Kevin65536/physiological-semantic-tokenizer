#!/usr/bin/env python3
"""Complete public-inventory Tak mismatch analysis with resumable local tasks.

Data admission and numerical definitions have their own library owners. This
entry point owns the immutable plan, fold fitting, scheduling and evidence only.
The default ``check`` opens no measured arrays. Substantial work must be launched
under the durable supervisor specified by the project operating contract.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

for _variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '1'

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import yaml

from src.inference.t3a_balloon_robust_ssm import BalloonFixedParameters, BalloonFreeParameters, BalloonParameters

DEFAULT_CONFIG = ROOT/'experiments/configs/physiology_semantic_tokenizer/tak_mismatch_v1.yaml'
ARTIFACT_ROOT = 'experiments/runs/physiology_semantic_tokenizer/tak_mismatch'
STAGES = ('check', 'synthetic', 'plan', 'prepare', 'run', 'summarize', 'audit-status')
CANDIDATES = ('nearest6', 'sixth_to_seventh', 'next6', 'random6')
METRICS = ('D1', 'D_EH', 'D_obs', 'D_obs_ar1_sd0p05', 'D_obs_ar1_sd0p2', 'D_obs_iid_sd0p1',
           'D_drive_H', 'Delta_spatial', 'D4', 'D_J_E', 'D_J_O', 'D_J_R',
           'HbO_RMSE_native', 'HbR_RMSE_native', 'D_bal_10s', 'D_bal_20s', 'D_bal_40s',
           'prediction_NMSE', 'selected_prediction_NMSE', 'zero_NMSE', 'template_NMSE',
           'wrong_NMSE', 'selected_wrong_NMSE', 'pairing_gain', 'selected_pairing_gain')
ROOT_METRICS = {'D1', 'D_EH', 'D_obs', 'D_obs_ar1_sd0p05', 'D_obs_ar1_sd0p2', 'D_obs_iid_sd0p1',
                'D_drive_H', 'D4', 'D_J_E', 'D_J_O', 'D_J_R',
                'HbO_RMSE_native', 'HbR_RMSE_native'}
FREE_METRICS = tuple(metric+'_free' for metric in ('D1', 'D_EH', 'D_obs', 'D_drive_H', 'D4',
    'D_J_E', 'D_J_O', 'D_J_R', 'D_bal_10s', 'D_bal_20s', 'D_bal_40s',
    'D_obs_ar1_sd0p05', 'D_obs_ar1_sd0p2', 'D_obs_iid_sd0p1'))
ROOT_METRICS.update(metric+'_free' for metric in tuple(ROOT_METRICS))


def serial(value):
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)): return [serial(v) for v in value]
    if isinstance(value, np.generic): return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(serial(value), ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    temporary.replace(path)


def write_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(temporary, **arrays); temporary.replace(path)


def token(identity):
    return hashlib.sha256(str(identity).encode()).hexdigest()[:24]


def read_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    expected = yaml.safe_load(DEFAULT_CONFIG.read_text())
    for key in expected:
        if key not in ('solver', 'resources') and cfg.get(key) != expected[key]:
            raise ValueError(f'versioned Tak mismatch contract drift: {key}')
    if cfg.get('schema') != 'tak_mismatch_v1' or cfg.get('artifact_root') != ARTIFACT_ROOT:
        raise ValueError('wrong versioned Tak mismatch namespace')
    flexible = {'max_iterations', 'max_evaluations_per_trial', 'gradient_tolerance'}
    for key, value in expected['solver'].items():
        if key not in flexible and cfg['solver'].get(key) != value:
            raise ValueError(f'fixed Tak solver contract drift: {key}')
    if any(not np.isfinite(cfg['solver'][k]) or cfg['solver'][k] <= 0 for k in flexible):
        raise ValueError('positive finite numerical budgets required')
    resource = cfg['resources']
    if (not 1 <= resource['max_workers'] <= 52 or resource['numerical_threads'] != 1
            or resource['in_flight_per_worker'] != 2 or not 1 <= resource['fit_chunk_size'] <= 128):
        raise ValueError('bounded workers, chunks and single numerical thread required')
    return cfg


def parameters(cfg):
    fixed = dict(cfg['fixed']); fixed.pop('viscoelastic_s')
    tau, kappa = fixed.pop('tau'), fixed.pop('kappa')
    return BalloonParameters(fixed=BalloonFixedParameters(**fixed), free=BalloonFreeParameters(tau=tau, kappa=kappa))


def check(cfg):
    parameters(cfg).validate()
    return dict(schema=cfg['schema'], experiment_id=cfg['experiment_id'], status='software_contract_valid',
        stages=STAGES, datasets=cfg['data']['datasets'], measured_arrays_read=0,
        protected_data='forbidden', inventory='all_eligible_public_v5_none_records_all_paired_Hb_channels',
        fixed_parameters=cfg['fixed'], primary=cfg['endpoints']['primary'])


def bounded_work(worker, payloads, workers, *, in_flight_per_worker=2, progress=None):
    """Bound submitted work as well as worker count; payloads may be a generator."""
    if workers < 1: raise ValueError('positive worker count required')
    iterator = iter(payloads); results = []
    if workers == 1:
        for payload in iterator:
            result = worker(payload); results.append(result)
            if progress: progress(len(results), result)
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        active = {}
        def submit():
            while len(active) < workers*in_flight_per_worker:
                try: payload = next(iterator)
                except StopIteration: break
                active[pool.submit(worker, payload)] = None
        submit()
        while active:
            ready, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in ready:
                active.pop(future)
                result = future.result(); results.append(result)
                if progress: progress(len(results), result)
            submit()
    return results


def _progress(out, stage, total, started):
    def update(count, result):
        elapsed = time.monotonic()-started
        write_json(Path(out)/'progress.json', dict(stage=stage, completed=count, planned=total,
            elapsed_seconds=elapsed, rate_per_second=count/max(elapsed, 1e-9),
            estimated_remaining_seconds=(total-count)*elapsed/max(count, 1),
            updated_utc=datetime.now(timezone.utc).isoformat()))
    return update


@contextmanager
def run_lock(out):
    Path(out).mkdir(parents=True, exist_ok=True)
    with (Path(out)/'.controller.lock').open('a') as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError('another Tak mismatch controller owns this run') from exc
        yield


def freeze_config(cfg, out):
    path = Path(out)/'resolved_config.json'
    if path.exists() and read_json(path) != serial(cfg):
        raise ValueError('resolved Tak mismatch configuration is immutable')
    if not path.exists(): write_json(path, cfg)


def plan_stage(cfg, out, project_root):
    from src.data.tak_mismatch import inventory_tak_mismatch
    plan = inventory_tak_mismatch(cfg, Path(project_root))
    path = Path(out)/'plan.json'
    if path.exists() and read_json(path) != serial(plan):
        raise ValueError('the complete admitted inventory changed; use a new versioned run')
    write_json(path, plan)
    return plan


def prepared_paths(out, record_id):
    stem = Path(out)/'prepared'/token(record_id)
    return stem.with_suffix('.json'), stem.with_suffix('.npz'), stem.with_suffix('.stats.npz')


def _pair_sets(pair):
    value = pair.get('channel_sets', {})
    value = value.get('sets', value)
    return {name: value.get(name, value.get('fixed_random6') if name == 'random6' else None)
            for name in CANDIDATES}


def _statistics(arrays, record):
    """Sufficient statistics avoid concatenating the complete public inventory."""
    hb = np.asarray(arrays['hb'], float)
    eeg = np.asarray(arrays.get('eeg_logpower', []), float)
    p = len(record['pairs']); statistics = dict(
        hb_count=np.zeros(p), hb_sum=np.zeros((p, 2)), hb_square=np.zeros((p, 2)),
        template_sum=np.zeros((p, 300, 2)), template_count=np.zeros((p, 300, 2)),
        eeg_count=np.zeros((p, 4)), eeg_sum=np.zeros((p, 4, 30)),
        eeg_square=np.zeros((p, 4, 30, 30)))
    if hb.ndim != 4 or hb.shape[1:] != (300, p, 2):
        raise ValueError('prepared Hb must be [window,300,pair,2]')
    if eeg.ndim != 4 or eeg.shape[:2] != hb.shape[:2] or eeg.shape[-1] != 5:
        raise ValueError('prepared EEG must be [matching_window,300,native_channel,5]')
    for j, pair in enumerate(record['pairs']):
        values = hb[:, :, j]
        usable = np.isfinite(values).all(axis=-1); usable[:, -1] = False
        flat = values[usable]
        statistics['hb_count'][j] = len(flat)
        if len(flat):
            statistics['hb_sum'][j] = flat.sum(0); statistics['hb_square'][j] = (flat*flat).sum(0)
        finite = np.isfinite(values)
        statistics['template_sum'][j] = np.where(finite, values, 0.).sum(0)
        statistics['template_count'][j] = finite.sum(0)
        for k, candidate in enumerate(CANDIDATES):
            channels = _pair_sets(pair)[candidate]
            if channels is None or eeg.ndim != 4: continue
            try: x = eeg[:, :, np.asarray(channels, int), :].reshape(-1, 30)
            except (IndexError, ValueError): continue
            good = np.isfinite(x).all(1)
            # The last forcing bin does not enter either fitting or scaling.
            good[np.arange(len(hb))*300+299] = False
            x = x[good]
            statistics['eeg_count'][j, k] = len(x)
            if len(x):
                statistics['eeg_sum'][j, k] = x.sum(0)
                statistics['eeg_square'][j, k] = x.T@x
    return statistics


@lru_cache(maxsize=1)
def _metadata_index(project_root, cache_root):
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    return CleanPhysiologyCacheIndex(Path(project_root)/cache_root)


def _prepare_record_job(payload):
    cfg, root, out, record = payload
    metadata_path, array_path, stats_path = prepared_paths(out, record['id'])
    if metadata_path.exists(): return read_json(metadata_path)
    started = time.monotonic()
    result = dict(record_id=record['id'], status='failed_preparation', success=False)
    try:
        from src.data.tak_mismatch import prepare_tak_mismatch_record
        arrays = prepare_tak_mismatch_record(Path(root), record,
            cache_root=cfg['data']['cache_root'], geometry_root=cfg['data']['geometry_root'],
            refed_geometry_root=cfg['data']['refed_geometry_root'], seed=cfg['seed'],
            index=_metadata_index(str(root), cfg['data']['cache_root']))
        metadata = arrays.pop('metadata', {})
        numeric = {key: value for key, value in arrays.items()
                   if isinstance(value, (np.ndarray, list, tuple, float, int, str))}
        write_npz(array_path, **numeric)
        write_npz(stats_path, **_statistics(numeric, record))
        result.update(status='completed', success=True, array_path=str(array_path),
            statistics_path=str(stats_path), metadata=metadata,
            windows=len(record['windows']), pairs=len(record['pairs']))
        if 'continuous_hb' in numeric:
            from src.inference.tak_mismatch import weak_balance_residual
            continuous = np.asarray(numeric['continuous_hb'], float)
            valid = np.array(numeric.get('continuous_hb_valid_mask', np.isfinite(continuous)), bool, copy=True)
            clock = np.asarray(numeric.get('continuous_hb_time_s', np.arange(len(continuous))*.1))
            gaps = np.asarray(numeric.get('continuous_hb_gap_valid_mask', np.ones(len(continuous), bool)), bool)
            if gaps.ndim == 1: gaps = gaps[:, None, None]
            valid &= np.broadcast_to(gaps, valid.shape)
            segments = record.get('continuous_hb_segments', metadata.get('continuous_hb_segments', []))
            # One invalid sample at an internal session boundary prevents every
            # weak kernel spanning that discontinuity from entering the score.
            for segment in segments:
                for boundary in (segment.get('start_s'), segment.get('end_s')):
                    if boundary is not None and len(clock) and clock[0]+.05 < boundary < clock[-1]-.05:
                        index = int(np.argmin(abs(clock-float(boundary))))
                        valid[index] = False
            weak_arrays = {}; weak_summary = {}
            for j, pair in enumerate(record['pairs']):
                pair_valid = valid[:, j] if valid.ndim == 3 else valid
                scales = weak_balance_residual(continuous[:, j], cfg['tensor']['dt_s'], parameters(cfg),
                    valid=pair_valid,
                    **dict(support_widths_s=cfg['weak_balance']['support_widths_s'],
                         center_step_s=cfg['weak_balance']['center_step_s'],
                         excitation_floor=cfg['weak_balance']['excitation_floor'],
                         viscoelastic_s=cfg['fixed']['viscoelastic_s']))
                weak_summary[str(j)] = _store_weak(weak_arrays, j, scales)
            weak_path = Path(out)/'weak'/f'{token(record["id"])}.npz'
            weak_arrays['continuous_hb_time_s'] = clock
            write_npz(weak_path, **weak_arrays)
            result.update(weak_path=str(weak_path), weak_summary=weak_summary)
    except Exception as exc:
        result.update(status='failed_preparation', success=False, error=str(exc), traceback=traceback.format_exc())
    result['elapsed_seconds'] = time.monotonic()-started
    import resource
    result['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(metadata_path, result)
    return result


def _store_weak(arrays, pair_index, scales):
    """Keep library fields verbatim under deterministic pair/width prefixes."""
    summary = {}
    scale_values = scales.get('scales', scales) if isinstance(scales, dict) else scales
    for width, values in scale_values.items():
        if not isinstance(values, dict): continue
        try: label = f'{float(width):g}s'
        except (TypeError, ValueError):
            label = str(width).replace('scale_', '').replace('s', '')+'s'
        prefix = f'pair_{pair_index}_{label}_'
        for key, value in values.items():
            if isinstance(value, np.ndarray): arrays[prefix+key] = value
        summary[label] = {key: value for key, value in values.items()
                          if not isinstance(value, np.ndarray) and key not in ('centers', 'status_by_center')}
    return serial(summary)


def interleaved_records(cfg, records):
    groups = {dataset: sorted((r for r in records if r['dataset'] == dataset),
        key=lambda r: (r.get('status') != 'available', r['id'])) for dataset in cfg['data']['datasets']}
    index = 0
    while any(index < len(group) for group in groups.values()):
        for dataset in cfg['data']['datasets']:
            if index < len(groups[dataset]): yield groups[dataset][index]
        index += 1


def prepare_stage(cfg, out, root, plan, workers, *, max_records=None):
    records = [r for r in plan['records'] if (r.get('windows') or r.get('hb_only_windows')) and r.get('pairs')]
    if max_records is not None: records = list(interleaved_records(cfg, records))[:max_records]
    started = time.monotonic()
    results = bounded_work(_prepare_record_job,
        ((cfg, str(root), str(out), r) for r in records), workers,
        progress=_progress(out, 'prepare', len(records), started))
    result = dict(status='completed' if max_records is None else 'pilot_completed', planned=len(records), terminal=len(results),
        successful=sum(r.get('success', False) for r in results),
        statuses=dict(Counter(r['status'] for r in results)), elapsed_seconds=time.monotonic()-started,
        record_ids=[r['id'] for r in records],
        records=[{key: r.get(key) for key in ('record_id', 'status', 'success', 'elapsed_seconds',
            'max_rss_kib', 'windows', 'pairs', 'array_path')} for r in results])
    write_json(Path(out)/('prepare_summary.json' if max_records is None else 'prepare_pilot_summary.json'), result)
    return result


def inner_subject_split(subjects, *, seed, fraction=.2):
    ordered = sorted(set(subjects), key=lambda s: token(f'{seed}|inner|{s}'))
    if len(ordered) < 4: return ordered, []
    count = max(2, int(np.ceil(len(ordered)*fraction)))
    count = min(count, len(ordered)-2)
    return ordered[count:], ordered[:count]


def _combine_statistics(chunks):
    result = None
    for chunk in chunks:
        if result is None: result = {key: np.array(value, float, copy=True) for key, value in chunk.items()}
        else:
            for key, value in chunk.items(): result[key] += value
    return result


def fit_coordinate_statistics(statistics, cfg):
    if statistics is None or statistics['hb_count'] <= 1:
        return dict(status='unavailable_training_Hb')
    count = float(statistics['hb_count'])
    variance = statistics['hb_square']/count-(statistics['hb_sum']/count)**2
    if not np.isfinite(variance).all() or np.min(variance) <= 1e-24:
        return dict(status='unavailable_degenerate_Hb_scale')
    factor = cfg['coordinate']['hb_training_sd_target']/np.sqrt(np.mean(variance))
    sd_hb = np.sqrt(variance)*factor; pcs = {}; eigenvalues = {}
    for k, candidate in enumerate(CANDIDATES):
        n = float(statistics['eeg_count'][k])
        if n <= 1: continue
        mean = statistics['eeg_sum'][k]/n
        covariance = statistics['eeg_square'][k]/n-np.outer(mean, mean)
        values, vectors = np.linalg.eigh((covariance+covariance.T)/2)
        if values[-1] <= 1e-16: continue
        pc = vectors[:, -1]
        if pc[np.argmax(np.abs(pc))] < 0: pc = -pc
        pcs[candidate], eigenvalues[candidate] = pc, float(values[-1])
    baseline = eigenvalues.get('nearest6')
    eeg_factor = None if baseline is None else cfg['coordinate']['eeg_training_sd_target']/np.sqrt(baseline)
    template = np.divide(statistics['template_sum'], statistics['template_count'],
        out=np.full((300, 2), np.nan), where=statistics['template_count'] > 0)*factor
    return dict(status='completed' if baseline is not None else 'Hb_only_no_training_EEG',
        pc=pcs, eeg_factor=eeg_factor, hb_factor=float(factor),
        sd=np.r_[cfg['coordinate']['eeg_training_sd_target'], sd_hb],
        template=template, training_samples=int(count),
        candidate_eigenvalues=eigenvalues, common_Hb_scale=True,
        candidate_output_renormalized=False)


def _group_key(record, pair):
    return f'{record["dataset"]}|{pair["id"]}'


def coordinate_stage(cfg, out, plan):
    path = Path(out)/'coordinates.json'
    if path.exists(): return read_json(path)
    missing = [r['id'] for r in plan['records'] if r.get('windows') and r.get('pairs')
               and not prepared_paths(out, r['id'])[0].exists()]
    if missing: raise ValueError(f'complete preparation must be terminal before freezing training coordinates: {len(missing)} records missing')
    statistics = defaultdict(lambda: defaultdict(list))
    for record in plan['records']:
        meta, _, stats_path = prepared_paths(out, record['id'])
        if not meta.exists() or read_json(meta)['status'] != 'completed': continue
        with np.load(stats_path, allow_pickle=False) as archive:
            for j, pair in enumerate(record['pairs']):
                statistics[_group_key(record, pair)][record['subject']].append(
                    {key: archive[key][j].copy() for key in archive.files})
    subject_totals = {key: {subject: _combine_statistics(chunks) for subject, chunks in groups.items()}
                      for key, groups in statistics.items()}
    models = {}
    for key, subject_stats in sorted(subject_totals.items()):
        dataset = key.split('|', 1)[0]; models[key] = {}
        folds = plan['subject_folds'][dataset]
        for fold in range(cfg['data']['folds']):
            training = sorted(s for s in subject_stats if folds[s] != fold)
            inner_train, selection = inner_subject_split(training, seed=cfg['seed'],
                fraction=cfg['coordinate']['inner_selection_fraction'])
            outer = fit_coordinate_statistics(_combine_statistics(subject_stats[s] for s in training), cfg)
            inner = fit_coordinate_statistics(_combine_statistics(subject_stats[s] for s in inner_train), cfg)
            models[key][str(fold)] = dict(coordinate=outer, inner_coordinate=inner,
                training_subjects=training, inner_training_subjects=inner_train,
                selection_subjects=selection, selected_candidate=None,
                selection_status='pending' if selection else 'unavailable_insufficient_training_subjects')
    result = dict(schema='tak_mismatch_coordinates_v1', status='fitted_training_only', models=models,
        partition='outer_subject_holdout_then_inner_subject_selection',
        candidate_scaling='all_candidate_PC1s_use_nearest6_positive_factor',
        scoring='frozen_nearest6_training_SD_common_Hb_factor')
    write_json(path, result)
    return serial(result)


@lru_cache(maxsize=2)
def _load_prepared(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for value in arrays.values(): value.flags.writeable = False
    return arrays


@lru_cache(maxsize=64)
def reference_operator(steps=300, reference_steps=50, hb_phase_s=0.):
    reference = np.eye(steps); reference[:, :reference_steps] -= 1/reference_steps
    if not np.isfinite(hb_phase_s) or not 0 <= hb_phase_s < .1+1e-9:
        raise ValueError('Hb phase must be inside its actual 100ms bin')
    weight = hb_phase_s/.1
    interpolation = np.eye(steps)*(1-weight)
    interpolation[np.arange(steps-1), np.arange(1, steps)] += weight
    interpolation[-1, -1] = 1.
    hb = reference@interpolation
    operator = np.zeros((steps, 3, steps, 3))
    operator[:, 0, :, 0] = reference
    operator[:, 1, :, 1] = hb; operator[:, 2, :, 2] = hb
    return operator.reshape(3*steps, 3*steps)


def window_operator(arrays, index):
    phase = float(np.asarray(arrays.get('hb_bin_phase_s', np.zeros(len(arrays['hb']))))[index])
    if not np.isfinite(phase): phase = 0.
    return reference_operator(hb_phase_s=round(phase, 12))


def proxy_for_window(arrays, window_index, pair, coordinate, candidate='nearest6'):
    channels = _pair_sets(pair).get(candidate)
    pc = coordinate.get('pc', {}).get(candidate)
    factor = coordinate.get('eeg_factor')
    if channels is None or pc is None or factor is None: return None
    try:
        features = np.asarray(arrays['eeg_logpower'][window_index, :, np.asarray(channels, int), :], float)
        # A single advanced index in NumPy can move the channel axis to front.
        if features.shape == (6, 300, 5): features = features.transpose(1, 0, 2)
        features = features.reshape(300, 30)
    except (KeyError, IndexError, ValueError): return None
    if not np.isfinite(features).all(): return None
    return features@np.asarray(pc)*factor


def hb_nmse(target, prediction, sd, mask=None):
    values, estimate = np.asarray(target, float), np.asarray(prediction, float)
    valid = np.isfinite(values).all(1) & np.isfinite(estimate).all(1)
    valid[:50] = False; valid[299:] = False
    if mask is not None: valid &= np.asarray(mask, bool)
    if not valid.any(): return None
    return float(np.mean(((values[valid]-estimate[valid])/np.asarray(sd))**2))


@lru_cache(maxsize=16)
def _screen_map(config_json, hb_phase_s=0.):
    cfg = json.loads(config_json)
    from src.inference.low_capacity_eeg_proxy import build_tak_eeg_conditional_map
    s = cfg['solver']
    return build_tak_eeg_conditional_map(parameters(cfg), 300, .1,
        mean_operator=reference_operator(hb_phase_s=hb_phase_s),
        sd=np.array([cfg['coordinate']['eeg_training_sd_target'], 1., 1.]),
        penalty=s['penalty'], initial_penalty=s['initial_penalty'],
        flow_prior_weight=s['flow_prior_weight'], flow_prior_log_sd=s['flow_prior_log_sd'],
        data_weight=s['data_weight'], substeps=s['substeps'], viscoelastic_s=cfg['fixed']['viscoelastic_s'],
        initial_state_mode=cfg['model']['initial_state_mode'])


def _screen_record_job(payload):
    cfg, out, record, models = payload
    if isinstance(models, str): models = _read_shared_json(models)['models']
    path = Path(out)/'selection'/f'{token(record["id"])}.json'
    if path.exists(): return read_json(path)
    results = []
    try:
        meta, arrays_path, _ = prepared_paths(out, record['id'])
        if not meta.exists() or read_json(meta)['status'] != 'completed':
            write_json(path, results); return results
        arrays = _load_prepared(str(arrays_path))
        for j, pair in enumerate(record['pairs']):
            key = _group_key(record, pair)
            for fold, model in models.get(key, {}).items():
                if record['subject'] not in model['selection_subjects']: continue
                coord = model['inner_coordinate']
                if not coord.get('hb_factor'): continue
                target = np.asarray(arrays['hb'])[:, :, j]*coord['hb_factor']
                for candidate in CANDIDATES:
                    scores = []; failed = 0; window_scores = {}
                    for i, window in enumerate(record['windows']):
                        proxy = proxy_for_window(arrays, i, pair, coord, candidate)
                        score = None
                        if proxy is not None:
                            phase = float(np.asarray(arrays.get('hb_bin_phase_s', np.zeros(len(target))))[i])
                            mapping = _screen_map(json.dumps(cfg, sort_keys=True), round(phase, 12))
                            prediction = mapping.predict(proxy)['prediction'][:, 1:]
                            score = hb_nmse(target[i], prediction, np.asarray(coord['sd'])[1:])
                        if score is None: failed += 1
                        else: scores.append(score)
                        window_scores[window['id']] = score
                    results.append(dict(key=key, fold=fold, subject=record['subject'], record=record['id'],
                        candidate=candidate, planned=len(record['windows']), successful=len(scores), failed=failed,
                        nmse=float(np.mean(scores)) if scores else None, window_scores=window_scores))
    except Exception as exc:
        results = [dict(record=record['id'], status='failed_exception', error=str(exc), traceback=traceback.format_exc())]
    write_json(path, results); return results


def select_spatial_candidate(scores):
    """Use equal subjects, full failure counts and declared candidate tie order."""
    paired = defaultdict(lambda: defaultdict(list)); paired_counts = Counter()
    by_record = defaultdict(dict)
    for row in scores:
        by_record[(row['subject'], row.get('record', row['subject']))][row['candidate']] = row
    for (subject, record), candidates in by_record.items():
        if set(candidates) != set(CANDIDATES): continue
        common = set.intersection(*(set(row.get('window_scores', {})) for row in candidates.values()))
        common = sorted(window for window in common if all(
            row['window_scores'][window] is not None and np.isfinite(row['window_scores'][window])
            for row in candidates.values()))
        if not common: continue
        for candidate, row in candidates.items():
            paired[candidate][subject].append(float(np.mean([row['window_scores'][w] for w in common])))
            paired_counts[candidate] += len(common)
    assessment = {}
    for candidate in CANDIDATES:
        rows = [r for r in scores if r.get('candidate') == candidate]
        by_subject = paired[candidate]
        planned = sum(r['planned'] for r in rows); failed = sum(r['failed'] for r in rows)
        assessment[candidate] = dict(planned=planned, failed=failed,
            subjects=len(by_subject), failure_rate=failed/planned if planned else 1.,
            paired_windows=paired_counts[candidate],
            nmse=float(np.mean([np.mean(x) for x in by_subject.values()])) if by_subject else None)
    baseline = assessment['nearest6']
    eligible = [c for c in CANDIDATES if assessment[c]['subjects'] >= 2 and assessment[c]['nmse'] is not None
                and assessment[c]['planned'] == baseline['planned']
                and assessment[c]['failure_rate'] <= baseline['failure_rate']]
    winner = min(eligible, key=lambda c: (assessment[c]['nmse'], CANDIDATES.index(c))) if eligible else None
    return dict(selected_candidate=winner,
        selection_status='completed' if winner else 'unavailable_selection_evidence', assessment=assessment)


def spatial_selection_stage(cfg, out, plan, coordinates, workers):
    path = Path(out)/'spatial_selection.json'
    if path.exists(): return read_json(path)
    results = bounded_work(_screen_record_job,
        ((cfg, str(out), r, str(Path(out)/'coordinates.json')) for r in plan['records'] if r.get('windows') and r.get('pairs')),
        workers)
    scores = [r for group in results for r in group]; by_key = defaultdict(list)
    for row in scores:
        if 'key' in row: by_key[(row['key'], row['fold'])].append(row)
    for key, folds in coordinates['models'].items():
        for fold, model in folds.items():
            if model['selection_subjects']:
                model.update(select_spatial_candidate(by_key[(key, fold)]))
    result = dict(schema='tak_mismatch_spatial_selection_v1', status='completed',
        method='inner_subject_holdout_fixed_Tak_rest_tangent_screen_nonlinear_holdout_evaluation',
        models=coordinates['models'], failures=[r for r in scores if r.get('status') == 'failed_exception'])
    write_json(path, result); return serial(result)


def wrong_eeg_partners(windows):
    result = {}
    for window in windows:
        if (window.get('eeg_filter_start_s') is None or window.get('eeg_filter_end_s') is None
                or window.get('wrong_pairing_eligible') is False):
            result[window['id']] = None; continue
        candidates = [other for other in windows if other['id'] != window['id']
            and other.get('eeg_filter_start_s') is not None and other.get('eeg_filter_end_s') is not None
            and other.get('wrong_pairing_eligible') is not False
            and other.get('condition') == window.get('condition')
            and other.get('task') == window.get('task')
            and (other['eeg_filter_end_s'] <= window['eeg_filter_start_s']
                 or window['eeg_filter_end_s'] <= other['eeg_filter_start_s'])]
        result[window['id']] = min(candidates, key=lambda r: r['id'])['id'] if candidates else None
    return result


def task_manifest(plan):
    for record in plan['records']:
        partners = wrong_eeg_partners(record.get('windows', []))
        for i, window in enumerate(record.get('windows', [])):
            for j, pair in enumerate(record.get('pairs', [])):
                identity = f'{record["id"]}|{window["id"]}|{pair["id"]}|fixed'
                yield dict(id=token(identity), record_id=record['id'], dataset=record['dataset'],
                    subject=record['subject'], record=record.get('record', record['id']),
                    session=window.get('native_support_id') or window.get('session') or window.get('session_id') or
                        f'{record.get("record", record["id"])}|segment={window.get("segment_index", "native")}',
                    window=window['id'], pair=pair['id'], fold=record['fold'],
                    window_index=i, pair_index=j, parameter_mode='fixed',
                    start_s=window['start_s'] if window.get('start_s') is not None else window['hb_start_s'],
                    eeg_start_s=window.get('eeg_start_s'), hb_start_s=window['hb_start_s'],
                    end_s=(window['start_s'] if window.get('start_s') is not None else window['hb_start_s'])+30.,
                    actual_context_s=window['actual_context_s'], wrong_window=partners[window['id']],
                    duration_s=30., independent_unit='subject')


def _solver_kwargs(cfg):
    s = cfg['solver']
    return {key: s[key] for key in ('penalty', 'initial_penalty', 'flow_prior_weight', 'flow_prior_log_sd',
        'max_evaluations_per_trial', 'max_iterations', 'gradient_tolerance', 'substeps', 'numerical_backend', 'data_weight')}


def _validate_fit(fit, cfg, sd, operator=None):
    if 'prediction' not in fit:
        return dict(success=False, status='failed_numerical', solver_status=fit.get('status'))
    from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
    prediction = np.asarray(fit['prediction'])[0]
    fine = nonlinear_driver_forward(np.asarray(fit['driver'])[0], np.asarray(fit['initial_state'])[0],
        parameters(cfg), .1, substeps=cfg['solver']['validation_substeps'], derivative=False,
        numerical_backend=cfg['solver']['numerical_backend'], hemodynamics='tak',
        viscoelastic_s=cfg['fixed']['viscoelastic_s'])
    operator = reference_operator() if operator is None else operator
    fine_prediction = (operator@fine['canonical_prediction'].ravel()).reshape(300, 3)
    difference = float(np.max(np.abs(fine_prediction-prediction)/np.asarray(sd)))
    success = bool(fit.get('converged') and difference <= cfg['solver']['maximum_integration_difference_training_sd'])
    return dict(success=success, status='completed' if success else 'failed_integration' if
        difference > cfg['solver']['maximum_integration_difference_training_sd'] else 'failed_numerical',
        integration_difference_training_sd=difference, solver_status=fit.get('status'),
        converged=bool(fit.get('converged')), objective=fit.get('objective'),
        objective_components=fit.get('objective_components'), evaluations=fit.get('evaluations'))


def _fit_views(cfg, target, sd, views, operator=None):
    from src.inference.tak_mismatch import fit_tak_views
    return fit_tak_views(np.asarray(target)[None], parameters(cfg), .1, np.asarray(sd),
        mean_operator=reference_operator() if operator is None else operator,
        parameter_names=(), parameter_bounds=(),
        views=tuple(views), viscoelastic_s=cfg['fixed']['viscoelastic_s'],
        initial_state_mode=cfg['model']['initial_state_mode'], **_solver_kwargs(cfg))


def independent_eeg_input(proxy):
    """Construct the complete prediction input without accepting any target Hb."""
    values = np.full((300, 3), np.nan)
    values[:, 0] = proxy
    values[-1] = np.nan
    return values


def _task_job(cfg, out, task, record, selection):
    path = Path(out)/'tasks'/f'{task["id"]}.json'
    if path.exists(): return read_json(path)
    started = time.monotonic(); result = dict(task, status='failed_exception', success=False)
    try:
        metadata_path, arrays_path, _ = prepared_paths(out, record['id'])
        if not metadata_path.exists() or read_json(metadata_path)['status'] != 'completed':
            result['status'] = 'unavailable_preparation'; write_json(path, result); return result
        metadata = read_json(metadata_path); arrays = _load_prepared(str(arrays_path))
        i, j = task['window_index'], task['pair_index']; pair = record['pairs'][j]
        model = selection['models'].get(_group_key(record, pair), {}).get(str(record['fold']))
        if model is None or not model['coordinate'].get('hb_factor'):
            result['status'] = 'unavailable_training_coordinate'; write_json(path, result); return result
        coord = model['coordinate']; sd = np.asarray(coord['sd']); target = np.full((300, 3), np.nan)
        operator = window_operator(arrays, i)
        target[:, 1:] = np.asarray(arrays['hb'])[i, :, j]*coord['hb_factor']
        if not np.isfinite(target[:299, 1:]).all():
            result['status'] = 'unavailable_complete_Hb_pair'
            write_json(path, result); return result
        proxy = proxy_for_window(arrays, i, pair, coord)
        if proxy is not None: target[:, 0] = proxy
        target[-1] = np.nan
        views = ['hb_only'] if proxy is None else ['hb_only', 'eeg_only', 'joint']
        fits = _fit_views(cfg, target, sd, views, operator)
        validations = {name: _validate_fit(fit, cfg, sd, operator) for name, fit in fits.items()}
        valid = np.isfinite(target); valid[:50] = False; valid[299:] = False
        from src.inference.tak_mismatch import score_tak_mismatch, nonlinear_drive_gap
        nan_prediction = np.full((300, 3), np.nan)
        predictions = {name: np.asarray(fit['prediction'])[0] if 'prediction' in fit else nan_prediction.copy()
                       for name, fit in fits.items()}
        metrics = score_tak_mismatch(target, predictions.get('hb_only', nan_prediction),
            predictions.get('joint', nan_prediction), sd, visible=valid)
        # Only scalar fields belong in task records. Signed arrays remain NPZ.
        result.update({key: value for key, value in metrics.items()
                       if isinstance(value, (float, int, str, bool, np.generic))})
        # The library status describes component support; controller status
        # separately describes terminal optimization/integration outcomes.
        result['score_status'] = metrics.get('status')
        for name in ('D1', 'D_EH'):
            if not validations.get('hb_only', {}).get('success'): result[name] = None
        for name in ('D4', 'D_J_E', 'D_J_O', 'D_J_R'):
            if not validations.get('joint', {}).get('success'): result[name] = None
        saved = dict(time_s=np.arange(300)*.1, target=target, sd=sd,
            hb_factor=np.asarray(coord['hb_factor']), score_mask=valid,
            absolute_start_s=np.asarray(task['start_s']), hb_absolute_start_s=np.asarray(task['hb_start_s']))
        for name in ('hb_only', 'eeg_only', 'joint'):
            saved[name+'_prediction'] = predictions.get(name, nan_prediction)
            fit = fits.get(name, {})
            saved[{'hb_only': 'driver_h', 'eeg_only': 'driver_e', 'joint': 'driver_j'}[name]] = (
                np.asarray(fit['driver'])[0] if 'driver' in fit else np.full(300, np.nan))
            if 'states' in fit: saved[name+'_states'] = np.asarray(fit['states'])[0]
            if 'initial_state' in fit: saved[name+'_initial_state'] = np.asarray(fit['initial_state'])[0]
        if validations.get('hb_only', {}).get('success') and validations.get('eeg_only', {}).get('success'):
            gap = nonlinear_drive_gap(saved['driver_e'], saved['driver_h'], parameters(cfg), .1, sd,
                mean_operator=operator, valid=valid[:, 1:],
                substeps=cfg['solver']['substeps'], viscoelastic_s=cfg['fixed']['viscoelastic_s'])
            result['D_drive_H'] = gap.get('D_drive_H')
            result['drive_gap_status'] = gap.get('status')
            result['drive_gap_dc_gauge'] = gap.get('driver_dc_gauge')
            for key, value in gap.items():
                if isinstance(value, np.ndarray): saved['drive_gap_'+key] = value
            from src.inference.tak_mismatch import observable_drive_scenarios
            observable = observable_drive_scenarios(saved['driver_e'], saved['driver_h'], saved['driver_h'],
                np.asarray(fits['hb_only']['initial_state'])[0], parameters(cfg), .1, sd,
                driver_train_sd=float(sd[0]), mean_operator=operator, valid=valid[:, 1:],
                noise_scenarios=cfg['observability']['scenarios'],
                draws=cfg['observability']['draws'], quantile=cfg['observability']['quantile'],
                seed=cfg['seed'], substeps=cfg['solver']['substeps'],
                viscoelastic_s=cfg['fixed']['viscoelastic_s'])
            result.update(D_obs=observable.get('D_obs'), D_obs_status=observable.get('status'),
                D_obs_allrank=observable.get('D_obs_allrank'), observable_numerical_rank=observable.get('numerical_rank'),
                observable_retained_rank=observable.get('retained_rank'),
                observable_primary_noise_scenario=observable.get('primary_noise_scenario'))
            result['observable_scenarios'] = {
                name: {key: value for key, value in item.items() if not isinstance(value, np.ndarray)}
                for name, item in observable.get('scenarios', {}).items()}
            for name, item in observable.get('scenarios', {}).items():
                if name != observable.get('primary_noise_scenario'): result['D_obs_'+name] = item.get('D_obs')
            primary = observable.get('scenarios', {}).get(observable.get('primary_noise_scenario'), {})
            for key in ('standardized_hb_difference', 'projected_noise_threshold', 'singular_value_threshold', 'retained_modes'):
                if key in primary: saved['observable_'+key] = np.asarray(primary[key])
            for name, scenario in observable.get('scenarios', {}).items():
                for key in ('projected_noise_threshold', 'singular_value_threshold', 'retained_modes'):
                    if key in scenario: saved['observable_'+name+'_'+key] = np.asarray(scenario[key])
            for key, value in observable.get('mode_curves', {}).items():
                saved['observable_'+key] = np.asarray(value)
        else:
            result['D_obs'] = None; result['D_obs_status'] = 'unavailable_successful_independent_fits'
        result['prediction_NMSE'] = hb_nmse(target[:, 1:], saved['eeg_only_prediction'][:, 1:], sd[1:]) if validations.get('eeg_only', {}).get('success') else None
        selected = model.get('selected_candidate'); result['selected_candidate'] = selected
        saved['spatial_prediction'] = nan_prediction.copy()
        spatial_valid = None
        if selected == 'nearest6':
            saved['spatial_prediction'] = saved['eeg_only_prediction']; spatial_valid = validations.get('eeg_only')
        elif selected:
            selected_proxy = proxy_for_window(arrays, i, pair, coord, selected)
            if selected_proxy is not None:
                selected_fit = _fit_views(cfg, independent_eeg_input(selected_proxy), sd, ['eeg_only'], operator)['eeg_only']
                spatial_valid = _validate_fit(selected_fit, cfg, sd, operator)
                if 'prediction' in selected_fit: saved['spatial_prediction'] = np.asarray(selected_fit['prediction'])[0]
        result['selected_prediction_NMSE'] = hb_nmse(target[:, 1:], saved['spatial_prediction'][:, 1:], sd[1:]) if spatial_valid and spatial_valid.get('success') else None
        if result['prediction_NMSE'] is not None and result['selected_prediction_NMSE'] is not None:
            result['Delta_spatial'] = result['prediction_NMSE']-result['selected_prediction_NMSE']
        saved['zero_prediction'] = np.zeros((300, 3))
        saved['template_prediction'] = np.column_stack((np.full(300, np.nan), np.asarray(coord['template'])))
        result['zero_NMSE'] = hb_nmse(target[:, 1:], saved['zero_prediction'][:, 1:], sd[1:])
        result['template_NMSE'] = hb_nmse(target[:, 1:], saved['template_prediction'][:, 1:], sd[1:])
        donor_index = next((k for k, w in enumerate(record['windows']) if w['id'] == task['wrong_window']), None)
        wrong_statuses = {}
        for name, candidate in (('wrong', 'nearest6'), ('selected_wrong', selected)):
            saved[name+'_prediction'] = nan_prediction.copy(); result[name+'_NMSE'] = None
            if donor_index is None or candidate is None:
                wrong_statuses[name] = 'unavailable_nonoverlap_donor' if donor_index is None else 'unavailable_selection'
                continue
            if name == 'selected_wrong' and candidate == 'nearest6':
                saved[name+'_prediction'] = saved['wrong_prediction']; result[name+'_NMSE'] = result['wrong_NMSE']
                wrong_statuses[name] = wrong_statuses['wrong']; continue
            wrong_proxy = proxy_for_window(arrays, donor_index, pair, coord, candidate)
            if wrong_proxy is None:
                wrong_statuses[name] = 'unavailable_donor_EEG'; continue
            wrong_fit = _fit_views(cfg, independent_eeg_input(wrong_proxy), sd, ['eeg_only'], operator)['eeg_only']
            validation = _validate_fit(wrong_fit, cfg, sd, operator); wrong_statuses[name] = validation
            if 'prediction' in wrong_fit: saved[name+'_prediction'] = np.asarray(wrong_fit['prediction'])[0]
            if validation['success']: result[name+'_NMSE'] = hb_nmse(target[:, 1:], saved[name+'_prediction'][:, 1:], sd[1:])
        for name, left, right in (('pairing_gain', 'wrong_NMSE', 'prediction_NMSE'),
                                 ('selected_pairing_gain', 'selected_wrong_NMSE', 'selected_prediction_NMSE')):
            if result[left] is not None and result[right] is not None: result[name] = result[left]-result[right]
        for width, item in metadata.get('weak_summary', {}).get(str(j), {}).items():
            result['D_bal_'+width] = item.get('D_bal')
        result.update(status='completed' if all(v['success'] for v in validations.values()) else 'partial_or_failed_fit',
            success=bool(validations.get('hb_only', {}).get('success')),
            view_statuses=validations, spatial_status=spatial_valid, wrong_statuses=wrong_statuses,
            hb_only_success=bool(validations.get('hb_only', {}).get('success')),
            eeg_only_success=bool(validations.get('eeg_only', {}).get('success')),
            joint_success=bool(validations.get('joint', {}).get('success')),
            training_subjects=model['training_subjects'], selection_subjects=model['selection_subjects'],
            weak_path=metadata.get('weak_path'), weak_pair_index=j,
            axis_3_selection_status=model['selection_status'],
            units='registered_cached_Hb_native_units_then_training_common_factor',
            eeg_input_status='completed' if proxy is not None else 'unavailable_native_EEG_or_geometry',
            sigma=sd, fixed_parameters=cfg['fixed'])
        if validations.get('hb_only', {}).get('success'):
            residual = (target[:, 1:]-saved['hb_only_prediction'][:, 1:])/coord['hb_factor']
            for k, name in enumerate(('HbO_RMSE_native', 'HbR_RMSE_native')):
                common = valid[:, 1:].all(1) & np.isfinite(residual).all(1)
                result[name] = float(np.sqrt(np.mean(residual[common, k]**2))) if common.any() else None
        array_path = Path(out)/'tasks'/f'{task["id"]}.npz'
        write_npz(array_path, **saved); result['array_path'] = str(array_path)
    except Exception as exc:
        result.update(status='failed_exception', error=str(exc), traceback=traceback.format_exc())
    result['elapsed_seconds'] = time.monotonic()-started
    write_json(path, result); return result


@lru_cache(maxsize=4)
def _read_shared_json(path):
    return read_json(path)


def _chunk_job(payload):
    cfg, out, record, tasks = payload
    selection = _read_shared_json(str(Path(out)/'spatial_selection.json'))
    results = [_task_job(cfg, out, task, record, selection) for task in tasks]
    return dict(terminal=len(results), statuses=dict(Counter(r['status'] for r in results)))


def pilot_task_manifest(cfg, plan, maximum):
    groups = defaultdict(list)
    for record in interleaved_records(cfg, [r for r in plan['records'] if r.get('windows') and r.get('pairs')]):
        rows = [task for task in task_manifest(dict(records=[record]))
                if record['windows'][task['window_index']].get('status', 'available') == 'available'
                and record['pairs'][task['pair_index']].get('status', 'available') == 'available']
        rows.sort(key=lambda r: (max(r['window_index'], r['pair_index']), r['window_index'], r['pair_index']))
        groups[record['dataset']].extend(rows[:maximum])
        if all(len(groups[d]) >= maximum for d in cfg['data']['datasets']): break
    selected = []; offset = 0
    while len(selected) < maximum and any(offset < len(group) for group in groups.values()):
        for dataset in cfg['data']['datasets']:
            if offset < len(groups[dataset]): selected.append(groups[dataset][offset])
            if len(selected) == maximum: break
        offset += 1
    return selected


def task_chunks(cfg, out, plan, *, max_tasks=None, selected_tasks=None):
    records = {r['id']: r for r in plan['records']}; active = []; record_id = None; count = 0
    for task in task_manifest(plan) if selected_tasks is None else selected_tasks:
        if max_tasks is not None and count >= max_tasks: break
        count += 1
        if active and (task['record_id'] != record_id or len(active) >= cfg['resources']['fit_chunk_size']):
            yield cfg, str(out), records[record_id], active
            active = []
        record_id = task['record_id']; active.append(task)
    if active: yield cfg, str(out), records[record_id], active


def run_stage(cfg, out, plan, workers, *, max_tasks=None):
    coordinates = coordinate_stage(cfg, out, plan)
    spatial_selection_stage(cfg, out, plan, coordinates, workers)
    selected_tasks = None if max_tasks is None else pilot_task_manifest(cfg, plan, max_tasks)
    count = sum(1 for _ in task_manifest(plan)) if selected_tasks is None else len(selected_tasks)
    if selected_tasks is not None:
        write_json(Path(out)/'pilot_manifest.json', dict(status='planned', tasks=selected_tasks,
            selection='dataset_round_robin_metadata_admitted_windows_pairs_no_result_selection', requested=max_tasks))
    started = time.monotonic(); completed = 0
    def update(_, result):
        nonlocal completed
        completed += result['terminal']
        _progress(out, 'fixed_parameter_fit', count, started)(completed, result)
    # Chunk results are already persisted. Retain only the small timing/status
    # summary in memory instead of the complete per-window fit dictionaries.
    results = bounded_work(_chunk_job, task_chunks(cfg, out, plan, selected_tasks=selected_tasks), workers,
        progress=update)
    result = dict(status='fixed_parameter_stage_completed' if max_tasks is None else 'pilot_completed', planned=count,
        terminal=sum(r['terminal'] for r in results), elapsed_seconds=time.monotonic()-started,
        workers=workers, numerical_threads=1,
        status_counts=dict(sum((Counter(chunk['statuses']) for chunk in results), Counter())))
    write_json(Path(out)/('run_summary.json' if max_tasks is None else 'pilot_summary.json'), result)
    if max_tasks is None:
        companion = globals().get('current_dof_stage')
        if companion is not None:
            result['current_dof'] = companion(cfg, out, plan, workers)
            result['status'] = 'completed' if result['current_dof']['status'] == 'completed' else 'incomplete'
            write_json(Path(out)/'run_summary.json', result)
    return result


# CURRENT_DOF_OWNER: the root agent adds record/session/pair-shared companions here.

def current_dof_groups(plan):
    """Keep parameters inside one actual native continuous support and Hb pair."""
    for record in plan['records']:
        groups = defaultdict(list)
        for task in task_manifest(dict(records=[record])):
            window = record['windows'][task['window_index']]
            support = window.get('native_support_id', window.get('hb_segment_index', task['session']))
            groups[(str(support), task['pair'])].append(task)
        for (support, pair), tasks in groups.items():
            identity = f'{record["id"]}|{support}|{pair}|current_dof'
            yield record, dict(id=token(identity), record_id=record['id'], dataset=record['dataset'],
                subject=record['subject'], native_support_id=support, pair=pair, tasks=tasks)


def _current_parameters(cfg, values):
    from dataclasses import replace
    p = parameters(cfg)
    return replace(p, free=replace(p.free, **dict(zip(cfg['current_dof']['parameter_names'], values))))


def _current_warm(out, task, view):
    """A conditional fit may warm-start only from the same visible observations."""
    driver = np.zeros(300); initial = np.array([0., 1., 1., 1., 1.]); source = 'rest'
    path = Path(out)/'tasks'/f'{task["id"]}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            key = 'driver_h' if view == 'hb_only' else 'driver_j'
            if key in saved:
                r = np.asarray(saved[key], float)
                if view+'_initial_state' in saved:
                    z = np.asarray(saved[view+'_initial_state'], float)
                elif view+'_states' in saved:
                    z = np.asarray(saved[view+'_states'], float)[0, 1:]
                else:
                    z = initial
                if r.shape == (300,) and z.shape == (5,) and np.isfinite(r).all() and np.isfinite(z).all():
                    driver, initial, source = r.copy(), z.copy(), 'same_visibility_fixed_solution'
    return driver, initial, source


def _current_group_job(payload):
    cfg, out, record, group = payload
    path = Path(out)/'current_dof'/'tasks'/f'{group["id"]}.json'
    if path.exists():
        result = read_json(path)
        return dict(group_id=group['id'], status=result['status'], terminal=len(result['rows']))
    started = time.monotonic()
    rows = [dict(task, parameter_mode='current_dof', group_id=group['id'],
                 status='unavailable_preparation', success=False, fixed_task_id=task['id'])
            for task in group['tasks']]
    result = {key: value for key, value in group.items() if key != 'tasks'}
    result.update(status='failed_exception', rows=rows, views={})
    try:
        meta_path, prepared_path, _ = prepared_paths(out, record['id'])
        if not meta_path.exists() or read_json(meta_path)['status'] != 'completed':
            result['status'] = 'unavailable_preparation'
        else:
            arrays = _load_prepared(str(prepared_path))
            selection = _read_shared_json(str(Path(out)/'spatial_selection.json'))
            pair = record['pairs'][group['tasks'][0]['pair_index']]
            model = selection['models'].get(_group_key(record, pair), {}).get(str(record['fold']))
            if model is None or not model['coordinate'].get('hb_factor'):
                result['status'] = 'unavailable_training_coordinate'
                for row in rows: row['status'] = result['status']
            else:
                coord = model['coordinate']; sd = np.asarray(coord['sd'], float)
                n = len(rows); targets = np.full((n, 300, 3), np.nan)
                operators = []
                for k, task in enumerate(group['tasks']):
                    i, j = task['window_index'], task['pair_index']
                    targets[k, :, 1:] = arrays['hb'][i, :, j]*coord['hb_factor']
                    proxy = proxy_for_window(arrays, i, pair, coord)
                    if proxy is not None: targets[k, :, 0] = proxy
                    targets[k, -1] = np.nan
                    operators.append(window_operator(arrays, i))
                    rows[k]['status'] = 'unavailable_complete_Hb_pair'
                score_mask = np.isfinite(targets); score_mask[:, :50] = False; score_mask[:, 299:] = False
                saved = dict(window_ids=np.asarray([t['id'] for t in group['tasks']]),
                    time_s=np.arange(300)*.1, target=targets, sd=sd,
                    hb_factor=np.asarray(coord['hb_factor']), score_mask=score_mask)
                valid_views = {}; predictions = {}; fitted_parameters = {}
                from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_parameters, nonlinear_driver_forward
                for view in cfg['current_dof']['views']:
                    required = targets[:, :299, 1:] if view == 'hb_only' else targets[:, :299]
                    eligible = np.flatnonzero(np.isfinite(required).all(axis=(1, 2)))
                    prediction = np.full_like(targets, np.nan)
                    drivers = np.full((n, 300), np.nan); initials = np.full((n, 5), np.nan)
                    states = np.full((n, 300, 6), np.nan); validated = np.zeros(n, bool)
                    predictions[view] = prediction; valid_views[view] = validated
                    if not len(eligible):
                        result['views'][view] = dict(status='unavailable_complete_support', planned=n, fitted=0)
                        continue
                    y = targets[eligible].copy(); visible = np.isfinite(y)
                    if view == 'hb_only': visible[..., 0] = False
                    warm = [_current_warm(out, group['tasks'][k], view) for k in eligible]
                    starts = [dict(parameter_values=values, driver=np.array([w[0] for w in warm]),
                                   initial_state=np.array([w[1] for w in warm]))
                              for values in cfg['current_dof']['starts']]
                    selected_ops = [operators[k] for k in eligible]
                    operator = selected_ops[0] if all(np.array_equal(selected_ops[0], op) for op in selected_ops) else np.asarray(selected_ops)
                    settings = cfg['current_dof']
                    fit = fit_nonlinear_shared_parameters(np.where(visible, y, np.nan), parameters(cfg), .1,
                        parameter_names=settings['parameter_names'], parameter_bounds=settings['parameter_bounds'],
                        sd=sd, mean_operator=operator, visible=visible, starts=starts,
                        parameter_prior_log_mean=np.log(settings['reference']),
                        parameter_prior_precision=1/np.square(settings['log_prior_sd']),
                        hemodynamics='tak', viscoelastic_s=cfg['fixed']['viscoelastic_s'],
                        initial_state_mode='volume_tied', initial_coordinate_weights=[1., 1., 1., 0., 1.],
                        **_solver_kwargs(cfg))
                    fit_summary = {key: fit.get(key) for key in ('status', 'converged', 'parameter_values',
                        'parameter_names', 'parameter_bounds', 'starts', 'selected_start', 'objective',
                        'objective_components', 'projected_scaled_gradient_inf_norm', 'log_parameter_boundary_distance',
                        'evaluations', 'raw_parameter_kkt_inf_norm', 'information')}
                    fit_summary.update(planned=n, fitted=len(eligible), warm_start_sources=[w[2] for w in warm],
                        prior_multiplicity='once_per_group', integration_difference_training_sd=[])
                    result['views'][view] = fit_summary
                    if 'prediction' in fit:
                        fp = _current_parameters(cfg, fit['parameter_values']); fitted_parameters[view] = fp
                        prediction[eligible] = fit['prediction']; drivers[eligible] = fit['driver']
                        initials[eligible] = fit['initial_state']; states[eligible] = fit['states']
                        for local, k in enumerate(eligible):
                            difference = np.inf
                            try:
                                finer = nonlinear_driver_forward(drivers[k], initials[k], fp, .1,
                                    substeps=cfg['solver']['validation_substeps'], derivative=False,
                                    numerical_backend='numba', hemodynamics='tak',
                                    viscoelastic_s=cfg['fixed']['viscoelastic_s'])
                                fine_prediction = (operators[k]@finer['canonical_prediction'].ravel()).reshape(300, 3)
                                difference = float(np.max(np.abs(fine_prediction-prediction[k])/sd))
                            except (ValueError, FloatingPointError, OverflowError):
                                pass
                            validated[k] = bool(fit.get('converged') and difference <= cfg['solver']['maximum_integration_difference_training_sd'])
                            fit_summary['integration_difference_training_sd'].append(difference)
                            rows[k][view+'_success'] = bool(validated[k])
                            rows[k][view+'_status'] = ('completed' if validated[k] else
                                'failed_integration' if difference > cfg['solver']['maximum_integration_difference_training_sd'] else 'failed_convergence')
                            rows[k][view+'_parameters'] = dict(zip(settings['parameter_names'], fit['parameter_values']))
                            rows[k][view+'_parameter_boundary'] = bool(np.any(np.asarray(fit['log_parameter_boundary_distance']) <= settings['boundary_fraction']))
                    saved[view+'_prediction'] = prediction
                    saved[view+'_driver'] = drivers; saved[view+'_initial_state'] = initials
                    saved[view+'_states'] = states
                from src.inference.tak_mismatch import score_tak_mismatch, weak_balance_residual, observable_drive_scenarios, nonlinear_drive_gap
                missing_predictions = np.full_like(targets, np.nan)
                saved['observable_standardized_hb_difference'] = np.full((n, 300, 2), np.nan)
                saved['drive_gap_difference'] = np.full((n, 300, 2), np.nan)
                for k, row in enumerate(rows):
                    scores = score_tak_mismatch(targets[k], predictions.get('hb_only', missing_predictions)[k],
                        predictions.get('joint', missing_predictions)[k], sd, visible=score_mask[k])
                    for name in ('D1', 'D_EH', 'D4', 'D_J_E', 'D_J_O', 'D_J_R'):
                        view = 'hb_only' if name in ('D1', 'D_EH') else 'joint'
                        row[name+'_free'] = scores.get(name) if valid_views.get(view, np.zeros(n, bool))[k] else None
                    row['status'] = ('completed' if valid_views.get('hb_only', np.zeros(n, bool))[k]
                        and (not np.isfinite(targets[k, :299, 0]).all() or valid_views.get('joint', np.zeros(n, bool))[k])
                        else 'partial_or_failed_fit' if np.isfinite(targets[k, :299, 1:]).all() else 'unavailable_complete_Hb_pair')
                    row['success'] = bool(valid_views.get('hb_only', np.zeros(n, bool))[k])
                    row.update(training_subjects=model['training_subjects'], group_planned_windows=n,
                        group_native_support=group['native_support_id'], sigma=sd)
                    if not row['success']: continue
                    fp = fitted_parameters['hb_only']
                    residual = (targets[k, :, 1:]-predictions['hb_only'][k, :, 1:])/coord['hb_factor']
                    common = score_mask[k, :, 1:].all(axis=1)
                    for component, label in enumerate(('HbO_RMSE_native_free', 'HbR_RMSE_native_free')):
                        row[label] = float(np.sqrt(np.mean(residual[common, component]**2)))
                    fixed_path = Path(out)/'tasks'/f'{row["fixed_task_id"]}.json'
                    fixed = read_json(fixed_path) if fixed_path.exists() else {}
                    if fixed.get('eeg_only_success') and fixed.get('array_path'):
                        with np.load(fixed['array_path'], allow_pickle=False) as fixed_arrays:
                            driver_e = fixed_arrays['driver_e'].copy()
                        driver_h = saved['hb_only_driver'][k]
                        observable = observable_drive_scenarios(driver_e, driver_h, driver_h,
                            saved['hb_only_initial_state'][k], fp, .1, sd, driver_train_sd=float(sd[0]),
                            mean_operator=operators[k], valid=score_mask[k, :, 1:], nuisance_parameters=('kappa', 'tau'),
                            noise_scenarios=cfg['observability']['scenarios'], draws=cfg['observability']['draws'],
                            quantile=cfg['observability']['quantile'], seed=cfg['seed'],
                            substeps=cfg['solver']['substeps'], viscoelastic_s=cfg['fixed']['viscoelastic_s'])
                        row.update(D_obs_free=observable.get('D_obs'), D_obs_free_status=observable.get('status'),
                            D_obs_allrank_free=observable.get('D_obs_allrank'),
                            observable_retained_rank_free=observable.get('retained_rank'),
                            observable_scenarios_free={name: {a: b for a, b in item.items() if not isinstance(b, np.ndarray)}
                                for name, item in observable.get('scenarios', {}).items()})
                        primary = observable['scenarios'][observable['primary_noise_scenario']]
                        saved['observable_standardized_hb_difference'][k] = primary['standardized_hb_difference']
                        gap = nonlinear_drive_gap(driver_e, driver_h, fp, .1, sd, mean_operator=operators[k],
                            valid=score_mask[k, :, 1:], substeps=cfg['solver']['substeps'],
                            viscoelastic_s=cfg['fixed']['viscoelastic_s'])
                        row['D_drive_H_free'] = gap.get('D_drive_H')
                        if 'difference' in gap: saved['drive_gap_difference'][k] = gap['difference']
                        row['drive_gap_free_condition'] = 'fixed_EEG_drive_and_free_Hb_drive_same_Hb_group_parameters_rest_initial'
                    else:
                        row['D_obs_free'] = None; row['D_obs_free_status'] = 'unavailable_successful_fixed_EEG_fit'
                # Weak profiles use the full original support at the one fitted
                # Hb-only group tau, never a separately tuned tau at each time.
                if 'hb_only' in fitted_parameters and result['views']['hb_only'].get('converged'):
                    j = group['tasks'][0]['pair_index']; t = np.asarray(arrays['continuous_hb_time_s'])
                    continuous = np.asarray(arrays['continuous_hb'])[:, j]
                    mask = np.asarray(arrays['continuous_hb_valid_mask'])[:, j].copy()
                    if 'continuous_hb_gap_valid_mask' in arrays:
                        gap_mask = np.asarray(arrays['continuous_hb_gap_valid_mask'], bool)
                        mask &= gap_mask[:, None] if gap_mask.ndim == 1 else gap_mask[:, j]
                    native = next((s for s in record.get('continuous_hb_segments', [])
                                   if s['id'] == group['native_support_id']), None)
                    starts = [x['hb_start_s'] for x in group['tasks']]
                    low = native['start_s'] if native is not None else min(starts)
                    high = native['end_s'] if native is not None else max(starts)+30.
                    within = (t >= low-1e-9) & (t < high-1e-9)
                    mask &= within[:, None]
                    weak = weak_balance_residual(continuous, .1, fitted_parameters['hb_only'], valid=mask,
                        support_widths_s=cfg['weak_balance']['support_widths_s'],
                        center_step_s=cfg['weak_balance']['center_step_s'],
                        excitation_floor=cfg['weak_balance']['excitation_floor'],
                        viscoelastic_s=cfg['fixed']['viscoelastic_s'])
                    result['weak_summary'] = _store_weak(saved, j, weak)
                    saved['continuous_hb_time_s'] = t
                    for row in rows:
                        for width, item in result['weak_summary'].items(): row['D_bal_'+width+'_free'] = item.get('D_bal')
                array_path = path.with_suffix('.npz')
                write_npz(array_path, **saved)
                for k, row in enumerate(rows): row.update(array_path=str(array_path), array_index=k)
                result['status'] = 'completed' if all(r['status'] == 'completed' for r in rows) else 'partial_or_unavailable'
    except Exception as exc:
        result.update(status='failed_exception', error=str(exc), traceback=traceback.format_exc())
        for row in rows:
            if row['status'] == 'unavailable_preparation': row['status'] = 'failed_exception'
    result.update(elapsed_seconds=time.monotonic()-started, parameter_mode='current_dof')
    write_json(path, result)
    return dict(group_id=group['id'], status=result['status'], terminal=len(rows))


def current_dof_stage(cfg, out, plan, workers):
    """Complete all Hb-only/joint parameter groups after fixed-view production."""
    count = sum(1 for _ in current_dof_groups(plan)); started = time.monotonic()
    results = bounded_work(_current_group_job,
        ((cfg, str(out), record, group) for record, group in current_dof_groups(plan)), workers,
        progress=_progress(out, 'current_dof_group_fit', count, started))
    result = dict(schema='tak_mismatch_current_dof_summary_v1', status='completed', planned_groups=count,
        terminal_groups=len(results), planned_windows=sum(len(g['tasks']) for _, g in current_dof_groups(plan)),
        terminal_windows=sum(r['terminal'] for r in results),
        status_counts=dict(Counter(r['status'] for r in results)),
        elapsed_seconds=time.monotonic()-started, workers=workers,
        grouping=cfg['current_dof']['grouping'], views=cfg['current_dof']['views'])
    write_json(Path(out)/'current_dof'/'summary.json', result)
    return result


def _csv(path, rows, fields):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader()
        for row in rows: writer.writerow({key: '' if value is None else value for key, value in row.items() if key in fields})
    temporary.replace(path)


def subject_aggregate(rows, metric):
    by_record = defaultdict(list); planned = Counter(); available = Counter()
    for row in rows:
        key = (row['dataset'], row['subject']); planned[key] += 1
        value = row.get(metric)
        if value is None or not np.isfinite(value): continue
        by_record[(*key, row['record_id'])].append(float(value)); available[key] += 1
    by_subject = defaultdict(list)
    for (dataset, subject, record), values in by_record.items():
        data = np.asarray(values)
        by_subject[(dataset, subject)].append(float(np.mean(data*data if metric in ROOT_METRICS else data)))
    result = []
    for (dataset, subject), total in sorted(planned.items()):
        values = by_subject[(dataset, subject)]
        value = float(np.mean(values)) if values else None
        if value is not None and metric in ROOT_METRICS: value = float(np.sqrt(value))
        result.append(dict(dataset=dataset, subject=subject, metric=metric, value=value,
            planned=total, available=available[(dataset, subject)], native_records=len(values)))
    return result


def choose_examples(rows):
    examples = []; datasets = sorted({row['dataset'] for row in rows})
    for dataset in datasets:
        valid = sorted((r for r in rows if r['dataset'] == dataset and r.get('D1') is not None
                        and r.get('array_path')), key=lambda r: (r['D1'], r['id']))
        used = set(); selected = {}
        for role, quantile in (('max', 1.), ('median', .5), ('p90', .9)):
            if not valid: continue
            target = float(np.quantile([r['D1'] for r in valid], quantile))
            eligible = [r for r in valid if r['subject'] not in used] or valid
            chosen = min(eligible, key=lambda r: (abs(r['D1']-target), r['id']))
            used.add(chosen['subject'])
            selected[role] = dict(chosen, role=role, candidate_denominator=len(valid),
                rank=next(k+1 for k, row in enumerate(valid) if row['id'] == chosen['id']),
                target_D1_quantile=target, selection_metric='nearest6_Hb_only_D1')
        examples.extend(selected[role] for role in ('median', 'p90', 'max') if role in selected)
    return dict(schema='tak_mismatch_examples_v1',
        selection='per_dataset_D1_median_p90_max_distinct_subject_when_possible_descriptive_only', examples=examples)


def _quantiles(values, weights=None):
    values = np.asarray(values, float)
    if not len(values): return None
    probabilities = np.array([.1, .25, .5, .75, .9, .95, 1.])
    if weights is None: result = np.quantile(values, probabilities)
    else:
        weights = np.asarray(weights, float); order = np.argsort(values)
        values, weights = values[order], weights[order]
        cumulative = np.cumsum(weights)
        if not cumulative[-1] > 0: return None
        locations = np.searchsorted(cumulative, probabilities*cumulative[-1], side='left')
        result = values[np.minimum(locations, len(values)-1)]
    return dict(zip(('p10', 'q25', 'median', 'q75', 'p90', 'p95', 'max'), result.tolist()))


def add_time_weights(rows, cfg, metrics=METRICS):
    """Within each native time window, available Hb pairs share its duration."""
    for metric in metrics:
        if metric.startswith('D_bal_'): continue
        count = Counter((r['record_id'], r['window']) for r in rows
                        if r.get(metric) is not None and np.isfinite(r[metric]))
        for row in rows:
            available = row.get(metric) is not None and np.isfinite(row[metric])
            component_count = row.get('eeg_count') if metric == 'D_EH' else row.get('hb_time_count')
            support = float(component_count if component_count is not None else
                cfg['tensor']['score_steps'][1]-cfg['tensor']['score_steps'][0])*cfg['tensor']['dt_s']
            weight = support/count[(row['record_id'], row['window'])] if available else 0.
            row['time_weight_'+metric+'_s'] = weight
            if metric == 'D1': row['time_weight_s'] = weight


def _current_dof_tables(cfg, out, plan, fixed_rows):
    """Export the one group owner without rediscovering unrelated run files."""
    rows = []; missing_groups = []; group_statuses = Counter(); weak_parts = defaultdict(list)
    for record, group in current_dof_groups(plan):
        path = Path(out)/'current_dof'/'tasks'/f'{group["id"]}.json'
        retained = read_json(path) if path.exists() else None
        if retained is None: missing_groups.append(group['id'])
        group_statuses[retained['status'] if retained else 'missing'] += 1
        existing = {r['id']: r for r in retained.get('rows', [])} if retained else {}
        for task in group['tasks']:
            row = existing.get(task['id'], dict(task, parameter_mode='current_dof', group_id=group['id'],
                fixed_task_id=task['id'], status='missing', success=False))
            for view, label in (('hb_only', 'hb'), ('joint', 'joint')):
                values = row.get(view+'_parameters', {})
                row['kappa_'+label] = values.get('kappa'); row['tau_'+label] = values.get('tau')
            for name, scenario in row.get('observable_scenarios_free', {}).items():
                row['D_obs_'+name+'_free'] = scenario.get('D_obs')
            rows.append(row)
        for width in cfg['weak_balance']['support_widths_s']:
            label = f'{width:g}s'
            values = retained.get('weak_summary', {}).get(label, {}) if retained else {}
            weak_parts[(record['dataset'], record['subject'], record['id'], group['pair'], width)].append(values)
    weak_rows = []
    for (dataset, subject, record_id, pair, width), parts in weak_parts.items():
        usable = [part for part in parts if part.get('D_bal') is not None and part.get('denominator', 0.) > 0]
        denominator = sum(part['denominator'] for part in usable)
        numerator = sum(part['D_bal']**2*part['denominator'] for part in usable)
        weak_rows.append(dict(dataset=dataset, subject=subject, record_id=record_id, pair=pair,
            width_s=width, D_bal=np.sqrt(numerator/denominator) if denominator > 0 else None,
            status='completed' if len(usable) == len(parts) and usable else 'partial_or_unavailable',
            planned_groups=len(parts), available_groups=len(usable), denominator=denominator,
            planned_centers=sum(part.get('planned_centers', 0) for part in parts),
            valid_centers=sum(part.get('valid_centers', 0) for part in usable)))
    _csv(Path(out)/'weak_record_summary_current_dof.csv', weak_rows,
        ('dataset', 'subject', 'record_id', 'pair', 'width_s', 'D_bal', 'status', 'planned_groups',
         'available_groups', 'denominator', 'planned_centers', 'valid_centers'))
    add_time_weights(rows, cfg, FREE_METRICS)
    fields = ('id', 'fixed_task_id', 'dataset', 'subject', 'record_id', 'record', 'session',
        'window', 'pair', 'fold', 'parameter_mode', 'start_s', 'hb_start_s', 'end_s', 'status', 'success',
        'group_id', 'group_native_support', 'group_planned_windows', 'hb_only_success', 'joint_success',
        'hb_only_status', 'joint_status', 'kappa_hb', 'tau_hb', 'kappa_joint', 'tau_joint',
        'hb_only_parameter_boundary', 'joint_parameter_boundary', 'D_obs_free_status',
        'observable_retained_rank_free', 'array_path', 'array_index')
    _csv(Path(out)/'metrics_current_dof.csv', rows, (*fields, *FREE_METRICS,
        *('time_weight_'+metric+'_s' for metric in FREE_METRICS if not metric.startswith('D_bal_'))))
    per_metric = {}
    for metric in FREE_METRICS:
        if metric.startswith('D_bal_'):
            width = float(metric.removeprefix('D_bal_').removesuffix('s_free'))
            per_metric[metric] = [dict(r, **{metric: r.get('D_bal')}) for r in weak_rows if r['width_s'] == width]
        else: per_metric[metric] = rows
    subjects = [r for metric in FREE_METRICS for r in subject_aggregate(per_metric[metric], metric)]
    _csv(Path(out)/'subject_summary_current_dof.csv', subjects,
        ('dataset', 'subject', 'metric', 'value', 'planned', 'available', 'native_records'))
    by_dataset = {}; rng = np.random.default_rng(cfg['seed'])
    for dataset in cfg['data']['datasets']:
        current = [row for row in rows if row['dataset'] == dataset]; metrics = {}
        for metric in FREE_METRICS:
            eligible = [r for r in per_metric[metric] if r['dataset'] == dataset]
            values = np.asarray([r[metric] for r in eligible if r.get(metric) is not None], float)
            values = values[np.isfinite(values)]
            means = np.asarray([r['value'] for r in subjects if r['dataset'] == dataset
                and r['metric'] == metric and r['value'] is not None], float)
            ci = None
            if len(means) >= 2:
                draws = rng.choice(means, (cfg['endpoints']['bootstrap_repeats'], len(means)), replace=True)
                bootstrap = np.sqrt(np.mean(draws*draws, 1)) if metric in ROOT_METRICS else np.mean(draws, 1)
                ci = np.quantile(bootstrap, [.025, .975])
            metrics[metric] = dict(planned=len(eligible), available=len(values), subjects=len(means),
                window_quantiles=_quantiles(values), subject_quantiles=_quantiles(means),
                time_weighted_quantiles=None if metric.startswith('D_bal_') else _quantiles(
                    [r[metric] for r in eligible if r.get(metric) is not None and np.isfinite(r[metric])],
                    [r['time_weight_'+metric+'_s'] for r in eligible if r.get(metric) is not None and np.isfinite(r[metric])]),
                subject_equal_mean=(float(np.sqrt(np.mean(means*means))) if metric in ROOT_METRICS else float(np.mean(means))) if len(means) else None,
                subject_bootstrap_95pct=ci)
        by_dataset[dataset] = dict(planned=len(current), terminal=sum(r['status'] != 'missing' for r in current),
            statuses=dict(Counter(r['status'] for r in current)), metrics=metrics)
    paired = {}; fixed_by_id = {row['id']: row for row in fixed_rows}
    for dataset in cfg['data']['datasets']:
        paired[dataset] = {}
        for metric in ('D1', 'D_EH', 'D_obs', 'D_drive_H', 'D4', 'D_J_E', 'D_J_O', 'D_J_R'):
            paired_fixed = []; paired_free = []
            for row in rows:
                fixed = fixed_by_id.get(row['id'], {})
                if (row['dataset'] != dataset or row.get(metric+'_free') is None or fixed.get(metric) is None
                        or not np.isfinite(row[metric+'_free']) or not np.isfinite(fixed[metric])): continue
                paired_fixed.append(dict(fixed, **{metric: fixed[metric]}))
                paired_free.append(dict(row, **{metric: row[metric+'_free']}))
            left = {r['subject']: r['value'] for r in subject_aggregate(paired_fixed, metric)}
            right = {r['subject']: r['value'] for r in subject_aggregate(paired_free, metric)}
            difference = np.asarray([left[s]-right[s] for s in sorted(left.keys() & right.keys())], float)
            ci = None
            if len(difference) >= 2:
                ci = np.quantile(rng.choice(difference, (cfg['endpoints']['bootstrap_repeats'], len(difference)),
                    replace=True).mean(1), [.025, .975])
            paired[dataset][metric] = dict(paired_windows=len(paired_fixed), paired_subjects=len(difference),
                fixed_minus_free_subject_mean=float(np.mean(difference)) if len(difference) else None,
                subject_bootstrap_95pct=ci, positive_direction='free_parameter_model_has_lower_residual')
    result = dict(status='completed' if not missing_groups and all(r['status'] != 'missing' for r in rows) else 'incomplete',
        planned_groups=sum(group_statuses.values()), terminal_groups=sum(group_statuses.values())-len(missing_groups),
        missing_groups=len(missing_groups), planned_windows=len(rows), terminal_windows=sum(r['status'] != 'missing' for r in rows),
        statuses=dict(group_statuses))
    return result, by_dataset, paired


def summarize_stage(cfg, out, plan):
    rows = []; missing = []; statuses = Counter()
    for task in task_manifest(plan):
        path = Path(out)/'tasks'/f'{task["id"]}.json'
        if path.exists(): row = read_json(path)
        else:
            row = dict(task, status='missing', success=False); missing.append(task['id'])
        statuses[row['status']] += 1; rows.append(row)
    add_time_weights(rows, cfg)
    identity_fields = ('id', 'dataset', 'subject', 'record_id', 'record', 'session', 'window', 'pair', 'fold',
        'parameter_mode', 'start_s', 'end_s', 'hb_start_s', 'duration_s', 'actual_context_s',
        'status', 'success', 'hb_only_success', 'eeg_only_success', 'joint_success',
        'eeg_input_status',
        'selected_candidate', 'D_obs_status', 'axis_3_selection_status',
        'observable_numerical_rank', 'observable_retained_rank', 'observable_primary_noise_scenario',
        'array_path', 'weak_path', 'weak_pair_index', 'time_weight_s')
    weight_fields = tuple('time_weight_'+metric+'_s' for metric in METRICS if not metric.startswith('D_bal_'))
    _csv(Path(out)/'metrics.csv', rows, (*identity_fields, *METRICS, *weight_fields))
    weak_rows = []
    for record in plan['records']:
        metadata_path, _, _ = prepared_paths(out, record['id'])
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        for j, pair in enumerate(record['pairs']):
            for width in cfg['weak_balance']['support_widths_s']:
                label = f'{width:g}s'
                values = metadata.get('weak_summary', {}).get(str(j), {}).get(label,
                    dict(status='unavailable_preparation', D_bal=None, planned_centers=0, valid_centers=0))
                weak_rows.append(dict(dataset=record['dataset'], subject=record['subject'],
                    record_id=record['id'], record=record['record'], pair=pair['id'],
                    pair_index=j, width_s=float(width), weak_path=metadata.get('weak_path'), **values))
    _csv(Path(out)/'weak_record_summary.csv', weak_rows,
        ('dataset', 'subject', 'record_id', 'record', 'pair', 'pair_index', 'width_s',
         'D_bal', 'status', 'planned_centers', 'valid_centers', 'denominator', 'mean_denominator', 'weak_path'))
    metric_rows = {}
    for metric in METRICS:
        if metric.startswith('D_bal_'):
            width = float(metric[6:-1])
            metric_rows[metric] = [dict(r, **{metric: r.get('D_bal')}) for r in weak_rows if r['width_s'] == width]
        else: metric_rows[metric] = rows
    subject_rows = [r for metric in METRICS for r in subject_aggregate(metric_rows[metric], metric)]
    _csv(Path(out)/'subject_summary.csv', subject_rows,
         ('dataset', 'subject', 'metric', 'value', 'planned', 'available', 'native_records'))
    by_dataset = {}
    rng = np.random.default_rng(cfg['seed'])
    for dataset in cfg['data']['datasets']:
        current = [r for r in rows if r['dataset'] == dataset]; metrics = {}
        for metric in METRICS:
            eligible = [r for r in metric_rows[metric] if r['dataset'] == dataset]
            good = [r for r in eligible if r.get(metric) is not None and np.isfinite(r[metric])]
            values = np.array([r[metric] for r in good], float)
            subjects = np.array([r['value'] for r in subject_rows
                if r['dataset'] == dataset and r['metric'] == metric and r['value'] is not None], float)
            ci = None
            if len(subjects) >= 2:
                samples = rng.choice(subjects, (cfg['endpoints']['bootstrap_repeats'], len(subjects)), replace=True)
                if metric in ROOT_METRICS: boot = np.sqrt(np.mean(samples*samples, axis=1))
                else: boot = np.mean(samples, axis=1)
                ci = np.quantile(boot, [.025, .975])
            metrics[metric] = dict(planned=len(eligible), available=len(values), subjects=len(subjects),
                observational_unit='record_pair' if metric.startswith('D_bal_') else 'nonoverlapping_30s_window_pair',
                window_quantiles=_quantiles(values), subject_quantiles=_quantiles(subjects),
                time_weighted_quantiles=None if metric.startswith('D_bal_') else _quantiles(values,
                    [r['time_weight_'+metric+'_s'] for r in good]),
                scored_time_seconds=None if metric.startswith('D_bal_') else sum(r['time_weight_'+metric+'_s'] for r in good),
                subject_equal_mean=(float(np.sqrt(np.mean(subjects*subjects))) if metric in ROOT_METRICS else float(np.mean(subjects))) if len(subjects) else None,
                subject_bootstrap_95pct=ci)
        by_dataset[dataset] = dict(planned=len(current), terminal=sum(r['status'] != 'missing' for r in current),
            statuses=dict(Counter(r['status'] for r in current)), metrics=metrics)
    current_summary, by_dataset_current_dof, paired_current_dof = _current_dof_tables(cfg, out, plan, rows)
    summary = dict(schema='tak_mismatch_summary_v1',
        status='completed' if not missing and current_summary['status'] == 'completed' else 'incomplete',
        experiment_id=cfg['experiment_id'], parameter_mode='fixed', fixed_parameters=cfg['fixed'],
        planned=len(rows), terminal=len(rows)-len(missing), missing=len(missing), statuses=dict(statuses),
        primary=cfg['endpoints']['primary'], by_dataset=by_dataset,
        current_dof=current_summary, by_dataset_current_dof=by_dataset_current_dof,
        paired_current_dof=paired_current_dof,
        files=dict(plan='plan.json', metrics='metrics.csv', subjects='subject_summary.csv',
            weak='weak_record_summary.csv', examples='examples.json',
            current_dof_metrics='metrics_current_dof.csv', current_dof_subjects='subject_summary_current_dof.csv',
            current_dof_weak='weak_record_summary_current_dof.csv'),
        aggregation='time/species standardized MSE within window; pair-windows equal within native record; records equal within subject; subjects equal within dataset; root metrics squared before averaging and square-rooted afterwards',
        intervals='descriptive whole-subject percentile bootstrap; coordinates and selection frozen; no multiplicity claim',
        axis_3='nonlinear heldout prediction difference using inner subject linear screen frozen candidate',
        D_obs='local Hb-only drive tangent; initial states and EEG-reference DC nuisance removed; prespecified standardized-Hb AR1 noise scenarios; sensitivity, not empirical sensor-noise calibration',
        observability=dict(cfg['observability'], seed=cfg['seed']),
        weak_balance='continuous record complete-support 10/20/40s; distributions and subject statistics use independent record-pair rows from weak_record_summary.csv',
        time_weighting='each scored 24.9s native record-window is counted once across available Hb pairs; metric-specific pair inverse weights; these are duration-weighted window scores, not instantaneous residual exceedance probabilities',
        claim_limits=cfg['claim_limits'], metric_definitions={
            'D1': 'Hb-only fit: sqrt mean standardized HbO/HbR squared residual',
            'D_EH': 'EEG proxy minus Hb-only predicted EEG, divided by frozen training EEG SD, RMS',
            'D4': 'joint-fit EEG/HbO/HbR equal-component standardized RMS',
            'Delta_spatial': 'nearest6 minus train-selected geometry independent EEG-only Hb NMSE; positive favors selected',
            'D_drive_H': 'EEG-only and Hb-only drives replayed with identical resting vascular initial state, Hb standardized RMS difference',
            'D_bal': 'weak small-signal Tak relation relative discrepancy on complete continuous-record supports',
            'D_obs': 'local Hb-observable drive gap in modes exceeding prespecified synthetic-noise scenario thresholds; insufficient retained rank remains unavailable'})
    write_json(Path(out)/'summary.json', summary); write_json(Path(out)/'examples.json', choose_examples(rows))
    return serial(summary)


AUDIT_CATEGORIES = ('completed', 'numerical_failure', 'unavailable_support', 'execution_failure', 'missing')


def _audit_bucket():
    return dict(planned=0, fitted=0, **{name: 0 for name in AUDIT_CATEGORIES},
        raw_status_counts=dict(task={}, view={}, solver={}, classification_reason={}))


def _audit_count(bucket, category, count=1, *, task_status=None, view_status=None,
                 solver_status=None, reason=None, fitted=None):
    if category not in AUDIT_CATEGORIES or not isinstance(count, int) or count < 0:
        raise ValueError('invalid terminal audit category or count')
    bucket['planned'] += count; bucket[category] += count
    entered = category in ('completed', 'numerical_failure') if fitted is None else fitted
    if entered: bucket['fitted'] += count
    for field, value in (('task', task_status), ('view', view_status), ('solver', solver_status),
                         ('classification_reason', reason)):
        if value is not None:
            counts = bucket['raw_status_counts'][field]; key = str(value)
            counts[key] = counts.get(key, 0)+count


def _audit_rate(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
        value=numerator/denominator if denominator else None)


def _audit_finish_bucket(bucket):
    if sum(bucket[name] for name in AUDIT_CATEGORIES) != bucket['planned']:
        raise ValueError('terminal audit categories do not conserve the planned denominator')
    if bucket['fitted'] != bucket['completed']+bucket['numerical_failure']:
        raise ValueError('terminal audit optimizer-entry denominator is inconsistent')
    bucket['rates'] = dict(
        support_rate=_audit_rate(bucket['fitted'], bucket['planned']),
        numerical_failure_rate=_audit_rate(bucket['numerical_failure'], bucket['fitted']),
        completion_rate=_audit_rate(bucket['completed'], bucket['planned']))
    return bucket


def _audit_is_terminal(status):
    return isinstance(status, str) and (status == 'completed' or status.startswith(
        ('failed_', 'unavailable_', 'partial_or_')))


def _audit_fixed_view(row, view):
    """A failed numerical validation and an absent input are different facts."""
    task_status = row.get('status')
    entry = row.get('view_statuses', {}).get(view)
    if entry is not None:
        if not isinstance(entry, dict): raise ValueError('fixed view status must be an object')
        status, solver = entry.get('status'), entry.get('solver_status')
        if entry.get('success') is True: category, reason = 'completed', 'validated_view_success'
        elif any(isinstance(value, str) and value.startswith('unavailable_') for value in (status, solver)):
            category, reason = 'unavailable_support', 'explicit_view_or_solver_support_status'
        elif status in ('failed_numerical', 'failed_integration', 'failed_convergence'):
            category, reason = 'numerical_failure', 'view_did_not_pass_numerical_validation'
        elif status == 'failed_exception': category, reason = 'execution_failure', 'view_execution_exception'
        else: category, reason = 'execution_failure', 'unrecognized_or_missing_view_verdict'
        return category, status, solver, reason
    if task_status == 'unavailable_preparation':
        return 'execution_failure', None, None, 'preparation_did_not_produce_a_consumable_record'
    if isinstance(task_status, str) and task_status.startswith('unavailable_'):
        return 'unavailable_support', None, None, 'explicit_task_support_status'
    if task_status == 'failed_exception':
        return 'execution_failure', None, None, 'task_exception_before_view_verdict'
    if view != 'hb_only' and str(row.get('eeg_input_status', '')).startswith('unavailable_'):
        return 'unavailable_support', None, None, 'EEG_input_not_available_for_this_view'
    return 'execution_failure', None, None, 'terminal_task_missing_required_view_verdict'


def _audit_identity(row, task, *, label):
    if row.get('id') != task['id']:
        raise ValueError(f'{label}: retained task identity differs from the frozen plan')
    for key in ('dataset', 'subject', 'record_id', 'window', 'pair'):
        if key in row and row[key] != task[key]:
            raise ValueError(f'{label}: retained {key} differs from the frozen plan')


def _audit_extra_counts(output, row, *, current=False):
    fields = ('D_obs_free_status',) if current else ('D_obs_status', 'spatial_status', 'axis_3_selection_status')
    for field in fields:
        value = row.get(field, '__not_recorded__')
        if isinstance(value, dict): value = value.get('status', '__missing_status__')
        if value is None: value = '__not_run_or_unavailable__'
        counts = output.setdefault(field, {}); key = str(value)
        counts[key] = counts.get(key, 0)+1
    if not current:
        for name in ('wrong', 'selected_wrong'):
            entry = row.get('wrong_statuses', {}).get(name, '__not_recorded__')
            if isinstance(entry, dict): entry = entry.get('status', '__missing_status__')
            counts = output.setdefault('wrong_statuses.'+name, {}); key = str(entry)
            counts[key] = counts.get(key, 0)+1


def terminal_status_audit_stage(cfg, out):
    """Read retained terminal JSON only and create one fresh, exclusive audit.

    Incomplete snapshots are returned without writing an artifact. A completed
    audit is never replaced. Validated per-view outcomes remain usable when a
    later task/group operation raised; parent execution statuses are retained
    separately. A group view with a recorded ``fitted`` count has an exact
    optimizer-entry denominator: successes plus numerical validations not
    passed. A missing view after an execution exception is never called a lack
    of input support.
    """
    import inspect
    import subprocess
    import tempfile
    out = Path(out); target = out/'terminal_status_audit.json'
    if target.exists(): raise FileExistsError(f'retained terminal audit already exists: {target}')
    plan = read_json(out/'plan.json')
    resolved = read_json(out/'resolved_config.json')
    if resolved != serial(cfg): raise ValueError('audit config differs from the retained resolved contract')
    source_path, launch_path = out/'source_manifest.json', out/'launch.json'
    production = read_json(source_path) if source_path.exists() else None
    sources_complete = production is not None and launch_path.is_file()
    by_dataset = {}
    for dataset in cfg['data']['datasets']:
        by_dataset[dataset] = dict(
            fixed=dict(planned_tasks=0, terminal_tasks=0, task_status_counts={},
                       views={view: _audit_bucket() for view in ('hb_only', 'eeg_only', 'joint')}, extra_statuses={}),
            current_dof=dict(planned_groups=0, terminal_groups=0, group_status_counts={},
                       views={view: _audit_bucket() for view in cfg['current_dof']['views']}, extra_statuses={}))
    fixed_ids = set(); fixed_missing = 0
    for task in task_manifest(plan):
        if task['id'] in fixed_ids: raise ValueError('duplicate fixed task identity in frozen plan')
        fixed_ids.add(task['id']); section = by_dataset[task['dataset']]['fixed']
        section['planned_tasks'] += 1; path = out/'tasks'/f'{task["id"]}.json'
        row = read_json(path) if path.exists() else None
        if row is not None: _audit_identity(row, task, label=str(path))
        status = row.get('status') if row is not None else '__missing_artifact__'
        section['task_status_counts'][status] = section['task_status_counts'].get(status, 0)+1
        terminal = row is not None and _audit_is_terminal(status)
        if terminal: section['terminal_tasks'] += 1
        else: fixed_missing += 1
        for view, bucket in section['views'].items():
            if not terminal:
                _audit_count(bucket, 'missing', task_status=status, reason='terminal_task_record_not_available')
            else:
                category, view_status, solver, reason = _audit_fixed_view(row, view)
                _audit_count(bucket, category, task_status=status, view_status=view_status,
                    solver_status=solver, reason=reason)
        if row is not None: _audit_extra_counts(section['extra_statuses'], row)
    group_ids = set(); group_missing = 0; current_windows = 0; current_terminal_windows = 0
    for record, group in current_dof_groups(plan):
        if group['id'] in group_ids: raise ValueError('duplicate current-dof group identity in frozen plan')
        group_ids.add(group['id']); section = by_dataset[record['dataset']]['current_dof']
        section['planned_groups'] += 1; n = len(group['tasks']); current_windows += n
        path = out/'current_dof'/'tasks'/f'{group["id"]}.json'
        retained = read_json(path) if path.exists() else None
        status = retained.get('status') if retained is not None else '__missing_artifact__'
        section['group_status_counts'][status] = section['group_status_counts'].get(status, 0)+1
        terminal = retained is not None and _audit_is_terminal(status)
        if not terminal:
            group_missing += 1
            for bucket in section['views'].values():
                _audit_count(bucket, 'missing', n, task_status=status, reason='terminal_group_record_not_available')
            continue
        section['terminal_groups'] += 1; current_terminal_windows += n
        if retained.get('id', group['id']) != group['id']:
            raise ValueError(f'{path}: retained group identity differs from frozen plan')
        expected = {task['id']: task for task in group['tasks']}
        rows = retained.get('rows', [])
        if len(rows) != n or len({row.get('id') for row in rows}) != n or {row.get('id') for row in rows} != set(expected):
            raise ValueError(f'{path}: current-dof row denominator/identities differ from frozen group')
        for row in rows:
            _audit_identity(row, expected[row['id']], label=str(path))
            _audit_extra_counts(section['extra_statuses'], row, current=True)
        for view, bucket in section['views'].items():
            entry = retained.get('views', {}).get(view)
            if entry is None or 'fitted' not in entry:
                unavailable = isinstance(status, str) and status.startswith('unavailable_') and status != 'unavailable_preparation'
                _audit_count(bucket, 'unavailable_support' if unavailable else 'execution_failure', n,
                    task_status=status, reason='explicit_group_support_status' if unavailable else 'group_execution_missing_view_verdict')
                continue
            fitted = entry['fitted']; planned = entry.get('planned', n)
            if (isinstance(fitted, bool) or not isinstance(fitted, int) or not 0 <= fitted <= n or planned != n):
                raise ValueError(f'{path}: invalid {view} fitted/planned denominator')
            success = sum(row.get(view+'_success') is True for row in rows)
            if success > fitted: raise ValueError(f'{path}: {view} successes exceed optimizer entries')
            _audit_count(bucket, 'completed', success, task_status=status, view_status='completed',
                solver_status=entry.get('status'), reason='validated_group_view_success')
            failures = [row.get(view+'_status', '__no_row_numerical_verdict__') for row in rows
                        if row.get(view+'_success') is not True and view+'_status' in row]
            _audit_count(bucket, 'numerical_failure', fitted-success, task_status=status,
                view_status=entry.get('status'), solver_status=entry.get('status'),
                reason='recorded_optimizer_entries_minus_validated_successes')
            _audit_count(bucket, 'unavailable_support', n-fitted, task_status=status,
                view_status=entry.get('status'), reason='planned_group_windows_not_entering_optimizer')
            detail = bucket['raw_status_counts'].setdefault('unsuccessful_row_view', {})
            for value, count in Counter(failures).items(): detail[value] = detail.get(value, 0)+count
    if current_windows != len(fixed_ids):
        raise ValueError('fixed and current-dof planned window denominators differ')
    flat = []
    for dataset, modes in by_dataset.items():
        declared = plan.get('summary', {}).get(dataset, {}).get('planned_pair_windows')
        if declared is not None and declared != modes['fixed']['planned_tasks']:
            raise ValueError(f'{dataset}: reconstructed denominator differs from frozen inventory summary')
        for mode, section in modes.items():
            for view, bucket in section['views'].items():
                _audit_finish_bucket(bucket)
                if bucket['planned'] != by_dataset[dataset]['fixed']['planned_tasks']:
                    raise ValueError(f'{dataset}/{mode}/{view}: view denominator differs from complete task plan')
                flat.append(dict(dataset=dataset, parameter_mode=mode, view=view, **bucket))
    try:
        git_head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError): git_head = None
    helpers = (_audit_bucket, _audit_count, _audit_rate, _audit_finish_bucket, _audit_is_terminal,
               _audit_fixed_view, _audit_identity, _audit_extra_counts, token, wrong_eeg_partners,
               task_manifest, current_dof_groups, terminal_status_audit_stage)
    completed = fixed_missing == group_missing == 0 and sources_complete
    result = dict(schema='tak_mismatch_terminal_status_audit_v1', status='completed' if completed else 'incomplete',
        all_parameter_modes_terminal=fixed_missing == group_missing == 0, artifact_written=completed,
        created_utc=datetime.now(timezone.utc).isoformat(),
        fixed=dict(planned_tasks=len(fixed_ids), terminal_tasks=len(fixed_ids)-fixed_missing, missing_tasks=fixed_missing),
        current_dof=dict(planned_groups=len(group_ids), terminal_groups=len(group_ids)-group_missing,
            missing_groups=group_missing, planned_windows=current_windows, terminal_windows=current_terminal_windows),
        by_dataset=by_dataset, rows=flat, measured_arrays_read=0, fits_executed=0,
        production_sources=dict(plan='plan.json', resolved_config='resolved_config.json',
            source_manifest='source_manifest.json', launch='launch.json', complete=sources_complete,
            source_git_head=production.get('source_git_head') if production else None,
            source_snapshot=production.get('source_snapshot') if production else None),
        audit_source=dict(path=str(Path(__file__).resolve()), git_head=git_head,
            implementation='\n\n'.join(inspect.getsource(function) for function in helpers)),
        definitions=dict(
            support_rate='observed optimizer entries / all planned windows; execution failures and missing records remain separately visible',
            numerical_failure_rate='numerical validation not passed / actual optimizer entries; unavailable support is excluded',
            completed='successful convergence and integration validation recorded for the view',
            current_dof_fitted='owning group views[view].fitted; shared parameter optimization can cover several windows',
            exceptions='existing per-view numerical verdicts are retained; missing view verdicts after execution exceptions are execution failures',
            extras='verbatim raw supplementary statuses, without treating insufficient observable rank as a numerical fit failure'))
    if completed:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=out, prefix='.terminal_status_audit.',
                    suffix='.tmp', encoding='utf-8', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(json.dumps(serial(result), ensure_ascii=False, allow_nan=False, indent=2)+'\n')
            os.link(temporary, target)  # Atomic creation, failing if another writer won.
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)
    return serial(result)


def synthetic_stage(cfg, out, workers=1):
    """One full-grid numerical contract without measured/native array reads."""
    from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
    from src.inference.tak_mismatch import fit_tak_views, score_tak_mismatch, weak_balance_residual
    clock = np.arange(300)*.1; driver = .02*np.sin(2*np.pi*clock/15.)
    truth = nonlinear_driver_forward(driver, np.r_[0., np.ones(4)], parameters(cfg), .1,
        substeps=8, derivative=False, numerical_backend='numba', hemodynamics='tak', viscoelastic_s=2.)
    target = (reference_operator()@truth['canonical_prediction'].ravel()).reshape(300, 3)
    target[-1] = np.nan; scale = np.array([.025, .025, .015])
    started = time.monotonic(); fits = _fit_views(cfg, target, scale, ('hb_only', 'eeg_only', 'joint'))
    validation = {name: _validate_fit(fit, cfg, scale) for name, fit in fits.items()}
    mask = np.isfinite(target); mask[:50] = False; mask[-1] = False
    metrics = score_tak_mismatch(target, np.asarray(fits['hb_only']['prediction'])[0],
        np.asarray(fits['joint']['prediction'])[0], scale, visible=mask)
    weak = weak_balance_residual(np.zeros((900, 2)), .1, parameters(cfg),
        support_widths_s=cfg['weak_balance']['support_widths_s'],
        excitation_floor=cfg['weak_balance']['excitation_floor'], viscoelastic_s=2.)
    result = dict(schema='tak_mismatch_synthetic_v1', status='completed', measured_arrays_read=0,
        all_views_successful=all(r['success'] for r in validation.values()), views=validation,
        metrics={key: value for key, value in metrics.items() if np.isscalar(value)},
        weak_zero_excitation=serial(weak), elapsed_seconds=time.monotonic()-started,
        truth_used_for_fitting=False, workers=workers)
    write_json(Path(out)/'synthetic_summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--stage', choices=STAGES, default='check')
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-tasks', type=int, help='Explicit bounded throughput pilot; never marks the full campaign complete')
    parser.add_argument('--max-records', type=int, help='Prepare-only dataset-round-robin resource pilot; the complete plan is retained')
    args = parser.parse_args(argv); cfg = read_config(args.config)
    if not 1 <= args.workers <= cfg['resources']['max_workers']: parser.error('workers outside reviewed resource envelope')
    if args.max_tasks is not None and (args.stage != 'run' or args.max_tasks < 1):
        parser.error('--max-tasks is a positive run-stage throughput pilot only')
    if args.max_records is not None and (args.stage != 'prepare' or args.max_records < 1):
        parser.error('--max-records is a positive prepare-stage throughput pilot only')
    if args.stage == 'check':
        print(json.dumps(serial(check(cfg)), ensure_ascii=False, allow_nan=False)); return 0
    root = args.project_root.resolve()
    out = (args.run_dir or root/ARTIFACT_ROOT/cfg['run_name']).resolve()
    owner = root/ARTIFACT_ROOT
    if out.parent != owner.resolve() or out.name != cfg['run_name']:
        parser.error('this contract writes only its exact versioned owning run directory')
    if args.stage == 'audit-status':
        result = terminal_status_audit_stage(cfg, out)
        print(json.dumps(serial(result), ensure_ascii=False, allow_nan=False))
        return 0 if result['status'] == 'completed' else 2
    with run_lock(out):
        freeze_config(cfg, out)
        if args.stage == 'synthetic': result = synthetic_stage(cfg, out, args.workers)
        elif args.stage == 'plan': result = plan_stage(cfg, out, root)
        else:
            if not (out/'plan.json').exists(): raise ValueError('plan stage required first')
            plan = read_json(out/'plan.json')
            if args.stage == 'prepare': result = prepare_stage(cfg, out, root, plan, args.workers, max_records=args.max_records)
            elif args.stage == 'run': result = run_stage(cfg, out, plan, args.workers, max_tasks=args.max_tasks)
            else: result = summarize_stage(cfg, out, plan)
    display = result
    if args.stage == 'plan':
        display = {key: result[key] for key in ('schema', 'status', 'summary', 'measured_arrays_read')}
        display['record_count'] = len(result['records']); display['plan_path'] = str(out/'plan.json')
    print(json.dumps(serial(display), ensure_ascii=False, allow_nan=False)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
