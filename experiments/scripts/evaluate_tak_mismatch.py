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
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
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
VISUAL_CONFIG = ROOT/'experiments/configs/physiology_semantic_tokenizer/tak_mismatch_visual_corrected_v2.yaml'
DEFAULT_PHYSIOLOGY_CONFIG = ROOT/'experiments/configs/physiology_semantic_tokenizer/tak_mismatch_fixed_physiology_v1.yaml'
VISUAL_PHYSIOLOGY_CONFIG = ROOT/'experiments/configs/physiology_semantic_tokenizer/tak_mismatch_visual_physiology_v2.yaml'
ARTIFACT_ROOT = 'experiments/runs/physiology_semantic_tokenizer/tak_mismatch'
PHYSIOLOGY_STAGES = ('physiology-extract', 'physiology-analyze', 'physiology-ocular', 'physiology-counterfactual', 'physiology-shapes')
STAGES = ('check', 'synthetic', 'plan', 'prepare', 'run', 'fixed-run', 'summarize', 'audit-status', 'plot-plan', 'plot-run',
          'physiology-topography', 'physiology-patterns', 'physiology-four-dataset', *PHYSIOLOGY_STAGES)
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
    contracts = {'tak_mismatch_v1': DEFAULT_CONFIG,
                 'tak_mismatch_visual_corrected_v2': VISUAL_CONFIG}
    if cfg.get('schema') not in contracts:
        raise ValueError('Unknown versioned Tak mismatch contract')
    expected = yaml.safe_load(contracts[cfg['schema']].read_text())
    for key in expected:
        if key not in ('solver', 'resources') and cfg.get(key) != expected[key]:
            raise ValueError(f'versioned Tak mismatch contract drift: {key}')
    if cfg.get('artifact_root') != ARTIFACT_ROOT:
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
            visual_eeg_channel_identity=cfg['data'].get('visual_eeg_channel_identity'),
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


def run_stage(cfg, out, plan, workers, *, max_tasks=None, include_current_dof=True):
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
    if max_tasks is None and include_current_dof:
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


def _plot_task_labels(record, windows):
    """Metadata labels only; REFED videos remain separate plotting cells."""
    if record['dataset'] == 'refed':
        return ['emotion_video/'+record['record']]
    labels = sorted({w.get('task') for w in windows if w.get('task')})
    if labels:
        return labels
    if record['dataset'] == 'simultaneous_eeg_nirs':
        return [record['record'].removeprefix('cnt_')]
    if record['dataset'] == 'visual_cognitive_motivation':
        return ['visual_cognitive_motivation']
    return ['unlabeled_record/'+record['record']]


def _plot_pairs(record, count=3):
    """Deterministic farthest-point sampling in the admitted native geometry."""
    geometry = record.get('geometry_identity', {}).get('fnirs', [])
    points = {}
    for pair in record['pairs']:
        index = pair['component_indices'][0]
        if index >= len(geometry):
            continue
        row = geometry[index]
        point = np.array([row.get(k, np.nan) for k in ('x', 'y', 'z')], float)
        if pair.get('status') == 'available' and np.isfinite(point).all():
            points[pair['id']] = point
    if not points:
        return [p['id'] for p in sorted(record['pairs'], key=lambda p: p['id'])[:count]]
    center = np.mean(list(points.values()), axis=0)
    selected = [min(points, key=lambda k: (-float(np.linalg.norm(points[k]-center)), k))]
    while len(selected) < min(count, len(points)):
        selected.append(min((k for k in points if k not in selected),
            key=lambda k: (-min(float(np.linalg.norm(points[k]-points[s])) for s in selected), k)))
    return selected


def plot_priority_plan(cfg, out, plan):
    """Freeze a bounded plotting panel without modifying the full numerical plan."""
    directory = Path(out)/'plot_priority_v1'; directory.mkdir(exist_ok=True)
    target = directory/'plan.json'
    if target.exists():
        return read_json(target)
    chosen = {r['id']: _plot_pairs(r) for r in plan['records']}
    cells = defaultdict(list); excluded = []; seen = set()
    for record, group in current_dof_groups(plan):
        if group['pair'] not in chosen[record['id']]:
            continue
        windows = [record['windows'][t['window_index']] for t in group['tasks']]
        if not any(w.get('hb_support_admitted', True) for w in windows):
            excluded.append(dict(group_id=group['id'], record_id=record['id'], pair=group['pair'],
                planned_windows=len(windows), reason='no_complete_native_Hb_support'))
            continue
        labels = _plot_task_labels(record, windows)
        # A native support can contain several task labels. It remains one fit;
        # all labels and window conditions are retained for the coverage table.
        cell = (record['dataset'], record['subject'], '+'.join(labels))
        item = {k: v for k, v in group.items() if k != 'tasks'}
        item.update(task_label=cell[2], task_labels=labels,
            conditions=sorted({w['condition'] for w in windows if w.get('condition')}),
            pair_rank=chosen[record['id']].index(group['pair']),
            planned_windows=len(windows),
            paired_windows=sum(w.get('status') == 'available' for w in windows),
            window_ids=[t['id'] for t in group['tasks']])
        cells[cell].append(item); seen.add(record['id'])
    for items in cells.values():
        items.sort(key=lambda x: (x['pair_rank'], x['record_id'], x['native_support_id']))
    keys = {ds: sorted((k for k in cells if k[0] == ds),
        key=lambda k: (token(str(cfg['seed'])+'|'+k[1]), k[2])) for ds in cfg['data']['datasets']}
    ordered = []
    for round_index in range(max(map(len, cells.values()), default=0)):
        queues = {ds: [cells[k][round_index] for k in keys[ds] if round_index < len(cells[k])]
                  for ds in cfg['data']['datasets']}
        for offset in range(max(map(len, queues.values()), default=0)):
            for ds in cfg['data']['datasets']:
                if offset < len(queues[ds]):
                    ordered.append(dict(queues[ds][offset], coverage_round=round_index))
    result = dict(schema='tak_mismatch_plot_priority_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        source_plan=str(Path(out)/'plan.json'), numerical_source=str(Path(out)/'source_manifest.json'),
        selection='all_subjects_all_native_records_all_continuous_Hb_supports_three_geometry_farthest_pairs',
        ordering='first_one_group_per_subject_task_cell_then_additional_supports_and_pairs_dataset_interleaved',
        task_semantics='REFED videos separate; other native task labels; all parts/probes/sessions retained',
        selection_uses_fit_results=False, unchanged_group_windows=True, pairs_per_record=3,
        groups=ordered, planned_groups=len(ordered), planned_windows=sum(g['planned_windows'] for g in ordered),
        planned_cells=len(cells), first_coverage_groups=sum(g['coverage_round']==0 for g in ordered),
        excluded_support_groups=excluded,
        excluded_records=[dict(record_id=r['id'], dataset=r['dataset'], subject=r['subject'],
            reason=r.get('excluded_reason', 'no_admitted_windows')) for r in plan['records'] if r['id'] not in seen],
        by_dataset={ds: dict(groups=sum(g['dataset']==ds for g in ordered),
            cells=len(keys[ds]), subjects=len({k[1] for k in keys[ds]}),
            records=len({g['record_id'] for g in ordered if g['dataset']==ds}),
            windows=sum(g['planned_windows'] for g in ordered if g['dataset']==ds)) for ds in keys})
    write_json(target, result)
    return result


def plot_priority_stage(cfg, out, plan, workers):
    """Run only the frozen plotting panel using the unchanged group fitter."""
    directory = Path(out)/'plot_priority_v1'; schedule = read_json(directory/'plan.json')
    ids = {g['id'] for g in schedule['groups']}
    payloads = {g['id']: (cfg, str(out), r, g) for r, g in current_dof_groups(plan) if g['id'] in ids}
    if set(payloads) != ids:
        raise ValueError('plot panel differs from the full frozen numerical identities')
    existing = []; pending = []
    for g in schedule['groups']:
        p = Path(out)/'current_dof'/'tasks'/f'{g["id"]}.json'
        if p.exists():
            saved = read_json(p)
            if saved.get('id') != g['id'] or len(saved['rows']) != g['planned_windows']:
                raise ValueError('retained group identity/denominator mismatch')
            existing.append(g['id'])
        else:
            pending.append(g)
    started = time.monotonic(); finished = set(existing); samples = defaultdict(list)
    def progress(count, result):
        if result:
            finished.add(result['group_id'])
            value = read_json(Path(out)/'current_dof'/'tasks'/f'{result["group_id"]}.json')
            if value.get('views'):
                samples[value['dataset']].append(dict(seconds=value['elapsed_seconds'], windows=len(value['rows'])))
        elapsed = time.monotonic()-started
        cells = defaultdict(list)
        for g in schedule['groups']:
            cells[(g['dataset'], g['subject'], g['task_label'])].append(g)
        write_json(directory/'progress.json', dict(stage='plot_priority_shared_parameters',
            updated_utc=datetime.now(timezone.utc).isoformat(), started_utc=datetime.fromtimestamp(time.time()-elapsed, timezone.utc).isoformat(),
            planned=schedule['planned_groups'], terminal=len(finished), reused=len(existing), new_terminal=count,
            first_coverage_planned=schedule['first_coverage_groups'],
            first_coverage_terminal=sum(g['id'] in finished for g in schedule['groups'] if g['coverage_round']==0),
            cells_with_terminal=sum(any(g['id'] in finished for g in gg) for gg in cells.values()),
            planned_cells=len(cells), elapsed_seconds=elapsed, workers=workers,
            timing_samples=dict(samples), by_dataset={ds: dict(planned=v['groups'],
                terminal=sum(g['id'] in finished for g in schedule['groups'] if g['dataset']==ds))
                for ds,v in schedule['by_dataset'].items()}))
    progress(0, None)
    results = bounded_work(_current_group_job, (payloads[g['id']] for g in pending), workers, progress=progress)
    result = dict(status='completed_plot_panel', planned_groups=len(ids), terminal_groups=len(finished),
        reused_groups=len(existing), new_status_counts=dict(Counter(r['status'] for r in results)),
        elapsed_seconds=time.monotonic()-started, full_campaign_completed=False)
    write_json(directory/'summary.json', result)
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


PHYSIOLOGY_FIELDS = tuple(dict.fromkeys((
    'id', 'dataset', 'subject', 'record_id', 'record', 'record_unit', 'record_order',
    'record_order_source', 'repeat_id', 'repeat_kind', 'task_label', 'task_label_source',
    'task', 'condition', 'native_support_id', 'native_support_admitted', 'native_support_start_s',
    'native_support_end_s', 'window', 'window_index', 'pair', 'pair_index', 'fold',
    'parameter_mode', 'panel_selected', 'window_start_s', 'window_start_clock', 'start_s',
    'end_s', 'hb_start_s', 'eeg_start_s', 'eeg_filter_start_s', 'eeg_filter_end_s',
    'actual_context_s', 'segment_index', 'hb_segment_index', 'window_status', 'window_reason',
    'hb_support_admitted', 'hb_score_start_s', 'hb_score_stop_s', 'eeg_score_start_s',
    'eeg_score_stop_s', 'score_start_relative_s', 'score_stop_relative_s', 'score_last_sample_relative_s',
    'status', 'success', 'fixed_success', 'score_status', 'hb_only_success', 'eeg_only_success',
    'joint_success', 'hb_only_status', 'eeg_only_status', 'joint_status',
    'hb_only_solver_status', 'eeg_only_solver_status', 'joint_solver_status',
    'spatial_success', 'spatial_status', 'wrong_success', 'wrong_status',
    'selected_wrong_success', 'selected_wrong_status', 'eeg_input_status', 'selected_candidate',
    'axis_3_selection_status', 'D_obs_status', 'drive_gap_status', 'observable_numerical_rank',
    'observable_retained_rank', 'hb_time_count', 'hb_observation_count', 'eeg_count',
    'array_path', 'weak_path', 'weak_pair_index', 'weak_metric_source', 'error', *METRICS)))
PHYSIOLOGY_WEAK_FIELDS = ('dataset', 'subject', 'record_id', 'record', 'task_label', 'task_label_source',
    'pair', 'pair_index', 'width_s', 'D_bal', 'status', 'weak_path', 'weak_pair_index',
    'record_unit', 'record_order', 'record_order_source', 'repeat_id', 'repeat_kind',
    'planned_centers', 'valid_centers', 'denominator', 'mean_denominator', 'excitation', 'native_support_id')


def read_physiology_config(path=DEFAULT_PHYSIOLOGY_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    contracts = {'tak_mismatch_fixed_physiology_v1': DEFAULT_PHYSIOLOGY_CONFIG,
                 'tak_mismatch_visual_physiology_v2': VISUAL_PHYSIOLOGY_CONFIG}
    if cfg.get('schema') not in contracts:
        raise ValueError('Unknown fixed-physiology version')
    expected = yaml.safe_load(contracts[cfg['schema']].read_text())
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f'versioned fixed-physiology contract drift: {key}')
    return cfg


def _physiology_directory(cfg, parent, project_root):
    parent = Path(parent).resolve()
    if parent != (Path(project_root)/cfg['parent_run']).resolve():
        raise ValueError('fixed physiology reads only its exact retained parent run')
    directory = (parent/cfg['analysis_subdir']).resolve()
    if directory.parent != parent or directory.name not in ('fixed_physiology_v1', 'fixed_physiology_v2'):
        raise ValueError('fixed physiology writes only its own analysis directory')
    return directory


def _physiology_plan(cfg, parent):
    plan = read_json(Path(parent)/'plan.json')
    panel = read_json(Path(parent)/cfg['scope']['array_source'])
    planned = sum(len(r.get('windows', []))*len(r.get('pairs', [])) for r in plan['records'])
    panel_ids = {identity for group in panel['groups'] for identity in group['window_ids']}
    if planned != cfg['scope']['scalar_pair_windows'] or len(panel_ids) != cfg['scope']['array_pair_windows']:
        raise ValueError('retained scalar inventory or frozen array-panel denominator changed')
    return plan, panel, panel_ids


def _physiology_record_metadata(record):
    name, dataset = record['record'], record['dataset']
    result = dict(record_unit=record['id'], repeat_id=record['id'], record_order=0,
        record_order_source='not_applicable_single_record_per_task', repeat_kind='unavailable_native_repeat')
    if dataset == 'eeg_fnirs_single_trial':
        match = re.fullmatch(r'session_(\d+)', name)
        result.update(record_order=int(match.group(1)) if match else None,
            record_order_source='declared_native_session_index', repeat_kind='same_task_native_record')
    elif dataset == 'visual_cognitive_motivation':
        unit = re.sub(r'_Probe\d+$', '', record['id'], flags=re.IGNORECASE)
        match = re.search(r'_Part(\d+)(?:_|$)', name, flags=re.IGNORECASE)
        result.update(record_unit=unit, repeat_id=unit, record_order=int(match.group(1)) if match else None,
            record_order_source='declared_native_part_index', repeat_kind='same_visit_cross_part')
    elif dataset == 'refed':
        result['repeat_kind'] = 'unavailable_different_video_stimuli'
    return result


def _physiology_task_label(record, window=None):
    window = window or {}
    if record['dataset'] == 'refed':
        return 'emotion_video/'+record['record'], 'native_video_identity'
    if record['dataset'] == 'visual_cognitive_motivation':
        return 'visual_cognitive_motivation', 'native_record_single_task_no_condition_recovery'
    if record['dataset'] == 'simultaneous_eeg_nirs':
        return record['record'].removeprefix('cnt_'), 'native_record_task_identity'
    if window.get('task'):
        return window['task'], 'native_window_task'
    labels = {w['task'] for w in record.get('windows', []) if w.get('task')}
    if len(labels) == 1:
        return next(iter(labels)), 'record_unique_native_task_for_Hb_only_window'
    return 'unlabeled_record/'+record['record'], 'unavailable_unambiguous_native_task'


def _physiology_write_csv(path, rows, fields):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with gzip.open(temporary, 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader()
        for row in rows:
            writer.writerow({k: '' if v is None else v for k, v in serial(row).items() if k in fields})
    temporary.replace(path)


def _physiology_merge_csv(paths, output, fields):
    def rows():
        for path in paths:
            with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
                yield from csv.DictReader(stream)
    _physiology_write_csv(output, rows(), fields)


def _physiology_extract_record(payload):
    parent, directory, record, panel_ids = payload
    parent, directory = Path(parent), Path(directory)
    stem = directory/'fixed_records'/token(record['id'])
    target, summary_path = stem.with_suffix('.csv.gz'), stem.with_suffix('.json')
    weak_target = stem.with_suffix('.weak.csv.gz')
    planned = len(record.get('windows', []))*len(record.get('pairs', []))
    if summary_path.exists() and target.exists() and weak_target.exists():
        retained = read_json(summary_path)
        if retained.get('record_id') != record['id'] or retained.get('planned') != planned:
            raise ValueError('fixed extraction record identity changed')
        if retained.get('status') == 'completed': return retained
    metadata_path, _, _ = prepared_paths(parent, record['id'])
    prepared = read_json(metadata_path) if metadata_path.exists() else {}
    record_meta = _physiology_record_metadata(record)
    supports = {s['id']: s for s in record.get('continuous_hb_segments', [])}
    rows = []; statuses = Counter(); missing = unreadable = 0
    for task in task_manifest(dict(records=[record])):
        window = record['windows'][task['window_index']]
        path = parent/'tasks'/f'{task["id"]}.json'
        try:
            retained = read_json(path) if path.exists() else dict(status='missing', success=False)
            if not isinstance(retained, dict): raise ValueError('terminal JSON is not an object')
        except (OSError, ValueError) as exc:
            retained = dict(status='unreadable_terminal_json', success=False, error=str(exc)); unreadable += 1
        if any(retained.get(k, task[k]) != task[k] for k in ('id', 'record_id', 'window', 'pair', 'parameter_mode')):
            raise ValueError(f'fixed terminal identity changed: {task["id"]}')
        if retained.get('status') == 'missing': missing += 1
        label, label_source = _physiology_task_label(record, window)
        support = supports.get(window.get('native_support_id'), {})
        row = dict(task, **record_meta)
        row.update({k: v for k, v in retained.items() if k in PHYSIOLOGY_FIELDS})
        row.update(task_label=label, task_label_source=label_source, task=window.get('task'),
            condition=window.get('condition'), parameter_mode='fixed', panel_selected=task['id'] in panel_ids,
            native_support_id=window.get('native_support_id') or record['id']+'|unavailable_native_support',
            native_support_admitted=bool(window.get('hb_support_admitted', False)),
            native_support_start_s=support.get('start_s'), native_support_end_s=support.get('end_s'),
            window_start_s=task['hb_start_s'], window_start_clock='native_cached_Hb_clock',
            window_status=window.get('status'), window_reason=window.get('reason'),
            fixed_success=bool(retained.get('success', False)),
            score_start_relative_s=5., score_stop_relative_s=29.9, score_last_sample_relative_s=29.8,
            hb_score_start_s=task['hb_start_s']+5., hb_score_stop_s=task['hb_start_s']+29.9,
            eeg_score_start_s=task['eeg_start_s']+5. if task['eeg_start_s'] is not None else None,
            eeg_score_stop_s=task['eeg_start_s']+29.9 if task['eeg_start_s'] is not None else None,
            weak_path=retained.get('weak_path', prepared.get('weak_path')), weak_pair_index=task['pair_index'],
            weak_metric_source='prepared_complete_native_record_pair_summary')
        for key in ('eeg_filter_start_s', 'eeg_filter_end_s', 'segment_index', 'hb_segment_index', 'hb_support_admitted'):
            row[key] = window.get(key)
        for view in ('hb_only', 'eeg_only', 'joint'):
            value = retained.get('view_statuses', {}).get(view, {})
            row[view+'_success'] = bool(retained.get(view+'_success', value.get('success', False)))
            row[view+'_status'] = value.get('status', 'unavailable')
            row[view+'_solver_status'] = value.get('solver_status')
        for name, value in (('spatial', retained.get('spatial_status')), *retained.get('wrong_statuses', {}).items()):
            row[name+'_success'] = bool(value.get('success', False)) if isinstance(value, dict) else False
            row[name+'_status'] = value.get('status') if isinstance(value, dict) else value
        for width, weak in prepared.get('weak_summary', {}).get(str(task['pair_index']), {}).items():
            row['D_bal_'+width] = weak.get('D_bal')
        statuses[row['status']] += 1; rows.append(row)
    weak_rows = []
    label, source = _physiology_task_label(record)
    for j, pair in enumerate(record.get('pairs', [])):
        for width in (10., 20., 40.):
            weak = prepared.get('weak_summary', {}).get(str(j), {}).get(f'{width:g}s', {})
            weak_rows.append(dict(dataset=record['dataset'], subject=record['subject'], record_id=record['id'],
                record=record['record'], task_label=label, task_label_source=source, pair=pair['id'], pair_index=j,
                width_s=width, D_bal=weak.get('D_bal'), status=weak.get('status', 'unavailable_prepared_summary'),
                weak_path=prepared.get('weak_path'), weak_pair_index=j, **record_meta,
                planned_centers=weak.get('planned_centers', 0), valid_centers=weak.get('valid_centers', 0),
                denominator=weak.get('denominator'), mean_denominator=weak.get('mean_denominator'),
                excitation=weak.get('excitation'), native_support_id=record['id']+'|complete_record_weak_summary'))
    _physiology_write_csv(target, rows, PHYSIOLOGY_FIELDS)
    _physiology_write_csv(weak_target, weak_rows, PHYSIOLOGY_WEAK_FIELDS)
    result = dict(schema='tak_mismatch_fixed_record_extract_v1', status='completed' if not missing and not unreadable else 'incomplete',
        record_id=record['id'], dataset=record['dataset'], subject=record['subject'], planned=planned, rows=len(rows),
        missing=missing, unreadable=unreadable, statuses=dict(statuses), panel_selected=sum(r['panel_selected'] for r in rows),
        files=dict(metrics=str(target), weak=str(weak_target)), measured_arrays_read=0, model_fits=0)
    write_json(summary_path, result); return result


def physiology_extract_stage(cfg, parent, directory, plan, panel_ids, workers, *, max_records=None):
    records = list(interleaved_records(dict(data=dict(datasets=plan['datasets'])), plan['records']))
    pending = []
    for record in records:
        stem = Path(directory)/'fixed_records'/token(record['id'])
        summary = stem.with_suffix('.json')
        if (not summary.exists() or read_json(summary).get('status') != 'completed'
                or not stem.with_suffix('.csv.gz').exists() or not stem.with_suffix('.weak.csv.gz').exists()):
            pending.append(record)
    selected = pending[:max_records] if max_records is not None else pending
    started = time.monotonic()
    bounded_work(_physiology_extract_record,
        ((str(parent), str(directory), r, panel_ids) for r in selected), workers,
        progress=_progress(directory, 'physiology-extract', len(selected), started))
    results = [read_json(Path(directory)/'fixed_records'/f'{token(r["id"])}.json') for r in records
               if (Path(directory)/'fixed_records'/f'{token(r["id"])}.json').exists()]
    complete = len(results) == len(records) and all(r['status'] == 'completed' for r in results)
    files = {}
    if complete:
        if sum(r['rows'] for r in results) != cfg['scope']['scalar_pair_windows'] or sum(r['panel_selected'] for r in results) != len(panel_ids):
            raise ValueError('fixed extraction denominator disagrees with complete parent manifest')
        files = dict(windows='fixed_window_metrics.csv.gz', weak='weak_record_metrics.csv.gz')
        for key, fields in (('metrics', PHYSIOLOGY_FIELDS), ('weak', PHYSIOLOGY_WEAK_FIELDS)):
            output = files['windows' if key == 'metrics' else 'weak']
            _physiology_merge_csv([r['files'][key] for r in results], Path(directory)/output, fields)
    result = dict(schema='tak_mismatch_fixed_physiology_extract_v1', status='completed' if complete else 'incomplete',
        planned_records=len(records), terminal_records=len(results), new_records=len(selected),
        planned=cfg['scope']['scalar_pair_windows'], rows=sum(r['rows'] for r in results),
        panel_planned=len(panel_ids), panel_selected=sum(r['panel_selected'] for r in results),
        missing=sum(r['missing'] for r in results), unreadable=sum(r['unreadable'] for r in results),
        statuses=dict(sum((Counter(r['statuses']) for r in results), Counter())), files=files,
        workers=workers, elapsed_seconds=time.monotonic()-started, measured_arrays_read=0, model_fits=0)
    write_json(Path(directory)/'extraction_summary.json', result); return result


def _physiology_read_frame(path):
    import pandas as pd
    identities = ('id', 'dataset', 'subject', 'record_id', 'record', 'task_label', 'pair',
                  'window', 'native_support_id', 'record_unit', 'repeat_id', 'repeat_kind')
    return pd.read_csv(path, dtype={key: str for key in identities}, low_memory=False)


def _physiology_save_frame(directory, filename, frame):
    path = Path(directory)/filename
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    frame.to_csv(temporary, index=False, compression='gzip' if filename.endswith('.gz') else None)
    temporary.replace(path)
    return filename


def _physiology_examples(frame, cfg):
    examples = []
    panel = frame[frame.panel_selected.eq(True)].copy()
    for dataset, planned in panel.groupby('dataset', sort=True):
        used = set()
        for metric, roles in (('D1', (('max', 1.), ('median', .5), ('p90', .9))),
                              ('D_drive_H', (('drive_incompatible_p90', cfg['examples']['D_drive_H_quantile']),))):
            candidates = planned[np.isfinite(planned[metric].to_numpy(float)) & planned.array_path.notna()].copy()
            candidates = candidates.sort_values([metric, 'id'])
            if candidates.empty: continue
            values = candidates[metric].to_numpy(float)
            for role, quantile in roles:
                target = float(np.quantile(values, quantile))
                eligible = candidates[~candidates.subject.isin(used)]
                if eligible.empty: eligible = candidates
                distances = abs(eligible[metric]-target)
                chosen = eligible.assign(_distance=distances).sort_values(['_distance', 'id']).iloc[0]
                used.add(chosen.subject)
                row = {k: v for k, v in chosen.to_dict().items() if k != '_distance'}
                row.update(role=role, selection_metric=metric, selection_quantile=quantile,
                    selection_target=target, candidate_denominator=len(candidates),
                    planned_panel_denominator=len(planned), unavailable_panel_count=len(planned)-len(candidates),
                    selection_rank=int(np.flatnonzero(candidates.id.to_numpy() == chosen.id)[0])+1,
                    interpretation='descriptive_example_selection_not_tuning_or_independent_evidence')
                examples.append(row)
    order = {'median':0, 'p90':1, 'max':2, 'drive_incompatible_p90':3}
    examples.sort(key=lambda row:(row['dataset'],order[row['role']]))
    return dict(schema='tak_mismatch_fixed_physiology_examples_v1', examples=examples,
        selection='frozen_panel_D1_median_p90_max_and_D_drive_H_p90_distinct_subject_when_possible')


def _physiology_summary(cfg, parent, directory):
    stages = dict(extraction='extraction_summary.json', statistics='statistical_summary.json',
        ocular='ocular_stage_summary.json', counterfactual='counterfactual_summary.json', shapes='shapes_summary.json')
    retained = {stage: read_json(Path(directory)/name) for stage, name in stages.items()
                if (Path(directory)/name).exists()}
    statuses = {stage: retained.get(stage, {}).get('status', 'pending') for stage in stages}
    files = {key: value for record in retained.values() for key, value in record.get('files', {}).items()}
    fixed = read_json(Path(parent)/'resolved_config.json').get('fixed', {})
    result = dict(schema='tak_mismatch_fixed_physiology_analysis_v1',
        experiment_id=cfg['experiment_id'], created_utc=datetime.now(timezone.utc).isoformat(),
        status='completed' if all(status == 'completed' for status in statuses.values()) else 'incomplete',
        parameter_mode='fixed', fixed_parameters=fixed, parent_run=cfg['parent_run'],
        interpretation=cfg['interpretation'], stages=statuses, files=files,
        scope_description=f"全部{cfg['scope']['scalar_pair_windows']}个固定参数窗口×通道对用于标量分析；波形限定于预先冻结的{cfg['scope']['array_pair_windows']}个窗口×通道对三通道面板；EOG仅在声明的原生参照数据集计算。不重新拟合模型。",
        statistics=cfg['statistics'], examples=cfg['examples'],
        repeatability_scope=dict(same_task_native_record='同任务跨原生记录重复性；不宣称跨日',
            same_visit_cross_part='同次访问跨Part一致性；同步Probe不构成重复',
            within_record_early_late='记录内早晚时间一致性；窗口不充当独立被试',
            unavailable_different_video_stimuli='不同视频是不同刺激，不能估计原生记录重测'),
        report_findings=[], model_fits=0, protected_data='forbidden',
        claim_limits=['探索性冻结模型条件分析，不是随机任务因果效应或生理真值恢复',
            'Hb残差与零预测MSE之比不是能量分解或R²，可大于1',
            'EOG指标反映辅助通道活动关联，不识别脑源、眨眼真值或伪迹原因'])
    write_json(Path(directory)/'summary.json', result); return result


def _physiology_view_coverage(frame):
    """Reuse the terminal-audit classifier on the retained flattened verdicts."""
    result = {}
    columns = ['dataset', 'status', 'eeg_input_status']
    for view in ('hb_only', 'eeg_only', 'joint'):
        columns.extend((view+'_status', view+'_solver_status', view+'_success'))
    for dataset, group in frame[columns].groupby('dataset', sort=True):
        views = {view: _audit_bucket() for view in ('hb_only', 'eeg_only', 'joint')}
        for flat in group.to_dict('records'):
            row = dict(status=flat['status'], eeg_input_status=flat['eeg_input_status'], view_statuses={})
            for view in views:
                status = flat[view+'_status']
                if isinstance(status, str) and status != 'unavailable':
                    row['view_statuses'][view] = dict(status=status, solver_status=flat[view+'_solver_status'],
                        success=flat[view+'_success'] in (True, 1, 'True', 'true'))
                if flat['status'] == 'missing':
                    category, status, solver, reason = 'missing', None, None, 'missing_retained_terminal'
                else:
                    category, status, solver, reason = _audit_fixed_view(row, view)
                _audit_count(views[view], category, task_status=flat['status'], view_status=status,
                    solver_status=solver, reason=reason)
        result[dataset] = {view: _audit_finish_bucket(bucket) for view, bucket in views.items()}
    return dict(schema='tak_mismatch_fixed_physiology_view_coverage_v1', by_dataset=result,
        unit='planned_fixed_window_pair', classification_owner='_audit_fixed_view',
        interpretation='support_absence_numerical_nonconvergence_and_execution_failure_keep_separate_denominators')


def _physiology_task_statistics(payload):
    from src.analysis.tak_mismatch import paired_task_comparisons
    frame, options = payload
    return paired_task_comparisons(frame, **options)


def _physiology_within_statistics(payload):
    from src.analysis.tak_mismatch import within_record_consistency
    frame, options = payload
    return within_record_consistency(frame, **options)


def physiology_analyze_stage(cfg, parent, directory, workers=1):
    import pandas as pd
    from src.analysis import tak_mismatch as analysis
    extraction = read_json(Path(directory)/'extraction_summary.json')
    if extraction['status'] != 'completed': raise ValueError('complete fixed extraction required before statistics')
    frame = _physiology_read_frame(Path(directory)/extraction['files']['windows'])
    if len(frame) != cfg['scope']['scalar_pair_windows']:
        raise ValueError('fixed statistical input denominator changed')
    metrics = tuple(dict.fromkeys((*cfg['statistics']['primary_metrics'], *cfg['statistics']['diagnostic_metrics'])))
    tables = analysis.aggregate_fixed_metrics(frame, metrics=metrics)
    for table in tables.values(): table['observational_unit'] = 'nonoverlapping_30s_window_pair'
    # The weak metric is one retained complete-record/pair quantity.  Its
    # repeated aliases in the window CSV never define its sample denominator.
    weak = _physiology_read_frame(Path(directory)/extraction['files']['weak'])
    for width in (10., 20., 40.):
        name = f'D_bal_{width:g}s'
        source = weak[weak.width_s.eq(width)].copy()
        source[name] = source.D_bal
        source['window_start_s'] = 0.  # One record-pair observation, not a sampled time point.
        source['parameter_mode'] = 'fixed'
        current = analysis.aggregate_fixed_metrics(source, metrics=(name,), include_hb_ratio=False)
        for key, table in current.items():
            table['observational_unit'] = 'native_record_pair'
            tables[key] = pd.concat([tables[key], table], ignore_index=True)
    options = {key: cfg['statistics'][key] for key in ('bootstrap_samples', 'seed')}
    tables['subject_descriptives'] = analysis.summarize_subject_distributions(tables['subjects'], **options)
    tables['task_descriptives'] = analysis.summarize_subject_distributions(tables['subject_tasks'], by=('dataset', 'task_label'), **options)
    known = tables['subject_tasks'][~tables['subject_tasks'].task_label.str.startswith('unlabeled_record/')]
    task_options = dict(options, permutation_samples=cfg['statistics']['permutation_samples'],
        primary_metrics=cfg['statistics']['primary_metrics'])
    task_groups = [group for _,group in known.groupby('dataset', sort=True)]
    chunks = bounded_work(_physiology_task_statistics, ((group, task_options) for group in task_groups),
        min(workers, max(1,len(task_groups))))
    comparisons = pd.concat(chunks, ignore_index=True)
    # Dataset jobs produce p values independently; the correction family still
    # spans every dataset/task pair/primary endpoint in the complete report.
    for family in ('primary', 'diagnostic'):
        take = comparisons.fdr_family.eq(family)
        values = comparisons.loc[take,'p_value'].to_numpy(float)
        comparisons.loc[take,'q_value'] = analysis.benjamini_hochberg(values)
        comparisons.loc[take,'fdr_family_size'] = int(np.isfinite(values).sum())
    tables['task_comparisons'] = comparisons
    stability_metrics = (*analysis.DEFAULT_METRICS, analysis.HB_RATIO)
    repeat = analysis.native_record_repeatability(tables['native_pairs'][tables['native_pairs'].metric.isin(stability_metrics)], **options)
    columns = [*analysis.DEFAULT_METRICS, 'zero_NMSE', 'dataset', 'subject', 'task_label',
        'record_id', 'native_support_id', 'pair', 'window_start_s', 'parameter_mode']
    within_groups = [group for _,group in frame[columns].groupby('dataset', sort=True)]
    within_chunks = bounded_work(_physiology_within_statistics, ((group, options) for group in within_groups),
        min(workers,max(1,len(within_groups))))
    within = {key: pd.concat([chunk[key] for chunk in within_chunks], ignore_index=True)
              for key in ('pairs','summary','records')}
    tables.update(repeat_pairs=repeat['pairs'], repeat_summary=repeat['summary'],
        within_record_pairs=within['pairs'], within_record_summary=within['summary'], within_record_records=within['records'])
    files = {}
    for key, table in tables.items():
        filename = key+('.csv.gz' if key in ('native_pairs', 'native_records', 'within_record_records') else '.csv')
        files[key] = _physiology_save_frame(directory, filename, table)
    examples = _physiology_examples(frame, cfg)
    write_json(Path(directory)/'examples.json', examples); files['examples'] = 'examples.json'
    coverage = _physiology_view_coverage(frame)
    write_json(Path(directory)/'view_coverage.json', coverage); files['view_coverage'] = 'view_coverage.json'
    result = dict(schema='tak_mismatch_fixed_physiology_statistics_v1', status='completed', files=files,
        rows=len(frame), subjects=len(frame[['dataset', 'subject']].drop_duplicates()),
        planned_records=frame.record_id.nunique(), primary_metrics=cfg['statistics']['primary_metrics'],
        task_comparisons=len(tables['task_comparisons']), model_fits=0, measured_arrays_read=0)
    write_json(Path(directory)/'statistical_summary.json', result)
    _physiology_summary(cfg, parent, directory)
    return result


def _physiology_record_jobs(stage, directory, records, worker, payload, workers, max_records):
    """Shared bounded record scheduling for the three read-only array stages."""
    directory = Path(directory); started = time.monotonic()
    pending = [record for record in records
               if not (directory/(stage+'_records')/f'{token(record["id"])}.json').exists()]
    selected = pending[:max_records] if max_records is not None else pending
    bounded_work(worker, (payload(record) for record in selected), workers,
        progress=_progress(directory, 'physiology-'+stage, len(selected), started))
    results = [read_json(directory/(stage+'_records')/f'{token(record["id"])}.json')
               for record in records if (directory/(stage+'_records')/f'{token(record["id"])}.json').exists()]
    complete = len(results) == len(records) and all(r['status'] == 'completed' for r in results)
    return results, dict(status='completed' if complete else 'incomplete', planned_records=len(records),
        terminal_records=len(results), new_records=len(selected), workers=workers,
        elapsed_seconds=time.monotonic()-started, model_fits=0)


def _physiology_panel_tasks(record, panel_ids):
    return [task for task in task_manifest(dict(records=[record])) if task['id'] in panel_ids]


def _physiology_ocular_record(payload):
    parent_cfg, root, directory, record, selected_tasks = payload
    from src.data.tak_mismatch_ocular import prepare_tak_ocular_record
    directory = Path(directory); stem = directory/'ocular_records'/token(record['id'])
    selected_windows = {t['window'] for t in selected_tasks}
    indices = [i for i,w in enumerate(record['windows']) if w['id'] in selected_windows]
    result = dict(schema='tak_mismatch_fixed_ocular_record_v1', record_id=record['id'],
        dataset=record['dataset'], subject=record['subject'], status='completed', files={},
        selected_windows=len(indices), measured_signal_reads=0, available_windows=0)
    arrays = {}; metadata = dict(status='unavailable', reason='no_registered_native_EOG_auxiliary')
    try:
        if record['dataset'] in ('eeg_fnirs_single_trial', 'simultaneous_eeg_nirs'):
            output = prepare_tak_ocular_record(root, record, cache_root=parent_cfg['data']['cache_root'],
                index=_metadata_index(str(root), parent_cfg['data']['cache_root']))
            metadata = output.pop('metadata')
            original_count = len(record['windows'])
            for key, value in output.items():
                arrays[key] = value[indices] if key != 'time_s' and np.ndim(value) and np.shape(value)[0] == original_count else value
            arrays['window_indices'] = np.asarray(indices, np.int64)
            array_path = stem.with_suffix('.npz'); write_npz(array_path, **arrays)
            result['files']['arrays'] = str(array_path)
            result['measured_signal_reads'] = metadata.get('measured_signal_reads', 0)
            result['available_windows'] = int(np.asarray(arrays['complete']).all(axis=1).sum())
        result['metadata'] = metadata
    except Exception as exc:
        result.update(status='failed_exception', error=str(exc), traceback=traceback.format_exc(), metadata=metadata)
    rows = []
    for local, index in enumerate(indices):
        window = record['windows'][index]
        row = dict(dataset=record['dataset'], subject=record['subject'], record_id=record['id'],
            window=window['id'], window_index=index, ocular_status=metadata.get('status'),
            ocular_reason=metadata.get('reason'), eog_rms_combined=None,
            eog_HEOG_rms=None, eog_VEOG_rms=None, eog_HEOG_peak_abs=None, eog_VEOG_peak_abs=None,
            eog_HEOG_derivative_rms=None, eog_VEOG_derivative_rms=None)
        if arrays:
            row['eog_rms_combined'] = arrays['rms_combined'][local]
            for j, channel in enumerate(('HEOG', 'VEOG')):
                for field in ('rms', 'peak_abs', 'derivative_rms'):
                    row['eog_'+channel+'_'+field] = arrays[field][local, j]
            native = metadata.get('windows', [])[index]
            row.update(ocular_status=native.get('status'), ocular_reason=native.get('reason'))
        rows.append(row)
    fields = ('dataset', 'subject', 'record_id', 'window', 'window_index', 'ocular_status', 'ocular_reason',
        'eog_rms_combined', 'eog_HEOG_rms', 'eog_VEOG_rms', 'eog_HEOG_peak_abs', 'eog_VEOG_peak_abs',
        'eog_HEOG_derivative_rms', 'eog_VEOG_derivative_rms')
    path = stem.with_suffix('.csv.gz'); _physiology_write_csv(path, rows, fields)
    result.update(files=dict(result['files'], windows=str(path)), fields=fields)
    write_json(stem.with_suffix('.json'), result); return result


def physiology_ocular_stage(parent_cfg, cfg, root, parent, directory, plan, panel_ids, workers, *, max_records=None):
    records = list(interleaved_records(parent_cfg, plan['records']))
    payload = lambda record: (parent_cfg, str(root), str(directory), record, _physiology_panel_tasks(record, panel_ids))
    results, result = _physiology_record_jobs('ocular', directory, records, _physiology_ocular_record,
        payload, workers, max_records)
    result.update(schema='tak_mismatch_fixed_physiology_ocular_v1',
        measured_signal_reads=sum(r['measured_signal_reads'] for r in results),
        available_windows=sum(r['available_windows'] for r in results),
        planned_native_records=sum(r['dataset'] in cfg['scope']['ocular_datasets'] for r in records), files={})
    if result['status'] == 'completed':
        filename = 'ocular_window_metrics.csv.gz'
        _physiology_merge_csv([r['files']['windows'] for r in results], Path(directory)/filename, results[0]['fields'])
        result['files']['ocular_windows'] = filename
    write_json(Path(directory)/'ocular_stage_summary.json', result)
    _physiology_summary(cfg, parent, directory)
    return result


def _physiology_counterfactual_record(payload):
    parent_cfg, root, parent, directory, record, selected_tasks = payload
    from src.data.tak_mismatch_ocular import prepare_tak_ocular_counterfactual
    parent, directory = Path(parent), Path(directory)
    stem = directory/'counterfactual_records'/token(record['id'])
    result = dict(schema='tak_mismatch_fixed_counterfactual_record_v1', record_id=record['id'],
        dataset=record['dataset'], subject=record['subject'], status='completed', files={},
        measured_signal_reads=0, planned_window_pairs=len(selected_tasks), available_window_pairs=0)
    try:
        if not selected_tasks:
            result['metadata'] = dict(status='unavailable', reason='no_frozen_panel_pairs')
        else:
            selection = _read_shared_json(str(parent/'spatial_selection.json'))
            pair_ids = sorted({task['pair'] for task in selected_tasks})
            models = {pair['id']: selection['models'].get(_group_key(record, pair), {}).get(str(record['fold']), {}).get('coordinate', {})
                      for pair in record['pairs'] if pair['id'] in pair_ids}
            prepared_path = prepared_paths(parent, record['id'])[1]
            with np.load(prepared_path, allow_pickle=False) as retained:
                output = prepare_tak_ocular_counterfactual(root, record, models, pair_ids, prepared=retained,
                    cache_root=parent_cfg['data']['cache_root'], index=_metadata_index(str(root), parent_cfg['data']['cache_root']))
            metadata = output.pop('metadata')
            wanted = {task['window'] for task in selected_tasks}
            indices = [i for i,w in enumerate(record['windows']) if w['id'] in wanted]
            n = len(record['windows']); arrays = {}
            for key, value in output.items():
                arrays[key] = value[indices] if key not in ('time_s', 'pair_ids', 'frozen_pc', 'frozen_eeg_factor', 'channel_indices') and np.ndim(value) and np.shape(value)[0] == n else value
            arrays['window_indices'] = np.asarray(indices, np.int64)
            path = stem.with_suffix('.npz'); write_npz(path, **arrays)
            result.update(metadata=metadata, files=dict(arrays=str(path)),
                measured_signal_reads=metadata.get('measured_signal_reads', 0),
                available_window_pairs=sum(bool(arrays['complete'][indices.index(t['window_index']), pair_ids.index(t['pair'])]) for t in selected_tasks))
    except Exception as exc:
        result.update(status='failed_exception', error=str(exc), traceback=traceback.format_exc())
    write_json(stem.with_suffix('.json'), result); return result


def physiology_counterfactual_stage(parent_cfg, cfg, root, parent, directory, plan, panel_ids, workers, *, max_records=None):
    records = [r for r in interleaved_records(parent_cfg, plan['records']) if r['dataset'] in cfg['scope']['ocular_datasets']]
    if len(records) != cfg['scope']['ocular_native_records']:
        raise ValueError('native EOG record denominator changed')
    payload = lambda record: (parent_cfg, str(root), str(parent), str(directory), record, _physiology_panel_tasks(record, panel_ids))
    results, result = _physiology_record_jobs('counterfactual', directory, records, _physiology_counterfactual_record,
        payload, workers, max_records)
    result.update(schema='tak_mismatch_fixed_physiology_counterfactual_v1',
        measured_signal_reads=sum(r['measured_signal_reads'] for r in results),
        planned_window_pairs=sum(r['planned_window_pairs'] for r in results),
        available_window_pairs=sum(r['available_window_pairs'] for r in results), files={})
    write_json(Path(directory)/'counterfactual_summary.json', result)
    _physiology_summary(cfg, parent, directory)
    return result


def _physiology_read_aux(path):
    if not Path(path).exists(): return None
    with np.load(path, allow_pickle=False) as arrays:
        return {k: arrays[k] for k in arrays.files}


def _physiology_aux_window(arrays, window_index, pair=None):
    if arrays is None: return None
    positions = np.flatnonzero(np.asarray(arrays['window_indices']) == window_index)
    if len(positions) != 1: return None
    i = int(positions[0])
    if pair is None:
        return {key: arrays[key][i] for key in ('rms_combined', 'rms_10hz', 'valid_mask')}
    pairs = np.flatnonzero(np.asarray(arrays['pair_ids']).astype(str) == str(pair))
    if len(pairs) != 1: return None
    j = int(pairs[0])
    return {key: arrays[key][i, j] for key in ('raw_proxy', 'clean_proxy', 'amplitude_proxy')}


def _physiology_ocular_examples(panel, features, cfg, directory):
    merged = features.merge(panel, on='id', how='left', validate='one_to_one', suffixes=('_feature',''))
    required = ('eog_raw_DEH_mse', 'eog_clean_DEH_mse', 'eog_amplitude_DEH_mse', 'eog_rms')
    examples = []; selections = []
    for dataset in cfg['scope']['ocular_datasets']:
        planned = merged[merged.dataset.eq(dataset)].copy()
        valid = np.ones(len(planned),bool)
        for column in required:
            if column not in planned: valid[:] = False
            else: valid &= np.isfinite(planned[column].to_numpy(float))
        eligible = planned.loc[valid].copy()
        eligible['clean_minus_amplitude_DEH_mse'] = eligible.get('eog_clean_DEH_mse', np.nan)-eligible.get('eog_amplitude_DEH_mse', np.nan)
        used = set()
        roles = (('high_eog_p90','eog_rms',.9,None),
            ('improvement_p10','clean_minus_amplitude_DEH_mse',.1,'negative'),
            ('nonimprovement_median','clean_minus_amplitude_DEH_mse',.5,'nonnegative'))
        for role,metric,quantile,condition in roles:
            candidates = eligible
            if condition == 'negative': candidates = candidates[candidates[metric] < 0]
            elif condition == 'nonnegative': candidates = candidates[candidates[metric] >= 0]
            selection = dict(dataset=dataset, role=role, selection_metric=metric, selection_quantile=quantile,
                planned_panel_denominator=len(planned), valid_counterfactual_denominator=len(eligible),
                candidate_denominator=len(candidates), candidate_unit='frozen_window_pair',
                status='selected' if len(candidates) else 'unavailable_no_candidates')
            if candidates.empty:
                selections.append(selection); continue
            target = float(np.quantile(candidates[metric],quantile))
            distinct = candidates[~candidates.subject.isin(used)]
            if distinct.empty: distinct = candidates
            chosen = distinct.assign(_distance=abs(distinct[metric]-target)).sort_values(['_distance','id']).iloc[0]
            used.add(chosen.subject)
            stem = token(chosen.record_id)
            ocular_path = (Path(directory)/'ocular_records'/f'{stem}.npz').resolve()
            counter_path = (Path(directory)/'counterfactual_records'/f'{stem}.npz').resolve()
            ocular = _physiology_read_aux(ocular_path); counter = _physiology_read_aux(counter_path)
            if ocular is None or counter is None: raise ValueError('selected EOG example lacks retained auxiliary arrays')
            oi = np.flatnonzero(ocular['window_indices'] == chosen.window_index)
            ci = np.flatnonzero(counter['window_indices'] == chosen.window_index)
            pi = np.flatnonzero(counter['pair_ids'].astype(str) == str(chosen.pair))
            if any(len(index) != 1 for index in (oi,ci,pi)):
                raise ValueError('selected EOG example auxiliary identity is ambiguous')
            metadata = read_json(Path(directory)/'ocular_records'/f'{stem}.json').get('metadata',{})
            row = {k:v for k,v in chosen.to_dict().items() if k != '_distance'}
            row.update(selection, selection_target=target, fixed_array_path=chosen.array_path,
                ocular_array_path=str(ocular_path), counterfactual_array_path=str(counter_path),
                ocular_array_index=int(oi[0]), counterfactual_window_index=int(ci[0]),
                counterfactual_pair_index=int(pi[0]), source_units=metadata.get('native_unit','not_recorded'),
                interpretation='posthoc_descriptive_example_after_complete_overall_statistics_not_population_evidence')
            examples.append(row); selections.append(dict(selection, selection_target=target, selected_id=chosen.id))
    return dict(schema='tak_mismatch_fixed_ocular_examples_v1', examples=examples, selections=selections,
        selection='C_available_high_EOG_p90_negative_clean_minus_amplitude_p10_and_nonnegative_median',
        array_index_contract='window_index_is_original_native_identity; use_ocular_array_index_and_counterfactual_window_index_for_subset_arrays')


def _physiology_shape_record(payload):
    parent, directory, record, selected_tasks = payload
    from src.analysis.tak_mismatch import fixed_window_physiology_features
    parent, directory = Path(parent), Path(directory)
    stem = directory/'shape_records'/token(record['id'])
    frame = _physiology_read_frame(directory/'fixed_records'/f'{token(record["id"])}.csv.gz')
    selected_ids = {task['id'] for task in selected_tasks}
    frame = frame[frame.id.isin(selected_ids)]
    if len(frame) != len(selected_tasks): raise ValueError('frozen shape-panel identity changed')
    ocular = _physiology_read_aux(directory/'ocular_records'/f'{token(record["id"])}.npz')
    counter = _physiology_read_aux(directory/'counterfactual_records'/f'{token(record["id"])}.npz')
    rows = []; errors = 0; arrays_read = 0
    for row in frame.to_dict('records'):
        features = dict(id=row['id'], dataset=row['dataset'], subject=row['subject'], record_id=row['record_id'],
            pair=row['pair'], window=row['window'], window_index=row['window_index'], feature_status='unavailable_arrays')
        try:
            if isinstance(row.get('array_path'), str) and row['array_path']:
                path = Path(row['array_path']).resolve()
                if path != (parent/'tasks'/f'{row["id"]}.npz').resolve():
                    raise ValueError('shape stage reads only the exact retained fixed task array')
                with np.load(path, allow_pickle=False) as retained:
                    arrays_read += 1
                    features.update(fixed_window_physiology_features(retained, row,
                        ocular=_physiology_aux_window(ocular, row['window_index']),
                        counterfactual=_physiology_aux_window(counter, row['window_index'], row['pair'])))
        except Exception as exc:
            features.update(feature_status='failed_array_validation', error=str(exc)); errors += 1
        rows.append(features)
    import pandas as pd
    table = pd.DataFrame(rows)
    filename = stem.with_suffix('.csv.gz'); _physiology_save_frame(directory, str(filename.relative_to(directory)), table)
    result = dict(schema='tak_mismatch_fixed_shape_record_v1', status='completed' if not errors else 'incomplete',
        record_id=record['id'], rows=len(rows), arrays_read=arrays_read, errors=errors,
        files=dict(features=str(filename)))
    write_json(stem.with_suffix('.json'), result); return result


def physiology_shapes_stage(cfg, parent, directory, plan, panel_ids, workers, *, max_records=None):
    from src.analysis.tak_mismatch import analyze_fixed_physiology_features
    import pandas as pd
    for name in ('extraction', 'ocular', 'counterfactual'):
        summary = Path(directory)/(name+('_stage_summary.json' if name == 'ocular' else '_summary.json'))
        if not summary.exists() or read_json(summary)['status'] != 'completed':
            raise ValueError(f'complete {name} stage required before shapes')
    tasks = {r['id']: _physiology_panel_tasks(r, panel_ids) for r in plan['records']}
    records = [r for r in plan['records'] if tasks[r['id']]]
    payload = lambda record: (str(parent), str(directory), record, tasks[record['id']])
    results, result = _physiology_record_jobs('shape', directory, records, _physiology_shape_record,
        payload, workers, max_records)
    result.update(schema='tak_mismatch_fixed_physiology_shapes_v1', files={},
        planned=len(panel_ids), rows=sum(r['rows'] for r in results),
        arrays_read=sum(r['arrays_read'] for r in results), errors=sum(r['errors'] for r in results))
    if result['status'] == 'completed':
        features = pd.concat([_physiology_read_frame(r['files']['features']) for r in results], ignore_index=True)
        panel_frame = _physiology_read_frame(Path(directory)/'fixed_window_metrics.csv.gz')
        panel_frame = panel_frame[panel_frame.panel_selected.eq(True)]
        if len(features) != len(panel_ids) or set(features.id) != set(panel_frame.id):
            raise ValueError('fixed shape feature denominator changed')
        result['files']['shape_windows'] = _physiology_save_frame(directory, 'shape_window_features.csv.gz', features)
        analysis = analyze_fixed_physiology_features(panel_frame, features, config=cfg['statistics'])
        for key, value in analysis.items():
            filename = key+('.csv' if isinstance(value, pd.DataFrame) else '.json')
            if isinstance(value, pd.DataFrame): _physiology_save_frame(directory, filename, value)
            else: write_json(Path(directory)/filename, value)
            result['files'][key] = filename
        examples = _physiology_ocular_examples(panel_frame, features, cfg, directory)
        write_json(Path(directory)/'ocular_examples.json', examples)
        result['files']['ocular_examples'] = 'ocular_examples.json'
    write_json(Path(directory)/'shapes_summary.json', result)
    _physiology_summary(cfg, parent, directory)
    return result


def _spatial_dataset_tables(payload):
    from src.analysis import tak_mismatch as analysis
    native, windows, geometry, examples = payload
    halves = analysis.spatial_half_metrics(windows)
    channels, maps = analysis.aggregate_spatial_maps(native, halves, geometry)
    records, ranking = analysis.rank_spatial_patterns(maps, examples_per_dataset=examples)
    return dict(channel_metrics=channels,map_values=maps,record_pattern_scores=records,
        subject_ranking=ranking,record_similarities=analysis.spatial_record_similarities(maps, ranking))


def physiology_topography_stage(cfg, parent, directory, output, *, examples_per_dataset=3, workers=1):
    """Derive spatial maps from completed scalar evidence; no signal reads or fits."""
    import shutil
    import pandas as pd
    from src.analysis import tak_mismatch as analysis
    from src.data import channel_geometry
    parent, directory, output = (Path(p).resolve() for p in (parent, directory, output))
    if output.parent != directory or not output.name.startswith('spatial_patterns_'):
        raise ValueError('spatial analysis writes a versioned direct child of its fixed-analysis owner')
    if output.exists():
        raise FileExistsError('spatial analysis requires a fresh output directory')
    source = read_json(directory/'summary.json')
    if (source.get('schema') != 'tak_mismatch_fixed_physiology_analysis_v1'
            or source.get('status') != 'completed' or source.get('parameter_mode') != 'fixed'
            or source.get('protected_data') != 'forbidden'):
        raise ValueError('completed public fixed-parameter analysis required')
    plan = read_json(parent/'plan.json')
    from src.data.unified_physiology import VISUAL_CHANNEL_IDENTITY_SCHEMA
    visual = [r for r in plan['records'] if r['dataset'] == 'visual_cognitive_motivation']
    corrected = bool(visual) and all(r.get('native_eeg_channel_identity', {}).get('schema')
                                    == VISUAL_CHANNEL_IDENTITY_SCHEMA for r in visual)
    if corrected:
        impact_path = parent/'input_migration_verification.json'
        impact = read_json(impact_path)
        if (impact.get('status') != 'passed_metadata_before_arrays' or impact.get('records') != len(visual)
                or any(len(r['eeg_channel_names']) != 31 or 'FCz' not in r['eeg_channel_names'] for r in visual)):
            raise ValueError('Corrected native Visual identity evidence differs from the frozen plan')
    else:
        impact_path = directory/'visual_channel_impact_audit_v1.json'
        impact = read_json(impact_path)
        if impact.get('status') != 'confirmed' or Path(impact['retained_run']).resolve() != parent:
            raise ValueError('the confirmed Visual impact audit must belong to this retained run')
    output.mkdir()
    sources = output/'source'; sources.mkdir()
    for path in (Path(__file__), Path(analysis.__file__), Path(channel_geometry.__file__)):
        shutil.copy2(path, sources/path.name)
    started = time.monotonic()
    geometry = analysis.spatial_geometry_from_plan(plan)
    native = pd.read_csv(directory/source['files']['native_pairs'], dtype={'subject':str})
    columns = ['dataset','subject','task_label','record_id','record_unit','pair','native_support_id',
        'window_start_s','parameter_mode','D1','zero_NMSE']
    windows = pd.read_csv(directory/source['files']['windows'], usecols=columns, dtype={'subject':str})
    if len(windows) != cfg['scope']['scalar_pair_windows']:
        raise ValueError('retained scalar window denominator changed')
    if set(windows.dataset) - set(('eeg_fnirs_single_trial','simultaneous_eeg_nirs','visual_cognitive_motivation','refed')):
        raise ValueError('only registered public scalar evidence is admitted')
    payloads = [(native[native.dataset.eq(ds)].copy(), windows[windows.dataset.eq(ds)].copy(),
        geometry[geometry.dataset.eq(ds)].copy(), examples_per_dataset) for ds in sorted(windows.dataset.unique())]
    chunks = bounded_work(_spatial_dataset_tables, payloads, min(workers,len(payloads)))
    tables = {name:pd.concat([chunk[name] for chunk in chunks],ignore_index=True) for name in chunks[0]}
    tables['channel_geometry'] = geometry
    ranking = tables['subject_ranking']
    files = {}
    for name, frame in tables.items():
        suffix = '.csv.gz' if name in ('channel_metrics','map_values') else '.csv'
        files[name] = _physiology_save_frame(output, name+suffix, frame)
    selected = ranking[ranking.selected]
    result = dict(schema='tak_mismatch_spatial_patterns_v1',status='completed',created_utc=datetime.now(timezone.utc).isoformat(),
        parent_run=str(parent),source_analysis=str(directory),owning_summary=str(directory/'summary.json'),
        visual_impact_audit=str(impact_path),files=files,elapsed_s=time.monotonic()-started,
        visual_native_eeg_identity=VISUAL_CHANNEL_IDENTITY_SCHEMA if corrected else 'retained_invalid_EEG_identity',
        planned_window_pairs=len(windows),native_records=int(windows.record_id.nunique()),
        subjects=len(ranking),selected_subjects=selected[['dataset','subject','rank','score']].to_dict('records'),
        metrics=list(analysis.SPATIAL_METRICS),workers=min(workers,len(payloads)),model_fits=0,raw_signal_reads=0,protected_reads=0,
        omitted_plan_records=[dict(record_id=r['id'],status=r.get('status'),reason=r.get('excluded_reason'))
            for r in plan['records'] if r['id'] not in set(windows.record_id)],
        selection=dict(endpoint='D1',examples_per_dataset=examples_per_dataset,minimum_window_fraction=.8,
            minimum_common_locations=8,minimum_location_fraction=.8,minimum_valid_windows_per_half=2,
            record_score='((P90-P10)/median) * clip(Moran_I,0,1) * clip(early_late_Spearman,0,1)',
            subject_score='median eligible record score * eligible/planned acquisition records',
            neighbours='symmetric 4 nearest projected positions; edges over 2.5 median nearest distance removed',
            split='original record planned times; equal early/late counts; odd center omitted',
            interpretation='data-selected illustration; early/late agreement also used in selection, not independent confirmation'),
        aggregation=dict(record='all finite retained endpoints displayed, including partial time support; strict 80% coverage used only for selection',
            coincident_positions='location-coincident channels average available numerators/denominators, with all members counted',
            subject='available acquisition records equally weighted at each location; ROOT metrics squared before average; available/planned records retained',
            Hb_ratio='matched-support mean residual MSE divided by mean zero-prediction MSE, not mean window ratios',
            weak='complete native-record pair diagnostic; never count its repeated window aliases',
            visual=('simultaneous Probe1/2 form one acquisition map; corrected native EEG geometry and frozen-fold coordinates'
                    if corrected else 'simultaneous Probe1/2 form one acquisition map; EEG-dependent metrics not spatially interpreted'),
            normalization='frozen location-specific and outer-training-fold SD, not a common physical concentration unit'),
        geometry='retained registry coordinates in plan; orthographic/template top view; coincident projected positions merged',
        claim_limits=['no cortical source localization','selection is exploratory','windows/channels are not independent subjects',
            'different REFED videos are different stimuli',
            'corrected channel identity does not establish cortical localization' if corrected else 'Visual Hb-only exemption does not repair EEG identity'])
    write_json(output/'summary.json', result)
    return {key:result[key] for key in ('schema','status','subjects','selected_subjects','planned_window_pairs','elapsed_s')}


def physiology_patterns_stage(parent, directory, spatial, panel_export, output):
    """Version descriptive mechanism-review tables from completed scalar evidence."""
    import shutil
    import pandas as pd
    from src.analysis import tak_mismatch as analysis
    parent, directory, spatial, panel_export, output = (
        Path(p).resolve() for p in (parent, directory, spatial, panel_export, output))
    if output.parent != directory or not output.name.startswith('mechanism_patterns_'):
        raise ValueError('pattern review writes a fresh mechanism_patterns_* analysis child')
    if output.exists():
        raise FileExistsError('pattern review requires a fresh output')
    fixed = read_json(directory/'summary.json')
    if (fixed.get('schema') != 'tak_mismatch_fixed_physiology_analysis_v1'
            or fixed.get('status') != 'completed' or fixed.get('protected_data') != 'forbidden'):
        raise ValueError('completed public fixed physiology analysis required')
    spatial_summary = read_json(spatial/'summary.json')
    if (spatial.parent != directory or spatial_summary.get('status') != 'completed'
            or spatial_summary.get('schema') != 'tak_mismatch_spatial_patterns_v1'
            or Path(spatial_summary['parent_run']).resolve() != parent):
        raise ValueError('completed spatial analysis must belong to this retained run')
    panel_summary = read_json(panel_export/'summary.json')
    if (panel_export.parent != parent/'plot_priority_v1'/'exports'
            or panel_summary.get('schema') != 'tak_mismatch_plot_snapshot_v1'
            or panel_summary.get('terminal_groups') != panel_summary.get('planned_groups')):
        raise ValueError('completed bounded shared-parameter export required')
    sources = dict(spatial_summary=str(spatial/'summary.json'),
        maps=str(spatial/spatial_summary['files']['map_values']),
        similarities=str(spatial/spatial_summary['files']['record_similarities']),
        ranking=str(spatial/spatial_summary['files']['subject_ranking']),
        plan=str(parent/'plan.json'), fixed_summary=str(directory/'summary.json'),
        shape_subjects=str(directory/fixed['files']['shape_subjects']),
        ocular_associations=str(directory/fixed['files']['ocular_associations']),
        visual_audit=str(spatial_summary['visual_impact_audit']),
        shared_summary=str(panel_export/'summary.json'), shared_windows=str(panel_export/'window_metrics.csv'),
        shared_groups=str(panel_export/'group_status.csv'))
    audit = read_json(Path(sources['visual_audit']))
    if audit.get('status') != 'confirmed' or Path(audit['retained_run']).resolve() != parent:
        raise ValueError('confirmed Visual interpretation audit required')
    maps = pd.read_csv(sources['maps'], dtype={'subject':str})
    windows = pd.read_csv(sources['shared_windows'], dtype={'subject':str})
    groups = pd.read_csv(sources['shared_groups'], dtype={'subject':str})
    if len(groups) != panel_summary['planned_groups'] or not groups.terminal.all():
        raise ValueError('shared group completion inventory changed')
    tables = {**analysis.spatial_region_patterns(maps), **analysis.shared_parameter_patterns(windows, groups)}
    output.mkdir()
    files = {name:_physiology_save_frame(output, name+'.csv', frame) for name,frame in tables.items()}
    source_dir = output/'source';source_dir.mkdir()
    for path in (Path(__file__), Path(analysis.__file__)):
        shutil.copy2(path,source_dir/path.name)
    result = dict(schema='tak_mismatch_mechanism_patterns_v1',status='completed',
        created_utc=datetime.now(timezone.utc).isoformat(),parent_run=str(parent),sources=sources,files=files,
        selection=spatial_summary['selected_subjects'],
        method=dict(regions='AF/Fp=frontal; C/FC=central; P/O=posterior; sensor-location groups, not cortical ROIs',
            region_datasets=['eeg_fnirs_single_trial','simultaneous_eeg_nirs'],
            regional_value='median of retained channel values; require all planned positions in region',
            subject_map='original record-equal subject map, then regional median',
            task_map='native-record regional medians, then record-equal mean within subject/task',
            contrasts='same subject, all three regions complete; posterior-minus-central etc.; no new significance tests',
            parameter_unit='one native-record/continuous-Hb-support/channel group; not repeated windows',
            valid_parameter='at least one retained numerically accepted window; complete-support count also retained',
            parameter_failed='finite terminal estimates retained as failed, never counted as valid',
            fit_pair='identical successful fixed/free window identities within each group; RMS from mean squares',
            parameter_boundary='tau=12s is the engineering upper bound, not a physiological reference',
            visual='excluded from EEG-dependent parameter comparisons; retained audit unchanged'),
        counts={name:len(frame) for name,frame in tables.items()},
        model_fits=0,raw_signal_reads=0,protected_reads=0,new_statistical_tests=0)
    write_json(output/'summary.json',result)
    return dict(status=result['status'],output=str(output),counts=result['counts'])


def physiology_four_dataset_fixed_stage(parent, directory, spatial, output, reference):
    """Read-only fixed census synthesis; stopped shared fits are irrelevant."""
    import shutil
    import pandas as pd
    from src.analysis import tak_mismatch as analysis
    from src.data.unified_physiology import VISUAL_CHANNEL_IDENTITY_SCHEMA
    parent,directory,spatial,output,reference=(Path(p).resolve() for p in (parent,directory,spatial,output,reference))
    if output.parent!=directory or not output.name.startswith('four_dataset_patterns_') or output.exists():
        raise ValueError('Fixed review requires a fresh four_dataset_patterns_* child')
    old=reference/'fixed_physiology_v1'; old_spatial=old/'spatial_patterns_v2'
    summaries=[read_json(p/'summary.json') for p in (old,directory,old_spatial,spatial)]
    if any(s.get('status')!='completed' for s in summaries):raise ValueError('Completed fixed evidence required')
    if any(s.get('parameter_mode')!='fixed' or s.get('protected_data')!='forbidden' for s in summaries[:2]):
        raise ValueError('Public fixed-only analysis required')
    if summaries[3].get('visual_native_eeg_identity')!=VISUAL_CHANNEL_IDENTITY_SCHEMA:
        raise ValueError('Corrected Visual native identity required')
    for owner,s in zip((reference,parent,reference,parent),summaries):
        if Path(s['parent_run']).resolve()!=owner:raise ValueError('Evidence owner mismatch')
    visual='visual_cognitive_motivation'
    def read(p):return pd.read_csv(p,dtype={'subject':str},low_memory=False)
    def combine(a,b,require_four=True):
        if set(b.dataset)-{visual}:raise ValueError('Only corrected Visual can replace legacy rows')
        result=pd.concat([a[~a.dataset.eq(visual)],b],ignore_index=True)
        if require_four and set(result.dataset)!={'eeg_fnirs_single_trial','simultaneous_eeg_nirs',visual,'refed'}:
            raise ValueError('Four dataset evidence required')
        return result
    names=['subjects','subject_descriptives','subject_tasks','repeat_summary','task_comparisons','shape_subjects']
    tables={name:combine(read(old/summaries[0]['files'][name]),read(directory/summaries[1]['files'][name]),
                        require_four=name!='task_comparisons') for name in names}
    maps=combine(read(old_spatial/'map_values.csv.gz'),read(spatial/'map_values.csv.gz'))
    tables.update(analysis.fixed_population_patterns(maps))
    tables['ocular_associations']=read(old/'ocular_associations.csv')
    sources=dict(reference_run=str(reference),corrected_run=str(parent),reference_fixed_summary=str(old/'summary.json'),
        corrected_fixed_summary=str(directory/'summary.json'),reference_maps=str(old_spatial/'map_values.csv.gz'),
        corrected_maps=str(spatial/'map_values.csv.gz'),reference_plan=str(reference/'plan.json'),corrected_plan=str(parent/'plan.json'),
        input_repair=str(parent/'repair_pilot_verification.json'),ocular=str(old/'ocular_associations.csv'))
    a=read(old/'subjects.csv');b=read(directory/'subjects.csv')
    revision=a[a.dataset.eq(visual)].merge(b,on=['dataset','subject','metric'],suffixes=('_old','_corrected'),validate='one_to_one')
    revision['difference']=revision.value_corrected-revision.value_old
    hb=revision[revision.metric.isin(['D1',analysis.HB_RATIO,'D_bal_10s','D_bal_20s','D_bal_40s'])]
    if not np.allclose(hb.value_old,hb.value_corrected,atol=1e-9,rtol=0,equal_nan=True):
        raise ValueError('Pure-Hb endpoint migration check failed')
    tables['visual_revision']=revision
    population=[]
    for (ds,metric,location),g in maps[maps.period.eq('subject_mean')].groupby(['dataset','metric','location_id']):
        p=g.iloc[0];population.append(dict(dataset=ds,metric=metric,location_id=location,x=p.x,y=p.y,
            pair_members=p.pair_members,display_value=g.display_value.median(),planned_subjects=len(g),
            available_subjects=int(g.display_value.notna().sum())))
    tables['all_population_maps']=pd.DataFrame(population)
    coverage={};window_tables=[];examples=[]
    for owner,run,s,corrected in ((old,reference,summaries[0],False),(directory,parent,summaries[1],True)):
        view=read_json(owner/'view_coverage.json')['by_dataset'];plan=read_json(run/'plan.json')
        for ds,v in plan['summary'].items():
            if (ds==visual)==corrected:coverage[ds]=dict(plan=v,views=view[ds],source=str(owner))
        columns=['dataset','subject','task_label','record_id','native_support_id','pair','window_start_s','parameter_mode',
                 'status','D1','D_EH','D_drive_H','Delta_spatial','D4','prediction_NMSE','zero_NMSE','template_NMSE']
        w=pd.read_csv(owner/s['files']['windows'],usecols=columns,dtype={'subject':str})
        w=w[w.dataset.eq(visual) if corrected else ~w.dataset.eq(visual)]
        window_tables.append(w)
        for e in read_json(owner/'examples.json')['examples']:
            if (e['dataset']==visual)==corrected and e['role']=='median':examples.append(dict(e,source_run=str(run)))
    windows=pd.concat(window_tables,ignore_index=True)
    if len(windows)!=sum(c['plan']['planned_pair_windows'] for c in coverage.values()):
        raise ValueError('Full fixed denominator changed')
    metric_coverage=[]
    for ds,g in windows.groupby('dataset'):
        for metric in ('D1','D_EH','D_drive_H','Delta_spatial','D4'):
            metric_coverage.append(dict(dataset=ds,metric=metric,planned=len(g),available=int(np.isfinite(g[metric]).sum())))
    tables['metric_coverage']=pd.DataFrame(metric_coverage)
    gains=['prediction_gain_over_zero','prediction_gain_over_template']
    for name,baseline in zip(gains,['zero_NMSE','template_NMSE']):
        good=np.isfinite(windows.prediction_NMSE)&np.isfinite(windows[baseline])
        windows[name]=(windows[baseline]-windows.prediction_NMSE).where(good)
    tables['independent_prediction_subjects']=analysis.aggregate_fixed_metrics(windows,metrics=gains,include_hb_ratio=False)['subjects']
    output.mkdir();files={k:_physiology_save_frame(output,k+'.csv',v) for k,v in tables.items()}
    write_json(output/'examples.json',dict(examples=examples,selection='existing frozen three-pair fixed-D1 median examples; no reselection'))
    source_dir=output/'source';source_dir.mkdir()
    for path in (Path(__file__),Path(analysis.__file__)):shutil.copy2(path,source_dir/path.name)
    result=dict(schema='tak_mismatch_four_dataset_fixed_patterns_v1',status='completed',created_utc=datetime.now(timezone.utc).isoformat(),
        parent_run=str(parent),sources=sources,files=files,coverage=coverage,planned_windows=len(windows),planned_subjects=sum(c['plan']['subjects'] for c in coverage.values()),
        model_fits=0,raw_signal_reads=0,protected_reads=0,shared_parameter_reads=0,
        verification=dict(full_fixed_denominator=True,old_Visual_rows_replaced=True,unchanged_Hb_subject_endpoints=len(hb),
            maximum_Hb_revision_difference=float(hb.difference.abs().max())),
        methods=dict(region='location median within three complete regions; posterior minus frontal per subject; six exploratory tests across two montages and D1/relative residual/D_bal40; 5000 subject bootstrap means; 100000 sign flips; BH family of six',
            spatial='all subjects; at least8 and80percent common sites; median record-pair Spearman per subject; 5000 subject-bootstrap median CI, descriptive',
            centered='subtract other-subject location median separately for each acquisition before spatial rank correlation; same fitted/scaled coordinates retained',
            hotspots='top ceil(25percent of common sites) per map; overlap divided by top-set size; random-reference k/common-sites is descriptive, not a p-value; each subject contributes median pair overlap',
            persistent_sites='per-site top-quarter frequency across eligible record maps; at least2 and80percent planned records; 80percent frequency is a descriptive display threshold, not a physiological cutoff; ties retained',
            repeats='Single-Trial all same-task record pairs; Simultaneous all cross-task pairs; Visual cross-Part; REFED all different-video pairs, not same-condition retest',
            early_late='retained time halves; minimum2 valid windows per half; at least8 and80percent common sites; all-subject median across records',
            inference='posthoc exploratory selection; nominal intervals and BH values do not correct the whole historical exploration; fixed fits and scales; no causal attribution'),
        limitations=['population-template subtraction does not remove subject-specific fixed sensor or normalization effects',
            'spatial patterns are sensor/template locations, not cortical localization','joint and shared-parameter comparisons are outside this report'])
    write_json(output/'summary.json',result)
    return dict(status='completed',output=str(output),planned_windows=len(windows),planned_subjects=result['planned_subjects'])


def physiology_four_dataset_stage(parent, directory, spatial, panel_export, output, reference):
    """Replace the invalid Visual evidence, retaining each other dataset's owner."""
    import shutil
    import pandas as pd
    from src.analysis import tak_mismatch as analysis
    from src.data.unified_physiology import VISUAL_CHANNEL_IDENTITY_SCHEMA
    parent, directory, spatial, panel_export, output, reference = (
        Path(p).resolve() for p in (parent, directory, spatial, panel_export, output, reference))
    if output.parent != directory or not output.name.startswith('four_dataset_patterns_'):
        raise ValueError('Four-dataset review belongs to a fresh child of the corrected analysis')
    if output.exists():
        raise FileExistsError('Four-dataset review requires a fresh export')
    fixed = read_json(directory/'summary.json')
    spatial_summary = read_json(spatial/'summary.json')
    panel = read_json(panel_export/'summary.json')
    current_plan = read_json(parent/'plan.json')
    if (fixed.get('status') != 'completed' or fixed.get('protected_data') != 'forbidden'
            or Path(fixed['parent_run']).name != parent.name
            or spatial_summary.get('status') != 'completed'
            or spatial_summary.get('visual_native_eeg_identity') != VISUAL_CHANNEL_IDENTITY_SCHEMA
            or Path(spatial_summary['parent_run']).resolve() != parent
            or panel.get('terminal_groups') != panel.get('planned_groups')
            or panel_export.parent != parent/'plot_priority_v1'/'exports'):
        raise ValueError('Completed corrected Visual scalar, spatial and common-panel evidence required')
    if set(current_plan['datasets']) != {'visual_cognitive_motivation'}:
        raise ValueError('Only Visual may replace the retained four-dataset evidence')
    old_directory = reference/'fixed_physiology_v1'
    old_fixed = read_json(old_directory/'summary.json')
    old_review = read_json(old_directory/'mechanism_patterns_v1'/'summary.json')
    old_plan = read_json(reference/'plan.json')
    if (old_fixed.get('status') != 'completed' or old_fixed.get('protected_data') != 'forbidden'
            or Path(old_review['parent_run']).resolve() != reference):
        raise ValueError('Completed public reference analysis required')
    old_records = {r['id']: r for r in old_plan['records']}
    for record in current_plan['records']:
        before = old_records[record['id']]
        for key in ('source_identity', 'array_shapes', 'fold', 'windows', 'continuous_hb_segments', 'hb_only_admitted'):
            if record.get(key) != before.get(key):
                raise ValueError('Visual revision changed an undeclared source, support or subject fold')
    sources = dict(reference_run=str(reference), corrected_run=str(parent),
        reference_fixed_summary=str(old_directory/'summary.json'), corrected_fixed_summary=str(directory/'summary.json'),
        reference_plan=str(reference/'plan.json'), corrected_plan=str(parent/'plan.json'),
        reference_maps=old_review['sources']['maps'], corrected_maps=str(spatial/spatial_summary['files']['map_values']),
        reference_shared_windows=old_review['sources']['shared_windows'], corrected_shared_windows=str(panel_export/'window_metrics.csv'),
        reference_shared_groups=old_review['sources']['shared_groups'], corrected_shared_groups=str(panel_export/'group_status.csv'),
        reference_spatial_summary=old_review['sources']['spatial_summary'], corrected_spatial_summary=str(spatial/'summary.json'),
        input_repair=str(parent/'repair_pilot_verification.json'))
    def read(path):
        return pd.read_csv(path, dtype={'subject':str})
    visual = 'visual_cognitive_motivation'
    def replace_visual(old, new):
        if set(new.dataset) != {visual}:
            raise ValueError('Corrected source contains an undeclared dataset')
        return pd.concat([old[~old.dataset.eq(visual)], new], ignore_index=True)
    scalar_old = read(old_directory/old_fixed['files']['subjects'])
    scalar_new = read(directory/fixed['files']['subjects'])
    subjects = replace_visual(scalar_old, scalar_new)
    shapes = replace_visual(read(old_directory/old_fixed['files']['shape_subjects']), read(directory/fixed['files']['shape_subjects']))
    maps = replace_visual(read(sources['reference_maps']), read(sources['corrected_maps']))
    shared_windows = replace_visual(read(sources['reference_shared_windows']), read(sources['corrected_shared_windows']))
    shared_groups = replace_visual(read(sources['reference_shared_groups']), read(sources['corrected_shared_groups']))
    if not shared_groups.terminal.all() or not shared_windows.current_terminal.all():
        raise ValueError('Common shared-parameter panel has unfinished groups')
    old_ids = set(read(sources['reference_shared_windows']).query('dataset == @visual').id)
    if old_ids != set(shared_windows.query('dataset == @visual').id):
        raise ValueError('Corrected Visual shared panel changed the preselected window/pair identities')
    tables = analysis.four_dataset_pattern_tables(subjects, shapes, maps, shared_windows, shared_groups)
    columns = ['dataset','subject','task_label','record_id','native_support_id','pair','window_start_s',
               'parameter_mode','prediction_NMSE','zero_NMSE','template_NMSE']
    independent = replace_visual(
        pd.read_csv(old_directory/old_fixed['files']['windows'], usecols=columns, dtype={'subject':str}),
        pd.read_csv(directory/fixed['files']['windows'], usecols=columns, dtype={'subject':str}))
    gains = ('prediction_gain_over_zero', 'prediction_gain_over_template')
    for metric, baseline in zip(gains, ('zero_NMSE','template_NMSE')):
        common = np.isfinite(independent.prediction_NMSE) & np.isfinite(independent[baseline])
        independent[metric] = (independent[baseline]-independent.prediction_NMSE).where(common)
    tables['independent_prediction_subjects'] = analysis.aggregate_fixed_metrics(
        independent, metrics=gains, include_hb_ratio=False)['subjects']
    revision = scalar_old[scalar_old.dataset.eq(visual)].merge(scalar_new,
        on=['dataset','subject','metric'], suffixes=('_old','_corrected'), validate='one_to_one')
    revision['corrected_minus_old'] = revision.value_corrected - revision.value_old
    hb_revision = revision[revision.metric.isin(['D1','D_bal_10s','D_bal_20s','D_bal_40s'])]
    if (hb_revision.empty or not np.allclose(hb_revision.value_old, hb_revision.value_corrected,
                                            rtol=0, atol=1e-9, equal_nan=True)):
        raise ValueError('Unchanged Hb-only endpoints failed the full subject-level migration check')
    tables['visual_revision_subjects'] = revision
    population = []
    for (dataset, metric, location), block in maps[maps.period.eq('subject_mean')].groupby(['dataset','metric','location_id']):
        row = block.iloc[0]
        values = block.display_value.dropna()
        population.append(dict(dataset=dataset, metric=metric, location_id=location,
            x=row.x, y=row.y, pair_members=row.pair_members, display_value=values.median(),
            available_subjects=len(values), planned_subjects=len(block)))
    tables['population_maps'] = pd.DataFrame(population)
    full_visual = None
    full_summary_path = parent/'summary.json'
    if full_summary_path.exists() and read_json(full_summary_path).get('status') == 'completed':
        full_status = read_json(parent/'current_dof'/'summary.json')
        expected_windows = sum(len(r['windows'])*len(r['pairs']) for r in current_plan['records'])
        if (full_status.get('status') != 'completed' or full_status.get('terminal_windows') != expected_windows
                or full_status.get('planned_windows') != expected_windows):
            raise ValueError('Full Visual shared-parameter denominator is incomplete')
        needed = ['id','group_id','dataset','subject','record_id','pair','hb_only_success','joint_success',
                  'kappa_hb','tau_hb','kappa_joint','tau_joint','D1_free','D_EH_free','D4_free']
        full_windows = pd.read_csv(parent/'metrics_current_dof.csv',usecols=needed,dtype={'subject':str})
        if len(full_windows) != expected_windows or set(full_windows.dataset) != {visual}:
            raise ValueError('Full Visual shared table differs from its terminal plan')
        full_windows = full_windows.rename(columns={'kappa_hb':'hb_only_kappa','tau_hb':'hb_only_tau',
            'kappa_joint':'joint_kappa','tau_joint':'joint_tau'})
        full_fixed = pd.read_csv(parent/'metrics.csv',usecols=['id','D1','D_EH','D4'])
        full_windows = full_windows.merge(full_fixed,on='id',validate='one_to_one')
        labels = {r['id']:_physiology_task_label(r)[0] for r in current_plan['records']}
        full_windows['task_label'] = full_windows.record_id.map(labels)
        full_groups = []
        for record, group in current_dof_groups(current_plan):
            retained = read_json(parent/'current_dof'/'tasks'/f"{group['id']}.json")
            if len(retained['rows']) != len(group['tasks']):
                raise ValueError('Full shared group lost planned rows')
            full_groups.append(dict(group_id=group['id'],dataset=record['dataset'],subject=record['subject'],
                record_id=record['id'],task_label=labels[record['id']],pair=group['pair'],
                status=retained['status'],planned_windows=len(group['tasks'])))
        if len(full_groups) != full_status['planned_groups'] or len(full_groups) != full_status['terminal_groups']:
            raise ValueError('Full shared group inventory changed')
        full_tables = analysis.shared_parameter_patterns(full_windows,pd.DataFrame(full_groups))
        tables.update({'visual_full_'+name:table for name,table in full_tables.items()})
        site = full_tables['parameter_site_summary']
        parameter_maps = []
        for _, location in maps[maps.dataset.eq(visual)&maps.period.eq('subject_mean')&maps.metric.eq('D1')].iterrows():
            for view in ('hb_only','joint'):
                members = site[site.subject.eq(location.subject)&site.view.eq(view)&site.pair.isin(location.pair_members.split(';'))]
                values = members.tau_paired_view_median.dropna()
                parameter_maps.append(dict(dataset=visual,subject=location.subject,view=view,
                    location_id=location.location_id,x=location.x,y=location.y,pair_members=location.pair_members,
                    record_ids=location.record_ids,display_value=values.median(),
                    available_members=len(values),planned_members=len(members),
                    paired_groups=int(members.paired_view_groups.sum()),planned_groups=int(members.planned_groups.sum())))
        tables['visual_full_parameter_maps'] = pd.DataFrame(parameter_maps)
        full_visual = dict(status='completed',groups=len(full_groups),windows=len(full_windows),
            summary=str(parent/'current_dof'/'summary.json'),terminal_audit=str(parent/'terminal_status_audit.json'),
            metrics=str(parent/'metrics_current_dof.csv'),
            map_aggregation='same-group paired native-support parameter median per channel, then median across location-coincident channel medians',
            interpretation='all native Visual Hb pairs; separate extension, not a matched-coverage comparison to other datasets')
    examples = []
    for owner, summary, replace in ((old_directory, old_fixed, False), (directory, fixed, True)):
        for example in read_json(owner/summary['files']['examples'])['examples']:
            if (example['dataset'] == visual) != replace or example['role'] not in ('median', 'p90'):
                continue
            examples.append(dict(example, source_run=str(parent if replace else reference)))
    if len(examples) != 8 or len({(r['dataset'],r['role']) for r in examples}) != 8:
        raise ValueError('Exactly median and P90 examples from every dataset are required')
    coverage = {}
    for plan, owner, summary, replace in ((old_plan, old_directory, old_fixed, False), (current_plan, directory, fixed, True)):
        view = read_json(owner/summary['files']['view_coverage'])['by_dataset']
        for dataset, value in plan['summary'].items():
            if (dataset == visual) == replace:
                coverage[dataset] = dict(plan=value, fixed_views=view[dataset], source_run=str(parent if replace else reference))
    output.mkdir()
    files = {name:_physiology_save_frame(output,name+'.csv',frame) for name,frame in tables.items()}
    write_json(output/'examples.json',dict(examples=examples,
        selection='frozen_three_pair_panel_fixed_D1_nearest_median_and_P90_distinct_subjects_when_possible'))
    files['examples'] = 'examples.json'
    source_dir = output/'source'; source_dir.mkdir()
    for path in (Path(__file__), Path(analysis.__file__)):
        shutil.copy2(path,source_dir/path.name)
    result = dict(schema='tak_mismatch_four_dataset_patterns_v1', status='completed',
        created_utc=datetime.now(timezone.utc).isoformat(), parent_run=str(parent), sources=sources, files=files,
        coverage=coverage, planned_subjects=sum(v['plan']['subjects'] for v in coverage.values()),
        shared_groups=len(shared_groups), shared_windows=len(shared_windows), raw_signal_reads=0, model_fits=0, protected_reads=0,
        methods=dict(scalar='retained window-square pooling, native-record equal, subject equal; each dataset separate',
            shapes='preselected three-pair panel; moments within record, equal records within subject before ratios',
            maps='subject median at each registered location; interpolation is display only',
            spatial_contrast='per acquisition (P90-P10)/median, at least8 and80percent available locations; subject median over acquisitions',
            early_late='original temporal halves; two valid windows per half and at least8/80percent common sites; subject median across acquisitions',
            shared_error='same successful windows per group, equal groups within record, equal records within subject, then square root',
            shared_parameters='valid group log-estimates averaged within native record then subject; exponentiated; failures and tau12s boundary retained',
            independent_prediction='same-window baseline NMSE minus EEG-only NMSE; require both finite before aggregation; window then record then subject',
            visual_revision='data-integrity comparison; EEG target coordinates changed, so differences are not accuracy improvements',
            inference='descriptive, no new significance tests; no cortical source or unique physiological truth claim'))
    result['verification'] = dict(unchanged_Hb_subject_endpoints=True, tolerance=1e-9,
        matched_subject_endpoints=len(hb_revision), maximum_absolute_difference=float(hb_revision.corrected_minus_old.abs().max()),
        four_datasets_present=True, old_Visual_EEG_rows_replaced=True, shared_panel_window_identities_unchanged=True)
    result['visual_full_shared'] = full_visual
    write_json(output/'summary.json',result)
    return {k:result[k] for k in ('schema','status','planned_subjects','shared_groups','shared_windows')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--analysis-config', type=Path, default=DEFAULT_PHYSIOLOGY_CONFIG)
    parser.add_argument('--stage', choices=STAGES, default='check')
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--spatial-output', type=Path, help='Fresh spatial_patterns_* child of the completed fixed-physiology analysis')
    parser.add_argument('--spatial-analysis', type=Path, help='Explicit completed spatial analysis for physiology-patterns')
    parser.add_argument('--panel-export', type=Path, help='Explicit completed bounded shared-parameter snapshot for physiology-patterns')
    parser.add_argument('--pattern-output', type=Path, help='Fresh mechanism_patterns_* child of the fixed-physiology analysis')
    parser.add_argument('--reference-run', type=Path, help='Explicit retained four-dataset run for the corrected-Visual review')
    parser.add_argument('--fixed-only', action='store_true', help='Four-dataset fixed census review; no shared-fit dependency')
    parser.add_argument('--spatial-examples', type=int, default=3, help='Illustrative subjects selected per dataset')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-tasks', type=int, help='Explicit bounded throughput pilot; never marks the full campaign complete')
    parser.add_argument('--max-records', type=int, help='Prepare-only dataset-round-robin resource pilot; the complete plan is retained')
    args = parser.parse_args(argv); cfg = read_config(args.config)
    if not 1 <= args.workers <= cfg['resources']['max_workers']: parser.error('workers outside reviewed resource envelope')
    if args.max_tasks is not None and (args.stage != 'run' or args.max_tasks < 1):
        parser.error('--max-tasks is a positive run-stage throughput pilot only')
    pilot_stages = ('prepare', 'physiology-extract', 'physiology-ocular', 'physiology-counterfactual', 'physiology-shapes')
    if args.max_records is not None and (args.stage not in pilot_stages or args.max_records < 1):
        parser.error('--max-records is a positive record-stage throughput pilot only')
    if args.stage == 'check':
        print(json.dumps(serial(check(cfg)), ensure_ascii=False, allow_nan=False)); return 0
    root = args.project_root.resolve()
    out = (args.run_dir or root/ARTIFACT_ROOT/cfg['run_name']).resolve()
    owner = root/ARTIFACT_ROOT
    if out.parent != owner.resolve() or out.name != cfg['run_name']:
        parser.error('this contract writes only its exact versioned owning run directory')
    if args.stage == 'physiology-four-dataset':
        if not all((args.spatial_analysis, args.pattern_output, args.reference_run)) or (not args.fixed_only and not args.panel_export):
            parser.error('four-dataset review requires spatial-analysis, panel-export, pattern-output and reference-run')
        directory = _physiology_directory(read_physiology_config(args.analysis_config), out, root)
        if args.fixed_only:
            if args.panel_export:parser.error('fixed-only review does not consume shared-parameter panels')
            result=physiology_four_dataset_fixed_stage(out,directory,args.spatial_analysis,args.pattern_output,args.reference_run)
        else:
            result = physiology_four_dataset_stage(out,directory,args.spatial_analysis,args.panel_export,
                args.pattern_output,args.reference_run)
        print(json.dumps(serial(result),ensure_ascii=False,allow_nan=False));return 0
    if args.fixed_only:parser.error('--fixed-only belongs only to physiology-four-dataset')
    if args.reference_run:
        parser.error('--reference-run belongs only to physiology-four-dataset')
    if args.stage == 'physiology-patterns':
        if not all((args.spatial_analysis,args.panel_export,args.pattern_output)) or args.spatial_output:
            parser.error('physiology-patterns requires --spatial-analysis, --panel-export and --pattern-output')
        directory = _physiology_directory(read_physiology_config(args.analysis_config), out, root)
        result = physiology_patterns_stage(out,directory,args.spatial_analysis,args.panel_export,args.pattern_output)
        print(json.dumps(serial(result),ensure_ascii=False,allow_nan=False));return 0
    if any((args.spatial_analysis,args.panel_export,args.pattern_output)):
        parser.error('pattern-review arguments belong to physiology-patterns only')
    if args.stage == 'physiology-topography':
        analysis_cfg = read_physiology_config(args.analysis_config)
        directory = _physiology_directory(analysis_cfg, out, root)
        if args.spatial_output is None or args.spatial_examples < 1:
            parser.error('physiology-topography requires a fresh --spatial-output and positive --spatial-examples')
        result = physiology_topography_stage(analysis_cfg, out, directory, args.spatial_output,
            examples_per_dataset=args.spatial_examples, workers=args.workers)
        print(json.dumps(serial(result), ensure_ascii=False, allow_nan=False)); return 0
    if args.spatial_output:
        parser.error('--spatial-output belongs to physiology-topography only')
    if args.stage in PHYSIOLOGY_STAGES:
        analysis_cfg = read_physiology_config(args.analysis_config)
        if args.workers > analysis_cfg['resources']['max_workers']:
            parser.error('workers outside fixed-physiology resource envelope')
        directory = _physiology_directory(analysis_cfg, out, root)
        with run_lock(directory):
            freeze_config(analysis_cfg, directory)
            plan, panel, panel_ids = _physiology_plan(analysis_cfg, out)
            if args.stage == 'physiology-extract':
                result = physiology_extract_stage(analysis_cfg, out, directory, plan, panel_ids, args.workers,
                    max_records=args.max_records)
            elif args.stage == 'physiology-analyze':
                result = physiology_analyze_stage(analysis_cfg, out, directory, args.workers)
            elif args.stage == 'physiology-ocular':
                result = physiology_ocular_stage(cfg, analysis_cfg, root, out, directory, plan, panel_ids,
                    args.workers, max_records=args.max_records)
            elif args.stage == 'physiology-counterfactual':
                result = physiology_counterfactual_stage(cfg, analysis_cfg, root, out, directory, plan, panel_ids,
                    args.workers, max_records=args.max_records)
            else:
                result = physiology_shapes_stage(analysis_cfg, out, directory, plan, panel_ids,
                    args.workers, max_records=args.max_records)
        print(json.dumps(serial(result), ensure_ascii=False, allow_nan=False))
        return 0 if args.max_records is not None or result.get('status') == 'completed' else 2
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
            if args.stage == 'plot-plan': result = plot_priority_plan(cfg, out, plan)
            elif args.stage == 'plot-run': result = plot_priority_stage(cfg, out, plan, args.workers)
            elif args.stage == 'prepare': result = prepare_stage(cfg, out, root, plan, args.workers, max_records=args.max_records)
            elif args.stage in ('run', 'fixed-run'):
                result = run_stage(cfg, out, plan, args.workers, max_tasks=args.max_tasks,
                    include_current_dof=args.stage == 'run')
            else: result = summarize_stage(cfg, out, plan)
    display = result
    if args.stage == 'plot-plan':
        display = {key: result[key] for key in ('schema', 'planned_groups', 'planned_windows', 'planned_cells', 'by_dataset')}
    elif args.stage == 'plan':
        display = {key: result[key] for key in ('schema', 'status', 'summary', 'measured_arrays_read')}
        display['record_count'] = len(result['records']); display['plan_path'] = str(out/'plan.json')
    print(json.dumps(serial(display), ensure_ascii=False, allow_nan=False)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
