#!/usr/bin/env python3
"""Bounded Step5 observation diagnostics; retained Step5 evidence is immutable.

Uses only original training trials. Cross-validation repeats channel/projection
fitting inside every fold. W curves are joint likelihoods or single-modality
refits; segment increments preserve chronological filtering history.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

for _thread_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_var] = '1'

import numpy as np
import scipy
import yaml
from scipy.signal import butter, periodogram, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import evaluate_step5 as step5
from src.inference.observation_baselines import (
    bridge_transform, first_difference_noise, linear_features, ridge_fit,
    ridge_predict, robust_mad, student_difference_mad,
)

DEFAULT_CONFIG = ROOT / 'experiments/configs/physiology_semantic_tokenizer/step5_observation_diagnostic_v1.yaml'
COORDINATES = ('broadband_pca', 'local_F3_alpha')
VARIANTS = ('model', 'baseline', 'fnirs_filter', 'resample_roundtrip',
            'common_scale', 'noise_estimate', 'combined')
SOURCE_FILES = (
    'experiments/evaluate_step5_observation_diagnostic.py',
    'experiments/evaluate_step5.py',
    'experiments/evaluate_step5a_inference_consistency.py',
    'experiments/evaluate_t3_multisession_loso.py',
    'src/inference/t3a_balloon_joint_ssm.py',
    'src/inference/t3a_balloon_robust_ssm.py',
    'src/data/homer2_preprocessing.py',
    'src/inference/observation_baselines.py',
    'src/metrics/trajectory_reliability.py',
    'src/data/unified_physiology.py',
    'src/data/clean_physiology_cache.py',
    'experiments/build_clean_eeg_fnirs_cache.py',
    'experiments/evaluate_shared_neural_driver_unified.py',
    'experiments/configs/physiology_semantic_tokenizer/step5_v1.yaml',
    'experiments/configs/physiology_semantic_tokenizer/step5b_v2.yaml',
    'experiments/configs/physiology_semantic_tokenizer/t3_multisession_loso_v1.yaml',
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['schema'] != 'step5_observation_diagnostic_v1' or cfg['scope'] != 'training_only_observation_diagnostic':
        raise ValueError('diagnostic schema/scope mismatch')
    if cfg['subjects'] != ['subject_01', 'subject_09', 'subject_18']:
        raise ValueError('only the three preselected development subjects are in scope')
    if cfg['measured_config'] != str(step5.MEASURED_CONFIG.relative_to(ROOT)):
        raise ValueError('measured owner changed')
    base, measured, metadata = step5.load_measured_config(ROOT / cfg['measured_config'])
    if (cfg['coordinates'] != list(COORDINATES) or cfg['curve']['bounds'] != base['axes']['W']['bounds']
            or cfg['curve']['fixed_w'] != [0., -.5] or cfg['bridge']['variants'] != list(VARIANTS)):
        raise ValueError('coordinate, parameter support or bridge panel drift')
    if cfg['local_eeg']['channel'] != 'F3' or cfg['local_eeg']['band_hz'] != [8., 13.]:
        raise ValueError('local EEG coordinate was not preregistered')
    if cfg['interpretation']['teacher_qualification'] != 'none':
        raise ValueError('diagnostic cannot grant teacher qualification')
    if (cfg['curve']['grid_points'] != 17 or cfg['curve']['quadrature_order'] != 13
            or cfg['curve']['quadrature_check_order'] != 17
            or cfg['bridge']['replicates'] != 24 or cfg['linear']['outer_folds'] != 4
            or cfg['linear']['inner_folds'] != 3):
        raise ValueError('frozen diagnostic budget changed')
    base['model']['steps'] = round(measured['window_seconds'] * measured['sampling_hz'])
    return cfg, base, measured, metadata


def training_events(events, base):
    """Filter by trial identity before slicing, validity checks or transforms."""
    events = sorted((e for e in events if e['label'] == base['measured']['condition']),
                    key=lambda e: int(e['event_index']))
    if len(events) != 10:
        raise ValueError('exact ten MA event identities per session required')
    return [(i, e) for i, e in enumerate(events)
            if i not in base['measured']['heldout_trial_positions']]


def local_eeg_feature(eeg, names, cfg, mask_name=None):
    local = cfg['local_eeg']
    if list(names).count(local['channel']) != 1:
        raise ValueError('registered local EEG channel missing or ambiguous; no substitute')
    values = np.array(eeg[:, list(names).index(local['channel'])], dtype=float, copy=True)
    count = len(values) // 50
    hidden = np.zeros(count, dtype=bool)
    if mask_name == 'center_EEG':
        _, hidden, _ = step5.masked_input(np.zeros((count, 3)), mask_name, 16)
        native_hidden = np.repeat(hidden, 50)
        visible = ~native_hidden
        values[native_hidden] = np.interp(np.flatnonzero(native_hidden), np.flatnonzero(visible), values[visible])
    filtered = sosfiltfilt(butter(local['filter_order'], local['band_hz'], btype='bandpass',
                                 fs=200., output='sos'), values)
    power = np.log(np.maximum(np.mean(filtered.reshape(count, 50)**2, axis=1), 1e-12))
    power[hidden] = np.nan
    return power


def load_training_subject(subject, cfg, base, measured, metadata, *, retain_native=False,
                          retain_feature_boundary=False, data_root=None):
    """Single native entry for this diagnostic; scope checked before any reader.

    Native files store entire sessions. Only the original eight training windows
    per session are sliced or transformed. No original held-out trial is scored.
    """
    if subject not in cfg['subjects'] or subject not in base['measured']['subjects']:
        raise ValueError('subject outside diagnostic training boundary')
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    from src.data.unified_physiology import load_native_eeg_record
    from experiments.build_clean_eeg_fnirs_cache import _pair_single_trial_wavelengths
    data_root = ROOT if data_root is None else Path(data_root).resolve()
    index = CleanPhysiologyCacheIndex(data_root / metadata['data']['cache_root'])
    records = sorted((r for r in index.records if r.dataset_id == base['measured']['dataset_id']
                      and r.canonical_subject_id == subject and r.base_record_id in base['measured']['sessions']),
                     key=lambda r: r.base_record_id)
    if [r.base_record_id for r in records] != base['measured']['sessions']:
        raise ValueError('exact three admitted records required before native access')
    trials, identities, sources = [], [], {}
    channel_identity = None
    for record in records:
        events = training_events(index.events_by_join_key[record.join_key], base)
        native = load_native_eeg_record(data_root, record)
        with np.load(record.npz_path, allow_pickle=False) as arrays:
            paired, pairs = _pair_single_trial_wavelengths(arrays['native_input_fnirs'], arrays['native_channel_names'])
        identity = (tuple(native.channel_names), tuple(pairs))
        if channel_identity is not None and identity != channel_identity:
            raise ValueError('cross-session channel identity drift')
        channel_identity = identity
        if native.sample_rate_hz != 200. or record.sample_rate_hz != 10.:
            raise ValueError('native clock drift')
        for path in (native.source_path, record.npz_path):
            sources[str(path.relative_to(data_root))] = digest(path)
        for ordinal, (position, event) in enumerate(events):
            starts = {m: round((event[f'{m}_time_ms']/1000 - 5.)*hz)
                      for m, hz in [('eeg', 200.), ('fnirs', 10.)]}
            if min(starts.values()) < 0:
                raise ValueError('negative training trial start')
            eeg = native.values[starts['eeg']:starts['eeg']+6000].copy()
            fnirs = paired[starts['fnirs']:starts['fnirs']+300].copy()
            if eeg.shape[0] != 6000 or fnirs.shape[0] != 300 or not np.isfinite(eeg).all() or not np.isfinite(fnirs).all():
                raise ValueError('training native support invalid')
            views = {}
            for mask in (None, 'center_EEG', 'center_fNIRS'):
                features = step5.preprocess_native_trial(
                    eeg, fnirs, mask_name=mask,
                    retain_feature_boundary=retain_feature_boundary and mask is None)
                features['local_eeg'] = local_eeg_feature(eeg, native.channel_names, cfg, mask)
                views[mask or 'target'] = features
            trials.append(dict(views=views, eligible=np.all(fnirs > 0, axis=(0, 2)),
                               session=record.base_record_id, ordinal=ordinal))
            if retain_native:
                auxiliary = native.auxiliary_values
                eog = (np.empty((len(eeg), 0)) if auxiliary is None else
                       auxiliary[starts['eeg']:starts['eeg']+6000].copy())
                if eog.shape[0] != len(eeg) or not np.isfinite(eog).all():
                    raise ValueError('training EOG support invalid')
                trials[-1]['native_eeg'] = eeg
                trials[-1]['native_eog'] = eog
            identities.append(dict(subject=subject, session=record.base_record_id,
                original_ma_trial_position=position, training_ordinal=ordinal,
                event_index=int(event['event_index']), native_start_samples=starts,
                eeg_time_ms=event['eeg_time_ms'], fnirs_time_ms=event['fnirs_time_ms'],
                sample_id=f"{record.join_key}|event={event['event_index']}|offset=-5.0|duration=30.0"))
    if len(trials) != 24:
        raise ValueError('exact 24 original training trials required')
    detail = dict(subject=subject, trials=identities, source_sha256=sources,
                  eeg_channels=channel_identity[0], fnirs_pairs=channel_identity[1],
                  eog_channels=list(native.auxiliary_channel_names),
                  original_heldout_trials_processed=0,
                  source_storage='whole native session files; only original training windows processed')
    return trials, detail


def fit_projection(trials, indices, base, measured):
    features = [trials[i]['views']['target'] for i in indices]
    eligible = np.logical_and.reduce([trials[i]['eligible'] for i in indices])
    projection = step5.fit_measured_projection(features, base, measured, eligible)
    local = np.concatenate([f['local_eeg']-f['local_eeg'][:20].mean() for f in features])
    projection['local_factor'] = projection['gauge']['eeg']/max(float(robust_mad(local)), 1e-8)
    return projection


def project_trial(trial, projection, coordinate, view='target'):
    if not trial['eligible'][projection['fnirs_pair']]:
        raise ValueError('training-selected fNIRS pair invalid on validation trial; no replacement')
    features = trial['views'][view]
    values = step5.apply_measured_projection(features, projection)
    if coordinate == 'local_F3_alpha':
        local = features['local_eeg']
        values[:, 0] = (local-local[:20].mean())*projection['local_factor']
    elif coordinate != 'broadband_pca':
        raise ValueError('unknown coordinate')
    return values


def filter_trace(y, config, w, order, residuals=False):
    """Read-only instrumentation of the existing joint filter's exact updates."""
    p, c = step5.localization.model(config, 'W', float(w))
    spec = step5.core.BalloonObservationSpec().resolved(p.fixed)
    mean = np.zeros(6)
    covariance = np.diag(np.square(c.initial_state_std))
    q = np.diag(np.square(p.fixed.process_std))*c.dt
    increments, predictive_residuals, standardized = [], [], []
    h = step5.core._observation_physical_matrix(p, spec)
    for t, observation in enumerate(y):
        if t:
            mean, a = step5.core.rk4_transition_with_jacobian(mean, p, c)
            covariance = step5.core._project_psd(a@covariance@a.T+q)
        if residuals:
            physical_mean, physical_cov = step5.core.transformed_gaussian_moments(mean, covariance)
            predicted = step5.core._observation_map_unchecked(physical_mean, p, spec)
            variance = np.diag(h@physical_cov@h.T)+np.square(spec.observation_scale)*spec.student_nu/(spec.student_nu-2)
            predictive_residuals.append(observation-predicted)
            standardized.append((observation-predicted)/np.sqrt(variance))
        mean, covariance, ll, _ = step5.joint.joint_observation_update(
            mean, covariance, observation, np.isfinite(observation), p, spec, order, False)
        increments.append(ll)
    return dict(increments=np.array(increments), predictive_residuals=np.array(predictive_residuals),
                standardized_predictive_residuals=np.array(standardized))


def residual_summary(traces, hz=4., max_seconds=5.):
    values = np.array(traces)
    result = {}
    for col, name in enumerate(('EEG', 'HbO', 'HbR')):
        rows = values[:, :, col]
        if not np.isfinite(rows).any():
            continue
        # Center each trial separately and never correlate across trial resets.
        centered = rows-rows.mean(axis=1, keepdims=True)
        acf = [1.]
        for lag in range(1, int(max_seconds*hz)+1):
            left, right = centered[:, :-lag], centered[:, lag:]
            acf.append(float(np.sum(left*right)/max(np.sqrt(np.sum(left**2)*np.sum(right**2)), 1e-20)))
        frequency, psd = periodogram(centered, fs=hz, axis=1)
        psd = psd.mean(axis=0)
        result[name] = dict(mean=float(rows.mean()), rms=float(np.sqrt(np.mean(rows**2))),
            mean_squared_trial_bias=float(np.mean(rows.mean(axis=1)**2)),
            acf=acf, lag_seconds=np.arange(len(acf))/hz,
            frequency_hz=frequency, mean_psd=psd,
            low_frequency_power_fraction=float(psd[(frequency > 0)&(frequency <= .2)].sum()/max(psd.sum(), 1e-20)))
    return result


def curve_job(config, observations, modality, w, cfg, order=None):
    y = np.array(observations, copy=True)
    if modality == 'EEG_only':
        y[:, :, 1:] = np.nan
    elif modality == 'fNIRS_only':
        y[:, :, 0] = np.nan
    elif modality != 'joint':
        raise ValueError('unknown modality')
    fixed = w in cfg['curve']['fixed_w']
    traces = [filter_trace(row, config, w, order or cfg['curve']['quadrature_order'], fixed) for row in y]
    increments = np.array([t['increments'] for t in traces])
    clock = np.arange(y.shape[1])*config['model']['dt']-5.
    segments = {name: float(increments[:, (clock >= bounds[0]) & (clock < bounds[1])].sum())
                for name, bounds in cfg['curve']['segments_seconds'].items()}
    result = dict(w=w, modality=modality, parameter_log_likelihood=float(increments.sum()),
                  chronological_segment_log_likelihood=segments, trial_log_likelihood=increments.sum(axis=1))
    if fixed:
        result['standardized_predictive_residuals'] = residual_summary(
            [t['standardized_predictive_residuals'] for t in traces], max_seconds=cfg['curve']['residual_acf_seconds'])
        errors = np.array([t['standardized_predictive_residuals'] for t in traces])
        result['segment_standardized_predictive_residual_mean'] = {
            name: {label: float(np.nanmean(errors[:, (clock >= b[0]) & (clock < b[1]), col]))
                   for col, label in enumerate(('EEG', 'HbO', 'HbR'))
                   if np.isfinite(errors[:, (clock >= b[0]) & (clock < b[1]), col]).any()}
            for name, b in cfg['curve']['segments_seconds'].items()}
    return result


def bridge_job(cfg, base, replicate, noise_constant):
    generated = step5.localization.generate_matched(base, 'W', cfg['bridge']['truth_w'], cfg['seed']+1000+replicate)
    training = step5.localization.generate_matched(base, 'W', cfg['bridge']['truth_w'], cfg['seed']+2000+replicate)
    rows = []
    for variant in cfg['bridge']['variants']:
        config = copy.deepcopy(base)
        y = bridge_transform(generated['observations'], variant, cfg)
        clean = bridge_transform(generated['clean'], variant, cfg)
        estimated = first_difference_noise([bridge_transform(training['observations'], variant, cfg)], noise_constant)
        if variant in ('noise_estimate', 'combined'):
            config['model']['observation_scale'] = np.maximum(estimated, base['model']['observation_scale']).tolist()
        for w in cfg['curve']['fixed_w']:
            try:
                p, c = step5.localization.model(config, 'W', w)
                inferred = step5.joint.smooth_balloon_joint(y, p, config=c, quadrature_order=cfg['curve']['quadrature_order'])
                truth, mean, var = step5.targets_and_moments(generated, inferred)
                transformed_truth = np.column_stack((generated['states'][:, 0], clean))
                rows.append(dict(replicate=replicate, variant=variant, w=w, execution='completed',
                    parameter_log_likelihood=inferred.parameter_log_likelihood,
                    model_truth=step5.truth_metrics(truth, mean, var),
                    processed_truth=step5.truth_metrics(transformed_truth, mean, var),
                    observation_coverage95=np.mean(abs(y-inferred.trajectory_mean) <= 1.95996398454*np.sqrt(inferred.total_observation_variance), axis=0),
                    noise_estimate=estimated, effective_noise_scale=config['model']['observation_scale'],
                    physical_checks=inferred.physical_checks))
            except Exception as exc:
                rows.append(dict(replicate=replicate, variant=variant, w=w, execution='failed', error=repr(exc), traceback=traceback.format_exc()))
    return dict(replicate=replicate, rows=rows, seeds=[cfg['seed']+1000+replicate, cfg['seed']+2000+replicate])


def fold_examples(trials, train, validation, coordinate, modality, cfg, base, measured):
    projection = fit_projection(trials, train, base, measured)
    targets = {i: project_trial(trials[i], projection, coordinate) for i in set(train)|set(validation)}
    normalizer = np.maximum(np.std(np.concatenate([targets[i] for i in train]), axis=0), 1e-8)
    blocks = {}
    for name, indices in [('train', train), ('validation', validation)]:
        examples = []
        for i in indices:
            # A training example's task template also excludes its own trial.
            peers = [j for j in train if trials[j]['session'] == trials[i]['session'] and j != i]
            if not peers:
                raise ValueError('no independent same-session training task template')
            template = np.mean([targets[j] for j in peers], axis=0)
            masked = project_trial(trials[i], projection, coordinate, 'center_'+modality)
            x0, x1, hidden, cols = linear_features(masked, template, modality, cfg)
            donor = targets[peers[trials[i]['ordinal'] % len(peers)]]
            shifted = np.roll(targets[i], round(len(masked)*cfg['linear']['circular_shift_fraction']), axis=0)
            _, xn, _, _ = linear_features(masked, template, modality, cfg, donor)
            _, xs, _, _ = linear_features(masked, template, modality, cfg, shifted)
            examples.append(dict(trial=i, basic=x0, joint=x1, independent_pairing=xn, circular_shift=xs,
                target=targets[i][hidden][:, cols], normalization=normalizer[cols]))
        blocks[name] = examples
    return blocks, projection


def cv_fold_job(cfg, base, measured, trials, coordinate, modality, fold):
    all_indices = list(range(len(trials)))
    train = [i for i in all_indices if trials[i]['ordinal'] % cfg['linear']['outer_folds'] != fold]
    validation = [i for i in all_indices if i not in train]
    alphas = cfg['linear']['ridge_alphas']
    losses = {kind: np.zeros(len(alphas)) for kind in ('basic', 'joint')}
    for inner in range(cfg['linear']['inner_folds']):
        # Reindex the remaining trials in each session before assigning inner folds.
        ordinal = {i: j for session in sorted({trials[i]['session'] for i in train})
                   for j, i in enumerate([k for k in train if trials[k]['session'] == session])}
        inner_train = [i for i in train if ordinal[i] % cfg['linear']['inner_folds'] != inner]
        inner_val = [i for i in train if i not in inner_train]
        blocks, _ = fold_examples(trials, inner_train, inner_val, coordinate, modality, cfg, base, measured)
        y = np.concatenate([v['target'] for v in blocks['train']])
        for kind in losses:
            x = np.concatenate([v[kind] for v in blocks['train']])
            for a, alpha in enumerate(alphas):
                model = ridge_fit(x, y, alpha)
                losses[kind][a] += np.mean([np.mean(((ridge_predict(model, v[kind])-v['target'])/v['normalization'])**2)
                                           for v in blocks['validation']])
    selected = {kind: alphas[int(np.argmin(value))] for kind, value in losses.items()}
    blocks, projection = fold_examples(trials, train, validation, coordinate, modality, cfg, base, measured)
    y = np.concatenate([v['target'] for v in blocks['train']])
    models = {kind: ridge_fit(np.concatenate([v[kind] for v in blocks['train']]), y, selected[kind]) for kind in losses}
    rows = []
    for example in blocks['validation']:
        score = {}
        for kind in ('basic', 'joint', 'independent_pairing', 'circular_shift'):
            prediction = ridge_predict(models['basic' if kind == 'basic' else 'joint'], example[kind])
            score[kind] = -float(np.mean(((prediction-example['target'])/example['normalization'])**2))
        rows.append(dict(trial=example['trial'], scores=score,
                         increments={k: score['joint']-score[k] for k in ('basic', 'independent_pairing', 'circular_shift')}))
    return dict(coordinate=coordinate, modality=modality, fold=fold, train=train, validation=validation,
                selected_alphas=selected, inner_losses=losses, projection=projection, rows=rows)


def guarded_job(kind, arguments):
    try:
        function = {'bridge': bridge_job, 'curve': curve_job, 'linear': cv_fold_job}[kind]
        return dict(execution='completed', result=function(*arguments))
    except Exception as exc:
        return dict(execution='failed', error=repr(exc), traceback=traceback.format_exc())


def execute_jobs(jobs, run_dir, workers):
    results = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        futures = {pool.submit(guarded_job, kind, args): key for key, kind, args in jobs}
        for future in as_completed(futures):
            key = futures[future]
            try:
                value = future.result()
            except Exception as exc:
                value = dict(execution='failed', error=repr(exc), traceback=traceback.format_exc())
            step5.write_json(run_dir / (key+'.json'), value)
            results[key] = value
            if len(results) % 12 == 0 or value['execution'] == 'failed':
                print(f'{len(results)}/{len(jobs)} {key}: {value["execution"]}', flush=True)
    return results


def summarize(cfg, bridge, measured, preparation):
    bridge_rows = [r for v in bridge.values() if v['execution'] == 'completed' for r in v['result']['rows']]
    bridge_summary = {}
    for variant in cfg['bridge']['variants']:
        subset = [r for r in bridge_rows if r['variant'] == variant]
        complete = [r for r in subset if r['execution'] == 'completed']
        by_w = {}
        for w in cfg['curve']['fixed_w']:
            rows = [r for r in complete if r['w'] == w]
            by_w[str(w)] = dict(completed=len(rows), model_truth={target: {
                metric: float(np.mean([r['model_truth'][target][metric] for r in rows]))
                for metric in ('nrmse', 'coverage95', 'correlation')}
                for target in step5.localization.TARGETS},
                processed_truth={target: {metric: float(np.mean([r['processed_truth'][target][metric] for r in rows]))
                    for metric in ('nrmse', 'coverage95')} for target in step5.localization.TARGETS},
                effective_noise_scale=np.mean([r['effective_noise_scale'] for r in rows], axis=0))
        deltas = []
        for replicate in range(cfg['bridge']['replicates']):
            pair = {r['w']: r for r in complete if r['replicate'] == replicate}
            if set(pair) == {0., -.5}:
                deltas.append(pair[-.5]['parameter_log_likelihood']-pair[0.]['parameter_log_likelihood'])
        bridge_summary[variant] = dict(expected=cfg['bridge']['replicates']*2, completed=len(complete), fixed_w=by_w,
            boundary_minus_truth_ll=cluster(deltas, cfg), boundary_preference_fraction=float(np.mean(np.array(deltas)>0)))
    curves, linear = {}, {}
    for key, value in measured.items():
        if value['execution'] != 'completed':
            continue
        if key.startswith('curve_'):
            _, subject, coordinate, modality, _ = key.split('__')
            curves.setdefault(subject, {}).setdefault(coordinate, {}).setdefault(modality, []).append(value['result'])
        elif key.startswith('linear_'):
            _, subject, coordinate, modality, _ = key.split('__')
            linear.setdefault(subject, {}).setdefault(coordinate, {}).setdefault(modality, []).append(value['result'])
    curve_summary = {}
    for subject, coordinates in curves.items():
        curve_summary[subject] = {}
        for coordinate, modalities in coordinates.items():
            curve_summary[subject][coordinate] = {}
            for modality, rows in modalities.items():
                rows.sort(key=lambda r: r['w'])
                fixed = {r['w']: r for r in rows if r['w'] in cfg['curve']['fixed_w']}
                curve_summary[subject][coordinate][modality] = dict(rows=rows,
                    expected_grid_points=cfg['curve']['grid_points'], completed_grid_points=len(rows),
                    likelihood_maximum_w=max(rows, key=lambda r: r['parameter_log_likelihood'])['w'],
                    boundary_minus_zero_ll=(fixed[-.5]['parameter_log_likelihood']-fixed[0.]['parameter_log_likelihood']) if len(fixed)==2 else None,
                    segment_boundary_minus_zero_ll={s: fixed[-.5]['chronological_segment_log_likelihood'][s]-fixed[0.]['chronological_segment_log_likelihood'][s]
                                                   for s in cfg['curve']['segments_seconds']} if len(fixed)==2 else {})
    linear_summary = {}
    for coordinate in cfg['coordinates']:
        linear_summary[coordinate] = {}
        for modality in ('EEG', 'fNIRS'):
            subject_values, counts = {}, {}
            for subject in cfg['subjects']:
                folds = linear.get(subject, {}).get(coordinate, {}).get(modality, [])
                rows = [r for fold in folds for r in fold['rows']]
                counts[subject] = len(rows)
                if len(rows) == 24 and len({r['trial'] for r in rows}) == 24:
                    subject_values[subject] = {control: float(np.mean([r['increments'][control] for r in rows]))
                                              for control in ('basic', 'independent_pairing', 'circular_shift')}
            linear_summary[coordinate][modality] = dict(trial_counts=counts, subject_increments=subject_values,
                increments={control: cluster([v[control] for v in subject_values.values()], cfg)
                            for control in ('basic', 'independent_pairing', 'circular_shift')},
                interpretation='descriptive nested CV within original training inventory; three subject clusters')
    failures = {k: v for k, v in {**bridge, **measured}.items() if v['execution'] != 'completed'}
    failures.update({f'bridge_case_{i}': r for i, r in enumerate(bridge_rows) if r['execution'] != 'completed'})
    quadrature = {k: v for k, v in measured.items() if k.startswith('quadrature_')}
    return dict(schema=cfg['schema'], bridge=bridge_summary, curves=curve_summary, linear=linear_summary,
                preparation=preparation, failures=failures, quadrature_checks=quadrature,
                teacher_qualification='not_evaluated', comprehensive_uq='not_executed')


def cluster(values, cfg):
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return dict(mean=None, ci95=None, independent_units=0)
    rng = np.random.default_rng(cfg['seed'])
    boot = values[rng.integers(0, len(values), (cfg['linear']['bootstrap_repetitions'], len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), ci95=np.quantile(boot, [.025, .975]), independent_units=len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=('bridge', 'all'), default='all')
    args = parser.parse_args()
    cfg, base, measured, metadata = load_config(args.config)
    run_dir = args.run_dir.resolve()
    if run_dir.parent != (ROOT/base['output_root']).resolve():
        raise ValueError('use a fresh direct child of the existing Step5 artifact root')
    run_dir.mkdir(exist_ok=False)
    started = time.time()
    source_hashes = {p: digest(ROOT/p) for p in SOURCE_FILES}
    manifest = dict(schema=cfg['schema'], execution='running', stage=args.stage,
        started_at=datetime.now(timezone.utc).isoformat(), source_sha256=source_hashes,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_worktree=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        config_sha256=digest(args.config), software=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        subjects=cfg['subjects'] if args.stage=='all' else [], original_heldout_trials_processed=0)
    step5.write_json(run_dir/'manifest.json', manifest)
    for name, value in [('resolved_config.yaml', cfg), ('resolved_base.yaml', base), ('resolved_measured.yaml', measured)]:
        (run_dir/name).write_text(yaml.safe_dump(value, sort_keys=False))
    (run_dir/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for source in ('src/inference/observation_baselines.py', 'src/metrics/trajectory_reliability.py'):
        (run_dir/(Path(source).stem+'_snapshot.py')).write_bytes((ROOT/source).read_bytes())
    constant = student_difference_mad(base['model']['student_nu'])
    bridge = execute_jobs([(f'bridge_{i:02d}', 'bridge', (cfg, base, i, constant))
                           for i in range(cfg['bridge']['replicates'])], run_dir, cfg['workers'])
    measured_results, preparation = {}, {}
    if args.stage == 'all':
        from experiments.evaluate_t3_multisession_loso import _validate_metadata
        metadata_summary, inventory, hashes = _validate_metadata(metadata)
        step5.write_json(run_dir/'metadata_boundary.json', dict(summary=metadata_summary, hashes=hashes,
            selected_inventory=[r for r in inventory if r.get('subject_id', r.get('subject')) in cfg['subjects']]))
        jobs = []
        grid = np.linspace(*cfg['curve']['bounds'], cfg['curve']['grid_points'])
        for subject in cfg['subjects']:
            try:
                trials, detail = load_training_subject(subject, cfg, base, measured, metadata)
                projection = fit_projection(trials, list(range(24)), base, measured)
                detail['projection'] = projection
                preparation[subject] = dict(execution='completed', trials=24,
                    selected_fnirs_pair=detail['fnirs_pairs'][projection['fnirs_pair']],
                    pca_explained_fraction=projection['pca_explained_fraction'])
                step5.write_json(run_dir/f'prepared_{subject}.json', detail)
                arrays = {}
                for coordinate in cfg['coordinates']:
                    observations = np.array([project_trial(t, projection, coordinate) for t in trials])
                    arrays[coordinate] = observations
                    config = copy.deepcopy(base)
                    config['model']['observation_scale'] = np.maximum(first_difference_noise(observations, constant), base['model']['observation_scale']).tolist()
                    preparation[subject][coordinate+'_noise_scale'] = config['model']['observation_scale']
                    for modality in cfg['curve']['modalities']:
                        for i, w in enumerate(grid):
                            jobs.append((f'curve__{subject}__{coordinate}__{modality}__{i:02d}', 'curve',
                                         (config, observations, modality, float(w), cfg)))
                        for w in cfg['curve']['fixed_w']:
                            small = observations[cfg['curve']['quadrature_check_trial_positions']]
                            for order in (13, 17):
                                jobs.append((f'quadrature__{subject}__{coordinate}__{modality}__{w}__{order}', 'curve',
                                             (config, small, modality, w, cfg, order)))
                    for modality in ('EEG', 'fNIRS'):
                        for fold in range(cfg['linear']['outer_folds']):
                            jobs.append((f'linear__{subject}__{coordinate}__{modality}__{fold}', 'linear',
                                (cfg, base, measured, trials, coordinate, modality, fold)))
                np.savez_compressed(run_dir/f'prepared_{subject}.npz', **arrays)
                print(f'Prepared {subject}: 24 original training trials, {preparation[subject]["selected_fnirs_pair"]}', flush=True)
            except Exception as exc:
                preparation[subject] = dict(execution='failed', error=repr(exc), traceback=traceback.format_exc())
                step5.write_json(run_dir/f'preparation_failure_{subject}.json', preparation[subject])
                print(f'Preparation failure {subject}: {exc}', flush=True)
        measured_results = execute_jobs(jobs, run_dir, cfg['workers'])
    summary = summarize(cfg, bridge, measured_results, preparation)
    step5.write_json(run_dir/'summary.json', summary)
    manifest.update(execution='completed', elapsed_seconds=time.time()-started,
                    finished_at=datetime.now(timezone.utc).isoformat(), failures=len(summary['failures']),
                    preparation_failures=sum(v['execution'] != 'completed' for v in preparation.values()))
    step5.write_json(run_dir/'manifest.json', manifest)
    print(json.dumps(dict(run_dir=str(run_dir), failures=manifest['failures'], preparation_failures=manifest['preparation_failures'])), flush=True)


if __name__ == '__main__':
    main()
