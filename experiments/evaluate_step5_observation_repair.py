#!/usr/bin/env python3
"""Observation contract repairs: bounded regression, temporal reference and replay.

Old configurations and evidence are immutable. Measured access reuses the one
training-only diagnostic loader; no held-out or protected trial is processed.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from unittest.mock import patch

for _thread_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_var] = '1'

import numpy as np
import scipy
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import evaluate_step5_observation_diagnostic as diagnostic
from src.inference.observation_baselines import (
    first_difference_noise, student_difference_mad, bridge_transform,
)

step5 = diagnostic.step5
core, joint = step5.core, step5.joint
DEFAULT_CONFIG = ROOT / 'experiments/configs/physiology_semantic_tokenizer/step5_observation_repair_v1.yaml'
MASK_CONFIG = DEFAULT_CONFIG.with_name('step5_observation_repair_v2.yaml')
SOURCE_FILES = tuple(dict.fromkeys((*diagnostic.SOURCE_FILES,
    'experiments/evaluate_step5_observation_repair.py',
    'experiments/configs/physiology_semantic_tokenizer/step5_observation_repair_v1.yaml')))


def load_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['schema'] == 'step5_observation_repair_v2':
        return load_mask_repair_config(cfg)
    if cfg['schema'] != 'step5_observation_repair_v1' or cfg['scope'] != 'observation_contract_regression_training_only':
        raise ValueError('repair schema/scope mismatch')
    if cfg['diagnostic_config'] != str(diagnostic.DEFAULT_CONFIG.relative_to(ROOT)):
        raise ValueError('training diagnostic owner changed')
    dc, base, measured, metadata = diagnostic.load_config(ROOT/cfg['diagnostic_config'])
    if cfg['previous_run'] != base['output_root']+'/20260908_observation_diagnostic_v1':
        raise ValueError('replay must use the specified original diagnostic evidence')
    if (cfg['teacher_qualification'] != 'none' or cfg['measured']['measurement_gain_candidate'] != 'none'
            or cfg['measured']['coordinate'] != 'broadband_pca' or cfg['measured']['fixed_w'] != 0.
            or cfg['measured']['outer_folds'] != dc['linear']['outer_folds']):
        raise ValueError('bounded measured comparison changed')
    if (cfg['scale']['diagonal'] != [1., .5, .5] or cfg['scale']['grid_points'] != 17
            or cfg['scale']['reused_replicates'] != 24 or cfg['scale']['independent_replicates'] != 24
            or cfg['trajectory']['steps'] != 64 or cfg['trajectory']['replicates_per_law'] != 24
            or cfg['trajectory']['variants'] != ['model', 'baseline', 'fnirs_filter', 'combined']
            or cfg['trajectory']['laws'] != ['linearized_gaussian', 'nonlinear_student_t']
            or cfg['replay']['coordinates'] != dc['coordinates']
            or cfg['replay']['modalities'] != dc['curve']['modalities']
            or cfg['replay']['grid_points'] != dc['curve']['grid_points']):
        raise ValueError('frozen repair panel changed')
    return cfg, dc, base, measured, metadata


def load_mask_repair_config(cfg):
    if (cfg['scope'] != 'flow_domain_and_mask_specific_feature_bridge'
            or cfg['diagnostic_config'] != str(diagnostic.DEFAULT_CONFIG.relative_to(ROOT))):
        raise ValueError('mask repair scope/diagnostic owner changed')
    dc, base, measured, metadata = diagnostic.load_config(ROOT/cfg['diagnostic_config'])
    if (cfg['previous_run'] != base['output_root']+'/20260908_observation_diagnostic_v1'
            or cfg['flow']['trials'] != [['subject_01', 4], ['subject_01', 7], ['subject_09', 12]]
            or cfg['flow']['coordinate'] != 'broadband_pca' or cfg['flow']['modality'] != 'fNIRS_only'
            or cfg['flow']['fixed_w'] != [0., -.5] or cfg['flow']['quadrature_order'] != 13
            or cfg['flow']['update_history'] != 4):
        raise ValueError('three-trial training replay boundary changed')
    t = cfg['trajectory']
    if (t['steps'] != 64 or t['center_steps'] != 16 or t['context_check_steps'] != [64, 120]
            or t['replicates_per_law'] != 24 or t['fixed_w'] != [0., -.5]
            or t['laws'] != ['linearized_gaussian', 'nonlinear_student_t']
            or t['variants'] != ['model', 'combined']
            or t['masks'] != ['full', 'center_EEG', 'center_fNIRS', 'whole_EEG', 'whole_fNIRS']
            or t['branches'] != ['pointwise_student_t', 'mask_specific_gaussian_reference']
            or cfg['teacher_qualification'] != 'none'
            or cfg['measured']['comparison'] != 'not_executable_without_nonlinear_student_t_temporal_validation'):
        raise ValueError('bounded mask bridge contract changed')
    if (not 0 < t['rank_rtol'] < 1 or any(not 0 < x < 1 for x in t['rank_check_rtols'])
            or not isinstance(cfg['workers'], int) or not 1 <= cfg['workers'] <= 6):
        raise ValueError('invalid rank tolerance or bounded worker count')
    return cfg, dc, base, measured, metadata


def maximum_error(a, b):
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))


def scale_job(cfg, dc, base, cohort, replicate):
    seed = dc['seed']+1000+replicate if cohort == 'reused' else cfg['scale']['independent_seed_start']+replicate
    generated = step5.localization.generate_matched(base, 'W', 0., seed)
    grid = np.linspace(*base['axes']['W']['bounds'], cfg['scale']['grid_points'])
    diagonal = np.array(cfg['scale']['diagonal'])
    fits, rows = {}, {}
    errors = dict(log_likelihood=0., transformed_mean=0., transformed_covariance=0.,
                  physical_mean=0., physical_covariance=0., canonical_clean_mean=0., canonical_clean_variance=0.)
    for branch in ('original', 'observation_only', 'synchronized'):
        fits[branch], rows[branch] = [], []
        for w in grid:
            p, c = step5.localization.model(base, 'W', float(w))
            spec = core.BalloonObservationSpec().resolved(p.fixed)
            y = generated['observations']
            if branch != 'original':
                y = y*diagonal
            if branch == 'synchronized':
                spec = spec.reexpress(diagonal)
            result = joint.smooth_balloon_joint(y, p, config=c, observation_spec=spec,
                                              quadrature_order=dc['curve']['quadrature_order'])
            fits[branch].append(result)
            rows[branch].append(result.parameter_log_likelihood)
            if branch == 'synchronized':
                original = fits['original'][len(fits[branch])-1]
                values = dict(
                    log_likelihood=abs(result.parameter_log_likelihood+spec.log_abs_det(np.isfinite(y))-original.parameter_log_likelihood),
                    transformed_mean=maximum_error(result.transformed_mean, original.transformed_mean),
                    transformed_covariance=maximum_error(result.transformed_covariance, original.transformed_covariance),
                    physical_mean=maximum_error(result.state_mean, original.state_mean),
                    physical_covariance=maximum_error(result.state_covariance, original.state_covariance),
                    canonical_clean_mean=maximum_error(result.trajectory_mean/diagonal, original.trajectory_mean),
                    canonical_clean_variance=maximum_error(result.state_posterior_variance/diagonal**2, original.state_posterior_variance))
                errors = {k: max(errors[k], values[k]) for k in errors}
    posteriors = {branch: step5.localization.posterior_grid(base, 'W', grid, ll)
                  for branch, ll in rows.items()}
    errors['parameter_cdf'] = maximum_error(posteriors['original']['cdf'], posteriors['synchronized']['cdf'])
    mixtures, metrics = {}, {}
    truth = np.column_stack((generated['states'][:, 0], generated['clean']))
    for branch, results in fits.items():
        means = np.array([r.state_mean for r in results])
        weights = posteriors[branch]['weights']
        mixture = np.einsum('k,kti->ti', weights, means)
        variance = np.einsum('k,kti->ti', weights, np.array([
            np.diagonal(r.state_covariance, axis1=1, axis2=2) for r in results])+(means-mixture)**2)
        mixtures[branch] = (mixture, variance)
        result = results[int(np.flatnonzero(grid == 0.)[0])]
        d = diagonal if branch == 'synchronized' else np.ones(3)
        mean = np.column_stack((result.state_mean[:, 0], result.trajectory_mean/d))
        var = np.column_stack((result.state_covariance[:, 0, 0], result.state_posterior_variance/d**2))
        processed_truth = truth.copy()
        if branch != 'original':
            processed_truth[:, 1:] *= diagonal
        declared_mean = np.column_stack((result.state_mean[:, 0], result.trajectory_mean))
        declared_var = np.column_stack((result.state_covariance[:, 0, 0], result.state_posterior_variance))
        metrics[branch] = dict(model_truth=step5.truth_metrics(truth, mean, var),
            processed_truth=step5.truth_metrics(processed_truth, declared_mean, declared_var),
            boundary_minus_zero_ll=rows[branch][0]-rows[branch][len(grid)//2])
    errors['mixture_physical_mean'] = maximum_error(mixtures['original'][0], mixtures['synchronized'][0])
    errors['mixture_physical_variance'] = maximum_error(mixtures['original'][1], mixtures['synchronized'][1])
    passed = errors['log_likelihood'] <= cfg['scale']['likelihood_absolute_tolerance'] and all(
        v <= cfg['scale']['posterior_absolute_tolerance'] for k, v in errors.items() if k != 'log_likelihood')
    return dict(cohort=cohort, replicate=replicate, seed=seed, grid=grid, log_likelihood=rows,
                posterior_cdf={k: v['cdf'] for k, v in posteriors.items()}, metrics=metrics,
                invariance_errors=errors, passed=passed)


def trajectory_operator(steps, variant, dc):
    """Construct the actual finite-window bridge operator by linear impulses."""
    operators = np.empty((3, steps, steps))
    for channel in range(3):
        for t in range(steps):
            impulse = np.zeros((steps, 3)); impulse[t, channel] = 1.
            operators[channel, :, t] = bridge_transform(impulse, variant, dc)[:, channel]
    b = dc['bridge']
    processing = []
    if variant == 'combined':
        processing.append(dict(operation='resample_poly_roundtrip', hz=[4., 10., 4.],
                               boundary='scipy default constant zero padding; crop original length', channels=['HbO', 'HbR']))
    if variant in ('fnirs_filter', 'combined'):
        processing.append(dict(operation='bandpass_fnirs', hz=b['filter_hz'], order=b['filter_order'],
                               boundary='existing bandpass_fnirs finite-window sosfiltfilt', channels=['HbO', 'HbR']))
    if variant in ('baseline', 'combined'):
        weights = np.zeros(steps); weights[:b['baseline_samples']] = 1/b['baseline_samples']
        processing.append(dict(operation='subtract_weighted_baseline', weights=weights.tolist(), channels=list(core.OBSERVATION_NAMES)))
    if variant == 'combined':
        processing.append(dict(operation='known_common_scale', factor=b['common_fnirs_factor'], channels=['HbO', 'HbR']))
    processing.append(dict(input_units='canonical model observation', output_units='processed model observation',
                           mask='discard outputs depending on hidden pre-processing inputs'))
    clock = np.arange(steps)/b['model_hz']-5.
    return joint.TrajectoryObservationSpec(operators, clock, clock, tuple(processing))


def generate_linear_gaussian(base, steps, seed):
    p, c = step5.localization.model(base, 'W', 0.)
    _, a = core.rk4_transition_with_jacobian(np.zeros(6), p, c)
    rng = np.random.default_rng(seed)
    z = np.empty((steps, 6)); z[0] = rng.normal(size=6)*c.initial_state_std
    for t in range(1, steps):
        z[t] = a@z[t-1]+rng.normal(size=6)*np.array(p.fixed.process_std)*np.sqrt(c.dt)
    h = core.observation_jacobian(np.zeros(6), p)
    clean = z@h.T
    variance = np.square(p.fixed.observation_scale)*p.fixed.student_nu/(p.fixed.student_nu-2)
    return dict(transformed_states=z, states=np.column_stack((z[:, :2], np.exp(z[:, 2:]))), clean=clean,
                observations=clean+rng.normal(size=clean.shape)*np.sqrt(variance))


def trajectory_metrics(generated, result, operator):
    truth = np.column_stack((generated['states'][:, 0], generated['clean']))
    mean = np.column_stack((result['transformed_mean'][:, 0], result['canonical_clean_mean']))
    variance = np.column_stack((np.diag(result['transformed_covariance'])[::6],
                               np.diag(result['canonical_clean_covariance']).reshape(-1, 3)))
    processed_truth = np.column_stack((generated['states'][:, 0], operator.apply(generated['clean'])))
    processed_mean = np.column_stack((result['transformed_mean'][:, 0], result['processed_clean_mean']))
    processed_variance = np.column_stack((variance[:, 0], np.diag(result['processed_clean_covariance']).reshape(-1, 3)))
    return dict(model_truth=step5.truth_metrics(truth, mean, variance),
                processed_truth=step5.truth_metrics(processed_truth, processed_mean, processed_variance))


def trajectory_job(cfg, dc, base, law, replicate):
    count = cfg['trajectory']['steps']
    if law == 'linearized_gaussian':
        seed = cfg['trajectory']['gaussian_seed_start']+replicate
        generated = generate_linear_gaussian(base, count, seed)
    else:
        seed = dc['seed']+1000+replicate
        generated = {k: v[:count] for k, v in step5.localization.generate_matched(base, 'W', 0., seed).items()}
    identity = trajectory_operator(count, 'model', dc)
    rows, reference, checks = [], {}, []
    for variant in cfg['trajectory']['variants']:
        operator = trajectory_operator(count, variant, dc)
        y = operator.apply(generated['observations'])
        for w in cfg['trajectory']['fixed_w']:
            p, c = step5.localization.model(base, 'W', w)
            noise = np.square(p.fixed.observation_scale)*p.fixed.student_nu/(p.fixed.student_nu-2)
            for branch in (('synchronized',) if variant == 'model' else ('observation_only', 'synchronized')):
                inference_operator = operator if branch == 'synchronized' else identity
                try:
                    result = joint.smooth_balloon_trajectory_reference(y, p, config=c,
                        trajectory_spec=inference_operator, noise_variance=noise, rank_rtol=cfg['trajectory']['rank_rtol'])
                    if variant == 'model':
                        reference[w] = result
                    metrics = trajectory_metrics(generated, result, operator)
                    rows.append(dict(variant=variant, branch=branch, w=w, execution='completed',
                        parameter_log_likelihood=result['parameter_log_likelihood'], metrics=metrics,
                        retained_rank=result['retained_rank'], observed_coordinates=result['observed_coordinates'],
                        discarded_singular_values=result['discarded_singular_values'], support_residual_norm=result['support_residual_norm'],
                        minimum_state_variance_change=float(np.min(np.diag(result['transformed_covariance']-reference[w]['transformed_covariance'])))))
                    if (variant in ('fnirs_filter', 'combined') and branch == 'synchronized'
                            and replicate in cfg['trajectory']['rank_check_replicates']):
                        for rtol in cfg['trajectory']['rank_check_rtols']:
                            alternate = joint.smooth_balloon_trajectory_reference(y, p, config=c,
                                trajectory_spec=operator, noise_variance=noise, rank_rtol=rtol)
                            checks.append(dict(variant=variant, w=w, rank_rtol=rtol, rank=alternate['retained_rank'],
                                maximum_transformed_mean_change=maximum_error(alternate['transformed_mean'], result['transformed_mean']),
                                maximum_transformed_variance_change=maximum_error(np.diag(alternate['transformed_covariance']), np.diag(result['transformed_covariance'])),
                                log_likelihood_change=alternate['parameter_log_likelihood']-result['parameter_log_likelihood']))
                except Exception as exc:
                    rows.append(dict(variant=variant, branch=branch, w=w, execution='failed', error=repr(exc), traceback=traceback.format_exc()))
    approximation = None
    if law == 'nonlinear_student_t':
        p, c = step5.localization.model(base, 'W', 0.)
        nonlinear = joint.smooth_balloon_joint(generated['observations'], p, config=c)
        approximation = dict(unprocessed_driver_mean_max_difference=maximum_error(reference[0.]['transformed_mean'][:, 0], nonlinear.state_mean[:, 0]),
            unprocessed_log_likelihood_difference=reference[0.]['parameter_log_likelihood']-nonlinear.parameter_log_likelihood,
            meaning='linearized/Gaussian approximation discrepancy before time mixing; not a convergence certificate')
    return dict(law=law, replicate=replicate, seed=seed, rows=rows, rank_checks=checks, approximation_check=approximation)


def traced_trial(y, config, w, order, extreme_threshold):
    """Record internal RK4 evaluation extremes, including completed saturation."""
    extraction = core._extraction
    diagnostic_values = dict(minimum_evaluated_flow=None, saturation_evaluations=0,
                             extreme_flow_evaluations=0, minimum_log_complement=None)
    def traced(f, e0):
        previous = diagnostic_values['minimum_evaluated_flow']
        diagnostic_values['minimum_evaluated_flow'] = float(f) if previous is None else min(previous, float(f))
        diagnostic_values['extreme_flow_evaluations'] += int(f < extreme_threshold)
        result = extraction(f, e0)
        log_complement = core.extraction_log_complement(f, e0)
        previous = diagnostic_values['minimum_log_complement']
        diagnostic_values['minimum_log_complement'] = log_complement if previous is None else min(previous, log_complement)
        diagnostic_values['saturation_evaluations'] += int(result[0] == 1.)
        return result
    try:
        with patch.object(core, '_extraction', traced):
            trace = diagnostic.filter_trace(y, config, w, order, w in (0., -.5))
        return dict(execution='completed', trace=trace, numerical=diagnostic_values)
    except Exception as exc:
        return dict(execution='failed', error=repr(exc), traceback=traceback.format_exc(), numerical=diagnostic_values)


def replay_curve_job(cfg, dc, config, observations, subject, coordinate, modality, w):
    y = observations.copy()
    if modality == 'EEG_only':
        y[:, :, 1:] = np.nan
    elif modality == 'fNIRS_only':
        y[:, :, 0] = np.nan
    trials = [traced_trial(row, config, w, dc['curve']['quadrature_order'], cfg['replay']['extreme_flow_threshold']) for row in y]
    completed = [t for t in trials if t['execution'] == 'completed']
    complete = len(completed) == len(trials)
    increments = np.array([t['trace']['increments'] for t in completed])
    result = dict(subject=subject, coordinate=coordinate, modality=modality, w=w,
        curve_execution='completed' if complete else 'failed', expected_trials=len(trials), completed_trials=len(completed),
        parameter_log_likelihood=float(increments.sum()) if complete else None,
        trials=[{k: v for k, v in t.items() if k != 'trace'} | dict(trial=i,
            parameter_log_likelihood=float(t['trace']['increments'].sum()) if t['execution'] == 'completed' else None)
            for i, t in enumerate(trials)])
    if completed and w in (0., -.5):
        result['standardized_predictive_residuals'] = diagnostic.residual_summary(
            [t['trace']['standardized_predictive_residuals'] for t in completed])
        result['residual_summary_complete'] = complete
    return result


def load_replay_inputs(cfg, dc, base, subject, *, data_root=None):
    """Scope and exact training identities precede opening prepared arrays."""
    if subject not in dc['subjects'] or subject not in base['measured']['subjects']:
        raise ValueError('replay subject outside training boundary')
    data_root = ROOT if data_root is None else Path(data_root).resolve()
    previous = data_root/cfg['previous_run']
    detail = json.loads((previous/f'prepared_{subject}.json').read_text())
    identities = detail['trials']
    expected = {(session, position) for session in base['measured']['sessions']
                for position in range(10) if position not in base['measured']['heldout_trial_positions']}
    if (len(identities) != 24 or any(t['subject'] != subject for t in identities)
            or {(t['session'], t['original_ma_trial_position']) for t in identities} != expected):
        raise ValueError('prepared replay identity crosses original training boundary')
    path = previous/f'prepared_{subject}.npz'
    with np.load(path, allow_pickle=False) as archive:
        arrays = {coordinate: archive[coordinate].copy() for coordinate in dc['coordinates']}
    if any(y.shape != (24, 120, 3) or not np.isfinite(y).all() for y in arrays.values()):
        raise ValueError('prepared replay array support changed')
    np.testing.assert_array_equal(arrays['broadband_pca'][:, :, 1:], arrays['local_F3_alpha'][:, :, 1:])
    return arrays, dict(input_sha256=diagnostic.digest(path), identity_sha256=diagnostic.digest(previous/f'prepared_{subject}.json'), trials=identities)


def paired_cv_job(cfg, dc, base, measured, trials, subject, modality, fold, noise_constant):
    coordinate = cfg['measured']['coordinate']
    linear = diagnostic.cv_fold_job(dc, base, measured, trials, coordinate, modality, fold)
    train, validation, projection = linear['train'], linear['validation'], linear['projection']
    targets = {i: diagnostic.project_trial(trials[i], projection, coordinate) for i in train+validation}
    config = copy.deepcopy(base)
    config['model']['observation_scale'] = np.maximum(
        first_difference_noise([targets[i] for i in train], noise_constant), base['model']['observation_scale']).tolist()
    normalization = np.maximum(np.std(np.concatenate([targets[i] for i in train]), axis=0), 1e-8)
    p, c = step5.localization.model(config, 'W', cfg['measured']['fixed_w'])
    cols, source = ([0], [1, 2]) if modality == 'EEG' else ([1, 2], [0])
    rows = []
    for i, linear_row in zip(validation, linear['rows']):
        if linear_row['trial'] != i:
            raise ValueError('linear/SSM fold trial mismatch')
        masked = diagnostic.project_trial(trials[i], projection, coordinate, 'center_'+modality)
        hidden = np.isnan(masked[:, cols[0]])
        peers = [j for j in train if trials[j]['session'] == trials[i]['session']]
        donor = targets[peers[trials[i]['ordinal'] % len(peers)]]
        shifted = np.roll(targets[i], round(len(masked)*dc['linear']['circular_shift_fraction']), axis=0)
        results = {}
        for condition in ('own_context', 'joint', 'independent_pairing', 'circular_shift'):
            y = masked.copy()
            if condition == 'own_context':
                y[:, source] = np.nan
            elif condition != 'joint':
                y[:, source] = (donor if condition == 'independent_pairing' else shifted)[:, source]
            try:
                result = joint.smooth_balloon_joint(y, p, config=c, quadrature_order=dc['curve']['quadrature_order'])
                error = (result.trajectory_mean[hidden][:, cols]-targets[i][hidden][:, cols])/normalization[cols]
                results[condition] = dict(execution='completed', score=-float(np.mean(error**2)),
                    physical_checks=result.physical_checks,
                    observation_coverage95=float(np.mean(abs(result.trajectory_mean[hidden][:, cols]-targets[i][hidden][:, cols])
                        <= 1.95996398454*np.sqrt(result.total_observation_variance[hidden][:, cols]))))
            except Exception as exc:
                results[condition] = dict(execution='failed', error=repr(exc), traceback=traceback.format_exc())
        complete = all(r['execution'] == 'completed' for r in results.values())
        increments = None
        if complete:
            joint_score = results['joint']['score']
            increments = {k: joint_score-results[k]['score'] for k in ('own_context', 'independent_pairing', 'circular_shift')}
            increments['linear_basic'] = joint_score-linear_row['scores']['basic']
            increments['linear_joint'] = joint_score-linear_row['scores']['joint']
        rows.append(dict(trial=i, ssm=results, linear=linear_row, increments=increments, complete=complete))
    return dict(subject=subject, modality=modality, fold=fold, train=train, validation=validation,
        coordinate=coordinate, projection=projection, observation_scale=config['model']['observation_scale'],
        selected_linear_alphas=linear['selected_alphas'], rows=rows)


def guarded_job(kind, arguments):
    try:
        result = dict(scale=scale_job, trajectory=trajectory_job, replay=replay_curve_job,
                      paired=paired_cv_job, mask_bridge=mask_bridge_job)[kind](*arguments)
        return dict(execution='completed', result=result)
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
            step5.write_json(run_dir/(key+'.json'), value)
            results[key] = value
            if len(results) % 12 == 0 or value['execution'] != 'completed':
                print(f'{len(results)}/{len(jobs)} {key}: {value["execution"]}', flush=True)
    return results


def summarize(cfg, dc, results, old_summary=None):
    failures = {key: value for group in results.values() for key, value in group.items() if value['execution'] != 'completed'}
    completed = {key: value['result'] for group in results.values() for key, value in group.items() if value['execution'] == 'completed'}
    scale = {}
    for cohort in ('reused', 'independent'):
        rows = [v for k, v in completed.items() if k.startswith('scale__'+cohort+'__')]
        scale[cohort] = dict(expected=cfg['scale'][cohort+'_replicates'], completed=len(rows),
            passed=sum(r['passed'] for r in rows),
            maximum_errors={k: max(r['invariance_errors'][k] for r in rows) for k in rows[0]['invariance_errors']} if rows else {},
            branches={branch: dict(
                boundary_minus_zero_ll=diagnostic.cluster([r['metrics'][branch]['boundary_minus_zero_ll'] for r in rows], dc),
                slow_preference_count=sum(r['metrics'][branch]['boundary_minus_zero_ll'] > 0 for r in rows),
                model_truth={target: {metric: float(np.mean([r['metrics'][branch]['model_truth'][target][metric] for r in rows]))
                    for metric in ('coverage95', 'nrmse')} for target in step5.localization.TARGETS},
                processed_truth={target: {metric: float(np.mean([r['metrics'][branch]['processed_truth'][target][metric] for r in rows]))
                    for metric in ('coverage95', 'nrmse')} for target in step5.localization.TARGETS})
                for branch in ('original', 'observation_only', 'synchronized')} if rows else {})
    trajectory = {}
    for law in cfg['trajectory']['laws']:
        cases = [v for k, v in completed.items() if k.startswith('trajectory__'+law+'__')]
        groups = {}
        for variant in cfg['trajectory']['variants']:
            for branch in (('synchronized',) if variant == 'model' else ('observation_only', 'synchronized')):
                rows = [r | dict(replicate=v['replicate']) for v in cases for r in v['rows'] if r['variant'] == variant and r['branch'] == branch]
                good = [r for r in rows if r['execution'] == 'completed']
                for r in rows:
                    if r['execution'] != 'completed':
                        failures[f'trajectory__{law}__{r["replicate"]}__{variant}__{branch}__{r["w"]}'] = r
                deltas = []
                for replicate in range(cfg['trajectory']['replicates_per_law']):
                    pair = {r['w']: r for r in good if r['replicate'] == replicate}
                    if set(pair) == {0., -.5}:
                        deltas.append(pair[-.5]['parameter_log_likelihood']-pair[0.]['parameter_log_likelihood'])
                fixed = [r for r in good if r['w'] == 0.]
                groups[variant+'__'+branch] = dict(expected=2*cfg['trajectory']['replicates_per_law'], completed=len(good),
                    boundary_minus_zero_ll=diagnostic.cluster(deltas, dc), slow_preference_count=sum(d > 0 for d in deltas),
                    retained_ranks=sorted({r['retained_rank'] for r in fixed}),
                    minimum_state_variance_change=min((r['minimum_state_variance_change'] for r in fixed), default=None),
                    model_truth={target: {metric: float(np.mean([r['metrics']['model_truth'][target][metric] for r in fixed]))
                        for metric in ('coverage95', 'nrmse')} for target in step5.localization.TARGETS} if fixed else {},
                    processed_truth={target: {metric: float(np.mean([r['metrics']['processed_truth'][target][metric] for r in fixed]))
                        for metric in ('coverage95', 'nrmse')} for target in step5.localization.TARGETS} if fixed else {})
        rank_checks = []
        for case in cases:
            for variant in ('fnirs_filter', 'combined'):
                for tolerance in cfg['trajectory']['rank_check_rtols']:
                    pair = {r['w']: r for r in case['rank_checks'] if r['variant'] == variant and r['rank_rtol'] == tolerance}
                    if set(pair) == {0., -.5}:
                        rank_checks.append(dict(replicate=case['replicate'], variant=variant, rank_rtol=tolerance,
                            rank=pair[0.]['rank'], maximum_transformed_mean_change=max(v['maximum_transformed_mean_change'] for v in pair.values()),
                            boundary_minus_zero_ll_change=pair[-.5]['log_likelihood_change']-pair[0.]['log_likelihood_change']))
        trajectory[law] = dict(expected_trials=cfg['trajectory']['replicates_per_law'], completed_trials=len(cases),
            groups=groups, rank_checks=rank_checks, approximation_checks=[r['approximation_check'] for r in cases if r['approximation_check']],
            interpretation='exact only for rest-linearized Gaussian contract on retained subspace; nonlinear Student-t use is a moment approximation')
    curve_groups = {}
    saturation_trials, extreme_trials = set(), set()
    minimum_flow = None
    for key, row in completed.items():
        if not key.startswith('replay__'):
            continue
        group = '__'.join([row['subject'], row['coordinate'], row['modality']])
        curve_groups.setdefault(group, []).append(row)
        for trial in row['trials']:
            numerical = trial['numerical']
            if numerical['saturation_evaluations']:
                saturation_trials.add((row['subject'], trial['trial']))
            if numerical['extreme_flow_evaluations']:
                extreme_trials.add((row['subject'], trial['trial']))
            f = numerical['minimum_evaluated_flow']
            if f is not None:
                minimum_flow = f if minimum_flow is None else min(minimum_flow, f)
        if row['curve_execution'] != 'completed':
            failures[key] = dict(subject=row['subject'], coordinate=row['coordinate'], modality=row['modality'], w=row['w'],
                failed_trials=[r for r in row['trials'] if r['execution'] != 'completed'])
    curves = {}
    for group, rows in curve_groups.items():
        rows.sort(key=lambda r: r['w'])
        good = [r for r in rows if r['curve_execution'] == 'completed']
        eeg_only = group.endswith('EEG_only')
        expected = 2 if eeg_only else cfg['replay']['grid_points']
        complete = len(good) == expected
        span = float(np.ptp([r['parameter_log_likelihood'] for r in good])) if good else None
        endpoints = {r['w']: r for r in good if r['w'] in (0., -.5)}
        curves[group] = dict(expected=expected, completed=len(good), log_likelihood_span=span,
            interpretation='incomplete_curve' if not complete else ('flat_no_W_information' if eeg_only else 'complete_diagnostic_curve'),
            likelihood_maximum_w=max(good, key=lambda r: r['parameter_log_likelihood'])['w'] if complete and not eeg_only else None,
            boundary_minus_zero_ll=endpoints[-.5]['parameter_log_likelihood']-endpoints[0.]['parameter_log_likelihood'] if len(endpoints) == 2 else None,
            endpoints={str(w): r.get('standardized_predictive_residuals', {}) for w, r in endpoints.items()},
            points=[dict(w=r['w'], execution=r['curve_execution'], log_likelihood=r['parameter_log_likelihood'],
                         completed_trials=r['completed_trials']) for r in rows])
    restored = []
    if old_summary:
        for key in old_summary['failures']:
            if not key.startswith('curve__'):
                continue
            _, subject, coordinate, modality, index = key.split('__')
            if modality == 'fNIRS_only':
                coordinate = 'broadband_pca'
            replay_key = '__'.join(('replay', subject, coordinate, modality, index))
            value = completed.get(replay_key)
            restored.append(dict(old_task=key, replay_task=replay_key,
                completed=value is not None and value['curve_execution'] == 'completed'))
    paired = {}
    for modality in ('EEG', 'fNIRS'):
        subjects = {}
        for subject in dc['subjects']:
            cases = [v for k, v in completed.items() if k.startswith('paired__'+subject+'__'+modality+'__')]
            rows = [r for v in cases for r in v['rows']]
            good = [r for r in rows if r['complete']]
            complete = len(rows) == len(good) == 24 and len({r['trial'] for r in good}) == 24
            for row in rows:
                for condition, result in row['ssm'].items():
                    if result['execution'] != 'completed':
                        failures[f'paired__{subject}__{modality}__{row["trial"]}__{condition}'] = result
            subjects[subject] = dict(expected_trials=24, completed_paired_trials=len(good), complete=complete,
                increments={k: float(np.mean([r['increments'][k] for r in good])) for k in good[0]['increments']} if complete else None,
                successful_subset_increments={k: float(np.mean([r['increments'][k] for r in good])) for k in good[0]['increments']} if good and not complete else None,
                linear_increments={k: float(np.mean([r['linear']['increments'][k] for r in rows]))
                    for k in ('basic', 'independent_pairing', 'circular_shift')} if len(rows) == 24 else None,
                ssm_scores={k: float(np.mean([r['ssm'][k]['score'] for r in good]))
                    for k in ('own_context', 'joint', 'independent_pairing', 'circular_shift')} if complete else None,
                linear_scores={k: float(np.mean([r['linear']['scores'][k] for r in rows]))
                    for k in ('basic', 'joint', 'independent_pairing', 'circular_shift')} if len(rows) == 24 else None)
        complete_subjects = [r for r in subjects.values() if r['complete']]
        paired[modality] = dict(subjects=subjects, completed_subjects=len(complete_subjects),
            increments={k: diagnostic.cluster([r['increments'][k] for r in complete_subjects], dc)
                        for k in ('own_context', 'linear_basic', 'linear_joint', 'independent_pairing', 'circular_shift')},
            all_subjects_complete=len(complete_subjects) == len(dc['subjects']),
            interpretation='three-subject descriptive comparison; incomplete subjects cannot support a full-panel claim')
    return dict(schema=cfg['schema'], scale=scale, trajectory=trajectory,
        replay=dict(curves=curves, original_failed_tasks=restored,
            restored_original_tasks=sum(r['completed'] for r in restored),
            unique_original_failed_tasks=len({r['replay_task'] for r in restored}),
            saturation_unique_trials=sorted(saturation_trials), extreme_flow_unique_trials=sorted(extreme_trials),
            minimum_internal_evaluated_flow=minimum_flow,
            fnirs_alias='local_F3_alpha fNIRS-only curves alias broadband_pca; identical fNIRS inputs and noise',
            independent_new_trials=0),
        paired=paired, failures=failures, teacher_qualification='none', comprehensive_uq='not_executed')


def write_report(run_dir, summary):
    lines = ['# Observation contract repair and regression', '',
        'This is engineering regression and a bounded training-only diagnostic. No teacher qualification is granted.', '',
        '## Known coordinate scaling', '',
        '| Cohort | Completed / expected | Invariance passed | Max density error | Max parameter CDF error |',
        '|---|---:|---:|---:|---:|']
    for name, row in summary['scale'].items():
        errors = row['maximum_errors']
        lines.append(f'| {name} | {row["completed"]}/{row["expected"]} | {row["passed"]} | {errors.get("log_likelihood", float("nan")):.3g} | {errors.get("parameter_cdf", float("nan")):.3g} |')
    lines += ['', '| Cohort / branch | Δ log L(−0.5 − 0) [95% descriptive CI] | Slow preference | r coverage | HbO coverage | HbR coverage |', '|---|---:|---:|---:|---:|---:|']
    for cohort, case in summary['scale'].items():
        for branch, row in case['branches'].items():
            delta = row['boundary_minus_zero_ll']; metrics = row['model_truth']
            lines.append(f'| {cohort} / {branch} | {delta["mean"]:.4f} [{delta["ci95"][0]:.4f}, {delta["ci95"][1]:.4f}] | {row["slow_preference_count"]}/{case["completed"]} | '+
                ' | '.join(f'{metrics[t]["coverage95"]:.3%}' for t in ('r', 'clean_HbO', 'clean_HbR'))+' |')
    lines += ['', 'The deliberately mismatched branch scales observations alone. Synchronized scaling transforms the operator and noise once; canonical latent states, parameter posterior and density correction are tested. Reused trials are regression inputs, not independent validation.', '',
        '## Temporal reference', '',
        'The batch reference is exact only for a rest-linearized Gaussian path and Gaussian observation noise on the retained SVD subspace. Nonlinear Student-t inputs use a moment approximation. This is not a transformed independent Student-t likelihood and does not inherit A0 qualification.', '',
        '| Generating law / processing / branch | Complete fits | Retained rank | r / HbO / HbR canonical coverage | Δ log L(−0.5 − 0) |', '|---|---:|---:|---|---:|']
    for law, case in summary['trajectory'].items():
        for name, row in case['groups'].items():
            metrics = row['model_truth']
            coverage = ' / '.join(f'{metrics[t]["coverage95"]:.3%}' for t in ('r', 'clean_HbO', 'clean_HbR')) if metrics else 'unavailable'
            delta = row['boundary_minus_zero_ll']['mean']
            lines.append(f'| {law} / {name} | {row["completed"]}/{row["expected"]} | {row["retained_ranks"]} | {coverage} | {delta if delta is None else round(delta, 4)} |')
    checks = [r for v in summary['trajectory'].values() for r in v['rank_checks']]
    if checks:
        lines += ['', f'Rank-tolerance sensitivity: maximum transformed-mean change {max(r["maximum_transformed_mean_change"] for r in checks):.6g}; maximum change in the W likelihood difference {max(abs(r["boundary_minus_zero_ll_change"]) for r in checks):.6g}. Absolute log densities on different retained subspaces are not directly comparable. Full ranks, discarded singular values, canonical and processed coverage, and approximation discrepancies are retained in JSON.']
    replay = summary['replay']
    lines += ['', '## Identical-input numerical continuation', '',
        f'Original failed tasks restored: {replay["restored_original_tasks"]}/{len(replay["original_failed_tasks"])}; these represent {replay["unique_original_failed_tasks"]} unique input/curve tasks because fNIRS-only coordinates were duplicated. No independent trial was added.', '',
        f'Saturation occurred in {len(replay["saturation_unique_trials"])} unique subject/trial identities; f<0.01 occurred in {len(replay["extreme_flow_unique_trials"])}. Minimum internal evaluated flow: {replay["minimum_internal_evaluated_flow"]}. These are internal RK4 evaluation diagnostics, not counts of independent failures or a physical qualification.', '',
        '| Curve | Completed | W maximum (complete curves only) | Δ log L(−0.5 − 0) |', '|---|---:|---:|---:|']
    for name, row in replay['curves'].items():
        lines.append(f'| {name} | {row["completed"]}/{row["expected"]} | {row["likelihood_maximum_w"]} | {row["boundary_minus_zero_ll"]} |')
    lines += ['', '## Same-fold fixed SSM and linear control', '',
        'All dynamics are fixed at W=0. Inputs, outer folds, native masks, target coordinates, training variance normalization and null donors match. No extra measurement-gain candidate is selected. Scores are negative normalized MSE, so a positive increment favors the correctly paired fixed SSM.', '',
        '| Target / contrast | Complete subjects | Increment [95% descriptive CI] |', '|---|---:|---:|']
    for modality, case in summary['paired'].items():
        for control, delta in case['increments'].items():
            estimate = 'unavailable' if delta['mean'] is None else f'{delta["mean"]:.6f} [{delta["ci95"][0]:.6f}, {delta["ci95"][1]:.6f}]'
            lines.append(f'| {modality} / {control} | {case["completed_subjects"]}/3 | {estimate} |')
    lines += ['', f'Retained failed jobs/curves/conditions: {len(summary["failures"])}. Missing cases do not become successful-case qualification evidence.', '',
        'Original trial positions 4/9 and subjects 19–29 are not processed. The temporal reference has not replaced measured pointwise inference. Persistent bias, extreme flow, W pressure and failure to beat pairing controls remain separate questions from numerical/coordinate correctness. No protected campaign, comprehensive UQ, support expansion, tokenizer training or promotion is performed.', '']
    (run_dir/'summary.md').write_text('\n'.join(lines))


def render_figure(run_dir, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    branches = ('original', 'observation_only', 'synchronized')
    labels = ('Original', 'Observation only', 'Synchronized')
    for cohort, offset, color in (('reused', -.07, '#216b8a'), ('independent', .07, '#bb613f')):
        values = summary['scale'][cohort]['branches']
        if not values:
            continue
        delta = [values[b]['boundary_minus_zero_ll'] for b in branches]
        means = np.array([r['mean'] for r in delta]); ci = np.array([r['ci95'] for r in delta])
        axes[0].errorbar(np.arange(3)+offset, means, yerr=np.array([means-ci[:, 0], ci[:, 1]-means]), fmt='o', capsize=3, label=cohort, color=color)
    axes[0].axhline(0, color='gray', lw=1); axes[0].set_xticks(range(3), labels, rotation=20)
    axes[0].set(title='Known scaling: W pressure', ylabel='log L(W=-0.5) - log L(W=0)'); axes[0].legend()
    law = summary['trajectory']['linearized_gaussian']['groups']
    groups = ('model__synchronized', 'baseline__observation_only', 'baseline__synchronized', 'fnirs_filter__observation_only', 'fnirs_filter__synchronized', 'combined__observation_only', 'combined__synchronized')
    if law and all(law[g]['model_truth'] for g in groups):
        for target in ('r', 'clean_HbO', 'clean_HbR'):
            axes[1].plot(range(len(groups)), [law[g]['model_truth'][target]['coverage95'] for g in groups], 'o-', label=target)
    axes[1].axhline(.95, color='gray', ls='--', lw=1)
    axes[1].set_xticks(range(7), ['Model', 'B: mismatch', 'B: matched', 'F: mismatch', 'F: matched', 'All: mismatch', 'All: matched'], rotation=45, ha='right')
    axes[1].set(title='Gaussian temporal reference', ylabel='Canonical truth 95% coverage', ylim=(0, 1.03)); axes[1].legend(fontsize=8)
    controls = ('own_context', 'linear_basic', 'independent_pairing', 'circular_shift')
    for modality, offset in (('EEG', -.08), ('fNIRS', .08)):
        values = summary['paired'][modality]['increments']
        for i, control in enumerate(controls):
            value = values[control]
            if value['mean'] is not None:
                mean, ci = value['mean'], value['ci95']
                axes[2].errorbar(i+offset, mean, yerr=[[mean-ci[0]], [ci[1]-mean]], fmt='o', capsize=3,
                    color='#216b8a' if modality == 'EEG' else '#bb613f', label=modality if i == 0 else None)
    axes[2].axhline(0, color='gray', lw=1); axes[2].set_xticks(range(4), ['Own context', 'Linear basic', 'Pair null', 'Shift null'], rotation=35, ha='right')
    axes[2].set(title='Fixed SSM: same-fold increments', ylabel='Negative normalized MSE increment')
    if axes[2].get_legend_handles_labels()[0]:
        axes[2].legend()
    fig.suptitle('Observation contract regression · no teacher qualification', fontsize=14)
    for suffix in ('png', 'pdf'):
        fig.savefig(run_dir/f'observation_repair.{suffix}', dpi=160)
    plt.close(fig)


def bridge_mask(count, name, center_steps):
    if name == 'full':
        return np.ones((count, 3), dtype=bool)
    return np.isfinite(step5.masked_input(np.zeros((count, 3)), name, center_steps)[0])


def mask_bridge_job(cfg, dc, base, law, replicate):
    """Same processed inputs, old nonlinear pointwise vs temporal moment reference.

    The second branch still is NOT nonlinear Student-t temporal inference.
    Nonlinear generating trials explicitly measure that remaining discrepancy.
    """
    tcfg = cfg['trajectory']
    count, seed = tcfg['steps'], tcfg['seed_start'][law]+replicate
    if law == 'linearized_gaussian':
        generated = generate_linear_gaussian(base, count, seed)
    else:
        generated = {k: v[:count] for k, v in step5.localization.generate_matched(base, 'W', 0., seed).items()}
    truth = np.column_stack((generated['states'][:, 0], generated['clean']))
    rows, rank_checks, operators = [], [], {}
    for variant in tcfg['variants']:
        processing = trajectory_operator(count, variant, dc)
        for name in tcfg['masks']:
            mask = bridge_mask(count, name, tcfg['center_steps'])
            operator = processing.with_visible_interpolation(mask, mask)
            y = operator.apply(generated['observations'], mask)
            # Paired counterfactual intervention on hidden feature values.
            changed = generated['observations'].copy(); changed[~mask] = 1e90
            np.testing.assert_array_equal(y, operator.apply(changed, mask))
            np.testing.assert_array_equal(np.isfinite(y), mask)
            matrix, available = operator.matrix(mask)
            assert not np.any(matrix[:, ~mask.ravel()])
            operators[variant+'__'+name] = dict(visible_inputs=int(mask.sum()),
                visible_outputs=int(available.sum()),
                conservative_outputs=int(processing.matrix(mask)[1].sum()),
                hidden_input_intervention_max_error=0.)
            for w in tcfg['fixed_w']:
                p, c = step5.localization.model(base, 'W', w)
                noise = np.square(p.fixed.observation_scale)*p.fixed.student_nu/(p.fixed.student_nu-2)
                for branch in tcfg['branches']:
                    row = dict(law=law, replicate=replicate, seed=seed, variant=variant,
                               mask=name, w=w, branch=branch)
                    try:
                        if branch == 'pointwise_student_t':
                            fitted = joint.smooth_balloon_joint(y, p, config=c,
                                quadrature_order=cfg['flow']['quadrature_order'])
                            _, mean, variance = step5.targets_and_moments(generated, fitted)
                            transformed = fitted.transformed_mean
                            ll = fitted.parameter_log_likelihood
                            processed_mean = fitted.trajectory_mean
                            processed_variance = fitted.state_posterior_variance
                        else:
                            fitted = joint.smooth_balloon_trajectory_reference(y, p, config=c,
                                trajectory_spec=operator, noise_variance=noise,
                                input_mask=mask, rank_rtol=tcfg['rank_rtol'])
                            mean = np.column_stack((fitted['transformed_mean'][:, 0], fitted['canonical_clean_mean']))
                            variance = np.column_stack((np.diag(fitted['transformed_covariance'])[::6],
                                np.diag(fitted['canonical_clean_covariance']).reshape(count, 3)))
                            transformed = fitted['transformed_mean']
                            ll = fitted['parameter_log_likelihood']
                            processed_mean = fitted['processed_clean_mean']
                            processed_variance = np.diag(fitted['processed_clean_covariance']).reshape(count, 3)
                            row.update(retained_rank=fitted['retained_rank'],
                                observed_coordinates=fitted['observed_coordinates'],
                                support_residual_norm=fitted['support_residual_norm'],
                                discarded_singular_values=fitted['discarded_singular_values'])
                            if variant == 'combined' and replicate in tcfg['rank_check_replicates']:
                                for rtol in tcfg['rank_check_rtols']:
                                    other = joint.smooth_balloon_trajectory_reference(y, p, config=c,
                                        trajectory_spec=operator, noise_variance=noise, input_mask=mask, rank_rtol=rtol)
                                    rank_checks.append(dict(mask=name, w=w, rank_rtol=rtol,
                                        rank=other['retained_rank'], reference_rank=fitted['retained_rank'],
                                        maximum_mean_change=maximum_error(other['transformed_mean'], transformed),
                                        maximum_variance_change=maximum_error(np.diag(other['transformed_covariance']),
                                                                            np.diag(fitted['transformed_covariance'])),
                                        log_likelihood=other['parameter_log_likelihood']))
                        metrics = step5.truth_metrics(truth, mean, variance)
                        for j, target in enumerate(step5.localization.TARGETS):
                            metrics[target]['bias'] = float(np.mean(mean[:, j]-truth[:, j]))
                        hidden = ~mask.all(axis=1)
                        hidden_metrics = step5.truth_metrics(truth[hidden], mean[hidden], variance[hidden]) if hidden.any() else None
                        processed_truth = operator.apply(generated['clean'], mask)
                        processed_metrics = {}
                        for j, target in enumerate(core.OBSERVATION_NAMES):
                            visible = mask[:, j]
                            if visible.any():
                                error = processed_mean[visible, j]-processed_truth[visible, j]
                                processed_metrics[target] = dict(rmse=float(np.sqrt(np.mean(error**2))),
                                    bias=float(np.mean(error)), coverage95=float(np.mean(abs(error) <=
                                        1.95996398454*np.sqrt(np.maximum(processed_variance[visible, j], 0.)))))
                        domain_exits = []
                        for i, state in enumerate(transformed[:-1]):
                            try:
                                core._require_flow_drift_domain(state, p, c.dt)
                            except core.FlowDomainExit as exc:
                                domain_exits.append(dict(time_index=i, drift=exc.diagnostic))
                        row.update(execution='completed', parameter_log_likelihood=ll,
                            all_time_metrics=metrics, hidden_time_metrics=hidden_metrics,
                            processed_visible_metrics=processed_metrics,
                            minimum_flow_at_transformed_mean=float(np.exp(transformed[:, 2]).min()),
                            smoothed_mean_drift_domain_exits=domain_exits)
                    except Exception as exc:
                        row.update(execution='failed', error=repr(exc), traceback=traceback.format_exc())
                        if isinstance(exc, core.FlowDomainExit):
                            row['flow_domain_exit'] = exc.diagnostic
                    rows.append(row)
    return dict(law=law, replicate=replicate, seed=seed, rows=rows,
                operator_checks=operators, rank_checks=rank_checks)


def flow_replay(cfg, dc, base):
    """Only the three named old training trials; use the owning joint filter."""
    preparation, rows = {}, []
    old = json.loads((ROOT/cfg['previous_run']/'summary.json').read_text())
    for subject in dict.fromkeys(subject for subject, _ in cfg['flow']['trials']):
        arrays, identity = load_replay_inputs(cfg, dc, base, subject)
        preparation[subject] = identity
        config = copy.deepcopy(base)
        config['model']['observation_scale'] = old['preparation'][subject]['broadband_pca_noise_scale']
        for selected_subject, trial in cfg['flow']['trials']:
            if selected_subject != subject:
                continue
            y = arrays['broadband_pca'][trial].copy(); y[:, 0] = np.nan
            for w in cfg['flow']['fixed_w']:
                p, c = step5.localization.model(config, 'W', w)
                spec = core.BalloonObservationSpec().resolved(p.fixed)
                row = dict(subject=subject, training_trial_index=trial,
                    trial_identity=identity['trials'][trial], w=w, modality='fNIRS_only',
                    observation_scale=config['model']['observation_scale'])
                try:
                    ll, _, fm, _, _, _, _, _ = joint._filter(y, p, c, spec,
                        cfg['flow']['quadrature_order'], True)
                    row.update(execution='completed', parameter_log_likelihood=ll,
                        minimum_filtered_flow=float(np.exp(fm[:, 2]).min()))
                except Exception as exc:
                    row.update(execution='failed', error=repr(exc), traceback=traceback.format_exc())
                    if isinstance(exc, core.FlowDomainExit):
                        row['flow_domain_exit'] = exc.diagnostic
                        index = exc.diagnostic['transition_index']
                        row['event_relative_zero_time_s'] = (index-1)*c.dt-5+exc.diagnostic['first_zero_time_s']
                rows.append(row)
    return dict(preparation=preparation, rows=rows,
        unique_training_trials=len(cfg['flow']['trials']), independent_new_trials=0,
        previous_summary_sha256=diagnostic.digest(ROOT/cfg['previous_run']/'summary.json'))


def summarize_mask_repair(cfg, synthetic, replay, context):
    rows = [r for v in synthetic.values() if v['execution'] == 'completed' for r in v['result']['rows']]
    groups = {}
    tcfg = cfg['trajectory']
    for law in tcfg['laws']:
        for variant in tcfg['variants']:
            for mask in tcfg['masks']:
                for branch in tcfg['branches']:
                    selected = [r for r in rows if (r['law'], r['variant'], r['mask'], r['branch']) == (law, variant, mask, branch)]
                    zero = [r for r in selected if r['w'] == 0 and r['execution'] == 'completed']
                    slow = {r['replicate']: r for r in selected if r['w'] == -.5 and r['execution'] == 'completed'}
                    deltas = [slow[r['replicate']]['parameter_log_likelihood']-r['parameter_log_likelihood']
                              for r in zero if r['replicate'] in slow]
                    metrics = {}
                    for support in ('all_time_metrics', 'hidden_time_metrics'):
                        available = [r[support] for r in zero if r[support] is not None]
                        metrics[support] = {target: {metric: float(np.mean([a[target][metric] for a in available]))
                            for metric in available[0][target]} for target in step5.localization.TARGETS} if available else None
                    processed = [r['processed_visible_metrics'] for r in zero]
                    metrics['processed_visible_metrics'] = {
                        target: {metric: float(np.mean([a[target][metric] for a in processed if target in a]))
                                 for metric in ('rmse', 'bias', 'coverage95')}
                        for target in core.OBSERVATION_NAMES if any(target in a for a in processed)}
                    w_informative = mask != 'whole_fNIRS'
                    key = '__'.join((law, variant, mask, branch))
                    groups[key] = dict(law=law, variant=variant, mask=mask, branch=branch,
                        expected_trials=tcfg['replicates_per_law'], completed_truth_w_trials=len(zero),
                        expected_fits=tcfg['replicates_per_law']*2,
                        completed_fits=sum(r['execution'] == 'completed' for r in selected),
                        retained_ranks=sorted(set(r['retained_rank'] for r in zero if 'retained_rank' in r)),
                        w_interpretation='fixed_W_comparison' if w_informative else 'EEG_only_no_W_information',
                        slow_preference_count=sum(x > 0 for x in deltas) if w_informative else None,
                        paired_w_trials=len(deltas),
                        mean_boundary_minus_zero_ll=float(np.mean(deltas)) if deltas and w_informative else None,
                        maximum_absolute_w_ll_difference=max(map(abs, deltas), default=None),
                        minimum_flow_at_transformed_mean=min((r['minimum_flow_at_transformed_mean'] for r in zero), default=None),
                        smoothed_mean_exit_trials=sum(bool(r['smoothed_mean_drift_domain_exits']) for r in zero),
                        **metrics)
    bounds = tcfg['coverage95_diagnostic_bounds']
    coverage_checks = {law: [] for law in tcfg['laws']}
    for key, group in groups.items():
        if group['branch'] != 'mask_specific_gaussian_reference':
            continue
        metrics = group['all_time_metrics']
        passed = group['completed_truth_w_trials'] == group['expected_trials'] and metrics is not None and all(
            bounds[0] <= metrics[target]['coverage95'] <= bounds[1] for target in ('r', 'clean_HbO', 'clean_HbR'))
        coverage_checks[group['law']].append(dict(group=key, diagnostic_coverage_in_bounds=passed))
    return dict(schema=cfg['schema'], groups=groups, coverage_checks=coverage_checks,
        context_checks=context, replay=replay,
        failures=[r for r in rows if r['execution'] != 'completed'],
        job_failures={k: v for k, v in synthetic.items() if v['execution'] != 'completed'},
        rank_checks={k: v['result']['rank_checks'] for k, v in synthetic.items() if v['execution'] == 'completed' and v['result']['rank_checks']},
        measured_comparison=dict(execution='not_executed',
            reason='No nonlinear Student-t temporal inference has been validated under the new mask contract. The Gaussian moment reference cannot supply that prerequisite.',
            native_preprocessing_validation='not_executed; this run tests controlled linear feature coordinates'),
        teacher_qualification='none', comprehensive_uq='not_executed')


def write_mask_report(run_dir, summary):
    lines = ['# Flow-domain and mask-specific observation experiment', '',
        'The six-state drift, oxygen extraction, noise, W support and measurement gain are unchanged. Failures are classified and retained; no state is projected or redrawn. No new measured comparison or teacher qualification is claimed.', '',
        '## Three original training trials', '',
        '| Subject / training index | W | Outcome | Transition | First zero after update (s) | Event-relative zero (s) |',
        '|---|---:|---|---:|---:|---:|']
    for row in summary['replay'].get('rows', []):
        event = row.get('flow_domain_exit', {})
        lines.append(f'| {row["subject"]} / {row["training_trial_index"]} | {row["w"]} | '
            f'{"flow_domain_exit" if event else row["execution"]} | {event.get("transition_index")} | '
            f'{event.get("first_zero_time_s")} | {row.get("event_relative_zero_time_s")} |')
    lines += ['', 'Indices identify the original prepared training inventory, not native trial positions. Two W evaluations of one trial are not independent trials. Subject_09/trial_12 at W=0 already completed in the old replay; its completion is not a rescue. Full pre/post observation means, covariance matrices, and modality visibility for the last four updates are retained in replay.json.', '',
        '## Mask-specific linear bridge', '',
        '| Window | Mask | Old conservative EEG / HbO / HbR outputs | New outputs |', '|---:|---|---|---|']
    for row in summary['context_checks']:
        lines.append(f'| {row["steps"]} | {row["mask"]} | {row["conservative_output_counts"]} | {row["output_counts"]} |')
    lines += ['', 'The actual visible-only interpolation → processing → output-selection matrix transforms both means and the complete time-noise covariance. Hidden-center interpolants are not observations. Impulse construction and hidden-value interventions are checked independently. This does not validate raw EEG power, optical conversion or motion suppression.', '',
        '## Same-input synthetic comparison at truth W=0', '',
        'Each generating law has 24 independent trials. W=0 and W=−0.5 fits share each trial. The pointwise branch uses the existing nonlinear Student-t filter; the mask-specific branch remains a resting-state Gaussian moment reference, including on nonlinear Student-t data. Coverage below is canonical clean truth over all times; hidden-time metrics are separately retained in summary.json. Completed subsets cannot qualify an incomplete panel.', '',
        '| Law / processing / mask / inference | W=0 trials | r / HbO / HbR coverage | HbR bias | Mean log L(−0.5)−log L(0) |',
        '|---|---:|---|---:|---:|']
    for key, row in summary['groups'].items():
        metrics = row['all_time_metrics']
        coverage = ' / '.join(f'{metrics[t]["coverage95"]:.2%}' for t in ('r', 'clean_HbO', 'clean_HbR')) if metrics else 'unavailable'
        bias = f'{metrics["clean_HbR"]["bias"]:.5f}' if metrics else 'unavailable'
        delta = row['mean_boundary_minus_zero_ll']
        delta_text = 'W unidentifiable (EEG only)' if row['w_interpretation'] == 'EEG_only_no_W_information' else str(delta if delta is None else round(delta, 5))
        lines.append(f'| {key} | {row["completed_truth_w_trials"]}/24 | {coverage} | {bias} | {delta_text} |')
    lines += ['', 'Likelihood differences compare W only within the same mask/operator and retained support. EEG-only W is checked as invariant; floating-point signs near zero are not slow-W preferences. Absolute pointwise and temporal densities are not comparable. SVD rank, discarded directions, support residuals and tolerance sensitivity are retained; numerical rank truncation is an approximation.', '',
        '## Remaining prerequisite', '', summary['measured_comparison']['reason'], '',
        'Therefore the requested new measured same-fold linear/pairing/shift comparison is not executed. Three old training failures were replayed for classification only. Subjects 19–29 and original held-out trial positions remain closed; old A0 evidence and Gaussian-reference coverage are not substituted for nonlinear temporal qualification.', '']
    (run_dir/'summary.md').write_text('\n'.join(lines))


def run_mask_repair(args, cfg, dc, base, measured, metadata):
    if args.stage == 'measured':
        raise ValueError('new measured comparison requires validated nonlinear Student-t temporal inference')
    run_dir = args.run_dir.resolve()
    if run_dir.parent != (ROOT/base['output_root']).resolve():
        raise ValueError('use a fresh direct child of the existing Step5 artifact root')
    if args.dry_run:
        print(json.dumps(dict(schema=cfg['schema'], stage=args.stage,
            synthetic_trials=48 if args.stage in ('synthetic', 'all') else 0,
            replay_trials=cfg['flow']['trials'] if args.stage in ('replay', 'all') else [],
            new_measured_comparison='not executable: nonlinear temporal qualification missing')))
        return
    run_dir.mkdir(exist_ok=False)
    started = time.time()
    source_files = (*SOURCE_FILES, str(MASK_CONFIG.relative_to(ROOT)))
    manifest = dict(schema=cfg['schema'], execution='running', stage=args.stage,
        started_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_worktree=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        source_sha256={p: diagnostic.digest(ROOT/p) for p in source_files},
        config_sha256=diagnostic.digest(args.config),
        software=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        original_heldout_trials_processed=0, native_data_reads=0)
    step5.write_json(run_dir/'manifest.json', manifest)
    for name, value in [('config', cfg), ('diagnostic', dc), ('base', base)]:
        (run_dir/f'resolved_{name}.yaml').write_text(yaml.safe_dump(value, sort_keys=False))
    for file in ('experiments/evaluate_step5_observation_repair.py',
                 'src/inference/observation_baselines.py',
                 'src/inference/t3a_balloon_robust_ssm.py', 'src/inference/t3a_balloon_joint_ssm.py'):
        (run_dir/(Path(file).stem+'_snapshot.py')).write_bytes((ROOT/file).read_bytes())
    context = []
    for count in cfg['trajectory']['context_check_steps']:
        processing = trajectory_operator(count, 'combined', dc)
        for name in cfg['trajectory']['masks']:
            mask = bridge_mask(count, name, cfg['trajectory']['center_steps'])
            operator = processing.with_visible_interpolation(mask, mask)
            np.testing.assert_array_equal(operator.matrix(mask)[1], mask)
            context.append(dict(steps=count, mask=name, output_counts=operator.matrix(mask)[1].sum(axis=0),
                conservative_output_counts=processing.matrix(mask)[1].sum(axis=0)))
    step5.write_json(run_dir/'trajectory_contract.json', {
        variant+'__'+name: trajectory_operator(cfg['trajectory']['steps'], variant, dc).with_visible_interpolation(
            bridge_mask(cfg['trajectory']['steps'], name, cfg['trajectory']['center_steps']),
            bridge_mask(cfg['trajectory']['steps'], name, cfg['trajectory']['center_steps'])).processing
        for variant in cfg['trajectory']['variants'] for name in cfg['trajectory']['masks']})
    jobs = [(f'mask_bridge__{law}__{i:02d}', 'mask_bridge', (cfg, dc, base, law, i))
        for law in cfg['trajectory']['laws'] for i in range(cfg['trajectory']['replicates_per_law'])] if args.stage in ('synthetic', 'all') else []
    synthetic = execute_jobs(jobs, run_dir, cfg['workers']) if jobs else {}
    replay = flow_replay(cfg, dc, base) if args.stage in ('replay', 'all') else {}
    step5.write_json(run_dir/'replay.json', replay)
    summary = summarize_mask_repair(cfg, synthetic, replay, context)
    step5.write_json(run_dir/'summary.json', summary)
    write_mask_report(run_dir, summary)
    manifest.update(execution='completed', elapsed_seconds=time.time()-started,
        finished_at=datetime.now(timezone.utc).isoformat(), synthetic_jobs=len(synthetic),
        synthetic_job_failures=len(summary['job_failures']), synthetic_fit_failures=len(summary['failures']),
        replay_model_failures=sum(r['execution'] != 'completed' for r in replay.get('rows', [])))
    step5.write_json(run_dir/'manifest.json', manifest)
    print(json.dumps(dict(run_dir=str(run_dir), **manifest)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=('synthetic', 'replay', 'measured', 'all'), default='all')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cfg, dc, base, measured, metadata = load_config(args.config)
    if cfg['schema'] == 'step5_observation_repair_v2':
        return run_mask_repair(args, cfg, dc, base, measured, metadata)
    if args.dry_run:
        raise ValueError('--dry-run is implemented for the v2 mask-specific bridge')
    run_dir = args.run_dir.resolve()
    if run_dir.parent != (ROOT/base['output_root']).resolve():
        raise ValueError('use a fresh direct child of the existing Step5 artifact root')
    run_dir.mkdir(exist_ok=False)
    started = time.time()
    manifest = dict(schema=cfg['schema'], execution='running', stage=args.stage,
        started_at=datetime.now(timezone.utc).isoformat(), source_sha256={p: diagnostic.digest(ROOT/p) for p in SOURCE_FILES},
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_worktree=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        config_sha256=diagnostic.digest(args.config), software=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        original_heldout_trials_processed=0, subjects=dc['subjects'] if args.stage != 'synthetic' else [])
    step5.write_json(run_dir/'manifest.json', manifest)
    for name, value in [('config', cfg), ('diagnostic', dc), ('base', base), ('measured', measured)]:
        (run_dir/f'resolved_{name}.yaml').write_text(yaml.safe_dump(value, sort_keys=False))
    (run_dir/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    # Preserve executable source identity, including the changed forward owner.
    for name in ('t3a_balloon_robust_ssm', 't3a_balloon_joint_ssm', 'observation_baselines'):
        (run_dir/f'{name}_snapshot.py').write_bytes((ROOT/f'src/inference/{name}.py').read_bytes())
    jobs = []
    if args.stage in ('synthetic', 'all'):
        jobs.extend((f'scale__{cohort}__{i:02d}', 'scale', (cfg, dc, base, cohort, i))
            for cohort in ('reused', 'independent') for i in range(cfg['scale'][cohort+'_replicates']))
        jobs.extend((f'trajectory__{law}__{i:02d}', 'trajectory', (cfg, dc, base, law, i))
            for law in cfg['trajectory']['laws'] for i in range(cfg['trajectory']['replicates_per_law']))
        step5.write_json(run_dir/'trajectory_contract.json', {variant: trajectory_operator(cfg['trajectory']['steps'], variant, dc).processing
                                                             for variant in cfg['trajectory']['variants']})
    synthetic = execute_jobs(jobs, run_dir, cfg['workers']) if jobs else {}
    # Mathematical regression is required before new measured evaluation. This
    # is validation sequencing, not an authorization flag or teacher admission.
    scale_values = [v for k, v in synthetic.items() if k.startswith('scale__')]
    if args.stage == 'all' and any(v['execution'] != 'completed' or not v['result']['passed'] for v in scale_values):
        manifest.update(execution='failed', reason='coordinate invariance regression failed', elapsed_seconds=time.time()-started)
        step5.write_json(run_dir/'manifest.json', manifest)
        raise RuntimeError('coordinate regression failed; retain results before measured evaluation')
    jobs, preparation, old_summary = [], {}, None
    if args.stage in ('replay', 'all'):
        old_summary = json.loads((ROOT/cfg['previous_run']/'summary.json').read_text())
        manifest['previous_summary_sha256'] = diagnostic.digest(ROOT/cfg['previous_run']/'summary.json')
        for subject in dc['subjects']:
            arrays, identity = load_replay_inputs(cfg, dc, base, subject)
            for coordinate in dc['coordinates']:
                np.testing.assert_array_equal(
                    old_summary['preparation'][subject][coordinate+'_noise_scale'][1:],
                    old_summary['preparation'][subject]['broadband_pca_noise_scale'][1:])
            preparation[subject] = identity
            for coordinate in dc['coordinates']:
                config = copy.deepcopy(base)
                config['model']['observation_scale'] = old_summary['preparation'][subject][coordinate+'_noise_scale']
                for modality in dc['curve']['modalities']:
                    # fNIRS inputs/noise are identical in both EEG coordinates.
                    # Run once and record aliases, not independent replicates.
                    if modality == 'fNIRS_only' and coordinate != 'broadband_pca':
                        continue
                    grid = ([-.5, 0.] if modality == 'EEG_only' else np.linspace(*base['axes']['W']['bounds'], cfg['replay']['grid_points']))
                    for i, w in enumerate(grid):
                        jobs.append((f'replay__{subject}__{coordinate}__{modality}__{i:02d}', 'replay',
                            (cfg, dc, config, arrays[coordinate], subject, coordinate, modality, float(w))))
        step5.write_json(run_dir/'replay_inputs.json', preparation)
    replay = execute_jobs(jobs, run_dir, cfg['workers']) if jobs else {}
    jobs = []
    if args.stage in ('measured', 'all'):
        from experiments.evaluate_t3_multisession_loso import _validate_metadata
        metadata_summary, inventory, hashes = _validate_metadata(metadata)
        step5.write_json(run_dir/'metadata_boundary.json', dict(summary=metadata_summary, hashes=hashes,
            selected_inventory=[r for r in inventory if r.get('subject_id', r.get('subject')) in dc['subjects']]))
        constant = student_difference_mad(base['model']['student_nu'])
        for subject in dc['subjects']:
            trials, detail = diagnostic.load_training_subject(subject, dc, base, measured, metadata)
            # Verify native regeneration against the identical previous inputs.
            projection = diagnostic.fit_projection(trials, list(range(24)), base, measured)
            arrays, identity = load_replay_inputs(cfg, dc, base, subject)
            regenerated = np.array([diagnostic.project_trial(t, projection, 'broadband_pca') for t in trials])
            np.testing.assert_allclose(regenerated, arrays['broadband_pca'], rtol=1e-12, atol=1e-12)
            detail['regenerated_input_max_absolute_difference'] = maximum_error(regenerated, arrays['broadband_pca'])
            detail['replay_identity'] = identity
            step5.write_json(run_dir/f'prepared_{subject}.json', detail)
            for modality in ('EEG', 'fNIRS'):
                for fold in range(dc['linear']['outer_folds']):
                    jobs.append((f'paired__{subject}__{modality}__{fold}', 'paired',
                        (cfg, dc, base, measured, trials, subject, modality, fold, constant)))
    paired = execute_jobs(jobs, run_dir, cfg['workers']) if jobs else {}
    results = dict(synthetic=synthetic, replay=replay, paired=paired)
    step5.write_json(run_dir/'results.json', results)
    summary = summarize(cfg, dc, results, old_summary)
    step5.write_json(run_dir/'summary.json', summary)
    write_report(run_dir, summary)
    render_figure(run_dir, summary)
    manifest.update(execution='completed', elapsed_seconds=time.time()-started,
        finished_at=datetime.now(timezone.utc).isoformat(),
        job_failures=sum(v['execution'] != 'completed' for group in results.values() for v in group.values()),
        retained_failures=len(summary['failures']),
        expected_jobs={name: len(group) for name, group in results.items()})
    step5.write_json(run_dir/'manifest.json', manifest)
    print(json.dumps(dict(run_dir=str(run_dir), elapsed_seconds=manifest['elapsed_seconds'], job_failures=manifest['job_failures'])), flush=True)


if __name__ == '__main__':
    main()
