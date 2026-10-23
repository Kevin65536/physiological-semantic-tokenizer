#!/usr/bin/env python3
"""Finite Suite 1: source observability, priors, and known-truth recovery.

Only synthetic arrays and two explicit retained-public evidence packages are
admitted. There is no raw loader, protected-data reader, or source discovery.
Substantial pilot/run stages must be launched by a durable supervisor.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
import csv
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.shared_driver_modes import temporal_mode_basis
from src.inference.shared_driver_spatial import forward_spatial_shared_driver
from src.inference.source_observability import (
    fit_free_physiology, information_decomposition, nuisance_profile_information,
    pack_spatial_linearization, weighted_jacobian,
)
from src.inference.t3a_balloon_robust_ssm import BalloonFreeParameters, BalloonParameters

DEFAULT_CONFIG = ROOT/'experiments/configs/physiology_semantic_tokenizer/source_observability_v1.yaml'
ARTIFACT_ROOT = 'experiments/runs/physiology_semantic_tokenizer/source_observability'
SPATIAL = 'experiments/runs/physiology_semantic_tokenizer/shared_driver_spatial/20261020_measured_v2'
SUPPORT = 'experiments/runs/physiology_semantic_tokenizer/ssm_output_support/20261010_v1'
SOURCES = ['experiments/scripts/evaluate_source_observability.py',
           'src/inference/source_observability.py', 'src/inference/shared_driver_spatial.py',
           'src/inference/shared_driver_reconstruction.py', 'src/inference/shared_driver_modes.py',
           'src/inference/t3a_balloon_robust_ssm.py', 'src/inference/shared_driver_rk4.py']


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serial(v) for v in value]
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def token(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def read_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg.get('schema') != 'source_observability_v1'
            or cfg.get('experiment_id') != 'SOURCE-OBSERVABILITY-v1'
            or cfg.get('artifact_root') != ARTIFACT_ROOT
            or cfg['data'].get('protected_data') != 'forbidden'
            or cfg['data'].get('raw_data') != 'forbidden'
            or cfg['data'].get('spatial_run') != SPATIAL
            or cfg['data'].get('output_support_run') != SUPPORT):
        raise ValueError('wrong suite identity or retained-only data boundary')
    s, p, q = cfg['synthetic'], cfg['priors'], cfg['solver']
    if (s['steps'] != 120 or s['temporal_modes'] != 12 or s['dt_s'] != .25
            or s['tau_truths'] != [1.3, 2.7] or s['kappa_truths'] != [.45, .9]
            or len(s['seeds']) != 4 or len(set(s['seeds'])) != 4
            or s['noise_multipliers'] != [0., 1.]
            or p['multipliers'] != [0., .1, 1., 10.]
            or len(q['starts']) != 3 or q['max_nfev'] < 1
            or len(cfg['profiles']['tau_grid']) < 7 or len(cfg['profiles']['kappa_grid']) < 7):
        raise ValueError('finite synthetic recovery contract changed')
    if any(np.allclose(start, truth) for start in q['starts'] for truth in
           itertools.product(s['tau_truths'], s['kappa_truths'])):
        raise ValueError('a fit start equals physiological truth')
    return cfg


def modality_mask(shape, modality, eeg_features=3):
    t, c = shape
    result = np.zeros(shape, bool)
    if modality == 'EEG_only':
        result[:, :eeg_features] = True
    elif modality == 'HbO_only':
        result[:, eeg_features::2] = True
    elif modality in ('Hb_only', 'Hb_half_rate'):
        result[:, eeg_features:] = True
        if modality == 'Hb_half_rate':
            result[1::2] = False
    elif modality == 'EEG_Hb':
        result[:] = True
    else:
        raise ValueError('unregistered modality')
    return result


def synthetic_case(cfg, task):
    s, solver = cfg['synthetic'], cfg['solver']
    basis = temporal_mode_basis(s['steps'], s['temporal_modes'])
    rng = np.random.default_rng(task['seed'])
    coef = rng.normal(size=(s['temporal_modes'], 1))/np.arange(1, s['temporal_modes']+1)[:, None]**s['coefficient_decay']
    coef *= s['driver_rms']/np.sqrt(np.mean((basis@coef)**2))
    # All five vascular coordinates depart from rest, with a seed-fixed sign.
    initial = np.asarray(s['initial_coordinates'])[None]*(1 if task['seed'] % 2 else -1)
    parameters = BalloonParameters(free=BalloonFreeParameters(tau=task['tau_truth'], kappa=task['kappa_truth']))
    forward_kwargs = dict(temporal_basis=basis, eeg_loadings=np.asarray(s['eeg_loadings']),
        hb_loadings=np.asarray(s['hb_loadings']), substeps=solver['substeps'],
        numerical_backend=solver['numerical_backend'])
    truth = forward_spatial_shared_driver(coef, initial, parameters, s['dt_s'], derivative=False,
        **dict(forward_kwargs, substeps=s['generator_substeps']))
    noise = np.random.default_rng(task['seed']+100003).normal(size=truth['prediction'].shape)*s['noise_sd']
    target = truth['prediction']+task.get('noise_multiplier', 1.)*noise
    return dict(truth=truth, parameters=parameters, target=target,
                forward_kwargs=forward_kwargs, truth_coefficients=coef, truth_initial=initial)


def prior_options(cfg, multiplier, *, shifted_center=None):
    p = cfg['priors']
    result = {name: multiplier*p[name] for name in
              ('driver_curvature_weight', 'initial_weight', 'flow_prior_weight')}
    result.update(flow_log_sd=p['flow_log_sd'], physiology_prior_weight=p['physiology_prior_weight'],
                  physiology_prior_center=p['physiology_prior_center'],
                  physiology_prior_log_sd=p['physiology_prior_log_sd'])
    if shifted_center is not None:
        result.update(physiology_prior_center=shifted_center,
                      physiology_prior_weight=p['shifted_prior_weight'])
    return result


def scales_for(cfg, modes, k=1):
    info = cfg['information']
    return np.r_[np.full(modes*k, info['coefficient_scale']),
        np.tile(info['initial_coordinate_scales'], k), info['log_physiology_scales']]


def recovery_metrics(fit, truth, task, cfg):
    if 'prediction' not in fit:
        return {}
    predicted, clean = fit['prediction'], truth['prediction']
    driver, true_driver = fit['drivers'], truth['drivers']
    difference = driver-true_driver
    metrics = dict(tau=fit['tau'], kappa=fit['kappa'],
        tau_absolute_error=abs(fit['tau']-task['tau_truth']),
        kappa_absolute_error=abs(fit['kappa']-task['kappa_truth']),
        tau_log_error=float(np.log(fit['tau']/task['tau_truth'])),
        kappa_log_error=float(np.log(fit['kappa']/task['kappa_truth'])),
        driver_nrmse=float(np.sqrt(np.mean(difference**2)/np.mean(true_driver**2))),
        driver_dc_error=float(difference.mean()),
        initial_physical_rmse=float(np.sqrt(np.mean((fit['initial_state']-truth['initial_state'])**2))),
        prediction_clean_rmse=float(np.sqrt(np.mean((predicted-clean)**2))),
        hb_clean_rmse=float(np.sqrt(np.mean((predicted[:, 3:]-clean[:, 3:])**2))))
    scale = np.asarray(cfg['synthetic']['weighting_sd'])
    active = modality_mask(predicted.shape, task['modality'])
    standardized = (predicted-clean)/scale
    metrics['visible_clean_standardized_nrmse'] = float(np.sqrt(np.mean(standardized[active]**2)))
    metrics['hb_clean_standardized_nrmse'] = float(np.sqrt(np.mean(standardized[:, 3:]**2)))
    metrics['theta_log_rmse'] = float(np.sqrt((metrics['tau_log_error']**2+metrics['kappa_log_error']**2)/2))
    for col, name in enumerate(('r', 's', 'f', 'v', 'p', 'q')):
        metrics[f'state_{name}_rmse'] = float(np.sqrt(np.mean((fit['states'][:, :, col]-truth['states'][:, :, col])**2)))
    return metrics


def fit_task(cfg, task):
    case = synthetic_case(cfg, task)
    s, solver = cfg['synthetic'], cfg['solver']
    mask = modality_mask(case['target'].shape, task['modality'])
    target = np.where(mask, case['target'], np.nan)
    options = dict(case['forward_kwargs'], sd=s['weighting_sd'], visible=mask,
        feature_weights=s['feature_weights'], **prior_options(cfg, task['prior_multiplier'],
            shifted_center=task.get('shifted_center')),
        tau_bounds=solver['tau_bounds'], kappa_bounds=solver['kappa_bounds'],
        max_nfev=solver['max_nfev'], gradient_tolerance=solver['gradient_tolerance'],
        finite_difference_step=solver['finite_difference_step'])
    if task['kind'] == 'profile':
        options['fixed_'+task['profile_parameter']] = task['profile_value']
    if task.get('oracle') in ('known_driver', 'known_driver_initial'):
        options['oracle_coefficients'] = case['truth_coefficients']
    if task.get('oracle') == 'known_driver_initial':
        options['oracle_initial_coordinates'] = case['truth_initial']
    starts, fits = [], []
    for index, start in enumerate(solver['starts']):
        fit = fit_free_physiology(target, case['parameters'], s['dt_s'], start_tau_kappa=start, **options)
        row = dict(start=index, success=fit['success'], status=fit['status'],
            convergence_reason=fit['convergence_reason'], nfev=fit['nfev'],
            forward_calls=fit['forward_calls'], elapsed_s=fit['elapsed_s'],
            objective=fit.get('objective'), objective_components=fit.get('objective_components'),
            metrics=recovery_metrics(fit, case['truth'], task, cfg),
            failure_log=fit['failure_log'], trace=fit['trace'])
        starts.append(row)
        fits.append(fit)
    finite = [i for i, f in enumerate(fits) if np.isfinite(f.get('objective', np.inf))]
    eligible = [i for i in finite if fits[i]['success']] or finite
    if not eligible:
        return dict(**task, status='failed_all_starts', success=False, starts=starts), {}
    best = min(eligible, key=lambda i: fits[i]['objective'])
    fit = fits[best]
    result = dict(**task, status=fit['status'], success=fit['success'], selected_start=best,
        selection='minimum_objective_among_converged_else_minimum_finite_retained_as_failure',
        objective=fit['objective'], objective_components=fit['objective_components'],
        metrics=recovery_metrics(fit, case['truth'], task, cfg), starts=starts,
        independent_unit='generator_seed_with_all_truth_noise_modality_prior_variants_paired')
    arrays = {key: fit[key] for key in ('prediction', 'drivers', 'states', 'coefficients', 'initial_coordinates')}
    arrays.update(truth_prediction=case['truth']['prediction'], truth_drivers=case['truth']['drivers'],
        truth_states=case['truth']['states'], truth_initial=case['truth']['initial_state'],
        target=case['target'], visible=mask,
        start_tau_kappa=np.asarray([[f.get('tau', np.nan), f.get('kappa', np.nan)] for f in fits]))
    return result, arrays


def information_rows(cfg, lin, *, scales, masks, sd, feature_weights):
    rows, arrays = [], {}
    rcond = cfg['information']['rank_relative_tolerance']
    for modality, mask in masks.items():
        factors = feature_weights[modality] if isinstance(feature_weights, dict) else feature_weights
        data = weighted_jacobian(lin['observation_jacobian'], sd, mask, factors)
        for multiplier in cfg['priors']['multipliers']:
            prior = np.sqrt(multiplier)*lin['prior_jacobian']
            decomp = information_decomposition(data, prior, scales, rcond)
            profile = nuisance_profile_information(data*scales, [-2 % len(scales), -1 % len(scales)], rcond)
            total_profile = nuisance_profile_information(np.vstack((data, prior))*scales,
                [len(scales)-2, len(scales)-1], rcond)
            fractions = decomp['prior_fraction_by_total_direction']
            row = dict(modality=modality, prior_multiplier=multiplier,
                data_rank=decomp['data_rank'], total_rank=decomp['total_rank'],
                unknowns=len(scales), data_condition=decomp['data_condition'],
                total_condition=decomp['total_condition'],
                data_smallest_singular=float(decomp['data_singular_values'][-1]),
                total_smallest_singular=float(decomp['total_singular_values'][-1]),
                prior_fraction_mean=float(np.nanmean(fractions)) if np.isfinite(fractions).any() else None,
                tau_compensation_fraction=float(profile['compensation_fraction'][0]),
                kappa_compensation_fraction=float(profile['compensation_fraction'][1]),
                theta_profile_min_eigen=float(np.linalg.eigvalsh(profile['profiled_information'])[0]),
                theta_augmented_profile_min_eigen=float(np.linalg.eigvalsh(total_profile['profiled_information'])[0]))
            rows.append(row)
            prefix = f'{modality}__prior{multiplier:g}__'
            for name in ('data_singular_values', 'total_singular_values', 'prior_fraction_by_total_direction', 'total_directions'):
                arrays[prefix+name] = decomp[name]
            arrays[prefix+'profiled_theta_information'] = profile['profiled_information']
            arrays[prefix+'nuisance_compensation'] = profile['compensating_nuisance']
    return rows, arrays


def local_task(cfg, task):
    case = synthetic_case(cfg, task)
    s = cfg['synthetic']
    masks = {m: modality_mask(case['target'].shape, m) for m in s['local_modalities']}
    rows, arrays = [], {}
    for modes in cfg['information']['driver_mode_controls']:
        basis = temporal_mode_basis(s['steps'], modes)
        # Dense-basis control represents exactly the same true r(t), rather
        # than changing the generator signal while enlarging nuisance space.
        coefficient = np.linalg.lstsq(basis, case['truth']['drivers'], rcond=1e-12)[0]
        if not np.allclose(basis@coefficient, case['truth']['drivers'], rtol=1e-12, atol=1e-13):
            raise ValueError('driver-mode control changed generator trajectory')
        lin = pack_spatial_linearization(coefficient, case['truth_initial'],
            case['parameters'], s['dt_s'], sd=s['weighting_sd'],
            feature_weights=s['feature_weights'], **prior_options(cfg, 1.),
            **dict(case['forward_kwargs'], temporal_basis=basis))
        values, blocks = information_rows(cfg, lin, scales=scales_for(cfg, modes),
            masks=masks, sd=s['weighting_sd'], feature_weights=s['feature_weights'])
        rows.extend(dict(**row, driver_modes=modes) for row in values)
        arrays.update({('' if modes == s['temporal_modes'] else f'M{modes}__')+key: value
                       for key, value in blocks.items()})
    return dict(**task, status='completed', success=True, information=rows,
        interpretation='local_weighted_engineering_sensitivity_at_generator_truth_not_global_identifiability'), arrays


def retained_task(cfg, task):
    root = ROOT/cfg['data']['spatial_run']
    spec = task['source_cell']
    metadata = read_json(root/'cells'/f"{spec['id']}.json")
    if not metadata['success']:
        return dict(**task, status='unavailable_source_failure', success=False,
                    source_status=metadata['status']), {}
    calibration = read_json(root/'calibration'/f"{spec['arm']}.json")
    coordinate, source_cfg = read_json(root/'coordinate.json'), read_json(root/'resolved_config.json')
    record_id = spec['record_id']
    # The source owner's exact retained identity hash; never glob inputs.
    prepared_id = hashlib.sha256(record_id.encode()).hexdigest()[:24]
    prepared_meta = read_json(root/'prepared'/f'{prepared_id}.json')
    if prepared_meta.get('status') != 'completed' or prepared_meta['row']['id'] != record_id:
        raise ValueError('retained prepared identity/status mismatch')
    with np.load(root/'prepared'/f'{prepared_id}.npz', allow_pickle=False) as data:
        eeg_op, hb_op = data['eeg_operator'].copy(), data['hb_operator'].copy()
    with np.load(root/'cells'/f"{spec['id']}.npz", allow_pickle=False) as data:
        values = {key: data[key].copy() for key in data.files}
    p = source_cfg['fixed']
    parameters = BalloonParameters(free=BalloonFreeParameters(tau=p['tau'], kappa=p['kappa']))
    parameters = replace(parameters, fixed=replace(parameters.fixed,
        **{name: p[name] for name in ('alpha', 'E0', 'gamma', 'P0', 'Q0', 'eeg_loading', 'neurovascular_gain')}))
    loading = np.asarray(calibration['readout']['eeg_loadings'])
    t, f = len(values['prediction']), len(loading)
    sd = np.asarray(coordinate['sd'])
    source_solver = source_cfg['solver']
    lin = pack_spatial_linearization(values['coefficients'], values['initial_coordinates'],
        parameters, source_cfg['tensor']['dt_s'], sd=sd, temporal_basis=temporal_mode_basis(t, values['coefficients'].shape[0]),
        eeg_loadings=loading, hb_loadings=calibration['design']['hb_loadings'],
        eeg_operator=eeg_op, hb_operator=hb_op, substeps=source_solver['substeps'],
        numerical_backend=source_solver['numerical_backend'],
        **{name: source_solver[name] for name in ('driver_curvature_weight', 'initial_weight', 'flow_prior_weight', 'flow_log_sd')})
    error = float(np.max(np.abs(lin['forward']['prediction']-values['prediction'])))
    if error > 1e-10:
        raise ValueError(f'retained forward reproduction failed: {error}')
    # Fixed complete-panel modality weights make partial/full controls nested.
    factors = np.r_[np.full(f, 1/np.sqrt(f)), np.full(len(sd)-f, 1/np.sqrt(len(sd)-f))]
    masks = {m: modality_mask(values['prediction'].shape, m, f) for m in
             ('HbO_only', 'Hb_half_rate', 'Hb_only', 'EEG_Hb')}
    masks['retained_fit_support'] = values['visible'].copy()
    if metadata['eeg_weight'] == 0:
        masks['retained_fit_support'][:, :f] = False
    # Actual selected EEG multiplier is part of the source objective. The nested
    # positive EEG control is explicitly hypothetical at this fixed solution.
    weights = {name: factors.copy() for name in masks}
    weights['retained_fit_support'][:f] *= np.sqrt(metadata['eeg_weight'])
    rows, arrays = information_rows(cfg, lin,
        scales=scales_for(cfg, values['coefficients'].shape[0], values['coefficients'].shape[1]),
        masks=masks, sd=sd, feature_weights=weights)
    return dict(**task, status='completed', success=True, information=rows,
        retained_forward_max_absolute_error=error, source_eeg_weight=metadata['eeg_weight'],
        subject=spec['subject'], dataset=spec['dataset'], record_id=record_id,
        source_parameter_inference='fixed_physiology_conditional_reconstruction',
        nested_positive_EEG='hypothetical_added_readout_sensitivity_at_same_retained_solution'), arrays


def task_manifest(cfg, *, pilot=False):
    s, p = cfg['synthetic'], cfg['profiles']
    tasks = []
    def add(**task):
        task['id'] = token(task)
        tasks.append(task)
    for tau, kappa, seed in itertools.product(s['tau_truths'], s['kappa_truths'], s['seeds']):
        base = dict(tau_truth=tau, kappa_truth=kappa, seed=seed)
        add(kind='local_information', **base, noise_multiplier=0.)
        for noise, modality, prior in itertools.product(s['noise_multipliers'], s['recovery_modalities'], cfg['priors']['multipliers']):
            add(kind='recovery', **base, noise_multiplier=noise, modality=modality, prior_multiplier=prior)
        for modality, oracle in itertools.product(s['recovery_modalities'], cfg['controls']['oracles']):
            add(kind='oracle', **base, noise_multiplier=cfg['controls']['oracle_noise_multiplier'],
                modality=modality, prior_multiplier=cfg['controls']['oracle_prior_multiplier'], oracle=oracle)
        if seed == cfg['controls']['shifted_prior_seed']:
            for modality, center in itertools.product(s['recovery_modalities'], cfg['priors']['shifted_prior_centers']):
                add(kind='shifted_prior', **base, noise_multiplier=cfg['controls']['shifted_prior_noise_multiplier'],
                    modality=modality, prior_multiplier=cfg['controls']['shifted_prior_base_multiplier'], shifted_center=center)
        if seed in p['seeds']:
            for modality, prior, parameter in itertools.product(s['recovery_modalities'], p['prior_multipliers'], ('tau', 'kappa')):
                for value in p[parameter+'_grid']:
                    add(kind='profile', **base, noise_multiplier=p['noise_multiplier'], modality=modality,
                        prior_multiplier=prior, profile_parameter=parameter, profile_value=value)
    if read_json(ROOT/SPATIAL/'summary.json').get('execution') != 'completed':
        raise ValueError('retained spatial source is not completed')
    spatial_plan = read_json(ROOT/SPATIAL/'plan.json')
    for cell in spatial_plan['cells']:
        if (cell['arm'] in cfg['data']['retained_arms'] and cell['mode'] == cfg['data']['retained_mode']
                and cell['pairing'] == cfg['data']['retained_pairing'] and cell['method'] == cfg['data']['retained_method']):
            add(kind='retained_information', source_cell=cell)
    if pilot:
        selectors = [
            lambda x: x['kind'] == 'recovery' and x['modality'] == 'Hb_only' and x['prior_multiplier'] == 0.,
            lambda x: x['kind'] == 'recovery' and x['modality'] == 'EEG_Hb' and x['prior_multiplier'] == 1.,
            lambda x: x['kind'] == 'recovery' and x['modality'] == 'Hb_only' and x['prior_multiplier'] == 10.,
            lambda x: x['kind'] == 'profile', lambda x: x['kind'] == 'oracle',
            lambda x: x['kind'] == 'shifted_prior', lambda x: x['kind'] == 'local_information',
            lambda x: x['kind'] == 'retained_information']
        tasks = [next(task for task in tasks if predicate(task)) for predicate in selectors]
    counts = {k: sum(t['kind'] == k for t in tasks) for k in sorted({t['kind'] for t in tasks})}
    return dict(schema='source_observability_manifest_v1', experiment_id=cfg['experiment_id'],
        mode='pilot' if pilot else 'formal', tasks=tasks, planned=len(tasks), task_counts=counts,
        planned_optimizer_starts=3*sum(t['kind'] not in ('retained_information', 'local_information') for t in tasks),
        scientific_verdict='not_assigned', source_identity={path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in SOURCES},
        config_identity=token(cfg), protected_data_read=False, raw_data_read=False,
        denominator='fixed_manifest_each_start_retained_no_failure_exclusion')


def prepare(cfg, out, *, pilot=False):
    out = Path(out).resolve()
    if out.parent != (ROOT/ARTIFACT_ROOT).resolve():
        raise ValueError('output must be one fresh run directly below owning artifact root')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        manifest = read_json(out/'manifest.json')
        if manifest['config_identity'] != token(cfg) or manifest['mode'] != ('pilot' if pilot else 'formal'):
            raise ValueError('immutable run config/mode differs')
        current = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in SOURCES}
        if current != manifest['source_identity']:
            raise ValueError('source changed since plan; use a fresh versioned run')
        return manifest
    manifest = task_manifest(cfg, pilot=pilot)
    for path in SOURCES:
        dest = out/'source_snapshot'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/path, dest)
    write_json(out/'resolved_config.json', cfg)
    write_json(out/'manifest.json', manifest)
    return manifest


def execute_task(payload):
    cfg, out, task = payload
    path = Path(out)/'tasks'/f"{task['id']}.json"
    if path.exists():
        return read_json(path)
    start = time.monotonic()
    try:
        if task['kind'] == 'local_information':
            result, arrays = local_task(cfg, task)
        elif task['kind'] == 'retained_information':
            result, arrays = retained_task(cfg, task)
        else:
            result, arrays = fit_task(cfg, task)
        if arrays:
            write_npz(path.with_suffix('.npz'), **arrays)
    except Exception as exc:
        result = dict(**task, success=False, status='failed_exception', error=str(exc), traceback=traceback.format_exc())
    result['elapsed_s'] = time.monotonic()-start
    write_json(path, result)
    return result


def run_tasks(cfg, out, manifest, workers):
    if workers < 1:
        raise ValueError('workers must be positive')
    started = time.monotonic()
    tasks = [task for task in manifest['tasks'] if not (Path(out)/'tasks'/f"{task['id']}.json").exists()]
    completed = manifest['planned']-len(tasks)
    counts = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending, iterator = set(), iter(tasks)
        while True:
            while len(pending) < workers*cfg['resources']['max_in_flight_per_worker']:
                try:
                    pending.add(pool.submit(execute_task, (cfg, str(out), next(iterator))))
                except StopIteration:
                    break
            if not pending:
                break
            ready, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                result = future.result()
                completed += 1
                counts[result['status']] = counts.get(result['status'], 0)+1
                progress = dict(terminal=completed, planned=manifest['planned'], elapsed_s=time.monotonic()-started,
                    new_status_counts=counts, workers=workers)
                write_json(Path(out)/'progress.json', progress)
                print(json.dumps(progress), flush=True)
    return summarize(cfg, out)


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{k: json.dumps(serial(v)) if isinstance(v, (list, dict)) else serial(v)
                          for k, v in row.items()} for row in rows])


def figures(out, records, information):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
                         'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10})
    folder = Path(out)/'figures'
    folder.mkdir(exist_ok=True)
    paths = []
    def save(fig, name):
        fig.savefig(folder/name, dpi=240, bbox_inches='tight')
        plt.close(fig)
        paths.append('figures/'+name)
    local = next((r for r in records if r['kind'] == 'local_information' and r['success']), None)
    if local:
        with np.load(Path(out)/'tasks'/f"{local['id']}.npz", allow_pickle=False) as arrays:
            fig, axes = plt.subplots(2, 1, figsize=(7, 6.5))
            for modality in ('Hb_only', 'EEG_Hb'):
                for kind, style in (('data', '-'), ('total', '--')):
                    v = arrays[f'{modality}__prior1__{kind}_singular_values']
                    axes[0].semilogy(np.arange(1, len(v)+1), np.maximum(v, 1e-14), style, label=f'{modality}: {kind}')
                axes[1].plot(arrays[f'{modality}__prior1__prior_fraction_by_total_direction'], label=modality)
            axes[0].set(xlabel='Fixed-scale singular direction', ylabel='Weighted sensitivity')
            axes[1].set(xlabel='Total-information direction (strong to weak)', ylabel='Prior / total information', ylim=(-.03, 1.03))
            for ax in axes:
                ax.legend(fontsize=10)
                ax.grid(alpha=.2)
            fig.suptitle('One declared generator truth; local geometry, no posterior interpretation')
            fig.tight_layout()
            save(fig, 'data_prior_information.png')
    recovery = [r for r in records if r['kind'] == 'recovery' and r.get('metrics') and r['noise_multiplier'] == 1.]
    if recovery:
        fig, axes = plt.subplots(2, 1, figsize=(7, 7))
        for modality, marker in (('Hb_only', 'o'), ('EEG_Hb', '^')):
            part = [r for r in recovery if r['modality'] == modality]
            for ax, key in zip(axes, ('theta_log_rmse', 'driver_nrmse')):
                points = ax.scatter([r['metrics']['hb_clean_rmse'] for r in part],
                    [r['metrics'][key] for r in part], c=[r['prior_multiplier'] for r in part],
                    marker=marker, cmap='viridis', vmin=0, vmax=10, label=modality, alpha=.7, s=25)
                ax.set(xlabel='Clean Hb prediction RMSE (model units)', ylabel=key)
                ax.grid(alpha=.2)
        for ax in axes:
            ax.legend(fontsize=10)
        fig.colorbar(points, ax=axes.tolist(), label='Prior multiplier', pad=.1)
        fig.suptitle('Each point: truth × seed × modality × prior; variants are paired\nFailed finite fits remain visible and are counted in tables')
        save(fig, 'fit_versus_recovery.png')
    profile = [r for r in records if r['kind'] == 'profile' and 'objective_components' in r]
    if profile:
        example = min((r['tau_truth'], r['kappa_truth'], r['seed']) for r in profile)
        part = [r for r in profile if (r['tau_truth'], r['kappa_truth'], r['seed']) == example]
        fig, axes = plt.subplots(2, 2, figsize=(7, 6.5))
        for col, parameter in enumerate(('tau', 'kappa')):
            for modality in ('Hb_only', 'EEG_Hb'):
                for prior in (0., 1.):
                    group = sorted([r for r in part if r['profile_parameter'] == parameter
                        and r['modality'] == modality and r['prior_multiplier'] == prior], key=lambda r: r['profile_value'])
                    if not group:
                        continue
                    x = [r['profile_value'] for r in group]
                    label = f'{modality}, prior={prior:g}'
                    axes[0, col].plot(x, [r['objective_components']['data'] for r in group], 'o-', label=label)
                    axes[1, col].plot(x, [r['objective']-r['objective_components']['data'] for r in group], 'o-')
            for row in (0, 1):
                axes[row, col].axvline(example[col], color='k', ls=':', label='Truth' if row == 0 else None)
                axes[row, col].set_xscale('log')
                axes[row, col].set(xlabel=parameter, ylabel='Weighted data SSE' if row == 0 else 'Separate prior penalty')
                axes[row, col].grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10)
        fig.suptitle('Nonlinear profiles: free nuisance refit at every node\nEngineering objective; no confidence-interval claim')
        fig.tight_layout()
        fig.subplots_adjust(bottom=.2, top=.88)
        save(fig, 'nonlinear_profiles.png')
    if recovery:
        # Prespecified display rule: lowest tau/kappa truth, first seed, Hb-only;
        # show all available prior variants without choosing a flattering fit.
        first = min((r['tau_truth'], r['kappa_truth'], r['seed']) for r in recovery)
        group = sorted([r for r in recovery if (r['tau_truth'], r['kappa_truth'], r['seed']) == first
                        and r['modality'] == 'Hb_only'], key=lambda r: r['prior_multiplier'])
        if group:
            fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
            for index, row in enumerate(group):
                with np.load(Path(out)/'tasks'/f"{row['id']}.npz", allow_pickle=False) as data:
                    time_s = np.arange(len(data['prediction']))*.25
                    if index == 0:
                        axes[0].plot(time_s, data['truth_prediction'][:, 3], 'k--', label='Noise-free generated HbO')
                        axes[1].plot(time_s, data['truth_drivers'][:, 0], 'k--', label='Generator driver')
                        axes[2].plot(time_s, data['truth_states'][:, 0, 2], 'k--', label='Generator relative flow')
                    label = f"prior={row['prior_multiplier']:g}; tau={row['metrics']['tau']:.2f}, kappa={row['metrics']['kappa']:.2f}"
                    axes[0].plot(time_s, data['prediction'][:, 3], label=label)
                    axes[1].plot(time_s, data['drivers'][:, 0])
                    axes[2].plot(time_s, data['states'][:, 0, 2])
            for ax, ylabel in zip(axes, ('HbO (model units)', 'Driver (fixed gauge)', 'Relative flow f')):
                ax.set_ylabel(ylabel)
                ax.grid(alpha=.2)
                ax.legend(fontsize=10)
            axes[-1].set_xlabel('Generator time (s)')
            fig.suptitle('Predetermined first Hb-only noisy case; prior tradeoff shown without posthoc alignment')
            fig.tight_layout()
            save(fig, 'prediction_driver_state_example.png')
    return paths


def summarize(cfg, out):
    out = Path(out)
    manifest = read_json(out/'manifest.json')
    records = [read_json(out/'tasks'/f"{task['id']}.json") for task in manifest['tasks']
               if (out/'tasks'/f"{task['id']}.json").exists()]
    counts, rows, starts, information = {}, [], [], []
    for record in records:
        counts[record['status']] = counts.get(record['status'], 0)+1
        base = {k: record.get(k) for k in ('id', 'kind', 'seed', 'tau_truth', 'kappa_truth',
            'noise_multiplier', 'modality', 'prior_multiplier', 'profile_parameter', 'profile_value', 'oracle', 'shifted_center')}
        if record.get('metrics'):
            rows.append(dict(**base, success=record['success'], status=record['status'],
                objective=record['objective'], elapsed_s=record['elapsed_s'],
                **record['metrics'], **{'objective_'+k: v for k, v in record['objective_components'].items()}))
        for start in record.get('starts', []):
            starts.append(dict(**base, start=start['start'], success=start['success'], status=start['status'],
                objective=start.get('objective'), nfev=start['nfev'], elapsed_s=start['elapsed_s'], **start['metrics']))
        for info in record.get('information', []):
            ibase = {k: v for k, v in base.items() if k not in ('modality', 'prior_multiplier')}
            information.append(dict(**ibase, subject=record.get('subject'),
                arm=record.get('source_cell', {}).get('arm'), **info))
    write_csv(out/'recovery_metrics.csv', rows)
    write_csv(out/'optimizer_starts.csv', starts)
    write_csv(out/'information_metrics.csv', information)
    groups = {}
    def group_key(row):
        return (row['kind'], row.get('modality'), row.get('prior_multiplier'), row.get('noise_multiplier'),
            row.get('oracle') or '', json.dumps(row.get('shifted_center') or []),
            row.get('profile_parameter') or '', row.get('profile_value') or 0.)
    planned_groups = {}
    for task in manifest['tasks']:
        if task['kind'] not in ('retained_information', 'local_information'):
            planned_groups.setdefault(group_key(task), []).append(task)
    for values, planned in planned_groups.items():
        kind, modality, prior, noise, oracle, center, parameter, profile_value = values
        terminal = [record for record in records if group_key(record) == values]
        valid = [r for r in rows if group_key(r) == values and r['success']]
        key = f'{kind}|{modality}|prior={prior:g}|noise={noise:g}'
        if oracle:
            key += '|'+oracle
        if center != '[]':
            key += '|center='+center
        if parameter:
            key += f'|{parameter}={profile_value:g}'
        groups[key] = dict(planned=len(planned), terminal=len(terminal), succeeded=len(valid),
            failures=sum(not record['success'] for record in terminal), pending=len(planned)-len(terminal),
            summary_scope='successful_subset_with_complete_failure_denominator_shown',
            medians={metric: float(np.median([r[metric] for r in valid])) if valid else None for metric in
                ('theta_log_rmse', 'tau_absolute_error', 'kappa_absolute_error', 'driver_nrmse',
                 'initial_physical_rmse', 'hb_clean_rmse', 'hb_clean_standardized_nrmse')})
    source_support = ROOT/SUPPORT
    support_rows = list(csv.DictReader((source_support/'local_information.csv').open()))
    source_excerpt = dict(path=SUPPORT, rows=len(support_rows),
        interpretation='retained_output_information_witnesses_not_parameter_recovery',
        information_definitions='original_source_fields_retained_verbatim',
        first_rows=support_rows[:4])
    write_json(out/'retained_output_support_context.json', source_excerpt)
    figure_paths = figures(out, records, information)
    summary = dict(schema='source_observability_summary_v1', experiment_id=cfg['experiment_id'],
        execution='completed' if len(records) == manifest['planned'] else 'incomplete',
        planned=manifest['planned'], terminal=len(records), status_counts=counts,
        task_counts=manifest['task_counts'], optimizer_starts=len(starts),
        successful_optimizer_starts=sum(bool(r['success']) for r in starts), groups=groups,
        finite_manifest=True, raw_data_read=False, protected_data_read=False,
        evidence_tables=['recovery_metrics.csv', 'optimizer_starts.csv', 'information_metrics.csv'],
        figures=figure_paths, claim_limits=cfg['claim_limits'],
        implementation_semantics=dict(data='fixed_SD_and_fixed_feature_weight_standardized_sum_of_squares',
            priors='curvature_physical_initial_departure_logflow_separate_with_optional_logtheta_anchor',
            profile='each_fixed_tau_or_kappa_refits_all_other_free_coordinates_from_three_starts',
            truth='noise_free_nonlinear_generator_with_nonrest_initials_substeps8_fit_substeps4',
            local='analytic_driver_initial_plus_central_logtau_logkappa_derivatives',
            retained='old_fixed_physiology_fits_local_diagnostics_only_no_measured_physiology_recovery'))
    write_json(out/'summary.json', summary)
    return summary


def check(cfg):
    task = dict(tau_truth=1.3, kappa_truth=.45, seed=cfg['synthetic']['seeds'][0], noise_multiplier=0.)
    case = synthetic_case(cfg, task)
    assert np.isfinite(case['truth']['states']).all()
    return dict(status='pass', schema=cfg['schema'], synthetic_shape=list(case['target'].shape),
                raw_data_read=False, protected_data_read=False)


# The feasibility diagnostic reuses this suite's retained-evidence boundary and
# artifact owner, while preserving the original suite's config and dispatch.
FEASIBILITY_CONFIG = ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_feasibility_v1.yaml'
FEASIBILITY_ARMS = ('K1_true_H4_E12', 'K2_true_H4_E12')
FEASIBILITY_STAGES = ('check', 'plan', 'synthetic', 'pilot', 'bounds', 'vascular',
                      'frontier', 'profiles', 'summary')


def feasibility_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    sources = cfg.get('sources', cfg.get('data', {}))
    source = sources.get('source_run', cfg.get('source_run'))
    if (cfg.get('schema') != 'shared_driver_feasibility_v1'
            or cfg.get('experiment_id') != 'SSM-FEASIBILITY-v1'
            or cfg.get('artifact_root') != ARTIFACT_ROOT or source != SPATIAL
            or sources.get('protected_data') != 'forbidden'
            or sources.get('raw_or_cache_reads', sources.get('raw_data')) != 'forbidden'):
        raise ValueError('exact retained public spatial source and feasibility identity required')
    if tuple(cfg.get('model', {}).get('arms', cfg.get('arms', []))) != FEASIBILITY_ARMS:
        raise ValueError('the two frozen true-geometry full-panel arms are required')
    if (sources.get('split')!='evaluation' or cfg['model'].get('steps')!=120
            or cfg['model'].get('dt_s')!=.25 or cfg['model'].get('temporal_modes')!=16
            or cfg['resources'].get('numerical_threads')!=1
            or not 1<=cfg['resources'].get('max_workers',0)<=24
            or cfg['solver'].get('validation_substeps',0)<=cfg['solver'].get('substeps',0)
            or cfg['frontier'].get('parameter_branches')!=['fixed','free_tau_kappa']):
        raise ValueError('frozen full-visible tensor, parameter branches and bounded numerical resources required')
    scoring = cfg.get('scoring', {})
    if (scoring.get('score_steps', [20, 120]) != [20, 120]
            or scoring.get('primary_tolerance', .5) != .5
            or scoring.get('tolerances', [.25, .5, .75, 1., 1.5]) != [.25, .5, .75, 1., 1.5]):
        raise ValueError('fixed score support and separate EEG/HbO/HbR tolerances required')
    return cfg


def feasibility_source(cfg):
    """Validate the one public metadata boundary before any np.load call."""
    source = (ROOT/SPATIAL).resolve()
    if source != ROOT.resolve()/SPATIAL:
        raise ValueError('the explicit parent path must not redirect outside its retained identity')
    names=('run_identity.json','resolved_config.json','plan.json','coordinate.json','summary.json','verification.json')
    if any(not (source/name).resolve().is_relative_to(source) for name in names):
        raise ValueError('parent metadata symlink escaped the explicit retained parent')
    identity, parent, plan, coordinate, summary = [read_json(source/name) for name in
        ('run_identity.json', 'resolved_config.json', 'plan.json', 'coordinate.json', 'summary.json')]
    verification = read_json(source/'verification.json')
    if (identity.get('schema') != 'shared_driver_spatial_v1' or identity.get('family') != 'public'
            or parent.get('schema') != 'shared_driver_spatial_v1'
            or parent.get('data', {}).get('datasets') != ['eeg_fnirs_single_trial']
            or parent['data'].get('protected_data') != 'forbidden'
            or parent['tensor'].get('steps') != 120 or parent['tensor'].get('dt_s') != .25
            or parent['tensor'].get('temporal_modes') != 16
            or parent['evaluation'].get('score_steps') != [20, 120]
            or summary.get('execution') != 'completed' or summary.get('planned') != summary.get('terminal')
            or verification.get('status') != 'passed' or plan.get('status') != 'available'):
        raise ValueError('completed independently verified public spatial parent required')
    groups, ids = {}, set()
    for row in plan['rows']:
        if (row.get('dataset') != 'eeg_fnirs_single_trial'
                or row.get('split') not in ('train', 'selection', 'evaluation')
                or row['id'] in ids or row['split'] != groups.setdefault(row['subject'], row['split'])):
            raise ValueError('parent row identity, public dataset or subject split changed')
        ids.add(row['id'])
    training = {r['id'] for r in plan['rows'] if r['split'] == 'train'}
    if set(coordinate['training_ids']) != training:
        raise ValueError('frozen coordinate training identities changed')
    sd = np.asarray(coordinate['sd'], float)
    if sd.shape != (68,) or not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError('positive frozen training SD for all 68 features required')
    calibrations = {}
    for arm in FEASIBILITY_ARMS:
        if not (source/'calibration'/(arm+'.json')).resolve().is_relative_to(source):
            raise ValueError('calibration metadata symlink escaped the explicit retained parent')
        calibration = read_json(source/'calibration'/(arm+'.json'))
        loading = np.asarray(calibration.get('readout', {}).get('eeg_loadings'), float)
        k = int(arm[1])
        if (calibration.get('status') != 'completed' or calibration.get('eeg_readout_status') != 'learned'
                or calibration.get('evaluation_data_used') is not False
                or not set(calibration['readout']['training_ids']) <= training
                or loading.shape != (60, k) or not np.isfinite(loading).all()
                or np.linalg.matrix_rank(loading) != k
                or np.shape(calibration['design']['hb_loadings']) != (4, k)):
            raise ValueError('frozen train-only full-rank EEG readout and Hb geometry required')
        calibrations[arm] = calibration
    return dict(path=source, identity=identity, parent=parent, plan=plan,
                coordinate=coordinate, calibrations=calibrations)


def feasibility_load(source, row):
    if row not in source['plan']['rows'] or row.get('split') != 'evaluation':
        raise ValueError('only exact retained evaluation identities may select arrays')
    name = hashlib.sha256(row['id'].encode()).hexdigest()[:24]
    path = source['path']/'prepared'/(name+'.npz')
    if (not path.resolve().is_relative_to(source['path'].resolve())
            or not path.with_suffix('.json').resolve().is_relative_to(source['path'].resolve())):
        raise ValueError('prepared symlink escaped the explicit retained parent')
    meta = read_json(path.with_suffix('.json'))
    if meta.get('status') != 'completed' or meta.get('row') != row:
        raise ValueError('prepared identity not completed or mismatched before array read')
    with np.load(path, allow_pickle=False) as values:
        arrays = {key: values[key].copy() for key in ('eeg', 'hb', 'eeg_operator', 'hb_operator')}
    shapes = dict(eeg=(120, 12, 5), hb=(120, 4, 2), eeg_operator=(120, 120), hb_operator=(120, 120))
    if any(arrays[key].shape != shape or not np.isfinite(arrays[key]).all() for key, shape in shapes.items()):
        raise ValueError('finite complete frozen 120-step spatial arrays required')
    from src.inference.observation_baselines import transform_spatial_coordinates
    eeg, hb = transform_spatial_coordinates(arrays['eeg'], arrays['hb'], source['coordinate'])
    arrays.update(eeg=eeg, hb=hb)
    return arrays


def feasibility_forward(source, record, arm):
    calibration = source['calibrations'][arm]
    return dict(temporal_basis=temporal_mode_basis(120, 16),
        eeg_loadings=np.asarray(calibration['readout']['eeg_loadings'], float),
        hb_loadings=np.asarray(calibration['design']['hb_loadings'], float),
        hb_gains=np.ones(4), eeg_operator=record['eeg_operator'], hb_operator=record['hb_operator'],
        substeps=4, numerical_backend=source['parent']['solver']['numerical_backend'])


def feasibility_metrics(prediction, target, sd, score_steps=(20, 120)):
    p, y, scale = np.asarray(prediction, float), np.asarray(target, float), np.asarray(sd, float)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != 68 or scale.shape != (68,):
        raise ValueError('feasibility metrics require fixed [T,68] prediction/target and SD')
    take = slice(*score_steps)
    if not np.isfinite(p[take]).all() or not np.isfinite(y[take]).all():
        raise ValueError('finite complete scoring support required')
    result = {}
    for name, columns in [('EEG', np.arange(60)), ('HbO', np.arange(60, 68, 2)), ('HbR', np.arange(61, 68, 2))]:
        a, b, s = p[take][:, columns], y[take][:, columns], scale[columns]
        delta = (a-b)/s
        ac, bc = a-a.mean(axis=0), b-b.mean(axis=0)
        denominator = np.linalg.norm(ac, axis=0)*np.linalg.norm(bc, axis=0)
        correlation = np.divide(np.sum(ac*bc, axis=0), denominator,
            out=np.full(len(columns), np.nan), where=denominator > 0)
        target_rms = float(np.sqrt(np.mean((b/s)**2)))
        ratio = float(np.sqrt(np.mean((a/s)**2))/target_rms) if target_rms > 0 else None
        centered_target_energy = float(np.mean((bc/s)**2))
        centered_error = float(np.mean(((ac-bc)/s)**2))
        nmse = float(np.mean(delta**2))
        result[name] = dict(nmse=nmse, nrmse=float(np.sqrt(nmse)),
            centered_nrmse=float(np.sqrt(np.mean(((ac-bc)/s)**2))),
            centered_local_nrmse=float(np.sqrt(centered_error/centered_target_energy)) if centered_target_energy > 0 else None,
            centered_amplitude_ratio=float(np.sqrt(np.mean((ac/s)**2)/centered_target_energy)) if centered_target_energy > 0 else None,
            mean_feature_correlation=float(np.nanmean(correlation)) if np.isfinite(correlation).any() else None,
            feature_correlations=serial(correlation), amplitude_rms_ratio=ratio,
            n_scored=int(a.size), denominator='frozen_parent_training_SD',
            morphology='time_centering_per_feature_only_for_centered_error_and_correlation')
    return result


def feasibility_baseline(source, row, arm):
    matches = [cell for cell in source['plan']['cells'] if cell['record_id'] == row['id']
        and cell['arm'] == arm and cell['mode'] == 'full' and cell['pairing'] == 'paired' and cell['method'] == 'SSM']
    if len(matches) != 1:
        raise ValueError('one exact retained full conditional reconstruction required')
    spec = matches[0]
    path = source['path']/'cells'/(spec['id']+'.json')
    if (not path.resolve().is_relative_to(source['path'].resolve())
            or not path.with_suffix('.npz').resolve().is_relative_to(source['path'].resolve())):
        raise ValueError('baseline symlink escaped the explicit retained parent')
    meta = read_json(path)
    if any(meta.get(key) != spec.get(key) for key in ('id', 'record_id', 'arm', 'mode', 'pairing', 'method')):
        raise ValueError('retained baseline identity mismatch before array read')
    arrays = None
    if path.with_suffix('.npz').exists():
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as values:
            arrays = {key: values[key].copy() for key in
                ('prediction', 'coefficients', 'initial_coordinates', 'states')}
    return meta, arrays


def feasibility_plan(cfg, source):
    rows = [row for row in source['plan']['rows'] if row['split'] == 'evaluation']
    tasks = []
    for row, arm in itertools.product(rows, FEASIBILITY_ARMS):
        task = dict(kind='bounds', arm=arm, row=row)
        task['id'] = token(dict(record_id=row['id'], arm=arm, kind='bounds'))
        tasks.append(task)
    return dict(schema='shared_driver_feasibility_plan_v1', experiment_id=cfg['experiment_id'],
        source_run=SPATIAL, config_identity=token(cfg), rows=rows, tasks=tasks,
        planned=len(tasks), evaluation_windows=len(rows), evaluation_subjects=len({r['subject'] for r in rows}),
        score_steps=[20, 120], conditional_support='full_visible_EEG_and_all_four_Hb_pairs_offline_reconstruction',
        selection='all_parent_evaluation_windows_no_error_screening', protected_data_read=False, raw_data_read=False,
        metadata_only=True)


def feasibility_prepare(cfg, out, source, *, freeze=False):
    out = Path(out).resolve()
    if out.parent != (ROOT/ARTIFACT_ROOT).resolve() or out == source['path']:
        raise ValueError('a versioned run directly below the existing source-observability root is required')
    out.mkdir(parents=True, exist_ok=True)
    manifest = feasibility_plan(cfg, source)
    for name, value in [('resolved_config.json', cfg), ('manifest.json', manifest)]:
        if (out/name).exists() and read_json(out/name) != serial(value):
            raise ValueError('immutable feasibility config or plan changed')
        if not (out/name).exists():
            write_json(out/name, value)
    if freeze:
        paths = list(dict.fromkeys(SOURCES+['src/inference/observation_baselines.py']))
        identity = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths}
        if (out/'source_identity.json').exists():
            if read_json(out/'source_identity.json') != identity:
                raise ValueError('source changed since execution; use a new versioned run')
        else:
            for path in paths:
                dest = out/'source_snapshot'/path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT/path, dest)
            write_json(out/'source_identity.json', identity)
    return manifest


def feasibility_bounds_task(cfg, source, task):
    from src.inference.source_observability import eeg_representation_lower_bound, hb_representation_lower_bound
    record = feasibility_load(source, task['row'])
    kwargs = feasibility_forward(source, record, task['arm'])
    sd = np.asarray(source['coordinate']['sd'], float)
    em = np.zeros((120, 60), bool); em[20:] = True
    hm = np.zeros((120, 4, 2), bool); hm[20:] = True
    eeg = eeg_representation_lower_bound(record['eeg'], temporal_basis=kwargs['temporal_basis'],
        eeg_loadings=kwargs['eeg_loadings'], sd=sd[:60], score_mask=em, eeg_operator=record['eeg_operator'])
    hb = hb_representation_lower_bound(record['hb'], hb_loadings=kwargs['hb_loadings'],
        sd=sd[60:].reshape(4, 2), score_mask=hm, hb_operator=record['hb_operator'])
    ep=eeg['certified_prediction']
    hp=hb['certified_prediction']
    prediction = np.column_stack((ep, hp.reshape(120, 8)))
    target = np.column_stack((record['eeg'], record['hb'].reshape(120, 8)))
    metrics = feasibility_metrics(prediction, target, sd)
    baseline, baseline_arrays = feasibility_baseline(source, task['row'], task['arm'])
    baseline_metrics = feasibility_metrics(baseline_arrays['prediction'], target, sd) if baseline_arrays else None
    tolerances = cfg.get('scoring', {}).get('tolerances', [.25, .5, .75, 1., 1.5])
    margin=cfg.get('representation',{}).get('exclusion_margin',1e-7)
    coverage = {str(tol): dict(
        **{name: metrics[name]['nrmse'] <= tol for name in ('EEG', 'HbO', 'HbR')},
        simultaneous_necessary_condition=all(metrics[name]['nrmse'] <= tol+margin for name in ('EEG', 'HbO', 'HbR')),
        certified_exclusion={name:metrics[name]['nrmse']>tol+margin for name in ('EEG','HbO','HbR')},
        numerical_boundary={name:abs(metrics[name]['nrmse']-tol)<=margin for name in ('EEG','HbO','HbR')})
        for tol in tolerances}
    result = dict(status='completed', success=True, bounds=metrics, tolerance_coverage=coverage,
        numerical_projection=feasibility_metrics(np.column_stack((eeg['prediction'],hb['prediction'].reshape(120,8))),target,sd),
        representation_certificate={name:{key:value[key] for key in ('certificate_type','certificate_relaxed',
            'certificate_projection_dimension','discarded_direction_count','numerical_projection_sse_excess')}
            for name,value in [('EEG',eeg),('Hb',hb)]},
        baseline=dict(status=baseline.get('status'), success=baseline.get('success'),
            eeg_weight=baseline.get('eeg_weight'), metrics=baseline_metrics,
            interpretation='retained_full_conditional_fit_original_selected_weight_not_new_joint_optimum'),
        zero_prediction=feasibility_metrics(np.zeros_like(target), target, sd),
        EEG_rank=eeg['rank'], Hb_rank=hb['rank'],
        normal_equation_residual_norm=dict(EEG=eeg['normal_equation_residual_norm'], Hb=hb['normal_equation_residual_norm']),
        interpretation='certified_conservative_relaxed_linear_representation_lower_bounds_not_necessarily_exact_distance_or_joint_witness',
        EEG_solver=eeg['solver'], Hb_solver=hb['solver'])
    arrays = dict(eeg_bound_prediction=ep, hb_bound_prediction=hp,
        numerical_eeg_projection=eeg['prediction'],numerical_hb_projection=hb['prediction'],
        eeg_coefficients=eeg['coefficients'], latent_hb_prediction=hb['latent_hb_prediction'], target=target,sd=sd)
    if baseline_arrays:
        arrays['baseline_prediction'] = baseline_arrays['prediction']
    return result, arrays


def feasibility_synthetic(cfg):
    """Small exact numerical checks, with no parent metadata or array reads."""
    from src.inference.source_observability import eeg_representation_lower_bound, hb_representation_lower_bound
    rng = np.random.default_rng(20261022)
    t, f = 24, 6
    basis = temporal_mode_basis(t, 4)
    op = np.eye(t); op[:, :4] -= .25
    loading = rng.normal(size=(f, 2))
    coefficients = rng.normal(scale=.02, size=(4, 2))
    signal = op@basis@coefficients@loading.T
    mask = np.zeros_like(signal, bool); mask[4:] = True
    eeg = eeg_representation_lower_bound(signal, temporal_basis=basis,
        eeg_loadings=loading, sd=np.linspace(.01, .03, f), score_mask=mask, eeg_operator=op)
    latent = rng.normal(scale=.01, size=(t, 1, 2))
    h = np.repeat((op@latent[:, 0])[:, None], 3, axis=1)
    hb = hb_representation_lower_bound(h, hb_loadings=np.ones((3, 1)), sd=np.ones((3, 2)), hb_operator=op)
    if eeg['nrmse'] > 1e-10 or hb['nrmse'] > 1e-10:
        raise ValueError('exact representable synthetic reconstruction failed')
    disturbed = h.copy(); disturbed[:, 1, 0] += np.linspace(-.1, .1, t)
    conflict = hb_representation_lower_bound(disturbed, hb_loadings=np.ones((3, 1)), sd=np.ones((3, 2)), hb_operator=op)
    if conflict['nrmse'] <= .001:
        raise ValueError('shared-waveform mismatch lower bound was not detected')
    return dict(schema='shared_driver_feasibility_synthetic_v1', status='passed', success=True,
        representable_EEG_nrmse=eeg['nrmse'], representable_Hb_nrmse=hb['nrmse'],
        incompatible_common_Hb_nrmse=conflict['nrmse'], measured_arrays_read=0,
        interpretation='linear_projection_and_shared_waveform_software_controls_only')


def feasibility_parameters(source, node=None):
    from src.inference.t3a_balloon_robust_ssm import BalloonFixedParameters
    fixed = dict(source['parent']['fixed'])
    tau, kappa = fixed.pop('tau'), fixed.pop('kappa')
    node = node or {}
    for name in ('alpha', 'E0'):
        if name in node:
            fixed[name] = node[name]
    result = BalloonParameters(fixed=BalloonFixedParameters(**fixed),
        free=BalloonFreeParameters(tau=node.get('tau', tau), kappa=node.get('kappa', kappa)))
    result.validate()
    return result


def feasibility_synthetic_plan(cfg, *, pilot=False):
    tasks=[]
    combinations=list(itertools.product(cfg['synthetic']['scenarios'],cfg['synthetic']['seeds']))
    if pilot:
        combinations=[(scenario,cfg['synthetic']['seeds'][0]) for scenario in cfg['synthetic']['scenarios'][:4]]
    for scenario,seed in combinations:
        task=dict(kind='synthetic',scenario=scenario,seed=seed,arm='K1_true_H4_E12')
        task['id']=token(task)
        tasks.append(task)
    return tasks


def feasibility_synthetic_task(cfg,source,task):
    """New generated arrays; parent contributes metadata and frozen calibration only."""
    from src.inference.observation_baselines import disjoint_native_bin_operator
    from src.inference.source_observability import (eeg_representation_lower_bound,
        hb_representation_lower_bound,fit_feasibility_frontier)
    from src.inference.shared_driver_reconstruction import fit_vascular_flow_relaxation
    s=cfg['synthetic']; rng=np.random.default_rng(task['seed']); solver=cfg['solver']
    operators=disjoint_native_bin_operator(np.arange(300)/10.,steps=120,dt=.25,reference_steps=20)
    record=dict(eeg_operator=operators['eeg'],hb_operator=operators['model'])
    kwargs=feasibility_forward(source,record,task['arm'])
    basis=kwargs['temporal_basis']; k=kwargs['eeg_loadings'].shape[1]
    coefficients=rng.normal(size=(16,k))/np.maximum(np.arange(16)[:,None],1.)**1.5
    coefficients*=s['driver_rms']/np.sqrt(np.mean((basis@coefficients)**2))
    initial=np.tile(np.asarray(s['initial_coordinates'],float),(k,1))
    param=feasibility_parameters(source)
    truth=forward_spatial_shared_driver(coefficients,initial,param,.25,
        **dict(kwargs,substeps=s['generator_substeps']),derivative=False)
    sd=np.asarray(source['coordinate']['sd'],float)
    target=truth['prediction'].copy(); mask=np.zeros((120,68),bool); mask[20:]=True
    if task['scenario']=='matched_noisy':
        target+=s['noise_training_SD']*rng.normal(size=target.shape)*sd
    elif task['scenario']=='EEG_orthogonal':
        disturbance=rng.normal(size=(120,60))*sd[:60]
        projection=eeg_representation_lower_bound(disturbance,temporal_basis=basis,
            eeg_loadings=kwargs['eeg_loadings'],sd=sd[:60],score_mask=mask[:,:60],eeg_operator=operators['eeg'])
        disturbance-=projection['prediction']
        disturbance*=s['orthogonal_RMS_training_SD']/np.sqrt(np.mean((disturbance[20:]/sd[:60])**2))
        target[:,:60]+=disturbance
    elif task['scenario']=='Hb_spatial_orthogonal':
        disturbance=rng.normal(size=(120,4,2))*sd[60:].reshape(4,2)
        projection=hb_representation_lower_bound(disturbance,hb_loadings=kwargs['hb_loadings'],
            sd=sd[60:],score_mask=mask[:,60:].reshape(120,4,2),hb_operator=operators['model'])
        disturbance-=projection['prediction']
        disturbance*=s['orthogonal_RMS_training_SD']/np.sqrt(np.mean((disturbance[20:]/sd[60:].reshape(4,2))**2))
        target[:,60:]+=disturbance.reshape(120,8)
    elif task['scenario']=='shared_driver_conflict':
        shifted=coefficients.copy(); shifted[1:]*=-1
        target[:,:60]=(operators['eeg']@basis)@shifted@kwargs['eeg_loadings'].T
    eeg=eeg_representation_lower_bound(target[:,:60],temporal_basis=basis,
        eeg_loadings=kwargs['eeg_loadings'],sd=sd[:60],score_mask=mask[:,:60],eeg_operator=operators['eeg'])
    hb=hb_representation_lower_bound(target[:,60:].reshape(120,4,2),hb_loadings=kwargs['hb_loadings'],
        sd=sd[60:],score_mask=mask[:,60:].reshape(120,4,2),hb_operator=operators['model'])
    prediction=np.column_stack((eeg['certified_prediction'],hb['certified_prediction'].reshape(120,8)))
    bounds=feasibility_metrics(prediction,target,sd)
    arrays=dict(target=target,truth_prediction=truth['prediction'],truth_coefficients=coefficients,
        truth_initial_coordinates=initial,truth_states=truth['states'],truth_drivers=truth['drivers'],
        eeg_bound_prediction=eeg['prediction'],hb_bound_prediction=hb['prediction'],sd=sd)
    fits={}; vascular=None
    # The matched cases calibrate both vascular and joint finite-search behavior.
    # The other controls are still fitted where the exact EEG bound permits it.
    if task['scenario'] in ('matched_noiseless','matched_noisy'):
        vm=np.zeros((120,2),bool);vm[20:]=True
        starts=[dict(label=start['label'],coefficients=np.zeros(120),initial_state=start['initial_physical'])
            for start in cfg['vascular']['starts']]
        vascular=fit_vascular_flow_relaxation(target[:,60:62],param,.25,
            temporal_basis=temporal_mode_basis(120,120),hb_operator=np.kron(operators['model'],np.eye(2)),sd=sd[60:62],
            score_mask=vm,starts=starts,max_nfev=solver['vascular_max_nfev'],
            substeps=solver['substeps'],validation_substeps=solver['validation_substeps'],
            log_flow_bounds=cfg['vascular']['log_flow_bounds'],initial_log_bounds=cfg['vascular']['initial_log_bounds'],
            integration_tolerance=solver['integration_tolerance'])
        for key in ('prediction','states','flow','coefficients'):
            if vascular.get(key) is not None:arrays['vascular_'+key]=vascular[key]
    if bounds['EEG']['nrmse']<=.5+cfg['representation']['exclusion_margin']:
        common=dict(kwargs);common.pop('substeps')
        starts=[dict(label='EEG_projection_rest',coefficients=eeg['coefficients'],initial_coordinates=np.zeros((k,5)),
            tau=param.free.tau,kappa=param.free.kappa)]
        for branch in ('fixed','free_tau_kappa'):
            if branch=='free_tau_kappa' and fits.get('fixed',{}).get('physical_valid'):
                starts=starts+[feasibility_candidate_start(fits['fixed'],'nested_fixed_candidate')]
                if fits['fixed'].get('joint_witness') is not None:
                    starts=starts+[feasibility_candidate_start(fits['fixed']['joint_witness'],'nested_fixed_joint_witness')]
            fit=fit_feasibility_frontier(target[:,:60],target[:,60:].reshape(120,4,2),param,.25,
                **common,sd=sd,score_mask=mask,epsilon=.5,starts=starts,free_physiology=branch=='free_tau_kappa',
                tau_bounds=cfg['frontier']['tau_bounds'],kappa_bounds=cfg['frontier']['kappa_bounds'],
                max_nfev=solver['max_nfev'],substeps=solver['substeps'],verification_substeps=solver['validation_substeps'],
                validation_tolerance=solver['integration_tolerance'],constraint_tolerance=solver['constraint_tolerance'],ftol=solver['ftol'])
            fits[branch]=fit
            for key in ('prediction','states','coefficients','initial_coordinates'):
                if fit.get(key) is not None:arrays[branch+'__'+key]=fit[key]
    recovery={}
    for branch,fit in fits.items():
        if fit.get('coefficients') is not None:
            recovered=basis@fit['coefficients']
            recovery[branch]=dict(driver_nrmse=float(np.sqrt(np.mean((recovered-truth['drivers'])**2))/s['driver_rms']),
                initial_coordinate_rmse=float(np.sqrt(np.mean((fit['initial_coordinates']-initial)**2))),
                tau=float(fit['tau']),kappa=float(fit['kappa']),truth_tau=param.free.tau,truth_kappa=param.free.kappa,
                interpretation='nominal_truth_conditional_recovery_not_non_nominal_parameter_qualification')
    return dict(status='completed',success=True,bounds=bounds,
        vascular=feasibility_fit_metadata(vascular),frontier=feasibility_fit_metadata(fits),recovery=recovery,
        measured_arrays_read=0,truth_as_start=False,
        interpretation='known_truth_finite_solver_and_exclusion_controls_no_measured_or_individual_qualification'),arrays


def feasibility_fit_metadata(value):
    """Keep traces/negative outcomes in JSON, and numerical arrays in the NPZ."""
    from dataclasses import is_dataclass, asdict
    if is_dataclass(value):
        return feasibility_fit_metadata(asdict(value))
    if isinstance(value, np.ndarray):
        return dict(array_shape=list(value.shape))
    if isinstance(value, dict):
        result={key:feasibility_fit_metadata(item) for key,item in value.items()}
        states=value.get('states')
        if isinstance(states,np.ndarray) and states.size:
            diagnostics={}
            if states.ndim==3 and states.shape[2]==6:
                diagnostics['cores']=[{name:dict(min=float(states[:,core,index].min()),max=float(states[:,core,index].max()))
                    for name,index in [('flow',2),('v',3),('p',4),('q',5)]} for core in range(states.shape[1])]
                initials=value.get('initial_coordinates')
                if isinstance(initials,np.ndarray):
                    diagnostics['physical_initials']=np.column_stack((initials[:,0],np.exp(initials[:,1:]))).tolist()
            elif states.ndim==2 and states.shape[1]==3:
                core={name:dict(min=float(states[:,index].min()),max=float(states[:,index].max()))
                    for name,index in [('v',0),('p',1),('q',2)]}
                flow=value.get('flow')
                if isinstance(flow,np.ndarray):
                    core['flow']=dict(min=float(flow.min()),max=float(flow.max()))
                diagnostics['cores']=[core]
                if value.get('initial_state') is not None:
                    diagnostics['physical_initials']=np.asarray(value['initial_state']).tolist()
            cores=diagnostics.get('cores',[])
            diagnostics['flow_outside_engineering_reference']=any(
                core.get('flow',{}).get('min',1.)<.25 or core.get('flow',{}).get('max',1.)>4. for core in cores)
            initial=value.get('initial_coordinates')
            initial_values=np.exp(initial[:,1:]) if isinstance(initial,np.ndarray) else np.asarray(value.get('initial_state',[]))
            diagnostics['initial_outside_engineering_reference']=bool(np.any(initial_values<.5) or np.any(initial_values>2.))
            diagnostics['parameter_bound_hits']={name:bool(abs(float(value[name])-lo)<=1e-5*(hi-lo)
                or abs(float(value[name])-hi)<=1e-5*(hi-lo)) for name,lo,hi in [('tau',.6,4.),('kappa',.2,1.3)]
                if value.get(name) is not None}
            diagnostics['reference']='descriptive_engineering_flow_[.25,4]_initial_[.5,2];not_a_physiological_gate'
            result['state_diagnostics']=diagnostics
        return result
    if isinstance(value, (list,tuple)):
        return [feasibility_fit_metadata(item) for item in value]
    return serial(value)


def feasibility_candidate_start(candidate, label):
    return dict(label=label, coefficients=np.asarray(candidate['coefficients']).copy(),
        initial_coordinates=np.asarray(candidate['initial_coordinates']).copy(),
        tau=float(candidate['tau']), kappa=float(candidate['kappa']))


def feasibility_fit_plan(cfg, source, out, stage):
    out = Path(out)
    rows = [row for row in source['plan']['rows'] if row['split']=='evaluation']
    tasks = []
    if stage == 'vascular':
        for row,node in itertools.product(rows, cfg['vascular']['parameter_nodes']):
            task = dict(kind='vascular', row=row, node=node, arm='single_target_pair')
            task['id'] = token(dict(kind=stage, record_id=row['id'], node=node))
            tasks.append(task)
    elif stage == 'frontier':
        for row,arm in itertools.product(rows, FEASIBILITY_ARMS):
            task = dict(kind=stage, row=row, arm=arm)
            task['id'] = token(dict(kind=stage, record_id=row['id'], arm=arm))
            tasks.append(task)
    elif stage == 'profiles':
        for row,arm in itertools.product(rows, FEASIBILITY_ARMS):
            frontier_id = token(dict(kind='frontier', record_id=row['id'], arm=arm))
            path = out/'frontier'/(frontier_id+'.json')
            if not path.exists():
                raise ValueError('all frontier tasks must be terminal before the profile gate is evaluated')
            candidate = read_json(path)
            if not candidate.get('joint_witness_found'):
                continue
            for tau,kappa in itertools.product(cfg['profiles']['tau_grid'],cfg['profiles']['kappa_grid']):
                task = dict(kind=stage, row=row, arm=arm, frontier_id=frontier_id, tau=tau, kappa=kappa)
                task['id'] = token(dict(kind=stage, record_id=row['id'], arm=arm, tau=tau, kappa=kappa))
                tasks.append(task)
    else:
        raise ValueError('unknown feasibility fitting stage')
    manifest = dict(schema='shared_driver_feasibility_stage_manifest_v1', stage=stage,
        tasks=tasks, planned=len(tasks), gate='joint_8_substep_verified_three_modality_witness' if stage=='profiles' else None,
        execution='not_entered' if not tasks else 'planned',
        no_transfer='no_non_nominal_recovery_and_comparable_independent_record_contract')
    path = out/(stage+'_manifest.json')
    if path.exists() and read_json(path) != serial(manifest):
        raise ValueError('immutable feasibility stage manifest changed')
    if not path.exists():
        write_json(path,manifest)
    return tasks


def feasibility_fit_task(cfg, source, stage, task, out):
    record = feasibility_load(source, task['row'])
    sd = np.asarray(source['coordinate']['sd'],float)
    solver = cfg['solver']
    if stage == 'vascular':
        from src.inference.shared_driver_reconstruction import fit_vascular_flow_relaxation
        v = cfg['vascular']; target_pair = v['target_pair']
        mask = np.zeros((120,2),bool); mask[20:] = True
        starts = [dict(label=start['label'],coefficients=np.zeros(v['temporal_modes']),
            initial_state=start['initial_physical']) for start in v['starts']]
        fit = fit_vascular_flow_relaxation(record['hb'][:,target_pair],
            feasibility_parameters(source,task['node']),.25,
            temporal_basis=temporal_mode_basis(120,v['temporal_modes']),
            hb_operator=np.kron(record['hb_operator'] if record['hb_operator'].ndim==2 else record['hb_operator'][target_pair],np.eye(2)),
            sd=sd[60+2*target_pair:62+2*target_pair],
            score_mask=mask,starts=starts,max_nfev=solver['vascular_max_nfev'],
            substeps=solver['substeps'],validation_substeps=solver['validation_substeps'],
            log_flow_bounds=v['log_flow_bounds'],initial_log_bounds=v['initial_log_bounds'],
            integration_tolerance=solver['integration_tolerance'])
        good = bool(fit.get('legal_witness') and all(
            fit['by_chromophore'][name]['nrmse'] <= .5 for name in ('HbO','HbR')))
        arrays = {key:fit[key] for key in ('prediction','optimization_prediction','canonical_prediction',
            'states','initial_state','flow','log_flow','coefficients') if fit.get(key) is not None}
        arrays.update(target=record['hb'][:,target_pair],sd=sd[60+2*target_pair:62+2*target_pair])
        return dict(status='completed',success=True,legal_witness=fit.get('legal_witness',False),
            good_Hb_witness=good,fit=feasibility_fit_metadata(fit),
            interpretation='upstream_relaxed_single_target_pair_finite_engineering_search;residual_is_upper_bound_not_nonlinear_exclusion'),arrays
    from src.inference.source_observability import fit_feasibility_frontier
    kwargs = feasibility_forward(source,record,task['arm'])
    kwargs.pop('substeps')
    mask = np.zeros((120,68),bool); mask[20:] = True
    base_args = dict(**kwargs,sd=sd,score_mask=mask,
        tau_bounds=cfg['frontier']['tau_bounds'],kappa_bounds=cfg['frontier']['kappa_bounds'],
        max_nfev=solver['max_nfev'],substeps=solver['substeps'],
        verification_substeps=solver['validation_substeps'],validation_tolerance=solver['integration_tolerance'],
        constraint_tolerance=solver['constraint_tolerance'],ftol=solver['ftol'])
    bounds_id=token(dict(record_id=task['row']['id'],arm=task['arm'],kind='bounds'))
    bound_path=Path(out)/'bounds'/(bounds_id+'.json')
    if not bound_path.exists() or not read_json(bound_path).get('success'):
        raise ValueError('completed exact representation bounds required before nonlinear search')
    bound=read_json(bound_path)
    with np.load(bound_path.with_suffix('.npz'),allow_pickle=False) as values:
        coefficients=values['eeg_coefficients'].copy()
    k=coefficients.shape[1]
    fixed=feasibility_parameters(source)
    starts=[dict(label='EEG_projection_rest',coefficients=coefficients,
        initial_coordinates=np.zeros((k,5)),tau=fixed.free.tau,kappa=fixed.free.kappa)]
    baseline, baseline_arrays=feasibility_baseline(source,task['row'],task['arm'])
    if baseline_arrays and np.isfinite(baseline_arrays['coefficients']).all():
        starts.insert(0,dict(label='retained_full_fit',coefficients=baseline_arrays['coefficients'],
            initial_coordinates=baseline_arrays['initial_coordinates'],tau=fixed.free.tau,kappa=fixed.free.kappa))
    if stage=='profiles':
        frontier_path=Path(out)/'frontier'/(task['frontier_id']+'.json')
        parent=read_json(frontier_path)
        if not parent.get('joint_witness_found'):
            raise ValueError('a verified joint witness is required before profile array access')
        with np.load(frontier_path.with_suffix('.npz'),allow_pickle=False) as values:
            warm=dict(coefficients=values['joint_coefficients'].copy(),
                initial_coordinates=values['joint_initial_coordinates'].copy())
        warm.update(label='verified_joint_witness',tau=task['tau'],kappa=task['kappa'])
        fit=fit_feasibility_frontier(record['eeg'],record['hb'],
            feasibility_parameters(source,dict(tau=task['tau'],kappa=task['kappa'])),.25,
            epsilon=.5,starts=[warm]+[dict(start,tau=task['tau'],kappa=task['kappa']) for start in starts],
            free_physiology=False,**base_args)
        joint=fit.get('joint_witness')
        good=bool(joint is not None or (fit.get('witness_feasible') and all(
            fit['scores'][name]['nrmse']<=.5 for name in ('EEG','HbO','HbR'))))
        chosen=joint if joint is not None else fit
        arrays={key:chosen[key] for key in ('prediction','states','coefficients','initial_coordinates') if chosen.get(key) is not None}
        return dict(status='completed',success=True,admissible_grid_witness=good,
            fit=feasibility_fit_metadata(fit),
            interpretation='discovered_admissible_nodes_only_failed_search_does_not_narrow_parameter_set'),arrays
    nodes=[]; arrays={}; previous={}; joint_candidates=[]
    margin=cfg.get('representation',{}).get('exclusion_margin',1e-7)
    for epsilon in sorted(cfg['frontier']['epsilon_values']):
        for branch in cfg['frontier']['parameter_branches']:
            label=f'{branch}__eps{epsilon:g}'
            if bound['bounds']['EEG']['nrmse']>epsilon+margin:
                nodes.append(dict(epsilon=epsilon,branch=branch,status='excluded_by_certified_EEG_representation_bound',
                    EEG_lower_bound=bound['bounds']['EEG']['nrmse'],witness_feasible=False))
                continue
            local_starts=list(starts)
            if branch in previous:
                local_starts.append(feasibility_candidate_start(previous[branch],'narrower_epsilon_candidate'))
                if previous[branch].get('joint_witness') is not None:
                    local_starts.append(feasibility_candidate_start(previous[branch]['joint_witness'],'narrower_epsilon_joint_witness'))
            if branch=='free_tau_kappa' and 'fixed_current' in previous:
                local_starts.append(feasibility_candidate_start(previous['fixed_current'],'nested_fixed_candidate'))
                if previous['fixed_current'].get('joint_witness') is not None:
                    local_starts.append(feasibility_candidate_start(previous['fixed_current']['joint_witness'],'nested_fixed_joint_witness'))
            fit=fit_feasibility_frontier(record['eeg'],record['hb'],fixed,.25,
                epsilon=epsilon,starts=local_starts,free_physiology=branch=='free_tau_kappa',**base_args)
            node=dict(epsilon=epsilon,branch=branch,fit=feasibility_fit_metadata(fit),
                status=fit.get('status'),witness_feasible=fit.get('witness_feasible',False))
            if fit.get('physical_valid') and fit.get('coefficients') is not None:
                previous[branch]=fit
                if branch=='fixed':
                    previous['fixed_current']=fit
            nodes.append(node)
            for key in ('prediction','states','coefficients','initial_coordinates'):
                if fit.get(key) is not None:
                    arrays[label+'__'+key]=fit[key]
            joint=fit.get('joint_witness')
            if joint is None and fit.get('witness_feasible') and all(
                    fit['scores'][name]['nrmse']<=.5 for name in ('EEG','HbO','HbR')):
                joint=fit
            if joint is not None:
                joint_candidates.append((max(joint['scores'][name]['nrmse'] for name in ('EEG','HbO','HbR')),joint,label))
    best=min(joint_candidates,key=lambda item:item[0]) if joint_candidates else None
    if best:
        for key in ('prediction','states','coefficients','initial_coordinates'):
            if best[1].get(key) is not None:
                arrays['joint_'+key]=best[1][key]
    return dict(status='completed',success=True,nodes=nodes,joint_witness_found=best is not None,
        joint_witness=feasibility_fit_metadata(best[1]) if best else None,joint_witness_node=best[2] if best else None,
        Hb_only_diagnostic_gate='not_used_for_execution_or_exclusion',
        interpretation='best_found_feasible_envelope_and_retained_joint_witnesses_not_global_Pareto_or_impossibility'),arrays


def feasibility_execute(payload):
    cfg, out, stage, task = payload
    path = Path(out)/stage/(task['id']+'.json')
    if path.exists():
        return read_json(path)
    started = time.monotonic()
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            source = feasibility_source(cfg)
            if stage in ('synthetic','pilot'):
                value, arrays = feasibility_synthetic_task(cfg, source, task)
            elif stage == 'bounds':
                value, arrays = feasibility_bounds_task(cfg, source, task)
            else:
                value, arrays = feasibility_fit_task(cfg, source, stage, task, out)
        if arrays:
            write_npz(path.with_suffix('.npz'), **arrays)
        result = dict(**task, **value)
    except Exception as exc:
        result = dict(**task, success=False, status='failed_exception', error=str(exc), traceback=traceback.format_exc())
    result['elapsed_s'] = time.monotonic()-started
    write_json(path, result)
    return result


def feasibility_run(cfg, out, stage, tasks, workers):
    if workers < 1:
        raise ValueError('workers must be positive')
    pending_tasks = [task for task in tasks if not (Path(out)/stage/(task['id']+'.json')).exists()]
    started = time.monotonic(); terminal = len(tasks)-len(pending_tasks)
    if pending_tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending, iterator = set(), iter(pending_tasks)
            while True:
                while len(pending) < workers*cfg.get('resources', {}).get('max_in_flight_per_worker', 2):
                    try:
                        pending.add(pool.submit(feasibility_execute, (cfg, str(out), stage, next(iterator))))
                    except StopIteration:
                        break
                if not pending:
                    break
                ready, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    future.result(); terminal += 1
                    progress = dict(stage=stage, planned=len(tasks), terminal=terminal,
                        workers=workers, elapsed_s=time.monotonic()-started)
                    write_json(Path(out)/(stage+'_progress.json'), progress)
                    print(json.dumps(progress), flush=True)
    result = dict(stage=stage, planned=len(tasks), terminal=terminal,
        execution=('not_entered' if not tasks else 'completed' if terminal == len(tasks) else 'incomplete'), elapsed_s=time.monotonic()-started)
    write_json(Path(out)/(stage+'_stage.json'), result)
    return result


def feasibility_aggregate(records, accessor, *, repeats=1000, seed=20261022):
    subjects = {}
    for record in records:
        value = accessor(record)
        if value is not None and np.isfinite(value):
            row = record['row']
            subjects.setdefault(row['subject'], {}).setdefault(row['record'], []).append(float(value))
    unit = {subject: float(np.mean([np.mean(values) for values in sessions.values()]))
        for subject, sessions in subjects.items()}
    values = np.asarray(list(unit.values()), float)
    if len(values):
        rng = np.random.default_rng(seed)
        draws = np.sqrt(values[rng.integers(len(values), size=(repeats, len(values)))].mean(axis=1))
        center, interval = float(np.sqrt(values.mean())), np.quantile(draws, [.025, .975]).tolist()
    else:
        center, interval = None, None
    return dict(nrmse=center, descriptive_subject_bootstrap_95=interval,
        subjects=len(unit), windows=sum(sum(len(v) for v in sessions.values()) for sessions in subjects.values()),
        subject_nmse=unit, pooling='mean_time_feature_MSE_then_window_mean_within_record_then_record_mean_within_subject_then_subject_equal_MSE_and_sqrt')


def feasibility_mean(records, accessor, *, repeats=1000, seed=20261022):
    subjects={}
    for record in records:
        value=accessor(record)
        if value is not None and np.isfinite(value):
            row=record['row']
            subjects.setdefault(row['subject'],{}).setdefault(row['record'],[]).append(float(value))
    units={subject:float(np.mean([np.mean(values) for values in sessions.values()]))
        for subject,sessions in subjects.items()}
    values=np.asarray(list(units.values()),float)
    if len(values):
        rng=np.random.default_rng(seed)
        draws=values[rng.integers(len(values),size=(repeats,len(values)))].mean(axis=1)
        mean,interval=float(values.mean()),np.quantile(draws,[.025,.975]).tolist()
    else:mean,interval=None,None
    return dict(mean=mean,descriptive_subject_bootstrap_95=interval,subjects=len(units),
        windows=sum(sum(len(v) for v in sessions.values()) for sessions in subjects.values()),
        subject_means=units,pooling='window_then_record_then_subject_equal_arithmetic_mean')


def feasibility_state_summary(fits):
    diagnostics=[fit['state_diagnostics'] for fit in fits if fit.get('state_diagnostics')]
    cores=[core for item in diagnostics for core in item.get('cores',[])]
    ranges={name:dict(min=min(core[name]['min'] for core in cores if name in core),
        max=max(core[name]['max'] for core in cores if name in core)) for name in ('flow','v','p','q')
        if any(name in core for core in cores)}
    return dict(candidates=len(diagnostics),all_candidate_extrema=ranges,
        flow_outside_engineering_reference=sum(bool(item.get('flow_outside_engineering_reference')) for item in diagnostics),
        initial_outside_engineering_reference=sum(bool(item.get('initial_outside_engineering_reference')) for item in diagnostics),
        parameter_bound_hits={name:sum(bool(item.get('parameter_bound_hits',{}).get(name)) for item in diagnostics) for name in ('tau','kappa')},
        interpretation='descriptive_engineering_state_and_parameter_extrema_not_physiological_qualification')


def feasibility_summary(cfg, out):
    out = Path(out); manifest = read_json(out/'manifest.json')
    records = [read_json(out/'bounds'/(task['id']+'.json')) for task in manifest['tasks']
        if (out/'bounds'/(task['id']+'.json')).exists()]
    groups, csv_rows = {}, []
    for arm in FEASIBILITY_ARMS:
        part = [r for r in records if r['arm'] == arm]
        good = [r for r in part if r.get('success')]
        group = dict(planned=sum(t['arm'] == arm for t in manifest['tasks']), terminal=len(part),
            succeeded=len(good), failed=len(part)-len(good), subjects=manifest['evaluation_subjects'], metrics={})
        for modality in ('EEG', 'HbO', 'HbR'):
            group['metrics'][modality] = feasibility_aggregate(good,
                lambda r, m=modality: r['bounds'][m]['nmse'])
        group['comparators']={}
        for comparator in ('baseline','zero_prediction'):
            comparable=[r for r in good if comparator=='zero_prediction' or r['baseline'].get('metrics')]
            group['comparators'][comparator]=dict(common_available_windows=len(comparable),
                excluded_or_missing=len(good)-len(comparable),
                interpretation='retained_original_selected_weight_full_fit' if comparator=='baseline' else 'zero_prediction_in_parent_coordinate',
                metrics={m:feasibility_aggregate(comparable,lambda r,m=m,c=comparator:
                    r[c]['metrics'][m]['nmse'] if c=='baseline' else r[c][m]['nmse']) for m in ('EEG','HbO','HbR')},
                bound_same_subset={m:feasibility_aggregate(comparable,lambda r,m=m:r['bounds'][m]['nmse']) for m in ('EEG','HbO','HbR')},
                paired_nmse_comparator_minus_bound={m:feasibility_mean(comparable,lambda r,m=m,c=comparator:
                    (r[c]['metrics'][m]['nmse'] if c=='baseline' else r[c][m]['nmse'])-r['bounds'][m]['nmse'])
                    for m in ('EEG','HbO','HbR')})
        group['morphology']={m:{field:feasibility_mean(good,lambda r,m=m,field=field:
            r['numerical_projection'][m].get(field))
            for field in ('centered_nrmse','centered_local_nrmse','centered_amplitude_ratio','amplitude_rms_ratio','mean_feature_correlation')}
            for m in ('EEG','HbO','HbR')}
        group['coverage'] = {}
        for tolerance in cfg.get('scoring', {}).get('tolerances', [.25, .5, .75, 1., 1.5]):
            key = str(tolerance)
            group['coverage'][key] = dict(
                passed_all_necessary=sum(r['tolerance_coverage'][key]['simultaneous_necessary_condition'] for r in good),
                failed_at_least_one_necessary=sum(not r['tolerance_coverage'][key]['simultaneous_necessary_condition'] for r in good),
                unavailable=len(part)-len(good), planned=group['planned'],
                by_modality={m:sum(r['tolerance_coverage'][key][m] for r in good) for m in ('EEG','HbO','HbR')})
        groups[arm] = group
        for row in good:
            for modality in ('EEG','HbO','HbR'):
                csv_rows.append(dict(id=row['id'], arm=arm, record_id=row['row']['id'], subject=row['row']['subject'],
                    modality=modality, **row['bounds'][modality], baseline=row['baseline'].get('metrics'),
                    zero=row['zero_prediction'][modality]['nrmse']))
    stages = {stage:read_json(out/(stage+'_stage.json')) for stage in ('pilot','synthetic','bounds','vascular','frontier','profiles')
        if (out/(stage+'_stage.json')).exists()}
    stage_records={}
    stage_manifests={}
    for stage in ('pilot','synthetic','vascular','frontier','profiles'):
        path=out/(stage+'_manifest.json')
        if path.exists():
            stage_manifests[stage]=read_json(path)
            stage_records[stage]=[read_json(out/stage/(task['id']+'.json')) for task in stage_manifests[stage]['tasks']
                if (out/stage/(task['id']+'.json')).exists()]
    vascular_groups={}
    for node in cfg['vascular']['parameter_nodes']:
        part=[r for r in stage_records.get('vascular',[]) if r['node']['label']==node['label']]
        finite=[r for r in part if r.get('success') and r.get('legal_witness') and r.get('fit',{}).get('by_chromophore')]
        vascular_groups[node['label']]=dict(node=node,planned=manifest['evaluation_windows'],terminal=len(part),
            task_exceptions=sum(not r.get('success') for r in part),
            legal_witnesses=sum(bool(r.get('legal_witness')) for r in part),
            good_Hb_witnesses=sum(bool(r.get('good_Hb_witness')) for r in part),
            optimizer_converged=sum(bool(r.get('fit',{}).get('optimizer_converged')) for r in part),
            failure_counts=dict(
                exceptions=sum(not r.get('success') for r in part),
                illegal_or_unverified=sum(r.get('success') and not r.get('legal_witness') for r in part),
                integration=sum(r.get('fit',{}).get('status')=='failed_integration_validation' for r in part),
                optimizer_nonconverged=sum(r.get('success') and not r.get('fit',{}).get('optimizer_converged') for r in part),
                evaluation_budget=sum(any(start.get('status')=='evaluation_budget' for start in r.get('fit',{}).get('starts',[])) for r in part)),
            metrics={m:feasibility_aggregate(finite,lambda r,m=m:r['fit']['by_chromophore'][m]['nrmse']**2)
                for m in ('HbO','HbR')},
            state_diagnostics=feasibility_state_summary([r['fit'] for r in finite]),
            interpretation='joint_qualified_preferred_selected_candidate_residual_is_upper_bound;minimum_legal_SSE_is_separate_retained_candidate_diagnostic')
    frontier_groups={}
    frontier_records=stage_records.get('frontier',[])
    for arm,branch,epsilon in itertools.product(FEASIBILITY_ARMS,cfg['frontier']['parameter_branches'],cfg['frontier']['epsilon_values']):
        arm_records=[r for r in frontier_records if r['arm']==arm]
        nodes=[dict(node,row=r['row']) for r in arm_records for node in r.get('nodes',[])
            if node['branch']==branch and node['epsilon']==epsilon]
        fits=[dict(node['fit'],row=node['row']) for node in nodes if node.get('fit') and node['fit'].get('scores')]
        feasible=[r for r in fits if r.get('witness_feasible')]
        frontier_groups[f'{arm}|{branch}|epsilon={epsilon:g}']=dict(arm=arm,branch=branch,epsilon=epsilon,
            planned=manifest['evaluation_windows'],terminal=len(nodes),
            task_exceptions=sum(not r.get('success') for r in arm_records),
            certified_EEG_exclusions=sum(node['status']=='excluded_by_certified_EEG_representation_bound' for node in nodes),
            legal_EEG_feasible_witnesses=len(feasible),
            numerical_or_search_undetermined=sum(node.get('fit') is not None and not node.get('witness_feasible') for node in nodes),
            failure_counts=dict(
                exceptions=sum(not r.get('success') for r in arm_records),
                physical_or_domain=sum(bool(node.get('fit')) and not node['fit'].get('physical_valid') for node in nodes),
                integration=sum(bool(node.get('fit')) and node['fit'].get('integration_error') is not None
                    and node['fit']['integration_error']>cfg['solver']['integration_tolerance'] for node in nodes),
                evaluation_budget=sum(any('budget' in str(start.get('message',start.get('optimizer_message',''))).lower()
                    for start in node.get('fit',{}).get('starts',[])) for node in nodes)),
            metrics={m:feasibility_aggregate(feasible,lambda r,m=m:r['scores'][m]['nmse']) for m in ('EEG','Hb','HbO','HbR')},
            state_diagnostics=feasibility_state_summary(feasible),
            summary_scope='verified_EEG_feasible_subset_with_complete_planned_denominator')
    paired_frontier={}
    for arm,epsilon in itertools.product(FEASIBILITY_ARMS,cfg['frontier']['epsilon_values']):
        lookups={branch:{r['row']['id']:dict(node['fit'],row=r['row'])
            for r in frontier_records if r['arm']==arm for node in r.get('nodes',[])
            if node['branch']==branch and node['epsilon']==epsilon and node.get('witness_feasible')
            and node.get('fit') and node['fit'].get('scores')}
            for branch in ('fixed','free_tau_kappa')}
        common=sorted(set(lookups['fixed'])&set(lookups['free_tau_kappa']))
        pairs=[dict(row=lookups['fixed'][identity]['row'],fixed=lookups['fixed'][identity],
            free=lookups['free_tau_kappa'][identity]) for identity in common]
        paired_frontier[f'{arm}|epsilon={epsilon:g}']=dict(arm=arm,epsilon=epsilon,
            planned=manifest['evaluation_windows'],paired_windows=len(pairs),
            paired_subjects=len({r['row']['subject'] for r in pairs}),
            fixed_feasible_windows=len(lookups['fixed']),free_feasible_windows=len(lookups['free_tau_kappa']),
            missing_or_not_jointly_EEG_feasible=manifest['evaluation_windows']-len(pairs),
            nesting_violations=sum(r['free']['scores']['Hb']['nmse']>r['fixed']['scores']['Hb']['nmse']+1e-8 for r in pairs),
            metrics={modality:dict(
                fixed=feasibility_aggregate(pairs,lambda r,m=modality:r['fixed']['scores'][m]['nmse']),
                free=feasibility_aggregate(pairs,lambda r,m=modality:r['free']['scores'][m]['nmse']),
                paired_nmse_fixed_minus_free=feasibility_mean(pairs,lambda r,m=modality:
                    r['fixed']['scores'][m]['nmse']-r['free']['scores'][m]['nmse'])) for modality in ('Hb','HbO','HbR')},
            delta_direction='positive_fixed_minus_free_NMSE_favors_free_parameters',
            interpretation='same_records_and_epsilon_verified_feasible_pairing_only')
    profiles=stage_records.get('profiles',[])
    profile_summary=dict(execution=stages.get('profiles',{}).get('execution','not_entered'),
        joint_witness_windows_arms=sum(bool(r.get('joint_witness_found')) for r in frontier_records),
        planned=stage_manifests.get('profiles',{}).get('planned',0),terminal=len(profiles),
        admissible_grid_nodes=sum(bool(r.get('admissible_grid_witness')) for r in profiles),
        grid_nodes=[dict(id=r['id'],subject=r['row']['subject'],record_id=r['row']['id'],arm=r['arm'],
            tau=r['tau'],kappa=r['kappa'],admissible=r.get('admissible_grid_witness',False),status=r['status']) for r in profiles],
        transfer='not_entered_missing_non_nominal_recovery_and_comparable_independent_record_contract',
        interpretation='admissible_grid_witnesses_not_confidence_intervals;failed_nodes_do_not_prove_exclusion')
    generated_summaries={}
    for stage in ('pilot','synthetic'):
        generated=stage_records.get(stage,[])
        scenarios={}
        for scenario in cfg['synthetic']['scenarios']:
            selected=[r for r in generated if r['scenario']==scenario]
            good=[r for r in selected if r.get('success')]
            scenarios[scenario]=dict(planned=sum(task['scenario']==scenario for task in stage_manifests.get(stage,{}).get('tasks',[])),
                terminal=len(selected),task_exceptions=len(selected)-len(good),
                bounds_median_nrmse={m:float(np.median([r['bounds'][m]['nrmse'] for r in good])) if good else None
                    for m in ('EEG','HbO','HbR')},
                vascular_legal_witnesses=sum(bool((r.get('vascular') or {}).get('legal_witness')) for r in good),
                vascular_good_Hb_witnesses=sum(bool((r.get('vascular') or {}).get('joint_hb_witness')) for r in good),
                frontier={branch:dict(planned=sum(task['scenario']==scenario for task in stage_manifests.get(stage,{}).get('tasks',[])),
                    executed=sum(branch in r.get('frontier',{}) for r in good),
                    verified_EEG_witnesses=sum(bool(r.get('frontier',{}).get(branch,{}).get('witness_feasible')) for r in good),
                    joint_witnesses=sum(r.get('frontier',{}).get(branch,{}).get('joint_witness') is not None for r in good),
                    driver_nrmse_median=float(np.median([r['recovery'][branch]['driver_nrmse'] for r in good
                        if branch in r.get('recovery',{})])) if any(branch in r.get('recovery',{}) for r in good) else None)
                    for branch in ('fixed','free_tau_kappa')})
        generated_summaries[stage]=dict(planned=stage_manifests.get(stage,{}).get('planned',0),terminal=len(generated),
            failed=sum(not r.get('success') for r in generated),scenarios=scenarios,
            task_elapsed_seconds=[r.get('elapsed_s') for r in generated],
            measured_arrays_read=0,truth_as_start=False,
            interpretation='generator_seed_replicates_not_independent_measured_subjects;nominal_truth_not_parameter_recovery_qualification')
    required=('synthetic','bounds','vascular','frontier','profiles')
    complete=(len(records)==manifest['planned'] and all(stages.get(stage,{}).get('execution') in ('completed','not_entered') for stage in required))
    task_counts=dict(bounds=dict(planned=manifest['planned'],terminal=len(records)))
    task_counts.update({stage:dict(planned=stage_manifests.get(stage,{}).get('planned',0),terminal=len(stage_records.get(stage,[])))
        for stage in ('synthetic','vascular','frontier','profiles')})
    result = dict(schema='shared_driver_feasibility_summary_v1', experiment_id=cfg['experiment_id'],
        execution='completed' if complete else 'incomplete',
        planned=sum(value['planned'] for value in task_counts.values()),
        terminal=sum(value['terminal'] for value in task_counts.values()),task_counts=task_counts,
        bounds_planned=manifest['planned'],bounds_terminal=len(records),groups=groups, stages=stages,
        evaluation_subjects=manifest['evaluation_subjects'], evaluation_windows=manifest['evaluation_windows'],
        vascular=vascular_groups,frontier=frontier_groups,paired_frontier=paired_frontier,profiles=profile_summary,
        frontier_subset_limit='different_epsilon_feasible_subsets_may_differ;unpaired_pooled_means_do_not_establish_Pareto_improvement',
        synthetic=generated_summaries['synthetic'],pilot=generated_summaries['pilot'],
        source_run=SPATIAL, raw_data_read=False, protected_data_read=False,
        primary_tolerance=.5, score_steps=[20,120],
        conclusion='representation_bounds_are_necessary_conditions_only;_local_fit_residuals_are_upper_bounds;no_global_vascular_impossibility_claim',
        scientific_verdict='conditional_model_feasibility_diagnostic_not_physiological_recovery')
    write_csv(out/'representation_bounds.csv', csv_rows)
    write_json(out/'summary.json', result)
    return result


def feasibility_main(args, parser):
    cfg = feasibility_config(args.config)
    if args.stage == 'check':
        print(json.dumps(serial(feasibility_synthetic(cfg))), flush=True)
        return
    if args.output is None:
        parser.error('--output is required outside check')
    if args.stage not in FEASIBILITY_STAGES:
        parser.error('stage is not part of the feasibility diagnostic')
    source = feasibility_source(cfg)
    # Validate output before creating its lock or any directory outside the owner.
    boundary = (ROOT/ARTIFACT_ROOT).resolve()
    if args.output.resolve().parent != boundary:
        parser.error('output must be one versioned child of the source-observability artifact root')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        formal = args.stage in ('bounds','vascular','frontier','profiles','pilot','synthetic')
        if formal and not any(os.environ.get(key) for key in ('INVOCATION_ID','SLURM_JOB_ID','PBS_JOBID')):
            parser.error('substantial stages require a durable systemd or scheduler supervisor')
        if not 1<=args.workers<=cfg['resources']['max_workers']:
            parser.error('workers must lie in the bounded configured resource range')
        plan = feasibility_prepare(cfg, args.output, source, freeze=formal)
        execution=dict(stage=args.stage,command=[sys.executable,str(Path(__file__)),*sys.argv[1:]],
            working_directory=str(Path.cwd()),started_epoch_s=time.time(),status='running',workers=args.workers,
            library_threads=1,cpu_affinity=sorted(os.sched_getaffinity(0)),
            supervisor={key:os.environ[key] for key in ('INVOCATION_ID','SYSTEMD_UNIT','SLURM_JOB_ID','PBS_JOBID') if key in os.environ})
        execution_path=args.output/'executions'/f'{args.stage}_{time.time_ns()}.json'
        write_json(execution_path,execution)
        try:
            result=feasibility_dispatch(cfg,args,source,plan)
            execution.update(status='completed',finished_epoch_s=time.time())
        except BaseException as exc:
            execution.update(status='failed',finished_epoch_s=time.time(),error=str(exc),traceback=traceback.format_exc())
            raise
        finally:
            write_json(execution_path,execution)
            write_json(args.output/'execution.json',execution)
        print(json.dumps(serial(result)), flush=True)


def feasibility_dispatch(cfg,args,source,plan):
    if args.stage=='plan':
        return {k:v for k,v in plan.items() if k not in ('rows','tasks')}
    if args.stage=='summary':
        return feasibility_summary(cfg,args.output)
    if args.stage in ('synthetic','pilot'):
        tasks=feasibility_synthetic_plan(cfg,pilot=args.stage=='pilot' or args.pilot_plan)
        path=args.output/(args.stage+'_manifest.json')
        stage_manifest=dict(stage=args.stage,tasks=tasks,planned=len(tasks),measured_arrays_read=0,truth_as_start=False)
        if path.exists() and read_json(path)!=serial(stage_manifest):
            raise ValueError('immutable synthetic stage manifest changed')
        if not path.exists():write_json(path,stage_manifest)
    else:
        tasks=plan['tasks'] if args.stage=='bounds' else feasibility_fit_plan(cfg,source,args.output,args.stage)
    result=feasibility_run(cfg,args.output,args.stage,tasks,args.workers)
    feasibility_summary(cfg,args.output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--stage', choices=tuple(dict.fromkeys(('check', 'plan', 'pilot', 'run', 'summary')+FEASIBILITY_STAGES)), default='check')
    parser.add_argument('--feasibility', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--pilot-plan', action='store_true')
    args = parser.parse_args(argv)
    if args.feasibility:
        if args.config == DEFAULT_CONFIG:
            args.config = FEASIBILITY_CONFIG
        return feasibility_main(args, parser)
    if args.stage not in ('check','plan','pilot','run','summary'):
        parser.error('this stage requires --feasibility')
    cfg = read_config(args.config)
    if args.stage == 'check':
        print(json.dumps(check(cfg)))
        return
    if args.output is None:
        parser.error('--output is required outside check')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'summary':
            result = summarize(cfg, args.output)
        else:
            manifest = prepare(cfg, args.output, pilot=args.stage == 'pilot' or args.pilot_plan)
            if args.stage == 'plan':
                result = {k: v for k, v in manifest.items() if k not in ('tasks', 'source_identity')}
            else:
                result = run_tasks(cfg, args.output, manifest, args.workers)
        print(json.dumps(serial(result)), flush=True)


if __name__ == '__main__':
    main()
