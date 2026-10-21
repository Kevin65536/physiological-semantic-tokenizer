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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--stage', choices=('check', 'plan', 'pilot', 'run', 'summary'), default='check')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--pilot-plan', action='store_true')
    args = parser.parse_args(argv)
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
