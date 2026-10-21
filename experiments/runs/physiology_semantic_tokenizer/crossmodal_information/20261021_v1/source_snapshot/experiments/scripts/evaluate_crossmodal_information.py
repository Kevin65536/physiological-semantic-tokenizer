#!/usr/bin/env python3
"""Frozen-readout EEG pairing destruction and training-background injection.

Only the specifically named completed public spatial run may supply arrays.
There is no raw-data loader, archive discovery or protected-data reader here.
Substantial pilot/run stages must be launched by the existing durable supervisor.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import yaml
from src.inference.crossmodal_information import (
    fit_condition, forward_kwargs, local_added_constraint, paired_statistics,
    physical_parameters, predict_ridge, scaled_record, score_target, validate_wrong_partners,
)
from src.inference.shared_driver_spatial import forward_spatial_shared_driver

DEFAULT_CONFIG = ROOT/'experiments/configs/physiology_semantic_tokenizer/crossmodal_information_v1.yaml'
ARTIFACT_ROOT = 'experiments/runs/physiology_semantic_tokenizer/crossmodal_information'
SOURCE = 'experiments/runs/physiology_semantic_tokenizer/shared_driver_spatial/20261020_measured_v2'
SCHEMA = 'crossmodal_information_v1'


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serial(x) for x in value]
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


def read_json(path):
    return json.loads(Path(path).read_text())


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem+f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def token(identity):
    return hashlib.sha256(str(identity).encode()).hexdigest()[:24]


def read_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg.get('schema') != SCHEMA or cfg.get('experiment_id') != 'CROSSMODAL-INFORMATION-v1'
            or cfg.get('artifact_root') != ARTIFACT_ROOT or cfg.get('source_run') != SOURCE
            or cfg.get('protected_data') != 'forbidden'
            or cfg.get('scope') != 'retained_public_development_arrays_only'):
        raise ValueError('wrong versioned suite or retained public read boundary')
    if (cfg['model']['arms'] != ['K1_true_H4_E12', 'K2_true_H4_E12']
            or cfg['model']['fixed_eeg_weight'] != 1. or cfg['model']['hidden_target_pair'] != 0
            or cfg['perturbations']['eeg_support'] != [32, 88]
            or cfg['perturbations']['conditions'] != ['Hb_only', 'paired', 'wrong_EEG', 'shift_m4s', 'shift_p4s', 'shift_m8s', 'shift_p8s']
            or cfg['perturbations']['shift_steps'] != dict(shift_m4s=-16, shift_p4s=16, shift_m8s=-32, shift_p8s=32)
            or cfg['perturbations']['no_wrong_fallback'] is not True
            or cfg['evaluation']['score_steps'] != [20, 120]
            or cfg['evaluation']['failure_penalty_nmse'] != 100.):
        raise ValueError('fixed information, common support or endpoint changed')
    if (cfg['synthetic']['arm'] != 'K2_true_H4_E12'
            or cfg['synthetic']['doses_training_target_sd_rms'] != [0., .25, .5, 1.]
            or cfg['synthetic']['conditions'] != ['Hb_only', 'paired', 'wrong_EEG']
            or cfg['synthetic']['calibration_seeds'] != 16 or cfg['synthetic']['evaluation_seeds'] != 16
            or cfg['resources']['numerical_threads'] != 1
            or not 1 <= cfg['resources']['max_workers'] <= 24):
        raise ValueError('fixed injection denominator or resource bounds changed')
    return cfg


def retained_source(cfg, root=ROOT):
    root = Path(root).resolve()
    source = (root/cfg['source_run']).resolve()
    if source != root/SOURCE:
        raise ValueError('retained run must be exactly the bounded public source')
    identity = read_json(source/'run_identity.json')
    summary = read_json(source/'summary.json')
    parent = read_json(source/'resolved_config.json')
    plan = read_json(source/'plan.json')
    coordinate = read_json(source/'coordinate.json')
    if (identity.get('schema') != 'shared_driver_spatial_v1' or identity.get('family') != 'public'
            or summary.get('execution') != 'completed' or summary['planned'] != summary['terminal']
            or parent.get('data', {}).get('protected_data') != 'forbidden'
            or parent['data']['datasets'] != ['eeg_fnirs_single_trial']
            or plan.get('status') != 'available'
            or parent['tensor']['steps'] != 120 or parent['tensor']['dt_s'] != .25):
        raise ValueError('completed public spatial source and fixed120x0.25s contract required')
    groups = {}
    for row in plan['rows']:
        if row['dataset'] != 'eeg_fnirs_single_trial' or row['split'] not in ('train', 'selection', 'evaluation'):
            raise ValueError('unexpected dataset or split in retained public plan')
        if row['subject'] in groups and groups[row['subject']] != row['split']:
            raise ValueError('parent subject split leakage')
        groups[row['subject']] = row['split']
    validate_wrong_partners(plan['rows'], plan['wrong_eeg_partners'])
    training_ids = {row['id'] for row in plan['rows'] if row['split'] == 'train'}
    if set(coordinate['training_ids']) != training_ids:
        raise ValueError('frozen coordinate does not cover exactly parent training rows')
    calibrations = {}
    for arm in cfg['model']['arms']:
        calibration = read_json(source/'calibration'/(arm+'.json'))
        if (calibration.get('status') != 'completed' or calibration.get('eeg_readout_status') != 'learned'
                or not set(calibration['readout']['training_ids']) <= training_ids
                or calibration.get('evaluation_data_used') is not False):
            raise ValueError('train-only frozen EEG readout required')
        for name in ('spatial_ridge__Hb_only', 'joint_ridge__paired'):
            if set(calibration['baselines'][name]['training_ids']) != training_ids:
                raise ValueError('baseline training membership changed')
        calibrations[arm] = calibration
    return dict(path=source, identity=identity, parent=parent, plan=plan,
                coordinate=coordinate, calibrations=calibrations)


def load_retained(source, row, *, fields=('eeg', 'hb', 'eeg_operator', 'hb_operator')):
    path = Path(source)/'prepared'/(token(row['id'])+'.npz')
    metadata = read_json(path.with_suffix('.json'))
    if metadata.get('status') != 'completed' or metadata.get('row') != row:
        raise ValueError('retained prepared identity not completed or mismatched')
    with np.load(path, allow_pickle=False) as z:
        # input_hb is not an authoritative mask; the suite masks before assembly.
        return {k: z[k].copy() for k in fields}


def manifest(cfg, source, pilot=False):
    rows = [row for row in source['plan']['rows'] if row['split'] == 'evaluation']
    if pilot:
        rows = rows[:2]
    cells = []
    for row in rows:
        for arm in cfg['model']['arms']:
            for condition in cfg['perturbations']['conditions']:
                for method in ('SSM_fixed1', 'raw_ridge'):
                    spec = dict(family='measured', record_id=row['id'], unit=row['subject'],
                        arm=arm, condition=condition, method=method,
                        wrong_eeg_id=source['plan']['wrong_eeg_partners'].get(row['id']))
                    spec['id'] = token(json.dumps(spec, sort_keys=True))
                    cells.append(spec)
    doses = [0., 1.] if pilot else cfg['synthetic']['doses_training_target_sd_rms']
    for partition in ('calibration', 'evaluation'):
        repeats = 1 if pilot else cfg['synthetic'][partition+'_seeds']
        for index in range(repeats):
            for dose in doses:
                for condition in cfg['synthetic']['conditions']:
                    for method in ('SSM_fixed1', 'raw_ridge'):
                        spec = dict(family='synthetic', unit=f'{partition}_{index:03}', partition=partition,
                            seed_index=index, dose=dose, arm=cfg['synthetic']['arm'], condition=condition, method=method)
                        spec['id'] = token(json.dumps(spec, sort_keys=True))
                        cells.append(spec)
    return dict(schema=SCHEMA, pilot=pilot, cells=cells, planned_cells=len(cells),
        source_run=cfg['source_run'], source_sha=source['identity']['source_sha'],
        parent_planned_evaluation_windows=sum(row['split'] == 'evaluation' for row in source['plan']['rows']),
        parent_training_windows=sum(row['split'] == 'train' for row in source['plan']['rows']),
        measured_evaluation_windows=len(rows), measured_independent_subjects=len({row['subject'] for row in rows}),
        paired_wrong_available=sum(source['plan']['wrong_eeg_partners'].get(row['id']) is not None for row in rows),
        method_input='all_donor_Hb_0:120; EEG32:88_for_all_positive_weight_conditions; hidden_pair_never_input',
        scoring='hidden_pair_HbO_HbR20:120; frozen_parent_training_SD',
        selected_parent_weights={arm: c['selected_eeg_weight'] for arm, c in source['calibrations'].items()},
        synthetic_calibration_units=1 if pilot else cfg['synthetic']['calibration_seeds'],
        synthetic_evaluation_units=1 if pilot else cfg['synthetic']['evaluation_seeds'])


def injected_pair(cfg, source, partition, index):
    """One target and an independent EEG-only companion; no eval background."""
    seed = cfg['seed']+cfg['synthetic']['seed_offset']+index+(0 if partition == 'calibration' else 1000000)
    rng = np.random.default_rng(seed)
    training = [r for r in source['plan']['rows'] if r['split'] == 'train']
    count = cfg['synthetic']['background_windows_per_modality']
    selected = rng.choice(len(training), size=3*count, replace=False)
    records, provenance = [], []
    for modality, indices in zip(('Hb_target', 'EEG_target', 'EEG_wrong'), selected.reshape(3, count)):
        weights = rng.normal(size=count)
        weights /= np.linalg.norm(weights)
        chunks = [scaled_record(load_retained(source['path'], training[i]), source['coordinate'], hide_target=False) for i in indices]
        array = np.asarray([x['hb'] if modality == 'Hb_target' else x['eeg'] for x in chunks])
        mixed = np.einsum('i,it...->t...', weights, array)
        records.append(mixed)
        flat = mixed.reshape(len(mixed), -1)
        provenance.append(dict(modality=modality, ids=[training[i]['id'] for i in indices], weights=weights,
            source_partition='train', source_windows_available=len(training),
            source_mean=np.mean(array, axis=(0, 1)), mixture_mean=flat.mean(axis=0),
            source_channel_sd=np.std(array.reshape(-1, flat.shape[1]), axis=0),
            mixture_channel_sd=flat.std(axis=0), mixture_covariance=np.cov(flat, rowvar=False)))
    reference_record = scaled_record(load_retained(source['path'], training[selected[0]]), source['coordinate'])
    calibration = source['calibrations'][cfg['synthetic']['arm']]
    kwargs = forward_kwargs(source['parent'], reference_record, calibration)
    basis = kwargs['temporal_basis']
    signals = []
    for _ in range(2):
        coefficient = rng.normal(size=(basis.shape[1], 2))/np.arange(1, basis.shape[1]+1)[:, None]**1.5
        coefficient *= .025/np.sqrt(np.mean((basis@coefficient)**2, axis=0))
        signal = forward_spatial_shared_driver(coefficient, np.zeros((2, 5)),
            physical_parameters(source['parent']), .25, **kwargs, derivative=False,
            substeps=source['parent']['solver']['validation_substeps'])
        sd = np.asarray(source['coordinate']['sd'])[60:62]
        norm = float(np.sqrt(np.mean((signal['hb_prediction'][20:, 0]/sd)**2)))
        if not norm > 0:
            raise ValueError('nonzero generated Hb response required')
        signals.append(dict(eeg=signal['eeg_prediction']/norm, hb=signal['hb_prediction']/norm,
                            coefficients=coefficient, drivers=signal['drivers'], states=signal['states'], normalization=norm))
    target = dict(eeg=records[1], hb=records[0], eeg_operator=reference_record['eeg_operator'],
                  hb_operator=reference_record['hb_operator'])
    wrong = dict(eeg=records[2])
    return dict(target=target, wrong=wrong, target_signal=signals[0], wrong_signal=signals[1],
        provenance=dict(seed=seed, partition=partition, unit=index, background_draws=provenance,
            crossmodal_draw_ids_disjoint=True, signal_generator='fixed_parent_K2_nonlinear_forward_then_common_observation_scalar',
            dose_normalization='sqrt(mean((clean_Hb_target20:120/training_SD_HbO_HbR)^2))',
            fixed_across_dose=['all_background_arrays', 'source_rows_and_weights', 'clean_response_shapes', 'frozen_scales_and_readouts'],
            independent_of_evaluation_prepared=True))


def synthetic_record(cfg, source, spec, out):
    unit_path = Path(out)/'synthetic_inputs'/(spec['unit']+'.npz')
    meta_path = unit_path.with_suffix('.json')
    # All workers recreate the same small deterministic arrays; only write-once
    # provenance is retained. Atomic replace is safe for identical content.
    if unit_path.exists() and meta_path.exists():
        with np.load(unit_path, allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
    else:
        pair = injected_pair(cfg, source, spec['partition'], spec['seed_index'])
        arrays = dict(background_eeg=pair['target']['eeg'], background_hb=pair['target']['hb'],
            wrong_background_eeg=pair['wrong']['eeg'], signal_eeg=pair['target_signal']['eeg'],
            signal_hb=pair['target_signal']['hb'], wrong_signal_eeg=pair['wrong_signal']['eeg'],
            eeg_operator=pair['target']['eeg_operator'], hb_operator=pair['target']['hb_operator'],
            truth_drivers=pair['target_signal']['drivers'], truth_states=pair['target_signal']['states'])
        write_npz(unit_path, **arrays)
        write_json(meta_path, pair['provenance'])
    dose = spec['dose']
    record = dict(eeg=arrays['background_eeg']+dose*arrays['signal_eeg'],
        hb=arrays['background_hb']+dose*arrays['signal_hb'],
        eeg_operator=arrays['eeg_operator'], hb_operator=arrays['hb_operator'])
    wrong = dict(eeg=arrays['wrong_background_eeg']+dose*arrays['wrong_signal_eeg'])
    return record, wrong, dose*arrays['signal_hb'][:, 0]


def evaluate_cell(payload):
    cfg, root, out, spec = payload
    path = Path(out)/'cells'/(spec['id']+'.json')
    if path.exists():
        return read_json(path)
    result = dict(spec, success=False)
    start = time.monotonic()
    try:
        source = retained_source(cfg, root)
        calibration = source['calibrations'][spec['arm']]
        clean = None
        if spec['family'] == 'measured':
            lookup = {row['id']: row for row in source['plan']['rows']}
            if spec['condition'] == 'wrong_EEG' and spec['wrong_eeg_id'] is None:
                result.update(status='unavailable_wrong_EEG')
                write_json(path, result)
                return result
            raw_record = load_retained(source['path'], lookup[spec['record_id']])
            target = np.asarray(raw_record['hb'])[:, 0]/float(source['coordinate']['hb_scale'])
            left, right = cfg['perturbations']['eeg_support']
            lag = cfg['perturbations']['shift_steps'].get(spec['condition'], 0)
            eeg_source_steps = [] if spec['condition'] in ('Hb_only', 'wrong_EEG') else np.arange(left, right)+lag
            record = scaled_record(raw_record, source['coordinate'], eeg_source_steps=eeg_source_steps)
            wrong = None
            if spec['condition'] == 'wrong_EEG':
                # The donor's Hb is never scaled/validated; EEG only.
                donor = load_retained(source['path'], lookup[spec['wrong_eeg_id']], fields=('eeg',))
                wrong = dict(eeg=(donor['eeg']/np.asarray(source['coordinate']['eeg_scale'])).reshape(120, 60))
        else:
            record, wrong, clean = synthetic_record(cfg, source, spec, out)
            target = record['hb'][:, 0]
        kwargs = dict(wrong=wrong, support=cfg['perturbations']['eeg_support'],
                      shift_steps=cfg['perturbations']['shift_steps'])
        if spec['method'] == 'SSM_fixed1':
            fit = fit_condition(record, calibration, source['coordinate'], source['parent'], spec['condition'], **kwargs)
            keep = ('success', 'status', 'converged', 'nfev', 'iterations', 'objective', 'objective_components',
                'integration_max_training_sd', 'scaled_gradient_inf_norm', 'failure_log', 'convergence_reason')
            result.update({k: fit[k] for k in keep if k in fit})
            prediction = fit['hb_prediction'][:, 0]
            arrays = dict(target_prediction=prediction, prediction=fit['prediction'],
                drivers=fit['drivers'], states=fit['states'], coefficients=fit['coefficients'],
                initial_coordinates=fit['initial_coordinates'], visible=fit['visible'])
            if fit['success'] and spec['condition'] == 'Hb_only':
                result['local_added_constraint'] = local_added_constraint(fit, record, calibration,
                    source['coordinate'], source['parent'], support=cfg['perturbations']['eeg_support'])
        else:
            prediction = predict_ridge(record, calibration, spec['condition'], **kwargs)
            arrays = dict(target_prediction=prediction)
            result.update(success=True, status='completed', independent_of_SSM_success=True,
                ridge_policy='frozen_joint_on_visible_EEG_rows_frozen_spatial_elsewhere')
        result.update(eeg_weight=0. if spec['condition'] == 'Hb_only' else 1.,
            target_hidden_at_all_times=True, input_EEG_steps=0 if spec['condition'] == 'Hb_only' else 56,
            input_Hb_donor_steps=120, frozen_parent_selected_weight=calibration['selected_eeg_weight'])
        write_npz(path.with_suffix('.npz'), **arrays)
        result['input_prediction_available'] = True
        if result['success']:
            result['input_fit_success'] = True
            sd = np.asarray(source['coordinate']['sd'])[60:62]
            try:
                result.update(score_target(prediction, target, sd, cfg['evaluation']['score_steps']))
            except ValueError as exc:
                result.update(success=False, status='failed_scoring', scoring_error=str(exc))
            if clean is not None:
                result['clean_response_score'] = score_target(prediction, clean, sd, cfg['evaluation']['score_steps'])
                result['actual_clean_target_training_sd_rms'] = float(np.sqrt(np.mean((clean[20:]/sd)**2)))
    except Exception as exc:
        result.update(status='failed_exception', success=False, error=str(exc), traceback=traceback.format_exc())
    result['elapsed_s'] = time.monotonic()-start
    write_json(path, result)
    return result


def bounded_run(payloads, workers, out):
    start, completed, failures = time.monotonic(), 0, 0
    payloads = iter(payloads)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = set()
        while True:
            while len(pending) < workers*2:
                try:
                    pending.add(pool.submit(evaluate_cell, next(payloads)))
                except StopIteration:
                    break
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                completed += 1
                failures += not result.get('success', False)
                write_json(Path(out)/'progress.json', dict(terminal=completed, noncompleted=failures,
                    workers=workers, max_in_flight=workers*2, numerical_threads=1,
                    elapsed_s=time.monotonic()-start, last_cell=result['id']))
    return dict(terminal=completed, noncompleted=failures, elapsed_s=time.monotonic()-start)


def valid(row):
    return bool(row.get('success')) and row.get('nmse') is not None


def unit_scores(rows, penalty, *, success_only=False, score_key='nmse'):
    groups = {}
    for row in rows:
        if success_only and not valid(row):
            continue
        score = row.get(score_key) if valid(row) else None
        groups.setdefault(row['unit'], []).append(float(score) if score is not None else penalty)
    return {key: float(np.sqrt(np.mean(values))) for key, values in groups.items()}


def compare(rows, arm, method, reference, cfg, *, common_success=False, restrict_record_ids=None):
    eligible = [r for r in rows if r['arm'] == arm and r['method'] == method]
    candidate = {r['record_id']: r for r in eligible if r['condition'] == 'paired'}
    control = {r['record_id']: r for r in eligible if r['condition'] == reference}
    ids = sorted(set(candidate) & set(control))
    if restrict_record_ids is not None:
        ids = sorted(set(ids) & set(restrict_record_ids))
    available = [key for key in ids if control[key]['status'] != 'unavailable_wrong_EEG']
    if common_success:
        available = [key for key in available if valid(candidate[key]) and valid(control[key])]
    left = [candidate[key] for key in available]
    right = [control[key] for key in available]
    scores = paired_statistics(unit_scores(left, cfg['evaluation']['failure_penalty_nmse']),
        unit_scores(right, cfg['evaluation']['failure_penalty_nmse']),
        repeats=cfg['evaluation']['bootstrap_repeats'], seed=cfg['seed'])
    scores.update(planned_window_pairs=len(ids), available_window_pairs=len(available),
        unavailable_window_pairs=len(ids)-sum(control[key]['status'] != 'unavailable_wrong_EEG' for key in ids),
        both_successful_window_pairs=sum(valid(candidate[key]) and valid(control[key]) for key in ids),
        candidate_failed=sum(not valid(candidate[key]) for key in available),
        reference_failed=sum(not valid(control[key]) for key in available), common_success_only=common_success)
    # NMSE is the pre-square-root primary paired increment.
    per_unit = {}
    for key in available:
        left_value = candidate[key]['nmse'] if valid(candidate[key]) else cfg['evaluation']['failure_penalty_nmse']
        right_value = control[key]['nmse'] if valid(control[key]) else cfg['evaluation']['failure_penalty_nmse']
        per_unit.setdefault(candidate[key]['unit'], []).append(right_value-left_value)
    scores['unit_delta_nmse_reference_minus_candidate'] = {key: float(np.mean(x)) for key, x in per_unit.items()}
    scores['mean_unit_delta_nmse_reference_minus_candidate'] = float(np.mean([np.mean(x) for x in per_unit.values()])) if per_unit else None
    def mean_nmse(records):
        groups = {}
        for record in records:
            value = record['nmse'] if valid(record) else cfg['evaluation']['failure_penalty_nmse']
            groups.setdefault(record['unit'], []).append(value)
        return {unit: float(np.mean(values)) for unit, values in groups.items()}
    scores['primary_nmse_statistics'] = paired_statistics(mean_nmse(left), mean_nmse(right),
        repeats=cfg['evaluation']['bootstrap_repeats'], seed=cfg['seed'], metric='nmse')
    return scores


def summarize(cfg, out, *, render=True):
    out = Path(out)
    plan = read_json(out/'manifest.json')
    rows = [read_json(out/'cells'/(spec['id']+'.json')) for spec in plan['cells'] if (out/'cells'/(spec['id']+'.json')).exists()]
    measured = [r for r in rows if r['family'] == 'measured']
    groups, comparisons = {}, {}
    for arm in cfg['model']['arms']:
        for method in ('SSM_fixed1', 'raw_ridge'):
            for condition in cfg['perturbations']['conditions']:
                selected = [r for r in measured if r['arm'] == arm and r['method'] == method and r['condition'] == condition]
                scores = unit_scores(selected, cfg['evaluation']['failure_penalty_nmse'])
                available = [r for r in selected if r['status'] != 'unavailable_wrong_EEG']
                groups['|'.join((arm, method, condition))] = dict(planned=len(selected), available=len(available),
                    succeeded=sum(valid(r) for r in selected), independent_units=len(scores),
                    unit_nrmse_with_penalty=scores, mean_unit_nrmse_with_penalty=float(np.mean(list(scores.values()))) if scores else None,
                    available_unit_nrmse=unit_scores(available, cfg['evaluation']['failure_penalty_nmse']),
                    status_counts={status: sum(r['status'] == status for r in selected) for status in sorted({r['status'] for r in selected})})
            for condition in cfg['perturbations']['conditions']:
                if condition == 'paired':
                    continue
                for success in (False, True):
                    comparisons['|'.join((arm, method, 'paired_vs_'+condition, 'common_success' if success else 'penalty'))] = compare(measured, arm, method, condition, cfg, common_success=success)
    joint_primary = {}
    for arm in cfg['model']['arms']:
        for method in ('SSM_fixed1', 'raw_ridge'):
            common_ids = {r['record_id'] for r in measured if r['arm'] == arm and r['method'] == method
                          and r['condition'] == 'wrong_EEG' and r['status'] != 'unavailable_wrong_EEG'}
            primary = {control: compare(measured, arm, method, control, cfg, restrict_record_ids=common_ids)
                       for control in ('Hb_only', 'wrong_EEG')}
            p_values = {control: value.get('primary_nmse_statistics', {}).get('exact_two_sided_sign_flip_p')
                        for control, value in primary.items()}
            ordered = sorted((p, control) for control, p in p_values.items() if p is not None)
            adjusted, previous = {}, 0.
            for rank, (p_value, control) in enumerate(ordered):
                previous = max(previous, min(1., p_value*(len(ordered)-rank)))
                adjusted[control] = previous
            joint_primary[arm+'|'+method] = dict(common_window_pairs=len(common_ids), comparisons=primary,
                holm_adjusted_two_sided_p=adjusted,
                evidence_requires='positive_primary_NMSE_increment_vs_both_controls_on_same_common_windows')
    injection = {}
    for method in ('SSM_fixed1', 'raw_ridge'):
        per_dose = {}
        for partition in ('calibration', 'evaluation'):
            for dose in cfg['synthetic']['doses_training_target_sd_rms']:
                sub = [r for r in rows if r['family'] == 'synthetic' and r['method'] == method and r['partition'] == partition and r['dose'] == dose]
                by_unit = {}
                for r in sub:
                    by_unit.setdefault(r['unit'], {})[r['condition']] = r
                units = {}
                for unit, conditions in by_unit.items():
                    if set(conditions) != set(cfg['synthetic']['conditions']):
                        continue
                    penalty = cfg['evaluation']['failure_penalty_nmse']
                    scores = {key: r['nmse'] if valid(r) else penalty for key, r in conditions.items()}
                    gains = dict(Hb_only=scores['Hb_only']-scores['paired'], wrong_EEG=scores['wrong_EEG']-scores['paired'])
                    units[unit] = dict(nmse=scores, gain_nmse=gains, minimum_gain_nmse=min(gains.values()),
                        all_success=all(valid(r) for r in conditions.values()),
                        status={key: r['status'] for key, r in conditions.items()},
                        clean_response_nmse={key: r.get('clean_response_score', {}).get('nmse') for key, r in conditions.items()})
                per_dose[f'{partition}|{dose:g}'] = dict(planned_units=len(by_unit), complete_units=len(units),
                    successful_units=sum(unit['all_success'] for unit in units.values()), units=units)
        calibration = per_dose['calibration|0']['units']
        null_values = [unit['minimum_gain_nmse'] for unit in calibration.values() if unit['all_success']]
        threshold = max(0., float(np.quantile(null_values, cfg['synthetic']['threshold_quantile']))) if null_values else None
        for group in per_dose.values():
            successful = [unit for unit in group['units'].values() if unit['all_success']]
            detections = sum(unit['minimum_gain_nmse'] > threshold for unit in successful) if threshold is not None else None
            group.update(empirical_reference_threshold_nmse=threshold, successful_unit_exceedances=detections,
                successful_unit_exceedance_rate=detections/len(successful) if successful and threshold is not None else None,
                all_planned_exceedance_rate=detections/group['planned_units'] if group['planned_units'] and detections is not None else None)
        injection[method] = dict(threshold_nmse=threshold, calibration_zero_successful=len(null_values),
            threshold_quantile=cfg['synthetic']['threshold_quantile'],
            criterion='per_seed_min_of_Hb_only_minus_paired_and_wrong_EEG_minus_paired_NMSE',
            threshold_fit='max(0,numpy_linear_quantile95_of_all_successful_calibration_zero_dose_seed_criteria)',
            exceedance_rule='strict_per_seed_criterion_greater_than_fixed_calibration_threshold',
            no_nominal_alpha_guarantee=True,
            interpretation=cfg['synthetic']['threshold_interpretation'], per_dose=per_dose,
            evaluation_zero_dose_false_positive_reference=per_dose['evaluation|0'])
    summary = dict(schema=SCHEMA, execution='completed' if len(rows) == plan['planned_cells'] else 'incomplete',
        planned=plan['planned_cells'], terminal=len(rows), status_counts={s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})},
        source_run=cfg['source_run'], source_manifest=plan, measured_groups=groups, measured_comparisons=comparisons,
        joint_primary=joint_primary, injection=injection, local_constraints=[dict(unit=r['unit'], arm=r['arm'], record_id=r.get('record_id'), **r['local_added_constraint']) for r in measured if 'local_added_constraint' in r],
        elapsed_cell_seconds=sum(r.get('elapsed_s', 0.) for r in rows),
        cell_elapsed_quantiles_s={family: np.quantile([r.get('elapsed_s', 0.) for r in rows if r['family'] == family and r['method'] == 'SSM_fixed1'], [.1, .5, .9]).tolist()
            for family in ('measured', 'synthetic') if any(r['family'] == family and r['method'] == 'SSM_fixed1' for r in rows)},
        claim_limits=cfg['claim_limits'])
    write_json(out/'summary.json', summary)
    if render:
        render_figures(summary, out)
    return summary


def render_figures(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = Path(out)/'figures'
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, arm in zip(axes, ('K1_true_H4_E12', 'K2_true_H4_E12')):
        labels = ('Hb_only', 'wrong_EEG', 'shift_m4s', 'shift_p4s', 'shift_m8s', 'shift_p8s')
        for offset, method, marker in ((-.10, 'SSM_fixed1', 'o'), (.10, 'raw_ridge', 's')):
            for i, control in enumerate(labels):
                key = '|'.join((arm, method, 'paired_vs_'+control, 'penalty'))
                data = summary['measured_comparisons'][key]
                if data.get('status') != 'computed':
                    continue
                y = data['delta_reference_minus_candidate']
                lower, upper = data['bootstrap_percentile95']
                ax.errorbar(i+offset, y, yerr=[[y-lower], [upper-y]], fmt=marker, capsize=3,
                    color='#28579a' if method == 'SSM_fixed1' else '#c36731', label=method if i == 0 else None)
        ax.axhline(0, color='black', lw=.8)
        ax.set_xticks(range(len(labels)), ['Hb only', 'Wrong EEG', '−4 s', '+4 s', '−8 s', '+8 s'], rotation=25)
        ax.set_title(arm+'; subject paired bootstrap')
        ax.set_ylabel('Reference − paired NRMSE (positive favors paired)')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Common EEG inputs [8,22) s; hidden Hb score [5,30) s; frozen weight 1')
    fig.tight_layout()
    fig.savefig(figures/'measured_pairing_information.png', dpi=240)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, arm in zip(axes, ('K1_true_H4_E12', 'K2_true_H4_E12')):
        for x, control in enumerate(('Hb_only', 'wrong_EEG')):
            for offset, method, marker, color in ((-.10, 'SSM_fixed1', 'o', '#28579a'),
                                                   (.10, 'raw_ridge', 's', '#c36731')):
                data = summary['joint_primary'][arm+'|'+method]['comparisons'][control]
                values = data.get('unit_delta_nmse_reference_minus_candidate', {})
                for index, (subject, value) in enumerate(sorted(values.items())):
                    ax.scatter(x+offset+(index-2.5)*.012, value, marker=marker, color=color,
                        alpha=.8, label=method if x == 0 and index == 0 else None)
                    ax.annotate(subject.replace('subject_', ''), (x+offset+(index-2.5)*.012, value),
                        xytext=(3, 2), textcoords='offset points', fontsize=6)
        ax.axhline(0, color='black', lw=.8)
        ax.set_xticks([0, 1], ['Hb only − paired', 'Wrong EEG − paired'])
        ax.set_ylabel('Subject mean NMSE difference (positive favors paired)')
        ax.set_title(arm+f'; {summary["source_manifest"]["paired_wrong_available"]} matched / {summary["source_manifest"]["measured_evaluation_windows"]} planned windows')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Each point is one subject; target Hb hidden at every time; fixed EEG weight1')
    fig.tight_layout()
    fig.savefig(figures/'measured_subject_information.png', dpi=240)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    doses = (0., .25, .5, 1.)
    for method, color in (('SSM_fixed1', '#28579a'), ('raw_ridge', '#c36731')):
        evidence = summary['injection'][method]
        for control, style in (('Hb_only', '-o'), ('wrong_EEG', '--s')):
            median, low, high = [], [], []
            for dose in doses:
                units = evidence['per_dose'][f'evaluation|{dose:g}']['units'].values()
                values = [u['gain_nmse'][control] for u in units if u['all_success']]
                if values:
                    q = np.quantile(values, [.1, .5, .9]); low.append(q[0]); median.append(q[1]); high.append(q[2])
                else:
                    low.append(np.nan); median.append(np.nan); high.append(np.nan)
            axes[0].plot(doses, median, style, color=color, label=method+' vs '+control)
            axes[0].fill_between(doses, low, high, color=color, alpha=.08)
        rates = [evidence['per_dose'][f'evaluation|{dose:g}']['successful_unit_exceedance_rate'] for dose in doses]
        axes[1].plot(doses, rates, '-o', color=color, label=method)
    axes[0].axhline(0, color='black', lw=.8)
    axes[0].set_ylabel('Control − paired NMSE; median and seed P10–P90')
    axes[1].set_ylabel('Independent evaluation seeds above empirical threshold')
    axes[1].set_ylim(-.03, 1.03)
    axes[0].set_title('Known response shape injection, matched K2')
    axes[1].set_title('Dose 0 is empirical false-positive reference')
    for ax in axes:
        ax.set_xlabel('Clean target RMS / frozen training Hb SD')
        ax.set_xticks(doses)
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(figures/'synthetic_dose_response.png', dpi=240)
    plt.close(fig)


def initialize(cfg, source, out, pilot):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    planned = manifest(cfg, source, pilot)
    for name, value in (('resolved_config.json', cfg), ('manifest.json', planned)):
        path = out/name
        if path.exists() and read_json(path) != serial(value):
            raise ValueError('immutable run identity/configuration changed; use fresh output')
        if not path.exists():
            write_json(path, value)
    if not (out/'execution.json').exists():
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout
        write_json(out/'execution.json', dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha=revision, source_worktree_status=status, parent_source_sha=source['identity']['source_sha'],
            command=sys.argv, cwd=str(ROOT), python=sys.executable,
            invocation_id=os.environ.get('INVOCATION_ID'), systemd_unit=os.environ.get('SYSTEMD_UNIT')))
        snapshot = out/'source_snapshot'
        for relative in ('experiments/scripts/evaluate_crossmodal_information.py', 'src/inference/crossmodal_information.py', 'src/inference/source_observability.py',
                         'src/inference/shared_driver_spatial.py', 'src/inference/observation_baselines.py',
                         'experiments/configs/physiology_semantic_tokenizer/crossmodal_information_v1.yaml'):
            destination = snapshot/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, destination)
    return planned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'dryrun', 'pilot', 'run', 'summary'), default='check')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    if not 1 <= args.workers <= cfg['resources']['max_workers']:
        parser.error('workers outside bounded resource contract')
    source = retained_source(cfg, args.project_root)
    plan = manifest(cfg, source, args.stage == 'pilot')
    if args.stage in ('check', 'dryrun'):
        print(json.dumps(serial(dict(status='completed', stage=args.stage, measured_arrays_read=0,
            planned_cells=plan['planned_cells'], measured_evaluation_windows=plan['measured_evaluation_windows'],
            paired_wrong_available=plan['paired_wrong_available'], frozen_parent_weights=plan['selected_parent_weights'],
            synthetic_calibration_seeds=cfg['synthetic']['calibration_seeds'], synthetic_evaluation_seeds=cfg['synthetic']['evaluation_seeds'])), indent=2))
        return 0
    if args.output is None:
        parser.error('--output required')
    out = args.output.resolve()
    boundary = (args.project_root/ARTIFACT_ROOT).resolve()
    if out.parent != boundary or out == source['path']:
        parser.error('output must be a fresh direct child of the owning artifact root')
    if args.stage == 'summary':
        result = summarize(cfg, out)
    else:
        out.mkdir(parents=True, exist_ok=True)
        with (out/'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            plan = initialize(cfg, source, out, args.stage == 'pilot')
            result = bounded_run(((cfg, str(args.project_root), str(out), spec) for spec in plan['cells']), args.workers, out)
            write_json(out/'completion.json', dict(status='completed', **result))
            result = summarize(cfg, out)
    print(json.dumps(serial(dict(stage=args.stage, execution=result['execution'], planned=result['planned'],
        terminal=result['terminal'], status_counts=result['status_counts'])), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
