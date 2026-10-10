#!/usr/bin/env python3
"""Public, versioned output-identifiability/continuity/cross-modal experiments."""
from __future__ import annotations

import argparse
from copy import deepcopy
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import yaml

from src.inference.observation_baselines import disjoint_native_bin_operator, eeg_band_power
from src.inference.ssm_output_audit import (
    OUTPUTS, OutputProblem, reference_parameters, visibility, hb_coordinates,
    fit_problem, worst_output_direction, output_profile,
)
from src.inference.shared_driver_rk4 import compiled_forward
from src.data.ssm_prepared import fit_feature_coordinate

DEFAULT = ROOT/'experiments/configs/physiology_semantic_tokenizer/ssm_output_support_v1.yaml'


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg['schema'] != 'ssm_output_support_v1' or cfg['experiment_id'] != 'SSM-OUTPUT-SUPPORT-v1'
            or cfg['data']['protected_data'] != 'forbidden' or cfg['data']['labels_as_inputs']
            or cfg['tensor']['steps'] != 360 or cfg['tensor']['dt_s'] != .25
            or cfg['tensor']['score_steps'] != [120, 240]
            or cfg['comparison']['arms'] != ['free', 'chain_matched', 'chain_first']
            or cfg['comparison']['masks'] != ['full', 'middle_Hb', 'middle_HbR']
            or cfg['resources']['numerical_threads'] != 1
            or cfg['data']['datasets'] != ['eeg_fnirs_single_trial', 'simultaneous_eeg_nirs', 'visual_cognitive_motivation']):
        raise ValueError('invalid output-support scope, tensor, masks or data boundary')
    if len(set(cfg['synthetic']['streams'].values())) != 3:
        raise ValueError('training/pilot/evaluation generator streams must be distinct')
    return cfg


@lru_cache(maxsize=1)
def base_config(path):
    return yaml.safe_load((ROOT/path).read_text())


def plan(cfg, out):
    parent = read(ROOT/cfg['source_pair_plan'])
    selected = []
    for ds in cfg['data']['datasets']:
        pairs = [p for p in parent['pairs'] if p['dataset'] == ds]
        training = [p for p in pairs if p['partition'] == 'development']
        available_sites = {p['site'] for p in training}
        for partition in ('development', 'evaluation'):
            subjects = sorted({p['subject'] for p in pairs if p['partition'] == partition})
            if partition == 'evaluation':
                subjects = subjects[:cfg['data']['evaluation_subjects_per_dataset']]
            for subject in subjects:
                candidates = [p for p in pairs if p['subject'] == subject and p['site'] in available_sites]
                if not candidates:
                    raise ValueError(f'no training-compatible native montage for {ds}/{subject}')
                candidate = sorted(candidates, key=lambda p: (not p['site'].startswith('prefrontal'), p['site']))[0]
                selected.append(dict(candidate, partition=partition))
        needed = {p['site'] for p in selected if p['dataset'] == ds and p['partition'] == 'evaluation'}
        present = {p['site'] for p in selected if p['dataset'] == ds and p['partition'] == 'development'}
        for site in sorted(needed-present):
            candidate = sorted([p for p in training if p['site'] == site], key=lambda p: p['subject'])[0]
            selected.append(dict(candidate, partition='development'))
    records = []
    for pair in selected:
        if len(pair['records']) != 2:
            raise ValueError('exactly two nonoverlapping matched-condition records required')
        a, b = pair['records']
        for key in ('dataset', 'subject', 'site', 'task', 'condition_counts', 'duration_s'):
            if a[key] != b[key]:
                raise ValueError('wrong-pair metadata mismatch: '+key)
        if a['anchor']['eeg_channels'] != b['anchor']['eeg_channels'] or a['anchor']['hb_channel'] != b['anchor']['hb_channel']:
            raise ValueError('wrong-pair montage/ROI mismatch')
        if a['record'] == b['record'] and a['eeg_start_s']+90 > b['eeg_start_s']+1e-8:
            raise ValueError('wrong EEG record overlaps target')
        for side, spec in enumerate(pair['records']):
            if spec['dataset'] == 'visual_cognitive_motivation' and spec['subject'] == 'S06' and 'Part1' in spec['record']:
                raise ValueError('excluded Visual S06 Part1 cannot enter the public panel')
            records.append(dict(spec, id=pair['id']+f'__r{side}', pair_id=pair['id'],
                side=side, donor_id=pair['id']+f'__r{1-side}', partition=pair['partition'],
                repeat_kind=pair['repeat_kind'], coordinate=pair['dataset']+'__'+pair['site']))
    panel = []
    for ds in cfg['data']['datasets']:
        eval_records = [r for r in records if r['dataset'] == ds and r['partition'] == 'evaluation']
        subjects = sorted({r['subject'] for r in eval_records})
        if len(subjects) != cfg['data']['evaluation_subjects_per_dataset']:
            raise ValueError('incomplete prespecified evaluation panel')
        panel.append(next(r['id'] for r in eval_records if r['subject'] == subjects[0] and r['side'] == 0))
    result = dict(schema='ssm_output_support_plan_v1', records=records, profile_panel=panel,
        metadata_source=cfg['source_pair_plan'], measured_arrays_read=0,
        calibration_subjects={ds: sorted({r['subject'] for r in records if r['dataset'] == ds and r['partition'] == 'development'}) for ds in cfg['data']['datasets']},
        evaluation_subjects={ds: sorted({r['subject'] for r in records if r['dataset'] == ds and r['partition'] == 'evaluation'}) for ds in cfg['data']['datasets']},
        confirmation='development_only_public_subjects_have_been_examined_in_earlier_generations',
        continuity='original_native_record_interval_not_concatenated_processed_trials',
        wrong_pair='same_subject_task_condition_counts_and_channels; onset_order_phase_not_exactly_matched')
    for ds in cfg['data']['datasets']:
        if set(result['calibration_subjects'][ds])&set(result['evaluation_subjects'][ds]):
            raise ValueError('calibration/evaluation subject overlap')
    write(out/'plan.json', result)
    return result


@lru_cache(maxsize=1)
def cache_index(root):
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    return CleanPhysiologyCacheIndex(Path(root))


def prepare_record(cfg, out, spec):
    """Only measured read boundary; registry identity and completed pilot precede it."""
    from src.data.unified_physiology import load_native_eeg_record, load_native_fnirs_record
    from src.data.homer2_preprocessing import modified_beer_lambert
    if spec['dataset'] not in cfg['data']['datasets'] or spec['partition'] not in ('development', 'evaluation'):
        raise ValueError('unregistered public identity')
    pilot = read(out/'pilot_summary.json')
    if not pilot['software_and_synthetic_pass']:
        raise ValueError('synthetic software qualification required before measured read')
    registry = cache_index(str(ROOT/cfg['data']['cache_root']))
    matches = [r for r in registry.records if r.join_key == spec['join_key']]
    if len(matches) != 1:
        raise ValueError('unique registered raw record required')
    record = matches[0]
    eeg = load_native_eeg_record(ROOT, record)
    hb = load_native_fnirs_record(ROOT, record)
    channels = {str(name).upper(): i for i, name in enumerate(eeg.channel_names)}
    selected = [channels[name.upper()] for name in spec['anchor']['eeg_channels']]
    start = round(spec['eeg_start_s']*eeg.sample_rate_hz)
    length = round(90*eeg.sample_rate_hz)
    values = eeg.values[start:start+length, selected]
    if values.shape != (length, 6) or not np.isfinite(values).all():
        raise ValueError('complete native continuous EEG support required')
    base = base_config(cfg['source_config'])
    power = eeg_band_power(values, bands=base['tensor']['eeg_bands_hz'],
        sample_rate=eeg.sample_rate_hz, target_rate=4.)
    if power.shape != (360, 6, 5) or np.any(power <= 0):
        raise ValueError('continuous bandpower tensor mismatch')
    native_mask = (hb['time_s'] >= spec['hb_start_s'])&(hb['time_s'] < spec['hb_start_s']+90)
    t = hb['time_s'][native_mask]-spec['hb_start_s']
    ops = disjoint_native_bin_operator(t)
    pair = spec['anchor']['hb_pair']
    native = hb['values'][native_mask, pair]
    optical_reference = None
    if hb['optical_intensity'] is not None:
        # Recompute the selected optical pair with ONLY the visible first 5s.
        # The generic native reader's full-record OD reference is never used.
        intensity = hb['optical_intensity'][native_mask, pair]
        if not np.isfinite(intensity).all() or np.any(intensity <= 0):
            raise ValueError('missing/nonpositive native intensity in registered interval')
        optical_reference = np.mean(intensity[t < 5.], axis=0)
        od = -np.log(intensity/optical_reference)
        converted, _ = modified_beer_lambert(od[:, None, :], wavelengths_nm=(760., 850.))
        native = converted[:, 0, :]
    if not np.isfinite(native).all():
        raise ValueError('nonfinite native Hb support')
    ef = ops['eeg']@np.log(power).reshape(360, 30)
    hf = ops['raw']@native
    # Use the operator's sample-support partition, including its explicit clock
    # roundoff tolerance, instead of a second independent timestamp threshold.
    hidden_native = (ops['native_groups'] >= 120)&(ops['native_groups'] < 240)
    changed = native.copy(); changed[hidden_native] += 1000*np.array([1., -2.])
    visible = np.r_[np.arange(120), np.arange(240, 360)]
    invariant = np.array_equal((ops['raw']@changed)[visible], hf[visible])
    if not invariant:
        raise AssertionError('hidden native Hb changed visible input features')
    path = out/'prepared'/(spec['id']+'.npz'); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, eeg=ef, hb=hf, eeg_operator=ops['eeg'], hb_operator=ops['model'],
        native_time=t, native_groups=ops['native_groups'])
    return dict(status='completed', native_eeg_source=str(eeg.source_path),
        eeg_rate=eeg.sample_rate_hz, eeg_unit=eeg.native_unit, hb_provenance=hb['provenance'],
        native_hb_rate=hb['sample_rate_hz'], eeg_clock_rounding_s=start/eeg.sample_rate_hz-spec['eeg_start_s'],
        feature_shapes=dict(eeg=ef.shape, hb=hf.shape), optical_reference=optical_reference,
        reference='first_5s_only_visible_not_latent_rest', native_samples=len(t),
        hidden_native_invariance=invariant, source_record=spec['join_key'])


def calibrate(cfg, out, pp):
    coordinates = {}
    for key in sorted({r['coordinate'] for r in pp['records']}):
        refs = [r for r in pp['records'] if r['coordinate'] == key and r['partition'] == 'development']
        good = [r for r in refs if read(out/'tasks'/'prepare'/(r['id']+'.json'))['status'] == 'completed']
        if not good:
            raise ValueError('no admitted training records for '+key)
        chunks = []
        for r in good:
            with np.load(out/'prepared'/(r['id']+'.npz')) as a:
                chunks.append((a['eeg'].copy(), a['hb'].copy()))
        co = fit_feature_coordinate(chunks)
        ys = [np.column_stack((e@co['pc']*co['eeg_factor'], h*co['hb_factor'])) for e, h in chunks]
        pooled = np.concatenate(ys)
        tx = hb_coordinates(pooled[:, 1:])
        co['output_scales'] = dict(T=float(tx[:, 0].std()), X=float(tx[:, 1].std()))
        ar = np.array([sum(y[:-1, k]@y[1:, k] for y in ys)/max(sum(y[:-1, k]@y[:-1, k] for y in ys), 1e-20) for k in range(3)])
        ar = np.clip(ar, 0., .98)
        innovations = np.concatenate([y[1:]-y[:-1]*ar for y in ys])
        cov = np.cov(innovations, rowvar=False, ddof=0)
        background_cov = cov.copy(); background_cov[0, 1:] = 0.; background_cov[1:, 0] = 0.
        background_cov += np.eye(3)*max(float(np.trace(cov)), 1e-12)*1e-8
        co.update(training_records=[r['id'] for r in good], training_subjects=sorted({r['subject'] for r in good}),
            training_planned_records=len(refs), training_failed_records=len(refs)-len(good),
            background_ar1=ar, measured_innovation_covariance=cov, background_innovation_covariance=background_cov,
            background_semantics='measured_training_feature_variation_not_pure_noise; EEG_Hb_innovation_crosscov_zeroed_for_known_dynamic_power',
            centering='common_visible_first_5s_90s_native_record_reference', covariance=np.cov(pooled, rowvar=False, ddof=0))
        coordinates[key] = co
    write(out/'coordinates.json', coordinates)
    return dict(status='completed', coordinates=list(coordinates), training_record_counts={k: len(v['training_records']) for k, v in coordinates.items()})


@lru_cache(maxsize=12)
def measured_arrays(out_string, record_id, coordinate):
    out = Path(out_string); co = read(out/'coordinates.json')[coordinate]
    with np.load(out/'prepared'/(record_id+'.npz')) as a:
        y = np.column_stack((a['eeg']@np.asarray(co['pc'])*co['eeg_factor'], a['hb']*co['hb_factor']))
        ops = dict(eeg=a['eeg_operator'].copy(), hb=a['hb_operator'].copy())
    return y, ops, np.asarray(co['sd']), co['output_scales']


@lru_cache(maxsize=1)
def synthetic_operators():
    ops = disjoint_native_bin_operator(np.arange(900)/10.)
    return dict(eeg=ops['eeg'], hb=ops['model'])


def process_canonical(canonical, ops):
    return np.column_stack((ops['eeg']@canonical[:, 0], ops['hb']@canonical[:, 1:]))


def synthetic_case(cfg, split, seed, scenario):
    gen = cfg['synthetic']; stream = gen['streams'][split]
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], stream, seed]))
    driver = gaussian_filter1d(rng.normal(size=360), gen['driver_smoothing_steps'])
    driver *= gen['driver_sd']/driver.std()
    initial = np.r_[0., np.ones(4)]
    if scenario == 'nonrest_common':
        initial += rng.normal(size=5)*gen['nonrest_sd']
    elif scenario != 'neural_only':
        raise ValueError('unregistered synthetic scenario')
    par = reference_parameters(base_config(cfg['source_config']))
    fwd = compiled_forward(driver, initial, par, .25, 8, False, False)
    ops = synthetic_operators(); physical = process_canonical(fwd['canonical_prediction'], ops)
    common = gaussian_filter1d(rng.normal(size=360), gen['component_smoothing_steps'])
    common *= gen['component_sd']/common.std()
    component = process_canonical(common[:, None]*np.array([0., .65, .35]), ops)
    if scenario == 'neural_only':
        component[:] = 0.
    noise = process_canonical(rng.normal(size=(360, 3))*np.asarray(gen['canonical_noise_sd']), ops)
    donor = gaussian_filter1d(rng.normal(size=360), gen['driver_smoothing_steps'])
    donor *= gen['driver_sd']/donor.std()
    return dict(target=physical+component+noise, physical_truth=physical,
        remaining_truth=component+noise, component_truth=component,
        driver=driver, states=fwd['states'], donor_eeg=ops['eeg']@donor,
        operators=ops, identity=seed, scenario=scenario, split=split)


def synthetic_coordinate(cfg):
    cases = [synthetic_case(cfg, 'training', i, 'neural_only') for i in range(cfg['synthetic']['training_seeds'])]
    pooled = np.concatenate([c['target'] for c in cases]); tx = hb_coordinates(pooled[:, 1:])
    return dict(sd=pooled.std(axis=0), output_scales=dict(T=tx[:, 0].std(), X=tx[:, 1].std()))


def metrics(problem, fit, original_target, mode, *, truth=None):
    v = fit['value']; sl = slice(120, 240)
    components = [2] if mode == 'middle_HbR' else [1, 2]
    error = (v['prediction'][sl][:, components]-original_target[sl][:, components])/problem.sd[components]
    result = dict(converged=fit['converged'], solver_status=fit['status'], evaluations=fit['evaluations'],
        iterations=fit['iterations'], scaled_gradient=fit['scaled_gradient'], objective=v['objective'], costs=v['costs'],
        score_nrmse=float(np.sqrt(np.mean(error**2))),
        HbO_nrmse=float(np.sqrt(np.mean(((v['prediction'][sl, 1]-original_target[sl, 1])/problem.sd[1])**2))),
        HbR_nrmse=float(np.sqrt(np.mean(((v['prediction'][sl, 2]-original_target[sl, 2])/problem.sd[2])**2))),
        visible_nrmse=float(np.sqrt(v['costs']['data']/problem.mask.sum())),
        remaining_closure_max=float(np.max(np.abs((v['physical']+fit['remaining']-problem.y)[problem.mask]))),
        missing_remaining_correct=bool(np.isnan(fit['remaining'][~problem.mask]).all()))
    if truth is not None:
        result.update(physical_truth_nrmse=float(np.sqrt(np.mean(((v['physical'][sl, 1:]-truth['physical_truth'][sl, 1:])/problem.sd[1:])**2))),
            remaining_truth_nrmse=float(np.sqrt(np.mean(((original_target[sl, 1:]-v['physical'][sl, 1:]-truth['remaining_truth'][sl, 1:])/problem.sd[1:])**2))),
            driver_truth_nrmse=float(np.sqrt(np.mean((v['states'][sl, 0]-truth['driver'][sl])**2))/.025),
            component_truth_nrmse=float(np.sqrt(np.mean(((v['component'][sl, 1:]-truth['component_truth'][sl, 1:])/problem.sd[1:])**2))))
        for name, index in (('flow', 2), ('volume', 3)):
            result[name+'_truth_rms_baseline_pct'] = float(np.sqrt(np.mean((v['states'][sl, index]-truth['states'][sl, index])**2))/.01)
        result['p_minus_v_truth_rms_baseline_pct'] = float(np.sqrt(np.mean(((v['states'][sl, 4]-v['states'][sl, 3])-(truth['states'][sl, 4]-truth['states'][sl, 3]))**2))/.01)
        truth_tx = hb_coordinates(truth['physical_truth'][sl, 1:])
        fit_tx = hb_coordinates(v['physical'][sl, 1:])
        for k, name in enumerate(('T', 'X')):
            result[name+'_truth_rmse'] = float(np.sqrt(np.mean((fit_tx[:, k]-truth_tx[:, k])**2)))
    return result


def save_fit(out, stage, key, fit, problem, target, output_scales):
    folder = out/'arrays'/stage; folder.mkdir(parents=True, exist_ok=True)
    v = fit['value']
    payload = dict(x=fit['x'], states=v['states'], physical=v['physical'], component=v['component'],
        prediction=v['prediction'], remaining=fit['remaining'], visible=problem.mask,
        target=target, input_target=problem.y, sd=problem.sd, T=hb_coordinates(v['physical'][:, 1:])[:, 0],
        X=hb_coordinates(v['physical'][:, 1:])[:, 1], T_scale=output_scales['T'], X_scale=output_scales['X'])
    np.savez_compressed(folder/(key+'.npz'), **payload)


def fit_spec(cfg, out, spec, *, synthetic=False):
    if synthetic:
        case = synthetic_case(cfg, spec['split'], spec['seed'], spec['scenario'])
        co = synthetic_coordinate(cfg); y = case['target'].copy(); ops = case['operators']
        sd, output_scales, donor = np.asarray(co['sd']), co['output_scales'], case['donor_eeg']
    else:
        r = spec['record']
        y, ops, sd, output_scales = measured_arrays(str(out), r['id'], r['coordinate'])
        y = y.copy(); case = None
        donor = measured_arrays(str(out), r['donor_id'], r['coordinate'])[0][:, 0] if spec['eeg'] == 'wrong_pair' else None
    original = y.copy()
    if spec['eeg'] == 'wrong_pair':
        y[:, 0] = donor
    mask = visibility(spec['mode'], eeg=spec['eeg'] != 'Hb_only', hb_context_steps=spec.get('context_steps', 360))
    y[~mask] = np.nan
    problem = OutputProblem(y, sd, ops, reference_parameters(base_config(cfg['source_config'])), cfg,
        arm=spec['arm'], visible=mask, boundaries=spec.get('boundaries', [120, 240]))
    fit = fit_problem(problem)
    result = metrics(problem, fit, original, spec['mode'], truth=case)
    stage = ('pilot' if spec['split'] == 'pilot' else 'synthetic') if synthetic else 'measured'
    save_fit(out, stage, spec['key'], fit, problem, original, output_scales)
    return dict(status='completed', **result)


def fit_key(record_id, arm, mode, eeg='paired', variant='standard'):
    return '__'.join((record_id, arm, mode, eeg, variant))


def fitting_variants():
    specs = []
    for arm in ('free', 'chain_matched', 'chain_first'):
        for mode in ('full', 'middle_Hb', 'middle_HbR'):
            specs.append(dict(arm=arm, mode=mode, eeg='paired', variant='standard'))
    for mode in ('middle_Hb', 'middle_HbR'):
        for eeg in ('wrong_pair', 'Hb_only'):
            specs.append(dict(arm='chain_matched', mode=mode, eeg=eeg, variant='standard'))
    return specs


def measured_specs(cfg, pp):
    specs = []
    for r in pp['records']:
        if r['partition'] != 'evaluation':
            continue
        variants = fitting_variants()
        for arm in ('free', 'chain_matched'):
            for boundaries in cfg['comparison']['shifted_boundaries']:
                variants.append(dict(arm=arm, mode='full', eeg='paired', boundaries=boundaries, variant='boundaries_'+'_'.join(map(str, boundaries))))
            for mode in ('full', 'middle_Hb'):
                variants.append(dict(arm=arm, mode=mode, eeg='paired', context_steps=300, variant='Hb_context_75s'))
        for v in variants:
            specs.append(dict(v, record=r, key=fit_key(r['id'], v['arm'], v['mode'], v['eeg'], v['variant'])))
    return specs


def synthetic_specs(cfg, *, pilot=False):
    specs = []
    seeds = range(2 if pilot else cfg['synthetic']['evaluation_seeds'])
    for seed in seeds:
        for scenario in cfg['synthetic']['scenarios']:
            for v in fitting_variants():
                identity = f'{scenario}__seed{seed:02d}'
                specs.append(dict(v, seed=seed, scenario=scenario, split='pilot' if pilot else 'evaluation',
                    key=fit_key(identity, v['arm'], v['mode'], v['eeg'], v['variant'])))
    return specs


def profile_specs(cfg, pp):
    specs = []
    for rid in pp['profile_panel']:
        r = next(r for r in pp['records'] if r['id'] == rid)
        for arm in cfg['comparison']['arms']:
            for objective in cfg['profiles']['objectives']:
                for name in cfg['profiles']['outputs']:
                    specs.append(dict(record=r, arm=arm, objective=objective, output=name, mode='full',
                        key='__'.join((rid, arm, objective, name, 'full'))))
    return specs


def profile_spec(cfg, out, spec):
    r = spec['record']; y, ops, sd, scales = measured_arrays(str(out), r['id'], r['coordinate'])
    mask = visibility(spec['mode'])
    problem = OutputProblem(y, sd, ops, reference_parameters(base_config(cfg['source_config'])), cfg,
        arm=spec['arm'], visible=mask, objective=spec['objective'])
    key = fit_key(r['id'], spec['arm'], spec['mode'])
    with np.load(out/'arrays'/'measured'/(key+'.npz')) as a:
        start = a['x'].copy()
    reference = fit_problem(problem, start)
    diagnostic, rows = output_profile(problem, reference, spec['output'], scales, cfg['profiles']['offsets'])
    local = []
    for mode in cfg['profiles']['local_modes']:
        m = visibility(mode)
        hidden = OutputProblem(y, sd, ops, problem.par, cfg, arm=spec['arm'], visible=m, objective='total')
        if mode == 'full':
            hidden_value = hidden.evaluate(reference['x'])
        else:
            hidden_key = fit_key(r['id'], spec['arm'], mode)
            with np.load(out/'arrays'/'measured'/(hidden_key+'.npz')) as a:
                hidden_value = hidden.evaluate(a['x'])
        for info in ('data', 'total'):
            d = worst_output_direction(hidden, hidden_value, spec['output'], scales, information=info)
            local.append(dict(mode=mode, information=info, **{k: v for k, v in d.items() if k not in ('direction', 'variable_direction')}))
    saved = {'reference_x': reference['x'], 'direction': diagnostic['direction']}
    for i, row in enumerate(rows):
        saved[f'x_{i}'] = row.pop('x')
        row['acceptable_witness'] = bool(reference['converged'] and row['converged']
            and abs(row['constraint_error']) <= cfg['solver']['constraint_tolerance']
            and row['objective_fraction'] <= cfg['profiles']['threshold_fraction'])
    folder = out/'arrays'/'profiles'; folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/(spec['key']+'.npz'), **saved)
    accepted = [r for r in rows if r['acceptable_witness']]
    return dict(status='completed', reference_converged=reference['converged'], reference_status=reference['status'],
        reference_gradient=reference['scaled_gradient'], reference_objective=reference['value']['objective'],
        diagnostic={k: v for k, v in diagnostic.items() if k not in ('direction', 'variable_direction')},
        local=local, profiles=rows,
        witnessed_min=min([0.]+[r['offset'] for r in accepted]),
        witnessed_max=max([0.]+[r['offset'] for r in accepted]),
        lower_objective_found=any(r['objective_fraction'] < -.001 for r in rows),
        interpretation='finite_valid_solution_witnesses_not_exhaustive_ranges_or_confidence_intervals')


def power_specs(cfg, out, pp):
    specs = []
    for dataset in cfg['data']['datasets']:
        key = sorted({r['coordinate'] for r in pp['records'] if r['dataset'] == dataset and r['partition'] == 'evaluation'})[0]
        for seed in range(cfg['power']['seeds_per_dataset']):
            for amplitude in cfg['power']['injection_Hb_training_SD']:
                for eeg in cfg['comparison']['eeg_conditions']:
                    for side in range(cfg['data']['records_per_subject']):
                        specs.append(dict(dataset=dataset, coordinate=key, seed=seed, side=side, amplitude=amplitude, eeg=eeg,
                            key=f'{dataset}__s{seed:02d}__r{side}__a{amplitude:g}__{eeg}'))
    return specs


def power_case(cfg, out, spec):
    co = read(out/'coordinates.json')[spec['coordinate']]
    ds_index = cfg['data']['datasets'].index(spec['dataset'])
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], 6103, ds_index, spec['seed'], spec['side']]))
    par = reference_parameters(base_config(cfg['source_config'])); ops = synthetic_operators()
    driver = gaussian_filter1d(rng.normal(size=360), 5.)
    driver *= .025/driver.std()
    sd = np.asarray(co['sd'])
    for _ in range(4):
        fw = compiled_forward(driver, np.r_[0., np.ones(4)], par, .25, 8, False, False)
        physical = process_canonical(fw['canonical_prediction'], ops)
        size = np.sqrt(np.mean((physical[120:240, 1:]/sd[1:])**2))
        if abs(size/spec['amplitude']-1) < 1e-4:
            break
        driver *= spec['amplitude']/max(size, 1e-8)
    fw = compiled_forward(driver, np.r_[0., np.ones(4)], par, .25, 8, False, False)
    physical = process_canonical(fw['canonical_prediction'], ops)
    amplitude = float(np.sqrt(np.mean((physical[120:240, 1:]/sd[1:])**2)))
    ar = np.asarray(co['background_ar1']); cov = np.asarray(co['background_innovation_covariance'])
    eps = rng.multivariate_normal(np.zeros(3), cov, size=760)
    bg = np.zeros_like(eps)
    for t in range(1, len(bg)):
        bg[t] = ar*bg[t-1]+eps[t]
    bg = process_canonical(bg[-360:], ops)
    donor = gaussian_filter1d(rng.normal(size=360), 5.)
    donor *= driver.std()/donor.std()
    target = physical+bg
    return dict(target=target, physical_truth=physical, remaining_truth=bg, component_truth=bg,
        driver=driver, states=fw['states'], donor_eeg=ops['eeg']@donor+bg[:, 0],
        operators=ops, achieved_amplitude=amplitude, sd=sd, output_scales=co['output_scales'])


def power_spec(cfg, out, spec):
    case = power_case(cfg, out, spec); y = case['target'].copy()
    if spec['eeg'] == 'wrong_pair':
        y[:, 0] = case['donor_eeg']
    mask = visibility('middle_Hb', eeg=spec['eeg'] != 'Hb_only')
    y[~mask] = np.nan
    problem = OutputProblem(y, case['sd'], case['operators'], reference_parameters(base_config(cfg['source_config'])), cfg,
        visible=mask, arm='chain_matched')
    fit = fit_problem(problem)
    result = metrics(problem, fit, case['target'], 'middle_Hb', truth=case)
    if spec['seed'] == 0:
        save_fit(out, 'power', spec['key'], fit, problem, case['target'], case['output_scales'])
    return dict(status='completed', achieved_amplitude=case['achieved_amplitude'], **result)


@lru_cache(maxsize=4)
def background_bank(parent_string, coordinate):
    """Only the frozen calibration records; no evaluation waveform in the bank."""
    parent = Path(parent_string); co = read(parent/'coordinates.json')[coordinate]
    plan = read(parent/'plan.json'); allowed = {r['id'] for r in plan['records'] if r['partition'] == 'development'}
    if not set(co['training_records']) <= allowed:
        raise ValueError('power background must use calibration identities only')
    bank = []
    for rid in co['training_records']:
        y, _, _, _ = measured_arrays(str(parent), rid, coordinate)
        bank.append(y)
    bank = np.asarray(bank)
    if bank.shape[1:] != (360, 3) or len(bank) < 2:
        raise ValueError('complete independent training feature bank required')
    mean = bank.mean(axis=0)
    return bank, mean, bank-mean, co


def draw_record_background(mean, centered, rng):
    """Preserve the empirical whole-trajectory moments, not just an AR(1) fit.

    The Hb pair shares one random weight vector. EEG has its own. The frozen
    per-time training mean is retained in both paired and wrong-pair controls.
    All first-5s references remain exactly the same linear observation contract.
    """
    count = len(centered)
    w_eeg = rng.normal(size=count)/np.sqrt(count)
    w_hb = rng.normal(size=count)/np.sqrt(count)
    return mean+np.column_stack((np.einsum('k,kt->t', w_eeg, centered[:, :, 0]),
        np.einsum('k,ktc->tc', w_hb, centered[:, :, 1:])))


def background_moment_audit(bank):
    mean = bank.mean(axis=0); centered = bank-mean
    grand = bank.mean(axis=(0, 1)); n = bank.shape[1]
    actual_cov = np.cov(bank.reshape(-1, 3), rowvar=False, ddof=0)
    expected = mean.T@mean/n-np.outer(grand, grand)
    random_cov = np.einsum('ktc,ktd->cd', centered, centered)/(len(bank)*n)
    expected[0, 0] += random_cov[0, 0]
    expected[1:, 1:] += random_cov[1:, 1:]
    rows = {}
    for lag in (1, 4, 20, 40, 120):
        actual = np.einsum('ktc,ktc->c', bank[:, :-lag], bank[:, lag:])/(len(bank)*(n-lag))
        predicted = np.mean(mean[:-lag]*mean[lag:], axis=0)+np.mean(centered[:, :-lag]*centered[:, lag:], axis=(0, 1))
        rows[str(lag)] = dict(actual=actual, expected=predicted, max_error=float(np.max(abs(actual-predicted))))
    fft_bank = np.fft.rfft(bank, axis=1)
    actual_spectrum = np.mean(abs(fft_bank)**2, axis=0)
    predicted_spectrum = abs(np.fft.rfft(mean, axis=0))**2+np.mean(abs(np.fft.rfft(centered, axis=1))**2, axis=0)
    return dict(training_covariance=actual_cov, expected_background_covariance=expected,
        Hb_covariance_max_error=float(np.max(abs(expected[1:, 1:]-actual_cov[1:, 1:]))),
        EEG_variance_error=float(abs(expected[0, 0]-actual_cov[0, 0])),
        lag_products=rows, expected_periodogram_max_error=float(np.max(abs(actual_spectrum-predicted_spectrum))),
        EEG_Hb_random_crosscovariance='zero; fixed_time_mean_shared_by_paired_and_wrong_pair',
        training_records=len(bank), mean=grand)


def power_case_v2(cfg, parent, spec):
    _, mean, centered, co = background_bank(str(parent), spec['coordinate'])
    ds_index = cfg['data']['datasets'].index(spec['dataset'])
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], 7103, ds_index, spec['seed'], spec['side']]))
    par = reference_parameters(base_config(cfg['source_config'])); ops = synthetic_operators()
    driver = gaussian_filter1d(rng.normal(size=360), 5.)
    driver *= .025/driver.std(); sd = np.asarray(co['sd'])
    if spec['amplitude'] == 0:
        driver[:] = 0.
    else:
        for _ in range(4):
            fw = compiled_forward(driver, np.r_[0., np.ones(4)], par, .25, 8, False, False)
            physical = process_canonical(fw['canonical_prediction'], ops)
            size = np.sqrt(np.mean((physical[120:240, 1:]/sd[1:])**2))
            if abs(size/spec['amplitude']-1) < 1e-4:
                break
            driver *= spec['amplitude']/max(size, 1e-8)
    fw = compiled_forward(driver, np.r_[0., np.ones(4)], par, .25, 8, False, False)
    physical = process_canonical(fw['canonical_prediction'], ops)
    bg = draw_record_background(mean, centered, rng)
    donor_bg = draw_record_background(mean, centered, rng)
    donor = gaussian_filter1d(rng.normal(size=360), 5.)
    donor *= driver.std()/donor.std()
    return dict(target=physical+bg, physical_truth=physical, remaining_truth=bg, component_truth=bg,
        driver=driver, states=fw['states'], donor_eeg=ops['eeg']@donor+donor_bg[:, 0],
        operators=ops, achieved_amplitude=float(np.sqrt(np.mean((physical[120:240, 1:]/sd[1:])**2))),
        sd=sd, output_scales=co['output_scales'])


def power_v2_spec(cfg, out, spec):
    parent = Path(cfg['power']['parent_run'])
    case = power_case_v2(cfg, parent, spec); y = case['target'].copy()
    if spec['eeg'] == 'wrong_pair':
        y[:, 0] = case['donor_eeg']
    mask = visibility('middle_Hb', eeg=spec['eeg'] != 'Hb_only'); y[~mask] = np.nan
    problem = OutputProblem(y, case['sd'], case['operators'], reference_parameters(base_config(cfg['source_config'])), cfg, visible=mask)
    fit = fit_problem(problem)
    result = metrics(problem, fit, case['target'], 'middle_Hb', truth=case)
    if spec['seed'] == 0:
        save_fit(out, 'power', spec['key'], fit, problem, case['target'], case['output_scales'])
    return dict(status='completed', achieved_amplitude=case['achieved_amplitude'], **result)


def run_power_calibration_v2(cfg, parent, contract_path, workers):
    contract = yaml.safe_load(Path(contract_path).read_text())
    if (contract['schema'] != 'ssm_output_power_calibration_v2' or contract['protected_data'] != 'forbidden'
            or (ROOT/contract['parent_run']).resolve() != parent.resolve()
            or contract['generator']['injection_Hb_training_SD'] != [0., .25, .5, 1.]
            or contract['generator']['records_per_identity'] != 2):
        raise ValueError('refined calibration scope/parent/mask mismatch')
    revised = deepcopy(cfg)
    revised['power'].update(parent_run=str(parent), calibration_version=2,
        injection_Hb_training_SD=contract['generator']['injection_Hb_training_SD'],
        seeds_per_dataset=contract['generator']['independent_identities_per_dataset'])
    out = parent/contract['output_subdirectory']; out.mkdir(parents=True, exist_ok=True)
    resolved = dict(contract=contract, parent_resolved_config=cfg)
    path = out/'resolved_config.yaml'
    if path.exists() and yaml.safe_load(path.read_text()) != resolved:
        raise ValueError('refined calibration is immutable')
    if not path.exists():
        path.write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
    snapshot(revised, out, 'power_v2')
    pp = read(parent/'plan.json'); specs = power_specs(revised, parent, pp)
    audits = {}
    for coordinate in sorted({s['coordinate'] for s in specs}):
        bank, _, _, co = background_bank(str(parent), coordinate)
        audits[coordinate] = dict(background_moment_audit(bank), training_record_ids=co['training_records'])
    if any(a['Hb_covariance_max_error'] > 1e-12 or a['expected_periodogram_max_error'] > 1e-9 for a in audits.values()):
        raise AssertionError('empirical moment calibration does not match its training evidence')
    write(out/'background_verification.json', dict(status='pass', coordinates=audits, measured_arrays_read=0))
    run_tasks(revised, out, 'power_v2', specs, workers)
    frame = task_rows(out, 'power_v2', specs, cfg['endpoints']['failure_penalty'])
    frame.to_csv(out/'metrics.csv', index=False)
    comparisons = compare_power(revised, frame); comparisons.to_csv(out/'paired.csv', index=False)
    matched = compare_power(revised, frame[frame.seed < 6]); matched.to_csv(out/'paired_n6.csv', index=False)
    result = dict(schema=contract['schema'], status='completed', planned=len(specs), terminal=len(frame),
        task_failures=int((frame.status != 'completed').sum()), converged=int(frame.accepted.sum()),
        numerical_failures=int(((frame.status == 'completed')&~frame.accepted).sum()),
        parent_run=contract['parent_run'], completed_at=datetime.now(timezone.utc).isoformat(),
        comparisons=comparisons.to_dict('records'), matched_n6_comparisons=matched.to_dict('records'),
        background_moment_verification='pass', interpretation=contract['statistics']['interpretation'])
    write(out/'summary.json', result)
    return {k: v for k, v in result.items() if k not in ('comparisons', 'matched_n6_comparisons')}


def worker(payload):
    cfg, out, stage, spec = payload; out = Path(out); began = time.monotonic()
    def timeout(*_):
        raise TimeoutError('registered per-task wall limit 3600s')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(3600)
    try:
        if stage == 'prepare':
            value = prepare_record(cfg, out, spec)
        elif stage in ('pilot', 'synthetic'):
            value = fit_spec(cfg, out, spec, synthetic=True)
        elif stage == 'measured':
            value = fit_spec(cfg, out, spec)
        elif stage == 'profiles':
            value = profile_spec(cfg, out, spec)
        elif stage == 'power':
            value = power_spec(cfg, out, spec)
        elif stage == 'power_v2':
            value = power_v2_spec(cfg, out, spec)
        else:
            raise ValueError(stage)
    except Exception as exc:
        value = dict(status='failed_task', error=repr(exc), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
    return dict(value, seconds=time.monotonic()-began, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def run_tasks(cfg, out, stage, specs, workers):
    folder = out/'tasks'/stage; folder.mkdir(parents=True, exist_ok=True)
    specs = [dict(s, key=s['id'] if stage == 'prepare' else s['key']) for s in specs]
    if len({s['key'] for s in specs}) != len(specs):
        raise ValueError('each registered task must have one distinct canonical identity')
    existing = [s for s in specs if (folder/(s['key']+'.json')).exists()]
    pending = [s for s in specs if not (folder/(s['key']+'.json')).exists()]
    began = time.monotonic(); terminal = len(existing)
    def update():
        current = dict(stage=stage, status='completed' if terminal == len(specs) else 'running',
            planned=len(specs), terminal=terminal, resumed=len(existing), workers=workers,
            elapsed_s=time.monotonic()-began,
            estimated_remaining_s=(len(specs)-terminal)*(time.monotonic()-began)/max(terminal-len(existing), 1),
            completed_at=datetime.now(timezone.utc).isoformat() if terminal == len(specs) else None)
        write(out/(stage+'_manifest.json'), current)
        return current
    update()
    iterator = iter(pending); in_flight = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def submit():
            if len(in_flight) >= workers*cfg['resources']['in_flight_per_worker']:
                return
            for spec in iterator:
                future = pool.submit(worker, (cfg, str(out), stage, spec)); in_flight[future] = spec
                if len(in_flight) >= workers*cfg['resources']['in_flight_per_worker']:
                    break
        submit()
        while in_flight:
            done, _ = wait(in_flight, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                spec = in_flight.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = dict(status='failed_worker', error=repr(exc))
                write(folder/(spec['key']+'.json'), dict(spec=spec, **result))
                terminal += 1
                print(stage, terminal, '/', len(specs), spec['key'], result['status'],
                    result.get('solver_status', ''), round(result.get('seconds', 0.), 2), flush=True)
            update(); submit()
    manifest = update()
    manifest['task_statuses'] = dict(Counter(read(folder/(s['key']+'.json'))['status'] for s in specs))
    write(out/(stage+'_manifest.json'), manifest)
    return manifest


def pilot_summary(cfg, out):
    specs = synthetic_specs(cfg, pilot=True)
    rows = [read(out/'tasks'/'pilot'/(s['key']+'.json')) for s in specs]
    completed = [r for r in rows if r['status'] == 'completed']
    value = dict(status='completed', planned=len(rows), task_failures=len(rows)-len(completed),
        converged=sum(r.get('converged', False) for r in rows),
        max_closure=max((r['remaining_closure_max'] for r in completed), default=np.inf),
        software_and_synthetic_pass=bool(len(completed) == len(rows) and all(r['missing_remaining_correct'] for r in completed)
            and all(r['remaining_closure_max'] < 1e-12 for r in completed)
            and sum(r['converged'] for r in completed)/len(rows) >= .95),
        measured_arrays_read=0)
    write(out/'pilot_summary.json', value)
    return value


def task_rows(out, stage, specs, penalty):
    rows = []
    for spec in specs:
        path = out/'tasks'/stage/(spec['key']+'.json')
        result = read(path) if path.exists() else dict(status='missing_task')
        flat = {k: v for k, v in spec.items() if k not in ('record', 'key')}
        if 'record' in spec:
            flat.update({k: spec['record'][k] for k in ('dataset', 'subject', 'side', 'pair_id', 'id', 'repeat_kind')})
        elif 'seed' in spec:
            flat.update(subject=str(spec['seed']), id=spec['key'])
        flat.update(key=spec['key'], **{k: v for k, v in result.items() if k not in ('spec', 'traceback', 'costs')})
        flat['accepted'] = bool(result.get('converged', False) and result['status'] == 'completed')
        for metric in ('score_nrmse', 'HbO_nrmse', 'HbR_nrmse', 'physical_truth_nrmse', 'component_truth_nrmse',
                       'remaining_truth_nrmse', 'driver_truth_nrmse'):
            if metric in result or stage in ('synthetic', 'power', 'power_v2') or metric in ('score_nrmse', 'HbO_nrmse', 'HbR_nrmse'):
                flat[metric+'_penalized'] = result.get(metric, penalty) if flat['accepted'] else penalty
                flat.setdefault(metric, np.nan)
        rows.append(flat)
    return pd.DataFrame(rows)


def paired_summary(left, right, *, cluster='subject', join=('subject', 'side'), metric='score_nrmse', repeats=2000, seed=20261010):
    """Left comparator minus right candidate, with declared full failure denominator."""
    columns = list(dict.fromkeys(list(join)+[cluster]))
    a = left[columns+[metric+'_penalized', metric, 'accepted']]
    b = right[columns+[metric+'_penalized', metric, 'accepted']]
    merged = a.merge(b, on=columns, suffixes=('_comparator', '_candidate'), validate='one_to_one')
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError('incomplete paired denominator')
    merged['delta'] = merged[metric+'_penalized_comparator']-merged[metric+'_penalized_candidate']
    grouped = merged.groupby(cluster, sort=True)[['delta', metric+'_penalized_comparator', metric+'_penalized_candidate']].mean()
    d = grouped['delta'].to_numpy()
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(len(d), size=(repeats, len(d)))].mean(axis=1)
    success = merged.accepted_comparator & merged.accepted_candidate
    success_rows = merged[success]
    good = success_rows.groupby(cluster, sort=True)[[metric+'_comparator', metric+'_candidate']].mean()
    return dict(planned_pairs=len(merged), paired_success=int(success.sum()), independent_units=len(grouped),
        comparator_failures=int((~merged.accepted_comparator).sum()), candidate_failures=int((~merged.accepted_candidate).sum()),
        comparator_mean=float(grouped[metric+'_penalized_comparator'].mean()),
        candidate_mean=float(grouped[metric+'_penalized_candidate'].mean()),
        improvement=float(d.mean()), lower=float(np.quantile(boot, .025)), upper=float(np.quantile(boot, .975)),
        units_improved=int((d > 0).sum()),
        successful_comparator_mean=float(good[metric+'_comparator'].mean()) if len(good) else np.nan,
        successful_candidate_mean=float(good[metric+'_candidate'].mean()) if len(good) else np.nan,
        success_paired_units=len(good),
        successful_relative_improvement=float((good[metric+'_comparator'].mean()-good[metric+'_candidate'].mean())/max(good[metric+'_comparator'].mean(), 1e-12)) if len(good) else np.nan)


def compare_measured(cfg, frame):
    rows = []
    standard = frame[(frame.variant == 'standard') & (frame.eeg == 'paired')]
    for dataset in cfg['data']['datasets']:
        for mode in cfg['comparison']['masks']:
            subset = standard[(standard.dataset == dataset) & (standard['mode'] == mode)]
            for arm in ('chain_matched', 'chain_first'):
                for metric in ('score_nrmse', 'HbO_nrmse', 'HbR_nrmse'):
                    rows.append(dict(dataset=dataset, mode=mode, comparison='continuity', comparator='free', candidate=arm, metric=metric,
                        **paired_summary(subset[subset.arm == 'free'], subset[subset.arm == arm], metric=metric,
                            repeats=cfg['endpoints']['bootstrap_repeats'])))
        for mode in ('middle_Hb', 'middle_HbR'):
            subset = frame[(frame.dataset == dataset)&(frame['mode'] == mode)&(frame.arm == 'chain_matched')&(frame.variant == 'standard')]
            for null in ('wrong_pair', 'Hb_only'):
                rows.append(dict(dataset=dataset, mode=mode, comparison='EEG_increment', comparator=null, candidate='paired', metric='score_nrmse',
                    **paired_summary(subset[subset.eeg == null], subset[subset.eeg == 'paired'], repeats=cfg['endpoints']['bootstrap_repeats'])))
        for arm in ('free', 'chain_matched'):
            subset = frame[(frame.dataset == dataset)&(frame['mode'] == 'middle_Hb')&(frame.arm == arm)&(frame.eeg == 'paired')]
            rows.append(dict(dataset=dataset, mode='middle_Hb', comparison='Hb_context', comparator='75s', candidate='90s', metric='score_nrmse', arm=arm,
                **paired_summary(subset[subset.variant == 'Hb_context_75s'], subset[subset.variant == 'standard'], repeats=cfg['endpoints']['bootstrap_repeats'])))
    return pd.DataFrame(rows)


def output_stability(cfg, out, frame):
    rows = []
    for _, alternative in frame[(frame.variant != 'standard')&(frame['mode'] == 'full')].iterrows():
        reference_key = fit_key(alternative['id'], alternative.arm, 'full')
        reference = frame[frame.key == reference_key].iloc[0]
        row = dict(dataset=alternative.dataset, subject=alternative.subject, side=alternative.side,
            id=alternative['id'], arm=alternative.arm, variant=alternative.variant,
            accepted=bool(reference.accepted and alternative.accepted), reference_key=reference_key, alternative_key=alternative.key)
        if row['accepted']:
            with np.load(out/'arrays'/'measured'/(reference_key+'.npz')) as a, np.load(out/'arrays'/'measured'/(alternative.key+'.npz')) as b:
                sl = slice(120, 240); sd = a['sd']
                for field, key, scale in (('T_change', 'T', a['T_scale']), ('X_change', 'X', a['X_scale'])):
                    difference = (b[key][sl]-a[key][sl])/scale
                    row[field] = float(np.sqrt(np.mean(difference**2)))
                    row[field+'_10s'] = float(np.sqrt(np.mean(difference.reshape(3, 40).mean(axis=1)**2)))
                for name in ('physical', 'component', 'prediction', 'remaining'):
                    row[name+'_change'] = float(np.sqrt(np.mean(((b[name][sl, 1:]-a[name][sl, 1:])/sd[1:])**2)))
                row['volume_change_baseline_pct'] = float(np.sqrt(np.mean((b['states'][sl, 3]-a['states'][sl, 3])**2))/.01)
                row['p_minus_v_change_baseline_pct'] = float(np.sqrt(np.mean(((b['states'][sl, 4]-b['states'][sl, 3])-(a['states'][sl, 4]-a['states'][sl, 3]))**2))/.01)
        else:
            for field in ('T_change', 'X_change', 'T_change_10s', 'X_change_10s', 'physical_change', 'component_change', 'prediction_change', 'remaining_change', 'volume_change_baseline_pct', 'p_minus_v_change_baseline_pct'):
                row[field] = cfg['endpoints']['failure_penalty']
        rows.append(row)
    table = pd.DataFrame(rows)
    paired = []
    table['perturbation'] = np.where(table.variant.str.startswith('boundaries'), 'boundary', 'Hb_context')
    for dataset in cfg['data']['datasets']:
        for perturbation in ('boundary', 'Hb_context'):
            sub = table[(table.dataset == dataset)&(table.perturbation == perturbation)].copy()
            for metric in ('T_change', 'X_change', 'physical_change', 'remaining_change', 'prediction_change', 'volume_change_baseline_pct', 'p_minus_v_change_baseline_pct'):
                sub[metric+'_penalized'] = sub[metric]
                paired.append(dict(dataset=dataset, perturbation=perturbation, metric=metric,
                    **paired_summary(sub[sub.arm == 'free'], sub[sub.arm == 'chain_matched'],
                        join=('subject', 'side', 'variant'), metric=metric, repeats=cfg['endpoints']['bootstrap_repeats'])))
    return table, pd.DataFrame(paired)


def compare_synthetic(cfg, frame):
    rows = []
    standard = frame[(frame.variant == 'standard')&(frame.eeg == 'paired')]
    for scenario in cfg['synthetic']['scenarios']:
        for mode in cfg['comparison']['masks']:
            group = standard[(standard.scenario == scenario)&(standard['mode'] == mode)]
            for arm in ('chain_matched', 'chain_first'):
                for metric in ('score_nrmse', 'physical_truth_nrmse', 'component_truth_nrmse', 'remaining_truth_nrmse', 'driver_truth_nrmse'):
                    rows.append(dict(scenario=scenario, mode=mode, arm=arm, metric=metric,
                        **paired_summary(group[group.arm == 'free'], group[group.arm == arm], join=('subject',), metric=metric,
                            repeats=cfg['endpoints']['bootstrap_repeats'])))
    return pd.DataFrame(rows)


def compare_power(cfg, frame):
    rows = []
    for dataset in cfg['data']['datasets']:
        for amplitude in cfg['power']['injection_Hb_training_SD']:
            group = frame[(frame.dataset == dataset)&(frame.amplitude == amplitude)]
            for null in ('wrong_pair', 'Hb_only'):
                rows.append(dict(dataset=dataset, amplitude=amplitude, comparator=null, candidate='paired',
                    **paired_summary(group[group.eeg == null], group[group.eeg == 'paired'],
                        repeats=cfg['endpoints']['bootstrap_repeats'])))
    return pd.DataFrame(rows)


def verify_results(cfg, out, pp, measured):
    from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
    errors = []; max_score = 0.; max_closure = 0.; checked = 0
    for row in measured.to_dict('records'):
        path = out/'arrays'/'measured'/(row['key']+'.npz')
        if not path.exists():
            if row['status'] == 'completed':
                errors.append(row['key']+':missing_arrays')
            continue
        with np.load(path) as a:
            mask = a['visible']; components = [2] if row['mode'] == 'middle_HbR' else [1, 2]
            value = np.sqrt(np.mean(((a['prediction'][120:240][:, components]-a['target'][120:240][:, components])/a['sd'][components])**2))
            max_score = max(max_score, abs(value-row['score_nrmse']))
            closure = np.max(np.abs((a['physical']+a['remaining']-a['input_target'])[mask]))
            max_closure = max(max_closure, float(closure))
            if not np.isnan(a['remaining'][~mask]).all():
                errors.append(row['key']+':missing_remaining')
            if not np.allclose(a['physical']+a['component'], a['prediction'], rtol=0, atol=1e-14):
                errors.append(row['key']+':component_closure')
            if not np.allclose(a['T'], a['physical'][:, 1]+a['physical'][:, 2], rtol=0, atol=1e-14):
                errors.append(row['key']+':T_identity')
            checked += 1
    replays = []
    par = reference_parameters(base_config(cfg['source_config']))
    for rid in pp['profile_panel']:
        r = next(r for r in pp['records'] if r['id'] == rid)
        _, ops, sd, _ = measured_arrays(str(out), rid, r['coordinate'])
        for arm in cfg['comparison']['arms']:
            key = fit_key(rid, arm, 'full'); path = out/'arrays'/'measured'/(key+'.npz')
            if not path.exists():
                errors.append(key+':replay_unavailable'); continue
            with np.load(path) as a:
                x, retained = a['x'].copy(), a['states'].copy()
            pieces = [(0, 120), (120, 240), (240, 360)] if arm == 'free' else [(0, 360)]
            four = []; eight = []
            for i, (left, right) in enumerate(pieces):
                initial = np.r_[x[360+5*i], np.exp(x[361+5*i:365+5*i])]
                four.append(nonlinear_driver_forward(x[left:right], initial, par, .25, substeps=4, derivative=False, numerical_backend='python')['states'])
                eight.append(nonlinear_driver_forward(x[left:right], initial, par, .25, substeps=8, derivative=False, numerical_backend='python')['states'])
            four, eight = np.concatenate(four), np.concatenate(eight)
            delta_h = np.column_stack((par.fixed.P0*(eight[:, 4]-four[:, 4])-par.fixed.Q0*(eight[:, 5]-four[:, 5]), par.fixed.Q0*(eight[:, 5]-four[:, 5])))
            delta_h = ops['hb']@delta_h
            replay = dict(key=key, independent_python_state_max=float(np.max(abs(four-retained))),
                substep_refinement_hb_rms_SD=float(np.sqrt(np.mean((delta_h/sd[1:])**2))))
            if replay['independent_python_state_max'] > 1e-9 or replay['substep_refinement_hb_rms_SD'] > 1e-3:
                errors.append(key+':integration_check')
            replays.append(replay)
    prepared = [read(out/'tasks'/'prepare'/(r['id']+'.json')) for r in pp['records']]
    hidden_invariance_count = sum(bool(r.get('hidden_native_invariance')) for r in prepared)
    result = dict(status='pass' if not errors and max_score < 1e-12 and max_closure < 1e-12 else 'fail',
        errors=errors, measured_arrays_verified=checked, score_recalculation_max_error=max_score,
        visible_decomposition_closure_max=max_closure, independent_replays=replays,
        native_hidden_invariance_pass=hidden_invariance_count, native_preparation_planned=len(prepared),
        data_boundary='declared_public_subjects_only; no_protected_campaign',
        information_scope='native_Hb_derived_components; EEG_90s_offline_power_features')
    write(out/'verification.json', result)
    return result


def summarize(cfg, out, pp):
    penalty = cfg['endpoints']['failure_penalty']
    stages = dict(measured=measured_specs(cfg, pp), synthetic=synthetic_specs(cfg),
                  power=power_specs(cfg, out, pp), profiles=profile_specs(cfg, pp))
    pending = []
    for stage, specs in stages.items():
        pending += [stage+'/'+s['key'] for s in specs if not (out/'tasks'/stage/(s['key']+'.json')).exists()]
    if pending:
        raise ValueError(f'cannot finalize while {len(pending)} registered tasks are missing')
    frames = {stage: task_rows(out, stage, stages[stage], penalty) for stage in ('measured', 'synthetic', 'power')}
    for stage, frame in frames.items():
        frame.to_csv(out/(stage+'_metrics.csv'), index=False)
    paired = compare_measured(cfg, frames['measured']); paired.to_csv(out/'measured_paired.csv', index=False)
    stability, stability_paired = output_stability(cfg, out, frames['measured'])
    stability.to_csv(out/'output_stability.csv', index=False); stability_paired.to_csv(out/'output_stability_paired.csv', index=False)
    synthetic = compare_synthetic(cfg, frames['synthetic']); synthetic.to_csv(out/'synthetic_paired.csv', index=False)
    power = compare_power(cfg, frames['power']); power.to_csv(out/'power_paired.csv', index=False)
    profiles = []; nodes = []; local = []
    for spec in stages['profiles']:
        result = read(out/'tasks'/'profiles'/(spec['key']+'.json'))
        base = dict(dataset=spec['record']['dataset'], subject=spec['record']['subject'], arm=spec['arm'],
            objective=spec['objective'], output=spec['output'], mode=spec['mode'], key=spec['key'])
        if result['status'] == 'completed':
            profiles.append(dict(base, status='completed', reference_converged=result['reference_converged'],
                reference_objective=result['reference_objective'], reference_gradient=result['reference_gradient'],
                witnessed_min=result['witnessed_min'], witnessed_max=result['witnessed_max'],
                width=result['witnessed_max']-result['witnessed_min'],
                acceptable_nodes=sum(r['acceptable_witness'] for r in result['profiles']),
                converged_nodes=sum(r['converged'] for r in result['profiles']),
                lower_objective_found=result['lower_objective_found'],
                **{k: v for k, v in result['diagnostic'].items() if k != 'compensation_fractions'},
                **{'direction_'+k: v for k, v in result['diagnostic']['compensation_fractions'].items()}))
            nodes.extend(dict(base, objective_kind=spec['objective'], **{k: v for k, v in r.items() if k != 'costs'}) for r in result['profiles'])
            local.extend(dict(base, **r) for r in result['local'])
        else:
            profiles.append(dict(base, status=result['status'], reference_converged=False, error=result.get('error')))
    pd.DataFrame(profiles).to_csv(out/'profile_ranges.csv', index=False)
    pd.DataFrame(nodes).to_csv(out/'profile_nodes.csv', index=False)
    pd.DataFrame(local).drop_duplicates(['dataset', 'subject', 'arm', 'output', 'mode', 'information', 'objective']).to_csv(out/'local_information.csv', index=False)
    verification = verify_results(cfg, out, pp, frames['measured'])
    result = dict(schema='ssm_output_support_summary_v1', experiment_id=cfg['experiment_id'], status='completed',
        completed_at=datetime.now(timezone.utc).isoformat(), scope=cfg['scope'],
        stages={stage: dict(planned=len(frame), terminal=len(frame), task_failures=int((frame.status != 'completed').sum()),
            converged=int(frame.accepted.sum()), numerical_failures=int(((frame.status == 'completed')&~frame.accepted).sum())) for stage, frame in frames.items()},
        profiles=dict(planned=len(profiles), completed=sum(r['status'] == 'completed' for r in profiles),
            reference_converged=sum(r['reference_converged'] for r in profiles),
            nodes_planned=len(profiles)*len(cfg['profiles']['offsets']), nodes_returned=len(nodes),
            nodes_converged=sum(r['converged'] for r in nodes), acceptable_witnesses=sum(r['acceptable_witness'] for r in nodes)),
        calibration_subjects=pp['calibration_subjects'], evaluation_subjects=pp['evaluation_subjects'],
        measured_comparisons=paired.to_dict('records'), stability_comparisons=stability_paired.to_dict('records'),
        synthetic_comparisons=synthetic.to_dict('records'), power_comparisons=power.to_dict('records'),
        verification=verification['status'],
        output_contract=dict(decomposition='y_visible=SSM_observation+complete_remaining; missing_remaining=NaN',
            T='processed P0(p-1)', X='processed Q0(p-q)', states='model_conditional_diagnostic_trajectories',
            parameters='tau=2, gain=1, kappa=.64 fixed_by_strategy_not_data_identified',
            intervals='engineering_solution_witnesses_and_subject_bootstrap_not_posterior'),
        evidence_tables=['measured_metrics.csv', 'measured_paired.csv', 'output_stability.csv', 'output_stability_paired.csv',
            'synthetic_metrics.csv', 'synthetic_paired.csv', 'power_metrics.csv', 'power_paired.csv',
            'profile_ranges.csv', 'profile_nodes.csv', 'local_information.csv', 'verification.json'])
    write(out/'summary.json', result)
    return dict(status=result['status'], stages=result['stages'], profiles=result['profiles'], verification=result['verification'])


def snapshot(cfg, out, stage):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = out/'source_snapshots'/(stage+'_'+stamp)
    paths = [Path(__file__).relative_to(ROOT), Path('src/inference/ssm_output_audit.py'),
        Path('src/inference/shared_driver_rk4.py'), Path('src/inference/observation_baselines.py'),
        Path('src/inference/shared_driver_reconstruction.py'),
        Path('src/data/ssm_prepared.py'), Path('src/data/unified_physiology.py'),
        Path('tests/test_ssm_output_support.py')]
    folder.mkdir(parents=True)
    for path in paths:
        dest = folder/path; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/path, dest)
    write(folder/'identity.json', dict(source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        dirty_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        snapshots=[str(p) for p in paths], created_at=datetime.now(timezone.utc).isoformat()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['plan', 'pilot', 'prepare', 'calibrate', 'synthetic', 'measured', 'profiles', 'power', 'power_calibration_v2', 'summary'], required=True)
    parser.add_argument('--calibration-contract', type=Path, default=ROOT/'experiments/configs/physiology_semantic_tokenizer/ssm_output_power_calibration_v2.yaml')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='pilot/prepare throughput benchmark prefix; not a final experiment')
    args = parser.parse_args(); cfg = load_config(args.config)
    out = args.run_dir.resolve(); root = (ROOT/cfg['artifact_root']).resolve()
    if not out.is_relative_to(root) or out == root:
        raise ValueError('versioned owning artifact root required')
    if not 1 <= args.workers <= cfg['resources']['max_workers']:
        raise ValueError('workers outside reviewed resource envelope')
    if args.limit and args.stage not in ('pilot', 'prepare'):
        raise ValueError('partial prefixes only for throughput/software preparation')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = out/'resolved_config.yaml'
        if path.exists() and yaml.safe_load(path.read_text()) != cfg:
            raise ValueError('existing run config is immutable')
        if not path.exists():
            path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        snapshot(cfg, out, args.stage)
        if args.stage == 'plan':
            result = plan(cfg, out)
            print(json.dumps({k: result[k] for k in ('calibration_subjects', 'evaluation_subjects', 'profile_panel')}, ensure_ascii=False), flush=True)
            return
        pp = read(out/'plan.json')
        if args.stage == 'power_calibration_v2':
            result = run_power_calibration_v2(cfg, out, args.calibration_contract, args.workers)
        elif args.stage == 'calibrate':
            result = calibrate(cfg, out, pp)
        elif args.stage == 'summary':
            # The summarizer is kept in the same owning executable.
            result = summarize(cfg, out, pp)
        else:
            if args.stage in ('pilot', 'synthetic'):
                specs = synthetic_specs(cfg, pilot=args.stage == 'pilot')
            elif args.stage == 'prepare':
                specs = pp['records']
            elif args.stage == 'measured':
                specs = measured_specs(cfg, pp)
            elif args.stage == 'profiles':
                specs = profile_specs(cfg, pp)
            else:
                specs = power_specs(cfg, out, pp)
            if args.limit:
                specs = specs[:args.limit]
            result = run_tasks(cfg, out, args.stage, specs, args.workers)
            if args.stage == 'pilot' and not args.limit:
                result = pilot_summary(cfg, out)
        print(json.dumps(clean(result), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
