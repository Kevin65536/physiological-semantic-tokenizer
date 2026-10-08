#!/usr/bin/env python3
"""Bounded SSM response/independent-observation development experiment.

One controller, exact public parent identities, fixed-budget terminal task
records. No raw-data reader, protected evaluation, tokenizer or automatic
confirmation launch. All numerical fitting uses the existing Balloon solver.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import yaml

from src.inference.observation_baselines import native_feature_operators
from src.inference.shared_driver_attribution import (
    equal_capacity_hb_basis, fit_conditioned_shared_driver, fit_reduced_rank_readout,
    predict_reduced_rank_readout, project_component, grouped_component_overlap,
    fit_standardized_ridge, predict_standardized_ridge,
)
from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
from src.inference.t3a_balloon_robust_ssm import BalloonFixedParameters, BalloonFreeParameters, BalloonParameters

DEFAULT_CONFIG = CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/ssm_strengthening_v1.yaml'
SOURCE_RUN = 'experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1'
ATTRIBUTION_RUN = 'experiments/runs/physiology_semantic_tokenizer/shared_driver_attribution/20261002_v1'
SOURCE_CONFIG = 'experiments/configs/physiology_semantic_tokenizer/shared_driver_physiology_semantics_v1.yaml'
ARTIFACT_ROOT = 'experiments/runs/physiology_semantic_tokenizer/ssm_strengthening'
RESPONSE_GRID = [(2., .64), (1., .64), (4., .64), (2., .32), (2., 1.28),
                 (1., .32), (1., 1.28), (4., .32), (4., 1.28)]
SCENARIOS = ['shared_neural_only', 'common_colored', 'common_correlated', 'global_component_only',
             'target_neural_only', 'donor_contamination', 'driver_amplitude', 'driver_amplitude_gain']


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
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg.get('schema') != 'ssm_strengthening_v1' or cfg.get('experiment_id') != 'SSM-STRENGTHEN-v1'
            or cfg.get('source_run') != SOURCE_RUN or cfg.get('attribution_run') != ATTRIBUTION_RUN
            or cfg.get('source_config') != SOURCE_CONFIG or cfg.get('artifact_root') != ARTIFACT_ROOT
            or cfg.get('data_boundary') != 'exact_public_parent_panel_training_selection_and_synchronous_donors'
            or cfg.get('protected_data') != 'forbidden'):
        raise ValueError('versioned identity or public data boundary mismatch')
    if (cfg['tensor'] != dict(steps=120, dt_s=.25,
            components=['EEG_logpower_PCA', 'relative_HbO', 'relative_HbR'], state_order=['r','s','f','v','p','q'])
            or cfg['response']['tau_s'] != [1.,2.,4.] or cfg['response']['kappa'] != [.32,.64,1.28]
            or cfg['response']['crossfit_folds'] != 3 or cfg['observation']['modes'] != 4
            or cfg['observation']['rho'] != .35 or cfg['observation']['ranks'] != [1,2]
            or cfg['observation']['ridge_weights'] != [.1,1.,10.]
            or cfg['synthetic']['scenarios'] != SCENARIOS
            or cfg['measured']['datasets'] != ['eeg_fnirs_single_trial','simultaneous_eeg_nirs','visual_cognitive_motivation']
            or cfg['measured']['modes'] != ['full','HbO_hidden','HbR_hidden','center_Hb','Hb_hidden','EEG_hidden']
            or cfg['synthetic']['modes'] != ['full','Hb_hidden','EEG_hidden']
            or cfg['response']['selection_modes'] != ['HbO_hidden','HbR_hidden','center_Hb']
            or cfg['resources']['numerical_threads'] != 1
            or not 1 <= cfg['resources']['max_workers'] <= 48):
        raise ValueError('tensor, candidate, mask, split or resource contract mismatch')
    streams = cfg['synthetic']['seed_streams']
    if len(set(streams.values())) != 4 or set(streams) != {'train','selection','evaluation','pilot'}:
        raise ValueError('independent synthetic seed streams required')
    for key in ('train_repeats','selection_repeats','evaluation_repeats'):
        if not isinstance(cfg['synthetic'][key], int) or cfg['synthetic'][key] < 3:
            raise ValueError('at least three independent identities per synthetic partition')
    return cfg


def parameters(base, candidate=0, *, truth=None):
    tau, kappa = RESPONSE_GRID[candidate] if truth is None else truth
    fixed = {k: v for k, v in base['fixed'].items() if k not in ('tau','kappa')}
    return BalloonParameters(fixed=BalloonFixedParameters(**fixed),
                             free=BalloonFreeParameters(tau=tau, kappa=kappa))


@lru_cache(maxsize=1)
def operators():
    native = native_feature_operators(120)
    op = np.zeros((360,360))
    op[0::3,0::3] = native['eeg']
    op[1::3,1::3] = op[2::3,2::3] = native['fnirs']@native['native_interpolation']
    cosines = np.cos(np.pi*(np.arange(120)[:,None]+.5)*np.arange(1,5)/120)
    temporal = op[1::3,1::3]@cosines
    q, _ = np.linalg.qr(temporal)
    return op, equal_capacity_hb_basis(op, modes=4, rho=.35), q@q.T


def visible_mask(mode):
    mask = np.ones((120,3), bool)
    if mode == 'full':
        pass
    elif mode == 'HbO_hidden':
        mask[:,1] = False
    elif mode == 'HbR_hidden':
        mask[:,2] = False
    elif mode == 'Hb_hidden':
        mask[:,1:] = False
    elif mode == 'EEG_hidden':
        mask[:,0] = False
    elif mode == 'center_Hb':
        mask[52:68,1:] = False
    else:
        raise ValueError(f'unregistered mask: {mode}')
    return mask


def fit(base, target, sd, cal, candidate=0, *, mode='full', component=False,
        mean=None, rho=.35, truth=None, jacobian=False):
    solver = base['solver']
    options = {k: solver[k] for k in ('numerical_backend','substeps','penalty','initial_penalty',
                                     'flow_prior_weight','flow_prior_log_sd','gradient_tolerance')}
    if component:
        options.update(observation_basis=equal_capacity_hb_basis(operators()[0], modes=4, rho=rho),
                       observation_coefficient_sd=np.asarray(cal['coefficient_sd'])*cal['coefficient_multiplier'])
    return fit_conditioned_shared_driver(target, parameters(base, candidate, truth=truth), .25,
        component_mean=mean, visible=visible_mask(mode), sd=sd, mean_operator=operators()[0],
        max_evaluations=solver['max_evaluations_per_trial'], return_jacobian=jacobian, **options)


def rmse(values):
    return float(np.sqrt(np.mean(np.asarray(values, float)**2)))


def metrics(result, case, mode, *, support=None):
    target, sd = case['target'], np.asarray(case['sd'])
    score = ~visible_mask(mode) if mode != 'full' else np.ones_like(target, bool)
    if support is not None:
        score &= np.asarray(support)[:,None]
    row = dict(converged=bool(result.get('converged', False)), status=result.get('status','missing'),
               objective=result.get('objective'), evaluations=result.get('evaluations'),
               score_count=int(score.sum()), score_nrmse=None, Hb_nrmse=None,
               EEG_nrmse=None, HbO_nrmse=None, HbR_nrmse=None)
    if 'prediction' not in result:
        return row
    error = (result['prediction']-target)/sd
    row['score_nrmse'] = rmse(error[score])
    for i, label in enumerate(('EEG','HbO','HbR')):
        row[label+'_nrmse'] = rmse(error[score[:,i],i]) if score[:,i].any() else None
    hb = score[:,1:]
    row['Hb_nrmse'] = rmse(error[:,1:][hb]) if hb.any() else None
    if 'physical_truth' in case and 'driver' in result:
        truth = case['driver']
        centered = result['driver']-result['driver'][:20].mean()
        target_centered = truth-truth[:20].mean()
        row.update(physical_error=rmse((result['physical_prediction'][:,1:]-case['physical_truth'][:,1:])/sd[1:]),
                   component_error=rmse((result['observation_component'][:,1:]-case['observation_truth'][:,1:])/sd[1:]),
                   driver_error=rmse(result['driver']-truth)/.025,
                   driver_shape_error=rmse(centered-target_centered)/.025,
                   driver_5s_error=rmse(centered.reshape(6,20).mean(1)-target_centered.reshape(6,20).mean(1))/.025)
    return row


def synthetic_case(cfg, base, population, split, repeat, spectrum, initial, scenario):
    if scenario not in SCENARIOS or split not in cfg['synthetic']['seed_streams']:
        raise ValueError('unregistered synthetic identity')
    seed = [cfg['seed'], cfg['synthetic']['seed_streams'][split], repeat,
            int(spectrum == 'mixed'), int(initial == 'nonrest')]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    shared = gaussian_filter1d(rng.normal(size=120), 5.)
    if spectrum == 'mixed':
        shared += .18*gaussian_filter1d(rng.normal(size=120), .7)
    shared *= .025/shared.std()
    drivers = np.repeat(shared[None], 3, axis=0)
    for i in (1,2):
        local = gaussian_filter1d(rng.normal(size=120), 5.)
        drivers[i] = .85*shared+.15*.025*local/local.std()
    starts = np.repeat([[0.,1.,1.,1.,1.]],3,axis=0)
    perturbations = rng.normal(size=(3,5))*.015
    if initial == 'nonrest':
        starts += perturbations
    if scenario == 'global_component_only':
        drivers[:] = 0.
    elif scenario == 'target_neural_only':
        drivers[0] *= 1.5
    elif scenario in ('driver_amplitude','driver_amplitude_gain'):
        drivers *= 1.5
    truth = cfg['synthetic']['populations'][population]
    physical, states = [], []
    for r, x0 in zip(drivers,starts):
        forward = nonlinear_driver_forward(r, x0, parameters(base, truth=truth), .25,
            substeps=8, derivative=False, numerical_backend='numba')
        physical.append((operators()[0]@forward['canonical_prediction'].ravel()).reshape(120,3))
        states.append(forward['states'])
    physical = np.asarray(physical)
    noise = rng.normal(size=(3,120,3))*np.asarray(cfg['synthetic']['canonical_noise_sd'])
    noise = np.array([(operators()[0]@v.ravel()).reshape(120,3) for v in noise])
    colored = gaussian_filter1d(rng.normal(size=120),8.)
    correlated = gaussian_filter1d(shared,8.)
    common = correlated if scenario == 'common_correlated' else colored
    common = cfg['synthetic']['common_sd']*common/common.std()
    component = np.zeros_like(physical)
    if scenario in ('common_colored','common_correlated','global_component_only'):
        for i, gain in enumerate((1., .8, 1.2)):
            raw = gain*common[:,None]*np.array([0.,.65,.35])
            component[i] = (operators()[0]@raw.ravel()).reshape(120,3)
    if scenario == 'donor_contamination':
        dirty = gaussian_filter1d(rng.normal(size=120),3.)
        dirty *= .06/dirty.std()
        raw = dirty[:,None]*np.array([0.,.2,.8])
        component[1] = (operators()[0]@raw.ravel()).reshape(120,3)
    if scenario == 'driver_amplitude_gain':
        component[:,:,1:] = .3*physical[:,:,1:]
    observations = physical+component+noise
    if scenario == 'driver_amplitude_gain':
        observations[:,:,0] = .7*physical[:,:,0]+noise[:,:,0]
    return dict(target=observations[0], donors=observations[1:], donor_ratios=np.ones(2),
        donor_signature=['region_1','region_2'], raw_donors=np.concatenate([v[:,1:] for v in observations[1:]],axis=1),
        physical_truth=physical[0], observation_truth=component[0], driver=drivers[0], states=states[0],
        truth_response=truth, seed=seed, subject=f'{split}_{repeat}_{spectrum}_{initial}', task=scenario, condition=scenario,
        population=population, scenario=scenario)


def synthetic_calibration(cfg, base):
    cases = [synthetic_case(cfg,base,'H0','train',i,'slow' if i%2 == 0 else 'mixed',
                            'rest' if i%4 < 2 else 'nonrest','shared_neural_only')
             for i in range(cfg['synthetic']['train_repeats'])]
    return dict(sd=np.std(np.concatenate([c['target'] for c in cases]),axis=0),
        coefficient_sd=cfg['synthetic']['coefficient_sd'], coefficient_multiplier=cfg['synthetic']['coefficient_multiplier'],
        driver_sd=.025, source='H0_training_seeds_only_fixed_across_response_populations', selected_rho=.35)


def attach_scales(case, cal):
    case['sd'] = np.asarray(cal['sd'])
    if 'donor_sds' not in case:
        case['donor_sds'] = np.repeat(np.asarray(cal['sd'])[None],len(case['donors']),axis=0)
    return case


def donor_features(base, case, cal, candidate):
    residuals, records = [], []
    for y, sd, ratio in zip(case['donors'],case['donor_sds'],case['donor_ratios']):
        result = fit(base,y,sd,cal,candidate)
        records.append(dict(converged=result['converged'], status=result['status']))
        residuals.append((y[:,1:]-result['physical_prediction'][:,1:])*ratio
                         if 'physical_prediction' in result else np.full((120,2),np.nan))
    return (np.concatenate(residuals,axis=1) if residuals else np.empty((120,0)),
            bool(records and all(r['converged'] for r in records)), records)


def response_choice(rows, *, component=None, exclude_subjects=()):
    frame = pd.DataFrame([r for r in rows if r['subject'] not in exclude_subjects
                          and (component is None or r['component'] == component)])
    if frame.empty:
        raise ValueError('empty independent response selection')
    frame['risk'] = [r.score_nrmse if r.converged and pd.notna(r.score_nrmse) and np.isfinite(r.score_nrmse) else 100.
                     for r in frame.itertuples()]
    scores = frame.groupby(['candidate','subject']).risk.mean().groupby('candidate').mean()
    if set(scores.index) != set(range(9)):
        raise ValueError('incomplete candidate denominator')
    families = {'H0':[0], 'Htau':[0,1,2], 'Hkappa':[0,3,4], 'Htau_kappa':list(range(9))}
    chosen = {name:min(indices,key=lambda i:(float(scores[i]),i)) for name,indices in families.items()}
    return dict(selected=chosen['Htau_kappa'], families=chosen, scores=scores.to_dict(),
                selection_subjects=sorted(frame.subject.unique()), excluded_subjects=sorted(exclude_subjects))


def training_worker(payload):
    cfg, base, root, spec, cal = payload
    out = Path(root)
    arrays, scores, cases_meta = {}, [], []
    if spec['kind'] == 'synthetic':
        cases = [attach_scales(synthetic_case(cfg,base,spec['population'],spec['split'],spec['repeat'],
                    spec['spectrum'],spec['initial'],scenario),cal)
                 for scenario in cfg['synthetic']['training_scenarios']]
    else:
        cases = [load_measured_case(spec['case'], cal)]
    for ci, case in enumerate(cases):
        arrays[f'{ci}_target'] = case['target']
        arrays[f'{ci}_raw_donors'] = case['raw_donors']
        item = {k:case[k] for k in ('subject','task','condition','donor_signature')}
        item.update(case_index=ci, signature='|'.join(case['donor_signature']), candidates=[])
        for candidate in range(9):
            for mode in cfg['response']['selection_modes']:
                for component in (False,True):
                    result = fit(base,case['target'],case['sd'],cal,candidate,mode=mode,component=component)
                    scores.append(dict(subject=case['subject'],case_index=ci,candidate=candidate,
                        component=component,mode=mode,**metrics(result,case,mode)))
            hidden = fit(base,case['target'],case['sd'],cal,candidate,mode='Hb_hidden')
            donors, available, statuses = donor_features(base,case,cal,candidate)
            arrays[f'{ci}_{candidate}_physical'] = hidden.get('physical_prediction',np.full((120,3),np.nan))
            arrays[f'{ci}_{candidate}_donors'] = donors
            item['candidates'].append(dict(candidate=candidate, target_converged=hidden['converged'],
                donor_available=available, target_status=hidden['status'], donor_statuses=statuses))
        cases_meta.append(item)
    path = out/'training'/spec['key']
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'),**arrays)
    return dict(status='completed',spec=spec,rows=scores,cases=cases_meta,array_path=str(path.with_suffix('.npz')))


def synchronized_donors(ref, refs):
    candidates = sorted((r for r in refs if r['key'] == ref['key'] and r['window'] == ref['window']
        and r['site'] != ref['site'] and r['hb_channel'] != ref['hb_channel']
        and r['dataset'] == ref['dataset'] and r['subject'] == ref['subject']
        and abs(r['eeg_start_s']-ref['eeg_start_s']) < 1e-6
        and abs(r['hb_start_s']-ref['hb_start_s']) < 1e-6),key=lambda r:r['site'])
    chosen, seen = [], set()
    for r in candidates:
        if r['hb_channel'] not in seen:
            chosen.append(r)
            seen.add(r['hb_channel'])
    return chosen


def validate_public_ref(ref, parent, datasets):
    path = Path(ref['array_path']).resolve()
    if (ref['dataset'] not in datasets or not path.is_relative_to(Path(parent).resolve()/'prepared')
            or (ref['dataset'] == 'visual_cognitive_motivation' and ref['subject'] == 'S06'
                and 'Part1' in ref['record'])):
        raise ValueError('public parent boundary or excluded identity before array access')
    return path


def measured_plan(cfg, project_root):
    """Read identity/calibration metadata only; no measured array reads here."""
    parent = (Path(project_root)/cfg['source_run']).resolve()
    previous = (Path(project_root)/cfg['attribution_run']).resolve()
    metadata = [parent/'cohort.json',parent/'diagnostic_plan.json',parent/'summary.json',
                previous/'summary.json',previous/'measured_plan.json']
    if any(not p.is_file() for p in metadata):
        raise FileNotFoundError('missing_input: exact parent metadata is unavailable')
    if any(read_json(p)['execution'] != 'completed' for p in (parent/'summary.json',previous/'summary.json')):
        raise ValueError('historical parent must be terminal')
    refs = read_json(parent/'cohort.json')['refs']
    lookup = {r['id']:r for r in refs}
    panel = read_json(parent/'diagnostic_plan.json')['windows']
    if len(panel) != 72 or len({(r['dataset'],r['subject']) for r in panel}) != 18:
        raise ValueError('unchanged 72-window/18-subject development panel required')
    if [r['id'] for r in panel] != [r['id'] for r in read_json(previous/'measured_plan.json')['windows']]:
        raise ValueError('historical paired panel mismatch')
    by_window = {}
    for r in refs:
        by_window.setdefault((r['key'],r['window']),[]).append(r)
    coordinate_cache, groups, files = {}, {}, set(metadata)

    def coordinate(key):
        if key not in coordinate_cache:
            path = parent/'coordinates'/f'{key}.json'
            files.add(path)
            value = read_json(path)
            coordinate_cache[key] = {k:value[k] for k in ('key','pc','eeg_factor','hb_factor','sd','training_subjects')}
        return coordinate_cache[key]

    def case_spec(ref, key, *, evaluation):
        fold = key.rsplit('__outer',1)[1]
        coord = coordinate(key)
        if evaluation and ref['subject'] in coord['training_subjects']:
            raise ValueError('target subject leaked into target coordinate')
        target_path = validate_public_ref(ref,parent,cfg['measured']['datasets'])
        files.add(target_path)
        donors = synchronized_donors(ref,by_window[ref['key'],ref['window']])
        items = []
        for d in donors:
            files.add(validate_public_ref(d,parent,cfg['measured']['datasets']))
            dc = coordinate(f'{d["dataset"]}__{d["site"]}__outer{fold}')
            if evaluation and ref['subject'] in dc['training_subjects']:
                raise ValueError('target subject leaked into a donor coordinate')
            items.append(dict(ref=d,coordinate=dc))
        return dict(ref=ref,coordinate=coord,donors=items,parent=str(parent))

    for ref in panel:
        key = f'{ref["dataset"]}__{ref["site"]}__outer{ref["subject_fold"]}'
        if key not in groups:
            path = previous/'calibration'/f'{key}.json'
            files.add(path)
            old = read_json(path)
            train = [lookup[v] for v in old['training_ids']]
            selection = [lookup[v] for v in old['selection_ids']]
            if (len(train) != cfg['measured']['training_windows'] or len(selection) != cfg['measured']['selection_windows']
                    or {r['subject'] for r in train}&{r['subject'] for r in selection}):
                raise ValueError('historical inner training/selection identity mismatch')
            cal = dict(sd=coordinate(key)['sd'],coefficient_sd=old['direction_scales']['D-fixed'],
                coefficient_multiplier=old['coefficient_multiplier'],driver_sd=old['driver_sd'],
                selected_rho=old['selected_rho'],trained_coefficient_sd=old['direction_scales']['D-trained'],
                parent_spatial_loading=old['spatial_loading']['D-fixed'],parent_spatial_ridge=old['spatial_ridge'],
                parent_training_template=old['training_template'])
            groups[key] = dict(key=key,calibration=cal,training=[case_spec(r,key,evaluation=False) for r in train],
                selection=[case_spec(r,key,evaluation=False) for r in selection],evaluation=[])
            # Inner selection inherits frozen OUTER coordinates, explicitly not
            # a fully nested coordinate fit. Outer evaluation is checked below.
        group = groups[key]
        forbidden = {s['ref']['subject'] for phase in ('training','selection') for s in group[phase]}
        if ref['subject'] in forbidden:
            raise ValueError('evaluation subject entered response/readout training')
        group['evaluation'].append(case_spec(ref,key,evaluation=True))
    profiles = []
    for dataset in cfg['measured']['datasets']:
        group = sorted((r for r in panel if r['dataset']==dataset),key=lambda r:r['id'])
        for subject in sorted({r['subject'] for r in group})[:cfg['profiles']['measured_windows_per_dataset']]:
            profiles.append(next(r['id'] for r in group if r['subject']==subject))
    for identity in profiles:
        files.add(previous/'measured'/identity/'result.json')
        files.add(previous/'measured'/identity/'result.npz')
    return dict(groups=groups,profile_ids=profiles,files=sorted(map(str,files)),
                interpretation='public_development_only_conditional_on_frozen_outer_coordinates')


def load_measured_case(spec, cal):
    from src.data.ssm_prepared import prepared_target

    ref, coord = spec['ref'], spec['coordinate']
    datasets = ['eeg_fnirs_single_trial','simultaneous_eeg_nirs','visual_cognitive_motivation']
    validate_public_ref(ref,spec['parent'],datasets)
    target = prepared_target(ref,coord)
    donors, scales, ratios, names, raw = [], [], [], [], []
    for entry in spec['donors']:
        d, dc = entry['ref'],entry['coordinate']
        if d['id'] == ref['id'] or d['hb_channel'] == ref['hb_channel']:
            raise ValueError('target Hb cannot enter independent donor reads')
        validate_public_ref(d,spec['parent'],datasets)
        value = prepared_target(d,dc)
        ratio = coord['hb_factor']/dc['hb_factor']
        donors.append(value); scales.append(dc['sd']); ratios.append(ratio); names.append(d['site'])
        raw.append(value[:,1:]*ratio)
    return dict(target=target,sd=np.asarray(cal['sd']),donors=donors,donor_sds=scales,
        donor_ratios=ratios,donor_signature=names,raw_donors=np.concatenate(raw,axis=1) if raw else np.empty((120,0)),
        subject=ref['subject'],task=ref['task'],condition=ref['condition'],scenario='measured',
        population=ref['dataset'],ref=ref)


def hash_inputs(plan, out):
    records, missing = [], []
    for name in plan['files']:
        path = Path(name)
        if not path.is_file():
            missing.append(name)
        else:
            records.append(dict(path=name,size=path.stat().st_size,sha256=sha256(path)))
    result = dict(status='missing_input' if missing else 'passed',files=records,missing_input=missing,
        hash_scope='current_local_retained_identity_not_an_invented_historical_seal')
    write_json(Path(out)/'input_identity.json',result)
    if missing:
        raise FileNotFoundError(f'missing_input: {len(missing)} retained inputs; no regeneration permitted')
    return result


def training_cases(results):
    cases = []
    for result in results:
        if result['status'] != 'completed':
            raise ValueError('training task failure prevents selecting on a success subset')
        with np.load(result['array_path'],allow_pickle=False) as data:
            for meta in result['cases']:
                ci = meta['case_index']
                cases.append(dict(meta, key=result['spec']['key'], target=data[f'{ci}_target'].copy(),
                    raw_donors=data[f'{ci}_raw_donors'].copy(),
                    physical=[data[f'{ci}_{i}_physical'].copy() for i in range(9)],
                    residuals=[data[f'{ci}_{i}_donors'].copy() for i in range(9)]))
    return cases


def spatial_fit(cfg, training, selection, scores, selected, cal):
    """Cross-generate training residuals; select readouts on independent people."""
    subjects = sorted({r['subject'] for r in training})
    folds = {s:i%cfg['response']['crossfit_folds'] for i,s in enumerate(subjects)}
    choices, crossfit = {}, []
    for fold in sorted(set(folds.values())):
        excluded = [s for s in subjects if folds[s]==fold]
        choice = response_choice(scores,component=False,exclude_subjects=excluded)
        if set(choice['selection_subjects'])&set(excluded):
            raise ValueError('crossfit response leakage')
        choices[fold] = choice['selected']
        crossfit.append(dict(fold=fold,**choice))
    models, score_rows = {}, []
    projector = operators()[2]
    signatures = sorted({r['signature'] for r in training if r['signature']})
    for regime in ('H0','Hselected'):
        maps = {}
        for signature in signatures:
            usable = []
            for row in training:
                candidate = 0 if regime == 'H0' else choices[folds[row['subject']]]
                state = row['candidates'][candidate]
                if row['signature'] == signature and state['donor_available'] and state['target_converged']:
                    usable.append((row,candidate))
            if len({r['subject'] for r,_ in usable}) < 3:
                continue
            x = np.concatenate([projector@r['residuals'][c] for r,c in usable])
            y = np.concatenate([projector@(r['target'][:,1:]-r['physical'][c][:,1:]) for r,c in usable])
            for rank in cfg['observation']['ranks']:
                for weight in cfg['observation']['ridge_weights']:
                    model = fit_reduced_rank_readout(x,y,rank=rank,weight=weight)
                    key = f'{rank}:{weight:g}'
                    maps.setdefault(key,{})[signature] = dict(model,
                        training_ids=[r['key'] for r,_ in usable],training_subjects=sorted({r['subject'] for r,_ in usable}),
                        crossfit_candidates=[c for _,c in usable],donor_signature=signature)
        candidate = 0 if regime == 'H0' else selected
        for rank in cfg['observation']['ranks']:
            for weight in cfg['observation']['ridge_weights']:
                key = f'{rank}:{weight:g}'
                for r in selection:
                    state = r['candidates'][candidate]
                    model = maps.get(key,{}).get(r['signature'])
                    available = bool(model and state['donor_available'] and state['target_converged'])
                    risk = 100.
                    if available:
                        component = predict_reduced_rank_readout(model,projector@r['residuals'][candidate])
                        risk = rmse((r['physical'][candidate][:,1:]+component-r['target'][:,1:])/np.asarray(cal['sd'])[1:])
                    score_rows.append(dict(regime=regime,rank=rank,weight=weight,subject=r['subject'],
                        key=r['key'],case_index=r['case_index'],available=available,risk=risk,signature=r['signature']))
        frame = pd.DataFrame([r for r in score_rows if r['regime']==regime])
        risks = frame.groupby(['rank','weight','subject']).risk.mean().groupby(['rank','weight']).mean()
        selected_by_rank = {rank:min(cfg['observation']['ridge_weights'],key=lambda w:(risks[rank,w],w))
                            for rank in cfg['observation']['ranks']}
        rank = min(cfg['observation']['ranks'],key=lambda r:(risks[r,selected_by_rank[r]],r))
        models[regime] = dict(selected_rank=rank,by_rank={str(r):maps.get(f'{r}:{w:g}',{})
                                                     for r,w in selected_by_rank.items()},
                             selected_weights=selected_by_rank)
    # Baselines use the same permitted identities. The historical mean/ridge
    # calibration remains the measured S-mean/spatial-ridge reference.
    baseline_maps = {}
    for signature in signatures:
        usable = [r for r in training if r['signature']==signature and r['candidates'][0]['donor_available']
                  and r['candidates'][0]['target_converged']]
        if len(usable) < 3:
            continue
        basis = operators()[1]
        components = []
        for r in usable:
            mean_residual = np.zeros((120,3))
            mean_residual[:,1:] = r['residuals'][0].reshape(120,-1,2).mean(1)
            a = project_component(mean_residual,basis,cal['sd'],np.asarray(cal['coefficient_sd'])*cal['coefficient_multiplier'])
            components.append((basis@a).reshape(120,3)[:,1:])
        x = np.concatenate(components)/np.asarray(cal['sd'])[1:]
        y = np.concatenate([r['target'][:,1:]-r['physical'][0][:,1:] for r in usable])/np.asarray(cal['sd'])[1:]
        loading = float(np.clip(np.sum(x*y)/(np.sum(x*x)+1e-12),0.,2.))
        ridge = fit_standardized_ridge(np.concatenate([r['raw_donors'] for r in usable]),
                                      np.concatenate([r['target'][:,1:] for r in usable]),.1)
        baseline_maps[signature] = dict(loading=loading,ridge=ridge)
    return dict(models=models,crossfit=crossfit,selection_scores=score_rows,baselines=baseline_maps,
                template=np.mean([r['target'] for r in training],axis=0))


def calibration_worker(payload):
    cfg, root, group, training_keys, selection_keys, cal = payload
    out = Path(root)
    train_results = [read_json(out/'training'/f'{k}.json') for k in training_keys]
    select_results = [read_json(out/'training'/f'{k}.json') for k in selection_keys]
    train_rows = [r for result in train_results for r in result['rows']]
    select_rows = [r for result in select_results for r in result['rows']]
    choice = response_choice(select_rows)
    training, selection = training_cases(train_results), training_cases(select_results)
    if {r['subject'] for r in training}&{r['subject'] for r in selection}:
        raise ValueError('training and inner selection subjects overlap')
    spatial = spatial_fit(cfg,training,selection,train_rows,choice['selected'],cal)
    sensitivity = [rmse((r['physical'][0][:,1:]-r['physical'][choice['selected']][:,1:])/np.asarray(cal['sd'])[1:])
                   for r in selection if r['candidates'][0]['target_converged']
                   and r['candidates'][choice['selected']]['target_converged']]
    return dict(status='completed',key=group,response=choice,spatial=spatial,calibration=cal,
                training_keys=training_keys,selection_keys=selection_keys,
                sensitivity_threshold=float(np.median(sensitivity)) if sensitivity else None,
                response_scores=select_rows,training_subjects=sorted({r['subject'] for r in training}),
                selection_subjects=sorted({r['subject'] for r in selection}))


def failed_fit(status):
    return dict(converged=False,status=status)


def fixed_prediction(result, prediction, component=None):
    result = dict(result)
    result['prediction'] = prediction
    if component is not None:
        result['observation_component'] = component
    return result


def spatial_mean(frozen, regime, rank, signature, residual):
    model = frozen['spatial']['models'][regime]['by_rank'][str(rank)].get(signature)
    if model is None:
        return None
    mean = np.zeros((120,3))
    mean[:,1:] = predict_reduced_rank_readout(model,operators()[2]@residual)
    return mean


def old_mean(cal, frozen, signature, residual):
    if not residual.shape[1]:
        return None
    loading = cal.get('parent_spatial_loading')
    if loading is None:
        baseline = frozen['spatial']['baselines'].get(signature)
        if not baseline:
            return None
        loading = baseline['loading']
    mean = np.zeros((120,3))
    mean[:,1:] = residual.reshape(120,-1,2).mean(1)
    a = project_component(mean,operators()[1],cal['sd'],
                          np.asarray(cal['coefficient_sd'])*cal['coefficient_multiplier'])
    return loading*(operators()[1]@a).reshape(120,3)


def ridge_prediction(cal, frozen, signature, raw):
    if not raw.shape[1]:
        return None
    if 'parent_spatial_ridge' in cal:
        return predict_standardized_ridge(cal['parent_spatial_ridge'],raw.reshape(120,-1,2).mean(1))
    baseline = frozen['spatial']['baselines'].get(signature)
    return predict_standardized_ridge(baseline['ridge'],raw) if baseline else None


def wrong_training_case(out, frozen, case):
    candidates = []
    for key in frozen['training_keys']:
        result = read_json(Path(out)/'training'/f'{key}.json')
        for meta in result['cases']:
            if (meta['subject'] != case['subject'] and meta['task'] == case['task']
                    and meta['condition'] == case['condition'] and meta['donor_signature'] == case['donor_signature']):
                candidates.append((key,meta['case_index'],result))
    if not candidates:
        return None
    key, index, result = min(candidates,key=lambda v:(v[0],v[1]))
    return next(r for r in training_cases([result]) if r['case_index']==index)


def evaluate_case(cfg, base, case, frozen, out, modes):
    cal = frozen['calibration']
    selected = frozen['response']['selected']
    signature = '|'.join(case['donor_signature'])
    donors = {i:donor_features(base,case,cal,i) for i in sorted({0,selected})}
    rows, saved, fits = [], {}, {}
    spatial_definitions = {}
    for regime, candidate in (('H0',0),('Hselected',selected)):
        for rank in (1,2):
            spatial_definitions[f'{regime}-Srank{rank}'] = (regime,candidate,rank)
    for mode in modes:
        cached = {}
        for arm,candidate,component in (('B0',0,False),('BC',0,True),
                                        ('Hselected',selected,False),('HselectedC',selected,True)):
            key = (candidate,component)
            if key not in cached:
                cached[key] = fit(base,case['target'],case['sd'],cal,candidate,mode=mode,component=component)
            fits[mode,arm] = cached[key]
        if 'trained_coefficient_sd' in cal:
            other = dict(cal,coefficient_sd=cal['trained_coefficient_sd'])
            fits[mode,'D-trained'] = fit(base,case['target'],case['sd'],other,mode=mode,
                                       component=True,rho=cal['selected_rho'])
        for arm,(regime,candidate,rank) in spatial_definitions.items():
            residual,available,_ = donors[candidate]
            mean = spatial_mean(frozen,regime,rank,signature,residual) if available else None
            if mean is None:
                result = failed_fit('missing_donor_or_training_signature')
            elif mode == 'Hb_hidden':
                physical = cached[candidate,False]
                result = (fixed_prediction(physical,physical['physical_prediction']+mean,mean)
                          if 'physical_prediction' in physical else physical)
            else:
                result = fit(base,case['target'],case['sd'],cal,candidate,mode=mode,mean=mean)
            fits[mode,arm] = result
        for arm,regime in (('F01','H0'),('F11','Hselected')):
            rank = frozen['spatial']['models'][regime]['selected_rank']
            fits[mode,arm] = fits[mode,f'{regime}-Srank{rank}']
        fits[mode,'F00'] = fits[mode,'BC']
        fits[mode,'F10'] = fits[mode,'HselectedC']
        if 'physical_truth' in case:
            fits[mode,'oracle_response'] = fit(base,case['target'],case['sd'],cal,mode=mode,
                truth=case['truth_response'],component=True)
            fits[mode,'oracle_component'] = fit(base,case['target'],case['sd'],cal,mode=mode,
                mean=case['observation_truth'])
            fits[mode,'oracle_both'] = fit(base,case['target'],case['sd'],cal,mode=mode,
                truth=case['truth_response'],mean=case['observation_truth'])
        template = np.asarray(cal.get('parent_training_template',frozen['spatial']['template']))
        fits[mode,'training_template'] = dict(converged=True,status='frozen_template',prediction=template)
        for (fit_mode,arm), result in list(fits.items()):
            if fit_mode != mode:
                continue
            rows.append(dict(arm=arm,mode=mode,pairing='real',donor_count=len(case['donors']),**metrics(result,case,mode)))
            # Primary independent predictions and full truth decompositions are
            # retained; all fit failures remain in the scalar task records.
            if mode in ('full','Hb_hidden','center_Hb') and arm in (
                    'B0','BC','Hselected','HselectedC','F01','F11','oracle_response','oracle_component','oracle_both'):
                for name in ('driver','prediction','physical_prediction','observation_component'):
                    if name in result:
                        saved[f'{mode}__{arm}__{name}'] = result[name]
    # Independent-observation nulls are scored on the fully hidden Hb pair.
    if 'Hb_hidden' in modes:
        wrong = wrong_training_case(out,frozen,case)
        for pairing in ('real','wrong_subject','real_shift_support','nonwrapping_shift_12s'):
            support = np.arange(120)<72 if 'shift' in pairing else np.ones(120,bool)
            available0 = donors[0][1]
            raw = case['raw_donors']
            residual0 = donors[0][0]
            if pairing == 'wrong_subject':
                available0 = bool(wrong and wrong['candidates'][0]['donor_available'])
                raw = wrong['raw_donors'] if wrong else np.empty((120,0))
                residual0 = wrong['residuals'][0] if wrong else np.empty((120,0))
            mean = old_mean(cal,frozen,signature,residual0) if available0 else None
            direct = ridge_prediction(cal,frozen,signature,raw) if available0 else None
            predictions = {'S-mean':(0,mean),'spatial_ridge':(0,direct)}
            if pairing != 'real':
                for arm,regime,candidate in (('F01','H0',0),('F11','Hselected',selected)):
                    rank = frozen['spatial']['models'][regime]['selected_rank']
                    residual,available,_ = donors[candidate]
                    if pairing == 'wrong_subject':
                        available = bool(wrong and wrong['candidates'][candidate]['donor_available'])
                        residual = wrong['residuals'][candidate] if wrong else np.empty((120,0))
                    predictions[arm] = (candidate,spatial_mean(frozen,regime,rank,signature,residual) if available else None)
            for arm,(candidate,predicted) in predictions.items():
                physical = fits['Hb_hidden','B0' if candidate==0 else 'Hselected']
                if predicted is None or 'prediction' not in physical:
                    result = failed_fit('missing_donor_or_matched_null')
                else:
                    predicted = predicted.copy()
                    if pairing == 'nonwrapping_shift_12s':
                        shifted = np.zeros_like(predicted)
                        shifted[:72] = predicted[48:]
                        predicted = shifted
                    if arm == 'spatial_ridge':
                        pred = physical['prediction'].copy(); pred[:,1:] = predicted
                        result = dict(converged=physical['converged'],status=physical['status'],prediction=pred)
                    else:
                        result = fixed_prediction(physical,physical['physical_prediction']+predicted,predicted)
                rows.append(dict(arm=arm,mode='Hb_hidden',pairing=pairing,donor_count=len(case['donors']),
                                 **metrics(result,case,'Hb_hidden',support=support)))
        left,right = fits['Hb_hidden','B0'],fits['Hb_hidden','Hselected']
        sensitivity = (rmse((left['physical_prediction'][:,1:]-right['physical_prediction'][:,1:])/case['sd'][1:])
                       if left['converged'] and right['converged'] else None)
        for row in rows:
            row['sensitivity_index'] = sensitivity
            threshold = frozen['sensitivity_threshold']
            row['low_sensitivity'] = bool(sensitivity is not None and threshold is not None and sensitivity<=threshold)
    return dict(rows=rows,arrays=saved,donor_fits={i:d[2] for i,d in donors.items()})


def evaluation_worker(payload):
    cfg,base,root,spec = payload
    out = Path(root)
    frozen = read_json(out/'calibration'/f'{spec["group"]}.json')
    cal = frozen['calibration']
    cases = ([attach_scales(synthetic_case(cfg,base,spec['population'],'evaluation',spec['repeat'],
                spec['spectrum'],spec['initial'],scenario),cal) for scenario in cfg['synthetic']['scenarios']]
             if spec['kind']=='synthetic' else [load_measured_case(spec['case'],cal)])
    modes = cfg[spec['kind']]['modes']
    rows, arrays, donor_records = [], {}, []
    for ci, case in enumerate(cases):
        result = evaluate_case(cfg,base,case,frozen,out,modes)
        for row in result['rows']:
            rows.append(dict(case_index=ci,scenario=case['scenario'],population=case['population'],
                subject=case['subject'],**row))
        for name,value in result['arrays'].items():
            arrays[f'{ci}__{name}'] = value
        for name in ('target','sd','driver','physical_truth','observation_truth'):
            if name in case:
                arrays[f'{ci}__truth__{name}'] = case[name]
        donor_records.append(dict(case_index=ci,signature=case['donor_signature'],fits=result['donor_fits']))
    path = out/spec['kind']/spec['key']
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'),**arrays)
    return dict(status='completed',spec=spec,rows=rows,donors=donor_records,array_path=str(path.with_suffix('.npz')))


def confounding_profile(cfg, base, case, cal, candidate, *, mode, arm):
    result = fit(base,case['target'],case['sd'],cal,candidate,mode=mode,component=True,jacobian=True)
    if not result['converged']:
        return dict(rows=[dict(arm=arm,mode=mode,converged=False,status=result['status'])],overlap=[],arrays={})
    n, count = 120, 125
    op, basis, _ = operators()
    p = parameters(base,candidate)
    forward = nonlinear_driver_forward(result['driver'],result['initial_state'],p,.25,substeps=4,
                                       derivative=True,return_flow_jacobian=True,numerical_backend='numba')
    j = op@forward['jacobian']
    columns, flow_columns = [], []
    for index in (0,1):
        plus,minus = list(RESPONSE_GRID[candidate]),list(RESPONSE_GRID[candidate])
        plus[index] *= np.exp(1e-4); minus[index] *= np.exp(-1e-4)
        alternatives = [nonlinear_driver_forward(result['driver'],result['initial_state'],parameters(base,truth=v),
                            .25,substeps=4,derivative=False,numerical_backend='numba') for v in (plus,minus)]
        columns.append(op@(alternatives[0]['canonical_prediction']-alternatives[1]['canonical_prediction']).ravel()/2e-4)
        flow_columns.append((alternatives[0]['states'][:,2]-alternatives[1]['states'][:,2])/2e-4)
    j = np.column_stack([j]+columns)
    mask = visible_mask(mode).ravel()
    scale = np.tile(case['sd'],n)
    jdata, bdata = j[mask]/scale[mask,None], basis[mask]/scale[mask,None]
    curvature = np.zeros((n-2,count+2))
    curvature[:,:n] = np.sqrt(base['solver']['penalty'])*np.diff(np.eye(n),n=2,axis=0)/.25**1.5
    initial = np.zeros((5,count+2))
    initial[:,n:count] = np.sqrt(base['solver']['initial_penalty'])*np.diag(np.r_[1.,result['initial_state'][1:]])
    flow = (np.sqrt(base['solver']['flow_prior_weight']*.25)/base['solver']['flow_prior_log_sd']
            /forward['states'][:,2])[:,None]*np.column_stack([forward['flow_jacobian']]+flow_columns)
    cs = np.asarray(cal['coefficient_sd'])*cal['coefficient_multiplier']
    groups = dict(driver=np.arange(n),initial=np.arange(n,count),response=np.arange(count,count+2),all=np.arange(count+2))
    overlap = grouped_component_overlap(jdata,bdata,groups=groups,
        physical_penalty=np.vstack((curvature,initial,flow)),component_penalty=np.diag(1/cs))
    rows, overlap_rows, arrays = [], [], {}
    for group, diagnostic in overlap.items():
        overlap_rows.append(dict(arm=arm,mode=mode,group=group,
            data_cosine=max(diagnostic['data']['cosines'],default=0.),
            objective_cosine=max(diagnostic['objective']['cosines'],default=0.),
            physical_rank=diagnostic['data']['rank'],
            response_tangent='diagnostic_log_tau_log_kappa_not_freed_evaluation_parameters'))
        direction = np.asarray(diagnostic['data']['coefficient_direction'])
        size = rmse((basis@direction).reshape(n,3)[:,1:]/case['sd'][1:])
        if size <= 1e-12:
            continue
        direction /= size
        for offset in cfg['profiles']['offsets_sd']:
            coefficients = result['observation_coefficients']+offset*direction
            component = (basis@coefficients).reshape(n,3)
            alternative = fit(base,case['target'],case['sd'],cal,candidate,mode=mode,mean=component)
            objective = alternative.get('objective',np.nan)+float(np.sum((coefficients/cs)**2))
            fraction = (objective-result['objective'])/max(result['objective'],1e-12)
            row = dict(arm=arm,mode=mode,group=group,offset_sd=offset,objective=objective,
                reference_objective=result['objective'],objective_fraction=fraction,
                near_equivalent=bool(alternative['converged'] and fraction<=cfg['profiles']['near_objective_fraction']),
                **{k:v for k,v in metrics(alternative,case,mode).items() if k!='objective'})
            if 'driver' in alternative:
                delta = np.r_[alternative['driver']-result['driver'],
                    alternative['initial_state'][0]-result['initial_state'][0],
                    np.log(alternative['initial_state'][1:]/result['initial_state'][1:])]
                r_effect = rmse((j[:,:n]@delta[:n]).reshape(n,3)[:,1:]/case['sd'][1:])
                i_effect = rmse((j[:,n:count]@delta[n:]).reshape(n,3)[:,1:]/case['sd'][1:])
                row.update(driver_change_sd=rmse(alternative['driver']-result['driver'])/cal['driver_sd'],
                    physical_change_sd=rmse((alternative['physical_prediction'][:,1:]-result['physical_prediction'][:,1:])/case['sd'][1:]),
                    component_change_sd=rmse((component-result['observation_component'])[:,1:]/case['sd'][1:]),
                    driver_tangent_Hb_change_sd=r_effect,initial_tangent_Hb_change_sd=i_effect,
                    initial_share_of_sum_tangent_RMS=i_effect/max(r_effect+i_effect,1e-12))
                for name in ('driver','prediction','physical_prediction','observation_component'):
                    arrays[f'{arm}__{mode}__{group}__{offset:g}__{name}'] = alternative[name]
            rows.append(row)
    return dict(rows=rows,overlap=overlap_rows,arrays=arrays)


def profile_worker(payload):
    cfg,base,root,spec,baseline_cal = payload
    out = Path(root)
    if spec.get('baseline'):
        cal, selected = baseline_cal, 0
    else:
        frozen = read_json(out/'calibration'/f'{spec["group"]}.json')
        cal, selected = frozen['calibration'],frozen['response']['selected']
    if spec['kind']=='synthetic':
        case = attach_scales(synthetic_case(cfg,base,spec['population'],
            'pilot' if spec.get('baseline') else 'evaluation',spec['repeat'],'mixed','nonrest',spec['scenario']),cal)
    else:
        case = load_measured_case(spec['case'],cal)
    rows,overlap,arrays = [],[],{}
    arms = [('BC',0)] if spec.get('baseline') else [('BC',0),('HselectedC',selected)]
    for arm,candidate in arms:
        for mode in cfg['profiles']['modes']:
            result = confounding_profile(cfg,base,case,cal,candidate,mode=mode,arm=arm)
            rows += result['rows']; overlap += result['overlap']; arrays.update(result['arrays'])
    path = out/'profiles'/spec['key']
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'),**arrays)
    return dict(status='completed',spec=spec,rows=rows,overlap=overlap,
        spatial_profile_boundary='fixed independent means have no target-fitted component direction; stability by constraint is not data identification',
        array_path=str(path.with_suffix('.npz')))


def replay_worker(payload):
    cfg,base,root,project_root,spec,cal = payload
    case = load_measured_case(spec,cal)
    old_path = Path(project_root)/cfg['attribution_run']/'measured'/spec['ref']['id']/'result.npz'
    comparisons = []
    with np.load(old_path,allow_pickle=False) as retained:
        comparisons.append(dict(name='target',max_abs_difference=float(np.max(abs(case['target']-retained['target'])))))
        for arm,component in (('M0',False),('D-fixed',True)):
            for mode in ('full','HbO_hidden','HbR_hidden','center_Hb','Hb_hidden','EEG_hidden'):
                result = fit(base,case['target'],case['sd'],cal,mode=mode,component=component)
                name = f'{mode}__{arm}__prediction'
                if name not in retained or 'prediction' not in result:
                    comparisons.append(dict(name=name,status='missing_input_or_fit'))
                else:
                    comparisons.append(dict(name=name,converged=result['converged'],
                        max_abs_difference=float(np.max(abs(result['prediction']-retained[name])))))
    passed = all(r.get('max_abs_difference',np.inf) <= 1e-8 for r in comparisons)
    return dict(status='completed' if passed else 'failed_replay',identity=spec['ref']['id'],checks=comparisons)


def software_probe(cfg, base, cal):
    case = attach_scales(synthetic_case(cfg,base,'Htau','pilot',0,'mixed','nonrest','common_correlated'),cal)
    mean = .5*case['observation_truth']
    mask = visible_mask('Hb_hidden')
    a = fit(base,case['target'],case['sd'],cal,mode='Hb_hidden',component=True,mean=mean)
    changed = case['target'].copy(); changed[~mask] = 1e100
    b = fit(base,changed,case['sd'],cal,mode='Hb_hidden',component=True,mean=mean)
    hidden_difference = float(np.max(abs(a['prediction']-b['prediction'])))
    unit = np.array([1e3,1e-3,1e-3])
    options = {k:base['solver'][k] for k in ('numerical_backend','substeps','penalty','initial_penalty',
                                           'flow_prior_weight','flow_prior_log_sd','gradient_tolerance')}
    scaled = fit_conditioned_shared_driver(case['target']*unit,parameters(base),.25,
        component_mean=mean*unit,visible=mask,mean_operator=np.tile(unit,120)[:,None]*operators()[0],
        observation_basis=np.tile(unit,120)[:,None]*operators()[1],
        observation_coefficient_sd=np.asarray(cal['coefficient_sd'])*cal['coefficient_multiplier'],
        sd=case['sd']*unit,max_evaluations=base['solver']['max_evaluations_per_trial'],**options)
    unit_difference = float(np.max(abs(a['prediction']-scaled['prediction']/unit)))
    shape_ok = all(c['target'].shape==(120,3) and c['donors'].shape==(2,120,3)
                   for c in [case])
    passed = bool(a['converged'] and b['converged'] and scaled['converged']
                  and hidden_difference==0. and unit_difference<1e-8 and shape_ok)
    return dict(status='passed' if passed else 'failed',hidden_max_abs_difference=hidden_difference,
                unit_change_max_abs_difference=unit_difference,shape_ok=shape_ok,
                converged=[a['converged'],b['converged'],scaled['converged']],measured_arrays_read=0)


def pilot_worker(payload):
    cfg,base,index,cal = payload
    case = attach_scales(synthetic_case(cfg,base,'Htau_kappa','pilot',index,'mixed','nonrest','common_colored'),cal)
    rows = []
    for mode in ('full','Hb_hidden'):
        for component in (False,True):
            start = time.monotonic()
            result = fit(base,case['target'],case['sd'],cal,mode=mode,component=component)
            rows.append(dict(mode=mode,component=component,seconds=time.monotonic()-start,**metrics(result,case,mode)))
    return dict(status='completed',rows=rows)


WORKERS = dict(training=training_worker,calibration=calibration_worker,evaluation=evaluation_worker,
               profile=profile_worker,replay=replay_worker,pilot=pilot_worker)


def execute_task(kind, payload, timeout):
    def expired(signum, frame):
        raise TimeoutError('fixed task time budget exhausted')
    signal.signal(signal.SIGALRM,expired)
    signal.alarm(timeout)
    start = time.monotonic()
    try:
        result = WORKERS[kind](payload)
    except Exception as exc:
        result = dict(status='timeout' if isinstance(exc,TimeoutError) else 'task_failed',
                      error=repr(exc),traceback=traceback.format_exc())
        if kind in ('training','evaluation','profile'):
            result['spec'] = payload[3]
    finally:
        signal.alarm(0)
    result.update(seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    return result


def run_tasks(cfg, out, stage, kind, keyed_payloads, workers, *, folder=None):
    """Bound in-flight tasks; every terminal failure is retained and never retried."""
    out = Path(out)
    folder = folder or stage
    pairs = list(keyed_payloads)
    task_plan = [key for key,_ in pairs]
    if len(set(task_plan)) != len(task_plan):
        raise ValueError('duplicate task identities')
    manifest_path = out/f'{stage}_manifest.json'
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous['tasks'] != task_plan:
            raise ValueError('immutable task plan mismatch')
    state = dict(stage=stage,tasks=task_plan,planned=len(pairs),execution='running',controller_pid=os.getpid(),
                 workers=workers,completed=0,failed=0,started_at=datetime.now(timezone.utc).isoformat())
    write_json(manifest_path,state)
    results,remaining = [],[]
    for key,payload in pairs:
        path = out/folder/f'{key}.json'
        if path.exists():
            results.append(read_json(path))
        else:
            remaining.append((key,payload))
    start = time.monotonic()
    iterator = iter(remaining)
    with ProcessPoolExecutor(max_workers=min(workers,max(1,len(remaining)))) as pool:
        pending = {}
        while True:
            while len(pending) < 2*workers:
                item = next(iterator,None)
                if item is None:
                    break
                key,payload = item
                pending[pool.submit(execute_task,kind,payload,cfg['resources']['task_timeout_s'])] = key
            if not pending:
                break
            done,_ = wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                key = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = dict(status='worker_crash',error=repr(exc))
                write_json(out/folder/f'{key}.json',result)
                results.append(result)
            state.update(completed=len(results),failed=sum(r['status']!='completed' for r in results),
                         elapsed_seconds=time.monotonic()-start,updated_at=datetime.now(timezone.utc).isoformat())
            write_json(manifest_path,state)
            print(json.dumps({k:state[k] for k in ('stage','completed','planned','failed','elapsed_seconds')}),flush=True)
    state.update(completed=len(results),failed=sum(r['status']!='completed' for r in results),
        execution='completed' if all(r['status']=='completed' for r in results) else 'completed_with_task_failures',
        elapsed_seconds=time.monotonic()-start,finished_at=datetime.now(timezone.utc).isoformat())
    write_json(manifest_path,state)
    return results,state


def synthetic_plans(cfg):
    training, groups, evaluation, profiles = [], {}, [], []
    for population in cfg['synthetic']['populations']:
        group = f'synthetic__{population}'
        members = dict(training=[],selection=[])
        for split,n in (('train',cfg['synthetic']['train_repeats']),('selection',cfg['synthetic']['selection_repeats'])):
            for repeat in range(n):
                key = f'{group}__{split}_{repeat:03d}'
                spec = dict(kind='synthetic',key=key,group=group,population=population,split=split,repeat=repeat,
                            spectrum='slow' if repeat%2==0 else 'mixed',initial='rest' if repeat%4<2 else 'nonrest')
                training.append(spec)
                members['training' if split=='train' else 'selection'].append(key)
        groups[group] = members
        for repeat in range(cfg['synthetic']['evaluation_repeats']):
            for spectrum in cfg['synthetic']['spectra']:
                for initial in cfg['synthetic']['initial']:
                    evaluation.append(dict(kind='synthetic',group=group,population=population,repeat=repeat,
                        spectrum=spectrum,initial=initial,key=f'{group}__r{repeat}_{spectrum}_{initial}'))
        for scenario in ('shared_neural_only','common_correlated'):
            profiles.append(dict(kind='synthetic',group=group,population=population,repeat=0,
                                 scenario=scenario,key=f'{group}__{scenario}'))
    return dict(training=training,groups=groups,evaluation=evaluation,profiles=profiles)


def require_stage(results, stage):
    failed = [r.get('error',r['status']) for r in results if r['status']!='completed']
    if failed:
        raise RuntimeError(f'{stage}: {len(failed)} terminal task failures; retained without automatic retry')


def historical_source_audit(cfg, project_root, out):
    snapshot = Path(project_root)/cfg['attribution_run']/'source_snapshot'
    paths = ['src/inference/shared_driver_reconstruction.py','src/inference/shared_driver_rk4.py',
             'src/inference/observation_baselines.py','src/inference/t3a_balloon_robust_ssm.py',
             cfg['source_config']]
    rows = []
    for name in paths:
        previous,current = snapshot/name,CODE_ROOT/name
        row = dict(path=name,retained_exists=previous.is_file(),current_exists=current.is_file())
        if previous.is_file() and current.is_file():
            row.update(retained_sha256=sha256(previous),current_sha256=sha256(current))
            row['equal'] = row['retained_sha256']==row['current_sha256']
        rows.append(row)
    passed = all(r.get('equal',False) for r in rows)
    result = dict(status='passed' if passed else 'missing_input_or_source_mismatch',files=rows)
    write_json(Path(out)/'historical_source_audit.json',result)
    if not passed:
        raise ValueError('historical solver/input identity mismatch; no silent replacement')
    return result


def pilot(cfg, base, out, ceiling):
    cal = synthetic_calibration(cfg,base)
    probe = software_probe(cfg,base,cal)
    write_json(out/'software_probe.json',probe)
    if probe['status']!='passed':
        raise ValueError('synthetic software/physical-domain probe failed')
    benchmarks = []
    for workers in sorted(set([min(8,ceiling),min(24,ceiling),ceiling])):
        stage = f'pilot_w{workers}'
        pairs = [(f'{i:03d}',(cfg,base,i,cal)) for i in range(96)]
        results,state = run_tasks(cfg,out,stage,'pilot',pairs,workers)
        require_stage(results,stage)
        fits = [row for result in results for row in result['rows']]
        benchmarks.append(dict(workers=workers,tasks=len(results),fits=len(fits),
            converged=sum(r['converged'] for r in fits),wall_seconds=state['elapsed_seconds'],
            fits_per_second=len(fits)/max(state['elapsed_seconds'],1e-9),
            max_worker_rss_bytes=max(r['peak_rss_bytes'] for r in results)))
    selected = max(benchmarks,key=lambda r:(r['fits_per_second'],-r['workers']))
    result = dict(status='completed',benchmarks=benchmarks,workers=selected['workers'],
        numerical_threads=1,max_in_flight=2*selected['workers'],
        selection='highest_measured_end_to_end_throughput_among_8_24_ceiling_within_preinspected_cpu_memory_limits',
        measured_arrays_read=0)
    write_json(out/'pilot.json',result)
    return result


def measured_tasks(cfg, base, out, project_root, measured):
    replays,train,selection,evaluation,profiles,baseline = [],[],[],[],[],[]
    for key,group in measured['groups'].items():
        members = dict(training=[],selection=[])
        for phase in ('training','selection'):
            for case in group[phase]:
                identity = key+'__'+phase+'__'+case['ref']['id']
                spec = dict(kind='measured',key=identity,group=key,case=case,split=phase)
                train.append((identity,(cfg,base,str(out),spec,group['calibration'])))
                members[phase].append(identity)
        selection.append((key,(cfg,str(out),key,members['training'],members['selection'],group['calibration'])))
        for case in group['evaluation']:
            spec = dict(kind='measured',key=case['ref']['id'],group=key,case=case)
            evaluation.append((spec['key'],(cfg,base,str(out),spec)))
            if spec['key'] in measured['profile_ids']:
                profiles.append((spec['key'],(cfg,base,str(out),spec,None)))
                replays.append((spec['key'],(cfg,base,str(out),str(project_root),case,group['calibration'])))
                initial = dict(spec,baseline=True,key='baseline_measured__'+spec['key'])
                baseline.append((initial['key'],(cfg,base,str(out),initial,group['calibration'])))
    return replays,train,selection,evaluation,profiles,baseline


def run_experiment(cfg, base, out, project_root, workers):
    probe = read_json(out/'software_probe.json')
    if probe['status'] != 'passed':
        raise ValueError('complete synthetic software probe before any measured read')
    historical_source_audit(cfg,project_root,out)
    measured = measured_plan(cfg,project_root)
    write_json(out/'measured_plan.json',measured)
    hash_inputs(measured,out)
    replays,train,selection,evaluation,profiles,baseline = measured_tasks(cfg,base,out,project_root,measured)
    results,_ = run_tasks(cfg,out,'historical_replay','replay',replays,workers)
    require_stage(results,'historical_replay')
    cal = synthetic_calibration(cfg,base)
    write_json(out/'synthetic_calibration_scales.json',cal)
    plan = synthetic_plans(cfg)
    write_json(out/'synthetic_plan.json',plan)
    baseline_specs = [dict(s,baseline=True,key='baseline__'+s['key']) for s in plan['profiles']]
    baseline += [(s['key'],(cfg,base,str(out),s,cal)) for s in baseline_specs]
    results,_ = run_tasks(cfg,out,'baseline_profiles','profile',baseline,workers,folder='profiles')
    require_stage(results,'baseline_profiles')
    write_json(out/'baseline_profile_summary.json',profile_summary(results))
    results,_ = run_tasks(cfg,out,'synthetic_training','training',
        [(s['key'],(cfg,base,str(out),s,cal)) for s in plan['training']],workers,folder='training')
    require_stage(results,'synthetic_training')
    results,_ = run_tasks(cfg,out,'synthetic_selection','calibration',
        [(key,(cfg,str(out),key,m['training'],m['selection'],cal)) for key,m in plan['groups'].items()],workers,folder='calibration')
    require_stage(results,'synthetic_selection')
    results,_ = run_tasks(cfg,out,'synthetic_evaluation','evaluation',
        [(s['key'],(cfg,base,str(out),s)) for s in plan['evaluation']],workers,folder='synthetic')
    require_stage(results,'synthetic_evaluation')
    summarize_synthetic(cfg,base,out)
    results,_ = run_tasks(cfg,out,'synthetic_profiles','profile',
        [(s['key'],(cfg,base,str(out),s,None)) for s in plan['profiles']],workers,folder='profiles')
    require_stage(results,'synthetic_profiles')
    # Public development only. Fresh confirmation is deliberately not an
    # executable stage of this phase-one contract.
    results,_ = run_tasks(cfg,out,'measured_training','training',train,workers,folder='training')
    require_stage(results,'measured_training')
    results,_ = run_tasks(cfg,out,'measured_selection','calibration',selection,workers,folder='calibration')
    require_stage(results,'measured_selection')
    results,_ = run_tasks(cfg,out,'measured_evaluation','evaluation',evaluation,workers,folder='measured')
    require_stage(results,'measured_evaluation')
    results,_ = run_tasks(cfg,out,'measured_profiles','profile',profiles,workers,folder='profiles')
    require_stage(results,'measured_profiles')
    summarize(cfg,base,out)
    verify(cfg,out)


def profile_summary(results):
    overlap = [r for result in results for r in result.get('overlap',[])]
    profiles = [r for result in results for r in result.get('rows',[])]
    groups = {}
    for name in ('driver','initial','response','all'):
        angles = [r for r in overlap if r['group']==name]
        rows = [r for r in profiles if r.get('group')==name and r.get('offset_sd')!=0]
        groups[name] = dict(diagnostics=len(angles),profiles=len(rows),
            median_data_cosine=float(np.median([r['data_cosine'] for r in angles])) if angles else None,
            median_objective_cosine=float(np.median([r['objective_cosine'] for r in angles])) if angles else None,
            near_equivalent=sum(r.get('near_equivalent',False) for r in rows))
    initial = []
    for result in results:
        angles = {(r['arm'],r['mode'],r['group']):r['data_cosine'] for r in result.get('overlap',[])}
        initial += [r for r in result.get('rows',[]) if r.get('group')=='initial' and r.get('near_equivalent')
            and r.get('component_change_sd',0)>=.25 and r.get('initial_share_of_sum_tangent_RMS',0)>.5
            and angles.get((r['arm'],r['mode'],'initial'),0)>=.95]
    return dict(groups=groups,initial_dominated_near_equivalent_rows=len(initial),
        recommend_S3_small_continuous_synthetic=bool(initial),
        interpretation='local_tangent_RMS_share_is_not_explained_variance; profiles_not_confidence_intervals')


def expected_case_rows(kind, cfg):
    arms = ['B0','BC','Hselected','HselectedC','H0-Srank1','H0-Srank2',
            'Hselected-Srank1','Hselected-Srank2','F00','F10','F01','F11','training_template']
    arms += ['oracle_response','oracle_component','oracle_both'] if kind=='synthetic' else ['D-trained']
    rows = [dict(mode=mode,arm=arm,pairing='real') for mode in cfg[kind]['modes'] for arm in arms]
    rows += [dict(mode='Hb_hidden',arm=arm,pairing=pairing)
             for pairing in ('real','wrong_subject','real_shift_support','nonwrapping_shift_12s')
             for arm in (['S-mean','spatial_ridge'] if pairing=='real' else ['S-mean','spatial_ridge','F01','F11'])]
    return rows


def collect_metrics(cfg, out, kind, specs):
    rows, failures = [], []
    for spec in specs:
        path = Path(out)/kind/f'{spec["key"]}.json'
        result = read_json(path) if path.is_file() else dict(status='missing_task')
        if result['status']=='completed':
            rows += [dict(kind=kind,key=spec['key'],group=spec['group'],**r) for r in result['rows']]
        else:
            failures.append(dict(key=spec['key'],status=result['status']))
            scenarios = cfg['synthetic']['scenarios'] if kind=='synthetic' else ['measured']
            for ci,scenario in enumerate(scenarios):
                subject = (f'evaluation_{spec["repeat"]}_{spec["spectrum"]}_{spec["initial"]}' if kind=='synthetic'
                           else spec['case']['ref']['subject'])
                population = spec['population'] if kind=='synthetic' else spec['case']['ref']['dataset']
                for cell in expected_case_rows(kind,cfg):
                    rows.append(dict(kind=kind,key=spec['key'],group=spec['group'],case_index=ci,subject=subject,
                        population=population,scenario=scenario,converged=False,status=result['status'],**cell))
    return pd.DataFrame(rows),failures


def paired_summary(cfg, frame):
    metrics_names = ['score_nrmse','Hb_nrmse','HbO_nrmse','HbR_nrmse',
                     'physical_error','driver_shape_error','driver_error','component_error']
    pairs = [('B0','Hselected'),('BC','HselectedC'),('BC','F01'),('HselectedC','F11'),('F00','F11'),
             ('B0','F01'),('Hselected','F11'),('BC','oracle_response'),('BC','oracle_component'),('BC','oracle_both')]
    index = ['kind','key','case_index','subject','population','scenario','mode','pairing']
    rows = []
    for baseline,candidate in pairs:
        a = frame[(frame.arm==baseline)&(frame.pairing=='real')]
        b = frame[(frame.arm==candidate)&(frame.pairing=='real')]
        merged = a.merge(b,on=index,suffixes=('_baseline','_candidate'),validate='one_to_one')
        for key,group in merged.groupby(['kind','population','scenario','mode'],sort=True):
            for metric in metrics_names:
                names = [metric+'_baseline',metric+'_candidate']
                if any(n not in group for n in names) or not group[names].notna().any().any():
                    continue
                success_a = group.converged_baseline & group[names[0]].notna()
                success_b = group.converged_candidate & group[names[1]].notna()
                common = success_a & success_b
                risk_a = group[names[0]].where(success_a,cfg['gates']['numerical_failure_penalty'])
                risk_b = group[names[1]].where(success_b,cfg['gates']['numerical_failure_penalty'])
                units = pd.DataFrame(dict(subject=group.subject,a=risk_a,b=risk_b)).groupby('subject')[['a','b']].mean()
                delta = (units.a-units.b).to_numpy()
                rng = np.random.default_rng(cfg['seed'])
                means = delta[rng.integers(0,len(delta),(cfg['gates']['bootstrap_repeats'],len(delta)))].mean(1)
                low,high = np.quantile(means,[.025,.975])
                success_gain = (group.loc[common,names[0]]-group.loc[common,names[1]])
                success_gain = success_gain.groupby(group.loc[common,'subject']).mean()
                rows.append(dict(zip(('kind','population','scenario','mode'),key),baseline=baseline,candidate=candidate,
                    metric=metric,planned=len(group),paired_success=int(common.sum()),independent_units=len(units),
                    baseline_failures=int((~success_a).sum()),candidate_failures=int((~success_b).sum()),
                    baseline_risk=float(units.a.mean()),candidate_risk=float(units.b.mean()),
                    gain=float(delta.mean()),relative_gain=float(delta.mean()/max(units.a.mean(),1e-12)),
                    ci_low=float(low),ci_high=float(high),common_success_gain=float(success_gain.mean()) if len(success_gain) else None,
                    interval='subject_block_percentile_95_frozen_fits_failure_penalty_100_no_multiplicity_claim'))
    return pd.DataFrame(rows)


def null_summary(cfg, frame):
    rows = []
    for arm in ('F01','F11','S-mean','spatial_ridge'):
        for real,null in (('real','wrong_subject'),('real_shift_support','nonwrapping_shift_12s')):
            a = frame[(frame.arm==arm)&(frame['mode']=='Hb_hidden')&(frame.pairing==real)]
            b = frame[(frame.arm==arm)&(frame['mode']=='Hb_hidden')&(frame.pairing==null)]
            merged = a.merge(b,on=['kind','key','case_index','subject','population','scenario'],suffixes=('_real','_null'),validate='one_to_one')
            for key,g in merged.groupby(['kind','population','scenario']):
                good = g.converged_real & g.converged_null & g.Hb_nrmse_real.notna() & g.Hb_nrmse_null.notna()
                diff = (g.loc[good,'Hb_nrmse_null']-g.loc[good,'Hb_nrmse_real']).groupby(g.loc[good,'subject']).mean().to_numpy()
                if len(diff):
                    rng = np.random.default_rng(cfg['seed'])
                    interval = np.quantile(diff[rng.integers(len(diff),size=(1000,len(diff)))].mean(1),[.025,.975])
                else:
                    interval = [np.nan,np.nan]
                rows.append(dict(zip(('kind','population','scenario'),key),arm=arm,null=null,planned=len(g),
                    paired_success=int(good.sum()),independent_units=len(diff),
                    gain=float(diff.mean()) if len(diff) else None,ci_low=interval[0],ci_high=interval[1],
                    interpretation='common_success_null_minus_real_same_support; missing_nulls_preserved'))
    return pd.DataFrame(rows)


def attribution_table(cfg, out, specs):
    loaded,success = {},{}
    for spec in specs:
        path = Path(out)/'synthetic'/f'{spec["key"]}.json'
        result = read_json(path)
        if result['status']!='completed':
            continue
        identity = (spec['population'],spec['repeat'],spec['spectrum'],spec['initial'])
        with np.load(result['array_path'],allow_pickle=False) as data:
            loaded[identity] = {k:data[k].copy() for k in data.files}
        success[identity] = {(r['case_index'],r['arm']):r['converged'] for r in result['rows']
                             if r['mode']=='full' and r['pairing']=='real'}
    rows = []
    arms = ['B0','BC','Hselected','HselectedC','F01','F11','oracle_response','oracle_component','oracle_both']
    for identity,data in loaded.items():
        for ci,scenario in enumerate(cfg['synthetic']['scenarios']):
            comparisons = []
            if ci:
                comparisons.append(('scenario_change',identity,0))
            if identity[0]!='H0':
                comparisons.append(('response_change',('H0',*identity[1:]),ci))
            for intervention,control_id,cj in comparisons:
                if control_id not in loaded:
                    continue
                control = loaded[control_id]
                sd = data[f'{ci}__truth__sd'][1:]
                true_physical = data[f'{ci}__truth__physical_truth'][:,1:]-control[f'{cj}__truth__physical_truth'][:,1:]
                true_component = data[f'{ci}__truth__observation_truth'][:,1:]-control[f'{cj}__truth__observation_truth'][:,1:]
                observed = data[f'{ci}__truth__target'][:,1:]-control[f'{cj}__truth__target'][:,1:]
                for arm in arms:
                    prefix,cp = f'{ci}__full__{arm}__',f'{cj}__full__{arm}__'
                    converged = success[identity].get((ci,arm),False) and success[control_id].get((cj,arm),False)
                    row = dict(population=identity[0],repeat=identity[1],spectrum=identity[2],initial=identity[3],
                        scenario=scenario,intervention=intervention,arm=arm,converged=converged)
                    if prefix+'physical_prediction' in data and cp+'physical_prediction' in control:
                        physical = data[prefix+'physical_prediction'][:,1:]-control[cp+'physical_prediction'][:,1:]
                        component = data[prefix+'observation_component'][:,1:]-control[cp+'observation_component'][:,1:]
                        residual = observed-physical-component
                        row.update(physical_change_error=rmse((physical-true_physical)/sd),
                            component_change_error=rmse((component-true_component)/sd),
                            true_physical_change_sd=rmse(true_physical/sd),true_component_change_sd=rmse(true_component/sd))
                        for truth_name,truth in (('physiology',true_physical),('structure',true_component),('observed',observed)):
                            denominator = float(np.sum((truth/sd)**2))
                            for output_name,output in (('physical',physical),('component',component),('residual',residual)):
                                row[f'{truth_name}_to_{output}_projection'] = (float(np.sum((output/sd)*(truth/sd))/denominator)
                                                                            if denominator>1e-12 else None)
                    rows.append(row)
    return pd.DataFrame(rows)


def risk_table(cfg, frame):
    rows = []
    subset = frame[(frame.kind=='synthetic')&(frame['mode']=='Hb_hidden')&(frame.pairing=='real')
                   &frame.arm.isin(['B0','BC','Hselected','HselectedC','F01','F11'])]
    for key,g in subset.groupby(['population','arm']):
        score = g.sensitivity_index.fillna(np.inf)
        errors = g[['physical_error','driver_shape_error','component_error']].max(axis=1)
        errors = errors.where(g.converged & errors.notna(),np.inf)
        for coverage in cfg['risk']['coverages']:
            finite = score[np.isfinite(score)]
            threshold = float(np.quantile(finite,coverage)) if len(finite) else -np.inf
            chosen = score<=threshold
            low = g.low_sensitivity.fillna(False)
            rows.append(dict(population=key[0],arm=key[1],requested_coverage=coverage,planned=len(g),
                actual_coverage=float(chosen.mean()),selected_units=int(g.loc[chosen,'subject'].nunique()),
                mean_risk=float(errors[chosen].mean()) if chosen.any() else None,
                false_reassurance_fraction=float((errors[low]>cfg['risk']['error_threshold_sd']).mean()) if low.any() else None,
                frozen_low_sensitivity_coverage=float(low.mean()),ties='all_ties_retained',
                interpretation='empirical_synthetic_error_not_calibrated_posterior'))
    return pd.DataFrame(rows)


def calibration_tables(cfg, base, out, keys):
    responses,spatial,curves = [],[],[]
    for key in keys:
        frozen = read_json(Path(out)/'calibration'/f'{key}.json')
        for family,index in frozen['response']['families'].items():
            tau,kappa = RESPONSE_GRID[index]
            responses.append(dict(group=key,family=family,selected_candidate=index,tau_s=tau,kappa=kappa,
                score=frozen['response']['scores'][str(index)],
                selected_for_evaluation=index==frozen['response']['selected'],
                training_subjects=json.dumps(frozen['training_subjects']),selection_subjects=json.dumps(frozen['selection_subjects'])))
        for regime,m in frozen['spatial']['models'].items():
            for rank,maps in m['by_rank'].items():
                for signature,model in maps.items():
                    spatial.append(dict(group=key,regime=regime,rank=int(rank),weight=model['weight'],
                        selected=int(rank)==m['selected_rank'],donor_signature=signature,
                        training_ids=json.dumps(model['training_ids']),training_subjects=json.dumps(model['training_subjects']),
                        crossfit_candidates=json.dumps(model['crossfit_candidates']),target_local_free_coefficients=0))
    drive = np.zeros(120); drive[20:24] = .025
    reference = None
    for index,(tau,kappa) in enumerate(RESPONSE_GRID):
        f = nonlinear_driver_forward(drive,[0.,1.,1.,1.,1.],parameters(base,index),.25,
                                     substeps=8,derivative=False,numerical_backend='numba')
        hb = f['canonical_prediction'][:,1:]
        if reference is None:
            reference=hb
        for t in range(120):
            curves.append(dict(candidate=index,tau_s=tau,kappa=kappa,time_s=t*.25,
                HbO=hb[t,0],HbR=hb[t,1],HbO_peak_delay_s=np.argmax(hb[:,0])*.25-5.,
                HbR_trough_delay_s=np.argmin(hb[:,1])*.25-5.,response_RMS_difference_H0=rmse(hb-reference)))
    pd.DataFrame(responses).to_csv(Path(out)/'response_selection.csv',index=False)
    pd.DataFrame(spatial).to_csv(Path(out)/'spatial_readout.csv',index=False)
    pd.DataFrame(curves).to_csv(Path(out)/'response_curves.csv',index=False)


def summarize_synthetic(cfg, base, out):
    plan = read_json(out/'synthetic_plan.json')
    frame,failures = collect_metrics(cfg,out,'synthetic',plan['evaluation'])
    frame.to_csv(out/'synthetic_metrics.csv',index=False)
    paired_summary(cfg,frame).to_csv(out/'synthetic_paired.csv',index=False)
    null_summary(cfg,frame).to_csv(out/'synthetic_nulls.csv',index=False)
    attribution_table(cfg,out,plan['evaluation']).to_csv(out/'mechanism_attribution.csv',index=False)
    risk_table(cfg,frame).to_csv(out/'risk_coverage.csv',index=False)
    calibration_tables(cfg,base,out,plan['groups'])
    result = dict(execution='completed' if not failures else 'completed_with_task_failures',
        tasks=len(plan['evaluation']),rows=len(frame),converged=int(frame.converged.sum()),
        nonconverged_or_unavailable=int((~frame.converged).sum()),task_failures=failures,
        cells_include_aliases='F00=BC,F10=HselectedC,F01/F11=training_selected_rank; these are not additional fits',
        truth='synthetic_generator_states_and_noiseless_components',
        finished_at=datetime.now(timezone.utc).isoformat())
    write_json(out/'synthetic_summary.json',result)
    return frame


def make_decision(cfg, out, paired, nulls, profiles):
    synthetic = paired[paired.kind=='synthetic']
    measured = paired[paired.kind=='measured']
    margin = cfg['gates']['synthetic_error_increase_max_sd']
    control_scenarios = ['shared_neural_only','target_neural_only','donor_contamination']
    controls = synthetic[(synthetic['mode']=='full')&synthetic.scenario.isin(control_scenarios)
                         &synthetic.metric.isin(['physical_error','driver_shape_error'])]
    s1_controls = controls[(controls.population=='H0')&(controls.baseline=='BC')&(controls.candidate=='HselectedC')]
    s2_controls = controls[(controls.baseline=='Hselected')&(controls.candidate=='F11')]
    response = synthetic[(synthetic['mode']=='full')&(synthetic.population!='H0')
        &(synthetic.baseline=='BC')&(synthetic.candidate=='HselectedC')&(synthetic.metric=='physical_error')]
    attribution = pd.read_csv(out/'mechanism_attribution.csv')
    mechanism = attribution[(attribution.intervention=='response_change')&attribution.converged
                            &attribution.arm.isin(['BC','HselectedC'])]
    leakage = mechanism.groupby('arm').physiology_to_component_projection.apply(lambda s:float(s.abs().median())).to_dict()
    mechanism_checks = dict(
        response_physical_mean_gain=float(response.gain.mean()) if len(response) else None,
        response_controls_pass=bool(len(s1_controls) and (s1_controls.gain>=-margin).all()),
        spatial_shared_neural_controls_pass=bool(len(s2_controls) and (s2_controls.gain>=-margin).all()),
        response_change_absolute_component_projection=leakage,
        response_attribution_improved=bool('BC' in leakage and 'HselectedC' in leakage and leakage['HselectedC']<=leakage['BC']))
    decisions = []
    for dataset in cfg['measured']['datasets']:
        for stage,baseline,candidate in (('S1','BC','HselectedC'),('S2','BC','F01'),('F11','F00','F11')):
            g = measured[(measured.population==dataset)&(measured.baseline==baseline)&(measured.candidate==candidate)
                         &(measured.metric=='score_nrmse')&measured['mode'].isin(['HbO_hidden','HbR_hidden','center_Hb','Hb_hidden'])]
            independent = g[g['mode']=='Hb_hidden']
            no_single_harm = bool(len(g)==4 and (g.gain>=0).all())
            predicted = bool(len(independent) and independent.iloc[0].relative_gain>=cfg['gates']['hidden_relative_improvement_min']
                             and independent.iloc[0].ci_low>0)
            failures = bool(len(g) and ((g.candidate_failures-g.baseline_failures)/g.planned<=cfg['gates']['failure_rate_increase_max']).all())
            donor = nulls[(nulls.kind=='measured')&(nulls.population==dataset)&(nulls.arm==('F01' if stage=='S2' else 'F11'))]
            specificity = bool(len(donor)==2 and (donor.ci_low>0).all() and (donor.paired_success==donor.planned).all())
            mechanism_pass = (mechanism_checks['response_controls_pass'] and mechanism_checks['response_attribution_improved']
                              if stage=='S1' else mechanism_checks['spatial_shared_neural_controls_pass'])
            supported = predicted and no_single_harm and failures and mechanism_pass and (stage=='S1' or specificity)
            decisions.append(dict(dataset=dataset,stage=stage,baseline=baseline,candidate=candidate,
                disposition='promising_development_candidate' if supported else 'not_supported_for_expansion',
                hidden_pair_gain_5pct_and_interval=predicted,no_hidden_component_worsening=no_single_harm,
                numerical_failure_margin=failures,synthetic_mechanism=bool(mechanism_pass),
                donor_specificity=specificity if stage!='S1' else None))
    return dict(experiment_id=cfg['experiment_id'],scope=cfg['scope'],mechanism_checks=mechanism_checks,decisions=decisions,
        S3=dict(execution='not_launched',triggered=profiles['recommend_S3_small_continuous_synthetic'],
            reason='first require a continuous synthetic implementation and verified native continuous support; prepared windows are not concatenated'),
        S2_soft_correction=dict(execution='not_launched',reason='fixed independent mean is evaluated first; no automatic capacity expansion'),
        S5=dict(execution='development_2x2_only',fresh_confirmation='not_launched',
            reason='no demonstrably untouched confirmation identity inventory; parent cohort has participated in earlier research'),
        source_status='unresolved',teacher_qualification=False,posterior_claim=False,
        failures='all planned cells retained; success-subset intervals do not erase missing donors or numerical failures')


def summarize(cfg, base, out):
    synthetic = summarize_synthetic(cfg,base,out)
    plan = read_json(out/'measured_plan.json')
    specs = [dict(kind='measured',key=c['ref']['id'],group=key,case=c)
             for key,g in plan['groups'].items() for c in g['evaluation']]
    measured,failures = collect_metrics(cfg,out,'measured',specs)
    measured.to_csv(out/'measured_metrics.csv',index=False)
    frame = pd.concat([synthetic,measured],ignore_index=True)
    paired = paired_summary(cfg,frame)
    paired.to_csv(out/'paired_hidden_metrics.csv',index=False)
    nulls = null_summary(cfg,frame)
    nulls.to_csv(out/'spatial_nulls.csv',index=False)
    groups = ['kind','population','scenario','mode','pairing','arm','donor_count']
    summary_rows = []
    for key,g in frame.groupby(groups,dropna=False):
        row = dict(zip(groups,key),planned=len(g),converged=int(g.converged.sum()),
                   failures=int((~g.converged).sum()),independent_units=g.subject.nunique())
        for metric in ('Hb_nrmse','HbO_nrmse','HbR_nrmse','physical_error','driver_error','driver_shape_error','component_error'):
            values = g.loc[g.converged,metric].dropna() if metric in g else pd.Series(dtype=float)
            row[metric+'_median'] = values.median() if len(values) else None
            row[metric+'_p90'] = values.quantile(.9) if len(values) else None
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out/'metric_summary.csv',index=False)
    synthetic_plan = read_json(out/'synthetic_plan.json')
    profile_keys = [s['key'] for s in synthetic_plan['profiles']]+plan['profile_ids']
    profile_results = [read_json(out/'profiles'/f'{key}.json') for key in profile_keys]
    pd.DataFrame([dict(key=k,**r) for k,result in zip(profile_keys,profile_results) for r in result['rows']]).to_csv(out/'confounding_profiles.csv',index=False)
    pd.DataFrame([dict(key=k,**r) for k,result in zip(profile_keys,profile_results) for r in result['overlap']]).to_csv(out/'jacobian_overlap.csv',index=False)
    profile_result = profile_summary(profile_results)
    write_json(out/'profile_summary.json',profile_result)
    calibration_tables(cfg,base,out,list(synthetic_plan['groups'])+list(plan['groups']))
    decision = make_decision(cfg,out,paired,nulls,profile_result)
    write_json(out/'decision.json',decision)
    summary = dict(schema='ssm_strengthening_summary_v1',experiment_id=cfg['experiment_id'],
        execution='completed' if not failures else 'completed_with_task_failures',
        synthetic=read_json(out/'synthetic_summary.json'),measured=dict(windows=len(specs),subjects=18,
            rows=len(measured),converged=int(measured.converged.sum()),task_failures=failures),
        profiles=profile_result,endpoint_files=['response_selection.csv','spatial_readout.csv','paired_hidden_metrics.csv',
            'mechanism_attribution.csv','confounding_profiles.csv','risk_coverage.csv','decision.json'],
        cell_count_warning='factorial aliases retained as comparison labels, not additional independent fits',
        claim_boundary='public_development_frozen_outer_coordinates; no fresh confirmation, unique source or calibrated posterior',
        finished_at=datetime.now(timezone.utc).isoformat())
    write_json(out/'summary.json',summary)
    return summary


def verify(cfg, out):
    checks = []
    synthetic_plan = read_json(out/'synthetic_plan.json')
    measured = read_json(out/'measured_plan.json')
    plans = [('synthetic',synthetic_plan['evaluation']),('measured',[
        dict(key=c['ref']['id']) for g in measured['groups'].values() for c in g['evaluation']])]
    for kind,specs in plans:
        for spec in specs:
            result = read_json(out/kind/f'{spec["key"]}.json')
            count = len(expected_case_rows(kind,cfg))*(len(cfg['synthetic']['scenarios']) if kind=='synthetic' else 1)
            cells = {(r['case_index'],r['mode'],r['arm'],r['pairing']) for r in result.get('rows',[])}
            checks.append(dict(name=f'{kind}/{spec["key"]}/denominator',passed=result['status']=='completed'
                               and len(result['rows'])==count and len(cells)==count,expected=count,actual=len(cells)))
            if result['status']!='completed':
                continue
            with np.load(result['array_path'],allow_pickle=False) as arrays:
                for name in arrays.files:
                    if name.endswith('__physical_prediction'):
                        prefix = name[:-len('physical_prediction')]
                        error = float(np.max(abs(arrays[prefix+'prediction']-arrays[name]-arrays[prefix+'observation_component'])))
                        checks.append(dict(name=f'{kind}/{spec["key"]}/{prefix}decomposition',passed=error<1e-10,max_abs_error=error))
    for key in list(synthetic_plan['groups'])+list(measured['groups']):
        cal = read_json(out/'calibration'/f'{key}.json')
        separated = not set(cal['training_subjects'])&set(cal['selection_subjects'])
        crossfit = all(not set(r['selection_subjects'])&set(r['excluded_subjects']) for r in cal['spatial']['crossfit'])
        if key in measured['groups']:
            evaluation = {r['ref']['subject'] for r in measured['groups'][key]['evaluation']}
            separated &= not evaluation&(set(cal['training_subjects'])|set(cal['selection_subjects']))
        checks.append(dict(name=f'calibration/{key}/subject_and_crossfit_separation',passed=bool(separated and crossfit)))
    for name in ('software_probe.json','historical_source_audit.json','input_identity.json'):
        checks.append(dict(name=name,passed=read_json(out/name)['status']=='passed'))
    identities = read_json(out/'input_identity.json')
    for record in identities['files']:
        path=Path(record['path'])
        checks.append(dict(name='input_unchanged/'+record['path'],passed=path.is_file() and sha256(path)==record['sha256']))
    failures = [r for r in checks if not r['passed']]
    result = dict(schema='ssm_strengthening_verification_v1',status='passed' if not failures else 'failed',
                  checks=checks,failures=failures,checked_at=datetime.now(timezone.utc).isoformat())
    write_json(out/'verification.json',result)
    if failures:
        raise ValueError(f'{len(failures)} final verification failures')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--project-root',type=Path,default=CODE_ROOT)
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--workers',type=int,default=48)
    parser.add_argument('--stage',choices=['pilot','run','summarize','verify'],default='run')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    cfg=read_config(args.config)
    base=yaml.safe_load((CODE_ROOT/cfg['source_config']).read_text())
    if args.check_only:
        plan=synthetic_plans(cfg)
        print(json.dumps(dict(status='contract_valid',experiment_id=cfg['experiment_id'],measured_arrays_read=0,
            synthetic_training_tasks=len(plan['training']),synthetic_evaluation_tasks=len(plan['evaluation']),
            synthetic_planned_rows=len(plan['evaluation'])*len(SCENARIOS)*len(expected_case_rows('synthetic',cfg)))))
        return
    if args.run_dir is None or not 1<=args.workers<=cfg['resources']['max_workers']:
        raise ValueError('explicit run directory and bounded workers required')
    out=args.run_dir.resolve()
    if out.parent!=(args.project_root/cfg['artifact_root']).resolve():
        raise ValueError('run must be a direct child of the registered artifact root')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resolved=out/'resolved_config.yaml'
    if resolved.exists() and yaml.safe_load(resolved.read_text())!=cfg:
        raise ValueError('immutable run contract mismatch')
    if not resolved.exists():
        resolved.write_text(yaml.safe_dump(cfg,sort_keys=False))
    stage=args.stage
    state=dict(experiment_id=cfg['experiment_id'],stage=stage,execution='running',controller_pid=os.getpid(),
               command=sys.argv,cwd=str(Path.cwd()),source_root=str(CODE_ROOT),workers=args.workers,
               started_at=datetime.now(timezone.utc).isoformat())
    write_json(out/'run_manifest.json',state)
    try:
        if stage=='pilot':
            pilot(cfg,base,out,args.workers)
        elif stage=='run':
            p=read_json(out/'pilot.json')
            workers=min(args.workers,p['workers'])
            run_experiment(cfg,base,out,args.project_root,workers)
        elif stage=='summarize':
            summarize(cfg,base,out)
        else:
            verify(cfg,out)
        state.update(execution='completed',finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        state.update(execution='failed',error=repr(exc),traceback=traceback.format_exc(),
                     finished_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        write_json(out/'run_manifest.json',state)


if __name__=='__main__':
    main()
