#!/usr/bin/env python3
"""Versioned regional spectral-mode diagnostic on exact public parent features.

The parent producer remains the sole measured-feature owner. This experiment
fits fixed log-power readouts and the existing Balloon core, with no raw reader,
task-label fitting, protected access, tokenizer training, or causal claim.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

for _thread_name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_name] = '1'
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
import numpy as np
import pandas as pd
import yaml

from src.inference.observation_baselines import native_feature_operators
from src.inference.shared_driver_attribution import equal_capacity_hb_basis, fit_standardized_ridge, predict_standardized_ridge
from src.inference.shared_driver_modes import fit_shared_driver_modes, forward_shared_driver_modes, spectral_loadings, temporal_mode_basis
from src.inference.t3a_balloon_robust_ssm import BalloonFixedParameters, BalloonFreeParameters, BalloonParameters

DEFAULT_CONFIG = CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_modes_v1.yaml'
ARMS = ('R0_PCA_refit', 'R1_broadband', 'R2_zero', 'R2_selected', 'independent')
MODES = ('full', 'center_EEG', 'center_Hb', 'HbO_hidden', 'HbR_hidden', 'Hb_hidden', 'EEG_channels_hidden')
DATASETS = ('eeg_fnirs_single_trial', 'simultaneous_eeg_nirs', 'visual_cognitive_motivation')
SOURCE_RUN = 'experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1'
SPLIT_RUN = 'experiments/runs/physiology_semantic_tokenizer/shared_driver_teacher_robustness/20261001_v1'


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


def read_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg.get('schema') != 'shared_driver_modes_v1' or cfg.get('source_run') != SOURCE_RUN
            or cfg.get('split_run') != SPLIT_RUN or cfg.get('protected_data') != 'forbidden'
            or cfg.get('data_boundary') != 'exact_public_parent_prepared_regions_and_frozen_training_selection_identities'):
        raise ValueError('wrong versioned source or public data boundary')
    t, m, c = cfg['tensor'], cfg['model'], cfg['calibration']
    if (t['steps'] != 120 or t['dt_s'] != .25 or t['eeg_channels'] != 6 or t['eeg_features'] != 30
            or t['eeg_bands_hz'] != [[1,4],[4,8],[8,13],[13,30],[30,45]]
            or t['feature_order'] != 'channel_major_band_minor' or t['reference_steps'] != 20
            or t['Hb_components'] != ['HbO','HbR'] or t['temporal_modes'] != 24
            or tuple(m['arms']) != ARMS or tuple(cfg['measured']['modes']) != MODES
            or tuple(cfg['data']['datasets']) != DATASETS
            or cfg['data']['panel_windows'] != 72 or cfg['data']['panel_subjects'] != 18
            or m['gamma_candidates'] != [-1.,-.5,0.,.5,1.] or m['common_Hb_modes'] != 4
            or m['common_Hb_rho'] != .35 or m['parameter_policy'] != 'H0_fixed_parent_Balloon'
            or m['vascular_projection'] != 'equal_norm_a_plus_gamma_b_over_sqrt_one_plus_gamma_squared'
            or c['training_windows'] != 18 or c['selection_windows'] != 6
            or cfg['measured']['hidden_EEG_channels'] != [0,2,4]
            or cfg['measured']['shift_steps'] != 48 or cfg['measured']['center_steps'] != 16
            or cfg['resources']['numerical_threads'] != 1 or not 1 <= cfg['resources']['max_workers'] <= 48):
        raise ValueError('tensor, loading, split, mask or resource contract mismatch')
    if m['state_sd'] != .025 or m['ar_phi'] != .95 or m['ar_weight'] != .1 or m['initial_weight'] != 100.:
        raise ValueError('frozen dynamical prior changed')
    if cfg['solver']['max_nfev'] < 1 or cfg['solver']['substeps'] < 1:
        raise ValueError('positive solver budgets required')
    return cfg


def parameters(cfg, project_root=CODE_ROOT):
    source = yaml.safe_load((Path(project_root)/cfg['source_config']).read_text())
    fixed = {k:v for k,v in source['fixed'].items() if k not in ('tau','kappa')}
    p = BalloonParameters(fixed=BalloonFixedParameters(**fixed),
        free=BalloonFreeParameters(tau=source['fixed']['tau'], kappa=source['fixed']['kappa']))
    p.validate()
    return p


@lru_cache(maxsize=2)
def operators(steps=120):
    native = native_feature_operators(steps)
    eeg, hb = native['eeg'], native['fnirs']@native['native_interpolation']
    block = np.zeros((3*steps,3*steps))
    block[0::3,0::3] = eeg
    block[1::3,1::3] = hb
    block[2::3,2::3] = hb
    common = equal_capacity_hb_basis(block, modes=4, rho=.35).reshape(steps,3,4)[:,1:]
    return dict(eeg=eeg, hb=hb, common=common, temporal=temporal_mode_basis(steps,24))


def visible_mask(mode, cfg):
    t, channels = cfg['tensor']['steps'], cfg['tensor']['eeg_features']
    mask = np.ones((t,channels+2), dtype=bool)
    center = cfg['measured']['center_steps']
    left, right = (t-center)//2, (t+center)//2
    if mode == 'full':
        pass
    elif mode == 'center_EEG':
        mask[left:right,:channels] = False
    elif mode == 'center_Hb':
        mask[left:right,channels:] = False
    elif mode == 'HbO_hidden':
        mask[:,channels] = False
    elif mode == 'HbR_hidden':
        mask[:,channels+1] = False
    elif mode == 'Hb_hidden':
        mask[:,channels:] = False
    elif mode == 'EEG_channels_hidden':
        for channel in cfg['measured']['hidden_EEG_channels']:
            mask[:,channel*5:(channel+1)*5] = False
    else:
        raise ValueError('unregistered information mask')
    return mask


def endpoint_mask(mode, mask, support=None):
    endpoint = np.ones_like(mask) if mode == 'full' else ~mask
    if support is not None:
        endpoint &= np.asarray(support, bool)[:,None]
    return endpoint


def validate_ref(ref, parent):
    if ref['dataset'] not in DATASETS:
        raise ValueError('unsupported or protected dataset before array read')
    if ref['dataset']=='visual_cognitive_motivation' and ref['subject']=='S06' and 'Part1' in ref['record']:
        raise ValueError('excluded Visual S06 Part1 before array read')
    path = Path(ref['array_path']).resolve()
    if not path.is_relative_to(Path(parent).resolve()/'prepared'):
        raise ValueError('array outside exact public parent prepared namespace')
    if len(ref['eeg_channels']) != 6 or ref['array_index'] < 0:
        raise ValueError('invalid feature or window identity')
    return path


def synchronous_regions(ref, refs):
    return sorted([r for r in refs if r['key']==ref['key'] and r['window']==ref['window']
        and r['hb_channel']!=ref['hb_channel'] and r['site']!=ref['site']
        and abs(r['eeg_start_s']-ref['eeg_start_s'])<1e-6
        and abs(r['hb_start_s']-ref['hb_start_s'])<1e-6], key=lambda r:r['id'])


def region_split_ids(ids, refs, target_site):
    """Map the retained inner split by native record/window, never array order."""
    lookup = {r['id']:r for r in refs}
    mapped, missing = [], []
    by_window = {}
    for ref in refs:
        by_window.setdefault((ref['key'],ref['window'],ref['site']), []).append(ref)
    for identity in ids:
        source = lookup[identity]
        matches = by_window.get((source['key'],source['window'],target_site), [])
        matches = [r for r in matches if abs(r['eeg_start_s']-source['eeg_start_s'])<1e-6
            and abs(r['hb_start_s']-source['hb_start_s'])<1e-6]
        if len(matches)!=1:
            missing.append(dict(source_id=identity, target_site=target_site, reason='regional_identity_not_unique_or_missing'))
            continue
        mapped.append(matches[0]['id'])
    return mapped, missing


def parent_plan(cfg, project_root):
    """Inspect only exact public metadata and reject scope before payload reads."""
    root = Path(project_root)
    parent, split_root = root/cfg['source_run'], root/cfg['split_run']
    if read_json(parent/'summary.json')['execution']!='completed' or read_json(split_root/'summary.json')['execution']!='completed':
        raise ValueError('parent or retained split evidence is not terminal')
    refs = read_json(parent/'cohort.json')['refs']
    for ref in refs:
        validate_ref(ref,parent)
    lookup = {r['id']:r for r in refs}
    if len(lookup)!=len(refs):
        raise ValueError('duplicate parent identity')
    panel = read_json(parent/'diagnostic_plan.json')['windows']
    old_panel = read_json(split_root/'measured_plan.json')['windows']
    if (len(panel)!=72 or len({(r['dataset'],r['subject']) for r in panel})!=18
            or [r['id'] for r in panel]!=[r['id'] for r in old_panel]
            or any(r['id'] not in lookup or lookup[r['id']]!=r for r in panel)):
        raise ValueError('frozen public panel identity mismatch')
    expanded, calibration, exclusions,region_eligibility = [], {}, [], []
    for source in panel:
        neighbors = synchronous_regions(source,refs)
        present={r['region'] for r in [source]+neighbors}
        for region in cfg['data']['region_names']:
            region_eligibility.append(dict(parent_panel_id=source['id'],dataset=source['dataset'],subject=source['subject'],
                record=source['record'],window=source['window'],region=region,
                status='eligible' if region in present else 'unavailable_parent_geometry_or_prepared_support'))
        for ref in [source]+neighbors:
            key = f'{ref["dataset"]}__{ref["site"]}__outer{ref["subject_fold"]}'
            source_key = f'{source["dataset"]}__{source["site"]}__outer{source["subject_fold"]}'
            expanded.append(dict(ref=ref, parent_panel_id=source['id'], calibration_key=key,
                synchronous_region_ids=[r['id'] for r in synchronous_regions(ref,refs)]))
            if key in calibration:
                if calibration[key]['source_key']!=source_key:
                    raise ValueError('ambiguous regional split owner')
                continue
            old = read_json(split_root/'calibration'/(source_key+'.json'))
            train, missing_train = region_split_ids(old['training_ids'],refs,ref['site'])
            select, missing_select = region_split_ids(old['selection_ids'],refs,ref['site'])
            coordinate = read_json(parent/'coordinates'/(key+'.json'))
            train_subjects = {lookup[i]['subject'] for i in train}
            selection_subjects = {lookup[i]['subject'] for i in select}
            outer_eval = {r['subject'] for r in panel if r['dataset']==ref['dataset'] and r['subject_fold']==ref['subject_fold']}
            if train_subjects & selection_subjects or outer_eval & (train_subjects|selection_subjects|set(coordinate['training_subjects'])):
                raise ValueError('training, selection or coordinate leaked evaluation subject')
            original_train=set(old['training_subjects'])
            if (not train_subjects.issubset(original_train) or not selection_subjects.issubset(set(old['selection_subjects']))
                    or (len(train)==18 and train_subjects!=original_train)):
                raise ValueError('regional mapping changed retained subject split')
            status = 'eligible' if len(train)==18 and len(select)==6 else 'ineligible_missing_regional_training_support'
            retained_coordinate = {k:coordinate[k] for k in ('key','pc','hb_factor','sd','training_subjects','training_windows')}
            calibration[key] = dict(key=key,source_key=source_key,site=ref['site'],dataset=ref['dataset'],
                training_ids=train,selection_ids=select,training_subjects=sorted(train_subjects),
                selection_subjects=sorted(selection_subjects),outer_evaluation_subjects=sorted(outer_eval),
                parent_coordinate=retained_coordinate,parent_coordinate_path=str(parent/'coordinates'/(key+'.json')),
                status=status,missing=missing_train+missing_select)
            exclusions.extend(missing_train+missing_select)
    if len({x['ref']['id'] for x in expanded})!=len(expanded):
        raise ValueError('duplicate expanded panel identity')
    needed={item['ref']['id'] for item in expanded}
    for item in expanded:
        needed.update(item['synchronous_region_ids'])
    for item in calibration.values():
        needed.update(item['training_ids']);needed.update(item['selection_ids'])
    retained_refs=[lookup[identity] for identity in sorted(needed)]
    return dict(schema='shared_driver_modes_public_plan_v1',parent=str(parent),split_root=str(split_root),
        original_windows=len(panel),original_subjects=18,windows=expanded,calibration=list(calibration.values()),
        refs=retained_refs,regional_exclusions=exclusions,region_eligibility=region_eligibility,
        planned_region_slots=len(region_eligibility),expanded_windows=len(expanded),
        selection='unchanged_parent_QC_selected_panel_and_metadata_only_synchronous_regions',
        interpretation='cross_fitted_public_development_not_protected_confirmation')


@lru_cache(maxsize=8)
def prepared_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        e, h = arrays['eeg_features'].copy(), arrays['hb'].copy()
    if e.ndim!=3 or e.shape[1:]!=(120,30) or h.shape!=(len(e),120,2):
        raise ValueError('parent prepared array shape mismatch')
    return e,h


def raw_target(ref, parent):
    path = validate_ref(ref,parent)
    e,h = prepared_arrays(str(path))
    if ref['array_index'] >= len(e):
        raise ValueError('native window array index out of range')
    e,h = e[ref['array_index']],h[ref['array_index']]
    if not np.isfinite(e).all() or not np.isfinite(h).all():
        raise ValueError('parent prepared window lacks complete finite support')
    return e,h


def scaled_target(ref, parent, calibration):
    e,h = raw_target(ref,parent)
    return np.column_stack((e*calibration['eeg_factor'],h*calibration['hb_factor']))


def fit_arm(cfg, p, target, calibration, arm, visible=None, *, gamma=None, common=True):
    if arm not in ARMS:
        raise ValueError('unregistered arm')
    design = operators(cfg['tensor']['steps'])
    loading = spectral_loadings()
    if arm=='R0_PCA_refit':
        loading = np.asarray(calibration['pc'],float)[:,None]
    elif arm=='R1_broadband':
        loading = loading[:,:1]
    g = (calibration['selected_gamma'] if gamma is None else gamma) if arm=='R2_selected' else 0.
    if arm=='independent':
        g = 0.
    return fit_shared_driver_modes(target[:,:30],target[:,30:],p,cfg['tensor']['dt_s'],
        loadings=loading,temporal_basis=design['temporal'],sd=np.asarray(calibration['sd'],float),
        visible=visible,eeg_operator=design['eeg'],hb_operator=design['hb'],
        hb_basis=design['common'] if common else None,gamma=g,independent=arm=='independent',
        state_sd=cfg['model']['state_sd'],ar_phi=cfg['model']['ar_phi'],ar_weight=cfg['model']['ar_weight'],
        initial_weight=cfg['model']['initial_weight'],component_sd=np.asarray(calibration['component_sd'],float) if common else None,
        max_nfev=cfg['solver']['max_nfev'],substeps=cfg['solver']['substeps'],
        numerical_backend=cfg['solver']['numerical_backend'])


def score_prediction(prediction, target, sd, endpoint, *, physical=None, component=None, eeg_factor=1., hb_factor=1.):
    pred,y,scale,mask = map(np.asarray,(prediction,target,sd,endpoint))
    if pred.shape!=y.shape or mask.shape!=y.shape or pred.shape[1]!=32 or scale.shape!=(32,):
        raise ValueError('metric shape or coordinate mismatch')
    error = pred-y
    result = dict(scored_EEG_points=int(mask[:,:30].sum()),scored_HbO_points=int(mask[:,30].sum()),scored_HbR_points=int(mask[:,31].sum()))
    modality_errors=[]
    for label,cols in [('EEG',slice(0,30)),('Hb',slice(30,32)),('HbO',slice(30,31)),('HbR',slice(31,32))]:
        support = mask[:,cols]
        if not support.any():
            result[label+'_nrmse']=None
            continue
        normalized = error[:,cols]/scale[cols]
        result[label+'_nrmse']=float(np.sqrt(np.mean(normalized[support]**2)))
        factor = eeg_factor if label=='EEG' else hb_factor
        result[label+'_rmse_native_feature']=float(np.sqrt(np.mean((error[:,cols][support]/factor)**2)))
        if label in ('EEG','Hb'):
            modality_errors.append(result[label+'_nrmse']**2)
    result['total_nrmse']=float(np.sqrt(np.mean(modality_errors))) if modality_errors else None
    if physical is not None and mask[:,30:].any():
        result['physical_Hb_nrmse']=float(np.sqrt(np.mean((((np.asarray(physical)[:,30:]-y[:,30:])/scale[30:])[mask[:,30:]])**2)))
    if component is not None:
        result['common_Hb_RMS_training_SD']=float(np.sqrt(np.mean((np.asarray(component)[:,30:]/scale[30:])**2)))
        component_energy=float(np.mean(np.asarray(component)[:,30:]**2))
        prediction_energy=float(np.mean(pred[:,30:]**2))
        result['common_Hb_energy_over_total_prediction']=component_energy/max(prediction_energy,1e-24)
    return result


def fit_score(fit, target, calibration, endpoint):
    result = dict(converged=bool(fit.get('converged',False)),status=fit.get('status','missing_status'),
        objective=fit.get('objective'),evaluations=fit.get('evaluations',fit.get('nfev')),
        scaled_gradient_inf_norm=fit.get('scaled_gradient_inf_norm'),convergence_reason=fit.get('convergence_reason'),
        complexity=fit.get('complexity'),failure_log=fit.get('failure_log',[]))
    if 'prediction' in fit:
        result.update(score_prediction(fit['prediction'],target,np.asarray(calibration['sd']),endpoint,
            physical=fit.get('physical_prediction'),component=fit.get('observation_component'),
            eeg_factor=calibration.get('eeg_factor',1.),hb_factor=calibration.get('hb_factor',1.)))
    if 'r' in fit:
        for index in range(fit['r'].shape[1]):
            result[f'r{index}_reference_centered_RMS']=float(np.sqrt(np.mean(centered(fit['r'][:,index])**2)))
            result[f'r{index}_reference_mean']=float(np.mean(fit['r'][:20,index]))
        result['u_H_reference_centered_RMS']=float(np.sqrt(np.mean(centered(fit['u_h'])**2)))
        result['u_H_reference_mean']=float(np.mean(fit['u_h'][:20]))
    return result


def interpolation_inputs(target, mask, template):
    """Only visible own-context values enter interpolation; no zero fill."""
    out = np.asarray(template,float).copy()
    for j in range(target.shape[1]):
        support = np.flatnonzero(mask[:,j])
        if len(support):
            out[:,j] = np.interp(np.arange(len(target)),support,target[support,j])
    return out


def calibration_worker(payload):
    cfg,project_root,run_root,spec,refs,parent = payload
    out = Path(run_root); path=out/'calibration'/(spec['key']+'.json')
    if path.exists():
        return read_json(path)
    started=time.monotonic()
    if spec['status']!='eligible':
        result=dict(spec)
        result.update(status='failed_missing_training_support',seconds=0.)
        write_json(path,result);return result
    lookup={r['id']:r for r in refs}; train=[lookup[i] for i in spec['training_ids']]
    raw=[raw_target(r,parent) for r in train]
    e=np.array([v[0] for v in raw]); h=np.array([v[1] for v in raw])
    w=spectral_loadings(); projections=e@w
    projection_rms=np.sqrt(np.mean(projections**2,axis=(0,1)))
    if projection_rms[0]<=1e-12:
        raise ValueError('degenerate training broadband mode projection')
    factor=cfg['model']['eeg_mode_training_RMS_target']/projection_rms[0]
    hb_factor=spec['parent_coordinate']['hb_factor']
    targets=np.concatenate((e*factor,h*hb_factor),axis=2)
    scale=np.maximum(targets.std(axis=(0,1)),1e-8)
    cal=dict(key=spec['key'],dataset=spec['dataset'],site=spec['site'],source_key=spec['source_key'],
        training_ids=spec['training_ids'],selection_ids=spec['selection_ids'],training_subjects=spec['training_subjects'],
        selection_subjects=spec['selection_subjects'],outer_evaluation_subjects=spec['outer_evaluation_subjects'],
        pc=spec['parent_coordinate']['pc'],eeg_factor=factor,hb_factor=hb_factor,sd=scale,
        selected_gamma=0.,component_sd=np.ones(4)*.02,training_mode_projection_RMS_raw=projection_rms,
        training_mode_projection_RMS_scaled=projection_rms*factor,training_template=targets.mean(axis=0),
        EEG_scaling='one_positive_scalar_before_fixed_loadings_no_per_feature_target_transform',
        SD_interpretation='training_feature_variability_weights_not_calibrated_sensor_noise',
        coordinate='relative_logpower_and_relative_Hb_effective_modes',fit_partition='retained_18_training_windows_only')
    p=parameters(cfg,project_root); common=operators()['common']; coefficients=[]; training_rows=[]
    for ref,y in zip(train,targets):
        fit=fit_arm(cfg,p,y,cal,'R1_broadband',common=False)
        training_rows.append(dict(id=ref['id'],**fit_score(fit,y,cal,np.ones_like(y,bool))))
        if fit.get('converged'):
            residual=y[:,30:]-fit['physical_prediction'][:,30:]
            design=common.reshape(240,4)/np.tile(scale[30:],120)[:,None]
            coefficients.append(np.linalg.lstsq(design,(residual/scale[30:]).ravel(),rcond=1e-10)[0])
    cal['training_R1_records']=training_rows
    if len(coefficients)<3:
        cal.update(status='failed_insufficient_converged_training_R1',seconds=time.monotonic()-started)
        write_json(path,cal);return cal
    cal['component_sd']=np.maximum(np.sqrt(np.mean(np.array(coefficients)**2,axis=0)),scale[30:].mean()*1e-6)
    rows=[]
    for gamma in cfg['model']['gamma_candidates']:
        for identity in spec['selection_ids']:
            ref=lookup[identity]; y=scaled_target(ref,parent,cal)
            for mode in cfg['calibration']['gamma_selection_modes']:
                cell=out/'calibration_selection'/spec['key']/f'gamma_{gamma:g}'/identity/(mode+'.json')
                if cell.exists():
                    row=read_json(cell)
                else:
                    mask=visible_mask(mode,cfg); fit=fit_arm(cfg,p,y,cal,'R2_selected',mask,gamma=gamma)
                    row=dict(id=identity,gamma=gamma,mode=mode,**fit_score(fit,y,cal,endpoint_mask(mode,mask)))
                    write_json(cell,row)
                rows.append(row)
    ranking=[]
    for gamma in cfg['model']['gamma_candidates']:
        candidate=[r for r in rows if r['gamma']==gamma]
        losses=[r.get('Hb_nrmse') if r.get('converged') and r.get('Hb_nrmse') is not None else cfg['calibration']['failure_penalty'] for r in candidate]
        ranking.append(dict(gamma=gamma,score=float(np.mean(losses)),planned=len(candidate),converged=sum(r['converged'] for r in candidate)))
    ranking.sort(key=lambda r:(r['score'],abs(r['gamma']),r['gamma']))
    cal['selected_gamma']=ranking[0]['gamma']; cal['gamma_ranking']=ranking
    cal['selection_coordinate']='new_EEG_scale_fit_on_18_training_windows_parent_Hb_scale_frozen_outer_fold'
    cal['ridge_models']={}
    for mode in MODES[1:]:
        mask=visible_mask(mode,cfg)
        inputs=np.concatenate([interpolation_inputs(y,mask,cal['training_template']) for y in targets])
        cal['ridge_models'][mode]=fit_standardized_ridge(inputs,targets.reshape(-1,32),cfg['measured']['ridge_weight'])
    cal.update(status='completed',seconds=time.monotonic()-started)
    write_json(path,cal)
    return cal


def pairing_target(cfg, y, mode, pairing, *, donor_eeg=None):
    target=np.asarray(y,float); fit_target=target.copy(); mask=visible_mask(mode,cfg)
    support=np.ones(len(target),bool)
    if pairing in ('real_shift_support','nonwrapping_shift_12s'):
        shift=cfg['measured']['shift_steps']; support[:shift]=False; mask[:shift,:30]=False
        if pairing=='nonwrapping_shift_12s':
            fit_target[shift:,:30]=target[:-shift,:30]
        fit_target[:shift,:30]=np.nan
    elif pairing in ('wrong_training_subject','other_region_EEG'):
        if donor_eeg is None or np.asarray(donor_eeg).shape!=(120,30):
            raise ValueError('exact supported donor EEG is required')
        fit_target[:,:30]=donor_eeg
    elif pairing!='real':
        raise ValueError('unregistered pairing')
    return fit_target,mask,endpoint_mask(mode,visible_mask(mode,cfg),support)


def measured_cells(cfg, plan):
    cells=[]
    for item in plan['windows']:
        for arm in ARMS:
            for mode in MODES:
                cells.append(dict(**item,arm=arm,mode=mode,pairing='real'))
        for mode in cfg['measured']['null_modes']:
            for pairing in cfg['measured']['pairings'][1:]:
                cells.append(dict(**item,arm=cfg['measured']['null_arm'],mode=mode,pairing=pairing))
    return cells


def cell_path(root, spec):
    return Path(root)/'measured'/spec['ref']['id']/spec['arm']/spec['pairing']/(spec['mode']+'.json')


def save_fit_arrays(path, fit, target=None, **extra):
    arrays={k:np.asarray(fit[k]) for k in ('prediction','physical_prediction','observation_component','r','u_h','states','coefficients','initial_coordinates') if k in fit}
    if target is not None:
        arrays['target']=np.asarray(target)
    arrays.update({k:np.asarray(v) for k,v in extra.items()})
    temporary=Path(path).with_name(Path(path).name+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    temporary.replace(path)


def measured_worker(payload):
    cfg,project_root,run_root,spec,refs,parent=payload
    path=cell_path(run_root,spec)
    if path.exists():
        return read_json(path)
    started=time.monotonic(); ref=spec['ref']; cal=read_json(Path(run_root)/'calibration'/(spec['calibration_key']+'.json'))
    identity={k:ref[k] for k in ('id','dataset','subject','record','site','region','task','condition','window')}
    row=dict(**identity,arm=spec['arm'],mode=spec['mode'],pairing=spec['pairing'],parent_panel_id=spec['parent_panel_id'],
        calibration_key=spec['calibration_key'],eeg_channels=ref['eeg_channels'],hb_channel=ref['hb_channel'])
    if cal['status']!='completed':
        row.update(converged=False,status='calibration_unavailable',seconds=time.monotonic()-started)
        write_json(path,row);return row
    y=scaled_target(ref,parent,cal); donor=None; lookup={r['id']:r for r in refs}
    if spec['pairing']=='wrong_training_subject':
        donors=[lookup[i] for i in cal['training_ids'] if lookup[i]['subject']!=ref['subject']
            and lookup[i]['task']==ref['task'] and lookup[i]['condition']==ref['condition']]
        donor=sorted(donors,key=lambda r:r['id'])[sum(ref['id'].encode())%len(donors)] if donors else None
    elif spec['pairing']=='other_region_EEG':
        donors=[lookup[i] for i in spec['synchronous_region_ids']]
        donor=sorted(donors,key=lambda r:r['id'])[0] if donors else None
    if spec['pairing'] in ('wrong_training_subject','other_region_EEG') and donor is None:
        row.update(converged=False,status='matched_donor_unavailable',seconds=time.monotonic()-started)
        write_json(path,row);return row
    donor_eeg=raw_target(donor,parent)[0]*cal['eeg_factor'] if donor is not None else None
    if donor is not None:
        row.update(donor_id=donor['id'],donor_subject=donor['subject'],donor_region=donor['region'],
            donor_eeg_channels=donor['eeg_channels'],shared_EEG_channels=sorted(set(ref['eeg_channels'])&set(donor['eeg_channels'])))
    fitted,mask,endpoint=pairing_target(cfg,y,spec['mode'],spec['pairing'],donor_eeg=donor_eeg)
    fit=fit_arm(cfg,parameters(cfg,project_root),fitted,cal,spec['arm'],mask)
    row.update(fit_score(fit,y,cal,endpoint),selected_gamma=cal['selected_gamma'])
    if spec['mode']=='full' and spec['pairing']=='real' and 'prediction' in fit:
        path.parent.mkdir(parents=True,exist_ok=True)
        save_fit_arrays(path.with_suffix('.npz'),fit,y)
        row['arrays']=str(path.with_suffix('.npz').relative_to(run_root))
    if spec['arm']=='R2_selected' and spec['pairing']=='real':
        baseline=[]; template=np.asarray(cal['training_template']); interpolation=interpolation_inputs(y,mask,template)
        predictions={'training_template':template}
        if spec['mode']!='full':
            predictions['own_context']=interpolation
            ridge={k:np.asarray(v) for k,v in cal['ridge_models'][spec['mode']].items()}
            predictions['ridge_cross']=predict_standardized_ridge(ridge,interpolation)
        for name,pred in predictions.items():
            baseline.append(dict(arm=name,converged=True,status='completed',**score_prediction(pred,y,np.asarray(cal['sd']),endpoint,
                eeg_factor=cal['eeg_factor'],hb_factor=cal['hb_factor'])))
        row['baselines']=baseline
    row.update(seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_json(path,row);return row


def synthetic_case(cfg, repeat, scenario, stream):
    """Independent seeded interventions, including misspecified generating modes."""
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],stream,repeat,sum(scenario.encode())]))
    design=operators(); basis=design['temporal']; w=spectral_loadings(); truth_loading=w.copy()
    weights=1/(1+np.arange(24))**1.25; weights[0]=.25
    coefficients=rng.normal(size=(24,3))*weights[:,None]
    drivers=basis@coefficients
    drivers*=cfg['synthetic']['truth_state_RMS']/np.sqrt(np.mean(drivers**2,axis=0))[None,:]
    if scenario=='a_only':
        drivers[:,1]=0.
    elif scenario=='b_only_gamma0':
        drivers[:,0]=0.
    gamma=0. if scenario=='b_only_gamma0' else cfg['synthetic']['truth_gamma']
    u=(drivers[:,0]+gamma*drivers[:,1])/np.sqrt(1+gamma**2)
    if scenario=='independent_Hb':
        u=drivers[:,2]
    elif scenario=='lag':
        u=np.r_[np.zeros(8),u[:-8]]
    elif scenario=='wrong_spectral':
        alternate=rng.normal(size=30)
        alternate-=w@(w.T@alternate)
        truth_loading[:,1]=alternate/np.linalg.norm(alternate)
    drivers[:,2]=u
    coefficients=np.linalg.lstsq(basis,drivers,rcond=1e-12)[0]
    delay_projection_rmse=float(np.sqrt(np.mean((basis@coefficients[:,2]-u)**2)))
    common_coefficients=np.zeros(4)
    if scenario=='common_Hb':
        common_coefficients=rng.normal(size=4)*cfg['synthetic']['common_coefficient_SD']
    p=parameters(cfg)
    truth=forward_shared_driver_modes(coefficients,np.zeros(5),p,.25,loadings=truth_loading,
        temporal_basis=basis,eeg_operator=design['eeg'],hb_operator=design['hb'],hb_basis=design['common'],
        component_coefficients=common_coefficients,independent=True,derivative=False,
        substeps=cfg['solver']['substeps'],numerical_backend=cfg['solver']['numerical_backend'])
    truth['generation']='DCT_projected_two_second_delay' if scenario=='lag' else 'registered_DCT_modes'
    truth['delay_projection_RMSE']=delay_projection_rmse if scenario=='lag' else 0.
    noise=np.r_[np.full(30,cfg['synthetic']['noise_SD'][0]/np.sqrt(30)),cfg['synthetic']['noise_SD'][1:]]
    target=truth['prediction']+rng.normal(size=(120,32))*noise
    score_sd=np.r_[np.full(30,cfg['synthetic']['score_SD'][0]/np.sqrt(30)),cfg['synthetic']['score_SD'][1:]]
    cal=dict(pc=w[:,0],sd=score_sd,selected_gamma=0.,component_sd=cfg['synthetic']['common_coefficient_SD'],eeg_factor=1.,hb_factor=1.)
    return target,truth,cal


def centered(values, reference_steps=20):
    values=np.asarray(values,float)
    return values-values[:reference_steps].mean(axis=0)


def recovery_metrics(fit, truth, reference_steps=20):
    if 'r' not in fit:
        return {}
    result={}
    for label,pred,true in [('a',fit['r'][:,0],truth['r'][:,0]),('u_H',fit['u_h'],truth['u_h'])]:
        result[label+'_reference_centered_RMSE']=float(np.sqrt(np.mean((centered(pred,reference_steps)-centered(true,reference_steps))**2)))
        result[label+'_raw_RMSE']=float(np.sqrt(np.mean((pred-true)**2)))
        result[label+'_reference_mean_error']=float(np.mean(pred[:reference_steps]-true[:reference_steps]))
    if fit['r'].shape[1]>1:
        result['b_reference_centered_RMSE']=float(np.sqrt(np.mean((centered(fit['r'][:,1],reference_steps)-centered(truth['r'][:,1],reference_steps))**2)))
        result['b_raw_RMSE']=float(np.sqrt(np.mean((fit['r'][:,1]-truth['r'][:,1])**2)))
        estimates=centered(fit['r'],reference_steps); sources=centered(truth['r'],reference_steps)
        for out in range(2):
            for source in range(2):
                result[f'mode_{out}_on_truth_{source}_slope']=float(sources[:,source]@estimates[:,out]/(sources[:,source]@sources[:,source]+1e-12))
    return result


def synthetic_selection(cfg, run_root):
    path=Path(run_root)/'synthetic_calibration.json'
    if path.exists():
        return read_json(path)
    p=parameters(cfg);rows=[]
    for repeat in range(cfg['synthetic']['calibration_repeats']):
        scenario=cfg['synthetic']['selection_scenarios'][repeat%len(cfg['synthetic']['selection_scenarios'])]
        y,truth,cal=synthetic_case(cfg,repeat,scenario,cfg['synthetic']['calibration_seed_stream'])
        for gamma in cfg['model']['gamma_candidates']:
            mask=visible_mask('Hb_hidden',cfg)
            fit=fit_arm(cfg,p,y,cal,'R2_selected',mask,gamma=gamma)
            rows.append(dict(repeat=repeat,scenario=scenario,gamma=gamma,**fit_score(fit,y,cal,endpoint_mask('Hb_hidden',mask))))
    ranking=[]
    for gamma in cfg['model']['gamma_candidates']:
        candidate=[r for r in rows if r['gamma']==gamma]
        score=np.mean([r['Hb_nrmse'] if r['converged'] and r.get('Hb_nrmse') is not None else cfg['calibration']['failure_penalty'] for r in candidate])
        ranking.append(dict(gamma=gamma,score=float(score),converged=sum(r['converged'] for r in candidate)))
    ranking.sort(key=lambda r:(r['score'],abs(r['gamma']),r['gamma']))
    result=dict(status='completed',selected_gamma=ranking[0]['gamma'],ranking=ranking,rows=rows,
        scope='independent_synthetic_calibration_seeds_no_measured_arrays',seed_stream=cfg['synthetic']['calibration_seed_stream'])
    write_json(path,result);return result


def synthetic_specs(cfg):
    return [dict(repeat=i,scenario=scenario,arm=arm,mode=mode) for i in range(cfg['synthetic']['repeats'])
        for scenario in cfg['synthetic']['scenarios'] for arm in ARMS for mode in cfg['synthetic']['modes']]


def synthetic_path(root,spec):
    return Path(root)/'synthetic'/spec['scenario']/f'repeat_{spec["repeat"]:03d}'/spec['arm']/(spec['mode']+'.json')


def synthetic_worker(payload):
    cfg,run_root,spec,gamma=payload
    path=synthetic_path(run_root,spec)
    if path.exists():
        retained=read_json(path)
        if retained.get('selected_gamma')!=gamma:
            raise ValueError('synthetic resume gamma differs from frozen selected gamma')
        return retained
    started=time.monotonic()
    y,truth,cal=synthetic_case(cfg,spec['repeat'],spec['scenario'],cfg['synthetic']['evaluation_seed_stream']);cal['selected_gamma']=gamma
    mask=visible_mask(spec['mode'],cfg);fit=fit_arm(cfg,parameters(cfg),y,cal,spec['arm'],mask)
    row=dict(**spec,selected_gamma=gamma,seed_stream=cfg['synthetic']['evaluation_seed_stream'],
        truth_generation=truth['generation'],delay_projection_RMSE=truth['delay_projection_RMSE'],
        **fit_score(fit,y,cal,endpoint_mask(spec['mode'],mask)),**recovery_metrics(fit,truth),seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    if spec['mode']=='full' and spec['repeat']==0 and 'prediction' in fit:
        path.parent.mkdir(parents=True,exist_ok=True)
        save_fit_arrays(path.with_suffix('.npz'),fit,y,truth_r=truth['r'],truth_u_h=truth['u_h'],truth_prediction=truth['prediction'])
        row['arrays']=str(path.with_suffix('.npz').relative_to(run_root))
    write_json(path,row);return row


def safe_worker(payload):
    kind,args=payload
    started=time.monotonic()
    worker=calibration_worker if kind=='calibrate' else measured_worker if kind=='measured' else synthetic_worker
    try:
        return worker(args)
    except Exception as exc:
        if kind=='calibrate':
            path=Path(args[2])/'calibration'/(args[3]['key']+'.json');identity=dict(key=args[3]['key'])
        elif kind=='measured':
            path=cell_path(args[2],args[3]);identity={k:args[3]['ref'][k] for k in ('id','dataset','subject','record','site','region')}
            identity.update({k:args[3][k] for k in ('arm','mode','pairing')})
        else:
            path=synthetic_path(args[1],args[2]);identity=args[2]
        row=dict(**identity,status='failed_exception',converged=False,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),
            seconds=time.monotonic()-started,evaluations=0,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write_json(path,row);return row


def parallel_stage(kind,payloads,workers,run_root):
    """Bound the queue; durable process supervision is supplied by the launcher."""
    started=time.monotonic(); payloads=iter(payloads); pending={}; done=failed=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        exhausted=False
        while pending or not exhausted:
            while not exhausted and len(pending)<2*workers:
                try:
                    payload=next(payloads)
                except StopIteration:
                    exhausted=True;break
                pending[pool.submit(safe_worker,(kind,payload))]=None
            if not pending:
                break
            finished,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in finished:
                pending.pop(future)
                row=future.result();done+=1;failed+=int(row.get('converged') is False or row.get('status','').startswith('failed'))
            write_json(Path(run_root)/f'{kind}_progress.json',dict(stage=kind,completed=done,failed=failed,
                in_flight=len(pending),workers=workers,seconds=time.monotonic()-started))
    return dict(stage=kind,completed=done,failed=failed,seconds=time.monotonic()-started)


def summary_table(frame,groups,metrics):
    rows=[]
    for key,group in frame.groupby(groups,dropna=False):
        key=key if isinstance(key,tuple) else (key,)
        row=dict(zip(groups,key)); good=group[group.converged]
        row.update(planned=len(group),converged=int(group.converged.sum()),failed=int((~group.converged).sum()),success_rate=float(group.converged.mean()))
        for metric in metrics:
            if metric not in group:
                continue
            values=pd.to_numeric(good[metric],errors='coerce').dropna()
            if 'subject' in good and len(values):
                subject_values=good.assign(_value=pd.to_numeric(good[metric],errors='coerce')).groupby('subject')._value.mean().dropna()
                row[metric]=float(subject_values.mean())
                row[metric+'_subjects']=len(subject_values)
            else:
                row[metric]=float(values.mean()) if len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def paired_rows(frame, *, controls, groups, value='Hb_nrmse',seed=20261002):
    """Resample subjects as whole blocks; retain all planned-pair denominators."""
    result=[];rng=np.random.default_rng(seed)
    for keys,group in frame.groupby(groups,dropna=False):
        keys=keys if isinstance(keys,tuple) else (keys,)
        for control in controls:
            c=group[group.arm==control]
            for arm in sorted(set(group.arm)-{control}):
                a=group[group.arm==arm]
                merged=a.merge(c,on=['id'],suffixes=('_arm','_control'))
                if value+'_arm' not in merged or value+'_control' not in merged:
                    continue
                a_value=pd.to_numeric(merged[value+'_arm'],errors='coerce')
                c_value=pd.to_numeric(merged[value+'_control'],errors='coerce')
                finite=np.isfinite(a_value)&np.isfinite(c_value)
                if not finite.any():
                    continue
                good=merged[merged.converged_arm & merged.converged_control & finite].copy()
                good['gain']=pd.to_numeric(good[value+'_control'],errors='coerce')-pd.to_numeric(good[value+'_arm'],errors='coerce')
                subject=good.groupby('subject_arm').gain.mean().dropna()
                bootstrap=np.mean(rng.choice(subject.to_numpy(),size=(1000,len(subject)),replace=True),axis=1) if len(subject) else np.array([])
                interval=np.quantile(bootstrap,[.025,.975]) if len(bootstrap) else [np.nan,np.nan]
                result.append(dict(zip(groups,keys),arm=arm,control=control,planned_pairs=len(merged),common_success_pairs=len(good),
                    failed_or_invalid_pairs=len(merged)-len(good),
                    subjects=len(subject),gain=float(subject.mean()) if len(subject) else None,ci_low=interval[0],ci_high=interval[1],
                    metric=value,positive_gain='arm_better',interval='subject_block_1000_frozen_coordinates_exploratory'))
    return result


def summarize(cfg,out):
    out=Path(out); result=dict(schema='shared_driver_modes_summary_v1',experiment_id=cfg['experiment_id'],
        completed_at=datetime.now(timezone.utc).isoformat(),scope='public_development_fixed_modes_not_physiological_truth',
        limitations=['QC_selected_parent_panel','offline_feature_not_native_future_prediction','coarse_region_geometry',
            'relative_logpower_modes_not_EI_synchrony_or_metabolism','scalar_Hb_input_retains_Balloon_HbO_HbR_structural_limit',
            'R0_PCA_refit_has_new_temporal_prior_and_is_not_historical_M0'])
    synthetic=[]
    for spec in synthetic_specs(cfg):
        path=synthetic_path(out,spec)
        if path.exists():
            synthetic.append(read_json(path))
    if synthetic:
        frame=pd.DataFrame(synthetic);frame.to_csv(out/'synthetic_cells.csv',index=False)
        summary_table(frame,['scenario','mode','arm'],['total_nrmse','EEG_nrmse','Hb_nrmse','HbO_nrmse','HbR_nrmse',
            'a_reference_centered_RMSE','b_reference_centered_RMSE','u_H_reference_centered_RMSE']).to_csv(out/'synthetic_summary.csv',index=False)
        result['synthetic']=dict(planned=len(synthetic_specs(cfg)),recorded=len(synthetic),converged=int(frame.converged.sum()))
    measured=[]
    if (out/'measured_plan.json').exists():
        plan=read_json(out/'measured_plan.json');specs=measured_cells(cfg,plan)
        for spec in specs:
            path=cell_path(out,spec)
            if path.exists():
                row=read_json(path);measured.append(row)
                for baseline in row.get('baselines',[]):
                    identity_keys=('id','dataset','subject','record','site','region','task','condition','window','mode','pairing',
                        'parent_panel_id','calibration_key','eeg_channels','hb_channel')
                    measured.append({**{k:row[k] for k in identity_keys if k in row},**baseline})
        frame=pd.DataFrame(measured)
        if len(frame):
            frame.to_csv(out/'measured_cells.csv',index=False)
            summary_table(frame,['dataset','region','mode','pairing','arm'],['total_nrmse','EEG_nrmse','Hb_nrmse','HbO_nrmse','HbR_nrmse',
                'physical_Hb_nrmse','common_Hb_RMS_training_SD','common_Hb_energy_over_total_prediction',
                'r0_reference_centered_RMS','r1_reference_centered_RMS','u_H_reference_centered_RMS']).to_csv(out/'measured_summary.csv',index=False)
            real=frame[frame.pairing=='real']
            controls=['R1_broadband','R0_PCA_refit','independent','training_template','own_context','ridge_cross']
            contrasts=paired_rows(real[real['mode'].isin(['full','center_Hb','HbO_hidden','HbR_hidden','Hb_hidden'])],controls=controls,
                groups=['dataset','region','mode'],seed=cfg['seed'])
            contrasts.extend(paired_rows(real[real['mode'].isin(['full','center_EEG','EEG_channels_hidden'])],controls=controls,
                groups=['dataset','region','mode'],value='EEG_nrmse',seed=cfg['seed']))
            pd.DataFrame(contrasts).to_csv(out/'measured_paired.csv',index=False)
            null=[]
            for pairing,control in [('wrong_training_subject','real'),('other_region_EEG','real'),('nonwrapping_shift_12s','real_shift_support')]:
                pairs=frame[(frame.arm=='R2_selected') & frame.pairing.isin([pairing,control])].copy()
                pairs['arm']=pairs['pairing']
                null.extend(paired_rows(pairs,controls=[pairing],groups=['dataset','region','mode'],seed=cfg['seed']))
            pd.DataFrame(null).to_csv(out/'pairing_nulls.csv',index=False)
        recorded=sum(cell_path(out,s).exists() for s in specs)
        result['measured']=dict(planned=len(specs),recorded=recorded,expanded_windows=plan['expanded_windows'],original_windows=72,
            planned_region_slots=plan['planned_region_slots'],region_eligibility=plan['region_eligibility'],
            subjects=18,regions=sorted({s['ref']['region'] for s in specs}),
            converged=sum(bool(read_json(cell_path(out,s)).get('converged')) for s in specs if cell_path(out,s).exists()),
            regional_exclusions=plan['regional_exclusions'])
        calibration=[read_json(out/'calibration'/(s['key']+'.json')) for s in plan['calibration'] if (out/'calibration'/(s['key']+'.json')).exists()]
        pd.DataFrame([{k:c.get(k) for k in ('key','dataset','site','status','selected_gamma','eeg_factor','hb_factor','seconds')} for c in calibration]).to_csv(out/'calibration_summary.csv',index=False)
    result['execution']='completed' if ('synthetic' in result and result['synthetic']['recorded']==result['synthetic']['planned']
        and 'measured' in result and result['measured']['recorded']==result['measured']['planned']) else 'partial'
    result['verdict']='exploratory_candidate_requires_hidden_prediction_and_true_pair_gain; consult_endpoint_tables'
    write_json(out/'summary.json',result);return result


def dryrun(cfg,out):
    w=spectral_loadings();design=operators();masks={mode:visible_mask(mode,cfg) for mode in MODES}
    result=dict(status='completed',measured_arrays_read=0,arms=ARMS,loadings=w,
        shapes=dict(loadings=w.shape,temporal_basis=design['temporal'].shape,eeg_operator=design['eeg'].shape,
            Hb_operator=design['hb'].shape,common_Hb_basis=design['common'].shape),
        visible_counts={m:int(mask.sum()) for m,mask in masks.items()},
        EEG_constant_operator_max_abs=float(abs(design['eeg']@np.ones(120)).max()),
        Hb_constant_operator_max_abs=float(abs(design['hb']@np.ones(120)).max()),
        temporal_frequency_max_hz=23/(2*120*.25),protected_data='forbidden')
    parameters(cfg).validate();write_json(Path(out)/'dryrun.json',result);return result


def pilot(cfg,out,workers):
    """Synthetic-only throughput pilot; measured access is a later stage."""
    specs=[dict(repeat=0,scenario='both',arm=arm,mode=mode) for arm in ARMS for mode in ('full','Hb_hidden')]
    started=time.monotonic();rows=[]
    for spec in specs:
        rows.append(synthetic_worker((cfg,str(Path(out)/'pilot_cells'),spec,cfg['synthetic']['truth_gamma'])))
    result=dict(status='completed',scope='synthetic_only_no_measured_arrays',cells=len(rows),seconds=time.monotonic()-started,
        converged=sum(r['converged'] for r in rows),median_cell_seconds=float(np.median([r['seconds'] for r in rows])),
        max_peak_rss_bytes=max(r['peak_rss_bytes'] for r in rows),planned_workers=workers,
        note='pilot_gamma_is_registered_truth_not_selected_model_evidence; synthetic_stage_reuses_only_matching_selected_gamma')
    write_json(Path(out)/'pilot.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--project-root',type=Path,default=CODE_ROOT)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['dryrun','synthetic','pilot','calibrate','measured','summarize','all'],default='dryrun')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args(argv);cfg=read_config(args.config)
    if not 1<=args.workers<=cfg['resources']['max_workers']:
        parser.error('workers outside frozen resource bound')
    root=args.project_root.resolve();out=args.output.resolve()
    if not out.is_relative_to(root/cfg['artifact_root']):
        parser.error('output must be below the owning versioned artifact root')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('an existing supervisor already owns this run')
        resolved=out/'resolved_config.json'
        if resolved.exists() and read_json(resolved)!=serial(cfg):
            parser.error('resume cannot change the frozen configuration')
        if not resolved.exists():
            write_json(resolved,cfg)
        manifest=dict(schema='shared_driver_modes_run_v1',experiment_id=cfg['experiment_id'],execution='running',
            stage=args.stage,command=sys.argv,working_directory=str(CODE_ROOT),project_root=str(root),output=str(out),
            workers=args.workers,numerical_threads=1,supervisor=os.environ.get('SYSTEMD_EXEC_PID') or os.environ.get('INVOCATION_ID'),
            started_at=datetime.now(timezone.utc).isoformat())
        write_json(out/'execution.json',manifest)
        stages=['dryrun','synthetic','pilot','calibrate','measured','summarize'] if args.stage=='all' else [args.stage]
        try:
            for stage in stages:
                manifest['stage']=stage;write_json(out/'execution.json',manifest)
                if stage=='dryrun':
                    result=dryrun(cfg,out)
                elif stage=='pilot':
                    result=pilot(cfg,out,args.workers)
                elif stage=='synthetic':
                    if not (out/'dryrun.json').exists():
                        raise ValueError('dryrun must complete before synthetic execution')
                    selected=synthetic_selection(cfg,out)
                    result=parallel_stage('synthetic',[(cfg,str(out),spec,selected['selected_gamma']) for spec in synthetic_specs(cfg)],args.workers,out)
                    summary=summarize(cfg,out)
                    write_json(out/'synthetic_completion.json',dict(status='completed',**summary['synthetic']))
                elif stage in ('calibrate','measured'):
                    completion=read_json(out/'synthetic_completion.json')
                    if completion['recorded']!=completion['planned']:
                        raise ValueError('all registered synthetic cells must finish before measured arrays are read')
                    plan=parent_plan(cfg,root);write_json(out/'measured_plan.json',plan)
                    lookup={r['id']:r for r in plan['refs']}
                    if stage=='calibrate':
                        payloads=[(cfg,str(root),str(out),spec,[lookup[i] for i in spec['training_ids']+spec['selection_ids']],plan['parent']) for spec in plan['calibration']]
                        result=parallel_stage('calibrate',payloads,args.workers,out)
                    else:
                        if any(not (out/'calibration'/(s['key']+'.json')).exists() for s in plan['calibration']):
                            raise ValueError('every calibration must have a retained terminal record')
                        calibrations={s['key']:read_json(out/'calibration'/(s['key']+'.json')) for s in plan['calibration']}
                        payloads=[]
                        for spec in measured_cells(cfg,plan):
                            required=set(calibrations[spec['calibration_key']].get('training_ids',[]))|set(spec['synchronous_region_ids'])
                            payloads.append((cfg,str(root),str(out),spec,[lookup[i] for i in sorted(required)],plan['parent']))
                        result=parallel_stage('measured',payloads,args.workers,out)
                else:
                    result=summarize(cfg,out)
                print(json.dumps(serial(result),ensure_ascii=False),flush=True)
            manifest.update(execution='completed',finished_at=datetime.now(timezone.utc).isoformat())
        except Exception as exc:
            manifest.update(execution='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),finished_at=datetime.now(timezone.utc).isoformat())
            write_json(out/'execution.json',manifest)
            raise
        write_json(out/'execution.json',manifest)


if __name__=='__main__':
    main()
