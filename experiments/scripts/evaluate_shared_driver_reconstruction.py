#!/usr/bin/env python3
"""Bounded shared-driver linear feasibility screen, synthetic before measured.

The only measured read boundary is an exact retained projection file. Scales,
folds and feature processing remain frozen. No teacher or calibrated posterior
is produced. A durable supervisor owns substantial execution.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import traceback

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter1d
import yaml
from src.inference.t3a_balloon_robust_ssm import BalloonParameters, BalloonFixedParameters, BalloonFreeParameters
from src.inference.shared_driver_reconstruction import build_shared_driver_design, fit_shared_driver, replay_nonlinear_driver
from src.inference.observation_baselines import native_feature_operators, visible_feature_interpolation

DEFAULT = CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_reconstruction_v1.yaml'
MODALITIES = ('EEG', 'HbO', 'HbR')


def serial(v):
    if isinstance(v, dict):
        return {str(k): serial(x) for k, x in v.items()}
    if isinstance(v, (tuple, list, np.ndarray)):
        return [serial(x) for x in v]
    if isinstance(v, np.generic):
        return serial(v.item())
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def read_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['schema'] != 'shared_driver_reconstruction_v1':
        raise ValueError('unrecognized experiment contract')
    if (cfg['source_run'] != 'experiments/runs/physiology_semantic_tokenizer/ssm_overnight/20260917_measurement_retest_v1'
            or cfg['protected_boundary'] != 'only_exact_parent_prepared_arrays_no_raw_or_protected_loaders'
            or cfg['sessions'] != ['session_01','session_03','session_05']):
        raise ValueError('source/scope boundary differs from versioned contract')
    if (cfg['methods'] != ['fixed_tau','fitted_tau','source_prior_tau','oracle_lower_bound']
            or cfg['modes'] != ['full','center_EEG','center_fNIRS','EEG_only','fNIRS_only','center_EEG_shift','center_fNIRS_shift']):
        raise ValueError('unregistered method or mask')
    if cfg['subjects'] != ['subject_01', 'subject_09', 'subject_18'] or cfg['outer_folds'] != [0, 1, 2, 3]:
        raise ValueError('retained scope mismatch')
    if cfg['tensor'] != dict(trials_per_subject=24, steps=120, components=list(MODALITIES), dt_s=.25):
        raise ValueError('tensor/clock mismatch')
    if cfg['original_positions'] != [0, 1, 2, 3, 5, 6, 7, 8] or cfg['center_steps'] != 16:
        raise ValueError('identity/mask mismatch')
    grid = cfg['tau_grid']
    if not 0 < grid['minimum'] < cfg['reference_tau_s'] < grid['maximum'] or grid['log_points'] < 3:
        raise ValueError('invalid parameter grid')
    for key in ('driver_second_difference_weight','initial_state_weight','tau_log_sd'):
        if not np.isfinite(cfg['regularization'][key]) or cfg['regularization'][key] <= 0:
            raise ValueError('invalid regularization')
    if not 1<=cfg['resources']['max_workers']<=24 or cfg['resources']['numerical_threads_per_worker'] != 1:
        raise ValueError('invalid resource contract')
    return cfg


def parameters(cfg, tau):
    fixed = {k: v for k, v in cfg['fixed'].items() if k != 'kappa'}
    return BalloonParameters(BalloonFixedParameters(**fixed),
                             BalloonFreeParameters(kappa=cfg['fixed']['kappa'], tau=float(tau)))


def tau_grid(cfg):
    g = cfg['tau_grid']
    return np.unique(np.r_[np.geomspace(g['minimum'], g['maximum'], g['log_points']),
                           g['include'], cfg['reference_tau_s']])


def view_operators(cfg, mode):
    n = cfg['tensor']['steps']
    op = native_feature_operators(n)
    mask = np.ones((n, 3), dtype=bool)
    el, er, hl, hr = 0, 0, 0, 0
    left, right = (n-cfg['center_steps'])//2, (n+cfg['center_steps'])//2
    if mode.startswith('center_EEG'):
        mask[left:right, 0] = False
        el, er = left, right
    elif mode.startswith('center_fNIRS'):
        mask[left:right, 1:] = False
        hl, hr = int(2.5*left), int(2.5*right)
    elif mode == 'EEG_only':
        mask[:, 1:] = False
        hl, hr = 0, int(2.5*n)
    elif mode == 'fNIRS_only':
        mask[:, 0] = False
        el, er = 0, n
    elif mode != 'full':
        raise ValueError('unknown mode')
    pe = op['eeg']@visible_feature_interpolation(n, el, er)
    ph = op['fnirs']@visible_feature_interpolation(int(2.5*n), hl, hr)
    mean = np.zeros((3*n, 3*n))
    mean[0::3, 0::3] = pe
    mean[1::3, 1::3] = mean[2::3, 2::3] = ph@op['native_interpolation']
    return pe, ph, mean, mask


def view(eeg, hb, cfg, mode):
    pe, ph, mean, mask = view_operators(cfg, mode)
    if mode.endswith('_shift'):
        if mode.startswith('center_EEG'):
            hb = np.roll(hb, len(hb)//2, axis=0)
        else:
            eeg = np.roll(eeg, len(eeg)//2)
    y = np.column_stack((pe@eeg, ph@hb))
    y[~mask] = np.nan
    return y, mean, mask


def load_measured(cfg, project_root, subject, outer):
    """Validate frozen identity/split metadata BEFORE dereferencing exact arrays."""
    if subject not in cfg['subjects'] or outer not in cfg['outer_folds']:
        raise ValueError('out-of-scope subject/fold')
    if cfg['source_run'] != 'experiments/runs/physiology_semantic_tokenizer/ssm_overnight/20260917_measurement_retest_v1':
        raise ValueError('out-of-scope retained source')
    source = Path(project_root)/cfg['source_run']/'prepared'
    info = json.loads((source/f'{subject}_o{outer}_E0.json').read_text())
    identity = json.loads((source/f'{subject}.json').read_text())
    expected = [(s, p) for s in cfg['sessions'] for p in cfg['original_positions']]
    observed = [(r['session'], r['original_ma_trial_position']) for r in identity['trials']]
    validation = [i for i in range(24) if (i % 8) % 4 == outer]
    train = [i for i in range(24) if i not in validation]
    if (observed != expected or info['train'] != train or info['validation'] != validation
            or info['subject'] != subject or info['outer'] != outer or info['inner'] is not None):
        raise ValueError('retained identities or fold differ from the contract')
    with np.load(source/f'{subject}_o{outer}_E0.npz', allow_pickle=False) as a:
        arrays = {k: a[k] for k in ('feature_eeg', 'feature_fnirs', 'target', 'normalizer')}
    expected_shapes = dict(feature_eeg=(24,120), feature_fnirs=(24,300,2), target=(24,120,3), normalizer=(3,))
    if any(arrays[k].shape != shape or not np.isfinite(arrays[k]).all() for k, shape in expected_shapes.items()):
        raise ValueError('retained tensor/finite contract failed')
    if not np.allclose(arrays['normalizer'], info['normalization_sd'], rtol=1e-12, atol=1e-14):
        raise ValueError('normalizer differs from owning metadata')
    if np.any(arrays['normalizer'] <= 1e-8):
        raise ValueError('low_training_variance')
    reproduced = np.array([view(e,h,cfg,'full')[0] for e,h in zip(arrays['feature_eeg'],arrays['feature_fnirs'])])
    if np.max(abs(reproduced-arrays['target'])/arrays['normalizer']) > 1e-6:
        raise ValueError('measurement operator does not reproduce retained targets')
    return arrays, dict(info, trials=identity['trials'])


def load_fixed_roi_measured(cfg, project_root, subject, outer):
    """Fixed retained Hb pair; unchanged fold EEG and training-only Hb gauge.

    Validate every declared source identity using JSON before reading arrays.
    This is a new observation preparation, never a rewrite of the retained run.
    """
    from src.inference.observation_baselines import robust_mad
    expected_roi=dict(pair_name='AF7Fp1',pair_index=0,
        selection='predeclared_first_registry_pair_not_validation_score',
        eeg='exact_parent_projection_unchanged',
        fnirs_scale='training_common_MAD_with_frozen_parent_observation_loading')
    if (cfg.get('fixed_roi')!=expected_roi
            or cfg.get('source_run')!='experiments/runs/physiology_semantic_tokenizer/ssm_overnight/20260917_measurement_retest_v1'
            or cfg.get('protected_boundary')!='only_exact_parent_prepared_arrays_no_raw_or_protected_loaders'
            or cfg.get('subjects')!=['subject_01','subject_09','subject_18']
            or cfg.get('outer_folds')!=[0,1,2,3]
            or cfg.get('sessions')!=['session_01','session_03','session_05']
            or cfg.get('original_positions')!=[0,1,2,3,5,6,7,8]
            or cfg.get('tensor')!=dict(trials_per_subject=24,steps=120,components=list(MODALITIES),dt_s=.25)
            or subject not in cfg['subjects'] or outer not in cfg['outer_folds']):
        raise ValueError('fixed ROI retained scope contract mismatch')
    source=Path(project_root)/cfg['source_run']/'prepared'
    expected=[(session,position) for session in cfg['sessions'] for position in cfg['original_positions']]
    for source_subject in cfg['subjects']:
        detail=json.loads((source/f'{source_subject}.json').read_text())
        if (detail.get('subject')!=source_subject or detail.get('synthetic') is not False
                or detail.get('original_heldout_trials_processed')!=0
                or len(detail.get('fnirs_pairs',[]))!=36 or detail['fnirs_pairs'].count('AF7Fp1')!=1
                or detail['fnirs_pairs'][0]!='AF7Fp1'
                or [(r['session'],r['original_ma_trial_position']) for r in detail['trials']]!=expected
                or any(r.get('subject')!=source_subject for r in detail['trials'])):
            raise ValueError('fixed ROI source identity/pair mismatch')
        for source_outer in cfg['outer_folds']:
            info=json.loads((source/f'{source_subject}_o{source_outer}_E0.json').read_text())
            validation=[i for i in range(24) if i%4==source_outer]
            train=[i for i in range(24) if i not in validation]
            projection=info['projection']
            eligible=np.asarray(projection['pair_eligible'])
            gauge=projection['observation_loading']['fnirs_common']
            if (info.get('subject')!=source_subject or info.get('outer')!=source_outer
                    or info.get('inner') is not None or info['train']!=train or info['validation']!=validation
                    or eligible.shape!=(36,) or eligible.dtype!=np.dtype(bool) or not eligible[0]
                    or not np.isfinite(gauge) or gauge<=0):
                raise ValueError('fixed ROI source fold/eligibility/gauge mismatch')
    # The existing boundary validates the requested fold arrays and freezes EEG.
    arrays,meta=load_measured(cfg,project_root,subject,outer)
    with np.load(source/f'{subject}.npz',allow_pickle=False) as archive:
        hb=archive['feature_fnirs']
        eligible=archive['eligible']
    if (hb.shape!=(24,300,36,2) or eligible.shape!=(24,36) or eligible.dtype!=np.dtype(bool)
            or not eligible[:,0].all() or not np.isfinite(hb[:,:,0,:]).all()):
        raise ValueError('fixed ROI feature shape/finite/eligibility mismatch')
    selected=hb[:,:,0,:].copy()
    op=native_feature_operators(cfg['tensor']['steps'])
    unscaled=np.array([op['fnirs']@trial for trial in selected])
    train=meta['train']
    common=max(float(robust_mad(np.concatenate([trial.reshape(-1) for trial in unscaled[train]]))),1e-12)
    gauge=float(meta['projection']['observation_loading']['fnirs_common'])
    factor=gauge/common
    feature=selected*factor
    target_hb=np.array([op['fnirs']@trial for trial in feature])
    hb_sd=np.std(np.concatenate(target_hb[train]),axis=0)
    if not np.isfinite(hb_sd).all() or np.any(hb_sd<=1e-8):
        raise ValueError('fixed ROI low training variance')
    target=arrays['target'].copy();target[:,:,1:]=target_hb
    normalizer=arrays['normalizer'].copy();normalizer[1:]=hb_sd
    frozen_projection=meta['projection']
    projection=dict(frozen_projection,fnirs_pair=0,fnirs_factor=factor,inverse_fnirs_factor=1/factor,
        measurement_scale=dict(frozen_projection['measurement_scale'],fnirs_common=common),
        computational_scale=dict(frozen_projection['computational_scale'],fnirs_common=1/common),
        selection_interpretation=expected_roi['selection'])
    # No original-pair selection scores or noise estimates describe the new ROI.
    projection.pop('pair_scores',None)
    metadata={k:v for k,v in meta.items() if k not in ('projection','feature_noise','noise','normalization_sd','array_sha256',
        'legacy_full_recomposition_max_error_training_sd')}
    metadata.update(projection=projection,normalization_sd=normalizer.tolist(),
        fixed_roi=dict(expected_roi,parent_pair_index=frozen_projection['fnirs_pair'],
            parent_projection_array_sha256=meta.get('array_sha256'),
            source_preparation_identity='source_preparation_sha256_identifies_retained_fullsubject_arrays_not_this_projection',
            source_subject_json=str(source/f'{subject}.json'),source_subject_npz=str(source/f'{subject}.npz'),
            source_projection_json=str(source/f'{subject}_o{outer}_E0.json'),
            training_trials=train,common_training_MAD=common,frozen_observation_loading=gauge,
            fnirs_factor=factor,noise='not_reestimated_or_reused; objective_uses_training_SD',
            coordinate_interpretation='relative_Hb_fixed_reference_gauge_not_absolute_physiological_calibration'))
    return dict(arrays,feature_fnirs=feature,target=target,normalizer=normalizer),metadata


def synthetic_panel(cfg, tau, replicate):
    n, dt = cfg['tensor']['steps'], cfg['tensor']['dt_s']
    rng = np.random.default_rng(cfg['synthetic']['seed']+replicate*100+int(tau*10))
    design = build_shared_driver_design(parameters(cfg,tau), n, dt)
    op = native_feature_operators(n)
    eeg, hb, driver = [], [], []
    clean = []
    for i in range(24):
        r = gaussian_filter1d(rng.normal(size=n), 3.)
        r += rng.normal()*.3*np.exp(-((np.arange(n)-rng.uniform(25,70))/18)**2)
        r *= cfg['synthetic']['driver_sd']/np.std(r)
        initial = rng.normal(size=5)*cfg['synthetic']['initial_state_sd']
        coeff = np.r_[r,initial]
        canonical = (design.canonical_design@coeff).reshape(n,3)+design.canonical_offset
        e, h = canonical[:,0], op['native_interpolation']@canonical[:,1:]
        clean.append(view(e,h,cfg,'full')[0])
        noise = cfg['synthetic']['noise_fraction']*np.std(canonical,axis=0)
        eeg.append(e+rng.normal(size=n)*noise[0])
        hb.append(h+rng.normal(size=h.shape)*noise[1:])
        driver.append(r)
    train = [i for i in range(24) if i%4 != 0]
    validation = [i for i in range(24) if i%4 == 0]
    target = np.array([view(e,h,cfg,'full')[0] for e,h in zip(eeg,hb)])
    arrays = dict(feature_eeg=np.array(eeg),feature_fnirs=np.array(hb), target=target,
                  normalizer=np.std(np.concatenate(target[train]),axis=0),truth=np.array(driver),clean=np.array(clean))
    ids = [dict(session=cfg['sessions'][i//8], sample_id=f'synthetic/{i}') for i in range(24)]
    return arrays, dict(train=train,validation=validation,trials=ids,true_tau=tau,replicate=replicate)


def fit(y, design, sd, cfg, *, lower_bound=False, rcond=None):
    reg = cfg['regularization']
    return fit_shared_driver(y,design, observation_scale=sd,
        penalty=0. if lower_bound else reg['driver_second_difference_weight'],
        initial_penalty=reg['initial_state_weight'], small_signal_limit=cfg['small_signal_limit'],
        rcond=cfg['svd_rcond'] if rcond is None else rcond)


def fit_cost(result):
    return (result['weighted_data_sse']+result['penalty']*result['roughness']+
            result['initial_penalty']*np.sum(result['initial_state']**2))


def metric_row(result, target, sd, cfg, mode, identity, truth=None):
    selection = np.ones_like(target,dtype=bool)
    n = len(target)
    left, right = (n-cfg['center_steps'])//2, (n+cfg['center_steps'])//2
    if mode.startswith('center_'):
        selection[:] = False
        selection[left:right,0 if mode.startswith('center_EEG') else slice(1,3)] = True
    elif mode == 'EEG_only':
        selection[:,0] = False
    elif mode == 'fNIRS_only':
        selection[:,1:] = False
    row = dict(identity, mode=mode,status='completed',effective_df=result['effective_df'],
        physical_valid=result['physical_validity']['mathematical_valid'],
        small_signal_valid=result['physical_validity']['small_signal_valid'],
        max_excursion=result['physical_validity']['max_fractional_excursion'])
    if 'rank' in result:
        singular=result['singular_values']
        row.update(rank=result['rank'],svd_rcond=result['rcond'],
            smallest_retained_singular=float(singular[result['rank']-1]) if result['rank'] else None,
            largest_discarded_singular=float(singular[result['rank']]) if result['rank']<len(singular) else None)
    for k,name in enumerate(MODALITIES):
        error = (target[selection[:,k],k]-result['prediction'][selection[:,k],k])/sd[k]
        row['nmse_'+name] = float(np.mean(error**2)) if len(error) else None
        row['nrmse_'+name] = np.sqrt(row['nmse_'+name]) if len(error) else None
    row['driver_nrmse'] = float(np.sqrt(np.mean((truth-result['driver'])**2))/np.std(truth)) if truth is not None else None
    return row


def run_group(payload):
    cfg,out,project_root,kind,subject,outer,true_tau,replicate = payload
    group = subject+f'_o{outer}'
    path = Path(out)/'groups'/f'{group}.json'
    if path.exists():
        return json.loads(path.read_text())
    started = time.monotonic()
    rows = []
    try:
        arrays,meta = (synthetic_panel(cfg,true_tau,replicate) if kind == 'synthetic' else
                       load_measured(cfg,project_root,subject,outer))
        sd = arrays['normalizer']
        n,dt = cfg['tensor']['steps'],cfg['tensor']['dt_s']
        full_operator = view_operators(cfg,'full')[2]
        grid = tau_grid(cfg)
        designs = {float(t):build_shared_driver_design(parameters(cfg,t),n,dt,processed_mean_operator=full_operator) for t in grid}
        profiles = []
        for tau,design in designs.items():
            fits = [fit(arrays['target'][i],design,sd,cfg) for i in meta['train']]
            objective = sum(fit_cost(f) for f in fits)
            prior = (np.log(tau/cfg['reference_tau_s'])/cfg['regularization']['tau_log_sd'])**2
            profiles.append(dict(tau=tau,train_sse=sum(f['weighted_data_sse'] for f in fits),
                train_penalized=objective,source_prior_objective=objective+prior,
                train_valid=sum(f['physical_validity']['valid'] for f in fits)))
        selected = dict(fixed_tau=cfg['reference_tau_s'],
            fitted_tau=min(profiles,key=lambda r:r['train_penalized'])['tau'],
            source_prior_tau=min(profiles,key=lambda r:r['source_prior_objective'])['tau'])
        saved = dict(target=arrays['target'][meta['validation']],normalizer=sd,trial_indices=np.array(meta['validation']))
        if kind == 'synthetic':
            saved['true_driver'] = arrays['truth'][meta['validation']]
            saved['clean'] = arrays['clean'][meta['validation']]
        for method,tau in selected.items():
            for mode in cfg['modes']:
                predictions,states = [],[]
                operator = view_operators(cfg,mode)[2]
                design = build_shared_driver_design(parameters(cfg,tau),n,dt,processed_mean_operator=operator)
                for i in meta['validation']:
                    y,_,_ = view(arrays['feature_eeg'][i],arrays['feature_fnirs'][i],cfg,mode)
                    f = fit(y,design,sd,cfg)
                    # Score the complete target map, not a masked/interpolated target.
                    f['prediction'] = (full_operator@f['canonical_prediction'].ravel()).reshape(n,3)
                    identity = dict(kind=kind,group=group,subject=subject,outer=outer,trial=i,
                        session=meta['trials'][i]['session'],method=method,tau=tau,true_tau=true_tau)
                    rows.append(metric_row(f,arrays['target'][i],sd,cfg,mode,identity,
                        arrays['truth'][i] if kind=='synthetic' else None))
                    predictions.append(f['prediction']); states.append(f['states'])
                saved[f'prediction_{method}_{mode}'] = np.array(predictions)
                saved[f'states_{method}_{mode}'] = np.array(states)
        predictions,states,sensitivity = [],[],[]
        for i in meta['validation']:
            candidates = [(tau,fit(arrays['target'][i],d,sd,cfg,lower_bound=True)) for tau,d in designs.items()]
            tau,f = min(candidates,key=lambda item:item[1]['weighted_data_sse'])
            identity = dict(kind=kind,group=group,subject=subject,outer=outer,trial=i,
                session=meta['trials'][i]['session'],method='oracle_lower_bound',tau=tau,true_tau=true_tau)
            rows.append(metric_row(f,arrays['target'][i],sd,cfg,'full',identity,
                arrays['truth'][i] if kind=='synthetic' else None))
            predictions.append(f['prediction']); states.append(f['states'])
            for rcond in cfg['lower_bound_sensitivity_rconds']:
                alternatives=[(t,fit(arrays['target'][i],d,sd,cfg,lower_bound=True,rcond=rcond)) for t,d in designs.items()]
                best_t,best=min(alternatives,key=lambda item:item[1]['weighted_data_sse'])
                sensitivity.append(dict(trial=i,rcond=rcond,tau=best_t,rank=best['rank'],
                    mean_component_nmse=best['weighted_data_sse']/(3*n),
                    reference_mean_component_nmse=f['weighted_data_sse']/(3*n),
                    max_excursion=best['physical_validity']['max_fractional_excursion']))
        saved['prediction_oracle_lower_bound_full'] = np.array(predictions)
        saved['states_oracle_lower_bound_full'] = np.array(states)
        for mode in ('center_EEG','center_fNIRS'):
            predictions=[]
            for i in meta['validation']:
                y,_,mask = view(arrays['feature_eeg'][i],arrays['feature_fnirs'][i],cfg,mode)
                pred = y.copy()
                for col in range(3):
                    seen = np.flatnonzero(mask[:,col])
                    pred[:,col] = np.interp(np.arange(n),seen,y[seen,col])
                f=dict(prediction=pred,effective_df=float('nan'),physical_validity=dict(
                    mathematical_valid=False,small_signal_valid=False,max_fractional_excursion=float('nan')))
                identity=dict(kind=kind,group=group,subject=subject,outer=outer,trial=i,
                    session=meta['trials'][i]['session'],method='own_context',tau=None,true_tau=true_tau)
                rows.append(metric_row(f,arrays['target'][i],sd,cfg,mode,identity))
                predictions.append(pred)
            saved[f'prediction_own_context_{mode}']=np.array(predictions)
        directory=Path(out)/'predictions'; directory.mkdir(exist_ok=True)
        np.savez_compressed(directory/f'{group}.npz',**saved)
        result=dict(group=group,kind=kind,status='completed',selected_tau=selected,profiles=profiles,
            metadata=meta,rows=rows,lower_bound_rank_sensitivity=sensitivity,
            seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    except Exception as exc:
        result=dict(group=group,kind=kind,status='failed',error=repr(exc),traceback=traceback.format_exc(),
                    rows=rows,seconds=time.monotonic()-started)
    write_json(path,result)
    return result


def tasks(cfg,out,root,kind):
    if kind == 'synthetic':
        return [(cfg,str(out),str(root),kind,f'synthetic_t{t:g}_r{r}',0,t,r)
            for t in cfg['synthetic']['taus_s'] for r in range(cfg['synthetic']['replicates'])]
    return [(cfg,str(out),str(root),kind,s,o,None,None) for s in cfg['subjects'] for o in cfg['outer_folds']]


def summarize(out, expected):
    records=[json.loads(p.read_text()) for p in sorted((out/'groups').glob('*.json'))]
    rows=[row for r in records for row in r['rows']]
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'metrics.csv',index=False)
    summary=dict(expected_groups=expected,completed_groups=sum(r['status']=='completed' for r in records),
        failed_groups=sum(r['status']=='failed' for r in records),records=len(rows),aggregates=[],
        expected_records=expected*6*(3*7+1+2),
        aggregation_scope='completed rows including retained partial failures; group and row completeness required for conclusions')
    if len(frame):
        fields=[f'{prefix}_{m}' for prefix in ('nmse','nrmse') for m in MODALITIES]
        for (kind,method,mode), group in frame.groupby(['kind','method','mode']):
            # Trial -> session -> subject equal; folds partition validation trials.
            means=group.groupby(['subject','session'])[fields].mean().groupby('subject').mean().mean()
            values=means.to_dict()
            for modality in MODALITIES:
                values['mean_trial_nrmse_'+modality]=values['nrmse_'+modality]
                values['nrmse_'+modality]=np.sqrt(values['nmse_'+modality])
            summary['aggregates'].append(dict(kind=kind,method=method,mode=mode,count=len(group),
                physical_valid=int(group.physical_valid.sum()),small_signal_valid=int(group.small_signal_valid.sum()),
                **values))
    write_json(out/'summary.json',summary)
    return summary


def replay_group(payload):
    cfg,parent_cfg,out,parent,group=payload
    destination=Path(out)/'groups'/f'{group}.json'
    if destination.exists():
        return json.loads(destination.read_text())
    started=time.monotonic()
    original=json.loads((Path(parent)/'groups'/f'{group}.json').read_text())
    if original['status']!='completed' or original['group']!=group:
        raise ValueError('replay requires a completed exact source group')
    source_rows=[r for r in original['rows'] if r['mode']=='full']
    with np.load(Path(parent)/'predictions'/f'{group}.npz',allow_pickle=False) as a:
        data={k:a[k] for k in a.files if k in ('target','normalizer','trial_indices') or k.endswith('_full')}
    if data['target'].shape!=(6,120,3) or data['normalizer'].shape!=(3,):
        raise ValueError('source prediction tensor mismatch')
    expected_trials=np.asarray(original['metadata']['validation'])
    if (data['trial_indices'].shape!=(6,) or len(set(data['trial_indices'].tolist()))!=6
            or not np.array_equal(data['trial_indices'],expected_trials)
            or not np.isfinite(data['target']).all() or not np.isfinite(data['normalizer']).all()
            or np.any(data['normalizer']<=1e-8)):
        raise ValueError('source identity/normalizer/finite contract failed')
    for method in cfg['methods']:
        for key,shape in ((f'states_{method}_full',(6,120,6)),(f'prediction_{method}_full',(6,120,3))):
            if data[key].shape!=shape or not np.isfinite(data[key]).all():
                raise ValueError('source model tensor/finite contract failed')
    saved={k:data[k] for k in ('target','normalizer','trial_indices')}
    operator=view_operators(parent_cfg,'full')[2]
    rows=[]
    for method in cfg['methods']:
        predictions=[]; states=[]
        for index,trial in enumerate(data['trial_indices']):
            matched=[r for r in source_rows if r['method']==method and r['trial']==int(trial)]
            if len(matched)!=1: raise ValueError('source identity is not unique')
            source=matched[0]
            row={k:source[k] for k in ('kind','subject','outer','trial','session','method','tau','true_tau')}
            row.update(group=group,mode='full',status='failed_domain')
            z=data[f'states_{method}_full'][index]
            runs=[replay_nonlinear_driver(z[:,0],z[0,1:],parameters(parent_cfg,source['tau']),.25,substeps=s)
                  for s in (cfg['substeps'],cfg['integration_check_substeps'])]
            row['checks']=[dict(status=r['status'],checks=r['checks'],failure=r.get('failure')) for r in runs]
            row['physical_valid']=all(r['status']=='completed' for r in runs)
            pred=np.full((120,3),np.nan); state=np.full((120,6),np.nan)
            for name in MODALITIES:
                row['linear_nrmse_'+name]=source['nrmse_'+name]
                row['linear_nmse_'+name]=source['nmse_'+name]
            if row['physical_valid']:
                predictions_by_resolution=[(operator@r['canonical_prediction'].ravel()).reshape(120,3) for r in runs]
                delta=np.max(abs(predictions_by_resolution[0]-predictions_by_resolution[1])/data['normalizer'])
                row['integration_max_difference_training_sd']=float(delta)
                row['status']='completed' if delta<=cfg['maximum_integration_difference_training_sd'] else 'failed_integration_check'
                pred=predictions_by_resolution[0];state=runs[0]['states']
                linear=data[f'prediction_{method}_full'][index]
                for col,name in enumerate(MODALITIES):
                    row['nmse_'+name]=float(np.mean(((pred[:,col]-data['target'][index,:,col])/data['normalizer'][col])**2))
                    row['nrmse_'+name]=np.sqrt(row['nmse_'+name])
                    row['replay_gap_nrmse_'+name]=float(np.sqrt(np.mean(((pred[:,col]-linear[:,col])/data['normalizer'][col])**2)))
            row['seconds']=time.monotonic()-started
            rows.append(row);predictions.append(pred);states.append(state)
        saved[f'prediction_{method}_full']=np.array(predictions)
        saved[f'states_{method}_full']=np.array(states)
    directory=Path(out)/'predictions';directory.mkdir(exist_ok=True)
    np.savez_compressed(directory/f'{group}.npz',**saved)
    result=dict(group=group,kind=original['kind'],status='completed',rows=rows,
        metadata=original['metadata'],selected_tau=original['selected_tau'],profiles=[],seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_json(destination,result)
    return result


def replay_main(args):
    cfg=yaml.safe_load(args.config.read_text())
    if (cfg['schema']!='shared_driver_nonlinear_replay_v1' or
        cfg['source_run']!='experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_linear_screen_v1' or
        cfg['protected_boundary']!='only_exact_parent_prediction_arrays_no_raw_or_protected_loaders' or
        cfg['substeps']!=4 or cfg['integration_check_substeps']!=8 or
        cfg['methods']!=['fixed_tau','fitted_tau','source_prior_tau','oracle_lower_bound']):
        raise ValueError('replay contract differs from frozen scope')
    parent=(args.project_root/cfg['source_run']).resolve()
    if args.replay_of.resolve()!=parent:
        raise ValueError('replay-of differs from exact source contract')
    if args.check_only:
        r=replay_nonlinear_driver(np.zeros(12),np.array([0.,1.,1.,1.,1.]),BalloonParameters(),.25)
        assert r['status']=='completed'
        print(json.dumps(dict(status='passed',source_arrays_read=0)))
        return
    if args.run_dir is None or not 1<=args.workers<=cfg['resources']['max_workers']:
        raise ValueError('run directory and bounded worker count required')
    manifest_source=json.loads((parent/'manifest.json').read_text())
    if manifest_source['execution']!='completed': raise ValueError('parent is incomplete')
    parent_cfg=read_config(parent/'resolved_config.yaml')
    out=args.run_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())['execution']=='completed':
        raise ValueError('completed evidence is immutable')
    resolved=out/'resolved_config.yaml'
    if resolved.exists() and yaml.safe_load(resolved.read_text())!=cfg: raise ValueError('replay resume config mismatch')
    if not resolved.exists():resolved.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    manifest=dict(experiment_id=cfg['experiment_id'],execution='running',source_run=str(parent),pilot=args.pilot,
        controller_pid=os.getpid(),supervisor=os.environ.get('SSM_SYSTEMD_UNIT'),workers=args.workers,
        started_at=datetime.now(timezone.utc).isoformat(),source_root=str(CODE_ROOT),
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    write_json(out/'manifest.json',manifest);(out/'groups').mkdir(exist_ok=True)
    all_rows=[];group_results=[]
    for kind in ('synthetic','measured'):
        if args.pilot and kind=='measured':break
        group_names=[t[4]+f'_o{t[5]}' for t in tasks(parent_cfg,out,args.project_root,kind)]
        if args.pilot:group_names=group_names[:1]
        with ProcessPoolExecutor(max_workers=min(args.workers,len(group_names))) as pool:
            futures={pool.submit(replay_group,(cfg,parent_cfg,str(out),str(parent),g)):g for g in group_names}
            for future in as_completed(futures):
                try:
                    result=future.result()
                except Exception as exc:
                    result=dict(group=futures[future],kind=kind,status='failed',rows=[],seconds=None,
                        error=repr(exc),traceback=traceback.format_exc())
                    write_json(out/'groups'/f"{result['group']}.json",result)
                all_rows.extend(result['rows']);group_results.append(result)
                print(json.dumps(dict(group=result['group'],seconds=result['seconds'],
                    valid=sum(r['status']=='completed' for r in result['rows']),expected=24)),flush=True)
        frame=pd.DataFrame(all_rows);frame.to_csv(out/'metrics.csv',index=False)
        summary=dict(expected_groups=1 if args.pilot else 24,completed_groups=sum(r['status']=='completed' for r in group_results),
            failed_groups=sum(r['status']=='failed' for r in group_results),
            expected_method_trials=24 if args.pilot else 576,completed_method_trials=sum(r['status']=='completed' for r in all_rows),
            aggregates=[],interpretation='all metrics conditional on successful nonlinear and resolution checks; failures remain in fixed denominator')
        for (k,m),group in frame.groupby(['kind','method']) if len(frame) else []:
            good=group[group.status=='completed']
            result=dict(kind=k,method=m,expected=6 if args.pilot else 72,observed_rows=len(group),
                        completed=len(good),failure_counts=group.status.value_counts().to_dict())
            if len(good):
                fields=[prefix+name for prefix in ('nmse_','linear_nmse_') for name in MODALITIES]
                averages=good.groupby(['subject','session'])[fields].mean().groupby('subject').mean().mean()
                for name in MODALITIES:
                    result['nrmse_'+name]=float(np.sqrt(averages['nmse_'+name]))
                    result['linear_nrmse_same_subset_'+name]=float(np.sqrt(averages['linear_nmse_'+name]))
            summary['aggregates'].append(result)
        write_json(out/'summary.json',summary)
    manifest.update(execution='completed',completed_at=datetime.now(timezone.utc).isoformat(),summary='summary.json',
        scientific_verdict='replay_diagnostic_only')
    write_json(out/'manifest.json',manifest)


NONLINEAR_METHODS = ['fixed_tau_100','fixed_tau_0p01','linear_trained_tau_0p01','linear_prior_tau_0p01']


def read_nonlinear_fit_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if (cfg['schema']!='shared_driver_nonlinear_fit_v1'
            or cfg['source_run']!='experiments/runs/physiology_semantic_tokenizer/ssm_overnight/20260917_measurement_retest_v1'
            or cfg['protected_boundary']!='only_exact_parent_prepared_arrays_no_raw_or_protected_loaders'
            or cfg['subjects']!=['subject_01','subject_09','subject_18']
            or cfg['outer_folds']!=[0,1,2,3]
            or cfg['sessions']!=['session_01','session_03','session_05']
            or cfg['original_positions']!=[0,1,2,3,5,6,7,8]
            or cfg['tensor']!=dict(trials_per_subject=24,steps=120,components=list(MODALITIES),dt_s=.25)
            or cfg['methods']!=NONLINEAR_METHODS or cfg['modes']!=['full','center_EEG','center_fNIRS']
            or cfg['center_steps']!=16):
        raise ValueError('nonlinear fit contract/scope mismatch')
    if (cfg['synthetic']['taus_s']!=[1.,2.,4.] or cfg['synthetic']['spectra']!=['slow','mixed']
            or cfg['synthetic']['replicates']!=2 or cfg['synthetic']['fast_fraction']!=.7
            or cfg['regularization']['linear_profile_weight']!=.01
            or cfg['regularization']['initial_state_weight']!=100.
            or cfg['regularization']['tau_log_sd']!=.05
            or cfg['nonlinear']['substeps']!=4 or cfg['nonlinear']['integration_check_substeps']!=8):
        raise ValueError('nonlinear numerical/synthetic contract mismatch')
    resource_cfg=cfg['resources']
    if not 1<=resource_cfg['max_workers']<=48 or resource_cfg['numerical_threads_per_worker']!=1:
        raise ValueError('nonlinear resource limit')
    if not 1<=resource_cfg['max_in_flight']<=2*resource_cfg['max_workers']:
        raise ValueError('invalid in-flight bound')
    options=cfg['nonlinear'];grid=cfg['tau_grid']
    if (cfg['reference_tau_s']!=2. or not 0<grid['minimum']<2.<grid['maximum']
            or type(grid['log_points']) is not int or grid['log_points']<3
            or type(options['max_evaluations']) is not int or options['max_evaluations']<1
            or not 0<cfg['svd_rcond']<1):
        raise ValueError('invalid parameter grid or optimization budget')
    positive=[options['gradient_tolerance'],options['maximum_integration_difference_training_sd'],
        cfg['synthetic']['driver_sd'],cfg['synthetic']['initial_state_sd'],cfg['synthetic']['noise_fraction']]
    if not np.isfinite(positive).all() or np.any(np.asarray(positive)<=0):
        raise ValueError('nonlinear numerical scales must be positive finite')
    parameters(cfg,cfg['reference_tau_s']).validate()
    return cfg


def nonlinear_group_specs(cfg,kind):
    if kind=='measured':
        return [dict(kind=kind,group=f'{subject}_o{outer}',subject=subject,outer=outer)
                for subject in cfg['subjects'] for outer in cfg['outer_folds']]
    return [dict(kind=kind,group=f'synthetic_tau{tau:g}_{spectrum}_r{rep}',
                 subject=f'synthetic_tau{tau:g}_{spectrum}_r{rep}',outer=0,true_tau=float(tau),
                 spectrum=spectrum,replicate=rep)
            for tau in cfg['synthetic']['taus_s'] for spectrum in cfg['synthetic']['spectra']
            for rep in range(cfg['synthetic']['replicates'])]


def nonlinear_synthetic_panel(cfg,spec):
    n,dt=cfg['tensor']['steps'],cfg['tensor']['dt_s']
    syn=cfg['synthetic'];tau=spec['true_tau']
    conditional=cfg['schema']=='shared_driver_conditional_optical_gain_v1'
    tau_seed=0 if cfg['schema'] in ('shared_driver_fixed_roi_tau_v1','shared_driver_fixed_roi_initial_v1','shared_driver_fixed_roi_optical_v1','shared_driver_conditional_optical_gain_v1') else int(tau*1000)
    rng=np.random.default_rng(syn['seed']+tau_seed+100*spec['replicate']+int(spec['spectrum']=='mixed'))
    op=native_feature_operators(n);p=parameters(cfg,tau)
    if conditional:
        from src.inference.observation_baselines import conditional_optical_hb_mapping
        mapping=conditional_optical_hb_mapping(syn['mapping_fnirs_factor'])
        p=replace(p,fixed=replace(p.fixed,neurovascular_gain=spec['true_relative_gain']*mapping['beta_reference']))
    features_eeg=[];features_hb=[];truth=[];truth_states=[];clean=[]
    initial_only=[];driver_only=[]
    for trial in range(24):
        innovations=rng.normal(size=n);slow=np.zeros(n)
        for t in range(1,n):slow[t]=.93*slow[t-1]+innovations[t]
        slow=gaussian_filter1d(slow,3.)
        slow+=rng.normal()*.3*np.exp(-((np.arange(n)-rng.uniform(25,70))/18.)**2)
        slow/=max(np.std(slow),1e-12)
        driver=slow
        if spec['spectrum']=='mixed':
            fast=gaussian_filter1d(rng.normal(size=n),.55)
            fast/=max(np.std(fast),1e-12)
            driver=np.sqrt(1.-syn['fast_fraction'])*slow+np.sqrt(syn['fast_fraction'])*fast
        driver=driver*syn['driver_sd']/np.std(driver)
        initial=np.r_[0.,np.ones(4)]+rng.normal(size=5)*syn['initial_state_sd']
        if cfg['schema']=='shared_driver_fixed_roi_initial_v1':
            initial[3]=initial[2]*np.exp(spec['true_initial_log_p_over_v'])
        generated=replay_nonlinear_driver(driver,initial,p,dt,substeps=cfg['nonlinear']['integration_check_substeps'])
        if generated['status']!='completed':
            raise ValueError(f'synthetic generator domain failure trial={trial}: {generated.get("failure")}')
        canonical=generated['canonical_prediction'].copy()
        if conditional:canonical[:,1:]=canonical[:,1:]@mapping['matrix'].T
        eeg=canonical[:,0];hb=op['native_interpolation']@canonical[:,1:]
        clean.append(view(eeg,hb,cfg,'full')[0]);truth.append(driver);truth_states.append(generated['states'])
        if conditional:
            for destination,probe_driver,probe_initial in [(initial_only,np.zeros(n),initial),(driver_only,driver,np.r_[0.,np.ones(4)])]:
                probe=replay_nonlinear_driver(probe_driver,probe_initial,p,dt,substeps=cfg['nonlinear']['integration_check_substeps'])
                if probe['status']!='completed':raise ValueError('conditional truth component domain failure')
                native=probe['canonical_prediction'].copy();native[:,1:]=native[:,1:]@mapping['matrix'].T
                destination.append(view(native[:,0],op['native_interpolation']@native[:,1:],cfg,'full')[0])
        noise=syn['noise_fraction']*np.std(canonical,axis=0)
        features_eeg.append(eeg+rng.normal(size=n)*noise[0])
        features_hb.append(hb+rng.normal(size=hb.shape)*noise[1:])
    train=[i for i in range(24) if i%4!=0];validation=[i for i in range(24) if i%4==0]
    target=np.array([view(e,h,cfg,'full')[0] for e,h in zip(features_eeg,features_hb)])
    arrays=dict(feature_eeg=np.array(features_eeg),feature_fnirs=np.array(features_hb),target=target,
        normalizer=np.std(np.concatenate(target[train]),axis=0),truth=np.array(truth),
        truth_states=np.array(truth_states),clean=np.array(clean))
    if np.any(arrays['normalizer']<=1e-8):raise ValueError('synthetic low training variance')
    trials=[dict(session=cfg['sessions'][i//8],sample_id=f'{spec["group"]}/{i}') for i in range(24)]
    meta=dict(train=train,validation=validation,trials=trials,**spec)
    if conditional:
        arrays.update(truth_initial_only_clean=np.array(initial_only),truth_driver_only_clean=np.array(driver_only))
        meta['truth_component_diagnostic']=dict(interpretation='separate_nonlinear_forward_probes_not_additive_variance_explanation',
            full_hb_rms=np.sqrt(np.mean(arrays['clean'][:,:,1:]**2,axis=(0,1))),
            initial_only_hb_rms=np.sqrt(np.mean(arrays['truth_initial_only_clean'][:,:,1:]**2,axis=(0,1))),
            driver_only_hb_rms=np.sqrt(np.mean(arrays['truth_driver_only_clean'][:,:,1:]**2,axis=(0,1))))
    return arrays,meta


def prepare_nonlinear_group(payload):
    cfg,out,project_root,spec,pilot=payload
    out=Path(out);path=out/'prepared'/f'{spec["group"]}.json'
    if path.exists():return json.loads(path.read_text())
    started=time.monotonic()
    try:
        arrays,meta=(nonlinear_synthetic_panel(cfg,spec) if spec['kind']=='synthetic'
                     else load_measured(cfg,project_root,spec['subject'],spec['outer']))
        profiles=[];selected={m:cfg['reference_tau_s'] for m in NONLINEAR_METHODS}
        if not pilot:
            full=view_operators(cfg,'full')[2]
            for tau in tau_grid(cfg):
                design=build_shared_driver_design(parameters(cfg,tau),cfg['tensor']['steps'],cfg['tensor']['dt_s'],processed_mean_operator=full)
                fits=[fit_shared_driver(arrays['target'][i],design,observation_scale=arrays['normalizer'],
                       penalty=cfg['regularization']['linear_profile_weight'],initial_penalty=100.,
                       small_signal_limit=cfg['small_signal_limit'],rcond=cfg['svd_rcond']) for i in meta['train']]
                objective=sum(fit_cost(f) for f in fits)
                prior=(np.log(tau/cfg['reference_tau_s'])/cfg['regularization']['tau_log_sd'])**2
                profiles.append(dict(tau=tau,train_sse=sum(f['weighted_data_sse'] for f in fits),
                    train_penalized=objective,source_prior_objective=objective+prior))
            selected['linear_trained_tau_0p01']=min(profiles,key=lambda r:r['train_penalized'])['tau']
            selected['linear_prior_tau_0p01']=min(profiles,key=lambda r:r['source_prior_objective'])['tau']
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.npz.tmp')
        with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
        tmp.replace(path.with_suffix('.npz'))
        result=dict(status='completed',spec=spec,metadata=meta,profiles=profiles,selected_tau=selected,seconds=time.monotonic()-started)
    except Exception as exc:
        result=dict(status='failed_preparation',spec=spec,error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started)
    write_json(path,result)
    return result


def nonlinear_cell_id(spec,method,mode):
    return f'{spec["group"]}__{method}__{mode}'


def nonlinear_cell(payload):
    from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_driver
    from src.inference.t3a_balloon_robust_ssm import run_physical_checks
    cfg,out,spec,method,mode,pilot=payload
    started=time.monotonic();out=Path(out);identifier=nonlinear_cell_id(spec,method,mode)
    destination=out/'cells'/identifier;path=destination/'result.json'
    if path.exists():return json.loads(path.read_text())
    prepared=json.loads((out/'prepared'/f'{spec["group"]}.json').read_text())
    if 'continuation' in cfg and (spec['kind']=='synthetic' or method.startswith('fixed_gain_')):
        parent=Path(prepared['continuation']['parent_record']).parent.parent
        return inherit_gain_evidence(parent/'cells'/identifier/'result.json',path,action='inherited_validation')
    fallback_validation=[i for i in range(24) if i%4==spec['outer']]
    validation=prepared.get('metadata',{}).get('validation',fallback_validation)
    if pilot:validation=validation[:2]
    rows=[];attempt_results=[];predictions=[];states=[]
    gain_branch=cfg['schema']=='shared_driver_gain_prior_v1'
    initial_branch=cfg['schema']=='shared_driver_fixed_roi_initial_v1'
    conditional_branch=cfg['schema']=='shared_driver_conditional_optical_gain_v1'
    optical_branch=cfg['schema']=='shared_driver_fixed_roi_optical_v1'
    flow_branch=conditional_branch or initial_branch or optical_branch or cfg['schema']=='shared_driver_fixed_roi_flow_v1'
    tau_branch=flow_branch or cfg['schema']=='shared_driver_fixed_roi_tau_v1'
    training=None;gain=1.;amplitude_weight=0.
    inherited_method=(conditional_branch and spec['kind']=='measured' and method=='fixed_tau_logflow1') or (optical_branch and spec['kind']=='synthetic' and not pilot) or (initial_branch and spec['kind']=='measured' and method in FIXED_ROI_INITIAL_FREE) or (
        flow_branch and not initial_branch and not optical_branch and method in FIXED_ROI_FLOW_INHERITED)
    if inherited_method:
        parent=Path(prepared['inheritance']['parent_record']).parent.parent
        return inherit_flow_evidence(parent/'cells'/identifier/'result.json',path,arrays_name='trajectories.npz')
    if tau_branch and method not in ('fixed_roi_fixed_tau','fixed_tau_logflow1','fixed_tau_logflow1_tied','conditional_fixed_gain'):
        selection=out/'training'/f"{spec['group']}__{method}"/'selection.json'
        if not selection.exists():raise ValueError('tau training must be terminal before dependent validation')
        training=json.loads(selection.read_text())
    if gain_branch:
        amplitude_weight=0. if method.endswith('_no_amplitude') else 1.
        if method.startswith('trained_gain_'):
            selection=out/'training'/f"{spec['group']}__amplitude{int(amplitude_weight)}"/'selection.json'
            if not selection.exists():raise ValueError('gain training must be terminal before dependent validation')
            training=json.loads(selection.read_text())
            if training['status']=='completed':gain=float(training['parameter_value'])
    inference_ready=prepared['status']=='completed' and (training is None or training['status']=='completed')
    if prepared['status']=='completed':
        with np.load(out/'prepared'/f'{spec["group"]}.npz',allow_pickle=False) as archive:
            arrays={k:archive[k] for k in archive.files}
        meta=prepared['metadata'];tau=(training['parameter_value'] if tau_branch and not conditional_branch and training is not None and inference_ready
                                      else cfg['reference_tau_s'] if flow_branch else prepared['selected_tau'][method]);p=parameters(cfg,tau)
        if conditional_branch:
            gain=(1. if method=='fixed_tau_logflow1' else training['parameter_value'] if training is not None and inference_ready else prepared['conditional_mapping']['beta_reference'])
        if gain_branch or conditional_branch:p=replace(p,fixed=replace(p.fixed,neurovascular_gain=gain))
        penalty=100. if method=='fixed_tau_100' else .01
        driver_prior_sd=float(arrays['normalizer'][0]/p.fixed.eeg_loading)
        full_operator=view_operators(cfg,'full')[2];n=cfg['tensor']['steps'];dt=cfg['tensor']['dt_s']
        fit_operator=view_operators(cfg,mode)[2]
        if conditional_branch:
            full_operator=conditional_prediction_operator(full_operator,prepared['conditional_mapping'],method)
            fit_operator=conditional_prediction_operator(fit_operator,prepared['conditional_mapping'],method)
        design=build_shared_driver_design(p,n,dt,processed_mean_operator=fit_operator)
    for trial in validation:
        identity=dict(spec,method=method,mode=mode,trial=trial,session=cfg['sessions'][trial//8])
        row=dict(identity,status='failed_preparation',converged=False)
        if flow_branch:row.update(**flow_options(cfg,method),flow_prior_cost=None)
        if conditional_branch and prepared['status']=='completed':
            ref=prepared['conditional_mapping']['beta_reference']
            row.update(beta_reference=ref,relative_gain=gain/ref if inference_ready else None,mean_mapping='identity' if method=='fixed_tau_logflow1' else 'conditional_B')
        if initial_branch:row['tie_total_hb_to_volume']=method.endswith('_tied')
        trial_result=dict(trial=trial,status='failed_preparation',error=prepared.get('error'))
        if gain_branch or tau_branch:
            row.update(tau=cfg['reference_tau_s'],neurovascular_gain=gain if inference_ready else None,
                driver_amplitude_weight=amplitude_weight,training_status=None if training is None else training['status'])
            if prepared['status']=='completed' and not inference_ready:
                row['status']='failed_training'
                trial_result.update(status='failed_training',training=training)
        prediction=np.full((cfg['tensor']['steps'],3),np.nan);state=np.full((cfg['tensor']['steps'],6),np.nan)
        try:
            if inference_ready:
                y,mean,visible=view(arrays['feature_eeg'][trial],arrays['feature_fnirs'][trial],cfg,mode)
                if conditional_branch:mean=conditional_prediction_operator(mean,prepared['conditional_mapping'],method)
                constraint_options=dict(tie_total_hb_to_volume=method.endswith('_tied')) if initial_branch else {}
                linear=fit_shared_driver(y,design,observation_scale=arrays['normalizer'],penalty=penalty,
                    initial_penalty=100.,small_signal_limit=cfg['small_signal_limit'],rcond=cfg['svd_rcond'],**constraint_options)
                initial=linear['states'][0,1:]
                checks=run_physical_checks(np.r_[linear['driver'][0],initial][None,:],p)
                required=('finite','positive_fvpq','oxygen_extraction_in_unit_interval','absolute_hb_nonnegative','hbr_not_above_hbt','rest_equilibrium')
                legal=all(checks[k] for k in required)
                starts=[]
                if legal:starts.append(dict(driver=linear['driver'],initial_state=initial))
                starts.append(dict(driver=np.zeros(n),initial_state=np.r_[0.,np.ones(4)]))
                options=cfg['nonlinear']
                amplitude_options=(dict(driver_amplitude_weight=amplitude_weight,driver_prior_sd=driver_prior_sd)
                                   if gain_branch else {})
                if flow_branch:amplitude_options.update(flow_options(cfg,method))
                if conditional_branch:amplitude_options.update(initial_coordinates='independent_logs',tie_total_hb_to_volume=False)
                if initial_branch:amplitude_options.update(constraint_options)
                result=fit_nonlinear_shared_driver(y,p,dt,mean_operator=mean,sd=arrays['normalizer'],
                    penalty=penalty,initial_penalty=100.,starts=starts,visible=visible,
                    max_evaluations=options['max_evaluations'],substeps=options['substeps'],gradient_tolerance=options['gradient_tolerance'],**amplitude_options)
                row.update(tau=tau,penalty=penalty,status=result['status'],converged=result['converged'],
                    linear_start_valid=legal,evaluations=result.get('evaluations'),selected_start=result.get('selected_start'),
                    objective=result.get('objective'),gradient_inf_norm=result.get('gradient_inf_norm'),
                    scaled_gradient_inf_norm=result.get('scaled_gradient_inf_norm'),rejected_steps=result.get('rejected_steps'))
                if flow_branch:
                    row.update(**flow_options(cfg,method),flow_prior_cost=result.get('flow_prior_cost'))
                if gain_branch:
                    row.update(driver_prior_sd=driver_prior_sd,driver_amplitude_cost=result.get('driver_amplitude_cost'))
                physical=result.get('physical_checks',{})
                row['physical_valid']=bool(physical.get('all_intermediate_valid',False)
                    and all(physical.get('completed_states',{}).get(k,False) for k in required))
                if 'states' in result:
                    state=result['states'];prediction=(full_operator@result['canonical_prediction'].ravel()).reshape(n,3)
                    fine=replay_nonlinear_driver(result['driver'],result['initial_state'],p,dt,substeps=options['integration_check_substeps'])
                    row['fine_physical_valid']=fine['status']=='completed'
                    row['physical_valid']=row['physical_valid'] and row['fine_physical_valid']
                    gap=float('inf')
                    if fine['status']=='completed':
                        fine_prediction=(full_operator@fine['canonical_prediction'].ravel()).reshape(n,3)
                        gap=float(np.max(abs(fine_prediction-prediction)/arrays['normalizer']))
                    row['integration_max_difference_training_sd']=gap
                    if row['status']=='completed' and gap>options['maximum_integration_difference_training_sd']:
                        row.update(status='failed_integration_check',converged=False)
                    result['integration_check']=dict(status=fine['status'],difference_training_sd=gap,failure=fine.get('failure'),
                        canonical_max_abs_difference=(float(np.max(abs(fine['canonical_prediction']-result['canonical_prediction'])))
                            if fine['status']=='completed' else None),
                        state_max_abs_difference=(np.max(abs(fine['states']-result['states']),axis=0)
                            if fine['status']=='completed' else None),state_order=['r','s','f','v','p','q'],
                        interpretation='diagnostic_only_state_and_canonical_differences_no_additional_gate')
                    left,right=(n-cfg['center_steps'])//2,(n+cfg['center_steps'])//2
                    selection=np.ones((n,3),bool)
                    if mode!='full':
                        selection[:]=False;selection[left:right,0 if mode=='center_EEG' else slice(1,3)]=True
                    success=row['status']=='completed' and row['converged']
                    for j,name in enumerate(MODALITIES):
                        error=(prediction[selection[:,j],j]-arrays['target'][trial,selection[:,j],j])/arrays['normalizer'][j]
                        nmse=float(np.mean(error**2)) if len(error) else None
                        row['last_valid_nmse_'+name]=nmse
                        row['nmse_'+name]=nmse if success else None
                        row['nrmse_'+name]=float(np.sqrt(nmse)) if success and nmse is not None else None
                    if spec['kind']=='synthetic':
                        driver_error=float(np.sqrt(np.mean((result['driver']-arrays['truth'][trial])**2))/np.std(arrays['truth'][trial]))
                        row['last_valid_driver_nrmse']=driver_error
                        row['driver_nrmse']=driver_error if success else None
                trial_result={k:v for k,v in result.items() if k not in ('prediction','canonical_prediction','states','driver')}
                trial_result.update(trial=trial,scoring_status=row['status'],linear_start=dict(valid=legal,physical_initial=initial,checks=checks,
                    disposition='passed_as_first_start' if legal else 'invalid_linear_start_excluded_without_clipping'),
                    start_order=(['linear','rest'] if legal else ['rest']))
                if initial_branch:
                    trial_result['linear_start']['coordinate_fit']=('reduced_SVD_with_p0_equals_v0_no_projection'
                        if method.endswith('_tied') else 'unconstrained_linear_SVD')
        except Exception as exc:
            row.update(status='failed_exception',converged=False,error=repr(exc))
            trial_result.update(status='failed_exception',error=repr(exc),traceback=traceback.format_exc())
        rows.append(row);attempt_results.append(trial_result);predictions.append(prediction);states.append(state)
    destination.mkdir(parents=True,exist_ok=True)
    saved=dict(prediction=np.array(predictions),states=np.array(states),trial_indices=np.array(validation))
    if prepared['status']=='completed':
        saved.update(target=arrays['target'][validation],normalizer=arrays['normalizer'])
        if spec['kind']=='synthetic':saved.update(truth=arrays['truth'][validation],truth_states=arrays['truth_states'][validation])
    with (destination/'trajectories.npz.tmp').open('wb') as stream:np.savez_compressed(stream,**saved)
    (destination/'trajectories.npz.tmp').replace(destination/'trajectories.npz')
    result=dict(cell=identifier,spec=spec,method=method,mode=mode,status='completed',rows=rows,trial_results=attempt_results,
        seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_json(path,result)
    return result


def bounded_nonlinear_work(worker,payloads,max_workers,max_in_flight):
    """Submit only a bounded number of tasks; failures become parent evidence."""
    iterator=iter(payloads)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        pending={}
        def fill():
            while len(pending)<max_in_flight:
                try:payload=next(iterator)
                except StopIteration:return
                pending[pool.submit(worker,payload)]=payload
        fill()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                payload=pending.pop(future)
                try:yield payload,future.result(),None
                except Exception as exc:yield payload,None,repr(exc)
            fill()


def summarize_nonlinear_fit(out,plan,pilot,methods=None):
    rows=[];terminal=0
    for spec,method,mode in plan:
        path=out/'cells'/nonlinear_cell_id(spec,method,mode)/'result.json'
        if path.exists():
            result=json.loads(path.read_text());rows.extend(result['rows']);terminal+=1
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics.csv',index=False)
    summary=dict(expected_cells=len(plan),terminal_cells=terminal,expected_rows=len(plan)*(2 if pilot else 6),
        observed_rows=len(rows),completed_rows=sum(r['status']=='completed' and r.get('converged',False) for r in rows),aggregates=[],
        interpretation='errors conditional on each arm successful subset; use paired common-success records before cross-arm improvement claims')
    for kind in ('synthetic','measured'):
        for method in (NONLINEAR_METHODS if methods is None else methods):
            for mode in ('full','center_EEG','center_fNIRS'):
                pipelines=list(dict.fromkeys(s.get('optical_pipeline') for s,m,v in plan if s['kind']==kind and m==method and v==mode))
                for pipeline in pipelines:
                    expected=sum(s['kind']==kind and m==method and v==mode and s.get('optical_pipeline')==pipeline for s,m,v in plan)*(2 if pilot else 6)
                    if not expected:continue
                    group=frame[(frame.kind==kind)&(frame.method==method)&(frame['mode']==mode)] if len(frame) else frame
                    if pipeline is not None and len(group):group=group[group.optical_pipeline==pipeline]
                    good=group[(group.status=='completed')&group.converged.eq(True)] if len(group) else group
                    aggregate=dict(kind=kind,method=method,mode=mode,expected=expected,observed_rows=len(group),completed=len(good),
                        failure_counts=group.status.value_counts().to_dict() if len(group) else {},missing=expected-len(group))
                    if pipeline is not None:aggregate['optical_pipeline']=pipeline
                    if len(good):
                        fields=[f'nmse_{c}' for c in MODALITIES if f'nmse_{c}' in good]
                        means=good.groupby(['subject','session'])[fields].mean().groupby('subject').mean().mean()
                        aggregate.update({key:float(value) for key,value in means.items()})
                        aggregate.update({key.replace('nmse_','nrmse_'):float(np.sqrt(value)) for key,value in means.items()})
                        if 'driver_nrmse' in good:aggregate['mean_driver_nrmse']=float(good.driver_nrmse.mean())
                    summary['aggregates'].append(aggregate)
    write_json(out/'summary.json',summary)
    return summary


def nonlinear_fit_main(args):
    cfg=read_nonlinear_fit_config(args.config)
    if args.check_only:
        from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_driver
        assert callable(fit_nonlinear_shared_driver)
        for mode in cfg['modes']:
            assert view_operators(cfg,mode)[2].shape==(360,360)
        assert len(nonlinear_group_specs(cfg,'synthetic'))==12 and len(nonlinear_group_specs(cfg,'measured'))==12
        print(json.dumps(dict(status='passed',source_arrays_read=0,synthetic_groups=12,measured_groups=12)))
        return
    if args.run_dir is None or not 1<=args.workers<=cfg['resources']['max_workers']:
        raise ValueError('run-dir and bounded workers required')
    if args.pilot and args.phase=='measured':raise ValueError('pilot is synthetic only')
    out=args.run_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest_path=out/'manifest.json';resolved=out/'resolved_config.yaml'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['execution']=='completed':raise ValueError('completed evidence is immutable')
        if (old['pilot']!=args.pilot or old['source_root']!=str(CODE_ROOT)
                or old['project_root']!=str(args.project_root.resolve())):raise ValueError('resume source/project/pilot mismatch')
    else:old={}
    if resolved.exists() and yaml.safe_load(resolved.read_text())!=cfg:raise ValueError('resume config mismatch')
    if not resolved.exists():resolved.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    specs={k:nonlinear_group_specs(cfg,k) for k in ('synthetic','measured')}
    if args.pilot:specs=dict(synthetic=specs['synthetic'][:1],measured=[])
    methods=['fixed_tau_0p01'] if args.pilot else cfg['methods'];modes=['full'] if args.pilot else cfg['modes']
    plan=[(s,m,v) for kind in specs for s in specs[kind] for m in methods for v in modes]
    if args.phase=='measured':
        synthetic_plan=[t for t in plan if t[0]['kind']=='synthetic']
        prior=summarize_nonlinear_fit(out,synthetic_plan,args.pilot)
        if prior['terminal_cells']!=prior['expected_cells'] or prior['observed_rows']!=prior['expected_rows']:
            raise ValueError('all synthetic tasks and their complete failure denominator must be terminal before measured access')
    manifest=dict(old,experiment_id=cfg['experiment_id'],execution='running',pilot=args.pilot,
        started_at=old.get('started_at',datetime.now(timezone.utc).isoformat()),controller_pid=os.getpid(),
        supervisor=os.environ.get('SSM_SYSTEMD_UNIT'),workers=args.workers,max_in_flight=min(cfg['resources']['max_in_flight'],2*args.workers),
        source_root=str(CODE_ROOT),project_root=str(args.project_root.resolve()),versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    write_json(manifest_path,manifest)
    for kind in ('synthetic','measured'):
        if not specs[kind] or args.phase not in ('all',kind):continue
        if kind=='measured':
            synthetic_plan=[t for t in plan if t[0]['kind']=='synthetic']
            prior=summarize_nonlinear_fit(out,synthetic_plan,args.pilot)
            if prior['terminal_cells']!=prior['expected_cells'] or prior['observed_rows']!=prior['expected_rows']:
                raise ValueError('all synthetic tasks and their complete failure denominator must be terminal before measured access')
        manifest['phase']=kind+'_preparation';write_json(manifest_path,manifest)
        prep=[(cfg,str(out),str(args.project_root.resolve()),s,args.pilot) for s in specs[kind]]
        for payload,result,error in bounded_nonlinear_work(prepare_nonlinear_group,prep,min(args.workers,len(prep)),manifest['max_in_flight']):
            if error:
                result=dict(status='failed_preparation',spec=payload[3],error=error)
                write_json(out/'prepared'/f'{payload[3]["group"]}.json',result)
            print(json.dumps(dict(event='prepared',group=result['spec']['group'],status=result['status'],seconds=result.get('seconds'))),flush=True)
        manifest['phase']=kind+'_fit';write_json(manifest_path,manifest)
        work=[(cfg,str(out),s,m,v,args.pilot) for s,m,v in plan if s['kind']==kind]
        for payload,result,error in bounded_nonlinear_work(nonlinear_cell,work,min(args.workers,len(work)),manifest['max_in_flight']):
            if error:
                _,_,spec,method,mode,pilot=payload;identifier=nonlinear_cell_id(spec,method,mode)
                validation=[i for i in range(24) if i%4==spec['outer']]
                if pilot:validation=validation[:2]
                result=dict(cell=identifier,status='failed_worker',error=error,rows=[dict(spec,method=method,mode=mode,
                    trial=i,session=cfg['sessions'][i//8],status='failed_worker',converged=False) for i in validation])
                write_json(out/'cells'/identifier/'result.json',result)
            print(json.dumps(dict(event='cell_terminal',cell=result['cell'],seconds=result.get('seconds'),
                completed=sum(r['status']=='completed' and r.get('converged',False) for r in result['rows']),expected=len(result['rows']))),flush=True)
            manifest['last_completed_cell']=result['cell'];write_json(manifest_path,manifest)
        current_plan=[t for t in plan if t[0]['kind']=='synthetic' or kind=='measured']
        summary=summarize_nonlinear_fit(out,current_plan,args.pilot)
        print(json.dumps(dict(event='phase_complete',phase=kind,summary=summary)),flush=True)
        manifest[kind+'_terminal']=True;manifest[kind+'_summary']=dict(expected_cells=summary['expected_cells'],terminal_cells=summary['terminal_cells'],observed_rows=summary['observed_rows'],completed_rows=summary['completed_rows'])
        write_json(manifest_path,manifest)
    complete=bool(args.pilot or manifest.get('measured_terminal'))
    manifest.update(execution='completed' if complete else 'synthetic_terminal',scientific_verdict='nonlinear_fit_diagnostic_only',summary='summary.json')
    if complete:manifest['completed_at']=datetime.now(timezone.utc).isoformat()
    write_json(manifest_path,manifest)


GAIN_PRIOR_METHODS = ['fixed_gain_no_amplitude','fixed_gain_amplitude',
                      'trained_gain_no_amplitude','trained_gain_amplitude']
FIXED_ROI_TAU_METHODS = ['fixed_roi_fixed_tau','fixed_roi_trained_tau','fixed_roi_prior_tau']


def read_fixed_roi_tau_config(path):
    return validate_fixed_roi_tau_config(yaml.safe_load(Path(path).read_text()))


def validate_fixed_roi_tau_config(cfg):
    expected=dict(schema='shared_driver_fixed_roi_tau_v1',experiment_id='SSM-SHARED-DRIVER-FIXED-ROI-TAU-v1',
        source_run='experiments/runs/physiology_semantic_tokenizer/ssm_overnight/20260917_measurement_retest_v1',
        output_root='experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction',
        protected_boundary='only_exact_parent_prepared_arrays_no_raw_or_protected_loaders',
        subjects=['subject_01','subject_09','subject_18'],outer_folds=[0,1,2,3],
        sessions=['session_01','session_03','session_05'],original_positions=[0,1,2,3,5,6,7,8],
        tensor=dict(trials_per_subject=24,steps=120,components=list(MODALITIES),dt_s=.25),
        methods=FIXED_ROI_TAU_METHODS,modes=['full','center_EEG','center_fNIRS'],center_steps=16,
        reference_tau_s=2.,split='frozen_parent_18_train_6_validation_each_outer_fold',
        scope='nonprotected_retained_public_development_diagnostic',
        model='original_nonlinear_Balloon_single_ZOH_driver_no_vascular_innovations',
        phase_rule='separate_synthetic_and_measured_calls_with_full_synthetic_terminal_evidence_before_measured_access',
        target='fixed_AF7Fp1_relative_HbO_HbR_and_exact_parent_EEG_log_power_PCA',
        normalizer='exact_parent_EEG_SD_and_fixed_ROI_Hb_component_SD_from_18_training_trials',
        mask='feature_missing_before_linear_processing_with_visible_interpolation',
        fixed=dict(kappa=.64,gamma=.32,alpha=.32,E0=.32,P0=1.,Q0=.35,neurovascular_gain=1.,eeg_loading=1.),
        fixed_roi=dict(pair_name='AF7Fp1',pair_index=0,selection='predeclared_first_registry_pair_not_validation_score',
            eeg='exact_parent_projection_unchanged',fnirs_scale='training_common_MAD_with_frozen_parent_observation_loading'))
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('fixed ROI tau scope/coordinate contract mismatch')
    declared={
        'tau_training':dict(parameter_name='tau',bounds=[.5,8.],start_values=[1.,2.,4.],prior_mean=2.,
            prior_log_sd=float(np.log(2)),max_evaluations=3600,max_iterations=90,gradient_tolerance=1e-4,
            information_rcond=1e-10,record_trace=True,
            initialization='zero_driver_and_physical_rest_for_every_training_trial_at_each_declared_tau',
            scope='one_tau_shared_across_all_18_training_trials_then_frozen_for_validation',
            selection='minimum_converged_group_objective_after_all_three_starts_terminal'),
        'regularization':dict(initial_state_weight=100.,driver_curvature_weight=.01,driver_amplitude_weight=0.),
        'nonlinear':dict(max_evaluations=160,gradient_tolerance=1e-4,substeps=4,integration_check_substeps=8,
            maximum_integration_difference_training_sd=.005),
        'synthetic':dict(seed=2026092603,spectra=['slow','mixed'],replicates=2,driver_sd=.04,
            initial_state_sd=.005,noise_fraction=.1,fast_fraction=.7,taus_s=[1.,2.,4.],
            tau_pairing='same_driver_initial_and_standardized_noise_draws_across_taus_within_spectrum_replicate')}
    if any(cfg[section].get(k)!=v for section,values in declared.items() for k,v in values.items()):
        raise ValueError('fixed ROI tau optimizer/synthetic contract mismatch')
    budget=cfg['resources']
    if (not 1<=budget['max_workers']<=48 or not 1<=budget['max_in_flight']<=48
            or budget['numerical_threads_per_worker']!=1 or not 0<cfg['svd_rcond']<1
            or cfg['small_signal_limit']!=.25):raise ValueError('fixed ROI tau numerical resource mismatch')
    parameters(cfg,2.).validate()
    return cfg


FIXED_ROI_FLOW_INHERITED = ['fixed_roi_fixed_tau','fixed_roi_trained_tau']
FIXED_ROI_FLOW_TRAINED = ['trained_tau_logflow01','trained_tau_logflow1','trained_tau_logflow4']
FIXED_ROI_FLOW_METHODS = FIXED_ROI_FLOW_INHERITED + ['fixed_tau_logflow1'] + FIXED_ROI_FLOW_TRAINED
FIXED_ROI_FLOW_OVERRIDES = dict(
    schema='shared_driver_fixed_roi_flow_v1',experiment_id='SSM-SHARED-DRIVER-FIXED-ROI-FLOW-v1',
    parent_run='experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_tau_v1',
    methods=FIXED_ROI_FLOW_METHODS,
    diagnostic='flow_anchor_sensitivity_tau_truth_recovery_multistart_information_boundary_and_complete_failures',
    baseline='exact_inherited_fixed_ROI_fixed_tau2_and_unregularized_trained_tau_cells_including_failures',
    null='log_flow_anchor_cannot_preserve_reconstruction_and_tau_recovery_while_reducing_extreme_flow_and_domain_failures',
    stopping='12_synthetic_then_12_measured_groups_each_3_new_trainable_arms_3_starts_and_6_methods_3_modes_6_validation_trials_no_scope_expansion',
    flow_prior=dict(log_sd=float(np.log(2)),weights=dict(fixed_tau_logflow1=1.,trained_tau_logflow01=.1,
        trained_tau_logflow1=1.,trained_tau_logflow4=4.),
        residual='sqrt_weight_dt_times_log_flow_over_log_sd_at_all_output_times_including_initial',
        interpretation='engineering_soft_anchor_not_physiological_population_range_or_calibrated_uncertainty',
        inheritance='parent_prepared_arrays_metadata_and_two_baseline_methods_inherited_nonindependent',
        tau_prior='none_in_all_new_training_arms'))
FIXED_ROI_FLOW_CLAIMS = [
    'inherited_baseline_and_prepared_evidence_is_not_an_independent_replication',
    'flow_anchor_weights_and_log_sd_are_engineering_sensitivity_choices_not_empirical_normal_ranges',
    'flow_regularization_changes_the_estimator_and_can_bias_tau_driver_and_states',
    'tau_has_zero_direct_flow_sensitivity_flow_anchor_affects_tau_indirectly_through_driver_and_initial_states',
    'objectives_are_comparable_only_between_starts_within_the_same_weight_arm',
    'all_six_methods_are_prespecified_not_validation_selected_winners']


def read_fixed_roi_flow_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if any(cfg.get(k)!=v for k,v in FIXED_ROI_FLOW_OVERRIDES.items()):
        raise ValueError('fixed ROI flow contract mismatch')
    normalized=deepcopy(cfg)
    normalized.update(schema='shared_driver_fixed_roi_tau_v1',experiment_id='SSM-SHARED-DRIVER-FIXED-ROI-TAU-v1',
                      methods=FIXED_ROI_TAU_METHODS)
    validate_fixed_roi_tau_config(normalized)
    if cfg.get('claim_limits',[])[-len(FIXED_ROI_FLOW_CLAIMS):]!=FIXED_ROI_FLOW_CLAIMS:
        raise ValueError('fixed ROI flow claim limits mismatch')
    return cfg


FIXED_ROI_INITIAL_FREE = ['fixed_tau_logflow1','trained_tau_logflow1']
FIXED_ROI_INITIAL_TRAINED = ['trained_tau_logflow1','trained_tau_logflow1_tied']
FIXED_ROI_INITIAL_METHODS = FIXED_ROI_INITIAL_FREE+['fixed_tau_logflow1_tied','trained_tau_logflow1_tied']
FIXED_ROI_INITIAL_OVERRIDES = dict(
    schema='shared_driver_fixed_roi_initial_v1',experiment_id='SSM-SHARED-DRIVER-FIXED-ROI-INITIAL-v1',
    parent_run='experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_flow_v1',
    methods=FIXED_ROI_INITIAL_METHODS,
    diagnostic='initial_total_Hb_volume_tie_restriction_tau_recovery_misspecification_and_transient_compensation',
    baseline='new_free_initial_synthetic_fits_and_exact_inherited_measured_weight1_free_initial_fits_including_failures',
    null='tying_initial_total_Hb_to_volume_cannot_preserve_reconstruction_and_tau_recovery_without_misspecification_bias',
    stopping='36_synthetic_groups_216_training_starts_2592_validation_rows_then_12_measured_groups_36_new_36_inherited_training_starts_864_validation_rows_no_scope_expansion',
    flow_prior=dict(FIXED_ROI_FLOW_OVERRIDES['flow_prior'],
        weights={m:1. for m in FIXED_ROI_INITIAL_METHODS},
        inheritance='measured_prepared_arrays_and_two_free_initial_methods_inherited_nonindependent_synthetic_all_new'),
    initial_constraint=dict(tied_methods=['fixed_tau_logflow1_tied','trained_tau_logflow1_tied'],
        relation='p0_equals_v0_in_reduced_optimization_coordinates',
        per_trial_nuisance_free_parameters={m:124 if m.endswith('_tied') else 125 for m in FIXED_ROI_INITIAL_METHODS},
        free_initial_coordinate_order={m:(['s0','logf0','logv0','logq0'] if m.endswith('_tied') else
            ['s0','logf0','logv0','logp0','logq0']) for m in FIXED_ROI_INITIAL_METHODS},
        full_physical_initial_order=['s0','f0','v0','p0','q0'],
        dynamics='original_equations_unchanged',
        initial_penalty='retain_all_five_physical_terms_including_both_volume_and_total_Hb',
        training_start='zero_driver_and_rest_for_both_free_and_tied',
        validation_start='tied_linear_reduced_SVD_fit_then_rest_no_projection_of_unconstrained_solution',
        interpretation='model_restriction_diagnostic_not_observed_initial_condition_or_population_range'))
FIXED_ROI_INITIAL_SECTION_OVERRIDES = dict(
    nonlinear=dict(fit_variables='per_method_reduced_initial_coordinates_full_physical_state_returned'),
    tau_training=dict(prior_interpretation='unused_no_parameter_prior_in_initial_contrast'))
FIXED_ROI_INITIAL_SYNTHETIC = dict(
    initial_log_p_over_v=[-.05,0.,.05],
    initial_relation='p0_equals_v0_times_exp_declared_log_ratio_other_initial_draws_unchanged',
    pairing='same_driver_other_initial_components_and_standardized_noise_across_tau_and_initial_ratio',
    ratio_interpretation='prespecified_model_misspecification_stress_not_physiological_normal_range',
    law='original_nonlinear_core_8_substeps_beta1_slow_or_mixed_driver_declared_initial_p_over_v_native_white_noise')
FIXED_ROI_INITIAL_CLAIMS = [
    'only_measured_free_initial_baselines_are_inherited_new_synthetic_truth_family_is_not_old_synthetic_evidence',
    'weight1_is_a_prespecified_middle_engineering_mechanism_control_not_a_selected_winner_or_physiological_prior',
    'p0_equals_v0_restricts_a_transient_mode_and_does_not_establish_individual_baseline_Hb_calibration',
    'nonzero_initial_log_p_over_v_is_a_misspecification_stress_not_a_population_normal_range',
    'all_four_initial_methods_are_prespecified_not_validation_selected_winners']


def read_fixed_roi_initial_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if (any(cfg.get(k)!=v for k,v in FIXED_ROI_INITIAL_OVERRIDES.items())
            or any(cfg['synthetic'].get(k)!=v for k,v in FIXED_ROI_INITIAL_SYNTHETIC.items())
            or any(cfg[section].get(k)!=v for section,values in FIXED_ROI_INITIAL_SECTION_OVERRIDES.items()
                   for k,v in values.items())
            or cfg.get('claim_limits',[])[-len(FIXED_ROI_INITIAL_CLAIMS):]!=FIXED_ROI_INITIAL_CLAIMS):
        raise ValueError('fixed ROI initial constraint contract mismatch')
    normalized=deepcopy(cfg)
    normalized.update(schema='shared_driver_fixed_roi_tau_v1',experiment_id='SSM-SHARED-DRIVER-FIXED-ROI-TAU-v1',
                      methods=FIXED_ROI_TAU_METHODS)
    validate_fixed_roi_tau_config(normalized)
    return cfg


def initial_group_specs(cfg,kind):
    if kind not in ('synthetic','measured'):raise ValueError('initial group requires explicit synthetic/measured kind')
    base=nonlinear_group_specs(cfg,kind)
    if kind=='measured':return base
    result=[]
    for spec in base:
        for ratio in cfg['synthetic']['initial_log_p_over_v']:
            group=spec['group']+f'_logpv{ratio:+.2f}'
            result.append(dict(spec,group=group,subject=group,true_initial_log_p_over_v=float(ratio)))
    return result


def validate_initial_parent(cfg,project_root,kind=None):
    """Only parent metadata is read here, including complete synthetic records."""
    parent=Path(project_root)/cfg['parent_run']
    manifest=json.loads((parent/'manifest.json').read_text())
    if (manifest.get('execution')!='completed' or manifest.get('pilot') is not False
            or manifest.get('experiment_id')!='SSM-SHARED-DRIVER-FIXED-ROI-FLOW-v1'
            or manifest.get('synthetic_terminal') is not True or manifest.get('measured_terminal') is not True
            or manifest.get('project_root')!=str(Path(project_root).resolve())):
        raise ValueError('initial parent must be exact completed nonpilot fixed ROI flow run')
    parent_cfg=read_fixed_roi_flow_config(parent/'resolved_config.yaml')
    expected=deepcopy(parent_cfg);expected.update(deepcopy(FIXED_ROI_INITIAL_OVERRIDES))
    expected['synthetic'].update(FIXED_ROI_INITIAL_SYNTHETIC)
    for section,values in FIXED_ROI_INITIAL_SECTION_OVERRIDES.items():expected[section].update(values)
    expected['claim_limits']=[c for c in expected['claim_limits'] if c!='all_six_methods_are_prespecified_not_validation_selected_winners']+FIXED_ROI_INITIAL_CLAIMS
    if cfg!=expected:raise ValueError('initial parent configuration changed outside declared contrast')
    validate_inherited_phase_records(parent_cfg,parent,'synthetic',FIXED_ROI_INITIAL_FREE,'trained_tau_logflow1')
    if kind is not None:
        if kind not in ('synthetic','measured'):raise ValueError('initial requires explicit synthetic or measured phase')
        if kind=='measured':
            validate_inherited_phase_records(parent_cfg,parent,kind,FIXED_ROI_INITIAL_FREE,'trained_tau_logflow1')
    return parent


FIXED_ROI_OPTICAL_METHODS = ['fixed_tau_logflow1','trained_tau_logflow1']
FIXED_ROI_OPTICAL_OVERRIDES = dict(
    schema='shared_driver_fixed_roi_optical_v1',experiment_id='SSM-SHARED-DRIVER-FIXED-ROI-OPTICAL-v1',
    parent_run='experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_flow_v1',
    optical_audit_run='experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_optical_motion_audit_v1',
    optical_pipelines=['no_motion','mne_tddr'],methods=FIXED_ROI_OPTICAL_METHODS,
    primary='per_pipeline_full_joint_shared_only_component_NRMSE_below_0.5_on_same_72_unique_outer_validation_trials',
    diagnostic='fixed_roi_optical_processing_sensitivity_free_initial_flowweight1',
    baseline='exact_inherited_weight1_synthetic_evidence_including_failures_not_new_replication',
    null='optical_processing_changes_do_not_change_shared_state_reconstruction_or_extreme_state_behavior',
    stopping='12_inherited_synthetic_groups_432_rows_then_24_new_measured_groups_72_training_starts_864_rows',
    flow_prior=dict(FIXED_ROI_FLOW_OVERRIDES['flow_prior'],weights={m:1. for m in FIXED_ROI_OPTICAL_METHODS},
        inheritance='synthetic_weight1_evidence_inherited_nonindependent_measured_new_optical_targets'))
FIXED_ROI_OPTICAL_TAU_INTERPRETATION = 'unused_no_parameter_prior_in_optical_contrast'
FIXED_ROI_OPTICAL_CLAIMS = [
    'different_optical_targets_and_training_SD_are_not_same_task_NRMSE_rankings',
    'no_motion_and_MNE_are_processing_sensitivity_conditions_not_physiological_truth',
    'synthetic_SSM_evidence_is_inherited_nonindependent_not_recovery_after_TDDR',
    'motion_synthetic_controls_remain_in_separate_completed_optical_audit',
    'weight1_is_prespecified_engineering_anchor_not_population_prior']


def read_fixed_roi_optical_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if (any(cfg.get(k)!=v for k,v in FIXED_ROI_OPTICAL_OVERRIDES.items())
            or cfg['tau_training'].get('prior_interpretation')!=FIXED_ROI_OPTICAL_TAU_INTERPRETATION
            or cfg.get('claim_limits',[])[-len(FIXED_ROI_OPTICAL_CLAIMS):]!=FIXED_ROI_OPTICAL_CLAIMS):
        raise ValueError('fixed ROI optical contract mismatch')
    normalized=deepcopy(cfg)
    normalized.update(schema='shared_driver_fixed_roi_tau_v1',experiment_id='SSM-SHARED-DRIVER-FIXED-ROI-TAU-v1',methods=FIXED_ROI_TAU_METHODS)
    validate_fixed_roi_tau_config(normalized)
    return cfg


def optical_group_specs(cfg,kind):
    base=nonlinear_group_specs(cfg,kind)
    if kind=='synthetic':return base
    if kind!='measured':raise ValueError('optical phase must be explicit')
    return [dict(s,group=pipeline+'__'+s['group'],optical_pipeline=pipeline)
            for pipeline in cfg['optical_pipelines'] for s in base]


def validate_optical_parent(cfg,project_root,kind=None):
    """Metadata only. Synthetic never dereferences optical/measured metadata."""
    root=Path(project_root);parent=root/cfg['parent_run']
    manifest=json.loads((parent/'manifest.json').read_text())
    if (manifest.get('execution')!='completed' or manifest.get('pilot') is not False
            or manifest.get('experiment_id')!='SSM-SHARED-DRIVER-FIXED-ROI-FLOW-v1'
            or manifest.get('synthetic_terminal') is not True or manifest.get('measured_terminal') is not True
            or manifest.get('project_root')!=str(root.resolve())):
        raise ValueError('optical parent must be completed nonpilot flow run')
    old=read_fixed_roi_flow_config(parent/'resolved_config.yaml')
    expected=deepcopy(old);expected.update(deepcopy(FIXED_ROI_OPTICAL_OVERRIDES))
    expected['tau_training']['prior_interpretation']=FIXED_ROI_OPTICAL_TAU_INTERPRETATION
    expected['claim_limits']=[c for c in old['claim_limits'] if c!='all_six_methods_are_prespecified_not_validation_selected_winners']+FIXED_ROI_OPTICAL_CLAIMS
    if cfg!=expected:raise ValueError('optical configuration changed outside declared contrast')
    validate_inherited_phase_records(old,parent,'synthetic',FIXED_ROI_OPTICAL_METHODS,'trained_tau_logflow1')
    if kind not in (None,'synthetic','measured'):raise ValueError('optical phase must be explicit')
    if kind!='measured':return parent
    audit=root/cfg['optical_audit_run'];a=json.loads((audit/'manifest.json').read_text())
    audit_cfg=yaml.safe_load((audit/'resolved_config.yaml').read_text())
    required=dict(schema='shared_driver_optical_motion_audit_v1',subjects=cfg['subjects'],sessions=cfg['sessions'],
        original_positions=cfg['original_positions'],pair_name='AF7Fp1',sampling_hz=10.,window_samples=300,
        pipelines=['no_motion','current','mne_tddr'],wavelengths_nm=[760.,850.],mne_version='1.11.0')
    if any(audit_cfg.get(k)!=v for k,v in required.items()):raise ValueError('optical audit source contract mismatch')
    summary=json.loads((audit/'measured_summary.json').read_text())
    syn=json.loads((audit/'synthetic_summary.json').read_text())
    if (a.get('experiment_id')!='SSM-SHARED-DRIVER-OPTICAL-MOTION-AUDIT-v1' or a.get('execution')!='completed'
            or a.get('result_status')!='completed' or a.get('project_root')!=str(root.resolve())
            or summary.get('status')!='completed' or summary.get('windows')!=72
            or summary.get('current_pipeline_reproduced') is not True or syn.get('status')!='completed'):
        raise ValueError('completed optical audit with exact current reproduction required')
    rows=summary['rows'];checks=summary['reproduction_checks']
    if len(rows)!=72 or len(checks)!=216 or any(c.get('matches') is not True for c in checks):
        raise ValueError('optical audit requires all 216 reproduction checks')
    identities={}
    for row in rows:
        ident=row['identity'];key=(ident['subject'],ident['trial_index'])
        if key in identities:raise ValueError('duplicate optical identity')
        identities[key]=ident
    if set(identities)!={(subject,i) for subject in cfg['subjects'] for i in range(24)}:
        raise ValueError('optical subject/trial scope mismatch')
    expected_checks=set()
    for spec in nonlinear_group_specs(old,'measured'):
        record=json.loads((parent/'prepared'/f"{spec['group']}.json").read_text())
        validate_flow_prepared(record,old,spec)
        for i,trial in enumerate(record['metadata']['trials']):
            ident=identities.get((spec['subject'],i),{})
            if (any(ident.get(k)!=v for k,v in trial.items())
                    or ident.get('selected_labels')!=['AF7Fp1lowWL','AF7Fp1highWL']):
                raise ValueError('audit and model retained trial identity mismatch')
            for stage in ('od','hb_native','hb_4hz'):expected_checks.add((trial['sample_id'],stage))
    if {(c['sample_id'],c['stage']) for c in checks}!=expected_checks:
        raise ValueError('optical reproduction identities mismatch')
    return parent


def prepare_fixed_roi_optical_group(payload):
    cfg,out,project_root,spec,pilot=payload
    if spec not in optical_group_specs(cfg,spec['kind']):raise ValueError('optical group outside scope')
    parent=validate_optical_parent(cfg,project_root,kind=spec['kind'])
    out=Path(out);path=out/'prepared'/f"{spec['group']}.json"
    if path.exists():return json.loads(path.read_text())
    if spec['kind']=='synthetic':
        if pilot:return prepare_fixed_roi_tau_group(payload)
        return inherit_flow_evidence(parent/'prepared'/path.name,path,arrays_name='prepared')
    if pilot:raise ValueError('optical pilot is synthetic only')
    from src.inference.observation_baselines import robust_mad
    base=dict(spec);base.pop('optical_pipeline');base['group']=spec['group'].split('__',1)[1]
    source=parent/'prepared'/f"{base['group']}.json";record=json.loads(source.read_text());meta=deepcopy(record['metadata'])
    audit=Path(project_root)/cfg['optical_audit_run'];summary=json.loads((audit/'measured_summary.json').read_text())
    indices=[next(j for j,r in enumerate(summary['rows']) if r['identity']['subject']==spec['subject']
                  and r['identity']['trial_index']==i) for i in range(24)]
    with np.load(source.with_suffix('.npz'),allow_pickle=False) as a:
        arrays={k:a[k] for k in a.files}
    expected=dict(target=(24,120,3),feature_eeg=(24,120),feature_fnirs=(24,300,2),normalizer=(3,))
    if any(arrays[k].shape!=shape or not np.isfinite(arrays[k]).all() for k,shape in expected.items()):
        raise ValueError('optical parent tensor mismatch')
    with np.load(audit/'measured_traces.npz',allow_pickle=False) as a:
        traces={p:a[p+'__hb_native'] for p in ['current']+cfg['optical_pipelines']}
    if any(v.shape!=(72,300,2) or not np.isfinite(v).all() for v in traces.values()):
        raise ValueError('optical traces shape/finite mismatch')
    fixed=meta['fixed_roi'];old_factor=float(fixed['fnirs_factor'])
    if not np.allclose(traces['current'][indices]*old_factor,arrays['feature_fnirs'],rtol=1e-10,atol=1e-12):
        raise ValueError('current audit Hb does not reproduce parent scaled feature')
    if (not np.allclose(arrays['normalizer'],meta['normalization_sd'],rtol=1e-12,atol=1e-14)
            or np.any(arrays['normalizer']<=0)):
        raise ValueError('optical parent normalization mismatch')
    reproduced=np.array([view(e,h,cfg,'full')[0] for e,h in zip(arrays['feature_eeg'],arrays['feature_fnirs'])])
    if np.max(abs(reproduced-arrays['target'])/arrays['normalizer'])>1e-6:
        raise ValueError('optical parent target reproduction mismatch')
    raw=traces[spec['optical_pipeline']][indices].copy();op=native_feature_operators(120)['fnirs']
    train=meta['train'];unscaled=np.array([op@h for h in raw])
    common=max(float(robust_mad(unscaled[train].reshape(-1))),1e-12)
    gauge=float(fixed['frozen_observation_loading']);factor=gauge/common
    feature=raw*factor;target_hb=np.array([op@h for h in feature]);sd=np.std(np.concatenate(target_hb[train]),axis=0)
    if not np.isfinite(sd).all() or np.any(sd<=1e-8):raise ValueError('optical low training variance')
    arrays['feature_fnirs']=feature;arrays['raw_optical_hb_native']=raw
    arrays['target']=arrays['target'].copy();arrays['target'][:,:,1:]=target_hb
    arrays['normalizer']=arrays['normalizer'].copy();arrays['normalizer'][1:]=sd
    meta['normalization_sd']=arrays['normalizer'].tolist()
    meta['optical_processing']=dict(pipeline=spec['optical_pipeline'],source_audit=str(audit),parent_prepared=str(source),
        source_row_indices=indices,training_trials=train,common_training_MAD=common,frozen_observation_loading=gauge,
        fnirs_factor=factor,units='relative_Hb_not_calibrated_uM',interpretation='different_target_and_SD_not_same_task_ranking')
    meta['fixed_roi']=dict(fixed,common_training_MAD=common,fnirs_factor=factor)
    projection=meta['projection'];projection.update(fnirs_factor=factor,inverse_fnirs_factor=1/factor)
    projection['measurement_scale']['fnirs_common']=common;projection['computational_scale']['fnirs_common']=1/common
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.npz.tmp').open('wb') as stream:np.savez_compressed(stream,**arrays)
    path.with_suffix('.npz.tmp').replace(path.with_suffix('.npz'))
    result=dict(status='completed',spec=spec,metadata=meta,selected_tau={m:2. for m in cfg['methods']},driver_prior_sd=record['driver_prior_sd'])
    write_json(path,result);return result


CONDITIONAL_GAIN_METHODS=['fixed_tau_logflow1','conditional_fixed_gain','conditional_trained_gain']
CONDITIONAL_GAIN_OVERRIDES=dict(schema='shared_driver_conditional_optical_gain_v1',
    experiment_id='SSM-SHARED-DRIVER-CONDITIONAL-OPTICAL-GAIN-v1',
    parent_run='experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_optical_v1',
    methods=CONDITIONAL_GAIN_METHODS,
    primary='same_target_full_joint_shared_only_component_NRMSE_below_0.5_and_paired_identity_map_baseline_contrast_per_pipeline',
    diagnostic='conditional_optical_scale_mapping_and_relative_gain_at_fixed_tau2',
    baseline='exact_inherited_measured_identity_map_beta1_fixed_tau2_including_all_failures_synthetic_all_fresh',
    null='conditional_prediction_mapping_and_shared_gain_do_not_improve_same_target_reconstruction_or_remove_extreme_states',
    stopping='12_new_synthetic_groups_36_gain_starts_648_rows_then_24_measured_groups_72_gain_starts_1296_rows_432_inherited_no_scope_expansion',
    flow_prior=dict(FIXED_ROI_FLOW_OVERRIDES['flow_prior'],weights={m:1. for m in CONDITIONAL_GAIN_METHODS},
        inheritance='only_measured_identity_map_fixed_tau2_inherited_nonindependent'),
    conditional_mapping=dict(old_extinction=[[.148,.384],[.252,.179]],decadic_extinction_M_inv_cm_inv=[[586.,1548.52],[1058.,691.32]],
        natural_log_conversion='ln10_times_1e-6',P0_assumed_uM=71.,tissue_sensitivity_assumed=1.,path_length_cm=3.,dpf=6.,
        formula='B=k*inv(Eold)@(ln(10)*1e-6*Edecadic*71); equal_18cm_paths_cancel',
        k='exact_parent_optical_processing_fnirs_factor_from_18_train',
        predictor='existing_mean_operator@kron(I,blockdiag(1,B));targets_and_SD_unchanged',
        beta_reference='1/sqrt(det(B))_engineering_amplitude_gauge_not_independent_NVC_calibration',
        initial_coordinates='independent_logs',tau_s=2.,P0_primary_source='https://www.fil.ion.ucl.ac.uk/~wpenny/publications/tak-penny15.pdf'),
    synthetic_engineering_screen=dict(expected_groups=12,expected_starts_per_group=3,minimum_successful_starts=1,
        maximum_selected_relative_g_error=.1,maximum_successful_g_span_over_true=.05,require_successful_starts_interior=True,
        failure_action='retain_negative_synthetic_evidence_and_stop_before_measured_arrays'),
    conditional_gain_training=dict(parameter_name='neurovascular_gain',relative_bounds=[.1,10.],relative_start_values=[.5,1.,2.],
        parameter_prior=None,max_evaluations=3600,max_iterations=90,gradient_tolerance=1e-4,information_rcond=1e-10,
        record_trace=True,initialization='zero_driver_physical_rest_all18trials',selection='minimum_converged_objective_after_all_declared_starts_terminal'))
CONDITIONAL_GAIN_SYNTHETIC=dict(relative_gains=[.5,1.,2.],mapping_fnirs_factor=40.,driver_sd=.04,initial_state_sd=.005,noise_fraction=.1,
    law='original_nonlinear_core_tau2_beta_g_times_reference_conditional_Hb_map_before_native_noise',
    pairing='same_driver_initial_draws_and_standardized_native_noise_across_relative_gain',
    interpretation='new_conditional_model_software_recovery_not_inverse_TDDR_or_measured_calibration')
CONDITIONAL_GAIN_CLAIMS=[
    'P0_71uM_site_sensitivity1_and_DPF6_are_conditional_assumptions_not_subject_calibration',
    'B_has_signed_cross_component_terms_no_clipping_and_only_changes_prediction_mean',
    'beta_and_beta_over_reference_are_confounded_with_optical_sensitivity_and_driver_gauge_not_independent_NVC',
    'same_target_and_SD_within_each_pipeline_enable_paired_baseline_comparison_not_cross_pipeline_ranking',
    'positive_Hb_optimization_coordinates_off_independent_logs_isolate_mean_mapping_hypothesis',
    'new_synthetic_conditional_truth_is_not_recovery_of_real_Hb_after_motion_processing',
    'identity_to_conditional_fixed_changes_both_map_and_beta_reference_not_pure_map_effect',
    'fixed_to_trained_conditional_arm_is_the_shared_beta_contribution_at_the_same_map',
    'B_amplifies_initial_state_contributions_while_physical_initial_prior_is_unchanged_and_flow_penalty_changes_with_physical_states',
    'Tak71_is_cortical_source_compartment_assumption_not_individual_intravascular_venous_concentration; wavelength_sensitivity1_and_omitted_pial_terms_are_not_exact_Tak_reproduction']


def read_conditional_optical_gain_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if (cfg.get('optical_pipelines')!=['no_motion','mne_tddr'] or any(cfg.get(k)!=v for k,v in CONDITIONAL_GAIN_OVERRIDES.items())
            or any(cfg['synthetic'].get(k)!=v for k,v in CONDITIONAL_GAIN_SYNTHETIC.items())
            or cfg.get('claim_limits',[])[-len(CONDITIONAL_GAIN_CLAIMS):]!=CONDITIONAL_GAIN_CLAIMS):
        raise ValueError('conditional optical gain contract mismatch')
    normalized=deepcopy(cfg)
    normalized.update(schema='shared_driver_fixed_roi_tau_v1',experiment_id='SSM-SHARED-DRIVER-FIXED-ROI-TAU-v1',methods=FIXED_ROI_TAU_METHODS)
    validate_fixed_roi_tau_config(normalized)
    return cfg


def conditional_gain_specs(cfg,kind):
    if kind=='measured':return optical_group_specs(cfg,kind)
    if kind!='synthetic':raise ValueError('conditional gain phase must be explicit')
    from src.inference.observation_baselines import conditional_optical_hb_mapping
    ref=conditional_optical_hb_mapping(cfg['synthetic']['mapping_fnirs_factor'])['beta_reference']
    return [dict(kind=kind,group=f'synthetic_g{g:g}_{spectrum}_r{rep}',subject=f'synthetic_g{g:g}_{spectrum}_r{rep}',
        outer=0,true_tau=2.,true_relative_gain=float(g),true_neurovascular_gain=float(g*ref),spectrum=spectrum,replicate=rep)
        for g in cfg['synthetic']['relative_gains'] for spectrum in cfg['synthetic']['spectra'] for rep in range(cfg['synthetic']['replicates'])]


def conditional_prediction_operator(operator,mapping,method):
    if method=='fixed_tau_logflow1':return operator
    if method not in CONDITIONAL_GAIN_METHODS:raise ValueError('unknown conditional mean method')
    block=np.eye(3);block[1:,1:]=np.asarray(mapping['matrix'],float)
    if operator.ndim!=2 or operator.shape[1]%3:raise ValueError('conditional prediction operator shape mismatch')
    return operator@np.kron(np.eye(operator.shape[1]//3),block)


def validate_conditional_gain_parent(cfg,project_root,kind=None):
    parent=Path(project_root)/cfg['parent_run'];manifest=json.loads((parent/'manifest.json').read_text())
    if (manifest.get('execution')!='completed' or manifest.get('pilot') is not False
            or manifest.get('experiment_id')!='SSM-SHARED-DRIVER-FIXED-ROI-OPTICAL-v1'
            or manifest.get('synthetic_terminal') is not True or manifest.get('measured_terminal') is not True
            or manifest.get('project_root')!=str(Path(project_root).resolve())):
        raise ValueError('conditional gain parent must be exact completed nonpilot optical run')
    old=read_fixed_roi_optical_config(parent/'resolved_config.yaml');expected=deepcopy(old)
    expected.update(deepcopy(CONDITIONAL_GAIN_OVERRIDES));expected['synthetic'].update(CONDITIONAL_GAIN_SYNTHETIC)
    expected['claim_limits']=[c for c in old['claim_limits'] if c!='synthetic_SSM_evidence_is_inherited_nonindependent_not_recovery_after_TDDR']+CONDITIONAL_GAIN_CLAIMS
    if cfg!=expected:raise ValueError('conditional gain changed outside declared contrast')
    if kind not in (None,'synthetic','measured'):raise ValueError('conditional gain phase must be explicit')
    if kind!='measured':return parent
    for spec in optical_group_specs(old,'measured'):
        record=json.loads((parent/'prepared'/f"{spec['group']}.json").read_text())
        if record.get('spec')!=spec or record.get('status') not in ('completed','failed_preparation'):
            raise ValueError('conditional parent prepared identity/terminal mismatch')
        val=[i for i in range(24) if i%4==spec['outer']];train=[i for i in range(24) if i not in val]
        if record['status']=='completed':
            meta=record['metadata'];opt=meta['optical_processing'];trials=meta['trials']
            if (meta.get('subject')!=spec['subject'] or meta.get('outer')!=spec['outer'] or meta['train']!=train or meta['validation']!=val
                    or opt['pipeline']!=spec['optical_pipeline'] or opt['training_trials']!=train
                    or opt.get('source_audit')!=str(Path(project_root)/cfg['optical_audit_run'])
                    or opt.get('source_row_indices')!=list(range(cfg['subjects'].index(spec['subject'])*24,(cfg['subjects'].index(spec['subject'])+1)*24))
                    or not np.isclose(opt['fnirs_factor']*opt['common_training_MAD'],opt['frozen_observation_loading'],rtol=1e-12,atol=1e-14)
                    or not np.isfinite(opt['fnirs_factor']) or opt['fnirs_factor']<=0
                    or opt['fnirs_factor']!=meta['fixed_roi']['fnirs_factor']
                    or opt['fnirs_factor']!=meta['projection']['fnirs_factor']
                    or opt['common_training_MAD']!=meta['fixed_roi']['common_training_MAD']
                    or opt['common_training_MAD']!=meta['projection']['measurement_scale']['fnirs_common']
                    or opt['frozen_observation_loading']!=meta['fixed_roi']['frozen_observation_loading']
                    or len(trials)!=24 or len({t['sample_id'] for t in trials})!=24
                    or any(t['subject']!=spec['subject'] or t['session']!=cfg['sessions'][i//8]
                        or t['original_ma_trial_position']!=cfg['original_positions'][i%8] for i,t in enumerate(trials))):
                raise ValueError('conditional parent 18/6 identity/training-scale mismatch')
        for mode in cfg['modes']:
            identifier=nonlinear_cell_id(spec,'fixed_tau_logflow1',mode)
            cell=json.loads((parent/'cells'/identifier/'result.json').read_text())
            if (cell.get('cell')!=identifier or len(cell.get('rows',[]))!=6 or [r['trial'] for r in cell['rows']]!=val
                    or any(any(r.get(k)!=v for k,v in dict(spec,method='fixed_tau_logflow1',mode=mode,
                        session=cfg['sessions'][r['trial']//8]).items()) for r in cell['rows'])
                    or any(r['status'] not in ('completed','failed_training','failed_domain','failed_numerical','failed_replay_validation',
                        'failed_integration_check','failed_exception','failed_worker','failed_preparation') for r in cell['rows'])):
                raise ValueError('conditional parent baseline terminal identity mismatch')
    return parent


def prepare_conditional_gain_group(payload):
    from src.inference.observation_baselines import conditional_optical_hb_mapping
    cfg,out,project_root,spec,pilot=payload
    if spec not in conditional_gain_specs(cfg,spec['kind']):raise ValueError('conditional group outside scope')
    if pilot and spec['kind']!='synthetic':raise ValueError('conditional gain pilot is synthetic only')
    parent=validate_conditional_gain_parent(cfg,project_root,kind=spec['kind']);out=Path(out)
    path=out/'prepared'/f"{spec['group']}.json"
    if path.exists():
        existing=json.loads(path.read_text())
        if existing.get('status')=='completed' and 'conditional_mapping' not in existing:
            raise ValueError('incomplete conditional prepared metadata cannot be resumed')
        return existing
    if spec['kind']=='synthetic':
        arrays,meta=nonlinear_synthetic_panel(cfg,spec)
        record=dict(status='completed',spec=spec,metadata=meta,selected_tau={m:2. for m in cfg['methods']},
                    driver_prior_sd=float(arrays['normalizer'][0]/cfg['fixed']['eeg_loading']))
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix('.npz.tmp').open('wb') as stream:np.savez_compressed(stream,**arrays)
        path.with_suffix('.npz.tmp').replace(path.with_suffix('.npz'))
    else:
        source=parent/'prepared'/path.name
        record=json.loads(source.read_text())
        if record['status']=='completed':
            with np.load(source.with_suffix('.npz'),allow_pickle=False) as a:
                expected=dict(target=(24,120,3),feature_eeg=(24,120),feature_fnirs=(24,300,2),normalizer=(3,),raw_optical_hb_native=(24,300,2))
                if any(a[k].shape!=shape or not np.isfinite(a[k]).all() for k,shape in expected.items()) or np.any(a['normalizer']<=0):
                    raise ValueError('conditional parent array shape/finite mismatch')
                if not np.array_equal(a['normalizer'],np.asarray(record['metadata']['normalization_sd'])):
                    raise ValueError('conditional exact parent SD identity mismatch')
                meta=record['metadata'];k=meta['optical_processing']['fnirs_factor']
                if not np.allclose(a['raw_optical_hb_native']*k,a['feature_fnirs'],rtol=1e-12,atol=1e-14):
                    raise ValueError('conditional parent raw Hb/factor closure mismatch')
                reproduced=np.array([view(e,h,cfg,'full')[0] for e,h in zip(a['feature_eeg'],a['feature_fnirs'])])
                if np.max(abs(reproduced-a['target'])/a['normalizer'])>1e-6:
                    raise ValueError('conditional parent target processing closure mismatch')
                parent_target=a['target'].copy();parent_sd=a['normalizer'].copy()
            for mode in cfg['modes']:
                cell=parent/'cells'/nonlinear_cell_id(spec,'fixed_tau_logflow1',mode)
                evidence=json.loads((cell/'result.json').read_text());trajectory=cell/'trajectories.npz'
                if not trajectory.exists():
                    if any(r['status']=='completed' and r.get('converged',False) for r in evidence['rows']):
                        raise ValueError('successful inherited baseline trajectory missing')
                    continue
                with np.load(trajectory,allow_pickle=False) as a:
                    if (not np.array_equal(a['trial_indices'],meta['validation']) or not np.array_equal(a['normalizer'],parent_sd)
                            or not np.array_equal(a['target'],parent_target[meta['validation']],equal_nan=True)):
                        raise ValueError('inherited baseline target/SD/trial closure mismatch')
        path.parent.mkdir(parents=True,exist_ok=True)
        if source.with_suffix('.npz').exists():
            temp=path.with_suffix('.npz.tmp');shutil.copyfile(source.with_suffix('.npz'),temp);temp.replace(path.with_suffix('.npz'))
        record['inheritance']=dict(status='inherited_nonindependent',parent_record=str(source.resolve()),new_optimization_evaluations=0)

    if record['status']=='completed':
        k=cfg['synthetic']['mapping_fnirs_factor'] if spec['kind']=='synthetic' else record['metadata']['optical_processing']['fnirs_factor']
        record['conditional_mapping']=conditional_optical_hb_mapping(k)
        record['conditional_mapping']['source']='prespecified_synthetic_k40' if spec['kind']=='synthetic' else 'exact_parent_training_scale_metadata'
    write_json(path,record);return record


def conditional_gain_screen(out,specs,pilot=False):
    rows=[];expected_starts=1 if pilot else 3
    for spec in specs:
        folder=gain_training_directory(out,spec,'conditional_trained_gain');path=folder/'selection.json'
        choice=json.loads(path.read_text()) if path.exists() else {}
        starts=[]
        for i in ([1] if pilot else range(3)):
            p=folder/f'start_{i}'/'result.json'
            if p.exists():starts.append(json.loads(p.read_text()))
        good=[r for r in starts if r.get('status')=='completed' and r.get('converged',False)]
        true=spec['true_relative_gain'];selected=choice.get('relative_gain')
        err=abs(selected/true-1) if selected is not None else float('inf')
        gains=[r.get('relative_gain',float('nan')) for r in good]
        span=(max(gains)-min(gains))/true if gains and np.isfinite(gains).all() else float('inf')
        interior=bool(good) and all(r.get('boundary_status')=='INTERIOR' for r in good)
        passed=(len(starts)==expected_starts and all(r.get('status')=='completed' or str(r.get('status','')).startswith('failed_') for r in starts) and choice.get('terminal_starts')==expected_starts
            and choice.get('status')=='completed' and len(good)>=1 and err<=.1 and span<=.05 and interior)
        rows.append(dict(group=spec['group'],true_relative_gain=true,selected_relative_gain=selected,
            terminal_starts=len(starts),successful_starts=len(good),selected_relative_error=err,
            successful_span_over_true=span,successful_starts_interior=interior,passed=bool(passed)))
    return dict(expected_groups=1 if pilot else 12,observed_groups=len(rows),all_groups_passed=len(rows)==(1 if pilot else 12) and all(r['passed'] for r in rows),
        groups=rows,interpretation='prespecified_synthetic_engineering_recovery_screen_not_measured_identifiability')


def flow_options(cfg,method):
    if method not in cfg['flow_prior']['weights']:
        raise ValueError('flow options only apply to declared newly fitted methods')
    return dict(flow_prior_weight=cfg['flow_prior']['weights'][method],flow_prior_log_sd=cfg['flow_prior']['log_sd'])


def validate_flow_prepared(record,cfg,spec):
    """Check exact identity and split metadata before any array read or copy."""
    if spec.get('kind') not in ('synthetic','measured') or spec not in nonlinear_group_specs(cfg,spec.get('kind')):
        raise ValueError('flow group outside declared phase identities')
    if record.get('status')!='completed' or record.get('spec')!=spec:
        raise ValueError('flow parent prepared identity/status mismatch')
    meta=record['metadata'];validation=[i for i in range(24) if i%4==spec['outer']]
    train=[i for i in range(24) if i not in validation]
    if meta.get('train')!=train or meta.get('validation')!=validation:
        raise ValueError('flow parent split must be exact 18/6 outer partition')
    trials=meta.get('trials',[])
    if (len(trials)!=24 or len({t.get('sample_id') for t in trials})!=24
            or any(not t.get('sample_id') or t.get('session')!=cfg['sessions'][i//8] for i,t in enumerate(trials))):
        raise ValueError('flow parent trial identities/session mismatch')
    if spec['kind']=='measured':
        if (meta.get('subject')!=spec['subject'] or meta.get('outer')!=spec['outer']
                or any(meta.get('fixed_roi',{}).get(k)!=v for k,v in cfg['fixed_roi'].items())
                or any(t.get('subject')!=spec['subject'] or t.get('original_ma_trial_position')!=cfg['original_positions'][i%8]
                       for i,t in enumerate(trials))):
            raise ValueError('flow parent measured ROI/trial identity mismatch')
    elif any(meta.get(k)!=v for k,v in spec.items()):
        raise ValueError('flow parent synthetic truth identity mismatch')


def validate_flow_parent(cfg,project_root,kind=None):
    """Parent metadata audit only; phase arrays are opened only by preparation."""
    parent=Path(project_root)/cfg['parent_run']
    manifest=json.loads((parent/'manifest.json').read_text())
    if (manifest.get('execution')!='completed' or manifest.get('pilot') is not False
            or manifest.get('experiment_id')!='SSM-SHARED-DRIVER-FIXED-ROI-TAU-v1'
            or manifest.get('synthetic_terminal') is not True or manifest.get('measured_terminal') is not True
            or manifest.get('project_root')!=str(Path(project_root).resolve())):
        raise ValueError('flow parent must be exact completed nonpilot fixed ROI tau run')
    parent_cfg=read_fixed_roi_tau_config(parent/'resolved_config.yaml')
    expected=deepcopy(parent_cfg);expected.update(deepcopy(FIXED_ROI_FLOW_OVERRIDES))
    expected['claim_limits']=[c for c in expected['claim_limits'] if c!='all_three_methods_are_prespecified_not_validation_selected_winners']+FIXED_ROI_FLOW_CLAIMS
    if cfg!=expected:raise ValueError('flow parent configuration changed outside declared flow experiment')
    if kind is None:return parent
    if kind not in ('synthetic','measured'):raise ValueError('flow requires explicit synthetic or measured phase')
    return validate_inherited_phase_records(cfg,parent,kind,FIXED_ROI_FLOW_INHERITED,'fixed_roi_trained_tau')


def validate_inherited_phase_records(cfg,parent,kind,methods,training_method):
    for spec in nonlinear_group_specs(cfg,kind):
        prepared=parent/'prepared'/f"{spec['group']}.json"
        record=json.loads(prepared.read_text());validate_flow_prepared(record,cfg,spec)
        if not prepared.with_suffix('.npz').is_file():raise ValueError('flow parent prepared arrays missing')
        for method in methods:
            for mode in cfg['modes']:
                identifier=nonlinear_cell_id(spec,method,mode)
                cell=json.loads((parent/'cells'/identifier/'result.json').read_text())
                rows=cell.get('rows',[])
                validation=record['metadata']['validation']
                if (cell.get('cell')!=identifier or len(rows)!=6
                        or [r.get('trial') for r in rows]!=validation
                        or any(any(r.get(k)!=v for k,v in dict(spec,method=method,mode=mode,
                            session=cfg['sessions'][r['trial']//8]).items()) for r in rows)
                        or any(r.get('status') not in ('completed','failed_training','failed_domain','failed_numerical',
                            'failed_replay_validation','failed_integration_check','failed_exception','failed_worker','failed_preparation') for r in rows)):
                    raise ValueError('flow parent validation identities/terminal status mismatch')
                if not (parent/'cells'/identifier/'trajectories.npz').is_file():
                    raise ValueError('flow parent validation arrays missing')
        directory=gain_training_directory(parent,spec,training_method)
        selection=json.loads((directory/'selection.json').read_text())
        if (selection.get('spec')!=spec or selection.get('method')!=training_method
                or selection.get('expected_starts')!=3 or selection.get('terminal_starts')!=3
                or selection.get('status') not in ('completed','failed_training')):
            raise ValueError('flow parent training selection mismatch')
        for i,tau in enumerate(cfg['tau_training']['start_values']):
            result=json.loads((directory/f'start_{i}'/'result.json').read_text())
            if (result.get('spec')!=spec or result.get('method')!=training_method
                    or result.get('start_index')!=i or result.get('start_tau')!=tau
                    or result.get('expected_trials')!=18 or result.get('status') not in
                        ('completed','failed_numerical','failed_domain','failed_integration_check','failed_exception','failed_worker','failed_preparation','failed_replay_validation','failed_missing_trajectory')):
                raise ValueError('flow parent training start identity/status mismatch')
    return parent


def inherit_flow_evidence(source,destination,*,arrays_name=None):
    """Keep failed and successful parent records, with explicit nonindependence."""
    source=Path(source);destination=Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('cannot overwrite parent evidence')
    if destination.exists():return json.loads(destination.read_text())
    record=json.loads(source.read_text());destination.parent.mkdir(parents=True,exist_ok=True)
    arrays=source.with_suffix('.npz') if arrays_name=='prepared' else source.parent/arrays_name if arrays_name else None
    if arrays is not None and arrays.is_file():
        target=destination.with_suffix('.npz') if arrays_name=='prepared' else destination.parent/arrays_name
        temp=target.with_suffix('.npz.tmp');shutil.copyfile(arrays,temp);temp.replace(target)
    record['inheritance']=dict(status='inherited_nonindependent',parent_record=str(source.resolve()),
                               new_optimization_evaluations=0)
    write_json(destination,record)
    return record


def prepare_fixed_roi_flow_group(payload):
    cfg,out,project_root,spec,pilot=payload
    # This function never generates new subjects or synthetic draws.
    if pilot and spec.get('kind')!='synthetic':raise ValueError('flow pilot is synthetic only')
    manifest_path=Path(out)/'manifest.json'
    if manifest_path.exists():
        phase=json.loads(manifest_path.read_text()).get('phase','')
        if not phase.startswith(spec.get('kind','invalid')+'_'):
            raise ValueError('flow prepared identity does not match active phase')
    if cfg['schema']=='shared_driver_fixed_roi_initial_v1':
        parent=validate_initial_parent(cfg,project_root)
        if spec not in initial_group_specs(cfg,spec.get('kind')):raise ValueError('initial group outside declared scope')
        if spec['kind']=='synthetic':return prepare_fixed_roi_tau_group(payload)
    else:parent=validate_flow_parent(cfg,project_root)
    source=parent/'prepared'/f"{spec['group']}.json"
    record=json.loads(source.read_text());validate_flow_prepared(record,cfg,spec)
    if not source.with_suffix('.npz').is_file():raise ValueError('flow parent prepared arrays missing')
    with np.load(source.with_suffix('.npz'),allow_pickle=False) as a:
        expected=dict(target=(24,120,3),feature_eeg=(24,120),feature_fnirs=(24,300,2),normalizer=(3,))
        if spec['kind']=='synthetic':expected.update(truth=(24,120),truth_states=(24,120,6))
        if any(k not in a or a[k].shape!=shape for k,shape in expected.items()):
            raise ValueError('flow parent prepared tensor contract mismatch')
        if not np.isfinite(a['normalizer']).all() or np.any(a['normalizer']<=0):
            raise ValueError('flow parent normalization must be finite positive')
    return inherit_flow_evidence(source,Path(out)/'prepared'/source.name,arrays_name='prepared')


def inherit_flow_training(cfg,project_root,out,spec,method='fixed_roi_trained_tau'):
    parent=Path(project_root)/cfg['parent_run']
    source=gain_training_directory(parent,spec,method)
    destination=gain_training_directory(out,spec,method)
    for i in range(3):
        inherit_flow_evidence(source/f'start_{i}'/'result.json',destination/f'start_{i}'/'result.json',
                              arrays_name='trajectories.npz')
    return inherit_flow_evidence(source/'selection.json',destination/'selection.json')


def prepare_fixed_roi_tau_group(payload):
    cfg,out,project_root,spec,pilot=payload
    out=Path(out);path=out/'prepared'/f"{spec['group']}.json"
    if path.exists():return json.loads(path.read_text())
    started=time.monotonic()
    try:
        arrays,meta=(nonlinear_synthetic_panel(cfg,spec) if spec['kind']=='synthetic'
                     else load_fixed_roi_measured(cfg,project_root,spec['subject'],spec['outer']))
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix('.npz.tmp').open('wb') as stream:np.savez_compressed(stream,**arrays)
        path.with_suffix('.npz.tmp').replace(path.with_suffix('.npz'))
        result=dict(status='completed',spec=spec,metadata=meta,
            selected_tau={m:2. for m in cfg['methods']},
            driver_prior_sd=float(arrays['normalizer'][0]/cfg['fixed']['eeg_loading']),
            seconds=time.monotonic()-started)
    except Exception as exc:
        result=dict(status='failed_preparation',spec=spec,error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started)
    write_json(path,result)
    return result


def summarize_fixed_roi_tau(out,plan,pilot):
    out=Path(out)
    methods=list(dict.fromkeys(method for _,method,_ in plan))
    summary=summarize_nonlinear_fit(out,plan,pilot,methods=methods)
    specs={s['group']:s for s,_,_ in plan};training=[]
    for spec in specs.values():
        for method in methods:
            if method in ('fixed_roi_fixed_tau','fixed_tau_logflow1','fixed_tau_logflow1_tied','conditional_fixed_gain'):continue
            path=gain_training_directory(out,spec,method)/'selection.json'
            if path.exists():
                r=json.loads(path.read_text());training.append(dict(group=spec['group'],kind=spec['kind'],method=method,
                    status=r['status'],parameter_value=r['parameter_value'],expected_starts=r['expected_starts'],
                    terminal_starts=r['terminal_starts'],successful_starts=r['successful_starts'],
                    **({'inheritance':r['inheritance']} if 'inheritance' in r else {})))
    summary.update(training=training,expected_training_groups=sum(m not in ('fixed_roi_fixed_tau','fixed_tau_logflow1','fixed_tau_logflow1_tied','conditional_fixed_gain') for m in methods)*len(specs),terminal_training_groups=len(training),
                   successful_training_groups=sum(r['status']=='completed' for r in training))
    if any(m in FIXED_ROI_FLOW_TRAINED for m in methods):
        initial_branch=any(m.endswith('_tied') for m in methods)
        inherited_methods=FIXED_ROI_INITIAL_FREE if initial_branch else FIXED_ROI_FLOW_INHERITED
        inherited_rows=sum(a['observed_rows'] for a in summary['aggregates'] if a['method'] in inherited_methods
                           and (not initial_branch or a['kind']=='measured'))
        expected_inherited=sum(m in inherited_methods and (not initial_branch or s['kind']=='measured')
                               for s,m,_ in plan)*(2 if pilot else 6)
        summary.update(expected_inherited_baseline_rows=expected_inherited,
            inherited_baseline_rows=inherited_rows,new_validation_rows=summary['observed_rows']-inherited_rows,
            inherited_training_groups=sum('inheritance' in r for r in training),
            new_training_groups=sum('inheritance' not in r for r in training),
            inheritance_interpretation='inherited_baselines_and_prepared_arrays_are_not_independent_replication')
        if initial_branch:
            expected_inherited_groups=sum(s['kind']=='measured' for s in specs.values())
            summary.update(expected_inherited_training_starts=3*expected_inherited_groups,
                expected_new_training_starts=(summary['expected_training_groups']-expected_inherited_groups)*(1 if pilot else 3),
                expected_new_validation_rows=summary['expected_rows']-expected_inherited)
    if 'conditional_trained_gain' not in methods and (any('optical_pipeline' in s for s in specs.values()) or (out/'resolved_config.yaml').exists() and yaml.safe_load((out/'resolved_config.yaml').read_text()).get('schema')=='shared_driver_fixed_roi_optical_v1'):
        inherited=sum(s['kind']=='synthetic' and not pilot for s,m,v in plan)*6
        summary.update(expected_inherited_baseline_rows=inherited,inherited_baseline_rows=sum(a['observed_rows'] for a in summary['aggregates'] if a['kind']=='synthetic' and not pilot),
            new_validation_rows=sum(a['observed_rows'] for a in summary['aggregates'] if a['kind']=='measured' or pilot),expected_new_validation_rows=summary['expected_rows']-inherited,
            expected_inherited_training_starts=sum(s['kind']=='synthetic' and not pilot for s in specs.values())*3,
            expected_new_training_starts=sum(s['kind']=='measured' or pilot for s in specs.values())*(1 if pilot else 3),
            interpretation='pipeline_targets_and_training_SD_differ_no_same_task_ranking')
    if 'conditional_trained_gain' in methods:
        summary['synthetic_engineering_screen']=conditional_gain_screen(out,[s for s in specs.values() if s['kind']=='synthetic'],pilot)
        inherited=sum(s['kind']=='measured' and m=='fixed_tau_logflow1' for s,m,v in plan)*(2 if pilot else 6)
        observed_inherited=sum(a['observed_rows'] for a in summary['aggregates'] if a['kind']=='measured' and a['method']=='fixed_tau_logflow1')
        summary.update(expected_inherited_baseline_rows=inherited,inherited_baseline_rows=observed_inherited,
            expected_new_validation_rows=summary['expected_rows']-inherited,new_validation_rows=summary['observed_rows']-observed_inherited,
            expected_new_training_starts=len(specs)*(1 if pilot else 3),expected_inherited_training_starts=0,
            interpretation='same_target_SD_within_pipeline_conditional_mapping_gain_comparison_not_absolute_calibration')
    write_json(out/'summary.json',summary)
    return summary


def read_gain_prior_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if (cfg['schema']!='shared_driver_gain_prior_v1'
            or cfg['source_run']!='experiments/runs/physiology_semantic_tokenizer/ssm_overnight/20260917_measurement_retest_v1'
            or cfg['protected_boundary']!='only_exact_parent_prepared_arrays_no_raw_or_protected_loaders'
            or cfg['subjects']!=['subject_01','subject_09','subject_18']
            or cfg['outer_folds']!=[0,1,2,3] or cfg['sessions']!=['session_01','session_03','session_05']
            or cfg['original_positions']!=[0,1,2,3,5,6,7,8]
            or cfg['tensor']!=dict(trials_per_subject=24,steps=120,components=list(MODALITIES),dt_s=.25)
            or cfg['methods']!=GAIN_PRIOR_METHODS or cfg['modes']!=['full','center_EEG','center_fNIRS']
            or cfg['reference_tau_s']!=2. or cfg['center_steps']!=16):
        raise ValueError('gain-prior scope/tensor/method contract mismatch')
    train=cfg['gain_training'];regularization=cfg['regularization'];nonlinear=cfg['nonlinear'];syn=cfg['synthetic']
    if (train['bounds']!=[.125,8.] or train['start_values']!=[.5,1.,2.]
            or train['prior_mean']!=1. or not np.isclose(train['prior_log_sd'],np.log(2.),rtol=0,atol=1e-12)
            or train['max_evaluations']!=(3600 if 'continuation' in cfg else 1800)
            or train['max_iterations']!=(90 if 'continuation' in cfg else 60) or train['gradient_tolerance']!=1e-4
            or regularization['driver_amplitude_weight']!=1.
            or regularization['driver_curvature_weight']!=.01 or regularization['initial_state_weight']!=100.
            or nonlinear['max_evaluations']!=160 or nonlinear['substeps']!=4 or nonlinear['integration_check_substeps']!=8
            or nonlinear['gradient_tolerance']!=1e-4
            or syn['gains']!=[.5,1.,2.] or syn['spectra']!=['slow','mixed'] or syn['replicates']!=2):
        raise ValueError('gain-prior numerical/synthetic contract mismatch')
    if (train['parameter_name']!='neurovascular_gain'
            or regularization['driver_prior_sd']!='frozen_training_EEG_normalizer_divided_by_fixed_EEG_loading'
            or cfg['normalizer']!='frozen_parent_outer_training_SD_per_component_unchanged_for_all_candidates'
            or cfg['mask']!='feature_missing_before_linear_processing_with_visible_interpolation'
            or cfg['target']!='parent_processed_EEG_log_power_PCA_and_relative_HbO_HbR'
            or cfg['fixed']!=dict(kappa=.64,gamma=.32,alpha=.32,E0=.32,P0=1.,Q0=.35,
                                 neurovascular_gain=1.,eeg_loading=1.)):
        raise ValueError('gain-prior parameter/coordinate contract mismatch')
    if (not np.isfinite(nonlinear['maximum_integration_difference_training_sd'])
            or nonlinear['maximum_integration_difference_training_sd']<=0
            or not np.isfinite([syn['driver_sd'],syn['initial_state_sd'],syn['noise_fraction'],syn['fast_fraction']]).all()
            or syn['driver_sd']<=0 or syn['initial_state_sd']<0 or syn['noise_fraction']<0
            or not 0<=syn['fast_fraction']<=1
            or not np.isfinite(train['information_rcond']) or not 0<train['information_rcond']<1):
        raise ValueError('gain-prior positive finite generation/integration settings required')
    parameters(cfg,2.).validate()
    budget=cfg['resources']
    if (not 1<=budget['max_workers']<=48 or budget['numerical_threads_per_worker']!=1
            or not 1<=budget['max_in_flight']<=48):
        raise ValueError('gain-prior resource contract mismatch')
    return cfg


def gain_prior_group_specs(cfg,kind):
    if kind=='measured':return nonlinear_group_specs(cfg,kind)
    return [dict(kind=kind,group=f'synthetic_gain{gain:g}_{spectrum}_r{rep}',
                 subject=f'synthetic_gain{gain:g}_{spectrum}_r{rep}',outer=0,true_gain=float(gain),
                 true_tau=2.,spectrum=spectrum,replicate=rep)
            for gain in cfg['synthetic']['gains'] for spectrum in cfg['synthetic']['spectra']
            for rep in range(cfg['synthetic']['replicates'])]


GAIN_CONTINUATION = dict(
    parent_run='experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_gain_prior_v1',
    eligibility='failed_numerical_evaluation_budget_without_domain_or_derivative_rejections',
    expected_measured_restarts=35,
    initialization='last_valid_parent_training_solution_with_reset_damping',
    unchanged_starts='retain_parent_terminal_evidence',
    synthetic='reuse_parent_completed_evidence_after_synthetic_restart_tests',
    validation='reuse_parent_synthetic_and_measured_fixed_gain_cells_refit_measured_trainable_gain_cells')
GAIN_BUDGET_TEXT = dict(
    stopping='35_measured_no_domain_budget_starts_at_most_90_additional_iterations_3600_additional_trial_evaluations_each_other_starts_retained_complete_original_72_validation_identities_no_scope_expansion',
    phase_rule='separate_synthetic_and_measured_calls_parent_terminal_and_synthetic_restart_checks_before_measured_continuation')
GAIN_BUDGET_CLAIMS = [
    'inherited_synthetic_and_fixed_gain_evidence_is_not_an_independent_replication',
    'warm_restarts_reset_LM_damping_and_do_not_reproduce_the_exact_uninterrupted_optimizer_path',
    'unchanged_failed_domain_starts_and_complete_identity_denominators_are_retained']


def gain_restart_eligible(record):
    attempts=record.get('starts',[])
    return bool(record.get('status')=='failed_numerical' and not record.get('converged')
        and record.get('expected_trials')==18 and record.get('completed_trials')==18
        and len(attempts)==1 and attempts[0].get('convergence_reason')=='evaluation_budget'
        and attempts[0].get('domain_rejections')==0 and attempts[0].get('derivative_rejections')==0
        and np.isfinite(record.get('objective',np.nan)) and np.isfinite(record.get('parameter_value',np.nan))
        and len(record.get('physical_checks',[]))==18
        and all(x.get('all_intermediate_valid',False) for x in record['physical_checks']))


def validate_gain_continuation(cfg,project_root):
    """Read only parent JSON identities before allowing any parent array access."""
    if (cfg['experiment_id']!='SSM-SHARED-DRIVER-GAIN-PRIOR-BUDGET-v1'
            or cfg['continuation']!=GAIN_CONTINUATION
            or any(cfg[k]!=v for k,v in GAIN_BUDGET_TEXT.items())
            or cfg['gain_training']['initialization']!='retain_original_start_lineage_restart_eligible_last_valid_training_solution'
            or cfg['claim_limits'][-3:]!=GAIN_BUDGET_CLAIMS):
        raise ValueError('gain continuation contract mismatch')
    parent=Path(project_root)/cfg['continuation']['parent_run']
    manifest=json.loads((parent/'manifest.json').read_text())
    if (manifest.get('execution')!='completed' or manifest.get('pilot') is not False
            or manifest.get('experiment_id')!='SSM-SHARED-DRIVER-GAIN-PRIOR-v1'
            or not manifest.get('synthetic_terminal') or not manifest.get('measured_terminal')
            or Path(manifest['project_root']).resolve()!=Path(project_root).resolve()
            or Path(manifest['source_root']).resolve()!=(parent/'source_snapshot').resolve()):
        raise ValueError('continuation parent is not completed with exact source identity')
    original=read_gain_prior_config(parent/'resolved_config.yaml')
    proposed=json.loads(json.dumps(cfg));proposed.pop('continuation')
    proposed['experiment_id']=original['experiment_id']
    for key in ('max_evaluations','max_iterations','initialization'):
        proposed['gain_training'][key]=original['gain_training'][key]
    for key in GAIN_BUDGET_TEXT:proposed[key]=original[key]
    proposed['claim_limits']=proposed['claim_limits'][:-3]
    if proposed!=original:raise ValueError('continuation changes parent objective or scope/config')
    specs=[s for kind in ('synthetic','measured') for s in gain_prior_group_specs(cfg,kind)]
    restarts=0
    for spec in specs:
        prep=json.loads((parent/'prepared'/f"{spec['group']}.json").read_text())
        validation=[i for i in range(24) if i%4==spec['outer']]
        train=[i for i in range(24) if i not in validation]
        if (prep.get('spec')!=spec or prep.get('status')!='completed'
                or prep['metadata']['train']!=train or prep['metadata']['validation']!=validation):
            raise ValueError('continuation parent preparation identity mismatch')
        if spec['kind']=='measured':
            meta=prep['metadata']
            expected=[(session,position) for session in cfg['sessions'] for position in cfg['original_positions']]
            observed=[(row['session'],row['original_ma_trial_position']) for row in meta['trials']]
            if meta['subject']!=spec['subject'] or meta['outer']!=spec['outer'] or observed!=expected:
                raise ValueError('continuation parent measured provenance mismatch')
        for amplitude in (0,1):
            directory=gain_training_directory(parent,spec,amplitude)
            selection=json.loads((directory/'selection.json').read_text())
            if selection.get('terminal_starts')!=3 or selection.get('expected_starts')!=3:
                raise ValueError('continuation parent training is incomplete')
            for index,gain in enumerate(cfg['gain_training']['start_values']):
                record=json.loads((directory/f'start_{index}'/'result.json').read_text())
                if (record.get('spec')!=spec or record.get('start_index')!=index
                        or record.get('start_gain')!=gain or record.get('amplitude_weight')!=amplitude
                        or record.get('training_trials')!=train or record.get('status') not in ('completed','failed_numerical')
                        or (spec['kind']=='synthetic' and not record.get('converged'))):
                    raise ValueError('continuation parent start identity/status mismatch')
                restarts+=int(spec['kind']=='measured' and gain_restart_eligible(record))
        for method in GAIN_PRIOR_METHODS:
            for mode in cfg['modes']:
                cell=json.loads((parent/'cells'/nonlinear_cell_id(spec,method,mode)/'result.json').read_text())
                rows=cell.get('rows',[])
                if (cell.get('spec')!=spec or cell.get('method')!=method or cell.get('mode')!=mode
                        or [r.get('trial') for r in rows]!=validation
                        or any(r.get('subject')!=spec['subject'] or r.get('outer')!=spec['outer']
                               or r.get('method')!=method or r.get('mode')!=mode for r in rows)
                        or (spec['kind']=='synthetic' and any(r.get('status')!='completed' or not r.get('converged') for r in rows))):
                    raise ValueError('continuation parent cell identity/denominator mismatch')
    if restarts!=cfg['continuation']['expected_measured_restarts']:
        raise ValueError('continuation eligible measured restart count mismatch')
    return parent


def inherit_gain_evidence(source,destination,*,action):
    """Copy exact arrays and retain terminal status; never mutate parent evidence."""
    source=Path(source);destination=Path(destination)
    if source.resolve()==destination.resolve() or destination.exists():
        raise ValueError('continuation inheritance cannot overwrite evidence')
    record=json.loads(source.read_text())
    destination.parent.mkdir(parents=True,exist_ok=True)
    arrays=source.with_suffix('.npz') if source.name!='result.json' else source.parent/'trajectories.npz'
    target=destination.with_suffix('.npz') if destination.name!='result.json' else destination.parent/'trajectories.npz'
    if not arrays.is_file():raise ValueError('continuation parent arrays missing')
    temporary=target.with_suffix('.npz.tmp');shutil.copyfile(arrays,temporary);temporary.replace(target)
    record['continuation']=dict(action=action,parent_record=str(source.resolve()),inherited=True)
    write_json(destination,record)
    return record


def gain_restart_arrays(parent_record,parent_arrays,target,sd,train):
    with np.load(parent_arrays,allow_pickle=False) as a:
        if (not np.array_equal(a['trial_indices'],train) or not np.array_equal(a['target'],target)
                or not np.array_equal(a['normalizer'],sd)):
            raise ValueError('continuation training target/normalizer/train_indices mismatch')
        driver=a['driver'].copy();initial=a['initial_state'].copy()
    if (driver.shape!=target.shape[:2] or initial.shape!=(18,5)
            or not np.isfinite(driver).all() or not np.isfinite(initial).all()):
        raise ValueError('continuation last valid training solution mismatch')
    return dict(parameter_value=float(parent_record['parameter_value']),driver=driver,initial_state=initial)


def prepare_gain_prior_group(payload):
    cfg,out,project_root,spec,pilot=payload
    out=Path(out);path=out/'prepared'/f"{spec['group']}.json"
    if path.exists():return json.loads(path.read_text())
    started=time.monotonic()
    if 'continuation' in cfg:
        source=Path(project_root)/cfg['continuation']['parent_run']/'prepared'/path.name
        return inherit_gain_evidence(source,path,action='inherited_preparation')
    try:
        if spec['kind']=='synthetic':
            generated_cfg=dict(cfg,fixed=dict(cfg['fixed'],neurovascular_gain=spec['true_gain']))
            arrays,meta=nonlinear_synthetic_panel(generated_cfg,spec)
        else:arrays,meta=load_measured(cfg,project_root,spec['subject'],spec['outer'])
        prior_sd=float(arrays['normalizer'][0]/cfg['fixed']['eeg_loading'])
        if not np.isfinite(prior_sd) or prior_sd<=0:raise ValueError('invalid frozen driver prior SD')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix('.npz.tmp').open('wb') as stream:np.savez_compressed(stream,**arrays)
        path.with_suffix('.npz.tmp').replace(path.with_suffix('.npz'))
        result=dict(status='completed',spec=spec,metadata=meta,profiles=[],
            selected_tau={method:2. for method in GAIN_PRIOR_METHODS},driver_prior_sd=prior_sd,
            prior_sd_source='frozen_outer_training_EEG_SD_divided_by_fixed_EEG_loading',
            seconds=time.monotonic()-started)
    except Exception as exc:
        result=dict(status='failed_preparation',spec=spec,error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started)
    write_json(path,result)
    return result


def gain_training_directory(out,spec,amplitude):
    if isinstance(amplitude,str):
        return Path(out)/'training'/f"{spec['group']}__{amplitude}"
    return Path(out)/'training'/f"{spec['group']}__amplitude{int(amplitude)}"


def train_gain_prior_start(payload):
    """One complete shared-parameter start is one worker; never nest a pool."""
    from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_parameter
    cfg,out,spec,amplitude,start_index,start_gain=payload
    destination=gain_training_directory(out,spec,amplitude)/f'start_{start_index}'
    initial_branch=cfg['schema']=='shared_driver_fixed_roi_initial_v1'
    conditional_branch=cfg['schema']=='shared_driver_conditional_optical_gain_v1'
    optical_branch=cfg['schema']=='shared_driver_fixed_roi_optical_v1'
    flow_branch=conditional_branch or initial_branch or optical_branch or cfg['schema']=='shared_driver_fixed_roi_flow_v1'
    tau_branch=flow_branch or cfg['schema']=='shared_driver_fixed_roi_tau_v1'
    method=amplitude if tau_branch else None
    if tau_branch:amplitude=0.
    path=destination/'result.json'
    if path.exists():return json.loads(path.read_text())
    started=time.monotonic();prepared=json.loads((Path(out)/'prepared'/f"{spec['group']}.json").read_text())
    parent_record=None;parent_path=None
    if 'continuation' in cfg:
        parent=Path(prepared['continuation']['parent_record']).parent.parent
        parent_path=gain_training_directory(parent,spec,amplitude)/f'start_{start_index}'/'result.json'
        parent_record=json.loads(parent_path.read_text())
        if spec['kind']=='synthetic' or not gain_restart_eligible(parent_record):
            return inherit_gain_evidence(parent_path,path,action='inherited_training')
    record=dict(spec=spec,amplitude_weight=float(amplitude),start_index=start_index,start_gain=start_gain,
        status='failed_preparation',converged=False,expected_trials=18,completed_trials=0)
    if flow_branch:record.update(**flow_options(cfg,method))
    if initial_branch:record['tie_total_hb_to_volume']=method.endswith('_tied')
    if tau_branch:
        record.pop('start_gain')
        record.update(method=method,start_tau=start_gain)
        if conditional_branch:record.pop('start_tau');record['start_relative_gain']=float(start_gain)
    if parent_record is not None:
        record['continuation']=dict(action='restarted_training',inherited=False,parent_record=str(parent_path),
            initialization='last_valid_training_solution_reset_damping_not_exact_optimizer_resume',
            original_start_gain=start_gain,parent_gain=parent_record['parameter_value'],resumed_gain=None,
            parent_evaluations=parent_record['evaluations'],new_evaluations=0,
            cumulative_evaluations=parent_record['evaluations'],parent_objective=parent_record['objective'],
            initial_objective=None,objective_continuity=False,objective_continuity_rtol=1e-10,objective_continuity_atol=1e-8)
    result={};saved={}
    try:
        if prepared['status']!='completed':record['error']=prepared.get('error')
        else:
            meta=prepared['metadata'];train=meta['train']
            if (len(train)!=18 or len(set(train))!=18 or len(meta['validation'])!=6
                    or set(train)&set(meta['validation']) or sorted(train+meta['validation'])!=list(range(24))):
                raise ValueError('gain training split is not 18 disjoint trials')
            with np.load(Path(out)/'prepared'/f"{spec['group']}.npz",allow_pickle=False) as a:
                # Only the declared training targets enter this optimizer.
                target=a['target'][train].copy();sd=a['normalizer'].copy()
            n=cfg['tensor']['steps'];dt=cfg['tensor']['dt_s'];operator=view_operators(cfg,'full')[2]
            p=parameters(cfg,2.);options=cfg['tau_training'] if tau_branch else cfg['gain_training']
            if conditional_branch:
                mapping=prepared['conditional_mapping'];ref=mapping['beta_reference']
                operator=conditional_prediction_operator(operator,mapping,method)
                options=dict(cfg['conditional_gain_training'],bounds=np.asarray(cfg['conditional_gain_training']['relative_bounds'])*ref)
                record.update(start_relative_gain=float(start_gain),beta_reference=ref)
                start_gain=float(start_gain)*ref
                record.pop('start_tau',None);record.update(start_gain=start_gain)
            prior_sd=float(prepared['driver_prior_sd'])
            starts=[dict(parameter_value=float(start_gain),driver=np.zeros((len(train),n)),
                         initial_state=np.tile(np.r_[0.,np.ones(4)],(len(train),1)))]
            trace_options=dict(record_trace=True) if tau_branch else {}
            if flow_branch:trace_options.update(flow_options(cfg,method))
            if conditional_branch:trace_options.update(initial_coordinates='independent_logs',tie_total_hb_to_volume=False)
            if initial_branch:trace_options['tie_total_hb_to_volume']=method.endswith('_tied')
            if parent_record is not None:
                starts=[gain_restart_arrays(parent_record,parent_path.parent/'trajectories.npz',target,sd,train)]
                trace_options=dict(record_trace=True)
            use_prior=(not tau_branch or method=='fixed_roi_prior_tau') and not conditional_branch
            result=fit_nonlinear_shared_parameter(target,p,dt,parameter_name='tau' if tau_branch and not conditional_branch else 'neurovascular_gain',
                parameter_bounds=options['bounds'],mean_operator=operator,sd=sd,
                penalty=.01,initial_penalty=100.,parameter_prior_mean=options['prior_mean'] if use_prior else None,
                parameter_prior_log_sd=options['prior_log_sd'] if use_prior else None,starts=starts,
                max_evaluations=options['max_evaluations'],max_iterations=options['max_iterations'],
                gradient_tolerance=options['gradient_tolerance'],substeps=cfg['nonlinear']['substeps'],
                information_rcond=options['information_rcond'],driver_amplitude_weight=float(amplitude),driver_prior_sd=prior_sd,**trace_options)
            record.update({k:v for k,v in result.items() if k not in
                           ('driver','initial_state','states','prediction','canonical_prediction','jacobian')})
            record.update(training_trials=train,driver_prior_sd=prior_sd,parameter_name='tau' if tau_branch and not conditional_branch else 'neurovascular_gain')
            if parent_record is not None:
                trace=result.get('trace',[])
                first=trace[0]['objective'] if trace else None
                continuous=bool(first is not None and np.isclose(first,parent_record['objective'],rtol=1e-10,atol=1e-8))
                record['continuation']=dict(action='restarted_training',inherited=False,parent_record=str(parent_path),
                    initialization='last_valid_training_solution_reset_damping_not_exact_optimizer_resume',
                    original_start_gain=start_gain,parent_gain=parent_record['parameter_value'],resumed_gain=result.get('parameter_value'),
                    parent_evaluations=parent_record['evaluations'],new_evaluations=result.get('evaluations',0),
                    cumulative_evaluations=parent_record['evaluations']+result.get('evaluations',0),
                    parent_objective=parent_record['objective'],initial_objective=first,objective_continuity=continuous,
                    objective_continuity_rtol=1e-10,objective_continuity_atol=1e-8,
                    parent_optimization_evaluations=parent_record.get('optimization_evaluations'),
                    new_optimization_evaluations=result.get('optimization_evaluations'),
                    cumulative_optimization_evaluations=(parent_record.get('optimization_evaluations',0)+result.get('optimization_evaluations',0)))
                if not continuous:record.update(status='failed_objective_continuity',converged=False)
            if conditional_branch:record['relative_gain']=result.get('parameter_value')/ref if result.get('parameter_value') is not None else None
            for key in ('driver','initial_state','states','prediction','canonical_prediction'):
                if key in result:saved[key]=result[key]
            saved.update(trial_indices=np.asarray(train),normalizer=sd,target=target)
            record['integration_checks']=[]
            if 'driver' in result and 'initial_state' in result:
                gain=float(result['parameter_value'])
                fitted=parameters(cfg,gain) if tau_branch and not conditional_branch else replace(p,fixed=replace(p.fixed,neurovascular_gain=gain))
                checks=[]
                for index,trial in enumerate(train):
                    fine=replay_nonlinear_driver(result['driver'][index],result['initial_state'][index],fitted,dt,
                        substeps=cfg['nonlinear']['integration_check_substeps'])
                    gap=float('inf')
                    if fine['status']=='completed':
                        pred=(operator@fine['canonical_prediction'].ravel()).reshape(n,3)
                        gap=float(np.max(abs(pred-result['prediction'][index])/sd))
                    checks.append(dict(trial=trial,status=fine['status'],difference_training_sd=gap,
                        state_max_abs_difference=(np.max(abs(fine['states']-result['states'][index]),axis=0)
                            if fine['status']=='completed' else None),failure=fine.get('failure')))
                record['integration_checks']=checks
                if record.get('status')=='completed' and record.get('converged'):
                    if any(c['status']!='completed' for c in checks):
                        record.update(status='failed_domain',converged=False)
                    elif max(c['difference_training_sd'] for c in checks)>cfg['nonlinear']['maximum_integration_difference_training_sd']:
                        record.update(status='failed_integration_check',converged=False)
            elif record.get('status')=='completed':
                record.update(status='failed_missing_trajectory',converged=False)
    except Exception as exc:
        record.update(status='failed_exception',converged=False,error=repr(exc),traceback=traceback.format_exc())
    destination.mkdir(parents=True,exist_ok=True)
    if saved:
        with (destination/'trajectories.npz.tmp').open('wb') as stream:np.savez_compressed(stream,**saved)
        (destination/'trajectories.npz.tmp').replace(destination/'trajectories.npz')
    record.update(seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_json(path,record)
    return record


def select_gain_prior_training(out,spec,amplitude,start_indices):
    """Select only after every declared start is terminal; never use a failed gain."""
    directory=gain_training_directory(out,spec,amplitude);path=directory/'selection.json'
    if path.exists():return json.loads(path.read_text())
    starts=[]
    for index in start_indices:
        p=directory/f'start_{index}'/'result.json'
        if not p.exists():raise ValueError('gain selection requires all declared training starts terminal')
        record=json.loads(p.read_text());record['task_start_index']=index;starts.append(record)
    eligible=[r for r in starts if r.get('status')=='completed' and r.get('converged')
              and r.get('completed_trials')==18 and r.get('expected_trials')==18
              and np.isfinite(r.get('objective',np.nan)) and np.isfinite(r.get('parameter_value',np.nan))]
    result=dict(spec=spec,amplitude_weight=amplitude,expected_starts=len(start_indices),terminal_starts=len(starts),
        successful_starts=len(eligible),expected_training_trials=18,
        starts=[dict(start_index=r['task_start_index'],status=r['status'],converged=r.get('converged',False),
                     objective=r.get('objective'),parameter_value=r.get('parameter_value')) for r in starts],
        status='failed_training',converged=False,parameter_value=None)
    if isinstance(amplitude,str):result.update(method=amplitude,amplitude_weight=0.,parameter_name='neurovascular_gain' if amplitude=='conditional_trained_gain' else 'tau')
    if eligible:
        best=min(eligible,key=lambda r:(r['objective'],r['task_start_index']))
        result.update(status='completed',converged=True,parameter_value=float(best['parameter_value']),
            objective=best['objective'],selected_training_start=best['task_start_index'],
            source_result=str(directory/f"start_{best['task_start_index']}"/'result.json'))
    if amplitude=='conditional_trained_gain' and eligible:
        result.update(beta_reference=best['beta_reference'],relative_gain=best['relative_gain'])
    write_json(path,result)
    return result


def summarize_gain_prior(out,plan,pilot):
    summary=summarize_nonlinear_fit(Path(out),plan,pilot,methods=GAIN_PRIOR_METHODS)
    specs={s['group']:s for s,_,_ in plan};training=[];continuation_actions={}
    for spec in specs.values():
        for amplitude in ([1] if pilot else [0,1]):
            for index in ([1] if pilot else [0,1,2]):
                start_path=gain_training_directory(out,spec,amplitude)/f'start_{index}'/'result.json'
                if start_path.exists():
                    action=json.loads(start_path.read_text()).get('continuation',{}).get('action')
                    if action:continuation_actions[action]=continuation_actions.get(action,0)+1
            p=gain_training_directory(out,spec,amplitude)/'selection.json'
            if p.exists():
                d=json.loads(p.read_text());training.append(dict(group=spec['group'],kind=spec['kind'],
                    amplitude_weight=amplitude,status=d['status'],expected_starts=d['expected_starts'],
                    terminal_starts=d['terminal_starts'],successful_starts=d['successful_starts'],parameter_value=d['parameter_value']))
    summary.update(training=training,expected_training_groups=len(specs)*(1 if pilot else 2),
        terminal_training_groups=len(training),successful_training_groups=sum(r['status']=='completed' for r in training))
    if continuation_actions:
        summary['continuation_training_actions']=continuation_actions
        summary['continuation_interpretation']='inherited_evidence_is_not_independent_replication; restarted_training_resets_damping'
    write_json(Path(out)/'summary.json',summary)
    return summary


def gain_prior_fit_main(args):
    conditional_branch=getattr(args,'conditional_optical_gain_fit',False)
    optical_branch=getattr(args,'fixed_roi_optical_fit',False)
    initial_branch=getattr(args,'fixed_roi_initial_fit',False)
    flow_branch=conditional_branch or optical_branch or initial_branch or getattr(args,'fixed_roi_flow_fit',False)
    tau_branch=flow_branch or getattr(args,'fixed_roi_tau_fit',False)
    cfg=(read_conditional_optical_gain_config(args.config) if conditional_branch else read_fixed_roi_optical_config(args.config) if optical_branch else read_fixed_roi_initial_config(args.config) if initial_branch else
         read_fixed_roi_flow_config(args.config) if flow_branch else
         read_fixed_roi_tau_config(args.config) if tau_branch else read_gain_prior_config(args.config))
    group_specs=conditional_gain_specs if conditional_branch else optical_group_specs if optical_branch else initial_group_specs if initial_branch else nonlinear_group_specs if tau_branch else gain_prior_group_specs
    prepare=prepare_conditional_gain_group if conditional_branch else prepare_fixed_roi_optical_group if optical_branch else prepare_fixed_roi_flow_group if flow_branch else prepare_fixed_roi_tau_group if tau_branch else prepare_gain_prior_group
    summarize=summarize_fixed_roi_tau if tau_branch else summarize_gain_prior
    if 'continuation' in cfg:
        validate_gain_continuation(cfg,args.project_root.resolve())
        import inspect
        from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_parameter
        if 'record_trace' not in inspect.signature(fit_nonlinear_shared_parameter).parameters:
            raise ValueError('continuation requires shared-parameter trace interface')
        if args.pilot:raise ValueError('continuation uses synthetic restart tests, not a new pilot campaign')
    if flow_branch:
        (validate_conditional_gain_parent if conditional_branch else validate_optical_parent if optical_branch else validate_initial_parent if initial_branch else validate_flow_parent)(cfg,args.project_root.resolve())
        import inspect
        from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_parameter,fit_nonlinear_shared_driver
        if any('flow_prior_weight' not in inspect.signature(f).parameters for f in
                (fit_nonlinear_shared_parameter,fit_nonlinear_shared_driver)):
            raise ValueError('flow candidate requires both library flow-prior interfaces')
        if initial_branch and any('tie_total_hb_to_volume' not in inspect.signature(f).parameters for f in
                (fit_nonlinear_shared_parameter,fit_nonlinear_shared_driver,fit_shared_driver)):
            raise ValueError('initial candidate requires reduced-coordinate nonlinear and linear interfaces')
    if args.check_only:
        from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_parameter
        assert callable(fit_nonlinear_shared_parameter)
        synthetic_count=36 if initial_branch else 12
        measured_count=24 if conditional_branch or optical_branch else 12
        assert len(group_specs(cfg,'synthetic'))==synthetic_count and len(group_specs(cfg,'measured'))==measured_count
        print(json.dumps(dict(status='passed',source_arrays_read=0,synthetic_groups=synthetic_count,measured_groups=measured_count,
            training_starts_per_group=3 if conditional_branch or optical_branch else 6 if initial_branch else 9 if flow_branch else 6,
            new_measured_training_starts=72 if conditional_branch or optical_branch else 36 if initial_branch else None,
            validation_rows_per_group=54 if conditional_branch else 36 if optical_branch else 72 if initial_branch else 108 if flow_branch else 54 if tau_branch else 72)))
        return
    if args.phase=='all':raise ValueError('gain-prior requires explicit synthetic then separately requested measured phase')
    if args.pilot and args.phase!='synthetic':raise ValueError('gain-prior pilot is synthetic only')
    if args.run_dir is None or not 1<=args.workers<=cfg['resources']['max_workers']:
        raise ValueError('run-dir and bounded workers required')
    out=args.run_dir.resolve();project_root=args.project_root.resolve()
    if out.parent!=(project_root/cfg['output_root']).resolve():raise ValueError('run must be a child of the owning output root')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest_path=out/'manifest.json';resolved=out/'resolved_config.yaml'
    old=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if old:
        if old['execution']=='completed':raise ValueError('completed evidence is immutable')
        if (old.get('pilot')!=args.pilot or old.get('source_root')!=str(CODE_ROOT)
                or old.get('project_root')!=str(project_root) or old.get('experiment_id')!=cfg['experiment_id']):
            raise ValueError('resume source/project/pilot identity mismatch')
    if resolved.exists() and yaml.safe_load(resolved.read_text())!=cfg:raise ValueError('resume config mismatch')
    if not resolved.exists():resolved.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    specs={k:group_specs(cfg,k) for k in ('synthetic','measured')}
    if args.pilot:
        chosen=([s for s in specs['synthetic'] if s['true_tau']==2. and s['spectrum']=='mixed' and s['replicate']==0]
                if tau_branch else specs['synthetic'][:1])
        if conditional_branch:chosen=[s for s in chosen if s['true_relative_gain']==1.]
        if initial_branch:chosen=[s for s in chosen if s['true_initial_log_p_over_v']==0.]
        specs=dict(synthetic=chosen,measured=[])
    methods=(FIXED_ROI_TAU_METHODS[1:] if args.pilot else FIXED_ROI_TAU_METHODS) if tau_branch else (
        ['fixed_gain_amplitude','trained_gain_amplitude'] if args.pilot else GAIN_PRIOR_METHODS)
    if flow_branch:
        methods=CONDITIONAL_GAIN_METHODS if conditional_branch else FIXED_ROI_OPTICAL_METHODS if optical_branch else FIXED_ROI_INITIAL_METHODS if initial_branch else (['fixed_tau_logflow1','trained_tau_logflow1'] if args.pilot else FIXED_ROI_FLOW_METHODS)
    modes=['full'] if args.pilot else cfg['modes']
    plan=[(s,m,v) for kind in specs for s in specs[kind] for m in methods for v in modes]
    if args.phase=='measured':
        prior=summarize(out,[t for t in plan if t[0]['kind']=='synthetic'],False)
        if (prior['terminal_cells']!=prior['expected_cells'] or prior['observed_rows']!=prior['expected_rows']
                or prior['terminal_training_groups']!=prior['expected_training_groups']):
            raise ValueError('complete synthetic terminal evidence required before measured access')
        if conditional_branch and not prior.get('synthetic_engineering_screen',{}).get('all_groups_passed',False):
            raise ValueError('conditional synthetic gain recovery screen failed; measured access stopped')
    if flow_branch:(validate_conditional_gain_parent if conditional_branch else validate_optical_parent if optical_branch else validate_initial_parent if initial_branch else validate_flow_parent)(cfg,project_root,kind=args.phase)
    manifest=dict(old,experiment_id=cfg['experiment_id'],execution='running',pilot=args.pilot,
        started_at=old.get('started_at',datetime.now(timezone.utc).isoformat()),controller_pid=os.getpid(),
        supervisor=os.environ.get('SSM_SYSTEMD_UNIT'),workers=args.workers,
        max_in_flight=min(cfg['resources']['max_in_flight'],2*args.workers),source_root=str(CODE_ROOT),
        project_root=str(project_root),versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    write_json(manifest_path,manifest)
    kind=args.phase;manifest['phase']=kind+'_preparation';write_json(manifest_path,manifest)
    prep=[(cfg,str(out),str(project_root),s,args.pilot) for s in specs[kind]]
    for payload,result,error in bounded_nonlinear_work(prepare,prep,min(args.workers,len(prep)),manifest['max_in_flight']):
        if error:
            result=dict(status='failed_preparation',spec=payload[3],error=error)
            write_json(out/'prepared'/f"{payload[3]['group']}.json",result)
        print(json.dumps(dict(event='prepared',group=result['spec']['group'],status=result['status'])),flush=True)
    manifest['phase']=kind+'_training';write_json(manifest_path,manifest)
    amplitudes=FIXED_ROI_TAU_METHODS[1:] if tau_branch else ([1] if args.pilot else [0,1])
    if conditional_branch:
        amplitudes=['conditional_trained_gain']
    elif optical_branch:
        amplitudes=['trained_tau_logflow1'] if args.pilot or kind=='measured' else []
        if kind=='synthetic' and not args.pilot:
            for spec in specs[kind]:inherit_flow_training(cfg,project_root,out,spec,method='trained_tau_logflow1')
    elif initial_branch:
        amplitudes=FIXED_ROI_INITIAL_TRAINED if kind=='synthetic' else ['trained_tau_logflow1_tied']
        if kind=='measured':
            for spec in specs[kind]:inherit_flow_training(cfg,project_root,out,spec,method='trained_tau_logflow1')
    elif flow_branch:
        amplitudes=['trained_tau_logflow1'] if args.pilot else FIXED_ROI_FLOW_TRAINED
        if not args.pilot:
            for spec in specs[kind]:inherit_flow_training(cfg,project_root,out,spec)
    options=(dict(start_values=cfg['conditional_gain_training']['relative_start_values']) if conditional_branch else cfg['tau_training'] if tau_branch else cfg['gain_training'])
    starts=[(1,1. if conditional_branch else 2. if tau_branch else 1.)] if args.pilot else list(enumerate(options['start_values']))
    work=[(cfg,str(out),s,a,index,gain) for s in specs[kind] for a in amplitudes for index,gain in starts]
    for payload,result,error in bounded_nonlinear_work(train_gain_prior_start,work,max(1,min(args.workers,len(work))),manifest['max_in_flight']):
        if error:
            _,_,spec,amplitude,index,gain=payload
            result=dict(spec=spec,amplitude_weight=amplitude,start_index=index,start_gain=gain,status='failed_worker',
                converged=False,error=error,expected_trials=18,completed_trials=0)
            if tau_branch:result.update(method=amplitude,amplitude_weight=0.,start_tau=gain)
            if conditional_branch:
                result.pop('start_tau',None);result.pop('start_gain',None);result.update(start_relative_gain=gain,parameter_name='neurovascular_gain')
            write_json(gain_training_directory(out,spec,amplitude)/f'start_{index}'/'result.json',result)
        progress=dict(event='training_start_terminal',group=result['spec']['group'],
            amplitude_weight=result['amplitude_weight'],start_index=result['start_index'],status=result['status'],
            seconds=result.get('seconds'))
        if tau_branch:
            progress.update(parameter_name='neurovascular_gain' if conditional_branch else 'tau',parameter_value=result.get('parameter_value'),method=result.get('method'))
        else:progress['gain']=result.get('parameter_value')
        print(json.dumps(progress),flush=True)
    for spec in specs[kind]:
        for amplitude in amplitudes:select_gain_prior_training(out,spec,amplitude,[index for index,_ in starts])
    manifest['phase']=kind+'_validation';write_json(manifest_path,manifest)
    work=[(cfg,str(out),s,m,v,args.pilot) for s,m,v in plan if s['kind']==kind]
    for payload,result,error in bounded_nonlinear_work(nonlinear_cell,work,min(args.workers,len(work)),manifest['max_in_flight']):
        if error:
            _,_,spec,method,mode,pilot=payload;identifier=nonlinear_cell_id(spec,method,mode)
            validation=[i for i in range(24) if i%4==spec['outer']]
            if pilot:validation=validation[:2]
            result=dict(cell=identifier,status='failed_worker',error=error,rows=[dict(spec,method=method,mode=mode,
                trial=i,session=cfg['sessions'][i//8],status='failed_worker',converged=False) for i in validation])
            write_json(out/'cells'/identifier/'result.json',result)
        print(json.dumps(dict(event='cell_terminal',cell=result['cell'],
            completed=sum(r['status']=='completed' and r.get('converged',False) for r in result['rows']),expected=len(result['rows']))),flush=True)
        manifest['last_completed_cell']=result['cell'];write_json(manifest_path,manifest)
    current_plan=[t for t in plan if t[0]['kind']=='synthetic' or kind=='measured']
    summary=summarize(out,current_plan,args.pilot)
    manifest[kind+'_terminal']=True
    manifest[kind+'_summary']={k:summary[k] for k in ('expected_cells','terminal_cells','expected_rows','observed_rows','completed_rows','successful_training_groups')}
    complete=args.pilot or kind=='measured'
    manifest.update(execution='completed' if complete else 'synthetic_terminal',
        scientific_verdict='conditional_optical_gain_diagnostic_only' if conditional_branch else 'fixed_roi_optical_diagnostic_only' if optical_branch else 'fixed_roi_initial_diagnostic_only' if initial_branch else 'fixed_roi_flow_diagnostic_only' if flow_branch else 'fixed_roi_tau_diagnostic_only' if tau_branch else 'gain_prior_diagnostic_only',summary='summary.json')
    if complete:manifest['completed_at']=datetime.now(timezone.utc).isoformat()
    write_json(manifest_path,manifest)
    print(json.dumps(dict(event='phase_complete',phase=kind,summary=summary)),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=DEFAULT)
    ap.add_argument('--run-dir',type=Path)
    ap.add_argument('--project-root',type=Path,default=CODE_ROOT)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--replay-of',type=Path)
    ap.add_argument('--nonlinear-fit',action='store_true')
    ap.add_argument('--gain-prior-fit',action='store_true')
    ap.add_argument('--fixed-roi-tau-fit',action='store_true')
    ap.add_argument('--fixed-roi-flow-fit',action='store_true')
    ap.add_argument('--fixed-roi-initial-fit',action='store_true')
    ap.add_argument('--fixed-roi-optical-fit',action='store_true')
    ap.add_argument('--conditional-optical-gain-fit',action='store_true')
    ap.add_argument('--phase',choices=['all','synthetic','measured'],default='all')
    args=ap.parse_args()
    if args.conditional_optical_gain_fit:
        if args.fixed_roi_optical_fit or args.fixed_roi_initial_fit or args.fixed_roi_flow_fit or args.fixed_roi_tau_fit or args.gain_prior_fit or args.nonlinear_fit or args.replay_of is not None:
            ap.error('conditional-optical-gain-fit cannot be combined with another fit/replay mode')
        return gain_prior_fit_main(args)
    if args.fixed_roi_optical_fit:
        if args.fixed_roi_initial_fit or args.fixed_roi_flow_fit or args.fixed_roi_tau_fit or args.gain_prior_fit or args.nonlinear_fit or args.replay_of is not None:
            ap.error('fixed-roi-optical-fit cannot be combined with another fit/replay mode')
        return gain_prior_fit_main(args)
    if args.fixed_roi_initial_fit:
        if args.fixed_roi_flow_fit or args.fixed_roi_tau_fit or args.gain_prior_fit or args.nonlinear_fit or args.replay_of is not None:
            ap.error('fixed-roi-initial-fit cannot be combined with another fit/replay mode')
        return gain_prior_fit_main(args)
    if args.fixed_roi_flow_fit:
        if args.fixed_roi_tau_fit or args.gain_prior_fit or args.nonlinear_fit or args.replay_of is not None:
            ap.error('fixed-roi-flow-fit cannot be combined with another fit/replay mode')
        return gain_prior_fit_main(args)
    if args.fixed_roi_tau_fit:
        if args.gain_prior_fit or args.nonlinear_fit or args.replay_of is not None:
            ap.error('fixed-roi-tau-fit cannot be combined with another fit/replay mode')
        return gain_prior_fit_main(args)
    if args.gain_prior_fit:
        if args.nonlinear_fit or args.replay_of is not None:ap.error('gain-prior-fit cannot be combined with another fit/replay mode')
        return gain_prior_fit_main(args)
    if args.nonlinear_fit:
        if args.replay_of is not None:ap.error('nonlinear-fit and replay-of are mutually exclusive')
        return nonlinear_fit_main(args)
    if args.replay_of is not None:
        return replay_main(args)
    cfg=read_config(args.config)
    if args.check_only:
        op=view_operators(cfg,'full')[2]
        d=build_shared_driver_design(parameters(cfg,2.),120,.25,processed_mean_operator=op)
        assert d.observation_design.shape==(360,125)
        print(json.dumps(dict(status='passed',grid_points=len(tau_grid(cfg)),measured_arrays_read=0)))
        return
    if args.run_dir is None or not 1<=args.workers<=cfg['resources']['max_workers']:
        ap.error('run-dir and a bounded worker count are required')
    out=args.run_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest_path=out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['execution']=='completed':
        raise ValueError('completed evidence is immutable')
    if (out/'resolved_config.yaml').exists():
        if yaml.safe_load((out/'resolved_config.yaml').read_text())!=cfg:
            raise ValueError('resume config differs')
    else:
        (out/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    manifest=dict(experiment_id=cfg['experiment_id'],execution='running',pilot=args.pilot,
        started_at=datetime.now(timezone.utc).isoformat(),controller_pid=os.getpid(),
        supervisor=os.environ.get('SSM_SYSTEMD_UNIT'),workers=args.workers,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        source_root=str(CODE_ROOT),project_root=str(args.project_root.resolve()))
    write_json(manifest_path,manifest)
    (out/'groups').mkdir(exist_ok=True)
    expected=1 if args.pilot else 24
    for kind in ('synthetic','measured'):
        if args.pilot and kind=='measured': break
        work=tasks(cfg,out,args.project_root,kind)
        if args.pilot: work=work[:1]
        with ProcessPoolExecutor(max_workers=min(args.workers,len(work))) as pool:
            for future in as_completed([pool.submit(run_group,t) for t in work]):
                result=future.result()
                print(json.dumps({k:result.get(k) for k in ('group','status','seconds','selected_tau','error')}),flush=True)
                manifest['finished_groups']=len(list((out/'groups').glob('*.json')))
                write_json(manifest_path,manifest)
        summary=summarize(out,expected)
        if kind=='synthetic':
            baseline=[a for a in summary['aggregates'] if a['kind']=='synthetic' and a['method']=='fitted_tau' and a['mode']=='full']
            expected_synthetic=len(work)
            truth_rows=[row for p in (out/'groups').glob('synthetic*.json')
                        for row in json.loads(p.read_text())['rows']
                        if row['method']=='fitted_tau' and row['mode']=='full']
            driver_error=float(np.mean([r['driver_nrmse'] for r in truth_rows])) if truth_rows else None
            tau_error=float(np.median([abs(r['tau']/r['true_tau']-1) for r in truth_rows])) if truth_rows else None
            passed=(summary['failed_groups']==0 and summary['completed_groups']==expected_synthetic
                and len(baseline)==1 and baseline[0]['count']==6*expected_synthetic and all(
                baseline[0]['nrmse_'+m]<cfg['synthetic']['maximum_mean_reconstruction_nrmse'] for m in MODALITIES))
            manifest['synthetic_check']=dict(passed=passed,scope='conditional_linear_software_recovery_only',
                mean_driver_nrmse=driver_error,median_tau_relative_error=tau_error,
                physiological_qualification=False,
                interpretation='only reconstruction software gate; state/parameter truth and validity remain separate diagnostic outcomes')
            write_json(manifest_path,manifest)
            if not passed:
                manifest['execution']='stopped_synthetic_check'
                write_json(manifest_path,manifest)
                return
    manifest.update(execution='completed',completed_at=datetime.now(timezone.utc).isoformat(),
                    scientific_verdict='diagnostic_only',summary='summary.json')
    write_json(manifest_path,manifest)


if __name__=='__main__':
    main()
