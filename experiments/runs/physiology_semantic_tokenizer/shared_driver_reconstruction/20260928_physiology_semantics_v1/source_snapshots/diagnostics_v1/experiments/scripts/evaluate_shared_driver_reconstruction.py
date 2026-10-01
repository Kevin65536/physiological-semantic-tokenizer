#!/usr/bin/env python3
"""Shared-driver reconstruction and fixed-structure physiology diagnostics.

Retained diagnostic modes read exact parent projections. The separately
versioned physiology-semantics mode uses central public native readers and
explicit identity/split manifests. No teacher or calibrated posterior is
produced. A durable supervisor owns substantial execution.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
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
    started=time.monotonic();cpu_started=time.process_time()
    prepared=json.loads((Path(out)/'prepared'/f"{spec['group']}.json").read_text())
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
            if 'training_step_control' in cfg:
                trace_options['step_control']=cfg['training_step_control']
                record['step_control']=cfg['training_step_control']
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
    record.update(seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu_started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
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


STEP_CONTROL_SCOPE=['no_motion__subject_01_o1','no_motion__subject_18_o0',
                    'mne_tddr__subject_01_o3','mne_tddr__subject_18_o3']
STEP_CONTROL_CONTRACT=dict(schema='shared_driver_conditional_step_control_v1',
    experiment_id='SSM-SHARED-DRIVER-CONDITIONAL-STEP-CONTROL-v1',
    parent_run='experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_conditional_optical_gain_v1',
    output_root='experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction',
    scope='training_only_retained_synthetic_then_four_declared_public_measured_groups_no_validation_fits',
    step_control='quadratic_interpolation',baseline_step_control='armijo',
    pilot='separate_output_synthetic_g1_mixed_r0_start_index1_no_measured',
    measured_groups=STEP_CONTROL_SCOPE,synthetic_groups=12,synthetic_starts=36,measured_starts=12,
    resources=dict(max_workers=48,max_in_flight=48,numerical_threads_per_worker=1),
    stopping='all_declared_starts_terminal_preserve_failures_no_validation_or_NRMSE_claim',
    phase_rule='synthetic_terminal_and_original_recovery_screen_before_measured_prepared_access')


STEP_CONTROL_PANEL_CONTRACT=dict(STEP_CONTROL_CONTRACT,
    schema='shared_driver_conditional_step_control_v2',
    experiment_id='SSM-SHARED-DRIVER-CONDITIONAL-STEP-CONTROL-v2',
    scope='paired_training_only_all_retained_public_groups_no_validation_fits',
    measured_groups=[f'{pipeline}__{subject}_o{fold}' for pipeline in ('no_motion','mne_tddr')
        for subject in ('subject_01','subject_09','subject_18') for fold in range(4)],
    measured_starts=72,paired_baseline=True,task_order_seed=20260928,
    primary_endpoint='same_budget_success_without_loss_of_any_armijo_success',
    performance_endpoint='at_least_10_percent_lower_total_measured_optimization_evaluations',
    secondary_endpoints=['process_cpu_seconds','wall_seconds','objective','gain','trajectory_agreement'],
    null='no_convergence_or_forward_evaluation_improvement',
    phase_rule='both_arms_synthetic_recovery_and_no_start_regressions_before_measured_access')


def read_conditional_step_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    expected=STEP_CONTROL_PANEL_CONTRACT if cfg.get('schema')=='shared_driver_conditional_step_control_v2' else STEP_CONTROL_CONTRACT
    if {k:v for k,v in cfg.items() if k!='base_contract'}!=expected:
        raise ValueError('step-control diagnostic scope/config mismatch')
    base=read_conditional_optical_gain_config(CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_conditional_optical_gain_v1.yaml')
    if cfg.get('base_contract')!=base:raise ValueError('step-control objective differs from frozen base contract')
    return cfg


def step_control_specs(cfg,kind):
    specs=conditional_gain_specs(cfg['base_contract'],kind)
    return specs if kind=='synthetic' else [s for s in specs if s['group'] in cfg['measured_groups']]


def validate_step_parent(cfg,root,kind):
    """JSON-only fixed parent/identity closure; phase controller gates measured use."""
    parent=Path(root)/cfg['parent_run'];m=json.loads((parent/'manifest.json').read_text())
    if (m.get('execution')!='completed' or m.get('pilot') is not False
            or m.get('experiment_id')!=cfg['base_contract']['experiment_id']
            or m.get('project_root')!=str(Path(root).resolve())
            or not m.get('synthetic_terminal') or not m.get('measured_terminal')
            or yaml.safe_load((parent/'resolved_config.yaml').read_text())!=cfg['base_contract']):
        raise ValueError('step-control parent must be exact completed conditional run')
    for spec in step_control_specs(cfg,kind):
        record=json.loads((parent/'prepared'/f"{spec['group']}.json").read_text())
        train=[i for i in range(24) if i%4!=spec['outer']];val=[i for i in range(24) if i%4==spec['outer']]
        if (record.get('status')!='completed' or record.get('spec')!=spec
                or record['metadata']['train']!=train or record['metadata']['validation']!=val
                or len(record['metadata']['trials'])!=24 or 'conditional_mapping' not in record):
            raise ValueError('step-control prepared identity/18/6 contract mismatch')
        if kind=='measured' and record['metadata']['optical_processing']['pipeline']!=spec['optical_pipeline']:
            raise ValueError('step-control pipeline mismatch')
        for i,g in enumerate(cfg['base_contract']['conditional_gain_training']['relative_start_values']):
            r=json.loads((gain_training_directory(parent,spec,'conditional_trained_gain')/f'start_{i}'/'result.json').read_text())
            if (r.get('spec')!=spec or r.get('start_index')!=i or r.get('start_relative_gain')!=g
                    or r.get('training_trials')!=train or r.get('parameter_name')!='neurovascular_gain'
                    or r.get('status') not in ('completed','failed_numerical')
                    or r.get('tau')!=2. or r.get('parameter_prior_cost')!=0.):
                raise ValueError('step-control baseline start identity/objective mismatch')
    return parent


def prepare_step_control_group(cfg,out,root,spec,start_indices=(0,1,2)):
    if spec not in step_control_specs(cfg,spec.get('kind')):raise ValueError('step-control group outside declared scope')
    parent=Path(root)/cfg['parent_run'];out=Path(out)
    source=parent/'prepared'/f"{spec['group']}.json";record=json.loads(source.read_text());meta=record['metadata']
    with np.load(source.with_suffix('.npz'),allow_pickle=False) as a:
        target=a['target'][meta['train']].copy();sd=a['normalizer'].copy()
        if a['target'].shape!=(24,120,3) or sd.shape!=(3,) or not np.isfinite(target).all() or not np.isfinite(sd).all() or np.any(sd<=0):
            raise ValueError('step-control prepared tensor mismatch')
    sources=[(source,out/'prepared'/source.name),(source.with_suffix('.npz'),out/'prepared'/source.with_suffix('.npz').name)]
    for i in start_indices:
        folder=gain_training_directory(parent,spec,'conditional_trained_gain')/f'start_{i}'
        with np.load(folder/'trajectories.npz',allow_pickle=False) as a:
            if (not np.array_equal(a['trial_indices'],meta['train']) or not np.array_equal(a['target'],target)
                    or not np.array_equal(a['normalizer'],sd)):
                raise ValueError('step-control parent training target/SD/index mismatch')
        dest=out/'baseline_training'/f"{spec['group']}__conditional_trained_gain"/f'start_{i}'
        sources.extend((folder/name,dest/name) for name in ('result.json','trajectories.npz'))
    # Publish only after every declared source artifact has passed closure.
    for src,dest in sources:
        if dest.exists():
            if src.read_bytes()!=dest.read_bytes():raise ValueError('step-control resume inherited bytes differ')
        else:
            dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix(dest.suffix+'.tmp');shutil.copyfile(src,tmp);tmp.replace(dest)
    return record


def summarize_step_control(cfg,out,kind,pilot=False):
    rows=[];specs=step_control_specs(cfg,kind);out=Path(out)
    if pilot:specs=[s for s in specs if s['group']=='synthetic_g1_mixed_r0']
    for spec in specs:
        for i in ([1] if pilot else range(3)):
            path=gain_training_directory(out,spec,'conditional_trained_gain')/f'start_{i}'/'result.json'
            if not path.exists():continue
            r=json.loads(path.read_text());old=json.loads((out/'baseline_training'/f"{spec['group']}__conditional_trained_gain"/f'start_{i}'/'result.json').read_text())
            success=r['status']=='completed' and r.get('converged',False)
            old_success=old['status']=='completed' and old.get('converged',False)
            rows.append(dict(group=spec['group'],start=i,status=r['status'],converged=r.get('converged',False),
                objective=r.get('objective'),relative_gain=r.get('relative_gain'),evaluations=r.get('evaluations'),
                optimization_evaluations=r.get('optimization_evaluations'),
                convergence_reason=r.get('convergence_reason'),seconds=r.get('seconds'),cpu_seconds=r.get('cpu_seconds'),
                iterations=r.get('starts',[{}])[0].get('iterations'),
                scaled_gradient=r.get('projected_scaled_gradient_inf_norm'),
                interpolation_counts=r.get('starts',[{}])[0].get('interpolation_counts',{}),
                baseline_success=old_success,regression=bool(old_success and not success),
                baseline_status=old['status'],baseline_objective=old.get('objective'),baseline_relative_gain=old.get('relative_gain'),
                baseline_evaluations=old.get('evaluations'),baseline_optimization_evaluations=old.get('optimization_evaluations')))
    expected=(1 if pilot else 3)*len(specs);result=dict(kind=kind,expected_starts=expected,terminal_starts=len(rows),
        successful_starts=sum(r['status']=='completed' and r['converged'] for r in rows),rows=rows,
        regressions=sum(r['regression'] for r in rows),
        interpretation='training_numerical_diagnostic_only_baseline_is_inherited_not_independent_no_validation_scores')
    if kind=='synthetic':result['engineering_screen']=conditional_gain_screen(out,specs,pilot=pilot)
    write_json(out/f'{kind}_training_summary.json',result)
    return result


def summarize_step_panel(cfg,out,kind,pilot=False):
    """Pair fresh arms by exact group/start; historical results remain separate."""
    out=Path(out);summaries={arm:summarize_step_control(cfg,out/'arms'/arm,kind,pilot)
        for arm in ('armijo','quadratic_interpolation')}
    indexed={arm:{(r['group'],r['start']):r for r in s['rows']} for arm,s in summaries.items()}
    rows=[]
    for key,old in indexed['armijo'].items():
        new=indexed['quadratic_interpolation'].get(key)
        if new is None:continue
        success=lambda r:r['status']=='completed' and r['converged']
        rows.append(dict(group=key[0],start=key[1],baseline=old,candidate=new,
            recovered=bool(success(new) and not success(old)),regressed=bool(success(old) and not success(new))))
    expected=summaries['armijo']['expected_starts']
    complete=len(rows)==expected and all(s['terminal_starts']==expected for s in summaries.values())
    old_calls=sum(r['baseline'].get('optimization_evaluations') or 0 for r in rows)
    new_calls=sum(r['candidate'].get('optimization_evaluations') or 0 for r in rows)
    cost_records=[r[arm] for r in rows for arm in ('baseline','candidate')]
    costs_available=bool(cost_records) and all(
        r['status'] in ('completed','failed_numerical') and
        isinstance(r.get('optimization_evaluations'),(int,float)) and
        np.isfinite(r['optimization_evaluations']) and 0<r['optimization_evaluations']<=3600
        for r in cost_records)
    regressions=sum(r['regressed'] for r in rows)
    old_success=summaries['armijo']['successful_starts'];new_success=summaries['quadratic_interpolation']['successful_starts']
    numerical_pass=complete and regressions==0 and new_success>=old_success
    recovery_pass=kind!='synthetic' or all(s['engineering_screen']['all_groups_passed'] and s['regressions']==0 for s in summaries.values())
    result=dict(kind=kind,expected_pairs=expected,terminal_pairs=len(rows),complete=complete,
        baseline_successes=old_success,candidate_successes=new_success,
        recovered=sum(r['recovered'] for r in rows),regressions=regressions,
        baseline_optimization_evaluations=old_calls,candidate_optimization_evaluations=new_calls,
        cost_records_valid=bool(costs_available),
        evaluation_reduction_fraction=1-new_calls/old_calls if costs_available and old_calls else None,
        numerical_screen_passed=bool(numerical_pass and recovery_pass),
        performance_screen_passed=bool(complete and numerical_pass and costs_available and old_calls and new_calls<=.9*old_calls),
        rows=rows,interpretation='fresh_paired_training_numerics_no_validation_or_physiology_claim')
    write_json(out/f'{kind}_paired_summary.json',result)
    return result


def conditional_step_control_main(args):
    cfg=read_conditional_step_config(args.config);root=args.project_root.resolve()
    # Check-only never reads measured metadata or any arrays.
    validate_step_parent(cfg,root,'synthetic')
    if args.check_only:
        print(json.dumps(dict(status='passed',source_arrays_read=0,synthetic_starts=cfg['synthetic_starts'],
            measured_starts=cfg['measured_starts'],arms=2 if cfg.get('paired_baseline') else 1,validation_fits=0)));return
    if args.phase not in ('synthetic','measured') or args.pilot and args.phase!='synthetic':
        raise ValueError('step-control requires explicit phase; pilot is synthetic only')
    if args.run_dir is None or not 1<=args.workers<=cfg['resources']['max_workers']:raise ValueError('bounded workers and run-dir required')
    out=args.run_dir.resolve()
    if out.parent!=(root/cfg['output_root']).resolve():raise ValueError('step-control output outside owning namespace')
    out.mkdir(parents=True,exist_ok=True);lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=out/'manifest.json';old=json.loads(path.read_text()) if path.exists() else {}
    if old and (old.get('execution')=='completed' or old.get('source_root')!=str(CODE_ROOT)
            or old.get('project_root')!=str(root) or old.get('experiment_id')!=cfg['experiment_id'] or old.get('pilot')!=args.pilot):
        raise ValueError('step-control immutable/resume identity mismatch')
    resolved=out/'resolved_config.yaml'
    if resolved.exists() and yaml.safe_load(resolved.read_text())!=cfg:raise ValueError('step-control resume config mismatch')
    if args.phase=='measured':
        syn=(summarize_step_panel if cfg.get('paired_baseline') else summarize_step_control)(cfg,out,'synthetic')
        passed=(syn['numerical_screen_passed'] if cfg.get('paired_baseline') else
            syn['terminal_starts']==36 and syn['regressions']==0 and syn['engineering_screen']['all_groups_passed'])
        if not passed:
            raise ValueError('step-control synthetic recovery gate must pass before measured access')
    validate_step_parent(cfg,root,args.phase)
    if not resolved.exists():resolved.write_text(yaml.safe_dump(cfg,sort_keys=False))
    manifest=dict(old,experiment_id=cfg['experiment_id'],execution='running',phase=args.phase+'_training',pilot=args.pilot,
        source_root=str(CODE_ROOT),project_root=str(root),controller_pid=os.getpid(),supervisor=os.environ.get('SSM_SYSTEMD_UNIT'),
        workers=args.workers,max_in_flight=cfg['resources']['max_in_flight'],step_control=cfg['step_control'],
        progress=dict(phase=args.phase,terminal=0,expected=(1 if args.pilot else cfg[args.phase+'_starts'])*(2 if cfg.get('paired_baseline') else 1)),
        started_at=old.get('started_at',datetime.now(timezone.utc).isoformat()))
    write_json(path,manifest);specs=step_control_specs(cfg,args.phase)
    if args.pilot:specs=[s for s in specs if s['group']=='synthetic_g1_mixed_r0']
    indices=[1] if args.pilot else [0,1,2]
    arms=['armijo',cfg['step_control']] if cfg.get('paired_baseline') else [cfg['step_control']]
    work=[]
    for arm in arms:
        arm_out=out/'arms'/arm if cfg.get('paired_baseline') else out
        for spec in specs:prepare_step_control_group(cfg,arm_out,root,spec,indices)
        base=deepcopy(cfg['base_contract']);base['training_step_control']=arm
        work.extend((base,str(arm_out),s,'conditional_trained_gain',i,g) for s in specs
            for i,g in enumerate(base['conditional_gain_training']['relative_start_values']) if i in indices)
    if cfg.get('paired_baseline'):np.random.default_rng(cfg['task_order_seed']).shuffle(work)
    write_json(out/f'{args.phase}_task_order.json',[dict(arm=w[0]['training_step_control'],group=w[2]['group'],start=w[4]) for w in work])
    terminal=0
    for payload,result,error in bounded_nonlinear_work(train_gain_prior_start,work,min(args.workers,len(work)),cfg['resources']['max_in_flight']):
        if error:
            base,arm_out,s,method,i,g=payload;result=dict(spec=s,start_index=i,start_relative_gain=g,method=method,status='failed_worker',converged=False,error=error,expected_trials=18,completed_trials=0,step_control=base['training_step_control'])
            write_json(gain_training_directory(arm_out,s,method)/f'start_{i}'/'result.json',result)
        terminal+=1
        progress=dict(event='training_start_terminal',phase=args.phase,group=payload[2]['group'],start=payload[4],status=result['status'],
            arm=payload[0]['training_step_control'],terminal=terminal,expected=len(work))
        print(json.dumps(progress),flush=True)
        manifest['progress']=progress;write_json(path,manifest)
    for arm in arms:
        arm_out=out/'arms'/arm if cfg.get('paired_baseline') else out
        for spec in specs:select_gain_prior_training(arm_out,spec,'conditional_trained_gain',indices)
    summary=(summarize_step_panel if cfg.get('paired_baseline') else summarize_step_control)(cfg,out,args.phase,pilot=args.pilot)
    passed=(summary['numerical_screen_passed'] if cfg.get('paired_baseline') else
        args.phase=='measured' or summary['regressions']==0 and summary['engineering_screen']['all_groups_passed'])
    manifest.update(execution=('completed' if args.phase=='measured' or args.pilot else 'synthetic_terminal') if passed or args.phase=='measured' else 'stopped_synthetic_check',
        numerical_screen_passed=bool(passed),
        **{args.phase+'_terminal':True},finished_at=datetime.now(timezone.utc).isoformat(),
        summary=f"{args.phase}_{'paired' if cfg.get('paired_baseline') else 'training'}_summary.json")
    write_json(path,manifest)


def read_waveform_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    filename={'shared_driver_waveform_diagnostic_v1':'shared_driver_waveform_diagnostic_v1.yaml',
              'shared_driver_waveform_volume_fraction_v1':'shared_driver_waveform_volume_fraction_v1.yaml'}.get(cfg.get('schema'))
    if filename is None:raise ValueError('unknown waveform schema')
    expected=CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer'/filename
    if cfg!=yaml.safe_load(expected.read_text()):
        raise ValueError('waveform diagnostic must use the complete versioned contract')
    return cfg


def waveform_parameters(base, name=None, value=None):
    if name is None:return base
    if name=='beta_factor':return replace(base,fixed=replace(base.fixed,neurovascular_gain=base.fixed.neurovascular_gain*value))
    if name in ('kappa','tau'):return replace(base,free=replace(base.free,**{name:value}))
    if name=='venous_viscoelastic_s':return base
    return replace(base,fixed=replace(base.fixed,**{name:value}))


def waveform_parent(cfg,root,group):
    """Exact retained public read boundary, metadata before array access."""
    parent=Path(root)/cfg['parent_run']
    manifest=json.loads((parent/'manifest.json').read_text())
    if manifest.get('execution')!='completed' or manifest.get('experiment_id')!='SSM-SHARED-DRIVER-CONDITIONAL-OPTICAL-GAIN-v1':
        raise ValueError('waveform parent is not the completed conditional experiment')
    allowed=[f'{p}__{s}_o3' for p in cfg['optical_pipelines'] for s in cfg['subjects']]
    if group not in allowed:raise ValueError('waveform group outside declared scope')
    info=json.loads((parent/'prepared'/f'{group}.json').read_text())
    m=info['metadata'];validation=[3,7,11,15,19,23];train=[i for i in range(24) if i not in validation]
    if (info.get('status')!='completed' or m['validation']!=validation or m['train']!=train
        or info['spec']['group']!=group or m['outer']!=3 or len(m['trials'])!=24
        or m['fixed_roi']['pair_name']!='AF7Fp1'):
        raise ValueError('parent identity, ROI or split mismatch')
    base_cfg=read_conditional_optical_gain_config(parent/'resolved_config.yaml')
    selection=json.loads((parent/'training'/f'{group}__conditional_trained_gain'/'selection.json').read_text())
    if selection['status']!='completed':raise ValueError('retained shared parameter unavailable')
    with np.load(parent/'prepared'/f'{group}.npz',allow_pickle=False) as z:
        target=z['target'];sd=z['normalizer']
    if (target.shape!=(24,120,3) or sd.shape!=(3,) or not np.isfinite(target).all()
        or not np.isfinite(sd).all() or np.any(sd<=0) or not np.allclose(sd,m['normalization_sd'],rtol=1e-12,atol=1e-14)):
        raise ValueError('parent tensor or training SD mismatch')
    p=parameters(base_cfg,2.)
    p=replace(p,fixed=replace(p.fixed,neurovascular_gain=selection['parameter_value']))
    operator=conditional_prediction_operator(view_operators(base_cfg,'full')[2],info['conditional_mapping'],'conditional_trained_gain')
    return parent,info,target,sd,p,operator


def waveform_linear_group(payload):
    from src.inference.shared_driver_reconstruction import fit_waveform_subspace,waveform_component_basis
    cfg,out,root,group=payload;started=time.monotonic()
    dest=Path(out)/'linear'/group
    if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
    parent,info,y,sd,p,operator=waveform_parent(cfg,root,group)
    train=info['metadata']['train'];valid=info['metadata']['validation'];rows=[];profiles=[];saved={}
    n,dt=120,.25
    for mode in ('joint','Hb_only'):
        mask=np.ones((n,3),bool)
        if mode=='Hb_only':mask[:,0]=False
        def fit_at(name=None,value=None,extra=None,rcond=1e-10):
            pp=waveform_parameters(p,name,value)
            design=build_shared_driver_design(pp,n,dt,processed_mean_operator=operator,
                venous_viscoelastic_s=value if name=='venous_viscoelastic_s' else 0.)
            basis=None if extra is None else waveform_component_basis(pp,n,dt,operator,extra,modes=cfg['slow_modes'])
            return fit_waveform_subspace(y,design,sd,visible=mask,extra_design=basis,rcond=rcond)
        methods=[('fixed',None,fit_at())]
        for name,grid in cfg['parameter_grids'].items():
            fits=[fit_at(name,value) for value in grid]
            losses=[float(np.mean(f['visible_sse'][train])) for f in fits]
            selected=int(np.argmin(losses))
            methods.append((name,float(grid[selected]),fits[selected]))
            for value,loss,f in zip(grid,losses,fits):
                profiles.append(dict(group=group,mode=mode,parameter=name,value=value,train_sse=loss,
                    validation_nrmse=np.sqrt(np.mean(f['nrmse'][valid]**2,axis=0)),
                    focus_nrmse=f['nrmse'][3],rank=f['rank']))
        for mechanism in ('volume','exchange'):
            methods.append(('slow_'+mechanism,None,fit_at(extra=mechanism)))
        for rcond in cfg['svd_rconds'][1:]:
            methods.append((f'fixed_rcond{rcond:g}',None,fit_at(rcond=rcond)))
        for method,value,f in methods:
            key=mode+'__'+method;saved[key]=f['prediction'][valid]
            for j in valid:
                rows.append(dict(group=group,mode=mode,method=method,parameter_value=value,trial=j,
                    rank=f['rank'],max_fractional_excursion=f['max_fractional_excursion'][j],
                    **{f'nrmse_{c}':float(f['nrmse'][j,k]) for k,c in enumerate(MODALITIES)}))
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'predictions.npz',target=y[valid],normalizer=sd,trial_indices=valid,**saved)
    result=dict(status='completed',group=group,rows=rows,profiles=profiles,
        seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_json(dest/'result.json',result);return result


def waveform_nonlinear_case(payload):
    from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_driver,nonlinear_driver_forward
    cfg,out,root,group,arm=payload;started=time.monotonic();dest=Path(out)/'nonlinear'/f'{group}__{arm}'
    if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
    parent,info,targets,sd,p,operator=waveform_parent(cfg,root,group)
    cell=parent/'cells'/f'{group}__conditional_trained_gain__full'
    with np.load(cell/'trajectories.npz',allow_pickle=False) as z:
        idx=int(np.flatnonzero(z['trial_indices']==3)[0]);state=z['states'][idx];old=z['prediction'][idx]
        np.testing.assert_array_equal(z['target'][idx],targets[3])
    mask=np.ones((120,3),bool)
    if arm=='Hb_only_no_penalties':mask[:,0]=False
    options=dict(penalty=.01,initial_penalty=100.,flow_prior_weight=1.)
    if arm in ('no_curvature','no_penalties','Hb_only_no_penalties'):options['penalty']=0.
    if arm in ('no_initial_prior','no_penalties','Hb_only_no_penalties'):options['initial_penalty']=0.
    if arm in ('no_flow_prior','no_penalties','Hb_only_no_penalties'):options['flow_prior_weight']=0.
    starts=[dict(driver=state[:,0],initial_state=state[0,1:]),dict(driver=np.zeros(120),initial_state=np.r_[0.,np.ones(4)])]
    result=fit_nonlinear_shared_driver(targets[3],p,.25,mean_operator=operator,sd=sd,visible=mask,
        starts=starts,max_evaluations=cfg['nonlinear_max_evaluations'],gradient_tolerance=cfg['nonlinear_gradient_tolerance'],
        substeps=4,flow_prior_log_sd=np.log(2.),**options)
    if 'prediction' in result:
        pred=result['prediction']
        result.update(optimization_status=result['status'],optimization_converged=result['converged'])
        try:
            fine=nonlinear_driver_forward(result['driver'],result['initial_state'],p,.25,substeps=8,derivative=False)
            fine_pred=(operator@fine['canonical_prediction'].ravel()).reshape(120,3)
            gap=float(np.max(abs(fine_pred-pred)/sd))
            result['integration_max_difference_training_sd']=gap
            if gap>cfg['maximum_integration_difference_training_sd']:result.update(status='failed_integration_check',converged=False)
        except (FloatingPointError,ValueError,OverflowError) as exc:
            # Retain the coarse trajectory and optimizer failure even if the
            # independent fine-grid check leaves the physical domain.
            result.update(status='failed_integration_check',converged=False,
                integration_max_difference_training_sd=None,integration_error=repr(exc))
        result['nrmse']=np.sqrt(np.mean(((pred-targets[3])/sd)**2,axis=0))
        result['retained_nrmse']=np.sqrt(np.mean(((old-targets[3])/sd)**2,axis=0))
        dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'trajectory.npz',target=targets[3],normalizer=sd,prediction=pred,states=result['states'],retained=old)
    result={k:v for k,v in result.items() if k not in ('prediction','canonical_prediction','driver','initial_state','states')}
    result.update(group=group,arm=arm,trial=3,seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_json(dest/'result.json',result);return result


def waveform_synthetic(cfg):
    from src.inference.shared_driver_reconstruction import fit_waveform_subspace,waveform_component_basis,nonlinear_driver_forward
    from src.inference.observation_baselines import conditional_optical_hb_mapping
    old=read_conditional_optical_gain_config(CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_conditional_optical_gain_v1.yaml')
    p=parameters(old,2.);n=120;dt=.25;rng=np.random.default_rng(cfg['synthetic']['seed'])
    mapping=conditional_optical_hb_mapping(40.);operator=conditional_prediction_operator(view_operators(old,'full')[2],mapping,'conditional_fixed_gain')
    p=replace(p,fixed=replace(p.fixed,neurovascular_gain=mapping['beta_reference']))
    t=np.arange(n)*dt;driver=.04*(np.sin(.45*t)+.4*np.sin(1.1*t))
    initial=np.array([0.,1.,1.,1.,1.]);rows=[];checks={}
    for truth in ('matched','tau4','kappa_low','viscoelastic4','volume','exchange'):
        pp=waveform_parameters(p,'tau',4.) if truth=='tau4' else waveform_parameters(p,'kappa',.16) if truth=='kappa_low' else p
        f=nonlinear_driver_forward(driver,initial,pp,dt,substeps=8,derivative=False)
        y=(operator@f['canonical_prediction'].ravel()).reshape(n,3)
        if truth=='viscoelastic4':
            generated=build_shared_driver_design(p,n,dt,processed_mean_operator=operator,venous_viscoelastic_s=4.)
            y=(generated.observation_design@np.r_[driver,np.zeros(5)]).reshape(n,3)+generated.offset
        if truth in ('volume','exchange'):
            basis=waveform_component_basis(p,n,dt,operator,truth,modes=cfg['slow_modes'])
            coef=cfg['synthetic']['injection_amplitude']*rng.normal(size=cfg['slow_modes'])
            y=y+(basis@coef).reshape(n,3)
        sd=np.maximum(np.std(y,axis=0),1e-6);design=build_shared_driver_design(p,n,dt,processed_mean_operator=operator)
        for method in ('fixed','volume','exchange','tau4','kappa_low','viscoelastic4'):
            pp=waveform_parameters(p,'tau',4.) if method=='tau4' else waveform_parameters(p,'kappa',.16) if method=='kappa_low' else p
            d=build_shared_driver_design(pp,n,dt,processed_mean_operator=operator,venous_viscoelastic_s=4. if method=='viscoelastic4' else 0.)
            extra=waveform_component_basis(p,n,dt,operator,method,modes=cfg['slow_modes']) if method in ('volume','exchange') else None
            fit=fit_waveform_subspace(y[None],d,sd,extra_design=extra)
            err=float(np.mean(fit['nrmse']**2));rows.append(dict(truth=truth,method=method,nrmse=fit['nrmse'][0],mean_nmse=err))
            if method==truth or truth=='matched' and method=='fixed':checks[truth]=err<.001
    return dict(status='passed' if all(checks.values()) else 'failed',checks=checks,rows=rows,
        interpretation='known_mechanism_representability_controls_not_unique_attribution_or_independent_replicates')


def waveform_volume_fit(cfg,y,sd,p,operator,train,valid):
    from src.inference.shared_driver_reconstruction import fit_waveform_subspace,waveform_component_basis
    d=build_shared_driver_design(p,120,.25,processed_mean_operator=operator)
    profiles=[];fits=[]
    for fraction in cfg['volume_deoxy_fraction_grid']:
        basis=waveform_component_basis(p,120,.25,operator,'volume',modes=6,volume_deoxy_fraction=fraction)
        f=fit_waveform_subspace(y,d,sd,extra_design=basis);fits.append(f)
        profiles.append(dict(fraction=fraction,training_sse=float(np.mean(f['visible_sse'][train])),
            validation_nrmse=np.sqrt(np.mean(f['nrmse'][valid]**2,axis=0)),focus_nrmse=f['nrmse'][3]))
    selected=int(np.argmin([v['training_sse'] for v in profiles]));f= fits[selected]
    return dict(selected_fraction=cfg['volume_deoxy_fraction_grid'][selected],profiles=profiles,
                rows=[dict(trial=i,nrmse=f['nrmse'][i],max_fractional_excursion=f['max_fractional_excursion'][i]) for i in valid]), f['prediction'][valid]


def waveform_volume_group(payload):
    cfg,base,out,root,group=payload;start=time.monotonic()
    _,info,y,sd,p,operator=waveform_parent(base,root,group)
    result,pred=waveform_volume_fit(cfg,y,sd,p,operator,info['metadata']['train'],info['metadata']['validation'])
    result.update(group=group,status='completed',seconds=time.monotonic()-start)
    dest=Path(out)/group;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'predictions.npz',prediction=pred,target=y[info['metadata']['validation']],normalizer=sd)
    write_json(dest/'result.json',result);return result


def waveform_volume_main(args,cfg):
    from src.inference.shared_driver_reconstruction import nonlinear_driver_forward,waveform_component_basis
    from src.inference.observation_baselines import conditional_optical_hb_mapping
    base=read_waveform_config(CODE_ROOT/cfg['base_config']);out=args.run_dir.resolve();root=args.project_root.resolve()
    if out.parent!=(root/base['output_root']).resolve():raise ValueError('volume output outside owning namespace')
    if args.phase not in ('synthetic','measured') or args.pilot or not 1<=args.workers<=6:raise ValueError('separate phases and at most six group workers required')
    out.mkdir(parents=True,exist_ok=True);lock=(out/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resolved=out/'resolved_config.yaml'
    if resolved.exists() and yaml.safe_load(resolved.read_text())!=cfg:raise ValueError('volume resume config mismatch')
    resolved.write_text(yaml.safe_dump(cfg,sort_keys=False));path=out/'manifest.json';manifest=json.loads(path.read_text()) if path.exists() else {}
    if args.phase=='measured' and manifest.get('synthetic_status')!='passed':raise ValueError('synthetic recovery must precede measured access')
    manifest.update(experiment_id=cfg['experiment_id'],execution='running',phase=args.phase,project_root=str(root),controller_pid=os.getpid(),workers=args.workers)
    write_json(path,manifest)
    if args.phase=='synthetic':
        old=read_conditional_optical_gain_config(CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_conditional_optical_gain_v1.yaml')
        mapping=conditional_optical_hb_mapping(40.);p=parameters(old,2.);p=replace(p,fixed=replace(p.fixed,neurovascular_gain=mapping['beta_reference']))
        operator=conditional_prediction_operator(view_operators(old,'full')[2],mapping,'conditional_trained_gain')
        rng=np.random.default_rng(cfg['synthetic_seed']);t=np.arange(120)*.25;results=[]
        for fraction in cfg['synthetic_true_fractions']:
            basis=waveform_component_basis(p,120,.25,operator,'volume',volume_deoxy_fraction=fraction)
            ys=[]
            for _ in range(24):
                driver=.04*np.sin(.45*t+rng.uniform(-3,3))+.01*np.sin(1.1*t)
                f=nonlinear_driver_forward(driver,np.r_[0.,np.ones(4)],p,.25,substeps=8,derivative=False)
                y=(operator@f['canonical_prediction'].ravel()+basis@(rng.normal(size=6)*.004)).reshape(120,3);ys.append(y)
            ys=np.array(ys);sd=np.std(ys[:18],axis=(0,1));r,_=waveform_volume_fit(cfg,ys,sd,p,operator,list(range(18)),list(range(18,24)))
            results.append(dict(true_fraction=fraction,**r))
        passed=all(r['selected_fraction']==r['true_fraction'] for r in results)
        write_json(out/'synthetic_summary.json',dict(passed=passed,results=results))
        manifest.update(execution='synthetic_terminal',synthetic_status='passed' if passed else 'failed');write_json(path,manifest);return
    parent_manifest=json.loads((root/cfg['diagnostic_parent']/'manifest.json').read_text())
    if parent_manifest.get('execution')!='completed':raise ValueError('volume followup requires completed diagnostic parent')
    payloads=[(cfg,base,str(out),str(root),f'{p}__{s}_o3') for p in base['optical_pipelines'] for s in base['subjects']]
    results=[]
    for payload,r,error in bounded_nonlinear_work(waveform_volume_group,payloads,args.workers,args.workers):
        if error:r=dict(group=payload[-1],status='failed_worker',error=error)
        results.append(r);print(json.dumps({k:r.get(k) for k in ['group','status','selected_fraction','seconds']}),flush=True)
    write_json(out/'summary.json',dict(expected=6,terminal=len(results),results=results,scientific_verdict='exploratory_loading_screen_only'))
    manifest.update(execution='completed',summary='summary.json');write_json(path,manifest)


def waveform_diagnostic_main(args):
    cfg=read_waveform_config(args.config)
    if args.check_only:
        print(json.dumps(dict(status='passed',groups=6,focus_nonlinear_cells=12 if cfg['schema']=='shared_driver_waveform_diagnostic_v1' else 0,measured_arrays_read=0)));return
    if cfg['schema']=='shared_driver_waveform_volume_fraction_v1':return waveform_volume_main(args,cfg)
    if args.phase not in ('synthetic','measured') or args.pilot:raise ValueError('waveform diagnostic requires separate phases and no pilot')
    root=args.project_root.resolve();out=args.run_dir.resolve()
    if out.parent!=(root/cfg['output_root']).resolve():raise ValueError('waveform output outside owning namespace')
    if not 1<=args.workers<=cfg['resources']['max_workers']:raise ValueError('worker limit exceeded')
    out.mkdir(parents=True,exist_ok=True);resolved=out/'resolved_config.yaml'
    if resolved.exists() and yaml.safe_load(resolved.read_text())!=cfg:raise ValueError('resume contract mismatch')
    resolved.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    path=out/'manifest.json';manifest=json.loads(path.read_text()) if path.exists() else {}
    if args.phase=='measured' and manifest.get('synthetic_status')!='passed':raise ValueError('synthetic checks must pass before measured arrays')
    lock=(out/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest.update(experiment_id=cfg['experiment_id'],execution='running',phase=args.phase,
        project_root=str(root),controller_pid=os.getpid(),workers=args.workers,
        started_at=manifest.get('started_at',datetime.now(timezone.utc).isoformat()),
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        supervisor=os.environ.get('WAVEFORM_SUPERVISOR','unspecified'))
    write_json(path,manifest)
    if args.phase=='synthetic':
        result=waveform_synthetic(cfg);write_json(out/'synthetic_summary.json',result)
        manifest.update(execution='synthetic_terminal',synthetic_status=result['status']);write_json(path,manifest)
        print(json.dumps(serial(result)));return
    groups=[f'{p}__{s}_o3' for p in cfg['optical_pipelines'] for s in cfg['subjects']]
    for worker,payloads,label in [
        (waveform_linear_group,[(cfg,str(out),str(root),g) for g in groups],'linear'),
        (waveform_nonlinear_case,[(cfg,str(out),str(root),f'{p}__subject_09_o3',a) for p in cfg['optical_pipelines'] for a in cfg['nonlinear_arms']],'nonlinear')]:
        results=[]
        for payload,result,error in bounded_nonlinear_work(worker,payloads,args.workers,args.workers):
            if error:result=dict(status='failed_worker',group=payload[3],arm=payload[4] if len(payload)>4 else None,error=error)
            results.append(result);print(json.dumps({k:result.get(k) for k in ('group','arm','status','seconds','nrmse')} ,default=serial),flush=True)
            write_json(out/f'{label}_summary.json',dict(expected=len(payloads),terminal=len(results),results=results))
        manifest[label+'_terminal']=True;write_json(path,manifest)
    linear=json.loads((out/'linear_summary.json').read_text());nonlinear=json.loads((out/'nonlinear_summary.json').read_text())
    rows=[r for result in linear['results'] for r in result.get('rows',[])]
    pd.DataFrame(rows).to_csv(out/'linear_metrics.csv',index=False)
    write_json(out/'summary.json',dict(linear_groups=len(linear['results']),linear_success=sum(r['status']=='completed' for r in linear['results']),
        nonlinear_cells=len(nonlinear['results']),nonlinear_success=sum(r['status']=='completed' and r.get('converged',False) for r in nonlinear['results']),
        scientific_verdict='exploratory_waveform_mechanism_diagnostic_only',parent=cfg['parent_run']))
    manifest.update(execution='completed',finished_at=datetime.now(timezone.utc).isoformat(),summary='summary.json');write_json(path,manifest)


def semantics_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get('schema') != 'shared_driver_physiology_semantics_v1':
        raise ValueError('wrong physiology semantics contract')
    if (cfg['tensor']['steps'] != 120 or cfg['tensor']['dt_s'] != .25 or
            cfg['tensor']['state_order'] != ['r', 's', 'f', 'v', 'p', 'q'] or
            cfg['data']['protected_data'] != 'forbidden' or
            cfg['data']['cache_root'] != 'data/cache/physiology_semantic_clean_v5' or
            set(cfg['data']['datasets']) != {'eeg_fnirs_single_trial','simultaneous_eeg_nirs','visual_cognitive_motivation','refed'} or
            cfg['data']['motion_method'] != 'none' or cfg['splits']['labels_as_model_inputs'] or
            cfg['report']['format'] != 'pptx'):
        raise ValueError('physiology semantics tensor, data, model or report contract violated')
    semantics_parameters(cfg).validate()
    return cfg


def semantics_parameters(cfg, **values):
    fixed = {k: v for k, v in cfg['fixed'].items() if k not in ('tau', 'kappa')}
    fixed.update({k: v for k, v in values.items() if k not in ('tau', 'kappa')})
    return BalloonParameters(fixed=BalloonFixedParameters(**fixed),
        free=BalloonFreeParameters(tau=values.get('tau', cfg['fixed']['tau']),
                                  kappa=values.get('kappa', cfg['fixed']['kappa'])))


def semantics_solver_options(cfg):
    s = cfg['solver']
    return {k: s[k] for k in ('numerical_backend', 'substeps', 'penalty',
        'initial_penalty', 'flow_prior_weight', 'flow_prior_log_sd', 'gradient_tolerance')}


def semantics_anchors(record, geometry):
    """Select actual channels from metadata only; never use fitted errors."""
    meta = record.manifest['measurement']
    eeg = geometry.for_channels(record=record, modality='eeg', channel_names=meta['eeg_channel_names'])
    hb = geometry.for_channels(record=record, modality='fnirs', channel_names=meta['fnirs_channel_names'][::2])
    if record.dataset_id == 'refed':
        return [], 'REFED EEG template is not registered to native Hb geometry'
    ex = np.array([[r.get(k) if r.get(k) is not None else np.nan for k in ('x','y','z')] for r in eeg])
    hx = np.array([[r.get(k) if r.get(k) is not None else np.nan for k in ('x','y','z')] for r in hb])
    if not np.isfinite(ex).all() or not np.isfinite(hx).all():
        return [], 'incomplete registered geometry'
    if any(r['coordinate_units'] != 'normalized_head_unit' for r in eeg+hb):
        return [], 'incompatible geometry units'
    targets = dict(prefrontal=[-.45,.89,.1], motor=[-.65,0,.72], posterior=[-.15,-.95,.1])
    if record.dataset_id == 'visual_cognitive_motivation':
        targets = {k: [v[1],-v[0],v[2]] for k,v in targets.items()}
    unit = hx/np.linalg.norm(hx, axis=1)[:,None]
    anchors, used = [], set()
    for region, target in targets.items():
        target = np.asarray(target); target = target/np.linalg.norm(target)
        index = int(np.argmax(unit@target))
        if float(unit[index]@target) < np.cos(np.deg2rad(50)) or index in used:
            continue
        used.add(index)
        neighbors = np.argsort(np.linalg.norm(ex-hx[index], axis=1))[:6]
        anchors.append(dict(region=region, hb_pair=index, hb_channel=hb[index]['base_channel_name'],
            eeg_indices=neighbors.tolist(), eeg_channels=[eeg[i]['channel_name'] for i in neighbors],
            hb_geometry=hb[index], eeg_geometry=[eeg[i] for i in neighbors],
            spatial_claim='coarse_template' if record.dataset_id == 'visual_cognitive_motivation' else 'registered_dataset_geometry_not_individual_source_localization'))
    return anchors, None if anchors else 'no supported regional anchor'


def semantics_window_plan(record, events):
    from src.data.event_alignment import window_within_alignment_support
    shape = record.manifest['array_shapes']
    duration_e, duration_h = shape['eeg'][0]/200., shape['fnirs'][0]/10.
    windows, rejected = [], []
    if record.dataset_id == 'refed':
        if not events:
            return [], [dict(reason='missing_event')]
        e = events[0]
        for i, start in enumerate(np.arange(0, min(duration_e, duration_h)-30+1e-6, 30)):
            windows.append(dict(window=i, eeg_start_s=float(start), hb_start_s=float(start),
                event_id=e['event_index'], task='emotion_video', condition=e['label'],
                independent_block=record.base_record_id))
        return windows, rejected
    last_end = -float('inf')
    for event in sorted(events, key=lambda x: x.get('eeg_time_ms') or 0):
        if event.get('eeg_time_ms') is None or event.get('fnirs_time_ms') is None:
            continue
        condition = str(event['metadata'].get('condition_label', event.get('label', 'unknown')))
        if condition == 'unknown':
            rejected.append(dict(event_id=event['event_index'], reason='unknown_condition')); continue
        es, hs = event['eeg_time_ms']/1000.-5., event['fnirs_time_ms']/1000.-5.
        if (min(es, hs) < 0 or es+30 > duration_e or hs+30 > duration_h or
                not window_within_alignment_support(event, -5., 30.)):
            rejected.append(dict(event_id=event['event_index'], reason='clock_or_alignment_support')); continue
        if record.dataset_id != 'eeg_fnirs_single_trial' and es < last_end:
            continue  # Dense events: continuous non-overlapping response windows.
        support = event['metadata'].get('alignment_support_ms', {})
        block = json.dumps(support, sort_keys=True)
        windows.append(dict(window=len(windows), eeg_start_s=es, hb_start_s=hs,
            event_id=event['event_index'], task=event['metadata'].get('task'), condition=condition,
            independent_block=block, native_session=event['metadata'].get('session_idx'),
            block_index=event['metadata'].get('block_index')))
        last_end = es+30
    return windows, rejected


def semantics_inventory(cfg, project_root, out):
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    from src.data.unified_physiology import ChannelGeometryIndex, DEFAULT_ADMISSIBLE_ALIGNMENT_CASES
    index = CleanPhysiologyCacheIndex(Path(project_root)/cfg['data']['cache_root'])
    geometry = ChannelGeometryIndex(index.cache_root)
    records, excluded = [], []
    for record in index.records:
        if record.dataset_id not in cfg['data']['datasets']:
            continue
        reports = index.reports_by_join_key.get(record.join_key, [])
        reason = None
        if record.dataset_id == 'visual_cognitive_motivation' and record.canonical_subject_id == 'S06' and 'Part1' in record.base_record_id:
            reason = 'retained_S06_Part1_alignment_exclusion'
        elif not reports or any(r.get('alignment_case') not in DEFAULT_ADMISSIBLE_ALIGNMENT_CASES
                                or r.get('label_sequence_match') is False for r in reports):
            reason = 'unverified_alignment'
        if reason == 'retained_S06_Part1_alignment_exclusion':
            excluded.append(dict(join_key=record.join_key, reason=reason)); continue
        anchors, spatial_reason = semantics_anchors(record, geometry)
        windows, rejected = semantics_window_plan(record, index.events_by_join_key.get(record.join_key, []))
        if reason:
            rejected.append(dict(reason=reason, planned_windows=len(windows)))
            windows = []
        key = record.join_key.replace('|','__').replace('/','_')
        repeat_group = record.base_record_id
        if record.dataset_id == 'visual_cognitive_motivation':
            repeat_group = record.base_record_id.split('_Probe')[0]
        records.append(dict(key=key, join_key=record.join_key, dataset=record.dataset_id,
            subject=record.canonical_subject_id, record=record.base_record_id, repeat_group=repeat_group,
            anchors=anchors, spatial_unavailable_reason=spatial_reason, windows=windows,
            rejected_windows=rejected, array_shapes=record.manifest['array_shapes'],
            source_files=record.manifest['source_files'],
            eeg_unit=record.manifest['measurement']['eeg_preprocessing_state']['canonical_unit'],
            hb_unit=record.manifest['measurement']['fnirs_preprocessing_state']['canonical_unit']))
    result = dict(schema='physiology_semantics_inventory_v1', records=records, excluded=excluded,
        input_cache=str(index.cache_root), measured_arrays_read=0,
        subject_counts={ds: len({r['subject'] for r in records if r['dataset']==ds}) for ds in cfg['data']['datasets']},
        record_counts={ds: sum(r['dataset']==ds for r in records) for ds in cfg['data']['datasets']})
    write_json(Path(out)/'inventory.json', result)
    return result


def semantics_native_qc(native):
    from scipy.signal import detrend
    values = native['values']; rate = native['sample_rate_hz']
    rows = []
    for i in range(values.shape[1]):
        pair = values[:,i]
        valid = np.isfinite(pair).all(axis=1)
        x = pair[valid]
        row = dict(pair=i, support=float(valid.mean()), samples=int(valid.sum()))
        if len(x) < 30 or min(np.std(x,axis=0)) <= 1e-14:
            row.update(status='flat_or_insufficient', rho=None); rows.append(row); continue
        delta = np.diff(x,axis=0); med = np.median(delta,axis=0)
        mad = np.maximum(1.4826*np.median(abs(delta-med),axis=0), 1e-14)
        step = max(1, round(rate))
        diff = x[step:]-x[:-step]
        row.update(status='supported', rho=float(np.corrcoef(x.T)[0,1]),
            detrended_rho=float(np.corrcoef(detrend(x,axis=0).T)[0,1]),
            difference_1s_rho=float(np.corrcoef(diff.T)[0,1]),
            exact_HbO_HbR_duplicate=bool(np.array_equal(x[:,0],x[:,1])),
            flat_step_fraction=np.mean(delta==0,axis=0),
            jump_fraction=np.mean(abs(delta-med)>20*mad,axis=0),
            extrema_repeat_fraction=[float(max(np.mean(x[:,j]==x[:,j].min()),np.mean(x[:,j]==x[:,j].max()))) for j in range(2)])
        rows.append(row)
    finite = np.isfinite(values).all(axis=(0,2)) & (np.std(values,axis=0).min(axis=1)>1e-14)
    spatial = None
    if finite.sum()>1:
        total = values[::max(1,round(rate)),finite].sum(axis=2)
        c = np.corrcoef(total.T)
        spatial = float(np.median(c[np.triu_indices(len(c),1)]))
    result = dict(channels=rows, spatial_total_Hb_median_rho=spatial,
        native_provenance=native['provenance'], time_strictly_increasing=bool(np.all(np.diff(native['time_s'])>0)),
        quality_flag_policy='descriptive_only_not_fit_dependent_or_automatic_exclusion')
    if native.get('dependent_total_hb') is not None:
        residual = native['dependent_total_hb']-values.sum(axis=2)
        result['dependent_HbT_closure'] = dict(max_absolute=float(np.nanmax(abs(residual))),
            rms=float(np.sqrt(np.nanmean(residual**2))), independent_measurement=False)
        result['absorbance_shape'] = list(native['absorbance'].shape)
    return result


def semantics_qc_detail_worker(payload):
    import hashlib
    from scipy.signal import coherence,detrend
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    from src.data.unified_physiology import load_native_fnirs_record
    cfg,root,project_root,spec=payload;path=Path(root)/'qc_detail'/(spec['key']+'.json')
    if path.exists():return json.loads(path.read_text())
    started=time.monotonic()
    index=CleanPhysiologyCacheIndex(Path(project_root)/cfg['data']['cache_root'])
    record=next(r for r in index.records if r.join_key==spec['join_key'])
    native=load_native_fnirs_record(Path(project_root),record);x=native['values'];time_s=native['time_s']
    flat=x.reshape(len(x),-1);lookup={};duplicates=[]
    for j in range(flat.shape[1]):
        column=flat[:,j]
        if not np.isfinite(column).all() or np.std(column)<=1e-14:continue
        digest=hashlib.blake2b(np.ascontiguousarray(column).tobytes(),digest_size=16).hexdigest()
        if digest in lookup and np.array_equal(column,flat[:,lookup[digest]]):
            duplicates.append([native['channel_names'][lookup[digest]],native['channel_names'][j]])
        lookup[digest]=j
    rate=native['sample_rate_hz'];uniform=bool(np.allclose(np.diff(time_s),1/rate,rtol=.001,atol=1e-6))
    coherences=[]
    if uniform:
        for pair in x.transpose(1,0,2):
            if not np.isfinite(pair).all() or min(np.std(pair,axis=0))<=1e-14:
                coherences.append(None);continue
            nperseg=min(round(128*rate),len(pair)//2)
            f,c=coherence(pair[:,0],pair[:,1],fs=rate,nperseg=nperseg)
            selected=(f>=.01)&(f<=.2)
            coherences.append(float(np.mean(c[selected])) if selected.any() else None)
    total=x[::max(1,round(rate))].sum(axis=2)
    keep=np.isfinite(total).all(axis=0)&(np.std(total,axis=0)>1e-14)
    common={}
    for label,value in [('native',total[:,keep]),('detrended',detrend(total[:,keep],axis=0))]:
        if value.shape[1]>1:
            sd=np.std(value,axis=0);good=sd>1e-14
            c=np.corrcoef(value[:,good].T)
            common[label]=dict(channels=int(good.sum()),first_component_variance_fraction=float(np.linalg.eigvalsh(c)[-1]/len(c))) if good.sum()>1 else None
    outside=[w['window'] for w in spec['windows'] if w['hb_start_s']<time_s[0]-1e-9 or w['hb_start_s']+29.9>time_s[-1]+1e-9]
    result=dict(spec_key=spec['key'],dataset=spec['dataset'],subject=spec['subject'],record=spec['record'],
        status='completed',exact_nonflat_channel_duplicates=duplicates,coherence_001_02Hz=coherences,
        native_clock_uniform=uniform,native_time_support=[float(time_s[0]),float(time_s[-1])],
        windows_outside_native_support=outside,common_HbT_variance=common,
        full_record_fingerprint=hashlib.blake2b(np.ascontiguousarray(x).tobytes(),digest_size=16).hexdigest(),
        full_record_shape=list(x.shape),seconds=time.monotonic()-started,
        interpretation='descriptive data checks; coherence and common variance do not identify a physiological source')
    write_json(path,result);return result


def semantics_prepare_worker(payload):
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    from src.data.unified_physiology import load_native_eeg_record, load_native_fnirs_record
    from src.inference.observation_baselines import eeg_band_power
    cfg, root, project_root, spec = payload
    path = Path(root)/'prepared'/spec['key']/'record.json'
    if path.exists(): return json.loads(path.read_text())
    started = time.monotonic()
    index = CleanPhysiologyCacheIndex(Path(project_root)/cfg['data']['cache_root'])
    record = next(r for r in index.records if r.join_key == spec['join_key'])
    native = load_native_fnirs_record(Path(project_root), record)
    qc = semantics_native_qc(native)
    prepared, failures = [], []
    if spec['anchors'] and spec['windows']:
        eeg = load_native_eeg_record(Path(project_root), record)
        lookup = {str(name).upper():i for i,name in enumerate(eeg.channel_names)}
        operator = native_feature_operators(120)
        for anchor in spec['anchors']:
            names = [n.upper() for n in anchor['eeg_channels']]
            if not all(n in lookup for n in names):
                failures.append(dict(region=anchor['region'], reason='native_EEG_channel_identity')); continue
            selected = [lookup[n] for n in names]
            features, hb, identities = [], [], []
            for window in spec['windows']:
                start = round(window['eeg_start_s']*eeg.sample_rate_hz)
                length = round(30*eeg.sample_rate_hz)
                signal = eeg.values[start:start+length,selected]
                times = window['hb_start_s']+np.arange(300)/10.
                pair = native['values'][:,anchor['hb_pair']]
                h = np.column_stack([np.interp(times,native['time_s'],pair[:,j]) for j in range(2)])
                if signal.shape[0]!=length or not np.isfinite(signal).all() or not np.isfinite(h).all():
                    failures.append(dict(region=anchor['region'], window=window['window'], reason='nonfinite_or_native_support')); continue
                power = eeg_band_power(signal, bands=cfg['tensor']['eeg_bands_hz'],
                                       sample_rate=eeg.sample_rate_hz,target_rate=4.)
                if power.shape != (120,6,5) or np.any(power<=0):
                    failures.append(dict(region=anchor['region'], window=window['window'], reason='invalid_power')); continue
                logpower = np.log(power).reshape(120,-1)
                features.append(operator['eeg']@logpower)
                hb.append(operator['fnirs']@h)
                identities.append(window)
            path.parent.mkdir(parents=True,exist_ok=True)
            if features:
                np.savez_compressed(path.parent/(anchor['region']+'.npz'),
                    eeg_features=np.array(features), hb=np.array(hb))
                prepared.append(dict(**anchor, windows=identities, array_file=anchor['region']+'.npz',
                    eeg_native_unit=eeg.native_unit, eeg_native_rate=eeg.sample_rate_hz,
                    native_eeg_source=str(eeg.source_path)))
    result = dict(spec=spec, qc=qc, prepared=prepared, failures=failures, status='completed',
        processing='native_window_local_bandpower_and_Hb_none_bandpass_resampling',
        reference='first_5_seconds_observed_reference_not_latent_rest_or_always_preevent',
        split_safety='no_temporal_operation_crosses_window_boundary; record grouping prevents overlapping folds',
        seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_json(path,result)
    return result


@lru_cache(maxsize=8)
def semantics_prepared_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        return arrays['eeg_features'].copy(), arrays['hb'].copy()


def semantics_coordinate(refs, out, key, cfg):
    """Fit one explicit training-only PCA and common Hb gain; keep DC reference."""
    sx=np.zeros(30);xx=np.zeros((30,30));sh=np.zeros(2);hh=np.zeros(2);count=0
    for path in sorted({r['array_path'] for r in refs}):
        selected=[r['array_index'] for r in refs if r['array_path']==path]
        e,h=semantics_prepared_arrays(path)
        x=e[selected].reshape(-1,30);v=h[selected].reshape(-1,2)
        sx+=x.sum(axis=0);xx+=x.T@x;sh+=v.sum(axis=0);hh+=(v*v).sum(axis=0);count+=len(x)
    if not count:raise ValueError('empty coordinate training partition')
    covariance=xx/count-np.outer(sx/count,sx/count)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance);pc=eigenvectors[:,-1]
    if pc[np.argmax(abs(pc))]<0:pc=-pc
    variance_h=np.maximum(hh/count-(sh/count)**2,0.)
    if eigenvalues[-1]<=1e-16 or min(variance_h)<=1e-24:raise ValueError('degenerate training coordinate')
    eeg_factor=cfg['coordinate']['eeg_training_sd_target']/np.sqrt(eigenvalues[-1])
    hb_factor=cfg['coordinate']['common_Hb_training_sd_target']/np.sqrt(np.mean(variance_h))
    coordinate=dict(key=key,pc=pc,eeg_factor=eeg_factor,hb_factor=hb_factor,
        sd=np.r_[np.sqrt(eigenvalues[-1])*eeg_factor,np.sqrt(variance_h)*hb_factor],
        training_subjects=sorted({r['subject'] for r in refs}),training_windows=len(refs),
        training_ids=[r['id'] for r in refs],eeg_channels=refs[0]['eeg_channels'],
        centering='first_5s_reference_only; covariance_center_used_for_PCA_not_subtracted_from_target',
        gain_interpretation='effective_in_frozen_dataset_site_fold_gauge_not_absolute_physiology')
    write_json(Path(out)/'coordinates'/(key+'.json'),coordinate)
    return coordinate


def semantics_target(ref, coordinate):
    e,h=semantics_prepared_arrays(ref['array_path'])
    return np.column_stack((e[ref['array_index']]@np.asarray(coordinate['pc'])*coordinate['eeg_factor'],
                            h[ref['array_index']]*coordinate['hb_factor']))


def semantics_balanced_select(refs, limit, seed):
    """Select records round-robin before windows; selection never reads signals."""
    rng=np.random.default_rng(seed);groups={}
    for r in refs:groups.setdefault((r['subject'],r['record']),[]).append(r)
    keys=sorted(groups);rng.shuffle(keys)
    for key in keys:rng.shuffle(groups[key])
    chosen=[]
    while len(chosen)<limit and any(groups.values()):
        for key in keys:
            if groups[key] and len(chosen)<limit:chosen.append(groups[key].pop())
    return chosen


def semantics_cohort(cfg,out):
    inventory=json.loads((out/'inventory.json').read_text())
    cohorts=[];unavailable=[]
    channel_sets={}
    for spec in inventory['records']:
        for anchor in spec['anchors']:
            channel_sets.setdefault((spec['dataset'],anchor['region']),set()).add(tuple(anchor['eeg_channels']))
    subject_folds={}
    for ds in cfg['data']['datasets']:
        subjects=sorted({r['subject'] for r in inventory['records'] if r['dataset']==ds})
        rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],21,cfg['data']['datasets'].index(ds)]))
        rng.shuffle(subjects)
        subject_folds[ds]={s:i%cfg['splits']['subject_folds'] for i,s in enumerate(subjects)}
    for spec in inventory['records']:
        path=out/'prepared'/spec['key']/'record.json'
        if not path.exists():unavailable.append(dict(key=spec['key'],reason='preparation_failed'));continue
        record=json.loads(path.read_text())
        for anchor in record['prepared']:
            variants=sorted(channel_sets[(spec['dataset'],anchor['region'])])
            site=anchor['region']+(f'__montage{variants.index(tuple(anchor["eeg_channels"]))}' if len(variants)>1 else '')
            windows=anchor['windows'];groups=list(dict.fromkeys(w['independent_block'] for w in windows))
            calibration=windows[:3]
            end=max((w['eeg_start_s']+30 for w in calibration),default=float('inf'))
            for i,w in enumerate(windows):
                repeat_unit=spec['repeat_group']
                if spec['dataset']=='eeg_fnirs_single_trial':
                    repeat=int(spec['record'].rsplit('_',1)[-1])//2%2
                elif spec['dataset']=='simultaneous_eeg_nirs':
                    repeat=groups.index(w['independent_block'])%2
                    repeat_unit+='::'+str(groups.index(w['independent_block']))
                else:
                    # Preserve both probes of the same EEG Part in the same fold.
                    import re
                    match=re.search(r'Part(\d+)',spec['record'],re.I)
                    repeat=(int(match.group(1))-1)%2 if match else 0
                channel_qc=record['qc']['channels'][anchor['hb_pair']]
                cohorts.append(dict(id=f'{spec["key"]}__{site}__w{w["window"]}',key=spec['key'],
                    dataset=spec['dataset'],subject=spec['subject'],record=spec['record'],site=site,
                    region=anchor['region'],eeg_channels=anchor['eeg_channels'],hb_channel=anchor['hb_channel'],
                    array_path=str(path.parent/anchor['array_file']),array_index=i,
                    subject_fold=subject_folds[spec['dataset']][spec['subject']],repeat_fold=repeat,
                    repeat_unit=repeat_unit,record_calibration=i<3,record_evaluation=w['eeg_start_s']>=end,
                    native_rho=channel_qc.get('rho'),quality_burden=float(np.mean(channel_qc.get('jump_fraction',[1,1])))+
                        float(np.mean(channel_qc.get('flat_step_fraction',[1,1])))+1-channel_qc['support'],**w))
    # Subjects with only one independent recording can only support disjoint-block repeatability.
    for ds,subject,site in sorted({(r['dataset'],r['subject'],r['site']) for r in cohorts}):
        group=[r for r in cohorts if (r['dataset'],r['subject'],r['site'])==(ds,subject,site)]
        if len({r['repeat_unit'] for r in group})==1:
            center=(min(r['eeg_start_s'] for r in group)+max(r['eeg_start_s']+30 for r in group))/2
            for r in group:
                r['repeat_fold']=0 if r['eeg_start_s']+30<=center else 1 if r['eeg_start_s']>=center else -1
                r['repeat_kind']='within_record_disjoint_blocks'
        else:
            for r in group:r['repeat_kind']='independent_record_or_native_session'
    write_json(out/'cohort.json',dict(refs=cohorts,unavailable=unavailable,subject_folds=subject_folds))
    coordinates={}
    for ds,site in sorted({(r['dataset'],r['site']) for r in cohorts}):
        group=[r for r in cohorts if (r['dataset'],r['site'])==(ds,site)]
        for fold in sorted({r['subject_fold'] for r in group}):
            key=f'{ds}__{site}__outer{fold}'
            train=[r for r in group if r['subject_fold']!=fold]
            coordinates[key]=semantics_coordinate(train,out,key,cfg)
    return cohorts,coordinates


def semantics_training_plan(cfg,out,refs):
    screen=json.loads((out/'synthetic_feature_summary.json').read_text())
    names=[name for name,row in screen['parameters'].items() if row['passed']]
    plan=[]
    for ds,site in sorted({(r['dataset'],r['site']) for r in refs}):
        group=[r for r in refs if (r['dataset'],r['site'])==(ds,site)]
        for fold in sorted({r['subject_fold'] for r in group}):
            key=f'{ds}__{site}__outer{fold}'
            train=semantics_balanced_select([r for r in group if r['subject_fold']!=fold],
                    cfg['splits']['training_windows'],cfg['seed']+31+fold)
            for name in names:
                plan.append(dict(key=key+'__H1__'+name,coordinate=key,hierarchy='H1_dataset',
                    parameter=name,train=train,subject=None,record=None,repeat=None))
        for subject in sorted({r['subject'] for r in group}):
            own=[r for r in group if r['subject']==subject];fold=own[0]['subject_fold']
            key=f'{ds}__{site}__outer{fold}'
            for repeat in (0,1):
                train=semantics_balanced_select([r for r in own if r['repeat_fold']==repeat],
                    cfg['splits']['training_windows'],cfg['seed']+41+repeat)
                if not train:continue
                for name in names:
                    for hierarchy in ('H2_subject','H2s_partial_pooling'):
                        plan.append(dict(key=key+f'__{subject}__rep{repeat}__{hierarchy}__'+name,
                            coordinate=key,hierarchy=hierarchy,parameter=name,train=train,
                            subject=subject,record=None,repeat=repeat,parent=key+'__H1__'+name))
            for record in sorted({r['record'] for r in own}):
                train=[r for r in own if r['record']==record and r['record_calibration']]
                if not train:continue
                for name in names:
                    plan.append(dict(key=key+f'__{subject}__{record}__H3__'+name,coordinate=key,
                        hierarchy='H3_record',parameter=name,train=train,subject=subject,
                        record=record,repeat=None))
    write_json(out/'training_plan.json',dict(jobs=plan,qualified_parameters=names,
        primary_baseline='H0 always evaluated; failed synthetic parameter arms not expanded',
        population_window_sampling='18 identity-selected windows with record round-robin; coordinate scaling uses all permitted training windows'))
    return plan


def semantics_training_worker(payload):
    from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_parameter
    cfg,root,spec=payload;out=Path(root);path=out/'training'/spec['key']/'result.json'
    if path.exists():return json.loads(path.read_text())
    started=time.monotonic();coordinate=json.loads((out/'coordinates'/(spec['coordinate']+'.json')).read_text())
    y=np.array([semantics_target(ref,coordinate) for ref in spec['train']]);name=spec['parameter']
    starts=[dict(parameter_value=v,driver=np.zeros(y.shape[:2]),
                 initial_state=np.tile(np.r_[0.,np.ones(4)],(len(y),1))) for v in cfg['parameters'][name]['starts']]
    prior={}
    if spec['hierarchy']=='H2s_partial_pooling':
        parent=json.loads((out/'training'/spec['parent']/'result.json').read_text())
        if not parent.get('converged'):
            result=dict(spec=spec,status='unavailable_population_fit',converged=False)
            write_json(path,result);return result
        # Prespecified engineering shrinkage range derives only from the parameter box.
        bounds=cfg['parameters'][name]['bounds']
        prior=dict(parameter_prior_mean=parent['parameter_value'],parameter_prior_log_sd=np.log(bounds[1]/bounds[0])/4.)
    result=fit_nonlinear_shared_parameter(y,semantics_parameters(cfg),.25,parameter_name=name,
        parameter_bounds=cfg['parameters'][name]['bounds'],sd=coordinate['sd'],starts=starts,
        mean_operator=semantics_model_operator(),max_evaluations=cfg['solver']['max_evaluations_per_trial']*len(y),
        max_iterations=cfg['solver']['max_iterations'],step_control=cfg['solver']['step_control'],
        **semantics_solver_options(cfg),**prior)
    compact=dict(semantics_compact_fit(result),spec=spec,coordinate=coordinate['key'],prior=prior,
        seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_json(path,compact)
    if 'driver' in result:
        np.savez_compressed(path.with_suffix('.npz'),driver=result['driver'],initial_state=result['initial_state'])
    return compact


def semantics_parameter_candidates(cfg,out,ref,qualified):
    coordinate=f'{ref["dataset"]}__{ref["site"]}__outer{ref["subject_fold"]}'
    candidates=[dict(method='H0_fixed',parameter=None,value=None,status='completed',converged=True)]
    for name in qualified:
        paths=[('H1_dataset',coordinate+'__H1__'+name)]
        if ref['repeat_fold'] in (0,1):
            paths.extend([
                ('H2_subject',coordinate+f'__{ref["subject"]}__rep{1-ref["repeat_fold"]}__H2_subject__'+name),
                ('H2s_partial_pooling',coordinate+f'__{ref["subject"]}__rep{1-ref["repeat_fold"]}__H2s_partial_pooling__'+name)])
        if ref['record_evaluation']:
            paths.append(('H3_record',coordinate+f'__{ref["subject"]}__{ref["record"]}__H3__'+name))
        for level,key in paths:
            path=Path(out)/'training'/key/'result.json'
            result=json.loads(path.read_text()) if path.exists() else dict(status='unavailable_independent_training',converged=False)
            candidates.append(dict(method=level+'__'+name,parameter=name,value=result.get('parameter_value'),
                status=result['status'],converged=result.get('converged',False),training_key=key,
                at_boundary=result.get('boundary_status') in ('LOWER','UPPER')))
    return candidates


def semantics_fit_metrics(y,fit,sd):
    prediction=fit['prediction'];residual=y-prediction
    rmse=np.sqrt(np.mean(residual**2,axis=0))/sd
    corr=[]
    for i in range(3):
        corr.append(float(np.corrcoef(y[:,i],prediction[:,i])[0,1])
                    if min(np.std(y[:,i]),np.std(prediction[:,i]))>1e-14 else None)
    amplitude=(np.ptp(prediction,axis=0)-np.ptp(y,axis=0))/sd
    return dict(nrmse=rmse,all_below_half=bool(np.all(rmse<.5) and fit['converged']),
        correlation=corr,amplitude_error=amplitude,
        peak_time_error_s=(np.argmax(prediction,axis=0)-np.argmax(y,axis=0))*.25,
        trough_time_error_s=(np.argmin(prediction,axis=0)-np.argmin(y,axis=0))*.25,
        residual_Hb_rho=float(np.corrcoef(residual[:,1:].T)[0,1]) if min(np.std(residual[:,1:],axis=0))>1e-14 else None)


def semantics_evaluation_worker(payload):
    from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_driver
    cfg,root,refs,qualified=payload[:4];out=Path(root)
    phase=payload[4] if len(payload)>4 else 'measured'
    first=refs[0];key=first['key']+'__'+first['site'];destination=out/phase/key
    terminal=destination/'result.json'
    if terminal.exists():return json.loads(terminal.read_text())
    coordinate_key=f'{first["dataset"]}__{first["site"]}__outer{first["subject_fold"]}'
    coordinate=json.loads((out/'coordinates'/(coordinate_key+'.json')).read_text())
    sd=np.asarray(coordinate['sd']);started=time.monotonic();rows=[]
    for ref in refs:
        path=destination/(f'w{ref["window"]}.json')
        if path.exists():rows.extend(json.loads(path.read_text())['rows']);continue
        y=semantics_target(ref,coordinate);saved=dict(target=y,sd=sd);window_rows=[]
        for candidate in semantics_parameter_candidates(cfg,out,ref,qualified):
            inherited=out/'fixed'/key/(f'w{ref["window"]}.json')
            if phase=='measured' and candidate['method']=='H0_fixed' and inherited.exists():
                old=json.loads(inherited.read_text());row=dict(old['rows'][0],inherited_from=str(inherited))
                with np.load(inherited.with_suffix('.npz'),allow_pickle=False) as a:
                    if not np.array_equal(y,a['target']) or not np.array_equal(sd,a['sd']):
                        raise ValueError('fixed baseline target/coordinate identity changed')
                    for field in a.files:
                        if field.startswith('H0_fixed__'):saved[field]=a[field].copy()
                window_rows.append(row);continue
            row=dict(identity=ref['id'],dataset=ref['dataset'],subject=ref['subject'],record=ref['record'],
                site=ref['site'],region=ref['region'],window=ref['window'],method=candidate['method'],
                parameter=candidate['parameter'],parameter_value=candidate['value'],
                native_rho=ref['native_rho'],quality_burden=ref['quality_burden'],
                record_evaluation=ref['record_evaluation'],coordinate=coordinate_key,
                status='failed_training',converged=False,all_below_half=False)
            if not candidate['converged']:
                row['training_status']=candidate['status'];window_rows.append(row);continue
            values={} if candidate['parameter'] is None else {candidate['parameter']:candidate['value']}
            p=semantics_parameters(cfg,**values)
            fit=fit_nonlinear_shared_driver(y,p,.25,sd=sd,mean_operator=semantics_model_operator(),
                starts=None,max_evaluations=cfg['solver']['max_evaluations_per_trial'],
                **semantics_solver_options(cfg))
            row.update(status=fit['status'],converged=fit['converged'],
                evaluations=fit.get('evaluations'),starts=fit['starts'],
                gradient=fit.get('scaled_gradient_inf_norm'),objective=fit.get('objective'),
                weighted_data_sse=fit.get('weighted_data_sse'),flow_prior_cost=fit.get('flow_prior_cost'),
                initial_state_penalty_cost=fit.get('initial_state_penalty_cost'),
                state_ranges=fit.get('state_ranges'),training_key=candidate.get('training_key'))
            if 'prediction' in fit:
                row.update(semantics_fit_metrics(y,fit,sd))
                method=candidate['method']
                for field in ('prediction','driver','states','initial_state'):
                    saved[method+'__'+field]=fit[field]
            window_rows.append(row)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path.with_suffix('.npz'),**saved)
        write_json(path,dict(ref=ref,rows=window_rows,endpoint='conditional_full_observation_reconstruction'))
        rows.extend(window_rows)
    result=dict(key=key,status='completed',windows=len(refs),rows=rows,seconds=time.monotonic()-started,
        nuisance_initialization='same_zero_driver_and_physical_rest_for_all_parameter_methods',
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_json(terminal,result);return result


def semantics_parallel_stage(cfg,out,stage,worker,payloads,workers,key_fn):
    started=time.monotonic();failed=[]
    for k,(payload,result,error) in enumerate(bounded_nonlinear_work(worker,payloads,workers,2*workers),1):
        key=key_fn(payload)
        if error:
            failed.append(key);write_json(out/(stage+'_failures')/(key+'.json'),dict(error=error))
        progress=dict(stage=stage,completed=k,expected=len(payloads),last_key=key,error=error,
            elapsed_s=time.monotonic()-started,estimated_remaining_s=(time.monotonic()-started)*(len(payloads)-k)/k)
        write_json(out/(stage+'_progress.json'),progress);print(json.dumps(progress),flush=True)
    return failed


def semantics_diagnostic_plan(cfg,out):
    refs=json.loads((out/'cohort.json').read_text())['refs']
    inventory=json.loads((out/'inventory.json').read_text())['records']
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],61]))
    selected={};strata=[];windows=[]
    for ds in cfg['data']['datasets']:
        burdens={}
        for spec in inventory:
            if spec['dataset']!=ds:continue
            record=json.loads((out/'prepared'/spec['key']/'record.json').read_text())
            scores=[1-c['support']+float(np.mean(c.get('jump_fraction',[1,1])))+
                    float(np.mean(c.get('flat_step_fraction',[1,1]))) for c in record['qc']['channels']]
            burdens.setdefault(spec['subject'],[]).extend(scores)
        order=sorted(burdens,key=lambda s:(float(np.mean(burdens[s])),s))
        selected[ds]=[]
        for stratum,block in enumerate(np.array_split(order,3)):
            choice=rng.choice(block,size=min(2,len(block)),replace=False).tolist()
            selected[ds].extend(choice)
            strata.extend(dict(dataset=ds,subject=s,stratum=stratum,quality_burden=float(np.mean(burdens[s]))) for s in choice)
        for subject in selected[ds]:
            own=[r for r in refs if r['dataset']==ds and r['subject']==subject and r['record_evaluation']]
            if not own:continue
            sites=sorted({r['site'] for r in own},key=lambda s:(not s.startswith('prefrontal'),s))
            site=sites[0]
            for repeat in (0,1):
                available=sorted([r for r in own if r['site']==site and r['repeat_fold']==repeat],
                                 key=lambda r:(r['record'],r['eeg_start_s']))
                if not available:continue
                record=available[0]['record'];available=[r for r in available if r['record']==record]
                for index in sorted(set((0,len(available)//2))):windows.append(available[index])
    plan=dict(selected_subjects=selected,strata=strata,windows=windows,
        selection='seeded two subjects per QC-burden tercile; no SSM errors or fitted parameters read',
        unsupported='REFED remains Hb QC only; no invented cross-modal spatial correspondence')
    write_json(out/'diagnostic_plan.json',plan);return plan


def semantics_missing_mask(mode,n=120):
    mask=np.ones((n,3),dtype=bool);left=(n-16)//2
    if mode=='center_EEG':mask[left:left+16,0]=False
    elif mode=='center_Hb':mask[left:left+16,1:]=False
    elif mode=='HbO_hidden':mask[:,1]=False
    elif mode=='HbR_hidden':mask[:,2]=False
    elif mode=='EEG_hidden':mask[:,0]=False
    elif mode=='Hb_hidden':mask[:,1:]=False
    elif mode=='prefix_Hb':mask[40:,1:]=False
    else:raise ValueError('unknown missing-feature endpoint')
    return mask


def semantics_hidden_scores(prediction,target,mask,sd):
    scores=[]
    for j in range(3):
        hidden=~mask[:,j]
        scores.append(float(np.sqrt(np.mean((prediction[hidden,j]-target[hidden,j])**2))/sd[j])
            if hidden.any() and np.isfinite(prediction[hidden,j]).all() else None)
    return scores


def semantics_ridge_completion(train,target,mask,*,cross):
    """Training-only ridge on identical feature supports; kernel form for small B."""
    prediction=np.full_like(target,np.nan)
    for j in range(3):
        hidden=~mask[:,j]
        if not hidden.any():continue
        inputs=mask.copy()
        if not cross:
            inputs[:]=False;inputs[:,j]=mask[:,j]
        if not inputs.any():continue
        x=train[:,inputs];v=target[inputs]
        center=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-8)
        z=(x-center)/scale;point=(v-center)/scale
        # Fixed engineering ridge strength; no heldout values choose it.
        labels=train[:,hidden,j];mean=labels.mean(axis=0)
        coefficient=np.linalg.solve(z@z.T/len(z)+np.eye(len(z)),labels-mean)
        prediction[hidden,j]=point@z.T/len(z)@coefficient+mean
    return prediction


def semantics_missing_worker(payload):
    from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_driver
    cfg,root,ref=payload;out=Path(root);path=out/'missing'/ref['id']/'result.json'
    if path.exists():return json.loads(path.read_text())
    started=time.monotonic();key=f'{ref["dataset"]}__{ref["site"]}__outer{ref["subject_fold"]}'
    coordinate=json.loads((out/'coordinates'/(key+'.json')).read_text());sd=np.asarray(coordinate['sd'])
    y=semantics_target(ref,coordinate)
    refs=json.loads((out/'cohort.json').read_text())['refs']
    library=[r for r in refs if r['dataset']==ref['dataset'] and r['site']==ref['site'] and
             r['subject']==ref['subject'] and r['repeat_fold']==1-ref['repeat_fold']]
    if not library:
        library=[r for r in refs if r['dataset']==ref['dataset'] and r['site']==ref['site'] and r['subject_fold']!=ref['subject_fold']]
    library=semantics_balanced_select(library,cfg['splits']['training_windows'],cfg['seed']+71)
    train=np.array([semantics_target(r,coordinate) for r in library])
    donors=[i for i,r in enumerate(library) if r['task']==ref['task'] and r['condition']==ref['condition']]
    donor=train[donors[0]] if donors else None
    spatial=[r for r in refs if r['key']==ref['key'] and r['window']==ref['window'] and r['region']!=ref['region']]
    spatial_hb=None
    if spatial:
        _,hb=semantics_prepared_arrays(spatial[0]['array_path'])
        spatial_hb=hb[spatial[0]['array_index']]*coordinate['hb_factor']
    qualified=json.loads((out/'training_plan.json').read_text())['qualified_parameters']
    candidates=[c for c in semantics_parameter_candidates(cfg,out,ref,qualified)
                if c['method']=='H0_fixed' or c['method'].startswith('H2_subject__')]
    rows=[];saved=dict(target=y,sd=sd)
    for mode in cfg['endpoints']['missing_modes']:
        mask=semantics_missing_mask(mode)
        own=np.full_like(y,np.nan)
        for j in range(3):
            visible=np.flatnonzero(mask[:,j])
            if len(visible)>=2:own[:,j]=np.interp(np.arange(120),visible,y[visible,j])
        baselines=dict(training_template=train.mean(axis=0),own_context=own,
            ridge_own=semantics_ridge_completion(train,y,mask,cross=False),
            ridge_cross=semantics_ridge_completion(train,y,mask,cross=True))
        for name,pred in baselines.items():
            scores=semantics_hidden_scores(pred,y,mask,sd)
            available=any(v is not None for v in scores)
            rows.append(dict(mode=mode,method=name,arm='real',converged=available,
                status='completed' if available else 'not_applicable_no_visible_same_modality_context',
                hidden_nrmse=scores))
            saved[mode+'__'+name]=pred
        arms=[('real',y.copy(),mask.copy())]
        other=[1,2] if mode in ('center_EEG','EEG_hidden') else [0]
        if donor is not None:
            null=y.copy();null[:,other]=donor[:,other]
            arms.append(('training_donor',null,mask.copy()))
        if mode in ('center_EEG','center_Hb'):
            support=mask.copy();support[-48:,other]=False
            shifted=y.copy();shifted[:-48,other]=y[48:,other];shifted[-48:,other]=np.nan
            arms.extend([('real_shift_support',y.copy(),support),('nonwrapping_shift_12s',shifted,support)])
        if mode=='center_EEG' and spatial_hb is not None:
            null=y.copy();null[:,1:]=spatial_hb;arms.append(('spatial_Hb_anchor',null,mask.copy()))
        for candidate in candidates:
            for arm,data,visible in arms:
                row=dict(mode=mode,method=candidate['method'],arm=arm,converged=False,
                    training_key=candidate.get('training_key'),parameter_value=candidate['value'])
                if not candidate['converged']:
                    row['status']='failed_training';rows.append(row);continue
                values={} if candidate['parameter'] is None else {candidate['parameter']:candidate['value']}
                # Hidden target values never initialize nuisance states or normalization.
                data=data.copy();data[~visible]=np.nan
                fit=fit_nonlinear_shared_driver(data,semantics_parameters(cfg,**values),.25,
                    sd=sd,visible=visible,mean_operator=semantics_model_operator(),
                    max_evaluations=cfg['solver']['max_evaluations_per_trial'],**semantics_solver_options(cfg))
                row.update(status=fit['status'],converged=fit['converged'],evaluations=fit.get('evaluations'))
                if 'prediction' in fit:
                    # Score only the original endpoint; additional support masks are null controls.
                    row['hidden_nrmse']=semantics_hidden_scores(fit['prediction'],y,mask,sd)
                    if arm=='real':saved[mode+'__'+candidate['method']]=fit['prediction']
                rows.append(row)
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**saved)
    result=dict(ref=ref,training_ids=[r['id'] for r in library],donor_id=library[donors[0]]['id'] if donors else None,
        spatial_donor_id=spatial[0]['id'] if spatial else None,rows=rows,status='completed',
        endpoint='processed_feature_missing_completion_not_raw_sensor_prediction',seconds=time.monotonic()-started)
    write_json(path,result);return result


def semantics_synthetic_specs(cfg):
    from itertools import product
    s = cfg['synthetic']
    return [dict(tau=t, gain=g, spectrum=f, repeat=r, initial=i,
                 key=f't{t:g}_g{g:g}_{f}_r{r}_{i}')
            for t, g, f, r, i in product(s['tau'], s['gain'], s['spectra'],
                range(s['repeats']), s['initial_conditions'])]


@lru_cache(maxsize=3)
def semantics_model_operator(steps=120):
    op = native_feature_operators(steps)
    mean = np.zeros((3*steps,3*steps))
    mean[0::3,0::3] = op['eeg']
    mean[1::3,1::3] = mean[2::3,2::3] = op['fnirs']@op['native_interpolation']
    return mean


def semantics_generate(cfg, spec):
    from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
    s, n, dt = cfg['synthetic'], cfg['tensor']['steps'], cfg['tensor']['dt_s']
    # Pair innovations/noise across physiological truth and initial-state arms.
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], 11,
        spec['repeat'], int(spec['spectrum'] == 'mixed')]))
    p = semantics_parameters(cfg, tau=spec['tau'], neurovascular_gain=spec['gain'],
                            kappa=spec.get('kappa',cfg['fixed']['kappa']))
    drivers, states, clean = [], [], []
    for _ in range(s['train_trials']+s['validation_trials']):
        r = gaussian_filter1d(rng.normal(size=n), 5.)
        if spec['spectrum'] == 'mixed':
            r += .18*gaussian_filter1d(rng.normal(size=n), .7)
        r = s['driver_sd']*r/r.std()
        perturbation = rng.normal(size=5)*.015
        initial = np.r_[0., np.ones(4)] + (perturbation if spec['initial'] == 'nonrest' else 0)
        forward = nonlinear_driver_forward(r, initial, p, dt, substeps=8,
                                           derivative=False, numerical_backend='numba')
        drivers.append(r); states.append(forward['states']); clean.append(forward['canonical_prediction'])
    clean = np.array(clean)
    if spec.get('coordinate') == 'processed':
        operator = semantics_model_operator(n)
        clean = np.array([(operator@v.ravel()).reshape(n,3) for v in clean])
    noise_sd = np.std(clean[:s['train_trials']], axis=(0, 1))*s['noise_fraction']
    y = clean+rng.normal(size=clean.shape)*noise_sd
    return dict(target=y, clean=clean, driver=np.array(drivers), states=np.array(states),
                sd=np.std(y[:s['train_trials']], axis=(0, 1)))


def semantics_stress_generate(cfg,spec):
    """Paired observation faults with continuous extra samples for clock shifts."""
    wide=deepcopy(cfg);wide['tensor']['steps']=168
    source=semantics_generate(wide,dict(spec,coordinate='canonical'))
    clean=source['clean'][:,:120].copy();y=source['target'][:,:120].copy()
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],81,spec['repeat'],int(spec['spectrum']=='mixed')]))
    scale=np.std(source['clean'][:18,:120],axis=(0,1));kind=spec['stress']
    common=gaussian_filter1d(rng.normal(size=(len(y),120)),8.,axis=1)
    common/=np.maximum(common.std(axis=1,keepdims=True),1e-12)
    if kind=='common_colored':y[:,:,1:]+=.5*common[:,:,None]*np.mean(scale[1:])
    elif kind=='independent_colored':
        noise=gaussian_filter1d(rng.normal(size=(len(y),120,2)),8.,axis=1)
        noise/=np.maximum(noise.std(axis=1,keepdims=True),1e-12)
        y[:,:,1:]+=.5*noise*scale[1:]
    elif kind=='drift':y[:,:,1:]+=.5*np.linspace(-1,1,120)[None,:,None]*scale[1:]
    elif kind=='jump':y[:,60:,1:]+=scale[1:]
    elif kind=='swapped_Hb':y[:,:,1:]=y[:,:,[2,1]]
    elif kind=='lag':y[:,:,1:]=source['target'][:,48:168,1:]
    elif kind!='control':raise ValueError('unknown observation stress')
    operator=semantics_model_operator()
    clean=np.array([(operator@v.ravel()).reshape(120,3) for v in clean])
    y=np.array([(operator@v.ravel()).reshape(120,3) for v in y])
    return dict(target=y,clean=clean,driver=source['driver'][:,:120],states=source['states'][:,:120],
                sd=np.std(clean[:18],axis=(0,1)))


def semantics_compact_fit(result):
    excluded = {'driver', 'prediction', 'canonical_prediction', 'initial_state', 'states',
                'jacobian', 'parameter_jacobian', 'driver_prior_sd'}
    return {k: v for k, v in result.items() if k not in excluded}


def semantics_synthetic_worker(payload):
    from src.inference.shared_driver_reconstruction import (fit_nonlinear_shared_parameter,
        fit_nonlinear_shared_driver, nonlinear_driver_forward)
    cfg, root, spec = payload
    folder = ('synthetic_stress' if 'stress' in spec else
              'synthetic_feature' if spec.get('coordinate') == 'processed' else 'synthetic')
    path = Path(root)/folder/spec['key']/'result.json'
    if path.exists():
        return json.loads(path.read_text())
    started = time.monotonic()
    arrays = semantics_stress_generate(cfg,spec) if 'stress' in spec else semantics_generate(cfg, spec)
    ntrain = cfg['synthetic']['train_trials']; n = cfg['tensor']['steps']
    train, test = arrays['target'][:ntrain], arrays['target'][ntrain:]
    options = semantics_solver_options(cfg)
    if spec.get('coordinate') == 'processed':
        options['mean_operator'] = semantics_model_operator(n)
    rows, trajectories = [], {}
    for name, truth in (('tau', spec['tau']), ('neurovascular_gain', spec['gain'])):
        # Qualification of one parameter with the other physiology fixed to truth.
        p = semantics_parameters(cfg, tau=spec['tau'], neurovascular_gain=spec['gain'])
        starts = [dict(parameter_value=value, driver=np.zeros((ntrain, n)),
                       initial_state=np.tile(np.r_[0., np.ones(4)], (ntrain, 1)))
                  for value in cfg['parameters'][name]['starts']]
        result = fit_nonlinear_shared_parameter(train, p, .25, parameter_name=name,
            parameter_bounds=cfg['parameters'][name]['bounds'], sd=arrays['sd'], starts=starts,
            max_evaluations=cfg['solver']['max_evaluations_per_trial']*ntrain,
            max_iterations=cfg['solver']['max_iterations'],
            step_control=cfg['solver']['step_control'], **options)
        row = dict(parameter=name, truth=truth, **semantics_compact_fit(result))
        row['relative_error'] = (abs(result['parameter_value']/truth-1)
                                 if result['parameter_value'] is not None else None)
        pfit = semantics_parameters(cfg, tau=result.get('tau', spec['tau']),
            neurovascular_gain=result.get('neurovascular_gain', spec['gain']))
        validation = []
        for i, y in enumerate(test):
            fit = fit_nonlinear_shared_driver(y, pfit, .25, sd=arrays['sd'],
                max_evaluations=cfg['solver']['max_evaluations_per_trial'], **options)
            item = semantics_compact_fit(fit)
            if 'prediction' in fit:
                item['nrmse'] = np.sqrt(np.mean((fit['prediction']-y)**2, axis=0))/arrays['sd']
                item['driver_nrmse'] = np.sqrt(np.mean((fit['driver']-arrays['driver'][ntrain+i])**2))/np.std(arrays['driver'][:ntrain])
                item['state_rmse'] = np.sqrt(np.mean((fit['states']-arrays['states'][ntrain+i])**2, axis=0))
                fine = nonlinear_driver_forward(fit['driver'], fit['initial_state'], pfit, .25,
                    substeps=8, derivative=False, numerical_backend='numba')
                item['substep_4_8_max_difference'] = float(np.max(abs(fine['canonical_prediction']-fit['canonical_prediction'])))
                trajectories[name+f'_prediction_{i}'] = fit['prediction']
            validation.append(item)
        row['validation'] = validation
        rows.append(row)
    result = dict(spec=spec, status='completed', rows=rows, seconds=time.monotonic()-started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'), **arrays, **trajectories)
    write_json(path, result)
    return result


def semantics_observation_math(cfg,out):
    from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
    from src.inference import t3a_balloon_robust_ssm as core
    from scipy.linalg import expm
    p=semantics_parameters(cfg);driver=.025*np.sin(np.arange(120)/12)+.01
    initial=np.array([.01,1.02,1.01,1.03,1.02])
    baseline=nonlinear_driver_forward(driver,initial,p,.25,numerical_backend='numba',derivative=False)
    gauges=[]
    for factor in (.25,1.,4.):
        alternate=replace(p,fixed=replace(p.fixed,eeg_loading=p.fixed.eeg_loading/factor,
                                         neurovascular_gain=p.fixed.neurovascular_gain/factor))
        replay=nonlinear_driver_forward(factor*driver,initial,alternate,.25,numerical_backend='numba',derivative=False)
        gauges.append(dict(driver_scale=factor,eeg_loading=alternate.fixed.eeg_loading,
            beta=alternate.fixed.neurovascular_gain,prediction_max_difference=float(np.max(abs(
                replay['canonical_prediction']-baseline['canonical_prediction']))),
            vascular_state_max_difference=float(np.max(abs(replay['states'][:,1:]-baseline['states'][:,1:]))),
            curvature_weight_for_equivalent_objective=cfg['solver']['penalty']/factor**2))
    h=core.observation_jacobian(np.zeros(6),p)
    a=core.balloon_rhs_jacobian(np.zeros(6),p)[1:,1:]
    transition=expm(a*.25)
    observability=np.vstack([h[:,1:]@np.linalg.matrix_power(transition,k) for k in range(120)])
    sv=np.linalg.svd(observability,compute_uv=False)
    y=baseline['canonical_prediction']
    total=y[:,1]+y[:,2]
    result=dict(instantaneous_observation_rank=int(np.linalg.matrix_rank(h)),state_count=6,
        known_input_vascular_linear_observability_rank=int(np.linalg.matrix_rank(observability)),
        known_input_vascular_singular_values=sv,gauge_examples=gauges,
        HbT_identity_max_error=float(np.max(abs(total-p.fixed.P0*(baseline['states'][:,4]-1)))),
        direct_coordinates=['r given fixed EEG loading','p from HbO+HbR given P0','q from HbR given Q0'],
        dynamically_constrained=['s','f','v'],
        source_separation='no unique scalp/cortex, arterial/venous or metabolic decomposition from these three observations',
        temporal_rank_interpretation='local known-input rest linearization; not finite-noise nonlinear identifiability',
        citations=[dict(title='Raue et al. 2009',url='https://pubmed.ncbi.nlm.nih.gov/19505944/'),
                   dict(title='Kirilina et al. 2012',url='https://pmc.ncbi.nlm.nih.gov/articles/PMC3348501/'),
                   dict(title='Arand et al. 2015',url='https://pmc.ncbi.nlm.nih.gov/articles/PMC4335185/')])
    write_json(out/'observation_math.json',result);return result


def semantics_hierarchy_generate(cfg,spec):
    from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],91,spec['repeat']]))
    shape=tuple(cfg['synthetic']['hierarchy_shape']);indices=list(np.ndindex(shape))
    dataset_effect=np.array([-.45,-.15,.15,.45])
    subject_effect=rng.normal(0,.25,size=shape[:2])
    record_effect=rng.normal(0,.25,size=shape[:3])
    trial_effect=rng.normal(0,.25,size=shape)
    kind=spec['mechanism'];observations=[];drivers=[];states=[];truth=[];clean=[]
    op=semantics_model_operator()
    for ds,subject,record,trial in indices:
        effect=0.
        if kind=='dataset':effect=dataset_effect[ds]
        elif kind=='subject':effect=subject_effect[ds,subject]
        elif kind=='record':effect=record_effect[ds,subject,record]
        elif kind=='trial':effect=trial_effect[ds,subject,record,trial]
        tau=2*np.exp(effect)
        driver=gaussian_filter1d(rng.normal(size=120),4.)
        driver=.025*driver/driver.std()
        initial=np.r_[0.,np.ones(4)]+rng.normal(0,.01,5)
        forward=nonlinear_driver_forward(driver,initial,semantics_parameters(cfg,tau=tau),.25,
            substeps=8,numerical_backend='numba',derivative=False)
        value=forward['canonical_prediction'].copy()
        if kind=='eeg_gain_only':value[:,0]*=np.exp(2*subject_effect[ds,subject])
        elif kind=='hb_gain_only':value[:,1:]*=np.exp(2*subject_effect[ds,subject])
        transformed=(op@value.ravel()).reshape(120,3)
        noise=.05*np.std(transformed,axis=0)
        if kind=='noise_only':noise*=np.exp(2*subject_effect[ds,subject])
        observations.append(transformed+rng.normal(size=transformed.shape)*noise)
        clean.append(transformed);drivers.append(driver);states.append(forward['states']);truth.append(tau)
    observations=np.array(observations);train=[i for i,index in enumerate(indices) if index[2]<2]
    return dict(target=observations,truth_tau=np.array(truth),indices=indices,
        driver=np.array(drivers),states=np.array(states),clean=np.array(clean),
        sd=np.std(observations[train],axis=(0,1)))


def semantics_hierarchy_worker(payload):
    from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_parameter,fit_nonlinear_shared_driver
    cfg,root,spec=payload;out=Path(root);path=out/'synthetic_hierarchy'/spec['key']/'result.json'
    if path.exists():return json.loads(path.read_text())
    started=time.monotonic();a=semantics_hierarchy_generate(cfg,spec);y=a['target'];indices=a['indices']
    options=semantics_solver_options(cfg);fits={};rows=[];rng=np.random.default_rng(cfg['seed']+101+spec['repeat'])
    def fit_group(key,ids,prior=None):
        checkpoint=path.parent/'fits'/(key+'.json')
        if checkpoint.exists():
            fits[key]=json.loads(checkpoint.read_text());return fits[key]
        values=y[ids]
        starts=[dict(parameter_value=v,driver=np.zeros((len(ids),120)),
                     initial_state=np.tile(np.r_[0.,np.ones(4)],(len(ids),1))) for v in cfg['parameters']['tau']['starts']]
        result=fit_nonlinear_shared_parameter(values,semantics_parameters(cfg),.25,parameter_name='tau',
            parameter_bounds=cfg['parameters']['tau']['bounds'],sd=a['sd'],mean_operator=semantics_model_operator(),
            starts=starts,max_evaluations=cfg['solver']['max_evaluations_per_trial']*len(ids),
            max_iterations=cfg['solver']['max_iterations'],step_control=cfg['solver']['step_control'],
            **options,**(prior or {}))
        fits[key]=dict(value=result.get('parameter_value'),converged=result['converged'],status=result['status'],
            boundary=result.get('boundary_status'),train_ids=ids,truth_tau=a['truth_tau'][ids],
            evaluations=result.get('evaluations'),objective=result.get('objective'),starts=result['starts'],
            information=result.get('information'))
        if 'driver' in result:
            fits[key]['driver_nrmse']=float(np.sqrt(np.mean((result['driver']-a['driver'][ids])**2))/np.std(a['driver'][ids]))
            fits[key]['state_rmse']=np.sqrt(np.mean((result['states']-a['states'][ids])**2,axis=(0,1)))
        if key.startswith('H4') and 'prediction' in result:
            fits[key]['conditional_training_nrmse']=np.sqrt(np.mean((result['prediction']-values)**2,axis=(0,1)))/a['sd']
        write_json(checkpoint,fits[key])
        return fits[key]
    train=[i for i,x in enumerate(indices) if x[2]<2]
    fit_group('H1_global',sorted(rng.choice(train,18,replace=False).tolist()))
    for ds in range(4):
        ids=[i for i,x in enumerate(indices) if x[0]==ds and x[2]<2]
        parent=fit_group(f'H1_dataset_{ds}',sorted(rng.choice(ids,18,replace=False).tolist()))
        for subject in range(4):
            ids=[i for i,x in enumerate(indices) if x[:2]==(ds,subject) and x[2]<2]
            fit_group(f'H2_subject_{ds}_{subject}',ids)
            if parent['converged']:
                fit_group(f'H2s_{ds}_{subject}',ids,dict(parameter_prior_mean=parent['value'],
                    parameter_prior_log_sd=np.log(16)/4.))
            for record in range(3):
                ids=[i for i,x in enumerate(indices) if x[:3]==(ds,subject,record) and x[3]<3]
                fit_group(f'H3_record_{ds}_{subject}_{record}',ids)
    for i,index in enumerate(indices):fit_group(f'H4_trial_{i}',[i])
    # The third record and its last three windows are independent of all H0--H3 calibrations.
    validation=[i for i,x in enumerate(indices) if x[2]==2 and x[3]>=3]
    for i in validation:
        ds,subject,record,trial=indices[i]
        candidates=[('H0_fixed',dict(value=2.,converged=True)),('H1_global',fits['H1_global']),
            ('H1_dataset',fits[f'H1_dataset_{ds}']),('H2_subject',fits[f'H2_subject_{ds}_{subject}']),
            ('H2s_partial_pooling',fits.get(f'H2s_{ds}_{subject}',dict(converged=False))),
            ('H3_record',fits[f'H3_record_{ds}_{subject}_{record}'])]
        for method,parent in candidates:
            checkpoint=path.parent/'validation'/(f'{i}__{method}.json')
            if checkpoint.exists():rows.append(json.loads(checkpoint.read_text()));continue
            row=dict(index=i,identity=indices[i],method=method,converged=False,true_tau=a['truth_tau'][i],
                     estimated_tau=parent.get('value'))
            if parent['converged']:
                result=fit_nonlinear_shared_driver(y[i],semantics_parameters(cfg,tau=parent['value']),.25,
                    sd=a['sd'],mean_operator=semantics_model_operator(),
                    max_evaluations=cfg['solver']['max_evaluations_per_trial'],**options)
                row.update(status=result['status'],converged=result['converged'])
                if 'prediction' in result:row.update(semantics_fit_metrics(y[i],result,a['sd']))
            write_json(checkpoint,row)
            rows.append(row)
    result=dict(spec=spec,status='completed',fits=fits,validation=rows,
        truth_tau=a['truth_tau'],indices=indices,seconds=time.monotonic()-started,
        parameter='tau_only_other_physiology_fixed',H4_interpretation='same_trial_conditional_fit_not_independent_prediction')
    write_json(path,result);return result


def semantics_error_quantiles(errors):
    values=np.asarray(errors,dtype=float)
    if np.isfinite(values).all():return np.quantile(values,[.5,.9],axis=0)
    values=np.sort(np.where(np.isfinite(values),values,np.inf),axis=0)
    output=[]
    for q in (.5,.9):
        location=(len(values)-1)*q;left=int(np.floor(location));right=int(np.ceil(location))
        if left==right:output.append(values[left]);continue
        with np.errstate(invalid='ignore'):
            value=values[left]+(values[right]-values[left])*(location-left)
        output.append(np.where(np.isfinite(values[right]),value,np.inf))
    return np.array(output)


def semantics_synthetic_summary(cfg, out, specs, folder='synthetic'):
    rows = []
    for spec in specs:
        path = out/folder/spec['key']/'result.json'
        if path.exists():
            result = json.loads(path.read_text())
            for row in result.get('rows', []):
                rows.append(dict(**spec, parameter=row['parameter'], truth=row['truth'],
                    estimate=row.get('parameter_value'), converged=row.get('converged', False),
                    relative_error=row.get('relative_error'), boundary=row.get('boundary_status'),
                    validation_driver_nrmse=np.median([x.get('driver_nrmse', np.nan) for x in row['validation']]),
                    seconds=result['seconds']))
    pd.DataFrame(rows).to_csv(out/(folder+'_metrics.csv'), index=False)
    summary = dict(expected_panels=len(specs), terminal_panels=len({r['key'] for r in rows}), parameters={})
    for name in ('tau', 'neurovascular_gain'):
        selected = [r for r in rows if r['parameter'] == name]
        errors = [r['relative_error'] if r['converged'] and r['relative_error'] is not None
                  else float('inf') for r in selected]
        median, p90 = semantics_error_quantiles(errors) if errors else (float('inf'), float('inf'))
        success = sum(r['converged'] for r in selected)/len(specs)
        passed = (len(selected) == len(specs) and success >= cfg['gates']['optimizer_success_min']
            and median <= cfg['gates']['synthetic_median_relative_error_max']
            and p90 <= cfg['gates']['synthetic_p90_relative_error_max'])
        summary['parameters'][name] = dict(passed=bool(passed), success_rate=success,
            median_relative_error=median, p90_relative_error=p90,
            boundary_count=sum(r['boundary'] in ('LOWER', 'UPPER') for r in selected),
            median_validation_driver_nrmse=np.nanmedian([r['validation_driver_nrmse'] for r in selected]) if selected else None)
    write_json(out/(folder+'_summary.json'), summary)
    return summary


def semantics_joint_worker(payload):
    from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_parameters,fit_nonlinear_shared_driver
    cfg,root,spec=payload;out=Path(root);path=out/'synthetic_joint'/spec['key']/'result.json'
    if path.exists():return json.loads(path.read_text())
    started=time.monotonic();a=semantics_generate(cfg,dict(spec,coordinate='processed'))
    count=cfg['synthetic']['train_trials'];names=['tau','neurovascular_gain']
    starts=[dict(parameter_values=np.array(values)) for values in ((1.,.5),(2.,1.),(4.,2.),(1.,2.),(4.,.5))]
    fit=fit_nonlinear_shared_parameters(a['target'][:count],semantics_parameters(cfg),.25,
        parameter_names=names,parameter_bounds=[cfg['parameters'][name]['bounds'] for name in names],
        sd=a['sd'],mean_operator=semantics_model_operator(),starts=starts,
        max_iterations=cfg['solver']['max_iterations'],
        max_evaluations_per_trial=cfg['solver']['max_evaluations_per_trial'],**semantics_solver_options(cfg))
    row=semantics_compact_fit(fit)
    row['relative_errors']=abs(np.asarray(fit['parameter_values'])/np.array([spec['tau'],spec['gain']])-1) if fit['parameter_values'] is not None else None
    validation=[]
    if fit['parameter_values'] is not None:
        p=semantics_parameters(cfg,**dict(zip(names,fit['parameter_values'])))
        for i,y in enumerate(a['target'][count:]):
            v=fit_nonlinear_shared_driver(y,p,.25,sd=a['sd'],mean_operator=semantics_model_operator(),
                max_evaluations=cfg['solver']['max_evaluations_per_trial'],**semantics_solver_options(cfg))
            item=dict(status=v['status'],converged=v['converged'])
            if 'driver' in v:
                item.update(driver_nrmse=float(np.sqrt(np.mean((v['driver']-a['driver'][count+i])**2))/np.std(a['driver'][:count])),
                            nrmse=np.sqrt(np.mean((v['prediction']-y)**2,axis=0))/a['sd'])
            validation.append(item)
    result=dict(spec=spec,**row,validation=validation,seconds=time.monotonic()-started)
    write_json(path,result);return result


def semantics_joint_summary(cfg,out,specs):
    rows=[]
    for spec in specs:
        path=out/'synthetic_joint'/spec['key']/'result.json'
        if not path.exists():continue
        r=json.loads(path.read_text());errors=r['relative_errors'] if r.get('converged') else [float('inf')]*2
        rows.append(dict(**spec,converged=r.get('converged',False),tau_error=errors[0],gain_error=errors[1],
            estimated_tau=r['parameter_values'][0] if r.get('parameter_values') else None,
            estimated_gain=r['parameter_values'][1] if r.get('parameter_values') else None))
    frame=pd.DataFrame(rows);frame.to_csv(out/'synthetic_joint_metrics.csv',index=False)
    success=sum(r['converged'] for r in rows)/len(specs)
    quantiles=semantics_error_quantiles([[r['tau_error'],r['gain_error']] for r in rows]) if rows else np.full((2,2),np.inf)
    passed=len(rows)==len(specs) and success>=cfg['gates']['optimizer_success_min'] and bool(
        np.all(quantiles[0]<=cfg['gates']['synthetic_median_relative_error_max']) and
        np.all(quantiles[1]<=cfg['gates']['synthetic_p90_relative_error_max']))
    summary=dict(expected=len(specs),terminal=len(rows),success_rate=success,median_relative_errors=quantiles[0],
        p90_relative_errors=quantiles[1],passed=passed,parameter_order=['tau','neurovascular_gain'])
    write_json(out/'synthetic_joint_summary.json',summary);return summary


def semantics_kappa_worker(payload):
    from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_parameters,fit_nonlinear_shared_driver
    cfg,root,spec=payload;path=Path(root)/'synthetic_kappa'/spec['key']/'result.json'
    if path.exists():return json.loads(path.read_text())
    started=time.monotonic();a=semantics_generate(cfg,dict(spec,coordinate='processed'))
    fit=fit_nonlinear_shared_parameters(a['target'][:18],semantics_parameters(cfg),.25,
        parameter_names=['kappa'],parameter_bounds=[cfg['parameters']['kappa']['bounds']],
        sd=a['sd'],mean_operator=semantics_model_operator(),
        starts=[dict(parameter_values=[v]) for v in cfg['parameters']['kappa']['starts']],
        max_iterations=cfg['solver']['max_iterations'],max_evaluations_per_trial=cfg['solver']['max_evaluations_per_trial'],
        **semantics_solver_options(cfg))
    value=fit['parameter_values'][0] if fit['parameter_values'] is not None else None
    validation=[]
    if value is not None:
        for i,target in enumerate(a['target'][18:]):
            check=fit_nonlinear_shared_driver(target,semantics_parameters(cfg,kappa=value),.25,
                sd=a['sd'],mean_operator=semantics_model_operator(),
                max_evaluations=cfg['solver']['max_evaluations_per_trial'],**semantics_solver_options(cfg))
            row=dict(converged=check['converged'],status=check['status'])
            if 'prediction' in check:
                row.update(semantics_fit_metrics(target,check,a['sd']),
                    driver_nrmse=float(np.sqrt(np.mean((check['driver']-a['driver'][18+i])**2))/np.std(a['driver'][:18])))
            validation.append(row)
    result=dict(semantics_compact_fit(fit),spec=spec,relative_error=abs(value/spec['kappa']-1) if value is not None else None,
        validation=validation,seconds=time.monotonic()-started,interpretation='conditional_kappa_recovery_tau_and_gain_fixed_to_truth')
    write_json(path,result);return result


def semantics_joint_fit(cfg,y,sd,*,fixed=None,visible=None,flow_weight=None):
    from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_parameters
    options=semantics_solver_options(cfg)
    if flow_weight is not None:options['flow_prior_weight']=flow_weight
    return fit_nonlinear_shared_parameters(y,semantics_parameters(cfg,**(fixed or {})),.25,
        parameter_names=['tau','neurovascular_gain'],parameter_bounds=[cfg['parameters'][n]['bounds'] for n in ('tau','neurovascular_gain')],
        sd=sd,visible=visible,mean_operator=semantics_model_operator(),
        starts=[dict(parameter_values=v) for v in ([1.,.5],[2.,1.],[4.,2.],[1.,2.],[4.,.5])],
        max_iterations=cfg['solver']['max_iterations'],max_evaluations_per_trial=cfg['solver']['max_evaluations_per_trial'],**options)


def semantics_profile_worker(payload):
    from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_parameter,fit_nonlinear_shared_driver
    cfg,root,ref=payload;out=Path(root);path=out/'profiles'/ref['id']/'result.json'
    if path.exists():return json.loads(path.read_text())
    started=time.monotonic();refs=json.loads((out/'cohort.json').read_text())['refs']
    key=f'{ref["dataset"]}__{ref["site"]}__outer{ref["subject_fold"]}'
    coordinate=json.loads((out/'coordinates'/(key+'.json')).read_text());sd=np.array(coordinate['sd'])
    train=[r for r in refs if r['key']==ref['key'] and r['site']==ref['site'] and r['record_calibration']]
    y=np.array([semantics_target(r,coordinate) for r in train]);rows=[]
    joint=semantics_joint_fit(cfg,y,sd)
    grids=dict(tau=[.5,.75,1.,1.5,2.,3.,4.,6.,8.],neurovascular_gain=[.1,.2,.35,.5,1.,2.,3.5,6.,10.])
    for profiled,grid in grids.items():
        other='neurovascular_gain' if profiled=='tau' else 'tau'
        for value in grid:
            starts=[dict(parameter_value=v,driver=np.zeros(y.shape[:2]),initial_state=np.tile(np.r_[0.,np.ones(4)],(len(y),1))) for v in cfg['parameters'][other]['starts']]
            fit=fit_nonlinear_shared_parameter(y,semantics_parameters(cfg,**{profiled:value}),.25,
                parameter_name=other,parameter_bounds=cfg['parameters'][other]['bounds'],sd=sd,
                mean_operator=semantics_model_operator(),starts=starts,
                max_evaluations=cfg['solver']['max_evaluations_per_trial']*len(y),max_iterations=cfg['solver']['max_iterations'],
                step_control=cfg['solver']['step_control'],**semantics_solver_options(cfg))
            rows.append(dict(profiled=profiled,value=value,reoptimized=[other,'driver','initial'],**semantics_compact_fit(fit)))
    kappa_path=out/'synthetic_kappa_summary.json'
    if kappa_path.exists() and json.loads(kappa_path.read_text())['passed']:
        for value in (.2,.32,.5,.64,.8,1.1,1.5):
            fit=semantics_joint_fit(cfg,y,sd,fixed=dict(kappa=value))
            rows.append(dict(profiled='kappa',value=value,reoptimized=['tau','neurovascular_gain','driver','initial'],**semantics_compact_fit(fit)))
    sensitivities=[]
    for fixed_name in ('alpha','E0','gamma'):
        for value in (.2,.5):
            fit=semantics_joint_fit(cfg,y,sd,fixed={fixed_name:value})
            sensitivities.append(dict(kind='fixed_parameter',parameter=fixed_name,value=value,**semantics_compact_fit(fit)))
    for weight in (.1,1.,4.):
        fit=semantics_joint_fit(cfg,y,sd,flow_weight=weight)
        sensitivities.append(dict(kind='logflow_penalty',value=weight,**semantics_compact_fit(fit)))
    target=semantics_target(ref,coordinate);trial=[];trajectories=dict(target=target,sd=sd)
    prefix_mask=semantics_missing_mask('prefix_Hb')
    prefix_baselines={name:semantics_hidden_scores(pred,target,prefix_mask,sd) for name,pred in {
        'training_template':y.mean(axis=0),
        'ridge_own':semantics_ridge_completion(y,target,prefix_mask,cross=False),
        'ridge_cross':semantics_ridge_completion(y,target,prefix_mask,cross=True),
        'last_visible_value':np.repeat(target[39:40],120,axis=0)}.items()}
    prefix_observed=target.copy();prefix_observed[~prefix_mask]=np.nan
    prefix_fixed=fit_nonlinear_shared_driver(prefix_observed,semantics_parameters(cfg),.25,sd=sd,
        visible=prefix_mask,mean_operator=semantics_model_operator(),
        max_evaluations=cfg['solver']['max_evaluations_per_trial'],**semantics_solver_options(cfg))
    prefix_baselines['H0_fixed']=dict(status=prefix_fixed['status'],converged=prefix_fixed['converged'],
        hidden_nrmse=semantics_hidden_scores(prefix_fixed['prediction'],target,prefix_mask,sd) if 'prediction' in prefix_fixed else None)
    for mode in ('same_trial_full','first_10s_feature_calibration'):
        visible=np.ones((1,120,3),dtype=bool)
        if mode!='same_trial_full':visible[:,40:]=False
        masked=target[None].copy();masked[~visible]=np.nan
        fitted=semantics_joint_fit(cfg,masked,sd,visible=visible)
        row=dict(mode=mode,parameter_fit=semantics_compact_fit(fitted))
        if fitted.get('converged'):
            p=semantics_parameters(cfg,**dict(zip(('tau','neurovascular_gain'),fitted['parameter_values'])))
            mask=semantics_missing_mask('prefix_Hb') if mode!='same_trial_full' else np.ones_like(target,dtype=bool)
            observed=target.copy();observed[~mask]=np.nan
            final=fit_nonlinear_shared_driver(observed,p,.25,sd=sd,visible=mask,mean_operator=semantics_model_operator(),
                max_evaluations=cfg['solver']['max_evaluations_per_trial'],**semantics_solver_options(cfg))
            row['completion']=semantics_compact_fit(final)
            if 'prediction' in final:
                row['completion']['nrmse']=np.sqrt(np.mean((final['prediction']-target)**2,axis=0))/sd
                row['completion']['hidden_nrmse']=semantics_hidden_scores(final['prediction'],target,mask,sd)
                trajectories[mode]=final['prediction']
        trial.append(row)
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**trajectories)
    result=dict(ref=ref,training_ids=[r['id'] for r in train],joint=semantics_compact_fit(joint),profiles=rows,
        sensitivity=sensitivities,trial=trial,prefix_baselines=prefix_baselines,status='completed',seconds=time.monotonic()-started,
        interpretation='nuisance_reoptimized_engineering_objective_profiles_not_likelihood_confidence_intervals',
        prefix_interpretation='processed_feature_calibration_and_EEG_visible_Hb_completion_not_raw_causal_forecast')
    write_json(path,result);return result


def semantics_context_worker(payload):
    """One continuous raw support; identical central target across context lengths."""
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    from src.data.unified_physiology import load_native_eeg_record,load_native_fnirs_record
    from src.data.event_alignment import window_within_alignment_support
    from src.inference.observation_baselines import eeg_band_power
    from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_driver
    cfg,root,project_root,ref=payload;out=Path(root);path=out/'context'/ref['id']/'result.json'
    if path.exists():return json.loads(path.read_text())
    started=time.monotonic();index=CleanPhysiologyCacheIndex(Path(project_root)/cfg['data']['cache_root'])
    record=next(r for r in index.records if r.join_key.replace('|','__').replace('/','_')==ref['key'])
    event=next(e for e in index.events_by_join_key[record.join_key] if e['event_index']==ref['event_id'])
    if not window_within_alignment_support(event,-50.,120.):
        result=dict(ref=ref,status='unavailable_alignment_support',rows=[])
        write_json(path,result);return result
    eeg=load_native_eeg_record(Path(project_root),record);native=load_native_fnirs_record(Path(project_root),record)
    es,hs=ref['eeg_start_s']-45,ref['hb_start_s']-45
    start=round(es*eeg.sample_rate_hz);length=round(120*eeg.sample_rate_hz)
    if min(es,hs)<0 or start+length>len(eeg.values) or hs<native['time_s'][0] or hs+119.75>native['time_s'][-1]:
        result=dict(ref=ref,status='unavailable_native_support',rows=[])
        write_json(path,result);return result
    prepared=json.loads((out/'prepared'/ref['key']/'record.json').read_text())
    anchor=next(a for a in prepared['prepared'] if a['region']==ref['region'])
    lookup={str(n).upper():i for i,n in enumerate(eeg.channel_names)}
    indices=[lookup[n.upper()] for n in ref['eeg_channels']]
    power=eeg_band_power(eeg.values[start:start+length,indices],bands=cfg['tensor']['eeg_bands_hz'],sample_rate=eeg.sample_rate_hz,target_rate=4.)
    pair=native['values'][:,anchor['hb_pair']];times=hs+np.arange(480)/4.
    hb=np.column_stack([np.interp(times,native['time_s'],pair[:,j]) for j in range(2)])
    key=f'{ref["dataset"]}__{ref["site"]}__outer{ref["subject_fold"]}'
    coordinate=json.loads((out/'coordinates'/(key+'.json')).read_text());sd=np.array(coordinate['sd'])
    features=np.log(power).reshape(480,-1)
    y=np.column_stack([features@np.array(coordinate['pc'])*coordinate['eeg_factor'],hb*coordinate['hb_factor']])
    y-=y[:20].mean(axis=0);center=y[180:300].copy();rows=[];saved=dict(target=y,central_target=center,sd=sd)
    qualified=json.loads((out/'training_plan.json').read_text())['qualified_parameters']
    candidates=[c for c in semantics_parameter_candidates(cfg,out,ref,qualified) if c['method']=='H0_fixed' or c['method'].startswith('H2_subject')]
    for seconds in cfg['tensor']['context_s']:
        n=int(seconds*4);left=(480-n)//2;target=y[left:left+n];central=slice(180-left,300-left)
        if not np.array_equal(target[central],center):raise ValueError('context endpoint changed')
        for candidate in candidates:
            row=dict(seconds=seconds,method=candidate['method'],parameter_value=candidate['value'],converged=False,status='failed_training')
            if candidate['converged']:
                values={} if candidate['parameter'] is None else {candidate['parameter']:candidate['value']}
                fit=fit_nonlinear_shared_driver(target,semantics_parameters(cfg,**values),.25,sd=sd,
                    max_evaluations=cfg['solver']['max_evaluations_per_trial'],**semantics_solver_options(cfg))
                row.update(semantics_compact_fit(fit))
                if 'prediction' in fit:
                    central_fit=dict(prediction=fit['prediction'][central],converged=fit['converged'])
                    row.update(semantics_fit_metrics(center,central_fit,sd))
                    row['central_driver_mean']=float(fit['driver'][central].mean())
                    row['central_state_mean']=fit['states'][central].mean(axis=0)
                    for field in ('prediction','driver','states','initial_state'):saved[f'{seconds}__{candidate["method"]}__{field}']=fit[field]
            rows.append(row)
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**saved)
    dt=np.diff(native['time_s'])
    result=dict(ref=ref,status='completed',rows=rows,seconds=time.monotonic()-started,
        eeg_support_s=[es,es+120],hb_support_s=[hs,hs+120],native_clock_step_quantiles_s=np.quantile(dt,[0,.01,.5,.99,1]),
        processing='continuous_120s_logpower_PCA_and_unfiltered_timestamp_interpolated_Hb; common_first5s_reference; nested_slices_identity_mean',
        interpretation='complete_observation_context_diagnostic; distinct_from_primary_window_filtered_target; no_forecasting_claim')
    write_json(path,result);return result


def semantics_transfer_worker(payload):
    from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_driver
    cfg,root,ref=payload;out=Path(root);path=out/'transfer'/ref['id']/'result.json'
    if path.exists():return json.loads(path.read_text())
    started=time.monotonic();plan=json.loads((out/'training_plan.json').read_text())['jobs'];sources={}
    for spec in plan:
        if spec['hierarchy']!='H1_dataset' or spec['parameter']!='tau':continue
        train=spec['train'];ds=train[0]['dataset']
        if ds==ref['dataset'] or train[0]['region']!=ref['region']:continue
        fit=json.loads((out/'training'/spec['key']/'result.json').read_text())
        if fit['converged']:sources.setdefault(ds,[]).append(dict(key=spec['key'],tau=fit['parameter_value']))
    key=f'{ref["dataset"]}__{ref["site"]}__outer{ref["subject_fold"]}'
    coordinate=json.loads((out/'coordinates'/(key+'.json')).read_text());sd=np.array(coordinate['sd']);y=semantics_target(ref,coordinate)
    candidates=[dict(method='H0_fixed',tau=cfg['fixed']['tau'])]
    own=json.loads((out/'training'/(key+'__H1__tau')/'result.json').read_text())
    if own['converged']:candidates.append(dict(method='target_H1_tau',tau=own['parameter_value']))
    for ds,values in sources.items():
        candidates.append(dict(method='source_'+ds,tau=float(np.exp(np.mean(np.log([v['tau'] for v in values]))))))
    if sources:
        candidates.append(dict(method='source_equal_dataset_log_ensemble',tau=float(np.exp(np.mean(np.log([c['tau'] for c in candidates if c['method'].startswith('source_')]))))))
    rows=[]
    for candidate in candidates:
        fit=fit_nonlinear_shared_driver(y,semantics_parameters(cfg,tau=candidate['tau']),.25,sd=sd,
            mean_operator=semantics_model_operator(),max_evaluations=cfg['solver']['max_evaluations_per_trial'],**semantics_solver_options(cfg))
        row=dict(candidate,status=fit['status'],converged=fit['converged'],all_below_half=False)
        if 'prediction' in fit:row.update(semantics_fit_metrics(y,fit,sd))
        rows.append(row)
    result=dict(ref=ref,rows=rows,sources=sources,status='completed',seconds=time.monotonic()-started,
        interpretation='calibrated_observation_coordinate_tau_transfer; source_fold_estimates_are_overlapping_ensemble_members_not_independent_repeats',
        unsupported='strict_zero_shot_amplitude_or_beta_transfer; REFED_cross_modal_geometry',
        target_calibration_subjects=coordinate['training_subjects'])
    write_json(path,result);return result


def semantics_linear_worker(payload):
    """Retained linear owner as a same-feature reference, with validity flags intact."""
    cfg,root,refs=payload;out=Path(root);first=refs[0]
    key=f'{first["dataset"]}__{first["site"]}__outer{first["subject_fold"]}';path=out/'linear'/key/'result.json'
    if path.exists():return json.loads(path.read_text())
    started=time.monotonic();coordinate=json.loads((out/'coordinates'/(key+'.json')).read_text());sd=np.array(coordinate['sd'])
    design=build_shared_driver_design(semantics_parameters(cfg),120,.25,processed_mean_operator=semantics_model_operator())
    rows=[]
    for ref in refs:
        target=semantics_target(ref,coordinate)
        for method,penalty in [('linear_regularized',cfg['solver']['penalty']),('linear_unregularized',0.)]:
            fit=fit_shared_driver(target,design,observation_scale=sd,penalty=penalty,initial_penalty=cfg['solver']['initial_penalty'])
            metrics=semantics_fit_metrics(target,dict(prediction=fit['prediction'],converged=True),sd)
            rows.append(dict(identity=ref['id'],dataset=ref['dataset'],subject=ref['subject'],record=ref['record'],site=ref['site'],region=ref['region'],method=method,
                converged=True,**metrics,physical_valid=fit['physical_validity']['mathematical_valid'],small_signal_valid=fit['physical_validity']['small_signal_valid'],
                rank=fit['rank'],effective_df=fit['effective_df'],max_fractional_excursion=fit['physical_validity']['max_fractional_excursion']))
    result=dict(coordinate=key,status='completed',rows=rows,seconds=time.monotonic()-started,
        interpretation='same-feature rest-linearized reference; regularized arm shares curvature/initial penalties but has no nonlinear log-flow prior; unregularized arm is a linear-subspace residual bound, not a physical model qualification')
    write_json(path,result);return result


def semantics_icc_absolute(values):
    """Two-way random, single-measure absolute agreement; no clipping of negatives."""
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or x.shape[1]!=2 or len(x)<3 or not np.isfinite(x).all():return None
    n,k=x.shape;mean=x.mean();row=x.mean(axis=1);column=x.mean(axis=0)
    msr=k*np.sum((row-mean)**2)/(n-1);msc=n*np.sum((column-mean)**2)/(k-1)
    mse=np.sum((x-row[:,None]-column[None,:]+mean)**2)/((n-1)*(k-1))
    denominator=msr+(k-1)*mse+k*(msc-mse)/n
    return float((msr-mse)/denominator) if denominator>1e-20 else None


def semantics_bootstrap_mean(values,seed,repeats=2000):
    values=np.asarray(values,dtype=float);values=values[np.isfinite(values)]
    if len(values)<2:return [None,None]
    rng=np.random.default_rng(seed)
    return np.quantile(values[rng.integers(len(values),size=(repeats,len(values)))].mean(axis=1),[.025,.975]).tolist()


def semantics_training_summary(cfg,out):
    plan=json.loads((out/'training_plan.json').read_text())['jobs'];rows=[]
    for job in plan:
        result=json.loads((out/'training'/job['key']/'result.json').read_text());ref=job['train'][0]
        starts=[s.get('parameter_value') for s in result.get('starts',[]) if s.get('converged') and s.get('parameter_value') is not None]
        rows.append(dict(key=job['key'],dataset=ref['dataset'],subject=job['subject'],record=job['record'],site=ref['site'],region=ref['region'],
            coordinate=job['coordinate'],repeat=job['repeat'],repeat_kind=ref['repeat_kind'],hierarchy=job['hierarchy'],parameter=job['parameter'],
            value=result.get('parameter_value'),converged=result.get('converged',False),status=result.get('status'),
            boundary=result.get('boundary_status'),training_windows=len(job['train']),objective=result.get('objective'),
            start_value_min=min(starts) if starts else None,start_value_max=max(starts) if starts else None))
    frame=pd.DataFrame(rows);frame.to_csv(out/'training_parameters.csv',index=False)
    grouped=[]
    for key,part in frame.groupby(['dataset','hierarchy','parameter']):
        good=part[part.converged];grouped.append(dict(zip(['dataset','hierarchy','parameter'],key),n=len(part),successes=len(good),
            success_rate=len(good)/len(part),boundary_rate=part.boundary.isin(['LOWER','UPPER']).mean(),
            median=good.value.median(),p10=good.value.quantile(.1),p90=good.value.quantile(.9)))
    pd.DataFrame(grouped).to_csv(out/'training_summary.csv',index=False)
    pairs=[]
    for key,part in frame[frame.hierarchy.isin(['H2_subject','H2s_partial_pooling'])].groupby(['dataset','subject','site','coordinate','hierarchy','parameter']):
        item=dict(zip(['dataset','subject','site','coordinate','hierarchy','parameter'],key));item['repeat_kind']=part.iloc[0].repeat_kind
        a=part[part['repeat']==0];b=part[part['repeat']==1]
        item.update(eligible_pair=len(a)==1 and len(b)==1,successful_pair=False)
        if len(a)==1 and len(b)==1 and bool(a.iloc[0].converged) and bool(b.iloc[0].converged):
            x,z=float(a.iloc[0].value),float(b.iloc[0].value)
            item.update(successful_pair=True,A=x,B=z,symmetric_relative_difference=2*abs(x-z)/(x+z),log_difference=np.log(x/z),
                boundary_pair=a.iloc[0].boundary in ('LOWER','UPPER') or b.iloc[0].boundary in ('LOWER','UPPER'))
        pairs.append(item)
    paired=pd.DataFrame(pairs);paired.to_csv(out/'repeatability_pairs.csv',index=False);summary=[];iccs=[]
    for key,part in paired.groupby(['dataset','hierarchy','parameter']):
        good=part[part.successful_pair];subject=good.groupby('subject').symmetric_relative_difference.mean()
        summary.append(dict(zip(['dataset','hierarchy','parameter'],key),expected_pairs=int(part.eligible_pair.sum()),successful_pairs=len(good),
            subjects=good.subject.nunique(),median_pair_difference=good.symmetric_relative_difference.median(),
            subject_equal_mean_difference=subject.mean(),subject_mean_difference_CI=semantics_bootstrap_mean(subject,cfg['seed']+211),
            boundary_pair_fraction=good.boundary_pair.mean(),fraction_pairs_within_20pct=(good.symmetric_relative_difference<=.2).mean()))
    # Each ICC stratum has the same outer-trained EEG/Hb gauge, site and repeat type.
    for key,part in paired[paired.successful_pair].groupby(['dataset','site','coordinate','repeat_kind','hierarchy','parameter']):
        values=np.log(part[['A','B']].to_numpy(float));estimate=semantics_icc_absolute(values)
        rng=np.random.default_rng(cfg['seed']+212);samples=[]
        if estimate is not None:
            for _ in range(2000):
                v=semantics_icc_absolute(values[rng.integers(len(values),size=len(values))])
                if v is not None:samples.append(v)
        interval=np.quantile(samples,[.025,.975]).tolist() if len(samples)>=1000 else [None,None]
        iccs.append(dict(zip(['dataset','site','coordinate','repeat_kind','hierarchy','parameter'],key),subjects=len(part),
            ICC_A1_log_parameter=estimate,lower=interval[0],upper=interval[1],valid_bootstraps=len(samples),
            within_log_variance=float(np.mean((values[:,0]-values[:,1])**2)/2),between_log_variance=float(np.var(values.mean(axis=1),ddof=1)) if len(values)>1 else None,
            lower_above_06=interval[0] is not None and interval[0]>=.6))
    pd.DataFrame(iccs).to_csv(out/'repeatability_icc.csv',index=False)
    result=dict(groups=grouped,repeatability=summary,icc=iccs,
        interpretation='independent_record_or_block_repeats; ICC only within identical coordinate/site/repeat type; bootstrap conditions on the frozen outer calibration',
        limitation='calibration PCA/scale uncertainty not refitted in this conditional bootstrap; no across-day claim; no confirmatory nested selection of sharing level')
    write_json(out/'training_summary.json',result);return result


def semantics_qc_summary(cfg,out):
    inventory=json.loads((out/'inventory.json').read_text());rows=[];details=[];fingerprints={};duplicates=[]
    for spec in inventory['records']:
        record=json.loads((out/'prepared'/spec['key']/'record.json').read_text());detail=json.loads((out/'qc_detail'/(spec['key']+'.json')).read_text())
        for channel in record['qc']['channels']:
            rows.append(dict(dataset=spec['dataset'],subject=spec['subject'],record=spec['record'],**channel))
        identity=(spec['dataset'],detail['full_record_fingerprint'],tuple(detail['full_record_shape']))
        if identity in fingerprints:duplicates.append([fingerprints[identity],spec['key']])
        fingerprints[identity]=spec['key']
        detail['closure_max']=record['qc'].get('dependent_HbT_closure',{}).get('max_absolute');details.append(detail)
    frame=pd.DataFrame(rows);frame.to_csv(out/'native_qc_channels.csv',index=False);groups=[]
    for ds,part in frame.groupby('dataset'):
        subject=part.assign(positive=part.rho>.5).groupby('subject').agg(positive=('positive','mean'),rho=('rho','median'))
        d=[r for r in details if r['dataset']==ds]
        groups.append(dict(dataset=ds,subjects=len(subject),records=len(d),channels=len(part),supported=int((part.status=='supported').sum()),
            subject_equal_whole_record_positive_rho_gt05=subject.positive.mean(),subject_equal_median_rho=subject.rho.mean(),
            nonuniform_native_clocks=sum(not r['native_clock_uniform'] for r in d),
            exact_nonflat_channel_duplicates=sum(len(r['exact_nonflat_channel_duplicates']) for r in d),
            outside_native_support=sum(len(r['windows_outside_native_support']) for r in d),
            common_HbT_PC1_fraction_median=np.median([r['common_HbT_variance']['native']['first_component_variance_fraction'] for r in d if r['common_HbT_variance'].get('native')]),
            dependent_HbT_closure_max=max((r['closure_max'] for r in d if r['closure_max'] is not None),default=None)))
    result=dict(groups=groups,duplicate_full_records=duplicates,excluded=inventory['excluded'],
        denominator='native whole-record channel pairs; distinct from earlier 30s window prevalence',
        source_attribution='QC flags and covariance do not establish corrupted data or a physiological source')
    write_json(out/'native_qc_summary.json',result);return result


def semantics_synthetic_diagnostic_summary(cfg,out):
    stress=[];hierarchy=[];fitrows=[]
    for spec in json.loads((out/'synthetic_stress_plan.json').read_text())['specs']:
        result=json.loads((out/'synthetic_stress'/spec['key']/'result.json').read_text())
        for row in result['rows']:
            valid=[v for v in row['validation'] if v['converged'] and 'nrmse' in v]
            stress.append(dict(stress=spec['stress'],spectrum=spec['spectrum'],repeat=spec['repeat'],parameter=row['parameter'],
                converged=row['converged'],value=row.get('parameter_value'),relative_error=row.get('relative_error') if row['converged'] else None,boundary=row.get('boundary_status'),
                validation_success=len(valid)/6,validation_nrmse=float(np.mean([np.mean(v['nrmse']) for v in valid])) if valid else None,
                driver_nrmse=float(np.mean([v['driver_nrmse'] for v in valid])) if valid else None,
                state_rmse=np.mean([v['state_rmse'] for v in valid],axis=0).tolist() if valid else None))
    pd.DataFrame(stress).to_csv(out/'stress_metrics.csv',index=False)
    for spec in json.loads((out/'synthetic_hierarchy_plan.json').read_text())['specs']:
        result=json.loads((out/'synthetic_hierarchy'/spec['key']/'result.json').read_text())
        for row in result['validation']:
            hierarchy.append(dict(mechanism=spec['mechanism'],repeat=spec['repeat'],method=row['method'],subject=row['identity'][:2],
                converged=row['converged'],relative_error=abs(row['estimated_tau']/row['true_tau']-1) if row.get('estimated_tau') else None,
                all_below_half=row.get('all_below_half',False),nrmse=float(np.mean(row['nrmse'])) if row.get('converged') and 'nrmse' in row else None))
        for key,fit in result['fits'].items():
            fitrows.append(dict(mechanism=spec['mechanism'],repeat=spec['repeat'],key=key,hierarchy=key.rsplit('_',1)[0] if key.startswith('H4') else key.split('_')[0],
                value=fit.get('value'),truth_mean=float(np.mean(fit['truth_tau'])),converged=fit['converged'],boundary=fit.get('boundary'),
                driver_nrmse=fit.get('driver_nrmse'),state_rmse=fit.get('state_rmse'),conditional_training_nrmse=fit.get('conditional_training_nrmse')))
    pd.DataFrame(hierarchy).to_csv(out/'hierarchy_validation.csv',index=False);pd.DataFrame(fitrows).to_csv(out/'hierarchy_parameters.csv',index=False)
    h=pd.DataFrame(hierarchy);s=pd.DataFrame(stress);summary=[]
    for key,part in h.groupby(['mechanism','method']):
        cohorts=part.groupby('repeat').agg(nrmse=('nrmse','mean'),relative_error=('relative_error','median'),success=('converged','mean'),passed=('all_below_half','mean'))
        summary.append(dict(zip(['mechanism','method'],key),cohorts=len(cohorts),validation_success=cohorts.success.mean(),
            median_parameter_relative_error=cohorts.relative_error.median(),nrmse=cohorts.nrmse.mean(),pass_rate=cohorts.passed.mean(),
            nrmse_cohort_CI=semantics_bootstrap_mean(cohorts.nrmse,cfg['seed']+221)))
    stress_summary=[]
    for key,part in s.groupby(['stress','parameter']):
        control=s[(s.stress=='control')&(s.parameter==key[1])]
        paired=part.merge(control,on=['spectrum','repeat'],suffixes=('','_control'))
        paired=paired[paired.converged&paired.converged_control]
        stress_summary.append(dict(zip(['stress','parameter'],key),panels=len(part),success_rate=part.converged.mean(),
            median_relative_error=part.relative_error.median(),boundary_rate=part.boundary.isin(['LOWER','UPPER']).mean(),
            driver_nrmse=part.driver_nrmse.mean(),nrmse=part.validation_nrmse.mean(),
            paired_parameter_shift_median=np.median(abs(paired.value-paired.value_control)/paired.value_control)))
    allocation=[];f=pd.DataFrame(fitrows)
    for (kind,repeat),part in f.groupby(['mechanism','repeat']):
        for prefix in ['H1_dataset_','H2_subject_','H2s_','H3_record_','H4_trial_']:
            group=part[part.key.str.startswith(prefix)&part.converged].dropna(subset=['value'])
            if not len(group):continue
            estimated=np.log(group.value.to_numpy());true=np.log(group.truth_mean.to_numpy())
            i,j=np.triu_indices(len(group),1);nonzero=abs(true[i]-true[j])>1e-12
            direction=float(np.mean(np.sign(estimated[i[nonzero]]-estimated[j[nonzero]])==np.sign(true[i[nonzero]]-true[j[nonzero]]))) if nonzero.any() else None
            allocation.append(dict(mechanism=kind,repeat=int(repeat),hierarchy=prefix.rstrip('_'),fits=len(group),
                estimated_log_sd=float(np.std(estimated)),true_log_sd=float(np.std(true)),nonzero_true_pair_count=int(nonzero.sum()),direction_retention=direction,
                median_relative_error=float(np.median(abs(np.exp(estimated-true)-1)))))
    pd.DataFrame(allocation).to_csv(out/'hierarchy_variation_recovery.csv',index=False)
    result=dict(stress=stress_summary,hierarchy=summary,variation_recovery=allocation,
        interpretation='paired model truth; H4 fits its own target and is not a heldout sharing competitor')
    write_json(out/'synthetic_diagnostics_summary.json',result);return result


def semantics_hierarchical_scores(frame,metrics):
    """Windows, records, montage variants, regions, then subjects receive equal weight."""
    keys=['dataset','method','subject','region','site','record']
    record=frame.groupby(keys,dropna=False)[metrics].mean()
    site=record.groupby(level=['dataset','method','subject','region','site']).mean()
    region=site.groupby(level=['dataset','method','subject','region']).mean()
    return region.groupby(level=['dataset','method','subject']).mean().reset_index()


def semantics_block_bootstrap_gain(paired,seed,repeats=2000):
    """Paired subject/record/120-s block bootstrap, retaining dependent ROI draws."""
    rng=np.random.default_rng(seed);subject_draws=[]
    for subject,own in paired.groupby('subject'):
        time_choices={};record_choices={};site_values={}
        for (region,site,record),part in own.groupby(['region','site','record']):
            blocks=part.assign(block=np.floor(part.eeg_start_s/120).astype(int)).groupby('block').delta.agg(['sum','count'])
            cache_key=(record.split('_Probe')[0],tuple(blocks.index))
            if cache_key not in time_choices:time_choices[cache_key]=rng.integers(len(blocks),size=(repeats,len(blocks)))
            choice=time_choices[cache_key]
            value=blocks['sum'].to_numpy()[choice].sum(axis=1)/blocks['count'].to_numpy()[choice].sum(axis=1)
            site_values.setdefault((region,site),[]).append((record,value))
        region_values={}
        for (region,site),records in site_values.items():
            records=sorted(records,key=lambda p:p[0]);cache_key=tuple(r[0].split('_Probe')[0] for r in records)
            if cache_key not in record_choices:record_choices[cache_key]=rng.integers(len(records),size=(repeats,len(records)))
            values=np.stack([r[1] for r in records],axis=1)
            selected=values[np.arange(repeats)[:,None],record_choices[cache_key]].mean(axis=1)
            region_values.setdefault(region,[]).append(selected)
        subject_draws.append(np.mean([np.mean(v,axis=0) for v in region_values.values()],axis=0))
    if len(subject_draws)<2:return [None,None]
    values=np.stack(subject_draws,axis=1);indices=rng.integers(len(subject_draws),size=(repeats,len(subject_draws)))
    return np.quantile(values[np.arange(repeats)[:,None],indices].mean(axis=1),[.025,.975]).tolist()


def semantics_reconstruction_summary(cfg,out):
    refs=json.loads((out/'cohort.json').read_text())['refs'];lookup={r['id']:r for r in refs};groups={}
    for ref in refs:groups.setdefault(ref['key']+'__'+ref['site'],[]).append(ref)
    rows=[]
    for key,group in groups.items():
        record=json.loads((out/'measured'/key/'result.json').read_text());scales={}
        for ref in group:
            path=out/'measured'/key/(f'w{ref["window"]}.npz')
            with np.load(path,allow_pickle=False) as a:
                y=a['target'];sd=a['sd'];scale=np.std(y,axis=0)
                scales[ref['id']]=dict(local_ratio=sd/np.maximum(scale,1e-14),feature_hb_rho=float(np.corrcoef(y[:,1:].T)[0,1]),
                    target_local_sd=scale.tolist())
        for row in record['rows']:
            ref=lookup[row['identity']];info=scales[ref['id']]
            value={k:row.get(k) for k in ('identity','dataset','subject','record','site','region','window','method','parameter_value','status','converged','all_below_half','native_rho','quality_burden','record_evaluation','coordinate','training_key','objective','weighted_data_sse','flow_prior_cost','initial_state_penalty_cost')}
            value.update(repeat_fold=ref['repeat_fold'],repeat_unit=ref['repeat_unit'],eeg_start_s=ref['eeg_start_s'],feature_hb_rho=info['feature_hb_rho'])
            for j,name in enumerate(MODALITIES):
                for metric,field in [('nrmse','nrmse'),('correlation','correlation'),('amplitude_error','amplitude_error'),('peak_time_error_s','peak_time_error_s')]:
                    value[f'{metric}_{name}']=row.get(field,[None]*3)[j] if row.get('converged') else None
                value['local_nrmse_'+name]=value['nrmse_'+name]*info['local_ratio'][j] if value['nrmse_'+name] is not None else None
            rows.append(value)
    frame=pd.DataFrame(rows);frame.to_csv(out/'reconstruction_windows.csv',index=False)
    expected=sum(1+2*(1+(2 if r['repeat_fold'] in (0,1) else 0)+(1 if r['record_evaluation'] else 0)) for r in refs)
    if len(frame)!=expected:raise ValueError(f'planned reconstruction denominator changed: {len(frame)} != {expected}')
    metrics=['converged','all_below_half']+[f'{metric}_{m}' for metric in ('nrmse','correlation','local_nrmse') for m in MODALITIES]
    tables={};subject_tables={}
    for scope,part in [('all_eligible',frame),('common_later_windows',frame[frame.record_evaluation & frame.repeat_fold.isin([0,1])])]:
        subjects=semantics_hierarchical_scores(part,metrics);subjects.to_csv(out/(scope+'_subject_scores.csv'),index=False);subject_tables[scope]=subjects
        aggregate=[]
        for (ds,method),group in subjects.groupby(['dataset','method']):
            original=part[(part.dataset==ds)&(part.method==method)]
            item=dict(dataset=ds,method=method,subjects=len(group),planned_windows=len(original),successes=int(original.converged.sum()),
                pooled_passes=int(original.all_below_half.sum()),success_rate=group.converged.mean(),pass_rate=group.all_below_half.mean(),
                pass_rate_subject_CI=semantics_bootstrap_mean(group.all_below_half,cfg['seed']+231))
            for name in MODALITIES:
                item['nrmse_'+name]=group['nrmse_'+name].mean()
                item['window_median_nrmse_'+name]=original['nrmse_'+name].median()
                item['window_p90_nrmse_'+name]=original['nrmse_'+name].quantile(.9)
                item['correlation_'+name]=group['correlation_'+name].mean()
                item['local_nrmse_'+name]=group['local_nrmse_'+name].mean()
            aggregate.append(item)
        pd.DataFrame(aggregate).to_csv(out/('reconstruction_'+scope+'.csv'),index=False);tables[scope]=aggregate
    paired=[];subjects=subject_tables['common_later_windows'];rng=np.random.default_rng(cfg['seed']+232)
    for ds,part in subjects.groupby('dataset'):
        pivot=part.pivot(index='subject',columns='method',values='all_below_half');family=[]
        for method in pivot.columns:
            if method=='H0_fixed':continue
            valid=pivot[[method,'H0_fixed']].dropna();delta=valid[method]-valid.H0_fixed
            random_sign=rng.choice([-1,1],size=(10000,len(delta)))
            null=(random_sign*delta.to_numpy()).mean(axis=1)
            p=(1+np.sum(abs(null)>=abs(delta.mean())-1e-14))/(len(null)+1)
            original=frame[(frame.dataset==ds)&frame.record_evaluation&frame.repeat_fold.isin([0,1])]
            paired_windows=original[original.method==method].merge(original[original.method=='H0_fixed'][['identity','all_below_half']],on='identity',suffixes=('','_baseline'))
            paired_windows['delta']=paired_windows.all_below_half.astype(float)-paired_windows.all_below_half_baseline.astype(float)
            family.append(dict(dataset=ds,method=method,subjects=len(delta),pass_rate_gain=delta.mean(),
                conditional_subject_CI=semantics_bootstrap_mean(delta,cfg['seed']+233),
                subject_record_block_CI=semantics_block_bootstrap_gain(paired_windows,cfg['seed']+234),signflip_p=p))
        order=sorted(range(len(family)),key=lambda i:family[i]['signflip_p']);last=0.
        for rank,i in enumerate(order):
            last=max(last,min(1.,family[i]['signflip_p']*(len(order)-rank)));family[i]['holm_p']=last
        paired.extend(family)
    pd.DataFrame(paired).to_csv(out/'sharing_paired_comparisons.csv',index=False)
    from scipy.stats import spearmanr
    association=[];fixed=frame[frame.method=='H0_fixed'].copy();fixed['Hb_nrmse']=fixed[['nrmse_HbO','nrmse_HbR']].mean(axis=1)
    record=fixed.groupby(['dataset','subject','site','record']).agg(Hb_nrmse=('Hb_nrmse','mean'),native_rho=('native_rho','first'),feature_hb_rho=('feature_hb_rho','mean'),quality_burden=('quality_burden','first')).reset_index()
    for key,part in record.groupby(['dataset','subject','site']):
        good=part.dropna(subset=['Hb_nrmse','native_rho'])
        if len(good)>=3 and min(good[['Hb_nrmse','native_rho']].std())>1e-12:
            association.append(dict(zip(['dataset','subject','site'],key),records=len(good),native_rho_vs_Hb_error=float(spearmanr(good.native_rho,good.Hb_nrmse).statistic)))
    pd.DataFrame(association).to_csv(out/'within_subject_QC_association.csv',index=False)
    result=dict(planned_regional_windows=len(refs),planned_method_windows=expected,tables=tables,paired=paired,
        numerical_status_counts=frame.status.value_counts().to_dict(),within_subject_association=association,
        aggregation='window -> record -> site -> region -> subject equal weights; common comparison uses later nonoverlapping calibration-held windows',
        uncertainty='paired gains: subject/record/120s temporal block bootstrap; marginal rates: subject bootstrap; both condition on fitted coordinates/parameters without calibration refitting; no calibrated parameter CI',
        selection='descriptive external-fold comparison; no nested learned sharing selector, no confirmatory optimal-level claim')
    write_json(out/'reconstruction_summary.json',result);return result


def semantics_completion_summary(cfg,out):
    plan=json.loads((out/'diagnostic_plan.json').read_text());rows=[]
    for ref in plan['windows']:
        result=json.loads((out/'missing'/ref['id']/'result.json').read_text())
        for row in result['rows']:
            scores=[v for v in row.get('hidden_nrmse',[]) if v is not None]
            value=dict(identity=ref['id'],dataset=ref['dataset'],subject=ref['subject'],record=ref['record'],site=ref['site'],region=ref['region'],
                mode=row['mode'],method=row['method'],arm=row['arm'],status=row.get('status'),converged=row.get('converged',False),
                applicable=not str(row.get('status','')).startswith('not_applicable'),
                error=float(np.mean(scores)) if row.get('converged') and scores else None,
                passed=bool(row.get('converged') and scores and max(scores)<.5))
            for j,m in enumerate(MODALITIES):value['nrmse_'+m]=row.get('hidden_nrmse',[None]*3)[j] if row.get('converged') else None
            rows.append(value)
    frame=pd.DataFrame(rows);frame.to_csv(out/'completion_windows.csv',index=False);summary=[];comparisons=[]
    for key,part in frame.groupby(['dataset','mode','method','arm']):
        available=part[part.applicable];subject=available.groupby('subject')[['error','passed','converged']].mean()
        summary.append(dict(zip(['dataset','mode','method','arm'],key),planned=len(part),applicable=len(available),subjects=len(subject),
            success_rate=subject.converged.mean(),error=subject.error.mean(),pass_rate=subject.passed.mean(),
            error_subject_CI=semantics_bootstrap_mean(subject.error,cfg['seed']+241)))
    for key,part in frame[frame.method.str.startswith('H')].groupby(['dataset','mode','method']):
        for arm in ('training_donor','nonwrapping_shift_12s','spatial_Hb_anchor'):
            real='real_shift_support' if arm=='nonwrapping_shift_12s' else 'real'
            paired=part[part.arm==arm].merge(part[part.arm==real],on=['identity','subject'],suffixes=('_null','_real'))
            if not len(paired):continue
            paired['error_advantage']=paired.error_null-paired.error_real
            paired['pass_advantage']=paired.passed_real.astype(float)-paired.passed_null.astype(float)
            subjects=paired.groupby('subject')[['error_advantage','pass_advantage']].mean()
            comparisons.append(dict(zip(['dataset','mode','method'],key),null=arm,planned_pairs=len(paired),subjects=len(subjects),
                real_error_advantage=subjects.error_advantage.mean(),error_advantage_CI=semantics_bootstrap_mean(subjects.error_advantage,cfg['seed']+242),
                real_pass_advantage=subjects.pass_advantage.mean()))
    pd.DataFrame(summary).to_csv(out/'completion_summary.csv',index=False);pd.DataFrame(comparisons).to_csv(out/'completion_null_comparisons.csv',index=False)
    result=dict(groups=summary,nulls=comparisons,endpoint='processed_feature_completion; not raw_sensor_or_causal_future_prediction',
        interpretation='mean errors conditional on successful fits; pass/success rates retain failed-training and solver failures; unavailable baselines explicit')
    write_json(out/'completion_summary.json',result);return result


def semantics_profile_summary(cfg,out):
    selected={}
    for ref in json.loads((out/'diagnostic_plan.json').read_text())['windows']:selected.setdefault((ref['dataset'],ref['subject']),ref)
    profiles=[];subjects=[];sensitivity=[];trials=[];context=[];transfer=[]
    for ref in selected.values():
        result=json.loads((out/'profiles'/ref['id']/'result.json').read_text());joint=result['joint']
        bounds=[]
        for name,value in zip(['tau','neurovascular_gain'],joint.get('parameter_values') or []):
            lo,hi=np.log(cfg['parameters'][name]['bounds']);v=np.log(value)
            bounds.append('LOWER' if abs(v-lo)<1e-8 else 'UPPER' if abs(v-hi)<1e-8 else 'INTERIOR')
        subjects.append(dict(dataset=ref['dataset'],subject=ref['subject'],identity=ref['id'],converged=joint['converged'],
            tau=joint['parameter_values'][0] if joint.get('parameter_values') is not None else None,
            gain=joint['parameter_values'][1] if joint.get('parameter_values') is not None else None,
            boundary=bounds,objective=joint.get('objective')))
        for row in result['profiles']:
            value=dict(dataset=ref['dataset'],subject=ref['subject'],identity=ref['id'],profiled=row['profiled'],grid=row['value'],
                converged=row['converged'],objective=row.get('objective'),weighted_data_sse=row.get('weighted_data_sse'),
                flow_prior_cost=row.get('flow_prior_cost'),initial_state_penalty_cost=row.get('initial_state_penalty_cost'))
            profiles.append(value)
        for row in result['sensitivity']:
            values=row.get('parameter_values')
            sensitivity.append(dict(dataset=ref['dataset'],subject=ref['subject'],kind=row['kind'],parameter=row.get('parameter','flow_weight'),value=row['value'],
                converged=row['converged'],tau=values[0] if values is not None else None,gain=values[1] if values is not None else None,
                base_tau=joint['parameter_values'][0] if joint.get('parameter_values') is not None else None,
                base_gain=joint['parameter_values'][1] if joint.get('parameter_values') is not None else None,objective=row.get('objective')))
        for row in result['trial']:
            fitted=row['parameter_fit'];final=row.get('completion',{});error=final.get('hidden_nrmse') if row['mode']!='same_trial_full' else final.get('nrmse')
            valid=[v for v in (error or []) if v is not None]
            trials.append(dict(dataset=ref['dataset'],subject=ref['subject'],method=row['mode'],
                converged=bool(fitted.get('converged') and final.get('converged')),error=np.mean(valid) if valid else None,
                passed=bool(fitted.get('converged') and final.get('converged') and valid and max(valid)<.5),
                tau=fitted['parameter_values'][0] if fitted.get('parameter_values') is not None else None,
                gain=fitted['parameter_values'][1] if fitted.get('parameter_values') is not None else None))
        for name,value in result['prefix_baselines'].items():
            error=value.get('hidden_nrmse') if isinstance(value,dict) else value;valid=[v for v in (error or []) if v is not None]
            success=bool(value.get('converged') and valid) if isinstance(value,dict) else bool(valid)
            trials.append(dict(dataset=ref['dataset'],subject=ref['subject'],method=name,converged=success,error=np.mean(valid) if success else None,passed=bool(success and max(valid)<.5)))
        c=json.loads((out/'context'/ref['id']/'result.json').read_text())
        for row in c['rows']:
            context.append(dict(dataset=ref['dataset'],subject=ref['subject'],method=row['method'],seconds=row['seconds'],
                converged=row['converged'],error=np.mean(row['nrmse']) if row.get('converged') and 'nrmse' in row else None,
                passed=row.get('all_below_half',False),central_driver_mean=row.get('central_driver_mean'),
                central_state_mean=row.get('central_state_mean'),status=row['status']))
        t=json.loads((out/'transfer'/ref['id']/'result.json').read_text())
        for row in t['rows']:
            transfer.append(dict(dataset=ref['dataset'],subject=ref['subject'],method=row['method'],tau=row['tau'],converged=row['converged'],
                error=np.mean(row['nrmse']) if row.get('converged') and 'nrmse' in row else None,passed=row.get('all_below_half',False)))
    for name,rows in [('profiles',profiles),('profile_subjects',subjects),('sensitivity',sensitivity),('trial_calibration',trials),('context',context),('transfer',transfer)]:
        pd.DataFrame(rows).to_csv(out/(name+'_metrics.csv'),index=False)
    widths=[];frame=pd.DataFrame(profiles)
    for key,part in frame.groupby(['dataset','subject','profiled']):
        good=part[part.converged].dropna(subset=['objective'])
        if not len(good):continue
        minimum=good.objective.min();near=good[good.objective<=minimum*1.01+1e-12]
        widths.append(dict(zip(['dataset','subject','parameter'],key),successful_grid_points=len(good),planned_grid_points=len(part),
            grid_minimum_value=good.loc[good.objective.idxmin(),'grid'],within_1pct_grid_min=near.grid.min(),within_1pct_grid_max=near.grid.max(),
            within_1pct_grid_ratio=near.grid.max()/near.grid.min(),
            grid_minimum_at_endpoint=good.loc[good.objective.idxmin(),'grid'] in (part.grid.min(),part.grid.max())))
    groups=[]
    for panel,rows,keys in [('prefix',trials,['dataset','method']),('context',context,['dataset','method','seconds']),('transfer',transfer,['dataset','method'])]:
        for key,part in pd.DataFrame(rows).groupby(keys):
            groups.append(dict(panel=panel,**dict(zip(keys,key)),planned=len(part),success_rate=part.converged.mean(),error=part.error.mean(),pass_rate=part.passed.mean()))
    result=dict(subjects=subjects,profile_widths=widths,groups=groups,
        profile_rule='1% grid objective contour is exploratory engineering flatness diagnostic, not a confidence interval; all nuisance quantities reoptimized',
        context='nested common center uses distinct unfiltered Hb diagnostic target; never compare its NRMSE directly against primary filtered target',
        transfer='target observation calibrated on other target subjects; source-fold tau ensemble only; no strict zero-shot beta/amplitude claim')
    write_json(out/'profile_diagnostics_summary.json',result);return result


def semantics_summary(cfg,out):
    required=['prepare','qc_detail','cohort','synthetic','synthetic_feature','synthetic_joint','synthetic_kappa',
              'synthetic_hierarchy','synthetic_stress','observation_math','training','measured','missing','profiles','diagnostics']
    manifests={name:json.loads((out/(name+'_manifest.json')).read_text()) for name in required}
    if any(m['execution']!='completed' for m in manifests.values()):raise ValueError('all planned stages must be terminal before the final summary')
    training=semantics_training_summary(cfg,out);qc=semantics_qc_summary(cfg,out)
    synthetic=semantics_synthetic_diagnostic_summary(cfg,out);measured=semantics_reconstruction_summary(cfg,out)
    completion=semantics_completion_summary(cfg,out);diagnostics=semantics_profile_summary(cfg,out)
    linear_rows=[]
    for path in sorted((out/'linear').glob('*/result.json')):
        for row in json.loads(path.read_text())['rows']:
            linear_rows.append({k:v for k,v in row.items() if k not in ['nrmse','correlation','amplitude_error','peak_time_error_s','trough_time_error_s']}|
                {f'nrmse_{m}':row['nrmse'][i] for i,m in enumerate(MODALITIES)})
    linear_frame=pd.DataFrame(linear_rows);linear_frame.to_csv(out/'linear_reference_windows.csv',index=False)
    linear_subjects=semantics_hierarchical_scores(linear_frame,['all_below_half','physical_valid','small_signal_valid','effective_df']+['nrmse_'+m for m in MODALITIES])
    linear_groups=linear_subjects.groupby(['dataset','method']).mean(numeric_only=True).reset_index().to_dict('records')
    write_json(out/'linear_reference_summary.json',dict(groups=linear_groups,planned=len(linear_frame),interpretation='linear residual screen with separate physical and small-signal validity; objective differs from nonlinear flow-regularized SSM'))
    inventory=json.loads((out/'inventory.json').read_text());refs=json.loads((out/'cohort.json').read_text())['refs']
    reconstruction_pass={}
    for ds in sorted({r['dataset'] for r in refs}):
        options=[r for r in measured['tables']['all_eligible'] if r['dataset']==ds]
        reconstruction_pass[ds]=[r['method'] for r in options if r['success_rate']>=.95 and r['pass_rate']>=.8]
    stable=[r for r in training['repeatability'] if r['hierarchy']=='H2_subject' and r['median_pair_difference']<=.2]
    result=dict(schema='shared_driver_physiology_semantics_summary_v1',experiment_id=cfg['experiment_id'],execution='completed',
        completed_at=datetime.now(timezone.utc).isoformat(),source_root=str(CODE_ROOT),
        scope=dict(native_subjects=inventory['subject_counts'],native_records=inventory['record_counts'],
            joint_subjects={ds:len({r['subject'] for r in refs if r['dataset']==ds}) for ds in sorted({r['dataset'] for r in refs})},regional_windows=len(refs)),
        synthetic={name:json.loads((out/(name+'_summary.json')).read_text()) for name in ['synthetic','synthetic_feature','synthetic_joint','synthetic_kappa']},
        source_files=dict(qc='native_qc_summary.json',training='training_summary.json',reconstruction='reconstruction_summary.json',
            completion='completion_summary.json',profiles='profile_diagnostics_summary.json',stress_hierarchy='synthetic_diagnostics_summary.json',observation='observation_math.json',linear='linear_reference_summary.json'),
        verdict=dict(reconstruction_gate_methods=reconstruction_pass,stable_H2_median_screen_groups=stable,
            parameter_claim='conditional_model_time_parameter_and_effective_gain_only; no absolute_individual_physiology_release',
            source_claim='canonical_observation_combinations_and_model_residual; no_unique_cortical_systemic_or_metabolic_separation'),
        report_format='pptx',limitations=['REFED cross-modal geometry unavailable; Hb QC only',
            'Full observation validation refits its driver/initial state; reconstruction not prediction',
            'Missing/prefix tasks at processed-feature level, not raw causal forecasting',
            'Frozen feature gauges and calibration uncertainty not refitted in bootstrap',
            'Sharing policy comparisons descriptive; no nested inner-selected one-SE policy evaluated',
            'H2s shrinkage fixed from search bounds, not tuned on inner folds; H3 uses three calibration windows vs up to eighteen for H1/H2',
            'Native-to-window-local processing changes target relative to prior whole-record cache reports',
            'Data covariance and QC flags do not by themselves identify corruption or a physiological source'])
    write_json(out/'summary.json',result);return result


def physiology_semantics_main(args):
    cfg = semantics_config(args.config)
    if args.check_only:
        specs = semantics_synthetic_specs(cfg)
        semantics_parameters(cfg).validate()
        print(json.dumps(dict(status='passed', synthetic_panels=len(specs),
                              measured_arrays_read=0, report_format='pptx')))
        return
    if args.run_dir is None or not 1 <= args.workers <= cfg['resources']['max_workers']:
        raise ValueError('explicit run-dir and bounded workers required')
    out = args.run_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    stage = args.semantics_stage
    lock = (out/(stage+'_controller.lock')).open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    resolved = out/'resolved_config.yaml'
    if resolved.exists() and yaml.safe_load(resolved.read_text()) != cfg:
        raise ValueError('cannot change resolved campaign contract')
    if not resolved.exists():
        resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
    stage = args.semantics_stage
    manifest_path = out/(stage+'_manifest.json')
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous.get('execution')=='completed':raise ValueError('completed stage evidence is immutable')
        stamp=previous['started_at'].replace(':','').replace('+','_')
        retained=out/(stage+'_manifest_attempt_'+stamp+'.json')
        if not retained.exists():write_json(retained,previous)
    manifest = dict(experiment_id=cfg['experiment_id'], stage=stage, execution='running',
        started_at=datetime.now(timezone.utc).isoformat(), source_root=str(CODE_ROOT),
        project_root=str(args.project_root.resolve()), supervisor=os.environ.get('SSM_SYSTEMD_UNIT'),
        controller_pid=os.getpid(), workers=args.workers, pilot=args.pilot)
    write_json(manifest_path, manifest)
    if stage == 'summarize':
        manifest.update(execution='completed',summary=semantics_summary(cfg,out))
    elif stage == 'diagnostics':
        if json.loads((out/'training_manifest.json').read_text())['execution']!='completed':
            raise ValueError('terminal training required for context and calibrated transfer')
        plan=json.loads((out/'diagnostic_plan.json').read_text());selected={}
        for ref in plan['windows']:selected.setdefault((ref['dataset'],ref['subject']),ref)
        failures=semantics_parallel_stage(cfg,out,'context',semantics_context_worker,
            [(cfg,str(out),str(args.project_root.resolve()),ref) for ref in selected.values()],args.workers,lambda p:p[3]['id'])
        failures.extend(semantics_parallel_stage(cfg,out,'transfer',semantics_transfer_worker,
            [(cfg,str(out),ref) for ref in selected.values()],args.workers,lambda p:p[2]['id']))
        groups={}
        for ref in json.loads((out/'cohort.json').read_text())['refs']:
            groups.setdefault((ref['dataset'],ref['site'],ref['subject_fold']),[]).append(ref)
        failures.extend(semantics_parallel_stage(cfg,out,'linear',semantics_linear_worker,
            [(cfg,str(out),refs) for refs in groups.values()],args.workers,lambda p:f'{p[2][0]["dataset"]}__{p[2][0]["site"]}__outer{p[2][0]["subject_fold"]}'))
        manifest.update(execution='failed' if failures else 'completed',failures=failures,expected_subjects=len(selected))
    elif stage == 'profiles':
        joint=json.loads((out/'synthetic_joint_summary.json').read_text())
        if not joint['passed']:
            manifest.update(execution='completed',scientific_result='not_expanded_failed_joint_recovery')
        else:
            plan=json.loads((out/'diagnostic_plan.json').read_text());selected={}
            for ref in plan['windows']:selected.setdefault((ref['dataset'],ref['subject']),ref)
            payloads=[(cfg,str(out),ref) for ref in selected.values()]
            failures=semantics_parallel_stage(cfg,out,stage,semantics_profile_worker,payloads,args.workers,lambda p:p[2]['id'])
            manifest.update(execution='failed' if failures else 'completed',failures=failures,expected_profiles=len(payloads))
    elif stage == 'synthetic_kappa':
        if not json.loads((out/'synthetic_joint_summary.json').read_text())['passed']:
            raise ValueError('joint tau/gain recovery required before kappa expansion')
        specs=[dict(tau=2.,gain=1.,kappa=kappa,spectrum=spectrum,repeat=repeat,initial=initial,
                    coordinate='processed',key=f'k{kappa:g}_{spectrum}_r{repeat}_{initial}')
               for kappa in cfg['parameters']['kappa']['starts'] for spectrum in cfg['synthetic']['spectra']
               for repeat in range(cfg['synthetic']['repeats']) for initial in cfg['synthetic']['initial_conditions']]
        failures=semantics_parallel_stage(cfg,out,stage,semantics_kappa_worker,
            [(cfg,str(out),spec) for spec in specs],args.workers,lambda p:p[2]['key'])
        results=[json.loads((out/'synthetic_kappa'/s['key']/'result.json').read_text()) for s in specs
                 if (out/'synthetic_kappa'/s['key']/'result.json').exists()]
        errors=[r['relative_error'] if r['converged'] and r['relative_error'] is not None else float('inf') for r in results]
        median,p90=semantics_error_quantiles(errors) if errors else (np.inf,np.inf)
        success=sum(r['converged'] for r in results)/len(specs)
        passed=len(results)==len(specs) and success>=cfg['gates']['optimizer_success_min'] and median<=.1 and p90<=.25
        summary=dict(expected=len(specs),terminal=len(results),median_relative_error=median,p90_relative_error=p90,
                     success_rate=success,passed=bool(passed))
        write_json(out/'synthetic_kappa_summary.json',summary)
        manifest.update(execution='failed' if failures else 'completed',summary=summary,failures=failures)
    elif stage == 'qc_detail':
        specs=json.loads((out/'inventory.json').read_text())['records']
        failures=semantics_parallel_stage(cfg,out,stage,semantics_qc_detail_worker,
            [(cfg,str(out),str(args.project_root.resolve()),spec) for spec in specs],args.workers,lambda p:p[3]['key'])
        manifest.update(execution='failed' if failures else 'completed',failures=failures,expected_records=len(specs))
    elif stage == 'synthetic_hierarchy':
        screen=json.loads((out/'synthetic_feature_summary.json').read_text())
        if not screen['parameters']['tau']['passed']:
            manifest.update(execution='completed',scientific_result='not_expanded_failed_tau_recovery')
        else:
            specs=[dict(mechanism=mechanism,repeat=repeat,key=f'{mechanism}__r{repeat}')
                for mechanism in ('global','dataset','subject','record','trial','eeg_gain_only','hb_gain_only','noise_only')
                for repeat in range(cfg['synthetic']['hierarchy_repeats'])]
            write_json(out/'synthetic_hierarchy_plan.json',dict(specs=specs,cohort_shape=cfg['synthetic']['hierarchy_shape'],
                parameter='tau_only_minimal_time_parameter; other_physiology_fixed',
                validation='third_record_last_three_windows; H4 is same-target descriptive only'))
            failures=semantics_parallel_stage(cfg,out,stage,semantics_hierarchy_worker,
                [(cfg,str(out),spec) for spec in specs],args.workers,lambda p:p[2]['key'])
            manifest.update(execution='failed' if failures else 'completed',failures=failures,expected_cohorts=len(specs))
    elif stage == 'observation_math':
        manifest.update(execution='completed',summary=semantics_observation_math(cfg,out))
    elif stage == 'synthetic_stress':
        specs=[]
        for stress in ['control']+cfg['synthetic']['stress']:
            for spectrum in cfg['synthetic']['spectra']:
                for repeat in range(cfg['synthetic']['repeats']):
                    specs.append(dict(tau=2.,gain=1.,spectrum=spectrum,repeat=repeat,initial='nonrest',
                        coordinate='processed',stress=stress,key=f'{stress}__{spectrum}__r{repeat}'))
        write_json(out/'synthetic_stress_plan.json',dict(specs=specs,interpretation='known observation faults; model structure unchanged'))
        failures=semantics_parallel_stage(cfg,out,stage,semantics_synthetic_worker,
            [(cfg,str(out),spec) for spec in specs],args.workers,lambda p:p[2]['key'])
        manifest.update(execution='failed' if failures else 'completed',failures=failures,expected_panels=len(specs))
    elif stage == 'missing':
        if json.loads((out/'training_manifest.json').read_text())['execution']!='completed':
            raise ValueError('terminal training required for independent completion')
        plan=json.loads((out/'diagnostic_plan.json').read_text())
        payloads=[(cfg,str(out),ref) for ref in plan['windows']]
        failures=semantics_parallel_stage(cfg,out,stage,semantics_missing_worker,payloads,args.workers,lambda p:p[2]['id'])
        manifest.update(execution='failed' if failures else 'completed',failures=failures,expected_windows=len(payloads))
    elif stage == 'synthetic_joint':
        screen=json.loads((out/'synthetic_feature_summary.json').read_text())
        if not all(v['passed'] for v in screen['parameters'].values()):
            manifest.update(execution='completed',scientific_result='not_expanded_failed_scalar_recovery')
        else:
            specs=semantics_synthetic_specs(cfg)
            if args.pilot:specs=specs[:4]
            payloads=[(cfg,str(out),spec) for spec in specs]
            failures=semantics_parallel_stage(cfg,out,stage,semantics_joint_worker,payloads,args.workers,lambda p:p[2]['key'])
            manifest.update(execution='failed' if failures else 'completed',failures=failures,
                            summary=semantics_joint_summary(cfg,out,specs))
    elif stage == 'training':
        refs=json.loads((out/'cohort.json').read_text())['refs']
        plan=semantics_training_plan(cfg,out,refs)
        failures=[]
        for label,jobs in [('population',[j for j in plan if j['hierarchy']=='H1_dataset']),
                           ('individual',[j for j in plan if j['hierarchy']!='H1_dataset'])]:
            if args.pilot:jobs=jobs[:min(4,len(jobs))]
            payloads=[(cfg,str(out),job) for job in jobs]
            failures.extend(semantics_parallel_stage(cfg,out,stage+'_'+label,semantics_training_worker,
                payloads,args.workers,lambda p:p[2]['key']))
        manifest.update(execution='failed' if failures else 'completed',failures=failures,expected_jobs=len(plan))
    elif stage in ('fixed','measured'):
        if stage=='measured' and json.loads((out/'training_manifest.json').read_text())['execution']!='completed':
            raise ValueError('terminal training phase required')
        refs=json.loads((out/'cohort.json').read_text())['refs']
        qualified=json.loads((out/'training_plan.json').read_text())['qualified_parameters'] if stage=='measured' else []
        groups={}
        for ref in refs:groups.setdefault((ref['key'],ref['site']),[]).append(ref)
        payloads=[(cfg,str(out),group,qualified,stage) for group in groups.values()]
        if args.pilot:payloads=payloads[:4]
        failures=semantics_parallel_stage(cfg,out,stage,semantics_evaluation_worker,payloads,args.workers,
            lambda p:p[2][0]['key']+'__'+p[2][0]['site'])
        manifest.update(execution='failed' if failures else 'completed',failures=failures,expected_groups=len(payloads))
    elif stage == 'cohort':
        preparation=json.loads((out/'prepare_manifest.json').read_text())
        if preparation['execution']!='completed':raise ValueError('completed preparation required')
        refs,coordinates=semantics_cohort(cfg,out)
        manifest.update(execution='completed',windows=len(refs),coordinates=len(coordinates))
    elif stage == 'inventory':
        manifest['summary'] = semantics_inventory(cfg, args.project_root, out)['record_counts']
        manifest['execution'] = 'completed'
    elif stage == 'prepare':
        inventory = json.loads((out/'inventory.json').read_text())
        specs = inventory['records']
        if args.pilot:
            specs = [next(r for r in specs if r['dataset']==ds) for ds in cfg['data']['datasets']]
        payloads = [(cfg, str(out), str(args.project_root.resolve()), spec) for spec in specs]
        started = time.monotonic(); failed = []
        for k, (payload, result, error) in enumerate(bounded_nonlinear_work(
                semantics_prepare_worker, payloads, args.workers, args.workers*2), 1):
            if error:
                failed.append(payload[3]['key'])
                write_json(out/'prepared'/payload[3]['key']/'failure.json', dict(error=error))
            progress = dict(stage=stage, completed=k, expected=len(specs), last_key=payload[3]['key'],
                error=error, elapsed_s=time.monotonic()-started,
                estimated_remaining_s=(time.monotonic()-started)*(len(specs)-k)/k)
            write_json(out/'prepare_progress.json', progress); print(json.dumps(progress), flush=True)
        manifest.update(execution='failed' if failed else 'completed', failed_records=failed,
                        expected_records=len(specs))
    elif stage in ('synthetic','synthetic_feature'):
        specs = semantics_synthetic_specs(cfg)
        if stage == 'synthetic_feature':
            specs = [dict(s,coordinate='processed') for s in specs]
        if args.pilot:
            specs = [specs[i] for i in (0, 3, 16, 21)]
        payloads = [(cfg, str(out), spec) for spec in specs]
        started = time.monotonic()
        for k, (payload, result, error) in enumerate(bounded_nonlinear_work(
                semantics_synthetic_worker, payloads, args.workers, args.workers*2), 1):
            if error:
                write_json(out/stage/payload[2]['key']/'failure.json', dict(error=error))
            progress = dict(completed=k, expected=len(specs), last_key=payload[2]['key'],
                error=error, elapsed_s=time.monotonic()-started,
                estimated_remaining_s=(time.monotonic()-started)*(len(specs)-k)/k)
            write_json(out/'progress.json', progress)
            print(json.dumps(progress), flush=True)
        summary = semantics_synthetic_summary(cfg, out, specs, stage)
        manifest['summary'] = summary
        manifest['execution'] = 'completed' if summary['terminal_panels'] == len(specs) else 'failed'
    else:
        raise ValueError('unknown physiology semantics stage')
    manifest['ended_at'] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)


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
    ap.add_argument('--conditional-step-control',action='store_true')
    ap.add_argument('--waveform-diagnostic',action='store_true')
    ap.add_argument('--physiology-semantics',action='store_true')
    ap.add_argument('--semantics-stage',choices=['synthetic','synthetic_feature','synthetic_joint','synthetic_kappa','synthetic_stress','synthetic_hierarchy','observation_math','inventory','prepare','qc_detail','cohort','training','fixed','measured','missing','profiles','diagnostics','summarize'],default='synthetic')
    ap.add_argument('--phase',choices=['all','synthetic','measured'],default='all')
    args=ap.parse_args()
    if args.physiology_semantics:
        if any((args.waveform_diagnostic, args.conditional_step_control,
                args.conditional_optical_gain_fit, args.fixed_roi_optical_fit,
                args.fixed_roi_initial_fit, args.fixed_roi_flow_fit, args.fixed_roi_tau_fit,
                args.gain_prior_fit, args.nonlinear_fit, args.replay_of is not None)):
            ap.error('physiology-semantics is a separate versioned contract')
        try:
            return physiology_semantics_main(args)
        except Exception as exc:
            if args.run_dir is not None:
                path=args.run_dir.resolve()/(args.semantics_stage+'_manifest.json')
                if path.exists():
                    owned=json.loads(path.read_text())
                    if owned.get('execution')=='running' and owned.get('controller_pid')==os.getpid():
                        owned.update(execution='failed',error=repr(exc),traceback=traceback.format_exc(),
                            ended_at=datetime.now(timezone.utc).isoformat())
                        write_json(path,owned)
            raise
    if args.waveform_diagnostic:
        if any((args.conditional_step_control,args.conditional_optical_gain_fit,args.fixed_roi_optical_fit,args.fixed_roi_initial_fit,args.fixed_roi_flow_fit,args.fixed_roi_tau_fit,args.gain_prior_fit,args.nonlinear_fit,args.replay_of is not None)):
            ap.error('waveform-diagnostic cannot combine with model modes')
        return waveform_diagnostic_main(args)
    if args.conditional_step_control:
        if any((args.conditional_optical_gain_fit,args.fixed_roi_optical_fit,args.fixed_roi_initial_fit,args.fixed_roi_flow_fit,args.fixed_roi_tau_fit,args.gain_prior_fit,args.nonlinear_fit,args.replay_of is not None)):
            ap.error('conditional-step-control cannot combine with model modes')
        return conditional_step_control_main(args)
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
