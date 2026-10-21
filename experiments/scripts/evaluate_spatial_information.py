#!/usr/bin/env python3
"""Retained public spatial derivatives and new matched synthetic controls.

No original dataset/cache reader is imported. All measured reads select exact
completed-run identities. New fits see masked observations before assembly.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
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
from src.inference.observation_baselines import spatial_observation_design, transform_spatial_coordinates
from src.inference.shared_driver_spatial import (forward_spatial_shared_driver,
    fit_spatial_shared_driver, fit_tied_spatial_shared_driver,
    collapse_tied_spatial_design, temporal_mode_basis)
from src.inference.source_observability import (weighted_jacobian, information_decomposition,
    nuisance_profile_information, pack_spatial_linearization)
from src.inference.spatial_information import (panel_mask, legacy_objective_weights,
    weak_direction_lift, generic_equal_rank_readout)
from src.inference.t3a_balloon_robust_ssm import BalloonParameters, BalloonFixedParameters, BalloonFreeParameters
DEFAULT = ROOT/'experiments/configs/physiology_semantic_tokenizer/spatial_information_v1.yaml'
ARTIFACT_ROOT = ROOT/'experiments/runs/physiology_semantic_tokenizer/spatial_information'


def serial(value):
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [serial(v) for v in value]
    if isinstance(value, np.ndarray): return serial(value.tolist())
    if isinstance(value, (np.bool_, np.integer)): return value.item()
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    return value


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2)+'\n'); tmp.replace(path)


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != serial(value): raise ValueError(f'frozen artifact changed: {path}')
    else: write(path, value)


def token(identity):
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


def cfg_read(path=DEFAULT):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get('schema') != 'spatial_information_v1' or cfg.get('experiment_id') != 'SPATIAL-INFORMATION-v1':
        raise ValueError('wrong suite identity')
    if cfg['sources']['protected_data'] != 'forbidden' or cfg['sources']['raw_or_cache_reads'] != 'forbidden':
        raise ValueError('retained public reads only')
    if cfg['sources']['measured_partition'] != 'evaluation': raise ValueError('frozen evaluation partition required')
    for field in ('measured_run', 'synthetic_run'):
        source = (ROOT/cfg['sources'][field]).resolve()
        allowed = ROOT/'experiments/runs/physiology_semantic_tokenizer/shared_driver_spatial'
        if not source.is_relative_to(allowed.resolve()): raise ValueError('source outside explicit spatial evidence owner')
    if cfg['information']['method'] != 'SSM_fixed1' or cfg['information']['mode'] != 'heldout_pair':
        raise ValueError('common fixed-weight hidden-pair analysis required')
    if not 0 < cfg['information']['rcond'] < 1 or cfg['statistics']['failure_penalty_nmse'] <= 0:
        raise ValueError('invalid numerical/statistical controls')
    if cfg['synthetic']['models'] != ['free_K2', 'tied_K2']:
        raise ValueError('exact nested free/tied comparison required')
    if cfg['synthetic']['controls'] != ['true', 'sham', 'generic_equal_rank']:
        raise ValueError('missing frozen geometry control')
    if cfg['solver']['validation_substeps'] <= cfg['solver']['substeps']:
        raise ValueError('finer integration verification required')
    return cfg


def parameters(source_cfg, **changes):
    fixed = dict(source_cfg['fixed']); tau, kappa = fixed.pop('tau'), fixed.pop('kappa')
    return BalloonParameters(fixed=BalloonFixedParameters(**fixed),
        free=BalloonFreeParameters(tau=changes.get('tau', tau), kappa=changes.get('kappa', kappa)))


def load_npz(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}


def source_context(cfg):
    measured = ROOT/cfg['sources']['measured_run']
    synthetic = ROOT/cfg['sources']['synthetic_run']
    for source in (measured, synthetic):
        summary = read(source/'summary.json')
        if summary['execution'] != 'completed' or summary['planned'] != summary['terminal']:
            raise ValueError(f'source is incomplete: {source}')
        if read(source/'verification.json')['status'] != 'passed': raise ValueError('source not independently verified')
    return measured, synthetic, read(measured/'resolved_config.json'), read(measured/'plan.json')


def select_representatives(plan, measured, count):
    candidates = []
    for spec in plan['cells']:
        if (spec['arm'], spec['method'], spec['mode'], spec['pairing']) != (
                'K2_true_H4_E12', 'SSM_fixed1', 'heldout_pair', 'paired'): continue
        cell = read(measured/'cells'/(spec['id']+'.json'))
        if cell.get('success') and cell.get('nmse') is not None:
            candidates.append((cell['nmse'], spec))
    candidates.sort(key=lambda x: (x[0], x[1]['id']))
    chosen, used = [], set()
    for quantile in np.linspace(0, 1, count):
        target = quantile*(len(candidates)-1)
        indices = sorted(range(len(candidates)), key=lambda i: abs(i-target))
        for index in indices:
            nmse, spec = candidates[index]
            if spec['subject'] not in used:
                chosen.append(dict(spec, retained_nmse=nmse, selection_quantile=float(quantile)))
                used.add(spec['subject']); break
    return chosen


def build_plan(cfg, *, pilot=False):
    measured, synthetic, source_cfg, plan = source_context(cfg)
    records = {r['id']: r for r in plan['rows'] if r['split']=='evaluation'}
    identities = sorted(records)
    if pilot: identities = identities[:2]
    info = []
    for spec in plan['cells']:
        if (spec['arm'] in cfg['information']['arms'] and spec['method']=='SSM_fixed1'
                and spec['mode']=='heldout_pair' and spec['pairing']=='paired'
                and spec['record_id'] in identities):
            info.append(dict(kind='information', source_cell=spec, row=records[spec['record_id']]))
    reps = select_representatives(plan, measured, 1 if pilot else cfg['nonlinear']['representatives'])
    tasks = info.copy()
    for representative in reps:
        for name in cfg['nonlinear']['profile_parameters']:
            for multiplier in ([.85, 1., 1.15] if pilot else cfg['nonlinear']['multipliers']):
                tasks.append(dict(kind='profile', representative=representative,
                    row=records[representative['record_id']], parameter=name, multiplier=multiplier))
    for scenario in cfg['synthetic']['scenarios']:
        for seed in cfg['synthetic']['seeds'][:1] if pilot else cfg['synthetic']['seeds']:
            for control in cfg['synthetic']['controls']:
                panels = cfg['synthetic']['panels'] if control=='true' else cfg['synthetic']['panels'][-1:]
                for panel in panels:
                    for model in cfg['synthetic']['models']:
                        tasks.append(dict(kind='synthetic', scenario=scenario, seed=seed,
                            control=control, panel=panel, model=model))
    for task in tasks:
        task['id'] = token(json.dumps(task, sort_keys=True))
    return dict(schema='spatial_information_plan_v1', pilot=pilot, tasks=tasks,
        planned=len(tasks), counts={kind: sum(t['kind']==kind for t in tasks)
            for kind in ('information','profile','synthetic')},
        measured_records=len(identities), measured_subjects=len({records[x]['subject'] for x in identities}),
        representatives=reps, sources=cfg['sources'], source_experiment_id=source_cfg['experiment_id'],
        selection='metadata_frozen_source_plan_all_evaluation_windows; representative_error_selection_descriptive_only')


def measured_arrays(cfg, task):
    source = ROOT/cfg['sources']['measured_run']; spec = task.get('source_cell', task.get('representative'))
    cell = read(source/'cells'/(spec['id']+'.json'))
    prepared = source/'prepared'/(token(task['row']['id'])+'.npz')
    meta = read(prepared.with_suffix('.json'))
    if meta['status'] != 'completed' or meta['row'] != task['row']: raise ValueError('prepared identity mismatch')
    record = load_npz(prepared)
    arrays = load_npz(source/'cells'/(spec['id']+'.npz')) if (source/'cells'/(spec['id']+'.npz')).exists() else None
    calibration = read(source/'calibration'/(spec['arm']+'.json'))
    coordinate = read(source/'coordinate.json'); source_cfg = read(source/'resolved_config.json')
    loading = np.asarray(calibration['readout']['eeg_loadings'])
    args = dict(eeg_loadings=loading, hb_loadings=np.asarray(calibration['design']['hb_loadings']),
        temporal_basis=temporal_mode_basis(source_cfg['tensor']['steps'],source_cfg['tensor']['temporal_modes']),
        hb_gains=np.ones(4), eeg_operator=record['eeg_operator'], hb_operator=record['hb_operator'],
        substeps=cfg['solver']['substeps'], numerical_backend=cfg['solver']['numerical_backend'])
    return cell, arrays, record, coordinate, source_cfg, args


def compact_information(value):
    omitted = ('data_information','prior_information','total_directions','projected_jacobian')
    return {k:v for k,v in value.items() if k not in omitted}


def linear_information(cfg, coefficients, initial, param, dt, sd, args):
    t, f, j = len(args['temporal_basis']), len(args['eeg_loadings']), len(args['hb_loadings'])
    mask = panel_mask(t,f,j,observed_eeg=f,observed_hb=j)
    s = cfg['solver']; rcond = cfg['information']['rcond']
    linear = pack_spatial_linearization(coefficients, initial, param, dt, sd=sd, visible=mask,
        finite_difference_step=cfg['information']['physiology_log_difference'],
        **{k:s[k] for k in ('driver_curvature_weight','initial_weight','flow_prior_weight','flow_log_sd')}, **args)
    jac = linear['observation_jacobian']; n = jac.shape[1]-2
    panels, weighted = [], {}
    for panel in cfg['information']['panels']:
        name = f'H{panel["hb_pairs"]}_E{panel["eeg_channels"]}'
        visible = panel_mask(t,f,j, observed_eeg=panel['eeg_channels']*5,observed_hb=panel['hb_pairs'])
        spectra = {}
        for weighting in cfg['information']['weighting']:
            scale = np.ones_like(sd) if weighting=='raw' else sd
            factors = legacy_objective_weights(visible,f) if weighting=='legacy_modality_mean_objective' else None
            data = weighted_jacobian(jac, scale, visible, factors)
            fixed = information_decomposition(data[:,:n], linear['prior_jacobian'][:,:n], rcond=rcond)
            # Physics information explicitly profiles all fixed-coordinate nuisance columns.
            profile = nuisance_profile_information(data, [n,n+1], rcond=rcond)
            spectra[weighting] = dict(fixed_physiology=compact_information(fixed),
                physiology_profile=compact_information(profile), visible_rows=int(visible.sum()))
            if weighting=='training_sd_independent_rows': weighted[name]=data[:,:n]
        panels.append(dict(panel=name, **spectra))
    full = weighted['H4_E12']
    lifts = {name:weak_direction_lift(data, full,
        directions=min(cfg['information']['weak_directions'],n),rcond=rcond)
        for name,data in weighted.items() if name!='H4_E12'}
    return dict(panels=panels, weak_lifts=lifts, fixed_unknowns=n, expanded_physiology_unknowns=n+2,
        parameter_names=linear['parameter_names'], scales='native_fixed_coordinate_no_column_rescaling',
        covariance='diagonal_training_SD_scale_assumption_not_estimated_noise_covariance'), linear


def nonlinear_weak_checks(cfg, result, linear, coefficients, initial, param, dt, sd, args):
    t, c = linear['forward']['prediction'].shape; f = len(args['eeg_loadings'])
    base = linear['forward']['prediction']; n = coefficients.size+initial.size
    checks=[]
    for panel, lift in result['weak_lifts'].items():
        hp, ep = panel.split('_'); mask = panel_mask(t,f,len(args['hb_loadings']),observed_eeg=int(ep[1:])*5,observed_hb=int(hp[1:]))
        full = panel_mask(t,f,len(args['hb_loadings']),observed_eeg=f,observed_hb=len(args['hb_loadings']))
        direction=np.asarray(lift['old_directions'])[:,-1]
        for step in cfg['nonlinear']['weak_direction_steps']:
            for sign in (-1,1):
                x=np.r_[coefficients.ravel(),initial.ravel()]+sign*step*direction
                row=dict(panel=panel,step=step,sign=sign, old_direction='weakest_old_fixed_physiology_direction')
                try:
                    actual=forward_spatial_shared_driver(x[:coefficients.size].reshape(coefficients.shape),
                        x[coefficients.size:].reshape(initial.shape),param,dt,derivative=False,**args)['prediction']-base
                    predicted=(linear['observation_jacobian'][:,:n]@(sign*step*direction)).reshape(t,c)
                    for name,m in [('old',mask),('added',full&~mask)]:
                        a=(actual/sd)[m]; b=(predicted/sd)[m]
                        row[name+'_actual_RMS_training_SD']=float(np.sqrt(np.mean(a*a)))
                        row[name+'_linear_RMS_training_SD']=float(np.sqrt(np.mean(b*b)))
                        row[name+'_nonlinear_remainder_RMS_training_SD']=float(np.sqrt(np.mean((a-b)**2)))
                    row['status']='completed'
                except Exception as exc: row.update(status='failed_domain',error=str(exc))
                checks.append(row)
    return checks


def information_job(cfg, task, representatives):
    cell,a,record,coordinate,source_cfg,args=measured_arrays(cfg,task)
    result=dict(source_cell=task['source_cell'],subject=task['source_cell']['subject'],
        record_id=task['row']['id'],source_fit_status=cell['status'],source_fit_success=cell['success'],
        source_nmse=cell.get('nmse'),source_eeg_weight=cell.get('eeg_weight'))
    if not cell['success'] or a is None: return dict(result,status='unavailable_source_fit_failure')
    param=parameters(source_cfg);dt=source_cfg['tensor']['dt_s'];sd=np.asarray(coordinate['sd'])
    information,linear=linear_information(cfg,a['coefficients'],a['initial_coordinates'],param,dt,sd,args)
    replay=float(np.max(np.abs(linear['forward']['prediction']-a['prediction'])/sd))
    if replay > 1e-8: raise ValueError(f'fixed fit replay changed: {replay}')
    result.update(status='completed',information=information,replay_max_training_SD=replay)
    if task['row']['id'] in {r['record_id'] for r in representatives} and task['source_cell']['arm']=='K2_true_H4_E12':
        result['nonlinear_weak_checks']=nonlinear_weak_checks(cfg,information,linear,a['coefficients'],a['initial_coordinates'],param,dt,sd,args)
    return result


def fit_and_verify(cfg, eeg, hb, sd, param, dt, args, *, tied=False, visible=None):
    s=cfg['solver'];fit=(fit_tied_spatial_shared_driver if tied else fit_spatial_shared_driver)(
        eeg,hb,param,dt,sd=sd,visible=visible,eeg_weight=1.,
        **{k:s[k] for k in ('driver_curvature_weight','initial_weight','flow_prior_weight','flow_log_sd','max_nfev','gradient_tolerance')},**args)
    if fit['success']:
        replay_args=dict(args,substeps=s['validation_substeps'])
        if tied: replay_args.update(collapse_tied_spatial_design(args['eeg_loadings'],args['hb_loadings']))
        replay=forward_spatial_shared_driver(fit['coefficients'],fit['initial_coordinates'],param,dt,derivative=False,**replay_args)
        error=float(np.max(np.abs(replay['prediction']-fit['prediction'])/sd))
        fit['integration_max_training_SD']=error
        if error>s['integration_tolerance_training_sd']: fit.update(success=False,status='failed_integration')
    return fit


def fit_meta(fit):
    return {k:fit[k] for k in ('status','success','nfev','objective','objective_components','convergence_reason',
        'scaled_gradient_inf_norm','failure_log','integration_max_training_SD','coefficients','initial_coordinates') if k in fit}


def profile_job(cfg,task):
    cell,a,record,coordinate,source_cfg,args=measured_arrays(cfg,task)
    param=parameters(source_cfg,**{task['parameter']:source_cfg['fixed'][task['parameter']]*task['multiplier']})
    fullmask=panel_mask(len(record['eeg']),60,4,observed_eeg=60,observed_hb=4)
    # Mask native values before transforming or assembling any fitter input.
    e,h=transform_spatial_coordinates(np.where(fullmask[:,:60].reshape(record['eeg'].shape),record['eeg'],np.nan),
        np.where(fullmask[:,60:].reshape(record['hb'].shape),record['hb'],np.nan),coordinate)
    sd=np.asarray(coordinate['sd']);fit=fit_and_verify(cfg,e,h,sd,param,source_cfg['tensor']['dt_s'],args,visible=fullmask)
    result=dict(representative=task['representative'],parameter=task['parameter'],multiplier=task['multiplier'],
        value=source_cfg['fixed'][task['parameter']]*task['multiplier'],**fit_meta(fit),
        interpretation='other_physiology_fixed_all_driver_initial_coordinates_refitted')
    if fit['success']:
        target=record['hb'][:,0]/coordinate['hb_scale'];error=(fit['hb_prediction'][20:,0]-target[20:])/sd[60:62]
        result['hidden_nmse']=float(np.mean(error**2));result['hidden_nrmse']=np.sqrt(result['hidden_nmse'])
    return result


def synthetic_case(cfg,source_cfg,scenario,seed):
    s=cfg['synthetic'];rng=np.random.default_rng(cfg['seed']+10007*seed)
    t,m=s['steps'],s['temporal_modes'];basis=temporal_mode_basis(t,m)
    angles=np.linspace(-1.1,1.1,12);ep=np.column_stack((np.sin(angles),np.cos(angles),.2*np.cos(2*angles)))
    hp=np.asarray([[-.8,.6,.1],[-.25,.97,.15],[.35,.94,.2],[.85,.53,.1]])
    design=spatial_observation_design(ep,hp,2,seed=cfg['seed'])
    bands=np.asarray([[1.,-.7],[-.6,1.],[1.2,.8],[.4,-.8],[-.8,.5]])
    load=(design['spatial_eeg'][:,None,:]*bands[None]).reshape(60,2)
    coef=rng.normal(size=(m,2))/np.arange(1,m+1)[:,None]**1.5
    coef*=s['driver_rms']/np.sqrt(np.mean((basis@coef)**2,axis=0))
    if scenario=='identical_sources':coef[:,1]=coef[:,0]
    elif scenario=='near_tied_sources':coef[:,1]=(1-s['near_tied_fraction'])*coef[:,0]+s['near_tied_fraction']*coef[:,1]
    elif scenario!='two_sources':raise ValueError('unknown synthetic scenario')
    operator=np.eye(t)-np.ones((t,1))@np.r_[np.full(s['reference_steps'],1/s['reference_steps']),np.zeros(t-s['reference_steps'])][None]
    args=dict(eeg_loadings=load,hb_loadings=design['hb_loadings'],temporal_basis=basis,eeg_operator=operator,
        hb_operator=operator,substeps=cfg['solver']['substeps'],numerical_backend=cfg['solver']['numerical_backend'])
    truth=forward_spatial_shared_driver(coef,np.zeros((2,5)),parameters(source_cfg),s['dt_s'],derivative=False,
        **dict(args,substeps=cfg['solver']['validation_substeps']))
    noise=s['noise_sd'];e=truth['eeg_prediction']+operator@rng.normal(scale=noise['eeg'],size=(t,60))
    h=truth['hb_prediction']+np.einsum('st,tjc->sjc',operator,rng.normal(size=(t,4,2))*[noise['HbO'],noise['HbR']])
    return dict(eeg=e,hb=h,truth=truth,args=args,coordinates=dict(eeg=ep,hb=hp),bands=bands)


def synthetic_coordinate(cfg,source_cfg):
    rows=[]
    for scenario in cfg['synthetic']['scenarios']:
        for seed in cfg['synthetic']['training_seeds']:
            case=synthetic_case(cfg,source_cfg,scenario,seed)
            rows.append(np.column_stack((case['eeg'],case['hb'].reshape(len(case['eeg']),8))))
    values=np.concatenate(rows)
    return dict(sd=np.maximum(values.std(axis=0,ddof=1),1e-5),training_seeds=cfg['synthetic']['training_seeds'],
        scenarios=cfg['synthetic']['scenarios'],training_windows=len(rows),evaluation_used=False,
        definition='one_common_training_SD_per_feature_all_scenarios_separate_training_seeds_no_refit_by_panel_or_control')


def synthetic_job(cfg,task,coordinate,source_cfg):
    case=synthetic_case(cfg,source_cfg,task['scenario'],task['seed']);args=dict(case['args']);control=task['control']
    if control=='sham':
        des=spatial_observation_design(case['coordinates']['eeg'],case['coordinates']['hb'],2,geometry='sham',seed=cfg['seed']+13)
        args.update(eeg_loadings=(des['spatial_eeg'][:,None,:]*case['bands'][None]).reshape(60,2),hb_loadings=des['hb_loadings'])
    elif control=='generic_equal_rank':
        generic=generic_equal_rank_readout(args['eeg_loadings'],args['hb_loadings'],seed=cfg['seed']+31)
        args.update({k:generic[k] for k in ('eeg_loadings','hb_loadings')})
    c,j=task['panel']['eeg_channels'],task['panel']['hb_pairs'];sd=np.r_[np.asarray(coordinate['sd'])[:c*5],np.asarray(coordinate['sd'])[60:60+2*j]]
    args.update(eeg_loadings=args['eeg_loadings'][:c*5],hb_loadings=args['hb_loadings'][:j])
    mask=panel_mask(len(case['eeg']),c*5,j,observed_eeg=c*5,observed_hb=j)
    e=np.where(mask[:,:c*5],case['eeg'][:,:c*5],np.nan)
    h=np.where(mask[:,c*5:].reshape(len(e),j,2),case['hb'][:,:j],np.nan)
    tied=task['model']=='tied_K2';fit=fit_and_verify(cfg,e,h,sd,parameters(source_cfg),cfg['synthetic']['dt_s'],args,tied=tied,visible=mask)
    result=dict(scenario=task['scenario'],seed=task['seed'],control=control,panel=f'H{j}_E{c}',model=task['model'],
        **fit_meta(fit),eeg_loading_singular_values=np.linalg.svd(args['eeg_loadings'],compute_uv=False),
        hb_loading_singular_values=np.linalg.svd(args['hb_loadings'],compute_uv=False),
        fixed_eeg_weight=1.,readout='known_generator_or_declared_mismatch_no_fit',hidden_input_all_nonfinite=bool(np.isnan(h[:,0]).all()))
    if fit['success']:
        left,right=cfg['synthetic']['score_steps'];truth=case['truth'];pred=fit['hb_prediction'][left:right,0]
        target=case['hb'][left:right,0];noiseless=truth['hb_prediction'][left:right,0]
        result['nmse']=float(np.mean(((pred-target)/sd[c*5:c*5+2])**2))
        result['noiseless_nmse']=float(np.mean(((pred-noiseless)/sd[c*5:c*5+2])**2))
        drivers=np.repeat(fit['drivers'],2,axis=1) if tied else fit['drivers']
        states=np.repeat(fit['states'],2,axis=1) if tied else fit['states']
        result['driver_relative_RMSE']=float(np.sqrt(np.mean((drivers-truth['drivers'])**2)/np.mean(truth['drivers']**2)))
        reference=cfg['synthetic']['reference_steps']
        driver_dc=drivers[:reference].mean(axis=0)-truth['drivers'][:reference].mean(axis=0)
        centered_driver_error=(drivers-truth['drivers'])-driver_dc[None]
        centered_driver_truth=truth['drivers']-truth['drivers'][:reference].mean(axis=0,keepdims=True)
        result['driver_reference_centered_NRMSE']=float(np.sqrt(np.mean(centered_driver_error**2)/np.mean(centered_driver_truth**2)))
        result['driver_reference_DC_error_by_core']=driver_dc
        result['driver_reference_steps']=reference
        state_dc=states[:reference].mean(axis=0)-truth['states'][:reference].mean(axis=0)
        state_error=(states-truth['states'])-state_dc[None]
        centered_state_truth=truth['states']-truth['states'][:reference].mean(axis=0,keepdims=True)
        state_scale=np.sqrt(np.mean(centered_state_truth**2,axis=(0,1)))
        result['state_reference_centered_NRMSE_by_state']=np.divide(np.sqrt(np.mean(state_error**2,axis=(0,1))),state_scale,
            out=np.full_like(state_scale,np.nan),where=state_scale>1e-12)
        result['state_reference_DC_error_by_core_and_state']=state_dc
        # State errors use each physical state's own true trajectory RMS;
        # resting offsets are retained, so this is not variation recovery.
        scale=np.sqrt(np.mean(truth['states']**2,axis=(0,1)))
        result['state_relative_RMSE_by_state']=np.sqrt(np.mean((states-truth['states'])**2,axis=(0,1)))/np.maximum(scale,1e-12)
        result['driver_difference_relative_RMSE']=float(np.sqrt(np.mean(((drivers[:,0]-drivers[:,1])-(truth['drivers'][:,0]-truth['drivers'][:,1]))**2))/cfg['synthetic']['driver_rms'])
    if control=='true' and task['model']=='free_K2' and task['panel']==cfg['synthetic']['panels'][-1]:
        result['truth_information'],_=linear_information(cfg,truth['coefficients'],truth['initial_coordinates'],parameters(source_cfg),cfg['synthetic']['dt_s'],np.asarray(coordinate['sd']),case['args'])
    return result


def pooled(rows, *, unit, penalty):
    groups={};successes={}
    for row in rows:
        key=str(row[unit]);valid=row.get('success') and row.get('status')=='completed' and row.get('nmse') is not None and np.isfinite(row['nmse'])
        groups.setdefault(key,[]).append(float(row['nmse']) if valid else penalty)
        if valid:successes.setdefault(key,[]).append(float(row['nmse']))
    scores={k:float(np.sqrt(np.mean(v))) for k,v in groups.items()}
    return dict(planned=len(rows),succeeded=sum(len(v) for v in successes.values()),failed=len(rows)-sum(len(v) for v in successes.values()),
        independent_units=len(groups),unit_nrmse_with_failure_penalty=scores,
        mean_nrmse_with_failure_penalty=float(np.mean(list(scores.values()))) if scores else None,
        success_subset_mean_nrmse=float(np.mean([np.sqrt(np.mean(v)) for v in successes.values()])) if successes else None,
        failure_penalty_nmse=penalty,pooling='within_unit_mean_NMSE_then_sqrt_then_equal_unit_mean')


def paired(candidate,reference,cfg,*,unit,key_fields):
    a={tuple(str(r[k]) for k in key_fields):r for r in candidate};b={tuple(str(r[k]) for k in key_fields):r for r in reference}
    common=sorted(set(a)&set(b));available=[k for k in common if not a[k].get('status','').startswith('unavailable') and not b[k].get('status','').startswith('unavailable')]
    p=cfg['statistics']['failure_penalty_nmse'];left=pooled([a[k] for k in available],unit=unit,penalty=p);right=pooled([b[k] for k in available],unit=unit,penalty=p)
    units=sorted(set(left['unit_nrmse_with_failure_penalty'])&set(right['unit_nrmse_with_failure_penalty']))
    diff=np.asarray([right['unit_nrmse_with_failure_penalty'][u]-left['unit_nrmse_with_failure_penalty'][u] for u in units])
    interval=None
    if len(diff):
        rng=np.random.default_rng(cfg['seed']);reps=cfg['statistics']['bootstrap_repeats']
        interval=np.quantile(diff[rng.integers(0,len(diff),(reps,len(diff)))].mean(axis=1),[.025,.975])
    return dict(planned_common_cells=len(common),unavailable_pairs=len(common)-len(available),available_pairs=len(available),
        candidate_unmatched=len(set(a)-set(b)),reference_unmatched=len(set(b)-set(a)),candidate=left,reference=right,
        independent_unit=unit,independent_units=len(units),mean_nrmse_improvement=float(diff.mean()) if len(diff) else None,
        interval_95=interval,direction='positive_favors_candidate',bootstrap_repeats=cfg['statistics']['bootstrap_repeats'],
        uncertainty=cfg['statistics']['interval'],numerical_failures_kept_with_penalty=True)


def retained_comparisons(cfg):
    measured,synthetic,_,_=source_context(cfg)
    roots=[('measured',measured)]+[(scenario,synthetic/'scenarios'/scenario) for scenario in read(synthetic/'resolved_config.json')['synthetic']['scenarios']]
    result=[];group_table=[]
    filters=[('true_vs_sham_K2',('K2_true_H4_E12','SSM'),('K2_sham_H4_E12','SSM')),
        ('K2_vs_K1',('K2_true_H4_E12','SSM'),('K1_true_H4_E12','SSM')),
        ('free_vs_exact_tied_fixed1',('K2_true_H4_E12','SSM_fixed1'),('K2_true_H4_E12','SSM_tied_fixed1')),
        ('H4_vs_H2',('K1_true_H4_E12','SSM'),('K1_true_H2_E12','SSM')),
        ('E12_vs_E6',('K1_true_H4_E12','SSM'),('K1_true_H4_E6','SSM')),
        ('K1_vs_same_input_ridge',('K1_true_H4_E12','SSM'),('K1_true_H4_E12','joint_ridge')),
        ('K2_vs_same_input_ridge',('K2_true_H4_E12','SSM'),('K2_true_H4_E12','joint_ridge')),
        ('K2_selected_vs_fixed1',('K2_true_H4_E12','SSM'),('K2_true_H4_E12','SSM_fixed1'))]
    for label,root in roots:
        plan=read(root/'plan.json');groups={};actual=0
        for spec in plan['cells']:
            # Read all terminal records, retaining the complete owning denominator.
            path=root/'cells'/(spec['id']+'.json')
            cell=read(path) if path.exists() else dict(spec,status='missing',success=False)
            actual+=path.exists()
            if spec['mode']=='heldout_pair' and spec['pairing']=='paired':
                groups.setdefault((spec['arm'],spec['method']),[]).append(cell)
        for (arm,method),rows in groups.items():
            group_table.append(dict(source=label,arm=arm,method=method,source_plan_cells=len(plan['cells']),source_terminal_cells=actual,
                **pooled(rows,unit='subject',penalty=cfg['statistics']['failure_penalty_nmse'])))
        for name,aa,bb in filters:
            if aa in groups and bb in groups:result.append(dict(source=label,comparison=name,
                candidate_arm=aa,reference_arm=bb,**paired(groups[aa],groups[bb],cfg,unit='subject',key_fields=['record_id'])))
    return dict(groups=group_table,comparisons=result,source='recomputed_from_all_frozen_plan_terminal_cell_JSON_not_copied_summary',
        source_measured=str(measured.relative_to(ROOT)),source_synthetic=str(synthetic.relative_to(ROOT)),
        old_selection_refitted=False)


def execute_one(payload):
    cfg,task,out,coordinate,source_cfg,representatives=payload;path=Path(out)/'tasks'/(task['id']+'.json')
    if path.exists():return read(path)
    started=time.monotonic()
    try:
        if task['kind']=='information':value=information_job(cfg,task,representatives)
        elif task['kind']=='profile':value=profile_job(cfg,task)
        else:value=synthetic_job(cfg,task,coordinate,source_cfg)
    except Exception as exc:value=dict(status='failed_exception',success=False,error=str(exc),traceback=traceback.format_exc())
    result=dict(task=task,**value,elapsed_seconds=time.monotonic()-started)
    write(path,result);return result


def run_tasks(cfg,plan,out,workers,coordinate,source_cfg):
    unfinished=[t for t in plan['tasks'] if not (out/'tasks'/(t['id']+'.json')).exists()]
    payloads=iter((cfg,t,str(out),coordinate,source_cfg,plan['representatives']) for t in unfinished)
    done=plan['planned']-len(unfinished);started=time.monotonic();counts={}
    def record(result):
        nonlocal done
        done+=1;status=result['status'];counts[status]=counts.get(status,0)+1
        write(out/'progress.json',dict(planned=plan['planned'],terminal=done,current_stage_status_counts=counts,
            elapsed_seconds=time.monotonic()-started,workers=workers,last_task=result['task']['id']))
    if workers==1:
        for payload in payloads:record(execute_one(payload))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending=set();exhausted=False
            while pending or not exhausted:
                while not exhausted and len(pending)<workers*cfg['resources']['in_flight_per_worker']:
                    try:pending.add(pool.submit(execute_one,next(payloads)))
                    except StopIteration:exhausted=True
                if pending:
                    ready,pending=wait(pending,return_when=FIRST_COMPLETED)
                    for future in ready:record(future.result())


def figure_exports(cfg,out,rows,retained):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=out/'figures';figures.mkdir(exist_ok=True)
    paths=[]
    def save(fig,name):
        fig.tight_layout();path=figures/(name+'.png');fig.savefig(path,dpi=240);plt.close(fig);paths.append(str(path.relative_to(out)))
    infos=[r for r in rows if r['task']['kind']=='information' and r['status']=='completed']
    if infos:
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        colors={'H2_E12':'#c87121','H4_E6':'#5963b9','H4_E12':'#12817d'}
        for ri,arm in enumerate(cfg['information']['arms']):
            for ci,weight in enumerate(cfg['information']['weighting']):
                ax=axes[ri,ci]
                for panel in colors:
                    data=[p[weight]['fixed_physiology']['data_singular_values'] for r in infos if r['source_cell']['arm']==arm for p in r['information']['panels'] if p['panel']==panel]
                    if data:ax.semilogy(np.arange(1,len(data[0])+1),np.median(data,axis=0),label=panel,color=colors[panel])
                ax.set_title(arm+'\n'+weight.replace('_',' '),fontsize=9);ax.set_xlabel('Singular direction (strong to weak)');ax.set_ylabel('Median local singular value');ax.legend(fontsize=8)
        save(fig,'fixed_coordinate_singular_spectra')
        # Standalone figures are sized for the final seven-inch report width.
        # Their labels retain their point size when embedded at that width.
        titles={'raw':'Raw observation-coordinate sensitivity',
            'training_sd_independent_rows':'Frozen training-SD sensitivity (independent-row assumption)',
            'legacy_modality_mean_objective':'Legacy modality-mean objective sensitivity'}
        for weight in cfg['information']['weighting']:
            fig,axes=plt.subplots(2,1,figsize=(7,6.5))
            for ax,arm in zip(axes,cfg['information']['arms']):
                for panel,color in colors.items():
                    data=[p[weight]['fixed_physiology']['data_singular_values'] for r in infos
                        if r['source_cell']['arm']==arm for p in r['information']['panels'] if p['panel']==panel]
                    if data:ax.semilogy(np.arange(1,len(data[0])+1),np.median(data,axis=0),label=panel,color=color)
                ax.set_title(arm,fontsize=11);ax.set_xlabel('Singular direction (strong to weak)',fontsize=10)
                ax.set_ylabel('Median local singular value',fontsize=10);ax.tick_params(labelsize=9);ax.legend(fontsize=10)
            fig.suptitle(titles[weight],fontsize=11)
            save(fig,'fixed_coordinate_spectrum_'+weight)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,arm in zip(axes,cfg['information']['arms']):
            for panel,color in list(colors.items())[:2]:
                vals=[r['information']['weak_lifts'][panel]['energy_gain_ratio'] for r in infos if r['source_cell']['arm']==arm]
                if vals:ax.plot(np.arange(1,len(vals[0])+1),np.nanmedian(np.asarray(vals,dtype=float),axis=0),marker='o',label=panel+' to H4_E12',color=color)
            ax.axhline(1,color='gray',ls='--');ax.set_yscale('log');ax.set_title(arm);ax.set_xlabel('Old weak direction (weakest at right)');ax.set_ylabel('Full / old directional energy');ax.legend(fontsize=8)
        save(fig,'old_weak_direction_lift')
    syn=[r for r in rows if r['task']['kind']=='synthetic']
    if syn:
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,scenario in zip(axes,cfg['synthetic']['scenarios']):
            for model,color in [('free_K2','#12817d'),('tied_K2','#b96355')]:
                data=[]
                for panel in cfg['synthetic']['panels']:
                    name=f'H{panel["hb_pairs"]}_E{panel["eeg_channels"]}'
                    subset=[r for r in syn if r.get('scenario')==scenario and r.get('control')=='true' and r.get('model')==model and r.get('panel')==name]
                    data.append(pooled(subset,unit='seed',penalty=cfg['statistics']['failure_penalty_nmse'])['mean_nrmse_with_failure_penalty'])
                if all(x is not None for x in data):ax.plot(range(3),data,'o-',color=color,label=model)
            ax.set_xticks(range(3),['H2 E12','H4 E6','H4 E12']);ax.set_title(scenario.replace('_',' '));ax.set_ylabel('Hidden pair NRMSE (failure penalty kept)');ax.legend(fontsize=8)
        save(fig,'synthetic_hidden_pair_channel_count')
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,scenario in zip(axes,cfg['synthetic']['scenarios']):
            for model,color in [('free_K2','#12817d'),('tied_K2','#b96355')]:
                data=[]
                for control in cfg['synthetic']['controls']:
                    vals=[r['driver_relative_RMSE'] for r in syn if r.get('scenario')==scenario and r.get('control')==control and r.get('model')==model and r.get('panel')=='H4_E12' and r.get('success')]
                    data.append(np.mean(vals) if vals else np.nan)
                ax.plot(range(3),data,'o-',color=color,label=model)
            ax.set_xticks(range(3),['true','sham','generic']);ax.set_title(scenario.replace('_',' '));ax.set_ylabel('Driver relative RMSE (successful fits)');ax.legend(fontsize=8)
        save(fig,'synthetic_geometry_driver_recovery_success_subset')
        fig,axes=plt.subplots(1,2,figsize=(9,4))
        for control in cfg['synthetic']['controls']:
            example=next((r for r in syn if r.get('control')==control and r.get('panel')=='H4_E12' and 'eeg_loading_singular_values' in r),None)
            if example:
                for ax,field in zip(axes,['eeg_loading_singular_values','hb_loading_singular_values']):ax.plot([1,2],example[field],'o-',label=control);ax.set_title(field.replace('_',' '));ax.set_xlabel('Singular direction');ax.legend()
        save(fig,'geometry_control_loading_spectra')
    profiles=[r for r in rows if r['task']['kind']=='profile']
    if profiles:
        representatives=list(dict.fromkeys(r['task']['representative']['id'] for r in profiles))
        fig,axes=plt.subplots(len(representatives),2,figsize=(11,3.3*len(representatives)),squeeze=False)
        for ri,identity in enumerate(representatives):
            for ci,param in enumerate(cfg['nonlinear']['profile_parameters']):
                ax=axes[ri,ci];selected=sorted([r for r in profiles if r['task']['representative']['id']==identity and r['task']['parameter']==param],key=lambda x:x['task']['multiplier'])
                for component,color in [('eeg_data','#5963b9'),('hb_data','#12817d'),('flow_prior','#c87121'),('driver_curvature','#b96355'),('initial_state','#929292')]:
                    ax.plot([r['task']['multiplier'] for r in selected],[r.get('objective_components',{}).get(component,np.nan) for r in selected],'o-',label=component,color=color)
                failed=[r for r in selected if not r.get('success')]
                if failed:ax.scatter([r['task']['multiplier'] for r in failed],[0]*len(failed),marker='x',color='red',label='failed fit')
                ax.set_title(selected[0]['task']['representative']['subject']+' '+param);ax.set_xlabel('Parameter / fixed baseline (others fixed)');ax.set_ylabel('Objective SSE component');ax.legend(fontsize=7)
        save(fig,'nonlinear_conditional_profile_components')
    groups=retained['groups'];fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,source in zip(axes,['measured','two_sources']):
        chosen=[next((r for r in groups if r['source']==source and r['arm']==arm and r['method']==method),None) for arm,method in [('K1_true_H4_E12','SSM'),('K2_true_H4_E12','SSM'),('K2_true_H4_E12','SSM_tied_fixed1'),('K1_true_H4_E12','joint_ridge')]]
        vals=[r['mean_nrmse_with_failure_penalty'] if r else np.nan for r in chosen]
        ax.bar(range(4),vals,color=['#5963b9','#12817d','#b96355','#c87121']);ax.set_xticks(range(4),['K1 selected','K2 selected','K2 tied fixed1','joint ridge'],rotation=15);ax.set_title('Retained '+source);ax.set_ylabel('Subject/generator-unit mean hidden NRMSE')
    save(fig,'retained_hidden_prediction_full_denominator')
    return paths


def summarize(cfg,plan,out):
    rows=[read(out/'tasks'/(t['id']+'.json')) if (out/'tasks'/(t['id']+'.json')).exists() else dict(task=t,status='missing',success=False) for t in plan['tasks']]
    retained=read(out/'retained_comparisons.json');synthetic=[r for r in rows if r['task']['kind']=='synthetic']
    groups=[];comparisons=[]
    for scenario in cfg['synthetic']['scenarios']:
        for control in cfg['synthetic']['controls']:
            for panel in cfg['synthetic']['panels']:
                name=f'H{panel["hb_pairs"]}_E{panel["eeg_channels"]}'
                for model in cfg['synthetic']['models']:
                    selected=[r for r in synthetic if r['task']['scenario']==scenario and r['task']['control']==control and r['task']['panel']==panel and r['task']['model']==model]
                    # Failed exceptions still carry plan identity and unit.
                    selected=[dict(r,seed=r['task']['seed']) for r in selected]
                    if selected:groups.append(dict(scenario=scenario,control=control,panel=name,model=model,**pooled(selected,unit='seed',penalty=cfg['statistics']['failure_penalty_nmse']),
                        success_subset_driver_relative_RMSE=float(np.mean([r['driver_relative_RMSE'] for r in selected if r.get('success')])) if any(r.get('success') for r in selected) else None,
                        success_subset_driver_reference_centered_NRMSE=float(np.mean([r['driver_reference_centered_NRMSE'] for r in selected if r.get('success') and 'driver_reference_centered_NRMSE' in r])) if any(r.get('success') and 'driver_reference_centered_NRMSE' in r for r in selected) else None,
                        success_subset_driver_absolute_DC_error=float(np.mean([np.mean(np.abs(r['driver_reference_DC_error_by_core'])) for r in selected if r.get('success') and 'driver_reference_DC_error_by_core' in r])) if any(r.get('success') and 'driver_reference_DC_error_by_core' in r for r in selected) else None))
    def subset(scenario,control,panel,model):return [dict(r,seed=r['task']['seed']) for r in synthetic if (r['task']['scenario'],r['task']['control'],r['task']['panel'],r['task']['model'])==(scenario,control,panel,model)]
    for scenario in cfg['synthetic']['scenarios']:
        full=cfg['synthetic']['panels'][-1]
        candidates=subset(scenario,'true',full,'free_K2')
        refs=[('free_vs_exact_tied',subset(scenario,'true',full,'tied_K2'))]
        refs += [('true_vs_'+control,subset(scenario,control,full,'free_K2')) for control in ('sham','generic_equal_rank')]
        refs += [('full_vs_H'+str(p['hb_pairs'])+'_E'+str(p['eeg_channels']),subset(scenario,'true',p,'free_K2')) for p in cfg['synthetic']['panels'][:-1]]
        for label,reference in refs:
            comparisons.append(dict(scenario=scenario,comparison=label,**paired(candidates,reference,cfg,unit='seed',key_fields=['seed'])))
    write(out/'synthetic_results.json',dict(groups=groups,comparisons=comparisons))
    write(out/'information_results.json',dict(rows=[r for r in rows if r['task']['kind']=='information']))
    write(out/'profile_results.json',dict(rows=[r for r in rows if r['task']['kind']=='profile']))
    figures=figure_exports(cfg,out,rows,retained)
    result=dict(schema='spatial_information_summary_v1',experiment_id=cfg['experiment_id'],
        execution='completed' if all(r['status']!='missing' for r in rows) else 'incomplete',planned=len(rows),terminal=sum(r['status']!='missing' for r in rows),
        status_counts={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},
        task_counts=plan['counts'],measured_records=plan['measured_records'],measured_subjects=plan['measured_subjects'],
        synthetic_groups=groups,synthetic_comparisons=comparisons,retained_comparisons=retained['comparisons'],
        information_completed=sum(r['task']['kind']=='information' and r['status']=='completed' for r in rows),
        profile_completed=sum(r['task']['kind']=='profile' and r.get('success',False) for r in rows),
        figures=figures,claim_limits=cfg['claim_limits'],scientific_verdict='not_assigned',
        owning_numeric_artifacts=['tasks/*.json','retained_comparisons.json','information_results.json','profile_results.json','synthetic_results.json'])
    write(out/'summary.json',result);return result


def snapshot(out,config_path):
    sources=[Path(__file__),ROOT/'src/inference/spatial_information.py',ROOT/'src/inference/source_observability.py',
        ROOT/'src/inference/shared_driver_spatial.py',ROOT/'src/inference/observation_baselines.py',
        ROOT/'src/inference/shared_driver_rk4.py',ROOT/'src/inference/shared_driver_modes.py',
        ROOT/'src/inference/shared_driver_reconstruction.py',ROOT/'src/inference/t3a_balloon_robust_ssm.py',Path(config_path),ROOT/'tests/test_spatial_information.py']
    entries=[]
    for source in sources:
        relative=source.resolve().relative_to(ROOT);target=out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        data=source.read_bytes()
        if target.exists() and target.read_bytes()!=data:raise ValueError('source snapshot differs; use new versioned output')
        if not target.exists():target.write_bytes(data)
        entries.append(dict(path=str(relative),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    immutable(out/'source_snapshot/files.json',entries)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default=str(DEFAULT));parser.add_argument('--stage',choices=['check','pilot','run','summarize'],default='check');parser.add_argument('--output');parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args(argv);cfg=cfg_read(args.config)
    if args.workers<1:parser.error('workers must be positive')
    if args.stage=='check':
        # Configuration-only check performs no retained-data or measured reads.
        print(json.dumps(dict(schema=cfg['schema'],status='passed',measured_reads=False,artifact_root=cfg['artifact_root'],
            source_paths=cfg['sources'],synthetic_scopes=cfg['synthetic']['scenarios'])));return 0
    if not args.output:parser.error('--output is required')
    out=Path(args.output).resolve()
    if not out.is_relative_to(ARTIFACT_ROOT.resolve()) or out==ARTIFACT_ROOT.resolve():parser.error('output must be a fresh named run inside owning artifact root')
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='summarize':
            plan=read(out/'plan.json');print(json.dumps(serial(summarize(cfg,plan,out))));return 0
        stage=read(out/'plan.json')['pilot'] if (out/'plan.json').exists() else args.stage=='pilot'
        if stage != (args.stage=='pilot'):raise ValueError('pilot/full runs cannot share output')
        immutable(out/'resolved_config.json',cfg);snapshot(out,args.config)
        plan=build_plan(cfg,pilot=args.stage=='pilot');immutable(out/'plan.json',plan)
        source_cfg=read(ROOT/cfg['sources']['measured_run']/'resolved_config.json')
        coordinate=synthetic_coordinate(cfg,source_cfg);immutable(out/'synthetic_training_coordinate.json',coordinate)
        if not (out/'retained_comparisons.json').exists():write(out/'retained_comparisons.json',retained_comparisons(cfg))
        execution=dict(stage=args.stage,status='running',created_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
            workers=args.workers,command=sys.argv,working_directory=str(ROOT),python=sys.executable,
            supervisor_unit=os.environ.get('SYSTEMD_UNIT') or os.environ.get('SYSTEMD_EXEC_PID'),
            invocation_id=os.environ.get('INVOCATION_ID'),library_threads=1,in_flight=2*args.workers,
            affinity=list(os.sched_getaffinity(0)),cpu_count=os.cpu_count())
        write(out/'execution.json',execution);started=time.monotonic()
        run_tasks(cfg,plan,out,args.workers,coordinate,source_cfg)
        summary=summarize(cfg,plan,out);execution.update(status='completed',elapsed_seconds=time.monotonic()-started)
        write(out/'execution.json',execution)
        print(json.dumps(serial(dict(execution=summary['execution'],planned=summary['planned'],status_counts=summary['status_counts'],elapsed_seconds=execution['elapsed_seconds']))))
    return 0


if __name__=='__main__':raise SystemExit(main())
