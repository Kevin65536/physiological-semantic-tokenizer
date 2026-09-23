#!/usr/bin/env python3
"""Independent information inventory (A) and nonlinear counterfactual screen (B).

A reads only the established cache index and the preceding audit's identities.
B is a conditional deterministic inverse problem using the existing six-state
Balloon drift, fixed alpha/E0 gauge and an unknown bandlimited neural drive.
It is not a new teacher, stochastic SSM likelihood or measured-data fit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time
import traceback

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('HBO_CALIBRATION_REPO_ROOT', CODE_ROOT)).resolve()
sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import pandas as pd
import yaml
from scipy.integrate import solve_ivp
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
from scipy.signal import resample_poly

from src.data.homer2_preprocessing import bandpass_fnirs, modified_beer_lambert, DEFAULT_EXTINCTION_COEFFICIENTS
from src.inference.t3a_balloon_robust_ssm import (
    BalloonFixedParameters, BalloonFreeParameters, BalloonParameters,
    balloon_rhs, balloon_rhs_jacobian, observation_map, run_physical_checks,
)

CONFIG = CODE_ROOT / 'experiments/configs/physiology_semantic_tokenizer/hbo_hbr_calibration_v1.yaml'


def read_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['schema'] != 'hbo_hbr_calibration_v1' or cfg['scope'] != 'metadata_only_A_and_synthetic_nonlinear_B':
        raise ValueError('unsupported calibration contract')
    s, f = cfg['synthetic'], cfg['fit']
    if s['native_rate_hz'] != 10 or s['processed_rate_hz'] != 2 or s['duration_s'] != 30:
        raise ValueError('this version requires 300 native samples and 60 processed samples')
    if s['replicates'] < 1 or s['noise_calibration_records'] < 128:
        raise ValueError('insufficient independent calibration records')
    if f['methods'] != ['fixed', 'joint_target', 'oracle', 'independent']:
        raise ValueError('four comparison arms required')
    if len(set(s['conditions'])) != len(s['conditions']) or set(s['conditions']) != set(CONDITIONS):
        raise ValueError('condition inventory mismatch')
    return cfg


CONDITIONS = ('white', 'independent_slow', 'shared_slow', 'gain2_white',
              'gain2_independent_slow', 'dpf_mixing', 'dpf_mixing_nonrest')


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def basis(time_s, frequencies):
    angles = 2*np.pi*np.asarray(time_s)[..., None]*np.asarray(frequencies)
    return np.concatenate((np.sin(angles), np.cos(angles)), axis=-1)


def parameters(tau, eta, cfg):
    s = cfg['synthetic']
    return BalloonParameters(BalloonFixedParameters(alpha=s['alpha'], E0=s['E0'],
        gamma=s['gamma'], P0=s['P0'], Q0=eta*s['P0']),
        BalloonFreeParameters(kappa=s['kappa'], tau=tau))


def forward(x, cfg, *, derivative=False, tight=False):
    """Integrate core nonlinear drift and exact parameter/driver sensitivities.

    r(t) is an unknown forcing trajectory; the five other states follow the
    unmodified core equations. Initial s/log(f,v,p,q) are all fitted nuisances.
    """
    s, f = cfg['synthetic'], cfg['fit']
    frequencies = s['driver_frequencies_hz']
    ncoef = 2*len(frequencies)
    x = np.asarray(x)
    tau, eta = x[:2]
    coeff, initial = x[2:2+ncoef], x[2+ncoef:]
    if initial.shape != (5,) or not np.isfinite(x).all():
        raise ValueError('invalid physiological/driver tensor')
    p = parameters(tau, eta, cfg)
    nparam = len(x)
    t = np.arange(round(s['duration_s']*s['native_rate_hz']))/s['native_rate_hz']
    init_jac = np.zeros((5, nparam)); init_jac[:, 2+ncoef:] = np.eye(5)
    def fun(now, vector):
        b = basis(now, frequencies)
        z = np.r_[b@coeff, vector[:5]]
        dz = balloon_rhs(z, p)[1:]
        if not derivative:
            return dz
        j = balloon_rhs_jacobian(z, p)
        tangent = j[1:, 1:]@vector[5:].reshape(5, nparam)
        tangent[:, 2:2+ncoef] += j[1:, 0, None]*b
        tangent[2:, 0] -= dz[2:]/tau
        return np.r_[dz, tangent.ravel()]
    result = solve_ivp(fun, (t[0], t[-1]), np.r_[initial, init_jac.ravel()] if derivative else initial,
        t_eval=t, rtol=1e-10 if tight else f['integration_rtol'],
        atol=1e-12 if tight else f['integration_atol'], method='DOP853')
    if not result.success or result.y.shape[1] != len(t):
        raise FloatingPointError('nonlinear integration failed: '+result.message)
    r = basis(t, frequencies)@coeff
    physical = np.column_stack((r, result.y[0], np.exp(result.y[1:5].T)))
    clean = np.array([observation_map(row, p)[1:] for row in physical])
    if not derivative:
        return clean, r, physical
    tangent = result.y[5:].T.reshape(len(t), 5, nparam)
    dh = np.empty((len(t), 2, nparam))
    dh[:, 1] = p.fixed.Q0*physical[:, 5, None]*tangent[:, 4]
    dh[:, 1, 1] += s['P0']*(physical[:, 5]-1)
    dh[:, 0] = s['P0']*physical[:, 4, None]*tangent[:, 3]-dh[:, 1]
    return clean, dh, r, physical


def processing(values):
    filtered, info = bandpass_fnirs(values, sample_rate_hz=10.)
    if info['status'] != 'applied':
        raise ValueError('processing failed')
    y = resample_poly(filtered, 1, 5, axis=0)
    return y-y[:10].mean(axis=0)


@lru_cache(maxsize=1)
def temporal_operator():
    op = processing(np.eye(300))
    u, singular, _ = np.linalg.svd(op, full_matrices=False)
    keep = singular > singular[0]*1e-8
    return op, u[:, keep], singular


def effective(tau, eta, cfg):
    s = cfg['synthetic']
    c = 1+(1-s['E0'])*np.log1p(-s['E0'])/s['E0']
    return np.array([1/tau, eta*c, eta*(1-c)/(s['alpha']*tau)])


def observation_matrix(condition, cfg):
    if condition.startswith('gain2'):
        return np.diag([1., 2.])
    if condition.startswith('dpf'):
        e = np.array([DEFAULT_EXTINCTION_COEFFICIENTS[w] for w in (760., 850.)])
        return np.linalg.solve(e, np.diag(cfg['synthetic']['dpf_relative_factors'])@e)
    return np.eye(2)


def optical_replay(clean, condition, cfg):
    e = np.array([DEFAULT_EXTINCTION_COEFFICIENTS[w] for w in (760., 850.)])
    factors = cfg['synthetic']['dpf_relative_factors'] if condition.startswith('dpf') else [1., 1.]
    intensity = np.exp(-(clean@e.T)*18*np.asarray(factors))
    hb, _ = modified_beer_lambert(-np.log(intensity)[:, None, :], wavelengths_nm=[760., 850.])
    result = hb[:, 0]
    if condition.startswith('gain2'):
        result = result@np.diag([1., 2.])
    return result


def noise_factor(condition, cfg):
    s = cfg['synthetic']
    op, u, _ = temporal_operator()
    native = np.eye(600)*s['white_sd']
    if 'slow' in condition or condition.startswith('dpf'):
        t = np.arange(300)/10
        slow = basis(t, s['slow_frequencies_hz'])*s['slow_sd']/np.sqrt(len(s['slow_frequencies_hz']))
        slow = np.kron(slow, np.ones((2, 1)) if condition == 'shared_slow' else np.eye(2))
        native = np.column_stack((native, slow))
    return np.kron(u.T@op, np.eye(2))@native


def calibrate_optics(matrix, cfg, seed):
    """Independent known Hb standards; no target physiology or curves enter."""
    s = cfg['synthetic']; rng = np.random.default_rng(seed)
    x = rng.normal(0, s['optical_standard_sd'], (s['optical_standard_samples'], 2))
    y = x@matrix.T+rng.normal(0, s['optical_standard_noise_sd'], x.shape)
    estimate = np.linalg.lstsq(x, y, rcond=None)[0].T
    residual = y-x@estimate.T
    covariance = np.kron(residual.T@residual/(len(x)-2), np.linalg.inv(x.T@x))
    return estimate, covariance


def generate_panel(condition, replicate, tau, cfg):
    s = cfg['synthetic']; seed = s['seed']+10000*replicate
    # The same r and noise draw are paired across tau and observation conditions.
    coeff = np.random.default_rng(seed).normal(0, s['driver_coefficient_sd'], 2*len(s['driver_frequencies_hz']))
    initial = s['nonrest_transformed_initial'] if condition.endswith('nonrest') else [0.]*5
    truth = np.r_[tau, s['eta'], coeff, initial]
    clean, driver, physical = forward(truth, cfg, tight=True)
    matrix = observation_matrix(condition, cfg)
    factor = noise_factor(condition, cfg)
    rng = np.random.default_rng(seed+1)
    noise = factor@rng.normal(size=factor.shape[1])
    op, u, _ = temporal_operator()
    native_mean = optical_replay(clean, condition, cfg)
    target = (u.T@op@native_mean).ravel()+noise
    covariance = factor@factor.T
    calibration_rng = np.random.default_rng(seed+2)
    noise_records = factor@calibration_rng.normal(size=(factor.shape[1], s['noise_calibration_records']))
    noise_covariance = noise_records@noise_records.T/s['noise_calibration_records']
    optical, optical_covariance = calibrate_optics(matrix, cfg, seed+3)
    return dict(truth=truth, clean=clean, driver=driver, physical=physical,
        matrix=matrix, target=target, covariance=covariance,
        calibration_covariance=noise_covariance, optical=optical, optical_covariance=optical_covariance)


def fit_target(target, matrix, covariance, cfg, *, free_matrix=False):
    f = cfg['fit']; ncoef = 2*len(cfg['synthetic']['driver_frequencies_hz']); nphys = 7+ncoef
    op, u, _ = temporal_operator(); project = u.T@op
    chol = np.linalg.cholesky(covariance)
    low = [f['tau_bounds'][0], f['eta_bounds'][0]]+[f['driver_bounds'][0]]*ncoef+[f['initial_bounds'][0]]*5
    high = [f['tau_bounds'][1], f['eta_bounds'][1]]+[f['driver_bounds'][1]]*ncoef+[f['initial_bounds'][1]]*5
    if free_matrix:
        low += f['observation_matrix_bounds'][0]; high += f['observation_matrix_bounds'][1]
    cache = {}
    def evaluate(x):
        if 'x' not in cache or not np.array_equal(x, cache['x']):
            clean, dh, driver, physical = forward(x[:nphys], cfg, derivative=True)
            a = x[nphys:].reshape(2,2) if free_matrix else matrix
            reduced = project@clean
            pred = (reduced@a.T).ravel()
            derivative = np.einsum('nt,tck,dc->ndk', project, dh, a).reshape(len(target), nphys)
            da = np.zeros((len(reduced), 2, 4))
            da[:, 0, :2] = reduced; da[:, 1, 2:] = reduced
            da = da.reshape(len(target),4)
            if free_matrix:
                derivative = np.column_stack((derivative, da))
            cache.update(x=x.copy(), residual=solve_triangular(chol, pred-target, lower=True),
                jac=solve_triangular(chol, derivative, lower=True),
                matrix_jac=solve_triangular(chol, da, lower=True), clean=clean, driver=driver, physical=physical)
        return cache
    attempts=[]; best=None
    for tau, eta in f['starts']:
        x = np.r_[tau, eta, np.zeros(ncoef+5)]
        if free_matrix: x = np.r_[x, np.eye(2).ravel()]
        try:
            result = least_squares(lambda q: evaluate(q)['residual'], x,
                jac=lambda q: evaluate(q)['jac'], bounds=(low, high),
                max_nfev=f['max_nfev'], ftol=f['ftol'], xtol=f['ftol'], gtol=f['ftol'], x_scale='jac')
            attempts.append(dict(success=bool(result.success), cost=float(result.cost), nfev=result.nfev,
                optimality=float(result.optimality), message=result.message))
            if result.success and (best is None or result.cost < best.cost): best=result
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            attempts.append(dict(success=False, error=str(exc)))
    if best is None:
        return dict(status='failed', attempts=attempts)
    e = evaluate(best.x)
    _, singular, vt = np.linalg.svd(e['jac'], full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-8))
    fullrank = rank == len(best.x)
    cov = (vt.T/np.maximum(singular, singular[0]*1e-8)**2)@vt
    active = bool(np.any(np.asarray(best.active_mask)!=0))
    physical_checks = run_physical_checks(e['physical'], parameters(best.x[0], best.x[1], cfg))
    physical_ok = all(physical_checks[k] for k in ('finite', 'positive_fvpq',
        'oxygen_extraction_in_unit_interval', 'absolute_hb_nonnegative', 'hbr_not_above_hbt'))
    return dict(status='completed' if physical_ok else 'failed_physical', x=best.x,
        clean=e['clean'], driver=e['driver'], covariance=cov, jac=e['jac'], matrix_jac=e['matrix_jac'],
        intervals_valid=fullrank and not active and physical_ok, rank=rank, active_bound=active,
        cost=float(best.cost), singular_values=singular, attempts=attempts)


def summarize_fit(fit, panel, cfg, *, calibration_covariance=None):
    if fit['status'] != 'completed': return {k:v for k,v in fit.items() if k in ('status','attempts')}
    x=fit['x']; truth=panel['truth']; ncoef=2*len(cfg['synthetic']['driver_frequencies_hz'])
    cov=fit['covariance'].copy()
    if calibration_covariance is not None:
        sensitivity = -np.linalg.pinv(fit['jac'], rcond=1e-8)@fit['matrix_jac']
        cov += sensitivity@calibration_covariance@sensitivity.T
    comb=effective(x[0],x[1],cfg); expected=effective(truth[0],truth[1],cfg)
    error=fit['driver']-panel['driver']
    row=dict(status=fit['status'],tau=float(x[0]),eta=float(x[1]),effective=comb.tolist(),
        combination_relative_error=float(np.sqrt(np.mean(((comb-expected)/expected)**2))),
        driver_nrmse=float(np.linalg.norm(error)/np.linalg.norm(panel['driver'])),
        driver_bias=float(error.mean()), driver_correlation=float(np.corrcoef(fit['driver'],panel['driver'])[0,1]),
        clean_Hb_rmse=np.sqrt(np.mean((fit['clean']-panel['clean'])**2,axis=0)).tolist(),
        objective=fit['cost'],rank=fit['rank'],active_bound=fit['active_bound'],
        parameters=x.tolist(),attempts=fit['attempts'],intervals_valid=fit['intervals_valid'])
    if fit['intervals_valid']:
        sd=np.sqrt(np.maximum(np.diag(cov),0)); b=basis(np.arange(300)/10,cfg['synthetic']['driver_frequencies_hz'])
        rsd=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',b,cov[2:2+ncoef,2:2+ncoef],b),0))
        ec=np.array([[-1/x[0]**2,0],[0,comb[1]/x[1]],[-comb[2]/x[0],comb[2]/x[1]]])
        esd=np.sqrt(np.maximum(np.diag(ec@cov[:2,:2]@ec.T),0))
        row.update(tau_95_interval=[float(x[0]-1.96*sd[0]),float(x[0]+1.96*sd[0])],
            tau_95_covers=bool(abs(x[0]-truth[0])<=1.96*sd[0]),
            combination_95_covers=(abs(comb-expected)<=1.96*esd).tolist(),
            driver_pointwise_95_coverage=float(np.mean(abs(error)<=1.96*rsd)),
            driver_interval_mean_width=float(np.mean(3.92*rsd)))
    return row


def run_task(task):
    key, condition, replicate, tau, cfg = task; started=time.monotonic()
    try:
        panel=generate_panel(condition,replicate,tau,cfg)
        white=noise_factor('white',cfg); white=white@white.T
        models={}; traces={}
        for method in cfg['fit']['methods']:
            matrix=(panel['matrix'] if method=='oracle' else panel['optical'] if method=='independent' else np.eye(2))
            cov=(panel['covariance'] if method=='oracle' else panel['calibration_covariance'] if method=='independent' else white)
            fit=fit_target(panel['target'],matrix,cov,cfg,free_matrix=method=='joint_target')
            models[method]=summarize_fit(fit,panel,cfg,calibration_covariance=panel['optical_covariance'] if method=='independent' else None)
            if fit['status']=='completed': traces[method]=fit['driver'].tolist()
        eig,vec=np.linalg.eigh(panel['optical_covariance']); delta=(1.96*np.sqrt(eig[-1])*vec[:,-1]).reshape(2,2)
        for sign,label in [(-1,'independent_minus'),(1,'independent_plus')]:
            fit=fit_target(panel['target'],panel['optical']+sign*delta,panel['calibration_covariance'],cfg)
            models[label]=summarize_fit(fit,panel,cfg)
        return dict(key=key,condition=condition,replicate=replicate,true_tau=tau,status='completed',
            models=models,driver_traces=traces,seconds=time.monotonic()-started,
            true_matrix=panel['matrix'].tolist(),estimated_matrix=panel['optical'].tolist(),
            calibration_matrix_covariance=panel['optical_covariance'].tolist())
    except Exception:
        return dict(key=key,condition=condition,replicate=replicate,true_tau=tau,status='failed',
            error=traceback.format_exc(),seconds=time.monotonic()-started)


def information_inventory(cfg, out):
    """Join the existing bounded record identities without loading signal arrays."""
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    from src.data.registry import get_dataset_registration
    identities = pd.read_csv(ROOT/cfg['parent_inventory'], dtype={'subject':str})[
        ['dataset_id','subject','record']].drop_duplicates()
    index = CleanPhysiologyCacheIndex(ROOT/cfg['cache_root'])
    rows=[]; details=[]
    for identity in identities.itertuples(index=False):
        key='|'.join(identity)
        matches=index.records_by_join_key.get(key,[])
        if len(matches)!=1: raise ValueError(f'nonunique/missing record {key}')
        record=matches[0]; m=record.manifest; d=record.dataset_id
        reg=get_dataset_registration(d)
        state=m['measurement']['fnirs_preprocessing_state']
        native=state['native_contract']; device=m.get('metadata',{}).get('device_metadata',{})
        sources=[ref.relative_path for ref in reg.documentation]
        if d=='eeg_fnirs_single_trial':
            wavelength='760/850 nm'; geometry='30 mm; source documentation'
            optical='dual wavelength intensities in registered native source'
            aux='ECG and respiration explicitly documented; current loader exposes EOG only; auxiliary record mapping unverified'
        elif d=='simultaneous_eeg_nirs':
            wavelength='760/850 nm'; geometry='30 mm all 36 channels; source documentation'
            optical='released Hb; vendor documentation describes wl1/wl2 but local raw optical mapping unverified'
            aux='EOG documented; independent systemic reference not established'
        elif d=='refed':
            wavelength='780/805/830 nm'; geometry='nominal 30 mm; original publication catalog'
            optical='released Hb and separate three Abs exports; optical calibration not established'
            aux='no systemic reference established in inspected README/registered metadata'
        else:
            wavelength='/'.join(device.get('Wave[nm]',[]))+' nm (nominal; actual channel wavelengths in metadata)'
            geometry='probe identity retained; subject-specific pathlength and short separation not established'
            optical='released Oxy/Deoxy; device gain settings retained but not a calibrated Hb mixing matrix'
            aux='no systemic reference established in inspected readme/registered metadata'
        base=dict(dataset_id=d,subject=record.canonical_subject_id,record=record.base_record_id,
            join_key=key,unit=state['canonical_unit'],native_rate_hz=m['native_sample_rate_hz'],
            source_evidence=';'.join(sources),source_record=';'.join(x['path'] for x in m['source_files']))
        facts=[
            ('known_processing','fixed',json.dumps({k:state.get(k) for k in ['scaling','filter_band_hz','canonical_sample_rate_hz']},ensure_ascii=False),'verified_record_metadata'),
            ('native_units','fixed' if native.get('unit_status')=='verified' else 'unknown',state['canonical_unit'],'record_metadata; relative unit is not absolute calibration'),
            ('wavelengths_and_optical_entry','fixed',wavelength+'; '+optical,'documented; record metadata when present'),
            ('geometry','fixed' if d!='visual_cognitive_motivation' else 'unknown',geometry,'nominal geometry does not identify DPF'),
            ('relative_gain_and_mixing','unknown','no independent known-Hb standards or calibrated conversion identified','not_identified_in_inspected_sources'),
            ('DPF','unknown',f"implemented assumption={native.get('partial_pathlength_factor','not exposed')}; no empirical interval",'implementation assumption not calibration'),
            ('P0_Q0_absolute_baselines','unknown','Hb changes do not establish absolute physiological baselines','not_identified_in_inspected_sources'),
            ('instrument_and_slow_noise','unknown','no independent dark/noise-only calibration identified; resting activity is not pure noise','not_identified_in_inspected_sources'),
            ('auxiliary_source_coefficients','estimable_after_auxiliary_mapping' if d=='eeg_fnirs_single_trial' else 'unknown',aux,'availability lead; no coefficient estimated from target Hb'),
            ('calibration_uncertainty_interval','unknown','no defensible empirical interval; synthetic DPF factors are sensitivity only','no_interval_invented'),
            ('continuous_support','fixed',f"cache record length={m['array_shapes']['fnirs'][0]/m['sample_rate_hz']:.1f}s; do not concatenate trials; append/support boundaries require review",'length metadata only; not strict split independence'),
        ]
        for name,status,value,confidence in facts:
            rows.append(dict(**base,parameter=name,classification=status,value=value,confidence=confidence))
        details.append(dict(**base,native_contract=native,device_metadata=device,
            fnirs_preprocessing=state,eeg_auxiliary_names=m['measurement']['eeg_preprocessing_state'].get('auxiliary_channel_names',[])))
    pd.DataFrame(rows).to_csv(out/'calibration_information.csv',index=False)
    atomic_json(out/'record_evidence.json',details)
    result=dict(records=len(identities),datasets=int(identities.dataset_id.nunique()),
        independently_calibrated_record_count=0,
        measured_candidate_status='not_established_by_inspected_evidence',
        auxiliary_lead='Single-Trial ECG/respiration documented; needs explicit record/clock/loader mapping',
        scope='prior_audit_record_identities_and_registered_metadata_no_signal_arrays',
        absence_semantics='not_found_in_inspected_documentation_and_metadata_not_claim_of_nonexistence')
    atomic_json(out/'information_summary.json',result)
    return result


def software_checks(cfg):
    ncoef=2*len(cfg['synthetic']['driver_frequencies_hz'])
    x=np.r_[2.3,.37,np.linspace(-.012,.016,ncoef),[.005,.01,-.01,.015,-.02]]
    h,j,r,physical=forward(x,cfg,derivative=True,tight=True)
    errors=[]
    for k in [0,1,2,len(x)-2]:
        step=1e-5; delta=np.zeros(len(x));delta[k]=step
        fd=(forward(x+delta,cfg,tight=True)[0]-forward(x-delta,cfg,tight=True)[0])/(2*step)
        errors.append(float(np.max(abs(fd-j[:,:,k]))))
    if max(errors)>2e-6: raise AssertionError(f'nonlinear sensitivity mismatch {errors}')
    op,u,singular=temporal_operator(); rng=np.random.default_rng(98); y=rng.normal(size=(300,2))
    processing_error=float(np.max(abs(op@y-processing(y))))
    if processing_error>1e-10: raise AssertionError('compiled processing mismatch')
    e=np.array([DEFAULT_EXTINCTION_COEFFICIENTS[w] for w in (760.,850.)])
    # Full intensity -> natural OD -> the existing MBLL mapping, including DPF error.
    intensity=np.exp(-(h@e.T)*18*np.asarray(cfg['synthetic']['dpf_relative_factors']))
    converted,_=modified_beer_lambert(-np.log(intensity)[:,None,:],wavelengths_nm=[760.,850.])
    optical_error=float(np.max(abs(converted[:,0]-h@observation_matrix('dpf_mixing',cfg).T)))
    if optical_error>1e-12: raise AssertionError('MBLL counterfactual mapping mismatch')
    a=generate_panel('gain2_white',0,1.,cfg);b=generate_panel('gain2_white',0,4.,cfg)
    if not np.array_equal(a['driver'],b['driver']): raise AssertionError('counterfactual driver pairing lost')
    if not np.array_equal(a['optical'],b['optical']): raise AssertionError('calibration depends on target physiology')
    return dict(derivative_max_errors=errors,compiled_processing_error=processing_error,
        intensity_MBll_closure_error=optical_error,native_shape=list(h.shape),processed_shape=[60,2],
        retained_temporal_rank=u.shape[1],discarded_singular_values=singular[u.shape[1]:].tolist(),
        driver_sign_scale='P0=1,positive_core_neurovascular_gain,no_posthoc_alignment',
        inference='conditional_deterministic_unknown_driver_not_full_stochastic_SSM')


def inventory(cfg):
    return [(f'{cond}|r{rep:02d}|tau{tau:g}',cond,rep,tau,cfg)
        for cond in cfg['synthetic']['conditions'] for rep in range(cfg['synthetic']['replicates'])
        for tau in cfg['synthetic']['tau_seconds']]


def finish(out, tasks, cfg):
    results=[json.loads((out/'tasks'/f'{task[0]}.json').read_text()) for task in tasks]
    rows=[]; paired=[]
    for result in results:
        for method in cfg['fit']['methods']+['independent_minus','independent_plus']:
            fit=result.get('models',{}).get(method,dict(status='task_failed'))
            row={k:result[k] for k in ['key','condition','replicate','true_tau']}
            row.update(method=method,**{k:v for k,v in fit.items() if not isinstance(v,(list,dict))})
            for k,v in zip(['HbO_rmse','HbR_rmse'],fit.get('clean_Hb_rmse',[None,None])):row[k]=v
            rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(out/'recovery_metrics.csv',index=False)
    summaries=[]
    for (condition,method),g in frame.groupby(['condition','method']):
        good=g[g.status=='completed']; valid=good[good.intervals_valid.fillna(False)] if len(good) else good
        row=dict(condition=condition,method=method,total=len(g),completed=len(good),valid_intervals=len(valid))
        for metric in ['combination_relative_error','driver_nrmse','HbO_rmse','HbR_rmse']:
            row[metric+'_median']=float(good[metric].median()) if len(good) else None
        row['tau_95_coverage']=float(valid.tau_95_covers.mean()) if len(valid) else None
        row['driver_pointwise_95_coverage']=float(valid.driver_pointwise_95_coverage.mean()) if len(valid) else None
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(out/'recovery_summary.csv',index=False)
    for (condition,rep,method),g in frame.groupby(['condition','replicate','method']):
        low=g[(g.true_tau==1)&(g.status=='completed')];high=g[(g.true_tau==4)&(g.status=='completed')]
        if len(low) and len(high):
            paired.append(dict(condition=condition,replicate=int(rep),method=method,true_delta_tau=3.,
                estimated_delta_tau=float(high.iloc[0].tau-low.iloc[0].tau),
                contrast_preserved=bool(high.iloc[0].tau>low.iloc[0].tau)))
    pd.DataFrame(paired).to_csv(out/'physiology_counterfactual.csv',index=False)
    invariance=[];by_key={(r['condition'],r['replicate'],r['true_tau']):r for r in results}
    for result in results:
        ref=by_key.get(('white',result['replicate'],result['true_tau']),{})
        for method in cfg['fit']['methods']:
            x=result.get('driver_traces',{}).get(method);y=ref.get('driver_traces',{}).get(method)
            if x is not None and y is not None:
                invariance.append(dict(key=result['key'],method=method,
                    driver_change_relative_to_white=float(np.linalg.norm(np.array(x)-y)/max(np.linalg.norm(y),1e-15))))
    pd.DataFrame(invariance).to_csv(out/'observation_counterfactual.csv',index=False)
    summary=dict(experiment_id=cfg['experiment_id'],tasks_total=len(tasks),
        tasks_completed=sum(r['status']=='completed' for r in results),
        fits_total=len(frame),fits_completed=int((frame.status=='completed').sum()),
        primary=cfg['primary'],by_condition=summaries,
        C_status='not_started_independent_measured_calibration_not_established',
        D_status='not_started_C_positive_evidence_required',
        verdict='screen_requires_recovery_coverage_and_independent_information_review_not_teacher_qualification')
    atomic_json(out/'summary.json',summary)
    # The report is generated from saved rows, never from embedded result numbers.
    columns=['condition','method','total','completed','valid_intervals','combination_relative_error_median','driver_nrmse_median','tau_95_coverage']
    table=pd.DataFrame(summaries)[columns].to_markdown(index=False,floatfmt='.4g')
    text='# 独立观测信息与非线性双向反事实筛查\n\n'
    text+='本轮 A 只读取上一轮身份与当前缓存元数据；B 使用非线性六状态核心的受迫确定性子模型。'
    text+='alpha/E0/kappa 固定，拟合 tau、eta、未知频带内 r 系数及全部五个血流初态。'
    text+='这是有利条件下的恢复筛查，不是完整随机 SSM 或 teacher 资格。\n\n'
    text+=f"完整分母：{summary['tasks_total']} 个面板、{summary['fits_total']} 次拟合（含校准敏感性），{summary['fits_completed']} 次收敛。\n\n"
    text+=table+'\n\n'
    text+='组合误差为 lambda/a/k 的相对 RMS；r NRMSE 不做逐条符号、幅度或时移对齐。'
    text+='局部 Gaussian 区间在秩亏或触界时记不可用；覆盖率只在有效区间上计算，同时保留完整分母。'
    text+='光学标准和噪声记录与目标独立；慢成分在加性 Hb 层生成，滤波/重采样和基线算子同步传播到均值及协方差。'
    text+='DPF 0.8/1.2 是机制敏感性设定，没有被当成实测 DPF 置信区间。\n\n'
    text+='[逐记录校准表](calibration_information.csv)、[逐次拟合](recovery_metrics.csv)、[真实生理变化](physiology_counterfactual.csv)、[观测变化](observation_counterfactual.csv)。\n\n'
    text+='A 尚未建立可用于本次实测候选的独立标定；Single-Trial ECG/呼吸的发布说明提供后续线索。C、D 按依赖条件保留未启动。未生成 PDF，未检查 WPS。\n'
    (out/'REPORT.md').write_text(text)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--inventory-only',action='store_true')
    parser.add_argument('--pilot',type=int,default=0,help='Run exactly the first N synthetic panels as a throughput pilot')
    parser.add_argument('--workers',type=int)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();cfg=read_config(args.config)
    if args.check_only:
        print(json.dumps(software_checks(cfg),indent=2));return
    if args.output_dir is None:parser.error('--output-dir required')
    out=args.output_dir.resolve()
    if out.parent!=ROOT/cfg['output_namespace']:raise ValueError('output must be a fresh child of owning audit root')
    workers=args.workers or cfg['resources']['workers']
    if not 1<=workers<=len(os.sched_getaffinity(0)):raise ValueError('invalid worker count')
    if args.resume:
        manifest=json.loads((out/'manifest.json').read_text())
        if manifest['resolved_config']!=cfg or manifest['pilot']!=args.pilot:raise ValueError('resume contract mismatch')
        if manifest['status']=='completed':raise ValueError('completed evidence is read only')
    else:
        allowed={'source_snapshot','launch.json','resources.json','run.log'}
        if out.exists() and any(p.name not in allowed for p in out.iterdir()):
            raise ValueError('existing evidence requires --resume; completed evidence is read only')
        out.mkdir(exist_ok=True);(out/'tasks').mkdir()
        manifest=dict(experiment_id=cfg['experiment_id'],resolved_config=cfg,status='preparing',
            started_at=time.time(),command=sys.argv,pilot=args.pilot,workers=workers,
            signal_array_reads=0,primary=cfg['primary'])
        atomic_json(out/'manifest.json',manifest)
    import fcntl
    with (out/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        checks=software_checks(cfg);atomic_json(out/'software_checks.json',checks)
        info=information_inventory(cfg,out)
        if args.inventory_only:
            manifest.update(status='completed_inventory',information=info);atomic_json(out/'manifest.json',manifest);return
        tasks=inventory(cfg)
        if args.pilot:tasks=tasks[:args.pilot]
        atomic_json(out/'task_inventory.json',[dict(key=t[0],condition=t[1],replicate=t[2],tau=t[3]) for t in tasks])
        pending_tasks=[t for t in tasks if not (out/'tasks'/f'{t[0]}.json').exists()]
        completed=len(tasks)-len(pending_tasks);started=time.monotonic();manifest.update(status='running',tasks_total=len(tasks),completed_tasks=completed)
        atomic_json(out/'manifest.json',manifest)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            todo=iter(pending_tasks);pending={}
            for _ in range(min(len(pending_tasks),workers,cfg['resources']['max_in_flight'])):
                task=next(todo);pending[pool.submit(run_task,task)]=task[0]
            while pending:
                done,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
                for future in done:
                    key=pending.pop(future);result=future.result()
                    atomic_json(out/'tasks'/f'{key}.json',result);completed+=1
                    task=next(todo,None)
                    if task is not None:pending[pool.submit(run_task,task)]=task[0]
                elapsed=time.monotonic()-started
                newly_completed=completed-(len(tasks)-len(pending_tasks))
                manifest.update(completed_tasks=completed,elapsed_this_invocation_seconds=elapsed,
                    estimated_remaining_seconds=elapsed*(len(tasks)-completed)/newly_completed if newly_completed else None)
                atomic_json(out/'manifest.json',manifest)
                print(f"panels {completed}/{len(tasks)} elapsed={elapsed:.1f}s",flush=True)
        summary=finish(out,tasks,cfg)
        manifest.update(status='completed',completed_at=time.time(),completed_tasks=len(tasks),
            fits_completed=summary['fits_completed'],fits_total=summary['fits_total'])
        atomic_json(out/'manifest.json',manifest)
        print(json.dumps({k:v for k,v in summary.items() if k!='by_condition'},indent=2))


if __name__=='__main__':main()
