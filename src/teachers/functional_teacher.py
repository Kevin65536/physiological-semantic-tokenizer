"""Frozen pre-Tak conditional TK teacher with the inlet-balance SSM owner.

The physiology, solver and observation matrices are reused unchanged. New
targets are not a qualification or a relabeling of old teacher evidence.
It is a historical consumer, not the default Tak teacher. New Tak-derived
targets need their own versioned target/consumer contract.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.inference.ssm_output_audit import (
    OutputProblem,reference_parameters,fit_problem,hb_coordinates,output_profile,
)
from src.inference.observation_baselines import disjoint_native_bin_operator
from src.inference.shared_driver_rk4 import compiled_forward


def teacher_problem(arrays,coordinate,ssm_config,physical_config,mode='joint'):
    ops=disjoint_native_bin_operator(arrays['native_time'], steps=360, dt=.25, reference_steps=20)
    target=np.column_stack((arrays['features']@np.asarray(coordinate['pc'])*coordinate['eeg_factor'],
        arrays['hb']*coordinate['hb_factor']))
    mask=np.ones_like(target,bool)
    if mode=='hb_only': mask[:,0]=False
    elif mode=='eeg_only': mask[:,1:]=False
    elif mode!='joint': raise ValueError('unknown teacher visibility')
    problem=OutputProblem(target,np.asarray(coordinate['sd']),dict(eeg=ops['eeg'],hb=ops['model']),
        reference_parameters(physical_config),ssm_config,arm='chain_matched',visible=mask)
    return problem


def produce_teacher(arrays,coordinate,ssm_config,physical_config,*,mode='joint',alternative=False):
    problem=teacher_problem(arrays,coordinate,ssm_config,physical_config,mode)
    fit=fit_problem(problem)
    value=fit['value']
    status={k:fit[k] for k in ('converged','status','iterations','evaluations','scaled_gradient','rejected_steps')}
    status.update(objective=value['objective'],costs=value['costs'],mode=mode)
    payload=dict(u=problem.y,physical=value['physical'],remaining=fit['remaining'],
        states=value['states'],unknowns=fit['x'],prediction=value['prediction'],
        visible=problem.mask,component=value['component'])
    if not np.allclose(payload['u'][problem.mask],(payload['physical']+payload['remaining'])[problem.mask],atol=1e-10):
        raise AssertionError('full remaining identity failed')
    return payload,status


def alternative_teacher(arrays,coordinate,ssm_config,physical_config,offsets,threshold):
    problem=teacher_problem(arrays,coordinate,ssm_config,physical_config)
    fit=fit_problem(problem)
    if not fit['converged']: return None,dict(status='reference_failed')
    scales={k:max(float(v),1e-8) for k,v in zip(('T','X'),hb_coordinates(problem.y[:,1:]).std(0))}
    _,witnesses=output_profile(problem,fit,'T',scales,offsets,information='total')
    good=[w for w in witnesses if w['converged'] and w['objective_fraction']<=threshold]
    audit=[{k:v for k,v in w.items() if k!='x'} for w in witnesses]
    if not good: return None,dict(status='no_feasible_witness',witnesses=audit)
    # A single whole-trajectory representative; never select independently in time.
    chosen=min(good,key=lambda w:(abs(w['offset']),w['offset']))
    value=problem.evaluate(chosen['x'],False)
    return value['physical'],dict(status='completed',offset=chosen['offset'],witnesses=audit,
        interpretation='finite_whole_trajectory_witness_not_posterior_or_complete_feasible_set')


def match_smoothing(observations,targets,sigmas):
    """Train-only normalized roughness matching, independently for E/X/T."""
    o=np.asarray(observations,float);t=np.asarray(targets,float)
    def roughness(x):
        return np.mean(np.diff(x,axis=1)**2,axis=(0,1))/np.maximum(np.var(x,axis=(0,1)),1e-12)
    desired=roughness(t)
    candidates=[gaussian_filter1d(o,float(s),axis=1,mode='nearest') for s in sigmas]
    values=np.stack([roughness(x) for x in candidates])
    selected=np.argmin(np.abs(np.log(np.maximum(values,1e-12))-np.log(np.maximum(desired,1e-12))),axis=0)
    return dict(sigmas=[float(sigmas[i]) for i in selected],teacher_roughness=desired.tolist(),
        smooth_roughness=values[selected,np.arange(o.shape[-1])].tolist(),
        mismatch_log_ratio=np.log(np.maximum(values[selected,np.arange(o.shape[-1])],1e-12)/np.maximum(desired,1e-12)).tolist())


def apply_smoothing(observations,selection):
    return np.stack([gaussian_filter1d(observations[...,k],s,axis=-1,mode='nearest')
        for k,s in enumerate(selection['sigmas'])],axis=-1)


def synthetic_interventions(cfg,physical_config,seed):
    """Paired generator identities; every variant shares driver/noise/carriers.

    Raw EEG carriers encode a model-conditional feature, not EEG source truth.
    The exact-alias row has identical observations but a different attribution.
    """
    rng=np.random.default_rng(seed)
    n=360;t=np.arange(n)*.25
    driver=gaussian_filter1d(rng.normal(size=n),6)*.07
    params=reference_parameters(physical_config)
    initial=np.r_[0.,np.ones(4)]
    component=gaussian_filter1d(rng.normal(size=n),8)*.05
    raw_t=np.arange(18000)/200
    phases=rng.uniform(-np.pi,np.pi,size=(6,5))
    frequencies=np.array([2.,6.,10.,20.,37.])
    carriers=np.sin(2*np.pi*raw_t[:,None,None]*frequencies[None,None,:]+phases)
    noise=rng.normal(0,.05,size=(18000,6))
    native_time=np.arange(900)/10.
    operators=disjoint_native_bin_operator(native_time, steps=360, dt=.25, reference_steps=20)
    hb_noise=rng.normal(0,.0001,size=(900,2))
    result={}
    for scenario in cfg['mechanism']['scenarios']:
        p=deepcopy(params);d=driver.copy()
        amp=1.35 if scenario in ('driver_amplitude','combined') else 1.
        d*=amp
        if scenario=='driver_shape': d+=.015*np.exp(-.5*((t-40)/3)**2)
        if scenario=='tau':
            from dataclasses import replace
            p=replace(p,free=replace(p.free,tau=3.))
        forward=compiled_forward(d,initial,p,.25,8,False,False,state=True)
        st=forward['states'];f=p.fixed
        truth=np.column_stack((f.eeg_loading*st[:,0],f.P0*(st[:,4]-1)-f.Q0*(st[:,5]-1),f.Q0*(st[:,5]-1)))
        observed=truth.copy()
        if scenario in ('D_common','combined'): observed[:,1:]+=component[:,None]*np.array([.65,.35])
        if scenario=='off_D': observed[:,1:]+=component[:,None]*np.array([.2,.8])
        envelope=np.exp(np.interp(raw_t,t,observed[:,0]/.025)*.3)
        eeg=(carriers*envelope[:,None,None]).sum(-1)+noise
        hb=operators['model']@observed[:,1:]+operators['raw']@hb_noise
        if scenario=='exact_alias':
            truth[:,1:]+=component[:,None]*np.array([.65,.35])
        result[scenario]=dict(eeg=eeg[6000:10000].astype('float32'),hb=hb[120:200].astype('float32'),
            truth=truth[120:200],driver=d[120:200],tau=p.free.tau,amplitude=amp,
            functional=np.column_stack((operators['eeg']@truth[:,0],
                hb_coordinates(operators['model']@truth[:,1:])[:,::-1]))[120:200])
        if scenario=='exact_alias':
            result[scenario]['eeg']=result['base']['eeg'].copy()
            result[scenario]['hb']=result['base']['hb'].copy()
    return result
