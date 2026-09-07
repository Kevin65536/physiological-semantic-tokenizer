#!/usr/bin/env python3
"""Versioned Step5 stages; old Step1--4 and Step5A0 evidence stay immutable."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import time

for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'

import numpy as np
import scipy
import yaml
from scipy.special import logsumexp, poch
from scipy.stats import kstest, norm, spearmanr, t as student_t

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments import evaluate_step5a_inference_consistency as localization
from src.inference import t3a_balloon_joint_ssm as joint
from src.inference import t3a_balloon_robust_ssm as core

CONFIG=ROOT/'experiments/configs/physiology_semantic_tokenizer/step5_v1.yaml'


def adaptive_particle_filter(y,cfg,axis,theta,particles,seed,resample_fraction=.5):
    """Bootstrap PF with correctly carried importance weights between resamples."""
    rng=np.random.default_rng(seed); m=cfg['model']; raw=localization.raw_parameters(cfg,axis,theta)
    z=rng.normal(size=(particles,6))*m['initial_state_std']
    weights=np.ones(particles)/particles; labels=np.arange(particles)
    ll,min_ess,resampling_count=0.,1.,0
    scales=np.array(m['observation_scale']); nu=m['student_nu']
    constant=np.log(poch(nu/2,.5))-.5*np.log(nu*np.pi)-np.log(scales)
    for t,row in enumerate(y):
        if t:
            if 1/np.sum(weights**2)<resample_fraction*particles:
                cdf=np.cumsum(weights); cdf[-1]=1.
                ix=np.searchsorted(cdf,(rng.random()+np.arange(particles))/particles)
                z=z[ix]; labels=labels[ix]; weights=np.ones(particles)/particles
                resampling_count+=1
            z=localization.independent_transition(z,raw,cfg)+rng.normal(size=z.shape)*np.array(m['process_std'])*np.sqrt(m['dt'])
        available=np.isfinite(row)
        residual=(row[available]-localization.clean_from_z(z)[:,available])/scales[available]
        log_density=np.sum(constant[available]-(nu+1)/2*np.log1p(residual**2/nu),axis=1)
        with np.errstate(divide='ignore'):
            log_weights=np.log(weights)+log_density
        increment=logsumexp(log_weights)
        ll+=increment; weights=np.exp(log_weights-increment)
        min_ess=min(min_ess,1/(particles*np.sum(weights**2)))
    return dict(parameter_log_likelihood=float(ll),minimum_ess_fraction=float(min_ess),
                unique_ancestor_fraction=len(np.unique(labels))/particles,resampling_count=resampling_count)


def joint_grid(y,cfg,axis,grid,order=None,smooth=False):
    order=order or cfg['inference']['quadrature_order']
    scores=[]; results=[]
    for value in grid:
        p,c=localization.model(cfg,axis,float(value))
        if smooth:
            result=joint.smooth_balloon_joint(y,p,config=c,quadrature_order=order)
            results.append(result); scores.append(result.parameter_log_likelihood)
        else:
            scores.append(joint.parameter_log_likelihood(y,p,config=c,quadrature_order=order))
    return np.array(scores),results


def reference_job(cfg,axis,level,run,index,theta,y):
    seed=cfg['seed']+10_000_000+list(cfg['axes']).index(axis)*1_000_000+level*100_000+run*1000+index
    return adaptive_particle_filter(y,cfg,axis,theta,cfg['reference']['particles'][level],seed,
                                    cfg['reference']['resample_ess_fraction'])


def write_json(path,value):
    localization.write_json(Path(path),value)


def reference_diagnostic(cfg,run_dir,workers):
    """Development reference check on the already retained localization cases."""
    run_dir=Path(run_dir)
    source=ROOT/cfg['reference']['localization_run']
    ref=cfg['reference']; R=ref['independent_runs']; N=ref['grid_points']
    rows={}; payload={}; logs={a:np.zeros((2,R,N)) for a in cfg['axes']}
    minima={a:dict(ess=1.,ancestry=1.) for a in cfg['axes']}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={}
        for axis in cfg['axes']:
            old=json.loads((source/f'case_{axis}_00.json').read_text())
            arrays=np.load(source/f'case_{axis}_00.npz')
            y=arrays['matched_model_calibration_observations'][:ref['steps']]
            grid=np.linspace(*cfg['axes'][axis]['bounds'],N)
            payload[axis]=(old,y,grid)
            for level in range(2):
                for r in range(R):
                    for j,theta in enumerate(grid):
                        job=pool.submit(reference_job,cfg,axis,level,r,j,theta,y)
                        jobs[job]=(axis,level,r,j)
        completed=0
        for job in as_completed(jobs):
            axis,level,r,j=jobs[job]; out=job.result(); logs[axis][level,r,j]=out['parameter_log_likelihood']
            minima[axis]['ess']=min(minima[axis]['ess'],out['minimum_ess_fraction'])
            minima[axis]['ancestry']=min(minima[axis]['ancestry'],out['unique_ancestor_fraction'])
            completed+=1
            if completed%100==0:print(f'reference {completed}/{len(jobs)}',flush=True)
    for axis,(old,y,grid) in payload.items():
        ll=logsumexp(logs[axis],axis=1)-np.log(R)
        cdfs=[localization.posterior_grid(cfg,axis,grid,l)['cdf'] for l in ll]
        split=[localization.posterior_grid(cfg,axis,grid,logsumexp(l,axis=0)-np.log(len(l)))['cdf'] for l in (logs[axis][1,:R//2],logs[axis][1,R//2:])]
        normalized=np.exp(logs[axis][1]-logs[axis][1].max(axis=0))
        se=normalized.std(axis=0,ddof=1)/np.sqrt(R)/normalized.mean(axis=0)
        coarse=localization.posterior_grid(cfg,axis,grid[::2],ll[1,::2])['cdf']
        budget_diff=float(np.max(abs(cdfs[0]-cdfs[1])))
        split_diff=float(np.max(abs(split[0]-split[1])))
        grid_diff=float(np.max(abs(cdfs[1]-np.interp(grid,grid[::2],coarse))))
        order=cfg['inference']['quadrature_order']
        jll,_=joint_grid(y,cfg,axis,grid,order)
        refined,_=joint_grid(y,cfg,axis,grid,cfg['inference']['quadrature_check_order'])
        jc=localization.posterior_grid(cfg,axis,grid,jll)['cdf']
        rc=localization.posterior_grid(cfg,axis,grid,refined)['cdf']
        oldgrid=np.array(old['nonlinear_reference']['grid'])
        oldc=np.interp(grid,oldgrid,localization.posterior_grid(cfg,axis,oldgrid,np.array(old['nonlinear_reference']['predictive_score']))['cdf'])
        likelihood_pass=bool(max(se)<=ref['max_log_likelihood_se'] and max(budget_diff,split_diff,grid_diff)<=ref['max_cdf_difference'] and minima[axis]['ess']>=ref['minimum_ess_fraction'])
        path_pass=bool(minima[axis]['ancestry']>=ref['minimum_unique_ancestor_fraction'])
        rows[axis]=dict(reference_precision_pass=likelihood_pass and path_pass,
                       parameter_likelihood_precision_pass=likelihood_pass,smoothing_path_diversity_pass=path_pass,
                       maximum_log_likelihood_se=float(max(se)),
                       minimum_ess_fraction=minima[axis]['ess'],minimum_unique_ancestor_fraction=minima[axis]['ancestry'],
                       particle_budget_cdf_difference=budget_diff,split_run_cdf_difference=split_diff,
                       reference_grid_cdf_difference=grid_diff,joint_to_reference_cdf_difference=float(max(abs(jc-cdfs[1]))),
                       old_score_to_reference_cdf_difference=float(max(abs(oldc-cdfs[1]))),
                       quadrature_cdf_difference=float(max(abs(jc-rc))),quadrature_log_likelihood_max_abs=float(max(abs(jll-refined))),
                       grid=grid,independent_reference_log_likelihoods=logs[axis],reference_parameter_log_likelihood=ll[1],joint_parameter_log_likelihood=jll)
        write_json(run_dir/f'reference_{axis}.json',rows[axis])
    result=dict(stage='a0_reference',axes={a:{k:v for k,v in row.items() if k not in {'grid','independent_reference_log_likelihoods','reference_parameter_log_likelihood','joint_parameter_log_likelihood'}} for a,row in rows.items()})
    write_json(run_dir/'summary.json',result)
    return result
