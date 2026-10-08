#!/usr/bin/env python3
"""Synthetic, same-observation state-continuity diagnostic for SSM-STRENGTHEN.

Three independently referenced 30 s observation blocks come from one continuous
90 s generator. Only vascular state propagation differs between the two arms.
No measured reader, extra source, cross-boundary driver or component prior.
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
from scipy.linalg import block_diag
from scipy.ndimage import gaussian_filter1d
import yaml

from src.inference.observation_baselines import native_feature_operators
from src.inference.shared_driver_attribution import (
    equal_capacity_hb_basis, fit_conditioned_shared_driver, grouped_component_overlap,
)
from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
from src.inference.t3a_balloon_robust_ssm import BalloonParameters, BalloonFixedParameters, BalloonFreeParameters


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(serial(value), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg['schema'] != 'ssm_state_continuity_v1' or cfg['experiment_id'] != 'SSM-STRENGTHEN-S3-v1'
            or cfg['data_boundary'] != 'synthetic_only_no_measured_arrays' or cfg['protected_data'] != 'forbidden'
            or cfg['tensor'] != dict(segments=3, segment_steps=120, dt_s=.25, state_order=['r','s','f','v','p','q'])
            or cfg['comparison']['driver_curvature_breaks'] != [120,240]
            or cfg['observation']['modes_per_segment'] != 4 or cfg['observation']['rho'] != .35
            or cfg['modes'] != ['full','center_Hb','Hb_hidden']
            or cfg['scenarios'] != ['neural_only','common_colored'] or cfg['initial'] != ['rest','nonrest']
            or set(cfg['seed_streams']) != {'training','pilot','evaluation'}
            or len(set(cfg['seed_streams'].values())) != 3
            or cfg['resources']['max_workers'] != 8 or cfg['resources']['numerical_threads'] != 1):
        raise ValueError('synthetic-only continuous tensor, split or capacity contract mismatch')
    if cfg['training_identities'] < 3 or cfg['evaluation_identities'] < 3:
        raise ValueError('independent training and evaluation identities required')
    return cfg


def parameters(base):
    fixed = {k:v for k,v in base['fixed'].items() if k not in ('tau','kappa')}
    return BalloonParameters(fixed=BalloonFixedParameters(**fixed),
        free=BalloonFreeParameters(tau=base['fixed']['tau'], kappa=base['fixed']['kappa']))


@lru_cache(maxsize=2)
def operators(segment_steps=120):
    native = native_feature_operators(segment_steps)
    local = np.zeros((3*segment_steps,3*segment_steps))
    local[0::3,0::3] = native['eeg']
    local[1::3,1::3] = local[2::3,2::3] = native['fnirs']@native['native_interpolation']
    basis = equal_capacity_hb_basis(local,modes=4,rho=.35)
    return local, basis, block_diag(local,local,local), block_diag(basis,basis,basis)


def visibility(mode, n=120):
    mask = np.ones((3*n,3),bool)
    if mode == 'center_Hb':
        mask[n+n//2-8:n+n//2+8,1:] = False
    elif mode == 'Hb_hidden':
        mask[n:2*n,1:] = False
    elif mode != 'full':
        raise ValueError('unregistered mask')
    return mask


def synthetic_case(cfg, base, split, identity, initial, scenario):
    if split not in cfg['seed_streams'] or initial not in cfg['initial'] or scenario not in cfg['scenarios']:
        raise ValueError('unregistered generator identity')
    n, dt = cfg['tensor']['segment_steps'], cfg['tensor']['dt_s']
    seed = [cfg['seed'],cfg['seed_streams'][split],identity]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    generator = cfg['generator']
    driver = gaussian_filter1d(rng.normal(size=3*n),generator['driver_smoothing_steps'])
    driver *= generator['driver_sd']/driver.std()
    perturbation = rng.normal(size=5)*generator['nonrest_sd']
    x0 = np.r_[0.,np.ones(4)]+(perturbation if initial == 'nonrest' else 0.)
    forward = nonlinear_driver_forward(driver,x0,parameters(base),dt,
        substeps=generator['substeps'],derivative=False,numerical_backend='numba')
    op = operators(n)[2]
    physical = (op@forward['canonical_prediction'].ravel()).reshape(3*n,3)
    colored = gaussian_filter1d(rng.normal(size=3*n),generator['common_smoothing_steps'])
    colored *= generator['common_sd']/colored.std()
    component = (op@(colored[:,None]*np.array([0.,.65,.35])).ravel()).reshape(3*n,3)
    if scenario == 'neural_only':
        component[:] = 0.
    noise = rng.normal(size=(3*n,3))*np.asarray(generator['canonical_noise_sd'])
    noise = (op@noise.ravel()).reshape(3*n,3)
    return dict(target=physical+component+noise, physical_truth=physical,
        component_truth=component,driver=driver,states=forward['states'],seed=seed,
        identity=identity,initial=initial,scenario=scenario)


def calibration(cfg, base):
    cases = [synthetic_case(cfg,base,'training',i,'rest','neural_only')
             for i in range(cfg['training_identities'])]
    return dict(sd=np.std(np.concatenate([c['target'] for c in cases]),axis=0),
        source='training_stream_neural_only_continuous_trajectories',
        identities=[c['seed'] for c in cases],driver_sd=cfg['generator']['driver_sd'])


def fit_one(cfg, base, target, sd, visible, *, chained=False, mean=None, jacobian=False):
    n, dt = cfg['tensor']['segment_steps'], cfg['tensor']['dt_s']
    local, basis, op, chain_basis = operators(n)
    s = base['solver']
    options = {k:s[k] for k in ('penalty','initial_penalty','flow_prior_weight','flow_prior_log_sd',
                               'substeps','gradient_tolerance','numerical_backend')}
    options.update(mean_operator=op if chained else local,
        max_evaluations=s['max_evaluations_per_trial'],return_jacobian=jacobian,
        curvature_breaks=[n,2*n] if chained else ())
    if mean is None:
        options.update(observation_basis=chain_basis if chained else basis,
            observation_coefficient_sd=np.tile(cfg['observation']['coefficient_sd'],3 if chained else 1))
    return fit_conditioned_shared_driver(target,parameters(base),dt,component_mean=mean,
        sd=sd,visible=visible,**options)


def combine_free(results):
    combined = dict(converged=all(r['converged'] for r in results),
        status='completed' if all(r['converged'] for r in results) else 'failed_segment',
        segment_status=[r['status'] for r in results],
        evaluations=sum(r['evaluations'] for r in results),
        objective=sum(r.get('objective',np.nan) for r in results))
    for key in ('driver','states','prediction','physical_prediction','observation_component','observation_coefficients'):
        if all(key in r for r in results):
            combined[key] = np.concatenate([r[key] for r in results])
    return combined


def fit_arms(cfg, base, case, sd, mode, *, fixed=None, jacobian=False):
    n = cfg['tensor']['segment_steps']
    mask = visibility(mode,n)
    free = [fit_one(cfg,base,case['target'][j*n:(j+1)*n],sd,mask[j*n:(j+1)*n],
            mean=None if fixed is None else fixed['C_free'][j*n:(j+1)*n],jacobian=jacobian)
            for j in range(3)]
    chain = fit_one(cfg,base,case['target'],sd,mask,chained=True,
        mean=None if fixed is None else fixed['C_chain'],jacobian=jacobian)
    return dict(C_free=combine_free(free),C_chain=chain), free


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value)**2)))


def metrics(cfg, case, result, sd, mode, *, target_only=False):
    n = cfg['tensor']['segment_steps']
    segment = slice(n,2*n)
    prediction_segment = slice(None) if target_only else segment
    score = ~visibility(mode,n)[segment] if mode != 'full' else np.ones((n,3),bool)
    row = dict(converged=bool(result['converged']),status=result['status'],
        objective=result.get('objective'),evaluations=result.get('evaluations'),score_count=int(score.sum()))
    if 'prediction' not in result:
        return row
    error = (result['prediction'][prediction_segment]-case['target'][segment])/sd
    row['score_nrmse'] = rms(error[score])
    for i,label in enumerate(('EEG','HbO','HbR')):
        row[label+'_nrmse'] = rms(error[score[:,i],i]) if score[:,i].any() else None
    row.update(physical_error=rms((result['physical_prediction'][prediction_segment,1:]-case['physical_truth'][segment,1:])/sd[1:]),
        component_error=rms((result['observation_component'][prediction_segment,1:]-case['component_truth'][segment,1:])/sd[1:]),
        driver_error=rms(result['driver'][prediction_segment]-case['driver'][segment])/cfg['generator']['driver_sd'])
    return row


def profiles(cfg, base, case, cal, mode, reference, free):
    n = cfg['tensor']['segment_steps']
    if not all(r['converged'] for r in reference.values()):
        return [],{}
    mask = visibility(mode,n)[n:2*n].ravel()
    local,basis,_,chain_basis = operators(n)
    scale = np.tile(cal['sd'],n)
    j = local@free[1]['canonical_jacobian']
    diagnostic = grouped_component_overlap(j[mask]/scale[mask,None],basis[mask]/scale[mask,None],
        groups={'initial':np.arange(n,n+5)})['initial']['data']
    direction = np.asarray(diagnostic['coefficient_direction'])
    size = rms((basis@direction).reshape(n,3)[:,1:]/np.asarray(cal['sd'])[1:])
    if size < 1e-12:
        return [],{}
    direction /= size
    full_direction = np.r_[np.zeros(4),direction,np.zeros(4)]
    rows, arrays = [],{}
    coefficient_sd = np.tile(cfg['observation']['coefficient_sd'],3)
    for offset in cfg['profiles']['offsets_sd']:
        coefficients = {arm:r['observation_coefficients']+offset*full_direction for arm,r in reference.items()}
        fixed = {arm:(chain_basis@a).reshape(3*n,3) for arm,a in coefficients.items()}
        alternatives,_ = fit_arms(cfg,base,case,cal['sd'],mode,fixed=fixed)
        for arm,alternative in alternatives.items():
            objective = alternative.get('objective',np.nan)+float(np.sum((coefficients[arm]/coefficient_sd)**2))
            delta = objective-reference[arm]['objective']
            fraction = delta/max(reference[arm]['objective'],1e-12)
            row = dict(arm=arm,mode=mode,offset_sd=offset,objective_delta=delta,
                reference_objective=reference[arm]['objective'],objective_fraction=fraction,
                direction_reference_C_free_initial_data_cosine=max(diagnostic['cosines'],default=0.),
                near_equivalent=bool(alternative['converged'] and fraction<=cfg['profiles']['near_objective_fraction']),
                **metrics(cfg,case,alternative,np.asarray(cal['sd']),mode))
            row['objective'] = objective
            if 'prediction' in alternative:
                row.update(physical_change_sd=rms((alternative['physical_prediction'][n:2*n,1:]-reference[arm]['physical_prediction'][n:2*n,1:])/np.asarray(cal['sd'])[1:]),
                    component_change_sd=rms((fixed[arm][n:2*n,1:]-reference[arm]['observation_component'][n:2*n,1:])/np.asarray(cal['sd'])[1:]),
                    driver_change_sd=rms(alternative['driver'][n:2*n]-reference[arm]['driver'][n:2*n])/cal['driver_sd'])
                for name in ('driver','prediction','physical_prediction','observation_component'):
                    arrays[f'{mode}__{arm}__{offset:g}__{name}'] = alternative[name]
            rows.append(row)
    return rows,arrays


def evaluate_task(payload):
    cfg,base,cal,out,split,spec,with_profiles = payload
    started = time.monotonic()
    def timeout(_signum,_frame):
        raise TimeoutError('fixed task budget exhausted')
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(cfg['resources']['task_timeout_s'])
    try:
        case = synthetic_case(cfg,base,split,**spec)
        rows,profile_rows,arrays = [],[],{}
        n = cfg['tensor']['segment_steps']
        for mode in cfg['modes'] if split == 'evaluation' else ['full']:
            do_profile = with_profiles and spec['identity'] in cfg['profiles']['identities'] and mode in cfg['profiles']['modes']
            arms,free = fit_arms(cfg,base,case,np.asarray(cal['sd']),mode,jacobian=do_profile)
            for arm,result in arms.items():
                rows.append(dict(**spec,mode=mode,arm=arm,**metrics(cfg,case,result,np.asarray(cal['sd']),mode)))
                for name in ('driver','states','prediction','physical_prediction','observation_component'):
                    if name in result:
                        arrays[f'{mode}__{arm}__{name}'] = result[name]
            rows.append(dict(**spec,mode=mode,arm='target_only',alias_of='C_free_middle',
                **metrics(cfg,case,free[1],np.asarray(cal['sd']),mode,target_only=True)))
            if do_profile:
                records,trajectories = profiles(cfg,base,case,cal,mode,arms,free)
                profile_rows.extend([dict(**spec,**r) for r in records])
                arrays.update(trajectories)
        if out is not None:
            key = task_key(spec)
            destination = Path(out)/'arrays'/f'{key}.npz'
            destination.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(destination,**arrays,**{k:case[k] for k in ('target','driver','states','physical_truth','component_truth')})
        return dict(status='completed',spec=spec,rows=rows,profiles=profile_rows,
            elapsed_seconds=time.monotonic()-started,max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    except Exception as exc:
        return dict(status='timeout' if isinstance(exc,TimeoutError) else 'task_error',spec=spec,
            error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
    finally:
        signal.alarm(0)


def task_key(spec):
    return f"{spec['identity']:02d}__{spec['initial']}__{spec['scenario']}"


def plans(cfg):
    return [dict(identity=i,initial=initial,scenario=scenario)
        for i in range(cfg['evaluation_identities']) for initial in cfg['initial'] for scenario in cfg['scenarios']]


def pilot(cfg,base,cal,out):
    specs = [dict(identity=i,initial='nonrest' if i%2 else 'rest',scenario='common_colored') for i in range(8)]
    measurements = []
    for workers in (2,4,8):
        start = time.monotonic()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(evaluate_task,[(cfg,base,cal,None,'pilot',s,False) for s in specs]))
        seconds = time.monotonic()-start
        measurement = dict(workers=workers,seconds=seconds,tasks=len(specs),tasks_per_second=len(specs)/seconds,
            task_failures=sum(r['status']!='completed' for r in results),
            numerical_failures=sum(not row['converged'] for r in results for row in r.get('rows',[]) if row['arm']!='target_only'),
            max_rss_mib=max(r.get('max_rss_mib',0) for r in results),records=results)
        measurements.append(measurement)
        write_json(out/'pilot.json',dict(measurements=measurements,execution='running'))
        print(json.dumps({k:v for k,v in measurement.items() if k!='records'}),flush=True)
    selected = max(measurements,key=lambda row:row['tasks_per_second'])['workers']
    result = dict(measurements=measurements,execution='completed',selected_workers=selected,
        reason='fastest_measured_concurrency_up_to_8_while_phase1_uses_48_workers')
    write_json(out/'pilot.json',result)
    if any(r['task_failures'] for r in measurements):
        raise RuntimeError('pilot task failure; retain evidence and stop')
    return selected


def run(cfg,base,cal,out,workers):
    specs = plans(cfg)
    state = dict(experiment_id=cfg['experiment_id'],execution='running',planned=len(specs),completed=0,
        tasks=[task_key(s) for s in specs],started_at=datetime.now(timezone.utc).isoformat(),
        controller_pid=os.getpid(),workers=workers,measured_arrays_read=0)
    write_json(out/'run_manifest.json',state)
    iterator = iter(s for s in specs if not (out/'tasks'/f'{task_key(s)}.json').exists())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}
        while True:
            while len(pending)<2*workers:
                spec = next(iterator,None)
                if spec is None:
                    break
                pending[pool.submit(evaluate_task,(cfg,base,cal,str(out),'evaluation',spec,True))] = spec
            if not pending:
                break
            done,_ = wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                spec = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = dict(status='worker_crash',error=repr(exc),spec=spec)
                write_json(out/'tasks'/f'{task_key(spec)}.json',result)
            available = [read_json(out/'tasks'/f'{task_key(s)}.json') for s in specs if (out/'tasks'/f'{task_key(s)}.json').exists()]
            state.update(completed=len(available),task_failures=sum(r['status']!='completed' for r in available),
                updated_at=datetime.now(timezone.utc).isoformat())
            write_json(out/'run_manifest.json',state)
            print(json.dumps({k:state[k] for k in ('completed','planned','task_failures')}),flush=True)
    summary = summarize(cfg,out)
    state.update(execution='completed' if summary['task_failures']==0 else 'completed_with_task_failures',
        completed=len(specs),finished_at=datetime.now(timezone.utc).isoformat())
    write_json(out/'run_manifest.json',state)


def summarize(cfg,out):
    rows,profile_rows,failures = [],[],[]
    for spec in plans(cfg):
        path = out/'tasks'/f'{task_key(spec)}.json'
        result = read_json(path) if path.exists() else dict(status='missing')
        if result['status'] != 'completed':
            failures.append(dict(**spec,status=result['status']))
            rows.extend(dict(**spec,mode=mode,arm=arm,converged=False,status=result['status'])
                for mode in cfg['modes'] for arm in ('C_free','C_chain','target_only'))
        else:
            rows.extend(result['rows'])
            profile_rows.extend(result['profiles'])
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'state_continuity.csv',index=False)
    profile = pd.DataFrame(profile_rows)
    profile.to_csv(out/'confounding_profiles.csv',index=False)
    paired = []
    endpoints = ['score_nrmse','HbO_nrmse','HbR_nrmse','physical_error','component_error','driver_error']
    rng = np.random.default_rng(cfg['seed'])
    for (initial,scenario,mode),part in frame.groupby(['initial','scenario','mode']):
        a = part[part.arm=='C_free'].set_index('identity')
        b = part[part.arm=='C_chain'].set_index('identity').reindex(a.index)
        good = a.converged & b.converged
        for endpoint in endpoints:
            av = pd.to_numeric(a.get(endpoint,pd.Series(np.nan,index=a.index)),errors='coerce')
            bv = pd.to_numeric(b.get(endpoint,pd.Series(np.nan,index=b.index)),errors='coerce')
            finite = good & np.isfinite(av) & np.isfinite(bv)
            if not finite.any() and a.converged.any() and b.converged.any():
                continue
            ar = np.where(a.converged & np.isfinite(av),av,cfg['gates']['failure_penalty'])
            br = np.where(b.converged & np.isfinite(bv),bv,cfg['gates']['failure_penalty'])
            delta = br-ar
            bootstrap = np.mean(delta[rng.integers(0,len(delta),(cfg['gates']['bootstrap_repeats'],len(delta)))],axis=1)
            paired.append(dict(initial=initial,scenario=scenario,mode=mode,endpoint=endpoint,
                planned=len(a),paired_success=int(finite.sum()),independent_identities=len(a),
                free_mean=float(ar.mean()),chain_mean=float(br.mean()),delta_chain_minus_free=float(delta.mean()),
                relative_gain=float(1-br.mean()/ar.mean()) if ar.mean()>1e-12 else None,
                delta_CI95_low=float(np.quantile(bootstrap,.025)),delta_CI95_high=float(np.quantile(bootstrap,.975)),
                paired_success_delta=float((bv[finite]-av[finite]).mean()) if finite.any() else None,
                failure_rate_free=float(1-a.converged.mean()),failure_rate_chain=float(1-b.converged.mean())))
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(out/'paired_hidden_metrics.csv',index=False)
    profile_summary = []
    if not profile.empty:
        for keys,part in profile[profile.offset_sd!=0].groupby(['initial','scenario','mode','arm']):
            profile_summary.append(dict(zip(['initial','scenario','mode','arm'],keys),planned=len(part),
                converged=int(part.converged.sum()),near_equivalent=int(part.near_equivalent.sum()),
                median_objective_delta=float(part.objective_delta.median()),
                median_objective_fraction=float(part.objective_fraction.median())))
    hidden = [r for r in paired if r['mode']=='Hb_hidden' and r['endpoint']=='score_nrmse']
    truth = [r for r in paired if r['endpoint'] in ('physical_error','driver_error','component_error')]
    hidden_pass = bool(hidden) and all(r['relative_gain'] is not None and r['relative_gain']>=cfg['gates']['hidden_relative_improvement_min'] for r in hidden)
    truth_pass = bool(truth) and all(r['delta_chain_minus_free']<=cfg['gates']['synthetic_error_increase_max_sd'] for r in truth)
    failure_pass = all(r['failure_rate_chain']-r['failure_rate_free']<=cfg['gates']['failure_rate_increase_max'] for r in hidden)
    decision = dict(hidden_gate=hidden_pass,truth_noninferiority=truth_pass,failure_gate=failure_pass,
        retain='promising_synthetic_candidate_requires_measured_continuous_validation' if hidden_pass and truth_pass and failure_pass else 'do_not_expand_without_resolving_failed_controls',
        measured_confirmation=False,interpretation='offline_processed_feature_smoothing_not_forecasting_or_source_identification',
        profile_scope=cfg['profiles']['interpretation'],
        initial_prior=cfg['comparison']['initial_prior'])
    summary = dict(experiment_id=cfg['experiment_id'],tasks=len(plans(cfg)),task_failures=len(failures),failures=failures,
        numerical_failures=frame.groupby('arm').converged.apply(lambda v:int((~v).sum())).to_dict(),
        planned_rows=len(frame),independent_evaluation_identities=cfg['evaluation_identities'],
        paired=paired,profile_summary=profile_summary,decision=decision,measured_arrays_read=0)
    write_json(out/'summary.json',summary)
    write_json(out/'decision.json',decision)
    write_json(out/'verification.json',dict(passed=len(frame)==len(plans(cfg))*len(cfg['modes'])*3,
        target_only_is_alias=True,figures_created=False,measured_arrays_read=0,task_failures=len(failures),
        statistical_unit='generator_identity_conditions_paired_not_independent_subjects'))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/ssm_state_continuity_v1.yaml')
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--stage',choices=['pilot','run','summarize'],default='run')
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args()
    cfg = read_config(args.config)
    base = yaml.safe_load((CODE_ROOT/cfg['source_config']).read_text())
    if args.check_only:
        print(json.dumps(dict(passed=True,tensor=cfg['tensor'],tasks=len(plans(cfg)),measured_arrays_read=0)))
        return
    if args.run_dir is None:
        parser.error('--run-dir required')
    out = args.run_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        resolved = out/'resolved_config.yaml'
        if resolved.exists() and yaml.safe_load(resolved.read_text())!=cfg:
            raise ValueError('immutable resolved configuration mismatch')
        if not resolved.exists():
            resolved.write_text(yaml.safe_dump(cfg,sort_keys=False))
        if args.stage=='summarize':
            summarize(cfg,out)
            return
        cal = read_json(out/'calibration.json') if (out/'calibration.json').exists() else calibration(cfg,base)
        if not (out/'calibration.json').exists():
            write_json(out/'calibration.json',cal)
        if args.stage=='pilot':
            pilot(cfg,base,cal,out)
        else:
            run(cfg,base,cal,out,read_json(out/'pilot.json')['selected_workers'])


if __name__=='__main__':
    main()
