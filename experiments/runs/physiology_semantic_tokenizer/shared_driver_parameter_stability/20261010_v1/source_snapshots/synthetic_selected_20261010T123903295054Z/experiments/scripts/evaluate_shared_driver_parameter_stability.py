#!/usr/bin/env python3
"""Versioned public parameter-stability suite; durable, resumable per-task evidence."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime,timezone
import fcntl
from functools import lru_cache
import itertools
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import yaml
from src.inference.shared_driver_stability import (
    PARAMETERS,REFERENCE,BOUNDS,parameter_object,boundary_distance,symmetric_relative,
    icc_absolute,bootstrap_icc,estimate_working_covariance,fit_record,compact_fit,warm_start,
    profile_pair,profile_intersections,synthetic_pair,oracle_fit,record_arrays,
)
from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
from src.inference.observation_baselines import native_feature_operators,eeg_band_power
from src.data.ssm_prepared import fit_feature_coordinate
DEFAULT=ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_parameter_stability_v1.yaml'
RETRY_FAILED=False
SYNTHETIC_TRAINING_SUBJECTS=8


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def read(path):return json.loads(Path(path).read_text())


def active_models(out):
    decision=Path(out)/'dimension_decision.json'
    return ['P1_tau','P1_beta','P2']+(['P3'] if decision.exists() and read(decision)['run_P3'] else [])


def config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if (cfg['schema']!='shared_driver_parameter_stability_v1' or cfg['data']['protected_data']!='forbidden'
        or cfg['tensor']!=dict(dt_s=.25,block_steps=120,blocks=3,duration_s=90,state_order=['r','s','f','v','p','q'],
                              components=['EEG_logpower_PCA','relative_HbO','relative_HbR'])
        or cfg['data']['labels_as_model_inputs'] or cfg['resources']['numerical_threads']!=1):
        raise ValueError('invalid stability scope/tensor/data contract')
    for name,bounds in zip(PARAMETERS,BOUNDS):
        if cfg['parameters'][name]['bounds']!=bounds.tolist():raise ValueError('original parameter bounds must stay fixed')
    return cfg


@lru_cache(maxsize=1)
def base_config(path):return yaml.safe_load((ROOT/path).read_text())


def pair_candidates(spec, events, anchor, site, seconds=90):
    """Metadata-only fixed-duration candidates. Labels are pairing metadata only."""
    from src.data.event_alignment import window_within_alignment_support
    result=[];de=spec['array_shapes']['eeg'][0]/200.;dh=spec['array_shapes']['fnirs'][0]/10.
    ordered=sorted([e for e in events if e.get('eeg_time_ms') is not None and e.get('fnirs_time_ms') is not None],key=lambda e:e['eeg_time_ms'])
    for e in ordered:
        es=e['eeg_time_ms']/1000.-5.;hs=e['fnirs_time_ms']/1000.-5.
        if min(es,hs)<0 or es+seconds>de or hs+seconds>dh or not window_within_alignment_support(e,-5.,seconds):continue
        within=[q for q in ordered if es<=q['eeg_time_ms']/1000.<es+seconds]
        conditions=Counter(str(q['metadata'].get('condition_label',q['label'])) for q in within)
        if 'unknown' in conditions or not conditions:continue
        task=e['metadata'].get('task');signature=tuple(sorted(conditions.items()))
        result.append(dict(key=spec['key'],join_key=spec['join_key'],record=spec['record'],
            dataset=spec['dataset'],subject=spec['subject'],site=site,region=anchor['region'],
            task=task,event_id=e['event_index'],eeg_start_s=es,hb_start_s=hs,duration_s=seconds,
            condition_counts=dict(signature),anchor=anchor,signature=signature,
            alignment_support=e['metadata'].get('alignment_support_ms'),
            native_session=e['metadata'].get('session_idx'),source_files=spec['source_files']))
    return result


def select_pair(candidates, priorities):
    """Prefer independent records, then nonoverlapping same-record blocks."""
    for independent in (True,False):
        for task in priorities:
            rows=sorted([r for r in candidates if r['task']==task],key=lambda r:(r['record'],r['eeg_start_s']))
            bysignature=defaultdict(list)
            for r in rows:
                sig=(r['signature'],tuple(r['anchor']['eeg_channels']),r['anchor']['hb_channel'])
                for previous in bysignature[sig]:
                    distinct=previous['record']!=r['record']
                    if independent!=distinct:continue
                    if not distinct and previous['eeg_start_s']+previous['duration_s']>r['eeg_start_s']+1e-9:continue
                    # Probe1 and Probe2 are different locations, not repeat recordings.
                    if distinct and r['dataset']=='visual_cognitive_motivation':
                        if previous['record'].split('_Probe')[-1]!=r['record'].split('_Probe')[-1]:continue
                    return [previous,r],('independent_native_record' if distinct else 'within_record_disjoint_blocks')
                bysignature[sig].append(r)
    return None,'no_exact_task_condition_duration_montage_pair'


def plan(cfg,out):
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    parent=ROOT/cfg['source_run'];inventory=read(parent/'inventory.json');diagnostic=read(parent/'diagnostic_plan.json')
    cohort=read(parent/'cohort.json')['refs']
    development={ds:diagnostic['selected_subjects'][ds] for ds in cfg['data']['datasets']}
    index=CleanPhysiologyCacheIndex(ROOT/cfg['data']['cache_root'])
    sites={(r['key'],r['region']):r['site'] for r in cohort}
    groups=defaultdict(list)
    for spec in inventory['records']:
        if spec['dataset'] not in development:continue
        if spec['dataset']=='visual_cognitive_motivation' and spec['subject']=='S06' and 'Part1' in spec['record']:continue
        for anchor in spec['anchors']:
            site=sites.get((spec['key'],anchor['region']))
            if site is None:continue
            rows=pair_candidates(spec,index.events_by_join_key.get(spec['join_key'],[]),anchor,site)
            groups[(spec['dataset'],spec['subject'],site)].extend(rows)
    pairs=[];excluded=[]
    for (ds,subject,site),candidates in sorted(groups.items()):
        pair,kind=select_pair(candidates,cfg['data']['task_priority'][ds])
        identity=dict(id=f'{ds}__{subject}__{site}',dataset=ds,subject=subject,site=site,
                      partition='development' if subject in development[ds] else 'evaluation')
        if pair is None:excluded.append(dict(**identity,reason=kind));continue
        pairs.append(dict(**identity,repeat_kind=kind,records=pair))
    subjects={ds:sorted({r['subject'] for r in inventory['records'] if r['dataset']==ds}) for ds in development}
    # The panel retains one spatial coordinate per development subject, using
    # a metadata preference. Boundary/interior coverage is audited after N0.
    panel=[]
    for ds in development:
        for subject in development[ds]:
            available=[p for p in pairs if p['dataset']==ds and p['subject']==subject]
            if available:panel.append(sorted(available,key=lambda p:(not p['site'].startswith('prefrontal'),p['site']))[0]['id'])
    value=dict(schema='parameter_stability_pair_plan_v1',development_subjects=development,
        candidate_subjects={ds:[s for s in subjects[ds] if s not in development[ds]] for ds in development},
        pairs=pairs,excluded=excluded,profile_panel=panel,measured_arrays_read=0,
        pairing='exact task, condition onset counts, native channel identity, duration; no fitted values',
        primary='independent_native_record',auxiliary='within_record_disjoint_blocks',
        historical_confirmatory_status='previously_examined_public_subjects_not_fresh_confirmation')
    write(out/'pair_plan.json',value)
    print('PLAN',json.dumps({ds:dict(development=len(development[ds]),candidate=len(value['candidate_subjects'][ds]),
        pairs=Counter(p['partition'] for p in pairs if p['dataset']==ds),kinds=Counter(p['repeat_kind'] for p in pairs if p['dataset']==ds)) for ds in development}),flush=True)
    return value


@lru_cache(maxsize=1)
def cache_index(root):
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    return CleanPhysiologyCacheIndex(Path(root))


@lru_cache(maxsize=2)
def native_record(cache_root,join_key):
    from src.data.unified_physiology import load_native_eeg_record,load_native_fnirs_record
    index=cache_index(cache_root);record=next(r for r in index.records if r.join_key==join_key)
    return load_native_eeg_record(ROOT,record),load_native_fnirs_record(ROOT,record)


def prepare_worker(payload):
    cfg,out,pair=payload;out=Path(out);began=time.monotonic()
    arrays={};details=[]
    for side,spec in enumerate(pair['records']):
        eeg,hb=native_record(str(ROOT/cfg['data']['cache_root']),spec['join_key'])
        channels={str(c).upper():i for i,c in enumerate(eeg.channel_names)}
        selected=[channels[n.upper()] for n in spec['anchor']['eeg_channels']]
        op=native_feature_operators(120);features=[];hemo=[]
        for block in range(3):
            es=spec['eeg_start_s']+30*block;hs=spec['hb_start_s']+30*block
            start=round(es*eeg.sample_rate_hz);length=round(30*eeg.sample_rate_hz)
            signal=eeg.values[start:start+length,selected]
            times=hs+np.arange(300)/10.;native=hb['values'][:,spec['anchor']['hb_pair']]
            if times[0]<hb['time_s'][0] or times[-1]>hb['time_s'][-1]:raise ValueError('Hb native support overrun')
            h=np.column_stack([np.interp(times,hb['time_s'],native[:,j]) for j in range(2)])
            if signal.shape!=(length,6) or not np.isfinite(signal).all() or not np.isfinite(h).all():raise ValueError('nonfinite native support')
            power=eeg_band_power(signal,bands=base_config(cfg['source_config'])['tensor']['eeg_bands_hz'],sample_rate=eeg.sample_rate_hz,target_rate=4.)
            if power.shape!=(120,6,5) or np.any(power<=0):raise ValueError('power shape/positivity')
            features.append(op['eeg']@np.log(power).reshape(120,30));hemo.append(op['fnirs']@h)
        arrays[f'eeg_{side}']=np.array(features);arrays[f'hb_{side}']=np.array(hemo)
        details.append(dict(native_eeg_source=str(eeg.source_path),native_eeg_rate=eeg.sample_rate_hz,
            native_eeg_unit=eeg.native_unit,native_hb_unit=hb.get('native_unit'),
            clock_rounding_error_s=round(spec['eeg_start_s']*eeg.sample_rate_hz)/eeg.sample_rate_hz-spec['eeg_start_s']))
    path=out/'prepared'/(pair['id']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)
    return dict(status='completed',pair_id=pair['id'],seconds=time.monotonic()-began,provenance=details,
                feature_shapes={k:v.shape for k,v in arrays.items()},reference='first_5s_of_each_30s_block_not_assumed_rest')


def coordinate_key(pair):return pair['dataset']+'__'+pair['site']


def fit_coordinates(cfg,out,plan):
    coordinates={};prepared={p['id']:read(out/'prepare'/(p['id']+'.json')) for p in plan['pairs']}
    for key in sorted({coordinate_key(p) for p in plan['pairs']}):
        training=[p for p in plan['pairs'] if coordinate_key(p)==key and p['partition']=='development' and prepared[p['id']]['status']=='completed']
        def chunks():
            for p in training:
                with np.load(out/'prepared'/(p['id']+'.npz')) as a:
                    for side in range(2):yield a[f'eeg_{side}'],a[f'hb_{side}']
        if not training:raise ValueError(f'no development coordinate for {key}')
        co=fit_feature_coordinate(chunks())
        co.update(key=key,training_pair_ids=[p['id'] for p in training],training_subjects=sorted({p['subject'] for p in training}),
                  training_record_ids=[r['join_key'] for p in training for r in p['records']])
        coordinates[key]=co
    write(out/'coordinates.json',coordinates)
    return coordinates


@lru_cache(maxsize=16)
def target_pair(out,pair_id,key):
    co=read(Path(out)/'coordinates.json')[key]
    with np.load(Path(out)/'prepared'/(pair_id+'.npz')) as a:
        y=np.array([np.column_stack((a[f'eeg_{s}'].reshape(-1,30)@np.asarray(co['pc'])*co['eeg_factor'],
                     a[f'hb_{s}'].reshape(-1,2)*co['hb_factor'])) for s in range(2)])
    return y,np.asarray(co['sd'])


def save_fit_arrays(out,folder,key,fits):
    arrays={}
    for side,fit in enumerate(fits):
        for name in ('driver','initial_state','prediction','states'):
            if name in fit:arrays[f'{name}_{side}']=fit[name]
    path=Path(out)/folder/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)


def measured_worker(payload):
    cfg,out,pair,model,arm,variant,calibration=payload
    targets,sd=target_pair(out,pair['id'],coordinate_key(pair));base=base_config(cfg['source_config'])
    covariance=calibration.get('covariance') if calibration else None
    prior=calibration.get('prior') if calibration else None
    fixed=calibration.get('fixed') if calibration else None
    fits=[]
    for side,target in enumerate(targets):
        fits.append(fit_record(target,sd,cfg,base,'P0' if fixed is not None else model,
            arm=arm,covariance=covariance,prior=prior,fixed=fixed,
            n0=(variant=='N0' or (variant=='Hselected' and prior is None))))
    key='__'.join((pair['id'],model,variant));save_fit_arrays(out,'fits',key,fits)
    transferred=[]
    for side in range(2):
        if 'parameters' not in fits[side]:transferred.append(dict(status='source_fit_failed',accepted=False));continue
        other=1-side
        transfer=fit_record(targets[other],sd,cfg,base,'P0',arm=arm,fixed=fits[side]['parameters'],covariance=covariance)
        # A transfer uses only theta from the other record. Its driver and
        # initial state are independently re-estimated from destination data.
        transferred.append(compact_fit(transfer))
    selection_transfer=[]
    for side,own in (calibration.get('record_folds',{}).items() if calibration else []):
        side=int(side)
        training=fit_record(targets[side],sd,cfg,base,'P0' if 'fixed' in own else model,
            prior=own.get('prior'),fixed=own.get('fixed'))
        if 'parameters' in training:
            applied=fit_record(targets[1-side],sd,cfg,base,'P0',fixed=training['parameters'])
            selection_transfer.append(compact_fit(applied))
        else:selection_transfer.append(dict(status='failed',accepted=False))
    return dict(status='completed',pair_id=pair['id'],dataset=pair['dataset'],subject=pair['subject'],site=pair['site'],
        repeat_kind=pair['repeat_kind'],partition=pair['partition'],model=model,variant=variant,arm=arm,
        fits=[compact_fit(f) for f in fits],transfer=transferred,
        selection_transfer=selection_transfer,
        transfer_interpretation='parameter_transfer_conditioned_reconstruction_not_unconditional_prediction',
        seconds=sum(f['seconds'] for f in fits)+sum(f.get('seconds',0) for f in transferred))


def synthetic_worker(payload):
    cfg,out,subject,level,model=payload;base=base_config(cfg['source_config']);data=synthetic_pair(cfg,base,subject,level,model)
    results={};names=cfg['models'][model]
    for arm in ('C0a','C0b','C1'):
        fits=[fit_record(target,data['sd'],cfg,base,model,arm=arm,n0=arm=='C0a') for target in data['target']]
        results[arm]=[compact_fit(f) for f in fits]
    oracle=[oracle_fit(data['target'][s],data['driver'][s],data['initial'][s],data['sd'],cfg,base,names) for s in range(2)]
    return dict(status='completed',subject=subject,level=level,model=model,truth=dict(zip(PARAMETERS,data['truth'])),
        fits=results,oracle=oracle,truth_kind='known_generator_parameters',independent_records=2,
        fixed_reference=dict(zip(PARAMETERS,REFERENCE)))


def profile_worker(payload):
    cfg,out,pair,model,objective=payload;targets,sd=target_pair(out,pair['id'],coordinate_key(pair))
    names=cfg['models'][model]
    baseline=read(Path(out)/'development'/('__'.join((pair['id'],model,'N0'))+'.json'))
    candidates=[[f['parameters'][name] for name in names] for f in baseline['fits'] if 'parameters' in f]
    candidates.append([REFERENCE[PARAMETERS.index(name)] for name in names])
    result=profile_pair(targets,sd,cfg,base_config(cfg['source_config']),names,objective=objective,candidate_values=candidates)
    if objective=='current_without_parameter_shrinkage':result['independent_reference']=baseline['fits']
    reference=read(Path(out)/'development'/('__'.join((pair['id'],'P0','N0'))+'.json'))
    denominator=[f['objective'] for f in reference['fits']]
    result.update(pair_id=pair['id'],dataset=pair['dataset'],subject=pair['subject'],site=pair['site'],model=model,
        intersections=profile_intersections(result,denominator,nrmse_margin=cfg['profile']['per_record_modality_nrmse_margin']))
    return result


def crossfit_worker(payload):
    cfg,out,pair=payload;targets,sd=target_pair(out,pair['id'],coordinate_key(pair));base=base_config(cfg['source_config'])
    residuals=[];counts=[]
    for target in targets:
        prediction=np.full_like(target,np.nan)
        for fold in range(3):
            mask=np.ones((3,120,3),bool);mask[:,fold*40:(fold+1)*40]=False
            fit=fit_record(target,sd,cfg,base,'P0',visible=mask)
            if 'prediction' not in fit:raise ValueError('crossfit conditional prediction failed')
            pred=np.asarray(fit['prediction']).reshape(-1,3);hidden=~mask.reshape(-1,3)
            prediction[hidden]=pred[hidden];counts.append(fit['accepted'])
        residuals.extend(((prediction-target)/sd).reshape(3,120,3))
    path=Path(out)/'crossfit'/(pair['id']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,residual=np.asarray(residuals))
    return dict(status='completed',pair_id=pair['id'],successful_folds=sum(counts),expected_folds=6,
                hidden_support='40_contiguous_samples_per_120_sample_block_all_components',coordinate_key=coordinate_key(pair))


def calibration_worker(payload):
    cfg,out,key,pairs,model=payload;out=Path(out);base=base_config(cfg['source_config'])
    residuals=[]
    for pair in pairs:
        path=out/'crossfit'/(pair['id']+'.npz')
        if path.exists():
            with np.load(path) as a:residuals.extend(a['residual'])
    wc=estimate_working_covariance(residuals,shrinkage=cfg['weighting']['shrinkage'],floor=cfg['weighting']['eigenvalue_floor'],ar_clip=cfg['weighting']['ar1_clip'])
    models={}
    for model in [model]:
        names=cfg['models'][model];ys=[];nuisance=[];values=[];sd=None
        for pair in pairs:
            y,sd=target_pair(str(out),pair['id'],key);ys.append(y)
            record=read(out/'development'/('__'.join((pair['id'],model,'N0'))+'.json'))
            with np.load(out/'fits'/('__'.join((pair['id'],model,'N0'))+'.npz')) as a:
                nuisance.append([dict(driver=a[f'driver_{side}'].copy(),initial_state=a[f'initial_state_{side}'].copy()) for side in range(2)])
            values.append([[f['parameters'][n] for n in names] for f in record['fits']])
        values=np.asarray(values);folds={}
        for label,sides in [('all',[0,1]),('0',[0]),('1',[1])]:
            all_y=np.concatenate([y[side] for y in ys for side in sides])
            mean0=np.mean(np.log(values[:,sides]),axis=(0,1))
            starts=[x[side] for x in nuisance for side in sides]
            start=dict(parameter_values=np.exp(mean0),driver=np.concatenate([s['driver'] for s in starts]),
                       initial_state=np.concatenate([s['initial_state'] for s in starts]))
            pooled=fit_record(all_y,sd,cfg,base,model,
                starts=[start,dict(parameter_values=np.exp(mean0))],profile=True)
            mu=np.log([pooled['parameters'][n] for n in names]);logs=np.log(values)
            if label=='all':variance=np.maximum(np.mean((logs[:,0]-mu)*(logs[:,1]-mu),axis=0),.01)
            else:variance=np.maximum(np.mean((logs[:,int(label)]-mu)**2,axis=0),.01)
            folds[label]=dict(mean=dict(zip(names,mu)),variance=dict(zip(names,variance)),pooled_fit=compact_fit(pooled),
                             training_repeat_indices=sides)
        models[model]=dict(**folds['all'],record_folds={s:folds[s] for s in ('0','1')},
            training_pairs=[p['id'] for p in pairs],
            scale_interpretation='regularization_working_log_scale_not_a_physiological_normal_distribution')
    return dict(status='completed',covariance=wc,models=models,training_subjects=sorted({p['subject'] for p in pairs}))


def calibration(cfg,out,plan,workers):
    coordinates=sorted({coordinate_key(p) for p in plan['pairs']});jobs=[];keys=[]
    for key in coordinates:
        pairs=[p for p in plan['pairs'] if p['partition']=='development' and coordinate_key(p)==key and (out/'prepared'/(p['id']+'.npz')).exists()]
        for model in active_models(out):
            jobs.append((cfg,str(out),key,pairs,model));keys.append(key+'__'+model)
    run_tasks(cfg,out,'calibration_components',calibration_worker,jobs,keys,workers)
    result={}
    for job,task_key in zip(jobs,keys):
        value=read(out/'calibration_components'/(task_key+'.json'))
        if value.get('status')=='failed':raise ValueError('development calibration failed; evaluation stays closed')
        key=job[2]
        if key not in result:result[key]=dict(value,models={})
        result[key]['models'].update(value['models'])
    write(out/'development_calibration.json',result);return result


def shrinkage_calibration(cal,model,strength,*,development=False):
    def one(fit):
        if strength=='full':return dict(fixed={k:float(np.exp(v)) for k,v in fit['mean'].items()})
        return dict(prior=dict(mean=fit['mean'],precision={k:float(strength)/v for k,v in fit['variance'].items()}))
    result=one(cal['models'][model])
    if development:result['record_folds']={side:one(v) for side,v in cal['models'][model]['record_folds'].items()}
    return result


def method_jobs(cfg,out,pairs,cal):
    jobs=[]
    for pair in pairs:
        own=cal[coordinate_key(pair)]
        for model in active_models(out):
            for arm in ('C0b','C1'):jobs.append((cfg,str(out),pair,model,arm,arm,None))
            jobs.append((cfg,str(out),pair,model,'C0a','W1',dict(covariance=own['covariance'])))
            for strength in (.25,1.,4.,'full'):
                jobs.append((cfg,str(out),pair,model,'C0a','H'+str(strength),shrinkage_calibration(own,model,strength,development=True)))
    return jobs


def result_metrics(result,cfg,baseline=None):
    names=cfg['models'][result['model']];fits=result.get('fits',[]);trans=result.get('transfer',[])
    row={k:result.get(k) for k in ('pair_id','dataset','subject','site','partition','repeat_kind','model','variant','arm')}
    row.update(completed=False,accepted=False,near_01=np.nan,near_05=np.nan,near_10=np.nan,exact_boundary=np.nan,
               raw_fit_pass=False,transfer_pass=False,transfer_accepted=False,accepted_record_fraction=0.,transfer_record_fraction=0.,
               repeat_error=np.nan,effective_pair=False,individual_parameter_model=bool(names) and result.get('variant')!='Hfull')
    if not names:
        for field in ('near_01','near_05','near_10','exact_boundary'):row[field]=np.nan
    if len(fits)!=2 or not all('parameters' in f for f in fits):return row
    row['completed']=True;row['accepted']=all(f.get('accepted',False) for f in fits)
    row['accepted_record_fraction']=sum(f.get('accepted',False) for f in fits)/2
    row['transfer_record_fraction']=sum(f.get('accepted',False) for f in trans)/2
    values=np.array([[f['parameters'][name] for name in names] for f in fits])
    if names:
        distance=boundary_distance(values,np.array([cfg['parameters'][n]['bounds'] for n in names]))
        row.update(near_01=bool((distance<.01).any()),near_05=bool((distance<.05).any()),near_10=bool((distance<.1).any()),
                   exact_boundary=bool((distance<1e-8).any()),repeat_error=float(np.max(symmetric_relative(*values))))
        for j,name in enumerate(names):
            row[name+'_A']=values[0,j];row[name+'_B']=values[1,j];row[name+'_repeat']=float(symmetric_relative(*values[:,j]))
    for j,label in enumerate(('EEG','HbO','HbR')):
        row[label+'_nrmse']=float(np.mean([f['nrmse'][j] for f in fits]))
        row[label+'_transfer']=float(np.mean([f['nrmse'][j] for f in trans])) if len(trans)==2 and all('nrmse' in f for f in trans) else np.nan
    row['transfer_accepted']=len(trans)==2 and all(f.get('accepted',False) for f in trans)
    row['seconds']=result.get('seconds',np.nan)
    original=baseline or result
    raw_ok=False;transfer_ok=False
    if len(original.get('fits',[]))==2 and all('nrmse' in f for f in original['fits']):
        raw_ok=bool((np.array([f['nrmse'] for f in fits])-np.array([f['nrmse'] for f in original['fits']])<=.01+1e-12).all())
    if len(original.get('transfer',[]))==2 and all('nrmse' in f for f in original['transfer']) and all('nrmse' in f for f in trans) and len(trans)==2:
        transfer_ok=bool((np.array([f['nrmse'] for f in trans])-np.array([f['nrmse'] for f in original['transfer']])<=.01+1e-12).all())
    row.update(raw_fit_pass=raw_ok,transfer_pass=transfer_ok,
        effective_pair=bool(row['individual_parameter_model'] and row['accepted'] and row['transfer_accepted'] and not row['near_05'] and row['repeat_error']<=.2 and raw_ok and transfer_ok))
    return row


def synthetic_summary(cfg,out):
    rows=[]
    for path in sorted((out/'synthetic').glob('*.json')):
        d=read(path)
        if d.get('status')!='completed':continue
        for variant,fits in d['fits'].items():
            for side,f in enumerate(fits):
                for name in cfg['models'][d['model']]:
                    true=d['truth'][name];value=f.get('parameters',{}).get(name,np.nan)
                    rows.append(dict(subject=d['subject'],level=d['level'],model=d['model'],variant=variant,side=side,
                        parameter=name,truth=true,value=value,relative_error=abs(value/true-1),accepted=f.get('accepted',False)))
        for side,f in enumerate(d['oracle']):
            for name,value in zip(cfg['models'][d['model']],f['values']):
                true=d['truth'][name]
                rows.append(dict(subject=d['subject'],level=d['level'],model=d['model'],variant='oracle',side=side,
                    parameter=name,truth=true,value=value,relative_error=abs(value/true-1),accepted=f['converged']))
        for name in cfg['models'][d['model']]:
            value=d['fixed_reference'][name];true=d['truth'][name]
            for side in range(2):rows.append(dict(subject=d['subject'],level=d['level'],model=d['model'],variant='fixed',side=side,
                parameter=name,truth=true,value=value,relative_error=abs(value/true-1),accepted=True))
    frame=pd.DataFrame(rows);frame.to_csv(out/'synthetic_estimates.csv',index=False);summary=[]
    for (model,variant,parameter,level),g in frame.groupby(['model','variant','parameter','level']):
        paired=g.pivot(index='subject',columns='side',values='value')
        between=g.groupby('subject')[['truth','value']].mean()
        slope=float(np.polyfit(np.log(between.truth),np.log(between.value),1)[0]) if np.std(np.log(between.truth))>1e-8 else None
        expected=cfg['synthetic']['subjects']
        summary.append(dict(model=model,variant=variant,parameter=parameter,level=int(level),n_subjects=len(between),
            planned_subjects=expected,generation_or_task_failures=expected-len(between),
            convergence=float(g.accepted.sum()/(2*expected)),conditional_convergence=float(g.accepted.mean()),median_relative_error=float(g.relative_error.median()),
            p90_relative_error=float(g.relative_error.quantile(.9)),log_recovery_slope=slope,
            repeat_median=float(np.median(symmetric_relative(paired[0],paired[1]))),
            between_log_variance=float(np.var(np.log(between.value),ddof=1)),
            truth_log_variance=float(np.var(np.log(between.truth),ddof=1)),
            vector_scoring='all released parameters must pass, never best parameter only'))
    write(out/'synthetic_summary.json',summary);return summary


def select_methods(cfg,out,pp):
    cal=read(out/'development_calibration.json');models=active_models(out);selection={};details=[]
    dev=[p for p in pp['pairs'] if p['partition']=='development' and (out/'prepared'/(p['id']+'.npz')).exists()]
    strengths={}
    for key in sorted(cal):
        for model in models:
            scores=[]
            for strength in (0,.25,1.,4.):
                variant='N0' if strength==0 else 'H'+str(float(strength))
                records=[]
                for p in dev:
                    if coordinate_key(p)!=key:continue
                    directory='development' if strength==0 else 'development_methods'
                    d=read(out/directory/('__'.join((p['id'],model,variant))+'.json'))
                    fits=d.get('selection_transfer') or d.get('transfer',[])
                    score=float(np.mean([f['nrmse'] for f in fits])) if len(fits)==2 and all(f.get('accepted') for f in fits) else float('inf')
                    records.append(score)
                working=cal[key]['models'][model]
                center_accepted=bool(working['pooled_fit']['accepted'] and
                    all(f['pooled_fit']['accepted'] for f in working['record_folds'].values()))
                admissible=strength==0 or center_accepted
                scores.append(dict(strength=strength,subject_equal_transfer_nrmse=float(np.mean(records)) if admissible else float('inf'),
                    observed_transfer_nrmse=float(np.mean(records)),admissible=admissible,
                    reason='numerically_accepted_training_centers' if admissible else 'unconverged_training_center'))
            best=min(s['subject_equal_transfer_nrmse'] for s in scores)
            chosen=next((s['strength'] for s in scores if s['subject_equal_transfer_nrmse']<=best+cfg['sharing']['tie_tolerance']),0)
            strengths[(key,model)]=chosen
            details.append(dict(coordinate=key,model=model,chosen_strength=chosen,scores=scores))
    # Select at most two single changes across all development subjects for each
    # released parameter model. Evaluation identities/results never enter here.
    for model in models:
        candidates=[]
        for variant in ('C1','W1','Hselected'):
            rows=[]
            for p in dev:
                base=read(out/'development'/('__'.join((p['id'],model,'N0'))+'.json'))
                chosen=strengths[(coordinate_key(p),model)]
                actual=('N0' if chosen==0 else 'H'+str(float(chosen))) if variant=='Hselected' else variant
                d=base if actual=='N0' else read(out/'development_methods'/('__'.join((p['id'],model,actual))+'.json'))
                a=result_metrics(base,cfg);b=result_metrics(d,cfg,base)
                rows.append(dict(subject=p['dataset']+'|'+p['subject'],repeat_gain=a['repeat_error']-b['repeat_error'],
                    effective_gain=float(b['effective_pair'])-float(a['effective_pair']),
                    raw_pass=b.get('raw_fit_pass',False),transfer_pass=b.get('transfer_pass',False),accepted=b['accepted']))
            g=pd.DataFrame(rows).groupby('subject').mean(numeric_only=True)
            metrics=dict(variant=variant,repeat_gain=float(g.repeat_gain.median()),effective_gain=float(g.effective_gain.mean()),
                raw_pass_fraction=float(g.raw_pass.mean()),transfer_pass_fraction=float(g.transfer_pass.mean()),convergence_pair_fraction=float(g.accepted.mean()))
            # Engineering selection keeps improvements worth testing even when
            # absolute physiological qualification fails; that verdict is separate.
            metrics['retain']=bool(metrics['repeat_gain']>.01 and metrics['raw_pass_fraction']>=.8 and metrics['transfer_pass_fraction']>=.8)
            candidates.append(metrics)
        retained=sorted([c for c in candidates if c['retain']],key=lambda c:(-c['effective_gain'],-c['repeat_gain'],c['variant']))[:2]
        selection[model]=dict(singles=[r['variant'] for r in retained],candidates=candidates,
            combine=len(retained)==2,sharing_strengths={key:strengths[(key,model)] for key in cal})
    synth=synthetic_summary(cfg,out)
    p2=[r for r in synth if r['model']=='P2' and r['variant']=='C0a' and r['level']>0]
    recovery=bool(p2) and all(r['convergence']>=.95 and r['median_relative_error']<=.1 and r['p90_relative_error']<=.25
        and r['log_recovery_slope'] is not None and .8<=r['log_recovery_slope']<=1.2 for r in p2)
    ratios=[]
    for p in dev:
        d=read(out/'development'/('__'.join((p['id'],'P2','N0'))+'.json'))
        for f in d.get('fits',[]):
            eig=np.asarray(f.get('information',{}).get('data_only',{}).get('eigenvalues',[]),float)
            if len(eig)==2:ratios.append(eig[0]/max(eig[-1],1e-20))
    p3=bool(recovery and ratios and np.median(ratios)>1e-4)
    value=dict(schema='parameter_stability_frozen_selection_v1',status='frozen_before_evaluation',
        models=selection,sharing_development_leave_record_scores=details,
        P3_trigger=dict(run=p3,synthetic_P2_pass=recovery,median_effective_information_ratio=float(np.median(ratios)) if ratios else None),
        development_subjects=pp['development_subjects'],evaluation_results_read=0,
        selection_margins=dict(repeat_gain_min=.01,per_pair_raw_and_transfer_pass_fraction_min=.8,max_singles=2),
        selection_is_not_physiological_qualification=True)
    path=out/'frozen_selection.json'
    if path.exists():
        if read(path)!=clean(value):raise ValueError('frozen selection differs; retained evaluation cannot be reselected')
        return read(path)
    write(path,value);return value


def evaluation_jobs(cfg,out,pairs,selection,cal):
    jobs=[]
    for p in pairs:
        jobs.append((cfg,str(out),p,'P0','C0a','N0',None))
        for model,spec in selection['models'].items():
            key=coordinate_key(p);own=cal[key];strength=spec['sharing_strengths'][key]
            jobs.append((cfg,str(out),p,model,'C0a','N0',None))
            jobs.append((cfg,str(out),p,model,'C0a','Hfull',shrinkage_calibration(own,model,'full')))
            def options(variants):
                arm='C1' if 'C1' in variants else 'C0a';settings={}
                if 'W1' in variants:settings['covariance']=own['covariance']
                if 'Hselected' in variants and strength>0:settings.update(shrinkage_calibration(own,model,strength))
                return arm,settings or None
            for variant in spec['singles']:
                arm,settings=options([variant]);jobs.append((cfg,str(out),p,model,arm,variant,settings))
            if spec['combine']:
                arm,settings=options(spec['singles']);jobs.append((cfg,str(out),p,model,arm,'combined',settings))
    return jobs


def selected_synthetic_training_sample_worker(payload):
    cfg,out,model,level,subject=payload;out=Path(out);base=base_config(cfg['source_config'])
    data=synthetic_pair(cfg,base,subject,level,model)
    fits=[fit_record(t,data['sd'],cfg,base,model,n0=True) for t in data['target']]
    if not all('parameters' in f for f in fits):return dict(status='failed',reason='training_fit_domain',subject=subject)
    arrays=dict(target=data['target'],sd=data['sd'])
    for side,f in enumerate(fits):
        arrays['driver_'+str(side)]=f['driver'];arrays['initial_state_'+str(side)]=f['initial_state']
    np.savez_compressed(out/'synthetic_selected_training_samples'/f'{model}__l{level}__s{subject:02d}.npz',**arrays)
    return dict(status='completed',subject=subject,level=level,model=model,fits=[compact_fit(f) for f in fits])


def selected_synthetic_training_worker(payload):
    cfg,out,model,level=payload;out=Path(out);base=base_config(cfg['source_config']);names=cfg['models'][model]
    targets=[];nuisance=[];values=[];failed=[]
    # Eight extra training identities are disjoint from the 32 scored identities.
    for subject in range(cfg['synthetic']['subjects'],cfg['synthetic']['subjects']+SYNTHETIC_TRAINING_SUBJECTS):
        path=out/'synthetic_selected_training_samples'/f'{model}__l{level}__s{subject:02d}.json'
        sample=read(path)
        if sample.get('status')!='completed':failed.append(dict(subject=subject,error=sample.get('error',sample.get('reason'))));continue
        fits=sample['fits']
        with np.load(path.with_suffix('.npz')) as a:
            targets.extend(a['target'])
            for side,f in enumerate(fits):nuisance.append(dict(driver=a['driver_'+str(side)],initial_state=a['initial_state_'+str(side)]))
        values.append([[f['parameters'][n] for n in names] for f in fits])
    if len(values)<3:raise ValueError('insufficient synthetic-only training subjects')
    values=np.asarray(values);mean0=np.log(values).mean(axis=(0,1))
    start=dict(parameter_values=np.exp(mean0),driver=np.concatenate([f['driver'] for f in nuisance]),
               initial_state=np.concatenate([f['initial_state'] for f in nuisance]))
    pooled=fit_record(np.concatenate(targets),np.full(3,.025),cfg,base,model,profile=True,
        starts=[start,dict(parameter_values=np.exp(mean0))])
    mu=np.log([pooled['parameters'][n] for n in names]);logs=np.log(values)
    variance=np.maximum(np.mean((logs[:,0]-mu)*(logs[:,1]-mu),axis=0),.01)
    return dict(status='completed',model=model,level=level,training_subject_ids=list(range(cfg['synthetic']['subjects'],cfg['synthetic']['subjects']+SYNTHETIC_TRAINING_SUBJECTS)),
        available_training_subjects=len(values),failures=failed,mean=dict(zip(names,mu)),variance=dict(zip(names,variance)),
        pooled_fit=compact_fit(pooled),truth_values_used_for_training=False)


def selected_synthetic_worker(payload):
    cfg,out,model,level,subject,strength,arm=payload;out=Path(out);base=base_config(cfg['source_config'])
    training=read(out/'synthetic_selected_training'/(model+'__l'+str(level)+'.json'))
    original=read(out/'synthetic'/f'{model}__l{level}__s{subject:02d}.json')
    if original.get('status')!='completed':return dict(status='generation_unavailable',subject=subject,level=level,model=model,
        source_failure=str(out/'synthetic'/f'{model}__l{level}__s{subject:02d}.json'))
    data=synthetic_pair(cfg,base,subject,level,model)
    prior=dict(mean=training['mean'],precision={n:strength/v for n,v in training['variance'].items()})
    fits=[fit_record(t,data['sd'],cfg,base,model,arm=arm,prior=prior) for t in data['target']]
    return dict(status='completed',subject=subject,level=level,model=model,variant=f'H{strength:g}_'+arm,
        strength=strength,arm=arm,truth=original['truth'],fits=[compact_fit(f) for f in fits],
        training_center_accepted=training['pooled_fit']['accepted'],
        fully_shared_estimate={n:float(np.exp(v)) for n,v in training['mean'].items()})


def selected_synthetic(cfg,out,workers):
    selection=read(out/'frozen_selection.json');cases=[]
    for model,entry in selection['models'].items():
        if 'Hselected' not in entry['singles']:continue
        for strength in sorted(set(entry['sharing_strengths'].values())):
            if strength<=0:continue
            cases.append((model,float(strength),'C0a'))
            if entry['combine'] and 'C1' in entry['singles']:cases.append((model,float(strength),'C1'))
    training_cases=sorted({(model,level) for model,_,_ in cases for level in range(3)})
    sample_jobs=[(cfg,str(out),model,level,subject) for model,level in training_cases
        for subject in range(cfg['synthetic']['subjects'],cfg['synthetic']['subjects']+SYNTHETIC_TRAINING_SUBJECTS)]
    run_tasks(cfg,out,'synthetic_selected_training_samples',selected_synthetic_training_sample_worker,sample_jobs,
        [f'{m}__l{l}__s{s:02d}' for _,_,m,l,s in sample_jobs],workers)
    jobs=[(cfg,str(out),model,level) for model,level in training_cases]
    run_tasks(cfg,out,'synthetic_selected_training',selected_synthetic_training_worker,jobs,[m+'__l'+str(l) for m,l in training_cases],workers)
    jobs=[(cfg,str(out),model,level,subject,strength,arm) for model,strength,arm in cases for level in range(3) for subject in range(cfg['synthetic']['subjects'])]
    keys=[f'{m}__l{l}__s{s:02d}__H{strength:g}_{arm}' for _,_,m,l,s,strength,arm in jobs]
    run_tasks(cfg,out,'synthetic_selected',selected_synthetic_worker,jobs,keys,workers)
    rows=[]
    for key in keys:
        d=read(out/'synthetic_selected'/(key+'.json'))
        if d['status']!='completed':continue
        for side,f in enumerate(d['fits']):
            for name in cfg['models'][d['model']]:
                true=d['truth'][name];value=f.get('parameters',{}).get(name,np.nan)
                rows.append(dict(subject=d['subject'],side=side,level=d['level'],model=d['model'],variant=d['variant'],
                    parameter=name,truth=true,value=value,relative_error=abs(value/true-1),accepted=f.get('accepted',False)))
    frame=pd.DataFrame(rows);frame.to_csv(out/'synthetic_selected_estimates.csv',index=False);scores=[]
    if len(frame):
        for key,g in frame.groupby(['model','variant','level','parameter']):
            means=g.groupby('subject')[['truth','value']].mean()
            slope=float(np.polyfit(np.log(means.truth),np.log(means.value),1)[0]) if np.std(np.log(means.truth))>1e-8 else None
            scores.append(dict(zip(['model','variant','level','parameter'],key),available_subjects=len(means),
                planned_subjects=cfg['synthetic']['subjects'],convergence=float(g.accepted.sum()/(2*cfg['synthetic']['subjects'])),
                median_relative_error=float(g.relative_error.median()),p90_relative_error=float(g.relative_error.quantile(.9)),log_recovery_slope=slope,
                between_log_variance=float(np.var(np.log(means.value),ddof=1)),truth_log_variance=float(np.var(np.log(means.truth),ddof=1))))
    write(out/'synthetic_selected_summary.json',dict(status='completed',cases=cases,estimates=scores,
        scored_subject_ids=list(range(cfg['synthetic']['subjects'])),
        disjoint_training_subject_ids=list(range(cfg['synthetic']['subjects'],cfg['synthetic']['subjects']+SYNTHETIC_TRAINING_SUBJECTS)),
        calibration_contract='same generator/noise/processing; fit pooled center and paired-log-crossmoment scale from extra training records at each heterogeneity level; no scored identity or individual truth used',
        training_centers=[dict(model=m,level=l,accepted=read(out/'synthetic_selected_training'/(m+'__l'+str(l)+'.json'))['pooled_fit']['accepted']) for m,l in training_cases],
        reason='only retained sharing strengths require additional recovery calibration; rejected measured arms do not advance'))
    return scores


def synthetic_information_summary(cfg,out):
    rows=[]
    for path in (out/'synthetic').glob('*.json'):
        d=read(path)
        if d.get('status')!='completed':continue
        names=cfg['models'][d['model']];bounds=np.asarray([cfg['parameters'][n]['bounds'] for n in names])
        for variant,fits in d['fits'].items():
            if not all(f.get('accepted') and 'information' in f for f in fits):continue
            values=np.asarray([[f['parameters'][n] for n in names] for f in fits])
            if (boundary_distance(values,bounds)<.05).any():continue
            matrices=[np.asarray(f['information']['data_only']['matrix']) for f in fits]
            if any(np.linalg.eigvalsh(mat)[0]<1e-12 for mat in matrices):continue
            covariance=[cfg['synthetic']['noise_fraction']**2*np.linalg.inv(mat) for mat in matrices]
            variance=np.diag(covariance[0]+covariance[1]);squared=(np.log(values[0])-np.log(values[1]))**2
            for j,name in enumerate(names):
                rows.append(dict(model=d['model'],level=d['level'],variant=variant,parameter=name,subject=d['subject'],
                    actual_squared_repeat=squared[j],linearized_repeat_variance=variance[j],ratio=squared[j]/variance[j]))
    frame=pd.DataFrame(rows);summary=[]
    if len(frame):
        for key,g in frame.groupby(['model','level','variant','parameter']):
            summary.append(dict(zip(['model','level','variant','parameter'],key),accepted_inner_subjects=len(g),
                mean_actual_squared_repeat=float(g.actual_squared_repeat.mean()),
                mean_linearized_repeat_variance=float(g.linearized_repeat_variance.mean()),
                mean_standardized_squared_repeat=float(g.ratio.mean()),median_standardized_squared_repeat=float(g.ratio.median())))
    return dict(interpretation='known synthetic observation-noise variance times inverse effective data curvature; local linear diagnostic at fitted interior estimates, not a measured physiological bound',
        formula='var(log theta_A-log theta_B) approximately sigma^2 * diag(Ieff_A^-1 + Ieff_B^-1)',
        inclusion='both accepted, all parameters 5% interior, positive full-rank effective information; excluded pairs are not recovery successes',rows=summary)


def followup_decisions(cfg,out,pp):
    selection=read(out/'frozen_selection.json')
    groups={model:{key:False for key in pp['profile_panel']} for model in active_models(out)}
    searched=Counter()
    for path in (out/'profiles').glob('*__regularized.json'):
        d=read(path)
        if 'rows' not in d:continue
        searched[d['model']]+=1
        n0=read(out/'development'/(d['pair_id']+'__'+d['model']+'__N0.json'))
        ref=read(out/'development'/(d['pair_id']+'__P0__N0.json'))
        hits=[]
        for side in range(2):
            old=n0['fits'][side]
            for row in d['rows']:
                fit=row['fits'][side]
                if fit.get('accepted',False) and (boundary_distance(row['values'],np.asarray(d['bounds']))>=.05).all():
                    gain=(old['objective']-fit['objective'])/ref['fits'][side]['objective']
                    if min(old['log_parameter_boundary_distance'])<.05 and gain>.01:hits.append(side)
        groups[d['model']][d['pair_id']]=bool(hits)
    numerical={model:dict(triggered=sum(cases.values())/len(cases)>=.25,
        groups_with_better_interior=sum(cases.values()),searched_groups=searched[model],registered_groups=len(cases),
        definition='N0 near boundary and an accepted profile inner point improves original objective by >1% of frozen P0 reference')
        for model,cases in groups.items()}
    continuous=[model for model,d in selection['models'].items() if 'C1' in d['singles']]
    value=dict(status='evaluated',N2=numerical,continuous_length_models=continuous,
        run_N2=any(v['triggered'] for v in numerical.values()),run_length_followup=bool(continuous),
        length_stopping_reason=None if continuous else 'no continuous arm retained under joint repeat/fidelity/transfer development criteria',
        parameter_expansion=read(out/'dimension_decision.json'))
    write(out/'followup_decisions.json',value);return value


def summarize(cfg,out,pp):
    rows=[];parameter_rows=[];all_records={}
    # Reconstruct the registered jobs so a failed/missing task stays in every
    # endpoint denominator, even when its exception record has no metadata.
    dev=[p for p in pp['pairs'] if p['partition']=='development']
    expected={'development':[(cfg,str(out),p,m,'C0a','N0',None) for p in dev for m in ['P0',*active_models(out)]]}
    if (out/'development_calibration.json').exists():
        cal=read(out/'development_calibration.json')
        expected['development_methods']=method_jobs(cfg,out,dev,cal)
        if (out/'frozen_selection.json').exists():
            expected['evaluation']=evaluation_jobs(cfg,out,[p for p in pp['pairs'] if p['partition']=='evaluation'],read(out/'frozen_selection.json'),cal)
    for stage in ('development','development_methods','evaluation'):
        directory=out/stage
        if not directory.exists():continue
        for _,_,p,model,arm,variant,_ in expected.get(stage,[]):
            path=directory/('__'.join((p['id'],model,variant))+'.json')
            d=read(path) if path.exists() else dict(status='missing')
            if 'pair_id' not in d:
                d=dict(d,pair_id=p['id'],model=model,arm=arm,variant=variant,
                    **{k:p[k] for k in ('dataset','subject','site','partition','repeat_kind')})
            all_records[(d['partition'],d['pair_id'],d['model'],d['variant'])]=d
    for key,d in all_records.items():
        baseline=all_records.get((d['partition'],d['pair_id'],d['model'],'N0'))
        row=result_metrics(d,cfg,baseline);rows.append(row)
        for name in cfg['models'][d['model']]:
            parameter_rows.append(dict(**{k:row[k] for k in ('dataset','subject','site','partition','repeat_kind','model','variant','accepted')},
                parameter=name,A=row.get(name+'_A',np.nan),B=row.get(name+'_B',np.nan),repeat=row.get(name+'_repeat',np.nan)))
    frame=pd.DataFrame(rows);parameters=pd.DataFrame(parameter_rows)
    frame.to_csv(out/'pair_metrics.csv',index=False);parameters.to_csv(out/'parameter_estimates.csv',index=False)
    def aggregate_group(key,g,columns):
        subject=g.groupby('subject').mean(numeric_only=True)
        # Continuous scores average the two records, then sites per subject,
        # then subjects. Pair pass counts retain every registered ROI pair.
        entry=dict(zip(columns,key));entry.update(planned_pairs=len(g),independent_subjects=g.subject.nunique(),
            completed_pairs=int(g.completed.sum()),accepted_pairs=int(g.accepted.sum()),
            effective_pairs=int(g.effective_pair.sum()),effective_pair_rate=float(g.effective_pair.mean()),
            subject_equal_effective_pair_rate=float(subject.effective_pair.mean()),
            near_01_pair_rate=float(g.near_01.mean()),near_05_pair_rate=float(g.near_05.mean()),
            near_10_pair_rate=float(g.near_10.mean()),exact_boundary_pair_rate=float(g.exact_boundary.mean()),
            repeat_median=float(subject.repeat_error.median()),repeat_p90=float(subject.repeat_error.quantile(.9)),
            convergence_pair_rate=float(g.accepted.mean()),available_repeat_pairs=int(g.repeat_error.notna().sum()))
        entry.update(convergence_record_rate=float(g.accepted_record_fraction.mean()),
            transfer_convergence_record_rate=float(g.transfer_record_fraction.mean()),available_boundary_pairs=int(g.near_05.notna().sum()))
        for label in ('EEG','HbO','HbR'):
            for metric in ('nrmse','transfer'):
                entry[label+'_'+metric]=float(subject[label+'_'+metric].mean())
        return entry
    by=['partition','dataset','repeat_kind','model','variant']
    grouped=[aggregate_group(key,g,by) for key,g in frame.groupby(by)]
    coordinate_by=by+['site']
    coordinate_grouped=[aggregate_group(key,g,coordinate_by) for key,g in frame.groupby(coordinate_by)]
    iccs=[]
    for key,g in parameters.groupby(['partition','dataset','site','repeat_kind','model','variant','parameter']):
        good=g[g.accepted].sort_values('subject');v=good[['A','B']].to_numpy()
        logs=np.log(v) if len(v) else v
        iccs.append(dict(zip(['partition','dataset','site','repeat_kind','model','variant','parameter'],key),
            planned_subjects=len(g),paired_accepted_subjects=len(good),icc=icc_absolute(logs),
            interval=bootstrap_icc(logs,cfg['seed'],cfg['endpoints']['bootstrap_repeats']),
            scale='log_parameter',interval_type='subject_percentile_bootstrap_95pct_fixed_training_objects',
            between_log_variance=float(np.var(logs.mean(1),ddof=1)) if len(logs)>1 else None))
    profile_rows=[];numerical=[];seen_profiles=set();profile_optimization=defaultdict(Counter)
    for path in sorted((out/'profiles').glob('*.json')) if (out/'profiles').exists() else []:
        d=read(path)
        if 'intersections' not in d:continue
        seen_profiles.add((d['pair_id'],d['model'],d['objective']))
        counts=profile_optimization[(d['model'],d['objective'])];counts['completed_groups']+=1
        for node in d['rows']:
            counts['record_node_fits']+=len(node['fits'])
            counts['accepted_record_node_fits']+=sum(f.get('accepted',False) for f in node['fits'])
        derived=profile_intersections(d,d['intersections'][0]['reference_objectives'],
            nrmse_margin=cfg['profile']['per_record_modality_nrmse_margin'],tolerances=cfg['profile']['objective_tolerances'],
            inners=cfg['profile']['inner_fractions'],narrow_fraction=cfg['profile']['narrow_log_range_fraction'])
        for row in derived:
            profile_rows.append(dict(**{k:d[k] for k in ('dataset','subject','site','model','objective','pair_id')},**row))
        if d['objective']!='current_without_parameter_shrinkage':continue
        n0=all_records[('development',d['pair_id'],d['model'],'N0')]
        ref=all_records[('development',d['pair_id'],'P0','N0')]
        for side in range(2):
            candidates=[(r['fits'][side],r['values']) for r in d['rows'] if 'objective' in r['fits'][side]]
            f,theta=min(candidates,key=lambda a:a[0]['objective']);old=n0['fits'][side]
            improved=(old['objective']-f['objective'])/ref['fits'][side]['objective']
            interior=bool((boundary_distance(theta,np.asarray(d['bounds']))>=.05).all())
            old_interior=bool((np.asarray(old['log_parameter_boundary_distance'])>=.05).all())
            better_inner=any(fit.get('accepted',False) and (boundary_distance(values,np.asarray(d['bounds']))>=.05).all()
                and (old['objective']-fit['objective'])/ref['fits'][side]['objective']>.01 for fit,values in candidates)
            numerical.append(dict(pair_id=d['pair_id'],dataset=d['dataset'],model=d['model'],side=side,
                objective_gain_reference_fraction=max(0.,improved),profile_best_accepted=f['accepted'],
                profile_best_values=theta,profile_best_interior=interior,N0_interior=old_interior,
                N0_objective=old['objective'],N1_objective=min(f['objective'],old['objective']),
                trigger_N2=bool(better_inner and not old_interior),
                paths=d['paths'].__len__(),accepted_nodes=sum(r['fits'][side]['accepted'] for r in d['rows']),total_nodes=len(d['rows'])))
    for p in pp['pairs']:
        if p['id'] not in pp['profile_panel']:continue
        for model in active_models(out):
            for objective in cfg['profile']['objectives']:
                if (p['id'],model,objective) in seen_profiles:continue
                for inner in cfg['profile']['inner_fractions']:
                    for tolerance in cfg['profile']['objective_tolerances']:
                        profile_rows.append(dict(dataset=p['dataset'],subject=p['subject'],site=p['site'],pair_id=p['id'],model=model,objective=objective,
                            inner_fraction=inner,tolerance=tolerance,exists=False,accepted_nodes=0,log_range_width=[np.nan]*len(cfg['models'][model]),
                            delta_share=np.nan,delta_inner=np.nan,delta_share_reference_fraction=np.nan,delta_inner_reference_fraction=np.nan,sampled_classification='profile_search_failed_or_missing',
                            unrestricted_common_nodes=0,interior_nodes_failing_nuisance_stationarity=None,witness=None))
    profile_frame=pd.DataFrame(profile_rows);profile_frame.to_json(out/'profile_intersections.json',orient='records',indent=2,force_ascii=False)
    pd.DataFrame(numerical).to_csv(out/'solver_reference.csv',index=False)
    synthetic=synthetic_summary(cfg,out)
    result=dict(schema='parameter_stability_summary_v1',execution='completed' if (out/'verification.json').exists() else 'analysis_in_progress',
        experiment_id=cfg['experiment_id'],scope=cfg['scope'],groups=grouped,coordinate_groups=coordinate_grouped,icc=iccs,
        synthetic=synthetic,profile_intersections=profile_rows,numerical_reference=numerical,
        profile_optimization=[dict(model=model,objective=objective,registered_groups=len(pp['profile_panel']),**counts)
            for (model,objective),counts in sorted(profile_optimization.items())],
        planned_subjects={ds:dict(development=len(pp['development_subjects'][ds]),evaluation=len(pp['candidate_subjects'][ds])) for ds in cfg['data']['datasets']},
        paired_scope={ds:dict(Counter(p['repeat_kind'] for p in pp['pairs'] if p['dataset']==ds)) for ds in cfg['data']['datasets']},
        frozen_selection=read(out/'frozen_selection.json') if (out/'frozen_selection.json').exists() else None,
        mathematical_and_numerical_verification=read(out/'verification.json') if (out/'verification.json').exists() else None,
        interpretation='best_found_conditional_estimates_not_measured_physiological_ground_truth',
        metric_contract=dict(nrmse='RMSE in original frozen observation coordinate / development SD; record mean then site mean then subject mean',
            repeat='maximum 2|A-B|/(|A|+|B|) across released parameters; site mean within subject then median/P90 across subjects',
            effective_pair='both accepted, every released parameter >=5% log-interior, worst relative difference<=.20, every record/modality fit and transfer <=N0+.01; fixed/shared null excluded',
            boundary_rate='among finite paired parameter estimates; missing estimates are separately retained as ineffective pairs in the full registered denominator',
            intervals='subject bootstrap per dataset/site; no pooled cross-dataset ICC; frozen calibration not refitted',
            profile='sampled engineering tolerance sets; no confidence interval/global optimality/nonexistence claim'))
    result['conditional_followups']=read(out/'followup_decisions.json') if (out/'followup_decisions.json').exists() else None
    if not (result['mathematical_and_numerical_verification'] or {}).get('verification_pass',False):
        result['execution']='analysis_in_progress'
    followups=result['conditional_followups'] or {}
    result['pending_conditional_steps']=[]
    if followups.get('run_N2') and not (out/'numerical_trust_region_summary.json').exists():result['pending_conditional_steps'].append('N2')
    if followups.get('run_length_followup') and not (out/'length_followup_summary.json').exists():result['pending_conditional_steps'].append('continuous_lengths')
    if result['pending_conditional_steps']:result['execution']='analysis_in_progress'
    result['selected_synthetic']=read(out/'synthetic_selected_summary.json') if (out/'synthetic_selected_summary.json').exists() else None
    result['synthetic_information_calibration']=synthetic_information_summary(cfg,out)
    result['synthetic_vector_recovery']=[]
    synthetic_sources=[out/'synthetic_estimates.csv',out/'synthetic_selected_estimates.csv']
    for path in synthetic_sources:
        if not path.exists() or path.stat().st_size<10:continue
        estimate=pd.read_csv(path)
        per_record=estimate.groupby(['model','variant','level','subject','side']).agg(relative_error=('relative_error','max'),accepted=('accepted','all')).reset_index()
        for key,g in per_record.groupby(['model','variant','level']):
            result['synthetic_vector_recovery'].append(dict(zip(['model','variant','level'],key),
                planned_records=2*cfg['synthetic']['subjects'],finite_records=int(g.relative_error.notna().sum()),
                median_worst_parameter_error=float(g.relative_error.median()),p90_worst_parameter_error=float(g.relative_error.quantile(.9)),
                all_parameter_convergence=float(g.accepted.sum()/(2*cfg['synthetic']['subjects']))))
    result['parameter_distributions']=[]
    for key,g in parameters.groupby(['partition','dataset','site','repeat_kind','model','variant','parameter']):
        available=g[['A','B']].notna().all(axis=1);v=g.loc[available,['A','B']].to_numpy()
        if not len(v):continue
        logs=np.log(v);average=logs.mean(axis=1)
        result['parameter_distributions'].append(dict(zip(['partition','dataset','site','repeat_kind','model','variant','parameter'],key),
            planned_subjects=len(g),available_subjects=int(available.sum()),accepted_subjects=int(g.accepted.sum()),
            median_A=float(np.median(v[:,0])),median_B=float(np.median(v[:,1])),
            median_geometric_AB=float(np.exp(np.median(average))),
            geometric_AB_p10=float(np.exp(np.quantile(average,.1))),geometric_AB_p90=float(np.exp(np.quantile(average,.9))),
            between_log_variance=float(np.var(average,ddof=1)) if len(v)>1 else None))
    result['fidelity_diagnostics']=[]
    for key,g in frame.groupby(by):
        subject=g.groupby('subject').mean(numeric_only=True)
        result['fidelity_diagnostics'].append(dict(zip(by,key),
            raw_pair_pass=float(g.raw_fit_pass.mean()) if 'raw_fit_pass' in g else 0.,
            transfer_pair_pass=float(g.transfer_pass.mean()) if 'transfer_pass' in g else 0.,
            subject_equal_raw_pass=float(subject.raw_fit_pass.mean()) if 'raw_fit_pass' in subject else 0.,
            subject_equal_transfer_pass=float(subject.transfer_pass.mean()) if 'transfer_pass' in subject else 0.))
    result['paired_method_differences']=[]
    for key,g in frame[frame.variant!='N0'].groupby(by):
        partition,dataset,kind,model,variant=key
        baseline=frame[(frame.partition==partition)&(frame.dataset==dataset)&(frame.repeat_kind==kind)
            &(frame.model==model)&(frame.variant=='N0')].set_index('pair_id')
        joined=g.set_index('pair_id').join(baseline,rsuffix='_baseline')
        for metric in ('repeat_error','effective_pair','EEG_nrmse','HbO_nrmse','HbR_nrmse','EEG_transfer','HbO_transfer','HbR_transfer'):
            if metric not in joined or metric+'_baseline' not in joined:continue
            difference=joined[metric].astype(float)-joined[metric+'_baseline'].astype(float)
            subject=difference.groupby(joined.subject).mean().dropna().to_numpy()
            if not len(subject):continue
            rng=np.random.default_rng(cfg['seed'])
            samples=rng.integers(0,len(subject),(cfg['endpoints']['bootstrap_repeats'],len(subject)))
            interval=np.quantile(subject[samples].mean(axis=1),[.025,.975])
            result['paired_method_differences'].append(dict(zip(by,key),metric=metric,
                planned_pairs=len(g),finite_pairs=int(difference.notna().sum()),available_subjects=len(subject),
                mean_subject_difference=float(subject.mean()),interval=interval,
                direction='candidate_minus_N0',interval_type='paired_subject_percentile_bootstrap_95pct_fixed_training_objects',
                multiplicity='descriptive_unadjusted_no_confirmatory_significance_claim'))
    result['sharing_information_budget']=[]
    if result['frozen_selection']:
        budget_rows=[]
        for d in all_records.values():
            if d['partition']!='evaluation' or d['variant']!='Hselected':continue
            key=coordinate_key(d);model=d['model']
            strength=result['frozen_selection']['models'][model]['sharing_strengths'][key]
            if strength==0:continue
            precision=shrinkage_calibration(cal[key],model,strength)['prior']['precision']
            for fit in d.get('fits',[]):
                if not fit.get('accepted') or 'information' not in fit:continue
                data=np.diag(fit['information']['data_only']['matrix'])
                regularized=np.diag(fit['information']['nuisance_regularized']['matrix'])
                for j,name in enumerate(cfg['models'][model]):
                    total=regularized[j]+precision[name]
                    budget_rows.append(dict(dataset=d['dataset'],site=d['site'],repeat_kind=d['repeat_kind'],
                        model=model,parameter=name,subject=d['subject'],strength=strength,
                        data_fraction=data[j]/total,original_regularization_fraction=(regularized[j]-data[j])/total,
                        added_parameter_prior_fraction=precision[name]/total))
        budgets=pd.DataFrame(budget_rows)
        if len(budgets):
            columns=['dataset','site','repeat_kind','model','parameter','strength']
            for key,g in budgets.groupby(columns):
                subject=g.groupby('subject').mean(numeric_only=True)
                result['sharing_information_budget'].append(dict(zip(columns,key),accepted_records=len(g),
                    available_subjects=len(subject),**{name:float(subject[name].mean()) for name in
                        ('data_fraction','original_regularization_fraction','added_parameter_prior_fraction')},
                    interpretation='local_log_parameter_Gauss_Newton_diagonal_after_nuisance_elimination_not_global_identification_probability'))
    # Descriptive rankings never feed back into the frozen method selection.
    # State the required numerical/repeat gates separately from physiological
    # identification; passing the former is not a physiological qualification.
    result['evaluation_assessment']=[]
    dataset_labels={'eeg_fnirs_single_trial':'Single-Trial','simultaneous_eeg_nirs':'Simultaneous','visual_cognitive_motivation':'Visual'}
    for dataset in cfg['data']['datasets']:
        kind='within_record_disjoint_blocks' if dataset=='simultaneous_eeg_nirs' else 'independent_native_record'
        for model in active_models(out):
            candidates=[g for g in grouped if g['partition']=='evaluation' and g['dataset']==dataset
                and g['repeat_kind']==kind and g['model']==model and g['variant']!='Hfull']
            if not candidates:continue
            winner=min(candidates,key=lambda g:(-g['effective_pair_rate'],g['repeat_median'],g['variant']))
            baseline=next(g for g in candidates if g['variant']=='N0')
            convergence=winner['convergence_record_rate']>=cfg['endpoints']['success']['convergence_min']
            repeat=winner['repeat_median']<=cfg['endpoints']['success']['repeat_symmetric_relative_median_max']
            result['evaluation_assessment'].append(dict(dataset=dataset,repeat_kind=kind,model=model,
                descriptive_best_variant=winner['variant'],ranking='effective_pair_rate_descending_then_repeat_median_ascending_not_reselection',
                baseline_repeat_median=baseline['repeat_median'],best_repeat_median=winner['repeat_median'],
                baseline_near_05_pair_rate=baseline['near_05_pair_rate'],best_near_05_pair_rate=winner['near_05_pair_rate'],
                effective_pairs=winner['effective_pairs'],planned_pairs=winner['planned_pairs'],
                convergence_record_rate=winner['convergence_record_rate'],numerical_gate_pass=convergence,
                repeat_gate_pass=repeat,necessary_numerical_and_repeat_gates_pass=convergence and repeat,
                physiological_qualification='not_established_by_descriptive_ranking'))
    assessments=result['evaluation_assessment']
    if assessments:
        passes=sum(a['necessary_numerical_and_repeat_gates_pass'] for a in assessments)
        result['headline_conclusion']=(f'{len(assessments)}个数据集×参数层次中，{passes}个观察最佳结果同时满足数值通过率≥95%和重复差异中位≤20%。'
            '实测个体生理准确性仍未建立；共享的稳定收益须与真实差异恢复分开判断。')
        dataset_findings=[]
        for dataset,label in dataset_labels.items():
            own=[a for a in assessments if a['dataset']==dataset]
            chosen=min(own,key=lambda a:(-a['effective_pairs']/a['planned_pairs'],a['best_repeat_median'],a['model']))
            caveat='同记录辅助证据' if dataset=='simultaneous_eeg_nirs' else '独立记录评价'
            dataset_findings.append((label+'（'+caveat+'）',
                f"观察排名最高为{chosen['model']}／{chosen['descriptive_best_variant']}：重复差异中位{chosen['best_repeat_median']:.1%}，"
                f"5%近界配对{chosen['best_near_05_pair_rate']:.1%}，有效配对{chosen['effective_pairs']}/{chosen['planned_pairs']}。不能据此认定个体生理真值已恢复。"))
        result['final_findings']=dataset_findings+[
            ('后续用途与优先级','保留条件重建和带失败标记的参数摘要；个体生理解释仍需独立约束。先解决共同内点与差异恢复问题，再扩大自由参数或推广强共享。')]
        result['unexecuted_steps_text']=('N2新求解器未触发；' if not followups.get('run_N2') else 'N2按已登记触发执行；')+(
            '连续长度扩展未触发；' if not followups.get('run_length_followup') else '连续长度扩展按已登记触发执行；')+(
            '未保留的单项不进入评价，也无组合臂。未训练tokenizer或执行其他协议的受保护评价。'
            if not any(v['combine'] for v in (result['frozen_selection'] or {}).get('models',{}).values())
            else '仅冻结保留项进入评价与组合。未训练tokenizer或执行其他协议的受保护评价。')
    write(out/'summary.json',result);return result


def verify_worker(payload):
    cfg,out,stage,key,pair=payload;out=Path(out);d=read(out/stage/(key+'.json'))
    if 'fits' not in d:return dict(status='failed',key=key,reason='missing_result')
    targets,sd=target_pair(str(out),pair['id'],coordinate_key(pair));checks=[]
    base=base_config(cfg['source_config'])
    with np.load(out/'fits'/(key+'.npz')) as a:
        for side,fit in enumerate(d['fits']):
            if 'parameters' not in fit:checks.append(dict(side=side,status='failed_domain',accepted=False));continue
            p=parameter_object(base,fit['parameters']);driver=a[f'driver_{side}'];initial=a[f'initial_state_{side}'];pred=a[f'prediction_{side}']
            _,op,_=record_arrays(targets[side],d['arm']);differences=[];legal=True
            for r,i,stored in zip(driver,initial,pred):
                try:
                    forward=nonlinear_driver_forward(r,i,p,.25,substeps=cfg['solver']['validation_substeps'],numerical_backend='numba',derivative=False)
                    predicted=(op@forward['canonical_prediction'].ravel()).reshape(-1,3)
                    differences.append(float(np.max(abs(predicted-stored)/sd)))
                except (ValueError,FloatingPointError,OverflowError):legal=False;differences.append(float('inf'))
            computed=np.sqrt(np.mean(((pred.reshape(-1,3)-targets[side])/sd)**2,axis=0))
            metric_error=float(np.max(abs(computed-np.asarray(fit['nrmse']))))
            checks.append(dict(side=side,accepted=fit.get('accepted',False),legal_substeps8=legal,
                max_substep_difference_development_sd=max(differences),metric_max_abs_error=metric_error,
                status='completed' if legal and metric_error<1e-10 else 'failed'))
    return dict(status='completed',key=key,checks=checks)


def verify(cfg,out,pp,workers):
    lookup={p['id']:p for p in pp['pairs']};jobs=[];keys=[]
    for stage in ('development','development_methods','evaluation'):
        for path in sorted((out/stage).glob('*.json')) if (out/stage).exists() else []:
            d=read(path)
            if 'pair_id' not in d:continue
            jobs.append((cfg,str(out),stage,path.stem,lookup[d['pair_id']]));keys.append(stage+'__'+path.stem)
    run_tasks(cfg,out,'verify',verify_worker,jobs,keys,workers)
    outcomes=[read(out/'verify'/(k+'.json')) for k in keys];checks=[c for d in outcomes for c in d.get('checks',[])]
    co=read(out/'coordinates.json');dev=pp['development_subjects']
    isolation=all(set(c['training_subjects'])<=set(dev[key.split('__')[0]]) for key,c in co.items())
    accepted=[c for c in checks if c.get('accepted')]
    errors=[c.get('metric_max_abs_error',np.inf) for c in checks if 'metric_max_abs_error' in c]
    precision=[c.get('max_substep_difference_development_sd',np.inf) for c in accepted]
    summary=dict(status='completed',result_files=len(outcomes),record_checks=len(checks),
        verification_job_failures=sum(d['status']=='failed' for d in outcomes),training_coordinate_isolation=isolation,
        accepted_records=len(accepted),accepted_legal_substeps8=sum(c.get('legal_substeps8',False) for c in accepted),
        metric_max_abs_error=max(errors,default=None),accepted_max_substep_difference_development_sd=max(precision,default=None),
        accepted_substep_difference_over_0_001=sum(v>.001 for v in precision),
        tested_substeps=[4,8],expected_main_stage_failures_retained={stage:read(out/(stage+'_manifest.json')).get('failed')
            for stage in ('synthetic','prepare','development','development_methods','profiles','evaluation') if (out/(stage+'_manifest.json')).exists()},
        protected_arrays_read=0)
    summary['verification_pass']=bool(isolation and not summary['verification_job_failures']
        and summary['accepted_legal_substeps8']==len(accepted)
        and max(errors,default=np.inf)<1e-10 and not summary['accepted_substep_difference_over_0_001'])
    write(out/'verification.json',summary);return summary


def run_tasks(cfg,out,stage,worker,payloads,keys,workers):
    directory=out/stage;directory.mkdir(parents=True,exist_ok=True)
    pending=[]
    for payload,key in zip(payloads,keys):
        path=directory/(key+'.json')
        if not path.exists():pending.append((payload,key))
        elif RETRY_FAILED and read(path).get('status')=='failed':
            # Current-run software recovery retains the failed record verbatim.
            # No completed result is rewritten and no artifact is removed.
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            write(directory/'failed_attempts'/(key+'__'+stamp+'.json'),read(path))
            pending.append((payload,key))
    begun=time.time();done=len(keys)-len(pending)
    write(out/'progress.json',dict(stage=stage,completed=done,expected=len(keys),status='running',utc=datetime.now(timezone.utc).isoformat()))
    if not pending:return
    def store_result(future,key):
        nonlocal done
        try:result=future.result()
        except Exception as exc:result=dict(status='failed',error=repr(exc),traceback=traceback.format_exc())
        write(directory/(key+'.json'),result);done+=1
        elapsed=time.time()-begun
        write(out/'progress.json',dict(stage=stage,completed=done,expected=len(keys),status='running',elapsed_s=elapsed,
            remaining_estimate_s=elapsed/max(1,done-(len(keys)-len(pending)))*(len(keys)-done),utc=datetime.now(timezone.utc).isoformat()))
        print(f'{stage} {done}/{len(keys)} {key} {result.get("status","completed")}',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        iterator=iter(pending);active={}
        for _ in range(min(len(pending),2*workers)):
            payload,key=next(iterator);active[pool.submit(worker,payload)]=key
        while active:
            finished,_=wait(active,return_when=FIRST_COMPLETED)
            for future in finished:
                key=active.pop(future);store_result(future,key)
                try:payload,key=next(iterator)
                except StopIteration:continue
                active[pool.submit(worker,payload)]=key
    write(out/(stage+'_manifest.json'),dict(stage=stage,status='completed',expected=len(keys),
        failed=sum(read(directory/(k+'.json')).get('status')=='failed' for k in keys),elapsed_s=time.time()-begun,workers=workers))


def run_stage(cfg,out,stage,workers,pilot=0):
    if stage=='plan':return plan(cfg,out)
    if stage=='synthetic':
        jobs=[(cfg,str(out),s,l,m) for m in active_models(out) for l in range(3) for s in range(cfg['synthetic']['subjects'])]
        if pilot:jobs=jobs[:pilot]
        return run_tasks(cfg,out,stage,synthetic_worker,jobs,[f'{m}__l{l}__s{s:02d}' for _,_,s,l,m in jobs],workers)
    pp=read(out/'pair_plan.json');pairs=pp['pairs']
    if stage=='prepare':
        jobs=[(cfg,str(out),p) for p in pairs]
        if pilot:jobs=jobs[:pilot]
        return run_tasks(cfg,out,'prepare',prepare_worker,jobs,[j[2]['id'] for j in jobs],workers)
    if stage=='coordinates':return fit_coordinates(cfg,out,pp)
    available=[p for p in pairs if (out/'prepared'/(p['id']+'.npz')).exists()]
    dev=[p for p in available if p['partition']=='development']
    if stage=='development':
        jobs=[(cfg,str(out),p,m,'C0a','N0',None) for p in dev for m in ['P0',*active_models(out)]]
        if pilot:jobs=jobs[:pilot]
        return run_tasks(cfg,out,stage,measured_worker,jobs,['__'.join((j[2]['id'],j[3],j[5])) for j in jobs],workers)
    if stage=='profiles':
        jobs=[(cfg,str(out),p,m,o) for p in dev if p['id'] in pp['profile_panel'] for m in active_models(out) for o in cfg['profile']['objectives']]
        if pilot:jobs=jobs[:pilot]
        return run_tasks(cfg,out,stage,profile_worker,jobs,['__'.join((j[2]['id'],j[3],'obs' if j[4].startswith('observation') else 'regularized')) for j in jobs],workers)
    if stage=='crossfit':
        jobs=[(cfg,str(out),p) for p in dev]
        return run_tasks(cfg,out,stage,crossfit_worker,jobs,[p['id'] for p in dev],workers)
    if stage=='calibration':return calibration(cfg,out,pp,workers)
    if stage=='development_methods':
        cal=read(out/'development_calibration.json');jobs=method_jobs(cfg,out,dev,cal)
        if pilot:jobs=jobs[:pilot]
        return run_tasks(cfg,out,stage,measured_worker,jobs,['__'.join((j[2]['id'],j[3],j[5])) for j in jobs],workers)
    if stage=='select':return select_methods(cfg,out,pp)
    if stage=='evaluation':
        selection=read(out/'frozen_selection.json');cal=read(out/'development_calibration.json')
        jobs=evaluation_jobs(cfg,out,[p for p in available if p['partition']=='evaluation'],selection,cal)
        return run_tasks(cfg,out,stage,measured_worker,jobs,['__'.join((j[2]['id'],j[3],j[5])) for j in jobs],workers)
    if stage=='followup_decisions':return followup_decisions(cfg,out,pp)
    if stage=='synthetic_selected':return selected_synthetic(cfg,out,workers)
    if stage=='summary':return summarize(cfg,out,pp)
    if stage=='verify':return verify(cfg,out,pp,workers)
    raise ValueError(f'unsupported stage {stage}')


def main():
    global RETRY_FAILED
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(DEFAULT));ap.add_argument('--run-dir',required=True)
    ap.add_argument('--stage',required=True);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--pilot',type=int,default=0)
    ap.add_argument('--retry-failed',action='store_true',help='retain failed attempts and retry unfinished software tasks')
    args=ap.parse_args();cfg=config(args.config);out=Path(args.run_dir).resolve()
    RETRY_FAILED=args.retry_failed
    if not out.is_relative_to((ROOT/cfg['output_namespace']).resolve()):raise ValueError('output outside owning namespace')
    if not 1<=args.workers<=cfg['resources']['max_workers']:raise ValueError('worker budget')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resolved=out/'resolved_config.yaml'
    if resolved.exists() and yaml.safe_load(resolved.read_text())!=cfg:raise ValueError('existing run config is immutable')
    if not resolved.exists():resolved.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    import shutil
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    snapshot=out/'source_snapshots'/(args.stage+'_'+stamp)
    files=['experiments/scripts/evaluate_shared_driver_parameter_stability.py','src/inference/shared_driver_stability.py',
        'src/inference/shared_driver_reconstruction.py','src/inference/shared_driver_rk4.py',
        'src/inference/observation_baselines.py','src/data/ssm_prepared.py']
    for filename in files:
        dest=snapshot/filename;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/filename,dest)
    write(snapshot/'invocation.json',dict(argv=sys.argv,cwd=str(ROOT),python=sys.executable,workers=args.workers,
        source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        numerical_threads=1,utc=stamp))
    run_stage(cfg,out,args.stage,args.workers,args.pilot)


if __name__=='__main__':main()
