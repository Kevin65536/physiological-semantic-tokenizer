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
            arm=arm,covariance=covariance,prior=prior,fixed=fixed,n0=(variant=='N0')))
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
    names=cfg['models'][model];result=profile_pair(targets,sd,cfg,base_config(cfg['source_config']),names,objective=objective)
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
    cfg,out,key,pairs=payload;out=Path(out);base=base_config(cfg['source_config'])
    residuals=[]
    for pair in pairs:
        path=out/'crossfit'/(pair['id']+'.npz')
        if path.exists():
            with np.load(path) as a:residuals.extend(a['residual'])
    wc=estimate_working_covariance(residuals,shrinkage=cfg['weighting']['shrinkage'],floor=cfg['weighting']['eigenvalue_floor'],ar_clip=cfg['weighting']['ar1_clip'])
    models={}
    for model in active_models(out):
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
            pooled=fit_record(all_y,sd,cfg,base,model,starts=[start],profile=True)
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
    keys=sorted({coordinate_key(p) for p in plan['pairs']});jobs=[]
    for key in keys:
        pairs=[p for p in plan['pairs'] if p['partition']=='development' and coordinate_key(p)==key and (out/'prepared'/(p['id']+'.npz')).exists()]
        jobs.append((cfg,str(out),key,pairs))
    run_tasks(cfg,out,'calibration',calibration_worker,jobs,keys,workers)
    result={k:read(out/'calibration'/(k+'.json')) for k in keys}
    if any(v.get('status')=='failed' for v in result.values()):raise ValueError('development calibration failed; evaluation stays closed')
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
    row.update(completed=False,accepted=False,near_01=True,near_05=True,near_10=True,exact_boundary=True,
               repeat_error=np.nan,effective_pair=False,individual_parameter_model=bool(names) and result.get('variant')!='Hfull')
    if len(fits)!=2 or not all('parameters' in f for f in fits):return row
    row['completed']=True;row['accepted']=all(f.get('accepted',False) for f in fits)
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
                scores.append(dict(strength=strength,subject_equal_transfer_nrmse=float(np.mean(records))))
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
    write(out/'frozen_selection.json',value);return value


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


def run_tasks(cfg,out,stage,worker,payloads,keys,workers):
    directory=out/stage;directory.mkdir(parents=True,exist_ok=True)
    pending=[(payload,key) for payload,key in zip(payloads,keys) if not (directory/(key+'.json')).exists()]
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
    raise ValueError(f'unsupported stage {stage}')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(DEFAULT));ap.add_argument('--run-dir',required=True)
    ap.add_argument('--stage',required=True);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--pilot',type=int,default=0)
    args=ap.parse_args();cfg=config(args.config);out=Path(args.run_dir).resolve()
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
