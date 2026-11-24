"""Offline event modulation and conditional measured-Hb associations on real MA/BL trials."""
from collections import defaultdict
from dataclasses import replace
from itertools import product
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

import run_single_vq_relations as suite
from run_dual_component_tokenizer import read_json, write_json
from src.analysis.observation_task_probe import DATASET, TASK, DOWNSTREAM_ROOT, _event
from src.analysis.single_vq_relations import fit_ridge, interval, wave_attribute
from src.data.downstream_input_cache import DownstreamInputDataset
from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
from src.data.event_alignment import window_within_alignment_support
from src.data.dual_component_tokenizer import _base_joint_eeg_observations
from src.utils.io import save_npz_atomic

EEG=('F3','F4','P3','P4')
HB=('AF3Fp1','AF4Fp2')
EEG_PHASES={'pre':(-4,0),'entry':(0,2),'active':(2,10),'exit':(10,18),'recovery':(18,24)}
LAGS=(2,4,6)
SUPPORTED={}


def record_tasks():
    rows,_=suite.maps()
    for role in ('train','validation'):
        ex=suite.parent.capacity.PANEL['examples']['eeg',role,DATASET]
        people={rows[i]['subject'] for i in np.unique(ex[:,0])}
        for i,row in enumerate(rows):
            if (row['dataset']==DATASET and row['modality']=='eeg' and row['split']==role
                and row['subject'] in people and row['record'] in ('session_01','session_03','session_05')
                and any(c['positions'] for c in row['inputs']['score'])):
                SUPPORTED.setdefault((role,row['subject'],row['record']),[]).append(i)
    if read_json(DOWNSTREAM_ROOT/'verification.json')['status']!='passed':
        raise ValueError('unverified downstream clock or measurement')
    tasks=[]; exclusion=[]
    for role in ('train','validation'):
        reader=DownstreamInputDataset(DOWNSTREAM_ROOT,split=role,task=TASK)
        admitted={rows[i]['subject'] for i in np.unique(suite.parent.capacity.PANEL['examples']['eeg',role,DATASET][:,0])}
        if not admitted<=set(reader.plan['subject_partitions'][DATASET][role]):
            raise ValueError('subject partition disagreement before signals')
        samples=defaultdict(list)
        for raw in reader.samples:
            if raw['subject'] not in admitted:
                continue
            event=_event(raw,role)
            samples[event['subject'],event['record']].append((event,raw['event']))
        for (sub,rec),events in samples.items():
            key=role,sub,rec
            if key not in SUPPORTED:
                exclusion.extend(dict(split=role,subject=sub,record=rec,event=e['event_index'],reason='no EEG score crops') for e,_ in events)
                continue
            join=f'{DATASET}|{sub}|{rec}'
            owner=reader.records[join]
            relative=Path('data/cache/physiology_downstream_v2/records')/DATASET/sub/rec
            folder=suite.parent.capacity.ROOT/relative
            if owner.npz_path not in (folder,Path(suite.parent.PLAN['recorded_root'])/relative) or folder.resolve()!=folder:
                raise ValueError('downstream cache record identity changed')
            # Only replace an exactly matched historical workspace prefix, in memory.
            owner=replace(owner,npz_path=folder)
            markers=[e['eeg_time_ms']/1000 for e in reader.index.events_by_join_key[join]]
            tasks.append((key,events,owner,np.array(markers)))
        seen={item[0]['id'] for group in samples.values() for item in group}
        for row in reader.plan['samples']:
            if row['split']==role and row['task']==TASK and row['subject'] in admitted and row['id'] not in seen:
                exclusion.append(dict(split=role,subject=row['subject'],record=row['record'],event=row['event_index'],reason='downstream synchronized QC'))
        for row in reader.plan['exclusions']:
            if row['split']==role and row['task']==TASK and row['subject'] in admitted:
                exclusion.append(dict(split=role,subject=row['subject'],record=row['record'],event=row['event_index'],reason=row['reason']))
    pd.DataFrame(exclusion,columns=['split','subject','record','event','reason']).to_csv(suite.OUT/'upstream_event_exclusions.csv',index=False)
    return tasks


def eeg_track(row_ids):
    rows,scales=suite.maps()
    stop=max(rows[r]['native_start_sample']+(c['start_s']+30)*200 for r in row_ids for c in rows[r]['inputs']['score'])
    track=np.full((stop,len(EEG)),np.nan,np.float32)
    for i in row_ids:
        row=rows[i]
        names=row['positions']
        if row['native_rate_hz']!=200 or row['rate']!=200:
            raise ValueError('Single-Trial EEG clock changed')
        for c in row['inputs']['score']:
            selected=[(dest,names.index(name)) for dest,name in enumerate(EEG) if name in names and names.index(name) in c['positions']]
            if not selected:
                continue
            values,valid=_base_joint_eeg_observations(row,c['start_s'],scales,names)
            lo=row['native_start_sample']+c['start_s']*200
            for dest,pos in selected:
                target=track[lo:lo+6000,dest]
                if np.isfinite(target).any():
                    raise ValueError('overlapping registered physical EEG score crops')
                target[valid[:,pos]]=values[valid[:,pos],pos]
    return track


def context(track,marker,offset,channels):
    start=round((marker+offset)*200)
    if start<0 or start+6000>len(track):
        return None
    values=track[start:start+6000,:channels]
    if not np.isfinite(values).all():
        return None
    return values.T.reshape(channels,15,400)


def measured_hb(record):
    names=record.manifest['measurement']['fnirs_channel_names']
    columns=[names.index(p+'_'+color) for p in HB for color in ('HbO','HbR')]
    unit=record.manifest['measurement']['fnirs_preprocessing_state']['canonical_unit']
    if unit!='relative_Hb_repo_approximation' or record.sample_rate_hz!=10:
        raise ValueError('wrong independent Hb coordinates')
    keys=('fnirs','fnirs_supported_channels')
    for key in keys:
        f=record.npz_path/record.manifest['arrays'][key]
        if f.resolve().parent!=record.npz_path:
            raise ValueError('invalid Hb record path')
    data=CleanPhysiologyCacheIndex.load_record_arrays(record,keys)
    if not data['fnirs_supported_channels'][columns].all():
        return None
    return np.asarray(data['fnirs'][:,columns]).reshape(-1,2,2).mean(1)


def conventional(x):
    p=np.abs(np.fft.rfft((x-x.mean(-1,keepdims=True))*np.hanning(400),axis=-1))**2
    total=np.maximum(p[...,2:91].sum(-1),1e-30)
    return np.stack((p[...,8:16].sum(-1)/total,p[...,16:26].sum(-1)/total,
                     np.sqrt((x*x).mean(-1))),axis=-1)


def token_history(x,ids,maps):
    # Last three assigned patches from each branch and each electrode, in time order.
    # Conventional features cover the same three cells plus full-context summaries.
    fam=[np.eye(4)[maps[b][ids[b]]] for b in 'SO']
    tv=np.concatenate([v[:,-3:].ravel() for v in fam]+[v.mean(axis=1).ravel() for v in fam])
    v=conventional(x)
    return tv,np.r_[v[:,-3:].ravel(),v.mean(1).ravel()]


@torch.inference_mode()
def record_worker(task):
    key,events,record,markers=task
    role,subject,rec=key
    folder=suite.OUT/'events_by_record';folder.mkdir(exist_ok=True)
    dest=folder/f'{role}_{subject}_{rec}.npz'
    if dest.exists():
        return
    model=suite.parent.network('eeg',8192,17)
    maps={b:np.array(read_json(suite.OUT/f'eeg_17/{b}_family.json')['mapping']) for b in 'SO'}
    track=eeg_track(SUPPORTED[key])
    hb=measured_hb(record)
    data=defaultdict(list); logs=[]
    for event,native in events:
        e=event['marker_ms']['eeg']/1000
        h=round(event['marker_ms']['hb']/100)
        info=dict(split=role,subject=subject,record=rec,event=event['event_index'],condition=event['condition'],
                  eeg_marker=e,hb_marker=event['marker_ms']['hb']/1000)
        entry=dict(info)
        if hb is None or h<50 or h+240>len(hb) or not np.isfinite(hb[h-50:h+240]).all():
            entry.update(B='Hb panel unavailable',D='Hb panel unavailable');logs.append(entry);continue
        def hbmean(lo,hi):
            return hb[h+10*lo:h+10*hi].mean(0)
        def encode(x):
            # Full neural forward assigns actual integers with learned quantizers.
            result=model(torch.from_numpy(x.copy()))['ids']
            return {b:result[b].numpy() for b in 'SO'}
        x=context(track,e,-6,4)
        if x is None or not window_within_alignment_support(native,-6,30):
            entry['B']='incomplete admitted event context'
        else:
            entry['B']='used'
            ids=encode(x)
            high=maps['S'][ids['S']]==3
            alpha=wave_attribute(x,'eeg')
            vals=[]
            for area, sl in [('frontal',slice(0,2)),('posterior',slice(2,4))]:
                for phase,(lo,hi) in EEG_PHASES.items():
                    i,j=(lo+6)//2,(hi+6)//2
                    cur=high[sl,i:j];prev=high[sl,i-1:j-1]
                    vals.append(dict(info,area=area,phase=phase,occupancy=cur.mean(),enter=(cur&~prev).mean(),
                        leave=(~cur&prev).mean(),alpha=alpha[sl,i:j].mean(),HbO=hbmean(lo,hi)[0],HbR=hbmean(lo,hi)[1]))
            data['B'].extend(vals)
        x=context(track,e,-26,2)
        shift=context(track,e,-69,2)
        if x is None or not window_within_alignment_support(native,-26,42):
            entry['D']='incomplete aligned EEG history or clock'
        elif shift is None or not window_within_alignment_support(native,-69,30):
            entry['D']='no disjoint shifted context'
        elif np.min(np.abs(markers-(e-43)))<=4:
            entry['D']='shift aligns with another cue'
        else:
            entry['D']='candidate'
            data['D'].append(info)
            for label,v in [('aligned',x),('shift',shift)]:
                token,raw=token_history(v,encode(v),maps)
                data[label].append(token)
                if label=='aligned':
                    data['continuous'].append(raw)
            hist=np.concatenate([hbmean(t,t+2) for t in (-2,0,2)])
            base=np.r_[hist,float(event['condition']=='MA'),[float(rec==r) for r in ('session_01','session_03','session_05')],event['event_index']/20]
            data['base'].append(base)
            data['y'].append(np.stack([hbmean(4+lag,6+lag)-hist[-2:] for lag in LAGS]))
        logs.append(entry)
    serial={k:np.array([json.dumps(x) for x in data[k]]) for k in ('B','D')}
    serial['log']=np.array([json.dumps(row) for row in logs])
    save_npz_atomic(dest,**serial,**{k:np.asarray(v) for k,v in data.items() if k not in ('B','D')})


def donor_indices(meta):
    """Same person, recording, condition and marker phase; maximum disjoint matching."""
    from scipy.optimize import linear_sum_assignment
    donor=np.full(len(meta),-1)
    for _,g in meta.groupby(['split','subject','record','condition']):
        idx=g.index.to_numpy();t=g.eeg_marker.to_numpy()
        allowed=np.abs(t[:,None]-t)>=30.01
        rng=np.random.default_rng(1703)
        a,b=linear_sum_assignment(~allowed*1e4+rng.uniform(size=allowed.shape))
        ok=allowed[a,b];donor[idx[a[ok]]]=idx[b[ok]]
    return donor


def aggregate_b(frame):
    folder=suite.OUT
    frame.to_csv(folder/'event_trials.csv.gz',index=False,compression='gzip')
    cols=['occupancy','enter','leave','alpha','HbO','HbR']
    profile=frame.groupby(['split','subject','record','condition','area','phase'])[cols].mean().groupby(
        ['split','subject','condition','area','phase']).mean().reset_index()
    profile.to_csv(folder/'event_profile.csv',index=False)
    results=[]
    for area in ('frontal','posterior'):
        for phase in ('entry','active','exit','recovery'):
            sub=frame[(frame.split=='validation')&(frame.area==area)]
            # Common trials and sessions for both conditions, with each session equal per person.
            view=sub.pivot(index=['subject','record','condition','event'],columns='phase',values=cols)
            delta=pd.DataFrame({c:view[c][phase]-view[c]['pre'] for c in cols})
            session=delta.groupby(['subject','record','condition']).mean().unstack('condition')
            if not all((c,k) in session for c in cols for k in ('MA','BL')):
                continue
            diff=pd.DataFrame({c:session[c]['MA']-session[c]['BL'] for c in cols}).dropna().groupby('subject').mean()
            for c in cols:
                v=diff[c].values
                if len(v)==0:
                    continue
                avg,lo,hi=interval(v,np.zeros(len(v)))
                flips=np.array(list(product((-1,1),repeat=len(v))))
                p=np.mean(np.abs((flips*v).mean(1))>=abs(avg)-1e-14)
                primary=area=='posterior' and phase=='active' and c=='occupancy'
                results.append(dict(area=area,phase=phase,measure=c,people=len(v),ma_minus_bl_change=avg,ci_low=lo,ci_high=hi,
                                    primary=primary,primary_exact_p=p if primary else np.nan))
    pd.DataFrame(results).to_csv(folder/'event_contrasts.csv',index=False)
    correlations=[]
    for (sub,area),g in frame[frame.split=='validation'].groupby(['subject','area']):
        table=g.pivot(index=['record','condition','event'],columns='phase',values=['occupancy','alpha','HbO','HbR'])
        v=pd.DataFrame({k:table[k].active-table[k].pre for k in ('occupancy','alpha','HbO','HbR')})
        v=v-v.groupby(['record','condition']).transform('mean')
        for item in ('alpha','HbO','HbR'):
            correlations.append(dict(subject=sub,area=area,measure=item,trials=len(v),r=v.occupancy.corr(v[item])))
    pd.DataFrame(correlations).to_csv(folder/'event_associations.csv',index=False)


def coupling(meta, arrays):
    folder=suite.OUT
    donors=donor_indices(meta)
    meta['donor_event']=[meta.iloc[d].event if d>=0 else -1 for d in donors]
    meta['donor_eeg_marker_s']=[meta.iloc[d].eeg_marker if d>=0 else np.nan for d in donors]
    meta.to_csv(folder/'coupling_pairs.csv',index=False)
    keep=donors>=0
    if not keep.any():
        return dict(status='insufficient support',reason='no admissible cross-trial donor')
    values={k:v[keep] for k,v in arrays.items()}
    values['mismatch']=arrays['aligned'][donors[keep]]
    table=meta.loc[keep].reset_index(drop=True)
    tr=table.split.eq('train').values
    people=table.subject.to_numpy(); train_ids=np.unique(people[tr])
    if len(train_ids)<3 or not (~tr).any():
        return dict(status='insufficient support',training_people=len(train_ids),development_people=len(np.unique(people[~tr])))
    cv=[]
    for i,lag in enumerate(LAGS):
        errs=[]
        for fold in range(3):
            val=tr&np.isin(people,train_ids[fold::3]);fit=tr&~val
            y=values['y'][:,i]
            pred=fit_ridge(values['base'][fit],y[fit])(values['base'][val])
            mse=((pred-y[val])**2/np.maximum(y[fit].var(0),1e-16)).mean(1)
            errs.extend(pd.Series(mse,index=people[val]).groupby(level=0).mean())
        cv.append(np.mean(errs))
    selected=int(np.argmin(cv));y=values['y'][:,selected];var=y[tr].var(0)
    if (var<=0).any():
        raise ValueError('zero training target variance')
    predictions={}
    aligned_model=None
    for method in ('base','aligned','mismatch','shift','continuous'):
        x=values['base'] if method=='base' else np.column_stack((values['base'],values[method]))
        model=fit_ridge(x[tr],y[tr])
        predictions[method]=model(x[~tr])
        if method=='aligned':
            aligned_model=model
    for name in ('mismatch','shift'):
        predictions['frozen_'+name]=aligned_model(np.column_stack((values['base'],values[name]))[~tr])
    scores=[]
    ref=table[~tr].reset_index(drop=True)
    obs=ref[['subject','record','condition','event']].copy()
    for c,color in enumerate(('HbO','HbR')):
        obs[color+'_measured_change']=y[~tr,c]
        for name,p in predictions.items():
            obs[color+'_'+name]=p[:,c]
            g=ref.copy()
            g['nmse']=(p[:,c]-y[~tr,c])**2/var[c]
            # Within-session condition balancing needs a pair of conditions.
            per_session=g.groupby(['subject','record','condition']).nmse.mean().unstack('condition').reindex(columns=['MA','BL'])
            v=per_session.dropna().mean(axis=1).groupby('subject').mean()
            scores.extend(dict(color=color,method=name,subject=sub,nmse=val) for sub,val in v.items())
    obs.to_csv(folder/'coupling_predictions.csv',index=False)
    scores=pd.DataFrame(scores,columns=['color','method','subject','nmse'])
    scores.to_csv(folder/'coupling_subject.csv',index=False)
    contrasts=[]
    if len(scores):
        pooled=scores.groupby(['method','subject']).nmse.mean().reset_index().assign(color='two_color')
        for c,g in pd.concat((scores,pooled)).groupby('color'):
            pivot=g.pivot(index='subject',columns='method',values='nmse')
            for refname in ('base','mismatch','shift','continuous','frozen_mismatch','frozen_shift'):
                delta=pivot[refname]-pivot.aligned
                avg,lo,hi=interval(delta,np.zeros(len(delta)))
                contrasts.append(dict(color=c,reference=refname,reference_nmse=pivot[refname].mean(),
                    aligned_nmse=pivot.aligned.mean(),difference=avg,ci_low=lo,ci_high=hi,people=len(delta),
                    improved=int((delta>0).sum())))
    pd.DataFrame(contrasts).to_csv(folder/'coupling_contrasts.csv',index=False)
    return dict(status='scored' if len(scores) else 'insufficient paired condition sessions',
                lag_grid=LAGS,baseline_training_cv=cv,selected_lag_s=LAGS[selected],training_trials=int(tr.sum()),
                development_trials=int((~tr).sum()),training_people=len(train_ids),development_people=len(np.unique(people[~tr])),
                scored_people=scores.subject.nunique(),normalization=var)


def evaluate_events(workers):
    tasks=record_tasks()
    if not tasks:
        raise ValueError('no admitted record tasks')
    suite.run_jobs(record_worker,tasks,workers)
    batches=defaultdict(list);rows=defaultdict(list)
    for key,*_ in tasks:
        with np.load(suite.OUT/'events_by_record'/('_'.join(key)+'.npz')) as z:
            for name in ('log','B','D'):
                rows[name].extend(json.loads(s) for s in z[name])
            for name in set(z.files)-{'log','B','D'}:
                batches[name].append(z[name])
    logs=pd.DataFrame(rows['log']);logs.to_csv(suite.OUT/'event_support.csv',index=False)
    if rows['B']:
        aggregate_b(pd.DataFrame(rows['B']))
    result=coupling(pd.DataFrame(rows['D']),{k:np.concatenate(v) for k,v in batches.items()}) if rows['D'] else dict(status='insufficient support')
    write_json(suite.OUT/'events_summary.json',dict(records=len(tasks),planned_after_upstream_QC=len(logs),B_used=len(rows['B'])//10,
        B_by_split=logs[logs.B=='used'].groupby('split').size().to_dict(),D_before_pairing=len(rows['D']),D=result))


def verify():
    root=suite.OUT
    ev=read_json(root/'events_summary.json')
    models=[]
    for m,spec in suite.PLAN['models'].items():
        for seed in spec['seeds']:
            d=root/f'{m}_{seed}'
            info=read_json(d/'completion.json')
            if info['train_people']!=60 or info['dev_people']!=16:
                raise ValueError('model lost people')
            cov=pd.read_csv(d/'coverage.csv')
            if not (cov[cov.dataset=='pooled'].valid==info['valid_dev_patches']).all():
                raise ValueError('atlas denominator mismatch')
            for b in 'SO':
                if not (d/f'{b}_members.png').is_file():
                    raise ValueError('missing word member atlas')
            seq=pd.read_csv(d/'sequence.csv');pair=pd.read_csv(d/'conditional.csv')
            if not len(seq) or len(pair)!=15:
                raise ValueError('incomplete C')
            models.append(info)
    if ev['B_used']:
        p=pd.read_csv(root/'event_contrasts.csv')
        if p.primary.sum()!=1:
            raise ValueError('incomplete event contrast')
    if ev['D']['status']=='scored':
        pairs=pd.read_csv(root/'coupling_pairs.csv');pairs=pairs[pairs.donor_event>=0]
        if (pairs.donor_event==pairs.event).any() or (abs(pairs.donor_eeg_marker_s-pairs.eeg_marker)<30).any():
            raise ValueError('overlapping cross-trial control')
    from single_vq_relations_evidence import collect_evidence
    denominators=collect_evidence(root,[(m,s) for m,spec in suite.PLAN['models'].items() for s in spec['seeds']])
    write_json(root/'summary.json',dict(status='completed',experiment_id=suite.PLAN['experiment_id'],models=models,events=ev,event_denominators=denominators,
               scope='ordinary development only; offline event and measured-Hb associations'))
