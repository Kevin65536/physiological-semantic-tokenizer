"""Exhaustive saved-AS training support and independent-person raw-ID replication."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit, attributes, feature_means
from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_batch
from run_actual_id_coarsening import (FIELDS, native_blocks, sample_source_identity,
    additional_training_examples, sample_support_readout)
from audit_joint_tokenizer_patterns import hierarchy

STATE={}
K=5*8192


def write(path,value):
    def scalar(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(value,default=scalar,indent=2,ensure_ascii=False)+'\n')


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def budget(cfg):
    unit=os.environ.get('TOKENIZER_FULL_SUPPORT_UNIT')
    used=time.process_time()
    if unit:
        import subprocess
        value=subprocess.check_output(['systemctl','--user','show',unit,'-p','CPUUsageNSec','--value'],text=True).strip()
        if value.isdigit():used=int(value)/1e9
    if used>cfg['resources']['CPU_budget_seconds']-40:raise RuntimeError('bounded full-support CPU budget reached')
    return used


def census(cfg,rows,pop,lookup):
    units={tuple(v) for v in pop['subjects']}
    counts=defaultdict(lambda:dict(crops=0,positions=0,people=set(),records=set()))
    all_examples=[]
    for i,t in sorted(lookup):
        r=rows[i]
        if r['split']!='train' or r['modality']!='eeg' or (r['dataset'],r['subject']) not in units:
            raise ValueError('saved native crop escaped original train population')
        support=next(v['positions'] for v in r['inputs']['score'] if v['start_s']==t)
        current=counts[r['dataset']]
        current['crops']+=1;current['positions']+=len(support)
        current['people'].add(r['subject']);current['records'].add((r['subject'],r['record']))
        all_examples.extend((int(i),int(t),int(p)) for p in support)
    if len(lookup)!=5294 or len(all_examples)!=185823 or len(set(all_examples))!=len(all_examples):
        raise ValueError('metadata census changed; retained scope must be reviewed')
    return np.asarray(all_examples,np.int32),{d:{k:len(v) if isinstance(v,set) else v for k,v in c.items()} for d,c in counts.items()}


def prepare_dataset(dataset):
    cfg,rows,scales,out,lookup,names,pop,extra=(STATE[k] for k in ['cfg','rows','scales','out','lookup','names','population','extra'])
    target=out/'additional'/f'{dataset}.npz'
    if target.exists():return dict(dataset=dataset,status='resumed',path=str(target))
    take=extra[np.array([rows[i]['dataset']==dataset for i,t,p in extra])]
    data={}
    for key,shape,dtype in [('S',(15,),np.int32),('O',(15,),np.int32),('valid',(15,),bool),('background',(15,),np.int32),('S_attributes',(15,10),float),('O_attributes',(15,10),float),('fixed_band',(),np.int64),('subject',(),np.int64),('fold',(),np.int64)]:
        data[key]=np.zeros((len(take),*shape),dtype=dtype)
    units={tuple(v):j for j,v in enumerate(pop['subjects'])}
    folds={(v['dataset'],v['subject']):v['fold'] for v in pop['folds']}
    sites={tuple(v):j for j,v in enumerate(pop['background_sites'])}
    native={v.upper():j for j,v in enumerate(names[dataset])}
    signal_cache=OrderedDict();id_cache=OrderedDict()
    for start in range(0,len(take),cfg['resources']['batch_size']):
        used=budget(cfg)
        batch_ex=take[start:start+cfg['resources']['batch_size']]
        where=slice(start,start+len(batch_ex))
        for j,(i,t,p) in enumerate(batch_ex,start):
            row=rows[i];path,ordinal=lookup[int(i),int(t)]
            if path not in id_cache:
                with np.load(path,allow_pickle=False) as z:
                    if not np.array_equal(z['crops'][ordinal],[i,t]):raise ValueError('native integer ordinal mismatch')
                    id_cache[path]={b:z[b].copy() for b in 'SO'}
                if len(id_cache)>4:id_cache.popitem(last=False)
            position=native[row['positions'][p].upper()]
            for b in 'SO':data[b][j]=id_cache[path][b][ordinal,position]
            data['fixed_band'][j]=position%5
            unit=dataset,row['subject'];data['subject'][j]=units[unit];data['fold'][j]=folds[unit]
        batch=base_tokenizer_batch(rows,batch_ex,scales,array_cache=signal_cache,cache_limit=32)
        valid=batch['mask'].all(-1);data['valid'][where]=valid
        site=np.array([sites[rows[i]['dataset'],rows[i]['positions'][p]] for i,t,p in batch_ex])
        data['background'][where]=4*site[:,None]+np.searchsorted([.5,1.,2.],np.sqrt(np.mean(batch['x']**2,axis=-1)),side='right')
        for b in 'SO':
            raw,slices=attributes(batch[b.lower()][valid],'eeg')
            data[b+'_attributes'][where][valid]=np.concatenate([raw[:,slice(*slices[k])] for k in FIELDS],axis=1)
        if start%2048==0:write(out/'additional'/f'{dataset}_progress.json',dict(completed=start+len(batch_ex),planned=len(take),CPU_seconds=used))
    np.savez_compressed(target,examples=take,**data)
    return dict(dataset=dataset,position_windows=len(take),complete_patches=int(data['valid'].sum()),CPU_seconds=time.process_time())


def raw_profile(bank,fit,held,branch,backgrounds,with_cell_people=False):
    words=bank[branch]+8192*bank['fixed_band'][:,None]
    audit=SparsePatternAudit(K,backgrounds,[],10,FIELDS)
    selected=bank['valid']&fit[:,None]
    audit.fit(words[selected],bank['background'][selected],bank[branch+'_attributes'][selected])
    people=np.zeros(K,int)
    cell_people=defaultdict(int)
    for person in np.unique(bank['subject'][fit]):
        mask=bank['valid']&(bank['subject'][:,None]==person)
        ids=words[mask]
        people[np.unique(ids)]+=1
        if with_cell_people:
            for key in np.unique(bank['background'][mask]*K+ids):cell_people[int(key)]+=1
    v=bank['valid'][held];ids=words[held][v];bg=bank['background'][held][v]
    background,_,joint=audit.read(ids,bg)
    target=bank[branch+'_attributes'][held][v]
    error=np.zeros((*v.shape,2,3))
    error[v]=np.stack([feature_means((p-target)**2,FIELDS) for p in [background,joint]],axis=1)
    perwindow=error.sum(1)/v.sum(1).clip(1)[:,None,None]
    perwindow[~v.any(1)]=np.nan
    keys=bg*K+ids
    local_support=(audit.local.count[ids]>=8)&(people[ids]>=2)
    strong_cell=np.fromiter((audit.joint.count.get(int(k),0)>=8 and cell_people.get(int(k),0)>=2 for k in keys),bool,len(keys))
    seen=np.fromiter((int(k) in audit.joint.count for k in keys),bool,len(keys))
    variance=np.maximum((bank[branch+'_attributes'][selected]**2).mean(0)-bank[branch+'_attributes'][selected].mean(0)**2,1e-20)
    return dict(background=background,joint=joint,error=perwindow,local_support=local_support,cell_seen=seen,strong_cell=strong_cell,variance=variance,word_count=audit.local.count,word_people=people)


def weighted_correlation(a,b,fields):
    result=[]
    for first,last in fields.values():
        x=a[:,first:last].reshape(-1);y=b[:,first:last].reshape(-1)
        if len(x)<2 or x.std()==0 or y.std()==0:result.append(float('nan'))
        else:result.append(float(np.corrcoef(x,y)[0,1]))
    return result


def score_worker(task):
    fold,branch=task
    bank,pop,cfg,rows,out=(STATE[k] for k in ['bank','population','cfg','rows','out'])
    budget(cfg)
    directory=out/'folds';directory.mkdir(exist_ok=True)
    dest=directory/f'{branch}_fold{fold}.npz'
    meta=directory/f'{branch}_fold{fold}.json'
    if dest.exists() and meta.exists():return json.loads(meta.read_text())
    fit=bank['fold']!=fold
    held=(np.arange(len(bank['examples']))<15360)&(bank['fold']==fold)
    indices=np.flatnonzero(held)
    original=sample_support_readout(bank,15360,len(pop['background_sites']),True,fold,branch)
    score_rows,errors,flags,joint_count,bg_count,record=original
    np.testing.assert_array_equal(indices,score_rows)
    group=np.full(len(pop['subjects']),-1,np.int8)
    fit_people=sorted(np.unique(bank['subject'][fit]).tolist())
    for dataset in sorted({pop['subjects'][p][0] for p in fit_people}):
        people=[p for p in fit_people if pop['subjects'][p][0]==dataset]
        people.sort(key=lambda p:hashlib.sha256(json.dumps([cfg['replication']['seed'],*pop['subjects'][p]]).encode()).hexdigest())
        for number,person in enumerate(people):group[person]=number%2
    fits=[fit&(group[bank['subject']]==g) for g in range(2)]
    if set(bank['subject'][fits[0]])&set(bank['subject'][fits[1]]):raise ValueError('disjoint subject replication failed')
    halves=[raw_profile(bank,f,held,branch,4*(len(pop['background_sites'])+1),True) for f in fits]
    both_seen=halves[0]['cell_seen']&halves[1]['cell_seen']
    strong=halves[0]['strong_cell']&halves[1]['strong_cell']
    # A/B predictions are evaluated at exactly the same held physical patches.
    reliability={}
    full_fit_target=bank[branch+'_attributes'][bank['valid']&fit[:,None]]
    variance=np.maximum(full_fit_target.var(0),1e-20)
    for label,mask in [('all_held',np.ones(len(strong),bool)),('both_joint_seen',both_seen),('both_8patch_2person_joint',strong)]:
        a=halves[0]['joint'][mask];b=halves[1]['joint'][mask]
        ar=(halves[0]['joint']-halves[0]['background'])[mask]
        br=(halves[1]['joint']-halves[1]['background'])[mask]
        reliability[label]=dict(patches=int(mask.sum()),total_patches=len(mask),raw_mean_correlation=weighted_correlation(a,b,FIELDS),background_residual_correlation=weighted_correlation(ar,br,FIELDS),prediction_drift_over_fit_variance=feature_means((a-b)**2/variance,FIELDS).mean(0) if mask.any() else [float('nan')]*3)
    np.savez_compressed(dest,score_rows=indices,full_errors=errors,full_flags=flags,full_joint_count=joint_count,full_background_count=bg_count,A_errors=halves[0]['error'],B_errors=halves[1]['error'],A_word_count=halves[0]['word_count'],B_word_count=halves[1]['word_count'],A_word_people=halves[0]['word_people'],B_word_people=halves[1]['word_people'])
    result=dict(fold=fold,branch=branch,full_fit=record,A_people=np.unique(bank['subject'][fits[0]]),B_people=np.unique(bank['subject'][fits[1]]),reliability=reliability,CPU_seconds=time.process_time())
    write(meta,result)
    return result


def summarize(bank,pop,rows,out,worker_results,cfg):
    arrays={b:{k:np.zeros((15360,2,3)) for k in ['full','A','B']} for b in 'SO'}
    support={b:np.zeros((15360,15,3),bool) for b in 'SO'}
    cellcount={b:np.zeros((15360,15),np.int32) for b in 'SO'}
    for record in worker_results:
        b=record['branch'];fold=record['fold']
        with np.load(out/'folds'/f'{b}_fold{fold}.npz') as z:
            index=z['score_rows']
            for arm,k in [('full','full_errors'),('A','A_errors'),('B','B_errors')]:arrays[b][arm][index]=z[k]
            support[b][index]=z['full_flags'];cellcount[b][index]=z['full_joint_count']
    reference=json.loads((ROOT/cfg['source']['reference_summary']).read_text())
    examples=bank['examples'][:15360];valid=bank['valid'][:15360]
    result={}
    for b in 'SO':
        result[b]={}
        for arm,error in arrays[b].items():
            stats=hierarchy(error,examples,rows);mse=np.asarray(stats['macro']);gain=1-mse[1]/mse[0]
            positive=np.mean([np.asarray(s['value'])[1]<np.asarray(s['value'])[0] for s in stats['by_subject']],axis=0)
            result[b][arm]=dict(hierarchy=stats,gain=dict(zip(FIELDS,gain)),positive_subject_fraction=dict(zip(FIELDS,positive)))
        masses=support[b][valid].mean(0)
        g=np.array(list(result[b]['full']['gain'].values()));f=np.array(list(result[b]['full']['positive_subject_fraction'].values()))
        result[b]['full'].update(ID_seen_mass=float(masses[0]),qualified_ID_mass=float(masses[1]),background_ID_seen_mass=float(masses[2]),joint_fit_count_quantiles=np.quantile(cellcount[b][valid],[0,.1,.5,.9,1]),numerical_gate_pass=bool(min(g[:2])>0 and max(g)>=.05 and min(f[:2])>=.5 and masses[1]>=.9))
        old=np.asarray(reference['arms']['fit1024'][b]['hierarchy']['macro']);new=np.asarray(result[b]['full']['hierarchy']['macro'])
        result[b]['comparison_to_fit1024']=dict(original_background_MSE=old[0],full_background_MSE=new[0],original_conditional_MSE=old[1],full_conditional_MSE=new[1],background_relative_change=new[0]/old[0]-1,conditional_relative_change=new[1]/old[1]-1,full_conditional_gain_vs_original_background=1-new[1]/old[0])
    np.savez_compressed(out/'full_support_errors.npz',examples=examples,**{b+'_'+arm:error for b,values in arrays.items() for arm,error in values.items()},**{b+'_flags':v for b,v in support.items()})
    summary=dict(status='exhaustive_train_support_completed',full_training_positions=len(bank['examples']),full_complete_patches=int(bank['valid'].sum()),held_positions=15360,held_complete_patches=int(valid.sum()),subjects=60,physical_crops=5294,branches=result,replication=worker_results,all_full_gates=all(result[b]['full']['numerical_gate_pass'] for b in 'SO'),validation_reads=0,confirmation_reads=0,test_reads=0,AS_calls=0,exposure=cfg['crossfit']['exposure'])
    write(out/'summary.json',summary)
    print(json.dumps({b:{k:result[b]['full'][k] for k in ['gain','qualified_ID_mass','numerical_gate_pass']} for b in 'SO'}))


def replication_review_worker(task):
    fold,branch=task
    bank,pop,out=(STATE[k] for k in ['bank','population','out'])
    original=json.loads((out/'folds'/f'{branch}_fold{fold}.json').read_text())
    fit=bank['fold']!=fold
    held=(np.arange(len(bank['examples']))<15360)&(bank['fold']==fold)
    groups=[np.isin(bank['subject'],original[k]) for k in ['A_people','B_people']]
    if (groups[0]&groups[1]).any() or not np.array_equal(groups[0]|groups[1],fit):
        raise ValueError('retained disjoint person groups changed')
    half=[raw_profile(bank,g,held,branch,4*(len(pop['background_sites'])+1),True) for g in groups]
    train=bank[branch+'_attributes'][bank['valid']&fit[:,None]]
    full_variance=np.maximum(train.var(0),1e-20)
    prior_variance=(half[0]['variance']+half[1]['variance'])/2
    both_seen=half[0]['cell_seen']&half[1]['cell_seen']
    both_strong=half[0]['strong_cell']&half[1]['strong_cell']
    reviews={}
    for label,mask in [('all_held',np.ones(len(both_seen),bool)),('both_joint_seen',both_seen),('both_8patch_2person_joint',both_strong)]:
        difference=(half[0]['joint']-half[1]['joint'])[mask]
        old=feature_means(difference**2/prior_variance,FIELDS).mean(0)
        np.testing.assert_array_equal(old,original['reliability'][label]['prediction_drift_over_fit_variance'])
        updated=feature_means(difference**2/full_variance,FIELDS).mean(0)
        reviews[label]=dict(held_patches=int(mask.sum()),original_mean_group_variance_drift=old,
                           full_fit_variance_drift=updated)
    return dict(fold=fold,branch=branch,A_people=original['A_people'],B_people=original['B_people'],
                full_fit_attribute_variance=full_variance,original_mean_group_variance=prior_variance,
                corrected_drift=reviews,all_original_drift_values_independently_reproduced=True)


def replication_review(cfg,out,pop):
    with np.load(out/'full_support_bank.npz') as z:bank={k:z[k].copy() for k in z.files}
    STATE.update(bank=bank,population=pop,out=out)
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:
        results=list(pool.map(replication_review_worker,[(f,b) for f in range(4) for b in 'SO']))
    write(out/'replication_full_fit_variance_v2.json',dict(status='corrected_diagnostic_normalizer',
         original_primary_owner='summary.json',primary_scores_gains_support_unchanged=True,
         issue='Original diagnostic divided squared A/B prototype difference by the average of the two subgroup attribute variances; registered normalization uses the variance of every full-fit training patch.',
         correction='Retain every original value. Recompute only the diagnostic ratios with the registered full-fit training attribute variance for each outer fold; neither includes held people.',
         folds=results))
    print('Full-fit variance diagnostic review completed')


def review_worker(task):
    fold,branch=task
    bank,pop,out=(STATE[k] for k in ['bank','population','out'])
    fit=bank['fold']!=fold
    held=(np.arange(len(bank['examples']))<15360)&(bank['fold']==fold)
    profile=raw_profile(bank,fit,held,branch,4*(len(pop['background_sites'])+1))
    with np.load(out/'folds'/f'{branch}_fold{fold}.npz') as z:
        np.testing.assert_array_equal(profile['error'],z['full_errors'])
        valid=bank['valid'][held]
        np.testing.assert_array_equal(profile['local_support'],z['full_flags'][valid][:,1])
        np.testing.assert_array_equal(profile['cell_seen'],z['full_flags'][valid][:,2])
    return dict(fold=fold,branch=branch,errors_bitwise_equal=True,support_bitwise_equal=True,held_positions=int(held.sum()))


def review(cfg,out,rows,scales,pop,lookup):
    with np.load(out/'full_support_bank.npz') as z:bank={k:z[k].copy() for k in z.files}
    with np.load(ROOT/cfg['source']['initial_bank']) as z:
        for k in z.files:np.testing.assert_array_equal(bank[k][:61440],z[k])
    with np.load(ROOT/cfg['source']['vocabulary_checkpoint']) as z:names=json.loads(str(z['metadata']))['positions']
    native={d:{v.upper():j for j,v in enumerate(values)} for d,values in names.items()}
    cache=OrderedDict();blocks={};signatures={}
    expected_encoder=digest(ROOT/cfg['source']['encoder'])
    all_count=0
    for path in sorted({p for p,j in lookup.values()}):
        with np.load(path) as z:
            sig=json.loads(str(z['signature']))
            if sig['encoder_sha256']!=expected_encoder:raise ValueError('cached AS encoder identity changed')
            all_count+=len(z['crops']);signatures[str(path.relative_to(ROOT))]=digest(path)
    if all_count!=5294:raise ValueError('duplicated physical crop in native archive')
    exact=0
    for n,(i,t,p) in enumerate(bank['examples']):
        path,j=lookup[int(i),int(t)]
        if path not in cache:
            with np.load(path) as z:cache[path]={b:z[b].copy() for b in 'SO'}
            if len(cache)>4:cache.popitem(last=False)
        pos=native[rows[i]['dataset']][rows[i]['positions'][p].upper()]
        for b in 'SO':np.testing.assert_array_equal(bank[b][n],cache[path][b][j,pos])
        exact+=1
    probes=[]
    for dataset in sorted(names):
        eligible=[j for j,e in enumerate(bank['examples']) if rows[e[0]]['dataset']==dataset]
        selected=[eligible[0],next(j for j in eligible if j>=61440)]
        for j in selected:
            ex=bank['examples'][j:j+1]
            batch=base_tokenizer_batch(rows,ex,scales)
            valid=batch['mask'].all(-1)
            np.testing.assert_array_equal(valid[0],bank['valid'][j])
            for b in 'SO':
                raw,slices=attributes(batch[b.lower()][valid],'eeg')
                expected=np.concatenate([raw[:,slice(*slices[k])] for k in FIELDS],axis=1)
                np.testing.assert_array_equal(bank[b+'_attributes'][j][valid[0]],expected)
            probes.append(dict(dataset=dataset,example=ex[0].tolist(),attribute_and_mask_exact=True))
    STATE.update(bank=bank,population=pop,out=out)
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:
        results=list(pool.map(review_worker,[(f,b) for f in range(4) for b in 'SO']))
    write(out/'independent_review.json',dict(status='passed',all_full_bank_integers_compared=exact,original_prefix_fields_bitwise_equal=True,
         original_native_file_hashes=signatures,raw_target_probes=probes,fold_checks=results,method='Fresh bulk-held categorical reads versus original batched scoring, exact original ID archive, independent pointwise target checks',validation_reads=0,confirmation_reads=0,test_reads=0))
    print('Exhaustive saved-ID and independent readout review passed',exact)


def render(cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    summary=json.loads((out/'summary.json').read_text())
    ref=json.loads((ROOT/cfg['source']['reference_summary']).read_text())
    colors=['#a2b8c2','#4a8d99','#d17c42']
    fig,axes=plt.subplots(2,2,figsize=(13.2,8.5),layout='constrained')
    for j,b in enumerate('SO'):
        for k,(label,result) in enumerate([('256 per person',ref['arms']['fit256'][b]),('1024 per person',ref['arms']['fit1024'][b]),('All saved positions',summary['branches'][b]['full'])]):
            values=100*np.array(list(result['gain'].values()))
            axes[0,j].bar(np.arange(3)+(k-1)*.25,values,.24,color=colors[k],label=label)
        axes[0,j].axhline(0,color='#607080',lw=.8);axes[0,j].axhline(5,color='#607080',lw=.8,ls=':')
        axes[0,j].set(xticks=range(3),xticklabels=['Theta','Alpha','Local RMS'],ylabel='Error reduction over background (%)',title=b+' actual full-ID gain; same 15,360 held windows')
        axes[0,j].legend(fontsize=8)
        values=[100*ref['arms'][f][b]['qualified_ID_mass'] for f in ['fit256','fit1024']]+[100*summary['branches'][b]['full']['qualified_ID_mass']]
        axes[1,j].bar(range(3),values,color=colors)
        axes[1,j].axhline(90,color='#993d38',ls='--',label='Required 90%')
        for k,v in enumerate(values):axes[1,j].text(k,v+.7,f'{v:.2f}%',ha='center')
        axes[1,j].set(xticks=range(3),xticklabels=['256/person','1024/person','All 185,823'],ylim=(0,105),ylabel='Held complete-patch supported mass (%)',title=b+' code has >=8 fit patches from >=2 people')
        axes[1,j].legend(fontsize=8)
    for axis in axes[0]:axis.set_ylim(-22,7)
    fig.savefig(out/'exhaustive_actual_ID_information_v2.png',dpi=240,facecolor='white');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8.2),layout='constrained')
    for row,b in enumerate('SO'):
        for offset,arm in enumerate(['A','B','full']):
            values=summary['branches'][b][arm]
            axes[row,0].bar(np.arange(3)+(offset-1)*.25,100*np.array(list(values['gain'].values())),.24,color=colors[offset],label=arm)
        axes[row,0].axhline(0,color='gray',lw=.8);axes[row,0].set(xticks=range(3),xticklabels=['Theta','Alpha','Local RMS'],ylabel='Held error reduction (%)',title=b+' independent-person group readouts')
        axes[row,0].set_ylim(-22,3)
        axes[row,0].legend()
        records=sorted([v for v in summary['replication'] if v['branch']==b],key=lambda v:v['fold'])
        for col,(label,key) in enumerate([('All held patches','all_held'),('Both groups cell >=8 patches, >=2 people','both_8patch_2person_joint')],1):
            values=np.array([v['reliability'][key]['background_residual_correlation'] for v in records])
            coverage=[v['reliability'][key]['patches']/v['reliability'][key]['total_patches'] for v in records]
            for fold,line in enumerate(values):axes[row,col].plot(range(3),line,marker='o',label=f'Fold {fold}, mass {100*coverage[fold]:.1f}%')
            axes[row,col].axhline(0,color='gray',lw=.8);axes[row,col].set(xticks=range(3),xticklabels=['Theta','Alpha','Local RMS'],ylim=(-1,1),ylabel='A/B background-residual mean correlation',title=b+' '+label)
            axes[row,col].legend(fontsize=7.5)
    fig.savefig(out/'disjoint_person_profiles_v2.png',dpi=240,facecolor='white');plt.close(fig)
    write(out/'figure_manifest_v2.json',dict(status='rendered_pending_actual_image_review',figures=['exhaustive_actual_ID_information_v2.png','disjoint_person_profiles_v2.png'],scope='raw full IDs; profile replication is diagnostic; every original held window retained'))
    print('Rendered exhaustive support and disjoint-person profile figures')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--review',action='store_true');parser.add_argument('--render',action='store_true')
    parser.add_argument('--replication-review',action='store_true')
    args=parser.parse_args();cfg=yaml.safe_load((ROOT/args.config).read_text());out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    if args.render:
        render(cfg,out)
        return
    if cfg['data']['allowed_splits']!=['train'] or cfg['schema']!='dual_component_actual_id_full_support_v1':raise ValueError('training-only exhaustive support contract required')
    g0=yaml.safe_load((ROOT/cfg['source']['G0_contract']).read_text())
    manifest,scales,_,_=base_tokenizer_training_data(g0,ROOT);rows=manifest['rows']
    identity=sample_source_identity(cfg,manifest)
    pop=json.loads((ROOT/cfg['source']['original_population']).read_text());lookup=native_blocks(cfg)
    if args.review:
        review(cfg,out,rows,scales,pop,lookup)
        return
    if args.replication_review:
        replication_review(cfg,out,pop)
        return
    all_examples,counts=census(cfg,rows,pop,lookup)
    with np.load(ROOT/cfg['source']['initial_bank']) as z:bank={k:z[k].copy() for k in z.files}
    with np.load(ROOT/cfg['source']['original_bank']) as z:
        for k in z.files:np.testing.assert_array_equal(bank[k][:15360],z[k])
    oldset=set(map(tuple,bank['examples']))
    if not oldset.issubset(set(map(tuple,all_examples))):raise ValueError('initial support contains unavailable native crop position')
    extra=np.asarray([e for e in all_examples if tuple(e) not in oldset],np.int32)
    with np.load(ROOT/cfg['source']['vocabulary_checkpoint']) as z:names=json.loads(str(z['metadata']))['positions']
    (out/'additional').mkdir(exist_ok=True)
    STATE.update(cfg=cfg,rows=rows,scales=scales,out=out,lookup=lookup,names=names,population=pop,extra=extra)
    write(out/'census.json',dict(by_dataset=counts,total_positions=len(all_examples),already_collected=len(bank['examples']),additional_positions=len(extra),original_score_positions=15360,physical_crops=len(lookup),identity=identity,AS_calls=0))
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:
        prepared=list(pool.map(prepare_dataset,sorted(counts)))
    parts={k:[v] for k,v in bank.items()}
    for dataset in sorted(counts):
        with np.load(out/'additional'/f'{dataset}.npz') as z:
            for k in parts:parts[k].append(z[k].copy())
    bank={k:np.concatenate(v) for k,v in parts.items()}
    if len(bank['examples'])!=185823 or len(set(map(tuple,bank['examples'])))!=185823:raise ValueError('exhaustive bank duplicate or missing support')
    np.savez_compressed(out/'full_support_bank.npz',**bank)
    write(out/'full_population.json',dict(initial_positions=61440,full_positions=len(bank['examples']),complete_patches=int(bank['valid'].sum()),source_preparation=prepared,subject_population=pop,source_identities=identity))
    STATE.update(bank=bank)
    tasks=[(f,b) for f in range(4) for b in 'SO']
    results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:
        jobs=[pool.submit(score_worker,t) for t in tasks]
        for done in as_completed(jobs):
            results.append(done.result());write(out/'progress.json',dict(stage='full_fit_and_person_replication',completed=len(results),planned=8,CPU_seconds=budget(cfg)))
    summarize(bank,pop,rows,out,results,cfg)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
