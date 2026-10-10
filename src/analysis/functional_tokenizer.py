"""Frozen TK probes, integer-only boundary, grouped scores and paired effects.

No tokenizer/teacher is imported here. The hard reader accepts exactly integer
IDs, codebooks and identities. Probe fitting and choices use train/validation;
test labels are accessed only by scoring functions.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from pathlib import Path
import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler


def read_hard_tokens(path):
    """Strict on-disk information boundary: no pre/soft/O vectors can enter."""
    required = {'ids_eeg','ids_hb','codebook_eeg','codebook_hb','identities'}
    with np.load(path,allow_pickle=False) as data:
        if set(data.files) != required:
            raise ValueError('hard export must contain only IDs, codebooks and identities')
        identities = data['identities'].astype(str)
        values = {}
        for m in ('eeg','hb'):
            ids,book = data['ids_'+m],data['codebook_'+m]
            if (ids.dtype.kind not in 'iu' or ids.ndim!=2 or book.ndim!=2
                    or len(ids)!=len(identities) or np.any(ids<0) or np.any(ids>=len(book))
                    or not np.isfinite(book).all()):
                raise ValueError('invalid integer token/codebook contract')
            values[m]=book[ids]
        if len(set(identities))!=len(identities):
            raise ValueError('duplicate export identities')
    return identities,values


def join_export(expected,identities,arrays):
    if set(expected)!=set(identities):
        raise ValueError('export identity population mismatch')
    lookup={name:i for i,name in enumerate(identities)}
    order=[lookup[name] for name in expected]
    return {k:np.asarray(v)[order] for k,v in arrays.items()}


def trial_features(rows,values,regions):
    """One row per original event, all region replicas concatenated in order."""
    groups=defaultdict(list)
    for i,r in enumerate(rows): groups[r['trial_id']].append(i)
    metadata,features=[],[]
    for trial,indices in sorted(groups.items()):
        byregion={rows[i]['anchor']['region']:i for i in indices}
        if set(byregion)!=set(regions) or len(indices)!=len(regions):
            raise ValueError('complete unique region support required per native trial')
        metadata.append(rows[indices[0]])
        features.append(np.concatenate([np.asarray(values[byregion[region]]).ravel() for region in regions]))
    return metadata,np.asarray(features)


def budget_indices(rows,split,fraction,seed):
    """Whole native trials/blocks; validation labels count in the same budget.

    Hash ordering is feature/label blind and nested across budgets and arms.
    Regions have already been joined to the original event by trial_features.
    """
    indices=[i for i,r in enumerate(rows) if r['split']==split]
    indices.sort(key=lambda i:hashlib.sha256(f'{seed}:{rows[i]["trial_id"]}'.encode()).digest())
    return np.asarray(indices[:int(np.floor(len(indices)*fraction))],dtype=int)


def subject_scores(rows,truth,prediction,classes):
    scores=[]
    for subject in sorted({r['subject'] for r in rows}):
        records=[]
        for record in sorted({r['record'] for r in rows if r['subject']==subject}):
            ix=np.array([i for i,r in enumerate(rows) if r['subject']==subject and r['record']==record])
            # Balanced accuracy averages recalls over supported classes; missing
            # classes are retained in the support record, not fabricated as 0.
            recalls=[np.mean(prediction[ix][truth[ix]==c]==c) for c in classes if np.any(truth[ix]==c)]
            records.append(dict(balanced_accuracy=float(np.mean(recalls)),
                macro_F1=float(f1_score(truth[ix],prediction[ix],labels=classes,average='macro',zero_division=0)),
                classes_present=int(len(set(truth[ix]))),trials=len(ix)))
        scores.append(dict(subject=subject,balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in records])),
            macro_F1=float(np.mean([r['macro_F1'] for r in records])),
            records=len(records),trials=sum(r['trials'] for r in records),
            minimum_record_classes=min(r['classes_present'] for r in records)))
    return scores


def fixed_features(features,training,dimension,seed=0):
    """Same train-fitted standardized PCA budget for every representation."""
    features=np.asarray(features,float)
    if features.ndim!=2 or not np.isfinite(features).all() or len(training)<2:
        raise ValueError('finite features and nonempty training partition required')
    scale=StandardScaler().fit(features[training])
    x=scale.transform(features)
    if np.max(np.var(x[training],axis=0))<1e-14:
        return np.zeros((len(x),1)),dict(dimensions=1,fit_samples=len(training),constant_representation=True)
    d=min(dimension,x.shape[1],len(training)-1)
    pca=PCA(d,svd_solver='randomized',random_state=seed).fit(x[training])
    return pca.transform(x),dict(dimensions=d,fit_samples=len(training))


def classification_probe(rows,features,cfg,*,seed):
    results,failures=[],[]
    for task in cfg['data']['tasks']:
        selected=[i for i,r in enumerate(rows) if r['task']==task]
        if not selected: continue
        meta=[rows[i] for i in selected]
        classes=cfg['data']['task_labels'][task]
        y=np.array([classes.index(r['condition']) for r in meta])
        train_all=np.array([i for i,r in enumerate(meta) if r['split']=='train'])
        test=np.array([i for i,r in enumerate(meta) if r['split']=='test'])
        if len(train_all)<2 or not len(test):
            failures.append(dict(task=task,reason='empty_train_or_test'));continue
        x,coordinate=fixed_features(np.asarray(features)[selected],train_all,cfg['evaluation']['probe_dimensions'],seed)
        for fraction in cfg['evaluation']['label_fractions']:
            train=budget_indices(meta,'train',fraction,cfg['seed'])
            val=budget_indices(meta,'validation',fraction,cfg['seed'])
            identity=dict(task=task,label_fraction=fraction,training_labels=len(train),validation_labels=len(val),
                total_labels=len(train)+len(val),probe_dimensions=coordinate['dimensions'])
            if not len(train) or not len(val) or set(y[train])!=set(range(len(classes))) or set(y[val])!=set(range(len(classes))):
                failures.append(dict(identity,reason='label_budget_missing_class_or_support'));continue
            best=None
            for c in cfg['evaluation']['probe_C']:
                fit=LogisticRegression(C=c,max_iter=1000,solver='lbfgs',class_weight='balanced',random_state=seed)
                fit.fit(x[train],y[train])
                ss=subject_scores([meta[i] for i in val],y[val],fit.predict(x[val]),list(range(len(classes))))
                score=np.mean([r['balanced_accuracy'] for r in ss])
                if best is None or score>best[0]: best=(score,c,fit)
            prediction=best[2].predict(x[test])
            for score in subject_scores([meta[i] for i in test],y[test],prediction,list(range(len(classes)))):
                results.append(dict(identity,**score,probe_C=best[1],validation_score=float(best[0])))
    return results,failures


def session_probe_split(rows):
    """Secondary within-subject transfer, on subjects unseen by the tokenizer.

    The original event and region dependencies stay together. Inner probe
    training/validation/test are the first/middle/last native sessions. Tasks
    lacking three verified sessions are unavailable, never guessed from labels.
    """
    selected=[];indices=[]
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        if r['split']=='test': groups[(r['dataset'],r['subject'],r['task'])].append(i)
    for group in groups.values():
        session_times={}
        for i in group:
            r=rows[i];session_times[r['session']]=min(session_times.get(r['session'],float('inf')),r['eeg_start_s'])
        sessions=sorted(session_times,key=lambda s:(session_times[s],s))
        if len(sessions)<3: continue
        split={s:('test' if j==len(sessions)-1 else 'validation' if j==len(sessions)-2 else 'train') for j,s in enumerate(sessions)}
        for i in group:
            indices.append(i);selected.append(dict(rows[i],split=split[rows[i]['session']]))
    return selected,np.asarray(indices,dtype=int)


def ridge_readout(rows,features,target,cfg,*,group_field=None,reduce_features=True):
    """Trajectory RMSE in train-target SD, with subject-equal validation choice."""
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    val=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    test=np.array([i for i,r in enumerate(rows) if r['split']=='test'])
    if min(len(train),len(val),len(test))<2: raise ValueError('missing grouped ridge support')
    if reduce_features:
        x,_=fixed_features(features,train,cfg['evaluation']['probe_dimensions'])
    else:
        # Nested completion baselines keep every visible Hb context feature
        # unchanged when EEG is appended. Never refit a joint PCA that could
        # discard different Hb information in q0 and q1.
        scaler=StandardScaler().fit(np.asarray(features)[train])
        keep=scaler.var_>1e-14
        x=scaler.transform(features)[:,keep]
        if x.shape[1]==0:x=np.zeros((len(features),1))
    y=np.asarray(target).reshape(len(rows),-1)
    sd=max(float(y[train].std()),1e-8)
    best=None
    for alpha in cfg['evaluation']['ridge_alpha']:
        fit=Ridge(alpha=alpha).fit(x[train],y[train])
        errors=np.mean((fit.predict(x[val])-y[val])**2,axis=1)/sd**2
        score=np.mean([np.mean(errors[[rows[i]['subject']==s for i in val]]) for s in {rows[i]['subject'] for i in val}])
        if best is None or score<best[0]: best=(score,alpha,fit)
    prediction=best[2].predict(x[test])
    mse=np.mean((prediction-y[test])**2,axis=1)/sd**2
    mean_mse=np.mean((y[train].mean(0)-y[test])**2,axis=1)/sd**2
    result=[]
    for s in sorted({rows[i]['subject'] for i in test}):
        groups=sorted({rows[i][group_field] for i in test if rows[i]['subject']==s}) if group_field else [None]
        for group in groups:
            selected=np.array([rows[i]['subject']==s and (group_field is None or rows[i][group_field]==group) for i in test])
            row=dict(subject=s,NRMSE=float(np.sqrt(np.mean(mse[selected]))),
                mean_baseline_NRMSE=float(np.sqrt(np.mean(mean_mse[selected]))),ridge_alpha=best[1],target_training_SD=sd)
            if group_field:
                row.update({group_field:group,'estimated_delta_mean':prediction[selected].mean(0).tolist(),
                    'true_delta_mean':y[test][selected].mean(0).tolist()})
            result.append(row)
    return result


def code_health(ids,codebook):
    ids=np.asarray(ids);book=np.asarray(codebook)
    counts=np.bincount(ids.ravel(),minlength=len(book)); p=counts/max(counts.sum(),1)
    nz=p>0
    singular=np.linalg.svd(book-book.mean(0),compute_uv=False)**2
    participation=float(singular.sum()**2/max(np.sum(singular**2),1e-20))
    norms=np.linalg.norm(book,axis=1)
    cosine=(book@book.T)/np.maximum(norms[:,None]*norms[None,:],1e-12)
    np.fill_diagonal(cosine,-1)
    return dict(active_codes=int(nz.sum()),effective_codes=float(np.exp(-np.sum(p[nz]*np.log(p[nz])))),
        dead_codes=int((~nz).sum()),minimum_active_support=int(counts[nz].min()) if nz.any() else 0,
        maximum_mass=float(p.max()),participation_rank=participation,
        near_duplicate_pairs=int(np.triu(cosine>.99,1).sum()),counts=counts.tolist())


def align_ids(reference,other,training):
    """Train-only maximum-overlap alignment; test scores never choose permutation."""
    reference,other=np.asarray(reference),np.asarray(other)
    k=max(int(reference.max()),int(other.max()))+1
    contingency=np.zeros((k,k),int)
    np.add.at(contingency,(reference[training].ravel(),other[training].ravel()),1)
    left,right=linear_sum_assignment(-contingency)
    permutation=np.empty(k,int);permutation[right]=left
    return permutation[other],permutation


def paired_interval(values,seed,repeats=2000):
    values=np.asarray(values,float)
    if not len(values): return dict(mean=None,low=None,high=None,subjects=0)
    rng=np.random.default_rng(seed)
    samples=values[rng.integers(len(values),size=(repeats,len(values)))].mean(1)
    return dict(mean=float(values.mean()),low=float(np.quantile(samples,.025)),
        high=float(np.quantile(samples,.975)),subjects=len(values),interval='paired_subject_percentile_95pct')


def compare_arms(records,cfg):
    """Seeds are averaged within subject; never added to biological sample n."""
    comparisons=[('teacher_continuous','CT','pre','C0','pre'),
        ('teacher_discrete','QT','hard','Q0','hard'),('hard_retention','QT','hard','CT','pre'),
        ('vs_smooth','CT','pre','C-smooth','pre'),('vs_shuffle','CT','pre','C-shuffle','pre'),
        ('wide_vs_functional','C-wide','pre','CT','pre'),('stop_vs_joint','C-stop','pre','CT','pre')]
    output=[]
    for name,a,ap,b,bp in comparisons:
        for task in cfg['data']['tasks']:
            for representation in ('S','O','SO'):
                for modality in ('eeg','hb','both'):
                    for fraction in cfg['evaluation']['label_fractions']:
                        selected=[r for r in records if r['task']==task and r['representation']==representation
                            and r['modality']==modality and r['label_fraction']==fraction]
                        left={(r['subject'],r['seed']):r['balanced_accuracy'] for r in selected if r['arm']==a and r['path']==ap}
                        right={(r['subject'],r['seed']):r['balanced_accuracy'] for r in selected if r['arm']==b and r['path']==bp}
                        pairs=sorted(set(left)&set(right)); differences=defaultdict(list)
                        for subject,seed in pairs: differences[subject].append(left[(subject,seed)]-right[(subject,seed)])
                        complete={s:v for s,v in differences.items() if len(v)==len(cfg['training']['seeds'])}
                        interval=paired_interval([np.mean(v) for v in complete.values()],cfg['seed'],cfg['evaluation']['bootstrap_repeats'])
                        verdict='insufficient_evidence'
                        if not complete or len(pairs)!=len(left) or len(pairs)!=len(right): verdict='invalid_experiment'
                        elif interval['low']>cfg['evaluation']['task_gain_margin']: verdict='benefit'
                        elif interval['high']<-cfg['evaluation']['task_gain_margin']: verdict='degradation'
                        output.append(dict(comparison=name,task=task,representation=representation,modality=modality,
                            label_fraction=fraction,**interval,verdict=verdict,paired_seed_subjects=len(pairs),
                            seeds_per_subject=len(cfg['training']['seeds']),multiplicity='exploratory_unadjusted_intervals_no_significance_claim'))
                        if name=='hard_retention':
                            output[-1]['retention']=('invalid' if verdict=='invalid_experiment' else
                                'within_margin' if interval['low']>=-cfg['evaluation']['hard_retention_margin'] else
                                'exceeds_margin' if interval['high']<-cfg['evaluation']['hard_retention_margin'] else 'uncertain')
    return output


def branch_increments(records,cfg):
    output=[]
    for arm in cfg['training']['arms']:
        path='hard' if arm in ('Q0','QT') else 'pre'
        for task in cfg['data']['tasks']:
            for modality in ('eeg','hb','both'):
                for fraction in cfg['evaluation']['label_fractions']:
                    select=[r for r in records if r['arm']==arm and r['path']==path and r['task']==task
                        and r['modality']==modality and r['label_fraction']==fraction]
                    for name,reference in (('Delta_O','S'),('Delta_S','O')):
                        full={(r['subject'],r['seed']):r['balanced_accuracy'] for r in select if r['representation']=='SO'}
                        base={(r['subject'],r['seed']):r['balanced_accuracy'] for r in select if r['representation']==reference}
                        bysubject=defaultdict(list)
                        for key in set(full)&set(base):bysubject[key[0]].append(full[key]-base[key])
                        values=[np.mean(v) for v in bysubject.values() if len(v)==len(cfg['training']['seeds'])]
                        output.append(dict(arm=arm,task=task,modality=modality,label_fraction=fraction,
                            increment=name,**paired_interval(values,cfg['seed'],cfg['evaluation']['bootstrap_repeats']),
                            meaning='frozen_probe_prediction_increment_not_mutual_information_or_PID'))
    return output


def coupling_features(hb,extra,hidden=(32,48),shift=0,token_mask=None):
    """Inputs omit hidden raw bins before feature construction; no teacher inputs.

    Hb is [N,80,2]; EEG token embedding is [N,10,D]. Shift and support are
    propagated together; callers intersect support across all planned nulls.
    """
    h=np.asarray(hb,float).copy(); start,stop=hidden
    h[:,start:stop]=0
    mask=np.ones_like(h,bool);mask[:,start:stop]=False
    context=np.concatenate((h.reshape(len(h),-1),mask.reshape(len(h),-1)),1)
    if extra is None: return context,np.ones(len(h),bool)
    e=np.asarray(extra,float)
    support=np.ones(e.shape[:2],bool) if token_mask is None else np.asarray(token_mask,bool).copy()
    if support.shape!=e.shape[:2]: raise ValueError('token support shape mismatch')
    if shift:
        e=np.roll(e,shift,axis=1);support=np.roll(support,shift,axis=1)
        if shift>0: support[:,:shift]=False
        else: support[:,shift:]=False
    e=np.where(support[...,None],e,0)
    return np.concatenate((context,e.reshape(len(e),-1),support),1),support.any(1)


def frozen_prototypes(rows,ids,measurements,training=False):
    """Conditional associations, with explicit subject/record support."""
    result=[]
    for token in np.unique(ids):
        select=(ids==token)
        sample,patch=np.nonzero(select)
        if not len(sample): continue
        waves=measurements[sample,patch]
        durations=[]
        for sequence in select:
            edges=np.diff(np.r_[False,sequence,False].astype(int))
            durations.extend((np.flatnonzero(edges==-1)-np.flatnonzero(edges==1))*2.)
        result.append(dict(token=int(token),patches=len(sample),subjects=len({rows[i]['subject'] for i in sample}),
            records=len({(rows[i]['subject'],rows[i]['record']) for i in sample}),
            mean_waveform=waves.mean(0).tolist(),amplitude_mean=float(np.ptp(waves,axis=1).mean()),
            slope_per_second_mean=float(np.mean(waves[:,-1]-waves[:,0])/((waves.shape[1]-1)*.25)),
            run_duration_median_s=float(np.median(durations)),run_duration_P90_s=float(np.quantile(durations,.9)),
            duration_scope='consecutive_2s_patches_clipped_at_20s_window_boundaries',
            subject_patch_counts=dict(Counter(rows[i]['subject'] for i in sample)),
            record_patch_counts=dict(Counter(rows[i]['subject']+'/'+rows[i]['record'] for i in sample)),
            dataset_patch_counts=dict(Counter(rows[i]['dataset'] for i in sample)),
            region_patch_counts=dict(Counter(rows[i]['anchor']['region'] for i in sample)),
            meaning='conditional_response_association_not_individual_physiological_parameter'))
    return result
