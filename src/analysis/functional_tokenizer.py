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


def read_hard_tokens(path,*,return_ids=False):
    """Strict on-disk information boundary: no pre/soft/O vectors can enter."""
    required = {'ids_eeg','ids_hb','codebook_eeg','codebook_hb','identities'}
    with np.load(path,allow_pickle=False) as data:
        if set(data.files) != required:
            raise ValueError('hard export must contain only IDs, codebooks and identities')
        identities = data['identities'].astype(str)
        values,integer_ids = {},{}
        for m in ('eeg','hb'):
            ids,book = data['ids_'+m],data['codebook_'+m]
            if (ids.dtype.kind not in 'iu' or ids.ndim!=2 or book.ndim!=2
                    or len(ids)!=len(identities) or np.any(ids<0) or np.any(ids>=len(book))
                    or not np.isfinite(book).all()):
                raise ValueError('invalid integer token/codebook contract')
            values[m]=book[ids]
            integer_ids[m]=ids.copy()
        if len(set(identities))!=len(identities):
            raise ValueError('duplicate export identities')
    return (identities,values,integer_ids) if return_ids else (identities,values)


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


def relation_trial_layout(rows,regions):
    """Join complete regional replicas by identity, never by array order."""
    groups=defaultdict(list)
    for i,row in enumerate(rows): groups[row['trial_id']].append(i)
    metadata,indices=[],[]
    for trial,group in sorted(groups.items()):
        byregion={rows[i]['anchor']['region']:i for i in group}
        if len(group)!=len(regions) or set(byregion)!=set(regions):
            raise ValueError('relation probes require complete unique trial regions')
        first=rows[group[0]]
        fields=('dataset','subject','record','session','task','condition','split')
        if any(any(rows[i][k]!=first[k] for k in fields) for i in group):
            raise ValueError('regional trial metadata disagree')
        metadata.append(first);indices.append([byregion[r] for r in regions])
    return metadata,np.asarray(indices,dtype=int)


def relation_patch_targets(values):
    """Native/prepared E,HbO,HbR samples -> 2 s means in E,X,T coordinates."""
    values=np.asarray(values,float)
    if values.ndim!=3 or values.shape[1:]!=(80,3) or not np.isfinite(values).all():
        raise ValueError('finite [regional windows,80,3] target required')
    transformed=np.stack((values[...,0],.35*values[...,1]-.65*values[...,2],
        values[...,1]+values[...,2]),axis=-1)
    return transformed.reshape(len(values),10,8,3).mean(axis=2)


def relation_subject_mse(rows,row_errors):
    """Average patches upstream, then region -> trial -> record -> subject."""
    row_errors=np.asarray(row_errors,float)
    if row_errors.shape!=(len(rows),) or not np.isfinite(row_errors).all():
        raise ValueError('one finite mean squared error per regional window required')
    trials=defaultdict(list)
    for row,error in zip(rows,row_errors):
        key=(row['dataset'],row['subject'],row['record'],row['trial_id'])
        trials[key].append(float(error))
    records=defaultdict(list)
    for (dataset,subject,record,trial),errors in trials.items():
        records[(dataset,subject,record)].append(float(np.mean(errors)))
    subjects=defaultdict(list)
    for (dataset,subject,record),errors in records.items():
        subjects[(dataset,subject)].append((float(np.mean(errors)),len(errors)))
    return [dict(dataset=dataset,subject=subject,MSE=float(np.mean([e for e,_ in entries])),
        NRMSE=float(np.sqrt(np.mean([e for e,_ in entries]))),records=len(entries),
        trials=sum(n for _,n in entries)) for (dataset,subject),entries in sorted(subjects.items())]


def _relation_partitions(rows):
    if any(r['split'] not in ('train','validation','test') for r in rows):
        raise ValueError('unknown relation probe partition')
    partitions={s:np.array([r['split']==s for r in rows]) for s in ('train','validation','test')}
    populations={s:{(r['dataset'],r['subject']) for r in rows if r['split']==s} for s in partitions}
    if any(populations[a]&populations[b] for a,b in (('train','validation'),('train','test'),('validation','test'))):
        raise ValueError('relation probe subjects overlap across partitions')
    return partitions


def relation_patch_readout(rows,features,target,cfg,coordinate_names):
    """Train-fitted ridge on individual patches, subject-grouped selection/score.

    Every target coordinate has its own training SD. Region and patch copies
    contribute to fitting, but never count as independent subjects in scoring.
    """
    x,y=np.asarray(features,float),np.asarray(target,float)
    if x.ndim!=3 or y.ndim!=3 or x.shape[:2]!=y.shape[:2] or len(x)!=len(rows):
        raise ValueError('aligned [regional windows,patches,coordinates] required')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or len(coordinate_names)!=y.shape[-1]:
        raise ValueError('finite features and named target coordinates required')
    partitions=_relation_partitions(rows)
    if any(not mask.any() for mask in partitions.values()): raise ValueError('missing grouped patch partition')
    train=partitions['train'];val=partitions['validation'];test=partitions['test']
    scaler=StandardScaler().fit(x[train].reshape(-1,x.shape[-1]))
    standardized=scaler.transform(x.reshape(-1,x.shape[-1])).reshape(x.shape)
    target_sd=np.maximum(y[train].reshape(-1,y.shape[-1]).std(0),1e-8)
    normalized=y/target_sd
    train_x=standardized[train].reshape(-1,x.shape[-1]);train_y=normalized[train].reshape(-1,y.shape[-1])
    best=None
    for alpha in cfg['evaluation']['ridge_alpha']:
        fit=Ridge(alpha=float(alpha)).fit(train_x,train_y)
        prediction=fit.predict(standardized[val].reshape(-1,x.shape[-1])).reshape(normalized[val].shape)
        errors=np.square(prediction-normalized[val]).mean(axis=(1,2))
        scores=relation_subject_mse([r for r,m in zip(rows,val) if m],errors)
        score=float(np.mean([r['MSE'] for r in scores]))
        if best is None or score<best[0]: best=(score,float(alpha),fit)
    prediction=best[2].predict(standardized[test].reshape(-1,x.shape[-1])).reshape(normalized[test].shape)
    errors=np.square(prediction-normalized[test]).mean(axis=1)
    mean_prediction=normalized[train].mean(axis=(0,1))
    mean_errors=np.square(normalized[test]-mean_prediction).mean(axis=1)
    test_rows=[r for r,m in zip(rows,test) if m];result=[]
    coordinates=[(name,errors[:,i],mean_errors[:,i]) for i,name in enumerate(coordinate_names)]
    if len(coordinate_names)>1:
        coordinates.append(('coordinate_mean',errors.mean(axis=1),mean_errors.mean(axis=1)))
    for name,coordinate_errors,baseline_errors in coordinates:
        scores=relation_subject_mse(test_rows,coordinate_errors)
        baselines={(r['dataset'],r['subject']):r for r in relation_subject_mse(test_rows,baseline_errors)}
        for score in scores:
            result.append(dict(score,coordinate=name,ridge_alpha=best[1],validation_MSE=best[0],
                mean_baseline_NRMSE=baselines[(score['dataset'],score['subject'])]['NRMSE'],
                target_training_SD=dict(zip(coordinate_names,target_sd.tolist())),
                patches_per_regional_window=x.shape[1]))
    return result


def relation_null_donors(rows,regions,seed):
    """Whole-trial eligibility and no-self, within-record null donor indices."""
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        groups[tuple(row[k] for k in ('dataset','subject','record','condition','split'))].append(i)
    eligible=np.array(sorted(i for group in groups.values() if len(group)>=2 for i in group),dtype=int)
    kept=[rows[i] for i in eligible];local_groups=defaultdict(list)
    for i,row in enumerate(kept):
        local_groups[tuple(row[k] for k in ('dataset','subject','record','condition','split'))].append(i)
    rng=np.random.default_rng(seed)
    spatial=np.empty((len(kept),len(regions)),dtype=int)
    temporal=np.empty((len(kept),len(regions),10),dtype=int)
    for group in local_groups.values():
        for region in range(len(regions)):
            ordered=rng.permutation(group)
            spatial[ordered,region]=np.roll(ordered,int(rng.integers(1,len(group))))
            for patch in range(10):
                ordered=rng.permutation(group)
                temporal[ordered,region,patch]=np.roll(ordered,int(rng.integers(1,len(group))))
    return eligible,spatial,temporal


def relation_feature_blocks(values,ids=None,*,codebook_size=128):
    """Trial [region,patch,dimension] marginals, temporal and spatial blocks.

    Spatial lag-1 products retain both region directions; no covariance or
    causal interpretation is implied by these uncentered products.
    """
    z=np.asarray(values,float)
    if z.ndim!=4 or z.shape[2]!=10 or not np.isfinite(z).all():
        raise ValueError('finite [trial,region,10,dimension] representation required')
    n,regions,_,dimension=z.shape
    base=[z.mean(axis=2).reshape(n,regions*dimension),z.std(axis=2).reshape(n,regions*dimension)]
    if ids is not None:
        ids=np.asarray(ids)
        if ids.shape!=z.shape[:3] or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=codebook_size):
            raise ValueError('aligned integer IDs and complete codebook capacity required')
        histogram=np.zeros((n,regions,codebook_size))
        for token in range(codebook_size): histogram[:,:,token]=(ids==token).mean(axis=2)
        base.append(histogram.reshape(n,regions*codebook_size))
    temporal=np.concatenate([(z[:,:,:-lag]*z[:,:,lag:]).mean(axis=2) for lag in (1,2)]+
        [np.abs(np.diff(z,axis=2)).mean(axis=2),z[:,:,-1]-z[:,:,0]],axis=-1).reshape(n,-1)
    spatial=[]
    for left in range(regions):
        for right in range(left+1,regions):
            spatial.extend([(z[:,left]*z[:,right]).mean(axis=1),
                (z[:,left,:-1]*z[:,right,1:]).mean(axis=1),
                (z[:,right,:-1]*z[:,left,1:]).mean(axis=1)])
    return dict(marginal=np.concatenate(base,axis=-1),temporal=temporal,
        spatial=np.concatenate(spatial,axis=-1) if spatial else np.empty((n,0)))


def relation_classification(rows,features,integer_ids,cfg,*,seed):
    """Fixed-budget relationship screen with one common eligible population."""
    _relation_partitions(rows)
    keep=np.array([i for i,r in enumerate(rows) if r['task'] in cfg['data']['tasks']],dtype=int)
    rows=[rows[i] for i in keep]
    features={m:np.asarray(z)[keep] for m,z in features.items()}
    if integer_ids is not None: integer_ids={m:np.asarray(z)[keep] for m,z in integer_ids.items()}
    regions=cfg['data']['regions'];meta,layout=relation_trial_layout(rows,regions)
    eligible,spatial_donors,temporal_donors=relation_null_donors(meta,regions,seed)
    selected=[meta[i] for i in eligible]
    eligible_set=set(eligible.tolist())
    exclusions=Counter((r['dataset'],r['split']) for i,r in enumerate(meta) if i not in eligible_set)
    support=dict(planned_trials=len(meta),eligible_trials=len(eligible),excluded_trials=len(meta)-len(eligible),
        excluded_by_dataset_split=[dict(dataset=k[0],split=k[1],trials=v) for k,v in sorted(exclusions.items())],
        reason='fewer_than_two_trials_in_dataset_subject_record_condition_split',
        eligibility='identical_trial_population_for_all_relationship_variants')
    if not len(eligible): return [],[dict(reason='no_null_eligible_trials')],support
    blocks={};region_index=np.arange(len(regions))[None,:]
    for modality,z in features.items():
        values=np.asarray(z)[layout][eligible]
        token_ids=np.asarray(integer_ids[modality])[layout][eligible] if integer_ids is not None else None
        real=relation_feature_blocks(values,token_ids)
        spatial_values=values[spatial_donors,region_index]
        temporal_values=values[temporal_donors,region_index[:,:,None],np.arange(10)[None,None,:]]
        real['null_temporal']=relation_feature_blocks(temporal_values)['temporal']
        real['null_spatial']=relation_feature_blocks(spatial_values)['spatial']
        blocks[modality]=real
    records,failures=[],[]
    for modality in ('eeg','hb','both'):
        names=('eeg','hb') if modality=='both' else (modality,)
        combine=lambda name:np.concatenate([blocks[m][name] for m in names],axis=-1)
        marginal,temporal,spatial=combine('marginal'),combine('temporal'),combine('spatial')
        variants=dict(marginal=marginal,temporal=np.concatenate((marginal,temporal),axis=-1),
            spatial=np.concatenate((marginal,temporal,spatial),axis=-1),
            null_temporal=np.concatenate((marginal,combine('null_temporal')),axis=-1),
            null_spatial=np.concatenate((marginal,temporal,combine('null_spatial')),axis=-1))
        for variant,x in variants.items():
            good,bad=classification_probe(selected,x,cfg,seed=cfg['seed'])
            for row in good:
                datasets={r['dataset'] for r in selected if r['subject']==row['subject'] and r['task']==row['task']}
                if len(datasets)!=1: raise ValueError('classification subject/task identity spans datasets')
                records.append(dict(row,dataset=next(iter(datasets)),modality=modality,
                    variant=variant,input_dimensions=x.shape[1]))
            failures.extend(dict(r,modality=modality,variant=variant) for r in bad)
    return records,failures,support


def shape_smooth_coordinates(smooth,manifest):
    """Undo parent smooth standardization, then invert E,X,T -> E,HbO,HbR."""
    values=np.asarray(smooth,float)
    mean,scale=np.asarray(manifest['target_mean'],float),np.asarray(manifest['target_scale'],float)
    if (values.ndim!=3 or values.shape[1:]!=(80,3) or mean.shape!=(3,) or scale.shape!=(3,)
            or np.any(scale<=0) or not all(np.isfinite(v).all() for v in (values,mean,scale))):
        raise ValueError('finite parent smooth arrays and positive training coordinate scales required')
    original=values*scale+mean
    eeg,x,total=np.moveaxis(original,-1,0)
    return np.stack((eeg,x+.65*total,.35*total-x),axis=-1)


def read_shape_tokens(path):
    """Strict shape-ID consumer: no scales, ridge, continuous or soft fields."""
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!={'c_ids','r_ids','c_centers','r_centers','identities'}:
            raise ValueError('shape hard export must contain only IDs, frozen centers and identities')
        identities=data['identities'].astype(str);parts=[];shape=None
        if identities.ndim!=1 or len(set(identities))!=len(identities): raise ValueError('unique shape identities required')
        for name in ('c','r'):
            ids,centers=data[name+'_ids'],data[name+'_centers']
            if (ids.dtype.kind not in 'iu' or ids.shape!=(len(identities),10) or centers.ndim!=3
                    or centers.shape[1]!=8 or np.any(ids<0) or np.any(ids>=len(centers))
                    or not np.isfinite(centers).all()):
                raise ValueError('invalid integer shape IDs or frozen centers')
            if shape is not None and centers.shape!=shape: raise ValueError('shape component codebooks disagree')
            shape=centers.shape
            parts.append(centers[ids].reshape(len(identities),10,-1))
    return identities,np.concatenate(parts,axis=-1)


def read_shape_sidecars(path):
    """Continuous scales are a separate consumed representation."""
    required={'c_mu','r_mu','c_amplitude','r_amplitude','identities'}
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!=required: raise ValueError('scale export must contain only means, amplitudes and identities')
        identities=data['identities'].astype(str);parts=[];shape=None
        if identities.ndim!=1 or len(set(identities))!=len(identities): raise ValueError('unique scale identities required')
        for name in ('c','r'):
            mu,amplitude=data[name+'_mu'],data[name+'_amplitude']
            if (mu.ndim!=3 or mu.shape[:2]!=(len(identities),10) or amplitude.shape!=mu.shape[:2]
                    or not np.isfinite(mu).all() or not np.isfinite(amplitude).all() or np.any(amplitude<=0)):
                raise ValueError('invalid continuous scale sidecars')
            if shape is not None and mu.shape!=shape: raise ValueError('scale component supports disagree')
            shape=mu.shape;parts.extend((mu,amplitude[...,None]))
    return identities,np.concatenate(parts,axis=-1)


def read_shape_hybrid(hard_path,sidecar_path):
    """Fixed mu + RMS * shape decoder consuming both separate exports."""
    identities,shape=read_shape_tokens(hard_path)
    sidecar_ids,sidecars=read_shape_sidecars(sidecar_path)
    sidecars=join_export(identities,sidecar_ids,dict(scales=sidecars))['scales']
    channels=(sidecars.shape[-1]-2)//2
    if channels<1 or sidecars.shape[-1]!=2*channels+2 or shape.shape[-1]!=16*channels:
        raise ValueError('shape and scale modality dimensions disagree')
    components=[]
    for index in range(2):
        start=index*(channels+1)
        mu=sidecars[...,start:start+channels]
        amplitude=sidecars[...,start+channels]
        normalized=shape[...,index*8*channels:(index+1)*8*channels].reshape(shape.shape[:2]+(8,channels))
        components.append((mu[...,None,:]+amplitude[...,None,None]*normalized).reshape(shape.shape[:2]+(8*channels,)))
    return identities,np.concatenate(components,axis=-1)


def shape_conditional_fidelity(rows,estimated,physical,coordinate_names):
    """Same-patch waveform agreement with the common conditional model target."""
    prediction,target=np.asarray(estimated,float),np.asarray(physical,float)
    if (prediction.shape!=target.shape or prediction.ndim!=4 or prediction.shape[0]!=len(rows)
            or prediction.shape[1:3]!=(10,8) or prediction.shape[-1]!=len(coordinate_names)
            or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
        raise ValueError('aligned finite conditional waveforms required')
    partitions=_relation_partitions(rows)
    if not partitions['train'].any() or not partitions['test'].any(): raise ValueError('missing conditional fidelity partition')
    sd=np.maximum(target[partitions['train']].std(axis=(0,1,2)),1e-8)
    errors=np.square((prediction[partitions['test']]-target[partitions['test']])/sd).mean(axis=(1,2))
    test_rows=[r for r,m in zip(rows,partitions['test']) if m];result=[]
    coordinates=[(name,errors[:,i]) for i,name in enumerate(coordinate_names)]
    if len(coordinate_names)>1: coordinates.append(('coordinate_mean',errors.mean(axis=1)))
    for coordinate,coordinate_error in coordinates:
        result.extend(dict(score,coordinate=coordinate,target_training_SD=dict(zip(coordinate_names,sd.tolist())),
            target='same_parent_conditional_physical_waveform_not_physiological_truth')
            for score in relation_subject_mse(test_rows,coordinate_error))
    return result


def shape_forecast_readout(rows,features,hb,cfg,context,*,donors=None,perturbation=None):
    """Next-patch observed Hb changes from current/past local shape tokens.

    Independent readouts and target scales are fitted for each target ROI.
    Complete trial support makes the final equal-ROI subject MSE equivalent to
    patch -> ROI -> trial -> record -> subject aggregation.
    """
    if context not in ('own_current','own_history','spatial_history'):
        raise ValueError('unknown current/past shape readout context')
    z,h=np.asarray(features,float),np.asarray(hb,float)
    if (z.ndim!=3 or z.shape[:2]!=(len(rows),10) or h.shape!=(len(rows),10,8,2)
            or not np.isfinite(z).all() or not np.isfinite(h).all()):
        raise ValueError('finite aligned ten-patch features and native-bin Hb targets required')
    regions=cfg['data']['regions'];meta,layout=relation_trial_layout(rows,regions)
    if perturbation is not None:
        if (context,perturbation) not in (('own_history','wrong_previous'),('spatial_history','wrong_other_ROIs')):
            raise ValueError('relationship perturbation does not match readout context')
        donors=np.asarray(donors)
        if (donors.shape!=(len(meta),len(regions)) or donors.dtype.kind not in 'iu'
                or np.any(donors<0) or np.any(donors>=len(meta))
                or np.any(donors==np.arange(len(meta))[:,None])):
            raise ValueError('one no-self native trial donor per target ROI required')
        strata=('dataset','subject','record','condition','split')
        for target,row in enumerate(meta):
            if any(any(meta[int(donor)][k]!=row[k] for k in strata) for donor in donors[target]):
                raise ValueError('relationship donor crosses the frozen pairing stratum')
    elif donors is not None: raise ValueError('donors require an explicit relationship perturbation')
    means=h.mean(axis=2)
    target=np.stack((.35*means[...,0]-.65*means[...,1],means.sum(axis=-1)),axis=-1)
    region_results={}
    for index,region in enumerate(regions):
        take=layout[:,index];region_rows=[rows[i] for i in take]
        current=z[take,1:9]
        if context=='own_current': x=current
        elif context=='own_history':
            previous=layout[donors[:,index],index] if perturbation=='wrong_previous' else take
            x=np.concatenate((z[previous,:8],current),axis=-1)
        else:
            sources=[layout[donors[:,index],r] if perturbation=='wrong_other_ROIs' and r!=index
                else layout[:,r] for r in range(len(regions))]
            x=np.concatenate([part for source in sources
                for part in (z[source,:8],z[source,1:9])],axis=-1)
        y=target[take,2:10]-target[take,1:9]
        scores=relation_patch_readout(region_rows,x,y,cfg,['X','T'])
        train=np.array([r['split']=='train' for r in region_rows])
        test=np.array([r['split']=='test' for r in region_rows])
        sd=np.maximum(y[train].reshape(-1,2).std(axis=0),1e-8)
        zero_errors=np.square(y[test]/sd).mean(axis=1)
        test_rows=[r for r,m in zip(region_rows,test) if m]
        persistence={}
        for coordinate,errors in (('X',zero_errors[:,0]),('T',zero_errors[:,1]),
                ('coordinate_mean',zero_errors.mean(axis=1))):
            persistence.update({(r['dataset'],r['subject'],coordinate):r
                for r in relation_subject_mse(test_rows,errors)})
        region_results[region]={}
        for score in scores:
            key=(score['dataset'],score['subject'],score['coordinate'])
            region_results[region][key]=dict(score,persistence_MSE=persistence[key]['MSE'],
                persistence_NRMSE=persistence[key]['NRMSE'])
    expected=set(region_results[regions[0]])
    if any(set(scores)!=expected for scores in region_results.values()):
        raise ValueError('target ROI score populations disagree')
    result=[]
    for key in sorted(expected):
        scores=[region_results[region][key] for region in regions]
        mse=float(np.mean([score['MSE'] for score in scores]))
        persistence_mse=float(np.mean([score['persistence_MSE'] for score in scores]))
        result.append(dict(dataset=key[0],subject=key[1],coordinate=key[2],context=context,MSE=mse,
            NRMSE=float(np.sqrt(mse)),persistence_MSE=persistence_mse,persistence_NRMSE=float(np.sqrt(persistence_mse)),
            mean_baseline_NRMSE=float(np.sqrt(np.mean([score['mean_baseline_NRMSE']**2 for score in scores]))),
            selected_alpha_by_region={region:region_results[region][key]['ridge_alpha'] for region in regions},
            target_training_SD_by_region={region:region_results[region][key]['target_training_SD'] for region in regions},
            region_scores={region:region_results[region][key] for region in regions},
            records=scores[0]['records'],trials=scores[0]['trials'],patches_per_regional_window=8))
    return result


def shape_persistence_scores(scores):
    """Keep every zero-change baseline level consistent with its top score."""
    result=[]
    for score in scores:
        regions={region:dict(row,MSE=row['persistence_MSE'],NRMSE=row['persistence_NRMSE'],
            ridge_alpha=None,validation_MSE=None) for region,row in score.get('region_scores',{}).items()}
        result.append(dict(score,MSE=score['persistence_MSE'],NRMSE=score['persistence_NRMSE'],
            selected_alpha_by_region=None,region_scores=regions))
    return result


def shape_relationship_probes(rows,feature_sets,hb,cfg):
    """Frozen candidate relationship specificity on one paired trial support.

    Wrong-previous keeps the real current target-ROI token. Wrong-other-ROIs
    keeps the target ROI history and uses one common other-trial trajectory
    for both other ROIs, preserving their internal relationship.
    """
    regions=cfg['data']['regions'];meta,layout=relation_trial_layout(rows,regions)
    eligible,donors,_=relation_null_donors(meta,regions,cfg['seed'])
    eligible_set=set(eligible.tolist())
    excluded=Counter((r['dataset'],r['split']) for i,r in enumerate(meta) if i not in eligible_set)
    kept=[meta[i] for i in eligible]
    support=dict(planned_trials=len(meta),eligible_trials=len(eligible),excluded_trials=len(meta)-len(eligible),
        excluded_by_dataset_split=[dict(dataset=k[0],split=k[1],trials=v) for k,v in sorted(excluded.items())],
        eligible_trial_ids=[r['trial_id'] for r in kept],
        donor_trial_ids=[[kept[int(donor)]['trial_id'] for donor in group] for group in donors],
        regions=list(regions),donor_seed=cfg['seed'],
        eligibility='common_complete_three_ROI_trials_with_at_least_two_native_trials_in_each_pairing_stratum',
        stratum=['dataset','subject','record','condition','split'],
        interpretation='descriptive_relationship_perturbation_not_randomization_or_causal_test')
    if not len(eligible): return [],[dict(reason='no_relationship_eligible_trials')],support
    take=layout[eligible].ravel();selected=[rows[i] for i in take]
    target=np.asarray(hb)[take];scores=[];failures=[]
    for representation,values in feature_sets.items():
        features=np.asarray(values)[take]
        for context,wrong in (('own_history','wrong_previous'),('spatial_history','wrong_other_ROIs')):
            for variant in ('real',wrong):
                arguments={} if variant=='real' else dict(donors=donors,perturbation=wrong)
                try:
                    result=shape_forecast_readout(selected,features,target,cfg,context,**arguments)
                    scores.extend(dict(row,representation=representation,variant=variant) for row in result)
                except ValueError as exc:
                    failures.append(dict(representation=representation,context=context,variant=variant,reason=str(exc)))
    return scores,failures,support
