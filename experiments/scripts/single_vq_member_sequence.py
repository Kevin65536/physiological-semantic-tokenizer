"""Train-selected real members and conditional sequence comparisons."""
from collections import OrderedDict
import numpy as np
import pandas as pd
from src.analysis.frozen_token_readout import Means, ResidualRidge
from src.analysis.single_vq_relations import descriptors, fit_ridge, interval, profile, surrogate_batch
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.data.dual_component_tokenizer import base_tokenizer_batch
from run_dual_component_tokenizer import write_json
import run_single_vq_relations as suite


def hierarchy(ex, losses):
    rows,_ = suite.maps()
    names = list(losses)
    stacked = np.stack(list(losses.values()),axis=1)
    agg = base_training_aggregate({'nmse':stacked},ex,rows)['metrics']['nmse']
    output = []
    for r in agg['by_subject']:
        for name, v in zip(names,r['value']):
            output.append(dict(dataset=r['dataset'],subject=r['subject'],method=name,mse=v))
    return pd.DataFrame(output)


def crop_mean(a,errors):
    valid = a['valid']
    out = np.zeros((*valid.shape,errors.shape[1]))
    out[valid] = errors
    n = valid.sum(1)[:,None]
    return np.divide(out.sum(1),n,out=np.full_like(out[:,0],np.nan),where=n>0)


def complement(tr,te,books,fams,folder):
    base=Means(tr['bg'],tr['y'])
    preds = {'background':base.lookup(te['bg'])[0]}
    def design(data,kind):
        ids=data['ids']
        parts=[books[b][ids[:,j]] for j,b in enumerate('SO') if b in kind]
        if kind=='SO_interaction':
            parts.append(np.eye(16,dtype=np.float32)[fams['S'][ids[:,0]]*4+fams['O'][ids[:,1]]])
        return np.column_stack(parts)
    for name in ('S','O','SO','SO_interaction'):
        ridge=ResidualRidge(tr['bg'],design(tr,name),tr['y'],base)
        coef=ridge.coefficients(.01)
        pred=[]
        for i in range(0,len(te['bg']),8192):
            sl=slice(i,i+8192)
            pred.append(ridge.predict(te['bg'][sl],design({'ids':te['ids'][sl]},name),coef))
        preds[name]=np.concatenate(pred)
    by=[]
    for j,target in enumerate('XSO'):
        losses={name:crop_mean(te,(v-te['y'])**2)[:,j] for name,v in preds.items()}
        by.append(hierarchy(te['examples'],losses).assign(target=target))
    people=pd.concat(by)
    people.to_csv(folder/'conditional_subject.csv',index=False)
    comparisons=[]
    for target,grp in people.groupby('target'):
        pivot=grp.pivot(index=['dataset','subject'],columns='method',values='mse')
        macro=pivot.groupby(level=0).mean().mean()
        for left,right in [('background','S'),('background','O'),('S','SO'),('O','SO'),('SO','SO_interaction')]:
            delta=pivot[left]-pivot[right]
            point,lo,hi=interval(delta,pivot.index.get_level_values(0))
            comparisons.append(dict(target=target,reference=left,candidate=right,reference_mse=macro[left],
                                    candidate_mse=macro[right],gain=1-macro[right]/macro[left],difference=point,
                                    ci_low=lo,ci_high=hi,people=len(pivot),people_improved=int((delta>0).sum())))
    pd.DataFrame(comparisons).to_csv(folder/'conditional.csv',index=False)


def word_stats(tr,te,b,k):
    j='SO'.index(b)
    ids,dev=tr['ids'][:,j],te['ids'][:,j]
    bg=Means(tr['bg'],tr['y'][:,j+1:j+2])
    residual=tr['y'][:,j+1]-bg.lookup(tr['bg'])[0][:,0]
    count=np.bincount(ids,minlength=k)
    offset=np.bincount(ids,weights=residual,minlength=k)/(count+16)
    residual=te['y'][:,j+1]-bg.lookup(te['bg'])[0][:,0]
    frame=pd.DataFrame(dict(id=dev,person=te['person'],attribute=te['y'][:,j+1],x=te['y'][:,0],
                            mse_background=residual**2,mse_word=(residual-offset[dev])**2))
    per=frame.groupby(['id','person']).agg(patches=('x','size'),mean_x=('x','mean'),mean_attribute=('attribute','mean'),
                                           mse_bg=('mse_background','mean'),mse_word=('mse_word','mean')).reset_index()
    per['gain_mse']=per.mse_bg-per.mse_word
    stats=per.groupby('id').agg(dev_patches=('patches','sum'),dev_people=('person','size'),
                      dev_mean=('mean_attribute','mean'),mean_subject_improvement=('gain_mse','mean'),
                      improved_people=('gain_mse',lambda a:(a>0).sum())).reindex(range(k))
    stats[['dev_patches','dev_people','improved_people']]=stats[['dev_patches','dev_people','improved_people']].fillna(0).astype(int)
    stats['train_patches']=count
    stats['train_people']=np.bincount(np.unique(tr['person']*k+ids)%k,minlength=k)
    stats['train_residual']=offset
    stats['eligible']=(stats.train_patches>=20)&(stats.train_people>=3)
    stats['readable']=stats.eligible&(stats.dev_people>=3)&(stats.mean_subject_improvement>0)&(stats.improved_people*2>stats.dev_people)
    return stats,per


def indices(a):
    a['window'],a['patch']=np.nonzero(a['valid'])


def members(a, pick):
    rows,scales=suite.maps()
    ex=a['examples'][a['window'][pick]]
    unique,reverse=np.unique(ex,axis=0,return_inverse=True)
    full={k:[] for k in ('x','s','o')}
    cache=OrderedDict()
    for start in range(0,len(unique),64):
        v=base_tokenizer_batch(rows,unique[start:start+64],scales,array_cache=cache,cache_limit=48)
        for key in full:
            full[key].append(v[key])
    return np.stack([np.concatenate(v)[reverse,a['patch'][pick]] for v in full.values()],axis=1)


def member_atlas(tr,te,b,stats,folder,mod):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng=np.random.default_rng(413)
    j='SO'.index(b)
    eligible=stats[stats.eligible].sort_values('train_patches',kind='stable')
    picks=[(tag,int(eligible.index[round(q*(len(eligible)-1))])) for tag,q in [('low',.1),('middle',.5),('high',.9)]]
    failures=eligible[(eligible.dev_people>=3)&(eligible.mean_subject_improvement<=0)]
    if len(failures):
        picks.append(('heldout_failure',int(failures.mean_subject_improvement.idxmin())))
    fig,axes=plt.subplots(len(picks),4,figsize=(14,2.7*len(picks)),squeeze=False,layout='constrained')
    manifest=[]; scores=[]; saved={}
    for line,(tag,word) in enumerate(picks):
        arrays={}
        for role,a in [('train',tr),('validation',te)]:
            ids=a['ids'][:,j]
            hits=np.flatnonzero(ids==word)
            hits=rng.choice(hits,min(128,len(hits)),replace=False)
            # Search once per group; require same person, site, dataset and visible RMS bin.
            key=a['bg']+100000*a['person']
            pairs=[]
            for cell in np.unique(key[hits]):
                h=hits[key[hits]==cell]
                alternatives=np.flatnonzero((key==cell)&(ids!=word))
                if len(alternatives):
                    pairs.extend(zip(h,rng.choice(alternatives,len(h),replace=True)))
            if not pairs:
                continue
            pair=np.asarray(pairs)
            for name,pick in zip(('member','background'),pair.T):
                arrays[role,name]=members(a,pick)
                saved[f'{tag}_{word}_{role}_{name}']=arrays[role,name]
                for ix,n in enumerate(pick):
                    r,t,p=a['examples'][a['window'][n]]
                    row=suite.maps()[0][r]
                    manifest.append(dict(word=word,group=tag,split=role,kind=name,pair=ix,dataset=row['dataset'],
                                     subject=row['subject'],record=row['record'],parent=row['parent'],crop_s=t,
                                     position=row['positions'][p],patch=a['patch'][n],rms_bin=int(a['bg'][n]%4),
                                     x_attribute=a['y'][n,0],branch_attribute=a['y'][n,j+1]))
        if len(arrays)!=4:
            axes[line,0].set_title(f'{tag}: ID {word} insufficient matches')
            continue
        train=arrays['train','member']; obs=arrays['validation','member']; other=arrays['validation','background']
        for col,(signal,center) in enumerate([(0,False),(0,True),(j+1,False),(j+1,True)]):
            ax=axes[line,col]; ax.set_title(f'{tag} ID {word} · {"X" if signal==0 else b} {"centered" if center else "amplitude"}')
            for c,color in enumerate(['#195489'] if mod=='eeg' else ['#ab3524','#195489']):
                def wave(v):
                    x=v[:,signal].reshape(len(v),-1,1 if mod=='eeg' else 2)[:,:,c]
                    return x-x.mean(1,keepdims=True) if center else x
                z=wave(obs); mu=wave(train).mean(0); bgmean=wave(arrays['train','background']).mean(0)
                t=np.arange(len(mu))/(200 if mod=='eeg' else 10)
                ax.plot(t,z[:3].T,color=color,alpha=.17,lw=.8)
                ax.plot(t,mu,color=color,lw=1.3,ls='--')
                ax.plot(t,z.mean(0),color=color,lw=1.4)
                ax.plot(t,wave(other).mean(0),color=color,lw=1,ls=':')
                scores.append(dict(group=tag,id=word,signal='X' if signal==0 else b,center=center,color=c,
                    train_members=len(train),dev_members=len(obs),mse_template=float(np.mean((z-mu)**2)),
                    mse_matched_background=float(np.mean((z-bgmean)**2))))
            ax.set_xlabel('Patch time (s)'); ax.set_ylabel('Training input SD')
    fig.suptitle(f'{mod.upper()} {b} | real members: thin examples, solid development mean, dashed train template, dotted matched controls'
                 + (' | red HbO, blue HbR' if mod=='hb' else ''))
    fig.savefig(folder/f'{b}_members.png',dpi=180);plt.close(fig)
    np.savez_compressed(folder/f'{b}_member_waves.npz',**saved)
    pd.DataFrame(manifest).to_csv(folder/f'{b}_members.csv',index=False)
    pd.DataFrame(scores).to_csv(folder/f'{b}_wave_template.csv',index=False)


def sequence_rows(a, fam, j, adjacent):
    rows,_=suite.maps()
    grid=np.full(a['valid'].shape,-1,int);grid[a['valid']]=fam[a['ids'][:,j]]
    x=np.full(grid.shape,np.nan);x[a['valid']]=a['y'][:,0]
    complete=np.all(grid>=0,axis=1)
    if not adjacent:
        return grid[complete,:14],x[complete,-1],a['examples'][complete]
    # Join absolute native time and electrode identity; never roll between records.
    keys=[]
    for r,start,p in a['examples']:
        row=rows[r]
        tick=round(row['native_start_sample']*200/row['native_rate_hz'])+int(start)*200
        keys.append((row['dataset'],row['subject'],row['record'],row['positions'][p],tick))
    lookup={key:i for i,key in enumerate(keys)}
    ii=[]; jj=[]
    for key,i in lookup.items():
        n=lookup.get((*key[:-1],key[-1]+6000))
        if n is not None and complete[i] and np.isfinite(x[n,0]):
            ii.append(i);jj.append(n)
    return grid[ii,1:],x[jj,0],a['examples'][jj]


def sequences(tr,te,fams,folder):
    all_rows=[];summary=[]
    for j,b in enumerate('SO'):
        for adjacent in (False,True):
            train,y,_=sequence_rows(tr,fams[b],j,adjacent)
            test,z,ex=sequence_rows(te,fams[b],j,adjacent)
            audit=dict(branch=b,adjacent=adjacent,train_windows=len(y),dev_windows=len(z))
            if len(y)<16 or not len(z):
                summary.append(dict(audit,status='insufficient support'));continue
            preds={};models=[]
            for name,a,v in zip(('bag','bigram_dwell','motif'),descriptors(train),descriptors(test)):
                model=fit_ridge(a,y);models.append(model);preds[name]=(model(v)-z)**2
            changes={}
            for mode,kind in enumerate(('bag','transition','dwell')):
                loss=np.zeros(len(z)); changed=0
                for rep in range(10):
                    sim=surrogate_batch(test,mode,1700+rep)
                    changed+=np.any(test!=sim,axis=1).sum()
                    loss+=(models[-1](descriptors(sim)[-1])-z)**2/10
                preds['motif_'+kind+'_null']=loss
                changes[kind+'_changed_fraction']=changed/(len(z)*10)
            people=hierarchy(ex,preds)
            all_rows.append(people.assign(branch=b,adjacent=adjacent))
            table=people.pivot(index=['dataset','subject'],columns='method',values='mse')
            for ref,arm in [('bag','bigram_dwell'),('bigram_dwell','motif'),*[(v,'motif') for v in preds if v.endswith('_null')]]:
                delta=table[ref]-table[arm]
                avg=table.groupby(level=0).mean().mean()
                point,lo,hi=interval(delta,table.index.get_level_values(0))
                summary.append(dict(audit,**changes,status='scored',reference=ref,candidate=arm,gain=1-avg[arm]/avg[ref],
                               difference=point,ci_low=lo,ci_high=hi,people=len(delta),datasets=len(table.groupby(level=0))))
    pd.concat(all_rows).to_csv(folder/'sequence_subject.csv',index=False)
    pd.DataFrame(summary).to_csv(folder/'sequence.csv',index=False)


def analyze(job):
    mod,seed=job
    folder=suite.OUT/f'{mod}_{seed}';folder.mkdir(exist_ok=True)
    if (folder/'completion.json').exists():
        return
    tr,te=[suite.collect(mod,seed,s) for s in ('train','validation')]
    for a in (tr,te):
        indices(a)
    k=suite.PLAN['models'][mod]['K']
    fams={};books={};coverage=[]
    model=suite.parent.network(mod,k,seed)
    for j,b in enumerate('SO'):
        fams[b],_,means,bounds=profile(tr['ids'][:,j],tr['y'][:,j+1],k)
        stats,per=word_stats(tr,te,b,k)
        stats['family']=fams[b];stats['family_score']=means
        stats.to_csv(folder/f'{b}_words.csv')
        person_names={v:key for key,v in suite.parent.PEOPLE.items()}
        per['dataset']=[person_names[v][0] for v in per.person]
        per['subject']=[person_names[v][1] for v in per.person]
        per.to_csv(folder/f'{b}_word_people.csv.gz',index=False,compression='gzip')
        for ds,group in [('pooled',per),*per.groupby('dataset')]:
            n=int(group.patches.sum())
            good=stats.index[stats.readable]; elig=stats.index[stats.eligible]
            covered=int(group.loc[group.id.isin(good),'patches'].sum())
            coverage.append(dict(dataset=ds,branch=b,valid=n,train_eligible_patches=int(group.loc[group.id.isin(elig),'patches'].sum()),
                            readable_patches=covered,readable_fraction=covered/n,readable_ids=group[group.id.isin(good)].id.nunique()))
        write_json(folder/f'{b}_family.json',dict(thresholds=bounds,mapping=fams[b],attribute='alpha fraction' if mod=='eeg' else 'HbO slope'))
        books[b]=model.branches[b]['quantizer'].codebook.detach().numpy()
        member_atlas(tr,te,b,stats,folder,mod)
    pd.DataFrame(coverage).to_csv(folder/'coverage.csv',index=False)
    complement(tr,te,books,fams,folder)
    sequences(tr,te,fams,folder)
    write_json(folder/'completion.json',dict(status='completed',models=job,train_people=len(set(tr['person'])),
                                           dev_people=len(set(te['person'])),valid_dev_patches=len(te['person'])))


def supplementary(job):
    """Matched family main effects, same-support observed history, and across-seed correspondence."""
    from scipy.stats import spearmanr
    from sklearn.metrics import adjusted_rand_score
    mod,seed=job
    out=suite.OUT/f'{mod}_{seed}'
    tr,te=[suite.collect(mod,seed,s) for s in ('train','validation')]
    size=suite.PLAN['models'][mod]['K']
    model=suite.parent.network(mod,size,seed)
    fams=[pd.read_csv(out/f'{b}_words.csv').family.values for b in 'SO']
    books=[model.branches[b]['quantizer'].codebook.detach().numpy() for b in 'SO']
    a=tr['ids']; f0=fams[0][a[:,0]]; f1=fams[1][a[:,1]]
    pairs=f0*4+f1
    count=np.bincount(pairs,minlength=16)
    people=np.bincount(np.unique(tr['person']*16+pairs)%16,minlength=16)
    ok=(count>=20)&(people>=3)
    means=[np.bincount(f,minlength=4)/len(f) for f in (f0,f1)]
    def design(ids,cross):
        u,v=[np.eye(4,dtype='float32')[fams[j][ids[:,j]]] for j in range(2)]
        vals=[books[0][ids[:,0]],books[1][ids[:,1]],u,v]
        if cross:
            vals.append(((u-means[0])[:,:,None]*(v-means[1])[:,None,:]).reshape(-1,16)[:,ok])
        return np.column_stack(vals)
    base=Means(tr['bg'],tr['y']);loss={}
    for cross in (False,True):
        name='main_plus_interaction' if cross else 'main'
        ridge=ResidualRidge(tr['bg'],design(tr['ids'],cross),tr['y'],base);coef=ridge.coefficients(.01)
        pred=np.concatenate([ridge.predict(te['bg'][i:i+8192],design(te['ids'][i:i+8192],cross),coef) for i in range(0,len(te['ids']),8192)])
        loss[name]=crop_mean(te,(pred-te['y'])**2)
    report=[]
    for j,target in enumerate('XSO'):
        h=hierarchy(te['examples'],{k:v[:,j] for k,v in loss.items()})
        tab=h.pivot(index=['dataset','subject'],columns='method',values='mse')
        effect=tab.main-tab.main_plus_interaction
        avg,lo,hi=interval(effect,tab.index.get_level_values(0))
        m=tab.groupby(level=0).mean().mean()
        report.append(dict(target=target,main_mse=m.main,interaction_mse=m.main_plus_interaction,gain=1-m.main_plus_interaction/m.main,
                          delta_mse=avg,ci_low=lo,ci_high=hi,supported_family_pairs=int(ok.sum())))
    pd.DataFrame(report).to_csv(out/'interaction_vs_family_main.csv',index=False)
    thresholds=np.quantile(tr['y'][:,0],[.25,.5,.75])
    results=[]
    for adjacent in (False,True):
        arrays=[]
        for a in (tr,te):
            observed=np.searchsorted(thresholds,a['y'][:,0],side='right')
            arrays.append(sequence_rows(dict(a,ids=observed[:,None]),np.arange(4),0,adjacent))
        (seq,y,_),(seq2,z,ex)=arrays
        preds={}
        for name,x,v in zip(('bag','bigram_dwell','motif'),descriptors(seq),descriptors(seq2)):
            preds[name]=(fit_ridge(x,y)(v)-z)**2
        h=hierarchy(ex,preds);results.append(h.assign(adjacent=adjacent))
    pd.concat(results).to_csv(out/'observed_sequence_reference.csv',index=False)
    if mod=='hb' and seed==17:
        other=suite.collect(mod,43,'validation')
        for key in ('valid','examples','person'):
            np.testing.assert_array_equal(te[key],other[key])
        output=[]
        for j,b in enumerate('SO'):
            profiles=[pd.read_csv(suite.OUT/f'hb_{s}'/f'{b}_words.csv') for s in (17,43)]
            ids0,ids1=te['ids'][:,j],other['ids'][:,j]
            g0,g1=profiles[0].family.values[ids0],profiles[1].family.values[ids1]
            v0,v1=profiles[0].family_score.values[ids0],profiles[1].family_score.values[ids1]
            for (ds,sub),p in suite.parent.PEOPLE.items():
                hit=te['person']==p
                if hit.any():
                    output.append(dict(branch=b,dataset=ds,subject=sub,patches=int(hit.sum()),
                                       attribute_rank_r=float(spearmanr(v0[hit],v1[hit]).statistic),
                                       family_agreement=np.mean(g0[hit]==g1[hit]),family_ari=adjusted_rand_score(g0[hit],g1[hit])))
        pd.DataFrame(output).to_csv(suite.OUT/'hb_seed_correspondence.csv',index=False)
