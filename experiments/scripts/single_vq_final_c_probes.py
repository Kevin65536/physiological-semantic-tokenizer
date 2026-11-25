"""Readouts and same-center context replacement for frozen final C."""
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import torch

import run_single_vq_final_c_audit as audit
import score_single_vq_readout as scoring
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.analysis.tokenizer_pattern_audit import feature_means
from src.utils.io import save_npz_atomic

from run_dual_component_tokenizer import write_json


def readout_job(job):
    seed,b = job
    path = audit.OUT/f'readout_{seed}_{b}'
    path.mkdir(exist_ok=True)
    if (path/'scores.csv').exists():
        return
    book = audit.network('eeg',8192,seed).branches[b]['quantizer'].codebook.detach().numpy()
    rows,_ = audit.relations.maps()
    fields = audit.readout.attributes(np.zeros((1,400)), 'eeg')[1]

    def load(task):
        with np.load(audit.inputs(task)) as z, np.load(audit.readout.destination(task)) as parent:
            ex,valid = z['examples'],z['valid']
            for name in ('examples','valid','bg'):
                np.testing.assert_array_equal(z[name],parent[name])
            ids = z[f'{b}_{seed}_id'].astype(np.int64)
            people = [audit.readout.PEOPLE[rows[r]['dataset'],rows[r]['subject']] for r in ex[:,0]]
            part = dict(bg=z['bg'].astype(np.int64),id=ids,embedding=book[ids],
                        person=np.repeat(people,valid.sum(1)),y=parent[b+'_target'])
            if task[2]=='validation':
                part['decoded']=z[f'{b}_{seed}_wave']
        return part,ex,valid

    parts = defaultdict(list)
    for t in audit.tasks('train'):
        for name,v in load(t)[0].items():
            parts[name].append(v)
    train = {k:np.concatenate(v) for k,v in parts.items()}
    del parts
    ds_of = {p:ds for (ds,subject),p in audit.readout.PEOPLE.items()}
    raw,bal,ridge,coeff,params,selection = scoring.fit(train,8192,['embedding'],fields,ds_of)
    write_json(path/'selection.json',dict(selected=selection['selected'],cv_loss=selection['cv_loss'],
               folds=[dict(dataset=ds_of[p],person=int(p),fold=v) for p,v in selection['folds'].items()]))
    errors=[]; examples=[]; unseen=0; patches=0
    for t in audit.tasks('validation'):
        a,ex,valid=load(t)
        pooled=raw.read(a['bg'],a['id'])
        balanced=bal.read(a['bg'],a['id'])
        methods=dict(background=pooled[0],shrunk=raw.shrink(*pooled,params['shrunk']),
                     balanced_background=balanced[0],balanced_shrunk=bal.shrink(*balanced,params['balanced_shrunk']),
                     embedding=ridge['embedding'].predict(a['bg'],a['embedding'],coeff['embedding']),decoded=a['decoded'])
        err=np.stack([feature_means((v-a['y'])**2,fields) for v in methods.values()],axis=1)
        if not np.isfinite(err).all():
            raise ValueError('invalid complete patch readout')
        grid=np.zeros((*valid.shape,*err.shape[1:]))
        grid[valid]=err
        n=valid.sum(1)[:,None,None]
        errors.append(np.divide(grid.sum(1),n,out=np.full_like(grid[:,0],np.nan),where=n>0))
        examples.append(ex)
        unseen+=int((pooled[2]==0).sum()); patches+=len(err)
    agg=base_training_aggregate(dict(nmse=np.concatenate(errors)),np.concatenate(examples),rows)
    if agg['inputs']!=53476 or agg['subjects']!=16 or len(set(train['person']))!=60:
        raise ValueError('readout denominator changed')
    stat=agg['metrics']['nmse']
    records=[]
    for scope,subject,values in [('macro','',stat['macro']),*[(v['dataset'],v['subject'],v['value']) for v in stat['by_subject']]]:
        scores=dict(zip(methods,np.asarray(values)))
        for m,v in scores.items():
            base=scores['balanced_background' if m.startswith('balanced') else 'background']
            for name,e,ref in zip(fields,v,base):
                records.append(dict(dataset=scope,subject=subject,method=m,attribute=name,mse=e,background_mse=ref,gain=1-e/ref))
    write_json(path/'support.json',dict(patches=patches,unseen_joint=unseen))
    scoring.write_csv(path/'scores.csv',records)


def context_pairs(ex,valid,rows):
    rng=np.random.default_rng(1125)
    groups=defaultdict(list)
    people=defaultdict(list)
    for i,(r,t,p) in enumerate(ex):
        row=rows[r]
        if valid[i].all():
            person=(row['dataset'],row['subject'])
            key=(*person,row['record'],row['positions'][p])
            start=row['native_start_sample']/row['native_rate_hz']+t
            groups[key].append((i,start))
            people[person].append((i,key,start))
    pairs=[]; denominator=[]
    for person,group in sorted(people.items()):
        eligible=[]
        for i,key,start in group:
            donors=[j for j,t in groups[key] if abs(t-start)>=30-1e-8]
            if donors:
                eligible.append((i,rng.choice(donors)))
        order=rng.permutation(len(eligible))[:32]
        pairs.extend(eligible[i] for i in order)
        denominator.append(dict(dataset=person[0],subject=person[1],complete=len(group),eligible=len(eligible),used=len(order)))
    return np.asarray(pairs,dtype=int),denominator


@torch.inference_mode()
def context_audit():
    rows,scales=audit.relations.maps()
    a=audit.collect('eeg',17,'validation')
    ex=a['examples']
    pairs,den=context_pairs(ex,a['valid'],rows)
    pd.DataFrame(den).to_csv(audit.OUT/'context_support.csv',index=False)
    cache=OrderedDict()
    results=[]
    for seed in audit.SEEDS:
        maps={b:pd.read_csv(audit.OUT/f'eeg_{seed}'/f'{b}_words.csv') for b in 'SO'}
        model=audit.network('eeg',8192,seed)
        for i in range(0,len(pairs),32):
            u,v=pairs[i:i+32].T
            def wave(idx):
                return audit.base_tokenizer_batch(rows,ex[idx],scales,array_cache=cache)['x']
            x,donor=wave(u),wave(v)
            swap=donor.copy()
            swap[:,7]=x[:,7]
            np.testing.assert_array_equal(x[:,7],swap[:,7])
            orig,other,changed=[model(torch.from_numpy(w))['ids'] for w in (x,donor,swap)]
            check=model(torch.from_numpy(x.copy()))['ids']
            for b in 'SO':
                torch.testing.assert_close(orig[b],check[b],rtol=0,atol=0)
                ids,don,ids2=[v[b][:,7].numpy() for v in (orig,other,changed)]
                fam=maps[b].family.to_numpy(); attr=maps[b].family_score.to_numpy()
                for j,(word,d,new) in enumerate(zip(ids,don,ids2)):
                    row,t,p=ex[u[j]]; row2,t2,p2=ex[v[j]]
                    results.append(dict(seed=seed,branch=b,dataset=rows[row]['dataset'],subject=rows[row]['subject'],
                        record=rows[row]['record'],site=rows[row]['positions'][p],row=row,start=t,donor_row=row2,donor_start=t2,
                        id=word,swapped=new,donor_id=d,id_same=word==new,family_same=fam[word]==fam[new],
                        family_donor_same=fam[word]==fam[d],absolute_profile_drift=abs(attr[new]-attr[word])))
            if i==0:
                save_npz_atomic(audit.OUT/f'context_example_{seed}.npz',x=x[0],swap=swap[0],example=ex[u[0]],donor=ex[v[0]],
                    **{k+'_'+b:v[b][0].numpy() for k,v in [('original',orig),('changed',changed)] for b in 'SO'})
    pd.DataFrame(results).to_csv(audit.OUT/'context.csv',index=False)
