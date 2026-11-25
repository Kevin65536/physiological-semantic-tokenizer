"""Export traceable example windows and descriptive sequence counts for the C report."""
import numpy as np
import pandas as pd
import torch

import run_single_vq_final_c_audit as audit
import single_vq_event_coupling as event
from src.analysis.dual_component_tokenizer import base_training_metrics
from src.utils.io import save_npz_atomic
from run_dual_component_tokenizer import read_json, write_json


@torch.inference_mode()
def export(out):
    out.mkdir(exist_ok=True)
    audit.readout.panel()
    rows,scales=audit.relations.maps()
    cases=[]
    for mod in ('eeg','hb'):
        path=audit.readout.capacity.OUT/'review_20261124_v1/tables'/f'wave_{mod}.json'
        for label,c in zip(('median','difficult'),read_json(path),strict=True):
            r,t=c['row'],c['start_s']
            row=rows[r]; pos=row['positions'].index(c['position'])
            ex=np.array([[r,t,pos]])
            if row['split']!='validation' or not any(np.all(ex==v) for v in audit.readout.capacity.PANEL['examples'][mod,'validation',row['dataset']]):
                raise ValueError('unregistered example')
            raw=audit.base_tokenizer_batch(rows,ex,scales)
            if not raw['mask'].all():
                raise ValueError('incomplete locked example')
            model=audit.network(mod,8192,17) if mod=='eeg' else audit.readout.network(mod,1024,17)
            ids=model(torch.from_numpy(raw['x']))['ids']
            if mod=='eeg':
                with np.load(audit.spectral.folder(17,'C')/'evaluation'/f'{label}_waveforms.npz') as z:
                    np.testing.assert_array_equal(z['example'],ex[0])
                    for b in 'SO':
                        np.testing.assert_array_equal(ids[b][0].numpy(),z['ids_'+b])
            rec={k:v.numpy() for k,v in model.decode_ids(ids).items()}
            nmse=base_training_metrics(rec,raw,1 if mod=='eeg' else 2)['nmse'][0]
            name=f'{mod}_{label}'
            save_npz_atomic(out/f'{name}.npz', **{k:raw[k][0] for k in ('x','s','o','mask')},
                            **{k:v[0] for k,v in rec.items()},**{b+'_id':ids[b][0].numpy() for b in 'SO'})
            cases.append(dict(c,name=name,dataset=row['dataset'],mod=mod,nmse=nmse,
                              native_start_s=row['native_start_sample']/row['native_rate_hz']+t))
    write_json(out/'windows.json',cases)
    # Encode actual event-aligned posterior EEG and frontal EEG history separately.
    directory=audit.OUT/'events_17'
    support=pd.read_csv(directory/'event_support.csv').query("split=='validation' and B=='used'")
    pairs=pd.read_csv(directory/'coupling_pairs.csv').query("split=='validation' and donor_event>=0")
    keys=['subject','record','event','condition']
    joined=support.merge(pairs[keys],on=keys,validate='one_to_one').sort_values(keys[:3])
    matched=joined.groupby(['subject','record']).condition.transform('nunique')==2
    chosen=joined[matched].iloc[0]
    records=event.record_tasks(out)
    key=('validation',chosen.subject,chosen.record)
    _,_,record,_=next(v for v in records if v[0]==key)
    eeg=event.eeg_track(event.SUPPORTED[key]); hb=event.measured_hb(record)
    model=audit.network('eeg',8192,17)
    selected=joined[(joined.subject==chosen.subject)&(joined.record==chosen.record)].groupby('condition').head(1)
    for c in selected.itertuples():
        extract={}
        for label,offset,channels in [('event',-6,4),('history',-26,2)]:
            x=event.context(eeg,c.eeg_marker,offset,channels)
            ids=model(torch.from_numpy(x.copy()))['ids']
            rec=model.decode_ids(ids)
            extract.update({label+'_x':x,label+'_sum':rec['sum'].numpy(),
                            **{label+'_'+b:ids[b].numpy() for b in 'SO'}})
        h=round(10*c.hb_marker)
        save_npz_atomic(out/f'trial_{c.condition}.npz',**extract,hb=hb[h-50:h+240])
    selected.to_csv(out/'trial_selection.csv',index=False)
    # Across fully valid windows only; aggregate counts, never bridge a window boundary.
    stats=[]
    for seed in audit.SEEDS:
        a=audit.collect('eeg',seed,'validation')
        complete=a['valid'].all(1)
        for j,b in enumerate('SO'):
            words=pd.read_csv(audit.OUT/f'eeg_{seed}'/f'{b}_words.csv')
            ids=np.zeros(a['valid'].shape,int)
            ids[a['valid']]=a['ids'][:,j]
            ids=ids[complete]
            fam=words.family.to_numpy()[ids]
            counts=np.bincount((4*fam[:,:-1]+fam[:,1:]).ravel(),minlength=16).reshape(4,4)
            for i,row in enumerate(counts):
                for k,n in enumerate(row):
                    stats.append(dict(seed=seed,branch=b,source=i+1,dest=k+1,n=n,prob=n/row.sum(),
                        windows=len(fam),same_id=np.mean(ids[:,1:]==ids[:,:-1]),same_family=np.mean(fam[:,1:]==fam[:,:-1])))
    pd.DataFrame(stats).to_csv(out/'transitions.csv',index=False)
