"""Frozen real-X attribute readout conditional on the known projection, training-only fit."""
from pathlib import Path
import sys
import os
import subprocess

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from experiments.scripts.train_temporal_projection_vq import CFG, OUT, model, read
from src.data.so_supervision_cache import write_json
from src.analysis.dual_component_tokenizer import eeg_spectral_attributes


def audit():
    assert read(OUT/'summary.json')['status']=='completed'
    meta=pd.read_csv(OUT/'examples.csv',dtype={'subject':str})
    ex,patch=np.where(np.load(OUT/'mask.npy')&(meta.role.to_numpy()!='shift')[:,None])
    rows=meta.iloc[ex].reset_index(drop=True)
    train=rows.role.to_numpy()=='train'
    x=np.load(OUT/'x.npy',mmap_mode='r')
    px=np.load(OUT/'projected_x.npy',mmap_mode='r')
    def attr(a):
        out=[]
        for ix in np.array_split(np.arange(len(ex)),100):
            wave=torch.from_numpy(a[ex[ix]].reshape(-1,15,400)[np.arange(len(ix)),patch[ix]][:,None,:])
            out.append(eeg_spectral_attributes(wave,torch.ones(len(ix)))[:,0,:2].numpy())
        return np.concatenate(out).astype(float)
    target=attr(x); proj=attr(px)
    raw=x.reshape(-1,15,400)
    rms=np.concatenate([np.log(np.maximum(np.mean(raw[ex[ix],patch[ix]]**2,axis=1),1e-12)) for ix in np.array_split(np.arange(len(ex)),100)])
    design=np.column_stack([np.ones(len(ex)),proj,rms])
    fitted=np.empty_like(target)
    for _,g in rows.groupby(['dataset','site']):
        use=g.index[g.role=='train']
        coeff=np.linalg.lstsq(design[use],target[use],rcond=1e-10)[0]
        fitted[g.index]=design[g.index]@coeff
    residual=target-fitted
    metrics=[];members=[]
    device='cuda:0'
    for arm in CFG['arms']:
        net=model(True,device).eval()
        st=torch.load(OUT/arm/'resume.pt',map_location='cpu',weights_only=False)
        assert st['step']==500
        net.load_state_dict(st['net'])
        ids=np.zeros((len(meta),15),dtype=np.int64)
        with torch.no_grad():
            for idx in np.array_split(np.flatnonzero(meta.role!='shift'),100):
                wave=torch.as_tensor(x[idx].reshape(-1,15,400),device=device)
                ids[idx]=net(wave)['ids']['S'].cpu().numpy()
        token=ids[ex,patch];nt=np.bincount(token[train],minlength=8192)
        person=rows.dataset.astype(str)+'/'+rows.subject
        people=pd.DataFrame(dict(id=token[train],p=person[train])).groupby('id').p.nunique().reindex(range(8192),fill_value=0).to_numpy()
        allowed=(nt>=20)&(people>=3)
        correction=np.zeros((8192,2))
        for f in range(2):
            correction[:,f]=np.bincount(token[train],weights=residual[train,f],minlength=8192)/(nt+20)
        correction[~allowed]=0
        selected=int(np.argmax(np.where(allowed,nt,-1))) if allowed.any() else -1
        for (ds,sub),g in rows[~train].groupby(['dataset','subject']):
            ix=g.index
            before=np.mean(residual[ix]**2,axis=0)
            after=np.mean((residual[ix]-correction[token[ix]])**2,axis=0)
            metrics.append(dict(arm=arm,dataset=ds,subject=sub,sites_patches=len(ix),support=allowed[token[ix]].mean(),
                                theta_gain=1-after[0]/before[0],alpha_gain=1-after[1]/before[1]))
            hit=ix[token[ix]==selected]
            if len(hit):
                j=hit[0]
                members.append(dict(arm=arm,token=selected,train_members=int(nt[selected]),train_people=int(people[selected]),
                                    dataset=ds,subject=sub,example=ex[j],patch=patch[j],site=rows.site[j],matches=len(hit)))
    pd.DataFrame(metrics).to_csv(OUT/'word_attribute_gain.csv',index=False)
    pd.DataFrame(members).to_csv(OUT/'word_members.csv',index=False)
    write_json(OUT/'word_audit.json',dict(status='completed',attributes=['real X log theta','real X log alpha'],
        background='OLS by dataset/electrode on intercept, log projected-X theta and alpha power and log X mean square; shared background frozen on training',
        lookup='training residual mean per S ID times n/(n+20), zero unless >=20 training patches and >=3 training people',
        member_selection='most frequent supported training ID per arm, first retained validation occurrence per person, no validation substitution',
        train_patches=int(train.sum()),validation_patches=int((~train).sum())))


if __name__=='__main__':
    bus='00000000:4B:00.0'
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=pci.bus_id,uuid','--format=csv,noheader'],text=True)
    gpu=next(u for b,u in [s.split(', ') for s in lines.splitlines()] if b==bus)
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    torch.set_num_threads(1);audit()
