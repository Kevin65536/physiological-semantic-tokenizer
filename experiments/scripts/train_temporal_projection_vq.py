"""Short matched final-C VQ screen on the frozen temporal-projection target."""
import argparse
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import yaml

from experiments.scripts.evaluate_temporal_projection_so import CFG as TARGET, read
from experiments.scripts.run_dual_component_tokenizer import _base_training_model, _state_refinement_loss, owner_lock
from src.analysis.dual_component_tokenizer import eeg_spectral_attributes
from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_batch
from src.data.so_supervision_cache import write_json
from src.data.temporal_projection_so import complete_blocks, project_blocks

CFG=yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/temporal_projection_vq_screen_v1.yaml').read_text())
RUN=ROOT/CFG['parent']; OUT=ROOT/CFG['output']; T=CFG['training']
BASE=yaml.safe_load((ROOT/TARGET['training_reader']).read_text())


def pack(x,s,mask,origin,bins):
    """Input and outputs are per-electrode G0-scaled voltage, 30 s on the source clock."""
    assert x.shape==s.shape==mask.shape and x.shape[0]==6000 and np.isfinite(x).all()
    blocks,sl=complete_blocks(np.where(mask,s,0.),origin)
    xb,_=complete_blocks(x,origin)
    good=np.zeros_like(mask)
    good[sl]=np.broadcast_to(mask[sl].reshape(blocks.shape).all(1)[:,None,:],blocks.shape).reshape(-1,x.shape[1])
    valid=good.T.reshape(-1,15,400).all(-1)
    values={'x':x.T.astype('f4'), 'v3':np.where(mask,s,0.).T.astype('f4'), 'mask':valid}
    for key,b in [('projected_v3',blocks),('projected_x',xb)]:
        v=np.zeros_like(x,dtype='f4');v[sl]=project_blocks(b,bins);values[key]=v.T
    return values


def prepare():
    if not read(RUN/'summary.json')['advance_to_training']: raise ValueError('Target gate is closed')
    frozen=read(RUN/'frozen.json');bins=frozen['order'][:frozen['pairs']]
    m,scales,_,_=base_tokenizer_training_data(BASE,ROOT,recorded_root=TARGET['recorded_root'])
    rows=m['rows'];seen=Counter(); arrays=[]; meta=[]
    def append(x,s,mask,origin,records):
        a=pack(x,s,mask,origin,bins)
        if not a['mask'].any(1).all():raise ValueError('Empty training or evaluation case')
        arrays.append(a);meta.extend(records)
    for i,r in sorted(enumerate(rows),key=lambda z:z[1]['parent']):
        key=r['dataset'],r['subject']
        if r['split']!='train' or r['modality']!='eeg' or seen[key]>=2: continue
        choices=[g for g in r['inputs']['score'] if g['positions']]
        if not choices:continue
        g=min(choices,key=lambda g:g['start_s']);pp=g['positions']; start=g['start_s']
        raw=base_tokenizer_batch(rows,[(i,start,p) for p in pp],scales,array_cache=OrderedDict())
        origin=round(r['native_start_sample']*200/r['native_rate_hz'])+200*start
        records=[dict(dataset=key[0],subject=key[1],parent=r['parent'],start=start,role='train',site=r['positions'][j],
              scale=scales['positions']['eeg'][key[0]][r['positions'][j]]['scale']) for j in pp]
        append(*(raw[k].reshape(len(pp),6000).T for k in ('x','s','mask')),origin,records)
        seen[key]+=1
    if len(seen)!=60 or set(seen.values())!={2}:raise ValueError('Training panel drift')
    for c in read(RUN/'plan.json')['cases']:
        if (c['dataset'],c['subject']) in seen: raise ValueError('Split collision')
        names=[c['channels'][idx] for idx in c['valid']]
        scale=np.array([scales['positions']['eeg'][c['dataset']][k]['scale'] for k in names])
        with np.load(ROOT/TARGET['parent']/'targets'/str(c['i'])/'baseline.npz') as z:
            x=z['x']/scale;s=z['v3']/scale
        with np.load(RUN/'targets'/str(c['i'])/'baseline.npz') as f:
            first=int(f['start_in_window']); last=int(f['stop_in_window'])
            for start in (0,30,60,2):
                a=start*200;cut=slice(a,a+6000)
                good=np.broadcast_to((np.arange(a,a+6000)<17980)[:,None],x[cut].shape)
                records=[dict(dataset=c['dataset'],subject=c['subject'],parent=c['key'],start=start,
                              role='shift' if start==2 else 'validation',site=k,scale=v) for k,v in zip(names,scale)]
                append(x[cut],s[cut],good,c['source']['start_sample']+a,records)
                for arm in ['projected_v3','projected_x']:
                    batch=arrays[-1];common=np.repeat(batch['mask'],400,axis=1)
                    use=slice(max(first,a),min(last,a+6000))
                    local=slice(use.start-a,use.stop-a);published=slice(use.start-first,use.stop-first)
                    np.testing.assert_allclose(batch[arm][:,local][common[:,local]],(f[arm][published]/scale).T[common[:,local]],
                                               atol=2e-5,rtol=1e-5)
    for key in arrays[0]:
        data=np.concatenate([a[key] for a in arrays]);np.save(OUT/(key+'.npy'),data)
    pd.DataFrame(meta).to_csv(OUT/'examples.csv',index=False)
    write_json(OUT/'prepared.json',dict(config=CFG,subjects=len(seen),train_crops=sum(seen.values()),examples=len(meta),
               bins=bins,counts=dict(Counter(r['role'] for r in meta))))


def model(hard,device):
    cfg=dict(BASE,model=dict(BASE['model'],geometry_each_forward=False),
             training=dict(BASE['training'],batch_size=T['batch']),
             capacity=dict(sizes=[8192],uniform_count_fraction=.1),
             architectures=dict(waveform=dict(BASE['architectures']['waveform'],width=256,latent_dim=64)))
    task=dict(modality='eeg',architecture='waveform',candidate='vq8192' if hard else 'warmup')
    return _base_training_model(cfg,task,single_id_spectrum=hard).to(device)


class Batches:
    def __init__(self,arm,device):
        self.meta=pd.read_csv(OUT/'examples.csv',dtype={'subject':str})
        self.device=device
        # Keep the prepared panel in host memory; gather only the selected rows onto the GPU.
        x=np.load(OUT/'x.npy',mmap_mode='r');s=np.load(OUT/(arm+'.npy'),mmap_mode='r')
        mask=np.load(OUT/'mask.npy',mmap_mode='r')
        self.x=x;self.s=s;self.mask=mask; self.scale=self.meta.scale.to_numpy(dtype='f4')
        self.stats={k:np.empty(len(x),dtype='f4') for k in ('vx','vs','vo','dvx','dvs','dvo')}
        self.features=np.empty((len(x),2,4),dtype='f4')
        tr=self.meta[self.meta.role=='train']
        self.groups=[[g.index.to_numpy() for _,g in ds.groupby('subject')] for _,ds in tr.groupby('dataset')]
        for key,group in self.meta.groupby(['dataset','site']):
            to=group.index.to_numpy();idx=group[group.role=='train'].index.to_numpy()
            if not len(idx):raise ValueError(f'No training scale for {key}')
            good=np.repeat(mask[idx],400,axis=1)
            for name,data in [('x',x[idx]),('s',s[idx]),('o',x[idx]-s[idx])]:
                for prefix,vals in [('',data[good]),('d',np.diff(data,axis=1)[good[:,1:]&good[:,:-1]])]:
                    v=np.var(vals,dtype=np.float64)
                    if not v>0: raise ValueError(f'Zero training variance {key}')
                    self.stats[prefix+'v'+name][to]=v
        for ds,group in self.meta.groupby('dataset'):
            idx=group[group.role=='train'].index.to_numpy()
            features=[]
            for chunk in np.array_split(idx,max(1,len(idx)//64)):
                wave=torch.from_numpy(s[chunk].reshape(-1,15,400)).double()
                f=eeg_spectral_attributes(wave,torch.from_numpy(self.scale[chunk]).double())
                features.append(f.numpy()[mask[chunk]])
            f=np.concatenate(features);self.features[group.index]=np.stack((f.mean(0),f.std(0)))
        if not np.all(self.features[:,1]>0):raise ValueError('Degenerate training spectral coordinates')

    def draw(self,rng,size):
        ids=[]
        for _ in range(size):
            people=self.groups[rng.integers(len(self.groups))]
            choices=people[rng.integers(len(people))]
            ids.append(choices[rng.integers(len(choices))])
        return np.array(ids)

    def batch(self,idx):
        def tensor(x):return torch.as_tensor(x,device=self.device)
        x=tensor(self.x[idx].reshape(-1,15,400));s=tensor(self.s[idx].reshape(-1,15,400))
        b=dict(x=x,s=s,o=x-s,mask=tensor(self.mask[idx])[:,:,None].expand_as(x),
               input_scale=tensor(self.scale[idx]),feature_mean=tensor(self.features[idx,0,None,:]),
               feature_sd=tensor(self.features[idx,1,None,:]))
        b.update({k:tensor(v[idx]).reshape((-1,1) if k.startswith('d') else (-1,1,1)) for k,v in self.stats.items()})
        return b


def evaluate(net,data,arm,folder):
    net.eval();meta=data.meta; scores=[]; ids=[]; predictions=[]
    indices=meta[meta.role!='train'].index.to_numpy()
    with torch.no_grad():
        for chunk in np.array_split(indices,max(1,int(np.ceil(len(indices)/32)))):
            b=data.batch(chunk);res=net(b['x']);dec=net.decode_ids(res['ids'])
            wave={}
            for name,out in [('s','S'),('o','O'),('x','sum')]:
                torch.testing.assert_close(dec[out],res[out],atol=0,rtol=0)
                mask=b['mask']
                mse=((dec[out]-b[name])**2*mask).sum((1,2))/mask.sum((1,2))
                wave[name+'_nmse']=(mse/b['v'+name].flatten()).cpu().numpy()
                wave[name+'_native_mse']=mse.cpu().numpy()*data.scale[chunk]**2
            scores.append(pd.DataFrame(wave,index=chunk))
            ids.append(res['ids']['S'].cpu().numpy())
            sample=np.flatnonzero((meta.loc[chunk,'role']=='validation') & (meta.loc[chunk,'start']==30)
                                  & meta.loc[chunk,'site'].str.upper().isin(['FP1','AFP1']))
            for j in sample:
                predictions.append((int(chunk[j]),[dec[k][j].cpu().numpy() for k in ('sum','S','O')]))
    df=meta.loc[indices].join(pd.concat(scores));df.index.name='example'
    df.to_csv(folder/'scores.csv')
    both=np.concatenate(ids); array=pd.Series(list(both),index=indices)
    shifts=[]
    for key,g in df.groupby(['dataset','subject','site']):
        i0=int(g[g.start==0].index[0]);i2=int(g[g.start==2].index[0])
        ok=data.mask[i0,1:]&data.mask[i2,:-1]
        shifts.append(dict(dataset=key[0],subject=key[1],site=key[2],matches=int((array[i0][1:][ok]==array[i2][:-1][ok]).sum()),pairs=int(ok.sum())))
    pd.DataFrame(shifts).to_csv(folder/'context_ids.csv',index=False)
    np.savez_compressed(folder/'waveforms.npz',examples=[i for i,_ in predictions],predictions=[v for _,v in predictions])
    write_json(folder/'summary.json',dict(status='completed',arm=arm,steps=T['warm']+T['hard'],
               validation_crops=int(sum(df.role=='validation')),shifted_crops=int(sum(df.role=='shift')),integer_replay='exact'))


def train(arm):
    torch.set_num_threads(1);torch.manual_seed(T['seed'])
    device='cuda:0'
    torch.cuda.set_per_process_memory_fraction(min(.6,11*2**30/torch.cuda.get_device_properties(0).total_memory),0)
    folder=OUT/arm;folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists(): return
    data=Batches(arm,device); ck=folder/'resume.pt'
    state=torch.load(ck,map_location=device,weights_only=False) if ck.exists() else None
    step=state['step'] if state else 0
    net=model(step>T['warm'],device)
    opt=torch.optim.AdamW(net.parameters(),lr=T['learning_rate'],weight_decay=T['weight_decay'])
    rng=np.random.default_rng(T['seed'])
    if state:
        net.load_state_dict(state['net']);opt.load_state_dict(state['opt']);rng.bit_generator.state=state['rng']
        torch.cuda.set_rng_state(state['cuda'].cpu());torch.set_rng_state(state['cpu'].cpu())
    started=time.monotonic()
    while step<T['warm']+T['hard']:
        if step==T['warm']:
            hard=model(True,device)
            hard.load_state_dict({k:v for k,v in net.state_dict().items() if '.quantizer.' not in k},strict=False)
            with torch.no_grad():
                ids=np.random.default_rng(T['seed']+1000).choice(data.meta.index[data.meta.role=='train'],576,replace=False)
                lat=[hard.encode_continuous(data.batch(ix)['x']) for ix in np.array_split(ids,18)]
                for key in ('S','O'):
                    q=hard.branches[key]['quantizer']
                    q._kmeans_initialize(torch.cat([l[key] for l in lat]).reshape(-1,64))
            net=hard; opt=torch.optim.AdamW(net.parameters(),lr=T['learning_rate'],weight_decay=T['weight_decay'])
        b=data.batch(data.draw(rng,T['batch']))
        spec=BASE['architectures']['waveform']
        if step>=T['warm']:spec=dict(spec,spectral_objective=dict(location='single_id_spectrum',weight=T['spectral_weight']))
        loss,_=_state_refinement_loss(net(b['x']),b,spec)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),T['clip'],error_if_nonfinite=True);opt.step()
        step+=1
        if step%25==0:
            write_json(folder/'progress.json',dict(step=step,loss=float(loss.detach()),seconds=time.monotonic()-started))
        if step%100==0:
            torch.save(dict(step=step,net=net.state_dict(),opt=opt.state_dict(),rng=rng.bit_generator.state,
                            cuda=torch.cuda.get_rng_state(),cpu=torch.get_rng_state()),ck.with_suffix('.tmp'))
            ck.with_suffix('.tmp').replace(ck)
    evaluate(net,data,arm,folder)


def controller():
    OUT.mkdir(parents=True,exist_ok=True)
    with owner_lock(OUT):
        if (OUT/'summary.json').exists(): return
        if not (OUT/'prepared.json').exists():prepare()
        if read(OUT/'prepared.json')['config']!=CFG:raise ValueError('Changed training contract')
        query=subprocess.check_output(['nvidia-smi','--query-gpu=pci.bus_id,uuid','--format=csv,noheader'],text=True)
        devices=[u for bus,u in [s.split(', ') for s in query.splitlines()] if bus in ('00000000:4B:00.0','00000000:CA:00.0')]
        if len(devices)!=2:raise ValueError('Admitted GPU identities changed')
        def sequence(arms,gpu):
            for arm in arms:
                with (OUT/(arm+'.log')).open('a') as log:
                    subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),arm],
                                   env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=log,check=True)
        with ThreadPoolExecutor(2) as pool:
            jobs=[pool.submit(sequence,['v3','projected_x'],devices[0]),pool.submit(sequence,['projected_v3'],devices[1])]
            for job in jobs:job.result()
        tables=[]
        for a in CFG['arms']:
            d=pd.read_csv(OUT/a/'scores.csv')
            subjects=d[d.role=='validation'].groupby(['dataset','subject']).mean(numeric_only=True).reset_index()
            subjects['arm']=a
            for k in 'xso':subjects[k+'_native_rmse']=np.sqrt(subjects[k+'_native_mse'])
            tables.append(subjects)
        subjects=pd.concat(tables);subjects.to_csv(OUT/'subject_scores.csv',index=False)
        macro=subjects.groupby(['arm','dataset'])[['x_nmse','s_nmse','o_nmse']].mean().groupby('arm').mean()
        macro.to_csv(OUT/'macro.csv')
        improve=macro.loc['projected_v3']-macro.loc['v3']
        write_json(OUT/'summary.json',dict(status='completed',seed=T['seed'],steps=T['warm']+T['hard'],
                    macro=macro.to_dict('index'),joint_gain=bool(improve.x_nmse<0 and max(improve.s_nmse,improve.o_nmse)<=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['controller',*CFG['arms']])
    stage=parser.parse_args().stage
    if stage=='controller':
        torch.set_num_threads(1);controller()
    else:train(stage)
