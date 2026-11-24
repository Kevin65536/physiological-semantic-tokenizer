"""Fit the 14-bit innovation encoder and its integer decoder from filtered G0 training arrays."""
import json
import multiprocessing
import time
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from retrain_joint_qc_tokenizer import CFG, OLD, OUT, configuration, dump
from src.data.dual_component_tokenizer import base_tokenizer_joint_eeg, base_tokenizer_training_data, base_tokenizer_batch
from src.tokenizers.adaptive_band_allocation import band_waves, covariance_from_source, regular_covariance
from src.tokenizers.band_exchange_words import fit_books, paid_coordinates, source_properties
from src.tokenizers.cross_branch_innovations import (
    PILOT_SIZE, encode_second, fit_second_books, load_model, pilot_means, pilot_positions,
)
from src.tokenizers.cross_branch_sampling import calibration, covariance_blocks, encode_role, layouts
from src.tokenizers.eeg_transient_gain import (
    BoundedTransientSource, FEATURES, band_indices, feature_field, prepare_x, weighted_ridge_gain,
)
from src.utils.io import save_npz_atomic
from retrain_joint_qc_tokenizer import ROOT

DIR = OUT/'eeg'
STATE = {}
DATASETS = ['eeg_fnirs_single_trial','refed','simultaneous_eeg_nirs','visual_cognitive_motivation']


def save(path, **arrays):
    save_npz_atomic(path, **arrays)


def load(path):
    with np.load(path,allow_pickle=False) as z:
        return dict(z)


def plan():
    manifest,scales,_,_ = base_tokenizer_training_data(configuration(),ROOT)
    if manifest.get('window_exclusions')!='v3_window_exclusions_v1':
        raise ValueError('QC view required')
    rows=manifest['rows']
    positions=json.loads((OLD/'component_source_transfer/hard_as_v1/local_information_v1/population.json').read_text())['native_positions']
    cases={ds:[] for ds in DATASETS}
    for i,r in enumerate(rows):
        if r['split']=='train' and r['modality']=='eeg':
            cases[r['dataset']].extend(dict(row=i,**v) for v in r['inputs']['score'] if v['positions'])
    for dataset,group in cases.items():
        records=Counter((rows[c['row']]['subject'],rows[c['row']]['record']) for c in group)
        by_person=Counter(s for s,r in records)
        for c in group:
            row=rows[c['row']]
            s,r=row['subject'],row['record']
            c['weight']=1/(len(by_person)*by_person[s]*records[s,r])
        assert np.isclose(sum(c['weight'] for c in group),1)
    STATE.update(rows=rows,scales=scales,names=positions,cases=cases)
    dump(DIR/'population.json',{ds:{'crops':len(g),'positions':sum(len(c['positions']) for c in g),
         'people':sorted({rows[c['row']]['subject'] for c in g})} for ds,g in cases.items()})


def fit_transfer():
    path=DIR/'frequency.npz'
    if path.exists():
        return
    rng=np.random.default_rng(29)
    power=np.zeros(3001); cross={b:np.zeros(3001,complex) for b in 'SO'}
    used=0; cache=OrderedDict()
    for ds in DATASETS:
        group=STATE['cases'][ds]
        picks=rng.choice(len(group),CFG['eeg']['frequency_fit_sample']//4,p=[c['weight'] for c in group])
        triples=[(group[k]['row'],group[k]['start_s'],rng.choice(group[k]['positions'])) for k in picks]
        for i in range(0,len(triples),64):
            batch=base_tokenizer_batch(STATE['rows'],triples[i:i+64],STATE['scales'],array_cache=cache,cache_limit=64)
            keep=batch['mask'].all((1,2))
            if not keep.any():
                continue
            x=np.fft.rfft(batch['x'][keep].reshape(-1,6000))
            power+=np.square(abs(x)).sum(0)
            for b in 'SO':
                y=np.fft.rfft(batch[b.lower()][keep].reshape(-1,6000))
                cross[b]+=(x.conj()*y).sum(0)
            used+=len(x)
    if used==0: raise ValueError('no complete train positions for frequency fit')
    save(path,**{b:c/((1+CFG['eeg']['frequency_ridge_fraction'])*power+1e-12) for b,c in cross.items()})
    dump(DIR/'frequency_fit.json',dict(complete=used,draws=CFG['eeg']['frequency_fit_sample']))


def data(ds,stage):
    group=STATE['cases'][ds]; cache=OrderedDict(); began=time.monotonic()
    for i,c in enumerate(group):
        if i%32==0:
            dump(DIR/ds/'progress.json',dict(stage=stage,crops=i,planned=len(group),seconds=time.monotonic()-began))
        fields,seen,valid=base_tokenizer_joint_eeg(STATE['rows'],c,STATE['scales'],STATE['names'][ds],cache)
        yield c,fields,seen,valid


def source(ds):
    path=DIR/ds/'source.npz'
    if path.exists():return
    names=STATE['names'][ds];n=len(names);H=load(DIR/'frequency.npz')
    total=np.zeros((5,n));mass=np.zeros(n)
    for c,x,seen,_ in data(ds,'source_background'):
        prepared=prepare_x(x['x'],seen,H)
        w=c['weight']*seen
        total+=w*np.log(prepared['window_rms'].clip(1e-8));mass+=w
    bg=np.exp(total/mass.clip(1e-12))
    for band in bg:band[mass==0]=np.median(band[mass>0]) if (mass>0).any() else 1.
    gram=np.zeros((5,n,len(FEATURES),len(FEATURES)));rhs=np.zeros(gram.shape[:-1])
    ref={k.upper():v for k,v in STATE['scales']['positions']['eeg'][ds].items()}
    vs,vo=np.array([[ref[k.upper()]['target_variance'][b][0] for k in names] for b in ['s','o']])
    for c,x,seen,valid in data(ds,'source_transient_fit'):
        p=prepare_x(x['x'],seen,H)
        wanted=(x['s'].T*vo[:,None]+(p['prior']['sum']-x['o'].T)*vs[:,None])/(vs+vo)[:,None]
        bands=band_waves(np.fft.rfft(wanted))
        for j in range(5):
            carrier=p['carriers'][j].reshape(n,300,20)
            w=(1/vs+1/vo)*valid*c['weight']/300
            a=np.square(carrier).mean(-1)*w[:,None]
            b=(carrier*bands[j].reshape(n,300,20)).mean(-1)*w[:,None]
            phi=feature_field(p,bg,j)
            gram[j]+=np.einsum('cti,ctj,ct->cij',phi,phi,a,optimize=True)
            rhs[j]+=np.einsum('cti,ct->ci',phi,b,optimize=True)
    beta=weighted_ridge_gain(gram,rhs,CFG['eeg']['source_ridge_fraction'])
    save(path,background=bg,beta=beta)


def source_model(ds):
    meta=dict(positions={ds:STATE['names'][ds]},transient=True,gain_bounds=[0,3])
    return BoundedTransientSource(load(DIR/'frequency.npz'),{ds:load(DIR/ds/'source.npz')},meta)


class Sample:
    """Bounded weighted sampling without replacement from the eligible 2-second words."""
    def __init__(self,rng):
        self.rng=rng
        self.limit=CFG['eeg']['sample_patches_per_dataset_role_branch']
        self.keys=np.empty(0)
        self.values=None

    def add(self,x,properties,sites,weight):
        if not len(sites):return
        waves=x.reshape(-1,400)
        keys=np.log(self.rng.random(len(waves)).clip(1e-300))/weight
        selected=np.flatnonzero(keys>=self.keys.min()) if len(self.keys)==self.limit else np.arange(len(keys))
        if not len(selected):return
        new={'wave':waves[selected].astype('f4'),'prop':properties.reshape(-1,3)[selected],
             'site':np.repeat(sites,15)[selected]}
        keys=np.r_[self.keys,keys[selected]]
        keep=np.argpartition(keys,-self.limit)[-self.limit:] if len(keys)>self.limit else np.arange(len(keys))
        self.values={k:(np.concatenate((self.values[k],v)) if self.values is not None else v)[keep] for k,v in new.items()}
        self.keys=keys[keep]


def sample_path(ds,stage,b,j):return DIR/ds/f'{stage}_{b}_{j}_training.npz'
def book_path(stage,b,j):return DIR/'books'/f'{stage}_{b}_{j}.npz'


def books(stage):
    return {b:[load(book_path(stage,b,j)) for j in range(5)] for b in 'SO'}


def add_moment(moment,j,indices,target,word,w):
    if len(indices):
        moment[0,j,indices]+=w*np.square(target).mean((1,2))
        moment[1,j,indices]+=w*(target*word).mean((1,2))
        moment[2,j,indices]+=w*np.square(word).mean((1,2))
        moment[3,j,indices]+=w


def fit_arrays(task):
    ds,stage=task;folder=DIR/ds
    dest=folder/(stage+'_state.npz')
    if dest.exists():return
    model=source_model(ds)
    n=len(STATE['names'][ds]);pilot=pilot_positions(n)
    rng=np.random.default_rng(107+DATASETS.index(ds))
    samples={b:[Sample(rng) for j in range(5)] for b in 'SO'}
    mass=np.zeros((2*n,2*n));cov=np.zeros((5,2*n,2*n))
    moments=np.zeros((4,5,2*n));squares=np.zeros((5,2*n));counts=np.zeros_like(squares)
    state=load(folder/'pilot_state.npz') if stage!='pilot' else {}
    if stage in ('residual','calibrate'):state.update(load(folder/'calibration_state.npz'))
    if stage=='calibrate':state.update(load(folder/'residual_state.npz'))
    old=books('pilot') if stage!='pilot' else None
    new=books('residual') if stage=='calibrate' else None
    for c,x,seen,_ in data(ds,stage):
        if not seen.any():continue
        src=model.predict_continuous(x['x'].T.reshape(1,n,15,400),ds,seen[None])
        roles=layouts(seen)['staggered']
        waves={b:band_waves(np.fft.rfft(src[b][0].reshape(n,6000))) for b in 'SO'}
        prop={b:source_properties(src[b][0]) for b in 'SO'}
        if stage=='pilot':
            both_seen=np.tile(seen,2)
            cov+=c['weight']*covariance_from_source(np.fft.rfft(np.concatenate([src[b][0] for b in 'SO']).reshape(2*n,6000)))
            mass+=c['weight']*np.outer(both_seen,both_seen)
        else:
            coords=[];selected={}
            for b in 'SO':
                words,selected[b]=encode_role(waves[b],roles[b],prop[b],old[b])
                coords.append(paid_coordinates(words,old[b],'balanced',True)[0])
            coords=np.concatenate(coords);role=np.r_[roles['S'],roles['O']]
            if stage!='calibration':
                coords[~pilot]=0
                means,_=pilot_means(coords,role,state['joint'],state['pilot_gain'],state['pilot_noise'])
        for bi,b in enumerate('SO'):
            for j in range(5):
                sites=np.flatnonzero(roles[b]==j);ix=sites+bi*n
                if stage=='pilot':
                    samples[b][j].add(waves[b][j,sites],prop[b][sites],sites,c['weight']/seen.sum())
                elif stage=='calibration':
                    add_moment(moments,j,ix,selected[b][sites],coords[ix],c['weight'])
                else:
                    sites=sites[~pilot[ix]];ix=sites+bi*n
                    if not len(ix):continue
                    residual=selected[b][sites]-means[j,ix]
                    if stage=='residual':
                        squares[j,ix]+=np.square(residual).mean((1,2))*c['weight']
                        counts[j,ix]+=c['weight']
                        samples[b][j].add(residual,prop[b][sites],ix,c['weight']/seen.sum())
                    else:
                        scale=state['scale'][j,ix,None,None]
                        book=new[b][j]
                        ids=encode_second((residual/scale).reshape(-1,400),prop[b][sites].reshape(-1,3),book,'innovation',j)
                        index=(ids.astype(int)-PILOT_SIZE)%CFG['eeg']['residual_capacity']
                        pred=book['innovation_atoms'][index].reshape(-1,15,400)*scale
                        add_moment(moments,j,ix,residual,pred,c['weight'])
    if stage in ('pilot','residual'):
        for b in 'SO':
            for j,s in enumerate(samples[b]):
                v=s.values if s.values is not None else dict(wave=np.empty((0,400),'f4'),prop=np.empty((0,3),'f4'),site=np.empty(0,int))
                save(sample_path(ds,stage,b,j),**v)
    if stage=='pilot':
        cov/=np.maximum(mass,1e-20)
        marg=[np.stack([regular_covariance(z) for z in cov[:,s,s]]) for s in [slice(0,n),slice(n,None)]]
        _,joint,_=covariance_blocks(*marg,cov[:,:n,n:])
        save(dest,joint=joint)
    elif stage=='residual':
        fallback=np.sqrt(np.diagonal(state['joint'],axis1=1,axis2=2).clip(1e-16))
        scale=np.sqrt(squares/counts.clip(1e-20))
        save(dest,scale=np.where(counts>0,np.maximum(scale,1e-8*fallback),fallback))
    else:
        gain,noise=calibration(moments)
        name='pilot' if stage=='calibration' else 'innovation_mean'
        save(dest,**{name+'_gain':gain,name+'_noise':noise})


def fit_book(task):
    stage,b,j=task; dest=book_path(stage,b,j)
    if dest.exists():return
    waves=[];props=[]
    for ds in DATASETS:
        v=load(sample_path(ds,stage,b,j));w=v['wave']
        if stage=='residual':
            scale=load(DIR/ds/'residual_state.npz')['scale'][j,v['site']]
            w=(w/scale[:,None]).astype('f4')
        waves.append(w);props.append(v['prop'])
    waves=np.concatenate(waves);props=np.concatenate(props)
    if stage=='pilot':
        result=fit_books(waves,props,j,107+100*(b=='O')+j,CFG['eeg']['pilot_capacity'])
    else:
        energy=np.square(waves.astype(float)).mean(1)
        result=fit_second_books({'innovation':waves},props,energy,np.ones(len(waves)),j,
                                197+100*(b=='O')+j,CFG['eeg']['residual_capacity'])
    save(dest,**result)


def export():
    fields={}
    for stage in ['pilot','residual']:
        for b,group in books(stage).items():
            for j,book in enumerate(group):
                prefix='pilot' if stage=='pilot' else 'second'
                fields.update({f'{prefix}_{b}_{j}_{k}':v for k,v in book.items()})
    for ds in DATASETS:
        source=source_model(ds)
        fields[ds+'_source_metadata']=np.array(json.dumps(source.metadata))
        for key,group in [('transfer',source.transfer),('source_param',source.models[ds])]:
            fields.update({f'{ds}_{key}_{k}':v for k,v in group.items()})
        for stage in ['pilot','calibration','residual','calibrate']:
            fields.update({ds+'_state_'+k:v for k,v in load(DIR/ds/(stage+'_state.npz')).items()})
    dest=DIR/'model.npz'
    save(dest,metadata=np.array(json.dumps(dict(schema='cross_branch_innovation_eeg_v1',positions=STATE['names']))),**fields)
    model=load_model(dest)
    for ds in DATASETS:
        _,x,seen,_=next(data(ds,'integer_replay'))
        n=len(seen); values=x['x'].T.reshape(1,n,15,400)
        ids=model.encode(values,ds,seen[None]); pred=model.decode_ids(ids,ds)
        for b in 'SO':
            assert ids[b].shape==(1,n,15) and ids[b].max()<15351 and np.isfinite(pred[b]).all()
        np.testing.assert_allclose(pred['sum'],pred['S']+pred['O'])
    dump(DIR/'fit_complete.json',dict(status='completed',model=str(dest),integer_reload_datasets=DATASETS))


def run():
    DIR.mkdir(parents=True,exist_ok=True)
    plan()
    fit_transfer()
    ctx=multiprocessing.get_context('fork')
    for label,fn,tasks in [('source',source,DATASETS)]+[
        entry for stage in ['pilot','calibration','residual','calibrate'] for entry in
        ([(stage,fit_arrays,[(ds,stage) for ds in DATASETS])] +
         ([(stage+'_words',fit_book,[(stage,b,j) for b in 'SO' for j in range(5)])] if stage in ('pilot','residual') else []))]:
        dump(DIR/'progress.json',dict(stage=label,status='running',workers=len(tasks)))
        with ProcessPoolExecutor(max_workers=len(tasks),mp_context=ctx) as pool:
            list(pool.map(fn,tasks))
    export()
    dump(DIR/'progress.json',dict(stage='complete',status='completed'))


if __name__=='__main__':
    run()
