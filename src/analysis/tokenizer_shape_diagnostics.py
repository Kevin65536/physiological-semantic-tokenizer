"""Frozen-G0 diagnostics for the D0 held-out subjects. No tokenizer fitting."""
from collections import Counter, defaultdict
from pathlib import Path

import json

import numpy as np

from src.analysis.dual_component_tokenizer import base_training_aggregate


def training_weights(examples, rows):
    """Uniform dataset, subject, record, crop and position, in that order."""
    children = defaultdict(set)
    paths = []
    for i, start, position in examples:
        r = rows[i]
        path = (r['dataset'], r['subject'], r['record'], (r['parent'], start), position)
        for n, child in enumerate(path):
            children[path[:n]].add(child)
        paths.append(path)
    return np.array([1 / np.prod([len(children[p[:n]]) for n in range(len(p))]) for p in paths])


def training_moments(rows, modality):
    """G0 un-floored native means and variances, pooling exactly its training support."""
    records = {}
    for row in rows:
        if row['modality'] != modality or row['split'] != 'train':
            continue
        record = json.loads((Path(row['folder']).parent/'record.json').read_text())
        parent = next(r for r in record['rows'] if r['modality']==modality and r['parent']==row['parent'])
        if parent['positions'] != row['positions'] or parent['split'] != 'train':
            raise ValueError('training moment identity mismatch')
        for j, pos in enumerate(row['positions']):
            for k in ('x','s','o'):
                m = parent['moments'][k]
                value = np.array([m[f][j] for f in ('count','sum','second')])
                if (value[0] > 0).all():
                    key = (row['dataset'],pos,k,row['subject'],row['record'])
                    records[key] = records.get(key,0) + value
    subjects = defaultdict(list)
    for key, value in records.items():
        subjects[key[:-1]].append(value[1:]/value[0])
    positions = defaultdict(list)
    for key, values in subjects.items():
        positions[key[:-1]].append(np.mean(values,axis=0))
    return {key:(v[0],np.maximum(v[1]-v[0]**2,0))
            for key, values in positions.items() for v in [np.mean(values,axis=0)]}


class ShapeReadout:
    """Train global ID templates and position-only controls; score the same finite support."""
    def __init__(self, size, width, channels):
        self.size, self.width, self.channels = size, width, channels
        self.sums = np.zeros((2,2,size,width,channels))
        self.counts = np.zeros_like(self.sums)
        self.sites = {}
        self.samples = {}
        self.rng = np.random.default_rng(17)
        self.frequency = np.zeros((2,size),dtype=np.int64)

    def update(self, batch, ids, sites, weights=None, identities=None):
        mask = batch['mask'].reshape(-1,15,self.width,self.channels)
        counts = mask.sum(2,keepdims=True)
        if weights is None:
            scores = np.empty((len(mask),2,4,self.channels))
            coverage = np.empty((len(mask),2,self.channels))
        for b,key in enumerate(('s','o')):
            code = ids[key.upper()].reshape(-1)
            target = batch[key].reshape(mask.shape)
            mean = np.where(mask,target,0).sum(2,keepdims=True)/counts.clip(1)
            centered = np.where(mask,target-mean,0).reshape(-1,self.width,self.channels)
            valid = mask.reshape(centered.shape)
            if weights is not None:
                for i in np.unique(code):
                    select = code==i
                    weight = np.repeat(weights,15)[select,None,None]
                    for mode,w in enumerate((1.,weight)):
                        self.sums[b,mode,i] += (centered[select]*w).sum(0)
                        self.counts[b,mode,i] += (valid[select]*w).sum(0)
                for n,site in enumerate(sites):
                    key_site = (b,site)
                    if key_site not in self.sites:
                        self.sites[key_site] = np.zeros((2,self.width,self.channels))
                    sl = slice(15*n,15*n+15)
                    self.sites[key_site][0] += centered[sl].sum(0)
                    self.sites[key_site][1] += valid[sl].sum(0)
                continue
            site_groups = np.array([self.sites[b,site] for site in sites])
            site_mean = (site_groups[:,0]/site_groups[:,1].clip(1))[:,None]
            support = mask & (site_groups[:,1,None]>0)
            for mode in range(2):
                support &= self.counts[b,mode][code].reshape(mask.shape)>0
            prototype = [np.divide(self.sums[b,m],self.counts[b,m],out=np.zeros_like(self.sums[b,m]),
                                   where=self.counts[b,m]>0)[code].reshape(mask.shape) for m in range(2)]
            den = support.sum((1,2))
            coverage[:,b] = den/mask.sum((1,2)).clip(1)
            for j,p in enumerate([0.,site_mean,*prototype]):
                error = np.where(support,(centered.reshape(mask.shape)-p)**2,0).sum((1,2))
                scores[:,b,j] = np.divide(error,den,out=np.full_like(error,np.nan),where=den>0)/batch['v'+key][:,0,:self.channels]
            good = valid.all((1,2))
            priority = self.rng.random(len(code))
            for i in np.unique(code[good]):
                indices = np.flatnonzero(good & (code==i))
                self.frequency[b,i] += len(indices)
                choose = indices[np.argsort(priority[indices])[:64]]
                v = target.reshape(centered.shape)[choose]
                info = np.column_stack([identities[choose//15],choose%15])
                prev = self.samples.get((b,i))
                values = (priority[choose],v,info)
                if prev is not None:
                    values = tuple(np.concatenate([a,c]) for a,c in zip(prev,values))
                order = np.argsort(values[0])[:64]
                self.samples[b,i] = tuple(a[order] for a in values)
        if weights is None:
            return scores, coverage


def shape_gallery(readout, folder, rate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(folder)
    arrays = {}
    for (branch,code),(_,waves,ids) in readout.samples.items():
        label = f'{"SO"[branch]}_{code}'
        centered = waves-waves.mean(1,keepdims=True)
        arrays[label+'_waves'] = waves
        arrays[label+'_example_row_start_position_patch'] = ids
        arrays[label+'_shape_q10_q50_q90'] = np.quantile(centered,[.1,.5,.9],axis=0)
        if rate==10:
            t = np.arange(waves.shape[1])/rate
            arrays[label+'_slope_per_second'] = (centered*(t-t.mean())[None,:,None]).sum(1)/np.sum((t-t.mean())**2)
            arrays[label+'_peak_to_peak'] = np.ptp(waves,axis=1)
        else:
            power = np.abs(np.fft.rfft(centered*np.hanning(waves.shape[1])[None,:,None],axis=1))**2
            arrays[label+'_periodogram'] = power/(rate*np.sum(np.hanning(waves.shape[1])**2))
    np.savez_compressed(folder/'validation_same_ID.npz',frequency=readout.frequency,**arrays)
    fig, axes = plt.subplots(2,4,figsize=(14,6),layout='constrained')
    for b in range(2):
        ranked = np.argsort(-readout.frequency[b])[:2]
        for n,code in enumerate(ranked):
            if not readout.frequency[b,code]:continue
            label = f'{"SO"[b]}_{code}'
            w = arrays[label+'_waves']
            for c in range(2):
                ax = axes[b,2*n+c]
                if rate==200 and c==1:
                    y = arrays[label+'_periodogram'][:,:,0]
                    x = np.fft.rfftfreq(w.shape[1],1/rate)
                    ax.set(xlabel='Frequency (Hz)',ylabel='Voltage coordinate²/Hz',xlim=(0,45))
                else:
                    y = w[:,:,c]-w[:,:,c].mean(1,keepdims=True)
                    x = np.arange(w.shape[1])/rate
                    ax.set(xlabel='Time (s)',ylabel=('EEG' if rate==200 else ('HbO','HbR')[c])+' / train X SD, centered')
                lo,mid,hi = np.quantile(y,[.1,.5,.9],axis=0)
                ax.plot(x,y[:3].T,color='.65',lw=.6)
                ax.fill_between(x,lo,hi,alpha=.25)
                ax.plot(x,mid,lw=1.2)
                ax.set_title(f'{label}, {len(y)} held-out patches')
    fig.suptitle('Two most frequent complete IDs per branch | 3 examples, median and 10–90% band')
    fig.savefig(folder/'validation_patterns.png',dpi=200)
    plt.close(fig)


def compare_readouts(scores, coverage, examples, rows):
    values = base_training_aggregate(dict(nmse=scores,coverage=coverage),examples,rows)
    def gains(a):
        v = np.array(a)
        denom = v[:,:2,None,:]
        return (1-np.divide(v[:,None,2:,:],denom,out=np.full((2,2,2,v.shape[-1]),np.nan),where=denom>0)).tolist()
    metric = values['metrics']['nmse']
    return dict(error=metric, coverage=values['metrics']['coverage'],axes=['S/O','zero/site_mean/ID_pooled/ID_weighted','color'],
                gain_axes=['S/O','zero/site_mean','ID_pooled/ID_weighted','color'],gain_macro=gains(metric['macro']),
                gains_by_dataset={k:gains(v) for k,v in metric['by_dataset'].items()},
                gains_by_subject=[dict(dataset=r['dataset'],subject=r['subject'],gain=gains(r['value'])) for r in metric['by_subject']])
