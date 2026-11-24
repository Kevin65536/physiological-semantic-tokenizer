"""Census of numeric failures and reviewable waveform patterns on the V3 G0 score grid."""
import argparse
import fcntl
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import blake2b
import json
import multiprocessing
from pathlib import Path
import resource
import time
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT/'experiments/configs/physiology_semantic_tokenizer/tokenizer_observation_qc_census_v1.yaml'
CFG = yaml.safe_load(CONTRACT.read_text())
DST = ROOT/CFG['output']
Q = CFG['settings']
WORK = []
SCALES = {}


def classify(data, mask, fs, scale):
    """Columns are electrodes or positions of ONE colour; all fractions use real samples."""
    data = np.ascontiguousarray(data.T, dtype=np.float64)
    good = (mask.T & np.isfinite(data))
    n = data.shape[1]
    assert n == 30*fs and good.shape == data.shape
    hits = []
    peers = {}
    hz = np.fft.rfftfreq(n, 1/fs)
    for i, (x, g) in enumerate(zip(data, good)):
        def hit(k, value, peer=-1):
            hits.append((i, k, float(value), peer))
        support = g.mean()
        if support < 1:
            hit('missing_all' if support == 0 else 'missing_part', 1-support)
            continue
        lo, hi = x.min(), x.max()
        if lo == hi:
            hit('constant', 1)
            continue
        # Canonicalize sign and signed zero; verify full equality after matching a digest.
        canonical = x if x[np.flatnonzero(x)[0]] > 0 else -x
        digest = blake2b(np.where(canonical==0,0.,canonical).tobytes(),digest_size=16).digest()
        peers.setdefault(digest, []).append(i)
        repeats = np.diff(x) == 0
        if repeats.any():
            bounds=np.flatnonzero(np.diff(np.r_[False,repeats,False]))
            longest = 1+max(bounds[1::2]-bounds[::2])
            if longest >= fs*Q['flat_patch_seconds']:
                hit('frozen_segment', longest/fs)
        extremes = max(np.mean(x==lo), np.mean(x==hi))
        if extremes >= Q['rail_fraction']:
            hit('rail', extremes)
        if len(np.unique(x[::max(1,n//16)])) <= Q['sparse_levels_max']:
            levels = np.unique(x)
            if len(levels)<=Q['sparse_levels_max']:
                hit('few_levels',len(levels))
        sd_ratio = np.std(x)/scale[i] if np.isfinite(scale[i]) and scale[i]>0 else np.nan
        if sd_ratio < Q['near_flat_sd_over_training_sd']:
            hit('near_flat',sd_ratio)
        e=(x-np.median(x))**2
        integ=np.r_[0.,np.cumsum(e)]
        span=round(fs*Q['brief_seconds'])
        brief=np.max(integ[span:]-integ[:-span])/integ[-1]
        if brief >= Q['brief_energy_fraction']:
            hit('transient',brief)
        if fs != 200:
            continue
        energy=abs(np.fft.rfft(x))**2
        energy[1:-1]*=2
        energy/=energy.sum()
        for center in (50,60):
            frac=energy[(hz>=center-1)&(hz<=center+1)].sum()
            if frac > Q['mains_fraction']:
                hit(f'line{center}',frac)
        peak=np.argmax(energy)
        if hz[peak]>=1 and min(abs(hz[peak]-50),abs(hz[peak]-60))>1 and energy[peak]>=Q['other_tone_bin_fraction']:
            hit('other_tone',energy[peak])
        if sd_ratio>=Q['large_sd_over_training_sd']:
            for label,low,high in [('large_slow',1,4),('large_fast',30,45)]:
                fraction=energy[(hz>=low)&(hz<=high)].sum()
                if fraction>=Q['high_or_slow_fraction']:
                    hit(label,fraction)
    for indices in peers.values():
        if len(indices)>1:
            for i in indices:
                for j in indices:
                    if j!=i and (np.array_equal(data[i],data[j]) or np.array_equal(data[i],-data[j])):
                        hits.append((i,'duplicate',1.,j))
                        break
    return hits


def synthetic():
    rng=np.random.default_rng(7)
    t=np.arange(6000)/200
    x=rng.normal(size=(6000,12));x[:,0]=0;x[:,1]=np.sin(2*np.pi*50*t)
    x[:,2]=-x[:,1];x[300:900,3]=2;x[:,4]=np.sign(x[:,4]);x[2345,5]=1e5
    x[:,6]=np.sin(2*np.pi*20*t);x[:,7]*=1e-6
    x[:,10]=np.sin(2*np.pi*60*t);x[:,11]=100*np.sin(2*np.pi*35*t)
    mask=np.ones_like(x,bool);mask[:,8]=False;mask[0,9]=False
    tags={(i,k) for i,k,*_ in classify(x,mask,200,np.ones(12))}
    assert {(0,'constant'),(1,'line50'),(2,'duplicate'),(3,'frozen_segment'),(4,'few_levels'),
            (4,'rail'),(5,'transient'),(6,'other_tone'),(7,'near_flat'),(8,'missing_all'),
            (9,'missing_part'),(10,'line60'),(11,'large_fast')} <= tags
    h=rng.normal(size=(300,2)); assert not classify(h,np.ones_like(h,bool),10,np.ones(2))
    four=np.tile(np.array([0.,1.,1.,1.,1.,2.,2.,2.,2.,3.]),30)[:,None]
    assert any(k=='few_levels' for _,k,_,_ in classify(four,np.ones_like(four,bool),10,np.ones(1)))


def run_one(index):
    w=WORK[index];m=w['metadata'];s=m['source']
    out=DST/'parts'/f'{index:05d}.json'
    if out.exists(): return
    result=[]
    with np.load(m['array_file'],allow_pickle=False) as z:
        for mod,field,fs,names in [('EEG','eeg',200,m['EEG_channels']),('Hb','fnirs',10,m['fNIRS_positions'])]:
            x=z[field+'_x']; mask=z[field+'_observation_valid']
            assert x.shape==mask.shape and mask.dtype==bool and x.shape[:2]==(round(s['duration_s']*fs),len(names))
            scales=SCALES['eeg' if fs==200 else 'hb'][m['dataset']]
            scale=np.array([scales.get(p,{}).get('scale',np.nan) for p in names])
            for start in w['score_starts_s']:
                cut=slice(start*fs,(start+30)*fs)
                for c,label in enumerate([mod] if fs==200 else ['HbO','HbR']):
                    y=x[cut] if fs==200 else x[cut,:,c]
                    valid=mask[cut] if fs==200 else mask[cut,:,c]
                    hits=classify(y,valid,fs,scale)
                    result.append([start,label,len(names),int((~np.isfinite(scale)).sum()),hits])
    temp=out.with_suffix('.tmp')
    temp.write_text(json.dumps(result));temp.replace(out)


def init():
    global WORK,SCALES
    if not __debug__:
        raise RuntimeError('Audit identity and split checks require enabled assertions')
    parent=ROOT/CFG['source_plan']; cache=(ROOT/CFG['source_cache']).resolve()
    plan=json.loads(parent.read_text());assert plan['allowed_splits']==CFG['allowed_splits']==['train','validation']
    assert Path(plan['cache_root']).resolve()==cache and json.loads((cache/'verification.json').read_text())['status']=='passed'
    members={(r['dataset'],r['subject']):r['split'] for r in plan['subjects']}
    SCALES=json.loads((parent.parent/'scales.json').read_text())['positions']
    WORK=[w for w in plan['windows'] if w['score_starts_s']]
    cuts=defaultdict(list)
    for w in WORK:
        m=w['metadata'];s=m['source'];k=m['key']
        assert m['split']==s['split']==members[m['dataset'],m['subject']] and m['split'] in CFG['allowed_splits']
        assert m['key']==w['row']['key']==s['key'] and (m['dataset'],m['subject'])==(s['dataset'],s['subject'])
        expected=(cache/'windows'/(k+'.npz')).resolve()
        assert expected.is_relative_to(cache/'windows') and Path(m['array_file']).resolve()==expected
        for t in w['score_starts_s']:
            cuts[m['dataset'],m['subject'],s['native_record']].append(s['start_sample']+200*t)
    assert all(np.all(np.diff(sorted(a))>=6000) for a in cuts.values())


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['synthetic','pilot','census'])
    args=p.parse_args();synthetic()
    if args.stage=='synthetic':return
    init();(DST/'parts').mkdir(exist_ok=True,parents=True)
    lock=(DST/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    indices=list(range(len(WORK)))
    if args.stage=='pilot':
        # Disjoint interleaved subsets across the full metadata order; pilot work is retained.
        tests=[(n,np.linspace(j+1,len(WORK)-21+j,32,dtype=int).tolist()) for j,n in enumerate((2,6,12))]
    else:
        pilot=json.loads((DST/'pilot.json').read_text())
        tests=[(max(pilot,key=lambda v:v['parents_per_s'])['workers'],indices)]
    records=[]
    for workers,items in tests:
        before=time.monotonic()
        with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('fork')) as pool:
            for number,_ in enumerate(pool.map(run_one,items,chunksize=2),1):
                if number%250==0: print('parents',number,'/',len(items),flush=True)
        elapsed=time.monotonic()-before
        records.append(dict(workers=workers,parents=len(items),seconds=elapsed,parents_per_s=len(items)/elapsed,
                            peak_worker_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))
    (DST/(args.stage+'.json')).write_text(json.dumps(records,indent=2))
    print(records,flush=True)


if __name__=='__main__':
    main()
