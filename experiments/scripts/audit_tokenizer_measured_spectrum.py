"""Measure the retained train-only EEG voltage spectrum with the original point and patch masks."""
from collections import OrderedDict,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse,json,sys
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.data.dual_component_tokenizer import base_tokenizer_batch
from src.analysis.dual_component_tokenizer import base_training_aggregate
from run_montage_word_search import training_folds
from evaluate_joint_tokenizer import write
from src.utils.io import save_npz_atomic
STATE={}
BANDS=['below1','1through45','above45','49to51','59to61']


def fraction(x, duration, taper):
    f=np.fft.rfftfreq(x.shape[-1],1/200)
    wave=x*np.hanning(x.shape[-1]) if taper else x
    energy=abs(np.fft.rfft(wave))**2;energy[...,1:-1]*=2
    masks=[f<1,(f>=1)&(f<=45),f>45,(f>=49)&(f<=51),(f>=59)&(f<=61)]
    return np.stack([energy[...,mask].sum(-1) for mask in masks],-1)


def dataset(d):
    pop,rows,scales,out=STATE['pop'],STATE['rows'],STATE['scales'],STATE['out'];fields=defaultdict(list);cache=OrderedDict()
    for c in pop['cases']:
        if c['dataset']!=d:continue
        ex=np.array([(c['crop']['row'],c['crop']['start_s'],p) for p in c['crop']['positions']]);batch=base_tokenizer_batch(rows,ex,scales,array_cache=cache,cache_limit=5)
        m=batch['mask'].astype(bool);complete=m.all(2);fields['examples'].append(ex);fields['valid_points'].append(m.sum((1,2)));fields['complete_patches'].append(complete.sum(1))
        for b in 'xso':
            v=batch[b];whole=fraction(v.reshape(len(v),6000),30,False)
            patch=np.where(complete[:,:,None],fraction(v,2,True),0).sum(1)
            for k,value in [('whole',whole),('patch',patch)]:
                total=value[:,:3].sum(1,keepdims=True);fields[k+b].append(np.divide(value,total,out=np.full_like(value,np.nan,dtype=float),where=total>0))
    save_npz_atomic(out/(d+'.npz'),**{k:np.concatenate(v) for k,v in fields.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=yaml.safe_load(a.config.read_text());assert cfg['data']['validation']==cfg['data']['test']=='forbidden'
    _,pop,rows,scales=training_folds(cfg);out=a.output;out.mkdir(exist_ok=True,parents=True);STATE.update(pop=pop,rows=rows,scales=scales,out=out)
    with ProcessPoolExecutor(4) as pool:list(pool.map(dataset,sorted(pop['native_positions'])))
    chunks=defaultdict(list);idx={}
    for d in sorted(pop['native_positions']):
        with np.load(out/(d+'.npz')) as z:
            idx[d]=len(z['examples'])
            for k in z.files:chunks[k].append(z[k])
    arrays={k:np.concatenate(v) for k,v in chunks.items()};ex=arrays['examples'];assert len(ex)==17040
    result={}
    for k in ['whole','patch']:
        for b in 'xso':
            values=arrays[k+b];distribution={}
            for d in idx:
                sel=np.array([rows[i]['dataset']==d for i,t,pos in ex]);v=values[sel];out_of_band=v[:,0]+v[:,2]
                distribution[d]=dict(position_windows=int(sel.sum()),finite=int(np.isfinite(out_of_band).sum()),out_of_band_fraction_p50_p90_p99_max=np.nanquantile(out_of_band,[.5,.9,.99,1]),more_than_10pct_above45=int((v[:,2]>.1).sum()),more_than_half_line50=int((v[:,3]>.5).sum()))
            result[k+b]=dict(hierarchy=base_training_aggregate({'nmse':values},ex,rows)['metrics']['nmse'],distribution=distribution)
    write(out/'summary.json',dict(arrays=400,people=60,windows=len(ex),complete_patches=int(arrays['complete_patches'].sum()),columns=BANDS,
        whole='retained 6000-point score-input arrays as delivered by the original batch operator, no centering or taper; original valid point counts retained',
        patch='only original complete 400-point patches; Hann taper, summed squared spectra over valid patches within each position-window; denominator includes DC and Nyquist',
        verdict='These external-to-physiology-band energies diagnose the observed measurement support. Physiological scoring remains 1-45Hz.',results=result,validation_reads=0,test_reads=0))


if __name__=='__main__':main()
