"""Matched, unprotected development comparison for the November 24 four-branch refit."""
import argparse
import json
import multiprocessing
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from retrain_joint_qc_tokenizer import configuration, dump, OLD, OUT, ROOT
from apply_joint_tokenizer import checked_path, load_branch_pair
from audit_joint_tokenizer_patterns import pattern_attributes
from evaluate_joint_tokenizer import input_scales
from src.analysis.dual_component_tokenizer import base_training_aggregate, joint_reconstruction_metrics
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit, feature_means
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_joint_eeg, base_tokenizer_training_data
from src.utils.io import file_sha256, save_npz_atomic

HIST=OLD/'component_source_transfer/cross_branch_innovations_capacity_v2/portable_fold0_v1/bundle/bundle.json'
STATE={}


def bundle(arm):
    if arm=='previous':
        return HIST
    spec={'schema':'joint_tokenizer_inference_v1','experimental_status':'qc_retrained_development_reference'}
    for m,kind in [('eeg','cross_branch_innovation_eeg'),('hb','hb_native_id_readout')]:
        fit=json.loads((OUT/m/'fit_complete.json').read_text())
        path=Path(fit['model'])
        spec[m]=dict(kind=kind,path=str(path.relative_to(OUT)),sha256=file_sha256(path))
    scales=ROOT/configuration()['data']['run']/'scales.json'
    spec['scales']=dict(path=str(scales),sha256=file_sha256(scales))
    dest=OUT/'bundle.json'
    dump(dest,spec)
    return dest


def evaluation_arrays(mod,ds,role):
    rows=STATE['rows']
    allowed=STATE['allowed'][mod]
    if role=='train':
        return [ex for ex in STATE['patterns'][mod] if rows[ex[0]]['dataset']==ds and tuple(ex) in allowed]
    return [(i,g['start_s'],p) for i,r in enumerate(rows) if r['dataset']==ds and r['modality']==mod and r['split']=='validation'
            for g in r['inputs']['score'] for p in g['positions']]


def encode_dataset(task):
    mod,ds=task
    rows,scales=STATE['rows'],STATE['scales']
    loc=STATE['bundle']; spec=json.loads(loc.read_text())[mod]
    model=load_branch_pair(spec['kind'],checked_path(loc,spec))
    cache=OrderedDict()
    for role in ['train','validation']:
        dest=STATE['out']/f'{mod}_{ds}_{role}.npz'
        if dest.exists(): continue
        ex=np.asarray(evaluation_arrays(mod,ds,role),int)
        if not len(ex) or any(rows[i]['split']!=role for i in np.unique(ex[:,0])):
            raise ValueError('empty or wrong evaluation population')
        if mod=='eeg':
            # Contiguous groups preserve G0 row/start order; repeated crops in a ranked
            # training panel are sorted before encoding the full native montage.
            ex=ex[np.lexsort((ex[:,2],ex[:,1],ex[:,0]))]
            groups=np.split(ex,np.flatnonzero(np.any(np.diff(ex[:,:2],axis=0),axis=1))+1)
        else:
            groups=[ex[i:i+128] for i in range(0,len(ex),128)]
        metrics=defaultdict(list);ids=defaultdict(list)
        for i,take in enumerate(groups):
            batch=base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=64)
            if mod=='eeg':
                row,start=take[0,:2]
                names=model.metadata['positions'][ds]
                native={name.upper():j for j,name in enumerate(names)}
                pick=[native[rows[r]['positions'][p].upper()] for r,_,p in take]
                crop=dict(row=row,start_s=start,positions=take[:,2].tolist())
                fields,seen,_=base_tokenizer_joint_eeg(rows,crop,scales,names,cache,allowed_splits=(role,))
                values=fields['x'].T.reshape(1,len(names),15,400)
                all_ids=model.encode(values,ds,seen[None])
                if role=='validation':
                    waves=model.decode_ids(all_ids,ds)
                    pred={k:waves[k][0,pick] for k in ['S','O','sum']}
                words={k:all_ids[k][0,pick] for k in 'SO'}
            else:
                words=model.encode(batch['x'])
                if role=='validation':pred=model.decode_ids(words)
            for k in 'SO':
                assert words[k].shape==(len(take),15) and words[k].dtype.kind in 'iu'
                ids[k].append(words[k].astype('uint16'))
            if role=='validation':
                if any(not np.isfinite(pred[k]).all() for k in ['S','O','sum']):
                    raise FloatingPointError('nonfinite decoded waveforms')
                values=joint_reconstruction_metrics(pred,batch,1 if mod=='eeg' else 2,input_scales(rows,take,scales))
                for k,v in values['metrics'].items():metrics[k].append(v)
            if i%50==0:dump(dest.with_suffix('.progress.json'),dict(groups=i,total=len(groups)))
        save_npz_atomic(dest,examples=ex,**{k:np.concatenate(v) for k,v in (metrics|ids).items()})
    return mod,ds


def summarize(mod, state=None, code_sizes=None):
    state = STATE if state is None else state
    rows,scales=state['rows'],state['scales']
    labels=sorted({(r['dataset'],p) for r in rows if r['split']=='train' and r['modality']==mod for p in r['positions']})
    sites={k:i for i,k in enumerate(labels)}
    names=(['power_4_8','power_8_13','rms_envelope'] if mod=='eeg' else
           ['HbO_slope','HbR_slope','HbO_centered','HbR_centered'])
    width=400 if mod=='eeg' else 40
    _,fields=pattern_attributes(np.zeros((1,width)),mod)
    columns=np.concatenate([np.arange(*fields[k]) for k in names])
    offsets=np.cumsum([0]+[fields[k][1]-fields[k][0] for k in names])
    ranges=dict(zip(names,zip(offsets[:-1],offsets[1:])))
    codes=code_sizes or {'eeg':{'S':15351,'O':15351},'hb':{'S':129,'O':257}}[mod]
    audits={k:SparsePatternAudit(codes[k],4*(len(labels)+1),state['datasets'],len(columns),ranges) for k in 'SO'}
    cache=OrderedDict();examples=[];metrics=defaultdict(list)
    for role in ['train','validation']:
        for ds in state['datasets']:
            with np.load(state['out']/f'{mod}_{ds}_{role}.npz') as z:
                ex=z['examples'];words={b:z[b] for b in 'SO'}
                for b in 'SO':
                    assert words[b].min()>=0 and words[b].max()<codes[b]
                if role=='validation':
                    examples.append(ex)
                    for k in z.files:
                        if k not in ('examples','S','O'):metrics[k].append(z[k])
            for j in range(0,len(ex),64):
                take=ex[j:j+64]
                batch=base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=128)
                valid=batch['mask'].all(-1)
                site=np.array([sites.get((ds,rows[i]['positions'][p]),len(labels)) for i,_,p in take])
                bg=4*site[:,None]+np.digitize(np.sqrt((batch['x']**2).mean(axis=-1)),[.5,1,2])
                for b in 'SO':
                    audit=audits[b];ids=words[b][j:j+len(take)]
                    a,_=pattern_attributes(batch[b.lower()].reshape(-1,width),mod)
                    y=a[:,columns][valid.ravel()]
                    if role=='train':
                        if len(y):audit.fit(ids[valid],bg[valid],y)
                        continue
                    values=np.full((*valid.shape,2,len(names)),np.nan)
                    if len(y):
                        base,_,cond=audit.read(ids[valid],bg[valid])
                        values[valid]=np.stack([feature_means((t-y)**2,ranges) for t in [base,cond]],axis=1)
                    counts=valid.sum(1)
                    result=np.divide(np.nansum(values,axis=1),counts[:,None,None],
                                     out=np.full((len(take),2,len(names)),np.nan),where=counts[:,None,None]>0)
                    metrics[b+'_word'].append(result)
    combined={k:np.concatenate(v) for k,v in metrics.items()}
    full=base_training_aggregate(combined,np.concatenate(examples),rows)
    for b in 'SO':
        mse=np.array(full['metrics'][b+'_word']['macro'])
        full[b+'_single_word_gain']=dict(zip(names,(1-mse[1]/mse[0]).tolist()))
    dump(state['out']/(mod+'_summary.json'),full)


def run(arm):
    torch.set_num_threads(1)
    loc=bundle(arm)
    manifest,scales,_,_=base_tokenizer_training_data(configuration(),ROOT)
    rows=manifest['rows']
    allowed={m:{(i,g['start_s'],p) for i,r in enumerate(rows) if r['split']=='train' and r['modality']==m
                for g in r['inputs']['score'] for p in g['positions']} for m in ['eeg','hb']}
    with np.load(OLD/'panels.npz') as z:
        patterns={m:z[m+'_pattern_train'] for m in allowed}
    datasets=sorted({r['dataset'] for r in rows})
    folder=OUT/'evaluation'/arm
    folder.mkdir(parents=True,exist_ok=True)
    STATE.update(bundle=loc,rows=rows,scales=scales,patterns=patterns,allowed=allowed,out=folder,datasets=datasets)
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:
        list(pool.map(encode_dataset,[(m,d) for m in ['eeg','hb'] for d in datasets]))
    for m in ['eeg','hb']:summarize(m)
    if arm=='retrained':
        prior=folder.parent/'previous'
        comparison={}
        for m in ['eeg','hb']:
            for ds in datasets:
                for role in ['train','validation']:
                    file=f'{m}_{ds}_{role}.npz'
                    with np.load(prior/file) as old, np.load(folder/file) as new:
                        np.testing.assert_array_equal(old['examples'],new['examples'])
            old=json.loads((prior/(m+'_summary.json')).read_text())
            new=json.loads((folder/(m+'_summary.json')).read_text())
            differences={k:(np.array(old['metrics'][k]['macro'])-new['metrics'][k]['macro']).tolist()
                         for k in ('nmse','rmse_native')}
            old_subject={(x['dataset'],x['subject']):np.array(x['value'])
                         for x in old['metrics']['nmse']['by_subject']}
            subject_differences=[dict(dataset=x['dataset'],subject=x['subject'],
                value=(old_subject[x['dataset'],x['subject']]-x['value']).tolist())
                for x in new['metrics']['nmse']['by_subject']]
            if len(subject_differences)!=16: raise ValueError('incomplete development population')
            comparison[m]=dict(positions=new['inputs'],people=new['subjects'],
                previous_minus_retrained=differences,paired_subject_NMSE=subject_differences,
                single_word_gain={b:dict(previous=old[b+'_single_word_gain'],
                retrained=new[b+'_single_word_gain']) for b in 'SO'})
        dump(folder.parent/'paired_comparison.json',comparison)
    dump(folder/'completion.json',dict(status='completed',bundle=str(loc),confirmation_reads=0,test_reads=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm',choices=['previous','retrained'])
    run(parser.parse_args().arm)
