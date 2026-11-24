"""Compare the same physical EEG patch in two real, shifted 30-second montages."""
import argparse
from collections import OrderedDict, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.analysis.tokenizer_pattern_audit import attributes
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_joint_eeg, base_tokenizer_training_data
from src.tokenizers.spatial_waveform_dictionary import load_model
from audit_joint_tokenizer_patterns import hierarchy, write_json


def run(args):
    cfg=yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest,scales,_,_=base_tokenizer_training_data(cfg,ROOT)
    rows=manifest['rows']
    model=load_model(args.model)
    if (model.metadata['scales_identity']!=manifest['scales_identity'] or
            model.metadata['input_identity']!=manifest['cfg_identity']):
        raise ValueError('frozen G0 identity differs')
    with np.load(args.panels) as p:
        train=p['eeg_pattern_train']; val=p['eeg_screen']
    cache=OrderedDict()
    paired=defaultdict(list)
    sums={b:np.zeros(10) for b in 'SO'}
    squares={b:np.zeros(10) for b in 'SO'}
    count=0
    for i,s,p in train:
        if rows[i]['modality']!='eeg' or rows[i]['split']!='train':
            raise ValueError('train variance boundary')
    for start in range(0,len(train),64):
        x=base_tokenizer_batch(rows,train[start:start+64],scales,array_cache=cache,cache_limit=24)
        valid=x['mask'].all(-1)
        for b in 'SO':
            vals, fields=attributes(x[b.lower()][valid], 'eeg')
            a=np.column_stack([vals[:,slice(*fields[k])] for k in ('power_4_8','power_8_13','rms_envelope')])
            sums[b]+=a.sum(0); squares[b]+=(a*a).sum(0)
        count+=valid.sum()
    variance={b: squares[b]/count-(sums[b]/count)**2 for b in 'SO'}
    with np.load(args.profiles/'training_attribute_profiles.npz') as f:
        mean={b:f[b+'_train_attribute_mean'] for b in 'SO'}
        support={b:f[b+'_train_count'] for b in 'SO'}
    with np.load(args.profiles/'validation_IDs.npz') as f:
        np.testing.assert_array_equal(f['examples'],val)
        prior={b:f[b].copy() for b in 'SO'}
        bands=f['fixed_band'].copy()
    for j,(i,s,p) in enumerate(val):
        row=rows[i]
        if row['modality']!='eeg' or row['split']!='validation': raise ValueError('invalid screen row')
        times={v['start_s'] for v in row['inputs']['score'] if p in v['positions']}
        if s+30 in times: paired[int(i),int(s),1].append(j)
        elif s-30 in times: paired[int(i),int(s),-1].append(j)
    values={b:[] for b in 'SO'}
    kept=[]; availability_changes=0
    for (i,s,direction), idx in paired.items():
        record=rows[i]; dataset=record['dataset']; names=model.metadata['positions'][dataset]
        take=val[idx]
        original=base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=24)
        valid=original['mask'][:,7].all(-1)
        if not valid.any():continue
        arrays=[]; available=[]
        for time in (s,s+direction*30):
            fields, observed, _=base_tokenizer_joint_eeg(rows,dict(row=i,start_s=time,positions=[]),scales,names,
                                                        allowed_splits=('validation',))
            arrays.append(fields['x'].T.reshape(1,len(names),15,400))
            available.append(observed)
        both=available[0]&available[1]
        avail_changed=not np.array_equal(both,available[0])
        availability_changes+=int(avail_changed)
        x0,x1=arrays
        shifted=np.concatenate([x0[:,:,1:],x1[:,:,:1]],axis=2) if direction==1 else np.concatenate([x1[:,:,-1:],x0[:,:,:-1]],axis=2)
        shifted_idx=7-direction
        np.testing.assert_array_equal(x0[:,:,7],shifted[:,:,shifted_idx])
        second=model.encode(np.where(both[None,:,None,None],shifted,0.),dataset,both[None])
        first=model.encode(np.where(both[None,:,None,None],x0,0.),dataset,both[None])
        names_to_pos={n.upper():n0 for n0,n in enumerate(names)}
        for j,use in zip(idx,valid):
            if not use:continue
            native=names_to_pos[record['positions'][val[j,2]].upper()]
            if not both[native]:continue
            band=bands[j]
            kept.append(j)
            for b in 'SO':
                old=int(first[b][0,native,7]); new=int(second[b][0,native,shifted_idx])
                if not avail_changed and old!=prior[b][j,7]: raise ValueError('source code disagrees with original audit')
                known=support[b][band,old]>0 and support[b][band,new]>0
                change=(mean[b][band,new]-mean[b][band,old])**2
                if not known:change[:]=np.nan
                values[b].append([float(old!=new),float(known),change[0],change[1],np.mean(change[2:])])
    result={}
    for b in 'SO':
        stats=hierarchy(np.asarray(values[b]),val[kept],rows)
        macro=np.array(stats['macro'])
        scale=np.r_[variance[b][:2],np.mean(variance[b][2:])]
        relative=macro[2:]/scale
        result[b]=dict(hierarchy=stats,changed_fraction=macro[0],two_supported_fraction=macro[1],
                       change_over_train_variance=relative.tolist(),threshold=.25,passed=bool((relative<.25).all()))
    write_json(args.output/'context_summary.json',dict(status='completed',model=str(args.model),panel='screen',
        planned=len(val),paired_planned=sum(map(len,paired.values())),scored=len(kept),training_complete_patches=int(count),
        paired_full_montages_with_changed_aux_support=availability_changes,availability='intersection of original and adjacent observed positions for both encodes',
        attributes=['power_4_8','power_8_13','rms_envelope'],branches=result))
    np.savez_compressed(args.output/'paired_attributes.npz',examples=val[kept],**{b:np.array(values[b]) for b in 'SO'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('model','panels','profiles','output'):parser.add_argument('--'+k,type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'context_summary.json').exists():raise FileExistsError('retained completed context review')
    torch.set_num_threads(3)
    run(args)
