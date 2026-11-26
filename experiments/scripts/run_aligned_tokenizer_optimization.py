"""Inner-subject experiment on the aligned tokenizer, starting with matched continuations."""
import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

import retrain_single_vq_aligned as aligned
from run_dual_component_tokenizer import (
    _base_training_model, _base_training_batch, _state_refinement_loss, _torch_save,
    identity, owner_lock, read_json, train_base_tokenizer, write_json,
)
from src.data.dual_component_tokenizer import (
    base_tokenizer_batch, base_tokenizer_examples, base_tokenizer_sample, base_tokenizer_sampling_index,
    base_tokenizer_scales, base_tokenizer_training_data,
)
from src.analysis.dual_component_tokenizer import eeg_spectral_attributes

ROOT = aligned.ROOT
CONTRACT = aligned.CONTRACT.with_name('aligned_tokenizer_optimization_v1.yaml')
PLAN = yaml.safe_load(CONTRACT.read_text())
OUT = ROOT / PLAN['run']
ORIGINAL = aligned.spectral.PLAN['recorded_root']


def inner_subjects(rows):
    pools = {}
    for r in rows:
        if r['split'] == 'train' and any(g['positions'] for g in r['inputs']['score']):
            pools.setdefault(r['dataset'], set()).add(r['subject'])
    total = sum(map(len, pools.values()))
    if total != PLAN['scope']['train_people'] + PLAN['scope']['inner_people'] or len(pools) != 4:
        raise ValueError('unexpected original train population')
    quotas = {d: len(v)*PLAN['scope']['inner_people']/total for d, v in pools.items()}
    counts = {d: int(v) for d, v in quotas.items()}
    left = PLAN['scope']['inner_people'] - sum(counts.values())
    for d in sorted(pools, key=lambda d: (-(quotas[d]-counts[d]), d))[:left]:
        counts[d] += 1
    rng = np.random.default_rng(PLAN['split_seed'])
    held = set()
    for d in sorted(pools):
        held.update((d, str(s)) for s in rng.permutation(sorted(pools[d]))[:counts[d]])
    fit = {(d,s) for d, people in pools.items() for s in people} - held
    return fit, held


def qc_moments(row):
    """Replace pre-exclusion moments only for score groups actually affected by new QC."""
    if not any(g.get('new_qc_exclusions', 0) for g in row['unavailable_inputs']['score']):
        record = read_json(Path(row['folder']).parent/'record.json')
        matches = [v for v in record['rows'] if v['parent'] == row['parent']
                   and v['modality'] == row['modality'] and v['split'] == 'train'
                   and v['subject'] == row['subject'] and v['positions'] == row['positions']]
        if len(matches) != 1:
            raise ValueError('fit-row moment source mismatch')
        return matches[0]['moments']
    folder = Path(row['folder'])
    mask = np.load(folder/'mask.npy', mmap_mode='r')
    result = {}
    for k in 'xso':
        a = np.load(folder/(k+'.npy'), mmap_mode='r')
        for diff in (False, True):
            sums = np.zeros((3, len(row['positions']), row['channels']))
            for g in row['inputs']['score']:
                pos = g['positions']
                if not pos:
                    continue
                sl = slice(g['start_s']*row['rate'], (g['start_s']+30)*row['rate'])
                data = np.asarray(a[sl][:,pos], dtype=float)
                valid = mask[sl][:,pos]
                if diff:
                    data = np.diff(data, axis=0)
                    valid = valid[1:] & valid[:-1]
                values = np.where(valid, data, 0.)
                sums[:,pos] += np.stack([valid.sum(0), values.sum(0), (values**2).sum(0)])
            result[('d' if diff else '')+k] = dict(zip(('count','sum','second'), sums.tolist()))
    return result


def partition(manifest, mod):
    fit, held = inner_subjects(manifest['rows'])
    rows = []
    for source in manifest['rows']:
        key = source['dataset'], source['subject']
        if source['split'] != 'train' or source['modality'] != mod or key not in fit | held:
            continue
        row = dict(source)
        if key in held:
            row.update(split='validation', moments={})
        else:
            row['moments'] = qc_moments(row)
        rows.append(row)
    cfg, _ = aligned.settings(mod)
    norm = yaml.safe_load((ROOT/cfg['data']['contract']).read_text())['normalization']
    scales = base_tokenizer_scales(rows, norm)
    excluded = 0
    for row in rows:
        available = scales['positions'][mod][row['dataset']]
        for role, groups in list(row['inputs'].items()):
            kept = []
            for g in groups:
                positions = [p for p in g['positions'] if row['positions'][p] in available]
                if role == 'score':
                    excluded += len(g['positions'])-len(positions)
                kept.append(dict(g,positions=positions))
            row['inputs'][role] = kept
        row.pop('moments')
        row.pop('unavailable_inputs')
    index = base_tokenizer_sampling_index(rows,mod)
    g0 = yaml.safe_load((ROOT/cfg['data']['contract']).read_text())
    init = base_tokenizer_sample(np.random.default_rng(g0['sampling']['seed']),index,0,initialization=True)
    assert all(rows[i]['split']=='train' for i,_,_ in init)
    split = dict(fit=sorted(fit), inner=sorted(held), excluded_score_site_positions=excluded,
                 source=manifest['plan_identity'], scales=identity(scales), initialization_records=len(init))
    folder = OUT/mod
    if (folder/'split.json').exists() and read_json(folder/'split.json') != json.loads(json.dumps(split)):
        raise ValueError('inner population or scales changed')
    write_json(folder/'split.json',split)
    write_json(folder/'scales.json',scales)
    return dict(rows=rows,split=split,window_exclusions=manifest['window_exclusions']),scales,{mod:dict(initialization=init)}


def settings(mod):
    cfg,task=aligned.settings(mod)
    cfg['experiment_id']=PLAN['experiment_id']
    return cfg,task


def load(mod):
    cfg,_=settings(mod)
    meta,*_=base_tokenizer_training_data(cfg,ROOT,recorded_root=ORIGINAL)
    return partition(meta,mod)


def calibrate(rows,scales):
    cache=OrderedDict()
    result={}
    ex=base_tokenizer_examples(rows,'eeg','train')
    for ds in sorted({r['dataset'] for r in rows}):
        panel=[v for v in ex if rows[v[0]]['dataset']==ds]
        # Max 4096 distinct score positions per dataset, all fitting people only.
        rng=np.random.default_rng(PLAN['split_seed'])
        chosen=rng.choice(len(panel),min(len(panel),4096),replace=False)
        attr=[]
        for start in range(0,len(chosen),64):
            triples=[panel[j] for j in chosen[start:start+64]]
            assert all(rows[i]['split']=='train' for i,_,_ in triples)
            b=base_tokenizer_batch(rows,triples,scales,array_cache=cache)
            scale=torch.tensor([scales['positions']['eeg'][ds][rows[i]['positions'][pos]]['scale'] for i,_,pos in triples],dtype=torch.float64)
            a=eeg_spectral_attributes(torch.from_numpy(b['s']).double(),scale).numpy()
            attr.append(a[b['mask'].all(-1)])
        a=np.concatenate(attr)
        assert np.isfinite(a).all() and (a.std(0)>0).all()
        result[ds]=dict(mean=a.mean(0).tolist(),sd=a.std(0).tolist(),patches=len(a),examples=len(chosen))
    return result


def baseline(mod):
    cfg,task=settings(mod)
    folder=OUT/mod
    cached={}
    def view(manifest):
        if 'data' not in cached:
            cached['data']=partition(manifest,mod)
        return cached['data']
    args=dict(recorded_root=ORIGINAL,training_partition=view)
    train_base_tokenizer(cfg,ROOT,folder,dict(task,candidate='warmup'),'cuda:0',**args)
    obj=None
    if mod=='eeg':
        meta,scales,_=cached['data']
        params=calibrate(meta['rows'],scales)
        write_json(folder/'calibration.json',params)
        obj=dict(location='single_id_spectrum',weight=.01,calibration=params)
    train_base_tokenizer(cfg,ROOT,folder,task,'cuda:0',spectral_objective=obj,**args)


def frozen_decoder(model):
    for name,param in model.named_parameters():
        param.requires_grad_(any(f'.{part}.' in name for part in ('from_latent','decoder','output')))
    for branch in model.branches.values():
        branch['quantizer'].eval()
    if not any(v.requires_grad for v in model.parameters()):
        raise RuntimeError('empty decoder')


def continuation(mod,arm):
    cfg,task=settings(mod)
    meta,scales,_=load(mod)
    rows=meta['rows']
    groups=base_tokenizer_sampling_index(rows,mod)
    folder=OUT/mod/arm
    parent=OUT/mod/'models'/'__'.join(task.values())/'checkpoint.pt'
    obj=(dict(location='single_id_spectrum',weight=.01,calibration=read_json(OUT/mod/'calibration.json'))
         if mod=='eeg' else None)
    spec=dict(cfg['architectures']['waveform'])
    if obj:
        spec['spectral_objective']=obj
    model=_base_training_model(cfg,task,single_id_spectrum=bool(obj)).cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=cfg['training']['weight_decay'])
    saved=torch.load(folder/'resume.pt' if (folder/'resume.pt').exists() else parent,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model_state'])
    opt.load_state_dict(saved['optimizer_state'])
    if arm=='decoder':
        frozen_decoder(model)
    rng=np.random.default_rng()
    rng.bit_generator.state=saved['numpy_rng']
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state(saved['cuda_rng'])
    cache=OrderedDict()
    history=saved.get('continuation_history',[])
    for step in range(saved['step']+1,12001):
        batch=_base_training_batch(rows,base_tokenizer_sample(rng,groups,64),scales,'cuda:0',cache,cfg,
                                   calibration=obj['calibration'] if obj else None)
        loss,_=_state_refinement_loss(model(batch['x']),batch,spec)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        opt.step()
        if step%250==0:
            history.append(dict(step=step,loss=float(loss.detach())))
            state=dict(step=step,model_state=model.state_dict(),optimizer_state=opt.state_dict(),numpy_rng=rng.bit_generator.state,
                       torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),continuation_history=history)
            if step%1000==0:
                _torch_save(folder/f'step_{step}.pt',state)
            _torch_save(folder/'resume.pt',state)
            write_json(folder/'progress.json',dict(step=step,history=history))
    write_json(folder/'summary.json',dict(status='completed',arm=arm,step=12000,history=history))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['synthetic','baseline','full','decoder'])
    parser.add_argument('mod',choices=['eeg','hb'])
    args=parser.parse_args()
    torch.set_num_threads(1)
    folder=OUT/args.mod
    folder.mkdir(parents=True,exist_ok=True)
    if args.action=='synthetic':
        cfg,task=settings(args.mod)
        result=aligned.spectral.synthetic('C' if args.mod=='eeg' else 'A','cuda:0',24,64,
                                          modality=args.mod,dimension=128,size=PLAN['dimensions'][args.mod+'_K'])
        write_json(folder/'pilot.json',result)
    elif args.action=='baseline':
        baseline(args.mod)
    else:
        with owner_lock(folder/args.action):
            continuation(args.mod,args.action)


if __name__=='__main__':
    main()
