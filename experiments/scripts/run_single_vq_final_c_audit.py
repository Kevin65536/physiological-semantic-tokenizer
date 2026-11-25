"""Frozen C checkpoint replay on the registered single-VQ development panel."""
import argparse
from collections import OrderedDict, defaultdict
from functools import lru_cache
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

import run_single_vq_readout as readout
import run_single_vq_relations as relations
import run_single_vq_spectral_location as spectral
from run_dual_component_tokenizer import _base_training_model, identity, owner_lock, read_json, write_json
from src.data.dual_component_tokenizer import base_tokenizer_batch
from src.utils.io import save_npz_atomic

CONTRACT = spectral.CONTRACT.with_name('single_vq_final_c_audit_v1.yaml')
PLAN = yaml.safe_load(CONTRACT.read_text())
OUT = spectral.ROOT / PLAN['run']
SEEDS = PLAN['models']['eeg']['seeds']


@lru_cache(maxsize=4)
def network(mod, size, seed, device='cpu'):
    if (mod, size) != ('eeg', spectral.K) or seed not in SEEDS:
        raise ValueError('expected a frozen C checkpoint')
    cfg = spectral.training_cfg(seed)
    folder = spectral.folder(seed, 'C')
    training = read_json(folder/'summary.json')
    state = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=False)
    objective = dict(calibration=read_json(spectral.OUT/'calibration.json'),
                     location=spectral.PLAN['arms']['C'], weight=spectral.PLAN['weight'])
    signature = state['signature']
    if (training['status'] != 'completed' or signature != training['signature']
            or signature['config'] != identity(cfg) or signature['task'] != spectral.TASK
            or state['step'] != spectral.PLAN['training']['total_steps']
            or signature['spectral_objective'] != identity(objective)):
        raise ValueError('C checkpoint provenance mismatch')
    model = _base_training_model(cfg, spectral.TASK, single_id_spectrum=True)
    model.load_state_dict(state['model_state'])
    return model.eval().to(device)


def inputs(task):
    return OUT/'inputs'/readout.destination(task).name


def tasks(role=None):
    return [t for t in readout.shards() if t[0]=='eeg' and (role is None or t[2]==role)]


@lru_cache(maxsize=4)
def retained(seed, ds, role):
    path = spectral.folder(seed,'C')/'evaluation'/f'{ds}_{role}.npz'
    with np.load(path) as z:
        np.testing.assert_array_equal(z['examples'], readout.capacity.PANEL['examples']['eeg',role,ds])
        valid = z['valid']
        ids = np.full(valid.shape,-1,int)
        ids[valid] = z['id']
    return ids


@torch.inference_mode()
def extract(task, pilot=False):
    mod, ds, role, lo, hi = task
    dest = inputs(task)
    if dest.exists() and not pilot:
        return
    rows, scales = relations.maps()
    with np.load(relations.path(task)) as z:
        ex, valid = z['examples'], z['valid']
        np.testing.assert_array_equal(ex, readout.capacity.PANEL['examples'][mod,role,ds][lo:hi])
        values = {k:z[k] for k in ['bg','y']}
    cache = OrderedDict()
    data = defaultdict(list)
    began = time.monotonic()
    for i in range(0,min(len(ex),64) if pilot else len(ex),64):
        take = ex[i:i+64]
        batch = base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=64)
        good = batch['mask'].all(-1)
        np.testing.assert_array_equal(good,valid[i:i+len(take)])
        x = torch.as_tensor(batch['x'],device='cuda')
        for seed in SEEDS:
            model = network('eeg',8192,seed,'cuda')
            result = model(x)
            ref = retained(seed,ds,role)[lo+i:lo+i+len(take)]
            np.testing.assert_array_equal(good,ref>=0)
            for b in 'SO':
                ids = result['ids'][b].cpu().numpy()[good]
                data[f'{b}_{seed}_id'].append(ids.astype(np.uint16))
                if b=='S':
                    data[f'replay_diff_{seed}'].append(np.count_nonzero(ids!=ref[good]))
                if role=='validation':
                    feat,_ = readout.attributes(result[b].cpu().numpy(),mod)
                    data[f'{b}_{seed}_wave'].append(feat[good.ravel()])
            if pilot:
                for b,p in model.decode_ids(result['ids']).items():
                    torch.testing.assert_close(p,result[b],rtol=0,atol=0)
    for seed in SEEDS:
        diff = sum(data.pop(f'replay_diff_{seed}'))
        values[f'replay_diff_{seed}'] = diff
    if not pilot:
        save_npz_atomic(dest,examples=ex,valid=valid,**values,**{k:np.concatenate(v) for k,v in data.items()})
    return dict(dataset=ds,split=role,seconds=time.monotonic()-began,
                replay_disagreements=[int(values[f'replay_diff_{s}']) for s in SEEDS])


def collect(mod,seed,role):
    parts = defaultdict(list)
    for task in tasks(role):
        with np.load(inputs(task)) as z:
            for k in ('examples','valid','y','bg'):
                parts[k].append(z[k])
            parts['ids'].append(np.column_stack([z[f'{b}_{seed}_id'] for b in 'SO']).astype(np.int64))
    a = {k:np.concatenate(v) for k,v in parts.items()}
    rows,_ = relations.maps()
    a['persons'] = np.array([readout.PEOPLE[rows[r]['dataset'],rows[r]['subject']] for r in a['examples'][:,0]])
    a['person'] = np.repeat(a['persons'],a['valid'].sum(1))
    return a


def relation_job(seed):
    import single_vq_member_sequence as member
    source = sys.modules[__name__]
    member.analyze(('eeg',seed),source=source)
    member.supplementary(('eeg',seed),source=source)


def verify():
    for stage in ('extract','readout','relations','context','events'):
        if read_json(OUT/f'{stage}_status.json')['status']!='completed':
            raise ValueError(f'incomplete stage {stage}')
    counts=defaultdict(int)
    disagreement=defaultdict(int)
    for task in tasks():
        with np.load(inputs(task)) as z, np.load(relations.path(task)) as frozen:
            for k in ('examples','valid','bg','y'):
                np.testing.assert_array_equal(z[k],frozen[k])
            n=int(z['valid'].sum())
            counts[task[2]]+=n
            if z['valid'].shape!=(len(z['examples']),15):
                raise ValueError('token grid changed')
            for seed in SEEDS:
                disagreement[seed]+=int(z[f'replay_diff_{seed}'])
                ref=retained(seed,task[1],task[2])[task[3]:task[4]]
                np.testing.assert_array_equal(z[f'S_{seed}_id'],ref[z['valid']])
                for b in 'SO':
                    ids=z[f'{b}_{seed}_id']
                    if ids.shape!=(n,) or ids.max()>=8192:
                        raise ValueError('invalid final C IDs')
    context=pd.read_csv(OUT/'context.csv')
    ctx=context.groupby(['seed','branch','dataset','subject'])[
        ['id_same','family_same','family_donor_same','absolute_profile_drift']].mean().groupby(level=[0,1,2]).mean().groupby(level=[0,1]).mean()
    output=[]
    for seed in SEEDS:
        model=OUT/f'eeg_{seed}'
        info=read_json(model/'completion.json')
        if (info['train_people'],info['dev_people'],info['valid_dev_patches'])!=(60,16,counts['validation']):
            raise ValueError('incorrect relations support')
        directory=OUT/f'events_{seed}'
        for name in ('event_support.csv','upstream_event_exclusions.csv','coupling_pairs.csv'):
            pd.testing.assert_frame_equal(pd.read_csv(directory/name),pd.read_csv(relations.OUT/name))
        coupling=pd.read_csv(directory/'coupling_contrasts.csv')
        old=pd.read_csv(relations.OUT/'coupling_contrasts.csv')
        check=coupling.reference.isin(['base','continuous'])
        np.testing.assert_allclose(coupling.loc[check,'reference_nmse'],old.loc[check,'reference_nmse'],rtol=1e-10)
        evt=pd.read_csv(directory/'event_contrasts.csv')
        raw=evt.measure.isin(['alpha','HbO','HbR'])
        pd.testing.assert_frame_equal(evt[raw],pd.read_csv(relations.OUT/'event_contrasts.csv')[raw])
        event=evt[evt.primary].to_dict('records')
        if len(event)!=1 or event[0]['people']!=3:
            raise ValueError('primary event contrast support changed')
        readouts={}
        for b in 'SO':
            scores=pd.read_csv(OUT/f'readout_{seed}_{b}/scores.csv')
            frame=scores[scores.dataset=='macro']
            readouts[b]={m:g.set_index('attribute').gain.to_dict() for m,g in frame.groupby('method')}
            if len(scores.subject.dropna().unique())>16 or len(scores[scores.subject.notna()].groupby(['dataset','subject']))!=16:
                raise ValueError('readout people changed')
        order=pd.read_csv(model/'sequence.csv')
        conditional=pd.read_csv(model/'conditional.csv')
        output.append(dict(seed=seed,readout_gain=readouts,context=ctx.loc[seed].to_dict('index'),
                    measured_X_O_added=conditional.query("target=='X' and reference=='S' and candidate=='SO'").iloc[0].to_dict(),
                    adjacent_S_order=order.query("adjacent and branch=='S' and reference in ['bag','bigram_dwell']").to_dict('records'),
                    primary_event=event[0],two_color=coupling.query("color=='two_color'").to_dict('records'),
                    event_population=read_json(directory/'events_summary.json')))
    write_json(OUT/'summary.json',dict(status='completed',experiment_id=PLAN['experiment_id'],
              complete_patches=counts,S_replay_disagreements={str(s):disagreement[s] for s in SEEDS},
              population=dict(train=60,ordinary_development=16,eeg_development_positions=53476),
              event_support_replayed=True,continuous_comparators_replayed=True,models=output,
              interpretation='Development only. Envelope readout and within-modality complement replicated. Event direction differs between seeds; aligned EEG increases Hb history prediction error.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['pilot','extract','relations','events','readout','context','verify'])
    p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    torch.set_num_threads(1)
    with owner_lock(OUT,args.stage):
        if (OUT/'resolved_config.json').exists() and read_json(OUT/'resolved_config.json')!=PLAN:
            raise ValueError('contract changed')
        write_json(OUT/'resolved_config.json',PLAN)
        for status in (relations.OUT/'summary.json', spectral.OUT/'summary.json'):
            if read_json(status)['status']!='completed':
                raise ValueError(f'incomplete parent {status}')
        readout.panel()
        marker = OUT/(args.stage+'_status.json')
        if marker.exists() and read_json(marker)['status']=='completed':
            return
        write_json(marker,dict(status='running'))
        try:
            if args.stage=='pilot':
                results=[extract(next(t for t in tasks() if t[1]==ds and t[2]=='validation'),True) for ds in readout.capacity.DATASETS]
                write_json(OUT/'pilot.json',results)
            elif args.stage=='extract':
                for i,t in enumerate(tasks(),1):
                    r=extract(t)
                    print(i,len(tasks()),r,flush=True)
            elif args.stage=='verify':
                verify()
            elif args.stage=='relations':
                relations.run_jobs(relation_job,SEEDS,args.workers)
            elif args.stage=='events':
                from single_vq_event_coupling import evaluate_events
                for seed in SEEDS:
                    evaluate_events(args.workers,out=OUT/f'events_{seed}',network=network,seed=seed,families=OUT)
            else:
                from single_vq_final_c_probes import context_audit, readout_job
                if args.stage=='readout':
                    relations.run_jobs(readout_job,[(s,b) for s in SEEDS for b in 'SO'],args.workers)
                else:
                    context_audit()
            write_json(marker,dict(status='completed'))
        except BaseException as error:
            write_json(marker,dict(status='failed',error=repr(error)))
            raise


if __name__=='__main__':
    sys.modules['run_single_vq_final_c_audit']=sys.modules[__name__]
    main()
