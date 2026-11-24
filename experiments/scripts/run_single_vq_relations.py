"""Frozen four-dataset member and sequence audit, and event-anchored Single-Trial screen."""
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

import numpy as np
import torch
import yaml

import run_single_vq_readout as parent
from run_dual_component_tokenizer import owner_lock, read_json, write_json
from src.analysis.single_vq_relations import wave_attribute
from src.data.dual_component_tokenizer import base_tokenizer_batch
from src.utils.io import save_npz_atomic

CONTRACT = parent.CONTRACT.with_name('single_vq_relations_v1.yaml')
PLAN = yaml.safe_load(CONTRACT.read_text())
OUT = parent.capacity.ROOT/PLAN['run']


def maps():
    return parent.capacity.PANEL['rows'], parent.capacity.PANEL['scales']


def path(task):
    return OUT/'inputs'/parent.destination(task).name


def extract(task, pilot=False):
    start_time = time.monotonic()
    mod, ds, role, lo, hi = task
    if not pilot and path(task).is_file():
        return
    rows, scales = maps()
    ex = parent.capacity.PANEL['examples'][mod,role,ds][lo:hi]
    with np.load(parent.destination(task)) as z:
        np.testing.assert_array_equal(ex,z['examples'])
        valid = z['valid']
        result = {k:z[k] for k in z.files if k=='bg' or k.endswith('_id')}
    cache = OrderedDict()
    ys = []
    for j in range(0, min(len(ex),64) if pilot else len(ex),64):
        batch = base_tokenizer_batch(rows,ex[j:j+64],scales,array_cache=cache,cache_limit=64)
        good = batch['mask'].all(-1)
        np.testing.assert_array_equal(good,valid[j:j+64])
        ys.append(np.column_stack([wave_attribute(batch[k][good],mod) for k in ('x','s','o')]))
    y = np.concatenate(ys)
    if not np.isfinite(y).all():
        raise ValueError('nonfinite complete patch attribute')
    if not pilot:
        save_npz_atomic(path(task), examples=ex,valid=valid,y=y.astype('float32'),**result)
    return dict(modality=mod,dataset=ds,role=role,patches=len(y),seconds=time.monotonic()-start_time)


def collect(mod, seed, role):
    parts = []
    for task in parent.shards():
        if task[0]!=mod or task[2]!=role:
            continue
        with np.load(path(task)) as z:
            parts.append({key:z[key] for key in ('examples','valid','bg','y',f'S_{seed}_id',f'O_{seed}_id')})
    a = {key:np.concatenate([p[key] for p in parts]) for key in parts[0]}
    a['ids'] = np.column_stack([a.pop(f'{b}_{seed}_id') for b in 'SO'])
    rows,_ = maps()
    ex = a['examples']
    a['persons'] = np.array([parent.PEOPLE[rows[r]['dataset'],rows[r]['subject']] for r in ex[:,0]])
    a['person'] = np.repeat(a['persons'],a['valid'].sum(1))
    return a


def run_jobs(func,jobs,workers):
    with ProcessPoolExecutor(min(workers,len(jobs)),mp_context=multiprocessing.get_context('fork')) as pool:
        futures = {pool.submit(func,job):job for job in jobs}
        for n,f in enumerate(as_completed(futures),1):
            f.result()
            if n%16==0 or n==len(jobs):
                print(func.__name__,n,'/',len(jobs),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['pilot','extract','relations','controls','events','verify'])
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    torch.set_num_threads(1)
    with owner_lock(OUT,args.stage):
        cfg=OUT/'resolved_config.json'
        if cfg.exists() and read_json(cfg)!=PLAN:
            raise ValueError('contract changed; require a new run version')
        write_json(cfg,PLAN)
        if read_json(parent.OUT/'verification.json')['status']!='verified':
            raise ValueError('incomplete frozen readout parent')
        marker = OUT/(args.stage+'_status.json')
        if marker.exists() and read_json(marker)['status']=='completed':
            return
        write_json(marker,dict(status='running'))
        try:
            if args.stage!='verify':
                parent.panel()
            tasks=list(parent.shards()) if args.stage!='verify' else []
            if args.stage=='pilot':
                info=[extract(next(t for t in tasks if t[0]==m and t[2]==s),True)
                      for m in ('eeg','hb') for s in ('train','validation')]
                write_json(OUT/'pilot.json',info)
            if args.stage=='extract':
                run_jobs(extract,tasks,args.workers)
            if args.stage=='relations':
                from single_vq_member_sequence import analyze
                run_jobs(analyze,[(m,s) for m in PLAN['models'] for s in PLAN['models'][m]['seeds']],args.workers)
            if args.stage=='controls':
                from single_vq_member_sequence import supplementary
                run_jobs(supplementary,[(m,s) for m in PLAN['models'] for s in PLAN['models'][m]['seeds']],args.workers)
            if args.stage=='events':
                from single_vq_event_coupling import evaluate_events
                evaluate_events(args.workers)
            if args.stage=='verify':
                from single_vq_event_coupling import verify
                verify()
            write_json(marker,dict(status='completed'))
        except BaseException as e:
            write_json(marker,dict(status='failed',error=repr(e)))
            raise


if __name__=='__main__':
    main()
