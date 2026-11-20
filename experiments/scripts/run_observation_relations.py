#!/usr/bin/env python3
"""One-shot frozen-word development screen. Only --stage screen reads measured X.

Run the five named data-free test files and preflight first; run the measured
stage under the user's service manager. --output must be a fresh dated run in
the existing observation suite. No fit/update method is called on a real codec.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.data.observation_relation_inputs import digest, load_routers, replay
from src.data.observation_tokenizer_panel import G0_ROOT

CONFIG=ROOT/'experiments/configs/physiology_semantic_tokenizer/observation_relations_v1.yaml'
SUITE=ROOT/'experiments/runs/physiology_semantic_tokenizer/observation_factorization'
PARTS=('phenotypes','dynamics','coupling_primary','coupling_four','events')
NAMES=('observation_relations_screen','observation_token_phenotypes','observation_token_dynamics','observation_token_coupling','observation_token_events')
TESTS=[f'tests/test_{n}.py' for n in NAMES]
SOURCES=[CONFIG,Path(__file__), ROOT/'src/data/observation_relation_inputs.py', ROOT/'src/analysis/observation_word_limits.py',
         *[ROOT/f'src/analysis/{n}.py' for n in NAMES[1:]], *[ROOT/path for path in TESTS],
         *[ROOT/f'src/{f}' for f in ('analysis/observation_task_probe.py','analysis/observation_tokenizer_evaluation.py','data/observation_tokenizer_panel.py','tokenizers/factorized_observation.py')]]


def now(): return datetime.now(timezone.utc).isoformat()


def save(path,obj):
    def convert(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        if isinstance(x,Path): return str(x)
        raise TypeError(type(x).__name__)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2,default=convert)+'\n');os.replace(tmp,path)


def js(p): return json.loads(Path(p).read_text())


def config(path):
    c=yaml.safe_load(Path(path).read_text());p=ROOT/c['parent'];r=ROOT/c['release']
    if (json.loads(json.dumps(c,allow_nan=False))!=c or c!=yaml.safe_load(CONFIG.read_text()) or c['experiment_id']!='TK-OBS-RELATIONS-v1' or c['splits']!=['train','validation']
            or any(c[k] is not False for k in ('raw_reads','refit_tokenizer','teacher_reads')) or (c['test_access'],c['protected_access'])!=('forbidden','forbidden')
            or (p,r)!=(SUITE/'20261120_development_v3',SUITE/'20261120_release_v1') or ROOT/c['g0']!=G0_ROOT
            or c['event_cache']!='data/cache/physiology_downstream_v2' or c['k']!=16 or c['patch_s']!=2
            or c['modalities']!={'eeg':[1,400],'hb':[2,20]} or c['coupling_primary_datasets']!=['eeg_fnirs_single_trial','simultaneous_eeg_nirs']):
        raise ValueError('wrong frozen scope or tensors')
    return c


def store(out,name,result):
    paths={};remaining={}
    for key,value in result.items():
        if isinstance(value,pd.DataFrame):
            target=out/f'{name}_{key}.csv';value.to_csv(target,index=False);paths[key]=target.name
        else: remaining[key]=value
    if remaining:
        target=out/f'{name}.json';save(target,remaining);paths['json']=target.name
    return paths


def computation(name,out,c,frames=None,waves=None,routers=None):
    started=time.perf_counter()
    if name=='events':
        from src.analysis.observation_token_events import run_event_relations
        result=run_event_relations(cache_root=ROOT/c['event_cache'],g0_root=G0_ROOT,eeg_codec=routers['eeg'],hb_codec=routers['hb'])
        fingerprint=result['kept_keys_sha256'];prior=js(ROOT/c['parent']/'task_probe.json')
        retained=sorted((e['split'],e['subject'],e['record'],e['event_index'],e['condition']) for e in prior['events'] if e['status']=='included')
        if hashlib.sha256(json.dumps(retained,separators=(',',':')).encode()).hexdigest()!=fingerprint: raise ValueError('event support changed since b277a356')
        result['same_exact_event_identities_as_frozen_probe']=True
        # Retain the support fingerprint, drop the per-trial 7-site ID streams from the evidence surface.
        result.pop('events',None)
    elif name=='phenotypes':
        from src.analysis.observation_token_phenotypes import run_phenotypes
        result=run_phenotypes(frames,waves,routers)
    elif name=='dynamics':
        from src.analysis.observation_token_dynamics import run_dynamics
        atlas=pd.read_csv(ROOT/c['parent']/'atlas.csv')
        t=[run_dynamics(frames[m,'train'],frames[m,'validation'],atlas,m) for m in ('eeg','hb')]
        result={k:pd.concat([u[k] for u in t],ignore_index=True) for k in t[0]}
    elif name in ('coupling_primary','coupling_four'):
        from src.analysis.observation_token_coupling import run_coupling
        def select(m,s):
            f=frames[m,s];return f[f.dataset.isin(c['coupling_primary_datasets'])].copy() if name=='coupling_primary' else f
        result=run_coupling(select('eeg','train'),select('hb','train'),select('eeg','validation'),select('hb','validation'))
    else: raise ValueError(name)
    paths=store(out,name,result);res=dict(status='completed',part=name,files=paths,elapsed_s=time.perf_counter()-started,finished=now())
    save(out/f'{name}_completed.json',res);print(name,'completed in',round(res['elapsed_s'],2),'s',flush=True);return res


def preflight(out,c):
    if (out/'preflight.json').exists(): raise FileExistsError('do not replace retained preflight')
    if not all(p.is_file() for p in SOURCES): raise FileNotFoundError('incomplete source/tests')
    code={str(p.relative_to(ROOT)):digest(p) for p in SOURCES};save(out/'resolved_config.json',c);save(out/'source_manifest.json',code)
    # These unit/integration tests use only generated arrays; never fall through if one fails.
    with (out/'synthetic_tests.log').open('x') as log:
        proc=subprocess.run([sys.executable,'-m','pytest','-q',*TESTS],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode: raise RuntimeError('synthetic/contract tests failed; see synthetic_tests.log')
    from src.analysis.observation_word_limits import frozen_word_collisions
    routers,_=load_routers(ROOT/c['parent'],ROOT/c['release'],c['identity'])
    save(out/'collisions.json',frozen_word_collisions(routers))
    save(out/'preflight.json',dict(status='passed',read_measured_arrays=False,tests=TESTS,finished=now(),source_manifest='source_manifest.json',frozen_dictionary_witness='collisions.json'))


def screen(out,c):
    if (out/'controller.json').exists(): raise FileExistsError('use a fresh dated attempt; never overwrite a running/failed/completed screen')
    if (js(out/'preflight.json')['status']!='passed' or js(out/'resolved_config.json')!=c
            or js(out/'source_manifest.json')!={str(p.relative_to(ROOT)):digest(p) for p in SOURCES}):
        raise ValueError('run data-free preflight with this source identity first')
    t=time.perf_counter();p=ROOT/c['parent'];r=ROOT/c['release']
    save(out/'controller.json',dict(status='running',start=now(),pid=os.getpid(),stage='frozen_panel'))
    try:
        routers,blind=load_routers(p,r,c['identity']);frames,waves,audit=replay(p,c['identity'],routers,blind,c['resources']['io_workers'])
        totals={m:len(frames[m,'validation']) for m in ('eeg','hb')}
        if totals!=c['expected_validation_cells']: raise ValueError('frozen validation denominator mismatch')
        old_atlas=pd.read_csv(p/'atlas.csv')
        for (m,s),f in frames.items():
            target=old_atlas[(old_atlas.modality==m)&(old_atlas.split==s)].groupby('pattern_id').n_patches.first().sort_index().to_numpy()
            if not np.array_equal(target,np.bincount(f.pattern,minlength=16)): raise ValueError('per-ID occupancy differs from released train/validation atlas')
        audit['reproduced_all_64_split_modality_word_support_counts']=True
        pop=lambda s:set(map(tuple,frames['eeg',s][['dataset','subject']].drop_duplicates().values))
        if len(pop('train'))!=60 or len(pop('validation'))!=20 or pop('train')&pop('validation'): raise ValueError('population mismatch')
        save(out/'panel_replay.json',dict(audit,validation_cells=totals,status='passed',seconds=time.perf_counter()-t))
        save(out/'controller.json',dict(status='running',stage='five_prespecified_readouts',since=now(),pid=os.getpid()))
        print('frozen panel and native clocks passed',totals,'elapsed',round(time.perf_counter()-t,1),'s',flush=True)
        # No persisted copy of the measured X or event inputs: pipe pickled arrays only to the phenotype worker.
        tasks=[joblib.delayed(computation)(n,out,c,frames if n!='events' else None,waves if n=='phenotypes' else None,routers if n in ('events','phenotypes') else None) for n in ('events','phenotypes','dynamics','coupling_primary','coupling_four')]
        with joblib.parallel_config(backend='loky',inner_max_num_threads=1):
            results=joblib.Parallel(n_jobs=c['resources']['outer_processes'],pre_dispatch=2,batch_size=1,max_nbytes=None)(tasks)
        summary=dict(status='completed',experiment_id=c['experiment_id'],started_from=c['parent_git'],scope='previously examined public development panel; no test access',
                     validation_cells=totals,training_subjects=60,validation_subjects=20,parts=results,source='source_manifest.json',protocol='resolved_config.json',
                     seconds=time.perf_counter()-t,controller_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished=now())
        save(out/'summary.json',summary);save(out/'controller.json',dict(status='completed',finished=now(),summary='summary.json'))
    except BaseException as ex:
        save(out/'controller.json',dict(status='failed',finished=now(),reason=str(ex),trace=traceback.format_exc()));raise


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--config',type=Path,default=CONFIG);q.add_argument('--stage',choices=['preflight','screen'],required=True);q.add_argument('--output',type=Path,required=True);a=q.parse_args()
    c=config(a.config);out=a.output.absolute();
    if out.parent!=SUITE or out!=out.resolve() or '_relations_v' not in out.name: raise ValueError('new named run under the observation suite required')
    out.mkdir(exist_ok=True)
    with threadpool_limits(limits=1): (preflight if a.stage=='preflight' else screen)(out,c)


if __name__=='__main__': main()
