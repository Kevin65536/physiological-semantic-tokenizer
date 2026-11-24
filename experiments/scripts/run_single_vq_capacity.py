"""Run the matched, development-only single-level VQ capacity sequence."""
import argparse
from collections import defaultdict, OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from run_dual_component_tokenizer import (
    _base_training_model, owner_lock, read_json, train_base_tokenizer, write_json,
)
from retrain_joint_qc_tokenizer import configuration, OLD, OUT as QC, ROOT
from evaluate_full_qc_retrain import summarize
from apply_joint_tokenizer import checked_path, load_branch_pair
from evaluate_joint_tokenizer import input_scales
from src.analysis.dual_component_tokenizer import base_training_aggregate, joint_reconstruction_metrics
from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_batch, base_tokenizer_joint_eeg
from src.utils.io import save_npz_atomic

CONTRACT = ROOT/'experiments/configs/physiology_semantic_tokenizer/single_vq_capacity_v1.yaml'
PLAN = yaml.safe_load(CONTRACT.read_text())
OUT = ROOT/PLAN['run']
PREVIOUS = QC/'evaluation/previous'
SEARCH = OLD/'component_source_transfer/montage_word_search_v1/portable_fold0_v1/bundle/bundle_waveform.json'
G1 = OLD.parent/'20261122_unified_base_g1_v1/models'
DATASETS = ('eeg_fnirs_single_trial','refed','simultaneous_eeg_nirs','visual_cognitive_motivation')
# Forked read-only G0 manifest; each child opens its own maps/model.
PANEL = {}


def candidate(mod, dim, size, seed):
    return f'{mod}_d{dim}_k{size}_s{seed}'


def dimensions(mod):
    return PLAN[mod]['dimensions']


def resolved(mod, dim, seed):
    if mod not in ('eeg','hb') or dim not in dimensions(mod) or seed not in PLAN['seeds']:
        raise ValueError('unknown capacity arm')
    cfg = configuration()
    cfg['experiment_id'] = PLAN['experiment_id']
    cfg['training'].update(PLAN['training'], seed=seed)
    cfg['capacity'] = dict(sizes=PLAN[mod]['sizes'], uniform_count_fraction=PLAN['uniform_count_fraction'])
    cfg['architectures']['waveform'].update(width=PLAN['width'],latent_dim=dim)
    cfg['model']['geometry_each_forward'] = False
    cfg['evaluation'].update(steps=[], roles=['train','validation'], checkpoint_selection='external fixed terminal checkpoint')
    cfg['resources'].update(gpu_memory_limit_gib=15,torch_threads=1)
    return cfg


def model_task(mod, size):
    return dict(modality=mod, architecture='waveform',candidate=('continuous' if size==0 else f'vq{size}'))


def train_folder(mod,dim,seed):
    return OUT/'training'/f'{mod}_d{dim}_s{seed}'


def train(mod, dim, size, seed, *, warmup_only=False):
    cfg = resolved(mod,dim,seed)
    if size not in (0,*cfg['capacity']['sizes']):
        raise ValueError('unregistered codebook size')
    base=train_folder(mod,dim,seed)
    with owner_lock(base,'warmup',blocking=True):
        write_json(base/'resolved_config.json',cfg)
        warm=dict(model_task(mod,size),candidate='warmup')
        if not (base/'models'/f'{mod}__waveform__warmup/summary.json').exists():
            train_base_tokenizer(cfg,ROOT,base,warm,'cuda:0')
    if not warmup_only:
        train_base_tokenizer(cfg,ROOT,base,model_task(mod,size),'cuda:0')


def prepare_panel(*, recorded_root=None):
    cfg=configuration()
    manifest,scales,_,_=base_tokenizer_training_data(cfg,ROOT,recorded_root=recorded_root)
    if manifest.get('window_exclusions')!=PLAN['support']['qc']:
        raise ValueError('QC identity changed')
    rows=manifest['rows']
    subjects={s:set() for s in ('train','validation')}
    allowed=defaultdict(set)
    for i,row in enumerate(rows):
        for g in row['inputs']['score']:
            if g['positions']:
                subjects[row['split']].add((row['dataset'],row['subject']))
                allowed[row['modality'],row['split'],row['dataset']].update(
                    (i,g['start_s'],p) for p in g['positions'])
    if (len(subjects['train']),len(subjects['validation']))!=(60,16) or subjects['train']&subjects['validation']:
        raise ValueError('subject boundary drift')
    examples={}
    for (m,role,ds),valid in allowed.items():
        with np.load(PREVIOUS/f'{m}_{ds}_{role}.npz') as z:
            ex=z['examples']
        actual=set(map(tuple,ex))
        if not actual<=valid or (role=='validation' and (actual!=valid or len(actual)!=len(ex))):
            raise ValueError('frozen panel no longer matches G0')
        examples[m,role,ds]=ex
    for m in ('eeg','hb'):
        count=sum(len(examples[m,'validation',ds]) for ds in DATASETS)
        if count!=PLAN['support'][m+'_positions']:
            raise ValueError('score denominator changed')
    PANEL.update(rows=rows,scales=scales,examples=examples)


def neural_predict(model, batch):
    with torch.inference_mode():
        result=model(torch.as_tensor(batch['x'],dtype=torch.float32))
        pred=model.decode_ids(result['ids']) if model.hard else result
        if model.hard:
            for b in 'SO':
                torch.testing.assert_close(pred[b],result[b],rtol=0,atol=0)
        return ({k:pred[k].numpy() for k in ('S','O','sum')},
                {k:v.numpy() for k,v in result['ids'].items()})


def load_scoring_model(arm,mod,dim,size,seed):
    if arm=='montage':
        item=read_json(SEARCH)['eeg']
        return load_branch_pair(item['kind'],checked_path(SEARCH,item),item['encoder_variant'])
    task=model_task(mod,size)
    if arm=='original':
        cfg=configuration()
        path=G1/f'{mod}__waveform__vq128/checkpoint_step_10000.pt'
    else:
        cfg=resolved(mod,dim,seed)
        base=train_folder(mod,dim,seed)/'models'/'__'.join(task.values())
        if read_json(base/'summary.json')['status']!='completed':
            raise ValueError('unfinished model')
        path=base/'checkpoint.pt'
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['step']!=10000 or state['signature']['task']!=task:
        raise ValueError('wrong checkpoint or training budget')
    model=_base_training_model(cfg,task).eval()
    model.load_state_dict(state['model_state'])
    return model


def score_dataset(args):
    arm,mod,dim,size,seed,ds,folder=args
    torch.set_num_threads(1)
    rows,scales=PANEL['rows'],PANEL['scales']
    model=load_scoring_model(arm,mod,dim,size,seed)
    cache=OrderedDict()
    for role in ('train','validation') if size else ('validation',):
        ex=PANEL['examples'][mod,role,ds]
        dest=folder/f'{mod}_{ds}_{role}.npz'
        if dest.exists():
            with np.load(dest) as z:
                np.testing.assert_array_equal(z['examples'],ex)
            continue
        if arm=='montage':
            groups=np.split(ex,np.flatnonzero(np.any(np.diff(ex[:,:2],axis=0),axis=1))+1)
        else:
            groups=[ex[i:i+64] for i in range(0,len(ex),64)]
        kept=defaultdict(list)
        began=time.monotonic()
        for i,take in enumerate(groups):
            batch=base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=64)
            if arm=='montage':
                row,start=take[0,:2]
                names=model.metadata['positions'][ds]
                order={name.upper():j for j,name in enumerate(names)}
                pick=[order[rows[r]['positions'][p].upper()] for r,_,p in take]
                crop=dict(row=row,start_s=start,positions=take[:,2].tolist())
                arrays,seen,_=base_tokenizer_joint_eeg(rows,crop,scales,names,cache,allowed_splits=(role,))
                ids=model.encode(arrays['x'].T.reshape(1,len(names),15,400),ds,seen[None])
                pred=({k:v[0,pick] for k,v in model.decode_ids(ids,ds).items() if k in ('S','O','sum')}
                      if role=='validation' else {})
                words={k:ids[k][0,pick] for k in 'SO'}
            else:
                pred,words=neural_predict(model,batch)
            for b,v in words.items():
                if v.shape!=(len(take),15) or v.dtype.kind not in 'iu' or v.min()<0 or v.max()>=size:
                    raise ValueError('invalid integer sequence')
                kept[b].append(v.astype('uint16'))
            if role=='validation':
                if any(not np.isfinite(v).all() for v in pred.values()):
                    raise ValueError('nonfinite prediction')
                metrics=joint_reconstruction_metrics(pred,batch,1 if mod=='eeg' else 2,input_scales(rows,take,scales))
                for k,v in metrics['metrics'].items():
                    kept[k].append(v)
            if i%10==0 or i+1==len(groups):
                write_json(dest.with_suffix('.progress.json'),dict(done=i+1,total=len(groups),seconds=time.monotonic()-began))
        save_npz_atomic(dest,examples=ex,**{k:np.concatenate(v) for k,v in kept.items()})
    if arm=='candidate' and size and ds==DATASETS[0]:
        write_json(folder/'code_geometry.json',{b:{k:float(v) for k,v in model.branches[b]['quantizer'].geometry().items()} for b in 'SO'})


def evaluate(arm,mod,dim,size,seed):
    name=candidate(mod,dim,size,seed) if arm=='candidate' else f'{arm}_{mod}'
    folder=OUT/'evaluation'/name
    with owner_lock(folder):
        finish=folder/'completion.json'
        if finish.exists():
            return
        prepare_panel()
        args=[(arm,mod,dim,size,seed,ds,folder) for ds in DATASETS]
        with ProcessPoolExecutor(4,mp_context=multiprocessing.get_context('fork')) as pool:
            list(pool.map(score_dataset,args))
        if size:
            state=dict(PANEL,out=folder,datasets=DATASETS)
            summarize(mod,state,dict(S=size,O=size))
        else:
            metrics=defaultdict(list)
            examples=[]
            for ds in DATASETS:
                with np.load(folder/f'{mod}_{ds}_validation.npz') as z:
                    examples.append(z['examples'])
                    for k in z.files:
                        if k!='examples':
                            metrics[k].append(z[k])
            stats=base_training_aggregate({k:np.concatenate(v) for k,v in metrics.items()},np.concatenate(examples),PANEL['rows'])
            write_json(folder/f'{mod}_summary.json',stats)
        stats=read_json(folder/f'{mod}_summary.json')
        if stats['inputs']!=PLAN['support'][mod+'_positions'] or stats['subjects']!=16:
            raise ValueError('incomplete development summary')
        write_json(finish,dict(status='completed',modality=mod,model=name,subjects=16,
                               positions=stats['inputs'],macro_nmse=stats['selection_score']))


def call(action,arm,mod,dim,size,seed,uuid=''):
    argv=[sys.executable,str(Path(__file__).resolve()),action,arm,mod,str(dim),str(size),str(seed)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid)
    folder=OUT/'logs'
    folder.mkdir(exist_ok=True,parents=True)
    with (folder/f'{action}_{arm}_{candidate(mod,dim,size,seed)}.log').open('a') as log:
        subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def summary(mod,dim,size,seed):
    return read_json(OUT/'evaluation'/candidate(mod,dim,size,seed)/f'{mod}_summary.json')


def training_file(action, mod, dim, size, seed, filename):
    kind = 'warmup' if action == 'warmup' else model_task(mod, size)['candidate']
    return train_folder(mod, dim, seed)/'models'/f'{mod}__waveform__{kind}'/filename


def controller():
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(set(devices)) != 2 or not all(d.startswith('GPU-') for d in devices):
        raise ValueError('two PCI-selected GPU UUIDs required')
    policy = os.environ.get('VQ_RESOURCE_POLICY')
    slots = (read_json(policy)['slots'] if policy else
             {mode: dict.fromkeys(devices, 1) for mode in ('continuous', 'hard')})
    for mode in ('continuous', 'hard'):
        if set(slots[mode]) != set(devices) or any(n not in (1, 2) for n in slots[mode].values()):
            raise ValueError('invalid measured GPU concurrency')
    seed, other = PLAN['seeds']
    k0 = PLAN['eeg']['dimension_screen_K']
    eeg_screen = [('eeg', d, k0, seed) for d in dimensions('eeg')]
    hb_screen = [('hb', d, k, seed) for d in dimensions('hb') for k in PLAN['hb']['sizes']]
    controls = [(m, d, 0, seed) for m in ('eeg', 'hb') for d in dimensions(m)]
    planned = dict.fromkeys(eeg_screen + hb_screen + controls)
    references = [('original', m, 64, 128, seed) for m in ('eeg', 'hb')]
    references.append(('montage', 'eeg', 64, 15351, seed))
    chosen, second, replicas = None, [], {}
    running = {}

    def score_done(arm, mod, dim, size, seed):
        name = candidate(mod, dim, size, seed) if arm == 'candidate' else f'{arm}_{mod}'
        return (OUT/'evaluation'/name/'completion.json').is_file()

    def select_followups():
        nonlocal chosen, second
        if chosen is None and all(score_done('candidate', *a) for a in eeg_screen):
            chosen = min(dimensions('eeg'), key=lambda d: (summary('eeg', d, k0, seed)['selection_score'], d))
            second = [('eeg', chosen, k, seed) for k in PLAN['eeg']['sizes'] if k != k0]
            planned.update(dict.fromkeys(second))
        for mod, arms in [('eeg', eeg_screen + second), ('hb', hb_screen)]:
            if mod in replicas or (mod == 'eeg' and chosen is None):
                continue
            if not score_done('original', mod, 64, 128, seed) or not all(score_done('candidate', *a) for a in arms):
                continue
            ref = read_json(OUT/'evaluation'/f'original_{mod}'/f'{mod}_summary.json')
            eligible = []
            for a in arms:
                score = summary(*a)
                if score['selection_score'] < ref['selection_score'] and all(
                    np.mean(list(score[b+'_single_word_gain'].values())) > 0 for b in 'SO'):
                    eligible.append((score['selection_score'], a[2], a[1]))
            replicas[mod] = None
            if eligible:
                _, k, d = min(eligible)
                replicas[mod] = (mod, d, k, other)
                planned[replicas[mod]] = None

    with owner_lock(OUT), ThreadPoolExecutor(7) as pool:
        write_json(OUT/'baseline_sources.json',dict(previous=str(PREVIOUS),retrained=str(QC/'evaluation/retrained'),
                   montage_hb='identical previous Hb111',planned_positions=PLAN['support']))
        while True:
            for f in list(running):
                if f.done():
                    f.result()
                    del running[f]
            select_followups()
            jobs = []
            for ref in references:
                if not score_done(*ref):
                    jobs.append(('score', *ref))
            for a in planned:
                if score_done('candidate', *a):
                    continue
                mod, dim, size, s = a
                if training_file('train', *a, 'summary.json').is_file():
                    jobs.append(('score', 'candidate', *a))
                elif training_file('warmup', *a, 'summary.json').is_file():
                    jobs.append(('train', 'candidate', *a))
                else:
                    jobs.append(('warmup', 'candidate', mod, dim, 0, s))
            if not jobs and not running:
                break
            # Resume partial checkpoints first, then start independent arms.
            jobs = sorted(dict.fromkeys(jobs), key=lambda j:
                          0 if j[0] != 'score' and training_file(j[0], *j[2:], 'checkpoint.pt').is_file() else 1)
            for job in jobs:
                active = list(running.values())
                if any(task == job for task, _ in active):
                    continue
                device = ''
                if job[0] == 'score':
                    if sum(not dev for _, dev in active) >= 3:
                        continue
                else:
                    mode = 'hard' if job[4] else 'continuous'
                    free = []
                    for gpu in devices:
                        co = [task for task, dev in active if dev == gpu]
                        caps = [slots['hard' if t[4] else 'continuous'][gpu] for t in co]
                        if len(co) < min([slots[mode][gpu], *caps]):
                            free.append((len(co), gpu))
                    if not free:
                        continue
                    device = min(free)[1]
                running[pool.submit(call, *job, device)] = (job, device)
            write_json(OUT/'stage.json', dict(stage='dependency_queue', updated=time.time(),
                       jobs=list(planned), active=[dict(job=j, gpu=g) for j,g in running.values()],
                       eeg_dimension=chosen, replication=replicas, slots=slots))
            if not running:
                raise RuntimeError('unfinished jobs without a resource or dependency')
            wait(running, timeout=30, return_when=FIRST_COMPLETED)
        repeated = [replicas[m] for m in ('eeg','hb') if replicas[m] is not None]
        write_json(OUT/'selection.json',dict(eeg_dimension=chosen,replication=repeated))
        arms = eeg_screen + hb_screen + second + repeated
        write_json(OUT/'summary.json',dict(status='completed',selection='ordinary development',eeg_dimension=chosen,
                   models={candidate(*a):summary(*a)['selection_score'] for a in arms},replicated=repeated,
                   scientific_verdict='requires joint waveform, word and reproducibility review'))
        write_json(OUT/'stage.json',dict(stage='completed',finished=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('controller','warmup','train','score'))
    parser.add_argument('arm',nargs='?',choices=('original','montage','candidate'))
    parser.add_argument('mod',nargs='?',choices=('eeg','hb'))
    for key in ('dim','size','seed'):
        parser.add_argument(key,nargs='?',type=int)
    a=parser.parse_args()
    if a.action=='controller':
        try:
            controller()
        except Exception:
            import traceback
            write_json(OUT/f'controller_failure_{int(time.time())}.json',dict(error=traceback.format_exc()))
            raise
    elif a.action in ('train','warmup'):
        train(a.mod,a.dim,a.size,a.seed,warmup_only=a.action=='warmup')
    else:
        evaluate(a.arm,a.mod,a.dim,a.size,a.seed)


if __name__=='__main__':
    main()
