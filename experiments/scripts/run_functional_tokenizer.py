#!/usr/bin/env python3
"""TK0--TK5 public development suite; one resumable supervised entry point."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE))
import numpy as np
import yaml

from src.data.functional_tokenizer import (
    inventory,prepare_native_record,training_coordinates,label_blind_donors,identity_digest,
)
from src.teachers.functional_teacher import (
    produce_teacher,alternative_teacher,match_smoothing,apply_smoothing,synthetic_interventions,
)
from src.inference.ssm_output_audit import hb_coordinates
from src.analysis.functional_tokenizer import (
    read_hard_tokens,join_export,trial_features,classification_probe,ridge_readout,
    code_health,align_ids,paired_interval,compare_arms,coupling_features,frozen_prototypes,
    session_probe_split,branch_increments,
)

DEFAULT=SOURCE/'experiments/configs/physiology_semantic_tokenizer/functional_tokenizer_v1.yaml'


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)): return [clean(v) for v in value]
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def read(path): return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path,allow_pickle=False) as x: return {k:x[k].copy() for k in x.files}


def save_arrays(path,**values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f: np.savez_compressed(f,**values)
    temp.replace(path)


def config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    expected=['C0','CT','Q0','QT','C-wide','C-smooth','C-shuffle','C-stop']
    if (cfg['schema']!='functional_tokenizer_v1' or cfg['experiment_id']!='TK-FUNCTIONAL-v1'
        or cfg['data']['protected_data']!='forbidden' or cfg['data']['labels_as_training_inputs']
        or cfg['data']['datasets']!=['eeg_fnirs_single_trial','simultaneous_eeg_nirs']
        or cfg['training']['arms']!=expected or cfg['tensor']['teacher_steps']!=360
        or cfg['tensor']['student_steps']!=[120,200] or cfg['tensor']['patches']!=10
        or cfg['model']['codebook_size']!=128 or not cfg['model']['independent_encoders']
        or cfg['resources']['numerical_threads']!=1 or cfg['coupling']['training_loss']!='none'):
        raise ValueError('invalid versioned scope, hard path, tensor or training contract')
    tensor_assertions=dict(teacher_seconds=90,teacher_offset_s=-30,dt_s=.25,student_seconds=20,
        patch_seconds=2,eeg_rate_hz=200,hb_reference_support_event_s=[-30,-25])
    if any(cfg['tensor'][k]!=v for k,v in tensor_assertions.items()):
        raise ValueError('native clocks, prefix, patch or continuous-context mismatch')
    if (cfg['data']['eeg_channels_per_region']!=6
        or cfg['data']['regions']!=['prefrontal','left_motor','right_motor']
        or cfg['data']['train_fraction']!=.6 or cfg['data']['validation_fraction']!=.2
        or cfg['teacher']['state_arm']!='chain_matched'
        or cfg['teacher']['target']!='physical_prediction_not_total_fit_or_common_component'
        or cfg['coupling']['hidden_student_steps']!=[32,48]
        or cfg['model']['quantizer']!='EMA_hard_forward_ST_backward'):
        raise ValueError('unregistered split, observation target, state or hard-VQ contract')
    if (min(cfg['training']['steps'],cfg['training']['batch_size'],cfg['training']['checkpoint_every'])<1
        or any(not 0<f<=1 for f in cfg['evaluation']['label_fractions'])
        or any(abs(s)>=5 for s in cfg['coupling']['shifts_in_patches'])):
        raise ValueError('invalid exposure, label budget or common shift support')
    if len(set(cfg['mechanism']['streams'].values()))!=3: raise ValueError('independent synthetic partitions required')
    if cfg['training']['seeds']!=[17,29,43]: raise ValueError('three registered optimization seeds required')
    return cfg


def teacher_configs(cfg):
    ssm,physical=tuple(yaml.safe_load((SOURCE/cfg['teacher'][key]).read_text()) for key in ('contract','physical_config'))
    if (ssm['schema']!='ssm_output_support_v1' or physical['schema']!='shared_driver_physiology_semantics_v1'
        or any(physical['fixed'][k]!=v for k,v in dict(tau=2.,kappa=.64,neurovascular_gain=1.).items())):
        raise ValueError('frozen six-state reference teacher version changed')
    return ssm,physical


def freeze_run(cfg,out,project_root):
    out.mkdir(parents=True,exist_ok=True)
    contract=out/'resolved_config.json'
    resolved=dict(config=cfg,teacher_configs=teacher_configs(cfg),project_root=str(Path(project_root).resolve()))
    if (out/'launch.json').exists():
        source_root=read(out/'launch.json').get('source_root')
        if source_root and Path(source_root).resolve()!=SOURCE.resolve():
            raise ValueError('resume must use the recorded immutable source snapshot')
    if contract.exists():
        if read(contract)!=clean(resolved): raise ValueError('immutable run/config mismatch; use a new run')
    else: write(contract,resolved)
    return resolved


def prepare_worker(payload):
    project_root,cfg,out_string,rows=payload;out=Path(out_string)
    counts=Counter();t=time.monotonic()
    for row,a,error in prepare_native_record(project_root,cfg,rows):
        record=out/'tasks'/'prepare'/(row['id']+'.json')
        if error:
            write(record,dict(status='failed',reason=error,id=row['id']))
            counts['failed']+=1
        else:
            save_arrays(out/'prepared'/(row['id']+'.npz'),**a)
            write(record,dict(status='completed',id=row['id'],hidden_native_invariance=True,
                units=dict(eeg='uV',hb='relative_MBLL' if row['dataset']=='eeg_fnirs_single_trial' else 'released_mmol/L'),
                source_identity=row['source_identity']))
            counts['completed']+=1
    return dict(status='completed',counts=dict(counts),seconds=time.monotonic()-t,
        maximum_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)


def teacher_worker(payload):
    cfg,out_string,row,co,mode=payload;out=Path(out_string);t=time.monotonic()
    a=arrays(out/'prepared'/(row['id']+'.npz'))
    data,status=produce_teacher(a,co,*teacher_configs(cfg),mode=mode)
    save_arrays(out/'teacher'/mode/(row['id']+'.npz'),**data)
    return dict(status='completed' if status['converged'] else 'failed',fit=status,
        seconds=time.monotonic()-t,maximum_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)


def cpu_benchmark_worker(payload):
    cfg,parent_string,row,co=payload;parent=Path(parent_string);start=time.monotonic()
    _,status=produce_teacher(arrays(parent/'prepared'/(row['id']+'.npz')),co,*teacher_configs(cfg))
    return dict(status='completed',converged=status['converged'],seconds=time.monotonic()-start,
        maximum_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)


def cpu_benchmark(cfg,out,parent):
    """Read-only replay of the completed public pilot; no new raw reads."""
    parent=Path(parent).resolve()
    if parent.parent!=out.parent or not read(parent/'preparation_summary.json')['pilot']:
        raise ValueError('explicit completed sibling pilot required')
    if read(parent/'resolved_config.json')['teacher_configs']!=clean(teacher_configs(cfg)):
        raise ValueError('pilot teacher version mismatch')
    rows=[r for r in read(parent/'inventory.json')['rows'] if (parent/'tasks'/'teacher_joint'/(r['id']+'.json')).exists()]
    co=read(parent/'coordinates.json');scores=[]
    for workers in (8,16,24,40):
        jobs=[(f'{i:03d}',(cfg,str(parent),rows[i%len(rows)],co[rows[i%len(rows)]['coordinate']])) for i in range(80)]
        start=time.monotonic();results=bounded_tasks(out,'cpu_'+str(workers),jobs,cpu_benchmark_worker,workers)
        elapsed=time.monotonic()-start
        scores.append(dict(workers=workers,wall_seconds=elapsed,fits_per_second=len(jobs)/elapsed,
            maximum_worker_RSS_MiB=max(r.get('maximum_RSS_MiB',0) for r in results.values()),
            median_fit_seconds=float(np.median([r['seconds'] for r in results.values() if 'seconds' in r])),
            failed=sum(r['status']!='completed' for r in results.values())))
        write(out/'cpu_benchmark.json',dict(source=str(parent),scope='throughput_replay_not_scientific_replication',scores=scores))


def alternative_worker(payload):
    cfg,out_string,row,co=payload;out=Path(out_string)
    a=arrays(out/'prepared'/(row['id']+'.npz'))
    target,status=alternative_teacher(a,co,*teacher_configs(cfg),cfg['mechanism']['alternative_offsets'],
        cfg['mechanism']['alternative_objective_fraction'])
    if target is not None: save_arrays(out/'teacher'/'alternative'/(row['id']+'.npz'),physical=target)
    return status


def bounded_tasks(out,stage,items,worker,workers):
    """Atomic task outcomes, bounded futures and fail-preserving resume."""
    pending=[(key,payload) for key,payload in items if not (out/'tasks'/stage/(key+'.json')).exists()]
    started=time.monotonic();done=len(items)-len(pending)
    progress=out/'progress'/(stage+'.json')
    write(progress,dict(stage=stage,total=len(items),completed=done,started_at=now()))
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers,mp_context=context) as pool:
        queue=iter(pending);futures={}
        def fill():
            while len(futures)<workers*2:
                try: key,payload=next(queue)
                except StopIteration: break
                futures[pool.submit(worker,payload)]=key
        fill()
        while futures:
            ready,_=wait(futures,timeout=30,return_when=FIRST_COMPLETED)
            for future in ready:
                key=futures.pop(future)
                try: result=future.result()
                except Exception as exc:
                    result=dict(status='failed',reason=str(exc),traceback=traceback.format_exc())
                write(out/'tasks'/stage/(key+'.json'),result);done+=1
                print(json.dumps(dict(stage=stage,done=done,total=len(items),key=key,status=result.get('status'))),flush=True)
            fill()
            elapsed=time.monotonic()-started
            processed=done-(len(items)-len(pending))
            eta=(len(items)-done)*elapsed/processed if processed else None
            write(progress,dict(stage=stage,total=len(items),completed=done,elapsed_s=elapsed,
                eta_seconds=eta,updated_at=now(),workers=workers))
    return {key:read(out/'tasks'/stage/(key+'.json')) for key,_ in items}


def now(): return datetime.now(timezone.utc).isoformat()


def software_synthetic(cfg,out):
    """No measured data. Exercises physical integration, all arms and export."""
    import torch
    from src.tokenizers.functional_tokenizer import FunctionalTokenizer
    torch.set_num_threads(1)
    checks={}
    examples=synthetic_interventions(cfg,teacher_configs(cfg)[1],cfg['mechanism']['streams']['train'])
    checks['alias_eeg_identical']=np.array_equal(examples['base']['eeg'],examples['exact_alias']['eeg'])
    checks['alias_hb_identical']=np.array_equal(examples['base']['hb'],examples['exact_alias']['hb'])
    checks['alias_truth_different']=not np.array_equal(examples['base']['truth'],examples['exact_alias']['truth'])
    e=torch.tensor(np.stack([v['eeg'].reshape(10,2400) for v in examples.values()]))
    h=torch.tensor(np.stack([v['hb'].reshape(10,16)/.025 for v in examples.values()]))
    target={'eeg':torch.zeros(len(e),10,8),'hb':torch.zeros(len(e),10,16)}
    losses={}
    for arm in cfg['training']['arms']:
        torch.manual_seed(17);model=FunctionalTokenizer(cfg['model'],arm)
        optim=torch.optim.AdamW(model.parameters(),lr=.001)
        values=[]
        for _ in range(3):
            optim.zero_grad();output=model(e,h);loss=model.objective(output,e,h,target,cfg['training'])['total']
            loss.backward();optim.step();values.append(float(loss.detach()))
        checks[arm+'_finite']=bool(np.isfinite(values).all());losses[arm]=values
        model.eval()
        with torch.no_grad():
            output=model(e,h)
            if arm in ('Q0','QT'):
                hard={f'ids_{m}':output[m]['hard_ids'].numpy() for m in ('eeg','hb')}
                hard.update({f'codebook_{m}':getattr(model,m).quantizer.codebook.numpy() for m in ('eeg','hb')})
                path=out/'synthetic'/(arm+'_hard.npz')
                save_arrays(path,**hard,identities=np.array([str(i) for i in range(len(e))]))
                _,lookup=read_hard_tokens(path)
                for m in ('eeg','hb'):
                    checks[arm+'_'+m+'_hard_lookup']=np.allclose(lookup[m],output[m]['semantic'].numpy(),atol=1e-6)
        # A pure reconstruction backward must leave the C-stop source untouched.
        if arm=='C-stop':
            model.zero_grad();result=model(e,h)
            sum(result[m]['reconstruction'].square().mean() for m in result).backward()
            checks['stop_gradient']=all(p.grad is None or torch.count_nonzero(p.grad)==0
                for branch in (model.eeg,model.hb) for p in branch.source_encoder.parameters())
    mask=np.ones((len(h),80,2),bool);mask[:,32:48]=False
    visible,_=coupling_features(h.numpy().reshape(-1,80,2),None)
    changed=h.numpy().reshape(-1,80,2).copy();changed[:,32:48]=np.nan
    check,_=coupling_features(changed,None)
    checks['hidden_hb_poison']=np.array_equal(visible,check)
    result=dict(status='completed' if all(checks.values()) else 'failed',checks=checks,
        toy_optimization_losses=losses,interpretation='software_synthetic_smoke_not_measured_scientific_evidence')
    write(out/'synthetic_checks.json',result)
    if not all(checks.values()): raise AssertionError('synthetic software check failed')
    return result


def prepare(cfg,out,root,workers,pilot=False):
    if read(out/'synthetic_checks.json')['status']!='completed': raise ValueError('synthetic checks must precede measured reads')
    path=out/'inventory.json'
    if not path.exists(): write(path,inventory(root,cfg))
    plan=read(path);rows=plan['rows']
    if pilot:
        selected=[]
        for ds in cfg['data']['datasets']:
            for split in ('train','validation','test'):
                candidates=[r for r in rows if r['dataset']==ds and r['split']==split]
                trial=candidates[0]['trial_id']
                selected.extend(r for r in candidates if r['trial_id']==trial)
        rows=selected
    groups=defaultdict(list)
    for r in rows: groups[r['join_key']].append(r)
    items=[(identity_digest(k)[:20],(str(root),cfg,str(out),v)) for k,v in groups.items()]
    bounded_tasks(out,'prepare_records',items,prepare_worker,min(workers,len(items)))
    good=[]
    for r in rows:
        p=out/'tasks'/'prepare'/(r['id']+'.json')
        if p.exists() and read(p)['status']=='completed': good.append(r)
    if not good: raise ValueError('no valid native windows; inspect prepare failures')
    co_path=out/'coordinates.json'
    if co_path.exists(): coordinates=read(co_path)
    else:
        coordinates=training_coordinates(good,lambda r:arrays(out/'prepared'/(r['id']+'.npz')))
        write(co_path,coordinates)
    items=[(r['id'],(cfg,str(out),r,coordinates[r['coordinate']],'joint')) for r in good]
    timings=bounded_tasks(out,'teacher_joint',items,teacher_worker,workers)
    successful=[r for r in good if timings[r['id']]['status']=='completed']
    # Single-modality teachers serve TK0 only, on a deterministic small panel.
    audit=[]
    for ds in cfg['data']['datasets']:
        for split in ('train','validation','test'):
            candidates=[r for r in successful if r['dataset']==ds and r['split']==split]
            trialids=sorted({r['trial_id'] for r in candidates},key=lambda v:identity_digest(v))[:12]
            audit.extend(r for r in candidates if r['trial_id'] in trialids)
    for mode in ('eeg_only','hb_only'):
        bounded_tasks(out,'teacher_'+mode,[(r['id'],(cfg,str(out),r,coordinates[r['coordinate']],mode)) for r in audit],teacher_worker,workers)
    write(out/'preparation_summary.json',dict(planned_windows=len(rows),native_success=len(good),
        teacher_success=len(successful),teacher_failed=len(good)-len(successful),
        native_failed=len(rows)-len(good),pilot=pilot,
        fit_seconds=[r['seconds'] for r in timings.values() if 'seconds' in r],
        maximum_worker_RSS_MiB=max([r.get('maximum_RSS_MiB',0) for r in timings.values()]+[0])))
    return successful


def pack(cfg,out):
    plan=read(out/'inventory.json');coordinates=read(out/'coordinates.json')
    bytrial=defaultdict(list)
    for r in plan['rows']: bytrial[r['trial_id']].append(r)
    rows=[];failed=[]
    for trial,rs in sorted(bytrial.items()):
        if all((out/'tasks'/'teacher_joint'/(r['id']+'.json')).exists() and
               read(out/'tasks'/'teacher_joint'/(r['id']+'.json'))['status']=='completed' for r in rs):
            rows.extend(rs)
        else: failed.append(trial)
    if not rows: raise ValueError('no complete matched trial population')
    tensors=defaultdict(list)
    for r in rows:
        a=arrays(out/'prepared'/(r['id']+'.npz'));teacher=arrays(out/'teacher'/'joint'/(r['id']+'.npz'))
        c=coordinates[r['coordinate']]
        eeg=(a['eeg']-c['eeg_mean'])/c['eeg_scale']
        hb=a['hb']*c['hb_factor']/.025
        physical=teacher['physical'];u=teacher['u']
        functional=np.column_stack((physical[:,0],hb_coordinates(physical[:,1:])[:,::-1]))/.025
        observed=np.column_stack((u[:,0],hb_coordinates(u[:,1:])[:,::-1]))/.025
        for key,value in dict(eeg=eeg.reshape(10,2400),hb=hb[120:200].reshape(10,16),
            functional=functional,observed=observed,u=u[120:200]/.025,
            physical=physical[120:200]/.025,remaining=teacher['remaining'][120:200]/.025,
            bands=a['features'][120:200]).items(): tensors[key].append(value.astype('float32'))
    tensors={k:np.stack(v) for k,v in tensors.items()}
    training=np.array([r['split']=='train' for r in rows])
    selections={};smooth=np.zeros_like(tensors['functional'])
    for key in sorted(coordinates):
        ix=np.array([r['coordinate']==key for r in rows]);fit=ix&training
        selections[key]=match_smoothing(tensors['observed'][fit],tensors['functional'][fit],cfg['teacher']['smooth_sigma_steps'])
        smooth[ix]=apply_smoothing(tensors['observed'][ix],selections[key])
    write(out/'smooth_coordinates.json',selections)
    # Teacher normalization is train-only, common to all supervision controls.
    target_mean=tensors['functional'][training,120:200].mean((0,1))
    target_scale=np.maximum(tensors['functional'][training,120:200].std((0,1)),1e-5)
    tensors['teacher']=((tensors.pop('functional')[:,120:200]-target_mean)/target_scale).astype('float32')
    tensors['smooth']=((smooth[:,120:200]-target_mean)/target_scale).astype('float32')
    tensors['donors']=label_blind_donors(rows,cfg['seed'])
    tensors.pop('observed')
    cache=out/'cache';cache.mkdir(exist_ok=True)
    for key,value in tensors.items():
        tmp=cache/(key+'.tmp')
        with tmp.open('wb') as f: np.save(f,value,allow_pickle=False)
        tmp.replace(cache/(key+'.npy'))
    manifest=dict(schema='functional_training_cache_v1',rows=rows,identity=identity_digest(rows),
        target_mean=target_mean,target_scale=target_scale,teacher_coordinate='EEG,X,T divided by .025 then training standardized',
        shapes={k:v.shape for k,v in tensors.items()},failed_trials=failed,planned_trials=len(bytrial),
        retained_trials=len(rows)//len(cfg['data']['regions']),
        failures='all regions and all arms share complete converged-teacher trial support; failed trials remain in denominator',
        labels_as_training_inputs=False)
    write(cache/'manifest.json',manifest)
    return manifest


def load_cache(out):
    manifest=read(out/'cache'/'manifest.json')
    data={key:np.load(out/'cache'/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in manifest['shapes']}
    return manifest,data


def model_targets(values):
    import torch
    return dict(eeg=values[...,0].reshape(-1,10,8),hb=values[...,1:].reshape(-1,10,16))


def sampling_groups(rows):
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        if r['split']=='train': groups[r['dataset']].append(i)
    return [np.asarray(groups[k]) for k in sorted(groups)]


def export_model(model,cache,manifest,destination,device,batch_size=128):
    import torch
    model.eval();result=defaultdict(list);reconstruction=defaultdict(list);agreement=defaultdict(list)
    n=len(manifest['rows'])
    with torch.no_grad():
        for start in range(0,n,batch_size):
            sl=slice(start,min(start+batch_size,n))
            e=torch.as_tensor(np.array(cache['eeg'][sl]),device=device)
            h=torch.as_tensor(np.array(cache['hb'][sl]),device=device)
            output=model(e,h)
            teacher=model_targets(torch.as_tensor(np.array(cache['teacher'][sl]),device=device))
            for m,measurement in (('eeg',e),('hb',h)):
                r=output[m]
                for path in ('pre','soft','observation'):
                    result[m+'_'+path].append(r[path].cpu().numpy())
                if r['hard_ids'] is not None: result[m+'_ids'].append(r['hard_ids'].cpu().numpy())
                reconstruction[m].extend((r['reconstruction']-measurement).square().mean((1,2)).cpu().tolist())
                delta=r['teacher']-teacher[m]
                if m=='hb': delta=delta.reshape(-1,10,8,2)[...,0]
                agreement[m].extend(delta.square().mean((1,2)).cpu().tolist())
    result={k:np.concatenate(v) for k,v in result.items()}
    identities=np.array([r['id'] for r in manifest['rows']])
    continuous={k:v for k,v in result.items() if not k.endswith('_ids')}
    save_arrays(destination/'continuous.npz',**continuous,identities=identities)
    hard={}
    if model.arm in ('Q0','QT'):
        for m in ('eeg','hb'):
            hard['ids_'+m]=result[m+'_ids'].astype('int16')
            hard['codebook_'+m]=getattr(model,m).quantizer.codebook.cpu().numpy()
        save_arrays(destination/'hard.npz',**hard,identities=identities)
    rows=manifest['rows'];test=np.array([r['split']=='test' for r in rows])
    summary=dict(reconstruction_MSE=reconstruction,teacher_MSE=agreement,
        teacher_head_trained=model.arm not in ('C0','Q0'),
        bitrate=dict(semantic_bits_per_regional_patch=14 if hard else 2*16*32,
            semantic_serialized_scalar_bits_per_regional_patch=32 if hard else 2*16*32,
            observation_bits_per_regional_patch=2*32*32,
            regions=3,patches_per_trial=10,format_overhead_excluded=True,
            basis='nominal_log2_code_capacity; serialized_ID_arrays_are_int16; fixed_codebook_overhead_not_amortized'),
        code_health={m:code_health(hard['ids_'+m][test],hard['codebook_'+m]) for m in ('eeg','hb')} if hard else {},
        complete_finite=all(np.isfinite(v).all() for v in continuous.values()))
    if not summary['complete_finite']:
        raise FloatingPointError('nonfinite frozen representation export')
    write(destination/'metrics.json',summary)
    return summary


def train_one(cfg,out,arm,seed,device,*,panel=None,steps=None):
    import torch
    from src.tokenizers.functional_tokenizer import FunctionalTokenizer
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    if device.startswith('cuda'): torch.cuda.set_device(device);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    manifest,cache=load_cache(out);rows=manifest['rows']
    name=f'{arm}__seed{seed}' if panel is None else f'{arm}__seed{seed}__{panel}'
    dest=out/('models' if panel is None else 'sensitivity')/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists(): return read(dest/'summary.json')
    model=FunctionalTokenizer(cfg['model'],arm).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
    rng=np.random.default_rng(seed);begin=0;history=[]
    limit=steps or cfg['training']['steps']
    if panel:
        selection=read(out/'sensitivity_panel.json')
        indices=np.array([i for i,r in enumerate(rows) if r['id'] in selection['training_ids']])
        if len(indices)<2: raise ValueError('no valid paired sensitivity panel')
        alt=np.load(out/'cache'/'alternative.npy',allow_pickle=False,mmap_mode='r') if panel=='alternative' else cache['teacher']
    latest=dest/'resume.pt'
    if latest.exists():
        state=torch.load(latest,map_location=device,weights_only=False)
        if state['cache_identity']!=manifest['identity']: raise ValueError('resume cache identity changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state=state['rng'];torch.set_rng_state(state['torch_rng'].cpu())
        if device.startswith('cuda'): torch.cuda.set_rng_state(state['cuda_rng'].cpu(),device)
        begin=state['step'];history=state['history']
    started=time.monotonic();groups=sampling_groups(rows)
    for step in range(begin,limit):
        model.train()
        ix=(rng.choice(indices,cfg['training']['batch_size']) if panel else
            np.array([rng.choice(groups[int(rng.integers(len(groups)))]) for _ in range(cfg['training']['batch_size'])]))
        e=torch.tensor(np.array(cache['eeg'][ix]),device=device);h=torch.tensor(np.array(cache['hb'][ix]),device=device)
        source=alt if panel else cache['smooth'] if arm=='C-smooth' else cache['teacher']
        target_ix=np.asarray(cache['donors'][ix]) if arm=='C-shuffle' else ix
        target=model_targets(torch.tensor(np.array(source[target_ix]),device=device))
        optimizer.zero_grad(set_to_none=True)
        output=model(e,h);terms=model.objective(output,e,h,target,cfg['training'])
        if not torch.isfinite(terms['total']): raise FloatingPointError('nonfinite training objective')
        terms['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['gradient_clip'])
        optimizer.step()
        if (step+1)%50==0 or step==0:
            history.append(dict(step=step+1,**{k:float(v.detach()) for k,v in terms.items()},
                code_health={m:{k:float(v.detach()) for k,v in output[m]['health'].items()} for m in ('eeg','hb')}))
        if (step+1)%cfg['training']['checkpoint_every']==0 or step+1==limit:
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None,
                step=step+1,history=history,cache_identity=manifest['identity'],arm=arm,seed=seed)
            tmp=dest/'resume.tmp';torch.save(state,tmp);tmp.replace(latest)
            # One rotating recovery checkpoint, plus a final frozen export. Every
            # checkpoint creation exports all paths using its exact frozen book.
            export_model(model,cache,manifest,dest/'latest_export',device)
            write(dest/'progress.json',dict(step=step+1,total=limit,elapsed_s=time.monotonic()-started,
                steps_per_second=(step+1-begin)/max(time.monotonic()-started,1e-9),updated_at=now()))
            print(json.dumps(dict(model=name,step=step+1,total=limit,loss=float(terms['total'].detach()))),flush=True)
    final=dict(model=model.state_dict(),model_config=cfg['model'],arm=arm,seed=seed,
        cache_identity=manifest['identity'],steps=limit,source_sha=read(out/'launch.json').get('source_sha') if (out/'launch.json').exists() else None)
    tmp=dest/'checkpoint.tmp';torch.save(final,tmp);tmp.replace(dest/'checkpoint.pt')
    write(dest/'summary.json',dict(status='completed',arm=arm,seed=seed,steps=limit,
        seconds=time.monotonic()-started,history=history,cache_identity=manifest['identity'],
        final_export='latest_export',device=device,peak_GPU_MiB=torch.cuda.max_memory_allocated(device)/2**20 if device.startswith('cuda') else 0))
    return read(dest/'summary.json')


def training_jobs(cfg,out,root,devices,gpu_workers,*,sensitivity=False):
    if not devices or gpu_workers<1: raise ValueError('explicit GPU/CPU resource choice required')
    jobs=[]
    if sensitivity:
        for arm in cfg['mechanism']['paired_retrain_arms']:
            for seed in cfg['training']['seeds']:
                for panel in ('reference','alternative'): jobs.append((arm,seed,panel))
    else: jobs=[(arm,seed,None) for arm in cfg['training']['arms'] for seed in cfg['training']['seeds']]
    running=[];queue=iter(jobs);started=time.monotonic();done=0
    while True:
        while len(running)<gpu_workers:
            try: arm,seed,panel=next(queue)
            except StopIteration: break
            name=f'{arm}__seed{seed}'+('' if panel is None else '__'+panel)
            dest=out/('models' if panel is None else 'sensitivity')/name
            status=dest/'summary.json'
            if status.exists(): done+=1;continue
            device=devices[len(running)%len(devices)]
            used={r['slot'] for r in running};slot=next(s for s in range(gpu_workers) if s not in used)
            device=devices[slot%len(devices)]
            cmd=[sys.executable,str(Path(__file__).resolve()),'--config',str(out/'config.yaml'),'--output',str(out),
                '--project-root',str(root),'--stage','train-one','--arm',arm,'--seed',str(seed),'--device',device]
            if panel: cmd+=['--panel',panel]
            dest.mkdir(parents=True,exist_ok=True)
            log=(dest/'process.log').open('a')
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'))
            running.append(dict(process=process,log=log,name=name,dest=dest,slot=slot))
        if not running: break
        time.sleep(2)
        for job in running[:]:
            code=job['process'].poll()
            if code is not None:
                job['log'].close();running.remove(job);done+=1
                if code:
                    write(job['dest']/f'failure_{time.time_ns()}.json',dict(status='failed',returncode=code,log='process.log',at=now()))
                print(json.dumps(dict(training=job['name'],returncode=code,done=done,total=len(jobs))),flush=True)
        write(out/'progress.json',dict(stage='paired_retraining' if sensitivity else 'training',
            total=len(jobs),completed=done,active=[j['name'] for j in running],elapsed_s=time.monotonic()-started,
            updated_at=now()))


def gpu_pilot(cfg,out,devices,workers):
    """Timed synthetic training through the actual model and backward path."""
    import torch
    from src.tokenizers.functional_tokenizer import FunctionalTokenizer
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    device=devices[0];torch.manual_seed(17)
    if device.startswith('cuda'): torch.cuda.set_device(device)
    model=FunctionalTokenizer(cfg['model'],'QT').to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003)
    e=torch.randn(64,10,2400,device=device);h=torch.randn(64,10,16,device=device)
    y={'eeg':torch.randn(64,10,8,device=device),'hb':torch.randn(64,10,16,device=device)}
    start=None
    for i in range(40):
        if i==10:
            if device.startswith('cuda'): torch.cuda.synchronize(device)
            start=time.monotonic()
        opt.zero_grad();loss=model.objective(model(e,h),e,h,y,cfg['training'])['total'];loss.backward();opt.step()
    if device.startswith('cuda'): torch.cuda.synchronize(device)
    result=dict(device=device,steps=30,seconds=time.monotonic()-start,
        peak_GPU_MiB=torch.cuda.max_memory_allocated(device)/2**20 if device.startswith('cuda') else 0,
        steps_per_second=30/(time.monotonic()-start),pid=os.getpid(),workers=workers)
    write(out/'gpu_pilot.json',result);print(json.dumps(result),flush=True)
    return result


def audit_information(cfg,out):
    manifest,cache=load_cache(out);rows=manifest['rows']
    probes=[];failures=[];readouts=[]
    representations={k:cache[k] for k in ('u','physical','remaining')}
    representations['physical_plus_remaining']=np.concatenate((cache['physical'],cache['remaining']),-1)
    bands=np.asarray(cache['bands'])
    representations['EEG_band_statistics']=np.concatenate((bands.mean(1),bands.std(1),bands[:,-1]-bands[:,0]),-1)
    hb=np.asarray(cache['hb']).reshape(-1,80,2)
    representations['Hb_morphology']=np.concatenate((hb.mean(1),hb.std(1),hb.max(1),hb.min(1),hb[:,-1]-hb[:,0]),-1)
    representations['observed_X']=hb_coordinates(hb)[...,1]
    smooth=np.asarray(cache['smooth'])*np.asarray(manifest['target_scale'])+np.asarray(manifest['target_mean'])
    representations['smooth_X']=smooth[...,1]
    for name,values in representations.items():
        meta,x=trial_features(rows,values,cfg['data']['regions'])
        result,bad=classification_probe(meta,x,cfg,seed=cfg['seed'])
        probes.extend(dict(r,feature=name) for r in result);failures.extend(dict(r,feature=name) for r in bad)
    # Modal predictability is not physiological truth. The student readouts use
    # their own observed inputs only; no joint teacher is an input feature.
    targets=np.asarray(cache['teacher'])
    for modality,features in (('EEG_only',bands),('Hb_only',hb)):
        meta,x=trial_features(rows,features,cfg['data']['regions'])
        for coordinate,j in (('SSM_EEG',0),('SSM_X',1),('SSM_T',2)):
            _,y=trial_features(rows,targets[...,j],cfg['data']['regions'])
            try:
                result=ridge_readout(meta,x,y,cfg)
                readouts.extend(dict(r,input=modality,target=coordinate) for r in result)
            except ValueError as exc: failures.append(dict(input=modality,target=coordinate,reason=str(exc)))
    single=[]
    for r in rows:
        for mode in ('eeg_only','hb_only'):
            status=out/'tasks'/('teacher_'+mode)/(r['id']+'.json')
            if not status.exists(): continue
            if read(status)['status']!='completed':
                single.append(dict(id=r['id'],mode=mode,status='failed'));continue
            joint=arrays(out/'teacher'/'joint'/(r['id']+'.npz'))['physical'][120:200]
            self_teacher=arrays(out/'teacher'/mode/(r['id']+'.npz'))['physical'][120:200]
            single.append(dict(id=r['id'],subject=r['subject'],dataset=r['dataset'],split=r['split'],mode=mode,status='completed',
                EEG_RMS_difference=float(np.sqrt(np.mean((joint[:,0]-self_teacher[:,0])**2))/.025),
                X_RMS_difference=float(np.sqrt(np.mean((hb_coordinates(joint[:,1:])[:,1]-hb_coordinates(self_teacher[:,1:])[:,1])**2))/.025)))
    result=dict(status='completed',classification=probes,predictability=readouts,failures=failures,
        joint_vs_unimodal=single,parameters='fixed_reference_no_individual_variation_not_probed',
        interpretation='deterministic_decomposition_changes_readability_not_information_content',
        planned_trials=manifest['planned_trials'],retained_trials=manifest['retained_trials'])
    write(out/'TK0.json',result)
    return dict(status='completed',classification_rows=len(probes),readout_rows=len(readouts),failures=len(failures))


def evaluate_model_worker(payload):
    cfg,out_string,arm,seed=payload;out=Path(out_string)
    manifest,cache=load_cache(out);rows=manifest['rows'];expected=[r['id'] for r in rows]
    dest=out/'models'/f'{arm}__seed{seed}';export=dest/'latest_export'
    if not (dest/'summary.json').exists(): return dict(status='failed',reason='training_incomplete')
    continuous=arrays(export/'continuous.npz');ids=continuous.pop('identities').astype(str)
    continuous=join_export(expected,ids,continuous)
    paths={'pre':{m:continuous[m+'_pre'] for m in ('eeg','hb')},
           'soft':{m:continuous[m+'_soft'] for m in ('eeg','hb')}}
    if arm in ('Q0','QT'):
        identifiers,hard=read_hard_tokens(export/'hard.npz')
        paths['hard']=join_export(expected,identifiers,hard)
    records,failures,readouts,session_scores=[],[],[],[]
    for path,semantic in paths.items():
        if path=='soft' and arm not in ('Q0','QT'): continue
        for representation in ('S','O','SO'):
            if path=='hard' and representation=='S':
                continue  # Primary S-only result is produced by the separate strict reader.
            for modality in ('eeg','hb','both'):
                parts=[]
                for m in ('eeg','hb') if modality=='both' else (modality,):
                    if representation in ('S','SO'): parts.append(semantic[m])
                    if representation in ('O','SO'): parts.append(continuous[m+'_observation'])
                meta,x=trial_features(rows,np.concatenate(parts,-1),cfg['data']['regions'])
                result,bad=classification_probe(meta,x,cfg,seed=cfg['seed'])
                info=dict(arm=arm,seed=seed,path=path,representation=representation,modality=modality)
                records.extend(dict(r,**info) for r in result);failures.extend(dict(r,**info) for r in bad)
                if path==('hard' if arm in ('Q0','QT') else 'pre'):
                    session_meta,session_ix=session_probe_split(meta)
                    if len(session_ix):
                        ss,sbad=classification_probe(session_meta,x[session_ix],cfg,seed=cfg['seed'])
                        session_scores.extend(dict(r,**info,protocol='within_outer_test_subject_cross_session') for r in ss)
                        failures.extend(dict(r,**info,protocol='within_outer_test_subject_cross_session') for r in sbad)
                if modality in ('eeg','hb') and representation=='S':
                    target=np.asarray(cache['teacher'])[...,0 if modality=='eeg' else 1]
                    _,y=trial_features(rows,target,cfg['data']['regions'])
                    readouts.extend(dict(r,**info) for r in ridge_readout(meta,x,y,cfg))
    # Post-hoc clustering is fitted on TRAIN latents only. The evaluation path
    # reopens the integer-only artifact, never the original continuous vectors.
    cluster_scores=[]
    if arm in ('C0','CT'):
        from sklearn.cluster import MiniBatchKMeans
        train=np.array([r['split']=='train' for r in rows])
        for k in cfg['evaluation']['kmeans_sizes']:
            cluster={}
            for m in ('eeg','hb'):
                latent=continuous[m+'_pre'];flat=latent[train].reshape(-1,latent.shape[-1])
                fit=MiniBatchKMeans(k,random_state=seed,n_init=3,batch_size=1024,max_iter=100).fit(flat)
                cluster['ids_'+m]=fit.predict(latent.reshape(-1,latent.shape[-1])).reshape(latent.shape[:2]).astype('int16')
                cluster['codebook_'+m]=fit.cluster_centers_
            path=dest/'clusters'/f'K{k}.npz';save_arrays(path,**cluster,identities=np.array(expected))
            identifiers,lookup=read_hard_tokens(path);lookup=join_export(expected,identifiers,lookup)
            for modality in ('eeg','hb','both'):
                value=np.concatenate([lookup[m] for m in (('eeg','hb') if modality=='both' else (modality,))],-1)
                meta,x=trial_features(rows,value,cfg['data']['regions'])
                result,bad=classification_probe(meta,x,cfg,seed=cfg['seed'])
                cluster_scores.extend(dict(r,arm=arm,seed=seed,K=k,modality=modality) for r in result)
                failures.extend(dict(r,arm=arm,seed=seed,K=k,modality=modality) for r in bad)
    observation=read(export/'metrics.json')
    summary=dict(status='completed',classification=records,failures=failures,readouts=readouts,
        kmeans=cluster_scores,observation=observation,cross_session=session_scores)
    write(dest/'evaluation.json',summary)
    return dict(status='completed',classification_rows=len(records),failed_probes=len(failures))


def evaluate_hard_worker(payload):
    """Primary token evaluator: never opens the continuous/soft/O export.

    Inputs are integer IDs + frozen books + native identities. Labels and
    teacher targets below are scoring/probe targets only, never features.
    """
    cfg,out_string,arm,seed=payload;out=Path(out_string)
    manifest=read(out/'cache'/'manifest.json');rows=manifest['rows']
    dest=out/'models'/f'{arm}__seed{seed}'
    file=dest/'latest_export'/'hard.npz'
    if not file.exists(): return dict(status='failed',reason='missing_integer_export')
    identities,features=read_hard_tokens(file)
    features=join_export([r['id'] for r in rows],identities,features)
    records,failures,readouts,session_scores=[],[],[],[]
    targets=np.load(out/'cache'/'teacher.npy',mmap_mode='r',allow_pickle=False)
    for modality in ('eeg','hb','both'):
        value=np.concatenate([features[m] for m in (('eeg','hb') if modality=='both' else (modality,))],-1)
        meta,x=trial_features(rows,value,cfg['data']['regions'])
        info=dict(arm=arm,seed=seed,path='hard',representation='S',modality=modality)
        result,bad=classification_probe(meta,x,cfg,seed=cfg['seed'])
        records.extend(dict(r,**info) for r in result);failures.extend(dict(r,**info) for r in bad)
        sm,si=session_probe_split(meta)
        if len(si):
            ss,sbad=classification_probe(sm,x[si],cfg,seed=cfg['seed'])
            session_scores.extend(dict(r,**info,protocol='within_outer_test_subject_cross_session') for r in ss)
            failures.extend(dict(r,**info,protocol='within_outer_test_subject_cross_session') for r in sbad)
        if modality!='both':
            _,y=trial_features(rows,targets[...,0 if modality=='eeg' else 1],cfg['data']['regions'])
            readouts.extend(dict(r,**info) for r in ridge_readout(meta,x,y,cfg))
    write(dest/'hard_evaluation.json',dict(status='completed',classification=records,failures=failures,
        readouts=readouts,cross_session=session_scores,input_files=['latest_export/hard.npz','cache/manifest.json'],
        target_only_files=['cache/teacher.npy'],continuous_export_read=False))
    return dict(status='completed',classification_rows=len(records),failed_probes=len(failures))


def sensitivity_panel(cfg,out,workers):
    manifest,_=load_cache(out);rows=manifest['rows'];coordinates=read(out/'coordinates.json')
    selected=[]
    for ds in cfg['data']['datasets']:
        candidates=[r for r in rows if r['split']=='train' and r['dataset']==ds]
        trials=sorted({r['trial_id'] for r in candidates},key=lambda x:identity_digest(x))[:cfg['mechanism']['panel_training_trials_per_dataset']]
        selected.extend(r for r in candidates if r['trial_id'] in trials)
    results=bounded_tasks(out,'alternative',[(r['id'],(cfg,str(out),r,coordinates[r['coordinate']])) for r in selected],alternative_worker,workers)
    good=[r for r in selected if results[r['id']]['status']=='completed']
    _,cache=load_cache(out);alternative=np.array(cache['teacher'])
    lookup={r['id']:i for i,r in enumerate(rows)}
    for r in good:
        p=arrays(out/'teacher'/'alternative'/(r['id']+'.npz'))['physical']
        target=np.column_stack((p[:,0],hb_coordinates(p[:,1:])[:,::-1]))[120:200]/.025
        alternative[lookup[r['id']]]=(target-np.asarray(manifest['target_mean']))/np.asarray(manifest['target_scale'])
    with (out/'cache'/'alternative.npy').open('wb') as f: np.save(f,alternative,allow_pickle=False)
    info=dict(status='completed' if len(good)>=6 else 'insufficient_witnesses',training_ids=[r['id'] for r in good],
        planned=len(selected),feasible=len(good),failed=len(selected)-len(good),
        selection='hash_order_of_training_trial_identity_before_fitting',
        comparison='same_initialization_order_budget_and_panel_reference_vs_alternative')
    write(out/'sensitivity_panel.json',info);return info


def sensitivity_evaluation_worker(payload):
    from sklearn.cluster import MiniBatchKMeans
    from scipy.linalg import orthogonal_procrustes
    cfg,out_string,arm,seed=payload;out=Path(out_string)
    manifest,cache=load_cache(out);rows=manifest['rows']
    train=np.array([r['split']=='train' for r in rows]);test=np.array([r['split']=='test' for r in rows])
    results=[];sources={};classification=[];readouts=[];failures=[]
    for other_seed in cfg['training']['seeds']:
        for panel in ('reference','alternative') if other_seed==seed else ('reference',):
            p=out/'sensitivity'/f'{arm}__seed{other_seed}__{panel}'
            if (p/'summary.json').exists():
                sources[(other_seed,panel)]=arrays(p/'latest_export'/'continuous.npz')
    if (seed,'reference') not in sources or (seed,'alternative') not in sources:
        return dict(status='failed',reason='missing_paired_training')
    def aligned_rms(reference,other):
        rm=reference[train].mean((0,1));om=other[train].mean((0,1))
        rotation,_=orthogonal_procrustes((other[train]-om).reshape(-1,other.shape[-1]),
            (reference[train]-rm).reshape(-1,reference.shape[-1]))
        return float(np.sqrt(np.mean(((other[test]-om)@rotation+rm-reference[test])**2)))
    for m in ('eeg','hb'):
        baseline=sources[(seed,'reference')][m+'_pre'];alternative=sources[(seed,'alternative')][m+'_pre']
        codes=[]
        for value in (baseline,alternative):
            fit=MiniBatchKMeans(128,random_state=seed,n_init=3,batch_size=1024).fit(value[train].reshape(-1,value.shape[-1]))
            codes.append(fit.predict(value.reshape(-1,value.shape[-1])).reshape(value.shape[:2]))
        aligned,_=align_ids(codes[0],codes[1],train)
        other_seed=next((s for s in cfg['training']['seeds'] if s!=seed and (s,'reference') in sources),None)
        noise=None if other_seed is None else aligned_rms(baseline,sources[(other_seed,'reference')][m+'_pre'])
        results.append(dict(arm=arm,seed=seed,modality=m,paired_latent_RMS=float(np.sqrt(np.mean((alternative[test]-baseline[test])**2))),
            teacher_train_aligned_RMS=aligned_rms(baseline,alternative),seed_train_aligned_RMS=noise,
            aligned_token_agreement=float(np.mean(aligned[test]==codes[0][test])),alignment_support='training_only',
            interpretation='paired_retraining_teacher_sensitivity_not_evaluation_target_swap'))
    for panel in ('reference','alternative'):
        data=sources[(seed,panel)]
        for rep in ('S','O','SO'):
            for modality in ('eeg','hb','both'):
                parts=[]
                for m in (('eeg','hb') if modality=='both' else (modality,)):
                    if rep in ('S','SO'): parts.append(data[m+'_pre'])
                    if rep in ('O','SO'): parts.append(data[m+'_observation'])
                meta,x=trial_features(rows,np.concatenate(parts,-1),cfg['data']['regions'])
                scores,bad=classification_probe(meta,x,cfg,seed=cfg['seed'])
                info=dict(arm=arm,seed=seed,panel=panel,representation=rep,modality=modality)
                classification.extend(dict(r,**info) for r in scores);failures.extend(dict(r,**info) for r in bad)
                if rep=='S' and modality!='both':
                    _,target=trial_features(rows,cache['teacher'][...,0 if modality=='eeg' else 1],cfg['data']['regions'])
                    readouts.extend(dict(r,**info) for r in ridge_readout(meta,x,target,cfg))
    write(out/'sensitivity'/f'{arm}__seed{seed}_comparison.json',dict(status='completed',results=results,
        classification=classification,readouts=readouts,failures=failures))
    return dict(status='completed',results=results,classification_rows=len(classification),failed_probes=len(failures))


def evaluate_sensitivity(cfg,out,workers):
    tasks=[(f'{arm}__seed{seed}',(cfg,str(out),arm,seed)) for arm in cfg['mechanism']['paired_retrain_arms'] for seed in cfg['training']['seeds']]
    results=bounded_tasks(out,'sensitivity_evaluation',tasks,sensitivity_evaluation_worker,workers)
    write(out/'TK4_sensitivity.json',dict(status='completed',panel=read(out/'sensitivity_panel.json'),
        results=results,comparison_files='sensitivity/*_comparison.json'))


def mechanism_inputs(cfg,out):
    if (out/'mechanism'/'manifest.json').exists(): return
    ssm,physical=teacher_configs(cfg);rows=[];data=defaultdict(list)
    for split,count in (('train',cfg['mechanism']['train_identities']),('validation',cfg['mechanism']['validation_identities']),('test',cfg['mechanism']['test_identities'])):
        for identity in range(count):
            examples=synthetic_interventions(cfg,physical,cfg['mechanism']['streams'][split]+identity*101)
            for scenario,value in examples.items():
                rows.append(dict(id=f'{split}_{identity}_{scenario}',subject=f'{split}_{identity}',record=str(identity),
                    trial_id=f'{split}_{identity}_{scenario}',split=split,scenario=scenario,base_identity=f'{split}_{identity}'))
                for key in ('eeg','hb','functional','tau','amplitude'): data[key].append(value[key])
    save_arrays(out/'mechanism'/'inputs.npz',**{k:np.asarray(v) for k,v in data.items()})
    write(out/'mechanism'/'manifest.json',dict(rows=rows,support='all_variants_grouped_by_generator_identity',
        truth='noiseless_generator_response_and_known_intervention_delta',
        measured_interpretation='synthetic_transfer_stress_not_measured_parameter_recovery'))


def mechanism_worker(payload):
    cfg,out_string,arm,seed=payload;out=Path(out_string)
    import torch
    from src.tokenizers.functional_tokenizer import FunctionalTokenizer
    torch.set_num_threads(1)
    path=out/'models'/f'{arm}__seed{seed}'/'checkpoint.pt'
    if not path.exists(): return dict(status='failed',reason='missing_checkpoint')
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=FunctionalTokenizer(cfg['model'],arm);model.load_state_dict(state['model']);model.eval()
    data=arrays(out/'mechanism'/'inputs.npz');meta=read(out/'mechanism'/'manifest.json')['rows']
    # Synthetic EEG uses unit carriers; Hb is in the same .025 model gauge.
    # No per-record renormalization: intervention amplitudes remain observable.
    results=defaultdict(list);reconstruction=[]
    with torch.no_grad():
        for start in range(0,len(meta),64):
            e=torch.tensor(data['eeg'][start:start+64].reshape(-1,10,2400));h=torch.tensor(data['hb'][start:start+64].reshape(-1,10,16)/.025)
            y=model(e,h)
            for m in ('eeg','hb'):
                # Explicitly use lookup from frozen IDs for quantized models.
                v=(getattr(model,m).quantizer.get_embedding(y[m]['hard_ids']) if arm in ('QT','Q0') else y[m]['pre'])
                results[m].append(v.numpy());results[m+'_O'].append(y[m]['observation'].numpy())
            reconstruction.extend((y['hb']['reconstruction']-h).square().mean((1,2)).tolist())
    results={k:np.concatenate(v) for k,v in results.items()}
    index={r['id']:i for i,r in enumerate(meta)};records=[]
    for m in ('eeg','hb'):
        for rep in ('S','O','SO'):
            value=results[m] if rep=='S' else results[m+'_O'] if rep=='O' else np.concatenate((results[m],results[m+'_O']),-1)
            x=[];targets=[];rows=[]
            for i,r in enumerate(meta):
                if r['scenario']=='base': continue
                base=index[r['base_identity']+'_base']
                x.append((value[i]-value[base]).ravel())
                target=np.array([data['amplitude'][i]-data['amplitude'][base],data['tau'][i]-data['tau'][base],
                    np.sqrt(np.mean((data['functional'][i]-data['functional'][base])**2))])
                targets.append(target);rows.append(r)
            readouts=[]
            for j,coordinate in enumerate(('amplitude_multiplier','tau_seconds','functional_RMS')):
                readouts.extend(dict(r,coordinate=coordinate) for r in ridge_readout(rows,np.asarray(x),np.asarray(targets)[:,j:j+1],cfg,group_field='scenario'))
            for scenario in cfg['mechanism']['scenarios'][1:]:
                ix=[i for i,r in enumerate(rows) if r['scenario']==scenario and r['split']=='test']
                records.append(dict(modality=m,representation=rep,scenario=scenario,
                    representation_delta_RMS=float(np.sqrt(np.mean(np.asarray(x)[ix]**2))),
                    true_delta_mean=np.asarray(targets)[ix].mean(0).tolist(),test_identities=len(ix)))
            records.extend(dict(r,modality=m,representation=rep,endpoint='known_delta_frozen_ridge') for r in readouts)
    result=dict(status='completed',arm=arm,seed=seed,results=records,
        Hb_reconstruction_MSE=reconstruction,delta_order=['driver_amplitude','tau_seconds','functional_RMS'],
        alias='identical_input_distinct_attribution; deterministic_model_cannot_disambiguate',
        caveat='analytic_carrier_EEG_transfer_stress; no physiological naming or calibration claim')
    write(out/'models'/f'{arm}__seed{seed}'/'mechanism.json',result)
    return dict(status='completed',rows=len(records))


def coupling_worker(payload):
    cfg,out_string,arm,seed=payload;out=Path(out_string)
    manifest,cache=load_cache(out);rows=manifest['rows'];dest=out/'models'/f'{arm}__seed{seed}'
    if not (dest/'summary.json').exists(): return dict(status='failed',reason='missing_checkpoint')
    export=dest/'latest_export'
    if arm in ('Q0','QT'):
        identities,lookup=read_hard_tokens(export/'hard.npz')
        e=join_export([r['id'] for r in rows],identities,lookup)['eeg']
    else:
        a=arrays(export/'continuous.npz');e=join_export([r['id'] for r in rows],a.pop('identities').astype(str),a)['eeg_pre']
    h=np.asarray(cache['hb']).reshape(-1,80,2)
    # Only rows with every required compatible donor are compared, identically
    # for q0, q1 and all nulls. Donors cannot cross subject partitions.
    donors={};valid=np.ones(len(rows),bool)
    for mode in ('different_record_same_subject','same_task_condition'):
        choices=[]
        for i,r in enumerate(rows):
            pool=[j for j,s in enumerate(rows) if s['split']==r['split'] and s['coordinate']==r['coordinate']
                and s['subject']==r['subject'] and s['trial_id']!=r['trial_id']
                and (s['record']!=r['record'] if mode=='different_record_same_subject' else
                     s['task']==r['task'] and s['condition']==r['condition']
                     and (s['record']!=r['record'] or abs(s['eeg_start_s']-r['eeg_start_s'])>=90))]
            # A concatenated native file may hold several real sessions. A
            # different declared session is an independent-record null unit.
            if mode=='different_record_same_subject' and not pool:
                pool=[j for j,s in enumerate(rows) if s['split']==r['split'] and s['coordinate']==r['coordinate']
                    and s['subject']==r['subject'] and s['session']!=r['session']]
            if not pool: valid[i]=False;choices.append(i)
            else: choices.append(min(pool,key=lambda j:identity_digest(rows[j]['id']+r['id'])))
        donors[mode]=np.array(choices)
    # Whole trials retain all regions, so partial ROI support never changes n.
    good_trials={r['trial_id'] for r in rows}
    good_trials-={r['trial_id'] for i,r in enumerate(rows) if not valid[i]}
    selection=np.array([i for i,r in enumerate(rows) if r['trial_id'] in good_trials])
    if not len(selection):
        write(dest/'coupling.json',dict(status='unavailable',reason='no_common_null_support',planned=len(rows)))
        return dict(status='completed',scientific_status='unavailable')
    selected=[rows[i] for i in selection]
    target=h[selection,32:48]
    _,y=trial_features(selected,target,cfg['data']['regions'])
    controls={}
    controls['Hb_context'],_=coupling_features(h,None)
    controls['paired'],_=coupling_features(h,e)
    for mode,ix in donors.items(): controls[mode],_=coupling_features(h,e[ix])
    # Intersection of nonwrapping token-time support for every candidate shift.
    # All conditions use the identical central patches; mask shifts with data.
    shift_candidates=cfg['coupling']['shifts_in_patches'];margin=max(abs(s) for s in shift_candidates)
    common_mask=np.ones(e.shape[:2],bool);common_mask[:,:margin]=False;common_mask[:,-margin:]=False
    for shift in shift_candidates:
        moved=np.roll(e,shift,axis=1);support=np.ones(e.shape[:2],bool)
        if shift>0: support[:,:shift]=False
        else: support[:,shift:]=False
        moved=np.where(support[...,None],moved,0)
        moved[:,:margin]=0;moved[:,-margin:]=0
        controls['shift_'+str(shift)],_=coupling_features(h,moved,token_mask=support&common_mask)
    for name in ('paired','different_record_same_subject','same_task_condition'):
        values=e.copy() if name=='paired' else e[donors[name]].copy()
        values[:,:margin]=0;values[:,-margin:]=0
        controls[name],_=coupling_features(h,values,token_mask=common_mask)
    bands=np.asarray(cache['bands']).reshape(-1,10,8,30).mean(2)
    bands[:,:margin]=0;bands[:,-margin:]=0
    controls['simple_EEG_features'],_=coupling_features(h,bands,token_mask=common_mask)
    records=[]
    for mode,value in controls.items():
        meta,x=trial_features(selected,value[selection],cfg['data']['regions'])
        for ds in cfg['data']['datasets']:
            ix=[i for i,r in enumerate(meta) if r['dataset']==ds]
            if not ix: continue
            sub=[meta[i] for i in ix]
            try: result=ridge_readout(sub,x[ix],y[ix],cfg,reduce_features=False)
            except ValueError as exc:
                records.append(dict(mode=mode,dataset=ds,status='invalid',reason=str(exc)));continue
            records.extend(dict(r,mode=mode,dataset=ds,status='completed') for r in result)
    write(dest/'coupling.json',dict(status='completed',results=records,planned_windows=len(rows),
        paired_windows=len(selection),paired_trials=len(good_trials),common_support=True,
        target='actually_hidden_native_Hb_8_to_12s; first5s_teacher_reference_outside_student_target',
        controls='same_event_aligned_phase; same_subject_nulls; mask_moves_with_shift',
        predictor='ridge_on_identical_full_visible_Hb_context_plus_EEG; no_joint_PCA; train_scaling_validation_alpha',
        interpretation='offline_completion_not_causal; all_four_fixed_shifts_reported_no_test_lag_selection'))
    return dict(status='completed',rows=len(records),paired_trials=len(good_trials))


def atlas_and_seed_stability(cfg,out):
    manifest,cache=load_cache(out);rows=manifest['rows'];train=np.array([r['split']=='train' for r in rows])
    test=~train & np.array([r['split']=='test' for r in rows]);meta=[r for r,t in zip(rows,test) if t]
    eeg_test=np.asarray(cache['eeg'][test])
    records=[];alignment=[]
    for arm in ('Q0','QT'):
        reference={}
        for seed in cfg['training']['seeds']:
            path=out/'models'/f'{arm}__seed{seed}'/'latest_export'/'hard.npz'
            if not path.exists(): continue
            # Only hard artifacts enter token assignment summaries.
            a=arrays(path)
            if not np.array_equal(a['identities'],[r['id'] for r in rows]): raise ValueError('atlas identity mismatch')
            for m in ('eeg','hb'):
                ids=a['ids_'+m]
                if m in reference:
                    aligned,_=align_ids(reference[m],ids,train)
                    alignment.append(dict(arm=arm,seed=seed,modality=m,
                        heldout_agreement=float(np.mean(aligned[test]==reference[m][test])),
                        alignment='training_only_assignment_overlap_Hungarian'))
                else: reference[m]=ids
                measured=(np.asarray(cache['bands']).reshape(-1,10,8,30) if m=='eeg'
                    else np.asarray(cache['hb']).reshape(-1,10,8,2))
                if m=='hb':measured=np.concatenate((measured,hb_coordinates(measured)),axis=-1)
                for row in frozen_prototypes(meta,ids[test],measured[test]):
                    if m=='eeg':
                        sample,patch=np.nonzero(ids[test]==row['token'])
                        raw=eeg_test[sample,patch].reshape(-1,400,6)
                        row['mean_voltage_patch_training_SD']=raw.mean(0).tolist()
                    row['coordinate_names']=(['HbO','HbR','T','X'] if m=='hb' else
                        ['regional_channel_'+str(i)+'_band_'+str(j) for i in range(6) for j in range(5)])
                    row['coordinate_unit']='train_common_Hb_SD' if m=='hb' else 'log_power_change_from_visible_reference'
                    records.append(dict(row,arm=arm,seed=seed,modality=m))
    write(out/'TK5_atlas.json',dict(status='completed',prototypes=records,seed_alignment=alignment,
        caveat='token IDs arbitrary; conditional pattern names only; no unique physiological-source labels'))


def error_contrasts(cfg,out,jobs):
    readouts=[];coupling=[]
    for name,payload in jobs:
        arm,seed=payload[2:]
        for filename in ('evaluation.json','hard_evaluation.json'):
            p=out/'models'/name/filename
            if p.exists(): readouts.extend(read(p).get('readouts',[]))
        p=out/'models'/name/'coupling.json'
        if p.exists(): coupling.extend(dict(r,arm=arm,seed=seed) for r in read(p).get('results',[]) if r.get('status')=='completed')
    results=[]
    pairs=[('teacher_continuous','CT','pre','C0','pre'),('teacher_discrete','QT','hard','Q0','hard'),
        ('hard_vs_continuous','QT','hard','CT','pre'),('SSM_vs_smooth','CT','pre','C-smooth','pre')]
    for name,a,ap,b,bp in pairs:
        for modality in ('eeg','hb'):
            left={(r['subject'],r['seed']):r['NRMSE'] for r in readouts if r['arm']==a and r['path']==ap and r['modality']==modality}
            right={(r['subject'],r['seed']):r['NRMSE'] for r in readouts if r['arm']==b and r['path']==bp and r['modality']==modality}
            values=defaultdict(list)
            for key in set(left)&set(right):values[key[0]].append((right[key]-left[key])/max(right[key],1e-8))
            effect=paired_interval([np.mean(v) for v in values.values() if len(v)==len(cfg['training']['seeds'])],cfg['seed'],cfg['evaluation']['bootstrap_repeats'])
            results.append(dict(comparison=name,modality=modality,**effect,
                estimand='relative_readout_error_reduction_positive_is_better',engineering_margin=cfg['evaluation']['readout_relative_error_margin']))
    nulls=[]
    for arm in cfg['training']['arms']:
        for ds in cfg['data']['datasets']:
            select=[r for r in coupling if r['arm']==arm and r['dataset']==ds]
            paired={(r['subject'],r['seed']):r['NRMSE'] for r in select if r['mode']=='paired'}
            for mode in sorted({r['mode'] for r in select}-{'paired'}):
                reference={(r['subject'],r['seed']):r['NRMSE'] for r in select if r['mode']==mode}
                values=defaultdict(list)
                for key in set(paired)&set(reference):values[key[0]].append(reference[key]-paired[key])
                interval=paired_interval([np.mean(v) for v in values.values() if len(v)==len(cfg['training']['seeds'])],cfg['seed'],cfg['evaluation']['bootstrap_repeats'])
                nulls.append(dict(arm=arm,dataset=ds,comparator=mode,**interval,
                    estimand='comparator_NRMSE_minus_paired_NRMSE_positive_is_better',
                    interpretation='unadjusted_exploratory_paired_subject_interval_not_causal_evidence'))
    write(out/'readout_and_coupling_contrasts.json',dict(readouts=results,coupling=nulls))
    return results,nulls


def evaluate_suite(cfg,out,root,workers,devices,gpu_workers):
    if not (out/'TK0.json').exists(): audit_information(cfg,out)
    jobs=[(f'{a}__seed{s}',(cfg,str(out),a,s)) for a in cfg['training']['arms'] for s in cfg['training']['seeds']]
    bounded_tasks(out,'evaluation',jobs,evaluate_model_worker,workers)
    hard_jobs=[(name,payload) for name,payload in jobs if payload[2] in ('Q0','QT')]
    bounded_tasks(out,'hard_evaluation',hard_jobs,evaluate_hard_worker,workers)
    mechanism_inputs(cfg,out)
    bounded_tasks(out,'mechanism',jobs,mechanism_worker,min(workers,8))
    bounded_tasks(out,'coupling',jobs,coupling_worker,workers)
    if not (out/'sensitivity_panel.json').exists(): sensitivity_panel(cfg,out,workers)
    if read(out/'sensitivity_panel.json')['status']=='completed':
        training_jobs(cfg,out,root,devices,gpu_workers,sensitivity=True)
        evaluate_sensitivity(cfg,out,workers)
    else:
        write(out/'TK4_sensitivity.json',dict(status='unavailable',panel=read(out/'sensitivity_panel.json')))
    atlas_and_seed_stability(cfg,out)
    classification=[];failed=[]
    for name,_ in jobs:
        p=out/'models'/name/'evaluation.json'
        if p.exists(): classification.extend(read(p)['classification'])
        else: failed.append(name)
        hard=out/'models'/name/'hard_evaluation.json'
        if hard.exists(): classification.extend(read(hard)['classification'])
        elif name.startswith(('QT__','Q0__')) and name not in failed:failed.append(name)
    effects=compare_arms(classification,cfg)
    manifest,_=load_cache(out)
    if manifest['failed_trials']:
        for row in effects:
            row['success_subset_verdict']=row['verdict'];row['verdict']='invalid_experiment'
            row['reason']='unresolved_failed_trial_denominator; success_subset_kept_separately'
    write(out/'contrasts.json',dict(effects=effects,branch_increments=branch_increments(classification,cfg),
        failed_training_or_evaluation=failed))
    readout_effects,coupling_effects=error_contrasts(cfg,out,jobs)
    task_failures={}
    for stage in ('evaluation','hard_evaluation','mechanism','coupling','alternative','sensitivity_evaluation'):
        statuses=[read(p) for p in (out/'tasks'/stage).glob('*.json')]
        task_failures[stage]=dict(planned_or_attempted=len(statuses),
            completed=sum(r.get('status')=='completed' for r in statuses),
            failed_or_unavailable=sum(r.get('status')!='completed' for r in statuses))
    final=dict(status='completed' if not failed else 'completed_with_failures',experiment_id=cfg['experiment_id'],
        planned_training=24,completed_training=24-len(failed),failed=failed,
        planned_trials=manifest['planned_trials'],retained_trials=manifest['retained_trials'],
        failed_trials=len(manifest['failed_trials']),classification_rows=len(classification),
        stage_outcomes=task_failures,readout_effects=readout_effects,coupling_effects=coupling_effects,
        primary_effects=[r for r in effects if r['task'] in cfg['evaluation']['primary_tasks'] and r['representation']=='S'
            and r['modality']=='both' and r['label_fraction']==1.0],
        evidence=dict(TK0='TK0.json',TK1_TK3='contrasts.json',TK4='models/*/mechanism.json',
            representative_sensitivity='TK4_sensitivity.json',TK5='TK5_atlas.json',coupling='models/*/coupling.json'),
        interpretation='conditional_teacher_development_screen_no_qualification_or_fresh_confirmation',completed_at=now())
    write(out/'summary.json',final);write(out/'progress.json',dict(stage='completed',**final))
    print(json.dumps(clean(final),ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,default=SOURCE)
    parser.add_argument('--stage',choices=['plan','synthetic','pilot','all','prepare','train','train-one','evaluate','auxiliary','hard-probe','gpu-pilot','cpu-pilot'],default='all')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--devices',default='cuda:1')
    parser.add_argument('--gpu-workers',type=int,default=1)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--arm',default='CT');parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--panel',choices=['reference','alternative'])
    parser.add_argument('--pilot-source',type=Path)
    args=parser.parse_args();cfg=config(args.config);out=args.output.resolve();root=args.project_root.resolve()
    if args.workers<1 or args.workers>cfg['resources']['max_cpu_workers'] or args.gpu_workers>cfg['resources']['max_gpu_workers']:
        raise ValueError('resources exceed frozen maximum')
    devices=args.devices.split(',')
    freeze_run(cfg,out,root)
    if not (out/'config.yaml').exists(): (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    if args.stage=='train-one':
        train_one(cfg,out,args.arm,args.seed,args.device,panel=args.panel,
            steps=cfg['mechanism']['paired_retrain_steps'] if args.panel else None);return
    if args.stage=='hard-probe':
        evaluate_hard_worker((cfg,str(out),args.arm,args.seed));return
    if args.stage=='gpu-pilot': gpu_pilot(cfg,out,devices,args.gpu_workers);return
    if args.stage=='cpu-pilot':
        if args.pilot_source is None: raise ValueError('--pilot-source is required')
        cpu_benchmark(cfg,out,args.pilot_source);return
    if args.stage=='auxiliary':
        if not (out/'TK0.json').exists(): audit_information(cfg,out)
        mechanism_inputs(cfg,out)
        if not (out/'sensitivity_panel.json').exists(): sensitivity_panel(cfg,out,args.workers)
        return
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if args.stage=='plan':
            plan=inventory(root,cfg);write(out/'inventory.json',plan)
            print(json.dumps(dict(rows=len(plan['rows']),subjects=plan['subjects'],excluded=len(plan['excluded']))));return
        if args.stage in ('synthetic','pilot','all'):
            if not (out/'synthetic_checks.json').exists(): software_synthetic(cfg,out)
            if args.stage=='synthetic': return
        if args.stage in ('pilot','prepare','all'):
            write(out/'progress.json',dict(stage='preparing',details='progress/',updated_at=now()))
            prepare(cfg,out,root,args.workers,pilot=args.stage=='pilot')
            if args.stage=='pilot': return
            if not (out/'cache'/'manifest.json').exists(): pack(cfg,out)
        if args.stage in ('train','all'):
            # TK0 and the mechanism/alternative-target panel need no fitted
            # tokenizer. Run them on CPUs while the GPU matrix progresses.
            auxiliary_cmd=[sys.executable,str(Path(__file__).resolve()),'--config',str(out/'config.yaml'),
                '--output',str(out),'--project-root',str(root),'--stage','auxiliary','--workers',str(min(args.workers,8))]
            with (out/'auxiliary.log').open('a') as log:
                auxiliary=subprocess.Popen(auxiliary_cmd,stdout=log,stderr=subprocess.STDOUT)
                training_jobs(cfg,out,root,devices,args.gpu_workers)
                if auxiliary.wait()!=0: raise RuntimeError('TK0/mechanism preparation failed; see auxiliary.log')
        if args.stage in ('evaluate','all'): evaluate_suite(cfg,out,root,args.workers,devices,args.gpu_workers)
    except Exception as exc:
        failure=f'controller_failure_{time.time_ns()}.json'
        write(out/failure,dict(status='failed',reason=str(exc),traceback=traceback.format_exc(),at=now()))
        write(out/'progress.json',dict(stage='failed',failure=failure,updated_at=now()))
        raise


if __name__=='__main__': main()
