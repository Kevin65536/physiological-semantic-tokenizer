#!/usr/bin/env python3
"""Bounded Visual investigation; source lineage, fixed development controls.

No report export, historical protected artifacts or outer-evaluation signals.
The existing loader, neural fitter and scheduler remain their respective owners.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
from src.data.downstream_benchmark import DownstreamBenchmarkDataset, canonical_sample_id
from src.data.event_alignment import read_xlsx_rows, read_visual_fnirs_csv, window_within_alignment_support
from src.metrics.downstream_benchmark import score_predictions, classification_baselines
from comparative_methods.downstream_rebench_worker_v1 import fit_neural, fit_input_scaler, ResourceMonitor
from comparative_methods.rebench_baselines_v1 import SignalFeatures
from comparative_methods.run_downstream_rebench_v1 import atomic_json

ARMS=['lineage','prepare','probes','cbramod_lp','ft_mean','ft_ordered','ft_ordered_balanced','small_fit','aggregate']
CLASSES=['RR','RF','FF','FR']


def serial(value):
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    if isinstance(value,(np.ndarray,torch.Tensor)):return serial(value.tolist())
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    return value


def write(path,value):atomic_json(Path(path),serial(value))


def target_view(labels,name):
    labels=np.asarray(labels,dtype=int)
    if name=='four_class':return labels.copy(),4
    if name=='motivation':return np.isin(labels,[2,3]).astype(int),2
    if name=='recognition':return np.isin(labels,[1,2]).astype(int),2
    raise ValueError('unknown label decomposition')


def shared_profiles(inventory):
    profiles=deepcopy(inventory['tasks']['visual']['profiles'])
    p=profiles['multimodal'];e=profiles['eeg_only']
    p['channel_names']=e['channel_names'];p['eeg_geometry']=e['eeg_geometry']
    p['geometry_provenance']['eeg']=e['geometry_provenance']['eeg']
    p.update(task='visual',duration_s=8.,window_start_s=0.,input_window_schema='downstream_rebench_v1')
    return p


def corrected_storage_panel(profile,audit):
    """Resolve intended electrodes to v5 storage aliases without rewriting v5.

    The original XLS owns raw-channel identity. Location.ced supplies geometry
    by electrode name only. This is a versioned diagnostic selection view, not
    a relabelling of historical cache manifests or predictions.
    """
    rows=audit['channel_comparison']
    by_physical={r['documented'].casefold():r for r in rows if not r['currently_excluded']}
    selected=[by_physical[name.casefold()] for name in profile['channel_names']]
    aliases=[r['current'] for r in selected]
    if len(set(aliases))!=len(aliases) or any(x is None for x in aliases):
        raise ValueError('corrected electrode panel must resolve to unique retained channels')
    return aliases,[r['raw_channel'] for r in selected]


def lineage(config,project,run):
    from experiments.build_clean_event_index import iter_visual
    inventory=json.loads((project/config['inventory']).read_text())
    rows=[r for r in inventory['samples'] if r['task_id']=='visual']
    raw_root=project/'data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults'
    rebuilt,reports=iter_visual(raw_root,subject_limit=16,record_limit=100)
    by_identity={(e.subject,e.record_id,int(e.metadata['epoch_id'])):e for e in rebuilt}
    type_maps={subject:{int(float(r['Epoch_ID'])):r['Type'] for r in read_xlsx_rows(str(raw_root/subject/(subject+'_type.xlsx')))}
               for subject in sorted({r['subject_id'] for r in rows})}
    errors=[];clock_errors=[];native_files=set();response_delays=[];per_event=[];mark_cache={}
    for row in rows:
        subject=row['subject_id'];epoch=int(row['event_id']);native_label=type_maps[subject].get(epoch)
        if native_label!=row['label'] or CLASSES[row['target']]!=native_label:
            errors.append(dict(sample_id=row['sample_id'],kind='native_type_label',native=native_label))
        probe_times=[]
        for ref in row['event_refs']:
            record=inventory['records'][ref['join_key']]
            key=(subject,record['record_id'],epoch);event=by_identity.get(key)
            if event is None:
                errors.append(dict(sample_id=row['sample_id'],kind='missing_rebuilt_event',key=key));continue
            for modality in ('eeg','fnirs'):
                error=abs(getattr(event,f'{modality}_time_ms')-ref[f'{modality}_time_ms'])
                clock_errors.append(error)
                if error>1e-6:errors.append(dict(sample_id=row['sample_id'],kind='clock_mismatch',modality=modality,error_ms=error))
            if event.label!=row['label']:errors.append(dict(sample_id=row['sample_id'],kind='rebuilt_label'))
            probe_times.append(ref['eeg_time_ms'])
            native_files.update(event.metadata['source_files'])
            if ref.get('probe')=='Probe1':
                path=next(project/p for p in event.metadata['source_files'] if p.endswith('_Oxy.csv'))
                if path not in mark_cache:mark_cache[path]=read_visual_fnirs_csv(path)
                marks=mark_cache[path]['marks'];anchor=ref['fnirs_time_ms']
                next_onsets=[m['onset_ms'] for m in marks if m['mark']==1 and m['onset_ms']>anchor+1e-6]
                stop=min(next_onsets) if next_onsets else anchor+15000
                responses=[(m['onset_ms']-anchor)/1000 for m in marks if m['mark']==3 and anchor<=m['onset_ms']<stop]
                offset=[(m['onset_ms']-anchor)/1000 for m in marks if m['mark']==2 and anchor<=m['onset_ms']<stop]
                if len(responses)==1:response_delays.append(responses[0])
                per_event.append(dict(sample_id=row['sample_id'],subject_id=subject,label=row['label'],
                                      response_delays_s=responses,stimulus_offset_s=offset))
        if len(probe_times)!=2 or abs(probe_times[0]-probe_times[1])>1e-6:
            errors.append(dict(sample_id=row['sample_id'],kind='probe_pair_disagreement'))
    delays=np.asarray(response_delays)
    subject_counts={s:dict(Counter(r['label'] for r in rows if r['subject_id']==s)) for s in sorted(type_maps)}
    result=dict(status='passed' if not errors else 'failed',sample_count=len(rows),probe_event_count=sum(len(r['event_refs']) for r in rows),
                class_counts=dict(Counter(r['label'] for r in rows)),subject_class_counts=subject_counts,
                native_label_and_rebuilt_clock_mismatches=errors,max_clock_error_ms=max(clock_errors,default=None),
                response=dict(single_response_count=len(delays),missing_or_multiple_count=len(rows)-len(delays),
                    quantiles_s=np.quantile(delays,[0,.1,.5,.9,1]),after8s_count=int((delays>=8).sum()),after12s_count=int((delays>=12).sum())),
                native_source_files=sorted(native_files),per_event=per_event,
                builder_reports=[r.to_dict() for r in reports],
                boundary='native type/annotation/ETG-mark metadata only; no original waveform arrays; excluded S06 Part1 remains excluded')
    write(run/'lineage.json',result)
    if errors:raise ValueError('Visual source lineage mismatch; diagnostics stopped before signal access')
    return {k:v for k,v in result.items() if k not in ['native_source_files','per_event','builder_reports']}


class ArrayDataset:
    def __init__(self,arrays,indices):self.arrays=arrays;self.indices=list(map(int,indices))
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        i=self.indices[index];a=self.arrays
        return dict(eeg=a['eeg'][i],fnirs=a['fnirs'][i],target=a['target'][i],target_valid_mask=np.bool_(True),
            sample_id=str(a['sample_id'][i]),subject_id=str(a['subject_id'][i]),record_id=str(a['record_id'][i]),
            valid_mask={m:a[m+'_mask'][i].any(0) for m in ('eeg','fnirs')},channel_valid_mask={m:a[m+'_mask'][i] for m in ('eeg','fnirs')})


def prepare(config,project,run):
    audit=json.loads((run/'lineage.json').read_text())
    if audit['status']!='passed':raise ValueError('source audit has not passed')
    original=json.loads((project/config['inventory']).read_text());splits=json.loads((project/config['splits']).read_text())
    fold=splits['tasks']['visual']['folds'][config['outer_fold']];inner=fold['inner_folds'][config['inner_fold']]
    train=set(inner['train_subjects']);validation=set(inner['validation_subjects']);subjects=train|validation
    if train&validation or subjects&set(fold['eval_subjects']):raise ValueError('overlapping subjects')
    inventory=deepcopy(original);rows=[deepcopy(r) for r in original['samples'] if r['task_id']=='visual' and r['subject_id'] in subjects]
    profile=shared_profiles(original);inventory['samples']=rows
    storage_names=profile['channel_names'];channel_correction=None
    if config.get('channel_identity_audit'):
        audit_path=project/config['channel_identity_audit']
        audit=json.loads(audit_path.read_text())
        storage_names,raw_names=corrected_storage_panel(profile,audit)
        channel_correction=dict(schema='visual_native_channel_selection_v1',source=str(audit_path),
            physical_names=profile['channel_names'],raw_channel_names=raw_names,v5_storage_aliases=storage_names,
            boundary='same existing measurements and native events; correct physical electrode selection; v5 retained unchanged; A64 not added to the fixed 16-channel panel')
        profile['channel_identity_schema']='visual_native_channel_selection_v1'
    inventory['tasks']={'visual':{**original['tasks']['visual'],'profiles':{'multimodal':profile}}}
    plan=dict(task='visual',train_subjects=sorted(train),validation_subjects=sorted(validation),outer_subjects_not_read=fold['eval_subjects'],
              sample_ids=[r['sample_id'] for r in rows],train_count=sum(r['subject_id'] in train for r in rows),
              validation_count=sum(r['subject_id'] in validation for r in rows),profile=profile,
              windows_s=[[0,8],[0,12]],sample_policy='same native events in both windows and all methods; preserve masks',
              scope='single development split; no new untouched evaluation; no report export')
    if channel_correction is not None:plan['channel_correction']=channel_correction
    # Freeze all identities before signal arrays are opened.
    write(run/'plan.json',plan)
    idx=CleanPhysiologyCacheIndex(original['cache_root']);events={(e['join_key'],int(e['event_index'])):e for e in idx.events}
    for r in rows:
        for ref in r['event_refs']:
            if not window_within_alignment_support(events[(ref['join_key'],ref['event_index'])],0,12):
                raise ValueError('12s window crosses unverified alignment support; no method-specific deletion')
    inventory['tasks']['visual']['profiles']['multimodal']={**profile,'channel_names':storage_names,'duration_s':12.,'input_samples':2400,'fnirs_samples':120,'input_window_schema':'visual_12s_diagnostic_v1'}
    for row in rows:
        row['window_duration_s']=12.
        row['sample_id']=canonical_sample_id(row['dataset_id'],row['subject_id'],row['record_id'],'visual',row['event_id'],0,12)
    ds=DownstreamBenchmarkDataset(inventory,task_id='visual',modality_profile='multimodal')
    values={m:[] for m in ('eeg','fnirs','eeg_mask','fnirs_mask')}
    for sample in ds:
        for modality in ('eeg','fnirs'):
            values[modality].append(np.asarray(sample[modality],dtype=np.float32))
            values[modality+'_mask'].append(sample['channel_valid_mask'][modality])
    arrays={k:np.stack(v) for k,v in values.items()}
    arrays.update(target=np.asarray([r['target'] for r in rows]),sample_id=np.asarray(plan['sample_ids']),
        subject_id=np.asarray([r['subject_id'] for r in rows]),record_id=np.asarray([r['record_id'] for r in rows]),
        event_id=np.asarray([r['event_id'] for r in rows]),partition=np.asarray(['train' if r['subject_id'] in train else 'validation' for r in rows]))
    if not arrays['eeg_mask'].all():raise ValueError('incomplete EEG observation support requires a separate adapter contract')
    for m in ('eeg','fnirs'):
        if not np.isfinite(arrays[m][arrays[m+'_mask']]).all():raise ValueError('nonfinite valid signal')
    np.savez(run/'shared_12s_arrays.npz',**arrays)
    q=dict(sample_count=len(rows),eeg_shape=arrays['eeg'].shape,fnirs_shape=arrays['fnirs'].shape,
        partial_eeg_count=int((~arrays['eeg_mask'].all((1,2))).sum()),partial_fnirs_count=int((~arrays['fnirs_mask'].all((1,2))).sum()),
        train_count=plan['train_count'],validation_count=plan['validation_count'])
    write(run/'input_audit.json',q)
    return q


def window_arrays(arrays,seconds):
    out=dict(arrays)
    for m,rate in [('eeg',200),('fnirs',10)]:
        out[m]=arrays[m][...,:seconds*rate];out[m+'_mask']=arrays[m+'_mask'][...,:seconds*rate]
    return out


def batch_iter(a,indices,scaler,device,size=32):
    for start in range(0,len(indices),size):
        rows=np.asarray(indices[start:start+size]);out={}
        for m in ('eeg','fnirs'):
            x=torch.as_tensor(a[m][rows],device=device);mask=torch.as_tensor(a[m+'_mask'][rows],device=device)
            x=(x-torch.tensor(scaler[m]['mean'],device=device))/torch.tensor(scaler[m]['scale'],device=device)
            out[m]=torch.where(mask,x,0.)
        out['channel_valid_mask']={m:torch.as_tensor(a[m+'_mask'][rows],device=device) for m in ('eeg','fnirs')}
        yield rows,out


def metric(y,pred,subjects,classes):
    out=score_predictions(y,pred,np.ones(len(y),bool),subjects,'visual',classes)
    out.update(accuracy=float(accuracy_score(y,pred)),confusion=confusion_matrix(y,pred,labels=range(classes)),
               pooled_class_recall=recall_score(y,pred,labels=range(classes),average=None,zero_division=0))
    return out


def probe_fit(x,a,tr,va,config,target,weight,permutation=False):
    y,classes=target_view(a['target'],target);fit_y=y[tr].copy()
    if permutation:np.random.default_rng(config['seed']).shuffle(fit_y)
    scale=StandardScaler().fit(x[tr]);model=LogisticRegression(C=config['probes']['C'],class_weight=weight,max_iter=3000).fit(scale.transform(x[tr]),fit_y)
    train_pred=model.predict(scale.transform(x[tr]));pred=model.predict(scale.transform(x[va]))
    baseline=classification_baselines(y[tr],dict(target=y[va],mask=np.ones(len(va),bool),subject_id=a['subject_id'][va]),classes)
    return dict(train=metric(y[tr],train_pred,a['subject_id'][tr],classes),validation=metric(y[va],pred,a['subject_id'][va],classes),
                B0=baseline['b0'],feature_count=x.shape[1],iterations=model.n_iter_,class_weight=weight,target=target,
                permutation=permutation,train_count=len(tr),validation_count=len(va))


def chronological_indices(a):
    train=[];validation=[];gaps=[]
    # Only original training subjects; never use the development subjects here.
    for subject in sorted(set(a['subject_id'][a['partition']=='train'])):
        indices=np.flatnonzero((a['subject_id']==subject)&(a['partition']=='train'))
        indices=sorted(indices,key=lambda i:int(a['event_id'][i]))
        cut=int(len(indices)*.7);train.extend(indices[:cut]);gaps.extend(indices[cut:cut+1]);validation.extend(indices[cut+1:])
    return np.asarray(train),np.asarray(validation),gaps


def probes(config,a,profile,output,device):
    tr=np.flatnonzero(a['partition']=='train');va=np.flatnonzero(a['partition']=='validation');results=[]
    within_tr,within_va,gaps=chronological_indices(a)
    windows=config['probes'].get('windows_s',[8,12])
    if not windows or not set(windows).issubset({8,12}):raise ValueError('investigation supports only declared 8/12s windows')
    for seconds in windows:
        view=window_arrays(a,seconds);scaler=fit_input_scaler(ArrayDataset(view,tr))
        p={**profile,'duration_s':float(seconds),'input_samples':seconds*200,'fnirs_samples':seconds*10,
           'input_window_schema':'visual_12s_diagnostic_v1' if seconds==12 else 'downstream_rebench_v1'}
        features={}
        for modality in ('eeg','fnirs','multimodal'):
            model=SignalFeatures('visual',{**p,'baseline_modality':modality}).to(device);xs=[]
            with torch.no_grad():
                for rows,b in batch_iter(view,np.arange(len(view['target'])),scaler,device):
                    xs.append(model(b['eeg'],b['fnirs'] if modality!='eeg' else None,eeg_mask=b['channel_valid_mask']['eeg'],fnirs_mask=b['channel_valid_mask']['fnirs']).cpu().numpy())
            x=np.concatenate(xs);features[modality]=x
            for target in ('four_class','motivation','recognition'):
                for weight in (None,'balanced'):
                    results.append(dict(window_seconds=seconds,features=modality,split='cross_subject',**probe_fit(x,a,tr,va,config,target,weight)))
            results.append(dict(window_seconds=seconds,features=modality,split='cross_subject_null',**probe_fit(x,a,tr,va,config,'four_class','balanced',True)))
        # Within-subject comparison refits every preprocessing scale on its own
        # training rows; do not reuse the cross-subject population statistics.
        within_scale=fit_input_scaler(ArrayDataset(view,within_tr))
        model=SignalFeatures('visual',{**p,'baseline_modality':'multimodal'}).to(device);xs=[]
        with torch.no_grad():
            for rows,b in batch_iter(view,np.arange(len(view['target'])),within_scale,device):
                xs.append(model(b['eeg'],b['fnirs'],eeg_mask=b['channel_valid_mask']['eeg'],fnirs_mask=b['channel_valid_mask']['fnirs']).cpu().numpy())
        x=np.concatenate(xs)
        for target in ('four_class','motivation','recognition'):
            results.append(dict(window_seconds=seconds,features='multimodal',split='within_subject_chronological',**probe_fit(x,a,within_tr,within_va,config,target,'balanced')))
        small=np.concatenate([tr[np.flatnonzero(a['target'][tr]==c)[:8]] for c in range(4)])
        results.append(dict(window_seconds=seconds,features='multimodal',split='training_memorization_positive_control',
                            **probe_fit(features['multimodal'],a,small,small,config,'four_class',None)))
    return dict(results=results,chronological_gap_sample_ids=a['sample_id'][gaps],
                boundary='fixed C; development comparisons and chronological training-subject controls; no target permutation at evaluation')


def cbramod_lp(config,a,profile,output,device,project):
    from comparative_methods.CBraMod.rebench_v1 import create_audit_model
    a=window_arrays(a,8);tr=np.flatnonzero(a['partition']=='train');va=np.flatnonzero(a['partition']=='validation')
    scaler=fit_input_scaler(ArrayDataset(a,tr));p={**profile,'audit_readout':'flatten','audit_training_eeg_scaler':scaler['eeg'],
        'method_root':str(project/'comparative_methods/CBraMod'),'eeg_unit':'uV'}
    model=create_audit_model('visual',p,device).eval();features=[]
    with torch.no_grad():
        for rows,b in batch_iter(a,np.arange(len(a['target'])),scaler,device):
            features.append(model(b['eeg'],eeg_mask=b['channel_valid_mask']['eeg']).cpu().numpy())
    x=np.concatenate(features);mean=x.reshape(len(x),16,8,200).mean((1,2));result=[]
    for name,f in [('mean',mean),('ordered',x)]:
        for target in ('four_class','motivation','recognition'):
            for weight in (None,'balanced'):
                result.append(dict(readout=name,**probe_fit(f,a,tr,va,config,target,weight)))
    return dict(results=result)


def neural(arm,config,a,profile,output,device,project):
    from comparative_methods.CBraMod.rebench_v1 import CBraModAuditModel
    a=window_arrays(a,8);tr=np.flatnonzero(a['partition']=='train');va=np.flatnonzero(a['partition']=='validation')
    small=arm=='small_fit';balanced=arm=='ft_ordered_balanced';ordered=arm!='ft_mean'
    if small:
        tr=np.concatenate([tr[np.flatnonzero(a['target'][tr]==c)[:8]] for c in range(4)]);va=tr.copy()
    train=ArrayDataset(a,tr);validation=None if small else ArrayDataset(a,va)
    scaler=fit_input_scaler(train);counts=np.bincount(a['target'][tr],minlength=4)
    weights=len(tr)/(4*counts) if balanced else np.ones(4)
    p={**profile,'audit_training_eeg_scaler':scaler['eeg'],'audit_readout':'flatten' if ordered else 'mean',
       'audit_head':'source_mlp' if ordered else 'linear','method_root':str(project/'comparative_methods/CBraMod'),'eeg_unit':'uV'}
    class WeightedModel(CBraModAuditModel):
        def __init__(self,task,profile,device,trainable):
            super().__init__(task,profile,device,trainable)
            self.register_buffer('class_weights',torch.as_tensor(weights,dtype=torch.float32,device=device))
        def supervised_loss(self,eeg,fnirs,target,target_valid_mask=None,eeg_mask=None,**kwargs):
            logits=self(eeg,eeg_mask=eeg_mask)
            return dict(total=nn.functional.cross_entropy(logits,target.long(),weight=self.class_weights))
    module=SimpleNamespace(OUTPUT_KIND='predictions',create_model=lambda task,profile,device,trainable:WeightedModel(task,profile,device,trainable))
    budget=config['cbramod'];job=dict(job_id=arm,kind='diagnostic',task='visual',method='cbramod',seed=config['seed'],modality_profile='eeg_only',
        batch_size=budget['batch_size'],microbatch_size=budget['microbatch_size'],gradient_accumulation=budget['batch_size']//budget['microbatch_size'],
        precision='bf16',data_workers=0,training=dict(max_epochs=budget['max_epochs'],min_epochs=budget['min_epochs'],patience=budget['patience']))
    candidate={k:budget[k] for k in ['optimizer','learning_rate','weight_decay','gradient_clip_norm','lr_schedule']}
    model,_,training=fit_neural(module,p,train,validation,job,device,output,arm,candidate)
    model.eval();indices=np.concatenate([tr,va]);pred=[]
    with torch.no_grad():
        for rows,b in batch_iter(a,indices,model.input_scaler,device,budget['microbatch_size']):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):logits=model(b,job)
            pred.extend(logits.argmax(-1).cpu().tolist())
    pred=np.asarray(pred);n=len(tr)
    np.savez(output/'predictions.npz',sample_id=a['sample_id'][indices],prediction=pred,target=a['target'][indices])
    return dict(train=metric(a['target'][tr],pred[:n],a['subject_id'][tr],4),validation=metric(a['target'][va],pred[n:],a['subject_id'][va],4),
                class_weights=weights,training=training,readout=p['audit_readout'],head=p['audit_head'],
                scope='training-only memorization control' if small else 'single development split and seed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,default=ROOT);parser.add_argument('--arm',choices=ARMS,required=True);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();config=yaml.safe_load(args.config.read_text());run=args.run.resolve();output=run/args.arm;output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists() and json.loads((output/'result.json').read_text()).get('status')=='completed':return
    torch.set_num_threads(config['resources']['threads_per_job']);torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    write(output/'status.json',dict(status='running',arm=args.arm))
    try:
        with ResourceMonitor(args.device) as monitor:
            if args.arm=='lineage':result=lineage(config,args.project_root,run)
            elif args.arm=='prepare':result=prepare(config,args.project_root,run)
            elif args.arm=='aggregate':
                results={};failures=[]
                for arm in ARMS[:-1]:
                    path=run/arm/'result.json'
                    if path.exists() and (record:=json.loads(path.read_text())).get('status')=='completed':results[arm]=record
                    else:failures.append(arm)
                result=dict(status='completed' if not failures else 'complete_with_failures',results=results,failures=failures,
                    scope=config['scope'],report_inclusion='pending_user_decision_no_report_export')
                write(run/'summary.json',result)
            else:
                with np.load(run/'shared_12s_arrays.npz',allow_pickle=False) as saved:a={k:saved[k] for k in saved.files}
                profile=json.loads((run/'plan.json').read_text())['profile']
                if args.arm=='probes':result=probes(config,a,profile,output,args.device)
                elif args.arm=='cbramod_lp':result=cbramod_lp(config,a,profile,output,args.device,args.project_root)
                else:result=neural(args.arm,config,a,profile,output,args.device,args.project_root)
            usage=monitor.snapshot()
        write(output/'result.json',dict(status='completed',arm=args.arm,result=result,resources=usage));write(output/'resource_usage.json',usage)
        write(output/'status.json',dict(status='completed',arm=args.arm))
    except Exception as error:
        write(output/'status.json',dict(status='failed',arm=args.arm,error=repr(error)));raise


if __name__=='__main__':main()
