#!/usr/bin/env python3
"""Train-defined quiet-observation atoms on one frozen Hb tokenizer."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import yaml

from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_examples, base_tokenizer_batch
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics, joint_reconstruction_summary
from src.analysis.tokenizer_shape_diagnostics import training_weights
from src.tokenizers.contextual_latent_dictionary import load_model as load_contextual
from src.tokenizers.silent_observation_tokenizer import SilentObservationTokenizer, load_model
from experiments.scripts.run_dual_component_tokenizer import owner_lock


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    def encode(item):
        if isinstance(item,np.ndarray):return item.tolist()
        if isinstance(item,np.generic):return item.item()
        if isinstance(item,Path):return str(item)
        raise TypeError(type(item).__name__)
    path=Path(path); temporary=path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=encode)+'\n')
    temporary.replace(path)


def save(path,value):
    temporary=path.with_suffix('.pending.pt')
    torch.save(value,temporary);temporary.replace(path)


def preserve(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('retained run identity changed')
    else:write(path,value)


def weighted_quantile(values,weights,quantile):
    order=np.argsort(values,kind='stable')
    cumulative=np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative,quantile*cumulative[-1],side='left')]])


def sample_record(rows,example):
    index,start,position=map(int,example); row=rows[index]
    return dict(triple=[index,start,position],dataset=row['dataset'],subject=row['subject'],record=row['record'],
                parent=row['parent'],position=row['positions'][position],split=row['split'],start_s=start)


def amplitudes(rows,examples,scales,out,role,cfg):
    destination=out/(role+'_amplitude.npz')
    if destination.exists():
        with np.load(destination) as retained:
            np.testing.assert_array_equal(retained['examples'],examples)
            return {key:retained[key].copy() for key in retained.files if key!='examples'}
    values=defaultdict(list);cache=OrderedDict();began=time.monotonic()
    for start in range(0,len(examples),cfg['resources']['batch_size']):
        take=examples[start:start+cfg['resources']['batch_size']]
        batch=base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=256)
        x=batch['x'].reshape(len(take),300,2).astype(np.float64)
        values['rms_by_color'].append(np.sqrt(np.mean(x*x,axis=1)))
        values['joint_rms'].append(np.sqrt(np.mean(x*x,axis=(1,2))))
        values['max_abs'].append(abs(x).max(axis=(1,2)))
        if start%4096==0:write(out/'progress.json',dict(stage='amplitude_scan',role=role,completed=start+len(take),planned=len(examples),seconds=time.monotonic()-began))
    values={key:np.concatenate(parts) for key,parts in values.items()}
    np.savez_compressed(destination,examples=examples,**values)
    return values


def inspect_source(rows,scales,example):
    from src.data.so_supervision_cache import SOSupervisionDataset
    item=sample_record(rows,example)
    if item['split'] not in ('train','validation'):raise ValueError('development source only')
    cache=ROOT/'data/cache/physiology_so_supervision_v3'
    reader=SOSupervisionDataset(cache,split=item['split'])
    selected=next(i for i,row in enumerate(reader.windows) if row['key']==item['parent'])
    metadata_path=Path(reader.windows[selected]['metadata']).resolve()
    if not metadata_path.is_relative_to(cache/'windows'):raise ValueError('source metadata path drift')
    meta=json.loads(metadata_path.read_text());array_path=Path(meta['array_file']).resolve()
    if array_path!=metadata_path.with_suffix('.npz') or not array_path.is_relative_to(cache/'windows'):raise ValueError('source array path drift')
    if (meta['subject']!=item['subject'] or meta['dataset']!=item['dataset'] or meta['split']!=item['split']):raise ValueError('source population drift')
    p=meta['fNIRS_positions'].index(item['position'])
    task_path=Path(meta['source_task_paths'][p]).resolve()
    if not task_path.is_relative_to(cache/'tasks'):raise ValueError('source task path drift')
    task=json.loads(task_path.read_text())
    if task['identity']['subject']!=item['subject'] or task['identity']['split']!=item['split']:raise ValueError('task identity drift')
    source=reader[selected]['arrays'];t=item['start_s']*10;sl=slice(t,t+300)
    row=rows[item['triple'][0]];position=item['triple'][2]
    for key in ('x','s','o'):
        prepared=np.load(Path(row['folder'])/(key+'.npy'),mmap_mode='r')[sl,position]
        np.testing.assert_array_equal(prepared,source['fnirs_'+key][sl,p].astype(np.float32))
    batch=base_tokenizer_batch(rows,[example],scales);mask=batch['mask'].reshape(300,2)
    factor=scales['positions']['hb'][item['dataset']][item['position']]['scale']
    details={}
    for key in 'xso':
        a=batch[key].reshape(300,2).astype(float)
        details[key]=dict(rms_input_sd=np.sqrt(np.mean(a*a,axis=0)),rms_native=np.sqrt(np.mean(a*a,axis=0))*factor,
                          bias_native=a.mean(0)*factor,peak_to_peak_native=np.ptp(a,axis=0)*factor)
    with np.load(task_path.with_suffix('.npz')) as retained:
        factor_subject=float(source['fnirs_scale'])
        np.testing.assert_allclose(source['fnirs_s'][:,p],retained['prediction'][:,1:]*factor_subject,rtol=1e-12,atol=1e-12)
        task_rms=np.sqrt(np.mean(retained['target']**2,axis=0))
    return dict(example=item,statistics=details,G0_training_scale=factor,scored_points_by_color=mask.sum(0),
                task=str(task_path),task_sha256=sha(task_path),source_visibility=task['identity'].get('visibility'),
                teacher_window_seconds=task['duration_s'],teacher_qualified=task['qualified'],
                teacher_Hb_not_better_than_zero=task.get('Hb_not_better_than_zero'),full_task_target_RMS_EEG_HbO_HbR=task_rms,
                parent_Hb_reference_unscaled=source['fnirs_reference_unscaled'][p],subject_Hb_factor=float(source['fnirs_scale']),
                max_native_X_S_O_closure=float(np.max(abs(batch['x']-batch['s']-batch['o'])))*factor,
                array_and_task_replay='passed',raw_dataset_validity_cause='not established by this cached-source inspection')


def fit_models(cfg,out,rows,scales,train,stats,source,signature):
    if (out/'fit.json').exists():return json.loads((out/'fit.json').read_text())
    weights=training_weights(train,rows)
    minimum=float(np.min(stats['joint_rms']))
    positive=stats['joint_rms'][stats['joint_rms']>0]
    if not len(positive):raise ValueError('nonzero training amplitude reference required')
    outside=.5*float(positive.min())
    tail=weighted_quantile(stats['joint_rms'],weights,cfg['fit']['tail_weighted_quantile'])
    selected=np.flatnonzero(stats['joint_rms']<=tail)
    numerator=np.zeros(2);denominator=np.zeros(2);cache=OrderedDict()
    for start in range(0,len(selected),64):
        ids=selected[start:start+64];batch=base_tokenizer_batch(rows,train[ids],scales,array_cache=cache)
        mask=batch['mask'].reshape(len(ids),300,2)
        count=mask.sum(1);coefficient=weights[ids,None]/count
        s,o=[batch[key].reshape(len(ids),300,2).astype(float) for key in ('s','o')]
        vs,vo=[batch['v'+key][:,0,:2] for key in ('s','o')]
        numerator+=(coefficient*np.where(mask,s/vs[:,None]-o/vo[:,None],0.).sum(1)).sum(0)
        denominator+=(weights[ids,None]*(1/vs+1/vo)).sum(0)
    level=np.divide(numerator,denominator)
    source_sample=source.decode_ids(source.encode(base_tokenizer_batch(rows,train[:1],scales)['x']))
    output_dtype=source_sample['S'].dtype.name
    base_size=int(source.metadata['codebook_size'])
    variants={}
    for name,threshold,atom in [('outside_support',outside,np.zeros(2)),('training_tail',tail,level)]:
        metadata=dict(source.metadata,schema='silent_observation_tokenizer_v1',source_file='source.pt',
            source_sha256=signature['source_sha256'],base_codebook_size=base_size,codebook_size=base_size+1,
            quiet_threshold_input_sd=threshold,quiet_threshold_rule=cfg['fit']['outside_support_threshold'] if name=='outside_support' else 'G0_train_weighted_quantile_0p001',
            output_dtype=output_dtype,variant=name,paired_atom='constant_S_level_O_negative_S',
            fit_signature=signature,integer_count_per_branch_position_window=15,context_s=30,
            fixed_length_bits_per_ID=int(np.ceil(np.log2(base_size+1))),bits_per_branch_position_per_second=float(np.ceil(np.log2(base_size+1))/2))
        checkpoint=out/(name+'.pt');save(checkpoint,dict(metadata=metadata,level=np.asarray(atom,dtype=output_dtype)))
        variants[name]=dict(checkpoint=str(checkpoint),threshold_input_sd=threshold,level_S=atom,level_O=-atom,
            amplitude_eligible_training_inputs=int((stats['joint_rms']<=threshold).sum()),
            amplitude_eligible_training_weight=float(weights[stats['joint_rms']<=threshold].sum()))
    fit=dict(status='completed_train_only_fit',source=signature,train_inputs=len(train),minimum_training_joint_RMS=minimum,
        lower_quantiles={str(q):weighted_quantile(stats['joint_rms'],weights,q) for q in (0.,.0001,.001,.01,.1)},
        low_tail_inputs=len(selected),low_tail_by_dataset=Counter(rows[train[i,0]]['dataset'] for i in selected),
        low_tail_subjects=len({(rows[train[i,0]]['dataset'],rows[train[i,0]]['subject']) for i in selected}),
        variants=variants,source_parameters=sum(p.numel() for p in source.model.parameters()),
        interface=dict(ordinary_codebook_size=base_size,new_codebook_size=base_size+1,integers_per_position_30s_both_branches=30,
            fixed_length_bits_per_ID=int(np.ceil(np.log2(base_size+1))),fixed_length_bits_per_second_both_branches=int(np.ceil(np.log2(base_size+1))),
            observation_gate_encoded_in_reserved_ID=True,decoder_extra_window_values=0))
    write(out/'fit.json',fit);return fit


def metric_inputs(rows,examples,scales):
    return np.array([scales['positions']['hb'][rows[i]['dataset']][rows[i]['positions'][p]]['scale'] for i,_,p in examples])


def accumulate(collection,audited):
    for key,value in audited['metrics'].items():collection[key].append(value)
    for key,value in audited['support'].items():collection['support_'+key].append(value)


def unpack(collection):
    values={key:np.concatenate(value) for key,value in collection.items()}
    return values,{key:value for key,value in values.items() if not key.startswith('support_')},{key[8:]:value for key,value in values.items() if key.startswith('support_')}


def paired_training(cfg,out,rows,scales,train,stats,source,models):
    path=out/'training_pairs.json'
    if path.exists():return
    tail=np.flatnonzero(stats['joint_rms']<=models['training_tail'].metadata['quiet_threshold_input_sd'])
    rng=np.random.default_rng(cfg['fit']['training_probe_seed'])
    normal=np.flatnonzero(stats['joint_rms']>models['training_tail'].metadata['quiet_threshold_input_sd'])
    random=rng.choice(normal,min(len(normal),cfg['fit']['training_probe_random']),replace=False)
    selected=np.sort(np.unique(np.r_[tail,random]));examples=train[selected]
    results={name:defaultdict(list) for name in ('source','outside_support','training_tail','forced_zero_pair','forced_train_mean_pair')}
    counts=Counter();cache=OrderedDict()
    for start in range(0,len(examples),cfg['resources']['batch_size']):
        take=examples[start:start+cfg['resources']['batch_size']];batch=base_tokenizer_batch(rows,take,scales,array_cache=cache)
        ids=source.encode(batch['x']);before=source.decode_ids(ids)
        predictions={'source':before}
        for name,model in models.items():
            current=model.select_ids(batch['x'],ids,before);after=model.decode_ids(current)
            quiet=(current['S']==model.silent_id).all(1);counts[name]+=int(quiet.sum())
            for key in ('S','O','sum'):np.testing.assert_array_equal(after[key][~quiet],before[key][~quiet])
            predictions[name]=after
        for name,variant in [('forced_zero_pair','outside_support'),('forced_train_mean_pair','training_tail')]:
            model=models[variant];forced={branch:np.full((len(take),15),model.silent_id,np.int64) for branch in ('S','O')}
            predictions[name]=model.decode_ids(forced)
        for name,prediction in predictions.items():accumulate(results[name],joint_reconstruction_metrics(prediction,batch,2,metric_inputs(rows,take,scales)))
    summaries={}
    for name,collection in results.items():
        values,metrics,support=unpack(collection)
        np.savez_compressed(out/('training_'+name+'_metrics.npz'),examples=examples,**values)
        summaries[name]=joint_reconstruction_summary(metrics,support,examples,rows)
    write(path,dict(status='completed',planned_all_train=len(train),paired_probe=len(examples),all_low_tail_inputs=len(tail),
                    random_normal_controls=len(random),changed_actual=counts,unchanged_outputs_bitwise_verified=True,summaries=summaries))


def development(cfg,out,rows,scales,examples,source,models):
    if (out/'development_comparison.json').exists():return
    results={name:defaultdict(list) for name in ('source',*models)}
    codes={name:{branch:[] for branch in ('S','O')} for name in models}
    changed={name:[] for name in models};cache=OrderedDict();began=time.monotonic()
    for start in range(0,len(examples),cfg['resources']['batch_size']):
        take=examples[start:start+cfg['resources']['batch_size']];batch=base_tokenizer_batch(rows,take,scales,array_cache=cache)
        ids=source.encode(batch['x']);before=source.decode_ids(ids)
        accumulate(results['source'],joint_reconstruction_metrics(before,batch,2,metric_inputs(rows,take,scales)))
        for name,model in models.items():
            current=model.select_ids(batch['x'],ids,before);after=model.decode_ids(current)
            quiet=(current['S']==model.silent_id).all(1);changed[name].append(quiet)
            for branch in ('S','O'):
                np.testing.assert_array_equal(current[branch][~quiet],ids[branch][~quiet]);codes[name][branch].append(current[branch])
            for key in ('S','O','sum'):np.testing.assert_array_equal(after[key][~quiet],before[key][~quiet])
            accumulate(results[name],joint_reconstruction_metrics(after,batch,2,metric_inputs(rows,take,scales)))
        if start%1024==0:write(out/'progress.json',dict(stage='development_paired_forward',completed=start+len(take),planned=len(examples),seconds=time.monotonic()-began))
    summaries={};all_metrics={};all_support={}
    for name,collection in results.items():
        values,metrics,support=unpack(collection);all_metrics[name]=metrics;all_support[name]=support
        np.savez_compressed(out/('development_'+name+'_metrics.npz'),examples=examples,**values)
        summaries[name]=joint_reconstruction_summary(metrics,support,examples,rows)
        if name in models:
            changed[name]=np.concatenate(changed[name]);code={branch:np.concatenate(value) for branch,value in codes[name].items()}
            np.savez_compressed(out/('development_'+name+'_ids.npz'),examples=examples,changed=changed[name],**code)
    replay={}
    for name,model in models.items():
        retained=load_model(out/(name+'.pt'));collection=defaultdict(list);cache=OrderedDict()
        with np.load(out/('development_'+name+'_ids.npz')) as stored:code={branch:stored[branch].copy() for branch in ('S','O')}
        for start in range(0,len(examples),cfg['resources']['batch_size']):
            take=examples[start:start+cfg['resources']['batch_size']];batch=base_tokenizer_batch(rows,take,scales,array_cache=cache)
            predicted=retained.decode_ids({branch:value[start:start+len(take)] for branch,value in code.items()})
            accumulate(collection,joint_reconstruction_metrics(predicted,batch,2,metric_inputs(rows,take,scales)))
        _,metrics,support=unpack(collection)
        for key in metrics:np.testing.assert_array_equal(metrics[key],all_metrics[name][key])
        for key in support:np.testing.assert_array_equal(support[key],all_support[name][key])
        replay[name]=dict(status='passed',independent_model_reload=True,metrics_bitwise_equal=True)
    effects={}
    for name,mask in changed.items():
        groups=Counter(rows[examples[i,0]]['dataset'] for i in np.flatnonzero(mask))
        effects[name]=dict(changed_inputs=int(mask.sum()),planned_inputs=len(examples),changed_by_dataset=groups,
            unchanged_inputs=int((~mask).sum()),unchanged_predictions_and_IDs_bitwise_equal=True,
            changed_examples=[sample_record(rows,examples[i]) for i in np.flatnonzero(mask)],
            integer_storage_bytes=(out/('development_'+name+'_ids.npz')).stat().st_size,
            changed_metrics={key:joint_reconstruction_summary({key:value[mask]},all_support[name],examples,rows)
                for key,value in []})
    write(out/'development_comparison.json',dict(status='completed',population='entire_frozen_development_panel_core_and_tail',
        planned_inputs=len(examples),summaries=summaries,effects=effects,replay=replay,confirmation_arrays_read=0,test_arrays_read=0))


def run(args,cfg):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest,scales,_,_=base_tokenizer_training_data(cfg,ROOT);rows=manifest['rows']
    panel_path=ROOT/cfg['panel']['file'];panel_record=json.loads((panel_path.parent/'panel.json').read_text())
    if sha(panel_path)!=panel_record['panel_sha256'] or manifest['scales_identity']!=panel_record['scales_identity']:raise ValueError('frozen development panel identity drift')
    with np.load(panel_path) as archive:dev=archive[cfg['panel']['role']].copy()
    train=np.asarray(base_tokenizer_examples(rows,'hb','train'),np.int32)
    for index,_,_ in dev:
        row=rows[index]
        if row['split']!='validation' or row['subject']==panel_record['confirmation_subjects'][row['dataset']]:raise ValueError('confirmation/test excluded before arrays')
    source_path=ROOT/cfg['source']['checkpoint']
    signature=dict(config=cfg,source_sha256=sha(source_path),manifest_sha256=sha(ROOT/cfg['data']['run']/'cache/manifest.json'),
                   scales_sha256=sha(ROOT/cfg['data']['run']/'scales.json'),panel_sha256=sha(panel_path))
    preserve(out/'resolved_config.json',signature)
    if (out/'summary.json').exists():return
    stats=amplitudes(rows,train,scales,out,'train',cfg);dev_stats=amplitudes(rows,dev,scales,out,'development',cfg)
    source_copy=out/'source.pt'
    if source_copy.exists():
        if sha(source_copy)!=signature['source_sha256']:raise ValueError('retained source drift')
    else:shutil.copy(source_path,source_copy)
    source=load_contextual(source_copy)
    inspection=dict(target=inspect_source(rows,scales,[5501,46,36]),
        related_zero_case=inspect_source(rows,scales,[5515,0,36]),
        train_minimum=float(stats['joint_rms'].min()),development_minimum=float(dev_stats['joint_rms'].min()),
        train_below_target_amplitude=int((stats['joint_rms']<=dev_stats['joint_rms'][np.flatnonzero(np.all(dev==[5501,46,36],axis=1))[0]]).sum()))
    write(out/'source_inspection.json',inspection)
    if args.stage=='amplitudes':return
    fit=fit_models(cfg,out,rows,scales,train,stats,source,signature)
    models={name:load_model(out/(name+'.pt')) for name in fit['variants']}
    paired_training(cfg,out,rows,scales,train,stats,source,models)
    if args.stage=='fit':return
    development(cfg,out,rows,scales,dev,source,models)
    write(out/'summary.json',dict(status='completed',fit='fit.json',training='training_pairs.json',development='development_comparison.json',
        source_inspection='source_inspection.json',no_new_model_training=True,confirmation_arrays_read=0,test_arrays_read=0))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',default='experiments/configs/physiology_semantic_tokenizer/dual_component_hb_silent_atom_v1.yaml')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--stage',choices=['amplitudes','fit','all'],default='all')
    args=parser.parse_args();cfg=yaml.safe_load((ROOT/args.config).read_text())
    if cfg['schema']!='dual_component_hb_silent_atom_v1' or cfg['panel']['role']!='hb_development':raise ValueError('versioned train-fit and development-only contract required')
    torch.set_num_threads(cfg['resources']['torch_threads'])
    with owner_lock(args.output),threadpool_limits(limits=cfg['resources']['numerical_threads']):run(args,cfg)


if __name__=='__main__':main()
