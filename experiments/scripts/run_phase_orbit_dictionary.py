"""Train-only four-fold ordinary versus explicit shift/sign carrier atoms."""
from __future__ import annotations
import argparse
from collections import OrderedDict, defaultdict
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/scripts'))
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
from src.data.dual_component_tokenizer import base_tokenizer_training_data,base_tokenizer_batch,base_tokenizer_joint_eeg
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics,joint_reconstruction_summary
from src.analysis.tokenizer_pattern_audit import attributes
from src.tokenizers.anchored_spatial_carrier import AnchoredSpatialCarrier
from src.tokenizers.phase_orbit_dictionary import PhaseOrbitDictionary,fit_book,load_model,waveform_features
from src.tokenizers.spatial_waveform_dictionary import spectral_features
from evaluate_joint_tokenizer import input_scales,write
# These public evidence routines retain the exact prior categorical background,
# support rule and hierarchy, including the fully expanded band+atom identity.
from run_anchored_carrier_dictionary import pattern_fold,pattern_summary,FIELDS,rank


def validate(cfg):
    d,m=cfg['data'],cfg['model']
    if (cfg['schema']!='phase_orbit_experiment_v1' or d['fitting_splits']!=['train'] or d['evaluation_splits']!=['train']
        or any(d[k]!='forbidden' for k in ('validation_access','confirmation_access','test_access','protected_data'))
        or m['positive_atoms']!=m['parent_shapes']*m['shifts_per_parent']*2
        or m['vocabulary_including_missing']!=m['positive_atoms']+1 or m['vocabulary_including_missing']>1024
        or m['context_s']!=30 or m['token_s']!=2 or m['eeg_rate_hz']!=200
        or m['continuous_decoder_side_information'] or m['auxiliary_properties_in_assignment']!='none'):
        raise ValueError('train-only explicit K<=1024 native-voltage single-integer orbit contract required')


def context(cfg):
    g0=yaml.safe_load((ROOT/cfg['source']['g0_config']).read_text())
    manifest,scales,_,_=base_tokenizer_training_data(g0,ROOT)
    basis=ROOT/cfg['source']['basis_run']
    bc=yaml.safe_load((ROOT/cfg['source']['basis_config']).read_text())
    plan=json.loads((ROOT/bc['source']['native_train_plan']).read_text())['plan']
    folds=json.loads((basis/'fold_subjects.json').read_text())
    rows=manifest['rows']
    for dataset,people in plan.items():
        for subject,crops in people.items():
            if any(rows[c['row']]['split']!='train' or rows[c['row']]['subject']!=subject or rows[c['row']]['dataset']!=dataset for c in crops):
                raise ValueError('original train source identity before arrays')
    if sum(len(v) for p in plan.values() for v in p.values())!=400 or sum(map(len,plan.values()))!=60:
        raise ValueError('400 crops and 60 original training subjects required')
    sites=sorted({(r['dataset'],p) for r in rows if r['split']=='train' and r['modality']=='eeg' for p in r['positions']})
    return manifest,scales,rows,plan,folds,{v:i for i,v in enumerate(sites)}


def make_cases(source,rows,scales,plan,folds,site_index,out):
    cache=OrderedDict(); cases=[]
    for dataset,people in sorted(plan.items()):
        names=source.metadata['positions'][dataset];lookup={v.upper():i for i,v in enumerate(names)}
        roles=source.roles(dataset)
        for subject,crops in sorted(people.items()):
            for crop in crops:
                fields,observed,_=base_tokenizer_joint_eeg(rows,dict(crop,positions=[]),scales,names,cache)
                physical=source.continuous_source(fields['x'].T.reshape(1,len(names),15,400))
                carriers=source.analyze(physical,dataset,observed[None])
                ex=np.asarray([(crop['row'],crop['start_s'],p) for p in crop['positions']],np.int32)
                batch=base_tokenizer_batch(rows,ex,scales,array_cache=cache,cache_limit=16)
                # All variance fields are constant within a channel. Preserve
                # the exact owning values without thousands of repeated copies.
                batch={k:(v[:,:1,:1].copy() if k.startswith(('v','dv')) else v) for k,v in batch.items()}
                native=np.asarray([lookup[rows[crop['row']]['positions'][p].upper()] for p in crop['positions']])
                complete=batch['mask'].all(-1); target={}
                for b in 'SO':
                    raw,slices=attributes(batch[b.lower()][complete],'eeg')
                    a=np.zeros((*complete.shape,10))
                    a[complete]=np.concatenate([raw[:,slice(*slices[k])] for k in FIELDS],axis=1)
                    target[b]=a
                site=np.asarray([site_index[dataset,rows[crop['row']]['positions'][p]] for p in crop['positions']])
                background=4*site[:,None]+np.searchsorted([.5,1.,2.],np.sqrt(np.mean(batch['x']**2,axis=-1)),side='right')
                cases.append(dict(dataset=dataset,subject=subject,fold=folds[dataset][subject],crop=crop,observed=observed[None],
                    carriers=carriers,examples=ex,native=native,complete=complete,target=target,background=background,
                    batch=batch,scales=input_scales(rows,ex,scales),roles=roles))
                if len(cases)%40==0:write(out/'progress.json',dict(stage='source_carriers',crops=len(cases),CPU_seconds=time.process_time()))
    return cases


def training_patches(cases,fit,branch,band,cfg):
    grouped=defaultdict(list)
    for i in fit:
        c=cases[i];positions=np.flatnonzero(c['roles'][c['native']]==band)
        sample,patch=np.nonzero(c['complete'][positions])
        grouped[c['dataset'],c['subject']].extend((i,int(c['native'][p]),int(t)) for p,t in zip(positions[sample],patch))
    rng=np.random.default_rng(cfg['training']['seed']+band+100*('SO'.index(branch)))
    picked=[]
    for key,items in sorted(grouped.items()):
        take=rng.choice(len(items),min(cfg['training']['book_fit_max_patches_per_subject_branch_band'],len(items)),replace=False)
        picked.extend(items[j] for j in take)
    return np.stack([cases[i]['carriers'][branch][0,n,t] for i,n,t in picked]),rank(picked)


def fit_models(cfg,out,fold,source,cases,fit,source_path):
    books={a:{} for a in ('ordinary','orbit')};reports={a:{} for a in books}
    folder=out/'books';folder.mkdir(exist_ok=True)
    for b in 'SO':
        for k in range(5):
            data,sample_identity=training_patches(cases,fit,b,k,cfg)
            shifts=(np.arange(16)*cfg['model']['shift_step_samples_by_band'][k]).tolist()
            for method in books:
                path=folder/f'{method}_{b}_{k}.npz';report_path=folder/f'{method}_{b}_{k}.json'
                if path.exists() and report_path.exists():
                    report=json.loads(report_path.read_text())
                    if report['sample_identity']!=sample_identity or report['config_identity']!=rank(cfg):raise ValueError('book resume identity mismatch')
                    with np.load(path) as f:book={key:f[key].copy() for key in f.files}
                else:
                    started=time.process_time()
                    book,report=fit_book(data,k,method=method,parent_count=cfg['model']['parent_shapes'],shifts=shifts,
                        iterations=cfg['training']['lloyd_iterations'],seed=cfg['training']['seed']+k)
                    report.update(sample_identity=sample_identity,config_identity=rank(cfg),CPU_seconds=time.process_time()-started)
                    np.savez_compressed(path,**book);write(report_path,report)
                books[method][b,k]=book;reports[method][f'{b}_{k}']=report
                write(out/'progress.json',dict(stage='book_fit',fold=fold,branch=b,band=k,method=method,CPU_seconds=time.process_time(),last_book=report['CPU_seconds']))
    models={}
    for method in books:
        metadata=dict(schema='anchored_carrier_dictionary_v1',experiment_family='phase_orbit_dictionary_v1',method=method,
            codebook_size=cfg['model']['vocabulary_including_missing'],held_fold=fold,seed=cfg['training']['seed'],
            fitting_split='train',integers_per_branch_position_context=15,continuous_decoder_side_information=0,
            parent_ID_used_for_pattern=False,config_identity=rank(cfg))
        models[method]=PhaseOrbitDictionary(source,books[method],metadata)
        models[method].save(out/f'{method}.npz',source_path)
        write(out/f'{method}_fit.json',reports[method])
    return models


def aggregate_audits(records,examples,rows,out,name):
    metrics={k:np.concatenate([v['metrics'][k] for v in records]) for k in records[0]['metrics']}
    support={k:np.concatenate([v['support'][k] for v in records]) for k in records[0]['support']}
    np.savez_compressed(out/f'{name}_metrics.npz',examples=examples,**metrics,**{'support_'+k:v for k,v in support.items()})
    summary=joint_reconstruction_summary(metrics,support,examples.tolist(),rows)
    write(out/f'{name}_summary.json',summary)
    return summary


def usage(models,cases,codes,fit,held,out):
    rows=[]
    for method,model in models.items():
        for b in 'SO':
            for band in range(5):
                count=np.zeros((2,model.codebook_size),np.int64);subjects=defaultdict(set)
                for role,indices in enumerate((fit,held)):
                    for i in indices:
                        c=cases[i]; keep=c['complete'] & (c['roles'][c['native']][:,None]==band)
                        ids=codes[method][i][b][0,c['native']][keep]
                        count[role]+=np.bincount(ids,minlength=model.codebook_size)
                        if role==0:
                            for code in np.unique(ids):subjects[int(code)].add((c['dataset'],c['subject']))
                for code in range(model.codebook_size):
                    rows.append(dict(method=method,branch=b,band=band,final_atom_ID=code,parent_ID_for_audit='never_used',
                        fit_complete_patches=int(count[0,code]),fit_subjects=len(subjects[code]),held_complete_patches=int(count[1,code]),
                        supported=bool(count[0,code]>=8 and len(subjects[code])>=2)))
    with (out/'all_atom_support.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def score(cfg,out,fold,models,cases,fit,held,rows,site_count):
    codes={a:[m.assign(c['carriers'],None,c['dataset'],c['observed']) for c in cases] for a,m in models.items()}
    pattern={}
    for method,model in models.items():
        ex,err,support=pattern_fold(model,cases,codes[method],fit,held,4*site_count,sorted(source_datasets(cases)))
        np.savez_compressed(out/f'{method}_pattern.npz',examples=ex,S=err['S'],O=err['O'],support=np.asarray([support[b] for b in 'SO']))
        pattern[method]=pattern_summary(err,ex,rows,support)
        archive={f'case{i}_{b}':codes[method][i][b] for i in held for b in 'SO'}
        np.savez_compressed(out/f'{method}_held_IDs.npz',**archive)
    usage(models,cases,codes,fit,held,out)
    replay={a:load_model(out/f'{a}.npz') for a in models}; disk={}
    for method,m in replay.items():
        m.source.continuous_source=lambda *_: (_ for _ in ()).throw(AssertionError('integer decoder requested X'))
        m.source.analyze=lambda *_: (_ for _ in ()).throw(AssertionError('integer decoder requested continuous carrier'))
        with np.load(out/f'{method}_held_IDs.npz') as f:disk[method]={key:f[key].copy() for key in f.files}
    records={a:[] for a in ('ordinary','orbit','continuous','projected_continuous')};examples=[];checks=[]
    plot_records=[];plot_arrays={}; hardest=(-np.inf,None)
    fixed={}
    if fold==0:
        for d in source_datasets(cases):
            subject=min(c['subject'] for i,c in enumerate(cases) if i in held and c['dataset']==d)
            fixed[d]=min((i for i in held if cases[i]['dataset']==d and cases[i]['subject']==subject),key=lambda i:rank(cases[i]['crop']))
    for i in held:
        c=cases[i];source=models['ordinary'].source
        prediction={}
        for method,model in models.items():
            ids={b:disk[method][f'case{i}_{b}'] for b in 'SO'}
            expected=model.decode_ids(codes[method][i],c['dataset']);actual=replay[method].decode_ids(ids,c['dataset'])
            maximum=max(float(np.abs(actual[b]-expected[b]).max()) for b in ('S','O','sum'))
            if maximum!=0:raise AssertionError('saved integers did not replay exactly')
            prediction[method]=actual
            checks.append(dict(method=method,case=i,fold=fold,dataset=c['dataset'],subject=c['subject'],crop=c['crop'],
                integer_count=sum(v.size for v in ids.values()),max_abs=maximum,decoder_continuous_calls=0))
        prediction['continuous']=source.synthesize(c['carriers'],c['dataset'])
        projected={b:np.zeros_like(c['carriers'][b]) for b in 'SO'}
        for b in 'SO':
            for k in range(5):
                selected=np.flatnonzero(c['roles']==k)
                q=spectral_features(c['carriers'][b][:,selected].reshape(-1,400),k)
                projected[b][:,selected]=waveform_features(q,k).reshape(1,len(selected),15,400)
        prediction['projected_continuous']=source.synthesize(projected,c['dataset'])
        audited={}
        for method,pred in prediction.items():
            audited[method]=joint_reconstruction_metrics({b:pred[b][0,c['native']] for b in ('sum','S','O')},c['batch'],1,c['scales'])
            records[method].append(audited[method])
        examples.extend(c['examples'])
        error=audited['ordinary']['metrics']['nmse'][:,:,0].mean(1)
        if error.max()>hardest[0]:hardest=(float(error.max()),(i,int(error.argmax()),prediction))
        if i in fixed.values():
            errors=audited['ordinary']['metrics']['nmse'][:,1,0]
            for label,p in [('median',int(np.argsort(errors)[len(errors)//2])),('hard',int(errors.argmax()))]:
                retain_plot(plot_records,plot_arrays,c,i,p,label,prediction,rows)
        write(out/'progress.json',dict(stage='full_held_integer_replay_and_metrics',held_crops=len(records['ordinary']),CPU_seconds=time.process_time()))
    i,p,pred=hardest[1];retain_plot(plot_records,plot_arrays,cases[i],i,p,'fold_overall_hardest',pred,rows,hardness=hardest[0])
    np.savez_compressed(out/'plot_arrays.npz',**plot_arrays);write(out/'plot_records.json',plot_records)
    summaries={a:aggregate_audits(v,np.asarray(examples),rows,out,a) for a,v in records.items()}
    write(out/'replay.json',dict(status='passed',checks=checks,independent_model_reload=True,saved_integer_arrays=True,maximum_absolute_difference=0))
    write(out/'fold_summary.json',dict(fold=fold,status='completed',patterns=pattern,
        waveform_NMSE={a:s['metrics']['nmse'] for a,s in summaries.items()},CPU_seconds=time.process_time(),max_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def source_datasets(cases):return sorted({c['dataset'] for c in cases})


def pack_integer_payload(codes):
    values=np.asarray(codes)
    if values.ndim!=1 or values.dtype.kind not in 'iu' or np.any(values<0) or np.any(values>=993):
        raise ValueError('one-dimensional in-vocabulary integer payload required')
    bits=((values.astype(np.uint16)[:,None]>>np.arange(10))&1).astype(np.uint8)
    return np.packbits(bits.reshape(-1),bitorder='little').tobytes()


def unpack_integer_payload(payload,count):
    if len(payload)!=(count*10+7)//8:raise ValueError('exact declared payload size required')
    bits=np.unpackbits(np.frombuffer(payload,np.uint8),bitorder='little')
    if np.any(bits[count*10:]):raise ValueError('nonzero undeclared padding bits')
    values=(bits[:count*10].reshape(count,10)*2**np.arange(10)).sum(1).astype(np.int32)
    if np.any(values>=993):raise ValueError('out-of-vocabulary integer payload')
    return values


def export_packed(cfg,out):
    records=[]
    for fold in range(4):
        for method in ('ordinary','orbit'):
            job=out/f'fold{fold}'
            with np.load(job/f'{method}_held_IDs.npz') as f:
                values={k:f[k].copy() for k in sorted(f.files)}
            index=[];offset=0
            for key,value in values.items():
                index.append(dict(key=key,shape=list(value.shape),integer_offset=offset,count=value.size));offset+=value.size
            flat=np.concatenate([v.reshape(-1) for v in values.values()])
            path=job/f'{method}_held_IDs10.bin';path.write_bytes(pack_integer_payload(flat))
            restored=unpack_integer_payload(path.read_bytes(),len(flat))
            np.testing.assert_array_equal(flat,restored)
            write(job/f'{method}_held_IDs10_index.json',dict(schema='fixed_10bit_EEG_atom_IDs_v1',bitorder='little',bits_per_ID=10,
                vocabulary_including_missing=993,count=len(flat),index=index,metadata='only fixed integer array shapes and original case identities; no signal values'))
            records.append(dict(fold=fold,method=method,integers=len(flat),payload_bits=path.stat().st_size*8,
                padding_bits=path.stat().st_size*8-len(flat)*10,ID_max_abs_difference=0,
                all_array_shapes_and_IDs_equal_to_fully_replayed_NPZ=True,index_bytes=(job/f'{method}_held_IDs10_index.json').stat().st_size))
    write(out/'integer_transport_v1.json',dict(status='passed',records=records,bits_per_ID=10,
        EEG_S_O_payload_bits_per_position_second={a:sum(v['payload_bits'] for v in records if v['method']==a)/(17040*30) for a in ('ordinary','orbit')},
        interpretation='packed payload plus explicitly reported fixed-layout audit index; every expanded shift/sign is a distinct paid ID'))


def retain_plot(records,arrays,c,i,p,label,predictions,rows,hardness=None):
    key=f'{i}_{p}_{label}'
    if any(v['key']==key for v in records):return
    n=c['native'][p]
    records.append(dict(key=key,dataset=c['dataset'],subject=c['subject'],crop=c['crop'],position=rows[c['crop']['row']]['positions'][c['crop']['positions'][p]],
        source_unit=rows[c['crop']['row']].get('source_unit','native'),scale=float(c['scales'][p]),label=label,hardness=hardness))
    for field in ('x','s','o','mask'):arrays[key+'__target_'+field]=c['batch'][field][p]
    for method,pred in predictions.items():
        for b in ('sum','S','O'):arrays[key+'__'+method+'_'+b]=pred[b][0,n]


def worker(cfg,out,fold):
    job=out/f'fold{fold}';job.mkdir(parents=True,exist_ok=True)
    lock=(job/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (job/'fold_summary.json').exists():raise FileExistsError('completed fold is retained')
    limit=cfg['resources']['CPU_budget_seconds_per_fold'];resource.setrlimit(resource.RLIMIT_CPU,(limit-10,limit))
    manifest,scales,rows,plan,folds,sites=context(cfg)
    source_path=ROOT/cfg['source']['basis_run']/f'fold_model_{fold}.npz';source=AnchoredSpatialCarrier.load(source_path)
    if source.metadata['input_identity']!=manifest['cfg_identity'] or source.metadata['scales_identity']!=manifest['scales_identity']:raise ValueError('source/G0 identity')
    write(out/'resolved_config.json',cfg)
    cases=make_cases(source,rows,scales,plan,folds,sites,job)
    fit=[i for i,c in enumerate(cases) if c['fold']!=fold];held=[i for i,c in enumerate(cases) if c['fold']==fold]
    models=fit_models(cfg,job,fold,source,cases,fit,source_path)
    score(cfg,job,fold,models,cases,fit,held,rows,len(sites))


def merge(cfg,out):
    manifest,_,rows,plan,folds,_=context(cfg)
    summary={};patterns={};examples=None
    for method in ('ordinary','orbit','continuous','projected_continuous'):
        data=[]
        for fold in range(4):
            with np.load(out/f'fold{fold}'/f'{method}_metrics.npz') as f:data.append({k:f[k].copy() for k in f.files})
        examples=np.concatenate([v['examples'] for v in data])
        metrics={k:np.concatenate([v[k] for v in data]) for k in data[0] if k!='examples' and not k.startswith('support_')}
        support={k[8:]:np.concatenate([v[k] for v in data]) for k in data[0] if k.startswith('support_')}
        if len(examples)!=17040 or len(set(map(tuple,examples)))!=17040:raise ValueError('original full position-window denominator')
        summary[method]=joint_reconstruction_summary(metrics,support,examples.tolist(),rows)
        np.savez_compressed(out/f'{method}_full_metrics.npz',examples=examples,**metrics,**{'support_'+k:v for k,v in support.items()})
        write(out/f'{method}_full_summary.json',summary[method])
    for method in ('ordinary','orbit'):
        ex=[];errors={b:[] for b in 'SO'};support={b:np.zeros(2,np.int64) for b in 'SO'}
        for fold in range(4):
            with np.load(out/f'fold{fold}'/f'{method}_pattern.npz') as f:
                ex.append(f['examples'])
                for j,b in enumerate('SO'):errors[b].append(f[b]);support[b]+=f['support'][j]
        patterns[method]=pattern_summary({b:np.concatenate(v) for b,v in errors.items()},np.concatenate(ex),rows,support)
    ordinary,orbit,continuous=[np.asarray(summary[a]['metrics']['nmse']['macro']).reshape(3) for a in ('ordinary','orbit','continuous')]
    recovered=(ordinary.mean()-orbit.mean())/(ordinary.mean()-continuous.mean())
    projected=np.asarray(summary['projected_continuous']['metrics']['nmse']['macro']).reshape(3)
    recovered_projected=(ordinary.mean()-orbit.mean())/(ordinary.mean()-projected.mean())
    improving_datasets=sum(np.asarray(summary['orbit']['metrics']['nmse']['by_dataset'][d]).mean()<np.asarray(v).mean()
                          for d,v in summary['ordinary']['metrics']['nmse']['by_dataset'].items())
    main_cancellation_support=bool(recovered>=.5 and np.all(orbit[1:]<ordinary[1:]) and improving_datasets>=3)
    selection=cfg['selection'];guards={}
    for method in ('ordinary','orbit'):
        nmse=np.asarray(summary[method]['metrics']['nmse']['macro']).reshape(3)
        guards[method]=dict(waveform=bool(np.all(nmse<=selection['waveform_component_max_ratio_to_matched_continuous']*continuous)),
            dataset_waveform=all(np.all(np.asarray(v)<=selection['waveform_dataset_component_max_ratio_to_matched_continuous']*np.asarray(summary['continuous']['metrics']['nmse']['by_dataset'][d])) for d,v in summary[method]['metrics']['nmse']['by_dataset'].items()),
            pattern=all(v['support_mass']>=selection['support_mass_min'] and min(v['gain'][k] for k in ('power_4_8','power_8_13'))>0
                and max(v['gain'].values())>=selection['one_gain_min'] and min(v['positive_subject_fraction'][k] for k in ('power_4_8','power_8_13'))>=selection['positive_subject_fraction_theta_alpha_min'] for v in patterns[method].values()))
    # Independently verify every continuous triple against the retained original
    # spatial source evidence instead of trusting its prose summary.
    with np.load(out/'continuous_full_metrics.npz') as f:actual={tuple(ex):v for ex,v in zip(f['examples'],f['nmse'])}
    source_difference=0.
    for d in sorted(plan):
        with np.load(ROOT/cfg['source']['basis_run']/f'{d}_held.npz') as f:
            for ex,value in zip(f['examples'],f['anchored__nmse']):source_difference=max(source_difference,float(abs(actual[tuple(ex)]-value).max()))
    if source_difference>1e-12:raise AssertionError('matching source continuous denominator or input changed')
    counts={method:0 for method in ('ordinary','orbit')};storage={method:0 for method in counts};parameters={method:0 for method in counts}
    usage_summary=[]
    for fold in range(4):
        replay=json.loads((out/f'fold{fold}'/'replay.json').read_text())
        for row in replay['checks']:counts[row['method']]+=row['integer_count']
        for method in counts:
            storage[method]+=(out/f'fold{fold}'/f'{method}_held_IDs.npz').stat().st_size*8
            model=load_model(out/f'fold{fold}'/f'{method}.npz')
            source_values=list(model.source.analysis.values())+list(model.source.synthesis.values())+list(model.source.transfer.values())
            parameters[method]=sum(v.size*(2 if np.iscomplexobj(v) else 1) for v in source_values)+sum(v.size for book in model.books.values() for key,v in book.items() if key in ('mean','weight','centers','atoms'))
        rows_usage=list(csv.DictReader((out/f'fold{fold}'/'all_atom_support.csv').open()))
        for method in counts:
            for b in 'SO':
                for band in range(5):
                    take=[r for r in rows_usage if r['method']==method and r['branch']==b and int(r['band'])==band and int(r['final_atom_ID'])>0]
                    n=np.asarray([int(r['fit_complete_patches']) for r in take]);p=n/max(n.sum(),1);occupied=p>0
                    usage_summary.append(dict(fold=fold,method=method,branch=b,band=band,normal_IDs=992,occupied=int(occupied.sum()),dead=int((~occupied).sum()),
                        empirical_conditional_entropy_bits=float(-(p[occupied]*np.log2(p[occupied])).sum()),fit_patches=int(n.sum())))
    write(out/'all_book_usage.json',usage_summary)
    write(out/'summary.json',dict(status='completed',NMSE_X_S_O={a:np.asarray(v['metrics']['nmse']['macro']).reshape(3) for a,v in summary.items()},
        pattern=patterns,guards=guards,any_candidate_eligible=any(all(v.values()) for v in guards.values()),
        phase_excess_macro_recovered_fraction=recovered,projected_continuous_excess_recovered_fraction=recovered_projected,
        improving_datasets_macro=improving_datasets,main_phase_cancellation_explanation_supported=main_cancellation_support,
        source_continuous_max_abs_NMSE_difference=source_difference,
        original_position_windows=17040,physical_crops=400,subjects=60,folds=4,validation_arrays_read=0,
        integers_per_branch_position_context=15,actual_integers_per_method=counts,integer_NPZ_storage_bits_per_method=storage,
        inference_real_scalars_per_fold_model=parameters,vocabulary_including_missing=993,
        fixed_bits_per_ID=10,EEG_S_O_bits_per_position_second=10,parent_ID_used_for_semantics=False,
        source_exposure='old Wiener/scales all60train; basis/dictionary/background readouts exclude corresponding held fold'))


def render(cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[]
    for fold in range(4):
        values=json.loads((out/f'fold{fold}'/'plot_records.json').read_text())
        selected.extend((fold,v) for v in values if v['label']!='fold_overall_hardest')
    hardest=[]
    for fold in range(4):
        hardest.extend((fold,v) for v in json.loads((out/f'fold{fold}'/'plot_records.json').read_text()) if v['label']=='fold_overall_hardest')
    selected.append(max(hardest,key=lambda v:v[1]['hardness']))
    folder=out/'waveforms';folder.mkdir(exist_ok=True);records=[]
    for fold,item in selected:
        with np.load(out/f'fold{fold}'/'plot_arrays.npz') as f:values={k:f[k].copy() for k in f.files if k.startswith(item['key']+'__')}
        key=item['key'];scale=item['scale'];t=np.arange(6000)/200
        valid=values[key+'__target_mask'].reshape(6000)
        fig,axes=plt.subplots(3,3,figsize=(17,8.4))
        for col,(b,target) in enumerate((('sum','x'),('S','s'),('O','o'))):
            y=values[key+'__target_'+target].reshape(6000)*scale
            for row,(start,end) in enumerate(((0,30),(14,16),(13.5,14.5))):
                ax=axes[row,col];take=(t>=start)&(t<end)
                ax.plot(t[take],y[take],c='black',lw=.65,label='target')
                for name,color in [('continuous','#359264'),('ordinary','#e28d33'),('orbit','#217ac0')]:
                    p=values[key+'__'+name+'_'+b].reshape(6000)*scale
                    ax.plot(t[take],p[take],color=color,lw=.65,alpha=.8,label=name)
                for edge in np.arange(0,31,2):
                    if start<=edge<=end:ax.axvline(edge,color='gray',lw=.5,alpha=.4)
                ax.fill_between(t[take],0,1,where=~valid[take],transform=ax.get_xaxis_transform(),color='gray',alpha=.2)
                ax.set(xlim=(start,end),xlabel='time(s)',ylabel=item['source_unit'],title=b)
                if row==0:ax.legend(fontsize=7)
        fig.suptitle(f'{item["dataset"]} / {item["subject"]} / row{item["crop"]["row"]}+{item["crop"]["start_s"]}s / {item["position"]} / {item["label"]}; same axes, gray=unscored')
        fig.tight_layout();path=folder/f'{item["dataset"]}_{item["label"]}.png';fig.savefig(path,dpi=220);plt.close(fig)
        records.append(dict(item,path=str(path.relative_to(ROOT)),fold=fold,status='rendered_awaiting_actual_visual_review'))
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained');x=np.arange(5);labels=['1-4','4-8','8-13','13-30','30-45']
    for col,b in enumerate('SO'):
        fields={a:[json.loads((out/f'fold{fold}'/f'{a}_fit.json').read_text()) for fold in range(4)] for a in ('ordinary','orbit')}
        def average(method,k,name):
            values=[v[f'{b}_{k}'] for v in fields[method]]
            return np.average([v[name] for v in values],weights=[v['training_patches'] for v in values])
        aligned=[average('orbit',k,'aligned_parent_mean_energy_fraction') for k in range(5)]
        unaligned=[average('orbit',k,'unaligned_same_parent_mean_energy_fraction') for k in range(5)]
        axes[0,col].bar(x-.18,unaligned,.36,label='unaligned same-parent mean',color='#b1b1b1')
        axes[0,col].bar(x+.18,aligned,.36,label='shift/sign aligned parent mean',color='#3382b4')
        axes[0,col].set(title=b+' training parent means',ylabel='mean energy / member energy',xticks=x,xticklabels=labels,ylim=(0,1));axes[0,col].legend(fontsize=8)
        for method,color,offset in [('ordinary','#e28d33',-.18),('orbit','#217ac0',.18)]:
            axes[1,col].bar(x+offset,[average(method,k,'final_projected_MSE') for k in range(5)],.36,label=method,color=color)
        axes[1,col].set(ylabel='training projected carrier MSE (input SD squared)',xlabel='band (Hz)',xticks=x,xticklabels=labels);axes[1,col].legend(fontsize=8)
    fig.suptitle('Parent alignment diagnosis; waveform/pattern verdict uses complete expanded IDs')
    mechanism=folder/'phase_cancellation.png';fig.savefig(mechanism,dpi=220);plt.close(fig)
    write(folder/'manifest.json',dict(figures=records,mechanism_figure=str(mechanism.relative_to(ROOT))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--mode',choices=('fold','merge','render','pack'),required=True);p.add_argument('--fold',type=int)
    args=p.parse_args();cfg=yaml.safe_load(args.config.read_text());validate(cfg);out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(cfg['resources']['numerical_threads_per_fold']):
        if args.mode=='fold':
            if args.fold not in range(4):raise ValueError('original fold required')
            worker(cfg,out,args.fold)
        elif args.mode=='merge':merge(cfg,out)
        elif args.mode=='render':render(cfg,out)
        else:export_packed(cfg,out)


if __name__=='__main__':main()
