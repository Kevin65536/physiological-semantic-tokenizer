"""Paired old/new X-only sources with newly fitted hard-stratified single atoms."""
import argparse
from collections import OrderedDict,defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/scripts'))
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
from src.tokenizers.stratified_source_dictionary import StratifiedSourceDictionary,fit_book,load_model,project,solve_readout,pack_ids,unpack_ids
from src.tokenizers.spatial_waveform_dictionary import load_model as load_wiener
from src.tokenizers.eeg_transient_gain import load_model as load_source
from src.data.dual_component_tokenizer import base_tokenizer_training_data,base_tokenizer_joint_eeg,base_tokenizer_batch
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics,joint_reconstruction_summary,base_training_aggregate
from src.analysis.tokenizer_pattern_audit import attributes
from run_anchored_carrier_dictionary import pattern_fold,pattern_summary,FIELDS
from evaluate_joint_tokenizer import input_scales,write
ARMS=('wiener','transient');KINDS=('source','continuous','hard')


def rank(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def validate(cfg):
    if (cfg['schema']!='stratified_source_atoms_experiment_v1' or cfg['data']['allowed_splits']!=['train']
        or any(cfg['data'][v]!='forbidden' for v in ['validation_access','confirmation_access','test_access'])
        or cfg['model']['codebook_size']!=512 or cfg['model']['positive_atoms']!=511 or cfg['model']['group_depth']!=3
        or cfg['model']['context_seconds']!=30 or cfg['model']['token_seconds']!=2 or cfg['model']['EEG_rate_hz']!=200):
        raise ValueError('fixed train-only K512 hard-route native voltage contract required')


def context(cfg):
    g0=yaml.safe_load((ROOT/cfg['source']['g0_config']).read_text());manifest,scales,_,_=base_tokenizer_training_data(g0,ROOT)
    base=yaml.safe_load((ROOT/cfg['source']['base']).read_text())['source'];old=load_wiener(ROOT/base['waveform_dictionary'])
    plan=json.loads((ROOT/base['native_train_plan']).read_text())['plan'];folds=json.loads((ROOT/cfg['source']['folds']).read_text());rows=manifest['rows']
    expected=[]
    for d,people in plan.items():
        if set(people)!=set(folds[d]):raise ValueError('original fold subject identities changed')
        for s,crops in people.items():
            for c in crops:
                r=rows[c['row']]
                if (r['dataset'],r['subject'],r['split'],r['modality'])!=(d,s,'train','eeg'):raise ValueError('original train identities before signal reads')
                expected.extend((c['row'],c['start_s'],p) for p in c['positions'])
    if len(expected)!=17040 or sum(len(p) for p in plan.values())!=60 or sum(len(c) for p in plan.values() for c in p.values())!=400:raise ValueError('original400/60/17040 required')
    if old.metadata['input_identity']!=manifest['cfg_identity'] or old.metadata['scales_identity']!=manifest['scales_identity']:raise ValueError('source input/scale mismatch')
    sites=sorted({(r['dataset'],p) for r in rows if r['split']=='train' and r['modality']=='eeg' for p in r['positions']})
    return manifest,scales,rows,old,plan,folds,{s:i for i,s in enumerate(sites)}


def make_cases(cfg,fold,out,context_values):
    manifest,scales,rows,old,plan,folds,sites=context_values;cache=OrderedDict();cases=[]
    for d,people in sorted(plan.items()):
        names=old.metadata['positions'][d];lookup={v.upper():i for i,v in enumerate(names)};roles=np.arange(len(names))%5
        new=load_source(ROOT/cfg['source']['gain_run']/f'{d}_fold{fold}_transient.npz')
        fit_people={s for s in people if folds[d][s]!=fold}
        if set(new.metadata['fit_subjects'])!=fit_people or new.metadata['held_fold']!=fold:raise ValueError('new source trained on wrong outer fold')
        if new.metadata['input_identity']!=manifest['cfg_identity'] or new.metadata['scales_identity']!=manifest['scales_identity']:raise ValueError('new source identity')
        for s,crops in sorted(people.items()):
            records=Counter(rows[c['row']]['record'] for c in crops)
            for crop in crops:
                fields,seen,_=base_tokenizer_joint_eeg(rows,dict(crop,positions=[]),scales,names,cache,allowed_splits=('train',))
                x=fields['x'].T.reshape(1,len(names),15,400)
                physical=dict(wiener=old.predict_continuous(x),transient=new.predict_continuous(x,d,seen[None]))
                np.testing.assert_array_equal(physical['wiener']['sum'],physical['transient']['sum'])
                physical={a:{b:p[b] for b in 'SO'} for a,p in physical.items()}
                ex=np.array([(crop['row'],crop['start_s'],p) for p in crop['positions']],np.int32)
                batch=base_tokenizer_batch(rows,ex,scales,array_cache=cache,cache_limit=12)
                batch={k:(v[:,:1,:1].copy() if k.startswith(('v','dv')) else v) for k,v in batch.items()}
                native=np.array([lookup[rows[crop['row']]['positions'][p].upper()] for p in crop['positions']]);complete=batch['mask'].all(-1);target={}
                for b in 'SO':
                    raw,slices=attributes(batch[b.lower()][complete],'eeg');attribute=np.zeros((*complete.shape,10))
                    attribute[complete]=np.concatenate([raw[:,slice(*slices[k])] for k in FIELDS],axis=1);target[b]=attribute
                site=np.array([sites[d,rows[crop['row']]['positions'][p]] for p in crop['positions']])
                background=4*site[:,None]+np.searchsorted([.5,1,2],np.sqrt(np.mean(batch['x']**2,axis=-1)),side='right')
                cases.append(dict(dataset=d,subject=s,fold=folds[d][s],crop=crop,observed=seen[None],physical=physical,native=native,roles=roles,
                    examples=ex,complete=complete,target=target,background=background,batch=batch,scales=input_scales(rows,ex,scales),
                    weight=1/(len(fit_people)*len(records)*records[rows[crop['row']]['record']])))
            write(out/'progress.json',dict(stage='frozen_source_arrays',fold=fold,crops=len(cases),last_subject=s,CPU_seconds=time.process_time()))
    return cases


def training_patches(cases,fit,band,cfg):
    grouped=defaultdict(list)
    for i in fit:
        c=cases[i];p,t=np.nonzero(c['complete']&(c['roles'][c['native']][:,None]==band))
        grouped[c['dataset'],c['subject']].extend((i,int(c['native'][j]),int(patch)) for j,patch in zip(p,t))
    rng=np.random.default_rng(cfg['training']['seed']+band);picked=[]
    for key,items in sorted(grouped.items()):
        take=rng.choice(len(items),min(len(items),cfg['training']['max_patches_per_fit_subject_branch_band']),replace=False)
        picked.extend(items[j] for j in take)
    return picked


def continuous_carriers(model,physical,dataset):
    roles=model.roles(dataset);out={b:np.zeros_like(physical[b]) for b in 'SO'}
    for b in 'SO':
        for k in range(5):
            sites=np.flatnonzero(roles==k);out[b][:,sites]=project(physical[b][:,sites],k)
    return out


def fit_native_readout(cfg,model,cases,fit,arm,codes,kind):
    weights={}
    for d,names in model.metadata['positions'].items():
        roles=model.roles(d);C=len(names);grams={};crosses={};variance=np.zeros((C,3))
        for k in range(5):
            M=2*int((roles==k).sum());grams[k]=np.zeros((C,M,M));crosses[k]=np.zeros((C,M,3))
        for i in fit:
            c=cases[i]
            if c['dataset']!=d:continue
            carriers=model.atom_carriers(codes[i],d) if kind=='hard' else continuous_carriers(model,c['physical'][arm],d)
            valid=c['native'][c['complete'].all(1)];which=np.flatnonzero(c['complete'].all(1));weight=c['weight']
            for z,b in enumerate(['s','o','x']):variance[c['native'],z]=c['batch']['v'+b][:,0,0]
            for k in range(5):
                sites=np.flatnonzero(roles==k)
                f=np.concatenate([project(carriers[b][0,sites].reshape(len(sites),6000),k) for b in 'SO'])
                g=f@f.T/6000;grams[k][valid]+=weight*g
                target=np.stack([project(c['batch'][b][which].reshape(len(which),6000),k) for b in ['s','o','x']],axis=-1)
                crosses[k][valid]+=weight*np.einsum('mt,ctz->cmz',f,target,optimize=True)/6000
        if np.any(variance<=0):raise ValueError('positive original G0 component and X denominators required')
        for k in range(5):
            w=solve_readout(grams[k],crosses[k],variance,cfg['model']['readout_ridge_trace_fraction'])
            for j,b in enumerate('SO'):weights[d,b,k]=w[:,:,j]
    return weights


def audit_usage(models,cases,codes,fit,held,out):
    records=[]
    for arm,model in models.items():
        for b in 'SO':
            for band in range(5):
                counts=np.zeros((2,512),np.int64);people=defaultdict(set)
                for role,indices in enumerate([fit,held]):
                    for i in indices:
                        c=cases[i];keep=c['complete']&(c['roles'][c['native']][:,None]==band);ids=codes[arm][i][b][0,c['native']][keep]
                        counts[role]+=np.bincount(ids,minlength=512)
                        if role==0:
                            for code in np.unique(ids):people[int(code)].add((c['dataset'],c['subject']))
                for code in range(512):records.append(dict(arm=arm,branch=b,band=band,final_ID=code,fit_patches=int(counts[0,code]),fit_people=len(people[code]),held_patches=int(counts[1,code]),supported=bool(counts[0,code]>=8 and len(people[code])>=2)))
    with (out/'all_final_ID_support.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def aggregate(records,examples,rows,out,name):
    metrics={k:np.concatenate([v['metrics'][k] for v in records]) for k in records[0]['metrics']}
    support={k:np.concatenate([v['support'][k] for v in records]) for k in records[0]['support']}
    np.savez_compressed(out/f'{name}_metrics.npz',examples=examples,**metrics,**{'support_'+k:v for k,v in support.items()})
    summary=joint_reconstruction_summary(metrics,support,np.asarray(examples).tolist(),rows);write(out/f'{name}_summary.json',summary);return summary


def worker(task):
    cfg,out,fold=task;out=out/f'fold{fold}';out.mkdir(exist_ok=True,parents=True)
    lock=open(out/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'fold_summary.json').exists():raise FileExistsError('retain completed fold')
    started_CPU=time.process_time();limit=cfg['resources']['CPU_budget_seconds_per_fold'];deadline=int(np.ceil(started_CPU+limit))
    resource.setrlimit(resource.RLIMIT_CPU,(deadline-5,cfg['resources']['CPU_budget_seconds_total']))
    with threadpool_limits(cfg['resources']['numerical_threads_per_fold']):
        ctx=context(cfg);manifest,scales,rows,old,plan,folds,sites=ctx;cases=make_cases(cfg,fold,out,ctx)
        fit=[i for i,c in enumerate(cases) if c['fold']!=fold];held=[i for i,c in enumerate(cases) if c['fold']==fold]
        books={a:{} for a in ARMS};book_reports={a:{} for a in ARMS}
        for band in range(5):
            selected=training_patches(cases,fit,band,cfg)
            for b in 'SO':
                for arm in ARMS:
                    patches=np.stack([cases[i]['physical'][arm][b][0,n,t] for i,n,t in selected])
                    book=fit_book(patches,band,positive_atoms=511,seed=cfg['training']['seed']+band+100*'SO'.index(b),max_iter=cfg['training']['kmeans_max_iter'])
                    books[arm][b,band]=book;book_reports[arm][f'{b}_{band}']=dict(groups=book['report'],sample_identity=rank(selected),training_patches=len(selected))
                    write(out/'progress.json',dict(stage='hard_group_books',fold=fold,band=band,branch=b,arm=arm,CPU_seconds=time.process_time()))
        metadata=dict(schema='stratified_source_dictionary_v1',codebook_size=512,positions=old.metadata['positions'],held_fold=fold,config_identity=rank(cfg),
            input_identity=manifest['cfg_identity'],scales_identity=manifest['scales_identity'],integers_per_branch_native_position=15,bits_per_ID=9,
            context_seconds=30,token_seconds=2,continuous_decoder_inputs=0,readout='train_true_S_O_X_joint_native_spatial',fit_subjects={d:[s for s in p if folds[d][s]!=fold] for d,p in plan.items()})
        models={a:StratifiedSourceDictionary(dict(metadata,source=a),books[a],{}) for a in ARMS}
        codes={a:[model.assign(c['physical'][a],c['dataset'],c['observed']) for c in cases] for a,model in models.items()}
        continuous={}
        for a,model in models.items():
            model.weights=fit_native_readout(cfg,model,cases,fit,a,codes[a],'hard')
            continuous[a]=StratifiedSourceDictionary(dict(metadata,source=a,continuous_only=True),books[a],fit_native_readout(cfg,model,cases,fit,a,codes[a],'continuous'))
            model.save(out/f'{a}.npz');continuous[a].save(out/f'{a}_continuous_readout.npz');write(out/f'{a}_books.json',book_reports[a])
        audit_usage(models,cases,codes,fit,held,out);pattern={}
        for a,model in models.items():
            ex,err,supported=pattern_fold(model,cases,codes[a],fit,held,4*len(sites),sorted(plan))
            np.savez_compressed(out/f'{a}_pattern.npz',examples=ex,S=err['S'],O=err['O'],support=np.asarray([supported[b] for b in 'SO']))
            pattern[a]=pattern_summary(err,ex,rows,supported)
        reload={a:load_model(out/f'{a}.npz') for a in ARMS};transport=[];disk={}
        for a in ARMS:
            flat=[];index=[];offset=0
            for i in held:
                for b in 'SO':
                    v=codes[a][i][b];index.append(dict(case=i,branch=b,shape=list(v.shape),offset=offset,count=v.size));offset+=v.size;flat.append(v.reshape(-1))
            flat=np.concatenate(flat);path=out/f'{a}_held_IDs9.bin';path.write_bytes(pack_ids(flat));stored=unpack_ids(path.read_bytes(),len(flat));np.testing.assert_array_equal(stored,flat)
            disk[a]={(r['case'],r['branch']):stored[r['offset']:r['offset']+r['count']].reshape(r['shape']) for r in index}
            write(out/f'{a}_held_IDs9_index.json',dict(bits_per_ID=9,vocabulary=512,integers=len(flat),padding_bits=8*path.stat().st_size-len(flat)*9,index=index))
            transport.append(dict(arm=a,integers=len(flat),payload_bits=path.stat().st_size*8,replay_max_abs=0))
        records={a+'_'+k:[] for a in ARMS for k in KINDS};examples=[];plots={};plot_meta={};rules={}
        ref=ROOT/cfg['source']['gain_run']
        for d in plan:
            for rule in ['original_fixed_hash','original_highest_montage_S_NMSE','maximum_true_S_peak']:
                with np.load(ref/f'{d}_{rule}_wave.npz') as f:rules[tuple(f['triple'])]=d+'_'+rule
        for i in held:
            c=cases[i];predictions={}
            for a,m in models.items():
                ids={b:disk[a][i,b] for b in 'SO'};actual=reload[a].decode_ids(ids,c['dataset']);expected=m.decode_ids(codes[a][i],c['dataset'])
                for b in actual:np.testing.assert_array_equal(actual[b],expected[b])
                raw=dict(c['physical'][a],sum=c['physical']['wiener']['S']+c['physical']['wiener']['O'])
                predictions[a+'_source']=raw;predictions[a+'_hard']=actual
                predictions[a+'_continuous']=continuous[a].synthesize(continuous_carriers(m,c['physical'][a],c['dataset']),c['dataset'])
            for name,pred in predictions.items():records[name].append(joint_reconstruction_metrics({b:v[0,c['native']] for b,v in pred.items()},c['batch'],1,c['scales']))
            error=records['transient_hard'][-1]['metrics']['nmse'][:,1,0]
            choices=[(int(np.nanargmax(error)),c['dataset']+'_new_hard_S_NMSE',float(np.nanmax(error)))]
            for j,e in enumerate(c['examples']):
                if tuple(e) in rules:choices.append((j,rules[tuple(e)],0.))
            for j,label,score in choices:
                if label in plot_meta and score<=plot_meta[label]['score']:continue
                n=c['native'][j];arrays={k:c['batch'][k][j] for k in ['x','s','o','mask']}
                arrays.update({name+'_'+b:p[b][0,n] for name,p in predictions.items() for b in ['S','O','sum']})
                plots[label]=arrays;plot_meta[label]=dict(dataset=c['dataset'],subject=c['subject'],crop=c['crop'],position=rows[c['crop']['row']]['positions'][c['crop']['positions'][j]],scale=float(c['scales'][j]),unit=rows[c['crop']['row']]['source_unit'],score=score)
            examples.extend(c['examples']);write(out/'progress.json',dict(stage='all_held_waveforms',fold=fold,scored_positions=len(examples),CPU_seconds=time.process_time()))
        summaries={name:aggregate(value,examples,rows,out,name) for name,value in records.items()}
        np.savez_compressed(out/'plot_arrays.npz',**{label+'__'+k:v for label,arrays in plots.items() for k,v in arrays.items()});write(out/'plot_meta.json',plot_meta)
        write(out/'fold_summary.json',dict(status='complete',fold=fold,pattern=pattern,waveforms=summaries,transport=transport,CPU_seconds=time.process_time()-started_CPU,config_identity=rank(cfg)))
        return fold


def merge(cfg,out):
    _,_,rows,old,plan,folds,sites=context(cfg);waveforms={};patterns={}
    for name in [a+'_'+k for a in ARMS for k in KINDS]:
        data=[]
        for fold in range(4):
            with np.load(out/f'fold{fold}'/f'{name}_metrics.npz') as f:data.append({k:f[k].copy() for k in f.files})
        examples=np.concatenate([d['examples'] for d in data])
        if len(examples)!=17040 or len(set(map(tuple,examples)))!=17040:raise ValueError('missing original full held denominator')
        metrics={k:np.concatenate([d[k] for d in data]) for k in data[0] if k!='examples' and not k.startswith('support_')}
        support={k[8:]:np.concatenate([d[k] for d in data]) for k in data[0] if k.startswith('support_')}
        waveforms[name]=joint_reconstruction_summary(metrics,support,examples.tolist(),rows)
        np.savez_compressed(out/f'{name}_full_metrics.npz',examples=examples,**metrics,**{'support_'+k:v for k,v in support.items()})
    for a in ARMS:
        ex=[];errors={b:[] for b in 'SO'};support={b:np.zeros(2,np.int64) for b in 'SO'}
        for fold in range(4):
            with np.load(out/f'fold{fold}'/f'{a}_pattern.npz') as f:
                ex.append(f['examples'])
                for j,b in enumerate('SO'):errors[b].append(f[b]);support[b]+=f['support'][j]
        patterns[a]=pattern_summary({b:np.concatenate(v) for b,v in errors.items()},np.concatenate(ex),rows,support)
    rules=cfg['selection'];gates={}
    for a in ARMS:
        hard=waveforms[a+'_hard']['metrics'];cont=waveforms[a+'_continuous']['metrics'];nmse=np.ravel(hard['nmse']['macro']);cnmse=np.ravel(cont['nmse']['macro'])
        gates[a]=dict(waveform=bool(np.all(nmse<=rules['waveform_component_max_ratio_to_matched_role_continuous']*cnmse)),
            dataset_waveform=all(np.all(np.ravel(v)<=rules['waveform_dataset_component_max_ratio_to_matched_role_continuous']*np.ravel(cont['nmse']['by_dataset'][d])) for d,v in hard['nmse']['by_dataset'].items()),
            RMS=bool(np.all((np.ravel(hard['rms_gain']['macro'])>=.75)&(np.ravel(hard['rms_gain']['macro'])<=1.25))),
            theta_alpha=all(np.all(np.ravel(hard[k]['macro'])<=rules['theta_alpha_abs_db_max']) for k in ['band_4_8_abs_db','band_8_13_abs_db']),
            envelope=bool(np.all(np.ravel(hard['local_rms_correlation']['macro'])>=rules['local_RMS_correlation_min'])),
            pattern=all(v['support_mass']>=rules['support_mass_min'] and min(v['gain'][k] for k in ['power_4_8','power_8_13'])>0 and max(v['gain'].values())>=rules['one_gain_min'] and min(v['positive_subject_fraction'][k] for k in ['power_4_8','power_8_13'])>=rules['positive_subject_fraction_theta_alpha_min'] for v in patterns[a].values()))
    ratio=np.ravel(waveforms['transient_hard']['metrics']['nmse']['macro'])/np.ravel(waveforms['wiener_hard']['metrics']['nmse']['macro'])
    gates['transient']['paired']=bool(np.all(ratio[1:]<=rules['paired_S_O_NMSE_max_ratio_to_wiener_arm']) and ratio[0]<=rules['paired_X_NMSE_max_ratio_to_wiener_arm'])
    write(out/'summary.json',dict(status='complete',waveforms=waveforms,pattern=patterns,gates=gates,paired_ratios=ratio.tolist(),
        numeric_joint_pass=all(gates['transient'].values()),visual='pending_actual_image_review',validation_reads=0,
        same_actual_atomic_ID_not_coarse_group=True,codebook_size=512,bits_per_ID=9,EEG_pair_bits_per_native_position_second=9,
        source_exposure='new_source_books_and_readout_exclude_outer_held; originalWiener_G0historically_all60train',config_identity=rank(cfg)))
    print(json.dumps(dict(gates=gates,macro={k:v['metrics']['nmse']['macro'] for k,v in waveforms.items()},pattern={a:{b:{k:v for k,v in z.items() if k!='hierarchy'} for b,z in p.items()} for a,p in patterns.items()})),flush=True)


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths=out/'waveforms';paths.mkdir(exist_ok=True);selected={}
    for fold in range(4):
        for name,meta in json.loads((out/f'fold{fold}'/'plot_meta.json').read_text()).items():
            if name not in selected or meta['score']>selected[name][1]['score']:selected[name]=(fold,meta)
    styles={'wiener_hard':('#df8f00','-'),'transient_hard':('#0072b2','-'),
        'wiener_continuous':('#8b9c39','--'),'transient_continuous':('#188955','--'),'transient_source':('#a863a6',':')}
    manifest=[]
    for name,(fold,meta) in sorted(selected.items()):
        with np.load(out/f'fold{fold}'/'plot_arrays.npz') as saved:arrays={k.split('__',1)[1]:saved[k].copy() for k in saved if k.startswith(name+'__')}
        mask=arrays['mask'].reshape(6000).astype(bool);scale=meta['scale'];true=arrays['s'].reshape(6000)*scale
        if 'maximum_true_S_peak' in name:peak=int(np.argmax(np.where(mask,abs(true),-1)))
        else:peak=int(np.argmax(np.where(mask,abs(arrays['transient_hard_S'].reshape(6000)*scale-true),-1)))
        t=np.arange(6000)/200;center=peak/200;boundary=np.clip(2*round(center/2),2,28)
        spans=[(0,30),(max(0,min(28,center-1)),min(30,max(2,center+1))),(boundary-.25,boundary+.25)]
        unit='native REFED export' if meta['unit']=='unknown_REFED_EEG_export' else meta['unit']
        fig,axes=plt.subplots(3,3,figsize=(18,10),layout='constrained')
        for row,(b,target) in enumerate([('sum','x'),('S','s'),('O','o')]):
            for col,(lo,hi) in enumerate(spans):
                ax=axes[row,col];ax.plot(t,np.where(mask,arrays[target].reshape(6000)*scale,np.nan),c='black',lw=.85,label='target')
                for method,(color,style) in styles.items():ax.plot(t,arrays[method+'_'+b].reshape(6000)*scale,c=color,ls=style,lw=.65,alpha=.8,label=method)
                for edge in np.arange(0,31,2):
                    if lo<=edge<=hi:ax.axvline(edge,c='.6',ls=':',lw=.5)
                ax.fill_between(t,0,1,where=~mask,transform=ax.get_xaxis_transform(),color='.7',alpha=.4)
                ax.set_xlim(lo,hi);ax.set_xlabel('time (s)');ax.set_ylabel(b+'\n('+unit+')');ax.grid(alpha=.2)
                if row==0:ax.set_title(['30s native voltage','2s at '+('true S peak' if 'maximum_true_S_peak' in name else 'new hard S error'),'2s token boundary zoom'][col])
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
        fig.suptitle(f'{name}: {meta["subject"]}/{meta["crop"]["row"]}+{meta["crop"]["start_s"]}s/{meta["position"]}, fold{fold}\nK512 final atomic IDs; all lines same physical coordinates; gray=unscored; no smoothing')
        file=paths/(name+'.png');fig.savefig(file,dpi=180);plt.close(fig);manifest.append(dict(path=str(file),fold=fold,**meta,zoom_seconds=center))
    write(out/'figure_manifest.json',manifest)


def atlas(cfg,out):
    """Fixed fold0 actual-atom atlas; no fitting, no ordinary validation reads."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ctx=context(cfg);rows=ctx[2];folder=out/'atlas_fold0';folder.mkdir(exist_ok=True)
    cases=make_cases(cfg,0,folder,ctx);fit=[i for i,c in enumerate(cases) if c['fold']!=0];held=[i for i,c in enumerate(cases) if c['fold']==0]
    models={a:load_model(out/'fold0'/f'{a}.npz') for a in ARMS};exports={};member_index=[]
    for a,model in models.items():
        codes=[model.assign(c['physical'][a],c['dataset'],c['observed']) for c in cases]
        index=json.loads((out/'fold0'/f'{a}_held_IDs9_index.json').read_text());stored=unpack_ids((out/'fold0'/f'{a}_held_IDs9.bin').read_bytes(),index['integers'])
        for r in index['index']:np.testing.assert_array_equal(codes[r['case']][r['branch']],stored[r['offset']:r['offset']+r['count']].reshape(r['shape']))
        for b in 'SO':
            for band in range(5):
                values=[[],[]];labels=[[],[]];people=[defaultdict(set),defaultdict(set)];identity=[[],[]]
                for role,indices in enumerate([fit,held]):
                    for i in indices:
                        c=cases[i];p,t=np.nonzero(c['complete']&(c['roles'][c['native']][:,None]==band))
                        code=codes[i][b][0,c['native'][p],t]
                        values[role].append(c['batch'][b.lower()][p,t]);labels[role].append(code)
                        identity[role].extend((i,int(j),int(patch)) for j,patch in zip(p,t))
                        for v in np.unique(code):people[role][int(v)].add((c['dataset'],c['subject']))
                waves=[np.concatenate(v) for v in values];ids=[np.concatenate(v) for v in labels];counts=np.array([np.bincount(v,minlength=512) for v in ids])
                mean=np.full((2,512,400),np.nan);quantile=np.full((2,512,2,400),np.nan);representative=np.full((2,512,400),np.nan)
                attrs=np.full((2,512,10),np.nan)
                for role in range(2):
                    raw,slices=attributes(waves[role],'eeg');truth_attributes=np.concatenate([raw[:,slice(*slices[k])] for k in FIELDS],1)
                    for code in np.flatnonzero(counts[role]):
                        chosen=np.flatnonzero(ids[role]==code);data=waves[role][chosen]
                        mean[role,code]=data.mean(0);quantile[role,code]=np.quantile(data,[.1,.9],axis=0);attrs[role,code]=truth_attributes[chosen].mean(0)
                        index_in_role=min(chosen,key=lambda j:rank([cfg['training']['seed'],identity[role][j]]));representative[role,code]=waves[role][index_in_role]
                        i,p,t=identity[role][index_in_role];c=cases[i]
                        member_index.append(dict(arm=a,branch=b,band=band,ID=int(code),role='fit' if role==0 else 'held_train',dataset=c['dataset'],subject=c['subject'],crop=c['crop'],position=rows[c['crop']['row']]['positions'][c['crop']['positions'][p]],patch=t))
                prefix=f'{a}_{b}_{band}';exports.update({prefix+'_counts':counts,prefix+'_mean':mean,prefix+'_q10q90':quantile,prefix+'_representative':representative,prefix+'_attributes':attrs,
                    prefix+'_people':np.array([[len(people[r][k]) for k in range(512)] for r in range(2)])})
                order=np.argsort(-counts[0]);order=order[counts[0,order]>0];cum=np.cumsum(counts[0,order])/counts[0].sum();major=order[:int(np.searchsorted(cum,.9))+1]
                fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
                for ax,(field,sl) in zip(axes[0],FIELDS.items()):
                    target=attrs[:,:,slice(*sl)].mean(-1);good=np.isfinite(target).all(0)&(np.arange(512)>0)
                    ax.scatter(target[0,good],target[1,good],s=8+20*np.sqrt(counts[1,good]/max(counts[1].max(),1)),c=np.isin(np.flatnonzero(good),major),cmap='coolwarm',vmin=0,vmax=1,alpha=.6)
                    lo=float(np.nanmin(target[:,good]));hi=float(np.nanmax(target[:,good]));ax.plot([lo,hi],[lo,hi],c='black',ls=':',lw=.8)
                    ax.set_xlabel('train true attribute mean');ax.set_ylabel('held train true attribute mean');ax.set_title(field);ax.grid(alpha=.2)
                axes[1,0].plot(np.arange(len(order))+1,cum,c='black');axes[1,0].axhline(.9,c='red',ls=':');axes[1,0].set_xlabel('all actual atom IDs ranked by fit count');axes[1,0].set_ylabel('cumulative fit coverage')
                axes[1,1].plot(counts[0,order],label='fit');axes[1,1].plot(counts[1,order],label='held train');axes[1,1].set_xlabel('actual atom rank');axes[1,1].set_ylabel('complete real patches');axes[1,1].legend()
                supported=(counts[0]>=8)&np.array([len(people[0][k])>=2 for k in range(512)])
                axes[1,2].bar(['8 patches + 2 people','unsupported'],[counts[1,supported].sum(),counts[1,~supported].sum()]);axes[1,2].set_ylabel('all held real patches')
                fig.suptitle(f'fold0 {a}/{b}/band{band}: all512 actual atom IDs, {len(major)} reach90% fit mass\nred=90% major-ID set; blue=tail; marker size=held occupancy; means are attributes, not carrier phase semantics')
                fig.savefig(folder/(prefix+'_all_ID_attributes.png'),dpi=160);plt.close(fig)
                # Predefined occupancy-spaced examples; every other code and
                # all failed/rare codes remain in the arrays and support table.
                selected=major[np.unique(np.linspace(0,len(major)-1,min(12,len(major))).round().astype(int))]
                fig,axes=plt.subplots(4,3,figsize=(14,10),layout='constrained');time2=np.arange(400)/200
                for ax,code in zip(axes.ravel(),selected):
                    ax.fill_between(time2,*quantile[0,code],color='.6',alpha=.25,label='train10–90%')
                    ax.plot(time2,mean[0,code],c='black',lw=.8,label='train true mean');ax.plot(time2,representative[1,code],c='#0072b2',lw=.8,label='held real member')
                    ax.plot(time2,model.books[b,band]['atoms'][code],c='#df8f00',lw=.7,label='local role-band atom')
                    ax.set_title(f'ID{code}: train{counts[0,code]}/{len(people[0][code])}people; held{counts[1,code]}');ax.set_xlabel('local time (s)');ax.set_ylabel('fixed G0 input SD');ax.grid(alpha=.2)
                for ax in axes.ravel()[len(selected):]:ax.set_visible(False)
                handles,labs=axes.ravel()[0].get_legend_handles_labels();fig.legend(handles,labs,loc='outside lower center',ncol=4)
                fig.suptitle(f'fold0 {a}/{b}/band{band}: actual final ID examples spaced across90% mass\nlocal atom is one decoder input; complete30s integer-sequence reconstruction is audited separately; all512 retained in NPZ')
                fig.savefig(folder/(prefix+'_members.png'),dpi=160);plt.close(fig)
                write(folder/'progress.json',dict(stage='all_actual_ID_atlas',arm=a,branch=b,band=band,CPU_seconds=time.process_time()))
    np.savez_compressed(folder/'all_actual_ID_templates.npz',**exports);write(folder/'representative_members.json',member_index)
    write(folder/'summary.json',dict(status='complete',fold=0,arms=list(ARMS),all_actual_IDs_per_book=512,book_count=20,source_refits=0,book_refits=0,held_integer_reassignment_matches_saved_packed_IDs=True,
        raw_examples_rule='12 occupancy_spaced_major_IDs_not_best_waveforms',every_code_in_arrays=True,ordinary_validation_reads=0))


def replay(cfg,out):
    """New-process ID-only decoder followed by original masked scores and S-tail diagnostics."""
    _,scales,rows,old,plan,folds,_=context(cfg);cache=OrderedDict();all_cases=[]
    for d,people in sorted(plan.items()):
        for s,crops in sorted(people.items()):all_cases.extend(dict(dataset=d,subject=s,crop=c) for c in crops)
    def forbidden(*args,**kwargs):raise AssertionError('integer replay requested an encoder or raw-X source')
    old.predict_continuous=forbidden
    summary={}
    for arm in ARMS:
        records=[];examples=[];peak_parts=defaultdict(list);integers=0;payload_bits=0;index_bytes=0
        for fold in range(4):
            folder=out/f'fold{fold}';model=load_model(folder/f'{arm}.npz');model.assign=forbidden
            meta_path=folder/f'{arm}_held_IDs9_index.json';index=json.loads(meta_path.read_text());index_bytes+=meta_path.stat().st_size
            path=folder/f'{arm}_held_IDs9.bin';ids=unpack_ids(path.read_bytes(),index['integers']);integers+=len(ids);payload_bits+=path.stat().st_size*8
            by_case=defaultdict(dict)
            for r in index['index']:by_case[r['case']][r['branch']]=ids[r['offset']:r['offset']+r['count']].reshape(r['shape'])
            for i,code in by_case.items():
                c=all_cases[i];d=c['dataset'];crop=c['crop']
                if folds[d][c['subject']]!=fold:raise ValueError('saved integer case is not the corresponding held subject')
                # The decoder finishes before targets/X are read by the scorer.
                prediction=model.decode_ids(code,d)
                ex=np.array([(crop['row'],crop['start_s'],p) for p in crop['positions']],np.int32)
                batch=base_tokenizer_batch(rows,ex,scales,array_cache=cache,cache_limit=12);scale=input_scales(rows,ex,scales)
                lookup={v.upper():j for j,v in enumerate(model.metadata['positions'][d])};native=np.array([lookup[rows[crop['row']]['positions'][p].upper()] for p in crop['positions']])
                pred={b:v[0,native] for b,v in prediction.items()};records.append(joint_reconstruction_metrics(pred,batch,1,scale));examples.extend(ex)
                mask=batch['mask'].reshape(len(ex),6000);truth=batch['s'].reshape(len(ex),6000);p=pred['S'].reshape(len(ex),6000);variance=batch['vs'][:,0,0]
                for multiplier in [2,4]:
                    region=mask&(abs(truth)>=multiplier*np.sqrt(variance)[:,None]);count=region.sum(1);den=variance*count
                    error=np.divide(np.where(region,(p-truth)**2,0).sum(1),den,out=np.full(len(ex),np.nan),where=den>0)
                    energy=np.where(region,truth**2,0).sum(1);gain=np.divide(np.where(region,p*truth,0).sum(1),energy,out=np.full(len(ex),np.nan),where=energy>0)
                    for metric,value in [('NMSE',error),('signed_gain',gain),('point_count',count)]:peak_parts[f'true_S_ge_{multiplier}SD_{metric}'].append(value[:,None])
        metrics={k:np.concatenate([v['metrics'][k] for v in records]) for k in records[0]['metrics']};ex=np.asarray(examples)
        with np.load(out/f'{arm}_hard_full_metrics.npz') as expected:
            indices={tuple(v):j for j,v in enumerate(expected['examples'])};take=[indices[tuple(v)] for v in ex]
            maximum=max(float(np.nanmax(abs(v-expected[k][take]))) for k,v in metrics.items())
            for k,v in metrics.items():np.testing.assert_allclose(v,expected[k][take],rtol=1e-12,atol=1e-12,equal_nan=True)
        peaks={k:np.concatenate(v) for k,v in peak_parts.items()};np.savez_compressed(out/f'{arm}_integer_replay_peaks.npz',examples=ex,**peaks)
        peak_summary=base_training_aggregate(dict(nmse=peaks['true_S_ge_2SD_NMSE'],**peaks),ex,rows)
        peak_summary['component_order']=['S']
        peak_summary['denominator']='original fixed G0 S training variance times number of masked points meeting the true-S threshold'
        summary[arm]=dict(scored=len(ex),integer_count=integers,payload_bits=payload_bits,index_bytes=index_bytes,bitrate_payload_per_native_position_second=payload_bits/(len(ex)*30),
            score_max_abs_difference=maximum,score_tolerance=dict(atol=1e-12,rtol=1e-12),encoder_calls=0,continuous_source_calls=0,decoder_inputs=['saved9bit_IDs','frozen_atoms_and_readout','fixed_dataset_native_order'],true_S_high=peak_summary)
    write(out/'independent_integer_replay.json',dict(status='passed',arms=summary,validation_reads=0,identity='original400crops17040position_windows',read_order='decode_integer_output_before_target_or_X_scorer_read'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--synthetic',action='store_true');p.add_argument('--merge',action='store_true');p.add_argument('--render',action='store_true');p.add_argument('--atlas',action='store_true');p.add_argument('--replay',action='store_true')
    args=p.parse_args();cfg=yaml.safe_load(args.config.read_text());validate(cfg);out=args.output;out.mkdir(exist_ok=True,parents=True)
    if args.render:render(out);return
    if args.atlas:
        if not (out/'summary.json').exists():raise ValueError('finish paired CV before posthoc fixed atlas')
        with threadpool_limits(2):atlas(cfg,out)
        return
    if args.replay:
        with threadpool_limits(2):replay(cfg,out)
        return
    if args.synthetic:
        rng=np.random.default_rng(9);t=np.arange(400)/200;q=rng.uniform(.3,3,(1200,1))*np.sin(2*np.pi*rng.choice([6,10],1200)[:,None]*t+rng.uniform(0,6.28,(1200,1)))
        books={(b,k):fit_book(q,k,seed=9,max_iter=1) for b in 'SO' for k in range(5)}
        meta=dict(schema='stratified_source_dictionary_v1',codebook_size=512,positions={'fixture':list(map(str,range(10)))})
        weights={('fixture',b,k):rng.normal(size=(4,10)) for b in 'SO' for k in range(5)};m=StratifiedSourceDictionary(meta,books,weights);ids={b:rng.integers(0,512,(1,10,15),dtype=np.int16) for b in 'SO'}
        m.save(out/'synthetic_model.npz');restored=load_model(out/'synthetic_model.npz')
        for b,v in m.decode_ids(ids,'fixture').items():np.testing.assert_array_equal(v,restored.decode_ids(ids,'fixture')[b])
        write(out/'synthetic.json',dict(status='pass',measured_reads=0,single_atomic_ID=True,reload_max_abs=0));return
    if not (out/'synthetic.json').exists():raise ValueError('synthetic contract must precede measured reads')
    if args.merge:merge(cfg,out);return
    lock=open(out/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'summary.json').exists():raise FileExistsError('preserve completed paired experiment')
    write(out/'resolved_config.json',cfg)
    with ProcessPoolExecutor(max_workers=cfg['resources']['fold_workers'],mp_context=multiprocessing.get_context('fork')) as pool:list(pool.map(worker,[(cfg,out,f) for f in range(4)]))
    merge(cfg,out)


if __name__=='__main__':main()
