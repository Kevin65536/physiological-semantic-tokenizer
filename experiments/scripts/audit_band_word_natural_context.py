"""Train-person natural-context audit of actual frequency-tagged EEG words."""
from __future__ import annotations

import argparse
from collections import OrderedDict,defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv,hashlib,json,multiprocessing,os
from pathlib import Path
import subprocess,sys,time

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.data.dual_component_tokenizer import base_tokenizer_training_data,base_tokenizer_batch,base_tokenizer_joint_eeg_context
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit,attributes,feature_means
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.tokenizers.eeg_transient_gain import load_model as load_source
from src.tokenizers.band_exchange_words import encode_native,encode_slots,source_properties,ATOMS
from src.tokenizers.adaptive_band_allocation import band_waves,choose_exchanges,exchange_costs,extract_coordinates

STATE={}
FIELDS={'power_4_8':(0,1),'power_8_13':(1,2),'rms_envelope':(2,10)}


def write(path,value):
    def scalar(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(value,default=scalar,indent=2,ensure_ascii=False)+'\n')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def ranking(seed,c):
    return hashlib.sha256(json.dumps([seed,c['dataset'],c['subject'],c['crop']['row'],c['crop']['start_s']]).encode()).hexdigest()


def budget():
    unit=os.environ.get('TAGGED_NATURAL_UNIT')
    if unit:
        v=subprocess.check_output(['systemctl','--user','show',unit,'-p','CPUUsageNSec','--value'],text=True).strip()
        if v.isdigit() and int(v)/1e9>700:raise RuntimeError('natural-context CPU budget reached, retained completed dataset outputs')


def encode_cached(x,seen,dataset,fold,source,cov,books,weights,cache):
    prediction=source.predict_continuous(x[None],dataset,seen[None]);result={}
    for b in 'SO':
        fields=band_waves(np.fft.rfft(prediction[b][0].reshape(len(seen),6000)))
        key=fold,b,seen.tobytes()
        if key not in cache:
            cache[key]=exchange_costs(cov[b],seen,weights[b]*seen)
            if len(cache)>6:cache.popitem(last=False)
        role,_=choose_exchanges(fields,seen,cache[key]);slots=extract_coordinates(fields,role).astype(np.float32)
        prop=source_properties(prediction[b][0]);codes=np.zeros(role.shape,np.uint16)
        for tag,book in enumerate(books[b]):
            take=role==tag
            if take.any():codes[take]=encode_slots(slots[take],prop[take],tag,book,'balanced')
        result[b]=codes
    return result


def word_roles(words):
    values=np.asarray(words,dtype=np.int32)
    role=np.where(values==0,-1,(values-1)//1023)
    if STATE.get('family')=='innovation':
        second=values>=5116
        role[second]=(values[second]-5116)//2047
    return role


def build_profiles(run,hard,pop,bank,rows,out):
    models={};backgrounds=4*len(pop['background_sites']);proof=[]
    for fold in range(4):
        parts={b:[] for b in 'SO'};examples=[]
        for dataset in sorted(pop['native_positions']):
            with np.load(hard/dataset/f'fold_{fold}_ids.npz') as z:
                examples.append(z['examples'])
                for b in 'SO':parts[b].append(z[STATE.get('family','exchange_balanced')+'_'+b])
        np.testing.assert_array_equal(np.concatenate(examples),bank['examples'])
        fit=bank['fold']!=fold;held=~fit;fitmask=bank['valid']&fit[:,None]
        for b in 'SO':
            codes=np.concatenate(parts[b]);audit=SparsePatternAudit(15351 if STATE.get("family")=="innovation" else 5116,backgrounds,[],10,FIELDS)
            audit.fit(codes[fitmask],bank['background'][fitmask],bank[b+'_attributes'][fitmask])
            with np.load(hard/(STATE.get('family','exchange_balanced')+f'_{b}_word_metrics.npz')) as saved:
                np.testing.assert_array_equal(audit.local.count,saved[('fit_count_' if STATE.get('family')=='innovation' else 'fit_')+str(fold)])
                people=saved[f'fit_people_{fold}'].copy();expected=saved['word'][held]
            valid=bank['valid'][held];background,_,conditional=audit.read(codes[held][valid],bank['background'][held][valid])
            truth=bank[b+'_attributes'][held][valid];errors=np.zeros((*valid.shape,2,3))
            errors[valid]=np.stack([feature_means((v-truth)**2,FIELDS) for v in [background,conditional]],axis=1)
            errors=errors.sum(1)/valid.sum(1).clip(1)[:,None,None];errors[~valid.any(1)]=np.nan
            np.testing.assert_allclose(errors,expected,rtol=8*np.finfo(float).eps,atol=0)
            absolute_difference=float(np.nanmax(abs(errors-expected)))
            max_relative_difference=float(np.nanmax(abs(errors-expected)/np.maximum(abs(expected),1e-300)))
            train=bank[b+'_attributes'][fitmask];variance=np.maximum(train.var(0),1e-20)
            dataset_variance={}
            for d in pop['native_positions']:
                ds=np.array([rows[i]['dataset']==d for i,t,p in bank['examples']])
                dataset_variance[d]=np.maximum(bank[b+'_attributes'][fitmask&ds[:,None]].var(0),1e-20)
            models[fold,b]=dict(audit=audit,people=people,variance=variance,dataset_variance=dataset_variance)
            proof.append(dict(fold=fold,branch=b,fit_people=np.unique(bank['subject'][fit]),held_people=np.unique(bank['subject'][held]),original_errors_bitwise_equal=bool(np.array_equal(errors,expected,equal_nan=True)),original_errors_max_absolute_difference=absolute_difference,original_errors_max_relative_difference=max_relative_difference,replay_relative_tolerance=8*np.finfo(float).eps,fit_count_bitwise_equal=True))
    write(out/'profile_replay.json',dict(status='passed',folds=proof,fit_scope='other three original person folds, 400-array raw background plus actual paid combined-ID means',source_bank_sha256=sha(run/'component_source_transfer/hard_as_v1/local_information_v1/training_attributes.npz')))
    return models


def worker(dataset):
    rows,scales,pop,run,hard,out,cases,profiles=(STATE[k] for k in ['rows','scales','population','run','hard','out','selected','profiles'])
    names=pop['native_positions'][dataset];native={v.upper():j for j,v in enumerate(names)};site_index={tuple(v):j for j,v in enumerate(pop['background_sites'])}
    local=[c for c in cases if c['dataset']==dataset];directory=out/dataset;directory.mkdir(exist_ok=True)
    cache=OrderedDict();objects={};stored={};records=[];arrays=defaultdict(list);proofs=[];probe_folds=set()
    for number,c in enumerate(local):
        budget();fold=c['fold'];row=rows[c['crop']['row']]
        if row['split']!='train' or row['subject']!=c['subject']:raise ValueError('original train person required before context reads')
        if fold not in objects:
            source=load_source(run/'component_source_transfer/transient_gain_v1'/f'{dataset}_fold{fold}_transient.npz')
            expected={v['subject'] for v in pop['cases'] if v['dataset']==dataset and v['fold']!=fold}
            if set(source.metadata['fit_subjects'])!=expected:raise ValueError('source includes held person')
            if STATE.get('family')=='innovation':
                from src.tokenizers.cross_branch_innovations import InnovationEEG
                from src.tokenizers.band_exchange_words import load_fold_books
                with np.load(hard/dataset/f'fold_{fold}_state.npz') as z:
                    if set(z['fit_people'])!=expected:raise ValueError('innovation fit people differ')
                    state={k:z[k] for k in ('joint','pilot_gain','pilot_noise','scale','innovation_mean_gain','innovation_mean_noise')}
                pilot={b:load_fold_books(run/STATE['pilot_books'],fold,b) for b in 'SO'}
                second={b:load_fold_books(hard/'books',fold,b) for b in 'SO'}
                objects[fold]=InnovationEEG(dict(positions={dataset:names}),{dataset:source},pilot,second,{dataset:state})
            else:
                with np.load(run/'component_source_transfer/adaptive_band_v1'/dataset/f'fold_{fold}_model.npz') as z:
                    if set(z['fit_people'])!=expected:raise ValueError('covariance includes held person')
                    covariance={b:z[b].copy() for b in 'SO'}
                books={b:[] for b in 'SO'}
                for b in 'SO':
                    for tag in range(5):
                        with np.load(hard/'books'/f'fold_{fold}_{b}_{tag}.npz') as z:
                            meta=json.loads(str(z['metadata']))
                            if (meta['fold'],meta['branch'],meta['role'])!=(fold,b,tag):raise ValueError('wrong word book')
                            books[b].append({k:z[k].copy() for k in z.files if k!='metadata'})
                table={name.upper():v for name,v in scales['positions']['eeg'][dataset].items()}
                weights={b:np.array([1/np.ravel(table[name.upper()]['target_variance'][b.lower()])[0] for name in names]) for b in 'SO'}
                objects[fold]=(source,covariance,books,weights)
            original_cases=[v for v in pop['cases'] if v['dataset']==dataset]
            with np.load(hard/dataset/f'fold_{fold}_ids.npz') as z:
                idx={tuple(e):j for j,e in enumerate(z['examples'])}
                stored[fold]=dict(index=idx,**{b:z[STATE.get('family','exchange_balanced')+'_'+b].copy() for b in 'SO'})
        crop=dict(row=c['crop']['row'],start_s=c['crop']['start_s'],positions=c['paired_positions'])
        old,oldseen,oldraw=base_tokenizer_joint_eeg_context(rows,crop,scales,names,0,allowed_splits=('train',))
        shifted,newseen,newraw=base_tokenizer_joint_eeg_context(rows,crop,scales,names,c['direction'],allowed_splits=('train',))
        old_slice=slice(2800,3200);center=7-c['direction'];new_slice=slice(center*400,(center+1)*400)
        np.testing.assert_array_equal(oldraw['raw_normalized_x'][old_slice],newraw['raw_normalized_x'][new_slice])
        np.testing.assert_array_equal(oldraw['observation_valid'][old_slice],newraw['observation_valid'][new_slice])
        a,b=(slice(400,6000),slice(0,5600)) if c['direction']==1 else (slice(0,5600),slice(400,6000))
        np.testing.assert_array_equal(oldraw['raw_normalized_x'][a],newraw['raw_normalized_x'][b])
        np.testing.assert_array_equal(oldraw['observation_valid'][a],newraw['observation_valid'][b])
        x0=old.T.reshape(len(names),15,400).astype(np.float32);x1=shifted.T.reshape(len(names),15,400).astype(np.float32)
        if STATE.get('family')=='innovation':
            model=objects[fold]
            paid=[{b:ids[0] for b,ids in model.encode(x[None],dataset,seen[None]).items()} for x,seen in [(x0,oldseen),(x1,newseen)]]
            probe_folds.add(fold)
        else:
            source,covariance,books,weights=objects[fold]
            paid=[encode_cached(x,seen,dataset,fold,source,covariance,books,weights,cache) for x,seen in [(x0,oldseen),(x1,newseen)]]
            if fold not in probe_folds:
                probe=encode_native(x0,oldseen,dataset,source,books,covariance,weights,'exchange','balanced')
                probe2=encode_native(x1,newseen,dataset,source,books,covariance,weights,'exchange','balanced')
                for branch in 'SO':
                    np.testing.assert_array_equal(paid[0][branch],probe[branch]);np.testing.assert_array_equal(paid[1][branch],probe2[branch])
                probe_folds.add(fold)
        offsets=[stored[fold]['index'][c['crop']['row'],c['crop']['start_s'],p] for p in c['crop']['positions']]
        positions=np.asarray(c['native']);inverse=np.argsort(positions)
        saved={b:stored[fold][b][np.asarray(offsets)[inverse]] for b in 'SO'}
        mismatch={b:int((saved[b]!=paid[0][b]).sum()) for b in 'SO'}
        proof=dict(pair=c,raw_center_and28s_overlap_exact=True,original_model_center_zeroed=~oldseen,shifted_model_center_zeroed=~newseen,
                   full_original_ID_mismatch=mismatch,old_ID_slots=int(sum(v.size for v in saved.values())))
        proofs.append(proof)
        np.savez_compressed(directory/f'pair_{number:03d}_integers.npz',old_S=paid[0]['S'],old_O=paid[0]['O'],shifted_S=paid[1]['S'],shifted_O=paid[1]['O'],saved_S=saved['S'],saved_O=saved['O'],old_observed=oldseen,shifted_observed=newseen,
                            original_center_X=oldraw['raw_normalized_x'][old_slice],shifted_center_X=newraw['raw_normalized_x'][new_slice])
        examples=np.array([(crop['row'],crop['start_s'],p) for p in crop['positions']]);batch=base_tokenizer_batch(rows,examples,scales)
        valid=batch['mask'][:,7].astype(bool).all(-1);idx=np.flatnonzero(valid)
        if not len(idx):
            records.append(dict(pair=c,scored_positions=0,reason='no complete central teacher patch'));continue
        ex=examples[idx];native_sites=np.array([native[row['positions'][p].upper()] for i,t,p in ex])
        bg=np.array([4*site_index[dataset,row['positions'][p]] for i,t,p in ex])+np.searchsorted([.5,1,2],np.sqrt((batch['x'][idx,7]**2).mean(-1)),side='right')
        fields_by_b={}
        for branch in 'SO':
            full,slices=attributes(batch[branch.lower()][idx,7],'eeg');truth=np.concatenate([full[:,slice(*slices[k])] for k in FIELDS],axis=1)
            codes=np.array([saved[branch][native_sites,7],paid[0][branch][native_sites,7],paid[1][branch][native_sites,center]])
            model=profiles[fold,branch];head=model['audit'];background=[];pred=[];support=[]
            for stream in codes:
                back,_,estimate=head.read(stream,bg);background.append(back);pred.append(estimate)
                support.append((head.local.count[stream]>=8)&(model['people'][stream]>=2))
            for back in background[1:]:np.testing.assert_array_equal(back,background[0])
            errors=np.stack([feature_means((value-truth)**2,FIELDS) for value in [background[0],*pred]],axis=1)
            delta=feature_means((pred[1]-pred[2])**2,FIELDS)/np.array([model['variance'][a:b].mean() for a,b in FIELDS.values()])
            ds_delta=feature_means((pred[1]-pred[2])**2,FIELDS)/np.array([model['dataset_variance'][dataset][a:b].mean() for a,b in FIELDS.values()])
            arrays[branch+'_errors'].append(errors);arrays[branch+'_delta'].append(delta);arrays[branch+'_dataset_delta'].append(ds_delta)
            arrays[branch+'_codes'].append(codes.T);arrays[branch+'_support'].append(np.array(support).T)
            arrays[branch+'_truth'].append(truth);arrays[branch+'_predictions'].append(np.array(pred).transpose(1,0,2));arrays[branch+'_background'].append(background[0])
            fields_by_b[branch]=dict(codes=codes,truth=truth,prediction=pred)
        arrays['examples'].append(ex);arrays['fold'].append(np.full(len(ex),fold))
        arrays['model_center_equal'].append((old[old_slice]==shifted[new_slice]).all(0)[native_sites])
        arrays['observed_both'].append(oldseen[native_sites]&newseen[native_sites])
        arrays['true_x'].append(batch['x'][idx,7]);arrays['true_s'].append(batch['s'][idx,7]);arrays['true_o'].append(batch['o'][idx,7])
        records.append(dict(pair=c,scored_positions=len(ex),invalid_central_target_positions=int((~valid).sum()),introduced_real_support_s=[crop['start_s']+30,crop['start_s']+32] if c['direction']==1 else [crop['start_s']-2,crop['start_s']]))
        write(directory/'progress.json',dict(completed=number+1,planned=len(local),CPU_seconds=time.process_time()))
    np.savez_compressed(directory/'results.npz',**{k:np.concatenate(v) for k,v in arrays.items()})
    write(directory/'identity_and_population.json',dict(pairs=records,proofs=proofs,public_encode_native_probed_folds=sorted(probe_folds),saved_original_exact=all(all(v==0 for v in p['full_original_ID_mismatch'].values()) for p in proofs)))
    return dataset


def summarize(out,rows,pop):
    parts=defaultdict(list);identities={}
    for dataset in sorted(pop['native_positions']):
        with np.load(out/dataset/'results.npz') as z:
            for k in z.files:parts[k].append(z[k].copy())
        identities[dataset]=json.loads((out/dataset/'identity_and_population.json').read_text())
    arrays={k:np.concatenate(v) for k,v in parts.items()};examples=arrays['examples'];result={}
    for b in 'SO':
        errors=base_training_aggregate({'nmse':arrays[b+'_errors']},examples,rows)['metrics']['nmse']
        mse=np.asarray(errors['macro']);gain=1-mse[1:]/mse[0]
        delta=base_training_aggregate({'nmse':arrays[b+'_delta'][:,None]},examples,rows)['metrics']['nmse']
        dataset_delta=base_training_aggregate({'nmse':arrays[b+'_dataset_delta'][:,None]},examples,rows)['metrics']['nmse']
        codes=arrays[b+'_codes'];roles=word_roles(codes)
        changed=np.array([codes[:,1]!=codes[:,2],roles[:,1]!=roles[:,2],codes[:,0]!=codes[:,1]]).T
        agreement=base_training_aggregate({'nmse':changed[:,None].astype(float)},examples,rows)['metrics']['nmse']
        support=base_training_aggregate({'nmse':arrays[b+'_support'][:,None].astype(float)},examples,rows)['metrics']['nmse']
        result[b]=dict(attribute_errors=errors,gain_saved_original_original_shifted=gain,context_attribute_change=delta,ID_role_and_source_replay_change=agreement,support_saved_original_original_shifted=support,
                       context_tolerance_pass=bool(np.max(delta['macro'])<.25),context_attribute_change_dataset_variance_diagnostic=dataset_delta,
                       position_pair_tail_quantiles=dict(zip(['P50','P90','P99','maximum'],np.quantile(arrays[b+'_delta'],[.5,.9,.99,1],axis=0))),
                       position_pairs_above_or_equal_0_25=(arrays[b+'_delta']>=.25).sum(0),
                       people_above_or_equal_0_25=(np.asarray([v['value'] for v in delta['by_subject']]).reshape(-1,3)>=.25).sum(0),
                       people_positive_gain_original_shifted=np.array([[sum(v['value'][stream][a]<v['value'][0][a] for v in errors['by_subject']) for a in range(3)] for stream in [2,3]]))
    np.savez_compressed(out/'paired_center_results.npz',**arrays)
    write(out/'summary.json',dict(status='natural_context_completed',candidate_windows=400,selected_pairs=sum(len(v['pairs']) for v in identities.values()),scored_physical_position_pairs=len(examples),source_reencoding_exact=all(v['saved_original_exact'] for v in identities.values()),branches=result,
         dataset_identity=identities,readout_stream_order=['background','saved_original','original_reencoded','naturally_shifted'],change_normalizer='fitted three-fold attribute variance, within-field mean',observed_both_fraction=float(arrays['observed_both'].mean()),model_center_equal_fraction=float(arrays['model_center_equal'].mean()),scope='same 60 historical train people, same 400 original arrays, at most 2 selected contexts/person; no development/confirmation/test reads'))
    print(json.dumps({b:dict(gains=v['gain_saved_original_original_shifted'],change=v['context_attribute_change']['macro'],ID_role_change=v['ID_role_and_source_replay_change']['macro'],pass_context=v['context_tolerance_pass']) for b,v in result.items()},default=lambda v:v.tolist()))


def render(out,rows,scales,pop):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((out/'summary.json').read_text())
    figure_dir=out/'figures_v2';figure_dir.mkdir(exist_ok=True)
    selection=json.loads((out/'selected_population.json').read_text())
    with np.load(out/'paired_center_results.npz') as z:data={k:z[k].copy() for k in z.files}
    labels=['theta 4–8 Hz','alpha 8–13 Hz','8 local RMS values']
    colors=['#777777','#2171b5','#e6550d'];streams=['background','original ID','shifted ID']
    figures=[];datasets=sorted(pop['native_positions']);export=[]
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    for i,b in enumerate('SO'):
        result=summary['branches'][b]
        for j,d in enumerate(datasets):
            mse=np.asarray(result['attribute_errors']['by_dataset'][d]);g=1-mse[2:]/mse[0]
            for a in range(3):
                axes[i,0].plot([j-.13,j+.13],100*g[:,a],marker=['o','s','^'][a],color=['#238b45','#6a51a3','#756bb1'][a],alpha=.8,label=labels[a] if j==0 else None)
            change=np.asarray(result['context_attribute_change']['by_dataset'][d]).reshape(3)
            for a in range(3):axes[i,1].scatter(j+(a-1)*.15,change[a],marker=['o','s','^'][a],color=['#238b45','#6a51a3','#756bb1'][a])
        axes[i,0].axhline(0,color='k',lw=.7);axes[i,0].set_ylabel('Gain over same background (%)')
        axes[i,0].set_title(f'EEG-{b}: original (left) → true shifted context (right)')
        axes[i,1].axhline(.25,color='#b30000',ls='--',label='registered context threshold 0.25')
        axes[i,1].set_ylabel('Squared change / fit attribute variance')
        axes[i,1].set_title(f'EEG-{b}: same physical 2 s, fixed fit-person readout')
        for ax in axes[i]:ax.set_xticks(range(4),['SingleTrial','REFED','Simultaneous','Visual']);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f"Paid band-and-waveform IDs: {summary['selected_pairs']} context pairs / 60 historical train people\nSame background for both contexts; equal position→crop→record→person→dataset. No decoder waveforms in this audit.")
    path=figure_dir/'context_summary.png';fig.savefig(path,dpi=220);plt.close(fig);figures.append(dict(path=str(path),kind='all-dataset attribute gain and registered context change'))
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for i,b in enumerate('SO'):
        result=summary['branches'][b]
        for j,d in enumerate(datasets):
            ids=np.array([rows[int(e[0])]['dataset']==d for e in data['examples']]);change=data[b+'_delta'][ids]
            for a in range(3):
                q=np.quantile(change[:,a],[.5,.9,.99]);axes[i,a].plot([j-.12,j,j+.12],q,marker='o',label=d if a==0 else None)
        for a,ax in enumerate(axes[i]):ax.set_yscale('symlog',linthresh=.001);ax.set_xticks(range(4),['SingleTrial','REFED','Simultaneous','Visual']);ax.set_title(f'{b} {labels[a]}: P50 / P90 / P99');ax.set_ylabel('Position-pair squared change / fit variance');ax.axhline(.25,color='#b30000',ls='--');ax.grid(alpha=.2)
    fig.suptitle('Full retained physical-position tails; quantiles are descriptive, not independent people')
    path=figure_dir/'context_change_tails.png';fig.savefig(path,dpi=220);plt.close(fig);figures.append(dict(path=str(path),kind='all-position untrimmed context-change tails'))
    for b in 'SO':
        result=summary['branches'][b]
        for person in result['attribute_errors']['by_subject']:
            d,s=person['dataset'],person['subject'];mse=np.asarray(person['value']);gains=1-mse[1:]/mse[0]
            delta=next(v['value'] for v in result['context_attribute_change']['by_subject'] if v['dataset']==d and v['subject']==s)
            changes=next(v['value'] for v in result['ID_role_and_source_replay_change']['by_subject'] if v['dataset']==d and v['subject']==s)
            row=dict(dataset=d,subject=s,branch=b)
            for stream,values in zip(['saved','original','shifted'],gains):
                for label,value in zip(['theta','alpha','RMS'],values):row[f'{stream}_{label}_gain']=value
            for label,value in zip(['theta','alpha','RMS'],np.ravel(delta)):row[f'{label}_context_change']=value
            for label,value in zip(['ID','role','saved_replay'],np.ravel(changes)):row[f'{label}_change_fraction']=value
            export.append(row)
    with (out/'subject_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=export[0]);writer.writeheader();writer.writerows(export)
    for d in datasets:
        available=np.flatnonzero([rows[int(e[0])]['dataset']==d for e in data['examples']])
        first_case=next(c for c in selection['selected'] if c['dataset']==d)
        first=next(int(i) for i in available if tuple(data['examples'][i][:2])==(first_case['crop']['row'],first_case['crop']['start_s']))
        difficulty=np.maximum(data['S_delta'][available].max(1),data['O_delta'][available].max(1));worst=int(available[np.argmax(difficulty)])
        for rule,index in [('fixed_hash_first_position',first),('largest_registered_attribute_change',worst)]:
            ri,start,pos=map(int,data['examples'][index]);row=rows[ri];name=row['positions'][pos];scale=float(scales['positions']['eeg'][d][name]['scale'])
            case=next(c for c in selection['selected'] if c['crop']['row']==ri and c['crop']['start_s']==start)
            names=pop['native_positions'][d];site=[n.upper() for n in names].index(name.upper())
            crop=dict(row=ri,start_s=start,positions=case['paired_positions'])
            _,_,old=base_tokenizer_joint_eeg_context(rows,crop,scales,names,0,allowed_splits=('train',))
            _,_,new=base_tokenizer_joint_eeg_context(rows,crop,scales,names,case['direction'],allowed_splits=('train',))
            unit=row['source_unit']
            t=np.arange(400)/200+14;fig,axes=plt.subplots(4,2,figsize=(14,13),constrained_layout=True)
            ax=axes[0,0]
            for raw,shift,color,label,ls in [(old,0,colors[1],'original 30 s X','-'),(new,2*case['direction'],colors[2],'natural shifted 30 s X','--')]:
                ax.plot(np.arange(6000)/200+shift,raw['raw_normalized_x'][:,site]*scale,color=color,lw=.7,label=label,ls=ls)
            for boundary in range(-2,33,2):ax.axvline(boundary,color='.8',lw=.4)
            ax.axvspan(14,16,color='#fed976',alpha=.35);ax.set_title('Actual X at original physical time; shared 28 s exact');ax.set_xlabel('Seconds relative to original crop');ax.set_ylabel(unit);ax.legend(fontsize=8)
            for col,key in enumerate(['x','s','o']):
                ax=axes[0,1] if col==0 else axes[col,0]
                ax.plot(t,data['true_'+key][index]*scale,color='k',lw=.7)
                ax.set_title(f'Same measured 2 s: {key.upper()} supervision' if key!='x' else 'Same measured 2 s: X input');ax.set_xlabel('Seconds relative to original crop');ax.set_ylabel(unit);ax.grid(alpha=.2)
            for r,b in enumerate('SO',1):
                codes=data[b+'_codes'][index].astype(int);roles=word_roles(codes)
                truth=data[b+'_truth'][index];pred=data[b+'_predictions'][index];back=data[b+'_background'][index]
                values=np.array([truth,back,pred[1],pred[2]])
                ax=axes[r,1]
                for k,(v,label,color) in enumerate(zip(values,['true attribute',*streams],['#000000',*colors])):
                    ax.bar(np.arange(2)+(k-1.5)*.18,v[:2]*scale**2,width=.18,label=label,color=color)
                ax.set_xticks([0,1],['theta 4–8 Hz','alpha 8–13 Hz']);ax.set_ylabel(f'{unit}²');ax.set_title(f'{b}: ID {codes[1]} → {codes[2]}; band {roles[1]} → {roles[2]}');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
                ax=axes[3,r-1]
                for v,label,color,ls in zip(values,['true local RMS',*streams],['#000000',*colors],['-','--','-','-']):
                    ax.plot(np.arange(8)*.25+14.125,v[2:]*scale,marker='o',lw=1,label=label,ls=ls,color=color)
                ax.set_title(f'{b}: 0.25 s RMS attributes, same true physical patch');ax.set_xlabel('Seconds relative to original crop');ax.set_ylabel(unit);ax.legend(fontsize=8);ax.grid(alpha=.2)
            fig.suptitle(f"{d} / {row['subject']} / {row['record']} / {name} / {start} s; fold {case['fold']}\n{rule}; context shift {2*case['direction']:+d} s; paired teacher mask complete. Paid single-ID readout, not a waveform decoder.")
            path=figure_dir/f'{d}_{rule}.png';fig.savefig(path,dpi=220);plt.close(fig)
            figures.append(dict(path=str(path),kind='same physical voltage and conditional attribute readouts',rule=rule,example=[ri,start,pos],dataset=d,subject=row['subject'],record=row['record'],position=name,fold=case['fold'],direction=case['direction'],S_IDs=data['S_codes'][index],O_IDs=data['O_codes'][index],S_context_change=data['S_delta'][index],O_context_change=data['O_delta'][index]))
    write(out/'figure_manifest.json',dict(images=figures,view_status='not_yet_opened',unit_source='native row unit and frozen position scale',attribute_coordinate='true physical voltage squared for power, voltage for RMS',waveforms='measured input/teachers only, no integer waveform decode claim'))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Paid word natural context</title>'+''.join(f'<h3>{Path(v["path"]).name}</h3><img style="width:100%;max-width:1500px" src="{Path(v["path"]).relative_to(out)}">' for v in figures))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);parser.add_argument('--render',action='store_true');parser.add_argument('--summary-only',action='store_true')
    args=parser.parse_args();cfg=yaml.safe_load((ROOT/args.config).read_text());run=ROOT/cfg['source']['run'];hard=run/cfg['source']['hard_words'];out=ROOT/args.output;out.mkdir(exist_ok=True,parents=True)
    STATE.update(family=cfg['encoding'].get('family','exchange_balanced'),pilot_books=cfg['source'].get('pilot_books'))
    if cfg['scope']['split']!='train' or any(cfg['scope'].get(k)!='forbidden' for k in ('validation_access','confirmation_access','test_access')):raise ValueError('train only natural context')
    pop=json.loads((run/cfg['source']['population']).read_text());manifest,scales,_,_=base_tokenizer_training_data(yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text()),ROOT);rows=manifest['rows']
    if args.render:
        render(out,rows,scales,pop);return
    if args.summary_only:
        summarize(out,rows,pop);return
    if len(pop['cases'])!=cfg['scope']['original_native_crops'] or len({(c['dataset'],c['subject']) for c in pop['cases']})!=cfg['scope']['original_subjects']:
        raise ValueError('frozen train population differs from registered 400 arrays and 60 people')
    candidates=defaultdict(list);excluded=[]
    for c in pop['cases']:
        row=rows[c['crop']['row']]
        if row['split']!='train' or row['subject']!=c['subject']:raise ValueError('admitted training cohort before observations')
        support={v['start_s']:set(v['positions']) for v in row['inputs']['score']};start=c['crop']['start_s'];options=[]
        for direction in [1,-1]:
            common=set(c['crop']['positions'])&support.get(start+30*direction,set())
            if common:options.append((direction,sorted(common)))
        if not options:excluded.append(dict(case=c,reason='no same-parent adjacent admitted 30s score support'));continue
        direction,common=options[0];case=dict(c,direction=direction,paired_positions=common);candidates[c['dataset'],c['subject']].append(case)
    selected=[]
    for key,values in sorted(candidates.items()):
        ordered=sorted(values,key=lambda c:ranking(cfg['selection']['seed'],c));selected.append(ordered[0])
        others=ordered[1:];different=[c for c in others if rows[c['crop']['row']]['record']!=rows[ordered[0]['crop']['row']]['record']]
        if others:selected.append((different or others)[0])
    write(out/'selected_population.json',dict(original_candidates=400,eligible_candidates=sum(map(len,candidates.values())),ineligible=excluded,selected=selected,people=[dict(dataset=d,subject=s,eligible=len(v),selected=min(2,len(v))) for (d,s),v in sorted(candidates.items())],rule=cfg['selection']))
    with np.load(run/cfg['source']['attributes']) as z:bank={k:z[k].copy() for k in z.files}
    models=build_profiles(run,hard,pop,bank,rows,out)
    STATE.update(rows=rows,scales=scales,population=pop,run=run,hard=hard,out=out,selected=selected,profiles=models)
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:list(pool.map(worker,sorted(pop['native_positions'])))
    summarize(out,rows,pop)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
