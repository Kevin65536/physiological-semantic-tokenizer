"""Independent physical-voltage audit of frozen continuous EEG band exchanges."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_joint_eeg, base_tokenizer_batch
from src.tokenizers.eeg_transient_gain import load_model as load_source
from src.analysis.dual_component_tokenizer import base_training_aggregate

STATE={}
BANDS=[(1,4),(4,8),(8,13),(13,30),(30,45)]


def write(path,value):
    def scalar(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(value,indent=2,default=scalar,ensure_ascii=False)+'\n')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def independent_bands(values):
    wave=np.asarray(values,float).reshape(len(values),6000)
    spectrum=np.fft.rfft(wave,axis=-1)
    hz=np.fft.rfftfreq(6000,1/200)
    fields=[]
    for lo,hi in BANDS:
        mask=(hz>=lo)&((hz<=hi) if hi==45 else (hz<hi))
        fields.append(np.fft.irfft(spectrum*mask,n=6000,axis=-1).reshape(len(values),15,400))
    return np.array(fields)


def interpolate_patch(target,covariance,anchors):
    if not len(anchors):return np.zeros_like(target)
    # Solve for the actual 400 voltage samples, independent of the producer's
    # residual quadratic form and precomputed interpolation matrices.
    values=np.linalg.solve(covariance[np.ix_(anchors,anchors)],target[anchors])
    return covariance[:,anchors]@values


def physical_decode(fields,roles,covariance,weights):
    wave=np.zeros(fields.shape[1:]);sse=np.zeros((5,15))
    for band in range(5):
        for patch in range(15):
            target=fields[band,:,patch]
            anchors=np.flatnonzero(roles[:,patch]==band)
            prediction=interpolate_patch(target,covariance[band],anchors)
            wave[:,patch]+=prediction
            sse[band,patch]=np.einsum('ct,c,ct->',prediction-target,weights,prediction-target)
    hz=np.fft.rfftfreq(6000,1/200)
    spectrum=np.fft.rfft(wave.reshape(len(roles),6000),axis=-1)
    spectrum[:,(hz<1)|(hz>45)]=0
    return np.fft.irfft(spectrum,n=6000,axis=-1).reshape(wave.shape),sse.sum(0)


def exact_patch_search(fields,covariance,seen,weights,patch):
    sites=np.flatnonzero(seen);fixed=sites%5
    baseline={b:sites[fixed==b] for b in range(5)}
    def cost(b,anchors):
        target=fields[b,:,patch]
        diff=interpolate_patch(target,covariance[b],np.sort(anchors))-target
        return float(np.einsum('ct,c,ct->',diff,weights,diff))
    original=[cost(b,baseline[b]) for b in range(5)]
    choices=[];pairs_checked=0
    for left in range(5):
        for right in range(left+1,5):
            best=(0.,None)
            for a in baseline[left]:
                for b in baseline[right]:
                    aa=np.where(baseline[left]==a,b,baseline[left]);bb=np.where(baseline[right]==b,a,baseline[right])
                    change=cost(left,aa)+cost(right,bb)-original[left]-original[right]
                    pairs_checked+=1
                    if change<best[0]:best=(change,(int(a),int(b)))
            choices.append((left,right,*best))
    optimal=(0.,[])
    for j,(left,right,change,pair) in enumerate(choices):
        if change<optimal[0]:optimal=(change,[pair])
        for k in range(j):
            a,b,second,pair2=choices[k]
            if pair is None or pair2 is None or left in (a,b) or right in (a,b):continue
            if change+second<optimal[0]:optimal=(change+second,[pair,pair2])
    roles=np.full(len(seen),-1,np.int8);roles[sites]=fixed
    for a,b in optimal[1]:roles[a],roles[b]=roles[b],roles[a]
    return roles,float(optimal[0]),pairs_checked


def refit_covariance(cases,samples,source,names,rows,dataset):
    fitting=[(c,x) for c,x in zip(cases,samples) if c['fold']!=0]
    frequency=Counter((c['subject'],rows[c['crop']['row']]['record']) for c,x in fitting)
    record_count=Counter(person for person,record in frequency)
    n=len(names);sum_cross={b:np.zeros((5,n,n)) for b in 'SO'};mass=np.zeros((n,n))
    for c,(x,seen) in fitting:
        estimate=source.predict_continuous(x,dataset,seen[None])
        weight=1/(frequency[c['subject'],rows[c['crop']['row']]['record']]*record_count[c['subject']])
        for b in 'SO':
            field=independent_bands(estimate[b][0]).reshape(5,n,6000)
            sum_cross[b]+=weight*(field@field.swapaxes(-1,-2))/6000
        mass+=weight*(seen[:,None]&seen[None])
    result={}
    for b in 'SO':
        values=[]
        for gram in sum_cross[b]:
            cov=gram/np.maximum(mass,1e-12);values_eigen,vectors=np.linalg.eigh((cov+cov.T)/2)
            psd=(vectors*np.maximum(values_eigen,0))@vectors.T
            ridge=np.maximum(np.diag(psd),max(float(np.trace(psd))/n*1e-6,1e-12))
            values.append(psd+.01*np.diag(ridge))
        result[b]=np.array(values)
    return result


def worker(dataset):
    rows,scales,pop,run,out,selected=(STATE[k] for k in ['rows','scales','population','run','out','selected'])
    here=out/dataset;here.mkdir(exist_ok=True)
    cases=[c for c in pop['cases'] if c['dataset']==dataset];names=pop['native_positions'][dataset];n=len(names)
    samples=[]
    for c in cases:
        row=rows[c['crop']['row']]
        if row['split']!='train' or row['subject']!=c['subject']:raise ValueError('training scope before signal reads')
        fields,seen,_=base_tokenizer_joint_eeg(rows,dict(c['crop'],positions=[]),scales,names,allowed_splits=('train',))
        samples.append((fields['x'].T.reshape(1,n,15,400).astype(np.float32),seen))
    sources=[load_source(run/'component_source_transfer/transient_gain_v1'/f'{dataset}_fold{f}_transient.npz') for f in range(4)]
    covariance=[];cov_checks=[]
    for fold in range(4):
        expected={c['subject'] for c in cases if c['fold']!=fold}
        if set(sources[fold].metadata['fit_subjects'])!=expected or sources[fold].metadata['held_fold']!=fold:raise ValueError('source has held person')
        with np.load(run/'component_source_transfer/adaptive_band_v1'/dataset/f'fold_{fold}_model.npz') as z:
            if set(z['fit_people'])!=expected:raise ValueError('covariance fit people differ')
            np.testing.assert_array_equal(z['names'],names)
            covariance.append({b:z[b].copy() for b in 'SO'})
            for b in 'SO':
                if np.linalg.eigvalsh(z[b]).min()<=0:raise ValueError('covariance is not positive definite')
        cov_checks.append(dict(fold=fold,fit_people=sorted(expected),held_people=sorted({c['subject'] for c in cases if c['fold']==fold}),covariance_sha=sha(run/'component_source_transfer/adaptive_band_v1'/dataset/f'fold_{fold}_model.npz')))
    refit=refit_covariance(cases,samples,sources[0],names,rows,dataset)
    covariance_gap={b:float(np.abs(refit[b]-covariance[0][b]).max()) for b in 'SO'}
    for b in 'SO':np.testing.assert_allclose(refit[b],covariance[0][b],rtol=1e-10,atol=2e-11)
    measured=defaultdict(list);examples=[];probes=[];images=[];quotas=0;max_gap=0.;cache=OrderedDict();max_proxy_gap=0.
    for fold in range(4):
        held=[(c,x) for c,x in zip(cases,samples) if c['fold']==fold]
        with np.load(run/'component_source_transfer/adaptive_band_exchange_v1'/dataset/f'fold_{fold}_roles.npz') as z:roles={k:z[k].copy() for k in z.files}
        with np.load(run/'component_source_transfer/adaptive_band_exchange_v1'/dataset/f'fold_{fold}_metrics.npz') as z:metrics={k:z[k].copy() for k in z.files}
        with np.load(run/'component_source_transfer/adaptive_band_exchange_v1'/dataset/f'fold_{fold}_proxy.npz') as z:proxy=z['changes'].copy()
        expected_examples=np.asarray([(c['crop']['row'],c['crop']['start_s'],p) for c,x in held for p in c['crop']['positions']])
        np.testing.assert_array_equal(metrics['examples'],expected_examples)
        pointer=0
        for j,(c,(x,seen)) in enumerate(held):
            dense=sources[fold].predict_continuous(x,dataset,seen[None]);fields={b:independent_bands(dense[b][0]) for b in 'SO'}
            count=c['count'];crop=c['crop'];ex=np.asarray([(crop['row'],crop['start_s'],p) for p in crop['positions']]);examples.extend(ex)
            batch=base_tokenizer_batch(rows,ex,scales,array_cache=cache,cache_limit=8);native=c['native']
            predictions={arm:{} for arm in ['fixed','exchange']}
            for ib,b in enumerate('SO'):
                weights=np.array([1/np.ravel(scales['positions']['eeg'][dataset][name]['target_variance'][b.lower()])[0] for name in names]);weights[~seen]=0
                losses={}
                for arm in ['fixed','exchange']:
                    tags=roles[arm+'_'+b][j]
                    if tags.shape!=(n,15) or not np.all(tags[~seen]==-1):raise ValueError('invalid missing roles')
                    wanted=np.bincount(np.flatnonzero(seen)%5,minlength=5)
                    for t in range(15):
                        np.testing.assert_array_equal(np.bincount(tags[seen,t],minlength=5),wanted);quotas+=1
                    wave,losses[arm]=physical_decode(fields[b],tags,covariance[fold][b],weights)
                    predictions[arm][b]=wave[native]
                change=losses['exchange']-losses['fixed'];saved=proxy[2*j+ib]
                scale=max(1.,float(np.max(np.abs(losses['fixed']))))
                max_proxy_gap=max(max_proxy_gap,float(np.abs(change-saved).max()))
                np.testing.assert_allclose(change,saved,rtol=1e-9,atol=1e-9*scale)
                if change.max()>1e-9*scale:raise ValueError('physical source SSE increased')
                if j==0 and fold==0:
                    expected_roles,delta,pairs=exact_patch_search(fields[b],covariance[fold][b],seen,weights,7)
                    np.testing.assert_array_equal(expected_roles,roles['exchange_'+b][j,:,7])
                    np.testing.assert_allclose(delta,saved[7],rtol=1e-9,atol=1e-9*scale)
                    probes.append(dict(fold=fold,branch=b,crop=crop,patch=7,physical_pair_searches=pairs,minimum_voltage_SSE_change=delta,actual_roles_exact=True))
            for arm,pred in predictions.items():
                pred['sum']=pred['S']+pred['O'];score=[];rmse=[]
                for k,b in [('x','sum'),('s','S'),('o','O')]:
                    truth=batch[k].astype(float);mask=batch['mask'].astype(bool);error=pred[b]-truth
                    mse=np.where(mask,error**2,0).sum((1,2))/mask.sum((1,2));variance=batch['v'+k][:,0,0]
                    score.append(mse/variance)
                    factors=np.array([scales['positions']['eeg'][dataset][rows[i]['positions'][p]]['scale'] for i,t,p in ex])
                    rmse.append(np.sqrt(mse)*factors)
                score=np.array(score).T[...,None];native_rmse=np.array(rmse).T[...,None]
                expected=metrics[arm+'__nmse'][pointer:pointer+count]
                max_gap=max(max_gap,float(np.abs(score-expected).max()))
                np.testing.assert_allclose(score,expected,rtol=2e-11,atol=1e-11)
                np.testing.assert_allclose(native_rmse,metrics[arm+'__rmse_native'][pointer:pointer+count],rtol=2e-11,atol=1e-11)
                measured[arm].append(score)
            for pos_index,e in enumerate(ex):
                identity=tuple(map(int,e))
                if identity not in selected:continue
                record=dict(STATE['selected'][identity]);record['fold']=fold
                arrays={k:batch[k][pos_index] for k in ['x','s','o','mask']}
                for arm,pred in predictions.items():
                    for b,wave in pred.items():arrays[arm+'_'+b]=wave[pos_index]
                for b in 'SO':arrays['roles_'+b]=roles['exchange_'+b][j]
                arrays['observed']=seen
                path=here/f"example_{record['audit_index']:03d}.npz";np.savez_compressed(path,**arrays)
                record['array']=str(path.relative_to(out));images.append(record)
            pointer+=count
            if j%10==0:write(here/'progress.json',dict(fold=fold,completed_crops=j+1,planned=len(held),CPU_seconds=time.process_time()))
    np.savez_compressed(here/'independent_nmse.npz',examples=examples,**{k:np.concatenate(v) for k,v in measured.items()})
    result=dict(dataset=dataset,crops=len(cases),positions=len(examples),quota_checks=quotas,maximum_original_NMSE_gap=max_gap,
                maximum_saved_proxy_physical_SSE_gap=max_proxy_gap,covariance_fold0_time_domain_refit_gap=covariance_gap,
                covariance_and_source_folds=cov_checks,exact_physical_pair_searches=probes,examples=images,CPU_seconds=time.process_time())
    write(here/'review.json',result);return result


def render_dataset(task):
    run,out,dataset=task
    return render_review(run,out,dataset)


def render_review(run,out,dataset=None):
    if dataset is None:
        original=json.loads((out/'summary.json').read_text())
        datasets=sorted({v['dataset'] for v in original['examples']})
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:
            records=[v for part in pool.map(render_dataset,[(run,out,d) for d in datasets]) for v in part]
        records.sort(key=lambda v:v['identity']['audit_index'])
        write(out/'figure_manifest.json',dict(status='rendered_pending_actual_image_inspection',examples=records,figure_count=2*len(records),interface=original.get('interface_description','continuous source-band samples and role tags; independent integer compression belongs to a separate stage')))
        print('Rendered',2*len(records),'figures from',len(records),'predeclared and hard examples')
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((out/'summary.json').read_text())
    destination=out/'waveforms';destination.mkdir(exist_ok=True)
    palette={'target':'#234b68','fixed':'#cd8543','exchange':'#168274'}
    style={'target':'-','fixed':'--','exchange':':'}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    records=[]
    for case in summary['examples']:
        if case['dataset']!=dataset:continue
        index=case['audit_index'];scale=float(case['scale'])
        with np.load(out/case['array']) as z:a={k:z[k].copy() for k in z.files}
        waves={'target':{b:a[k].reshape(6000)*scale for b,k in [('X','x'),('S','s'),('O','o')]}}
        for model in ['fixed','exchange']:waves[model]={b:a[model+'_'+k].reshape(6000)*scale for b,k in [('X','sum'),('S','S'),('O','O')]}
        mask=a['mask'].reshape(6000).astype(bool);time_axis=np.arange(6000)/200
        identity=f"{case['dataset']} | {case['subject']} | {case['record']} | {case['position_name']} | crop +{case['start_s']}s"
        fig,axes=plt.subplots(3,3,figsize=(15,8.5),layout='constrained')
        for row,b in enumerate(['X','S','O']):
            values=np.array([waves[m][b] for m in waves]);lower=values.min();upper=values.max();margin=.07*(upper-lower or 1)
            for col,(left,right) in enumerate([(0,30),(14,16),(13.5,14.5)]):
                ax=axes[row,col]
                for model in waves:ax.plot(time_axis,waves[model][b],color=palette[model],ls=style[model],lw=.8,label=model)
                for boundary in range(0,32,2):ax.axvline(boundary,color='#b9c3c8',lw=.5,ls=':')
                ax.fill_between(time_axis,lower-margin,upper+margin,where=~mask,color='#c4c9cd',alpha=.5)
                ax.set(xlim=(left,right),ylim=(lower-margin,upper+margin),xlabel='Time within visible context (s)')
                if col==0:ax.set_ylabel(b+' / '+case['source_unit'])
                if row==0:ax.set_title(['Full 30 s','Fixed 14–16 s','Fixed 13.5–14.5 s boundary'][col])
        axes[0,0].legend(fontsize=8);fig.suptitle(identity+' | '+case.get('interface_label','continuous band waveform + role tags'),fontsize=11)
        stem=f"{index:03d}_{case['dataset']}_{case['subject']}_{case['position_name']}"
        path=destination/(stem+'_voltage.png');fig.savefig(path,dpi=220,facecolor='white');plt.close(fig)
        fig,axes=plt.subplots(3,2,figsize=(13.4,8.3),layout='constrained')
        patch_valid=a['mask'].astype(bool).all(-1);block_valid=mask.reshape(120,50).all(-1)
        hz=np.fft.rfftfreq(400,1/200);window=np.hanning(400)
        metrics={}
        for row,b in enumerate(['X','S','O']):
            metrics[b]={}
            for j,model in enumerate(waves):
                y=waves[model][b];envelope=np.sqrt((y.reshape(120,50)**2).mean(-1));envelope[~block_valid]=np.nan
                axes[row,0].plot(np.arange(120)*.25+.125,envelope,color=palette[model],ls=style[model],lw=1,label=model)
                patches=y.reshape(15,400)[patch_valid];centered=patches-patches.mean(-1,keepdims=True)
                power=np.abs(np.fft.rfft(centered*window,axis=-1))**2*2/(400*(window**2).sum())
                energy=[power[:,(hz>=lo)&((hz<=hi) if hi==45 else (hz<hi))].sum(-1).mean() for lo,hi in BANDS]
                axes[row,1].bar(np.arange(5)+(j-1)*.25,np.maximum(energy,1e-25),.24,color=palette[model],label=model,hatch='//' if model=='exchange' else None)
                metrics[b][model]=dict(band_energy=energy,complete_patches=int(patch_valid.sum()))
            axes[row,0].set(xlim=(0,30),xlabel='Time within visible context (s)',ylabel=b+' 0.25 s RMS / '+case['source_unit'])
            axes[row,1].set(xticks=range(5),xticklabels=['1–4','4–8','8–13','13–30','30–45'],yscale='log',xlabel='Frequency band (Hz)',ylabel=b+' mean patch power / source unit²')
        axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8);fig.suptitle(identity+' | complete-patch bands and quarter-second RMS',fontsize=10)
        dynamics=destination/(stem+'_dynamics.png');fig.savefig(dynamics,dpi=220,facecolor='white');plt.close(fig)
        peak={}
        for b in ['X','S','O']:
            selected_time=int(np.argmax(np.where(mask,abs(waves['target'][b]),-1)))
            peak[b]=dict(time_s=selected_time/200,target=float(waves['target'][b][selected_time]),fixed=float(waves['fixed'][b][selected_time]),exchange=float(waves['exchange'][b][selected_time]),fixed_RMSE=float(np.mean((waves['fixed'][b][mask]-waves['target'][b][mask])**2)**.5),exchange_RMSE=float(np.mean((waves['exchange'][b][mask]-waves['target'][b][mask])**2)**.5))
        record=dict(identity=case,voltage=str(path.relative_to(out)),dynamics=str(dynamics.relative_to(out)),target_extrema=peak,band_energy=metrics)
        records.append(record)
        write(out/dataset/'render_progress.json',dict(completed=len(records),planned=sum(c['dataset']==dataset for c in summary['examples'])))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args();run=ROOT/args.run;out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    if args.render:
        render_review(run,out)
        return
    poppath=run/'component_source_transfer/hard_as_v1/local_information_v1/population.json';pop=json.loads(poppath.read_text())
    manifest,scales,_,_=base_tokenizer_training_data(yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text()),ROOT)
    if (pop['cfg_identity'],pop['scales_identity'])!=(manifest['cfg_identity'],manifest['scales_identity']):raise ValueError('G0 identities changed')
    rows=manifest['rows'];gallery=json.loads((run/'component_source_transfer/integer_sum_source_v1/training_examples/waveform_examples.json').read_text())['examples']
    selected={}
    def add(c,rule):
        key=c['row'],c['start_s'],c['position']
        if key not in selected:selected[key]=dict(c,audit_selection=[rule],audit_index=len(selected))
        else:selected[key]['audit_selection'].append(rule)
    for c in gallery:add(c,'original_'+','.join(c['selection']))
    for dataset in pop['native_positions']:
        all_ex=[];scores=[]
        for f in range(4):
            with np.load(run/'component_source_transfer/adaptive_band_exchange_v1'/dataset/f'fold_{f}_metrics.npz') as z:all_ex.append(z['examples']);scores.append(z['exchange__nmse'])
        ex=np.concatenate(all_ex);score=np.concatenate(scores)
        for rule,value in [('new_joint_worst',score.mean((1,2))),('new_S_worst',score[:,1,0])]:
            i,t,p=map(int,ex[np.argmax(value)]);row=rows[i]
            case=dict(dataset=dataset,subject=row['subject'],record=row['record'],parent=row['parent'],row=i,start_s=t,position=p,position_name=row['positions'][p],scale=scales['positions']['eeg'][dataset][row['positions'][p]]['scale'],source_unit=row['source_unit'])
            add(case,rule)
    write(out/'selected_examples.json',dict(examples=list(selected.values()),rules='Original 17 fixed/random/K128 P90/old high-worst plus each dataset new joint and S worst, deduplicated without deleting any original.'))
    STATE.update(rows=rows,scales=scales,population=pop,run=run,out=out,selected=selected)
    results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:
        futures=[pool.submit(worker,d) for d in sorted(pop['native_positions'])]
        for future in as_completed(futures):results.append(future.result())
    all_ex=[];parts=defaultdict(list)
    for record in results:
        with np.load(out/record['dataset']/'independent_nmse.npz') as z:
            all_ex.append(z['examples']);parts['fixed'].append(z['fixed']);parts['exchange'].append(z['exchange'])
    ex=np.concatenate(all_ex);metrics={arm:base_training_aggregate({'nmse':np.concatenate(v)},ex,rows)['metrics']['nmse'] for arm,v in parts.items()}
    original=json.loads((run/'component_source_transfer/adaptive_band_exchange_v1/summary.json').read_text())
    for arm in parts:np.testing.assert_allclose(metrics[arm]['macro'],original['arms'][arm]['metrics']['nmse']['macro'],rtol=2e-11,atol=1e-11)
    write(out/'summary.json',dict(status='independent_physical_voltage_review_passed',stage='continuous_band_waveforms_and_role_tags',is_single_ID_codec=False,metrics=metrics,datasets=results,
        examples=sorted([c for r in results for c in r['examples']],key=lambda c:c['audit_index']),quota_checks=sum(r['quota_checks'] for r in results),validation_reads=0,confirmation_reads=0,test_reads=0))
    print('Independent covariance, quota, physical SSE and G0 score review passed',len(ex),'positions')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
