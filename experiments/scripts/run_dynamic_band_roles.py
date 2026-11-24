"""Diagnose one X-driven whole-context native EEG band-role change on training folds."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.analysis.dual_component_tokenizer import base_training_aggregate, joint_reconstruction_metrics
from src.data.dual_component_tokenizer import base_tokenizer_training_data,base_tokenizer_joint_eeg_context,base_tokenizer_batch
from src.tokenizers.adaptive_band_allocation import band_waves
from src.tokenizers.band_exchange_words import load_fold_books
from src.tokenizers.cross_branch_innovations import paid_fields,decode_words
from src.tokenizers.dynamic_band_roles import select_single_role
from src.tokenizers.eeg_transient_gain import load_model
from src.utils.io import save_npz_atomic,file_sha256

STATE={}
ARMS=('frozen_word','fixed_continuous','dynamic_continuous','source_continuous')


def write(path,value):
    def scalar(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    p=Path(path);temporary=p.with_name(p.name+'.pending')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=scalar)+'\n');temporary.replace(p)


def budget():
    unit=os.environ.get('DYNAMIC_ROLE_UNIT')
    if unit:
        used=int(subprocess.check_output(['systemctl','--user','show',unit,'-p','CPUUsageNSec','--value'],text=True))/1e9
        if used>=STATE['cfg']['resources']['stage_CPU_seconds']-30:
            raise RuntimeError('registered dynamic-role stage CPU budget reached')


def load_frozen(dataset,fold):
    cfg=STATE['cfg'];run=STATE['run'];folder=run/cfg['frozen']/dataset
    with np.load(folder/f'fold_{fold}_state.npz') as z:state=dict(z)
    with np.load(folder/f'fold_{fold}_ids.npz') as z:words=dict(z)
    old={b:load_fold_books(run/cfg['pilot'],fold,b) for b in 'SO'}
    new={b:load_fold_books(run/cfg['frozen']/'books',fold,b) for b in 'SO'}
    path=run/cfg['source']/f'{dataset}_fold{fold}_transient.npz'
    source=load_model(path)
    cases=[c for c in STATE['pop']['cases'] if c['dataset']==dataset]
    fit={c['subject'] for c in cases if c['fold']!=fold}
    held={c['subject'] for c in cases if c['fold']==fold}
    if fit&held or set(state['fit_people'])!=fit or set(state['held_people'])!=held or set(source.metadata['fit_subjects'])!=fit:
        raise ValueError('original train-person cross-fitting identity changed')
    return state,words,old,new,source


def case_predictions(case,models):
    dataset=case['dataset'];names=STATE['pop']['native_positions'][dataset]
    state,words,old,new,source=models;n=len(names)
    native=np.asarray(case['native']);examples=np.array([(case['crop']['row'],case['crop']['start_s'],p) for p in case['crop']['positions']])
    lookup={tuple(e):i for i,e in enumerate(words['examples'])}
    order=np.argsort(native);indices=np.array([lookup[tuple(e)] for e in examples])[order]
    np.testing.assert_array_equal(native[order],np.arange(n))
    sent={b:words['innovation_'+b][indices] for b in 'SO'}
    x,seen,_=base_tokenizer_joint_eeg_context(STATE['rows'],case['crop'],STATE['scales'],names,0,allowed_splits=('train',))
    x=x.T.reshape(n,15,400).astype(np.float32)
    predictions=source.predict_continuous(x[None],dataset,seen[None])
    full=np.concatenate([predictions[b][0] for b in 'SO'])
    fields=np.concatenate([band_waves(np.fft.rfft(predictions[b][0].reshape(n,6000))) for b in 'SO'],axis=1)
    _,_,role=paid_fields(sent,old,new,state['scale'],'innovation',False)
    refs={k.upper():v for k,v in STATE['scales']['positions']['eeg'][dataset].items()}
    weight=np.concatenate([np.array([1/np.ravel(refs[k.upper()]['target_variance'][b])[0] for k in names])*seen for b in ('s','o','x')])
    target=np.concatenate([full,x])
    before=time.process_time()
    fixed,dynamic,changed,audit=select_single_role(fields,role,state['joint'],target,weight)
    audit['geometry_CPU_seconds']=time.process_time()-before
    audit['selected_position_name']=names[audit['selected_native']]
    audit['initial_roles']=role;audit['selected_roles']=changed
    result={}
    for label,value in (('fixed_continuous',fixed),('dynamic_continuous',dynamic)):
        result[label]={b:v[native] for b,v in zip(('S','O','sum'),np.split(value,3))}
    result['source_continuous']={b:predictions[b][0,native] for b in 'SO'}
    result['source_continuous']['sum']=result['source_continuous']['S']+result['source_continuous']['O']
    frozen=decode_words(sent,old,new,state,state,'innovation',False)
    result['frozen_word']={b:v[native] for b,v in frozen.items()}
    return examples,result,audit


def run_dataset(dataset,probe=False):
    dest=STATE['out']/dataset;dest.mkdir(exist_ok=True)
    stage='probe' if probe else 'held'
    if (dest/f'{stage}_scores.npz').exists():return
    before=time.process_time();cases=[c for c in STATE['pop']['cases'] if c['dataset']==dataset]
    models={};metrics=defaultdict(list);snapshots={};records=[];cache=OrderedDict();examples_all=[]
    for number,case in enumerate(cases):
        budget();fold=case['fold']
        if fold not in models:models[fold]=load_frozen(dataset,fold)
        examples,predictions,record=case_predictions(case,models[fold])
        # Supervised targets become visible only after X-only role selection has finished.
        batch=base_tokenizer_batch(STATE['rows'],examples,STATE['scales'],array_cache=cache,cache_limit=4)
        factors=np.array([STATE['scales']['positions']['eeg'][dataset][STATE['rows'][i]['positions'][p]]['scale'] for i,_,p in examples])
        for arm in ARMS:
            scored=joint_reconstruction_metrics(predictions[arm],batch,1,factors)
            if scored['support']['nonfinite_prediction_points'].any():raise ValueError('nonfinite geometry prediction retained as failure')
            for k,v in scored['metrics'].items():metrics[arm+'__'+k].append(v)
        for j,e in enumerate(examples):
            key=tuple(map(int,e))
            if key in STATE['display']:
                label=STATE['display'][key]
                for k in ('x','s','o','mask'):snapshots[f'{label}_{k}']=batch[k][j]
                for arm in ARMS:
                    for k,v in predictions[arm].items():snapshots[f'{label}_{arm}_{k}']=v[j]
        record.update(row=case['crop']['row'],start_s=case['crop']['start_s'],dataset=dataset,subject=case['subject'],fold=fold)
        records.append(record);examples_all.append(examples)
        if (number+1)%10==0 or probe:print(dataset,number+1,'arrays CPU',round(time.process_time()-before,2),flush=True)
        if probe:break
    save_npz_atomic(dest/f'{stage}_scores.npz',examples=np.concatenate(examples_all),**{k:np.concatenate(v) for k,v in metrics.items()})
    save_npz_atomic(dest/f'{stage}_gallery.npz',**snapshots)
    write(dest/f'{stage}_decisions.json',dict(cases=records,CPU_seconds=time.process_time()-before,stage=stage,model_identity={str(f):dict(fit_people=m[0]['fit_people'],held_people=m[0]['held_people']) for f,m in models.items()}))
    print(dataset,stage,'complete CPU',round(time.process_time()-before,2),flush=True)


def summarize():
    arrays=defaultdict(list);details=[]
    for ds in sorted(STATE['pop']['native_positions']):
        with np.load(STATE['out']/ds/'held_scores.npz') as z:
            for k in z:arrays[k].append(z[k])
        details+=json.loads((STATE['out']/ds/'held_decisions.json').read_text())['cases']
    ex=np.concatenate(arrays.pop('examples'))
    expected=np.array([(c['crop']['row'],c['crop']['start_s'],p) for c in STATE['pop']['cases'] for p in c['crop']['positions']])
    if len(ex)!=17040 or sorted(map(tuple,ex))!=sorted(map(tuple,expected)) or len(details)!=400:
        raise ValueError('original full 400/17040 scoring denominator changed')
    arms={a:base_training_aggregate({k.split('__',1)[1]:np.concatenate(v) for k,v in arrays.items() if k.startswith(a+'__')},ex,STATE['rows']) for a in ARMS}
    original=json.loads((STATE['run']/STATE['cfg']['frozen']/'summary.json').read_text())['arms']['innovation_mean']['summary']['metrics']
    for key,value in original.items():
        np.testing.assert_allclose(arms['frozen_word']['metrics'][key]['macro'],value['macro'],rtol=1e-11,atol=1e-12,equal_nan=True)
    reference=json.loads((STATE['run']/'component_source_transfer/integer_sum_source_v1/summary.json').read_text())['arms']['high_transient_source']['metrics']['nmse']
    gates={}
    for arm,v in arms.items():
        m=v['metrics'];nm=np.ravel(m['nmse']['macro']);ref=np.ravel(reference['macro']);gain=np.ravel(m['rms_gain']['macro'])
        gates[arm]=dict(X=nm[0]<=1.05*ref[0],SO=bool(np.all(nm[1:]<=1.1*ref[1:])),datasets=all(np.all(np.asarray(x)<=1.1*np.asarray(reference['by_dataset'][d])) for d,x in m['nmse']['by_dataset'].items()),
            RMS=bool(np.all((gain>=.75)&(gain<=1.25))),bands=all(np.max(np.abs(m[k]['macro']))<=3.0103 for k in ('band_4_8_db','band_4_8_abs_db','band_8_13_db','band_8_13_abs_db')),
            correlation=min(np.ravel(m['correlation']['macro']))>0,envelope=min(np.ravel(m['local_rms_correlation']['macro']))>=.5)
    fixed=np.ravel(arms['fixed_continuous']['metrics']['nmse']['macro']);dynamic=np.ravel(arms['dynamic_continuous']['metrics']['nmse']['macro'])
    population=dict(arrays=400,position_windows=17040,people=len({c['subject']+'@'+c['dataset'] for c in STATE['pop']['cases']}),folds=4)
    npz={k:np.concatenate(v) for k,v in arrays.items()};selected={tuple(map(int,e)):i for i,e in enumerate(ex)}
    difficult={}
    for panel_index,c in enumerate(STATE['draws']):
        if c['position_name'] not in ('P7','CP5','FC2'):continue
        index=selected[c['row'],c['start_s'],c['position']]
        values={a:npz[a+'__nmse'][index].reshape(3).tolist() for a in ARMS}
        ratio=np.sqrt(values['dynamic_continuous'][0]/values['fixed_continuous'][0])
        group=difficult.setdefault(c['position_name'],dict(cases=[],X_RMSE_ratio_dynamic_over_fixed=0.))
        group['cases'].append(dict(identity=c,panel_index=panel_index,original_train_variance_NMSE_X_S_O=values,X_RMSE_ratio_dynamic_over_fixed=float(ratio)))
        # Both original FC2 windows must satisfy the same condition; retain each.
        group['X_RMSE_ratio_dynamic_over_fixed']=max(group['X_RMSE_ratio_dynamic_over_fixed'],float(ratio))
    checks=dict(macro_X_improvement_at_least_2percent=bool(dynamic[0]<=.98*fixed[0]),components_worsen_at_most_2percent=bool(np.all(dynamic[1:]<=1.02*fixed[1:])),
        at_least_two_difficult_X_RMSE_improve_10percent=sum(v['X_RMSE_ratio_dynamic_over_fixed']<=.9 for v in difficult.values())>=2,
        all_difficult_X_RMSE_worsen_at_most_10percent=all(v['X_RMSE_ratio_dynamic_over_fixed']<=1.1 for v in difficult.values()),all_three_preserved=len(difficult)==3)
    histogram=np.zeros((5,5),int)
    for c in details:histogram[c['old_band'],c['new_band']]+=1
    write(STATE['out']/'summary.json',dict(arms=arms,waveform_gates=gates,population=population,continuation_checks=checks,continue_real_word_calibration=all(checks.values()),
        difficult=difficult,role_old_to_new_counts=histogram,changed_arrays=int(sum(c['old_band']!=c['new_band'] for c in details)),
        geometry_CPU_seconds=sum(c['geometry_CPU_seconds'] for c in details),finite_candidate=False,interpretation='chosen_linear_geometry_continuous_information_diagnosis_not_global_bound_or_single_ID_result',validation_reads=0,confirmation_reads=0,test_reads=0))


def review():
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    dest=STATE['out']/'visual_review_v1';dest.mkdir(exist_ok=True)
    gallery={};chosen={tuple(c[k] for k in ('row','start_s','position')):dict(c,audit_index=i) for i,c in enumerate(STATE['draws'])}
    for ds in sorted(STATE['pop']['native_positions']):
        with np.load(STATE['out']/ds/'held_gallery.npz') as z:gallery.update(dict(z))
        with np.load(STATE['out']/ds/'held_scores.npz') as z:
            ex=z['examples'];metric=z['dynamic_continuous__nmse']
        for tag,value in (('dynamic_joint_worst',metric.mean((1,2))),('dynamic_S_worst',metric[:,1,0])):
            identity=tuple(map(int,ex[np.argmax(value)]))
            if identity in chosen:chosen[identity]['selection'].append(tag);continue
            i,t,p=identity;row=STATE['rows'][i]
            chosen[identity]=dict(row=i,start_s=t,position=p,dataset=ds,subject=row['subject'],record=row['record'],position_name=row['positions'][p],source_unit=row['source_unit'],scale=STATE['scales']['positions']['eeg'][ds][row['positions'][p]]['scale'],selection=[tag],audit_index=len(chosen))
    models={};case_metadata=[]
    for identity,case in chosen.items():
        label=case['audit_index'];original=next(c for c in STATE['pop']['cases'] if (c['crop']['row'],c['crop']['start_s'])==identity[:2])
        if f'{label}_x' not in gallery:
            key=original['dataset'],original['fold']
            if key not in models:models[key]=load_frozen(*key)
            examples,values,record=case_predictions(original,models[key])
            j=int(np.flatnonzero(examples[:,2]==identity[2])[0]);batch=base_tokenizer_batch(STATE['rows'],examples,STATE['scales'])
            for k in ('x','s','o','mask'):gallery[f'{label}_{k}']=batch[k][j]
            for arm in ARMS:
                for k,v in values[arm].items():gallery[f'{label}_{arm}_{k}']=v[j]
        decisions=json.loads((STATE['out']/case['dataset']/'held_decisions.json').read_text())['cases']
        decision=next(c for c in decisions if (c['row'],c['start_s'])==identity[:2])
        case_metadata.append(dict(case,role_decision=decision))
    save_npz_atomic(dest/'display_arrays.npz',**gallery)
    colors=dict(target='#243c51',frozen_word='#8b8d93',fixed_continuous='#b77e24',dynamic_continuous='#13756c',source_continuous='#8465a2')
    styles=dict(target='-',frozen_word=':',fixed_continuous='--',dynamic_continuous='-',source_continuous='-.')
    names=dict(target='target',frozen_word='frozen K15351 word',fixed_continuous='fixed continuous band',dynamic_continuous='one dynamic continuous band',source_continuous='uncompressed X-only source')
    t=np.arange(6000)/200;figures=[];numbers=[];plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    for case in case_metadata:
        i=case['audit_index'];scale=case['scale'];mask=gallery[f'{i}_mask'].reshape(6000).astype(bool)
        values={'target':{b:gallery[f'{i}_{k}'].reshape(6000)*scale for b,k in (('X','x'),('S','s'),('O','o'))}}
        values.update({a:{b:gallery[f'{i}_{a}_{k}'].reshape(6000)*scale for b,k in (('X','sum'),('S','S'),('O','O'))} for a in ARMS})
        j=int(np.argmax(np.where(mask,abs(values['target']['X']),-1)));start=float(min(28,2*(j//400)))
        identity=f"{i:02d}_{case['dataset']}_{case['subject']}_{case['position_name']}_r{case['row']}_s{case['start_s']}"
        d=case['role_decision'];detail=f"changed {d['selected_branch']}:{d['selected_position_name']} band {d['old_band']}→{d['new_band']} | one role for all 15 slots"
        fig,axes=plt.subplots(3,3,figsize=(15,8.7),layout='constrained')
        for row,b in enumerate(('X','S','O')):
            stack=np.asarray([v[b] for v in values.values()]);lo,hi=stack.min(),stack.max();margin=.06*(hi-lo or 1)
            for col,(left,right) in enumerate(((0,30),(start,start+2),(13.5,14.5))):
                ax=axes[row,col]
                for a,v in values.items():ax.plot(t,v[b],label=names[a],color=colors[a],ls=styles[a],lw=.8 if a=='target' else .65)
                for edge in range(0,31,2):ax.axvline(edge,color='#b7c0c5',ls=':',lw=.4)
                ax.fill_between(t,lo-margin,hi+margin,where=~mask,color='#ccc',alpha=.4)
                ax.set(xlim=(left,right),ylim=(lo-margin,hi+margin),xlabel='Time in 30 s context (s)')
                if col==0:ax.set_ylabel(b+' / '+case['source_unit'])
                if row==0:ax.set_title(('Full 30 s',f'2 s at largest |X|: {j/200:.3f} s','13.5–14.5 s token boundary')[col])
            for a in ARMS:
                pred,y=values[a][b],values['target'][b];p=int(np.argmax(np.where(mask,abs(y),-1)))
                numbers.append(dict(case=i,dataset=case['dataset'],subject=case['subject'],record=case['record'],position=case['position_name'],arm=a,branch=b,
                    RMSE_native=float(np.sqrt(np.mean(np.square(pred[mask]-y[mask])))),target_peak_time_s=p/200,target_peak=float(y[p]),prediction_same_time=float(pred[p]),
                    target_rms=float(np.sqrt(np.mean(y[mask]**2))),prediction_rms=float(np.sqrt(np.mean(pred[mask]**2)))))
        fig.suptitle(identity+'\n'+detail+' | band 0/1/2/3/4 = δ/θ/α/β/γ | continuous diagnosis',fontsize=11)
        axes[0,0].legend(fontsize=6,ncol=2);path=dest/(identity+'_waveform.png');fig.savefig(path,dpi=200);plt.close(fig)
        record=dict(case=i,identity=case,waveform=str(path))
        fig,axes=plt.subplots(3,2,figsize=(13,8.3),layout='constrained');centers=np.arange(15)*2+1;hz=np.fft.rfftfreq(400,1/200)
        for row,b in enumerate(('X','S','O')):
            for a,v in values.items():
                patches=v[b].reshape(15,400);rms=np.sqrt(np.mean(patches**2,axis=1));power=abs(np.fft.rfft(patches,axis=1))**2
                bands=[np.mean(power[:,(hz>=low)&(hz<(high if high<45 else high+.001))]) for low,high in ((1,4),(4,8),(8,13),(13,30),(30,45))]
                axes[row,0].plot(centers,rms,label=names[a],color=colors[a],ls=styles[a],lw=1)
                axes[row,1].plot(range(5),np.maximum(bands,1e-20),label=names[a],color=colors[a],ls=styles[a],marker='o',ms=3)
            axes[row,0].set(xlabel='2 s patch center (s)',ylabel=f'{b} local RMS / '+case['source_unit'],xlim=(0,30))
            axes[row,1].set(yscale='log',xticks=range(5),xticklabels=['1–4','4–8','8–13','13–30','30–45'],xlabel='Frequency band (Hz)',ylabel=f'{b} mean FFT power / '+case['source_unit']+'²')
        axes[0,0].legend(fontsize=6,ncol=2);fig.suptitle(identity+'\n'+detail,fontsize=11)
        path=dest/(identity+'_dynamics.png');fig.savefig(path,dpi=200);plt.close(fig);record['dynamics']=str(path);figures.append(record)
    with (dest/'example_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(numbers[0]));writer.writeheader();writer.writerows(numbers)
    contacts=[]
    for ds in sorted(STATE['pop']['native_positions']):
        subset=[f for f in figures if f['identity']['dataset']==ds]
        for key in ('waveform','dynamics'):
            thumb=[]
            for f in subset:
                im=Image.open(f[key]);im.thumbnail((1200,700));thumb.append(im.copy())
            canvas=Image.new('RGB',(2400,740*((len(thumb)+1)//2)),(255,255,255));draw=ImageDraw.Draw(canvas)
            for j,(im,f) in enumerate(zip(thumb,subset)):
                x=1200*(j%2);y=740*(j//2);canvas.paste(im,(x,y+25));draw.text((x+8,y+5),f"Case {f['case']}: {f['identity']['subject']} {f['identity']['position_name']}",fill=(0,0,0))
            path=dest/(ds+'_'+key+'_contact.jpg');canvas.save(path,quality=91);contacts.append(str(path))
    write(dest/'manifest.json',dict(figures=figures,contacts=contacts,actual_visual_audit='pending_actual_open',units='original native voltage',scaling='target_and_all_predictions_share_same_axis_no_display_smoothing',selection='original17_plus_each_dataset_dynamic_joint_and_S_maximum'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--stage',required=True,choices=('probe','held','summary','review','all'))
    args=parser.parse_args();cfg=yaml.safe_load(args.config.read_text())
    if cfg['schema']!='dynamic_band_roles_v1' or cfg['data']!={'fit':'train','score':'train_held_people','validation':'forbidden','confirmation':'forbidden','test':'forbidden'}:raise ValueError('fixed train-only dynamic-role contract required')
    run=ROOT/cfg['run'];pop=json.loads((run/cfg['population']).read_text())
    contract=yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest,scales,_,_=base_tokenizer_training_data(contract,ROOT)
    if (pop['cfg_identity'],pop['scales_identity'])!=(manifest['cfg_identity'],manifest['scales_identity']):raise ValueError('original v3 scale/input identity mismatch')
    if len(pop['cases'])!=400 or sum(c['count'] for c in pop['cases'])!=17040:raise ValueError('original array denominator required')
    for c in pop['cases']:
        r=manifest['rows'][c['crop']['row']]
        if (r['split'],r['dataset'],r['subject'])!=('train',c['dataset'],c['subject']):raise ValueError('train-only read boundary before observations')
    draws=json.loads((run/'component_source_transfer/integer_sum_source_v1/training_examples/waveform_examples.json').read_text())['examples']
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    identity=dict(config=cfg,population_sha256=file_sha256(run/cfg['population']),cfg_identity=manifest['cfg_identity'],scales_identity=manifest['scales_identity'])
    if (out/'resolved_config.json').exists() and json.loads((out/'resolved_config.json').read_text())!=identity:raise ValueError('retained dynamic-role contract changed')
    write(out/'resolved_config.json',identity)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    STATE.update(cfg=cfg,run=run,pop=pop,rows=manifest['rows'],scales=scales,out=out,draws=draws,display={tuple(c[k] for k in ('row','start_s','position')):i for i,c in enumerate(draws)})
    for stage in (('held','summary','review') if args.stage=='all' else (args.stage,)):
        if stage in ('summary','review'):{'summary':summarize,'review':review}[stage]()
        else:
            with ProcessPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(run_dataset,d,stage=='probe') for d in sorted(pop['native_positions'])]
                for future in futures:future.result()
        write(out/(stage+'_complete.json'),dict(stage=stage,completed_time=time.strftime('%Y-%m-%dT%H:%M:%S%z')))


if __name__=='__main__':main()
