"""Frozen train-person test of nonlinear, X-observable Hb event allocation."""
import argparse,fcntl,hashlib,json,os,pickle,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from threadpoolctl import threadpool_limits
from src.tokenizers.hb_event_source import event_features,centered,fit_event,predict_event,allocate_event,LAGS
from src.tokenizers.hb_native_id_readout import apply_head
from src.tokenizers.hb_pair_allocation import optimal_delta
from src.data.dual_component_tokenizer import base_tokenizer_training_data
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics,joint_reconstruction_summary
from src.analysis.tokenizer_shape_diagnostics import training_weights
STATE={};ARMS=['linear_local','nonlinear_local','nonlinear_array'];BRANCH=['sum','S','O']


def serial(v):
    if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [serial(x) for x in v]
    if isinstance(v,np.ndarray):return serial(v.tolist())
    if isinstance(v,np.generic):return serial(v.item())
    if isinstance(v,Path):return str(v)
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def write(p,v):Path(p).write_text(json.dumps(serial(v),ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(c):
    if (c['schema']!='hb_event_source_v1' or c['boundary']['numerical_read_splits']!=['train'] or any(c['boundary'][k]!='forbidden' for k in ['validation','confirmation','test']) or c['boundary']['teacher_refit'] or c['boundary']['raw_reads'] or c['model']['arms']!=ARMS or tuple(c['model']['lags_seconds'])!=LAGS or c['model']['sample_stride']!=5 or c['model']['tree_iterations']!=120 or c['model']['tree_learning_rate']!=.05 or c['model']['tree_max_leaves']!=15 or c['model']['tree_min_samples_leaf']!=128 or c['model']['tree_l2']!=10 or c['model']['tree_early_stopping'] or c['resources']['workers']>4 or c['resources']['memory_GiB']>12):raise ValueError('frozen train-only event source contract before arrays')


def budget():
    unit=os.environ.get('HB_EVENT_UNIT');used=time.process_time() if not unit else int(subprocess.check_output(['systemctl','--user','show',unit,'-p','CPUUsageNSec','--value'],text=True))/1e9
    used+=float(os.environ.get('HB_EVENT_PREVIOUS_CPU','0'))
    if used>STATE['cfg']['resources']['CPU_stop_seconds']:raise RuntimeError('registered event-source CPU budget reached')
    return used


def baseline(a,head):
    C,N=a['ids_S'].shape[:2];missing=np.all(a['ids_S']==129,2)&np.all(a['ids_O']==257,2)
    ids={b:np.where(missing[:,:,None],0,a['ids_'+b]).reshape(-1,15) for b in 'SO'}
    p=apply_head(ids,{b:a['raw_'+b].reshape(-1,15,40) for b in BRANCH},head)
    for b in p:p[b]=p[b].reshape(C,N,15,40);p[b][missing]=0
    return {b:p[b][a['crop_index'],a['native']] for b in p}


def measure(p,a,rows,scales,selected):
    ex=a['examples'][selected];bank={k:a[k][selected] for k in ['x','s','o','mask','vx','vs','vo','dvx','dvs','dvo']};p={b:v[selected] for b,v in p.items()}
    coordinate=np.array([scales['positions']['hb'][rows[i]['dataset']][rows[i]['positions'][j]]['scale'] for i,t,j in ex]);r=joint_reconstruction_metrics(p,bank,2,coordinate);m=bank['mask'].reshape(-1,300,2)
    metrics={k:[] for k in ['centered_nmse','centered_peak_nmse','peak_time_error_s','peak_amplitude_gain']}
    for b,k in zip(BRANCH,['x','s','o']):
        target=centered(bank[k].reshape(-1,300,2),m);pred=centered(p[b].reshape(-1,300,2),m);var=bank['v'+k][:,0,:2];support=m.sum(1).clip(1)
        metrics['centered_nmse'].append(np.where(m,(target-pred)**2,0).sum(1)/support/var)
        peak=np.argmax(np.where(m,abs(target),-np.inf),1);idx=np.arange(len(ex))[:,None];col=np.arange(2)[None]
        metrics['centered_peak_nmse'].append((pred[idx,peak,col]-target[idx,peak,col])**2/var)
        actual=np.argmax(np.where(m,abs(pred),-np.inf),1);metrics['peak_time_error_s'].append(abs(actual-peak)/10)
        metrics['peak_amplitude_gain'].append(np.max(np.where(m,abs(pred),0),1)/np.maximum(np.max(np.where(m,abs(target),0),1),1e-12))
    for k,v in metrics.items():
        value=np.stack(v,1);value[np.broadcast_to((m.sum(1)==0)[:,None],value.shape)]=np.nan;r['metrics'][k]=value
    return r,joint_reconstruction_summary(r['metrics'],r['support'],ex.tolist(),rows)


def worker(dataset):
    with threadpool_limits(1):
        torch.set_num_threads(1);c,rows,out,source,root=(STATE[k] for k in ['cfg','rows','out','source','root']);folder=out/dataset;folder.mkdir(exist_ok=True)
        items=STATE['plan']['datasets'][dataset]
        # Reject every non-train or ineligible row before compressed signal data.
        for v in items:
            row=rows[v['row']]
            if row['split']!='train' or not any(x['start_s']==v['start_s'] and set(v['positions']).issubset(x['positions']) for x in row['inputs']['train']):raise ValueError('original admitted G0 train support required before arrays')
        with np.load(source/dataset/'prepared.npz') as disk:a={k:disk[k] for k in disk.files}
        ex=a['examples'];E=len(ex);foldmap={(x['dataset'],x['subject']):x['fold'] for x in STATE['plan']['subject_folds']};fold=np.array([foldmap[dataset,str(rows[i]['subject'])] for i,t,j in ex]);C,N=a['ids_S'].shape[:2]
        quiet=np.all(a['ids_S'][a['crop_index'],a['native']]==128,1)&np.all(a['ids_O'][a['crop_index'],a['native']]==256,1)
        heads=torch.load(root/c['source']['native']/'decoded_waveform__ridge512_fold_heads.pt',map_location='cpu',weights_only=False)
        banks={k:a[k] for k in ['x','s','o','mask','vx','vs','vo','dvx','dvs','dvo']};mask=a['mask'].reshape(E,300,2);phase=np.array([int(hashlib.sha256(f'{i}:{t}:{j}'.encode()).hexdigest()[:12],16)%5 for i,t,j in ex]);timeindex=np.arange(60)[None]*5+phase[:,None]
        results={arm:{b:np.empty((E,15,40)) for b in BRANCH} for arm in ['baseline111',*ARMS]};details=[]
        whichfold=STATE.get('fold');whicharm=STATE.get('arm')
        for f in ([whichfold] if whichfold is not None else range(4)):
            budget();base=baseline(a,heads[f]);held=fold==f;fit=a['primary']&(fold!=f);fitpeople=sorted({str(rows[i]['subject']) for i,t,j in ex[fit]});heldpeople=sorted({str(rows[i]['subject']) for i,t,j in ex[held]})
            if set(fitpeople)&set(heldpeople):raise ValueError('person folds overlap')
            y=centered(-optimal_delta(base,banks),mask);weights=training_weights(ex[fit],rows);w=np.zeros((E,300,2));inverse=1/a['vs'][fit,0,:2]+1/a['vo'][fit,0,:2];w[fit]=weights[:,None,None]*mask[fit]/mask[fit].sum(1).clip(1)[:,None,:]*inverse[:,None];w[quiet]=0
            np.testing.assert_array_equal(held&a['primary'],(fold==f)&a['primary'])
            for b in BRANCH:results['baseline111'][b][held]=base[b][held]
            localbase=np.stack([base[b].reshape(E,300,2) for b in 'SO'],2).reshape(E,300,4)
            local=event_features(a['array_X'],a['observed'],localbase,a['crop_index'],a['native'],False);array=None
            for arm in ([whicharm] if whicharm else ARMS):
                budget();dest=folder/f'{arm}_fold{f}.npz';modelpath=folder/f'{arm}_fold{f}.pkl'
                if dest.exists():
                    with np.load(dest) as z:
                        np.testing.assert_array_equal(z['examples'],ex[held]);pp={b:z[b] for b in BRANCH}
                    for b in BRANCH:results[arm][b][held]=pp[b]
                    print(dataset,f,arm,'restored',flush=True);continue
                start=time.process_time()
                if arm=='nonlinear_array':
                    if array is None:array=event_features(a['array_X'],a['observed'],localbase,a['crop_index'],a['native'],True)
                    features=array
                else:features=local
                train=features[fit][np.arange(fit.sum())[:,None],timeindex[fit]].reshape(-1,features.shape[-1]);target=y[fit][np.arange(fit.sum())[:,None],timeindex[fit]].reshape(-1,2);weight=w[fit][np.arange(fit.sum())[:,None],timeindex[fit]].reshape(-1,2)
                fitted=fit_event(train,target,weight,'linear' if arm=='linear_local' else 'nonlinear',c['model']['seed']);fitted.update(fit_people=fitpeople,held_people=heldpeople,fold=f,dataset=dataset,config_SHA256=STATE['config_sha'],source_prepared_SHA256=STATE['assets'][str(source/dataset/'prepared.npz')])
                modelpath.write_bytes(pickle.dumps(fitted,protocol=5));restored=pickle.loads(modelpath.read_bytes());delta=predict_event(features[held],restored);output=allocate_event({b:base[b][held] for b in BRANCH},delta,quiet[held])
                # Reload equality from frozen inputs, and exact declared total/quiet.
                np.testing.assert_array_equal(predict_event(features[held][:2],fitted),predict_event(features[held][:2],restored))
                np.testing.assert_array_equal(output['sum'],base['sum'][held])
                for b in BRANCH:np.testing.assert_array_equal(output[b][quiet[held]],base[b][held][quiet[held]]);results[arm][b][held]=output[b]
                np.testing.assert_allclose(output['S']+output['O'],output['sum'],atol=1e-12,rtol=1e-12)
                np.savez_compressed(dest,examples=ex[held],**output)
                details.append(dict(arm=arm,fold=f,fit_people=fitpeople,held_people=heldpeople,fit_windows=int(fit.sum()),all_fit_windows_retained=True,sampled_time_rows=len(train),color_rows=fitted['color_rows'],feature_width=features.shape[-1],fit_CPU_seconds=time.process_time()-start,model_bytes=modelpath.stat().st_size,integer_interface=False))
                write(folder/'progress.json',dict(fold=f,arm=arm,CPU_seconds=budget(),details=details));print(dataset,f,arm,'CPU',budget(),flush=True)
            del local,array
        if whichfold is not None or whicharm:return dataset
        np.savez_compressed(folder/'predictions.npz',examples=ex,fold=fold,primary=a['primary'],**{arm+'_'+b:v for arm,p in results.items() for b,v in p.items()});summaries={}
        for arm,p in results.items():
            measured,s=measure(p,a,rows,STATE['scales'],a['primary']);summaries[arm]=s;np.savez_compressed(folder/(arm+'_metrics.npz'),examples=ex[a['primary']],**measured['metrics']);np.savez_compressed(folder/(arm+'_support.npz'),**measured['support'])
        write(folder/'summaries.json',summaries);write(folder/'fit_audit.json',dict(details=details,primary_windows=int(a['primary'].sum()),primary_arrays=int(a['primary_crop'].sum()),teacher_refits=0,validation_reads=0));return dataset


def merge():
    out,rows=STATE['out'],STATE['rows'];metrics={arm:{} for arm in ['baseline111',*ARMS]};supports={arm:{} for arm in metrics};examples=[];summaries={}
    for d in sorted(STATE['plan']['datasets']):
        for arm in metrics:
            with np.load(out/d/(arm+'_metrics.npz')) as z:
                if arm=='baseline111':examples.extend(z['examples'])
                for k in z.files:
                    if k!='examples':metrics[arm].setdefault(k,[]).append(z[k])
            with np.load(out/d/(arm+'_support.npz')) as z:
                for k in z.files:supports[arm].setdefault(k,[]).append(z[k])
    ex=np.array(examples);metrics={arm:{k:np.concatenate(v) for k,v in fields.items()} for arm,fields in metrics.items()}
    for arm,m in metrics.items():summaries[arm]=joint_reconstruction_summary(m,{k:np.concatenate(v) for k,v in supports[arm].items()},ex.tolist(),rows);np.savez_compressed(out/(arm+'_metrics.npz'),examples=ex,**m)
    decisions={};ref=summaries['baseline111']['metrics'];b=np.array(ref['nmse']['macro']);bc=np.array(ref['centered_nmse']['macro'])
    for arm in ARMS:
        mm=summaries[arm]['metrics'];a=np.array(mm['nmse']['macro']);ac=np.array(mm['centered_nmse']['macro']);gain=1-a[1:].mean()/b[1:].mean();cgain=1-ac[1:].mean()/bc[1:].mean();checks=dict(joint_gain=gain>=.05,centered_gain=cgain>=.05,macro_component=bool(np.all(a[1:]<=1.02*b[1:])))
        people=lambda x:{(r['dataset'],r['subject']):np.array(r['value'])[1:].mean() for r in x['nmse']['by_subject']};aa,bb=people(mm),people(ref);fraction=np.mean([aa[k]<bb[k] for k in aa]);checks['people']=fraction>=.60;libraries=0
        for d in ['macro',*mm['nmse']['by_dataset']]:
            get=lambda m,k:np.array(m[k]['macro'] if d=='macro' else m[k]['by_dataset'][d]);checks['peak_'+d]=bool(np.all(get(mm,'centered_peak_nmse')[1]<=1.05*get(ref,'centered_peak_nmse')[1]+1e-15));checks['boundary_'+d]=bool(np.all(get(mm,'boundary_jump_difference_nmse')[1:]<=1.05*get(ref,'boundary_jump_difference_nmse')[1:]+1e-15))
            for field in ['slope_correlation','curvature_correlation']:checks[field+'_'+d]=bool(np.all(get(mm,field)[1:]>=get(ref,field)[1:]-.02))
            if d!='macro':checks['components_'+d]=bool(np.all(get(mm,'nmse')[1:]<=1.05*get(ref,'nmse')[1:]));libraries+=int(np.all(get(mm,'nmse')[1:]<get(ref,'nmse')[1:]))
            use=np.ones(len(ex),bool) if d=='macro' else np.array([rows[i]['dataset']==d for i,t,j in ex])
            for q in [.9,.99]:checks[f'tail{q}_{d}']=bool(np.all(np.quantile(metrics[arm]['nmse'][use],q,0)[1:]<=1.05*np.quantile(metrics['baseline111']['nmse'][use],q,0)[1:]))
        checks['libraries']=libraries>=3;decisions[arm]=dict(eligible=all(checks.values()),checks=checks,joint_gain=gain,centered_gain=cgain,improved_people_fraction=fraction,improved_libraries=libraries)
    if len(ex)!=9343 or len({(rows[i]['dataset'],rows[i]['subject']) for i,t,j in ex})!=60:raise ValueError('full original9343/60 denominator changed')
    write(out/'summary.json',dict(status='numerical_complete_awaiting_visual_review',arms=summaries,decisions=decisions,primary_windows=len(ex),primary_arrays=236,people=60,validation_reads=0,confirmation_reads=0,test_reads=0,teacher_refits=0,CPU_seconds=budget()))


def render():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out,source,rows,scales=(STATE[k] for k in ['out','source','rows','scales']);original=json.loads((source/'review_v1/examples.json').read_text());chosen={tuple(v['example']):['original17'] for v in original}
    for d in sorted(STATE['plan']['datasets']):
        with np.load(out/d/'nonlinear_array_metrics.npz') as z:
            ex=z['examples'];scores={'new_joint_worst':z['nmse'][:,1:].mean((1,2)),'new_centered_S_peak_worst':z['centered_peak_nmse'][:,1].mean(1)}
            for label,v in scores.items():chosen.setdefault(tuple(ex[np.argmax(v)]),[]).append(label)
    folder=out/'figures';folder.mkdir(exist_ok=True);cases=[];time=np.arange(300)/10
    for d in sorted(STATE['plan']['datasets']):
        with np.load(source/d/'prepared.npz') as z:a={k:z[k] for k in z.files}
        with np.load(out/d/'predictions.npz') as z:p={k:z[k] for k in z.files}
        for identity,labels in chosen.items():
            i,st,j=identity;row=rows[i]
            if row['dataset']!=d:continue
            index,=np.flatnonzero(np.all(a['examples']==identity,1));mask=a['mask'][index].reshape(300,2);scale=scales['positions']['hb'][d][row['positions'][j]]['scale'];ys={b:a[k][index].reshape(300,2)*scale for b,k in zip(BRANCH,['x','s','o'])};ps={arm:{b:p[arm+'_'+b][index].reshape(300,2)*scale for b in BRANCH} for arm in ['baseline111',*ARMS]}
            peak=int(np.argmax(np.where(mask[:,0],abs(ys['S'][:,0]-ys['S'][mask[:,0],0].mean()),-np.inf)));lo=max(0,min(28,time[peak]-1));fig,axes=plt.subplots(6,4,figsize=(16,12),layout='constrained')
            for bi,b in enumerate(BRANCH):
                for color in range(2):
                    waves=[ys[b][:,color],*[ps[arm][b][:,color] for arm in ['baseline111',*ARMS]]];valid=mask[:,color]
                    for col,ax in enumerate(axes[2*bi+color]):
                        span=(0,30) if col in [0,2] else (lo,lo+2);sel=(time>=span[0])&(time<=span[1]);iscenter=col>=2
                        for v,label,style,co in zip(waves,['target','111','linear local','nonlinear local','nonlinear array'],['-','--','-.',':','-'],['#151515','#247a9f','#b7892e','#8a61ac','#c74a30']):
                            v=v-v[valid].mean() if iscenter else v;ax.plot(time,np.where(valid,v,np.nan),ls=style,color=co,lw=1.05,label=label)
                        for t in range(0,31,2):
                            if span[0]<=t<=span[1]:ax.axvline(t,color='.75',lw=.5)
                        if (~valid).any():ax.fill_between(time,0,1,where=~valid,transform=ax.get_xaxis_transform(),color='.8',step='mid',alpha=.7)
                        ax.set_xlim(span);ax.tick_params(labelsize=7);ax.set_xlabel('s',fontsize=8)
                        if col==0:ax.set_ylabel(b+' '+['HbO','HbR'][color],fontsize=8)
                        if bi==color==0:ax.set_title(['30s original amplitude','2s at true centered S-HbO peak','30s demeaned','2s demeaned'][col],fontsize=9)
            axes[0,0].legend(fontsize=6,ncol=2);fig.suptitle(f"{d} | {row['subject']} | {row['record']} | {row['positions'][j]} | row{i} +{st}s | {row['source_unit']} | fold{p['fold'][index]}\n"+', '.join(labels)+'; original masks; same curve coordinates; 2s token lines',fontsize=10)
            path=folder/f'{len(cases):02}_{d}_{row["subject"]}.png';fig.savefig(path,dpi=200);plt.close(fig);cases.append(dict(example=identity,dataset=d,subject=row['subject'],record=row['record'],position=row['positions'][j],start_s=st,unit=row['source_unit'],scale=scale,fold=p['fold'][index],selection=labels,file=str(path.relative_to(out)),primary=bool(a['primary'][index]),centered_S_HbO_peak_s=time[peak]))
    write(out/'figure_manifest.json',dict(status='rendered_pending_actual_opening',cases=cases))


def replay_worker(dataset):
    """No supervised waveform, variance or scoring mask is read here."""
    with threadpool_limits(1):
        torch.set_num_threads(1);source,out,root,c=(STATE[k] for k in ['source','out','root','cfg'])
        fields=['array_X','observed','ids_S','ids_O','raw_S','raw_O','raw_sum','crop_index','native','examples']
        with np.load(source/dataset/'prepared.npz') as z:a={k:z[k] for k in fields}
        heads=torch.load(root/c['source']['native']/'decoded_waveform__ridge512_fold_heads.pt',map_location='cpu',weights_only=False);checks=[]
        quiet=np.all(a['ids_S'][a['crop_index'],a['native']]==128,1)&np.all(a['ids_O'][a['crop_index'],a['native']]==256,1)
        for f in range(4):
            base=baseline(a,heads[f])
            with np.load(out/dataset/f'nonlinear_array_fold{f}.npz') as disk:
                wanted=disk['examples'];lookup={tuple(v):j for j,v in enumerate(a['examples'])};held=np.array([lookup[tuple(v)] for v in wanted])
            localbase=np.stack([base[b].reshape(-1,300,2)[held] for b in 'SO'],2).reshape(len(held),300,4);features={}
            for arm in ARMS:
                budget();array=arm=='nonlinear_array'
                if array not in features:features[array]=event_features(a['array_X'],a['observed'],localbase,a['crop_index'][held],a['native'][held],array)
                fitted=pickle.loads((out/dataset/f'{arm}_fold{f}.pkl').read_bytes());delta=predict_event(features[array],fitted);prediction=allocate_event({b:base[b][held] for b in BRANCH},delta,quiet[held])
                maximum=0.
                with np.load(out/dataset/f'{arm}_fold{f}.npz') as saved:
                    np.testing.assert_array_equal(saved['examples'],wanted)
                    for b in BRANCH:np.testing.assert_array_equal(prediction[b],saved[b]);maximum=max(maximum,float(abs(prediction[b]-saved[b]).max()))
                checks.append(dict(fold=f,arm=arm,examples=len(held),maximum_absolute_difference=maximum))
        write(out/dataset/'independent_replay.json',dict(status='passed',allowed_signal_fields=fields,supervised_target_fields_read=[],checks=checks,CPU_seconds=budget()));return dataset


def observability():
    """Target-assisted feasible projection and fixed event diagnostics, no fit."""
    out,source,rows,scales=(STATE[k] for k in ['out','source','rows','scales']);summaries={};matches={};events=[]
    for d in sorted(STATE['plan']['datasets']):
        budget()
        with np.load(source/d/'prepared.npz') as z:a={k:z[k] for k in z.files}
        with np.load(out/d/'predictions.npz') as z:p={k:z[k] for k in z.files}
        base={b:p['baseline111_'+b] for b in BRANCH};mask=a['mask'].reshape(-1,300,2);quiet=np.all(a['ids_S'][a['crop_index'],a['native']]==128,1)&np.all(a['ids_O'][a['crop_index'],a['native']]==256,1)
        delta=centered(-optimal_delta(base,a),mask);oracle=allocate_event(base,delta,quiet);r,s=measure(oracle,a,rows,scales,a['primary']);summaries[d]=s
        np.savez_compressed(out/d/'target_assisted_projected_metrics.npz',examples=a['examples'][a['primary']],**r['metrics'])
        with np.load(source/d/'baseline111_metrics.npz') as old,np.load(out/d/'baseline111_metrics.npz') as new:
            checks={k:float(np.nanmax(abs(new[k]-old[k]))) for k in old.files if k!='examples'}
            for k in checks:np.testing.assert_array_equal(new[k],old[k])
            matches[d]=checks
        selected=a['primary'];idx=np.flatnonzero(selected);E=len(idx);m=mask[selected];target=centered(a['s'][selected].reshape(-1,300,2),m);peak=np.argmax(np.where(m,abs(target),-np.inf),1);xx=centered(a['array_X'][a['crop_index'][selected],:,a['native'][selected]],None)
        for n,j in enumerate(idx):
            i,st,site=a['examples'][j];row=rows[i];scale=scales['positions']['hb'][d][row['positions'][site]]['scale'];case=dict(example=a['examples'][j],dataset=d,subject=row['subject'],record=row['record'],position=row['positions'][site],start_s=st,unit=row['source_unit'],quiet=quiet[j],fold=p['fold'][j],colors=[])
            for color in range(2):
                t=int(peak[n,color]);cell=dict(color=['HbO','HbR'][color],true_centered_S_peak_s=t/10,true_centered_S_peak_native=target[n,t,color]*scale,X_at_true_S_peak_native=xx[n,t,color]*scale,X_centered_peak_s=int(np.argmax(abs(xx[n,:,color])))/10,models={})
                for arm in ['baseline111',*ARMS]:
                    z=centered(p[arm+'_S'][j:j+1].reshape(1,300,2),mask[j:j+1])[0,:,color];cell['models'][arm]=dict(value_at_true_peak_native=z[t]*scale,own_abs_peak_s=int(np.argmax(np.where(mask[j,:,color],abs(z),-np.inf)))/10,centered_RMS_native=np.sqrt(np.mean(z[mask[j,:,color]]**2))*scale)
                case['colors'].append(cell)
            events.append(case)
    write(out/'observability.json',dict(status='completed',fit=False,target_assisted_role='diagnostic feasible centered correction in same0.2Hz/DC/quiet/fixed111sum output class; hidden target needed, not a deployed source and not a model-independent lower bound',target_assisted_by_dataset=summaries,original111_metric_maximum_differences=matches,all9343_event_table='event_observability_table.json',teacher_refits=0,validation_reads=0));write(out/'event_observability_table.json',events)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['synthetic','fit','render','replay','observability'],required=True);p.add_argument('--dataset');p.add_argument('--fold',type=int);p.add_argument('--arm',choices=ARMS);args=p.parse_args();c=yaml.safe_load(args.config.read_text());validate(c);out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.stage=='synthetic':
        from src.tokenizers.hb_event_source import project_event
        x=np.sin(np.arange(300)[None,:,None,None]/19)*np.ones((2,1,3,2));f=event_features(x,np.ones(x.shape[:3],bool),np.zeros((2,300,4)),[0,1],[0,2],True);np.testing.assert_allclose(project_event(x[:,:,0]).mean(1),0,atol=1e-14);write(out/'synthetic.json',dict(status='passed',feature_shape=f.shape,measured_arrays_read=0));return
    if not (out/'synthetic.json').exists():raise ValueError('synthetic checks required before measurements')
    lock=open(out/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=ROOT/c['joint_run'];source=root/c['source']['prepared'];plan=json.loads((source/'plan.json').read_text());m,scales,_,_=base_tokenizer_training_data(yaml.safe_load((ROOT/c['source']['g0_config']).read_text()),ROOT);rows=m['rows']
    if plan['G0']!=m['cfg_identity'] or plan['scales']!=m['scales_identity']:raise ValueError('original G0 and scales identity required')
    if sum(sum(x['primary'] for x in v) for v in plan['datasets'].values())!=236:raise ValueError('original236 crop denominator')
    assets={str(source/d/'prepared.npz'):sha(source/d/'prepared.npz') for d in plan['datasets']};assets[str(root/c['source']['native']/'decoded_waveform__ridge512_fold_heads.pt')]=sha(root/c['source']['native']/'decoded_waveform__ridge512_fold_heads.pt')
    state=dict(cfg=c,root=root,out=out,source=source,rows=rows,scales=scales,plan=plan,assets=assets,config_sha=sha(args.config),fold=args.fold,arm=args.arm);STATE.update(state)
    if args.stage=='render':render();return
    if args.stage=='observability':observability();return
    if args.stage=='replay':
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:list(pool.map(replay_worker,sorted(plan['datasets'])))
        write(out/'replay_complete.json',dict(status='passed',supervised_fields_read=[],validation_reads=0,CPU_seconds=budget()));return
    write(out/'resolved_config.json',c);write(out/'input_manifest.json',dict(assets=assets,G0=m['cfg_identity'],scales=m['scales_identity'],original_plan=str(source/'plan.json'),config_SHA256=sha(args.config),dataset_folds=plan['subject_folds'],primary_arrays=236,primary_windows=9343,people=60,input_teacher_mask=False,validation_reads=0));datasets=[args.dataset] if args.dataset else sorted(plan['datasets'])
    start=time.time()
    try:
        with ProcessPoolExecutor(max_workers=min(4,len(datasets)),mp_context=multiprocessing.get_context('fork')) as pool:list(pool.map(worker,datasets))
        if args.dataset is None and args.fold is None and args.arm is None:merge()
        write(out/('probe_complete.json' if args.fold is not None else 'fit_complete.json'),dict(CPU_seconds=budget(),wall_seconds=time.time()-start,stage=args.stage))
    except BaseException as e:write(out/'failure.json',dict(type=type(e).__name__,reason=str(e),CPU_seconds=budget()));raise


if __name__=='__main__':main()
