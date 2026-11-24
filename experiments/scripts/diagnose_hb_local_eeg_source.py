"""Train-only local anatomical EEG increment at a frozen strong Hb total."""
import argparse
from collections import OrderedDict,Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import fcntl,hashlib,json,multiprocessing,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/scripts'))
import numpy as np
import yaml
import torch
from threadpoolctl import threadpool_limits
from src.tokenizers.hb_local_eeg_source import fixed_neighborhoods,power_bank,pool_power,design,fit_ridge,allocate
from src.tokenizers.hb_native_id_readout import load_model
from src.tokenizers.hb_parent_distillation import validate_ids
from src.tokenizers.hb_pair_allocation import optimal_delta
from src.data.dual_component_tokenizer import base_tokenizer_training_data,base_tokenizer_joint_eeg,base_tokenizer_joint_hb
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics,base_training_aggregate
from evaluate_joint_tokenizer import write
STATE={};ARMS=('Hb_only','global_EEG','local_EEG','cyclic_EEG')


def validate(c):
    if (c['schema']!='hb_local_eeg_source_diagnostic_v1' or c['data']['allowed_splits']!=['train']
        or any(c['data'][s]!='forbidden' for s in ['validation_access','confirmation_access','test_access'])
        or c['model']['ridge']!=1.0 or c['resources']['maximum_threads']>4):raise ValueError('frozen train-only local-source contract required')


def budget():
    unit=os.environ.get('TOKENIZER_HB_LOCAL_UNIT');used=time.process_time() if not unit else int(subprocess.check_output(['systemctl','--user','show',unit,'-p','CPUUsageNSec','--value'],text=True))/1e9
    used+=float(os.environ.get('TOKENIZER_HB_LOCAL_PRIOR_CPU','0'))
    if used>=STATE['cfg']['resources']['stop_CPU_seconds']:raise RuntimeError('bounded local EEG/Hb CPU budget reached')
    return used


def batch_from_arrays(a,rows,scales):
    b={key:a['targets'][:,j] for j,key in enumerate('xso')};b['mask']=a['mask']
    for j,key in enumerate('xso'):
        b['v'+key]=np.tile(a['variances'][:,j,None,:],(1,15,20))
        dv=[scales['positions']['hb'][rows[i]['dataset']][rows[i]['positions'][p]]['difference_variance'][key] for i,_,p in a['examples']]
        b['dv'+key]=np.tile(np.asarray(dv)[:,None,:],(1,15,20))
    return b


def weights(examples,rows):
    groups=Counter((rows[i]['subject'],rows[i]['record'],rows[i]['parent'],int(start)) for i,start,p in examples)
    crops=Counter((s,r) for s,r,parent,start in groups);records=Counter(s for s,r in crops)
    return np.array([1/(records[rows[i]['subject']]*crops[rows[i]['subject'],rows[i]['record']]*groups[rows[i]['subject'],rows[i]['record'],rows[i]['parent'],int(start)]) for i,start,p in examples])


def fit_predictions(a,base,quiet,eeg,hb_valid,eeg_valid,rows,positions,folds,setting,out):
    ex=a['examples'];people=np.array([rows[i]['subject'] for i,_,p in ex]);site=np.array([positions.index(rows[i]['positions'][p]) for i,_,p in ex]);n=len(ex)
    bank=dict(s=a['targets'][:,1],o=a['targets'][:,2],mask=a['mask'],vs=a['variances'][:,1],vo=a['variances'][:,2]);target=optimal_delta(base,bank).reshape(n,300,2);mask=a['mask'].reshape(n,300,2)
    w=weights(ex,rows)[:,None,None]*mask/np.maximum(mask.sum(1)[:,None],1)*(1/a['variances'][:,1]+1/a['variances'][:,2])[:,None];w[quiet]=0
    predictions={};payload={};newfolds=np.array([folds[s] for s in people]);hb=a['targets'][:,0].reshape(n,300,2)
    for arm in ARMS:
        budget();which=ARMS.index(arm)-1;features=np.stack([design(hb[i],{b:base[b][i].reshape(300,2) for b in 'SO'},None if arm=='Hb_only' else eeg[i,:,which],hb_valid[i],eeg_valid[i,which],site[i],len(positions),setting['FIR_seconds']) for i in range(n)])
        width=features.shape[-1];stats={}
        for person in sorted(folds):
            take=people==person;x=features[take].reshape(-1,width).astype(float);y=target[take].reshape(-1,2);mass=w[take].reshape(-1,2)
            stats[person]=dict(gram=np.stack([x.T@(mass[:,c,None]*x) for c in range(2)]),right=np.stack([x.T@(mass[:,c]*y[:,c]) for c in range(2)]),mass=mass.sum(0))
        delta=np.zeros((n,300,2))
        for fold in range(3):
            fit=[s for s in folds if folds[s]!=fold];held=newfolds==fold
            gram=sum(stats[s]['gram'] for s in fit);right=sum(stats[s]['right'] for s in fit);mass=sum(stats[s]['mass'] for s in fit);coeff=[];sc=[]
            for c in range(2):
                beta,scale=fit_ridge(gram[c],right[c],mass[c],setting['ridge']);coeff.append(beta);sc.append(scale)
            beta=np.stack(coeff,1);delta[held]=features[held].astype(float)@beta;payload[f'{arm}_fold{fold}_coefficient']=beta;payload[f'{arm}_fold{fold}_scale']=np.stack(sc,1)
        predictions[arm]=allocate(base,delta.reshape(n,15,40),quiet);payload[f'{arm}_width']=np.array(width)
        write(out/'progress.json',dict(stage='crossfit_conditional_allocation',arm=arm,positions=n,CPU_seconds=budget()))
    payload['metadata']=np.array(json.dumps(dict(folds=folds,positions=positions,arms=list(ARMS),ridge=setting['ridge'],input='dense observed EEG/Hb X plus frozen Hb integer outputs; not integer-only candidate',subject_or_record_predictors=False)))
    np.savez_compressed(out/'fold_models.npz',**payload);return predictions


def dataset_worker(dataset):
    with threadpool_limits(1):
        torch.set_num_threads(1);cfg,prior,rows,scales,plan,mapping,out=(STATE[k] for k in ['cfg','prior','rows','scales','plan','mapping','out']);dest=out/dataset;dest.mkdir(exist_ok=True)
        with np.load(prior/dataset/'prepared.npz') as f:a={k:f[k].copy() for k in f if k!='features'};old_features=f['features'][:, :, 92:97].copy()
        if any(rows[i]['split']!='train' for i,_,p in a['examples']):raise ValueError('old panel train boundary')
        reference=load_model(ROOT/cfg['source']['Hb_reference'])
        if reference.metadata['cfg_identity']!=plan['signature']['G0'] or reference.metadata['scales_identity']!=plan['signature']['scales']:raise ValueError('frozen strong reference coordinates')
        ids=reference.encode(a['targets'][:,0]);base=reference.decode_ids(ids);_,quiet=validate_ids(ids);n=len(a['examples']);eeg=np.zeros((n,300,3,10),np.float32);eeg_valid=np.zeros((n,3,300),bool);hb_valid=np.zeros((n,300),bool);availability=[];cache=OrderedDict();globaldiff=0.
        positions=sorted(mapping['nearest6'][dataset]);lookup={tuple(map(int,e)):i for i,e in enumerate(a['examples'])}
        for subject,crops in plan['datasets'][dataset].items():
            for crop in crops:
                row=rows[crop['row']];hb,seen=base_tokenizer_joint_hb(rows,crop,scales,allowed_splits=('train',));power=None;observed=None;names=[];reason='missing_paired_EEG'
                if crop['eeg_row'] is not None:
                    erow=rows[crop['eeg_row']]
                    if erow['split']!='train' or erow['parent']!=row['parent']:raise ValueError('same train parent before EEG arrays')
                    admitted=any(v['start_s']==crop['start_s'] for group in erow['inputs'].values() for v in group)
                    if admitted:
                        fields,observed,_=base_tokenizer_joint_eeg(rows,dict(row=crop['eeg_row'],start_s=crop['start_s'],positions=[]),scales,erow['positions'],cache,allowed_splits=('train',));names=erow['positions'];power=power_bank(fields['x'],observed);reason='same_parent_same30s'
                for p in crop['positions']:
                    j=lookup[crop['row'],crop['start_s'],p];site=row['positions'][p];hb_valid[j]=seen[:,p].all(-1);count=[]
                    if power is not None:
                        byname={s.upper():k for k,s in enumerate(names)};allidx=np.arange(len(names));local=[byname[s.upper()] for s in mapping['nearest6'][dataset][site]];wrong=[byname[s.upper()] for s in mapping['nearest6'][dataset][mapping['cyclic_site'][dataset][site]]]
                        for mode,selected in enumerate([allidx,local,wrong]):
                            value,valid=pool_power(power,observed,selected);eeg[j,:,mode]=value;eeg_valid[j,mode]=valid;count.append(int(observed[selected].sum()))
                        globaldiff=max(globaldiff,float(abs(eeg[j,:,0,:5]-old_features[j]).max()))
                    availability.append(dict(example=a['examples'][j].tolist(),subject=subject,site=site,sync=reason,available_EEG_global_local_cyclic=count,quiet=bool(quiet[j])))
            write(dest/'progress.json',dict(stage='same_panel_local_features',subject=subject,CPU_seconds=budget()))
        if globaldiff>2e-6:raise ValueError('whole-head feature does not reproduce original EEG summary')
        np.savez_compressed(dest/'dense_features.npz',eeg=eeg,eeg_valid=eeg_valid,hb_valid=hb_valid,examples=a['examples']);np.savez_compressed(dest/'reference.npz',**base,**{'ids_'+b:v for b,v in ids.items()},quiet=quiet);write(dest/'availability.json',availability)
        predictions=fit_predictions(a,base,quiet,eeg,hb_valid,eeg_valid,rows,positions,plan['subject_folds'][dataset],cfg['model'],dest);predictions={'frozen_reference':base,**predictions};batch=batch_from_arrays(a,rows,scales);outputs={};peak={}
        for arm,pred in predictions.items():
            audit=joint_reconstruction_metrics(pred,batch,2,a['input_scales']);outputs.update({arm+'__'+k:v for k,v in audit['metrics'].items()})
            np.testing.assert_array_equal(pred['sum'],base['sum'])
            if np.max(abs(pred['S']+pred['O']-pred['sum']))>2e-12:raise ValueError('paired total closure')
            for b in base:np.testing.assert_array_equal(pred[b][quiet],base[b][quiet])
            truth=a['targets'][:,1].reshape(n,300,2);mask=a['mask'].reshape(n,300,2);at=np.argmax(np.where(mask,abs(truth),-1),axis=1);take=np.arange(n)[:,None];color=np.arange(2)[None];values=pred['S'].reshape(n,300,2);peak[arm]=np.where(mask.any(1),abs(values[take,at,color]-truth[take,at,color]),np.nan)
        np.savez_compressed(dest/'metrics.npz',examples=a['examples'],**outputs);np.savez_compressed(dest/'predictions.npz',**{arm+'_'+b:v for arm,p in predictions.items() for b,v in p.items()});np.savez_compressed(dest/'peak_errors.npz',examples=a['examples'],**peak)
        write(dest/'summary.json',dict(status='complete',position_windows=n,physical_crops=sum(len(v) for v in plan['datasets'][dataset].values()),quiet_windows=int(quiet.sum()),global_old_feature_max_abs_diff=globaldiff,CPU_seconds=time.process_time(),validation_reads=0));return dataset


def merge(out):
    cfg,rows,plan=STATE['cfg'],STATE['rows'],STATE['plan'];parts=defaultdict(list);peaks=defaultdict(list)
    for d in sorted(plan['datasets']):
        with np.load(out/d/'metrics.npz') as f:
            for k in f:parts[k].append(f[k])
        with np.load(out/d/'peak_errors.npz') as f:
            for k in f:peaks[k].append(f[k])
    parts={k:np.concatenate(v) for k,v in parts.items()};examples=parts.pop('examples');summaries={a:base_training_aggregate({k.split('__',1)[1]:v for k,v in parts.items() if k.startswith(a+'__')},examples,rows) for a in ['frozen_reference',*ARMS]};rule=cfg['selection'];local=summaries['local_EEG']['metrics'];controls={a:summaries[a]['metrics'] for a in ['Hb_only','global_EEG']};reference=summaries['frozen_reference']['metrics'];checks={};comparisons={};rng=np.random.default_rng(rule['bootstrap']['seed'])
    for control,metric in controls.items():
        l=np.asarray(local['nmse']['macro'])[1:];b=np.asarray(metric['nmse']['macro'])[1:];gain=1-l.mean()/b.mean();people={(v['dataset'],v['subject']):np.asarray(v['value'])[1:].mean() for v in metric['nmse']['by_subject']};lp={(v['dataset'],v['subject']):np.asarray(v['value'])[1:].mean() for v in local['nmse']['by_subject']};positive=np.mean([lp[k]<v for k,v in people.items()]);draw=[]
        ds={d:[k for k in people if k[0]==d] for d in plan['datasets']}
        for _ in range(rule['bootstrap']['replicates']):
            bv=[];lv=[]
            for d,keys in ds.items():
                pick=rng.integers(0,len(keys),len(keys));bv.append(np.mean([people[keys[i]] for i in pick]));lv.append(np.mean([lp[keys[i]] for i in pick]))
            draw.append(1-np.mean(lv)/np.mean(bv))
        interval=np.quantile(draw,[.025,.975]);goodds=sum(np.all(np.asarray(local['nmse']['by_dataset'][d])[1:].mean(0)<np.asarray(metric['nmse']['by_dataset'][d])[1:].mean(0)) for d in ds)
        comparisons[control]=dict(joint_relative_gain=float(gain),positive_people_fraction=float(positive),subject_bootstrap95=interval.tolist(),datasets_both_joint_colors_improve=int(goodds))
        checks[control]=bool(gain>=rule['local_joint_SO_relative_gain_over_Hb_and_global_min'] and np.all(l.mean(0)<b.mean(0)) and positive>=rule['improved_people_fraction_min'] and goodds>=rule['datasets_both_colors_improve_min'] and interval[0]>0 and all(np.all(np.asarray(local['nmse']['by_dataset'][d])[1:]<=rule['each_dataset_SO_color_max_ratio_to_Hb_and_global']*np.asarray(metric['nmse']['by_dataset'][d])[1:]) for d in ds))
    checks['frozen_reference']=bool(np.all(np.asarray(local['nmse']['macro'])[1:]<=rule['macro_each_SO_color_max_ratio_to_frozen_reference']*np.asarray(reference['nmse']['macro'])[1:]) and all(np.all(np.asarray(local['nmse']['by_dataset'][d])[1:]<=rule['each_dataset_SO_color_max_ratio_to_frozen_reference']*np.asarray(reference['nmse']['by_dataset'][d])[1:]) for d in plan['datasets']))
    peak_summaries={}
    for a in ['frozen_reference',*ARMS]:
        summary=base_training_aggregate({'nmse':np.concatenate(peaks[a])[:,None]},examples,rows)
        summary['metrics']['absolute_error_input_sd']=summary['metrics'].pop('nmse');summary.pop('selection_score');summary['component_order']=['S'];summary['units']='absolute voltage error in fixed G0 input SD at same masked true-S absolute peak'
        peak_summaries[a]=summary
    checks['true_S_peak']=bool(np.all(np.asarray(peak_summaries['local_EEG']['metrics']['absolute_error_input_sd']['macro'])<=rule['difficult_joint_error_and_true_S_peak_error_max_ratio_to_reference']*np.asarray(peak_summaries['frozen_reference']['metrics']['absolute_error_input_sd']['macro'])))
    tails={};dslabels=np.array([rows[i]['dataset'] for i,_,p in examples]);checks['tails']=True
    for d in plan['datasets']:
        take=np.flatnonzero(dslabels==d);rank=parts['frozen_reference__nmse'][take,1:].mean((1,2));tails[d]={}
        for q in rule['frozen_reference_difficulty']:
            chosen=take[rank>=np.quantile(rank,q)];ratio=float(parts['local_EEG__nmse'][chosen,1:].mean()/parts['frozen_reference__nmse'][chosen,1:].mean());tails[d][str(q)]=dict(count=len(chosen),local_to_reference_SO_NMSE_ratio=ratio,examples=examples[chosen].tolist());checks['tails']&=ratio<=rule['difficult_joint_error_and_true_S_peak_error_max_ratio_to_reference']
    write(out/'summary.json',dict(status='numerical_complete_pending_actual_images',arms=summaries,conditional_comparisons=comparisons,checks=checks,dense_eligible=bool(all(checks.values())),peak_errors=peak_summaries,tails=tails,position_windows=len(examples),physical_crops=772,people=60,validation_reads=0,confirmation_reads=0,test_reads=0,integer_stage_run=False,CPU_seconds=budget(),source_scope='dense EEG/Hb conditional allocation on historically exposed frozen reference; new heads exclude held subjects'))
    print(json.dumps(dict(comparisons=comparisons,checks=checks,NMSE={a:v['metrics']['nmse']['macro'] for a,v in summaries.items()})),flush=True)


def replay_worker(dataset):
    with threadpool_limits(1):
        torch.set_num_threads(1);cfg,prior,rows,scales,plan,mapping,out=(STATE[k] for k in ['cfg','prior','rows','scales','plan','mapping','out']);dest=out/dataset
        with np.load(prior/dataset/'prepared.npz') as f:a={k:f[k].copy() for k in f if k!='features'}
        with np.load(dest/'reference.npz') as f:ids={b:f['ids_'+b] for b in 'SO'};quiet=f['quiet'];saved={b:f[b] for b in ['S','O','sum']}
        reference=load_model(ROOT/cfg['source']['Hb_reference']);base=reference.decode_ids(ids)
        for b in base:np.testing.assert_array_equal(base[b],saved[b])
        with np.load(dest/'dense_features.npz') as f:eeg=f['eeg'];eeg_valid=f['eeg_valid'];hb_valid=f['hb_valid'];np.testing.assert_array_equal(f['examples'],a['examples'])
        people=np.array([rows[i]['subject'] for i,_,p in a['examples']]);positions=sorted(mapping['nearest6'][dataset]);site=[positions.index(rows[i]['positions'][p]) for i,_,p in a['examples']];folds=np.array([plan['subject_folds'][dataset][s] for s in people]);n=len(people);hb=a['targets'][:,0].reshape(n,300,2);batch=batch_from_arrays(a,rows,scales);maximum=0.
        with np.load(dest/'fold_models.npz') as models,np.load(dest/'predictions.npz') as expected,np.load(dest/'metrics.npz') as score:
            for arm in ARMS:
                budget();which=ARMS.index(arm)-1;features=np.stack([design(hb[i],{b:base[b][i].reshape(300,2) for b in 'SO'},None if arm=='Hb_only' else eeg[i,:,which],hb_valid[i],eeg_valid[i,which],site[i],len(positions),cfg['model']['FIR_seconds']) for i in range(n)])
                delta=np.zeros((n,300,2))
                for fold in range(3):
                    held=folds==fold;delta[held]=features[held].astype(float)@models[f'{arm}_fold{fold}_coefficient']
                prediction=allocate(base,delta.reshape(n,15,40),quiet)
                for b in prediction:np.testing.assert_array_equal(prediction[b],expected[arm+'_'+b])
                metrics=joint_reconstruction_metrics(prediction,batch,2,a['input_scales'])['metrics']
                for k,v in metrics.items():np.testing.assert_array_equal(v,score[arm+'__'+k])
        result=dict(status='passed',position_windows=n,reference_ID_only_output_exact=True,conditional_prediction_and_all_metrics_exact=True,max_difference=maximum,encoder_calls=0,native_EEG_reads=0,conditional_inputs='saved dense EEG features and observed Hb X plus frozen Hb integer outputs; not integer-only interface')
        write(dest/'independent_replay.json',result);return dataset,result


def render(cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    prior=ROOT/cfg['source']['prior'];folder=out/'figures';folder.mkdir(exist_ok=True);manifest=[]
    for d in sorted(p.name for p in out.iterdir() if p.is_dir() and (p/'predictions.npz').exists()):
        with np.load(prior/d/'prepared.npz') as f:a={k:f[k] for k in ['targets','mask','examples','input_scales']}
        with np.load(out/d/'predictions.npz') as f:pred={k:f[k] for k in f}
        with np.load(out/d/'metrics.npz') as f:error=f['frozen_reference__nmse'][:,1:].mean((1,2))
        peak=np.max(np.where(a['mask'].reshape(-1,300,2),abs(a['targets'][:,1].reshape(-1,300,2)),-1),axis=(1,2));first=min(range(len(a['examples'])),key=lambda i:tuple(a['examples'][i]));choices=[('fixed_first',first),('reference_hard',int(np.argmax(error))),('true_S_peak',int(np.argmax(peak)))];t=np.arange(300)/10
        for name,i in choices:
            target=a['targets'][i].reshape(3,300,2);mask=a['mask'][i].reshape(300,2);scale=a['input_scales'][i]
            if name=='fixed_first':center=15.
            elif name=='true_S_peak':center=np.unravel_index(np.argmax(np.where(mask,abs(target[1]),-1)),mask.shape)[0]/10
            else:
                pe=pred['frozen_reference_S'][i].reshape(300,2)-target[1];center=np.unravel_index(np.argmax(np.where(mask,pe*pe,-1)),mask.shape)[0]/10
            lo=max(0,min(28,center-1));edge=np.clip(2*round(center/2),2,28);spans=[(0,30),(lo,lo+2),(edge-.5,edge+.5)];fig,axes=plt.subplots(6,3,figsize=(18,17),layout='constrained')
            for b,(branch,truth) in enumerate([('sum',0),('S',1),('O',2)]):
                for color in range(2):
                    for column,(left,right) in enumerate(spans):
                        ax=axes[b*2+color,column];ax.plot(t,np.where(mask[:,color],target[truth,:,color]*scale,np.nan),c='black',lw=1,label='target')
                        for arm,c,style in [('frozen_reference','#d58b00',':'),('Hb_only','#777777','--'),('global_EEG','#0072b2','-'),('local_EEG','#009e73','-'),('cyclic_EEG','#cc79a7',':')]:ax.plot(t,pred[arm+'_'+branch][i].reshape(300,2)[:,color]*scale,c=c,ls=style,lw=.7,label=arm)
                        for boundary in range(0,31,2):
                            if left<=boundary<=right:ax.axvline(boundary,c='.7',ls=':',lw=.5)
                        ax.fill_between(t,0,1,where=~mask[:,color],transform=ax.get_xaxis_transform(),color='.8',alpha=.5);ax.set_xlim(left,right);ax.set_xlabel('time (s)');ax.set_ylabel(branch+'/'+['HbO','HbR'][color]+' native Hb');ax.grid(alpha=.2)
                        if b==0 and color==0:ax.set_title(['same30s target/reconstruction','fixed2s local interval','2s token boundary'][column])
            handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=6);fig.suptitle(f'{d}/{name}: originalG0 triple{tuple(a["examples"][i])}\nHbO/HbR in original subject-normalized Hb units; dense EEG/Hb diagnostic, fixed total and quiet; no smoothing for display')
            path=folder/f'{d}_{name}.png';fig.savefig(path,dpi=150);plt.close(fig);manifest.append(dict(dataset=d,rule=name,index=i,example=a['examples'][i].tolist(),path=str(path),center_seconds=center))
    write(out/'figure_manifest.json',manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--synthetic',action='store_true');p.add_argument('--render',action='store_true');p.add_argument('--replay',action='store_true');a=p.parse_args();cfg=yaml.safe_load(a.config.read_text());validate(cfg);out=a.output;out.mkdir(parents=True,exist_ok=True)
    if a.render:render(cfg,out);return
    if a.synthetic:
        x=np.sin(2*np.pi*6*np.arange(6000)/200)[:,None];power=power_bank(x,np.ones(1,bool));features,valid=pool_power(power,np.ones(1,bool),[0]);assert features.shape==(300,10) and valid.all();write(out/'synthetic.json',dict(status='passed',measured_arrays_read=0));return
    if not (out/'synthetic.json').exists():raise ValueError('synthetic before signal access')
    lock=open(out/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'summary.json').exists() and not a.replay:raise FileExistsError('preserve completed source diagnostic')
    prior=ROOT/cfg['source']['prior'];plan=json.loads((prior/'plan.json').read_text());oldcfg=yaml.safe_load((ROOT/cfg['source']['g0_contract']).read_text());manifest,scales,_,_=base_tokenizer_training_data(oldcfg,ROOT);rows=manifest['rows'];crops=[c for people in plan['datasets'].values() for cs in people.values() for c in cs]
    if len(crops)!=772 or sum(len(c['positions']) for c in crops)!=3088 or sum(len(v) for v in plan['datasets'].values())!=60 or any(rows[c['row']]['split']!='train' or (c['eeg_row'] is not None and rows[c['eeg_row']]['split']!='train') for c in crops):raise ValueError('exact prior train panel before signals')
    if manifest['cfg_identity']!=plan['signature']['G0'] or manifest['scales_identity']!=plan['signature']['scales']:raise ValueError('same original G0 cache and scale identities')
    positions={d:sorted(scales['positions']['hb'][d]) for d in plan['datasets']};source=json.loads((ROOT/cfg['source']['geometry_plan']).read_text());mapping=fixed_neighborhoods(source['records'],positions)
    STATE.update(cfg=cfg,prior=prior,rows=rows,scales=scales,plan=plan,mapping=mapping,out=out)
    if a.replay:
        if not (out/'summary.json').exists():raise ValueError('finish original source diagnostic first')
        if mapping!=json.loads((out/'fixed_geometry.json').read_text()):raise ValueError('fixed geometry changed')
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:result=dict(pool.map(replay_worker,sorted(plan['datasets'])))
        write(out/'independent_replay.json',dict(status='passed',datasets=result,position_windows=sum(v['position_windows'] for v in result.values()),integer_candidate=False,CPU_seconds=budget()));return
    write(out/'fixed_geometry.json',mapping);write(out/'resolved_config.json',cfg)
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:list(pool.map(dataset_worker,sorted(plan['datasets'])))
    merge(out)


if __name__=='__main__':main()
