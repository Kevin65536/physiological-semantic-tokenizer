"""Cross-person Hb allocation from visible native-array space/time information."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from collections import OrderedDict,defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from threadpoolctl import threadpool_limits
from src.tokenizers.hb_array_source import lagged_array,fit_transfer,predict_transfer,allocate,LAGS
from src.tokenizers.hb_native_id_readout import load_model,apply_head
from src.tokenizers.hb_pair_allocation import optimal_delta
from src.data.dual_component_tokenizer import base_tokenizer_training_data,base_tokenizer_batch,base_tokenizer_joint_hb
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics,joint_reconstruction_summary,base_training_aggregate
from src.analysis.tokenizer_shape_diagnostics import training_weights
STATE={};ARMS=['local_X','array_X','array_ID']


def serial(v):
    if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [serial(x) for x in v]
    if isinstance(v,np.ndarray):return serial(v.tolist())
    if isinstance(v,np.generic):return serial(v.item())
    if isinstance(v,Path):return str(v)
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def write(p,v):Path(p).write_text(json.dumps(serial(v),ensure_ascii=False,indent=2))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(c):
    if (c['schema']!='hb_array_source_v1' or c['data']['numerical_read_splits']!=['train'] or any(c['data'][k]!='forbidden' for k in ['validation','confirmation','test'])
        or c['data']['teacher_refit'] or c['data']['raw_reads'] or c['model']['arms']!=ARMS or c['model']['context_seconds']!=30 or c['model']['sample_rate_hz']!=10
        or tuple(c['model']['lags_seconds'])!=LAGS or c['model']['supervised_subspace_rank']!=8 or c['model']['ridge']!=.01 or c['resources']['workers']>4):raise ValueError('fixed train-only Hb space/time source contract before arrays')


def budget():
    unit=os.environ.get('HB_ARRAY_UNIT');used=time.process_time() if not unit else int(subprocess.check_output(['systemctl','--user','show',unit,'-p','CPUUsageNSec','--value'],text=True))/1e9
    if used>STATE['cfg']['resources']['CPU_stop_seconds']:raise RuntimeError('Hb array source CPU budget reached')
    return used


def audit(p,b,e,rows,scales):
    coordinate=np.array([scales['positions']['hb'][rows[i]['dataset']][rows[i]['positions'][j]]['scale'] for i,t,j in e])
    result=joint_reconstruction_metrics(p,b,2,coordinate)
    if result['support']['nonfinite_prediction_points'].any():raise ValueError('nonfinite source prediction retained as failure')
    mask=b['mask'].reshape(-1,300,2);indices=np.arange(len(e))[:,None];color=np.arange(2)[None];peak_error=[]
    for output,key in [('sum','x'),('S','s'),('O','o')]:
        target=b[key].reshape(-1,300,2);prediction=p[output].reshape(-1,300,2);point=np.argmax(np.where(mask,abs(target),-np.inf),axis=1)
        peak_error.append((prediction[indices,point,color]-target[indices,point,color])**2/b['v'+key][:,0,:2])
    result['metrics']['true_abs_peak_nmse']=np.stack(peak_error,axis=1)
    return result,joint_reconstruction_summary(result['metrics'],result['support'],e.tolist(),rows)


def post111(a,head):
    c,n=a['ids_S'].shape[:2];missing=np.all(a['ids_S']==129,axis=2)&np.all(a['ids_O']==257,axis=2)
    code={b:np.where(missing[:,:,None],0,a['ids_'+b]).reshape(-1,15) for b in 'SO'}
    p=apply_head(code,{b:a['raw_'+b].reshape(-1,15,40) for b in ['S','O','sum']},head)
    for b in p:p[b]=p[b].reshape(c,n,15,40);p[b][missing]=0
    return p,missing


def collect(dataset,items,names,rows,scales,root,out,native):
    model=load_model(native/'native_readout.pt');source=model.source;source_sha=digest(native/'current.pt')
    oldcfg=json.loads((root/'hb_neighbor_readout/v1/resolved_config.json').read_text())
    if 'config' in oldcfg:oldcfg=oldcfg['config']
    if digest(ROOT/oldcfg['source']['checkpoint'])!=source_sha:raise ValueError('old whole-array integers do not share exact frozen source')
    oldplan=json.loads((root/'hb_neighbor_readout/v1/training_plan.json').read_text());oldplan=[v for v in oldplan if v['dataset']==dataset];lookup={(v['row'],v['start_s']):k for k,v in enumerate(oldplan)}
    cache=OrderedDict();all_examples=[];parts=defaultdict(list);crop_indices=[];reused=0;verified=0
    with np.load(root/f'hb_neighbor_readout/v1/{dataset}_train_ids.npz') as old:
        for ordinal,item in enumerate(items):
            budget();i,t=item['row'],item['start_s'];row=rows[i]
            if row['split']!='train' or row['dataset']!=dataset or not any(v['start_s']==t and set(item['positions']).issubset(v['positions']) for v in row['inputs']['train']):raise ValueError('train and eligible original crop before arrays')
            x,seen=base_tokenizer_joint_hb(rows,dict(row=i,start_s=t),scales,allowed_splits=('train',));full=np.zeros((300,len(names),2));observed=np.zeros((300,len(names)),bool)
            for local,name in enumerate(row['positions']):
                position=names.index(name);full[:,position]=x[:,local];observed[:,position]=seen[:,local].all(1)
            available=observed.all(0);missing=~available
            if (i,t) in lookup:
                k=lookup[i,t];ids={b:old[f'ids_{b}_{k}'].copy() for b in 'SO'};reused+=1
                np.testing.assert_array_equal(np.all(ids['S']==129,1)&np.all(ids['O']==257,1),missing)
                actual=source.encode(full.transpose(1,0,2).reshape(len(names),15,40).astype(np.float32))
                for b in ids:np.testing.assert_array_equal(actual[b][available],ids[b][available])
                verified+=1
            else:
                ids=source.encode(full.transpose(1,0,2).reshape(len(names),15,40).astype(np.float32))
                for b in ids:ids[b][missing]={'S':129,'O':257}[b]
            safe={b:np.where(missing[:,None],0,ids[b]) for b in 'SO'};raw=source.decode_ids(safe)
            for b in raw:raw[b][missing]=0;parts['raw_'+b].append(raw[b])
            for b in ids:parts['ids_'+b].append(ids[b])
            parts['array_X'].append(full.astype(np.float32));parts['observed'].append(observed)
            take=np.array([(i,t,j) for j in item['positions']],np.int32);b=base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=8)
            for k in ['x','s','o','mask','vx','vs','vo','dvx','dvs','dvo']:parts[k].append(b[k])
            all_examples.extend(take);crop_indices.extend([ordinal]*len(take));parts['native'].extend([names.index(row['positions'][j]) for j in item['positions']])
            write(out/'progress.json',dict(stage='collect',arrays=ordinal+1,total_arrays=len(items),CPU_seconds=budget()))
    a={k:np.stack(v) if k in ['array_X','observed','ids_S','ids_O','raw_S','raw_O','raw_sum'] else np.concatenate(v) if k not in ['native'] else np.asarray(v) for k,v in parts.items()}
    a['examples']=np.asarray(all_examples,np.int32);a['crop_index']=np.asarray(crop_indices,np.int32);a['primary_crop']=np.array([v['primary'] for v in items],bool);a['primary']=a['primary_crop'][a['crop_index']]
    np.savez_compressed(out/'prepared.npz',**a)
    write(out/'input_audit.json',dict(status='passed',source_SHA256=source_sha,arrays=len(items),primary_arrays=int(a['primary_crop'].sum()),primary_windows=int(a['primary'].sum()),reused_full_integer_arrays=reused,independently_reencoded_arrays=verified,
        full_native_positions=names,teacher_mask_used_as_input=False,observation_only_input=True,original_denominators_unchanged=True,validation_reads=0,teacher_refits=0))
    return a


def fit_dataset(dataset):
    with threadpool_limits(1):
        torch.set_num_threads(1);c,rows,scales,root,out,names=(STATE[k] for k in ['cfg','rows','scales','root','out','names']);folder=out/dataset;folder.mkdir(exist_ok=True)
        items=STATE['plan'][dataset];native=root/c['source']['native_run'];a=collect(dataset,items,names[dataset],rows,scales,root,folder,native)
        heads=torch.load(native/'decoded_waveform__ridge512_fold_heads.pt',map_location='cpu',weights_only=False)
        ex=a['examples'];C,N=a['ids_S'].shape[:2];M=2*N;E=len(ex);crop_fold=np.array([STATE['folds'][dataset,str(rows[v['row']]['subject'])] for v in items]);example_fold=crop_fold[a['crop_index']]
        xfeatures=np.stack([lagged_array(a['array_X'][j],a['observed'][j]) for j in range(C)])
        baseline={b:np.empty((E,15,40)) for b in ['S','O','sum']};predictions={arm:{b:np.empty((E,15,40)) for b in baseline} for arm in ARMS};replay={};models={};quiet=np.all(a['ids_S']==128,2)&np.all(a['ids_O']==256,2)
        bank={k:a[k] for k in ['x','s','o','mask','vx','vs','vo','dvx','dvs','dvo']}
        for fold in range(4):
            budget();post,missing=post111(a,heads[fold]);flat={b:post[b][a['crop_index'],a['native']] for b in post};target=optimal_delta(flat,bank)
            fit_crop=a['primary_crop']&(crop_fold!=fold);held_crop=crop_fold==fold;fit_examples=a['primary']&(example_fold!=fold);held_examples=example_fold==fold
            people_fit={str(rows[i]['subject']) for i,t,p in ex[fit_examples]};people_held={str(rows[v['row']]['subject']) for j,v in enumerate(items) if held_crop[j]}
            if people_fit&people_held:raise ValueError('Hb array outer person overlap')
            y=np.zeros((C,300,N,2));w=np.zeros_like(y);valid=a['mask'].reshape(E,300,2)
            weights=training_weights(ex[fit_examples],rows);inv=1/a['vs'][fit_examples,0,:2]+1/a['vo'][fit_examples,0,:2]
            exact=weights[:,None,None]*valid[fit_examples]/np.maximum(valid[fit_examples].sum(1)[:,None],1)*inv[:,None]
            indices=np.flatnonzero(fit_examples)
            for e,mass in zip(indices,exact):
                j,p=a['crop_index'][e],a['native'][e];y[j,:,p]=-target[e];w[j,:,p]=mass
                if quiet[j,p]:w[j,:,p]=0
            for b in baseline:baseline[b][held_examples]=flat[b][held_examples]
            for arm in ARMS:
                if arm=='array_ID':
                    values=np.stack([post[b].reshape(C,N,300,2).transpose(0,2,1,3) for b in ['S','O']],axis=3).reshape(C,300,N,4)
                    features=np.stack([lagged_array(values[j],np.broadcast_to((~missing[j])[None],(300,N))) for j in range(C)])
                else:features=xfeatures
                start=time.process_time();fitted=fit_transfer(features[fit_crop].reshape(-1,features.shape[-1]),y[fit_crop].reshape(-1,M),w[fit_crop].reshape(-1,M),local=(N,2) if arm=='local_X' else None)
                key=f'{arm}_fold{fold}';np.savez_compressed(folder/(key+'.npz'),**{k:(-1 if v is None else v) for k,v in fitted.items()});models[key]=dict(mode=fitted['mode'],rank=fitted['rank'],fit_people=sorted(people_fit),held_people=sorted(people_held),input_width=features.shape[-1],scalar_parameters=sum(np.asarray(v).size for k,v in fitted.items() if k in ['coefficient','projection','readout']))
                delta=predict_transfer(features[held_crop].reshape(-1,features.shape[-1]),fitted).reshape(-1,300,N,2).transpose(0,2,1,3).reshape(-1,N,15,40)
                output=allocate({b:post[b][held_crop].reshape(-1,15,40) for b in post},delta.reshape(-1,15,40),quiet[held_crop].ravel())
                held_ids=np.flatnonzero(held_crop);lookup={j:k for k,j in enumerate(held_ids)}
                for b in output:
                    pp=output[b].reshape(-1,N,15,40);predictions[arm][b][held_examples]=np.stack([pp[lookup[j],p] for j,p in zip(a['crop_index'][held_examples],a['native'][held_examples])])
                    np.testing.assert_array_equal(output['sum'],post['sum'][held_crop].reshape(-1,15,40))
                with np.load(folder/(key+'.npz')) as disk:restored={k:(str(disk[k]) if disk[k].ndim==0 and disk[k].dtype.kind in 'US' else disk[k]) for k in disk.files}
                again=predict_transfer(features[held_crop].reshape(-1,features.shape[-1]),restored);maximum=float(abs(again-delta.reshape(-1,N,300,2).transpose(0,2,1,3).reshape(-1,M)).max())
                if maximum>1e-10:raise ValueError('frozen space/time operator failed independent reload')
                replay[key]=dict(maximum_transfer_difference=maximum,fit_CPU_seconds=time.process_time()-start,role='integer-only decoder' if arm=='array_ID' else 'visible-X source estimator, not integer-only')
                write(folder/'progress.json',dict(stage='fit',fold=fold,arm=arm,CPU_seconds=budget()));print(dataset,key,'CPU',budget(),flush=True)
        payload=dict(examples=ex,primary=a['primary'],fold=example_fold)
        payload.update({'baseline_'+b:v for b,v in baseline.items()});payload.update({arm+'_'+b:v for arm,p in predictions.items() for b,v in p.items()})
        np.savez_compressed(folder/'predictions.npz',**payload);summaries={}
        for arm,p in dict(baseline111=baseline,**predictions).items():
            chosen=a['primary'];measured,s=audit({b:v[chosen] for b,v in p.items()},{k:v[chosen] for k,v in bank.items()},ex[chosen],rows,scales);summaries[arm]=s
            np.savez_compressed(folder/(arm+'_metrics.npz'),examples=ex[chosen],**measured['metrics'])
        write(folder/'summaries.json',summaries);write(folder/'fitting.json',dict(models=models,replay=replay,fold_scope='original111head_k_on_both_fit_and_held',fit_only=True,held_selection=False,CPU_seconds=budget()))
        return dataset


def verdict(summaries,metrics,examples,rows,cfg):
    raw,local,base=[summaries[k]['metrics'] for k in ['array_X','local_X','baseline111']];a,l,b=[np.asarray(v['nmse']['macro']) for v in [raw,local,base]];rule=cfg['selection']
    gain=lambda x,y:1-x[1:].mean()/y[1:].mean()
    checks=dict(array_gain_vs_local=gain(a,l)>=.05,array_gain_vs111=gain(a,b)>=.05,both_colors_vs_local=bool(np.all(a[1:].sum(0)<l[1:].sum(0))),both_colors_vs111=bool(np.all(a[1:].sum(0)<b[1:].sum(0))),
        macro_component_guard=bool(np.all(a[1:]<=1.02*b[1:])),overall_RMS=bool(np.all((np.asarray(raw['rms_gain']['macro'])[1:]>=.75)&(np.asarray(raw['rms_gain']['macro'])[1:]<=1.25))))
    for metric in ['slope_correlation','curvature_correlation']:checks[metric+'_macro']=bool(np.all(np.asarray(raw[metric]['macro'])[1:]>=np.asarray(base[metric]['macro'])[1:]-.02))
    for d in ['macro',*raw['nmse']['by_dataset']]:
        get=lambda v,key:np.asarray(v[key]['macro'] if d=='macro' else v[key]['by_dataset'][d])
        checks['true_S_peak_'+d]=bool(np.all(get(raw,'true_abs_peak_nmse')[1]<=1.05*get(base,'true_abs_peak_nmse')[1]+1e-15))
        checks['boundary_'+d]=bool(np.all(get(raw,'boundary_jump_difference_nmse')[1:]<=np.maximum(4,1.05*get(base,'boundary_jump_difference_nmse')[1:])))
    for q in [.9,.99]:checks[f'tail{q}_macro']=bool(np.all(np.quantile(metrics['array_X']['nmse'],q,0)[1:]<=1.05*np.quantile(metrics['baseline111']['nmse'],q,0)[1:]))
    subject=lambda v:{(r['dataset'],r['subject']):np.asarray(r['value'])[1:].sum() for r in v['nmse']['by_subject']}
    aa,ll,bb=[subject(v) for v in [raw,local,base]];fraction=sum(aa[k]<ll[k] and aa[k]<bb[k] for k in aa)/len(aa);checks['subjects']=fraction>=.60
    libraries=0
    for d in raw['nmse']['by_dataset']:
        da,dl,db=[np.asarray(v['nmse']['by_dataset'][d]) for v in [raw,local,base]];libraries+=int(np.all(da[1:]<dl[1:]) and np.all(da[1:]<db[1:]))
        checks['component_'+d]=bool(np.all(da[1:]<=1.05*db[1:]));r0,r1=[np.asarray(v['rms_gain']['by_dataset'][d])[1:] for v in [base,raw]];inside=(r0>=.75)&(r0<=1.25)
        checks['RMS_'+d]=bool(np.all(np.where(inside,(r1>=.75)&(r1<=1.25),abs(r1-1)<=abs(r0-1)+.01)))
        for metric in ['slope_correlation','curvature_correlation']:checks[metric+'_'+d]=bool(np.all(np.asarray(raw[metric]['by_dataset'][d])[1:]>=np.asarray(base[metric]['by_dataset'][d])[1:]-.02))
        use=np.array([rows[i]['dataset']==d for i,t,p in examples])
        for q in [.9,.99]:checks[f'tail{q}_{d}']=bool(np.all(np.quantile(metrics['array_X']['nmse'][use],q,0)[1:]<=1.05*np.quantile(metrics['baseline111']['nmse'][use],q,0)[1:]))
    checks['libraries']=libraries>=3
    return dict(numerically_eligible=all(checks.values()),checks=checks,gain_vs_local=gain(a,l),gain_vs111=gain(a,b),subject_fraction=fraction,datasets_all_SO_colors_better=libraries)


def merge():
    c,rows,scales,out=(STATE[k] for k in ['cfg','rows','scales','out']);collections=defaultdict(list);examples=[];supports=defaultdict(lambda:defaultdict(list));metrics=defaultdict(lambda:defaultdict(list))
    for d in sorted(STATE['plan']):
        folder=out/d
        with np.load(folder/'prepared.npz') as a,np.load(folder/'predictions.npz') as p:
            selected=a['primary'];e=a['examples'][selected];examples.extend(e);b={k:a[k][selected] for k in ['x','s','o','mask','vx','vs','vo','dvx','dvs','dvo']}
            for arm in ['baseline111',*ARMS]:
                z={k:p[('baseline' if arm=='baseline111' else arm)+'_'+k][selected] for k in ['S','O','sum']};r,_=audit(z,b,e,rows,scales)
                for k,v in r['metrics'].items():metrics[arm][k].append(v)
                for k,v in r['support'].items():supports[arm][k].append(v)
    examples=np.array(examples,np.int32);metrics={arm:{k:np.concatenate(v) for k,v in fields.items()} for arm,fields in metrics.items()};summaries={}
    for arm in metrics:
        summaries[arm]=joint_reconstruction_summary(metrics[arm],{k:np.concatenate(v) for k,v in supports[arm].items()},examples.tolist(),rows);np.savez_compressed(out/(arm+'_metrics.npz'),examples=examples,**metrics[arm])
    decision=verdict(summaries,metrics,examples,rows,c);write(out/'summary.json',dict(status='numerical_complete_awaiting_visual_review',decision=decision,arms=summaries,primary_arrays=sum(sum(v['primary'] for v in items) for items in STATE['plan'].values()),primary_windows=len(examples),subjects=len({(rows[i]['dataset'],rows[i]['subject']) for i,t,p in examples}),CPU_seconds=budget(),validation_reads=0,confirmation_reads=0,test_reads=0,teacher_refits=0))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--synthetic',action='store_true');a=p.parse_args();c=yaml.safe_load(a.config.read_text());validate(c);out=a.output;out.mkdir(parents=True,exist_ok=True)
    if a.synthetic:
        q=np.arange(300);v=np.sin(q[:,None,None]/11)*np.ones((1,3,2));f=lagged_array(v,np.ones((300,3),bool));y=f[:,2:6];fit=fit_transfer(f,y,np.ones_like(y));assert predict_transfer(f,fit).shape==y.shape;write(out/'synthetic.json',dict(status='passed',measured_arrays_read=0));return
    if not (out/'synthetic.json').exists():raise ValueError('synthetic must pass before measured arrays')
    lock=open(out/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'summary.json').exists():raise FileExistsError('completed source experiment retained')
    root=ROOT/c['joint_run'];g0=yaml.safe_load((ROOT/c['source']['g0_config']).read_text());manifest,scales,_,_=base_tokenizer_training_data(g0,ROOT);rows=manifest['rows']
    prior=json.loads((root/c['source']['neighbor_run']/'training_plan.json').read_text());original=json.loads((root/c['source']['fixed17']).read_text())['cases'];plan=defaultdict(list);identity=set()
    for item in prior:
        v=dict(item,primary=True);plan[item['dataset']].append(v);identity.add((item['row'],item['start_s']))
    eligibility=[]
    for item in original:
        i,t,p=item['example'];row=rows[i];eligible=any(v['start_s']==t and p in v['positions'] for v in row['inputs']['train']);eligibility.append(dict(example=item['example'],dataset=row['dataset'],subject=row['subject'],split=row['split'],train_eligible=eligible,score_eligible=any(v['start_s']==t and p in v['positions'] for v in row['inputs']['score'])))
        if row['split']!='train' or not eligible:raise ValueError('fixed diagnostic is not accessible train input')
        if (i,t) not in identity:
            positions=next(v['positions'] for v in row['inputs']['train'] if v['start_s']==t);plan[row['dataset']].append(dict(dataset=row['dataset'],subject=row['subject'],record=row['record'],row=i,start_s=t,positions=positions,primary=False));identity.add((i,t))
    if len(prior)!=236 or len({(rows[v['row']]['dataset'],rows[v['row']]['subject']) for v in prior})!=60:raise ValueError('original236/60 train whole-array identity')
    for items in plan.values():
        for item in items:
            r=rows[item['row']]
            if r['split']!='train' or not any(v['start_s']==item['start_s'] and set(item['positions']).issubset(v['positions']) for v in r['inputs']['train']):raise ValueError('train eligibility before target data')
    folds={}
    with np.load(root/c['source']['folds']) as f:
        for d,s,k in zip(f['dataset'],f['subject'],f['fold']):
            if (str(d),str(s)) in folds and folds[str(d),str(s)]!=int(k):raise ValueError('inconsistent original subject fold')
            folds[str(d),str(s)]=int(k)
    names={d:json.loads((root/c['source']['neighbor_run']/f'{d}_summary.json').read_text())['positions'] for d in plan}
    for d, ordered in names.items():
        if len(ordered)!=len(set(ordered)) or set(ordered)!=set(scales['positions']['hb'][d]):
            raise ValueError('native channel identities differ from frozen whole-array encoder')
    write(out/'resolved_config.json',c);write(out/'plan.json',dict(datasets=plan,fixed17_eligibility=eligibility,subject_folds=[dict(dataset=d,subject=s,fold=k) for (d,s),k in sorted(folds.items())],G0=manifest['cfg_identity'],scales=manifest['scales_identity'],supplement_fitting=False))
    STATE.update(cfg=c,root=root,out=out,rows=rows,scales=scales,names=names,plan=dict(plan),folds=folds)
    try:
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:list(pool.map(fit_dataset,sorted(plan)))
        merge()
    except BaseException as e:write(out/'failure.json',dict(type=type(e).__name__,reason=str(e),CPU_seconds=budget()));raise


if __name__=='__main__':main()
