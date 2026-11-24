"""Frozen-Hb source/scale controls and paid-integer analysis-by-synthesis."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,multiprocessing,os,subprocess,sys,time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from threadpoolctl import threadpool_limits
from src.tokenizers.hb_component_search import FrozenHb111,objective
from src.data.dual_component_tokenizer import base_tokenizer_training_data
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics,joint_reconstruction_summary

STATE={};DATASETS=['eeg_fnirs_single_trial','refed','simultaneous_eeg_nirs','visual_cognitive_motivation']


def write(path,value):
    def scalar(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=scalar)+'\n')


def validate(c):
    if (c['schema']!='hb_component_search_v1' or c['scope']['split']!='train' or any(c['scope'][k]!='forbidden' for k in ['validation','confirmation','test']) or c['scope']['raw_reads'] or c['scope']['teacher_refits'] or c['scope']['primary_arrays']!=236 or c['scope']['primary_positions']!=9343 or c['scope']['subjects']!=60 or c['scope']['fixed_examples']!=17 or c['model']['fixed_scale_probes']!=[.5,1.,2.] or c['model']['visible_seconds']!=30 or c['model']['sample_rate_hz']!=10 or c['resources']['workers']!=4 or c['resources']['memory_GiB']!=12):raise ValueError('fixed236/9343/17 train-only single-ID contract before arrays')


def budget():
    unit=os.environ.get('HB_COMPONENT_UNIT');used=time.process_time()
    if unit:
        raw=subprocess.check_output(['systemctl','--user','show',unit,'-p','CPUUsageNSec','--value'],text=True).strip()
        if raw.isdigit():used=int(raw)/1e9
    if used>STATE['cfg']['resources']['stop_CPU_seconds']:raise RuntimeError('Hb component stage CPU budget reached; completed evidence retained')
    return used


def metrics(pred,a,selection,rows,scales):
    ex=a['examples'][selection];bank={k:a[k][selection] for k in ['x','s','o','mask','vx','vs','vo','dvx','dvs','dvo']}
    coordinate=np.array([scales['positions']['hb'][rows[i]['dataset']][rows[i]['positions'][p]]['scale'] for i,t,p in ex])
    result=joint_reconstruction_metrics({b:v[selection] for b,v in pred.items()},bank,2,coordinate)
    mask=bank['mask'].reshape(-1,300,2);ix=np.arange(len(ex))[:,None];color=np.arange(2)[None];peaks=[]
    for b,key in [('sum','x'),('S','s'),('O','o')]:
        y=bank[key].reshape(-1,300,2);p=pred[b][selection].reshape(-1,300,2);timepoint=np.argmax(np.where(mask,abs(y),-np.inf),axis=1)
        peaks.append((p[ix,timepoint,color]-y[ix,timepoint,color])**2/bank['v'+key][:,0,:2])
    result['metrics']['true_abs_peak_nmse']=np.stack(peaks,axis=1)
    return result


def synthetic(out,root,cfg):
    from src.tokenizers.hb_component_search import objective
    out.mkdir(exist_ok=True,parents=True);torch.set_num_threads(1);start=time.process_time();m=FrozenHb111(root/cfg['source']['native'])
    rng=np.random.default_rng(93017);x=rng.normal(scale=.2,size=(4,15,40)).astype(np.float32);ids=m.encode(x)
    timings={};t=time.process_time();p=m.decode(ids,0);timings['decode_4']=time.process_time()-t
    np.testing.assert_array_equal(m.decode(ids,0)['sum'],p['sum'])
    t=time.process_time();source=m.continuous(x,ids,0);timings['continuous_4']=time.process_time()-t
    v=np.ones((4,1,40));cv=dict(S=v,O=v)
    t=time.process_time();actual,report=m.search(ids,x,v,0,source,cv);timings['source_prior_search_4_full_sweep']=time.process_time()-t
    assert all(actual[b].dtype.kind in 'iu' and actual[b].shape==(4,15) for b in 'SO')
    assert np.all(report['final_objective']<=report['initial_objective']+1e-8)
    # All reserved identities are decided inside the paid sequences, not by a hidden side input.
    quiet=dict(S=np.full((1,15),128),O=np.full((1,15),256));p0=m.decode(quiet,0);p1=m.decode(quiet,0,.5)
    for b in p0:np.testing.assert_array_equal(p0[b],p1[b])
    write(out/'synthetic.json',dict(status='passed',measured_arrays_read=0,frozen_checkpoint_read=True,complete_decoder_repeat_exact=True,quiet_fixed_scale_outputs_exact=True,search_monotone=True,timings_CPU_seconds=timings,total_CPU_seconds=time.process_time()-start,estimated_search_9343_CPU_seconds=timings['source_prior_search_4_full_sweep']/4*9343))


def profile_dataset(dataset):
    with threadpool_limits(1):
        torch.set_num_threads(1);budget();root,out,cfg,rows,scales=(STATE[k] for k in ['root','out','cfg','rows','scales'])
        folder=out/dataset;folder.mkdir(exist_ok=True)
        with np.load(root/cfg['source']['prepared']/dataset/'prepared.npz') as z:a={k:z[k].copy() for k in z.files}
        items=STATE['plan']['datasets'][dataset];folds=STATE['folds'];crop_fold=np.array([folds[dataset,str(rows[v['row']]['subject'])] for v in items]);E=len(a['examples']);C,N=a['ids_S'].shape[:2]
        if C!=len(items) or not all(rows[int(i)]['split']=='train' and rows[int(i)]['dataset']==dataset for i,t,p in a['examples']):raise ValueError('exact existing train rows required before prediction')
        model=FrozenHb111(root/cfg['source']['native']);names=['original111','fixed_scale_half','fixed_scale_double','continuous_source'];pred={name:{b:np.empty((E,15,40)) for b in ['S','O','sum']} for name in names}
        codes={key:np.empty_like(a['ids_S']) for key in ['half_S','half_O','double_S','double_O']};continuous={b:np.empty((C,N,15,40)) for b in ['S','O','sum']};identity=[]
        for j in range(C):
            budget();x=a['array_X'][j].transpose(1,0,2).reshape(N,15,40).astype(np.float32);ids={b:a['ids_'+b][j] for b in 'SO'};available=~model.missing(ids)
            fresh=model.encode(x)
            for b in 'SO':np.testing.assert_array_equal(fresh[b][available],ids[b][available])
            keep=np.flatnonzero(a['crop_index']==j);sites=a['native'][keep];f=int(crop_fold[j]);base=model.decode(ids,f)
            for b in base:pred['original111'][b][keep]=base[b][sites]
            for gamma,label,key in [(.5,'fixed_scale_half','half'),(2.,'fixed_scale_double','double')]:
                encoded=model.encode(x,ids,gamma);decoded=model.decode(encoded,f,gamma)
                for b in decoded:pred[label][b][keep]=decoded[b][sites]
                for b in 'SO':codes[key+'_'+b][j]=encoded[b]
            cont=model.continuous(x,ids,f)
            for b in cont:pred['continuous_source'][b][keep]=cont[b][sites];continuous[b][j]=cont[b]
            identity.append(dict(crop=j,row=items[j]['row'],start_s=items[j]['start_s'],fold=f,available_positions=int(available.sum()),original_ID_reencoding_exact=True))
            write(folder/'profile_progress.json',dict(completed=j+1,total=C,CPU_seconds=budget()))
        summaries={};selection=a['primary'].astype(bool)
        for arm,values in pred.items():
            result=metrics(values,a,selection,rows,scales);summaries[arm]=joint_reconstruction_summary(result['metrics'],result['support'],a['examples'][selection].tolist(),rows)
            np.savez_compressed(folder/(arm+'_profile_metrics.npz'),examples=a['examples'][selection],**result['metrics'])
            if arm=='original111':
                with np.load(root/cfg['source']['prepared']/dataset/'baseline111_metrics.npz') as original:
                    np.testing.assert_array_equal(original['examples'],a['examples'][selection])
                    np.testing.assert_allclose(result['metrics']['nmse'],original['nmse'],rtol=1e-10,atol=1e-12)
        np.savez_compressed(folder/'profile_predictions.npz',examples=a['examples'],primary=a['primary'],**{arm+'_'+b:v for arm,p in pred.items() for b,v in p.items()})
        np.savez_compressed(folder/'continuous_source.npz',**continuous);np.savez_compressed(folder/'scale_probe_integers.npz',**codes)
        write(folder/'profile_summary.json',summaries);write(folder/'profile_identity.json',dict(status='passed',arrays=C,primary_arrays=int(a['primary_crop'].sum()),primary_windows=int(selection.sum()),full_native_original_reencoding=identity,scope='train only cached unified X/teachers, no new raw reads'))
        return dataset


def merge_profile():
    cfg,root,out,rows,scales=(STATE[k] for k in ['cfg','root','out','rows','scales']);arms=['original111','fixed_scale_half','fixed_scale_double','continuous_source'];all_metrics=defaultdict(lambda:defaultdict(list));support=defaultdict(lambda:defaultdict(list));examples=[]
    for ds in DATASETS:
        with np.load(root/cfg['source']['prepared']/ds/'prepared.npz') as z:a={k:z[k].copy() for k in z.files}
        with np.load(out/ds/'profile_predictions.npz') as z:p={k:z[k].copy() for k in z.files}
        primary=a['primary'].astype(bool);examples.append(a['examples'][primary])
        for arm in arms:
            r=metrics({b:p[arm+'_'+b] for b in ['S','O','sum']},a,primary,rows,scales)
            for k,v in r['metrics'].items():all_metrics[arm][k].append(v)
            for k,v in r['support'].items():support[arm][k].append(v)
    examples=np.concatenate(examples);summaries={}
    for arm in arms:
        mm={k:np.concatenate(v) for k,v in all_metrics[arm].items()};ss={k:np.concatenate(v) for k,v in support[arm].items()};summaries[arm]=joint_reconstruction_summary(mm,ss,examples.tolist(),rows)
        np.savez_compressed(out/(arm+'_profile_metrics.npz'),examples=examples,**mm)
    if len(examples)!=9343 or len({(rows[int(i)]['dataset'],rows[int(i)]['subject']) for i,t,p in examples})!=60:raise ValueError('full236/9343/60 denominator changed')
    write(out/'profile_summary.json',dict(status='completed_profile',primary_arrays=236,primary_windows=9343,subjects=60,fixed17_examples=17,arms=summaries,selection='fixed-scale controls are diagnostic, no held-target hyperparameter selection',CPU_seconds=budget(),development_reads=0,confirmation_reads=0,test_reads=0))
    print(json.dumps({a:s['metrics']['nmse']['macro'] for a,s in summaries.items()}),flush=True)


def oracle_dataset(dataset):
    with threadpool_limits(1):
        torch.set_num_threads(1);root,out,cfg,rows=(STATE[k] for k in ['root','out','cfg','rows'])
        paired=STATE.get('paired',False);destination=out/('paired_oracle17' if paired else 'oracle17');folder=destination/dataset;folder.mkdir(exist_ok=True)
        with np.load(root/cfg['source']['prepared']/dataset/'prepared.npz') as z:a={k:z[k].copy() for k in z.files}
        with np.load(out/dataset/'continuous_source.npz') as z:continuous={k:z[k].copy() for k in z.files}
        cases=[v for v in json.loads((root/cfg['source']['fixed17']).read_text()) if v['dataset']==dataset];lookup={tuple(e):j for j,e in enumerate(a['examples'])};model=FrozenHb111(root/cfg['source']['native']);records=[];saved={}
        for ordinal,case in enumerate(cases):
            budget();e=lookup[tuple(case['example'])];crop,site=int(a['crop_index'][e]),int(a['native'][e]);fold=int(case['fold']);x=a['x'][e:e+1].astype(float);mask=a['mask'][e:e+1].astype(bool)
            ids={b:a['ids_'+b][crop,site][None].copy() for b in 'SO'};cv={b:a['v'+b.lower()][e:e+1] for b in 'SO'};vx=a['vx'][e:e+1]
            source={b:continuous[b][crop,site][None] for b in 'SO'};opposite=dict(S=x-a['o'][e:e+1],O=x-a['s'][e:e+1]);baseline=model.decode(ids,fold)
            delta=max(float(abs(opposite[b][mask]-a[b.lower()][e:e+1][mask]).max()) for b in 'SO')
            record=dict(case=case,oracle_X_minus_opposite_max_closure_difference=delta,arms={});saved[f'example_{ordinal}']=np.asarray(case['example'])
            for b in baseline:saved[f'baseline_{b}_{ordinal}']=baseline[b]
            targets=[('visible_X_only',{}),('source_prior',dict(source=source,component_variance=cv)),('true_opposite',dict(oracle=opposite,component_variance=cv,mask=mask))]
            if paired:targets.append(('individual_source_prior_guard',dict(source=source,component_variance=cv)))
            for name,target in targets:
                searcher=model.search_pairs if paired and name!='individual_source_prior_guard' else model.search
                if paired:target=dict(target,keep_X_ratio=1.05)
                start=time.process_time();sent,info=searcher(ids,x,vx,fold,full=True,progress=lambda p,b:budget(),**target);decoded=model.decode(sent,fold);again=model.decode({b:sent[b].copy() for b in 'SO'},fold)
                for b in decoded:np.testing.assert_array_equal(decoded[b],again[b]);saved[f'{name}_{b}_{ordinal}']=decoded[b]
                for b in 'SO':saved[f'{name}_ids_{b}_{ordinal}']=sent[b]
                record['arms'][name]=dict(CPU_seconds=time.process_time()-start,replay_exact=True,**info)
                write(folder/'progress.json',dict(completed_cases=ordinal,case=case['example'],arm=name,CPU_seconds=budget()))
            records.append(record);write(folder/'results.json',records);np.savez_compressed(folder/'predictions.npz',**saved)
        return dataset


def merge_oracle():
    root,out,cfg,rows,scales=(STATE[k] for k in ['root','out','cfg','rows','scales']);paired=STATE.get('paired',False);directory=out/('paired_oracle17' if paired else 'oracle17');arms=['baseline','visible_X_only','source_prior','true_opposite']+(['individual_source_prior_guard'] if paired else []);mm=defaultdict(lambda:defaultdict(list));ss=defaultdict(lambda:defaultdict(list));examples=[];observability=[]
    for ds in DATASETS:
        with np.load(root/cfg['source']['prepared']/ds/'prepared.npz') as z:a={k:z[k].copy() for k in z.files}
        records=json.loads((directory/ds/'results.json').read_text());lookup={tuple(e):j for j,e in enumerate(a['examples'])}
        with np.load(directory/ds/'predictions.npz') as p:
            for j,record in enumerate(records):
                e=lookup[tuple(record['case']['example'])];mini={k:a[k][e:e+1] for k in ['examples','x','s','o','mask','vx','vs','vo','dvx','dvs','dvo']};selection=np.ones(1,bool);examples.append(mini['examples'][0])
                for arm in arms:
                    r=metrics({b:p[f'{arm}_{b}_{j}'] for b in ['S','O','sum']},mini,selection,rows,scales)
                    for k,v in r['metrics'].items():mm[arm][k].append(v)
                    for k,v in r['support'].items():ss[arm][k].append(v)
                yy=[mini[k].reshape(300,2) for k in ['x','s','o']];center=[v-v.mean(0) for v in yy]
                cancel=1-(center[0]**2).sum(0)/np.maximum((center[1]**2+center[2]**2).sum(0),1e-20)
                observability.append(dict(case=record['case'],centered_cancellation_fraction=cancel,original_component_RMS=[np.sqrt((v*v).mean(0)) for v in yy],closure_difference=record['oracle_X_minus_opposite_max_closure_difference']))
    if len(examples)!=17:raise ValueError('fixed17 diagnostic denominator changed')
    summaries={};examples=np.asarray(examples)
    for arm in arms:
        metrics0={k:np.concatenate(v) for k,v in mm[arm].items()};support={k:np.concatenate(v) for k,v in ss[arm].items()};summaries[arm]=joint_reconstruction_summary(metrics0,support,examples.tolist(),rows);np.savez_compressed(directory/(arm+'_metrics.npz'),examples=examples,**metrics0)
    write(directory/'summary.json',dict(status='completed_fixed17_integer_reachability_diagnostic',examples=17,arms=summaries,observability=observability,scope='all original17 train diagnostics, supplements never fit',oracle_interpretation=('paired128x256 Cartesian affine ranking, top4 actual pairs verified with visible-X 1.05 per-color guard; achieved feasible oracle, not certified global minimum' if paired else 'X minus true opposite component, exhaustive exact conditional coordinate minima in one interleaved sweep; feasible achieved oracle is not a certified global dictionary minimum'),CPU_seconds=budget()))
    print(json.dumps({a:s['metrics']['nmse']['macro'] for a,s in summaries.items()}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stage',choices=['synthetic','profile','oracle17','paired_oracle17'],required=True);args=parser.parse_args()
    cfg=yaml.safe_load(args.config.read_text());validate(cfg);root=ROOT/cfg['joint_run'];out=args.output;out.mkdir(exist_ok=True,parents=True)
    STATE.update(cfg=cfg,root=root,out=out)
    if args.stage=='synthetic':synthetic(out,root,cfg);return
    if json.loads((out/'synthetic.json').read_text())['status']!='passed':raise ValueError('successful synthetic required before new measured-data use')
    lock=open(out/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.stage=='profile' and (out/'profile_summary.json').exists():raise FileExistsError('retained profile already complete')
    plan=json.loads((root/cfg['source']['prepared']/'plan.json').read_text());folds={(v['dataset'],v['subject']):v['fold'] for v in plan['subject_folds']}
    if sum(sum(v['primary'] for v in values) for values in plan['datasets'].values())!=236:raise ValueError('original236 metadata before arrays')
    manifest,scales,_,_=base_tokenizer_training_data(yaml.safe_load((ROOT/cfg['source']['g0_config']).read_text()),ROOT);rows=manifest['rows']
    for ds,values in plan['datasets'].items():
        for v in values:
            r=rows[v['row']]
            if r['split']!='train' or r['dataset']!=ds or r['subject']!=v['subject']:raise ValueError('same permitted train input identity before cached values')
    STATE.update(rows=rows,scales=scales,plan=plan,folds=folds)
    if args.stage in ['oracle17','paired_oracle17']:
        STATE['paired']=args.stage=='paired_oracle17';target=out/args.stage;target.mkdir(exist_ok=True)
        if (target/'summary.json').exists():raise FileExistsError('retained fixed17 oracle complete')
        write(target/'resolved_stage_protocol.json',dict(source_config=cfg,selection='all original17, no selection or fitting',actual_candidates_per_coordinate='all128 S / all256 O; current candidate retained',sweeps=1,paired=STATE['paired'],visible_X_guard=1.05 if STATE['paired'] else None,matched_control='individual_source_prior_guard' if STATE['paired'] else 'same exhaustive coordinate sweep',objectives=['visible_X_only','visible_X_plus_frozen_continuous_source','X_minus_true_opposite'],acceptance='exact full30s frozen111 objective; original complete-pair quiet and missing sequences unchanged',oracle='privileged achieved feasible integer dictionary result, not global lower-error certificate'))
        (target/'source_execution.py').write_text(Path(__file__).read_text());(target/'source_library.py').write_text((ROOT/'src/tokenizers/hb_component_search.py').read_text())
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:list(pool.map(oracle_dataset,DATASETS))
        merge_oracle();return
    write(out/'resolved_config.json',cfg);(out/'source_profile.py').write_text(Path(__file__).read_text())
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:list(pool.map(profile_dataset,DATASETS))
    merge_profile()


if __name__=='__main__':
    with threadpool_limits(1):torch.set_num_threads(1);main()
