"""Versioned whole-population S/O production stages of run_dual_component_tokenizer."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
import os
import resource
import time

import numpy as np

from experiments.scripts import evaluate_shared_driver_reconstruction as solver
from experiments.scripts.run_dual_component_tokenizer import owner_lock, read_json, write_json, write_npz
from src.data import so_supervision_cache as so
from src.data.downstream_input_cache import DownstreamInputDataset
from src.data.registry import REGISTERED_DATASETS
from src.data.dual_component_tokenizer import BANDS, voltage_power_cap_target
from src.inference.observation_baselines import eeg_band_power


def require(value, message):
    if not value:
        raise ValueError(message)


def output_grid(g):
    """Retain the native identity while recording the actual output clock explicitly."""
    rate = g['rate']
    start = round(g['start_sample']*200/rate)
    n = round(g['duration_s']*10)
    require(rate in (200,500,1000) and n>50 and g['samples']==round(n*rate/10),
            'Invalid original support or sample grid')
    return dict(g, native_start_sample=g['start_sample'], native_rate_hz=rate,
                native_samples=g['samples'], rate=200, start_sample=start, samples=n*20,
                grid_rounding_s=start/200-g['start_sample']/rate)


def admitted(g, boundary, partitions):
    require(g['dataset'] in boundary['datasets'] and g['split'] in boundary['splits']
            and g['subject'] in partitions[g['dataset']][g['split']], 'Outside registered public split')


def choose_positions(old_names, names, valid, projection, statistics):
    supported=np.flatnonzero(valid)
    if not len(supported):
        return supported
    preferred=[names.index(k) for k in old_names if k in names]
    if len(preferred)==min(3,len(supported)) and all(valid[k] for k in preferred):
        return np.asarray(preferred)
    return supported[so.select_proxy_rows(statistics,projection[supported],3)]


def check_synthetic():
    g=output_grid(dict(start_sample=38,samples=30000,rate=500,duration_s=60))
    assert (g['start_sample'],g['samples'])==(15,12000)
    assert abs(g['grid_rounding_s'])<=.0025
    stats={'covariance':np.eye(4)}
    idx=choose_positions(['b','c','d'],list('abcd'),np.array([1,1,1,1],bool),np.eye(4),stats)
    np.testing.assert_array_equal(idx,[1,2,3])
    idx=choose_positions(['b','c','d'],list('abcd'),np.array([1,0,1,1],bool),np.eye(4),stats)
    np.testing.assert_array_equal(idx,[0,2,3])


def setup(cfg, root, out):
    global CFG, ROOT, OUT, PARENT, FIT, RECORDS, PARTITIONS, COORD, CORE, GROUPS
    CFG,ROOT,OUT=cfg,root,out
    PARENT=root/cfg['source_cache']; FIT=root/cfg['source_ssm']
    old=read_json(PARENT/'plan.json')
    require(old['schema']=='physiology_so_supervision_v2' and Path(old['source'])==FIT
            and read_json(PARENT/'verification.json')['status']=='passed'
            and read_json(FIT/'verification.json')['status']=='passed'
            and read_json(root/cfg['accepted_pilot']/'verification.json')['status']=='passed',
            'Completed, verified production sources and accepted pilot required')
    down=root/cfg['downstream_input']
    view=DownstreamInputDataset(down)
    require(view.schema=='physiology_downstream_input_v2' and read_json(down/'verification.json')['status']=='passed',
            'Expected verified downstream v2')
    PARTITIONS=view.plan['subject_partitions']
    RECORDS={}
    for e in view.plan['records']:
        admitted(e,cfg['boundary'],PARTITIONS)
        rec=view.records[e['join_key']]
        require(rec.npz_path.resolve().is_relative_to(down.resolve()/'records'), 'Downstream record outside source')
        key=e['split'],e['native_group']
        if key in RECORDS:
            require((rec.npz_path/'eeg.npy').samefile(RECORDS[key].npz_path/'eeg.npy'), 'Different EEG probe views')
        RECORDS[key]=rec
    COORD=read_json(FIT/'coordinates.json')['individual']
    CORE=read_json(FIT/'resolved_model_config.json')
    groups=[output_grid(g) for g in old['groups']]
    for g in groups:
        admitted(g,cfg['boundary'],PARTITIONS)
        require((g['split'],group_name(g)) in RECORDS and COORD[g['dataset']+'__'+g['subject']]['split']==g['split'],
                'Measurement and subject calibration split mismatch')
    GROUPS={g['key']:g for g in groups}
    require(len(GROUPS)==len(groups)==read_json(PARENT/'verification.json')['windows'],'Nonunique parent windows')
    refs=[g['key'] for g in old['reference_groups']]
    require(len(set(refs))==len(refs) and all(GROUPS[k]['split']=='train' for k in refs),'Reference split mismatch')
    counts={ds:dict(v, uncompressed_EEG_bytes=sum(g['samples']*len(g['EEG_channels'])*24 for g in groups if g['dataset']==ds))
            for ds,v in old['counts'].items()}
    return dict(schema=cfg['schema'],source=str(FIT),source_cache=str(PARENT),groups=groups,
                reference_keys=refs,counts=counts,source_pair_windows=old['source_pair_windows'],
                source_subjects=old['source_subjects'],source_records=old['source_records'])


def group_name(g):
    return '|'.join(g[k] for k in ('dataset','subject','native_record'))


def voltage(g):
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    admitted(g,CFG['boundary'],PARTITIONS)
    record=RECORDS[g['split'],group_name(g)]
    m=record.manifest['measurement']; state=m['eeg_preprocessing_state']
    dataset_root=(ROOT/REGISTERED_DATASETS[g['dataset']].default_root).resolve()
    require(Path(state['source_path']).resolve().is_relative_to(dataset_root)
            and state['source_path'] in [s['path'] for s in record.manifest['source_files']]
            and m['eeg_channel_names']==g['EEG_channels'] and state['filter_band_hz']==[1.,45.]
            and state['native_sample_rate_hz']==g['native_rate_hz'] and state['canonical_sample_rate_hz']==200
            and state['signal_branch']=='raw_with_ocular_artifact','Voltage identity mismatch')
    a=CleanPhysiologyCacheIndex.load_record_arrays(record,['eeg','eeg_supported_channels'])
    start=g['start_sample']
    x=a['eeg'][start:start+g['samples']]*np.array([u['inverse_factor'] for u in state['unit_conversions']])
    supported=np.asarray(a['eeg_supported_channels'],bool)
    require(x.shape==(g['samples'],len(supported)) and np.isfinite(x).all(),'Incomplete EEG input')
    return x,supported,state


def references(plan):
    # The accepted run already computed exactly these 348 reference moments.
    pilot=ROOT/CFG['accepted_pilot']
    require(read_json(pilot/'plan.json')['train_reference_windows']==len(plan['reference_keys']),
            'Reference population differs from pilot')
    summary={}
    for ds in plan['counts']:
        keys=[k for k in plan['reference_keys'] if GROUPS[k]['dataset']==ds]
        source=read_json(PARENT/'references'/f'{ds}.json')
        require(source['training_only'] and keys==source['source_groups'],'Changed reference identities')
        moments=[]
        for key in keys:
            with np.load(pilot/'moments'/(key+'.npz')) as z:
                require(int(z['sample_rate_hz'])==200,'Wrong training reference rate')
                moments.append(dict(z))
        # Independently reconstruct the reference, and replay one current input per dataset.
        x,support,_=voltage(GROUPS[keys[0]])
        require(support.all(),'Unsupported training reference')
        for k,v in so.reference_moments(x,200).items():
            np.testing.assert_allclose(v,moments[0][k],rtol=2e-12,atol=1e-15)
        merged=so.combine_reference_moments(moments)
        with np.load(pilot/'references'/f'{ds}.npz') as accepted:
            for k,v in merged.items(): np.testing.assert_allclose(v,accepted[k],rtol=2e-12,atol=1e-15)
        write_npz(OUT/'references'/f'{ds}.npz',**merged)
        summary[ds]=dict(status='completed',training_only=True,windows=len(keys),source_groups=keys,
                         channels=source['channels'])
        write_json(OUT/'references'/f'{ds}.json',summary[ds])
    write_json(OUT/'reference_summary.json',summary)


def old_metadata(g):
    admitted(g,CFG['boundary'],PARTITIONS)
    path=PARENT/'windows'/(g['key']+'.json')
    require(path.resolve().is_relative_to(PARENT/'windows'),'Invalid parent path')
    old=read_json(path)
    require(old['schema']=='physiology_so_supervision_v2' and output_grid(old['source'])==g and
            old['split']==g['split'] and Path(old['array_file'])==path.with_suffix('.npz'),
            'Parent S/O identity changed')
    return old


@lru_cache(maxsize=4)
def stats_for(dataset):
    with np.load(OUT/'references'/f'{dataset}.npz') as z: return dict(z)


def one_fit(g, source, y, co, valid, dest):
    old=read_json(source); spec=old['identity']; p=spec['pair']; w=spec['window']
    require(source==FIT/'tasks'/spec['record_key']/(w['key']+f"__p{p['native_index']:03d}.json") and
            any(m['record_key']==spec['record_key'] and m['window']==w for m in g['members'])
            and all(spec[k]==g[k] for k in ('split','subject','dataset')), 'Parent fit support mismatch')
    key=dest/'tasks'/spec['record_key']/source.name
    if key.exists():
        row=read_json(key)
        require(row['schema']==CFG['schema'] and row['identity']==spec and row['warm_source']==str(source)
                and (not row['qualified'] or key.with_suffix('.npz').exists()),'Resumed fit identity changed')
        if row['qualified']:
            with np.load(key.with_suffix('.npz')) as z:
                np.testing.assert_allclose(z['target'],y,rtol=1e-12,atol=1e-12)
                return key,row,z['prediction'].copy()
        return key,row,None
    row=dict(schema=CFG['schema'],identity=spec,warm_source=str(source),qualified=False,
             status='unsupported_observation',observation_weights=CFG['weights'])
    pred=None
    if valid:
        seed,info=solver.tak_cache_v3_seed(FIT,spec,len(y))
        weights=np.array(CFG['weights'])
        sd=np.array(co['sd'])/[co['eeg_factor'],co['hb_factor'],co['hb_factor']]
        # The individual SD defines the actual objective, common_sd only its auxiliary report.
        pop=POPULATION[g['dataset']]
        common=np.array(pop['sd'])/[pop['eeg_factor'],pop['hb_factor'],pop['hb_factor']]
        attempts=[]
        for initial in ([seed,None] if seed is not None else [None]):
            try:
                result,z=solver.tak_cache_fit_arrays(CORE,y,co,common,structured=True,
                    observation_weights=weights,initial_solution=initial)
            except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                result,z=dict(status='failed_numerical',qualified=False,error=repr(exc)),{}
            attempts.append(dict(route='warm' if initial is not None else 'standard',status=result['status'],
                                 qualified=result['qualified'],fits=result.get('fits',{}),error=result.get('error')))
            if result['qualified']: break
        row.update(result);row.update(attempts=attempts,warm_seed_status=info['status'])
        if row['qualified']:
            pred=z['prediction']
            require(z['fit_mask'][:-1].all() and not z['fit_mask'][-1].any(),'Last-bin scoring changed')
            expected=CORE['tensor']['dt_s']/CORE['solver']['observation_reference_dt_s']*np.sum(weights*((pred-y)[:-1]/sd)**2)
            np.testing.assert_allclose(expected,result['fits']['tak_tau_beta']['solver']['objective_components']['data'],rtol=1e-10,atol=1e-8)
            row['verified_weighted_data_cost']=float(expected)
        if z: write_npz(key.with_suffix('.npz'),**z)
    write_json(key,row)
    return key,row,pred


def produce(key):
    began=time.monotonic();g=GROUPS[key]; end=(OUT/'windows'/key).with_suffix('.json')
    if end.exists():
        row=read_json(end)
        require(row['schema']==CFG['schema'] and row['source']==g and Path(row['array_file']).is_file(),'Terminal window mismatch')
        return brief(row)
    x,supported,state=voltage(g);old=old_metadata(g);n=len(x)//20
    co=COORD[g['dataset']+'__'+g['subject']]
    with np.load(old['array_file']) as z:
        arrays={k:z[k] for k in ('fnirs_x','fnirs_observation_valid','fnirs_reference_unscaled','fnirs_scale',
                                   'fnirs_time_s','source_scoring_mask')}
    np.testing.assert_allclose(arrays['fnirs_scale'],co['hb_factor'],rtol=0,atol=0)
    names=old['fNIRS_positions'];hb=arrays['fnirs_x']/co['hb_factor']
    require(hb.shape==(n,len(names),2) and len(old['source_task_paths'])==len(names),'Hb population mismatch')
    features=np.log(np.maximum(eeg_band_power(x,BANDS,200,10),1e-20))
    features-=features[:50].mean(0)
    hs=np.full_like(hb,np.nan); proxies=np.full((n,len(names)),np.nan)
    projections=[];valid=[];neighbors=[];paths=[];statuses=[];refit_sites=0
    for j,src in enumerate(old['source_task_paths']):
        src=Path(src)
        require(src.is_relative_to(FIT/'tasks'),'Unexpected warm source')
        original=read_json(src);pair=original['identity']['pair']
        indices=pair.get('channel_sets',{}).get('nearest6',[])
        require(pair['id']==names[j], 'Hb regional identity mismatch')
        if pair['status']=='available':
            require(len(indices)==6 and len(set(indices))==6, 'Regional geometry mismatch')
            proj=so.window_projection(dict(eeg_channel_names=g['EEG_channels']),co['pc'],[pair])[0]
            y=np.column_stack((features[:,indices].reshape(n,30)@co['pc'],hb[:,j]))
            ok=arrays['fnirs_observation_valid'][:,j].all() and supported[indices].all()
        else:
            proj=np.zeros(x.shape[1]*5)
            y=np.column_stack((np.full(n,np.nan),hb[:,j]))
            ok=False
        path,row,pred=one_fit(g,src,y,co,ok,OUT)
        refit_sites+=bool(row.get('attempts'))
        if pred is not None: hs[:,j]=pred[:,1:]*co['hb_factor'];proxies[:,j]=pred[:,0]
        valid.append(row['qualified']);statuses.append(row['status']);paths.append(str(path))
        projections.append(proj);neighbors.append(indices)
    valid=np.asarray(valid,bool);projections=np.stack(projections)
    arrays.update(fnirs_s=hs,fnirs_o=arrays['fnirs_x']-hs,
        fnirs_teacher_valid=np.broadcast_to(valid[None,:,None],hs.shape),
        eeg_x=x,eeg_observation_valid=np.broadcast_to(supported,x.shape),eeg_time_s=np.arange(len(x))/200,
        eeg_source_scoring_mask=np.repeat(arrays['source_scoring_mask'],20))
    stats=stats_for(g['dataset'])
    idx=choose_positions(old.get('reference_selected_positions',[]),names,valid,projections,stats)
    direct=np.zeros(len(supported),bool)
    meta=dict(schema=CFG['schema'],key=key,source=g,split=g['split'],dataset=g['dataset'],subject=g['subject'],
        status='failed',EEG_available=False,EEG_channels=g['EEG_channels'],EEG_unit=old.get('EEG_unit',state['native_unit']),
        EEG_native_rate_hz=g['native_rate_hz'],EEG_output_rate_hz=200,fNIRS_rate_hz=10,fNIRS_unit='subject_normalized_Hb',
        fNIRS_positions=names,source_task_paths=paths,source_task_status=statuses,SSM_qualified_pairs=int(valid.sum()),
        SSM_failed_pairs=int((~valid).sum()),source_SSM_refits=refit_sites,normalization_source=str(FIT/'coordinates.json'),
        native_source=state['source_path'],source_failures=[dict(pair=k,status=s) for k,s,q in zip(names,statuses,valid) if not q],
        reference_selected_positions=[names[j] for j in idx],reference_candidates=int(valid.sum()))
    if len(idx) and supported.all():
        try:
            model=so.bind_reference(stats,projections[idx],n)
            proxy=proxies[:,idx]
            target=voltage_power_cap_target(x,model,proxy,cap=CFG['power_cap'],audit_final=True,
                                             frame_batch_size=CFG['resources']['frame_batch_size'])
            np.testing.assert_allclose(x,target['s']+target['o'],rtol=1e-12,atol=1e-10)
            arrays.update(eeg_s=target['s'],eeg_o=target['o'],eeg_teacher_valid=np.ones_like(x,bool),
                eeg_reference_source_indices=idx,eeg_reference_projection=projections[idx],eeg_reference_proxy=proxy,
                eeg_training_voltage_rms=stats['voltage_rms'],local_cap_ratio=target['local_output_ratio'],
                final_cap_ratio=target['final_reanalysis_ratio'])
            for j in idx: direct[neighbors[j]]=True
            meta.update(status='completed',EEG_available=True,reference_condition=float(model['completion_condition']),
                        local_cap_max=float(target['local_output_ratio'].max()),
                        final_cap_max=float(target['final_reanalysis_ratio'].max()),
                        final_cap_exceedance=float(np.mean(target['final_reanalysis_ratio']>1+1e-8)))
        except (np.linalg.LinAlgError,FloatingPointError,OverflowError,ValueError) as exc:
            meta['error']=repr(exc)
    if not meta['EEG_available']:
        meta.setdefault('error','Insufficient qualified regional or full-head voltage support')
        arrays.update(eeg_s=np.full_like(x,np.nan),eeg_o=np.full_like(x,np.nan),eeg_teacher_valid=np.zeros_like(x,bool))
    arrays['eeg_direct_proxy_coverage']=direct
    write_npz(end.with_suffix('.npz'),**arrays)
    meta.update(array_file=str(end.with_suffix('.npz')),seconds=time.monotonic()-began,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(end,meta)
    return brief(meta)


def brief(row):
    return dict(key=row['key'],status=row['status'],qualified_pairs=row['SSM_qualified_pairs'],
                failed_pairs=row['SSM_failed_pairs'],eeg_available=row['EEG_available'],
                seconds=row['seconds'],peak_rss_mib=row['peak_rss_mib'])


def verify(key):
    g=GROUPS[key]
    r=so.verify_worker((str(OUT),g))
    if r['issues']: raise ValueError(str(r))
    meta=read_json(OUT/'windows'/(key+'.json'))
    with np.load(meta['array_file']) as z, np.load(old_metadata(g)['array_file']) as parent:
        x,support,_=voltage(g)
        np.testing.assert_array_equal(x,z['eeg_x'])
        np.testing.assert_array_equal(z['eeg_observation_valid'],np.broadcast_to(support,x.shape))
        np.testing.assert_array_equal(z['eeg_source_scoring_mask'],np.repeat(z['source_scoring_mask'],20))
        for field in ('fnirs_x','fnirs_reference_unscaled','fnirs_scale','fnirs_observation_valid'):
            np.testing.assert_array_equal(z[field],parent[field])
    r['refit_sites']=meta['source_SSM_refits']
    return r


def pool(fn,keys,workers,on_result):
    it=iter(keys)
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('fork')) as executor:
        pending={}
        def submit():
            key=next(it,None)
            if key is not None: pending[executor.submit(fn,key)]=key
        for _ in range(min(len(keys),workers)): submit()
        while pending:
            done,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for f in done:
                key=pending.pop(f)
                try: on_result(f.result())
                except Exception as exc: raise RuntimeError(f'{fn.__name__} failed on {key}') from exc
                submit()


def run(cfg,args):
    from experiments.scripts.run_dual_component_tokenizer import finalize_long_so_cache
    global POPULATION
    root=args.project_root.resolve();out=(root/cfg['output']).resolve()
    require(args.output is None or args.output.resolve()==out,'Reviewed v3 output required')
    check_synthetic()
    if args.stage=='check':
        print('Unified S/O v3 contract and synthetic grid/selection checks passed');return
    require(args.stage in ('so-plan','so-produce','so-verify'),'Use so-plan, so-produce or so-verify for v3')
    out.mkdir(parents=True,exist_ok=True)
    with owner_lock(out):
        if (out/'cache_manifest.json').exists() and args.stage!='so-verify':
            raise ValueError('Completed generation is read-only')
        resolved=dict(config=cfg,project_root=str(root))
        if (out/'resolved_config.json').exists(): require(read_json(out/'resolved_config.json')==resolved,'Contract drift')
        else: write_json(out/'resolved_config.json',resolved)
        plan=setup(cfg,root,out)
        POPULATION=read_json(FIT/'coordinates.json')['population']
        if (out/'plan.json').exists(): require(read_json(out/'plan.json')==plan,'Frozen plan changed')
        else: write_json(out/'plan.json',plan)
        if args.stage=='so-plan':return
        require(os.environ.get('SSM_SYSTEMD_UNIT') and os.environ.get('INVOCATION_ID'),'Durable service required')
        workers=args.workers or cfg['resources']['production_workers']
        state=dict(status='running',stage='reference',unit=os.environ['SSM_SYSTEMD_UNIT'],workers=workers,
                   started_unix=time.time(),source=str(Path(__file__).resolve()))
        write_json(out/'execution.json',state)
        try:
            if args.stage=='so-produce':
                references(plan)
                # Use representative training cases as the first valid production outputs.
                pilot=[next(g['key'] for g in plan['groups'] if g['dataset']==ds and g['split']=='train' and g['duration_s']==90)
                       for ds in plan['counts']]
                pilot.append(next(g['key'] for g in plan['groups'] if g['split']=='train' and g['duration_s']<90))
                results=[];begin=time.monotonic()
                state['stage']='pilot';write_json(out/'execution.json',state)
                pool(produce,pilot,min(len(pilot),cfg['resources']['pilot_workers']),results.append)
                for key in pilot: verify(key)
                require(any(r['eeg_available'] for r in results),'No usable pilot voltage')
                write_json(out/'pilot_summary.json',dict(status='passed',rows=results,seconds=time.monotonic()-begin))
                state['stage']='production';write_json(out/'execution.json',state)
                progress=dict(completed_windows=0,qualified_pairs=0,failed_pairs=0,eeg_available=0,
                              planned_windows=len(GROUPS),planned_pairs=plan['source_pair_windows'])
                def update(row):
                    progress['completed_windows']+=1
                    for k in ('qualified_pairs','failed_pairs','eeg_available'):progress[k]+=row[k]
                    progress['updated_unix']=time.time()
                    write_json(out/'progress.json',progress)
                    if progress['completed_windows']%20==0:print(progress,flush=True)
                for r in results:update(r)
                keys=[g['key'] for g in plan['groups'] if g['key'] not in set(pilot)]
                # Spread datasets and support lengths across workers from the start.
                np.random.default_rng(20261121).shuffle(keys)
                pool(produce,keys,workers,update)
            state['stage']='verification';write_json(out/'execution.json',state)
            checks=[];pool(verify,list(GROUPS),cfg['resources']['verification_workers'],checks.append)
            finalize_long_so_cache(out,cfg,plan,checks)
            state.update(status='completed',finished_unix=time.time())
        except BaseException as exc:
            state.update(status='failed',error=repr(exc))
            raise
        finally:write_json(out/'execution.json',state)
