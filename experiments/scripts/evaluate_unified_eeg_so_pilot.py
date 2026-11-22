"""Paired native/unified EEG input refits on eight existing public S/O cases."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
import time
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.scripts import evaluate_shared_driver_reconstruction as recon
import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, sosfiltfilt

from src.data.downstream_input_cache import DownstreamInputDataset
from src.data.dual_component_tokenizer import BANDS, voltage_power_cap_target
from src.data.so_supervision_cache import bind_reference, combine_reference_moments, reference_moments
from src.data.unified_physiology import _bandpass, _resample
from src.inference.observation_baselines import eeg_band_power
from experiments.scripts.run_dual_component_tokenizer import read_json, write_json, write_npz

CFG = yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/unified_eeg_so_pilot_v1.yaml').read_text())
OUT, SO, FIT = (ROOT/CFG[k] for k in ('output_root','so','fit'))
WEIGHTS = np.array([1.,4.,4.])/3
ARMS = CFG['arms']
VARIANTS = CFG['variants']


@lru_cache(maxsize=1)
def model_config():
    return read_json(FIT/'resolved_model_config.json')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parallel(fn, items, workers, stage):
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('fork')) as pool:
        futures={pool.submit(fn,item):item for item in items}
        for n,f in enumerate(as_completed(futures),1):
            f.result()
            if n==1 or n%10==0 or n==len(items):
                print(f'{stage} {n}/{len(items)} {time.monotonic()-started:.1f}s',flush=True)


def plan():
    require(read_json(SO/'verification.json')['status']=='passed' and
            read_json(FIT/'verification.json')['status']=='passed','Verified production parents required')
    old=read_json(ROOT/CFG['panel'])
    full=read_json(SO/'plan.json')
    groups={(g['dataset'],g['subject'],m['record_key'],m['window']['key']):g
            for g in full['groups'] if g['split']=='validation' for m in g['members']}
    cases=[]
    for r in old['rows']:
        if r['split']!='validation': continue
        g=groups[r['dataset'],r['subject'],r['record_plan']['cache_key'],r['window']['key']]
        require(g['duration_s']==90,'Exact complete source window required')
        cases.append(g)
    require(len(cases)==8 and len({(g['dataset'],g['subject']) for g in cases})==8,'Changed panel')
    refs=full['reference_groups']
    require(all(g['split']=='train' for g in refs),'Training reference boundary')
    for ds in {g['dataset'] for g in cases}:
        expected=read_json(SO/'references'/f'{ds}.json')['source_groups']
        require([g['key'] for g in refs if g['dataset']==ds]==expected,'Training identities changed')
    return cases,refs


def downstream():
    require(read_json(ROOT/CFG['downstream']/'verification.json')['status']=='passed','Cache is incomplete')
    ds=DownstreamInputDataset(ROOT/CFG['downstream'],split='validation')
    entries={}
    for e in ds.plan['records']:
        if e['split'] not in ('train','validation'): continue
        require(e['subject'] in ds.plan['subject_partitions'][e['dataset']][e['split']], 'Split mismatch')
        key=e['split'],e['native_group']
        r=ds.records[e['join_key']]
        if key in entries:
            require((r.npz_path/'eeg.npy').samefile(entries[key].npz_path/'eeg.npy'),'EEG probe views differ')
        entries[key]=r
    return ds.index,entries


def unified(g):
    group='|'.join(g[k] for k in ('dataset','subject','native_record'))
    record=ENTRIES[g['split'],group]
    m=record.manifest['measurement']
    s=m['eeg_preprocessing_state']
    require(m['eeg_channel_names']==g['EEG_channels'] and s['filter_band_hz']==[1,45]
            and s['native_sample_rate_hz']==g['rate'] and s['canonical_sample_rate_hz']==200
            and s['signal_branch']=='raw_with_ocular_artifact','EEG contract mismatch')
    arrays=INDEX.load_record_arrays(record,['eeg','eeg_supported_channels'])
    require(arrays['eeg_supported_channels'].all(),'Unsupported EEG input')
    lo=round(g['start_sample']*200/g['rate'])
    inverse=np.array([c['inverse_factor'] for c in s['unit_conversions']])
    x=arrays['eeg'][lo:lo+18000]*inverse
    require(x.shape==(18000,len(inverse)) and np.isfinite(x).all(),'Invalid matched window')
    return x


def reference_one(g):
    require(g['split']=='train','Train only')
    path=OUT/'moments'/(g['key']+'.npz')
    if not path.exists(): write_npz(path,**reference_moments(unified(g),200))


def prepare_case(item):
    i,g=item
    require(g['split']=='validation','Validation only')
    wd=OUT/'inputs'/str(i)
    meta=read_json(SO/'windows'/(g['key']+'.json'))
    require(meta['source']==g and meta['split']=='validation' and meta['status']=='completed','Missing planned prior target')
    co=COORD[g['dataset']+'__'+g['subject']]
    require(co['split']=='validation' and co['status']=='completed','Wrong individual coordinates')
    with np.load(meta['array_file']) as f:
        idx=f['eeg_reference_source_indices']
        projection=f['eeg_reference_projection']
        x=f['eeg_x']
        wd.mkdir(exist_ok=True,parents=True)
        np.save(wd/'native.npy',x)
        np.save(wd/'unified.npy',unified(g))
        # Keep the frozen native S for an exact reproducibility comparison.
        np.save(wd/'cache_s.npy',f['eeg_s'])
        expected=f['eeg_reference_proxy']
        hb=f['fnirs_x'][:,idx]/co['hb_factor']
        require(len(idx)==3 and np.isfinite(hb).all(),'Three retained proxy sites required')
    sources=[]; neighbors=[]
    for j,k in enumerate(idx):
        path=Path(meta['source_task_paths'][k])
        t=read_json(path)
        require(t['qualified'] and t['identity']['split']=='validation' and
                t['identity']['subject']==g['subject'] and t['identity']['dataset']==g['dataset']
                and t['identity']['pair']['id']==meta['fNIRS_positions'][k] and
                any(t['identity']['window']==m['window'] for m in g['members']), 'Source fit identity mismatch')
        with np.load(path.with_suffix('.npz')) as a:
            np.testing.assert_allclose(a['prediction'][:,0],expected[:,j],rtol=0,atol=1e-12)
            np.testing.assert_allclose(a['target'][:,1:],hb[:,j],atol=1e-12)
        neighbors.append(t['identity']['pair']['channel_sets']['nearest6'])
        sources.append(str(path))
    np.save(wd/'projection.npy',projection)
    np.save(wd/'hb.npy',hb)
    write_json(wd/'identity.json',dict(group=g,co=co,sources=sources,neighbors=neighbors,
                                       positions=[meta['fNIRS_positions'][k] for k in idx]))


@lru_cache(maxsize=1)
def donor():
    with np.load(ROOT/CFG['donor']) as f:
        n=f['n_ocular']
        return n[:,np.argmax((n*n).sum(0))]


def native_delta(i,v,x,g,neighbors):
    fs=g['rate']; t=np.arange(len(x))/fs
    u=(t>=30)&(t<50)
    if v.startswith('ocular'):
        amp={'ocular05':.5,'ocular1':1.,'ocular2':2.}[v]
        return recon.cache_ocular_injection(x,fs,g['EEG_channels'],donor(),amp)[0]
    out=np.zeros_like(x)
    if v in ('muscle','neural10','neural35'):
        j=neighbors[0][0]
        rms=np.sqrt(np.mean(x[u,j]**2))
        rng=np.random.default_rng(CFG['seed']+i)
        env=np.sin(np.pi*np.clip((t-30)/20,0,1))**2*u
        if v=='muscle':
            wave=sosfiltfilt(butter(4,[25,70],btype='bandpass',fs=fs,output='sos'),rng.normal(size=len(t)))
        else: wave=np.sin(2*np.pi*int(v[6:])*t+rng.uniform(0,2*np.pi))
        wave=wave*env
        out[:,j]=wave*(rms/np.sqrt(np.mean(wave[u]**2)))
    return out


def filter_delta(delta,fs,start):
    # Align the polyphase grid to the actual record origin. The added pulse
    # is zero for 30 s on the left and 40 s on the right of its active span.
    if not np.any(delta): return np.zeros((18000,delta.shape[1]))
    period=fs//int(np.gcd(fs,200)); phase=start%period
    pad=np.pad(delta,((phase,period),(0,0)))
    result=_resample(_bandpass(pad,fs,(1.,45.)),fs,200)
    begin=round(start*200/fs)-(start-phase)*200//fs
    return result[begin:begin+18000]


def known_inputs(i,arm,v):
    wd=OUT/'inputs'/str(i)
    ident=read_json(wd/'identity.json'); g=ident['group']; co=ident['co']
    old=np.load(wd/'native.npy',mmap_mode='r')
    before=native_delta(i,v,old,g,ident['neighbors'])
    delta=before if arm=='native' else filter_delta(before,g['rate'],g['start_sample'])
    x=np.load(wd/(arm+'.npy'),mmap_mode='r')+delta
    fs=g['rate'] if arm=='native' else 200
    h=np.load(wd/'hb.npy'); dh=np.zeros_like(h)
    if v in ('hbo_noise','hbr_noise'):
        j=int(v=='hbr_noise')
        pulse=np.sin(np.pi*np.arange(200)/200)**2
        dh[300:500,0,j]=pulse/np.sqrt(np.mean(pulse**2))*co['sd'][j+1]/co['hb_factor']
    scale=lambda a,r: np.sqrt(np.mean(a[30*r:50*r]**2))
    transmission=float(scale(delta,fs)/scale(before,g['rate'])) if np.any(before) else None
    return ident,x,fs,delta,h+dh,dh,transmission


def fit_one(item):
    i,arm,v=item; wd=OUT/'fits'/str(i)/arm; path=wd/(v+'.json')
    if path.exists(): return
    ident,x,fs,delta,h,dh,_=known_inputs(i,arm,v)
    co=ident['co']; n=900; eeg=np.log(np.maximum(eeg_band_power(x,BANDS,fs,10),1e-20)); eeg-=eeg[:50].mean(0)
    pred=[]; target=[]; details=[]
    for j,source in enumerate(ident['sources']):
        source=Path(source); fit=read_json(source)['fits']['tak_tau_beta']
        ix=ident['neighbors'][j]
        y=np.column_stack((eeg[:,ix].reshape(n,30)@co['pc'],h[:,j]))
        with np.load(source.with_suffix('.npz')) as a:
            raw=a['target'];old=a['prediction'];sd=a['sd_common']
            seed=dict(driver=a['driver'][None],initial_state=a['initial_state'][None],
                      parameter_values=np.array([fit['tau'],fit['beta']]))
        if arm=='native' and v=='baseline':
            np.testing.assert_allclose(y,raw,atol=1e-9,rtol=1e-9)
            p=old; method='production_replay'
        elif v!='baseline':
            with np.load(wd/'baseline.npz') as base:
                if np.array_equal(base['target'][:,j],y):
                    d=read_json(wd/'baseline.json')['rows'][j]
                    pred.append(base['prediction'][:,j]);target.append(y);details.append(dict(d,method='identical_baseline'));continue
            p=None
        else: p=None
        if p is None:
            result,arr=recon.tak_cache_fit_arrays(model_config(),y,co,sd,structured=True,observation_weights=WEIGHTS,initial_solution=seed)
            method='warm'
            warm_ok=result['qualified']
            if not warm_ok:
                result,arr=recon.tak_cache_fit_arrays(model_config(),y,co,sd,structured=True,observation_weights=WEIGHTS)
                method='standard_after_failed_warm'
            fit=result['fits']['tak_tau_beta']
            p=arr['prediction'] if fit['qualified'] else np.full_like(y,np.nan)
        details.append(dict(qualified=fit['qualified'],tau=fit.get('tau'),beta=fit.get('beta'),method=method,
                            integration=fit.get('integration_max_subject_sd')))
        pred.append(p);target.append(y)
    write_npz(path.with_suffix('.npz'),target=np.stack(target,1),prediction=np.stack(pred,1))
    write_json(path,dict(rows=details,qualified=all(r['qualified'] for r in details)))


def synthetic():
    t=np.arange(90000)/1000
    for f in (10,35):
        x=(np.sin(2*np.pi*f*t)*np.sin(np.pi*np.clip((t-30)/20,0,1))**2)[:,None]
        y=filter_delta(x,1000,2)
        assert y.shape==(18000,1) and np.linalg.norm(y)>0
        np.testing.assert_allclose(filter_delta(2*x,1000,2),2*y)
    y,_,sd,_=recon.tak_task_parameter_synthetic(model_config(),dict(subject=0,repeat=0,task='A',scenario='unchanged_parameters'))
    r,a=recon.tak_cache_fit_arrays(model_config(),y[:100],dict(eeg_factor=1.,hb_factor=1.,sd=sd),sd,structured=True,observation_weights=WEIGHTS)
    require(r['qualified'],'Synthetic Tak fit failed')
    np.testing.assert_allclose(a['target'],a['prediction']+a['residual'],atol=1e-12)


def prepare_all(workers):
    global INDEX,ENTRIES,COORD
    cases,refs=plan()
    INDEX,ENTRIES=downstream()
    COORD=read_json(FIT/'coordinates.json')['individual']
    write_json(OUT/'plan.json',dict(config=CFG,cases=[g['key'] for g in cases],train_reference_windows=len(refs),
                                   workers=workers,planned_target_cells=len(cases)*len(ARMS)*len(VARIANTS)))
    need=[g for g in refs if not (OUT/'references'/(g['dataset']+'.npz')).exists()]
    parallel(reference_one,need,workers,'training references')
    for d in sorted({g['dataset'] for g in need}):
        moments=[]
        for g in refs:
            if g['dataset']==d:
                with np.load(OUT/'moments'/(g['key']+'.npz')) as a: moments.append(dict(a))
        write_npz(OUT/'references'/(d+'.npz'),**combine_reference_moments(moments))
    parallel(prepare_case,[(i,g) for i,g in enumerate(cases) if not (OUT/'inputs'/str(i)/'identity.json').exists()],min(workers,4),'paired inputs')
    return cases



def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def corr(x,y):
    x=np.asarray(x).ravel();y=np.asarray(y).ravel()
    x=x-x.mean();y=y-y.mean()
    den=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/den) if den>0 else np.nan


def change_score(ds,delta):
    den=np.sum(delta*delta)
    require(den>0, 'Nonzero perturbation required')
    do=delta-ds
    return dict(leak=rms(ds)/rms(delta), gain=float(np.sum(do*delta)/den),
                o_corr=corr(do,delta), preservation_error=rms(do)/rms(delta))


def target_one(item):
    i,arm,v=item
    folder=OUT/'targets'/str(i)/arm
    path=folder/(v+'.json')
    if path.exists():return
    start=time.monotonic()
    fits=OUT/'fits'/str(i)/arm
    fit=read_json(fits/(v+'.json'))
    base=read_json(fits/'baseline.json')
    row=dict(i=i,arm=arm,variant=v,qualified=False)
    if not fit['qualified'] or not base['qualified']:
        write_json(path,dict(row,reason='unqualified_roi_or_baseline'));return
    ident,x,fs,de,h,dh,transmission=known_inputs(i,arm,v)
    g=ident['group'];co=ident['co']
    row.update(dataset=g['dataset'],subject=g['subject'],transmission=transmission)
    with np.load(fits/(v+'.npz')) as f: pred=f['prediction']; y=f['target']
    np.testing.assert_allclose(h,y[:,:,1:],atol=1e-13)
    ref=(SO if arm=='native' else OUT)/'references'/(g['dataset']+'.npz')
    with np.load(ref) as f: model=bind_reference(dict(f),np.load(OUT/'inputs'/str(i)/'projection.npy'),900)
    try:
        cap=voltage_power_cap_target(x,model,pred[:,:,0],cap=1.,audit_final=v=='baseline',frame_batch_size=8)
    except (np.linalg.LinAlgError, FloatingPointError) as exc:
        write_json(path,dict(row,reason=f'{type(exc).__name__}: {exc}',seconds=time.monotonic()-start))
        return
    s=cap['s'];o=cap['o']
    ts=np.s_[30*fs:50*fs]; ht=np.s_[300:500]; lags=np.s_[300:850]
    sd=np.asarray(co['sd'])/np.array([co['eeg_factor'],co['hb_factor'],co['hb_factor']])
    hr=pred[:,:,1:];unscaled_error=(pred[:-1]-y[:-1])/sd
    row['qualified']=True
    row['closure_eeg']=float(cap['closure_max_abs'])/max(rms(x),1e-300)
    row['closure_hb']=float(np.max(np.abs(hr+(h-hr)-h)/sd[1:]))
    require(max(row['closure_eeg'],row['closure_hb'])<1e-9,'S/O closure changed')
    row['local_cap_max']=float(cap['local_output_ratio'].max())
    row['tau_mean']=np.mean([r['tau'] for r in fit['rows']]);row['beta_mean']=np.mean([r['beta'] for r in fit['rows']])
    for j,k in enumerate(('proxy','hbo','hbr')):
        row[k+'_nrmse']=rms(unscaled_error[:,:,j])
        row[k+'_corr']=float(np.mean([corr(pred[:-1,r,j],y[:-1,r,j]) for r in range(3)]))
    if v=='baseline':
        folder.mkdir(exist_ok=True,parents=True)
        np.save(folder/'baseline_S.npy',s)
        if arm=='native':
            saved=np.load(OUT/'inputs'/str(i)/'cache_s.npy',mmap_mode='r')
            row['old_S_replay_relative']=rms(s-saved)/rms(saved)
            require(row['old_S_replay_relative']<1e-8,'Cached S baseline replay differs')
        row['eeg_s_x_corr']=float(np.median([corr(s[ts,j],x[ts,j]) for j in range(x.shape[1])]))
        row['eeg_derivative_corr']=float(np.median([corr(np.diff(s[ts,j]),np.diff(x[ts,j])) for j in range(x.shape[1])]))
        row['s_rms_over_x']=rms(s[ts])/rms(x[ts])
        row['o_rms_over_x']=rms(o[ts])/rms(x[ts])
        row['final_cap_exceedance']=float(np.mean(cap['final_reanalysis_ratio']>1+1e-8))
        px=eeg_band_power(x,BANDS,fs,10)[ht].mean(0)
        ps=eeg_band_power(s,BANDS,fs,10)[ht].mean(0)
        for j,(lo,hi) in enumerate(BANDS):
            row[f's_power_over_x_{lo:g}_{hi:g}']=float(np.median(ps[:,j]/np.maximum(px[:,j],1e-30)))
        for j,k in enumerate(('hbo','hbr')):
            row[k+'_s_rms_over_x']=rms(hr[:-1,:,j])/rms(h[:-1,:,j])
        row['hb_raw_pair_corr']=np.mean([corr(h[:-1,j,0],h[:-1,j,1]) for j in range(3)])
        row['hb_s_pair_corr']=np.mean([corr(hr[:-1,j,0],hr[:-1,j,1]) for j in range(3)])
    else:
        s0=np.load(folder/'baseline_S.npy',mmap_mode='r')
        with np.load(fits/'baseline.npz') as f: p0=f['prediction']
        ds=s[ts]-s0[ts]; dp=pred-p0
        row.update(s_pair_corr=corr(s[ts],s0[ts]), s_change_over_base_S=rms(ds)/rms(s0[ts]),
                   eeg_change_train_rms=rms(ds/model['voltage_rms']),
                   hbo_spill_sd=rms(dp[lags,:,1]/sd[1]), hbr_spill_sd=rms(dp[lags,:,2]/sd[2]),
                   proxy_change_sd=rms(dp[ht,:,0]/sd[0]))
        if v.startswith('hb'):
            row.update(change_score(dp[ht,:,1:],dh[ht]))
        else:
            row.update(change_score(ds,de[ts]))
        if v.startswith('ocular'):
            row['ocular_gate']=row['leak']<=.30 and row['o_corr']>=.90
        # Same chosen frontal electrode/ROI in every variant; small display payload.
        display=np.argmax([1 if str(c).upper().startswith(('FP','AFP')) else 0 for c in g['EEG_channels']])
        step=fs//100
        write_npz(folder/(v+'_example.npz'),x=x[::step,display],s=s[::step,display],
                  delta=de[::step,display],base_s=s0[::step,display],hb=h[:,0],hb_s=hr[:,0],base_hb_s=p0[:,0,1:])
    row['seconds']=time.monotonic()-start
    write_json(path,row)


def run_targets(workers):
    cases=read_json(OUT/'plan.json')['cases']
    grid=[(i,a) for i in range(len(cases)) for a in ARMS]
    planned=[(i,a,v) for v in VARIANTS for i,a in grid]
    pending=[j for j in planned if not (OUT/'targets'/str(j[0])/j[1]/(j[2]+'.json')).exists()]
    active={}
    begin=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('fork')) as pool:
        while pending or active:
            for j in list(pending):
                if len(active)>=workers: break
                i,a,v=j
                if not (OUT/'fits'/str(i)/a/(v+'.json')).exists(): continue
                if v!='baseline' and not (OUT/'targets'/str(i)/a/'baseline.json').exists(): continue
                pending.remove(j);active[pool.submit(target_one,j)]=j
            if active:
                done,_=wait(active,timeout=3,return_when=FIRST_COMPLETED)
                for f in done:
                    j=active.pop(f);f.result()
                    print(f'target {len(planned)-len(pending)-len(active)}/{len(planned)} {j} {time.monotonic()-begin:.1f}s',flush=True)
            elif pending:
                import subprocess
                if subprocess.run(['systemctl','--user','is-active','--quiet','unified-eeg-so-fit-20261121']).returncode:
                    raise RuntimeError(f'Missing fits: {pending[:3]}')
                time.sleep(3)


def report():
    cases=read_json(OUT/'plan.json')['cases']
    rows=[]; fits=[]
    for i,case in enumerate(cases):
        for a in ARMS:
            for v in VARIANTS:
                row=read_json(OUT/'targets'/str(i)/a/(v+'.json'))
                row.update(dataset=case.split('/')[0],subject=case.split('/')[1])
                rows.append(row)
                for j,r in enumerate(read_json(OUT/'fits'/str(i)/a/(v+'.json'))['rows']):
                    fits.append(dict(i=i,arm=a,variant=v,roi=j,**r))
    table=pd.DataFrame(rows);ft=pd.DataFrame(fits)
    table.to_csv(OUT/'subject_metrics.csv',index=False);ft.to_csv(OUT/'fit_status.csv',index=False)
    ids=['dataset','i','variant'];numeric=[k for k in table.select_dtypes(include='number') if k not in ('i','seconds')]
    native=table[table.arm=='native'].set_index(ids); unified=table[table.arm=='unified'].set_index(ids)
    good=native.qualified & unified.qualified
    diff=(unified.loc[good,numeric]-native.loc[good,numeric])
    diff.to_csv(OUT/'paired_differences.csv')
    metrics=[]
    for name,data in [('native',native.loc[good,numeric]),('unified',unified.loc[good,numeric]),('unified_minus_native',diff)]:
        for (ds,v),r in data.groupby(level=['dataset','variant']).mean().iterrows():
            metrics.append(dict(arm=name,level='dataset',dataset=ds,variant=v,**r.to_dict()))
        for v,r in data.groupby(level=['dataset','variant']).mean().groupby(level='variant').mean().iterrows():
            metrics.append(dict(arm=name,level='macro',dataset='four_dataset_equal',variant=v,**r.to_dict()))
    pd.DataFrame(metrics).to_csv(OUT/'comparison.csv',index=False)
    gates={}
    for v in ['ocular05','ocular1','ocular2']:
        gates[v]={a:int(table.query('arm==@a and variant==@v').ocular_gate.eq(True).sum()) for a in ARMS}
    summary=dict(planned_targets=len(rows),qualified_targets=int(table.qualified.sum()),
                 planned_roi_conditions=len(fits),qualified_roi_conditions=int(ft.qualified.sum()),
                 roi_routes=ft.method.value_counts().to_dict(),paired_by_variant=good.groupby(level='variant').sum().to_dict(),
                 ocular_gate_of_8=gates,maximum_eeg_closure=table.closure_eeg.max(),
                 maximum_hb_closure=table.closure_hb.max(),maximum_native_replay=table.old_S_replay_relative.max(),
                 design=CFG)
    write_json(OUT/'summary.json',summary)
    return table



def observed_eog():
    """Unlabelled same-window association with registered ocular electrodes, no injection."""
    from src.data.unified_physiology import load_native_eeg_record
    view=DownstreamInputDataset(ROOT/CFG['downstream'],split='validation')
    results=[]
    for i,case in enumerate(read_json(OUT/'plan.json')['cases']):
        ident=read_json(OUT/'inputs'/str(i)/'identity.json');g=ident['group']
        if g['dataset'] not in ('eeg_fnirs_single_trial','simultaneous_eeg_nirs'):continue
        require(g['split']=='validation' and g['subject'] in view.plan['subject_partitions'][g['dataset']]['validation'],'Validation ocular boundary')
        key='|'.join(g[k] for k in ('dataset','subject','native_record'))
        rows=[r for r in view.plan['records'] if r['native_group']==key and r['split']=='validation']
        require(len(rows)==1,'Unique EOG record required')
        record=view.records[rows[0]['join_key']]
        state=record.manifest['measurement']['eeg_preprocessing_state']
        require(state['source_path'] in [f['path'] for f in record.manifest['source_files']],'Registered ocular source')
        raw=load_native_eeg_record(ROOT,record)
        require(raw.auxiliary_channel_names and raw.auxiliary_values is not None and g['rate']==raw.sample_rate_hz==200,'Native external EOG required')
        lo=g['start_sample'];cut=slice(lo,lo+18000)
        np.testing.assert_allclose(raw.values[cut],np.load(OUT/'inputs'/str(i)/'native.npy',mmap_mode='r'),rtol=0,atol=0)
        frontal=[j for j,name in enumerate(g['EEG_channels']) if name.upper().startswith(('FP','AF'))]
        require(frontal,'Anterior scalp support required')
        for arm in ARMS:
            y=(raw.auxiliary_values if arm=='native' else _bandpass(raw.auxiliary_values,200,(1.,45.)))[cut][:-20]
            q,sv,_=np.linalg.svd(y-y.mean(0),full_matrices=False)
            q=q[:,sv>sv[0]*1e-10]
            x=np.load(OUT/'inputs'/str(i)/(arm+'.npy'),mmap_mode='r')[:-20,frontal]
            s=np.load(OUT/'targets'/str(i)/arm/'baseline_S.npy',mmap_mode='r')[:-20,frontal]
            x=x-x.mean(0);s=s-s.mean(0)
            px=q.T@x;ps=q.T@s;po=px-ps
            results.append(dict(dataset=g['dataset'],subject=g['subject'],arm=arm,frontal_sites=len(frontal),
                auxiliary=','.join(raw.auxiliary_channel_names),rank=q.shape[1],
                x_eog_r2=np.sum(px**2)/np.sum(x**2),s_eog_r2=np.sum(ps**2)/np.sum(s**2),
                o_eog_r2=np.sum(po**2)/np.sum((x-s)**2),
                o_eog_signed_gain=np.sum(po*px)/np.sum(px**2),s_eog_relative_norm=np.linalg.norm(ps)/np.linalg.norm(px)))
    pd.DataFrame(results).to_csv(OUT/'natural_eog.csv',index=False)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'],'axes.unicode_minus':False})
    t=pd.read_csv(OUT/'subject_metrics.csv')
    c=pd.read_csv(OUT/'comparison.csv')
    m=c[c.level=='macro'].set_index(['arm','variant'])
    names=['眼电 0.5×','眼电 1×','眼电 2×','肌电样','HbO 扰动','HbR 扰动','10 Hz 探针','35 Hz 探针']
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),layout='constrained')
    for a,offset,color,label in [('native',-.19,'#49748a','原输入'),('unified',.19,'#ad4734','统一预处理')]:
        y=[m.loc[(a,v),'leak'] for v in VARIANTS[1:7]]
        axes[0].bar(np.arange(6)+offset,y,width=.36,color=color,label=label)
        measure=['leak','gain','preservation_error','preservation_error']
        cells=['ocular1','muscle','neural10','neural35']
        axes[1].bar(np.arange(4)+offset,[m.loc[(a,v),k] for v,k in zip(cells,measure)],width=.36,color=color,label=label)
    axes[0].set(xticks=np.arange(6),xticklabels=names[:6],title='S 增量 / 进入分解的扰动（越低越好）',ylabel='范数比')
    axes[1].set(xticks=np.arange(4),xticklabels=['眼电 S 泄漏','肌电样 O 投影','10 Hz 保留误差','35 Hz 保留误差'],title='噪声归属与神经样波形保留',ylabel='比值')
    for ax in axes: ax.legend(frameon=False); ax.tick_params(axis='x',rotation=18)
    fig.savefig(OUT/'comparison.png',dpi=220);plt.close(fig)

    # One ocular example per dataset, chosen by panel order before examining scores.
    fig,axes=plt.subplots(4,2,figsize=(12,9),layout='constrained')
    base=t[(t.variant=='baseline')&(t.arm=='native')].sort_values('i').drop_duplicates('dataset')
    for axis,(_,case) in zip(axes,base.iterrows()):
        for ax,arm in zip(axis,ARMS):
            pp=OUT/'targets'/str(case.i)/arm/'ocular1_example.npz'
            if not pp.is_file():
                ax.set_title(f'{case.subject} · 无合格配对');continue
            with np.load(pp) as z:
                sl=slice(3000,5000);d=z['delta'][sl];ds=z['s'][sl]-z['base_s'][sl]
                scale=rms(d)
                for v,label,style in [(d,'进入分解的眼电','-'),(ds,'S 增量','--'),(d-ds,'O 增量',':')]:
                    ax.plot(np.arange(2000)/100+30,v/scale,style,label=label,lw=.85)
            ax.set_title(f'{case.subject} · '+('原输入' if arm=='native' else '统一处理'))
            ax.set_xlabel('时间（秒）');ax.set_ylabel('本条件眼电 RMS')
        low=min(a.get_ylim()[0] for a in axis);high=max(a.get_ylim()[1] for a in axis)
        for ax in axis: ax.set_ylim(low,high)
    axes[0,0].legend(frameon=False,ncol=3,fontsize=8)
    fig.savefig(OUT/'ocular_examples.png',dpi=200);plt.close(fig)



def verify_and_render():
    """Check the stored fits, paired table, and all declared terminal denominators."""
    from html import escape
    table=pd.read_csv(OUT/'subject_metrics.csv')
    fits=pd.read_csv(OUT/'fit_status.csv')
    before=table[table.arm=='native'].set_index(['i','variant'])
    after=table[table.arm=='unified'].set_index(['i','variant'])
    require(before.index.equals(after.index) and len(fits)==3*len(table),'Terminal identity mismatch')
    replay_error=0.
    for row in table.itertuples():
        co=read_json(OUT/'inputs'/str(row.i)/'identity.json')['co']
        sd=np.array(co['sd'])/np.array([co['eeg_factor'],co['hb_factor'],co['hb_factor']])
        p=OUT/'fits'/str(row.i)/row.arm/(row.variant+'.npz')
        with np.load(p) as z:
            errors=(z['prediction'][:-1]-z['target'][:-1])/sd
            metrics=np.sqrt(np.mean(errors**2,axis=(0,1)))
        if row.qualified:
            expected=[row.proxy_nrmse,row.hbo_nrmse,row.hbr_nrmse]
            np.testing.assert_allclose(metrics,expected,rtol=1e-11,atol=1e-12)
            replay_error=max(replay_error,float(np.max(abs(metrics-expected))))
    paired=before.qualified & after.qualified
    comparison=pd.read_csv(OUT/'comparison.csv')
    for (i,v),r in before[paired].iterrows():
        j=after.loc[i,v]
        if v=='baseline': continue
        with np.load(OUT/'targets'/str(i)/'unified'/(v+'_example.npz')) as z:
            np.testing.assert_allclose(z['x']-z['s']-(z['x']-z['delta']-z['base_s']),
                                       z['delta']-(z['s']-z['base_s']),atol=1e-9)
        for metric in ('leak','gain','o_corr'):
            require(np.isfinite([r[metric],j[metric]]).all(),'Missing paired metric')
    for v in VARIANTS[1:]:
        x=(after.loc[paired,'leak']-before.loc[paired,'leak']).xs(v,level='variant')
        ds=before.loc[paired].xs(v,level='variant').dataset
        expected=x.groupby(ds).mean().mean()
        actual=comparison.query('variant==@v and level=="macro" and arm=="unified_minus_native"').iloc[0].leak
        np.testing.assert_allclose(actual,expected,atol=1e-12)
    write_json(OUT/'verification.json',dict(status='passed',fits_checked=len(fits),
        targets_checked=len(table),valid_pairs=int(paired.sum()),verified_nrmse_max_abs=replay_error,
        baseline_voltage_replay_max=float(table.old_S_replay_relative.max()),
        retained_failures=table.loc[~table.qualified,['dataset','subject','arm','variant','reason']].to_dict('records')))
    summary=read_json(OUT/'summary.json')
    macro=comparison[comparison.level=='macro'].set_index(['arm','variant'])
    natural=pd.read_csv(OUT/'natural_eog.csv')
    dual=lambda v,k:' → '.join(f'{macro.loc[(a,v),k]:.3f}' for a in ARMS)
    n=summary['paired_by_variant']
    label=dict(ocular05='眼电 0.5 倍',ocular1='眼电 1 倍',ocular2='眼电 2 倍',muscle='25–70 Hz 肌电样噪声',
               hbo_noise='HbO 单独扰动',hbr_noise='HbR 单独扰动',neural10='10 Hz 神经样探针',neural35='35 Hz 神经样探针')
    results=pd.DataFrame([dict(条件=label[v],配对窗口=f'{n[v]}/8',S泄漏=dual(v,'leak'),O有符号增益=dual(v,'gain'),
        O增量相关=dual(v,'o_corr'),S保留误差=dual(v,'preservation_error') if v.startswith('neural') else '—') for v in VARIANTS[1:]])
    baseline=pd.DataFrame([dict(指标=title,原输入到统一输入=dual('baseline',k)) for title,k in [
        ('EEG 功率代理 NRMSE','proxy_nrmse'),('HbO NRMSE','hbo_nrmse'),('HbR NRMSE','hbr_nrmse'),
        ('S 与观测电压的波形相关','eeg_s_x_corr'),('一阶差分相关','eeg_derivative_corr'),('S / X 电压 RMS 比','s_rms_over_x'),
        ('S / X 的 8–13 Hz 功率比','s_power_over_x_8_13'),('S / X 的 13–30 Hz 功率比','s_power_over_x_13_30'),
        ('HbO-S / HbR-S 波形相关','hb_s_pair_corr'),('合成后频带和局部窗的超限比例','final_cap_exceedance')]])
    eog=natural.pivot(index=['dataset','subject'],columns='arm',values='o_eog_signed_gain').round(3)
    reasons='；'.join(f'{r.dataset} {r.subject}（{r.arm}，{r.variant}）：{r.reason}' for r in table[~table.qualified].itertuples()) or '全部合格'
    html=f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>统一 EEG 输入前后的 S/O 重拟合</title>
<style>body{{max-width:980px;margin:3em auto;padding:0 1em;font:16px/1.65 system-ui;color:#24313b}}table{{border-collapse:collapse;width:100%;margin:1em 0}}td,th{{border-bottom:1px solid #ccd4d9;padding:.35em;text-align:left}}img{{width:100%}}a{{color:#235e80}}</style>
<h1>统一 EEG 输入前后的 S/O 重拟合</h1>
<p>四个公开数据集各两名原验证被试，各取历史权重实验预定的同一条 90 秒记录窗。每名被试沿用生产缓存的三个代理区域和全头皮 EEG。全量生产缓存未改动。当前检验是全窗可见的条件拟合，只有人工加入的增量有已知值。</p>
<h2>既有证据与本轮问题</h2>
<p>早期权重实验分别测量眼电、肌电样、Hb 平滑干扰和神经样改变；综合噪声误差曾改善，但 Hb 拟合变差，10/35 Hz 波形也进入 O。此前 12 人、24 窗的等权眼电确认中，0.5/1/2 倍有 9/12/19 个计划窗通过双门槛，每档计划 24 窗。冻结预测的对照曾显示功率约束本身已能产生低眼电泄漏。本次采用当前生产的个体主成分和 1:4:4 权重，历史等权数值只作为背景。</p>
<h2>输入与评分</h2>
<p>原分支采用原生采样率与生产训练参考。统一分支采用项目整记录 1–45 Hz 带通、200 Hz EEG，在原 348 个训练记录窗口重新估计电压功率参考。两分支保持 Hb、个体主成分、尺度、三个代理位置、模型方程与权重相同。重新拟合驱动、初态、τ 和 β。此比较同时包含 EEG 带通、采样率和参考统计改变。</p>
<p>原生电压中 30–50 秒加入相同扰动；统一臂将同一增量经相同滤波与时钟转换，不补回被滤掉的幅值。眼电使用实测 VEOG 派生波形及固定额前较强的混合。两种单色团脉冲直接加到已处理 Hb，保持另一色团和 EEG 输入不变。10/35 Hz 只改变 EEG，不能充当天然神经—血流耦合真值。</p>
<h2>无注入基线</h2>{baseline.to_html(index=False,border=0)}
<p>三路 NRMSE 用同一被试冻结 SD 逐路归一，在三个区域和前 899 个 100 ms 点上取 RMS。电压 S 与 X 的相关及频带比取 30–50 秒、先按电极取中位数。S+O=X 是构造恒等式；重建相关、分量保真和独立来源归属分别判断。</p>
<h2>加入同一扰动后的分配</h2>{results.to_html(index=False,border=0)}
<p>箭头为原输入到统一输入。S 泄漏为 S 增量范数 / 进入分解的已知扰动范数。O 有符号增益为 O 增量对扰动的内积 / 扰动平方范数。神经样保留误差为 S 增量与已知改变之差的范数 / 扰动范数。O 的增量相关描述形状。使用同条件合格配对，先库内平均被试再四库等权；门槛通过人数始终以全部八个窗口为分母。</p>
<img src="comparison.png" alt="逐条件分支归属比较"><img src="ocular_examples.png" alt="四库预定首个样例的眼电增量时序">
<h2>自然眼电</h2><p>具备真实 HEOG/VEOG 的四名被试中，以同窗眼电的零延迟线性子空间投影额前 X、S、O。表中是 O 的眼电相关投影占 X 相应投影的有符号增益。未注入观测的相关性不能识别真实的神经和眼电来源。</p>{eog.to_html(border=0)}
<h2>数值核验与范围</h2><p>432 个区域条件全部合格，其中 248 次实际重新优化、24 个原生产基线复用、160 个目标逐点相同的区域条件复用。合格 S/O 目标 {summary['qualified_targets']}/{summary['planned_targets']}。保存的数值失败为 {escape(reasons)}。八个原 S 基线复现相对误差最大为 {summary['maximum_native_replay']:.2g}。受试窗口未替换，局部约束和合成后重新滤波的超限是不同指标。</p>
<p><a href="subject_metrics.csv">逐人逐条件</a> · <a href="fit_status.csv">区域拟合与复用</a> · <a href="natural_eog.csv">自然眼电</a> · <a href="paired_differences.csv">配对差</a> · <a href="verification.json">独立数值核验</a></p></html>"""
    (OUT/'review.html').write_text(html)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['synthetic','fit','targets','report'])
    parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    if args.stage=='synthetic':
        synthetic();write_json(OUT/'synthetic.json',dict(status='passed',measured_reads=0))
    elif args.stage=='fit':
        require(read_json(OUT/'synthetic.json')['status']=='passed','Synthetic test required')
        cases=prepare_all(args.workers)
        grid=[(i,a) for i in range(len(cases)) for a in ARMS]
        parallel(fit_one,[(i,a,'baseline') for i,a in grid],args.workers,'baseline')
        parallel(fit_one,[(i,a,v) for i,a in grid for v in VARIANTS[1:]],args.workers,'perturbation')
    elif args.stage=='targets': run_targets(args.workers)
    else:
        report(); observed_eog(); figures(); verify_and_render()
