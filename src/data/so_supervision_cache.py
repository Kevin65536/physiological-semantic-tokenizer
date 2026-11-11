"""Long-window S/O targets from the completed public SSM cache.

Native voltage is read through the central loader after immutable parent/split/
source checks. No model fitting or protected evaluation is performed here.
"""
from collections import defaultdict, Counter
from functools import lru_cache
from pathlib import Path
import json
import os
import re
import time
import resource
import traceback

import numpy as np

from .clean_physiology_cache import CleanPhysiologyCacheIndex
from .registry import REGISTERED_DATASETS
from .unified_physiology import load_native_eeg_record
from .dual_component_tokenizer import voltage_power_cap_bank, voltage_power_cap_target

SCHEMA='physiology_so_supervision_v1'
SCHEMAS={SCHEMA,'physiology_so_supervision_v2'}


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def save_arrays(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.partial.npz')
    np.savez_compressed(temp,**arrays);temp.replace(path)


@lru_cache(maxsize=1)
def parent_plan(source):
    source=Path(source);verify=json.loads((source/'verification.json').read_text());summary=json.loads((source/'summary.json').read_text())
    if verify['status']!='passed' or verify['issues'] or summary['status']!='completed':raise ValueError('Completed verified SSM parent required')
    plan=json.loads((source/'plan.json').read_text())
    if plan['schema']!='shared_driver_tak_subject_cache_v4':raise ValueError('Exact v4 source required')
    return plan,{r['cache_key']:r for r in plan['records']},json.loads((source/'coordinates.json').read_text())


@lru_cache(maxsize=1)
def measurement_index(path):return CleanPhysiologyCacheIndex(path)


def window_projection(record,pc,positions=None):
    pairs=record['pairs'] if positions is None else positions
    a=np.zeros((len(pairs),len(record['eeg_channel_names'])*5));pc=np.asarray(pc).reshape(6,5)
    for row,p in enumerate(pairs):
        for j,c in enumerate(p['channel_sets']['nearest6']):a[row,c*5:c*5+5]=pc[j]
    return a


def select_proxy_rows(statistics,projection,maximum):
    from scipy.linalg import qr
    a=np.asarray(projection,float)
    if maximum<1 or not len(a):raise ValueError('Positive reference proxy count required')
    root=np.linalg.cholesky(statistics['covariance'])
    _,_,pivot=qr((a@root).T,mode='economic',pivoting=True)
    return np.sort(pivot[:min(maximum,len(a))])


def bind_reference(statistics,projection,steps):
    """Exact conditional completion through SVD, without squaring conditioning."""
    a=np.asarray(projection,float);cov=np.asarray(statistics['covariance']);root=np.linalg.cholesky(cov)
    if a.ndim!=2 or a.shape[1]!=len(cov) or len(a)==0:raise ValueError('Nonempty compatible projection required')
    u,s,vh=np.linalg.svd(a@root,full_matrices=False)
    if s[-1]<=s[0]*max(a.shape)*np.finfo(float).eps:raise ValueError('Rank-deficient joint proxy reference')
    lift=(root@vh.T/s[None,:])@u.T
    np.testing.assert_allclose(a@lift,np.eye(len(a)),rtol=1e-9,atol=1e-9)
    return dict(statistics,projection=a,lift=lift,proxy_steps=np.array(steps),completion_condition=np.array(s[0]/s[-1]))


def reference_moments(x,rate):
    """One 90 s training window, algebraically matching the original reference.

    Add ten consecutive 100ms Gram matrices to obtain overlapping one-second
    second moments. This avoids repeating 90 percent of the native products.
    """
    x=np.asarray(x,float);c=x.shape[1]
    if x.shape!=(90*rate,c):raise ValueError('Full 90s training reference window required')
    bank=voltage_power_cap_bank(x,sample_rate_hz=rate)
    power=(bank[:,:5].reshape(900,rate//10,5,c)**2).mean(1).transpose(0,2,1)
    log=np.log(np.maximum(power,1e-20));baseline=log[:50].mean(0).reshape(-1);ell=(log-log[:50].mean(0))[150:750].reshape(600,-1)
    # Frames centered at 15.0..74.9 s have support 14.5..75.4 s.
    bins=bank[int(14.5*rate):int(75.4*rate)].reshape(609,rate//10,6,c).transpose(0,2,3,1)
    grams=bins@bins.swapaxes(-1,-2);cumulative=np.concatenate([np.zeros_like(grams[:1]),np.cumsum(grams,axis=0)])
    second=(cumulative[10:]-cumulative[:-10])/rate
    diagonal=np.diagonal(second,axis1=-2,axis2=-1)
    if second.shape[0]!=600 or np.any(diagonal<=0):raise ValueError('Degenerate reference power')
    corr=(second/np.sqrt(diagonal[..., :,None]*diagonal[...,None,:])).mean(0)
    return dict(count=np.array(600),sum=ell.sum(0),second=ell.T@ell,baseline=baseline,correlation=corr,
        complement=diagonal[:,-1],voltage_power=np.mean(x*x,0),sample_rate_hz=np.array(rate))


def combine_reference_moments(items,shrinkage=.1):
    count=sum(int(v['count']) for v in items);mu=sum(v['sum'] for v in items)/count
    raw=sum(v['second'] for v in items)/count-np.outer(mu,mu);raw=(raw+raw.T)/2
    floor=max(float(np.diag(raw).max())*1e-8,1e-12)
    cov=(1-shrinkage)*raw+shrinkage*np.diag(np.maximum(np.diag(raw),floor))
    c=items[0]['correlation'].shape[-1];rate=int(items[0]['sample_rate_hz'])
    if any(int(v['sample_rate_hz'])!=rate for v in items):raise ValueError('Reference sample rates differ')
    return dict(mean=mu,covariance=cov,baseline_log_power=np.median([v['baseline'] for v in items],axis=0),
        correlation=(1-shrinkage)*np.mean([v['correlation'] for v in items],axis=0)+shrinkage*np.eye(c)[None],
        complement_power=np.median(np.concatenate([v['complement'] for v in items]),axis=0),
        voltage_rms=np.sqrt(np.mean([v['voltage_power'] for v in items],axis=0)),sample_rate_hz=np.array(rate),
        training_windows=np.array(len(items)),training_feature_rows=np.array(count),shrinkage=np.array(shrinkage))


def make_plan(root,cfg):
    root=Path(root);source=root/cfg['source_ssm'];plan,records,coordinates=parent_plan(str(source.resolve()))
    index=measurement_index(str((root/cfg['measurement']).resolve()));available={r.join_key:r for r in index.records}
    if json.loads((root/cfg['measurement']/'verification.json').read_text())['status']!='passed':raise ValueError('Verified measurement source required')
    if set(available)!={r['join_key'] for r in records.values()}:raise ValueError('Source inventory differs')
    groups={};calibration={};by_dataset=defaultdict(list)
    for r in plan['records']:
        if r['split'] not in cfg['boundary']['splits']:raise ValueError('Source split outside contract')
        if available[r['join_key']].manifest['source_files']!=r['source_identity']:raise ValueError('Measurement source identity changed')
        native=re.sub(r'_Probe[12]$','',r['record']) if r['dataset']=='visual_cognitive_motivation' else r['record']
        for w in r['cache_windows']:
            meta=json.loads((source/'features'/r['cache_key']/(w['key']+'.json')).read_text())
            if meta['status']!='completed' or meta['window']!=w:raise ValueError('Prepared window unavailable')
            rate=int(meta['provenance']['eeg_native_rate_hz']);start=round(w['eeg_start_s']*rate);n=round(w['duration_s']*rate)
            if rate not in (200,500,1000) or not (w['duration_s']==90. or (r['dataset']=='refed' and 5<w['duration_s']<90 and np.isclose(w['duration_s']*10,round(w['duration_s']*10)))):raise ValueError('Unreviewed native support')
            key=f"{r['dataset']}/{r['subject']}/{native}/e{start:010d}_n{n:07d}"
            group=groups.setdefault(key,dict(key=key,dataset=r['dataset'],subject=r['subject'],native_record=native,split=r['split'],rate=rate,start_sample=start,samples=n,duration_s=w['duration_s'],EEG_channels=r['eeg_channel_names'],members=[]))
            if group['EEG_channels']!=r['eeg_channel_names'] or group['split']!=r['split']:raise ValueError('Native grouping changes channel/split identity')
            group['members'].append(dict(record_key=r['cache_key'],window=w))
    jobs=list(groups.values())
    for g in jobs:
        if g['split']=='train' and g['duration_s']==90.:
            key=(g['dataset'],g['subject'],g['native_record'])
            if key not in calibration or g['start_sample']<calibration[key]['start_sample']:calibration[key]=g
        by_dataset[g['dataset']].append(g)
    train=list(calibration.values())
    if {g['dataset'] for g in train}!=set(by_dataset):raise ValueError('Missing dataset training references')
    for ds,gg in by_dataset.items():
        names=gg[0]['EEG_channels']
        if any(g['EEG_channels']!=names for g in gg):raise ValueError('Dataset channel order differs; explicit reference grouping required')
    pair_count=sum(len(records[m['record_key']]['pairs']) for g in jobs for m in g['members'])
    if pair_count!=plan['planned_pair_windows']:raise ValueError('Window grouping lost source fits')
    counts={ds:dict(windows=len(gg),subjects=len({g['subject'] for g in gg}),native_channels=len(gg[0]['EEG_channels']),
        reference_windows=sum(g['dataset']==ds for g in train),short_windows=sum(g['duration_s']<90 for g in gg),
        uncompressed_EEG_bytes=sum(g['samples']*len(g['EEG_channels'])*8*3 for g in gg)) for ds,gg in by_dataset.items()}
    return dict(schema=cfg['schema'],source=str(source.resolve()),measurement=str((root/cfg['measurement']).resolve()),groups=jobs,reference_groups=train,counts=counts,
        source_pair_windows=pair_count,source_records=len(records),source_subjects=len(coordinates['individual']),
        source_subject_partitions=plan['subject_partitions'],source_downstream_joins=plan['downstream_joins'],
        excluded_source_records=[r['cache_key'] for r in records.values() if not r['cache_windows']])


@lru_cache(maxsize=1)
def native_record(root,measurement,join_key):
    index=measurement_index(measurement)
    records=index.records_by_join_key[join_key]
    if len(records)!=1:raise ValueError('Unique native record required')
    return load_native_eeg_record(Path(root),records[0])


def read_native(root,cfg,group):
    root=Path(root);source=(root/cfg['source_ssm']).resolve();plan,records,co=parent_plan(str(source));index=measurement_index(str((root/cfg['measurement']).resolve()))
    member=group['members'][0];r=records[member['record_key']];w=member['window']
    if w not in r['cache_windows'] or r['split']!=group['split'] or group['split'] not in ('train','validation','test'):raise ValueError('Declared public SSM identity required before native read')
    if round(w['eeg_start_s']*group['rate'])!=group['start_sample'] or round(w['duration_s']*group['rate'])!=group['samples']:raise ValueError('Native time support changed before read')
    found=[v for v in index.records_by_join_key[r['join_key']] if v.record_id==r['cache_record_id']]
    if len(found)!=1:raise ValueError('Unique registered measurement record required')
    record=found[0];registered=(root/REGISTERED_DATASETS[r['dataset']].default_root).resolve()
    if record.manifest['source_files']!=r['source_identity'] or any(not (root/f['path']).resolve().is_relative_to(registered) for f in r['source_identity']):raise ValueError('Source boundary changed before raw access')
    native=native_record(str(root.resolve()),str((root/cfg['measurement']).resolve()),record.join_key)
    if int(native.sample_rate_hz)!=group['rate'] or list(native.channel_names)!=group['EEG_channels'] or str(native.source_path.resolve()) not in [str((root/f['path']).resolve()) for f in r['source_identity']]:raise ValueError('Native channel/rate/source mismatch')
    start=group['start_sample'];x=np.asarray(native.values[start:start+group['samples']],float).copy()
    if x.shape!=(group['samples'],len(group['EEG_channels'])) or not np.isfinite(x).all():raise ValueError('Incomplete finite native voltage support')
    return x,str(native.native_unit),str(native.source_path)


def reference_worker(payload):
    root,cfg,out,g=payload;out=Path(out);dest=out/'reference_windows'/g['key'];began=time.monotonic()
    if dest.with_suffix('.json').exists():return json.loads(dest.with_suffix('.json').read_text())
    if g['split']!='train':raise ValueError('Training-only reference before native read')
    try:
        x,unit,source=read_native(root,cfg,g);moments=reference_moments(x,g['rate']);save_arrays(dest.with_suffix('.npz'),**moments)
        row=dict(status='completed',key=g['key'],dataset=g['dataset'],source=source,unit=unit,channels=g['EEG_channels'],seconds=time.monotonic()-began,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    except Exception:row=dict(status='failed',key=g['key'],dataset=g['dataset'],error=traceback.format_exc())
    write_json(dest.with_suffix('.json'),row);return row


def finalize_references(out,plan,cfg):
    out=Path(out);result={}
    for ds in plan['counts']:
        items=[];groups=[g for g in plan['reference_groups'] if g['dataset']==ds];sources=[]
        for g in groups:
            p=out/'reference_windows'/g['key'];r=json.loads(p.with_suffix('.json').read_text())
            if r['status']!='completed':raise ValueError('Failed reference input: '+g['key'])
            with np.load(p.with_suffix('.npz')) as a:items.append(dict(a))
            sources.append(g['key'])
        ref=combine_reference_moments(items,cfg['reference']['shrinkage']);save_arrays(out/'references'/(ds+'.npz'),**ref)
        result[ds]=dict(status='completed',windows=len(items),channels=groups[0]['EEG_channels'],source_groups=sources,training_only=True)
        write_json(out/'references'/(ds+'.json'),result[ds])
    write_json(out/'reference_summary.json',result);return result


def supervision_worker(payload):
    root,cfg,out,g=payload;root=Path(root);out=Path(out);dest=out/'windows'/g['key'];began=time.monotonic()
    if dest.with_suffix('.json').exists():return json.loads(dest.with_suffix('.json').read_text())
    source=root/cfg['source_ssm'];_,records,coordinates=parent_plan(str(source.resolve()));coordinate=coordinates['individual'][g['dataset']+'__'+g['subject']]
    n=round(g['duration_s']*10);columns=[];hx=[];hs=[];ho=[];valid=[];references=[];proxies=[];projection=[];proxy_positions=[];proxy_electrodes=[];sources=[];states=[];failures=[];direct=np.zeros(len(g['EEG_channels']),bool)
    row=dict(schema=cfg['schema'],key=g['key'],dataset=g['dataset'],subject=g['subject'],split=g['split'],status='failed',EEG_available=False,source=g,source_SSM_refits=0)
    arrays={}
    try:
        for member in g['members']:
            r=records[member['record_key']];w=member['window'];feature=source/'features'/r['cache_key']/(w['key']+'.npz')
            with np.load(feature) as a:prepared={k:a[k] for k in ['eeg','hb','hb_reference','eeg_valid','hb_valid']}
            if 'eeg_features' in arrays:np.testing.assert_array_equal(arrays['eeg_features'],prepared['eeg'])
            else:arrays['eeg_features']=prepared['eeg']
            for pair in r['pairs']:
                i=pair.get('compact_index',pair['native_index']);p=source/'tasks'/r['cache_key']/(w['key']+f"__p{pair['native_index']:03d}.json");task=json.loads(p.read_text())
                if task['identity']['pair']['id']!=pair['id'] or task['identity']['window']!=w:raise ValueError('SSM task identity mismatch')
                columns.append(pair['id']);hx.append(prepared['hb'][:,i]*coordinate['hb_factor']);references.append(prepared['hb_reference'][i]);sources.append(str(p));states.append(task['status'])
                if task['qualified']:
                    with np.load(p.with_suffix('.npz')) as a:
                        np.testing.assert_allclose(a['target'][:,1:],prepared['hb'][:,i],rtol=0,atol=1e-12)
                        np.testing.assert_allclose(a['scale'],[coordinate['eeg_factor'],coordinate['hb_factor'],coordinate['hb_factor']],rtol=1e-12)
                        hs.append(a['prediction'][:,1:]*coordinate['hb_factor']);ho.append(a['residual'][:,1:]*coordinate['hb_factor']);proxies.append(a['prediction'][:,0])
                    projection.append(window_projection(r,coordinate['pc'],[pair])[0]);proxy_positions.append(len(columns)-1);proxy_electrodes.append(pair['channel_sets']['nearest6']);direct[pair['channel_sets']['nearest6']]=True;valid.append(True)
                else:
                    hs.append(np.full((n,2),np.nan));ho.append(np.full((n,2),np.nan));valid.append(False);failures.append(dict(pair=pair['id'],status=task['status']))
        arrays.update(fnirs_x=np.stack(hx,1),fnirs_s=np.stack(hs,1),fnirs_o=np.stack(ho,1),fnirs_teacher_valid=np.broadcast_to(np.array(valid)[None,:,None],(n,len(valid),2)).copy(),
            fnirs_observation_valid=np.isfinite(np.stack(hx,1)),fnirs_reference_unscaled=np.stack(references),fnirs_scale=np.array(coordinate['hb_factor']),
            fnirs_time_s=np.arange(n)/10,source_scoring_mask=np.r_[np.ones(n-1,bool),False])
        row.update(fNIRS_unit='subject_normalized_Hb',fNIRS_positions=columns,source_task_paths=sources,source_task_status=states,
            SSM_qualified_pairs=sum(valid),SSM_failed_pairs=len(valid)-sum(valid),source_failures=failures)
        eeg_features=arrays.pop('eeg_features');x,unit,native_source=read_native(root,cfg,g);bank=voltage_power_cap_bank(x,sample_rate_hz=g['rate'])
        computed=np.log(np.maximum((bank[:,:5].reshape(n,g['rate']//10,5,len(g['EEG_channels']))**2).mean(1).transpose(0,2,1),1e-20));computed-=computed[:50].mean(0)
        np.testing.assert_allclose(computed,eeg_features,rtol=1e-9,atol=1e-9);del bank
        arrays.update(eeg_x=x,fnirs_x=np.stack(hx,1),fnirs_s=np.stack(hs,1),fnirs_o=np.stack(ho,1),fnirs_teacher_valid=np.broadcast_to(np.array(valid)[None,:,None],(n,len(valid),2)).copy(),
            fnirs_observation_valid=np.isfinite(np.stack(hx,1)),fnirs_reference_unscaled=np.stack(references),fnirs_scale=np.array(coordinate['hb_factor']),
            eeg_time_s=np.arange(len(x))/g['rate'],fnirs_time_s=np.arange(n)/10,eeg_direct_proxy_coverage=direct,
            source_scoring_mask=np.r_[np.ones(n-1,bool),False],eeg_source_scoring_mask=np.repeat(np.r_[np.ones(n-1,bool),False],g['rate']//10))
        row.update(native_source=native_source,EEG_unit=unit,fNIRS_unit='subject_normalized_Hb',EEG_channels=g['EEG_channels'],fNIRS_positions=columns,
            source_task_paths=sources,source_task_status=states,SSM_qualified_pairs=sum(valid),SSM_failed_pairs=len(valid)-sum(valid),source_failures=failures,
            EEG_native_rate_hz=g['rate'],fNIRS_rate_hz=10,normalization_source=str(source/'coordinates.json'),native_feature_replay_max_abs=float(np.max(abs(computed-eeg_features))))
        if not proxies:raise ValueError('No qualified SSM proxy; no statistical-only fallback')
        with np.load(out/'references'/(g['dataset']+'.npz')) as a:statistics=dict(a)
        all_projection=np.stack(projection);selected=select_proxy_rows(statistics,all_projection,cfg['reference'].get('maximum_proxies',len(projection)))
        model=bind_reference(statistics,all_projection[selected],n);proxy=np.stack(proxies,1)[:,selected]
        direct[:]=False
        for i in selected:direct[proxy_electrodes[i]]=True
        arrays['eeg_direct_proxy_coverage']=direct
        arrays['eeg_reference_source_indices']=np.array(proxy_positions,dtype=np.int64)[selected]
        row.update(reference_candidates=len(projection),reference_selected_positions=[columns[proxy_positions[i]] for i in selected],reference_selection=cfg['reference'].get('selection','all_qualified'))
        target=voltage_power_cap_target(x,model,proxy,cap=cfg['EEG']['cap'],audit_final=True,frame_batch_size=cfg['resources']['frame_batch_size'])
        arrays.update(eeg_s=target['s'],eeg_o=target['o'],eeg_teacher_valid=np.ones_like(x,bool),eeg_observation_valid=np.ones_like(x,bool),
            eeg_reference_projection=model['projection'],eeg_reference_proxy=proxy,eeg_training_voltage_rms=statistics['voltage_rms'],
            local_cap_ratio=target['local_output_ratio'],final_cap_ratio=target['final_reanalysis_ratio'])
        np.testing.assert_allclose(arrays['eeg_x'],arrays['eeg_s']+arrays['eeg_o'],rtol=1e-12,atol=1e-10)
        mask=arrays['fnirs_teacher_valid'];np.testing.assert_allclose(arrays['fnirs_x'][mask],(arrays['fnirs_s']+arrays['fnirs_o'])[mask],rtol=1e-12,atol=1e-12)
        row.update(status='completed',EEG_available=True,reference_condition=float(model['completion_condition']),local_cap_max=float(target['local_output_ratio'].max()),
            final_cap_max=float(target['final_reanalysis_ratio'].max()),final_cap_exceedance=float(np.mean(target['final_reanalysis_ratio']>1+1e-8)),
            reference_log_power_min=float(np.log(target['reference_power']).min()),reference_log_power_max=float(np.log(target['reference_power']).max()),
            EEG_closure_max_abs=float(target['closure_max_abs']))
    except Exception:
        row['error']=traceback.format_exc()
        if 'eeg_x' in arrays:
            arrays.update(eeg_s=np.full_like(arrays['eeg_x'],np.nan),eeg_o=np.full_like(arrays['eeg_x'],np.nan),eeg_teacher_valid=np.zeros_like(arrays['eeg_x'],bool),eeg_observation_valid=np.isfinite(arrays['eeg_x']))
    if arrays:save_arrays(dest.with_suffix('.npz'),**arrays)
    row.update(seconds=time.monotonic()-began,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,array_file=str(dest.with_suffix('.npz').resolve()) if arrays else None)
    write_json(dest.with_suffix('.json'),row);return row


def verify_worker(payload):
    out,g=payload;path=Path(out)/'windows'/g['key'];d=json.loads(path.with_suffix('.json').read_text());errors=[];maxerr=0.
    try:
        if d['schema'] not in SCHEMAS or d['source']!=g:raise ValueError('Window identity mismatch')
        if d['array_file']:
            with np.load(d['array_file']) as a:
                for m in ['eeg','fnirs']:
                    if m+'_x' not in a:continue
                    x,s,o,mask=(a[m+'_'+k] for k in ['x','s','o','teacher_valid'])
                    if x.shape!=s.shape or x.shape!=mask.shape or x.dtype!=np.float64:raise ValueError('Target dtype/shape mismatch')
                    if not np.isfinite(s[mask]).all() or not np.isfinite(o[mask]).all():raise ValueError('Invalid finite target')
                    if np.isfinite(s[~mask]).any() or np.isfinite(o[~mask]).any():raise ValueError('Invalid targets must stay explicitly missing')
                    err=float(np.max(abs(x[mask]-s[mask]-o[mask]))) if mask.any() else 0.;maxerr=max(maxerr,err)
                    np.testing.assert_allclose(x[mask],(s+o)[mask],rtol=1e-12,atol=1e-10)
                if d['EEG_available']:
                    if a['eeg_x'].shape!=(g['samples'],len(g['EEG_channels'])) or a['local_cap_ratio'].max()>1+1e-8:raise ValueError('EEG local cap or native grid mismatch')
                if a['fnirs_x'].shape!=(round(g['duration_s']*10),len(d['fNIRS_positions']),2):raise ValueError('Hb shape mismatch')
                if a['source_scoring_mask'][-1] or not a['source_scoring_mask'][:-1].all():raise ValueError('Source scoring support changed')
                if d.get('normalization_source'):
                    source_proxy={}
                    for j,p in enumerate(d['source_task_paths']):
                        source=json.loads(Path(p).read_text())
                        if source['identity']['pair']['id']!=d['fNIRS_positions'][j]:raise ValueError('Source channel order changed')
                        if source['qualified']:
                            with np.load(Path(p).with_suffix('.npz')) as z:
                                for target,key in [('fnirs_s','prediction'),('fnirs_o','residual'),('fnirs_x','target')]:
                                    np.testing.assert_allclose(a[target][:,j],z[key][:,1:]*a['fnirs_scale'],rtol=1e-12,atol=1e-12)
                                source_proxy[j]=z['prediction'][:,0]
                        elif a['fnirs_teacher_valid'][:,j].any():raise ValueError('Failed SSM target became valid')
                    if d['EEG_available']:
                        indices=a['eeg_reference_source_indices'] if 'eeg_reference_source_indices' in a else list(source_proxy)
                        np.testing.assert_array_equal(a['eeg_reference_proxy'],np.stack([source_proxy[int(j)] for j in indices],1))
    except Exception:errors.append(traceback.format_exc())
    return dict(key=g['key'],dataset=g['dataset'],status=d['status'],EEG_available=d['EEG_available'],source_pairs=len(d.get('source_task_paths',[])),SSM_qualified_pairs=d.get('SSM_qualified_pairs',0),SSM_failed_pairs=d.get('SSM_failed_pairs',0),closure=maxerr,issues=errors)


class SOSupervisionDataset:
    """Explicit long-window cache reader; masks and original coordinates retained."""
    def __init__(self,root,split=None):
        self.root=Path(root);manifest=json.loads((self.root/'cache_manifest.json').read_text())
        if manifest['schema'] not in SCHEMAS or manifest['status']!='completed':raise ValueError('Completed supervision cache required')
        self.windows=[w for w in manifest['windows'] if split is None or w['split']==split]
    def __len__(self):return len(self.windows)
    def __getitem__(self,index):
        row=self.windows[index];meta=json.loads(Path(row['metadata']).read_text())
        arrays={}
        if meta['array_file']:
            with np.load(meta['array_file']) as a:arrays=dict(a)
        return dict(metadata=meta,arrays=arrays)
