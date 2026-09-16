#!/usr/bin/env python3
"""Reproducible, bounded data-scale report using the existing data/SSM owners."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import yaml

CONFIG = ROOT / 'experiments/configs/physiology_semantic_tokenizer/dataset_scaling_report_v1.yaml'
NAMES = {'eeg_fnirs_single_trial': 'Single-Trial', 'simultaneous_eeg_nirs': 'Simultaneous',
         'visual_cognitive_motivation': 'Visual', 'refed': 'REFED'}
ARMS = ['baseline', 'shared_data_only', 'shared_synchronized', 'separate_data_only',
        'separate_synchronized', 'shared_data_noise_only']
ARM_NAMES = ['原坐标', '共同缩放/仅数据', '共同缩放/同步', '分色团缩放/仅数据',
             '分色团缩放/同步', '共同缩放/数据+噪声']


def jsonable(x):
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return jsonable(x.tolist())
    if isinstance(x, np.generic): return jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    return x


def write_json(path, payload):
    Path(path).write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def config():
    c = yaml.safe_load(CONFIG.read_text())
    validate_scope(c)
    return c


def validate_scope(c):
    expected = {'eeg_fnirs_single_trial': ['subject_01', 'subject_09', 'subject_18'],
                'simultaneous_eeg_nirs': ['VP001', 'VP002', 'VP003'],
                'visual_cognitive_motivation': ['S01', 'S02', 'S03'], 'refed': ['1', '2', '3']}
    if c.get('schema') != 'dataset_scaling_report_v1' or set(c['datasets']) != set(expected):
        raise ValueError('invalid report scope')
    for d, subjects in expected.items():
        if c['datasets'][d]['subjects'] != subjects: raise ValueError('subject scope changed')
    if c['datasets']['eeg_fnirs_single_trial']['excluded_original_positions'] != [4, 9]:
        raise ValueError('original trial exclusion changed')
    if c['ssm']['measured_indices'] != [1, 5, 9, 13, 17, 21] or c['ssm']['arms'] != ARMS:
        raise ValueError('SSM diagnostic scope changed')
    if c['ssm']['steps'] != 120 or c['ssm']['measured_subjects'] != expected['eeg_fnirs_single_trial']:
        raise ValueError('SSM identity/shape changed')


def output(c): return ROOT / c['output_root'] / c['run_id']


def init_run(c):
    out = output(c)
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / 'resolved_config.yaml'
    if frozen.exists() and frozen.read_text() != CONFIG.read_text():
        raise ValueError('cannot replace frozen report config')
    frozen.write_text(CONFIG.read_text())
    (out / 'figures').mkdir(exist_ok=True)
    return out


def mad(x, axis=0):
    return 1.482602218505602 * np.nanmedian(abs(x - np.nanmedian(x, axis=axis, keepdims=True)), axis=axis)


def stats_rows(x, names, roles, meta):
    x = np.asarray(x, float)
    rows = []
    for j, name in enumerate(names):
        z = x[:, j]; v = z[np.isfinite(z)]
        rows.append(dict(meta, channel=name, component=roles[j], n_samples=len(z),
            finite_fraction=len(v)/len(z), mean=np.mean(v) if len(v) else np.nan,
            sd=np.std(v) if len(v) else np.nan, robust_sd=mad(v) if len(v) else np.nan,
            max_abs=np.max(abs(v)) if len(v) else np.nan,
            p99_abs=np.quantile(abs(v), .99) if len(v) else np.nan,
            peak_to_peak=np.ptp(v) if len(v) else np.nan,
            rms=np.sqrt(np.mean(v*v)) if len(v) else np.nan))
    return rows


def dry_run(c, out):
    from src.data.unified_physiology import UnifiedPhysiologyWindowDataset
    ds = UnifiedPhysiologyWindowDataset(ROOT/c['cache_root'], window_duration_s=30,
        window_offset_s=-5, eeg_signal_branch=c['current_eeg_branch'])
    records = [r for r in ds._selected_records() if r.dataset_id in c['datasets']
               and r.canonical_subject_id in c['datasets'][r.dataset_id]['subjects']
               and r.base_record_id in c['datasets'][r.dataset_id]['records']
               and r.join_key not in ds.excluded_alignment_records]
    rows = [dict(dataset_id=r.dataset_id, subject=r.canonical_subject_id, record=r.base_record_id,
                 join_key=r.join_key, cache=str(r.npz_path), native_unit=r.manifest['native_contract']['native_unit'])
            for r in records]
    if len(rows) != 28: raise ValueError(f'expected 28 admitted record identities, got {len(rows)}')
    write_json(out/'inventory.json', dict(scope=c['scope'], records=rows, array_reads=0,
        note=c['measured_scope_note'], synthetic_fits=24*2*6, measured_fits=18*2*6))
    print(json.dumps(dict(stage='dry_run', records=len(rows), synthetic_fits=288, measured_fits=216)), flush=True)


def select_windows(c, ds, record):
    events = [w.event for w in ds.windows if w.record.join_key == record.join_key]
    if record.dataset_id == 'eeg_fnirs_single_trial':
        from experiments.evaluate_step5_observation_diagnostic import training_events
        from experiments import evaluate_ssm_overnight_diagnostics as suite
        _, _, base, _, _ = suite.load_config(ROOT/'experiments/configs/physiology_semantic_tokenizer/ssm_overnight_v3.yaml')
        return [(p, dict(e), 0.) for p, e in training_events(events, base)]
    if record.dataset_id == 'refed':
        return [(k, dict(events[0]), float(5 + 30*k)) for k in range(3)]
    selected = []; last_eeg = last_hb = -1e9
    for e in events:
        if e.get('label') == 'unknown': continue
        te, th = e['eeg_time_ms']/1000., e['fnirs_time_ms']/1000.
        if min(te, th) < 5 or te-last_eeg < 30 or th-last_hb < 30: continue
        selected.append((int(e['event_index']), dict(e), 0.)); last_eeg, last_hb = te, th
        if len(selected) == c['max_windows_per_record']: break
    return selected


def collect_measurements(c, out):
    from src.data.unified_physiology import UnifiedPhysiologyWindowDataset, load_native_eeg_record
    from experiments.build_clean_eeg_fnirs_cache import _pair_single_trial_wavelengths
    inventory = json.loads((out/'inventory.json').read_text())['records']
    ds = UnifiedPhysiologyWindowDataset(ROOT/c['cache_root'], window_duration_s=30,
        window_offset_s=-5, eeg_signal_branch=c['current_eeg_branch'])
    index = {r.join_key:r for r in ds.index.records if r.signal_branch != 'absorbance_780_805_830'}
    rows=[]; pairs=[]; windows=[]; traces={}; provenance=[]; scale_data={d:[] for d in NAMES}
    candidate_inputs=[]
    for ix, entry in enumerate(inventory):
        r=index[entry['join_key']]; d=r.dataset_id
        # Exact manifest identity is checked before either raw or canonical array reader.
        if r.canonical_subject_id not in c['datasets'][d]['subjects'] or r.base_record_id not in c['datasets'][d]['records']:
            raise ValueError('record not in frozen scope')
        selected=select_windows(c,ds,r)
        native=load_native_eeg_record(ROOT,r)
        current=ds._load_canonical_record(r)
        with np.load(r.npz_path,allow_pickle=False) as f:
            raw=np.asarray(f['native_input_fnirs'],float)
            raw_names=[str(x) for x in f['native_channel_names']]
        if d=='eeg_fnirs_single_trial':
            raw, pnames=_pair_single_trial_wavelengths(raw,raw_names)
            raw=raw.reshape(len(raw),-1)
            raw_names=[f'{n}_{role}' for n in pnames for role in ('760nm','850nm')]
            raw_roles=['760nm','850nm']*(len(raw_names)//2)
        else: raw_roles=['HbO','HbR']*(len(raw_names)//2)
        hs=current['fnirs_preprocessing_state']; es=current['eeg_preprocessing_state']
        # Invert only the recorded final affine normalization, not filtering or motion correction.
        clean_h=current['fnirs'].astype(float)*np.array(hs['channel_scale'])+np.array(hs['channel_location'])
        clean_e=current['eeg'].astype(float)*np.array(es['channel_scale'])+np.array(es['channel_location'])
        unit_e=native.native_unit
        provenance.append(dict(entry, eeg_unit=unit_e, eeg_source=str(native.source_path),
            eeg_source_sha256=sha(native.source_path), fnirs_cache_sha256=sha(r.npz_path),
            native_fNIRS_sources=r.manifest['source_files'], preprocessing=dict(eeg=es,fnirs=hs),
            raw_fnirs_storage='native_input_fnirs retained float32 source copy',
            current_context='existing whole-record canonical preprocessing'))
        for pos,e,extra in selected:
            te=e['eeg_time_ms']/1000.-5+extra; th=e['fnirs_time_ms']/1000.-5+extra
            def crop(a,hz,start):
                i=round(start*hz);n=round(30*hz)
                if i<0 or i+n>len(a):raise ValueError('window support outside record')
                return a[i:i+n]
            raw_e=crop(native.values,native.sample_rate_hz,te)
            raw_h=crop(raw,r.sample_rate_hz,th)
            cur_e=crop(current['eeg'],200.,te);cur_h=crop(current['fnirs'],10.,th)
            cl_e=crop(clean_e,200.,te);cl_h=crop(clean_h,10.,th)
            wid=f'{r.join_key}|{pos}|{extra}'
            meta=dict(dataset_id=d,subject=r.canonical_subject_id,record=r.base_record_id,window_id=wid)
            windows.append(dict(meta,event_index=int(e['event_index']),original_position=pos,
                eeg_start_s=te,fnirs_start_s=th,duration_s=30,task=e.get('metadata',{}).get('task'),
                reference_kind='pretrial' if d=='eeg_fnirs_single_trial' else 'window_reference_not_verified_rest'))
            for stage,ee,hh in [('raw',raw_e,raw_h),('current',cur_e,cur_h),('clean_measurement',cl_e,cl_h)]:
                ru=raw_roles if stage=='raw' else current['fnirs_component_roles']
                hn=raw_names if stage=='raw' else current['fnirs_channel_names']
                hu=r.manifest['native_contract']['native_unit'] if stage=='raw' else ('robust_SD' if stage=='current' else ('relative_MB​​LL' if d=='eeg_fnirs_single_trial' else r.manifest['native_contract']['native_unit']))
                eu='robust_SD' if stage=='current' else unit_e
                rows+=stats_rows(ee,native.channel_names,['EEG']*ee.shape[1],dict(meta,stage=stage,unit=eu))
                rows+=stats_rows(hh,hn,ru,dict(meta,stage=stage,unit=hu))
            baseline_h=cl_h-cl_h[:50].mean(0)
            if r.canonical_subject_id in c['datasets'][d]['subjects'][:2]:
                scale_data[d].append(float(mad(baseline_h.ravel())))
            candidate_inputs.append((meta,baseline_h.astype(np.float32),list(current['fnirs_channel_names'])))
            for j in range(cl_h.shape[1]//2):
                so=np.std(cl_h[:,2*j]);sr=np.std(cl_h[:,2*j+1]); co=np.std(cur_h[:,2*j]);cr=np.std(cur_h[:,2*j+1])
                pairs.append(dict(meta,pair=j,clean_ratio=so/sr if sr>1e-12 else np.nan,
                    current_ratio=co/cr if cr>1e-12 else np.nan,
                    hbo_scale=hs['channel_scale'][2*j],hbr_scale=hs['channel_scale'][2*j+1],
                    scale_ratio=hs['channel_scale'][2*j]/hs['channel_scale'][2*j+1]))
            if d not in traces:
                traces[d]=dict(meta, eeg_names=list(native.channel_names),fnirs_names=list(current['fnirs_channel_names']),
                    raw_eeg=raw_e[:round(3*native.sample_rate_hz),:3], raw_eeg_hz=native.sample_rate_hz,
                    raw_h=raw_h[:, :2],raw_h_hz=r.sample_rate_hz,raw_h_unit=r.manifest['native_contract']['native_unit'],
                    raw_eeg_unit=unit_e,current_eeg=cur_e[:600,:3],clean_eeg=cl_e[:600,:3],
                    current_h=cur_h[:,:2],clean_h=cl_h[:,:2],candidate_h=baseline_h[:,:2],
                    candidate_reference='first 5 s window reference; not measured clean truth')
        print(json.dumps(dict(stage='measurements',record=ix+1,total=len(inventory),identity=r.join_key,windows=len(selected))),flush=True)
        ds._record_cache.clear()
    scales={d:float(np.median(v)) for d,v in scale_data.items()}
    if not all(np.isfinite(v) and v>1e-12 for v in scales.values()):raise ValueError('invalid candidate training scale')
    for meta,h,names in candidate_inputs:
        d=meta['dataset_id'];rows+=stats_rows(h/scales[d],names,['HbO','HbR']*(h.shape[1]//2),dict(meta,stage='candidate_shared',unit='training_group_shared_SD'))
    for d in traces:traces[d]['candidate_h']=traces[d]['candidate_h']/scales[d]
    pd.DataFrame(rows).to_csv(out/'channel_window_statistics.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'hbo_hbr_ratios.csv',index=False)
    write_json(out/'measurement_windows.json',windows);write_json(out/'measurement_provenance.json',provenance)
    write_json(out/'candidate_scales.json',dict(scales=scales,fit_subjects={d:c['datasets'][d]['subjects'][:2] for d in NAMES},interpretation='descriptive processing candidate, not clean truth or trained tokenizer'))
    write_json(out/'waveform_examples.json',traces)
    return dict(records=len(inventory),windows=len(windows),statistics_rows=len(rows),pairs=len(pairs))


def arm_transform(arm,separate,shared):
    if arm=='baseline':return np.ones(3),'none'
    d=np.array([1.,shared,shared]) if arm.startswith('shared') else np.array([1.,*separate])
    owner='all' if arm.endswith('synchronized') else ('noise' if arm.endswith('noise_only') else 'none')
    return d,owner


def fit_case(task):
    from src.inference import t3a_balloon_joint_ssm as joint
    from src.inference import t3a_balloon_robust_ssm as core
    from experiments.evaluate_step5a_inference_consistency import model
    arm=task['arm']; d,owner=arm_transform(arm,task['separate'],task['shared'])
    y=task['y'].copy();n=len(y);mask=np.zeros_like(y,dtype=bool)
    if task['mode']=='center_fNIRS':mask[(n-16)//2:(n+16)//2,1:]=True;y[mask]=np.nan
    p,c=model(task['base'],'W',0.)
    if task.get('noise') is not None:
        p=replace(p,fixed=replace(p.fixed,observation_scale=tuple(task['noise'])))
    spec=core.BalloonObservationSpec().resolved(p.fixed)
    if owner=='all':spec=spec.reexpress(d)
    elif owner=='noise':spec=replace(spec,observation_scale=tuple(np.array(spec.observation_scale)*d))
    start=time.monotonic(); row={k:task[k] for k in ['kind','identity','subject','arm','mode']}
    row.update(scale=d.tolist(),status='failed',physical_pass=False)
    try:
        fit=joint.smooth_balloon_joint(y*d,p,config=c,observation_spec=spec,quadrature_order=task['quadrature'])
        pred=fit.trajectory_mean/d
        state=fit.state_mean
        physical=core.run_physical_checks(state,p)
        passed=all(bool(physical[k]) for k in ['finite','positive_fvpq','oxygen_extraction_in_unit_interval','absolute_hb_nonnegative','hbr_not_above_hbt'])
        row.update(status='completed',physical_pass=passed,physical_checks=physical,
                   state=state, prediction=pred, log_likelihood=fit.parameter_log_likelihood+spec.log_abs_det(np.isfinite(y)),
                   state_variance=np.diagonal(fit.state_covariance,axis1=1,axis2=2))
        if not passed:row['status']='physical_failure'
        for j,name in enumerate(['EEG','HbO','HbR']):
            use=mask[:,j] if task['mode']=='center_fNIRS' and j>0 else np.ones(n,bool)
            error=pred[use,j]-task['target'][use,j]
            row[name+'_rmse']=float(np.sqrt(np.mean(error**2)))
            row[name+'_nrmse']=row[name+'_rmse']/float(task['sd'][j])
            row[name+'_bias']=float(np.mean(error))
        if task.get('truth') is not None:
            row['r_nrmse']=float(np.sqrt(np.mean((state[:,0]-task['truth'][:,0])**2))/np.std(task['truth'][:,0]))
    except Exception as e:row.update(error=f'{type(e).__name__}: {e}')
    row['seconds']=time.monotonic()-start
    return row


def run_fits(c,out,tasks,stage):
    rows=[]; examples={}; by_id={}
    started=time.monotonic()
    # Each worker runs one existing SSM fit; no parameter search or solver retries.
    with ProcessPoolExecutor(max_workers=c['ssm']['workers']) as pool:
        futures=[pool.submit(fit_case,t) for t in tasks]
        for f in as_completed(futures):
            r=f.result();key=(r['identity'],r['mode'])
            by_id.setdefault(key,{})[r['arm']]={k:r.get(k) for k in ['state','prediction','log_likelihood','status']}
            if r['identity']==tasks[0]['identity']:
                examples[f"{r['mode']}|{r['arm']}"]={k:r.get(k) for k in ['state','prediction','status']}
            r={k:v for k,v in r.items() if k not in ['state','prediction','state_variance']};rows.append(r)
            if len(rows)%24==0:print(json.dumps(dict(stage=stage,completed=len(rows),expected=len(tasks),elapsed=round(time.monotonic()-started,1))),flush=True)
            if time.monotonic()-started>c['ssm']['maximum_seconds']:raise TimeoutError('fixed stage wall budget exhausted')
    invariance=[]
    for (identity,mode),arms in by_id.items():
        base=arms['baseline']
        for arm in ['shared_synchronized','separate_synchronized']:
            b=arms[arm]
            inv=dict(identity=identity,mode=mode,arm=arm,baseline_status=base['status'],changed_status=b['status'])
            if base['state'] is not None and b['state'] is not None:
                inv.update(state_max_abs=float(np.max(abs(base['state']-b['state']))),
                           prediction_max_abs=float(np.max(abs(base['prediction']-b['prediction']))),
                           corrected_ll_error=float(abs(base['log_likelihood']-b['log_likelihood'])))
            invariance.append(inv)
    pd.DataFrame(rows).to_csv(out/f'{stage}_fits.csv',index=False)
    write_json(out/f'{stage}_invariance.json',invariance)
    write_json(out/f'{stage}_example.json',dict(identity=tasks[0]['identity'],input=tasks[0]['y'],target=tasks[0]['target'],truth=tasks[0].get('truth'),fits=examples))
    return dict(expected=len(tasks),completed=sum(r['status']=='completed' for r in rows),seconds=time.monotonic()-started)


def synthetic(c,out):
    from experiments.evaluate_step5a_inference_consistency import load_config,generate_matched
    base=load_config();base['model']['steps']=c['ssm']['steps']
    train=[generate_matched(base,'W',0.,c['ssm']['synthetic_seed_train']+i) for i in range(24)]
    clean=np.concatenate([a['clean'] for a in train]);observed=np.concatenate([a['observations'] for a in train])
    target_scale=float(mad(observed[:,1:].ravel()))
    separate=target_scale/mad(observed[:,1:],axis=0)
    write_json(out/'synthetic_projection.json',dict(separate_scale=separate,target_common_scale=target_scale,
        training_seeds=list(range(c['ssm']['synthetic_seed_train'],c['ssm']['synthetic_seed_train']+24)),base=base,
        source='matched Student-t observation / nonlinear discrete Balloon state model',
        scale_rule='training-only pooled Hb MAD divided by each chromophore MAD; no evaluation fit'))
    tasks=[]
    for i in range(24):
        g=generate_matched(base,'W',0.,c['ssm']['synthetic_seed_evaluate']+i)
        for mode in c['ssm']['modes']:
            for arm in ARMS:tasks.append(dict(kind='synthetic',identity=f'synthetic_{i:02d}',subject='synthetic',base=base,
                arm=arm,mode=mode,y=g['observations'],target=g['clean'],truth=g['states'],sd=np.std(clean,0),
                separate=separate,shared=c['ssm']['shared_multiplier'],quadrature=c['ssm']['quadrature_order']))
    return run_fits(c,out,tasks,'synthetic')


def measured_ssm(c,out):
    from experiments import evaluate_ssm_overnight_diagnostics as suite
    from src.inference.observation_baselines import robust_mad
    cfg,_,base,_,_=suite.load_config(ROOT/'experiments/configs/physiology_semantic_tokenizer/ssm_overnight_v3.yaml')
    run=ROOT/c['ssm']['retained_run'];tasks=[];sources=[]
    for subject in c['ssm']['measured_subjects']:
        path=run/'prepared'/f"{subject}_{c['ssm']['measured_projection']}"
        info=json.loads(path.with_suffix('.json').read_text()); detail=json.loads((run/'prepared'/f'{subject}.json').read_text())
        if info['validation']!=c['ssm']['measured_indices'] or set(info['train'])&set(info['validation']):raise ValueError('fold identity mismatch before array read')
        if any(x['original_ma_trial_position'] in [4,9] for x in detail['trials']):raise ValueError('excluded original trial in retained source')
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
        sources.append(dict(subject=subject,metadata_sha256=sha(path.with_suffix('.json')),arrays_sha256=sha(path.with_suffix('.npz')),train=info['train'],validation=info['validation']))
        target=arrays['target'];hb=target[info['train'],:,1:].reshape(-1,2)
        separate=float(robust_mad(hb.reshape(-1)))/robust_mad(hb)
        for i in info['validation']:
            identity=f'{subject}/{i}'
            view=suite.v3_view(cfg,arrays,info,detail['trials'],i,'full')
            variance=np.sum(view['noise_factor']**2,axis=1).reshape(120,3)
            p,_,_=suite.model(base,suite.BASE)
            noise=np.sqrt(np.mean(variance,axis=0))*np.sqrt((p.fixed.student_nu-2)/p.fixed.student_nu)
            for mode in c['ssm']['modes']:
                # This diagnostic isolates pointwise O0 amplitude semantics; v3 temporal O1/O2
                # and feature-interpolation missingness are retained evidence, not recreated here.
                for arm in ARMS:tasks.append(dict(kind='measured',identity=identity,subject=subject,base=base,
                    arm=arm,mode=mode,y=target[i],target=target[i],sd=arrays['normalizer'],noise=noise,
                    separate=separate,shared=c['ssm']['shared_multiplier'],quadrature=c['ssm']['quadrature_order']))
    write_json(out/'measured_ssm_sources.json',dict(rows=sources,interpretation='pointwise O0 scale sensitivity on saved outer-fold validation inputs; center missing applied to final observations; not a rerun or qualification of v3 feature-missing O1/O2'))
    return run_fits(c,out,tasks,'measured_ssm')


def markdown_table(frame):
    def cell(v):
        if isinstance(v,(float,np.floating)):return f'{v:.5g}' if np.isfinite(v) else '未定义'
        return str(v).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def fit_summary(frame):
    rows=[]
    for mode in ['full','center_fNIRS']:
        for arm,label in zip(ARMS,ARM_NAMES):
            all_rows=frame[(frame['mode']==mode)&(frame.arm==arm)]
            good=all_rows[all_rows.status=='completed']
            r={'模式':mode,'缩放':label,'完成':f'{len(good)}/{len(all_rows)}'}
            for k in ['r_nrmse','HbO_nrmse','HbR_nrmse']:
                if k in frame:r[k]=float(good[k].mean())
            rows.append(r)
    return pd.DataFrame(rows)


def tail_example(c,out,statistics):
    """Reopen one already-audited window, chosen explicitly by largest candidate |x|."""
    cached=out/'tail_example.json'
    if cached.exists():return json.loads(cached.read_text())
    from src.data.unified_physiology import UnifiedPhysiologyWindowDataset
    row=statistics[statistics.stage=='candidate_shared'].sort_values('max_abs',ascending=False).iloc[0]
    window=next(x for x in json.loads((out/'measurement_windows.json').read_text()) if x['window_id']==row.window_id)
    inventory=json.loads((out/'inventory.json').read_text())['records']
    key='|'.join(row.window_id.split('|')[:3])
    if key not in {x['join_key'] for x in inventory}:raise ValueError('tail preview outside frozen inventory')
    ds=UnifiedPhysiologyWindowDataset(ROOT/c['cache_root'],eeg_signal_branch=c['current_eeg_branch'])
    record=next(r for r in ds.index.records if r.join_key==key)
    current=ds._load_canonical_record(record)
    j=list(current['fnirs_channel_names']).index(row.channel);pair=j//2
    cols=[2*pair,2*pair+1];start=round(window['fnirs_start_s']*10)
    cur=current['fnirs'][start:start+300,cols]
    state=current['fnirs_preprocessing_state'];scale=np.array(state['channel_scale'])[cols]
    location=np.array(state['channel_location'])[cols]
    clean=cur*scale+location
    with np.load(record.npz_path,allow_pickle=False) as f:
        i=round(window['fnirs_start_s']*record.sample_rate_hz)
        raw=np.asarray(f['native_input_fnirs'][i:i+round(30*record.sample_rate_hz),cols],float)
    group=json.loads((out/'candidate_scales.json').read_text())['scales'][record.dataset_id]
    result=dict(identity=row.window_id,channel=row.channel,selection='largest candidate absolute amplitude among all audited windows/channels',
                raw=raw,raw_hz=record.sample_rate_hz,clean=clean,current=cur,candidate=(clean-clean[:50].mean(0))/group)
    write_json(cached,result);return jsonable(result)


def append_bitmap_figure(doc, png_path):
    """Place one complete PNG on a figure page, leaving room for its caption."""
    import fitz
    pix = fitz.Pixmap(str(png_path))
    height = min(650., 547. * pix.height / pix.width)
    width = height * pix.width / pix.height
    page = doc.new_page(width=595, height=842)
    left = (595. - width) / 2
    page.insert_image(fitz.Rect(left, 44, left + width, 44 + height), pixmap=pix)
    return page, height


def export_report_document(report, out, captions, *, title, date, compact=False):
    """Export selectable report text and complete bitmap figure pages."""
    import fitz
    import markdown

    (out/'REPORT.md').write_text(report,encoding='utf-8')
    body=markdown.markdown(report,extensions=['tables','fenced_code','toc'])
    # MuPDF's fallback font lacks several Unicode superscripts. HTML superscript
    # retains the exponent and selectable text instead of embedding null glyphs.
    superscripts=str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺ᵀ','0123456789-+T')
    body=re.sub('[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺ᵀ]+',lambda m:'<sup>'+m[0].translate(superscripts)+'</sup>',body)
    css='''body{font-family:sans-serif;color:#20303f;font-size:11pt;line-height:1.6} h1{font-size:26pt;color:#163f59} h2{font-size:18pt;color:#1d526c;margin-top:30px} h3{font-size:13pt;color:#277da8} table{border-collapse:collapse;width:100%;font-size:8.5pt;margin:12px 0} th,td{border:1px solid #c9d5de;padding:5px;vertical-align:top} th{background:#eaf1f5} img{max-width:100%;height:auto} code{font-size:9pt;color:#5a456d} a{color:#236484} p{margin:8px 0}'''
    if compact:css=css.replace('font-size:11pt;line-height:1.6','font-size:10.5pt;line-height:1.5')
    embedded=body
    for name in captions:
        data=base64.b64encode((out/'figures'/f'{name}.png').read_bytes()).decode()
        embedded=embedded.replace(f'src="figures/{name}.png"',f'src="data:image/png;base64,{data}"')
    (out/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+title+'</title><style>'+css+'body{max-width:1180px;margin:40px auto;padding:0 30px}</style><body>'+embedded+'</body></html>',encoding='utf-8')
    # Story can repeat a split table's header background behind unrelated text
    # on later pages. Borders and bold header text remain searchable and clear.
    css=css.replace('th{background:#eaf1f5}', 'th{}')
    # Give every scientific figure a bitmap page instead of letting HTML
    # shrink it to the space remaining at the bottom of a text page.
    doc=fitz.open()
    parts=re.split(r'(<p><img[^>]+></p>)',body)
    skip_caption='';pending_heading=''
    for part_index,part in enumerate(parts):
        if part.startswith('<p><img'):
            name=re.search(r'figures/([^"/]+)\.png',part).group(1)
            title,caption=captions[name]
            page,height=append_bitmap_figure(doc,out/'figures'/f'{name}.png')
            if pending_heading:
                heading=re.sub(r'</?h[23][^>]*>', '', pending_heading)
                spare,_=page.insert_htmlbox(fitz.Rect(42,8,553,41),heading,
                    css='body{font-family:sans-serif;font-size:11pt;color:#277da8}',scale_low=.85)
                if spare<0:raise RuntimeError('figure section heading did not fit')
                pending_heading=''
            caption_html=f'<p><b>图 {int(name[:2])}｜{title}。</b> {caption}</p>'
            spare,scale=page.insert_htmlbox(fitz.Rect(42,60+height,553,800),caption_html,css=css,scale_low=.85)
            if spare<0:raise RuntimeError('figure caption did not fit')
            skip_caption=f'<p><strong>图 {int(name[:2])}'
        else:
            if skip_caption and part.lstrip().startswith(skip_caption):
                part=re.sub(r'^\s*<p>.*?</p>','',part,count=1,flags=re.S);skip_caption=''
            if part_index+1<len(parts) and parts[part_index+1].startswith('<p><img'):
                heading=re.search(r'(<h[23][^>]*>[^<]+</h[23]>)\s*$',part)
                if heading:
                    pending_heading=heading[1];part=part[:heading.start()]
            if not re.sub('<[^>]+>','',part).strip():continue
            buffer=io.BytesIO();writer=fitz.DocumentWriter(buffer)
            story=fitz.Story('<html><body>'+part+'</body></html>',user_css=css,archive=fitz.Archive(str(out)))
            story.write(writer,lambda n,filled:(fitz.Rect(0,0,595,842),fitz.Rect(42,42,553,800),None));writer.close()
            textpdf=fitz.open(stream=buffer.getvalue(),filetype='pdf');doc.insert_pdf(textpdf);textpdf.close()
    page_count=len(doc)
    for i,page in enumerate(doc):
        page.insert_text((42,820),f'EEG-fNIRS | {date} | {i+1} / {page_count}',fontsize=8,color=(.4,.45,.5))
    doc.save(out/'REPORT.pdf',garbage=4,deflate=True,use_objstms=1)
    pdf_text=''.join(p.get_text() for p in doc)
    if '最终结论' not in pdf_text or len(pdf_text)<8000 or '\x00' in pdf_text:
        raise RuntimeError('PDF content missing, truncated or contains unavailable glyphs')
    doc.close()
    fitz.TOOLS.mupdf_warnings(reset=True)
    with fitz.open(out/'REPORT.pdf') as saved:
        for page in saved:page.get_pixmap(matrix=fitz.Matrix(.5,.5))
        report_images=sum(len(page.get_images()) for page in saved)
    with fitz.open(out/'FIGURES.pdf') as saved:
        atlas_images=sum(len(page.get_images()) for page in saved)
    if report_images!=len(captions) or atlas_images!=len(captions):
        raise RuntimeError('PDF bitmap figure count mismatch')
    render_warnings=fitz.TOOLS.mupdf_warnings(reset=True)
    if render_warnings:raise RuntimeError('PDF render warnings: '+render_warnings[:500])
    return dict(pdf_pages=page_count,pdf_text_characters=len(pdf_text),
                figures_embedded_in_html=embedded.count('data:image/png;base64,')==len(captions),
                pdf_image_count=report_images,atlas_image_count=atlas_images,
                pdf_bytes=(out/'REPORT.pdf').stat().st_size,wps_checked=False,
                pdf_all_pages_rendered=True,pdf_render_warnings=0)


def render_report(c,out):
    if any((out/name).exists() for name in ['REPORT.pdf', 'FIGURES.pdf', 'report_completion.json']):
        raise RuntimeError('retained report is immutable; use a new versioned export directory')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import markdown
    import fitz
    for required in ['measurements_completion.json','synthetic_completion.json','measured-ssm_completion.json']:
        if not (out/required).exists():raise RuntimeError(f'missing completed evidence {required}')
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
        'font.size':11,'axes.titlesize':12,'axes.labelsize':10,'legend.fontsize':9,
        # Type 3 preserves the CFF glyph outlines in this Noto TTC; emitting
        # them as Type 42 produced invalid glyph references in PDF readers.
        'axes.unicode_minus':False,'pdf.fonttype':3,'axes.spines.top':False,'axes.spines.right':False})
    stats=pd.read_csv(out/'channel_window_statistics.csv');pairs=pd.read_csv(out/'hbo_hbr_ratios.csv')
    syn=pd.read_csv(out/'synthetic_fits.csv');mea=pd.read_csv(out/'measured_ssm_fits.csv')
    examples=json.loads((out/'waveform_examples.json').read_text())
    windows=pd.DataFrame(json.loads((out/'measurement_windows.json').read_text()))
    counts=windows.groupby('dataset_id').agg(subjects=('subject','nunique'),windows=('window_id','nunique'))
    counts['records']=windows.drop_duplicates(['dataset_id','subject','record']).groupby('dataset_id').size()
    counts=counts[['subjects','records','windows']]
    source=ROOT/c['ssm']['retained_run'];history=json.loads((source/'summary.json').read_text())
    precision=json.loads((source/'precision_audit_v1.json').read_text())['rows']
    projections=[json.loads(f.read_text()) for f in sorted((source/'prepared').glob('subject_*_E0.json'))]
    captions={};atlas=fitz.open()
    def save(fig,name,title,caption):
        fig.suptitle(title,fontsize=16,fontweight='bold',y=.995)
        fig.tight_layout(rect=[0,.015,1,.965]);fig.savefig(out/'figures'/f'{name}.png',dpi=250,bbox_inches='tight')
        fig.savefig(out/'figures'/f'{name}.svg',bbox_inches='tight')
        append_bitmap_figure(atlas,out/'figures'/f'{name}.png');plt.close(fig)
        captions[name]=(title,caption)
    colors=['#277da8','#f4a261','#2a9d8f','#e76f51','#70a494','#9b5de5']
    datasets=list(NAMES)

    # 01: Current branches and candidate are labelled separately.
    fig,axes=plt.subplots(3,1,figsize=(12,6.5))
    lines=[['原生/发布测量\n单位、参考、时钟','缓存可用清理\n光学或色团入口','整 record\n逐通道 MAD','通用训练 loader\nEEG 200 / Hb 10 Hz'],
           ['Single-Trial\n限定原训练 trial','EEG log-power/PCA\nOD→近似 MBLL','训练折共同 Hb 尺度\n参考模型幅度桥','当前 SSM 观测\n3 个量 / 4 Hz'],
           ['有证据的单位\n真实支持/空间对应','保留 float64\n测量与特征边界','同步均值 + 噪声\n明确观测 loading','SSM / tokenizer\n各自派生输入']]
    for ax,line,label in zip(axes,lines,['当前通用入口','当前 SSM 入口','修订候选']):
        ax.set(xlim=(0,4),ylim=(0,1));ax.axis('off');ax.text(-.03,.5,label,ha='right',va='center',fontsize=11)
        for j,txt in enumerate(line):
            ax.text(j+.5,.5,txt,ha='center',va='center',bbox=dict(boxstyle='round,pad=.65',fc='#edf4f8',ec='#9bb6c6'),fontsize=11)
            if j<3:ax.annotate('',xy=(j+1.1,.5),xytext=(j+.9,.5),arrowprops=dict(arrowstyle='->',color='#4d6472'))
    save(fig,'01_pipeline','数据处理有两个实际消费入口','当前 SSM 不经过通用 loader 的逐通道 MAD；底行是修订规范，尚未用于 tokenizer 训练。')

    # 02: raw numeric scale is explicitly not a shared physical coordinate.
    fig,axes=plt.subplots(2,3,figsize=(12,7.5))
    for col,comp in enumerate(['EEG','HbO','HbR']):
        for row,metric in enumerate(['sd','max_abs']):
            ax=axes[row,col];values=[]
            for d in datasets:
                actual=('760nm' if comp=='HbO' else '850nm') if d=='eeg_fnirs_single_trial' and comp!='EEG' else comp
                a=stats[(stats.stage=='raw')&(stats.dataset_id==d)&(stats.component==actual)]
                values.append(a[metric].median() if metric=='sd' else a[metric].max())
            ax.bar(range(4),values,color=['#277da8','#43aa8b','#f9c74f','#e76f51']);ax.set_yscale('log')
            ax.set_xticks(range(4),[NAMES[d] for d in datasets],rotation=20)
            ax.set_title(comp+(' / Single-Trial 为波长' if comp!='EEG' else ''))
            ax.set_ylabel('通道×窗口 SD 中位数' if metric=='sd' else '已审计窗口最大 |x|')
            for j,v in enumerate(values):ax.annotate(f'{v:.3g}',(j,v),ha='center',va='bottom',fontsize=9)
    save(fig,'02_raw_scales','原始数值量级差异：先读单位，再读柱高','纵轴为各自原单位的数值，不能作生理强弱排名。EEG：Single-Trial/Simultaneous/Visual 为读取的 µV；REFED 仅有 loader 的 V 声明，证据未核定。fNIRS：Single-Trial 是光电 V，Simultaneous 是 mmol/L，其余是未知单位的色团导出。')

    fig,axes=plt.subplots(2,3,figsize=(12,7.5))
    for col,comp in enumerate(['EEG','HbO','HbR']):
        for row,metric in enumerate(['sd','max_abs']):
            ax=axes[row,col];data=[]
            for d in datasets:
                a=stats[(stats.stage=='current')&(stats.dataset_id==d)&(stats.component==comp)]
                data.append(a.groupby(['subject','record'])[metric].median().to_numpy())
            ax.boxplot(data,tick_labels=[NAMES[d] for d in datasets],showfliers=True)
            ax.tick_params(axis='x',rotation=20);ax.set_title(comp)
            ax.set_ylabel(('SD' if metric=='sd' else 'max |x|')+'：每记录通道×窗口中位数')
    save(fig,'03_current_scales','当前处理使主体尺度接近，但没有消除尾部和成分差异','每个箱图点来自一条记录的汇总；记录属于三个被试，不是独立总体样本。整体最大值另列统计表，避免记录中位数隐藏极端通道。')

    fig,axes=plt.subplots(4,2,figsize=(12,10))
    for row,d in enumerate(datasets):
        x=examples[d]
        for col,(key,hz) in enumerate([('raw_eeg',x['raw_eeg_hz']),('current_eeg',200.)]):
            a=np.asarray(x[key]);axes[row,col].plot(np.arange(len(a))/hz,a[:,0],lw=.75,color='#277da8')
            axes[row,col].set_title(f"{NAMES[d]} · {x['eeg_names'][0]} · "+('原始' if col==0 else '当前'))
            axes[row,col].set_ylabel(x['raw_eeg_unit']+('（单位未核定）' if d=='refed' else '') if col==0 else 'record/channel robust SD')
            axes[row,col].set_xlabel('窗口起点后 / s')
    save(fig,'04_eeg_waveforms','同一实际窗口的 EEG：原始与当前','固定展示每个数据集首个纳入窗口、第一通道的前 3 s，未按好坏选择。原始 EEG 按实际采样率绘制，未为作图无抗混叠降采样；处理结果是通用 loader 的实际输出。')

    fig,axes=plt.subplots(4,3,figsize=(12,11))
    for row,d in enumerate(datasets):
        x=examples[d]
        for col,key in enumerate(['raw_h','current_h','candidate_h']):
            a=np.array(x[key]);hz=x['raw_h_hz'] if col==0 else 10.
            labels=['760 nm','850 nm'] if col==0 and d=='eeg_fnirs_single_trial' else ['HbO','HbR']
            for j,color in enumerate(['#e76f51','#277da8']):axes[row,col].plot(np.arange(len(a))/hz,a[:,j],color=color,lw=1.2,label=labels[j])
            axes[row,col].set_title(NAMES[d]+' · '+['原始/发布','当前逐通道 MAD','候选共同训练尺度'][col])
            axes[row,col].set_ylabel(x['raw_h_unit'] if col==0 else ['','record/channel robust SD','group shared SD'][col])
            axes[row,col].axvspan(0,5,color='#8796a5',alpha=.12);axes[row,col].set_xlabel('窗口起点后 / s');axes[row,col].legend(loc='best')
    save(fig,'05_fnirs_waveforms','同一实际窗口的 fNIRS：原始、当前、成对尺度候选','各行固定展示首个窗口和首个光学位置。候选只逆当前末级仿射、减去首 5 s 参考并采用前两名被试拟合的共同尺度；不是新运动清理，也不是实测 clean 真值。Single-Trial 原始列与其余列属于不同测量阶段。')

    fig,axes=plt.subplots(1,4,figsize=(12,4))
    ratios=[]
    for ax,d in zip(axes,datasets):
        a=pairs[pairs.dataset_id==d];good=np.isfinite(a.clean_ratio)&np.isfinite(a.current_ratio)&(a.clean_ratio>0)&(a.current_ratio>0);a=a[good]
        ax.scatter(a.clean_ratio,a.current_ratio,s=6,alpha=.15,color='#e76f51');lo=min(a.clean_ratio.min(),a.current_ratio.min());hi=max(a.clean_ratio.max(),a.current_ratio.max())
        ax.plot([lo,hi],[lo,hi],color='#2a9d8f',lw=1.2,label='保持幅度比');ax.axhline(1,color='grey',ls='--')
        ax.set(xscale='log',yscale='log',xlabel='末级缩放前 SD(HbO)/SD(HbR)',ylabel='当前输出的同一比值',title=NAMES[d]);ax.legend(loc='upper left',fontsize=8)
        ratios.append(dict(数据集=NAMES[d],位置窗口数=len(a),缩放前比值中位数=a.clean_ratio.median(),当前比值中位数=a.current_ratio.median(),中位绝对log2畸变=np.median(abs(np.log2(a.current_ratio/a.clean_ratio)))))
    ratio_df=pd.DataFrame(ratios);ratio_df.to_csv(out/'ratio_summary.csv',index=False)
    save(fig,'06_ratio_distortion','逐通道缩放确实改变 HbO/HbR 幅度关系','每点为同一窗口/光学位置；红点偏离对角线就是末级仿射造成的比例改变，非滤波差异。共同正尺度与同时间支持的常量基线保持标准差比，位于绿色对角线。点之间有被试/记录依赖。')

    tail=tail_example(c,out,stats);fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key in zip(axes,['raw','current','candidate']):
        a=np.array(tail[key]);hz=tail['raw_hz'] if key=='raw' else 10
        ax.plot(np.arange(len(a))/hz,a[:,0],color='#e76f51',label='HbO');ax.plot(np.arange(len(a))/hz,a[:,1],color='#277da8',label='HbR')
        ax.set_title({'raw':'原发布色团','current':'当前逐通道 MAD','candidate':'候选共同训练尺度'}[key]);ax.set_xlabel('窗口起点后 / s');ax.legend()
    save(fig,'07_candidate_tail','共同尺度的负结果：极端通道不能靠换一个分母解决','明确按候选最大 |x| 选取的 QC 极端案例：'+tail['identity']+'，'+tail['channel']+'。它不代表典型数据；共同尺度保留幅度差异，也暴露严重尾部。此候选不宜直接进入 tokenizer 训练。')

    # Synthetic ideal is a known truth illustration, not a replacement for measured data.
    from src.data.unified_physiology import _robust_standardize
    t=np.arange(300)/10;response=np.where(t>5,((t-5)/4)**2*np.exp(-np.maximum(t-5,0)/4),0)
    truth=np.column_stack([2*response,-.5*response]);rng=np.random.default_rng(20693500)
    observed=truth+np.column_stack([.025*t,-.01*t])+rng.normal(0,.03,truth.shape)
    wrong,_=_robust_standardize(truth);shared=(truth-truth[:50].mean(0))/.5
    fig,axes=plt.subplots(2,2,figsize=(12,6.5))
    for ax,a,title in zip(axes.ravel(),[observed,truth,wrong,shared],['含漂移/噪声的合成测量','已知潜在 clean 信号（仅合成有真值）','逐色团 MAD：幅度关系被改写','共同已知尺度：保留 4:1 幅度关系']):
        ax.plot(t,a[:,0],color='#e76f51',label='HbO');ax.plot(t,a[:,1],color='#277da8',label='HbR');ax.axvspan(0,5,color='grey',alpha=.1);ax.set_title(title);ax.set_xlabel('时间 / s');ax.legend()
    save(fig,'08_ideal_fnirs','理想输入的含义：保留真实比例与响应，而非强求标准差等于 1','所有曲线为明确构造的合成示意；4:1 与反相关只是这个例子的设定，不是要求真实 HbO/HbR 满足的模板。理想 clean 信号不可直接从实测数据得知。')

    te=np.arange(6000)/200;env=1+.45*np.exp(-((te-12)/4)**2);clean=15*env*np.sin(2*np.pi*10*te)
    raw=clean+4*np.sin(2*np.pi*50*te)+40*np.exp(-((te-7)/.3)**2)+rng.normal(0,1,len(te))
    fig,axes=plt.subplots(3,1,figsize=(12,7))
    axes[0].plot(te,raw,color='#acb4ba',lw=.5,label='合成原始');axes[0].plot(te,clean,color='#277da8',lw=.4,label='已知电位 clean');axes[0].set_ylabel('µV');axes[0].legend();axes[0].set_xlim(5,9)
    axes[1].plot(te,np.log(env**2),color='#9b5de5');axes[1].set_ylabel('log(P/P_ref)');axes[1].set_title('SSM：带单位功率构造后的相对能量特征，另附观测映射')
    axes[2].plot(te[:1000],clean[:1000]/100,color='#2a9d8f',lw=.7);axes[2].set_ylabel('µV / 100');axes[2].set_title('波形 tokenizer：已知物理尺度下的电位；保留波形与通道身份')
    for ax in axes:ax.set_xlabel('时间 / s')
    save(fig,'09_ideal_eeg','SSM 与 tokenizer 应看到不同的、可追溯的观测量','合成示意。宽带/分带 log-power 是 EEG 代理，不等同神经驱动真值；µV/100 是已知固定尺度示例，不声称本项目 tokenizer 已采用该输入或已训练。')

    from experiments.evaluate_ssm_overnight_diagnostics import v3_native_operators
    op=v3_native_operators();a=op['fnirs'];cov=a@a.T;sd=np.sqrt(np.diag(cov));corr=cov/np.outer(sd,sd)
    fig,axes=plt.subplots(1,3,figsize=(12,4));axes[0].imshow(np.eye(120),vmin=-1,vmax=1,cmap='RdBu_r');axes[0].set_title('错误简化：处理后仍独立')
    im=axes[1].imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');axes[1].set_title('实际线性处理引入相关性');fig.colorbar(im,ax=axes[1],fraction=.045)
    singular=np.linalg.svd(a,compute_uv=False);axes[2].semilogy(singular/singular[0]);axes[2].axhline(1e-10,ls='--',color='#e76f51');axes[2].set_title('算子相对奇异值');axes[2].set_xlabel('奇异值序号')
    save(fig,'10_temporal_noise','滤波与 baseline 改变了噪声结构和有效信息维度','使用当前 v3 的 10 Hz→4 Hz、0.01–0.2 Hz 带通与 baseline 算子。此图是已知算子传播独立单位方差噪声的结果，不是把实测残差假定为白噪声的检验。')

    for frame,prefix,label in [(syn,'11_synthetic_metrics','合成：已知状态恢复'),(mea,'13_measured_metrics','实测：固定 18 个验证 trial 的观测预测')]:
        comps=['r_nrmse','HbO_nrmse','HbR_nrmse'] if prefix.startswith('11') else ['HbO_nrmse','HbR_nrmse']
        fig,axes=plt.subplots(2,len(comps),figsize=(12,7),squeeze=False)
        for row,mode in enumerate(['full','center_fNIRS']):
            for col,metric in enumerate(comps):
                ax=axes[row,col];data=[]
                for arm in ARMS:
                    f=frame[(frame['mode']==mode)&(frame.arm==arm)&(frame.status=='completed')]
                    data.append(f[metric].to_numpy())
                box=ax.boxplot(data,patch_artist=True,showfliers=True)
                for b,color in zip(box['boxes'],colors):b.set_facecolor(color);b.set_alpha(.65)
                ax.set_xticks(range(1,7),['基线','共/错','共/同步','分/错','分/同步','共/数据R'],rotation=30)
                ax.set_title(mode+' · '+metric);ax.set_ylabel('NRMSE（冻结训练 SD）')
        save(fig,prefix,label,'每个 trial 在六个分支间配对；完整分母与均值见表。“共/错”“分/错”为仅改数据的负对照；“共/数据R”同时改数据和噪声但不改均值，改变了观测映射。实测没有潜状态真值，不能用较低观测误差证明状态正确。')

    for filename,prefix,known in [('synthetic_example.json','12_synthetic_trace',True),('measured_ssm_example.json','14_measured_trace',False)]:
        x=json.loads((out/filename).read_text());target=np.array(x['target']);obs=np.array(x['input']);tt=np.arange(120)/4
        fig,axes=plt.subplots(2,2,figsize=(12,7))
        for row,j in enumerate([1,2]):
            for col,mode in enumerate(['full','center_fNIRS']):
                ax=axes[row,col];ax.plot(tt,obs[:,j],color='#b7bdc2',lw=.8,label='带噪测量')
                if known:ax.plot(tt,target[:,j],color='black',lw=1.7,label='已知 clean 真值')
                for arm,color in [('baseline','#277da8'),('shared_data_only','#f4a261'),('separate_data_only','#e76f51'),('shared_synchronized','#2a9d8f')]:
                    f=x['fits'][mode+'|'+arm]
                    if f['prediction'] is not None:ax.plot(tt,np.array(f['prediction'])[:,j],color=color,lw=1.25,ls='--' if 'synchronized' in arm else '-',label=ARM_NAMES[ARMS.index(arm)])
                if mode=='center_fNIRS':ax.axvspan(13,17,color='grey',alpha=.18)
                ax.set_title(['HbO','HbR'][row]+' · '+mode);ax.set_xlabel('窗口起点后 / s');ax.set_ylabel('原观测坐标');ax.legend(fontsize=8,ncol=2)
        save(fig,prefix,('合成真值' if known else '实测观测')+'与不同缩放下的 SSM 拟合','预先固定的首个评估身份 '+x['identity']+'，不按结果选择。所有预测均逆变换到相同目标坐标；灰色区域是最终观测层中心缺失。'+('同步变换的曲线与基线重合。' if known else '实测带噪曲线不是真实潜状态。缩小残差仍可能保留错误的形状/极性。'))

    fig,axes=plt.subplots(1,3,figsize=(12,4))
    rows=[r for r in history['temporal'] if r['law']=='nonlinear_gaussian' and r['variant']=='combined' and r['mode']=='full' and r['solver'] in ['O2_pointwise','O2_mean_only','O2']]
    lookup={r['solver']:r for r in rows};keys=['O2_pointwise','O2_mean_only','O2']
    axes[0].bar(range(3),[lookup[k]['r_nrmse'] for k in keys],color=colors[:3]);axes[0].set_xticks(range(3),['逐点','时间均值','均值+相关R'],rotation=20);axes[0].set_title('历史合成 r NRMSE / 24 trial')
    axes[1].bar(range(3),[lookup[k]['clean_HbR_nrmse'] for k in keys],color=colors[:3]);axes[1].set_xticks(range(3),['逐点','时间均值','均值+相关R'],rotation=20);axes[1].set_title('历史合成 clean HbR NRMSE')
    axes[2].bar(['O0','O1','O2'],[54,6,0],color=colors[:3]);axes[2].axhline(72,color='grey',ls='--');axes[2].set_ylim(0,80);axes[2].set_title('历史实测中心预测成功 / 72')
    save(fig,'15_retained_v3','历史 v3：时间合同的合成收益与实测失败并存','读取冻结保留报告与 summary，不重跑旧实验。左两图为同一 Gaussian MAP 的受控消融，右图的各 solver 成功集合不同，不能据此以成功子集误差排序。')

    fig,axes=plt.subplots(1,3,figsize=(12,4.5))
    for subject,color in zip(c['ssm']['measured_subjects'],colors[:3]):
        pp=[x for x in projections if x['subject']==subject]
        axes[0].scatter([x['projection']['fnirs_factor'] for x in pp],[x['normalization_sd'][2] for x in pp],label=subject,color=color)
    axes[0].set(xlabel='训练折共同 Hb factor',ylabel='处理后 HbR 训练 SD',title='幅度桥与弱 HbR');axes[0].legend(fontsize=8)
    eegnoise=[x['feature_noise']['eeg_sd'] for x in projections];axes[1].plot(eegnoise,'o',ms=3);axes[1].axhline(.08*np.sqrt(5/3),ls='--',color='#e76f51',label='合成噪声下限');axes[1].set(xlabel='39 份有效投影',ylabel='EEG 特征噪声 SD',title='最终 EEG 噪声均为下限');axes[1].legend()
    for key,label,color in [('legacy_error_training_sd','原精度路径','#e76f51'),('promoted_baseline_arithmetic_error_training_sd','仅 baseline 提精度','#277da8')]:
        value=[max(x[key]) for x in precision];axes[2].plot(value,'o',ms=3,label=label,color=color)
    axes[2].axhline(1e-6,ls='--',color='grey');axes[2].set(xlabel='48 份折内投影',ylabel='最大重组误差 / 训练 SD',title='仍有重组失败');axes[2].legend(fontsize=8)
    save(fig,'16_precision_and_floors','历史输入诊断：尺度、噪声下限与有限精度','前两图仅含成功保存的 39 份投影，不能外推未准备成功的折。第三图来自 48 份算术审计，原路径 9 份超阈值，仅提升 baseline 精度后仍有 5 份；不修改旧失败。')
    atlas.save(out/'FIGURES.pdf',garbage=4,deflate=True)
    atlas.close()

    # Tables and paired simulation contrasts have a single machine-readable source.
    summary=[]
    for d in datasets:
        for comp in ['EEG','HbO','HbR']:
            rawcomp=('760nm' if comp=='HbO' else '850nm') if d=='eeg_fnirs_single_trial' and comp!='EEG' else comp
            a=stats[(stats.dataset_id==d)&(stats.stage=='raw')&(stats.component==rawcomp)]
            b=stats[(stats.dataset_id==d)&(stats.stage=='current')&(stats.component==comp)]
            unit=str(a.unit.iloc[0]);unit='未核定（loader 写 V）' if d=='refed' and comp=='EEG' else unit
            summary.append(dict(数据集=NAMES[d],原始成分=rawcomp,原始单位=unit,
                原始SD中位数=a.sd.median(),原始最大绝对值=a.max_abs.max(),当前SD中位数=b.sd.median(),当前最大绝对值=b.max_abs.max()))
    sd_table=pd.DataFrame(summary);sd_table.to_csv(out/'amplitude_summary.csv',index=False)
    syntab=fit_summary(syn);meatab=fit_summary(mea)
    syinv=json.loads((out/'synthetic_invariance.json').read_text());meinv=json.loads((out/'measured_ssm_invariance.json').read_text())
    pairs_delta=[];rng=np.random.default_rng(20693510)
    for arm in ARMS[1:]:
        for metric in ['r_nrmse','HbO_nrmse','HbR_nrmse']:
            a=syn[(syn['mode']=='center_fNIRS')&(syn.arm==arm)&(syn.status=='completed')][['identity',metric]]
            b=syn[(syn['mode']=='center_fNIRS')&(syn.arm=='baseline')&(syn.status=='completed')][['identity',metric]]
            m=a.merge(b,on='identity',suffixes=('_a','_b'));diff=m[metric+'_a'].to_numpy()-m[metric+'_b'].to_numpy()
            boot=diff[rng.integers(0,len(diff),size=(2000,len(diff)))].mean(1)
            pairs_delta.append(dict(arm=arm,metric=metric,paired_trials=len(diff),mean_delta=diff.mean(),lower95=np.quantile(boot,.025),upper95=np.quantile(boot,.975)))
    pd.DataFrame(pairs_delta).to_csv(out/'synthetic_paired_differences.csv',index=False)
    write_json(out/'report_summary.json',dict(counts=counts.reset_index().to_dict('records'),amplitude=summary,
        ratios=ratios,synthetic_fits=syntab.to_dict('records'),measured_fits=meatab.to_dict('records'),
        synchronized_max_state_error=dict(synthetic=max(x['state_max_abs'] for x in syinv),measured=max(x['state_max_abs'] for x in meinv)),
        candidate_max_abs=float(stats[stats.stage=='candidate_shared'].max_abs.max()),
        conclusion='coordinate inconsistency harms synthetic recovery; measured fit improvement alone is insufficient; shared group scale candidate retains severe tails'))

    def figure(name):
        title,caption=captions[name]
        return f'\n\n![{title}](figures/{name}.png)\n\n**图 {int(name[:2])}｜{title}。** {caption}\n\n'
    scope_table=counts.reset_index().replace({'dataset_id':NAMES}).rename(columns={'dataset_id':'数据集','subjects':'被试','records':'记录','windows':'30 s 窗口'})
    # Detailed interpretation is generated with numerical tables, never inferred from a hand-picked waveform.
    report=f'''# EEG–fNIRS 数据统一化诊断报告

**2026-09-15 · 实际数据审计、受控实验与修订建议**

## 摘要

本报告覆盖用户要求的五个部分：数据来源/单位/流程，四数据集实测量级比较，原始—当前—期望输入可视化，缩放对 HbO/HbR 拟合的实验，以及逐项修订方案。

本次实际读取 **4 个数据集、每集 3 名被试、28 条记录、194 个 30 s 窗口**；生成 73,824 条通道×窗口×处理阶段统计和 6,870 个 HbO/HbR 位置窗口比较。完成 **288 次合成 SSM 拟合与 216 次实测 SSM 拟合**，各次均保留身份、处理分支、误差及物理检查。这是有界描述性与机制诊断，不是四数据集全量普查，也不支持总体显著性结论。

主要结果：

1. 原始数值量级明显不同，但光电 V、EEG 电位、mmol/L 与未知单位色团导出不能直接按数值比较生理幅度。
2. 当前通用 loader 确实改写了色团幅度关系：Single-Trial 和 Simultaneous 的 HbO/HbR 标准差比中位数从约 **2.06、2.66** 降到 **1.00、1.04**。这条通用路径与 SSM 的原生特征入口不同。
3. 在独立合成真值实验中，仅改数据而不同步模型明显损害 HbO/HbR 恢复；同步变换数据、均值和噪声后，状态最大误差仅 **{max(x['state_max_abs'] for x in syinv):.2g}**。已知正比例缩放本身不是状态失真的必然原因。
4. 实测分色团负对照反而降低 HbR 观测误差；结合其合成状态退化，说明“曲线拟合更好”可能来自错误补偿，不能据此认定生理解释改善。
5. 简单共同训练尺度候选保留了幅度比例，但本次最大值达到 **{stats[stats.stage=='candidate_shared'].max_abs.max():.1f}** 个组尺度单位，不能直接部署。新方案必须结合单位标定、真实支持、质量诊断和噪声模型。

## 1. 数据来源、采集单位与当前预处理

### 1.1 来源与原始测量语义

| 数据集 | 原始来源与采集 | EEG | fNIRS |
| --- | --- | --- | --- |
| Single-Trial | Shin 等，2017，Scientific Data；MI/MA 同步采集 | BrainAmp，30 scalp channels，linked-mastoid 参考；采集 1000 Hz，发布分析视图 200 Hz；读取 yUnit，缺失默认 uV 不算单位证据 | NIRScout，36 个双波长位置，760/850 nm；采集 12.5 Hz，发布视图 10 Hz；原值为光电探测 V，尚非血红蛋白浓度 |
| Simultaneous | Shin 等，2018，Scientific Data；n-back、DSR、word generation | BrainAmp，TP9 参考、TP10 ground；发布 200 Hz、28 scalp 加 HEOG/VEOG；读取单位为 µV | NIRScout，采集约 10.4 Hz、发布 10 Hz；36 位置；MATLAB oxy/deoxy 已是 mmol/L 浓度变化 |
| Visual | A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults，Data in Brief，2024 | Neurofax EEG-1100；论文 32 electrodes、500 Hz；连续 EDF 按物理/数字量程解码，本面板 header 为 µV；采集参考仍待核定 | ETG-7100、10 Hz；695/830 nm 设备信息；CSV 已是 Oxy/Deoxy，导出单位未报告 |
| REFED | NeurIPS 2025 Datasets and Benchmarks；情绪视频与连续主观标签 | ESI Neuroscan、64 channels、AFz 参考；README/正文 1000 Hz；loader 写 V，但原 MAT 数值单位证据尚未闭合 | LABNIRS、51 channels、47.62 Hz、780/805/830 nm；含 HbO/HbR/HbT 和三路 Abs；不能当成单一量纲 |

数据集目录、原作者说明、采集与发布频率辨析来自项目 `docs/DATASETS_DESCRIPTION.md` 的证据复核。REFED 论文附录的 200 Hz 与正文/发布说明 1000 Hz 不一致，本项目按实际入口 1000 Hz 解读。单位未知时保留未知，不从幅值猜测。

本报告的“原始 fNIRS”是当前缓存 `native_input_fnirs` 保存的未标准化发布输入副本（float32）；不是名为 `raw_native_fnirs` 的已处理分支。原 EEG 由项目原生 reader 读取；fNIRS 原文件 SHA 由 producer manifest 留存，本次另核对所读缓存 SHA。精度限制明确保留。

### 1.2 当前实际流程

通用 EEG：原参考/数据集对应清理分支 → 1–45 Hz → 200 Hz → 整 record 逐通道 median/MAD。Single-Trial 使用 v4 清理缓存，Simultaneous 使用 EOG-only 分支；Visual 连续 EDF 与 REFED 发布 MAT 保留各自上游差异。

通用 fNIRS：Single-Trial 强度比→OD→TDDR-like 运动处理→0.01–0.2 Hz→近似 MBLL；其他三集从已发布色团进入可用后处理 → 10 Hz → 整 record 逐通道 median/MAD。它不是完整 HOMER2 复现，固定近似系数、3 cm 与路径因子 6 不能自动赋予绝对 µM 标定。

SSM：限定原训练 trial → EEG 宽带 log-power/训练 PCA + 原生光学特征 → 训练折共同 Hb 尺度与参考模型幅度桥 → 4 Hz、3 个观测量。v3 进一步把 baseline、滤波、重采样与缺失插值纳入时序均值/噪声。下一代 tokenizer 尚无已完成训练结果。
{figure('01_pipeline')}

## 2. 四数据集实际数据对比

### 2.1 统计面板与定义

{markdown_table(scope_table)}

Single-Trial 为 subject_01/09/18 × session_01/03/05，原 MA 位置 4/9 排除；Simultaneous 为 VP001–003、三任务各取前八个非重叠允许窗口；Visual 为 S01–03 的 Probe1、S01 两 Part，各记录前八个非重叠窗口；REFED 为 1–3 号被试、video_1/2 各取三个连续非重叠 30 s 窗口。采样规则在读取信号前固定，不按幅值、拟合或标签好坏选择。

当前 loader 的历史整记录预处理上下文原样保留；原始与当前数值统计只覆盖上述窗口。Single-Trial 原 MA 位置 4/9 不进入统计或 SSM 拟合；24–29 号被试与受保护比较工件未打开。完整窗身份见 `measurement_windows.json`，来源/实际清理状态见 `measurement_provenance.json`。

**标准差**取每个通道每个窗口的时间标准差（ddof=0）；表中的 SD 是这些值的中位数。**最大幅值**指已审计支持中的最大绝对值，而不是正向最大值。另导出 RMS、峰峰值、稳健 SD、P99(|x|)、有限值比例。重叠依赖已尽量避免，但通道/记录/被试仍有层级依赖；73,824 条统计不等于同样多的独立样本。原始和处理后采样率可不同，对比的是同一实际时间支持，不逐点配对。

### 2.2 数值结果

{markdown_table(sd_table)}

当前列均为各通道的 record robust SD 坐标；Single-Trial 原始两行是 760/850 nm，处理后对应 HbO/HbR，只能作处理阶段对照，不能直接除出“浓度变化倍率”。REFED EEG 的小数值与另外三集有约六个数量级差异，与 V/µV 的数值差异相容，但这一观察不是单位确认。

对于单位已知的 Simultaneous，HbO/HbR 原始窗口 SD 中位数约为 **1.67/0.51 µM**（mmol/L 数值乘 1000）。其余色团单位未闭合，不能把这两个数与另外三集按共同 µM 坐标排名。
{figure('02_raw_scales')}
{figure('03_current_scales')}

处理后主体分布更接近，但 EEG 的最大绝对值仍约 15–58 个各自 robust SD；Visual HbR 还出现约 58.74。不能把 MAD=1 或有限值比例=1 当作质量已经一致。标准化可让典型量级相近，无法证明传感器噪声、生理信息和尾部风险相同。

## 3. 原始、当前与理想输入应是什么样

### 3.1 实际数据的逐阶段曲线
{figure('04_eeg_waveforms')}
{figure('05_fnirs_waveforms')}

这里的候选使用同一个数据集前两名被试的允许窗口估计一个共同 Hb 尺度，并冻结到第三名；仍沿用现有清理。首 5 s 只作为图示参考区间：Single-Trial 是指定任务前段，其他范式未统一核定为静息。**这些候选曲线不是理想生理真值，也没有证明跨被试泛化改善。**

### 3.2 已直接确认的幅度关系改变

{markdown_table(ratio_df)}
{figure('06_ratio_distortion')}

逐通道除以不同正尺度，保留各自波形形状，却改变 HbO/HbR 的相对幅度；若下游丢掉这些尺度信息，物理幅度关系就不再直接可用。共同正尺度保留同一位置的标准差比。正比例转换本身不会翻转 Pearson 相关的正负；因此 HbR 拟合形状或极性错误不能全部归因于正比例幅度缩放。

### 3.3 共同尺度候选也有明确负结果
{figure('07_candidate_tail')}

当前 record/channel 归一化把不同通道的绝对增益和大幅异常一起压缩；共同尺度把它们保留下来，模型可能反而面对更大的数值尾部。此次简单候选的极值远大于当前输出，所以不能“删掉逐通道 MAD 就直接训练”。应先区分测量无效、运动/系统性扰动与真实生理幅度，保留质量证据并采用明确的噪声/稳健目标。不能用任意 clip 捏造干净真值，也不能凭缓存伪迹标记静默删掉原合同允许的数据。

### 3.4 理想输入是保持信息与观测合同一致
{figure('08_ideal_fnirs')}
{figure('09_ideal_eeg')}

SSM 应看到有清晰观测映射的相对能量和 Hb 特征，连同尺度、baseline、时钟、mask 与噪声。波形 tokenizer 可看到已知单位缩放后的多通道测量；若使用 SSM teacher，则输出要区分 teacher 预测、残差和有效支持，不能把失败拟合伪造为训练目标。不同数据集不应被强迫变成相同 PSD、同一反相关模板或同一标准差。

## 4. 缩放对 HbO/HbR 拟合的实验

### 4.1 机制与实验设计

已知观测缩放 D 应满足 **y′=Dy、h′=Dh、R′=DRDᵀ**，并保留对应密度变换；观测单位改变时潜状态、Q 和初态先验不变。训练尺度冻结后不能随候选重拟合目标。对角变换可逆并同步使用时，理论上不改变同一模型的潜状态推断。未知观测 gain 是另一项统计假设。

| 分支 | 数据 | 观测均值 | 噪声 | 用途 |
| --- | --- | --- | --- | --- |
| 基线 | 原值 | 原值 | 原值 | 当前点式参考 |
| 共同/仅数据 | Hb 两列 ×0.5 | 不变 | 不变 | 错误坐标使用负对照 |
| 共同/同步 | Hb 两列 ×0.5 | 同步 ×0.5 | SD 同步 ×0.5 | 已知单位等价性 |
| 分色团/仅数据 | 各色团训练 MAD 因子 | 不变 | 不变 | 相对观测映射失配负对照 |
| 分色团/同步 | 同上 | 各自同步 | 各自同步 | 验证可逆缩放本身不丢信息 |
| 共同/数据+噪声 | Hb 两列 ×0.5 | 不变 | SD ×0.5 | 改变观测 loading 的敏感性 |

**负对照不声称是当前 SSM 的代码路径。** 它们检验了“错误使用缩放能否造成负面影响”的因果机制。当前 SSM 已用共同 Hb 因子；通用 loader 的逐通道方法确有第 3 节的比例畸变，但两条事实不能直接串成“当前所有 SSM 失败由逐通道 MAD 导致”。

合成训练 24 个独立 realization，用于拟合两个色团的尺度；另用 24 个独立 realization 评估。使用当前固定 W=0 的六状态 Balloon/Student-t 观测生成规律、120 点/4 Hz，固定求解器与 quadrature=7，无参数网格。每个评估 trial 比较六分支和完整/中心 fNIRS 缺失两模式，共 288 次。EEG/HbO/HbR 误差按冻结合成训练 clean SD 归一化；r 误差按各评估 realization 自身的真值 SD 归一化，六分支共用该分母；最后均对 trial 算术平均。中心缺失模式的 HbO/HbR 只在隐藏点评分，r 使用完整轨迹。这与旧 v3 的层级聚合风险 B 不同，不跨表排名。

实测使用历史 v3 保存的 subject_01/09/18 外折 o1 的六个验证 trial，共 18 个原训练身份；尺度只从对应 18 个训练 trial 拟合。模型/噪声基准来自同一冻结输入记录。这里用点式 O0 做幅度敏感性、在**最终观测层**施加中心缺失；不复现 v3 的原生特征缺失插值或 O1/O2 时序推断。只回答这组输入的尺度机制，不替代 72-trial/14-mode 主协议或 teacher 资格。

### 4.2 合成真值结果

{markdown_table(syntab)}
{figure('11_synthetic_metrics')}
{figure('12_synthetic_trace')}

24/24 评估 trial 在各分支/模式均完成且通过物理检查。中心缺失时，错误共同缩放使 HbO/HbR NRMSE 约由 0.120/0.107 升到 0.196/0.173；错误分色团缩放升到 0.392/0.415。神经驱动恢复也退化。同步缩放与原坐标完全一致，合成状态最大差为 {max(x['state_max_abs'] for x in syinv):.3g}；校正后 log-likelihood 最大差为 {max(x['corrected_ll_error'] for x in syinv):.3g}。

这说明负面影响来自变换后仍使用不匹配的观测均值/噪声，而不是“所有归一化都有害”。配对差与 2,000 次 bootstrap 描述区间见 `synthetic_paired_differences.csv`；独立样本量仍为 24，bootstrap 次数不是新增实验。这里只对该生成规律成立，未模拟完整原生光学、空间混合和 EEG 特征构造误差。

### 4.3 实测拟合结果与反例

{markdown_table(meatab)}
{figure('13_measured_metrics')}
{figure('14_measured_trace')}

216 次拟合全部完成并通过同一物理检查；每分支/模式分母都是 18。同步缩放保持相同状态，最大差为 {max(x['state_max_abs'] for x in meinv):.3g}。但错误分色团分支使隐藏 HbR NRMSE 从 2.931 降到 1.916，尽管同一规则在合成真值上损害状态恢复。**观测误差较低可以来自补偿了观测映射/先验的失配，而不代表正确的生理状态。**

固定首个实例 subject_01/1 的 full 基线中，HbR 目标 SD 约 0.0150，预测 SD 约 0.0386，平均偏差约 +0.0872，相关约 −0.730。分色团负对照把平均偏差压到约 +0.0356，但相关仍约 −0.740。这个实例说明幅度/偏置变小并未修正形状；它是预先固定实例，不能当作全部 trial 的相关结论。

本组未作 14 模式 null 评价、跨数据集 SSM 标定、参数可辨识性或 tokenizer 训练；18 个 trial 归属于 3 名被试，不能把它们当 18 名独立被试作总体显著性推断。原有参考幅度桥、宽带 PCA 与局部 Hb 位置之间的物理对应仍是待检验假设。

### 4.4 与过去实验证据合读
{figure('10_temporal_noise')}
{figure('15_retained_v3')}
{figure('16_precision_and_floors')}

旧 N5 在共同有效 51/72 身份上把 B 从 5.953 降至 4.387（约 26.3%），却令匹配合成 r NRMSE 增加约 0.103。最新 v3 的增益适配仍不完整/不确定；时间均值与相关噪声在合成上有效，但实测主风险仍不完整。因此目前应优先修复/验证观测合同，而非放宽生理参数或继续扩大 gain/Q 网格。

当前参考幅度桥把训练 MAD 匹配到模型先验预测 SD；39 份保留投影的 EEG 特征噪声最终值全为合成下限 0.10328；48 份输入重组中 9 份失败，仅提升 baseline 精度后仍有 5 份。这些是独立的工程/建模问题。正比例共同尺度不会改变 HbO/HbR 原始比例或直接翻转极性；单凭 HbR 拟合一直困难，不能证明共同尺度就是唯一病因。

## 5. 新方案：逐项修改与预期改善

以下“预期改善”是可检验目标，不是本次已经取得的训练增益。继续复用已有 reader、预处理与观测 spec；旧缓存、旧参数和失败证据保留原身份。

### 5.1 四个数据集分别怎样修改

| 数据集 | 测量层修订 | 进入 SSM / tokenizer 前的要求 |
| --- | --- | --- |
| Single-Trial | 保留读取到的 EEG µV 与原参考；光电 V 先转 OD，再用与波长、对数底、消光系数和光程一致的 MBLL。现有近似结果继续标为相对 Hb，不能补写成绝对 µM | 先在已限定训练支持上形成 float64 特征；HbO/HbR 同尺度，baseline 与处理算子同步进入均值和噪声；保留模型增益与数值尺度各自的来源 |
| Simultaneous | EEG 保留 µV；已发布 Hb 的 mmol/L 数值乘 1000 转 µM，避免再次执行 MBLL；保留原作者处理与 EOG-only 分支身份 | 用已知浓度进行单位等价性检查；保留 Hb 对比例和原任务时钟，再比较共同训练尺度；不把三种任务的任意窗口开头都当静息 |
| Visual | 按 EDF 校准得到 µV；保留 Part/Probe 与事件对齐边界。Oxy/Deoxy 的 ETG-7100 导出单位继续待核，不从数值推定光程或 µM | 在单位未核定时保留独立相对测量组；本次 CH6 极端支持先作质量定位，避免用当前逐通道缩放隐藏、或直接用共同尺度放大其训练权重 |
| REFED | 核实原 MAT 的 EEG 单位：只有确认 V 后才乘 10^6 转 µV。HbO/HbR/HbT 与 780/805/830 nm Abs 分列，已发布色团不重复光学转换 | 保留 47.62 Hz 原生时钟并显式记录重采样；核查 HbT 与 HbO+HbR 的同单位、同基线条件；未知色团单位保持相对组，不与已知 µM 混称物理一致 |

测量层统一后的建议顺序为：**有证据的物理单位 → 质量与时空支持 → 有来源的基线/特征 → 训练数据拟合并冻结的数值尺度 → 消费者输入。** 未知物理单位与已知单位可以共享相对特征研究，但不能借由归一化宣称绝对生理标定已经完成。

### 5.2 逐项实现与验收

| 优先级 / 修改点 | 具体修改 | 预期改善与验收 |
| --- | --- | --- |
| P0 单位证据 | 单位与证据来源分列；核实 EEG V/µV、色团 mmol/L/µM；未知设备保留未知组 | 消除可确认的数量级差；单位等价输入应产生等价状态 |
| P0 输入阶段 | 明确 intensity、OD、HbO/HbR、Abs；已发布 Hb 不重复 MBLL；审计系数、光程、对数底 | 避免把不同物理量混成一条归一化路径；已知浓度合成正例闭合 |
| P0 精度 | 新版本测量/特征边界保留 float64，统一实际处理与算子次序 | 消除可归因舍入的重组失败；不宣称恢复旧 float32 的丢失精度 |
| P0 EEG 功率下限 | 先统一电位单位，功率用明确单位，再做 log(P/P_ref)；floor 随单位转换 | 避免 V² 与 µV² 同常数 floor 导致 10^12 物理阈值差 |
| P0 观测同步 | 数据、均值、雅可比、R、输出逆变换与评分尺度同步 | 已知缩放不改变状态；保留错误缩放负对照作为回归测试 |
| P0 噪声证据 | 保存下限前估计、下限来源、最终值与触发率；分清 Student-t scale 和 SD | 判断噪声是否被合成下限支配，避免隐藏模型权重变化 |
| P1 成对幅度 | HbO/HbR 使用同一正尺度，训练组拟合后冻结；保留逆变换和组身份 | 保留比例、HbT 加和与幅度差；同时审查尾部，不能直接部署本次简单候选 |
| P1 baseline | 根据采集范式确认静息/事件前支持；不足时显式标记；baseline 算子进入均值和噪声 | 避免把任意前 5 s 当静息，保留基线不确定性与初态关系 |
| P1 时间处理 | 保留真实秒时钟、仪器偏移与生理延迟；滤波/重采样传播完整协方差 | 降低时间模型失配；30 s 窗不能视为充分测得 0.01 Hz 慢周期 |
| P1 质量与尾部 | 定位极端通道/时段；区分无效测量与伪迹注记；稳健噪声/损失候选独立验证 | 减少极端值主导训练，同时保留真实响应；不以任意 clip 冒充清理成功 |
| P1 空间与参考 | 保留 EEG 参考、电极覆盖、Hb 配对位置；先固定当前选择，再独立比较空间/参考修订 | 避免把“最平滑位置”误作同一局部脑源；不能由尺度统一替代空间对应 |
| P1 模型幅度桥 | 将模型无关数值缩放与观测 loading 分离；不随 Q 或参数候选重新配数据幅度 | 减少测量增益与潜状态幅度混淆；未知绝对标定保持相对解释 |
| P1 缺失 | 原生缺失在非线性之前施加；特征缺失单独声明；补值不计真实 support | 防止隐藏值经滤波、功率或统计量进入输入/目标 |
| P2 求解与动态 | 分类输入失败、预算失败、非法路径；核对非零 driver 积分/插值；gain 与 Q 单独检验 | 把剩余求解/动力学问题与缩放分开，避免通过放宽边界掩盖失配 |
| P2 tokenizer 接口 | 保留物理/相对测量坐标、训练尺度、模态/位置/mask；teacher 仅在合格支持上发出目标 | 降低输入适配负担，保持可追溯性；此项尚未训练，无准确率承诺 |
| P2 验收 | 同一输入身份、冻结 scoring SD、完整失败分母；合成状态与实测配对增量并列 | 防止只优化曲线/成功子集；完整 72/14-mode 诊断支持后再扩数据集 |

### 5.3 执行顺序与通过条件

建议执行顺序：**单位/精度/同步合同 → 独立合成验证 → 固定 72 个开发身份的完整对照 → 单独检验噪声/观测 gain/Q → 跨数据集验证与 tokenizer 输入实现。** 不将本报告的 18-trial 点式诊断当作完成下一轮主协议。

下一次开发验证应在运行前固定以下判断：已知单位与共同/分色团坐标重表达的状态最大差不超过 1e-8；成对共同尺度的 SD 比变化仅允许浮点舍入；所有尺度只在训练支持拟合；失败数、物理检查和全分母误差同时报告。幅度尾部不设一个缺乏测量依据的通用 clip 阈值，而是逐数据集报告异常来源、保留/排除依据和训练损失贡献。完整 SSM 验证沿用既有主协议，不由本报告另设资格规则。tokenizer 的改善还需独立训练对照，以损失、残差尺度及跨数据集任务结果验证，当前不能承诺准确率增益。

## 6. 开源 foundation model 的相关经验

LaBraM/CBraMod 的固定物理尺度思路可以借鉴，BIOT 的逐通道分位数尺度强调设备鲁棒性，REVE 显式利用通道空间与时钟，NormWear 使用时频输入适配多类信号。它们的目标、原生输入和数据组成不同，不能直接证明某个逐通道缩放适合生理 SSM。

| 方法 | 相关处理 | 本项目的取舍 |
| --- | --- | --- |
| LaBraM | 200 Hz、以 100 µV 为单位输入；频谱 target 另作标准化 | 原电位物理缩放与特征 target 归一化分开 |
| CBraMod | 200 Hz、TUEG 预训练使用 100 µV 阈值/固定尺度 | 幅度规则需有真实电位单位；不照搬到 Hb |
| BIOT | 通道绝对幅值 P95 缩放与通道/时间 token | 保留变通道与缺失语义；不要丢掉 Hb 对尺度关系 |
| REVE | recording-session z-score、3D/时间位置编码 | 空间信息可借鉴；session 统计不能冒充 train-only 规则 |
| NormWear | 去趋势/尺度处理与 CWT，原训练未包含 fNIRS | 按模态定义时频表示；不能作为 fNIRS 物理标定证据 |

## 7. 来源、复现与结论边界

### 7.1 一手资料

- Single-Trial：本地原始说明 `data/EEG+NIRS Single-Trial/Open access dataset for simultaneous EEG and NIRS Brain-Computer Interfaces (BCIs).html`。
- Simultaneous：本地 `data/Simultaneous EEG&NIRS/Dataset description_MATLAB.pdf` 与 `Dataset description_BrainVision and NIRx.pdf`。
- [Visual 原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10964074/)；本地发布 readme 与 EDF/CSV 头。
- [REFED 原论文](https://papers.neurips.cc/paper_files/paper/2025/file/2bf0ed7c35d9d84128e7f7c72ab76402-Paper-Datasets_and_Benchmarks_Track.pdf)；本地发布 README。
- [MNE MBLL 接口](https://mne.tools/stable/generated/mne.preprocessing.nirs.beer_lambert_law.html)、[SNIRF 测量元数据](https://fnirs.github.io/snirf/)。
- [LaBraM 论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf)、[CBraMod 论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/bbbd6d915cb90be21c1254a82d45cedd-Paper-Conference.pdf)、[BIOT 论文](https://arxiv.org/pdf/2305.10351)、[REVE 论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/20a917f77773ac0fa8bea2bdd6606b66-Paper-Conference.pdf)、[NormWear 论文](https://arxiv.org/html/2412.09758v1)。

### 7.2 复现与完整证据

入口：`experiments/scripts/analyze_dataset_scaling.py`；配置：`experiments/configs/physiology_semantic_tokenizer/dataset_scaling_report_v1.yaml`。阶段顺序为 `--stage dry-run`、`synthetic`、`measurements`、`measured-ssm`、`report`；已有完成阶段禁止覆盖。复现需使用新 run 身份并保留同一范围。

本目录的 `resolved_config.yaml`、`inventory.json`、各阶段 completion、`source_identity.json` 与 `source_snapshot/` 保留配置、确切阶段源码 SHA 和依赖身份。`channel_window_statistics.csv` 是完整数值统计，`amplitude_summary.csv`/`ratio_summary.csv` 是对应派生表；`synthetic_fits.csv`/`measured_ssm_fits.csv` 保存全部拟合行及失败字段，`*_invariance.json` 保存逐例等价性，`*_example.json` 保存固定实例曲线。HTML 内嵌图；PDF 可独立阅读；`FIGURES.pdf` 与 SVG 可用于放大/导出。

历史结果仍由原 N1–N7/v3 run 持有，本报告仅引用；新计算拥有自己的配置和身份。没有修改冻结数据、旧结果或 registry 状态，没有训练 tokenizer，没有开展 protected evaluation。新处理方案的总体效益、跨数据集 SSM 物理标定与 teacher 资格仍未建立。

**最终结论：过去逐通道处理的幅度关系损失已经在实测中确认；不一致缩放的状态恢复损害已经在受控合成中确认。但共同缩放本身并不必然损害 HbO/HbR 拟合，实测误差降低也不保证状态正确。下一版应以可追溯测量、成对幅度、同步观测/噪声合同和完整验证为核心，并先解决共同尺度暴露的极端尾部。**
'''
    exported=export_report_document(report,out,captions,title='EEG-fNIRS 数据统一化诊断',date='2026-09-15')
    validation=dict(exported,figures=len(captions),
        raw_windows=len(windows),statistics_rows=len(stats),synthetic_rows=len(syn),measured_rows=len(mea),
        expected_counts_pass=len(windows)==194 and len(syn)==288 and len(mea)==216,
        all_figures_exist=all((out/'figures'/f'{n}.png').exists() for n in captions),
        pdf_figures='separate PNG figure pages',
        unique_subject_record_count=int(counts['records'].sum()))
    write_json(out/'report_validation.json',validation)
    write_json(out/'figure_captions.json',captions)
    return validation


def alignment_fit_comparison(candidate, reference):
    """Compare retained fits by identity, including every failed trajectory."""
    keys = ['sample_id', 'mode', 'solver']
    candidate = candidate[candidate.solver.isin(['O0', 'O1', 'O2'])].copy()
    reference = reference[reference.solver.isin(['O0', 'O1', 'O2'])].copy()
    for frame in (candidate, reference):
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError('missing or duplicate fit identity')
        if not frame.status.isin(['completed', 'failed_domain', 'failed_numerical']).all():
            raise ValueError('unrecognized fit outcome')
    common = reference[keys + ['status']].merge(
        candidate[keys + ['status']], on=keys, how='left',
        suffixes=('_reference', '_candidate'), validate='one_to_one')
    if common.status_candidate.isna().any():
        raise ValueError('candidate is missing a retained reference fit')
    transitions = common.groupby(['solver', 'status_reference', 'status_candidate']).size().rename('count').reset_index()
    return candidate, reference, common, transitions


def render_alignment_report(out):
    """Dated, read-only synthesis of the alignment repair and retained S1/S2 run."""
    import os
    import shutil
    import fitz
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    root = ROOT / 'experiments/runs/physiology_semantic_tokenizer'
    out = Path(out).resolve()
    if not out.is_relative_to(root / 'data_quality_audit'):
        raise ValueError('alignment report must use the existing data_quality_audit artifact root')
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('retained export is immutable; choose an empty new versioned directory')
    run = root / 'ssm_overnight/20260916_measurement_alignment_v3'
    old = root / 'ssm_overnight/20260911_observation_contract_v3_continuation_v1'
    audit = root / 'data_quality_audit/20260916_dataset_alignment_v3'
    previous = root / 'data_quality_audit/20260915_dataset_scaling_report_v1'
    sources = set()

    def read_json(path):
        sources.add(path)
        return json.loads(path.read_text())

    def read_csv(path):
        sources.add(path)
        return pd.read_csv(path)

    def arrays(path):
        sources.add(path)
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k].copy() for k in f.files}

    summary = read_json(run / 'summary.json')
    manifest = read_json(run / 'manifest.json')
    if manifest['execution'] != 'completed' or not summary['final']:
        raise ValueError('report requires a completed retained run')
    comparison = read_json(run / 'measurement_comparison.json')
    prep = read_json(run / 'measurement_preparation_audit.json')['rows']
    engineering = read_json(audit / 'engineering_checks.json')
    mbll = read_json(audit / 'mbll_independent_check.json')
    smoke = read_json(audit / 'public_loader_smoke.json')
    qc = read_json(audit / 'visual_ch6_quality.json')
    scope = read_json(run / 'scope_inventory.json')
    candidate, reference, common, transitions = alignment_fit_comparison(
        read_csv(run / 'S2/trial_metrics.csv'), read_csv(old / 'S2/trial_metrics.csv'))
    if len(candidate) != 3024 or len(reference) != 2268 or len(common) != comparison['common_fits']:
        raise ValueError('unexpected fixed fit denominator')
    expected_ids = {r['sample_id'] for rows in scope.values() for r in rows}
    if len(expected_ids) != 72 or set(candidate.sample_id) != expected_ids:
        raise ValueError('retained fit identities disagree with scope inventory')
    if not (candidate.groupby(['sample_id', 'solver']).size() == 14).all():
        raise ValueError('incomplete 14-mode identity panel')
    if candidate.status.value_counts().to_dict() != comparison['candidate_statuses']:
        raise ValueError('retained fit outcomes disagree with comparison')
    if reference.status.value_counts().to_dict() != comparison['reference_statuses']:
        raise ValueError('retained reference outcomes disagree with comparison')
    retained_transitions=pd.DataFrame(comparison['transitions']).rename(columns={
        'reference_status':'status_reference','candidate_status':'status_candidate'})
    transition_keys=['solver','status_reference','status_candidate']
    pd.testing.assert_frame_equal(transitions.set_index(transition_keys).sort_index(),
                                  retained_transitions.set_index(transition_keys).sort_index())
    synthetic = read_csv(run / 'S1/solver_comparison.csv')
    synthetic_old = read_csv(old / 'S1/solver_comparison.csv')
    skeys = ['law', 'variant', 'mode', 'solver']
    snew = synthetic.set_index(skeys).sort_index()
    sold = synthetic_old.set_index(skeys).sort_index()
    pd.testing.assert_frame_equal(snew, sold, check_exact=True)
    if len(synthetic) != 150 or synthetic.completed.sum() != 3600:
        raise ValueError('incomplete synthetic panel')
    projections = [read_json(run / 'prepared' / (r['projection'] + '.json')) for r in prep]
    if len(projections) != 48:
        raise ValueError('unexpected projection denominator')
    errors = np.array([r['recomposition_error_training_sd'] for r in prep])
    sd_changes = [r['max_relative_scoring_sd_change'] for r in prep if 'reference_projection' in r]
    noise_eeg = np.array([p['feature_noise']['evidence']['eeg']['estimate_before_floor'] for p in projections])
    final_eeg = np.array([p['feature_noise']['eeg_sd'] for p in projections])
    noise_od = np.array([p['feature_noise']['evidence']['optical_density']['estimate_before_floor'] for p in projections])
    od_triggers = int(sum(sum(p['feature_noise']['evidence']['optical_density']['triggered']) for p in projections))
    p0 = read_json(run / 'prepared/subject_01_o0_E0.json')
    prepared = arrays(run / 'prepared/subject_01_o0_E0.npz')
    example = candidate[(candidate.subject == 'subject_01') & (candidate.trial == 0) & (candidate['mode'] == 'full')]
    example_arrays = {row.solver: arrays(run / row.trajectory_path) for row in example.itertuples()
                      if isinstance(row.trajectory_path, str)}
    ex0 = read_json(run / 'cells/v3_S2__subject_01__0__0__O0/0__full.json')
    ex1 = read_json(run / 'cells/v3_S2__subject_01__0__0__O1/0__full.json')
    np.testing.assert_array_equal(prepared['target'][0],example_arrays['O0']['target'])
    modes = read_csv(run / 'mode_status.csv')
    modes = modes[modes.rule.isin(['S2/O0', 'S2/O1', 'S2/O2'])].copy()
    for row in modes.itertuples():
        fits=candidate[(candidate.solver==row.rule.split('/')[1]) & (candidate['mode']==row.mode)]
        if len(fits)!=row.expected or int((fits.status=='completed').sum())!=row.completed:
            raise ValueError('retained mode summary disagrees with trajectory rows')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir()
    transitions.to_csv(out / 'paired_status_transitions.csv', index=False)
    counts = candidate.groupby(['solver', 'status']).size().unstack(fill_value=0)
    counts.to_csv(out / 'fit_status_counts.csv')
    modes.to_csv(out / 'mode_status.csv', index=False)
    synthetic.to_csv(out / 'synthetic_comparison.csv', index=False)
    pd.DataFrame(prep).to_csv(out / 'preparation_audit.csv', index=False)
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'font.size': 11, 'axes.titlesize': 12, 'legend.fontsize': 9,
                         'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
    captions = {}
    atlas = fitz.open()
    colors = ['#237f75', '#d58a34', '#bb5361', '#a8b1bb']

    def save(fig, name, title, caption):
        fig.suptitle(title, fontsize=16, fontweight='bold', y=.995)
        fig.tight_layout(rect=[0, .01, 1, .955])
        png = out / 'figures' / f'{name}.png'
        fig.savefig(png, dpi=250, bbox_inches='tight')
        plt.close(fig)
        captions[name] = (title, caption)

    fig, ax = plt.subplots(figsize=(12, 7.1)); ax.set(xlim=(0, 12), ylim=(0, 7)); ax.axis('off')
    rows = [
        (5.7, '原始记录', '已证实 EEG 电位 / 光电强度 / 已发布 Hb\n原单位、时钟、通道、参考、事件身份保留', '#e8eef2'),
        (3.6, '通用测量 loader', '连续记录处理；200 Hz EEG + 10 Hz Hb\nfloat64；不做逐通道 MAD；未知单位独立分组', '#e0f1ed'),
        (1.4, 'tokenizer 测量接口', '6 EEG × 4000 + 2 Hb × 200；10 个 2 s patch\n位置 / 单位 / mask 保留；尚未训练', '#e0f1ed')]
    for y, title, body, color in rows:
        ax.text(3, y, title + '\n' + body, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=.8', fc=color, ec='#8096a2'))
    ax.text(9.2, 3.6, '本轮 SSM 独立有界入口\n先裁 72 个训练窗口，再做特征\n4 Hz；120 × 3；fold 内拟合投影\n原生缺失和特征缺失分开', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=.8', fc='#fff0d9', ec='#bda17b'))
    ax.text(9.2, 1.4, 'SSM S1 / S2 已运行\n均值与噪声同步；仍有物理 / 数值失败\n未建立 teacher 资格', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=.8', fc='#fae6e8', ec='#c18d94'))
    for start, end in [((3, 5.0), (3, 4.4)), ((3, 2.8), (3, 2.15)),
                       ((6.0, 5.7), (9.2, 4.55)), ((9.2, 2.6), (9.2, 2.1))]:
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', lw=1.8, color='#526f81'))
    save(fig, '01_routes', '统一的是数据合同，消费者使用的张量仍各不相同',
         '绿色路径为当前可选 measurement loader 和本地 tokenizer 接口；橙色路径为本轮实际 SSM 输入。SSM 未经过通用逐通道 MAD 输出，也没有部署新的成对 adapter 作为统计候选。箭头表示处理顺序，不表示已完成 tokenizer 训练。')

    fig, axs = plt.subplots(2, 1, figsize=(11, 6.6))
    for ax in axs: ax.set_xlim(-5, 25); ax.set_xlabel('相对 MA 事件的时间 / s'); ax.axvline(0, color='#566e80', ls='--')
    axs[0].broken_barh([(-5, 30)], (.7, .32), facecolors='#dce7ef')
    axs[0].broken_barh([(-5, 5)], (.7, .32), facecolors=colors[0])
    axs[0].text(10, 1.25, '原生窗口先裁剪：EEG 6000 × 30；光强 300 × 36 × 2', ha='center')
    axs[0].text(-2.5, .86, '参考基线', ha='center', color='white')
    axs[0].text(12, .86, '基线不是已知潜在静息态', ha='center')
    axs[0].set(ylim=(.35, 1.65), yticks=[])
    axs[1].plot(np.arange(120)/4-5, prepared['target'][0,:,1], color=colors[0], label='实际 SSM HbO 坐标（固定首例）')
    axs[1].axvspan(-5, 0, color=colors[0], alpha=.12)
    axs[1].set_ylabel('模型观测坐标'); axs[1].legend(loc='upper left')
    save(fig, '02_time_support', '原始时钟、事件前基线与 SSM 输出时钟',
         '示例固定为 subject_01 / session_01 / event 1。30 s 原生窗口经过处理后输出 120 个 4 Hz 点；EEG 功率块和 fNIRS 的 10 Hz 特征先保留各自时钟。基线使用事件前 5 s 的 20 个输出点；双向滤波与全窗推断仍是离线处理。')

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    axs[0].semilogy(np.arange(1,49), errors, '.', color=colors[0], ms=7, label='48 份新投影')
    axs[0].axhline(1e-6, color=colors[2], ls='--', label='合同容差 1e-6')
    axs[0].set(xlabel='冻结投影编号', ylabel='最大重组误差 / 训练 SD', title='精度修复：全部低于容差')
    axs[0].legend()
    axs[1].plot(np.arange(1,49), noise_eeg, '.', color=colors[0], label='训练差分 MAD 估计')
    axs[1].plot(np.arange(1,49), final_eeg, '-', color=colors[2], label='最终值：合成模型噪声下限')
    axs[1].set(xlabel='冻结投影编号', ylabel='EEG 特征噪声 SD', title='噪声假设：48/48 仍被下限支配')
    axs[1].legend(loc='center right')
    save(fig, '03_precision_noise', '输入重组已修复，噪声标定问题仍独立存在',
         f'48/48 输入重组通过，最大误差 {errors.max():.4g} 个训练 SD。EEG 下限前估计范围 {noise_eeg.min():.5f}–{noise_eeg.max():.5f}，最终均为 {final_eeg[0]:.5f}；光学特征 {od_triggers}/96 个波长估计触发下限。本轮为隔离工程变化保留了原噪声规则。')

    fig, axs = plt.subplots(1, 3, figsize=(12, 4.8))
    laws = ['linearized_gaussian', 'nonlinear_gaussian', 'nonlinear_student_t']
    solvers = ['O0', 'O1', 'O2', 'O2_pointwise', 'O2_mean_only']
    palette = ['#a0acb8', '#399c94', '#315d99', '#c28557', '#b76276']
    for ax, metric, title in zip(axs, ['r_nrmse','clean_HbO_nrmse','clean_HbR_nrmse'], ['潜状态 r','干净 HbO','干净 HbR']):
        for i, solver in enumerate(solvers):
            frame = synthetic[(synthetic.variant=='combined') & (synthetic['mode']=='full') & (synthetic.solver==solver)].set_index('law')
            ax.bar(np.arange(3)+(i-2)*.15, frame.loc[laws,metric], .145, label=solver, color=palette[i])
        ax.set(xticks=np.arange(3), xticklabels=['线性高斯','非线性高斯','非线性 t'], ylabel='NRMSE', title=title, ylim=(0,1.08))
        ax.tick_params(axis='x', labelsize=9)
    axs[1].legend(ncol=3, loc='upper center', bbox_to_anchor=(.5,1.28), fontsize=8)
    save(fig, '04_synthetic', '受控合成：完整时间观测与噪声算子仍有效',
         '图示 combined/full 子面板，每个柱为 24 seeds 的均值；完整实验为 150 条条件汇总、3600 次拟合。O2_pointwise 忽略时间处理；O2_mean_only 只同步均值。新旧 150 条汇总数值完全相同，因此这些优势是保留的既有证据，不能归功于此次 loader 修复。O2 没有后验区间或边际似然。')

    fig, axs = plt.subplots(1, 2, figsize=(11, 5.3))
    status_order = ['completed','failed_domain','failed_numerical','unattempted']
    for ax, frame, title in zip(axs, [reference,candidate], ['历史 v3（计划分母）','本轮 measurement v3（计划分母）']):
        bottom = np.zeros(3)
        for status, color, label in zip(status_order, colors, ['通过检查','生理域失败','数值失败','输入依赖未执行']):
            values = [int(((frame.solver==s)&(frame.status==status)).sum()) if status!='unattempted'
                      else 1008-int((frame.solver==s).sum()) for s in ['O0','O1','O2']]
            ax.bar(['O0','O1','O2'], values, bottom=bottom, color=color, label=label)
            for j,v in enumerate(values):
                if v: ax.text(j,bottom[j]+v/2,str(v),ha='center',va='center',fontsize=10,color='white' if status!='unattempted' else '#263c4c')
            bottom += values
        ax.set(ylabel='拟合次数；每求解器计划 1008 次',title=title,ylim=(0,1060))
    axs[0].legend(ncol=2,loc='upper center',bbox_to_anchor=(1.08,1.23),fontsize=9)
    save(fig, '05_outcomes', '实测完整分母：输入可执行性改善，SSM 成功率未获全面改善',
         '每个求解器分母为 72 身份 × 14 模式 = 1008。旧实验各有 252 次因输入依赖未执行；本轮全部执行。通过检查仍只表示求解与生理路径检查通过，不表示重建误差低、参数正确或获得 teacher 资格。')

    order = modes[modes.rule=='S2/O0']['mode'].tolist()
    matrix = np.array([modes[modes.rule==f'S2/{s}'].set_index('mode').loc[order,'completed'].to_numpy() for s in ['O0','O1','O2']])
    fig, ax = plt.subplots(figsize=(12,5.2)); im=ax.imshow(matrix, vmin=0,vmax=72,cmap='YlGnBu',aspect='auto')
    for i in range(3):
        for j in range(14): ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>42 else '#263c4c')
    ax.set(xticks=np.arange(14),xticklabels=order,yticks=[0,1,2],yticklabels=['O0','O1','O2'])
    plt.setp(ax.get_xticklabels(),rotation=50,ha='right',fontsize=9)
    fig.colorbar(im,ax=ax,label='通过次数 / 72')
    save(fig, '06_modes', '14 模式逐项检查：O2 的联合 full 输入仍为 0/72',
         '每格分母固定为 72。center 指中心特征段隐藏；own/template/pairing/shift 是相应的自身模态、模板、错配、错时对照。all_missing 只检验无观测先验路径。O2 的 EEG_only、all_missing、center_EEG_own 各 72 次成功，合计 216 次，不构成联合血氧拟合成功。')

    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    target = example_arrays['O0']['target']; t=np.arange(120)/4-5
    for k,(ax,label) in enumerate(zip(axs,['EEG 投影','HbO','HbR'])):
        sd=example_arrays['O0']['normalization_sd'][k]
        ax.plot(t,target[:,k]/sd,color='#203647',lw=1.6,label='实际测量目标 / 冻结训练 SD')
        ax.plot(t,example_arrays['O0']['prediction'][:,k]/sd,color=colors[0],label='O0：通过路径检查')
        ax.plot(t,example_arrays['O1']['clean_mean'][:,k]/sd,color=colors[2],ls='--',label='O1 clean map：生理检查失败，仅示意')
        ax.axvspan(-5,0,color='#b6c6d5',alpha=.2);ax.axvline(0,color='#a6b6c2',ls=':')
        ax.set_ylabel(label+' / SD'); ax.grid(alpha=.15)
    axs[0].legend(ncol=1,loc='upper left',fontsize=9)
    axs[-1].set_xlabel('相对事件时间 / s')
    save(fig, '08_fixed_example', '固定首个实测身份：曲线可计算不等于生理拟合合格',
         'subject_01 / session_01 / event 1 / full，预先按身份顺序选取，没有挑选最佳拟合。O0 的 EEG/HbO/HbR NRMSE 分别为 0.9784/0.7673/4.3480。O1 虚线是失败的 clean map，不能作为有效预测；其正式评分使用条件 noisy-target 预测。O2 数值失败，无有效曲线，未画零线代替。')

    fig, ax = plt.subplots(figsize=(8.5,5.5))
    o2=transitions[transitions.solver=='O2']; states=status_order[:3]
    mat=np.zeros((3,3),int)
    for row in o2.itertuples():mat[states.index(row.status_reference),states.index(row.status_candidate)]=row.count
    im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=534)
    for i in range(3):
        for j in range(3):ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=17,color='white' if mat[i,j]>200 else '#263c4c')
    labels=['通过检查','生理域失败','数值失败'];ax.set(xticks=range(3),xticklabels=labels,yticks=range(3),yticklabels=labels,xlabel='本轮',ylabel='历史共同身份')
    save(fig, '07_paired_changes', '共同身份上的 O2：2 次恢复、1 次退化',
         'O2 共同分母为 756；另两个求解器各 756 条的状态完全不变。新增可执行身份与共同身份分开汇报，避免把输入恢复带来的分母增加误写成算法成功率改善。状态翻转都出现在 center_EEG_shift。')
    for name, (title, caption) in sorted(captions.items()):
        png=out/'figures'/f'{name}.png'
        page, height = append_bitmap_figure(atlas, png)
        spare, _ = page.insert_htmlbox(fitz.Rect(42, 60 + height, 553, 800),
                                      f'<p><b>图 {int(name[:2])}｜{title}。</b> {caption}</p>',
                                      css='body{font-family:sans-serif;font-size:10pt;line-height:1.5}', scale_low=.85)
        if spare < 0:
            raise RuntimeError('atlas caption does not fit')

    atlas.save(out/'FIGURES.pdf', garbage=4, deflate=True, use_objstms=1);atlas.close()

    def figure(name):
        title, caption = captions[name]
        return f'![{title}](figures/{name}.png)\n\n**图 {int(name[:2])}｜{title}。** {caption}'

    def link(path, label):
        return f'[{label}]({os.path.relpath(path, out)})'

    def table(headers, rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                         ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])

    status_table=table(['求解器','通过检查','生理域失败','数值失败','通过比例','14 模式全通过身份'],[
        [s,counts.loc[s,'completed'],counts.loc[s,'failed_domain'],counts.loc[s,'failed_numerical'],
         f"{counts.loc[s,'completed']/1008:.2%}",f"{summary['measured'][f'S2/{s}']['all_14_modes_complete']}/72"] for s in ['O0','O1','O2']])
    synth_table=table(['生成律（combined/full）','求解器','r NRMSE','HbO NRMSE','HbR NRMSE'],[
        [row.law,row.solver,f'{row.r_nrmse:.4f}',f'{row.clean_HbO_nrmse:.4f}',f'{row.clean_HbR_nrmse:.4f}']
        for row in synthetic[(synthetic.variant=='combined') & (synthetic['mode']=='full') & synthetic.solver.isin(['O0','O1','O2'])].itertuples()])
    smoke_table=table(['数据 / 记录','EEG 张量 / 单位','Hb 张量 / 单位','实测 support','参考'],[
        [r['join_key'].replace('|',' / '),str(r['shape']['eeg'])+' / '+r['unit']['eeg'],str(r['shape']['fnirs'])+' / '+r['unit']['fnirs'],
         'EEG 100%；Hb 100%',r['eeg_reference']] for r in smoke['rows']])
    report = f'''# 数据集对齐修复后的统一载入、模型输入与 SSM 拟合报告

版本：{out.name}；证据截止：2026-09-16；基于当前工作树的接口实现与已经结束的 measurement alignment v3 运行。

## 1. 主要结论与阅读范围

**本轮完成的是测量合同与输入工程修复；尚未证明真实 EEG–fNIRS 的 SSM 能稳定、合格地恢复生理状态，也没有新的 tokenizer 训练结果。** 已明确单位证据、光学输入阶段、float64 边界、事件和真实支持、成对幅度及噪声变换。SSM 的 48 份训练投影现在全部可重组，历史为 39/48；同一合成面板保持 3600/3600 完成且汇总数值不变。实测 3024 次均已执行，但只有 1552 次通过求解与生理路径检查；744 次生理域失败、728 次数值失败。三个求解器的完整 72 身份主指标仍不可定义。

“一致性”在这里有四层含义：数组里的量与单位对应；相同样本身份和时钟正确相连；均值、噪声、mask 和评分使用同一数学变换；训练统计不读取评价支持。它不等于所有数据被压成相同分布，也不等于跨设备物理标定已经完成。

本文承接 {link(previous/'REPORT.md','2026-09-15 数据缩放报告')} 的修改清单，但不重写它的历史证据。原报告的 18-trial、216-fit 点式缩放诊断与本轮 72-trial、14-mode、三求解器时间观测实验不同，二者成功率不能直接横比。本报告的新旧实测对照使用同一 v3 主协议的历史续跑，所有比较均显式保留分母。

### 1.1 三条数据路径必须分开理解

{figure('01_routes')}

| 路径 | 真正输出 | 已经验证到哪一步 |
| --- | --- | --- |
| 通用 loader 的 legacy_robust 分支 | 连续记录清理后逐通道 median/MAD；float32；数值单位为 robust SD | 保留为历史接口；默认参数尚未统一切换。旧 v1 缓存由当前 reader 显式拒绝，不会自动升级 |
| 通用 loader 的 measurement 分支 | EEG 200 Hz、Hb 10 Hz；有证据的 µV/µM 或分开的相对组；float64；不做逐通道 MAD | 三个公开数据记录的真实加载 smoke；接口与单位/mask 合同测试；不是全数据重建 |
| 本轮实际 SSM | 单独的有界原生窗口入口；30 s → 120 × 3，4 Hz；宽带 EEG 功率投影 + 单对 HbO/HbR | 固定 Single-Trial 72 身份的 S1/S2 已执行；保留完整失败记录 |
| tokenizer 测量接口 | 在 measurement 窗口上取 6 EEG + 一对 Hb，保留单位、位置、mask 和 float64 | 已接入现有 factory/local view 并做针对性测试；没有训练运行、准确率或 teacher 目标 |

不能把“新增 measurement 参数”写成“所有旧配置已经迁移”，也不能把 generic loader 的全记录清理写成本轮 SSM 的裁窗处理。当前接口实现和冻结 SSM 源码可能有不同提交时点；SSM 的数值结果以该 run 的 source_snapshot 为准。

## 2. 原始数据是什么，统一 loader 实际改了什么

### 2.1 单位证据：已知的转换，未知的显式保留

| 数据集 | 发布数组的 EEG | 发布数组的 fNIRS | 修复后的测量坐标及证据边界 |
| --- | --- | --- | --- |
| Single-Trial | MATLAB 分析视图 200 Hz、30 scalp 通道；读取 yUnit；当前证据为 µV；linked-mastoid reference | 10 Hz、36 对 760/850 nm，yUnit=V；它是光电探测电压 | EEG µV；光强 → 自然对数 OD → 近似 MBLL 相对 Hb。光电 V 不作为 EEG V 转换；真实数据仍不是经验证的绝对 µM |
| Simultaneous | MATLAB 200 Hz、28 scalp + EOG 辅助通道；yUnit 为 µV；TP9 reference | 已发布 36 对 oxy/deoxy，10 Hz，yUnit=mmol/L | EEG µV；Hb 数值 ×1000 转 µM；已是色团，不再执行 intensity→OD→MBLL |
| Visual | EDF 500 Hz；逐通道 physical/digital extrema 与单位字段决定校准；当前记录为 µV；采集参考尚不明 | ETG-7100 10 Hz Oxy/Deoxy CSV；已是色团语义，未找到发布数值单位证据 | EEG 解码为 µV 后重采样；Hb 保留 relative:visual_cognitive_motivation，保留 Part/Probe 与 CSV Time |
| REFED | MAT 64 通道、1000 Hz；AFz reference；原说明及作者代码未证实数组电位单位 | LABNIRS 47.62 Hz；HbO/HbR/HbT/Abs780/Abs805/Abs830 六种类型 | 去掉硬编码 V，不擅自 ×10⁶；EEG 保留 unknown_REFED_EEG_export 相对组；Hb 保留 relative:refed；当前共同分支只选 HbO/HbR |

此前缺单位证据的 **REFED EEG、REFED Hb、Visual Hb 三项仍未解决**，已标记 unknown/relative。原论文的设备型号、concentration/raw 措辞或典型幅值不能证明具体发布数组的单位；浓度×光程也不能擅自当浓度。Single-Trial 和 Simultaneous 的单位证据来自数组头字段，字段缺失时已取消默认 µV；Visual 保留每通道 EDF 解码来源。原始采集率和发布分析率分开：不能把 Single-Trial/Simultaneous 的采集 1000 Hz EEG 说成当前 reader 读取 1000 Hz。

此次报告引用 {link(ROOT/'docs/DATASETS_DESCRIPTION.md','单位证据复核 owner')} 中已经执行的原说明、原论文和作者读取代码核查；没有把新的无证据推断补成已知单位。

### 2.2 输入阶段、精度和光学转换

统一入口显式区分 intensity、OD、已发布色团和吸光度波段。Single-Trial 需要从强度比形成 OD；Simultaneous、Visual 和 REFED 的 HbO/HbR 分支不能再次 MBLL。REFED 的原生 HbT 是另一个导出量，只有同单位、同基线、同处理条件成立时，才有资格比较其与 HbO+HbR 的关系。

通用测量路径沿用有身份的清理分支：EEG 经过对应分支的 EOG/伪迹处理、1–45 Hz 带通并统一到 200 Hz；Hb 保留导数稳健处理与 0.01–0.2 Hz 带通，统一到 10 Hz。REFED 的 47.62→10 Hz 和 EEG 的 1000→200 Hz、Visual EEG 的 500→200 Hz 都是显式重采样。发布色团此前的作者处理不能被逆转或当成未处理原始光强；本轮新增清理步骤与上游来源分别记录。measurement 分支跳过旧逐通道 median/MAD，但没有跳过滤波、时钟核查和支持处理。

新 producer 以 float64 保留原生/测量值和处理输出，并记录 processing_schema。reader 在进入 measurement 分支前核查新生产版本、float64 和 processed support；它不会把已舍入的旧 float32 缓存转成 float64 后宣称恢复了精度。缓存结构版本、事件版本和处理版本是不同层：当前结构为 clean_eeg_fnirs_cache_v2，事件为 physiology_event_alignment_v2，新处理为 physiology_measurement_alignment_v3，测量窗口为 unified_physiology_measurement_window_v3。

独立 MBLL 正例使用 MNE {mbll['mne_version']} 的 Prahl 系数、760/850 nm、3 cm、PPF=6 和明确的自然对数系数约定。与独立 MNE 输出最大差 {mbll['maximum_error_uM']:.4g} µM，与已知合成浓度最大差 {mbll['maximum_truth_error_uM']:.4g} µM；同时记录 MNE 的 2.303 舍入因子与 ln(10) 的差别。**这证明指定参数下的变换闭合，不证明 Single-Trial 实际光程和系数已经被该正例标定。** 真实 SSM 仍沿用明确标记的近似相对 Hb 坐标，以隔离工程修复影响。

### 2.3 时间、身份、参考与空间位置

1. 事件配对依据原始身份和标签，不依赖数组顺序。单个漏标记只有在匹配唯一时修复；多重不匹配、非单调时间、标签冲突或歧义直接拒绝，不静默截短。
2. 时钟漂移在 offset segment 内以真实毫秒拟合；跨拼接跳变的全局斜率只作诊断。无法证实的拼接边界邻域不作为有效窗口支持。
3. 输出同时保留请求的事件锚点、实际网格起点、取样索引、舍入误差和模态各自时钟。仪器时间偏移不等于脑血流生理滞后；修正前者不能任意消除后者。
4. Visual 的 Oxy/Deoxy 时间与 marker 必须一致，先按 CSV Time 规则化再滤波；不删除坏行后挤压时轴。REFED 保留真实发布的 1 Hz 连续情绪标注，不把它拉伸到四舍五入后的 Hb 时长；硬件 offset/jitter 未知仍是未知。
5. EEG 参考、EOG 辅助信号和 Hb 配对名可追溯。模板坐标可用于邻近选择，不能作为个体精确共定位证据；没有通过通道镜像或复制填补位置缺失。

三条公开 smoke 的时间差例子说明为什么不能只按同一数组下标切片：Simultaneous 首事件 EEG 30107 ms、Hb 43403 ms，相差 13296 ms；实际 200/10 Hz 网格起点为 30105/43400 ms，分别有 −2/−3 ms 舍入。Visual 对应 2032/24500 ms，实际为 2030/24500 ms。REFED 的共同零点是发布方共享起点假设，不能据返回 offset=0 推断真实硬件零延迟。

### 2.4 缺失、补值、伪迹与幅度尾部

measurement 分支同时记录原生有限值支持和处理后真实支持。用于保持算子可计算的插值/零填充不变成新观测。双向 IIR 有整段依赖，只要补值影响处理通道，就保守地不把其输出当真实支持；EOG 回归耦合到多个 EEG 通道，辅助 EOG 缺失也会影响对应有效性。这是严格支持规则，可能减少可用数据；本次三个 smoke 所取窗口均为完整有限支持，没有由此被剔除。

伪迹注记、bad-channel 判定、真实缺失和 padding 各有语义。缓存 artifact mask 默认只是审计信息，不会自动把全部伪迹标记点改成无效测量。后续损失必须使用与其张量对应的 mask。原生缺失须在功率/OD 等非线性之前施加；本轮 14 模式的 feature-missing 是另一项干预，不能声称已经模拟全部原生传感器缺失。

Visual S02/Probe1 的既有异常窗口另作定位：299.2–329.2 s 的 CH6_HbO 有 300 个真实样本，SD={qc['channel_6']['sd']:.4f}，其余通道 SD 中位数={qc['median_other_channel_sd']:.4f}，相差 {qc['channel_6']['sd']/qc['median_other_channel_sd']:.2f} 倍；最大幅值 {qc['channel_6']['max_abs']:.4f} 出现在约 321.297 s。设备 ExceptionCh6=1，DigitalGain 为 163/163，BodyMovement 有标记，但这些注记本身不证明全部变化都是可删除伪迹。最大的 1% 样本只贡献约 4.01% 的中心化平方和，不能把这段幅度问题简化为几个尖峰。当前保留并注记，没有任意 clip、强制剔除或通过逐通道 MAD 隐藏它；独立稳健损失候选尚未完成实测比较。

### 2.5 本轮真实重处理范围

{smoke_table}

表中是各公开数据一个原始记录、各取一个 20 s 窗口的加载结果，两个模态均为 float64。索引在这些记录下分别列出 300、125、1 个事件窗口，不代表这些窗口都逐个完成 smoke。公开 cache 为 physiology_semantic_clean_v3_public_smoke_v2；第一次 record_limit smoke 曾暴露信号与事件 builder 的任务排序不一致，已统一排序并保留首次失败证据。

Single-Trial 的新 native-only cache 为 physiology_semantic_clean_v3_ssm_native，保存 3 被试 × 3 session 的 9 个原生记录与 180 条事件元数据；实际 SSM 只在已限定的 72 个 MA 窗口内做预处理。原始 MA 位置 4、9 未进入拟合/预处理，subjects 24–29 仍关闭。缓存保存会话数组不能被解释为对整个会话做了清理或对所有事件做了评分。没有执行四个数据集所有被试的全量缓存重建。

## 3. 原始数据到本轮 SSM：每一步改变了什么

### 3.1 实际张量与变换次序

| 层 | EEG | fNIRS | 保留 / 改变的语义 |
| --- | --- | --- | --- |
| 原生窗口 | 30 s × 200 Hz × 30 scalp，即 6000 × 30；单位有证据为 µV | 30 s × 10 Hz × 36 对 × 2 波长，即 300 × 36 × 2；探测电压 V | 先按精确允许身份裁窗，再做统计与非线性；原始文件保持不变 |
| 非线性特征 | 1–45 Hz 后每 250 ms 计算功率；log(max(P,10⁻¹² µV²)/(1 µV²))；120 × 30 | ln(I_ref/I)，TDDR-like 导数稳健处理，保留 300 × 36 × 2 的处理后 OD | EEG 不再是电位，也不是瞬时频带相位；OD 是相对于参考的光强变化 |
| 线性时间处理 | 功率时钟 4 Hz；fold 中拟合 EEG 中心、尺度、PCA 方向 | 固定近似 MBLL → 10 Hz 上 0.01–0.2 Hz 带通 → 10→4 Hz 重采样 | 顺序与有效观测算子完全一致；不可把 MBLL 与 filter 的不同舍入边界混用 |
| 选择与基线 | 30 维宽带 log-power 投到一维；事件前 20 输出点去均值 | 36 对中按训练 smoothness 指标选一对；HbO/HbR 使用共同尺度；相同基线区间 | 不是六通道局部 tokenizer 视图；PCA 与所选 Hb 对未证明对应同一个脑源 |
| SSM 观测 | EEG 投影为第 1 列 | HbO/HbR 为第 2/3 列 | y 为 float64 [120,3]，4 Hz；单位是冻结的模型观测坐标，不是 µV/绝对 µM |

SSM 的 EEG 功率下限在统一电位单位后定义：同一信号用 V 或 µV 表达时，变换结果最大差约 8.74×10⁻¹⁴；不再把相同字面 floor 错用于 V² 和 µV²。原生 EEG 的相位、逐通道高频细节和幅度单位在 log-power/PCA 后已经被有意改变，必须通过特征定义和变换来源解释，不能称“原始 EEG 无损送入 SSM”。

{figure('02_time_support')}

### 3.2 训练尺度与模型幅度桥：已分开记录，但桥还在

实际投影先在 fold 训练数据中估计 EEG 每通道 median/MAD 和 PCA；PCA 符号约定为 loading 总和为正，该约定不建立神经驱动的生理正负方向。Hb 配对选择依据训练的信号 MAD / 一阶差分 MAD，趋向平滑、变化稳定的测量位置，不能解释为已定位脑皮层源。

投影输出可写成：EEG_model = C(PCA(log-power)) × g_E / s_E；Hb_model = C(Hb_selected) × g_H / s_H。C 是事件前基线去均值算子；s_E、s_H 为训练数据拟合的数值尺度；g_E、g_H 来自被冻结的 reference_observation_gauge。代码现在分别序列化 computational_scale、measurement_scale、observation_loading 和逆变换，而不是只留下一个不透明因子。

固定 subject_01 / outer fold 0 的实际例子：s_E={p0['projection']['measurement_scale']['eeg']:.6f}、g_E={p0['projection']['observation_loading']['eeg']:.6f}，EEG 最终乘数为 {p0['projection']['eeg_factor']:.6f}；s_H={p0['projection']['measurement_scale']['fnirs_common']:.8f}、g_H={p0['projection']['observation_loading']['fnirs_common']:.6f}，成对 Hb 共同乘数为 {p0['projection']['fnirs_factor']:.6f}。该 fold 的三个评分 SD 为 {p0['normalization_sd'][0]:.6f}、{p0['normalization_sd'][1]:.6f}、{p0['normalization_sd'][2]:.6f}；这是训练评分对象，不是对 HbO/HbR 各自再做输入缩放。逆因子只恢复所选特征的尺度，不能逆转 PCA 降维、滤波或功率非线性。

**本轮仍保留先验参考幅度桥 g，并未证明已经消除模型增益与潜状态尺度的混淆。** 同一训练投影对象被后续候选复用，不随 Q/参数候选重新缩放数据。HbO/HbR 共用一个正数，不单独把两种色团各自压成单位方差。独立的 paired adapter v2 也已实现：要求训练记录身份、显式 baseline、有效支持和共同 pooled MAD，可序列化与逆变换；但本轮 SSM 没有用它替换旧统计候选。

共同正尺度保持相同基线后的 HbO/HbR 幅度比例和 HbO+HbR 的线性加和关系；它不能把未知单位变成已知单位，也不能恢复已经被滤波、基线去除或通道选择丢弃的信息。成对关系的“保留”有这些明确前提，不等于所有原始生理信息无损。

### 3.3 baseline、有效观测与协方差

本轮 30 s 窗从 MA 事件前 5 s 开始，baseline 标记为 pre_event_reference_not_latent_rest；20 个输出点共享真实支持。基线是参考区间，不是已知静息潜状态。通用 measurement_baseline 函数要求调用方提供区间、作用和证据；支持不足则返回 insufficient_support，不自动把任意记录开头 5 s 当静息。

可将基线算子写为 C = I − 1wᵀ，其中 w 在参考区间权重和为 1。对线性处理 L，数据、预测均值和雅可比同步左乘 L；噪声必须传播为 L R Lᵀ。若 L 包括基线、滤波、重采样，协方差一般不再逐点独立。EEG 特征原生时钟为 4 Hz，光学特征为 10 Hz；有效算子将其映射到相同 4 Hz 输出时钟，并携带 mask 的观测/目标交叉协方差。

完整观测的 120×3=360 个坐标在当前 rank_rtol=10⁻¹⁰ 下保留 357 维，符合三个基线线性约束。工程审计同时保留阈值敏感性：更宽的 10⁻⁸ 阈值下 full rank 为 346，10⁻¹² 时为 357；不能把数值秩绝对不变当成已通过结论。固定阈值内的单位密度校正、雅可比和条件噪声一致性已通过；O2 仍没有区间估计或边际似然。

30 s × 0.01 Hz = 0.3 个周期，短窗低频滤波受边界影响，不能据此声称可靠测出了 0.01 Hz 周期。双向滤波和全窗平滑使用未来点；本报告所有此类结果都是离线诊断，不作为因果或实时前瞻预测。

### 3.4 噪声的证据与工程复核

{figure('03_precision_noise')}

48 份投影全部满足 10⁻⁶ 个训练 SD 的重组容差，最大误差为 {errors.max():.6g}；旧版只有 39 份可用。39 份旧新均有记录的投影保留同一 Hb 对，冻结评分 SD 最大相对变化为 {max(sd_changes):.6g}（{max(sd_changes)*100:.6f}%）。评分 SD 在各自 fold/candidate 比较内冻结，但新旧不是逐位相同对象，不能直接宣称跨 run 风险计算完全同坐标。

独立工程检查：观测 gain 雅可比最大绝对误差约 3.91×10⁻¹¹；O2 线性均值误差 8.92×10⁻¹³、导数误差 6.64×10⁻¹²、密度单位误差 1.42×10⁻¹⁴；14 模式隐藏值干预误差为 0；同噪声条件预测一致性误差约 6.11×10⁻¹³。新的 float64 特征重组正例误差为 0。旧 float32 边界的 4.31×10⁻⁹ 绝对误差另行保留，未被新结果覆盖。

噪声清单现在同时记录下限前估计、下限来源、最终值和触发标记。合成 Student-t 的 scale=[0.08,0.025,0.015]、ν=5，其边际 SD 为 scale×sqrt(5/3)=[0.10328,0.03227,0.01936]，两者不再混称。同一噪声量的单位和所在特征层也显式保存。EEG 的 48/48 最终值都由合成下限控制；光学特征 91/96 触发下限，下限前 OD 估计范围 {noise_od.min():.6f}–{noise_od.max():.6f}。这些训练差分 MAD 仍是特征噪声近似，不能当成原始硬件噪声标定；本轮没有重新调低下限来制造更好的拟合曲线。

## 4. tokenizer 现在能够收到什么

### 4.1 已接入的实际输入接口

factory 的 data.output_coordinate="measurement" 已传入现有 UnifiedPhysiologyLocalViewDataset。对常见 20 s 窗，先从统一测量窗口中选一对有效 HbO/HbR，再按可比较的几何选 6 个邻近 EEG；Hb 对由稳定身份哈希确定，不按当前样本的拟合好坏选择。它不同于 SSM 的宽带 PCA 与训练 smoothness 选 Hb 对。

| 字段 / 张量 | 当前 measurement local view |
| --- | --- |
| eeg | float64 [6,4000]，200 Hz；是清理后的电位或未知单位相对测量，不是本轮 SSM 的一维 log-power |
| fnirs | float64 [2,200]，10 Hz，顺序固定 HbO/HbR；µM 或所属相对组 |
| patch 网格 | 每 2 s：EEG 400 点，Hb 20 点；20 s 共 10 个 token 位置 |
| token_valid_mask | 所选通道与时间点真实支持合取；填零不当成有效数据；任一必要点缺失时该 patch 不能冒充完整支持 |
| channel_valid_mask | 保留所选 EEG/Hb 的逐点真实支持，与 token 级汇总分开 |
| unit / selected channels | 保留单位组、六个 EEG 名、HbO/HbR 名、anchor 和稳定 sample_id/dependency_group_id |
| measurement_metadata_json | 序列化预处理状态、参考与几何证据；支持默认 batch collate，不依赖 None/变长字典被错误拼接 |
| teacher_target_status | not_generated_no_qualification；measurement 坐标拒绝旧 teacher sidecar 连接 |

该接口保留 float64 到消费者边界，消费者若为神经网络选择 float32，需要把数值变换与计算 dtype 作为明确的训练配置决定。接口不会暗中 fit 成对 adapter，也不会在每个 crop 上重新估计尺度。paired adapter 的 fit/apply/inverse 已有显式训练身份与 baseline 合同，实际部署比较仍需单独定义。当前没有新的训练 launcher，也没有训练 loss、分类准确率、跨数据集泛化或 token 生理解释的实测增益。

### 4.2 哪些语义仍在，哪些已经改变

| 信息 | loader 测量层 | 实际 SSM | tokenizer local view |
| --- | --- | --- | --- |
| 物理单位与源头 | 已证实的单位换算可逆，未知独立保留 | 通过 manifest/投影可追溯；观测本身是模型坐标 | 单位与处理状态跟随张量 |
| HbO/HbR 相对幅度 | 没有新增逐色团 MAD 抹平幅度 | 相同 baseline 区间和共同正尺度保留已选对的关系 | 保留选定 Hb 对原测量关系；后续训练尺度尚未自动部署 |
| 绝对 DC / 静息浓度 | 已发布 Hb 本来就是变化量，滤波进一步移除慢趋势 | baseline 明确去除均值；不能恢复绝对浓度或静息潜态 | 承接测量预处理，不创造被移除的 DC |
| EEG 相位与局部空间 | 滤波/清理有明确带宽与分支 | 功率/PCA 压缩丢失相位和逐通道信息 | 保留六通道时序，但只选局部覆盖 |
| 时间/缺失/任务 | 锚点、真实 support、标注范式可追溯 | 4 Hz 特征时钟；feature-missing 单独定义 | 200/10 Hz 后 patch 化；patch mask 保留真实支持 |
| 精确脑源与跨设备标定 | 参考、模板与未知项保留 | 未证明 EEG PCA 和所选 Hb 对是同一脑源 | 邻近模板不等于个体共定位；没有物理 teacher 资格 |

因此，“不丢失生理语义”的可执行含义是保留类型、单位、时空身份、变换与支持的解释，不把不同量混成同一个名字；它不是承诺滤波和特征压缩完全无损。原始数据和历史缓存仍保留，重要压缩/不可逆步骤在表中明确。

## 5. 目前 SSM 拟合结果

### 5.1 实验范围、求解器和指标

本轮只运行 S1 与 S2。S1 为 3 种生成律 × 2 种时间处理 variant × 5 种 mask × 5 个求解器 × 24 seeds，共 3600 次；S2 为 subject_01/09/18、session_01/03/05、每 session 8 个允许 MA，共 72 个身份 × 14 模式 × O0/O1/O2，共 3024 次。训练/评价 fold、原始位置排除、噪声规则与求解预算均由冻结配置约束。S3 gain 和 S4 Q 网格未在本轮执行；通用 summary 中的 S3/S4 空模板不是本轮漏跑结果。

O0 是原点式 Student-t 参考；O1 是含完整时间观测/噪声的线性 Gaussian 参考；O2 是非线性 MAP，使用相同有效观测合同但可能在有限预算下失败。O1/O2 正式缺失恢复评分使用 conditional noisy-target prediction，clean-map residual 另算；两者不混称同一预测。合成 NRMSE 有已知干净状态/观测真值，真实数据 NRMSE 则是对观测目标的恢复误差，不能作为潜状态真值误差。

实测每个 trial 先评分，然后 session 等权，再 subject 等权。中心隐藏段为 16 个 4 Hz 点（4 s）。每个 trial 的 B = 0.5×NMSE_EEG + 0.25×NMSE_HbO + 0.25×NMSE_HbR，其中 EEG 项来自 center_EEG 的隐藏段，Hb 项来自 center_fNIRS 的隐藏段，NMSE = MSE / 冻结训练 SD²。B 不是潜状态真值误差；完整主指标需要所有要求模式和身份都有有效结果。controller cell completed 只表示任务已结束并写出结果，不能替代每一条 trajectory 的 completed 状态。

### 5.2 合成结果：通过工程预检，但没有新的精度增益

{figure('04_synthetic')}

{synth_table}

3600/3600 全部完成；新旧 150 条条件汇总按 law/variant/mode/solver 对齐后逐值完全相同。冻结 Gaussian mean precheck 的 24/24 例通过，r 和 clean EEG NRMSE=0.48429，HbO=0.14862，HbR=0.16291，r correlation=0.87098；门槛为 NRMSE≤0.65、correlation≥0.8。这个预检是均值/工程检查，不是 Student-t 校准或真实数据 teacher 资格。

图中完整时间算子相对点式/仅均值近似的优势仍存在，说明工程修复没有破坏已有合成结果。它不能解决真实数据的观测增益、参考方式、空间对应、残余伪迹和动力学失配。O2 未估计可信区间；不得把点估计成功写成后验不确定性已校准。

### 5.3 实测完整分母与失败分布

{figure('05_outcomes')}

{status_table}

总计 1552/3024（51.32%）通过求解及路径检查，744/3024（24.60%）生理域失败，728/3024（24.07%）数值失败。三个 solver 都未达到完整 72/72 的 14 模式支持，因此 full_panel_B 全部为空。O0 在 27 个全模式成功身份上的 subset B=5.2268，O1 仅在 2 个身份上的 subset B=0.8311，O2 无此子集；这些数字不能横比成“哪个方法总体更好”。旧 O0 子集为 21 个，B=3.6479；新旧样本组成变化，也不能据 5.2268 对 3.6479 宣称性能恶化。

{figure('06_modes')}

full 联合模式 O0=71/72、O1=8/72、O2=0/72。O2 全部 237 次通过中，216 次来自 EEG_only、all_missing、center_EEG_own；其余为 center_EEG_template 的 1 次和 center_EEG_shift 的 20 次。O0 的主要失败集中在 center_EEG_shift（仅 28/72 通过），说明固定错时对照会触发生理路径问题；不能删除该模式后重新宣称完整面板合格。

### 5.4 同一身份的新旧比较

{figure('07_paired_changes')}

历史主协议 3024 次计划中只有 2268 次真正执行，756 次因输入依赖未执行；本轮把这些依赖补齐。新增 756 次包含 380 次通过、183 次生理域失败、193 次数值失败。这是新增可审计证据，不能与旧 run 的缺失行直接配对。

共同 2268 次中，O0 的 721 次通过和 35 次生理域失败完全保持；O1 的 273 次通过和 483 次生理域失败完全保持；O2 从 177/43/536（通过/生理域/数值失败）变为 178/43/535。两个恢复与一个退化均为 center_EEG_shift：subject_09/session_01/event14、subject_18/session_01/event1 恢复；subject_09/session_05/event4 退化。没有删除这一例退化或把它按较低目标函数值改记成功。

该退化例历史 rest 初始化在 168 次 evaluation 收敛，而本轮用满 200 次预算仍未达停止规则；新目标函数约 26464，历史约 55748，较低的目标值也不自动意味着收敛或合格。恢复例分别从历史 200 次预算耗尽变为 176/184 次收敛。微小数值变化能改变有限预算优化路径，本轮证据是“合成不变、实测共同身份多数不变且有双向翻转”，不是“已保证任何后续 SSM 都不受影响”。

### 5.5 固定实例与失败原因

{figure('08_fixed_example')}

固定首例 O0 虽通过路径检查，但 HbR NRMSE=4.3480，HbR 平均偏差为 +3.7881 个训练 SD；成功状态不保证好拟合。同一身份 O1 的状态数值有限、f/v/p/q 为正，仍因 absolute_hb_nonnegative 和 hbr_not_above_hbt 两项为 false 被判为生理域失败。O2 数值失败，没有有效曲线。这里保留失败曲线的明确标签，用来解释失败，不作为合格的恢复结果。

失败分为输入依赖、物理路径和求解预算/数值三个层次。本轮输入重组问题已解除；O1 的线性近似均值可能越出生理域，O2 的非线性路径优化仍受条件数、初始化、预算和模型失配影响。现有结果不支持将所有失败单因归结为共同 Hb 缩放，更不支持放宽生理边界、扩大 Q/gain 网格来遮盖负结果。

## 6. 与原报告设计清单逐项对照

| 原修改点 | 当前已经落地与核验 | 尚不能宣称完成的部分 |
| --- | --- | --- |
| P0 单位证据 | 证据字段、无证据默认值移除、已知 µV/µM 等价性 | REFED EEG/Hb、Visual Hb 仍未知；Single-Trial 仍相对 Hb |
| P0 输入阶段 | intensity/OD/已发布 Hb/Abs 分流；独立 MBLL 已知浓度正例 | 没有实际个体光程的物理标定 |
| P0 精度 | 新原生/特征 float64；算子次序一致；48/48 重组 | 旧缓存历史精度不能恢复；未全量迁移旧配置 |
| P0 EEG floor | µV² 与参考功率明确；V/µV 等价检查 | 未知单位 EEG 不能套用已标定物理阈值 |
| P0 观测同步 | 均值/雅可比/R/基线/密度与评分合同检查 | O2 后验区间和边际似然仍未估计 |
| P0 噪声证据 | floor 前后值、触发率、Student scale/SD 区分 | 原始传感器噪声未独立标定；floor 仍占主导 |
| P1 成对幅度 | 共同正尺度、显式 baseline/训练身份/逆变换接口 | 新 paired adapter 未作为实测统计候选部署验证 |
| P1 baseline | 有来源的事件前区间、真实共同支持、算子传播 | 不等于已知静息潜态；未知范式无自动静息推断 |
| P1 时间处理 | 漂移分段、原始 marker 身份、真实网格、完整时间协方差 | 短窗慢周期与因果解释不成立；全记录路径仍离线 |
| P1 质量尾部 | 缺失与注记分离；CH6 位置、幅值、设备注记已核查 | 未完成独立稳健损失比较；未擅自剪裁 |
| P1 空间参考 | 名称/参考/几何来源保留，坏通道与坐标兼容检查 | PCA 与 Hb 配对同源未证明，模板非个体共定位 |
| P1 幅度桥 | 数值尺度与 reference loading 分开记录并冻结 | 先验参考桥仍在，不能宣称完成绝对增益辨识 |
| P1 缺失 | 非线性前原生隐藏检查；14-mode 特征隐藏独立声明 | 特征缺失结果不等于全部原生缺失/未来预测资格 |
| P2 求解动态 | 失败分类、固定预算、driver replay 与导数检查 | 728 次数值失败与 744 次生理域失败未解决；未扩 Q/gain |
| P2 tokenizer | measurement local v2 与 factory 接通；mask/单位/float64/teacher 拒绝 | 没有新训练、合格 teacher 或准确率结果 |
| P2 验收 | S1/S2 固定身份全执行、完整分母、旧新配对 | 完整实测主指标不可定义；不作跨数据集/默认流程推广 |

已实现、已运行、统计候选通过和可默认推广是不同状态。当前宜先在既有开发支持上独立定位噪声下限、观测映射/空间对应和 O2 收敛条件，再按完整面板验证；跨数据集 SSM 与 tokenizer 训练需要各自的具体配置和结果，不能由此次工程通过自动推导。

## 7. 证据、复现与交付检查

### 7.1 唯一事实来源与可追溯文件

- 数据原始事实与原论文入口：{link(ROOT/'docs/DATASETS_DESCRIPTION.md','DATASETS_DESCRIPTION.md')}；运行边界与字段合同：{link(ROOT/'docs/DATA_CONTRACT.md','DATA_CONTRACT.md')}。
- 当前 loader 与 consumer：{link(ROOT/'src/data/unified_physiology.py','unified_physiology.py')}、{link(ROOT/'src/data/physiology_measurement_adapter.py','physiology_measurement_adapter.py')}、{link(ROOT/'src/data/physiology_semantic_local.py','physiology_semantic_local.py')} 与 {link(ROOT/'src/data/factory.py','factory.py')}；本报告保存读取时的源码快照。
- 实测/合成正式结论：{link(run/'summary.json','本轮 summary')}、{link(run/'resolved_config.yaml','冻结配置')}、{link(run/'source_snapshot_identity.json','运行源码身份')}、{link(run/'scope_inventory.json','精确样本身份')}、{link(run/'S2/trial_metrics.csv','逐拟合实测表')}、{link(run/'S1/solver_comparison.csv','合成条件表')}。本报告只是解释和绘图，不替代它们的事实源地位。
- 历史比较：{link(old/'summary.json','历史 v3 summary')}、{link(run/'measurement_comparison.json','同身份状态比较')}、{link(run/'measurement_preparation_audit.json','48 份投影精度审计')}。
- 工程与公开记录：{link(audit/'engineering_checks.json','工程数值检查')}、{link(audit/'mbll_independent_check.json','独立 MBLL')}、{link(audit/'visual_ch6_quality.json','CH6 质量定位')}、{link(audit/'public_loader_smoke.json','公开 loader smoke')}、{link(audit/'public_loader_smoke_failed_v1.json','保留的首次失败 smoke')}。
- 本导出附带 preparation_audit.csv、fit_status_counts.csv、paired_status_transitions.csv、mode_status.csv 与 synthetic_comparison.csv；均由上述 owner 派生，source_identity.json 给出读取来源，report_summary.json 为此报告的派生校验结果。

### 7.2 复现及检查边界

沿用既有 analyze_dataset_scaling.py 入口，新增 --stage alignment-report，并要求 --output-dir 指向 data_quality_audit 下新的空版本目录。该 stage 仅读取已保留证据，生成图表与报告；不执行原始数据加载、重建缓存、SSM 拟合或受保护评价，也不触发旧报告初始化。已有导出非空时拒绝覆盖。

本轮修复的留存测试记录包括 measurement 78 passed、integration 67 passed，另有针对性组检查；这些测试组有重叠，不能相加成独立测试总数。默认 collection 是 635 tests collected、11 deselected，表示收集检查，不表示全部 635 个测试都执行通过。报告生成器另做身份/分母校验、表格与 owner 一致性检查以及位图 PDF 输出测试。

REPORT.md、内嵌 PNG 的自包含 REPORT.html、REPORT.pdf 和 FIGURES.pdf 同时交付。图像统一为 250 dpi PNG，PDF 只栅格化整幅图，正文、表格和图注保持可选择/检索；源图不以 SVG 或矢量 PDF 页面嵌入。导出后逐页渲染检查、核对图像对象数，并检查页面标签与裁切；WPS 在本环境未检查。

**最终结论：数据单位、阶段、时间、支持和变换来源比修复前更明确；输入重组的工程障碍已解除，受控合成结果保持不变。但实测 SSM 联合拟合仍未通过完整验收，未知物理单位、模型幅度桥、噪声下限和空间对应等限制仍在。tokenizer 已具备可追溯测量输入接口，尚无训练与生理 teacher 资格结果。**
'''
    validation=export_report_document(report,out,captions,title='数据集对齐修复、模型输入与 SSM 拟合',date='2026-09-16',compact=True)
    validation.update(figures=len(captions),fit_denominators_checked=True,synthetic_exact_replay=True,
                      raw_array_reads=0,new_model_solves=0,protected_reads=0)
    write_json(out/'figure_captions.json',captions)
    live_paths = [Path(__file__),ROOT/'src/data/unified_physiology.py',ROOT/'src/data/physiology_measurement_adapter.py',
                  ROOT/'src/data/physiology_semantic_local.py',ROOT/'src/data/factory.py',ROOT/'src/data/event_alignment.py',
                  ROOT/'src/data/homer2_preprocessing.py',ROOT/'src/inference/observation_baselines.py',
                  ROOT/'experiments/evaluate_step5.py',ROOT/'experiments/evaluate_step5_observation_diagnostic.py',
                  ROOT/'experiments/evaluate_ssm_overnight_diagnostics.py',
                  ROOT/'docs/DATASETS_DESCRIPTION.md',ROOT/'docs/DATA_CONTRACT.md']
    for path in live_paths:
        dest=out/'source_snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    sources.update(live_paths)
    sources.update([previous/'REPORT.md',run/'resolved_config.yaml',run/'source_snapshot_identity.json'])
    write_json(out/'source_identity.json',dict(schema='dataset_alignment_report_v1',
        input_run=str(run.relative_to(ROOT)),reference_run=str(old.relative_to(ROOT)),
        note='live interface snapshot differs in time from frozen SSM source; retained runs are read-only',
        files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(sources)]))
    write_json(out/'report_summary.json',dict(source_run=str(run.relative_to(ROOT)),
        fit_status=counts.to_dict('index'),common_fits=len(common),transitions=transitions.to_dict('records'),
        fixed_example=dict(sample_id=ex0['sample_id'],O0_metrics=ex0['metrics'],
                           O1_physical_checks=ex1['physical_checks']),
        projections=len(prep),maximum_recomposition_training_sd=float(errors.max()),
        comparable_projections=len(sd_changes),maximum_scoring_sd_relative_change=max(sd_changes),
        synthetic_fits=3600,synthetic_summary_exactly_unchanged=True,
        all_modes_complete={s:summary['measured'][f'S2/{s}']['all_14_modes_complete'] for s in ['O0','O1','O2']},
        teacher_qualified=False,tokenizer_trained=False,default_promoted=False))
    write_json(out/'report_validation.json',validation)
    write_json(out/'report_completion.json',dict(stage='alignment-report',result=validation,source_sha256=sha(Path(__file__))))
    return validation


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['dry-run','synthetic','measurements','measured-ssm','report','alignment-report'],required=True)
    parser.add_argument('--output-dir',type=Path,help='new versioned alignment-report export directory')
    args=parser.parse_args()
    if args.stage=='alignment-report':
        if args.output_dir is None:parser.error('alignment-report requires --output-dir')
        print(json.dumps(jsonable(render_alignment_report(args.output_dir)),ensure_ascii=False),flush=True)
        return
    if args.output_dir is not None:parser.error('--output-dir applies only to alignment-report')
    c=config();out=init_run(c)
    if args.stage=='dry-run':dry_run(c,out);return
    if not (out/'inventory.json').exists():raise RuntimeError('dry-run required before execution')
    marker=out/(args.stage+'_completion.json')
    if marker.exists():raise RuntimeError('completed stage is immutable')
    if args.stage in ('measurements','measured-ssm'):
        if not (out/'synthetic_completion.json').exists():raise RuntimeError('synthetic stage required before measured access')
    if args.stage=='synthetic':result=synthetic(c,out)
    elif args.stage=='measurements':result=collect_measurements(c,out)
    elif args.stage=='measured-ssm':result=measured_ssm(c,out)
    else:result=render_report(c,out)
    write_json(marker,dict(stage=args.stage,result=result,source_sha256=sha(Path(__file__)),config_sha256=sha(CONFIG)))
    print(json.dumps(jsonable(result),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
