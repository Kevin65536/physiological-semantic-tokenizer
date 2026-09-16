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
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    body=markdown.markdown(report,extensions=['tables','fenced_code','toc'])
    css='''body{font-family:sans-serif;color:#20303f;font-size:11pt;line-height:1.6} h1{font-size:26pt;color:#163f59} h2{font-size:18pt;color:#1d526c;margin-top:30px} h3{font-size:13pt;color:#277da8} table{border-collapse:collapse;width:100%;font-size:8.5pt;margin:12px 0} th,td{border:1px solid #c9d5de;padding:5px;vertical-align:top} th{background:#eaf1f5} img{max-width:100%;height:auto} code{font-size:9pt;color:#5a456d} a{color:#236484} p{margin:8px 0}'''
    embedded=body
    for name in captions:
        data=base64.b64encode((out/'figures'/f'{name}.png').read_bytes()).decode()
        embedded=embedded.replace(f'src="figures/{name}.png"',f'src="data:image/png;base64,{data}"')
    (out/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>EEG-fNIRS 数据统一化诊断</title><style>'+css+'body{max-width:1180px;margin:40px auto;padding:0 30px}</style><body>'+embedded+'</body></html>',encoding='utf-8')
    # Give every scientific figure a bitmap page instead of letting HTML
    # shrink it to the space remaining at the bottom of a text page.
    doc=fitz.open()
    parts=re.split(r'(<p><img[^>]+></p>)',body)
    skip_caption=''
    for part in parts:
        if part.startswith('<p><img'):
            name=re.search(r'figures/([^"/]+)\.png',part).group(1)
            title,caption=captions[name]
            page,height=append_bitmap_figure(doc,out/'figures'/f'{name}.png')
            caption_html=f'<p><b>图 {int(name[:2])}｜{title}。</b> {caption}</p>'
            spare,scale=page.insert_htmlbox(fitz.Rect(42,60+height,553,800),caption_html,css=css,scale_low=.85)
            if spare<0:raise RuntimeError('figure caption did not fit')
            skip_caption=f'<p><strong>图 {int(name[:2])}'
        else:
            if skip_caption and part.lstrip().startswith(skip_caption):
                part=re.sub(r'^\s*<p>.*?</p>','',part,count=1,flags=re.S);skip_caption=''
            if not re.sub('<[^>]+>','',part).strip():continue
            buffer=io.BytesIO();writer=fitz.DocumentWriter(buffer)
            story=fitz.Story('<html><body>'+part+'</body></html>',user_css=css,archive=fitz.Archive(str(out)))
            story.write(writer,lambda n,filled:(fitz.Rect(0,0,595,842),fitz.Rect(42,42,553,800),None));writer.close()
            textpdf=fitz.open(stream=buffer.getvalue(),filetype='pdf');doc.insert_pdf(textpdf);textpdf.close()
    page_count=len(doc)
    for i,page in enumerate(doc):
        page.insert_text((42,820),f'EEG-fNIRS | 2026-09-15 | {i+1} / {page_count}',fontsize=8,color=(.4,.45,.5))
    doc.save(out/'REPORT.pdf',garbage=4,deflate=True,use_objstms=1)
    pdf_text=''.join(p.get_text() for p in doc)
    if '最终结论' not in pdf_text or len(pdf_text)<8000:raise RuntimeError('PDF content missing or truncated')
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
    validation=dict(figures=len(captions),pdf_pages=page_count,pdf_text_characters=len(pdf_text),
        raw_windows=len(windows),statistics_rows=len(stats),synthetic_rows=len(syn),measured_rows=len(mea),
        figures_embedded_in_html=embedded.count('data:image/png;base64,')==len(captions),
        expected_counts_pass=len(windows)==194 and len(syn)==288 and len(mea)==216,
        all_figures_exist=all((out/'figures'/f'{n}.png').exists() for n in captions),
        pdf_figures='separate PNG figure pages',pdf_bytes=(out/'REPORT.pdf').stat().st_size,
        pdf_image_count=report_images,atlas_image_count=atlas_images,wps_checked=False,
        pdf_all_pages_rendered=True,pdf_render_warnings=0,
        unique_subject_record_count=int(counts['records'].sum()))
    write_json(out/'report_validation.json',validation)
    write_json(out/'figure_captions.json',captions)
    return validation


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['dry-run','synthetic','measurements','measured-ssm','report'],required=True)
    args=parser.parse_args();c=config();out=init_run(c)
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
