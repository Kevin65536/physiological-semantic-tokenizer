#!/usr/bin/env python3
"""Read-only, identity-bounded optical preprocessing comparison; never fits an SSM."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import importlib
import json
import os
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
from scipy.signal import resample_poly
import yaml
from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
from src.data.event_alignment import ADMISSIBLE_ALIGNMENT_CASES, window_within_alignment_support
from src.data.homer2_preprocessing import (
    intensity_to_optical_density, robust_derivative_motion_suppression,
    modified_beer_lambert, bandpass_fnirs,
)

DEFAULT = ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_optical_motion_audit_v1.yaml'
SUBJECTS = ['subject_01','subject_09','subject_18']
SESSIONS = ['session_01','session_03','session_05']
POSITIONS = [0,1,2,3,5,6,7,8]
CACHE = 'data/cache/physiology_semantic_clean_v3_ssm_native'
RETAINED = 'experiments/runs/physiology_semantic_tokenizer/ssm_overnight/20260917_measurement_retest_v1/prepared'
PIPELINES = ['no_motion','current','mne_tddr']
KEYS = ['native_input_fnirs','native_channel_names']


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def read_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    exact=dict(schema='shared_driver_optical_motion_audit_v1',experiment_id='SSM-SHARED-DRIVER-OPTICAL-MOTION-AUDIT-v1',
        scope='read_only_public_optical_pipeline_audit_same_72_retained_windows_no_model_evaluation',
        cache_root=CACHE,retained_prepared=RETAINED,
        output_root='experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction',
        dataset_id='eeg_fnirs_single_trial',subjects=SUBJECTS,sessions=SESSIONS,original_positions=POSITIONS,
        allowed_array_keys=KEYS,pair_name='AF7Fp1',wavelengths_nm=[760.,850.],sampling_hz=10.,
        window_samples=300,window_offset_s=-5.,window_duration_s=30.,pipelines=PIPELINES,
        mne_version='1.11.0',processing_schema='physiology_measurement_alignment_v3',
        linear_filter=dict(low_hz=.01,high_hz=.2,order=3,resample_up=2,resample_down=5),
        reproduction=dict(absolute_tolerance=1e-12,relative_tolerance=1e-10,
            retained_keys=['feature_od','feature_fnirs','target_fnirs'],pair_index=0),
        synthetic=dict(seed=20260927,steps=1201,cases=['zero','constant','asymmetric_positive','asymmetric_negative','true_linear_trend','noise']),
        resources=dict(numerical_threads=1,max_workers=1),
        source_policy='published_subjects_01_to_29_public_policy_2026_09_17_scope_narrowed_to_exact_three_subjects',
        phase_rule='synthetic_terminal_before_measured_metadata_and_array_audit',
        stop_rule='retain_all_differences_and_block_pipeline_attribution_if_current_reproduction_fails')
    if any(cfg.get(k)!=v for k,v in exact.items()):raise ValueError('optical motion audit scope/processing contract mismatch')
    paths={f'{CACHE}/eeg_fnirs_single_trial/{s}/{session}.npz' for s in SUBJECTS for session in SESSIONS}
    if set(cfg.get('source_sha256',{}))!=paths or any(not re.fullmatch('[0-9a-f]{64}',v) for v in cfg['source_sha256'].values()):
        raise ValueError('exact nine native cache source identities required')
    return cfg


def validate_metadata(cfg, root, index=None):
    """Complete the 9-record/72-window audit before any array or digest read."""
    root=Path(root).resolve();index=CleanPhysiologyCacheIndex(root/cfg['cache_root']) if index is None else index
    manifest=index.cache_manifest
    if (manifest.get('processing_schema')!=cfg['processing_schema'] or manifest.get('native_only') is not True
            or Path(index.cache_root).resolve()!=(root/cfg['cache_root']).resolve()):
        raise ValueError('native-only cache manifest mismatch')
    plan=[];identities=set()
    for subject in SUBJECTS:
        detail_path=root/cfg['retained_prepared']/f'{subject}.json'
        detail=json.loads(detail_path.read_text())
        if (detail.get('subject')!=subject or detail.get('synthetic') is not False
                or detail.get('observation_contract')!='native_feature_missing_measurement_v3'
                or len(detail.get('trials',[]))!=24 or detail.get('fnirs_pairs',[]).count('AF7Fp1')!=1
                or detail['fnirs_pairs'][0]!='AF7Fp1'):
            raise ValueError('retained subject metadata identity/coordinate mismatch')
        for session_index,session in enumerate(SESSIONS):
            matches=[r for r in index.records if r.dataset_id==cfg['dataset_id'] and
                r.canonical_subject_id==subject and r.base_record_id==session]
            if len(matches)!=1:raise ValueError('exact unique indexed native record required')
            record=matches[0];relative=f'{CACHE}/eeg_fnirs_single_trial/{subject}/{session}.npz'
            m=record.manifest;contract=m.get('native_contract',{});meta=m.get('metadata',{})
            if (record.npz_path.resolve()!=(root/relative).resolve() or not record.npz_path.is_file()
                    or record.signal_branch!='homer2_wavelength_pair' or record.sample_rate_hz!=10.
                    or m.get('native_only') is not True or m.get('processing_schema')!=cfg['processing_schema']
                    or contract.get('measurement_family')!='optical_intensity' or contract.get('native_unit')!='V'
                    or contract.get('channel_roles')!=['lowWL_760nm','highWL_850nm']
                    or meta.get('homer2_pair_labels',[]).count('AF7Fp1')!=1
                    or detail.get('source_sha256',{}).get(relative)!=cfg['source_sha256'][relative]):
                raise ValueError('indexed record path/branch/units/source identity mismatch')
            reports=index.reports_by_join_key.get(record.join_key,[])
            if not reports or any(r.get('alignment_case') not in ADMISSIBLE_ALIGNMENT_CASES or not r.get('label_sequence_match') for r in reports):
                raise ValueError('record event alignment not admissible')
            events=sorted((e for e in index.events_by_join_key.get(record.join_key,[]) if e.get('label')=='MA'),key=lambda e:int(e['event_index']))
            if len(events)!=10 or len({e['event_index'] for e in events})!=10:raise ValueError('exact ten native MA event identities required')
            trials=[]
            for ordinal,position in enumerate(POSITIONS):
                trial_index=session_index*8+ordinal;trial=detail['trials'][trial_index];event=events[position]
                start=round((event['fnirs_time_ms']/1000.-5.)*10.)
                expected_id=f'{record.join_key}|event={event["event_index"]}|offset=-5.0|duration=30.0'
                if (trial.get('subject')!=subject or trial.get('session')!=session
                        or trial.get('original_ma_trial_position')!=position or trial.get('training_ordinal')!=ordinal
                        or trial.get('event_index')!=event['event_index'] or trial.get('fnirs_time_ms')!=event['fnirs_time_ms']
                        or trial.get('native_start_samples',{}).get('fnirs')!=start or start<0
                        or trial.get('sample_id')!=expected_id or expected_id in identities
                        or event.get('event_type')!='trial' or event.get('metadata',{}).get('task')!='mental_arithmetic'
                        or not window_within_alignment_support(event,-5.,30.)):
                    raise ValueError('retained event/window identity mismatch')
                identities.add(expected_id);trials.append(dict(trial_index=trial_index,start=start,stop=start+300,identity=trial))
            plan.append(dict(record=record,relative_path=relative,trials=trials,subject=subject,session=session))
    if len(plan)!=9 or len(identities)!=72:raise ValueError('audit must retain exact 72 windows from nine sessions')
    return index,plan


def paired_columns(names):
    names=[str(v) for v in names];selected=[]
    for marker in ['lowWL','highWL']:
        matches=[]
        for i,label in enumerate(names):
            found=re.fullmatch(r'\s*(.*?)\s*(lowWL|highWL)(.*?)\s*',label)
            if found and found.group(1).strip()=='AF7Fp1' and found.group(2)==marker:
                # If a numerical wavelength suffix is present it must agree with the role.
                suffix=found.group(3).strip();digits=re.findall(r'\d+',suffix)
                if suffix and not re.fullmatch(r'[_ ]*(760|850)\s*(nm)?',suffix):
                    raise ValueError('selected wavelength label has unrecognized suffix')
                if digits and digits!=(['760'] if marker=='lowWL' else ['850']):
                    raise ValueError('selected wavelength label disagrees with manifest role')
                matches.append(i)
        if len(matches)!=1:raise ValueError('selected low/high wavelength labels must each be unique')
        selected.append(matches[0])
    return selected


def load_admitted_windows(index,item,cfg,verified):
    """Only the unified cache loader may open the session container's arrays."""
    record=item['record'];relative=item['relative_path']
    if relative not in verified:
        digest=hashlib.sha256()
        with record.npz_path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=cfg['source_sha256'][relative]:raise ValueError('native source SHA256 mismatch')
        verified.add(relative)
    arrays=index.load_record_arrays(record,keys=KEYS)
    if set(arrays)!=set(KEYS):raise ValueError('required native cache keys missing')
    values=np.asarray(arrays['native_input_fnirs']);names=np.asarray(arrays['native_channel_names'])
    if values.ndim!=2 or names.ndim!=1 or values.shape[1]!=len(names):raise ValueError('native optical tensor/channel shape mismatch')
    columns=paired_columns(names)
    windows=[]
    for trial in item['trials']:
        if trial['stop']>len(values) or trial['stop']-trial['start']!=300:raise ValueError('retained window lacks exact 300-sample support')
        selected=np.asarray(values[trial['start']:trial['stop'],columns],dtype=float)
        if selected.shape!=(300,2) or not np.isfinite(selected).all() or np.any(selected<=0):
            raise ValueError('admitted optical pair must be finite positive intensity')
        windows.append(selected)
    return windows,[str(names[i]) for i in columns]


def mne_motion(od,sampling_hz):
    import mne
    from mne.preprocessing.nirs import temporal_derivative_distribution_repair
    if mne.__version__!='1.11.0':raise ValueError('this audit requires declared installed MNE 1.11.0')
    info=mne.create_info(['S1_D1 760','S1_D1 850'],sampling_hz,ch_types=['fnirs_od']*2)
    for ch,wavelength in zip(info['chs'],[760.,850.]):
        ch['loc'][:3]=[.015,0,0];ch['loc'][3:6]=[0,0,0];ch['loc'][6:9]=[.03,0,0];ch['loc'][9]=wavelength
    raw=mne.io.RawArray(np.asarray(od).T,info,verbose=False)
    return temporal_derivative_distribution_repair(raw,verbose=False).get_data().T


def compare_pipelines(intensity,cfg):
    od,_=intensity_to_optical_density(np.asarray(intensity,float),epsilon=np.finfo(float).tiny)
    current,quality=robust_derivative_motion_suppression(od)
    corrected=dict(no_motion=od,current=current,mne_tddr=mne_motion(od,cfg['sampling_hz']))
    output={}
    for name,values in corrected.items():
        hb,_=modified_beer_lambert(values[:,None,:],wavelengths_nm=cfg['wavelengths_nm'])
        hb=hb[:,0];filtered,_=bandpass_fnirs(hb,sample_rate_hz=10.,low_hz=.01,high_hz=.2,order=3)
        output[name]=dict(od=values,hb_native=hb,hb_filtered=filtered,hb_4hz=resample_poly(filtered,2,5,axis=0))
    return output,quality


def signal_metrics(values,hz):
    values=np.asarray(values);t=np.arange(len(values))/hz;width=min(round(5*hz),len(values)//2)
    return dict(endpoint_change=(values[-1]-values[0]).tolist(),
        linear_slope_per_s=np.polyfit(t,values,1)[0].tolist(),
        mean_derivative_per_sample=np.diff(values,axis=0).mean(axis=0).tolist(),
        late_minus_early_5s=(values[-width:].mean(axis=0)-values[:width].mean(axis=0)).tolist())


def summarize_window(output,identity):
    return dict(identity=identity,pipelines={name:{stage:signal_metrics(value,4. if stage=='hb_4hz' else 10.)
        for stage,value in arrays.items()} for name,arrays in output.items()},
        filtered_difference_rms_from_no_motion={name:np.sqrt(np.mean((arrays['hb_filtered']-output['no_motion']['hb_filtered'])**2,axis=0)).tolist()
            for name,arrays in output.items()})


def synthetic_phase(cfg,out):
    t=np.arange(cfg['synthetic']['steps'])/10.;phase=t%1
    pulse=np.where(phase<.2,.5-.5*np.cos(np.pi*phase/.2),.5+.5*np.cos(np.pi*(phase-.2)/.8))
    noise=np.random.default_rng(cfg['synthetic']['seed']).normal(0,.01,len(t));noise[-1]=noise[0]
    signals=dict(zero=np.zeros(len(t)),constant=np.ones(len(t))*.1,asymmetric_positive=.03*pulse,
        asymmetric_negative=-.03*pulse,true_linear_trend=.001*t,noise=noise)
    rows=[];saved={}
    for name,signal in signals.items():
        # Declared artificial intensity; natural OD removes only a constant reference.
        intensity=np.exp(-np.column_stack((signal,.7*signal)))
        output,quality=compare_pipelines(intensity,cfg);row=summarize_window(output,dict(synthetic_case=name))
        row['current_motion_quality']=quality;rows.append(row)
        for method,stages in output.items():
            for stage,value in stages.items():saved[f'{name}__{method}__{stage}']=value
    np.savez_compressed(out/'synthetic_traces.npz',**saved)
    result=dict(status='completed',cases=rows,interpretation='constructed_motion_free_controls_not_real_physiological_truth; MNE_can_remove_real_linear_trend')
    write_json(out/'synthetic_summary.json',result);return result


def measured_phase(cfg,root,out):
    index,plan=validate_metadata(cfg,root);verified=set();rows=[];reproduction=[];saved={};references={}
    for subject in SUBJECTS:
        with np.load(Path(root)/cfg['retained_prepared']/f'{subject}.npz',allow_pickle=False) as a:
            references[subject]={key:np.array(a[key][:,:,0,:],copy=True) for key in cfg['reproduction']['retained_keys']}
        expected={'feature_od':(24,300,2),'feature_fnirs':(24,300,2),'target_fnirs':(24,120,2)}
        if any(references[subject][k].shape!=shape for k,shape in expected.items()):raise ValueError('retained feature tensor contract mismatch')
    for item in plan:
        windows,labels=load_admitted_windows(index,item,cfg,verified)
        for trial,intensity in zip(item['trials'],windows):
            output,quality=compare_pipelines(intensity,cfg);identity=dict(trial['identity'],trial_index=trial['trial_index'],selected_labels=labels)
            row=summarize_window(output,identity);row['current_motion_quality']=quality;rows.append(row)
            for stage,key in [('od','feature_od'),('hb_native','feature_fnirs'),('hb_4hz','target_fnirs')]:
                actual=output['current'][stage];reference=references[item['subject']][key][trial['trial_index']]
                matches=bool(np.allclose(actual,reference,atol=cfg['reproduction']['absolute_tolerance'],rtol=cfg['reproduction']['relative_tolerance']))
                reproduction.append(dict(sample_id=identity['sample_id'],stage=stage,matches=matches,
                    maximum_absolute_difference=float(np.max(abs(actual-reference)))))
            for method,stages in output.items():
                for stage,value in stages.items():saved.setdefault(f'{method}__{stage}',[]).append(value)
    passed=all(r['matches'] for r in reproduction)
    np.savez_compressed(out/'measured_traces.npz',**{k:np.array(v) for k,v in saved.items()})
    result=dict(status='completed' if passed else 'reproduction_mismatch',windows=len(rows),rows=rows,
        reproduction_checks=reproduction,current_pipeline_reproduced=passed,
        pipeline_attribution='eligible_for_read_only_pipeline_comparison_not_physiological_truth' if passed else 'STOP_reproduction_mismatch_no_causal_attribution',
        verified_native_files=sorted(verified),
        storage_boundary='unified_loader_opens_nine_compressed_session_containers; transforms_only_exact_72_retained_300_sample_AF7Fp1_windows',
        units='natural_OD_and_repo_relative_Hb_not_calibrated_uM',
        limits=['no_motion_is_not_ground_truth','MNE_may_remove_real_slow_trends',
            'filter_edges_affect_window_shape','no_new_trials_subjects_raw_MAT_EEG_or_model_fits_or_scores'])
    write_json(out/'measured_summary.json',result);return result


def export_report(run_dir,export_dir):
    """Render completed retained evidence only; no producer/resume or source reads."""
    import csv
    import fitz
    import markdown
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    run_dir=Path(run_dir).resolve();export_dir=Path(export_dir).resolve()
    if export_dir.parent!=run_dir or (export_dir.exists() and any(export_dir.iterdir())):
        raise ValueError('report requires a new empty direct subdirectory of this run')
    manifest=json.loads((run_dir/'manifest.json').read_text())
    measured=json.loads((run_dir/'measured_summary.json').read_text())
    synthetic=json.loads((run_dir/'synthetic_summary.json').read_text())
    cfg=yaml.safe_load((run_dir/'resolved_config.yaml').read_text())
    if (manifest.get('execution')!='completed' or manifest.get('experiment_id')!='SSM-SHARED-DRIVER-OPTICAL-MOTION-AUDIT-v1'
            or measured.get('status') not in ('completed','reproduction_mismatch') or measured.get('windows')!=72
            or synthetic.get('status')!='completed'):
        raise ValueError('report requires completed retained optical audit evidence')
    rows=measured['rows'];checks=measured['reproduction_checks']
    expected={(subject,session,position) for subject in SUBJECTS for session in SESSIONS for position in POSITIONS}
    actual=[(r['identity']['subject'],r['identity']['session'],r['identity']['original_ma_trial_position']) for r in rows]
    if len(rows)!=72 or len(set(actual))!=72 or set(actual)!=expected:raise ValueError('report identities must retain all 72 windows')
    sample_ids={r['identity']['sample_id'] for r in rows}
    if (len(sample_ids)!=72 or len(checks)!=216 or len({(c['sample_id'],c['stage']) for c in checks})!=216
            or {(c['sample_id'],c['stage']) for c in checks}!={(i,s) for i in sample_ids for s in ['od','hb_native','hb_4hz']}):
        raise ValueError('report requires exact 216 reproduction identities')
    reproduced=all(c['matches'] is True for c in checks)
    if (reproduced!=measured['current_pipeline_reproduced'] or
            measured['status']!=('completed' if reproduced else 'reproduction_mismatch')):
        raise ValueError('reproduction summary disagrees with checks')
    if [r['identity']['synthetic_case'] for r in synthetic['cases']]!=cfg['synthetic']['cases']:
        raise ValueError('report synthetic identities mismatch')
    with np.load(run_dir/'measured_traces.npz',allow_pickle=False) as a:
        traces={key:np.array(a[key],copy=True) for key in a.files}
    with np.load(run_dir/'synthetic_traces.npz',allow_pickle=False) as a:
        syn_traces={key:np.array(a[key],copy=True) for key in a.files}
    for method in PIPELINES:
        for stage,n in [('od',300),('hb_native',300),('hb_filtered',300),('hb_4hz',120)]:
            value=traces[f'{method}__{stage}']
            if value.shape!=(72,n,2) or not np.isfinite(value).all():raise ValueError('report retained trace shape/finiteness mismatch')
            for i,row in enumerate(rows):
                expected_metrics=signal_metrics(value[i],4. if stage=='hb_4hz' else 10.)
                if any(not np.allclose(expected_metrics[k],row['pipelines'][method][stage][k],rtol=1e-10,atol=1e-12)
                       for k in expected_metrics):raise ValueError('summary/trace row ordering or metric mismatch')
    export_dir.mkdir(parents=True,exist_ok=True);figdir=export_dir/'figures';figdir.mkdir()
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':8,'axes.unicode_minus':False})
    colors=dict(no_motion='#2b6cb0',current='#c53030',mne_tddr='#2f855a')
    labels=dict(no_motion='不做运动校正',current='当前导数抑制',mne_tddr='MNE TDDR')
    def csv_file(name,records):
        with (export_dir/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    flat=[];paired=[];aggregate=[]
    for row in rows:
        identity=row['identity']
        for method in PIPELINES:
            for stage,metrics in row['pipelines'][method].items():
                for component,j in [('HbO',0),('HbR',1)] if stage!='od' else [('760nm',0),('850nm',1)]:
                    flat.append(dict(sample_id=identity['sample_id'],subject=identity['subject'],session=identity['session'],
                        trial_index=identity['trial_index'],method=method,stage=stage,component=component,
                        **{k:v[j] for k,v in metrics.items()}))
        for component,j in [('HbO',0),('HbR',1)]:
            current=row['pipelines']['current'];none=row['pipelines']['no_motion']
            paired.append(dict(sample_id=identity['sample_id'],subject=identity['subject'],trial_index=identity['trial_index'],component=component,
                native_endpoint_current_minus_no_motion=current['hb_native']['endpoint_change'][j]-none['hb_native']['endpoint_change'][j],
                native_slope_current_minus_no_motion=current['hb_native']['linear_slope_per_s'][j]-none['hb_native']['linear_slope_per_s'][j],
                filtered_endpoint_current_minus_no_motion=current['hb_filtered']['endpoint_change'][j]-none['hb_filtered']['endpoint_change'][j],
                filtered_difference_rms=row['filtered_difference_rms_from_no_motion']['current'][j]))
    for subject in SUBJECTS:
        for method in PIPELINES:
            for stage in ['hb_native','hb_filtered']:
                for component,j in [('HbO',0),('HbR',1)]:
                    values=[r['pipelines'][method][stage]['endpoint_change'][j] for r in rows if r['identity']['subject']==subject]
                    aggregate.append(dict(subject=subject,method=method,stage=stage,component=component,n=len(values),
                        endpoint_q25=float(np.quantile(values,.25)),endpoint_median=float(np.median(values)),endpoint_q75=float(np.quantile(values,.75))))
    csv_file('window_metrics.csv',flat);csv_file('paired_current_minus_no_motion.csv',paired)
    csv_file('subject_endpoint_distributions.csv',aggregate);csv_file('reproduction_checks.csv',checks)
    synthetic_metrics=[]
    for row in synthetic['cases']:
        for method in PIPELINES:
            for stage,metrics in row['pipelines'][method].items():
                for j in range(2):
                    synthetic_metrics.append(dict(case=row['identity']['synthetic_case'],method=method,stage=stage,component_index=j,
                        **{k:v[j] for k,v in metrics.items()}))
    csv_file('synthetic_metrics.csv',synthetic_metrics)
    figures=[]
    def save(fig,name):
        fig.tight_layout();fig.savefig(figdir/f'{name}.png',dpi=240);plt.close(fig);figures.append(name)
    fig,axes=plt.subplots(2,2,figsize=(7.1,6.5))
    for axis,(stage,component,j) in zip(axes.flat,[(s,c,j) for s in ['hb_native','hb_filtered'] for c,j in [('HbO',0),('HbR',1)]]):
        for method_index,method in enumerate(PIPELINES):
            for subject_index,subject in enumerate(SUBJECTS):
                values=np.array([r['pipelines'][method][stage]['endpoint_change'][j] for r in rows if r['identity']['subject']==subject])
                x=subject_index+(method_index-1)*.23
                axis.scatter(x+np.linspace(-.045,.045,len(values)),values,s=9,alpha=.65,color=colors[method],label=labels[method] if subject_index==0 else None)
                axis.plot([x-.07,x+.07],[np.median(values)]*2,color=colors[method],lw=2)
        axis.axhline(0,color='gray',lw=.5);axis.set_xticks(range(3),['S01','S09','S18'])
        axis.set_title(('原生' if stage=='hb_native' else '滤波后')+f' {component}：末值−初值')
        axis.set_ylabel('相对 Hb，非 μM');axis.grid(alpha=.15)
    axes[0,0].legend(fontsize=6);save(fig,'all_window_distributions')
    fig,axes=plt.subplots(2,2,figsize=(7.1,6.5))
    for j,component in enumerate(['HbO','HbR']):
        for row_index,(key,title) in enumerate([('native_endpoint_current_minus_no_motion','原生末−初变化的配对差'),('filtered_difference_rms','滤波后曲线差 RMS')]):
            ax=axes[row_index,j]
            for i,subject in enumerate(SUBJECTS):
                values=[r[key] for r in paired if r['subject']==subject and r['component']==component]
                ax.scatter(i+np.linspace(-.12,.12,len(values)),values,s=12,alpha=.75,color=colors['current'])
                ax.plot([i-.15,i+.15],[np.median(values)]*2,color='black',lw=2)
            ax.axhline(0,color='gray',lw=.5);ax.set_xticks(range(3),['S01','S09','S18'])
            ax.set_title(f'{component}：{title}');ax.set_ylabel('相对 Hb，非 μM');ax.grid(alpha=.15)
    save(fig,'paired_processing_differences')
    fixed_examples=[('subject_01',0),('subject_09',12),('subject_18',4)]
    example_text=[]
    for subject,trial in fixed_examples:
        matches=[i for i,row in enumerate(rows) if row['identity']['subject']==subject and row['identity']['trial_index']==trial]
        if len(matches)!=1:raise ValueError('fixed report example identity missing or ambiguous')
        i=matches[0];identity=rows[i]['identity'];fig,axes=plt.subplots(2,2,figsize=(7.1,6.2))
        for j,component in enumerate(['HbO','HbR']):
            for column,stage in enumerate(['hb_native','hb_filtered']):
                ax=axes[j,column]
                for method in PIPELINES:ax.plot(np.arange(300)/10.-5.,traces[f'{method}__{stage}'][i,:,j],color=colors[method],label=labels[method],lw=1.)
                ax.axvline(0,color='gray',ls=':',lw=.8);ax.set_title(component+('：原生' if stage=='hb_native' else '：滤波后'))
                ax.set_xlabel('相对 MA 标记时间（秒）');ax.set_ylabel('相对 Hb，非 μM');ax.grid(alpha=.15)
        axes[0,0].legend(fontsize=6)
        name=f'{subject}_trial{trial}';save(fig,name)
        example_text.append((name,identity))
    fig,axes=plt.subplots(2,2,figsize=(7.1,6.2))
    examples=[('asymmetric_positive','零净变化的正向非对称周期'),('asymmetric_negative','零净变化的反向非对称周期'),
              ('true_linear_trend','真实线性趋势：MNE 可将其移除'),('noise','固定种子噪声，端点相同')]
    for ax,(case,title) in zip(axes.flat,examples):
        for method in PIPELINES:
            y=syn_traces[f'{case}__{method}__od'][:,0];ax.plot(np.arange(len(y))/10.,y-y[0],color=colors[method],label=labels[method],lw=1.)
        ax.set_title(title);ax.set_xlabel('秒');ax.set_ylabel('OD−初始 OD（760 nm）');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=6);save(fig,'synthetic_controls')
    medians=[]
    for method in PIPELINES:
        medians.append('| '+labels[method]+' | '+' | '.join(f"{np.median([r['pipelines'][method][s]['endpoint_change'][j] for r in rows]):.6g}" for s,j in [('hb_native',0),('hb_native',1),('hb_filtered',0),('hb_filtered',1)])+' |')
    max_difference=max(c['maximum_absolute_difference'] for c in checks)
    attribution=('当前管道在 216 项 OD / 原生 Hb / 4 Hz Hb 对照中全部重现保留数据，因此三管道差异可解释为这组固定输入下的处理差异。'
        if reproduced else '重现检查未全部通过。本报告保留差异，但停止将保留数据中的下降归因于当前运动校正。')
    first=f'''# 光学运动校正只读审计

本报告比较同一 AF7Fp1 光学输入经“不做运动校正、当前导数抑制、MNE TDDR”三条管道后的曲线。范围为 3 名被试、9 个 session、原有 72 个窗口；没有增加 trial、拟合 SSM 或产生新模型评分。

{attribution} 最大绝对重现差异为 {max_difference:.6g}，判据来自运行配置：atol={cfg['reproduction']['absolute_tolerance']:g}、rtol={cfg['reproduction']['relative_tolerance']:g}。

## 全部 72 窗口的末值−初值中位数

| 管道 | 原生 HbO | 原生 HbR | 滤波 HbO | 滤波 HbR |
|---|---:|---:|---:|---:|
'''+ '\n'.join(medians)+f'''

所有 Hb 数值均为仓库近似 MBLL 的相对单位，未标定为 μM。“原生”在此指运动处理与 MBLL 之后、线性滤波之前，不是原始探测电压。各管道共用 −ln(I/median)、MBLL 和 10 Hz 下 0.01–0.2 Hz 三阶带通；4 Hz 对照另用 2/5 重采样。滤波窗口端点存在边缘效应。

不做运动校正和 MNE 均不是生理真值。合成负对照表明：当前处理可能给零净变化的非对称周期引入漂移；MNE 也能去除输入中真实存在的慢趋势。这些事实不能单独判定人体信号的真实趋势。

运行：{run_dir.name}；MNE {manifest['versions']['mne']}，NumPy {manifest['versions']['numpy']}。输入与实现来源见原 manifest.json 的 source_files / source_sha256；本导出不修改这些文件。所有 216 条检查与每窗指标均随报告提供 CSV。
'''
    sections=[first,
        '## 全部窗口分布\n\n每个点为一个窗口，每被试每管道 24 点；短横线为中位数。没有按结果筛选样本。\n\n![全部分布](figures/all_window_distributions.png)',
        '## 当前处理相对不校正的配对变化\n\n上排为同一窗口的原生 Hb 末−初差之差，下排为两条滤波曲线的差 RMS（描述处理差异，不是重建 NRMSE）。全部 72 窗口均保留。\n\n![配对变化](figures/paired_processing_differences.png)']
    for name,identity in example_text:
        sections.append(f"## 固定示例：{identity['subject']} / trial {identity['trial_index']}\n\n{identity['session']}，原始 MA 位置 {identity['original_ma_trial_position']}，事件 {identity['event_index']}。这三个身份在导出前已固定，未按曲线效果挑选。纵线为 MA 事件；显示原处理数值，未为作图再次平移基线。\n\n![固定曲线](figures/{name}.png)")
    sections.append('## 合成反例与解释边界\n\n六个确定性输入含零、常数、正/负非对称周期、真实线性趋势及噪声；图示四个非平凡案例。这里没有注入运动伪迹，周期形状也不是经过验证的人体脉搏模型。显示 OD 相对初值；MNE 消除真实趋势的例子必须与周期漂移例子一同解读。\n\n![合成对照](figures/synthetic_controls.png)\n\n所有图均为 240 dpi PNG；正文和表格可搜索。WPS 滚动与缩放未检查。')
    (export_dir/'REPORT.md').write_text('\n\n---\n\n'.join(sections)+'\n')
    css='body{font-family:sans-serif;font-size:9pt;line-height:1.45}h1{font-size:19pt}h2{font-size:13pt}table{font-size:8pt;border-collapse:collapse}td,th{border:0.4pt solid #aaa;padding:4pt}img{max-width:100%}'
    def page_rect(number,filled):
        if number>3:raise RuntimeError('report section unexpectedly overflows')
        paper=fitz.paper_rect('a4');return paper,paper+(36,30,-36,-30),None
    pdf=fitz.open()
    for section in sections:
        html=markdown.markdown(section,extensions=['tables']).replace('<img ','<img width="510" ')
        story=fitz.Story(html,user_css=css,archive=fitz.Archive(str(export_dir)))
        part=story.write_with_links(page_rect);pdf.insert_pdf(part);part.close()
    pdf.save(export_dir/'REPORT.pdf');pdf.close()
    document=fitz.open(export_dir/'REPORT.pdf')
    pages=[dict(page=i+1,images=len(page.get_images()),vector_paths=len(page.get_drawings()),searchable_characters=len(page.get_text())) for i,page in enumerate(document)]
    validation=dict(pages=len(document),page_details=pages,png_dpi=240,wps_checked=False,
        figure_pages=sum(p['images']>0 for p in pages),reproduction_passed=reproduced,
        report_source=str(Path(__file__).resolve()),report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if len(document)>8 or validation['figure_pages']!=6 or any(p['images'] and p['vector_paths'] for p in pages):
        raise ValueError('report pagination or bitmap figure validation failed')
    write_json(export_dir/'export_validation.json',validation)
    print(json.dumps(dict(status='completed',export_dir=str(export_dir),pages=len(document),reproduction_passed=reproduced)))
    document.close();return validation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT);parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--run-dir',type=Path);parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--phase',choices=['synthetic','measured','report'],default='synthetic')
    parser.add_argument('--export-dir',type=Path)
    args=parser.parse_args()
    if args.phase=='report':
        if args.run_dir is None or args.export_dir is None or args.check_only:
            parser.error('report requires run-dir and a new empty export-dir; check-only is producer-only')
        return export_report(args.run_dir,args.export_dir)
    if args.export_dir is not None:parser.error('export-dir only applies to report phase')
    cfg=read_config(args.config);root=args.project_root.resolve()
    if args.check_only:
        _,plan=validate_metadata(cfg,root)
        print(json.dumps(dict(status='passed',arrays_read=0,source_files_hashed=0,records=len(plan),windows=sum(len(p['trials']) for p in plan))));return
    if args.run_dir is None or args.run_dir.resolve().parent!=(root/cfg['output_root']).resolve():raise ValueError('run-dir must belong to declared artifact root')
    out=args.run_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest_path=out/'manifest.json';old=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if old.get('execution')=='completed':raise ValueError('completed evidence is immutable')
    if old and (old.get('experiment_id')!=cfg['experiment_id'] or old.get('project_root')!=str(root)
                or old.get('source_root')!=str(ROOT)):
        raise ValueError('resume source/project identity mismatch')
    if (out/(args.phase+'_summary.json')).exists():raise ValueError('terminal phase evidence is immutable')
    resolved=out/'resolved_config.yaml'
    if resolved.exists() and yaml.safe_load(resolved.read_text())!=cfg:raise ValueError('resume config mismatch')
    if not resolved.exists():resolved.write_text(yaml.safe_dump(cfg,sort_keys=False))
    if args.phase=='measured':
        previous=json.loads((out/'synthetic_summary.json').read_text())
        if (previous.get('status')!='completed' or
                [r.get('identity',{}).get('synthetic_case') for r in previous.get('cases',[])]!=cfg['synthetic']['cases']):
            raise ValueError('synthetic terminal evidence required')
    import mne
    if mne.__version__!=cfg['mne_version']:raise ValueError('declared MNE software version mismatch')
    sources=dict(runner=str(Path(__file__).resolve()),current_motion=inspect.getsourcefile(robust_derivative_motion_suppression),
        cache_index=inspect.getsourcefile(CleanPhysiologyCacheIndex),alignment=inspect.getsourcefile(window_within_alignment_support),
        mne_tddr=str(Path(importlib.import_module('mne.preprocessing.nirs._tddr').__file__).resolve()))
    source_hashes={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in sources.items()}
    if old and old.get('source_sha256')!=source_hashes:raise ValueError('resume implementation source identity mismatch')
    manifest=dict(experiment_id=cfg['experiment_id'],execution='running',phase=args.phase,project_root=str(root),
        supervisor=os.environ.get('SSM_SYSTEMD_UNIT'),controller_pid=os.getpid(),source_root=str(ROOT),
        source_files=sources,source_sha256=source_hashes,
        versions=dict(python=platform.python_version(),numpy=np.__version__,mne=mne.__version__),
        scope=cfg['scope'],
        input_boundary=dict(native_loader='CleanPhysiologyCacheIndex.load_record_arrays',native_keys=KEYS,
            container_access='nine_compressed_native_session_containers_opened_by_unified_loader_only_in_measured_phase',
            transformation_scope='exact_72_retained_300_sample_AF7Fp1_windows_only_no_new_trial_output',
            retained_reference_keys=cfg['reproduction']['retained_keys'],excluded='raw_MAT_EEG_other_subjects_model_fits_and_scores'),
        started_at=old.get('started_at',datetime.now(timezone.utc).isoformat()))
    write_json(manifest_path,manifest)
    try:
        result=synthetic_phase(cfg,out) if args.phase=='synthetic' else measured_phase(cfg,root,out)
    except Exception as exc:
        manifest.update(execution='failed',error=repr(exc),finished_at=datetime.now(timezone.utc).isoformat())
        write_json(manifest_path,manifest)
        raise
    manifest.update(execution='synthetic_terminal' if args.phase=='synthetic' else 'completed',
        result_status=result['status'],finished_at=datetime.now(timezone.utc).isoformat())
    write_json(manifest_path,manifest);print(json.dumps(dict(phase=args.phase,status=result['status'])))


if __name__=='__main__':main()
