#!/usr/bin/env python3
"""Bounded actual-file audit for the calibration followup; no measured model fit."""
from pathlib import Path
import argparse
import json
import sys
import zipfile
import numpy as np
import pandas as pd
import yaml
from scipy.io import whosmat
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
from src.data.registry import get_dataset_registration
from src.data.unified_physiology import load_native_eeg_record
from src.data.eeg_fnirs_dataset import load_mat_struct


def audit(cfg,out):
    follow=cfg['followup'];prior=ROOT/follow['parent_run']
    selected=[x for x in json.loads((prior/'record_evidence.json').read_text()) if x['dataset_id']=='eeg_fnirs_single_trial']
    expected={(s,r) for s in follow['measured_subjects'] for r in follow['measured_sessions']}
    if {(x['subject'],x['record']) for x in selected}!=expected:raise ValueError('public source identity mismatch')
    index=CleanPhysiologyCacheIndex(ROOT/cfg['cache_root'])
    registration=get_dataset_registration('eeg_fnirs_single_trial')
    source=ROOT/'data/EEG+NIRS Single-Trial';records=[];files=[];archives=[];examples=[]
    for subject in follow['measured_subjects']:
        number=int(subject[-2:]);label=f'subject {number:02d}'
        eegdir=source/'EEG_01-29'/label/'with occular artifact';nirsdir=source/'NIRS_01-29'/label
        # Reuse the registered dataset's MATLAB reader; inspect selected cells only.
        ecnt=load_mat_struct(str(eegdir/'cnt.mat'));ncnt=load_mat_struct(str(nirsdir/'cnt.mat'))
        emrk=load_mat_struct(str(eegdir/'mrk.mat'))['mrk'];nmrk=load_mat_struct(str(nirsdir/'mrk.mat'))['mrk']
        paths=list(eegdir.glob('*.mat'))+list(eegdir.parent.glob('*.mat'))+list(nirsdir.glob('*.mat'))
        for path in sorted(paths):
            entry=dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,variables=whosmat(path))
            # Artifact recordings are auxiliary candidates too: inspect channel metadata.
            if path.name.startswith('cnt_artifact'):
                payload=load_mat_struct(str(path));desc=[]
                for k,v in payload.items():
                    if k.startswith('__'):continue
                    for item in np.atleast_1d(v):
                        desc.append(dict(variable=k,fields=list(getattr(item,'_fieldnames',[])),
                            channel_names=[str(x) for x in np.atleast_1d(getattr(item,'clab',[]))],
                            sample_rate_hz=float(getattr(item,'fs',0))))
                entry['artifact_metadata']=desc
            files.append(entry)
        zipname={1:'EEG_01-05.zip',9:'EEG_06-10.zip',18:'EEG_16-20.zip'}[number]
        for name in [zipname,'NIRS_01-29.zip']:
            with zipfile.ZipFile(source/name) as archive:
                entries=[dict(name=i.filename,bytes=i.file_size) for i in archive.infolist() if f'/{label}/' in '/'+i.filename]
            archives.append(dict(archive=name,subject=subject,entries=entries))
        for record_id in follow['measured_sessions']:
            key=f'eeg_fnirs_single_trial|{subject}|{record_id}';matches=index.records_by_join_key.get(key,[])
            if len(matches)!=1:raise ValueError('record not uniquely registered')
            record=matches[0];session=int(record_id[-2:]);e=ecnt['cnt'][session];n=ncnt['cnt'][session]
            native=load_native_eeg_record(ROOT,record)
            labels=[str(x) for x in n.clab];lows=[x.removesuffix('lowWL') for x in labels if x.endswith('lowWL')];highs=[x.removesuffix('highWL') for x in labels if x.endswith('highWL')]
            if lows!=highs or len(lows)!=36:raise ValueError('optical wavelength identity mismatch')
            et=np.asarray(emrk[session].time,dtype=float).ravel();nt=np.asarray(nmrk[session].time,dtype=float).ravel()
            names=[str(x) for x in e.clab]
            candidates=[x for x in names if any(k in x.upper() for k in ['ECG','EKG','RESP','BREATH'])]
            reports=index.reports_by_join_key.get(key,[])
            arrays=index.load_record_arrays(record,('fnirs',))
            row=dict(join_key=key,subject=subject,record=record_id,task=str(e.title),
                eeg_path=str(native.source_path.relative_to(ROOT)),fnirs_path=str((nirsdir/'cnt.mat').relative_to(ROOT)),
                eeg_top_level_variables=[k for k in ecnt if not k.startswith('__')],eeg_fields=e._fieldnames,fnirs_fields=n._fieldnames,
                eeg_channel_names=names,eeg_shape=list(e.x.shape),eeg_hz=float(e.fs),eeg_unit=str(e.yUnit),
                exposed_eeg_channels=list(native.channel_names),exposed_auxiliary_channels=list(native.auxiliary_channel_names),
                systemic_auxiliary_candidates=candidates,fnirs_channel_names=labels,fnirs_shape=list(n.x.shape),fnirs_hz=float(n.fs),fnirs_unit=str(n.yUnit),
                wavelengths_nm=np.asarray(n.wavelengths).tolist(),n_sources=int(n.nSources),n_detectors=int(n.nDetectors),pairs=len(lows),
                eeg_seconds=len(e.x)/float(e.fs),fnirs_seconds=len(n.x)/float(n.fs),
                eeg_event_count=len(et),fnirs_event_count=len(nt),eeg_marker_first_ms=float(et[0]),fnirs_marker_first_ms=float(nt[0]),
                corresponding_marker_offsets_ms=(nt-et).tolist() if len(nt)==len(et) else None,
                existing_alignment_reports=reports,
                fnirs_preprocessing=record.manifest['measurement']['fnirs_preprocessing_state'],
                native_finite_fraction=float(np.isfinite(n.x).mean()),native_positive_fraction=float((n.x>0).mean()),
                conclusion='no_ECG_or_respiration_channel_in_selected_released_record; EOG_is_not_systemic_calibration')
            records.append(row)
            if session==1:
                count=min(1200,len(n.x));examples.append(dict(join_key=key,pair=lows[0],time_s=(np.arange(count)/float(n.fs)).tolist(),
                    intensity_V=np.asarray(n.x[:count,[0,36]]).tolist(),processed_relative_Hb=np.asarray(arrays['fnirs'][:count,:2]).tolist()))
            print(key, 'channels',len(names),'aux',native.auxiliary_channel_names,'systemic',candidates,flush=True)
    auxiliary_names=[x for f in files for m in f.get('artifact_metadata',[]) for x in m['channel_names'] if any(k in x.upper() for k in ['ECG','EKG','RESP','BREATH'])]
    result=dict(schema='hbo_calibration_actual_sources_v1',scope='nine_existing_public_records_three_subjects; archive filenames and artifact channel metadata only',
        records=records,inspected_files=files,archive_subject_members=archives,examples=examples,
        documentation=[r.relative_path for r in registration.documentation],
        artifact_systemic_channel_candidates=auxiliary_names,
        independent_calibration_established=False,
        C_status='not_executable_no_independently_grounded_measured_observation_candidate_in_inspected_files',
        D_status='not_executable_C_requires_positive_heldout_evidence',
        missing_semantics='not_found_in_selected_local_release_and_matching_archive_members; original_acquisition_may_have_unreleased_auxiliary_data',
        sensitivity='no_empirical_DPF_or_gain_bounds_established; synthetic_0.8_1.2_not_reused',
        pipeline_change='none; native_measurement_amplitude_masks_units_and_event_alignment_preserved')
    out.mkdir(exist_ok=True)
    (out/'source_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in records]).to_csv(out/'source_records.csv',index=False)
    print('audit completed',len(records),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();cfg=yaml.safe_load(args.config.read_text())
    if cfg['schema']!='hbo_hbr_calibration_v2' or args.output_dir.resolve().parent!=ROOT/cfg['output_namespace']:raise ValueError('wrong source audit contract/output root')
    if (args.output_dir/'source_audit.json').exists():raise ValueError('completed audit is immutable')
    audit(cfg,args.output_dir)
