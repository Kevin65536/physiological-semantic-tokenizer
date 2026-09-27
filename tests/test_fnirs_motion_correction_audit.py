"""Temporary public cache fixtures for the read-only optical pipeline audit."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from experiments.scripts import analyze_fnirs_motion_correction as run
from src.data.event_alignment import EVENT_ALIGNMENT_SCHEMA


@pytest.fixture
def fixture(tmp_path):
    cfg=run.read_config(run.DEFAULT);records=[];events=[];reports=[]
    root=tmp_path;cache=root/cfg['cache_root'];cache.mkdir(parents=True)
    for subject in run.SUBJECTS:
        detail=dict(subject=subject,synthetic=False,observation_contract='native_feature_missing_measurement_v3',
            fnirs_pairs=['AF7Fp1'],trials=[],source_sha256={})
        for session in run.SESSIONS:
            relative=f'{run.CACHE}/eeg_fnirs_single_trial/{subject}/{session}.npz';path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
            # Label order intentionally differs between wavelengths; selecting
            # AF7 by actual label must return columns 1 and 2, not positional 1/3.
            labels=np.array(['AF3Fp1lowWL','AF7Fp1lowWL','AF7Fp1highWL','AF3Fp1highWL'])
            values=np.column_stack([np.full(4000,value) for value in [9.,2.,3.,8.]])
            np.savez(path,native_input_fnirs=values,native_channel_names=labels)
            digest=hashlib.sha256(path.read_bytes()).hexdigest();cfg['source_sha256'][relative]=digest;detail['source_sha256'][relative]=digest
            subject_native='subject '+subject[-2:]
            key=f'eeg_fnirs_single_trial|{subject}|{session}'
            base=dict(dataset_id=cfg['dataset_id'],subject=subject_native,record_id=session)
            records.append(dict(base,schema='clean_eeg_fnirs_cache_v2',processing_schema=cfg['processing_schema'],native_only=True,
                record_npz=str(path),sample_rate_hz=10.,native_contract=dict(measurement_family='optical_intensity',native_unit='V',
                channel_roles=['lowWL_760nm','highWL_850nm']),metadata=dict(homer2_pair_labels=['AF7Fp1','AF3Fp1'])))
            reports.append(dict(base,alignment_case='stable_fixed_offset',label_sequence_match=True))
            for i in range(10):
                onset=10000.+i*35000.
                events.append(dict(base,event_index=i,event_type='trial',label='MA',fnirs_time_ms=onset,eeg_time_ms=onset,
                    metadata=dict(task='mental_arithmetic',alignment_support_ms=dict(eeg=[None,None],fnirs=[None,None]))))
                if i in run.POSITIONS:
                    detail['trials'].append(dict(subject=subject,session=session,original_ma_trial_position=i,
                        training_ordinal=run.POSITIONS.index(i),event_index=i,fnirs_time_ms=onset,
                        native_start_samples=dict(fnirs=round((onset/1000-5)*10)),
                        sample_id=f'{key}|event={i}|offset=-5.0|duration=30.0'))
        run.write_json(root/cfg['retained_prepared']/f'{subject}.json',detail)
    run.write_json(cache/'cache_manifest.json',dict(schema='clean_eeg_fnirs_cache_v2',processing_schema=cfg['processing_schema'],native_only=True,records=records))
    run.write_json(cache/'event_index/event_manifest.json',dict(event_alignment_schema=EVENT_ALIGNMENT_SCHEMA))
    for name,rows in [('events',events),('alignment_reports',reports)]:
        (cache/'event_index'/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return root,cfg


def test_metadata_preflight_reads_no_arrays_or_source_bytes(fixture,monkeypatch):
    root,cfg=fixture
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('metadata preflight opened arrays'))
    index,plan=run.validate_metadata(cfg,root)
    assert len(plan)==9 and sum(len(x['trials']) for x in plan)==72
    assert len({t['identity']['sample_id'] for item in plan for t in item['trials']})==72


@pytest.mark.parametrize('change',['path','duplicate_record','branch','event','start','sha','pair','session'])
def test_boundary_changes_fail_before_arrays(fixture,monkeypatch,change):
    root,cfg=fixture;index,plan=run.validate_metadata(cfg,root)
    if change=='path':
        item=plan[0]['record'];object.__setattr__(item,'npz_path',root/'unapproved.npz')
    elif change=='duplicate_record':index.records.append(index.records[0])
    elif change=='branch':object.__setattr__(plan[0]['record'],'signal_branch','other')
    elif change=='event':index.events_by_join_key[plan[0]['record'].join_key][0]['fnirs_time_ms']+=1000
    else:
        file=root/cfg['retained_prepared']/'subject_01.json';record=json.loads(file.read_text())
        if change=='start':record['trials'][0]['native_start_samples']['fnirs']+=1
        elif change=='sha':record['source_sha256'][plan[0]['relative_path']]='0'*64
        elif change=='pair':record['fnirs_pairs']=['AF3Fp1']
        else:record['trials'][0]['session']='session_03'
        run.write_json(file,record)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('invalid boundary opened arrays'))
    with pytest.raises(ValueError):run.validate_metadata(cfg,root,index)


def test_loader_keys_labels_and_file_hash_once(fixture,monkeypatch):
    root,cfg=fixture;index,plan=run.validate_metadata(cfg,root);item=plan[0]
    original=index.load_record_arrays;calls=[]
    def load(record,keys):
        calls.append(keys);return original(record,keys)
    monkeypatch.setattr(index,'load_record_arrays',load)
    verified=set();windows,labels=run.load_admitted_windows(index,item,cfg,verified)
    assert calls==[run.KEYS] and labels==['AF7Fp1lowWL','AF7Fp1highWL']
    assert len(windows)==8 and all(w.shape==(300,2) for w in windows)
    np.testing.assert_array_equal(windows[0],np.tile([2.,3.],(300,1)))
    assert verified=={item['relative_path']}
    monkeypatch.setattr(hashlib,'sha256',lambda *a,**k:pytest.fail('native file hashed twice'))
    run.load_admitted_windows(index,item,cfg,verified)


@pytest.mark.parametrize('names',[
    ['AF7Fp1lowWL','AF3Fp1highWL'],['AF7Fp1lowWL','AF7Fp1lowWL','AF7Fp1highWL'],
    ['AF7Fp1highWL760nm','AF7Fp1lowWL850nm'],['AF7Fp1lowWL','AF7Fp1highWL','AF7Fp1highWL']])
def test_ambiguous_or_swapped_wavelength_labels_rejected(names):
    with pytest.raises(ValueError):run.paired_columns(names)


def test_shape_and_hash_fail_closed(fixture,monkeypatch):
    root,cfg=fixture;index,plan=run.validate_metadata(cfg,root);item=plan[0]
    bad=deepcopy(cfg);bad['source_sha256'][item['relative_path']]='0'*64
    monkeypatch.setattr(index,'load_record_arrays',lambda *a,**k:pytest.fail('SHA mismatch reached array loader'))
    with pytest.raises(ValueError,match='SHA256'):run.load_admitted_windows(index,item,bad,set())
    monkeypatch.setattr(index,'load_record_arrays',lambda *a,**k:dict(native_input_fnirs=np.ones((2,2,2)),
        native_channel_names=np.array(['AF7Fp1lowWL','AF7Fp1highWL'])))
    with pytest.raises(ValueError,match='shape'):run.load_admitted_windows(index,item,cfg,set())


def test_current_pipeline_reproduces_owner_processing_without_reordering():
    from src.data.homer2_preprocessing import apply_homer2_aligned_contract
    cfg=run.read_config(run.DEFAULT);t=np.arange(300)/10.
    intensity=np.exp(-np.column_stack((.04*np.sin(t),.02*np.cos(t*.4))))
    result,_=run.compare_pipelines(intensity,cfg)
    owner=apply_homer2_aligned_contract(intensity[:,None,:],dataset_id='eeg_fnirs_single_trial',sample_rate_hz=10.,
        entry_stage='raw_intensity',wavelengths_nm=(760.,850.),processing_schema=cfg['processing_schema'],retain_feature_boundary=True)
    np.testing.assert_array_equal(result['current']['od'],owner.pre_linear_optical_density[:,0])
    np.testing.assert_array_equal(result['current']['hb_native'],owner.pre_linear_values[:,0])
    np.testing.assert_array_equal(result['current']['hb_filtered'],owner.values.reshape(300,1,2)[:,0])


def test_synthetic_controls_preserve_negative_motion_free_evidence(tmp_path):
    cfg=run.read_config(run.DEFAULT);result=run.synthetic_phase(cfg,tmp_path)
    rows={r['identity']['synthetic_case']:r for r in result['cases']}
    assert list(rows)==cfg['synthetic']['cases']
    for name in ['zero','constant']:
        for values in rows[name]['pipelines'].values():
            np.testing.assert_allclose(values['od']['endpoint_change'],0.,atol=1e-14)
    positive=rows['asymmetric_positive']['pipelines'];negative=rows['asymmetric_negative']['pipelines']
    assert abs(positive['no_motion']['od']['endpoint_change'][0])<1e-14
    drift=positive['current']['od']['endpoint_change'][0]
    assert abs(drift)>.01
    assert negative['current']['od']['endpoint_change'][0]==pytest.approx(-drift,abs=1e-12)
    trend=rows['true_linear_trend']['pipelines']
    assert trend['no_motion']['od']['endpoint_change'][0]>.1
    assert abs(trend['mne_tddr']['od']['endpoint_change'][0])<.1*trend['no_motion']['od']['endpoint_change'][0]
    assert 'not_real_physiological_truth' in result['interpretation']


@pytest.mark.parametrize('mismatch',[False,True])
def test_complete_window_reproduction_controls_attribution(fixture,tmp_path,monkeypatch,mismatch):
    root,cfg=fixture
    for subject in run.SUBJECTS:
        od=np.zeros((24,300,1,2));hb=od.copy();target=np.zeros((24,120,1,2))
        if mismatch and subject==run.SUBJECTS[0]:od[0,0,0,0]=.1
        np.savez(root/cfg['retained_prepared']/f'{subject}.npz',feature_od=od,feature_fnirs=hb,target_fnirs=target)
    def compare(intensity,config):
        return {name:dict(od=np.zeros((300,2)),hb_native=np.zeros((300,2)),
            hb_filtered=np.zeros((300,2)),hb_4hz=np.zeros((120,2))) for name in run.PIPELINES},{}
    monkeypatch.setattr(run,'compare_pipelines',compare)
    out=tmp_path/'audit';out.mkdir()
    result=run.measured_phase(cfg,root,out)
    assert result['windows']==72 and len(result['rows'])==72
    assert len(result['verified_native_files'])==9 and len(result['reproduction_checks'])==216
    assert result['current_pipeline_reproduced']==(not mismatch)
    if mismatch:
        assert result['status']=='reproduction_mismatch' and result['pipeline_attribution'].startswith('STOP_')
        assert sum(not r['matches'] for r in result['reproduction_checks'])==1
    else:assert result['status']=='completed'
    assert (out/'measured_traces.npz').is_file() and (out/'measured_summary.json').is_file()


def test_manifest_records_real_implementation_files_and_rejects_changed_source_resume(fixture,monkeypatch):
    root,cfg=fixture;config=root/'audit.yaml';config.write_text(yaml.safe_dump(cfg))
    out=root/cfg['output_root']/'fixture'
    def synthetic(config,destination):
        result=dict(status='completed',cases=[dict(identity=dict(synthetic_case=name)) for name in config['synthetic']['cases']])
        run.write_json(destination/'synthetic_summary.json',result);return result
    monkeypatch.setattr(run,'synthetic_phase',synthetic)
    argv=['audit','--config',str(config),'--project-root',str(root),'--run-dir',str(out)]
    monkeypatch.setattr(run.sys,'argv',argv+['--phase','synthetic']);run.main()
    manifest=json.loads((out/'manifest.json').read_text())
    assert set(manifest['source_files'])=={'runner','current_motion','cache_index','alignment','mne_tddr'}
    assert all(Path(p).is_file() for p in manifest['source_files'].values())
    assert manifest['input_boundary']['native_keys']==run.KEYS
    manifest['source_sha256']['current_motion']='0'*64;run.write_json(out/'manifest.json',manifest)
    monkeypatch.setattr(run,'measured_phase',lambda *a,**k:pytest.fail('changed implementation reached measured phase'))
    monkeypatch.setattr(run.sys,'argv',argv+['--phase','measured'])
    with pytest.raises(ValueError,match='implementation source identity'):run.main()


def test_completed_report_uses_retained_evidence_only_and_bitmap_figures(fixture,tmp_path,monkeypatch):
    root,cfg=fixture
    for subject in run.SUBJECTS:
        np.savez(root/cfg['retained_prepared']/f'{subject}.npz',feature_od=np.zeros((24,300,1,2)),
            feature_fnirs=np.zeros((24,300,1,2)),target_fnirs=np.zeros((24,120,1,2)))
    monkeypatch.setattr(run,'compare_pipelines',lambda x,c:({name:dict(od=np.zeros((300,2)),hb_native=np.zeros((300,2)),
        hb_filtered=np.zeros((300,2)),hb_4hz=np.zeros((120,2))) for name in run.PIPELINES},{}))
    out=tmp_path/'completed_audit';out.mkdir();run.measured_phase(cfg,root,out)
    # Build retained synthetic fixtures directly; report must not call processors.
    synthetic=[];arrays={}
    for name in cfg['synthetic']['cases']:
        processed={m:{stage:np.zeros((1201 if stage!='hb_4hz' else 481,2))
            for stage in ['od','hb_native','hb_filtered','hb_4hz']} for m in run.PIPELINES}
        synthetic.append(run.summarize_window(processed,dict(synthetic_case=name)))
        for method,stages in processed.items():
            for stage,value in stages.items():arrays[f'{name}__{method}__{stage}']=value
    np.savez(out/'synthetic_traces.npz',**arrays)
    run.write_json(out/'synthetic_summary.json',dict(status='completed',cases=synthetic))
    (out/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
    run.write_json(out/'manifest.json',dict(execution='completed',experiment_id=cfg['experiment_id'],
        versions=dict(mne='1.11.0',numpy=np.__version__)))
    original=(out/'manifest.json').read_bytes()
    monkeypatch.setattr(run,'validate_metadata',lambda *a,**k:pytest.fail('report accessed producer source metadata'))
    monkeypatch.setattr(run,'compare_pipelines',lambda *a,**k:pytest.fail('report reran numerical processing'))
    result=run.export_report(out,out/'report_v1')
    assert 5<=result['pages']<=8 and result['figure_pages']==6
    assert all(p['vector_paths']==0 for p in result['page_details'] if p['images'])
    assert sum(p['searchable_characters'] for p in result['page_details'])>1000
    assert (out/'manifest.json').read_bytes()==original
    assert (out/'report_v1/REPORT.md').is_file() and (out/'report_v1/reproduction_checks.csv').is_file()
    with pytest.raises(ValueError,match='new empty'):run.export_report(out,out/'report_v1')
