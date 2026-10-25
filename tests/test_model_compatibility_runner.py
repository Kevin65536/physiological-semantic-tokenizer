"""Data-free census contracts, resumability and exact hierarchical summaries."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.scripts import evaluate_model_compatibility as runner


def fixture_record(subject='s1',record='r1'):
    return SimpleNamespace(dataset_id='fixture',canonical_subject_id=subject,
        record_id=record,join_key=f'fixture|{subject}|{record}',signal_branch='oxy_deoxy')


def fixture_native(times=None):
    times=np.arange(1850)/10 if times is None else np.asarray(times)
    signal=np.sin(times/3)
    hb=np.stack((signal,.3*np.cos(times/4)),axis=-1)[:,None,:]
    hb=np.repeat(hb,3,axis=1)
    hb[:,2]=1
    return dict(values=hb,time_s=times,sample_rate_hz=10.,
        channel_names=['p1_HbO','p1_HbR','p2_HbO','p2_HbR','p3_HbO','p3_HbR'],
        provenance={'native_unit':'unreported_fixture','transform':'none'})


def test_fixed_contract_and_metadata_only_plan():
    config=runner.load_config()
    assert len(runner.tau_grid(config))==31
    assert len(config['panels']['linear']['tau_grid'])==16
    assert .9 in config['summary']['quantiles']
    record=fixture_record()
    record.manifest={'array_shapes':{'fnirs':[1850,6]}}
    record.sample_rate_hz=10.
    config=deepcopy(config)
    config['census'].update(datasets=['fixture'],expected_records=1,expected_subjects=1)
    index=SimpleNamespace(records=[record],events_by_join_key={})
    plan,inventory,records=runner.plan_inventory(index,config)
    assert plan['records']==1 and plan['estimated_pair_windows']==24
    assert inventory[0]['join_key']=='fixture|s1|r1' and records==[record]
    config['census']['expected_records']=2
    with pytest.raises(ValueError,match='fixed public inventory'):
        runner.plan_inventory(index,config)


def test_synthetic_stage_covers_known_truth_without_data_access(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('synthetic stage cannot read data')
    monkeypatch.setattr(runner,'load_native_fnirs_record',forbidden)
    monkeypatch.setattr(runner,'CleanPhysiologyCacheIndex',forbidden)
    result=runner.synthetic_checks(runner.load_config())
    assert result['execution']=='passed' and len(result['cases'])==34
    cases=pd.DataFrame(result['cases'])
    legal=cases[cases.case.isin(['legal_small_signal','legal_same_phase_initial','common_scale_offset'])]
    assert legal.nominal_D.max()<1e-12
    assert set(cases.sample_rate_hz)=={10.,47.62}
    assert set(cases[cases.case=='additive_independent_noise'].noise_relative_rms)=={0.,.03,.1}


def test_all_pair_rows_and_profiles_retain_exclusions_and_missing(tmp_path):
    native=fixture_native();native['values'][100,0,1]=np.nan
    event={'metadata':{'alignment_support_ms':{'fnirs':[45000,90000]}}}
    meta=runner.evaluate_record(fixture_record(),[event],runner.load_config(),tmp_path,
        reader=lambda *_:native)
    assert meta['execution']=='completed'
    frame=pd.read_csv(meta['csv_path'])
    assert len(frame)==24 and meta['windows']==8
    assert (frame.status=='excluded_alignment_boundary').sum()==6
    assert (frame.status=='missing_support').sum()==1
    assert frame.loc[frame.status!='ok',['nominal_D','best_D','best_tau']].isna().all().all()
    assert meta['native_provenance']['native_unit']=='unreported_fixture'
    with np.load(meta['profiles_path']) as values:
        assert values['profiles_30s'].shape==(6,3,31)
        assert values['profiles_90s'].shape==(2,3,31)
        assert np.isnan(values['profiles_30s'][1]).all()


@pytest.mark.parametrize('times',[np.delete(np.arange(600)/10,100),np.arange(350,950)/10,np.arange(599)/10])
def test_time_gaps_empty_first_window_and_incomplete_tail_not_integrated(tmp_path,times):
    native=fixture_native(times)
    calls=[]
    def metric(*args,**kwargs):
        calls.append(kwargs['times'])
        return runner.weak_hb_profiles(*args,**kwargs)
    meta=runner.evaluate_record(fixture_record(),[],runner.load_config(),tmp_path,
        reader=lambda *_:native,metric_fn=metric)
    assert meta['execution']=='completed'
    frame=pd.read_csv(meta['csv_path'])
    if times[0]>0 or np.max(np.diff(times))>.15:
        assert (frame[frame.window_id==0].status=='missing_time_support').all()
    assert all(np.max(np.diff(t))<=.15 for t in calls)
    # A final partial window is dropped rather than padded or scored.
    assert (frame.stop_s<=times[-1]+.1+1e-7).all()


def test_record_failure_and_resume_preserve_terminal_evidence(tmp_path,monkeypatch):
    def broken(*args):
        raise ValueError('fixture source failure')
    record=fixture_record()
    meta=runner.evaluate_record(record,[],runner.load_config(),tmp_path,reader=broken)
    assert meta['execution']=='failed' and meta['error_type']=='ValueError'
    monkeypatch.setattr(runner,'load_native_fnirs_record',broken)
    result=runner.execute_records([record],{},runner.load_config(),tmp_path,resume=True)
    assert result[0]['error']=='fixture source failure'
    with pytest.raises(ValueError,match='already exists'):
        runner.execute_records([record],{},runner.load_config(),tmp_path)


def test_summary_exact_weights_and_failures_use_separate_denominators(tmp_path):
    config=deepcopy(runner.load_config());config['census']['lengths_s']=[30]
    config['summary']['bootstrap_replicates']=20
    records=[fixture_record('a','r1'),fixture_record('a','r2'),fixture_record('b','r1'),fixture_record('c','failed')]
    inventory=[runner.record_identity(record) for record in records]
    (tmp_path/'record_inventory.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in inventory))
    data=[[(0,0.),(0,10.),(1,20.)],[(0,100.)],[(0,200.)]]
    for record,pairs in zip(records,data):
        path,meta_path,_=runner._paths(tmp_path,record);path.parent.mkdir(exist_ok=True)
        frame=pd.DataFrame([dict(length_s=30,window_id=window,status='ok',nominal_D=x,
            best_D=x/2,best_tau=2.,improvement=x/2) for window,x in pairs])
        frame.to_csv(path,index=False,compression='gzip')
        runner.write_json(meta_path,dict(**runner.record_identity(record),execution='completed',csv_path=str(path),
            rows=len(frame),status_counts={'ok':len(frame)}))
    runner.write_json(runner._paths(tmp_path,records[-1])[1],dict(**inventory[-1],execution='failed'))
    result=runner.summarize(tmp_path,config)
    summary=pd.read_csv(tmp_path/'summary.csv')
    row=summary[(summary.dataset_id=='fixture')&(summary.metric=='nominal_D')].iloc[0]
    assert row['mean']==pytest.approx(128.125)
    assert row.q50==100. and row.q90==200.
    assert row.planned_subjects==3 and row.valid_subjects==2 and row.failed_records==1
    assert row.raw_median==20.
    subjects=pd.read_csv(tmp_path/'subject_summary.csv')
    failed=subjects[(subjects.subject_id=='c')&(subjects.metric=='nominal_D')].iloc[0]
    assert failed.valid_records==0 and np.isnan(failed['mean'])
    assert result['unknown_native_window_denominator_records']==['fixture|c|failed']


def test_existing_resolved_contract_cannot_be_overwritten(tmp_path):
    config=runner.load_config();runner.prepare_output(tmp_path,config)
    changed=deepcopy(config);changed['metric']['rho']=.4
    with pytest.raises(ValueError,match='different resolved contract'):
        runner.prepare_output(tmp_path,changed)


def test_all_failed_dataset_keeps_NA_summary_and_readable_empty_tables(tmp_path):
    config=deepcopy(runner.load_config());config['census']['lengths_s']=[30]
    identity=runner.record_identity(fixture_record())
    (tmp_path/'record_inventory.jsonl').write_text(json.dumps(identity)+'\n')
    runner.write_json(tmp_path/'records'/f"{identity['record_token']}.meta.json",
        dict(**identity,execution='failed'))
    result=runner.summarize(tmp_path,config)
    frame=pd.read_csv(tmp_path/'summary.csv')
    assert len(frame)==8 and (frame.valid_subjects==0).all()
    assert frame.q50.isna().all() and (frame.failed_records==1).all()
    assert pd.read_csv(tmp_path/'record_summary.csv').empty
    assert pd.read_csv(tmp_path/'distribution_ecdf.csv.gz').empty
    assert result['failed_records']==1
