"""Public cache software contracts; no local measured or sealed artifacts."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.scripts import evaluate_shared_driver_reconstruction as runner
from src.data.ssm_prepared import fit_feature_coordinate
from src.data.tak_mismatch import tak_reconstruction_windows


def record(dataset='refed', duration=185., offset=3.03):
    return dict(dataset=dataset,subject='s',record='r',join_key=dataset+'|s|r',
        cache_key=dataset+'/s/r',split='train',status='available',hb_only_admitted=True,
        coverage={'segments':[dict(eeg_start_s=0.,eeg_end_s=duration,
            hb_start_s=offset,hb_end_s=offset+duration,offset_s=offset,
            alignment_support_ms={'eeg':[0,duration*1000],'fnirs':[offset*1000,(offset+duration)*1000]})]},
        continuous_hb_segments=[dict(start_s=offset,end_s=offset+duration)])


def test_subject_calibration_keeps_population_direction_and_preserves_scale_inverse():
    rng=np.random.default_rng(8)
    x=rng.normal(size=(2000,30));x[:,0]*=4
    h=rng.normal(size=(2000,2))*[2.,.5]
    population=fit_feature_coordinate([(x,h)])
    own=fit_feature_coordinate([(x*7,h*3)],fixed_pc=population['pc'])
    np.testing.assert_array_equal(own['pc'],population['pc'])
    assert own['eeg_factor']==pytest.approx(population['eeg_factor']/7)
    assert own['hb_factor']==pytest.approx(population['hb_factor']/3)
    np.testing.assert_allclose(own['sd'],population['sd'],rtol=1e-12)
    transformed=np.column_stack((x*7@own['pc']*own['eeg_factor'],h*3*own['hb_factor']))
    inverse=transformed/np.array([own['eeg_factor'],own['hb_factor'],own['hb_factor']])
    np.testing.assert_allclose(inverse,np.column_stack((x*7@population['pc'],h*3)))
    changed=x.copy();changed[:,1]*=20
    kept=fit_feature_coordinate([(changed,h)],fixed_pc=population['pc'])
    np.testing.assert_array_equal(kept['pc'],population['pc'])
    assert abs(fit_feature_coordinate([(changed,h)])['pc']@population['pc'])<.2
    with pytest.raises(ValueError,match='unit vector'):
        fit_feature_coordinate([(x,h)],fixed_pc=population['pc']*2)


def test_normalization_comparison_keeps_weights_and_objective_sd_fixed(tmp_path):
    cfg=runner.normalization_comparison_config()
    assert json.loads(json.dumps(cfg))==cfg
    assert cfg['controls']['explicit_component_weights']==[1.,1.,1.]
    assert runner.normalization_comparison_check(cfg,tmp_path)['status']=='passed'
    old=dict(pc=[1.,0.],eeg_factor=2.,hb_factor=3.,sd=[.025,.03,.01])
    own=dict(pc=[0.,1.]);lo=dict(eeg_factor=5.,hb_factor=7.);ln=dict(eeg_factor=11.,hb_factor=7.)
    arms=runner.normalization_comparison_coordinates(old,own,lo,ln)
    assert arms['individual_pca']['eeg_factor']==2.
    assert arms['local_scale']['eeg_factor']==5.
    assert arms['individual_pca_local_scale']['eeg_factor']==11.
    for co in arms.values():np.testing.assert_array_equal(co['sd'],old['sd'])


def test_normalization_selection_ignores_reconstruction_outcomes_and_overlap():
    rows=[]
    for subject,ratio in [('high',5.),('medium',1.),('low',.2),('other',.6)]:
        for k in range(12):
            rows.append(dict(subject=subject,available=True,input_hb_eeg_ratio=ratio*(1+k*.01),
                qualified=k%2==0,eeg_local_nrmse=1+k,
                window=dict(key=f'w{k}',hb_start_s=k*60.,eeg_start_s=k*60.,duration_s=90.)))
    a=runner.normalization_comparison_select(rows)
    changed=[dict(r,eeg_local_nrmse=1e6-r['eeg_local_nrmse'],qualified=not r['qualified']) for r in rows]
    b=runner.normalization_comparison_select(changed)
    assert a['selected_subjects']==b['selected_subjects']==['high','medium','low']
    assert [(r['subject'],r['window']['key']) for r in a['panel']]==[(r['subject'],r['window']['key']) for r in b['panel']]
    assert set(a['nonoverlapping_counts'].values())=={6}


def test_normalization_metrics_separate_input_mismatch_and_prediction_amplification():
    t=np.arange(100)/10.;y=np.column_stack((np.sin(t),4*np.cos(t),2*np.sin(t)))
    co=dict(eeg_factor=.1,hb_factor=.2,sd=[.025,.03,.01])
    p=y.copy();p[:,0]*=5;p[-1]=1e12
    m=runner.normalization_comparison_metrics(y,p,co)
    assert m['eeg_amplitude_ratio']==pytest.approx(5)
    assert m['eeg_correlation']==pytest.approx(1)
    assert m['hb_rmse']==0
    expected=np.sqrt(np.mean((y[:-1,1:].std(0)*.2)**2))/(y[:-1,0].std()*.1)
    assert m['input_hb_eeg_ratio']==pytest.approx(expected)




def test_windows_cover_complete_support_without_crossing_gaps_and_short_refed_is_fitted():
    r=record();windows,excluded=tak_reconstruction_windows(r)
    assert len(windows)==3 and not excluded
    assert all(w['duration_s']==90 and not w['short_record'] for w in windows)
    assert windows[0]['hb_start_s']==3.1
    assert windows[-1]['hb_start_s']+90==188.
    assert all(w['eeg_start_s']>=0 and w['eeg_start_s']+90<=185 for w in windows)
    short=record(duration=45.);windows,excluded=tak_reconstruction_windows(short)
    assert len(windows)==1 and not excluded
    assert windows[0]['short_record'] and windows[0]['duration_s']==44.9
    windows,excluded=tak_reconstruction_windows(record('eeg_fnirs_single_trial',45.))
    assert not windows and excluded[0]['reason']=='paired_support_shorter_than_90s'
    short=record(duration=4.)
    assert tak_reconstruction_windows(short)[1][0]['reason']=='support_does_not_extend_beyond_5s_reference'
    gap=record()
    gap['continuous_hb_segments']=[dict(start_s=3.03,end_s=70.),dict(start_s=95.,end_s=188.03)]
    windows,excluded=tak_reconstruction_windows(gap)
    assert all(w['hb_start_s']>=95 for w in windows)
    assert excluded


def test_refed_shorter_than_30_seconds_does_not_inherit_the_30s_diagnostic_exclusion():
    r=record(duration=20.)
    r['status']='hb_only'
    r['excluded_reason']='no_complete_30s_paired_support'
    windows,excluded=tak_reconstruction_windows(r)
    assert len(windows)==1 and windows[0]['duration_s']==19.9 and not excluded
    r['coverage']['segments']=[]
    assert not tak_reconstruction_windows(r)[0]


def test_sample_join_requires_both_clocks_and_chooses_one_context_without_labels():
    r=record(offset=0.)
    r['cache_windows'],_=tak_reconstruction_windows(r)
    sample=dict(id='sample',eeg_start_s=65.,hb_start_s=65.,duration_s=30.,condition='A')
    joined=runner.tak_cache_sample_join(sample,r)
    assert joined['window_key']=='w00001' and joined['eeg_offset_s']==5.
    assert runner.tak_cache_sample_join(dict(sample,condition='B'),r)==joined
    assert runner.tak_cache_sample_join(dict(sample,hb_start_s=180.),r)['status']=='unavailable'


def test_90s_and_short_observation_operators_match_central_clock_contract():
    from src.inference.observation_baselines import disjoint_native_bin_operator
    for n in (900,450,199):
        central=disjoint_native_bin_operator(np.arange(n)/10.,steps=n,dt=.1,reference_steps=50)
        operator=runner._tak_cache_operator(n)
        np.testing.assert_allclose(operator,np.kron(central['model'],np.eye(3)),atol=1e-12)
        target=np.zeros((n,3));prediction=np.tile([1.,2.,3.],(n,1));prediction[-1]=1e12
        assert runner._tak_cache_errors(target,prediction,np.ones(3))==dict(
            eeg_nmse=1.,hbo_nmse=4.,hbr_nmse=9.,hb_nmse=6.5)


def test_changed_long_window_is_rejected_before_native_or_hb_reads(monkeypatch,tmp_path):
    import src.data.tak_mismatch as data
    r=record(offset=0.)
    r['cache_windows'],_=tak_reconstruction_windows(r)
    r['cache_windows'][0]['duration_s']=91.
    monkeypatch.setattr(data,'_validated_tak_record',lambda *a,**k:(object(),['E'],None,[]))
    monkeypatch.setattr(data,'load_native_eeg_record',lambda *a:pytest.fail('native read before window validation'))
    monkeypatch.setattr(data,'load_cached_fnirs_record',lambda *a:pytest.fail('Hb read before window validation'))
    with pytest.raises(ValueError,match='windows changed'):
        list(data.iter_tak_reconstruction_record(tmp_path,r))


def test_population_pca_uses_training_only_while_individual_scales_use_only_own_subject(tmp_path):
    rng=np.random.default_rng(27);records=[]
    for subject,split,multiplier in [('train','train',1.),('test','test',20.)]:
        r=record('fixture',90.,0.);r.update(subject=subject,split=split,
            cache_key='fixture/'+subject+'/r',join_key='fixture|'+subject+'|r')
        r['cache_windows'],_=tak_reconstruction_windows(r)
        r['pairs']=[dict(status='available',native_index=0,channel_sets={'nearest6':list(range(6))})]
        path=tmp_path/'features'/r['cache_key']/'w00000.npz'
        x=rng.normal(size=(900,6,5))*multiplier;h=rng.normal(size=(900,1,2))*multiplier
        if split=='train':train_x=x.copy();train_h=h.copy()
        runner._tak_cache_npz(path,eeg=x,hb=h,eeg_valid=np.ones(6,bool),hb_valid=np.ones(1,bool))
        runner.write_json(path.with_suffix('.json'),dict(status='completed'))
        records.append(r)
    cfg=dict(schema='fixture',data={'datasets':['fixture']},calibration={'projection':'training_only'})
    result=runner.tak_cache_calibrate(cfg,tmp_path,dict(records=records))
    expected=fit_feature_coordinate([(train_x.reshape(900,30),train_h[:,0])])
    np.testing.assert_allclose(result['population']['fixture']['pc'],expected['pc'])
    assert result['population']['fixture']['subjects']==['train']
    own=result['individual']['fixture__test']
    assert {s['subject'] for s in own['sources']}=={'test'}
    np.testing.assert_array_equal(own['pc'],expected['pc'])
    assert own['eeg_factor']<expected['eeg_factor']/10


def test_individual_pca_uses_only_own_subject_and_freezes_across_records(tmp_path):
    cfg,_=runner.tak_cache_config()
    assert cfg['schema']=='shared_driver_tak_subject_cache_v2'
    cfg=deepcopy(cfg);cfg['data']['datasets']=['fixture']
    rng=np.random.default_rng(61);records=[];chunks={};paths={}
    for subject,split,axis in [('train','train',0),('heldout','test',12)]:
        chunks[subject]=[]
        for repeat in range(2):
            r=record('fixture',90.,0.)
            r.update(subject=subject,split=split,record=f'r{repeat}',
                cache_key=f'fixture/{subject}/r{repeat}',join_key=f'fixture|{subject}|r{repeat}')
            r['cache_windows'],_=tak_reconstruction_windows(r)
            r['pairs']=[dict(status='available',native_index=0,channel_sets={'nearest6':list(range(6))})]
            x=rng.normal(size=(900,30));x[:,axis]*=10
            h=rng.normal(size=(900,2))*[2.,.5]
            path=tmp_path/'features'/r['cache_key']/'w00000.npz'
            runner._tak_cache_npz(path,eeg=x.reshape(900,6,5),hb=h[:,None],
                eeg_valid=np.ones(6,bool),hb_valid=np.ones(1,bool))
            runner.write_json(path.with_suffix('.json'),dict(status='completed'))
            records.append(r);chunks[subject].append((x,h));paths[(subject,repeat)]=path
    result=runner.tak_cache_calibrate(cfg,tmp_path,dict(records=records))
    own=result['individual']['fixture__heldout']
    expected=fit_feature_coordinate(chunks['heldout'])
    np.testing.assert_array_equal(own['pc'],expected['pc'])
    assert own['eeg_factor']==expected['eeg_factor']
    assert own['PCA_source']=='fixture__heldout'
    assert len({s['record_key'] for s in own['sources']})==2
    assert {s['subject'] for s in own['sources']}=={'heldout'}
    assert abs(own['pc']@result['population']['fixture']['pc'])<.1
    # Change another person's covariance: neither this PC nor its scales may change.
    for repeat in range(2):
        x,h=chunks['train'][repeat];changed=x.copy();changed[:,20]*=100
        runner._tak_cache_npz(paths[('train',repeat)],eeg=changed.reshape(900,6,5),hb=h[:,None],
            eeg_valid=np.ones(6,bool),hb_valid=np.ones(1,bool))
    repeated=runner.tak_cache_calibrate(cfg,tmp_path,dict(records=records),destination=tmp_path/'recalibrated.json')
    same=repeated['individual']['fixture__heldout']
    for key in ('pc','eeg_factor','hb_factor','sd'):
        np.testing.assert_array_equal(same[key],own[key])
    assert abs(repeated['population']['fixture']['pc']@result['population']['fixture']['pc'])<.1
    # The persisted calibration is reused exactly, without rereading features.
    assert runner.serial(runner.tak_cache_calibrate(cfg,tmp_path,dict(records=records)))==runner.serial(result)
    old,_=runner.tak_cache_config(runner.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_subject_cache_v1.yaml')
    with pytest.raises(ValueError,match='Frozen calibration'):
        runner.tak_cache_calibrate(old,tmp_path,dict(records=records))
    specs=list(runner.tak_cache_specs(records,individual_pca=True))
    assert all(s['output_coordinates']=='individual_PC1_logpower_and_referenced_cached_Hb' for s in specs)


def test_cache_versions_keep_prior_contract_and_reject_unreviewed_projection(tmp_path):
    import yaml
    new,core=runner.tak_cache_config()
    old,old_core=runner.tak_cache_config(runner.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_subject_cache_v1.yaml')
    assert old['calibration']['projection']=='one_training_subject_only_PC1_per_dataset_all_available_sites'
    assert new['calibration']['projection']=='one_PC1_per_dataset_subject_all_available_sites'
    for key in ('data','windows','model','storage'):
        assert new[key]==old[key]
    for key in ('arms','solver','parameters'):
        assert core[key]==old_core[key]
    assert new['run_name']!=old['run_name']
    bad=deepcopy(new);bad['calibration']['projection']=old['calibration']['projection']
    path=tmp_path/'mixed.yaml';path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError,match='reviewed'):
        runner.tak_cache_config(path)
    checks=runner.tak_cache_check(new,core,tmp_path)
    assert 'individual_PCA_and_scaling' in checks['checks']


def test_existing_terminal_task_with_wrong_identity_is_not_reused(tmp_path):
    cfg,core=runner.tak_cache_config()
    spec=dict(record_key='fixture/s/r',window={'key':'w00000'},pair={'native_index':0})
    path=tmp_path/'tasks/fixture/s/r/w00000__p000.json'
    runner.write_json(path,dict(schema=cfg['schema'],identity={'different':True}))
    with pytest.raises(ValueError,match='identity'):
        runner._tak_cache_fit_worker((cfg,core,str(tmp_path),'unused',str(tmp_path),spec))


def test_cache_model_explicitly_uses_tau_beta_and_keeps_base_clock_and_penalties():
    cfg,core=runner.tak_cache_config();base=runner.tak_config(runner.DEFAULT)
    assert core['arms'][1]['fit']==['tau','neurovascular_gain']
    assert core['arms'][1]['log_sd']==[.125,.125]
    assert core['solver']==base['solver'] and core['parameters']==base['parameters']
    assert core['tensor']['dt_s']==.1 and cfg['windows']['duration_s']==90
    assert 'REFED_actual' in cfg['windows']['short_support']


def test_execution_capacity_respects_ancestor_quota_and_preserves_frozen_model(tmp_path,monkeypatch):
    root=tmp_path/'cgroup';child=root/'user.slice/worker.service';child.mkdir(parents=True)
    (root/'cpu.max').write_text('max 100000')
    (root/'user.slice/cpu.max').write_text('8000000 100000')
    (child/'cpu.max').write_text('max 100000')
    membership=tmp_path/'membership';membership.write_text('0::/user.slice/worker.service\n')
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:set(range(104)))
    before=runner.tak_cache_config()
    assert runner.tak_cache_worker_capacity(cgroup_root=root,membership_path=membership)==80
    (child/'cpu.max').write_text('4800000 100000')
    assert runner.tak_cache_worker_capacity(cgroup_root=root,membership_path=membership)==48
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:set(range(24)))
    assert runner.tak_cache_worker_capacity(cgroup_root=root,membership_path=membership)==24
    assert runner.tak_cache_config()==before


def test_execution_capacity_rejects_cgroup_path_escape(tmp_path,monkeypatch):
    membership=tmp_path/'membership';membership.write_text('0::/../../elsewhere\n')
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:set(range(64)))
    with pytest.raises(ValueError,match='cgroup path'):
        runner.tak_cache_worker_capacity(cgroup_root=tmp_path/'cg',membership_path=membership)
