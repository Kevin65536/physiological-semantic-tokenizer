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
