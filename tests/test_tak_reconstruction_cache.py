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


def test_amplitude_frequency_inclusive_thresholds_and_explicit_failure_denominators(tmp_path):
    import pandas as pd
    cfg=runner.amplitude_frequency_config()
    assert cfg['splits']==['train','validation']
    assert runner.amplitude_frequency_check(cfg,tmp_path)['status']=='passed'
    frame=pd.DataFrame(dict(subject=['a','a','b','b'],terminal=[True,True,True,False],
        finite=[True,True,True,False],qualified=[True,True,False,False],
        input_ratio=[1.5,1.49,3.,np.nan],amplitude=[2.,3.,4.,np.nan],error=[1.,2.,3.,np.nan]))
    rates=runner.amplitude_frequency_rates(frame,cfg['thresholds'])
    assert rates['planned']==4 and rates['qualified']==2 and rates['unqualified_finite']==1 and rates['missing']==1
    assert rates['joint_count']==1 and rates['joint_rate']==.5
    assert rates['amplified_count']==2 and rates['joint_subjects']==1
    assert rates['subjects']==2 and rates['subjects_with_qualified']==1


def test_amplitude_frequency_rejects_test_record_before_any_task_read(tmp_path):
    cfg=runner.amplitude_frequency_config()
    with pytest.raises(ValueError,match='Disallowed split'):
        runner._amplitude_frequency_record((cfg,str(tmp_path),str(tmp_path),
            dict(dataset='refed',split='test'),{},1.))


def test_weight_SO_changes_only_objective_denominator_and_keeps_total_weight(monkeypatch):
    cfg=runner.weight_so_config();target=np.ones((900,3));sd=np.array([.025,.03,.01])
    co=dict(eeg_factor=2.,hb_factor=3.,sd=sd)
    core=dict(arms=[dict(name='tak_fixed',fit=[])])
    weight=runner.weight_so_weights(cfg['weights']['hbo4'])
    assert weight.sum()==pytest.approx(3)
    def captured(core,arm,y,operator,actual_sd,mask,warm,**kwargs):
        np.testing.assert_array_equal(y[0],target*[2,3,3])
        np.testing.assert_allclose(actual_sd[0],sd/np.sqrt(weight))
        assert mask[0,:-1].all() and not mask[0,-1].any()
        assert kwargs['observation_reference_steps']==50
        raise RuntimeError('captured_weighted_objective')
    monkeypatch.setattr(runner,'tak_fit_group',captured)
    with pytest.raises(RuntimeError,match='captured_weighted_objective'):
        runner.tak_cache_fit_arrays(core,target,co,sd,structured=True,observation_weights=weight)
    np.testing.assert_array_equal(co['sd'],sd)


def test_weight_native_reader_rejects_test_split_before_cache_access(tmp_path):
    from src.data.tak_mismatch import read_tak_weight_voltage_window
    with pytest.raises(ValueError,match='Development split'):
        read_tak_weight_voltage_window(tmp_path,{'split':'test'}, {},split='test',cache_root='missing',
            geometry_root='missing',refed_geometry_root='missing',visual_eeg_channel_identity=None)


def test_weight_SO_delta_metrics_distinguish_rejection_preservation_and_units():
    rng=np.random.default_rng(4);base=rng.normal(size=(100,3));delta=rng.normal(size=(100,3))
    assert runner.weight_so_delta_metrics(base,base,delta,kind='noise')['normalized_error']==0
    assert runner.weight_so_delta_metrics(base,base+delta,delta,kind='noise')['normalized_error']==pytest.approx(1)
    assert runner.weight_so_delta_metrics(base,base+delta,delta,kind='neural')['normalized_error']<1e-15
    assert runner.weight_so_delta_metrics(base,base,delta,kind='neural')['normalized_error']==pytest.approx(1)
    a=runner.weight_so_delta_metrics(base,base+.3*delta,delta,kind='neural')
    b=runner.weight_so_delta_metrics(base*1e-6,(base+.3*delta)*1e-6,delta*1e-6,kind='neural')
    assert a['normalized_error']==pytest.approx(b['normalized_error'])


def test_weight_SO_selected_arm_owns_qualification_and_prediction(tmp_path):
    import json
    path=tmp_path/'fit.json'
    path.write_text(json.dumps(dict(qualified=False,fits=dict(
        tak_fixed=dict(qualified=True,tau=2.,beta=1.),
        tak_tau_beta=dict(qualified=False,tau=3.,beta=.5)))))
    np.savez(path.with_suffix('.npz'),fixed_prediction=np.full((900,3),2.),prediction=np.full((900,3),3.))
    pred,record=runner.weight_so_prediction(path,'tak_fixed')
    np.testing.assert_array_equal(pred,np.full((900,3),2.));assert record['tau']==2.
    with pytest.raises(ValueError,match='Unqualified SSM'):
        runner.weight_so_prediction(path,'tak_tau_beta')
    with pytest.raises(ValueError,match='Unknown SSM arm'):
        runner.weight_so_prediction(path,'unknown')


def test_tau_beta_SO_retained_boundary_rejects_protected_before_signal_reads():
    cfg=runner.tau_beta_so_config()
    with pytest.raises(ValueError,match='Disallowed split'):
        runner.tau_beta_so_validate_plan(dict(schema=cfg['source_schema'],rows=[dict(split='test')]),cfg)


def test_cache_ocular_injection_has_exact_frozen_RMS_and_compact_support():
    x=np.tile(np.sin(np.arange(18000)/200)[:,None],(1,3));donor=np.cos(np.arange(18000)/31)
    before=x.copy();n,meta=runner.cache_ocular_injection(x,200,['FP1','AF3','O1'],donor,.5)
    np.testing.assert_array_equal(x,before)
    assert not n[:6000].any() and not n[10000:].any()
    assert np.sqrt(np.mean(n[6000:10000]**2))/np.sqrt(np.mean(x[6000:10000]**2))==pytest.approx(.5)
    np.testing.assert_allclose(n[:,1],.65*n[:,0]);np.testing.assert_allclose(n[:,2],.08*n[:,0])
    assert meta['electrode']==0
    with pytest.raises(ValueError):runner.cache_ocular_injection(x,200,['FP1','AF3','O1'],donor,0)


def test_cache_ocular_selection_is_metadata_only_and_excludes_previous_and_test():
    cfg=runner.cache_ocular_config();cfg['selection'].update(expected_subjects=1,expected_windows=2)
    template=dict(dataset='fixture',split='train',record_plan=dict(eeg_channel_names=['F1']))
    reference=dict(datasets={'fixture':dict(validation_subjects=['s1'],pairs=['P'])},rows=[template])
    records=[]
    for split,subject in [('validation','s1'),('validation','s2'),('test','s3')]:
        for rec in ['a','b','c']:
            records.append(dict(dataset='fixture',subject=subject,split=split,record=rec,cache_key=subject+'/'+rec,
                eeg_channel_names=['F1'],pairs=[dict(id='P',status='available')],
                cache_windows=[dict(key=str(i),duration_s=90) for i in range(3)]))
    selected,counts=runner.cache_ocular_select(dict(records=records),reference,cfg)
    assert counts=={'fixture':1} and [r['record_plan']['record'] for r in selected]==['a','c']
    assert all(r['subject']=='s2' and r['window']['key']=='1' for r in selected)


def test_modality_weight_metrics_distinguish_model_change_and_unchanged_Hb():
    cfg=runner.modality_weight_config();y=np.zeros((900,3));sd=np.array([1.,2.,4.])
    p=y.copy();p[300:850]=sd;observed=y.copy();observed[300:850,0]=2.
    result=runner.modality_weight_delta_metrics(y,y,observed,p,sd,cfg)
    assert result['HbO_change']==pytest.approx(1)
    assert result['HbR_late_mean']==pytest.approx(1)
    assert result['EEG_outside_RMS']==0 and result['HbO_baseline_NRMSE']==0
    assert result['proxy_transfer_ratio']==pytest.approx(.5)
    assert runner.modality_weight_delta_metrics(y,y,y,p,sd,cfg)['proxy_transfer_ratio'] is None


def test_modality_weight_selection_keeps_fidelity_guard_and_stops_without_candidate():
    import pandas as pd
    cfg=runner.modality_weight_config();rows=[]
    for weight in cfg['weights']:
        rows.append(dict(weight=weight,HbO_change=1.,HbR_change=1.,EEG_change=1.,
            EEG_baseline_NRMSE=.1,HbO_baseline_NRMSE=.1,HbR_baseline_NRMSE=.1,EEG_S_correlation=.9))
    frame=pd.DataFrame(rows);frame.loc[frame.weight=='hb2',['HbO_change','HbR_change','EEG_change']]=[.7,.9,.8]
    ds=pd.concat([frame.assign(dataset=d) for d in ('a','b')]);counts={w:dict(qualified_fits=36,qualified_targets=12) for w in cfg['weights']}
    assert runner.modality_weight_select(frame,ds,counts,cfg)['candidate']=='hb2'
    matched=frame.set_index('weight').loc['equal'].copy();matched['HbO_change']=.8
    matched_ds=ds[ds.weight=='equal'].set_index('dataset').copy();matched_ds['HbO_change']=.8
    paired=runner.modality_weight_select(frame,ds,counts,cfg,{'hb2':dict(aggregate=matched,datasets=matched_ds,ROIs=17)})
    assert paired['candidate'] is None and not paired['candidates'][0]['checks']['HbO_reduction']
    frame.loc[frame.weight=='hb2','EEG_baseline_NRMSE']=.151
    result=runner.modality_weight_select(frame,ds,counts,cfg)
    assert result['candidate'] is None and not result['continuation_required']
    assert not result['candidates'][0]['checks']['baseline_fidelity']


def test_tau_beta_SO_frozen_control_changes_input_but_holds_prediction(tmp_path,monkeypatch):
    import json
    import src.data.dual_component_tokenizer as voltage_module
    import src.inference.observation_baselines as observations
    for name in ('inputs','references','variants','fits'):(tmp_path/name).mkdir()
    case='synthetic';rate=200;t=np.arange(18000)/rate;x=np.sin(t)[:,None]
    de=np.zeros_like(x);de[6000:10000]=np.sin(3*t[6000:10000,None]);hb=np.zeros((900,3,2))
    (tmp_path/'plan.json').write_text(json.dumps(dict(rows=[dict(case=case,dataset='fixture',subject='one',coordinate=dict(sd=[1,1,1],hb_factor=1))])))
    np.savez(tmp_path/'inputs'/f'{case}.npz',x=x,rate=rate,unit='fixture')
    np.savez(tmp_path/'references'/'fixture.npz',voltage_rms=np.ones(1))
    (tmp_path/'variants'/f'{case}.json').write_text(json.dumps(dict(electrode=0,changed_rois=dict(baseline=[0,1,2],ocular1=[0]))))
    np.savez(tmp_path/'variants'/f'{case}.npz',delta_eeg_baseline=np.zeros_like(x),delta_eeg_ocular1=de,
        delta_hb_baseline=hb,delta_hb_ocular1=hb,hb_baseline=hb,hb_ocular1=hb)
    for variant,level in [('baseline',1.),('ocular1',9.)]:
        for roi in range(3):
            path=tmp_path/'fits'/f'{case}__equal__{variant}__r{roi}.json'
            path.write_text(json.dumps(dict(fits=dict(tak_fixed=dict(qualified=True,tau=2,beta=1)))))
            np.savez(path.with_suffix('.npz'),fixed_prediction=np.full((900,3),level))
    calls=[]
    def mock_projection(signal,model,proxy,**kwargs):
        calls.append((signal.copy(),proxy.copy()))
        return dict(s=.5*signal,o=.5*signal,closure_max_abs=0,final_reanalysis_ratio=np.ones(1))
    monkeypatch.setattr(voltage_module,'voltage_power_cap_target',mock_projection)
    monkeypatch.setattr(observations,'eeg_band_power',lambda *args,**kwargs:np.ones((900,1,5)))
    result=runner._weight_so_target((str(tmp_path),case,'equal',dict(arm='tak_fixed',freeze_baseline=True,
        variants=['baseline','ocular1'],destination=str(tmp_path/'frozen'),audit_interference=True)))
    assert result['qualified']==2
    np.testing.assert_allclose(calls[1][0]-calls[0][0],de,atol=1e-15)
    np.testing.assert_array_equal(calls[1][1],calls[0][1])
    metric=json.loads((tmp_path/'frozen'/case/'equal'/'ocular1.json').read_text())
    assert metric['noise_error']==pytest.approx(.5)
    assert metric['O_signed_gain']==pytest.approx(.5)
    assert metric['O_delta_correlation']==pytest.approx(1.)


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
    cfg,_=runner.tak_cache_config(runner.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_subject_cache_v2.yaml')
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
    new,core=runner.tak_cache_config(runner.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_subject_cache_v2.yaml')
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


def test_so_feature_audit_uses_training_only_scales_and_preserves_pairing(tmp_path):
    """A changed evaluation target must not redefine its own feature scale."""
    import pandas as pd
    from src.analysis.physiological_patch_features import supervision_signal_features
    cfg=runner.so_features_config();rows=[]
    # One synthetic dataset suffices to test the analysis mechanics; production
    # panel cardinality and split selection are enforced by so_features_plan.
    for i,role in enumerate(['training']*3+['evaluation']):
        case=f'fixture_{i}';folder=tmp_path/'cases'/case;folder.mkdir(parents=True)
        rows.append(dict(case=case,role=role,variants=['baseline','ocular1'] if role=='evaluation' else ['baseline']))
        records=[]
        for modality in ('EEG','HbO','HbR'):
            fs=200 if modality=='EEG' else 10;t=np.arange(fs*(20 if modality=='EEG' else 80))/fs
            freq=(7+i) if modality=='EEG' else (.05+.0125*i)
            x=np.sin(2*np.pi*freq*t)+(.1+.1*i)*np.cos(2*np.pi*freq*1.8*t)
            for condition in rows[-1]['variants']:
                for branch,gain in [('X',1.)] if role=='training' else [('X',1.),('S',.5),('O',.5)]:
                    feats,_,_=supervision_signal_features((x*gain)[:,None],fs,modality,np.ones(1))
                    rec={k:float(v[0]) for k,v in feats.items()};rec.update(case=case,dataset='fixture',subject=str(i),role=role,condition=condition,branch=branch,modality=modality,point_key=f'{case}|{modality}|0',channel='C',segment=0,anchor=True,correlation_to_X=1.,valid_shape=True)
                    rec.update(native_rms=rec['rms'],native_std=rec['std'],native_mean=rec['mean'],native_excess_kurtosis=rec['excess_kurtosis']);records.append(rec)
        pd.DataFrame(records).to_csv(folder/'features.csv',index=False)
        pd.DataFrame([dict(case=case)]).to_csv(folder/'Hb_relation.csv',index=False)
        runner.write_json(folder/'status.json',dict(status='completed'))
    runner.write_json(tmp_path/'plan.json',dict(rows=rows))
    runner.so_features_prepare_embeddings(cfg,tmp_path)
    before=json.loads((tmp_path/'feature_scalers.json').read_text());pairs=pd.read_csv(tmp_path/'paired_features.csv')
    np.testing.assert_allclose(pairs.rms_ratio,.5)
    np.testing.assert_allclose(pairs.core_power_ref_ratio,.25)
    np.testing.assert_allclose(pairs.query('condition=="baseline"').feature_distance_shape,0,atol=1e-8)
    assert (pairs.query('condition=="baseline"').feature_distance_shape_amplitude>0).all()
    folder=tmp_path/'cases'/'fixture_3';a=pd.read_csv(folder/'features.csv');a.loc[a.branch=='S','log_rms_ref']+=20;a.to_csv(folder/'features.csv',index=False)
    runner.so_features_prepare_embeddings(cfg,tmp_path)
    assert json.loads((tmp_path/'feature_scalers.json').read_text())==before
    after=pd.read_csv(tmp_path/'paired_features.csv')
    assert (after.query('condition=="baseline" and branch=="S"').feature_distance_shape_amplitude>pairs.query('condition=="baseline" and branch=="S"').feature_distance_shape_amplitude).all()


def test_explicit_cached_start_replaces_prefit_but_preserves_new_weighted_objective(monkeypatch):
    _,core=runner.tak_cache_config(runner.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_subject_cache_v2.yaml')
    calls=[]
    def record_fit(cfg,arm,y,operator,sd,mask,warm,**kwargs):
        calls.append(dict(arm=arm['name'],y=y.copy(),sd=sd.copy(),mask=mask.copy(),warm=warm,**kwargs))
        return dict(converged=False,status='synthetic_no_trajectory')
    monkeypatch.setattr(runner,'tak_fit_group',record_fit)
    target=np.ones((60,3));co=dict(eeg_factor=2.,hb_factor=3.,sd=[.025,.03,.02]);weights=runner.weight_so_weights([1,4,4])
    start=dict(parameter_values=np.array([2.3,.9]),driver=np.full((1,60),.02),initial_state=np.array([[0.,1.,1.,1.,1.]]))
    runner.tak_cache_fit_arrays(core,target,co,np.ones(3),structured=True,observation_weights=weights)
    assert [r['arm'] for r in calls]==['tak_fixed','tak_tau_beta'];cold=calls[-1];calls.clear()
    runner.tak_cache_fit_arrays(core,target,co,np.ones(3),structured=True,observation_weights=weights,initial_solution=start)
    assert len(calls)==1 and calls[0]['arm']=='tak_tau_beta'
    np.testing.assert_array_equal(calls[0]['y'],cold['y']);np.testing.assert_array_equal(calls[0]['sd'],cold['sd']);np.testing.assert_array_equal(calls[0]['mask'],cold['mask'])
    np.testing.assert_allclose(calls[0]['sd'][0],np.asarray(co['sd'])/np.sqrt(weights))
    assert calls[0]['starts_override'][0] is start
    assert not calls[0]['mask'][0,-1].any()


def test_warmstart_refuses_test_split_before_signal_read(tmp_path):
    cfg=runner.warmstart_config()
    assert json.loads(json.dumps(cfg))==cfg
    with pytest.raises(ValueError,match='Training-only'):
        runner._warmstart_fit_worker((str(tmp_path),dict(id='forbidden',split='test')))
    with pytest.raises(ValueError,match='heldout/protected'):
        runner._warmstart_prepare_worker(({},str(tmp_path),dict(split='test')))


def test_v3_default_uses_individual_pca_new_weights_and_same_physiological_penalties():
    cfg,core=runner.tak_cache_config()
    old,base=runner.tak_cache_config(runner.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_subject_cache_v2.yaml')
    assert cfg['schema']=='shared_driver_tak_subject_cache_v3'
    assert cfg['calibration']==old['calibration'] and cfg['windows']==old['windows']
    np.testing.assert_allclose(cfg['model']['observation_weights'],[1/3,4/3,4/3])
    assert core['solver']==base['solver'] and core['arms']==base['arms']
    assert cfg['data']['protected_data']=='forbidden'


def test_v3_retains_failed_warm_attempt_and_refits_standard(tmp_path,monkeypatch):
    cfg,core=runner.tak_cache_config();out=tmp_path/'new';source=tmp_path/'old';n=60
    spec=dict(record_key='fixture/s/r',dataset='fixture',subject='s',split='train',window={'key':'w00000'},pair=dict(native_index=0,status='available',channel_sets={'nearest6':list(range(6))}))
    old=source/'tasks/fixture/s/r/w00000__p000.json'
    runner.write_json(old,dict(schema='old',identity=spec,qualified=True,fits={'tak_tau_beta':dict(tau=2.,beta=1.)}))
    runner._tak_cache_npz(old.with_suffix('.npz'),driver=np.zeros(n),initial_state=np.array([0.,1.,1.,1.,1.]),time_s=np.arange(n)/10.)
    seed,info=runner.tak_cache_v3_seed(source,spec,n)
    assert info['status']=='available' and seed['driver'].shape==(1,n)
    changed=deepcopy(spec);changed['subject']='other'
    with pytest.raises(ValueError,match='identity'):runner.tak_cache_v3_seed(source,changed,n)
    co=dict(status='completed',pc=[1.]+[0.]*29,eeg_factor=1.,hb_factor=1.,sd=[1.,1.,1.]);coordinate=out/'coordinates.json'
    runner.write_json(coordinate,dict(individual={'fixture__s':co},population={'fixture':co}));runner.write_json(out/'source_reuse.json',dict(source=str(source)))
    feature=out/'features/fixture/s/r/w00000.npz';runner._tak_cache_npz(feature,eeg=np.zeros((n,6,5)),hb=np.zeros((n,1,2)),eeg_valid=np.ones(6,bool),hb_valid=np.ones(1,bool),eeg_reference=np.zeros((6,5)),hb_reference=np.zeros((1,2)))
    runner.write_json(feature.with_suffix('.json'),dict(status='completed',split='train',provenance={}))
    starts=[]
    def fake_fit(core,target,coordinate,common,**kwargs):
        starts.append(kwargs['initial_solution'] if 'initial_solution' in kwargs else None)
        return dict(qualified=starts[-1] is None,status='completed' if starts[-1] is None else 'failed_numerical',fits={'tak_tau_beta':{'solver':{'objective_components':{'data':0.}}}}),dict(target=target,prediction=target.copy(),residual=np.zeros_like(target),sd_subject=np.ones(3))
    monkeypatch.setattr(runner,'tak_cache_fit_arrays',fake_fit)
    result=runner._tak_cache_v3_fit_worker((cfg,core,str(out),str(coordinate),str(out),spec))
    assert result['qualified'] and result['route']=='standard_after_unqualified_warm'
    assert len(starts)==2 and starts[0] is not None and starts[1] is None
    attempt=out/'warm_attempts/fixture/s/r/w00000__p000.json'
    assert json.loads(attempt.read_text())['qualified'] is False and attempt.with_suffix('.npz').exists()
    row=json.loads((out/'tasks/fixture/s/r/w00000__p000.json').read_text());assert row['failed_warm_attempt']==str(attempt.relative_to(out))
    assert runner._tak_cache_v3_fit_worker((cfg,core,str(out),str(coordinate),str(out),spec))['cached']


def test_v4_compact_hb_preserves_sparse_native_identity_and_never_copies_excluded(tmp_path):
    cfg,_=runner.tak_cache_config(runner.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_subject_cache_v4.yaml')
    cfg=deepcopy(cfg);cfg['source_cache']['run']='parent'
    out=tmp_path/'new';out.mkdir();lower=tmp_path/'measurement';lower.mkdir()
    rng=np.random.default_rng(4);raw=rng.normal(size=(100,3,2));eeg=rng.normal(size=(60,6,5));indices=[0,2]
    np.save(lower/'fnirs.npy',raw[:,indices].reshape(100,4))
    selection=dict(retained_pair_indices=indices)
    runner.write_json(lower/'record.json',dict(fnirs_selection=selection,arrays={'fnirs':'fnirs.npy'}))
    w=dict(key='w00000',hb_start_s=1.,duration_s=6.)
    r=dict(cache_key='visual/s/r',measurement_record=str(lower),fnirs_selection=selection,pairs=[dict(native_index=i) for i in indices],cache_windows=[w],split='train')
    p=tmp_path/'parent/features/visual/s/r/w00000.npz';reference=raw[10:60].mean(0)
    runner._tak_cache_npz(p,eeg=eeg,eeg_reference=np.ones((6,5)),eeg_valid=np.ones(6,bool),hb=raw[10:70]-reference,hb_reference=reference,hb_valid=np.ones(3,bool))
    runner.write_json(p.with_suffix('.json'),dict(status='completed',window=w,split='train',provenance={'hb':{}}))
    runner._tak_cache_v4_prepare_worker((cfg,str(tmp_path),str(out),r))
    with np.load(out/'features/visual/s/r/w00000.npz') as a:
        assert a['hb'].shape==(60,2,2)
        np.testing.assert_array_equal(a['eeg'],eeg)
        np.testing.assert_allclose(a['hb'],(raw[10:70]-reference)[:,indices],atol=1e-12)
    r.update(dataset='visual',subject='s',record='r',join_key='visual|s|r')
    r['pairs']=[dict(native_index=2,compact_index=1,id='CH3',status='available',channel_sets={})]
    spec=next(runner.tak_cache_specs([r],individual_pca=True))
    assert spec['pair']['native_index']==2 and spec['pair']['compact_index']==1 and spec['pair']['id']=='CH3'


def test_v4_sparse_pair_warm_seed_and_target_use_different_indices(tmp_path,monkeypatch):
    cfg,core=runner.tak_cache_config(runner.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_subject_cache_v4.yaml')
    out=tmp_path/'new';source=tmp_path/'old';n=60
    spec=dict(record_key='visual/s/r',dataset='visual',subject='s',split='train',window={'key':'w00000'},pair=dict(native_index=12,compact_index=1,id='CH13',status='available',channel_sets={'nearest6':list(range(6))}))
    oldspec=deepcopy(spec);oldspec['pair'].pop('compact_index')
    old=source/'tasks/visual/s/r/w00000__p012.json'
    runner.write_json(old,dict(schema='v3',identity=oldspec,qualified=True,fits={'tak_tau_beta':dict(tau=2.,beta=1.)}))
    runner._tak_cache_npz(old.with_suffix('.npz'),driver=np.zeros(n),initial_state=np.array([0.,1.,1.,1.,1.]),time_s=np.arange(n)/10.)
    co=dict(status='completed',pc=[1.]+[0.]*29,eeg_factor=1.,hb_factor=.2,sd=[1.,1.,1.]);coordinate=out/'coordinates.json'
    runner.write_json(coordinate,dict(individual={'visual__s':co},population={'visual':co}));runner.write_json(out/'source_reuse.json',dict(source=str(source)))
    feature=out/'features/visual/s/r/w00000.npz';hb=np.arange(n*4,dtype=float).reshape(n,2,2)
    runner._tak_cache_npz(feature,eeg=np.zeros((n,6,5)),hb=hb,eeg_valid=np.ones(6,bool),hb_valid=np.ones(2,bool),eeg_reference=np.zeros((6,5)),hb_reference=np.zeros((2,2)))
    runner.write_json(feature.with_suffix('.json'),dict(status='completed',split='train',provenance={}))
    def fake_fit(core,target,coordinate,common,**kwargs):
        np.testing.assert_array_equal(target[:,1:],hb[:,1]);assert coordinate['hb_factor']==.2 and kwargs['initial_solution'] is not None
        return dict(qualified=True,status='completed',fits={'tak_tau_beta':{'solver':{'objective_components':{'data':0.}}}}),dict(target=target,prediction=target.copy(),residual=np.zeros_like(target),sd_subject=np.ones(3))
    monkeypatch.setattr(runner,'tak_cache_fit_arrays',fake_fit)
    result=runner._tak_cache_fit_worker((cfg,core,str(out),str(coordinate),str(out),spec))
    assert result['qualified'] and result['route']=='cached_free_start'
    assert (out/'tasks/visual/s/r/w00000__p012.json').exists()
    assert not (out/'tasks/visual/s/r/w00000__p001.json').exists()


def test_v4_keeps_v3_model_and_explicit_visual_only_migration():
    cfg,core=runner.tak_cache_config(runner.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_subject_cache_v4.yaml')
    old,base=runner.tak_cache_config()
    assert cfg['model']==old['model'] and cfg['windows']==old['windows']
    assert core['arms']==base['arms'] and core['solver']==base['solver']
    assert cfg['migration']['dataset']=='visual_cognitive_motivation'
    assert cfg['data']['cache_root'].endswith('_v6') and cfg['data']['downstream_input'].endswith('_v2')


def test_v4_pilot_retains_individual_nonconvergence_without_reclassifying_channel(tmp_path):
    cfg,core=runner.tak_cache_config(runner.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_subject_cache_v4.yaml')
    records=[]
    for sub in ['good','failed']:
        records.append(dict(dataset='visual_cognitive_motivation',subject=sub,split='train',cache_key='visual_cognitive_motivation/'+sub+'/r',join_key=sub,record='r',cache_windows=[dict(key='w00000')],pairs=[dict(id='CH3',native_index=2,compact_index=0,status='available',channel_sets={})]))
    specs=list(runner.tak_cache_specs(records,pilot=True,individual_pca=True));pilot=tmp_path/'pilot'
    runner.write_json(pilot/'plan.json',dict(specs=specs,selection=cfg['pilot']['selection']))
    runner.write_json(pilot/'summary.json',dict(status='failed',reason='previous_all_examples_success_requirement'))
    original=(pilot/'summary.json').read_bytes()
    for folder in ['cpu_standard','cpu_warm','cuda_warm']:
        runner.write_json(pilot/(folder+'_execution.json'),dict(seconds=1.))
        for spec in specs:
            path=pilot/folder/'tasks'/spec['record_key']/'w00000__p002.json';good=spec['subject']=='good'
            runner.write_json(path,dict(identity=spec,qualified=good,seconds=1.,fits={'tak_tau_beta':dict(tau=2.,beta=1.,solver={'objective':3.})}))
            if good:runner._tak_cache_npz(path.with_suffix('.npz'),target=np.zeros((60,3)),prediction=np.zeros((60,3)),sd_subject=np.ones(3))
    review=runner.tak_cache_v4_pilot(cfg,core,tmp_path,records,[0,1])
    assert review['status']=='passed' and len(review['retained_unqualified_examples'])==1
    assert review['standard_subjects']==['visual_cognitive_motivation__failed']
    assert (pilot/'summary.json').read_bytes()==original
    assert (pilot/'verification_v2.json').exists()
