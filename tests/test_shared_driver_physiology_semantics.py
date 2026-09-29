"""Public-surface campaign contracts; no ignored or measured input fixtures."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.scripts import evaluate_shared_driver_reconstruction as runner


CONFIG = Path(__file__).resolve().parents[1]/'experiments/configs/physiology_semantic_tokenizer/shared_driver_physiology_semantics_v1.yaml'


def test_absolute_agreement_penalizes_systematic_repeat_offset():
    x=np.arange(1.,8.)
    assert runner.semantics_icc_absolute(np.column_stack([x,x]))==pytest.approx(1.)
    shifted=np.column_stack([x,x+20])
    assert np.corrcoef(shifted.T)[0,1]==pytest.approx(1.)
    assert runner.semantics_icc_absolute(shifted)<.1
    assert runner.semantics_icc_absolute(np.ones((5,2))) is None
    assert runner.semantics_icc_absolute(np.ones((2,2))) is None


def test_subject_record_block_bootstrap_does_not_treat_duplicate_windows_as_new_evidence():
    rows=[dict(subject=f'S{s}',region='motor',site='motor',record=f'r{r}',eeg_start_s=30.*w,
        delta=(s-1)*.1+(r-1)*.2+(w//4)*.05) for s in range(4) for r in range(3) for w in range(8)]
    frame=pd.DataFrame(rows)
    a=runner.semantics_block_bootstrap_gain(frame,37,300)
    b=runner.semantics_block_bootstrap_gain(pd.concat([frame,frame],ignore_index=True),37,300)
    np.testing.assert_allclose(a,b,atol=1e-15)
    assert a[0]<a[1]
    constant=frame.copy();constant['delta']=.25
    np.testing.assert_allclose(runner.semantics_block_bootstrap_gain(constant,37,300),[.25,.25])


def test_continuous_context_keeps_identical_endpoint_and_native_clock(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.data import clean_physiology_cache as cache,unified_physiology as unified
    from src.inference import shared_driver_reconstruction as inference
    cfg=runner.semantics_config(CONFIG);key='fixture__subject__record'
    ref=dict(id='fixture_window',key=key,dataset='fixture',subject='subject',record='record',site='prefrontal',region='prefrontal',
        event_id=0,eeg_start_s=60.,hb_start_s=60.,subject_fold=0,eeg_channels=[f'c{i}' for i in range(6)])
    record=SimpleNamespace(join_key='fixture|subject|record')
    event=dict(event_index=0,eeg_time_ms=65000.,fnirs_time_ms=65000.,metadata={})
    monkeypatch.setattr(cache,'CleanPhysiologyCacheIndex',lambda _:SimpleNamespace(records=[record],events_by_join_key={record.join_key:[event]}))
    rng=np.random.default_rng(22)
    eeg=SimpleNamespace(sample_rate_hz=200.,channel_names=ref['eeg_channels'],values=rng.normal(size=(36000,6)))
    t=np.arange(1800)/10.;t+=.002*np.sin(t)
    native=dict(time_s=t,values=np.stack([np.sin(t/4),np.cos(t/5)],axis=-1)[:,None,:])
    monkeypatch.setattr(unified,'load_native_eeg_record',lambda *_:eeg)
    monkeypatch.setattr(unified,'load_native_fnirs_record',lambda *_:native)
    folder=tmp_path/'prepared'/key;folder.mkdir(parents=True)
    (folder/'record.json').write_text(json.dumps(dict(prepared=[dict(region='prefrontal',hb_pair=0)])))
    folder=tmp_path/'coordinates';folder.mkdir()
    (folder/'fixture__prefrontal__outer0.json').write_text(json.dumps(dict(pc=[1.]+[0.]*29,eeg_factor=1.,hb_factor=1.,sd=[1.,1.,1.])))
    (tmp_path/'training_plan.json').write_text(json.dumps(dict(qualified_parameters=[])))
    targets=[]
    def fake_fit(target,*_,**kwargs):
        assert 'mean_operator' not in kwargs
        targets.append(target.copy());n=len(target)
        return dict(prediction=np.zeros_like(target),driver=np.zeros(n),states=np.ones((n,6)),initial_state=np.ones(5),converged=True,status='test_only')
    monkeypatch.setattr(inference,'fit_nonlinear_shared_driver',fake_fit)
    result=runner.semantics_context_worker((cfg,str(tmp_path),str(tmp_path),ref))
    assert [len(x) for x in targets]==[120,240,480]
    np.testing.assert_array_equal(targets[0],targets[1][60:180])
    np.testing.assert_array_equal(targets[0],targets[2][180:300])
    assert result['status']=='completed'
    assert result['native_clock_step_quantiles_s'][0]<.1<result['native_clock_step_quantiles_s'][-1]


def test_versioned_contract_and_truth_grid(tmp_path):
    cfg=runner.semantics_config(CONFIG)
    runner.semantics_parameters(cfg).validate()
    specs=runner.semantics_synthetic_specs(cfg)
    assert len(specs)==288 and len({s['key'] for s in specs})==288
    assert {s['initial'] for s in specs}=={'rest','nonrest'}
    bad=deepcopy(cfg);bad['data']['motion_method']='mne';path=tmp_path/'bad.yaml'
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError):runner.semantics_config(path)


def make_cohort(tmp_path):
    rng=np.random.default_rng(5);records=[]
    for subject in ('S01','S02','S03','S04','S05','S06'):
        for probe in (1,2):
            key=f'{subject}_Probe{probe}';folder=tmp_path/'prepared'/key;folder.mkdir(parents=True)
            windows=[dict(window=i,eeg_start_s=i*30.,hb_start_s=i*30.+2.,
                event_id=i,task='visual',condition='RR',independent_block='continuous') for i in range(6)]
            anchor=dict(region='prefrontal',eeg_channels=[f'c{i}' for i in range(6)],hb_pair=0,hb_channel='CH1',
                windows=windows,array_file='prefrontal.npz')
            spec=dict(key=key,dataset='visual_cognitive_motivation',subject=subject,record=key,
                repeat_group=subject,anchors=[anchor])
            records.append(spec)
            record=dict(prepared=[anchor],qc=dict(channels=[dict(rho=.4,status='supported',support=1,
                jump_fraction=[0,0],flat_step_fraction=[0,0])]))
            (folder/'record.json').write_text(json.dumps(record))
            eeg=rng.normal(size=(6,120,30));eeg-=eeg[:,:20].mean(axis=1,keepdims=True)
            hb=rng.normal(size=(6,120,2));hb-=hb[:,:20].mean(axis=1,keepdims=True)
            np.savez(folder/'prefrontal.npz',eeg_features=eeg,hb=hb)
    (tmp_path/'inventory.json').write_text(json.dumps(dict(records=records)))
    return runner.semantics_cohort(runner.semantics_config(CONFIG),tmp_path)


def test_native_records_without_part_have_disjoint_repeats_and_probe_grouping(tmp_path):
    refs,coordinates=make_cohort(tmp_path)
    assert len(refs)==72 and len(coordinates)==5
    for subject in sorted({r['subject'] for r in refs}):
        own=[r for r in refs if r['subject']==subject]
        for start in {r['eeg_start_s'] for r in own}:
            assert len({r['repeat_fold'] for r in own if r['eeg_start_s']==start})==1
        a=[r for r in own if r['repeat_fold']==0];b=[r for r in own if r['repeat_fold']==1]
        assert max(r['eeg_start_s']+30 for r in a)<=min(r['eeg_start_s'] for r in b)
        assert all(r['repeat_kind']=='within_record_disjoint_blocks' for r in own)
        for r in own:
            key=f'{r["dataset"]}__{r["site"]}__outer{r["subject_fold"]}'
            assert subject not in coordinates[key]['training_subjects']


def test_target_and_coordinate_do_not_fit_heldout_amplitudes(tmp_path):
    refs,coordinates=make_cohort(tmp_path);cfg=runner.semantics_config(CONFIG)
    held=refs[0];key=f'{held["dataset"]}__{held["site"]}__outer{held["subject_fold"]}'
    original=coordinates[key];train=[r for r in refs if r['subject_fold']!=held['subject_fold']]
    path=Path(held['array_path']);e,h=runner.semantics_prepared_arrays(str(path))
    np.savez(path,eeg_features=e*1000,hb=h-1e8);runner.semantics_prepared_arrays.cache_clear()
    revised=runner.semantics_coordinate(train,tmp_path,key+'_recheck',cfg)
    for k in ('pc','eeg_factor','hb_factor','sd'):np.testing.assert_array_equal(original[k],revised[k])
    target=runner.semantics_target(train[0],revised)
    assert target.shape==(120,3) and np.isfinite(target).all()
    np.testing.assert_allclose(target[:20].mean(axis=0),0,atol=1e-15)


def test_linear_reference_keeps_validity_separate_from_residual_bound(tmp_path):
    refs,_=make_cohort(tmp_path);cfg=runner.semantics_config(CONFIG)
    result=runner.semantics_linear_worker((cfg,str(tmp_path),refs[:2]))
    assert len(result['rows'])==4
    for identity in {r['identity'] for r in result['rows']}:
        pair={r['method']:r for r in result['rows'] if r['identity']==identity}
        low=pair['linear_unregularized'];regularized=pair['linear_regularized']
        assert np.sum(np.square(low['nrmse']))<=np.sum(np.square(regularized['nrmse']))+1e-12
        assert isinstance(low['physical_valid'],(bool,np.bool_))
        assert isinstance(low['small_signal_valid'],(bool,np.bool_))


def test_precision_audit_replays_frozen_driver_without_refitting(tmp_path):
    from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
    cfg=runner.semantics_config(CONFIG);parameters=runner.semantics_parameters(cfg)
    driver=.025*np.sin(np.arange(120)*.21);initial=np.r_[.01,1.02,1.01,1.01,1.]
    old=nonlinear_driver_forward(driver,initial,parameters,.25,substeps=4,derivative=False,numerical_backend='numba')
    fine=nonlinear_driver_forward(driver,initial,parameters,.25,substeps=8,derivative=False,numerical_backend='numba')
    operator=runner.semantics_model_operator();target=(operator@fine['canonical_prediction'].ravel()).reshape(120,3)
    prediction=(operator@old['canonical_prediction'].ravel()).reshape(120,3)
    ref=dict(key='fixture',site='motor',window=0,id='fixture_w0',dataset='fixture',subject='S0')
    path=tmp_path/'measured'/'fixture__motor'/'w0.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(rows=[dict(method='H0_fixed',parameter=None,parameter_value=None,converged=True,all_below_half=True)])))
    np.savez(path.with_suffix('.npz'),target=target,sd=np.ones(3)*.025,H0_fixed__prediction=prediction,
        H0_fixed__driver=driver,H0_fixed__initial_state=initial,H0_fixed__states=old['states'])
    result=runner.semantics_precision_worker((cfg,str(tmp_path),[ref]));row=result['rows'][0]
    assert row['fine_valid'] and row['all_below_half_8']
    assert 0<row['max_prediction_difference_sd']<1e-3
    np.testing.assert_allclose(row['nrmse_8'],0,atol=1e-14)


def test_processed_generator_has_matching_mean_contract():
    cfg=runner.semantics_config(CONFIG);spec=runner.semantics_synthetic_specs(cfg)[0]
    spec=dict(spec,coordinate='processed');a=runner.semantics_generate(cfg,spec)
    assert a['target'].shape==(24,120,3)
    assert a['states'].shape==(24,120,6)
    np.testing.assert_allclose(a['clean'][:,:20].mean(axis=1),0,atol=1e-13)
    assert np.linalg.matrix_rank(runner.semantics_model_operator())<360


def test_ridge_completion_hidden_value_intervention():
    rng=np.random.default_rng(17);train=rng.normal(size=(18,120,3));target=rng.normal(size=(120,3))
    for mode in ('center_EEG','center_Hb','HbO_hidden','HbR_hidden','EEG_hidden','Hb_hidden'):
        mask=runner.semantics_missing_mask(mode);changed=target.copy();changed[~mask]=1e12
        for cross in (False,True):
            a=runner.semantics_ridge_completion(train,target,mask,cross=cross)
            b=runner.semantics_ridge_completion(train,changed,mask,cross=cross)
            np.testing.assert_array_equal(a,b)
        assert (~mask).sum() in (16,32,120,240)


def test_observation_gauge_and_stress_pairing(tmp_path):
    cfg=runner.semantics_config(CONFIG)
    result=runner.semantics_observation_math(cfg,tmp_path)
    assert result['instantaneous_observation_rank']==3
    assert max(r['prediction_max_difference'] for r in result['gauge_examples'])<1e-13
    assert result['HbT_identity_max_error']<1e-13
    spec=dict(tau=2.,gain=1.,spectrum='slow',repeat=0,initial='nonrest',stress='control')
    control=runner.semantics_stress_generate(cfg,spec)
    perturbed=runner.semantics_stress_generate(cfg,dict(spec,stress='common_colored'))
    np.testing.assert_array_equal(control['driver'],perturbed['driver'])
    np.testing.assert_array_equal(control['states'],perturbed['states'])
    np.testing.assert_array_equal(control['target'][:,:,0],perturbed['target'][:,:,0])
    assert np.max(abs(control['target'][:,:,1:]-perturbed['target'][:,:,1:]))>1e-5


def test_hierarchy_truth_and_measurement_only_counterexample_are_distinct():
    cfg=runner.semantics_config(CONFIG)
    base=runner.semantics_hierarchy_generate(cfg,dict(mechanism='global',repeat=0))
    gain=runner.semantics_hierarchy_generate(cfg,dict(mechanism='hb_gain_only',repeat=0))
    assert base['target'].shape==(4*4*3*6,120,3)
    np.testing.assert_array_equal(base['truth_tau'],gain['truth_tau'])
    np.testing.assert_array_equal(base['driver'],gain['driver'])
    np.testing.assert_array_equal(base['states'],gain['states'])
    np.testing.assert_array_equal(base['target'][:,:,0],gain['target'][:,:,0])
    assert not np.allclose(base['target'][:,:,1:],gain['target'][:,:,1:])


def test_joint_summary_keeps_failed_fits_in_full_denominator(tmp_path):
    cfg=runner.semantics_config(CONFIG);specs=runner.semantics_synthetic_specs(cfg)[:3]
    for i,spec in enumerate(specs):
        path=tmp_path/'synthetic_joint'/spec['key']/'result.json';path.parent.mkdir(parents=True)
        row=dict(converged=i!=2,parameter_values=[spec['tau'],spec['gain']],relative_errors=[0.,0.])
        path.write_text(json.dumps(row))
    summary=runner.semantics_joint_summary(cfg,tmp_path,specs)
    assert summary['terminal']==3 and summary['success_rate']==2/3 and not summary['passed']
    assert (tmp_path/'synthetic_joint_metrics.csv').exists()


def test_profile_summary_does_not_promote_a_failed_parameter_fit(tmp_path):
    cfg=runner.semantics_config(CONFIG)
    ref=dict(id='fixture_w0',dataset='fixture',subject='S0')
    runner.write_json(tmp_path/'diagnostic_plan.json',dict(windows=[ref]))
    failed=dict(converged=False,parameter_values=[2.,1.],objective=1.)
    runner.write_json(tmp_path/'profiles'/ref['id']/'result.json',dict(joint=failed,
        profiles=[dict(profiled='tau',value=2.,converged=True,objective=1.)],
        sensitivity=[dict(kind='constant',parameter='alpha',value=.2,converged=True,parameter_values=[2.,1.],objective=1.)],
        trial=[dict(mode='first_10s_feature_calibration',parameter_fit=failed,
                    completion=dict(converged=True,hidden_nrmse=[None,.01,.01]))],prefix_baselines={}))
    runner.write_json(tmp_path/'context'/ref['id']/'result.json',dict(rows=[
        dict(method='H0_fixed',seconds=30,converged=True,nrmse=[.1,.1,.1],all_below_half=True,status='fixture')]))
    runner.write_json(tmp_path/'transfer'/ref['id']/'result.json',dict(rows=[
        dict(method='H0_fixed',tau=2.,converged=True,nrmse=[.1,.1,.1],all_below_half=True)]))
    result=runner.semantics_profile_summary(cfg,tmp_path)
    prefix=next(r for r in result['groups'] if r['panel']=='prefix')
    assert prefix['planned']==1 and prefix['success_rate']==0 and prefix['pass_rate']==0
    assert np.isnan(prefix['error'])
    sensitivity=pd.read_csv(tmp_path/'sensitivity_metrics.csv')
    assert sensitivity.base_tau.isna().all() and sensitivity.base_gain.isna().all()


def test_reconstruction_strata_preserve_failures_and_distinct_correlation_layers(tmp_path):
    cfg=runner.semantics_config(CONFIG);refs=[]
    methods=['H0_fixed']+[h+'__'+p for h in ['H1_dataset','H2_subject','H2s_partial_pooling','H3_record']
        for p in ['tau','neurovascular_gain']]
    for subject in range(3):
        key=f'fixture__S{subject}';rows=[]
        for window in range(4):
            ref=dict(id=f'{key}__motor__w{window}',key=key,dataset='fixture',subject=f'S{subject}',record='record',
                site='motor',region='motor',window=window,record_evaluation=True,repeat_fold=window%2,
                repeat_unit='record',eeg_start_s=30.*window,quality_burden=subject*.01)
            refs.append(ref);path=tmp_path/'measured'/(key+'__motor')/(f'w{window}.npz');path.parent.mkdir(parents=True,exist_ok=True)
            t=np.arange(120)/8;np.savez(path,target=np.column_stack([np.sin(t),np.cos(t),-np.cos(t)]),sd=np.ones(3))
            for method in methods:
                success=method!='H2_subject__tau'
                row={k:ref[k] for k in ('dataset','subject','record','site','region','window','record_evaluation','quality_burden')}
                row.update(identity=ref['id'],method=method,converged=success,all_below_half=success,
                    status='completed' if success else 'failed_training',native_rho=.9,
                    nrmse=[.1]*3,correlation=[.9]*3,amplitude_error=[.1]*3,peak_time_error_s=[0.]*3)
                rows.append(row)
        runner.write_json(tmp_path/'measured'/(key+'__motor')/'result.json',dict(rows=rows))
    runner.write_json(tmp_path/'cohort.json',dict(refs=refs))
    summary=runner.semantics_reconstruction_summary(cfg,tmp_path)
    assert summary['planned_method_windows']==108
    assert sum(r['planned_windows'] for r in summary['regional'])==108
    assert sum(r['planned_windows'] for r in summary['QC_strata'])==72
    for row in summary['QC_strata']:
        assert row['rho_group']==('rho>.8' if row['layer']=='native_record' else 'rho<-.5')
        if row['method']=='H2_subject__tau':assert row['success_rate']==row['pass_rate']==0
