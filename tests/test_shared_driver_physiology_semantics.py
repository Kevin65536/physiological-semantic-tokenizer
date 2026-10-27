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


def test_transfer_summary_includes_retained_target_panels_and_failed_denominators(tmp_path):
    out, old = tmp_path/'new', tmp_path/'retained'
    out.mkdir(); old.mkdir()
    datasets=['eeg_fnirs_single_trial','simultaneous_eeg_nirs','visual_cognitive_motivation','refed']
    refs=[dict(id=d+'__s1',dataset=d,subject='s1') for d in datasets]
    (out/'diagnostic_plan.json').write_text(json.dumps(dict(windows=refs[-1:])))
    (old/'diagnostic_plan.json').write_text(json.dumps(dict(windows=refs[:-1])))
    for ref in refs:
        p=out/'transfer'/ref['id'];p.mkdir(parents=True)
        row=dict(method='source_equal_dataset_log_ensemble',tau=2.,converged=ref['dataset']!='refed',
                 nrmse=[.1,.2,.3],all_below_half=False)
        (p/'result.json').write_text(json.dumps(dict(ref=ref,rows=[row],status='completed',sources={})))
    cfg=dict(transfer_source_runs=['retained'],data=dict(datasets=['refed']))
    result=runner.semantics_four_dataset_transfer_summary(cfg,out,tmp_path)
    assert result['planned_targets']==4 and result['converged']==3 and result['new_fits']==0
    frame=pd.read_csv(out/'four_dataset_transfer_v1/metrics.csv')
    assert set(frame.dataset)==set(datasets)
    assert frame.loc[frame.dataset=='refed','error'].isna().all()
    assert next(r for r in result['groups'] if r['dataset']=='refed')['planned']==1


def test_refed_planar_anchors_require_explicit_shared_frame_and_keep_native_identity():
    from types import SimpleNamespace
    names=['FP1','C3','O1','F3','FZ','C1','CP3','P3','PZ']
    positions=[[0,.8],[-.8,0],[0,-.8],[-.3,.5],[0,.5],[-.4,0],[-.5,-.2],[-.3,-.5],[0,-.5]]
    def row(name,position):
        return dict(channel_name=name,base_channel_name=name,x=position[0],y=position[1],z=0,
            coordinate_system='refed_paper_layout_2d_v1',coordinate_units='normalized_figure_radius',
            metadata={'spatial_dimension':2})
    eeg=[row(n,p) for n,p in zip(names,positions)]
    hb=[row('CH'+str(i+1),p) for i,p in enumerate([[0,.75],[-.75,0],[0,-.75]])]
    geometry=SimpleNamespace(for_channels=lambda **kw:eeg if kw['modality']=='eeg' else hb)
    record=SimpleNamespace(dataset_id='refed',manifest={'measurement':{
        'eeg_channel_names':names,'fnirs_channel_names':[f'CH{i}_{role}' for i in range(1,4) for role in ('HbO','HbR')]}})
    assert runner.semantics_anchors(record,geometry)[0]==[]  # frozen v1 remains excluded
    anchors,reason=runner.semantics_anchors(record,geometry,
        refed_region_eeg_anchors={'prefrontal':'FP1','motor':'C3','posterior':'O1'})
    assert reason is None and [a['hb_channel'] for a in anchors]==['CH1','CH2','CH3']
    assert all(len(a['eeg_channels'])==6 and a['seventh_eeg_channel'] not in a['eeg_channels'] for a in anchors)
    assert all(a['eeg_channels'][0]==name for a,name in zip(anchors,names[:3]))
    hb[0]['coordinate_system']='dataset_head_coordinates'
    with pytest.raises(ValueError,match='complete paper-layout'):
        runner.semantics_anchors(record,geometry,refed_region_eeg_anchors={'prefrontal':'FP1'})


def test_refed_video_folds_keep_every_window_of_a_video_together(tmp_path):
    cfg=runner.semantics_config(CONFIG.with_name('shared_driver_physiology_semantics_refed_v2.yaml'))
    retained=tmp_path/cfg['retained_run'];retained.mkdir(parents=True)
    subjects=[str(i) for i in range(1,7)]
    (retained/'cohort.json').write_text(json.dumps({'subject_folds':{'refed':{s:i%3 for i,s in enumerate(subjects)}}}))
    rng=np.random.default_rng(41);inventory=[]
    for subject in subjects:
        for video in range(1,5):
            key=f'refed__{subject}__video_{video}';folder=tmp_path/'prepared'/key;folder.mkdir(parents=True)
            windows=[dict(window=i,eeg_start_s=30.*i,hb_start_s=30.*i,event_id=video,
                task='emotion_video',condition='same_emotion',independent_block=f'video_{video}') for i in range(4)]
            anchor=dict(region='prefrontal',eeg_channels=[f'c{i}' for i in range(6)],hb_pair=0,hb_channel='CH1',
                windows=windows,array_file='prefrontal.npz')
            inventory.append(dict(key=key,dataset='refed',subject=subject,record=f'video_{video}',
                repeat_group=f'video_{video}',anchors=[anchor],windows=windows))
            (folder/'record.json').write_text(json.dumps(dict(prepared=[anchor],qc={'channels':[
                dict(rho=.2,support=1,jump_fraction=[0,0],flat_step_fraction=[0,0])]})))
            np.savez(folder/'prefrontal.npz',eeg_features=rng.normal(size=(4,120,30)),hb=rng.normal(size=(4,120,2)))
    (tmp_path/'inventory.json').write_text(json.dumps({'records':inventory}))
    refs,coordinates=runner.semantics_cohort(cfg,tmp_path,tmp_path)
    for subject in subjects:
        own=[r for r in refs if r['subject']==subject]
        assert {r['repeat_fold'] for r in own}=={0,1}
        for video in range(1,5):assert len({r['repeat_fold'] for r in own if r['record']==f'video_{video}'})==1
        for r in own:
            coordinate=coordinates[f'refed__prefrontal__outer{r["subject_fold"]}']
            assert subject not in coordinate['training_subjects']


def test_boundary_neighbor_fit_preserves_hidden_target_and_rejects_coordinate_leakage(tmp_path,monkeypatch):
    from src.inference import shared_driver_reconstruction as inference
    cfg=runner.semantics_config(CONFIG.with_name('shared_driver_physiology_semantics_refed_v2.yaml'))
    prepared=tmp_path/'prepared'/'record';prepared.mkdir(parents=True)
    rng=np.random.default_rng(2);hb=rng.normal(size=(1,120,2))
    for suffix in ('','_boundary_swap'):
        np.savez(prepared/('motor'+suffix+'.npz'),eeg_features=rng.normal(size=(1,120,30)),hb=hb)
    coordinates=tmp_path/'coordinates';coordinates.mkdir()
    c=dict(pc=[1.]+[0.]*29,eeg_factor=1.,hb_factor=1.,sd=[1.,2.,3.],training_subjects=['train'])
    for suffix in ('','__boundary_swap'):(coordinates/('refed__motor__outer0'+suffix+'.json')).write_text(json.dumps(c))
    ref=dict(id='test_ref',dataset='refed',subject='eval',site='motor',region='motor',subject_fold=0,
        array_path=str(prepared/'motor.npz'),array_index=0)
    calls=[]
    def fit(observed,*args,**kwargs):
        calls.append(observed.copy())
        return dict(status='fixture',converged=True,prediction=np.nan_to_num(observed))
    monkeypatch.setattr(inference,'fit_nonlinear_shared_driver',fit)
    result=runner.semantics_geometry_worker((cfg,str(tmp_path),ref))
    assert len(result['rows'])==4 and len(calls)==4
    assert np.isnan(calls[1][:,1:]).all() and np.isnan(calls[3][:,1:]).all()
    np.testing.assert_array_equal(calls[0][:,1:],calls[2][:,1:])
    c['training_subjects']=['eval'];(coordinates/'refed__motor__outer0__boundary_swap.json').write_text(json.dumps(c))
    with pytest.raises(ValueError,match='leaked evaluation'):
        runner.semantics_geometry_worker((cfg,str(tmp_path),dict(ref,id='different_ref')))


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


def test_retired_unfiltered_context_reads_only_retained_results(tmp_path, monkeypatch):
    from src.data import clean_physiology_cache as cache
    def forbidden(*args, **kwargs):
        raise AssertionError('retired context producer must not read measured data')
    monkeypatch.setattr(cache, 'CleanPhysiologyCacheIndex', forbidden)
    ref = dict(id='fixture_window')
    payload = ({}, str(tmp_path), str(tmp_path), ref)
    with pytest.raises(ValueError, match='Unfiltered context preparation is retired'):
        runner.semantics_context_worker(payload)
    path = tmp_path / 'context' / ref['id'] / 'result.json'
    path.parent.mkdir(parents=True)
    retained = dict(status='completed', processing='historical_unfiltered_target', rows=[])
    path.write_text(json.dumps(retained))
    assert runner.semantics_context_worker(payload) == retained
    assert json.loads(path.read_text()) == retained


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
