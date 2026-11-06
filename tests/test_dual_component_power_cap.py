"""Synthetic contracts for the conditional local voltage-power candidate."""
import numpy as np
import pytest

from src.data.dual_component_tokenizer import (
    fit_voltage_power_cap_reference, project_voltage_second_moment,
    voltage_power_cap_bank, voltage_power_cap_frames, voltage_power_cap_reference,
    voltage_power_cap_target,
)


def model_fixture(channels=4):
    rng = np.random.default_rng(52)
    a = rng.normal(size=(3, channels*5))*.1
    v = rng.normal(size=(channels*5, channels*5))
    v = v @ v.T + np.eye(channels*5)
    return dict(projection=a, mean=rng.normal(size=channels*5)*.1,
        covariance=v, lift=np.linalg.solve(a @ v @ a.T, a @ v).T,
        baseline_log_power=np.zeros(channels*5),
        correlation=np.tile(np.eye(channels), (6, 1, 1)),
        complement_power=np.ones(channels))


def test_second_moment_projection_matches_independent_singular_value_solution():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 60))*5
    q = rng.normal(size=(4, 4)); c = q @ q.T + np.eye(4)
    out, info = project_voltage_second_moment(x, c, 1.7)
    v, u = np.linalg.eigh(c)
    root = (u*np.sqrt(v)) @ u.T
    white = np.linalg.solve(root, x)
    left, singular, right = np.linalg.svd(white, full_matrices=False)
    expected = root @ (left*np.minimum(singular, np.sqrt(1.7*60))) @ right
    np.testing.assert_allclose(out, expected, atol=1e-12)
    assert np.linalg.eigvalsh(1.7*c-out @ out.T/60).min() > -1e-12
    assert info['max_output_eigenvalue'] <= 1.7+1e-12


def test_bound_does_not_imply_artifact_rejection_and_radial_saturation_is_local():
    x = np.array([[.1, -.1, .1, -.1]])
    n = np.array([[.05, .05, -.05, -.05]])
    before, _ = project_voltage_second_moment(x, np.eye(1), 1.)
    after, _ = project_voltage_second_moment(x+n, np.eye(1), 1.)
    np.testing.assert_array_equal(before, x)
    np.testing.assert_array_equal(after, x+n)
    np.testing.assert_allclose(after-before, n)
    saturated, _ = project_voltage_second_moment(20*x, np.eye(1), 1.)
    larger, _ = project_voltage_second_moment(200*x, np.eye(1), 1.)
    np.testing.assert_allclose(saturated, larger, atol=1e-14)


def test_projection_controls_uncentered_offsets_and_preserves_polarity():
    x = np.ones((2, 80))*20
    out, _ = project_voltage_second_moment(x, np.eye(2), 1.)
    negative, _ = project_voltage_second_moment(-x, np.eye(2), 1.)
    assert np.linalg.norm(out)/np.linalg.norm(x) < .1
    np.testing.assert_allclose(negative, -out, atol=1e-14)


def test_projection_is_invariant_to_extreme_channel_units():
    rng=np.random.default_rng(31);a=rng.normal(size=(4,4));c=a@a.T+np.eye(4)
    x=rng.normal(size=(4,80))*4
    before,_=project_voltage_second_moment(x,c,1.)
    units=np.array([1e-12,1e-3,1e3,1e12])
    after,info=project_voltage_second_moment(x*units[:,None],c*units[:,None]*units[None,:],1.)
    np.testing.assert_allclose(after/units[:,None],before,atol=2e-12)
    assert info['max_output_eigenvalue']<=1+1e-12


def test_spectral_cap_survives_highly_unequal_whitened_singular_values():
    rng=np.random.default_rng(32)
    left=np.linalg.qr(rng.normal(size=(4,4)))[0]
    right=np.linalg.qr(rng.normal(size=(80,4)))[0].T
    x=(left*np.array([1e12,1e6,.1,1e-5]))@right
    out,info=project_voltage_second_moment(x,np.eye(4),1.)
    assert np.linalg.eigvalsh(out@out.T/80).max()<=1+1e-12
    assert info['max_output_eigenvalue']<=1+1e-12


def test_reference_completion_satisfies_proxy_and_does_not_read_observed_nullspace():
    model = model_fixture(); p = np.random.default_rng(4).normal(size=(900, 3))*.1
    power = voltage_power_cap_reference(model, p, np.arange(900)/10)
    ell = np.log(power[:, :5].transpose(0, 2, 1).reshape(900, -1))-model['baseline_log_power']
    np.testing.assert_allclose(ell @ model['projection'].T, p, atol=1e-14)
    np.testing.assert_array_equal(power, voltage_power_cap_reference(model, p.copy(), np.arange(900)/10))
    with pytest.raises(ValueError, match='complete finite'):
        voltage_power_cap_reference(model, p[:800], np.arange(900)/10)


def test_complete_carrier_bank_and_overlap_add_are_identity_when_cap_is_inactive():
    x = np.random.default_rng(5).normal(size=(18000, 4))*.1 + .2
    bank = voltage_power_cap_bank(x)
    np.testing.assert_allclose(bank.sum(axis=1), x, atol=1e-15)
    frames, t = voltage_power_cap_frames(bank)
    assert frames.shape == (901, 6, 4, 200)
    np.testing.assert_array_equal(t[[0, -1]], [0., 90.])
    result = voltage_power_cap_target(x, model_fixture(), None, cap=1e6)
    np.testing.assert_allclose(result['s'], x, atol=1e-14)
    np.testing.assert_allclose(result['s']+result['o'], x, atol=1e-14)
    assert result['final_reanalysis_ratio'].shape == (901, 6)


def test_training_reference_rejects_validation_before_processing():
    with pytest.raises(ValueError, match='training windows only'):
        fit_voltage_power_cap_reference([dict(split='validation', x=None)], None)
    with pytest.raises(ValueError, match='positive reference'):
        project_voltage_second_moment(np.ones((2, 20)), np.diag([1., 0.]), 1.)


def test_train_calibration_retains_complete_band_units_and_fixed_positive_statistics():
    rng = np.random.default_rng(6)
    training = [dict(split='train', x=rng.normal(size=(18000, 4))) for _ in range(2)]
    model = fit_voltage_power_cap_reference(training, model_fixture()['projection'])
    assert np.linalg.eigvalsh(model['covariance']).min() > 0
    assert np.linalg.eigvalsh(model['correlation']).min() > 0
    p = voltage_power_cap_reference(model, np.zeros((900, 3)), np.arange(900)/10)
    assert p.shape == (900, 6, 4)
    assert np.isfinite(p).all() and (p>0).all()


def test_paired_score_support_keeps_O_error_identical_and_counts_cross_channel_leakage():
    from src.analysis.dual_component_tokenizer import separation_change_metrics
    n=np.zeros((40, 2));n[:,0]=1
    base=np.zeros_like(n);changed=np.column_stack((np.full(40,.2),np.full(40,.4)))
    row=separation_change_metrics(base,changed,n,'ocular',base_o=base,changed_o=n-changed,
        score_slice=slice(None))
    assert row['change_NMSE']==pytest.approx(.2)
    assert row['O_change_NMSE']==pytest.approx(.2)
    assert row['S_signed_gain']==pytest.approx(.2)
    assert row['O_signed_gain']==pytest.approx(.8)


def test_power_cap_contract_rejects_new_case_or_test_scope(tmp_path):
    import importlib.util
    from pathlib import Path
    import yaml
    root=Path(__file__).resolve().parents[1]
    module_spec=importlib.util.spec_from_file_location('power_cap_runner',root/'experiments/scripts/run_dual_component_tokenizer.py')
    runner=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(runner)
    path=root/'experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_power_cap_v1.yaml'
    good=runner.read_config(path)
    assert good['power_cap']['physical']['condition']=='joint_full'
    for key,value in [('case_id','unplanned'),('test_access','allowed')]:
        bad=yaml.safe_load(path.read_text());bad['power_cap'][key]=value
        file=tmp_path/'bad.yaml';file.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError,match='one-window power-cap'):
            runner.read_config(file)


def test_joint_fit_adapter_uses_exact_existing_teacher_clock(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    import src.data.dual_component_tokenizer as data
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('power_cap_clock_runner',root/'experiments/scripts/run_dual_component_tokenizer.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    a=dict(x=np.zeros((18000,16)),roi_indices=np.tile(np.arange(6),(3,1)),
        features_baseline=np.zeros((900,16,5)),hb=np.zeros((900,3,2)),
        hb_native_time=np.tile((np.arange(900)*.1)[:,None],(1,3)))
    plan=dict(parent=str(tmp_path),case=dict(rows=[dict(id='fixture',coordinate='fixture')]*3))
    monkeypatch.setattr(runner,'_voltage_teacher_read',lambda path:a)
    monkeypatch.setattr(runner,'read_json',lambda path:plan if Path(path).name=='plan.json'
        else {'fixture':{}} if Path(path).name=='coordinates.json' else {})
    calls=[]
    def fake(arrays,coordinate,cfg,*,eeg_visible):
        data._check_prepared(arrays)
        calls.append(eeg_visible)
        return None,dict(accepted=False)
    monkeypatch.setattr(data,'produce_teacher',fake)
    result=runner._power_cap_fit((str(tmp_path),'baseline',0))
    assert 'error' not in result
    assert calls==[True]


def test_teacher_weights_change_only_the_objective_scale(monkeypatch):
    from pathlib import Path
    import yaml
    import src.inference.shared_driver_reconstruction as inference
    from src.data.dual_component_tokenizer import produce_teacher
    from src.data.ssm_prepared import EEG_PROXY_INPUT_SCHEMA
    from src.data.unified_physiology import CACHED_FNIRS_INPUT_SCHEMA
    root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((root/'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_10hz_v2.yaml').read_text())
    a=dict(features=np.zeros((900,30)),hb=np.zeros((900,2)),eeg=np.zeros((4000,6)),
        native_time=np.arange(900)*.1,time_s=np.arange(900)*.1,identity=np.array('fixture'),
        eeg_proxy_input=np.array(EEG_PROXY_INPUT_SCHEMA),eeg_proxy_rate_hz=np.array(10.),
        fnirs_input=np.array(CACHED_FNIRS_INPUT_SCHEMA))
    original=np.array([.025,.032,.013])
    co=dict(pc=np.ones(30),eeg_factor=1.,hb_factor=1.,sd=original.copy(),eeg_proxy_input=EEG_PROXY_INPUT_SCHEMA)
    captured=[]
    def solver(observations,parameters,dt,**kwargs):
        captured.append(kwargs);return dict(status='fixture')
    monkeypatch.setattr(inference,'fit_nonlinear_shared_parameters',solver)
    _,status=produce_teacher(a,co,cfg,observation_weights=[.01,100.,100.])
    np.testing.assert_allclose(captured[-1]['sd'],original/np.sqrt([.01,100.,100.]),atol=0,rtol=1e-15)
    np.testing.assert_array_equal(co['sd'],original)
    assert captured[-1]['visible'].sum(axis=(0,1)).tolist()==[899,899,899]
    np.testing.assert_array_equal(status['observation_weights'],[.01,100.,100.])
    produce_teacher(a,co,cfg)
    np.testing.assert_array_equal(captured[-1]['sd'],original)
    with pytest.raises(ValueError,match='objective weights'):
        produce_teacher(a,co,cfg,observation_weights=[0,1,1])


def test_weighted_continuation_keeps_observations_and_acceptance_threshold(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    import src.data.dual_component_tokenizer as data
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('power_cap_resume_runner',root/'experiments/scripts/run_dual_component_tokenizer.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    source=tmp_path/'old.npz';source.touch()
    a=dict(x=np.zeros((18000,16)),roi_indices=np.tile(np.arange(6),(3,1)),
        features_baseline=np.zeros((900,16,5)),hb=np.zeros((900,3,2)),hb_native_time=np.zeros((900,3)))
    saved=dict(u=np.zeros((900,3)),driver=np.ones(900)*.01,initial_state=np.array([0.,1.,1.,1.,1.]))
    plan=dict(parent=str(tmp_path),continuation_source=str(tmp_path/'previous'),observation_weights=[.01,100.,100.],
        solver_budget=dict(max_iterations=1500,max_evaluations_per_trial=4000),
        case=dict(rows=[dict(id='fixture',coordinate='fixture')]*3))
    previous=dict(key='baseline__0',observation_weights=plan['observation_weights'],status='failed',array_path=str(source))
    def read(path):
        name=Path(path).name
        if name=='plan.json':return plan
        if name=='coordinates.json':return {'fixture':dict(pc=np.zeros(30),eeg_factor=1.,hb_factor=1.)}
        if name=='teacher_config.json':return {'solver':{'max_iterations':300,'max_evaluations_per_trial':600,'gradient_tolerance':1e-5}}
        return previous
    monkeypatch.setattr(runner,'read_json',read)
    monkeypatch.setattr(runner,'_voltage_teacher_read',lambda path:saved if Path(path)==source else a)
    captured=[]
    def fake(arrays,coordinate,cfg,**kwargs):
        captured.append((cfg,kwargs));return None,dict(accepted=False)
    monkeypatch.setattr(data,'produce_teacher',fake)
    result=runner._power_cap_fit((str(tmp_path),'baseline',0))
    assert 'error' not in result
    cfg,kwargs=captured[0]
    assert cfg['solver']['gradient_tolerance']==1e-5
    assert cfg['solver']['max_evaluations_per_trial']==4000
    np.testing.assert_array_equal(kwargs['warm_start']['driver'],saved['driver'])
    assert kwargs['observation_weights']==[.01,100.,100.]


def test_examples_select_four_other_subjects_before_any_array_read():
    from src.data.dual_component_tokenizer import voltage_power_cap_example_cases
    from copy import deepcopy
    rows=[dict(id=f'{s}_{slot}',subject=s,slot=slot,split='validation',phase='validation',
        dataset='simultaneous_eeg_nirs',record='cnt_nback')
        for s in ('VP002','VP005','VP014','VP017','VP019') for slot in (0,1)]
    panel=dict(schema='voltage_separation_panel_v1',protected_data='forbidden',test_access='forbidden',rows=rows)
    subjects=['VP005','VP014','VP017','VP019']
    selected=voltage_power_cap_example_cases(panel,'VP002_1',subjects)
    assert [r['id'] for r in selected]==[s+'_0' for s in subjects]
    bad=deepcopy(panel);bad['rows'][2]['split']='test'
    with pytest.raises(ValueError,match='validation'):
        voltage_power_cap_example_cases(bad,'VP002_1',subjects)
    with pytest.raises(ValueError,match='other validation'):
        voltage_power_cap_example_cases(panel,'VP002_1',['VP002',*subjects[1:]])


def test_examples_contract_keeps_weights_scope_and_no_search(tmp_path):
    import importlib.util
    from pathlib import Path
    import yaml
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('example_config_runner',root/'experiments/scripts/run_dual_component_tokenizer.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    path=root/'experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_power_cap_examples_v1.yaml'
    cfg=runner.read_config(path)
    assert cfg['power_cap']['physical']['observation_weights']==[.1,10.,10.]
    for key,value in [('allowed_split','test'),('subjects',['VP005','VP014','VP017','VP020']),('cap',.1)]:
        bad=yaml.safe_load(path.read_text());bad['examples'][key]=value
        p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError,match='four-subject'):
            runner.read_config(p)


def test_examples_summary_keeps_unavailable_pair_in_denominator(tmp_path):
    from src.analysis.dual_component_tokenizer import summarize_voltage_power_cap_examples
    entries=[dict(case=dict(id=f'case_{i}',subject=f'VP{i:03}'),run=str(tmp_path/f'case_{i}'),reused=False)
        for i in range(5)]
    result=summarize_voltage_power_cap_examples(entries,[])
    assert result['counts']['planned_pairs']==5
    assert result['counts']['completed_pairs']==0
    assert result['counts']['failed_pairs']==5
    assert all(r['status']=='failed' for r in result['rows'])


def test_posterior_contract_keeps_named_electrodes_and_development_scope(tmp_path):
    import importlib.util
    from pathlib import Path
    import yaml
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('posterior_config_runner',root/'experiments/scripts/run_dual_component_tokenizer.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    path=root/'experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_power_cap_posterior_v1.yaml'
    assert runner.read_config(path)['posterior']['channels']==17
    for key,value in [('channels',16),('allowed_splits',['train','test']),('affected_roi',1)]:
        bad=yaml.safe_load(path.read_text());bad['posterior'][key]=value
        p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError,match='posterior scope'):
            runner.read_config(p)


def test_posterior_reader_rejects_test_identity_before_signal_read(monkeypatch,tmp_path):
    import src.data.dual_component_tokenizer as data
    def forbidden(*a,**kw):raise AssertionError('signal reader must not run')
    monkeypatch.setattr(data,'load_native_eeg_record',forbidden)
    monkeypatch.setattr(data,'load_cached_fnirs_record',forbidden)
    entry=dict(split='test',source_case={})
    plan=dict(schema='voltage_power_cap_posterior_plan_v1',protected_data='forbidden',
        test_access='forbidden',rows=[entry],panel={'rows':[{}]})
    with pytest.raises(ValueError,match='before array access'):
        data.prepare_voltage_power_cap_posterior(tmp_path,{},plan,entry)


def test_posterior_injection_changes_the_modeled_posterior_features_only():
    from src.data.dual_component_tokenizer import separation_covariances
    rng=np.random.default_rng(81);x=rng.normal(size=(18000,17));changed=x.copy()
    changed[:,15]+=20*np.exp(-.5*((np.arange(18000)/200-40)/.3)**2)
    a=separation_covariances(x,dict(covariance_seconds=2))['features']
    b=separation_covariances(changed,dict(covariance_seconds=2))['features']
    assert a.shape==(900,17,5)
    np.testing.assert_array_equal(a[:,np.arange(17)!=15],b[:,np.arange(17)!=15])
    assert np.linalg.norm(a[:,15]-b[:,15])>1


def test_posterior_summary_uses_actual_display_channel_and_roi(tmp_path):
    import json
    from src.analysis.dual_component_tokenizer import summarize_voltage_power_cap_examples
    x=np.tile(np.sin(np.arange(18000)*.07)[:,None],(1,17))
    n=np.zeros_like(x);n[:,15]=np.sin(np.arange(18000)*.05)
    np.savez(tmp_path/'inputs.npz',x=x,n_ocular=n,channels=np.array([f'C{i}' for i in range(15)]+['O1','O2']))
    (tmp_path/'targets').mkdir();records=[]
    for variant,s,o in [('baseline',.5*x,.5*x),('ocular',.5*x+.25*n,.5*x+.75*n)]:
        p=tmp_path/'targets'/f'physical_refitted_reference__{variant}__cap1.npz'
        np.savez(p,s_score=s[6000:10000],o_score=o[6000:10000])
        p.with_suffix('.json').write_text(json.dumps({'status':'completed'}))
        arr=tmp_path/(variant+'.npz');np.savez(arr,u=np.zeros((900,3)),physical=np.ones((900,3))*.1)
        records.append(dict(variant=variant,roi=2,status='completed',array_path=str(arr)))
    (tmp_path/'physical_fits.json').write_text(json.dumps(records))
    (tmp_path/'coordinates.json').write_text(json.dumps({'occipital':{'sd':[1,2,4]}}))
    case=dict(id='fixture',subject='fixture',rows=[{}, {}, {'coordinate':'occipital'}])
    result=summarize_voltage_power_cap_examples([dict(case=case,run=str(tmp_path),reused=False,
        display_channel='O1',metric_roi=2)],[],planned_new_fits=2)
    row=result['rows'][0]
    assert row['status']=='completed'
    assert set(row['metrics'])=={'O1','all17'}
    assert row['metrics']['O1']['S_error_NRMSE']==pytest.approx(.25)
    np.testing.assert_allclose(row['model_NRMSE']['ocular']['full899'],[.1,.05,.025])
