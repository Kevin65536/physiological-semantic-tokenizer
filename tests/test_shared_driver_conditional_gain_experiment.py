"""Conditional scale/gain contracts use public synthetic fixtures only."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import yaml
from experiments.scripts import evaluate_shared_driver_reconstruction as run
from src.inference.observation_baselines import conditional_optical_hb_mapping
from src.inference.shared_driver_reconstruction import nonlinear_driver_forward

CONFIG=run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_conditional_optical_gain_v1.yaml'
@pytest.fixture
def cfg():return run.read_conditional_optical_gain_config(CONFIG)


def test_mapping_physical_log_base_units_and_signed_cross_terms():
    d=conditional_optical_hb_mapping(40.);old=np.array([[.148,.384],[.252,.179]]);eps=np.array([[586.,1548.52],[1058.,691.32]])
    state_fraction=np.array([.01,-.003]);natural_od=np.log(10.)*eps@(71e-6*state_fraction)*18.
    recovered=np.linalg.solve(old,natural_od/18.)*40.
    np.testing.assert_allclose(d['matrix']@state_fraction,recovered,rtol=1e-14)
    np.testing.assert_allclose(d['base_matrix'],[[.70109771,-.02726013],[-.02073205,.66977092]],atol=5e-9)
    assert d['matrix'][0,1]<0 and d['matrix'][1,0]<0
    assert np.isclose(d['beta_reference'],.036504726246241764)
    assert np.isclose(np.linalg.det(d['matrix'])*d['beta_reference']**2,1.)
    for bad in [0.,-1.,np.nan,np.inf]:
        with pytest.raises(ValueError):conditional_optical_hb_mapping(bad)


def test_composed_prediction_and_exact_jacobian_finite_difference(cfg):
    n=8;r=np.linspace(-.02,.03,n);initial=np.array([.001,1.002,.999,1.001,.998]);p=run.parameters(cfg,2.)
    mapping=conditional_optical_hb_mapping(40.);p=replace(p,fixed=replace(p.fixed,neurovascular_gain=mapping['beta_reference']))
    op=run.conditional_prediction_operator(np.eye(3*n),mapping,'conditional_fixed_gain')
    f=nonlinear_driver_forward(r,initial,p,.25)
    np.testing.assert_allclose((op@f['canonical_prediction'].ravel()).reshape(n,3)[:,1:],f['canonical_prediction'][:,1:]@mapping['matrix'].T)
    jac=op@f['jacobian'];step=1e-6
    for col in [0,4,n,n+1,n+3,n+4]:
        rp,rm=r.copy(),r.copy();ip,im=initial.copy(),initial.copy()
        if col<n:rp[col]+=step;rm[col]-=step
        elif col==n:ip[0]+=step;im[0]-=step
        else:ip[col-n]*=np.exp(step);im[col-n]*=np.exp(-step)
        fp=nonlinear_driver_forward(rp,ip,p,.25,derivative=False)['canonical_prediction']
        fm=nonlinear_driver_forward(rm,im,p,.25,derivative=False)['canonical_prediction']
        np.testing.assert_allclose(jac[:,col],op@((fp-fm)/(2*step)).ravel(),rtol=2e-5,atol=2e-8)
    plus=replace(p,fixed=replace(p.fixed,neurovascular_gain=p.fixed.neurovascular_gain+step));minus=replace(p,fixed=replace(p.fixed,neurovascular_gain=p.fixed.neurovascular_gain-step))
    fd=(nonlinear_driver_forward(r,initial,plus,.25,derivative=False)['canonical_prediction']-nonlinear_driver_forward(r,initial,minus,.25,derivative=False)['canonical_prediction'])/(2*step)
    np.testing.assert_allclose(op@f['neurovascular_gain_jacobian'].ravel(),op@fd.ravel(),rtol=2e-5,atol=2e-8)


@pytest.fixture
def parent(tmp_path,cfg):
    root=tmp_path/'root';p=root/cfg['parent_run'];p.mkdir(parents=True)
    old=run.read_fixed_roi_optical_config(run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_optical_v1.yaml')
    (p/'resolved_config.yaml').write_text(yaml.safe_dump(old));run.write_json(p/'manifest.json',dict(experiment_id=old['experiment_id'],execution='completed',pilot=False,synthetic_terminal=True,measured_terminal=True,project_root=str(root.resolve())))
    return root,p,old


def add_measured(parent,cfg):
    root,p,old=parent
    for spec in run.conditional_gain_specs(cfg,'measured'):
        validation=[i for i in range(24) if i%4==spec['outer']];train=[i for i in range(24) if i not in validation]
        k=20.+spec['outer'];idx=cfg['subjects'].index(spec['subject']);trials=[dict(subject=spec['subject'],session=cfg['sessions'][i//8],original_ma_trial_position=cfg['original_positions'][i%8],event_index=i+1,sample_id=spec['subject']+'/'+str(i)) for i in range(24)]
        meta=dict(subject=spec['subject'],outer=spec['outer'],train=train,validation=validation,trials=trials,normalization_sd=[1.,2.,3.],fixed_roi=dict(fnirs_factor=k,common_training_MAD=.01,frozen_observation_loading=k*.01),
            projection=dict(fnirs_factor=k,measurement_scale=dict(fnirs_common=.01)),
            optical_processing=dict(pipeline=spec['optical_pipeline'],training_trials=train,source_audit=str(root/cfg['optical_audit_run']),source_row_indices=list(range(idx*24,(idx+1)*24)),fnirs_factor=k,common_training_MAD=.01,frozen_observation_loading=k*.01))
        f=p/'prepared'/f"{spec['group']}.json";run.write_json(f,dict(status='completed',spec=spec,metadata=meta,driver_prior_sd=1.))
        f.with_suffix('.npz').write_bytes(b'array-free gate')
        for mode in cfg['modes']:
            cell=run.nonlinear_cell_id(spec,'fixed_tau_logflow1',mode);rows=[dict(spec,trial=i,session=cfg['sessions'][i//8],method='fixed_tau_logflow1',mode=mode,status='failed_numerical',converged=False) for i in validation]
            run.write_json(p/'cells'/cell/'result.json',dict(cell=cell,rows=rows))
    return root,p,old


def test_check_only_no_arrays_or_measured_metadata(parent,cfg,monkeypatch,capsys):
    root,p,old=parent;monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('array read'))
    run.gain_prior_fit_main(SimpleNamespace(config=CONFIG,conditional_optical_gain_fit=True,project_root=root,check_only=True))
    d=json.loads(capsys.readouterr().out);assert (d['synthetic_groups'],d['measured_groups'],d['training_starts_per_group'],d['validation_rows_per_group'])==(12,24,3,54)
    assert not (p/'prepared').exists()

@pytest.mark.parametrize('key,value',[('parent_run','wrong'),('methods',['conditional_fixed_gain']),('conditional_mapping',{}),('conditional_gain_training',{}),('optical_pipelines',['no_motion'])])
def test_strict_contract(cfg,tmp_path,key,value):
    cfg[key]=value;p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):run.read_conditional_optical_gain_config(p)

@pytest.mark.parametrize('mutation',['pipeline','rows','factor','train','identity'])
def test_parent_metadata_before_arrays(parent,cfg,monkeypatch,mutation):
    root,p,old=add_measured(parent,cfg);f=p/'prepared/no_motion__subject_01_o0.json';r=json.loads(f.read_text());m=r['metadata'];o=m['optical_processing']
    if mutation=='pipeline':o['pipeline']='mne_tddr'
    elif mutation=='rows':o['source_row_indices']=list(range(24,48))
    elif mutation=='factor':o['fnirs_factor']*=2.
    elif mutation=='train':o['training_trials']=m['validation']
    else:m['trials'][0]['subject']='subject_99'
    run.write_json(f,r);monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('array read before metadata gate'))
    with pytest.raises(ValueError):run.validate_conditional_gain_parent(cfg,root,'measured')


def test_exact_measured_copy_no_second_transform_and_failed_baseline_without_trajectory(parent,cfg,tmp_path):
    root,p,old=add_measured(parent,cfg);spec=run.conditional_gain_specs(cfg,'measured')[0];source=p/'prepared'/f"{spec['group']}.npz"
    rng=np.random.default_rng(3);arrays=dict(target=rng.normal(size=(24,120,3)),feature_eeg=rng.normal(size=(24,120)),feature_fnirs=rng.normal(size=(24,300,2)),normalizer=np.array([1.,2.,3.]),raw_optical_hb_native=rng.normal(size=(24,300,2)))
    arrays['feature_fnirs']=arrays['raw_optical_hb_native']*20.
    arrays['target']=np.array([run.view(e,h,cfg,'full')[0] for e,h in zip(arrays['feature_eeg'],arrays['feature_fnirs'])])
    np.savez(source,**arrays);out=tmp_path/'new';r=run.prepare_conditional_gain_group((cfg,out,root,spec,False))
    assert source.read_bytes()==(out/'prepared'/source.name).read_bytes()
    assert r['conditional_mapping']['fnirs_factor']==20.
    cell=run.nonlinear_cell((cfg,out,spec,'fixed_tau_logflow1','full',False))
    assert all(row['status']=='failed_numerical' for row in cell['rows']) and cell['inheritance']['status']=='inherited_nonindependent'
    assert not (out/'cells'/cell['cell']/'trajectories.npz').exists()


def test_synthetic_true_gain_and_mapping_before_noise_pairing(cfg,monkeypatch):
    # Fast observation fixture isolates generation/order; actual RK4 Jacobians tested above.
    from src.inference.t3a_balloon_robust_ssm import observation_map
    calls=[]
    def replay(driver,initial,p,dt,**kw):
        calls.append((driver.copy(),initial.copy(),p.fixed.neurovascular_gain))
        states=np.column_stack((driver,np.tile(initial,(len(driver),1))))
        states[:,2]+=p.fixed.neurovascular_gain*driver
        states[:,4]+=p.fixed.neurovascular_gain*driver
        states[:,5]-=.1*p.fixed.neurovascular_gain*driver
        return dict(status='completed',states=states,canonical_prediction=np.array([observation_map(x,p) for x in states]))
    monkeypatch.setattr(run,'replay_nonlinear_driver',replay)
    specs=[s for s in run.conditional_gain_specs(cfg,'synthetic') if s['spectrum']=='slow' and s['replicate']==0]
    first=[];standard_noise=[]
    for spec in specs:
        start=len(calls);a,m=run.nonlinear_synthetic_panel(cfg,spec);first.append(calls[start]);mapping=conditional_optical_hb_mapping(40.)
        mean=run.conditional_prediction_operator(run.view_operators(cfg,'full')[2],mapping,'conditional_fixed_gain')
        p=run.parameters(cfg,2.);p=replace(p,fixed=replace(p.fixed,neurovascular_gain=spec['true_neurovascular_gain']))
        canonical=np.array([observation_map(state,p) for state in a['truth_states'][0]])
        np.testing.assert_allclose(a['clean'][0],(mean@canonical.ravel()).reshape(120,3),rtol=1e-10,atol=1e-10)
        mapped=canonical.copy();mapped[:,1:]=mapped[:,1:]@mapping['matrix'].T
        native=run.native_feature_operators(120)['native_interpolation']@mapped[:,1:]
        noise=cfg['synthetic']['noise_fraction']*np.std(mapped,axis=0)
        standard_noise.append((a['feature_fnirs'][0]-native)/noise[1:])
        assert np.isclose(calls[start][2],spec['true_relative_gain']*mapping['beta_reference'])
        assert a['truth_initial_only_clean'].shape==a['truth_driver_only_clean'].shape==(24,120,3)
        assert m['truth_component_diagnostic']['interpretation'].startswith('separate_nonlinear')
    for row in first[1:]:np.testing.assert_array_equal(row[0],first[0][0]);np.testing.assert_array_equal(row[1],first[0][1])
    for z in standard_noise[1:]:np.testing.assert_allclose(z,standard_noise[0],rtol=1e-10,atol=1e-10)


def test_plan_counts(cfg,tmp_path):
    specs=run.conditional_gain_specs(cfg,'synthetic')+run.conditional_gain_specs(cfg,'measured')
    plan=[(s,m,v) for s in specs for m in cfg['methods'] for v in cfg['modes']]
    summary=run.summarize_fixed_roi_tau(tmp_path,plan,False)
    assert summary['expected_cells']==324 and summary['expected_rows']==1944
    assert summary['expected_training_groups']==36 and summary['expected_new_training_starts']==108
    assert summary['expected_inherited_baseline_rows']==432 and summary['expected_new_validation_rows']==1512


def software_prepared(out,cfg,spec,target=None):
    train=[i for i in range(24) if i%4!=spec['outer']];val=[i for i in range(24) if i not in train]
    record=dict(status='completed',spec=spec,metadata=dict(train=train,validation=val),driver_prior_sd=1.,
        conditional_mapping=conditional_optical_hb_mapping(40.),selected_tau={m:2. for m in cfg['methods']})
    run.write_json(out/'prepared'/f"{spec['group']}.json",record)
    rng=np.random.default_rng(9);e=rng.normal(size=(24,120));h=rng.normal(size=(24,300,2))
    if target is None:target=np.array([run.view(x,y,cfg,'full')[0] for x,y in zip(e,h)])
    np.savez(out/'prepared'/f"{spec['group']}.npz",target=target,normalizer=np.ones(3),feature_eeg=e,feature_fnirs=h,truth=np.ones((24,120)),truth_states=np.ones((24,120,6)))
    return record,e,h


def test_group_gain_training_routes_reference_B_and_fine_check_train_only(cfg,tmp_path,monkeypatch):
    from src.inference import shared_driver_reconstruction as lib
    spec=run.conditional_gain_specs(cfg,'synthetic')[0];mapping=conditional_optical_hb_mapping(40.);ref=mapping['beta_reference'];calls=[];fine_gains=[]
    expected=run.conditional_prediction_operator(run.view_operators(cfg,'full')[2],mapping,'conditional_trained_gain')
    def fit(y,p,dt,**kw):
        assert p.free.tau==2. and kw['parameter_name']=='neurovascular_gain'
        np.testing.assert_array_equal(kw['mean_operator'],expected)
        np.testing.assert_allclose(kw['parameter_bounds'],np.array([.1,10.])*ref)
        assert kw['starts'][0]['parameter_value']==.5*ref
        assert kw['parameter_prior_mean'] is None and kw['parameter_prior_log_sd'] is None
        assert kw['initial_coordinates']=='independent_logs' and kw['tie_total_hb_to_volume'] is False
        assert kw['flow_prior_weight']==1. and kw['substeps']==4 and kw['max_evaluations']==3600
        assert y.shape==(18,120,3) and np.all(y==2.);calls.append(y.copy())
        return dict(status='completed',converged=True,parameter_value=1.2*ref,objective=1.,boundary_status='INTERIOR',
            driver=np.zeros((18,120)),initial_state=np.tile([0.,1.,1.,1.,1.],(18,1)),states=np.tile([0.,0.,1.,1.,1.,1.],(18,120,1)),prediction=np.zeros((18,120,3)),canonical_prediction=np.zeros((18,120,3)))
    def replay(driver,initial,p,dt,**kw):
        assert kw['substeps']==8 and p.free.tau==2.;fine_gains.append(p.fixed.neurovascular_gain)
        return dict(status='completed',states=np.tile([0.,0.,1.,1.,1.,1.],(120,1)),canonical_prediction=np.zeros((120,3)))
    monkeypatch.setattr(lib,'fit_nonlinear_shared_parameter',fit);monkeypatch.setattr(run,'replay_nonlinear_driver',replay)
    for folder,value in [('a',3.),('b',1e8)]:
        target=np.full((24,120,3),2.);target[::4]=value;out=tmp_path/folder;software_prepared(out,cfg,spec,target)
        r=run.train_gain_prior_start((cfg,out,spec,'conditional_trained_gain',0,.5))
        assert r['status']=='completed' and r['relative_gain']==1.2 and r['start_relative_gain']==.5 and r['start_gain']==.5*ref
        assert all(c['difference_training_sd']==0. for c in r['integration_checks'])
    np.testing.assert_array_equal(*calls);np.testing.assert_allclose(fine_gains,1.2*ref)

@pytest.mark.parametrize('method',run.CONDITIONAL_GAIN_METHODS)
def test_synthetic_all_three_arms_fit_and_validation_mean_not_target_transformed(cfg,tmp_path,monkeypatch,method):
    from src.inference import shared_driver_reconstruction as lib
    spec=run.conditional_gain_specs(cfg,'synthetic')[0];record,e,h=software_prepared(tmp_path,cfg,spec)
    mapping=record['conditional_mapping'];ref=mapping['beta_reference'];gain=1. if method=='fixed_tau_logflow1' else ref if method=='conditional_fixed_gain' else ref*1.7
    if method=='conditional_trained_gain':run.write_json(run.gain_training_directory(tmp_path,spec,method)/'selection.json',dict(status='completed',parameter_value=gain))
    expected=run.conditional_prediction_operator(run.view_operators(cfg,'center_fNIRS')[2],mapping,method)
    calls=[]
    def design(p,n,dt,**kw):
        assert p.free.tau==2. and p.fixed.neurovascular_gain==gain;np.testing.assert_array_equal(kw['processed_mean_operator'],expected)
    monkeypatch.setattr(run,'build_shared_driver_design',design)
    monkeypatch.setattr(run,'fit_shared_driver',lambda *a,**kw:dict(driver=np.zeros(120),states=np.tile([0.,0.,1.,1.,1.,1.],(120,1))))
    def fit(y,p,dt,**kw):
        trial=record['metadata']['validation'][len(calls)];expected_y,_,visible=run.view(e[trial],h[trial],cfg,'center_fNIRS')
        np.testing.assert_array_equal(y,expected_y);np.testing.assert_array_equal(kw['visible'],visible)
        np.testing.assert_array_equal(kw['mean_operator'],expected)
        assert p.free.tau==2. and p.fixed.neurovascular_gain==gain
        assert kw['initial_coordinates']=='independent_logs' and kw['flow_prior_weight']==1.
        calls.append(y);return dict(status='failed_numerical',converged=False,starts=[])
    monkeypatch.setattr(lib,'fit_nonlinear_shared_driver',fit)
    result=run.nonlinear_cell((cfg,tmp_path,spec,method,'center_fNIRS',False))
    assert len(calls)==6 and 'inheritance' not in result
    assert all(r['neurovascular_gain']==gain and np.isclose(r['relative_gain'],gain/ref) for r in result['rows'])


def test_bad_array_preparation_does_not_publish_completed_resume(parent,cfg,tmp_path):
    root,p,old=add_measured(parent,cfg);spec=run.conditional_gain_specs(cfg,'measured')[0];source=p/'prepared'/f"{spec['group']}.npz";out=tmp_path/'bad'
    np.savez(source,target=np.zeros((24,120,3)),feature_eeg=np.zeros((24,120)),feature_fnirs=np.zeros((24,300,2)),normalizer=np.ones(3),raw_optical_hb_native=np.zeros((24,300,2)))
    for _ in range(2):
        with pytest.raises(ValueError,match='SD identity'):run.prepare_conditional_gain_group((cfg,out,root,spec,False))
        assert not (out/'prepared'/source.with_suffix('.json').name).exists()
    run.write_json(out/'prepared'/source.with_suffix('.json').name,dict(status='completed',spec=spec))
    with pytest.raises(ValueError,match='cannot be resumed'):run.prepare_conditional_gain_group((cfg,out,root,spec,False))

@pytest.mark.parametrize('mutation',['bad_recovery','spread','bound','failed','missing'])
def test_prespecified_synthetic_screen_rejects_each_failure(cfg,tmp_path,mutation):
    specs=run.conditional_gain_specs(cfg,'synthetic')
    for spec in specs:
        folder=run.gain_training_directory(tmp_path,spec,'conditional_trained_gain');g=spec['true_relative_gain']
        run.write_json(folder/'selection.json',dict(status='completed',relative_gain=g,terminal_starts=3))
        for i in range(3):run.write_json(folder/f'start_{i}'/'result.json',dict(status='completed',converged=True,relative_gain=g,boundary_status='INTERIOR'))
    assert run.conditional_gain_screen(tmp_path,specs)['all_groups_passed']
    folder=run.gain_training_directory(tmp_path,specs[0],'conditional_trained_gain');g=specs[0]['true_relative_gain']
    if mutation=='bad_recovery':run.write_json(folder/'selection.json',dict(status='completed',relative_gain=g*1.11,terminal_starts=3))
    elif mutation=='missing':(folder/'start_2/result.json').unlink()
    elif mutation=='failed':run.write_json(folder/'selection.json',dict(status='failed_training',relative_gain=None,terminal_starts=3))
    else:run.write_json(folder/'start_0/result.json',dict(status='completed',converged=True,relative_gain=g*1.06 if mutation=='spread' else g,boundary_status='LOWER' if mutation=='bound' else 'INTERIOR'))
    result=run.conditional_gain_screen(tmp_path,specs);assert not result['all_groups_passed'] and result['groups'][0]['passed'] is False


def test_measured_gate_requires_recovery_not_just_terminal(parent,cfg,monkeypatch):
    root,p,old=parent
    monkeypatch.setattr(run,'summarize_fixed_roi_tau',lambda *a,**k:dict(terminal_cells=108,expected_cells=108,observed_rows=648,expected_rows=648,terminal_training_groups=12,expected_training_groups=12,synthetic_engineering_screen=dict(all_groups_passed=False)))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('measured array read'))
    args=SimpleNamespace(config=CONFIG,conditional_optical_gain_fit=True,project_root=root,check_only=False,pilot=False,phase='measured',workers=1,run_dir=root/cfg['output_root']/'new')
    with pytest.raises(ValueError,match='recovery screen failed'):run.gain_prior_fit_main(args)

@pytest.mark.parametrize('mutation',['target','normalizer','indices'])
def test_baseline_trajectory_closure_checked_before_copy(parent,cfg,tmp_path,mutation):
    root,p,old=add_measured(parent,cfg);spec=run.conditional_gain_specs(cfg,'measured')[0]
    source=p/'prepared'/f"{spec['group']}.npz";target=np.zeros((24,120,3));normalizer=np.array([1.,2.,3.]);val=list(range(0,24,4))
    np.savez(source,target=target,feature_eeg=np.zeros((24,120)),feature_fnirs=np.zeros((24,300,2)),raw_optical_hb_native=np.zeros((24,300,2)),normalizer=normalizer)
    arrays=dict(target=target[val].copy(),normalizer=normalizer.copy(),trial_indices=np.array(val))
    if mutation=='target':arrays['target'][0,0,0]=1.
    elif mutation=='normalizer':arrays['normalizer'][1]=9.
    else:arrays['trial_indices'][0]=1
    cell=p/'cells'/run.nonlinear_cell_id(spec,'fixed_tau_logflow1','full');np.savez(cell/'trajectories.npz',**arrays)
    out=tmp_path/'destination'
    with pytest.raises(ValueError,match='baseline target/SD/trial'):run.prepare_conditional_gain_group((cfg,out,root,spec,False))
    assert not (out/'prepared'/source.with_suffix('.json').name).exists()
