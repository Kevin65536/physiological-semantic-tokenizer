import copy
import numpy as np
import pytest

from src.inference.observation_baselines import (
    first_difference_noise, student_difference_mad,
)
from scipy.integrate import quad
from scipy.stats import norm,t
from experiments import evaluate_step5 as step5


def test_full_contract_stops_scope_and_prior_drift():
    cfg=step5.load_config()
    for field,value in [('subjects',['subject_24']),('replication_subjects',['subject_19']),('sessions',['session_02']),('heldout_trial_positions',[0,1])]:
        bad=copy.deepcopy(cfg);bad['measured'][field]=value
        with pytest.raises(ValueError):step5.validate_config(bad)
    bad=copy.deepcopy(cfg);bad['axes']['Z']['prior']='logit_uniform'
    with pytest.raises(ValueError):step5.validate_config(bad)
    bad=copy.deepcopy(cfg);bad['replicates_per_axis']=4
    with pytest.raises(ValueError):step5.validate_config(bad)


def test_adaptive_reference_carries_nonresampled_weights():
    cfg=step5.load_config(); y=np.array([[.03,np.nan,np.nan]])
    std=cfg['model']['initial_state_std'][0];scale=cfg['model']['observation_scale'][0]
    exact=quad(lambda r:norm.pdf(r,scale=std)*t.pdf((.03-r)/scale,df=5)/scale,-1,1)[0]
    result=step5.adaptive_particle_filter(y,cfg,'G',0.,100000,19)
    assert result['parameter_log_likelihood']==pytest.approx(np.log(exact),abs=.01)
    missing=step5.adaptive_particle_filter(np.full((4,3),np.nan),cfg,'G',0.,512,15)
    assert missing['parameter_log_likelihood']==pytest.approx(0.,abs=1e-12)
    assert missing['resampling_count']==0
    assert missing['unique_ancestor_fraction']==1.
    # With every update missing, all resampling schedules estimate exactly one.
    every=step5.adaptive_particle_filter(np.full((4,3),np.nan),cfg,'G',0.,512,15,resample_fraction=1.01)
    assert every['parameter_log_likelihood']==pytest.approx(0.,abs=1e-12)


def test_small_calibration_keeps_likelihood_score_and_state_targets_distinct():
    cfg=step5.load_config();cfg['model']['steps']=12
    cfg['inference'].update(grid_points=5,maximum_grid_points=9,bootstrap_repetitions=20)
    cfg['oracle'].update(steps=24,grid_points=5,maximum_grid_points=9)
    row,arrays=step5.calibration_case(cfg,'G',10)
    assert row['parameter_log_likelihood'].shape==row['predictive_score'].shape
    assert not np.allclose(row['parameter_log_likelihood'],row['predictive_score'])
    assert set(row['conditional_true_parameter_state_metrics'])==set(step5.localization.TARGETS)
    assert arrays['truth_states'].shape==(12,6)
    assert arrays['conditional_clean_variance'].shape==(12,3)
    assert 0<=row['joint_posterior']['rank_u']<=1


def test_cluster_summary_uses_replicates_not_time_samples():
    cfg=step5.load_config();cfg['inference']['bootstrap_repetitions']=100
    result=step5.cluster_summary([.8,.9,1.],cfg,15)
    assert result['independent_units']==3
    assert result['mean']==pytest.approx(.9)
    with pytest.raises(ValueError):step5.cluster_summary(np.zeros((3,160)),cfg)


def test_parameter_mixture_and_masks_preserve_target_isolation():
    cfg=step5.load_config();cfg['model']['steps']=32
    cfg['teacher']['frozen_training_parameter_mixture_points']=3
    cfg['inference']['quadrature_order']=7
    generated=step5.localization.generate_matched(cfg,'G',.1,41)
    grid=np.linspace(-.6,.6,9)
    fit=dict(grid=grid,parameter_log_likelihood=-np.square(grid-.1)/.04)
    y=generated['observations']
    hidden,mask,columns=step5.masked_input(y,'center_fNIRS',8)
    changed=y.copy();changed[np.ix_(mask,columns)]=1e6
    hidden2,_,_=step5.masked_input(changed,'center_fNIRS',8)
    first=step5.mixture_state(hidden,cfg,'G',fit)
    second=step5.mixture_state(hidden2,cfg,'G',fit)
    np.testing.assert_array_equal(first['clean_mean'],second['clean_mean'])
    np.testing.assert_allclose(first['clean_variance'],first['conditional_clean_variance']+first['parameter_clean_variance'])
    assert np.isfinite(step5.predictive_log_score(y,first,mask,columns,cfg))
    assert np.all(first['parameter_clean_variance']>=0)


def test_small_teacher_case_uses_independent_training_distribution(tmp_path):
    cfg=step5.load_config();cfg['model']['steps']=24
    cfg['inference'].update(quadrature_order=7,grid_points=5,maximum_grid_points=9,mask_center_steps=8)
    cfg['teacher'].update(frozen_training_parameter_mixture_points=3,parameter_mixture_check_points=5,task_template_training_trials=2)
    grid=np.linspace(-.6,.6,5)
    step5.write_json(tmp_path/'case_G_00.json',dict(truth=.1,grid=grid,parameter_log_likelihood=-grid**2/.08))
    row,arrays=step5.teacher_case(cfg,'G',0,tmp_path)
    assert row['axis']=='G'
    for law,results in row['laws'].items():
        assert set(results)=={'U0_FIXED','U1_G'}
        for result in results.values():
            assert set(result['masks'])==set(cfg['teacher']['masks'])
            assert result['physical_pass']
            assert len(result['masks']['center_EEG']['controls'])==4
        assert arrays[f'{law}_observations'].shape==(24,3)
    seeds=[step5.case_seed(cfg,a,r,s) for a in ('G','W','Z','R') for r in range(60) for s in range(120)]
    assert len(seeds)==len(set(seeds))
    # Independent candidate panels must not inherit another axis's failure.
    # U0's explicitly cross-scenario sensitivity still needs every panel.
    w=copy.deepcopy(row);w['axis']='W'
    reference=copy.deepcopy(row);reference['axis']='R'
    for law in row['laws']:
        w['laws'][law]['U1_W']=w['laws'][law].pop('U1_G')
        reference['laws'][law].pop('U1_G')
    cfg['teacher']['matching_replicates']=1
    summary=step5.teacher_summary(cfg,[w,reference,dict(axis='G',replicate=0,execution='failed'),dict(axis='Z',replicate=0,execution='failed')],{})
    assert summary['candidates']['U1_W']['checks']['complete_registered_cases']
    assert not summary['candidates']['U0_FIXED']['checks']['complete_registered_cases']
    assert not summary['candidates']['U1_G']['teacher_qualified']
    assert not summary['candidates']['U1_Z']['teacher_qualified']


def test_u3_coordinate_map_preserves_gwz_interpretation():
    cfg=step5.load_config();g,w,zeta=.2,-.1,.8
    p,_=step5.multi_parameter_model(cfg,[g,w,zeta])
    ref=cfg['model']['reference']
    assert p.fixed.neurovascular_gain/p.fixed.gamma==pytest.approx(ref['beta']/ref['gamma']*np.exp(g))
    assert np.sqrt(p.fixed.gamma)==pytest.approx(np.sqrt(ref['gamma'])*np.exp(w))
    assert p.free.kappa/(2*np.sqrt(p.fixed.gamma))==pytest.approx(zeta)


@pytest.mark.parametrize('mask_name',['center_EEG','center_fNIRS','whole_EEG','whole_fNIRS'])
def test_native_preprocessing_hides_target_before_filter_power_and_resampling(mask_name):
    rng=np.random.default_rng(17)
    eeg=rng.normal(size=(4000,3));fnirs=np.exp(.01*rng.normal(size=(200,2,2)))
    changed_eeg=eeg.copy();changed_fnirs=fnirs.copy()
    values=changed_eeg if mask_name.endswith('EEG') else changed_fnirs
    hz=200 if mask_name.endswith('EEG') else 10
    hidden=np.ones(len(values),dtype=bool)
    if mask_name.startswith('center'):
        hidden=(np.arange(len(values))/hz>=8)&(np.arange(len(values))/hz<12)
    values[hidden]=1e9
    first=step5.preprocess_native_trial(eeg,fnirs,mask_name=mask_name)
    second=step5.preprocess_native_trial(changed_eeg,changed_fnirs,mask_name=mask_name)
    for key in first:np.testing.assert_array_equal(first[key],second[key])
    columns=[0] if mask_name.endswith('EEG') else [1,2]
    assert not first['observation_mask'][:,columns].all()
    for key in ('eeg_log_power','fnirs'):
        assert first[key].shape[0]==80


def test_teacher_failure_keeps_case_identity_and_cannot_qualify(monkeypatch,tmp_path):
    def fail(*args):raise FloatingPointError('fixture transition overflow')
    monkeypatch.setattr(step5,'teacher_case',fail)
    cfg=step5.load_config()
    row,arrays=step5.teacher_job(cfg,'Z',12,tmp_path)
    assert arrays is None and row['execution']=='failed'
    assert row['axis']=='Z' and row['replicate']==12
    assert 'fixture transition overflow' in row['traceback']
    summary=step5.teacher_summary(cfg,[row],{})
    assert summary['failed_cases']==1
    assert not summary['measured_stage_eligible']
    assert not summary['qualified_candidates']


def test_measured_boundary_fails_before_any_array_or_noise_evidence_read(monkeypatch,tmp_path):
    base,spec,metadata=step5.load_measured_config()
    with pytest.raises(ValueError,match='development boundary'):
        step5.prepare_measured_subject(base,spec,metadata,'subject_24',1.)
    step5.write_json(tmp_path/'manifest.json',dict(stage='a1',execution='completed'))
    step5.write_json(tmp_path/'summary.json',dict(candidates={'U1_W':{'teacher_qualified':False}}))
    def forbidden(*args,**kwargs):raise AssertionError('read reached before qualification')
    monkeypatch.setattr(step5,'noise_estimator_preflight',forbidden)
    monkeypatch.setattr(step5,'prepare_measured_subject',forbidden)
    with pytest.raises(ValueError,match='complete synthetic panel'):
        step5.run_measured(step5.MEASURED_CONFIG,tmp_path/'unused',tmp_path)


def test_student_difference_noise_scale_and_reference_gauge():
    base,spec,_=step5.load_measured_config()
    constant=student_difference_mad(5.)
    noise=np.random.default_rng(44).standard_t(5,size=(50000,3))*[.08,.025,.015]
    estimate=first_difference_noise([noise],constant)
    np.testing.assert_allclose(estimate,[.08,.025,.015],rtol=.015)
    gauge=step5.reference_observation_gauge(base,spec)
    assert gauge['eeg']>0 and gauge['fnirs_common']>0
    assert np.isfinite(gauge['coordinate_sd']).all()


def test_frozen_measured_projection_preserves_raw_mask_isolation():
    base,spec,_=step5.load_measured_config();rng=np.random.default_rng(21)
    trials=[(rng.normal(size=(6000,3)),np.exp(.02*rng.normal(size=(300,2,2)))) for _ in range(3)]
    features=[step5.preprocess_native_trial(*pair) for pair in trials[:2]]
    projection=step5.fit_measured_projection(features,base,spec)
    heldout=trials[2]
    for mask in ('center_EEG','center_fNIRS'):
        changed=[v.copy() for v in heldout]
        changed[0][2600:3400]=1e9 if mask.endswith('EEG') else changed[0][2600:3400]
        changed[1][130:170]=1e9 if mask.endswith('fNIRS') else changed[1][130:170]
        first=step5.apply_measured_projection(step5.preprocess_native_trial(*heldout,mask_name=mask),projection)
        second=step5.apply_measured_projection(step5.preprocess_native_trial(*changed,mask_name=mask),projection)
        np.testing.assert_array_equal(first,second)
    assert projection['training_trials']==2 and not projection['task_labels_used']
    eligible=np.ones(2,dtype=bool);eligible[projection['fnirs_pair']]=False
    selected=step5.fit_measured_projection(features,base,spec,eligible)
    assert selected['fnirs_pair']!=projection['fnirs_pair']
    # Excluded pairs retain finite diagnostics; -inf is only an argmax sentinel.
    import json
    json.dumps(step5.localization.jsonable(selected),allow_nan=False)
    with pytest.raises(ValueError,match='no valid training fNIRS pair'):
        step5.fit_measured_projection(features,base,spec,np.zeros(2,dtype=bool))


def test_small_measured_trial_has_frozen_nulls_and_observation_only_metrics():
    base,spec,_=step5.load_measured_config();cfg=copy.deepcopy(base);cfg['model']['steps']=32
    cfg['inference']['quadrature_order']=7;cfg['teacher']['frozen_training_parameter_mixture_points']=3
    spec['posterior']['mixture_check_points']=5
    y=step5.localization.generate_matched(cfg,'W',.05,19)['observations']
    donor=step5.localization.generate_matched(cfg,'W',.05,21)['observations']
    arrays=dict(targets=y[None],training=np.stack([y,donor]),donors=donor[None],templates=np.zeros_like(y)[None],
                masked_inputs=np.array([[step5.masked_input(y,m,16)[0] for m in base['teacher']['masks']]]))
    fit=dict(grid=np.linspace(-.5,.5,9),parameter_log_likelihood=-np.linspace(-.5,.5,9)**2/.08)
    detail=dict(trial_inventory=[dict(heldout=True,sample_id='synthetic-only')])
    row,payload=step5.measured_trial_case(base,spec,'synthetic',0,detail,arrays,cfg,fit)
    assert row['execution']=='completed'
    assert set(row['candidates'])=={'U0_FIXED','U1_W'}
    result=row['candidates']['U1_W']
    assert len(result['masks']['center_fNIRS']['controls'])==4
    assert 'observation_coverage95' in result['masks']['center_EEG']['metrics']['EEG']
    assert np.isfinite(payload['U1_W_full_mean']).all()
    assert result['physical_pass']


def test_local_grid_resolves_boundary_mass_without_changing_support(monkeypatch,tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from scipy.integrate import quad
    base,spec,_=step5.load_measured_config();spec['posterior']['quadrature_check_subject_positions']=[]
    lo,hi=base['axes']['W']['bounds'];slope=500.
    monkeypatch.setattr(step5,'ProcessPoolExecutor',lambda **kw:ThreadPoolExecutor(max_workers=2))
    monkeypatch.setattr(step5,'measured_likelihood_chunk',lambda config,training,points,order:list(-slope*(points-lo)))
    grid=np.linspace(lo,hi,65)
    fit={'subject_01':dict(grid=grid,parameter_log_likelihood=-slope*(grid-lo),grid_cdf_difference=.5,quadrature_cdf_difference=None)}
    prepared={'subject_01':dict(config=base,arrays={'training':np.zeros((1,2,3))})}
    result=step5.refine_measured_posterior_mass(base,spec,prepared,fit,2,tmp_path)['subject_01']
    assert result['grid_cdf_difference']<=.02
    assert set(grid)<=set(result['grid'])
    posterior=step5.localization.posterior_grid(base,'W',result['grid'],result['parameter_log_likelihood'])
    prior=step5.localization.prior(base,'W')
    normalizer=quad(lambda u:np.exp(-u)*prior.pdf(lo+u/slope),0,80)[0]
    checks=np.linspace(lo,lo+.02,41)
    exact=np.array([quad(lambda u:np.exp(-u)*prior.pdf(lo+u/slope),0,slope*(x-lo))[0]/normalizer for x in checks])
    assert max(abs(np.interp(checks,result['grid'],posterior['cdf'])-exact))<.025


def test_measured_fixed_baseline_failure_does_not_skip_primary_candidate(monkeypatch):
    base,spec,_=step5.load_measured_config();calls=[]
    def evaluate(*args):
        candidate=args[-1];calls.append(candidate)
        if candidate=='U0_FIXED':raise FloatingPointError('baseline fixture failure')
        return {'physical_pass':True},{}
    monkeypatch.setattr(step5,'measured_candidate_trial',evaluate)
    row,_=step5.measured_trial_case(base,spec,'synthetic',0,{'trial_inventory':[{'heldout':True}]},
                                   {'targets':np.zeros((1,8,3))},base,{})
    assert calls==['U0_FIXED','U1_W']
    assert row['execution']=='partial'
    assert row['candidates']['U0_FIXED']['execution']=='failed'
    assert row['candidates']['U1_W']['execution']=='completed'
