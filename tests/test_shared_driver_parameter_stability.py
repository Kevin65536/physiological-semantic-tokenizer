"""Array-only contracts for profiles, covariance, continuous support and priors."""
from dataclasses import replace
import numpy as np
import pytest
from src.inference.shared_driver_reconstruction import (
    fit_nonlinear_shared_parameters, nonlinear_driver_forward,
)
from src.inference.t3a_balloon_robust_ssm import BalloonParameters
from src.data.ssm_prepared import fit_feature_coordinate


def fixture():
    p=BalloonParameters()
    r=.025*np.sin(np.arange(28)*.3)
    y=nonlinear_driver_forward(r,np.r_[0.,np.ones(4)],p,.25,numerical_backend='numba')['canonical_prediction']
    return p,r,y[None]


def test_fixed_parameter_profile_and_prior_accounting():
    p,r,y=fixture()
    opts=dict(sd=[.02,.03,.02],penalty=.01,initial_penalty=[100.],flow_prior_weight=1.)
    fixed=fit_nonlinear_shared_parameters(y,p,.25,parameter_names=[],parameter_bounds=[],**opts)
    assert fixed['converged']
    assert fixed['parameter_values'].shape==(0,)
    np.testing.assert_allclose(fixed['objective'],sum(fixed['objective_components'].values()))
    shrunk=fit_nonlinear_shared_parameters(y,p,.25,parameter_names=['tau'],parameter_bounds=[[.5,8.]],
        parameter_prior_log_mean=[np.log(3.)],parameter_prior_precision=[4.],**opts)
    np.testing.assert_allclose(shrunk['objective_components']['parameter_prior'],4*np.log(shrunk['tau']/3.)**2)
    np.testing.assert_allclose(shrunk['raw_parameter_gradient'],sum(shrunk['raw_parameter_gradient_components'].values()))


def test_boundary_warm_start_allows_only_float_roundoff():
    p,r,y=fixture()
    opts=dict(parameter_names=['neurovascular_gain'],parameter_bounds=[[.1,10.]],sd=[.02,.03,.02],max_iterations=0)
    a=fit_nonlinear_shared_parameters(y,p,.25,starts=[dict(parameter_values=[np.exp(np.log(10.))])],**opts)
    assert a['parameter_values'][0]==pytest.approx(10.)
    with pytest.raises(ValueError,match='outside'):
        fit_nonlinear_shared_parameters(y,p,.25,starts=[dict(parameter_values=[10.00001])],**opts)


def test_visible_covariance_whitening_never_reads_hidden_targets():
    p,r,y=fixture();mask=np.ones_like(y,dtype=bool);mask[:,10:14,1:]=False
    count=mask.sum();w=np.eye(count);w[1:, :-1]+=.03*np.eye(count-1)
    opts=dict(parameter_names=['tau'],parameter_bounds=[[.5,8.]],sd=[.02,.03,.02],
              visible=mask,observation_whiteners=[w],max_iterations=200)
    a=fit_nonlinear_shared_parameters(y,p,.25,**opts)
    y[~mask]=1e15
    b=fit_nonlinear_shared_parameters(y,p,.25,**opts)
    np.testing.assert_array_equal(a['parameter_values'],b['parameter_values'])
    with pytest.raises(ValueError,match='whitener'):
        fit_nonlinear_shared_parameters(y,p,.25,**dict(opts,observation_whiteners=[np.eye(y.size)]))


def test_data_information_cannot_exceed_conditional_and_prior_is_separate():
    p,r,y=fixture()
    result=fit_nonlinear_shared_parameters(y,p,.25,parameter_names=['tau','neurovascular_gain'],
        parameter_bounds=[[.5,8.],[.1,10.]],sd=[.02,.03,.02],max_iterations=200)
    info=result['information']
    assert np.linalg.eigvalsh(info['conditional_data']['matrix']-info['data_only']['matrix']).min()>-1e-8
    assert np.linalg.eigvalsh(info['nuisance_regularized']['matrix']-info['data_only']['matrix']).min()>-1e-8


def test_continuity_regularization_omits_only_cross_boundary_curvature():
    p,r,y=fixture()
    # A piecewise-linear driver has curvature only at the artificial boundary.
    r=np.r_[np.arange(14)*.001, .1+np.arange(14)*.002]
    yy=nonlinear_driver_forward(r,np.r_[0.,np.ones(4)],p,.25,numerical_backend='numba')['canonical_prediction'][None]
    opts=dict(parameter_names=[],parameter_bounds=[],sd=[.02,.03,.02],starts=[dict(parameter_values=[],driver=r[None])],
        penalty=1.,flow_prior_weight=0.,initial_penalty=0.,gradient_tolerance=1e-9)
    a=fit_nonlinear_shared_parameters(yy,p,.25,curvature_breaks=[14],**opts)
    assert a['objective']<1e-20
    with pytest.raises(ValueError,match='curvature_breaks'):
        fit_nonlinear_shared_parameters(yy,p,.25,curvature_breaks=[2],**opts)


def test_coordinate_fits_training_only_and_preserves_hb_amplitude_ratio():
    rng=np.random.default_rng(4);e=rng.normal(size=(120,30));h=rng.normal(size=(120,1))*np.array([[4.,1.]])
    c=fit_feature_coordinate([(e,h)])
    assert c['sd'][1]/c['sd'][2]==pytest.approx(4.)
    assert np.linalg.norm(c['pc'])==pytest.approx(1.)


def test_common_profile_requires_both_records_and_each_modality():
    from src.inference.shared_driver_stability import profile_intersections
    rows=[]
    for value,costs,errors in [(1.,[10.,10.],[[.2,.2,.2],[.2,.2,.2]]),
                               (2.,[10.01,10.01],[[.2,.23,.2],[.2,.2,.2]]),
                               (4.,[10.01,10.01],[[.2,.2,.2],[.2,.2,.2]])]:
        rows.append(dict(values=[value],fits=[dict(objective=q,nrmse=e,accepted=True) for q,e in zip(costs,errors)]))
    profile=dict(rows=rows,bounds=[[1.,4.]])
    result=profile_intersections(profile,[20.,20.],inners=[.05],tolerances=[.01])
    assert not result[0]['exists']  # Only interior node damages HbO beyond margin.
    rows[1]['fits'][0]['nrmse'][1]=.205
    assert profile_intersections(profile,[20.,20.],inners=[.05],tolerances=[.01])[0]['exists']
    rows[1]['fits'][1]['accepted']=False
    assert not profile_intersections(profile,[20.,20.],inners=[.05],tolerances=[.01])[0]['exists']


def test_working_covariance_hidden_marginal_matches_full_covariance():
    from src.inference.shared_driver_stability import working_whitener
    cal=dict(ar1=.6,covariance=[[1.,.2,0],[.2,1.,.3],[0,.3,1.]])
    full=working_whitener(8,cal);mask=np.ones((8,3),bool);mask[2:4,1:]=False
    sub=working_whitener(8,cal,mask)
    expected=np.linalg.inv(full.T@full)[np.ix_(mask.ravel(),mask.ravel())]
    np.testing.assert_allclose(np.linalg.inv(sub.T@sub),expected,rtol=1e-12,atol=1e-12)


def test_metadata_pairing_requires_exact_proportions_and_same_native_montage():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).parents[1]/'experiments/scripts/evaluate_shared_driver_parameter_stability.py'
    spec=importlib.util.spec_from_file_location('parameter_stability_test_runner',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    base=dict(dataset='eeg_fnirs_single_trial',task='motor_imagery',eeg_start_s=5.,duration_s=90.,
              anchor=dict(eeg_channels=['C3','C4'],hb_channel='C3h'),signature=(('LMI',1),('RMI',2)))
    a=dict(base,record='session_00');b=dict(base,record='session_02',signature=(('LMI',2),('RMI',1)))
    assert module.select_pair([a,b],['motor_imagery'])[0] is None
    b['signature']=a['signature'];pair,kind=module.select_pair([a,b],['motor_imagery'])
    assert len(pair)==2 and kind=='independent_native_record'


def test_effective_pair_requires_entire_vector_and_excludes_shared_null():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).parents[1]/'experiments/scripts/evaluate_shared_driver_parameter_stability.py'
    spec=importlib.util.spec_from_file_location('stability_metrics_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=module.config(module.DEFAULT)
    def fit(tau,beta):
        return dict(parameters=dict(tau=tau,neurovascular_gain=beta,kappa=.64),accepted=True,nrmse=[.2,.3,.4])
    result=dict(model='P2',variant='N0',fits=[fit(2,1),fit(2.1,1.05)],transfer=[fit(2,1),fit(2.1,1.05)])
    assert module.result_metrics(result,cfg)['effective_pair']
    result['fits'][1]['parameters']['neurovascular_gain']=1.5
    assert not module.result_metrics(result,cfg)['effective_pair']
    result['fits'][1]=fit(2,1);result['variant']='Hfull'
    assert not module.result_metrics(result,cfg)['effective_pair']
    missing=module.result_metrics(dict(model='P2',variant='N0',status='failed'),cfg)
    assert not missing['completed'] and not missing['accepted'] and not missing['effective_pair']
    original=dict(model='P2',variant='N0',fits=[fit(2,1),fit(2.1,1.05)],transfer=[fit(2,1),fit(2.1,1.05)])
    audit=dict(checks=[dict(side=0,legal_substeps8=True,metric_max_abs_error=0.,max_substep_difference_development_sd=.001),
        dict(side=1,legal_substeps8=True,metric_max_abs_error=0.,max_substep_difference_development_sd=.014)])
    screened,excluded=module.apply_numerical_audit(original,audit)
    assert len(screened['fits'])==2 and original['fits'][1]['accepted']  # Evidence and denominator survive.
    assert screened['fits'][0]['accepted'] and not screened['fits'][1]['accepted']
    assert excluded[0]['reasons']==['integration_precision']
    assert not module.result_metrics(screened,cfg)['effective_pair']


def test_vectorized_icc_bootstrap_preserves_subject_resampling():
    from src.inference.shared_driver_stability import icc_absolute,bootstrap_icc
    values=np.random.default_rng(27).normal(size=(11,2));rng=np.random.default_rng(43)
    expected=[icc_absolute(values[rng.integers(0,11,11)]) for _ in range(200)]
    np.testing.assert_allclose(bootstrap_icc(values,43,200),np.quantile(expected,[.025,.975]),rtol=1e-13,atol=1e-13)
    assert np.isnan(bootstrap_icc(np.ones((6,2)),43,200)).all()
