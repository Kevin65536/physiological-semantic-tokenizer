from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, t, multivariate_normal
from experiments.evaluate_step5a_inference_consistency import load_config,model,generate_matched
from src.inference import t3a_balloon_joint_ssm as joint
from src.inference import t3a_balloon_robust_ssm as core


def test_joint_update_matches_scalar_student_t_integrals():
    cfg=load_config(); p,c=model(cfg,'G',0.)
    spec=core.BalloonObservationSpec().resolved(p.fixed)
    mean=np.zeros(6); cov=np.diag(np.square(c.initial_state_std))
    cov[0,1]=cov[1,0]=.001
    observed=np.array([.1,np.nan,np.nan]); available=np.isfinite(observed)
    integrand=lambda x: norm.pdf(x,scale=c.initial_state_std[0])*t.pdf((.1-x)/spec.observation_scale[0],df=spec.student_nu)/spec.observation_scale[0]
    density=quad(integrand,-2,2,epsabs=1e-12)[0]
    expected_mean=quad(lambda x:x*integrand(x),-2,2,epsabs=1e-12)[0]/density
    expected_var=quad(lambda x:(x-expected_mean)**2*integrand(x),-2,2,epsabs=1e-12)[0]/density
    updated,ucov,ll,_=joint.joint_observation_update(mean,cov,observed,available,p,spec,order=41)
    assert ll==pytest.approx(np.log(density),abs=1e-8)
    assert updated[0]==pytest.approx(expected_mean,abs=1e-8)
    assert ucov[0,0]==pytest.approx(expected_var,abs=1e-8)
    assert updated[1]==pytest.approx(cov[1,0]/cov[0,0]*expected_mean,abs=1e-8)


def test_linear_gaussian_joint_update_retains_cross_covariance(monkeypatch):
    cfg=load_config(); p,c=model(cfg,'G',0.)
    p=replace(p,fixed=replace(p.fixed,student_nu=1e12))
    spec=core.BalloonObservationSpec().resolved(p.fixed)
    h=core._observation_physical_matrix(p,spec)
    monkeypatch.setattr(joint,'_batch_observation',lambda z,*args:z@h.T)
    mean=np.zeros(6); cov=np.diag(np.square(c.initial_state_std))
    cov[0,4]=cov[4,0]=.0002
    observed=np.array([.03,.02,-.01])
    r=np.diag(np.square(spec.observation_scale)); pred=h@cov@h.T+r
    gain=np.linalg.solve(pred,h@cov).T
    expected_mean=gain@observed; expected_cov=cov-gain@h@cov
    updated,ucov,ll,_=joint.joint_observation_update(mean,cov,observed,np.ones(3,dtype=bool),p,spec,order=31,calculate_score=False)
    np.testing.assert_allclose(updated,expected_mean,atol=1e-9)
    np.testing.assert_allclose(ucov,expected_cov,atol=1e-9)
    assert ll==pytest.approx(multivariate_normal.logpdf(observed,cov=pred),abs=1e-8)


def test_missing_values_never_enter_quadrature_and_no_data_preserves_prior():
    cfg=load_config(); p,c=model(cfg,'W',.1)
    spec=core.BalloonObservationSpec().resolved(p.fixed)
    mean=np.zeros(6); cov=np.diag(np.square(c.initial_state_std))
    output=joint.joint_observation_update(mean,cov,np.full(3,np.nan),np.zeros(3,dtype=bool),p,spec)
    np.testing.assert_array_equal(output[0],mean)
    np.testing.assert_array_equal(output[1],cov)
    assert output[2:]==(0.,0.)
    visible=np.array([.1,np.nan,np.nan]); malicious=np.array([.1,1e100,-1e100])
    a=joint.joint_observation_update(mean,cov,visible,np.array([1,0,0],dtype=bool),p,spec)
    b=joint.joint_observation_update(mean,cov,malicious,np.array([1,0,0],dtype=bool),p,spec)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)


def test_joint_smoother_finite_masked_and_variances_have_distinct_meaning():
    cfg=load_config(); p,c=model(cfg,'Z',.8)
    generated=generate_matched(cfg,'Z',.8,70)
    y=generated['observations']; y[24:40,1:]=np.nan
    result=joint.smooth_balloon_joint(y,p,config=c,quadrature_order=13)
    assert np.isfinite(result.trajectory_mean).all()
    assert np.linalg.eigvalsh(result.state_covariance).min()>0
    assert np.all(result.state_posterior_variance>=0)
    np.testing.assert_allclose(result.total_observation_variance-result.state_posterior_variance,
                               np.broadcast_to(result.observation_variance,result.state_posterior_variance.shape))
    assert result.parameter_log_likelihood==pytest.approx(joint.parameter_log_likelihood(y,p,config=c),abs=1e-10)
    assert result.predictive_score!=result.parameter_log_likelihood
    with pytest.raises(ValueError):
        joint.parameter_log_likelihood(np.array([[np.inf,0.,0.]]),p,config=c)


def test_gaussian_driver_path_matches_dense_joint_conditioning_with_gaps():
    cfg=load_config();p,c=model(cfg,'G',0.)
    p=replace(p,fixed=replace(p.fixed,student_nu=1e12))
    count=20;observed=np.random.default_rng(15).normal(0,.04,count)
    y=np.full((count,3),np.nan);y[:,0]=observed;y[5:9,0]=np.nan
    available=np.isfinite(y[:,0])
    _,transition=core.rk4_transition_with_jacobian(np.zeros(6),p,c)
    rho=transition[0,0];innovation=p.fixed.process_std[0]**2*c.dt
    variances=[c.initial_state_std[0]**2]
    for _ in range(1,count):variances.append(rho**2*variances[-1]+innovation)
    prior=np.array([[rho**abs(i-j)*variances[min(i,j)] for j in range(count)] for i in range(count)])
    predicted=prior[np.ix_(available,available)]+np.eye(available.sum())*p.fixed.observation_scale[0]**2
    cross=prior[:,available]
    mean=cross@np.linalg.solve(predicted,observed[available])
    covariance=prior-cross@np.linalg.solve(predicted,cross.T)
    result=joint.smooth_balloon_joint(y,p,config=c,quadrature_order=31)
    np.testing.assert_allclose(result.state_mean[:,0],mean,atol=1e-9)
    np.testing.assert_allclose(result.state_covariance[:,0,0],np.diag(covariance),atol=1e-9)
    assert result.parameter_log_likelihood==pytest.approx(multivariate_normal.logpdf(observed[available],cov=predicted),abs=1e-8)
