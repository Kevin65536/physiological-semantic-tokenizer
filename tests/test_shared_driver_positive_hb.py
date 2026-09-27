"""Positive initial HbO coordinates preserve objective and legacy stationarity."""
from dataclasses import replace
import numpy as np
import pytest
from src.inference import shared_driver_reconstruction as lib
from src.inference import t3a_balloon_robust_ssm as core


def case():
    p=core.BalloonParameters();p=replace(p,free=replace(p.free,tau=1.6))
    r=.05*np.sin(np.arange(20)*.4)
    initial=np.array([.01,1.02,1.01,.96,.93])
    y=lib.nonlinear_driver_forward(r,initial,p,.25)['canonical_prediction']
    return p,r,initial,y


def test_objective_and_exact_chain_with_physical_penalties_and_flow():
    p,r,initial,y=case();n=len(r)
    x=np.r_[r,lib._encode_positive_hb_initial(initial,p)]
    old=np.r_[r,initial[0],np.log(initial[1:])]
    mask=np.ones_like(y,bool);mask[4:7,0]=False
    op=np.eye(3*n);op[0,3]=.1
    reg=np.zeros((n,n+5));reg[:,:n]=np.eye(n)*.2
    def evaluate(z,mode='positive_hb',derivative=True):
        return lib._nonlinear_trial_objective(z,p,.25,op,y,mask,np.ones_like(y)*.1,reg,100.,4,derivative,
            flow_prior_weight=1.,flow_prior_log_sd=np.log(2),initial_coordinates=mode)
    a,b=evaluate(x),evaluate(old,'independent_logs')
    np.testing.assert_allclose(a['residual'],b['residual'],atol=2e-14)
    np.testing.assert_allclose(a['stationarity_jacobian'],b['jacobian'],atol=2e-13)
    eps=1e-6
    numerical=np.column_stack([(evaluate(x+eps*d,derivative=False)['residual']-evaluate(x-eps*d,derivative=False)['residual'])/(2*eps) for d in np.eye(n+5)])
    np.testing.assert_allclose(a['jacobian'],numerical,rtol=2e-5,atol=2e-8)
    np.testing.assert_allclose(a['initial'],initial,atol=2e-16)


@pytest.mark.parametrize('shared',[False,True])
def test_single_and_shared_truth_recovery(shared):
    p,r,initial,y=case();kw=dict(sd=np.full(3,.01),penalty=0.,initial_penalty=0.,initial_coordinates='positive_hb',
        gradient_tolerance=1e-7,return_jacobian=True)
    if shared:
        result=lib.fit_nonlinear_shared_parameter(y[None],p,.25,parameter_name='tau',parameter_bounds=(.5,8.),
            starts=[dict(parameter_value=2.,driver=np.zeros((1,len(r))),initial_state=np.array([[0.,1.,1.,1.,1.]]))],
            max_evaluations=240,max_iterations=60,**kw)
        assert result['tau']==pytest.approx(p.free.tau,abs=2e-6)
        np.testing.assert_allclose(result['initial_state'][0],initial,atol=2e-6)
    else:
        result=lib.fit_nonlinear_shared_driver(y,p,.25,max_evaluations=160,**kw)
        np.testing.assert_allclose(result['initial_state'],initial,atol=2e-6)
    assert result['converged'] and result['objective']<1e-10
    assert result['initial_state_free_parameters']==5
    assert result['stationarity_coordinate_basis']=='legacy_free_initial_coordinates'


@pytest.mark.parametrize('shared',[False,True])
def test_near_zero_hbo_native_gradient_cannot_fake_convergence(shared):
    p=core.BalloonParameters();n=14;ratio=p.fixed.Q0/p.fixed.P0
    initial=np.array([0.,1.,1.,ratio+1e-10,1.]);driver=np.zeros(n)
    forward=lib.nonlinear_driver_forward(driver,initial,p,.25)
    j=forward['jacobian'];g=np.zeros(n+5);g[n+3]=1.;g[n+4]=-ratio/initial[3]
    if shared:
        j=np.column_stack((j,p.free.tau*forward['tau_jacobian'].ravel()));g=np.r_[g,0.]
    residual=np.linalg.lstsq(j.T,g,rcond=1e-12)[0]
    y=forward['canonical_prediction']-residual.reshape(n,3)
    options=dict(penalty=0.,initial_penalty=0.,max_evaluations=1,gradient_tolerance=1e-4,initial_coordinates='positive_hb')
    if shared:
        result=lib.fit_nonlinear_shared_parameter(y[None],p,.25,parameter_name='tau',parameter_bounds=(.5,8.),
            starts=[dict(parameter_value=p.free.tau,driver=driver[None],initial_state=initial[None])],**options)
        reported=result['projected_scaled_gradient_inf_norm']
    else:
        result=lib.fit_nonlinear_shared_driver(y,p,.25,starts=[dict(driver=driver,initial_state=initial)],**options)
        reported=result['scaled_gradient_inf_norm']
    assert result['optimization_coordinate_scaled_gradient_inf_norm']<1e-4
    assert reported>1e-3
    assert not result['converged'] and result['status']=='failed_numerical'


@pytest.mark.parametrize('shared',[False,True])
def test_options_and_zero_initial_hbo_fail_closed(shared):
    p,r,initial,y=case()
    fn=lib.fit_nonlinear_shared_parameter if shared else lib.fit_nonlinear_shared_driver
    options=dict(parameter_name='tau',parameter_bounds=(.5,8.)) if shared else {}
    values=y[None] if shared else y
    with pytest.raises(ValueError,match='cannot be combined'):
        fn(values,p,.25,initial_coordinates='positive_hb',tie_total_hb_to_volume=True,**options)
    with pytest.raises(ValueError,match='initial_coordinates'):
        fn(values,p,.25,initial_coordinates='typo',**options)
    initial[3]=p.fixed.Q0/p.fixed.P0*initial[4]
    start=dict(driver=r[None] if shared else r,initial_state=initial[None] if shared else initial)
    if shared:start['parameter_value']=p.free.tau
    result=fn(values,p,.25,initial_coordinates='positive_hb',starts=[start],**options)
    assert result['status']=='failed_domain' and not result['converged']
    assert result['initial_coordinates']=='positive_hb'


def test_near_boundary_start_same_physical_objective_not_a_claim_of_rescue():
    p=core.BalloonParameters();n=20;r=.04*np.sin(np.arange(n)*.4)
    truth=np.array([.01,1.,1.,.8,.8]);y=lib.nonlinear_driver_forward(r,truth,p,.25)['canonical_prediction']
    initial=np.array([.03,1.02,1.1,p.fixed.Q0/p.fixed.P0*.6+1e-7,.6])
    kwargs=dict(sd=np.full(3,.02),penalty=.01,initial_penalty=100.,flow_prior_weight=1.,flow_prior_log_sd=np.log(2),
        starts=[dict(driver=np.zeros(n),initial_state=initial)],max_evaluations=160,gradient_tolerance=1e-4)
    old=lib.fit_nonlinear_shared_driver(y,p,.25,**kwargs)
    new=lib.fit_nonlinear_shared_driver(y,p,.25,initial_coordinates='positive_hb',**kwargs)
    assert old['converged'] and new['converged']
    assert new['objective']==pytest.approx(old['objective'],abs=5e-8)
    assert np.all(p.fixed.P0*new['states'][:,4]>p.fixed.Q0*new['states'][:,5])
    assert new['scaled_gradient_inf_norm']<1e-4
    # No required speed advantage: positivity coordinates can need more calls.


@pytest.mark.parametrize('shared',[False,True])
def test_explicit_legacy_mode_exact_default(shared):
    p,r,initial,y=case();kw=dict(penalty=.01,initial_penalty=100.,max_evaluations=20,return_jacobian=True)
    fn=lib.fit_nonlinear_shared_parameter if shared else lib.fit_nonlinear_shared_driver
    if shared:kw.update(parameter_name='tau',parameter_bounds=(.5,8.))
    a=fn(y[None] if shared else y,p,.25,**kw)
    b=fn(y[None] if shared else y,p,.25,initial_coordinates='independent_logs',**kw)
    for key in ['driver','initial_state','states','prediction','objective','converged','evaluations']:
        np.testing.assert_array_equal(a[key],b[key])
