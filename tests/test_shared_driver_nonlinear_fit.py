"""Exact discrete sensitivities, honest convergence and hidden-value isolation."""
from dataclasses import replace

import numpy as np
import pytest

from src.inference import t3a_balloon_robust_ssm as core
from src.inference.shared_driver_reconstruction import (
    nonlinear_driver_forward, replay_nonlinear_driver, fit_nonlinear_shared_driver,
    fit_nonlinear_shared_parameter,
)


def unpack(vector, steps):
    return vector[:steps], np.r_[vector[steps], np.exp(vector[steps+1:])]


def test_exact_rk4_sensitivities_match_finite_differences_including_tau():
    p, steps, dt = core.BalloonParameters(), 13, .3
    r = .035*np.sin(np.arange(steps)*.4)
    initial = np.array([.005, 1.03, 1.02, 1.015, 1.01])
    vector = np.r_[r, initial[0], np.log(initial[1:])]
    result = nonlinear_driver_forward(r, initial, p, dt)
    epsilon = 1e-6
    numeric = []
    for col in range(steps+5):
        plus, minus = vector.copy(), vector.copy()
        plus[col] += epsilon
        minus[col] -= epsilon
        yp = nonlinear_driver_forward(*unpack(plus, steps), p, dt, derivative=False)['canonical_prediction']
        ym = nonlinear_driver_forward(*unpack(minus, steps), p, dt, derivative=False)['canonical_prediction']
        numeric.append(((yp-ym)/(2*epsilon)).ravel())
    np.testing.assert_allclose(result['jacobian'], np.array(numeric).T, atol=3e-9, rtol=2e-6)
    yp = nonlinear_driver_forward(r, initial, replace(p, free=replace(p.free, tau=p.free.tau+epsilon)),
                                  dt, derivative=False)['canonical_prediction']
    ym = nonlinear_driver_forward(r, initial, replace(p, free=replace(p.free, tau=p.free.tau-epsilon)),
                                  dt, derivative=False)['canonical_prediction']
    np.testing.assert_allclose(result['tau_jacobian'], (yp-ym)/(2*epsilon), atol=3e-9, rtol=2e-6)
    yp = nonlinear_driver_forward(r, initial,
        replace(p,fixed=replace(p.fixed,neurovascular_gain=p.fixed.neurovascular_gain+epsilon)),
        dt,derivative=False)['canonical_prediction']
    ym = nonlinear_driver_forward(r, initial,
        replace(p,fixed=replace(p.fixed,neurovascular_gain=p.fixed.neurovascular_gain-epsilon)),
        dt,derivative=False)['canonical_prediction']
    np.testing.assert_allclose(result['neurovascular_gain_jacobian'],(yp-ym)/(2*epsilon),atol=3e-9,rtol=2e-6)
    # Last driver affects its EEG sample, never earlier vascular observations.
    np.testing.assert_array_equal(result['jacobian'][1::3, steps-1], 0.)
    np.testing.assert_array_equal(result['jacobian'][2::3, steps-1], 0.)


def test_fast_forward_matches_independent_checked_replay_outside_linear_range():
    p = core.BalloonParameters()
    r = .05+.03*np.sin(np.arange(20)*.2)
    initial = np.array([.03, 1.3, 1.2, 1.25, 1.1])
    forward = nonlinear_driver_forward(r, initial, p, .25)
    checked = replay_nonlinear_driver(r, initial, p, .25)
    assert checked['status'] == 'completed'
    np.testing.assert_allclose(forward['states'], checked['states'], atol=1e-14, rtol=0.)
    np.testing.assert_allclose(forward['canonical_prediction'], checked['canonical_prediction'], atol=1e-14, rtol=0.)


def test_noisefree_fit_recovers_shared_driver_without_vascular_innovations():
    p, steps, dt = core.BalloonParameters(), 14, .4
    r = .03*np.sin(np.arange(steps)*.4)
    initial = np.array([.005, 1.02, 1.01, 1.015, 1.008])
    target = nonlinear_driver_forward(r, initial, p, dt)['canonical_prediction']
    fit = fit_nonlinear_shared_driver(target, p, dt, penalty=0., initial_penalty=0.,
        sd=np.array([.03, .01, .01]), max_evaluations=81, return_jacobian=True)
    assert fit['status'] == 'completed', fit['starts']
    assert fit['converged']
    assert fit['scaled_gradient_inf_norm'] <= 1e-6
    assert fit['weighted_data_sse'] < 1e-8
    np.testing.assert_allclose(fit['driver'], r, atol=1e-7)
    assert fit['canonical_jacobian'].shape == (3*steps, steps+5)
    assert fit['replay_max_abs_difference'] < 1e-12
    assert fit['initial_penalty'] == 0.


def test_zero_driver_penalty_preserves_independent_initial_state_regularization():
    p, steps, dt = core.BalloonParameters(), 12, .3
    rest = np.array([0.,1.,1.,1.,1.])
    initial = np.array([.02,1.1,1.1,1.1,1.1])
    driver = .02*np.sin(np.arange(steps)*.3)
    target = nonlinear_driver_forward(driver,initial,p,dt)['canonical_prediction']
    common = dict(starts=[dict(driver=driver,initial_state=initial)],
                  penalty=0.,max_evaluations=81)
    unregularized = fit_nonlinear_shared_driver(target,p,dt,initial_penalty=0.,**common)
    regularized = fit_nonlinear_shared_driver(target,p,dt,initial_penalty=100.,**common)
    assert unregularized['status'] == regularized['status'] == 'completed'
    assert unregularized['objective'] < 1e-20
    assert regularized['initial_penalty'] == 100.
    assert regularized['penalty'] == 0.
    assert np.linalg.norm(regularized['initial_state']-rest) < .1*np.linalg.norm(initial-rest)
    assert regularized['objective'] == pytest.approx(regularized['weighted_data_sse']+
        100*np.sum((regularized['initial_state']-rest)**2))


def test_masked_hidden_values_and_processing_do_not_change_fit():
    p, steps, dt = core.BalloonParameters(), 10, .3
    target = nonlinear_driver_forward(.02*np.sin(np.arange(steps)), [0.,1.,1.,1.,1.],p,dt)['canonical_prediction']
    operator = np.kron(np.eye(steps)-np.ones((steps,steps))/steps, np.eye(3))
    target = (operator@target.ravel()).reshape(steps,3)
    visible = np.ones((steps,3), dtype=bool)
    visible[3:7,1:] = False
    first = fit_nonlinear_shared_driver(target,p,dt,mean_operator=operator,visible=visible,
        penalty=.1,initial_penalty=.1,max_evaluations=41)
    changed = target.copy()
    changed[~visible] = np.nan
    second = fit_nonlinear_shared_driver(changed,p,dt,mean_operator=operator,visible=visible,
        penalty=.1,initial_penalty=.1,max_evaluations=41)
    assert first['status'] == second['status']
    assert first['starts'] == second['starts']
    np.testing.assert_array_equal(first['prediction'], second['prediction'])


def test_physical_initial_penalty_gradient_and_total_objective_match_contract():
    p, steps, dt = core.BalloonParameters(), 7, .3
    initial = np.array([.01,1.1,1.04,1.05,1.02])
    driver = .01*np.sin(np.arange(steps))
    target = np.zeros((steps,3))
    penalty, initial_penalty = .2, 3.
    fit = fit_nonlinear_shared_driver(target,p,dt,penalty=penalty,initial_penalty=initial_penalty,
        starts=[dict(driver=driver,initial_state=initial)],max_evaluations=1,return_jacobian=True)
    perturbation = initial-np.array([0.,1.,1.,1.,1.])
    expected = (fit['weighted_data_sse']+penalty*np.sum(np.diff(driver,n=2)**2)/dt**3
                +initial_penalty*np.sum(perturbation**2))
    assert fit['objective'] == pytest.approx(expected)
    residual = np.r_[fit['prediction'].ravel(), np.sqrt(penalty)*np.diff(driver,n=2)/dt**1.5,
                     np.sqrt(initial_penalty)*perturbation]
    gradient = 2*fit['residual_jacobian'].T@residual
    vector = np.r_[driver,initial[0],np.log(initial[1:])]
    numeric = []
    for col in range(steps,steps+5):
        costs = []
        for direction in (1.,-1.):
            x = vector.copy(); x[col] += direction*1e-6
            r, initial_shifted = unpack(x,steps)
            shifted = fit_nonlinear_shared_driver(target,p,dt,penalty=penalty,
                initial_penalty=initial_penalty,max_evaluations=1,
                starts=[dict(driver=r,initial_state=initial_shifted)])
            costs.append(shifted['objective'])
        numeric.append((costs[0]-costs[1])/2e-6)
    np.testing.assert_allclose(gradient[steps:],numeric,atol=1e-8,rtol=1e-6)


def test_domain_failure_and_evaluation_budget_never_claim_convergence():
    p = core.BalloonParameters()
    target = np.full((8,3), .02)
    invalid = fit_nonlinear_shared_driver(target,p,.3, starts=[
        dict(driver=np.zeros(8),initial_state=np.array([0.,1.,1.,1.,4.]))])
    assert invalid['status'] == 'failed_domain'
    assert not invalid['converged']
    assert invalid['starts'][0]['status'] == 'infeasible_initial'
    budget = fit_nonlinear_shared_driver(target,p,.3,max_evaluations=1)
    assert budget['status'] == 'failed_numerical'
    assert not budget['converged']
    assert budget['convergence_reason'] == 'evaluation_budget'
    assert budget['evaluations'] == 1
    assert budget['gradient_inf_norm'] > 0


def test_infeasible_proposals_are_rejected_not_replaced_by_constant_residual():
    p = core.BalloonParameters()
    target = np.zeros((10,3)); target[:,0] = -10.
    result = fit_nonlinear_shared_driver(target,p,.3,sd=[.1,1.,1.],
        penalty=.01,initial_penalty=1.,max_evaluations=11)
    assert result['rejected_steps'] > 0
    assert result['domain_rejections'] > 0
    assert result['rejected_steps'] == sum(result[k] for k in
        ('domain_rejections','nondecreasing_rejections','derivative_rejections'))
    assert result['evaluations'] <= 11
    assert not result['converged']
    assert np.isfinite(result['objective'])


@pytest.mark.parametrize('initial', [[0.,0.,1.,1.,1.],[0.,1.,1.,1.,4.]])
def test_forward_rejects_mathematical_domain(initial):
    with pytest.raises(FloatingPointError):
        nonlinear_driver_forward(np.zeros(5),initial,core.BalloonParameters(),.25)


def test_forward_midway_domain_and_invalid_input_contracts():
    p = core.BalloonParameters()
    with pytest.raises(FloatingPointError):
        nonlinear_driver_forward(np.full(20,-10.),[0.,1.,1.,1.,1.],p,.25)
    for kwargs in [dict(max_evaluations=0),dict(penalty=-1),dict(substeps=0),
                   dict(sd=[1.,0.,1.]),dict(mean_operator=np.eye(3)),
                   dict(visible=np.zeros((5,3),dtype=bool)),dict(starts=[])]:
        with pytest.raises(ValueError):
            fit_nonlinear_shared_driver(np.zeros((5,3)),p,.25,**kwargs)


def test_three_trial_shared_tau_recovers_known_nonlinear_truth_from_rest():
    p, steps, dt = core.BalloonParameters(), 20, .4
    truth = replace(p,free=replace(p.free,tau=1.4))
    drivers = np.array([.07*np.sin(np.arange(steps)*(.25+.1*i)+i)+.025*np.cos(np.arange(steps)*.6)
                        for i in range(3)])
    target = np.array([nonlinear_driver_forward(r,[0.,1.,1.,1.,1.],truth,dt)['canonical_prediction']
                       for r in drivers])
    fit = fit_nonlinear_shared_parameter(target,p,dt,parameter_name='tau',parameter_bounds=(.5,10.),
        sd=[.03,.005,.003],penalty=0.,initial_penalty=0.)
    assert fit['status'] == 'completed', fit['starts']
    assert fit['tau'] == pytest.approx(1.4,abs=1e-5)
    assert fit['objective'] < 1e-8
    assert fit['expected_trials'] == fit['completed_trials'] == 3
    np.testing.assert_allclose(fit['driver'],drivers,atol=1e-6)
    assert fit['states'].shape == (3,steps,6)
    assert fit['information']['data_only']['value'] > 1.
    assert fit['information']['data_only']['value'] == pytest.approx(fit['information']['nuisance_regularized']['value'])
    assert fit['evaluations'] == fit['optimization_evaluations']+fit['validation_evaluations']
    assert fit['optimization_evaluations'] <= 240
    assert fit['validation_evaluations'] == 3


def test_shared_parameter_prior_is_once_per_group_and_is_not_data_information():
    p, steps, batch = core.BalloonParameters(), 6, 3
    start = dict(parameter_value=3.,driver=np.zeros((batch,steps)),initial_state=np.tile([0.,1.,1.,1.,1.],(batch,1)))
    result = fit_nonlinear_shared_parameter(np.zeros((batch,steps,3)),p,.25,starts=[start],
        max_evaluations=batch,parameter_prior_mean=2.,parameter_prior_log_sd=.5,
        parameter_name='tau',parameter_bounds=(.5,10.))
    expected = (np.log(3./2.)/.5)**2
    assert result['parameter_prior_cost'] == pytest.approx(expected)
    assert result['objective'] == pytest.approx(expected)
    assert result['log_parameter_gradient'] == pytest.approx(2*np.log(3./2.)/.5**2)
    assert result['information']['data_only']['value'] < 1e-20
    assert result['information']['nuisance_regularized']['value'] < 1e-20
    assert result['information']['with_parameter_prior'] == pytest.approx(4.)
    assert not result['converged']


@pytest.mark.parametrize('parameter_name',['tau','neurovascular_gain'])
def test_shared_log_parameter_objective_gradient_matches_finite_difference_with_prior(parameter_name):
    p, steps, batch, dt = core.BalloonParameters(), 8, 2, .3
    drivers = np.array([.04*np.sin(np.arange(steps)*.4+i) for i in range(batch)])
    initial = np.tile([.01,1.02,1.03,1.02,1.01],(batch,1))
    y = np.zeros((batch,steps,3))
    kwargs = dict(parameter_name=parameter_name,parameter_bounds=(.5,10.),max_evaluations=batch,sd=[.03,.01,.01],penalty=.1,
        initial_penalty=2.,parameter_prior_mean=2.,parameter_prior_log_sd=.4)
    center = fit_nonlinear_shared_parameter(y,p,dt,starts=[dict(parameter_value=1.7,driver=drivers,initial_state=initial)],**kwargs)
    costs = []
    for sign in (1.,-1.):
        fit = fit_nonlinear_shared_parameter(y,p,dt,
            starts=[dict(parameter_value=1.7*np.exp(sign*1e-6),driver=drivers,initial_state=initial)],**kwargs)
        costs.append(fit['objective'])
    assert center['log_parameter_gradient'] == pytest.approx((costs[0]-costs[1])/2e-6,abs=2e-7,rel=1e-6)


def test_shared_fit_failure_never_drops_trial_and_counts_partial_evaluations():
    p, steps, batch = core.BalloonParameters(), 12, 3
    drivers = np.zeros((batch,steps));drivers[1] = -10.
    initials = np.tile([0.,1.,1.,1.,1.],(batch,1))
    result = fit_nonlinear_shared_parameter(np.zeros((batch,steps,3)),p,.3,
        starts=[dict(parameter_value=2.,driver=drivers,initial_state=initials)],max_evaluations=batch,
        parameter_name='tau',parameter_bounds=(.5,10.))
    assert result['status'] == 'failed_domain'
    assert result['expected_trials'] == 3 and result['completed_trials'] == 0
    assert result['evaluations'] == 2  # trial 0 succeeded; trial 1 failed; trial 2 was not evaluated
    assert result['starts'][0]['failed_trial_evaluations'] == [0,1,0]
    assert result['starts'][0]['evaluation_failures'][0]['trial'] == 1


def test_shared_fit_mask_isolation_with_per_trial_sd_and_processing():
    p, steps, batch = core.BalloonParameters(), 6, 2
    y = np.random.default_rng(83).normal(0,.01,(batch,steps,3))
    mask = np.ones_like(y,dtype=bool);mask[:,2:4,1:] = False
    operator = np.kron(np.eye(steps)-np.ones((steps,steps))/steps,np.eye(3))
    kwargs = dict(parameter_name='tau',parameter_bounds=(.5,10.),visible=mask,mean_operator=operator,
                  sd=np.array([[.03,.01,.02],[.04,.02,.03]]),
                  max_evaluations=20,penalty=.1)
    before = fit_nonlinear_shared_parameter(y,p,.3,**kwargs)
    y[~mask] = np.nan
    after = fit_nonlinear_shared_parameter(y,p,.3,**kwargs)
    assert before['starts'] == after['starts']
    np.testing.assert_array_equal(before['prediction'],after['prediction'])
    assert before['tau'] == after['tau']


@pytest.mark.parametrize('starting_tau',[.5,1.])
def test_shared_tau_bound_is_constrained_stationary_point_not_clipped_state(starting_tau):
    p, batch, steps = core.BalloonParameters(), 2, 5
    start=dict(parameter_value=starting_tau,driver=np.zeros((batch,steps)),initial_state=np.tile([0.,1.,1.,1.,1.],(batch,1)))
    result=fit_nonlinear_shared_parameter(np.zeros((batch,steps,3)),p,.25,starts=[start],
        parameter_name='tau',parameter_bounds=(.5,4.),parameter_prior_mean=.25,parameter_prior_log_sd=.5,max_evaluations=20)
    assert result['status'] == 'completed'
    assert result['boundary_status'] == 'LOWER'
    assert result['log_parameter_gradient'] > 0
    assert result['projected_log_parameter_gradient'] == 0.
    assert result['tau'] == .5


def test_projected_data_information_excludes_both_nuisance_and_parameter_priors():
    p, batch, steps = core.BalloonParameters(), 2, 9
    drivers=np.array([.04*np.sin(np.arange(steps)*.4+i) for i in range(batch)])
    start=dict(parameter_value=1.7,driver=drivers,initial_state=np.tile([.01,1.02,1.03,1.02,1.01],(batch,1)))
    kwargs=dict(parameter_name='tau',parameter_bounds=(.5,10.),starts=[start],max_evaluations=batch,sd=[.03,.005,.003])
    free=fit_nonlinear_shared_parameter(np.zeros((batch,steps,3)),p,.3,penalty=0.,initial_penalty=0.,**kwargs)
    regularized=fit_nonlinear_shared_parameter(np.zeros((batch,steps,3)),p,.3,
        penalty=2.,initial_penalty=100.,parameter_prior_mean=2.,parameter_prior_log_sd=.1,**kwargs)
    assert regularized['information']['data_only']['value'] == pytest.approx(free['information']['data_only']['value'])
    assert regularized['information']['nuisance_regularized']['value'] > free['information']['data_only']['value']
    assert regularized['information']['with_parameter_prior'] == pytest.approx(
        regularized['information']['nuisance_regularized']['value']+100.)
    assert regularized['objective'] == pytest.approx(regularized['weighted_data_sse']+
        regularized['penalty']*regularized['roughness']+regularized['initial_state_penalty_cost']+
        regularized['parameter_prior_cost'])
    assert all(rank <= steps+5 for rank in regularized['information']['data_only']['nuisance_ranks'])


def test_shared_tau_batch_contracts_fail_before_forward_calls():
    p, y = core.BalloonParameters(), np.zeros((2,5,3))
    for options in [dict(max_evaluations=1),dict(parameter_bounds=(0.,2.)),dict(parameter_prior_mean=2.),
                    dict(sd=np.ones((5,3))),dict(visible=np.zeros_like(y,dtype=bool)),
                    dict(information_rcond=0.),dict(initial_penalty=-1.),dict(starts=[])]:
        options.setdefault('parameter_bounds',(.5,10.))
        with pytest.raises(ValueError):fit_nonlinear_shared_parameter(y,p,.25,parameter_name='tau',**options)


def test_three_trial_shared_gain_recovery_preserves_tau_and_other_fixed_gauge():
    p,steps,dt=core.BalloonParameters(),20,.4
    truth=replace(p,fixed=replace(p.fixed,neurovascular_gain=.35))
    drivers=np.array([.07*np.sin(np.arange(steps)*(.25+.1*i)+i)+.025*np.cos(np.arange(steps)*.6)
                      for i in range(3)])
    target=np.array([nonlinear_driver_forward(r,[0.,1.,1.,1.,1.],truth,dt)['canonical_prediction']
                     for r in drivers])
    result=fit_nonlinear_shared_parameter(target,p,dt,parameter_name='neurovascular_gain',
        parameter_bounds=(.05,5.),sd=[.03,.005,.003],penalty=0.,initial_penalty=0.)
    assert result['status']=='completed',result['starts']
    assert result['parameter_name']=='neurovascular_gain'
    assert result['parameter_value']==pytest.approx(.35,abs=1e-5)
    assert result['neurovascular_gain']==result['parameter_value']
    assert result['tau']==p.free.tau
    assert p.fixed.neurovascular_gain==1.
    assert result['information']['coordinate']=='log_neurovascular_gain'
    assert result['objective']<1e-8


def test_driver_amplitude_residual_objective_and_exact_gradient_independent_of_other_weights():
    p,steps,dt=core.BalloonParameters(),7,.3
    driver=.03+.01*np.sin(np.arange(steps))
    initial=np.array([0.,1.,1.,1.,1.]);y=np.zeros((steps,3))
    prior_sd=np.linspace(.5,1.5,steps);weight=3.
    kwargs=dict(starts=[dict(driver=driver,initial_state=initial)],penalty=0.,initial_penalty=0.,
                max_evaluations=1,return_jacobian=True)
    base=fit_nonlinear_shared_driver(y,p,dt,**kwargs)
    zero=fit_nonlinear_shared_driver(y,p,dt,driver_amplitude_weight=0.,driver_prior_sd=prior_sd,**kwargs)
    fitted=fit_nonlinear_shared_driver(y,p,dt,driver_amplitude_weight=weight,driver_prior_sd=prior_sd,**kwargs)
    assert base['objective']==zero['objective']
    np.testing.assert_array_equal(base['residual_jacobian'],zero['residual_jacobian'])
    expected=weight*dt*np.sum((driver/prior_sd)**2)
    assert fitted['objective']-base['objective']==pytest.approx(expected)
    assert fitted['driver_amplitude_cost']==pytest.approx(expected)
    base_residual=np.r_[base['prediction'].ravel(),np.zeros(steps-2+5)]
    fitted_residual=np.r_[fitted['prediction'].ravel(),np.zeros(steps-2),
                         np.sqrt(weight*dt)*driver/prior_sd,np.zeros(5)]
    difference=2*(fitted['residual_jacobian'].T@fitted_residual-base['residual_jacobian'].T@base_residual)
    np.testing.assert_allclose(difference[:steps],2*weight*dt*driver/prior_sd**2,atol=1e-14)
    np.testing.assert_allclose(difference[steps:],0.,atol=1e-14)
    # Constant drive is no longer an unpenalized direction, unlike curvature.
    amp_j=fitted['residual_jacobian'][3*steps+steps-2:3*steps+2*steps-2,:steps]
    assert np.linalg.norm(amp_j@np.ones(steps))>0


def test_shared_driver_amplitude_prior_sd_is_per_trial_and_enters_cost_once_per_sample():
    p,batch,steps,dt=core.BalloonParameters(),2,7,.3
    drivers=np.tile(.03+.01*np.sin(np.arange(steps)),(batch,1))
    start=dict(parameter_value=1.,driver=drivers,initial_state=np.tile([0.,1.,1.,1.,1.],(batch,1)))
    fitted=fit_nonlinear_shared_parameter(np.zeros((batch,steps,3)),p,dt,
        parameter_name='neurovascular_gain',parameter_bounds=(.05,5.),starts=[start],
        max_evaluations=batch,penalty=0.,initial_penalty=0.,driver_amplitude_weight=3.,driver_prior_sd=[.5,1.])
    expected=3*dt*np.sum((drivers/np.array([.5,1.])[:,None])**2,axis=1)
    np.testing.assert_allclose(fitted['trial_driver_amplitude_cost'],expected)
    assert fitted['objective']==pytest.approx(fitted['weighted_data_sse']+sum(expected))
    assert fitted['driver_amplitude_cost']==pytest.approx(sum(expected))


@pytest.mark.parametrize('options',[dict(driver_amplitude_weight=-1.),dict(driver_prior_sd=0.),
    dict(driver_prior_sd=float('nan')),dict(driver_prior_sd=np.ones((1,5)))])
def test_single_driver_amplitude_contract_rejects_bad_weight_or_sd(options):
    with pytest.raises(ValueError):
        fit_nonlinear_shared_driver(np.zeros((5,3)),core.BalloonParameters(),.25,**options)


@pytest.mark.parametrize('options',[dict(driver_amplitude_weight=-1.),dict(driver_prior_sd=0.),
    dict(driver_prior_sd=[1.,float('inf')]),dict(driver_prior_sd=np.ones(5)),
    dict(parameter_name='alpha'),dict(parameter_name=['tau','neurovascular_gain'])])
def test_shared_parameter_and_amplitude_contract_rejects_undeclared_freedoms(options):
    kwargs=dict(parameter_name='neurovascular_gain',parameter_bounds=(.05,5.))
    kwargs.update(options)
    with pytest.raises(ValueError):
        fit_nonlinear_shared_parameter(np.zeros((2,5,3)),core.BalloonParameters(),.25,**kwargs)


def test_shared_parameter_trace_is_observational_and_failed_fit_can_warm_start():
    p, steps, dt, batch = core.BalloonParameters(), 16, .4, 3
    truth = replace(p, fixed=replace(p.fixed, neurovascular_gain=.35))
    drivers = np.array([.07*np.sin(np.arange(steps)*(.25+.1*i)+i)
                        for i in range(batch)])
    y = np.array([nonlinear_driver_forward(r, [0.,1.,1.,1.,1.], truth, dt)
                  ['canonical_prediction'] for r in drivers])
    options = dict(parameter_name='neurovascular_gain', parameter_bounds=(.05,5.),
                   sd=[.03,.005,.003], penalty=0., initial_penalty=0.,
                   gradient_tolerance=1e-6)
    # Initial evaluation plus one accepted step: deliberately unfinished.
    plain = fit_nonlinear_shared_parameter(y,p,dt,max_evaluations=3*batch,**options)
    short = fit_nonlinear_shared_parameter(y,p,dt,max_evaluations=3*batch,
                                          record_trace=True,**options)
    assert short['status']==plain['status']=='failed_numerical'
    assert short['convergence_reason']=='evaluation_budget'
    assert 'trace' not in plain and 'trace' not in plain['starts'][0]
    for key in ('driver','initial_state','states','parameter_value','objective',
                'optimization_evaluations','projected_scaled_gradient_inf_norm'):
        np.testing.assert_array_equal(short[key],plain[key])
    trace=short['trace']
    assert trace==short['starts'][short['selected_start']]['trace']
    assert trace[0]['iteration']==0 and trace[0]['evaluations']==batch
    assert trace[-1]['evaluations']==short['optimization_evaluations']
    assert trace[-1]['objective']==short['objective']
    assert trace[-1]['accepted_fraction']>0 and trace[-1]['accepted_step_norm']>0
    warm=dict(parameter_value=short['parameter_value'],driver=short['driver'],
              initial_state=short['initial_state'])
    resumed=fit_nonlinear_shared_parameter(y,p,dt,starts=[warm],max_evaluations=300,
                                           record_trace=True,**options)
    assert resumed['status']=='completed',resumed['starts']
    assert resumed['trace'][0]['objective']==pytest.approx(short['objective'],rel=1e-12,abs=1e-12)
    assert resumed['objective']<short['objective']
    assert resumed['parameter_value']==pytest.approx(.35,abs=1e-5)
    assert resumed['trace'][-1]['projected_scaled_gradient_inf_norm']<=1e-6
    assert all(b['objective']<=a['objective'] for a,b in zip(resumed['trace'],resumed['trace'][1:]))


def test_shared_parameter_trace_warm_start_preserves_all_regularization_terms():
    p,batch,steps,dt=core.BalloonParameters(),2,7,.3
    driver=np.tile(.03+.01*np.sin(np.arange(steps)),(batch,1))
    initial=np.tile([.02,1.03,.98,1.01,.99],(batch,1))
    options=dict(parameter_name='neurovascular_gain',parameter_bounds=(.05,5.),
        sd=[.3,.2,.1],penalty=.01,initial_penalty=100.,driver_amplitude_weight=1.,
        driver_prior_sd=.1,parameter_prior_mean=1.,parameter_prior_log_sd=.7,
        record_trace=True,max_evaluations=batch)
    first=fit_nonlinear_shared_parameter(np.zeros((batch,steps,3)),p,dt,
        starts=[dict(parameter_value=.7,driver=driver,initial_state=initial)],**options)
    second=fit_nonlinear_shared_parameter(np.zeros((batch,steps,3)),p,dt,
        starts=[dict(parameter_value=first['parameter_value'],driver=first['driver'],
                     initial_state=first['initial_state'])],**options)
    expected=(first['weighted_data_sse']+.01*first['roughness']+
        first['initial_state_penalty_cost']+first['driver_amplitude_cost']+first['parameter_prior_cost'])
    assert first['trace'][0]['objective']==pytest.approx(expected)
    assert second['trace'][0]['objective']==pytest.approx(first['objective'],rel=1e-13)
    assert len(first['trace'])==1
    assert first['trace'][0]['accepted_fraction'] is None
    assert not first['converged'] and not second['converged']


def test_rest_data_cannot_identify_tau_even_when_prior_selects_tau():
    p=core.BalloonParameters();y=np.zeros((3,8,3))
    for value in (1.,2.,4.):
        start=dict(parameter_value=value,driver=np.zeros((3,8)),initial_state=np.tile([0.,1.,1.,1.,1.],(3,1)))
        fit=fit_nonlinear_shared_parameter(y,p,.25,parameter_name='tau',parameter_bounds=(.5,8.),
            starts=[start],penalty=.01,initial_penalty=100.)
        assert fit['converged'] and fit['parameter_value']==pytest.approx(value)
        assert fit['information']['data_only']['value']<1e-20
        assert fit['information']['nuisance_regularized']['value']<1e-20
    prior=fit_nonlinear_shared_parameter(y,p,.25,parameter_name='tau',parameter_bounds=(.5,8.),
        starts=[start],parameter_prior_mean=2.,parameter_prior_log_sd=np.log(2.))
    assert prior['converged'] and prior['parameter_value']==pytest.approx(2.,abs=1e-5)
    assert prior['information']['data_only']['value']<1e-20
    assert prior['information']['with_parameter_prior']==pytest.approx(1/np.log(2.)**2)


@pytest.mark.parametrize('coordinate_factor', [.5, 2.])
def test_shared_tau_unit_reexpression_preserves_physical_fit(coordinate_factor, record_property):
    """Changing observation units is not shrinking the assumed physical state."""
    import json

    p, steps, dt, batch = core.BalloonParameters(), 10, .3, 2
    truth = replace(p, free=replace(p.free, tau=1.6))
    drivers = np.array([.025*np.sin(np.arange(steps)*(.3+.1*i)+i)
                        for i in range(batch)])
    physical_initial = np.array([[.002,1.015,1.008,1.01,1.006],
                                 [-.001,1.008,1.012,1.009,1.005]])
    canonical = np.array([nonlinear_driver_forward(r, initial, truth, dt)['canonical_prediction']
                          for r, initial in zip(drivers, physical_initial)])
    # A nonidentity observation map prevents the test from only exercising
    # scalar weighting. It is deliberately fixed, not estimated from targets.
    operator = np.eye(3*steps)+.15*np.diag(np.ones(3*steps-3), k=3)
    y = np.array([(operator@curve.ravel()).reshape(steps,3) for curve in canonical])
    sd = np.array([.03,.01,.008])
    options = dict(parameter_name='tau', parameter_bounds=(.5,4.),
        starts=[dict(parameter_value=2.3, driver=.5*drivers,
                     initial_state=np.tile([0.,1.,1.,1.,1.],(batch,1)))],
        penalty=.01, initial_penalty=2., parameter_prior_mean=1.8,
        parameter_prior_log_sd=.5, max_evaluations=200, max_iterations=40,
        gradient_tolerance=1e-6)
    # The parameter prior makes this a short, well-conditioned software check;
    # equality of fits is not a physiological tau-recovery qualification.
    original = fit_nonlinear_shared_parameter(y,p,dt,sd=sd,mean_operator=operator,**options)
    scaled = fit_nonlinear_shared_parameter(coordinate_factor*y,p,dt,
        sd=coordinate_factor*sd,mean_operator=coordinate_factor*operator,**options)
    assert original['status'] == scaled['status'] == 'completed', (original['starts'], scaled['starts'])
    assert original['converged'] and scaled['converged']
    assert original['convergence_reason'] == scaled['convergence_reason']
    assert original['optimization_evaluations'] == scaled['optimization_evaluations']
    for field in ('parameter_value','driver','initial_state','states','objective','canonical_prediction'):
        np.testing.assert_allclose(scaled[field],original[field],rtol=1e-10,atol=1e-11)
    np.testing.assert_allclose(scaled['prediction'],coordinate_factor*original['prediction'],rtol=1e-10,atol=1e-11)
    assert original['optimization_evaluations'] > batch  # Nontrivial optimization occurred.
    assert original['objective'] > 0.  # The same nonzero regularized objective was preserved.
    record_property('unit_reexpression',json.dumps(dict(
        coordinate_factor=coordinate_factor, batch=batch, steps=steps,
        original_status=original['status'], scaled_status=scaled['status'],
        tau_original=original['parameter_value'], tau_scaled=scaled['parameter_value'],
        objective_original=original['objective'], objective_scaled=scaled['objective'],
        driver_max_abs_difference=float(np.max(abs(scaled['driver']-original['driver']))),
        initial_max_abs_difference=float(np.max(abs(scaled['initial_state']-original['initial_state']))),
        optimization_evaluations=original['optimization_evaluations'],
        interpretation='software_unit_reexpression_invariance_not_physiological_recovery')))
