"""Delta-only intervention: independent dynamics, exact tangents and fit contracts."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.inference.shared_driver_reconstruction import (
    fit_decay_shared_driver, fit_nonlinear_shared_driver,
)
from src.inference.shared_driver_rk4 import compiled_forward, decay_forward
from src.inference.t3a_balloon_robust_ssm import BalloonParameters


def parameters():
    base = BalloonParameters()
    return replace(base, free=replace(base.free, kappa=.64))


def independent_forward(driver, initial, p, dt, multiplier, coefficients):
    """Adaptive solve_ivp, with independent physical RHS and ZOH boundaries."""
    f, tau, kappa = p.fixed, p.free.tau, p.free.kappa
    frequencies = np.pi*np.arange(len(coefficients))/(len(driver)*dt)

    def rhs(t, x, value):
        s, flow, volume, total, deoxy = x
        loss = volume**(1/f.alpha-1)
        volume_dot = (flow-volume**(1/f.alpha))/tau
        forcing = coefficients@np.cos(frequencies*(t+dt/2))
        extraction = 1-(1-f.E0)**(1/flow)
        return [f.neurovascular_gain*value-kappa*s-f.gamma*(flow-1), s,
                volume_dot, volume_dot-multiplier*loss*(total-volume)/tau+forcing,
                (flow*extraction/f.E0-loss*deoxy)/tau]

    states = [np.asarray(initial)]
    for i in range(len(driver)-1):
        solved = solve_ivp(rhs, (i*dt, (i+1)*dt), states[-1], args=(driver[i],),
                           rtol=2e-12, atol=2e-14, method='DOP853')
        assert solved.success
        states.append(solved.y[:, -1])
    return np.array(states)


@pytest.mark.parametrize('multiplier', [1., .1, 0.])
def test_matches_independent_adaptive_solver_with_continuous_input(multiplier):
    p, dt, n = parameters(), .2, 19
    driver = .035*np.sin(np.arange(n)*.35)
    initial = np.array([.01, 1.04, 1.025, .98, 1.01])
    coefficients = np.array([.002, -.0015, .0006])
    exact = independent_forward(driver, initial, p, dt, multiplier, coefficients)
    actual = decay_forward(driver, initial, p, dt, decay_multiplier=multiplier,
                           forcing_coefficients=coefficients, substeps=8)
    np.testing.assert_allclose(actual['states'][:, 1:], exact, rtol=2e-10, atol=8e-11)


@pytest.mark.parametrize('substeps', [4, 8])
def test_lambda_one_unforced_reproduces_existing_compiled_map_and_tangent(substeps):
    p, n = parameters(), 21
    driver = .03*np.cos(np.arange(n)/4)
    initial = np.array([.01, 1.05, 1.02, .99, 1.01])
    existing = compiled_forward(driver, initial, p, .25, substeps, True, True, state=True)
    actual = decay_forward(driver, initial, p, .25, substeps=substeps, derivative=True)
    for key in ('states', 'canonical_prediction', 'jacobian', 'flow_jacobian', 'state_jacobian'):
        np.testing.assert_allclose(actual[key], existing[key], atol=2e-14, rtol=2e-13)
    assert actual['forcing_coefficients'].shape == (0,)
    np.testing.assert_array_equal(actual['forcing'], np.zeros(n))


@pytest.mark.parametrize('multiplier', [1., .1, 0.])
def test_exact_tangent_for_every_driver_initial_and_input_coordinate(multiplier):
    p, dt, n = parameters(), .3, 11
    driver = .025*np.sin(np.arange(n)/2)
    initial = np.array([.01, 1.025, 1.015, 1.03, .99])
    coefficients = np.array([.001, -.0007, .0002])
    vector = np.r_[driver, initial[0], np.log(initial[1:]), coefficients]
    actual = decay_forward(driver, initial, p, dt, decay_multiplier=multiplier,
        forcing_coefficients=coefficients, derivative=True)
    columns, state_columns = [], []
    for j in range(len(vector)):
        observations, states = [], []
        for sign in (1, -1):
            x = vector.copy()
            x[j] += sign*1e-6
            shifted = decay_forward(x[:n], np.r_[x[n], np.exp(x[n+1:n+5])], p, dt,
                decay_multiplier=multiplier, forcing_coefficients=x[n+5:])
            observations.append(shifted['canonical_prediction'])
            states.append(shifted['states'])
        columns.append(((observations[0]-observations[1])/2e-6).ravel())
        state_columns.append((states[0]-states[1])/2e-6)
    np.testing.assert_allclose(actual['jacobian'], np.array(columns).T, atol=5e-10, rtol=3e-6)
    np.testing.assert_allclose(actual['state_jacobian'], np.stack(state_columns, axis=-1),
                               atol=1e-9, rtol=3e-6)
    np.testing.assert_array_equal(actual['jacobian'][1::3, n-1], 0.)
    np.testing.assert_array_equal(actual['jacobian'][2::3, n-1], 0.)
    np.testing.assert_array_equal(actual['state_jacobian'][:, [0, 1, 2, 3, 5], n+5:], 0.)


@pytest.mark.parametrize('initial_delta', [.1, -.1])
def test_zero_decay_preserves_signed_delta_while_positive_decay_loses_it(initial_delta):
    p, n, dt = parameters(), 81, .25
    clock = np.arange(n)*dt
    initial = np.array([0., 1., 1., 1.+initial_delta, 1.])
    for multiplier in (1., .1, 0.):
        actual = decay_forward(np.zeros(n), initial, p, dt, decay_multiplier=multiplier)
        delta = actual['states'][:, 4]-actual['states'][:, 3]
        expected = initial_delta*np.exp(-multiplier*clock/p.free.tau)
        np.testing.assert_allclose(delta, expected, atol=3e-10, rtol=2e-7)
        np.testing.assert_allclose(actual['retention_integral'], clock/p.free.tau, atol=1e-14)
        np.testing.assert_allclose(actual['decay_retention'], np.exp(-multiplier*clock/p.free.tau))
        assert np.all(np.sign(delta) == np.sign(initial_delta))
        if multiplier == 0.:
            np.testing.assert_allclose(delta, initial_delta, atol=4e-16, rtol=0.)


def test_sustained_input_is_integrated_into_delta_with_correct_units_and_phase():
    p, n, dt, coefficient = parameters(), 120, .25, .01
    clock = np.arange(n)*dt
    actual = decay_forward(np.zeros(n), [0., 1., 1., 1., 1.], p, dt,
        decay_multiplier=0., forcing_coefficients=[0., coefficient])
    omega = np.pi/(n*dt)
    expected = coefficient/omega*(np.sin(omega*(clock+dt/2))-np.sin(omega*dt/2))
    delta = actual['states'][:, 4]-actual['states'][:, 3]
    np.testing.assert_allclose(delta, expected, atol=5e-13, rtol=2e-10)
    np.testing.assert_allclose(actual['forcing'], coefficient*np.cos(omega*(clock+dt/2)))
    assert np.all(np.diff(delta[:n//2]) > 0)
    # Changing the initial output by a cosine would fail this initial condition.
    assert delta[0] == 0.
    assert delta[n//2-1] > .09
    short = decay_forward(np.zeros(120), [0., 1., 1., 1., 1.], p, .25,
        forcing_coefficients=np.zeros(7))
    long = decay_forward(np.zeros(360), [0., 1., 1., 1., 1.], p, .25,
        forcing_coefficients=np.zeros(19))
    assert short['forcing_angular_frequencies'][0] == 0.
    np.testing.assert_array_equal(short['forcing_basis'][:, 0], 1.)
    assert short['forcing_angular_frequencies'][-1] == pytest.approx(long['forcing_angular_frequencies'][-1])


def test_dc_input_grows_delta_linearly_and_survives_output_demeaning():
    p, n, dt, coefficient = parameters(), 120, .25, .01
    clock = np.arange(n)*dt
    actual = decay_forward(np.zeros(n), [0., 1., 1., 1., 1.], p, dt,
        decay_multiplier=0., forcing_coefficients=[coefficient], derivative=True)
    delta = actual['states'][:, 4]-actual['states'][:, 3]
    # Repeated physical-coordinate additions accumulate floating-point roundoff.
    np.testing.assert_allclose(delta, coefficient*clock, atol=8e-14, rtol=0.)
    np.testing.assert_array_equal(actual['forcing'], coefficient)
    assert delta[0] == 0.
    np.testing.assert_allclose(actual['jacobian'][1::3, n+5], p.fixed.P0*clock,
                               atol=2e-13, rtol=0.)
    operator = np.kron(np.eye(n)-np.ones((n, n))/n, np.eye(3))
    processed = (operator@actual['canonical_prediction'].ravel()).reshape(n, 3)
    np.testing.assert_allclose(processed[:, 1], p.fixed.P0*coefficient*(clock-clock.mean()),
                               atol=8e-14, rtol=0.)
    assert np.linalg.norm(processed[:, 1]) > .5
    # The DC coefficient remains a state/s input, with a sample-integrated prior.
    fitted = fit_decay_shared_driver(processed, p, dt, mean_operator=operator,
        decay_multiplier=0., forcing_modes=1, forcing_sd=.02, forcing_weight=.7,
        penalty=0., initial_penalty=0., flow_prior_weight=0., max_evaluations=1,
        starts=[dict(driver=np.zeros(n), initial_state=[0., 1., 1., 1., 1.],
                     forcing_coefficients=[coefficient])])
    assert fitted['weighted_data_sse'] < 1e-25
    assert fitted['forcing_prior_cost'] == pytest.approx(.7*n*dt*(coefficient/.02)**2)


def test_upstream_volume_deoxy_and_hbr_are_identical_for_fixed_driver_and_initial():
    p, n = parameters(), 40
    driver = .02+.03*np.sin(np.arange(n)/5)
    initial = np.array([.01, 1.04, .99, 1.05, 1.01])
    baseline = decay_forward(driver, initial, p, .25)
    for multiplier in (1., .1, 0.):
        actual = decay_forward(driver, initial, p, .25, decay_multiplier=multiplier,
                               forcing_coefficients=[.002, -.001])
        np.testing.assert_array_equal(actual['states'][:, [0, 1, 2, 3, 5]],
                                      baseline['states'][:, [0, 1, 2, 3, 5]])
        np.testing.assert_array_equal(actual['canonical_prediction'][:, [0, 2]],
                                      baseline['canonical_prediction'][:, [0, 2]])
        assert np.linalg.norm(actual['canonical_prediction'][:, 1]-baseline['canonical_prediction'][:, 1]) > .001


def test_lambda_one_no_input_fit_objective_matches_existing_solver():
    p, n, dt = parameters(), 18, .25
    initial = np.array([.005, 1.02, 1.01, 1.015, 1.005])
    driver = .02*np.sin(np.arange(n)/3)
    target = decay_forward(driver, initial, p, dt)['canonical_prediction']
    operator = np.kron(np.eye(n)-np.ones((n, n))/n, np.eye(3))
    target = (operator@target.ravel()).reshape(n, 3)
    common = dict(mean_operator=operator, sd=[.02, .01, .006], penalty=.01,
        initial_penalty=100., flow_prior_weight=1., flow_prior_log_sd=np.log(2),
        gradient_tolerance=1e-7, max_evaluations=150)
    existing = fit_nonlinear_shared_driver(target, p, dt, numerical_backend='numba', **common)
    actual = fit_decay_shared_driver(target, p, dt, **common)
    assert actual['converged'] and existing['converged']
    assert actual['objective'] == pytest.approx(existing['objective'], abs=2e-12)
    np.testing.assert_allclose(actual['prediction'], existing['prediction'], atol=2e-11)
    np.testing.assert_allclose(actual['initial_state'], existing['initial_state'], atol=2e-10)
    assert sum(actual['costs'].values()) == pytest.approx(actual['objective'])


def test_input_objective_and_gradient_use_same_sample_prior_for_each_decay():
    p, n, dt = parameters(), 10, .3
    driver = .02*np.sin(np.arange(n)/3)
    initial = np.array([.01, 1.03, 1.01, 1.02, .99])
    coefficients = np.array([.002, -.001])
    vector = np.r_[driver, initial[0], np.log(initial[1:]), coefficients]
    target = np.zeros((n, 3))
    common = dict(sd=[.03, .01, .006], penalty=.03, initial_penalty=3.,
        flow_prior_weight=.5, flow_prior_log_sd=.6, forcing_sd=.02, forcing_weight=.7,
        forcing_modes=2, max_evaluations=1, return_jacobian=True)

    def at(x, multiplier):
        return fit_decay_shared_driver(target, p, dt, decay_multiplier=multiplier,
            starts=[dict(driver=x[:n], initial_state=np.r_[x[n], np.exp(x[n+1:n+5])],
                         forcing_coefficients=x[n+5:])], **common)

    costs = []
    for multiplier in (1., .1, 0.):
        actual = at(vector, multiplier)
        assert not actual['converged']
        assert actual['evaluations'] == 1
        assert actual['convergence_reason'] == 'evaluation_budget'
        expected_prior = .7*dt*np.sum((actual['forcing']/.02)**2)
        assert actual['forcing_prior_cost'] == pytest.approx(expected_prior)
        assert actual['objective'] == pytest.approx(sum(actual['costs'].values()))
        np.testing.assert_allclose(actual['gradient'], 2*actual['residual_jacobian'].T@actual['residual'])
        numeric = []
        for j in range(len(vector)):
            plus, minus = vector.copy(), vector.copy()
            plus[j] += 1e-6
            minus[j] -= 1e-6
            numeric.append((at(plus, multiplier)['objective']-at(minus, multiplier)['objective'])/2e-6)
        np.testing.assert_allclose(actual['gradient'], numeric, atol=2e-7, rtol=3e-6)
        costs.append([actual['driver_curvature_cost'], actual['initial_prior_cost'],
                      actual['flow_prior_cost'], actual['forcing_prior_cost']])
    np.testing.assert_array_equal(costs, np.broadcast_to(costs[0], (3, 4)))


def test_forced_fit_recovers_synthetic_input_without_an_output_component():
    p, n, dt = parameters(), 24, .3
    driver = .02*np.sin(np.arange(n)/4)
    initial = np.array([.005, 1.02, 1.01, 1.015, 1.005])
    coefficients = np.array([.003, -.0015])
    target = decay_forward(driver, initial, p, dt, decay_multiplier=.1,
                           forcing_coefficients=coefficients)['canonical_prediction']
    actual = fit_decay_shared_driver(target, p, dt, sd=[.02, .001, .001],
        decay_multiplier=.1, forcing_modes=2, forcing_weight=0., penalty=0.,
        initial_penalty=0., flow_prior_weight=0., gradient_tolerance=1e-7,
        max_evaluations=180)
    assert actual['converged'], actual['starts']
    assert actual['weighted_data_sse'] < 1e-10
    np.testing.assert_allclose(actual['driver'], driver, atol=1e-8)
    np.testing.assert_allclose(actual['forcing_coefficients'], coefficients, atol=2e-7)
    np.testing.assert_array_equal(actual['prediction'], actual['physical_prediction'])
    assert actual['physical_checks']['all_intermediate_valid']


def test_hidden_targets_cannot_change_initialization_fit_or_selection():
    p, n, dt = parameters(), 16, .3
    target = decay_forward(.02*np.sin(np.arange(n)/3), [0., 1., 1., 1., 1.], p, dt,
        decay_multiplier=.1, forcing_coefficients=[.002, -.001])['canonical_prediction']
    operator = np.kron(np.eye(n)-np.ones((n, n))/n, np.eye(3))
    target = (operator@target.ravel()).reshape(n, 3)
    visible = np.ones((n, 3), bool)
    visible[5:11, 1:] = False
    common = dict(mean_operator=operator, visible=visible, sd=[.02, .006, .003],
        decay_multiplier=.1, forcing_modes=2, max_evaluations=100)
    actual = fit_decay_shared_driver(target, p, dt, **common)
    target[~visible] = np.nan
    changed = fit_decay_shared_driver(target, p, dt, **common)
    assert actual['starts'] == changed['starts']
    np.testing.assert_array_equal(actual['prediction'], changed['prediction'])
    np.testing.assert_array_equal(actual['gradient'], changed['gradient'])


@pytest.mark.parametrize('initial', [[0., 0., 1., 1., 1.], [0., 1., 1., 1., 4.]])
def test_initial_domain_is_rejected_and_never_claims_convergence(initial):
    p = parameters()
    with pytest.raises(FloatingPointError):
        decay_forward(np.zeros(10), initial, p, .25)
    fitted = fit_decay_shared_driver(np.zeros((10, 3)), p, .25,
        starts=[dict(driver=np.zeros(10), initial_state=initial)])
    assert fitted['status'] == 'failed_domain'
    assert not fitted['converged']
    assert fitted['starts'][0]['status'] == 'infeasible_initial'


def test_intermediate_stage_domain_rejection_never_clips_or_substitutes_residuals():
    p = parameters()
    for kwargs in (dict(driver=np.full(20, -100.)),
                   dict(driver=np.zeros(20), decay_multiplier=0., forcing_coefficients=[-10.])):
        with pytest.raises(FloatingPointError, match='RK'):
            decay_forward(initial=[0., 1., 1., 1., 1.], parameters=p, dt=.25, **kwargs)
    target = np.zeros((12, 3))
    target[:, 0] = -10.
    actual = fit_decay_shared_driver(target, p, .3, sd=[.1, 1., 1.], max_evaluations=11)
    assert actual['domain_rejections'] > 0
    assert np.isfinite(actual['objective'])
    assert not actual['converged']
    assert actual['evaluations'] <= 11


@pytest.mark.parametrize('kwargs', [dict(forcing_modes=True), dict(forcing_modes=10),
    dict(forcing_sd=0.), dict(forcing_weight=-1.), dict(decay_multiplier=-.1),
    dict(substeps=0), dict(max_evaluations=0), dict(sd=[1., 0., 1.]),
    dict(mean_operator=np.eye(3)), dict(starts=[])])
def test_fitter_rejects_invalid_contracts(kwargs):
    with pytest.raises(ValueError):
        fit_decay_shared_driver(np.zeros((10, 3)), parameters(), .25, **kwargs)
