"""Discrete derivatives and cost accounting for the optional engineering flow anchor."""
from dataclasses import replace

import numpy as np
import pytest

from src.inference import t3a_balloon_robust_ssm as core
from src.inference.shared_driver_reconstruction import (
    _nonlinear_trial_objective, nonlinear_driver_forward,
    fit_nonlinear_shared_driver, fit_nonlinear_shared_parameter,
)


def case():
    p = core.BalloonParameters()
    r = .04*np.sin(np.arange(12)*.3)
    initial = np.array([.01, 1.03, 1.02, 1.01, .99])
    y = nonlinear_driver_forward(r, initial, p, .25)['canonical_prediction']
    return p, r, initial, y


@pytest.mark.parametrize('name', ['tau', 'neurovascular_gain'])
def test_flow_residual_exact_discrete_jacobians(name):
    p, r, initial, y = case()
    x = np.r_[r, initial[0], np.log(initial[1:])]
    mask = np.ones_like(y, dtype=bool)
    mask[3:6, 1] = False
    operator = np.kron(np.eye(len(r))-np.ones((len(r), len(r)))/len(r), np.eye(3))
    def evaluate(z, parameters=p, derivative=True):
        return _nonlinear_trial_objective(z, parameters, .25, operator, y, mask,
            np.ones_like(y)*.3, np.eye(len(z))*.02, .7, 3, derivative,
            name, .4, .8)
    result = evaluate(x)
    eps = 1e-6
    numerical = np.column_stack([(evaluate(x+eps*row, derivative=False)['residual']-
        evaluate(x-eps*row, derivative=False)['residual'])/(2*eps) for row in np.eye(len(x))])
    np.testing.assert_allclose(result['jacobian'], numerical, rtol=1e-5, atol=2e-8)
    def changed(delta):
        if name == 'tau':
            return replace(p, free=replace(p.free, tau=p.free.tau*np.exp(delta)))
        return replace(p, fixed=replace(p.fixed, neurovascular_gain=p.fixed.neurovascular_gain*np.exp(delta)))
    value = p.free.tau if name == 'tau' else p.fixed.neurovascular_gain
    numerical_parameter = (evaluate(x, changed(eps), False)['residual']-
        evaluate(x, changed(-eps), False)['residual'])/(2*eps)
    np.testing.assert_allclose(value*result['parameter_jacobian'], numerical_parameter, rtol=1e-5, atol=2e-8)
    flow_part = result['parameter_jacobian'][-len(r):]
    if name == 'tau':
        np.testing.assert_array_equal(flow_part, 0.)
    else:
        assert np.linalg.norm(flow_part) > 0
    assert result['flow_prior_cost'] == pytest.approx(np.sum(result['residual'][-len(r):]**2))


@pytest.mark.parametrize('group', [False, True])
def test_zero_weight_exact_parity(group):
    p, r, initial, y = case()
    kwargs = dict(penalty=.01, initial_penalty=.1, max_evaluations=80, return_jacobian=True)
    if group:
        fn = fit_nonlinear_shared_parameter
        y = y[None]
        kwargs.update(parameter_name='tau', parameter_bounds=(.3, 8.), max_iterations=20)
    else:
        fn = fit_nonlinear_shared_driver
    old = fn(y, p, .25, **kwargs)
    zero = fn(y, p, .25, flow_prior_weight=0., flow_prior_log_sd=.4, **kwargs)
    for key in ['objective', 'prediction', 'driver', 'initial_state', 'evaluations', 'converged']:
        np.testing.assert_array_equal(old[key], zero[key])
    assert zero['flow_prior_cost'] == 0.


@pytest.mark.parametrize('group', [False, True])
def test_nonzero_fit_cost_and_information_accounting(group):
    p, r, initial, y = case()
    kwargs = dict(penalty=.003, initial_penalty=.02, driver_amplitude_weight=.01,
        flow_prior_weight=.02, flow_prior_log_sd=.7, max_evaluations=240,
        gradient_tolerance=1e-6, return_jacobian=True)
    if group:
        y = np.stack([y, y*.9])
        fit = fit_nonlinear_shared_parameter(y, p, .25, parameter_name='tau', parameter_bounds=(.3, 8.),
            parameter_prior_mean=2., parameter_prior_log_sd=.8, max_iterations=60, **kwargs)
    else:
        fit = fit_nonlinear_shared_driver(y, p, .25, **kwargs)
    assert fit['converged']
    flow_cost = .02*.25/.7**2*np.sum(np.log(fit['states'][..., 2])**2)
    assert flow_cost > 0
    assert fit['flow_prior_cost'] == pytest.approx(flow_cost)
    initial_cost = .02*np.sum((fit['initial_state']-[0, 1, 1, 1, 1])**2)
    assert fit['objective'] == pytest.approx(fit['weighted_data_sse'] + .003*fit['roughness'] +
        initial_cost + fit['driver_amplitude_cost'] + flow_cost + fit.get('parameter_prior_cost', 0.), abs=1e-14)
    if group:
        assert fit['flow_prior_cost'] == pytest.approx(sum(fit['trial_flow_prior_cost']))
        assert fit['objective'] == pytest.approx(sum(fit['trial_objectives'])+fit['parameter_prior_cost'])
        assert fit['parameter_prior_cost'] == pytest.approx((np.log(fit['tau']/2)/.8)**2)
        for label, n in [('data_only', y.shape[1]*3), ('nuisance_regularized', None)]:
            expected = 0.
            for j, b in zip(fit['residual_jacobians'], fit['residual_log_parameter_jacobians']):
                j, b = j[:n], b[:n]
                u, s, _ = np.linalg.svd(j, full_matrices=False)
                u = u[:, s > s[0]*1e-10]
                expected += np.sum((b-u@(u.T@b))**2)
            assert fit['information'][label]['value'] == pytest.approx(expected, abs=1e-13)


@pytest.mark.parametrize('group', [False, True])
def test_flow_anchor_does_not_clip_invalid_domains(group):
    p, r, initial, y = case()
    initial[1] = 0.
    kwargs = dict(flow_prior_weight=1.)
    if group:
        result = fit_nonlinear_shared_parameter(y[None], p, .25, parameter_name='tau',
            parameter_bounds=(.3, 8.), starts=[dict(parameter_value=2., driver=r[None],
            initial_state=initial[None])], **kwargs)
    else:
        result = fit_nonlinear_shared_driver(y, p, .25, starts=[dict(driver=r, initial_state=initial)], **kwargs)
    assert not result['converged']
    assert result['status'] == 'failed_domain'


@pytest.mark.parametrize('weight,sd', [(-1, 1), (np.nan, 1), (1, 0), (0, np.inf)])
@pytest.mark.parametrize('group', [False, True])
def test_invalid_flow_prior_rejected(weight, sd, group):
    p, _, _, y = case()
    fn = fit_nonlinear_shared_parameter if group else fit_nonlinear_shared_driver
    kwargs = dict(parameter_name='tau', parameter_bounds=(.3, 8.)) if group else {}
    with pytest.raises(ValueError, match='flow prior'):
        fn(y[None] if group else y, p, .25, flow_prior_weight=weight, flow_prior_log_sd=sd, **kwargs)


def test_flow_anchor_is_soft_and_covers_initial_sample():
    p, r, initial, y = case()
    initial[1] = 2.
    x = np.r_[r, initial[0], np.log(initial[1:])]
    result = _nonlinear_trial_objective(x, p, .25, None, y, np.ones_like(y, bool),
        np.ones_like(y), np.zeros((0, len(x))), 0., 4, True,
        flow_prior_weight=.5, flow_prior_log_sd=.7)
    assert result['forward']['states'][0, 2] == pytest.approx(2.)
    assert np.isfinite(result['objective'])
    assert result['residual'][-len(r)] == pytest.approx(np.sqrt(.5*.25)*np.log(2)/.7)
    assert result['jacobian'][-len(r), len(r)+1] == pytest.approx(np.sqrt(.5*.25)/.7)


def test_baseline_processing_cannot_identify_stationary_nonrest_flow_family():
    """Distinct prescribed-driver equilibria share zero processed observations.

    This is a data-likelihood ambiguity, not equality of penalized objectives:
    initial-state and optional flow anchors can prefer rest without observing it.
    The autonomous core r decay is outside the prescribed-driver dynamics.
    """
    from experiments.scripts.evaluate_shared_driver_reconstruction import view_operators
    pe, ph, operator, _ = view_operators(dict(tensor=dict(steps=120), center_steps=16), 'full')
    outputs = []
    for flow in [.7, 1., 1.3]:
        tau_outputs = []
        for tau in [1., 4.]:
            p = core.BalloonParameters()
            p = replace(p, free=replace(p.free, tau=tau))
            driver = p.fixed.gamma*(flow-1)/p.fixed.neurovascular_gain
            volume = flow**p.fixed.alpha
            extraction = -np.expm1(core.extraction_log_complement(flow, p.fixed.E0))
            state = np.array([driver, 0., flow, volume, volume, volume*extraction/p.fixed.E0])
            # r is externally prescribed; only the five vascular derivatives
            # must vanish, not the autonomous core's unused r derivative.
            rhs = core.balloon_rhs(core.physical_to_transformed(state), p)
            np.testing.assert_allclose(rhs[1:], 0., atol=1e-14)
            if flow != 1.:
                assert rhs[0] != 0.
            forward = nonlinear_driver_forward(np.full(120, driver), state[1:], p, .25,
                                               substeps=4, derivative=False)
            np.testing.assert_allclose(forward['states'], np.tile(state, (120, 1)), atol=1e-13, rtol=0.)
            checks = core.run_physical_checks(forward['states'], p)
            for key in ['finite', 'positive_fvpq', 'oxygen_extraction_in_unit_interval',
                        'absolute_hb_nonnegative', 'hbr_not_above_hbt']:
                assert checks[key]
            canonical = core.observation_map(state, p)
            if flow != 1.:
                assert np.linalg.norm(canonical) > .1
            processed = (operator@forward['canonical_prediction'].ravel()).reshape(120, 3)
            native_processed = np.column_stack((pe@np.full(120, canonical[0]),
                                                ph@np.tile(canonical[1:], (300, 1))))
            np.testing.assert_allclose(processed, 0., atol=1e-12, rtol=0.)
            np.testing.assert_allclose(native_processed, processed, atol=1e-12, rtol=0.)
            tau_outputs.append(canonical)
            outputs.append(processed)
        np.testing.assert_array_equal(*tau_outputs)
    np.testing.assert_allclose(outputs, 0., atol=1e-12, rtol=0.)
