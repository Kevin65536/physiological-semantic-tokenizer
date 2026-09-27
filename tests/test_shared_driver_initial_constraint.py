"""Reduced-coordinate checks for the explicit initial total-Hb/volume constraint."""
from dataclasses import replace

import numpy as np
import pytest

from src.inference import t3a_balloon_robust_ssm as core
from src.inference.shared_driver_reconstruction import (
    _volume_tied_initial_embedding, _nonlinear_trial_objective,
    nonlinear_driver_forward, fit_nonlinear_shared_driver, fit_nonlinear_shared_parameter,
)


def synthetic():
    p = core.BalloonParameters()
    p = replace(p, free=replace(p.free, tau=1.6))
    r = .06*np.sin(np.arange(20)*.4)+.02*np.cos(np.arange(20)*.9)
    initial = np.array([.01, 1.03, 1.02, 1.02, .98])
    y = nonlinear_driver_forward(r, initial, p, .3)['canonical_prediction']
    return p, r, initial, y


def test_reduced_residual_jacobian_and_log_tau_chain_match_finite_difference():
    p, r, initial, y = synthetic()
    n = len(r)
    embedding = _volume_tied_initial_embedding(n)
    reduced = np.r_[r, initial[0], np.log(initial[[1, 2, 4]])]
    mask = np.ones_like(y, bool)
    mask[4:9, 1] = False
    def evaluate(z, parameters=p, derivative=True):
        return _nonlinear_trial_objective(z, parameters, .3, None, y, mask,
            np.ones_like(y)*.2, np.eye(n+5)*.01, 3., 4, derivative,
            flow_prior_weight=.4, flow_prior_log_sd=.7, initial_embedding=embedding)
    result = evaluate(reduced)
    eps = 1e-6
    numerical = np.column_stack([(evaluate(reduced+eps*d, derivative=False)['residual']-
        evaluate(reduced-eps*d, derivative=False)['residual'])/(2*eps) for d in np.eye(n+4)])
    assert result['jacobian'].shape[1] == n+4
    np.testing.assert_allclose(result['jacobian'], numerical, rtol=2e-5, atol=2e-8)
    upper = replace(p, free=replace(p.free, tau=p.free.tau*np.exp(eps)))
    lower = replace(p, free=replace(p.free, tau=p.free.tau*np.exp(-eps)))
    numerical_tau = (evaluate(reduced, upper, False)['residual']-
                     evaluate(reduced, lower, False)['residual'])/(2*eps)
    np.testing.assert_allclose(p.free.tau*result['parameter_jacobian'], numerical_tau, rtol=2e-5, atol=2e-8)
    assert result['initial'][2] == result['initial'][3]


def test_constraint_retains_both_volume_and_total_hb_initial_penalties():
    p, r, _, y = synthetic()
    n = len(r)
    x = np.zeros(n+4)
    x[n+2] = np.log(1.2)
    result = _nonlinear_trial_objective(x, p, .3, None, np.zeros_like(y), np.ones_like(y, bool),
        np.ones_like(y), np.zeros((0, n+5)), 7., 4, True,
        initial_embedding=_volume_tied_initial_embedding(n))
    # All other initial components are at rest; these remain two independent
    # physical residual rows even though they share one free coordinate.
    initial_rows = result['residual'][-5:]
    np.testing.assert_allclose(initial_rows, np.sqrt(7)*np.array([0., 0., .2, .2, 0.]), atol=1e-15)
    assert initial_rows@initial_rows == pytest.approx(2*7*.2**2)
    np.testing.assert_allclose(result['jacobian'][-5:, n+2], np.sqrt(7)*np.array([0., 0., 1.2, 1.2, 0.]), atol=1e-15)


@pytest.mark.parametrize('group', [False, True])
def test_matched_nonrest_initial_constraint_recovers_states_driver_and_tau(group):
    p, r, initial, y = synthetic()
    n = len(r)
    kwargs = dict(sd=np.full(3, .01), penalty=0., initial_penalty=0.,
        gradient_tolerance=1e-8, tie_total_hb_to_volume=True, return_jacobian=True)
    if group:
        second_initial = initial.copy()
        second_initial[[2, 3]] = 1.01
        truth_initial = np.stack([initial, second_initial])
        truth_driver = np.stack([r, .7*r])
        second_y = nonlinear_driver_forward(.7*r, second_initial, p, .3)['canonical_prediction']
        fit = fit_nonlinear_shared_parameter(np.stack([y, second_y]), p, .3, parameter_name='tau',
            parameter_bounds=(.5, 8.), max_evaluations=400, max_iterations=60,
            starts=[dict(parameter_value=2., driver=np.zeros((2, n)),
                initial_state=np.tile([0., 1., 1., 1., 1.], (2, 1)))], **kwargs)
        assert fit['tau'] == pytest.approx(p.free.tau, abs=1e-8)
        assert fit['total_free_parameters'] == 2*(n+4)+1
        jacobians = fit['residual_jacobians']
        parameter_jacobians = fit['residual_log_parameter_jacobians']
        np.testing.assert_allclose(fit['initial_state'], truth_initial, atol=1e-8, rtol=0.)
        np.testing.assert_allclose(fit['driver'], truth_driver, atol=1e-8, rtol=0.)
        for label, rows in [('data_only', n*3), ('nuisance_regularized', None)]:
            values = []
            for jac, param in zip(jacobians, parameter_jacobians):
                assert jac.shape[1] == n+4
                jac, param = jac[:rows], param[:rows]
                u, s, _ = np.linalg.svd(jac, full_matrices=False)
                u = u[:, s>s[0]*1e-10]
                values.append(np.sum((param-u@(u.T@param))**2))
            assert fit['information'][label]['value'] == pytest.approx(sum(values), rel=1e-10)
            assert fit['information'][label]['nuisance_ranks'] == [n+4, n+4]
    else:
        fit = fit_nonlinear_shared_driver(y, p, .3, max_evaluations=120, **kwargs)
        assert fit['total_free_parameters'] == n+4
        assert fit['residual_jacobian'].shape[1] == n+4
        assert fit['canonical_jacobian'].shape == (n*3, n+4)
        np.testing.assert_allclose(fit['initial_state'], initial, atol=1e-8, rtol=0.)
        np.testing.assert_allclose(fit['driver'], r, atol=1e-8, rtol=0.)
        residual = ((fit['prediction']-y)/.01).ravel()
        j = fit['residual_jacobian'][:n*3]
        expected = np.max(abs(j.T@residual)/np.sqrt(np.maximum(np.sum(j*j, axis=0), 1.)))
        assert fit['scaled_gradient_inf_norm'] == pytest.approx(expected, abs=1e-15)
    assert fit['converged'] and fit['objective'] < 1e-14
    assert fit['initial_state_constraint'] == 'p0_equals_v0'
    assert fit['initial_state_free_parameters'] == 4
    assert fit['nuisance_free_parameters_per_trial'] == n+4
    np.testing.assert_array_equal(fit['initial_state'][..., 2], fit['initial_state'][..., 3])
    # The unchanged equations preserve p=v up to integration roundoff.
    np.testing.assert_allclose(fit['states'][..., 3], fit['states'][..., 4], atol=2e-14, rtol=0.)


@pytest.mark.parametrize('group', [False, True])
def test_false_constraint_exactly_preserves_unconstrained_results(group):
    p, r, initial, y = synthetic()
    kwargs = dict(penalty=.01, initial_penalty=.2, flow_prior_weight=.1,
                  max_evaluations=100, return_jacobian=True)
    if group:
        fn = fit_nonlinear_shared_parameter
        y = y[None]
        kwargs.update(parameter_name='tau', parameter_bounds=(.5, 8.), max_iterations=30)
    else:
        fn = fit_nonlinear_shared_driver
    first = fn(y, p, .3, **kwargs)
    second = fn(y, p, .3, tie_total_hb_to_volume=False, **kwargs)
    for key in ['objective', 'prediction', 'states', 'initial_state', 'driver', 'evaluations', 'converged']:
        np.testing.assert_array_equal(first[key], second[key])
    assert second['initial_state_free_parameters'] == 5
    assert second['initial_state_constraint'] == 'none'


@pytest.mark.parametrize('group', [False, True])
def test_unequal_volume_and_total_hb_start_is_rejected_without_projection(group):
    p, r, initial, y = synthetic()
    initial[3] = np.nextafter(initial[2], np.inf)
    with pytest.raises(ValueError, match='p0 == v0; no projection'):
        if group:
            fit_nonlinear_shared_parameter(y[None], p, .3, parameter_name='tau', parameter_bounds=(.5, 8.),
                starts=[dict(parameter_value=2., driver=r[None], initial_state=initial[None])], tie_total_hb_to_volume=True)
        else:
            fit_nonlinear_shared_driver(y, p, .3, starts=[dict(driver=r, initial_state=initial)], tie_total_hb_to_volume=True)


@pytest.mark.parametrize('group', [False, True])
def test_constraint_requires_boolean_option(group):
    p, _, _, y = synthetic()
    kwargs = dict(parameter_name='tau', parameter_bounds=(.5, 8.)) if group else {}
    fn = fit_nonlinear_shared_parameter if group else fit_nonlinear_shared_driver
    with pytest.raises(ValueError, match='must be boolean'):
        fn(y[None] if group else y, p, .3, tie_total_hb_to_volume='false', **kwargs)


@pytest.mark.parametrize('group', [False, True])
@pytest.mark.parametrize('failure', ['invalid_physical_initial', 'evaluation_budget'])
def test_failed_fits_keep_constraint_metadata_and_reduced_diagnostics(group, failure):
    p = core.BalloonParameters()
    n = 4
    initial = np.array([0., -1. if failure == 'invalid_physical_initial' else 1., 1., 1., 1.])
    y = np.ones((n, 3))*.01
    kwargs = dict(tie_total_hb_to_volume=True, max_evaluations=1, return_jacobian=True)
    if group:
        fit = fit_nonlinear_shared_parameter(y[None], p, .25, parameter_name='tau', parameter_bounds=(.5, 8.),
            starts=[dict(parameter_value=2., driver=np.zeros((1, n)), initial_state=initial[None])], **kwargs)
    else:
        fit = fit_nonlinear_shared_driver(y, p, .25,
            starts=[dict(driver=np.zeros(n), initial_state=initial)], **kwargs)
    assert not fit['converged']
    assert fit['initial_state_constraint'] == 'p0_equals_v0'
    assert fit['initial_state_free_parameters'] == 4
    assert fit['nuisance_free_parameters_per_trial'] == n+4
    assert fit['total_free_parameters'] == n+4+int(group)
    if failure == 'invalid_physical_initial':
        assert fit['status'] == 'failed_domain'
    else:
        assert fit['status'] == 'failed_numerical'
        assert fit['convergence_reason'] == 'evaluation_budget'
        jac = fit['residual_jacobians'][0] if group else fit['residual_jacobian']
        assert jac.shape[1] == n+4
        assert np.isfinite(jac).all()
        np.testing.assert_array_equal(fit['initial_state'][..., 2], fit['initial_state'][..., 3])


@pytest.mark.parametrize('penalty', [0., .07])
def test_linear_tied_svd_matches_independent_column_merge(penalty):
    from src.inference.shared_driver_reconstruction import build_shared_driver_design, fit_shared_driver
    p, _, _, _ = synthetic()
    n = 12
    design = build_shared_driver_design(p, n, .25)
    rng = np.random.default_rng(17)
    y = rng.normal(0, .01, (n, 3))
    visible = np.ones_like(y, bool)
    visible[3:6, 1:] = False
    sd = np.array([.1, .2, .3])
    result = fit_shared_driver(y, design, visible=visible, observation_scale=sd,
        penalty=penalty, initial_penalty=7., tie_total_hb_to_volume=True)
    scale = np.broadcast_to(sd, y.shape)
    a = design.observation_design[visible.ravel()]/scale[visible, None]
    b = (y[visible]-design.offset[visible])/scale[visible]
    curvature = np.zeros((n-2, n+5))
    curvature[:, :n] = np.diff(np.eye(n), n=2, axis=0)/.25**1.5
    initial = np.zeros((5, n+5))
    initial[:, n:] = np.sqrt(7.)*np.eye(5)
    system = np.vstack([a, np.sqrt(penalty)*curvature, initial]) if penalty else a
    rhs = np.r_[b, np.zeros(n+3)] if penalty else b
    # Independently merge p/v columns; no implementation embedding helper.
    def merge(matrix):
        matrix = matrix.copy()
        matrix[:, n+2] += matrix[:, n+3]
        return np.delete(matrix, n+3, axis=1)
    reduced = merge(system)
    u, s, vt = np.linalg.svd(reduced, full_matrices=False)
    keep = s>s[0]*1e-10
    inv = vt[keep].T/s[keep]
    z = inv@(u[:, keep].T@rhs)
    expected = np.insert(z, n+3, z[n+2])
    np.testing.assert_allclose(result['coefficients'], expected, atol=2e-12, rtol=2e-10)
    assert result['rank'] == int(keep.sum())
    assert result['effective_df'] == pytest.approx(np.sum((merge(a)@inv)**2))
    assert result['total_free_parameters'] == n+4
    assert result['coefficients'].shape == (n+5,)
    assert result['states'][0, 3] == result['states'][0, 4]
    np.testing.assert_allclose(result['states'][:, 3], result['states'][:, 4], atol=2e-14)
    if not penalty:
        assert result['initial_penalty'] == 0.
        assert result['residual_lower_bound_scope'] == 'tied_initial_state'
        assert result['effective_df'] == pytest.approx(result['rank'])
        unregularized = fit_shared_driver(y, design, visible=visible, observation_scale=sd,
            penalty=0., initial_penalty=0., tie_total_hb_to_volume=True)
        np.testing.assert_array_equal(result['coefficients'], unregularized['coefficients'])


def test_linear_default_false_exact_and_boolean_validation():
    from src.inference.shared_driver_reconstruction import build_shared_driver_design, fit_shared_driver
    p, r, initial, y = synthetic()
    design = build_shared_driver_design(p, len(r), .3)
    first = fit_shared_driver(y, design)
    second = fit_shared_driver(y, design, tie_total_hb_to_volume=False)
    for key in ('coefficients', 'states', 'prediction', 'singular_values', 'rank', 'effective_df'):
        np.testing.assert_array_equal(first[key], second[key])
    with pytest.raises(ValueError, match='must be boolean'):
        fit_shared_driver(y, design, tie_total_hb_to_volume=1)
