"""Tak/SPM equations, reduced state, discrete tangents and fitting contract."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.inference.shared_driver_reconstruction import (
    nonlinear_driver_forward, replay_nonlinear_driver,
    fit_nonlinear_shared_parameters, _volume_tied_initial_embedding,
)
from src.inference.t3a_balloon_robust_ssm import BalloonParameters, BalloonFreeParameters


def fixture(n=28):
    p = BalloonParameters(free=BalloonFreeParameters(kappa=.64, tau=2.))
    r = .03*np.sin(np.arange(n)*.23)
    initial = np.array([.02, 1.03, .98, .98, 1.01])
    return p, r, initial


def forward(p, r, initial, tv=2., **kwargs):
    return nonlinear_driver_forward(r, initial, p, .25, numerical_backend='numba',
        hemodynamics='tak', viscoelastic_s=tv, return_flow_jacobian=True, **kwargs)


def test_tak_matches_independent_spm_log_ode_and_preserves_ratio():
    p, r, initial = fixture()
    a = forward(p, r, initial)
    np.testing.assert_array_equal(a['states'][:, 3], a['states'][:, 4])
    # SPM state order: s, logf, logv, logq, logp. Also test the original
    # five-state ODE's non-unit invariant, which the reduced fitter excludes.
    z = np.r_[initial[0], np.log(initial[[1, 2, 4, 3]])]
    z[-1] += np.log(1.2)
    trajectory = [z.copy()]
    def rhs(_, x, drive):
        s = x[0]; f, v, q, total = np.exp(x[1:])
        sd = drive-.64*s-.32*(f-1)
        lv = (f-v**(1/.32))/((2.+2.)*v)
        out = v**(1/.32)+2.*v*lv
        ff = (1-(1-.32)**(1/f))/.32
        return [sd, s/f, lv, (f*ff-out*q/v)/(2.*q), (f-out)/(2.*v)]
    for drive in r[:-1]:
        z = solve_ivp(rhs, [0, .25], z, args=(drive,), rtol=1e-11, atol=1e-13).y[:, -1]
        trajectory.append(z.copy())
    physical = np.exp(np.array(trajectory)[:, 1:])
    np.testing.assert_allclose(physical[:, [0, 1, 2]], a['states'][:, [2, 3, 5]], atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(physical[:, 3]/physical[:, 1], 1.2, atol=1e-12)
    replay = replay_nonlinear_driver(r, initial, p, .25, hemodynamics='tak', viscoelastic_s=2.)
    assert replay['status'] == 'completed'
    np.testing.assert_allclose(replay['canonical_prediction'], a['canonical_prediction'], atol=1e-13)


def test_zero_viscoelastic_equivalence_including_initial_and_driver_tangents():
    p, r, initial = fixture()
    a = forward(p, r, initial, tv=0.)
    b = nonlinear_driver_forward(r, initial, p, .25, numerical_backend='numba',
        hemodynamics='inlet_balance', viscoelastic_s=0.)
    np.testing.assert_allclose(a['states'], b['states'], atol=2e-14)
    embedding = _volume_tied_initial_embedding(len(r))
    np.testing.assert_allclose(a['jacobian']@embedding, b['jacobian']@embedding, atol=1e-12)
    for name in ('tau', 'kappa', 'neurovascular_gain'):
        np.testing.assert_allclose(a[name+'_jacobian'], b[name+'_jacobian'], atol=1e-12)


@pytest.mark.parametrize('name', ['tau', 'kappa', 'gamma', 'viscoelastic_s', 'neurovascular_gain'])
def test_tak_parameter_tangent_matches_independent_finite_difference(name):
    p, r, initial = fixture()
    a = forward(p, r, initial)
    eps = 2e-5
    def at(delta):
        if name == 'viscoelastic_s': return forward(p, r, initial, tv=2+delta)
        part = 'free' if name in ('tau', 'kappa') else 'fixed'
        obj = getattr(p, part)
        return forward(replace(p, **{part: replace(obj, **{name: getattr(obj, name)+delta})}), r, initial)
    hi, lo = at(eps), at(-eps)
    np.testing.assert_allclose(a[name+'_jacobian'],
        (hi['canonical_prediction']-lo['canonical_prediction'])/(2*eps), atol=2e-9, rtol=1e-6)
    np.testing.assert_allclose(a['flow_'+name+'_jacobian'],
        (hi['states'][:, 2]-lo['states'][:, 2])/(2*eps), atol=2e-9, rtol=1e-6)


def test_tak_initial_driver_tangent_and_integration_refinement():
    p, r, initial = fixture()
    a = forward(p, r, initial)
    embedding = _volume_tied_initial_embedding(len(r))
    j = a['jacobian']@embedding
    x = np.r_[r, initial[0], np.log(initial[[1, 2, 4]])]
    direction = np.random.default_rng(9).normal(size=x.size)*.05
    def at(v):
        s, f, volume, q = v[len(r):]
        return forward(p, v[:len(r)], np.array([s, np.exp(f), np.exp(volume), np.exp(volume), np.exp(q)]))
    eps = 1e-5
    diff = (at(x+eps*direction)['canonical_prediction']-at(x-eps*direction)['canonical_prediction']).ravel()/(2*eps)
    np.testing.assert_allclose(j@direction, diff, atol=2e-9, rtol=1e-6)
    fine = forward(p, r, initial, substeps=8)
    assert np.max(abs(fine['canonical_prediction']-a['canonical_prediction'])) < 1e-8


def test_reduced_fit_hidden_isolation_rest_and_single_group_prior():
    p, r, _ = fixture(n=20)
    y = forward(p, r, np.r_[0., np.ones(4)])['canonical_prediction'][None]
    y = np.repeat(y, 2, axis=0)
    mask = np.ones_like(y, bool); mask[:, 9:12, 1:] = False
    opts = dict(parameter_names=['kappa', 'tau'], parameter_bounds=[[.1, 2.], [.3, 10.]],
        sd=[.02, .03, .02], visible=mask, initial_state_mode='volume_tied', hemodynamics='tak',
        viscoelastic_s=2., initial_coordinate_weights=[1, 1, 1, 0, 1],
        parameter_prior_log_mean=np.log([.64, 2.]), parameter_prior_precision=[16., 16.],
        max_iterations=250)
    a = fit_nonlinear_shared_parameters(y, p, .25, **opts)
    assert a['converged']
    assert a['initial_free_parameters'] == 4
    np.testing.assert_array_equal(a['states'][:, :, 3], a['states'][:, :, 4])
    np.testing.assert_allclose(a['objective_components']['parameter_prior'],
        np.sum(16*np.log(a['parameter_values']/[.64, 2.])**2))
    np.testing.assert_allclose(a['objective'], sum(a['objective_components'].values()))
    y[~mask] = 1e20
    b = fit_nonlinear_shared_parameters(y, p, .25, **opts)
    np.testing.assert_array_equal(a['parameter_values'], b['parameter_values'])
    rest = fit_nonlinear_shared_parameters(y, p, .25, **dict(opts, initial_state_mode='rest'))
    assert rest['converged']
    np.testing.assert_array_equal(rest['initial_state'], np.tile(np.r_[0., np.ones(4)], (2, 1)))
    with pytest.raises(ValueError, match='eliminates'):
        fit_nonlinear_shared_parameters(y, p, .25, **dict(opts, initial_state_mode='free'))


def test_invalid_tak_initial_and_unavailable_legacy_viscoelastic_rejected():
    p, r, initial = fixture()
    initial[3] += .01
    with pytest.raises(ValueError, match='p0 == v0'):
        forward(p, r, initial)
    with pytest.raises(ValueError, match='no viscoelastic'):
        nonlinear_driver_forward(r, initial, p, .25, hemodynamics='inlet_balance', viscoelastic_s=2.)


def test_default_baseline_is_tak_with_four_initial_states_and_one_volume_penalty():
    p, r, initial = fixture(n=20)
    four = initial[[0, 1, 2, 4]]
    default = nonlinear_driver_forward(r, four, p, .25)
    explicit = forward(p, r, initial)
    np.testing.assert_array_equal(default['states'], explicit['states'])
    replay = replay_nonlinear_driver(r, four, p, .25)
    np.testing.assert_allclose(replay['canonical_prediction'],default['canonical_prediction'],atol=1e-13)
    fit = fit_nonlinear_shared_parameters(default['canonical_prediction'][None],p,.25,
        parameter_names=[],parameter_bounds=[],sd=[.02,.03,.02],
        starts=[dict(parameter_values=[],driver=r[None],initial_state=four[None])],max_iterations=250)
    assert fit['converged'] and fit['hemodynamics']=='tak' and fit['viscoelastic_s']==2.
    assert fit['initial_state_mode']=='volume_tied' and fit['initial_free_parameters']==4
    np.testing.assert_array_equal(fit['initial_coordinate_weights'],[1,1,1,0,1])
    np.testing.assert_array_equal(fit['states'][:,:,3],fit['states'][:,:,4])
