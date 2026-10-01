"""Independent numerical parity, sensitivities, domain and optimizer checks."""
from dataclasses import replace

import numpy as np
import pytest

from src.inference.shared_driver_reconstruction import (
    nonlinear_driver_forward, fit_nonlinear_shared_driver,
)
from src.inference.t3a_balloon_robust_ssm import BalloonParameters


@pytest.mark.parametrize('substeps', [4, 8])
@pytest.mark.parametrize('derivative', [False, True])
def test_compiled_matches_reference_discrete_map(substeps, derivative):
    p = BalloonParameters()
    p = replace(p, fixed=replace(p.fixed, P0=71., Q0=23., eeg_loading=1.7,
                                neurovascular_gain=.6), free=replace(p.free, tau=3.1))
    driver = .04*np.sin(np.arange(24)/4)+.013
    initial = np.array([.015, 1.04, .97, 1.02, .99])
    a = nonlinear_driver_forward(driver, initial, p, .25, substeps=substeps,
                                 derivative=derivative, return_flow_jacobian=True)
    b = nonlinear_driver_forward(driver, initial, p, .25, substeps=substeps,
                                 derivative=derivative, return_flow_jacobian=True,
                                 numerical_backend='numba')
    for key, value in a.items():
        if value is not None:
            np.testing.assert_allclose(b[key], value, rtol=2e-12, atol=5e-13)


def test_compiled_sensitivity_independent_finite_difference():
    p = BalloonParameters()
    driver = .02*np.sin(np.arange(20)/3)
    initial = np.array([.01, 1.02, 1.01, 1.02, 1.])
    call = lambda r, i, pp: nonlinear_driver_forward(r, i, pp, .25, numerical_backend='numba')
    base = call(driver, initial, p)
    eps = 1e-5
    for name in ('tau', 'kappa', 'neurovascular_gain'):
        branch = 'fixed' if name == 'neurovascular_gain' else 'free'
        part = getattr(p, branch)
        plus = replace(p, **{branch: replace(part, **{name: getattr(part, name)+eps})})
        minus = replace(p, **{branch: replace(part, **{name: getattr(part, name)-eps})})
        fd = (call(driver, initial, plus)['canonical_prediction']-
              call(driver, initial, minus)['canonical_prediction'])/(2*eps)
        np.testing.assert_allclose(fd, base[name+'_jacobian'], atol=3e-10, rtol=2e-5)
    for col in (0, 8, 20, 21, 23, 24):
        args = []
        for sign in (1, -1):
            r, i = driver.copy(), initial.copy()
            if col < 20:
                r[col] += sign*eps
            elif col == 20:
                i[0] += sign*eps
            else:
                i[col-20] *= np.exp(sign*eps)
            args.append(call(r, i, p)['canonical_prediction'])
        np.testing.assert_allclose(((args[0]-args[1])/(2*eps)).ravel(),
                                   base['jacobian'][:, col], atol=3e-10, rtol=2e-5)


@pytest.mark.parametrize('backend', ['python', 'numba'])
def test_compiled_does_not_clip_invalid_rk_stages(backend):
    with pytest.raises(FloatingPointError):
        nonlinear_driver_forward(np.full(20, -100.), np.array([0., 1., 1., 1., 1.]),
                                 BalloonParameters(), .25, numerical_backend=backend)


def test_optimizer_and_hidden_value_isolation():
    p = BalloonParameters()
    y = nonlinear_driver_forward(.02*np.sin(np.arange(20)/3), np.array([0., 1., 1., 1., 1.]),
                                 p, .25)['canonical_prediction']
    mask = np.ones_like(y, dtype=bool); mask[8:12, 1:] = False
    args = dict(sd=np.array([.02, .03, .01]), penalty=.01, initial_penalty=100.,
                max_evaluations=150, visible=mask, flow_prior_weight=1., flow_prior_log_sd=np.log(2))
    a = fit_nonlinear_shared_driver(y, p, .25, **args)
    corrupted = y.copy(); corrupted[~mask] = 1e10
    b = fit_nonlinear_shared_driver(corrupted, p, .25, numerical_backend='numba', **args)
    assert a['converged'] and b['converged']
    np.testing.assert_allclose(a['prediction'], b['prediction'], atol=1e-10)
    assert b['replay_max_abs_difference'] < 1e-10
