"""Synthetic-only tests of fixed coordinates, exact derivatives and masking."""
import numpy as np
import pytest

from src.inference.shared_driver_attribution import equal_capacity_hb_basis
from src.inference.shared_driver_modes import (
    fit_shared_driver_modes, forward_shared_driver_modes,
    spectral_loadings, temporal_mode_basis,
)
from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
from src.inference.t3a_balloon_robust_ssm import BalloonParameters


def _fixture(steps=32, modes=6, independent=False):
    basis = temporal_mode_basis(steps, modes)
    loading = spectral_loadings(channels=2)
    rng = np.random.default_rng(321)
    coefficients = rng.normal(size=(modes, 3 if independent else 2))*.008
    coefficients[0] = .003
    return basis, loading, coefficients


def test_fixed_spectral_and_temporal_coordinates_have_declared_gauge():
    loading = spectral_loadings(spatial_weights=[1., .8, .6, .4, .2, .1])
    assert loading.shape == (30, 2)
    np.testing.assert_allclose(loading.T@loading, np.eye(2), atol=1e-15)
    assert np.all(loading[:, 0] > 0)
    contrast = loading[:, 1].reshape(6, 5)
    np.testing.assert_allclose(contrast.sum(axis=1), 0., atol=1e-16)
    assert np.all(contrast[:, 2:4] > 0)
    assert np.all(contrast[:, [0, 1, 4]] < 0)
    np.testing.assert_array_equal(loading, spectral_loadings(spatial_weights=[1., .8, .6, .4, .2, .1]))
    basis = temporal_mode_basis(120)
    assert basis.shape == (120, 24)
    np.testing.assert_array_equal(basis[:, 0], 1.)
    np.testing.assert_allclose(basis.T@basis/120, np.eye(24), atol=3e-15)
    with pytest.raises(ValueError, match="full-column-rank"):
        forward_shared_driver_modes(np.zeros((24, 2)), np.zeros(5), BalloonParameters(), .25,
            loadings=np.ones((3, 2)), temporal_basis=basis, numerical_backend="python")


@pytest.mark.parametrize("independent", [False, True])
def test_forward_analytic_jacobian_includes_operators_initials_and_common(independent):
    basis, loading, coefficients = _fixture(steps=18, modes=4, independent=independent)
    steps, channels = len(basis), len(loading)
    eeg_op = np.eye(steps)-np.ones((steps, 1))@np.r_[np.full(4, .25), np.zeros(steps-4)][None]
    hb_op = np.eye(steps)-np.ones((steps, steps))/steps
    operator = np.zeros((3*steps, 3*steps))
    operator[0::3, 0::3] = eeg_op
    operator[1::3, 1::3] = operator[2::3, 2::3] = hb_op
    common = equal_capacity_hb_basis(operator, modes=3).reshape(steps, 3, 3)[:, 1:]
    initial = np.array([.006, .004, -.003, .002, -.001])
    cc = np.array([.003, -.002, .001])
    kwargs = dict(loadings=loading, temporal_basis=basis, eeg_operator=eeg_op,
        hb_operator=hb_op, hb_basis=common, gamma=.7, independent=independent,
        numerical_backend="python")
    p = BalloonParameters()
    analytic = forward_shared_driver_modes(coefficients, initial, p, .25,
        component_coefficients=cc, **kwargs)
    x = np.r_[coefficients.ravel(), initial, cc]
    numerical = np.empty_like(analytic["jacobian"])
    for j in range(len(x)):
        predictions = []
        for sign in (-1., 1.):
            shifted = x.copy()
            shifted[j] += sign*1e-6
            predictions.append(forward_shared_driver_modes(
                shifted[:coefficients.size].reshape(coefficients.shape),
                shifted[coefficients.size:coefficients.size+5], p, .25,
                component_coefficients=shifted[-3:], derivative=False, **kwargs)["prediction"].ravel())
        numerical[:, j] = (predictions[1]-predictions[0])/2e-6
    assert analytic["jacobian"].shape == (steps*(channels+2), len(x))
    np.testing.assert_allclose(analytic["jacobian"], numerical, rtol=2e-6, atol=2e-9)
    np.testing.assert_allclose(analytic["prediction"]-analytic["physical_prediction"],
                               analytic["observation_component"], atol=1e-18)
    if independent:
        np.testing.assert_array_equal(analytic["u_h"], analytic["drivers"][:, 2])
    else:
        np.testing.assert_allclose(analytic["u_h"],
            (analytic["r"][:, 0]+.7*analytic["r"][:, 1])/np.sqrt(1+.7**2), atol=1e-17)


def test_scalar_Hb_route_is_exactly_the_existing_balloon_core():
    basis, loading, coefficients = _fixture()
    coefficients = coefficients[:, :1]
    forward = forward_shared_driver_modes(coefficients, np.zeros(5), BalloonParameters(), .25,
        loadings=loading[:, :1], temporal_basis=basis, numerical_backend="python")
    original = nonlinear_driver_forward(basis@coefficients[:, 0], np.r_[0., np.ones(4)],
        BalloonParameters(), .25, numerical_backend="python")
    np.testing.assert_array_equal(forward["hb_prediction"], original["canonical_prediction"][:, 1:])
    np.testing.assert_array_equal(forward["states"], original["states"])
    np.testing.assert_allclose(forward["jacobian"].reshape(32, 12, -1)[:, -2:, :6],
        original["jacobian"].reshape(32, 3, 37)[:, 1:, :32]@basis, atol=1e-16)


def test_compiled_backend_preserves_processed_mean_and_analytic_jacobian():
    basis, loading, coefficients = _fixture(steps=24, modes=4)
    kwargs = dict(loadings=loading, temporal_basis=basis, gamma=-.5,
        eeg_operator=np.eye(24)-np.ones((24, 24))/24,
        hb_operator=np.eye(24)-np.ones((24, 24))/24)
    reference = forward_shared_driver_modes(coefficients, np.zeros(5), BalloonParameters(),
        .25, numerical_backend="python", **kwargs)
    compiled = forward_shared_driver_modes(coefficients, np.zeros(5), BalloonParameters(),
        .25, numerical_backend="numba", **kwargs)
    for key in ("prediction", "states", "jacobian"):
        np.testing.assert_allclose(compiled[key], reference[key], rtol=2e-12, atol=2e-14)


def test_hidden_poison_cannot_change_initialization_fit_or_residuals():
    basis, loading, coefficients = _fixture(steps=24, modes=4)
    truth = forward_shared_driver_modes(coefficients, np.zeros(5), BalloonParameters(), .25,
        loadings=loading, temporal_basis=basis, gamma=.5, numerical_backend="python")
    mask = np.ones_like(truth["prediction"], bool)
    mask[7:15, -2:] = False
    mask[:, 2] = False
    kwargs = dict(loadings=loading, temporal_basis=basis, gamma=.5,
        sd=np.r_[np.full(10, .03), .015, .01], visible=mask, ar_weight=.0001,
        numerical_backend="python", max_nfev=100)
    clean = truth["prediction"]
    a = fit_shared_driver_modes(clean[:, :10], clean[:, 10:], BalloonParameters(), .25, **kwargs)
    assert a["success"], a["convergence_reason"]
    for poison in (1e30, np.nan, np.inf):
        changed = clean.copy()
        changed[~mask] = poison
        b = fit_shared_driver_modes(changed[:, :10], changed[:, 10:], BalloonParameters(), .25, **kwargs)
        np.testing.assert_array_equal(a["prediction"], b["prediction"])
        np.testing.assert_array_equal(a["coefficients"], b["coefficients"])
        assert a["objective"] == b["objective"] and a["nfev"] == b["nfev"]
        assert np.isnan(b["residual"][~mask]).all()


def test_independent_three_trajectory_recovery_preserves_mode_directions():
    basis, loading, coefficients = _fixture(steps=48, modes=7, independent=True)
    p = BalloonParameters()
    before = loading.copy()
    truth = forward_shared_driver_modes(coefficients, np.zeros(5), p, .25,
        loadings=loading, temporal_basis=basis, independent=True, numerical_backend="python")
    fitted = fit_shared_driver_modes(truth["eeg_prediction"], truth["hb_prediction"], p, .25,
        loadings=loading, temporal_basis=basis, independent=True, ar_weight=0.,
        sd=np.r_[np.full(10, .01), .001, .001], numerical_backend="python", max_nfev=100)
    assert fitted["success"], fitted["convergence_reason"]
    assert fitted["physical_check"]["valid"]
    assert fitted["complexity"]["driving_trajectories"] == 3
    np.testing.assert_allclose(fitted["r"], truth["r"], atol=2e-8)
    np.testing.assert_allclose(fitted["u_h"], truth["u_h"], atol=2e-6)
    np.testing.assert_array_equal(loading, before)
    rotated = coefficients.copy()
    rotated[:, :2] = coefficients[:, :2]@np.array([[0., 1.], [-1., 0.]])
    alternative = forward_shared_driver_modes(rotated, np.zeros(5), p, .25,
        loadings=loading, temporal_basis=basis, independent=True, numerical_backend="python")
    assert np.linalg.norm(alternative["eeg_prediction"]-truth["eeg_prediction"]) > .05


def test_state_AR_penalty_is_invariant_to_temporal_basis_units():
    basis, loading, coefficients = _fixture(steps=24, modes=4)
    truth = forward_shared_driver_modes(coefficients, np.zeros(5), BalloonParameters(), .25,
        loadings=loading, temporal_basis=basis, numerical_backend="python")
    kwargs = dict(loadings=loading, sd=np.r_[np.full(10, .03), .015, .01],
        ar_weight=.04, state_sd=.07, numerical_backend="python", max_nfev=100)
    fitted = fit_shared_driver_modes(truth["eeg_prediction"], truth["hb_prediction"],
        BalloonParameters(), .25, temporal_basis=basis, **kwargs)
    rescaled = fit_shared_driver_modes(truth["eeg_prediction"], truth["hb_prediction"],
        BalloonParameters(), .25, temporal_basis=basis*np.array([2., .5, 3., 1.]), **kwargs)
    assert fitted["success"] and rescaled["success"]
    np.testing.assert_allclose(fitted["prediction"], rescaled["prediction"], atol=1e-8)
    assert fitted["objective"] == pytest.approx(rescaled["objective"], abs=1e-11)


def test_budget_and_invalid_physical_forward_do_not_fake_success():
    basis, loading, coefficients = _fixture(steps=24, modes=4)
    truth = forward_shared_driver_modes(coefficients, np.zeros(5), BalloonParameters(), .25,
        loadings=loading, temporal_basis=basis, numerical_backend="python")
    fitted = fit_shared_driver_modes(truth["eeg_prediction"], truth["hb_prediction"],
        BalloonParameters(), .25, loadings=loading, temporal_basis=basis,
        sd=np.ones(12), max_nfev=1, numerical_backend="python")
    assert fitted["nfev"] == 1
    assert not fitted["success"] and fitted["status"] == "failed_numerical"
    with pytest.raises(FloatingPointError, match="Hb balance"):
        forward_shared_driver_modes(coefficients, np.array([0., 0., 0., -3., 0.]),
            BalloonParameters(), .25, loadings=loading, temporal_basis=basis,
            numerical_backend="python")
