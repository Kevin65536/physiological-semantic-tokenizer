"""Small synthetic checks of mode ownership and physical response comparisons."""
import inspect

import numpy as np
import pytest

from src.inference.semantic_response_ssm import (
    cascade_response_features, fit_fixed_driver_response, fixed_driver_response,
    gaussian_predictive_scores, infer_eeg_modes, response_rhs,
)
from src.inference.shared_driver_attribution import equal_capacity_hb_basis
from src.inference.shared_driver_modes import spectral_loadings, temporal_mode_basis
from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
from src.inference.t3a_balloon_robust_ssm import (
    BalloonParameters, balloon_rhs, physical_to_transformed,
)


def _driver(steps=48):
    t = np.arange(steps)*.25
    return .018*np.sin(t/2)+.009*np.exp(-((t-4)/2)**2)


def _hb_basis(steps):
    return equal_capacity_hb_basis(np.eye(3*steps), modes=4).reshape(steps, 3, 4)[:, 1:]


def test_eeg_mode_inference_has_no_hb_input_and_recovers_fixed_coordinates():
    assert not any("hb" in key.lower() for key in inspect.signature(infer_eeg_modes).parameters)
    basis, loading = temporal_mode_basis(48, 8), spectral_loadings(channels=2)
    coefficients = np.random.default_rng(123).normal(size=(8, 2))*.006
    eeg = (basis@coefficients)@loading.T
    result = infer_eeg_modes(eeg, loading, np.full(10, .01), temporal_basis=basis, ar_weight=0)
    np.testing.assert_allclose(result["r"], basis@coefficients, atol=1e-16)
    np.testing.assert_allclose(result["prediction"], eeg, atol=1e-16)
    assert result["support"]["observation_owner"] == "EEG_only"
    with pytest.raises(TypeError):
        infer_eeg_modes(eeg, loading, np.full(10, .01), hb=np.ones((48, 2)))


def test_eeg_hidden_poison_and_temporal_basis_units_cannot_change_modes():
    basis, loading = temporal_mode_basis(32, 6), spectral_loadings(channels=2)
    eeg = np.random.default_rng(234).normal(size=(32, 10))*.03
    mask = np.ones_like(eeg, bool)
    mask[8:20, :5] = False
    operator = np.eye(32)-np.ones((32, 32))/32
    kwargs = dict(visible=mask, eeg_operator=operator, state_sd=.025, ar_weight=.1)
    result = infer_eeg_modes(eeg, loading, .02, temporal_basis=basis, **kwargs)
    for poison in (1e30, np.nan, np.inf):
        changed = eeg.copy()
        changed[~mask] = poison
        alternative = infer_eeg_modes(changed, loading, .02, temporal_basis=basis, **kwargs)
        np.testing.assert_array_equal(alternative["r"], result["r"])
    alternative = infer_eeg_modes(eeg, loading, .02,
        temporal_basis=basis*np.array([2., .5, 3., 1., .7, 4.]), **kwargs)
    np.testing.assert_allclose(alternative["r"], result["r"], atol=1e-15)
    assert np.isnan(result["residual"][~mask]).all()


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_zero_extensions_are_exact_existing_forward(backend):
    driver, parameters = _driver(24), BalloonParameters()
    initial = np.array([.004, 1.008, 1.01, 1.009, 1.005])
    original = nonlinear_driver_forward(driver, initial, parameters, .25,
        derivative=True, numerical_backend=backend)
    result = fixed_driver_response(driver, parameters, .25, initial_state=initial,
                                  derivative=True, numerical_backend=backend)
    np.testing.assert_array_equal(result["states"], original["states"])
    np.testing.assert_array_equal(result["canonical_hb"], original["canonical_prediction"][:, 1:])
    np.testing.assert_array_equal(result["initial_jacobian"],
                                  original["jacobian"].reshape(24, 3, 29)[:, 1:, -5:])
    np.testing.assert_array_equal(result["driver"], driver)


def test_viscoelastic_outflow_has_consistent_volume_and_hb_mass_balances():
    parameters = BalloonParameters()
    state = np.array([.015, 1.13, 1.04, 1.08, 1.03])
    tau, tau_v = parameters.free.tau, 1.2
    derivative = response_rhs(state, .03, parameters, tau_v=tau_v)
    _, f, v, p, q = state
    outflow = (tau*v**(1/parameters.fixed.alpha)+tau_v*f)/(tau+tau_v)
    extraction = -np.expm1(np.log1p(-parameters.fixed.E0)/f)
    assert tau*derivative[2] == pytest.approx(f-outflow)
    assert tau*derivative[3] == pytest.approx(f-outflow*p/v)
    assert tau*derivative[4] == pytest.approx(f*extraction/parameters.fixed.E0-outflow*q/v)
    tied = state.copy()
    tied[3] = tied[2]
    d = response_rhs(tied, .03, parameters, tau_v=tau_v)
    assert d[2] == pytest.approx(d[3], abs=1e-15)
    legacy = balloon_rhs(physical_to_transformed(np.r_[.03, state]), parameters)[1:]*np.r_[1., state[1:]]
    np.testing.assert_allclose(response_rhs(state, .03, parameters), legacy, atol=1e-16)


def test_input_lag_and_venous_outflow_change_different_state_paths():
    driver, parameters = np.full(48, .025), BalloonParameters()
    baseline = fixed_driver_response(driver, parameters, .25)
    lagged = fixed_driver_response(driver, parameters, .25, tau_n=1.)
    compliant = fixed_driver_response(driver, parameters, .25, tau_v=1.)
    np.testing.assert_allclose(lagged["vascular_input"], .025*(1-np.exp(-np.arange(48)*.25)), atol=1e-17)
    np.testing.assert_array_equal(compliant["vascular_input"], baseline["vascular_input"])
    np.testing.assert_allclose(compliant["states"][:, 1:3], baseline["states"][:, 1:3], atol=2e-16)
    assert np.max(abs(lagged["states"][:, 1:3]-baseline["states"][:, 1:3])) > .005
    assert np.max(abs(compliant["states"][:, 3:]-baseline["states"][:, 3:])) > .001
    for result in (baseline, lagged, compliant):
        assert result["physical_check"]["valid"]
        np.testing.assert_array_equal(result["driver"], driver)


@pytest.mark.parametrize("tau_n,tau_v", [(0., 0.), (.7, 1.2)])
def test_initial_analytic_jacobian_matches_physical_log_perturbations(tau_n, tau_v):
    driver, parameters = _driver(24), BalloonParameters()
    coordinates = np.array([.004, .008, .007, .006, .005])
    operator = np.eye(24)-np.ones((24, 24))/24
    def evaluate(z, derivative=False):
        initial = np.r_[z[0], np.exp(z[1:])]
        return fixed_driver_response(driver, parameters, .25, initial_state=initial,
            tau_n=tau_n, tau_v=tau_v, hb_operator=operator, derivative=derivative)
    result = evaluate(coordinates, True)
    numerical = np.empty_like(result["initial_jacobian"])
    for j in range(5):
        plus, minus = coordinates.copy(), coordinates.copy()
        plus[j] += 1e-6
        minus[j] -= 1e-6
        numerical[:, :, j] = (evaluate(plus)["prediction"]-evaluate(minus)["prediction"])/2e-6
    np.testing.assert_allclose(result["initial_jacobian"], numerical, rtol=2e-7, atol=3e-10)


@pytest.mark.parametrize("initial_mode", ["rest", "tied_pv_logprior", "legacy_free_physicalprior"])
def test_visible_prefix_fit_ignores_hidden_values_and_keeps_driver_fixed(initial_mode):
    driver, parameters = _driver(), BalloonParameters()
    initial = np.array([.003, 1.008, 1.006, 1.006, 1.002])
    basis = _hb_basis(48)
    target = fixed_driver_response(driver, parameters, .25, initial_state=initial)["prediction"]
    target += np.einsum("tck,k->tc", basis, [.004, -.002, .001, .003])
    mask = np.zeros_like(target, bool)
    mask[:16] = True
    kwargs = dict(sd=.006, visible=mask, hb_basis=basis, initial_mode=initial_mode,
                  component_sd=.01, max_nfev=100)
    result = fit_fixed_driver_response(target, driver, parameters, .25, **kwargs)
    assert result["success"], result["convergence_reason"]
    assert result["converged"] and result["fit_performed"]
    for poison in (1e30, np.nan, np.inf):
        changed = target.copy()
        changed[~mask] = poison
        alternative = fit_fixed_driver_response(changed, driver, parameters, .25, **kwargs)
        np.testing.assert_array_equal(alternative["prediction"], result["prediction"])
        np.testing.assert_array_equal(alternative["initial_state"], result["initial_state"])
        assert alternative["objective"] == result["objective"]
    np.testing.assert_array_equal(result["driver"], driver)
    np.testing.assert_allclose(result["prediction"], result["physical_prediction"]+result["observation_component"], atol=1e-17)
    assert np.isnan(result["residual"][~mask]).all()
    if initial_mode == "tied_pv_logprior":
        assert result["initial_state"][2] == result["initial_state"][3]


def test_zero_prefix_returns_prior_prediction_without_fitted_convergence():
    driver, parameters = _driver(24), BalloonParameters()
    mask, mean = np.zeros((24, 2), bool), np.array([.002, .004, .005, .003])
    expected = fixed_driver_response(driver, parameters, .25,
        initial_state=np.r_[mean[0], np.exp(mean[1]), np.exp(mean[2]), np.exp(mean[2]), np.exp(mean[3])])
    for poison in (1e30, np.nan, np.inf):
        result = fit_fixed_driver_response(np.full((24, 2), poison), driver, parameters, .25,
            visible=mask, sd=.01, initial_mode="tied_pv_logprior", initial_prior_mean=mean,
            initial_prior_sd=[.1, .05, .04, .03], hb_basis=_hb_basis(24), component_sd=.02)
        assert result["status"] == "prior_prediction" and result["success"]
        assert not result["converged"] and not result["fit_performed"]
        assert result["evaluations"] == 1 and result["objective"] == 0
        np.testing.assert_array_equal(result["prediction"], expected["prediction"])
        np.testing.assert_array_equal(result["component_coefficients"], np.zeros(4))


def test_invalid_initial_log_prior_and_response_domains_fail_without_clipping():
    driver, parameters = _driver(24), BalloonParameters()
    for coordinate in (-1000., 1000.):
        mean = np.array([0., coordinate, 0., 0.])
        with pytest.raises((FloatingPointError, OverflowError)):
            fit_fixed_driver_response(np.zeros((24, 2)), driver, parameters, .25, sd=.01,
                initial_mode="tied_pv_logprior", initial_prior_mean=mean)
    with pytest.raises(FloatingPointError, match="initial state"):
        fixed_driver_response(driver, parameters, .25, initial_state=[0., -1., 1., 1., 1.])
    with pytest.raises(FloatingPointError):
        fixed_driver_response(np.full(24, -30.), parameters, .25, tau_v=.5)
    result = fit_fixed_driver_response(np.zeros((24, 2)), np.full(24, -30.), parameters,
                                       .25, sd=.01, tau_v=.5)
    assert result["status"] == "failed_domain" and not result["success"]
    assert result["evaluations"] == 1 and result["failure_log"]
    with pytest.raises(ValueError, match="proper"):
        fit_fixed_driver_response(np.zeros((24, 2)), driver, parameters, .25, sd=.01,
            initial_mode="tied_pv_logprior", initial_weight=0.)


def test_evaluation_budget_cannot_be_reported_as_convergence():
    result = fit_fixed_driver_response(np.full((24, 2), .03), _driver(24),
        BalloonParameters(), .25, sd=.01, hb_basis=_hb_basis(24), max_nfev=1)
    assert result["evaluations"] == 1 and not result["success"] and not result["converged"]
    assert result["convergence_reason"] == "evaluation_budget"


def test_cascades_have_analytic_step_response_and_strictly_causal_memory():
    dt, constants = .25, np.array([.5, 1., 2., 4., 8.])
    values = np.column_stack((np.ones(80), np.full(80, 2.)))
    features = cascade_response_features(values, dt)
    assert features.shape == (80, 12)
    features = features.reshape(80, 2, 6)
    time = np.arange(80)[:, None]*dt
    truth = 1-(1+time/constants)*np.exp(-time/constants)
    np.testing.assert_allclose(features[:, 0, 1:], truth, atol=1e-15)
    np.testing.assert_allclose(features[:, 1, 1:], 2*truth, atol=2e-15)
    np.testing.assert_array_equal(features[:, :, 0], values)
    impulse = np.zeros((80, 1))
    impulse[12] = 1.
    response = cascade_response_features(impulse, dt)
    assert not response[:13, 1:].any() and response[13, 1:].min() > 0
    assert np.isfinite(response).all() and np.all(response[:, 1:] >= 0)
    changed = impulse.copy()
    changed[40:] = 1e8
    np.testing.assert_array_equal(cascade_response_features(changed, dt)[:40], response[:40])
    tiny = cascade_response_features(np.ones((3, 1)), 1e-8, time_constants=[1.])
    assert tiny[1, 1] == pytest.approx(5e-17, rel=1e-7)
    stiff = cascade_response_features(np.ones((3, 1)), 1., time_constants=[1e-300])
    np.testing.assert_array_equal(stiff[:, 1], [0., 1., 1.])


def test_gaussian_scores_have_known_values_coordinate_means_and_hidden_isolation():
    target = np.zeros((3, 2))
    sd = np.array([1., 2.])
    result = gaussian_predictive_scores(target, target, sd)
    np.testing.assert_allclose(result["nll"], np.broadcast_to(np.log(sd)+.5*np.log(2*np.pi), (3, 2)))
    np.testing.assert_allclose(result["crps"], np.broadcast_to(sd*(np.sqrt(2)-1)/np.sqrt(np.pi), (3, 2)))
    np.testing.assert_allclose(result["nll_per_coordinate"], result["nll"].mean(axis=0))
    mask = np.ones((3, 2), bool)
    mask[1, 0] = False
    target[1, 0] = np.inf
    prediction = np.zeros((3, 2))
    prediction[1, 0] = np.nan
    selected = gaussian_predictive_scores(target, prediction, sd, visible=mask)
    assert selected["scored_count"] == 5 and np.isnan(selected["crps"][1, 0])
    assert selected["mean_nll"] == pytest.approx(selected["nll"][mask].mean())
    empty = gaussian_predictive_scores(target, prediction, sd, visible=np.zeros((3, 2), bool))
    assert empty["mean_nll"] is None and empty["mean_crps"] is None
    assert np.isnan(empty["nll_per_coordinate"]).all()
    for bad in (0., -1., np.nan, np.inf):
        with pytest.raises(ValueError, match="strictly positive"):
            gaussian_predictive_scores(np.zeros(2), np.zeros(2), bad)
