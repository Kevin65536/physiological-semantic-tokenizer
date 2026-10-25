"""Synthetic checks of descriptive distances; no local datasets or run reads."""
from dataclasses import replace

import numpy as np
import pytest

from src.inference import t3a_balloon_robust_ssm as core
from src.inference.model_compatibility import (
    linear_panel_dynamic_diagnostics, linear_panel_profiles, prepare_linear_panel_profiles,
    spatial_projection_residual, weak_hb_profiles,
)
from src.inference.shared_driver_reconstruction import build_shared_driver_design


def _continuous_hb(dt, tau=2., initial=None):
    """Independent frequency-domain solution plus legal arbitrary initial state."""
    time = np.arange(0, 30, dt)
    p = core.BalloonParameters(free=core.BalloonFreeParameters(tau=tau))
    jac = core.balloon_rhs_jacobian(np.zeros(6), p)
    observation = core.observation_jacobian(np.zeros(6), p)
    a, b = jac[1:, 1:], jac[1:, 0]
    state, drive = np.zeros((len(time), 5)), np.zeros(len(time))
    for frequency, amplitude, phase in ((1/30, .014, .2), (1/12, .009, 1.1), (1/7.5, .006, -.4)):
        oscillation = amplitude*np.exp(1j*(2*np.pi*frequency*time+phase))
        state += np.real(oscillation[:, None]*np.linalg.solve(2j*np.pi*frequency*np.eye(5)-a, b))
        drive += oscillation.real
    if initial is not None:
        eigenvalues, eigenvectors = np.linalg.eig(a)
        weights = np.linalg.solve(eigenvectors, initial)
        state += np.real((np.exp(time[:, None]*eigenvalues)*weights)@eigenvectors.T)
    y = state@observation[:, 1:].T + drive[:, None]*observation[:, 0]
    return time, y[:, 1:]


def test_weak_closure_profiles_legal_initial_state_and_plus_one_offsets():
    time, hb = _continuous_hb(.1, initial=[.01, .02, .03, .13, .01])
    result = weak_hb_profiles(hb+1, times=time, return_details=True)
    assert result["status"].tolist() == ["ok"]
    assert result["nominal_D"][0] < 2e-13
    assert result["best_tau"][0] == 2.
    assert result["nominal_nuisance_rank"] == 2
    detail = result["nominal_details"]
    np.testing.assert_allclose(detail["left"]-detail["right"], detail["residual"], atol=1e-17)
    np.testing.assert_allclose(np.sum(detail["residual"]**2, axis=0)/result["denominator"],
                               result["nominal_D"], rtol=1e-14)


def test_weak_closure_scale_offsets_and_batch_are_invariant():
    time, hb = _continuous_hb(.1)
    batch = np.stack((hb, 23*hb+[11., -7.], 1e-8*hb), axis=1)
    result = weak_hb_profiles(batch, times=time)
    assert result["profiles"].shape == (3, 31)
    assert result["status"].tolist() == ["ok"]*3
    np.testing.assert_allclose(result["profiles"][1:],
        np.broadcast_to(result["profiles"][0], (2, 31)), rtol=2e-10, atol=1e-13)
    np.testing.assert_array_equal(result["best_tau"], [2., 2., 2.])


def test_independent_hb_component_is_detected_at_fixed_and_profiled_tau():
    time, hb = _continuous_hb(.1)
    altered = hb.copy()
    altered[:, 1] += .03*np.sin(2*np.pi*time/6.3+.7)
    result = weak_hb_profiles(np.stack((hb, altered), axis=1), times=time)
    assert result["best_D"][0] < 1e-10
    assert result["nominal_D"][1] > .01
    assert result["best_D"][1] > 1e-3


def test_weak_quadrature_refinement_and_native_4762_clock():
    scores = []
    for dt in (.1, .05, 1/47.62):
        time, hb = _continuous_hb(dt, initial=[.01, .02, .03, .13, .01])
        actual = weak_hb_profiles(hb, times=time, taus=[2.])
        fixed_dt = weak_hb_profiles(hb, dt=dt, taus=[2.])
        np.testing.assert_array_equal(actual["profiles"], fixed_dt["profiles"])
        scores.append(actual["nominal_D"][0])
    assert scores[2] < scores[1] < scores[0] < 2e-13


def test_weak_zero_missing_and_short_support_are_not_passes():
    hb = np.zeros((300, 3, 2))
    hb[:, 0] = [2.3, -4.7]
    hb[4, 1, 0] = np.nan
    hb[:, 2, 0] = np.sin(np.arange(300)/30)
    result = weak_hb_profiles(hb, dt=.1)
    assert result["status"].tolist() == ["insufficient_energy", "missing_support", "ok"]
    assert np.isnan(result["nominal_D"][:2]).all()
    assert np.isnan(result["best_tau"][:2]).all()
    short = weak_hb_profiles(hb[:20], dt=.1, return_details=True)
    assert short["status"].tolist() == ["insufficient_support", "missing_support", "insufficient_support"]
    assert short["kernel_count"] == 0
    with pytest.raises(ValueError):
        weak_hb_profiles(hb, times=np.zeros(300))


def test_fixed_spatial_distance_retains_raw_energy_and_rank_bound():
    direction = np.array([1., 2., -1.])
    direction /= np.linalg.norm(direction)
    orthogonal = np.array([2., -1., 0.])
    orthogonal /= np.linalg.norm(orthogonal)
    time = np.arange(100)
    base = np.sin(time/10)[:, None]*direction
    good = spatial_projection_residual(base, direction[:, None])
    rotated = spatial_projection_residual(np.sin(time/10)[:, None]*orthogonal, direction[:, None])
    assert good["D"] < 1e-28 and good["rank"] == 1
    assert rotated["D"] == pytest.approx(1.)
    assert rotated["rank_D"] < 1e-28  # Low rank alone cannot recover a fixed readout.
    assert rotated["residual_energy"] == pytest.approx(rotated["energy"])
    zero = spatial_projection_residual(np.zeros_like(base), direction[:, None])
    assert zero["status"] == "insufficient_energy" and np.isnan(zero["D"])
    base[5, 0] = np.nan
    assert spatial_projection_residual(base, direction[:, None])["status"] == "missing_support"
    mask = np.arange(100) != 5
    assert spatial_projection_residual(base, direction[:, None], mask=mask)["status"] == "ok"


def _panel():
    n, dt = 40, .5
    p = core.BalloonParameters(free=core.BalloonFreeParameters(kappa=.64))
    # Distinct actual temporal operators; they need not commute with the
    # dynamics, and must therefore act on the forward mean before projection.
    time_operator = np.eye(n)
    time_operator[1:, :-1] += .2*np.eye(n-1)
    operator = np.kron(time_operator, np.diag([1., 2., .7]))
    design = build_shared_driver_design(p, n, dt, processed_mean_operator=operator)
    coefficient = np.r_[.02*np.sin(np.arange(n)*.3), [.01, .02, .03, .13, .01]]
    constants = operator@np.tile(np.eye(3), (n, 1))
    y = (design.observation_design@coefficient+constants@np.array([2., -3., 1.])).reshape(n, 3)+design.offset
    return y, operator, np.array([.1, .2, .3]), dt, p


def test_linear_panel_uses_actual_operator_initial_offsets_and_same_theta_predictions():
    y, operator, sd, dt, p = _panel()
    result = linear_panel_profiles(y, operator, sd, [1., 2., 4.], [.4, .64], dt=dt,
                                   parameters=p, return_predictions=True)
    nominal = result["nominal"]
    assert nominal["joint_nmse"] < 1e-23
    assert nominal["hb_nmse"] < 1e-23
    assert nominal["joint_rank"] < y.size
    pred = result["predictions"]["nominal"]
    np.testing.assert_allclose(pred["joint_prediction"], y, atol=2e-12)
    np.testing.assert_allclose(pred["hb_prediction"][:, 1:], y[:, 1:], atol=2e-12)
    assert np.isnan(pred["hb_prediction"][:, 0]).all()
    assert all(row["joint_sse"] >= row["hb_sse"]-1e-20 for row in result["profiles"])
    assert result["best_joint"]["tau"] == 2.
    assert result["best_joint"]["kappa"] == .64


def test_panel_cross_increment_and_component_scores_keep_joint_fit_distinct():
    y, operator, sd, dt, p = _panel()
    # An independent high-frequency EEG component violates the common driver.
    y[:, 0] += .1*np.cos(np.arange(len(y))*1.3)
    result = linear_panel_profiles(y, operator, sd, [2.], [.64], dt=dt, parameters=p)
    row = result["nominal"]
    assert row["hb_nmse"] < 1e-23
    assert row["cross_increment_nmse"] > 1e-5
    assert row["joint_nmse"] > row["hb_nmse"]
    assert row["joint_eeg_nmse"] == row["eeg_nmse"]
    assert row["joint_hb_nmse"] == pytest.approx((row["joint_hbo_nmse"]+row["joint_hbr_nmse"])/2)
    assert row["cross_increment_nrmse"]**2 == pytest.approx(row["cross_increment_nmse"])


def test_prepared_panel_matches_uncached_and_hides_unselected_values():
    y, operator, sd, dt, p = _panel()
    mask = np.ones(y.shape, bool)
    mask[:3] = False
    cache = prepare_linear_panel_profiles(len(y), dt, operator, sd, [2.], [.64], mask, parameters=p)
    one = linear_panel_profiles(y, operator, sd, [2.], [.64], mask, dt=dt, parameters=p, prepared=cache)
    y[~mask] = np.nan
    two = linear_panel_profiles(y, operator, sd, [2.], [.64], mask, dt=dt, parameters=p, prepared=cache)
    assert one["profiles"] == two["profiles"]
    y[5, 0] = np.nan
    assert linear_panel_profiles(y, operator, sd, [2.], [.64], mask,
        dt=dt, parameters=p, prepared=cache)["status"] == "missing_support"
    with pytest.raises(ValueError, match="do not match"):
        linear_panel_profiles(y, operator, sd*2, [2.], [.64], mask, dt=dt, parameters=p, prepared=cache)
    changed = replace(p, fixed=replace(p.fixed, alpha=.4))
    with pytest.raises(ValueError, match="do not match"):
        linear_panel_profiles(y, operator, sd, [2.], [.64], mask, dt=dt, parameters=changed, prepared=cache)


def test_panel_zero_or_transformed_constant_is_not_evidence_of_excitation():
    y, operator, sd, dt, p = _panel()
    cache = prepare_linear_panel_profiles(len(y), dt, operator, sd, [2.], [.64], parameters=p)
    zero = linear_panel_profiles(np.zeros_like(y), operator, sd, [2.], [.64],
        dt=dt, parameters=p, prepared=cache)
    assert zero["status"] == "ok"
    assert zero["nominal"]["joint_nmse"] == 0.
    assert zero["zero_prediction_nmse"] == 0.
    assert zero["dynamic_nmse"] == 0.
    assert not zero["dynamic_support"]
    assert zero["interpretation"] == "insufficient_excitation"
    constants = operator@np.tile(np.eye(3), (len(y), 1))
    constant_y = (constants@np.array([2., -3., 1.])).reshape(y.shape)
    constant = linear_panel_profiles(constant_y, operator, sd, [2.], [.64],
        dt=dt, parameters=p, prepared=cache)
    assert not constant["dynamic_support"]
    assert constant["zero_prediction_nmse"] > 1.
    assert constant["dynamic_nmse"] < 1e-25
    dynamic = linear_panel_profiles(y, operator, sd, [2.], [.64],
        dt=dt, parameters=p, prepared=cache)
    assert dynamic["dynamic_support"]


@pytest.mark.parametrize("steps", [6, 10, 40])
def test_annihilated_constants_do_not_create_roundoff_nuisance_directions(steps):
    operator = np.kron(np.eye(steps)-np.ones((steps, steps))/steps, np.eye(3))
    constants = operator@np.tile(np.eye(3), (steps, 1))
    u, singular, _ = np.linalg.svd(constants, full_matrices=False)
    assert 0 < singular[0] < 1e-13
    # This direction exposed the original bug: a relative SVD of constants
    # alone treated roundoff as rank three and erased a unit-energy target.
    target = u[:, 0].reshape(steps, 3)
    old_basis = u[:, singular > singular[0]*1e-10]
    old_energy = np.sum((target.ravel()-old_basis@(old_basis.T@target.ravel()))**2)
    assert old_energy < 1e-28
    diagnostics = linear_panel_dynamic_diagnostics(target, operator, np.ones(3))
    assert diagnostics["constant_rank"] == 0
    assert diagnostics["dynamic_energy"] == pytest.approx(1.)
    assert diagnostics["dynamic_support"]
    assert diagnostics["dynamic_nmse"] == diagnostics["zero_prediction_nmse"]
    # Main model bases and predictions do not use the constant-only basis.
    # Replacing it with the old bogus basis changes only diagnostic fields.
    prepared = prepare_linear_panel_profiles(steps, .25, operator, np.ones(3), [2.], [.64])
    old_prepared = dict(prepared, constant_basis=old_basis, constant_rank=old_basis.shape[1])
    fixed = linear_panel_profiles(target, operator, np.ones(3), [2.], [.64], dt=.25,
        prepared=prepared, return_predictions=True)
    old = linear_panel_profiles(target, operator, np.ones(3), [2.], [.64], dt=.25,
        prepared=old_prepared, return_predictions=True)
    assert fixed["profiles"] == old["profiles"]
    np.testing.assert_array_equal(fixed["predictions"]["nominal"]["joint_prediction"],
                                  old["predictions"]["nominal"]["joint_prediction"])


def test_constant_rank_reference_preserves_operator_and_unit_scaling():
    steps = 12
    target = np.tile([2., -3., 1.], (steps, 1))
    for multiplier in (1e-15, 1., 1e15):
        operator = multiplier*np.eye(3*steps)
        sd = multiplier*np.array([.1, .2, .3])
        result = linear_panel_dynamic_diagnostics(multiplier*target, operator, sd)
        assert result["constant_rank"] == 3
        assert not result["dynamic_support"]
    # One modality's constant survives; annihilated modalities contribute no
    # arbitrary extra rank even when combined with a genuinely visible one.
    center = np.eye(steps)-np.ones((steps, steps))/steps
    operator = np.kron(center, np.diag([0., 1., 1.]))+np.kron(np.eye(steps), np.diag([1., 0., 0.]))
    result = linear_panel_dynamic_diagnostics(target, operator, np.ones(3))
    assert result["constant_rank"] == 1
