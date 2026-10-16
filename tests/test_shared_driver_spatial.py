"""Synthetic spatial-core contracts; no dataset or retained-run access."""
import numpy as np
import pytest

from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
from src.inference.shared_driver_spatial import (
    collapse_tied_spatial_design, fit_spatial_shared_driver,
    fit_tied_spatial_shared_driver, forward_spatial_shared_driver, temporal_mode_basis,
)
from src.inference.t3a_balloon_robust_ssm import BalloonParameters


def _fixture(t=16, m=3, k=2):
    basis = temporal_mode_basis(t, m)
    eeg = np.array([[1., .3], [.2, 1.], [.6, -.4]])[:, :k]
    hb = np.array([[.9, .1], [.2, .8], [.5, .5]]) if k == 2 else np.ones((1, 1))
    coef = np.zeros((m, k))
    coef[0] = .003
    coef[1] = np.array([.015, -.009])[:k]
    coef[2] = np.array([.006, .011])[:k]
    kwargs = dict(eeg_loadings=eeg, hb_loadings=hb, temporal_basis=basis,
                  numerical_backend="python")
    return coef, kwargs


def _truth(t=16, m=3, k=2):
    coef, kwargs = _fixture(t, m, k)
    truth = forward_spatial_shared_driver(coef, np.zeros((k, 5)),
        BalloonParameters(), .25, **kwargs)
    return truth, kwargs


def test_k1_j1_is_exact_existing_core_including_nonrest_initials():
    p = BalloonParameters()
    coef, kwargs = _fixture(k=1)
    kwargs["eeg_loadings"] = np.array([[p.fixed.eeg_loading]])
    initial = np.array([[.006, .004, -.003, .002, -.001]])
    actual = forward_spatial_shared_driver(coef, initial, p, .25, **kwargs)
    physical = np.r_[initial[0, 0], np.exp(initial[0, 1:])]
    core = nonlinear_driver_forward(kwargs["temporal_basis"]@coef[:, 0], physical,
        p, .25, numerical_backend="python", return_flow_jacobian=True)
    np.testing.assert_array_equal(actual["prediction"], core["canonical_prediction"])
    np.testing.assert_array_equal(actual["states"][:, 0], core["states"])
    chain = np.zeros((16+5, 3+5))
    chain[:16, :3] = kwargs["temporal_basis"]
    chain[16:, 3:] = np.eye(5)
    np.testing.assert_allclose(actual["jacobian"], core["jacobian"]@chain, atol=1e-17)
    np.testing.assert_allclose(actual["flow_jacobian"][:, 0], core["flow_jacobian"]@chain, atol=1e-17)


def test_k2_has_distinct_hb_waveforms_and_preserves_total_hb():
    coef, kwargs = _fixture(t=32)
    kwargs["hb_loadings"] = np.eye(2)
    kwargs["hb_gains"] = np.array([1.3, .8])
    p = BalloonParameters()
    actual = forward_spatial_shared_driver(coef, np.zeros((2, 5)), p, .25, **kwargs)
    pair = actual["hb_prediction"]
    # Mere spatial loading of one waveform has rank one in each Hb species.
    assert np.linalg.svd(pair[:, :, 0], compute_uv=False)[1] > .002
    assert np.linalg.svd(pair[:, :, 1], compute_uv=False)[1] > .001
    np.testing.assert_allclose(pair.sum(axis=2),
        (actual["states"][:, :, 4]-1)*p.fixed.P0*kwargs["hb_gains"], atol=2e-16)
    assert actual["total_unknowns"] == 2*(3+5)


def test_identical_k2_cores_reduce_to_k1_after_spatial_mixing():
    """Two named spatial columns must not manufacture a second dynamic."""
    coef, kwargs = _fixture(t=24)
    coef[:, 1] = coef[:, 0]
    initial = np.tile(np.array([.004, .002, -.003, .001, -.002]), (2, 1))
    reference = np.eye(24)
    reference[:, :4] -= .25
    kwargs.update(hb_operator=reference, eeg_operator=reference)
    two = forward_spatial_shared_driver(coef, initial, BalloonParameters(), .25, **kwargs)
    one = forward_spatial_shared_driver(coef[:, :1], initial[:1], BalloonParameters(), .25,
        **dict(kwargs, eeg_loadings=kwargs["eeg_loadings"].sum(axis=1, keepdims=True),
               hb_loadings=np.ones((len(kwargs["hb_loadings"]), 1))))
    np.testing.assert_allclose(two["prediction"], one["prediction"], rtol=0, atol=3e-16)
    np.testing.assert_array_equal(two["states"][:, 0], two["states"][:, 1])


@pytest.mark.parametrize("channel_operators", [False, True])
def test_all_analytic_columns_match_finite_difference_with_nonrest_initials(channel_operators):
    coef, kwargs = _fixture(t=12)
    t = len(kwargs["temporal_basis"])
    kwargs["hb_gains"] = np.array([1.3, .7, 1.1])
    kwargs["eeg_operator"] = np.eye(t)-np.ones((t, 1))@np.r_[.5, .5, np.zeros(t-2)][None]
    operator = np.eye(t)-np.ones((t, t))/t
    kwargs["hb_operator"] = (np.stack([operator, np.eye(t), .7*operator])
                              if channel_operators else operator)
    initial = np.array([[.006, .004, -.003, .002, -.001],
                        [-.003, -.002, .005, -.004, .002]])
    p = BalloonParameters()
    actual = forward_spatial_shared_driver(coef, initial, p, .25, **kwargs)
    packed = np.r_[coef.ravel(), initial.ravel()]
    numerical = np.empty_like(actual["jacobian"])
    numerical_flow = np.empty_like(actual["flow_jacobian"])
    for column in range(len(packed)):
        predictions, flows = [], []
        for sign in (-1., 1.):
            shifted = packed.copy()
            shifted[column] += sign*1e-6
            forward = forward_spatial_shared_driver(shifted[:coef.size].reshape(coef.shape),
                shifted[coef.size:].reshape(2, 5), p, .25, derivative=False, **kwargs)
            predictions.append(forward["prediction"].ravel())
            flows.append(forward["states"][:, :, 2])
        numerical[:, column] = (predictions[1]-predictions[0])/2e-6
        numerical_flow[:, :, column] = (flows[1]-flows[0])/2e-6
    np.testing.assert_allclose(actual["jacobian"], numerical, rtol=3e-6, atol=4e-10)
    np.testing.assert_allclose(actual["flow_jacobian"], numerical_flow, rtol=3e-6, atol=4e-10)


def test_compiled_forward_matches_python():
    coef, kwargs = _fixture(t=12)
    initial = np.zeros((2, 5))
    reference = forward_spatial_shared_driver(coef, initial, BalloonParameters(), .25, **kwargs)
    kwargs["numerical_backend"] = "numba"
    compiled = forward_spatial_shared_driver(coef, initial, BalloonParameters(), .25, **kwargs)
    for key in ("prediction", "states", "jacobian", "flow_jacobian"):
        np.testing.assert_allclose(compiled[key], reference[key], rtol=2e-12, atol=2e-14)


def test_hidden_poison_never_changes_initialization_fit_or_residual():
    truth, kwargs = _truth()
    clean = truth["prediction"]
    mask = np.ones_like(clean, bool)
    mask[4:10, 3:5] = False
    mask[:, 1] = False
    options = dict(**kwargs, sd=np.full(clean.shape[1], .03), visible=mask,
        max_nfev=60, gradient_tolerance=2e-6)
    reference = fit_spatial_shared_driver(clean[:, :3], clean[:, 3:].reshape(16, 3, 2),
        BalloonParameters(), .25, **options)
    assert reference["success"], reference["convergence_reason"]
    for poison in (1e30, np.nan, np.inf):
        changed = clean.copy()
        changed[~mask] = poison
        actual = fit_spatial_shared_driver(changed[:, :3], changed[:, 3:].reshape(16, 3, 2),
            BalloonParameters(), .25, **options)
        for key in ("prediction", "coefficients", "initial_coefficients"):
            np.testing.assert_array_equal(actual[key], reference[key])
        assert actual["objective"] == reference["objective"]
        assert actual["nfev"] == reference["nfev"]
        assert np.isnan(actual["residual"][~mask]).all()


def test_zero_eeg_weight_does_not_use_eeg_even_for_initialization():
    truth, kwargs = _truth()
    options = dict(**kwargs, sd=np.full(9, .03), eeg_weight=0., max_nfev=45,
                   gradient_tolerance=2e-6)
    reference = fit_spatial_shared_driver(truth["eeg_prediction"], truth["hb_prediction"],
        BalloonParameters(), .25, **options)
    for poison in (1e30, np.nan, np.inf):
        actual = fit_spatial_shared_driver(np.full((16, 3), poison), truth["hb_prediction"],
            BalloonParameters(), .25, visible=np.ones((16, 9), bool), **options)
        np.testing.assert_array_equal(actual["prediction"], reference["prediction"])
        np.testing.assert_array_equal(actual["initial_coefficients"], np.zeros((3, 2)))
        assert actual["objective"] == reference["objective"]
        assert actual["objective_components"]["eeg_data"] == 0.
        assert actual["support"]["visible_eeg_loading_rank"] == 0
        assert np.isnan(actual["residual"][:, :3]).all()


def test_visible_modality_mean_uses_mask_only_and_preserves_modality_weight():
    t, f, j = 8, 3, 2
    options = dict(eeg_loadings=np.ones((f, 1)), hb_loadings=np.ones((j, 1)),
        temporal_basis=temporal_mode_basis(t, 1), sd=np.ones(f+2*j),
        max_nfev=1, numerical_backend="python")
    eeg, hb = np.zeros((t, f)), np.full((t, j, 2), .02)
    mask = np.ones((t, f+2*j), bool)
    mask[::2, f+2:] = False
    full = fit_spatial_shared_driver(eeg, hb, BalloonParameters(), .25, **options)
    masked = fit_spatial_shared_driver(eeg, hb, BalloonParameters(), .25,
        visible=mask, **options)
    assert full["objective_components"]["hb_data"] == pytest.approx(t*.02**2)
    assert masked["objective_components"]["hb_data"] == pytest.approx(t*.02**2)
    assert masked["support"]["effective_denominators"] == dict(eeg=3., hb=3.)
    assert full["complexity"]["effective_denominators"] == dict(eeg=3., hb=4.)
    poisoned = hb.copy().reshape(t, 2*j)
    poisoned[~mask[:, f:]] = np.inf
    poison_fit = fit_spatial_shared_driver(eeg, poisoned.reshape(t, j, 2),
        BalloonParameters(), .25, visible=mask, **options)
    assert poison_fit["objective"] == masked["objective"]
    assert poison_fit["support"]["effective_denominators"] == masked["support"]["effective_denominators"]
    with pytest.raises(ValueError, match="finite visible"):
        fit_spatial_shared_driver(eeg, poisoned.reshape(t, j, 2),
            BalloonParameters(), .25, **options)


def test_budget_failure_retains_last_physical_fit_and_requires_stationarity():
    truth, kwargs = _truth()
    actual = fit_spatial_shared_driver(truth["eeg_prediction"], truth["hb_prediction"],
        BalloonParameters(), .25, sd=np.full(9, .01), max_nfev=1, **kwargs)
    assert actual["nfev"] == 1
    assert not actual["success"] and not actual["converged"]
    assert actual["convergence_reason"] == "evaluation_budget"
    assert actual["physical_check"]["valid"]
    assert len(actual["physical_checks"]) == 2
    assert np.isfinite(actual["prediction"]).all()
    assert actual["objective"] == pytest.approx(sum(actual["objective_components"].values()))
    assert actual["failure_log"][-1]["reason"] == "evaluation_budget"
    assert actual["scaled_gradient_inf_norm"] > 1e-6


def test_physical_domain_rejections_are_not_clipped():
    coef, kwargs = _fixture(t=8)
    initial = np.zeros((2, 5))
    initial[1, 3] = -3.
    with pytest.raises(FloatingPointError, match="Hb balance"):
        forward_spatial_shared_driver(coef, initial, BalloonParameters(), .25, **kwargs)
    # A strong negative driver eventually reaches a nonpositive RK stage.
    coef[:] = 0
    coef[0] = -10.
    with pytest.raises(FloatingPointError):
        forward_spatial_shared_driver(coef, np.zeros((2, 5)), BalloonParameters(), .5, **kwargs)


def test_initialization_domain_failures_and_budget_are_retained():
    coef, kwargs = _fixture(t=8)
    extreme_eeg = np.full((8, 3), -100.)
    actual = fit_spatial_shared_driver(extreme_eeg, np.zeros((8, 3, 2)),
        BalloonParameters(), .5, sd=np.ones(9), max_nfev=1, **kwargs)
    assert not actual["success"] and actual["status"] == "failed_domain"
    assert actual["nfev"] == 1
    assert actual["failure_log"][0]["stage"] == "initialization"
    assert np.isnan(actual["prediction"]).all()


def test_integration_refinement_reduces_error_for_same_fixed_drivers():
    coef, kwargs = _fixture(t=20)
    coef *= 4
    initial = np.zeros((2, 5))
    results = [forward_spatial_shared_driver(coef, initial, BalloonParameters(), .8,
        substeps=n, derivative=False, **kwargs) for n in (1, 2, 8)]
    coarse = np.linalg.norm(results[0]["hb_prediction"]-results[2]["hb_prediction"])
    refined = np.linalg.norm(results[1]["hb_prediction"]-results[2]["hb_prediction"])
    assert coarse > 1e-6
    assert refined < coarse/8
    np.testing.assert_array_equal(results[0]["drivers"], results[1]["drivers"])


def test_support_reports_rank_collapse_without_rotating_coordinates():
    truth, kwargs = _truth(t=8)
    kwargs["hb_loadings"] = np.full((3, 2), .5)
    actual = fit_spatial_shared_driver(truth["eeg_prediction"], truth["hb_prediction"],
        BalloonParameters(), .25, sd=np.ones(9), max_nfev=1, **kwargs)
    assert actual["support"]["hb_loading_rank"] == 1
    assert actual["support"]["eeg_loading_rank"] == 2
    assert actual["complexity"]["total_parameters"] == 16
    assert actual["total_unknowns"] == 16


@pytest.mark.parametrize("change,match", [
    ("eeg_rank", "full-column-rank"), ("hb_negative", "nonnegative"),
    ("hb_sum", "row-sum"), ("hb_support", "supported"),
    ("gain", "positive finite"), ("basis_rank", "full-column-rank"),
    ("operator", "hb_operator"),
])
def test_invalid_fixed_design_fails_before_forward(change, match):
    coef, kwargs = _fixture(t=8)
    if change == "eeg_rank":
        kwargs["eeg_loadings"] = np.ones((3, 2))
    elif change == "hb_negative":
        kwargs["hb_loadings"][0] = [-.1, 1.1]
    elif change == "hb_sum":
        kwargs["hb_loadings"] *= 2
    elif change == "hb_support":
        kwargs["hb_loadings"][:] = [1., 0.]
    elif change == "gain":
        kwargs["hb_gains"] = np.array([1., 0., 1.])
    elif change == "basis_rank":
        kwargs["temporal_basis"][:, 2] = kwargs["temporal_basis"][:, 1]
    else:
        kwargs["hb_operator"] = np.eye(7)
    with pytest.raises(ValueError, match=match):
        forward_spatial_shared_driver(coef, np.zeros((2, 5)), BalloonParameters(), .25, **kwargs)


def test_curvature_dt_scaling_and_flow_physical_initial_components_are_exact():
    truth, kwargs = _truth(t=12)
    # The first evaluation retains the visible-EEG warm start, with rest initial.
    options = dict(**kwargs, sd=np.full(9, .02), driver_curvature_weight=.04,
                   flow_prior_weight=.7, flow_log_sd=.6, max_nfev=1)
    actual = fit_spatial_shared_driver(truth["eeg_prediction"], truth["hb_prediction"],
        BalloonParameters(), .25, **options)
    curvature = .04*.25*np.sum((np.diff(actual["drivers"], n=2, axis=0)/.25**2)**2)
    flow = .7*.25*np.sum(np.log(actual["states"][:, :, 2])**2)/.6**2
    assert actual["objective_components"]["driver_curvature"] == pytest.approx(curvature)
    assert actual["objective_components"]["flow_prior"] == pytest.approx(flow)
    assert actual["objective_components"]["initial_state"] == 0.


def test_tied_collapse_preserves_nonrest_forward_and_constrained_jacobian():
    coefficients, kwargs = _fixture(t=12)
    coefficients[:, 1] = coefficients[:, 0]
    initial = np.tile([.006, .004, -.003, .002, -.001], (2, 1))
    p, dt = BalloonParameters(), .25
    kwargs["hb_gains"] = np.array([1.3, .7, 1.1])
    operator = np.eye(12)-np.ones((12, 12))/12
    kwargs.update(eeg_operator=operator,
                  hb_operator=np.stack([operator, np.eye(12), .7*operator]))
    collapsed = collapse_tied_spatial_design(kwargs["eeg_loadings"], kwargs["hb_loadings"])
    np.testing.assert_array_equal(collapsed["eeg_loadings"],
                                  kwargs["eeg_loadings"].sum(axis=1, keepdims=True))
    assert np.linalg.norm(collapsed["eeg_loadings"]) != pytest.approx(1.)
    single_kwargs = dict(kwargs, **collapsed)
    full = forward_spatial_shared_driver(coefficients, initial, p, dt, **kwargs)
    single = forward_spatial_shared_driver(coefficients[:, :1], initial[:1], p, dt,
                                           **single_kwargs)
    np.testing.assert_allclose(single["prediction"], full["prediction"], rtol=0, atol=3e-16)
    np.testing.assert_allclose(single["states"][:, 0], full["states"][:, 0], rtol=0, atol=3e-16)
    m = len(coefficients)
    embedding = np.zeros((2*(m+5), m+5))
    embedding[:2*m] = np.repeat(np.eye(m, m+5), 2, axis=0)
    embedding[2*m:2*m+5, m:] = embedding[2*m+5:, m:] = np.eye(5)
    constrained = full["jacobian"]@embedding
    np.testing.assert_allclose(single["jacobian"], constrained, rtol=3e-15, atol=3e-16)
    packed = np.r_[coefficients[:, 0], initial[0]]
    numerical = np.empty_like(single["jacobian"])
    for index in range(len(packed)):
        predictions = []
        for direction in (-1., 1.):
            changed = packed.copy()
            changed[index] += direction*1e-6
            forward = forward_spatial_shared_driver(changed[:m, None],
                changed[m:][None], p, dt, derivative=False, **single_kwargs)
            predictions.append(forward["prediction"].ravel())
        numerical[:, index] = (predictions[1]-predictions[0])/2e-6
    np.testing.assert_allclose(single["jacobian"], numerical, rtol=3e-6, atol=4e-10)


def test_tied_fit_objective_equals_original_k2_restriction_including_all_priors():
    coefficients, kwargs = _fixture(t=12)
    coefficients[:, 1] = coefficients[:, 0]
    initial = np.tile([.006, .004, -.003, .002, -.001], (2, 1))
    p, dt = BalloonParameters(), .25
    kwargs["hb_gains"] = np.array([1.3, .7, 1.1])
    truth = forward_spatial_shared_driver(coefficients, initial, p, dt, **kwargs)
    target = truth["prediction"].copy()
    target[:, 3:] += .0002*np.sin(np.arange(12)[:, None])
    visible = np.ones_like(target, bool)
    visible[4:8, 3:5] = False
    target[~visible] = np.nan
    sd = np.linspace(.01, .02, 9)
    curvature_weight, initial_weight, flow_weight, flow_sd = .003, .2, .07, .6
    fitted = fit_tied_spatial_shared_driver(target[:, :3], target[:, 3:].reshape(12, 3, 2),
        p, dt, **kwargs, sd=sd, visible=visible,
        driver_curvature_weight=curvature_weight, initial_weight=initial_weight,
        flow_prior_weight=flow_weight, flow_log_sd=flow_sd, max_nfev=70,
        gradient_tolerance=2e-7)
    assert fitted["success"], fitted["convergence_reason"]
    assert fitted["states"].shape == (12, 1, 6)
    assert fitted["initial_coordinates"].shape == (1, 5)
    assert fitted["coefficients"].shape == (3, 1)
    assert fitted["total_unknowns"] == 8
    assert fitted["original_K"] == 2 and fitted["effective_K"] == 1 and fitted["tied"]
    assert fitted["complexity"]["unrestricted_total_parameters"] == 16
    full = forward_spatial_shared_driver(np.repeat(fitted["coefficients"], 2, axis=1),
        np.repeat(fitted["initial_coordinates"], 2, axis=0), p, dt, **kwargs)
    np.testing.assert_allclose(fitted["prediction"], full["prediction"], rtol=0, atol=3e-16)
    residual = np.full_like(target, np.nan)
    residual[visible] = ((full["prediction"]-target)/sd)[visible]
    components = dict(
        eeg_data=12/visible[:, :3].sum()*np.nansum(residual[:, :3]**2),
        hb_data=12/visible[:, 3:].sum()*np.nansum(residual[:, 3:]**2),
        driver_curvature=curvature_weight*dt*np.sum((np.diff(full["drivers"], n=2, axis=0)/dt**2)**2),
        initial_state=initial_weight*np.sum((full["initial_state"]-np.r_[0., np.ones(4)])**2),
        flow_prior=flow_weight*dt*np.sum(np.log(full["states"][:, :, 2])**2)/flow_sd**2)
    assert all(value > 0 for value in components.values())
    for name, value in components.items():
        assert fitted["objective_components"][name] == pytest.approx(value, rel=1e-12, abs=2e-15)
    assert fitted["objective"] == pytest.approx(sum(components.values()), rel=1e-12)
    replay = forward_spatial_shared_driver(fitted["coefficients"], fitted["initial_coordinates"],
        p, dt, **dict(kwargs, eeg_loadings=fitted["forward_eeg_loadings"],
                     hb_loadings=fitted["forward_hb_loadings"]))
    np.testing.assert_array_equal(replay["prediction"], fitted["prediction"])
    assert fitted["driver_curvature_weight"] == 2*curvature_weight
    assert fitted["initial_weight"] == 2*initial_weight
    assert fitted["flow_prior_weight"] == 2*flow_weight


@pytest.mark.parametrize("change,match", [
    ("cancel", "cancels"), ("near_cancel", "nearly zero"),
    ("negative", "nonnegative"), ("row_sum", "row-sum"),
    ("unsupported", "supported"), ("wrong_k", "K=2"),
])
def test_tied_invalid_design_fails_before_any_refit(monkeypatch, change, match):
    import src.inference.shared_driver_spatial as spatial
    _, kwargs = _fixture(t=8)
    if change == "cancel":
        kwargs["eeg_loadings"][:, 1] = -kwargs["eeg_loadings"][:, 0]
    elif change == "near_cancel":
        kwargs["eeg_loadings"][:, 1] = -kwargs["eeg_loadings"][:, 0]+np.array([1., -1., 1.])*1e-12
    elif change == "negative":
        kwargs["hb_loadings"][0] = [-.1, 1.1]
    elif change == "row_sum":
        kwargs["hb_loadings"] *= 2
    elif change == "unsupported":
        kwargs["hb_loadings"][:] = [1., 0.]
    else:
        kwargs["eeg_loadings"] = kwargs["eeg_loadings"][:, :1]
    def unexpected_refit(*args, **options):
        raise AssertionError("invalid original design must not trigger any refit")
    monkeypatch.setattr(spatial, "fit_spatial_shared_driver", unexpected_refit)
    with pytest.raises(ValueError, match=match):
        fit_tied_spatial_shared_driver(np.zeros((8, 3)), np.zeros((8, 3, 2)),
            BalloonParameters(), .25, **kwargs, sd=np.ones(9))
