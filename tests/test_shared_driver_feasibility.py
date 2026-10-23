"""Synthetic exact observation-space bounds; no measured evidence is read."""
import numpy as np
import pytest

from src.inference.shared_driver_modes import temporal_mode_basis
from src.inference.shared_driver_spatial import forward_spatial_shared_driver
from src.inference.source_observability import (
    eeg_representation_lower_bound, fit_feasibility_frontier, hb_representation_lower_bound,
)
from src.inference.t3a_balloon_robust_ssm import BalloonFreeParameters, BalloonParameters


def _eeg_design(basis, loading, operator, scale, mask):
    # Independent column construction fixes coefficient flattening/order.
    columns = []
    for row in range(basis.shape[1]):
        for column in range(loading.shape[1]):
            response = np.outer(operator@basis[:, row], loading[:, column])
            columns.append((response/scale)[mask])
    return np.column_stack(columns)


def _hb_prediction(latent, loading, gain, operators):
    t, j = len(latent), len(loading)
    prediction = np.zeros((t, j, 2))
    for channel in range(j):
        prediction[:, channel] = operators[channel]@(
            gain[channel]*np.einsum("k,tkc->tc", loading[channel], latent))
    return prediction


@pytest.mark.parametrize("k", [1, 2])
def test_eeg_matched_observation_map_has_zero_bound_and_minimum_norm(k):
    rng = np.random.default_rng(6+k)
    t, f, m = 15, 7, 4
    basis, loading = rng.normal(size=(t, m)), rng.normal(size=(f, k))
    operator = np.eye(t)-np.ones((t, t))/t
    coefficient, scale = rng.normal(size=(m, k)), np.linspace(.3, 2., f)
    target = operator@basis@coefficient@loading.T
    result = eeg_representation_lower_bound(target, temporal_basis=basis,
        eeg_loadings=loading, sd=scale, eeg_operator=operator)
    np.testing.assert_allclose(result["coefficients"], coefficient, atol=2e-14)
    np.testing.assert_allclose(result["prediction"], target, atol=3e-14)
    assert result["weighted_sse"] < 1e-24
    assert result["rank"] == m*k
    assert result["solver"] == "rectangular_two_sided_svd"
    assert result["normal_equation_residual_norm"] < 2e-11
    assert result["rank_relative_tolerance"] == np.finfo(float).eps*t*f


@pytest.mark.parametrize("rectangular", [False, True])
def test_eeg_arbitrary_masks_fixed_scale_and_minimum_norm_match_full_lstsq(rectangular):
    rng = np.random.default_rng(17)
    t, f = 12, 6
    base = rng.normal(size=(t, 3))
    basis = np.column_stack((base, base[:, 1]))  # Genuine rank deficiency.
    loading = rng.normal(size=(f, 2))
    loading[:, 1] = 2*loading[:, 0]
    operator = np.eye(t)+.08*rng.normal(size=(t, t))
    scale = np.geomspace(.2, 4., f)
    mask = rng.uniform(size=(t, f)) > .3
    if rectangular:
        mask = (np.arange(t) % 3 != 0)[:, None] & (np.arange(f) % 2 == 0)[None]
    target = rng.normal(size=(t, f))
    design = _eeg_design(basis, loading, operator, scale, mask)
    expected = np.linalg.lstsq(design, (target/scale)[mask], rcond=None)
    result = eeg_representation_lower_bound(np.where(mask, target, np.nan),
        temporal_basis=basis, eeg_loadings=loading, sd=scale,
        score_mask=mask, eeg_operator=operator)
    np.testing.assert_allclose(result["coefficients"].ravel(), expected[0], rtol=1e-12, atol=1e-13)
    residual = design@expected[0]-(target/scale)[mask]
    assert result["weighted_sse"] == pytest.approx(residual@residual, rel=2e-13)
    assert result["nmse"] == pytest.approx((residual@residual)/mask.sum())
    assert result["nrmse"] == pytest.approx(np.sqrt(result["nmse"]))
    assert result["n_scored"] == mask.sum()
    assert result["rank"] == expected[2] == 3
    assert result["normal_equation_residual_norm"] < 2e-10
    np.testing.assert_allclose(result["normal_equation_residual"].ravel(),
        design.T@residual, atol=2e-10)


def test_eeg_orthogonal_residual_is_retained_and_small_real_direction_is_not_truncated():
    t, f = 5, 3
    basis = np.eye(t)[:, :2]
    basis[:, 1] *= 1e-11  # A fixed 1e-8 cutoff would invent a positive bound.
    loading = np.array([[1.], [0.], [0.]])
    target = basis@np.array([[2.], [3e11]])@loading.T
    target[3, 2] = 4.
    result = eeg_representation_lower_bound(target, temporal_basis=basis,
        eeg_loadings=loading, sd=[1., 2., 2.])
    assert result["rank"] == 2
    assert result["weighted_sse"] == pytest.approx(4.)
    assert result["prediction"][1, 0] == pytest.approx(3.)
    assert result["prediction"][3, 2] == 0.
    assert result["normal_equation_residual_norm"] < 1e-13


@pytest.mark.parametrize("masked", [False, True])
def test_eeg_certificate_retains_genuine_directions_below_machine_rank(masked):
    basis = np.array([[1., 0.], [0., 1e-18], [0., 0.], [0., 0.]])
    loading = np.array([[1.], [2.]]) if masked else np.ones((1, 1))
    explicit_coefficient = np.array([[0.], [1e18]])
    target = basis@explicit_coefficient@loading.T
    mask = np.array([[True, True], [True, False], [True, True], [False, True]]) if masked else None
    result = eeg_representation_lower_bound(target, temporal_basis=basis,
        eeg_loadings=loading, sd=np.ones(len(loading)), score_mask=mask)
    assert result["rank"] == 1
    assert result["nrmse"] > .3  # Numerical minimum-norm fit discards a real direction.
    assert result["certified_nrmse"] < 1e-14
    assert result["certificate_relaxed"]
    assert result["certificate_projection_dimension"] == 2
    assert result["discarded_direction_count"] == 1
    assert result["numerical_projection_sse_excess"] > .9
    assert result["span_residual"] < 1e-14
    assert result["orthogonality_error"] < 1e-14
    assert result["certificate_type"] == "float64_conservative_projection_not_interval_arithmetic"
    explicit_error = np.mean((basis@explicit_coefficient@loading.T-target)**2)
    assert result["certified_nmse"] <= explicit_error+1e-28


def test_eeg_well_conditioned_certificate_and_numerical_projection_agree():
    rng = np.random.default_rng(12)
    basis, loading = rng.normal(size=(13, 4)), rng.normal(size=(7, 2))
    target, scale = rng.normal(size=(13, 7)), np.geomspace(.3, 2., 7)
    for mask in (None, rng.uniform(size=target.shape) > .2):
        result = eeg_representation_lower_bound(target, temporal_basis=basis,
            eeg_loadings=loading, sd=scale, score_mask=mask)
        assert not result["certificate_relaxed"]
        assert result["certified_nmse"] == pytest.approx(result["nmse"], rel=1e-13)


@pytest.mark.parametrize("k", [1, 2])
def test_hb_matched_shared_curves_channel_specific_operator_and_gain(k):
    rng = np.random.default_rng(21+k)
    t, j = 10, 4
    loading = np.ones((j, 1)) if k == 1 else np.array(
        [[.8, .2], [.1, .9], [.6, .4], [.3, .7]])
    gains = np.array([.4, 2., .8, 1.7])
    operators = np.eye(t)[None]+.03*rng.normal(size=(j, t, t))
    latent = rng.normal(size=(t, k, 2))
    target = _hb_prediction(latent, loading, gains, operators)
    result = hb_representation_lower_bound(target, hb_loadings=loading,
        sd=np.linspace(.3, 1.4, 2*j), hb_gains=gains, hb_operator=operators)
    np.testing.assert_allclose(result["latent_hb_prediction"], latent, atol=2e-14)
    np.testing.assert_allclose(result["prediction"], target, atol=4e-14)
    assert result["weighted_sse"] < 1e-24
    assert result["rank"] == 2*t*k
    assert result["normal_equation_residual_norm"] < 1e-10
    assert result["by_chromophore"]["HbO"]["n_scored"] == t*j
    assert result["by_chromophore"]["HbR"]["n_scored"] == t*j


def test_hb_arbitrary_mask_scale_and_rank_deficiency_match_independent_design():
    rng = np.random.default_rng(32)
    t, j, k = 9, 4, 2
    loading = np.full((j, k), .5)  # No ability to separate the latent curves.
    gains = np.array([1., .4, 2., 1.5])
    common_operator = np.eye(t)-np.ones((t, t))/t
    operators = np.broadcast_to(common_operator, (j, t, t))
    scale = np.array([[.3, 1.], [2., .5], [.8, 1.5], [1.1, .2]])
    mask = rng.uniform(size=(t, j, 2)) > .25
    target = rng.normal(size=(t, j, 2))
    result = hb_representation_lower_bound(np.where(mask, target, np.inf),
        hb_loadings=loading, sd=scale, score_mask=mask, hb_gains=gains,
        hb_operator=common_operator)
    summed_sse, summed_rank = 0., 0
    for chromophore, name in enumerate(("HbO", "HbR")):
        columns = []
        for time in range(t):
            for latent_index in range(k):
                unit = np.zeros((t, k, 2))
                unit[time, latent_index, chromophore] = 1.
                response = _hb_prediction(unit, loading, gains, operators)[:, :, chromophore]
                columns.append((response/scale[:, chromophore])[mask[:, :, chromophore]])
        design = np.column_stack(columns)
        rhs = (target[:, :, chromophore]/scale[:, chromophore])[mask[:, :, chromophore]]
        solution, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
        np.testing.assert_allclose(result["latent_hb_prediction"][:, :, chromophore].ravel(),
            solution, rtol=2e-13, atol=2e-13)
        residual = design@solution-rhs
        score = result["by_chromophore"][name]
        assert score["weighted_sse"] == pytest.approx(residual@residual, rel=2e-13)
        assert score["nmse"] == pytest.approx((residual@residual)/len(rhs))
        assert result["rank_by_chromophore"][name] == rank
        summed_sse += residual@residual
        summed_rank += rank
    assert result["weighted_sse"] == pytest.approx(summed_sse)
    assert result["rank"] == summed_rank
    assert result["normal_equation_residual_norm"] < 1e-10


def test_hb_k1_bound_is_shared_across_channels_and_chromophores_remain_independent():
    target = np.zeros((6, 2, 2))
    target[:, 0, 0], target[:, 1, 0] = 1., -1.
    target[:, :, 1] = np.arange(6)[:, None]
    result = hb_representation_lower_bound(target, hb_loadings=np.ones((2, 1)), sd=np.ones((2, 2)))
    assert result["weighted_sse"] == pytest.approx(12.)
    assert result["by_chromophore"]["HbO"]["weighted_sse"] == pytest.approx(12.)
    assert result["by_chromophore"]["HbR"]["weighted_sse"] < 1e-25
    np.testing.assert_allclose(result["latent_hb_prediction"][:, 0, 0], 0., atol=1e-15)
    np.testing.assert_allclose(result["latent_hb_prediction"][:, 0, 1], np.arange(6))


def test_hb_empty_single_chromophore_does_not_read_hidden_values():
    target = np.full((5, 2, 2), np.nan)
    target[:, :, 0] = np.arange(5)[:, None]
    mask = np.zeros_like(target, dtype=bool)
    mask[:, :, 0] = True
    result = hb_representation_lower_bound(target, hb_loadings=np.ones((2, 1)),
        sd=np.ones((2, 2)), score_mask=mask)
    assert result["rank_by_chromophore"]["HbR"] == 0
    assert result["by_chromophore"]["HbR"] == dict(
        weighted_sse=0., n_scored=0, nmse=None, nrmse=None)
    np.testing.assert_array_equal(result["latent_hb_prediction"][:, :, 1], 0.)


@pytest.mark.parametrize("common_operator", [False, True])
def test_hb_certificate_retains_genuine_directions_below_machine_rank(common_operator):
    t, j = 4, 2
    operators = np.array([np.diag([1., 1e-18, 0., 0.]), np.diag([1., 2e-18, 0., 0.])])
    if common_operator:
        operators[1] = operators[0]
    latent = np.zeros((t, 1, 2))
    latent[1, 0] = [1e18, -3e18]
    target = _hb_prediction(latent, np.ones((j, 1)), np.ones(j), operators)
    mask = np.ones_like(target, dtype=bool)
    if not common_operator:
        mask[2, 1, 0] = False
    result = hb_representation_lower_bound(target, hb_loadings=np.ones((j, 1)),
        sd=np.ones((j, 2)), hb_operator=operators, score_mask=mask)
    assert result["nrmse"] > .5
    assert result["certified_nrmse"] < 1e-14
    assert result["certificate_relaxed"]
    assert result["rank_by_chromophore"] == {"HbO": 1, "HbR": 1}
    assert result["discarded_direction_count"] == 6
    assert result["certified_prediction"].shape == target.shape
    assert result["span_residual"] < 1e-14
    for name in ("HbO", "HbR"):
        assert result["certified_by_chromophore"][name]["nmse"] < 1e-28


def test_hb_common_operator_rectangle_uses_factor_space_without_arbitrary_channel_directions():
    rng = np.random.default_rng(37)
    t, j = 12, 4
    mask = np.ones((t, j, 2), bool)
    mask[:4] = False
    target = rng.normal(size=(t, j, 2))
    result = hb_representation_lower_bound(target, hb_loadings=np.ones((j, 1)),
        sd=np.geomspace(.4, 2., 2*j), score_mask=mask,
        hb_operator=np.eye(t)-np.ones((t, t))/t)
    assert result["rank"] == 2*(t-4)
    assert result["certificate_projection_dimension"] == result["rank"]
    assert not result["certificate_relaxed"]
    assert result["certified_nmse"] == pytest.approx(result["nmse"], rel=1e-13)


@pytest.mark.parametrize("k", [1, 2])
def test_bounds_cannot_exceed_random_nonlinear_predictions_with_same_observation_map(k):
    rng = np.random.default_rng(50+k)
    t, f, j = 16, 5, 4
    basis = temporal_mode_basis(t, 4)
    eeg_loading = rng.normal(size=(f, k))
    hb_loading = np.ones((j, 1)) if k == 1 else np.array(
        [[.9, .1], [.1, .9], [.7, .3], [.2, .8]])
    gains = np.linspace(.6, 1.8, j)
    eeg_operator = np.eye(t)-np.ones((t, t))/t
    hb_operator = np.eye(t)[None]+.01*rng.normal(size=(j, t, t))
    options = dict(temporal_basis=basis, eeg_loadings=eeg_loading,
        hb_loadings=hb_loading, hb_gains=gains, eeg_operator=eeg_operator,
        hb_operator=hb_operator, numerical_backend="numba")
    nonlinear = [forward_spatial_shared_driver(rng.normal(0., .003, (4, k)),
        rng.normal(0., .001, (k, 5)), BalloonParameters(), .25,
        derivative=False, **options) for _ in range(3)]
    target = nonlinear[0]["prediction"]+rng.normal(0., .001, (t, f+2*j))
    mask = rng.uniform(size=target.shape) > .2
    scale = np.geomspace(.001, .01, f+2*j)
    eeg = eeg_representation_lower_bound(target[:, :f], temporal_basis=basis,
        eeg_loadings=eeg_loading, sd=scale[:f], score_mask=mask[:, :f],
        eeg_operator=eeg_operator)
    hb = hb_representation_lower_bound(target[:, f:].reshape(t, j, 2),
        hb_loadings=hb_loading, sd=scale[f:],
        score_mask=mask[:, f:].reshape(t, j, 2), hb_gains=gains, hb_operator=hb_operator)
    for candidate in nonlinear:
        weighted_error = ((candidate["prediction"]-target)/scale)**2
        assert eeg["weighted_sse"] <= weighted_error[:, :f][mask[:, :f]].sum()+1e-10
        assert hb["weighted_sse"] <= weighted_error[:, f:][mask[:, f:]].sum()+1e-10


@pytest.mark.parametrize("change", [
    {"temporal_basis": np.ones(4)}, {"eeg_loadings": [[np.nan]]},
    {"sd": [0., 1.]}, {"sd": [1.]}, {"score_mask": np.ones((4, 2))},
    {"score_mask": np.zeros((4, 2), bool)}, {"eeg_operator": np.eye(3)},
    {"eeg_operator": np.full((4, 4), np.inf)},
])
def test_eeg_rejects_malformed_inputs(change):
    options = dict(temporal_basis=np.ones((4, 1)), eeg_loadings=np.ones((2, 1)), sd=np.ones(2))
    options.update(change)
    with pytest.raises(ValueError):
        eeg_representation_lower_bound(np.ones((4, 2)), **options)


@pytest.mark.parametrize("change", [
    {"hb_loadings": [[-.1, 1.1], [.2, .8]]}, {"hb_loadings": np.ones((2, 2))},
    {"hb_loadings": [[1., 0.], [1., 0.]]}, {"hb_gains": [0., 1.]},
    {"hb_gains": [1.]}, {"hb_operator": np.eye(3)},
    {"hb_operator": np.full((2, 4, 4), np.nan)}, {"sd": np.ones(2)},
    {"sd": np.array([[1., np.inf], [1., 1.]])},
    {"score_mask": np.ones((4, 2, 2))}, {"score_mask": np.zeros((4, 2, 2), bool)},
])
def test_hb_rejects_malformed_inputs(change):
    options = dict(hb_loadings=np.ones((2, 1)), sd=np.ones((2, 2)))
    options.update(change)
    with pytest.raises(ValueError):
        hb_representation_lower_bound(np.ones((4, 2, 2)), **options)


def _frontier_fixture(t=48, modes=4):
    basis = temporal_mode_basis(t, modes)
    coefficient = np.array([.007, .018, -.011, .008, -.005, .01])[:modes, None]
    initial = np.array([[.008, .005, -.003, .002, -.001]])
    parameters = BalloonParameters(free=BalloonFreeParameters(tau=1.3, kappa=.45))
    design = dict(temporal_basis=basis, eeg_loadings=np.array([[1.], [-.7]]),
        hb_loadings=np.ones((2, 1)), hb_gains=np.array([.8, 1.4]), numerical_backend="numba")
    truth = forward_spatial_shared_driver(coefficient, initial, parameters, .25,
        substeps=8, derivative=False, **design)
    options = dict(**design, sd=np.array([.025, .025, .01, .005, .01, .005]))
    return coefficient, initial, parameters, truth, options


def _frontier_start(result, label):
    return {key: result[key] for key in ("coefficients", "initial_coordinates", "tau", "kappa")} | {"label": label}


def test_frontier_recovers_noiseless_known_driver_from_eeg_only_initialization():
    coefficient, initial, _, truth, options = _frontier_fixture(t=96, modes=6)
    wrong_parameters = BalloonParameters(free=BalloonFreeParameters(tau=2.3, kappa=.8))
    result = fit_feasibility_frontier(truth["eeg_prediction"], truth["hb_prediction"],
        wrong_parameters, .25, epsilon=.001, free_physiology=True, max_nfev=200, **options)
    assert result["witness_feasible"] and result["physical_valid"]
    assert result["optimizer_converged"]
    assert result["eeg_nrmse"] <= .001+1e-8
    assert result["hb_nmse"] < 2e-8
    assert result["tau"] == pytest.approx(1.3, rel=.005)
    assert result["kappa"] == pytest.approx(.45, rel=.005)
    assert result["integration_error"] < .005
    assert result["source_start_label"] == "eeg_least_squares_rest"
    assert result["starts"][0]["finite_difference_calls"] > 0
    assert np.linalg.norm(result["initial_coordinates"]) > 1e-4
    # All truth information above is used to generate/score, never as a start.
    assert not np.array_equal(initial, np.zeros_like(initial))
    assert np.linalg.norm(result["coefficients"]-coefficient) < 1e-4


def test_frontier_budget_failure_preserves_fixed_and_wider_epsilon_witnesses():
    _, _, parameters, truth, options = _frontier_fixture()
    narrow = fit_feasibility_frontier(truth["eeg_prediction"], truth["hb_prediction"],
        parameters, .25, epsilon=.01, max_nfev=1, **options)
    assert narrow["witness_feasible"]
    assert not narrow["optimizer_converged"]
    assert narrow["starts"][0]["optimizer_message"] == "differentiated_evaluation_budget"
    assert narrow["evaluations"] == 1
    assert narrow["cache_hits"] >= 2
    for free in (False, True):
        inherited = fit_feasibility_frontier(truth["eeg_prediction"], truth["hb_prediction"],
            parameters, .25, epsilon=.1, starts=[_frontier_start(narrow, "inherited_narrow_fixed")],
            max_nfev=1, free_physiology=free, **options)
        assert inherited["witness_feasible"]
        assert inherited["hb_nmse"] <= narrow["hb_nmse"]+1e-15
        assert inherited["source_start_label"] == "inherited_narrow_fixed"
        np.testing.assert_array_equal(inherited["prediction"], narrow["prediction"])
        assert inherited["optimization_scores"]["Hb"]["nmse"] == pytest.approx(
            narrow["optimization_scores"]["Hb"]["nmse"])


def test_frontier_k2_retains_independent_nontruth_initial_coordinates():
    rng = np.random.default_rng(91)
    t, f, j, k = 24, 3, 3, 2
    parameters = BalloonParameters()
    design = dict(temporal_basis=temporal_mode_basis(t, 4),
        eeg_loadings=np.array([[1., .2], [-.6, .7], [.3, -.5]]),
        hb_loadings=np.array([[.8, .2], [.3, .7], [.1, .9]]), numerical_backend="numba")
    truth = forward_spatial_shared_driver(rng.normal(0., .004, (4, k)),
        rng.normal(0., .001, (k, 5)), parameters, .25, derivative=False, substeps=8, **design)
    scale = np.r_[np.full(f, .03), np.tile([.01, .005], j)]
    initial = np.array([[.001, -.002, .003, -.001, .002],
                        [-.002, .003, -.001, .002, -.001]])
    projection = eeg_representation_lower_bound(truth["eeg_prediction"],
        temporal_basis=design["temporal_basis"], eeg_loadings=design["eeg_loadings"], sd=scale[:f])
    result = fit_feasibility_frontier(truth["eeg_prediction"], truth["hb_prediction"],
        parameters, .25, epsilon=.01, sd=scale, max_nfev=1,
        starts=[dict(label="independent_nontruth_initials", coefficients=projection["coefficients"],
                     initial_coordinates=initial)], **design)
    assert result["witness_feasible"]
    assert result["states"].shape == (t, k, 6)
    assert result["drivers"].shape == (t, k)
    np.testing.assert_array_equal(result["initial_coordinates"], initial)
    assert not np.array_equal(result["initial_coordinates"][0], result["initial_coordinates"][1])


def test_frontier_hard_constraint_and_gradients_use_only_scored_fixed_scale_targets(monkeypatch):
    from types import SimpleNamespace
    import scipy.optimize

    _, _, parameters, truth, options = _frontier_fixture(t=24)
    rng = np.random.default_rng(73)
    target = truth["prediction"]+rng.normal(0., .0001, truth["prediction"].shape)
    mask = rng.uniform(size=target.shape) > .2
    mask[:6] = False
    target[~mask] = np.nan
    checked = []

    def inspect_minimize(fun, x, *, jac, constraints, **kwargs):
        objective, gradient = fun(x), jac(x)
        constraint = constraints[0]
        constraint_gradient = constraint["jac"](x)
        for column in (0, 4, 7, 9, 10):
            step = np.zeros_like(x)
            step[column] = 1e-6
            numerical = (fun(x+step)-fun(x-step))/(2e-6)
            assert gradient[column] == pytest.approx(numerical, rel=2e-5, abs=2e-7)
            numerical_constraint = (constraint["fun"](x+step)-constraint["fun"](x-step))/(2e-6)
            assert constraint_gradient[column] == pytest.approx(numerical_constraint, abs=1e-8)
        assert objective > 0
        checked.append(True)
        return SimpleNamespace(success=False, status=9, message="gradient_test_stop")

    monkeypatch.setattr(scipy.optimize, "minimize", inspect_minimize)
    result = fit_feasibility_frontier(target[:, :2], target[:, 2:].reshape(24, 2, 2),
        parameters, .25, epsilon=.2, free_physiology=True, score_mask=mask,
        max_nfev=50, **options)
    assert checked and result["physical_valid"]
    actual = ((result["prediction"]-target)/options["sd"])**2
    assert result["hb_nmse"] == pytest.approx(actual[:, 2:][mask[:, 2:]].mean())
    assert result["eeg_nmse"] == pytest.approx(actual[:, :2][mask[:, :2]].mean())
    assert result["scores"]["HbO"]["n_scored"] == mask[:, 2::2].sum()
    assert result["scores"]["HbR"]["n_scored"] == mask[:, 3::2].sum()


def test_frontier_rejects_invalid_state_trials_and_retains_legal_start():
    _, _, parameters, truth, options = _frontier_fixture(t=24)
    bound = eeg_representation_lower_bound(truth["eeg_prediction"],
        temporal_basis=options["temporal_basis"], eeg_loadings=options["eeg_loadings"],
        sd=options["sd"][:2])
    invalid_initial = np.zeros((1, 5))
    invalid_initial[0, 3] = -20.  # Absolute HbT below HbR; no clipping is valid.
    starts = [dict(label="invalid_hb_balance", coefficients=bound["coefficients"],
                   initial_coordinates=invalid_initial),
              dict(label="legal_rest", coefficients=bound["coefficients"],
                   initial_coordinates=np.zeros((1, 5)))]
    result = fit_feasibility_frontier(truth["eeg_prediction"], truth["hb_prediction"],
        parameters, .25, epsilon=.1, starts=starts, max_nfev=1, **options)
    assert result["witness_feasible"]
    assert result["source_start_label"] == "legal_rest"
    assert result["invalid_trials"] == 1
    assert result["starts"][0]["best_candidate"] is None
    assert result["failure_log"][0]["start"] == "invalid_hb_balance"
    np.testing.assert_array_equal(starts[0]["initial_coordinates"], invalid_initial)


def test_frontier_fine_integration_requirement_is_independent_of_optimizer_status():
    _, _, parameters, truth, options = _frontier_fixture(t=24)
    result = fit_feasibility_frontier(truth["eeg_prediction"], truth["hb_prediction"],
        parameters, .25, epsilon=.1, max_nfev=1, substeps=1,
        validation_tolerance=0., **options)
    assert result["physical_valid"]
    assert result["eeg_nrmse"] < .1
    assert result["integration_error"] > 0.
    assert not result["witness_feasible"]
    assert result["status"] == "no_feasible_witness"


def test_frontier_retains_joint_witness_when_best_mean_hb_candidate_fails_one_chromophore():
    t = 8
    parameters = BalloonParameters()
    # At t=0 these starts predict standardized (HbO,HbR) errors (.4,.4)
    # and (.55,0). The latter has the smaller mean squared Hb error, while
    # only the former meets both .5 diagnostic thresholds.
    balanced = np.zeros((1, 5))
    balanced[0, 3] = np.log1p(.008/parameters.fixed.P0)
    balanced[0, 4] = np.log1p(.004/parameters.fixed.Q0)
    lower_mean = np.zeros((1, 5))
    lower_mean[0, 3] = np.log1p(.0055/parameters.fixed.P0)
    mask = np.zeros((t, 3), bool)
    mask[:, 0] = True
    mask[0, 1:] = True
    result = fit_feasibility_frontier(np.zeros((t, 1)), np.zeros((t, 1, 2)),
        parameters, .25, temporal_basis=temporal_mode_basis(t, 2),
        eeg_loadings=np.ones((1, 1)), hb_loadings=np.ones((1, 1)),
        sd=[1., .01, .01], epsilon=.1, score_mask=mask, max_nfev=1,
        starts=[dict(label="balanced_joint", coefficients=np.zeros((2, 1)), initial_coordinates=balanced),
                dict(label="lower_mean_hb", coefficients=np.zeros((2, 1)), initial_coordinates=lower_mean)])
    assert result["source_start_label"] == "lower_mean_hb"
    assert result["hbo_nrmse"] == pytest.approx(.55)
    assert result["hbr_nrmse"] < 1e-12
    assert result["joint_witness"]["source_start_label"] == "balanced_joint"
    assert result["joint_witness"]["hbo_nrmse"] == pytest.approx(.4)
    assert result["joint_witness"]["hbr_nrmse"] == pytest.approx(.4)
    assert result["hb_nmse"] < result["joint_witness"]["hb_nmse"]


def test_frontier_projection_exclusion_never_calls_nonlinear_solver(monkeypatch):
    import src.inference.source_observability as module

    def forbidden(*args, **kwargs):
        raise AssertionError("excluded representation must not run nonlinear search")

    monkeypatch.setattr(module, "forward_spatial_shared_driver", forbidden)
    eeg = np.column_stack((np.zeros(8), np.ones(8)))
    result = fit_feasibility_frontier(eeg, np.zeros((8, 1, 2)), BalloonParameters(), .25,
        temporal_basis=temporal_mode_basis(8, 2), eeg_loadings=np.array([[1.], [0.]]),
        hb_loadings=np.ones((1, 1)), sd=np.ones(4), epsilon=.1)
    assert result["status"] == "excluded_by_eeg_representation"
    assert result["eeg_representation_lower_bound"] == pytest.approx(np.sqrt(.5))
    assert result["evaluations"] == result["forward_calls"] == 0
    assert not result["witness_feasible"]


def test_frontier_does_not_exclude_a_machine_truncated_eeg_direction():
    t = 4
    target = np.array([[0.], [1.], [0.], [0.]])
    result = fit_feasibility_frontier(target, np.zeros((t, 1, 2)), BalloonParameters(), .25,
        temporal_basis=np.eye(t)[:, :2], eeg_loadings=np.ones((1, 1)),
        eeg_operator=np.diag([1., 1e-18, 1., 1.]), hb_loadings=np.ones((1, 1)),
        sd=np.ones(3), epsilon=.1, max_nfev=1)
    assert result["eeg_numerical_projection_nrmse"] == pytest.approx(.5)
    assert result["eeg_representation_lower_bound"] == 0.
    assert result["eeg_representation_lower_bound_relaxed"]
    assert result["status"] != "excluded_by_eeg_representation"
    assert result["evaluations"] == 1


@pytest.mark.parametrize("change", [
    {"epsilon": -1.}, {"max_nfev": 0}, {"verification_substeps": 4},
    {"validation_tolerance": -1.}, {"constraint_tolerance": np.inf},
    {"free_physiology": "yes"}, {"tau_bounds": (4., .6)},
    {"starts": []}, {"starts": [{"coefficients": np.zeros((4, 1))}]},
])
def test_frontier_rejects_malformed_search_controls(change):
    _, _, parameters, truth, options = _frontier_fixture(t=12)
    options = dict(options, epsilon=.1, max_nfev=1)
    options.update(change)
    with pytest.raises(ValueError):
        fit_feasibility_frontier(truth["eeg_prediction"], truth["hb_prediction"], parameters, .25, **options)
