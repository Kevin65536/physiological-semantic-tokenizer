import numpy as np
import pytest

from src.inference.t3a_balloon_robust_ssm import (
    BalloonConfig,
    BalloonFixedParameters,
    BalloonFreeParameters,
    BalloonObservationSpec,
    BalloonParameters,
    STATE_NAMES,
    balloon_rhs,
    balloon_rhs_jacobian,
    fit_balloon,
    observation_jacobian,
    observation_map,
    physical_to_transformed,
    rk4_transition,
    run_physical_checks,
    simulate_balloon,
    smooth_balloon,
    student_t_irls_weights,
    transformed_gaussian_moments,
    transformed_to_physical,
)


def _parameters() -> BalloonParameters:
    return BalloonParameters(
        fixed=BalloonFixedParameters(
            P0=1.0,
            Q0=0.4,
            process_std=(0.10, 0.05, 0.015, 0.015, 0.015, 0.015),
            observation_scale=(0.04, 0.04, 0.04),
        ),
        free=BalloonFreeParameters(kappa=0.65, tau=2.0),
    )


def _config(**kwargs: object) -> BalloonConfig:
    values = dict(rk4_substeps=2, irls_iterations=2, initial_state_std=(0.4,) * 6)
    values.update(kwargs)
    return BalloonConfig(**values)


def _driver(length: int = 100) -> np.ndarray:
    rng = np.random.default_rng(12)
    values = np.zeros(length, dtype=np.float64)
    for index in range(1, length):
        pulse = 0.05 * np.sin(2.0 * np.pi * index / 36.0)
        values[index] = 0.96 * values[index - 1] + pulse + 0.025 * rng.normal()
    return values


def test_rest_equilibrium_and_jacobian_are_exact_enough():
    parameters = _parameters()
    rest = physical_to_transformed((0.0, 0.0, 1.0, 1.0, 1.0, 1.0))
    np.testing.assert_allclose(balloon_rhs(rest, parameters), 0.0, atol=1e-12)
    np.testing.assert_allclose(rk4_transition(rest, parameters, _config()), rest, atol=1e-12)

    point = physical_to_transformed((0.2, -0.1, 1.1, 1.04, 1.02, 0.97))
    analytic = balloon_rhs_jacobian(point, parameters)
    numeric = np.empty_like(analytic)
    for index in range(len(point)):
        delta = np.zeros_like(point)
        delta[index] = 1e-6
        numeric[:, index] = (
            balloon_rhs(point + delta, parameters) - balloon_rhs(point - delta, parameters)
        ) / (2.0e-6)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-7)


def test_tak_p_balance_uses_minimal_tau_v_zero_equation():
    parameters = _parameters()
    physical = np.asarray((0.0, 0.0, 1.2, 1.1, 1.3, 0.9), dtype=np.float64)
    transformed = physical_to_transformed(physical)
    rhs = balloon_rhs(transformed, parameters)
    f, v, p, tau, alpha = physical[2], physical[3], physical[4], parameters.free.tau, parameters.fixed.alpha
    f_out = v ** (1.0 / alpha)
    expected_d_p = (f - f_out * p / v) / tau
    alternative_d_p = (f - f_out) * p / (tau * v)
    np.testing.assert_allclose(rhs[4] * p, expected_d_p, rtol=1e-12, atol=1e-12)
    assert not np.isclose(expected_d_p, alternative_d_p)


def test_neurovascular_gain_scales_driver_coupling_and_is_positive():
    parameters = BalloonParameters(
        fixed=BalloonFixedParameters(neurovascular_gain=1.7),
        free=BalloonFreeParameters(kappa=0.65, tau=2.0),
    )
    state = physical_to_transformed((0.2, 0.0, 1.0, 1.0, 1.0, 1.0))
    rhs = balloon_rhs(state, parameters)
    np.testing.assert_allclose(rhs[1], 1.7 * 0.2)
    with pytest.raises(ValueError, match="neurovascular_gain"):
        BalloonFixedParameters(neurovascular_gain=0.0).validate()


def test_observation_map_is_explicit_hbt_hbr_hbo_balance():
    parameters = _parameters()
    state = np.asarray((0.2, 0.0, 1.0, 1.0, 1.2, 0.8), dtype=np.float64)
    spec = BalloonObservationSpec().resolved(parameters.fixed)
    observed = observation_map(state, parameters, spec)
    expected_hbt = parameters.fixed.P0 * state[4]
    expected_hbr = parameters.fixed.Q0 * state[5]
    np.testing.assert_allclose(observed, (state[0], expected_hbt - expected_hbr - (parameters.fixed.P0 - parameters.fixed.Q0), expected_hbr - parameters.fixed.Q0))
    # The map is expressed in deltas around p=q=1, while the physical check
    # uses absolute P0*p and Q0*q values from the same single baseline.
    checks = run_physical_checks(state[None, :], parameters)
    assert checks["absolute_hb_nonnegative"]
    assert checks["hbr_not_above_hbt"]
    jacobian = observation_jacobian(physical_to_transformed(state), parameters, spec)
    assert jacobian.shape == (3, 6)


def test_fixed_eeg_sign_gauge_rejects_nonpositive_loading():
    try:
        BalloonFixedParameters(eeg_loading=0.0).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("nonpositive EEG loading must violate the fixed sign gauge")
    try:
        BalloonObservationSpec(eeg_loading=-1.0).resolved(BalloonFixedParameters())
    except ValueError:
        pass
    else:
        raise AssertionError("negative observation loading must violate the fixed sign gauge")


def test_simulation_known_coordinate_transform_also_transforms_noise_realization():
    parameters = _parameters()
    spec = BalloonObservationSpec().resolved(parameters.fixed)
    scale = np.array([2., -.5, 3.])
    original = simulate_balloon(_driver(12), parameters, config=_config(), observation_spec=spec,
                                rng=np.random.default_rng(39))
    changed = simulate_balloon(_driver(12), parameters, config=_config(), observation_spec=spec.reexpress(scale),
                               rng=np.random.default_rng(39))
    np.testing.assert_array_equal(changed.states, original.states)
    np.testing.assert_allclose(changed.observations, original.observations*scale, atol=1e-15)


def test_transformed_gaussian_moments_use_exact_lognormal_formulae():
    transformed_mean = np.asarray((0.2, -0.1, np.log(1.1), np.log(0.9), np.log(1.2), np.log(0.8)))
    covariance = np.diag((0.04, 0.01, 0.09, 0.04, 0.16, 0.25))
    covariance[0, 2] = covariance[2, 0] = 0.01
    covariance[1, 4] = covariance[4, 1] = -0.005
    mean, physical_covariance = transformed_gaussian_moments(transformed_mean, covariance)
    expected_f = np.exp(transformed_mean[2] + 0.5 * covariance[2, 2])
    expected_q = np.exp(transformed_mean[5] + 0.5 * covariance[5, 5])
    np.testing.assert_allclose(mean[[0, 1]], transformed_mean[[0, 1]])
    np.testing.assert_allclose(mean[2], expected_f)
    np.testing.assert_allclose(mean[5], expected_q)
    np.testing.assert_allclose(
        physical_covariance[2, 2], expected_f**2 * np.expm1(covariance[2, 2])
    )
    np.testing.assert_allclose(
        physical_covariance[0, 2], covariance[0, 2] * expected_f
    )
    np.testing.assert_allclose(
        physical_covariance[2, 5], mean[2] * mean[5] * np.expm1(covariance[2, 5])
    )


def test_transformed_underflow_and_extraction_domain_fail_closed():
    with pytest.raises(FloatingPointError):
        transformed_to_physical(np.asarray((0.0, 0.0, -1000.0, 0.0, 0.0, 0.0)))
    tiny_f = physical_to_transformed((0.0, 0.0, 1.0e-320, 1.0, 1.0, 1.0))
    with pytest.raises(FloatingPointError):
        balloon_rhs(tiny_f, _parameters())


@pytest.mark.parametrize('flow', [.003209156, .0001, .01, .2, 1., 3., 100., 1e12])
def test_extraction_matches_high_precision_domain_and_derivatives(flow):
    from decimal import Decimal, localcontext
    from src.inference import t3a_balloon_robust_ssm as core
    with localcontext() as ctx:
        ctx.prec = 100
        f, e0 = Decimal(str(flow)), Decimal('.32')
        log_base = (1-e0).ln()
        a = log_base/f
        complement = a.exp()
        extraction = 1-complement
        derivative = complement*log_base/(f*f)
        flux = f*extraction/e0
        flux_derivative = f*(extraction+f*derivative)/e0
        expected = list(map(float, (extraction, derivative, flux)))
    actual = core._extraction(flow, .32)
    np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=0)
    assert core.extraction_log_complement(flow, .32) == pytest.approx(float(a), rel=1e-14)
    assert core._flow_extraction_log_derivative(flow, .32, *actual[:2]) == pytest.approx(
        float(flux_derivative), rel=5e-12, abs=1e-300)
    if flow == .003209156:
        assert actual[0] == 1.0
        assert float(complement) > 0
        states = np.array([[0, 0, flow, 1, 1, 1.]])
        checks = core.run_physical_checks(states, _parameters())
        assert checks['oxygen_extraction_in_unit_interval']
        assert checks['extraction_float_saturation_count'] == 1
        assert checks['flow_below_0_01_count'] == 1


@pytest.mark.parametrize('flow', [0., -1., np.nan, np.inf, -np.inf])
def test_extraction_rejects_illegal_flow(flow):
    from src.inference import t3a_balloon_robust_ssm as core
    with pytest.raises(ValueError, match='extraction requires'):
        core._extraction(flow, .32)


def test_low_flow_rhs_jacobian_remains_correct_after_saturation():
    point = physical_to_transformed((.02, -.001, .003209156, .2, .3, .4))
    parameters = _parameters()
    numeric = np.column_stack([
        (balloon_rhs(point+delta, parameters)-balloon_rhs(point-delta, parameters))/2e-6
        for delta in np.eye(6)*1e-6])
    np.testing.assert_allclose(balloon_rhs_jacobian(point, parameters), numeric, rtol=1e-6, atol=1e-7)


def test_simulation_preserves_positive_compartments_and_extraction_domain():
    parameters = _parameters()
    simulation = simulate_balloon(_driver(), parameters, config=_config(), add_noise=False)
    assert simulation.states.shape == (100, 6)
    assert np.all(simulation.states[:, 2:] > 0.0)
    checks = run_physical_checks(simulation.states, parameters)
    assert checks["finite"]
    assert checks["positive_fvpq"]
    assert checks["oxygen_extraction_in_unit_interval"]
    assert checks["rest_equilibrium"]


def test_student_t_irls_downweights_outlier():
    weights = student_t_irls_weights(np.asarray([0.0, 10.0]), np.ones(2), nu=5.0)
    assert weights[0] > 1.0
    assert weights[1] < 0.1


def test_missing_aware_smoother_keeps_finite_state_and_increases_uncertainty():
    parameters = _parameters()
    config = _config()
    simulation = simulate_balloon(_driver(80), parameters, config=config, add_noise=False)
    full = smooth_balloon(simulation.observations, parameters, config=config)
    missing_mask = np.zeros_like(simulation.observation_mask, dtype=bool)
    missing = smooth_balloon(
        simulation.observations,
        parameters,
        config=config,
        observation_mask=missing_mask,
    )
    assert np.all(np.isfinite(missing.state_mean))
    assert np.all(np.isfinite(missing.total_variance))
    assert not missing.trajectory_valid_mask.any()
    assert np.all(missing.total_variance.mean(axis=0) > full.total_variance.mean(axis=0))
    assert np.all(missing.observation_residual_valid_mask == missing_mask)


def test_known_truth_driver_recovery_and_variance_identity():
    parameters = _parameters()
    config = _config()
    simulation = simulate_balloon(_driver(120), parameters, config=config, add_noise=False)
    result = smooth_balloon(simulation.observations, parameters, config=config)
    assert result.state_names == STATE_NAMES
    np.testing.assert_allclose(
        result.observation_residual,
        simulation.observations - result.observation_mean,
    )
    assert np.isfinite(result.predictive_log_likelihood)
    assert np.corrcoef(simulation.states[:, 0], result.state_mean[:, 0])[0, 1] > 0.9
    assert np.corrcoef(simulation.states[:, 4], result.state_mean[:, 4])[0, 1] > 0.85
    np.testing.assert_allclose(
        result.total_variance,
        result.aleatoric_variance + result.epistemic_variance,
        rtol=1e-12,
        atol=1e-12,
    )
    assert result.uncertainty_method.startswith("Student-t IRLS")


def test_fit_only_changes_kappa_and_tau_and_records_multistart_estimates():
    parameters = _parameters()
    config = _config(
        optimizer_max_iterations=4,
        optimizer_starts=((0.4, 1.2), (1.0, 3.0)),
        hessian_step=2e-3,
    )
    simulation = simulate_balloon(_driver(45), parameters, config=config, add_noise=False)
    fit = fit_balloon(
        simulation.observations,
        fixed=parameters.fixed,
        config=config,
    )
    assert len(fit.starts) == 2
    assert all("estimate" in record for record in fit.starts)
    assert fit.parameters.fixed == parameters.fixed
    assert config.kappa_bounds[0] <= fit.parameters.free.kappa <= config.kappa_bounds[1]
    assert config.tau_bounds[0] <= fit.parameters.free.tau <= config.tau_bounds[1]
    assert fit.hessian.shape == (2, 2)
    assert fit.likelihood_hessian.shape == (2, 2)
    assert fit.parameter_covariance.shape == (2, 2)


def test_fit_rejects_all_missing_data_before_prior_can_create_identifiability():
    parameters = _parameters()
    config = _config(optimizer_max_iterations=2)
    simulation = simulate_balloon(_driver(20), parameters, config=config, add_noise=False)
    missing_mask = np.zeros_like(simulation.observation_mask, dtype=bool)
    with pytest.raises(ValueError, match="at least two finite observations"):
        fit_balloon(
            simulation.observations,
            fixed=parameters.fixed,
            config=config,
            observation_mask=missing_mask,
        )
