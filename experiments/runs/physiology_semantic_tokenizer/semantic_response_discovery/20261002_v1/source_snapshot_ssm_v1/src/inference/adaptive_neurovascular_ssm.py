"""Adaptive, physiology-constrained EEG-fNIRS fixed-interval smoother.

The legacy exploratory filter in :mod:`src.inference.neurovascular_smc` uses a
causal scalar AR state followed by a fixed HRF.  That is useful as a baseline,
but a present fNIRS sample cannot revise an already emitted neural state when
the HRF has a several-second delay.  This module instead linearizes Croce's
five-state neurovascular equations around their resting point and applies an
RTS fixed-interval smoother.  The complete window can therefore negotiate a
single neural driver between instantaneous EEG evidence and delayed HbO/HbR
evidence.

The physiological parameters are bounded and fitted on training trials only.
They remain a constrained model family rather than unconstrained waveform
regression.  Observation-noise balance is selected on the training trials by
an explicit equal-modality reconstruction objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.linalg import expm, solve_discrete_lyapunov
from scipy.optimize import minimize


@dataclass(frozen=True)
class HemodynamicParameters:
    """Parameters of the linearized Croce/Balloon hemodynamic dynamics."""

    epsilon: float = 1.0
    kas: float = 0.65
    kaf: float = 0.41
    tau0: float = 2.0
    alpha: float = 0.32
    e0: float = 0.40


@dataclass(frozen=True)
class AdaptiveSSMFit:
    """Training-fold parameters needed to apply the adaptive smoother."""

    params: HemodynamicParameters
    transition: np.ndarray
    process_cov: np.ndarray
    observation: np.ndarray
    observation_cov: np.ndarray
    initial_cov: np.ndarray
    hbo_mean: float
    hbo_std: float
    hbr_mean: float
    hbr_std: float
    baseline_samples: int
    phi: float
    q_driver: float
    q_scale: float
    fnirs_noise_scale: float
    hbo_gain: float
    hbr_gain: float
    eeg_noise: float
    hbo_noise_base: float
    hbr_noise_base: float
    training_score: float
    optimizer_success: bool
    optimizer_objective: float


@dataclass(frozen=True)
class AdaptiveSmootherResult:
    """Posterior trajectories and reconstructed observations."""

    states: np.ndarray
    state_std: np.ndarray
    observation_predictive_std: np.ndarray
    eeg_reconstructed: np.ndarray
    hbo_reconstructed: np.ndarray
    hbr_reconstructed: np.ndarray
    predictive_log_likelihood: float
    observation_mode: str = "unknown"


@dataclass(frozen=True)
class AdaptiveStateGaugeResult:
    """Fold-aligned target coordinates and their transformed uncertainty.

    The dynamical state remains available in :class:`AdaptiveSmootherResult`.
    This view only maps the two chromophore coordinates through observation
    gains and train-fold scales into the canonical measurement space used by
    the tokenizer input.  It is therefore an observation-aligned target gauge,
    not a claim that the independently scaled coordinates remain a physical
    five-state trajectory.
    """

    states: np.ndarray
    state_std: np.ndarray
    scales: np.ndarray
    offsets: np.ndarray
    reconstruction_max_abs_delta: float


def _flow_extraction_slope(e0: float) -> float:
    """Derivative of f*E(f)/E0 at resting flow f=1."""

    e0 = float(np.clip(e0, 1e-4, 1.0 - 1e-4))
    one_minus = 1.0 - e0
    return float(1.0 + one_minus * np.log(one_minus) / e0)


def continuous_hemodynamic_matrices(
    params: HemodynamicParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the four-state linearized hemodynamic drift and neural input."""

    tau0 = max(float(params.tau0), 1e-6)
    alpha = max(float(params.alpha), 1e-6)
    flow_slope = _flow_extraction_slope(params.e0)
    drift = np.asarray([
        [-params.kas, -params.kaf, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0 / tau0, -1.0 / (alpha * tau0), 0.0],
        [0.0, flow_slope / tau0, -(1.0 / alpha - 1.0) / tau0, -1.0 / tau0],
    ], dtype=np.float64)
    neural_input = np.asarray([params.epsilon, 0.0, 0.0, 0.0], dtype=np.float64)
    return drift, neural_input


def discretize_hemodynamics(
    params: HemodynamicParameters,
    fs_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exactly discretize the linearized four-state ODE at ``fs_hz``."""

    drift, neural_input = continuous_hemodynamic_matrices(params)
    augmented = np.zeros((5, 5), dtype=np.float64)
    augmented[:4, :4] = drift
    augmented[:4, 4] = neural_input
    discrete = expm(augmented / float(fs_hz))
    return discrete[:4, :4], discrete[:4, 4]


def simulate_hemodynamics(
    driver: np.ndarray,
    params: HemodynamicParameters,
    fs_hz: float = 10.0,
    initial_state: np.ndarray | None = None,
) -> np.ndarray:
    """Forward-simulate ``[s, delta_f, delta_HbO, delta_HbR]`` from a driver."""

    values = np.asarray(driver, dtype=np.float64).reshape(-1)
    transition, neural_input = discretize_hemodynamics(params, fs_hz)
    state = np.zeros(4, dtype=np.float64) if initial_state is None else np.asarray(initial_state, dtype=np.float64).copy()
    output = np.zeros((len(values), 4), dtype=np.float64)
    for index, value in enumerate(values):
        state = transition @ state + neural_input * value
        output[index] = state
    return output


def _amplitude_calibrated_gain(source: np.ndarray, target: np.ndarray) -> float:
    source = np.asarray(source, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    source_energy = max(float(np.dot(source, source)), 1e-12)
    target_energy = max(float(np.dot(target, target)), 1e-12)
    covariance = float(np.dot(source, target))
    sign = -1.0 if covariance < 0.0 else 1.0
    return sign * np.sqrt(target_energy / source_energy)


def _parameter_vector(params: HemodynamicParameters) -> np.ndarray:
    return np.asarray([params.kas, params.kaf, params.tau0, params.alpha, params.e0], dtype=np.float64)


def _parameters_from_vector(values: Sequence[float]) -> HemodynamicParameters:
    kas, kaf, tau0, alpha, e0 = [float(value) for value in values]
    return HemodynamicParameters(kas=kas, kaf=kaf, tau0=tau0, alpha=alpha, e0=e0)


def fit_hemodynamic_parameters(
    drivers: Sequence[np.ndarray],
    hbo_targets: Sequence[np.ndarray],
    hbr_targets: Sequence[np.ndarray],
    *,
    fs_hz: float = 10.0,
    prior_strength: float = 0.03,
    max_iterations: int = 80,
    max_flow_perturbation: float = 0.25,
) -> tuple[HemodynamicParameters, float, float, float, bool]:
    """Fit bounded Croce dynamics and measurement gains on training trials."""

    if not drivers or len(drivers) != len(hbo_targets) or len(drivers) != len(hbr_targets):
        raise ValueError("drivers, HbO targets, and HbR targets must have equal non-zero length")
    canonical = HemodynamicParameters()
    center = _parameter_vector(canonical)
    bounds = np.asarray([
        [0.25, 1.50],  # signal decay
        [0.05, 0.90],  # flow-dependent feedback
        [0.60, 5.00],  # mean transit time
        [0.18, 0.55],  # Grubb exponent
        [0.20, 0.65],  # resting extraction
    ], dtype=np.float64)
    half_ranges = np.maximum((bounds[:, 1] - bounds[:, 0]) * 0.5, 1e-8)
    hbo_concat = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in hbo_targets])
    hbr_concat = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in hbr_targets])

    def evaluate(values: np.ndarray) -> tuple[float, float, float]:
        params = _parameters_from_vector(values)
        states = [simulate_hemodynamics(driver, params, fs_hz) for driver in drivers]
        hbo_state = np.concatenate([state[:, 2] for state in states])
        hbr_state = np.concatenate([state[:, 3] for state in states])
        hbo_gain = _amplitude_calibrated_gain(hbo_state, hbo_concat)
        hbr_gain = _amplitude_calibrated_gain(hbr_state, hbr_concat)
        residual = np.concatenate((hbo_gain * hbo_state - hbo_concat, hbr_gain * hbr_state - hbr_concat))
        mse = float(np.mean(residual**2))
        prior = float(np.mean(((values - center) / half_ranges) ** 2))
        return mse + float(prior_strength) * prior, hbo_gain, hbr_gain

    result = minimize(
        lambda values: evaluate(np.asarray(values, dtype=np.float64))[0],
        center,
        method="L-BFGS-B",
        bounds=[tuple(value) for value in bounds],
        options={"maxiter": int(max_iterations), "ftol": 1e-9},
    )
    values = np.asarray(result.x if np.all(np.isfinite(result.x)) else center, dtype=np.float64)
    objective, _, _ = evaluate(values)
    uncalibrated = _parameters_from_vector(values)

    # The EEG-derived driver has arbitrary units, so epsilon and both
    # chromophore gains contain an exact scale gauge.  Fix that gauge from
    # training data so the linearized flow perturbation remains in its valid
    # neighbourhood instead of plotting arbitrary-unit delta_f as physical
    # relative flow.  Observation reconstructions are invariant because gains
    # are recalibrated after scaling epsilon.
    uncalibrated_states = [simulate_hemodynamics(driver, uncalibrated, fs_hz) for driver in drivers]
    flow_reference = float(np.quantile(
        np.abs(np.concatenate([state[:, 1] for state in uncalibrated_states])),
        0.995,
    ))
    epsilon_scale = min(1.0, float(max_flow_perturbation) / max(flow_reference, 1e-12))
    params = HemodynamicParameters(
        epsilon=epsilon_scale,
        kas=uncalibrated.kas,
        kaf=uncalibrated.kaf,
        tau0=uncalibrated.tau0,
        alpha=uncalibrated.alpha,
        e0=uncalibrated.e0,
    )
    calibrated_states = [simulate_hemodynamics(driver, params, fs_hz) for driver in drivers]
    hbo_state = np.concatenate([state[:, 2] for state in calibrated_states])
    hbr_state = np.concatenate([state[:, 3] for state in calibrated_states])
    hbo_gain = _amplitude_calibrated_gain(hbo_state, hbo_concat)
    hbr_gain = _amplitude_calibrated_gain(hbr_state, hbr_concat)
    return params, hbo_gain, hbr_gain, objective, bool(result.success)


def estimate_driver_dynamics(drivers: Sequence[np.ndarray]) -> tuple[float, float]:
    """Estimate an unconstrained-enough AR(1) driver prior from training EEG."""

    previous = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1)[:-1] for value in drivers])
    current = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1)[1:] for value in drivers])
    phi = float(np.dot(previous, current) / max(float(np.dot(previous, previous)), 1e-12))
    phi = float(np.clip(phi, 0.45, 0.995))
    residual = current - phi * previous
    q = float(np.clip(np.var(residual), 1e-4, 4.0))
    return phi, q


def build_state_transition(
    params: HemodynamicParameters,
    phi: float,
    fs_hz: float = 10.0,
) -> np.ndarray:
    """Build the discrete five-state transition for ``[s,f,HbO,HbR,r]``."""

    hemodynamic, neural_input = discretize_hemodynamics(params, fs_hz)
    transition = np.zeros((5, 5), dtype=np.float64)
    transition[:4, :4] = hemodynamic
    transition[:4, 4] = neural_input
    transition[4, 4] = float(phi)
    return transition


def _initial_covariance(transition: np.ndarray, process_cov: np.ndarray) -> np.ndarray:
    try:
        covariance = solve_discrete_lyapunov(transition, process_cov)
    except Exception:
        covariance = np.eye(transition.shape[0], dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = (covariance + covariance.T) * 0.5
    diagonal_floor = np.maximum(np.diag(covariance), 1e-4)
    covariance += np.diag(diagonal_floor * 0.5 + 1e-6)
    return covariance


def rts_smoother(
    observations: np.ndarray,
    transition: np.ndarray,
    process_cov: np.ndarray,
    observation: np.ndarray,
    observation_cov: np.ndarray,
    initial_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run a missing-value-aware Kalman filter and RTS backward smoother."""

    values = np.asarray(observations, dtype=np.float64)
    steps = values.shape[0]
    state_dim = transition.shape[0]
    filtered_mean = np.zeros((steps, state_dim), dtype=np.float64)
    filtered_cov = np.zeros((steps, state_dim, state_dim), dtype=np.float64)
    predicted_mean = np.zeros_like(filtered_mean)
    predicted_cov = np.zeros_like(filtered_cov)
    mean = np.zeros(state_dim, dtype=np.float64)
    covariance = np.asarray(initial_cov, dtype=np.float64).copy()
    log_likelihood = 0.0

    for index in range(steps):
        if index:
            mean = transition @ mean
            covariance = transition @ covariance @ transition.T + process_cov
        predicted_mean[index] = mean
        predicted_cov[index] = covariance
        available = np.isfinite(values[index])
        if np.any(available):
            design = observation[available]
            noise = observation_cov[np.ix_(available, available)]
            predictive_residual = values[index, available] - design @ mean
            predictive_covariance = design @ covariance @ design.T + noise
            predictive_precision = np.linalg.pinv(predictive_covariance)
            gain = covariance @ design.T @ predictive_precision
            mean = mean + gain @ predictive_residual
            covariance = covariance - gain @ design @ covariance
            covariance = (covariance + covariance.T) * 0.5
            sign, logdet = np.linalg.slogdet(predictive_covariance)
            if sign > 0:
                log_likelihood += -0.5 * (
                    len(predictive_residual) * np.log(2.0 * np.pi)
                    + logdet
                    + float(predictive_residual @ predictive_precision @ predictive_residual)
                )
        filtered_mean[index] = mean
        filtered_cov[index] = covariance

    smoothed_mean = filtered_mean.copy()
    smoothed_cov = filtered_cov.copy()
    for index in range(steps - 2, -1, -1):
        smoothing_gain = filtered_cov[index] @ transition.T @ np.linalg.pinv(predicted_cov[index + 1])
        smoothed_mean[index] += smoothing_gain @ (smoothed_mean[index + 1] - predicted_mean[index + 1])
        smoothed_cov[index] += smoothing_gain @ (smoothed_cov[index + 1] - predicted_cov[index + 1]) @ smoothing_gain.T
        smoothed_cov[index] = (smoothed_cov[index] + smoothed_cov[index].T) * 0.5
    state_std = np.sqrt(np.maximum(np.diagonal(smoothed_cov, axis1=1, axis2=2), 0.0))
    return smoothed_mean, state_std, smoothed_cov, float(log_likelihood)


def _lowpass_residual_variance(signal: np.ndarray, cutoff_bins: int = 7) -> float:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if len(values) < 5:
        return 0.5
    width = min(int(cutoff_bins), len(values) if len(values) % 2 else len(values) - 1)
    width = max(width, 3)
    kernel = np.ones(width, dtype=np.float64) / width
    smooth = np.convolve(values, kernel, mode="same")
    return float(np.clip(np.var(values - smooth), 0.10, 1.50))


def fit_adaptive_ssm(
    eeg_drivers: Sequence[np.ndarray],
    hbo_targets: Sequence[np.ndarray],
    hbr_targets: Sequence[np.ndarray],
    *,
    fs_hz: float = 10.0,
    prior_strength: float = 0.03,
    max_iterations: int = 80,
    q_scale_candidates: Sequence[float] = (0.5, 1.0, 2.0),
    fnirs_noise_scale_candidates: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    balance_penalty: float = 0.25,
    baseline_samples: int = 0,
    max_flow_perturbation: float = 0.25,
) -> AdaptiveSSMFit:
    """Fit dynamics, observation adapters, and a training-only modality balance."""

    hbo_concat = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in hbo_targets])
    hbr_concat = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in hbr_targets])
    # Unified inputs are already event-baseline corrected.  Preserve zero as
    # the physiological rest coordinate instead of reintroducing a positive
    # training-set task mean at every new window.
    hbo_mean, hbr_mean = 0.0, 0.0
    hbo_std = max(float(np.std(hbo_concat)), 1e-8)
    hbr_std = max(float(np.std(hbr_concat)), 1e-8)
    hbo_norm = [(np.asarray(value, dtype=np.float64) - hbo_mean) / hbo_std for value in hbo_targets]
    hbr_norm = [(np.asarray(value, dtype=np.float64) - hbr_mean) / hbr_std for value in hbr_targets]
    drivers = [np.asarray(value, dtype=np.float64).reshape(-1) for value in eeg_drivers]
    params, hbo_gain, hbr_gain, objective, optimizer_success = fit_hemodynamic_parameters(
        drivers,
        hbo_norm,
        hbr_norm,
        fs_hz=fs_hz,
        prior_strength=prior_strength,
        max_iterations=max_iterations,
        max_flow_perturbation=max_flow_perturbation,
    )
    phi, q_driver = estimate_driver_dynamics(drivers)
    transition = build_state_transition(params, phi, fs_hz)
    observation = np.asarray([
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, hbo_gain, 0.0, 0.0],
        [0.0, 0.0, 0.0, hbr_gain, 0.0],
    ], dtype=np.float64)
    forward_states = [simulate_hemodynamics(driver, params, fs_hz) for driver in drivers]
    hbo_forward = np.concatenate([hbo_gain * state[:, 2] for state in forward_states])
    hbr_forward = np.concatenate([hbr_gain * state[:, 3] for state in forward_states])
    hbo_noise_base = float(np.clip(np.var(hbo_forward - np.concatenate(hbo_norm)), 0.20, 4.0))
    hbr_noise_base = float(np.clip(np.var(hbr_forward - np.concatenate(hbr_norm)), 0.20, 4.0))
    eeg_noise = float(np.mean([_lowpass_residual_variance(value) for value in drivers]))

    best: tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for q_scale in q_scale_candidates:
        process_cov = np.eye(5, dtype=np.float64) * 1e-8
        process_cov[4, 4] = max(float(q_driver) * float(q_scale), 1e-6)
        initial_cov = _initial_covariance(transition, process_cov)
        for noise_scale in fnirs_noise_scale_candidates:
            observation_cov = np.diag([
                eeg_noise,
                hbo_noise_base * float(noise_scale),
                hbr_noise_base * float(noise_scale),
            ]).astype(np.float64)
            modality_errors = []
            for eeg, hbo, hbr in zip(drivers, hbo_norm, hbr_norm):
                states, _, _, _ = rts_smoother(
                    np.column_stack((eeg, hbo, hbr)),
                    transition,
                    process_cov,
                    observation,
                    observation_cov,
                    initial_cov,
                )
                reconstruction = states @ observation.T
                if int(baseline_samples) > 0:
                    stop = min(int(baseline_samples), len(reconstruction))
                    reconstruction[:, 1:] -= np.mean(reconstruction[:stop, 1:], axis=0, keepdims=True)
                modality_errors.append(np.mean((reconstruction - np.column_stack((eeg, hbo, hbr))) ** 2, axis=0))
            errors = np.mean(modality_errors, axis=0)
            log_errors = np.log(np.maximum(errors, 1e-8))
            score = float(np.mean(errors) + float(balance_penalty) * np.std(log_errors))
            if best is None or score < best[0]:
                best = (score, float(q_scale), float(noise_scale), process_cov, observation_cov, initial_cov)
    if best is None:
        raise RuntimeError("no adaptive SSM balance candidates were evaluated")
    score, q_scale, noise_scale, process_cov, observation_cov, initial_cov = best
    return AdaptiveSSMFit(
        params=params,
        transition=transition,
        process_cov=process_cov,
        observation=observation,
        observation_cov=observation_cov,
        initial_cov=initial_cov,
        hbo_mean=hbo_mean,
        hbo_std=hbo_std,
        hbr_mean=hbr_mean,
        hbr_std=hbr_std,
        baseline_samples=int(baseline_samples),
        phi=phi,
        q_driver=q_driver,
        q_scale=q_scale,
        fnirs_noise_scale=noise_scale,
        hbo_gain=hbo_gain,
        hbr_gain=hbr_gain,
        eeg_noise=eeg_noise,
        hbo_noise_base=hbo_noise_base,
        hbr_noise_base=hbr_noise_base,
        training_score=score,
        optimizer_success=optimizer_success,
        optimizer_objective=objective,
    )


def apply_adaptive_ssm(
    eeg_driver: np.ndarray | None,
    fit: AdaptiveSSMFit,
    *,
    hbo_observation: np.ndarray | None = None,
    hbr_observation: np.ndarray | None = None,
    observation_mode: str | None = None,
) -> AdaptiveSmootherResult:
    """Apply joint, EEG-only, or fNIRS-only fixed-interval smoothing.

    ``observation_mode=None`` preserves the historical API: supplying both
    chromophores selects ``joint`` and omitting both selects ``eeg_only``.
    ``fnirs_only`` requires both chromophores, deliberately replaces the EEG
    update with missing values, and permits ``eeg_driver=None`` so callers do
    not have to pass a semantically unused EEG array. The fitted dynamics and
    observation operators are unchanged.
    """

    if (hbo_observation is None) != (hbr_observation is None):
        raise ValueError("HbO and HbR observations must either both be supplied or both omitted")
    inferred_mode = "joint" if hbo_observation is not None else "eeg_only"
    mode = inferred_mode if observation_mode is None else str(observation_mode)
    if mode not in {"joint", "eeg_only", "fnirs_only"}:
        raise ValueError("observation_mode must be joint, eeg_only, or fnirs_only")
    if mode in {"joint", "fnirs_only"} and hbo_observation is None:
        raise ValueError(f"{mode} requires both HbO and HbR observations")
    if mode == "eeg_only" and hbo_observation is not None:
        raise ValueError("eeg_only does not admit HbO/HbR observations")
    if eeg_driver is None:
        if mode != "fnirs_only":
            raise ValueError("eeg_driver=None is permitted only for fnirs_only")
        assert hbo_observation is not None
        steps = len(np.asarray(hbo_observation).reshape(-1))
        eeg_input = np.full(steps, np.nan, dtype=np.float64)
    else:
        eeg_input = np.asarray(eeg_driver, dtype=np.float64).reshape(-1)
        steps = len(eeg_input)
    if steps <= 0:
        raise ValueError("observations must contain at least one time point")

    eeg = (
        eeg_input
        if mode in {"joint", "eeg_only"}
        else np.full(steps, np.nan, dtype=np.float64)
    )
    hbo = np.full(steps, np.nan, dtype=np.float64)
    hbr = np.full(steps, np.nan, dtype=np.float64)
    if mode in {"joint", "fnirs_only"}:
        assert hbo_observation is not None and hbr_observation is not None
        hbo = (np.asarray(hbo_observation, dtype=np.float64).reshape(-1) - fit.hbo_mean) / fit.hbo_std
        hbr = (np.asarray(hbr_observation, dtype=np.float64).reshape(-1) - fit.hbr_mean) / fit.hbr_std
        if len(hbo) != steps or len(hbr) != steps:
            raise ValueError("all observations must have the same length")
    states, state_std, state_cov, log_likelihood = rts_smoother(
        np.column_stack((eeg, hbo, hbr)),
        fit.transition,
        fit.process_cov,
        fit.observation,
        fit.observation_cov,
        fit.initial_cov,
    )
    reconstruction = states @ fit.observation.T
    observation_predictive_variance = np.einsum(
        "oi,tij,oj->to",
        fit.observation,
        state_cov,
        fit.observation,
        optimize=True,
    ) + np.diag(fit.observation_cov)[None, :]
    observation_predictive_std = np.sqrt(
        np.maximum(observation_predictive_variance, 0.0)
    )
    if fit.baseline_samples > 0:
        stop = min(int(fit.baseline_samples), len(reconstruction))
        reconstruction[:, 1:] -= np.mean(reconstruction[:stop, 1:], axis=0, keepdims=True)
    observation_predictive_std[:, 1] *= abs(float(fit.hbo_std))
    observation_predictive_std[:, 2] *= abs(float(fit.hbr_std))
    return AdaptiveSmootherResult(
        states=states,
        state_std=state_std,
        observation_predictive_std=observation_predictive_std,
        eeg_reconstructed=reconstruction[:, 0],
        hbo_reconstructed=reconstruction[:, 1] * fit.hbo_std + fit.hbo_mean,
        hbr_reconstructed=reconstruction[:, 2] * fit.hbr_std + fit.hbr_mean,
        predictive_log_likelihood=log_likelihood,
        observation_mode=mode,
    )


def measurement_aligned_state_gauge(
    result: AdaptiveSmootherResult,
    fit: AdaptiveSSMFit,
) -> AdaptiveStateGaugeResult:
    """Return a train-fold observation-aligned gauge for teacher targets.

    ``hbo_gain``, ``hbr_gain``, and the chromophore scales are learned only
    from the training trials.  The event-baseline offset is the same declared
    deterministic transform already applied to the held-out reconstruction.
    Applying the gauge must therefore reproduce the emitted HbO/HbR clean
    means exactly while leaving the underlying smoother and reconstruction
    unchanged.
    """

    states = np.asarray(result.states, dtype=np.float64)
    state_std = np.asarray(result.state_std, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 5 or state_std.shape != states.shape:
        raise ValueError("adaptive state gauge expects matching [time, 5] state arrays")

    scales = np.ones(5, dtype=np.float64)
    scales[2] = float(fit.hbo_gain) * float(fit.hbo_std)
    scales[3] = float(fit.hbr_gain) * float(fit.hbr_std)
    if not np.all(np.isfinite(scales)) or np.any(np.abs(scales[[2, 3]]) < 1e-12):
        raise ValueError("chromophore observation gauge is non-finite or singular")

    offsets = np.zeros(5, dtype=np.float64)
    offsets[2] = float(fit.hbo_mean)
    offsets[3] = float(fit.hbr_mean)
    if int(fit.baseline_samples) > 0:
        stop = min(int(fit.baseline_samples), len(states))
        offsets[2] -= float(np.mean(states[:stop, 2]) * scales[2])
        offsets[3] -= float(np.mean(states[:stop, 3]) * scales[3])

    aligned = states * scales[None, :] + offsets[None, :]
    aligned_std = state_std * np.abs(scales)[None, :]
    reconstruction_delta = max(
        float(np.max(np.abs(aligned[:, 2] - np.asarray(result.hbo_reconstructed)))),
        float(np.max(np.abs(aligned[:, 3] - np.asarray(result.hbr_reconstructed)))),
    )
    if reconstruction_delta > 1e-8:
        raise RuntimeError(
            "observation-aligned gauge changed the physical reconstruction "
            f"(max abs delta={reconstruction_delta:.3e})"
        )
    return AdaptiveStateGaugeResult(
        states=aligned,
        state_std=aligned_std,
        scales=scales,
        offsets=offsets,
        reconstruction_max_abs_delta=reconstruction_delta,
    )


def fit_to_mapping(fit: AdaptiveSSMFit) -> Mapping[str, float | bool]:
    """Return the scalar fitted parameters for CSV/manifest serialization."""

    return {
        "epsilon": fit.params.epsilon,
        "kas": fit.params.kas,
        "kaf": fit.params.kaf,
        "tau0": fit.params.tau0,
        "alpha": fit.params.alpha,
        "e0": fit.params.e0,
        "phi": fit.phi,
        "q_driver": fit.q_driver,
        "q_scale": fit.q_scale,
        "fnirs_noise_scale": fit.fnirs_noise_scale,
        "hbo_gain": fit.hbo_gain,
        "hbr_gain": fit.hbr_gain,
        "hbo_mean": fit.hbo_mean,
        "hbo_std": fit.hbo_std,
        "hbr_mean": fit.hbr_mean,
        "hbr_std": fit.hbr_std,
        "hbo_state_measurement_scale": fit.hbo_gain * fit.hbo_std,
        "hbr_state_measurement_scale": fit.hbr_gain * fit.hbr_std,
        "eeg_noise": fit.eeg_noise,
        "hbo_noise_base": fit.hbo_noise_base,
        "hbr_noise_base": fit.hbr_noise_base,
        "baseline_samples": fit.baseline_samples,
        "training_score": fit.training_score,
        "optimizer_success": fit.optimizer_success,
        "optimizer_objective": fit.optimizer_objective,
    }
