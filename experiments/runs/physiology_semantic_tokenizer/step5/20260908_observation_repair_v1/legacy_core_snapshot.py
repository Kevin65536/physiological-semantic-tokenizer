"""Minimal robust nonlinear Balloon state-space teacher.

This module is deliberately self contained.  It implements the T3a
continuous-time core in transformed coordinates
``z = (r, s, log(f), log(v), log(p), log(q))`` so that all four Balloon
compartments remain positive during integration.  Inference is an iterated
extended-Kalman fixed-interval smoother with Student-t IRLS observation
updates.  The IRLS/Laplace approximation is a robust engineering posterior,
not an exact Student-t posterior.

The ``p`` equation uses the minimal ``tau_v=0`` Tak extension required by the
current contract, ``tau * dp/dt = f - f_out * p / v``.  It is intentionally
not the alternative ``(f-f_out)*p/v`` linearized form used by the old
adaptive implementation.

The module-level :func:`fit_balloon` API fits only ``kappa`` and ``tau``.
Other physiology values remain in :class:`BalloonFixedParameters`; an
experiment runner may compare staged fixed-parameter variants under its own
explicit contract.  ``P0``, ``Q0`` and the neural-driver/noise gauges remain
contract-supplied.  ``process_std`` is interpreted as a transformed-state
continuous diffusion standard deviation (per square-root second); the
discrete covariance used by the filter is ``Q = process_std**2 * dt``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as student_t


STATE_NAMES: tuple[str, ...] = ("r", "s", "f", "v", "p", "q")
OBSERVATION_NAMES: tuple[str, ...] = ("EEG", "HbO", "HbR")
_STATE_DIM = len(STATE_NAMES)
_OBS_DIM = len(OBSERVATION_NAMES)


@dataclass(frozen=True)
class BalloonFixedParameters:
    """Fixed T3a parameters and observation/noise gauges.

    ``P0`` and ``Q0`` are baseline total-Hb and deoxy-Hb model-coordinate
    scales in the declared observation coordinate.  Absolute total-Hb and
    deoxy-Hb checks use ``P0*p`` and ``Q0*q`` directly; this keeps one baseline
    convention and avoids a second fitted or hidden offset.  Entries in
    ``process_std`` are continuous diffusion standard deviations in transformed
    coordinates, not per-sample standard deviations.
    """

    alpha: float = 0.32
    E0: float = 0.32
    gamma: float = 0.32
    P0: float = 1.0
    Q0: float = 0.35
    driver_decay_per_s: float = 0.45
    process_std: tuple[float, ...] = (0.08, 0.01, 0.006, 0.004, 0.004, 0.004)
    observation_scale: tuple[float, ...] = (0.08, 0.025, 0.015)
    student_nu: float = 5.0
    eeg_loading: float = 1.0
    eeg_offset: float = 0.0
    # Kept at the end to preserve positional construction of older callers.
    neurovascular_gain: float = 1.0

    def validate(self) -> None:
        """Validate hard mathematical and observation-contract boundaries."""

        values = {
            "alpha": self.alpha,
            "E0": self.E0,
            "gamma": self.gamma,
            "neurovascular_gain": self.neurovascular_gain,
            "P0": self.P0,
            "Q0": self.Q0,
            "driver_decay_per_s": self.driver_decay_per_s,
            "student_nu": self.student_nu,
            "eeg_loading": self.eeg_loading,
            "eeg_offset": self.eeg_offset,
        }
        if any(not np.isfinite(float(value)) for value in values.values()):
            raise ValueError("fixed Balloon parameters must be finite")
        if self.alpha <= 0.0 or self.gamma <= 0.0 or self.neurovascular_gain <= 0.0:
            raise ValueError("alpha, gamma, and neurovascular_gain must be positive")
        if not 0.0 < self.E0 < 1.0:
            raise ValueError("E0 must lie strictly between zero and one")
        if self.P0 <= 0.0 or self.Q0 <= 0.0:
            raise ValueError("P0 and Q0 must be positive")
        if self.eeg_loading <= 0.0:
            raise ValueError("eeg_loading must be strictly positive for the fixed sign gauge")
        if self.driver_decay_per_s <= 0.0:
            raise ValueError("driver_decay_per_s must be positive")
        if self.student_nu <= 2.0:
            raise ValueError("student_nu must exceed two for finite variance")
        if self.Q0 > self.P0:
            raise ValueError("Q0 must not exceed P0 for absolute Hb balance")
        process_std = np.asarray(self.process_std, dtype=np.float64)
        observation_scale = np.asarray(self.observation_scale, dtype=np.float64)
        if process_std.shape != (_STATE_DIM,) or not np.all(np.isfinite(process_std)):
            raise ValueError("process_std must be finite with one entry per state")
        if np.any(process_std < 0.0):
            raise ValueError("process_std cannot be negative")
        if observation_scale.shape != (_OBS_DIM,) or not np.all(np.isfinite(observation_scale)):
            raise ValueError("observation_scale must have one positive entry per observation")
        if np.any(observation_scale <= 0.0):
            raise ValueError("observation_scale must be positive")


@dataclass(frozen=True)
class BalloonFreeParameters:
    """The only parameters allowed to vary in the initial T3a fit."""

    kappa: float = 0.65
    tau: float = 2.0

    def validate(self) -> None:
        if not np.isfinite(self.kappa) or not np.isfinite(self.tau):
            raise ValueError("kappa and tau must be finite")
        if self.kappa <= 0.0 or self.tau <= 0.0:
            raise ValueError("kappa and tau must be positive")


@dataclass(frozen=True)
class BalloonParameters:
    """Complete T3a parameter object, split into fixed and free sections."""

    fixed: BalloonFixedParameters = field(default_factory=BalloonFixedParameters)
    free: BalloonFreeParameters = field(default_factory=BalloonFreeParameters)

    def validate(self) -> None:
        self.fixed.validate()
        self.free.validate()


@dataclass(frozen=True)
class BalloonObservationSpec:
    """Explicit EEG and HbO/HbR observation operator contract."""

    coordinate_names: tuple[str, ...] = OBSERVATION_NAMES
    eeg_loading: float | None = None
    eeg_offset: float | None = None
    observation_scale: tuple[float, ...] | None = None
    student_nu: float | None = None

    def resolved(self, fixed: BalloonFixedParameters) -> "BalloonObservationSpec":
        names = tuple(str(name) for name in self.coordinate_names)
        if names != OBSERVATION_NAMES:
            raise ValueError(
                f"T3a currently requires observation order {OBSERVATION_NAMES}, got {names}"
            )
        loading = fixed.eeg_loading if self.eeg_loading is None else float(self.eeg_loading)
        offset = fixed.eeg_offset if self.eeg_offset is None else float(self.eeg_offset)
        scales = fixed.observation_scale if self.observation_scale is None else tuple(self.observation_scale)
        nu = fixed.student_nu if self.student_nu is None else float(self.student_nu)
        result = BalloonObservationSpec(
            coordinate_names=names,
            eeg_loading=loading,
            eeg_offset=offset,
            observation_scale=tuple(float(value) for value in scales),
            student_nu=nu,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if tuple(self.coordinate_names) != OBSERVATION_NAMES:
            raise ValueError("coordinate_names must be (EEG, HbO, HbR)")
        if self.eeg_loading is None or self.eeg_offset is None:
            raise ValueError("observation spec must be resolved against fixed parameters")
        if not np.isfinite(self.eeg_loading) or not np.isfinite(self.eeg_offset):
            raise ValueError("EEG loading and offset must be finite")
        if self.eeg_loading <= 0.0:
            raise ValueError("EEG loading must be strictly positive for the fixed sign gauge")
        if self.observation_scale is None or self.student_nu is None:
            raise ValueError("observation spec must be resolved against fixed parameters")
        scales = np.asarray(self.observation_scale, dtype=np.float64)
        if scales.shape != (_OBS_DIM,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("observation_scale must contain three positive finite values")
        if not np.isfinite(self.student_nu) or self.student_nu <= 2.0:
            raise ValueError("student_nu must exceed two")


@dataclass(frozen=True)
class BalloonConfig:
    """Numerical and fitting settings; these are not physiological parameters."""

    dt: float = 0.1
    rk4_substeps: int = 2
    irls_iterations: int = 3
    irls_weight_floor: float = 0.05
    initial_state_std: tuple[float, ...] = (0.5, 0.5, 0.08, 0.08, 0.08, 0.08)
    kappa_bounds: tuple[float, float] = (0.20, 1.50)
    tau_bounds: tuple[float, float] = (0.50, 5.00)
    optimizer_max_iterations: int = 60
    optimizer_starts: tuple[tuple[float, float], ...] = (
        (0.35, 1.0),
        (0.65, 2.0),
        (1.10, 3.5),
    )
    kappa_prior_mean: float = 0.64
    kappa_prior_sd: float = 0.20
    tau_prior_mean: float = 2.0
    tau_prior_sd: float = 0.75
    hessian_step: float = 1.0e-3

    def validate(self) -> None:
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if int(self.rk4_substeps) < 1:
            raise ValueError("rk4_substeps must be at least one")
        if int(self.irls_iterations) < 1:
            raise ValueError("irls_iterations must be at least one")
        if not 0.0 < self.irls_weight_floor <= 1.0:
            raise ValueError("irls_weight_floor must lie in (0, 1]")
        initial_std = np.asarray(self.initial_state_std, dtype=np.float64)
        if initial_std.shape != (_STATE_DIM,) or not np.all(np.isfinite(initial_std)):
            raise ValueError("initial_state_std must have one finite entry per state")
        if np.any(initial_std <= 0.0):
            raise ValueError("initial_state_std must be positive")
        for name, bounds in (("kappa", self.kappa_bounds), ("tau", self.tau_bounds)):
            if len(bounds) != 2 or not np.all(np.isfinite(bounds)):
                raise ValueError(f"{name}_bounds must contain two finite values")
            if bounds[0] <= 0.0 or bounds[1] <= bounds[0]:
                raise ValueError(f"{name}_bounds must be positive and ordered")
        prior_values = (
            self.kappa_prior_mean,
            self.kappa_prior_sd,
            self.tau_prior_mean,
            self.tau_prior_sd,
        )
        if not np.all(np.isfinite(prior_values)) or self.kappa_prior_sd <= 0.0 or self.tau_prior_sd <= 0.0:
            raise ValueError("free-parameter Gaussian priors must be finite with positive scales")
        if int(self.optimizer_max_iterations) < 1 or self.hessian_step <= 0.0:
            raise ValueError("optimizer_max_iterations and hessian_step must be positive")


@dataclass(frozen=True)
class BalloonFit:
    """Fit-fold parameters and local identifiability diagnostics."""

    parameters: BalloonParameters
    objective: float
    optimizer_success: bool
    optimizer_message: str
    starts: tuple[Mapping[str, Any], ...]
    hessian: np.ndarray
    parameter_covariance: np.ndarray
    identifiability_status: str
    boundary_status: str
    likelihood_hessian: np.ndarray = field(
        default_factory=lambda: np.full((2, 2), np.nan, dtype=np.float64)
    )

    @property
    def parameter_summary(self) -> Mapping[str, Any]:
        """Return a flat serialization-friendly parameter summary."""

        return {
            "kappa": float(self.parameters.free.kappa),
            "tau": float(self.parameters.free.tau),
            "alpha": float(self.parameters.fixed.alpha),
            "E0": float(self.parameters.fixed.E0),
            "gamma": float(self.parameters.fixed.gamma),
            "neurovascular_gain": float(self.parameters.fixed.neurovascular_gain),
            "P0": float(self.parameters.fixed.P0),
            "Q0": float(self.parameters.fixed.Q0),
            "driver_decay_per_s": float(self.parameters.fixed.driver_decay_per_s),
            "objective": float(self.objective),
            "optimizer_success": bool(self.optimizer_success),
            "identifiability_status": self.identifiability_status,
            "boundary_status": self.boundary_status,
        }


@dataclass(frozen=True)
class BalloonSmootherResult:
    """Common T3a trajectory, uncertainty, and mask output.

    ``epistemic_variance`` is the conditional posterior variance induced by
    the latent state covariance at the fixed fitted parameters.  It excludes
    parameter-estimation covariance; the latter remains in
    :class:`BalloonFit.parameter_covariance` and is not silently folded into
    predictive intervals.  ``observation_residual`` is
    ``observation - trajectory_mean`` after RTS smoothing; the one-step
    predictive residual used for the fit score is kept separate.
    """

    state_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    state_mean: np.ndarray
    state_variance: np.ndarray
    observation_mean: np.ndarray
    trajectory_mean: np.ndarray
    aleatoric_variance: np.ndarray
    epistemic_variance: np.ndarray
    total_variance: np.ndarray
    observation_residual: np.ndarray
    observation_mask: np.ndarray
    teacher_valid_mask: np.ndarray
    uncertainty_valid_mask: np.ndarray
    trajectory_valid_mask: np.ndarray
    observation_residual_valid_mask: np.ndarray
    predictive_log_likelihood: float
    physical_checks: Mapping[str, Any]
    parameters: BalloonParameters
    uncertainty_method: str = (
        "Student-t IRLS extended-Kalman Laplace approximation; "
        "epistemic excludes parameter covariance"
    )
    observation_residual_kind: str = "observation_minus_posterior_trajectory"

    @property
    def state_std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.state_variance, 0.0))

    @property
    def observation_std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.total_variance, 0.0))

    @property
    def shared_driver_mean(self) -> np.ndarray:
        return self.state_mean[:, 0]

    @property
    def shared_driver_variance(self) -> np.ndarray:
        return self.state_variance[:, 0]


@dataclass(frozen=True)
class BalloonSimulation:
    """Known-truth synthetic trajectory and observation arrays."""

    states: np.ndarray
    clean_observations: np.ndarray
    observations: np.ndarray
    observation_mask: np.ndarray


def _as_vector(value: Sequence[float] | np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape [{size}]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _safe_exp(value: np.ndarray | float) -> np.ndarray | float:
    with np.errstate(over="raise", invalid="raise"):
        try:
            return np.exp(value)
        except FloatingPointError as exc:
            raise FloatingPointError("transformed Balloon state overflowed") from exc


def physical_to_transformed(state: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map physical ``(r,s,f,v,p,q)`` to the positive transformed state."""

    values = _as_vector(state, _STATE_DIM, "state")
    if np.any(values[2:] <= 0.0):
        raise ValueError("f, v, p, and q must be strictly positive")
    return np.asarray(
        [values[0], values[1], np.log(values[2]), np.log(values[3]), np.log(values[4]), np.log(values[5])],
        dtype=np.float64,
    )


def transformed_to_physical(state: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map transformed ``(r,s,log f,log v,log p,log q)`` to physical state."""

    values = _as_vector(state, _STATE_DIM, "transformed state")
    return _transformed_to_physical_unchecked(values)


def _transformed_to_physical_unchecked(state: np.ndarray) -> np.ndarray:
    """Map a known-valid transformed state without repeated shape checks."""

    values = np.asarray(state, dtype=np.float64)
    positive = _safe_exp(values[2:])
    if np.any(np.asarray(positive) <= 0.0):
        raise FloatingPointError("transformed Balloon state underflowed to a non-positive compartment")
    output = np.concatenate((values[:2], np.asarray(positive, dtype=np.float64)))
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("transformed Balloon state produced non-finite values")
    return output


def _extraction(f: float, E0: float) -> tuple[float, float, float]:
    """Return E(f), dE/df, and f*E/E0 with stable logarithms."""

    if f <= 0.0 or not 0.0 < E0 < 1.0:
        raise ValueError("extraction requires f>0 and 0<E0<1")
    log_one_minus = np.log1p(-E0)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        exponent = log_one_minus / f
        one_minus_E = float(np.exp(exponent))
        E = float(-np.expm1(exponent))
        dE_df = float(one_minus_E * log_one_minus / (f * f))
        flow_extraction = float(f * E / E0)
    if (
        not np.isfinite(E)
        or not np.isfinite(dE_df)
        or not np.isfinite(flow_extraction)
        or not 0.0 < E < 1.0
    ):
        raise FloatingPointError("oxygen extraction left its strict physical domain")
    return E, dE_df, flow_extraction


def balloon_rhs(
    transformed_state: Sequence[float] | np.ndarray,
    parameters: BalloonParameters,
    *,
    dt: float = 0.1,
) -> np.ndarray:
    """Continuous transformed-coordinate Balloon dynamics.

    ``r`` follows a fixed continuous decay rate
    ``dr/dt = -driver_decay_per_s * r``.  This is a fixed driver prior, not a
    fitted physiological parameter.
    """

    parameters.validate()
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    z = _as_vector(transformed_state, _STATE_DIM, "transformed_state")
    return _balloon_rhs_unchecked(z, parameters)


def _balloon_rhs_unchecked(
    transformed_state: np.ndarray,
    parameters: BalloonParameters,
) -> np.ndarray:
    """Evaluate dynamics after the caller has validated parameters and shape."""

    fixed, free = parameters.fixed, parameters.free
    r, s, f, v, p, q = _transformed_to_physical_unchecked(transformed_state)
    f_out = float(v ** (1.0 / fixed.alpha))
    E, dE_df, flow_extraction = _extraction(f, fixed.E0)
    tau = free.tau
    d = f - f_out
    p_balance = f - f_out * p / v
    rhs = np.asarray(
        [
            -fixed.driver_decay_per_s * r,
            fixed.neurovascular_gain * r - free.kappa * s - fixed.gamma * (f - 1.0),
            s / f,
            d / (tau * v),
            p_balance / (tau * p),
            (flow_extraction - f_out * q / v) / (tau * q),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(rhs)):
        raise FloatingPointError("Balloon dynamics produced non-finite derivative")
    return rhs


def balloon_rhs_jacobian(
    transformed_state: Sequence[float] | np.ndarray,
    parameters: BalloonParameters,
    *,
    dt: float = 0.1,
) -> np.ndarray:
    """Analytic Jacobian of :func:`balloon_rhs` in transformed coordinates."""

    parameters.validate()
    z = _as_vector(transformed_state, _STATE_DIM, "transformed_state")
    return _balloon_rhs_jacobian_unchecked(z, parameters)


def _balloon_rhs_jacobian_unchecked(
    transformed_state: np.ndarray,
    parameters: BalloonParameters,
) -> np.ndarray:
    """Evaluate dynamics Jacobian after the caller has validated inputs."""

    fixed, free = parameters.fixed, parameters.free
    r, s, f, v, p, q = _transformed_to_physical_unchecked(transformed_state)
    alpha, tau = fixed.alpha, free.tau
    f_out = float(v ** (1.0 / alpha))
    E, dE_df, flow_extraction = _extraction(f, fixed.E0)
    d = f - f_out
    v_ratio = f_out / v
    jacobian = np.zeros((_STATE_DIM, _STATE_DIM), dtype=np.float64)
    jacobian[0, 0] = -fixed.driver_decay_per_s
    jacobian[1, 0] = fixed.neurovascular_gain
    jacobian[1, 1] = -free.kappa
    jacobian[1, 2] = -fixed.gamma * f
    jacobian[2, 1] = 1.0 / f
    jacobian[2, 2] = -s / f
    dv_log = -(f + (1.0 / alpha - 1.0) * f_out) / v
    jacobian[3, 2] = f / (tau * v)
    jacobian[3, 3] = dv_log / tau
    jacobian[4, 2] = f / (tau * p)
    jacobian[4, 3] = -(1.0 / alpha - 1.0) * v_ratio / tau
    jacobian[4, 4] = -f / (tau * p)
    dflow_dlogf = f * (E + f * dE_df) / fixed.E0
    jacobian[5, 2] = dflow_dlogf / (tau * q)
    jacobian[5, 3] = -(1.0 / alpha - 1.0) * v_ratio / tau
    jacobian[5, 5] = -flow_extraction / (tau * q)
    if not np.all(np.isfinite(jacobian)):
        raise FloatingPointError("Balloon Jacobian produced non-finite values")
    return jacobian


def _rk4_step(
    state: np.ndarray,
    parameters: BalloonParameters,
    step: float,
) -> np.ndarray:
    k1 = _balloon_rhs_unchecked(state, parameters)
    k2 = _balloon_rhs_unchecked(state + 0.5 * step * k1, parameters)
    k3 = _balloon_rhs_unchecked(state + 0.5 * step * k2, parameters)
    k4 = _balloon_rhs_unchecked(state + step * k3, parameters)
    output = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("RK4 Balloon transition produced non-finite state")
    return output


def _rk4_step_with_jacobian(
    state: np.ndarray,
    parameters: BalloonParameters,
    step: float,
    tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k1_state = _balloon_rhs_unchecked(state, parameters)
    k1_tangent = _balloon_rhs_jacobian_unchecked(state, parameters) @ tangent
    state_2 = state + 0.5 * step * k1_state
    tangent_2 = tangent + 0.5 * step * k1_tangent
    k2_state = _balloon_rhs_unchecked(state_2, parameters)
    k2_tangent = _balloon_rhs_jacobian_unchecked(state_2, parameters) @ tangent_2
    state_3 = state + 0.5 * step * k2_state
    tangent_3 = tangent + 0.5 * step * k2_tangent
    k3_state = _balloon_rhs_unchecked(state_3, parameters)
    k3_tangent = _balloon_rhs_jacobian_unchecked(state_3, parameters) @ tangent_3
    state_4 = state + step * k3_state
    tangent_4 = tangent + step * k3_tangent
    k4_state = _balloon_rhs_unchecked(state_4, parameters)
    k4_tangent = _balloon_rhs_jacobian_unchecked(state_4, parameters) @ tangent_4
    output = state + step * (k1_state + 2.0 * k2_state + 2.0 * k3_state + k4_state) / 6.0
    output_tangent = tangent + step * (
        k1_tangent + 2.0 * k2_tangent + 2.0 * k3_tangent + k4_tangent
    ) / 6.0
    if not np.all(np.isfinite(output)) or not np.all(np.isfinite(output_tangent)):
        raise FloatingPointError("RK4 Balloon tangent transition produced non-finite values")
    return output, output_tangent


def rk4_transition(
    transformed_state: Sequence[float] | np.ndarray,
    parameters: BalloonParameters,
    config: BalloonConfig = BalloonConfig(),
) -> np.ndarray:
    """Integrate one configured sample interval with RK4."""

    parameters.validate()
    config.validate()
    state = _as_vector(transformed_state, _STATE_DIM, "transformed_state")
    step = config.dt / int(config.rk4_substeps)
    for _ in range(int(config.rk4_substeps)):
        state = _rk4_step(state, parameters, step)
    return state


def rk4_transition_with_jacobian(
    transformed_state: Sequence[float] | np.ndarray,
    parameters: BalloonParameters,
    config: BalloonConfig = BalloonConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate one interval and its state-transition Jacobian."""

    parameters.validate()
    config.validate()
    state = _as_vector(transformed_state, _STATE_DIM, "transformed_state")
    tangent = np.eye(_STATE_DIM, dtype=np.float64)
    step = config.dt / int(config.rk4_substeps)
    for _ in range(int(config.rk4_substeps)):
        state, tangent = _rk4_step_with_jacobian(state, parameters, step, tangent)
    return state, tangent


def observation_map(
    state: Sequence[float] | np.ndarray,
    parameters: BalloonParameters,
    observation_spec: BalloonObservationSpec | None = None,
) -> np.ndarray:
    """Map physical state to ``(EEG, HbO, HbR)`` model coordinates."""

    parameters.validate()
    spec = (observation_spec or BalloonObservationSpec()).resolved(parameters.fixed)
    values = _as_vector(state, _STATE_DIM, "state")
    if np.any(values[2:] <= 0.0):
        raise ValueError("observation map requires positive f, v, p, and q")
    return _observation_map_unchecked(values, parameters, spec)


def _observation_map_unchecked(
    state: np.ndarray,
    parameters: BalloonParameters,
    spec: BalloonObservationSpec,
) -> np.ndarray:
    """Map a known-valid physical state without repeated contract checks."""

    r, _, _, _, p, q = state
    delta_hbt = parameters.fixed.P0 * (p - 1.0)
    delta_hbr = parameters.fixed.Q0 * (q - 1.0)
    delta_hbo = delta_hbt - delta_hbr
    output = np.asarray(
        [float(spec.eeg_loading) * r + float(spec.eeg_offset), delta_hbo, delta_hbr],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("observation map produced non-finite values")
    return output


def _observation_physical_matrix(
    parameters: BalloonParameters,
    observation_spec: BalloonObservationSpec,
) -> np.ndarray:
    """Linear part of the observation map in physical state coordinates."""

    matrix = np.zeros((_OBS_DIM, _STATE_DIM), dtype=np.float64)
    matrix[0, 0] = float(observation_spec.eeg_loading)
    matrix[1, 4] = parameters.fixed.P0
    matrix[1, 5] = -parameters.fixed.Q0
    matrix[2, 5] = parameters.fixed.Q0
    return matrix


def observation_jacobian(
    transformed_state: Sequence[float] | np.ndarray,
    parameters: BalloonParameters,
    observation_spec: BalloonObservationSpec | None = None,
) -> np.ndarray:
    """Jacobian of :func:`observation_map` with respect to transformed state."""

    parameters.validate()
    spec = (observation_spec or BalloonObservationSpec()).resolved(parameters.fixed)
    z = _as_vector(transformed_state, _STATE_DIM, "transformed_state")
    return _observation_jacobian_unchecked(z, parameters, spec)


def _observation_jacobian_unchecked(
    transformed_state: np.ndarray,
    parameters: BalloonParameters,
    spec: BalloonObservationSpec,
) -> np.ndarray:
    """Build the observation Jacobian without repeated contract checks."""

    _, _, f, v, p, q = _transformed_to_physical_unchecked(transformed_state)
    del f, v
    jacobian = np.zeros((_OBS_DIM, _STATE_DIM), dtype=np.float64)
    jacobian[0, 0] = float(spec.eeg_loading)
    jacobian[1, 4] = parameters.fixed.P0 * p
    jacobian[1, 5] = -parameters.fixed.Q0 * q
    jacobian[2, 5] = parameters.fixed.Q0 * q
    if not np.all(np.isfinite(jacobian)):
        raise FloatingPointError("observation Jacobian produced non-finite values")
    return jacobian


def student_t_irls_weights(
    residual: Sequence[float] | np.ndarray,
    scale: Sequence[float] | np.ndarray,
    nu: float,
    *,
    floor: float = 0.05,
) -> np.ndarray:
    """Return bounded latent precisions for diagonal Student-t updates."""

    residual_array = np.asarray(residual, dtype=np.float64)
    scale_array = np.asarray(scale, dtype=np.float64)
    if not np.isfinite(nu) or nu <= 2.0 or floor <= 0.0 or floor > 1.0:
        raise ValueError("Student-t degrees of freedom and IRLS floor are invalid")
    if residual_array.shape != scale_array.shape:
        raise ValueError("residual and scale must share a positive scale shape")
    if not np.all(np.isfinite(residual_array)) or not np.all(np.isfinite(scale_array)):
        raise ValueError("residual and scale must be finite")
    if np.any(scale_array <= 0.0):
        raise ValueError("residual and scale must use a positive scale")
    standardized = residual_array / scale_array
    weights = (nu + 1.0) / (nu + standardized * standardized)
    return np.maximum(weights, float(floor))


def _project_psd(covariance: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.float64)
    matrix = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, float(floor))
    return (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T


def transformed_gaussian_moments(
    transformed_mean: Sequence[float] | np.ndarray,
    transformed_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact physical moments of a transformed Gaussian state.

    The first two coordinates ``(r, s)`` are linear and the four Balloon
    compartments are represented as logarithms.  For a Gaussian
    ``z ~ N(mu, C)``, this helper maps to ``x=(r,s,exp(log f),...)`` using
    log-normal moments rather than the plug-in value ``exp(mu)``.  In
    particular, ``E[exp(z_i)] = exp(mu_i+C_ii/2)``,
    ``Var[exp(z_i)] = E_i^2*expm1(C_ii)``, linear/log covariance is
    ``C_ij*E_j``, and log/log covariance is
    ``E_i*E_j*expm1(C_ij)``.  The covariance returned here is therefore in
    physical coordinates and can be passed through the linear observation
    map without a delta-method approximation.
    """

    mean = _as_vector(transformed_mean, _STATE_DIM, "transformed_mean")
    covariance = np.asarray(transformed_covariance, dtype=np.float64)
    if covariance.shape != (_STATE_DIM, _STATE_DIM) or not np.all(np.isfinite(covariance)):
        raise ValueError("transformed_covariance must be a finite 6x6 matrix")
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(np.min(eigenvalues)) < -1.0e-8:
        raise ValueError("transformed_covariance must be positive semidefinite")
    # Smoothing can leave round-off-sized negative eigenvalues.  Clipping only
    # those values preserves the supplied Gaussian covariance contract.
    covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)[None, :]) @ eigenvectors.T

    physical_mean = np.empty(_STATE_DIM, dtype=np.float64)
    physical_mean[:2] = mean[:2]
    diagonal = np.diag(covariance)
    with np.errstate(over="raise", invalid="raise"):
        physical_mean[2:] = np.exp(mean[2:] + 0.5 * diagonal[2:])

    physical_covariance = np.empty((_STATE_DIM, _STATE_DIM), dtype=np.float64)
    for first in range(_STATE_DIM):
        for second in range(_STATE_DIM):
            first_log = first >= 2
            second_log = second >= 2
            if not first_log and not second_log:
                value = covariance[first, second]
            elif first_log and not second_log:
                value = covariance[first, second] * physical_mean[first]
            elif not first_log and second_log:
                value = covariance[first, second] * physical_mean[second]
            else:
                value = (
                    physical_mean[first]
                    * physical_mean[second]
                    * np.expm1(covariance[first, second])
                )
            physical_covariance[first, second] = value
    physical_covariance = (physical_covariance + physical_covariance.T) * 0.5
    if not np.all(np.isfinite(physical_mean)) or not np.all(np.isfinite(physical_covariance)):
        raise FloatingPointError("physical Gaussian moments are non-finite")
    # The exact moment matrix is PSD in exact arithmetic; project only tiny
    # numerical violations so downstream uncertainty fields remain valid.
    physical_covariance = _project_psd(physical_covariance, floor=0.0)
    return physical_mean, physical_covariance


def _initial_transformed_covariance(config: BalloonConfig) -> np.ndarray:
    return np.diag(np.square(np.asarray(config.initial_state_std, dtype=np.float64)))


def _resolved_process_covariance(parameters: BalloonParameters, config: BalloonConfig) -> np.ndarray:
    return np.diag(np.square(np.asarray(parameters.fixed.process_std, dtype=np.float64))) * config.dt


def _observation_update(
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    observed: np.ndarray,
    available: np.ndarray,
    parameters: BalloonParameters,
    spec: BalloonObservationSpec,
    config: BalloonConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Perform iterated EKF updates from one fixed prior using IRLS weights."""

    if not np.any(available):
        return prior_mean.copy(), prior_cov.copy(), 0.0
    indices = np.flatnonzero(available)
    y = observed[indices]
    scales = np.asarray(spec.observation_scale, dtype=np.float64)[indices]
    nu = float(spec.student_nu)
    # Proper score used by fitting: evaluate the observation before it is
    # consumed by the smoother, with the prior latent covariance folded into
    # a diagonal predictive scale.  The resulting Student-t score is an
    # explicit approximation to the convolution of Gaussian state uncertainty
    # and Student-t observation noise; it is not a same-point posterior score.
    prior_state_mean, prior_state_covariance = transformed_gaussian_moments(
        prior_mean, prior_cov
    )
    prior_prediction = _observation_map_unchecked(prior_state_mean, parameters, spec)[indices]
    observation_matrix = _observation_physical_matrix(parameters, spec)
    prior_observation_covariance = (
        observation_matrix @ prior_state_covariance @ observation_matrix.T
    )
    predictive_scale = np.sqrt(
        np.maximum(
            scales * scales
            + np.diag(prior_observation_covariance)[indices]
            * (nu - 2.0)
            / nu,
            1e-12,
        )
    )
    prior_residual = y - prior_prediction
    log_likelihood = float(
        np.sum(
            student_t.logpdf(prior_residual / predictive_scale, df=nu)
            - np.log(predictive_scale)
        )
    )
    mean = prior_mean.copy()
    covariance = prior_cov.copy()
    for _ in range(int(config.irls_iterations)):
        predicted = _observation_map_unchecked(
            _transformed_to_physical_unchecked(mean), parameters, spec
        )[indices]
        residual = y - predicted
        weights = student_t_irls_weights(
            residual,
            scales,
            float(spec.student_nu),
            floor=float(config.irls_weight_floor),
        )
        effective_noise = np.diag(np.square(scales) / weights)
        design = _observation_jacobian_unchecked(mean, parameters, spec)[indices]
        predictive_covariance = _project_psd(design @ prior_cov @ design.T + effective_noise)
        try:
            gain = prior_cov @ design.T @ np.linalg.pinv(predictive_covariance)
        except np.linalg.LinAlgError:
            gain = prior_cov @ design.T @ np.linalg.pinv(predictive_covariance + np.eye(len(indices)) * 1e-8)
        # Iterated EKF correction: linearize h at the current iterate while
        # keeping one fixed prior.  Omitting H(m-prior) would repeatedly
        # apply the same nonlinear residual and over-count the observation.
        linearized_residual = residual + design @ (mean - prior_mean)
        mean = prior_mean + gain @ linearized_residual
        identity = np.eye(_STATE_DIM, dtype=np.float64)
        covariance = _project_psd(
            (identity - gain @ design) @ prior_cov @ (identity - gain @ design).T
            + gain @ effective_noise @ gain.T
        )
    return mean, covariance, log_likelihood


def _validate_observation_inputs(
    observations: np.ndarray,
    observation_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != _OBS_DIM or values.shape[0] < 1:
        raise ValueError("observations must have shape [time, 3]")
    finite = np.isfinite(values)
    if observation_mask is None:
        mask = finite
    else:
        mask = np.asarray(observation_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("observation_mask must match observations")
        mask = mask & finite
    return values, mask


def _physical_checks(
    states: np.ndarray,
    parameters: BalloonParameters,
) -> dict[str, Any]:
    values = np.asarray(states, dtype=np.float64)
    checks: dict[str, Any] = {
        "finite": bool(np.all(np.isfinite(values))),
        "positive_fvpq": False,
        "oxygen_extraction_in_unit_interval": False,
        "absolute_hb_nonnegative": False,
        "hbr_not_above_hbt": False,
        "rest_equilibrium": False,
        "minimum_f": float("nan"),
        "minimum_v": float("nan"),
        "minimum_p": float("nan"),
        "minimum_q": float("nan"),
        "minimum_extraction": float("nan"),
        "maximum_extraction": float("nan"),
    }
    if values.ndim != 2 or values.shape[1] != _STATE_DIM or not checks["finite"]:
        return checks
    f, v, p, q = (values[:, index] for index in (2, 3, 4, 5))
    checks["minimum_f"] = float(np.min(f))
    checks["minimum_v"] = float(np.min(v))
    checks["minimum_p"] = float(np.min(p))
    checks["minimum_q"] = float(np.min(q))
    positive = (f > 0.0) & (v > 0.0) & (p > 0.0) & (q > 0.0)
    checks["positive_fvpq"] = bool(np.all(positive))
    if checks["positive_fvpq"]:
        extraction = np.asarray([_extraction(float(value), parameters.fixed.E0)[0] for value in f])
        checks["minimum_extraction"] = float(np.min(extraction))
        checks["maximum_extraction"] = float(np.max(extraction))
        checks["oxygen_extraction_in_unit_interval"] = bool(np.all((extraction > 0.0) & (extraction < 1.0)))
        hbt = parameters.fixed.P0 * p
        hbr = parameters.fixed.Q0 * q
        hbo = hbt - hbr
        checks["absolute_hb_nonnegative"] = bool(np.all((hbt >= 0.0) & (hbr >= 0.0) & (hbo >= 0.0)))
        checks["hbr_not_above_hbt"] = bool(np.all(hbr <= hbt + 1e-12))
    rest = physical_to_transformed((0.0, 0.0, 1.0, 1.0, 1.0, 1.0))
    checks["rest_equilibrium"] = bool(
        np.allclose(balloon_rhs(rest, parameters), 0.0, atol=1e-10)
    )
    return checks


def simulate_balloon(
    driver: Sequence[float] | np.ndarray,
    parameters: BalloonParameters | None = None,
    *,
    observation_spec: BalloonObservationSpec | None = None,
    config: BalloonConfig = BalloonConfig(),
    rng: np.random.Generator | None = None,
    add_noise: bool = True,
    observation_mask: np.ndarray | None = None,
) -> BalloonSimulation:
    """Generate known ``r`` truth, nonlinear Balloon states, and observations.

    The supplied ``driver`` is the known synthetic ``r(t)`` path.  The model
    still propagates the other compartments with the declared fixed dynamics;
    setting ``driver`` to an AR(1)-like path keeps the simulation and smoother
    priors aligned without hiding the driver from the truth arrays.
    """

    parameters = parameters or BalloonParameters()
    parameters.validate()
    config.validate()
    spec = (observation_spec or BalloonObservationSpec()).resolved(parameters.fixed)
    values = np.asarray(driver, dtype=np.float64).reshape(-1)
    if values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("driver must contain at least one finite value")
    transformed = physical_to_transformed((float(values[0]), 0.0, 1.0, 1.0, 1.0, 1.0))
    states = np.zeros((len(values), _STATE_DIM), dtype=np.float64)
    clean = np.zeros((len(values), _OBS_DIM), dtype=np.float64)
    for index, value in enumerate(values):
        transformed = transformed.copy()
        transformed[0] = float(value)
        states[index] = _transformed_to_physical_unchecked(transformed)
        clean[index] = _observation_map_unchecked(states[index], parameters, spec)
        if index + 1 < len(values):
            transformed = rk4_transition(transformed, parameters, config)
    generator = rng if rng is not None else np.random.default_rng(0)
    if add_noise:
        noise = generator.standard_t(
            df=float(spec.student_nu),
            size=clean.shape,
        ) * np.asarray(spec.observation_scale, dtype=np.float64)[None, :]
        observations = clean + noise
    else:
        observations = clean.copy()
    if observation_mask is None:
        mask = np.ones_like(observations, dtype=bool)
    else:
        mask = np.asarray(observation_mask, dtype=bool)
        if mask.shape != observations.shape:
            raise ValueError("observation_mask must match generated observations")
        observations = observations.copy()
        observations[~mask] = np.nan
    return BalloonSimulation(
        states=states,
        clean_observations=clean,
        observations=observations,
        observation_mask=mask,
    )


def smooth_balloon(
    observations: np.ndarray,
    parameters: BalloonParameters | None = None,
    *,
    observation_spec: BalloonObservationSpec | None = None,
    config: BalloonConfig = BalloonConfig(),
    observation_mask: np.ndarray | None = None,
) -> BalloonSmootherResult:
    """Run missing-aware Student-t IRLS EKF plus fixed-interval RTS smoothing."""

    parameters = parameters or BalloonParameters()
    parameters.validate()
    config.validate()
    spec = (observation_spec or BalloonObservationSpec()).resolved(parameters.fixed)
    values, mask = _validate_observation_inputs(observations, observation_mask)
    steps = values.shape[0]
    process_cov = _resolved_process_covariance(parameters, config)
    initial_mean = physical_to_transformed((0.0, 0.0, 1.0, 1.0, 1.0, 1.0))
    initial_cov = _project_psd(_initial_transformed_covariance(config))
    filtered_mean = np.zeros((steps, _STATE_DIM), dtype=np.float64)
    filtered_cov = np.zeros((steps, _STATE_DIM, _STATE_DIM), dtype=np.float64)
    predicted_mean = np.zeros_like(filtered_mean)
    predicted_cov = np.zeros_like(filtered_cov)
    transition_jacobians = np.zeros((steps, _STATE_DIM, _STATE_DIM), dtype=np.float64)
    filtered_mean_previous = initial_mean
    filtered_cov_previous = initial_cov
    log_likelihood = 0.0
    for index in range(steps):
        if index == 0:
            predicted_mean[index] = initial_mean
            predicted_cov[index] = initial_cov
            transition_jacobians[index] = np.eye(_STATE_DIM, dtype=np.float64)
        else:
            predicted_mean[index], transition_jacobians[index] = rk4_transition_with_jacobian(
                filtered_mean_previous,
                parameters,
                config,
            )
            predicted_cov[index] = _project_psd(
                transition_jacobians[index] @ filtered_cov_previous @ transition_jacobians[index].T
                + process_cov
            )
        filtered_mean[index], filtered_cov[index], step_log_likelihood = _observation_update(
            predicted_mean[index],
            predicted_cov[index],
            values[index],
            mask[index],
            parameters,
            spec,
            config,
        )
        filtered_cov[index] = _project_psd(filtered_cov[index])
        filtered_mean_previous = filtered_mean[index]
        filtered_cov_previous = filtered_cov[index]
        log_likelihood += float(step_log_likelihood)

    smoothed_mean = filtered_mean.copy()
    smoothed_cov = filtered_cov.copy()
    for index in range(steps - 2, -1, -1):
        predicted_next_cov = _project_psd(predicted_cov[index + 1])
        smoother_gain = filtered_cov[index] @ transition_jacobians[index + 1].T @ np.linalg.pinv(predicted_next_cov)
        smoothed_mean[index] = filtered_mean[index] + smoother_gain @ (
            smoothed_mean[index + 1] - predicted_mean[index + 1]
        )
        smoothed_cov[index] = _project_psd(
            filtered_cov[index]
            + smoother_gain @ (smoothed_cov[index + 1] - predicted_next_cov) @ smoother_gain.T
        )

    physical_moments = [
        transformed_gaussian_moments(smoothed_mean[index], smoothed_cov[index])
        for index in range(steps)
    ]
    state_mean = np.vstack([item[0] for item in physical_moments])
    state_covariance = np.stack([item[1] for item in physical_moments], axis=0)
    # Report exact log-normal marginal variances in physical coordinates,
    # rather than the plug-in/delta variance of exp(log f), etc.
    state_variance = np.maximum(
        np.diagonal(state_covariance, axis1=1, axis2=2),
        0.0,
    )
    observation_mean = np.vstack(
        [_observation_map_unchecked(row, parameters, spec) for row in state_mean]
    )
    epistemic_variance = np.zeros((steps, _OBS_DIM), dtype=np.float64)
    observation_matrix = _observation_physical_matrix(parameters, spec)
    for index in range(steps):
        epistemic_variance[index] = np.maximum(
            np.diag(observation_matrix @ state_covariance[index] @ observation_matrix.T),
            0.0,
        )
    scales = np.asarray(spec.observation_scale, dtype=np.float64)
    aleatoric_variance = np.broadcast_to(
        np.square(scales) * float(spec.student_nu) / (float(spec.student_nu) - 2.0),
        epistemic_variance.shape,
    ).copy()
    total_variance = aleatoric_variance + epistemic_variance
    observation_residual = values - observation_mean
    observation_residual[~mask] = np.nan
    teacher_valid = np.all(np.isfinite(state_mean), axis=1)
    uncertainty_valid = np.isfinite(total_variance) & (total_variance >= 0.0)
    trajectory_valid = mask & np.isfinite(observation_mean)
    observation_residual_valid = trajectory_valid & uncertainty_valid
    checks = _physical_checks(state_mean, parameters)
    return BalloonSmootherResult(
        state_names=STATE_NAMES,
        observation_names=OBSERVATION_NAMES,
        state_mean=state_mean,
        state_variance=state_variance,
        observation_mean=observation_mean,
        trajectory_mean=observation_mean.copy(),
        aleatoric_variance=aleatoric_variance,
        epistemic_variance=epistemic_variance,
        total_variance=total_variance,
        observation_residual=observation_residual,
        observation_mask=mask,
        teacher_valid_mask=np.broadcast_to(teacher_valid[:, None], (steps, _STATE_DIM)).copy(),
        uncertainty_valid_mask=uncertainty_valid,
        trajectory_valid_mask=trajectory_valid,
        observation_residual_valid_mask=observation_residual_valid,
        predictive_log_likelihood=float(log_likelihood),
        physical_checks=checks,
        parameters=parameters,
    )


def _parameters_from_vector(values: Sequence[float], fixed: BalloonFixedParameters) -> BalloonParameters:
    vector = _as_vector(values, 2, "free parameter vector")
    return BalloonParameters(fixed=fixed, free=BalloonFreeParameters(kappa=float(vector[0]), tau=float(vector[1])))


def _objective(
    vector: Sequence[float],
    observations: np.ndarray,
    fixed: BalloonFixedParameters,
    spec: BalloonObservationSpec,
    config: BalloonConfig,
    mask: np.ndarray | None,
    *,
    include_prior: bool = True,
) -> float:
    try:
        parameters = _parameters_from_vector(vector, fixed)
        result = smooth_balloon(
            observations,
            parameters,
            observation_spec=spec,
            config=config,
            observation_mask=mask,
        )
        if not np.isfinite(result.predictive_log_likelihood):
            return 1.0e12
        if not include_prior:
            return -float(result.predictive_log_likelihood)
        kappa, tau = float(vector[0]), float(vector[1])
        prior_penalty = 0.5 * (
            ((kappa - config.kappa_prior_mean) / config.kappa_prior_sd) ** 2
            + ((tau - config.tau_prior_mean) / config.tau_prior_sd) ** 2
        )
        return -float(result.predictive_log_likelihood) + prior_penalty
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return 1.0e12


def _finite_hessian(
    optimum: np.ndarray,
    objective: Any,
    step: float,
) -> np.ndarray:
    dimension = len(optimum)
    hessian = np.zeros((dimension, dimension), dtype=np.float64)
    base = float(objective(optimum))
    for index in range(dimension):
        delta = np.zeros(dimension, dtype=np.float64)
        delta[index] = float(step) * max(1.0, abs(float(optimum[index])))
        plus = float(objective(optimum + delta))
        minus = float(objective(optimum - delta))
        hessian[index, index] = (plus - 2.0 * base + minus) / max(delta[index] ** 2, 1e-12)
        for other in range(index):
            delta_other = np.zeros(dimension, dtype=np.float64)
            delta_other[other] = float(step) * max(1.0, abs(float(optimum[other])))
            pp = float(objective(optimum + delta + delta_other))
            pm = float(objective(optimum + delta - delta_other))
            mp = float(objective(optimum - delta + delta_other))
            mm = float(objective(optimum - delta - delta_other))
            denominator = max(4.0 * abs(delta[index] * delta_other[other]), 1e-12)
            value = (pp - pm - mp + mm) / denominator
            hessian[index, other] = value
            hessian[other, index] = value
    return (hessian + hessian.T) * 0.5


def fit_balloon(
    observations: np.ndarray,
    *,
    fixed: BalloonFixedParameters | None = None,
    observation_spec: BalloonObservationSpec | None = None,
    config: BalloonConfig = BalloonConfig(),
    observation_mask: np.ndarray | None = None,
    starts: Sequence[Sequence[float]] | None = None,
) -> BalloonFit:
    """Fit only ``kappa`` and ``tau`` with bounded multi-start L-BFGS-B."""

    fixed = fixed or BalloonFixedParameters()
    fixed.validate()
    config.validate()
    spec = (observation_spec or BalloonObservationSpec()).resolved(fixed)
    values, mask = _validate_observation_inputs(observations, observation_mask)
    valid_counts = np.sum(mask, axis=0)
    if np.any(valid_counts < 2):
        raise ValueError(
            "fit requires at least two finite observations in each EEG/HbO/HbR coordinate"
        )
    del values
    bounds = (tuple(config.kappa_bounds), tuple(config.tau_bounds))
    candidate_starts = tuple(tuple(float(item) for item in start) for start in (starts or config.optimizer_starts))
    if not candidate_starts:
        raise ValueError("at least one optimizer start is required")
    objective = lambda vector: _objective(
        vector, observations, fixed, spec, config, mask, include_prior=True
    )
    likelihood_objective = lambda vector: _objective(
        vector, observations, fixed, spec, config, mask, include_prior=False
    )
    records: list[Mapping[str, Any]] = []
    best_result: Any | None = None
    for start in candidate_starts:
        if len(start) != 2:
            raise ValueError("every optimizer start must contain kappa and tau")
        clipped = np.asarray(
            [
                np.clip(start[0], bounds[0][0], bounds[0][1]),
                np.clip(start[1], bounds[1][0], bounds[1][1]),
            ],
            dtype=np.float64,
        )
        result = minimize(
            objective,
            clipped,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(config.optimizer_max_iterations), "ftol": 1e-8},
        )
        records.append(
            {
                "start": tuple(float(item) for item in clipped),
                "estimate": tuple(float(item) for item in np.asarray(result.x, dtype=np.float64)),
                "objective": float(result.fun),
                "success": bool(result.success),
                "message": str(result.message),
            }
        )
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result
    if best_result is None or not np.all(np.isfinite(best_result.x)):
        raise RuntimeError("Balloon parameter optimization produced no finite result")
    optimum = np.asarray(best_result.x, dtype=np.float64)
    parameters = _parameters_from_vector(optimum, fixed)
    hessian = _finite_hessian(optimum, objective, float(config.hessian_step))
    likelihood_hessian = _finite_hessian(
        optimum, likelihood_objective, float(config.hessian_step)
    )
    try:
        eigenvalues = np.linalg.eigvalsh(hessian)
        curvature_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        positive_curvature = bool(np.all(eigenvalues > 1e-6 * curvature_scale))
        covariance = np.linalg.pinv(hessian) if positive_curvature else np.full((2, 2), np.nan)
        finite_covariance = bool(np.all(np.isfinite(covariance)))
    except np.linalg.LinAlgError:
        positive_curvature = False
        finite_covariance = False
        covariance = np.full((2, 2), np.nan, dtype=np.float64)
    tolerance = 1e-4
    boundary = (
        np.isclose(optimum[0], bounds[0][0], atol=tolerance)
        or np.isclose(optimum[0], bounds[0][1], atol=tolerance)
        or np.isclose(optimum[1], bounds[1][0], atol=tolerance)
        or np.isclose(optimum[1], bounds[1][1], atol=tolerance)
    )
    if boundary or not positive_curvature or not finite_covariance:
        covariance = np.full((2, 2), np.nan, dtype=np.float64)
    # Identifiability is a data property, not a consequence of adding the
    # Gaussian regularizer.  Check the likelihood-only Hessian and whether
    # the posterior actually contracts relative to the declared prior.
    try:
        data_eigenvalues = np.linalg.eigvalsh(likelihood_hessian)
        data_scale = max(1.0, float(np.max(np.abs(data_eigenvalues))))
        data_positive = bool(
            np.all(data_eigenvalues > 1e-6 * data_scale)
            and np.all(np.isfinite(data_eigenvalues))
        )
        data_condition = float(np.max(data_eigenvalues) / np.min(data_eigenvalues))
        data_well_conditioned = bool(np.isfinite(data_condition) and data_condition < 1e8)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        data_positive = False
        data_well_conditioned = False
    prior_precision = np.diag(
        [1.0 / config.kappa_prior_sd**2, 1.0 / config.tau_prior_sd**2]
    )
    prior_variance = np.diag(np.linalg.pinv(prior_precision))
    posterior_shrinkage = False
    if np.all(np.isfinite(covariance)):
        posterior_variance = np.maximum(np.diag(covariance), 0.0)
        posterior_shrinkage = bool(np.all(posterior_variance < 0.99 * prior_variance))
    if boundary or not positive_curvature or not finite_covariance or not data_positive or not data_well_conditioned:
        identifiability = "UNIDENTIFIABLE"
    elif not posterior_shrinkage:
        identifiability = "PRIOR_DOMINATED"
    else:
        identifiability = "IDENTIFIABLE"
    return BalloonFit(
        parameters=parameters,
        objective=float(best_result.fun),
        optimizer_success=bool(best_result.success),
        optimizer_message=str(best_result.message),
        starts=tuple(records),
        hessian=hessian,
        parameter_covariance=covariance,
        identifiability_status=identifiability,
        boundary_status="BOUNDARY" if boundary else "INTERIOR",
        likelihood_hessian=likelihood_hessian,
    )


def run_physical_checks(states: np.ndarray, parameters: BalloonParameters | None = None) -> Mapping[str, Any]:
    """Public physical validity report for known or inferred trajectories."""

    parameters = parameters or BalloonParameters()
    parameters.validate()
    return _physical_checks(np.asarray(states, dtype=np.float64), parameters)


__all__ = [
    "BalloonConfig",
    "BalloonFit",
    "BalloonFixedParameters",
    "BalloonFreeParameters",
    "BalloonObservationSpec",
    "BalloonParameters",
    "BalloonSimulation",
    "BalloonSmootherResult",
    "OBSERVATION_NAMES",
    "STATE_NAMES",
    "balloon_rhs",
    "balloon_rhs_jacobian",
    "fit_balloon",
    "observation_jacobian",
    "observation_map",
    "physical_to_transformed",
    "rk4_transition",
    "rk4_transition_with_jacobian",
    "run_physical_checks",
    "simulate_balloon",
    "smooth_balloon",
    "student_t_irls_weights",
    "transformed_gaussian_moments",
    "transformed_to_physical",
]
