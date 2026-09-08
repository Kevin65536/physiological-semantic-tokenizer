"""Versioned joint-observation Gaussian assumed-density Balloon inference.

The T3a module owns the physiological dynamics. This alternative
integrates the *joint* conditional Student-t density under each predicted
Gaussian state using Gauss-Hermite quadrature, and moment-matches the update.
It avoids the old independent-marginal score and IRLS curvature covariance.

The resulting parameter_log_likelihood still approximates the target SSM:
EKF transition propagation and a Gaussian filtering closure remain. Numerical
quadrature convergence, particle references, and matched-model calibration
must be reported; this module does not claim an exact posterior.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np
from scipy.special import logsumexp, poch, roots_hermitenorm
from scipy.linalg import cho_factor, cho_solve

from . import t3a_balloon_robust_ssm as core


@dataclass(frozen=True)
class JointBalloonResult:
    transformed_mean: np.ndarray
    transformed_covariance: np.ndarray
    state_mean: np.ndarray
    state_covariance: np.ndarray
    trajectory_mean: np.ndarray
    state_posterior_variance: np.ndarray
    observation_variance: np.ndarray
    observation_mask: np.ndarray
    parameter_log_likelihood: float
    predictive_score: float
    physical_checks: dict
    inference_method: str = 'joint Student-t quadrature / Gaussian moment matching / EKF-RTS'

    @property
    def total_observation_variance(self):
        return self.state_posterior_variance + self.observation_variance


@lru_cache(maxsize=16)
def normal_quadrature(order: int, dimension: int):
    if order < 3 or order % 2 != 1 or dimension not in (1, 2, 3):
        raise ValueError('quadrature requires odd order >= 3 and 1--3 dimensions')
    nodes, weights = roots_hermitenorm(order)
    weights /= np.sqrt(2*np.pi)
    indices = np.array(list(product(range(order), repeat=dimension)))
    points = nodes[indices]
    log_weights = np.log(weights[indices]).sum(axis=1)
    return points, log_weights


def _batch_observation(z, parameters, spec):
    with np.errstate(over='raise', invalid='raise'):
        p, q = np.exp(z[:, 4]), np.exp(z[:, 5])
    hbr = parameters.fixed.Q0*(q-1)
    return np.column_stack((spec.eeg_loading*z[:, 0]+spec.eeg_offset,
                            parameters.fixed.P0*(p-1)-hbr, hbr)) * spec.coordinate_scale


def joint_observation_update(mean, covariance, observation, available, parameters,
                             spec, order=13, calculate_score=True):
    """Joint predictive integral and first two exact quadrature update moments.

    Only r/log p/log q enter the observation. The remaining Gaussian state
    coordinates are integrated analytically by conditional regression.
    """
    available = np.asarray(available, dtype=bool)
    if not available.any():
        return mean.copy(), covariance.copy(), 0., 0.
    active = set()
    if available[0]:
        active.add(0)
    if available[1]:
        active.update((4, 5))
    if available[2]:
        active.add(5)
    active = sorted(active)
    c_active = covariance[np.ix_(active, active)]
    chol = np.linalg.cholesky(c_active)
    points, prior_log_weights = normal_quadrature(int(order), len(active))
    active_nodes = mean[active]+points@chol.T
    z = np.broadcast_to(mean, (len(points), 6)).copy()
    z[:, active] = active_nodes
    scales = spec.effective_noise_scale[available]
    nu = float(spec.student_nu)
    predictions = _batch_observation(z, parameters, spec)[:, available]
    residual = (np.asarray(observation)[available]-predictions)/scales
    constant = np.log(poch(nu/2, .5))-.5*np.log(nu*np.pi)-np.log(scales)
    log_density = (constant-(nu+1)/2*np.log1p(residual**2/nu)).sum(axis=1)
    log_likelihood = float(logsumexp(prior_log_weights+log_density))
    weights = np.exp(prior_log_weights+log_density-log_likelihood)
    active_mean = weights@active_nodes
    centered = active_nodes-active_mean
    active_covariance = (centered*weights[:, None]).T@centered
    regression = np.linalg.solve(c_active, covariance[active, :]).T
    updated_mean = mean+regression@(active_mean-mean[active])
    updated_covariance = core._project_psd(
        covariance + regression@(active_covariance-c_active)@regression.T)
    score = 0.
    if calculate_score:
        physical_mean, physical_cov = core.transformed_gaussian_moments(mean, covariance)
        h = core._observation_physical_matrix(parameters, spec)
        marginal_var = np.diag(h@physical_cov@h.T)[available]
        marginal_scale = np.sqrt(scales**2+marginal_var*(nu-2)/nu)
        center = core._observation_map_unchecked(physical_mean, parameters, spec)[available]
        res = (np.asarray(observation)[available]-center)/marginal_scale
        constant = np.log(poch(nu/2, .5))-.5*np.log(nu*np.pi)-np.log(marginal_scale)
        score = float(np.sum(constant-(nu+1)/2*np.log1p(res**2/nu)))
    return updated_mean, updated_covariance, log_likelihood, score


def _filter(observations, parameters, config, spec, order, keep_states):
    values = np.asarray(observations, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 1:
        raise ValueError('observations must have shape [time,3]')
    if np.isinf(values).any():
        raise ValueError('infinite observations are invalid; missing values use NaN')
    parameters.validate()
    config.validate()
    normal_quadrature(int(order), 1)
    mask = np.isfinite(values)
    mean = np.zeros(6)
    covariance = np.diag(np.square(config.initial_state_std))
    q = np.diag(np.square(parameters.fixed.process_std))*config.dt
    ll, score = 0., 0.
    filtered_mean, filtered_cov, predicted_mean, predicted_cov, transitions = [], [], [], [], []
    for t, observation in enumerate(values):
        a = np.eye(6)
        if t:
            mean, a = core.rk4_transition_with_jacobian(mean, parameters, config)
            covariance = core._project_psd(a@covariance@a.T+q)
        if keep_states:
            predicted_mean.append(mean.copy())
            predicted_cov.append(covariance.copy())
            transitions.append(a)
        mean, covariance, log_likelihood, marginal_score = joint_observation_update(
            mean, covariance, observation, mask[t], parameters, spec, order, keep_states)
        ll += log_likelihood
        score += marginal_score
        if keep_states:
            filtered_mean.append(mean.copy())
            filtered_cov.append(covariance.copy())
    return (ll, score, np.array(filtered_mean), np.array(filtered_cov),
            np.array(predicted_mean), np.array(predicted_cov), np.array(transitions), mask)


def parameter_log_likelihood(observations, parameters, *, config=core.BalloonConfig(),
                             observation_spec=None, quadrature_order=13):
    """Approximate joint SSM likelihood, not the old marginal predictive score."""
    spec = (observation_spec or core.BalloonObservationSpec()).resolved(parameters.fixed)
    return float(_filter(observations, parameters, config, spec, quadrature_order, False)[0])


def smooth_balloon_joint(observations, parameters, *, config=core.BalloonConfig(),
                         observation_spec=None, quadrature_order=13):
    spec = (observation_spec or core.BalloonObservationSpec()).resolved(parameters.fixed)
    ll, score, fm, fc, pm, pc, a, mask = _filter(
        observations, parameters, config, spec, quadrature_order, True)
    mean, covariance = fm.copy(), fc.copy()
    for t in range(len(fm)-2, -1, -1):
        gain = np.linalg.solve(pc[t+1], a[t+1]@fc[t]).T
        mean[t] += gain@(mean[t+1]-pm[t+1])
        covariance[t] = core._project_psd(fc[t]+gain@(covariance[t+1]-pc[t+1])@gain.T)
    moments = [core.transformed_gaussian_moments(m,c) for m,c in zip(mean,covariance)]
    state_mean = np.array([v[0] for v in moments])
    state_cov = np.array([v[1] for v in moments])
    h = core._observation_physical_matrix(parameters,spec)
    trajectory = np.array([core._observation_map_unchecked(m,parameters,spec) for m in state_mean])
    clean_var = np.einsum('ij,tjk,ik->ti',h,state_cov,h)
    observation_var = np.square(spec.effective_noise_scale)*spec.student_nu/(spec.student_nu-2)
    return JointBalloonResult(mean,covariance,state_mean,state_cov,trajectory,clean_var,
                             observation_var,mask,float(ll),float(score),
                             core._physical_checks(state_mean,parameters))


@dataclass(frozen=True)
class TrajectoryObservationSpec:
    """Explicit finite-window temporal map, separately for EEG/HbO/HbR.

    ``operators[m]`` maps the entire input clock to the output clock, in actual
    processing order (including boundary handling). ``processing`` records that
    order, baseline weights, filter/resampling choices and units at construction.
    Known diagonal unit changes remain owned by BalloonObservationSpec.

    Masks refer to input samples *before* time mixing: any output depending on
    a hidden input is unavailable. This conservative reference never fills a
    hidden target and never treats filtering as a pointwise observation map.
    """
    operators: np.ndarray  # [3, output_time, input_time]
    input_time: np.ndarray
    output_time: np.ndarray
    processing: tuple[dict, ...]

    def matrix(self, input_mask=None):
        operators = np.asarray(self.operators, dtype=float)
        if (operators.ndim != 3 or operators.shape[0] != 3
                or min(operators.shape[1:]) < 1 or not np.isfinite(operators).all()):
            raise ValueError('trajectory operators require finite [3,output_time,input_time]')
        nout, nin = operators.shape[1:]
        for clock, size in ((self.input_time, nin), (self.output_time, nout)):
            clock = np.asarray(clock)
            if clock.shape != (size,) or not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0):
                raise ValueError('trajectory clocks must be finite and strictly increasing')
        matrix = np.zeros((3*nout, 3*nin))
        for channel in range(3):
            matrix[channel::3, channel::3] = operators[channel]
        mask = np.ones((nin, 3), dtype=bool) if input_mask is None else np.asarray(input_mask, dtype=bool)
        if mask.shape != (nin, 3):
            raise ValueError('input mask must match the pre-processing clock and coordinates')
        # Exact dependency test; tiny nonzero filter coefficients still carry
        # target information and must not be thresholded into visibility.
        valid = ~np.any((matrix != 0) & ~mask.ravel()[None, :], axis=1)
        return matrix, valid.reshape(nout, 3)

    def apply(self, values, input_mask=None):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self.input_time), 3) or np.isinf(values).any():
            raise ValueError('trajectory values must match the input clock; missing uses NaN')
        mask = np.isfinite(values)
        if input_mask is not None:
            supplied = np.asarray(input_mask, dtype=bool)
            if supplied.shape != values.shape:
                raise ValueError('input mask must match values')
            mask &= supplied
        matrix, valid = self.matrix(mask)
        processed = (matrix @ np.where(mask, values, 0.).ravel()).reshape(valid.shape)
        processed[~valid] = np.nan
        return processed


def linearized_trajectory_prior(steps, parameters, config):
    """Full six-state Gaussian path covariance, linearized at resting state.

    This short-window reference has no nonlinear filtering closure. It is exact
    for the declared linearized transition, not for the nonlinear Balloon SSM.
    """
    parameters.validate(); config.validate()
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError('steps must be a positive integer')
    _, transition = core.rk4_transition_with_jacobian(np.zeros(6), parameters, config)
    covariance = np.zeros((steps, 6, steps, 6))
    covariance[0, :, 0, :] = np.diag(np.square(config.initial_state_std))
    q = np.diag(np.square(parameters.fixed.process_std))*config.dt
    for t in range(1, steps):
        for s in range(t):
            covariance[t, :, s, :] = transition @ covariance[t-1, :, s, :]
            covariance[s, :, t, :] = covariance[t, :, s, :].T
        covariance[t, :, t, :] = transition @ covariance[t-1, :, t-1, :] @ transition.T + q
    return covariance.reshape(6*steps, 6*steps)


def smooth_balloon_trajectory_reference(processed_observations, parameters, *,
        config, trajectory_spec, noise_variance, observation_spec=None,
        input_mask=None, rank_rtol=1e-10):
    """Batch reference with linearized physiology and explicitly Gaussian noise.

    ``noise_variance`` is the *pre-processing*, canonical independent Gaussian
    variance per coordinate. Supplying a Student-t marginal variance defines a
    moment approximation, not a transformed Student-t likelihood. Both mean and
    full noise covariance are transformed; no independent per-time t updates.

    SVD retains the observable subspace of the time operator. Discarded singular
    values and support residuals are returned: numerically truncated filtering
    is a reported approximation and cannot inherit earlier calibration. The
    density uses Euclidean volume on the retained output subspace. Irreversible
    preprocessing loses information; there is no invertible-Jacobian shortcut.
    """
    if not 0 < rank_rtol < 1:
        raise ValueError('rank_rtol must be strictly between zero and one')
    spec = (observation_spec or core.BalloonObservationSpec()).resolved(parameters.fixed)
    variance = np.asarray(noise_variance, dtype=float)
    if variance.shape != (3,) or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError('canonical Gaussian noise_variance must be three finite positive values')
    operator, available = trajectory_spec.matrix(input_mask)
    values = np.asarray(processed_observations, dtype=float)
    if values.shape != available.shape or np.isinf(values).any():
        raise ValueError('processed observations must match the output clock; missing uses NaN')
    available &= np.isfinite(values)
    selected = operator[available.ravel()]
    steps = len(trajectory_spec.input_time)
    prior = linearized_trajectory_prior(steps, parameters, config)
    canonical_spec = core.BalloonObservationSpec(
        eeg_loading=spec.eeg_loading, eeg_offset=spec.eeg_offset,
        observation_scale=spec.observation_scale, student_nu=spec.student_nu).resolved(parameters.fixed)
    canonical_h = np.kron(np.eye(steps), core.observation_jacobian(np.zeros(6), parameters, canonical_spec))
    canonical_offset = np.tile(core.observation_map(np.array([0., 0., 1., 1., 1., 1.]), parameters, canonical_spec), steps)
    scale = np.tile(spec.coordinate_scale, steps)
    h = scale[:, None] * canonical_h
    offset = scale * canonical_offset
    r = np.tile(variance, steps)*scale**2
    mean = np.zeros(steps*6)
    covariance = prior.copy()
    log_likelihood = 0.
    singular_values = np.array([])
    rank = 0
    support_error = 0.
    if len(selected):
        u, singular_values, vt = np.linalg.svd(selected, full_matrices=False)
        keep = singular_values > rank_rtol * singular_values[0]
        rank = int(keep.sum())
        residual = values[available] - selected @ offset
        support_error = float(np.linalg.norm(residual-u[:, keep]@(u[:, keep].T@residual)))
        support_tolerance = 100*rank_rtol*max(1., float(np.linalg.norm(residual)))
        if support_error > support_tolerance:
            raise ValueError('processed observations lie outside the retained operator support')
        if rank:
            # Divide out singular values first: working in the orthonormal
            # retained input subspace avoids squaring a filter's condition number.
            basis = vt[keep]
            target = (u[:, keep].T @ residual)/singular_values[keep]
            design = basis @ h
            noise = (basis*r) @ basis.T
            cross = prior @ design.T
            predictive = design @ cross + noise
            factor = cho_factor(predictive, lower=True)
            solved = cho_solve(factor, target)
            mean = cross @ solved
            covariance -= cross @ cho_solve(factor, cross.T)
            covariance = (covariance+covariance.T)/2
            log_likelihood = float(-.5*(rank*np.log(2*np.pi)
                + 2*np.log(np.diag(factor[0])).sum()+target@solved)
                - np.log(singular_values[keep]).sum())
    canonical_mean = canonical_offset + canonical_h @ mean
    canonical_covariance = canonical_h @ covariance @ canonical_h.T
    processed_h = operator @ h
    return dict(transformed_mean=mean.reshape(steps, 6), transformed_covariance=covariance,
        canonical_clean_mean=canonical_mean.reshape(steps, 3),
        canonical_clean_covariance=canonical_covariance,
        processed_clean_mean=(operator @ (scale*canonical_mean)).reshape(values.shape),
        processed_clean_covariance=processed_h @ covariance @ processed_h.T,
        processed_noise_covariance=(operator*r) @ operator.T,
        observation_mask=available, parameter_log_likelihood=log_likelihood,
        retained_rank=rank, observed_coordinates=int(available.sum()),
        operator_singular_values=singular_values, rank_rtol=rank_rtol,
        discarded_singular_values=singular_values[rank:], support_residual_norm=support_error,
        approximation='rest-linearized six-state Gaussian path and Gaussian observation noise; SVD subspace truncation')
