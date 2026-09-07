"""Versioned joint-observation Gaussian assumed-density Balloon inference.

The immutable T3a module owns the physiological dynamics. This alternative
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
                            parameters.fixed.P0*(p-1)-hbr, hbr))


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
    scales = np.asarray(spec.observation_scale)[available]
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
    observation_var = np.square(spec.observation_scale)*spec.student_nu/(spec.student_nu-2)
    return JointBalloonResult(mean,covariance,state_mean,state_cov,trajectory,clean_var,
                             observation_var,mask,float(ll),float(score),
                             core._physical_checks(state_mean,parameters))
