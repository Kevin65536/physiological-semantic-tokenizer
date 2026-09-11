"""Exploratory nonlinear six-state MAP with a mask-specific Gaussian time map.

The physiological transition and its analytic derivative remain owned by T3a.
No path variance, marginal likelihood or Student-t calibration is claimed.
Domain-invalid trial steps are rejected, never clipped; every evaluation counts
towards the per-start budget. Observation derivatives are dense in time.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from . import t3a_balloon_robust_ssm as core
from . import t3a_balloon_joint_ssm as joint


class TrajectoryObjective:
    def __init__(self, observations, parameters, config, trajectory_spec,
                 noise_variance, *, input_mask=None, observation_spec=None,
                 rank_rtol=1e-10, linear=False, noise_factor=None):
        parameters.validate()
        config.validate()
        if not 0 < rank_rtol < 1:
            raise ValueError('invalid SVD tolerance')
        self.p, self.c, self.linear = parameters, config, linear
        self.spec = (observation_spec or core.BalloonObservationSpec()).resolved(parameters.fixed)
        self.steps = len(trajectory_spec.input_time)
        self.matrix, self.available = trajectory_spec.matrix(input_mask)
        y = np.asarray(observations, dtype=float)
        variance = np.asarray(noise_variance, dtype=float) if noise_factor is None else None
        if noise_factor is not None and noise_variance is not None:
            raise ValueError('use one noise owner: noise_variance or explicit noise_factor')
        if (y.shape != self.available.shape or np.isinf(y).any() or
                (noise_factor is None and (variance.shape != (3,) or
                 not np.isfinite(variance).all() or np.any(variance <= 0)))):
            raise ValueError('invalid processed observations or pre-processing noise variance')
        self.available &= np.isfinite(y)
        self.rank_rtol = rank_rtol
        if noise_factor is not None:
            system = joint.whiten_trajectory_observations(y, self.matrix, self.available, noise_factor, rank_rtol)
            self.rank, self.discarded = system['rank'], system['discarded_singular_values']
            self.singular, self.support_error = system['singular_values'], system['support_residual_norm']
            self.observation_design, self.target = system['design'], system['target']
            self.observation_normalization = system['normalization']
        else:
            selected = self.matrix[self.available.ravel()]
            u, singular, vt = np.linalg.svd(selected, full_matrices=False)
            keep = singular > rank_rtol*singular[0] if len(singular) else np.zeros(0, bool)
            self.rank = int(keep.sum())
            self.discarded = singular[~keep]
            self.singular = singular
            target = y[self.available]
            self.support_error = float(np.linalg.norm(target-u[:, keep]@(u[:, keep].T@target)))
            if self.support_error > 100*rank_rtol*max(1., np.linalg.norm(target)):
                raise ValueError('processed observations outside retained operator support')
            r = np.tile(variance*np.square(self.spec.coordinate_scale), self.steps)
            basis = vt[keep]
            if self.rank:
                chol = np.linalg.cholesky((basis*r)@basis.T)
                self.observation_design = solve_triangular(chol, basis, lower=True)
                self.target = solve_triangular(chol, (u[:, keep].T@target)/singular[keep], lower=True)
                self.observation_normalization = float(
                    self.rank*np.log(2*np.pi)/2 + np.log(np.diag(chol)).sum() + np.log(singular[keep]).sum())
            else:
                self.observation_design = np.empty((0, 3*self.steps))
                self.target = np.empty(0)
                self.observation_normalization = 0.
        self.initial_sd = np.asarray(config.initial_state_std)
        self.process_sd = np.asarray(parameters.fixed.process_std)*np.sqrt(config.dt)
        if np.any(self.process_sd <= 0):
            raise ValueError('MAP path density requires strictly positive process noise')
        self.prior_normalization = float(
            3*self.steps*np.log(2*np.pi) + np.log(self.initial_sd).sum() +
            (self.steps-1)*np.log(self.process_sd).sum())
        self.transition = core.rk4_transition_with_jacobian(np.zeros(6), parameters, config)[1]
        self.h = core.observation_jacobian(np.zeros(6), parameters, self.spec)
        self.offset = core.observation_map(np.array([0., 0., 1., 1., 1., 1.]), parameters, self.spec)

    def evaluate(self, flat, *, derivative=True):
        z = np.asarray(flat, dtype=float).reshape(self.steps, 6)
        if not np.isfinite(z).all():
            raise FloatingPointError('nonfinite path')
        residual = np.empty((self.steps, 6))
        residual[0] = z[0]/self.initial_sd
        jac = np.zeros((6*self.steps+self.rank, 6*self.steps)) if derivative else None
        if derivative:
            jac[:6, :6] = np.diag(1/self.initial_sd)
        for t in range(1, self.steps):
            if self.linear:
                predicted, transition = self.transition@z[t-1], self.transition
            elif derivative:
                predicted, transition = core.rk4_transition_with_jacobian(z[t-1], self.p, self.c)
            else:
                predicted = core.rk4_transition(z[t-1], self.p, self.c)
            residual[t] = (z[t]-predicted)/self.process_sd
            if derivative:
                jac[6*t:6*t+6, 6*t-6:6*t] = -transition/self.process_sd[:, None]
                jac[6*t:6*t+6, 6*t:6*t+6] = np.diag(1/self.process_sd)
        if self.linear:
            clean = z@self.h.T+self.offset
            local = np.broadcast_to(self.h, (self.steps, 3, 6))
        else:
            physical = np.array([core.transformed_to_physical(row) for row in z])
            clean = np.array([core._observation_map_unchecked(row, self.p, self.spec) for row in physical])
            if derivative:
                local = np.array([core._observation_jacobian_unchecked(row, self.p, self.spec) for row in z])
        if derivative:
            # A_M mixes every visible time: no false sparse observation pattern.
            for t in range(self.steps):
                jac[6*self.steps:, 6*t:6*t+6] = self.observation_design[:, 3*t:3*t+3]@local[t]
        vector = np.concatenate((residual.ravel(), self.observation_design@clean.ravel()-self.target))
        if not np.isfinite(vector).all() or (derivative and not np.isfinite(jac).all()):
            raise FloatingPointError('nonfinite objective or derivative')
        return vector, jac


def _optimize(objective, initial, max_nfev):
    """Damped Gauss--Newton, with explicit domain rejection and call accounting."""
    x = np.asarray(initial, dtype=float).ravel().copy()
    calls, derivatives, rejected, iterations = 0, 0, [], 0
    convergence = False
    reason = 'evaluation_budget'
    damping = 1e-7
    final_cost = None
    while calls < max_nfev:
        calls += 1
        derivatives += 1
        try:
            r, jac = objective.evaluate(x)
        except (FloatingPointError, ValueError, OverflowError) as exc:
            return dict(status='infeasible_initial' if iterations == 0 else 'failed_numerical',
                        error=repr(exc), evaluations=calls, jacobian_evaluations=derivatives,
                        rejected_steps=rejected, converged=False)
        cost = float(r@r/2)
        final_cost = cost
        gradient = jac.T@r
        if np.max(np.abs(gradient)) < 1e-7:
            convergence, reason = True, 'gradient'
            break
        gram = jac.T@jac
        diagonal = np.maximum(np.diag(gram), 1.)
        accepted = False
        while calls < max_nfev:
            iterations += 1
            try:
                delta = cho_solve(cho_factor(gram+np.diag(diagonal*damping), lower=True), -gradient)
            except np.linalg.LinAlgError:
                damping *= 10
                if damping > 1e16:
                    reason = 'singular_step'
                    break
                continue
            calls += 1
            try:
                trial_residual, _ = objective.evaluate(x+delta, derivative=False)
                next_cost = float(trial_residual@trial_residual/2)
            except (FloatingPointError, ValueError, OverflowError) as exc:
                rejected.append(dict(evaluation=calls, reason=repr(exc), kind='domain_trial_step'))
                damping *= 10
                continue
            if next_cost < cost or (np.linalg.norm(delta) < 1e-11 and next_cost <= cost+1e-12):
                x += delta
                final_cost = next_cost
                damping = max(damping/5, 1e-14)
                accepted = True
                if (np.linalg.norm(delta) <= 1e-9*(1+np.linalg.norm(x)) or
                        abs(cost-next_cost) <= 1e-11*(1+cost)):
                    convergence, reason = True, 'step_or_objective'
                break
            rejected.append(dict(evaluation=calls, kind='nondecreasing_objective'))
            damping *= 10
        if convergence or not accepted:
            break
    return dict(status='completed' if convergence else 'failed_numerical',
                transformed_mean=x.reshape(objective.steps, 6), objective=final_cost,
                converged=convergence, convergence_reason=reason, evaluations=calls,
                jacobian_evaluations=derivatives, rejected_steps=rejected, iterations=iterations)


def smooth_balloon_trajectory_map(observations, parameters, *, config,
        trajectory_spec, noise_variance, observation_spec=None, input_mask=None,
        rank_rtol=1e-10, max_nfev=200, linear=False, noise_factor=None):
    if not 1 <= max_nfev <= 200:
        raise ValueError('max_nfev must be in [1,200] per fixed start')
    objective = TrajectoryObjective(observations, parameters, config, trajectory_spec,
        noise_variance, observation_spec=observation_spec, input_mask=input_mask,
        rank_rtol=rank_rtol, linear=linear, noise_factor=noise_factor)
    starts = [('rest', np.zeros((objective.steps, 6)))]
    initial_failure = None
    try:
        reference = joint.smooth_balloon_trajectory_reference(observations, parameters,
            config=config, trajectory_spec=trajectory_spec, noise_variance=noise_variance,
            observation_spec=observation_spec, input_mask=input_mask, rank_rtol=rank_rtol,
            noise_factor=noise_factor)
        starts.append(('linear_reference', reference['transformed_mean']))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        initial_failure = dict(start='linear_reference', status='infeasible_initial',
                               error=repr(exc), evaluations=0, converged=False)
    attempts = [dict(start=name, **_optimize(objective, initial, max_nfev)) for name, initial in starts]
    if initial_failure:
        attempts.append(initial_failure)
    complete = [a for a in attempts if a['converged']]
    result = dict(status='completed' if complete else 'failed_numerical',
        starts=[{k: v for k, v in a.items() if k != 'transformed_mean'} for a in attempts],
        retained_rank=objective.rank, observed_coordinates=int(objective.available.sum()),
        discarded_singular_values=objective.discarded, rank_rtol=rank_rtol,
        support_residual_norm=objective.support_error,
        observation_density_normalization=objective.observation_normalization,
        prior_density_normalization=objective.prior_normalization,
        uncertainty='NOT_ESTIMATED', parameter_interval='NOT_ESTIMATED',
        marginal_likelihood='NOT_ESTIMATED',
        approximation='nonlinear correlated Gaussian batch MAP' if not linear else 'linear Gaussian MAP special case')
    if not complete:
        return result
    best = min(complete, key=lambda a: a['objective'])
    z = best['transformed_mean']
    physical = np.array([core.transformed_to_physical(row) for row in z])
    clean = (z@objective.h.T+objective.offset if linear else
             np.array([core._observation_map_unchecked(row, parameters, objective.spec) for row in physical]))
    result.update(transformed_mean=z, state_mean=physical, canonical_clean_mean=clean,
        processed_clean_mean=(objective.matrix@clean.ravel()).reshape(np.asarray(observations).shape),
        objective=best['objective'], selected_start=best['start'],
        observation_mask=objective.available, physical_checks=core.run_physical_checks(physical, parameters),
        negative_joint_log_density=best['objective']+objective.observation_normalization+objective.prior_normalization,
        converged_start_max_path_difference=(float(np.max(np.abs(complete[0]['transformed_mean']-
            complete[1]['transformed_mean']))) if len(complete) == 2 else None))
    return result
