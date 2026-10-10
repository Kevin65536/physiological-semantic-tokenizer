"""Output-function profiles and matched-prior state-continuity diagnostics.

The physiology and RK4 tangents remain owned by shared_driver_rk4. This module
keeps every driver, free boundary state and structured remainder coefficient in
one explicit objective so an output profile can reoptimize *all* actual unknowns.
No feature SD is interpreted as calibrated sensor noise or posterior precision.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, svd

from .shared_driver_rk4 import compiled_forward
from .t3a_balloon_robust_ssm import BalloonParameters, BalloonFixedParameters, BalloonFreeParameters


OUTPUTS = ('SSM_Hb', 'T', 'X', 'volume', 'p_minus_v', 'flow', 'driver')


def reference_parameters(base):
    fixed = {k: v for k, v in base['fixed'].items() if k not in ('tau', 'kappa')}
    return BalloonParameters(fixed=BalloonFixedParameters(**fixed),
        free=BalloonFreeParameters(tau=base['fixed']['tau'], kappa=base['fixed']['kappa']))


def visibility(mode, steps=360, *, eeg=True, hb_context_steps=None):
    if steps != 360 or mode not in ('full', 'middle_Hb', 'middle_HbR'):
        raise ValueError('registered 90s three-component tensor and mask required')
    mask = np.ones((steps, 3), bool)
    if mode == 'middle_Hb':
        mask[120:240, 1:] = False
    elif mode == 'middle_HbR':
        mask[120:240, 2] = False
    if not eeg:
        mask[:, 0] = False
    if hb_context_steps is not None:
        if hb_context_steps not in (300, 360):
            raise ValueError('registered Hb context ends at 75 or 90 seconds')
        mask[hb_context_steps:, 1:] = False
    return mask


def hb_coordinates(hb, rho=.35):
    h = np.asarray(hb)
    return np.stack((h[..., 0]+h[..., 1], rho*h[..., 0]-(1-rho)*h[..., 1]), axis=-1)


def structured_basis(steps=360):
    """Frozen four cosine modes per ORIGINAL 30s block, independent of resets."""
    if steps != 360:
        raise ValueError('registered basis requires 360 samples')
    basis = np.zeros((steps, 3, 12))
    for block in range(3):
        wave = np.cos(np.pi*(np.arange(120)[:, None]+.5)*np.arange(1, 5)/120.)
        basis[120*block:120*(block+1), 1:, 4*block:4*(block+1)] = wave[:, None, :]*np.array([.65, .35])[None, :, None]
    return basis


class OutputProblem:
    """One legal SSM fit with an explicit visible-only residual contract."""

    def __init__(self, target, sd, operators, parameters, config, *, arm='chain_matched',
                 visible=None, boundaries=(120, 240), objective='total'):
        self.y = np.asarray(target, float)
        self.n = len(self.y)
        self.sd = np.asarray(sd, float)
        self.mask = np.isfinite(self.y) if visible is None else np.asarray(visible, bool)
        if (self.y.shape != (360, 3) or self.sd.shape != (3,) or self.mask.shape != self.y.shape
                or not np.isfinite(self.y[self.mask]).all() or not np.isfinite(self.sd).all()
                or np.any(self.sd <= 0) or not self.mask.any()):
            raise ValueError('finite visible [360,3] and positive training scales required')
        if arm not in ('free', 'chain_matched', 'chain_first') or objective not in ('data', 'total'):
            raise ValueError('unregistered state arm or objective')
        self.arm, self.objective = arm, objective
        self.cfg, self.par = config, parameters
        self.dt = config['tensor']['dt_s']
        self.boundaries = list(boundaries)
        if self.boundaries not in ([120, 240], [100, 220], [140, 260]):
            raise ValueError('unregistered state reset times')
        self.segments = list(zip([0]+self.boundaries, self.boundaries+[self.n])) if arm == 'free' else [(0, self.n)]
        self.initial_count = 5*len(self.segments)
        self.cstart = self.n+self.initial_count
        self.count = self.cstart+12
        self.op = [np.asarray(operators['eeg'], float), np.asarray(operators['hb'], float), np.asarray(operators['hb'], float)]
        if any(a.shape != (self.n, self.n) or not np.isfinite(a).all() for a in self.op):
            raise ValueError('finite shared observation operators required')
        basis = structured_basis(self.n)
        self.basis = np.stack([self.op[k]@basis[:, k] for k in range(3)], axis=1)
        self.csd = np.tile(config['observation']['coefficient_sd'], 3)
        curvature = np.diff(np.eye(self.n), n=2, axis=0)/self.dt**1.5
        for b in config['comparison']['fixed_driver_curvature_breaks']:
            curvature[b-2:b] = 0
        self.regularizer = np.zeros((len(curvature), self.count))
        self.regularizer[:, :self.n] = np.sqrt(config['solver']['penalty'])*curvature
        self.prior_times = [0] if arm == 'chain_first' else [0]+self.boundaries
        self.scale = np.tile(self.sd, (self.n, 1))
        self.variable_scales = np.r_[np.full(self.n, .025), np.full(self.initial_count, .02), self.csd]
        parameters.validate()

    def initial(self):
        return np.zeros(self.count)

    def evaluate(self, x, derivative=True):
        x = np.asarray(x, float)
        if x.shape != (self.count,) or not np.isfinite(x).all():
            raise ValueError('finite actual-unknown vector required')
        states = np.empty((self.n, 6))
        js = np.zeros((self.n, 6, self.count)) if derivative else None
        for k, (left, right) in enumerate(self.segments):
            loc = self.n+5*k
            with np.errstate(over='raise', invalid='raise'):
                initial = np.r_[x[loc], np.exp(x[loc+1:loc+5])]
            result = compiled_forward(x[left:right], initial, self.par, self.dt,
                self.cfg['solver']['substeps'], derivative, False, state=True)
            states[left:right] = result['states']
            if derivative:
                indices = np.r_[np.arange(left, right), np.arange(loc, loc+5)]
                js[left:right, :, indices] = result['state_jacobian']
        fixed = self.par.fixed
        canonical = np.column_stack((fixed.eeg_loading*states[:, 0],
            fixed.P0*(states[:, 4]-1)-fixed.Q0*(states[:, 5]-1), fixed.Q0*(states[:, 5]-1)))
        physical = np.column_stack([self.op[k]@canonical[:, k] for k in range(3)])
        component = np.einsum('tck,k->tc', self.basis, x[self.cstart:])
        prediction = physical+component
        data = (prediction[self.mask]-self.y[self.mask])/self.scale[self.mask]
        residuals = [data]
        physical_j = None
        jacobians = []
        if derivative:
            cj = np.stack((fixed.eeg_loading*js[:, 0], fixed.P0*js[:, 4]-fixed.Q0*js[:, 5], fixed.Q0*js[:, 5]), axis=1)
            physical_j = np.stack([self.op[k]@cj[:, k] for k in range(3)], axis=1)
            total_j = physical_j.copy()
            total_j[:, :, self.cstart:] += self.basis
            jacobians = [total_j[self.mask]/self.scale[self.mask, None]]
        costs = dict(data=float(data@data), driver=0., initial=0., flow=0., component=0.)
        if self.objective == 'total':
            sol = self.cfg['solver']
            initial_residual = np.sqrt(sol['initial_penalty'])*(states[self.prior_times, 1:]-np.r_[0., np.ones(4)])
            flow_factor = np.sqrt(sol['flow_prior_weight']*self.dt)/sol['flow_prior_log_sd']
            extra = [self.regularizer@x, initial_residual.ravel(),
                     flow_factor*np.log(states[:, 2]), x[self.cstart:]/self.csd]
            residuals += extra
            costs.update({name: float(v@v) for name, v in zip(('driver', 'initial', 'flow', 'component'), extra)})
            if derivative:
                jc = np.zeros((12, self.count)); jc[:, self.cstart:] = np.diag(1/self.csd)
                jacobians += [self.regularizer,
                    np.sqrt(sol['initial_penalty'])*js[self.prior_times, 1:].reshape(-1, self.count),
                    flow_factor*js[:, 2]/states[:, 2, None], jc]
        residual = np.concatenate(residuals)
        jac = np.vstack(jacobians) if derivative else None
        if not np.isfinite(residual).all() or (derivative and not np.isfinite(jac).all()):
            raise FloatingPointError('nonfinite legal objective')
        return dict(residual=residual, jacobian=jac, states=states, state_jacobian=js,
            physical=physical, physical_jacobian=physical_j, component=component,
            prediction=prediction, objective=float(residual@residual), costs=costs,
            data_rows=int(self.mask.sum()))

    def output(self, value, name, output_scales, *, score=slice(120, 240)):
        """Output in frozen reporting units, with physical/processed distinction."""
        st, j = value['states'], value['state_jacobian']
        h, jh = value['physical'][:, 1:], value['physical_jacobian']
        if name == 'SSM_Hb':
            v = h[score]/self.sd[1:]
            g = None if jh is None else jh[score, 1:]/self.sd[None, 1:, None]
        elif name in ('T', 'X'):
            weights = np.array([1., 1.]) if name == 'T' else np.array([.35, -.65])
            v = (h[score]@weights)/output_scales[name]
            g = None if jh is None else np.einsum('tck,c->tk', jh[score, 1:], weights)/output_scales[name]
        else:
            index = dict(volume=3, p_minus_v=4, flow=2, driver=0).get(name)
            if index is None:
                raise ValueError('unknown candidate output')
            unit = .025 if name == 'driver' else self.cfg['profiles']['state_output_unit']
            v = (st[score, index]-(st[score, 3] if name == 'p_minus_v' else 0. if name == 'driver' else 1.))/unit
            g = None if j is None else (j[score, index]-(j[score, 3] if name == 'p_minus_v' else 0.))/unit
        return v.ravel(), None if g is None else g.reshape(-1, self.count)


def fit_problem(problem, start=None, *, constraint=None, max_iterations=None):
    """Damped Gauss--Newton/SQP, backtracking only through legal RK4 solutions.

    A profile constraint is an exact scalar output projection. Every actual
    unknown is reoptimized, including all structured-component coefficients.
    Equality error and projected stationarity are both required for success.
    """
    cfg = problem.cfg['solver']
    x = problem.initial() if start is None else np.asarray(start, float).copy()
    evaluations, rejects, damping = 0, 0, 1e-5
    reason, converged, gradient = 'iteration_budget', False, np.inf
    limit = cfg['max_iterations'] if max_iterations is None else max_iterations
    current = problem.evaluate(x); evaluations += 1
    for iteration in range(limit+1):
        j, residual = current['jacobian'], current['residual']
        grad, gram = j.T@residual, j.T@j
        diagonal = np.maximum(np.diag(gram), 1.)
        error, a = (0., None) if constraint is None else constraint(current, True)
        if a is None:
            projected = grad
        else:
            denom = float((a/diagonal)@a)
            if denom < 1e-30:
                reason = 'constraint_locally_fixed'; break
            multiplier = float((a/diagonal)@grad)/denom
            projected = grad-multiplier*a
        gradient = float(np.max(np.abs(projected)/np.sqrt(diagonal)))
        if gradient <= cfg['gradient_tolerance'] and abs(error) <= cfg['constraint_tolerance']:
            converged, reason = True, 'stationary_feasible'; break
        if iteration == limit or evaluations+2 > cfg['max_evaluations']:
            break
        try:
            factor = cho_factor(gram+damping*np.diag(diagonal), lower=True, check_finite=False)
            delta = cho_solve(factor, -grad, check_finite=False)
            merit_weight = 100.
            if a is not None:
                normal = cho_solve(factor, a, check_finite=False)
                multiplier_step = float(a@delta+error)/float(a@normal)
                delta -= normal*multiplier_step
                merit_weight = max(merit_weight, 4*abs(multiplier_step))
        except (np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError):
            damping *= 10
            if damping > 1e16:
                reason = 'singular_step'; break
            continue
        merit = current['objective']+merit_weight*abs(error)
        accepted = False
        for power in range(20):
            if evaluations+2 > cfg['max_evaluations']:
                break
            step = 2.**(-power)
            trial_x = x+step*delta
            evaluations += 1
            try:
                candidate = problem.evaluate(trial_x, False)
                new_error = 0. if constraint is None else constraint(candidate, False)[0]
                trial_merit = candidate['objective']+merit_weight*abs(new_error)
                predicted = 2*float(grad@delta)-merit_weight*abs(error)
                if trial_merit <= merit+1e-4*step*min(predicted, 0.) and trial_merit < merit:
                    current = problem.evaluate(trial_x, True); evaluations += 1
                    x = trial_x; damping = max(damping/3, 1e-12); accepted = True
                    break
            except (ValueError, FloatingPointError, OverflowError):
                pass
            rejects += 1
        if not accepted:
            damping *= 10
            if damping > 1e12:
                reason = 'line_search_stalled'; break
    remaining = np.full_like(problem.y, np.nan)
    remaining[problem.mask] = problem.y[problem.mask]-current['physical'][problem.mask]
    return dict(x=x, value=current, remaining=remaining, converged=converged,
        status=reason, evaluations=evaluations, iterations=iteration,
        rejected_steps=rejects, scaled_gradient=gradient,
        constraint_error=0. if constraint is None else float(constraint(current, False)[0]))


def worst_output_direction(problem, value, name, output_scales, *, information='data'):
    """Worst RMS output change per visible-data/total local residual change.

    SVD truncation is replaced by a declared small ridge in standardized actual
    unknown coordinates. Reported gains and the ridge are local diagnostics,
    never data-based confidence intervals. The nonlinear profile uses the
    output projection, not a fixed parameter direction.
    """
    _, g = problem.output(value, name, output_scales)
    j = value['jacobian']
    if information == 'data':
        j = j[:value['data_rows']]
    elif information != 'total':
        raise ValueError('information must be data or total')
    d = problem.variable_scales
    j = j*d[None, :]/np.sqrt(value['data_rows'])
    gs = g*d[None, :]/np.sqrt(len(g))
    gram = j.T@j
    eigenvalues, vectors = eigh(gram, check_finite=False)
    floor = max(float(eigenvalues[-1]), 1.)*problem.cfg['profiles']['local_information_floor']**2
    whitener = vectors/np.sqrt(np.maximum(eigenvalues, 0.)+floor)[None, :]
    u, singular, vh = svd(gs@whitener, full_matrices=False, check_finite=False)
    v = d*(whitener@vh[0])
    output_direction = u[:, 0]
    pivot = int(np.argmax(np.abs(output_direction)))
    if output_direction[pivot] < 0:
        output_direction *= -1; v *= -1
    response = g@v
    norm = np.sqrt(np.mean(response**2))
    if norm > 1e-30:
        v /= norm
    standardized = v/d
    blocks = dict(driver=slice(0, problem.n), initial=slice(problem.n, problem.cstart), component=slice(problem.cstart, problem.count))
    mass = {k: float(np.sum(standardized[s]**2)) for k, s in blocks.items()}
    total = max(sum(mass.values()), 1e-30)
    return dict(direction=output_direction, variable_direction=v,
        gain=float(singular[0]), data_change_for_unit_output=float(np.linalg.norm(j@(v/d))),
        eigen_floor=floor, local_rank=int(np.sum(eigenvalues > floor)), unknowns=problem.count,
        compensation_fractions={k: a/total for k, a in mass.items()})


def output_profile(problem, reference, name, output_scales, offsets, *, information=None):
    value = reference['value']
    diagnostic = worst_output_direction(problem, value, name, output_scales,
        information=problem.objective if information is None else information)
    u = diagnostic['direction']
    g0, _ = problem.output(value, name, output_scales)
    normalizer = np.sqrt(len(g0))
    origin = float(u@g0/normalizer)
    records = []
    for sign in (-1, 1):
        warm = reference['x'].copy()
        for offset in sorted([a for a in offsets if np.sign(a) == sign], key=abs):
            target = origin+offset
            def constraint(candidate, derivative):
                g, jg = problem.output(candidate, name, output_scales)
                return float(u@g/normalizer-target), None if not derivative else u@jg/normalizer
            fit = fit_problem(problem, warm, constraint=constraint)
            if fit['converged']:
                warm = fit['x'].copy()
            new = fit['value']; g1, _ = problem.output(new, name, output_scales)
            records.append(dict(offset=offset, converged=fit['converged'], status=fit['status'],
                constraint_error=fit['constraint_error'], scaled_gradient=fit['scaled_gradient'],
                objective=new['objective'], reference_objective=value['objective'],
                objective_fraction=(new['objective']-value['objective'])/max(value['objective'], 1e-12),
                output_rms_change=float(np.sqrt(np.mean((g1-g0)**2))),
                SSM_Hb_rms_change=float(np.sqrt(np.mean(((new['physical'][120:240, 1:]-value['physical'][120:240, 1:])/problem.sd[1:])**2))),
                component_rms_change=float(np.sqrt(np.mean(((new['component'][120:240, 1:]-value['component'][120:240, 1:])/problem.sd[1:])**2))),
                total_prediction_rms_change=float(np.sqrt(np.mean(((new['prediction'][120:240, 1:]-value['prediction'][120:240, 1:])/problem.sd[1:])**2))),
                costs=new['costs'], x=fit['x'], evaluations=fit['evaluations']))
    return diagnostic, records
