"""Local data/prior information and genuinely free Balloon parameter refits.

All coordinates have an explicit fixed scale.  Information is local sensitivity
of a declared weighted objective, not a calibrated likelihood or global proof
of identifiability.  No data, registry, or runner is imported here.
"""
from __future__ import annotations

from dataclasses import replace
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .shared_driver_spatial import forward_spatial_shared_driver
from .t3a_balloon_robust_ssm import BalloonFreeParameters, run_physical_checks


def weighted_jacobian(jacobian, sd, visible=None, feature_weights=None):
    """Select time-major rows and divide by fixed per-feature standard deviations.

    ``jacobian`` is [T,C,P] or [T*C,P]; a flattened input requires a [T,C]
    mask. ``feature_weights`` multiplies rows and never changes with visibility.
    Hidden nonfinite entries are permitted; selected entries must be finite.
    No per-column or visible-count normalization is performed.
    """
    jac, scale = np.asarray(jacobian, float), np.asarray(sd, float)
    if scale.ndim != 1 or not len(scale) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("sd must be positive finite [C]")
    if jac.ndim == 3:
        t, c, p = jac.shape
        if c != len(scale):
            raise ValueError("Jacobian feature count differs from sd")
        jac = jac.reshape(t*c, p)
    elif jac.ndim == 2 and visible is not None:
        shape = np.asarray(visible).shape
        if len(shape) != 2 or shape[1] != len(scale) or np.prod(shape) != len(jac):
            raise ValueError("flattened Jacobian requires matching [T,C] visible")
        t, c = shape
    else:
        raise ValueError("Jacobian must be [T,C,P], or flat with explicit visible")
    mask = np.ones((t, c), bool) if visible is None else np.asarray(visible)
    if mask.shape != (t, c) or mask.dtype != np.bool_:
        raise ValueError("visible must be boolean [T,C]")
    factors = np.ones(c) if feature_weights is None else np.asarray(feature_weights, float)
    if factors.shape != (c,) or not np.isfinite(factors).all() or np.any(factors < 0):
        raise ValueError("feature_weights must be nonnegative finite [C]")
    active = mask & (factors[None] > 0)
    selected = jac[active.ravel()]*np.broadcast_to(factors/scale, (t, c))[active, None]
    if not np.isfinite(selected).all():
        raise ValueError("selected Jacobian rows must be finite")
    return selected


def _spectrum(jac, rcond):
    p = jac.shape[1]
    singular = np.linalg.svd(jac, compute_uv=False)
    singular = np.pad(singular, (0, p-len(singular)))
    rank = int(np.sum(singular > (singular[0]*rcond if len(singular) else 0)))
    condition = float(singular[0]/singular[-1]) if rank == p and p else float("inf")
    return singular, rank, condition


def information_decomposition(data_jacobian, prior_jacobian=None,
                              parameter_scales=None, rcond=1e-8):
    """Separate data-only and augmented sensitivity in predefined coordinates.

    Columns are multiplied by ``parameter_scales``, so unit perturbations mean
    the same prescribed physical/log perturbation in every comparison.
    Prior fractions are Rayleigh contributions along *total* right singular
    directions, ordered strongest first. Zero-total directions receive NaN.
    """
    data = np.asarray(data_jacobian, float)
    if data.ndim != 2 or not data.shape[1] or not np.isfinite(data).all():
        raise ValueError("data_jacobian must be finite [N,P>0]")
    p = data.shape[1]
    prior = np.zeros((0, p)) if prior_jacobian is None else np.asarray(prior_jacobian, float)
    scales = np.ones(p) if parameter_scales is None else np.asarray(parameter_scales, float)
    if (prior.ndim != 2 or prior.shape[1] != p or not np.isfinite(prior).all()
            or scales.shape != (p,) or not np.isfinite(scales).all() or np.any(scales <= 0)
            or not 0 < rcond < 1):
        raise ValueError("prior/scale/rcond mismatch")
    d, q = data*scales, prior*scales
    ds, dr, dc = _spectrum(d, rcond)
    ts, tr, tc = _spectrum(np.vstack((d, q)), rcond)
    values, vectors = np.linalg.eigh(d.T@d+q.T@q)
    values, vectors = np.maximum(values[::-1], 0), vectors[:, ::-1]
    fractions = np.divide(np.sum((q@vectors)**2, axis=0), values,
        out=np.full(p, np.nan), where=values > np.finfo(float).eps*max(values[0], 1))
    return dict(data_singular_values=ds, total_singular_values=ts,
        data_rank=dr, total_rank=tr, unknowns=p, data_condition=dc, total_condition=tc,
        data_information=d.T@d, prior_information=q.T@q,
        prior_fraction_by_total_direction=fractions,
        total_directions=vectors, parameter_scales=scales, rank_relative_tolerance=rcond)


def nuisance_profile_information(jacobian, interest_columns, rcond=1e-8):
    """Project interest sensitivities off all nuisance columns using truncated SVD.

    The compensating nuisance increment is -pinv(J_nuisance) J_interest.
    Information before/after projection is diagnostic local curvature, with
    no prior rows unless the caller explicitly supplies them.
    """
    jac = np.asarray(jacobian, float)
    interest = np.asarray(interest_columns, int)
    if (jac.ndim != 2 or not np.isfinite(jac).all() or interest.ndim != 1
            or not len(interest) or len(set(interest.tolist())) != len(interest)
            or np.any(interest < 0) or np.any(interest >= jac.shape[1]) or not 0 < rcond < 1):
        raise ValueError("invalid Jacobian or interest columns")
    nuisance = np.setdiff1d(np.arange(jac.shape[1]), interest)
    target, other = jac[:, interest], jac[:, nuisance]
    if other.shape[1]:
        u, s, vt = np.linalg.svd(other, full_matrices=False)
        retain = s > (s[0]*rcond if len(s) else 0)
        compensation = -(vt[retain].T/s[retain])@(u[:, retain].T@target)
        profiled = target+other@compensation
        rank = int(retain.sum())
    else:
        compensation, profiled, rank = np.zeros((0, len(interest))), target.copy(), 0
    raw, info = target.T@target, profiled.T@profiled
    fraction = np.divide(np.diag(info), np.diag(raw), out=np.full(len(interest), np.nan),
                         where=np.diag(raw) > 0)
    return dict(raw_information=raw, profiled_information=info,
        retained_fraction=fraction, compensation_fraction=1-fraction,
        compensating_nuisance=compensation, projected_jacobian=profiled,
        nuisance_columns=nuisance, interest_columns=interest, nuisance_rank=rank,
        rank_relative_tolerance=rcond)


def with_log_physiology(parameters, log_tau_kappa):
    values = np.exp(np.asarray(log_tau_kappa, float))
    return replace(parameters, free=BalloonFreeParameters(tau=float(values[0]), kappa=float(values[1])))


def pack_spatial_linearization(coefficients, initial_coordinates, parameters, dt, *,
        sd, visible=None, feature_weights=None, driver_curvature_weight=.01,
        initial_weight=100., flow_prior_weight=1., flow_log_sd=np.log(2),
        physiology_prior_weight=0., physiology_prior_center=(2., .64),
        physiology_prior_log_sd=.5, finite_difference_step=1e-4, **forward_kwargs):
    """Return data/prior Jacobians with log(tau),log(kappa) appended.

    Driver/initial derivatives are the owner's exact discrete derivatives;
    physiology columns use central log-coordinate differences of the same
    nonlinear solver. Flow-prior physiology derivatives follow the same rule.
    Prior residual/Jacobian blocks retain separate names and contributions.
    """
    controls = np.asarray([dt, driver_curvature_weight, initial_weight, flow_prior_weight,
        flow_log_sd, physiology_prior_weight, physiology_prior_log_sd, finite_difference_step], float)
    if (not np.isfinite(controls).all() or np.any(controls[[0, 4, 6, 7]] <= 0)
            or np.any(controls[[1, 2, 3, 5]] < 0)):
        raise ValueError("invalid prior or numerical controls")
    kwargs = dict(forward_kwargs)
    kwargs.pop("derivative", None)
    coef, initial = np.asarray(coefficients, float), np.asarray(initial_coordinates, float)
    forward = forward_spatial_shared_driver(coef, initial, parameters, dt, derivative=True, **kwargs)
    t, c = forward["prediction"].shape
    mask = np.ones((t, c), bool) if visible is None else np.asarray(visible)
    ncoef, k = coef.size, coef.shape[1]
    n = ncoef+5*k
    jac = np.zeros((t*c, n+2))
    jac[:, :n] = forward["jacobian"]
    flow_jac = np.zeros((t, k, n+2))
    flow_jac[:, :, :n] = forward["flow_jacobian"]
    logtheta = np.log([parameters.free.tau, parameters.free.kappa])
    for column in range(2):
        outputs = []
        for sign in (-1., 1.):
            changed = logtheta.copy()
            changed[column] += sign*finite_difference_step
            outputs.append(forward_spatial_shared_driver(coef, initial,
                with_log_physiology(parameters, changed), dt, derivative=False, **kwargs))
        jac[:, n+column] = ((outputs[1]["prediction"]-outputs[0]["prediction"])
                            /(2*finite_difference_step)).ravel()
        flow_jac[:, :, n+column] = ((outputs[1]["states"][:, :, 2]-outputs[0]["states"][:, :, 2])
                                    /(2*finite_difference_step))
    data_jac = weighted_jacobian(jac, sd, mask, feature_weights)
    basis = np.asarray(kwargs["temporal_basis"], float)
    curvature = np.zeros(((t-2)*k, n+2))
    second = np.sqrt(driver_curvature_weight)*np.diff(basis, n=2, axis=0)/dt**1.5
    for index in range(k):
        curvature[index::k, index:ncoef:k] = second
    packed = np.r_[coef.ravel(), initial.ravel(), logtheta]
    initial_jac = np.zeros((5*k, n+2))
    initial_jac[:, ncoef:n] = np.diag(np.sqrt(initial_weight)*np.column_stack(
        (np.ones(k), forward["initial_state"][:, 1:])).ravel())
    flow = forward["states"][:, :, 2]
    flow_factor = np.sqrt(flow_prior_weight*dt)/flow_log_sd
    theta_jac = np.zeros((2, n+2))
    theta_jac[:, -2:] = np.eye(2)*np.sqrt(physiology_prior_weight)/physiology_prior_log_sd
    center = np.asarray(physiology_prior_center, float)
    if center.shape != (2,) or not np.isfinite(center).all() or np.any(center <= 0):
        raise ValueError("physiology_prior_center must be positive [tau,kappa]")
    prior_residuals = dict(driver_curvature=curvature@packed,
        initial_state=(np.sqrt(initial_weight)*(forward["initial_state"]-np.r_[0., np.ones(4)])).ravel(),
        flow_prior=(flow_factor*np.log(flow)).ravel(),
        physiology_prior=np.sqrt(physiology_prior_weight)*(logtheta-np.log(center))/physiology_prior_log_sd)
    prior_jacobians = dict(driver_curvature=curvature, initial_state=initial_jac,
        flow_prior=((flow_factor/flow)[:, :, None]*flow_jac).reshape(t*k, n+2),
        physiology_prior=theta_jac)
    return dict(forward=forward, data_jacobian=data_jac,
        observation_jacobian=jac, prior_jacobian=np.vstack(list(prior_jacobians.values())),
        prior_residual=np.concatenate(list(prior_residuals.values())),
        prior_residuals=prior_residuals, prior_jacobians=prior_jacobians,
        parameter_names=[f"coefficient_{i}" for i in range(ncoef)]+
            [f"initial_{i}" for i in range(5*k)]+["log_tau", "log_kappa"])


def fit_free_physiology(target, parameters, dt, *, temporal_basis, eeg_loadings,
        hb_loadings, sd, visible=None, feature_weights=None, start_tau_kappa=(2., .64),
        start_coefficients=None, start_initial_coordinates=None,
        fixed_tau=None, fixed_kappa=None, oracle_coefficients=None,
        oracle_initial_coordinates=None, tau_bounds=(.6, 4.), kappa_bounds=(.2, 1.3),
        max_nfev=100, gradient_tolerance=1e-5, **kwargs):
    """Safeguarded joint Gauss--Newton; every profile point refits free nuisance.

    Starts use visible EEG only (or zero driver), rest initials, and prespecified
    physiology. Oracle arrays are explicit controls and never supplied by the
    ordinary recovery arm. Invalid nonlinear trials are rejected, not clipped.
    ``max_nfev`` counts differentiated candidate/iteration evaluations; detailed
    forward-call counts (including physiology differences) are also retained.
    """
    started = time.monotonic()
    target = np.asarray(target, float)
    basis, loading = np.asarray(temporal_basis, float), np.asarray(eeg_loadings, float)
    t, m = basis.shape
    f, k = loading.shape
    c = f+2*len(hb_loadings)
    if target.shape != (t, c):
        raise ValueError("target must match [T,F+2J]")
    mask = np.ones(target.shape, bool) if visible is None else np.asarray(visible)
    if mask.dtype != np.bool_ or mask.shape != target.shape:
        raise ValueError("visible must be boolean target shape")
    factors = np.ones(c) if feature_weights is None else np.asarray(feature_weights, float)
    active = mask & (factors[None] > 0)
    if not active.any() or not np.isfinite(target[active]).all():
        raise ValueError("at least one finite visible target is required")
    scale = np.asarray(sd, float)
    # Validate common shape/weight rules before reading any selected target.
    weighted_jacobian(np.zeros((t, c, 1)), scale, mask, factors)
    ncoef, n = m*k, m*k+5*k
    x = np.r_[np.zeros(n), np.log(start_tau_kappa)]
    if start_coefficients is not None:
        x[:ncoef] = np.asarray(start_coefficients).reshape(ncoef)
    elif active[:, :f].any():
        eegop = kwargs.get("eeg_operator", np.eye(t))
        design = np.einsum("tm,fk->tfmk", eegop@basis, loading).reshape(t, f, ncoef)
        ej = weighted_jacobian(design, scale[:f], active[:, :f], factors[:f])
        rhs = (target[:, :f]*np.broadcast_to(factors[:f]/scale[:f], (t, f)))[active[:, :f]]
        x[:ncoef] = np.linalg.lstsq(ej, rhs, rcond=1e-10)[0]
    if start_initial_coordinates is not None:
        x[ncoef:n] = np.asarray(start_initial_coordinates).reshape(5*k)
    free = np.ones(n+2, bool)
    for index, value in ((n, fixed_tau), (n+1, fixed_kappa)):
        if value is not None:
            x[index], free[index] = np.log(value), False
    if oracle_coefficients is not None:
        x[:ncoef], free[:ncoef] = np.asarray(oracle_coefficients).ravel(), False
    if oracle_initial_coordinates is not None:
        x[ncoef:n], free[ncoef:n] = np.asarray(oracle_initial_coordinates).ravel(), False
    lower, upper = np.log([tau_bounds[0], kappa_bounds[0]]), np.log([tau_bounds[1], kappa_bounds[1]])
    if np.any(x[-2:] < lower) or np.any(x[-2:] > upper) or not free.any():
        raise ValueError("invalid parameter bounds/start or no free coordinates")
    row_weights = np.broadcast_to(factors/scale, target.shape)[active]
    failure_log, trace = [], []
    evaluations, forward_calls = 0, 0
    options = dict(temporal_basis=basis, eeg_loadings=loading, hb_loadings=hb_loadings,
        sd=scale, visible=mask, feature_weights=factors, **kwargs)

    def evaluate(candidate):
        nonlocal evaluations, forward_calls
        evaluations += 1
        forward_calls += 5
        local = pack_spatial_linearization(candidate[:ncoef].reshape(m, k),
            candidate[ncoef:n].reshape(k, 5), with_log_physiology(parameters, candidate[-2:]),
            dt, **options)
        residual = np.r_[(local["forward"]["prediction"][active]-target[active])*row_weights,
                         local["prior_residual"]]
        jac = np.vstack((local["data_jacobian"], local["prior_jacobian"]))[:, free]
        value = float(residual@residual)
        if not np.isfinite(value) or not np.isfinite(jac).all():
            raise FloatingPointError("nonfinite objective")
        return local, residual, jac, value

    current = None
    for attempt in range(min(20, max_nfev)):
        try:
            current = evaluate(x)
            break
        except (FloatingPointError, OverflowError, ValueError) as exc:
            failure_log.append(dict(stage="initialization", evaluation=evaluations, error=str(exc)))
            x[:ncoef][free[:ncoef]] *= .5
    converged, reason, damping = False, "evaluation_budget", 1e-4
    gradient = float("inf")
    if current is not None:
        while True:
            local, residual, jac, objective = current
            gram, rhs = jac.T@jac, jac.T@residual
            diagonal = np.maximum(np.diag(gram), 1.)
            gradient = float(np.max(np.abs(rhs)/np.sqrt(diagonal)))
            trace.append(dict(evaluation=evaluations, objective=objective,
                              gradient=gradient, tau=float(np.exp(x[-2])), kappa=float(np.exp(x[-1]))))
            if gradient <= gradient_tolerance:
                converged, reason = True, "scaled_gradient"
                break
            if evaluations >= max_nfev:
                break
            try:
                delta = cho_solve(cho_factor(gram+damping*np.diag(diagonal), lower=True), -rhs)
            except np.linalg.LinAlgError:
                damping *= 10
                if damping > 1e12:
                    reason = "singular_step"
                    break
                continue
            accepted = False
            for power in range(18):
                if evaluations >= max_nfev:
                    break
                candidate = x.copy()
                candidate[free] += 2.**(-power)*delta
                if np.any(candidate[-2:] < lower) or np.any(candidate[-2:] > upper):
                    continue
                try:
                    trial = evaluate(candidate)
                except (FloatingPointError, OverflowError, ValueError) as exc:
                    failure_log.append(dict(stage="candidate", evaluation=evaluations, error=str(exc)))
                    continue
                if trial[3] < objective:
                    x, current, accepted = candidate, trial, True
                    damping = max(damping/3, 1e-12)
                    break
            if not accepted:
                damping *= 10
                if damping > 1e12 or np.linalg.norm(delta) < 1e-12*(1+np.linalg.norm(x[free])):
                    reason = "stagnation_without_gradient_convergence"
                    break
    if current is None:
        return dict(success=False, status="failed_domain", convergence_reason="no_feasible_initialization",
            nfev=evaluations, forward_calls=forward_calls, failure_log=failure_log,
            trace=trace, elapsed_s=time.monotonic()-started)
    local, residual, jac, objective = current
    forward = local["forward"]
    checks = [run_physical_checks(forward["states"][:, i], with_log_physiology(parameters, x[-2:]))
              for i in range(k)]
    physical = all(all(check[name] for name in ("finite", "positive_fvpq",
        "oxygen_extraction_in_unit_interval", "absolute_hb_nonnegative", "hbr_not_above_hbt"))
        for check in checks)
    components = {name: float(value@value) for name, value in local["prior_residuals"].items()}
    components["data"] = float(residual[:int(active.sum())]@residual[:int(active.sum())])
    return dict(**forward, tau=float(np.exp(x[-2])), kappa=float(np.exp(x[-1])),
        success=bool(converged and physical), status="completed" if converged and physical else "failed_numerical",
        convergence_reason=reason, nfev=evaluations, forward_calls=forward_calls,
        physical_checks=checks, scaled_gradient_inf_norm=gradient, objective=objective,
        objective_components=components, failure_log=failure_log, trace=trace,
        data_jacobian=local["data_jacobian"], prior_jacobian=local["prior_jacobian"],
        parameter_names=local["parameter_names"], free_coordinates=free,
        start_tau_kappa=list(start_tau_kappa), elapsed_s=time.monotonic()-started)
