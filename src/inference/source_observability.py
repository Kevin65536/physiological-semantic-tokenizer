"""Local data/prior information and genuinely free Balloon parameter refits.

All coordinates have an explicit fixed scale.  Information is local sensitivity
of a declared weighted objective, not a calibrated likelihood or global proof
of identifiability.  No data, registry, or runner is imported here.
"""
from __future__ import annotations

from dataclasses import replace
import time
from collections import OrderedDict

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .shared_driver_spatial import _validated_design, forward_spatial_shared_driver
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


def _representation_target(target, sd, score_mask, feature_shape):
    target, scale = np.asarray(target, float), np.asarray(sd, float)
    if (target.ndim != len(feature_shape)+1 or target.shape[0] < 1
            or target.shape[1:] != feature_shape):
        raise ValueError("target must match the declared time and feature dimensions")
    if (scale.shape != feature_shape or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError("sd must be positive finite with the declared feature shape")
    mask = np.ones(target.shape, bool) if score_mask is None else np.asarray(score_mask)
    if mask.dtype != np.bool_ or mask.shape != target.shape:
        raise ValueError("score_mask must be boolean with target shape")
    if not mask.any() or not np.isfinite(target[mask]).all():
        raise ValueError("at least one finite scored target is required")
    return target, scale, mask


def _representation_score(target, prediction, scale, mask):
    count = int(mask.sum())
    residual = ((prediction[mask]-target[mask])
                /np.broadcast_to(scale, target.shape)[mask])
    sse = float(residual@residual)
    return dict(weighted_sse=sse, n_scored=count,
        nmse=sse/count if count else None,
        nrmse=float(np.sqrt(sse/count)) if count else None)


def _projection_diagnostics(design, basis):
    """Float64 span/orthogonality checks for an untruncated projection basis."""
    norm = np.linalg.norm(design)
    return dict(span_residual=float(np.linalg.norm(design-basis@(basis.T@design))/norm) if norm else 0.,
        orthogonality_error=float(np.linalg.norm(basis.T@basis-np.eye(basis.shape[1]))))


def eeg_representation_lower_bound(eeg, *, temporal_basis, eeg_loadings, sd,
        score_mask=None, eeg_operator=None):
    """Numerical EEG fit and a conservative numerical representation bound.

    The observation map is exactly ``eeg_operator @ temporal_basis @ C @
    eeg_loadings.T``. No dynamics, prior or penalty restricts C. ``nmse`` is
    sum(((prediction-eeg)/sd)**2)/n_scored and ``nrmse`` its square root;
    neither scale nor denominator is fitted to the scored target. Hidden target
    entries may be nonfinite and do not enter the fit or any score.

    Rectangular time/feature support uses the two small factor SVDs, retaining
    their product directions according to the *full* whitened design's machine
    precision threshold. Other masks use a selected, whitened least-squares
    design. Both return the numerical minimum-norm C at the reported machine
    rank tolerance eps*max(n_scored,M*K). Its residual can exceed the actual
    representation minimum when genuine tiny singular directions are discarded.

    Only ``certified_*`` fields are lower-bound evidence. They project onto an
    enlarged space retaining every SVD factor column, or every complete thin-QR
    column for a masked design, without inverting tiny singular values. Thus the
    certified prediction need not be representable by any C. This is a float64
    conservative numerical relaxation, NOT an interval-arithmetic certificate
    or the exact distance to the original model space. Span and orthogonality
    errors audit its numerical quality; scalar diagnostics are maxima over the
    projection factors. Unscored certified predictions use the numerical fit.
    """
    basis, loading = np.asarray(temporal_basis, float), np.asarray(eeg_loadings, float)
    if (basis.ndim != 2 or min(basis.shape, default=0) < 1
            or not np.isfinite(basis).all()):
        raise ValueError("temporal_basis must be finite nonempty [T,M]")
    if (loading.ndim != 2 or min(loading.shape, default=0) < 1
            or not np.isfinite(loading).all()):
        raise ValueError("eeg_loadings must be finite nonempty [F,K]")
    t, m = basis.shape
    f, k = loading.shape
    target, scale, mask = _representation_target(eeg, sd, score_mask, (f,))
    if target.shape[0] != t:
        raise ValueError("eeg and temporal_basis time dimensions differ")
    operator = np.eye(t) if eeg_operator is None else np.asarray(eeg_operator, float)
    if operator.shape != (t, t) or not np.isfinite(operator).all():
        raise ValueError("eeg_operator must be finite [T,T]")
    temporal = operator@basis
    if not np.isfinite(temporal).all():
        raise ValueError("observed temporal basis must remain finite")
    rows, columns = mask.any(axis=1), mask.any(axis=0)
    rcond = np.finfo(float).eps*max(int(mask.sum()), m*k)
    if np.array_equal(mask, rows[:, None] & columns[None, :]):
        left, sl, vl = np.linalg.svd(temporal[rows], full_matrices=False)
        right, sr, vr = np.linalg.svd(loading[columns]/scale[columns, None],
                                    full_matrices=False)
        singular_grid = sl[:, None]*sr[None, :]
        threshold = float(singular_grid.max(initial=0)*rcond)
        retained = singular_grid > threshold
        core = left.T@(target[np.ix_(rows, columns)]/scale[columns])@right
        coordinates = np.divide(core, singular_grid, out=np.zeros_like(core), where=retained)
        coefficients = vl.T@coordinates@vr
        singular = np.sort(singular_grid.ravel())[::-1]
        rank, solver = int(retained.sum()), "rectangular_two_sided_svd"
        certificate_selected = (left@core@right.T)*scale[columns]
        certificate_dimension = left.shape[1]*right.shape[1]
        certificate_diagnostics = dict(
            temporal=_projection_diagnostics(temporal[rows], left),
            spatial=_projection_diagnostics(loading[columns]/scale[columns, None], right))
    else:
        ti, fi = np.nonzero(mask)
        design = (temporal[ti, :, None]*loading[fi, None, :]).reshape(len(ti), m*k)
        design /= scale[fi, None]
        rhs = target[mask]/scale[fi]
        solution, _, rank, singular = np.linalg.lstsq(design, rhs, rcond=None)
        coefficients = solution.reshape(m, k)
        rank, solver = int(rank), "masked_whitened_lstsq"
        complete_basis = np.linalg.qr(design, mode="reduced")[0]
        certificate_selected = (complete_basis@(complete_basis.T@rhs))*scale[fi]
        certificate_dimension = complete_basis.shape[1]
        certificate_diagnostics = dict(design=_projection_diagnostics(design, complete_basis))
    prediction = temporal@coefficients@loading.T
    certified_prediction = prediction.copy()
    if solver == "rectangular_two_sided_svd":
        certified_prediction[np.ix_(rows, columns)] = certificate_selected
    else:
        certified_prediction[mask] = certificate_selected
    score = _representation_score(target, prediction, scale, mask)
    certificate = _representation_score(target, certified_prediction, scale, mask)
    standardized_residual = np.zeros_like(target)
    standardized_residual[mask] = ((prediction[mask]-target[mask])
                                 /np.broadcast_to(scale, target.shape)[mask])
    normal_residual = temporal.T@(standardized_residual/scale)@loading
    return dict(**score,
        prediction=prediction, coefficients=coefficients, rank=rank,
        certified_prediction=certified_prediction,
        certified_weighted_sse=certificate["weighted_sse"], certified_nmse=certificate["nmse"],
        certified_nrmse=certificate["nrmse"], certificate_projection_dimension=certificate_dimension,
        discarded_direction_count=certificate_dimension-rank,
        certificate_relaxed=bool(certificate_dimension > rank),
        numerical_projection_sse_excess=max(score["weighted_sse"]-certificate["weighted_sse"], 0.),
        certificate_type="float64_conservative_projection_not_interval_arithmetic",
        certificate_diagnostics=certificate_diagnostics,
        span_residual=max(item["span_residual"] for item in certificate_diagnostics.values()),
        orthogonality_error=max(item["orthogonality_error"] for item in certificate_diagnostics.values()),
        normal_equation_residual=normal_residual,
        normal_equation_residual_norm=float(np.linalg.norm(normal_residual)),
        singular_values=singular, rank_relative_tolerance=float(rcond), solver=solver)


def hb_representation_lower_bound(hb, *, hb_loadings, sd, score_mask=None,
        hb_gains=None, hb_operator=None):
    """Relax all vascular dynamics to free canonical HbO/HbR curves [T,K,2].

    A channel first mixes the K canonical curves by its fixed nonnegative,
    row-normalized loading and positive gain, then applies its own fixed time
    operator. The two chromophores are independent least-squares problems;
    each design has only T*K columns. With K=1, *every* channel shares the same
    two free curves before its gain/operator. This is a lower bound for the
    entire nonlinear model using that fixed observation map, regardless of the
    vascular dynamics, driver basis or initial conditions.

    ``sd`` is [J,2] or channel-major [2J]; ``score_mask`` is boolean [T,J,2].
    Joint and per-chromophore nmse/nrmse use the same fixed-scale definition as
    ``eeg_representation_lower_bound``. The latent curves have minimum Euclidean
    norm. A chromophore without scored entries has rank zero, zero curves/SSE,
    and undefined (None) nmse/nrmse. No giant joint chromophore Kronecker matrix
    or physiological constraint is introduced. Numerical fits use the reported
    machine rank. Only ``certified_*`` fields are conservative numerical lower
    bounds: all factor-SVD directions for a shared operator/rectangular support,
    otherwise all complete thin-QR columns, including unresolved directions.
    As in the EEG function, this may enlarge the actual representation space;
    it is not an interval-arithmetic certificate or an exact original-space
    distance. Scalar span/orthogonality diagnostics are maxima over all bases.
    """
    loading = np.asarray(hb_loadings, float)
    if (loading.ndim != 2 or not loading.shape[0] or loading.shape[1] not in (1, 2)
            or not np.isfinite(loading).all() or np.any(loading < 0)
            or not np.allclose(loading.sum(axis=1), 1., rtol=0., atol=1e-10)
            or np.any(loading.sum(axis=0) <= 0)):
        raise ValueError("hb_loadings must be nonnegative [J,K=1 or 2], row-sum one, with every latent supported")
    j, k = loading.shape
    scale = np.asarray(sd, float)
    if scale.shape == (2*j,):
        scale = scale.reshape(j, 2)
    target, scale, mask = _representation_target(hb, scale, score_mask, (j, 2))
    t = target.shape[0]
    gain = np.ones(j) if hb_gains is None else np.asarray(hb_gains, float)
    if gain.shape != (j,) or not np.isfinite(gain).all() or np.any(gain <= 0):
        raise ValueError("hb_gains must be positive finite [J]")
    operators = np.eye(t) if hb_operator is None else np.asarray(hb_operator, float)
    if operators.shape == (t, t):
        operators = np.broadcast_to(operators, (j, t, t))
    if operators.shape != (j, t, t) or not np.isfinite(operators).all():
        raise ValueError("hb_operator must be finite [T,T] or [J,T,T]")
    mixing = gain[:, None]*loading
    design = np.einsum("jst,jk->sjtk", operators, mixing).reshape(t*j, t*k)
    if not np.isfinite(design).all():
        raise ValueError("Hb observation design must remain finite")
    latent = np.zeros((t, k, 2))
    normal_residual = np.zeros_like(latent)
    ranks, rconds, singular = {}, {}, {}
    certificate_values = np.zeros_like(target)
    certificate_dimensions, certificate_diagnostics = {}, {}
    common_operator = np.array_equal(operators, np.broadcast_to(operators[0], operators.shape))
    for chromophore, name in enumerate(("HbO", "HbR")):
        selected = mask[:, :, chromophore].ravel()
        factors = np.broadcast_to(scale[:, chromophore], (t, j)).ravel()[selected]
        whitened = design[selected]/factors[:, None]
        rhs = target[:, :, chromophore].ravel()[selected]/factors
        solution, _, rank, values = np.linalg.lstsq(whitened, rhs, rcond=None)
        latent[:, :, chromophore] = solution.reshape(t, k)
        normal_residual[:, :, chromophore] = (whitened.T@(whitened@solution-rhs)).reshape(t, k)
        ranks[name], rconds[name], singular[name] = int(rank), float(
            np.finfo(float).eps*max(len(rhs), t*k)), values
        chromophore_mask = mask[:, :, chromophore]
        rows, columns = chromophore_mask.any(axis=1), chromophore_mask.any(axis=0)
        if common_operator and selected.any() and np.array_equal(
                chromophore_mask, rows[:, None] & columns[None, :]):
            time_design = operators[0][rows]
            space_design = mixing[columns]/scale[columns, chromophore, None]
            left = np.linalg.svd(time_design, full_matrices=False)[0]
            right = np.linalg.svd(space_design, full_matrices=False)[0]
            observed = target[:, :, chromophore][np.ix_(rows, columns)]/scale[columns, chromophore]
            relaxed = (left@(left.T@observed@right)@right.T)*scale[columns, chromophore]
            certificate_values[:, :, chromophore][np.ix_(rows, columns)] = relaxed
            certificate_dimensions[name] = left.shape[1]*right.shape[1]
            certificate_diagnostics[name] = dict(temporal=_projection_diagnostics(time_design, left),
                                                 spatial=_projection_diagnostics(space_design, right))
        else:
            complete_basis = np.linalg.qr(whitened, mode="reduced")[0]
            certificate_values[:, :, chromophore].flat[np.flatnonzero(selected)] = (
                complete_basis@(complete_basis.T@rhs))*factors
            certificate_dimensions[name] = complete_basis.shape[1]
            certificate_diagnostics[name] = dict(design=_projection_diagnostics(whitened, complete_basis))
    canonical = np.einsum("jk,tkc->tjc", mixing, latent)
    prediction = np.einsum("jst,tjc->sjc", operators, canonical)
    certified_prediction = np.where(mask, certificate_values, prediction)
    by_chromophore = {name: _representation_score(target[:, :, index],
        prediction[:, :, index], scale[:, index], mask[:, :, index])
        for index, name in enumerate(("HbO", "HbR"))}
    certified_by_chromophore = {name: _representation_score(target[:, :, index],
        certified_prediction[:, :, index], scale[:, index], mask[:, :, index])
        for index, name in enumerate(("HbO", "HbR"))}
    score = _representation_score(target, prediction, scale, mask)
    certificate = _representation_score(target, certified_prediction, scale, mask)
    diagnostics = [item for chromophore in certificate_diagnostics.values() for item in chromophore.values()]
    discarded = {name: certificate_dimensions[name]-ranks[name] for name in ranks}
    return dict(**score,
        prediction=prediction, latent_hb_prediction=latent,
        canonical_hb_prediction=canonical, by_chromophore=by_chromophore,
        certified_prediction=certified_prediction,
        certified_weighted_sse=certificate["weighted_sse"], certified_nmse=certificate["nmse"],
        certified_nrmse=certificate["nrmse"], certified_by_chromophore=certified_by_chromophore,
        certificate_projection_dimension=sum(certificate_dimensions.values()),
        certificate_projection_dimension_by_chromophore=certificate_dimensions,
        discarded_direction_count=sum(discarded.values()), discarded_direction_count_by_chromophore=discarded,
        certificate_relaxed=bool(any(discarded.values())),
        numerical_projection_sse_excess=max(score["weighted_sse"]-certificate["weighted_sse"], 0.),
        certificate_type="float64_conservative_projection_not_interval_arithmetic",
        certificate_diagnostics=certificate_diagnostics,
        span_residual=max(item["span_residual"] for item in diagnostics),
        orthogonality_error=max(item["orthogonality_error"] for item in diagnostics),
        rank=sum(ranks.values()), rank_by_chromophore=ranks,
        normal_equation_residual=normal_residual,
        normal_equation_residual_norm=float(np.linalg.norm(normal_residual)),
        singular_values=singular, rank_relative_tolerance=rconds,
        solver="chromophore_masked_whitened_lstsq")


def fit_feasibility_frontier(eeg, hb, parameters, dt, *, temporal_basis,
        eeg_loadings, hb_loadings, sd, epsilon, starts=None,
        free_physiology=False, score_mask=None, hb_gains=None,
        eeg_operator=None, hb_operator=None, tau_bounds=(.6, 4.),
        kappa_bounds=(.2, 1.3), max_nfev=300, substeps=4,
        verification_substeps=8, validation_tolerance=.005,
        constraint_tolerance=1e-8, ftol=1e-9, finite_difference_step=1e-4,
        numerical_backend="numba", joint_tolerance=.5):
    """Search Hb NMSE subject to a hard EEG NRMSE <= epsilon constraint.

    No prior or penalty changes the valid-domain objective. SLSQP uses exact
    coefficient/independent-initial-state Jacobians; free log(tau,kappa) columns
    use domain-checked finite differences. Alpha, E0 and all remaining physical
    and observation parameters remain exactly those of ``parameters``. Bounds
    on tau/kappa are engineering search bounds, applied only when they are free.

    Each explicit start supplies coefficients [M,K], initial_coordinates [K,5],
    optional tau/kappa and label. Without starts, the only initialization is the
    EEG least-squares coefficients and rest initials. Hidden scored-out targets
    are never read. ``max_nfev`` bounds differentiated forward evaluations *per
    start*; finite differences and verification calls are counted separately.
    A four-entry cache prevents repeated callbacks from rerunning the solver.
    Invalid nonlinear trials receive infinite objective, are recorded and never
    clipped. Initial candidates and all later legal candidates remain eligible,
    so passing an earlier witness as a start preserves model/epsilon nesting.

    Returned predictions and scores use ``verification_substeps``. A feasible
    witness requires physical validity at both resolutions, verified EEG NRMSE
    <= epsilon+constraint_tolerance, and integration_error <=
    validation_tolerance. Integration error is the largest fixed-sd RMS change
    between resolutions on any scored EEG/HbO/HbR support. This is independent
    of optimizer convergence. Failure to find a witness is not an infeasibility
    proof; only an EEG representation bound above the threshold excludes a
    condition directly. ``joint_witness`` independently retains the best legal
    feasible candidate whose EEG, HbO and HbR NRMSE each meet joint_tolerance,
    even if a later mean-Hb optimum fails one of these diagnostic thresholds.
    This local search never certifies a global frontier.
    """
    from scipy.optimize import minimize

    parameters.validate()
    loading, mixing, basis, gains, eeg_op, hb_op = _validated_design(
        eeg_loadings, hb_loadings, temporal_basis, hb_gains, eeg_operator, hb_operator)
    controls = np.asarray([dt, epsilon, validation_tolerance, constraint_tolerance,
                           ftol, finite_difference_step, joint_tolerance], float)
    if (not np.isfinite(controls).all() or np.any(controls[[0, 4, 5]] <= 0)
            or np.any(controls[[1, 2, 3, 6]] < 0)):
        raise ValueError("invalid time step, threshold, tolerance or finite-difference step")
    for name, value in (("max_nfev", max_nfev), ("substeps", substeps),
                        ("verification_substeps", verification_substeps)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if verification_substeps <= substeps:
        raise ValueError("verification_substeps must exceed substeps")
    if not isinstance(free_physiology, (bool, np.bool_)):
        raise ValueError("free_physiology must be boolean")
    if numerical_backend not in ("numba", "python"):
        raise ValueError("numerical_backend must be numba or python")
    bounds = np.asarray([tau_bounds, kappa_bounds], float)
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] <= 0) or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError("tau_bounds and kappa_bounds must be positive finite ordered pairs")
    t, m = basis.shape
    f, k = loading.shape
    j = len(mixing)
    e, h = np.asarray(eeg, float), np.asarray(hb, float)
    if e.shape != (t, f) or h.shape != (t, j, 2):
        raise ValueError("eeg and hb must match [T,F] and [T,J,2]")
    target = np.column_stack((e, h.reshape(t, 2*j)))
    target, scale, mask = _representation_target(target, sd, score_mask, (f+2*j,))
    if not mask[:, :f].any() or not mask[:, f:].any():
        raise ValueError("score_mask must include both EEG and Hb observations")
    lower = eeg_representation_lower_bound(e, temporal_basis=basis,
        eeg_loadings=loading, sd=scale[:f], score_mask=mask[:, :f], eeg_operator=eeg_op)
    common = dict(epsilon=float(epsilon), validation_tolerance=float(validation_tolerance),
        constraint_tolerance=float(constraint_tolerance),
        eeg_representation_lower_bound=lower["certified_nrmse"],
        eeg_numerical_projection_nrmse=lower["nrmse"],
        eeg_representation_lower_bound_relaxed=lower["certificate_relaxed"],
        free_physiology=bool(free_physiology),
        substeps=int(substeps), verification_substeps=int(verification_substeps),
        joint_tolerance=float(joint_tolerance))
    if lower["certified_nrmse"] > epsilon+constraint_tolerance:
        return dict(**common, prediction=None, states=None, drivers=None, coefficients=None, initial_coordinates=None,
            tau=None, kappa=None, eeg_nmse=None, hb_nmse=None, hbo_nmse=None, hbr_nmse=None,
            eeg_nrmse=None, hb_nrmse=None, hbo_nrmse=None, hbr_nrmse=None, scores=None,
            witness_feasible=False, physical_valid=False, integration_error=None,
            optimizer_success=False, optimizer_converged=False, status="excluded_by_eeg_representation",
            source_start_label=None, joint_witness=None, starts=[], evaluations=0, forward_calls=0,
            finite_difference_calls=0, verification_calls=0, cache_hits=0, invalid_trials=0,
            failure_log=[])
    if starts is None:
        starts = [dict(label="eeg_least_squares_rest", coefficients=lower["coefficients"],
                       initial_coordinates=np.zeros((k, 5)))]
    if not isinstance(starts, (list, tuple)) or not starts:
        raise ValueError("starts must be a nonempty explicit list")
    ncoef, nphysical = m*k, m*k+5*k
    n = nphysical+2*int(free_physiology)
    log_bounds = np.log(bounds)
    normalized_starts = []
    for index, start in enumerate(starts):
        if not isinstance(start, dict) or "coefficients" not in start or "initial_coordinates" not in start:
            raise ValueError("each start requires coefficients and independent initial_coordinates")
        coef, initial = np.asarray(start["coefficients"], float), np.asarray(start["initial_coordinates"], float)
        theta = np.asarray([start.get("tau", parameters.free.tau),
                            start.get("kappa", parameters.free.kappa)], float)
        if (coef.shape != (m, k) or initial.shape != (k, 5)
                or not np.isfinite(coef).all() or not np.isfinite(initial).all()
                or theta.shape != (2,) or not np.isfinite(theta).all() or np.any(theta <= 0)):
            raise ValueError("start shapes/values must match finite [M,K], [K,5] and positive tau/kappa")
        if free_physiology and (np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1])):
            raise ValueError("free-physiology starts must lie inside tau/kappa bounds")
        if not free_physiology and not np.array_equal(theta, [parameters.free.tau, parameters.free.kappa]):
            raise ValueError("fixed-physiology start tau/kappa must match parameters")
        label = start.get("label", f"start_{index}")
        if not isinstance(label, str) or not label:
            raise ValueError("start label must be a nonempty string")
        packed = np.r_[coef.ravel(), initial.ravel()]
        if free_physiology:
            packed = np.r_[packed, np.log(theta)]
        normalized_starts.append((label, packed))
    forward_kwargs = dict(temporal_basis=basis, eeg_loadings=loading,
        hb_loadings=mixing, hb_gains=gains, eeg_operator=eeg_op, hb_operator=hb_op,
        numerical_backend=numerical_backend)
    eeg_mask, hb_mask = mask[:, :f], mask[:, f:]
    eeg_t, eeg_f = np.nonzero(eeg_mask)
    eeg_design = ((eeg_op@basis)[eeg_t, :, None]*loading[eeg_f, None, :]).reshape(-1, ncoef)
    eeg_design /= scale[eeg_f, None]
    eeg_rhs = e[eeg_mask]/scale[eeg_f]
    eeg_count, hb_count = len(eeg_rhs), int(hb_mask.sum())
    hb_scale = np.broadcast_to(scale[f:], (t, 2*j))[hb_mask]
    hb_rhs = target[:, f:][hb_mask]/hb_scale
    hb_chromophore_mask = hb_mask.reshape(t, j, 2)
    physical_names = ("finite", "positive_fvpq", "oxygen_extraction_in_unit_interval",
                      "absolute_hb_nonnegative", "hbr_not_above_hbt")
    failures, records = [], []

    def unpack(x):
        physical = with_log_physiology(parameters, x[-2:]) if free_physiology else parameters
        return x[:ncoef].reshape(m, k), x[ncoef:nphysical].reshape(k, 5), physical

    def eeg_constraint(x):
        residual = eeg_design@x[:ncoef]-eeg_rhs
        return float(epsilon**2-residual@residual/eeg_count)

    def eeg_constraint_jacobian(x):
        gradient = np.zeros(n)
        gradient[:ncoef] = -2*eeg_design.T@(eeg_design@x[:ncoef]-eeg_rhs)/eeg_count
        return gradient

    def scores(prediction, reference=None):
        reference = target if reference is None else reference
        eeg_reference = reference[:, :f]
        hb_reference = reference[:, f:].reshape(t, j, 2)
        hb_prediction = prediction[:, f:].reshape(t, j, 2)
        return dict(EEG=_representation_score(eeg_reference, prediction[:, :f], scale[:f], eeg_mask),
            Hb=_representation_score(hb_reference, hb_prediction, scale[f:].reshape(j, 2), hb_chromophore_mask),
            HbO=_representation_score(hb_reference[:, :, 0], hb_prediction[:, :, 0], scale[f::2], hb_chromophore_mask[:, :, 0]),
            HbR=_representation_score(hb_reference[:, :, 1], hb_prediction[:, :, 1], scale[f+1::2], hb_chromophore_mask[:, :, 1]))

    def check_physical(forward, physical):
        checks = [run_physical_checks(forward["states"][:, coordinate], physical) for coordinate in range(k)]
        return checks, all(all(item[name] for name in physical_names) for item in checks)

    class EvaluationBudget(Exception):
        pass

    class DerivativeDomainFailure(Exception):
        pass

    for label, initial_x in normalized_starts:
        counts = dict(evaluations=0, forward_calls=0, finite_difference_calls=0,
                      verification_calls=0, cache_hits=0, invalid_trials=0)
        cache, best, best_joint = OrderedDict(), None, None

        def call_forward(x, phase, derivative, steps):
            counts["forward_calls"] += 1
            if phase == "finite_difference":
                counts["finite_difference_calls"] += 1
            elif phase == "verification":
                counts["verification_calls"] += 1
            coef, initial, physical = unpack(x)
            try:
                return forward_spatial_shared_driver(coef, initial, physical, dt,
                    derivative=derivative, substeps=steps, **forward_kwargs)
            except (FloatingPointError, OverflowError, ValueError) as exc:
                counts["invalid_trials"] += 1
                failures.append(dict(start=label, evaluation=counts["evaluations"],
                                     phase=phase, error=str(exc)))
                return None

        def retain_candidate(x, coarse):
            nonlocal best, best_joint
            _, _, physical = unpack(x)
            coarse_checks, coarse_valid = check_physical(coarse, physical)
            if not coarse_valid:
                counts["invalid_trials"] += 1
                failures.append(dict(start=label, evaluation=counts["evaluations"],
                                     phase="optimization_physical_checks", error="failed physical checks"))
                return
            fine = call_forward(x, "verification", False, verification_substeps)
            if fine is None:
                return
            fine_checks, fine_valid = check_physical(fine, physical)
            if not fine_valid:
                counts["invalid_trials"] += 1
                failures.append(dict(start=label, evaluation=counts["evaluations"],
                                     phase="verification_physical_checks", error="failed physical checks"))
                return
            metric, coarse_metric = scores(fine["prediction"]), scores(coarse["prediction"])
            change = fine["prediction"]-coarse["prediction"]
            changes = scores(change, np.zeros_like(target))
            integration = {name: changes[name]["nrmse"] for name in ("EEG", "HbO", "HbR")}
            integration_error = max(value for value in integration.values() if value is not None)
            feasible = (metric["EEG"]["nrmse"] <= epsilon+constraint_tolerance
                        and integration_error <= validation_tolerance)
            candidate = dict(prediction=fine["prediction"].copy(),
                states=fine["states"].copy(), drivers=fine["drivers"].copy(),
                coefficients=x[:ncoef].reshape(m, k).copy(),
                initial_coordinates=x[ncoef:nphysical].reshape(k, 5).copy(),
                tau=float(physical.free.tau), kappa=float(physical.free.kappa),
                scores=metric, optimization_scores=coarse_metric,
                witness_feasible=bool(feasible), physical_valid=True,
                integration_error=float(integration_error), integration_error_by_modality=integration,
                physical_checks=fine_checks, optimization_physical_checks=coarse_checks,
                source_start_label=label)
            for modality, prefix in (("EEG", "eeg"), ("Hb", "hb"), ("HbO", "hbo"), ("HbR", "hbr")):
                for name in ("nmse", "nrmse"):
                    candidate[f"{prefix}_{name}"] = metric[modality][name]
            # Verified witnesses dominate all diagnostics. Otherwise retain the
            # closest legal candidate rather than an illegal optimizer endpoint.
            rank = (not feasible, max(metric["EEG"]["nrmse"]-epsilon-constraint_tolerance, 0.)
                    +max(integration_error-validation_tolerance, 0.), metric["Hb"]["nmse"])
            if best is None or rank < best[0]:
                best = (rank, candidate)
            if feasible and all(metric[name]["nrmse"] is not None
                    and metric[name]["nrmse"] <= joint_tolerance for name in ("EEG", "HbO", "HbR")):
                if best_joint is None or metric["Hb"]["nmse"] < best_joint["hb_nmse"]:
                    best_joint = candidate

        def evaluate(x):
            key = np.asarray(x, float).tobytes()
            if key in cache:
                counts["cache_hits"] += 1
                cache.move_to_end(key)
                return cache[key]
            if counts["evaluations"] >= max_nfev:
                raise EvaluationBudget
            counts["evaluations"] += 1
            forward = call_forward(x, "optimization", True, substeps)
            if forward is None:
                value = (float("inf"), np.zeros(n))
            else:
                retain_candidate(x, forward)
                residual = forward["prediction"][:, f:][hb_mask]/hb_scale-hb_rhs
                jac = np.zeros((hb_count, n))
                jac[:, :nphysical] = forward["jacobian"].reshape(t, f+2*j, nphysical)[:, f:][hb_mask]/hb_scale[:, None]
                if free_physiology:
                    for coordinate in range(2):
                        outputs = []
                        for sign in (-1., 1.):
                            theta = x[-2+coordinate]+sign*finite_difference_step
                            if not log_bounds[coordinate, 0] <= theta <= log_bounds[coordinate, 1]:
                                continue
                            shifted = x.copy()
                            shifted[-2+coordinate] = theta
                            trial = call_forward(shifted, "finite_difference", False, substeps)
                            if trial is not None:
                                outputs.append((theta, trial["prediction"][:, f:][hb_mask]))
                        if not outputs:
                            raise DerivativeDomainFailure
                        if len(outputs) == 1:
                            outputs.append((x[-2+coordinate], forward["prediction"][:, f:][hb_mask]))
                        jac[:, nphysical+coordinate] = ((outputs[1][1]-outputs[0][1])
                            /(outputs[1][0]-outputs[0][0])/hb_scale)
                value = (float(residual@residual/hb_count), 2*jac.T@residual/hb_count)
            cache[key] = value
            while len(cache) > 4:
                cache.popitem(last=False)
            return value

        success, optimizer_status, message = False, None, "not_started"
        try:
            # Evaluate and retain the original start before SLSQP can move it.
            objective, _ = evaluate(initial_x)
            if np.isfinite(objective):
                optimizer_bounds = [(None, None)]*nphysical
                if free_physiology:
                    optimizer_bounds += [tuple(pair) for pair in log_bounds]
                result = minimize(lambda x: evaluate(x)[0], initial_x,
                    jac=lambda x: evaluate(x)[1], method="SLSQP", bounds=optimizer_bounds,
                    constraints=[dict(type="ineq", fun=eeg_constraint, jac=eeg_constraint_jacobian)],
                    options=dict(ftol=ftol, maxiter=max_nfev, disp=False))
                success, optimizer_status, message = bool(result.success), int(result.status), str(result.message)
            else:
                message = "invalid_initial_domain"
        except EvaluationBudget:
            message = "differentiated_evaluation_budget"
        except DerivativeDomainFailure:
            message = "no_valid_physiology_difference"
        records.append(dict(label=label, optimizer_success=success, optimizer_converged=success,
            optimizer_status=optimizer_status, optimizer_message=message,
            best_candidate=best[1] if best is not None else None,
            joint_witness=best_joint, **counts))
    available = [record for record in records if record["best_candidate"] is not None]
    totals = {name: sum(record[name] for record in records) for name in
              ("evaluations", "forward_calls", "finite_difference_calls", "verification_calls", "cache_hits", "invalid_trials")}
    if not available:
        return dict(**common, prediction=None, states=None, drivers=None, coefficients=None, initial_coordinates=None,
            tau=None, kappa=None, eeg_nmse=None, hb_nmse=None, hbo_nmse=None, hbr_nmse=None,
            eeg_nrmse=None, hb_nrmse=None, hbo_nrmse=None, hbr_nrmse=None, scores=None,
            witness_feasible=False, physical_valid=False, integration_error=None,
            optimizer_success=False, optimizer_converged=False, status="failed_domain",
            source_start_label=None, joint_witness=None, starts=records, failure_log=failures, **totals)
    selected = min(available, key=lambda item: (
        not item["best_candidate"]["witness_feasible"],
        max(item["best_candidate"]["eeg_nrmse"]-epsilon-constraint_tolerance, 0.)
            +max(item["best_candidate"]["integration_error"]-validation_tolerance, 0.),
        item["best_candidate"]["hb_nmse"]))
    candidate = selected["best_candidate"]
    joint_candidates = [item["joint_witness"] for item in records if item["joint_witness"] is not None]
    joint_witness = min(joint_candidates, key=lambda item: item["hb_nmse"]) if joint_candidates else None
    return dict(**common, **candidate, optimizer_success=selected["optimizer_success"],
        optimizer_converged=selected["optimizer_converged"],
        status=("feasible_witness" if selected["optimizer_success"] else
                "feasible_witness_without_optimizer_convergence") if candidate["witness_feasible"] else "no_feasible_witness",
        joint_witness=joint_witness, starts=records, failure_log=failures, **totals)
