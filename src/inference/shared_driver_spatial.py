"""Retained pre-Tak spatial readouts of independent inlet-balance cores.

Each spatial coordinate owns one driver and five initial vascular coordinates.
Physiology, temporal coordinates, spatial loadings, gains and observation
operators stay fixed. There are no graph edges or extra Hb components. These
are conditional reconstructions, not physiological parameter posteriors.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .shared_driver_modes import temporal_mode_basis
from .shared_driver_reconstruction import nonlinear_driver_forward
from .t3a_balloon_robust_ssm import run_physical_checks


def _validated_spatial_loadings(eeg_loadings, hb_loadings):
    eeg = np.asarray(eeg_loadings, float)
    hb = np.asarray(hb_loadings, float)
    if (eeg.ndim != 2 or eeg.shape[1] not in (1, 2) or not eeg.shape[0]
            or not np.isfinite(eeg).all()
            or np.linalg.matrix_rank(eeg) != eeg.shape[1]):
        raise ValueError("eeg_loadings must be finite full-column-rank [F,K=1 or 2]")
    k = eeg.shape[1]
    if (hb.ndim != 2 or hb.shape[1] != k or not hb.shape[0]
            or not np.isfinite(hb).all() or np.any(hb < 0)
            or not np.allclose(hb.sum(axis=1), 1., rtol=0., atol=1e-10)
            or np.any(hb.sum(axis=0) <= 0)):
        raise ValueError("hb_loadings must be nonnegative [J,K], row-sum one, with every latent supported")
    return eeg, hb


def _validated_design(eeg_loadings, hb_loadings, temporal_basis, hb_gains,
                      eeg_operator, hb_operator):
    eeg, hb = _validated_spatial_loadings(eeg_loadings, hb_loadings)
    basis = np.asarray(temporal_basis, float)
    if (basis.ndim != 2 or basis.shape[0] < 3
            or not 1 <= basis.shape[1] <= basis.shape[0]
            or not np.isfinite(basis).all()
            or np.linalg.matrix_rank(basis) != basis.shape[1]):
        raise ValueError("temporal_basis must be finite full-column-rank [T>=3,M<=T]")
    t, j = len(basis), len(hb)
    gain = np.ones(j) if hb_gains is None else np.asarray(hb_gains, float)
    if gain.shape != (j,) or not np.isfinite(gain).all() or np.any(gain <= 0):
        raise ValueError("hb_gains must be positive finite [J]")
    eeg_op = np.eye(t) if eeg_operator is None else np.asarray(eeg_operator, float)
    if eeg_op.shape != (t, t) or not np.isfinite(eeg_op).all():
        raise ValueError("eeg_operator must be finite [T,T]")
    hb_op = np.eye(t) if hb_operator is None else np.asarray(hb_operator, float)
    if hb_op.shape == (t, t):
        hb_op = np.broadcast_to(hb_op, (j, t, t))
    if hb_op.shape != (j, t, t) or not np.isfinite(hb_op).all():
        raise ValueError("hb_operator must be finite [T,T] or [J,T,T]")
    return eeg, hb, basis, gain, eeg_op, hb_op


def collapse_tied_spatial_design(eeg_loadings, hb_loadings):
    """Collapse the exact K2 restriction r1=r2 and initial1=initial2.

    Returns keyword arguments for ``forward_spatial_shared_driver``. The EEG
    loading is the unnormalized sum of the original columns; convex Hb rows
    become one. The original K2 design must be valid. A loading sum at most
    1e-10 times the original Frobenius norm is rejected explicitly, rather than
    normalizing a cancelled readout or fitting a different model.
    """
    eeg = np.asarray(eeg_loadings, float)
    if (eeg.ndim != 2 or eeg.shape[1] != 2 or not eeg.shape[0]
            or not np.isfinite(eeg).all()):
        raise ValueError("tied restriction requires finite eeg_loadings [F,K=2]")
    with np.errstate(over="raise", invalid="raise"):
        try:
            summed = eeg.sum(axis=1, keepdims=True)
        except FloatingPointError as exc:
            raise ValueError("tied EEG loading sum must remain finite") from exc
    scale = float(np.max(np.abs(eeg)))
    if (scale == 0 or np.linalg.norm(summed/scale)
            <= 1e-10*np.linalg.norm(eeg/scale)):
        raise ValueError("tied EEG loading sum cancels or is nearly zero relative to original loadings")
    _, hb = _validated_spatial_loadings(eeg, hb_loadings)
    return dict(eeg_loadings=summed, hb_loadings=np.ones((len(hb), 1)))


def forward_spatial_shared_driver(coefficients, initial_coordinates, parameters,
        dt, *, eeg_loadings, hb_loadings, temporal_basis, hb_gains=None,
        eeg_operator=None, hb_operator=None, derivative=True, substeps=4,
        numerical_backend="numba"):
    """Generate spatial EEG and Hb means and their exact discrete Jacobian.

    Coefficients are [M,K]; initial coordinates [K,5] are
    [s0,log(f0),log(v0),log(p0),log(q0)]. EEG is drivers@eeg_loadings.T;
    the loading already includes its fixed observation scale. Each Hb channel
    is a nonnegative convex mixture of the K canonical Hb pairs, multiplied by
    its fixed positive gain. Its time operator acts after mixing. This preserves
    the core total-Hb = HbO+HbR identity before and after linear processing.

    Prediction columns are EEG features followed by channel-major HbO,HbR.
    Jacobian rows are time-major; columns are coefficients.ravel(C), followed
    by initial_coordinates.ravel(C). Invalid RK stages raise without clipping.
    Hb loading rank may be below K and is reported by the fitter, rather than
    silently manufacturing spatial information.
    """
    eeg, hb, basis, gain, eeg_op, hb_op = _validated_design(
        eeg_loadings, hb_loadings, temporal_basis, hb_gains, eeg_operator, hb_operator)
    parameters.validate()
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive finite")
    if not isinstance(derivative, (bool, np.bool_)):
        raise ValueError("derivative must be boolean")
    t, m = basis.shape
    f, k = eeg.shape
    j = len(hb)
    coef = np.asarray(coefficients, float)
    initial = np.asarray(initial_coordinates, float)
    if coef.shape != (m, k) or not np.isfinite(coef).all():
        raise ValueError("coefficients must be finite [M,K]")
    if initial.shape != (k, 5) or not np.isfinite(initial).all():
        raise ValueError("initial_coordinates must be finite [K,5]")
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        physical_initial = np.column_stack((initial[:, 0], np.exp(initial[:, 1:])))
    drivers = basis@coef
    cores = [nonlinear_driver_forward(drivers[:, index], physical_initial[index],
        parameters, dt, derivative=derivative, substeps=substeps,
        return_flow_jacobian=derivative, numerical_backend=numerical_backend, hemodynamics='inlet_balance', viscoelastic_s=0.)
        for index in range(k)]
    states = np.stack([core["states"] for core in cores], axis=1)
    latent_hb = np.stack([core["canonical_prediction"][:, 1:] for core in cores], axis=1)
    canonical_eeg = drivers@eeg.T
    canonical_hb = np.einsum("jk,tkc->tjc", gain[:, None]*hb, latent_hb)
    eeg_prediction = eeg_op@canonical_eeg
    hb_prediction = np.einsum("jst,tjc->sjc", hb_op, canonical_hb)
    prediction = np.column_stack((eeg_prediction, hb_prediction.reshape(t, 2*j)))
    jacobian = flow_jacobian = None
    count, ncoef = m*k+5*k, m*k
    if derivative:
        jac = np.zeros((t, f+2*j, count))
        jac[:, :f, :ncoef] = np.einsum(
            "tm,fk->tfmk", eeg_op@basis, eeg).reshape(t, f, ncoef)
        hb_jac = np.zeros((t, j, 2, count))
        flow_jacobian = np.zeros((t, k, count))
        for index, core in enumerate(cores):
            chain = np.zeros((t+5, count))
            chain[:t, index:ncoef:k] = basis
            chain[t:, ncoef+5*index:ncoef+5*(index+1)] = np.eye(5)
            latent_jac = core["jacobian"].reshape(t, 3, t+5)[:, 1:]@chain
            hb_jac += np.einsum("j,tcp->tjcp", gain*hb[:, index], latent_jac)
            flow_jacobian[:, index] = core["flow_jacobian"]@chain
        hb_jac = np.einsum("jst,tjcp->sjcp", hb_op, hb_jac)
        jac[:, f:] = hb_jac.reshape(t, 2*j, count)
        jacobian = jac.reshape(t*(f+2*j), count)
    return dict(prediction=prediction, eeg_prediction=eeg_prediction,
        hb_prediction=hb_prediction,
        canonical_prediction=np.column_stack((canonical_eeg, canonical_hb.reshape(t, 2*j))),
        canonical_eeg_prediction=canonical_eeg, canonical_hb_prediction=canonical_hb,
        latent_hb_prediction=latent_hb, drivers=drivers, states=states,
        initial_state=physical_initial, coefficients=coef.copy(),
        initial_coordinates=initial.copy(), jacobian=jacobian,
        flow_jacobian=flow_jacobian, total_unknowns=count)


def fit_spatial_shared_driver(eeg, hb, parameters, dt, *, eeg_loadings,
        hb_loadings, temporal_basis, sd, visible=None, hb_gains=None,
        eeg_operator=None, hb_operator=None, eeg_weight=1.,
        driver_curvature_weight=.01, initial_weight=100., flow_prior_weight=1.,
        flow_log_sd=np.log(2), max_nfev=100, gradient_tolerance=1e-6,
        substeps=4, numerical_backend="numba"):
    """Safeguarded analytic Gauss--Newton in fixed spatial coordinates.

    Each modality's data objective is its visible standardized SSE
    times T/n_visible, also multiplied by eeg_weight for EEG. With all targets
    visible these factors are eeg_weight/F and 1/(2J). The denominator depends
    only on the supplied mask; an empty modality contributes zero. This is an
    engineering objective, not a likelihood.

    Remaining terms are driver_curvature_weight*dt*sum((diff(driver,2)/dt**2)**2),
    initial_weight times squared physical departures from rest, and
    flow_prior_weight*dt*sum(log(f)**2)/flow_log_sd**2. No scale is target-fitted.

    Hidden values never enter initialization, residuals, or finiteness checks.
    EEG with zero weight is also excluded completely. Invalid physical trials
    are rejected and retained in failure_log. Every attempted spatial forward
    call counts against max_nfev, including failed calls. Success requires the
    scaled gradient criterion and physical validity in every Balloon core.
    """
    loading, mixing, basis, gains, eeg_op, hb_op = _validated_design(
        eeg_loadings, hb_loadings, temporal_basis, hb_gains, eeg_operator, hb_operator)
    parameters.validate()
    controls = [dt, eeg_weight, driver_curvature_weight, initial_weight,
                flow_prior_weight, flow_log_sd, gradient_tolerance]
    if (not np.isfinite(controls).all() or min(dt, flow_log_sd, gradient_tolerance) <= 0
            or min(eeg_weight, driver_curvature_weight, initial_weight, flow_prior_weight) < 0):
        raise ValueError("scales and tolerance must be positive finite; weights nonnegative finite")
    if (isinstance(max_nfev, bool) or not isinstance(max_nfev, (int, np.integer))
            or max_nfev < 1):
        raise ValueError("max_nfev must be a positive integer")
    t, m = basis.shape
    f, k = loading.shape
    j = len(mixing)
    e, h = np.asarray(eeg, float), np.asarray(hb, float)
    if e.shape != (t, f) or h.shape != (t, j, 2):
        raise ValueError("eeg and hb must match [T,F] and [T,J,2]")
    target = np.column_stack((e, h.reshape(t, 2*j)))
    mask = np.ones(target.shape, bool) if visible is None else np.asarray(visible)
    if mask.shape != target.shape or mask.dtype != np.bool_:
        raise ValueError("visible must be boolean [T,F+2J]")
    active = mask.copy()
    if eeg_weight == 0:
        active[:, :f] = False
    if not active.any() or not np.isfinite(target[active]).all():
        raise ValueError("at least one finite visible observation with positive weight is required")
    scale = np.asarray(sd, float)
    if scale.shape != (f+2*j,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("sd must be positive finite [F+2J]")
    eeg_count, hb_count = int(active[:, :f].sum()), int(active[:, f:].sum())
    denominators = dict(eeg=eeg_count/t if eeg_count else None,
                        hb=hb_count/t if hb_count else None)
    eeg_factor = np.sqrt(eeg_weight/denominators["eeg"]) if eeg_count else 0.
    hb_factor = 1/np.sqrt(denominators["hb"]) if hb_count else 0.
    weight = np.r_[np.full(f, eeg_factor), np.full(2*j, hb_factor)]/scale
    grid_weight = np.broadcast_to(weight, target.shape)
    eeg_rows = np.broadcast_to(np.arange(f+2*j) < f, target.shape)[active]
    ncoef, count = m*k, m*k+5*k
    curvature = np.zeros(((t-2)*k, count))
    second = np.sqrt(driver_curvature_weight)*np.diff(basis, n=2, axis=0)/dt**1.5
    for index in range(k):
        curvature[index::k, index:ncoef:k] = second
    failure_log, evaluations, iterations = [], 0, 0
    forward_kwargs = dict(eeg_loadings=loading, hb_loadings=mixing,
        temporal_basis=basis, hb_gains=gains, eeg_operator=eeg_op,
        hb_operator=hb_op, substeps=substeps, numerical_backend=numerical_backend)
    initial_factor = np.sqrt(initial_weight)
    flow_factor = np.sqrt(flow_prior_weight*dt)/flow_log_sd
    rest = np.r_[0., np.ones(4)]

    def evaluate(x, derivative):
        nonlocal evaluations
        evaluations += 1
        forward = forward_spatial_shared_driver(x[:ncoef].reshape(m, k),
            x[ncoef:].reshape(k, 5), parameters, dt,
            derivative=derivative, **forward_kwargs)
        data = (forward["prediction"][active]-target[active])*grid_weight[active]
        initial_residual = initial_factor*(forward["initial_state"]-rest).ravel()
        flow = forward["states"][:, :, 2]
        flow_residual = flow_factor*np.log(flow).ravel()
        curvature_residual = curvature@x
        residual = np.r_[data, curvature_residual, initial_residual, flow_residual]
        jac = None
        if derivative:
            initial_jac = np.zeros((5*k, count))
            initial_jac[:, ncoef:] = np.diag(initial_factor*np.column_stack(
                (np.ones(k), forward["initial_state"][:, 1:])).ravel())
            flow_jac = ((flow_factor/flow)[:, :, None]*forward["flow_jacobian"]).reshape(t*k, count)
            jac = np.vstack((forward["jacobian"][active.ravel()]*grid_weight[active, None],
                curvature, initial_jac, flow_jac))
        # Data rows retain time-major ordering, so identify their modality using
        # the same active indexing rather than regrouping observations.
        components = dict(eeg_data=float(data[eeg_rows]@data[eeg_rows]),
            hb_data=float(data[~eeg_rows]@data[~eeg_rows]),
            driver_curvature=float(curvature_residual@curvature_residual),
            initial_state=float(initial_residual@initial_residual),
            flow_prior=float(flow_residual@flow_residual))
        objective = float(residual@residual)
        if not np.isfinite(objective) or (derivative and not np.isfinite(jac).all()):
            raise FloatingPointError("nonfinite spatial objective or derivative")
        return forward, residual, jac, objective, components

    x = np.zeros(count)
    # Only positive-weight, visible EEG enters the linear warm start. In
    # particular Hb-only fitting cannot inherit any information from EEG.
    if eeg_weight > 0 and active[:, :f].any():
        design = np.einsum("tm,fk->tfmk", eeg_op@basis, loading).reshape(t, f, ncoef)
        emask = active[:, :f]
        system = design[emask]*np.broadcast_to(weight[:f], (t, f))[emask, None]
        rhs = e[emask]*np.broadcast_to(weight[:f], (t, f))[emask]
        x[:ncoef] = np.linalg.lstsq(np.vstack((system, curvature[:, :ncoef],
            1e-8*np.eye(ncoef))), np.r_[rhs, np.zeros(len(curvature)+ncoef)], rcond=1e-10)[0]
    initial_coefficients = x[:ncoef].reshape(m, k).copy()
    current = None
    initialization_attempts = min(22, max_nfev)
    for attempt in range(initialization_attempts):
        try:
            current = evaluate(x, True)
            break
        except (FloatingPointError, OverflowError) as exc:
            failure_log.append(dict(evaluation=evaluations, stage="initialization", error=str(exc)))
            if attempt+1 < initialization_attempts:
                x[:ncoef] *= .5
    support = dict(visible_count=int(mask.sum()), effective_count=int(active.sum()),
        weighting="visible_modality_mean", effective_denominators=denominators,
        eeg_counts=active[:, :f].sum(axis=0), hb_counts=active[:, f:].reshape(t, j, 2).sum(axis=0),
        eeg_loading_rank=int(np.linalg.matrix_rank(loading)),
        hb_loading_rank=int(np.linalg.matrix_rank(mixing)),
        visible_eeg_loading_rank=int(np.linalg.matrix_rank(loading[active[:, :f].any(axis=0)]))
            if active[:, :f].any() else 0,
        visible_hb_loading_rank=int(np.linalg.matrix_rank(mixing[active[:, f:].reshape(t, j, 2).any(axis=(0, 2))]))
            if active[:, f:].any() else 0,
        interpretation="fixed conditional observation support; not physiological identifiability")
    complexity = dict(driving_trajectories=k, temporal_modes=m, trajectory_parameters=ncoef,
        initial_parameters=5*k, total_parameters=count, extra_hb_parameters=0,
        graph_parameters=0, fixed_coordinate_gauge=True, weighting="visible_modality_mean",
        effective_denominators=denominators)
    if current is None:
        return dict(success=False, converged=False, status="failed_domain",
            convergence_reason="no_feasible_initialization", nfev=evaluations,
            evaluations=evaluations, iterations=0, objective=float("inf"),
            objective_components={}, failure_log=failure_log, support=support,
            complexity=complexity, total_unknowns=count, physical_checks=[],
            physical_check=dict(valid=False), initial_coefficients=initial_coefficients,
            prediction=np.full_like(target, np.nan), eeg_prediction=np.full_like(e, np.nan),
            hb_prediction=np.full_like(h, np.nan), states=np.full((t, k, 6), np.nan),
            drivers=basis@x[:ncoef].reshape(m, k), coefficients=x[:ncoef].reshape(m, k),
            initial_coordinates=x[ncoef:].reshape(k, 5), initial_state=np.tile(rest, (k, 1)),
            residual=np.full_like(target, np.nan), jacobian=None,
            gradient_norm=float("inf"), scaled_gradient_inf_norm=float("inf"))
    damping, converged, reason = 1e-5, False, "evaluation_budget"
    while True:
        forward, residual, jac, objective, components = current
        gradient, gram = jac.T@residual, jac.T@jac
        diagonal = np.maximum(np.diag(gram), 1.)
        scaled_gradient = float(np.max(np.abs(gradient)/np.sqrt(diagonal)))
        if scaled_gradient <= gradient_tolerance:
            converged, reason = True, "scaled_gradient"
            break
        if evaluations+2 > max_nfev:
            break
        iterations += 1
        try:
            delta = cho_solve(cho_factor(gram+damping*np.diag(diagonal), lower=True), -gradient)
        except np.linalg.LinAlgError as exc:
            failure_log.append(dict(evaluation=evaluations, stage="solve", error=str(exc)))
            reason = "singular_step"
            break
        if not np.isfinite(delta).all():
            reason = "nonfinite_step"
            break
        accepted = False
        for power in range(20):
            if evaluations+2 > max_nfev:
                break
            fraction = 2.**(-power)
            try:
                with np.errstate(over="raise", invalid="raise"):
                    candidate = x+fraction*delta
                    trial = evaluate(candidate, False)
                if (trial[3] >= objective
                        or trial[3] > objective+2e-4*fraction*float(gradient@delta)):
                    continue
                differentiated = evaluate(candidate, True)
            except (FloatingPointError, OverflowError) as exc:
                failure_log.append(dict(evaluation=evaluations, stage="candidate", error=str(exc)))
                continue
            x, current = candidate, differentiated
            damping = max(damping/3, 1e-12)
            accepted = True
            break
        if not accepted:
            damping *= 10
            if damping > 1e12 or np.linalg.norm(delta) <= 1e-12*(1+np.linalg.norm(x)):
                reason = "stagnation_without_gradient_convergence"
                break
    forward, residual, jac, objective, components = current
    required = ("finite", "positive_fvpq", "oxygen_extraction_in_unit_interval",
                "absolute_hb_nonnegative", "hbr_not_above_hbt", "rest_equilibrium")
    checks = []
    for index in range(k):
        check = dict(run_physical_checks(forward["states"][:, index], parameters))
        check["valid"] = all(check[name] for name in required)
        checks.append(check)
    physical_valid = all(check["valid"] for check in checks)
    success = bool(converged and physical_valid)
    if not success:
        failure_log.append(dict(evaluation=evaluations, stage="termination",
            reason=reason if physical_valid else "physical_checks_failed"))
    visible_residual = np.full_like(target, np.nan)
    visible_residual[active] = forward["prediction"][active]-target[active]
    support["driver_data_column_norms"] = np.linalg.norm(
        jac[:int(active.sum()), :ncoef], axis=0).reshape(m, k)
    return dict(**forward, residual=visible_residual, success=success,
        converged=bool(converged), status="completed" if success else
            "failed_physical" if not physical_valid else "failed_numerical",
        convergence_reason=reason, iterations=iterations, nfev=evaluations,
        evaluations=evaluations, objective=objective, objective_components=components,
        weighted_data_sse=components["eeg_data"]+components["hb_data"],
        data_sse=float(visible_residual[active]@visible_residual[active]),
        gradient_norm=float(np.max(np.abs(jac.T@residual))),
        scaled_gradient_inf_norm=scaled_gradient, physical_checks=checks,
        physical_check=dict(valid=physical_valid), support=support,
        complexity=complexity, failure_log=failure_log,
        initial_coefficients=initial_coefficients, eeg_weight=float(eeg_weight),
        driver_curvature_weight=float(driver_curvature_weight),
        initial_weight=float(initial_weight), flow_prior_weight=float(flow_prior_weight),
        flow_log_sd=float(flow_log_sd), weighting="visible_modality_mean",
        parameter_inference="fixed_physiology_conditional_reconstruction_no_posterior")


def fit_tied_spatial_shared_driver(eeg, hb, parameters, dt, *, eeg_loadings,
        hb_loadings, temporal_basis, sd, visible=None, hb_gains=None,
        eeg_operator=None, hb_operator=None, eeg_weight=1.,
        driver_curvature_weight=.01, initial_weight=100., flow_prior_weight=1.,
        flow_log_sd=np.log(2), max_nfev=100, gradient_tolerance=1e-6,
        substeps=4, numerical_backend="numba"):
    """Fit the exact original K2 objective restricted to identical trajectories.

    One effective core represents both original drivers and their identical
    five initial coordinates. Its EEG loading is L[:,0]+L[:,1], without
    normalization. Every original nonnegative row-sum-one Hb mixture therefore
    becomes a single loading of one. Fixed gains and observation operators are
    unchanged. Curvature, physical initial and log-flow weights are doubled
    because each original core contributes its own identical prior term.

    Returned coefficient/initial/state arrays have effective K=1. Replay uses
    ``collapse_tied_spatial_design`` or the returned ``forward_*_loadings``.
    The data mask and modality denominators are those of the original target.
    Degenerate readouts raise before fitting; there is no alternate refit.
    """
    collapsed = collapse_tied_spatial_design(eeg_loadings, hb_loadings)
    fitted = fit_spatial_shared_driver(eeg, hb, parameters, dt, **collapsed,
        temporal_basis=temporal_basis, sd=sd, visible=visible, hb_gains=hb_gains,
        eeg_operator=eeg_operator, hb_operator=hb_operator, eeg_weight=eeg_weight,
        driver_curvature_weight=2*driver_curvature_weight,
        initial_weight=2*initial_weight, flow_prior_weight=2*flow_prior_weight,
        flow_log_sd=flow_log_sd, max_nfev=max_nfev,
        gradient_tolerance=gradient_tolerance, substeps=substeps,
        numerical_backend=numerical_backend)
    metadata = dict(original_K=2, effective_K=1, tied=True,
        objective_exactness="exact_original_K2_objective_on_tied_subspace",
        regularization_multiplicity=2)
    fitted.update(metadata, forward_eeg_loadings=collapsed["eeg_loadings"],
        forward_hb_loadings=collapsed["hb_loadings"])
    fitted["complexity"].update(metadata,
        unrestricted_total_parameters=2*fitted["total_unknowns"])
    return fitted


__all__ = ["temporal_mode_basis", "forward_spatial_shared_driver",
    "fit_spatial_shared_driver", "collapse_tied_spatial_design",
    "fit_tied_spatial_shared_driver"]
