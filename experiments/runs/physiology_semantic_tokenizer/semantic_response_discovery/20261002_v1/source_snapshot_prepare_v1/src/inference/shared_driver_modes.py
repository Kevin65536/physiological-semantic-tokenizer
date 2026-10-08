"""Fixed spectral/spatial neural modes with the existing scalar Balloon core.

Modes are signed changes in baseline-relative log power. The second mode is
an effective spectral contrast, not a cellular synchrony or E/I estimate.
Loadings, observation operators, prior scales and physiology are supplied and
frozen before fitting a target. The fits are conditional reconstructions, not
calibrated physiological posteriors.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .shared_driver_reconstruction import nonlinear_driver_forward
from .t3a_balloon_robust_ssm import run_physical_checks


def spectral_loadings(channels=6, bands=5, spatial_weights=None):
    """Return fixed unit-norm [channel*band,2] loadings, channel-major.

    Bands 2 and 3 are the declared alpha/beta coordinates. The first column
    changes all bands in the same direction; the second contrasts alpha/beta
    against all remaining bands, with zero band mean in every channel.
    Nonnegative spatial weights preserve that sign convention. These are
    linear effective log-power readouts, not a raw EEG power generator.
    """
    for name, value, minimum in (("channels", channels, 1), ("bands", bands, 4)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    spatial = np.ones(channels) if spatial_weights is None else np.asarray(spatial_weights, float)
    if (spatial.shape != (channels,) or not np.isfinite(spatial).all()
            or np.any(spatial < 0) or not np.any(spatial > 0)):
        raise ValueError("spatial_weights must be finite nonnegative [channels], not all zero")
    contrast = np.full(bands, -2. / (bands-2))
    contrast[[2, 3]] = 1.
    loading = np.column_stack((np.outer(spatial, np.ones(bands)).ravel(),
                               np.outer(spatial, contrast).ravel()))
    return loading / np.linalg.norm(loading, axis=0)


def temporal_mode_basis(steps, modes=24):
    """Fixed DCT-II basis including DC, with RMS-one columns.

    Non-DC column k has period 2*steps*dt/k seconds. Coefficients therefore
    retain the signed trajectory units; no data-dependent temporal rotation
    or scale is introduced.
    """
    if (isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 3
            or isinstance(modes, bool) or not isinstance(modes, (int, np.integer))
            or not 1 <= modes <= steps):
        raise ValueError("basis needs integer steps >= 3 and 1 <= modes <= steps")
    basis = np.sqrt(2.)*np.cos(np.pi*(np.arange(steps)[:, None]+.5)*np.arange(modes)/steps)
    basis[:, 0] = 1.
    return basis


def _validated_design(loadings, temporal_basis, eeg_operator, hb_operator,
                      hb_basis, gamma, independent):
    loading, basis = np.asarray(loadings, float), np.asarray(temporal_basis, float)
    if (loading.ndim != 2 or loading.shape[1] not in (1, 2) or not loading.shape[0]
            or not np.isfinite(loading).all()
            or np.linalg.matrix_rank(loading) != loading.shape[1]):
        raise ValueError("loadings must be finite full-column-rank [C,1 or 2]")
    if (basis.ndim != 2 or basis.shape[0] < 3 or not 1 <= basis.shape[1] <= basis.shape[0]
            or not np.isfinite(basis).all() or np.linalg.matrix_rank(basis) != basis.shape[1]):
        raise ValueError("temporal_basis must be finite full-column-rank [T>=3,K<=T]")
    if not isinstance(independent, (bool, np.bool_)) or (independent and loading.shape[1] != 2):
        raise ValueError("independent requires two EEG modes and one separate Hb driver")
    if not np.isfinite(gamma) or (loading.shape[1] == 1 and gamma != 0):
        raise ValueError("gamma must be finite; scalar arms require gamma=0")
    steps = len(basis)
    operators = []
    for name, operator in (("eeg_operator", eeg_operator), ("hb_operator", hb_operator)):
        operator = np.eye(steps) if operator is None else np.asarray(operator, float)
        if operator.shape != (steps, steps) or not np.isfinite(operator).all():
            raise ValueError(f"{name} must be finite [T,T]")
        operators.append(operator)
    common = np.empty((steps, 2, 0)) if hb_basis is None else np.asarray(hb_basis, float)
    if (common.ndim != 3 or common.shape[:2] != (steps, 2)
            or not np.isfinite(common).all() or common.shape[2] >= steps):
        raise ValueError("hb_basis must be finite processed-coordinate [T,2,Kc<T]")
    return loading, basis, *operators, common


def forward_shared_driver_modes(coefficients, initial_coordinates, parameters, dt, *,
        loadings, temporal_basis, eeg_operator=None, hb_operator=None, hb_basis=None,
        component_coefficients=None, gamma=0., independent=False, substeps=4,
        derivative=True, numerical_backend="numba"):
    """Forward and analytic Jacobian in fixed coordinates, without clipping.

    ``coefficients`` is [K,d+independent]. The first d columns are EEG modes;
    shared Hb input is (a+gamma*b)/sqrt(1+gamma**2) (or a for scalar), while the independent arm
    uses a third trajectory. Initial coordinates are [s0,logf0,logv0,logp0,
    logq0]. Jacobian columns are coefficients.ravel(C-order), initial[5],
    component coefficients. Operators act after physical mean generation.
    ``hb_basis`` already lives in processed observation coordinates, so it is
    not processed twice. Invalid RK stages raise FloatingPointError.
    """
    loading, basis, eeg_op, hb_op, common = _validated_design(
        loadings, temporal_basis, eeg_operator, hb_operator, hb_basis, gamma, independent)
    steps, modes = basis.shape
    channels, dimensions = loading.shape
    drivers_count = dimensions+int(independent)
    coef, initial = np.asarray(coefficients, float), np.asarray(initial_coordinates, float)
    cc = (np.zeros(common.shape[2]) if component_coefficients is None
          else np.asarray(component_coefficients, float))
    if coef.shape != (modes, drivers_count) or not np.isfinite(coef).all():
        raise ValueError("coefficients must be finite [K,d+independent]")
    if initial.shape != (5,) or not np.isfinite(initial).all():
        raise ValueError("initial_coordinates must be finite [5]")
    if cc.shape != (common.shape[2],) or not np.isfinite(cc).all():
        raise ValueError("component_coefficients must be finite [Kc]")
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        physical_initial = np.r_[initial[0], np.exp(initial[1:])]
    drivers = basis@coef
    hb_weights = np.zeros(drivers_count)
    if independent:
        hb_weights[-1] = 1.
    else:
        hb_weights[0] = 1.
        if dimensions == 2:
            hb_weights[1] = gamma
            hb_weights /= np.sqrt(1+gamma**2)
    u_h = drivers@hb_weights
    balloon = nonlinear_driver_forward(u_h, physical_initial, parameters, dt,
        substeps=substeps, derivative=derivative, numerical_backend=numerical_backend)
    canonical_eeg = drivers[:, :dimensions]@loading.T
    canonical_hb = balloon["canonical_prediction"][:, 1:]
    canonical = np.column_stack((canonical_eeg, canonical_hb))
    physical = np.column_stack((eeg_op@canonical_eeg, hb_op@canonical_hb))
    component = np.zeros_like(physical)
    component[:, channels:] = np.einsum("tck,k->tc", common, cc)
    prediction = physical+component
    jacobian = None
    if derivative:
        count = modes*drivers_count+5+len(cc)
        jac = np.zeros((steps, channels+2, count))
        eeg_design = np.zeros((steps, channels, modes, drivers_count))
        eeg_design[:, :, :, :dimensions] = np.einsum("tk,cd->tckd", eeg_op@basis, loading)
        jac[:, :channels, :modes*drivers_count] = eeg_design.reshape(steps, channels, -1)
        chain = np.zeros((steps+5, modes*drivers_count+5))
        chain[:steps, :modes*drivers_count] = (basis[:, :, None]*hb_weights).reshape(steps, -1)
        chain[steps:, modes*drivers_count:] = np.eye(5)
        hb_jac = balloon["jacobian"].reshape(steps, 3, steps+5)[:, 1:]
        hb_jac = np.einsum("st,tcp->scp", hb_op, hb_jac)@chain
        jac[:, channels:, :modes*drivers_count+5] = hb_jac
        jac[:, channels:, modes*drivers_count+5:] = common
        jacobian = jac.reshape(steps*(channels+2), count)
    return dict(prediction=prediction, eeg_prediction=prediction[:, :channels],
        hb_prediction=prediction[:, channels:], canonical_prediction=canonical,
        physical_prediction=physical, observation_component=component,
        r=drivers[:, :dimensions], drivers=drivers, u_h=u_h, states=balloon["states"],
        initial_state=physical_initial, coefficients=coef.copy(),
        initial_coordinates=initial.copy(), component_coefficients=cc.copy(),
        jacobian=jacobian)


def fit_shared_driver_modes(eeg, hb, parameters, dt, *, loadings, temporal_basis,
        sd, visible=None, eeg_operator=None, hb_operator=None, hb_basis=None,
        gamma=0., independent=False, state_sd=.1, ar_phi=.95, ar_weight=.1,
        initial_weight=100., component_sd=None, max_nfev=100, substeps=4,
        numerical_backend="numba", gradient_tolerance=1e-6):
    """Visible-only conditional MAP fit using safeguarded analytic Gauss--Newton.

    The stationary AR(1) penalty is sqrt(ar_weight)*r0/state_sd followed by
    sqrt(ar_weight)*(r[t]-phi*r[t-1])/(state_sd*sqrt(1-phi**2)), separately for
    every trajectory. The initial penalty is sqrt(initial_weight) times the
    five physical departures from rest. Common coefficients use c/component_sd.
    All scales are caller-supplied training values; none is estimated here.
    EEG coordinate residuals get 1/sqrt(C), Hb gets 1/sqrt(2), balancing total
    modality weight. The same weights apply when a modality is partly hidden.
    Hidden targets never enter initialization, fitting or returned residuals.

    Invalid candidate RK stages are rejected and logged, never clipped or
    replaced by artificial residuals. Every forward call, including rejected
    trials, counts toward max_nfev. Success requires actual scaled-gradient
    stationarity and physical checks, not a small step or objective change.
    """
    loading, basis, eeg_op, hb_op, common = _validated_design(
        loadings, temporal_basis, eeg_operator, hb_operator, hb_basis, gamma, independent)
    parameters.validate()
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite positive")
    if (not np.isfinite([ar_phi, ar_weight, initial_weight, gradient_tolerance]).all()
            or not -1 < ar_phi < 1 or ar_weight < 0 or initial_weight < 0
            or gradient_tolerance <= 0):
        raise ValueError("AR must be stable; weights nonnegative and gradient tolerance positive")
    if (isinstance(max_nfev, bool) or not isinstance(max_nfev, (int, np.integer))
            or max_nfev < 1):
        raise ValueError("max_nfev must be a positive integer")
    steps, modes = basis.shape
    channels, dimensions = loading.shape
    driver_count, common_count = dimensions+int(independent), common.shape[2]
    e, h = np.asarray(eeg, float), np.asarray(hb, float)
    if e.shape != (steps, channels) or h.shape != (steps, 2):
        raise ValueError("eeg and hb must match [T,C] and [T,2]")
    target = np.column_stack((e, h))
    mask = np.isfinite(target) if visible is None else np.asarray(visible)
    if mask.shape != target.shape or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("visible must be nonempty boolean [T,C+2]")
    if not np.isfinite(target[mask]).all():
        raise ValueError("visible observations must be finite")
    scale = np.asarray(sd, float)
    if scale.shape == (channels+2,):
        scale = np.broadcast_to(scale, target.shape)
    if scale.shape != target.shape or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("sd must be positive finite [C+2] or [T,C+2]")
    prior_sd = np.broadcast_to(np.asarray(state_sd, float), (driver_count,)).copy()
    if not np.isfinite(prior_sd).all() or np.any(prior_sd <= 0):
        raise ValueError("state_sd must be positive finite scalar or [drivers]")
    if common_count and component_sd is None:
        raise ValueError("training component_sd is required with hb_basis")
    cs = np.broadcast_to(np.asarray(1. if component_sd is None else component_sd, float), (common_count,)).copy()
    if not np.isfinite(cs).all() or np.any(cs <= 0):
        raise ValueError("component_sd must be positive finite scalar or [Kc]")
    weight = 1./(scale*np.r_[np.full(channels, np.sqrt(channels)), np.sqrt(2.), np.sqrt(2.)])
    ncoef, count = modes*driver_count, modes*driver_count+5+common_count
    ar = basis.copy()
    ar[1:] = (basis[1:]-ar_phi*basis[:-1])/np.sqrt(1-ar_phi**2)
    ar_design = np.zeros((steps, driver_count, modes, driver_count))
    for j in range(driver_count):
        ar_design[:, j, :, j] = np.sqrt(ar_weight)*ar/prior_sd[j]
    regularizer = np.zeros((steps*driver_count+common_count, count))
    regularizer[:steps*driver_count, :ncoef] = ar_design.reshape(steps*driver_count, ncoef)
    if common_count:
        regularizer[steps*driver_count:, ncoef+5:] = np.diag(1./cs)
    failure_log, nfev, iterations = [], 0, 0

    def evaluate(x, derivative):
        nonlocal nfev
        nfev += 1
        forward = forward_shared_driver_modes(x[:ncoef].reshape(modes, driver_count),
            x[ncoef:ncoef+5], parameters, dt, loadings=loading, temporal_basis=basis,
            eeg_operator=eeg_op, hb_operator=hb_op, hb_basis=common,
            component_coefficients=x[ncoef+5:], gamma=gamma, independent=independent,
            substeps=substeps, derivative=derivative, numerical_backend=numerical_backend)
        data = (forward["prediction"][mask]-target[mask])*weight[mask]
        initial = np.sqrt(initial_weight)*(forward["initial_state"]-np.r_[0., np.ones(4)])
        residual = np.r_[data, regularizer@x, initial]
        jac = None
        if derivative:
            initial_jac = np.zeros((5, count))
            initial_jac[:, ncoef:ncoef+5] = np.sqrt(initial_weight)*np.diag(np.r_[1., forward["initial_state"][1:]])
            jac = np.vstack((forward["jacobian"][mask.ravel()]*weight[mask, None],
                             regularizer, initial_jac))
        return forward, residual, jac, float(residual@residual)

    # A linear visible EEG solve supplies only the EEG modes. Hb starts at rest.
    eeg_design = np.zeros((steps, channels, modes, driver_count))
    eeg_design[:, :, :, :dimensions] = np.einsum("tk,cd->tckd", eeg_op@basis, loading)
    eeg_mask = mask[:, :channels]
    system = eeg_design.reshape(steps, channels, ncoef)[eeg_mask]*weight[:, :channels][eeg_mask, None]
    rhs = target[:, :channels][eeg_mask]*weight[:, :channels][eeg_mask]
    x = np.zeros(count)
    x[:ncoef] = np.linalg.lstsq(np.vstack((system, regularizer[:, :ncoef], 1e-8*np.eye(ncoef))),
        np.r_[rhs, np.zeros(len(regularizer)+ncoef)], rcond=1e-10)[0]
    current = None
    # Feasible initialization is found by reducing initial coefficients, with
    # every rejection retained; the physical trajectory itself is never altered.
    for _ in range(min(21, max_nfev)):
        try:
            current = evaluate(x, True)
            break
        except (FloatingPointError, OverflowError) as exc:
            failure_log.append(dict(evaluation=nfev, stage="initialization", error=str(exc)))
            x[:ncoef] *= .5
    support = dict(visible_count=int(mask.sum()), eeg_counts=mask[:, :channels].sum(axis=0),
        hb_counts=mask[:, channels:].sum(axis=0), loading_rank=dimensions,
        interpretation="conditional observation support; not calibrated mode certainty")
    complexity = dict(eeg_modes=dimensions, driving_trajectories=driver_count,
        temporal_modes=modes, trajectory_parameters=ncoef, initial_parameters=5,
        common_parameters=common_count, total_parameters=count,
        fixed_coordinate_gauge=True, independent_hb_driver=bool(independent))
    if current is None:
        return dict(success=False, converged=False, status="failed_domain", iterations=0,
            nfev=nfev, evaluations=nfev, objective=float("inf"), gradient_norm=float("inf"),
            failure_log=failure_log, support=support, complexity=complexity,
            physical_check=dict(valid=False))
    damping, success, reason = 1e-5, False, "evaluation_budget"
    while True:
        forward, residual, jac, objective = current
        gradient, gram = jac.T@residual, jac.T@jac
        diagonal = np.maximum(np.diag(gram), 1.)
        scaled_gradient = float(np.max(np.abs(gradient)/np.sqrt(diagonal)))
        if scaled_gradient <= gradient_tolerance:
            success, reason = True, "scaled_gradient"
            break
        if nfev+2 > max_nfev:
            break
        iterations += 1
        try:
            delta = cho_solve(cho_factor(gram+damping*np.diag(diagonal), lower=True), -gradient)
        except np.linalg.LinAlgError as exc:
            failure_log.append(dict(evaluation=nfev, stage="solve", error=str(exc)))
            reason = "singular_step"
            break
        accepted = False
        for power in range(20):
            if nfev+2 > max_nfev:
                break
            fraction = 2.**(-power)
            candidate = x+fraction*delta
            try:
                trial = evaluate(candidate, False)
                if trial[3] >= objective or trial[3] > objective+2e-4*fraction*float(gradient@delta):
                    continue
                differentiated = evaluate(candidate, True)
            except (FloatingPointError, OverflowError) as exc:
                failure_log.append(dict(evaluation=nfev, stage="candidate", error=str(exc)))
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
    forward, residual, jac, objective = current
    checks = dict(run_physical_checks(forward["states"], parameters))
    required = ("finite", "positive_fvpq", "oxygen_extraction_in_unit_interval",
                "absolute_hb_nonnegative", "hbr_not_above_hbt", "rest_equilibrium")
    physical_valid = all(checks[k] for k in required)
    success = bool(success and physical_valid)
    checks["valid"] = physical_valid
    visible_residual = np.full_like(target, np.nan)
    visible_residual[mask] = forward["prediction"][mask]-target[mask]
    weighted_data_sse = float(np.sum((visible_residual[mask]*weight[mask])**2))
    # Local sensitivities describe the fixed coordinate directions; they are
    # not converted into posteriors or a claim that both modalities see each mode.
    support["driver_data_column_norms"] = np.linalg.norm(jac[:int(mask.sum()), :ncoef], axis=0).reshape(modes, driver_count)
    return dict(**forward, residual=visible_residual, success=success, converged=success,
        status="completed" if success else "failed_physical" if not physical_valid else "failed_numerical",
        convergence_reason=reason, iterations=iterations, nfev=nfev, evaluations=nfev,
        objective=objective, weighted_data_sse=weighted_data_sse,
        data_sse=float(visible_residual[mask]@visible_residual[mask]),
        gradient_norm=float(np.max(np.abs(jac.T@residual))),
        scaled_gradient_inf_norm=scaled_gradient, physical_check=checks,
        support=support, complexity=complexity, failure_log=failure_log,
        ar_phi=float(ar_phi), ar_weight=float(ar_weight), state_sd=prior_sd,
        initial_weight=float(initial_weight), component_sd=cs,
        parameter_inference="fixed_physiology_conditional_reconstruction_no_posterior")
