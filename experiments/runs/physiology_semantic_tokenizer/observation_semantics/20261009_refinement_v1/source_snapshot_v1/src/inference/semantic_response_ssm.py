"""EEG-owned continuous modes and fixed-driver Hb response comparisons.

This array-only library neither loads recordings nor selects a data partition.
Training normalization, projection weights and predictive noise scales belong
to the caller. Initial-state priors are explicit engineering priors; finite
states and a fitted response do not establish physiological source semantics.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr

from .shared_driver_modes import temporal_mode_basis
from .shared_driver_reconstruction import nonlinear_driver_forward
from .t3a_balloon_robust_ssm import run_physical_checks


def _operator(value, steps, name):
    result = np.eye(steps) if value is None else np.asarray(value, float)
    if result.shape != (steps, steps) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite [T,T]")
    return result


def _visible(values, supplied):
    result = np.isfinite(values) if supplied is None else np.asarray(supplied)
    if result.shape != values.shape or result.dtype != np.bool_:
        raise ValueError("visible must be boolean with the observation shape")
    if not np.isfinite(values[result]).all():
        raise ValueError("visible observations must be finite")
    return result


def _positive_scale(value, shape, name):
    try:
        result = np.broadcast_to(np.asarray(value, float), shape)
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to {shape}") from exc
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def infer_eeg_modes(eeg, loadings, sd, *, temporal_basis=None, eeg_operator=None,
                    visible=None, state_sd=.1, ar_phi=.95, ar_weight=.1):
    """Infer two fixed-coordinate trajectories using EEG observations alone.

    Inputs are EEG [T,C], fixed full-rank loadings [C,2] and caller-frozen
    observation SD. A supplied temporal basis [T,K] keeps its own units; the
    default is the existing RMS-one DCT basis with at most 24 columns. The
    stationary AR(1) penalty acts on trajectories, never on basis coefficients.
    Data residuals use 1/(SD*sqrt(C)), retaining the declared coordinate-average
    EEG weight. Hidden coordinates cannot affect the linear solve. No Hb values
    or per-target scaling are accepted or estimated.
    """
    y, loading = np.asarray(eeg, float), np.asarray(loadings, float)
    if y.ndim != 2 or y.shape[0] < 3 or not y.shape[1]:
        raise ValueError("eeg must be [T>=3,C>=1]")
    steps, channels = y.shape
    if (loading.shape != (channels, 2) or not np.isfinite(loading).all()
            or np.linalg.matrix_rank(loading) != 2):
        raise ValueError("loadings must be finite full-column-rank [C,2]")
    basis = (temporal_mode_basis(steps, min(24, steps)) if temporal_basis is None
             else np.asarray(temporal_basis, float))
    if (basis.ndim != 2 or basis.shape[0] != steps or not 1 <= basis.shape[1] <= steps
            or not np.isfinite(basis).all() or np.linalg.matrix_rank(basis) != basis.shape[1]):
        raise ValueError("temporal_basis must be finite full-column-rank [T,K<=T]")
    if (not np.isfinite([ar_phi, ar_weight]).all() or not -1 < ar_phi < 1
            or ar_weight < 0):
        raise ValueError("AR phi must be stable and its weight nonnegative")
    mask = _visible(y, visible)
    scale = _positive_scale(sd, y.shape, "sd")
    prior_sd = _positive_scale(state_sd, (2,), "state_sd")
    operator = _operator(eeg_operator, steps, "eeg_operator")
    modes = basis.shape[1]
    design = np.einsum("tk,cd->tckd", operator@basis, loading).reshape(steps, channels, 2*modes)
    weight = 1/(np.sqrt(channels)*scale)
    ar = basis.copy()
    ar[1:] = (basis[1:]-ar_phi*basis[:-1])/np.sqrt(1-ar_phi**2)
    regularizer = np.zeros((steps, 2, modes, 2))
    for coordinate in range(2):
        regularizer[:, coordinate, :, coordinate] = np.sqrt(ar_weight)*ar/prior_sd[coordinate]
    regularizer = regularizer.reshape(2*steps, 2*modes)
    system = np.vstack((design[mask]*weight[mask, None], regularizer))
    target = np.r_[y[mask]*weight[mask], np.zeros(2*steps)]
    coefficients, _, rank, singular = np.linalg.lstsq(system, target, rcond=1e-10)
    coefficients = coefficients.reshape(modes, 2)
    trajectories = basis@coefficients
    prediction = operator@(trajectories@loading.T)
    residual = np.full_like(y, np.nan)
    residual[mask] = prediction[mask]-y[mask]
    return dict(r=trajectories, a=trajectories[:, 0], b=trajectories[:, 1],
        prediction=prediction, coefficients=coefficients, residual=residual,
        visible_count=int(mask.sum()), rank=int(rank), singular_values=singular,
        state_sd=prior_sd.copy(), ar_phi=float(ar_phi), ar_weight=float(ar_weight),
        support=dict(coordinate_counts=mask.sum(axis=0), observation_owner="EEG_only",
                     temporal_scope="offline_full_window", fixed_loading_rank=2),
        source_status="fixed_effective_spectral_coordinates_not_identified_neural_sources")


@njit(cache=True)
def _physical_response_slope(x, tangent, value, parameters, tau_v, derivative):
    alpha, e0, gamma, p0, q0, gain, kappa, tau = parameters
    if not np.isfinite(x).all() or min(x[1:]) <= 0 or q0*x[4] > p0*x[3]:
        raise FloatingPointError("invalid physical response RK stage")
    s, f, v, p, q = x
    elastic = v**(1/alpha)
    outflow = (tau*elastic+tau_v*f)/(tau+tau_v)
    loss = outflow/v
    log_complement = np.log1p(-e0)/f
    extraction = -np.expm1(log_complement)
    rhs = np.array([gain*value-kappa*s-gamma*(f-1), s,
        (f-elastic)/(tau+tau_v), (f-loss*p)/tau,
        (f*extraction/e0-loss*q)/tau])
    result = np.zeros_like(tangent)
    if derivative:
        outflow_f = tau_v/(tau+tau_v)
        outflow_v = tau/(tau+tau_v)*elastic/(alpha*v)
        loss_f, loss_v = outflow_f/v, outflow_v/v-outflow/v**2
        jac = np.zeros((5, 5))
        jac[0, 0], jac[0, 1], jac[1, 0] = -kappa, -gamma, 1.
        jac[2, 1], jac[2, 2] = 1/(tau+tau_v), -elastic/(alpha*v*(tau+tau_v))
        jac[3, 1], jac[3, 2], jac[3, 3] = (1-p*loss_f)/tau, -p*loss_v/tau, -loss/tau
        extraction_slope = (extraction+np.exp(log_complement)*log_complement)/e0
        jac[4, 1], jac[4, 2], jac[4, 4] = (extraction_slope-q*loss_f)/tau, -q*loss_v/tau, -loss/tau
        result = jac@tangent
    if not np.isfinite(rhs).all() or not np.isfinite(result).all():
        raise FloatingPointError("nonfinite physical response derivative")
    return rhs, result


@njit(cache=True)
def _extended_integrate(driver, initial, parameters, dt, substeps, tau_n, tau_v, derivative):
    steps = len(driver)
    states = np.empty((steps, 6))
    tangents = np.zeros((steps, 5, 5 if derivative else 0))
    vascular_input = np.empty(steps)
    vascular_input[0] = 0. if tau_n > 0 else driver[0]
    x = initial.copy()
    tangent = np.zeros((5, 5 if derivative else 0))
    if derivative:
        tangent[0, 0] = 1.
        for coordinate in range(1, 5):
            tangent[coordinate, coordinate] = initial[coordinate]
    states[0, 0], states[0, 1:] = vascular_input[0], x
    tangents[0] = tangent
    h = dt/substeps
    for t in range(steps-1):
        for j in range(substeps):
            u1 = u2 = u4 = driver[t]
            if tau_n > 0:
                departure = vascular_input[t]-driver[t]
                u1 += departure*np.exp(-j*h/tau_n)
                u2 += departure*np.exp(-(j+.5)*h/tau_n)
                u4 += departure*np.exp(-(j+1)*h/tau_n)
            k1, d1 = _physical_response_slope(x, tangent, u1, parameters, tau_v, derivative)
            k2, d2 = _physical_response_slope(x+h*k1/2, tangent+h*d1/2, u2, parameters, tau_v, derivative)
            k3, d3 = _physical_response_slope(x+h*k2/2, tangent+h*d2/2, u2, parameters, tau_v, derivative)
            k4, d4 = _physical_response_slope(x+h*k3, tangent+h*d3, u4, parameters, tau_v, derivative)
            x = x+h*(k1+2*k2+2*k3+k4)/6
            tangent = tangent+h*(d1+2*d2+2*d3+d4)/6
            _physical_response_slope(x, tangent, u4, parameters, tau_v, False)
        vascular_input[t+1] = (driver[t]+(vascular_input[t]-driver[t])*np.exp(-dt/tau_n)
                               if tau_n > 0 else driver[t+1])
        states[t+1, 0], states[t+1, 1:] = vascular_input[t+1], x
        tangents[t+1] = tangent
    return states, tangents, vascular_input


def _parameter_array(parameters):
    fixed, free = parameters.fixed, parameters.free
    return np.array([fixed.alpha, fixed.E0, fixed.gamma, fixed.P0, fixed.Q0,
                     fixed.neurovascular_gain, free.kappa, free.tau])


def _initial_domain(initial, parameters):
    state = np.asarray(initial, float)
    if state.shape != (5,) or not np.isfinite(state).all():
        raise ValueError("initial_state must be finite physical [s0,f0,v0,p0,q0]")
    if np.any(state[1:] <= 0) or parameters.fixed.Q0*state[4] > parameters.fixed.P0*state[3]:
        raise FloatingPointError("initial state outside positive flow/volume/Hb domain")
    return state


def _response_arguments(driver, parameters, dt, initial_state, tau_n, tau_v, substeps, backend):
    parameters.validate()
    r = np.asarray(driver, float)
    if r.ndim != 1 or len(r) < 3 or not np.isfinite(r).all():
        raise ValueError("driver must be finite [T>=3]")
    if not np.isfinite([dt, tau_n, tau_v]).all() or dt <= 0 or min(tau_n, tau_v) < 0:
        raise ValueError("dt must be positive and tau_n/tau_v nonnegative finite")
    if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    if backend not in ("python", "numba"):
        raise ValueError("numerical_backend must be python or numba")
    initial = _initial_domain(np.r_[0., np.ones(4)] if initial_state is None else initial_state, parameters)
    return r, initial


def response_rhs(vascular_state, vascular_input, parameters, *, tau_v=0.):
    """Physical derivative [s,f,v,p,q], with one consistent venous outflow.

    tau_v is a nonnegative time in seconds. It modifies volume compliance and
    the same outflow in both Hb balances; it never rescales neural gain.
    """
    parameters.validate()
    x = _initial_domain(vascular_state, parameters)
    if not np.isfinite([vascular_input, tau_v]).all() or tau_v < 0:
        raise ValueError("input must be finite and tau_v nonnegative")
    return _physical_response_slope(x, np.empty((5, 0)), float(vascular_input),
                                    _parameter_array(parameters), float(tau_v), False)[0]


def fixed_driver_response(driver, parameters, dt, *, initial_state=None, tau_n=0.,
                          tau_v=0., substeps=4, hb_operator=None, derivative=False,
                          numerical_backend="numba"):
    """Nonlinear Hb response to a fixed supplied trajectory, without Hb fitting.

    A first-order input filter has initial value zero and exact ZOH values at
    every RK stage: du/dt=(driver-u)/tau_n. tau_n=0 means instantaneous input.
    Viscoelastic outflow satisfies tau*dv/dt=f-f_out, with
    f_out=(tau*v**(1/alpha)+tau_v*f)/(tau+tau_v). The existing inlet total-Hb
    and deoxy-Hb mass balances use this same f_out. Zero extensions call the
    original forward function. No state or RK stage is clipped.

    States [T,6] are [actual vascular input,s,f,v,p,q]; original driver is
    returned separately. Initial Jacobian [T,2,5] uses columns
    [s0,logf0,logv0,logp0,logq0]. The Hb operator acts after physical generation.
    """
    r, initial = _response_arguments(driver, parameters, dt, initial_state, tau_n, tau_v,
                                     substeps, numerical_backend)
    operator = _operator(hb_operator, len(r), "hb_operator")
    initial_jacobian = None
    if tau_n == 0 and tau_v == 0:
        original = nonlinear_driver_forward(r, initial, parameters, dt, substeps=substeps,
            derivative=derivative, numerical_backend=numerical_backend)
        states, canonical = original["states"], original["canonical_prediction"][:, 1:]
        vascular_input = r.copy()
        if derivative:
            initial_jacobian = original["jacobian"].reshape(len(r), 3, len(r)+5)[:, 1:, -5:]
    else:
        integrate = _extended_integrate if numerical_backend == "numba" else _extended_integrate.py_func
        states, tangents, vascular_input = integrate(r, initial, _parameter_array(parameters),
            float(dt), int(substeps), float(tau_n), float(tau_v), bool(derivative))
        fixed = parameters.fixed
        canonical = np.column_stack((fixed.P0*(states[:, 4]-1)-fixed.Q0*(states[:, 5]-1),
                                     fixed.Q0*(states[:, 5]-1)))
        if derivative:
            initial_jacobian = np.stack((fixed.P0*tangents[:, 3]-fixed.Q0*tangents[:, 4],
                                        fixed.Q0*tangents[:, 4]), axis=1)
    if derivative:
        initial_jacobian = np.einsum("st,tck->sck", operator, initial_jacobian)
    outflow = (parameters.free.tau*states[:, 3]**(1/parameters.fixed.alpha)
               +tau_v*states[:, 2])/(parameters.free.tau+tau_v)
    prediction = operator@canonical
    checks = dict(run_physical_checks(states, parameters))
    checks["valid"] = all(checks[k] for k in ("finite", "positive_fvpq",
        "oxygen_extraction_in_unit_interval", "absolute_hb_nonnegative", "hbr_not_above_hbt", "rest_equilibrium"))
    return dict(driver=r.copy(), states=states, vascular_input=vascular_input,
        initial_state=initial.copy(), canonical_hb=canonical, hb_prediction=prediction,
        prediction=prediction, initial_jacobian=initial_jacobian, outflow=outflow,
        physical_check=checks, tau_n=float(tau_n), tau_v=float(tau_v),
        initial_coordinate_order=["s0", "logf0", "logv0", "logp0", "logq0"],
        interpretation="fixed_EEG_driver_response_not_physiological_source_truth")


def _decode_initial(coordinates, mode, parameters):
    if mode == "rest":
        return np.r_[0., np.ones(4)], np.empty((5, 0))
    if mode == "legacy_free_physicalprior":
        initial = _initial_domain(coordinates, parameters)
        return initial, np.diag(1/np.r_[1., initial[1:]])
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        s, f, v, q = coordinates[0], *np.exp(coordinates[1:])
    initial = _initial_domain(np.array([s, f, v, v, q]), parameters)
    chain = np.array([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0],
                      [0, 0, 1., 0], [0, 0, 0, 1.]])
    return initial, chain


def fit_fixed_driver_response(hb, driver, parameters, dt, *, sd, visible=None,
        hb_operator=None, hb_basis=None, initial_mode="rest", initial_prior_mean=None,
        initial_prior_sd=None, initial_weight=1., component_sd=1., max_nfev=80,
        gradient_tolerance=1e-6, tau_n=0., tau_v=0., substeps=4, numerical_backend="numba"):
    """Fit visible Hb initial/component nuisance coordinates with driver frozen.

    rest fixes the physical initial state. tied_pv_logprior has coordinates
    [s0,logf0,logv0,logq0] and sets p0=v0; legacy_free_physicalprior uses
    physical [s0,f0,v0,p0,q0] as an optional diagnostic. Caller-frozen mean/SD
    and positive prior weight define proper Gaussian coordinate penalties.
    Default means are rest, default SD is .1 in the chosen coordinates.

    hb_basis is a fixed, already-processed [T,2,4] cosine/component design.
    Its zero-mean coefficient prior is proper. Only visible values enter the
    fit, and allfalse visibility returns a prior_prediction with no fitted
    convergence. Data residuals are divided by caller-supplied SD, without
    re-estimating driver, scales or future Hb. Every rejected forward counts
    toward max_nfev; actual scaled-gradient convergence is required.
    """
    y = np.asarray(hb, float)
    if y.ndim != 2 or y.shape[1] != 2 or len(y) < 3:
        raise ValueError("hb must be [T>=3,2]")
    r, _ = _response_arguments(driver, parameters, dt, None, tau_n, tau_v, substeps, numerical_backend)
    if len(r) != len(y):
        raise ValueError("driver and Hb clocks must match")
    operator = _operator(hb_operator, len(r), "hb_operator")
    mask, scale = _visible(y, visible), _positive_scale(sd, y.shape, "sd")
    modes = {"rest": 0, "tied_pv_logprior": 4, "legacy_free_physicalprior": 5}
    if initial_mode not in modes:
        raise ValueError("unknown initial_mode")
    ninitial = modes[initial_mode]
    if (not np.isfinite([initial_weight, gradient_tolerance]).all() or initial_weight < 0
            or gradient_tolerance <= 0 or (ninitial and initial_weight == 0)):
        raise ValueError("initial priors must be proper and gradient tolerance positive")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, (int, np.integer)) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    mean = (np.zeros(ninitial) if initial_mode != "legacy_free_physicalprior"
            else np.r_[0., np.ones(4)]) if initial_prior_mean is None else np.asarray(initial_prior_mean, float)
    if mean.shape != (ninitial,) or not np.isfinite(mean).all():
        raise ValueError("initial_prior_mean must match the chosen coordinates")
    if not ninitial and initial_prior_sd is not None:
        raise ValueError("rest fixes initial state and takes no initial prior SD")
    prior_sd = _positive_scale(.1 if initial_prior_sd is None else initial_prior_sd,
                                (ninitial,), "initial_prior_sd")
    _decode_initial(mean, initial_mode, parameters)
    common = np.empty((len(y), 2, 0)) if hb_basis is None else np.asarray(hb_basis, float)
    if (common.ndim != 3 or common.shape[:2] != y.shape or common.shape[2] not in (0, 4)
            or not np.isfinite(common).all() or common.shape[2] >= len(y)):
        raise ValueError("hb_basis must be finite processed-coordinate [T,2,4] or absent")
    ncommon = common.shape[2]
    coefficient_sd = _positive_scale(component_sd, (ncommon,), "component_sd")
    count = ninitial+ncommon
    x = np.r_[mean, np.zeros(ncommon)]
    prior_design = np.zeros((count, count))
    if ninitial:
        prior_design[:ninitial, :ninitial] = np.diag(np.sqrt(initial_weight)/prior_sd)
    if ncommon:
        prior_design[ninitial:, ninitial:] = np.diag(1/coefficient_sd)
    prior_center = x.copy()
    evaluations, iterations, failure_log = 0, 0, []

    def evaluate(coordinates, derivative):
        nonlocal evaluations
        evaluations += 1
        initial, chain = _decode_initial(coordinates[:ninitial], initial_mode, parameters)
        forward = fixed_driver_response(r, parameters, dt, initial_state=initial,
            tau_n=tau_n, tau_v=tau_v, substeps=substeps, hb_operator=operator,
            derivative=derivative and ninitial > 0, numerical_backend=numerical_backend)
        component = np.einsum("tck,k->tc", common, coordinates[ninitial:])
        prediction = forward["hb_prediction"]+component
        data = (prediction[mask]-y[mask])/scale[mask]
        residual = np.r_[data, prior_design@(coordinates-prior_center)]
        jacobian = None
        if derivative:
            design = np.empty((*y.shape, count))
            if ninitial:
                design[:, :, :ninitial] = forward["initial_jacobian"]@chain
            if ncommon:
                design[:, :, ninitial:] = common
            jacobian = np.vstack((design[mask]/scale[mask, None], prior_design))
        return forward, component, prediction, residual, jacobian, float(residual@residual)

    try:
        current = evaluate(x, bool(mask.any() and count))
    except (FloatingPointError, OverflowError) as exc:
        return dict(success=False, converged=False, status="failed_domain", fit_performed=False,
            evaluations=evaluations, nfev=evaluations, iterations=0, driver=r.copy(),
            failure_log=[dict(evaluation=evaluations, stage="prior_initialization", error=str(exc))],
            physical_check=dict(valid=False), initial_mode=initial_mode,
            uncertainty="NOT_ESTIMATED")
    reason = "prior_prediction" if not mask.any() else "fixed_prediction" if count == 0 else "evaluation_budget"
    converged = bool(mask.any() and count == 0)
    damping, gradient = 1e-5, 0.
    while mask.any() and count:
        jacobian, residual, objective = current[4], current[3], current[5]
        derivative, gram = jacobian.T@residual, jacobian.T@jacobian
        diagonal = np.maximum(np.diag(gram), 1.)
        gradient = float(np.max(np.abs(derivative)/np.sqrt(diagonal)))
        if gradient <= gradient_tolerance:
            converged, reason = True, "scaled_gradient"
            break
        if evaluations+2 > max_nfev:
            break
        iterations += 1
        try:
            delta = cho_solve(cho_factor(gram+damping*np.diag(diagonal), lower=True), -derivative)
        except np.linalg.LinAlgError as exc:
            failure_log.append(dict(evaluation=evaluations, stage="solve", error=str(exc)))
            reason = "singular_step"
            break
        accepted = False
        for power in range(20):
            if evaluations+2 > max_nfev:
                break
            fraction = 2.**(-power)
            candidate = x+fraction*delta
            try:
                trial = evaluate(candidate, False)
                if trial[5] >= objective or trial[5] > objective+2e-4*fraction*float(derivative@delta):
                    continue
                differentiated = evaluate(candidate, True)
            except (FloatingPointError, OverflowError) as exc:
                failure_log.append(dict(evaluation=evaluations, stage="candidate", error=str(exc)))
                continue
            x, current, accepted = candidate, differentiated, True
            damping = max(damping/3, 1e-12)
            break
        if not accepted:
            damping *= 10
            if damping > 1e12 or np.linalg.norm(delta) <= 1e-12*(1+np.linalg.norm(x)):
                reason = "stagnation_without_gradient_convergence"
                break
    forward, component, prediction, residual, _, objective = current
    physical_valid = forward["physical_check"]["valid"]
    success = bool(physical_valid and (converged or not mask.any()))
    status = ("prior_prediction" if not mask.any() and physical_valid else "completed"
              if converged and physical_valid else "failed_physical" if not physical_valid else "failed_numerical")
    visible_residual = np.full_like(y, np.nan)
    visible_residual[mask] = prediction[mask]-y[mask]
    result = dict(forward)
    result.update(prediction=prediction, physical_prediction=forward["hb_prediction"],
        observation_component=component, residual=visible_residual,
        component_coefficients=x[ninitial:].copy(), fitted_initial_coordinates=x[:ninitial].copy(),
        initial_mode=initial_mode, initial_prior_mean=mean.copy(), initial_prior_sd=prior_sd.copy(),
        initial_weight=float(initial_weight), component_sd=coefficient_sd.copy(),
        success=success, converged=bool(converged and physical_valid), status=status,
        fit_performed=bool(mask.any() and count), visible_count=int(mask.sum()),
        evaluations=evaluations, nfev=evaluations, iterations=iterations, objective=objective,
        data_sse=float(visible_residual[mask]@visible_residual[mask]),
        weighted_data_sse=float(residual[:int(mask.sum())]@residual[:int(mask.sum())]),
        scaled_gradient_inf_norm=gradient, convergence_reason=reason, failure_log=failure_log,
        uncertainty="NOT_ESTIMATED", prior_interpretation="proper_coordinate_prior_not_physical_truth",
        driver_owner="fixed_EEG_input_never_reestimated_from_Hb",
        fitted_initial_coordinate_order=[] if initial_mode == "rest" else
            ["s0", "logf0", "logv0", "logq0"] if initial_mode == "tied_pv_logprior" else
            ["s0", "f0", "v0", "p0", "q0"])
    return result


def cascade_response_features(values, dt, *, time_constants=(.5, 1., 2., 4., 8.)):
    """Instantaneous input plus exact-ZOH two-stage causal lowpass cascades.

    Input [T,C] produces [T,C*(1+K)] in channel-major order. Each cascade has
    two equal time constants and both states start at zero. At sample t its
    output depends on inputs strictly before t; the instantaneous feature is
    values[t]. These states are computational features without source labels.
    """
    x, constants = np.asarray(values, float), np.asarray(time_constants, float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("values must be finite [T,C]")
    if (not np.isfinite(dt) or dt <= 0 or constants.ndim != 1 or not len(constants)
            or not np.isfinite(constants).all() or np.any(constants <= 0)):
        raise ValueError("positive finite dt and time constants required")
    with np.errstate(over="ignore", under="ignore"):
        ratio = dt/constants
        decay = np.exp(-ratio)
    coupling = np.zeros_like(ratio)
    retained = ratio < 745.
    coupling[retained] = decay[retained]*ratio[retained]
    complement = -np.expm1(-ratio)
    input_weight = complement-coupling
    small = ratio < 1e-4
    z = ratio[small]
    input_weight[small] = z*z*(.5-z/3+z*z/8-z**3/30+z**4/144)
    first, second = np.zeros((x.shape[1], len(constants))), np.zeros((x.shape[1], len(constants)))
    result = np.zeros((len(x), x.shape[1], 1+len(constants)))
    result[:, :, 0] = x
    for t in range(len(x)-1):
        value = x[t, :, None]
        second = decay*second+coupling*first+input_weight*value
        first = decay*first+complement*value
        result[t+1, :, 1:] = second
    return result.reshape(len(x), -1)


def gaussian_predictive_scores(target, prediction, predictive_sd, *, visible=None):
    """Gaussian NLL/CRPS using strictly positive caller-calibrated predictive SD.

    SD must be frozen from permitted training/selection residuals by the caller;
    this helper never estimates it from evaluation targets or a Hessian. Point
    arrays keep target shape and use NaN for unscored entries. Coordinate means
    average all leading axes; empty support has no mean (None / NaN).
    """
    y, mean = np.asarray(target, float), np.asarray(prediction, float)
    if y.ndim < 1 or mean.shape != y.shape:
        raise ValueError("target and prediction must have matching non-scalar shape")
    mask = _visible(y, visible)
    if not np.isfinite(mean[mask]).all():
        raise ValueError("scored predictions must be finite")
    scale = _positive_scale(predictive_sd, y.shape, "predictive_sd")
    z = (y[mask]-mean[mask])/scale[mask]
    nll, crps = np.full_like(y, np.nan), np.full_like(y, np.nan)
    nll[mask] = np.log(scale[mask])+.5*np.log(2*np.pi)+.5*z*z
    crps[mask] = scale[mask]*(z*(2*ndtr(z)-1)+np.sqrt(2/np.pi)*np.exp(-.5*z*z)-1/np.sqrt(np.pi))
    axes = tuple(range(y.ndim-1)) if y.ndim > 1 else (0,)
    counts = mask.sum(axis=axes)
    def coordinate_mean(points):
        total = np.where(mask, points, 0.).sum(axis=axes)
        return np.divide(total, counts, out=np.full_like(total, np.nan, dtype=float), where=counts > 0)
    return dict(nll=nll, crps=crps, mean_nll=float(nll[mask].mean()) if mask.any() else None,
        mean_crps=float(crps[mask].mean()) if mask.any() else None,
        nll_per_coordinate=coordinate_mean(nll), crps_per_coordinate=coordinate_mean(crps),
        scored_count=int(mask.sum()), counts_per_coordinate=counts,
        predictive_sd_source="caller_frozen_training_or_selection_not_Hessian")
