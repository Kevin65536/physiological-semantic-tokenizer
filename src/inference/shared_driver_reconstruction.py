"""Single-driver Balloon reconstruction diagnostics.

Only the driver trajectory and five initial vascular coordinates are fitted.
Vascular states have no per-sample innovations. Linear screening, fixed-driver
nonlinear replay and nonlinear MAP-style fitting remain distinct diagnostics;
none provides a calibrated SSM posterior or a physiological qualification.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import cho_factor, cho_solve, expm, svd

from . import t3a_balloon_robust_ssm as core


@dataclass(frozen=True)
class SharedDriverDesign:
    parameters: core.BalloonParameters
    steps: int
    dt: float
    observation_design: np.ndarray
    canonical_design: np.ndarray
    state_design: np.ndarray
    offset: np.ndarray
    canonical_offset: np.ndarray
    transition: np.ndarray
    input: np.ndarray


def build_shared_driver_design(parameters, steps, dt, *, processed_mean_operator=None):
    """Build time-major [EEG,HbO,HbR] design, columns [r[0:T],initial[5]].

    The driver is held constant on [t,t+dt); its last value is observed in
    EEG but cannot influence an earlier vascular state. A supplied processing
    operator acts on the complete canonical mean, before selecting visible
    processed observations. It does not implement raw-sensor masking.
    """
    parameters.validate()
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 3:
        raise ValueError('steps must be an integer of at least three')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    j = core.balloon_rhs_jacobian(np.zeros(6), parameters)
    augmented = np.zeros((6, 6))
    augmented[:5, :5], augmented[:5, 5] = j[1:, 1:], j[1:, 0]
    discrete = expm(augmented*dt)
    transition, drive = discrete[:5, :5], discrete[:5, 5]
    state = np.zeros((steps, 5, steps+5))
    state[0, :, steps:] = np.eye(5)
    for t in range(1, steps):
        state[t] = transition@state[t-1]
        state[t, :, t-1] += drive
    h = core.observation_jacobian(np.zeros(6), parameters)
    canonical = np.einsum('ij,tjk->tik', h[:, 1:], state)
    canonical[:, 0, :steps] += parameters.fixed.eeg_loading*np.eye(steps)
    canonical = canonical.reshape(3*steps, steps+5)
    rest = core.observation_map(np.array([0., 0., 1., 1., 1., 1.]), parameters)
    canonical_offset = np.broadcast_to(rest, (steps, 3)).copy()
    if processed_mean_operator is None:
        observed, offset = canonical.copy(), canonical_offset.copy()
    else:
        operator = np.asarray(processed_mean_operator, dtype=float)
        if operator.shape != (3*steps, 3*steps) or not np.isfinite(operator).all():
            raise ValueError('processed_mean_operator must be finite [3T,3T]')
        observed = operator@canonical
        offset = (operator@canonical_offset.ravel()).reshape(steps, 3)
    return SharedDriverDesign(parameters, int(steps), float(dt), observed,
        canonical, state, offset, canonical_offset, transition, drive)


def fit_shared_driver(observations, design, *, visible=None, penalty=1.,
                      initial_penalty=1., observation_scale=None,
                      small_signal_limit=.25, rcond=1e-10,
                      tie_total_hb_to_volume=False):
    """Fit visible rows by SVD, optionally penalizing curvature and initial state.

    ``penalty=0`` disables *all* regularization, including initial_penalty,
    and provides the numerical least-residual lower bound at the given rank
    tolerance. Weights are fixed observation standard deviations, not fitted
    noise. ``data_sse`` is always the unweighted visible-coordinate SSE.
    Curvature is dt*sum((second_difference(r)/dt**2)**2).
    The small-signal limit is a declared linearization screen, not a medical
    range. Physically invalid minima remain visible and explicitly flagged.
    With tie_total_hb_to_volume, solve in T+4 coordinates with delta_p0=delta_v0;
    both initial penalty rows remain. Returned coefficients retain T+5 entries.
    """
    if not isinstance(tie_total_hb_to_volume, (bool, np.bool_)):
        raise ValueError('tie_total_hb_to_volume must be boolean')
    y = np.asarray(observations, dtype=float)
    shape = (design.steps, 3)
    if y.shape != shape:
        raise ValueError('observations must have shape [T,3]')
    if visible is None:
        mask = np.isfinite(y)
        if np.isinf(y).any():
            raise ValueError('infinite observations')
    else:
        mask = np.asarray(visible)
        if mask.shape != shape or mask.dtype != np.bool_:
            raise ValueError('visible must be boolean [T,3]')
        if not np.isfinite(y[mask]).all():
            raise ValueError('visible observations must be finite')
    if not mask.any():
        raise ValueError('at least one visible observation is required')
    if (not np.isfinite([penalty, initial_penalty, small_signal_limit, rcond]).all()
            or penalty < 0 or initial_penalty < 0 or small_signal_limit <= 0
            or not 0 < rcond < 1):
        raise ValueError('invalid penalty, small-signal limit or SVD tolerance')
    scale = np.ones(shape) if observation_scale is None else np.asarray(observation_scale, dtype=float)
    if scale.shape == (3,):
        scale = np.broadcast_to(scale, shape)
    if scale.shape != shape or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('observation_scale must be positive finite [3] or [T,3]')
    x = design.observation_design[mask.ravel()]/scale[mask, None]
    target = (y[mask]-design.offset[mask])/scale[mask]
    curvature = np.zeros((design.steps-2, design.steps+5))
    curvature[:, :design.steps] = np.diff(np.eye(design.steps), n=2, axis=0)/design.dt**1.5
    if penalty:
        initial = np.zeros((5, design.steps+5))
        initial[:, design.steps:] = np.sqrt(initial_penalty)*np.eye(5)
        system = np.vstack((x, np.sqrt(penalty)*curvature, initial))
        rhs = np.r_[target, np.zeros(design.steps+3)]
    else:
        system, rhs = x, target
    embedding = _volume_tied_initial_embedding(design.steps) if tie_total_hb_to_volume else None
    if embedding is not None:
        system = system@embedding
        x = x@embedding
    u, singular, vt = svd(system, full_matrices=False, check_finite=False)
    keep = singular > (rcond*singular[0] if len(singular) else 0.)
    inverse = vt[keep].T/singular[keep]
    coefficients = inverse@(u[:, keep].T@rhs)
    effective_df = float(np.sum(np.square(x@inverse)))
    if embedding is not None:
        coefficients = embedding@coefficients
    canonical = (design.canonical_design@coefficients).reshape(shape)+design.canonical_offset
    prediction = (design.observation_design@coefficients).reshape(shape)+design.offset
    vascular = np.einsum('tij,j->ti', design.state_design, coefficients)
    states = np.column_stack((coefficients[:design.steps], vascular))
    states[:, 2:] += 1.
    mathematical = dict(core.run_physical_checks(states, design.parameters))
    required = ('finite', 'positive_fvpq', 'oxygen_extraction_in_unit_interval',
                'absolute_hb_nonnegative', 'hbr_not_above_hbt', 'rest_equilibrium')
    mathematical_valid = all(mathematical[key] for key in required)
    excursion = float(np.max(np.abs(vascular[:, 1:])))
    small_signal = excursion <= small_signal_limit
    residual = prediction[mask]-y[mask]
    return dict(prediction=prediction, canonical_prediction=canonical,
        driver=coefficients[:design.steps], states=states, vascular_states=vascular,
        initial_state=coefficients[design.steps:], coefficients=coefficients,
        data_sse=float(residual@residual), weighted_data_sse=float(np.sum((residual/scale[mask])**2)),
        roughness=float(np.sum((curvature@coefficients)**2)), effective_df=effective_df,
        rank=int(keep.sum()), singular_values=singular, visible_count=int(mask.sum()),
        penalty=float(penalty), initial_penalty=float(initial_penalty) if penalty else 0.,
        physical_validity=dict(mathematical=mathematical,
            mathematical_valid=mathematical_valid, small_signal_valid=small_signal,
            max_fractional_excursion=excursion, small_signal_limit=float(small_signal_limit),
            valid=bool(mathematical_valid and small_signal)),
        approximation='rest-linearized single-driver necessary-condition screen',
        residual_lower_bound=bool(penalty == 0), rcond=float(rcond),
        tie_total_hb_to_volume=bool(tie_total_hb_to_volume),
        initial_state_constraint='p0_equals_v0' if tie_total_hb_to_volume else 'none',
        initial_state_free_parameters=4 if tie_total_hb_to_volume else 5,
        nuisance_free_parameters_per_trial=design.steps+(4 if tie_total_hb_to_volume else 5),
        total_free_parameters=design.steps+(4 if tie_total_hb_to_volume else 5),
        residual_lower_bound_scope='tied_initial_state' if tie_total_hb_to_volume else 'free_initial_state')


def replay_nonlinear_driver(driver, initial_state, parameters, dt, *, substeps=4):
    """Replay fixed ZOH driver through nonlinear core, without refitting.

    ``initial_state`` is physical [s,f,v,p,q], unlike the linear design's
    initial perturbations. Classical RK4 evolves these five coordinates;
    derivatives come only from the existing transformed Balloon drift.
    Every RK stage and accepted endpoint undergoes the core physical checks.
    An invalid stage returns ``failed_domain`` and a NaN-filled uncomputed
    suffix; no clipping, projection or automatic step-size retry occurs.
    ``intermediate_checks`` retains all tested stage states, including failure.
    This deterministic diagnostic provides no posterior or fitted parameters.
    """
    parameters.validate()
    r = np.asarray(driver, dtype=float)
    initial = np.asarray(initial_state, dtype=float)
    if r.ndim != 1 or not len(r) or not np.isfinite(r).all():
        raise ValueError('driver must be nonempty finite [T]')
    if initial.shape != (5,) or not np.isfinite(initial).all():
        raise ValueError('initial_state must be finite physical [s,f,v,p,q]')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
        raise ValueError('substeps must be a positive integer')
    states = np.full((len(r), 6), np.nan)
    prediction = np.full((len(r), 3), np.nan)
    history = []
    required = ('finite', 'positive_fvpq', 'oxygen_extraction_in_unit_interval',
                'absolute_hb_nonnegative', 'hbr_not_above_hbt', 'rest_equilibrium')
    failure = None

    def check(vascular, value, step, substep, stage):
        nonlocal failure
        physical = np.r_[value, vascular]
        try:
            checks = dict(core.run_physical_checks(physical[None, :], parameters))
            valid = all(checks[key] for key in required)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            checks, valid = dict(error=repr(exc)), False
        record = dict(step=int(step), substep=int(substep), stage=stage,
                      state=physical.copy(), valid=bool(valid), checks=checks)
        history.append(record)
        if not valid:
            failure = dict(step=int(step), substep=int(substep), stage=stage,
                           reason='invalid_nonlinear_physical_domain', checks=checks)
        return valid

    def derivative(vascular, value):
        transformed = core.physical_to_transformed(np.r_[value, vascular])
        dz = core.balloon_rhs(transformed, parameters)[1:]
        return dz*np.r_[1., vascular[1:]]

    def result(status):
        finite = np.isfinite(states).all(axis=1)
        completed_checks = (dict(core.run_physical_checks(states[finite], parameters))
                            if finite.any() else None)
        return dict(status=status, states=states, canonical_prediction=prediction,
            checks=dict(all_intermediate_valid=all(h['valid'] for h in history),
                completed_samples=int(finite.sum()), requested_samples=len(r),
                completed_states=completed_checks, substeps=int(substeps), dt=float(dt),
                integrator='classical_RK4_physical_coordinates_fixed_ZOH_driver'),
            intermediate_checks=history, failure=failure)

    vascular = initial.copy()
    if not check(vascular, r[0], 0, -1, 'initial'):
        return result('failed_domain')
    states[0] = np.r_[r[0], vascular]
    prediction[0] = core.observation_map(states[0], parameters)
    h = float(dt)/substeps
    for step in range(len(r)-1):
        value = r[step]
        for substep in range(substeps):
            derivatives = []
            for stage in range(4):
                point = (vascular if stage == 0 else
                    vascular+(h if stage == 3 else h/2)*derivatives[-1])
                if not check(point, value, step, substep, f'k{stage+1}'):
                    return result('failed_domain')
                try:
                    slope = derivative(point, value)
                    if not np.isfinite(slope).all():
                        raise FloatingPointError('nonfinite physical derivative')
                except (ValueError, FloatingPointError, OverflowError) as exc:
                    history[-1]['valid'] = False
                    history[-1]['checks']['derivative_error'] = repr(exc)
                    failure = dict(step=step, substep=substep, stage=f'k{stage+1}',
                                   reason='invalid_nonlinear_derivative', error=repr(exc))
                    return result('failed_domain')
                derivatives.append(slope)
            k1, k2, k3, k4 = derivatives
            candidate = vascular+h*(k1+2*k2+2*k3+k4)/6
            if not check(candidate, value, step, substep, 'accepted_endpoint'):
                return result('failed_domain')
            vascular = candidate
        states[step+1] = np.r_[r[step+1], vascular]
        prediction[step+1] = core.observation_map(states[step+1], parameters)
    return result('completed')


def _require_nonlinear_domain(vascular, parameters):
    """Fast equivalent of core mathematical checks; no small-signal bound."""
    if not np.isfinite(vascular).all() or np.any(vascular[1:] <= 0):
        raise FloatingPointError('nonfinite or nonpositive nonlinear compartment')
    hbt, hbr = parameters.fixed.P0*vascular[3], parameters.fixed.Q0*vascular[4]
    if not np.isfinite([hbt, hbr]).all() or hbr > hbt:
        raise FloatingPointError('nonlinear absolute Hb balance violated')
    # Positive finite f and validated E0 imply 0<E<1 mathematically. The core
    # extraction implementation retains its stable log-complement convention.


def nonlinear_driver_forward(driver, initial_physical, parameters, dt, *,
                             substeps=4, derivative=True, return_flow_jacobian=False):
    """Nonlinear ZOH mean and exact derivatives of the discrete RK4 map.

    Jacobian columns are [r[0:T],s0,log(f0),log(v0),log(p0),log(q0)].
    ``tau_jacobian`` and ``neurovascular_gain_jacobian`` differentiate the same
    discretization with respect to tau and the effective gain, respectively.
    return_flow_jacobian optionally exposes the compact physical flow tangents.
    Initial physical values stay fixed. No parameter is fitted here. Invalid RK stages
    raise FloatingPointError without clipping or residual substitution.
    """
    parameters.validate()
    r = np.asarray(driver, dtype=float)
    initial = np.asarray(initial_physical, dtype=float)
    if r.ndim != 1 or not len(r) or not np.isfinite(r).all():
        raise ValueError('driver must be nonempty finite [T]')
    if initial.shape != (5,) or not np.isfinite(initial).all():
        raise ValueError('initial_physical must be finite [s,f,v,p,q]')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
        raise ValueError('substeps must be a positive integer')
    _require_nonlinear_domain(initial, parameters)
    steps, count = len(r), len(r)+5
    states = np.empty((steps, 6))
    state_derivative = np.zeros((steps, 5, count+2)) if derivative else None
    vascular = initial.copy()
    tangent = np.zeros((5, count+2)) if derivative else None
    if derivative:
        tangent[:, steps:count] = np.diag(np.r_[1., initial[1:]])
        state_derivative[0] = tangent
    states[0] = np.r_[r[0], vascular]
    h = float(dt)/substeps

    def slope(point, sensitivity, step):
        _require_nonlinear_domain(point, parameters)
        transformed = np.r_[r[step], point[0], np.log(point[1:])]
        dz = core._balloon_rhs_unchecked(transformed, parameters)[1:]
        diagonal = np.r_[1., point[1:]]
        value = diagonal*dz
        if not derivative:
            return value, None
        jt = core._balloon_rhs_jacobian_unchecked(transformed, parameters)[1:]
        physical_j = (diagonal[:, None]*jt[:, 1:])/diagonal[None, :]
        physical_j += np.diag(np.r_[0., dz[1:]])
        ds = physical_j@sensitivity
        ds[:, step] += diagonal*jt[:, 0]
        ds[2:, count] -= value[2:]/parameters.free.tau
        ds[0, count+1] += r[step]
        if not np.isfinite(value).all() or not np.isfinite(ds).all():
            raise FloatingPointError('nonfinite nonlinear tangent')
        return value, ds

    for step in range(steps-1):
        for _ in range(substeps):
            k1, d1 = slope(vascular, tangent, step)
            k2, d2 = slope(vascular+h*k1/2, tangent+h*d1/2 if derivative else None, step)
            k3, d3 = slope(vascular+h*k2/2, tangent+h*d2/2 if derivative else None, step)
            k4, d4 = slope(vascular+h*k3, tangent+h*d3 if derivative else None, step)
            vascular = vascular+h*(k1+2*k2+2*k3+k4)/6
            _require_nonlinear_domain(vascular, parameters)
            if derivative:
                tangent = tangent+h*(d1+2*d2+2*d3+d4)/6
        states[step+1] = np.r_[r[step+1], vascular]
        if derivative:
            state_derivative[step+1] = tangent
    spec = core.BalloonObservationSpec().resolved(parameters.fixed)
    canonical = np.array([core._observation_map_unchecked(s, parameters, spec) for s in states])
    jac, tau_jac, gain_jac = None, None, None
    if derivative:
        obs = core._observation_physical_matrix(parameters, spec)
        sensitivity = np.einsum('ij,tjk->tik', obs[:, 1:], state_derivative)
        sensitivity[:, 0, :steps] += parameters.fixed.eeg_loading*np.eye(steps)
        jac = sensitivity[:, :, :count].reshape(3*steps, count)
        tau_jac, gain_jac = sensitivity[:, :, count], sensitivity[:, :, count+1]
        if not np.isfinite(jac).all() or not np.isfinite(tau_jac).all() or not np.isfinite(gain_jac).all():
            raise FloatingPointError('nonfinite observation tangent')
    result = dict(states=states, canonical_prediction=canonical, jacobian=jac,
                tau_jacobian=tau_jac, neurovascular_gain_jacobian=gain_jac)
    if return_flow_jacobian:
        result.update(flow_jacobian=state_derivative[:, 1, :count].copy() if derivative else None,
            flow_tau_jacobian=state_derivative[:, 1, count].copy() if derivative else None,
            flow_neurovascular_gain_jacobian=state_derivative[:, 1, count+1].copy() if derivative else None)
    return result


def _volume_tied_initial_embedding(steps):
    """Map [r, s0, logf0, logv0, logq0] into the existing full coordinates."""
    embedding = np.eye(steps+5)[:, np.r_[np.arange(steps+3), steps+4]]
    embedding[steps+3, steps+2] = 1.
    return embedding


def _validate_initial_coordinates(mode, tied, parameters):
    if mode not in ('independent_logs', 'positive_hb'):
        raise ValueError('initial_coordinates must be independent_logs or positive_hb')
    if mode == 'positive_hb':
        if tied:
            raise ValueError('positive_hb cannot be combined with tie_total_hb_to_volume')
        if not 0 < parameters.fixed.Q0/parameters.fixed.P0 < 1:
            raise ValueError('positive_hb requires 0 < Q0/P0 < 1')


def _encode_positive_hb_initial(initial, parameters):
    ratio = parameters.fixed.Q0/parameters.fixed.P0
    h = initial[3]-ratio*initial[4]
    if not np.isfinite(h) or h <= 0:
        raise FloatingPointError('positive_hb requires strictly positive initial HbO')
    return np.r_[initial[0], np.log(initial[1:3]), np.log(h/(1-ratio)), np.log(initial[4])]


def _positive_hb_to_legacy(x, steps, parameters):
    """Exact coordinate map; no clipping or changes to physical penalties."""
    ratio = parameters.fixed.Q0/parameters.fixed.P0
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        h = (1-ratio)*np.exp(x[steps+3])
        q = np.exp(x[steps+4])
        p = ratio*q+h
    if not np.isfinite([h,q,p]).all() or min(h,q,p) <= 0 or p <= ratio*q:
        raise FloatingPointError('positive_hb initial coordinate underflow or invalid balance')
    legacy = x.copy();legacy[steps+3] = np.log(p)
    chain = np.eye(steps+5)
    chain[steps+3, steps+3] = h/p
    chain[steps+3, steps+4] = ratio*q/p
    return legacy, chain


def _nonlinear_trial_objective(x, parameters, dt, operator, y, mask, scale,
                               regularizer, initial_weight, substeps, derivative,
                               parameter_name='tau', flow_prior_weight=0., flow_prior_log_sd=1.,
                               initial_embedding=None, initial_coordinates='independent_logs'):
    """Actual common residual contract for individual and shared-parameter fitting."""
    steps, count = len(y), len(y)+5
    if initial_embedding is not None:
        x = initial_embedding@x
    coordinate_chain = None
    if initial_coordinates == 'positive_hb':
        x, coordinate_chain = _positive_hb_to_legacy(x, steps, parameters)
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        physical_initial = np.r_[x[steps], np.exp(x[steps+1:])]
    forward = nonlinear_driver_forward(x[:steps], physical_initial, parameters,
                                       dt, substeps=substeps, derivative=derivative,
                                       return_flow_jacobian=bool(flow_prior_weight))
    clean = forward['canonical_prediction'].ravel()
    prediction = (operator@clean if operator is not None else clean).reshape(y.shape)
    residual_data = (prediction[mask]-y[mask])/scale[mask]
    initial_residual = physical_initial-np.array([0., 1., 1., 1., 1.])
    residual = np.r_[residual_data, regularizer@x, np.sqrt(initial_weight)*initial_residual]
    flow_residual = np.empty(0)
    if flow_prior_weight:
        flow_factor = np.sqrt(flow_prior_weight*dt)/flow_prior_log_sd
        flow = forward['states'][:, 2]
        flow_residual = flow_factor*np.log(flow)
        residual = np.r_[residual, flow_residual]
    jacobian, parameter_jacobian = None, None
    if derivative:
        jf = forward['jacobian']
        jdata = ((operator@jf) if operator is not None else jf)[mask.ravel()]/scale[mask, None]
        ji = np.zeros((5, count))
        ji[:, steps:] = np.sqrt(initial_weight)*np.diag(np.r_[1., physical_initial[1:]])
        jacobian = np.vstack((jdata, regularizer, ji))
        jt = forward[parameter_name+'_jacobian'].ravel()
        jt = (operator@jt if operator is not None else jt)[mask.ravel()]/scale[mask]
        parameter_jacobian = np.r_[jt, np.zeros(len(residual)-len(jt))]
        if flow_prior_weight:
            jacobian = np.vstack((jacobian, (flow_factor/flow)[:, None]*forward['flow_jacobian']))
            parameter_jacobian[-steps:] = flow_factor/flow*forward['flow_'+parameter_name+'_jacobian']
    if derivative and initial_embedding is not None:
        jacobian = jacobian@initial_embedding
    stationarity_jacobian = jacobian
    if derivative and coordinate_chain is not None:
        jacobian = jacobian@coordinate_chain
    if not np.isfinite(residual).all() or (derivative and not np.isfinite(jacobian).all()):
        raise FloatingPointError('nonfinite nonlinear objective')
    return dict(residual=residual, jacobian=jacobian, parameter_jacobian=parameter_jacobian, forward=forward,
        stationarity_jacobian=stationarity_jacobian, coordinate_chain=coordinate_chain,
        prediction=prediction, initial=physical_initial, objective=float(residual@residual),
        flow_prior_cost=float(flow_residual@flow_residual))


def fit_nonlinear_shared_driver(observations, parameters, dt, *, mean_operator=None,
        sd=None, penalty=1., initial_penalty=1., starts=None, max_evaluations=80,
        visible=None, substeps=4, gradient_tolerance=1e-6, return_jacobian=False,
        driver_amplitude_weight=0., driver_prior_sd=1., flow_prior_weight=0., flow_prior_log_sd=1.,
        tie_total_hb_to_volume=False, initial_coordinates="independent_logs"):
    """Fixed-parameter nonlinear driver/initial-state fit by damped Gauss--Newton.

    ``starts`` contains dictionaries with driver[T] and physical initial_state[5].
    The default is zero driver and rest. Each start has max_evaluations forward
    calls, including derivatives and rejected trial steps. Convergence requires
    a scaled gradient test, never just a small step or substituted large cost.
    Initial regularization uses [s0,f0-1,v0-1,p0-1,q0-1] in physical coordinates.
    Driver and initial-state penalties are independent; both must be zero to
    disable those regularizers. Optional amplitude residuals are
    sqrt(driver_amplitude_weight*dt)*r/driver_prior_sd (scalar or [T] SD).
    Optional flow residuals are sqrt(flow_prior_weight*dt)*log(f)/flow_prior_log_sd
    at every output time, including the initial state. These engineering anchors
    are not physiological normal ranges or calibrated uncertainty.
    Weight zero preserves the old objective. The fixed processing mean operator acts before visible-row selection; hidden values
    cannot initialize or otherwise affect fitting. No posterior is estimated.
    tie_total_hb_to_volume removes logp0 from the optimization coordinates and
    sets logp0=logv0, retaining both physical initial-state penalty rows. Starts
    must have p0==v0 exactly. Dynamics and the mathematical domain stay unchanged.
    Returned Jacobians use the actual free coordinates.
    initial_coordinates='positive_hb' (untied only) represents initial HbO
    as (1-Q0/P0)*exp(eta_h), with p0=(Q0/P0)*q0+HbO and q0=exp(logq0).
    Physical penalties are unchanged. The success gradient is always evaluated
    in the original independent-log coordinates, preventing false convergence
    when the new HbO coordinate approaches zero. No positivity clipping occurs.
    """
    parameters.validate()
    _validate_initial_coordinates(initial_coordinates, tie_total_hb_to_volume, parameters)
    if not isinstance(tie_total_hb_to_volume, (bool, np.bool_)):
        raise ValueError('tie_total_hb_to_volume must be boolean')
    if not np.isfinite([flow_prior_weight, flow_prior_log_sd]).all() or flow_prior_weight < 0 or flow_prior_log_sd <= 0:
        raise ValueError('flow prior requires finite nonnegative weight and positive log SD')
    y = np.asarray(observations, dtype=float)
    if y.ndim != 2 or y.shape[1] != 3 or len(y) < 3:
        raise ValueError('observations must be [T>=3,3]')
    steps, count = len(y), len(y)+5
    embedding = _volume_tied_initial_embedding(steps) if tie_total_hb_to_volume else None
    constraint_metadata = dict(tie_total_hb_to_volume=bool(tie_total_hb_to_volume),
        initial_state_constraint='p0_equals_v0' if tie_total_hb_to_volume else 'none',
        initial_state_free_parameters=4 if tie_total_hb_to_volume else 5,
        nuisance_free_parameters_per_trial=steps+4 if tie_total_hb_to_volume else steps+5,
        total_free_parameters=steps+4 if tie_total_hb_to_volume else steps+5,
        initial_state_coordinate_order=['s0','logf0','logv0','logq0'] if tie_total_hb_to_volume else
            ['s0','logf0','logv0','logp0','logq0'])
    constraint_metadata['initial_coordinates'] = initial_coordinates
    constraint_metadata['stationarity_coordinate_basis'] = 'legacy_free_initial_coordinates'
    if initial_coordinates == 'positive_hb':
        constraint_metadata['initial_state_coordinate_order'] = ['s0','logf0','logv0','eta_h','logq0']
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
        raise ValueError('substeps must be a positive integer')
    if (isinstance(max_evaluations, bool) or not isinstance(max_evaluations, (int, np.integer))
            or max_evaluations < 1):
        raise ValueError('max_evaluations must be a positive integer')
    if (not np.isfinite([penalty, initial_penalty, gradient_tolerance, driver_amplitude_weight]).all()
            or penalty < 0 or initial_penalty < 0 or gradient_tolerance <= 0 or driver_amplitude_weight < 0):
        raise ValueError('invalid regularization or gradient tolerance')
    mask = np.isfinite(y) if visible is None else np.asarray(visible)
    if mask.shape != y.shape or mask.dtype != np.bool_ or not mask.any():
        raise ValueError('visible must be boolean [T,3] with at least one entry')
    if not np.isfinite(y[mask]).all() or (visible is None and np.isinf(y).any()):
        raise ValueError('visible observations must be finite')
    scale = np.ones_like(y) if sd is None else np.asarray(sd, dtype=float)
    if scale.shape == (3,):
        scale = np.broadcast_to(scale, y.shape)
    if scale.shape != y.shape or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('sd must be finite positive [3] or [T,3]')
    if mean_operator is not None:
        operator = np.asarray(mean_operator, dtype=float)
        if operator.shape != (3*steps, 3*steps) or not np.isfinite(operator).all():
            raise ValueError('mean_operator must be finite [3T,3T]')
    else:
        operator = None
    initial_weight = initial_penalty
    curvature = np.zeros((steps-2, count))
    curvature[:, :steps] = np.diff(np.eye(steps), n=2, axis=0)/dt**1.5
    regularizer = np.sqrt(penalty)*curvature
    prior_sd = np.asarray(driver_prior_sd,dtype=float)
    if prior_sd.ndim == 0:
        prior_sd = np.full(steps,float(prior_sd))
    if prior_sd.shape != (steps,) or not np.isfinite(prior_sd).all() or np.any(prior_sd <= 0):
        raise ValueError('driver_prior_sd must be finite positive scalar or [T]')
    if driver_amplitude_weight:
        amplitude = np.zeros((steps,count))
        amplitude[:,:steps] = np.diag(np.sqrt(driver_amplitude_weight*dt)/prior_sd)
        regularizer = np.vstack((regularizer,amplitude))
    starts = ([dict(driver=np.zeros(steps), initial_state=np.array([0., 1., 1., 1., 1.]))]
              if starts is None else list(starts))
    if not starts:
        raise ValueError('at least one start is required')

    def evaluate(x, derivative):
        return _nonlinear_trial_objective(x, parameters, dt, operator, y, mask, scale,
            regularizer, initial_weight, substeps, derivative,
            flow_prior_weight=flow_prior_weight, flow_prior_log_sd=flow_prior_log_sd,
            initial_embedding=embedding, initial_coordinates=initial_coordinates)

    attempts, candidates = [], []
    for index, start in enumerate(starts):
        r0, i0 = np.asarray(start['driver'], dtype=float), np.asarray(start['initial_state'], dtype=float)
        if r0.shape != (steps,) or i0.shape != (5,) or not np.isfinite(r0).all() or not np.isfinite(i0).all():
            raise ValueError('start requires finite driver[T] and physical initial_state[5]')
        if tie_total_hb_to_volume and i0[3] != i0[2]:
            raise ValueError('tie_total_hb_to_volume requires every start p0 == v0; no projection')
        evaluations, rejections, iterations = 0, 0, 0
        domain_rejections, nondecreasing_rejections, derivative_rejections = 0, 0, 0
        record = dict(start=index, converged=False, status='infeasible_initial', evaluations=0,
            rejected_steps=0, domain_rejections=0, nondecreasing_rejections=0,
            derivative_rejections=0, iterations=0)
        try:
            _require_nonlinear_domain(i0, parameters)
            x = np.r_[r0, i0[0], np.log(i0[1:])]
            if initial_coordinates == 'positive_hb':
                x = np.r_[r0, _encode_positive_hb_initial(i0, parameters)]
            if tie_total_hb_to_volume:x = np.delete(x, steps+3)
            evaluations += 1
            current = evaluate(x, True)
        except (FloatingPointError, OverflowError, ValueError) as exc:
            record.update(evaluations=evaluations, error=repr(exc))
            attempts.append(record)
            continue
        damping, reason, converged = 1e-5, 'evaluation_budget', False
        gradient_inf, scaled_gradient = np.inf, np.inf
        while True:
            jac, residual = current['jacobian'], current['residual']
            gradient = jac.T@residual
            gram = jac.T@jac
            diagonal = np.maximum(np.diag(gram), 1.)
            gradient_inf = float(2*np.max(np.abs(gradient)))
            scaled_gradient = float(np.max(np.abs(gradient)/np.sqrt(diagonal)))
            coordinate_scaled_gradient = scaled_gradient
            if initial_coordinates == 'positive_hb':
                legacy_j = current['stationarity_jacobian']
                legacy_gradient = legacy_j.T@residual
                gradient_inf = float(2*np.max(np.abs(legacy_gradient)))
                scaled_gradient = float(np.max(abs(legacy_gradient)/np.sqrt(np.maximum(np.sum(legacy_j*legacy_j,axis=0),1.))))
            if scaled_gradient <= gradient_tolerance:
                converged, reason = True, 'scaled_gradient'
                break
            # Reserve a derivative call for the accepted candidate: reported
            # objective and gradient must always describe the same final x.
            if evaluations+2 > max_evaluations:
                break
            iterations += 1
            try:
                delta = cho_solve(cho_factor(gram+damping*np.diag(diagonal), lower=True), -gradient)
            except np.linalg.LinAlgError:
                damping *= 10
                if damping > 1e16:
                    reason = 'singular_step'
                    break
                continue
            accepted = False
            for power in range(20):
                if evaluations+2 > max_evaluations:
                    break
                fraction = 2.**(-power)
                trial_x = x+fraction*delta
                evaluations += 1
                try:
                    trial = evaluate(trial_x, False)
                except (FloatingPointError, OverflowError, ValueError):
                    rejections += 1
                    domain_rejections += 1
                    continue
                armijo = current['objective']+2e-4*fraction*float(gradient@delta)
                if trial['objective'] <= armijo and trial['objective'] < current['objective']:
                    evaluations += 1
                    try:
                        differentiated = evaluate(trial_x, True)
                    except (FloatingPointError, OverflowError, ValueError):
                        rejections += 1
                        derivative_rejections += 1
                        continue
                    x, current = trial_x, differentiated
                    damping = max(damping/3, 1e-12)
                    accepted = True
                    break
                rejections += 1
                nondecreasing_rejections += 1
            if not accepted:
                if evaluations+2 > max_evaluations:
                    reason = 'evaluation_budget'
                    break
                damping *= 10
                if damping > 1e12 or np.linalg.norm(delta) <= 1e-12*(1+np.linalg.norm(x)):
                    reason = 'stagnation_without_gradient_convergence'
                    break
        record.update(status='completed' if converged else 'failed_numerical',
            converged=converged, convergence_reason=reason, objective=current['objective'],
            gradient_inf_norm=gradient_inf, scaled_gradient_inf_norm=scaled_gradient,
            evaluations=evaluations, rejected_steps=rejections, iterations=iterations,
            domain_rejections=domain_rejections, nondecreasing_rejections=nondecreasing_rejections,
            derivative_rejections=derivative_rejections)
        if initial_coordinates == 'positive_hb':
            record['optimization_coordinate_scaled_gradient_inf_norm'] = coordinate_scaled_gradient
        attempts.append(record)
        candidates.append((record, x, current))
    if not candidates:
        return dict(status='failed_domain', starts=attempts, converged=False,
            evaluations=sum(a['evaluations'] for a in attempts),
            rejected_steps=sum(a['rejected_steps'] for a in attempts),
            domain_rejections=sum(a['domain_rejections'] for a in attempts),
            nondecreasing_rejections=sum(a['nondecreasing_rejections'] for a in attempts),
            derivative_rejections=sum(a['derivative_rejections'] for a in attempts), **constraint_metadata)
    converged_candidates = [candidate for candidate in candidates if candidate[0]['converged']]
    record, x, best = min(converged_candidates or candidates, key=lambda candidate: candidate[0]['objective'])
    replay = replay_nonlinear_driver(x[:steps], best['initial'], parameters, dt, substeps=substeps)
    replay_error = (float(np.max(abs(replay['canonical_prediction']-best['forward']['canonical_prediction'])))
                    if replay['status'] == 'completed' else float('inf'))
    validated = replay['status'] == 'completed' and replay_error <= 1e-10
    states = best['forward']['states']
    residual_data = best['prediction'][mask]-y[mask]
    full_x = embedding@x if tie_total_hb_to_volume else x
    result = dict(status=record['status'] if validated else 'failed_replay_validation',
        converged=bool(record['converged'] and validated), starts=attempts,
        selected_start=record['start'], objective=best['objective'],
        gradient_inf_norm=record['gradient_inf_norm'], scaled_gradient_inf_norm=record['scaled_gradient_inf_norm'],
        convergence_reason=record['convergence_reason'],
        evaluations=sum(a['evaluations'] for a in attempts), rejected_steps=sum(a['rejected_steps'] for a in attempts),
        domain_rejections=sum(a['domain_rejections'] for a in attempts),
        nondecreasing_rejections=sum(a['nondecreasing_rejections'] for a in attempts),
        derivative_rejections=sum(a['derivative_rejections'] for a in attempts),
        prediction=best['prediction'], canonical_prediction=best['forward']['canonical_prediction'],
        driver=x[:steps], initial_state=best['initial'], states=states,
        state_ranges=dict(minimum=states.min(axis=0), maximum=states.max(axis=0)),
        physical_checks=replay['checks'], replay_max_abs_difference=replay_error,
        replay_failure=replay['failure'], data_sse=float(residual_data@residual_data),
        weighted_data_sse=float(np.sum((residual_data/scale[mask])**2)),
        roughness=float(np.sum((curvature@full_x)**2)), penalty=float(penalty), initial_penalty=float(initial_weight),
        driver_amplitude_weight=float(driver_amplitude_weight),driver_prior_sd=prior_sd,
        driver_amplitude_cost=float(driver_amplitude_weight*dt*np.sum((x[:steps]/prior_sd)**2)),
        flow_prior_weight=float(flow_prior_weight), flow_prior_log_sd=float(flow_prior_log_sd),
        flow_prior_cost=best['flow_prior_cost'],
        flow_prior_interpretation='engineering_log_flow_anchor_not_physiological_range_or_calibrated_uncertainty',
        visible_count=int(mask.sum()), parameter_inference='fixed_parameters_no_posterior',
        uncertainty='NOT_ESTIMATED')
    result.update(constraint_metadata)
    if initial_coordinates == 'positive_hb':
        result['optimization_coordinate_scaled_gradient_inf_norm'] = record['optimization_coordinate_scaled_gradient_inf_norm']
    if return_jacobian:
        result.update(canonical_jacobian=(best['forward']['jacobian']@embedding if tie_total_hb_to_volume
            else best['forward']['jacobian']@best['coordinate_chain'] if initial_coordinates == 'positive_hb'
            else best['forward']['jacobian']), residual_jacobian=best['jacobian'],
            stationarity_jacobian=best['stationarity_jacobian'])
    return result


def fit_nonlinear_shared_parameter(observations, parameters, dt, *, parameter_name,
        parameter_bounds, mean_operator=None, sd=None, penalty=1., initial_penalty=1.,
        parameter_prior_mean=None, parameter_prior_log_sd=None, starts=None, max_evaluations=240,
        max_iterations=40, visible=None, substeps=4, gradient_tolerance=1e-6,
        information_rcond=1e-10, return_jacobian=False, record_trace=False,
        driver_amplitude_weight=0., driver_prior_sd=1., flow_prior_weight=0., flow_prior_log_sd=1.,
        tie_total_hb_to_volume=False, initial_coordinates="independent_logs"):
    """Fit exactly one declared positive shared parameter using block GN.

    Arrays are [B,T,3], with one common mean operator. sd accepts [3], [B,3]
    or [B,T,3]. parameter_name is 'tau' or 'neurovascular_gain', with explicit
    positive parameter_bounds. Each start supplies parameter_value, driver[B,T],
    initial_state[B,5] in physical coordinates; the default is the input
    parameter value and rest for every
    trial. The caller owns the train/validation split: only training arrays
    belong here. All other Balloon parameters and penalties stay fixed.
    Optional amplitude residuals use sqrt(driver_amplitude_weight*dt)*r/SD;
    driver_prior_sd accepts scalar, [B] or [B,T]. This is an engineering anchor.
    Optional flow residuals are sqrt(flow_prior_weight*dt)*log(f)/flow_prior_log_sd
    per trial at all output times. This soft engineering anchor is not a
    physiological normal range or calibrated uncertainty; zero weight adds no rows.
    record_trace retains loop diagnostics per start and on the selected result;
    it does not change the objective, evaluation budget, or stopping rule.

    tie_total_hb_to_volume removes each trial's independent logp0 coordinate,
    with logp0=logv0 and both physical penalty rows retained. Starts with p0!=v0
    are rejected, not projected. Gradients, returned Jacobians and local nuisance
    information use the reduced coordinates; full physical states are returned.
    Untied initial_coordinates='positive_hb' changes optimization coordinates
    only, preserving all physical penalties and the original legacy-coordinate
    stationarity test. Local information uses that same legacy nuisance basis
    to avoid a coordinate-induced rank loss near zero initial HbO.

    A single optional log(value/parameter_prior_mean)/parameter_prior_log_sd residual is
    added per group, never per trial. Nuisance blocks are eliminated from each
    damped Gauss--Newton step, not assumed to be fully optimized profiles.
    A rejected proposal rejects the entire group; failed trials are never
    omitted. max_evaluations counts individual trial forward calls per start,
    including partial failed group evaluations. Final independent replay calls
    are reported separately and included in total evaluations.

    Local projected squared-residual information is returned with and without
    nuisance penalties, in the declared log-parameter coordinate. It is not a calibrated Fisher
    information, profile interval or physiological identifiability verdict.
    """
    parameters.validate()
    _validate_initial_coordinates(initial_coordinates, tie_total_hb_to_volume, parameters)
    if not isinstance(tie_total_hb_to_volume, (bool, np.bool_)):
        raise ValueError('tie_total_hb_to_volume must be boolean')
    if not np.isfinite([flow_prior_weight, flow_prior_log_sd]).all() or flow_prior_weight < 0 or flow_prior_log_sd <= 0:
        raise ValueError('flow prior requires finite nonnegative weight and positive log SD')
    if parameter_name not in ('tau','neurovascular_gain'):
        raise ValueError('parameter_name must be tau or neurovascular_gain')
    reference_value = (parameters.free.tau if parameter_name=='tau'
                       else parameters.fixed.neurovascular_gain)
    def parameter_object(value):
        if parameter_name == 'tau':
            return replace(parameters,free=replace(parameters.free,tau=float(value)))
        return replace(parameters,fixed=replace(parameters.fixed,neurovascular_gain=float(value)))
    y = np.asarray(observations, dtype=float)
    if y.ndim != 3 or y.shape[0] < 1 or y.shape[1] < 3 or y.shape[2] != 3:
        raise ValueError('observations must be [B>=1,T>=3,3]')
    batch, steps, _ = y.shape
    count = steps+5
    embedding = _volume_tied_initial_embedding(steps) if tie_total_hb_to_volume else None
    constraint_metadata = dict(tie_total_hb_to_volume=bool(tie_total_hb_to_volume),
        initial_state_constraint='p0_equals_v0' if tie_total_hb_to_volume else 'none',
        initial_state_free_parameters=4 if tie_total_hb_to_volume else 5,
        nuisance_free_parameters_per_trial=steps+4 if tie_total_hb_to_volume else steps+5,
        total_free_parameters=batch*(steps+4 if tie_total_hb_to_volume else steps+5)+1,
        initial_state_coordinate_order=['s0','logf0','logv0','logq0'] if tie_total_hb_to_volume else
            ['s0','logf0','logv0','logp0','logq0'])
    constraint_metadata['initial_coordinates'] = initial_coordinates
    constraint_metadata['stationarity_coordinate_basis'] = 'legacy_free_initial_coordinates'
    if initial_coordinates == 'positive_hb':
        constraint_metadata['initial_state_coordinate_order'] = ['s0','logf0','logv0','eta_h','logq0']
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be positive and finite')
    for value, name, minimum in ((substeps, 'substeps', 1),
            (max_evaluations, 'max_evaluations', batch), (max_iterations, 'max_iterations', 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if (not np.isfinite([penalty, initial_penalty, gradient_tolerance, information_rcond,driver_amplitude_weight]).all()
            or min(penalty, initial_penalty,driver_amplitude_weight) < 0 or gradient_tolerance <= 0 or not 0 < information_rcond < 1):
        raise ValueError('invalid penalty or numerical tolerance')
    bounds = np.asarray(parameter_bounds, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
        raise ValueError('parameter_bounds must be ordered finite positive values')
    lower, upper = np.log(bounds)
    if (parameter_prior_mean is None) != (parameter_prior_log_sd is None):
        raise ValueError('provide both parameter_prior_mean and parameter_prior_log_sd or neither')
    prior_precision = 0.
    if parameter_prior_mean is not None:
        if not np.isfinite([parameter_prior_mean, parameter_prior_log_sd]).all() or min(parameter_prior_mean, parameter_prior_log_sd) <= 0:
            raise ValueError('value prior mean and log SD must be finite and positive')
        prior_precision = 1/float(parameter_prior_log_sd)**2
    mask = np.isfinite(y) if visible is None else np.asarray(visible)
    if (mask.shape != y.shape or mask.dtype != np.bool_ or not np.all(mask.any(axis=(1,2)))
            or not np.isfinite(y[mask]).all() or (visible is None and np.isinf(y).any())):
        raise ValueError('each trial requires finite visible data and a boolean [B,T,3] mask')
    scale = np.ones_like(y) if sd is None else np.asarray(sd, dtype=float)
    if scale.shape == (3,):
        scale = np.broadcast_to(scale, y.shape)
    elif scale.shape == (batch, 3):
        scale = np.broadcast_to(scale[:, None, :], y.shape)
    if scale.shape != y.shape or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('sd must be positive finite [3], [B,3] or [B,T,3]')
    operator = None if mean_operator is None else np.asarray(mean_operator, dtype=float)
    if operator is not None and (operator.shape != (3*steps, 3*steps) or not np.isfinite(operator).all()):
        raise ValueError('mean_operator must be finite [3T,3T]')
    curvature = np.zeros((steps-2, count))
    curvature[:, :steps] = np.diff(np.eye(steps), n=2, axis=0)/dt**1.5
    regularizer = np.sqrt(penalty)*curvature
    prior_sd = np.asarray(driver_prior_sd,dtype=float)
    if prior_sd.ndim == 0:
        prior_sd = np.full((batch,steps),float(prior_sd))
    elif prior_sd.shape == (batch,):
        prior_sd = np.broadcast_to(prior_sd[:,None],(batch,steps))
    if prior_sd.shape != (batch,steps) or not np.isfinite(prior_sd).all() or np.any(prior_sd<=0):
        raise ValueError('driver_prior_sd must be positive finite scalar, [B] or [B,T]')
    regularizers = [regularizer]*batch
    if driver_amplitude_weight:
        regularizers=[]
        for trial in range(batch):
            amplitude=np.zeros((steps,count))
            amplitude[:,:steps]=np.diag(np.sqrt(driver_amplitude_weight*dt)/prior_sd[trial])
            regularizers.append(np.vstack((regularizer,amplitude)))
    starts = ([dict(parameter_value=reference_value, driver=np.zeros((batch, steps)),
                   initial_state=np.tile([0.,1.,1.,1.,1.], (batch,1)))] if starts is None else list(starts))
    if not starts:
        raise ValueError('at least one start is required')
    attempts, candidates = [], []
    for start_index, start in enumerate(starts):
        drivers, initial = np.asarray(start['driver'], dtype=float), np.asarray(start['initial_state'], dtype=float)
        value = float(start['parameter_value'])
        if (drivers.shape != (batch,steps) or initial.shape != (batch,5)
                or not np.isfinite(drivers).all() or not np.isfinite(initial).all() or not np.isfinite(value)):
            raise ValueError('start requires finite value, driver[B,T], initial_state[B,5]')
        if tie_total_hb_to_volume and np.any(initial[:,3] != initial[:,2]):
            raise ValueError('tie_total_hb_to_volume requires every start p0 == v0; no projection')
        evaluations = 0
        failed_trials, evaluation_failures = np.zeros(batch, dtype=int), []
        domain_rejections = nondecreasing_rejections = derivative_rejections = 0
        iterations, damping = 0, 1e-5
        record = dict(start=start_index, status='infeasible_initial', converged=False,
            evaluations=0, iterations=0, rejected_steps=0, expected_trials=batch,completed_trials=0,
            parameter_name=parameter_name,parameter_value=value)
        if record_trace:
            record['trace'] = []
        accepted_fraction = accepted_step_norm = None

        def append_trace():
            if not record_trace:
                return
            point = dict(iteration=iterations, evaluations=evaluations,
                objective=current['objective'],
                projected_scaled_gradient_inf_norm=float(scaled_gradient),
                parameter_value=float(np.exp(log_parameter)), damping=float(damping),
                domain_rejections=domain_rejections,
                nondecreasing_rejections=nondecreasing_rejections,
                derivative_rejections=derivative_rejections,
                accepted_fraction=accepted_fraction, accepted_step_norm=accepted_step_norm)
            if initial_coordinates == 'positive_hb':
                point['optimization_coordinate_scaled_gradient_inf_norm'] = float(coordinate_scaled_gradient)
            if not record['trace'] or point != record['trace'][-1]:
                record['trace'].append(point)

        def evaluate_group(vectors, log_parameter, derivative):
            nonlocal evaluations
            current_p = parameter_object(np.exp(log_parameter))
            trials = []
            for trial in range(batch):
                # All callers reserve a full group budget before entering.
                if evaluations >= max_evaluations:
                    raise RuntimeError('internal group evaluation budget exceeded')
                evaluations += 1
                try:
                    trials.append(_nonlinear_trial_objective(vectors[trial], current_p, dt, operator,
                        y[trial], mask[trial], scale[trial], regularizers[trial], initial_penalty,
                        substeps, derivative,parameter_name, flow_prior_weight, flow_prior_log_sd, embedding, initial_coordinates))
                except (FloatingPointError, OverflowError, ValueError) as exc:
                    failed_trials[trial] += 1
                    evaluation_failures.append(dict(evaluation=evaluations, trial=trial, derivative=derivative, error=repr(exc)))
                    raise
            prior_residual = ((log_parameter-np.log(parameter_prior_mean))/parameter_prior_log_sd
                              if parameter_prior_mean is not None else 0.)
            return dict(trials=trials, prior_residual=float(prior_residual),
                        objective=float(sum(t['objective'] for t in trials)+prior_residual**2))

        try:
            if not bounds[0] <= value <= bounds[1]:
                raise FloatingPointError('initial value outside declared bounds')
            for trial, state in enumerate(initial):
                try:
                    _require_nonlinear_domain(state, parameters)
                    if initial_coordinates == 'positive_hb':
                        _encode_positive_hb_initial(state, parameters)
                except FloatingPointError:
                    failed_trials[trial] += 1
                    evaluation_failures.append(dict(evaluation=0, trial=trial, derivative=False,
                                                   error='invalid physical initial state'))
                    raise
            x = np.column_stack((drivers, initial[:, 0], np.log(initial[:, 1:])))
            if initial_coordinates == 'positive_hb':
                x = np.column_stack((drivers, np.array([_encode_positive_hb_initial(v, parameters) for v in initial])))
            if tie_total_hb_to_volume:x = np.delete(x, steps+3, axis=1)
            log_parameter = float(np.log(value))
            current = evaluate_group(x, log_parameter, True)
        except (FloatingPointError, OverflowError, ValueError) as exc:
            record.update(error=repr(exc), evaluations=evaluations,
                failed_trial_evaluations=failed_trials.tolist(), evaluation_failures=evaluation_failures)
            attempts.append(record)
            continue
        converged, reason = False, 'evaluation_budget'
        while True:
            value = float(np.exp(log_parameter))
            blocks = []
            parameter_gradient = current['prior_residual']*np.sqrt(prior_precision)
            parameter_diagonal = prior_precision
            for trial in current['trials']:
                j, r = trial['jacobian'], trial['residual']
                jt = value*trial['parameter_jacobian']
                gram, gradient, cross = j.T@j, j.T@r, j.T@jt
                blocks.append((gram, gradient, cross))
                parameter_gradient += float(jt@r)
                parameter_diagonal += float(jt@jt)
            lower_active, upper_active = log_parameter <= lower+1e-12, log_parameter >= upper-1e-12
            projected_parameter_gradient = (0. if (lower_active and parameter_gradient >= 0)
                or (upper_active and parameter_gradient <= 0) else parameter_gradient)
            raw_gradient = max([abs(parameter_gradient)]+[np.max(abs(b[1])) for b in blocks])
            scaled_gradient = max([abs(projected_parameter_gradient)/np.sqrt(max(parameter_diagonal,1.))]+
                [np.max(abs(g)/np.sqrt(np.maximum(np.diag(h),1.))) for h,g,_ in blocks])
            coordinate_scaled_gradient = scaled_gradient
            if initial_coordinates == 'positive_hb':
                legacy_blocks = [(t['stationarity_jacobian'].T@t['stationarity_jacobian'],
                                  t['stationarity_jacobian'].T@t['residual']) for t in current['trials']]
                raw_gradient = max([abs(parameter_gradient)]+[np.max(abs(g)) for h,g in legacy_blocks])
                scaled_gradient = max([abs(projected_parameter_gradient)/np.sqrt(max(parameter_diagonal,1.))]+
                    [np.max(abs(g)/np.sqrt(np.maximum(np.diag(h),1.))) for h,g in legacy_blocks])
            append_trace()
            if scaled_gradient <= gradient_tolerance:
                converged, reason = True, 'projected_scaled_gradient'
                break
            if evaluations+2*batch > max_evaluations:
                reason = 'evaluation_budget'
                break
            if iterations >= max_iterations:
                reason = 'iteration_budget'
                break
            iterations += 1
            try:
                eliminated = [cho_solve(cho_factor(h+damping*np.diag(np.maximum(np.diag(h),1.)),lower=True),
                                        np.column_stack((g,c))) for h,g,c in blocks]
                schur = parameter_diagonal+damping*max(parameter_diagonal,1.)-sum(c@inv[:,1] for (_,_,c),inv in zip(blocks,eliminated))
                rhs = parameter_gradient-sum(c@inv[:,0] for (_,_,c),inv in zip(blocks,eliminated))
                if not np.isfinite(schur) or schur <= 0:
                    raise np.linalg.LinAlgError('nonpositive shared-parameter Schur complement')
                delta_parameter = -rhs/schur
                if (lower_active and delta_parameter < 0) or (upper_active and delta_parameter > 0):
                    delta_parameter = 0.
                delta_x = np.array([-inv[:,0]-inv[:,1]*delta_parameter for inv in eliminated])
            except np.linalg.LinAlgError:
                damping *= 10
                if damping > 1e16:
                    reason = 'singular_step'
                    break
                continue
            directional = parameter_gradient*delta_parameter+sum(g@d for (_,g,_),d in zip(blocks,delta_x))
            if not np.isfinite(directional) or directional >= 0:
                damping *= 10
                if damping > 1e12:
                    reason = 'non_descent_step'
                    break
                continue
            maximum_fraction = 1.
            if delta_parameter > 0:
                maximum_fraction = min(1., (upper-log_parameter)/delta_parameter)
            elif delta_parameter < 0:
                maximum_fraction = min(1., (lower-log_parameter)/delta_parameter)
            accepted = False
            for power in range(20):
                if evaluations+2*batch > max_evaluations:
                    break
                fraction = maximum_fraction*2.**(-power)
                trial_log_parameter = log_parameter+fraction*delta_parameter
                # Exact bound intersection can round a few ulps out of range.
                if abs(trial_log_parameter-lower) < 1e-14: trial_log_parameter = float(lower)
                if abs(trial_log_parameter-upper) < 1e-14: trial_log_parameter = float(upper)
                trial_x = x+fraction*delta_x
                try:
                    trial_group = evaluate_group(trial_x, trial_log_parameter, False)
                except (FloatingPointError, OverflowError, ValueError):
                    domain_rejections += 1
                    continue
                if (trial_group['objective'] < current['objective'] and
                        trial_group['objective'] <= current['objective']+2e-4*fraction*directional):
                    try:
                        differentiated = evaluate_group(trial_x, trial_log_parameter, True)
                    except (FloatingPointError, OverflowError, ValueError):
                        derivative_rejections += 1
                        continue
                    x, log_parameter, current = trial_x, trial_log_parameter, differentiated
                    if record_trace:
                        accepted_fraction = float(fraction)
                        accepted_step_norm = float(fraction*np.sqrt(np.sum(delta_x**2)+delta_parameter**2))
                    damping = max(damping/3,1e-12)
                    accepted = True
                    break
                nondecreasing_rejections += 1
            if not accepted:
                if evaluations+2*batch > max_evaluations:
                    reason = 'evaluation_budget'
                    break
                damping *= 10
                if damping > 1e12 or np.linalg.norm(delta_x)+abs(delta_parameter) < 1e-12:
                    reason = 'stagnation_without_gradient_convergence'
                    break
        append_trace()
        record.update(status='completed' if converged else 'failed_numerical',converged=converged,
            convergence_reason=reason,parameter_value=float(np.exp(log_parameter)),objective=current['objective'],
            gradient_inf_norm=float(2*raw_gradient),projected_scaled_gradient_inf_norm=float(scaled_gradient),
            log_parameter_gradient=float(2*parameter_gradient),projected_log_parameter_gradient=float(2*projected_parameter_gradient),
            evaluations=evaluations,iterations=iterations,
            rejected_steps=domain_rejections+nondecreasing_rejections+derivative_rejections,
            domain_rejections=domain_rejections,nondecreasing_rejections=nondecreasing_rejections,
            derivative_rejections=derivative_rejections,failed_trial_evaluations=failed_trials.tolist(),
            evaluation_failures=evaluation_failures,completed_trials=batch)
        attempts.append(record)
        if initial_coordinates == 'positive_hb':
            record['optimization_coordinate_scaled_gradient_inf_norm'] = float(coordinate_scaled_gradient)
        candidates.append((record,x,log_parameter,current))
    optimization_evaluations = sum(a['evaluations'] for a in attempts)
    if not candidates:
        return dict(status='failed_domain',converged=False,starts=attempts,expected_trials=batch,
            completed_trials=0,optimization_evaluations=optimization_evaluations,
            validation_evaluations=0,evaluations=optimization_evaluations,
            parameter_name=parameter_name,parameter_bounds=bounds,parameter_value=None,
            parameter_reference_value=reference_value, **constraint_metadata)
    successful = [c for c in candidates if c[0]['converged']]
    record,x,log_parameter,best = min(successful or candidates,key=lambda c:c[0]['objective'])
    value = float(np.exp(log_parameter))
    fitted_p = parameter_object(value)
    initial = np.array([t['initial'] for t in best['trials']])
    validations = [replay_nonlinear_driver(x[i,:steps],initial[i],fitted_p,dt,substeps=substeps) for i in range(batch)]
    validation_errors = [float(np.max(abs(v['canonical_prediction']-t['forward']['canonical_prediction'])))
        if v['status']=='completed' else float('inf') for v,t in zip(validations,best['trials'])]
    valid = all(v['status']=='completed' and e<=1e-10 for v,e in zip(validations,validation_errors))
    information = {}
    for label in ('data_only','nuisance_regularized'):
        values, ranks = [], []
        for i,trial in enumerate(best['trials']):
            n = int(mask[i].sum()) if label=='data_only' else len(trial['residual'])
            j = (trial['stationarity_jacobian'] if initial_coordinates == 'positive_hb' else trial['jacobian'])[:n]
            b = value*trial['parameter_jacobian'][:n]
            u,s,_ = svd(j,full_matrices=False,check_finite=False)
            keep = s > (information_rcond*s[0] if len(s) else 0.)
            remaining = b-u[:,keep]@(u[:,keep].T@b)
            values.append(float(remaining@remaining)); ranks.append(int(keep.sum()))
        information[label] = dict(value=float(sum(values)),by_trial=values,nuisance_ranks=ranks)
    information.update(coordinate='log_'+parameter_name,parameter_name=parameter_name,parameter_prior_precision=prior_precision,
        with_parameter_prior=float(information['nuisance_regularized']['value']+prior_precision),
        rank_rtol=float(information_rcond),interpretation='local_squared_residual_information_not_calibrated_CI')
    states = np.array([t['forward']['states'] for t in best['trials']])
    data_sse = np.array([float(np.sum((t['prediction'][mask[i]]-y[i][mask[i]])**2))
                         for i,t in enumerate(best['trials'])])
    weighted_sse = np.array([float(np.sum(((t['prediction'][mask[i]]-y[i][mask[i]])/scale[i][mask[i]])**2))
                             for i,t in enumerate(best['trials'])])
    full_x = x@embedding.T if tie_total_hb_to_volume else x
    roughness = np.sum((full_x@curvature.T)**2,axis=1)
    initial_cost = float(initial_penalty*np.sum((initial-np.array([0.,1.,1.,1.,1.]))**2))
    amplitude_cost = driver_amplitude_weight*dt*np.sum((x[:,:steps]/prior_sd)**2,axis=1)
    result = dict(status=record['status'] if valid else 'failed_replay_validation',
        converged=bool(record['converged'] and valid),starts=attempts,selected_start=record['start'],
        parameter_name=parameter_name,parameter_value=value,parameter_bounds=bounds,
        parameter_reference_value=reference_value,tau=fitted_p.free.tau,
        neurovascular_gain=fitted_p.fixed.neurovascular_gain,objective=best['objective'],
        trial_objectives=np.array([t['objective'] for t in best['trials']]),parameter_prior_cost=best['prior_residual']**2,
        data_sse=float(data_sse.sum()),weighted_data_sse=float(weighted_sse.sum()),
        trial_data_sse=data_sse,trial_weighted_data_sse=weighted_sse,
        roughness=float(roughness.sum()),trial_roughness=roughness,initial_state_penalty_cost=initial_cost,
        driver_amplitude_weight=float(driver_amplitude_weight),driver_prior_sd=prior_sd,
        driver_amplitude_cost=float(amplitude_cost.sum()),trial_driver_amplitude_cost=amplitude_cost,
        flow_prior_weight=float(flow_prior_weight), flow_prior_log_sd=float(flow_prior_log_sd),
        flow_prior_cost=float(sum(t['flow_prior_cost'] for t in best['trials'])),
        trial_flow_prior_cost=np.array([t['flow_prior_cost'] for t in best['trials']]),
        flow_prior_interpretation='engineering_log_flow_anchor_not_physiological_range_or_calibrated_uncertainty',
        expected_trials=batch,completed_trials=batch,visible_counts=mask.sum(axis=(1,2)),
        optimization_evaluations=optimization_evaluations,validation_evaluations=batch,
        evaluations=optimization_evaluations+batch,gradient_inf_norm=record['gradient_inf_norm'],
        projected_scaled_gradient_inf_norm=record['projected_scaled_gradient_inf_norm'],
        log_parameter_gradient=record['log_parameter_gradient'],projected_log_parameter_gradient=record['projected_log_parameter_gradient'],
        convergence_reason=record['convergence_reason'],
        boundary_status='LOWER' if abs(log_parameter-lower)<1e-8 else 'UPPER' if abs(log_parameter-upper)<1e-8 else 'INTERIOR',
        driver=x[:,:steps],initial_state=initial,states=states,
        prediction=np.array([t['prediction'] for t in best['trials']]),
        canonical_prediction=np.array([t['forward']['canonical_prediction'] for t in best['trials']]),
        physical_checks=[v['checks'] for v in validations],replay_failures=[v['failure'] for v in validations],
        replay_max_abs_difference=np.array(validation_errors),information=information,
        state_ranges=dict(minimum=states.min(axis=(0,1)),maximum=states.max(axis=(0,1))),
        penalty=float(penalty),initial_penalty=float(initial_penalty),uncertainty='NOT_ESTIMATED')
    result.update(constraint_metadata)
    if initial_coordinates == 'positive_hb':
        result['optimization_coordinate_scaled_gradient_inf_norm'] = record['optimization_coordinate_scaled_gradient_inf_norm']
    result['parameter_interpretation'] = ('conditional_compartment_transit_constant_seconds' if parameter_name=='tau'
        else 'effective_neurovascular_amplitude_in_fixed_observation_gauge_not_molecular_efficacy')
    if record_trace:
        result['trace'] = record['trace']
    if return_jacobian:
        result.update(residual_jacobians=[t['jacobian'] for t in best['trials']],
                      stationarity_jacobians=[t['stationarity_jacobian'] for t in best['trials']],
                      residual_log_parameter_jacobians=[value*t['parameter_jacobian'] for t in best['trials']])
    return result
