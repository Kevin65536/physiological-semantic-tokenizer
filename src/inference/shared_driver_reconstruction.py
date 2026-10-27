"""Tak-baseline shared-driver reconstruction and explicit historical controls.

Fits expose the driver trajectory, declared physiological parameters and
permitted initial coordinates. Vascular states have no per-sample innovations. Linear screening, fixed-driver
nonlinear replay and nonlinear MAP-style fitting remain distinct diagnostics;
none provides a calibrated SSM posterior or a physiological qualification.

The explicit delta-decay intervention additionally supports fixed-bandwidth
input forcing; it retains fixed physiology and has no added output component.
The default Tak core eliminates p in favor of v and restores venous
viscoelasticity. Its joint fitter supports four or fixed resting vascular
initial coordinates, with no independent vascular innovations.
New fits use fit_nonlinear_shared_parameters (batch size one is supported).
The older single-parameter/component fitters and linear/decay interventions
retain their recorded inlet-balance equations for historical comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import cho_factor, cho_solve, expm, svd

from . import t3a_balloon_robust_ssm as core


def _information_svd(matrix):
    """Retain the default decomposition; use QR SVD if divide-and-conquer fails.

    This is only a post-fit nuisance-space diagnostic. It changes neither the
    fitted objective nor any optimization step. The caller records the driver.
    """
    try:
        u,s,vh=svd(matrix,full_matrices=False,check_finite=False)
        return u,s,vh,'gesdd'
    except np.linalg.LinAlgError:
        u,s,vh=svd(matrix,full_matrices=False,check_finite=True,lapack_driver='gesvd')
        return u,s,vh,'gesvd_fallback'


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


def build_shared_driver_design(parameters, steps, dt, *, processed_mean_operator=None,
                               venous_viscoelastic_s=0.):
    """Historical inlet-balance design, columns [r[0:T],initial[5]].

    The driver is held constant on [t,t+dt); its last value is observed in
    EEG but cannot influence an earlier vascular state. A supplied processing
    operator acts on the complete canonical mean, before selecting visible
    processed observations. It does not implement raw-sensor masking.
    The optional linear diagnostic uses f_out=v**(1/alpha)+tau_v*dv/dt.
    It changes this legacy rest Jacobian only; it is not the reduced Tak
    baseline linearization. New Tak work uses the nonlinear forward/tangents.
    """
    parameters.validate()
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 3:
        raise ValueError('steps must be an integer of at least three')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if not np.isfinite(venous_viscoelastic_s) or venous_viscoelastic_s < 0:
        raise ValueError('venous_viscoelastic_s must be finite and nonnegative')
    j = core.balloon_rhs_jacobian(np.zeros(6), parameters)
    if venous_viscoelastic_s:
        tau, alpha = parameters.free.tau, parameters.fixed.alpha
        fraction = venous_viscoelastic_s/(tau+venous_viscoelastic_s)
        # Eliminate the implicit outflow before linearizing mass balances.
        j[3, 2] = 1/(tau+venous_viscoelastic_s)
        j[3, 3] = -1/(alpha*(tau+venous_viscoelastic_s))
        for row in (4, 5):
            j[row, 2] -= fraction/tau
            j[row, 3] += fraction/(alpha*tau)
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


def waveform_component_basis(parameters, steps, dt, mean_operator, mechanism, *, modes=6,
                             volume_deoxy_fraction=None):
    """Equal-dimension slow observation perturbations, not new physiology fits.

    Volume changes hold saturation fixed: [HbO,HbR]=[P0-Q0,Q0]*b(t).
    Exchange holds total Hb fixed: [-Q0,Q0]*b(t). Both use the same six
    predeclared cosine modes (periods 2*T*dt/k), before the mean operator.
    No direction or temporal basis is estimated from the target.
    A supplied volume_deoxy_fraction describes a separate compartment's
    fixed HbR/HbT loading; it does not change the Balloon's Q0 or E0.
    """
    if mechanism not in ('volume', 'exchange'):
        raise ValueError('unknown waveform mechanism')
    if not isinstance(modes, int) or not 1 <= modes < steps or dt <= 0:
        raise ValueError('invalid waveform basis clock or dimension')
    f = parameters.fixed
    q = f.Q0
    if volume_deoxy_fraction is not None:
        if mechanism != 'volume' or not np.isfinite(volume_deoxy_fraction) or not 0 < volume_deoxy_fraction < 1:
            raise ValueError('volume fraction must be in (0,1) for volume only')
        q = f.P0*volume_deoxy_fraction
    loading = np.array([0., f.P0-q, q] if mechanism == 'volume'
                       else [0., -f.Q0, f.Q0])
    basis = np.cos(np.pi*(np.arange(steps)[:, None]+.5)*np.arange(1, modes+1)/steps)
    canonical = (basis[:, None, :]*loading[None, :, None]).reshape(3*steps, modes)
    operator = np.asarray(mean_operator, float)
    if operator.shape != (3*steps, 3*steps) or not np.isfinite(operator).all():
        raise ValueError('mean operator must be finite [3T,3T]')
    return operator@canonical


def fit_waveform_subspace(target, design, sd, *, visible=None, extra_design=None, rcond=1e-10):
    """Batch unregularized SVD screen; never a physically qualified estimate.

    Targets [trial,T,3], training SD [3]. Nuisance trajectories are refit to
    each visible target. A parameter grid must be selected on training trials
    outside this function; validation reconstruction is not prediction.
    """
    y = np.asarray(target, float)
    scale = np.asarray(sd, float)
    n = design.steps
    if y.ndim != 3 or y.shape[1:] != (n, 3):
        raise ValueError('target must have shape [trial,T,3]')
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('sd must be positive finite [3]')
    mask = np.ones((n, 3), bool) if visible is None else np.asarray(visible)
    if mask.shape != (n, 3) or mask.dtype != np.bool_ or not mask.any():
        raise ValueError('visible must be nonempty boolean [T,3]')
    if not np.isfinite(y[:, mask]).all() or not 0 < rcond < 1:
        raise ValueError('invalid visible data or rcond')
    x = design.observation_design
    if extra_design is not None:
        extra = np.asarray(extra_design, float)
        if extra.ndim != 2 or extra.shape[0] != 3*n or not np.isfinite(extra).all():
            raise ValueError('extra design must be finite [3T,K]')
        x = np.column_stack((x, extra))
    w = np.tile(scale, n)[mask.ravel()]
    u, s, vt = svd(x[mask.ravel()]/w[:, None], full_matrices=False, check_finite=False)
    keep = s > rcond*s[0]
    rhs = ((y-design.offset)[:, mask]/w).T
    coef = (vt[keep].T/s[keep])@(u[:, keep].T@rhs)
    pred = (x@coef).T.reshape(y.shape)+design.offset
    vascular = np.einsum('tij,jb->bti', design.state_design, coef[:n+5])
    excursion = np.max(np.abs(vascular[:, :, 1:]), axis=(1, 2))
    return dict(prediction=pred, coefficients=coef.T, rank=int(keep.sum()),
                singular_values=s, max_fractional_excursion=excursion,
                nrmse=np.sqrt(np.mean(((pred-y)/scale)**2, axis=1)),
                visible_sse=np.sum(((pred-y)[:, mask]/w)**2, axis=1))


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


def replay_nonlinear_driver(driver, initial_state, parameters, dt, *, substeps=4,
                            hemodynamics='tak', viscoelastic_s=2.):
    """Replay fixed ZOH driver through nonlinear core, without refitting.

    The default is deterministic Tak with tau_v=2 seconds. A Tak initial state
    can be [s,f,v,q], or the retained [s,f,v,p,q] view with p==v exactly.
    p is derived, not an independent degree of freedom. Explicit inlet_balance
    with viscoelastic_s=0 retains the historical five-coordinate RK4 replay.
    Every RK stage and accepted endpoint undergoes the core physical checks.
    An invalid stage returns ``failed_domain`` and a NaN-filled uncomputed
    suffix; no clipping, projection or automatic step-size retry occurs.
    ``intermediate_checks`` retains all tested stage states, including failure.
    This deterministic diagnostic provides no posterior or fitted parameters.
    """
    parameters.validate()
    r = np.asarray(driver, dtype=float)
    initial = np.asarray(initial_state, dtype=float)
    if hemodynamics == 'tak' and initial.shape == (4,):
        initial = initial[[0, 1, 2, 2, 3]]
    if r.ndim != 1 or not len(r) or not np.isfinite(r).all():
        raise ValueError('driver must be nonempty finite [T]')
    if initial.shape != (5,) or not np.isfinite(initial).all():
        raise ValueError('initial_state must be finite physical [s,f,v,p,q]')
    _validate_hemodynamics(hemodynamics, viscoelastic_s)
    if hemodynamics == 'tak' and initial[2] != initial[3]:
        raise ValueError('Tak reduced core requires p0 == v0 exactly')
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
        if hemodynamics == 'tak':
            # Independent scalar transcription of the SPM equations. It does
            # not call the compiled forward or its state/parameter tangents.
            s, f, v, p, q = vascular
            fixed, free = parameters.fixed, parameters.free
            dv = (f-v**(1/fixed.alpha))/(free.tau+viscoelastic_s)
            out = v**(1/fixed.alpha)+viscoelastic_s*dv
            ef = -np.expm1(np.log1p(-fixed.E0)/f)/fixed.E0
            return np.array([fixed.neurovascular_gain*value-free.kappa*s-fixed.gamma*(f-1),
                s, dv, dv, (f*ef-out*q/v)/free.tau])
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


def _validate_hemodynamics(hemodynamics, viscoelastic_s):
    if hemodynamics not in ('inlet_balance', 'tak'):
        raise ValueError('hemodynamics must be inlet_balance or tak')
    if not np.isfinite(viscoelastic_s) or viscoelastic_s < 0:
        raise ValueError('viscoelastic_s must be finite and nonnegative')
    if hemodynamics == 'inlet_balance' and viscoelastic_s != 0:
        raise ValueError('inlet_balance has no viscoelastic term')


def nonlinear_driver_forward(driver, initial_physical, parameters, dt, *,
                             substeps=4, derivative=True, return_flow_jacobian=False,
                             numerical_backend='numba', hemodynamics='tak',
                             viscoelastic_s=2.):
    """Nonlinear ZOH mean and exact derivatives of the discrete RK4 map.

    Jacobian columns are [r[0:T],s0,log(f0),log(v0),log(p0),log(q0)].
    ``tau_jacobian`` and ``neurovascular_gain_jacobian`` differentiate the same
    discretization with respect to tau and the effective gain, respectively.
    return_flow_jacobian optionally exposes the compact physical flow tangents.
    Initial physical values stay fixed. No parameter is fitted here. Invalid RK stages
    raise FloatingPointError without clipping or residual substitution.
    Tak is the research baseline: four physical initial states [s,f,v,q],
    deterministic p=v, and tau_v=2 seconds by default. The retained five-entry
    initial/state view is also accepted, but its p entry must equal v exactly.
    Historical callers explicitly select inlet_balance and viscoelastic_s=0.
    The returned Jacobian keeps its recorded column layout: under Tak the
    log(p0) column is zero and log(v0) differentiates both v and derived p.
    The current joint fitter removes that redundant column exactly.
    """
    parameters.validate()
    r = np.asarray(driver, dtype=float)
    initial = np.asarray(initial_physical, dtype=float)
    if hemodynamics == 'tak' and initial.shape == (4,):
        initial = initial[[0, 1, 2, 2, 3]]
    if r.ndim != 1 or not len(r) or not np.isfinite(r).all():
        raise ValueError('driver must be nonempty finite [T]')
    if initial.shape != (5,) or not np.isfinite(initial).all():
        raise ValueError('initial_physical must be finite [s,f,v,p,q]')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
        raise ValueError('substeps must be a positive integer')
    _require_nonlinear_domain(initial, parameters)
    _validate_hemodynamics(hemodynamics, viscoelastic_s)
    if hemodynamics == 'tak':
        if numerical_backend != 'numba':
            raise ValueError('Tak forward requires numba; independent replay is available separately')
        from .shared_driver_rk4 import tak_forward
        return tak_forward(r, initial, parameters, dt, substeps, derivative,
                           return_flow_jacobian, viscoelastic_s)
    if numerical_backend == 'numba':
        from .shared_driver_rk4 import compiled_forward
        return compiled_forward(r, initial, parameters, dt, substeps, derivative,
                                return_flow_jacobian)
    if numerical_backend != 'python':
        raise ValueError('numerical_backend must be python or numba')
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


def _vascular_flow_rk4(log_flow, basis, initial, tau, alpha, e0, p0, q0,
                       dt, substeps, derivative):
    """Three physical states and exact tangents of this discrete RK4 map."""
    steps, modes = basis.shape
    count = modes+3
    width = count if derivative else 0
    states = np.empty((steps, 3))
    tangents = np.zeros((steps, 3, width))
    point = initial.copy()
    tangent = np.zeros((3, width))
    if derivative:
        for j in range(3):
            tangent[j, modes+j] = initial[j]
    states[0] = point
    tangents[0] = tangent
    h = dt/substeps
    exponent = 1/alpha-1
    log_complement = np.log1p(-e0)

    def slope(state, sensitivity, step, fraction):
        if (not np.isfinite(state).all() or np.any(state <= 0)
                or p0*state[1] < q0*state[2]):
            raise FloatingPointError('invalid vascular RK4 state or absolute Hb balance')
        value = (1-fraction)*log_flow[step]+fraction*log_flow[step+1]
        flow = np.exp(value)
        if not np.isfinite(flow) or flow <= 0:
            raise FloatingPointError('invalid vascular flow or extraction')
        a = log_complement/flow
        extraction = -np.expm1(a)
        if not np.isfinite(a) or a >= 0 or not 0 < extraction <= 1:
            raise FloatingPointError('invalid vascular flow or extraction')
        g = state[0]**exponent
        velocity = np.array([(flow-g*state[0])/tau,
                             (flow-g*state[1])/tau,
                             (flow*extraction/e0-g*state[2])/tau])
        ds = np.zeros((3, width))
        if derivative:
            dg = exponent*g/state[0]
            ds[0] = -(exponent+1)*g*sensitivity[0]/tau
            ds[1] = -(dg*state[1]*sensitivity[0]+g*sensitivity[1])/tau
            ds[2] = -(dg*state[2]*sensitivity[0]+g*sensitivity[2])/tau
            # Stable d(f*E(f)/E0)/d(log f), as in the core extraction owner.
            if abs(a) < 1e-3:
                polynomial = .5+a*(1/3+a*(1/8+a*(1/30+a/144)))
                dphi = (flow*extraction/e0)*a*(a/extraction)*polynomial
            else:
                dphi = flow*(extraction+a*np.exp(a))/e0
            for j in range(modes):
                weight = (1-fraction)*basis[step, j]+fraction*basis[step+1, j]
                ds[0, j] += flow*weight/tau
                ds[1, j] += flow*weight/tau
                ds[2, j] += dphi*weight/tau
        if not np.isfinite(velocity).all() or not np.isfinite(ds).all():
            raise FloatingPointError('nonfinite vascular slope or tangent')
        return velocity, ds

    for step in range(steps-1):
        for substep in range(substeps):
            left, middle, right = substep/substeps, (substep+.5)/substeps, (substep+1)/substeps
            k1, d1 = slope(point, tangent, step, left)
            k2, d2 = slope(point+h*k1/2, tangent+h*d1/2, step, middle)
            k3, d3 = slope(point+h*k2/2, tangent+h*d2/2, step, middle)
            k4, d4 = slope(point+h*k3, tangent+h*d3, step, right)
            point = point+h*(k1+2*k2+2*k3+k4)/6
            tangent = tangent+h*(d1+2*d2+2*d3+d4)/6
            if (not np.isfinite(point).all() or np.any(point <= 0)
                    or p0*point[1] < q0*point[2]):
                raise FloatingPointError('invalid vascular RK4 endpoint or absolute Hb balance')
        states[step+1] = point
        tangents[step+1] = tangent
    return states, tangents


def vascular_flow_forward(coefficients, initial_physical, parameters, dt, *,
                          temporal_basis=None, substeps=4, derivative=True,
                          numerical_backend='numba'):
    """Externally prescribed positive flow, independent initial [v,p,q].

    ``log f = temporal_basis @ coefficients``; omitted basis is the identity.
    Log flow is linearly interpolated at every RK4 stage. Parameters, including
    tau/alpha/E0 and P0/Q0, stay fixed. No EEG/upstream state, measured derivative,
    vascular innovation, clipping, or data access is involved. Jacobian columns
    are [flow coefficients,log(v0),log(p0),log(q0)].
    """
    parameters.validate()
    a, initial = np.asarray(coefficients, float), np.asarray(initial_physical, float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('coefficients must be nonempty finite [K]')
    basis = np.eye(len(a)) if temporal_basis is None else np.asarray(temporal_basis, float)
    if (basis.ndim != 2 or basis.shape[0] < 2 or basis.shape[1] != len(a)
            or not np.isfinite(basis).all()):
        raise ValueError('temporal_basis must be finite [T,K] with T >= 2')
    if initial.shape != (3,) or not np.isfinite(initial).all():
        raise ValueError('initial_physical must be finite [v,p,q]')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
        raise ValueError('substeps must be a positive integer')
    if not isinstance(derivative, (bool, np.bool_)):
        raise ValueError('derivative must be boolean')
    if numerical_backend not in ('python', 'numba'):
        raise ValueError('numerical_backend must be python or numba')
    kernel = _vascular_flow_rk4
    if numerical_backend == 'numba':
        if not hasattr(vascular_flow_forward, '_compiled_kernel'):
            from numba import njit
            vascular_flow_forward._compiled_kernel = njit(cache=True)(_vascular_flow_rk4)
        kernel = vascular_flow_forward._compiled_kernel
    log_flow = basis@a
    fixed = parameters.fixed
    states, tangent = kernel(log_flow, basis, initial, parameters.free.tau,
        fixed.alpha, fixed.E0, fixed.P0, fixed.Q0, float(dt), int(substeps), bool(derivative))
    full = np.column_stack((np.zeros((len(states), 2)), np.exp(log_flow), states))
    spec = core.BalloonObservationSpec().resolved(fixed)
    canonical = np.array([core._observation_map_unchecked(s, parameters, spec)[1:] for s in full])
    observation = core._observation_physical_matrix(parameters, spec)[1:, 3:]
    jacobian = (np.einsum('ij,tjk->tik', observation, tangent).reshape(2*len(states), -1)
                if derivative else None)
    return dict(states=states, flow=np.exp(log_flow), log_flow=log_flow,
                canonical_prediction=canonical, jacobian=jacobian,
                state_jacobian=tangent if derivative else None,
                physical_checks=core.run_physical_checks(full, parameters))


def fit_vascular_flow_relaxation(hb, parameters, dt, *, temporal_basis=None,
        hb_operator=None, sd=(1., 1.), score_mask=None, starts=None, max_nfev=300,
        substeps=4, validation_substeps=8, log_flow_bounds=(-np.log(4), np.log(4)),
        initial_log_bounds=(-np.log(2), np.log(2)), integration_tolerance=1e-4,
        optimizer_tolerance=1e-12, joint_hb_tolerance=.5):
    """Search a fixed-physiology Hb subsystem for legal forward trajectories.

    Minimize unregularized, training-SD-weighted Hb residuals after the complete
    [2T,2T] observation operator. Only score_mask entries of hb are read. The
    pointwise log-flow bounds constrain ``B@a``, including for a supplied DCT
    basis; the three initial log coordinates remain independent.

    The default deterministic starts use no target or truth information. The
    max_nfev budget applies separately to each start. Retain every coarse-SSE
    improvement and every candidate whose two coarse NRMSEs are at most
    joint_hb_tolerance+integration_tolerance; the latter cannot exclude a joint
    witness passing the fine/coarse integration check. Fine-validate all retained
    candidates, even at the optimization budget. Prefer a legal candidate meeting
    both chromophore tolerances, then minimize its fine weighted SSE. A legal
    witness certifies mathematical/integration checks only; its reported error
    must be compared with a caller-declared tolerance. Finite basis, engineering
    bounds and local optimization cannot certify incompatibility or a global
    lower bound when no good witness is found. No physiological parameters fit.
    """
    from scipy.optimize import Bounds, LinearConstraint, minimize

    parameters.validate()
    y, scale = np.asarray(hb, float), np.asarray(sd, float)
    if y.ndim != 2 or y.shape[0] < 3 or y.shape[1] != 2:
        raise ValueError('hb must have shape [T,2], T >= 3')
    n = len(y)
    basis = np.eye(n) if temporal_basis is None else np.asarray(temporal_basis, float)
    if (basis.ndim != 2 or basis.shape[0] != n or not 1 <= basis.shape[1] <= n
            or not np.isfinite(basis).all()):
        raise ValueError('temporal_basis must be finite [T,K], 1 <= K <= T')
    operator = np.eye(2*n) if hb_operator is None else np.asarray(hb_operator, float)
    if operator.shape != (2*n, 2*n) or not np.isfinite(operator).all():
        raise ValueError('hb_operator must be finite [2T,2T]')
    mask = np.ones(y.shape, bool) if score_mask is None else np.asarray(score_mask)
    if mask.shape != y.shape or mask.dtype != np.bool_ or not mask.any():
        raise ValueError('score_mask must be nonempty boolean [T,2]')
    if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('sd must be positive finite [2]')
    if not np.isfinite(y[mask]).all():
        raise ValueError('scored hb values must be finite')
    flow_bounds, initial_bounds = np.asarray(log_flow_bounds, float), np.asarray(initial_log_bounds, float)
    if (flow_bounds.shape != (2,) or initial_bounds.shape != (2,)
            or not np.isfinite(np.r_[flow_bounds, initial_bounds]).all()
            or flow_bounds[0] >= flow_bounds[1] or initial_bounds[0] >= initial_bounds[1]):
        raise ValueError('log-flow and initial-log bounds must be finite increasing pairs')
    for name, value in (('max_nfev', max_nfev), ('substeps', substeps),
                        ('validation_substeps', validation_substeps)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(name+' must be a positive integer')
    if validation_substeps <= substeps:
        raise ValueError('validation_substeps must exceed substeps')
    if (not np.isfinite([integration_tolerance, optimizer_tolerance, joint_hb_tolerance]).all()
            or integration_tolerance <= 0 or optimizer_tolerance <= 0 or joint_hb_tolerance <= 0):
        raise ValueError('integration, optimizer and joint Hb tolerances must be positive finite')
    modes = basis.shape[1]
    if starts is None:
        neutral = np.full(3, initial_bounds.mean())
        delta = .05*np.diff(initial_bounds)[0]
        baseline = np.linalg.lstsq(basis, np.full(n, flow_bounds.mean()), rcond=None)[0]
        starts = [dict(coefficients=baseline, initial_state=np.exp(neutral+direction*delta*np.array([1., .5, .5])))
                  for direction in (0, -1, 1)]
    else:
        starts = list(starts)
    if not starts:
        raise ValueError('starts must be nonempty')
    vectors = []
    for start in starts:
        if not isinstance(start, dict) or 'coefficients' not in start or 'initial_state' not in start:
            raise ValueError('each start requires coefficients and initial_state')
        a, initial = np.asarray(start['coefficients'], float), np.asarray(start['initial_state'], float)
        if (a.shape != (modes,) or initial.shape != (3,) or not np.isfinite(a).all()
                or not np.isfinite(initial).all() or np.any(initial <= 0)):
            raise ValueError('start requires finite coefficients[K] and positive initial_state[3]')
        vectors.append(np.r_[a, np.log(initial)])
    constraint_matrix = np.column_stack((basis, np.zeros((n, 3))))
    constraints = [LinearConstraint(constraint_matrix, *flow_bounds)]
    bounds = Bounds(np.r_[np.full(modes, -np.inf), np.full(3, initial_bounds[0])],
                    np.r_[np.full(modes, np.inf), np.full(3, initial_bounds[1])])
    selected_operator = operator[mask.ravel()]/np.broadcast_to(scale, y.shape)[mask, None]
    observed = y[mask]/np.broadcast_to(scale, y.shape)[mask]
    attempts, candidates = [], []

    def chromophore_scores(prediction):
        scores = {}
        for j, name in enumerate(('HbO', 'HbR')):
            residual = prediction[mask[:, j], j]-y[mask[:, j], j]
            rmse = float(np.sqrt(np.mean(residual**2))) if len(residual) else None
            scores[name] = dict(n_scored=len(residual), rmse=rmse,
                nrmse=float(rmse/scale[j]) if rmse is not None else None,
                weighted_sse=float(np.sum((residual/scale[j])**2)))
        return scores

    class EvaluationBudget(Exception):
        pass

    for index, x0 in enumerate(vectors):
        record = dict(start=index, evaluations=0, domain_rejections=0,
                      optimizer_converged=False, status='infeasible_initial', message='',
                      retained_candidates=0, legal_witness=False, joint_hb_witness=False)
        best = None
        cache_x, cache_value = None, None

        def in_bounds(x):
            flow = basis@x[:modes]
            return bool(np.all(flow >= flow_bounds[0]-1e-10) and np.all(flow <= flow_bounds[1]+1e-10)
                and np.all(x[modes:] >= initial_bounds[0]-1e-10)
                and np.all(x[modes:] <= initial_bounds[1]+1e-10))

        def objective(x):
            nonlocal best, cache_x, cache_value
            if cache_x is not None and np.array_equal(x, cache_x):
                return cache_value
            if record['evaluations'] >= max_nfev:
                raise EvaluationBudget
            record['evaluations'] += 1
            try:
                forward = vascular_flow_forward(x[:modes], np.exp(x[modes:]), parameters, dt,
                    temporal_basis=basis, substeps=substeps)
                residual = selected_operator@forward['canonical_prediction'].ravel()-observed
                jacobian = selected_operator@forward['jacobian']
                sse = float(residual@residual)
                value = sse/(2*len(residual)), jacobian.T@residual/len(residual)
                if in_bounds(x):
                    improvement = best is None or sse < best['sse']
                    coarse_prediction = (operator@forward['canonical_prediction'].ravel()).reshape(n, 2)
                    coarse_scores = chromophore_scores(coarse_prediction)
                    possible_joint_witness = all(item['nrmse'] is not None and
                        item['nrmse'] <= joint_hb_tolerance+integration_tolerance
                        for item in coarse_scores.values())
                    if improvement or possible_joint_witness:
                        # Retained candidates need trajectories, not large derivative arrays.
                        retained_forward = {key: item for key, item in forward.items()
                                            if key not in ('jacobian', 'state_jacobian')}
                        candidate = dict(x=x.copy(), sse=sse, forward=retained_forward,
                            gradient_inf_norm=float(np.max(np.abs(value[1]))), start=index,
                            evaluation=record['evaluations'])
                        candidates.append(candidate)
                        record['retained_candidates'] += 1
                        if improvement:
                            best = candidate
            except (FloatingPointError, OverflowError):
                record['domain_rejections'] += 1
                value = (np.inf, np.zeros_like(x))
            cache_x, cache_value = x.copy(), value
            return value

        if in_bounds(x0):
            try:
                initial_value = objective(x0)[0]
                if np.isfinite(initial_value):
                    result = minimize(objective, x0, method='SLSQP', jac=True, bounds=bounds,
                        constraints=constraints, options=dict(maxiter=max_nfev, ftol=optimizer_tolerance))
                    record.update(optimizer_converged=bool(result.success),
                        status='converged' if result.success else 'optimizer_failed',
                        message=str(result.message), optimizer_status=int(result.status),
                        iterations=int(result.nit))
            except EvaluationBudget:
                record.update(status='evaluation_budget', message='forward evaluation budget exhausted')
        if best is not None:
            record['weighted_data_sse'] = best['sse']
        attempts.append(record)
    common = dict(starts=attempts, evaluations=sum(a['evaluations'] for a in attempts),
        parameters=parameters, temporal_basis=basis, log_flow_bounds=flow_bounds,
        initial_log_bounds=initial_bounds, score_mask=mask, sd=scale,
        joint_hb_tolerance=joint_hb_tolerance, retained_candidates=len(candidates),
        validation_evaluations=len(candidates),
        search_scope='finite_basis_bounded_flow_subsystem_local_search',
        incompatibility_certified=False, global_lower_bound_certified=False)
    if not candidates:
        return dict(common, status='failed_domain', legal_witness=False,
                    optimizer_converged=False, converged=False, prediction=None,
                    joint_hb_witness=False)
    required = ('finite', 'positive_fvpq', 'oxygen_extraction_in_unit_interval',
                'absolute_hb_nonnegative', 'hbr_not_above_hbt')
    for candidate in candidates:
        x, coarse = candidate['x'], candidate['forward']
        prediction_coarse = (operator@coarse['canonical_prediction'].ravel()).reshape(n, 2)
        fine, integration_error, validation_message = None, np.inf, ''
        try:
            fine = vascular_flow_forward(x[:modes], np.exp(x[modes:]), parameters, dt,
                temporal_basis=basis, substeps=validation_substeps, derivative=False)
            prediction_fine = (operator@fine['canonical_prediction'].ravel()).reshape(n, 2)
            integration_error = float(np.max(np.abs((prediction_fine-prediction_coarse)[mask]
                                                   /np.broadcast_to(scale, y.shape)[mask])))
        except (FloatingPointError, OverflowError) as error:
            validation_message = str(error)
        forward = fine if fine is not None else coarse
        prediction = (operator@forward['canonical_prediction'].ravel()).reshape(n, 2)
        physical = all(forward['physical_checks'][key] for key in required)
        integration_passed = bool(fine is not None and integration_error <= integration_tolerance)
        legal = bool(physical and integration_passed)
        scores = chromophore_scores(prediction)
        joint = bool(legal and all(item['nrmse'] is not None and item['nrmse'] <= joint_hb_tolerance
                                  for item in scores.values()))
        residual = (prediction[mask]-y[mask])/np.broadcast_to(scale, y.shape)[mask]
        candidate.update(validated_forward=forward, prediction=prediction,
            prediction_coarse=prediction_coarse, validation_sse=float(residual@residual),
            legal=legal, joint=joint, by_chromophore=scores,
            integration=dict(passed=integration_passed, substeps=substeps,
                validation_substeps=validation_substeps, tolerance=integration_tolerance,
                max_abs_training_sd_difference=integration_error, message=validation_message))
        record = attempts[candidate['start']]
        record.update(legal_witness=bool(record['legal_witness'] or legal),
            joint_hb_witness=bool(record['joint_hb_witness'] or joint),
            validation_weighted_data_sse=min(record.get('validation_weighted_data_sse', np.inf),
                                             candidate['validation_sse']))
    valid = [candidate for candidate in candidates if candidate['legal']]
    joint = [candidate for candidate in valid if candidate['joint']]
    best = min(joint or valid or candidates, key=lambda item: item['validation_sse'])
    x, forward, prediction = best['x'], best['validated_forward'], best['prediction']
    per_chromophore = best['by_chromophore']
    converged = attempts[best['start']]['optimizer_converged']
    return dict(common, status='completed' if best['legal'] else 'failed_integration_validation',
        legal_witness=best['legal'], optimizer_converged=converged, converged=converged,
        joint_hb_witness=bool(joint), selected_evaluation=best['evaluation'],
        selection='joint_hb_witness_then_weighted_sse',
        minimum_legal_weighted_data_sse=min((item['validation_sse'] for item in valid), default=None),
        selected_start=best['start'], coefficients=x[:modes].copy(),
        initial_state=np.exp(x[modes:]), states=forward['states'], flow=forward['flow'],
        log_flow=forward['log_flow'], prediction=prediction,
        canonical_prediction=forward['canonical_prediction'],
        optimization_prediction=best['prediction_coarse'], optimization_weighted_data_sse=best['sse'],
        objective_gradient_inf_norm=best['gradient_inf_norm'],
        weighted_data_sse=sum(item['weighted_sse'] for item in per_chromophore.values()),
        by_chromophore=per_chromophore, physical_checks=forward['physical_checks'],
        integration=best['integration'],
        optimizer_status=attempts[best['start']]['status'])


def _decay_trial_objective(x, parameters, dt, operator, y, mask, scale,
                           regularizer, initial_weight, substeps, multiplier,
                           forcing_regularizer, flow_weight, flow_log_sd, derivative):
    """Visible-only objective with input forcing and unchanged baseline priors."""
    from .shared_driver_rk4 import decay_forward

    n, count = len(y), len(x)
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        initial = np.r_[x[n], np.exp(x[n+1:n+5])]
    forward = decay_forward(x[:n], initial, parameters, dt,
        decay_multiplier=multiplier, forcing_coefficients=x[n+5:],
        substeps=substeps, derivative=derivative)
    clean = forward['canonical_prediction'].ravel()
    prediction = (operator@clean if operator is not None else clean).reshape(y.shape)
    data = (prediction[mask]-y[mask])/scale[mask]
    curvature = regularizer@x
    initial_residual = np.sqrt(initial_weight)*(initial-np.array([0., 1., 1., 1., 1.]))
    residual = np.r_[data, curvature, initial_residual]
    flow_residual, forcing_residual = np.empty(0), np.empty(0)
    if flow_weight:
        flow_factor = np.sqrt(flow_weight*dt)/flow_log_sd
        flow = forward['states'][:, 2]
        flow_residual = flow_factor*np.log(flow)
        residual = np.r_[residual, flow_residual]
    if forcing_regularizer.shape[1]:
        forcing_residual = forcing_regularizer@x[n+5:]
        residual = np.r_[residual, forcing_residual]
    jacobian = None
    if derivative:
        jf = forward['jacobian']
        jdata = ((operator@jf) if operator is not None else jf)[mask.ravel()]/scale[mask, None]
        ji = np.zeros((5, count))
        ji[:, n:n+5] = np.sqrt(initial_weight)*np.diag(np.r_[1., initial[1:]])
        jacobian = np.vstack((jdata, regularizer, ji))
        if flow_weight:
            jacobian = np.vstack((jacobian, (flow_factor/flow)[:, None]*forward['flow_jacobian']))
        if forcing_regularizer.shape[1]:
            ju = np.zeros((len(forcing_residual), count))
            ju[:, n+5:] = forcing_regularizer
            jacobian = np.vstack((jacobian, ju))
    if not np.isfinite(residual).all() or (derivative and not np.isfinite(jacobian).all()):
        raise FloatingPointError('nonfinite decay objective')
    costs = dict(weighted_data_sse=float(data@data), driver_curvature_cost=float(curvature@curvature),
        initial_prior_cost=float(initial_residual@initial_residual),
        flow_prior_cost=float(flow_residual@flow_residual),
        forcing_prior_cost=float(forcing_residual@forcing_residual))
    return dict(residual=residual, jacobian=jacobian, forward=forward, prediction=prediction,
                initial=initial, costs=costs, objective=float(residual@residual))


def fit_decay_shared_driver(target, parameters, dt, *, mean_operator=None, sd=None,
        visible=None, decay_multiplier=1., forcing_modes=0, forcing_sd=.01,
        forcing_weight=1., penalty=.01, initial_penalty=100., flow_prior_weight=1.,
        flow_prior_log_sd=np.log(2), max_evaluations=600, gradient_tolerance=1e-5,
        substeps=4, starts=None, return_jacobian=False):
    """Fixed-physiology fit of a driver, five initial states and optional input.

    The intervention is p_dot=v_dot-lambda*v**(1/alpha-1)*(p-v)/tau+u(t).
    The finite input has M=forcing_modes, duration L=T*dt, frequencies
    pi*k/L (k=0..M-1), and a fixed dt/2 phase. Its first coefficient is a
    constant input; M=7/19 means DC plus 6/18 cosines for 30/90 s. It is
    integrated continuously at RK stages. No output component or physiological
    parameter is fitted. The same basis applies at every decay multiplier.
    Input regularization is forcing_weight*dt*sum(u(t)**2)/forcing_sd**2 at
    output times, including the initial time; coefficients have state/s units.
    Driver curvature, physical initial-state and log-flow priors use precisely
    the existing nonlinear objective conventions and stay the same for lambda.

    Supplied starts require driver[T], physical initial_state[5], and, when
    M>0, forcing_coefficients[M]. Only visible target values enter the fit.
    Each start has a separate hard forward-evaluation budget. The same scaled
    gradient criterion and damped Gauss--Newton steps as the baseline are used;
    an exhausted budget or a small step cannot claim convergence. The caller
    owns independent integration-resolution audits (for example 4/8 substeps).
    """
    parameters.validate()
    y = np.asarray(target, float)
    if y.ndim != 2 or y.shape[1] != 3 or len(y) < 3:
        raise ValueError('target must be [T>=3,3]')
    n = len(y)
    if (isinstance(forcing_modes, (bool, np.bool_))
            or not isinstance(forcing_modes, (int, np.integer)) or not 0 <= forcing_modes < n):
        raise ValueError('forcing_modes must be an integer in [0,T)')
    count = n+5+forcing_modes
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    for name, value in (('substeps', substeps), ('max_evaluations', max_evaluations)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(name+' must be a positive integer')
    if (not np.isfinite([decay_multiplier, forcing_sd, forcing_weight, penalty,
            initial_penalty, flow_prior_weight, flow_prior_log_sd, gradient_tolerance]).all()
            or min(decay_multiplier, forcing_weight, penalty, initial_penalty, flow_prior_weight) < 0
            or min(forcing_sd, flow_prior_log_sd, gradient_tolerance) <= 0):
        raise ValueError('invalid decay multiplier, regularization, scales or gradient tolerance')
    mask = np.isfinite(y) if visible is None else np.asarray(visible)
    if mask.shape != y.shape or mask.dtype != np.bool_ or not mask.any():
        raise ValueError('visible must be nonempty boolean [T,3]')
    if not np.isfinite(y[mask]).all() or (visible is None and np.isinf(y).any()):
        raise ValueError('visible target values must be finite')
    scale = np.ones_like(y) if sd is None else np.asarray(sd, float)
    if scale.shape == (3,):
        scale = np.broadcast_to(scale, y.shape)
    if scale.shape != y.shape or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('sd must be finite positive [3] or [T,3]')
    operator = None if mean_operator is None else np.asarray(mean_operator, float)
    if operator is not None and (operator.shape != (3*n, 3*n) or not np.isfinite(operator).all()):
        raise ValueError('mean_operator must be finite [3T,3T]')
    curvature = np.zeros((n-2, count))
    curvature[:, :n] = np.diff(np.eye(n), n=2, axis=0)/dt**1.5
    regularizer = np.sqrt(penalty)*curvature
    basis = np.cos(np.pi*(np.arange(n)[:, None]+.5)*np.arange(forcing_modes)/n)
    forcing_regularizer = np.sqrt(forcing_weight*dt)/forcing_sd*basis
    starts = ([dict(driver=np.zeros(n), initial_state=np.array([0., 1., 1., 1., 1.]),
                    forcing_coefficients=np.zeros(forcing_modes))]
              if starts is None else list(starts))
    if not starts:
        raise ValueError('at least one start is required')

    def evaluate(x, derivative):
        return _decay_trial_objective(x, parameters, dt, operator, y, mask, scale,
            regularizer, initial_penalty, substeps, decay_multiplier, forcing_regularizer,
            flow_prior_weight, flow_prior_log_sd, derivative)

    attempts, candidates = [], []
    for index, start in enumerate(starts):
        if (not isinstance(start, dict) or 'driver' not in start or 'initial_state' not in start
                or (forcing_modes and 'forcing_coefficients' not in start)):
            raise ValueError('each start requires driver, initial_state and enabled forcing_coefficients')
        r0, i0 = np.asarray(start['driver'], float), np.asarray(start['initial_state'], float)
        c0 = np.asarray(start.get('forcing_coefficients', np.empty(0)), float)
        if (r0.shape != (n,) or i0.shape != (5,) or c0.shape != (forcing_modes,)
                or not np.isfinite(r0).all() or not np.isfinite(i0).all() or not np.isfinite(c0).all()):
            raise ValueError('start requires finite driver[T], initial_state[5], forcing_coefficients[M]')
        evaluations, rejections, iterations = 0, 0, 0
        domain_rejections, nondecreasing_rejections, derivative_rejections = 0, 0, 0
        record = dict(start=index, converged=False, status='infeasible_initial', evaluations=0,
            rejected_steps=0, domain_rejections=0, nondecreasing_rejections=0,
            derivative_rejections=0, iterations=0)
        try:
            _require_nonlinear_domain(i0, parameters)
            x = np.r_[r0, i0[0], np.log(i0[1:]), c0]
            evaluations += 1
            current = evaluate(x, True)
        except (FloatingPointError, OverflowError, ValueError) as exc:
            record.update(evaluations=evaluations, error=repr(exc))
            attempts.append(record)
            continue
        damping, reason, converged = 1e-5, 'evaluation_budget', False
        while True:
            jac, residual = current['jacobian'], current['residual']
            gradient = jac.T@residual
            gram = jac.T@jac
            diagonal = np.maximum(np.diag(gram), 1.)
            gradient_inf = float(2*np.max(np.abs(gradient)))
            scaled_gradient = float(np.max(np.abs(gradient)/np.sqrt(diagonal)))
            if scaled_gradient <= gradient_tolerance:
                converged, reason = True, 'scaled_gradient'
                break
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
        attempts.append(record)
        candidates.append((record, x, current))
    common = dict(starts=attempts, evaluations=sum(a['evaluations'] for a in attempts),
        rejected_steps=sum(a['rejected_steps'] for a in attempts),
        domain_rejections=sum(a['domain_rejections'] for a in attempts),
        nondecreasing_rejections=sum(a['nondecreasing_rejections'] for a in attempts),
        derivative_rejections=sum(a['derivative_rejections'] for a in attempts),
        decay_multiplier=float(decay_multiplier), forcing_modes=int(forcing_modes),
        forcing_sd=float(forcing_sd), forcing_weight=float(forcing_weight),
        penalty=float(penalty), initial_penalty=float(initial_penalty),
        flow_prior_weight=float(flow_prior_weight), flow_prior_log_sd=float(flow_prior_log_sd),
        visible_count=int(mask.sum()), total_free_parameters=count, substeps=int(substeps),
        parameter_inference='fixed_parameters_no_posterior', uncertainty='NOT_ESTIMATED')
    if not candidates:
        return dict(common, status='failed_domain', converged=False)
    converged_candidates = [candidate for candidate in candidates if candidate[0]['converged']]
    record, x, best = min(converged_candidates or candidates, key=lambda candidate: candidate[0]['objective'])
    forward = best['forward']
    states = forward['states']
    difference = best['prediction'][mask]-y[mask]
    objective_gradient = 2*best['jacobian'].T@best['residual']
    result = dict(common, status=record['status'], converged=record['converged'],
        convergence_reason=record['convergence_reason'], selected_start=record['start'],
        objective=best['objective'], costs=best['costs'], objective_components=best['costs'],
        **best['costs'], gradient=objective_gradient, objective_gradient=objective_gradient,
        gradient_inf_norm=record['gradient_inf_norm'], scaled_gradient_inf_norm=record['scaled_gradient_inf_norm'],
        prediction=best['prediction'], physical_prediction=best['prediction'],
        canonical_prediction=forward['canonical_prediction'], driver=x[:n].copy(),
        initial_state=best['initial'], states=states, forcing_coefficients=x[n+5:].copy(),
        forcing=forward['forcing'], forcing_basis=forward['forcing_basis'],
        forcing_angular_frequencies=forward['forcing_angular_frequencies'],
        forcing_phase_s=forward['forcing_phase_s'], forcing_duration_s=forward['forcing_duration_s'],
        retention_integral=forward['retention_integral'], decay_retention=forward['decay_retention'],
        roughness=float(np.sum((curvature@x)**2)), data_sse=float(difference@difference),
        state_ranges=dict(minimum=states.min(axis=0), maximum=states.max(axis=0)),
        physical_checks=dict(all_intermediate_valid=True, completed_samples=n, requested_samples=n,
            completed_states=dict(core.run_physical_checks(states, parameters)), substeps=int(substeps),
            dt=float(dt), integrator='classical_RK4_physical_coordinates_ZOH_driver_continuous_cosine_input'))
    if return_jacobian:
        result.update(canonical_jacobian=forward['jacobian'], residual_jacobian=best['jacobian'],
                      residual=best['residual'])
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
                               initial_embedding=None, initial_coordinates='independent_logs',
                               numerical_backend='python', observation_component=None,
                               observation_whitener=None, hemodynamics='inlet_balance',
                               viscoelastic_s=0., initial_coordinate_weights=None,
                               data_weight=1.):
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
                                       return_flow_jacobian=bool(flow_prior_weight),
                                       numerical_backend=numerical_backend,
                                       hemodynamics=hemodynamics, viscoelastic_s=viscoelastic_s)
    clean = forward['canonical_prediction'].ravel()
    prediction = (operator@clean if operator is not None else clean).reshape(y.shape)
    residual_data = (prediction[mask]-y[mask])/scale[mask]
    if observation_whitener is not None:
        residual_data = observation_whitener@residual_data
    residual_data = np.sqrt(data_weight)*residual_data
    initial_residual = physical_initial-np.array([0., 1., 1., 1., 1.])
    initial_factor = np.sqrt(initial_weight) * (np.ones(5) if initial_coordinate_weights is None
        else np.sqrt(initial_coordinate_weights))
    residual = np.r_[residual_data, regularizer@x, initial_factor*initial_residual]
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
        if observation_whitener is not None:
            jdata = observation_whitener@jdata
        jdata = np.sqrt(data_weight)*jdata
        ji = np.zeros((5, count))
        ji[:, steps:] = np.diag(initial_factor*np.r_[1., physical_initial[1:]])
        jacobian = np.vstack((jdata, regularizer, ji))
        names = [parameter_name] if isinstance(parameter_name, str) else list(parameter_name)
        parameter_columns = []
        for name in names:
            jt = forward[name+'_jacobian'].ravel()
            jt = (operator@jt if operator is not None else jt)[mask.ravel()]/scale[mask]
            if observation_whitener is not None:
                jt = observation_whitener@jt
            jt = np.sqrt(data_weight)*jt
            column = np.r_[jt, np.zeros(len(residual)-len(jt))]
            if flow_prior_weight:
                column[-steps:] = flow_factor/flow*forward['flow_'+name+'_jacobian']
            parameter_columns.append(column)
        parameter_jacobian = (parameter_columns[0] if isinstance(parameter_name, str) else
            np.column_stack(parameter_columns) if parameter_columns else np.empty((len(residual), 0)))
        if flow_prior_weight:
            jacobian = np.vstack((jacobian, (flow_factor/flow)[:, None]*forward['flow_jacobian']))
    physical_prediction = prediction.copy()
    component = np.zeros_like(prediction)
    coefficients = np.empty(0)
    component_cost = 0.
    if observation_component is not None:
        # Exact variable projection of a fixed, low-rank observation component.
        # Its Gaussian coefficient penalty is part of the engineering objective;
        # it is not an extra physiological state or a calibrated likelihood.
        basis, weighted_basis, coefficient_map, coefficient_sd = observation_component
        data_count = int(mask.sum())
        coefficients = coefficient_map@residual[:data_count]
        component = (basis@coefficients).reshape(y.shape)
        prediction = physical_prediction+component
        residual[:data_count] += weighted_basis@coefficients
        component_residual = coefficients/coefficient_sd
        residual = np.r_[residual, component_residual]
        component_cost = float(component_residual@component_residual)
        if derivative:
            dc = coefficient_map@jacobian[:data_count]
            jacobian[:data_count] += weighted_basis@dc
            jacobian = np.vstack((jacobian, dc/coefficient_sd[:, None]))
            dp = coefficient_map@parameter_jacobian[:data_count]
            parameter_jacobian[:data_count] += weighted_basis@dp
            parameter_jacobian = np.concatenate((parameter_jacobian,
                dp/coefficient_sd if dp.ndim == 1 else dp/coefficient_sd[:, None]), axis=0)
    if derivative and initial_embedding is not None:
        jacobian = jacobian@initial_embedding
    stationarity_jacobian = jacobian
    if derivative and coordinate_chain is not None:
        jacobian = jacobian@coordinate_chain
    if not np.isfinite(residual).all() or (derivative and not np.isfinite(jacobian).all()):
        raise FloatingPointError('nonfinite nonlinear objective')
    return dict(residual=residual, jacobian=jacobian, parameter_jacobian=parameter_jacobian, forward=forward,
        stationarity_jacobian=stationarity_jacobian, coordinate_chain=coordinate_chain,
        prediction=prediction, physical_prediction=physical_prediction,
        observation_component=component, observation_coefficients=coefficients,
        observation_component_cost=component_cost,
        initial=physical_initial, objective=float(residual@residual),
        flow_prior_cost=float(flow_residual@flow_residual))


def fit_nonlinear_shared_driver(observations, parameters, dt, *, mean_operator=None,
        sd=None, penalty=1., initial_penalty=1., starts=None, max_evaluations=80,
        visible=None, substeps=4, gradient_tolerance=1e-6, return_jacobian=False,
        driver_amplitude_weight=0., driver_prior_sd=1., flow_prior_weight=0., flow_prior_log_sd=1.,
        tie_total_hb_to_volume=False, initial_coordinates="independent_logs",
        numerical_backend='python', observation_basis=None, observation_coefficient_sd=1.,
        curvature_breaks=()):
    """Historical inlet-balance fixed-parameter/component fit.

    New Tak fits use fit_nonlinear_shared_parameters with parameter_names=[]
    and batch size one when appropriate. This retained entry keeps the old
    component and initial-state contract for explicit historical callers.

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
    observation_basis is an optional fixed processed-coordinate [3T,K] basis
    with zero EEG rows. Coefficients are exactly profiled using visible data
    and a proper zero-mean coefficient penalty; their scale is frozen by the
    caller on training data. Full, physical and extra predictions are returned
    separately. The component has no source-identification interpretation.
    initial_coordinates='positive_hb' (untied only) represents initial HbO
    as (1-Q0/P0)*exp(eta_h), with p0=(Q0/P0)*q0+HbO and q0=exp(logq0).
    Physical penalties are unchanged. The success gradient is always evaluated
    in the original independent-log coordinates, preventing false convergence
    when the new HbO coordinate approaches zero. No positivity clipping occurs.
    curvature_breaks optionally removes second-difference rows that cross the
    declared segment boundaries. The driver remains pointwise and the vascular
    trajectory remains continuous; this adds no cross-segment driver prior.
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
    component_projection = None
    if observation_basis is not None:
        basis = np.asarray(observation_basis, dtype=float)
        if (basis.ndim != 2 or basis.shape[0] != 3*steps or not 1 <= basis.shape[1] < steps
                or not np.isfinite(basis).all() or np.any(basis[0::3] != 0)):
            raise ValueError('observation_basis must be finite [3T,K<T] with zero EEG rows')
        coefficient_sd = np.broadcast_to(np.asarray(observation_coefficient_sd, float), (basis.shape[1],)).copy()
        if not np.isfinite(coefficient_sd).all() or np.any(coefficient_sd <= 0):
            raise ValueError('observation coefficient SD must be finite and positive')
        weighted_basis = basis[mask.ravel()]/scale[mask, None]
        gram = weighted_basis.T@weighted_basis+np.diag(coefficient_sd**-2)
        coefficient_map = -cho_solve(cho_factor(gram, lower=True), weighted_basis.T)
        component_projection = (basis, weighted_basis, coefficient_map, coefficient_sd)
        constraint_metadata['profiled_observation_coefficients'] = basis.shape[1]
        constraint_metadata['optimized_driver_initial_coordinates'] = constraint_metadata['total_free_parameters']
        constraint_metadata['total_free_parameters'] += basis.shape[1]
        constraint_metadata['nuisance_free_parameters_per_trial'] += basis.shape[1]
    initial_weight = initial_penalty
    curvature = np.zeros((steps-2, count))
    curvature[:, :steps] = np.diff(np.eye(steps), n=2, axis=0)/dt**1.5
    breaks = list(curvature_breaks)
    if (any(isinstance(b, (bool, np.bool_)) or not isinstance(b, (int, np.integer)) for b in breaks)
            or breaks != sorted(set(breaks)) or any(b < 3 or b > steps-3 for b in breaks)
            or any(b-a < 3 for a, b in zip(breaks, breaks[1:]))):
        raise ValueError('curvature_breaks must be sorted unique boundaries with segments >=3')
    for boundary in breaks:
        curvature[boundary-2:boundary] = 0.
    if breaks:
        constraint_metadata['curvature_breaks'] = breaks
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
            initial_embedding=embedding, initial_coordinates=initial_coordinates,
            numerical_backend=numerical_backend, observation_component=component_projection)

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
    replay = replay_nonlinear_driver(x[:steps], best['initial'], parameters, dt, substeps=substeps, hemodynamics='inlet_balance', viscoelastic_s=0.)
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
        physical_prediction=best['physical_prediction'], observation_component=best['observation_component'],
        observation_coefficients=best['observation_coefficients'],
        observation_component_cost=best['observation_component_cost'],
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
        tie_total_hb_to_volume=False, initial_coordinates="independent_logs",
        step_control="armijo", numerical_backend='python'):
    """Historical inlet-balance scalar-parameter fit using block GN.

    New Tak fits, including one-parameter profiles, use
    fit_nonlinear_shared_parameters with an explicit parameter list.

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

    step_control='quadratic_interpolation' optionally tries one extra objective
    evaluation after a feasible Armijo candidate whose actual/predicted GN
    reduction ratio is below .25. The quadratic fraction is clipped to [.1,.5]
    times that candidate fraction; only a lower feasible Armijo objective wins.
    All residual rows (including flow and the single group parameter prior)
    enter the prediction. The final derivative group is budget-reserved; an
    unavailable or invalid extra candidate leaves the original candidate intact.
    Trace step_control_diagnostics describe the original candidate and selection;
    interpolation domain failures also enter ordinary domain/evaluation failures.
    interpolation_counts.selected counts objective selections, which can still
    fail the subsequent derivative evaluation; it is not accepted iterations.
    Default 'armijo' preserves the original arithmetic and result fields.

    Local projected squared-residual information is returned with and without
    nuisance penalties, in the declared log-parameter coordinate. It is not a calibrated Fisher
    information, profile interval or physiological identifiability verdict.
    """
    if step_control not in ("armijo", "quadratic_interpolation"):
        raise ValueError("step_control must be armijo or quadratic_interpolation")
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
    if step_control != 'armijo':
        constraint_metadata['step_control'] = step_control
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
        control_diagnostics = None
        interpolation_counts = dict(attempts=0, selected=0, domain_rejections=0,
                                    not_selected=0, budget_skips=0)
        if step_control != 'armijo':
            record.update(step_control=step_control, interpolation_counts=interpolation_counts)

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
            if step_control != 'armijo':
                point['step_control_diagnostics'] = control_diagnostics
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
                        substeps, derivative,parameter_name, flow_prior_weight, flow_prior_log_sd, embedding,
                        initial_coordinates, numerical_backend))
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
                    if step_control == 'quadratic_interpolation':
                        # Full arrow Jacobian: include all cross terms and the
                        # group parameter prior once, outside trial summation.
                        jd_squared = prior_precision*delta_parameter**2
                        for trial, dx in zip(current['trials'], delta_x):
                            jd = trial['jacobian']@dx + value*trial['parameter_jacobian']*delta_parameter
                            jd_squared += float(jd@jd)
                        predicted = -2*fraction*directional-fraction**2*jd_squared
                        actual = current['objective']-trial_group['objective']
                        ratio = actual/predicted if np.isfinite(predicted) and predicted > 0 else None
                        control_diagnostics = dict(candidate_fraction=float(fraction),
                            predicted_reduction=float(predicted), actual_reduction=float(actual),
                            gain_ratio=None if ratio is None else float(ratio),
                            directional_derivative=float(2*directional),
                            interpolation_fraction=None, interpolation_status='not_needed')
                        if ratio is not None and np.isfinite(ratio) and ratio < .25:
                            if evaluations+2*batch > max_evaluations:
                                interpolation_counts['budget_skips'] += 1
                                control_diagnostics['interpolation_status'] = 'budget_skip'
                            else:
                                denominator = trial_group['objective']-current['objective']-2*fraction*directional
                                interpolated_fraction = float(np.clip(
                                    -directional*fraction**2/denominator, .1*fraction, .5*fraction))
                                control_diagnostics['interpolation_fraction'] = interpolated_fraction
                                interpolation_counts['attempts'] += 1
                                interpolated_x = x+interpolated_fraction*delta_x
                                interpolated_log_parameter = log_parameter+interpolated_fraction*delta_parameter
                                try:
                                    interpolated = evaluate_group(interpolated_x, interpolated_log_parameter, False)
                                except (FloatingPointError, OverflowError, ValueError):
                                    domain_rejections += 1
                                    interpolation_counts['domain_rejections'] += 1
                                    control_diagnostics['interpolation_status'] = 'domain_rejection'
                                else:
                                    control_diagnostics['interpolated_objective'] = interpolated['objective']
                                    if (interpolated['objective'] < trial_group['objective'] and
                                            interpolated['objective'] <= current['objective']+2e-4*interpolated_fraction*directional):
                                        fraction, trial_x, trial_log_parameter, trial_group = (
                                            interpolated_fraction, interpolated_x, interpolated_log_parameter, interpolated)
                                        interpolation_counts['selected'] += 1
                                        control_diagnostics['interpolation_status'] = 'selected'
                                    else:
                                        interpolation_counts['not_selected'] += 1
                                        control_diagnostics['interpolation_status'] = 'not_selected'
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
    validations = [replay_nonlinear_driver(x[i,:steps],initial[i],fitted_p,dt,substeps=substeps, hemodynamics='inlet_balance', viscoelastic_s=0.) for i in range(batch)]
    validation_errors = [float(np.max(abs(v['canonical_prediction']-t['forward']['canonical_prediction'])))
        if v['status']=='completed' else float('inf') for v,t in zip(validations,best['trials'])]
    valid = all(v['status']=='completed' and e<=1e-10 for v,e in zip(validations,validation_errors))
    information = {}
    for label in ('data_only','nuisance_regularized'):
        values, ranks, svd_drivers = [], [], []
        for i,trial in enumerate(best['trials']):
            n = int(mask[i].sum()) if label=='data_only' else len(trial['residual'])
            j = (trial['stationarity_jacobian'] if initial_coordinates == 'positive_hb' else trial['jacobian'])[:n]
            b = value*trial['parameter_jacobian'][:n]
            u,s,_,svd_driver = _information_svd(j)
            svd_drivers.append(svd_driver)
            keep = s > (information_rcond*s[0] if len(s) else 0.)
            remaining = b-u[:,keep]@(u[:,keep].T@b)
            values.append(float(remaining@remaining)); ranks.append(int(keep.sum()))
        information[label] = dict(value=float(sum(values)),by_trial=values,nuisance_ranks=ranks,
                                 svd_drivers=svd_drivers)
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


def fit_nonlinear_shared_parameters(observations, parameters, dt, *, parameter_names,
        parameter_bounds, sd, mean_operator=None, visible=None, starts=None,
        penalty=.01, initial_penalty=100., flow_prior_weight=1., flow_prior_log_sd=np.log(2),
        substeps=4, max_iterations=150, max_evaluations_per_trial=600,
        gradient_tolerance=1e-5, numerical_backend='numba', curvature_breaks=(),
        observation_whiteners=None, parameter_prior_log_mean=None,
        parameter_prior_precision=None, return_jacobian=False,
        hemodynamics='tak', viscoelastic_s=2., initial_state_mode='volume_tied',
        initial_coordinate_weights=None, data_weight=1.):
    """Small joint-parameter diagnostic with exact block Gauss--Newton steps.

    Uses the same trial residual and physical domain as the scalar solver.
    Bounds apply to positive log parameters; rejected RK stages reject the
    proposal, with no clipping of states or substituted residuals. Each start
    receives a fixed forward-call budget. This supplies an engineering objective
    and local projected information, not a likelihood or confidence interval.

    An empty parameter_names sequence performs the nuisance optimization for a
    fixed-parameter profile. initial_penalty accepts one value per trajectory;
    curvature_breaks removes differences crossing declared original segments.
    Optional whiteners act on the *visible standardized residual vector* only,
    and must be constructed from the corresponding visible covariance. A log
    parameter prior contributes once per group, independently of trial count.
    Tak is the default and eliminates p as an independent state. Its default
    volume_tied initial mode fits four coordinates and penalizes volume once;
    rest is the fixed-initial literature control. Tak starts can contain [B,4]
    physical [s,f,v,q], or the retained [B,5] view with p==v. Returned states
    preserve [r,s,f,v,p,q] as an observation-compatible view with derived p.
    Historical callers explicitly select inlet_balance, viscoelastic_s=0 and
    their initial-state policy. Per-record operators may be [B,3T,3T].
    data_weight multiplies only the standardized observation SSE and tangents.
    A denser sampling grid can use dt/reference_dt to preserve the same
    per-second observation weight without changing the reported training SD.
    """
    parameters.validate()
    if not np.isfinite(data_weight) or data_weight <= 0:
        raise ValueError('data_weight must be finite and positive')
    _validate_hemodynamics(hemodynamics, viscoelastic_s)
    if initial_state_mode not in ('free', 'volume_tied', 'rest'):
        raise ValueError('initial_state_mode must be free, volume_tied or rest')
    if hemodynamics == 'tak' and initial_state_mode == 'free':
        raise ValueError('Tak eliminates independent p; use volume_tied or rest')
    if initial_coordinate_weights is None:
        initial_coordinate_weights = [1., 1., 1., 0., 1.] if hemodynamics == 'tak' else np.ones(5)
    initial_coordinate_weights = np.asarray(initial_coordinate_weights, float)
    if (initial_coordinate_weights.shape != (5,) or not np.isfinite(initial_coordinate_weights).all()
            or np.any(initial_coordinate_weights < 0)):
        raise ValueError('five finite nonnegative initial coordinate weights required')
    names = tuple(parameter_names)
    supported = {'tau','neurovascular_gain','kappa'}
    if hemodynamics == 'tak': supported |= {'gamma', 'viscoelastic_s'}
    if len(set(names)) != len(names) or not set(names) <= supported:
        raise ValueError('distinct supported physiological parameter names required')
    if 'kappa' in names and numerical_backend != 'numba':
        raise ValueError('joint kappa tangent requires the validated compiled backend')
    bounds = np.asarray(parameter_bounds, dtype=float)
    if not names and bounds.size == 0: bounds = bounds.reshape(0, 2)
    if bounds.shape != (len(names),2) or not np.isfinite(bounds).all() or np.any(bounds[:,0]<=0) or np.any(bounds[:,1]<=bounds[:,0]):
        raise ValueError('ordered finite positive bounds per parameter required')
    y = np.asarray(observations, dtype=float)
    if y.ndim != 3 or y.shape[0]<1 or y.shape[1]<3 or y.shape[2]!=3:
        raise ValueError('observations must be [B,T,3]')
    batch,n,_ = y.shape; count=n+5
    embedding = (_volume_tied_initial_embedding(n) if initial_state_mode == 'volume_tied'
        else np.eye(count)[:, :n] if initial_state_mode == 'rest' else None)
    mask = np.isfinite(y) if visible is None else np.asarray(visible)
    if mask.shape != y.shape or mask.dtype != bool or not mask.any(axis=(1,2)).all() or not np.isfinite(y[mask]).all():
        raise ValueError('finite visible observations and boolean mask required')
    scale = np.asarray(sd,dtype=float)
    if scale.shape == (3,):scale=np.broadcast_to(scale,y.shape)
    if scale.shape == (batch,3):scale=np.broadcast_to(scale[:,None,:],y.shape)
    if scale.shape != y.shape or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('positive finite observation scales required')
    initial_weights = np.broadcast_to(np.asarray(initial_penalty, dtype=float), (batch,))
    if (not np.isfinite([dt,penalty,flow_prior_weight,flow_prior_log_sd,gradient_tolerance]).all()
            or not np.isfinite(initial_weights).all() or np.any(initial_weights < 0)
            or min(dt,flow_prior_log_sd,gradient_tolerance)<=0 or min(penalty,flow_prior_weight)<0):
        raise ValueError('invalid objective controls')
    whiteners = [None]*batch if observation_whiteners is None else list(observation_whiteners)
    if len(whiteners) != batch:
        raise ValueError('one visible-residual whitener per trial required')
    for i, w in enumerate(whiteners):
        if w is not None and (np.shape(w) != (int(mask[i].sum()),)*2 or not np.isfinite(w).all()):
            raise ValueError('whitener must match visible residuals; hidden values cannot enter it')
    if (parameter_prior_log_mean is None) != (parameter_prior_precision is None):
        raise ValueError('both log prior mean and precision are required')
    prior_mean = np.zeros(len(names)) if parameter_prior_log_mean is None else np.asarray(parameter_prior_log_mean, float)
    prior_precision = np.zeros(len(names)) if parameter_prior_precision is None else np.asarray(parameter_prior_precision, float)
    if (prior_mean.shape != (len(names),) or prior_precision.shape != (len(names),)
            or not np.isfinite(prior_mean).all() or not np.isfinite(prior_precision).all() or np.any(prior_precision < 0)):
        raise ValueError('finite per-parameter log means and nonnegative diagonal precisions required')
    operator = None if mean_operator is None else np.asarray(mean_operator,dtype=float)
    if operator is not None and (operator.shape not in ((3*n,3*n), (batch,3*n,3*n)) or not np.isfinite(operator).all()):
        raise ValueError('finite common or per-record observation operator required')
    for v,minimum in ((substeps,1),(max_iterations,0),(max_evaluations_per_trial,3)):
        if isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<minimum:
            raise ValueError('positive integer numerical budgets required')
    regularizer=np.zeros((n-2,count));regularizer[:,:n]=np.sqrt(penalty)*np.diff(np.eye(n),n=2,axis=0)/dt**1.5
    breaks = list(curvature_breaks)
    if (any(isinstance(b, (bool, np.bool_)) or not isinstance(b, (int, np.integer)) for b in breaks)
            or breaks != sorted(set(breaks)) or any(b < 3 or b > n-3 for b in breaks)
            or any(b-a < 3 for a,b in zip(breaks, breaks[1:]))):
        raise ValueError('curvature_breaks requires ordered unique segment boundaries')
    for boundary in breaks: regularizer[boundary-2:boundary] = 0.
    lower,upper=np.log(bounds).T
    fixed_names = {'neurovascular_gain', 'gamma'}
    reference=np.array([viscoelastic_s if name == 'viscoelastic_s' else
        getattr(parameters.fixed if name in fixed_names else parameters.free,name) for name in names])
    def at(values):
        fixed={name:float(v) for name,v in zip(names,values) if name in fixed_names}
        free={name:float(v) for name,v in zip(names,values) if name in {'tau', 'kappa'}}
        tv = float(values[names.index('viscoelastic_s')]) if 'viscoelastic_s' in names else viscoelastic_s
        return replace(parameters,fixed=replace(parameters.fixed,**fixed),free=replace(parameters.free,**free)), tv
    if starts is None:starts=[dict(parameter_values=reference)]
    if not len(starts):raise ValueError('at least one start required')
    attempts,candidates=[],[]
    for start_index,start in enumerate(starts):
        values=np.asarray(start['parameter_values'],dtype=float)
        tolerance=8*np.spacing(bounds)
        if (values.shape != reference.shape or not np.isfinite(values).all()
                or np.any(values<bounds[:,0]-tolerance[:,0]) or np.any(values>bounds[:,1]+tolerance[:,1])):
            raise ValueError('start values outside declared parameter box')
        # exp(log(10)) can exceed 10 by one ULP in retained boundary solutions.
        # Only roundoff-sized excursions are projected, never outside starts.
        values=np.clip(values,bounds[:,0],bounds[:,1])
        driver=np.asarray(start.get('driver',np.zeros((batch,n))),dtype=float)
        initial=np.asarray(start.get('initial_state',np.tile(np.r_[0.,np.ones(4)],(batch,1))),dtype=float)
        if hemodynamics == 'tak' and initial.shape == (batch,4):
            initial = initial[:, [0, 1, 2, 2, 3]]
        if driver.shape!=(batch,n) or initial.shape!=(batch,5) or not np.isfinite(driver).all():
            raise ValueError('invalid nuisance start shape')
        for row in initial:_require_nonlinear_domain(row,parameters)
        if initial_state_mode == 'volume_tied' and np.any(initial[:,2] != initial[:,3]):
            raise ValueError('volume_tied starts require p0 == v0 exactly')
        if initial_state_mode == 'rest' and np.any(initial != np.r_[0.,np.ones(4)]):
            raise ValueError('rest starts require the fixed resting initial state')
        x=np.column_stack((driver,initial[:,0],np.log(initial[:,1:])))
        if initial_state_mode == 'volume_tied': x = np.delete(x, n+3, axis=1)
        elif initial_state_mode == 'rest': x = x[:, :n]
        logp=np.log(values);evaluations=0;rejections=0;damping=1e-3;converged=False
        def evaluate(xx,zz,derivative):
            nonlocal evaluations
            pp,tv=at(np.exp(zz));result=[]
            for i in range(batch):
                evaluations+=1
                if evaluations>max_evaluations_per_trial*batch:
                    raise RuntimeError('internal evaluation budget overflow')
                op = operator[i] if operator is not None and operator.ndim == 3 else operator
                result.append(_nonlinear_trial_objective(xx[i],pp,dt,op,y[i],mask[i],scale[i],
                    regularizer,initial_weights[i],substeps,derivative,names,flow_prior_weight,
                    flow_prior_log_sd,numerical_backend=numerical_backend,observation_whitener=whiteners[i],
                    initial_embedding=embedding, hemodynamics=hemodynamics, viscoelastic_s=tv,
                    initial_coordinate_weights=initial_coordinate_weights,data_weight=data_weight))
            return result,float(sum(t['objective'] for t in result)+np.sum(prior_precision*(zz-prior_mean)**2))
        try:current,cost=evaluate(x,logp,True)
        except (FloatingPointError,ValueError,OverflowError) as exc:
            attempts.append(dict(start=start_index,converged=False,status='failed_domain',error=repr(exc),evaluations=evaluations));continue
        reason='iteration_budget';scaled=float('inf')
        for iteration in range(max_iterations+1):
            values=np.exp(logp);pg=np.zeros(len(names));ph=np.zeros((len(names),len(names)));blocks=[]
            for t in current:
                j,r=t['jacobian'],t['residual'];jp=t['parameter_jacobian']*values[None,:]
                h,g,c=j.T@j,j.T@r,j.T@jp
                blocks.append((h,g,c));pg+=jp.T@r;ph+=jp.T@jp
            pg += prior_precision*(logp-prior_mean)
            ph += np.diag(prior_precision)
            active=((logp<=lower+1e-12)&(pg>=0))|((logp>=upper-1e-12)&(pg<=0))
            projected=pg.copy();projected[active]=0
            scaled=max([float(np.max(abs(projected)/np.sqrt(np.maximum(np.diag(ph),1.)))) if names else 0.]+
                [float(np.max(abs(g)/np.sqrt(np.maximum(np.diag(h),1.)))) for h,g,c in blocks])
            if scaled<=gradient_tolerance:converged=True;reason='projected_scaled_gradient';break
            if iteration==max_iterations:break
            if evaluations+2*batch>max_evaluations_per_trial*batch:reason='evaluation_budget';break
            try:
                eliminated=[cho_solve(cho_factor(h+damping*np.diag(np.maximum(np.diag(h),1.)),lower=True),np.column_stack((g,c))) for h,g,c in blocks]
                schur=ph+damping*np.diag(np.maximum(np.diag(ph),1.))
                rhs=pg.copy()
                for (_,_,c),inv in zip(blocks,eliminated):schur-=c.T@inv[:,1:];rhs-=c.T@inv[:,0]
                dp=np.linalg.solve(schur,-rhs)
                outward=((logp<=lower+1e-12)&(dp<0))|((logp>=upper-1e-12)&(dp>0))
                if outward.any():
                    keep=~outward;dp[:]=0
                    if keep.any():dp[keep]=np.linalg.solve(schur[np.ix_(keep,keep)],-rhs[keep])
                dx=np.array([-inv[:,0]-inv[:,1:]@dp for inv in eliminated])
                directional=float(pg@dp+sum(g@d for (_,g,_),d in zip(blocks,dx)))
                if not np.isfinite(directional) or directional>=0:raise np.linalg.LinAlgError('non-descent')
            except np.linalg.LinAlgError:
                damping*=10
                if damping>1e14:reason='singular_step';break
                continue
            fraction=1.
            for j in range(len(names)):
                if dp[j]>0:fraction=min(fraction,(upper[j]-logp[j])/dp[j])
                elif dp[j]<0:fraction=min(fraction,(lower[j]-logp[j])/dp[j])
            accepted=False
            for half in range(20):
                if evaluations+2*batch>max_evaluations_per_trial*batch:break
                step=fraction*2.**(-half)
                candidate_log=np.clip(logp+step*dp,lower,upper) # roundoff at exact parameter bound only
                try:trial,newcost=evaluate(x+step*dx,candidate_log,False)
                except (FloatingPointError,ValueError,OverflowError):rejections+=1;continue
                if newcost<cost and newcost<=cost+2e-4*step*directional:
                    try:updated,updatedcost=evaluate(x+step*dx,candidate_log,True)
                    except (FloatingPointError,ValueError,OverflowError):rejections+=1;continue
                    x+=step*dx;logp=candidate_log;current,cost=updated,updatedcost
                    damping=max(1e-12,damping/3);accepted=True;break
            if not accepted:
                damping*=10
                if damping>1e14:reason='line_search_failed';break
        record=dict(start=start_index,converged=converged,status='completed' if converged else 'failed_convergence',
            convergence_reason=reason,evaluations=evaluations,iterations=iteration,objective=cost,
            projected_scaled_gradient_inf_norm=scaled,parameter_values=np.exp(logp),domain_rejections=rejections)
        attempts.append(record);candidates.append((record,x,logp,current))
    if not candidates:return dict(status='failed_domain',converged=False,starts=attempts,parameter_names=names,parameter_values=None)
    good=[c for c in candidates if c[0]['converged']]
    record,x,logp,current=min(good or candidates,key=lambda c:c[0]['objective'])
    values=np.exp(logp);fitted,tv=at(values);initial=np.array([t['initial'] for t in current])
    replay=[replay_nonlinear_driver(x[i,:n],initial[i],fitted,dt,substeps=substeps,
        hemodynamics=hemodynamics, viscoelastic_s=tv) for i in range(batch)]
    errors=[float(np.max(abs(v['canonical_prediction']-t['forward']['canonical_prediction'])))
            if v['status']=='completed' else float('inf') for v,t in zip(replay,current)]
    valid=max(errors)<=1e-10
    info={}
    for label in ('conditional_data','data_only','nuisance_regularized'):
        matrix=np.zeros((len(names),len(names)));svd_drivers=[]
        for i,t in enumerate(current):
            length=int(mask[i].sum()) if label!='nuisance_regularized' else len(t['residual'])
            j=t['jacobian'][:length];jp=t['parameter_jacobian'][:length]*values
            if label == 'conditional_data' or not names:
                remainder = jp
            else:
                u,s,_,svd_driver=_information_svd(j);u=u[:,s>1e-10*s[0]];svd_drivers.append(svd_driver)
                remainder=jp-u@(u.T@jp)
            matrix+=remainder.T@remainder
        info[label]=dict(matrix=matrix,eigenvalues=np.linalg.eigvalsh(matrix),svd_drivers=svd_drivers)
    gradients={label:np.zeros(len(names)) for label in ('data','driver','initial','flow')}
    costs={label:0. for label in gradients}
    for i,t in enumerate(current):
        stops = [0, int(mask[i].sum()), int(mask[i].sum())+n-2, int(mask[i].sum())+n+3, len(t['residual'])]
        for label,left,right in zip(gradients,stops,stops[1:]):
            residual=t['residual'][left:right]
            gradients[label]+=2*t['parameter_jacobian'][left:right].T@residual
            costs[label]+=float(residual@residual)
    gradients['parameter_prior']=2*prior_precision*(logp-prior_mean)/values
    costs['parameter_prior']=float(np.sum(prior_precision*(logp-prior_mean)**2))
    raw_gradient=sum(gradients.values())
    projected_raw=raw_gradient.copy()
    projected_raw[((logp<=lower+1e-10)&(raw_gradient>=0))|((logp>=upper-1e-10)&(raw_gradient<=0))]=0.
    result = dict(status=record['status'] if valid else 'failed_replay_validation',
        converged=bool(record['converged'] and valid),parameter_names=names,parameter_values=values,
        parameter_bounds=bounds,starts=attempts,selected_start=record['start'],objective=record['objective'],
        projected_scaled_gradient_inf_norm=record['projected_scaled_gradient_inf_norm'],
        weighted_data_sse=data_weight*sum(np.sum(((t['prediction'][mask[i]]-y[i][mask[i]])/scale[i][mask[i]])**2) for i,t in enumerate(current)),
        flow_prior_cost=sum(t['flow_prior_cost'] for t in current),driver=x[:,:n],initial_state=initial,
        states=np.array([t['forward']['states'] for t in current]),prediction=np.array([t['prediction'] for t in current]),
        canonical_prediction=np.array([t['forward']['canonical_prediction'] for t in current]),
        replay_max_abs_difference=errors,information=info,uncertainty='NOT_ESTIMATED',
        physical_checks=[v['checks'] for v in replay], objective_components=costs,
        raw_parameter_gradient=raw_gradient, raw_parameter_gradient_components=gradients,
        raw_parameter_kkt_inf_norm=float(np.max(abs(projected_raw))) if names else 0.,
        projected_raw_parameter_gradient=projected_raw,
        log_parameter_boundary_distance=np.minimum((logp-lower)/(upper-lower),(upper-logp)/(upper-lower)),
        evaluations=sum(a['evaluations'] for a in attempts)+batch,
        tau=fitted.free.tau,neurovascular_gain=fitted.fixed.neurovascular_gain,kappa=fitted.free.kappa,
        gamma=fitted.fixed.gamma, viscoelastic_s=tv, hemodynamics=hemodynamics,
        initial_state_mode=initial_state_mode, initial_coordinate_weights=initial_coordinate_weights,
        data_weight=float(data_weight),
        initial_free_parameters={'free':5, 'volume_tied':4, 'rest':0}[initial_state_mode])
    if return_jacobian:
        result.update(residual_jacobians=[t['jacobian'] for t in current],
            residual_parameter_jacobians=[t['parameter_jacobian'] for t in current],
            residuals=[t['residual'] for t in current])
    return result
