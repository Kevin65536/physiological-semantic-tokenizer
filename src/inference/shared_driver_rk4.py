"""Compiled RK4 for the Tak baseline and explicit historical Balloon controls.

This module is a numerical backend, not a different model or optimizer. The
public wrapper validates inputs and the original implementation independently
replays every selected fit. No fastmath, clipping, or additional states.

The separately named decay_forward diagnostic changes only the p-v decay
equation and optionally integrates a declared finite-dimensional input there.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _tak_slope(x, d, r, step, count, p, derivative):
    """Tak/SPM hemodynamics in four physical coordinates [s,f,v,q]."""
    alpha, e0, gamma, p0, q0, beta, kappa, tau, tv = p
    s, f, v, q = x
    if not np.isfinite(x).all() or min(x[1:]) <= 0 or q0*q > p0*v:
        raise FloatingPointError('invalid Tak RK stage')
    power, total = 1/alpha, tau+tv
    steady = v**power
    dv = (f-steady)/total
    out = steady+tv*dv
    loge = np.log1p(-e0)
    extraction = -np.expm1(loge/f)
    dq = (f*extraction/e0-out*q/v)/tau
    value = np.array([beta*r-kappa*s-gamma*(f-1), s, dv, dq])
    ds = np.zeros_like(d)
    if derivative:
        out_f, out_v = tv/total, tau/total*power*steady/v
        a = np.zeros((4, 4))
        a[0, 0], a[0, 1], a[1, 0] = -kappa, -gamma, 1.
        a[2, 1], a[2, 2] = 1/total, -power*steady/v/total
        a[3, 1] = ((extraction+np.exp(loge/f)*loge/f)/e0-out_f*q/v)/tau
        a[3, 2], a[3, 3] = -(out_v-out/v)*q/v/tau, -out/v/tau
        ds = a@d
        ds[0, step] += beta
        # Physical derivatives: tau, gain, kappa, gamma, viscoelastic_s.
        ds[2, count] -= dv/total
        ds[3, count] += tv*dv*q/(v*total*tau)-dq/tau
        ds[0, count+1] += r
        ds[0, count+2] -= s
        ds[0, count+3] -= f-1
        ds[2, count+4] -= dv/total
        ds[3, count+4] -= dv*q/(v*total)
    if not np.isfinite(value).all() or not np.isfinite(ds).all():
        raise FloatingPointError('nonfinite Tak RK tangent')
    return value, ds


@njit(cache=True)
def _integrate_tak(r, initial, p, dt, substeps, derivative):
    n, count = len(r), len(r)+5
    states = np.empty((n, 6))
    tangents = np.zeros((n, 4, count+5 if derivative else 0))
    x = initial[np.array([0, 1, 2, 4])].copy()
    d = np.zeros((4, count+5 if derivative else 0))
    if derivative:
        d[0, n], d[1, n+1], d[2, n+2], d[3, n+4] = 1., x[1], x[2], x[3]
    h = dt/substeps
    for t in range(n):
        # p is derived from v, never independently integrated or innovated.
        states[t] = np.array([r[t], x[0], x[1], x[2], x[2], x[3]])
        tangents[t] = d
        if t == n-1:
            break
        for _ in range(substeps):
            k1, d1 = _tak_slope(x, d, r[t], t, count, p, derivative)
            k2, d2 = _tak_slope(x+h*k1/2, d+h*d1/2, r[t], t, count, p, derivative)
            k3, d3 = _tak_slope(x+h*k2/2, d+h*d2/2, r[t], t, count, p, derivative)
            k4, d4 = _tak_slope(x+h*k3, d+h*d3, r[t], t, count, p, derivative)
            x += h*(k1+2*k2+2*k3+k4)/6
            d += h*(d1+2*d2+2*d3+d4)/6
            if not np.isfinite(x).all() or min(x[1:]) <= 0 or p[4]*x[3] > p[3]*x[2]:
                raise FloatingPointError('invalid Tak RK endpoint')
    return states, tangents


def tak_forward(r, initial, parameters, dt, substeps, derivative, flow, viscoelastic_s):
    """Reduced deterministic Tak core; caller validates the shared input contract.

    The legacy-shaped initial Jacobian has a zero log(p0) column. Its log(v0)
    column already differentiates both v and p. The fitting embedding removes
    the unused p coordinate, so it adds no parameter or stochastic innovation.
    """
    if initial[3] != initial[2]:
        raise ValueError('Tak reduced core requires p0 == v0 exactly')
    if not np.isfinite(viscoelastic_s) or viscoelastic_s < 0:
        raise ValueError('Tak viscoelastic_s must be finite and nonnegative')
    fixed, free = parameters.fixed, parameters.free
    p = np.array([fixed.alpha, fixed.E0, fixed.gamma, fixed.P0, fixed.Q0,
                  fixed.neurovascular_gain, free.kappa, free.tau, viscoelastic_s])
    states, tangent = _integrate_tak(r, initial, p, dt, substeps, derivative)
    n, count = len(r), len(r)+5
    prediction = np.column_stack((fixed.eeg_loading*r,
        fixed.P0*(states[:, 3]-1)-fixed.Q0*(states[:, 5]-1),
        fixed.Q0*(states[:, 5]-1)))
    result = dict(states=states, canonical_prediction=prediction, jacobian=None,
                  hemodynamics='tak', viscoelastic_s=float(viscoelastic_s))
    names = ('tau', 'neurovascular_gain', 'kappa', 'gamma', 'viscoelastic_s')
    for name in names:
        result[name+'_jacobian'] = None
        if flow:
            result['flow_'+name+'_jacobian'] = None
    if flow:
        result['flow_jacobian'] = None
    if derivative:
        j = np.zeros((n, 3, count+5))
        j[:, 0, :n] = fixed.eeg_loading*np.eye(n)
        j[:, 1] = fixed.P0*tangent[:, 2]-fixed.Q0*tangent[:, 3]
        j[:, 2] = fixed.Q0*tangent[:, 3]
        result['jacobian'] = j[:, :, :count].reshape(3*n, count)
        for k, name in enumerate(names):
            result[name+'_jacobian'] = j[:, :, count+k]
            if flow:
                result['flow_'+name+'_jacobian'] = tangent[:, 1, count+k].copy()
        if flow:
            result['flow_jacobian'] = tangent[:, 1, :count].copy()
    return result


@njit(cache=True)
def _slope(x, d, r, step, count, p, derivative):
    alpha, e0, gamma, p0, q0, beta, kappa, tau = p
    if not np.isfinite(x).all() or min(x[1:]) <= 0 or q0*x[4] > p0*x[3]:
        raise FloatingPointError('invalid physical RK stage')
    s, f, v, total, deoxy = x
    power = 1/alpha
    out = v**power
    loss = out/v
    loge = np.log1p(-e0)
    extraction = -np.expm1(loge/f)
    value = np.array([beta*r-kappa*s-gamma*(f-1), s,
                      (f-out)/tau, (f-loss*total)/tau,
                      (f*extraction/e0-loss*deoxy)/tau])
    ds = np.zeros_like(d)
    if derivative:
        a = np.zeros((5, 5))
        a[0, 0], a[0, 1], a[1, 0] = -kappa, -gamma, 1.
        a[2, 1], a[2, 2] = 1/tau, -power*loss/tau
        a[3, 1], a[3, 2], a[3, 3] = 1/tau, -(power-1)*loss*total/v/tau, -loss/tau
        a[4, 1] = (extraction+np.exp(loge/f)*loge/f)/e0/tau
        a[4, 2], a[4, 4] = -(power-1)*loss*deoxy/v/tau, -loss/tau
        ds = a@d
        ds[0, step] += beta
        ds[2:, count] -= value[2:]/tau
        ds[0, count+1] += r
        ds[0, count+2] -= s
    if not np.isfinite(value).all() or not np.isfinite(ds).all():
        raise FloatingPointError('nonfinite physical RK tangent')
    return value, ds


@njit(cache=True)
def _integrate(r, initial, p, dt, substeps, derivative):
    n = len(r)
    count = n+5
    states = np.empty((n, 6))
    tangents = np.zeros((n, 5, count+3 if derivative else 0))
    x = initial.copy()
    d = np.zeros((5, count+3 if derivative else 0))
    if derivative:
        d[0, n] = 1.
        for k in range(1, 5):
            d[k, n+k] = initial[k]
    states[0, 0], states[0, 1:] = r[0], x
    tangents[0] = d
    h = dt/substeps
    for t in range(n-1):
        for _ in range(substeps):
            k1, d1 = _slope(x, d, r[t], t, count, p, derivative)
            k2, d2 = _slope(x+h*k1/2, d+h*d1/2, r[t], t, count, p, derivative)
            k3, d3 = _slope(x+h*k2/2, d+h*d2/2, r[t], t, count, p, derivative)
            k4, d4 = _slope(x+h*k3, d+h*d3, r[t], t, count, p, derivative)
            x += h*(k1+2*k2+2*k3+k4)/6
            d += h*(d1+2*d2+2*d3+d4)/6
            if not np.isfinite(x).all() or min(x[1:]) <= 0 or p[4]*x[4] > p[3]*x[3]:
                raise FloatingPointError('invalid physical RK endpoint')
        states[t+1, 0], states[t+1, 1:] = r[t+1], x
        tangents[t+1] = d
    return states, tangents


def compiled_forward(r, initial, parameters, dt, substeps, derivative, flow, *, state=False):
    fixed, free = parameters.fixed, parameters.free
    p = np.array([fixed.alpha, fixed.E0, fixed.gamma, fixed.P0, fixed.Q0,
                  fixed.neurovascular_gain, free.kappa, free.tau])
    states, tangent = _integrate(r, initial, p, dt, substeps, derivative)
    n, count = len(r), len(r)+5
    prediction = np.column_stack((fixed.eeg_loading*r,
        fixed.P0*(states[:, 4]-1)-fixed.Q0*(states[:, 5]-1),
        fixed.Q0*(states[:, 5]-1)))
    result = dict(states=states, canonical_prediction=prediction, jacobian=None,
                  tau_jacobian=None, neurovascular_gain_jacobian=None, kappa_jacobian=None)
    if derivative:
        j = np.zeros((n, 3, count+3))
        j[:, 0, :n] = fixed.eeg_loading*np.eye(n)
        j[:, 1] = fixed.P0*tangent[:, 3]-fixed.Q0*tangent[:, 4]
        j[:, 2] = fixed.Q0*tangent[:, 4]
        result['jacobian'] = j[:, :, :count].reshape(3*n, count)
        for k, name in enumerate(('tau', 'neurovascular_gain', 'kappa')):
            result[name+'_jacobian'] = j[:, :, count+k]
    if flow:
        result['flow_jacobian'] = tangent[:, 1, :count].copy() if derivative else None
        for k, name in enumerate(('tau', 'neurovascular_gain', 'kappa')):
            result['flow_'+name+'_jacobian'] = tangent[:, 1, count+k].copy() if derivative else None
    if state:
        # Optional physical-state derivatives of exactly the same discrete map.
        # Existing callers retain their original result and allocation contract.
        result['state_jacobian'] = None
        result['state_parameter_jacobian'] = None
        if derivative:
            full = np.zeros((n, 6, count+3))
            full[:, 0, :n] = np.eye(n)
            full[:, 1:] = tangent
            result['state_jacobian'] = full[:, :, :count]
            result['state_parameter_jacobian'] = full[:, :, count:]
    return result


@njit(cache=True)
def _decay_slope(x, d, r, step, p, multiplier, coefficients, clock, length,
                 phase, derivative):
    """Physical RHS and tangent for an intervention only on delta=p-v."""
    alpha, e0, gamma, p0, q0, beta, kappa, tau = p
    if not np.isfinite(x).all() or min(x[1:]) <= 0 or q0*x[4] > p0*x[3]:
        raise FloatingPointError('invalid physical decay RK stage')
    s, f, v, total, deoxy = x
    power = 1/alpha
    out = v**power
    loss = out/v
    loge = np.log1p(-e0)
    extraction = -np.expm1(loge/f)
    value = np.array([beta*r-kappa*s-gamma*(f-1), s,
                      (f-out)/tau, (f-loss*total)/tau,
                      (f*extraction/e0-loss*deoxy)/tau])
    if multiplier != 1.:
        value[3] = value[2]-multiplier*loss*(total-v)/tau
    basis = np.empty(len(coefficients))
    for j in range(len(coefficients)):
        basis[j] = np.cos(np.pi*j*(clock+phase)/length)
        value[3] += coefficients[j]*basis[j]
    ds = np.zeros_like(d)
    if derivative:
        a = np.zeros((5, 5))
        a[0, 0], a[0, 1], a[1, 0] = -kappa, -gamma, 1.
        a[2, 1], a[2, 2] = 1/tau, -power*loss/tau
        a[3, 1], a[3, 2], a[3, 3] = 1/tau, -(power-1)*loss*total/v/tau, -loss/tau
        if multiplier != 1.:
            a[3, 2] = ((multiplier-power)*loss
                       -multiplier*(power-1)*loss*(total-v)/v)/tau
            a[3, 3] = -multiplier*loss/tau
        a[4, 1] = (extraction+np.exp(loge/f)*loge/f)/e0/tau
        a[4, 2], a[4, 4] = -(power-1)*loss*deoxy/v/tau, -loss/tau
        ds = a@d
        ds[0, step] += beta
        first_coefficient = d.shape[1]-len(coefficients)
        for j in range(len(coefficients)):
            ds[3, first_coefficient+j] += basis[j]
    if not np.isfinite(value).all() or not np.isfinite(ds).all() or not np.isfinite(loss):
        raise FloatingPointError('nonfinite physical decay RK tangent')
    return value, ds, loss/tau


@njit(cache=True)
def _integrate_decay(r, initial, p, dt, substeps, multiplier, coefficients, derivative):
    n = len(r)
    width = n+5+len(coefficients) if derivative else 0
    states = np.empty((n, 6))
    tangents = np.zeros((n, 5, width))
    integral = np.zeros(n)
    accumulated = 0.
    x = initial.copy()
    d = np.zeros((5, width))
    if derivative:
        d[0, n] = 1.
        for k in range(1, 5):
            d[k, n+k] = initial[k]
    states[0, 0], states[0, 1:] = r[0], x
    tangents[0] = d
    h, length, phase = dt/substeps, n*dt, dt/2
    for t in range(n-1):
        for substep in range(substeps):
            clock = t*dt+substep*h
            k1, d1, i1 = _decay_slope(x, d, r[t], t, p, multiplier, coefficients,
                                      clock, length, phase, derivative)
            k2, d2, i2 = _decay_slope(x+h*k1/2, d+h*d1/2, r[t], t, p, multiplier,
                                      coefficients, clock+h/2, length, phase, derivative)
            k3, d3, i3 = _decay_slope(x+h*k2/2, d+h*d2/2, r[t], t, p, multiplier,
                                      coefficients, clock+h/2, length, phase, derivative)
            k4, d4, i4 = _decay_slope(x+h*k3, d+h*d3, r[t], t, p, multiplier,
                                      coefficients, clock+h, length, phase, derivative)
            x += h*(k1+2*k2+2*k3+k4)/6
            d += h*(d1+2*d2+2*d3+d4)/6
            accumulated += h*(i1+2*i2+2*i3+i4)/6
            if not np.isfinite(x).all() or min(x[1:]) <= 0 or p[4]*x[4] > p[3]*x[3]:
                raise FloatingPointError('invalid physical decay RK endpoint')
        states[t+1, 0], states[t+1, 1:] = r[t+1], x
        tangents[t+1] = d
        integral[t+1] = accumulated
    return states, tangents, integral


def decay_forward(driver, initial, parameters, dt, *, decay_multiplier=1.,
                  forcing_coefficients=None, substeps=4, derivative=False):
    """ZOH driver with a continuous, finite cosine input to delta=p-v.

    Only p_dot is changed: v_dot-lambda*v**(1/alpha-1)*(p-v)/tau+u(t).
    The other four physical equations and the Hb observation map are retained.
    All physiology is supplied and fixed. Input coefficients have normalized
    state/s units; u(t)=sum(c[k]*cos(pi*k*(t+dt/2)/(T*dt))), k=0..M-1.
    The first coefficient is a constant input. Thus M=7 at 30 s and M=19
    at 90 s include DC plus 6/18 cosines at the same maximum bandwidth.
    The basis is evaluated at every RK stage, never added to the output.

    Exact discrete Jacobian columns are [r[T],s0,logf0,logv0,logp0,logq0,c[M]].
    retention_integral is the RK quadrature of integral(v**(1/alpha-1)/tau),
    and decay_retention=exp(-lambda*retention_integral). The latter describes
    homogeneous continuous-time retention; it is not a confidence interval.
    Every RK stage and accepted endpoint must satisfy f,v,p,q>0 and P0*p>=Q0*q.
    No clipping or physiological parameter fitting is performed.
    """
    parameters.validate()
    r, initial = np.asarray(driver, float), np.asarray(initial, float)
    coefficients = (np.empty(0) if forcing_coefficients is None
                    else np.asarray(forcing_coefficients, float))
    if r.ndim != 1 or not len(r) or not np.isfinite(r).all():
        raise ValueError('driver must be nonempty finite [T]')
    if initial.shape != (5,) or not np.isfinite(initial).all():
        raise ValueError('initial must be finite physical [s,f,v,p,q]')
    if coefficients.ndim != 1 or len(coefficients) >= len(r) or not np.isfinite(coefficients).all():
        raise ValueError('forcing_coefficients must be finite [M], 0 <= M < T')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if not np.isfinite(decay_multiplier) or decay_multiplier < 0:
        raise ValueError('decay_multiplier must be finite and nonnegative')
    if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
        raise ValueError('substeps must be a positive integer')
    if not isinstance(derivative, (bool, np.bool_)):
        raise ValueError('derivative must be boolean')
    fixed, free = parameters.fixed, parameters.free
    if min(initial[1:]) <= 0 or fixed.Q0*initial[4] > fixed.P0*initial[3]:
        raise FloatingPointError('invalid physical decay initial state')
    p = np.array([fixed.alpha, fixed.E0, fixed.gamma, fixed.P0, fixed.Q0,
                  fixed.neurovascular_gain, free.kappa, free.tau])
    states, tangent, integral = _integrate_decay(r, initial, p, float(dt), int(substeps),
        float(decay_multiplier), coefficients, bool(derivative))
    n, count = len(r), len(r)+5+len(coefficients)
    basis = np.cos(np.pi*(np.arange(n)[:, None]+.5)*np.arange(len(coefficients))/n)
    prediction = np.column_stack((fixed.eeg_loading*r+fixed.eeg_offset,
        fixed.P0*(states[:, 4]-1)-fixed.Q0*(states[:, 5]-1),
        fixed.Q0*(states[:, 5]-1)))
    result = dict(states=states, canonical_prediction=prediction, jacobian=None,
        flow_jacobian=None, state_jacobian=None, forcing=basis@coefficients,
        forcing_basis=basis, forcing_coefficients=coefficients.copy(),
        forcing_angular_frequencies=np.pi*np.arange(len(coefficients))/(n*dt),
        forcing_phase_s=float(dt)/2, forcing_duration_s=n*float(dt),
        retention_integral=integral, decay_retention=np.exp(-decay_multiplier*integral))
    if derivative:
        j = np.zeros((n, 3, count))
        j[:, 0, :n] = fixed.eeg_loading*np.eye(n)
        j[:, 1] = fixed.P0*tangent[:, 3]-fixed.Q0*tangent[:, 4]
        j[:, 2] = fixed.Q0*tangent[:, 4]
        result['jacobian'] = j.reshape(3*n, count)
        result['flow_jacobian'] = tangent[:, 1].copy()
        full = np.zeros((n, 6, count))
        full[:, 0, :n] = np.eye(n)
        full[:, 1:] = tangent
        result['state_jacobian'] = full
    return result
