"""Optional compiled arithmetic for the existing ZOH Balloon RK4 map.

This module is a numerical backend, not a different model or optimizer. The
public wrapper validates inputs and the original implementation independently
replays every selected fit. No fastmath, clipping, or additional states.
"""
import numpy as np
from numba import njit


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
