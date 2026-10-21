"""Diagnostics for a scalar structured Hb component, without source labels."""
from __future__ import annotations

import numpy as np


def hb_diagnostic_coordinates(values, rho=.35):
    """Preserve the native relative Hb amplitudes; this is not saturation."""
    values = np.asarray(values, float)
    if values.shape[-1] != 2 or not 0 < rho < 1:
        raise ValueError('expected paired HbO/HbR and rho in (0,1)')
    return np.stack((values[..., 0]+values[..., 1],
                     rho*values[..., 0]-(1-rho)*values[..., 1]), axis=-1)


def equal_capacity_hb_basis(operator, *, modes=4, rho=.35, exchange=False):
    """One direction and K cosines, with the same norm as [0.65,0.35]."""
    operator = np.asarray(operator, float)
    n = operator.shape[0]//3
    if operator.shape != (3*n, 3*n) or not np.isfinite(operator).all():
        raise ValueError('operator must be finite [3T,3T]')
    if not 0 < rho < 1 or not 1 <= modes < n:
        raise ValueError('invalid rho or temporal rank')
    loading = np.array([-1., 1.]) if exchange else np.array([1-rho, rho])
    loading *= np.linalg.norm([.65, .35])/np.linalg.norm(loading)
    time = np.cos(np.pi*(np.arange(n)[:, None]+.5)*np.arange(1, modes+1)/n)
    canonical = time[:, None, :]*np.r_[0., loading][None, :, None]
    return operator@canonical.reshape(3*n, modes)


def project_component(residual, basis, sd, coefficient_sd=None, visible=None):
    """Visible-only projection; proper ridge when a coefficient SD is supplied."""
    residual = np.asarray(residual, float)
    mask = np.isfinite(residual) if visible is None else np.asarray(visible, bool)
    if residual.ndim != 2 or residual.shape[1] != 3 or mask.shape != residual.shape:
        raise ValueError('residual and visible must be [T,3]')
    scale = np.broadcast_to(np.asarray(sd, float), residual.shape)
    if np.any(scale <= 0) or not np.isfinite(scale).all() or not np.isfinite(residual[mask]).all():
        raise ValueError('visible values and positive scale required')
    basis = np.asarray(basis, float)
    if basis.ndim != 2 or basis.shape[0] != residual.size or np.any(basis[0::3]):
        raise ValueError('basis must have zero EEG rows and [3T,K] shape')
    design = basis[mask.ravel()]/scale[mask, None]
    target = residual[mask]/scale[mask]
    if coefficient_sd is None:
        return np.linalg.lstsq(design, target, rcond=1e-10)[0]
    cs = np.broadcast_to(np.asarray(coefficient_sd, float), (basis.shape[1],))
    if np.any(cs <= 0) or not np.isfinite(cs).all():
        raise ValueError('positive coefficient SD required')
    return np.linalg.solve(design.T@design+np.diag(cs**-2), design.T@target)


def subspace_overlap(jacobian, basis, sd):
    """Observation-space principal cosines, not an identifiability verdict."""
    scale = np.tile(np.asarray(sd, float), basis.shape[0]//3)
    j = np.asarray(jacobian, float)/scale[:, None]
    b = np.asarray(basis, float)/scale[:, None]
    def orth(x):
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        return u[:, s > (s[0]*1e-9 if len(s) else 0)]
    qj, qb = orth(j), orth(b)
    _, singular, vh = np.linalg.svd(qj.T@qb, full_matrices=False)
    direction = qb@vh[0] if len(singular) else np.zeros(len(scale))
    coefficients = np.linalg.lstsq(b, direction, rcond=1e-10)[0]
    return dict(principal_cosines=np.clip(singular, 0, 1),
                physical_rank=qj.shape[1], component_rank=qb.shape[1],
                coefficient_direction=coefficients)


def component_change_metrics(reference, alternative, sd, driver_sd):
    """Track physical and extra outputs separately, including component shape."""
    if 'driver' not in reference or 'driver' not in alternative:
        return dict(available=False)
    c0, c1 = reference['observation_component'][:, 1:], alternative['observation_component'][:, 1:]
    a, b = c0.ravel(), c1.ravel()
    cosine = float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if min(np.linalg.norm(a), np.linalg.norm(b)) > 1e-12 else np.nan
    return dict(available=True,
        driver_change_SD=float(np.sqrt(np.mean((reference['driver']-alternative['driver'])**2))/driver_sd),
        physical_change_SD=float(np.sqrt(np.mean(((reference['physical_prediction'][:,1:]-alternative['physical_prediction'][:,1:])/np.asarray(sd)[1:])**2))),
        component_change_SD=float(np.sqrt(np.mean(((c0-c1)/np.asarray(sd)[1:])**2))),
        component_shape_cosine=cosine)


def fit_standardized_ridge(features, targets, weight):
    """Training-only affine ridge; retain the exact training coordinate."""
    x, y = np.asarray(features, float), np.asarray(targets, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError('ridge needs aligned [N,D] training arrays')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.maximum(scale, 1e-8)
    target_mean = y.mean(axis=0)
    z = (x-mean)/scale
    coefficient = np.linalg.solve(z.T@z+weight*len(x)*np.eye(x.shape[1]), z.T@(y-target_mean))
    return dict(mean=mean, scale=scale, target_mean=target_mean, coefficient=coefficient)


def predict_standardized_ridge(model, features):
    return ((np.asarray(features)-model['mean'])/model['scale'])@model['coefficient']+model['target_mean']


def fit_conditioned_shared_driver(observations, parameters, dt, *, component_mean=None,
                                 visible=None, **options):
    """Fit the existing profiled solver around an independently supplied mean.

    The mean is a fixed processed-coordinate observation, not a physiological
    state. Optional profiled coefficients describe deviations from that mean;
    their existing proper penalty is unchanged. Hidden target values never
    enter subtraction, initialization, or the solver. The caller owns the
    independent donor/training boundary.
    """
    from .shared_driver_reconstruction import fit_nonlinear_shared_driver

    y = np.asarray(observations, dtype=float)
    mask = np.isfinite(y) if visible is None else np.asarray(visible)
    if y.ndim != 2 or y.shape[1] != 3 or mask.shape != y.shape or mask.dtype != np.bool_:
        raise ValueError('observations and boolean visible must be [T,3]')
    mean = np.zeros_like(y) if component_mean is None else np.asarray(component_mean, dtype=float)
    if mean.shape != y.shape or not np.isfinite(mean).all() or np.any(mean[:, 0] != 0):
        raise ValueError('independent component mean must be finite [T,3] with zero EEG')
    shifted = np.full_like(y, np.nan)
    shifted[mask] = y[mask]-mean[mask]
    result = fit_nonlinear_shared_driver(shifted, parameters, dt, visible=mask, **options)
    if 'prediction' in result:
        result['component_deviation'] = result['observation_component'].copy()
        result['component_mean'] = mean.copy()
        result['observation_component'] = mean+result['component_deviation']
        result['prediction'] = result['physical_prediction']+result['observation_component']
    return result


def fit_reduced_rank_readout(features, targets, *, rank, weight):
    """Exact reduced-rank ridge, with training RMS scaling and no intercept.

    Minimize ||Y-X B||_F^2 + N*weight*||B||_F^2 subject to rank(B)<=rank,
    after dividing donor columns by their training RMS. No target centering,
    per-window normalization, target residual fitting, or free time course is
    added. With a shared linear temporal projector this preserves its support.
    """
    x, y = np.asarray(features, float), np.asarray(targets, float)
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or isinstance(rank, bool) or not isinstance(rank, (int, np.integer))
            or not 1 <= rank <= min(x.shape[1], y.shape[1])
            or not np.isfinite(weight) or weight <= 0):
        raise ValueError('finite aligned training arrays, admissible rank and positive ridge required')
    scale = np.maximum(np.sqrt(np.mean(x*x, axis=0)), 1e-12)
    z = x/scale
    chol = np.linalg.cholesky(z.T@z+len(z)*weight*np.eye(z.shape[1]))
    transformed = np.linalg.solve(chol, z.T@y)
    u, singular, vt = np.linalg.svd(transformed, full_matrices=False)
    coefficient = np.linalg.solve(chol.T, (u[:, :rank]*singular[:rank])@vt[:rank])
    return dict(scale=scale, coefficient=coefficient, rank=int(rank), weight=float(weight),
                training_rows=len(x), singular_values=singular)


def predict_reduced_rank_readout(model, features):
    x = np.asarray(features, float)
    scale, coefficient = np.asarray(model['scale']), np.asarray(model['coefficient'])
    if x.ndim != 2 or x.shape[1] != len(scale) or not np.isfinite(x).all():
        raise ValueError('complete independent donor features with frozen column order required')
    return (x/scale)@coefficient


def grouped_component_overlap(data_jacobian, component_basis, *, groups,
                              physical_penalty=None, component_penalty=None):
    """Principal cosines for each tangent group, in supplied weighted rows.

    Data and augmented engineering-objective spaces are kept separate. Penalty
    rows for physical coordinates and component coefficients are orthogonal
    blocks. These are local diagnostics, not likelihood confidence intervals.
    """
    j, b = np.asarray(data_jacobian, float), np.asarray(component_basis, float)
    if j.ndim != 2 or b.ndim != 2 or len(j) != len(b):
        raise ValueError('aligned weighted data Jacobian and component basis required')
    p = np.zeros((0, j.shape[1])) if physical_penalty is None else np.asarray(physical_penalty, float)
    q = np.zeros((0, b.shape[1])) if component_penalty is None else np.asarray(component_penalty, float)
    if (p.ndim != 2 or p.shape[1] != j.shape[1] or q.ndim != 2 or q.shape[1] != b.shape[1]
            or not all(np.isfinite(v).all() for v in (j, b, p, q))):
        raise ValueError('finite compatible penalty rows required')

    def compare(a, c):
        def orth(v):
            u, s, _ = np.linalg.svd(v, full_matrices=False)
            return u[:, s > (s[0]*1e-9 if len(s) else 0)]
        qa, qc = orth(a), orth(c)
        if not qa.shape[1] or not qc.shape[1]:
            return dict(cosines=[], coefficient_direction=np.zeros(b.shape[1]), rank=qa.shape[1])
        _, cosines, vt = np.linalg.svd(qa.T@qc, full_matrices=False)
        direction = np.linalg.lstsq(c, qc@vt[0], rcond=1e-10)[0]
        return dict(cosines=np.clip(cosines, 0, 1), coefficient_direction=direction, rank=qa.shape[1])

    result = {}
    for name, columns in groups.items():
        columns = np.asarray(columns, dtype=int)
        a = np.vstack((j[:, columns], p[:, columns], np.zeros((len(q), len(columns)))))
        c = np.vstack((b, np.zeros((len(p), b.shape[1])), q))
        result[name] = dict(data=compare(j[:, columns], b), objective=compare(a, c))
    return result
