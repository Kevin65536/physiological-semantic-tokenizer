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
