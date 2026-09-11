"""Shared observation transforms, noise estimates and linear SSM baselines.

These array-only helpers use the caller's existing observation contract. Callers
own train/validation membership, masks, candidate selection and data access.
Importing this module neither loads recordings nor changes process settings.
"""

import numpy as np
from scipy.signal import resample_poly
from scipy.stats import t as student_t


def robust_mad(values, axis=0):
    """Gaussian-consistent median absolute deviation along one array axis."""
    values = np.asarray(values, dtype=float)
    median = np.median(values, axis=axis, keepdims=True)
    return 1.482602218505602 * np.median(abs(values - median), axis=axis)


def student_difference_mad(nu):
    """Median absolute difference of two independent unit-scale Student draws."""
    from scipy.integrate import quad
    from scipy.optimize import brentq
    def mass(q):
        return quad(lambda x:student_t.pdf(x,nu)*(student_t.cdf(x+q,nu)-student_t.cdf(x-q,nu)),
                    -np.inf,np.inf,epsabs=1e-10)[0]-.5
    return float(brentq(mass,.01,10.))


def first_difference_noise(trials, constant):
    """Estimate scale within trials, using the specified difference-law constant."""
    differences = np.concatenate([np.diff(y, axis=0) for y in trials], axis=0)
    return robust_mad(differences) / 1.482602218505602 / constant


def bridge_transform(values, variant, cfg):
    """Apply the configured finite-window transform to EEG/HbO/HbR columns."""
    y = np.array(values, copy=True)
    b = cfg['bridge']
    if variant in ('resample_roundtrip', 'combined'):
        y[:, 1:] = resample_poly(resample_poly(y[:, 1:], 5, 2, axis=0), 2, 5, axis=0)[:len(y)]
    if variant in ('fnirs_filter', 'combined'):
        # Import the data package only when its filter is used; importing noise
        # or ridge helpers must not initialize its visualization dependencies.
        from src.data.homer2_preprocessing import bandpass_fnirs

        y[:, 1:], info = bandpass_fnirs(y[:, 1:], sample_rate_hz=4.,
            low_hz=b['filter_hz'][0], high_hz=b['filter_hz'][1], order=b['filter_order'])
        if info['status'] != 'applied':
            raise ValueError('bridge filter was skipped')
    if variant in ('baseline', 'combined'):
        y -= y[:b['baseline_samples']].mean(axis=0)
    if variant in ('common_scale', 'combined'):
        y[:, 1:] *= b['common_fnirs_factor']
    return y


def ridge_fit(x, y, alpha):
    """Fit centering, scaling and ridge coefficients on supplied training rows."""
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
    z = (x-center)/scale
    target_center = y.mean(axis=0)
    coefficients = np.linalg.solve(z.T@z/len(z)+alpha*np.eye(z.shape[1]), z.T@(y-target_center)/len(z))
    return dict(center=center, scale=scale, target_center=target_center, coefficients=coefficients)


def ridge_predict(model, x):
    """Apply frozen training normalization and coefficients."""
    return (x-model['center'])/model['scale']@model['coefficients']+model['target_center']


def linear_features(masked, template, modality, cfg, other=None):
    """Build own/template and lagged cross-modal features at hidden target rows."""
    columns, source = ([0], [1, 2]) if modality == 'EEG' else ([1, 2], [0])
    hidden = np.isnan(masked[:, columns[0]])
    t = np.flatnonzero(hidden)
    visible = np.flatnonzero(~hidden)
    own = np.column_stack([np.interp(t, visible, masked[visible, col]) for col in columns])
    basic = np.column_stack((own, template[t][:, columns]))
    other = masked if other is None else other
    # Positive lag: preceding EEG for fNIRS; future fNIRS for offline EEG reconstruction.
    sign = 1 if modality == 'EEG' else -1
    lagged = []
    for lag in cfg['linear']['lag_seconds']:
        indices = np.clip(t+sign*round(lag*4), 0, len(masked)-1)
        lagged.append(other[indices][:, source])
    return basic, np.column_stack((basic, *lagged)), hidden, columns
