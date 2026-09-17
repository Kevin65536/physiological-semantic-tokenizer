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


def noise_floor_evidence(estimate, floor, *, layer, unit, source, family='gaussian', nu=None):
    """Audit the existing noise policy without changing its statistical weight."""
    estimate, floor = np.broadcast_arrays(np.asarray(estimate, dtype=float), np.asarray(floor, dtype=float))
    if not np.isfinite(estimate).all() or not np.isfinite(floor).all() or np.any(estimate < 0) or np.any(floor < 0):
        raise ValueError('noise estimate and floor must be finite and nonnegative')
    if family not in ('gaussian', 'student_t') or (family == 'student_t' and (nu is None or nu <= 2)):
        raise ValueError('noise family and finite-variance Student degrees of freedom required')
    final = np.maximum(estimate, floor)
    sd_factor = np.sqrt(nu/(nu-2)) if family == 'student_t' else 1.
    return dict(estimate_before_floor=estimate.tolist(), floor=floor.tolist(), final=final.tolist(),
                triggered=(estimate < floor).tolist(), trigger_fraction=float(np.mean(estimate < floor)),
                layer=layer, unit=unit, floor_source=source, family=family, student_nu=nu,
                parameter='scale' if family == 'student_t' else 'SD',
                standard_deviation=(final*sd_factor).tolist())


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


def eeg_band_power(values, bands=((8., 13.), (13., 30.), (30., 45.)),
                   sample_rate=200., target_rate=4.):
    """Trial-local power in uV^2; caller verifies units before this boundary.

    The last axis preserves band identity. No baseline, fitted scaling or
    missing-feature interpolation is applied before returning these features.
    """
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(values, dtype=float)
    ratio = int(round(sample_rate / target_rate))
    if (x.ndim != 2 or not np.isfinite(x).all() or ratio < 1 or
            sample_rate / target_rate != ratio or len(x) % ratio):
        raise ValueError('complete trial [native time, channel] and integer clock ratio required')
    powers = []
    for low, high in bands:
        filtered = sosfiltfilt(butter(4, [low, high], btype='bandpass',
                                    fs=sample_rate, output='sos'), x, axis=0)
        powers.append(np.mean(filtered.reshape(-1, ratio, x.shape[1])**2, axis=1))
    return np.stack(powers, axis=-1)


def gaussian_driver_statistic(values, loading, covariance):
    """Exact sufficient statistic for e = loading*r + correlated Gaussian noise.

    Returns the linear weights and variance, retaining signed loadings and all
    off-diagonal noise terms. For a common temporal operator/mask this reduction
    commutes with that operator. Channel-specific masks require a new reduction
    on the visible submatrix; zero-filling channels is not supported.
    """
    x, ell, cov = map(lambda v: np.asarray(v, dtype=float), (values, loading, covariance))
    if (ell.ndim != 1 or x.shape[-1] != len(ell) or cov.shape != (len(ell), len(ell))
            or not all(np.isfinite(v).all() for v in (x, ell, cov))
            or not np.allclose(cov, cov.T, atol=1e-12, rtol=1e-12)):
        raise ValueError('incompatible finite observation/loading/covariance')
    np.linalg.cholesky(cov)
    precision_loading = np.linalg.solve(cov, ell)
    information = float(ell @ precision_loading)
    if information <= 1e-12:
        raise ValueError('no identifiable EEG driver loading')
    weights = precision_loading/information
    return dict(values=x@weights, variance=1/information, weights=weights)


def signed_loading_fit(features, driver, weights=None, *, bands=1,
                       ridge=.01, geometry_penalty=0., gauge=1.):
    """Training-only anchor regression with optional soft geometry per band.

    Both fitting and penalties are per-coordinate averages. The supplied
    driver is a fixed reference proxy, never described as measured neural truth.
    Channel-major/band-minor order is explicit. The geometry amplitude is free
    and signed; the orthogonal deviation is penalized without changing capacity.
    """
    x = np.asarray(features, dtype=float).reshape(-1, features.shape[-1])/gauge
    r = np.asarray(driver, dtype=float).ravel()/gauge
    k = x.shape[1]
    if (len(x) != len(r) or k % bands or not np.isfinite(x).all()
            or not np.isfinite(r).all() or ridge <= 0 or geometry_penalty < 0):
        raise ValueError('invalid loading fit support or regularization')
    penalty = np.zeros((k, k))
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != (k//bands,) or not np.isfinite(w).all() or w@w <= 0:
            raise ValueError('geometry must match actual channel support')
        projector = np.eye(len(w))-np.outer(w, w)/(w@w)
        for b in range(bands):
            ix = np.arange(b, k, bands)
            penalty[np.ix_(ix, ix)] = projector
    elif geometry_penalty:
        raise ValueError('positive geometry penalty requires geometry')
    return np.linalg.solve((np.mean(r*r)+ridge)*np.eye(k)+geometry_penalty*penalty,
                           np.mean(x*r[:, None], axis=0))


def correlated_feature_noise(trials, *, floor, shrinkage=.1):
    """Training difference covariance with robust marginal SD and fixed shrinkage.

    This is a feature-noise approximation; slow physiological variability and
    nonlinear power extraction do not establish sensor noise calibration.
    """
    x = np.asarray(trials, dtype=float)
    if x.ndim != 3 or not np.isfinite(x).all() or not 0 < shrinkage <= 1:
        raise ValueError('finite [trial,time,feature] training array required')
    d = np.diff(x, axis=1).reshape(-1, x.shape[-1])
    estimate = robust_mad(d)/np.sqrt(2.)
    sd = np.maximum(estimate, floor)
    if x.shape[-1] == 1:
        correlation = np.ones((1, 1))
    else:
        z = (d-d.mean(0))/np.maximum(d.std(0), 1e-12)
        correlation = z.T@z/len(z)
        np.fill_diagonal(correlation, 1.)
    correlation = (1-shrinkage)*correlation+shrinkage*np.eye(len(sd))
    return dict(covariance=sd[:, None]*correlation*sd[None, :],
                estimate_before_floor=estimate, floor=floor, final_sd=sd,
                floor_fraction=float(np.mean(estimate < floor)), shrinkage=shrinkage)


def geometry_neighbours(rows, channel_names, anchor_name, maximum=6):
    """Compatible coordinates only; never substitute an anatomical fallback."""
    targets = [r for r in rows if r.get('modality') == 'fnirs' and r.get('channel_name') == anchor_name]
    for target in targets:
        xyz = [target.get(k) for k in ('x', 'y', 'z')]
        if None in xyz or not np.isfinite(xyz).all():
            continue
        found = []
        for i, name in enumerate(channel_names):
            matches = [r for r in rows if r.get('modality') == 'eeg' and r.get('channel_name') == name
                       and r.get('coordinate_system') == target.get('coordinate_system')
                       and r.get('coordinate_units') == target.get('coordinate_units')
                       and all(r.get(k) is not None and np.isfinite(r[k]) for k in ('x', 'y', 'z'))]
            if len(matches) == 1:
                row = matches[0]
                distance = float(np.linalg.norm(np.array([row[k] for k in ('x', 'y', 'z')])-xyz))
                found.append((distance, name, i, row))
        found.sort(key=lambda v: (v[0], v[1]))
        found = found[:maximum]
        if len(found) < 2:
            continue
        distances = np.array([v[0] for v in found])
        width = max(float(np.median(distances)), 1e-8)
        weights = np.exp(-.5*(distances/width)**2)
        weights /= np.linalg.norm(weights)
        return dict(indices=[v[2] for v in found], channels=[v[1] for v in found],
                    distances=distances, weights=weights, kernel_width=width,
                    coordinate_system=target['coordinate_system'], coordinate_units=target['coordinate_units'],
                    source_files=[target.get('source_file')]+[v[3].get('source_file') for v in found],
                    interpretation='dataset montage proximity; not individual cortical co-registration')
    raise ValueError('no compatible EEG/fNIRS geometry; no substitute channels')


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
