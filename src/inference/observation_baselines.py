"""Shared observation transforms, noise estimates and linear SSM baselines.

These array-only helpers use the caller's existing observation contract. Callers
own train/validation membership, masks, candidate selection and data access.
Importing this module neither loads recordings nor changes process settings.
"""

from functools import lru_cache

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


def disjoint_native_bin_operator(native_time, *, steps=360, dt=.25, reference_steps=20):
    """Native sample means with explicit disjoint raw support and a fixed reference.

    Returns raw-to-feature and model-to-feature matrices. Each unreferenced row
    uses only native samples in [k*dt,(k+1)*dt); there is no bidirectional filter
    or interpolation of measured values across a missing interval. The model
    is linearly interpolated to those exact native timestamps before binning.
    The caller keeps the first reference_steps bins visible for every Hb task.
    """
    t = np.asarray(native_time, float)
    if (t.ndim != 1 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)
            or steps < reference_steps or reference_steps < 1 or dt <= 0
            or np.any(t < -1e-9) or np.any(t >= steps*dt)):
        raise ValueError('finite ordered native timestamps within the declared interval required')
    groups = np.floor(np.maximum(t, 0.)/dt+1e-10).astype(int)
    counts = np.bincount(groups, minlength=steps)
    if len(counts) != steps or np.any(counts == 0):
        raise ValueError('every model bin must contain actual native samples')
    bins = np.zeros((steps, len(t)))
    bins[groups, np.arange(len(t))] = 1/counts[groups]
    interp = np.zeros((len(t), steps))
    z = np.minimum(t/dt, steps-1)
    left = z.astype(int); right = np.minimum(left+1, steps-1); weight = z-left
    interp[np.arange(len(t)), left] += 1-weight
    interp[np.arange(len(t)), right] += weight
    reference = np.eye(steps)
    reference[:, :reference_steps] -= 1/reference_steps
    return dict(raw=reference@bins, model=reference@bins@interp, bins=bins,
                eeg=reference, native_counts=counts, native_groups=groups)


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


@lru_cache(maxsize=8)
def native_feature_operators(steps=120):
    from scipy.signal import resample_poly
    from src.data.homer2_preprocessing import bandpass_fnirs, modified_beer_lambert
    from src.data.physiology_measurement_adapter import measurement_baseline
    if steps <= 20 or (steps*5) % 2:
        raise ValueError('feature time window must support the five-second baseline')
    native_steps = steps*5//2
    _, weights, baseline_evidence = measurement_baseline(
        np.zeros((steps,3)),np.arange(steps)/4.-5.,np.ones(steps,dtype=bool),
        interval_s=(-5.,0.),role='pre_event_reference_not_latent_rest',
        evidence='Step5 admitted 30 s window begins 5 s before MA event',minimum_samples=20)
    baseline = np.eye(steps)-np.ones((steps,1))*weights
    filtered, quality = bandpass_fnirs(np.eye(native_steps), sample_rate_hz=10.)
    if quality['status'] != 'applied':
        raise ValueError('native feature filter unavailable at the declared support')
    fnirs = baseline@resample_poly(filtered, 2, 5, axis=0)
    model_clock, native_clock = np.arange(steps)/4., np.arange(native_steps)/10.
    interpolation = np.column_stack([np.interp(native_clock, model_clock, col)
                                     for col in np.eye(steps)])
    optical_basis = np.zeros((16, 1, 2))
    optical_basis[:2, 0] = np.eye(2)
    converted, _ = modified_beer_lambert(optical_basis, wavelengths_nm=(760., 850.))
    mbll = converted[:2, 0, :].T
    return dict(eeg=baseline, fnirs=fnirs, native_interpolation=interpolation,
                model_time=model_clock, fnirs_time=native_clock, mbll=mbll,
                baseline_evidence=baseline_evidence)


@lru_cache(maxsize=8)
def native_model_operator(steps=120, *, block_steps=None):
    """Push model trajectories through the measured feature-time operator.

    block_steps preserves independently processed original windows when the
    latent trajectory is continuous; no additional centering/filtering occurs.
    """
    from scipy.linalg import block_diag
    if block_steps is not None:
        if steps % block_steps:
            raise ValueError('trajectory length must contain complete feature blocks')
        return block_diag(*([native_model_operator(block_steps)]*(steps//block_steps)))
    op=native_feature_operators(steps)
    mean=np.zeros((3*steps,3*steps))
    mean[0::3,0::3]=op['eeg']
    mean[1::3,1::3]=mean[2::3,2::3]=op['fnirs']@op['native_interpolation']
    return mean


@lru_cache(maxsize=96)
def visible_feature_interpolation(length, hidden_left, hidden_right):
    mask = np.ones(length, dtype=bool)
    mask[hidden_left:hidden_right] = False
    selected = np.flatnonzero(mask)
    if not len(selected):
        return np.zeros((length, length))
    if len(selected) < 2:
        raise ValueError('partly visible feature needs two real input samples')
    result = np.zeros((length, length))
    result[:, selected] = np.column_stack(
        [np.interp(np.arange(length), selected, col) for col in np.eye(len(selected))])
    return result


def conditional_optical_hb_mapping(fnirs_factor):
    """Conditional physical-to-retained Hb map, not individual calibration.

    Both Beer--Lambert paths are 3 cm * DPF 6 and cancel. The published
    decadic extinction table uses M^-1 cm^-1; ln(10)*1e-6 converts to natural
    OD per micromolar. P0=71 micromolar and tissue sensitivity one are fixed
    diagnostic assumptions. Negative off-diagonals are retained, not clipped.
    """
    k=float(fnirs_factor)
    if not np.isfinite(k) or k<=0:
        raise ValueError('fnirs_factor must be finite positive')
    old=np.array([[.148,.384],[.252,.179]])
    decadic=np.array([[586.,1548.52],[1058.,691.32]])
    bbar=np.linalg.solve(old,np.log(10.)*1e-6*decadic*71.)
    matrix=k*bbar
    reference=float(1./np.sqrt(np.linalg.det(matrix)))
    return dict(matrix=matrix,base_matrix=bbar,fnirs_factor=k,beta_reference=reference,
                P0_assumed_uM=71.,tissue_sensitivity_assumed=1.,path_length_cm=3.,dpf=6.,
                interpretation='conditional_prediction_map_not_individual_absolute_calibration')


def spatial_observation_design(eeg_positions, hb_positions, modes=1, *,
                               geometry='true', seed=0, kernel_width=None,
                               bands=5):
    """Freeze one or two spatial readouts in the supplied montage coordinates.

    Hb row zero is the declared target. A second anchor, when requested, is
    the farthest supplied Hb donor, with input order breaking distance ties.
    Gaussian EEG columns have unit L2 norm. Hb rows are convex kernel mixtures;
    for one core every Hb loading is one. The default width is the median
    positive sensor-to-anchor distance in these coordinates, not a calibrated
    physiological distance.

    A sham only reassigns the true design's rows to the unchanged signal order.
    It preserves each matrix's singular values, anchors and parameter count.
    The target Hb row remains fixed; two Hb channels have no donor permutation.
    A meaningful sham requires an EEG row assignment that changes its readout.
    """
    eeg, hb = np.asarray(eeg_positions, float), np.asarray(hb_positions, float)
    if (eeg.ndim != 2 or hb.ndim != 2 or not len(eeg) or not len(hb)
            or not eeg.shape[1] or eeg.shape[1] != hb.shape[1]
            or not np.isfinite(eeg).all() or not np.isfinite(hb).all()):
        raise ValueError('finite EEG [C,D] and Hb [J,D] coordinates are required')
    if (isinstance(modes, (bool, np.bool_)) or modes not in (1, 2)
            or isinstance(bands, (bool, np.bool_)) or bands != 5):
        raise ValueError('spatial designs support K=1 or 2 and exactly five EEG bands')
    k = int(modes)
    if geometry not in ('true', 'uniform', 'sham'):
        raise ValueError('geometry must be true, uniform or sham')
    if geometry == 'uniform' and k != 1:
        raise ValueError('uniform geometry is defined only for K=1')
    anchors = [0]
    if k == 2:
        if len(hb) < 2:
            raise ValueError('K=2 requires a distinct Hb donor anchor')
        distances = np.linalg.norm(hb[1:]-hb[0], axis=1)
        farthest = int(np.argmax(distances))+1
        if not np.isfinite(distances).all() or distances[farthest-1] <= 0:
            raise ValueError('K=2 requires spatially distinct Hb anchors')
        anchors.append(farthest)
    anchor_positions = hb[anchors]
    eeg_distance = np.linalg.norm(eeg[:, None, :]-anchor_positions, axis=2)
    hb_distance = np.linalg.norm(hb[:, None, :]-anchor_positions, axis=2)
    if not np.isfinite(eeg_distance).all() or not np.isfinite(hb_distance).all():
        raise ValueError('coordinate distances must be finite')
    if kernel_width is None:
        distances = np.r_[eeg_distance.ravel(), hb_distance.ravel()]
        positive = distances[distances > 0]
        if not len(positive):
            raise ValueError('kernel width needs a positive montage distance')
        width = float(np.median(positive))
    else:
        width = float(kernel_width)
    if not np.isfinite(width) or width <= 0:
        raise ValueError('kernel_width must be positive finite')
    if geometry == 'uniform':
        spatial_eeg = np.ones((len(eeg), 1))/np.sqrt(len(eeg))
    else:
        # Subtracting a column constant preserves its normalized Gaussian
        # kernel while avoiding underflow at a distant montage origin.
        log_eeg = -.5*(eeg_distance/width)**2
        spatial_eeg = np.exp(np.maximum(log_eeg-log_eeg.max(axis=0), -700.))
        spatial_eeg /= np.linalg.norm(spatial_eeg, axis=0)
    if k == 1:
        hb_loadings = np.ones((len(hb), 1))
    else:
        log_hb = -.5*(hb_distance/width)**2
        hb_loadings = np.exp(np.maximum(log_hb-log_hb.max(axis=1, keepdims=True), -700.))
        hb_loadings /= hb_loadings.sum(axis=1, keepdims=True)
    eeg_rank = int(np.linalg.matrix_rank(spatial_eeg))
    hb_rank = int(np.linalg.matrix_rank(hb_loadings))
    if eeg_rank != k or hb_rank != k:
        raise ValueError('spatial EEG and Hb readouts must both have rank K')
    eeg_permutation, hb_permutation = np.arange(len(eeg)), np.arange(len(hb))
    eeg_changed = hb_changed = False
    if geometry == 'sham':
        rng = np.random.default_rng(seed)
        eeg_permutation = rng.permutation(len(eeg))
        if np.array_equal(spatial_eeg[eeg_permutation], spatial_eeg):
            different = np.flatnonzero(np.any(spatial_eeg != spatial_eeg[0], axis=1))
            if not len(different):
                raise ValueError('sham needs a nontrivial EEG geometry-to-signal assignment')
            eeg_permutation = np.arange(len(eeg))
            other = int(different[0])
            eeg_permutation[[0, other]] = eeg_permutation[[other, 0]]
        if len(hb) > 2:
            hb_permutation[1:] = rng.permutation(np.arange(1, len(hb)))
            if np.array_equal(hb_permutation, np.arange(len(hb))):
                hb_permutation[1:] = np.roll(hb_permutation[1:], 1)
        eeg_changed = not np.array_equal(spatial_eeg[eeg_permutation], spatial_eeg)
        hb_changed = not np.array_equal(hb_loadings[hb_permutation], hb_loadings)
        spatial_eeg = spatial_eeg[eeg_permutation]
        hb_loadings = hb_loadings[hb_permutation]
    return dict(spatial_eeg=spatial_eeg, hb_loadings=hb_loadings,
                anchors=anchors, anchor_positions=anchor_positions.copy(), K=k,
                ranks=dict(eeg=eeg_rank, hb=hb_rank), bands=int(bands),
                permutation=dict(eeg=eeg_permutation, hb=hb_permutation),
                eeg_changed=eeg_changed, hb_changed=hb_changed,
                geometry=geometry, seed=seed, kernel_width=width,
                learned_band_coefficients=int(bands)*k,
                interpretation='supplied_montage_coordinates_not_individual_cortical_registration')


def fit_spatial_coordinates(chunks, training_ids=None, *, target_rms=.025):
    """Fit frozen observation coordinates from explicitly supplied train pairs.

    Each pair is already referenced EEG [T,C,5], Hb [T,J,2]. Each EEG band's
    RMS is pooled over training time and channels, then shared by all channels
    so fixed spatial amplitudes remain intact. A single RMS across Hb channels
    and both species preserves HbO/HbR ratios and the HbO+HbR identity. No
    record is centered or independently normalized. Callers own the training
    membership; this API never receives evaluation arrays.
    """
    target_rms = float(target_rms)
    if not np.isfinite(target_rms) or target_rms <= 0:
        raise ValueError('target_rms must be positive finite')
    eeg_chunks, hb_chunks = [], []
    shape = None
    for eeg, hb in chunks:
        eeg, hb = np.asarray(eeg, float), np.asarray(hb, float)
        if (eeg.ndim != 3 or eeg.shape[2] != 5 or not eeg.shape[1]
                or hb.ndim != 3 or hb.shape[2] != 2 or not hb.shape[1]
                or not len(eeg) or len(eeg) != len(hb)
                or not np.isfinite(eeg).all() or not np.isfinite(hb).all()):
            raise ValueError('finite training EEG [T,C,5] and Hb [T,J,2] pairs are required')
        current = (eeg.shape[1:], hb.shape[1:])
        if shape is not None and shape != current:
            raise ValueError('training chunks must preserve channel and band identity')
        shape = current
        eeg_chunks.append(eeg)
        hb_chunks.append(hb)
    if not eeg_chunks:
        raise ValueError('at least one training pair is required')
    ids = None if training_ids is None else list(training_ids)
    if ids is not None and len(ids) != len(eeg_chunks):
        raise ValueError('training_ids must identify exactly the supplied training chunks')
    eeg, hb = np.concatenate(eeg_chunks), np.concatenate(hb_chunks)
    if len(eeg) < 2:
        raise ValueError('at least two training time rows are required')
    with np.errstate(over='ignore', invalid='ignore'):
        band_scale = np.sqrt(np.mean(eeg**2, axis=(0, 1)))/target_rms
        eeg_scale = np.broadcast_to(band_scale, eeg.shape[1:]).copy()
        hb_scale = float(np.sqrt(np.mean(hb**2))/target_rms)
    if (not np.isfinite(eeg_scale).all() or np.any(eeg_scale <= 0)
            or not np.isfinite(hb_scale) or hb_scale <= 0):
        raise ValueError('training RMS must provide positive finite observation scales')
    scaled = np.column_stack(((eeg/eeg_scale).reshape(len(eeg), -1),
                              (hb/hb_scale).reshape(len(hb), -1)))
    sd_before_floor = scaled.std(axis=0)
    sd_floor = target_rms*1e-6
    return dict(eeg_scale=eeg_scale, hb_scale=hb_scale,
                sd=np.maximum(sd_before_floor, sd_floor),
                sd_before_floor=sd_before_floor, sd_floor=sd_floor,
                sd_floor_applied=sd_before_floor < sd_floor,
                training_count=len(eeg_chunks), training_rows=len(eeg),
                training_ids=ids, target_rms=target_rms,
                centering='none_already_referenced_inputs',
                eeg_scale_definition='per_band_training_RMS_pooled_over_time_and_channels_divided_by_target_rms',
                eeg_shape=shape[0], hb_shape=shape[1])


def transform_spatial_coordinates(eeg, hb, coordinate):
    """Divide by frozen training scales; retain hidden NaNs without time mixing.

    Returns EEG [T,C*5] in channel-major/band-minor order and Hb [T,J,2].
    EEG may enter as either [T,C,5] or the same declared flattened ordering.
    The elementwise operation cannot let a hidden target alter visible values.
    """
    eeg, hb = np.asarray(eeg, float), np.asarray(hb, float)
    scale = np.asarray(coordinate['eeg_scale'], float)
    hb_scale = float(coordinate['hb_scale'])
    if (scale.ndim != 2 or scale.shape[1] != 5 or not len(scale)
            or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not np.isfinite(hb_scale) or hb_scale <= 0):
        raise ValueError('frozen spatial observation scales must be positive finite')
    if eeg.ndim == 2 and eeg.shape[1] == scale.size:
        eeg = eeg.reshape(len(eeg), *scale.shape)
    if (eeg.ndim != 3 or eeg.shape[1:] != scale.shape
            or hb.ndim != 3 or hb.shape[2] != 2 or len(hb) != len(eeg)
            or tuple(hb.shape[1:]) != tuple(coordinate['hb_shape'])
            or np.isinf(eeg).any() or np.isinf(hb).any()):
        raise ValueError('observation shape/order must match the frozen training coordinates')
    return (eeg/scale).reshape(len(eeg), -1), hb/hb_scale


def fit_spatial_eeg_loadings(eeg_chunks, driver_chunks, spatial_eeg, *,
                             ridge=.1, training_ids=None):
    """Learn exactly five signed band coefficients per fixed spatial core.

    For each band the pooled training equations are
    E[t,c,b] = sum_k spatial_eeg[c,k]*a[b,k]*driver[t,k].
    Each band's Gram is averaged across training time/channel rows. The ridge
    penalty is ridge*trace(Gram)/K, so ridge is dimensionless and changing the
    supplied driver units changes loadings inversely while preserving predicted
    EEG. No channel coefficient, intercept, loading renormalization or
    driver/gain change can undo geometry. The caller's fixed driver gauge is
    retained, including its sign and scale.
    """
    spatial = np.asarray(spatial_eeg, float)
    ridge = float(ridge)
    if (spatial.ndim != 2 or not len(spatial) or spatial.shape[1] not in (1, 2)
            or not np.isfinite(spatial).all()
            or np.linalg.matrix_rank(spatial) != spatial.shape[1]
            or not np.isfinite(ridge) or ridge < 0):
        raise ValueError('finite full-rank [C,K] geometry and nonnegative ridge are required')
    c, k = spatial.shape
    eeg_values, drivers = list(eeg_chunks), list(driver_chunks)
    if not eeg_values or len(eeg_values) != len(drivers):
        raise ValueError('paired training EEG and driver chunks are required')
    ids = None if training_ids is None else list(training_ids)
    if ids is not None and len(ids) != len(eeg_values):
        raise ValueError('training_ids must identify exactly the supplied training chunks')
    designs, targets = [], []
    for eeg, driver in zip(eeg_values, drivers):
        eeg, driver = np.asarray(eeg, float), np.asarray(driver, float)
        if eeg.ndim == 2 and eeg.shape[1] == c*5:
            eeg = eeg.reshape(len(eeg), c, 5)
        if (eeg.ndim != 3 or eeg.shape[1:] != (c, 5) or not len(eeg)
                or driver.shape != (len(eeg), k)
                or not np.isfinite(eeg).all() or not np.isfinite(driver).all()):
            raise ValueError('finite paired training EEG [T,C,5] and driver [T,K] are required')
        designs.append((driver[:, None, :]*spatial[None, :, :]).reshape(-1, k))
        targets.append(eeg.reshape(-1, 5))
    design, target = np.concatenate(designs), np.concatenate(targets)
    design_rank = int(np.linalg.matrix_rank(design))
    if design_rank != k:
        raise ValueError('training driver/geometry equations are rank deficient')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        gram = design.T@design/len(design)
    gram_trace = float(np.trace(gram))
    if not np.isfinite(gram_trace) or gram_trace <= np.finfo(float).tiny:
        raise ValueError('EEG loading calibration unavailable: training Gram trace is too small or nonfinite')
    # Every band has the same complete channel/time support and driver design.
    # Keep the five effective penalties explicit for the retained fit contract.
    effective_ridge = np.full(5, ridge*gram_trace/k)
    coefficients = np.linalg.solve(gram+effective_ridge[0]*np.eye(k),
                                   design.T@target/len(design)).T
    loading = (spatial[:, None, :]*coefficients[None, :, :]).reshape(c*5, k)
    norm = np.linalg.norm(loading, axis=0)
    if not np.isfinite(loading).all() or np.any(norm <= 1e-10):
        raise ValueError('fitted EEG loading has a near-zero core column')
    singular = np.linalg.svd(loading, compute_uv=False)
    rank = int(np.linalg.matrix_rank(loading))
    if rank != k or singular[-1] <= 1e-8*singular[0]:
        raise ValueError('fitted EEG loading is rank deficient or nearly singular')
    return dict(eeg_loadings=loading, band_coefficients=coefficients,
                rank=rank, condition=float(singular[0]/singular[-1]),
                design_rank=design_rank, design_condition=float(np.linalg.cond(design)),
                training_ids=ids, training_count=len(eeg_values),
                training_rows=sum(len(value) for value in eeg_values),
                learned_coefficients=5*k, ridge=ridge,
                effective_ridge=effective_ridge, gram_trace=np.full(5, gram_trace),
                ridge_definition='penalty_b = ridge * trace(X_b.T @ X_b / n_rows) / K',
                driver_gauge='unchanged_fixed_reference')
