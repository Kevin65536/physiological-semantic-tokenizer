"""Observation-coordinate, event and hardware diagnostics without source labels.

Array-only computations. The caller owns sample identity, training partitions,
native clocks and masks. A numerical identification set is not a posterior.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .shared_driver_attribution import hb_diagnostic_coordinates


def lagged(values, shifts):
    """Non-wrapping lags; positive shifts mean past inputs. Missing stays NaN."""
    x = np.asarray(values, float)
    if x.ndim != 2:
        raise ValueError('expected [time, channel]')
    output = []
    for shift in shifts:
        if int(shift) != shift or abs(shift) >= len(x):
            raise ValueError('integer lag must be shorter than the window')
        shift = int(shift)
        y = np.full_like(x, np.nan)
        if shift > 0:
            y[shift:] = x[:-shift]
        elif shift < 0:
            y[:shift] = x[-shift:]
        else:
            y[:] = x
        output.append(y)
    return np.concatenate(output, axis=1)


def event_impulses(eog, training_scale, *, rate, height=3., refractory=.5):
    """VEOG excursions and HEOG slope excursions, not annotated blinks/saccades.

    Inputs have columns [HEOG, VEOG]. Scales are fitted on training records for
    [VEOG, HEOG derivative per second]. Both signs are retained as metadata.
    """
    x, scale = np.asarray(eog, float), np.asarray(training_scale, float)
    if x.ndim != 2 or x.shape[1] != 2 or scale.shape != (2,):
        raise ValueError('EOG [T,2] and two frozen training scales required')
    if not np.isfinite(x).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('complete EOG support and positive training scales required')
    traces = np.column_stack((x[:, 1], np.gradient(x[:, 0])*rate)) / scale
    impulses = np.zeros_like(traces)
    rows = []
    for j in range(2):
        peaks, _ = find_peaks(abs(traces[:, j]), height=height,
                              distance=max(1, round(refractory*rate)))
        impulses[peaks, j] = abs(traces[peaks, j])
        rows.extend(dict(sample=int(k), kind=j, signed_height=float(traces[k, j])) for k in peaks)
    return impulses, rows


def coordinate_stability(candidates, training_scale, *, threshold=.2, minimum=2):
    """Pointwise range of candidate HbT/HbX, in shared training coordinate SD.

    HbO/HbR are transformed BEFORE scaling, preserving their amplitude ratio.
    A single candidate is insufficient evidence of stability.
    """
    values = np.asarray(candidates, float)
    scale = np.asarray(training_scale, float)
    if (values.ndim != 3 or values.shape[-1] != 2 or scale.shape != (2,)
            or not np.isfinite(values).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0) or len(values) < 1):
        raise ValueError('finite candidates [K,T,2] and positive coordinate scales required')
    coords = hb_diagnostic_coordinates(values)
    lower, upper = coords.min(axis=0), coords.max(axis=0)
    width = (upper-lower)/scale
    rms_width = np.sqrt(np.mean(width**2, axis=0))
    enough = len(values) >= minimum
    stable = (width <= threshold) & enough
    amplitude = np.sqrt(np.mean(coords**2, axis=1))/scale
    peaks = np.argmax(abs(coords), axis=1)*.25
    norms = np.linalg.norm(coords, axis=1)
    cosine = np.divide(np.sum(coords*coords[0], axis=1), norms*norms[0],
                       out=np.full_like(norms, np.nan), where=norms*norms[0] > 1e-12)
    return dict(lower=lower, upper=upper, rms_range_SD=rms_width,
                pointwise_stable=stable, stable_fraction=stable.mean(axis=0),
                stable_coordinate=(rms_width <= threshold) & enough,
                amplitude_min_SD=amplitude.min(axis=0), amplitude_max_SD=amplitude.max(axis=0),
                peak_time_range_s=peaks.max(axis=0)-peaks.min(axis=0), shape_cosines=cosine,
                candidate_count=len(values), sufficient_candidates=enough)


def hardware_donors(rows, target):
    """One donor per arm, selected using geometry alone with deterministic ties.

    Distances are template-coordinate distances; no millimetre/depth claim.
    Unknown optodes never count as a shared optode or a known disjoint optode.
    """
    def supported(row):
        return (row.get('source_index') is not None and row.get('detector_index') is not None
                and all(row.get(k) is not None and np.isfinite(row[k]) for k in ('x', 'y', 'z')))
    t = rows[target]
    if not supported(t):
        return dict(available=False, reason='missing_optode_or_position')
    pos = np.array([t[k] for k in ('x', 'y', 'z')])
    shared, disjoint = [], []
    for i, r in enumerate(rows):
        if i == target or not supported(r):
            continue
        d = float(np.linalg.norm(np.array([r[k] for k in ('x', 'y', 'z')])-pos))
        item = (d, i)
        if r['source_index'] == t['source_index'] or r['detector_index'] == t['detector_index']:
            shared.append(item)
        else:
            disjoint.append(item)
    if not shared or len(disjoint) < 2:
        return dict(available=False, reason='insufficient_disjoint_or_shared_donors')
    sd, si = min(shared)
    nd, ni = min(disjoint, key=lambda item: (abs(item[0]-sd), item[1]))
    fd, fi = max((v for v in disjoint if v[1] != ni), key=lambda item: (item[0], -item[1]))
    return dict(available=True, indices=dict(shared_optode=si, disjoint_distance_matched=ni,
                disjoint_far=fi), distances=dict(shared_optode=sd,
                disjoint_distance_matched=nd, disjoint_far=fd), distance_mismatch=nd-sd)


def block_interval(values, *, seed=20261009, repeats=2000):
    """Mean and percentile interval; caller has already reduced to independent units."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(mean=np.nan, low=np.nan, high=np.nan, n=0)
    rng = np.random.default_rng(seed)
    estimates = x[rng.integers(len(x), size=(repeats, len(x)))].mean(axis=1)
    lo, hi = np.quantile(estimates, [.025, .975])
    return dict(mean=float(x.mean()), low=float(lo), high=float(hi), n=len(x))


def sign_flip_p(values, *, seed=20261009, repeats=20000):
    """One-sided paired mean > 0; exact for up to 16 independent units."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan
    if len(x) <= 16:
        signs = 2*((np.arange(2**len(x))[:, None] >> np.arange(len(x))) & 1)-1
        return float(np.mean((signs*x).mean(axis=1) >= x.mean()-1e-14))
    rng = np.random.default_rng(seed)
    samples = (rng.choice([-1., 1.], (repeats, len(x)))*x).mean(axis=1)
    return float((1+(samples >= x.mean()-1e-14).sum())/(repeats+1))


def holm(pvalues):
    p = np.asarray(pvalues, float)
    result = np.full_like(p, np.nan)
    indices = np.flatnonzero(np.isfinite(p))
    order = indices[np.argsort(p[indices])]
    result[order] = np.minimum(1., np.maximum.accumulate(p[order]*np.arange(len(order), 0, -1)))
    return result


def candidate_inference(observation, predictions, noise_sd, *, anchors=None,
                        anchor_predictions=None, anchor_sd=None, cutoff=3.841459):
    """Fixed Gaussian-distance candidate comparison, with optional measured anchors.

    No mechanism truth or candidate label enters the score. The fixed LR set
    is a diagnostic identification set; its empirical coverage must be checked.
    """
    y, pred = np.asarray(observation, float), np.asarray(predictions, float)
    sd = np.asarray(noise_sd, float)
    if pred.ndim != y.ndim+1 or pred.shape[1:] != y.shape or not np.isfinite(pred).all():
        raise ValueError('candidate predictions must be [K,*observation.shape]')
    if not np.isfinite(y).all() or not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError('finite observation and positive noise SD required')
    cost = np.sum(((pred-y)/sd)**2, axis=tuple(range(1, pred.ndim)))
    if anchors is not None:
        a, pa, sa = np.asarray(anchors), np.asarray(anchor_predictions), np.asarray(anchor_sd)
        if pa.shape != (len(pred),)+a.shape or not all(np.isfinite(v).all() for v in (a, pa, sa)) or np.any(sa <= 0):
            raise ValueError('finite aligned auxiliary measurements and positive noise required')
        cost += np.sum(((pa-a)/sa)**2, axis=tuple(range(1, pa.ndim)))
    best = int(np.argmin(cost))
    return dict(best=best, cost=cost, admissible=cost-cost[best] <= cutoff)
