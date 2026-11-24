"""Bounded X-driven geometry diagnosis using unquantized band observations."""

import numpy as np


def band_prediction(fields, role, covariance, band):
    """Conditional mean from exact source observations at the declared native sites."""
    sites = np.flatnonzero(role == band)
    if not len(sites):
        return np.zeros(fields.shape[1:])
    cov = covariance[band]
    operator = np.linalg.solve(cov[np.ix_(sites, sites)], cov[sites]).T
    return (operator @ fields[band, sites].reshape(len(sites), -1)).reshape(fields.shape[1:])


def with_sum(values):
    n = len(values) // 2
    return np.concatenate((values, values[:n] + values[n:]))


def continuous_geometry(fields, role, covariance):
    bands = [band_prediction(fields, role, covariance, b) for b in range(5)]
    return np.asarray(bands), with_sum(np.sum(bands, axis=0))


def select_single_role(fields, role, covariance, target, weight):
    """Choose one visible maximum-error site, then its best whole-context band.

    All fields and targets here are X-only source estimates or actual observed X.
    No supervised held-person target is used for either decision.
    """
    role = np.asarray(role)
    n = len(role) // 2
    if fields.shape != (5, 2*n, 15, 400) or target.shape != (3*n, 15, 400):
        raise ValueError("five bands and one complete native 30-second context required")
    if covariance.shape != (5, 2*n, 2*n) or weight.shape != (3*n,):
        raise ValueError("fixed joint covariance and native training weights required")
    if np.any(weight < 0) or not np.isfinite(weight).all():
        raise ValueError("nonnegative finite frozen weights required")
    pieces, original = continuous_geometry(fields, role, covariance)
    error = np.square(target-original).mean((1, 2)) * weight
    importance = error[:2*n] + np.tile(error[2*n:]/2, 2)
    importance[(role < 0) | (weight[:2*n] <= 0)] = -np.inf
    selected = int(np.argmax(importance))
    before = float(error.sum())
    records = [dict(band=int(role[selected]), objective=before)]
    chosen = role.copy()
    best = original.copy()
    objective = before
    old_band = int(role[selected])
    removed_role = role.copy()
    removed_role[selected] = -1
    removed = band_prediction(fields, removed_role, covariance, old_band)
    reduced = np.sum(pieces, axis=0) - pieces[old_band] + removed
    for band in range(5):
        if band == old_band:
            continue
        trial = role.copy()
        trial[selected] = band
        replacement = band_prediction(fields, trial, covariance, band)
        proposal = with_sum(reduced-pieces[band]+replacement)
        loss = float(weight @ np.square(target-proposal).mean((1, 2)))
        records.append(dict(band=band, objective=loss))
        if loss < objective:
            objective = loss
            best = proposal
            chosen = trial
    return original, best, chosen, dict(
        selected_joint_native=selected,
        selected_branch="SO"[selected//n],
        selected_native=selected % n,
        selected_error=float(importance[selected]),
        old_band=old_band,
        new_band=int(chosen[selected]),
        objective_before=before,
        objective_after=objective,
        trials=records,
    )
