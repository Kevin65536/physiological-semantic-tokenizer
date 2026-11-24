"""Select and reconstruct one EEG frequency band per available native position."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .eeg_transient_gain import band_indices


def covariance_from_source(spectrum):
    return np.stack(
        [2 * (spectrum[:, hz] @ spectrum[:, hz].conj().T).real / 6000**2 for hz in band_indices()]
    )


def band_waves(spectrum):
    bands = []
    for hz in band_indices():
        field = np.zeros_like(spectrum)
        field[:, hz] = spectrum[:, hz]
        bands.append(np.fft.irfft(field, n=6000).reshape(len(field), 15, 400))
    return np.stack(bands)


def regular_covariance(cov):
    vals, vec = np.linalg.eigh((cov + cov.T) / 2)
    positive = (vec * np.maximum(vals, 0)) @ vec.T
    scale = np.maximum(np.diag(positive), max(float(np.trace(positive)) / len(cov) * 1e-6, 1e-12))
    return positive + np.diag(0.01 * scale)


def choose_roles(fields, covariances, seen, strategy):
    """The role tag is transmitted. No amplitude, ranking, or fitted per-crop weights are sent."""
    sites = np.flatnonzero(seen)
    roles = np.full((len(seen), 15), -1, np.int8)
    if not len(sites):
        return roles
    slots = sites % 5
    if strategy == "fixed":
        roles[sites] = slots[:, None]
        return roles
    if strategy not in ("energy", "innovation"):
        raise ValueError("unknown layout policy")
    scores = []
    total = np.diagonal(covariances, axis1=1, axis2=2).sum(0)[sites].clip(1e-12)
    for band, covariance in enumerate(covariances):
        v = fields[band, sites]
        if strategy == "innovation":
            precision = np.linalg.inv(covariance[np.ix_(sites, sites)])
            predictor = precision / np.diag(precision)[:, None]
            v = (predictor @ v.reshape(len(sites), -1)).reshape(v.shape)
        scores.append(np.mean(v * v, axis=-1) / total[:, None])
    scores = np.stack(scores, axis=-1)
    for t in range(15):
        row, col = linear_sum_assignment(scores[:, t, :][:, slots], maximize=True)
        roles[sites[row], t] = slots[col]
    return roles


def extract_coordinates(fields, roles):
    values = np.zeros((*roles.shape, 400))
    for b in range(5):
        select = roles == b
        values[select] = fields[b, select]
    return values


def synthesize(coordinates, roles, covariances):
    if coordinates.shape != (*roles.shape, 400) or roles.shape[1] != 15:
        raise ValueError("expected a dated native layout and 15 two-second patches")
    total = np.zeros_like(coordinates)
    for b, cov in enumerate(covariances):
        for t in range(15):
            anchors = np.flatnonzero(roles[:, t] == b)
            if len(anchors):
                inverse_product = np.linalg.solve(cov[np.ix_(anchors, anchors)], cov[anchors])
                total[:, t] += inverse_product.T @ coordinates[anchors, t]
    wave = total.reshape(len(roles), 6000)
    spectrum = np.fft.rfft(wave)
    hz = np.fft.rfftfreq(6000, 1 / 200)
    spectrum[:, (hz < 1) | (hz > 45)] = 0
    return np.fft.irfft(spectrum, n=6000).reshape(total.shape)


def interpolation_error_operator(covariance, anchors, weights):
    """Quadratic form in observed band voltage for the exact spatial interpolation loss."""
    n = len(covariance)
    residual = np.eye(n)
    if len(anchors):
        mapping = np.linalg.solve(covariance[np.ix_(anchors, anchors)], covariance[anchors]).T
        residual[:, anchors] -= mapping
    return residual.T @ (weights[:, None] * residual)


def exchange_costs(covariances, seen, weights):
    """Precompute the change in quadratic cost of each quota-preserving proposed exchange."""
    sites = np.flatnonzero(seen)
    tags = sites % 5
    bands = np.unique(tags)
    specs = []
    for b in bands:
        anchors = sites[tags == b]
        base = interpolation_error_operator(covariances[b], anchors, weights)
        for other in bands[bands > b]:
            partners = sites[tags == other]
            ref = interpolation_error_operator(covariances[other], partners, weights)
            pairs = []
            lhs, rhs = [], []
            for i, site in enumerate(anchors):
                for j, partner in enumerate(partners):
                    a = anchors.copy()
                    a[i] = partner
                    h = partners.copy()
                    h[j] = site
                    pairs.append((site, partner))
                    lhs.append(interpolation_error_operator(covariances[b], a, weights) - base)
                    rhs.append(interpolation_error_operator(covariances[other], h, weights) - ref)
            specs.append(
                (
                    b,
                    other,
                    np.asarray(pairs),
                    np.asarray(lhs).reshape(len(lhs), -1),
                    np.asarray(rhs).reshape(len(rhs), -1),
                )
            )
    return specs


def choose_exchanges(fields, seen, specs):
    """Choose at most two disjoint band-pair exchanges minimizing X-derived source error."""
    time_steps = fields.shape[2]
    roles = np.full((len(seen), time_steps), -1, np.int8)
    roles[seen] = (np.flatnonzero(seen) % 5)[:, None]
    per_band = fields.transpose(0, 2, 1, 3)
    grams = (per_band @ per_band.swapaxes(-2, -1)).reshape(5, time_steps, -1)
    alternatives = []
    for b, other, pairs, left, right in specs:
        costs = left @ grams[b].T + right @ grams[other].T
        indices = costs.argmin(axis=0)
        alternatives.append(
            (int(b), int(other), pairs[indices], costs[indices, np.arange(time_steps)])
        )
    accepted = np.zeros(time_steps)
    for t in range(time_steps):
        best_cost = 0.0
        best = ()
        for i, (b, h, _, cost) in enumerate(alternatives):
            if cost[t] < best_cost:
                best_cost, best = cost[t], (i,)
            for j in range(i):
                c, d, _, second = alternatives[j]
                if b in (c, d) or h in (c, d):
                    continue
                if cost[t] + second[t] < best_cost:
                    best_cost, best = cost[t] + second[t], (i, j)
        for i in best:
            a, b = alternatives[i][2][t]
            roles[a, t], roles[b, t] = roles[b, t], roles[a, t]
        accepted[t] = best_cost
    return roles, accepted
