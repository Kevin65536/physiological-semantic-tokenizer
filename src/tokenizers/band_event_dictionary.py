"""Finite full-waveform event prototypes alongside the original paid EEG words."""

from __future__ import annotations
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .band_exchange_words import nearest
from .spatial_waveform_dictionary import BANDS

ORIGINAL_SIZE = 5116
EVENT_CAPACITY = 2047
VOCABULARY = ORIGINAL_SIZE + 5 * EVENT_CAPACITY


def event_word_ids(role, prototype):
    values = np.asarray(prototype)
    if (
        role not in range(5)
        or values.dtype.kind not in "ui"
        or np.any(values < 0)
        or np.any(values >= EVENT_CAPACITY)
    ):
        raise ValueError("valid paid event prototype required")
    return (ORIGINAL_SIZE + role * EVENT_CAPACITY + values).astype(np.uint16)


def word_roles(words):
    values = np.asarray(words)
    if values.dtype.kind not in "ui" or np.any(values < 0) or np.any(values >= VOCABULARY):
        raise ValueError("valid paid event vocabulary required")
    roles = np.where(values > 0, (values.astype(np.int32) - 1) // 1023, -1)
    events = values >= ORIGINAL_SIZE
    roles[events] = (values[events].astype(np.int32) - ORIGINAL_SIZE) // EVENT_CAPACITY
    return roles, events


def weighted_quantile(values, weights, fractions):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return np.interp(np.asarray(fractions) * cumulative[-1], cumulative, values[order])


def fit_basis(waves, weights, width):
    weight = np.asarray(weights, float)
    center = np.average(waves, axis=0, weights=weight).astype(np.float32)
    residual = waves.astype(float) - center
    covariance = (residual.T * weight) @ residual / weight.sum()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, -width:].T[::-1].astype(np.float32)
    return (
        center,
        basis,
        float(
            np.maximum(eigenvalues[-width:], 0).sum() / np.maximum(eigenvalues.clip(0).sum(), 1e-14)
        ),
    )


def fit_clusters(waves, weights, coordinates, k, seed):
    rng = np.random.default_rng(seed)
    energy = np.square(waves.astype(float)).sum(1)
    probability = np.asarray(weights, float)
    capped = np.minimum(energy, np.quantile(energy, 0.99))
    weighted_energy = probability * capped
    probability = 0.5 * probability / probability.sum() + 0.5 * weighted_energy / max(
        weighted_energy.sum(), 1e-14
    )
    probability /= probability.sum()
    initial = rng.choice(len(waves), k, replace=False, p=probability)
    model = MiniBatchKMeans(
        n_clusters=k,
        init=coordinates[initial].copy(),
        n_init=1,
        max_iter=30,
        max_no_improvement=8,
        batch_size=2048,
        reassignment_ratio=0,
        random_state=seed,
    )
    model.fit(coordinates, sample_weight=np.asarray(weights, np.float32))
    centers = model.cluster_centers_.astype(np.float32)
    assigned = nearest(coordinates, centers)
    mass = np.bincount(assigned, weights=weights, minlength=k)
    count = np.bincount(assigned, minlength=k)
    total = np.zeros((k, 400))
    np.add.at(total, assigned, waves * weights[:, None])
    atoms = total / np.maximum(mass, 1e-20)[:, None]
    empty = mass == 0
    if empty.any():
        atoms[empty] = waves[nearest(centers[empty], coordinates)]
    norm = np.bincount(assigned, weights=weights * energy, minlength=k) / np.maximum(mass, 1e-20)
    gain = np.sqrt(norm / np.square(atoms).sum(1).clip(1e-14)).clip(1, 2)
    return centers, atoms.astype(np.float32), gain.astype(np.float32), count


def fit_event_dictionary(waves, weights, role, seed, capacity=EVENT_CAPACITY):
    if waves.ndim != 2 or waves.shape[1] != 400 or len(waves) != len(weights) or len(waves) < 1:
        raise ValueError("complete weighted fit-event waveforms required")
    waves = np.asarray(waves, np.float32)
    weights = np.asarray(weights, float)
    if not np.isfinite(waves).all() or not np.all(np.isfinite(weights) & (weights > 0)):
        raise ValueError("finite training event waves and positive weights required")
    weights = weights / weights.mean()
    count = max(1, min(capacity, len(waves) // 8))
    width = min(4 * (BANDS[role][1] - BANDS[role][0]) + 8, 76, len(waves))
    mean, basis, fraction = fit_basis(waves, weights, width)
    coordinates = np.ascontiguousarray((waves - mean) @ basis.T)
    center, atoms, gain, n = fit_clusters(waves, weights, coordinates, count, seed)
    output = dict(
        wave_mean=mean,
        wave_basis=basis,
        wave_centers=center,
        wave_atoms=atoms,
        wave_gain=gain,
        wave_count=n,
        wave_PCA_fraction=np.array(fraction),
        fit_events=np.array(len(waves)),
    )
    rms = np.sqrt(np.square(waves.astype(float)).mean(1))
    edges = weighted_quantile(rms, weights, [0.25, 0.5, 0.75])
    tier = np.searchsorted(edges, rms, side="right")
    output["shape_edges"] = edges
    output["shape_tier"] = []
    shape_atoms = []
    shape_gain = []
    shape_count = []
    assigned_base = 0
    for level in range(4):
        selected = tier == level
        if not selected.any():
            output[f"shape_{level}_count"] = np.array(0)
            continue
        local = waves[selected]
        normalized = (local / rms[selected, None].clip(1e-12)).astype(np.float32)
        local_weight = weights[selected]
        local_weight /= local_weight.mean()
        local_count = max(1, min(capacity // 4 + (level < capacity % 4), len(local) // 8))
        cm, bm, retained = fit_basis(normalized, local_weight, width)
        features = np.ascontiguousarray((normalized - cm) @ bm.T)
        cc, aa, gg, nn = fit_clusters(local, local_weight, features, local_count, seed + 10 + level)
        output.update(
            {
                f"shape_{level}_mean": cm,
                f"shape_{level}_basis": bm,
                f"shape_{level}_centers": cc,
                f"shape_{level}_offset": np.array(assigned_base),
                f"shape_{level}_count": np.array(local_count),
                f"shape_{level}_PCA_fraction": np.array(retained),
            }
        )
        assigned_base += local_count
        shape_atoms.append(aa)
        shape_gain.append(gg)
        shape_count.append(nn)
    output.pop("shape_tier")
    output["shape_atoms"] = np.concatenate(shape_atoms)
    output["shape_gain"] = np.concatenate(shape_gain)
    output["shape_count"] = np.concatenate(shape_count)
    return output


def encode_events(waves, model, mode):
    wave = np.asarray(waves, np.float32)
    if mode == "wave":
        return nearest(
            np.ascontiguousarray((wave - model["wave_mean"]) @ model["wave_basis"].T),
            model["wave_centers"],
        )
    if mode != "shape":
        raise ValueError("unknown event codebook")
    rms = np.sqrt(np.square(wave.astype(float)).mean(1))
    tiers = np.searchsorted(model["shape_edges"], rms, side="right")
    result = np.zeros(len(wave), np.int32)
    available = [i for i in range(4) if int(model[f"shape_{i}_count"]) > 0]
    for level in range(4):
        selected = tiers == level
        if not selected.any():
            continue
        actual = level if level in available else min(available, key=lambda i: abs(i - level))
        normalized = (wave[selected] / rms[selected, None].clip(1e-12)).astype(np.float32)
        coords = np.ascontiguousarray(
            (normalized - model[f"shape_{actual}_mean"]) @ model[f"shape_{actual}_basis"].T
        )
        result[selected] = nearest(coords, model[f"shape_{actual}_centers"]) + int(
            model[f"shape_{actual}_offset"]
        )
    return result


def decode_event_coordinates(words, original_atoms, event_models, mode, energy):
    """The event branch uses only the transmitted new word and its frozen waveform."""
    values = np.asarray(words)
    roles, events = word_roles(values)
    result = np.zeros((*values.shape, 400))
    for role in range(5):
        base = (roles == role) & ~events
        old_atoms, old_gain = original_atoms[role]
        old = (values[base] - 1) % 1023
        result[base] = old_atoms[old] * old_gain[old, None]
        selected = (roles == role) & events
        idx = (values[selected].astype(np.int32) - ORIGINAL_SIZE) % EVENT_CAPACITY
        model = event_models[role]
        available = idx < len(model[mode + "_atoms"])
        wave = np.zeros((len(idx), 400))
        wave[available] = model[mode + "_atoms"][idx[available]].astype(float)
        if energy:
            wave[available] *= model[mode + "_gain"][idx[available], None]
        result[selected] = wave
    return roles, result


def propagation_scores(continuous, atomic, roles, covariances, weights):
    """Score each isolated anchor correction in weighted native voltage coordinates."""
    if continuous.shape != atomic.shape or continuous.shape != (*roles.shape, 400):
        raise ValueError("matched two-second band voltages and roles required")
    scores = np.zeros(roles.shape)
    for band, covariance in enumerate(covariances):
        for t in range(roles.shape[1]):
            anchors = np.flatnonzero(roles[:, t] == band)
            if not len(anchors):
                continue
            columns = np.linalg.solve(covariance[np.ix_(anchors, anchors)], covariance[anchors])
            norm = np.square(columns) @ weights
            error = np.square(continuous[anchors, t] - atomic[anchors, t]).sum(axis=-1)
            scores[anchors, t] = norm * error
    return scores


def select_slots(scores, roles, fraction):
    if not 0 <= fraction <= 1 or scores.shape != roles.shape:
        raise ValueError("registered fraction and matching role slots required")
    available = np.flatnonzero(roles.ravel() >= 0)
    if not np.isfinite(scores.ravel()[available]).all():
        raise ValueError("nonfinite encoder correction score")
    selected = np.zeros(roles.shape, bool)
    count = int(np.floor(fraction * len(available)))
    ordered = available[np.argsort(-scores.ravel()[available], kind="stable")[:count]]
    selected.ravel()[ordered] = True
    return selected


def fit_residual_stages(residual, weights, basis, seed, stages=3, capacity=255):
    """Fit extra full-waveform words, retaining a literal zero at every stage."""
    current = np.asarray(residual, np.float32).copy()
    weight = np.asarray(weights, float)
    weight /= weight.mean()
    result = {}
    k = max(1, min(capacity, len(current) // 8))
    for layer in range(1, stages + 1):
        coords = np.ascontiguousarray(current @ basis.T)
        _, atoms, _, _ = fit_clusters(current, weight, coords, k, seed + layer)
        atoms = np.concatenate((np.zeros((1, 400), np.float32), atoms))
        ids = nearest(current, atoms)
        result[f"layer_{layer}_atoms"] = atoms
        result[f"layer_{layer}_count"] = np.bincount(ids, minlength=k + 1)
        before = np.average(np.square(current.astype(float)).sum(1), weights=weight)
        current -= atoms[ids]
        after = np.average(np.square(current.astype(float)).sum(1), weights=weight)
        if after > before + 1e-5 * max(before, 1):
            raise ValueError("zero residual word failed its nonincreasing source error control")
        result[f"layer_{layer}_fit_residual_energy"] = np.array([before, after])
    return result


def encode_residual_stages(residual, model, stages=3):
    current = np.asarray(residual, np.float32).copy()
    output = []
    for layer in range(1, stages + 1):
        atoms = model[f"layer_{layer}_atoms"]
        ids = nearest(current, atoms)
        output.append(ids.astype(np.uint8))
        current -= atoms[ids]
    return np.stack(output, axis=-1)


def add_residual_words(coordinates, primary, residual_codes, models, layers):
    """Apply only the residual words present after each event's primary integer."""
    roles, events = word_roles(primary)
    if (
        layers not in (1, 2, 4)
        or residual_codes.shape != (int(events.sum()), 3)
        or residual_codes.dtype.kind not in "ui"
    ):
        raise ValueError("declared event tuple and a fixed readout layer count required")
    result = coordinates.copy()
    event_roles = roles[events]
    selected = result[events]
    for role, model in enumerate(models):
        match = event_roles == role
        for layer in range(1, layers):
            ids = residual_codes[match, layer - 1]
            atoms = model[f"layer_{layer}_atoms"]
            if len(ids) and ids.max() >= len(atoms):
                raise ValueError("unfitted residual word")
            selected[match] += atoms[ids]
    result[events] = selected
    return result
