"""A single frequency-tagged EEG atom at each native site and two-second position."""

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .adaptive_band_allocation import synthesize
from .spatial_waveform_dictionary import BANDS

ATOMS = 1023


def nearest(values, centers, block_size=768):
    result = np.empty(len(values), np.int32)
    norms = np.einsum("ij,ij->i", centers, centers)
    for start in range(0, len(values), block_size):
        stop = min(start + block_size, len(values))
        distance = norms - 2 * (values[start:stop] @ centers.T)
        result[start:stop] = np.argmin(distance, axis=1)
    return result


def feature_coordinates(waveforms, features, model, mode):
    projection = ((waveforms - model["mean"]) @ model["basis"].T) / model["wave_scale"]
    if mode == "wave":
        return projection
    if mode != "balanced":
        raise ValueError("unknown assignment rule")
    properties = (features - model["property_mean"]) / model["property_scale"]
    return np.column_stack((projection, properties / np.float32(np.sqrt(3))))


def fit_books(waves, properties, role, seed, atom_count=ATOMS):
    if len(waves) < atom_count or waves.shape[1] != 400 or properties.shape != (len(waves), 3):
        raise ValueError("complete two-second training waves and local attributes required")
    mean = waves.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = waves - mean
    matrix = centered.astype(np.float64)
    eigenvalues, vectors = np.linalg.eigh(matrix.T @ matrix / len(waves))
    width = 4 * (BANDS[role][1] - BANDS[role][0]) + 8
    retain = np.maximum(eigenvalues[-width:], 0).sum()
    model = dict(
        mean=mean,
        basis=vectors[:, -width:].T[::-1].astype(np.float32),
        wave_scale=np.float32(max(np.sqrt(retain), 1e-8)),
        property_mean=properties.mean(0, dtype=np.float64).astype(np.float32),
        property_scale=properties.std(0, dtype=np.float64).clip(1e-7).astype(np.float32),
    )
    model["pca_variance_fraction"] = np.array(
        retain / np.maximum(np.maximum(eigenvalues, 0).sum(), 1e-14)
    )
    energy = np.einsum("ij,ij->i", waves, waves, dtype=np.float64)
    rng = np.random.default_rng(seed)
    cap = np.quantile(energy, 0.99)
    weights = np.minimum(energy, cap)
    weights = 0.5 / len(waves) + 0.5 * weights / max(weights.sum(), 1e-20)
    weights /= weights.sum()
    starting = rng.choice(len(waves), atom_count, replace=False, p=weights)
    for mode in ("wave", "balanced"):
        coords = feature_coordinates(waves, properties, model, mode)
        kmeans = MiniBatchKMeans(
            n_clusters=atom_count,
            init=coords[starting].copy(),
            n_init=1,
            max_iter=12,
            max_no_improvement=6,
            batch_size=1024,
            reassignment_ratio=0,
            random_state=seed,
        )
        kmeans.fit(coords)
        centers = np.ascontiguousarray(kmeans.cluster_centers_, dtype=np.float32)
        ids = nearest(coords, centers)
        count = np.bincount(ids, minlength=atom_count)
        atoms = np.zeros((atom_count, 400))
        np.add.at(atoms, ids, waves)
        atoms /= np.maximum(count, 1)[:, None]
        empty = count == 0
        # Preserve a meaningful waveform for a center absent in the final batch assignment.
        # Its nearest training example supplies a deterministic decode, and zero support is retained.
        if np.any(empty):
            candidates = nearest(centers[empty], coords)
            atoms[empty] = waves[candidates]
        expected_energy = np.bincount(ids, weights=energy, minlength=atom_count) / np.maximum(
            1, count
        )
        atom_energy = np.einsum("ij,ij->i", atoms, atoms)
        gain = np.sqrt(expected_energy / np.maximum(atom_energy, 1e-14)).clip(1.0, 2.0)
        model[mode + "_centers"] = centers
        model[mode + "_atoms"] = atoms.astype(np.float32)
        model[mode + "_gain"] = gain.astype(np.float32)
        model[mode + "_count"] = count
    return model


def load_fold_books(directory, fold, branch):
    """Load the five paid-frequency books fitted together for one outer fold."""
    import json
    from pathlib import Path

    books = []
    for role in range(5):
        with np.load(Path(directory) / f"fold_{fold}_{branch}_{role}.npz", allow_pickle=False) as file:
            metadata = json.loads(str(file["metadata"]))
            if (metadata["fold"], metadata["branch"], metadata["role"]) != (fold, branch, role):
                raise ValueError("wrong folded atom dictionary")
            books.append({key: file[key].copy() for key in file if key != "metadata"})
    return books


def encode_slots(waves, properties, role, model, mode):
    coords = feature_coordinates(waves, properties, model, mode)
    selected = nearest(coords, model[mode + "_centers"])
    return (1 + role * ATOMS + selected).astype(np.uint16)


def paid_coordinates(ids, books, mode, energy_gain=False):
    """Decode the frequency layout from the paid ID itself, including missing zero."""
    values = np.asarray(ids)
    if (
        values.ndim != 2
        or values.shape[1] != 15
        or values.dtype.kind not in "iu"
        or np.any(values < 0)
        or np.any(values > 5 * ATOMS)
    ):
        raise ValueError("valid native EEG integer array required")
    tags = np.where(values == 0, -1, (values.astype(np.int32) - 1) // ATOMS)
    waves = np.zeros((*values.shape, 400), np.float64)
    for role, model in enumerate(books):
        mask = tags == role
        index = (values[mask] - 1) % ATOMS
        atoms = model[mode + "_atoms"][index].astype(np.float64)
        if energy_gain:
            atoms *= model[mode + "_gain"][index, None]
        waves[mask] = atoms
    return waves, tags


def decode_branch(ids, books, covariance, mode, energy_gain=False):
    waves, tags = paid_coordinates(ids, books, mode, energy_gain)
    return synthesize(waves, tags, covariance)


def source_properties(source_wave):
    """Local physiological attributes available at the encoder from the X-only source estimate."""
    from src.analysis.tokenizer_pattern_audit import attributes

    shape = source_wave.shape
    full, fields = attributes(source_wave.reshape(-1, 400), "eeg")
    theta = full[:, fields["power_4_8"][0]]
    alpha = full[:, fields["power_8_13"][0]]
    rms = np.sqrt(np.square(source_wave.reshape(-1, 400)).mean(1))
    return np.log1p(np.column_stack((theta, alpha, rms))).astype(np.float32).reshape(*shape[:-1], 3)


def encode_native(x, observed, dataset, source, books, covariances, weights, layout, mode):
    """Reference encoder for one complete native 30-s array in the frozen training coordinate."""
    from .adaptive_band_allocation import (
        band_waves,
        choose_roles,
        choose_exchanges,
        exchange_costs,
        extract_coordinates,
    )

    expected = (len(source.metadata["positions"][dataset]), 15, 400)
    if x.shape != expected or observed.dtype != bool or observed.shape != (expected[0],):
        raise ValueError("one complete native array and its observation mask required")
    if layout not in ("fixed", "exchange"):
        raise ValueError("unknown band assignment")
    values = source.predict_continuous(x[None], dataset, observed[None])
    result = {}
    for b in "SO":
        waveform = values[b][0]
        fields = band_waves(np.fft.rfft(waveform.reshape(expected[0], 6000)))
        if layout == "fixed":
            role = choose_roles(fields, covariances[b], observed, "fixed")
        else:
            spec = exchange_costs(covariances[b], observed, weights[b] * observed)
            role, _ = choose_exchanges(fields, observed, spec)
        slots = extract_coordinates(fields, role).astype(np.float32)
        prop = source_properties(waveform)
        sent = np.zeros(role.shape, np.uint16)
        for band, book in enumerate(books[b]):
            take = role == band
            if np.any(take):
                sent[take] = encode_slots(slots[take], prop[take], band, book, mode)
        result[b] = sent
    return result


def decode_native(ids, dataset, books, covariances, source, mode, method):
    """Full reconstruction from only the two paid native word arrays and frozen state."""
    if set(ids) != {"S", "O"} or method not in ("mean", "energy", "allocated"):
        raise ValueError("two tagged integer streams and a declared readout required")
    observed = (ids["S"] != 0).any(axis=1)
    if not np.array_equal(observed, (ids["O"] != 0).any(axis=1)):
        raise ValueError("inconsistent missing native site")
    waves = {b: decode_branch(ids[b], books[b], covariances[b], mode, method != "mean") for b in "SO"}
    total = waves["S"] + waves["O"]
    if method == "allocated":
        estimate = source.predict_continuous(total[None], dataset, observed[None])["S"][0]
        waves["S"] = np.where(observed[:, None, None], estimate, waves["S"])
        waves["O"] = total - waves["S"]
    return dict(**waves, sum=total)


RESIDUAL_SLOTS = (("own", 0), ("other", 0), ("own", -1), ("own", 1), ("other", -1), ("other", 1))


def neighbor_words(words, offset):
    """Select a paid word from the same 30-second position, with missing edges."""
    if words.ndim != 2 or words.shape[1] != 15 or offset not in (-1, 0, 1):
        raise ValueError("expected fifteen integers per window and a one-patch offset")
    if offset == 0:
        return words
    shifted = np.zeros_like(words)
    if offset == -1:
        shifted[:, 1:] = words[:, :-1]
    else:
        shifted[:, :-1] = words[:, 1:]
    return shifted


def residual_table(words, errors, weights, size, shrinkage, block=8192):
    """Fit a complete-waveform correction for each observed integer in training."""
    if errors.ndim != 2 or words.shape != weights.shape or len(errors) != len(words):
        raise ValueError("one weighted target per integer required")
    numerator = np.zeros((size, errors.shape[1]), np.float64)
    mass = np.bincount(words, weights=weights, minlength=size)
    for start in range(0, len(words), block):
        key = words[start : start + block]
        order = np.argsort(key, kind="stable")
        code, first = np.unique(key[order], return_index=True)
        weighted = errors[start : start + block][order] * weights[start : start + block][order, None]
        numerator[code] += np.add.reduceat(weighted, first, axis=0)
    result = (numerator / (mass + shrinkage)[:, None]).astype(np.float32)
    result[0] = 0
    return result, mass


def apply_residual_tables(base, words, tables, steps):
    """Read only frozen tables and real paid words for the native waveform correction."""
    if steps not in (1, len(RESIDUAL_SLOTS)):
        raise ValueError("unregistered residual readout")
    prediction = {}
    for branch, other in (("S", "O"), ("O", "S")):
        y = base[branch].copy()
        for i, (source, offset) in enumerate(RESIDUAL_SLOTS[:steps]):
            key = neighbor_words(words[branch if source == "own" else other], offset)
            y += tables[branch][i][key]
        prediction[branch] = y
    prediction["sum"] = prediction["S"] + prediction["O"]
    return prediction


class BandExchangeEEG:
    """One portable frozen outer-fold dictionary; no training or data paths."""

    def __init__(self, metadata, books, covariances, sources, weights):
        self.metadata = metadata
        self.books = books
        self.covariances = covariances
        self.sources = sources
        self.weights = weights
        if (metadata['schema'] != 'band_exchange_eeg_v1'
                or metadata['layout'] != 'exchange' or metadata['assignment'] != 'balanced'
                or metadata['synthesis'] != 'energy' or metadata['codebook_size'] != 5116):
            raise ValueError('frozen exchange-balanced-energy single-integer model required')

    def encode(self, values, dataset, observed):
        x, seen = np.asarray(values), np.asarray(observed)
        count = len(self.metadata['positions'][dataset])
        if (x.ndim != 4 or x.shape[1:] != (count, 15, 400)
                or seen.shape != x.shape[:2] or seen.dtype != bool or not len(x)):
            raise ValueError('complete native 30-second arrays and boolean observations required')
        result = [encode_native(row, mask, dataset, self.sources[dataset], self.books,
                                self.covariances[dataset], self.weights[dataset],
                                'exchange', 'balanced') for row, mask in zip(x, seen)]
        return {b: np.stack([v[b] for v in result]) for b in 'SO'}

    def decode_ids(self, ids, dataset):
        if set(ids) != {'S', 'O'}:
            raise ValueError('both paid native integer streams required')
        values = {b: np.asarray(ids[b]) for b in 'SO'}
        count = len(self.metadata['positions'][dataset])
        if (values['S'].ndim != 3 or values['S'].shape[1:] != (count, 15)
                or values['S'].shape != values['O'].shape or not len(values['S'])):
            raise ValueError('matching native [batch,position,15] sequences required')
        decoded = [decode_native({b: values[b][i] for b in 'SO'}, dataset, self.books,
                                 self.covariances[dataset], self.sources[dataset],
                                 'balanced', 'energy') for i in range(len(values['S']))]
        return {b: np.stack([v[b] for v in decoded]) for b in ('S', 'O', 'sum')}

    def save(self, path):
        import json
        from pathlib import Path
        fields = {'metadata': np.array(json.dumps(self.metadata))}
        required = ('mean', 'basis', 'wave_scale', 'property_mean', 'property_scale',
                    'balanced_centers', 'balanced_atoms', 'balanced_gain')
        for b in 'SO':
            for role, book in enumerate(self.books[b]):
                for name in required: fields[f'book_{b}_{role}_{name}'] = book[name]
        for dataset, source in self.sources.items():
            fields[f'{dataset}_source_metadata'] = np.array(json.dumps(source.metadata))
            for b in 'SO':
                fields[f'{dataset}_transfer_{b}'] = source.transfer[b]
                fields[f'{dataset}_cov_{b}'] = self.covariances[dataset][b]
                fields[f'{dataset}_weights_{b}'] = self.weights[dataset][b]
            for name in ('background', 'beta'):
                fields[f'{dataset}_source_{name}'] = source.models[dataset][name]
        with Path(path).open('xb') as stream: np.savez_compressed(stream, **fields)


def load_model(path):
    """Load only self-contained numerical state and fixed dataset metadata."""
    import json
    from .eeg_transient_gain import BoundedTransientSource
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata']))
        if metadata['schema'] != 'band_exchange_eeg_v1':
            raise ValueError('portable frozen band-exchange model required')
        books = {b: [{k[len(f'book_{b}_{role}_'):]: z[k].copy() for k in z
                      if k.startswith(f'book_{b}_{role}_')} for role in range(5)] for b in 'SO'}
        covariances, sources, weights = {}, {}, {}
        for dataset in metadata['positions']:
            source_metadata = json.loads(str(z[f'{dataset}_source_metadata']))
            if set(source_metadata['positions']) != {dataset}:
                raise ValueError('one native dataset per folded source required')
            sources[dataset] = BoundedTransientSource(
                {b: z[f'{dataset}_transfer_{b}'].copy() for b in 'SO'},
                {dataset: {name: z[f'{dataset}_source_{name}'].copy() for name in ('background','beta')}},
                source_metadata)
            covariances[dataset] = {b: z[f'{dataset}_cov_{b}'].copy() for b in 'SO'}
            weights[dataset] = {b: z[f'{dataset}_weights_{b}'].copy() for b in 'SO'}
    return BandExchangeEEG(metadata, books, covariances, sources, weights)
