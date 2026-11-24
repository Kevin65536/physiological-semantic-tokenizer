"""Single-ID carrier quantization with state derived solely from preceding IDs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .anchored_spatial_carrier import AnchoredSpatialCarrier
from .spatial_waveform_dictionary import spectral_features


def next_log_scale(state, decoded_atom_rms, settings, adaptive):
    if not adaptive:
        return np.zeros_like(state)
    increment = np.log(np.maximum(decoded_atom_rms, 1e-12)/settings['reference_atom_RMS'])
    increment = np.clip(increment, -settings['maximum_log_step'], settings['maximum_log_step'])
    return np.clip(state + increment, np.log(settings['minimum_ratio']), np.log(settings['maximum_ratio']))


def quantize_sequences(features, initial_scale, book, settings, adaptive, observed=None):
    """Encode all 15 patches, using the exact decoder state before every decision."""
    values = np.asarray(features, float)
    initial = np.asarray(initial_scale, float)
    if values.ndim != 3 or values.shape[1] != 15 or initial.shape != values.shape[:1] or np.any(initial <= 0):
        raise ValueError('fifteen carrier feature patches and one frozen initial scale per sequence required')
    available = np.ones(len(values), bool) if observed is None else np.asarray(observed, bool)
    codes = np.zeros(values.shape[:2], np.int32)
    gains = np.empty(values.shape[:2])
    state = np.zeros(len(values))
    centers = book['centers']
    norm = np.square(centers).sum(1)
    atom_rms = np.sqrt(np.mean(np.square(book['atoms']), axis=1))
    for patch in range(15):
        scale = initial*np.exp(state)
        gains[:, patch] = scale
        query = values[:, patch]/scale[:, None]
        distance = np.square(query).sum(1)[:, None] + norm - 2*query@centers.T
        chosen = distance.argmin(1) + 1
        chosen[~available] = 0
        codes[:, patch] = chosen
        updated = next_log_scale(state, atom_rms[chosen], settings, adaptive)
        state = np.where(available, updated, state)
    return codes, gains


def decode_sequences(codes, initial_scale, book, settings, adaptive):
    ids = np.asarray(codes)
    initial = np.asarray(initial_scale, float)
    if (ids.ndim != 2 or ids.shape[1] != 15 or ids.dtype.kind not in 'iu'
            or np.any(ids < 0) or np.any(ids >= len(book['atoms']))
            or initial.shape != ids.shape[:1] or np.any(initial <= 0)):
        raise ValueError('in-range integer sequences and frozen initial scales required')
    state = np.zeros(len(ids))
    gains = np.empty(ids.shape)
    output = np.empty((*ids.shape, 400))
    atom_rms = np.sqrt(np.mean(np.square(book['atoms']), axis=1))
    for patch in range(15):
        gains[:, patch] = initial*np.exp(state)
        chosen = ids[:, patch]
        output[:, patch] = book['atoms'][chosen]*gains[:, patch, None]
        updated = next_log_scale(state, atom_rms[chosen], settings, adaptive)
        state = np.where(chosen != 0, updated, state)
    return output, gains


class BackwardAdaptiveCarrier:
    def __init__(self, source, books, initial_scales, settings, metadata):
        self.source = source
        self.books = books
        self.initial_scales = initial_scales
        self.settings = settings
        self.metadata = dict(metadata, positions=source.metadata['positions'])
        self.codebook_size = metadata['codebook_size']
        self.adaptive = metadata['adaptive']
        for dataset, positions in self.metadata['positions'].items():
            for branch in ('S', 'O'):
                value = initial_scales[dataset, branch]
                if np.shape(value) != (len(positions),) or not np.isfinite(value).all() or np.any(value <= 0):
                    raise ValueError('finite positive fixed training scale per native position required')
        for branch in ('S', 'O'):
            for band in range(5):
                book = books[branch, band]
                width = spectral_features(np.zeros((1, 400)), band).shape[1]
                if (book['atoms'].shape != (self.codebook_size, 400)
                        or book['centers'].shape != (self.codebook_size-1, width)
                        or not np.isfinite(book['atoms']).all() or not np.isfinite(book['centers']).all()
                        or np.any(book['atoms'][0] != 0)):
                    raise ValueError('finite fixed carrier codewords and reserved zero required')
                setting = settings[branch, band]
                if not (0 < setting['minimum_ratio'] <= 1 <= setting['maximum_ratio']
                        and setting['reference_atom_RMS'] > 0 and setting['maximum_log_step'] > 0):
                    raise ValueError('stable positive bounded state update required')

    def roles(self, dataset, branch=None):
        return self.source.roles(dataset)

    def encode(self, values, dataset, observed):
        physical = self.source.continuous_source(values)
        carriers = self.source.analyze(physical, dataset, observed)
        return self.assign(carriers, dataset, observed)

    def assign(self, carriers, dataset, observed, *, return_gains=False):
        roles = self.roles(dataset)
        shape = np.shape(carriers['S'])
        if (len(shape) != 4 or shape[1:] != (len(roles), 15, 400)
                or np.shape(carriers['O']) != shape or np.shape(observed) != shape[:2]):
            raise ValueError('complete native-layout carriers required')
        result, all_gains = {}, {}
        for branch in ('S', 'O'):
            ids = np.zeros(shape[:3], np.int32)
            gains = np.empty(shape[:3])
            for band in range(5):
                positions = np.flatnonzero(roles == band)
                if not len(positions):
                    continue
                feature = spectral_features(carriers[branch][:, positions].reshape(-1, 400), band)
                feature = feature.reshape(shape[0]*len(positions), 15, -1)
                initial = np.broadcast_to(self.initial_scales[dataset, branch][positions], (shape[0], len(positions))).reshape(-1)
                code, gain = quantize_sequences(feature, initial, self.books[branch, band], self.settings[branch, band],
                    self.adaptive, np.asarray(observed)[:, positions].reshape(-1))
                ids[:, positions] = code.reshape(shape[0], len(positions), 15)
                gains[:, positions] = gain.reshape(shape[0], len(positions), 15)
            result[branch], all_gains[branch] = ids, gains
        return (result, all_gains) if return_gains else result

    def decode_carriers(self, ids, dataset, *, return_gains=False):
        if set(ids) != {'S', 'O'}:
            raise ValueError('both branch integer sequences required')
        roles = self.roles(dataset)
        result, all_gains = {}, {}
        shape = np.shape(ids['S'])
        if len(shape) != 3 or shape[1:] != (len(roles), 15) or np.shape(ids['O']) != shape:
            raise ValueError('complete matching native integer arrays required')
        for branch in ('S', 'O'):
            values = np.empty((*shape, 400))
            gains = np.empty(shape)
            for band in range(5):
                positions = np.flatnonzero(roles == band)
                if not len(positions):
                    continue
                initial = np.broadcast_to(self.initial_scales[dataset, branch][positions], (shape[0], len(positions))).reshape(-1)
                decoded, gain = decode_sequences(np.asarray(ids[branch])[:, positions].reshape(-1, 15), initial,
                    self.books[branch, band], self.settings[branch, band], self.adaptive)
                values[:, positions] = decoded.reshape(shape[0], len(positions), 15, 400)
                gains[:, positions] = gain.reshape(shape[0], len(positions), 15)
            result[branch], all_gains[branch] = values, gains
        return (result, all_gains) if return_gains else result

    def decode_ids(self, ids, dataset):
        # Every floating value supplied to synthesis is recomputed from IDs and
        # frozen model metadata inside decode_carriers. No per-window state enters.
        return self.source.synthesize(self.decode_carriers(ids, dataset), dataset)

    def save(self, path, source_path):
        path, source_path = Path(path), Path(source_path).resolve()
        metadata = dict(self.metadata, source_file=os.path.relpath(source_path, path.parent),
                        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())
        arrays = dict(metadata=np.asarray(json.dumps(metadata)),
                      settings=np.asarray(json.dumps({f'{b}_{k}': v for (b, k), v in self.settings.items()})))
        for (branch, band), book in self.books.items():
            for key, value in book.items():
                arrays[f'{branch}_{band}__{key}'] = value
        for (dataset, branch), value in self.initial_scales.items():
            arrays[f'initial_{dataset}_{branch}'] = value
        pending = path.with_suffix('.pending.npz')
        np.savez_compressed(pending, **arrays)
        pending.replace(path)


def load_model(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored['metadata']))
        if metadata['schema'] != 'backward_adaptive_carrier_v1':
            raise ValueError('versioned backward-adaptive carrier model required')
        source_path = (path.parent/metadata['source_file']).resolve()
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != metadata['source_sha256']:
            raise ValueError('frozen anchored carrier source changed')
        source = AnchoredSpatialCarrier.load(source_path)
        settings = json.loads(str(stored['settings']))
        settings = {(b, k): settings[f'{b}_{k}'] for b in ('S', 'O') for k in range(5)}
        books = {(b, k): {key: stored[f'{b}_{k}__{key}'].copy() for key in ('atoms', 'centers', 'counts')}
                 for b in ('S', 'O') for k in range(5)}
        initial = {(d, b): stored[f'initial_{d}_{b}'].copy() for d in source.metadata['positions'] for b in ('S', 'O')}
    return BackwardAdaptiveCarrier(source, books, initial, settings, metadata)
