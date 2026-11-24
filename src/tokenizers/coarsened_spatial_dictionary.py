"""Training-fitted many-to-one remapping of an existing atomic EEG vocabulary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .spatial_analysis_synthesis import load_encoder
from .spatial_local_gain import load_decoder


def supported_remap(atoms, count, subjects, means, capacity, semantic_fraction):
    """Map every fine ID to a supported, frequency-ranked representative atom."""
    if not 0 <= semantic_fraction < 1 or capacity < 2:
        raise ValueError("positive vocabulary and a proper joint distance required")
    eligible = np.flatnonzero((count >= 8) & (subjects >= 2))
    eligible = eligible[eligible != 0]
    if not len(eligible):
        raise ValueError("no training-supported representative")
    order = np.lexsort((eligible, -count[eligible]))
    anchors = eligible[order[: capacity - 1]]
    weights = count.astype(float)
    weights[0] = 0
    norm = np.sqrt(np.sum(weights[:, None] * atoms**2) / max(weights.sum(), 1))
    carrier = atoms / max(norm, 1e-12)
    semantics = np.log(np.maximum(means, 1e-12))
    center = np.average(semantics, axis=0, weights=weights)
    scale = np.sqrt(np.average((semantics - center) ** 2, axis=0, weights=weights)).clip(1e-8)
    semantics = (semantics - center) / scale / np.sqrt(semantics.shape[1])
    mapping = np.zeros(len(count), np.int32)
    anchor_carrier = carrier[anchors]
    carrier_norm = np.square(anchor_carrier).sum(1)
    semantic_norm = np.square(semantics[anchors]).sum(1)
    for start in range(1, len(count), 256):
        stop = min(start + 256, len(count))
        c = carrier[start:stop]
        distance = np.maximum(0, np.square(c).sum(1)[:, None] + carrier_norm - 2 * c @ anchor_carrier.T)
        if semantic_fraction:
            s = semantics[start:stop]
            semantic_distance = np.maximum(0, np.square(s).sum(1)[:, None] + semantic_norm - 2 * s @ semantics[anchors].T)
            available = count[start:stop] > 0
            distance[available] = ((1 - semantic_fraction) * distance[available]
                                   + semantic_fraction * semantic_distance[available])
        mapping[start:stop] = distance.argmin(1) + 1
    representatives = np.zeros(capacity, np.int32)
    representatives[1: len(anchors) + 1] = anchors
    representatives[len(anchors) + 1:] = anchors[0]
    if not np.array_equal(mapping[anchors], np.arange(1, len(anchors) + 1)):
        # Distinct fine IDs may have identical atoms and attributes. A designated
        # representative retains its own coarse identity even in such ties.
        mapping[anchors] = np.arange(1, len(anchors) + 1)
    return mapping, representatives


class CoarsenedSpatialDictionary:
    """Decode only coarse integers by replacing them with frozen representative atoms."""

    def __init__(self, source, encoder, metadata, mapping, representatives):
        self.source = source
        self.encoder = encoder
        self.metadata = metadata
        self.mapping = mapping
        self.representatives = representatives
        self.codebook_size = int(metadata['codebook_size'])
        for branch in ('S', 'O'):
            for band in range(5):
                table = mapping[branch, band]
                lift = representatives[branch, band]
                if (table.ndim != 1 or table.dtype.kind not in 'iu' or table[0] != 0
                        or np.any(table[1:] <= 0) or np.any(table >= self.codebook_size)
                        or lift.shape != (self.codebook_size,) or lift.dtype.kind not in 'iu'
                        or lift[0] != 0 or np.any(lift[1:] <= 0) or np.any(lift >= len(table))):
                    raise ValueError('complete coarse remap with a unique missing index required')

    def remap(self, ids, dataset):
        return self._translate(ids, dataset, self.mapping, self.encoder.codebook_size)

    def _translate(self, ids, dataset, tables, vocabulary):
        if set(ids) != {'S', 'O'}:
            raise ValueError('both branch sequences required')
        roles = np.arange(len(self.metadata['positions'][dataset])) % 5
        result = {}
        for branch in ('S', 'O'):
            values = np.asarray(ids[branch])
            if (values.ndim != 3 or values.shape[1:] != (len(roles), 15)
                    or values.dtype.kind not in 'iu' or (values < 0).any()
                    or (values >= vocabulary).any()):
                raise ValueError('complete in-vocabulary native-layout integers required')
            transformed = np.empty(values.shape, np.int32)
            for band in range(5):
                selected = roles == band
                transformed[:, selected] = tables[branch, band][values[:, selected]]
            result[branch] = transformed
        if result['S'].shape != result['O'].shape:
            raise ValueError('matched branch shapes required')
        return result

    def encode(self, values, dataset, observed):
        return self.remap(self.encoder.encode(values, dataset, observed), dataset)

    def decode_ids(self, ids, dataset):
        fine = self._translate(ids, dataset, self.representatives, self.codebook_size)
        return self.source.decode_ids(fine, dataset)

    def save(self, path):
        arrays = {'metadata': np.asarray(json.dumps(self.metadata))}
        for name in ('mapping', 'representatives'):
            for (branch, band), value in getattr(self, name).items():
                arrays[f'{name}_{branch}_{band}'] = value
        np.savez_compressed(path, **arrays)


def load_model(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        if metadata['schema'] != 'coarsened_AS_spatial_dictionary_v1':
            raise ValueError('versioned coarse AS model required')
        for key in ('source_decoder', 'source_encoder'):
            original = (path.parent / metadata[key]).resolve()
            if hashlib.sha256(original.read_bytes()).hexdigest() != metadata[key + '_sha256']:
                raise ValueError('frozen source changed')
        encoder = load_encoder((path.parent / metadata['source_encoder']).resolve())
        source = load_decoder((path.parent / metadata['source_decoder']).resolve())
        tables = {name: {(branch, band): data[f'{name}_{branch}_{band}'].copy()
                         for branch in ('S', 'O') for band in range(5)}
                  for name in ('mapping', 'representatives')}
    return CoarsenedSpatialDictionary(source, encoder, metadata, **tables)
