"""Full two-second atoms with local attributes and complementary spatial carriers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .spatial_temporal_dictionary import SpatialTemporalDictionary
from .spectral_waveform_dictionary import band_components, raw_features
from .supervised_waveform_dictionary import SupervisedWaveformDictionary


def semantic_slot_features(patches, slot, slot_samples=100):
    patches = np.asarray(patches, dtype=np.float64).reshape(-1, 400)
    semantics = raw_features(patches, None, None, 'theta_alpha_rms')
    begin = slot*slot_samples
    return np.concatenate((semantics, patches[:, begin:begin+slot_samples]), axis=1)


class SpatialSemanticDictionary(SpatialTemporalDictionary):
    def __init__(self, metadata, source, books, weights):
        self.metadata = metadata
        self.source = source
        self.books = books
        self.weights = weights
        self.slot_samples = int(metadata['slot_samples'])
        self.slots = 400//self.slot_samples
        self.codebook_size = int(metadata['codebook_size'])
        if (metadata['schema'] != 'spatial_semantic_dictionary_v1' or not metadata['interface_eligible']
                or self.slot_samples not in (50, 100) or source.input_dim != 400):
            raise ValueError('versioned full two-second single-ID spatial-semantic model required')

    def encode(self, values, dataset, observed):
        values, observed = self._values(values, dataset, observed)
        prediction = self.predict_continuous(values, dataset, observed)
        roles = self.roles(dataset)
        result = {}
        for branch in ('S', 'O'):
            codes = np.zeros((*values.shape[:2], 15), dtype=np.int32)
            for slot in range(self.slots):
                positions = np.flatnonzero(roles == slot)
                book = self.books[branch, slot]
                features = semantic_slot_features(prediction[branch][:, positions], slot, self.slot_samples)
                features = (features-book['feature_mean'])/book['feature_scale']
                centers = book['centers']
                norm = np.square(centers).sum(1)
                selected = []
                for begin in range(0, len(features), 256):
                    block = features[begin:begin+256]
                    distance = np.square(block).sum(1, keepdims=True)+norm-2*block@centers.T
                    selected.append(distance.argmin(1)+1)
                if selected:
                    codes[:, positions] = np.concatenate(selected).reshape(len(values), len(positions), 15)
            result[branch] = np.where(observed[:, :, None], codes, 0).astype(np.int32)
        return result

    def integer_sources(self, ids, dataset):
        roles = self.roles(dataset)
        if set(ids) != {'S', 'O'}:
            raise ValueError('both branch integer sequences required')
        result = {}
        expected = None
        for branch in ('S', 'O'):
            codes = np.asarray(ids[branch])
            if (codes.ndim != 3 or codes.shape[1:] != (len(roles), 15) or codes.dtype.kind not in 'iu'
                    or (codes < 0).any() or (codes >= self.codebook_size).any()):
                raise ValueError('one in-vocabulary integer per native position and two-second patch required')
            if expected is not None and codes.shape != expected:
                raise ValueError('matching branch integer supports required')
            expected = codes.shape
            for slot in range(self.slots):
                positions = np.flatnonzero(roles == slot)
                begin = slot*self.slot_samples
                atoms = self.books[branch, slot]['atoms'][codes[:, positions]]
                result[branch, slot] = atoms[..., begin:begin+self.slot_samples]
        return result

    def decode_ids(self, ids, dataset, *, calibrated=True):
        result = self.decode_sources(self.integer_sources(ids, dataset), dataset, 'quantized')
        if not calibrated:
            return result
        roles = self.roles(dataset)
        for branch in ('S', 'O'):
            wave = result[branch]
            parts = band_components(wave.reshape(-1, 400))
            energy = np.zeros((*wave.shape[:-1], 5))
            for slot in range(self.slots):
                positions = np.flatnonzero(roles == slot)
                energy[:, positions] = self.books[branch, slot]['band_energy'][ids[branch][:, positions]]
            desired = energy.reshape(-1, 5)
            current = np.mean(parts**2, axis=-1)
            factor = np.sqrt(desired/np.maximum(current, 1e-20))
            factor = np.minimum(factor, self.metadata['calibration_gain_cap'])
            factor[np.asarray(ids[branch]).reshape(-1) == 0] = 1.
            remainder = wave.reshape(-1, 400)-parts.sum(1)
            result[branch] = (remainder+(parts*factor[..., None]).sum(1)).reshape(wave.shape)
        result['sum'] = result['S']+result['O']
        return result

    def save(self, path):
        payload = dict(metadata=np.asarray(json.dumps(self.metadata)), source_metadata=np.asarray(json.dumps(self.source.metadata)))
        for name in ('transfer', 'means', 'bases', 'atoms'):
            for branch, value in getattr(self.source, name).items():
                payload[f'source_{name}_{branch}'] = value
        for (branch, slot), book in self.books.items():
            for field, value in book.items():
                payload[f'book_{branch}_{slot}_{field}'] = value
        for (dataset, mode, branch, slot), value in self.weights.items():
            payload[f'weights_{dataset}_{mode}_{branch}_{slot}'] = value
        temporary = Path(path).with_suffix('.pending.npz')
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)


def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        source_metadata = json.loads(str(archive['source_metadata']))
        fields = {name: {branch: archive[f'source_{name}_{branch}'] for branch in ('S', 'O')}
                  for name in ('transfer', 'means', 'bases', 'atoms')}
        source = SupervisedWaveformDictionary(source_metadata, **fields)
        slots = 400//metadata['slot_samples']
        book_fields = ('feature_mean', 'feature_scale', 'centers', 'atoms', 'band_energy', 'rms_envelope', 'counts')
        books = {(branch, slot): {field: archive[f'book_{branch}_{slot}_{field}'] for field in book_fields}
                 for branch in ('S', 'O') for slot in range(slots)}
        weights = {(dataset, mode, branch, slot): archive[f'weights_{dataset}_{mode}_{branch}_{slot}']
                   for dataset in metadata['positions'] for mode in ('continuous', 'quantized')
                   for branch in ('S', 'O') for slot in range(slots)}
    return SpatialSemanticDictionary(metadata, source, books, weights)
