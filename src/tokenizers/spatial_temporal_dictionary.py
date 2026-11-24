"""Complementary spatial time slots with frozen integer-only linear decoding."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .supervised_waveform_dictionary import SupervisedWaveformDictionary


class SpatialTemporalDictionary:
    def __init__(self, metadata, source, weights):
        self.metadata = metadata
        self.source = source
        self.weights = weights
        self.slot_samples = int(metadata['slot_samples'])
        self.slots = 400 // self.slot_samples
        self.codebook_size = source.codebook_size + 1
        if (metadata['schema'] != 'spatial_temporal_dictionary_v1' or self.slot_samples != source.atom_dim
                or 400 % self.slot_samples or metadata['interface_eligible']):
            raise ValueError('explicit subpatch-slot diagnostic and matching frozen short atoms required')
        self.atoms = {branch: np.concatenate((np.zeros((1, self.slot_samples)), source.atoms[branch]))
                      for branch in ('S', 'O')}

    def roles(self, dataset):
        return np.arange(len(self.metadata['positions'][dataset])) % self.slots

    def _values(self, values, dataset, observed):
        values = np.asarray(values, dtype=np.float64)
        observed = np.asarray(observed)
        positions = len(self.metadata['positions'][dataset])
        if (values.ndim != 4 or values.shape[1:] != (positions, 15, 400)
                or observed.shape != values.shape[:2] or observed.dtype != bool
                or not np.isfinite(values).all()):
            raise ValueError('finite X in fixed native position order and observation availability required')
        return values, observed

    def predict_continuous(self, values, dataset, observed):
        values, observed = self._values(values, dataset, observed)
        shape = values.shape
        output = self.source.predict_continuous(values.reshape(-1, 15, 400))
        return {key: np.where(observed[:, :, None, None], value.reshape(shape), 0.)
                for key, value in output.items()}

    def encode(self, values, dataset, observed):
        values, observed = self._values(values, dataset, observed)
        prediction = self.predict_continuous(values, dataset, observed)
        roles = self.roles(dataset)
        selected = np.arange(15)[None, None, :]*self.slots + roles[None, :, None]
        result = {}
        for branch in ('S', 'O'):
            all_ids = self.source.nearest_ids(prediction[branch].reshape(-1, 15, 400), branch)
            all_ids = all_ids.reshape(len(values), len(roles), 15*self.slots)
            codes = np.take_along_axis(all_ids, np.broadcast_to(selected, (len(values), len(roles), 15)), axis=2)+1
            result[branch] = np.where(observed[:, :, None], codes, 0).astype(np.int32)
        return result

    def slot_sources(self, prediction, observed, dataset):
        """Continuous X-only subsegments for the declared prequantization control."""
        roles = self.roles(dataset)
        result = {}
        for branch in ('S', 'O'):
            wave = np.asarray(prediction[branch])
            if wave.shape[1:] != (len(roles), 15, 400):
                raise ValueError('full branch waveform in native position order required')
            for slot in range(self.slots):
                positions = np.flatnonzero(roles == slot)
                begin = slot*self.slot_samples
                result[branch, slot] = np.where(observed[:, positions, None, None],
                    wave[:, positions, :, begin:begin+self.slot_samples], 0.)
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
                raise ValueError('in-vocabulary integer [batch,native_position,15] sequences required')
            if expected is not None and codes.shape != expected:
                raise ValueError('matching branch sequence supports required')
            expected = codes.shape
            for slot in range(self.slots):
                positions = np.flatnonzero(roles == slot)
                result[branch, slot] = self.atoms[branch][codes[:, positions]]
        return result

    def decode_sources(self, sources, dataset, mode):
        roles = self.roles(dataset)
        example = sources['S', 0]
        result = {}
        for branch in ('S', 'O'):
            output = np.zeros((len(example), len(roles), 15, 400))
            for slot in range(self.slots):
                begin = slot*self.slot_samples
                prediction = np.einsum('ngpt,gk->nkpt', sources[branch, slot], self.weights[dataset, mode, branch, slot], optimize=True)
                output[:, :, :, begin:begin+self.slot_samples] = prediction
            result[branch] = output
        result['sum'] = result['S']+result['O']
        return result

    def decode_ids(self, ids, dataset):
        return self.decode_sources(self.integer_sources(ids, dataset), dataset, 'quantized')

    def save(self, path):
        payload = dict(metadata=np.asarray(json.dumps(self.metadata)),
                       source_metadata=np.asarray(json.dumps(self.source.metadata)))
        for name in ('transfer', 'means', 'bases', 'atoms'):
            for branch, value in getattr(self.source, name).items():
                payload[f'{name}_{branch}'] = value
        for (dataset, mode, branch, slot), value in self.weights.items():
            payload[f'weights_{dataset}_{mode}_{branch}_{slot}'] = value
        temporary = Path(path).with_suffix('.pending.npz')
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)


def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        source_metadata = json.loads(str(archive['source_metadata']))
        fields = {name: {branch: archive[f'{name}_{branch}'] for branch in ('S', 'O')}
                  for name in ('transfer', 'means', 'bases', 'atoms')}
        source = SupervisedWaveformDictionary(source_metadata, **fields)
        slots = 400//metadata['slot_samples']
        weights = {(dataset, mode, branch, slot): archive[f'weights_{dataset}_{mode}_{branch}_{slot}']
                   for dataset in metadata['positions'] for mode in ('continuous', 'quantized')
                   for branch in ('S', 'O') for slot in range(slots)}
    return SpatialTemporalDictionary(metadata, source, weights)
