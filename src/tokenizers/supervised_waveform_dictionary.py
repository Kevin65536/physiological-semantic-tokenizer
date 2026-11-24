"""Training-fitted frequency regression followed by one waveform atom per patch.

The encoder may use the entire declared 30-second observation. The decoder
only indexes frozen two-second waveform atoms. Each branch has one integer ID
per patch. Hb atoms jointly contain HbO and HbR in time-major order.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def nearest_masked_waveform_ids(targets, mask, atoms, *, target_block_size=32, atom_block_size=256):
    """Exhaustively minimize full-waveform MSE on each target's valid points.

    Inputs are [patch, coordinate] targets/mask and [code, coordinate] atoms.
    Distances use direct float64 differences and every atom, including waveform
    dimensions outside an encoder PCA basis. Exact ties select the lowest ID.
    A patch with zero valid points receives ID 0 and NaN MSE, with a zero count.
    """
    targets = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(mask)
    atoms = np.asarray(atoms, dtype=np.float64)
    if (targets.ndim != 2 or atoms.ndim != 2 or mask.shape != targets.shape
            or mask.dtype != np.bool_ or atoms.shape[1] != targets.shape[1]
            or len(atoms) == 0 or target_block_size < 1 or atom_block_size < 1):
        raise ValueError('aligned target masks and nonempty full-waveform atoms required')
    if not np.isfinite(atoms).all() or np.any(mask & ~np.isfinite(targets)):
        raise ValueError('finite atoms and scored target coordinates required')
    safe_targets = np.where(mask, targets, 0.)
    counts = mask.sum(axis=1)
    ids = np.zeros(len(targets), dtype=np.int32)
    costs = np.full(len(targets), np.inf)
    for start in range(0, len(targets), target_block_size):
        stop = min(start + target_block_size, len(targets))
        target_block = safe_targets[start:stop]
        valid = mask[start:stop]
        best_cost = costs[start:stop]
        best_ids = ids[start:stop]
        for code_start in range(0, len(atoms), atom_block_size):
            residual = target_block[:, None] - atoms[None, code_start:code_start + atom_block_size]
            np.square(residual, out=residual)
            distances = np.einsum('nkd,nd->nk', residual, valid, optimize=False)
            local_ids = distances.argmin(axis=1)
            local_cost = distances[np.arange(len(target_block)), local_ids]
            improved = local_cost < best_cost
            best_cost[improved] = local_cost[improved]
            best_ids[improved] = code_start + local_ids[improved]
    mse = np.divide(costs, counts, out=np.full(len(targets), np.nan), where=counts > 0)
    return ids, mse, counts


class SupervisedWaveformDictionary:
    def __init__(self, metadata, transfer, means, bases, atoms):
        self.metadata = metadata
        self.modality = metadata['modality']
        self.channels = 1 if self.modality == 'eeg' else 2
        self.input_dim = 400 if self.modality == 'eeg' else 40
        self.codebook_size = int(metadata['codebook_size'])
        self.atom_dim = int(metadata.get('atom_dim', self.input_dim))
        if self.atom_dim < 1 or (15 * self.input_dim) % self.atom_dim:
            raise ValueError('atom support must partition the declared 30-second input')
        self.atom_patches = 15 * self.input_dim // self.atom_dim
        if self.atom_patches != 15 and metadata.get('interface_eligible') is not False:
            raise ValueError('changed time unit must be explicitly declared as an interface diagnostic')
        if self.atom_dim % self.channels or metadata.get('atom_patches', self.atom_patches) != self.atom_patches:
            raise ValueError('atomic support must preserve paired-color and integer-count coordinates')
        for branch in ('S', 'O'):
            if (np.shape(transfer[branch]) != ((3001 if self.modality == 'eeg' else 151), self.channels, self.channels)
                    or np.shape(means[branch]) != (self.atom_dim,)
                    or np.asarray(bases[branch]).ndim != 2
                    or np.shape(bases[branch])[1] != self.atom_dim
                    or np.shape(atoms[branch]) != (self.codebook_size, self.atom_dim)
                    or any(not np.isfinite(field[branch]).all() for field in (transfer, means, bases, atoms))):
                raise ValueError('finite frozen arrays must match the declared waveform coordinates')
        self.transfer = transfer
        self.means = means
        self.bases = bases
        self.atoms = atoms
        self._atom_latent = {branch: (atoms[branch] - means[branch]) @ bases[branch].T
                             for branch in ('S', 'O')}

    def _values(self, values):
        values = np.asarray(values)
        if values.ndim != 3 or values.shape[1:] != (15, self.input_dim):
            raise ValueError('declared [batch,15,patch samples times channels] required')
        if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
            raise ValueError('finite floating observations required')
        return values.astype(np.float64, copy=False)

    def predict_continuous(self, values):
        values = self._values(values)
        wave = values.reshape(len(values), -1, self.channels)
        spectrum = np.fft.rfft(wave, axis=1)
        result = {}
        for branch in ('S', 'O'):
            transformed = np.einsum('nfi,fij->nfj', spectrum, self.transfer[branch], optimize=True)
            result[branch] = np.fft.irfft(transformed, n=wave.shape[1], axis=1).reshape(values.shape)
        result['sum'] = result['S'] + result['O']
        return result

    def nearest_ids(self, targets, branch, block_size=512):
        values = self._values(targets).reshape(-1, self.atom_dim)
        projected = (values - self.means[branch]) @ self.bases[branch].T
        centers = self._atom_latent[branch]
        center_norm = np.square(centers).sum(axis=1)
        result = []
        for start in range(0, len(projected), block_size):
            block = projected[start:start + block_size]
            distances = np.square(block).sum(axis=1, keepdims=True) + center_norm - 2 * block @ centers.T
            result.append(distances.argmin(axis=1))
        return np.concatenate(result).reshape(len(targets), self.atom_patches).astype(np.int32)

    def encode(self, values):
        prediction = self.predict_continuous(values)
        return {branch: self.nearest_ids(prediction[branch], branch) for branch in ('S', 'O')}

    def decode_ids(self, ids):
        if set(ids) != {'S', 'O'}:
            raise ValueError('both declared branch integer sequences required')
        result = {}
        shape = None
        for branch in ('S', 'O'):
            values = np.asarray(ids[branch])
            if values.ndim != 2 or values.shape[1] != self.atom_patches or not np.issubdtype(values.dtype, np.integer):
                raise ValueError('integer sequences matching the frozen atom count required')
            if shape is not None and values.shape != shape:
                raise ValueError('same branch sequence supports required')
            if (values < 0).any() or (values >= self.codebook_size).any():
                raise ValueError('integer ID outside the frozen vocabulary')
            shape = values.shape
            result[branch] = self.atoms[branch][values].reshape(len(values), 15, self.input_dim)
        result['sum'] = result['S'] + result['O']
        return result

    def __call__(self, values):
        ids = self.encode(values)
        return dict(self.decode_ids(ids), ids=ids)

    def save(self, path):
        payload = {'metadata': np.asarray(json.dumps(self.metadata, ensure_ascii=False))}
        for name in ('transfer', 'means', 'bases', 'atoms'):
            for branch, value in getattr(self, name).items():
                payload[name + '_' + branch] = value
        destination = Path(path)
        temporary = destination.with_suffix('.pending.npz')
        np.savez_compressed(temporary, **payload)
        temporary.replace(destination)


def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if metadata['schema'] != 'supervised_waveform_dictionary_v1':
            raise ValueError('supported frozen checkpoint required')
        fields = {name: {branch: archive[name + '_' + branch] for branch in ('S', 'O')}
                  for name in ('transfer', 'means', 'bases', 'atoms')}
    return SupervisedWaveformDictionary(metadata, **fields)
