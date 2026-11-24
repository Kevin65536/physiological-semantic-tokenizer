"""Single-ID, position-assigned spectral dictionaries with integer-only spatial decoding."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def spectral_features(patches, band):
    patches = np.asarray(patches, dtype=np.float64)
    frequency = np.fft.rfftfreq(400, 1 / 200)
    lower, upper = BANDS[band]
    selected = (frequency >= lower) & (frequency < upper)
    spectrum = np.fft.rfft(patches, axis=-1, norm='ortho')[..., selected]
    return np.concatenate((spectrum.real, spectrum.imag), axis=-1) * np.sqrt(2)


def joint_assignment_features(patches, band):
    from .spectral_waveform_dictionary import raw_features
    carrier = spectral_features(patches, band)
    if not len(carrier):
        return np.empty((0, carrier.shape[1]+3))
    semantics = raw_features(patches, None, None, mode='theta_alpha_rms')
    return np.column_stack((carrier, semantics))


def band_filter(waveforms, band, axis=-1):
    frequency = np.fft.rfftfreq(waveforms.shape[axis], 1 / 200)
    lower, upper = BANDS[band]
    keep = (frequency >= lower) & (frequency < upper)
    shape = [1] * waveforms.ndim
    shape[axis] = len(keep)
    spectrum = np.fft.rfft(waveforms, axis=axis)
    return np.fft.irfft(spectrum * keep.reshape(shape), n=waveforms.shape[axis], axis=axis)


class SpatialWaveformDictionary:
    def __init__(self, metadata, transfer, centers, atoms, weights, readout=None, feature_weights=None):
        self.metadata = metadata
        self.transfer = transfer
        self.centers = centers
        self.atoms = atoms
        self.weights = weights
        self.readout = readout or {}
        self.feature_weights = feature_weights or {}
        self.codebook_size = metadata['codebook_size']

    def roles(self, dataset, branch=None):
        offset = self.metadata.get('branch_role_offsets', {}).get(branch, 0)
        return (np.arange(len(self.metadata['positions'][dataset])) + offset) % len(BANDS)

    def predict_continuous(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 4 or values.shape[-2:] != (15, 400) or not np.isfinite(values).all():
            raise ValueError('finite [batch,native_position,15,400] observations required')
        waveforms = values.reshape(*values.shape[:2], 6000)
        spectrum = np.fft.rfft(waveforms, axis=-1)
        result = {branch: np.fft.irfft(spectrum * self.transfer[branch].reshape(1, 1, -1), n=6000, axis=-1).reshape(values.shape)
                  for branch in ('S', 'O')}
        result['sum'] = result['S'] + result['O']
        return result

    def assignment_features(self, patches, branch, band):
        if self.feature_weights:
            return joint_assignment_features(patches, band) * self.feature_weights[branch, band]
        return spectral_features(patches, band)

    def encode(self, values, dataset, observed):
        values = np.asarray(values)
        roles = self.roles(dataset)
        observed = np.asarray(observed)
        if values.shape[1] != len(roles) or observed.shape != values.shape[:2] or observed.dtype != bool:
            raise ValueError('fixed native position order and observation-only availability required')
        return self.encode_positions(values, dataset, np.arange(len(roles)), observed)

    def encode_positions(self, values, dataset, native_indices, observed=None):
        """Exact position-independent encoder subset, retaining each fixed band role.

        Spatial reconstruction still requires the complete native-layout integer
        sequence. This entry only avoids encoding unrelated positions for code
        support and local-attribute readouts.
        """
        values = np.asarray(values)
        indices = np.broadcast_to(np.asarray(native_indices), values.shape[:2])
        if (indices.dtype.kind not in 'iu' or (indices < 0).any()
                or (indices >= len(self.roles(dataset))).any()):
            raise ValueError('valid fixed native indices required')
        if observed is None:
            observed = np.ones(values.shape[:2], dtype=bool)
        observed = np.asarray(observed)
        if observed.shape != values.shape[:2] or observed.dtype != bool:
            raise ValueError('observation availability must match selected positions')
        prediction = self.predict_continuous(values)
        return self._assign_positions(prediction, dataset, indices, observed)

    def _assign_positions(self, prediction, dataset, indices, observed):
        """Nearest-atom initialization from already computed observation-only source estimates."""
        result = {}
        for branch in ('S', 'O'):
            roles = self.roles(dataset, branch)[indices]
            ids = np.zeros((*observed.shape, 15), dtype=np.int32)
            for band in range(len(BANDS)):
                positions = roles == band
                patches = prediction[branch][positions].reshape(-1, 400)
                features = self.assignment_features(patches, branch, band)
                centers = self.centers[branch, band]
                center_norm = np.square(centers).sum(axis=1)
                nearest = []
                for begin in range(0, len(features), 256):
                    block = features[begin:begin + 256]
                    distance = np.square(block).sum(axis=1, keepdims=True) + center_norm - 2 * block @ centers.T
                    nearest.append(distance.argmin(axis=1) + 1)
                if nearest:
                    ids[positions] = np.concatenate(nearest).reshape(-1, 15)
            ids[~observed] = 0
            result[branch] = ids
        return result

    def decode_ids(self, ids, dataset, *, spatial=True, cross_branch=True):
        if set(ids) != {'S', 'O'}:
            raise ValueError('both declared EEG branch integer sequences required')
        if spatial and self.metadata.get('joint_spatial_inputs') == 'both_branch_atoms':
            if not cross_branch:
                raise ValueError('joint spatial model requires both branches; use explicit reserved-zero IDs for an intervention')
            independent = self.decode_ids(ids, dataset, spatial=False)
            positions = len(self.metadata['positions'][dataset])
            result = {branch: np.zeros_like(independent[branch]) for branch in ('S', 'O')}
            for band in range(len(BANDS)):
                sources = []
                for branch in ('S', 'O'):
                    wave = independent[branch].reshape(len(ids[branch]), positions, 6000)
                    anchors = np.flatnonzero(self.roles(dataset, branch) == band)
                    sources.append(band_filter(wave[:, anchors], band).transpose(0, 2, 1))
                joined = np.concatenate(sources, axis=-1)
                for branch in ('S', 'O'):
                    prediction = joined @ self.weights[dataset, branch, band]
                    result[branch] += prediction.transpose(0, 2, 1).reshape(result[branch].shape)
            result['sum'] = result['S'] + result['O']
            return result
        roles = self.roles(dataset)
        result = {}
        expected_shape = None
        for branch in ('S', 'O'):
            roles = self.roles(dataset, branch)
            codes = np.asarray(ids[branch])
            if (codes.ndim != 3 or codes.shape[1:] != (len(roles), 15) or codes.dtype.kind not in 'iu'
                    or (codes < 0).any() or (codes >= self.codebook_size).any()):
                raise ValueError('in-vocabulary integer [batch,native_position,15] sequences required')
            if expected_shape is not None and expected_shape != codes.shape:
                raise ValueError('matching branch integer supports required')
            expected_shape = codes.shape
            decoded = np.zeros((*codes.shape, 400), dtype=np.float64)
            for band in range(len(BANDS)):
                positions = np.flatnonzero(roles == band)
                decoded[:, positions] = self.atoms[branch, band][codes[:, positions]]
            waveforms = decoded.reshape(len(codes), len(roles), 6000).transpose(0, 2, 1)
            if spatial:
                reconstructed = waveforms.copy()
                for band in range(len(BANDS)):
                    positions = np.flatnonzero(roles == band)
                    baseline = band_filter(waveforms, band, axis=1)
                    predicted = baseline[:, :, positions] @ self.weights[dataset, branch, band]
                    available = (codes[:, positions] != 0).any(axis=(1, 2))
                    reconstructed[available] += predicted[available] - baseline[available]
                waveforms = reconstructed
            result[branch] = waveforms.transpose(0, 2, 1).reshape(len(codes), len(roles), 15, 400)
        projection = self.metadata.get('output_projection_hz')
        if projection is not None:
            if projection != [1, 45]:
                raise ValueError('declared 1-45 Hz output projection required')
            frequency = np.fft.rfftfreq(6000, 1 / 200)
            retained = (frequency >= 1) & (frequency < 45)
            for branch in ('S', 'O'):
                values = result[branch]
                spectrum = np.fft.rfft(values.reshape(*values.shape[:2], 6000), axis=-1)
                result[branch] = np.fft.irfft(spectrum * retained[None, None], n=6000, axis=-1).reshape(values.shape)
        if self.readout and spatial and cross_branch:
            original = {branch: result[branch].reshape(len(codes), len(roles), 6000) for branch in ('S', 'O')}
            combined = {branch: values.copy() for branch, values in original.items()}
            for band in range(len(BANDS)):
                filtered = np.stack([band_filter(original[branch], band) for branch in ('S', 'O')], axis=-1)
                transformed = np.einsum('bpti,pio->bpto', filtered, self.readout[dataset][band], optimize=True)
                for column, branch in enumerate(('S', 'O')):
                    combined[branch] += transformed[..., column] - filtered[..., column]
            for branch in ('S', 'O'):
                result[branch] = combined[branch].reshape(len(codes), len(roles), 15, 400)
        result['sum'] = result['S'] + result['O']
        return result

    def save(self, path):
        payload = {'metadata': np.asarray(json.dumps(self.metadata))}
        for branch, value in self.transfer.items():
            payload['transfer_' + branch] = value
        for name in ('centers', 'atoms', 'feature_weights'):
            for (branch, band), value in getattr(self, name).items():
                payload[f'{name}_{branch}_{band}'] = value
        for (dataset, branch, band), value in self.weights.items():
            payload[f'weights_{dataset}_{branch}_{band}'] = value
        for dataset, value in self.readout.items():
            payload['readout_' + dataset] = value
        destination = Path(path)
        temporary = destination.with_suffix('.pending.npz')
        np.savez_compressed(temporary, **payload)
        temporary.replace(destination)


def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if metadata['schema'] == 'fixed_role_mixed_spatial_v1':
            from .role_mixed_spatial import load_role_mixed_decoder
            return load_role_mixed_decoder(path)
        if metadata['schema'] == 'integer_sum_source_decoder_v1':
            from .eeg_component_transfer import load_integer_decoder
            return load_integer_decoder(path)
        if metadata['schema'] == 'backward_adaptive_carrier_v1':
            from .backward_adaptive_carrier import load_model as load_backward
            return load_backward(path)
        if metadata['schema'] == 'spatial_AS_local_gain_v1':
            from .spatial_local_gain import load_decoder
            return load_decoder(path)
        if metadata['schema'] in ('montage_joint_tokenizer_v1','montage_joint_phase_readout_v1'):
            from .montage_joint_tokenizer import load_model as load_joint
            return load_joint(path)
        if metadata['schema'] not in ('spatial_waveform_dictionary_v1', 'spatial_waveform_dictionary_v2', 'spatial_waveform_dictionary_v3'):
            raise ValueError('versioned spatial dictionary required')
        if metadata['schema'] == 'spatial_waveform_dictionary_v3':
            offsets = metadata.get('branch_role_offsets')
            if (metadata.get('joint_spatial_inputs') != 'both_branch_atoms' or offsets is None
                    or set(offsets) != {'S', 'O'} or any(type(value) is not int or not 0 <= value < len(BANDS)
                                                       for value in offsets.values())
                    or metadata.get('output_projection_hz') != [1, 45]):
                raise ValueError('fixed branch roles and complete joint integer spatial decoder required')
        transfer = {branch: archive['transfer_' + branch] for branch in ('S', 'O')}
        fields = {name: {(branch, band): archive[f'{name}_{branch}_{band}']
                         for branch in ('S', 'O') for band in range(len(BANDS))} for name in ('centers', 'atoms')}
        feature_weights = {}
        if metadata.get('semantic_distance_fraction') is not None:
            feature_weights = {(branch, band): archive[f'feature_weights_{branch}_{band}']
                               for branch in ('S', 'O') for band in range(len(BANDS))}
        weights = {(dataset, branch, band): archive[f'weights_{dataset}_{branch}_{band}']
                   for dataset in metadata['positions'] for branch in ('S', 'O') for band in range(len(BANDS))}
        if metadata['schema'] == 'spatial_waveform_dictionary_v3':
            for dataset, names in metadata['positions'].items():
                for band in range(len(BANDS)):
                    width = sum(int(np.sum((np.arange(len(names)) + offsets[branch]) % len(BANDS) == band))
                                for branch in ('S', 'O'))
                    for branch in ('S', 'O'):
                        if weights[dataset, branch, band].shape != (width, len(names)):
                            raise ValueError('joint spatial coefficient support must match declared branch roles')
        readout = {dataset: archive['readout_' + dataset] for dataset in metadata['positions']
                   if 'readout_' + dataset in archive.files}
        if metadata['schema'] == 'spatial_waveform_dictionary_v2' and set(readout) != set(metadata['positions']):
            raise ValueError('complete frozen cross-branch readout required')
    return SpatialWaveformDictionary(metadata, transfer, weights=weights, readout=readout, feature_weights=feature_weights, **fields)
