"""Single-ID EEG atoms assigned by training-scaled dynamics and waveform features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.tokenizers.supervised_waveform_dictionary import SupervisedWaveformDictionary


BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def band_components(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[-1] != 400 or not np.isfinite(values).all():
        raise ValueError('finite two-second 200 Hz EEG patches required')
    frequency = np.fft.rfftfreq(400, 1 / 200)
    spectrum = np.fft.rfft(values, axis=-1)
    components = []
    for low, high in BANDS:
        support = (frequency >= low) & ((frequency <= high) if high == 45 else (frequency < high))
        components.append(np.fft.irfft(spectrum * support, n=400, axis=-1))
    return np.stack(components, axis=1)


def raw_features(values, mean, basis, mode='dynamics_waveform'):
    values = np.asarray(values).reshape(-1, 400)
    if mode in ('theta_alpha_rms', 'semantic_waveform'):
        from src.analysis.tokenizer_pattern_audit import attributes
        parts = []
        for offset in range(0, len(values), 512):
            statistics, fields = attributes(values[offset:offset + 512], 'eeg')
            columns = [statistics[:, fields[name][0]:fields[name][1]].mean(axis=-1)
                       for name in ('power_4_8', 'power_8_13', 'rms_envelope')]
            parts.append(np.log(np.stack(columns, axis=-1).clip(1e-12)))
        semantic = np.concatenate(parts)
        if mode == 'theta_alpha_rms':
            return semantic
        return np.concatenate((semantic, (values - mean) @ basis.T), axis=-1)
    if mode != 'dynamics_waveform':
        raise ValueError('declared spectral feature mode required')
    components = band_components(values)
    # Eight fixed quarter-second energy blocks retain within-token timing.
    log_rms = .5 * np.log(np.mean(components.reshape(-1, 5, 8, 50) ** 2, axis=-1) + 1e-12)
    waveform = (values - mean) @ basis.T
    return np.concatenate((log_rms.reshape(len(values), 40), waveform), axis=-1)


def calibrated_atoms(values, labels, means, representatives, size):
    """Restore training-conditional band energy using fixed training phases."""
    values = np.asarray(values)
    counts = np.bincount(labels, minlength=size)
    energy = np.zeros((size, 5), dtype=np.float64)
    # Chunking bounds temporary memory for complete 61k-patch training support.
    for offset in range(0, len(values), 512):
        components = band_components(values[offset:offset + 512])
        np.add.at(energy, labels[offset:offset + 512], np.mean(components ** 2, axis=-1))
    energy /= np.maximum(counts[:, None], 1)
    components = band_components(means)
    representative_parts = band_components(representatives)
    source_energy = np.mean(components ** 2, axis=-1)
    # A vanishing conditional mean carries no stable phase. Its fixed training
    # representative supplies the atom phase, with the source identity retained.
    fallback = source_energy < 1e-12
    components[fallback] = representative_parts[fallback]
    source_energy = np.mean(components ** 2, axis=-1)
    factor = np.sqrt(energy / np.maximum(source_energy, 1e-30))
    result = means - band_components(means).sum(axis=1)
    result += (components * factor[..., None]).sum(axis=1)
    return result, energy, fallback


class SpectralWaveformDictionary(SupervisedWaveformDictionary):
    def __init__(self, metadata, transfer, means, bases, atoms, feature_mean, feature_scale, centers):
        super().__init__(metadata, transfer, means, bases, atoms)
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        self.centers = centers

    def nearest_ids(self, targets, branch, block_size=256):
        values = self._values(targets)
        features = raw_features(values, self.means[branch], self.bases[branch],
                                self.metadata.get('feature_mode', 'dynamics_waveform'))
        standardized = (features - self.feature_mean[branch]) / self.feature_scale[branch]
        centers = self.centers[branch]
        norm = np.sum(centers ** 2, axis=-1)
        result = []
        for offset in range(0, len(features), block_size):
            block = standardized[offset:offset + block_size]
            distance = np.sum(block ** 2, axis=-1, keepdims=True) + norm - 2 * block @ centers.T
            result.append(distance.argmin(axis=-1))
        return np.concatenate(result).reshape(values.shape[:2]).astype(np.int32)

    def save(self, path):
        payload = dict(metadata=np.asarray(json.dumps(self.metadata, ensure_ascii=False)))
        for name in ('transfer', 'means', 'bases', 'atoms', 'feature_mean', 'feature_scale', 'centers'):
            for branch, value in getattr(self, name).items():
                payload[name + '_' + branch] = value
        path = Path(path)
        temporary = path.with_suffix('.pending.npz')
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)


def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if metadata['schema'] != 'spectral_waveform_dictionary_v1' or metadata['modality'] != 'eeg':
            raise ValueError('frozen single-ID EEG spectral dictionary required')
        fields = {name: {branch: archive[name + '_' + branch] for branch in ('S', 'O')}
                  for name in ('transfer', 'means', 'bases', 'atoms', 'feature_mean', 'feature_scale', 'centers')}
    return SpectralWaveformDictionary(metadata, **fields)
