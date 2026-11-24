"""Bounded local EEG energy correction from frozen integer decoder outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from src.analysis.dual_component_tokenizer import eeg_complete_patch_band_energy
from .spatial_waveform_dictionary import band_filter, load_model

CRITICAL = (1, 2)


def patch_rms(wave):
    values = np.asarray(wave, dtype=np.float64).reshape(-1, 1, 400, 1)
    power = eeg_complete_patch_band_energy(values, np.ones_like(values, dtype=bool))
    return np.sqrt(np.stack([power[key][:, 0] for key in ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))], axis=-1))


def integer_features(wave, ids):
    """Own, neighbouring-time and robust field energy, plus fixed native position."""
    wave = np.asarray(wave, dtype=np.float64)
    positions = wave.shape[1]
    rms = np.stack([patch_rms(wave[branch]).reshape(positions, 15, 5) for branch in range(2)], axis=2)
    log_rms = np.log(np.maximum(rms, 1e-8))
    own = log_rms.reshape(positions, 15, 10)
    critical = log_rms[:, :, :, CRITICAL].reshape(positions, 15, 4)
    temporal = np.concatenate([critical[:, np.clip(np.arange(15) + offset, 0, 14)] for offset in (-1, 1)], axis=-1)
    medians = np.zeros((15, 2, 5))
    spread = np.zeros((15, 2, 2))
    for branch, name in enumerate(('S', 'O')):
        available = np.asarray(ids[name])[0] != 0
        for patch in range(15):
            selected = log_rms[available[:, patch], patch, branch]
            if len(selected):
                quartiles = np.quantile(selected, [.25, .5, .75], axis=0)
                medians[patch, branch] = quartiles[1]
                spread[patch, branch] = (quartiles[2] - quartiles[0])[list(CRITICAL)]
    field = np.broadcast_to(medians.reshape(1, 15, 10), (positions, 15, 10))
    iqr = np.broadcast_to(spread.reshape(1, 15, 4), (positions, 15, 4))
    native = np.broadcast_to(np.eye(positions)[:, None], (positions, 15, positions))
    return np.concatenate((own, temporal, field, iqr, native), axis=-1), rms


def filtered_bank(wave):
    return np.stack([[band_filter(wave[branch], band) for band in range(5)] for branch in range(2)])


def smooth_gain(log_gain, settings):
    time = np.arange(6000) / 200
    centers = np.arange(15) * 2 + 1
    interpolated = np.stack([np.interp(time, centers, row) for row in log_gain])
    sigma = settings['smoothing_sigma_s'] * 200
    radius = int(4 * sigma + .5)
    kernel = np.exp(-.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(interpolated, ((0, 0), (radius, radius)), mode='edge')
    return np.exp(fftconvolve(padded, kernel[None], mode='valid', axes=-1))


def fixed_prediction(wave, bands, gain):
    gain = np.asarray(gain).reshape(2, 5)
    return wave + np.sum(bands * (gain - 1)[:, :, None, None], axis=1)


def local_prediction(wave, bands, features, state, fixed_gain, strength, settings):
    fixed_gain = np.asarray(fixed_gain).reshape(2, 5)
    standard = (features - state['mean']) / state['scale']
    predicted = standard @ state['coefficient'].T + state['intercept']
    predicted = predicted.reshape(len(wave[0]), 15, 2, 2)
    lower, upper = np.log(settings['residual_gain_bounds'])
    predicted = np.clip(predicted, lower, upper) * strength
    output = fixed_prediction(wave, bands, fixed_gain)
    for branch in range(2):
        for column, band in enumerate(CRITICAL):
            logarithm = predicted[:, :, branch, column] + np.log(fixed_gain[branch, band])
            lo, hi = np.log(settings['total_gain_bounds'])
            gain = smooth_gain(np.clip(logarithm, lo, hi), settings)
            output[branch] += bands[branch, band] * (gain - fixed_gain[branch, band])
    frequency = np.fft.rfftfreq(6000, 1 / 200)
    projected = np.fft.rfft(output, axis=-1)
    projected[:, :, ~((frequency >= 1) & (frequency < 45))] = 0
    return np.fft.irfft(projected, n=6000, axis=-1)


class LocalEnergyGainDecoder:
    def __init__(self, source, states, fixed_gains, settings, strength, metadata):
        self.source = source
        self.states = states
        self.fixed_gains = fixed_gains
        self.settings = settings
        self.strength = strength
        self.metadata = dict(metadata, positions=source.metadata['positions'])

    def decode_ids(self, ids, dataset):
        decoded = self.source.decode_ids(ids, dataset)
        wave = np.stack([decoded[name][0].reshape(-1, 6000) for name in ('S', 'O')])
        features, _ = integer_features(wave, ids)
        value = local_prediction(wave, filtered_bank(wave), features, self.states[dataset], self.fixed_gains[dataset], self.strength, self.settings)
        result = {name: value[branch][None].reshape(decoded[name].shape) for branch, name in enumerate(('S', 'O'))}
        result['sum'] = result['S'] + result['O']
        return result

    def save(self, path):
        metadata = dict(self.metadata, settings=self.settings, strength=self.strength, fixed_gains=self.fixed_gains)
        arrays = {'metadata': np.asarray(json.dumps(metadata))}
        for dataset, state in self.states.items():
            for key, value in state.items():
                arrays[dataset + '__' + key] = value
        path = Path(path)
        temporary = path.with_suffix('.pending.npz')
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)


def load_decoder(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if metadata['schema'] != 'spatial_AS_local_gain_v1':
            raise ValueError('versioned ID-only local energy decoder required')
        source = (Path(path).parent / metadata['source_decoder']).resolve()
        if hashlib.sha256(source.read_bytes()).hexdigest() != metadata['source_decoder_sha256']:
            raise ValueError('frozen spatial source decoder changed')
        states = {dataset: {key: archive[dataset + '__' + key] for key in ('mean', 'scale', 'coefficient', 'intercept')}
                  for dataset in metadata['fixed_gains']}
    return LocalEnergyGainDecoder(load_model(source), states, metadata['fixed_gains'], metadata['settings'], metadata['strength'], metadata)
