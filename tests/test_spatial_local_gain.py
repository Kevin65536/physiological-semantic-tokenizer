"""ID-only local gains are smooth, bounded and independently replayable."""
import hashlib

import numpy as np
import pytest

from src.tokenizers.spatial_local_gain import LocalEnergyGainDecoder, integer_features, load_decoder, smooth_gain
from src.tokenizers.spatial_waveform_dictionary import SpatialWaveformDictionary, spectral_features


def source():
    names = ['p' + str(i) for i in range(5)]
    metadata = dict(schema='spatial_waveform_dictionary_v1', codebook_size=3, positions={'fixture': names}, output_projection_hz=[1, 45])
    atoms, centers, weights = {}, {}, {}
    for band, frequency in enumerate([2, 6, 10, 20, 35]):
        wave = np.sin(2 * np.pi * frequency * np.arange(400) / 200)
        for branch in ('S', 'O'):
            atoms[branch, band] = np.array([0., 1., 2.])[:, None] * wave
            centers[branch, band] = spectral_features(atoms[branch, band][1:], band)
            weight = np.zeros((1, 5))
            weight[0, band] = 1.
            weights['fixture', branch, band] = weight
    transfer = {branch: np.ones(3001) * .5 for branch in ('S', 'O')}
    return SpatialWaveformDictionary(metadata, transfer, centers, atoms, weights)


def test_smooth_log_gain_preserves_bounds_and_has_no_token_step():
    values = np.tile(np.log([.5, 2., .5]), 5)[None]
    gain = smooth_gain(values, dict(smoothing_sigma_s=.25))
    assert gain.shape == (1, 6000)
    assert gain.min() >= .5 - 1e-12 and gain.max() <= 2 + 1e-12
    assert np.max(abs(np.diff(gain))) < .02


def test_integer_only_local_energy_gain_and_saved_reload(tmp_path):
    base = source()
    path = tmp_path / 'source.npz'
    base.save(path)
    ids = {branch: np.ones((1, 5, 15), dtype=np.int32) for branch in ('S', 'O')}
    original = base.decode_ids(ids, 'fixture')
    wave = np.stack([original[branch][0].reshape(5, 6000) for branch in ('S', 'O')])
    features, _ = integer_features(wave, ids)
    assert features.shape == (5, 15, 37)
    state = dict(mean=np.zeros(37), scale=np.ones(37), coefficient=np.zeros((4, 37)), intercept=np.full(4, np.log(2.)))
    settings = dict(residual_gain_bounds=[.5, 2], total_gain_bounds=[.25, 4], smoothing_sigma_s=.25)
    metadata = dict(schema='spatial_AS_local_gain_v1', source_decoder=str(path), source_decoder_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    fitted = LocalEnergyGainDecoder(base, {'fixture': state}, {'fixture': [1.] * 10}, settings, 1., metadata)
    prediction = fitted.decode_ids(ids, 'fixture')
    for branch in ('S', 'O'):
        np.testing.assert_allclose(prediction[branch][:, [1, 2]], 2 * original[branch][:, [1, 2]], atol=1e-12)
        np.testing.assert_allclose(prediction[branch][:, [0, 3, 4]], original[branch][:, [0, 3, 4]], atol=1e-12)
    fitted.save(tmp_path / 'local.npz')
    np.savez(tmp_path / 'ids.npz', **ids)
    with np.load(tmp_path / 'ids.npz') as saved:
        replay = load_decoder(tmp_path / 'local.npz').decode_ids(dict(saved), 'fixture')
    for branch in ('S', 'O', 'sum'):
        np.testing.assert_array_equal(prediction[branch], replay[branch])
    with pytest.raises(ValueError, match='integer'):
        fitted.decode_ids({'S': ids['S'].astype(float), 'O': ids['O']}, 'fixture')
    with pytest.raises(ValueError, match='both declared'):
        fitted.decode_ids(dict(ids, scale=np.ones(1)), 'fixture')
