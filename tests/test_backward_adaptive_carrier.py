"""Decoder state is a bounded causal function of integer history within one window."""
import numpy as np
import pytest

from src.tokenizers.backward_adaptive_carrier import decode_sequences, quantize_sequences


def fixture():
    atoms = np.zeros((4, 400))
    atoms[1] = 1.
    atoms[2] = 2.
    atoms[3] = .5
    book = dict(atoms=atoms, centers=np.array([[1.], [2.], [.5]]))
    settings = dict(reference_atom_RMS=1., maximum_log_step=np.log(2), minimum_ratio=.25, maximum_ratio=4.)
    return book, settings


def test_state_uses_previous_ID_and_resets_every_call():
    book, settings = fixture()
    ids = np.ones((1, 15), np.int32)
    ids[0, 0] = 2
    decoded, gains = decode_sequences(ids, np.ones(1), book, settings, True)
    np.testing.assert_array_equal(gains[0, :3], [1., 2., 2.])
    np.testing.assert_array_equal(decoded[0, :2, 0], [2., 2.])
    changed = ids.copy()
    changed[0, 4:] = 3
    later, changed_gains = decode_sequences(changed, np.ones(1), book, settings, True)
    np.testing.assert_array_equal(later[:, :4], decoded[:, :4])
    np.testing.assert_array_equal(changed_gains[:, :5], gains[:, :5])
    np.testing.assert_array_equal(decode_sequences(ids, np.ones(1), book, settings, True)[0], decoded)
    assert changed_gains.min() >= .25 and changed_gains.max() <= 4.


def test_encoder_and_integer_decoder_propagate_identical_state():
    book, settings = fixture()
    features = np.full((2, 15, 1), 2.)
    features[:, 4:10] = 7.
    ids, encoder_gain = quantize_sequences(features, np.ones(2), book, settings, True)
    _, replay_gain = decode_sequences(ids, np.ones(2), book, settings, True)
    np.testing.assert_array_equal(encoder_gain, replay_gain)
    fixed, fixed_gain = quantize_sequences(features, np.ones(2), book, settings, False)
    np.testing.assert_array_equal(fixed_gain, 1.)
    with pytest.raises(ValueError, match='integer'):
        decode_sequences(ids.astype(float), np.ones(2), book, settings, True)


def test_complete_model_reload_has_no_cross_window_state_or_gain_input(tmp_path):
    from src.tokenizers.anchored_spatial_carrier import AnchoredSpatialCarrier
    from src.tokenizers.backward_adaptive_carrier import BackwardAdaptiveCarrier, load_model
    from src.tokenizers.spatial_waveform_dictionary import BANDS, spectral_features

    positions = {'study': [f'site{k}' for k in range(5)]}
    metadata = dict(schema='anchored_spatial_carrier_v1', positions=positions, bands_hz=[list(v) for v in BANDS])
    analysis = {('study', b, k): np.eye(5)[:, k:k+1] for b in ('S', 'O') for k in range(5)}
    synthesis = {key: value.T for key, value in analysis.items()}
    source = AnchoredSpatialCarrier(metadata, {b: np.full(3001, v) for b, v in [('S', .7), ('O', .3)]}, analysis, synthesis)
    source_path = tmp_path/'source.npz'
    source.save(source_path)
    books, settings = {}, {}
    for b in ('S', 'O'):
        for band, (lo, hi) in enumerate(BANDS):
            wave = np.sin(2*np.pi*((lo+hi)/2)*np.arange(400)/200)
            atoms = np.array([wave*0, wave, wave*2])
            books[b, band] = dict(atoms=atoms, centers=spectral_features(atoms[1:], band), counts=np.array([0, 1, 1]))
            settings[b, band] = dict(reference_atom_RMS=.7, maximum_log_step=np.log(2), minimum_ratio=.25, maximum_ratio=4.)
    codec = BackwardAdaptiveCarrier(source, books, {('study', b): np.ones(5) for b in ('S', 'O')}, settings,
        dict(schema='backward_adaptive_carrier_v1', codebook_size=3, adaptive=True))
    x = np.random.default_rng(17).normal(size=(1, 5, 15, 400))
    ids = codec.encode(x, 'study', np.ones((1, 5), bool))
    expected = codec.decode_ids(ids, 'study')
    codec.encode(x*20, 'study', np.ones((1, 5), bool))
    path = tmp_path/'codec.npz'
    codec.save(path, source_path)
    from src.tokenizers.spatial_waveform_dictionary import load_model as load_public
    restored = load_public(path)
    actual = restored.decode_ids(ids, 'study')
    for b in ('S', 'O', 'sum'):
        np.testing.assert_array_equal(expected[b], actual[b])
    with pytest.raises(TypeError):
        restored.decode_ids(ids, 'study', frame_gain=np.ones(5))
