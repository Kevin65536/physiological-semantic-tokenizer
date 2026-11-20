"""Synthetic checks only: no G0 arrays, measured loaders or launchers."""
import pickle

import numpy as np
import pytest

from src.tokenizers.factorized_observation import FactorizedObservationTokenizer as Codec


SHAPE = {'eeg': (1, 400), 'hb': (2, 20)}
MODES = ('prototype', 'full', 'flat', 'flat_pattern')


def noise(modality, n=40, seed=29):
    return np.random.default_rng(seed).normal(size=(n, *SHAPE[modality])).astype('f4')


def test_eeg_features_exact_band_edges_nyquist_and_no_phase_or_amplitude_charge():
    c = Codec('eeg')
    t = np.arange(400) / 200
    tones = np.cos(2 * np.pi * np.array([.5, 4, 8, 13, 30, 45])[:, None] * t)
    assert c.pattern_feature_names[0] == 'eeg_rel_0p5_4hz'
    np.testing.assert_allclose(c.pattern_features(tones[:, None]), np.eye(6), atol=1e-24)
    # Equal amplitudes: interior cosine variance 1/2, Nyquist cosine variance 1.
    mix = tones[1] + (-1.) ** np.arange(400)
    x = np.stack((mix, 100 - 7 * np.roll(mix, 19), np.ones(400)))[:, None]
    f = c.pattern_features(x)
    np.testing.assert_allclose(f[:2], np.tile([0, 1 / 3, 0, 0, 0, 2 / 3], (2, 1)), atol=1e-14)
    np.testing.assert_array_equal(f[2], np.zeros(6))
    # More generally, varying every independent rFFT phase keeps bin energy,
    # not arbitrary subwindow envelopes. No imaginary DC or Nyquist.
    x = noise('eeg', 1).astype(float)
    spectrum = np.fft.rfft(x)
    phase = np.random.default_rng(2).uniform(-np.pi, np.pi, spectrum.shape)
    phase[..., [0, -1]] = 0
    randomized = np.fft.irfft(spectrum * np.exp(1j * phase), n=400)
    assert not np.allclose(x, randomized)
    np.testing.assert_allclose(c.pattern_features(x), c.pattern_features(randomized), atol=1e-14)


def test_hb_joint_named_physical_units_and_common_readout():
    c = Codec('hb')
    t = (np.arange(20) - 9.5) / 10
    x = np.stack((2 + 3 * t, -5 - .5 * t))[None]
    assert c.pattern_feature_names == ('hbo_mean', 'hbr_mean', 'hbo_slope_per_s', 'hbr_slope_per_s')
    np.testing.assert_allclose(c.pattern_features(x), [[2, -5, 3, -.5]])
    np.testing.assert_allclose(c.observation_features(x[:, ::-1]), [[-5, 2, -.5, 3, 0, 0]], atol=1e-14)


@pytest.mark.parametrize('modality', SHAPE)
@pytest.mark.parametrize('routing', ('pattern', 'waveform'))
def test_integer_payload_replay_exact_bit_order_and_matched_readouts(modality, routing):
    x = noise(modality)
    c = Codec(modality, n_patterns=5, detail_count=2, detail_bits=5, seed=11, routing=routing).fit(x)
    state = pickle.dumps(c)
    for mode, charge in zip(MODES, (3, 13, 13, 3)):
        rate = c.rate(mode)
        width = rate['detail_widths']
        assert rate['bits_per_item'] == rate['pattern_bits'] + sum(width) == charge
        assert rate['bits_per_second'] == charge / 2
        codes = c.encode(x[:7], mode)
        expected = {'prototype': {'pattern'}, 'full': {'pattern', 'detail'}, 'flat': {'detail'}, 'flat_pattern': {'detail'}}
        assert set(codes) == expected[mode]
        assert all(v.dtype == np.int32 for v in codes.values())
        parts = []
        for i in range(7):
            if 'pattern' in codes:
                assert 0 <= codes['pattern'][i] < 5
                parts.append(f'{codes["pattern"][i]:03b}')
            if 'detail' in codes:
                assert len(codes['detail'][i]) == len(width)
                for symbol, w in zip(codes['detail'][i], width):
                    assert 0 <= symbol < (1 << w)
                    parts.append(f'{symbol:0{w}b}')
        bitstring = ''.join(parts)
        assert len(bitstring) == charge * 7
        padded = bitstring + '0' * (-len(bitstring) % 8)
        payload = c.pack(codes, mode)
        assert payload == int(padded, 2).to_bytes(len(padded) // 8, 'big')
        restored = pickle.loads(state)  # Decoder shares train state, not input rows or mutable batch context.
        replay = restored.unpack(payload, 7, mode)
        for k, value in codes.items():
            np.testing.assert_array_equal(replay[k], value)
        y = restored.decode(replay, mode)
        assert y.shape == x[:7].shape and np.isfinite(y).all()
        np.testing.assert_array_equal(c.decode(codes, mode), y)
        np.testing.assert_array_equal(c.readout(codes, mode), c.observation_features(y))
        assert c.readout(codes, mode).shape == (7, len(c.readout_feature_names))
    assert pickle.dumps(c) == state  # No held-out update or caching the last input.


@pytest.mark.parametrize('modality,m,b,total', [('eeg', 2, 5, 14), ('eeg', 63, 4, 256),
                           ('eeg', 127, 4, 512), ('hb', 2, 5, 14), ('hb', 7, 4, 32), ('hb', 15, 4, 64)])
def test_candidate_budgets_charge_all_coefficients_and_exactly_reassign_id_bits(modality, m, b, total):
    c = Codec(modality, n_patterns=16, detail_count=m, detail_bits=b).fit(noise(modality, 24))
    assert c.rate()['detail_widths'] == (b,) * m
    assert c.rate()['pattern_bits'] == c.rate('prototype')['bits_per_item'] == 4
    assert c.rate('flat')['pattern_bits'] == 0
    assert c.rate()['bits_per_item'] == c.rate('flat')['bits_per_item'] == total
    assert c.rate('flat_pattern')['bits_per_item'] == 4


def test_flat_spends_all_four_saved_bits_on_real_precision_not_an_appended_label():
    # Only one TRAIN direction carries any material variation (joint Hb DC).
    amp = np.linspace(-1, 1, 129)
    x = np.tile(amp[:, None, None], (1, 2, 20))
    c = Codec('hb', n_patterns=16, detail_count=2, detail_bits=5).fit(x)
    assert c.rate('flat')['detail_widths'] == (14,)
    assert c.rate('flat_pattern')['detail_widths'] == (4,)
    detail = c.encode(x, 'flat')['detail']
    assert np.unique(detail).size == 129 > 2 ** c.detail_bits
    assert detail.min() == 0 and detail.max() == 2 ** 14 - 1
    np.testing.assert_allclose(c.decode({'detail': detail}, 'flat'), x, atol=7e-5)


def test_phase_blind_slow_id_has_distinct_holdout_test_from_using_routing_bits():
    # Balanced quadrature training -> carrier phases cancel in each slow prototype.
    t = np.arange(400) / 200
    phase = np.arange(16) * 2 * np.pi / 16
    def rhythms(offset):
        return np.concatenate([np.sin(2 * np.pi * f * t + phase[:, None] + offset)
                               for f in (6., 23.)])[:, None]
    train, held = rhythms(0.), 1.3 * rhythms(.107)
    c = Codec('eeg', n_patterns=2, detail_count=12, detail_bits=4, seed=9).fit(train)
    p_train = c.encode(train, 'prototype')['pattern']
    p = c.encode(held, 'prototype')['pattern']
    # Test a known synthetic rhythm on unseen phase and gain; no phase is an
    # ID feature. This is a construction test, not evidence for task semantics.
    classes = np.repeat([0, 1], len(phase))
    table = [np.bincount(classes[p_train == j]).argmax() for j in range(2)]
    np.testing.assert_array_equal(np.asarray(table)[p], classes)
    for j in range(2):
        np.testing.assert_allclose(c.pattern_centers_[j], c.pattern_features(train[p_train == j]).mean(0))
        np.testing.assert_allclose(c.decode({'pattern': np.array([j])}, 'prototype'),
                                   train[p_train == j].mean(0, keepdims=True), atol=1e-14)
    # Separate mechanistic counterfactual: exactly same integers except ID.
    full = c.encode(held)
    np.testing.assert_array_equal(full['pattern'], p)
    truth = np.mean((c.decode(full) - held) ** 2)
    wrong = {**full, 'pattern': 1 - p}
    wrong_error = np.mean((c.decode(wrong) - held) ** 2)
    proto_error = np.mean((c.decode({'pattern': p}, 'prototype') - held) ** 2)
    assert truth < .2 * proto_error and wrong_error > 3 * truth
    assert c.rate('full')['bits_per_item'] == c.rate('flat')['bits_per_item']


def test_blind_waveform_routing_control_can_keep_polarity_that_pattern_discards():
    u = noise('eeg', 1)
    x = np.concatenate((u, -u, u, -u))
    slow = Codec('eeg', n_patterns=2, detail_count=3, detail_bits=2).fit(x)
    blind = Codec('eeg', n_patterns=2, detail_count=3, detail_bits=2, routing='waveform').fit(x)
    slow_codes = slow.encode(x, 'prototype')
    blind_codes = blind.encode(x, 'prototype')
    assert len(np.unique(slow_codes['pattern'])) == 1
    assert len(np.unique(blind_codes['pattern'])) == 2
    np.testing.assert_allclose(slow.decode(slow_codes, 'prototype'), 0, atol=1e-13)
    np.testing.assert_allclose(blind.decode(blind_codes, 'prototype'), x, atol=1e-13)
    for mode in MODES:
        assert slow.rate(mode)['bits_per_item'] == blind.rate(mode)['bits_per_item']
    # Unrouted stream and raw physical-property reader do not use KMeans.
    np.testing.assert_array_equal(slow.encode(x, 'flat')['detail'], blind.encode(x, 'flat')['detail'])
    np.testing.assert_array_equal(slow.pattern_features(x), blind.pattern_features(x))


@pytest.mark.parametrize('modality', SHAPE)
def test_shape_seed_batch_order_and_dictionary_freeze(modality):
    x, held = noise(modality, 64), noise(modality, 9, seed=204)
    c, other = [Codec(modality, n_patterns=7, seed=31).fit(x, partition='train') for _ in range(2)]
    np.testing.assert_array_equal(c.pattern_counts_, other.pattern_counts_)
    assert c.pattern_counts_.shape == (7,) and c.pattern_counts_.sum() == len(x)
    state = pickle.dumps(c)
    for mode in MODES:
        left, right = c.encode(held, mode), other.encode(held[::-1], mode)
        for field in left:
            np.testing.assert_array_equal(left[field], right[field][::-1])
            np.testing.assert_array_equal(left[field][:1], c.encode(held[:1], mode)[field])
        empty = c.encode(held[:0], mode)
        assert c.pack(empty, mode) == b''
        assert c.decode(c.unpack(b'', 0, mode), mode).shape == held[:0].shape
    with pytest.raises(ValueError, match='partition'):
        c.fit(held, partition='validation')
    assert state == pickle.dumps(c)


def test_saturation_including_float32_extrema_and_degenerate_ranges():
    x = np.stack([np.full((2, 20), level) for level in (-1., 1.)])
    c = Codec('hb', n_patterns=1, detail_count=1, detail_bits=5).fit(x)
    extreme = x * np.finfo(np.float32).max
    for mode in ('full', 'flat'):
        codes = c.encode(extreme, mode)
        np.testing.assert_array_equal(codes['detail'], [[0], [31]])
        np.testing.assert_allclose(c.decode(codes, mode), x, atol=1e-14)
    # More requested IDs than unique training values. Empty IDs never make
    # normalization NaN or silently reduce the charge; fallback stays finite.
    zero = np.zeros((5, 2, 20))
    constant = Codec('hb', n_patterns=9, detail_count=40, detail_bits=16).fit(zero)
    assert constant.pattern_counts_.tolist() == [5] + [0] * 8
    assert constant.rate('flat')['bits_per_item'] == 4 + 40 * 16
    for mode in MODES:
        codes = constant.encode(extreme, mode)
        if 'pattern' in codes:
            codes['pattern'][:] = 8
        np.testing.assert_array_equal(constant.decode(codes, mode), np.zeros_like(x))
    with pytest.raises(ValueError, match='moment range'):
        c.encode(x * np.finfo(float).max)


def test_zero_bit_stream_and_strict_integer_validation():
    x = noise('hb', 6)
    empty = Codec('hb', n_patterns=1, detail_count=0).fit(x)
    for mode in MODES:
        assert empty.rate(mode)['bits_per_item'] == 0
        codes = empty.encode(x, mode)
        assert empty.pack(codes, mode) == b''
        np.testing.assert_allclose(empty.decode(empty.unpack(b'', len(x), mode), mode),
                                   np.tile(x.astype(float).mean(0, keepdims=True), (len(x), 1, 1)))
    c = Codec('hb', n_patterns=3, detail_count=2, detail_bits=4).fit(x)
    d = c.encode(x)
    for bad in ({**d, 'scale': np.ones(len(x))}, {'detail': d['detail']},
                {**d, 'pattern': d['pattern'][:, None]}, {**d, 'detail': d['detail'][:, :1]},
                {**d, 'detail': d['detail'][:-1]}, {**d, 'pattern': np.full(len(x), 3)},
                {**d, 'detail': np.full_like(d['detail'], 16)}, {**d, 'detail': -1 - d['detail']}):
        with pytest.raises(ValueError):
            c.decode(bad)
    for bad_type in (float, bool, complex, object):
        with pytest.raises(TypeError, match='integer'):
            c.decode({**d, 'pattern': d['pattern'].astype(bad_type)})
    with pytest.raises(ValueError, match='pattern'):
        c.unpack(b'\xc0', 1, 'prototype')  # leading 11=3 is outside K=3
    with pytest.raises(ValueError, match='padding'):
        c.unpack(b'\x01', 1, 'prototype')
    with pytest.raises(ValueError, match='length'):
        c.unpack(b'\x00\x00', 1, 'prototype')


@pytest.mark.parametrize('modality', SHAPE)
def test_reject_wrong_shape_nonfinite_unfitted_or_impossible_configuration(modality):
    x = noise(modality, 6)
    c = Codec(modality, n_patterns=2)
    with pytest.raises(RuntimeError, match='fit'):
        c.encode(x)
    for invalid in (x[0], x.transpose(0, 2, 1), x[:, :, :-1], x * np.nan, x.astype(complex), x[:0]):
        with pytest.raises(ValueError):
            c.fit(invalid)
    c.fit(x)
    with pytest.raises(ValueError, match='mode'):
        c.encode(x, mode='unmetered')
    for kwargs in ({'detail_bits': 0}, {'detail_count': -1}, {'detail_count': x[0].size + 1},
                   {'n_patterns': 0}, {'seed': -1}, {'n_patterns': True}, {'detail_bits': 4.5},
                   {'routing': 'labels'}):
        with pytest.raises((ValueError, TypeError)):
            Codec(modality, **kwargs)
