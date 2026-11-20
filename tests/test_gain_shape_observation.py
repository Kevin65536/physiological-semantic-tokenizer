"""EEG gain/shape on known synthetic waveforms only; no measured loaders."""
import pickle

import numpy as np
import pytest

from src.tokenizers.adaptive_observation_detail import AdaptiveObservationDetail
from src.tokenizers.factorized_observation import FactorizedObservationTokenizer as Codec
from src.tokenizers.gain_shape_observation import GainShapeObservation as GainShape


MODES = ('conditional', 'independent')


def ac_rms(x):
    return np.sqrt(np.mean((x - x.mean(axis=-1, keepdims=True)) ** 2, axis=-1))[:, 0]


def spectral_panel():
    t = np.arange(400) / 200
    ratios = np.linspace(.15, 1.4, 16)
    def shape(ratio, phase):
        y = np.sin(2 * np.pi * 10 * t + phase) + ratio * np.cos(2 * np.pi * 37 * t - .7 * phase)
        y -= y.mean()
        return y / np.sqrt(np.mean(y ** 2))
    train = np.stack([amp * shape(r, phase) for r in ratios for amp in (2 ** -9, 2 ** -5, .5, 8.)
                      for phase in np.arange(4) * np.pi / 2])[:, None]
    # New phase and amplitudes. Within each pair exactly the same spectral
    # shape spans a factor of 1024 or 128 in amplitude (not different bands).
    held = np.stack([amp * shape(ratios[i], .37) for amp, i in
                     ((.004, 3), (.004 * 1024, 3), (.01, 10), (1.28, 10))])[:, None]
    return train, held


def bandpower(x):
    fft = np.fft.rfft(x - x.mean(axis=-1, keepdims=True), axis=-1)
    return np.sum(np.abs(fft[:, 0, 60:90]) ** 2, axis=1)  # [30,45) Hz; common factors cancel in a ratio


@pytest.mark.parametrize('budget', (256, 512, 1024))
def test_heldout_weak_and_strong_same_band_exact_gain_account_and_original_route(budget):
    x, held = spectral_panel()
    base = Codec('eeg', n_patterns=16, seed=0).fit(x)
    g = GainShape(base, budget).fit(x)
    absolute = AdaptiveObservationDetail(base, budget).fit(x)
    assert np.allclose(g.positive_gain_range_, (2 ** -9, 8.))
    state = pickle.dumps(g)
    for mode in MODES:
        codes = g.encode(held, mode)
        np.testing.assert_array_equal(codes['pattern'], base.encode(held, 'prototype')['pattern'])
        y = g.decode(codes, mode)
        np.testing.assert_allclose(ac_rms(y), g.gain_lut_[codes['gain']], rtol=1e-12)
        np.testing.assert_allclose(y.mean(axis=(1, 2)), g.level_lut_[codes['level']], atol=2e-14)
        # Half a log-grid step (no extrapolation) bounds global gain error.
        step = np.log(8. / (2 ** -9)) / 62
        assert np.max(np.abs(np.log(ac_rms(y) / ac_rms(held)))) <= step / 2 + 1e-12
        for i in (0, 2):
            assert codes['pattern'][i] == codes['pattern'][i + 1]
            np.testing.assert_array_equal(codes['detail'][i], codes['detail'][i + 1])
        db = 10 * np.log10(bandpower(y) / bandpower(held))
        assert np.max(np.abs(db)) < 1.5
        # Controlled synthetic demonstration of the hypothesized failure;
        # not a conclusion about measured or independent physiological data.
        old = absolute.decode(absolute.encode(held, mode), mode)
        assert np.min(10 * np.log10(bandpower(old)[::2] / bandpower(held)[::2])) > 10.
        assert g.rate(mode) == dict(pattern_bits=4, level_bits=8, gain_bits=6, detail_bits=budget - 18,
                                    bits_per_item=budget, bits_per_second=budget / 2)
        assert sum(g.detail_layout(int(codes['pattern'][0]), mode)['widths']) == budget - 18
        # MSB order independently reconstructed: four fields, not an aligned
        # header dict interpreted as if it occupied 18 bytes or had free RMS.
        parts = []
        for p, level, gain, row in zip(codes['pattern'], codes['level'], codes['gain'], codes['detail']):
            parts.append(f'{p:04b}{level:08b}{gain:06b}' + ''.join(f'{b:08b}' for b in row)[:budget - 18])
        wire = g.pack(codes, mode)
        assert wire == int(''.join(parts), 2).to_bytes(len(held) * budget // 8, 'big')
        decoder = pickle.loads(state)
        replay = decoder.unpack(wire, len(held), mode)
        for field in codes:
            np.testing.assert_array_equal(replay[field], codes[field])
        np.testing.assert_array_equal(decoder.decode(replay, mode), y)
        np.testing.assert_array_equal(g.readout(codes, mode), base.observation_features(y))
        swapped = {**codes, 'pattern': (codes['pattern'] + 1) % 16}
        if mode == 'independent':
            np.testing.assert_array_equal(g.decode(swapped, mode), y)
        else:
            assert not np.allclose(g.decode(swapped, mode), y)
    assert pickle.dumps(g) == state


def test_independent_shape_stream_is_the_same_adaptive_allocator_after_charging_header():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 1, 400)) * rng.uniform(.01, 4, (80, 1, 1)) + rng.uniform(-2, 2, (80, 1, 1))
    held = rng.normal(size=(8, 1, 400)) + rng.normal(size=(8, 1, 1))
    def unit(a):
        return (a - a.mean(axis=2, keepdims=True)) / ac_rms(a)[:, None, None]
    base = Codec('eeg').fit(x)
    g = GainShape(base, 512).fit(x)
    adaptive = AdaptiveObservationDetail(base, 512 - 14).fit(unit(x))
    codes = g.encode(held, 'independent')
    reference = adaptive.encode(unit(held), 'independent')
    np.testing.assert_array_equal(codes['detail'], reference['detail'])
    np.testing.assert_array_equal(g.encode(held, 'conditional')['gain'], codes['gain'])
    np.testing.assert_array_equal(g.encode(held, 'conditional')['level'], codes['level'])
    # Counterfactual packet edit acts through the charged header only.
    force = {**codes, 'level': np.full(8, 255), 'gain': np.full(8, 63)}
    y = g.decode(force, 'independent')
    np.testing.assert_allclose(ac_rms(y), g.gain_lut_[-1])
    np.testing.assert_allclose(y.mean(axis=(1, 2)), g.level_lut_[-1])
    zero = g.decode({**force, 'gain': np.zeros(8, dtype='u1')}, 'independent')
    np.testing.assert_array_equal(zero, np.full_like(held, g.level_lut_[-1]))
    assert g.level_range_ == pytest.approx((x.mean(axis=-1).min(), x.mean(axis=-1).max()))
    assert np.all(np.diff(g.level_lut_) >= 0) and np.all(np.diff(g.gain_lut_) >= 0)
    np.testing.assert_allclose(g.level_lut_, np.sort(np.r_[0., np.quantile(x.mean(axis=-1), np.linspace(0, 1, 255))]), atol=1e-14)


def test_constant_zero_missing_gain_and_extreme_saturation_no_carrier_fabrication():
    x = np.tile(np.repeat([-2., .1, 0., 3.], 5)[:, None, None], (1, 1, 400))
    base = Codec('eeg').fit(x)
    g = GainShape(base, 256).fit(x)
    assert g.positive_gain_range_ is None and (g.gain_lut_ == 0).all()
    np.testing.assert_array_equal(g.level_range_, [-2, 3])
    extreme = np.stack((np.full(400, np.finfo(np.float32).max), np.full(400, -1e70), (-1.) ** np.arange(400)))[:, None]
    for mode in MODES:
        codes = g.encode(x, mode)
        np.testing.assert_array_equal(codes['gain'], 0)
        np.testing.assert_array_equal(g.decode(codes, mode), x)
        held = g.encode(extreme, mode)
        assert (held['gain'] == 0).all()
        np.testing.assert_array_equal(g.level_lut_[held['level']], [3., -2., 0.])
        np.testing.assert_array_equal(g.decode(held, mode)[:, 0, 0], [3., -2., 0.])
        assert g.pack(g.encode(x[:0], mode), mode) == b''
        assert g.decode(g.unpack(b'', 0, mode), mode).shape == (0, 1, 400)
    # Fit with AC: an out-of-training-range positive gain saturates rather
    # than receiving a freshly computed test-window scale. True zero is exact.
    train, _ = spectral_panel()
    g = GainShape(Codec('eeg').fit(train), 256).fit(train)
    codes = g.encode(extreme)
    assert codes['gain'][:2].tolist() == [0, 0]
    for amp, endpoint in ((1e-90, 1), (1e70, 63)):
        wave = amp * (-1.) ** np.arange(400)[None, None]
        assert g.encode(wave)['gain'][0] == endpoint


def test_require_frozen_train_router_valid_budget_and_strict_integer_only_packet():
    x = np.random.default_rng(3).normal(size=(32, 1, 400))
    for invalid in (Codec('eeg'), Codec('eeg', n_patterns=2).fit(x), Codec('eeg', routing='waveform').fit(x),
                    Codec('hb').fit(np.zeros((2, 2, 20)))):
        with pytest.raises(ValueError, match='router'):
            GainShape(invalid, 512)
    base = Codec('eeg').fit(x)
    for bits in (14, 260, 513, True, 512.5):
        with pytest.raises((ValueError, TypeError)):
            GainShape(base, bits)
    g = GainShape(base, 512)
    with pytest.raises(RuntimeError):
        g.encode(x)
    for bad in (x[:0], x[:, 0], x * np.nan):
        with pytest.raises(ValueError):
            g.fit(bad)
    g.fit(x, partition='train')
    snap = pickle.dumps(g)
    with pytest.raises(ValueError, match='partition'):
        g.fit(x, partition='validation')
    assert pickle.dumps(g) == snap
    c = g.encode(x[:1])
    for bad in ({**c, 'level': np.array([256])}, {**c, 'gain': np.array([64])}, {**c, 'pattern': np.array([-1])},
                {**c, 'gain': c['gain'].astype(float)}, {**c, 'level': c['level'][:, None]},
                {**c, 'amp': np.array([1.13])}, {**c, 'detail': c['detail'].astype(float)},
                {**c, 'detail': c['detail'] | 1}, {'pattern': c['pattern'], 'detail': c['detail']}):
        with pytest.raises((ValueError, TypeError)):
            g.decode(bad)
    with pytest.raises(ValueError, match='length'):
        g.unpack(g.pack(c) + b'\x00', 1)
    with pytest.raises(ValueError, match='mode'):
        g.decode(c, 'flat')
    # Neither inference nor a later independent refit of the caller's base
    # router changes this training snapshot.
    base.fit(np.zeros_like(x))
    np.testing.assert_array_equal(c['pattern'], g.encode(x[:1])['pattern'])
    assert pickle.dumps(g) == snap
