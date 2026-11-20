"""Synthetic-only counterfactuals for matched rate/allocation; no data I/O."""
import pickle

import numpy as np
import pytest

from src.tokenizers.adaptive_observation_detail import AdaptiveObservationDetail as Adaptive
from src.tokenizers.factorized_observation import FactorizedObservationTokenizer as Codec


SHAPES = {'eeg': (1, 400), 'hb': (2, 20)}
MODES = ('conditional', 'independent')


def noise(modality, n=48, seed=20):
    return np.random.default_rng(seed).normal(size=(n, *SHAPES[modality])).astype('f4')


def hb_groups(n, seed):
    # Two levels define the slow groups. On the low group, only joint DC
    # varies. The high group has six variable, zero-level/zero-slope modes.
    rng = np.random.default_rng(seed)
    low = np.tile((-20 + rng.uniform(-1, 1, n))[:, None, None], (1, 2, 20))
    basis = np.cos(np.pi * np.array([2, 4, 6])[:, None] * (np.arange(20) + .5) / 20)
    high = 20 + np.einsum('nck,kt->nct', rng.normal(size=(n, 2, 3)), basis)
    return np.concatenate((low, high))


def test_exact_local_and_pooled_equivalence_to_individually_refitted_flat_codecs():
    # Reference exact integers, not just same reported total. No fixed-width
    # vs water-filling or unequal quantizer calibration can confound this pair.
    x, held = hb_groups(52, 1), hb_groups(7, 4)
    base = Codec('hb', n_patterns=2, detail_count=13, detail_bits=1).fit(x)
    a = Adaptive(base, 14).fit(x)
    ptrain = base.encode(x, 'prototype')['pattern']
    pheld = base.encode(held, 'prototype')['pattern']
    assert len(set(ptrain)) == 2 and not (ptrain[:52] == ptrain[-1]).any()
    layouts = [a.detail_layout(p) for p in range(2)]
    assert all(sum(l['widths']) == 13 for l in layouts)
    assert sorted(len(l['indices']) for l in layouts)[0] == 1
    assert sorted(len(l['indices']) for l in layouts)[1] > 1
    for mode in MODES:
        codes = a.encode(held, mode)
        np.testing.assert_array_equal(codes['pattern'], pheld)
        y = a.decode(codes, mode)
        for p in range(2):
            train_rows = ptrain == p if mode == 'conditional' else np.ones(len(x), dtype=bool)
            rows = np.flatnonzero(pheld == p)
            # A K=1 reference flat codec spends *exactly* the 13 residual bits.
            ref = Codec('hb', n_patterns=1, detail_count=13, detail_bits=1).fit(x[train_rows])
            assert a.detail_layout(p, mode)['widths'] == ref.rate('flat')['detail_widths']
            reference = ref.encode(held[rows], 'flat')
            for i, row in enumerate(rows):
                expected_bytes = ref.pack({'detail': reference['detail'][i:i + 1]}, 'flat')
                assert codes['detail'][row].tobytes() == expected_bytes
            np.testing.assert_array_equal(y[rows], ref.decode(reference, 'flat'))
        wrong_route = {**codes, 'pattern': 1 - pheld}  # same actual 13-bit strings
        intervention = a.decode(wrong_route, mode)
        if mode == 'independent':
            np.testing.assert_array_equal(intervention, y)
        else:
            assert np.mean((intervention - held) ** 2) > 10 * np.mean((y - held) ** 2)
    assert base.rate('flat')['bits_per_item'] == 14  # no-ID/full-budget flat, not the attached-ID arm


@pytest.mark.parametrize('modality', SHAPES)
@pytest.mark.parametrize('routing', ('pattern', 'waveform'))
def test_pack_actual_rate_readout_snapshot_and_batch_replay(modality, routing):
    train, held = noise(modality), noise(modality, 7, 13)
    base = Codec(modality, n_patterns=5, seed=7, routing=routing).fit(train)
    base_before = pickle.dumps(base)
    a = Adaptive(base, 14).fit(train, partition='train')
    assert pickle.dumps(base) == base_before
    np.testing.assert_array_equal(a.pattern_counts_, base.pattern_counts_)
    snapshot = pickle.dumps(a)
    for mode in MODES:
        codes = a.encode(held, mode)
        assert set(codes) == {'pattern', 'detail'} and codes['detail'].dtype == np.uint8
        assert codes['detail'].shape == (7, 2) and not (codes['detail'][:, -1] & 31).any()
        rate = a.rate(mode)
        assert rate == dict(pattern_bits=3, detail_bits=11, bits_per_item=14, bits_per_second=7)
        bits = ''.join(f'{p:03b}' + ''.join(f'{byte:08b}' for byte in row)[:11]
                       for p, row in zip(codes['pattern'], codes['detail']))
        padded = bits + '0' * 6  # 7*14=98 payload bits; 104 bits including batch-end padding
        payload = a.pack(codes, mode)
        assert payload == int(padded, 2).to_bytes(13, 'big')
        decoder = pickle.loads(snapshot)
        replay = decoder.unpack(payload, 7, mode)
        # Use a different item order between training/decoding; never use a
        # cached input window to turn an integer roundtrip into an identity.
        encoded_backwards = a.encode(held[::-1], mode)
        for field in codes:
            np.testing.assert_array_equal(replay[field], codes[field])
            np.testing.assert_array_equal(encoded_backwards[field][::-1], codes[field])
        y = decoder.decode(replay, mode)
        np.testing.assert_array_equal(y, a.decode(codes, mode))
        np.testing.assert_array_equal(a.readout(codes, mode), base.observation_features(y))
        for start in range(7):
            single = a.encode(held[start:start + 1], mode)
            np.testing.assert_array_equal(a.decode(single, mode), y[start:start + 1])
        empty = a.encode(held[:0], mode)
        assert a.pack(empty, mode) == b''
        assert a.decode(a.unpack(b'', 0, mode), mode).shape == held[:0].shape
    assert pickle.dumps(a) == snapshot
    # Caller can reuse/refit its copy without moving an ID in this codec.
    original = a.encode(held, 'conditional')
    base.fit(np.zeros_like(train))
    for field in original:
        np.testing.assert_array_equal(a.encode(held, 'conditional')[field], original[field])


def test_independent_detail_does_not_see_routing_features_while_id_is_still_charged():
    x, held = noise('eeg', 70), noise('eeg', 14, 9)
    models = [Adaptive(Codec('eeg', n_patterns=16, seed=13, routing=route).fit(x), 32).fit(x)
              for route in ('pattern', 'waveform')]
    ca, cb = [a.encode(held, 'independent') for a in models]
    assert (ca['pattern'] != cb['pattern']).any()
    np.testing.assert_array_equal(ca['detail'], cb['detail'])
    np.testing.assert_array_equal(models[0].decode(ca, 'independent'), models[1].decode(cb, 'independent'))
    assert all(a.rate('independent')['detail_bits'] == 28 for a in models)


@pytest.mark.parametrize('modality,bits', [('eeg', 14), ('hb', 32), ('hb', 64), ('eeg', 256),
                                           ('hb', 512), ('hb', 964), ('eeg', 1024)])
def test_full_advertised_budget_spent_for_every_layout_including_unused_ids(modality, bits):
    x = noise(modality, 5)
    base = Codec(modality, n_patterns=16).fit(x)
    a = Adaptive(base, bits).fit(x)
    assert a.pattern_counts_.tolist() == [1] * 5 + [0] * 11
    for mode in MODES:
        for p in (0, 2, 15):
            layout = a.detail_layout(p, mode)
            assert len(layout['indices']) == len(layout['widths']) == len(set(layout['indices']))
            assert min(layout['widths']) >= 1 and max(layout['widths']) <= 24
            assert sum(layout['widths']) + 4 == bits
        codes = a.encode(x, mode)
        # Exercise an otherwise unoccupied ID and arbitrary maximum unsigned
        # detail symbols. Zero byte-alignment padding is NOT available as data.
        codes['pattern'][:] = 15
        codes['detail'][:] = 255
        pad = -(bits - 4) % 8
        if pad:
            codes['detail'][:, -1] = 256 - (1 << pad)
        payload = a.pack(codes, mode)
        assert len(payload) == (len(x) * bits + 7) // 8
        recovered = a.unpack(payload, len(x), mode)
        np.testing.assert_array_equal(recovered['detail'], codes['detail'])
        assert np.isfinite(a.decode(recovered, mode)).all()
    assert a.detail_layout(15, 'conditional') == a.detail_layout(0, 'independent')


def test_exact_twenty_four_bit_extrema_including_heldout_saturation():
    x = np.stack([np.full((2, 20), v) for v in (-2., 2.)])
    base = Codec('hb', n_patterns=1).fit(x)
    a = Adaptive(base, 24).fit(x)
    for mode in MODES:
        assert a.detail_layout(0, mode)['widths'] == (24,)
        c = a.encode(x * np.finfo(np.float32).max, mode)
        assert c['detail'].tobytes() == b'\x00\x00\x00\xff\xff\xff'
        np.testing.assert_allclose(a.decode(c, mode), x, atol=1e-14)


def test_no_detail_has_true_conditional_mean_vs_pooled_attached_id_and_zero_bit_case():
    x = hb_groups(9, 0)
    for k in (1, 2):
        base = Codec('hb', n_patterns=k).fit(x)
        a = Adaptive(base, k - 1).fit(x)
        local = a.encode(x, 'conditional')
        attached = a.encode(x, 'independent')
        np.testing.assert_array_equal(local['pattern'], attached['pattern'])
        assert local['detail'].shape == attached['detail'].shape == (len(x), 0)
        y = a.decode(local, 'conditional')
        z = a.decode(attached, 'independent')
        np.testing.assert_allclose(z, np.tile(x.mean(0), (len(x), 1, 1)), atol=1e-14)
        for p in np.unique(local['pattern']):
            rows = local['pattern'] == p
            np.testing.assert_allclose(y[rows][0], x[rows].mean(0), atol=1e-14)
        for mode in MODES:
            payload = a.pack(local, mode)
            assert len(payload) == (len(x) * (k - 1) + 7) // 8
            replay = a.unpack(payload, len(x), mode)
            np.testing.assert_array_equal(replay['pattern'], local['pattern'])


def test_strict_fields_partition_bounds_and_padding():
    x = noise('hb', 10)
    base = Codec('hb', n_patterns=3)
    with pytest.raises(ValueError, match='fit'):
        Adaptive(base, 14)
    base.fit(x)
    for invalid in (1, 963, True, 20.1):  # p=2, 40*24=960 max detail bits
        with pytest.raises((ValueError, TypeError)):
            Adaptive(base, invalid)
    a = Adaptive(base, 14)
    with pytest.raises(RuntimeError, match='fit'):
        a.encode(x)
    for bad in (x[:0], x[:, 0], x * np.nan):
        with pytest.raises(ValueError):
            a.fit(bad)
    a.fit(x)
    before = pickle.dumps(a)
    with pytest.raises(ValueError, match='partition'):
        a.fit(x, partition='test')
    assert pickle.dumps(a) == before
    d = a.encode(x[:1])
    for bad in ({**d, 'phase': x[:, 0]}, {'pattern': d['pattern']},
                {**d, 'pattern': np.array([3])}, {**d, 'pattern': np.array([-1])},
                {**d, 'pattern': d['pattern'][:, None]}, {**d, 'detail': d['detail'][:, :1]},
                {**d, 'pattern': d['pattern'].astype(float)}, {**d, 'detail': d['detail'].astype(float)},
                {**d, 'detail': d['detail'].astype(int)}, {**d, 'detail': np.array([[0, 1]], dtype='u1')}):
        with pytest.raises((ValueError, TypeError)):
            a.decode(bad)
    for bad_payload, why in ((b'\xc0\x00', 'pattern'), (b'\x00\x01', 'padding'), (b'\x00', 'length')):
        with pytest.raises(ValueError, match=why):
            a.unpack(bad_payload, 1)
    with pytest.raises(ValueError, match='mode'):
        a.decode(d, 'flat')
