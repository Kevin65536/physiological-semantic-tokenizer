"""Full-field development audit keeps AS encoding separate from integer decoding."""
from collections import OrderedDict

import numpy as np
import pytest

from experiments.scripts import audit_spatial_development as audit


@pytest.fixture
def scope(monkeypatch, tmp_path):
    calls = []
    codes = {b: np.ones((1, 2, 15), dtype=np.int32) for b in ('S', 'O')}
    class Encoder:
        def encode(self, values, dataset, observed):
            calls.append('encode')
            assert values.shape == (1, 2, 15, 400)
            np.testing.assert_array_equal(observed, [[True, False]])
            return codes
    class Decoder:
        metadata = {'positions': {'fixture': ['a', 'b']}}
        def decode_ids(self, ids, dataset):
            for b in ('S', 'O'):
                np.testing.assert_array_equal(ids[b], codes[b])
            return {b: np.ones((1, 2, 15, 400)) for b in ('S', 'O', 'sum')}
    def fields(*args, **kwargs):
        calls.append('read_X')
        assert kwargs['allowed_splits'] == ('validation',)
        return {'x': np.zeros((6000, 2)), 's': np.full((6000, 2), 999.)}, np.array([True, False]), np.array([False, True])
    monkeypatch.setattr(audit, 'base_tokenizer_joint_eeg', fields)
    monkeypatch.setattr(audit, 'base_tokenizer_batch', lambda *a, **kw: {})
    monkeypatch.setattr(audit, 'joint_reconstruction_metrics', lambda *a, **kw: {'metrics': {'nmse': np.ones((1, 3, 1))}, 'support': {'valid': np.ones(1, bool)}})
    monkeypatch.setattr(audit, 'edge_metrics', lambda *a: {})
    audit.STATE.clear()
    audit.STATE.update(rows=[{'positions': ['a', 'b']}], scales={'positions': {'eeg': {'fixture': {'a': {'scale': 2.}}}}},
        encoder=Encoder(), model=Decoder(), reused={}, output=tmp_path, jobs={'fixture': [(0, 0, [0])]},
        signature={'fixture': 'immutable'}, threads=1, checkpoint=tmp_path/'decoder.npz')
    yield calls, codes
    audit.STATE.clear()


def test_forward_uses_full_array_X_AS_encoder_and_observation_only_support(scope):
    calls, codes = scope
    _, saved, _ = audit.evaluate_crop(audit.STATE['model'], 'fixture', (0, 0, [0]), OrderedDict())
    assert calls == ['read_X', 'encode']
    assert saved is codes
    audit.STATE['reused'][(0, 0)] = codes
    audit.evaluate_crop(audit.STATE['model'], 'fixture', (0, 0, [0]), OrderedDict())
    assert calls == ['read_X', 'encode']


def test_both_disk_replays_bypass_encoder_and_match_all_metrics(scope, monkeypatch):
    calls, _ = scope
    loaded = []
    def reload(path):
        loaded.append(str(path))
        return audit.STATE['model']
    monkeypatch.setattr(audit, 'load_checkpoint', reload)
    for phase in ('forward', 'replay', 'replay2'):
        audit.worker(('fixture', phase))
    assert calls == ['read_X', 'encode']
    assert len(loaded) == 2
    out = audit.STATE['output']/'spatial/fixture'
    with np.load(out/'forward_0000.npz') as forward, np.load(out/'replay2_0000.npz') as replay:
        for key in ('nmse', 'support_valid', 'ids_S', 'ids_O'):
            np.testing.assert_array_equal(forward[key], replay[key])


def test_continuous_carrier_control_respects_frequency_roles_and_missing_sites():
    from experiments.scripts.audit_spatial_tokenizer_distortion import projected_continuous, temporary_wave_bank
    from src.tokenizers.spatial_waveform_dictionary import SpatialWaveformDictionary

    t = np.arange(6000) / 200
    signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 45 * t)
    x = np.tile(signal, (1, 5, 1)).reshape(1, 5, 15, 400)
    meta = dict(codebook_size=32, positions={'fixture': list('abcde')}, output_projection_hz=[1, 45])
    weights = {('fixture', branch, band): np.ones((1, 5)) for branch in ('S', 'O') for band in range(5)}
    base = SpatialWaveformDictionary(meta, {}, {}, {}, weights)
    available = np.array([True, True, True, True, False])
    synthetic = dict(S=x, O=x * .5)
    carrier, lookup = temporary_wave_bank(base, 'fixture', synthetic, available)
    for branch in ('S', 'O'):
        np.testing.assert_array_equal(lookup[branch][0, 4], 0)
    result = carrier.decode_ids(lookup, 'fixture')
    # The 8–13 Hz source is observed. Every site receives its carrier on synthesis.
    np.testing.assert_allclose(result['S'], np.broadcast_to(np.sin(2*np.pi*10*t), (1, 5, 6000)).reshape(x.shape), atol=1e-12)
    direct = projected_continuous({k:v[:,0] for k,v in synthetic.items()})
    np.testing.assert_allclose(result['sum'][:,0], direct['sum'], atol=1e-12)


def test_masked_voltage_floor_is_exact_on_full_windows_and_brackets_partial_windows():
    from experiments.scripts.audit_spatial_tokenizer_distortion import band_limited_bounds
    t = np.arange(6000) / 200
    x = np.tile(np.sin(2*np.pi*10*t) + 2*np.sin(2*np.pi*50*t), (2, 3, 1))
    mask = np.ones((2,6000), bool)
    mask[1,:20] = False
    x[1,:,:20] = np.nan
    variance = np.ones((2,3))
    lower,upper = band_limited_bounds(x, mask, variance)
    np.testing.assert_allclose(lower[0],2.,atol=1e-12)
    np.testing.assert_array_equal(lower[0],upper[0])
    np.testing.assert_array_equal(lower[1],0.)
    assert (upper[1]>0).all()
