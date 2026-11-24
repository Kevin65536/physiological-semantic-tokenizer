import numpy as np
import pytest

from experiments.scripts.audit_montage_word_search import decode_integers, integer_atoms, pack14, unpack14
from src.tokenizers.cross_branch_innovations import decode_words, paid_fields


def test_real_alphabet_transport_and_padding():
    words = np.array([0, 1, 5115, 5116, 7163, 15350, 12], np.uint16).reshape(1, 7)
    payload = pack14(words)
    assert len(payload) == 13
    np.testing.assert_array_equal(unpack14(payload, words.shape), words)
    with pytest.raises(ValueError): pack14(np.array([15351], np.uint16))
    with pytest.raises(ValueError): unpack14(payload[:-1] + bytes([payload[-1] | 128]), words.shape)


def test_fresh_paid_lookup_and_frozen_waveform_match():
    rng = np.random.default_rng(81)
    old = {b: [dict(balanced_atoms=rng.normal(size=(3, 400)).astype(np.float32),
                    balanced_gain=np.array([1, 1.2, 1.5], np.float32)) for _ in range(5)] for b in 'SO'}
    new = {b: [dict(capacity=np.int32(2047), innovation_atoms=rng.normal(size=(3, 400)).astype(np.float32))
               for _ in range(5)] for b in 'SO'}
    state = dict(scale=rng.uniform(.2, 2, (5, 4)), joint=np.repeat((np.eye(4) + .2)[None], 5, 0),
                 pilot_gain=np.ones((5, 4)), pilot_noise=np.ones((5, 4)) * .3,
                 innovation_mean_gain=np.ones((5, 4)) * .9, innovation_mean_noise=np.ones((5, 4)) * .2)
    sent = {'S': np.array([[1 + 1023 + 1] * 15, [5116 + 2047 * 3 + 2] * 15], np.uint16),
            'O': np.array([[5116 + 2047 * 2 + 1] * 15, [1 + 1023 * 4 + 2] * 15], np.uint16)}
    fresh = integer_atoms(sent, old, new, state)
    official = paid_fields(sent, old, new, state['scale'], 'innovation', False)
    for a, b in zip(fresh, official): np.testing.assert_array_equal(a, b)
    expected = decode_words(sent, old, new, state, state, 'innovation', False)
    actual = decode_integers(sent, old, new, state)
    for b in actual: np.testing.assert_array_equal(actual[b], expected[b])
    sent['S'][0, 0] = 5116
    with pytest.raises(ValueError): integer_atoms(sent, old, new, state)
