import numpy as np
import pytest
from src.tokenizers.band_exchange_words import ATOMS, encode_slots, fit_books, decode_branch
from src.tokenizers.adaptive_band_allocation import synthesize


def test_paid_index_carries_role_and_complete_waveform():
    rng = np.random.default_rng(19)
    waves = rng.normal(size=(96, 400)).astype(np.float32)
    props = rng.normal(size=(96, 3)).astype(np.float32)
    book = fit_books(waves, props, 1, 94, atom_count=8)
    ids = encode_slots(waves[:15], props[:15], 1, book, "balanced")
    assert np.all((ids > ATOMS) & (ids <= ATOMS + 8))
    sent = np.stack([ids, np.zeros_like(ids)])
    cov = np.repeat(np.eye(2)[None], 5, axis=0)
    result = decode_branch(sent, [book] * 5, cov, "balanced")
    raw = np.zeros((2, 15, 400))
    raw[0] = book["balanced_atoms"][(ids - 1) % ATOMS]
    np.testing.assert_array_equal(result, synthesize(raw, np.stack([np.ones(15), -np.ones(15)]), cov))
    with pytest.raises(ValueError):
        decode_branch(sent.astype(float), [book] * 5, cov, "balanced")


def test_weighted_word_residual_and_window_edges():
    from src.tokenizers.band_exchange_words import neighbor_words, residual_table

    words = np.array([[1] * 14 + [2], [3] + [4] * 14], dtype=np.uint16)
    assert neighbor_words(words, -1)[1, 0] == neighbor_words(words, 1)[0, -1] == 0
    keys = np.array([0, 1, 1, 2])
    values = np.array([[90.0, 7.0], [2.0, 6.0], [4.0, 2.0], [1.0, 3.0]])
    table, mass = residual_table(keys, values, np.array([1.0, 3.0, 1.0, 2.0]), 4, 4)
    np.testing.assert_array_equal(mass, [1, 4, 2, 0])
    np.testing.assert_allclose(table, [[0, 0], [1.25, 2.5], [1 / 3, 1], [0, 0]])
