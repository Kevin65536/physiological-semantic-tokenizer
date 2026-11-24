import numpy as np
import pytest
from src.tokenizers.predictive_word_order import encode_words, decode_words, fit_site, word_stats


def test_all_symbols_are_reversible_after_branch_reset_and_missing_slots():
    rng = np.random.default_rng(14)
    size = 32
    words = rng.integers(size, size=(8, 15), dtype=np.int32)
    valid = words != 0
    attributes = rng.normal(size=(*words.shape, 2))
    count, total = word_stats(words[valid], attributes[valid], size)
    state, maps = fit_site(
        words, valid, attributes, np.arange(size), count, total / np.maximum(count[:, None], 1)
    )
    all_words = np.tile(np.arange(size), (5, 1))
    words = np.concatenate([all_words, all_words[:, ::-1]], axis=1)
    for table in maps.values():
        np.testing.assert_array_equal(np.sort(table), np.broadcast_to(np.arange(size), table.shape))
        packed = encode_words(words, state, table)
        assert np.all(packed[words == 0] == 0)
        inverse = np.argsort(table, axis=1)
        np.testing.assert_array_equal(decode_words(packed, state, inverse), words)
        with pytest.raises(ValueError):
            decode_words(np.array([[size]]), state, inverse)


def test_compact_encoder_uses_reconstructed_predecessor_for_dictionary_choice():
    from src.tokenizers.predictive_word_order import compact_words

    state = np.array([0, 1, 2, 3, 4, 4], np.uint8)
    atoms = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [2.1]])
    candidates = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 4], [0, 3, 4], [0, 1, 5]], np.uint16)
    source = np.array([[5, 4, 0, 2]])
    sent, recovered, contexts = compact_words(source, state, candidates, atoms)
    np.testing.assert_array_equal(contexts, [[0, 2, 4, 0]])
    np.testing.assert_array_equal(recovered, [[2, 4, 0, 2]])
    np.testing.assert_array_equal(decode_words(sent, state, candidates), recovered)
    with pytest.raises(ValueError):
        decode_words(np.array([[3]]), state, candidates)
