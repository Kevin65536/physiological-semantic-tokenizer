"""A zero residual is always an available word and the decoder receives every extra ID."""

import numpy as np
from threadpoolctl import threadpool_limits
from src.tokenizers.band_event_dictionary import (
    fit_residual_stages,
    encode_residual_stages,
    add_residual_words,
    event_word_ids,
)


def test_residual_zero_and_saved_tuple_reproduce_every_stage():
    rng = np.random.default_rng(173)
    residual = rng.normal(size=(128, 400)).astype(np.float32)
    with threadpool_limits(1):
        model = fit_residual_stages(
            residual, np.ones(len(residual)), np.eye(400, dtype=np.float32)[:32], 173, capacity=15
        )
        ids = encode_residual_stages(residual, model)
    current = residual.copy()
    for layer in range(1, 4):
        before = np.square(current.astype(float)).sum()
        current -= model[f"layer_{layer}_atoms"][ids[:, layer - 1]]
        assert np.square(current.astype(float)).sum() <= before + 1e-5
        np.testing.assert_array_equal(model[f"layer_{layer}_atoms"][0], np.zeros(400))
    words = np.zeros((2, 15), np.uint16)
    words[0, :2] = event_word_ids(1, np.array([1, 2], np.int32))
    model_list = [model] * 5
    code = np.array([[1, 2, 3], [4, 5, 6]], np.uint8)
    decoded = add_residual_words(np.zeros((2, 15, 400)), words, code, model_list, 4)
    expected = sum(model[f"layer_{j}_atoms"][code[:, j - 1]].astype(np.float64) for j in (1, 2, 3))
    np.testing.assert_array_equal(decoded[0, :2], expected)
    np.testing.assert_array_equal(
        add_residual_words(np.zeros((2, 15, 400)), words, code, model_list, 1), np.zeros((2, 15, 400))
    )
