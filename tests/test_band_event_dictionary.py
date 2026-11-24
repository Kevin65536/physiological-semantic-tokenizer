"""Finite event words carry their own complete waveform and amplitude tier."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from src.tokenizers.band_event_dictionary import (
    EVENT_CAPACITY,
    ORIGINAL_SIZE,
    VOCABULARY,
    decode_event_coordinates,
    encode_events,
    event_word_ids,
    fit_event_dictionary,
    word_roles,
)


def test_event_word_is_self_contained_and_old_word_keeps_its_waveform():
    old = [
        (np.repeat(np.arange(1023)[:, None], 400, axis=1).astype(float), np.ones(1023))
        for _ in range(5)
    ]
    models = [
        dict(
            wave_atoms=np.stack([np.ones(400) * 7, np.ones(400) * 13]), wave_gain=np.array([1.5, 2.0])
        )
        for _ in range(5)
    ]
    words = np.zeros((2, 15), np.uint16)
    words[0, 0] = 1 + 2 * 1023 + 4
    words[0, 1] = event_word_ids(3, np.array([1], np.int32))[0]
    roles, wave = decode_event_coordinates(words, old, models, "wave", True)
    np.testing.assert_allclose(wave[0, 0], 4)
    np.testing.assert_allclose(wave[0, 1], 26)
    assert roles[0, 1] == 3 and np.all(wave[1] == 0)
    # No overridden original integer exists at the event decoder.
    changed_old = [(v[0] * 100, v[1]) for v in old]
    _, changed = decode_event_coordinates(words, changed_old, models, "wave", True)
    np.testing.assert_array_equal(changed[0, 1], wave[0, 1])
    assert VOCABULARY == 15351 and int(words.max()) < 2**14
    reserved = words.copy()
    reserved[0, 1] = event_word_ids(3, np.array([2000], np.int32))[0]
    _, unused = decode_event_coordinates(reserved, old, models, "wave", True)
    np.testing.assert_array_equal(unused[0, 1], np.zeros(400))


def test_shape_dictionary_stratifies_amplitude_and_emits_one_index():
    rng = np.random.default_rng(151)
    time = np.arange(400) / 200
    phase = rng.uniform(-np.pi, np.pi, 128)
    wave = (
        np.sin(2 * np.pi * 6 * time[None] + phase[:, None])
        * np.repeat([0.1, 0.5, 2.0, 9.0], 32)[:, None]
    ).astype(np.float32)
    with threadpool_limits(1):
        model = fit_event_dictionary(wave, np.ones(len(wave)), 1, 151, capacity=31)
        for mode in ("wave", "shape"):
            ids = encode_events(wave, model, mode)
            assert ids.shape == (128,) and ids.max() < EVENT_CAPACITY
            paid = event_word_ids(1, ids)
            roles, events = word_roles(paid)
            assert (roles == 1).all() and events.all()
    assert len(model["shape_atoms"]) <= 31
    assert np.all(np.diff(model["shape_edges"]) >= 0)
    with pytest.raises(ValueError):
        word_roles(np.array([VOCABULARY]))
