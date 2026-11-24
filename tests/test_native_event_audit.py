"""Independent decoding identities for the retained native-event audit."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.tokenizers.montage_word_search import dictionaries
from src.tokenizers.native_event_words import decode_events, event_wave, pulse_shapes

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_event_audit", ROOT / "experiments/scripts/audit_native_event_words.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_every_actual_event_wave_has_independent_parameter_decode():
    pulses = pulse_shapes()
    for start in range(audit.FIRST, audit.CAPACITY, 2048):
        codes = np.arange(start, min(start + 2048, audit.CAPACITY), dtype=np.uint16)
        np.testing.assert_allclose(audit.event_wave_independent(codes), event_wave(codes, pulses), rtol=2e-13, atol=2e-13)
    assert audit.event_attributes(np.array([0, 15350, 15351, 26742, 26743, 26775, 58742])).tolist() == [0, 0, 1, 1, 2, 3, 3]


def test_transport16_preserves_entire_paid_alphabet_and_rejects_unused_cells():
    values = np.arange(audit.CAPACITY, dtype=np.uint16)
    payload, restored = audit.transport16(values)
    assert len(payload) == 2 * audit.CAPACITY
    np.testing.assert_array_equal(restored, values)
    with pytest.raises(ValueError):
        audit.transport16(np.array([65535], dtype=np.uint16))


def test_independent_native_replacement_with_missing_sites_matches_consumer():
    n = 5
    time = np.arange(400)
    wave = np.stack([np.cos(2 * np.pi * time / 40), np.sin(2 * np.pi * time / 50)])
    old, new = {}, {}
    for b in "SO":
        old[b], new[b] = [], []
        for band in range(5):
            old[b].append(dict(balanced_atoms=wave, balanced_gain=np.array([0.7, 1.3]), balanced_centers=np.zeros((2, 3)), balanced_count=np.ones(2), property_mean=np.zeros(3), property_scale=np.ones(3)))
            new[b].append(dict(innovation_atoms=wave[::-1], innovation_centers=np.zeros((2, 3)), innovation_count=np.ones(2), capacity=2047, property_mean=np.zeros(3), property_scale=np.ones(3)))
    cov = np.eye(2 * n) + 0.1 * np.ones((2 * n, 2 * n))
    state = dict(joint=np.repeat(cov[None], 5, axis=0), pilot_gain=np.ones((5, 2*n)), pilot_noise=np.full((5, 2*n), 0.2), innovation_mean_gain=np.ones((5, 2*n)), innovation_mean_noise=np.full((5, 2*n), 0.3), scale=np.full((5, 2*n), 1.2))
    seen = np.array([True, True, False, True, True])
    roles = np.arange(n) % 5
    other = roles.copy();other[seen] = np.roll(roles[seen], 2)
    sent = {b: np.zeros((n, 15), np.uint16) for b in "SO"}
    for bi, b in enumerate("SO"):
        for j in np.flatnonzero(seen):
            pilot = (j % 2 == 0) if b == "S" else (j % 2 == 1)
            role = [roles, other][bi][j]
            sent[b][j] = (1 + role * 1023) if pilot else (5116 + role * 2047)
    sent["S"][0, 3] = 15351
    sent["O"][1, 9] = 26743
    sent["S"][4, 11] = 58742
    scales = np.linspace(0.5, 2.5, 2*n)
    got = audit.decode_integer_events(sent, old, new, state, scales)
    expected = decode_events(sent, dictionaries(old, new), state, scales, pulse_shapes())
    for b in got:
        np.testing.assert_allclose(got[b], expected[b], rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(got["sum"], got["S"] + got["O"])
