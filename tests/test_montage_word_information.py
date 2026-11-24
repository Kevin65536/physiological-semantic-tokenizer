import numpy as np
import pytest
from experiments.scripts.audit_montage_word_information import restore_montage_streams, type_role


def archive(examples, values):
    result = {"examples": np.array(examples)}
    for name in ("semantic64_S", "semantic64_O", "waveform_S", "waveform_O"):
        result[name] = np.repeat(np.array(values, np.uint16)[:, None], 15, axis=1)
    return result


def test_restore_words_by_physical_identity_and_reject_wrong_person_partition():
    examples = np.array([[1, 30, 0], [1, 30, 1], [2, 60, 0]])
    folds = np.array([0, 0, 1])
    fit = archive(examples[2:], [11])
    held = archive(examples[[1, 0]], [8, 9])
    restored = restore_montage_streams(examples, folds, 0, fit, held)
    np.testing.assert_array_equal(restored["waveform_S"][:, 0], [9, 8, 11])
    with pytest.raises(ValueError):
        restore_montage_streams(examples, folds, 0, held, fit)
    repeated = archive(examples[[0, 0]], [9, 9])
    with pytest.raises(ValueError):
        restore_montage_streams(examples, folds, 0, fit, repeated)


def test_actual_paid_type_and_band_include_residual_offset():
    np.testing.assert_array_equal(
        type_role(np.array([0, 1, 1023, 1024, 5115, 5116, 7162, 7163, 15350])),
        [0, 1, 1, 2, 5, 6, 6, 7, 10],
    )


def test_shared_join_accepts_declared_event_vocabulary_and_keeps_limits():
    ex = np.array([[1, 30, 0], [2, 60, 0]])
    folds = np.array([0, 1])
    fit = {"examples": ex[1:], "event_S": np.full((1, 15), 50000, np.uint16)}
    held = {"examples": ex[:1], "event_S": np.full((1, 15), 58742, np.uint16)}
    actual = restore_montage_streams(
        ex, folds, 0, fit, held, stream_fields=("event_S",), capacity=58743
    )
    np.testing.assert_array_equal(actual["event_S"][:, 0], [58742, 50000])
    with pytest.raises(ValueError):
        restore_montage_streams(ex, folds, 0, fit, held, stream_fields=("event_S",), capacity=50000)
