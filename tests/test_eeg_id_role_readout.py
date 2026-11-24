import numpy as np
from experiments.scripts.audit_eeg_id_role_information import relative_word_readout


def test_relative_word_profile_shares_across_sites_and_preserves_unseen_background():
    words = np.tile([1, 2], 200)
    bg = np.repeat([0, 1], 200)
    y = (1 + (words == 2)) * (2 + 2 * bg)
    reference, predicted, shuffled = relative_word_readout(
        words,
        bg,
        y[:, None],
        np.array([1, 2, 3, 0]),
        np.array([0, 1, 2, 1]),
        vocabulary=8192,
        prior=8,
        subjects=np.zeros(400, int),
        seed=43,
    )
    np.testing.assert_allclose(reference.ravel(), [3, 6, 4.5, 6])
    assert 2 < predicted[0, 0] < 2.1 and 7.9 < predicted[1, 0] < 8
    np.testing.assert_array_equal(predicted[2:], reference[2:])
    assert np.linalg.norm(shuffled - reference) < np.linalg.norm(predicted - reference)


def test_absolute_loss_recovers_the_same_word_least_squares_coefficient():
    codes = np.array([1, 2, 1, 2])
    groups = np.array([0, 0, 1, 1])
    y = np.array([0, 2, 15, 5])[:, None]
    base, prediction, _ = relative_word_readout(
        codes,
        groups,
        y,
        np.array([1]),
        np.array([1]),
        vocabulary=8192,
        prior=8,
        subjects=np.zeros(4, int),
        seed=43,
        loss="absolute",
    )
    slope = ((-1) * 1 + 5 * 10) / (1 + 10**2)
    np.testing.assert_allclose(prediction, base * (1 + slope * 2 / 10))
