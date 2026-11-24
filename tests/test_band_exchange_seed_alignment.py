import numpy as np
from experiments.scripts.audit_band_exchange_seed import align_role_words


def test_role_alignment_recovers_permuted_ids_and_keeps_missing():
    first = np.array([0, 1, 1, 2, 3, 4, 5, 5, 6], np.uint16)
    other = np.array([0, 2, 2, 3, 1, 6, 4, 4, 5], np.uint16)
    mapping = align_role_words(first, other, atom_count=3, roles=2)
    np.testing.assert_array_equal(mapping[other], first)
    np.testing.assert_array_equal(np.sort(mapping), np.arange(7))
    assert mapping[0] == 0


def test_alignment_uses_fit_members_for_the_mapping():
    first = np.array([1, 1, 1, 2, 2, 3], np.uint16)
    other = np.array([3, 3, 3, 1, 1, 2], np.uint16)
    mapping = align_role_words(first, other, atom_count=3, roles=1)
    held_first = np.array([1, 1, 2, 3], np.uint16)
    held_other = np.array([1, 3, 1, 2], np.uint16)
    assert np.mean(mapping[held_other] == held_first) == 0.75
    np.testing.assert_array_equal(mapping, align_role_words(first, other, atom_count=3, roles=1))
