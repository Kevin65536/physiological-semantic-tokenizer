import numpy as np
from src.analysis.nback_function_screen import path_features, phase, probe


def test_reversal_preserves_occupancy_and_runs_but_transposes_pairs():
    paths = np.random.default_rng(14).integers(0, 4, (6, 4, 4))
    occ, pairs = path_features(paths)
    rev, backward = path_features(paths[..., ::-1])
    np.testing.assert_array_equal(occ, rev)
    np.testing.assert_array_equal(pairs.swapaxes(-1, -2), backward)
    np.testing.assert_allclose(pairs.sum(axis=(-1, -2)), 1)
    assert not np.array_equal(pairs, backward)
    x = np.arange(15)[None, None, :, None]
    np.testing.assert_array_equal(phase(x).ravel(), [2, 7, 12])


def test_development_labels_never_select_or_fit_readout():
    rng = np.random.default_rng(1127)
    groups = np.repeat(np.arange(8), 6)
    y = np.tile(np.eye(3), (16, 1))
    y[:6] = y[[0, 1, 0, 1, 0, 1]]  # One training person lacks a condition.
    x = y + 0.1 * rng.normal(size=y.shape)
    tr, dev = np.arange(36), np.arange(36, 48)
    pr, dirty, reg = probe(x, y, groups, tr, dev, classification=True, dirty=x[dev])
    changed = y.copy()
    changed[dev] = np.roll(y[dev], 1, axis=1)
    alt, _, second_reg = probe(x, changed, groups, tr, dev, classification=True)
    assert reg == second_reg
    np.testing.assert_allclose(pr, alt)
    np.testing.assert_allclose(pr, dirty)
    np.testing.assert_array_equal(pr.argmax(1), y[dev].argmax(1))
