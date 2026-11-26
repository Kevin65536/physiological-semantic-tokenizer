import numpy as np
from src.analysis.task_pattern_expansion import auc_columns, family_features, features, spectral_db


def test_direction_ties_and_region_identity():
    y = np.array([0, 0, 1, 1])
    x = np.column_stack([y, 1 - y, np.ones(4)])
    np.testing.assert_allclose(auc_columns(x, y), [1, 0, 0.5])
    assert np.isnan(auc_columns(x, np.ones(4))).all()
    e = np.zeros((4, 6, 15, 3))
    e[:, 4, 3:7, 1] = 2
    h = np.zeros((4, 4, 15, 2))
    ef = np.zeros((4, 6, 15, 2), int)
    hf = ef[:, :4]
    f = features(e, h, ef, hf)
    np.testing.assert_array_equal(f["EEG", "motor_alpha_lr"], 2)
    occ, t = family_features(hf)
    assert occ.shape == (4, 128) and t.shape == (4, 512)
    wave = np.sin(2 * np.pi * 10 * np.arange(400) / 200)
    band = spectral_db(np.broadcast_to(wave, (1, 6, 15, 400)))
    np.testing.assert_allclose(band[..., 1], 10 * np.log10(0.5), atol=0.005)
