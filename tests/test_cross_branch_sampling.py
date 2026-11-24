import numpy as np
from src.tokenizers.cross_branch_sampling import layouts, covariance_blocks, calibration, reconstruct


def test_staggering_preserves_observed_role_quota_and_marginals():
    seen = np.array([True, False, True, True, True, True, True])
    role = layouts(seen)
    a = role["aligned"]["O"]
    b = role["staggered"]["O"]
    np.testing.assert_array_equal(np.sort(a), np.sort(b))
    assert b[1] == -1 and (a[seen] != b[seen]).any()
    cov = np.tile(np.eye(2), (5, 1, 1))
    separate, joint, _ = covariance_blocks(cov, 2 * cov, 3 * cov)
    assert np.linalg.eigvalsh(joint).min() > 0
    np.testing.assert_array_equal(separate[:, :2, :2], joint[:, :2, :2])


def test_other_branch_observation_recovers_correlation_with_expected_sign_and_scale():
    v = np.sin(2 * np.pi * 6 * np.arange(6000) / 200)
    q = np.zeros((2, 15, 400))
    q[1] = v.reshape(15, 400)
    cov = np.tile([[1.0, -0.6], [-0.6, 1.0]], (5, 1, 1))
    p = reconstruct(q, np.array([0, 1]), cov, np.ones((5, 2)), np.zeros((5, 2)))
    np.testing.assert_allclose(p["S"].ravel(), -0.6 * v, atol=2e-13)
    np.testing.assert_allclose(p["O"].ravel(), v, atol=2e-13)
    np.testing.assert_array_equal(p["sum"], p["S"] + p["O"])
    g, n = calibration(np.array([[4.0, 0], [2.0, 0], [3.0, 0], [2.0, 0]]))
    np.testing.assert_array_equal(g, [0.5, 1])
    np.testing.assert_array_equal(n, [1, 0])
