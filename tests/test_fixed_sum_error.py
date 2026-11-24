import numpy as np
from experiments.scripts.audit_fixed_sum_component_error import decompose


def test_fixed_sum_oracle_is_sharp_with_mask_and_unequal_scales():
    s = np.array([[[1.0], [3.0], [110.0]]])
    o = np.array([[[2.0], [-2.0], [7.0]]])
    m = np.array([[[True], [True], [False]]])
    vs = np.array([[2.0]])
    vo = np.array([[3.0]])
    d = decompose(s, o, m, vs, vo)
    np.testing.assert_allclose(d[0, 3, 0], 1.0)
    e = s + o
    ideal = decompose(e * 0.4, e * 0.6, m, vs, vo)
    np.testing.assert_allclose(ideal[:, 2], d[:, 3])
    np.testing.assert_allclose(ideal[:, 4], 0, atol=1e-20)
