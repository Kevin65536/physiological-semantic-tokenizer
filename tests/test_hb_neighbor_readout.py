import numpy as np
import pytest

from src.tokenizers.hb_neighbor_readout import (
    decode_native,
    apply_coefficients,
    fit_site,
)


class Source:
    silent_ids = {"S": 128, "O": 256}

    @staticmethod
    def decode_ids(ids):
        s = np.repeat(ids["S"][:, :, None], 40, axis=2).astype(float) * 0.01
        o = np.repeat(ids["O"][:, :, None], 40, axis=2).astype(float) * 0.02
        return {"S": s, "O": o, "sum": s + o}


def test_full_array_decoder_uses_neighbor_integers_and_preserves_quiet_and_missing():
    ids = {
        "S": np.tile(np.array([1, 128, 129])[:, None], (1, 15)),
        "O": np.tile(np.array([2, 256, 257])[:, None], (1, 15)),
    }
    values, missing, quiet = decode_native(Source(), ids, {"S": 129, "O": 257})
    coefficient = np.zeros((3, 12, 4))
    coefficient[0, 4, 0] = 1.0
    coefficient[1, 0, 0] = 7.0
    coefficient[2, 0, 0] = 8.0
    prediction = apply_coefficients(values, missing, quiet, coefficient)
    np.testing.assert_allclose(prediction["S"][0].reshape(300, 2)[:, 0], 1.29)
    np.testing.assert_allclose(prediction["S"][0].reshape(300, 2)[:, 1], .01)
    np.testing.assert_array_equal(prediction["S"][1], 1.28)
    np.testing.assert_array_equal(prediction["S"][2], 0.0)
    np.testing.assert_array_equal(prediction["sum"], prediction["S"] + prediction["O"])
    broken = {k: v.copy() for k, v in ids.items()}
    broken["S"][2, 0] = 0
    with pytest.raises(ValueError, match="complete repeated"):
        decode_native(Source(), broken, {"S": 129, "O": 257})


def test_neighbor_low_rank_fit_improves_a_residual_own_features_cannot_predict():
    rng = np.random.default_rng(10)
    features = rng.normal(size=(1000, 8))
    target = np.zeros((1000, 4))
    target[:, 0] = features[:, 4]
    target[:, 2] = -features[:, 4]
    gram = features.T @ features / len(features)
    cross = features.T @ target / len(features)
    variance = np.ones((3, 2))
    local = fit_site(gram, cross, variance, 0, 0, 0.01)
    other = fit_site(gram, cross, variance, 0, 4, 0.01)
    assert np.mean((features @ other - target) ** 2) < 0.02 * np.mean(
        (features @ local - target) ** 2
    )
    np.testing.assert_allclose(other[:, 0] + other[:, 2], 0, atol=1e-14)
