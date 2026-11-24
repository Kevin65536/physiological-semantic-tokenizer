import numpy as np
import pytest

from src.tokenizers.learned_role_dictionary import (
    LearnedRoleDictionary,
    balanced_roles,
    fit_roles,
    load_model,
)
from src.tokenizers.spatial_waveform_dictionary import spectral_features


def test_balanced_training_selection_finds_informative_electrodes_and_keeps_capacity():
    count = 10
    original = np.arange(count) % 5
    statistics = dict(
        gram=np.broadcast_to(np.eye(count), (5, count, count, count)).copy(),
        cross=np.zeros((5, count, count)),
        energy=np.ones((5, count)),
    )
    for band, informative in enumerate([7, 8, 9, 0, 1]):
        statistics["cross"][band, :, informative] = 1
    _, before = fit_roles(statistics, original, 0.01)
    roles, _, trace = balanced_roles(statistics, original, 0.01, 12, 1e-4)
    _, after = fit_roles(statistics, roles, 0.01)
    np.testing.assert_array_equal(
        np.bincount(roles, minlength=5), np.bincount(original, minlength=5)
    )
    assert after.sum() < before.sum() * 0.01
    assert all(b["objective"] < a["objective"] for a, b in zip(trace, trace[1:]))


def test_learned_roles_control_assignment_and_integer_reload_without_window_values(
    tmp_path,
):
    metadata = dict(
        schema="learned_role_dictionary_v1",
        codebook_size=3,
        positions={"fixture": ["a", "b", "c", "d", "e"]},
        native_roles={"fixture": {"S": [4, 3, 2, 1, 0], "O": [0, 1, 2, 3, 4]}},
        output_projection_hz=[1, 45],
    )
    rng = np.random.default_rng(7)
    atoms = {
        (b, k): np.vstack([np.zeros(400), rng.normal(size=(2, 400))])
        for b in "SO"
        for k in range(5)
    }
    centers = {
        (b, k): spectral_features(atoms[b, k][1:], k) for b in "SO" for k in range(5)
    }
    weights = {("fixture", b, k): np.ones((1, 5)) for b in "SO" for k in range(5)}
    transfer = {b: np.ones(3001) * 0.5 for b in "SO"}
    model = LearnedRoleDictionary(metadata, transfer, centers, atoms, weights)
    inputs = rng.normal(size=(1, 5, 15, 400))
    observed = np.array([[True, True, True, True, False]])
    ids = model.encode(inputs, "fixture", observed)
    assert all((x[:, 4] == 0).all() for x in ids.values())
    expected = model.decode_ids(ids, "fixture")
    file = tmp_path / "roles.npz"
    model.save(file)
    restored = load_model(file)
    for key, value in restored.decode_ids(ids, "fixture").items():
        np.testing.assert_array_equal(value, expected[key])
    metadata["native_roles"]["fixture"]["S"] = [4, 4, 2, 1, 0]
    with pytest.raises(ValueError, match="capacity"):
        LearnedRoleDictionary(metadata, transfer, centers, atoms, weights)


def test_role_fit_covariance_uses_true_sample_mask_and_training_variance():
    from experiments.scripts.run_learned_spatial_roles import moments
    from src.tokenizers.spatial_waveform_dictionary import band_filter

    rng = np.random.default_rng(23)
    carrier = rng.normal(size=(2, 5, 6000))
    truth = carrier.copy()
    mask = np.ones((5, 6000), bool)
    mask[0, 17:23] = False
    variance = np.full((2, 5), 3.0)
    statistics = moments(carrier, truth, mask, variance)
    b, k, p = 0, 2, 0
    anchors = np.array([0, 2])
    from src.tokenizers.learned_role_dictionary import fit_band

    weights, error = fit_band(
        statistics["gram"][b, k],
        statistics["cross"][b, k],
        statistics["energy"][b, k],
        anchors,
        0.01,
    )
    x = band_filter(carrier[b], k).T
    y = band_filter(truth[b], k).T
    direct = np.mean((x[:, anchors] @ weights[:, p] - y[:, p])[mask[p]] ** 2) / 3.0
    np.testing.assert_allclose(error[p], direct, rtol=1e-12, atol=1e-12)
