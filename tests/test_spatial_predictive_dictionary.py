import numpy as np
import pytest
from src.tokenizers.spatial_predictive_dictionary import (
    SpatialPredictiveDictionary,
    fit_tree,
    load_model,
)


def test_tree_decode_uses_reconstructed_prefix_and_frozen_scale(tmp_path):
    order, parent, gain, scale = fit_tree(np.array([[1.0, 0.8], [0.8, 1.0]]))
    assert parent[order[0]] == -1
    tree = dict(order=order, parent=parent, gain=gain, scale=scale)
    t = np.arange(400) / 200
    atoms = np.zeros((512, 400))
    atoms[1] = np.sin(2 * np.pi * 5 * t)
    book = dict(
        atoms=atoms,
        mean=np.zeros(400),
        basis=np.eye(400)[:, :64],
        centers=np.zeros((511, 64)),
    )
    model = SpatialPredictiveDictionary(
        dict(schema="spatial_predictive_dictionary_v1"),
        {b: tree for b in "SO"},
        {b: book for b in "SO"},
    )
    ids = {b: np.ones((1, 2, 15), np.int32) for b in "SO"}
    wave = model.decode_ids(ids)
    root, child = order
    np.testing.assert_allclose(
        wave["S"][0, child],
        (gain[child] * scale[root] + scale[child]) * np.tile(atoms[1], (15, 1)),
        atol=1e-12,
    )
    path = tmp_path / "model.npz"
    model.save(path)
    for k, v in load_model(path).decode_ids(ids).items():
        np.testing.assert_array_equal(v, wave[k])
    ids["S"] = ids["S"].astype(float)
    with pytest.raises(ValueError):
        model.decode_ids(ids)


def test_encoder_quantizes_error_against_decoded_parent_instead_of_true_parent():
    t = np.arange(400) / 200
    wave = np.sin(2 * np.pi * 5 * t)
    atoms = np.zeros((512, 400))
    atoms[1] = wave
    atoms[2] = 2 * wave
    centers = np.full((511, 64), 100.0, np.float32)
    centers[0] = wave[:64]
    centers[1] = 2 * wave[:64]
    book = dict(
        atoms=atoms, mean=np.zeros(400), basis=np.eye(400)[:, :64], centers=centers
    )
    tree = dict(
        order=np.array([0, 1]),
        parent=np.array([-1, 0]),
        gain=np.array([0.0, 0.8]),
        scale=np.array([1.0, 0.6]),
    )
    model = SpatialPredictiveDictionary(
        dict(schema="spatial_predictive_dictionary_v1"),
        {b: tree for b in "SO"},
        {b: book for b in "SO"},
    )
    source = np.stack([np.tile(1.8 * wave, (15, 1)), np.tile(2.46 * wave, (15, 1))])[
        None
    ]
    codes = model.encode_source({b: source for b in "SO"}, np.array([[True, True]]))
    assert np.all(codes["S"][0, 0] == 2)
    assert np.all(codes["S"][0, 1] == 1)
    source[:, 0] = np.nan
    codes = model.encode_source({b: source for b in "SO"}, np.array([[False, True]]))
    assert np.all(codes["S"][0, 0] == 0)
    assert np.all(codes["S"][0, 1] == 2)


@pytest.mark.parametrize("field,value", [("vocabulary", 256), ("context_seconds", 29)])
def test_changed_fixed_model_contract_fails_before_data_reads(
    tmp_path, monkeypatch, field, value
):
    import yaml
    import experiments.scripts.run_spatial_predictive_dictionary as runner

    config = yaml.safe_load(
        (
            runner.ROOT
            / "experiments/configs/physiology_semantic_tokenizer/dual_component_spatial_predictive_v1.yaml"
        ).read_text()
    )
    config["model"][field] = value
    file = tmp_path / "config.yaml"
    file.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        "sys.argv", ["run", "--config", str(file), "--output", str(tmp_path / "out")]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("contract rejection must precede prepared data access")

    monkeypatch.setattr(runner, "base_tokenizer_training_data", forbidden)
    with pytest.raises(ValueError, match="train-only original population"):
        runner.main()
