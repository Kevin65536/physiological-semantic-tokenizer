import numpy as np
import pytest

from src.tokenizers.distributed_measurement import (
    atomic_ids, infer, measure, probes, spatial_directions, word_features,
)


def test_rank_one_measurement_matches_explicit_native_space_time_operator():
    rng = np.random.default_rng(107)
    wave = rng.normal(size=(3, 15, 400))
    state = dict(SD=np.array([1., 2., 3.]), temporal=probes(3, 19), spatial=spatial_directions(np.eye(3)))
    seen = np.array([True, False, True])
    actual = measure(wave, seen, state, "spatial")
    normalized = np.where(seen[:, None, None], wave / state["SD"][:, None, None], 0)
    expected = np.einsum("cjk,kpt,cjt->cpj", state["spatial"], normalized, state["temporal"])
    expected[~seen] = 0
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(state["temporal"], axis=-1), 1)


def test_actual_atomic_alphabet_reserves_missing_without_an_extra_bit():
    values = np.ones((2, 15, 10))
    prop = np.ones((2, 15, 3))
    ids = atomic_ids(values, prop, np.array([True, False]), np.zeros((2, 3)), "spatial_attributes")
    assert np.all(ids[0] == 1023) and np.all(ids[1] == 0)
    levels = np.tile(np.array([-.5, .5]), (2, 3, 1))
    features = word_features(ids, np.ones((2, 10)), levels, "spatial_attributes")
    assert np.all(features[0, :, 0] == 0)  # merged last two cells, unknown sign
    assert np.all(features[0, :, 1:7] == 1)
    assert np.all(features[0, :, 7:] == .5)
    assert np.all(features[1] == 0)
    with pytest.raises(ValueError):
        word_features(ids.astype(float), np.ones((2, 10)), levels, "spatial_attributes")


def test_fixed_head_has_no_window_statistics_and_preserves_voltage_grid():
    rng = np.random.default_rng(12)
    features = rng.normal(size=(2, 15, 10)).astype(np.float32)
    head = dict(xmean=np.zeros(20, np.float32), ymean=np.zeros(800, np.float32),
                coefficient=rng.normal(size=(20, 800)).astype(np.float32))
    expected = features.transpose(1, 0, 2).reshape(15, 20) @ head["coefficient"]
    actual = infer(features, head)
    assert actual.shape == (2, 15, 400)
    np.testing.assert_array_equal(actual.transpose(1, 0, 2).reshape(15, 800), expected)
    np.testing.assert_allclose(infer(2 * features, head), 2 * actual, rtol=1e-6)


def test_spatial_covariance_thresholds_and_scalar_levels_exclude_held_people(monkeypatch):
    from experiments.scripts import run_distributed_measurement as runner

    rng = np.random.default_rng(107)
    blocks = [dict(case=dict(subject=f"s{i}", fold=fold, crop=dict(row=i)),
                   source={b: rng.normal(size=(2, 15, 400)).astype(np.float32) for b in "SO"},
                   seen=np.ones(2, bool), complete=np.ones(15, bool)) for i, fold in enumerate((1, 2, 0))]
    table = {n: dict(target_variance=dict(s=[1.], o=[1.])) for n in ("A", "B")}
    monkeypatch.setattr(runner, "STATE", dict(rows=[dict(record="r")] * 3,
         names=dict(study=["A", "B"]), cases=dict(study=[]),
         scales=dict(positions=dict(eeg=dict(study=table)))))
    before = runner.make_states("study", blocks, 0)
    for b in "SO":
        blocks[-1]["source"][b] *= 1e6
    after = runner.make_states("study", blocks, 0)
    for b in before:
        for key in before[b]:
            np.testing.assert_array_equal(before[b][key], after[b][key])
