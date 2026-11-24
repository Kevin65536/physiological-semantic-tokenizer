import numpy as np

from src.tokenizers.staggered_time_carrier import (
    StaggeredTimeCarrier,
    sample_layout,
    stationary_moments,
    fit_interpolator,
)


def fixture(staggered):
    count = 16
    t = np.arange(6000) / 200
    train = np.sin(2 * np.pi * 7 * t) + 0.8 * np.cos(2 * np.pi * 13 * t)
    wave = np.tile(train, (count, 1))
    moments = stationary_moments(wave, np.ones(count, bool))
    indices, _ = sample_layout(count, 4, staggered)
    op = fit_interpolator(moments, indices, 1e-6)
    model = StaggeredTimeCarrier(
        dict(
            schema="staggered_time_carrier_v1",
            real_samples_per_slot=4,
            positions={"fixture": [str(i) for i in range(count)]},
        ),
        {("fixture", b): np.ones(count) for b in "SO"},
        {"fixture": indices},
        {("fixture", b): op for b in "SO"},
    )
    held = np.cos(2 * np.pi * 7 * t + 0.4) + 0.6 * np.sin(2 * np.pi * 13 * t + 0.7)
    data = np.tile(held[None, None], (1, count, 1)).reshape(1, count, 15, 400)
    return model, dict(S=data, O=data * 0.5)


def test_staggered_actual_samples_resolve_the_aligned_alias_on_held_phase():
    errors = []
    for staggered in [False, True]:
        model, waves = fixture(staggered)
        z = model.analyze(waves, "fixture", np.ones((1, 16), bool))
        predicted = model.synthesize(z, "fixture")
        errors.append(np.mean((predicted["S"] - waves["S"]) ** 2))
    assert errors[1] < 1e-6
    assert errors[0] > 0.1


def test_real_sample_mapping_missing_slots_and_frozen_reload(tmp_path):
    for m in [4, 16, 32]:
        indices, phase = sample_layout(30, m, True)
        assert indices.shape == (30, m)
        assert all(len(np.unique(row)) == m for row in indices)
        assert indices.min() >= 0 and indices.max() < 400
        assert np.all(np.diff(indices, axis=1) > 0)
    model, waves = fixture(True)
    observed = np.ones((1, 16), bool)
    observed[:, 3] = False
    z = model.analyze(waves, "fixture", observed)
    for b in "SO":
        np.testing.assert_array_equal(z[b][:, 3], 0)
    file = tmp_path / "model.npz"
    model.save(file)
    loaded = StaggeredTimeCarrier.load(file)
    for b, v in model.synthesize(z, "fixture").items():
        np.testing.assert_array_equal(v, loaded.synthesize(z, "fixture")[b])
    pred = loaded.synthesize(z, "fixture")
    np.testing.assert_array_equal(pred["S"] + pred["O"], pred["sum"])


def test_covariance_lag_orientation_matches_direct_periodic_time_average():
    rng = np.random.default_rng(95)
    waves = rng.normal(size=(3, 6000))
    moments = stationary_moments(waves, np.array([True, False, True]))
    for lag in [-123, 0, 97]:
        expected = np.mean(waves[1] * np.roll(waves[2], -lag))
        np.testing.assert_allclose(
            moments["target_observation"][1, 2, lag + 400], expected, atol=1e-15
        )
    np.testing.assert_array_equal(moments["observation"][1], 0)
