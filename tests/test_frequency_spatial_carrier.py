import numpy as np

from src.tokenizers.frequency_spatial_carrier import (
    FREQUENCY_INDICES,
    FrequencySpatialCarrier,
    fit_modes,
)


def fixture():
    time = np.arange(400) / 200
    carrier = np.stack(
        [np.sin(2 * np.pi * 5 * time), 2 * np.sin(2 * np.pi * 10 * time)]
    )
    waves = np.tile(carrier[None, :, None, :], (1, 1, 15, 1))
    spectrum = np.fft.rfft(waves[0].transpose(1, 0, 2), axis=-1) / 400
    covariance = np.stack(
        [
            np.einsum("qp,qr->pr", spectrum[:, :, f], spectrum[:, :, f].conj()) / 15
            for f in FREQUENCY_INDICES
        ]
    )
    selected = fit_modes(covariance, 1)
    meta = dict(
        schema="frequency_spatial_carrier_v1",
        complex_coordinates_per_slot=1,
        positions={"fixture": ["a", "b"]},
    )
    model = FrequencySpatialCarrier(
        meta,
        {("fixture", b): np.ones(2) for b in "SO"},
        {("fixture", b): selected for b in "SO"},
    )
    return model, dict(S=waves, O=waves * 0.5)


def test_short_window_DC_coordinates_reduce_in_band_noninteger_patch_frequency_loss(
    tmp_path,
):
    t = np.arange(6000) / 200
    wave = np.sin(2 * np.pi * 1.3 * t).reshape(1, 1, 15, 400)
    errors = []
    for indices in [FREQUENCY_INDICES, np.arange(90)]:
        covariance = np.ones((len(indices), 1, 1), dtype=complex)
        basis = fit_modes(covariance, len(indices), frequency_indices=indices)
        model = FrequencySpatialCarrier(
            dict(
                schema="frequency_spatial_carrier_v2",
                positions={"fixture": ["a"]},
                complex_coordinates_per_slot=len(indices),
                local_frequency_indices=indices.tolist(),
            ),
            {("fixture", b): np.ones(1) for b in "SO"},
            {("fixture", b): basis for b in "SO"},
        )
        coordinates = model.analyze(
            dict(S=wave, O=wave), "fixture", np.ones((1, 1), bool)
        )
        result = model.synthesize(coordinates, "fixture")["S"]
        errors.append(float(np.mean((result - wave) ** 2)))
        spectrum = np.fft.rfft(result.ravel())
        frequency = np.fft.rfftfreq(6000, 1 / 200)
        assert np.max(abs(spectrum[(frequency < 1) | (frequency >= 45)])) < 1e-10
        file = tmp_path / f"m{len(indices)}.npz"
        model.save(file)
        np.testing.assert_array_equal(
            FrequencySpatialCarrier.load(file).synthesize(coordinates, "fixture")["S"],
            result,
        )
    # Full local frequency support still discards the crop's >=45Hz leakage.
    # The hypothesis here is positive restoration, not an assumed exact inverse.
    assert errors[1] < errors[0]


def test_DC_mode_selection_uses_its_single_real_parseval_weight():
    covariance = np.asarray([[[50.0]], [[30.0]]], dtype=complex)
    selected = fit_modes(covariance, 1, frequency_indices=np.asarray([0, 1]))
    assert selected["frequency"].item() == 1
    assert selected["training_total_energy"] == 110


def test_power_only_local_audit_is_invariant_to_carrier_phase():
    from experiments.scripts.run_frequency_spatial_carrier import coordinate_features

    z = np.asarray([[1 + 2j, 3 - 4j]])
    np.testing.assert_allclose(
        coordinate_features(z, "power_only"),
        coordinate_features(z * np.exp(0.7j), "power_only"),
    )
    x = np.asarray([[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(coordinate_features(x, "identity"), x)


def test_projection_diagnostic_separates_known_crop_DC_low_and_high_frequency_floor():
    from experiments.scripts.run_frequency_spatial_carrier import (
        temporal_projection_diagnostics,
    )

    t = np.arange(6000) / 200
    truth = (
        5
        + 3 * np.sin(2 * np.pi * 0.4 * t)
        + 2 * np.sin(2 * np.pi * 8 * t)
        + 4 * np.sin(2 * np.pi * 60 * t)
    )
    values = np.tile(truth.reshape(1, 15, 400), (2, 1, 1))
    mask = np.ones_like(values, dtype=bool)
    mask[1, 0, :100] = False
    batch = dict(
        x=values,
        s=values,
        o=values,
        mask=mask,
        vx=np.ones_like(values),
        vs=np.ones_like(values),
        vo=np.ones_like(values),
    )
    source = {b: values[None] for b in "SO"}
    predictions, complete, below, above = temporal_projection_diagnostics(
        source, batch, np.asarray([0, 1])
    )
    np.testing.assert_array_equal(complete, [True, False])
    np.testing.assert_allclose(below[0], 29.5, atol=1e-10)
    np.testing.assert_allclose(above[0], 8.0, atol=1e-10)
    assert np.isnan(below[1]).all() and np.isnan(above[1]).all()
    oracle = predictions["independent_target_projection_oracle"]["sum"][0, 0].ravel()
    np.testing.assert_allclose(oracle, 2 * np.sin(2 * np.pi * 8 * t), atol=1e-10)
    np.testing.assert_allclose(np.mean((truth - oracle) ** 2), 37.5, atol=1e-10)
    # Measured caches use float32; the mathematical floor accumulates in double.
    for key in ["x", "s", "o"]:
        batch[key] = batch[key].astype(np.float32)
    predictions, _, below, above = temporal_projection_diagnostics(
        source, batch, np.asarray([0, 1])
    )
    oracle = predictions["independent_target_projection_oracle"]["sum"][0, 0].ravel()
    error = np.mean((batch["x"][0].astype(np.float64).ravel() - oracle) ** 2)
    np.testing.assert_allclose(
        error, below[0, 0, 0] + above[0, 0, 0], rtol=1e-12, atol=1e-12
    )


def test_frequency_selection_reconstructs_known_in_support_modes_and_exact_slot_count(
    tmp_path,
):
    model, waves = fixture()
    observed = np.ones((1, 2), bool)
    coordinates = model.analyze(waves, "fixture", observed)
    assert all(v.shape == (1, 2, 15, 1) for v in coordinates.values())
    assert set(model.modes["fixture", "S"]["frequency"].ravel()) == {10, 20}
    expected = model.synthesize(coordinates, "fixture")
    for b in "SO":
        np.testing.assert_allclose(expected[b], waves[b], atol=1e-12)
    file = tmp_path / "model.npz"
    model.save(file)
    loaded = FrequencySpatialCarrier.load(file)
    for b, value in loaded.synthesize(coordinates, "fixture").items():
        np.testing.assert_array_equal(value, expected[b])
    np.testing.assert_array_equal(expected["sum"], expected["S"] + expected["O"])


def test_missing_physical_slot_suppresses_its_complex_coordinates_without_renormalizing():
    model, waves = fixture()
    observed = np.array([[True, False]])
    coordinates = model.analyze(waves, "fixture", observed)
    for b in "SO":
        np.testing.assert_array_equal(coordinates[b][:, 1], 0)
    result = model.synthesize(coordinates, "fixture")
    assert all(np.isfinite(v).all() for v in result.values())
    # The 45Hz endpoint is absent from every declared mode, including a zero covariance.
    zero = fit_modes(np.zeros((len(FREQUENCY_INDICES), 2, 2), complex), 1)
    assert np.all((zero["frequency"] >= 2) & (zero["frequency"] < 90))


def test_complex_spatial_phase_and_nonuniform_native_scales_are_preserved():
    time = np.arange(400) / 200
    carrier = np.stack(
        [
            np.sin(2 * np.pi * 7.5 * time),
            np.cos(2 * np.pi * 7.5 * time),
            2 * np.sin(2 * np.pi * 7.5 * time + 0.7),
        ]
    )
    sd = np.array([0.5, 2.0, 4.0])
    wave = np.tile(carrier[None, :, None, :], (1, 1, 15, 1))
    spectra = np.fft.rfft(wave[0] / sd[:, None, None], axis=-1) / 400
    covariance = np.stack(
        [
            np.einsum("pq,rq->pr", spectra[:, :, f], spectra[:, :, f].conj()) / 15
            for f in FREQUENCY_INDICES
        ]
    )
    chosen = fit_modes(covariance, 1)
    model = FrequencySpatialCarrier(
        dict(
            schema="frequency_spatial_carrier_v1",
            positions={"fixture": ["a", "b", "c"]},
            complex_coordinates_per_slot=1,
        ),
        {("fixture", b): sd for b in "SO"},
        {("fixture", b): chosen for b in "SO"},
    )
    values = model.analyze(dict(S=wave, O=wave), "fixture", np.ones((1, 3), bool))
    for name, output in model.synthesize(values, "fixture").items():
        np.testing.assert_allclose(
            output, wave * (2 if name == "sum" else 1), atol=1e-11
        )
