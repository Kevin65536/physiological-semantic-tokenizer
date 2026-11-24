import numpy as np
from src.tokenizers.anchored_spatial_carrier import AnchoredSpatialCarrier, fit_band


def test_projection_retains_shared_mode_with_physical_orientation_and_target_scales():
    # Principal mode has opposite signs at these electrodes. Align its sign to the second site.
    unit = np.array([0.6, -0.8])
    cov = 4 * np.outer(unit, unit)
    scales = np.array([2.0, 3.0])
    analysis, synthesis, physical = fit_band(cov, scales, np.array([1]), 0.001)
    wave = np.sin(np.arange(100) / 10)[:, None] * (scales * unit)
    slot = wave @ analysis
    np.testing.assert_allclose(slot, wave[:, 1:], atol=1e-12)
    np.testing.assert_allclose(slot @ synthesis, wave, atol=1e-12)
    np.testing.assert_allclose(wave[:, 1:] @ physical, wave / 1.001, atol=1e-12)


def test_synthesis_uses_slot_band_and_preserves_native_order_and_missing_sites(
    tmp_path,
):
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    # Assign only site k to band k and push its carrier to all native sites.
    mats = {("d", b, k): np.eye(5)[:, [k]] for b in "SO" for k in range(5)}
    inv = {key: np.ones((1, 5)) for key in mats}
    basis = AnchoredSpatialCarrier(
        dict(
            schema="anchored_spatial_carrier_v1",
            positions={"d": list("ABCDE")},
            bands_hz=[list(k) for k in bands],
        ),
        {b: np.full(3001, 0.5) for b in "SO"},
        mats,
        inv,
    )
    t = np.arange(6000) / 200
    x = np.tile(
        np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 45 * t), (1, 5, 1)
    ).reshape(1, 5, 15, 400)
    pred = basis.continuous_source(x)
    coded = basis.analyze(pred, "d", np.ones((1, 5), bool))
    y = basis.synthesize(coded, "d")
    np.testing.assert_allclose(
        y["sum"].reshape(1, 5, 6000),
        np.tile(np.sin(2 * np.pi * 10 * t), (1, 5, 1)),
        atol=1e-12,
    )
    mask = np.array([[True, True, False, True, True]])
    missing = basis.analyze(pred, "d", mask)
    np.testing.assert_array_equal(missing["S"][:, 2], 0)
    np.testing.assert_allclose(basis.synthesize(missing, "d")["sum"], 0, atol=1e-12)
    f = tmp_path / "basis.npz"
    basis.save(f)
    np.testing.assert_array_equal(
        AnchoredSpatialCarrier.load(f).synthesize(coded, "d")["sum"], y["sum"]
    )
