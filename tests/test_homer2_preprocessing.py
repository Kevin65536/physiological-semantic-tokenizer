import numpy as np
import pytest

from src.data.homer2_preprocessing import (
    HOMER2_ALIGNMENT_SCHEMA,
    MEASUREMENT_ALIGNMENT_SCHEMA,
    apply_homer2_aligned_contract,
    get_homer2_dataset_compatibility,
    homer2_compatibility_manifest,
    intensity_to_optical_density,
    modified_beer_lambert,
)


def test_compatibility_manifest_marks_full_and_partial_datasets():
    single = get_homer2_dataset_compatibility("eeg_fnirs_single_trial")
    simultaneous = get_homer2_dataset_compatibility("simultaneous_eeg_nirs")
    manifest = homer2_compatibility_manifest()

    assert manifest["schema"] == HOMER2_ALIGNMENT_SCHEMA
    assert single.entry_stage == "raw_intensity"
    assert "intensity_to_optical_density" in single.possible_steps
    assert simultaneous.completeness == "partial_post_conversion"
    assert "raw_wl1_wl2_intensity" in simultaneous.missing_inputs


def test_intensity_to_optical_density_is_finite_and_baselined():
    time = np.arange(400) / 10.0
    intensity = np.stack(
        (
            1.2 + 0.01 * np.sin(time),
            0.9 + 0.02 * np.cos(time),
        ),
        axis=1,
    )
    od, quality = intensity_to_optical_density(intensity)

    assert od.shape == intensity.shape
    assert np.isfinite(od).all()
    assert abs(float(np.median(od))) < 1e-3
    assert quality["clamped_nonpositive_fraction"] == 0.0


def test_raw_intensity_branch_applies_od_filter_and_mbll():
    time = np.arange(1000) / 10.0
    base = np.ones((time.size, 3, 2), dtype=np.float64)
    base[:, :, 0] += 0.02 * np.sin(2 * np.pi * 0.05 * time)[:, None]
    base[:, :, 1] += 0.01 * np.cos(2 * np.pi * 0.04 * time)[:, None]

    result = apply_homer2_aligned_contract(
        base,
        dataset_id="eeg_fnirs_single_trial",
        sample_rate_hz=10.0,
        entry_stage="raw_intensity",
        wavelengths_nm=(760.0, 850.0),
    )

    assert result.values.shape == (time.size, 6)
    assert np.isfinite(result.values).all()
    assert "intensity_to_optical_density" in result.state.applied_steps
    assert "modified_beer_lambert" in result.state.applied_steps
    assert result.state.schema == HOMER2_ALIGNMENT_SCHEMA


def test_chromophore_branch_records_missing_raw_homer2_inputs():
    time = np.arange(1000) / 10.0
    values = np.column_stack(
        (
            np.sin(2 * np.pi * 0.04 * time),
            -0.5 * np.cos(2 * np.pi * 0.04 * time),
        )
    )

    result = apply_homer2_aligned_contract(
        values,
        dataset_id="simultaneous_eeg_nirs",
        sample_rate_hz=10.0,
        entry_stage="chromophore",
        wavelengths_nm=(760.0, 850.0),
    )

    assert result.values.shape == values.shape
    assert "bandpass" in result.state.applied_steps
    assert "intensity_to_optical_density" in result.state.skipped_steps
    assert "raw_light_intensity" in result.state.missing_inputs
    assert "modified_beer_lambert" in result.state.skipped_steps


def test_modified_beer_lambert_returns_hbo_hbr_axis():
    od = np.zeros((50, 4, 2), dtype=np.float64)
    od[:, :, 0] = 0.01
    od[:, :, 1] = -0.005
    concentration, quality = modified_beer_lambert(od, wavelengths_nm=(760.0, 850.0))

    assert concentration.shape == (50, 4, 2)
    assert np.isfinite(concentration).all()
    assert quality["wavelengths_nm"] == [760.0, 850.0]


def test_physical_mbll_closes_known_concentration_and_log_bases():
    # Independent forward law: decadic extinction in 1/(M cm), 3 cm * DPF 6.
    extinction = np.array([[586., 1548.52], [1058., 691.32]])
    time = np.arange(100) / 10
    truth = np.stack([2*np.sin(time), -.4*np.cos(time)], axis=-1)[:, None, :]
    natural_od = (truth * 1e-6) @ extinction.T * 18 * np.log(10.)
    for od, base in [(natural_od, "e"), (natural_od / np.log(10.), "10")]:
        hb, evidence = modified_beer_lambert(
            od, wavelengths_nm=[760, 850], extinction_coefficients=extinction,
            extinction_unit="1/(M cm)", coefficient_log_base="10", od_log_base=base,
            coefficient_source="Prahl tabulated extinction; synthetic test")
        np.testing.assert_allclose(hb, truth, atol=2e-15)
        assert evidence['output_unit'] == 'uM'
    with pytest.raises(ValueError, match='unknown wavelength'):
        modified_beer_lambert(natural_od, wavelengths_nm=[761, 850])
    with pytest.raises(ValueError, match='pathlength'):
        modified_beer_lambert(natural_od, wavelengths_nm=[760, 850], partial_pathlength_factor=0)


def test_published_hb_is_converted_once_and_never_runs_mbll():
    time = np.arange(300) / 10
    values = np.stack([np.sin(time), -.2*np.sin(time)], axis=-1)
    kwargs = dict(dataset_id='simultaneous_eeg_nirs', sample_rate_hz=10.,
                  entry_stage='chromophore', processing_schema=MEASUREMENT_ALIGNMENT_SCHEMA,
                  unit_evidence='MAT oxy/deoxy.yUnit')
    mmol = apply_homer2_aligned_contract(values / 1000, native_unit='mmol/L', **kwargs)
    micro = apply_homer2_aligned_contract(values, native_unit='uM', **kwargs)
    np.testing.assert_allclose(mmol.values, micro.values)
    assert 'modified_beer_lambert' not in mmol.state.applied_steps
    with pytest.raises(ValueError, match='cannot re-enter'):
        apply_homer2_aligned_contract(values, dataset_id='simultaneous_eeg_nirs',
                                      sample_rate_hz=10, entry_stage='raw_intensity')


def test_float64_feature_producer_closes_linear_operator():
    from scipy.signal import resample_poly
    from src.data.homer2_preprocessing import bandpass_fnirs
    t = np.arange(300) / 10
    intensity = np.stack([1.3+.013*np.sin(t), .9+.004*np.cos(t)], axis=-1)[:, None]
    result = apply_homer2_aligned_contract(
        intensity, dataset_id='eeg_fnirs_single_trial', sample_rate_hz=10,
        entry_stage='raw_intensity', wavelengths_nm=[760,850],
        retain_feature_boundary=True, processing_schema=MEASUREMENT_ALIGNMENT_SCHEMA)
    filtered, _ = bandpass_fnirs(np.eye(300), sample_rate_hz=10)
    operator = resample_poly(filtered, 2, 5, axis=0)
    actual = resample_poly(result.values, 2, 5, axis=0)
    replay = operator @ result.pre_linear_values.reshape(300, 2)
    error = np.max(abs(actual-replay)) / np.std(actual)
    assert result.values.dtype == np.float64
    assert result.pre_linear_values.dtype == np.float64
    assert result.state.schema == MEASUREMENT_ALIGNMENT_SCHEMA
    assert error < 1e-6


def test_native_missing_is_hidden_before_od_motion_statistics_and_preserves_support():
    rng = np.random.default_rng(142)
    raw = np.exp(.02*rng.normal(size=(300,2,2)))
    visible = np.ones_like(raw,dtype=bool);visible[100:120,0] = False
    changed = raw.copy();changed[~visible] = 1e10
    kwargs = dict(dataset_id='eeg_fnirs_single_trial',sample_rate_hz=10,entry_stage='raw_intensity',
                  wavelengths_nm=[760,850],processing_schema=MEASUREMENT_ALIGNMENT_SCHEMA,
                  valid_mask=visible,retain_feature_boundary=True)
    a = apply_homer2_aligned_contract(raw,**kwargs)
    b = apply_homer2_aligned_contract(changed,**kwargs)
    np.testing.assert_array_equal(a.values,b.values)
    np.testing.assert_array_equal(a.pre_linear_optical_density,b.pre_linear_optical_density)
    assert not a.recorded_mask[100:120,0].any()
    assert not a.processed_valid_mask[:,:2].any()
    assert a.processed_valid_mask[:,2:].all()
    assert a.values.shape == (300,4)
