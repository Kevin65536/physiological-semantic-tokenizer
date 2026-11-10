import json

import numpy as np
import pytest

from src.analysis.physiological_patch_features import (
    CANONICAL_SIGNAL_UNIT,
    DEFAULT_FEATURE_SPEC,
    FrequencyBand,
    PhysiologicalPatchFeatureSpec,
    extract_eeg_patch_features,
    extract_fnirs_patch_features,
    supervision_signal_features,
    supervision_spectral_triplet,
)


def test_supervision_features_resolve_band_and_preserve_unit_invariance():
    fs=200;t=np.arange(4000)/fs;x=np.sin(2*np.pi*10*t)[:,None]
    a,f,p=supervision_signal_features(x,fs,'EEG',np.ones(1))
    b,_,_=supervision_signal_features(x*1e-6,fs,'EEG',np.ones(1)*1e-6)
    assert a['spectral_peak'][0]==10 and a['relative_alpha'][0]==pytest.approx(1.)
    assert a['power_alpha_ref'][0]==pytest.approx(.5)
    assert a['excess_kurtosis'][0]==pytest.approx(-1.5,abs=.002)
    for name in ['excess_kurtosis','skewness','hjorth_mobility','hjorth_complexity','log_rms_ref','relative_alpha']:
        np.testing.assert_allclose(a[name],b[name],atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(b['rms'],a['rms']*1e-6)


def test_supervision_spectra_keep_negative_cross_term_instead_of_energy_partition():
    fs=200;t=np.arange(4000)/fs;x=np.sin(2*np.pi*10*t)[:,None]
    f,p,cross,error=supervision_spectral_triplet(x,2*x,-x,fs,'EEG',np.ones(1))
    assert error<1e-12
    np.testing.assert_allclose(p[1],4*p[0],atol=1e-12);np.testing.assert_allclose(p[2],p[0],atol=1e-12)
    np.testing.assert_allclose(cross,-4*p[0],atol=1e-12)


def test_supervision_zero_variance_and_Hb_outband_are_explicit():
    a,_,_=supervision_signal_features(np.ones((4000,1)),200,'EEG',np.ones(1))
    assert not a['valid_shape'][0] and np.isnan(a['excess_kurtosis'][0])
    t=np.arange(800)/10;x=(np.sin(2*np.pi*.075*t)+.5*np.sin(2*np.pi*.5*t))[:,None]
    a,_,_=supervision_signal_features(x,10,'HbO',np.ones(1))
    assert a['outband_fraction'][0]==pytest.approx(.2,abs=1e-10)


def _coordinate(batch, name):
    return batch.values[..., batch.feature_names.index(name)]


def test_eeg_hann_psd_resolves_channel_level_alpha_and_beta():
    sample_rate_hz = 200.0
    sample_count = 400
    time = np.arange(sample_count) / sample_rate_hz
    alpha = np.sin(2.0 * np.pi * 10.0 * time)
    beta = np.sin(2.0 * np.pi * 20.0 * time)
    signal = np.stack((alpha, beta), axis=0)[None, None, ...]

    batch = extract_eeg_patch_features(
        signal,
        sample_rate_hz=sample_rate_hz,
        channel_names=("alpha_channel", "beta_channel"),
    )

    assert batch.values.shape == (1, 1, 2, 21)
    assert batch.channel_names == ("alpha_channel", "beta_channel")
    assert _coordinate(batch, "peak_frequency")[0, 0, 0] == pytest.approx(10.0)
    assert _coordinate(batch, "peak_frequency")[0, 0, 1] == pytest.approx(20.0)
    alpha_power = _coordinate(batch, "log_absolute_power_alpha")[0, 0]
    beta_power = _coordinate(batch, "log_absolute_power_beta")[0, 0]
    assert alpha_power[0] > beta_power[0] + 10.0
    assert beta_power[1] > alpha_power[1] + 10.0
    assert batch.flatten_channels().shape == (1, 1, 42)
    assert batch.flattened_feature_names[0] == "alpha_channel/mean"


def test_eeg_continuous_layout_matches_explicit_patch_layout():
    rng = np.random.default_rng(7)
    continuous = rng.normal(size=(2, 3, 800))
    patch_major = continuous.reshape(2, 3, 2, 400).transpose(0, 2, 1, 3)

    legacy = extract_eeg_patch_features(
        continuous,
        sample_rate_hz=200.0,
        patch_size=400,
    )
    explicit = extract_eeg_patch_features(
        patch_major,
        sample_rate_hz=200.0,
    )

    assert np.allclose(legacy.values, explicit.values, equal_nan=True)
    assert np.array_equal(
        legacy.feature_valid_mask, explicit.feature_valid_mask
    )
    assert legacy.manifest.input_layout == "batch_channel_sample"
    assert explicit.manifest.input_layout == "batch_patch_channel_sample"


def test_masks_and_nan_produce_explicit_invalid_coordinates_without_infinity():
    rng = np.random.default_rng(11)
    signal = rng.normal(size=(1, 2, 1, 400))
    signal[0, 1, 0, 10] = np.nan
    token_mask = np.array([[True, True]])

    batch = extract_eeg_patch_features(
        signal,
        sample_rate_hz=200.0,
        valid_mask=token_mask,
    )

    assert batch.channel_valid_mask.tolist() == [[[True], [False]]]
    assert np.all(batch.feature_valid_mask[0, 0, 0])
    assert not np.any(batch.feature_valid_mask[0, 1, 0])
    assert np.all(np.isnan(batch.values[0, 1, 0]))
    assert not np.any(np.isinf(batch.values))
    assert batch.valid_sample_fraction[0, 1, 0] == pytest.approx(399 / 400)


def test_empty_or_unresolved_eeg_bands_are_nan_and_manifested():
    time = np.arange(40) / 20.0
    signal = np.sin(2.0 * np.pi * 3.0 * time)[None, None, None, :]

    batch = extract_eeg_patch_features(signal, sample_rate_hz=20.0)

    assert batch.manifest.band_availability["delta"] is True
    assert batch.manifest.band_availability["alpha"] is False
    assert batch.manifest.band_availability["beta"] is False
    assert batch.manifest.band_availability["low_gamma"] is False
    assert np.isnan(_coordinate(batch, "log_absolute_power_alpha")).all()
    assert np.isnan(_coordinate(batch, "log_relative_power_delta")).all()
    assert not np.any(np.isinf(batch.values))


def test_fnirs_two_second_patches_have_morphology_but_no_band_power():
    sample_rate_hz = 10.0
    time = np.arange(20) / sample_rate_hz
    signal = np.stack((time, 2.0 * time), axis=0)[None, None, ...]

    batch = extract_fnirs_patch_features(
        signal,
        sample_rate_hz=sample_rate_hz,
        channel_names=("hbo", "hbr"),
    )

    assert batch.values.shape == (1, 1, 2, 8)
    assert _coordinate(batch, "slope")[0, 0].tolist() == pytest.approx(
        [1.0, 2.0]
    )
    assert _coordinate(batch, "endpoint_delta")[0, 0].tolist() == pytest.approx(
        [1.9, 3.8]
    )
    assert _coordinate(batch, "derivative_spike")[0, 0].tolist() == pytest.approx(
        [1.0, 2.0]
    )
    assert batch.manifest.patch_duration_seconds == pytest.approx(2.0)
    assert batch.manifest.band_availability == {}
    assert not any(
        "power" in name or "spectral" in name for name in batch.feature_names
    )
    assert "Frequency-band power is intentionally excluded" in " ".join(
        batch.manifest.notes
    )


def test_feature_spec_hash_is_stable_sensitive_and_manifest_is_json_safe():
    equivalent = PhysiologicalPatchFeatureSpec()
    changed = PhysiologicalPatchFeatureSpec(
        eeg_bands=(
            FrequencyBand("delta", 1.0, 4.0),
            FrequencyBand("theta", 4.0, 8.0),
        )
    )
    assert DEFAULT_FEATURE_SPEC.spec_hash == equivalent.spec_hash
    assert DEFAULT_FEATURE_SPEC.spec_hash != changed.spec_hash

    batch = extract_fnirs_patch_features(
        np.zeros((1, 1, 1, 20)),
        sample_rate_hz=10.0,
    )
    manifest = batch.manifest.to_dict()
    json.dumps(manifest)
    assert manifest["feature_spec_hash"] == DEFAULT_FEATURE_SPEC.spec_hash
    assert manifest["input_unit"] == CANONICAL_SIGNAL_UNIT
    assert all("volt" not in unit.lower() for unit in batch.feature_units)
    assert all("molar" not in unit.lower() for unit in batch.feature_units)


def test_invalid_layout_and_patch_size_fail_loudly():
    with pytest.raises(ValueError, match="patch_size is required"):
        extract_eeg_patch_features(
            np.zeros((1, 2, 400)),
            sample_rate_hz=200.0,
        )
    with pytest.raises(ValueError, match="does not match patch axis"):
        extract_fnirs_patch_features(
            np.zeros((1, 1, 2, 20)),
            sample_rate_hz=10.0,
            patch_size=10,
        )
