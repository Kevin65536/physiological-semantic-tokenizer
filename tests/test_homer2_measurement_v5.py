"""The general cache preserves signals without legacy derivative suppression."""
import numpy as np
import pytest

from src.data import homer2_preprocessing as hp


@pytest.mark.parametrize('method', ['none', 'mne_tddr'])
def test_v5_optical_is_identical_to_explicit_v4(method):
    t = np.arange(1201) / 10
    od = np.stack((.02 * np.sin(t), .001 * t), axis=1)[:, None, :]
    options = dict(dataset_id='eeg_fnirs_single_trial', sample_rate_hz=10.,
                   entry_stage='optical_density', wavelengths_nm=(760., 850.),
                   motion_method=method, retain_feature_boundary=True)
    old = hp.apply_homer2_aligned_contract(od, processing_schema=hp.MEASUREMENT_ALIGNMENT_V4_SCHEMA, **options)
    new = hp.apply_homer2_aligned_contract(od, processing_schema=hp.MEASUREMENT_ALIGNMENT_V5_SCHEMA, **options)
    np.testing.assert_array_equal(new.values, old.values)
    np.testing.assert_array_equal(new.pre_linear_optical_density, old.pre_linear_optical_density)
    np.testing.assert_array_equal(new.processed_valid_mask, old.processed_valid_mask)


@pytest.mark.parametrize('dataset,unit,factor', [
    ('simultaneous_eeg_nirs', 'mmol/L', 1000.),
    ('refed', 'unknown', 1.),
    ('visual_cognitive_motivation', 'unknown', 1.),
])
def test_released_hb_skips_motion_and_mbll_but_preserves_units_and_support(dataset, unit, factor, monkeypatch):
    t = np.arange(1201) / 10
    hb = np.stack((.001 * t + .002 * np.sin(t), -.0004 * t), axis=1)
    def forbidden(*args, **kwargs):
        raise AssertionError('No motion/MBLL may touch published Hb')
    monkeypatch.setattr(hp, 'robust_derivative_motion_suppression', forbidden)
    monkeypatch.setattr(hp, 'modified_beer_lambert', forbidden)
    options = dict(dataset_id=dataset, sample_rate_hz=10., entry_stage='chromophore',
                   native_unit=unit, unit_evidence='synthetic known header' if factor != 1 else '',
                   processing_schema=hp.MEASUREMENT_ALIGNMENT_V5_SCHEMA, motion_method='none',
                   retain_feature_boundary=True)
    result = hp.apply_homer2_aligned_contract(hb, **options)
    np.testing.assert_array_equal(result.pre_linear_values, hb * factor)
    reference, _ = hp.bandpass_fnirs(hb * factor, sample_rate_hz=10.)
    np.testing.assert_array_equal(result.values, reference)
    assert result.values.dtype == np.float64
    assert result.processed_valid_mask.all()
    mask = np.ones_like(hb, bool)
    mask[80:100, 0] = False
    changed = hb.copy()
    changed[~mask] = 1e9
    first = hp.apply_homer2_aligned_contract(hb, valid_mask=mask, **options)
    second = hp.apply_homer2_aligned_contract(changed, valid_mask=mask, **options)
    np.testing.assert_array_equal(first.values, second.values)
    assert not first.processed_valid_mask[:, 0].any()
    assert first.processed_valid_mask[:, 1].all()


@pytest.mark.parametrize('method', [None, 'legacy', 'mne_tddr'])
def test_v5_released_hb_rejects_implicit_or_unsupported_motion(method):
    with pytest.raises(ValueError):
        hp.apply_homer2_aligned_contract(np.ones((100, 2)), dataset_id='refed',
            sample_rate_hz=10., entry_stage='chromophore', motion_method=method,
            processing_schema=hp.MEASUREMENT_ALIGNMENT_V5_SCHEMA)
