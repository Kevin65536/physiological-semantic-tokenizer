from __future__ import annotations

import math

import pytest
import torch

from comparative_methods.rebench_baselines_v1 import create_model


TASKS = ("motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual", "refed_regression")


def profile(task="refed_regression", modality="multimodal"):
    duration = 20 if task == "refed_regression" else (2 if task == "dsr" else 8)
    return dict(channel_names=["C3", "C4"], fnirs_channel_names=["S1D1:HbO", "S1D1:HbR"],
                sampling_rate_hz=200, fnirs_sampling_rate_hz=10,
                input_samples=duration * 200, fnirs_samples=duration * 10, baseline_modality=modality)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("modality,dimension", (("eeg", 12), ("fnirs", 8), ("multimodal", 20)))
def test_all_tasks_have_fixed_finite_parameter_free_features(task, modality, dimension):
    p = profile(task, modality)
    model = create_model(task, p, "cpu", trainable=True)
    eeg = torch.zeros(2, 2, p["input_samples"])
    fnirs = torch.zeros(2, 2, p["fnirs_samples"])
    values = model(eeg, fnirs)
    assert list(model.parameters()) == []
    assert values.shape == ((2, 20, dimension) if task == "refed_regression" else (2, dimension))
    assert bool(torch.isfinite(values).all())
    assert model.feature_dim == dimension
    assert model.supports_mask is True
    assert model.sequence_features is (task == "refed_regression")
    assert len(model.metadata["feature_names"]) == dimension
    torch.testing.assert_close(model(eeg, fnirs), values, rtol=0, atol=0)


@pytest.mark.parametrize("frequency,band", ((2, 0), (4, 1), (10, 2), (13, 3), (40, 4), (45, 4)))
@pytest.mark.parametrize("task", ("visual", "refed_regression"))
def test_bandpower_matches_sinusoid_variance_and_nonoverlapping_band_boundaries(frequency, band, task):
    p = profile(task, "eeg")
    time = torch.arange(p["input_samples"], dtype=torch.float64) / 200
    eeg = (2 * torch.sin(2 * math.pi * frequency * time))[None, None].expand(1, 2, -1)
    model = create_model(task, p, "cpu")
    features = model(eeg)
    channel = features[0, 0, :6] if task == "refed_regression" else features[0, :6]
    expected = torch.zeros(6, dtype=torch.float64)
    expected[band] = math.log1p(2.0)  # amplitude^2/2 = 2
    expected[-1] = 1
    torch.testing.assert_close(channel, expected, atol=1e-12, rtol=1e-12)
    half_amplitude = model(eeg / 2)
    half_band = half_amplitude[0, 0, band] if task == "refed_regression" else half_amplitude[0, band]
    torch.testing.assert_close(half_band, torch.tensor(math.log1p(0.5), dtype=torch.float64), atol=1e-12, rtol=1e-12)


def test_hb_statistics_use_original_valid_timestamps_and_population_sd():
    p = profile("refed_regression", "fnirs")
    time = torch.arange(200, dtype=torch.float64) / 10
    fnirs = (2 + 3 * time)[None, None].expand(1, 2, -1).clone()
    mask = torch.ones_like(fnirs, dtype=torch.bool)
    mask[:, 0, 1:4] = False
    model = create_model("refed_regression", p, "cpu")
    values = model(torch.empty(1, 0, 0), fnirs, fnirs_mask=mask)
    chosen = fnirs[0, 0, :10][mask[0, 0, :10]]
    torch.testing.assert_close(values[0, 0, 0], chosen.mean())
    torch.testing.assert_close(values[0, 0, 1], chosen.std(correction=0))
    torch.testing.assert_close(values[0, :, 2], torch.full((20,), 3.0, dtype=torch.float64))
    torch.testing.assert_close(values[0, 0, 3], torch.tensor(0.7, dtype=torch.float64))


def test_partial_masks_ignore_padding_values_and_preserve_feature_and_input_support():
    p = profile()
    model = create_model("refed_regression", p, "cpu")
    eeg, fnirs = torch.randn(2, 2, 4000), torch.randn(2, 2, 200)
    em, hm = torch.ones_like(eeg, dtype=torch.bool), torch.ones_like(fnirs, dtype=torch.bool)
    em[0, :, 200:] = False
    hm[0, :, 10:] = False
    em[1, 0, -100:] = False
    hm[1, 0, -5:] = False
    original = model(eeg, fnirs, eeg_mask=em, fnirs_mask=hm)
    altered_eeg = torch.where(em, eeg, float("nan"))
    altered_fnirs = torch.where(hm, fnirs, float("inf"))
    actual = model(altered_eeg, altered_fnirs, eeg_mask=em, fnirs_mask=hm)
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert bool((actual[0, 1:] == 0).all())
    # The last partial second retains real input coverage, independently per channel.
    assert actual[1, -1, 5] == 0.5
    assert actual[1, -1, 15] == 0.5
    feature_mask = model.feature_mask(eeg, fnirs, eeg_mask=em, fnirs_mask=hm)
    assert feature_mask.shape == actual.shape
    assert bool((~feature_mask[0, 1:, :5]).all())
    assert bool(feature_mask[0, 1:, 5].all())  # zero coverage remains observed metadata
    expected = torch.ones(2, 20, dtype=torch.bool)
    expected[0, 1:] = False
    torch.testing.assert_close(model.sequence_feature_mask(eeg, fnirs, eeg_mask=em, fnirs_mask=hm), expected)
    with pytest.raises(TypeError, match="target_valid_mask"):
        model(eeg, fnirs, eeg_mask=em, fnirs_mask=hm, target_valid_mask=torch.zeros(2, 2, 20, dtype=torch.bool))


def test_single_sample_and_empty_input_have_explicit_feature_masks():
    model = create_model("refed_regression", profile(), "cpu")
    eeg, fnirs = torch.zeros(1, 2, 4000), torch.zeros(1, 2, 200)
    em, hm = torch.zeros_like(eeg, dtype=torch.bool), torch.zeros_like(fnirs, dtype=torch.bool)
    eeg[0, 0, 50] = 7
    fnirs[0, 0, 5] = 3
    em[0, 0, 50] = True
    hm[0, 0, 5] = True
    values = model(eeg, fnirs, eeg_mask=em, fnirs_mask=hm)
    support = model.feature_mask(eeg, fnirs, eeg_mask=em, fnirs_mask=hm)
    assert not bool(support[0, 0, :5].any())
    torch.testing.assert_close(values[0, 0, 12:16], torch.tensor([3.0, 0.0, 0.0, 0.1]))
    torch.testing.assert_close(support[0, 0, 12:16], torch.tensor([True, True, False, True]))
    assert bool((values[0, 1:] == 0).all())
    assert model.sequence_feature_mask(eeg, fnirs, eeg_mask=em, fnirs_mask=hm)[0, 0]


def test_multimodal_is_exact_concatenation_and_unselected_input_cannot_change_features():
    p = profile("dsr")
    eeg, fnirs = torch.randn(1, 2, 400), torch.randn(1, 2, 20)
    ee = create_model("dsr", {**p, "baseline_modality": "eeg"}, "cpu")
    hh = create_model("dsr", {**p, "baseline_modality": "fnirs"}, "cpu")
    both = create_model("dsr", p, "cpu")
    torch.testing.assert_close(both(eeg, fnirs), torch.cat((ee(eeg, fnirs), hh(eeg, fnirs)), -1))
    torch.testing.assert_close(ee(eeg, fnirs), ee(eeg, fnirs * float("nan")))
    torch.testing.assert_close(hh(eeg, fnirs), hh(eeg * float("nan"), fnirs))
    with pytest.raises(ValueError, match="must be finite"):
        both(eeg * float("nan"), fnirs)
    with pytest.raises(ValueError, match="same.*shape"):
        both(eeg, fnirs, eeg_mask=torch.ones(1, 400, dtype=torch.bool))
    with pytest.raises(ValueError, match="nonempty"):
        create_model("dsr", {**p, "fnirs_channel_names": []}, "cpu")
    with pytest.raises(TypeError, match="unknown"):
        both(eeg, fnirs, unknown=True)
