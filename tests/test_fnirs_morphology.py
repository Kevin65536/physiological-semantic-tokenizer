"""Synthetic invariants for original-axis fNIRS morphology diagnostics."""

import json

import numpy as np
import pytest

from src.metrics.fnirs_morphology import MorphologySettings, fnirs_morphology_metrics


def _waveforms(duration=60, dt=0.1):
    time = np.arange(0, duration, dt)
    hbo = np.sin(2 * np.pi * time / 12) + 0.17 * np.sin(2 * np.pi * time / 5)
    return time, np.column_stack((hbo, -0.6 * hbo))


def _score(target, prediction, time, **kwargs):
    result = fnirs_morphology_metrics(target, prediction, time_s=time, **kwargs)
    # This is the actual serialization contract, including nested event details.
    json.dumps(result, allow_nan=False)
    return result


def _pulses(time, centers, width=0.6):
    return sum(np.exp(-0.5 * ((time - center) / width) ** 2) for center in centers)


def test_identity_scores_shape_slopes_events_and_joint_relationship():
    time, target = _waveforms()
    result = _score(target, target, time, training_sd=[2, 3])
    assert result["schema"] == "fnirs_morphology_v1"
    assert result["settings"]["local_window_s"] == 20
    for component in result["components"]:
        assert component["status"] == "ok"
        assert component["state"] == "dynamic"
        assert component["requested_points"] == component["scored_points"] == len(time)
        assert component["global_pearson_r"] == pytest.approx(1)
        assert component["within_segment_r"] == pytest.approx(1)
        assert component["within_segment_amplitude_ratio"] == pytest.approx(1)
        assert component["bias_training_sd"] == 0
        assert component["nrmse_training_sd"] == 0
        assert component["local_correlation"]["median"] == pytest.approx(1)
        assert component["local_correlation"]["valid_window_count"] == 9
        for slope in component["slope_direction"]:
            assert slope["target_slope_energy_weighted_unrecovered_rate"] == 0
        for kind in ("peaks", "troughs"):
            events = component["events"][kind]
            assert events["target_count"] > 0
            assert events["f1"] == 1
            assert events["timing_absolute_median_s"] == 0
    pair = result["pair"]
    assert pair["target_within_segment_hbo_hbr_r"] == pytest.approx(-1)
    assert pair["correlation_absolute_difference"] == 0
    assert pair["local_correlation_difference"]["absolute_median"] == 0
    assert all(entry["target_slope_product_weighted_unrecovered_rate"] == 0 for entry in pair["slope_relationship"])


def test_positive_gain_and_bias_preserve_shape_but_expose_amplitude_and_bias():
    time, target = _waveforms()
    prediction = target * [2.5, 0.4] + [9, -7]
    result = _score(target, prediction, time, training_sd=[2, 4])
    for column, component in enumerate(result["components"]):
        assert component["within_segment_r"] == pytest.approx(1)
        assert component["local_correlation"]["p10"] == pytest.approx(1)
        assert component["within_segment_amplitude_ratio"] == pytest.approx([2.5, 0.4][column])
        residual = (prediction[:, column] - target[:, column]) / [2, 4][column]
        assert component["bias_training_sd"] == pytest.approx(np.mean(residual))
        assert component["nrmse_training_sd"] == pytest.approx(np.sqrt(np.mean(residual ** 2)))
        assert component["events"]["peaks"]["f1"] == 1
    assert result["pair"]["correlation_absolute_difference"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("flip_both", [False, True])
def test_sign_flips_are_separate_component_and_relationship_failures(flip_both):
    time, target = _waveforms()
    prediction = target * ([-1, -1] if flip_both else [-1, 1])
    result = _score(target, prediction, time)
    assert result["components"][0]["within_segment_r"] == pytest.approx(-1)
    assert result["components"][0]["local_correlation"]["median"] == pytest.approx(-1)
    for slope in result["components"][0]["slope_direction"]:
        assert slope["target_slope_energy_weighted_unrecovered_rate"] == 1
    expected = 0 if flip_both else 1
    for slope in result["pair"]["slope_relationship"]:
        assert slope["target_slope_product_weighted_unrecovered_rate"] == expected
    assert result["pair"]["correlation_absolute_difference"] == pytest.approx(0 if flip_both else 2)


def test_known_shift_is_measured_without_alignment():
    time = np.arange(0, 60, 0.05)
    y = _pulses(time, [10, 25, 40, 50])
    p = _pulses(time, [11, 26, 41, 51])
    result = _score(np.column_stack((y, -y)), np.column_stack((p, -p)), time)
    component = result["components"][0]
    peaks = component["events"]["peaks"]
    assert peaks["target_count"] == peaks["matched_count"] == 4
    assert peaks["timing_signed_median_s"] == pytest.approx(1)
    assert peaks["timing_absolute_p90_s"] == pytest.approx(1)
    assert component["within_segment_r"] < 0.6
    assert component["local_correlation"]["median"] < 0.8


def test_missing_and_extra_peaks_use_full_event_denominators():
    time = np.arange(0, 60, 0.05)
    y = _pulses(time, [10, 25, 40])
    p = _pulses(time, [10, 40, 52])
    result = _score(np.column_stack((y, -y)), np.column_stack((p, -p)), time)
    peaks = result["components"][0]["events"]["peaks"]
    assert (peaks["target_count"], peaks["prediction_count"], peaks["matched_count"]) == (3, 3, 2)
    assert peaks["precision"] == pytest.approx(2 / 3)
    assert peaks["recall"] == pytest.approx(2 / 3)
    assert peaks["f1"] == pytest.approx(2 / 3)


def test_wider_pulses_retain_timing_but_have_width_error():
    time = np.arange(0, 60, 0.05)
    # Interior troughs establish both bases; an isolated pulse ending at its
    # prominence bases is conservatively censored by the width contract.
    centers = np.arange(-5, 66, 10)
    y = _pulses(time, centers, width=0.6)
    p = _pulses(time, centers, width=1.2)
    result = _score(np.column_stack((y, -y)), np.column_stack((p, -p)), time)
    peaks = result["components"][0]["events"]["peaks"]
    assert peaks["timing_absolute_median_s"] == 0
    assert peaks["width_valid_match_count"] > 0
    assert peaks["width_absolute_median_s"] == pytest.approx(2 * np.sqrt(2 * np.log(2)) * 0.6, rel=0.01)


def test_no_matches_has_no_timing_zero():
    time = np.arange(0, 60, 0.1)
    y = _pulses(time, [10, 30])
    p = _pulses(time, [15, 35])
    result = _score(np.column_stack((y, -y)), np.column_stack((p, -p)), time)
    peaks = result["components"][0]["events"]["peaks"]
    assert peaks["matched_count"] == 0
    assert peaks["f1"] == 0
    assert peaks["timing_absolute_median_s"] is None
    assert peaks["timing_absolute_p90_s"] is None


def test_ordered_matching_maximizes_count_before_minimizing_delay():
    time = np.arange(0, 30, 0.05)
    y = _pulses(time, [10, 13], width=0.2)
    p = _pulses(time, [8.1, 11.9], width=0.2)
    result = _score(np.column_stack((y, -y)), np.column_stack((p, -p)), time)
    # Nearest-first would match 10 -> 11.9 and lose the second reference.
    peaks = result["components"][0]["events"]["peaks"]
    assert peaks["matched_count"] == 2
    details = result["components"][0]["segments"][0]["events"]["peaks"]["matches"]
    assert [entry["prediction_time_s"] for entry in details] == pytest.approx([8.1, 11.9])


def test_endpoint_controlled_width_is_censored_but_event_remains_counted():
    time = np.arange(-1, 1.001, 0.01)
    y = np.exp(-0.5 * (time / 0.6) ** 2)
    result = _score(np.column_stack((y, -y)), np.column_stack((y, -y)), time)
    component = result["components"][0]
    event = component["segments"][0]["events"]["peaks"]["target"][0]
    assert event["width_censored"] is True
    assert event["width_s"] is None
    assert event["segment_local_width_s"] > 0
    assert component["events"]["peaks"]["matched_count"] == 1
    assert component["events"]["peaks"]["width_valid_match_count"] == 0
    assert component["events"]["peaks"]["width_absolute_median_s"] is None


def test_flat_prediction_reports_collapse_and_event_misses():
    time, target = _waveforms()
    result = _score(target, np.zeros_like(target), time)
    for component in result["components"]:
        assert component["status"] == "ok"
        assert component["state"] == "collapsed"
        assert component["collapsed"] is True
        assert component["within_segment_r"] is None
        assert component["within_segment_amplitude_ratio"] == 0
        assert component["local_correlation"]["valid_window_count"] == 0
        assert component["local_correlation"]["total_window_count"] == 9
        assert component["events"]["peaks"]["recall"] == 0
        assert component["events"]["peaks"]["f1"] == 0
        assert component["events"]["peaks"]["timing_absolute_median_s"] is None
        assert all(item["target_slope_energy_weighted_unrecovered_rate"] == 1 for item in component["slope_direction"])


def test_flat_target_keeps_spurious_prediction_and_undefined_reference_rates():
    time, prediction = _waveforms()
    result = _score(np.zeros_like(prediction), prediction, time)
    for component in result["components"]:
        assert component["spurious"] is True
        assert component["state"] == "spurious"
        peaks = component["events"]["peaks"]
        assert peaks["precision"] == 0
        assert peaks["recall"] is None
        assert peaks["f1"] == 0
        assert all(item["target_slope_energy_weighted_unrecovered_rate"] is None for item in component["slope_direction"])
    flat = _score(np.ones_like(prediction), np.ones_like(prediction), time)
    for component in flat["components"]:
        assert component["state"] == "both_low_dynamic"
        assert all(component["events"]["peaks"][key] is None for key in ("precision", "recall", "f1"))
        assert component["bias_training_sd"] is None
        assert component["nrmse_training_sd"] is None
    assert result["pair"]["slope_relationship"][0]["na_reason"] == "no_target_joint_variation"


def test_nonfinite_support_fails_component_without_changing_denominator():
    time, target = _waveforms()
    prediction = target.copy()
    prediction[100, 0] = np.nan
    result = _score(target, prediction, time)
    assert result["components"][0]["status"] == "failed_nonfinite"
    assert result["components"][0]["requested_points"] == len(time)
    assert result["components"][0]["scored_points"] == 0
    assert result["components"][1]["status"] == "ok"
    assert result["components"][1]["scored_points"] == len(time)
    assert result["pair"]["status"] == "failed_nonfinite"
    assert result["pair"]["requested_points"] == len(time)
    assert result["pair"]["scored_points"] == 0
    mask = np.ones(len(time), dtype=bool)
    mask[100] = False
    masked = _score(target, prediction, time, score_mask=mask)
    assert masked["components"][0]["status"] == "ok"
    assert masked["components"][0]["scored_points"] == len(time) - 1


def test_pair_nonfinite_check_uses_its_own_joint_support():
    time, target = _waveforms()
    prediction = target.copy()
    prediction[100, 0] = np.inf
    mask = np.ones(target.shape, dtype=bool)
    mask[100, 1] = False
    result = _score(target, prediction, time, score_mask=mask)
    assert result["components"][0]["status"] == "failed_nonfinite"
    assert result["pair"]["status"] == "ok"
    assert result["pair"]["scored_points"] == len(time) - 1


def test_mask_gap_cannot_create_events_windows_or_matches_across_segments():
    time = np.arange(0, 60, 0.1)
    target = np.column_stack((time, -time))
    prediction = target.copy()
    prediction[300:] += [100, -100]
    mask = np.ones(len(time), dtype=bool)
    mask[290:310] = False
    result = _score(target, prediction, time, score_mask=mask)
    component = result["components"][0]
    assert component["segment_count"] == 2
    assert component["global_pearson_r"] is None
    assert component["within_segment_r"] == pytest.approx(1)
    assert component["within_segment_amplitude_ratio"] == pytest.approx(1)
    assert component["events"]["peaks"]["target_count"] == 0
    assert component["events"]["peaks"]["prediction_count"] == 0
    assert component["local_correlation"]["valid_window_count"] == 0
    for window in component["local_correlation"]["windows"]:
        assert window["end_s_exclusive"] <= 29 or window["start_s"] >= 31
        assert window["na_reason"] == "both_low_dynamic"
    assert result["pair"]["segment_count"] == 2


def test_mask_gap_cannot_pair_nearby_events_on_opposite_sides():
    time = np.arange(0, 40, 0.05)
    y = _pulses(time, [19.5], width=0.1)
    p = _pulses(time, [20.5], width=0.1)
    mask = np.abs(time - 20) > 0.15
    result = _score(np.column_stack((y, -y)), np.column_stack((p, -p)), time, score_mask=mask)
    peaks = result["components"][0]["events"]["peaks"]
    assert peaks["target_count"] == peaks["prediction_count"] == 1
    assert peaks["matched_count"] == 0


def test_pooled_correlation_does_not_zscore_away_segment_amplitude_errors():
    time = np.arange(0, 60, 0.1)
    signal = np.sin(2 * np.pi * time / 6)
    y = signal * np.where(time < 30, 1, 10)
    p = signal * np.where(time < 30, 10, 1)
    mask = time != 30
    result = _score(np.column_stack((y, -y)), np.column_stack((p, -p)), time, score_mask=mask)
    component = result["components"][0]
    assert component["within_segment_r"] < 0.3
    assert component["local_correlation"]["median"] == pytest.approx(1)


def test_timestamp_jump_splits_support_and_irregular_sampling_is_rejected():
    time, target = _waveforms()
    jumped = time.copy()
    jumped[300:] += 7
    result = _score(target, target, jumped)
    assert result["components"][0]["segment_count"] == 2
    assert result["components"][0]["global_pearson_r"] is None
    assert result["pair"]["segment_count"] == 2
    irregular = time.copy()
    irregular[100] += 0.01
    with pytest.raises(ValueError, match="uniform"):
        _score(target, target, irregular)
    for invalid in (time[::-1], np.zeros_like(time), np.full_like(time, np.nan)):
        with pytest.raises(ValueError):
            _score(target, target, invalid)


@pytest.mark.parametrize("scale", [1e-12, 1e-6, 1e6])
def test_native_unit_scaling_preserves_shape_and_rates(scale):
    time, target = _waveforms()
    prediction = target * [0.7, -1.2]
    baseline = _score(target, prediction, time, training_sd=[2, 3])
    scaled = _score(target * scale, prediction * scale, time, training_sd=np.array([2, 3]) * scale)
    for normal, converted in zip(baseline["components"], scaled["components"]):
        assert converted["within_segment_r"] == pytest.approx(normal["within_segment_r"])
        assert converted["within_segment_amplitude_ratio"] == pytest.approx(normal["within_segment_amplitude_ratio"])
        assert converted["nrmse_training_sd"] == pytest.approx(normal["nrmse_training_sd"])
        assert converted["local_correlation"]["median"] == pytest.approx(normal["local_correlation"]["median"])
        for event_type in ("peaks", "troughs"):
            assert converted["events"][event_type]["target_count"] == normal["events"][event_type]["target_count"]
        for a, b in zip(normal["slope_direction"], converted["slope_direction"]):
            assert b["target_slope_energy_weighted_unrecovered_rate"] == pytest.approx(a["target_slope_energy_weighted_unrecovered_rate"])


def test_linear_detrend_roundoff_does_not_become_local_shape_information():
    time = np.arange(0, 60, 0.1)
    y = 0.37 * time + 2.3
    p = 0.41 * time + 3.7
    result = _score(np.column_stack((y, -y)), np.column_stack((p, -p)), time)
    local = result["components"][0]["local_correlation"]
    assert local["total_window_count"] == 9
    assert local["valid_window_count"] == 0
    assert local["median"] is None
    assert result["pair"]["local_correlation_difference"]["valid_window_count"] == 0


def test_tiny_but_resolvable_dynamics_are_not_filtered_by_training_sd():
    time, target = _waveforms()
    result = _score(target * 1e-12, target * 1e-12, time, training_sd=[1e10, 1e10])
    assert result["components"][0]["state"] == "dynamic"
    assert result["components"][0]["within_segment_r"] == pytest.approx(1)
    assert result["components"][0]["events"]["peaks"]["target_count"] > 0


def test_giant_offset_exposes_precision_floor_instead_of_shape_certainty():
    time, target = _waveforms()
    result = _score(target + 1e16, target + 1e16, time)
    assert result["components"][0]["state"] == "both_low_dynamic"
    assert result["components"][0]["within_segment_r"] is None
    assert result["components"][0]["local_correlation"]["valid_window_count"] == 0


@pytest.mark.parametrize("kwargs", [
    {"target": np.ones((4, 1))},
    {"prediction": np.ones((3, 2))},
    {"time_s": np.arange(3)},
    {"score_mask": np.ones((4, 3))},
    {"score_mask": np.full(4, 0.3)},
    {"training_sd": [1, 0]},
    {"training_sd": [1, 2, 3]},
    {"settings": {}},
])
def test_schema_errors_are_rejected(kwargs):
    arguments = {"target": np.ones((4, 2)), "prediction": np.ones((4, 2)), "time_s": np.arange(4)}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        fnirs_morphology_metrics(**arguments)


def test_empty_support_is_explicit_and_settings_are_validated():
    time, target = _waveforms()
    result = _score(target, target, time, score_mask=np.zeros(len(time), dtype=bool))
    assert all(component["status"] == "no_support" for component in result["components"])
    assert result["pair"]["status"] == "no_support"
    for kwargs in ({"local_window_s": 0}, {"event_match_tolerance_s": -1}, {"timestamp_gap_factor": 1}, {"slope_windows_s": (2, 2)}):
        with pytest.raises(ValueError):
            MorphologySettings(**kwargs)
