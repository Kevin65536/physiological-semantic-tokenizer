"""Array-only fNIRS shape diagnostics on the original, unaligned time axis.

These engineering diagnostics do not establish physiological validity.  Each
mask-contiguous, uniformly sampled segment is scored independently: missing
support is never bridged, and a nonfinite value on requested support fails the
whole affected component rather than reducing its denominator.  All undefined
statistics are ``None`` so the result can be written with ``allow_nan=False``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.signal import find_peaks, peak_widths


@dataclass(frozen=True)
class MorphologySettings:
    """Engineering defaults, reported verbatim; none is a pass threshold.

    A signal is numerically dynamic only when its population SD exceeds
    ``numerical_relative_floor * max(abs(original_signal))``.  Detrended
    residuals use the *original* local signal for that scale, so floating point
    residuals of an exact line do not become apparent shape information.
    """

    local_window_s: float = 20.0
    local_step_s: float = 5.0
    slope_windows_s: tuple[float, ...] = (2.0, 5.0)
    slope_step_s: float = 1.0
    event_prominence_sd: float = 0.5
    event_min_distance_s: float = 2.0
    event_match_tolerance_s: float = 2.0
    timestamp_gap_factor: float = 1.5
    timestamp_uniform_rtol: float = 1e-3
    timestamp_uniform_atol_s: float = 1e-9
    numerical_relative_floor: float = 64.0 * np.finfo(np.float64).eps

    def __post_init__(self) -> None:
        positive = (
            "local_window_s", "local_step_s", "slope_step_s",
            "event_prominence_sd", "event_min_distance_s",
            "timestamp_uniform_rtol", "numerical_relative_floor",
        )
        for key in positive:
            value = getattr(self, key)
            if (
                not isinstance(value, (int, float, np.integer, np.floating))
                or isinstance(value, bool) or not np.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")
        for key in ("event_match_tolerance_s", "timestamp_uniform_atol_s"):
            value = getattr(self, key)
            if (
                not isinstance(value, (int, float, np.integer, np.floating))
                or isinstance(value, bool) or not np.isfinite(value) or value < 0
            ):
                raise ValueError(f"{key} must be finite and nonnegative")
        if (
            not isinstance(self.timestamp_gap_factor, (int, float, np.integer, np.floating))
            or not np.isfinite(self.timestamp_gap_factor) or self.timestamp_gap_factor <= 1
        ):
            raise ValueError("timestamp_gap_factor must exceed one")
        try:
            widths = tuple(float(value) for value in self.slope_windows_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("slope_windows_s must contain positive durations") from exc
        if not widths or any(not np.isfinite(value) or value <= 0 for value in widths):
            raise ValueError("slope_windows_s must contain positive durations")
        if len(set(widths)) != len(widths):
            raise ValueError("slope_windows_s must not contain duplicates")
        object.__setattr__(self, "slope_windows_s", widths)


def _numeric_array(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a real numeric array")
    return np.asarray(raw, dtype=np.float64)


def _validate_inputs(target, prediction, time_s, score_mask, training_sd, settings):
    if settings is None:
        settings = MorphologySettings()
    if not isinstance(settings, MorphologySettings):
        raise ValueError("settings must be MorphologySettings or None")
    target = _numeric_array(target, "target")
    prediction = _numeric_array(prediction, "prediction")
    if target.ndim != 2 or target.shape[1] != 2 or len(target) == 0:
        raise ValueError("target must have nonempty shape [T, 2] in HbO/HbR order")
    if prediction.shape != target.shape:
        raise ValueError("prediction must match target shape [T, 2]")
    time = _numeric_array(time_s, "time_s")
    if time.shape != (len(target),) or not np.all(np.isfinite(time)):
        raise ValueError("time_s must be finite with shape [T]")
    differences = np.diff(time)
    if np.any(differences <= 0):
        raise ValueError("time_s must be strictly increasing")
    gaps = np.zeros(max(0, len(time) - 1), dtype=bool)
    dt = None
    if len(differences):
        nominal = float(np.median(differences))
        gaps = differences > settings.timestamp_gap_factor * nominal
        dt = float(np.median(differences[~gaps]))
        gaps = differences > settings.timestamp_gap_factor * dt
        if np.any(~np.isclose(
            differences[~gaps], dt, rtol=settings.timestamp_uniform_rtol,
            atol=settings.timestamp_uniform_atol_s,
        )):
            raise ValueError("time_s must be uniform within timestamp-separated segments")
    if score_mask is None:
        mask = np.ones(target.shape, dtype=bool)
    else:
        raw_mask = np.asarray(score_mask)
        if raw_mask.shape not in ((len(target),), target.shape):
            raise ValueError("score_mask must have shape [T] or [T, 2]")
        if raw_mask.dtype.kind not in "biuf" or not np.all(np.isin(raw_mask, [0, 1])):
            raise ValueError("score_mask must contain only boolean or binary values")
        mask = raw_mask.astype(bool)
        if mask.ndim == 1:
            mask = np.repeat(mask[:, None], 2, axis=1)
    if training_sd is None:
        scales = None
    else:
        scales = _numeric_array(training_sd, "training_sd")
        if scales.ndim == 0:
            scales = np.repeat(scales, 2)
        if scales.shape != (2,) or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("training_sd must be a positive finite scalar or [2] array")
    return target, prediction, time, mask, scales, settings, gaps, dt


def _segments(mask: np.ndarray, gaps: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    split = (np.diff(indices) != 1) | gaps[indices[:-1]]
    return list(np.split(indices, np.flatnonzero(split) + 1))


def _sd(centered: np.ndarray) -> float:
    if not len(centered):
        return 0.0
    scale = float(np.max(np.abs(centered)))
    if scale == 0:
        return 0.0
    return float(scale * np.sqrt(np.mean((centered / scale) ** 2)))


def _center(values: np.ndarray) -> np.ndarray:
    # Subtract a nearby origin first to retain small dynamics on a large bias.
    shifted = values - values[0]
    return shifted - np.mean(shifted)


def _dynamic(centered: np.ndarray, original: np.ndarray, settings) -> bool:
    return bool(
        len(centered) >= 2
        and _sd(centered) > settings.numerical_relative_floor * float(np.max(np.abs(original)))
    )


def _correlation(y, p, settings, *, reference_y=None, reference_p=None):
    yc, pc = _center(y), _center(p)
    dy = _dynamic(yc, y if reference_y is None else reference_y, settings)
    dp = _dynamic(pc, p if reference_p is None else reference_p, settings)
    if not dy or not dp:
        reason = "both_low_dynamic" if not dy and not dp else (
            "target_low_dynamic" if not dy else "prediction_low_dynamic"
        )
        return None, reason
    ys, ps = yc / np.max(np.abs(yc)), pc / np.max(np.abs(pc))
    r = np.dot(ys, ps) / np.sqrt(np.dot(ys, ys) * np.dot(ps, ps))
    return float(np.clip(r, -1, 1)), None


def _bounds(indices, time, segment_id):
    return {
        "segment_id": segment_id,
        "start_index": int(indices[0]), "end_index_inclusive": int(indices[-1]),
        "start_s": float(time[indices[0]]), "end_s": float(time[indices[-1]]),
        "point_count": len(indices),
    }


def _windows(indices, time, duration, step, dt, minimum_points):
    if dt is None or len(indices) < minimum_points:
        return
    times = time[indices]
    tolerance = max(1e-10, dt * 1e-7)
    available = times[-1] + dt - times[0]
    if available + tolerance < duration:
        return
    count = int(np.floor((available - duration + tolerance) / step)) + 1
    for number in range(count):
        start = times[0] + number * step
        end = start + duration
        lo = int(np.searchsorted(times, start - tolerance, side="left"))
        hi = int(np.searchsorted(times, end - tolerance, side="left"))
        chosen = indices[lo:hi]
        if len(chosen) >= minimum_points:
            yield chosen, float(start), float(end)


def _slope(values, times, settings):
    tc = _center(times)
    vc = _center(values)
    denominator = float(np.dot(tc, tc))
    if denominator == 0 or not _dynamic(vc, values, settings):
        return 0.0
    result = float(np.dot(tc, vc) / denominator)
    numerical_floor = settings.numerical_relative_floor * float(np.max(np.abs(values)))
    return 0.0 if abs(result) * float(np.max(np.abs(tc))) <= numerical_floor else result


def _detrend(values, times):
    tc, vc = _center(times), _center(values)
    return vc - tc * (np.dot(tc, vc) / np.dot(tc, tc))


def _summary(values):
    return {
        "median": float(np.median(values)) if values else None,
        "p10": float(np.percentile(values, 10)) if values else None,
    }


def _local_correlation(y, p, indices, time, dt, settings, segment_id):
    details = []
    for chosen, start, end in _windows(
        indices, time, settings.local_window_s, settings.local_step_s, dt, 3,
    ):
        r, reason = _correlation(
            _detrend(y[chosen], time[chosen]), _detrend(p[chosen], time[chosen]),
            settings, reference_y=y[chosen], reference_p=p[chosen],
        )
        details.append({
            "segment_id": segment_id, "start_s": start, "end_s_exclusive": end,
            "point_count": len(chosen), "r": r, "na_reason": reason,
        })
    return details


def _local_summary(details):
    values = [entry["r"] for entry in details if entry["r"] is not None]
    return {**_summary(values), "valid_window_count": len(values),
            "total_window_count": len(details), "windows": details}


def _slope_details(y, p, indices, time, dt, settings, segment_id, width):
    details = []
    for chosen, start, end in _windows(
        indices, time, width, settings.slope_step_s, dt, 2,
    ):
        sy, sp = _slope(y[chosen], time[chosen], settings), _slope(p[chosen], time[chosen], settings)
        details.append({
            "segment_id": segment_id, "start_s": start, "end_s_exclusive": end,
            "point_count": len(chosen), "target_slope_per_s": sy,
            "prediction_slope_per_s": sp,
            "unrecovered": bool(np.sign(sy) * np.sign(sp) <= 0) if sy != 0 else None,
        })
    return details


def _slope_summary(details, width):
    informative = [entry for entry in details if entry["unrecovered"] is not None]
    rate = None
    if informative:
        magnitudes = np.asarray([abs(entry["target_slope_per_s"]) for entry in informative])
        weights = (magnitudes / np.max(magnitudes)) ** 2
        failed = np.asarray([entry["unrecovered"] for entry in informative], dtype=bool)
        rate = float(weights[failed].sum() / weights.sum())
    return {
        "window_s": width, "target_slope_energy_weighted_unrecovered_rate": rate,
        "informative_window_count": len(informative), "total_window_count": len(details),
        "na_reason": "no_target_slope_energy" if rate is None else None,
        "windows": details,
    }


def _events(values, times, settings, dt, kind):
    centered = _center(values)
    if dt is None or len(values) < 3 or not _dynamic(centered, values, settings):
        return []
    z = centered / _sd(centered)
    if kind == "troughs":
        z = -z
    peaks, properties = find_peaks(
        z, prominence=settings.event_prominence_sd,
        distance=max(1, int(np.ceil(settings.event_min_distance_s / dt - 1e-10))),
    )
    if not len(peaks):
        return []
    _, _, left, right = peak_widths(
        z, peaks, rel_height=0.5,
        prominence_data=(properties["prominences"], properties["left_bases"], properties["right_bases"]),
    )
    output = []
    for index, peak in enumerate(peaks):
        lb, rb = int(properties["left_bases"][index]), int(properties["right_bases"][index])
        at_boundary = lb == 0 or rb == len(values) - 1
        # The higher base controls prominence. If that base touches an
        # endpoint, its actual minimum is not observed; equal endpoint bases
        # are also conservative censored cases (a symmetric truncated pulse).
        controlling_boundary = (
            (lb == 0 and z[lb] >= z[rb] - settings.numerical_relative_floor)
            or (rb == len(values) - 1 and z[rb] >= z[lb] - settings.numerical_relative_floor)
        )
        censored = bool(controlling_boundary or left[index] <= 0 or right[index] >= len(values) - 1)
        left_s = float(np.interp(left[index], np.arange(len(times)), times))
        right_s = float(np.interp(right[index], np.arange(len(times)), times))
        output.append({
            "time_s": float(times[peak]), "sample_index_in_segment": int(peak),
            "prominence_sd": float(properties["prominences"][index]),
            "width_s": None if censored else right_s - left_s,
            "segment_local_width_s": right_s - left_s,
            "width_censored": censored, "prominence_base_at_boundary": bool(at_boundary),
            "left_half_prominence_s": None if censored else left_s,
            "right_half_prominence_s": None if censored else right_s,
        })
    return output


def _match_events(target, prediction, tolerance):
    """Order-preserving maximum cardinality, then minimum absolute delay."""
    n, m = len(target), len(prediction)
    counts = np.zeros((n + 1, m + 1), dtype=np.int32)
    costs = np.zeros((n + 1, m + 1), dtype=np.float64)
    actions = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            count, cost, action = counts[i - 1, j], costs[i - 1, j], 1
            candidate = (counts[i, j - 1], costs[i, j - 1])
            if candidate[0] > count or (candidate[0] == count and candidate[1] < cost):
                count, cost, action = *candidate, 2
            delay = prediction[j - 1]["time_s"] - target[i - 1]["time_s"]
            if abs(delay) <= tolerance:
                candidate = (counts[i - 1, j - 1] + 1, costs[i - 1, j - 1] + abs(delay))
                if candidate[0] > count or (candidate[0] == count and candidate[1] <= cost):
                    count, cost, action = *candidate, 3
            counts[i, j], costs[i, j], actions[i, j] = count, cost, action
    matched = []
    i, j = n, m
    while i > 0 and j > 0:
        action = actions[i, j]
        if action == 1:
            i -= 1
        elif action == 2:
            j -= 1
        else:
            i, j = i - 1, j - 1
            yt, pt = target[i], prediction[j]
            matched.append({
                "target_event_index": i, "prediction_event_index": j,
                "target_time_s": yt["time_s"], "prediction_time_s": pt["time_s"],
                "timing_error_s": pt["time_s"] - yt["time_s"],
                "width_absolute_error_s": (
                    abs(pt["width_s"] - yt["width_s"])
                    if pt["width_s"] is not None and yt["width_s"] is not None else None
                ),
            })
    return list(reversed(matched))


def _event_summary(segments, kind):
    target_count = sum(len(segment["events"][kind]["target"]) for segment in segments)
    prediction_count = sum(len(segment["events"][kind]["prediction"]) for segment in segments)
    matches = [item for segment in segments for item in segment["events"][kind]["matches"]]
    matched = len(matches)
    timing = [item["timing_error_s"] for item in matches]
    widths = [item["width_absolute_error_s"] for item in matches if item["width_absolute_error_s"] is not None]
    return {
        "target_count": target_count, "prediction_count": prediction_count,
        "matched_count": matched,
        "precision": matched / prediction_count if prediction_count else None,
        "recall": matched / target_count if target_count else None,
        "f1": 2 * matched / (target_count + prediction_count) if target_count + prediction_count else None,
        "timing_signed_median_s": float(np.median(timing)) if timing else None,
        "timing_absolute_median_s": float(np.median(np.abs(timing))) if timing else None,
        "timing_absolute_p90_s": float(np.percentile(np.abs(timing), 90)) if timing else None,
        "width_absolute_median_s": float(np.median(widths)) if widths else None,
        "width_absolute_p90_s": float(np.percentile(widths, 90)) if widths else None,
        "width_valid_match_count": len(widths),
    }


def _component(y, p, mask, time, gaps, dt, scale, settings, name):
    pieces = _segments(mask, gaps)
    requested = int(np.count_nonzero(mask))
    bad_y = int(np.count_nonzero(~np.isfinite(y[mask])))
    bad_p = int(np.count_nonzero(~np.isfinite(p[mask])))
    status = "failed_nonfinite" if bad_y or bad_p else ("ok" if requested else "no_support")
    output = {
        "component": name, "status": status, "state": None,
        "requested_points": requested, "scored_points": requested if status == "ok" else 0,
        "segment_count": len(pieces), "target_nonfinite_points": bad_y,
        "prediction_nonfinite_points": bad_p, "global_pearson_r": None,
        "within_segment_r": None, "within_segment_amplitude_ratio": None,
        "target_dynamic_sd": None, "prediction_dynamic_sd": None,
        "training_sd": None if scale is None else float(scale),
        "bias_training_sd": None, "nrmse_training_sd": None,
        "collapsed": None, "spurious": None,
        "local_correlation": _local_summary([]),
        "slope_direction": [_slope_summary([], width) for width in settings.slope_windows_s],
        "events": None, "segments": [],
    }
    if status != "ok":
        output["segments"] = [{**_bounds(piece, time, sid), "status": status} for sid, piece in enumerate(pieces)]
        return output
    all_yc, all_pc, local = [], [], []
    slopes = {width: [] for width in settings.slope_windows_s}
    dy_flags, dp_flags = [], []
    for sid, piece in enumerate(pieces):
        yc, pc = _center(y[piece]), _center(p[piece])
        dy, dp = _dynamic(yc, y[piece], settings), _dynamic(pc, p[piece], settings)
        dy_flags.append(dy)
        dp_flags.append(dp)
        all_yc.append(yc)
        all_pc.append(pc)
        segment = {
            **_bounds(piece, time, sid), "status": "ok",
            "target_dynamic": dy, "prediction_dynamic": dp,
            "target_dynamic_sd": _sd(yc), "prediction_dynamic_sd": _sd(pc),
            "collapsed": dy and not dp, "spurious": not dy and dp,
            "local_correlation_windows": _local_correlation(y, p, piece, time, dt, settings, sid),
            "slope_direction": [], "events": {},
        }
        local.extend(segment["local_correlation_windows"])
        for width in settings.slope_windows_s:
            details = _slope_details(y, p, piece, time, dt, settings, sid, width)
            slopes[width].extend(details)
            segment["slope_direction"].append(_slope_summary(details, width))
        for kind in ("peaks", "troughs"):
            ye, pe = _events(y[piece], time[piece], settings, dt, kind), _events(p[piece], time[piece], settings, dt, kind)
            segment["events"][kind] = {
                "target": ye, "prediction": pe,
                "matches": _match_events(ye, pe, settings.event_match_tolerance_s),
            }
        output["segments"].append(segment)
    yc, pc = np.concatenate(all_yc), np.concatenate(all_pc)
    ysd, psd = _sd(yc), _sd(pc)
    collapsed = any(segment["collapsed"] for segment in output["segments"])
    spurious = any(segment["spurious"] for segment in output["segments"])
    if all(dy_flags) and all(dp_flags):
        state = "dynamic"
    elif not any(dy_flags) and not any(dp_flags):
        state = "both_low_dynamic"
    elif collapsed and not spurious:
        state = "collapsed"
    elif spurious and not collapsed:
        state = "spurious"
    else:
        state = "mixed_dynamic_support"
    output.update({
        "state": state, "target_dynamic_sd": ysd, "prediction_dynamic_sd": psd,
        "collapsed": collapsed, "spurious": spurious,
        "within_segment_amplitude_ratio": psd / ysd if any(dy_flags) else None,
        "local_correlation": _local_summary(local),
        "slope_direction": [_slope_summary(slopes[width], width) for width in settings.slope_windows_s],
        "events": {kind: _event_summary(output["segments"], kind) for kind in ("peaks", "troughs")},
    })
    if any(dy_flags) and any(dp_flags):
        output["within_segment_r"] = _correlation(yc, pc, settings)[0]
    if len(pieces) == 1:
        output["global_pearson_r"] = _correlation(y[mask], p[mask], settings)[0]
    if scale is not None:
        residual = (p[mask] - y[mask]) / scale
        output["bias_training_sd"] = float(np.mean(residual))
        output["nrmse_training_sd"] = _sd(residual)
    return output


def _pair_slope_summary(details, width):
    informative = [entry for entry in details if entry["unrecovered"] is not None]
    rate = None
    if informative:
        o = np.asarray([abs(entry["target_slopes_per_s"][0]) for entry in informative])
        r = np.asarray([abs(entry["target_slopes_per_s"][1]) for entry in informative])
        weights = (o / np.max(o)) * (r / np.max(r))
        failed = np.asarray([entry["unrecovered"] for entry in informative], dtype=bool)
        rate = float(weights[failed].sum() / weights.sum())
    return {
        "window_s": width, "target_slope_product_weighted_unrecovered_rate": rate,
        "informative_window_count": len(informative), "total_window_count": len(details),
        "na_reason": "no_target_joint_variation" if rate is None else None, "windows": details,
    }


def _pair(target, prediction, mask, time, gaps, dt, settings):
    joint = mask[:, 0] & mask[:, 1]
    pieces = _segments(joint, gaps)
    failed = bool(np.any(~np.isfinite(target[joint])) or np.any(~np.isfinite(prediction[joint])))
    requested = int(np.count_nonzero(joint))
    status = "failed_nonfinite" if failed else ("ok" if requested else "no_support")
    output = {
        "status": status, "requested_points": requested,
        "scored_points": requested if status == "ok" else 0, "segment_count": len(pieces),
        "target_within_segment_hbo_hbr_r": None,
        "prediction_within_segment_hbo_hbr_r": None,
        "correlation_absolute_difference": None,
        "local_correlation_difference": {
            "signed_median": None, "absolute_median": None, "absolute_p90": None,
            "valid_window_count": 0, "total_window_count": 0, "windows": [],
        },
        "slope_relationship": [_pair_slope_summary([], width) for width in settings.slope_windows_s],
        "segments": [],
    }
    if status != "ok":
        output["segments"] = [
            {**_bounds(piece, time, sid), "status": status}
            for sid, piece in enumerate(pieces)
        ]
        return output
    target_centered, prediction_centered, local = [], [], []
    target_dynamic = [False, False]
    prediction_dynamic = [False, False]
    slopes = {width: [] for width in settings.slope_windows_s}
    for sid, piece in enumerate(pieces):
        yc = np.column_stack([_center(target[piece, column]) for column in range(2)])
        pc = np.column_stack([_center(prediction[piece, column]) for column in range(2)])
        target_centered.append(yc)
        prediction_centered.append(pc)
        for column in range(2):
            target_dynamic[column] |= _dynamic(yc[:, column], target[piece, column], settings)
            prediction_dynamic[column] |= _dynamic(pc[:, column], prediction[piece, column], settings)
        segment = {**_bounds(piece, time, sid), "local_correlation_windows": [], "slope_relationship": []}
        for chosen, start, end in _windows(piece, time, settings.local_window_s, settings.local_step_s, dt, 3):
            yr, yn = _correlation(
                *[_detrend(target[chosen, column], time[chosen]) for column in range(2)],
                settings, reference_y=target[chosen, 0], reference_p=target[chosen, 1],
            )
            pr, pn = _correlation(
                *[_detrend(prediction[chosen, column], time[chosen]) for column in range(2)],
                settings, reference_y=prediction[chosen, 0], reference_p=prediction[chosen, 1],
            )
            entry = {
                "segment_id": sid, "start_s": start, "end_s_exclusive": end,
                "point_count": len(chosen), "target_r": yr, "prediction_r": pr,
                "difference": pr - yr if yr is not None and pr is not None else None,
                "target_na_reason": yn, "prediction_na_reason": pn,
            }
            local.append(entry)
            segment["local_correlation_windows"].append(entry)
        for width in settings.slope_windows_s:
            details = []
            for chosen, start, end in _windows(piece, time, width, settings.slope_step_s, dt, 2):
                sy = [_slope(target[chosen, column], time[chosen], settings) for column in range(2)]
                sp = [_slope(prediction[chosen, column], time[chosen], settings) for column in range(2)]
                informative = all(value != 0 for value in sy)
                # Compare signs without multiplying physical slopes; this
                # remains meaningful for very small dataset-native units.
                signs_y = np.sign(sy[0]) * np.sign(sy[1])
                signs_p = np.sign(sp[0]) * np.sign(sp[1])
                details.append({
                    "segment_id": sid, "start_s": start, "end_s_exclusive": end,
                    "point_count": len(chosen), "target_slopes_per_s": sy,
                    "prediction_slopes_per_s": sp,
                    "unrecovered": bool(signs_y * signs_p <= 0) if informative else None,
                })
            slopes[width].extend(details)
            segment["slope_relationship"].append(_pair_slope_summary(details, width))
        output["segments"].append(segment)
    yc, pc = np.concatenate(target_centered), np.concatenate(prediction_centered)
    yr = _correlation(yc[:, 0], yc[:, 1], settings)[0] if all(target_dynamic) else None
    pr = _correlation(pc[:, 0], pc[:, 1], settings)[0] if all(prediction_dynamic) else None
    differences = [entry["difference"] for entry in local if entry["difference"] is not None]
    output.update({
        "target_within_segment_hbo_hbr_r": yr,
        "prediction_within_segment_hbo_hbr_r": pr,
        "correlation_absolute_difference": abs(pr - yr) if yr is not None and pr is not None else None,
        "local_correlation_difference": {
            "signed_median": float(np.median(differences)) if differences else None,
            "absolute_median": float(np.median(np.abs(differences))) if differences else None,
            "absolute_p90": float(np.percentile(np.abs(differences), 90)) if differences else None,
            "valid_window_count": len(differences), "total_window_count": len(local), "windows": local,
        },
        "slope_relationship": [_pair_slope_summary(slopes[width], width) for width in settings.slope_windows_s],
    })
    return output


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def fnirs_morphology_metrics(
    target, prediction, *, time_s, score_mask=None, training_sd=None,
    settings: MorphologySettings | None = None,
) -> dict[str, Any]:
    """Score ``[T, 2]`` HbO/HbR arrays without alignment or refiltering.

    Local windows have half-open time support and are retained only when the
    whole requested duration is inside one segment (including its final sample
    interval). Slopes are ordinary least squares in native units per second.
    ``within_segment_r`` pools separately centered segment values; the dynamic
    SD and amplitude ratio use those same values, without per-segment z scores.
    Peak/trough detection alone uses each curve's segment SD. Widths are at
    half the observed prominence; endpoint-truncated flanks have no width.

    Bias is ``mean(prediction-target)/training_sd``; NRMSE uses the RMS of that
    same scaled residual. Without an explicitly supplied training SD both are
    undefined. Errors of shape/schema/timestamps raise ``ValueError``; numeric
    failures on requested signal support are returned as component statuses.
    """
    target, prediction, time, mask, scales, settings, gaps, dt = _validate_inputs(
        target, prediction, time_s, score_mask, training_sd, settings,
    )
    components = [
        _component(target[:, column], prediction[:, column], mask[:, column], time,
                   gaps, dt, None if scales is None else scales[column], settings, name)
        for column, name in enumerate(("HbO", "HbR"))
    ]
    return _json_safe({
        "schema": "fnirs_morphology_v1", "settings": asdict(settings),
        "components": components,
        "pair": _pair(target, prediction, mask, time, gaps, dt, settings),
    })


__all__ = ["MorphologySettings", "fnirs_morphology_metrics"]
