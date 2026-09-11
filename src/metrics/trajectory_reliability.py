"""Mask-aware reliability metrics for reconstructed observation trajectories.

Reconstruction deviation and posterior uncertainty are deliberately reported as
separate quantities.  A large residual is not itself a posterior standard
deviation, and a narrow posterior does not imply an accurate reconstruction.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_matching_arrays(
    observed: np.ndarray,
    reconstructed: np.ndarray,
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed_array = np.asarray(observed, dtype=np.float64)
    reconstructed_array = np.asarray(reconstructed, dtype=np.float64)
    if observed_array.shape != reconstructed_array.shape:
        raise ValueError("observed and reconstructed trajectories must match")
    if observed_array.size == 0:
        raise ValueError("trajectory reliability requires at least one point")

    if valid_mask is None:
        mask = np.ones(observed_array.shape, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != observed_array.shape:
            raise ValueError("valid_mask must match the trajectory shape")
    mask = mask & np.isfinite(observed_array) & np.isfinite(reconstructed_array)
    if not np.any(mask):
        raise ValueError("trajectory reliability has no finite valid points")
    return observed_array, reconstructed_array, mask


def trajectory_reliability_metrics(
    observed: np.ndarray,
    reconstructed: np.ndarray,
    *,
    predictive_std: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    observed_std_floor: float = 1e-8,
    predictive_std_floor: float = 1e-8,
    coverage_z: float = 1.959963984540054,
) -> dict[str, Any]:
    """Return deviation, spread, and predictive-calibration diagnostics.

    ``trajectory_deviation_nrmse`` is undefined when the observed trajectory
    has negligible variance.  Predictive diagnostics use the subset of valid
    points with finite, strictly positive predictive standard deviation; they
    remain undefined when no such points exist.
    """

    if observed_std_floor < 0.0 or predictive_std_floor < 0.0:
        raise ValueError("standard-deviation floors must be non-negative")
    if not np.isfinite(coverage_z) or coverage_z <= 0.0:
        raise ValueError("coverage_z must be finite and positive")

    observed_array, reconstructed_array, mask = _as_matching_arrays(
        observed, reconstructed, valid_mask
    )
    observed_valid = observed_array[mask]
    reconstructed_valid = reconstructed_array[mask]
    residual = observed_valid - reconstructed_valid
    observed_sd = float(np.std(observed_valid, ddof=0))
    reconstructed_sd = float(np.std(reconstructed_valid, ddof=0))
    rmse = float(np.sqrt(np.mean(residual**2)))
    low_observed_variance = bool(
        not np.isfinite(observed_sd) or observed_sd <= float(observed_std_floor)
    )
    deviation = float("nan") if low_observed_variance else rmse / observed_sd
    sd_ratio = (
        float("nan")
        if low_observed_variance
        else reconstructed_sd / observed_sd
    )

    output: dict[str, Any] = {
        "trajectory_deviation_nrmse": float(deviation),
        "observed_temporal_sd": observed_sd,
        "reconstructed_temporal_sd": reconstructed_sd,
        "temporal_sd_ratio": float(sd_ratio),
        "valid_point_count": int(np.count_nonzero(mask)),
        "low_observed_variance": low_observed_variance,
        "posterior_predictive_sd_mean": float("nan"),
        "posterior_predictive_sd_median": float("nan"),
        "standardized_residual_rms": float("nan"),
        "predictive_95_coverage": float("nan"),
        "predictive_valid_point_count": 0,
    }

    if predictive_std is None:
        return output
    predictive = np.asarray(predictive_std, dtype=np.float64)
    if predictive.shape != observed_array.shape:
        raise ValueError("predictive_std must match the trajectory shape")
    predictive_mask = (
        mask
        & np.isfinite(predictive)
        & (predictive > float(predictive_std_floor))
    )
    if not np.any(predictive_mask):
        return output

    predictive_valid = predictive[predictive_mask]
    predictive_residual = (
        observed_array[predictive_mask] - reconstructed_array[predictive_mask]
    )
    standardized = predictive_residual / predictive_valid
    output.update(
        {
            "posterior_predictive_sd_mean": float(np.mean(predictive_valid)),
            "posterior_predictive_sd_median": float(np.median(predictive_valid)),
            "standardized_residual_rms": float(
                np.sqrt(np.mean(standardized**2))
            ),
            "predictive_95_coverage": float(
                np.mean(np.abs(predictive_residual) <= coverage_z * predictive_valid)
            ),
            "predictive_valid_point_count": int(np.count_nonzero(predictive_mask)),
        }
    )
    return output


__all__ = ["trajectory_reliability_metrics"]


_LEGACY_RESIDUAL_FIELDS = {
    'innovations': 'predictive_residuals',
    'standardized_innovations': 'standardized_predictive_residuals',
    'segment_standardized_innovation_mean': 'segment_standardized_predictive_residual_mean',
    'innovation_structure': 'predictive_residual_structure',
    'process_innovation_rms': 'state_transition_residual_rms',
    'standardized_process_innovation_rms': 'standardized_state_transition_residual_rms',
    'process_innovation_status': 'state_transition_residual_status',
}


def canonical_residual_fields(value):
    """Read retained diagnostic fields into the current schema in memory."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = _LEGACY_RESIDUAL_FIELDS.get(key, key)
            if name in result:
                raise ValueError(f'duplicate residual field: {name}')
            result[name] = canonical_residual_fields(item)
        return result
    if isinstance(value, list):
        return [canonical_residual_fields(item) for item in value]
    return value
