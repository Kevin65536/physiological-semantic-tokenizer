"""Leakage-safe train-to-validation information ledger for token representations.

The ledger asks how much held-out physiological variation is linearly
recoverable from each representation.  Ridge regularization is selected using
``GroupKFold`` over *training subjects only*.  The selected regularization is
then frozen, the probe is refit on all eligible training subjects, and scores
are computed once on the validation set.

Missing and constant targets are reported as unavailable rather than being
assigned a misleading score.  The public result is composed of JSON-compatible
objects so skipped analyses remain explicit in persisted manifests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


TOKEN_INFORMATION_LEDGER_SCHEMA_VERSION = "token_information_ledger_v1"


@dataclass(frozen=True)
class InformationLedgerConfig:
    """Configuration for :func:`evaluate_information_ledger`."""

    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    max_group_folds: int = 5
    bootstrap_iterations: int = 1000
    bootstrap_confidence: float = 0.95
    seed: int = 0
    variance_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not self.alphas or any(
            not np.isfinite(alpha) or alpha < 0.0 for alpha in self.alphas
        ):
            raise ValueError("alphas must be finite, non-empty, and non-negative")
        if self.max_group_folds < 2:
            raise ValueError("max_group_folds must be at least two")
        if self.bootstrap_iterations < 0:
            raise ValueError("bootstrap_iterations must be non-negative")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise ValueError(
                "bootstrap_confidence must lie strictly between zero and one"
            )
        if self.variance_tolerance < 0.0:
            raise ValueError("variance_tolerance must be non-negative")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _skipped_ledger(
    reason: str,
    *,
    config: InformationLedgerConfig,
    representations: Mapping[str, Any] | None = None,
    **diagnostics: Any,
) -> dict[str, Any]:
    return {
        "schema_version": TOKEN_INFORMATION_LEDGER_SCHEMA_VERSION,
        "status": "skipped",
        "skipped_reason": reason,
        "config": _json_value(asdict(config)),
        "representations": _json_value(representations or {}),
        **_json_value(diagnostics),
    }


def _safe_r2(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    variance_tolerance: float,
) -> float | None:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    finite = np.isfinite(target) & np.isfinite(prediction)
    if int(np.sum(finite)) < 2:
        return None
    target = target[finite]
    prediction = prediction[finite]
    centered_sum = float(np.sum(np.square(target - np.mean(target))))
    if centered_sum <= variance_tolerance:
        return None
    value = 1.0 - float(np.sum(np.square(target - prediction))) / centered_sum
    return value if np.isfinite(value) else None


def _predictive_cv_r2(
    train_target: np.ndarray,
    validation_target: np.ndarray,
    prediction: np.ndarray,
    *,
    variance_tolerance: float,
) -> float | None:
    """Out-of-group R2 using the fold-training mean as the null prediction."""

    finite_train = np.isfinite(train_target)
    finite_validation = np.isfinite(validation_target) & np.isfinite(prediction)
    if not np.any(finite_train) or int(np.sum(finite_validation)) < 1:
        return None
    baseline = float(np.mean(train_target[finite_train]))
    denominator = float(
        np.sum(np.square(validation_target[finite_validation] - baseline))
    )
    if denominator <= variance_tolerance:
        return None
    numerator = float(
        np.sum(
            np.square(
                validation_target[finite_validation] - prediction[finite_validation]
            )
        )
    )
    value = 1.0 - numerator / denominator
    return value if np.isfinite(value) else None


def _standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return mean, scale


def _mean_finite(values: Sequence[float | None]) -> float | None:
    finite = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    return float(np.mean(finite)) if len(finite) else None


def _coordinate_train_status(
    target: np.ndarray,
    representation_valid: np.ndarray,
    subjects: np.ndarray,
    *,
    variance_tolerance: float,
) -> tuple[np.ndarray, str | None]:
    valid = representation_valid & np.isfinite(target)
    if int(np.sum(valid)) < 2:
        return valid, "insufficient_finite_train_samples"
    if len(np.unique(subjects[valid])) < 2:
        return valid, "insufficient_train_subjects"
    centered_sum = float(np.sum(np.square(target[valid] - np.mean(target[valid]))))
    if centered_sum <= variance_tolerance:
        return valid, "constant_train_target"
    return valid, None


def _select_alpha_grouped(
    features: np.ndarray,
    target: np.ndarray,
    subjects: np.ndarray,
    representation_valid: np.ndarray,
    *,
    config: InformationLedgerConfig,
) -> tuple[float | None, dict[str, Any]]:
    """Select one alpha across coordinates using training-subject folds only."""

    alpha_values = tuple(float(alpha) for alpha in config.alphas)
    scores: dict[float, list[float]] = {alpha: [] for alpha in alpha_values}
    coordinate_fold_counts = np.zeros(target.shape[1], dtype=np.int64)
    coordinate_train_status: list[dict[str, Any]] = []
    coordinate_valid_masks: list[np.ndarray] = []
    eligible_coordinates: list[int] = []
    maximum_folds = 0
    ridge_fit_count = 0

    for coordinate in range(target.shape[1]):
        valid, reason = _coordinate_train_status(
            target[:, coordinate],
            representation_valid,
            subjects,
            variance_tolerance=config.variance_tolerance,
        )
        coordinate_valid_masks.append(valid)
        coordinate_train_status.append(
            {
                "coordinate_index": coordinate,
                "status": "ok" if reason is None else "skipped",
                "skipped_reason": reason,
                "finite_train_samples": int(np.sum(valid)),
                "train_subject_count": int(len(np.unique(subjects[valid]))),
            }
        )
        if reason is None:
            eligible_coordinates.append(coordinate)

    common_mask = bool(eligible_coordinates) and all(
        np.array_equal(
            coordinate_valid_masks[coordinate],
            coordinate_valid_masks[eligible_coordinates[0]],
        )
        for coordinate in eligible_coordinates[1:]
    )
    fit_strategy = (
        "multi_output_common_mask" if common_mask else "coordinatewise_missing_mask"
    )

    if common_mask:
        valid = coordinate_valid_masks[eligible_coordinates[0]]
        coordinate_groups = subjects[valid]
        fold_count = min(config.max_group_folds, len(np.unique(coordinate_groups)))
        splitter = GroupKFold(n_splits=fold_count)
        x = features[valid]
        y = target[valid][:, eligible_coordinates]
        for train_index, validation_index in splitter.split(
            x, y, groups=coordinate_groups
        ):
            x_train = x[train_index]
            y_train = y[train_index]
            x_validation = x[validation_index]
            y_validation = y[validation_index]
            if len(x_train) < 2:
                continue
            mean, scale = _standardizer(x_train)
            standardized_train = (x_train - mean) / scale
            standardized_validation = (x_validation - mean) / scale
            fold_had_score = np.zeros(len(eligible_coordinates), dtype=bool)
            for alpha in alpha_values:
                try:
                    ridge_fit_count += 1
                    model = Ridge(alpha=alpha).fit(standardized_train, y_train)
                    prediction = np.asarray(
                        model.predict(standardized_validation), dtype=np.float64
                    ).reshape(len(x_validation), len(eligible_coordinates))
                except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                    continue
                for local_coordinate in range(len(eligible_coordinates)):
                    score = _predictive_cv_r2(
                        y_train[:, local_coordinate],
                        y_validation[:, local_coordinate],
                        prediction[:, local_coordinate],
                        variance_tolerance=config.variance_tolerance,
                    )
                    if score is not None:
                        scores[alpha].append(score)
                        fold_had_score[local_coordinate] = True
            for local_coordinate, had_score in enumerate(fold_had_score):
                if had_score:
                    coordinate_fold_counts[
                        eligible_coordinates[local_coordinate]
                    ] += 1
        maximum_folds = max(maximum_folds, fold_count)
    else:
        for coordinate in eligible_coordinates:
            valid = coordinate_valid_masks[coordinate]
            coordinate_groups = subjects[valid]
            group_count = len(np.unique(coordinate_groups))
            fold_count = min(config.max_group_folds, group_count)
            splitter = GroupKFold(n_splits=fold_count)
            x = features[valid]
            y = target[valid, coordinate]

            for train_index, validation_index in splitter.split(
                x, y, groups=coordinate_groups
            ):
                x_train = x[train_index]
                y_train = y[train_index]
                x_validation = x[validation_index]
                y_validation = y[validation_index]
                if len(x_train) < 2:
                    continue
                mean, scale = _standardizer(x_train)
                standardized_train = (x_train - mean) / scale
                standardized_validation = (x_validation - mean) / scale
                fold_had_score = False
                for alpha in alpha_values:
                    try:
                        ridge_fit_count += 1
                        model = Ridge(alpha=alpha).fit(
                            standardized_train, y_train
                        )
                        prediction = np.asarray(
                            model.predict(standardized_validation),
                            dtype=np.float64,
                        )
                    except (
                        FloatingPointError,
                        ValueError,
                        np.linalg.LinAlgError,
                    ):
                        continue
                    score = _predictive_cv_r2(
                        y_train,
                        y_validation,
                        prediction,
                        variance_tolerance=config.variance_tolerance,
                    )
                    if score is not None:
                        scores[alpha].append(score)
                        fold_had_score = True
                if fold_had_score:
                    coordinate_fold_counts[coordinate] += 1
            maximum_folds = max(maximum_folds, fold_count)

    means = {
        alpha: (float(np.mean(values)) if values else None)
        for alpha, values in scores.items()
    }
    eligible = [alpha for alpha in alpha_values if means[alpha] is not None]
    if not eligible:
        any_train_coordinate = any(
            row["status"] == "ok" for row in coordinate_train_status
        )
        return None, {
            "status": "skipped",
            "skipped_reason": (
                "group_cv_failed_no_finite_scores"
                if any_train_coordinate
                else "no_evaluable_train_target_coordinates"
            ),
            "selection_data": "training_only",
            "splitter": "GroupKFold_by_subject",
            "fit_strategy": fit_strategy,
            "mean_predictive_r2_by_alpha": {
                str(alpha): means[alpha] for alpha in alpha_values
            },
            "fold_score_count_by_alpha": {
                str(alpha): len(scores[alpha]) for alpha in alpha_values
            },
            "coordinate_scored_fold_counts": coordinate_fold_counts.tolist(),
            "coordinate_train_status": coordinate_train_status,
            "maximum_fold_count": maximum_folds,
            "ridge_fit_count": ridge_fit_count,
        }

    selected = max(eligible, key=lambda alpha: (float(means[alpha]), -alpha))
    return selected, {
        "status": "ok",
        "skipped_reason": None,
        "selected_alpha": selected,
        "selection_data": "training_only",
        "splitter": "GroupKFold_by_subject",
        "fit_strategy": fit_strategy,
        "score_definition": "predictive_R2_against_fold_training_target_mean",
        "mean_predictive_r2_by_alpha": {
            str(alpha): means[alpha] for alpha in alpha_values
        },
        "fold_score_count_by_alpha": {
            str(alpha): len(scores[alpha]) for alpha in alpha_values
        },
        "coordinate_scored_fold_counts": coordinate_fold_counts.tolist(),
        "coordinate_train_status": coordinate_train_status,
        "maximum_fold_count": maximum_folds,
        "ridge_fit_count": ridge_fit_count,
    }


def _subject_scores(
    target: np.ndarray,
    prediction: np.ndarray,
    subjects: np.ndarray,
    *,
    variance_tolerance: float,
) -> dict[str, list[float | None]]:
    return {
        str(subject): [
            _safe_r2(
                target[subjects == subject, coordinate],
                prediction[subjects == subject, coordinate],
                variance_tolerance=variance_tolerance,
            )
            for coordinate in range(target.shape[1])
        ]
        for subject in np.unique(subjects)
    }


def _nanmean_columns(matrix: np.ndarray) -> np.ndarray:
    finite = np.isfinite(matrix)
    counts = np.sum(finite, axis=0)
    sums = np.sum(np.where(finite, matrix, 0.0), axis=0)
    return np.divide(
        sums,
        counts,
        out=np.full(matrix.shape[1], np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _subject_bootstrap(
    subject_scores: Mapping[str, Sequence[float | None]],
    *,
    config: InformationLedgerConfig,
    seed: int,
) -> dict[str, Any]:
    if config.bootstrap_iterations == 0:
        return {
            "status": "skipped",
            "skipped_reason": "bootstrap_disabled",
            "iterations": 0,
        }
    subject_names = list(subject_scores)
    if not subject_names:
        return {
            "status": "skipped",
            "skipped_reason": "no_validation_subjects",
            "iterations": config.bootstrap_iterations,
        }
    matrix = np.asarray(
        [
            [np.nan if value is None else float(value) for value in subject_scores[name]]
            for name in subject_names
        ],
        dtype=np.float64,
    )
    if not np.any(np.isfinite(matrix)):
        return {
            "status": "skipped",
            "skipped_reason": "no_defined_subject_level_r2",
            "iterations": config.bootstrap_iterations,
            "subject_count": len(subject_names),
        }

    rng = np.random.default_rng(seed)
    coordinate_draws = np.empty(
        (config.bootstrap_iterations, matrix.shape[1]), dtype=np.float64
    )
    overall_draws = np.full(config.bootstrap_iterations, np.nan, dtype=np.float64)
    for iteration in range(config.bootstrap_iterations):
        selected = rng.integers(0, len(matrix), size=len(matrix))
        coordinate_draws[iteration] = _nanmean_columns(matrix[selected])
        finite = coordinate_draws[iteration][
            np.isfinite(coordinate_draws[iteration])
        ]
        if len(finite):
            overall_draws[iteration] = float(np.mean(finite))

    tail = (1.0 - config.bootstrap_confidence) / 2.0
    coordinate_low = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    coordinate_high = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    for coordinate in range(matrix.shape[1]):
        finite_draws = coordinate_draws[
            np.isfinite(coordinate_draws[:, coordinate]), coordinate
        ]
        if len(finite_draws):
            coordinate_low[coordinate] = np.quantile(finite_draws, tail)
            coordinate_high[coordinate] = np.quantile(
                finite_draws, 1.0 - tail
            )
    finite_overall = overall_draws[np.isfinite(overall_draws)]
    return _json_value(
        {
            "status": "ok",
            "skipped_reason": None,
            "iterations": config.bootstrap_iterations,
            "confidence": config.bootstrap_confidence,
            "resampling_unit": "validation_subject",
            "subject_count": len(subject_names),
            "coordinate_subject_mean_r2": _nanmean_columns(matrix),
            "coordinate_ci_low": coordinate_low,
            "coordinate_ci_high": coordinate_high,
            "overall_subject_mean_r2": float(
                np.mean(_nanmean_columns(matrix)[np.isfinite(_nanmean_columns(matrix))])
            ),
            "overall_ci_low": (
                float(np.quantile(finite_overall, tail))
                if len(finite_overall)
                else np.nan
            ),
            "overall_ci_high": (
                float(np.quantile(finite_overall, 1.0 - tail))
                if len(finite_overall)
                else np.nan
            ),
        }
    )


def _evaluate_representation(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_target: np.ndarray,
    validation_target: np.ndarray,
    train_subjects: np.ndarray,
    validation_subjects: np.ndarray,
    coordinate_names: Sequence[str],
    *,
    config: InformationLedgerConfig,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if train_features.ndim != 2 or train_features.shape[0] != len(train_target):
        return {
            "status": "skipped",
            "skipped_reason": "invalid_train_representation_shape",
            "observed_shape": list(train_features.shape),
            "expected_sample_count": len(train_target),
        }
    if (
        validation_features.ndim != 2
        or validation_features.shape[0] != len(validation_target)
    ):
        return {
            "status": "skipped",
            "skipped_reason": "invalid_validation_representation_shape",
            "observed_shape": list(validation_features.shape),
            "expected_sample_count": len(validation_target),
        }
    if train_features.shape[1] != validation_features.shape[1]:
        return {
            "status": "skipped",
            "skipped_reason": "train_validation_representation_dimension_mismatch",
            "train_dimension": train_features.shape[1],
            "validation_dimension": validation_features.shape[1],
        }
    if train_features.shape[1] == 0:
        return {
            "status": "skipped",
            "skipped_reason": "empty_representation_dimension",
        }

    train_valid = np.all(np.isfinite(train_features), axis=1)
    validation_valid = np.all(np.isfinite(validation_features), axis=1)
    if int(np.sum(train_valid)) < 2:
        return {
            "status": "skipped",
            "skipped_reason": "insufficient_finite_train_representation_rows",
            "finite_train_rows": int(np.sum(train_valid)),
        }
    if len(np.unique(train_subjects[train_valid])) < 2:
        return {
            "status": "skipped",
            "skipped_reason": "fewer_than_two_train_subjects",
            "train_subject_count": int(len(np.unique(train_subjects[train_valid]))),
        }
    if int(np.sum(validation_valid)) < 2:
        return {
            "status": "skipped",
            "skipped_reason": "insufficient_finite_validation_representation_rows",
            "finite_validation_rows": int(np.sum(validation_valid)),
        }

    selected_alpha, selection = _select_alpha_grouped(
        train_features,
        train_target,
        train_subjects,
        train_valid,
        config=config,
    )
    if selected_alpha is None:
        return {
            "status": "skipped",
            "skipped_reason": selection["skipped_reason"],
            "probe_selection": selection,
            "finite_train_rows": int(np.sum(train_valid)),
            "finite_validation_rows": int(np.sum(validation_valid)),
        }

    coordinate_train_masks: list[np.ndarray] = []
    coordinate_train_reasons: list[str | None] = []
    for coordinate in range(train_target.shape[1]):
        coordinate_train_valid, train_reason = _coordinate_train_status(
            train_target[:, coordinate],
            train_valid,
            train_subjects,
            variance_tolerance=config.variance_tolerance,
        )
        coordinate_train_masks.append(coordinate_train_valid)
        coordinate_train_reasons.append(train_reason)

    prediction = np.full(validation_target.shape, np.nan, dtype=np.float64)
    fit_errors: dict[int, str] = {}
    fast_fit = selection["fit_strategy"] == "multi_output_common_mask"
    eligible_coordinates = [
        coordinate
        for coordinate, reason in enumerate(coordinate_train_reasons)
        if reason is None
    ]
    if fast_fit and eligible_coordinates:
        common_train_valid = coordinate_train_masks[eligible_coordinates[0]]
        x_train = train_features[common_train_valid]
        y_train = train_target[common_train_valid][:, eligible_coordinates]
        mean, scale = _standardizer(x_train)
        try:
            model = Ridge(alpha=selected_alpha).fit(
                (x_train - mean) / scale, y_train
            )
            fast_prediction = np.asarray(
                model.predict(
                    (validation_features[validation_valid] - mean) / scale
                ),
                dtype=np.float64,
            ).reshape(int(np.sum(validation_valid)), len(eligible_coordinates))
            prediction[
                np.ix_(
                    np.flatnonzero(validation_valid),
                    np.asarray(eligible_coordinates, dtype=np.int64),
                )
            ] = fast_prediction
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
            for coordinate in eligible_coordinates:
                fit_errors[coordinate] = type(error).__name__

    coordinate_status: list[dict[str, Any]] = []
    for coordinate, coordinate_name in enumerate(coordinate_names):
        coordinate_train_valid = coordinate_train_masks[coordinate]
        train_reason = coordinate_train_reasons[coordinate]
        if train_reason is not None:
            coordinate_status.append(
                {
                    "coordinate_index": coordinate,
                    "coordinate_name": coordinate_name,
                    "status": "skipped",
                    "skipped_reason": train_reason,
                    "finite_train_samples": int(np.sum(coordinate_train_valid)),
                }
            )
            continue

        if not fast_fit:
            x_train = train_features[coordinate_train_valid]
            y_train = train_target[coordinate_train_valid, coordinate]
            mean, scale = _standardizer(x_train)
            try:
                model = Ridge(alpha=selected_alpha).fit(
                    (x_train - mean) / scale, y_train
                )
                prediction[validation_valid, coordinate] = model.predict(
                    (validation_features[validation_valid] - mean) / scale
                )
            except (
                FloatingPointError,
                ValueError,
                np.linalg.LinAlgError,
            ) as error:
                fit_errors[coordinate] = type(error).__name__

        if coordinate in fit_errors:
            coordinate_status.append(
                {
                    "coordinate_index": coordinate,
                    "coordinate_name": coordinate_name,
                    "status": "skipped",
                    "skipped_reason": "ridge_fit_failed",
                    "error_type": fit_errors[coordinate],
                }
            )
            continue

        validation_coordinate_valid = validation_valid & np.isfinite(
            validation_target[:, coordinate]
        )
        score = _safe_r2(
            validation_target[validation_coordinate_valid, coordinate],
            prediction[validation_coordinate_valid, coordinate],
            variance_tolerance=config.variance_tolerance,
        )
        if score is None:
            validation_values = validation_target[
                validation_coordinate_valid, coordinate
            ]
            if len(validation_values) < 2:
                reason = "insufficient_finite_validation_samples"
            else:
                reason = "constant_validation_target"
            coordinate_status.append(
                {
                    "coordinate_index": coordinate,
                    "coordinate_name": coordinate_name,
                    "status": "skipped",
                    "skipped_reason": reason,
                    "finite_validation_samples": int(
                        np.sum(validation_coordinate_valid)
                    ),
                }
            )
        else:
            coordinate_status.append(
                {
                    "coordinate_index": coordinate,
                    "coordinate_name": coordinate_name,
                    "status": "ok",
                    "skipped_reason": None,
                    "finite_train_samples": int(np.sum(coordinate_train_valid)),
                    "finite_validation_samples": int(
                        np.sum(validation_coordinate_valid)
                    ),
                }
            )

    coordinate_r2 = [
        _safe_r2(
            validation_target[:, coordinate],
            prediction[:, coordinate],
            variance_tolerance=config.variance_tolerance,
        )
        for coordinate in range(validation_target.shape[1])
    ]
    if not any(score is not None for score in coordinate_r2):
        return {
            "status": "skipped",
            "skipped_reason": "no_evaluable_validation_target_coordinates",
            "selected_alpha": selected_alpha,
            "probe_selection": selection,
            "coordinate_names": list(coordinate_names),
            "coordinate_r2": coordinate_r2,
            "coordinate_status": coordinate_status,
            "finite_train_rows": int(np.sum(train_valid)),
            "finite_validation_rows": int(np.sum(validation_valid)),
            "frozen_fit_strategy": selection["fit_strategy"],
        }

    by_subject = _subject_scores(
        validation_target,
        prediction,
        validation_subjects,
        variance_tolerance=config.variance_tolerance,
    )
    return _json_value(
        {
            "status": "ok",
            "skipped_reason": None,
            "selected_alpha": selected_alpha,
            "probe_selection": selection,
            "coordinate_names": list(coordinate_names),
            "coordinate_r2": coordinate_r2,
            "mean_r2": _mean_finite(coordinate_r2),
            "subject_r2": by_subject,
            "subject_bootstrap": _subject_bootstrap(
                by_subject,
                config=config,
                seed=bootstrap_seed,
            ),
            "representation_dimension": train_features.shape[1],
            "train_sample_count": len(train_features),
            "validation_sample_count": len(validation_features),
            "finite_train_rows": int(np.sum(train_valid)),
            "finite_validation_rows": int(np.sum(validation_valid)),
            "nonfinite_train_rows": int(np.sum(~train_valid)),
            "nonfinite_validation_rows": int(np.sum(~validation_valid)),
            "train_subject_count": int(len(np.unique(train_subjects[train_valid]))),
            "validation_subject_count": int(
                len(np.unique(validation_subjects[validation_valid]))
            ),
            "coordinate_status": coordinate_status,
            "frozen_fit_strategy": selection["fit_strategy"],
            "validation_used_for_model_selection": False,
        }
    )


def evaluate_information_ledger(
    train_target: np.ndarray,
    validation_target: np.ndarray,
    train_subjects: Sequence[Any],
    validation_subjects: Sequence[Any],
    train_representations: Mapping[str, np.ndarray],
    validation_representations: Mapping[str, np.ndarray],
    *,
    coordinate_names: Sequence[str] | None = None,
    config: InformationLedgerConfig | None = None,
) -> dict[str, Any]:
    """Evaluate representations with train-subject CV and frozen validation.

    Invalid top-level shapes and representation-specific failures are returned
    with ``status="skipped"`` and a ``skipped_reason``.  This behavior is
    deliberate: downstream reports must not silently replace unavailable R2
    values with zero or a favorable finite number.
    """

    config = config or InformationLedgerConfig()
    try:
        train_target = np.asarray(train_target, dtype=np.float64)
    except (TypeError, ValueError):
        return _skipped_ledger("invalid_train_target_values", config=config)
    try:
        validation_target = np.asarray(validation_target, dtype=np.float64)
    except (TypeError, ValueError):
        return _skipped_ledger("invalid_validation_target_values", config=config)
    if train_target.ndim != 2:
        return _skipped_ledger(
            "invalid_train_target_shape",
            config=config,
            observed_shape=list(train_target.shape),
        )
    if validation_target.ndim != 2:
        return _skipped_ledger(
            "invalid_validation_target_shape",
            config=config,
            observed_shape=list(validation_target.shape),
        )
    if train_target.shape[1] != validation_target.shape[1]:
        return _skipped_ledger(
            "train_validation_target_dimension_mismatch",
            config=config,
            train_target_dimension=train_target.shape[1],
            validation_target_dimension=validation_target.shape[1],
        )
    if train_target.shape[1] == 0:
        return _skipped_ledger("empty_target_dimension", config=config)

    train_subjects = np.asarray(train_subjects, dtype=object)
    validation_subjects = np.asarray(validation_subjects, dtype=object)
    if train_subjects.ndim != 1 or len(train_subjects) != len(train_target):
        return _skipped_ledger(
            "invalid_train_subject_shape",
            config=config,
            observed_shape=list(train_subjects.shape),
            expected_sample_count=len(train_target),
        )
    if (
        validation_subjects.ndim != 1
        or len(validation_subjects) != len(validation_target)
    ):
        return _skipped_ledger(
            "invalid_validation_subject_shape",
            config=config,
            observed_shape=list(validation_subjects.shape),
            expected_sample_count=len(validation_target),
        )
    if len(np.unique(train_subjects)) < 2:
        return _skipped_ledger(
            "fewer_than_two_train_subjects",
            config=config,
            train_subject_count=int(len(np.unique(train_subjects))),
        )

    if coordinate_names is None:
        coordinate_names = [
            f"feature_{index}" for index in range(train_target.shape[1])
        ]
    else:
        coordinate_names = [str(name) for name in coordinate_names]
        if len(coordinate_names) != train_target.shape[1]:
            return _skipped_ledger(
                "coordinate_name_count_mismatch",
                config=config,
                coordinate_name_count=len(coordinate_names),
                target_dimension=train_target.shape[1],
            )

    train_names = {str(name) for name in train_representations}
    validation_names = {str(name) for name in validation_representations}
    common_names = sorted(train_names & validation_names)
    diagnostics = {
        "coordinate_names": list(coordinate_names),
        "train_only_representations": sorted(train_names - validation_names),
        "validation_only_representations": sorted(validation_names - train_names),
        "common_representations": common_names,
        "train_target_nonfinite_count": int(np.sum(~np.isfinite(train_target))),
        "validation_target_nonfinite_count": int(
            np.sum(~np.isfinite(validation_target))
        ),
        "model_selection_partition": "training_subjects_only",
        "validation_role": "frozen_evaluation_only",
    }
    if not common_names:
        return _skipped_ledger(
            "no_common_representations",
            config=config,
            **diagnostics,
        )

    results: dict[str, Any] = {}
    for index, name in enumerate(common_names):
        try:
            train_features = np.asarray(
                train_representations[name], dtype=np.float64
            )
        except (TypeError, ValueError):
            results[name] = {
                "status": "skipped",
                "skipped_reason": "invalid_train_representation_values",
            }
            continue
        try:
            validation_features = np.asarray(
                validation_representations[name], dtype=np.float64
            )
        except (TypeError, ValueError):
            results[name] = {
                "status": "skipped",
                "skipped_reason": "invalid_validation_representation_values",
            }
            continue
        results[name] = _evaluate_representation(
            train_features,
            validation_features,
            train_target,
            validation_target,
            train_subjects,
            validation_subjects,
            coordinate_names,
            config=config,
            bootstrap_seed=config.seed + index,
        )

    successful = sum(result["status"] == "ok" for result in results.values())
    if successful == len(results):
        status = "ok"
        skipped_reason = None
    elif successful:
        status = "partial"
        skipped_reason = "some_representations_skipped"
    else:
        status = "skipped"
        skipped_reason = "all_common_representations_skipped"
    return _json_value(
        {
            "schema_version": TOKEN_INFORMATION_LEDGER_SCHEMA_VERSION,
            "status": status,
            "skipped_reason": skipped_reason,
            "config": asdict(config),
            "representations": results,
            **diagnostics,
        }
    )


def build_token_representations(
    *,
    continuous_latent: np.ndarray | None = None,
    hard_ids: np.ndarray | None = None,
    posterior: np.ndarray | None = None,
    codebook_embedding: np.ndarray | None = None,
    codebook: np.ndarray | None = None,
    codebook_size: int | None = None,
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Construct NumPy probe representations while preserving sample order.

    Arrays may have arbitrary leading sample dimensions, for example
    ``[batch, token, dimension]``.  The leading dimensions must agree and are
    flattened to ``N``.  ``valid_mask=False`` rows and invalid hard IDs are
    encoded as ``NaN`` rows; the ledger reports and excludes them without
    changing alignment with the physiological target.

    ``codebook_embedding`` accepts an already gathered ``[..., D]`` array.
    Alternatively, ``codebook`` and ``hard_ids`` construct the gathered hard
    codebook embedding without requiring PyTorch.
    """

    sample_shape: tuple[int, ...] | None = None

    def register_shape(name: str, shape: tuple[int, ...]) -> None:
        nonlocal sample_shape
        if sample_shape is None:
            sample_shape = shape
        elif sample_shape != shape:
            raise ValueError(
                f"{name} leading sample shape {shape} does not match "
                f"{sample_shape}"
            )

    matrices: dict[str, np.ndarray] = {}
    if continuous_latent is not None:
        latent = np.asarray(continuous_latent, dtype=np.float64)
        if latent.ndim < 2 or latent.shape[-1] == 0:
            raise ValueError("continuous_latent must have shape [...,D] with D>0")
        register_shape("continuous_latent", latent.shape[:-1])
        matrices["continuous_latent"] = latent.reshape(-1, latent.shape[-1])

    hard_array: np.ndarray | None = None
    if hard_ids is not None:
        hard_array = np.asarray(hard_ids)
        if hard_array.ndim < 1:
            raise ValueError("hard_ids must have shape [...]")
        register_shape("hard_ids", hard_array.shape)

    posterior_array: np.ndarray | None = None
    if posterior is not None:
        posterior_array = np.asarray(posterior, dtype=np.float64)
        if posterior_array.ndim < 2 or posterior_array.shape[-1] == 0:
            raise ValueError("posterior must have shape [...,K] with K>0")
        register_shape("posterior", posterior_array.shape[:-1])
        matrices["posterior"] = posterior_array.reshape(
            -1, posterior_array.shape[-1]
        )

    if codebook_embedding is not None and codebook is not None:
        raise ValueError(
            "provide either gathered codebook_embedding or codebook, not both"
        )
    if codebook_embedding is not None:
        embedding = np.asarray(codebook_embedding, dtype=np.float64)
        if embedding.ndim < 2 or embedding.shape[-1] == 0:
            raise ValueError("codebook_embedding must have shape [...,D] with D>0")
        register_shape("codebook_embedding", embedding.shape[:-1])
        matrices["codebook_embedding"] = embedding.reshape(
            -1, embedding.shape[-1]
        )

    lookup: np.ndarray | None = None
    if codebook is not None:
        lookup = np.asarray(codebook, dtype=np.float64)
        if lookup.ndim != 2 or min(lookup.shape) <= 0:
            raise ValueError("codebook must have shape [K,D] with K,D>0")
        if hard_array is None:
            raise ValueError("hard_ids are required to gather codebook embeddings")

    inferred_sizes = []
    if codebook_size is not None:
        if codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        inferred_sizes.append(int(codebook_size))
    if posterior_array is not None:
        inferred_sizes.append(int(posterior_array.shape[-1]))
    if lookup is not None:
        inferred_sizes.append(int(lookup.shape[0]))
    if len(set(inferred_sizes)) > 1:
        raise ValueError("codebook_size, posterior, and codebook K must agree")

    hard_flat: np.ndarray | None = None
    hard_valid: np.ndarray | None = None
    if hard_array is not None:
        try:
            hard_float = np.asarray(hard_array, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as error:
            raise ValueError("hard_ids must be numeric") from error
        if inferred_sizes:
            resolved_codebook_size = inferred_sizes[0]
        else:
            integral_nonnegative = (
                np.isfinite(hard_float)
                & (hard_float >= 0)
                & (hard_float == np.floor(hard_float))
            )
            if not np.any(integral_nonnegative):
                raise ValueError(
                    "cannot infer codebook_size when every hard ID is invalid"
                )
            resolved_codebook_size = int(np.max(hard_float[integral_nonnegative])) + 1
        hard_valid = (
            np.isfinite(hard_float)
            & (hard_float >= 0)
            & (hard_float < resolved_codebook_size)
            & (hard_float == np.floor(hard_float))
        )
        hard_flat = np.zeros(len(hard_float), dtype=np.int64)
        hard_flat[hard_valid] = hard_float[hard_valid].astype(np.int64)
        one_hot = np.full(
            (len(hard_flat), resolved_codebook_size), np.nan, dtype=np.float64
        )
        one_hot[hard_valid] = 0.0
        one_hot[np.flatnonzero(hard_valid), hard_flat[hard_valid]] = 1.0
        matrices["hard_one_hot"] = one_hot
        if lookup is not None:
            gathered = np.full(
                (len(hard_flat), lookup.shape[1]), np.nan, dtype=np.float64
            )
            gathered[hard_valid] = lookup[hard_flat[hard_valid]]
            matrices["codebook_embedding"] = gathered

    if sample_shape is None:
        return {}

    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != sample_shape:
            raise ValueError(
                f"valid_mask shape {mask.shape} does not match {sample_shape}"
            )
        flat_mask = mask.reshape(-1)
        for name, values in matrices.items():
            values = values.copy()
            values[~flat_mask] = np.nan
            matrices[name] = values

    return matrices


__all__ = [
    "InformationLedgerConfig",
    "TOKEN_INFORMATION_LEDGER_SCHEMA_VERSION",
    "build_token_representations",
    "evaluate_information_ledger",
]
