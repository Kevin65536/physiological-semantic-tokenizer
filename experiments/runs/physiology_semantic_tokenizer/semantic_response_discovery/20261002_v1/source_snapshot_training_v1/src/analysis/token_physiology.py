"""Auditable token-conditioned physiology summaries.

The utilities in this module deliberately treat codebook IDs as nominal
categories.  They summarize token support, assignment uncertainty, and
token-conditioned feature distributions without assigning physiological
meaning to an ID by construction.

All profile estimates use subjects as the statistical unit: measurements are
first averaged within subject and token, then subjects receive equal weight.
The implementation never materializes an ``[N, K, F]`` tensor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


TOKEN_PHYSIOLOGY_SCHEMA_VERSION = "token_physiology_v1"


@dataclass(frozen=True)
class TokenPhysiologyConfig:
    """Configuration for :func:`analyze_token_physiology`.

    ``rare_count`` defaults to ``min_count`` when left as ``None``.  A token is
    considered insufficiently supported when either its hard count or subject
    coverage falls below the corresponding minimum.
    """

    codebook_size: int | None = None
    min_count: int = 30
    min_subjects: int = 5
    rare_count: int | None = None
    bootstrap_iterations: int = 1000
    bootstrap_confidence: float = 0.95
    seed: int = 0
    max_state_categories: int = 64
    max_metadata_categories: int = 64

    def __post_init__(self) -> None:
        if self.codebook_size is not None and self.codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        if self.min_count < 0 or self.min_subjects < 0:
            raise ValueError("support thresholds must be non-negative")
        if self.rare_count is not None and self.rare_count < 0:
            raise ValueError("rare_count must be non-negative")
        if self.bootstrap_iterations < 0:
            raise ValueError("bootstrap_iterations must be non-negative")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise ValueError("bootstrap_confidence must lie strictly between 0 and 1")
        if self.max_state_categories <= 0 or self.max_metadata_categories <= 0:
            raise ValueError("category-count limits must be positive")


@dataclass
class TokenPhysiologyResult:
    """Structured, serialization-friendly token physiology result."""

    schema_version: str
    manifest: dict[str, Any]
    support_rows: list[dict[str, Any]]
    profile_rows: list[dict[str, Any]]
    signature_rows: list[dict[str, Any]]
    state_rows: list[dict[str, Any]]
    metadata_rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "manifest": _json_value(self.manifest),
            "support_rows": _json_value(self.support_rows),
            "profile_rows": _json_value(self.profile_rows),
            "signature_rows": _json_value(self.signature_rows),
            "state_rows": _json_value(self.state_rows),
            "metadata_rows": _json_value(self.metadata_rows),
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
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


def _finite_or_none(value: float | np.floating[Any]) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _is_missing_category(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return not np.isfinite(value)
    try:
        missing = np.isnat(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if np.ndim(missing) == 0 else False


def _category_key(value: Any) -> tuple[str, str]:
    scalar = _json_value(value)
    return type(scalar).__name__, repr(scalar)


def _as_categorical_vector(
    values: Sequence[Any],
    *,
    name: str,
    length: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must be aligned shape [N]")
    return array


def _categorical_groups(
    values: np.ndarray,
    valid_mask: np.ndarray,
) -> list[tuple[Any, np.ndarray]]:
    grouped: dict[tuple[str, str], tuple[Any, list[int]]] = {}
    for index in np.flatnonzero(valid_mask):
        value = values[index]
        if _is_missing_category(value):
            continue
        key = _category_key(value)
        if key not in grouped:
            grouped[key] = (_json_value(value), [])
        grouped[key][1].append(int(index))
    return [
        (grouped[key][0], np.asarray(grouped[key][1], dtype=np.int64))
        for key in sorted(grouped)
    ]


def _prepare_named_categories(
    values: Mapping[str, Sequence[Any]] | Sequence[Any] | None,
    *,
    default_name: str,
    length: int,
) -> dict[str, np.ndarray]:
    if values is None:
        return {}
    if isinstance(values, Mapping):
        return {
            str(name): _as_categorical_vector(vector, name=str(name), length=length)
            for name, vector in values.items()
        }
    return {
        default_name: _as_categorical_vector(
            values,
            name=default_name,
            length=length,
        )
    }


def _prepare_hard_ids(
    hard_ids: Sequence[Any],
    *,
    input_mask: np.ndarray,
    requested_codebook_size: int | None,
    posterior_size: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    raw = np.asarray(hard_ids)
    if raw.ndim != 1 or len(raw) != len(input_mask):
        raise ValueError("hard_ids must be aligned shape [N]")
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("hard_ids must contain integer-valued nominal codes") from error
    finite = np.isfinite(numeric)
    non_integral = input_mask & finite & (numeric != np.floor(numeric))
    if np.any(non_integral):
        raise ValueError("hard_ids must contain integer-valued nominal codes")

    non_negative = input_mask & finite & (numeric >= 0)
    inferred_size = (
        int(np.max(numeric[non_negative])) + 1 if np.any(non_negative) else None
    )
    size_candidates = [
        value
        for value in (requested_codebook_size, posterior_size, inferred_size)
        if value is not None
    ]
    if not size_candidates:
        raise ValueError(
            "cannot infer codebook size from empty IDs; provide codebook_size or posterior"
        )
    codebook_size = int(
        requested_codebook_size
        if requested_codebook_size is not None
        else posterior_size if posterior_size is not None else inferred_size
    )
    if posterior_size is not None and posterior_size != codebook_size:
        raise ValueError("posterior width must match codebook_size")
    if inferred_size is not None and inferred_size > codebook_size:
        raise ValueError("hard_ids contain a code outside codebook_size")

    ids = np.full(len(numeric), -1, dtype=np.int64)
    convertible = finite & non_negative & (numeric < codebook_size)
    ids[convertible] = numeric[convertible].astype(np.int64)
    return ids, convertible, codebook_size


def _prepare_posterior(
    posterior: np.ndarray | None,
    *,
    length: int,
    codebook_size: int | None,
    base_mask: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray, int | None]:
    if posterior is None:
        return None, np.zeros(length, dtype=bool), None
    values = np.asarray(posterior, dtype=np.float64)
    if values.ndim != 2 or len(values) != length:
        raise ValueError("posterior must be aligned shape [N,K]")
    if values.shape[1] == 0:
        raise ValueError("posterior must have at least one code")
    if codebook_size is not None and values.shape[1] != codebook_size:
        raise ValueError("posterior width must match codebook_size")
    finite_rows = np.all(np.isfinite(values), axis=1)
    if np.any(base_mask[:, None] & np.isfinite(values) & (values < 0.0)):
        raise ValueError("posterior probabilities must be non-negative")
    row_sum = np.sum(np.where(np.isfinite(values), values, 0.0), axis=1)
    valid = base_mask & finite_rows & (row_sum > 0.0)
    normalized = np.zeros_like(values)
    normalized[valid] = values[valid] / row_sum[valid, None]
    return normalized, valid, int(values.shape[1])


def _subject_equal_marginal(
    features: np.ndarray,
    subject_indices: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    feature_count = features.shape[1]
    subject_means = np.full((len(subject_indices), feature_count), np.nan)
    for subject_index, indices in enumerate(subject_indices):
        for feature_index in range(feature_count):
            values = features[indices, feature_index]
            values = values[np.isfinite(values)]
            if len(values):
                subject_means[subject_index, feature_index] = np.mean(values)
    marginal_mean = np.full(feature_count, np.nan)
    marginal_scale = np.full(feature_count, np.nan)
    for feature_index in range(feature_count):
        means = subject_means[:, feature_index]
        valid_subjects = np.isfinite(means)
        if not np.any(valid_subjects):
            continue
        marginal_mean[feature_index] = np.mean(means[valid_subjects])
        subject_mse: list[float] = []
        for indices in subject_indices:
            values = features[indices, feature_index]
            values = values[np.isfinite(values)]
            if len(values):
                subject_mse.append(
                    float(np.mean(np.square(values - marginal_mean[feature_index])))
                )
        if subject_mse:
            marginal_scale[feature_index] = np.sqrt(np.mean(subject_mse))
    return marginal_mean, marginal_scale


def _subject_token_means(
    features: np.ndarray,
    hard_ids: np.ndarray,
    subject_indices: list[np.ndarray],
    *,
    codebook_size: int,
    posterior: np.ndarray | None,
    posterior_valid: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Compute subject-token means without an ``[N,K,F]`` broadcast."""

    feature_count = features.shape[1]
    hard_means = np.full(
        (len(subject_indices), codebook_size, feature_count),
        np.nan,
        dtype=np.float64,
    )
    soft_means = (
        np.full_like(hard_means, np.nan) if posterior is not None else None
    )
    hard_finite_counts = np.zeros((codebook_size, feature_count), dtype=np.int64)
    soft_weight_sum = (
        np.zeros((codebook_size, feature_count), dtype=np.float64)
        if posterior is not None
        else None
    )
    soft_weight_square_sum = (
        np.zeros((codebook_size, feature_count), dtype=np.float64)
        if posterior is not None
        else None
    )

    for subject_index, indices in enumerate(subject_indices):
        subject_features = features[indices]
        subject_codes = hard_ids[indices]
        for feature_index in range(feature_count):
            values = subject_features[:, feature_index]
            finite = np.isfinite(values)
            if np.any(finite):
                counts = np.bincount(
                    subject_codes[finite],
                    minlength=codebook_size,
                )
                sums = np.bincount(
                    subject_codes[finite],
                    weights=values[finite],
                    minlength=codebook_size,
                )
                active = counts > 0
                hard_means[subject_index, active, feature_index] = (
                    sums[active] / counts[active]
                )
                hard_finite_counts[:, feature_index] += counts

            if posterior is None:
                continue
            soft_rows = finite & posterior_valid[indices]
            if not np.any(soft_rows):
                continue
            weights = posterior[indices[soft_rows]]
            soft_values = values[soft_rows]
            denominator = np.sum(weights, axis=0)
            numerator = weights.T @ soft_values
            active = denominator > 0.0
            assert soft_means is not None
            soft_means[subject_index, active, feature_index] = (
                numerator[active] / denominator[active]
            )
            assert soft_weight_sum is not None
            assert soft_weight_square_sum is not None
            soft_weight_sum[:, feature_index] += denominator
            soft_weight_square_sum[:, feature_index] += np.sum(
                np.square(weights),
                axis=0,
            )

    return (
        hard_means,
        soft_means,
        hard_finite_counts,
        soft_weight_sum,
        soft_weight_square_sum,
    )


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    iterations: int,
    confidence: float,
    seed_components: Sequence[int],
) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values) or iterations == 0:
        return None, None
    rng = np.random.default_rng(np.random.SeedSequence(seed_components))
    selected = rng.integers(0, len(values), size=(iterations, len(values)))
    samples = np.mean(values[selected], axis=1)
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(samples, tail)),
        float(np.quantile(samples, 1.0 - tail)),
    )


def _subject_equal_selected_mean(
    values: np.ndarray,
    selected: np.ndarray,
    subject_indices: list[np.ndarray],
) -> float | None:
    means: list[float] = []
    for indices in subject_indices:
        subject_values = values[indices]
        subject_selected = selected[indices] & np.isfinite(subject_values)
        if np.any(subject_selected):
            means.append(float(np.mean(subject_values[subject_selected])))
    return float(np.mean(means)) if means else None


def _support_rows(
    hard_ids: np.ndarray,
    analysis_mask: np.ndarray,
    subject_indices: list[np.ndarray],
    *,
    codebook_size: int,
    min_count: int,
    min_subjects: int,
    rare_count: int,
    posterior: np.ndarray | None,
    posterior_valid: np.ndarray,
) -> list[dict[str, Any]]:
    posterior_entropy: np.ndarray | None = None
    posterior_normalized_entropy: np.ndarray | None = None
    posterior_margin: np.ndarray | None = None
    if posterior is not None:
        safe = np.where(posterior > 0.0, posterior, 1.0)
        posterior_entropy = -np.sum(posterior * np.log(safe), axis=1)
        if codebook_size > 1:
            posterior_normalized_entropy = posterior_entropy / np.log(codebook_size)
            top_two = np.partition(posterior, -2, axis=1)[:, -2:]
            posterior_margin = top_two[:, 1] - top_two[:, 0]
        else:
            posterior_normalized_entropy = np.zeros(len(posterior))
            posterior_margin = posterior[:, 0]

    rows: list[dict[str, Any]] = []
    for token_id in range(codebook_size):
        selected = analysis_mask & (hard_ids == token_id)
        subject_counts = np.asarray(
            [int(np.sum(selected[indices])) for indices in subject_indices],
            dtype=np.int64,
        )
        subject_counts = subject_counts[subject_counts > 0]
        count = int(np.sum(subject_counts))
        subject_count = int(len(subject_counts))
        if count:
            fractions = subject_counts / count
            subject_entropy = float(-np.sum(fractions * np.log(fractions)))
            effective_subjects = float(np.exp(subject_entropy))
            max_subject_fraction = float(np.max(fractions))
            normalized_subject_entropy = (
                subject_entropy / np.log(subject_count) if subject_count > 1 else 0.0
            )
        else:
            subject_entropy = 0.0
            effective_subjects = 0.0
            max_subject_fraction = None
            normalized_subject_entropy = 0.0
        insufficient = count < min_count or subject_count < min_subjects
        assignment_selected = selected & posterior_valid
        row: dict[str, Any] = {
            "token_id": token_id,
            "count": count,
            "subject_count": subject_count,
            "effective_subjects": effective_subjects,
            "max_subject_fraction": max_subject_fraction,
            "subject_entropy": subject_entropy,
            "normalized_subject_entropy": normalized_subject_entropy,
            "inactive": count == 0,
            "rare": 0 < count < rare_count,
            "insufficient_support": insufficient,
            "support_status": (
                "inactive"
                if count == 0
                else "insufficient"
                if insufficient
                else "sufficient"
            ),
            "posterior_valid_count": int(np.sum(assignment_selected)),
            "posterior_entropy_subject_equal_mean": None,
            "posterior_normalized_entropy_subject_equal_mean": None,
            "posterior_margin_subject_equal_mean": None,
        }
        if posterior_entropy is not None:
            row["posterior_entropy_subject_equal_mean"] = (
                _subject_equal_selected_mean(
                    posterior_entropy,
                    assignment_selected,
                    subject_indices,
                )
            )
            assert posterior_normalized_entropy is not None
            row["posterior_normalized_entropy_subject_equal_mean"] = (
                _subject_equal_selected_mean(
                    posterior_normalized_entropy,
                    assignment_selected,
                    subject_indices,
                )
            )
            assert posterior_margin is not None
            row["posterior_margin_subject_equal_mean"] = (
                _subject_equal_selected_mean(
                    posterior_margin,
                    assignment_selected,
                    subject_indices,
                )
            )
        rows.append(row)
    return rows


def _profile_rows(
    subject_means: np.ndarray,
    *,
    profile_type: str,
    feature_names: Sequence[str],
    finite_counts: np.ndarray,
    weight_sums: np.ndarray,
    effective_counts: np.ndarray,
    marginal_mean: np.ndarray,
    marginal_scale: np.ndarray,
    support_rows: list[dict[str, Any]],
    config: TokenPhysiologyConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    codebook_size = subject_means.shape[1]
    rows: list[dict[str, Any]] = []
    signatures: list[dict[str, Any]] = []
    for token_id in range(codebook_size):
        signature_mean: list[float | None] = []
        signature_effect: list[float | None] = []
        for feature_index, feature_name in enumerate(feature_names):
            values = subject_means[:, token_id, feature_index]
            values = values[np.isfinite(values)]
            subject_count = int(len(values))
            if subject_count:
                subject_equal_mean = float(np.mean(values))
                median = float(np.median(values))
                q25, q75 = np.quantile(values, (0.25, 0.75))
            else:
                subject_equal_mean = np.nan
                median = np.nan
                q25 = np.nan
                q75 = np.nan
            scale = marginal_scale[feature_index]
            effect = (
                (subject_equal_mean - marginal_mean[feature_index]) / scale
                if np.isfinite(subject_equal_mean)
                and np.isfinite(scale)
                and scale > 0.0
                else np.nan
            )
            profile_code = 0 if profile_type == "hard" else 1
            ci_low, ci_high = _bootstrap_mean_ci(
                values,
                iterations=config.bootstrap_iterations,
                confidence=config.bootstrap_confidence,
                seed_components=(config.seed, profile_code, token_id, feature_index),
            )
            mean_value = _finite_or_none(subject_equal_mean)
            effect_value = _finite_or_none(effect)
            rows.append(
                {
                    "profile_type": profile_type,
                    "token_id": token_id,
                    "feature_index": feature_index,
                    "feature_name": feature_name,
                    "finite_count": int(finite_counts[token_id, feature_index]),
                    "weight_sum": float(weight_sums[token_id, feature_index]),
                    "effective_count": float(
                        effective_counts[token_id, feature_index]
                    ),
                    "subject_count": subject_count,
                    "subject_equal_mean": mean_value,
                    "subject_equal_median": _finite_or_none(median),
                    "subject_equal_q25": _finite_or_none(q25),
                    "subject_equal_q75": _finite_or_none(q75),
                    "subject_equal_iqr": (
                        _finite_or_none(q75 - q25)
                        if np.isfinite(q25) and np.isfinite(q75)
                        else None
                    ),
                    "marginal_subject_equal_mean": _finite_or_none(
                        marginal_mean[feature_index]
                    ),
                    "marginal_subject_equal_scale": _finite_or_none(scale),
                    "marginal_standardized_effect": effect_value,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "bootstrap_unit": "subject",
                    "insufficient_support": support_rows[token_id][
                        "insufficient_support"
                    ],
                }
            )
            signature_mean.append(mean_value)
            signature_effect.append(effect_value)
        signatures.append(
            {
                "profile_type": profile_type,
                "token_id": token_id,
                "feature_names": list(feature_names),
                "subject_equal_mean": signature_mean,
                "marginal_standardized_effect": signature_effect,
                "count": support_rows[token_id]["count"],
                "subject_count": support_rows[token_id]["subject_count"],
                "insufficient_support": support_rows[token_id][
                    "insufficient_support"
                ],
            }
        )
    return rows, signatures


def _normalized_mutual_information(joint: np.ndarray) -> float:
    joint = np.asarray(joint, dtype=np.float64)
    total = float(np.sum(joint))
    if total <= 0.0:
        return 0.0
    probability = joint / total
    token_probability = np.sum(probability, axis=1)
    state_probability = np.sum(probability, axis=0)
    expected = token_probability[:, None] * state_probability[None, :]
    occupied = probability > 0.0
    mutual_information = max(
        0.0,
        float(
            np.sum(
                probability[occupied]
                * np.log(probability[occupied] / expected[occupied])
            )
        ),
    )
    token_occupied = token_probability > 0.0
    state_occupied = state_probability > 0.0
    token_entropy = max(
        0.0,
        float(
            -np.sum(
                token_probability[token_occupied]
                * np.log(token_probability[token_occupied])
            )
        ),
    )
    state_entropy = max(
        0.0,
        float(
            -np.sum(
                state_probability[state_occupied]
                * np.log(state_probability[state_occupied])
            )
        ),
    )
    denominator = np.sqrt(token_entropy * state_entropy)
    return mutual_information / denominator if denominator > 0.0 else 0.0


def _contingency_rows(
    values: Mapping[str, np.ndarray],
    *,
    row_kind: str,
    hard_ids: np.ndarray,
    subjects: np.ndarray,
    analysis_mask: np.ndarray,
    codebook_size: int,
    support_rows: list[dict[str, Any]],
    max_categories: int | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for variable_name, vector in values.items():
        groups = _categorical_groups(vector, analysis_mask)
        if max_categories is not None and len(groups) > max_categories:
            skipped.append(variable_name)
            continue
        state_valid = analysis_mask & np.asarray(
            [not _is_missing_category(value) for value in vector],
            dtype=bool,
        )
        total = int(np.sum(state_valid))
        if total == 0:
            continue
        token_counts = np.bincount(
            hard_ids[state_valid],
            minlength=codebook_size,
        ).astype(np.int64)
        joint = np.zeros((codebook_size, len(groups)), dtype=np.int64)
        category_counts = np.zeros(len(groups), dtype=np.int64)
        for category_index, (_, indices) in enumerate(groups):
            selected_indices = indices[state_valid[indices]]
            if len(selected_indices):
                joint[:, category_index] = np.bincount(
                    hard_ids[selected_indices],
                    minlength=codebook_size,
                )
                category_counts[category_index] = len(selected_indices)
        nmi = _normalized_mutual_information(joint)
        category_codes = np.full(len(vector), -1, dtype=np.int64)
        for category_index, (_, indices) in enumerate(groups):
            category_codes[indices] = category_index
        subject_contingencies: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, int]
        ] = []
        for _, subject_indices in _categorical_groups(subjects, state_valid):
            flat = (
                hard_ids[subject_indices] * len(groups)
                + category_codes[subject_indices]
            )
            subject_joint = np.bincount(
                flat,
                minlength=codebook_size * len(groups),
            ).reshape(codebook_size, len(groups))
            subject_token_counts = subject_joint.sum(axis=1)
            subject_category_counts = subject_joint.sum(axis=0)
            subject_contingencies.append(
                (
                    subject_joint,
                    subject_token_counts,
                    subject_category_counts,
                    int(len(subject_indices)),
                )
            )
        subject_nmi = [
            _normalized_mutual_information(subject_joint)
            for subject_joint, _, _, _ in subject_contingencies
        ]
        for token_id in range(codebook_size):
            for category_index, (category, _) in enumerate(groups):
                joint_count = int(joint[token_id, category_index])
                token_count = int(token_counts[token_id])
                category_count = int(category_counts[category_index])
                p_category_given_token = (
                    joint_count / token_count if token_count else None
                )
                p_token_given_category = (
                    joint_count / category_count if category_count else None
                )
                p_category = category_count / total
                p_token = token_count / total
                lift = (
                    p_category_given_token / p_category
                    if p_category_given_token is not None and p_category > 0.0
                    else None
                )
                subject_p_category_given_token = [
                    float(subject_joint[token_id, category_index])
                    / float(subject_token_counts[token_id])
                    for (
                        subject_joint,
                        subject_token_counts,
                        _,
                        _,
                    ) in subject_contingencies
                    if subject_token_counts[token_id] > 0
                ]
                subject_p_token_given_category = [
                    float(subject_joint[token_id, category_index])
                    / float(subject_category_counts[category_index])
                    for (
                        subject_joint,
                        _,
                        subject_category_counts,
                        _,
                    ) in subject_contingencies
                    if subject_category_counts[category_index] > 0
                ]
                subject_marginal_category = [
                    float(subject_category_counts[category_index])
                    / float(subject_total)
                    for (
                        _,
                        _,
                        subject_category_counts,
                        subject_total,
                    ) in subject_contingencies
                ]
                subject_marginal_token = [
                    float(subject_token_counts[token_id])
                    / float(subject_total)
                    for (
                        _,
                        subject_token_counts,
                        _,
                        subject_total,
                    ) in subject_contingencies
                ]
                subject_equal_p_category_given_token = (
                    float(np.mean(subject_p_category_given_token))
                    if subject_p_category_given_token
                    else None
                )
                subject_equal_p_token_given_category = (
                    float(np.mean(subject_p_token_given_category))
                    if subject_p_token_given_category
                    else None
                )
                subject_equal_p_category = (
                    float(np.mean(subject_marginal_category))
                    if subject_marginal_category
                    else None
                )
                subject_equal_p_token = (
                    float(np.mean(subject_marginal_token))
                    if subject_marginal_token
                    else None
                )
                rows.append(
                    {
                        "row_kind": row_kind,
                        "variable_name": variable_name,
                        "token_id": token_id,
                        "category": category,
                        "joint_count": joint_count,
                        "token_count": token_count,
                        "category_count": category_count,
                        "total_count": total,
                        "p_category_given_token": p_category_given_token,
                        "p_token_given_category": p_token_given_category,
                        "p_category": p_category,
                        "p_token": p_token,
                        "lift": lift,
                        "normalized_mutual_information": nmi,
                        "subject_equal_p_category_given_token": (
                            subject_equal_p_category_given_token
                        ),
                        "subject_count_p_category_given_token": len(
                            subject_p_category_given_token
                        ),
                        "subject_equal_p_token_given_category": (
                            subject_equal_p_token_given_category
                        ),
                        "subject_count_p_token_given_category": len(
                            subject_p_token_given_category
                        ),
                        "subject_equal_p_category": subject_equal_p_category,
                        "subject_equal_p_token": subject_equal_p_token,
                        "subject_equal_lift": (
                            subject_equal_p_category_given_token
                            / subject_equal_p_category
                            if subject_equal_p_category_given_token is not None
                            and subject_equal_p_category is not None
                            and subject_equal_p_category > 0.0
                            else None
                        ),
                        "subject_equal_mean_normalized_mutual_information": (
                            float(np.mean(subject_nmi))
                            if subject_nmi
                            else None
                        ),
                        "subject_equal_total_subject_count": len(
                            subject_contingencies
                        ),
                        "counting_unit": "valid token-aligned patch",
                        "association_scope": (
                            "patch-weighted and subject-equal descriptive "
                            "contingency; not causal inference"
                        ),
                        "insufficient_support": support_rows[token_id][
                            "insufficient_support"
                        ],
                    }
                )
    return rows, skipped


def analyze_token_physiology(
    features: np.ndarray,
    hard_ids: Sequence[Any],
    subjects: Sequence[Any],
    *,
    feature_names: Sequence[str] | None = None,
    posterior: np.ndarray | None = None,
    states: Mapping[str, Sequence[Any]] | Sequence[Any] | None = None,
    metadata: Mapping[str, Sequence[Any]] | None = None,
    valid_mask: Sequence[bool] | None = None,
    config: TokenPhysiologyConfig | None = None,
) -> TokenPhysiologyResult:
    """Build subject-balanced token physiology tables.

    Parameters
    ----------
    features:
        Continuous physiological descriptors with shape ``[N,F]``.
    hard_ids:
        Nominal hard code assignments with shape ``[N]``. NaN or negative IDs
        are excluded and counted in the manifest.
    subjects:
        Subject identifiers with shape ``[N]``. Missing subject identifiers are
        excluded because subject-balanced inference is otherwise undefined.
    posterior:
        Optional soft assignments with shape ``[N,K]``. Rows are normalized
        internally; non-finite or zero-mass rows are excluded from soft profiles.
    states:
        One categorical physiological-state vector or a mapping of named state
        vectors. Continuous features should remain in ``features`` rather than
        being discretized solely for this analysis.
    metadata:
        Optional named categorical nuisance/context vectors. Low-cardinality
        fields receive the same directional contingency summaries as states.
    """

    config = config or TokenPhysiologyConfig()
    feature_values = np.asarray(features, dtype=np.float64)
    if feature_values.ndim != 2 or len(feature_values) == 0:
        raise ValueError("features must be a non-empty matrix [N,F]")
    sample_count, feature_count = feature_values.shape
    if feature_count == 0:
        raise ValueError("features must contain at least one coordinate")
    if feature_names is None:
        names = tuple(f"feature_{index}" for index in range(feature_count))
    else:
        names = tuple(str(name) for name in feature_names)
        if len(names) != feature_count or len(set(names)) != len(names):
            raise ValueError("feature_names must be unique and align with features")

    if valid_mask is None:
        input_mask = np.ones(sample_count, dtype=bool)
    else:
        input_mask = np.asarray(valid_mask, dtype=bool)
        if input_mask.ndim != 1 or len(input_mask) != sample_count:
            raise ValueError("valid_mask must be aligned shape [N]")

    subject_values = _as_categorical_vector(
        subjects,
        name="subjects",
        length=sample_count,
    )
    subject_present = np.asarray(
        [not _is_missing_category(value) for value in subject_values],
        dtype=bool,
    )
    categorical_mask = input_mask & subject_present

    posterior_width = None
    if posterior is not None:
        posterior_array = np.asarray(posterior)
        if posterior_array.ndim != 2 or len(posterior_array) != sample_count:
            raise ValueError("posterior must be aligned shape [N,K]")
        posterior_width = int(posterior_array.shape[1])
    ids, hard_id_valid, codebook_size = _prepare_hard_ids(
        hard_ids,
        input_mask=categorical_mask,
        requested_codebook_size=config.codebook_size,
        posterior_size=posterior_width,
    )
    analysis_mask = categorical_mask & hard_id_valid
    normalized_posterior, posterior_valid, _ = _prepare_posterior(
        posterior,
        length=sample_count,
        codebook_size=codebook_size,
        base_mask=analysis_mask,
    )

    subject_groups = _categorical_groups(subject_values, analysis_mask)
    subject_indices = [indices for _, indices in subject_groups]
    if not subject_indices:
        raise ValueError("no valid subject-token rows remain after masking")

    rare_count = (
        config.min_count if config.rare_count is None else config.rare_count
    )
    support_rows = _support_rows(
        ids,
        analysis_mask,
        subject_indices,
        codebook_size=codebook_size,
        min_count=config.min_count,
        min_subjects=config.min_subjects,
        rare_count=rare_count,
        posterior=normalized_posterior,
        posterior_valid=posterior_valid,
    )
    (
        hard_subject_means,
        soft_subject_means,
        hard_finite_counts,
        soft_weight_sum,
        soft_weight_square_sum,
    ) = _subject_token_means(
        feature_values,
        ids,
        subject_indices,
        codebook_size=codebook_size,
        posterior=normalized_posterior,
        posterior_valid=posterior_valid,
    )
    marginal_mean, marginal_scale = _subject_equal_marginal(
        feature_values,
        subject_indices,
    )

    hard_weight = hard_finite_counts.astype(np.float64)
    hard_effective = hard_weight.copy()
    profile_rows, signature_rows = _profile_rows(
        hard_subject_means,
        profile_type="hard",
        feature_names=names,
        finite_counts=hard_finite_counts,
        weight_sums=hard_weight,
        effective_counts=hard_effective,
        marginal_mean=marginal_mean,
        marginal_scale=marginal_scale,
        support_rows=support_rows,
        config=config,
    )
    if soft_subject_means is not None:
        assert soft_weight_sum is not None
        assert soft_weight_square_sum is not None
        soft_effective = np.divide(
            np.square(soft_weight_sum),
            soft_weight_square_sum,
            out=np.zeros_like(soft_weight_sum),
            where=soft_weight_square_sum > 0.0,
        )
        soft_finite_counts = np.zeros_like(hard_finite_counts)
        for feature_index in range(feature_count):
            finite = (
                analysis_mask
                & posterior_valid
                & np.isfinite(feature_values[:, feature_index])
            )
            if np.any(finite):
                positive = normalized_posterior[finite] > 0.0
                soft_finite_counts[:, feature_index] = np.sum(positive, axis=0)
        soft_profiles, soft_signatures = _profile_rows(
            soft_subject_means,
            profile_type="soft",
            feature_names=names,
            finite_counts=soft_finite_counts,
            weight_sums=soft_weight_sum,
            effective_counts=soft_effective,
            marginal_mean=marginal_mean,
            marginal_scale=marginal_scale,
            support_rows=support_rows,
            config=config,
        )
        profile_rows.extend(soft_profiles)
        signature_rows.extend(soft_signatures)

    state_values = _prepare_named_categories(
        states,
        default_name="state",
        length=sample_count,
    )
    metadata_values = _prepare_named_categories(
        metadata,
        default_name="metadata",
        length=sample_count,
    )
    state_rows, skipped_states = _contingency_rows(
        state_values,
        row_kind="physiological_state",
        hard_ids=ids,
        subjects=subject_values,
        analysis_mask=analysis_mask,
        codebook_size=codebook_size,
        support_rows=support_rows,
        max_categories=config.max_state_categories,
    )
    metadata_rows, skipped_metadata = _contingency_rows(
        metadata_values,
        row_kind="metadata",
        hard_ids=ids,
        subjects=subject_values,
        analysis_mask=analysis_mask,
        codebook_size=codebook_size,
        support_rows=support_rows,
        max_categories=config.max_metadata_categories,
    )

    manifest = {
        "schema_version": TOKEN_PHYSIOLOGY_SCHEMA_VERSION,
        "sample_count": sample_count,
        "input_valid_count": int(np.sum(input_mask)),
        "analysis_valid_count": int(np.sum(analysis_mask)),
        "masked_count": int(sample_count - np.sum(input_mask)),
        "missing_subject_count": int(np.sum(input_mask & ~subject_present)),
        "invalid_hard_id_count": int(
            np.sum(categorical_mask & ~hard_id_valid)
        ),
        "subject_count": len(subject_indices),
        "feature_count": feature_count,
        "feature_names": list(names),
        "feature_nonfinite_count": np.sum(
            analysis_mask[:, None] & ~np.isfinite(feature_values),
            axis=0,
        ).astype(int).tolist(),
        "codebook_size": codebook_size,
        "profile_statistical_unit": "subject",
        "profile_distribution_unit": "within-subject token means",
        "marginal_effect_definition": (
            "(subject-equal token mean - subject-equal marginal mean) / "
            "subject-equal marginal scale"
        ),
        "posterior_provided": normalized_posterior is not None,
        "posterior_valid_count": int(np.sum(posterior_valid)),
        "posterior_invalid_count": (
            int(np.sum(analysis_mask & ~posterior_valid))
            if normalized_posterior is not None
            else 0
        ),
        "soft_profile_available": soft_subject_means is not None,
        "state_fields": list(state_values),
        "skipped_state_fields": skipped_states,
        "metadata_fields": list(metadata_values),
        "skipped_metadata_fields": skipped_metadata,
        "support_thresholds": {
            "min_count": config.min_count,
            "min_subjects": config.min_subjects,
            "rare_count": rare_count,
        },
        "bootstrap": {
            "iterations": config.bootstrap_iterations,
            "confidence": config.bootstrap_confidence,
            "seed": config.seed,
            "unit": "subject",
        },
        "config": asdict(config),
    }
    return TokenPhysiologyResult(
        schema_version=TOKEN_PHYSIOLOGY_SCHEMA_VERSION,
        manifest=manifest,
        support_rows=support_rows,
        profile_rows=profile_rows,
        signature_rows=signature_rows,
        state_rows=state_rows,
        metadata_rows=metadata_rows,
    )


def _signature_map(
    result: TokenPhysiologyResult,
    *,
    profile_type: str,
    value_field: str,
) -> tuple[list[str], dict[int, np.ndarray]]:
    selected = [
        row
        for row in result.signature_rows
        if row["profile_type"] == profile_type
    ]
    if not selected:
        raise ValueError(f"result has no {profile_type!r} signatures")
    names = list(selected[0]["feature_names"])
    signatures: dict[int, np.ndarray] = {}
    for row in selected:
        if list(row["feature_names"]) != names:
            raise ValueError("signature feature order is inconsistent")
        if value_field not in row:
            raise ValueError(f"unknown signature value field {value_field!r}")
        signatures[int(row["token_id"])] = np.asarray(
            [
                np.nan if value is None else float(value)
                for value in row[value_field]
            ],
            dtype=np.float64,
        )
    return names, signatures


def _pairwise_cosine(
    left: np.ndarray,
    right: np.ndarray,
    *,
    min_feature_overlap: int,
) -> tuple[float, int]:
    shared = np.isfinite(left) & np.isfinite(right)
    overlap = int(np.sum(shared))
    if overlap < min_feature_overlap:
        return -1.0, overlap
    left_values = left[shared]
    right_values = right[shared]
    left_norm = float(np.linalg.norm(left_values))
    right_norm = float(np.linalg.norm(right_values))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0, overlap
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0, overlap
    return float(np.dot(left_values, right_values) / (left_norm * right_norm)), overlap


def match_token_signatures(
    left: TokenPhysiologyResult,
    right: TokenPhysiologyResult,
    *,
    profile_type: str = "hard",
    value_field: str = "marginal_standardized_effect",
    min_count: int | None = None,
    min_subjects: int | None = None,
    min_feature_overlap: int = 1,
    bootstrap_iterations: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Hungarian-match supported token phenotype signatures.

    Optional bootstrap intervals resample the already matched token pairs; the
    alignment is deliberately held fixed.  Subject-level uncertainty should be
    assessed from the profile confidence intervals before interpreting a match.
    """

    if min_feature_overlap <= 0:
        raise ValueError("min_feature_overlap must be positive")
    if bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be non-negative")
    left_names, left_signatures = _signature_map(
        left,
        profile_type=profile_type,
        value_field=value_field,
    )
    right_names, right_signatures = _signature_map(
        right,
        profile_type=profile_type,
        value_field=value_field,
    )
    common_names = [name for name in left_names if name in set(right_names)]
    if not common_names:
        raise ValueError("signature results share no feature names")
    left_columns = [left_names.index(name) for name in common_names]
    right_columns = [right_names.index(name) for name in common_names]

    left_support = {
        int(row["token_id"]): row for row in left.support_rows
    }
    right_support = {
        int(row["token_id"]): row for row in right.support_rows
    }
    if min_count is None:
        min_count = max(
            int(left.manifest["support_thresholds"]["min_count"]),
            int(right.manifest["support_thresholds"]["min_count"]),
        )
    if min_subjects is None:
        min_subjects = max(
            int(left.manifest["support_thresholds"]["min_subjects"]),
            int(right.manifest["support_thresholds"]["min_subjects"]),
        )
    if min_count < 0 or min_subjects < 0:
        raise ValueError("support thresholds must be non-negative")

    active_left = [
        token_id
        for token_id, row in left_support.items()
        if row["count"] >= min_count
        and row["subject_count"] >= min_subjects
        and token_id in left_signatures
        and np.any(np.isfinite(left_signatures[token_id][left_columns]))
    ]
    active_right = [
        token_id
        for token_id, row in right_support.items()
        if row["count"] >= min_count
        and row["subject_count"] >= min_subjects
        and token_id in right_signatures
        and np.any(np.isfinite(right_signatures[token_id][right_columns]))
    ]
    if not active_left or not active_right:
        return {
            "schema_version": "token_signature_match_v1",
            "profile_type": profile_type,
            "value_field": value_field,
            "common_feature_names": common_names,
            "matched_count": 0,
            "mean_cosine": None,
            "fixed_alignment_bootstrap_ci_low": None,
            "fixed_alignment_bootstrap_ci_high": None,
            "matches": [],
            "unmatched_left_codes": active_left,
            "unmatched_right_codes": active_right,
            "thresholds": {
                "min_count": min_count,
                "min_subjects": min_subjects,
                "min_feature_overlap": min_feature_overlap,
            },
        }

    cosine = np.empty((len(active_left), len(active_right)), dtype=np.float64)
    overlap = np.empty_like(cosine, dtype=np.int64)
    for left_index, left_code in enumerate(active_left):
        left_vector = left_signatures[left_code][left_columns]
        for right_index, right_code in enumerate(active_right):
            right_vector = right_signatures[right_code][right_columns]
            cosine[left_index, right_index], overlap[left_index, right_index] = (
                _pairwise_cosine(
                    left_vector,
                    right_vector,
                    min_feature_overlap=min_feature_overlap,
                )
            )
    left_assignment, right_assignment = linear_sum_assignment(1.0 - cosine)
    matches = [
        {
            "left_code": int(active_left[left_index]),
            "right_code": int(active_right[right_index]),
            "cosine": float(cosine[left_index, right_index]),
            "feature_overlap": int(overlap[left_index, right_index]),
            "left_count": int(
                left_support[active_left[left_index]]["count"]
            ),
            "right_count": int(
                right_support[active_right[right_index]]["count"]
            ),
            "left_subject_count": int(
                left_support[active_left[left_index]]["subject_count"]
            ),
            "right_subject_count": int(
                right_support[active_right[right_index]]["subject_count"]
            ),
        }
        for left_index, right_index in zip(left_assignment, right_assignment)
    ]
    matched_left = {row["left_code"] for row in matches}
    matched_right = {row["right_code"] for row in matches}
    match_cosines = np.asarray([row["cosine"] for row in matches])
    ci_low: float | None = None
    ci_high: float | None = None
    if bootstrap_iterations:
        ci_low, ci_high = _bootstrap_mean_ci(
            match_cosines,
            iterations=bootstrap_iterations,
            confidence=0.95,
            seed_components=(seed, 112358),
        )
    return {
        "schema_version": "token_signature_match_v1",
        "profile_type": profile_type,
        "value_field": value_field,
        "common_feature_names": common_names,
        "matched_count": len(matches),
        "mean_cosine": float(np.mean(match_cosines)),
        "fixed_alignment_bootstrap_ci_low": ci_low,
        "fixed_alignment_bootstrap_ci_high": ci_high,
        "bootstrap_iterations": bootstrap_iterations,
        "matches": matches,
        "unmatched_left_codes": [
            code for code in active_left if code not in matched_left
        ],
        "unmatched_right_codes": [
            code for code in active_right if code not in matched_right
        ],
        "thresholds": {
            "min_count": min_count,
            "min_subjects": min_subjects,
            "min_feature_overlap": min_feature_overlap,
        },
    }


__all__ = [
    "TOKEN_PHYSIOLOGY_SCHEMA_VERSION",
    "TokenPhysiologyConfig",
    "TokenPhysiologyResult",
    "analyze_token_physiology",
    "match_token_signatures",
]
