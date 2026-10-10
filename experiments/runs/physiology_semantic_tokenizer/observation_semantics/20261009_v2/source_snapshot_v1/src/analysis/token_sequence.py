"""Standard, mask-aware sequence diagnostics for discrete physiology tokens.

The helpers in this module deliberately operate within recorded windows.  They
do not concatenate unrelated trials or overlapping windows, and they use
whole-window circular shifts for the default cross-modal null so local
autocorrelation is not destroyed by token-wise shuffling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SEQUENCE_ANALYSIS_SCHEMA = "physiology_token_sequence_v1"
EPS = 1e-12


@dataclass(frozen=True)
class SequenceSummary:
    """Compact sequence certificate with array artifacts kept separate."""

    schema: str
    token_count: int
    transition_count: int
    self_transition_fraction: float | None
    occupancy_entropy_nats: float | None
    transition_entropy_nats: float | None
    token_rows: tuple[Mapping[str, Any], ...]


def _validated_sequences(
    tokens: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    codebook_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tokens, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("tokens must have shape [samples,positions]")
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    if valid_mask is None:
        mask = np.ones(values.shape, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must match token shape")
    invalid_id = mask & ((values < 0) | (values >= codebook_size))
    if invalid_id.any():
        raise ValueError("valid token IDs must lie in [0, codebook_size)")
    return values, mask


def occupancy_counts(
    tokens: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    codebook_size: int,
) -> np.ndarray:
    values, mask = _validated_sequences(tokens, valid_mask, codebook_size=codebook_size)
    return np.bincount(values[mask], minlength=codebook_size).astype(np.int64)


def transition_counts(
    tokens: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    codebook_size: int,
    lag: int = 1,
) -> np.ndarray:
    """Count within-window transitions without crossing sample boundaries.

    ``lag`` is measured in token positions and must be a positive integer
    strictly smaller than the number of positions in each window.
    """

    values, mask = _validated_sequences(tokens, valid_mask, codebook_size=codebook_size)
    if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)):
        raise ValueError("lag must be an integer")
    lag = int(lag)
    if lag <= 0 or lag >= values.shape[1]:
        raise ValueError("lag must satisfy 0 < lag < positions")
    pair_mask = mask[:, :-lag] & mask[:, lag:]
    left = values[:, :-lag][pair_mask]
    right = values[:, lag:][pair_mask]
    flat = left * codebook_size + right
    return np.bincount(
        flat, minlength=codebook_size * codebook_size
    ).reshape(codebook_size, codebook_size).astype(np.int64)


def _entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    positive = values > 0
    return float(-np.sum(values[positive] * np.log(values[positive])))


def _run_lengths_by_token(
    tokens: np.ndarray,
    valid_mask: np.ndarray,
    *,
    codebook_size: int,
) -> list[list[int]]:
    runs: list[list[int]] = [[] for _ in range(codebook_size)]
    for row, row_mask in zip(tokens, valid_mask):
        current: int | None = None
        length = 0
        for value, valid in zip(row, row_mask):
            if not valid:
                if current is not None:
                    runs[current].append(length)
                current, length = None, 0
                continue
            value = int(value)
            if current == value:
                length += 1
            else:
                if current is not None:
                    runs[current].append(length)
                current, length = value, 1
        if current is not None:
            runs[current].append(length)
    return runs


def summarize_sequences(
    tokens: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    codebook_size: int,
) -> SequenceSummary:
    values, mask = _validated_sequences(tokens, valid_mask, codebook_size=codebook_size)
    occupancy = occupancy_counts(values, mask, codebook_size=codebook_size)
    if values.shape[1] >= 2:
        transitions = transition_counts(values, mask, codebook_size=codebook_size)
    else:
        transitions = np.zeros((codebook_size, codebook_size), dtype=np.int64)
    token_count = int(occupancy.sum())
    transition_count = int(transitions.sum())
    occupancy_probability = occupancy / max(token_count, 1)
    transition_rows = transitions.sum(axis=1)
    conditional = np.divide(
        transitions,
        transition_rows[:, None],
        out=np.zeros_like(transitions, dtype=np.float64),
        where=transition_rows[:, None] > 0,
    )
    row_entropies = np.asarray([_entropy(row) for row in conditional])
    transition_entropy = (
        float(np.sum(row_entropies * transition_rows) / transition_count)
        if transition_count
        else None
    )
    runs = _run_lengths_by_token(values, mask, codebook_size=codebook_size)
    transition_columns = transitions.sum(axis=0)
    token_rows: list[Mapping[str, Any]] = []
    for code in range(codebook_size):
        lengths = np.asarray(runs[code], dtype=np.int64)
        outgoing = transitions[code]
        outgoing_probability = (
            outgoing / transition_rows[code]
            if transition_rows[code]
            else np.zeros(codebook_size, dtype=np.float64)
        )
        token_rows.append(
            {
                "token_id": code,
                "occupancy": int(occupancy[code]),
                "transition_out_count": int(transition_rows[code]),
                "transition_in_count": int(transition_columns[code]),
                "self_transition_count": int(transitions[code, code]),
                "self_transition_fraction": (
                    float(transitions[code, code] / transition_rows[code])
                    if transition_rows[code]
                    else None
                ),
                "distinct_successor_count": int(np.count_nonzero(outgoing)),
                "distinct_predecessor_count": int(
                    np.count_nonzero(transitions[:, code])
                ),
                "transition_out_entropy_nats": (
                    _entropy(outgoing_probability) if transition_rows[code] else None
                ),
                "run_count": int(len(lengths)),
                "run_length_mean_tokens": float(lengths.mean()) if len(lengths) else None,
                "run_length_median_tokens": float(np.median(lengths)) if len(lengths) else None,
                "run_length_max_tokens": int(lengths.max()) if len(lengths) else None,
            }
        )
    return SequenceSummary(
        schema=SEQUENCE_ANALYSIS_SCHEMA,
        token_count=token_count,
        transition_count=transition_count,
        self_transition_fraction=(
            float(np.trace(transitions) / transition_count) if transition_count else None
        ),
        occupancy_entropy_nats=(
            _entropy(occupancy_probability) if token_count else None
        ),
        transition_entropy_nats=transition_entropy,
        token_rows=tuple(token_rows),
    )


def _validated_lags(lags: Iterable[int], *, positions: int) -> np.ndarray:
    """Return integer lags under the public EEG-to-fNIRS lag convention."""

    if positions <= 0:
        raise ValueError("token positions must be positive to analyze lags")
    raw_lags = tuple(lags)
    if not raw_lags:
        raise ValueError("at least one lag is required")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw_lags
    ):
        raise ValueError("lags must contain integers")
    lag_values = np.asarray(raw_lags, dtype=np.int64)
    if np.any(np.abs(lag_values) >= positions):
        raise ValueError(
            "each lag must satisfy abs(lag) < positions; positive lag pairs "
            "EEG[t] with fNIRS[t + lag]"
        )
    return lag_values


def cross_modal_lag_counts(
    eeg_tokens: np.ndarray,
    fnirs_tokens: np.ndarray,
    *,
    eeg_valid_mask: np.ndarray | None = None,
    fnirs_valid_mask: np.ndarray | None = None,
    lags: Iterable[int],
    eeg_codebook_size: int,
    fnirs_codebook_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Count aligned EEG/fNIRS token pairs at each requested lag.

    A positive lag pairs EEG at position ``t`` with fNIRS at
    ``t + lag``; a negative lag pairs EEG at ``t`` with an earlier fNIRS
    position.  Zero lag is allowed.  Every lag must be an integer satisfying
    ``abs(lag) < positions``.  Pairs never cross window boundaries and are
    counted only when both modality masks are valid.
    """

    eeg, eeg_mask = _validated_sequences(
        eeg_tokens, eeg_valid_mask, codebook_size=eeg_codebook_size
    )
    fnirs, fnirs_mask = _validated_sequences(
        fnirs_tokens, fnirs_valid_mask, codebook_size=fnirs_codebook_size
    )
    if eeg.shape != fnirs.shape:
        raise ValueError("EEG and fNIRS token windows must share shape")
    lag_values = _validated_lags(lags, positions=eeg.shape[1])
    counts = np.zeros(
        (len(lag_values), eeg_codebook_size, fnirs_codebook_size), dtype=np.int64
    )
    for index, lag in enumerate(lag_values):
        if lag >= 0:
            usable = eeg.shape[1] - lag
            left = eeg[:, :usable]
            right = fnirs[:, lag : lag + usable]
            mask = eeg_mask[:, :usable] & fnirs_mask[:, lag : lag + usable]
        else:
            offset = -lag
            usable = eeg.shape[1] - offset
            left = eeg[:, offset : offset + usable]
            right = fnirs[:, :usable]
            mask = eeg_mask[:, offset : offset + usable] & fnirs_mask[:, :usable]
        flat = left[mask] * fnirs_codebook_size + right[mask]
        counts[index] = np.bincount(
            flat, minlength=eeg_codebook_size * fnirs_codebook_size
        ).reshape(eeg_codebook_size, fnirs_codebook_size)
    return lag_values, counts


def coupling_metrics_from_counts(counts: np.ndarray) -> Mapping[str, float | int | None]:
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("counts must have shape [eeg_token,fnirs_token]")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("counts must be finite and non-negative")
    total = float(values.sum())
    possible_cells = int(values.size)
    active_eeg_tokens = int(np.count_nonzero(values.sum(axis=1)))
    active_fnirs_tokens = int(np.count_nonzero(values.sum(axis=0)))
    nonzero_values = values[values > 0]
    if total <= 0:
        return {
            "pair_count": 0,
            "nonzero_cells": 0,
            "possible_cells": possible_cells,
            "nonzero_cell_fraction": 0.0,
            "active_eeg_tokens": 0,
            "active_fnirs_tokens": 0,
            "singleton_cell_fraction": None,
            "median_nonzero_cell_count": None,
            "mean_pairs_per_cell": 0.0,
            "mutual_information_nats": None,
            "normalized_mutual_information": None,
            "conditional_entropy_fnirs_given_eeg_nats": None,
            "conditional_entropy_eeg_given_fnirs_nats": None,
        }
    joint = values / total
    p_eeg = joint.sum(axis=1)
    p_fnirs = joint.sum(axis=0)
    expected = p_eeg[:, None] * p_fnirs[None, :]
    positive = joint > 0
    mutual_information = float(
        np.sum(
            joint[positive]
            * np.log(joint[positive] / np.maximum(expected[positive], EPS))
        )
    )
    h_eeg, h_fnirs = _entropy(p_eeg), _entropy(p_fnirs)
    return {
        "pair_count": int(total),
        "nonzero_cells": int(np.count_nonzero(values)),
        "possible_cells": possible_cells,
        "nonzero_cell_fraction": float(np.count_nonzero(values) / possible_cells),
        "active_eeg_tokens": active_eeg_tokens,
        "active_fnirs_tokens": active_fnirs_tokens,
        "singleton_cell_fraction": float(np.mean(nonzero_values == 1)),
        "median_nonzero_cell_count": float(np.median(nonzero_values)),
        "mean_pairs_per_cell": float(total / values.size),
        "mutual_information_nats": mutual_information,
        "normalized_mutual_information": float(
            mutual_information / np.sqrt(max(h_eeg * h_fnirs, EPS))
        ),
        "conditional_entropy_fnirs_given_eeg_nats": float(
            max(h_fnirs - mutual_information, 0.0)
        ),
        "conditional_entropy_eeg_given_fnirs_nats": float(
            max(h_eeg - mutual_information, 0.0)
        ),
    }


def circular_shift_coupling_null(
    eeg_tokens: np.ndarray,
    fnirs_tokens: np.ndarray,
    *,
    eeg_valid_mask: np.ndarray | None,
    fnirs_valid_mask: np.ndarray | None,
    lags: Sequence[int],
    eeg_codebook_size: int,
    fnirs_codebook_size: int,
    permutations: int,
    seed: int,
) -> np.ndarray:
    """Null NMI from independently circular-shifting each EEG trial window.

    Tokens and their validity mask are rolled together, preserving each
    window's internal gaps and never moving values between trials.  A
    one-position window has no non-zero circular shift, so requested null
    samples are returned as ``NaN`` rather than treating the unchanged data as
    randomized.
    """

    if permutations < 0:
        raise ValueError("permutations must be non-negative")
    eeg, eeg_mask = _validated_sequences(
        eeg_tokens, eeg_valid_mask, codebook_size=eeg_codebook_size
    )
    fnirs, fnirs_mask = _validated_sequences(
        fnirs_tokens, fnirs_valid_mask, codebook_size=fnirs_codebook_size
    )
    if eeg.shape != fnirs.shape:
        raise ValueError("EEG and fNIRS token windows must share shape")
    lag_values = _validated_lags(lags, positions=eeg.shape[1])
    if permutations == 0:
        return np.empty((0, len(lag_values)), dtype=np.float64)
    if eeg.shape[1] == 1:
        return np.full((permutations, len(lag_values)), np.nan, dtype=np.float64)
    result = np.empty((permutations, len(lag_values)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    positions = eeg.shape[1]
    for iteration in range(permutations):
        shifts = rng.integers(1, positions, size=eeg.shape[0])
        shifted_tokens = np.empty_like(eeg)
        shifted_mask = np.empty_like(eeg_mask)
        for row, shift in enumerate(shifts):
            shifted_tokens[row] = np.roll(eeg[row], int(shift))
            shifted_mask[row] = np.roll(eeg_mask[row], int(shift))
        _, matrices = cross_modal_lag_counts(
            shifted_tokens,
            fnirs,
            eeg_valid_mask=shifted_mask,
            fnirs_valid_mask=fnirs_mask,
            lags=lag_values,
            eeg_codebook_size=eeg_codebook_size,
            fnirs_codebook_size=fnirs_codebook_size,
        )
        for lag_index, matrix in enumerate(matrices):
            metric = coupling_metrics_from_counts(matrix)["normalized_mutual_information"]
            result[iteration, lag_index] = np.nan if metric is None else float(metric)
    return result


def analyze_cross_modal_lags(
    eeg_tokens: np.ndarray,
    fnirs_tokens: np.ndarray,
    *,
    eeg_valid_mask: np.ndarray | None = None,
    fnirs_valid_mask: np.ndarray | None = None,
    lags: Sequence[int] = (-2, -1, 0, 1, 2),
    eeg_codebook_size: int,
    fnirs_codebook_size: int,
    permutations: int = 0,
    seed: int = 0,
    patch_duration_s: float = 2.0,
) -> tuple[list[Mapping[str, Any]], np.ndarray, np.ndarray]:
    if not np.isfinite(patch_duration_s) or patch_duration_s <= 0:
        raise ValueError("patch_duration_s must be positive")
    lag_values, matrices = cross_modal_lag_counts(
        eeg_tokens,
        fnirs_tokens,
        eeg_valid_mask=eeg_valid_mask,
        fnirs_valid_mask=fnirs_valid_mask,
        lags=lags,
        eeg_codebook_size=eeg_codebook_size,
        fnirs_codebook_size=fnirs_codebook_size,
    )
    null = circular_shift_coupling_null(
        eeg_tokens,
        fnirs_tokens,
        eeg_valid_mask=eeg_valid_mask,
        fnirs_valid_mask=fnirs_valid_mask,
        lags=lags,
        eeg_codebook_size=eeg_codebook_size,
        fnirs_codebook_size=fnirs_codebook_size,
        permutations=permutations,
        seed=seed,
    )
    rows: list[Mapping[str, Any]] = []
    positions = np.asarray(eeg_tokens).shape[1]
    null_degenerate_reason = (
        "fewer than 2 token positions; no non-zero circular shift"
        if permutations and positions < 2
        else None
    )
    for index, (lag, matrix) in enumerate(zip(lag_values, matrices)):
        metrics = dict(coupling_metrics_from_counts(matrix))
        observed = metrics["normalized_mutual_information"]
        if permutations and observed is not None:
            finite_null = null[:, index][np.isfinite(null[:, index])]
            null_mean = float(np.mean(finite_null)) if len(finite_null) else None
            empirical_p = (
                float((1 + np.sum(finite_null >= float(observed))) / (len(finite_null) + 1))
                if len(finite_null)
                else None
            )
        else:
            null_mean, empirical_p = None, None
        rows.append(
            {
                "lag_tokens": int(lag),
                "lag_seconds": float(lag * patch_duration_s),
                **metrics,
                "null_policy": "within-window whole-sequence circular shift",
                "null_shift_unit": "trial window (tokens and validity mask together)",
                "null_permutations": int(permutations),
                "null_finite_permutations": int(
                    np.count_nonzero(np.isfinite(null[:, index]))
                ),
                "null_degenerate_reason": null_degenerate_reason,
                "null_nmi_mean": null_mean,
                "nmi_above_null": (
                    float(observed) - null_mean
                    if observed is not None and null_mean is not None
                    else None
                ),
                "nmi_empirical_p": empirical_p,
            }
        )
    return rows, matrices, null


def markov_log_loss(
    train_tokens: np.ndarray,
    validation_tokens: np.ndarray,
    *,
    train_valid_mask: np.ndarray | None = None,
    validation_valid_mask: np.ndarray | None = None,
    codebook_size: int,
    alpha: float = 0.5,
) -> Mapping[str, float | int]:
    """Compare smoothed order-0 and order-1 validation log loss."""

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    train, train_mask = _validated_sequences(
        train_tokens, train_valid_mask, codebook_size=codebook_size
    )
    validation, validation_mask = _validated_sequences(
        validation_tokens, validation_valid_mask, codebook_size=codebook_size
    )
    unigram = occupancy_counts(train, train_mask, codebook_size=codebook_size).astype(np.float64)
    unigram_probability = (unigram + alpha) / (unigram.sum() + alpha * codebook_size)
    if train.shape[1] >= 2:
        transitions = transition_counts(
            train, train_mask, codebook_size=codebook_size
        ).astype(np.float64)
    else:
        transitions = np.zeros(
            (codebook_size, codebook_size), dtype=np.float64
        )
    transition_probability = (transitions + alpha * unigram_probability[None, :])
    transition_probability /= np.maximum(
        transition_probability.sum(axis=1, keepdims=True), EPS
    )
    pair_mask = validation_mask[:, :-1] & validation_mask[:, 1:]
    left = validation[:, :-1][pair_mask]
    right = validation[:, 1:][pair_mask]
    if not len(right):
        return {
            "validation_transition_count": 0,
            "order0_log_loss_nats": float("nan"),
            "order1_log_loss_nats": float("nan"),
            "order1_gain_nats": float("nan"),
        }
    order0 = float(-np.mean(np.log(np.maximum(unigram_probability[right], EPS))))
    order1 = float(
        -np.mean(np.log(np.maximum(transition_probability[left, right], EPS)))
    )
    return {
        "validation_transition_count": int(len(right)),
        "order0_log_loss_nats": order0,
        "order1_log_loss_nats": order1,
        "order1_gain_nats": order0 - order1,
    }


__all__ = [
    "SEQUENCE_ANALYSIS_SCHEMA",
    "SequenceSummary",
    "analyze_cross_modal_lags",
    "circular_shift_coupling_null",
    "coupling_metrics_from_counts",
    "cross_modal_lag_counts",
    "markov_log_loss",
    "occupancy_counts",
    "summarize_sequences",
    "transition_counts",
]
