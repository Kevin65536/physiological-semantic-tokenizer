"""Frozen Tak mismatch summaries; records and subjects retain their own roles.

This module reads no data and performs no fitting.  Its input is one row per
planned fixed-parameter record/pair/window, including unavailable endpoints.
The main summaries preserve the evaluator's equal-record estimand.  A native
continuous support is *not* promoted to an independent record or subject.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata


SCHEMA = "tak_mismatch_fixed_analysis_v1"
ROOT_METRICS = frozenset(("D1", "D_EH", "D_drive_H", "D_obs", "D4", "D_J_E", "D_J_O", "D_J_R"))
DEFAULT_METRICS = ("D1", "D_EH", "D_drive_H", "D_obs", "Delta_spatial", "D4")
PRIMARY_METRICS = ("D1", "D_EH", "D_drive_H")
HB_RATIO = "Hb_residual_to_zero_MSE_ratio"
METRIC_DEFINITIONS = {
    "D1": dict(direction="lower", unit="training Hb SD", description="Hb-only standardized Hb residual RMS"),
    "D_EH": dict(direction="lower", unit="training EEG SD", description="Hb-only predicted EEG minus observed EEG proxy RMS"),
    "D_drive_H": dict(direction="lower", unit="training Hb SD", description="EEG-only and Hb-only drive Hb replay difference RMS"),
    "D_obs": dict(direction="lower", unit="training Hb SD", description="local Hb-readout effect of the observable drive difference under the frozen synthetic-noise scenario"),
    "Delta_spatial": dict(direction="positive_favors_selected", unit="training Hb SD squared", description="nearest-neighborhood minus selected-neighborhood independent EEG-to-Hb MSE"),
    "D4": dict(direction="lower", unit="training component SD", description="joint EEG/Hb standardized residual RMS"),
    HB_RATIO: dict(direction="lower", unit="ratio", description="matched-support Hb residual MSE divided by zero-prediction MSE; can exceed one; not explained variance or an energy decomposition"),
}

SPATIAL_METRICS = ("D1", HB_RATIO, "D_bal_10s", "D_bal_20s", "D_bal_40s",
                   "D_EH", "D_drive_H", "D_obs", "D4", "D_J_E", "D_J_O", "D_J_R", "Delta_spatial")
SPATIAL_HB_METRICS = frozenset(("D1", HB_RATIO, "D_bal_10s", "D_bal_20s", "D_bal_40s"))

PHYSIOLOGY_FEATURE_DEFINITIONS = {
    'Hb_bias_fraction': dict(label='Hb残差偏置平方比', unit='0–1', description='先窗口内计算两色团残差均值平方与总均方；记录、被试等权汇总后相除。不是生理来源占比。'),
    'Hb_centered_residual_RMS': dict(label='Hb去均值残差RMS', unit='训练Hb SD', description='去除每窗残差均值后的均方，先记录、再被试等权汇总并开根。'),
    'Hb_residual_OR_correlation': dict(label='HbO/HbR残差相关', unit='Pearson相关', description='各窗去均值的协方差及方差先记录、再被试等权，再构成相关系数。'),
    'Hb_common_direction_fraction': dict(label='标准化共同方向平方比', unit='0–1', description='c=(rO+rR)/sqrt(2), d=(rO-rR)/sqrt(2)；记录等权的mean(c²)/(mean(c²)+mean(d²))。r用各色团训练SD标准化；不是HbT、血容量或生理来源份额。'),
    'EEG_Hb_readout_correlation': dict(label='EEG与Hb解读出形态相关', unit='平均窗口Pearson相关', description='同一评分支持上EEG代理与Hb-only EEG读出的去均值相关；先窗口对、记录、被试等权。'),
    'drive_E_H_correlation': dict(label='两独立驱动形态相关', unit='平均窗口Pearson相关', description='EEG-only与Hb-only驱动在同一评分支持去均值后的相关；不同于经过血流重放的D_drive_H。'),
}
PHYSIOLOGY_FEATURE_DEFINITIONS['Hb_bias_fraction']['latex'] = r'F_{\mathrm{bias}}=\frac{\langle(\bar r_O^2+\bar r_R^2)/2\rangle}{\langle\operatorname{mean}_t[(r_O^2+r_R^2)/2]\rangle}'
PHYSIOLOGY_FEATURE_DEFINITIONS['Hb_centered_residual_RMS']['latex'] = r'R_{\mathrm{centered}}=\sqrt{\langle\operatorname{mean}_{t,c}[(r_c(t)-\bar r_c)^2]\rangle}'
PHYSIOLOGY_FEATURE_DEFINITIONS['Hb_residual_OR_correlation']['latex'] = r'\rho_{OR}=\frac{\langle\operatorname{cov}_t(r_O,r_R)\rangle}{\sqrt{\langle\operatorname{var}_t(r_O)\rangle\langle\operatorname{var}_t(r_R)\rangle}}'
PHYSIOLOGY_FEATURE_DEFINITIONS['Hb_common_direction_fraction']['latex'] = r'c=(r_O+r_R)/\sqrt{2},\quad d=(r_O-r_R)/\sqrt{2},\quad F_c=\frac{\langle\operatorname{mean}_t(c^2)\rangle}{\langle\operatorname{mean}_t(c^2+d^2)\rangle}'

_COUNTS = ("planned_windows", "available_windows", "missing_windows", "planned_records", "available_records")
_VALUES = ("metric", "value", "aggregation_numerator", "aggregation_denominator", *_COUNTS)
_SUBJECT_TASK = ("dataset", "subject", "task_label")
_META = ("record_unit", "repeat_id", "repeat_kind", "record_order")
_COMPARISON_COLUMNS = ("dataset", "metric", "task_a", "task_b", "planned_a", "planned_b", "available_a", "available_b", "paired_subjects", "missing_a", "missing_b", "mean_difference", "median_difference", "positive_fraction", "cohen_dz", "ci_low", "ci_high", "p_value", "q_value", "fdr_family", "fdr_family_size", "permutation_method", "permutations", "status")
_STABILITY_SUMMARY = ("dataset", "task_label", "repeat_kind", "metric", "planned_subjects", "subjects", "missing_subjects", "spearman_rho", "rho_ci_low", "rho_ci_high", "median_signed_difference", "median_absolute_difference", "status")


def _checked_windows(frame: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    required = {*_SUBJECT_TASK, "record_id", "native_support_id", "pair", "window_start_s"}
    columns = set(metrics) - {HB_RATIO}
    if HB_RATIO in metrics:
        columns |= {"D1", "zero_NMSE"}
    missing = (required | columns) - set(frame.columns)
    if missing:
        raise ValueError(f"missing fixed-analysis columns: {sorted(missing)}")
    data = frame.copy()
    if "parameter_mode" in data and not data["parameter_mode"].eq("fixed").all():
        raise ValueError("fixed analysis requires fixed rows only")
    identities = list(required - {"window_start_s"})
    if data[identities].isna().any().any():
        raise ValueError("fixed-analysis identities must not be missing")
    for column in identities:
        data[column] = data[column].astype(str)
    times = pd.to_numeric(data["window_start_s"], errors="coerce")
    if not np.isfinite(times).all():
        raise ValueError("window_start_s must be finite")
    data["window_start_s"] = times
    identity = ["dataset", "subject", "record_id", "pair", "window_start_s"]
    if data.duplicated(identity).any():
        raise ValueError("duplicate fixed record/pair/window identity")
    for metric in columns:
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        if metric in ROOT_METRICS and (data[metric].dropna() < 0).any():
            raise ValueError(f"{metric} must be nonnegative")
    # These are acquisition metadata, not alternative aggregation identities.
    for column in _META:
        if column in data:
            counts = data.groupby(["dataset", "subject", "record_id"], dropna=False)[column].nunique(dropna=False)
            if (counts > 1).any():
                raise ValueError(f"{column} must be constant within record_id")
    return data


def _parts(data: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if metric == HB_RATIO:
        value = data["D1"].to_numpy(float)
        denominator = data["zero_NMSE"].to_numpy(float)
        valid = np.isfinite(value) & np.isfinite(denominator) & (denominator > 0)
        return value * value, denominator, valid
    value = data[metric].to_numpy(float)
    return value * value if metric in ROOT_METRICS else value, np.ones(len(value)), np.isfinite(value)


def _value(numerator: float, denominator: float, metric: str) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    value = numerator / denominator
    return float(np.sqrt(value)) if metric in ROOT_METRICS else float(value)


def _window_cells(data: pd.DataFrame, keys: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    for identity, group in data.groupby(list(keys), sort=True, dropna=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        base = dict(zip(keys, identity))
        for column in _META:
            if column in group and column not in base:
                base[column] = group[column].iloc[0]
        base["native_supports"] = int(group["native_support_id"].nunique())
        for metric in metrics:
            numerator, denominator, valid = _parts(group, metric)
            n = int(valid.sum())
            num = float(np.mean(numerator[valid])) if n else np.nan
            den = float(np.mean(denominator[valid])) if n else np.nan
            rows.append(dict(base, metric=metric, value=_value(num, den, metric),
                aggregation_numerator=num, aggregation_denominator=den,
                planned_windows=len(group), available_windows=n, missing_windows=len(group)-n,
                planned_records=1, available_records=int(n > 0)))
    return pd.DataFrame(rows, columns=[*keys, *(c for c in _META if c not in keys and c in data), "native_supports", *_VALUES])


def _record_equal(records: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    rows = []
    for identity, group in records.groupby([*keys, "metric"], sort=True, dropna=False):
        base = dict(zip([*keys, "metric"], identity))
        good = np.isfinite(group["value"].to_numpy(float))
        num = float(group.loc[good, "aggregation_numerator"].mean()) if good.any() else np.nan
        den = float(group.loc[good, "aggregation_denominator"].mean()) if good.any() else np.nan
        rows.append(dict(base, value=_value(num, den, base["metric"]),
            aggregation_numerator=num, aggregation_denominator=den,
            planned_windows=int(group.planned_windows.sum()), available_windows=int(group.available_windows.sum()),
            missing_windows=int(group.missing_windows.sum()), planned_records=len(group), available_records=int(good.sum())))
    return pd.DataFrame(rows, columns=[*keys, *_VALUES])


def aggregate_fixed_metrics(frame: pd.DataFrame, *, metrics: Sequence[str] = DEFAULT_METRICS,
                            include_hb_ratio: bool = True) -> dict[str, pd.DataFrame]:
    """Window/pairs -> equal original records -> subjects, preserving failures.

    ``native_pairs`` and ``native_records`` use the original ``record_id``.
    ``native_support_id`` counts continuous supports but does not define the
    independent record.  ROOT metrics are squared before averaging and rooted
    after averaging.  The Hb ratio averages its matched-support numerator and
    denominator separately, and divides only after the final aggregation.
    ``fixed_success`` is not a cross-endpoint gate: each retained finite metric
    uses its own endpoint's availability, as in the evaluator.
    """
    metrics = tuple(dict.fromkeys((*metrics, *((HB_RATIO,) if include_hb_ratio else ()))))
    data = _checked_windows(frame, metrics)
    pair_keys = (*_SUBJECT_TASK, "record_id", "pair")
    record_keys = (*_SUBJECT_TASK, "record_id")
    pairs = _window_cells(data, pair_keys, metrics)
    records = _window_cells(data, record_keys, metrics)
    # A record can contain several tasks.  The subject's overall value must
    # pool its windows before the equal-record step, rather than count that
    # record once for every task it happened to contain.
    pooled = _window_cells(data, ("dataset", "subject", "record_id"), metrics)
    return dict(native_pairs=pairs, native_records=records,
        subject_tasks=_record_equal(records, _SUBJECT_TASK),
        subjects=_record_equal(pooled, ("dataset", "subject")))


def _bootstrap(values: np.ndarray, statistic, samples: int, rng: np.random.Generator) -> np.ndarray:
    result = np.empty(samples)
    for start in range(0, samples, 256):
        stop = min(start+256, samples)
        draws = values[rng.integers(len(values), size=(stop-start, len(values)))]
        result[start:stop] = statistic(draws)
    return result


def summarize_subject_distributions(subjects: pd.DataFrame, *, by: Sequence[str] = ("dataset",),
                                    bootstrap_samples: int = 5000, seed: int = 17) -> pd.DataFrame:
    """Subject distributions and whole-subject percentile intervals, frozen fits.

    The equal-subject center uses RMS for ROOT metrics to preserve the owning
    evaluator's estimator.  Quantiles describe individual subject values.
    Intervals with fewer than six available subjects remain unavailable.
    """
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    rows = []
    for identity, group in subjects.groupby([*by, "metric"], sort=True, dropna=False):
        base = dict(zip([*by, "metric"], identity))
        good = group[np.isfinite(group.value.to_numpy(float))]
        values = good.value.to_numpy(float)
        metric = base["metric"]
        num = good.aggregation_numerator.to_numpy(float)
        den = good.aggregation_denominator.to_numpy(float)
        center = _value(float(num.mean()), float(den.mean()), metric) if len(good) else np.nan
        lo = hi = np.nan
        if len(good) >= 6:
            indices = np.arange(len(good))
            def statistic(draws):
                picked = draws.astype(int)
                ratio = np.mean(num[picked], axis=1) / np.mean(den[picked], axis=1)
                return np.sqrt(ratio) if metric in ROOT_METRICS else ratio
            draws = _bootstrap(indices, statistic, bootstrap_samples, rng)
            lo, hi = np.quantile(draws, [.025, .975])
        quantiles = np.quantile(values, [.1, .25, .5, .75, .9]) if len(values) else np.full(5, np.nan)
        rows.append(dict(base, planned_subjects=len(group), available_subjects=len(good),
            missing_subjects=len(group)-len(good), subject_equal_value=center,
            **dict(zip(("p10", "q25", "median", "q75", "p90"), quantiles)),
            ci_low=lo, ci_high=hi, planned_windows=int(group.planned_windows.sum()),
            available_windows=int(group.available_windows.sum()), planned_records=int(group.planned_records.sum()),
            available_records=int(group.available_records.sum()),
            status="available" if len(good) >= 6 else "insufficient_subjects"))
    columns = [*by, "metric", "planned_subjects", "available_subjects", "missing_subjects", "subject_equal_value", "p10", "q25", "median", "q75", "p90", "ci_low", "ci_high", "planned_windows", "available_windows", "planned_records", "available_records", "status"]
    return pd.DataFrame(rows, columns=columns)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """BH adjusted values among finite tests; retain unavailable entries as NaN."""
    values = np.asarray(p_values, float)
    if values.ndim != 1 or ((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1)).any():
        raise ValueError("p values must be a one-dimensional sequence in [0,1]")
    output = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid], kind="stable")]
    if len(order):
        adjusted = values[order] * len(order) / np.arange(1, len(order)+1)
        output[order] = np.minimum(1., np.minimum.accumulate(adjusted[::-1])[::-1])
    return output


def _sign_flip(difference: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, str, int]:
    n = len(difference)
    observed = abs(float(np.mean(difference)))
    tolerance = np.finfo(float).eps * max(float(np.max(abs(difference))), observed, np.finfo(float).tiny) * n * 8
    exact = n <= 16
    total = 2**n if exact else samples
    exceed = 0
    for start in range(0, total, 4096):
        count = min(4096, total-start)
        if exact:
            bits = (np.arange(start, start+count, dtype=np.uint64)[:, None] >> np.arange(n, dtype=np.uint64)) & 1
            signs = bits.astype(float)*2-1
        else:
            signs = rng.integers(0, 2, (count, n))*2-1
        statistics = abs(signs @ difference / n)
        exceed += int(np.count_nonzero(statistics >= observed-tolerance))
    return ((exceed/total if exact else (exceed+1)/(total+1)),
        "exact_sign_flip" if exact else "monte_carlo_sign_flip_plus_one", total)


def paired_task_comparisons(subject_tasks: pd.DataFrame, *, bootstrap_samples: int = 5000,
                            permutation_samples: int = 100000, seed: int = 17,
                            primary_metrics: Sequence[str] = PRIMARY_METRICS) -> pd.DataFrame:
    """Exploratory within-dataset, paired-subject task differences, B minus A.

    No inference is produced below six paired subjects.  The primary BH family
    contains every testable dataset/task-pair/primary-endpoint row in this call.
    Other metrics form one separately labelled diagnostic BH family.  No task
    difference constitutes a randomized intervention or a general emotion effect.
    """
    if bootstrap_samples < 1 or permutation_samples < 1:
        raise ValueError("bootstrap and permutation samples must be positive")
    keys = ["dataset", "subject", "task_label", "metric"]
    if subject_tasks.duplicated(keys).any():
        raise ValueError("subject/task/metric summaries must be unique")
    rng = np.random.default_rng(seed)
    rows = []
    for (dataset, metric), group in subject_tasks.groupby(["dataset", "metric"], sort=True):
        tasks = sorted(group.task_label.unique())
        for task_a, task_b in combinations(tasks, 2):
            a = group[group.task_label == task_a].set_index("subject").value
            b = group[group.task_label == task_b].set_index("subject").value
            available_a = a[np.isfinite(a.to_numpy(float))]
            available_b = b[np.isfinite(b.to_numpy(float))]
            common = sorted(set(available_a.index) & set(available_b.index))
            d = available_b.loc[common].to_numpy(float)-available_a.loc[common].to_numpy(float)
            n = len(d)
            mean = float(d.mean()) if n else np.nan
            standard_deviation = float(d.std(ddof=1)) if n >= 2 else np.nan
            row = dict(dataset=dataset, metric=metric, task_a=task_a, task_b=task_b,
                planned_a=len(a), planned_b=len(b), available_a=len(available_a), available_b=len(available_b),
                paired_subjects=n, missing_a=len(a)-len(available_a), missing_b=len(b)-len(available_b),
                mean_difference=mean, median_difference=float(np.median(d)) if n else np.nan,
                positive_fraction=float(np.mean(d > 0)) if n else np.nan,
                cohen_dz=mean/standard_deviation if standard_deviation > 0 else np.nan,
                ci_low=np.nan, ci_high=np.nan, p_value=np.nan, q_value=np.nan,
                fdr_family="primary" if metric in primary_metrics else "diagnostic",
                fdr_family_size=0, permutation_method="unavailable", permutations=0,
                status="available" if n >= 6 else "insufficient_paired_subjects")
            if n >= 6:
                boot = _bootstrap(d, lambda x: x.mean(axis=1), bootstrap_samples, rng)
                row["ci_low"], row["ci_high"] = np.quantile(boot, [.025, .975])
                row["p_value"], row["permutation_method"], row["permutations"] = _sign_flip(d, permutation_samples, rng)
            rows.append(row)
    result = pd.DataFrame(rows, columns=_COMPARISON_COLUMNS)
    for family in ("primary", "diagnostic"):
        selector = result.fdr_family.eq(family)
        values = result.loc[selector, "p_value"].to_numpy(float)
        result.loc[selector, "q_value"] = benjamini_hochberg(values)
        result.loc[selector, "fdr_family_size"] = int(np.isfinite(values).sum())
    return result


def _rho(first: np.ndarray, second: np.ndarray) -> float:
    a, b = rankdata(first), rankdata(second)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _rho_batch(draws: np.ndarray) -> np.ndarray:
    first, second = rankdata(draws[:, :, 0], axis=1), rankdata(draws[:, :, 1], axis=1)
    first -= first.mean(axis=1, keepdims=True)
    second -= second.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(first*first, axis=1)*np.sum(second*second, axis=1))
    result = np.full(len(draws), np.nan)
    np.divide(np.sum(first*second, axis=1), denominator, out=result, where=denominator > 0)
    return result


def summarize_stability(pairs: pd.DataFrame, *, bootstrap_samples: int = 5000,
                         seed: int = 17) -> pd.DataFrame:
    """Spearman and paired changes; only whole subjects are bootstrap units."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    rows = []
    keys = ["dataset", "task_label", "repeat_kind", "metric"]
    for identity, group in pairs.groupby(keys, sort=True, dropna=False):
        good = np.isfinite(group.value_a.to_numpy(float)) & np.isfinite(group.value_b.to_numpy(float))
        first = group.loc[good, "value_a"].to_numpy(float)
        second = group.loc[good, "value_b"].to_numpy(float)
        n = len(first)
        rho = _rho(first, second)
        lo = hi = np.nan
        if n >= 6 and np.isfinite(rho):
            values = np.column_stack((first, second))
            draws = _bootstrap(values, _rho_batch, bootstrap_samples, rng)
            finite = draws[np.isfinite(draws)]
            if len(finite):
                lo, hi = np.quantile(finite, [.025, .975])
        rows.append(dict(zip(keys, identity), planned_subjects=len(group), subjects=n,
            missing_subjects=len(group)-n, spearman_rho=rho, rho_ci_low=lo, rho_ci_high=hi,
            median_signed_difference=float(np.median(second-first)) if n else np.nan,
            median_absolute_difference=float(np.median(abs(second-first))) if n else np.nan,
            status=("insufficient_subjects" if n < 6 else "constant_values" if not np.isfinite(rho) else "available")))
    return pd.DataFrame(rows, columns=_STABILITY_SUMMARY)


def _paired_value(first: pd.DataFrame, second: pd.DataFrame, metric: str) -> tuple[float, float, int, int]:
    """Use only the same available physical pairs at both endpoints."""
    planned = set(first.pair) & set(second.pair)
    first = first[first.pair.isin(planned)].set_index("pair")
    second = second[second.pair.isin(planned)].set_index("pair")
    common = [pair for pair in sorted(planned)
              if np.isfinite(first.loc[pair, "value"]) and np.isfinite(second.loc[pair, "value"])]
    def value(data):
        if not common:
            return np.nan
        good = data.loc[common]
        return _value(float(good.aggregation_numerator.mean()), float(good.aggregation_denominator.mean()), metric)
    return value(first), value(second), len(planned), len(common)


def native_record_repeatability(native_pairs: pd.DataFrame, *, bootstrap_samples: int = 5000,
                                 seed: int = 17) -> dict[str, pd.DataFrame]:
    """Earliest/latest declared repeats of the same subject/task and channels.

    ``repeat_id`` identifies an acquisition unit; simultaneous Probe branches
    must share it. ``record_order`` must come from metadata, not guessed string
    ordering. ``repeat_kind`` states the scope (e.g. same-visit cross-Part).
    Missing metadata, fewer than two units, and failed chosen endpoints remain
    explicit; successful intermediate units never replace chosen endpoints.
    """
    rows = []
    keys = [*_SUBJECT_TASK, "metric"]
    for identity, group in native_pairs.groupby(keys, sort=True, dropna=False):
        base = dict(zip(keys, identity))
        row = dict(base, repeat_kind="unavailable", repeat_a=None, repeat_b=None,
            value_a=np.nan, value_b=np.nan, planned_pairs=0, available_pairs=0, missing_pairs=0,
            available_pair_windows_a=0, available_pair_windows_b=0,
            status="unavailable_repeat_metadata")
        if not all(column in group for column in ("repeat_id", "repeat_kind", "record_order")):
            rows.append(row); continue
        group = group.dropna(subset=["repeat_id", "repeat_kind", "record_order"])
        if group.empty:
            rows.append(row); continue
        kinds = group.repeat_kind.astype(str).unique()
        if len(kinds) != 1:
            raise ValueError("each subject/task must have one repeat_kind")
        row["repeat_kind"] = kinds[0]
        order = group.groupby("repeat_id", sort=False).record_order.agg(["min", "max"])
        if (order["min"] != order["max"]).any():
            raise ValueError("record_order must be constant within repeat_id")
        order = order.sort_values("min", kind="stable")
        if len(order) < 2:
            row["status"] = "unavailable_independent_repeat"; rows.append(row); continue
        if order["min"].duplicated().any():
            row["status"] = "unavailable_repeat_order"; rows.append(row); continue
        a, b = order.index[0], order.index[-1]
        row.update(repeat_a=a, repeat_b=b)
        # A source may carry synchronous branches; the same physical pair must
        # not appear twice in one acquisition unit.
        selected = group[group.repeat_id.isin((a, b))]
        if selected.duplicated(["repeat_id", "pair"]).any():
            raise ValueError("duplicate physical pair within declared repeat_id")
        first, second = selected[selected.repeat_id == a], selected[selected.repeat_id == b]
        value_a, value_b, planned, available = _paired_value(first, second, base["metric"])
        common = set(first[np.isfinite(first.value)].pair) & set(second[np.isfinite(second.value)].pair)
        row.update(value_a=value_a, value_b=value_b, planned_pairs=planned, available_pairs=available, missing_pairs=planned-available,
            available_pair_windows_a=int(first[first.pair.isin(common)].available_windows.sum()),
            available_pair_windows_b=int(second[second.pair.isin(common)].available_windows.sum()),
            status="available" if available else "unavailable_common_pairs")
        rows.append(row)
    columns = [*keys, "repeat_kind", "repeat_a", "repeat_b", "value_a", "value_b", "planned_pairs", "available_pairs", "missing_pairs", "available_pair_windows_a", "available_pair_windows_b", "status"]
    pairs = pd.DataFrame(rows, columns=columns)
    return dict(pairs=pairs, summary=summarize_stability(pairs, bootstrap_samples=bootstrap_samples, seed=seed))


def within_record_consistency(frame: pd.DataFrame, *, metrics: Sequence[str] = DEFAULT_METRICS,
                              include_hb_ratio: bool = True, bootstrap_samples: int = 5000,
                              seed: int = 17) -> dict[str, pd.DataFrame]:
    """Early/late nonoverlapping window halves; this is temporal consistency.

    Halves are formed from declared original-record times within task.  Both
    halves need two distinct windows; an odd central window is dropped.  The
    same physical channel pairs must be available in both halves.  Original
    records are then equally weighted within subject/task.  No new fitting or
    independent test/retest claim is involved.
    """
    metrics = tuple(dict.fromkeys((*metrics, *((HB_RATIO,) if include_hb_ratio else ()))))
    data = _checked_windows(frame, metrics)
    record_rows = []
    for identity, group in data.groupby([*_SUBJECT_TASK, "record_id"], sort=True):
        times = np.sort(group.window_start_s.unique())
        half = len(times)//2
        for metric in metrics:
            base = dict(zip([*_SUBJECT_TASK, "record_id"], identity), metric=metric,
                planned_windows=len(times), windows_per_half=half, dropped_middle_windows=len(times)%2,
                value_a=np.nan, value_b=np.nan, numerator_a=np.nan, numerator_b=np.nan,
                denominator_a=np.nan, denominator_b=np.nan, planned_pairs=0, available_pairs=0,
                available_windows_a=0, available_windows_b=0,
                status="insufficient_nonoverlapping_windows")
            if half < 2:
                record_rows.append(base); continue
            first = group[group.window_start_s.isin(times[:half])]
            second = group[group.window_start_s.isin(times[-half:])]
            a = _window_cells(first, ("pair",), (metric,))
            b = _window_cells(second, ("pair",), (metric,))
            value_a, value_b, planned, available = _paired_value(a, b, metric)
            common = set(a[np.isfinite(a.value)].pair) & set(b[np.isfinite(b.value)].pair)
            base.update(value_a=value_a, value_b=value_b, planned_pairs=planned, available_pairs=available,
                status="available" if available else "unavailable_common_pairs")
            if available:
                # Pair-window equal within each half, as in the original owner.
                for label, source in (("a", first), ("b", second)):
                    selected = source[source.pair.isin(common)]
                    numerator, denominator, valid = _parts(selected, metric)
                    base["available_windows_"+label] = int(selected.loc[valid, "window_start_s"].nunique())
                    base["numerator_"+label] = float(numerator[valid].mean())
                    base["denominator_"+label] = float(denominator[valid].mean())
                    base["value_"+label] = _value(base["numerator_"+label], base["denominator_"+label], metric)
                if min(base["available_windows_a"], base["available_windows_b"]) < 2:
                    base["status"] = "insufficient_available_half_windows"
                    for label in ("a", "b"):
                        for field in ("value_", "numerator_", "denominator_"):
                            base[field+label] = np.nan
            record_rows.append(base)
    record_columns = [*_SUBJECT_TASK, "record_id", "metric", "planned_windows", "windows_per_half", "dropped_middle_windows", "value_a", "value_b", "numerator_a", "numerator_b", "denominator_a", "denominator_b", "planned_pairs", "available_pairs", "available_windows_a", "available_windows_b", "status"]
    records = pd.DataFrame(record_rows, columns=record_columns)
    rows = []
    for identity, group in records.groupby([*_SUBJECT_TASK, "metric"], sort=True):
        good = group.status.eq("available")
        base = dict(zip([*_SUBJECT_TASK, "metric"], identity), repeat_kind="within_record_early_late",
            planned_records=len(group), available_records=int(good.sum()), value_a=np.nan, value_b=np.nan,
            status="available" if good.any() else "unavailable_complete_halves")
        if good.any():
            for label in ("a", "b"):
                base["value_"+label] = _value(float(group.loc[good, "numerator_"+label].mean()),
                    float(group.loc[good, "denominator_"+label].mean()), base["metric"])
        rows.append(base)
    pairs = pd.DataFrame(rows, columns=[*_SUBJECT_TASK, "metric", "repeat_kind", "planned_records", "available_records", "value_a", "value_b", "status"])
    return dict(records=records, pairs=pairs,
        summary=summarize_stability(pairs, bootstrap_samples=bootstrap_samples, seed=seed))


def _finite_pearson(first, second):
    a, b = np.asarray(first, float), np.asarray(second, float)
    valid = np.isfinite(a)&np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def fixed_window_physiology_features(arrays, row, *, ocular=None, counterfactual=None):
    """Describe retained fixed fits and EOG sensitivity; never fit or read data.

    The EOG lag comparison uses the same 149 central scoring bins for all
    0/-5/+5 s alignments. Residual component moments are retained so that the
    later subject summaries divide averaged squared terms, not window ratios.
    """
    result = dict(feature_status='unavailable_arrays')
    if not arrays or 'target' not in arrays:
        return result
    y = np.asarray(arrays['target'], float); sd = np.asarray(arrays['sd'], float)
    mask = np.asarray(arrays['score_mask'], bool)
    if y.shape != (300, 3) or mask.shape != y.shape or sd.shape != (3,) or not np.all(sd > 0):
        raise ValueError('retained fixed physiology arrays must preserve the 300x3 training coordinate')
    time = np.asarray(arrays.get('time_s', np.arange(300)*.1))
    if not np.allclose(time, np.arange(300)*.1, atol=1e-10, rtol=0):
        raise ValueError('retained fixed physiology time grid differs')
    hb = mask[:, 1:].all(axis=1)&np.isfinite(y[:, 1:]).all(axis=1)
    eeg = mask[:, 0]&np.isfinite(y[:, 0])
    if hb.any(): result['target_hb_ms'] = float(np.mean((y[hb, 1:]/sd[1:])**2))
    if eeg.any(): result['eeg_proxy_ms'] = float(np.mean((y[eeg, 0]/sd[0])**2))
    good = lambda key: row.get(key) in (True, 1, 'true', 'True')
    h = np.asarray(arrays.get('hb_only_prediction', np.full_like(y, np.nan)))
    result['feature_status'] = 'available_observations'
    if good('hb_only_success') and hb.any() and np.isfinite(h[hb, 1:]).all():
        r = (y[hb, 1:]-h[hb, 1:])/sd[1:]
        mean = r.mean(axis=0); centered = r-mean
        total = float(np.mean(r*r))
        result.update(hb_residual_total_ms=total, hb_residual_bias_ms=float(np.mean(mean*mean)),
            hb_residual_centered_ms=float(np.mean(centered*centered)),
            hb_residual_var_o=float(np.mean(centered[:, 0]**2)),
            hb_residual_var_r=float(np.mean(centered[:, 1]**2)),
            hb_residual_cov_or=float(np.mean(centered[:, 0]*centered[:, 1])),
            hb_residual_common_ms=float(np.mean((r[:, 0]+r[:, 1])**2)/2),
            hb_residual_difference_ms=float(np.mean((r[:, 0]-r[:, 1])**2)/2),
            feature_status='available_Hb_residual')
        if row.get('D1') is not None and np.isfinite(float(row['D1'])):
            if not np.isclose(np.sqrt(total), float(row['D1']), atol=1e-8, rtol=1e-8):
                raise ValueError('residual arrays do not reproduce retained fixed D1')
    series = {}
    if eeg.any():
        centered = y[:, 0]/sd[0]-np.mean(y[eeg, 0]/sd[0])
        series['EEG_proxy_amplitude'] = np.where(eeg, centered*centered, np.nan)
    if good('hb_only_success') and eeg.any() and np.isfinite(h[eeg, 0]).all():
        residual = (y[:, 0]-h[:, 0])/sd[0]
        series['Hb_readout_disagreement'] = np.where(eeg, residual*residual, np.nan)
        result['eeg_hb_readout_corr'] = _finite_pearson(y[eeg, 0], h[eeg, 0])
    if good('hb_only_success') and good('eeg_only_success') and eeg.any():
        re, rh = (np.asarray(arrays.get(k, np.full(300, np.nan))) for k in ('driver_e','driver_h'))
        support = eeg&np.isfinite(re)&np.isfinite(rh)
        if support.any():
            diff = (re-np.mean(re[support]))-(rh-np.mean(rh[support]))
            series['independent_drive_disagreement'] = np.where(support, (diff/sd[0])**2, np.nan)
            result['drive_e_h_corr'] = _finite_pearson(re[support], rh[support])
    if ocular is not None:
        value = float(np.asarray(ocular.get('rms_combined', np.nan)))
        result['eog_rms'] = value
        bins = np.asarray(ocular.get('rms_10hz', np.full((300,2),np.nan)),float)
        valid = np.asarray(ocular.get('valid_mask', np.isfinite(bins)),bool)
        if bins.shape != (300,2) or valid.shape != bins.shape:
            raise ValueError('EOG anchor shape must preserve 300x2 HEOG/VEOG bins')
        eog = np.sqrt(np.mean(bins*bins,axis=1));eog[~valid.all(axis=1)] = np.nan
        center = np.arange(100,249)
        for name, values in series.items():
            good_time = np.isfinite(values[center])
            for shift in (-50,0,50):good_time &= np.isfinite(eog[center+shift])
            take = center[good_time]
            result['eog_lag_'+name+'_bins'] = len(take)
            if len(take) < 3:continue
            rho = [_rho(values[take],eog[take+shift]) for shift in (-50,0,50)]
            result['eog_lag_'+name+'_rho0'] = rho[1]
            result['eog_lag_'+name+'_shift_mean'] = float(np.mean([rho[0],rho[2]]))
            result['eog_lag_'+name+'_contrast'] = rho[1]-float(np.mean([rho[0],rho[2]]))
    if counterfactual is not None and good('hb_only_success') and eeg.any() and np.isfinite(h[eeg,0]).all():
        proxies = {name:np.asarray(counterfactual.get(name+'_proxy',np.full(300,np.nan)),float)
                   for name in ('raw','clean','amplitude')}
        if any(v.shape != (300,) for v in proxies.values()):raise ValueError('counterfactual proxies must preserve 300 bins')
        common = eeg.copy()
        for v in proxies.values(): common &= np.isfinite(v)
        result['eog_counterfactual_bins'] = int(common.sum())
        if not np.array_equal(common,eeg):return result
        difference = float(np.max(abs(proxies['raw'][common]-y[common,0])))
        result['eog_raw_proxy_reproduction_error'] = difference
        if difference > 1e-10:raise ValueError('counterfactual raw proxy does not reproduce fixed target')
        for name,v in proxies.items():result['eog_'+name+'_DEH_mse'] = float(np.mean(((v[common]-h[common,0])/sd[0])**2))
        if row.get('D_EH') is not None and np.isfinite(float(row['D_EH'])):
            if not np.isclose(result['eog_raw_DEH_mse'],float(row['D_EH'])**2,rtol=1e-8,atol=1e-8):
                raise ValueError('retained raw EOG-sensitivity score differs from D_EH')
    return result


def _effect_summary(subjects, *, bootstrap_samples=5000, permutation_samples=100000, seed=17):
    rng = np.random.default_rng(seed);rows=[]
    keys=['analysis','dataset','endpoint','contrast','fdr_family']
    for identity, group in subjects.groupby(keys,sort=True,dropna=False):
        values=group.value.to_numpy(float);values=values[np.isfinite(values)];n=len(values)
        row=dict(zip(keys,identity),planned_subjects=len(group),subjects=n,
            mean=float(np.mean(values)) if n else np.nan,median=float(np.median(values)) if n else np.nan,
            positive_fraction=float(np.mean(values>0)) if n else np.nan,
            ci_low=np.nan,ci_high=np.nan,p_value=np.nan,q_value=np.nan,
            resampling_unit='whole_subject',status='available' if n>=6 else 'insufficient_subjects')
        if n>=6:
            draws=_bootstrap(values,lambda a:a.mean(axis=1),bootstrap_samples,rng)
            row['ci_low'],row['ci_high']=np.quantile(draws,[.025,.975])
            row['p_value'],row['permutation_method'],row['permutations']=_sign_flip(values,permutation_samples,rng)
        rows.append(row)
    result=pd.DataFrame(rows)
    if len(result):
        for family in result.fdr_family.unique():
            take=result.fdr_family==family;result.loc[take,'q_value']=benjamini_hochberg(result.loc[take,'p_value'])
            result.loc[take,'fdr_family_size']=int(result.loc[take,'p_value'].notna().sum())
    return result


def analyze_fixed_physiology_features(panel_frame, feature_frame, *, config):
    """Record/subject summaries of fixed residuals and the EOG A/B/C protocol."""
    if panel_frame.id.duplicated().any() or feature_frame.id.duplicated().any():
        raise ValueError('one physiology feature record per frozen panel identity required')
    if set(panel_frame.id)!=set(feature_frame.id):raise ValueError('physiology feature denominator differs from frozen panel')
    feature_columns=[c for c in feature_frame if c not in panel_frame or c=='id']
    data=panel_frame.merge(feature_frame[feature_columns],on='id',validate='one_to_one')
    moments=['hb_residual_total_ms','hb_residual_bias_ms','hb_residual_centered_ms','hb_residual_var_o',
        'hb_residual_var_r','hb_residual_cov_or','hb_residual_common_ms','hb_residual_difference_ms',
        'eeg_hb_readout_corr','drive_e_h_corr','eog_rms','eeg_proxy_ms','target_hb_ms']
    for key in moments:
        if key not in data:data[key]=np.nan
    record=data.groupby(['dataset','subject','record_id'],sort=True)[moments].mean().reset_index()
    subject=record.groupby(['dataset','subject'],sort=True)[moments].mean().reset_index()
    shape_rows=[]
    for row in subject.to_dict('records'):
        block=data[(data.dataset==row['dataset'])&(data.subject==row['subject'])]
        positive=lambda name:np.isfinite(row[name]) and row[name]>0
        values={
            'Hb_bias_fraction':row['hb_residual_bias_ms']/row['hb_residual_total_ms'] if positive('hb_residual_total_ms') else np.nan,
            'Hb_centered_residual_RMS':np.sqrt(row['hb_residual_centered_ms']),
            'Hb_residual_OR_correlation':row['hb_residual_cov_or']/np.sqrt(row['hb_residual_var_o']*row['hb_residual_var_r']) if positive('hb_residual_var_o') and positive('hb_residual_var_r') else np.nan,
            'Hb_common_direction_fraction':row['hb_residual_common_ms']/(row['hb_residual_common_ms']+row['hb_residual_difference_ms']) if positive('hb_residual_total_ms') else np.nan,
            'EEG_Hb_readout_correlation':row['eeg_hb_readout_corr'],
            'drive_E_H_correlation':row['drive_e_h_corr']}
        for feature,value in values.items():
            key=('eeg_hb_readout_corr' if feature=='EEG_Hb_readout_correlation' else
                 'drive_e_h_corr' if feature=='drive_E_H_correlation' else 'hb_residual_total_ms')
            shape_rows.append(dict(dataset=row['dataset'],subject=row['subject'],feature=feature,value=value,
                planned_windows=len(block),available_windows=int(block[key].notna().sum()),
                planned_records=block.record_id.nunique(),available_records=block.loc[block[key].notna(),'record_id'].nunique()))
    shapes=pd.DataFrame(shape_rows)
    # Collapse the three Hb-pair copies before relating any metric to one EOG
    # trace. Every record, and then every subject, receives equal weight.
    keys=['dataset','subject','record_id','window_start_s']
    for metric in PRIMARY_METRICS:data[metric+'_mse_for_eog']=pd.to_numeric(data[metric],errors='coerce')**2
    scalar_columns=[m+'_mse_for_eog' for m in PRIMARY_METRICS]+['eog_rms','eeg_proxy_ms','zero_NMSE']
    scalar=data.groupby(keys,sort=True)[scalar_columns].mean().reset_index()
    eog_datasets=sorted(data.loc[np.isfinite(data.eog_rms),'dataset'].unique())
    record_effects=[]
    for (dataset,subject_id,record_id),block in scalar[scalar.dataset.isin(eog_datasets)].groupby(keys[:3],sort=True):
        for endpoint in PRIMARY_METRICS:
            metric=endpoint+'_mse_for_eog'
            good=np.isfinite(block.eog_rms)&np.isfinite(block[metric]);n=int(good.sum())
            rho=_rho(block.loc[good,'eog_rms'].to_numpy(),block.loc[good,metric].to_numpy()) if n>=6 else np.nan
            base=dict(dataset=dataset,subject=subject_id,record_id=record_id,endpoint=endpoint,
                planned_windows=len(block),available_windows=n)
            record_effects.append(dict(base,analysis='cross_window',contrast='EOG_metric_rank_correlation',
                fdr_family='ocular_A_primary',value=rho))
            good &= np.isfinite(block.eeg_proxy_ms)&np.isfinite(block.zero_NMSE)
            n=int(good.sum());partial=np.nan
            if n>=10:
                b=block[good];controls=np.column_stack((np.ones(n),rankdata(b.eeg_proxy_ms),rankdata(b.zero_NMSE)))
                a=rankdata(b.eog_rms);y=rankdata(b[metric])
                ar=a-controls@np.linalg.lstsq(controls,a,rcond=None)[0]
                yr=y-controls@np.linalg.lstsq(controls,y,rcond=None)[0]
                if np.linalg.norm(ar)>1e-8*np.linalg.norm(a) and np.linalg.norm(yr)>1e-8*np.linalg.norm(y):
                    partial=_finite_pearson(ar,yr)
            record_effects.append(dict(base,available_windows=n,analysis='cross_window_partial',
                contrast='rank_correlation_given_EEG_and_Hb_amplitude',fdr_family='ocular_A_amplitude_diagnostic',value=partial))
    lag_series=('EEG_proxy_amplitude','Hb_readout_disagreement','independent_drive_disagreement')
    for endpoint in lag_series:
        key='eog_lag_'+endpoint+'_contrast'
        if key not in data:continue
        cols=[key,'eog_lag_'+endpoint+'_rho0','eog_lag_'+endpoint+'_shift_mean']
        windows=data.groupby(keys,sort=True)[cols].mean().reset_index()
        for (dataset,subject_id,record_id),block in windows[windows.dataset.isin(eog_datasets)].groupby(keys[:3],sort=True):
            values=block[key];valid=np.isfinite(values)
            record_effects.append(dict(dataset=dataset,subject=subject_id,record_id=record_id,endpoint=endpoint,
                analysis='lag_specificity',contrast='rho0_minus_mean_nonwrapping_pm5s',fdr_family='ocular_B_primary',
                value=float(values[valid].mean()) if valid.any() else np.nan,
                rho0=float(block.loc[valid,cols[1]].mean()),shift_mean=float(block.loc[valid,cols[2]].mean()),
                planned_windows=len(block),available_windows=int(valid.sum())))
    cf_columns=['eog_'+name+'_DEH_mse' for name in ('raw','clean','amplitude')]
    cf_subjects=pd.DataFrame()
    if all(c in data for c in cf_columns):
        valid=data[cf_columns].notna().all(axis=1)
        values=data[cf_columns].where(valid)
        windows=pd.concat((data[keys],values),axis=1).groupby(keys,sort=True)[cf_columns].mean().reset_index()
        for (dataset,subject_id,record_id),block in windows[windows.dataset.isin(eog_datasets)].groupby(keys[:3],sort=True):
            good=block[cf_columns].notna().all(axis=1);v=block.loc[good,cf_columns].mean()
            for left,right,family in [('clean','raw','ocular_C_primary'),('clean','amplitude','ocular_C_primary'),('amplitude','raw','ocular_C_scale_diagnostic')]:
                record_effects.append(dict(dataset=dataset,subject=subject_id,record_id=record_id,endpoint='D_EH_squared',
                    analysis='counterfactual',contrast=left+'_minus_'+right,fdr_family=family,
                    value=float(v['eog_'+left+'_DEH_mse']-v['eog_'+right+'_DEH_mse']) if good.any() else np.nan,
                    raw_mse=v.get(cf_columns[0],np.nan),clean_mse=v.get(cf_columns[1],np.nan),amplitude_mse=v.get(cf_columns[2],np.nan),
                    planned_windows=len(block),available_windows=int(good.sum())))
        cf_records=windows.groupby(keys[:3],sort=True)[cf_columns].mean().reset_index()
        cf_subjects=cf_records.groupby(keys[:2],sort=True)[cf_columns].mean().reset_index()
    effects=pd.DataFrame(record_effects)
    subject_effects=[]
    if len(effects):
        effect_keys=['analysis','dataset','subject','endpoint','contrast','fdr_family']
        for identity,group in effects.groupby(effect_keys,sort=True):
            good=np.isfinite(group.value);row=dict(zip(effect_keys,identity))
            row.update(value=float(group.loc[good,'value'].mean()) if good.any() else np.nan,
                planned_records=len(group),available_records=int(good.sum()),
                planned_windows=int(group.planned_windows.sum()),available_windows=int(group.available_windows.sum()))
            subject_effects.append(row)
    subject_effects=pd.DataFrame(subject_effects)
    settings=config.get('statistics',config)
    associations=_effect_summary(subject_effects,
        bootstrap_samples=int(settings.get('bootstrap_samples',5000)),
        permutation_samples=int(settings.get('permutation_samples',100000)),seed=int(settings.get('seed',17))) if len(subject_effects) else pd.DataFrame()
    shape_summary=dict(status='completed',scope_description='预先冻结的三个空间通道面板；已验证固定Hb-only残差，记录和被试等权。',
        planned_windows=len(data),feature_definitions=PHYSIOLOGY_FEATURE_DEFINITIONS,
        report_findings=['共同/相反方向只是标准化残差的统计坐标分解；不把共同方向称为血容量、HbT或系统血流来源。',
            '偏置平方与去均值残差平方分开呈现；25秒支持及既有带通不用于识别Mayer波、呼吸或心血管来源。'])
    ocular_summary=dict(status='completed',scope_description='原生EOG辅助活动锚点与固定三通道失配的探索性关联；只有Single-Trial和Simultaneous有登记HEOG/VEOG。',
        datasets=eog_datasets,planned_panel_windows=len(data),available_EOG_pair_windows=int(data.eog_rms.notna().sum()),
        method=dict(A='先同窗通道对均方聚合，record内至少6窗Spearman，record等权到subject；partial至少10窗并控制EEG代理与Hb零预测幅度。',
            B='EEG代理中心化平方、Hb读出差平方、独立驱动去DC差平方；共同149点10.0–24.8秒，0与非循环±5秒，先窗口、record、subject等权。',
            C='原记录EOG回归后仅重算冻结PCA的EEG代理；raw/clean/amplitude共同评分与训练SD，固定Hb-only读出不动；负差表示新观察坐标更接近该读出。'),
        report_findings=['本次固定Tak EEG代理没有经过项目通用缓存的EOG回归；当前关联不应描述为清理后的残余。',
            'EOG活动可包含眼动、肌肉、电极或其他噪声；相关与非循环位移对照不证明眼电造成失配。',
            '清理改变了EEG目标坐标。幅度匹配对照只控制均值/尺度；即使clean优于该对照，也不证明纯眼源去除或神经信号准确性改善。'])
    return dict(shape_subjects=shapes,shape_summary=shape_summary,ocular_records=effects,
        ocular_subjects=subject_effects,ocular_associations=associations,ocular_counterfactual_subjects=cf_subjects,
        ocular_summary=ocular_summary)


def spatial_geometry_from_plan(plan: dict) -> pd.DataFrame:
    """Use the frozen channel registry carried by the retained plan, without I/O."""
    from src.data.channel_geometry import project_head_topview
    rows = []
    for record in plan["records"]:
        geometry = record.get("geometry_identity", {}).get("fnirs", [])
        for pair in record.get("pairs", []):
            indices = pair["component_indices"]
            base = dict(dataset=record["dataset"], subject=str(record["subject"]),
                record_id=record["id"], pair=pair["id"], native_pair_index=pair["native_index"],
                x=np.nan, y=np.nan, location_id="unavailable:"+pair["id"],
                geometry_status="unavailable", coordinate_system="unavailable", geometry_source="")
            if len(indices) != 2 or max(indices) >= len(geometry):
                rows.append(base); continue
            oxy, deoxy = [geometry[i] for i in indices]
            if oxy.get("component") != "HbO" or deoxy.get("component") != "HbR":
                raise ValueError("spatial map requires the retained ordered HbO/HbR pair")
            try:
                xy = project_head_topview(oxy)
                if not np.array_equal(xy, project_head_topview(deoxy)):
                    raise ValueError("paired Hb geometry differs")
            except ValueError as exc:
                base["geometry_status"] = str(exc); rows.append(base); continue
            base.update(x=float(xy[0]), y=float(xy[1]),
                location_id=f"{xy[0]:.7f},{xy[1]:.7f}", geometry_status="available",
                coordinate_system=oxy["coordinate_system"], geometry_source=oxy.get("source_file", ""))
            rows.append(base)
    frame = pd.DataFrame(rows)
    if frame.duplicated(["dataset", "subject", "record_id", "pair"]).any():
        raise ValueError("duplicate planned channel geometry identity")
    return frame


def spatial_half_metrics(windows: pd.DataFrame) -> pd.DataFrame:
    """Retained D1 and matched Hb ratio in early/late halves, at each Hb site.

    The split uses the original record's ordered planned times.  An odd middle
    window is omitted.  Two valid distinct windows per site/half are required;
    this is a selection diagnostic, not an independent validation experiment.
    """
    data = _checked_windows(windows, ("D1", HB_RATIO))
    data["period"] = "middle"
    for _, group in data.groupby(["dataset", "subject", "record_id"], sort=False):
        times = np.sort(group.window_start_s.unique())
        half = len(times)//2
        if half < 2:
            continue
        data.loc[group.index[group.window_start_s.isin(times[:half])], "period"] = "early"
        data.loc[group.index[group.window_start_s.isin(times[-half:])], "period"] = "late"
    data = data[data.period.ne("middle")].copy()
    keys = ["dataset", "subject", "task_label", "record_id", "record_unit", "pair", "period"]
    outputs = []
    for metric in ("D1", HB_RATIO):
        numerator, denominator, valid = _parts(data, metric)
        source = data[keys].copy()
        source["aggregation_numerator"] = np.where(valid, numerator, np.nan)
        source["aggregation_denominator"] = np.where(valid, denominator, np.nan)
        source["available_windows"] = valid.astype(int)
        grouped = source.groupby(keys, sort=True, dropna=False)
        table = grouped.agg(aggregation_numerator=("aggregation_numerator", "mean"),
            aggregation_denominator=("aggregation_denominator", "mean"),
            planned_windows=("available_windows", "size"), available_windows=("available_windows", "sum")).reset_index()
        table["metric"] = metric
        table["value"] = table.aggregation_numerator / table.aggregation_denominator
        if metric in ROOT_METRICS:
            table["value"] = np.sqrt(table.value)
        table.loc[table.available_windows < 2, ["value", "aggregation_numerator", "aggregation_denominator"]] = np.nan
        outputs.append(table)
    return pd.concat(outputs, ignore_index=True)


def aggregate_spatial_maps(native_pairs: pd.DataFrame, halves: pd.DataFrame,
                           geometry: pd.DataFrame, *, minimum_window_fraction: float = .8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Record/site maps and record-equal subject maps with explicit missingness.

    Coincident projected positions are combined before plotting.  ROOT metrics
    average squared quantities, and Hb ratios average their matched numerator
    and denominator separately. Visual probes share one acquisition map but
    remain distinct native records in the channel table.
    """
    if not 0 < minimum_window_fraction <= 1:
        raise ValueError("minimum_window_fraction must be in (0, 1]")
    source = native_pairs[native_pairs.metric.isin(SPATIAL_METRICS)].copy()
    source["period"] = "full"
    source = pd.concat([source, halves], ignore_index=True)
    channel = source.merge(geometry, on=["dataset", "subject", "record_id", "pair"], how="left", validate="many_to_one")
    if channel.location_id.isna().any():
        raise ValueError("retained channel metrics are missing their planned geometry identity")
    channel["finite_value"] = np.isfinite(channel.value)
    channel["eligible_member"] = (channel.finite_value & channel.geometry_status.eq("available")
        & (channel.available_windows >= minimum_window_fraction * channel.planned_windows))
    keys = ["dataset", "subject", "task_label", "record_unit", "period", "location_id", "metric"]
    joined = lambda values: ";".join(sorted(set(values.astype(str))))
    maps = channel.groupby(keys, sort=True, dropna=False).agg(
        aggregation_numerator=("aggregation_numerator", "mean"), aggregation_denominator=("aggregation_denominator", "mean"),
        x=("x", "first"), y=("y", "first"), pair_members=("pair", joined), record_ids=("record_id", joined),
        planned_members=("pair", "size"), available_members=("finite_value", "sum"), eligible_members=("eligible_member", "sum"),
        planned_windows=("planned_windows", "sum"), available_windows=("available_windows", "sum")).reset_index()
    maps["value"] = maps.aggregation_numerator / maps.aggregation_denominator
    take = maps.metric.isin(ROOT_METRICS)
    maps.loc[take, "value"] = np.sqrt(maps.loc[take, "value"])
    invalid_visual = maps.dataset.eq("visual_cognitive_motivation") & ~maps.metric.isin(SPATIAL_HB_METRICS)
    maps["interpretation"] = np.where(invalid_visual, "invalid_visual_EEG_identity_not_mapped", "retained_conditional_metric")
    # Display every computable retained endpoint, including partial time
    # support. The stricter coverage threshold belongs to pattern selection,
    # and must not erase a measured record from the descriptive maps.
    maps["display_valid"] = (np.isfinite(maps.value) & np.isfinite(maps.x) & np.isfinite(maps.y) & ~invalid_visual)
    maps["selection_valid"] = maps.display_valid & maps.eligible_members.eq(maps.planned_members)
    maps["display_value"] = maps.value.where(maps.display_valid)
    maps["planned_records"] = 1
    maps["available_records"] = maps.display_valid.astype(int)
    maps["map_id"] = maps.record_unit
    full = maps[maps.period.eq("full")].copy()
    for column in ("aggregation_numerator", "aggregation_denominator"):
        full[column] = full[column].where(full.display_valid)
    group = ["dataset", "subject", "location_id", "metric"]
    means = full.groupby(group, sort=True).agg(
        aggregation_numerator=("aggregation_numerator", "mean"), aggregation_denominator=("aggregation_denominator", "mean"),
        x=("x", "first"), y=("y", "first"), pair_members=("pair_members", "first"),
        record_ids=("record_ids", joined), planned_records=("record_unit", "size"),
        available_records=("display_valid", "sum"), planned_windows=("planned_windows", "sum"),
        available_windows=("available_windows", "sum"), interpretation=("interpretation", "first")).reset_index()
    means["value"] = means.aggregation_numerator / means.aggregation_denominator
    take = means.metric.isin(ROOT_METRICS)
    means.loc[take, "value"] = np.sqrt(means.loc[take, "value"])
    means["display_valid"] = np.isfinite(means.value)
    means["selection_valid"] = means.display_valid & (means.available_records >= minimum_window_fraction*means.planned_records)
    means["display_value"] = means.value.where(means.display_valid)
    means["period"] = "subject_mean"
    means["task_label"] = "all_native_records_equal"
    means["record_unit"] = "subject_mean"
    means["map_id"] = "subject_mean"
    return channel, pd.concat([maps, means], ignore_index=True)


def spatial_neighbour_weights(xy: np.ndarray, neighbours: int = 4) -> np.ndarray:
    """Symmetric near-neighbour graph in the declared display projection."""
    from scipy.spatial.distance import cdist
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all() or len(xy) < 3:
        raise ValueError("at least three finite projected positions required")
    distance = cdist(xy, xy)
    np.fill_diagonal(distance, np.inf)
    if np.min(distance) <= 1e-8:
        raise ValueError("coincident projected positions must be combined first")
    spacing = float(np.median(distance.min(axis=1)))
    order = np.argsort(distance, axis=1, kind="stable")[:, :min(neighbours, len(xy)-1)]
    weights = np.zeros(distance.shape)
    weights[np.arange(len(xy))[:, None], order] = 1.
    weights = np.maximum(weights, weights.T)
    weights[distance > 2.5*spacing] = 0.
    return weights


def spatial_moran(values: np.ndarray, weights: np.ndarray) -> float:
    """Descriptive spatial coherence; no location-wise significance claim."""
    values = np.asarray(values, float)
    centered = values-values.mean()
    denominator = float(centered@centered)
    if denominator <= 1e-24 or weights.sum() <= 0:
        return np.nan
    return float(len(values)/weights.sum() * (centered@weights@centered)/denominator)


def rank_spatial_patterns(maps: pd.DataFrame, *, examples_per_dataset: int = 3,
                          minimum_location_fraction: float = .8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select illustrative D1 patterns by contrast, neighbours and early/late agreement.

    Selection uses the same evidence shown later and is explicitly exploratory.
    Every candidate and every unavailable/zero-score record is retained.
    """
    if examples_per_dataset < 1:
        raise ValueError("positive illustrative subject count required")
    rows = []
    source = maps[maps.metric.eq("D1") & maps.period.ne("subject_mean")]
    for identity, group in source.groupby(["dataset", "subject", "record_unit", "task_label"], sort=True):
        full = group[group.period.eq("full")].set_index("location_id")
        a = group[group.period.eq("early")].set_index("location_id")
        b = group[group.period.eq("late")].set_index("location_id")
        common = full.index.intersection(a.index).intersection(b.index)
        common = [site for site in common if full.loc[site, "selection_valid"] and a.loc[site, "selection_valid"] and b.loc[site, "selection_valid"]]
        row = dict(zip(["dataset", "subject", "record_unit", "task_label"], identity),
            planned_locations=len(full), common_locations=len(common), spatial_contrast=np.nan,
            spatial_moran_I=np.nan, early_late_rho=np.nan, score=0., eligible=False,
            status="insufficient_common_early_late_locations")
        if len(common) >= max(8, int(np.ceil(minimum_location_fraction*len(full)))):
            values = full.loc[common, "display_value"].to_numpy(float)
            median = float(np.median(values))
            contrast = float((np.quantile(values,.9)-np.quantile(values,.1))/median) if median > 1e-12 else 0.
            weights = spatial_neighbour_weights(full.loc[common, ["x", "y"]].to_numpy(float))
            moran = spatial_moran(values, weights)
            x, y = a.loc[common, "display_value"].to_numpy(float), b.loc[common, "display_value"].to_numpy(float)
            rho = float(np.corrcoef(rankdata(x), rankdata(y))[0,1]) if np.ptp(x)>1e-12 and np.ptp(y)>1e-12 else np.nan
            finite = np.isfinite([contrast, moran, rho]).all()
            score = contrast*np.clip(moran,0,1)*np.clip(rho,0,1) if finite else 0.
            row.update(spatial_contrast=contrast, spatial_moran_I=moran, early_late_rho=rho,
                score=float(score), eligible=bool(finite), status="available" if finite else "constant_or_degenerate_map")
        rows.append(row)
    records = pd.DataFrame(rows)
    candidates = []
    for identity, group in records.groupby(["dataset", "subject"], sort=True):
        good = group[group.eligible]
        fraction = len(good)/len(group)
        candidates.append(dict(dataset=identity[0], subject=identity[1], planned_records=len(group), eligible_records=len(good),
            eligible_record_fraction=fraction, median_record_score=float(good.score.median()) if len(good) else 0.,
            score=float(good.score.median()*fraction) if len(good) else 0.,
            median_spatial_contrast=float(good.spatial_contrast.median()) if len(good) else np.nan,
            median_moran_I=float(good.spatial_moran_I.median()) if len(good) else np.nan,
            median_early_late_rho=float(good.early_late_rho.median()) if len(good) else np.nan,
            eligible=bool(len(good)), status="available" if len(good) else "no_record_with_two_valid_windows_per_half"))
    ranking = pd.DataFrame(candidates).sort_values(["dataset", "eligible", "score", "subject"], ascending=[True, False, False, True])
    ranking["rank"] = ranking.groupby("dataset").cumcount()+1
    ranking["selected"] = ranking.eligible & (ranking["rank"] <= examples_per_dataset)
    return records, ranking.reset_index(drop=True)


def spatial_record_similarities(maps: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    """Record map concordance; different videos/tasks remain distinct conditions."""
    chosen = ranking[ranking.selected][["dataset", "subject"]]
    source = maps[maps.period.eq("full")].merge(chosen, on=["dataset", "subject"], validate="many_to_one")
    rows = []
    for (dataset, subject, metric), group in source.groupby(["dataset", "subject", "metric"], sort=True):
        units = {key: g.set_index("location_id") for key,g in group.groupby("record_unit", sort=True)}
        for first, second in combinations(units, 2):
            a,b = units[first], units[second]
            common = a.index.intersection(b.index)
            common = [site for site in common if a.loc[site,"display_valid"] and b.loc[site,"display_valid"]]
            rho = np.nan
            if len(common) >= 8:
                x,y = a.loc[common,"display_value"].to_numpy(float),b.loc[common,"display_value"].to_numpy(float)
                if np.ptp(x)>1e-12 and np.ptp(y)>1e-12:
                    rho = float(np.corrcoef(rankdata(x),rankdata(y))[0,1])
            rows.append(dict(dataset=dataset,subject=subject,metric=metric,record_a=first,record_b=second,
                task_a=a.task_label.iloc[0],task_b=b.task_label.iloc[0],
                same_task=bool(a.task_label.iloc[0]==b.task_label.iloc[0]),
                common_locations=len(common),spearman_rho=rho,
                interpretation="descriptive_selected_case_not_independent_confirmation"))
    return pd.DataFrame(rows)
