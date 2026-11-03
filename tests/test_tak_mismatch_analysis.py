"""Synthetic, artifact-independent checks of frozen Tak descriptive inference."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import tak_mismatch as analysis


def _window(subject="s", task="a", record="r", pair="p", start=0., value=1., **metadata):
    row = dict(dataset="d", subject=subject, task_label=task, record_id=record,
        native_support_id=record+"/support", pair=pair, window_start_s=start,
        parameter_mode="fixed", D1=value, D_EH=value, D_drive_H=value,
        D_obs=value, Delta_spatial=value, D4=value, zero_NMSE=1.)
    row.update(metadata)
    return row


def _value(frame, subject="s", metric="D1", **selectors):
    group = frame[(frame.subject == subject) & (frame.metric == metric)]
    for column, value in selectors.items():
        group = group[group[column] == value]
    assert len(group) == 1
    return group.iloc[0]


def test_record_equal_owner_and_endpoint_failures_are_preserved():
    rows = [_window(record="long", start=30*i, value=1., fixed_success=False) for i in range(10)]
    rows += [_window(record="short", value=3.),
             _window(subject="failed", record="unavailable", value=np.nan)]
    result = analysis.aggregate_fixed_metrics(pd.DataFrame(rows))
    subject = _value(result["subjects"])
    assert subject.value == pytest.approx(np.sqrt(5.))
    assert subject.planned_windows == 11
    assert subject.available_windows == 11  # A joint failure is not a D1 gate.
    assert subject.planned_records == 2
    assert subject.available_records == 2
    failed = _value(result["subjects"], subject="failed")
    assert np.isnan(failed.value)
    assert failed.planned_windows == failed.missing_windows == 1
    assert failed.planned_records == 1 and failed.available_records == 0


def test_subject_totals_remain_record_equal_rather_than_task_equal():
    rows = [_window(task="a", record=f"a{i}", value=1.) for i in range(3)]
    rows += [_window(task="b", record="b0", value=3.)]
    result = analysis.aggregate_fixed_metrics(pd.DataFrame(rows))
    assert _value(result["subjects"]).value == pytest.approx(np.sqrt(3.))
    assert _value(result["subject_tasks"], task_label="b").value == 3.


def test_mixed_task_record_is_pooled_before_overall_equal_record_step():
    rows = [_window(task="a", record="mixed", start=30*i, value=1.) for i in range(3)]
    rows += [_window(task="b", record="mixed", start=90., value=3.),
             _window(task="a", record="other", value=5.)]
    result = analysis.aggregate_fixed_metrics(pd.DataFrame(rows))
    # mixed-record MSE=(1+1+1+9)/4=3; other-record MSE=25.
    assert _value(result["subjects"]).value == pytest.approx(np.sqrt(14.))
    assert _value(result["subjects"]).planned_records == 2
    assert _value(result["subject_tasks"], task_label="a").value == pytest.approx(np.sqrt(13.))


def test_continuous_supports_do_not_become_independent_records():
    rows = [_window(record="one_record", start=30*i, value=value,
                    native_support_id=f"support{i}") for i, value in enumerate((1., 3.))]
    result = analysis.aggregate_fixed_metrics(pd.DataFrame(rows))
    assert len(result["native_records"].query("metric == 'D1'")) == 1
    row = _value(result["native_records"])
    assert row.native_supports == 2 and row.planned_records == 1
    assert row.value == pytest.approx(np.sqrt(5.))


def test_ratio_is_ratio_of_matched_record_means_and_may_exceed_one():
    rows = [_window(record="long", start=30*i, value=1., zero_NMSE=1.) for i in range(10)]
    rows += [_window(record="short", value=3., zero_NMSE=3.),
             _window(record="unmatched", value=100., zero_NMSE=np.nan),
             _window(record="zero", value=2., zero_NMSE=0.)]
    result = analysis.aggregate_fixed_metrics(pd.DataFrame(rows))
    ratio = _value(result["subjects"], metric=analysis.HB_RATIO)
    assert ratio.value == pytest.approx(2.5)  # ((1+9)/2) / ((1+3)/2).
    assert ratio.available_windows == 11 and ratio.planned_windows == 13
    assert ratio.available_records == 2 and ratio.planned_records == 4
    assert ratio.value != pytest.approx((1.+3.)/2)  # Not an average of window ratios.


@pytest.mark.parametrize("change,match", [
    (lambda x: pd.concat([x, x]), "duplicate"),
    (lambda x: x.assign(parameter_mode="current_dof"), "fixed rows only"),
    (lambda x: x.assign(D1=-1.), "nonnegative"),
    (lambda x: x.drop(columns="native_support_id"), "missing fixed-analysis"),
])
def test_fixed_input_contract(change, match):
    with pytest.raises(ValueError, match=match):
        analysis.aggregate_fixed_metrics(change(pd.DataFrame([_window()])))


def test_subject_descriptives_keep_failed_denominators_and_whole_subject_center():
    rows = [_window(subject=str(i), value=float(i+1)) for i in range(6)]
    rows += [_window(subject="missing", value=np.nan)]
    tables = analysis.aggregate_fixed_metrics(pd.DataFrame(rows))
    summary = analysis.summarize_subject_distributions(tables["subjects"], bootstrap_samples=80)
    row = summary[summary.metric == "D1"].iloc[0]
    assert row.planned_subjects == 7 and row.available_subjects == 6
    assert row.subject_equal_value == pytest.approx(np.sqrt(np.mean(np.arange(1, 7)**2)))
    assert row["median"] == pytest.approx(3.5)
    assert row.ci_low <= row.subject_equal_value <= row.ci_high


def _task_table(n=6):
    rows = []
    for subject in range(n):
        for task, value in (("a", subject+1.), ("b", subject+2.)):
            rows.append(_window(subject=str(subject), task=task, record=task, value=value))
    return analysis.aggregate_fixed_metrics(pd.DataFrame(rows))["subject_tasks"]


def test_task_comparisons_are_paired_and_sign_flip_is_exact():
    table = _task_table()
    # An unavailable subject contributes a denominator, not a zero difference.
    failed = table[table.subject == "0"].assign(subject="failed", value=np.nan)
    table = pd.concat([table, failed], ignore_index=True)
    result = analysis.paired_task_comparisons(table, bootstrap_samples=80)
    d1 = result[result.metric == "D1"].iloc[0]
    assert d1.mean_difference == 1. and d1.median_difference == 1.
    assert d1.paired_subjects == 6 and d1.planned_a == 7
    assert d1.available_a == 6 and d1.missing_a == 1
    assert d1.positive_fraction == 1. and np.isnan(d1.cohen_dz)
    assert d1.ci_low == d1.ci_high == 1.
    assert d1.p_value == pytest.approx(2/64)
    assert d1.permutations == 64 and d1.permutation_method == "exact_sign_flip"
    assert d1.fdr_family == "primary" and d1.fdr_family_size == 3
    assert result[result.metric == "D4"].iloc[0].fdr_family == "diagnostic"


def test_small_task_support_is_descriptive_and_duplicate_subjects_rejected():
    table = _task_table(5)
    result = analysis.paired_task_comparisons(table, bootstrap_samples=30)
    assert result.p_value.isna().all() and result.q_value.isna().all()
    assert result.ci_low.isna().all() and result.fdr_family_size.eq(0).all()
    assert result.status.eq("insufficient_paired_subjects").all()
    with pytest.raises(ValueError, match="must be unique"):
        analysis.paired_task_comparisons(pd.concat([table, table.iloc[:1]]))


def test_monte_carlo_sign_flip_is_seeded_and_uses_plus_one():
    table = _task_table(17)
    first = analysis.paired_task_comparisons(table, bootstrap_samples=20, permutation_samples=100, seed=4)
    second = analysis.paired_task_comparisons(table, bootstrap_samples=20, permutation_samples=100, seed=4)
    pd.testing.assert_frame_equal(first, second)
    row = first[first.metric == "D1"].iloc[0]
    assert row.permutation_method == "monte_carlo_sign_flip_plus_one"
    assert row.p_value >= 1/101
    assert row.permutations == 100


def test_bh_excludes_only_unavailable_p_values_and_preserves_positions():
    result = analysis.benjamini_hochberg([.01, np.nan, .04, .03])
    np.testing.assert_allclose(result[[0, 2, 3]], [.03, .04, .04])
    assert np.isnan(result[1])


def test_repeats_use_declared_units_endpoints_and_same_physical_pairs():
    rows = []
    for subject in range(6):
        for record, order, value in (("first", 0, subject+1.), ("middle", 1, 999.), ("last", 2, subject+2.)):
            rows.append(_window(subject=str(subject), record=record, value=value,
                repeat_id=record, repeat_kind="native_record", record_order=order))
        rows.append(_window(subject=str(subject), record="first", pair="unmatched", value=999.,
            repeat_id="first", repeat_kind="native_record", record_order=0))
    result = analysis.aggregate_fixed_metrics(pd.DataFrame(rows))
    repeat = analysis.native_record_repeatability(result["native_pairs"], bootstrap_samples=50)
    pairs = repeat["pairs"].query("metric == 'D1'")
    assert pairs.repeat_a.eq("first").all() and pairs.repeat_b.eq("last").all()
    assert pairs.planned_pairs.eq(1).all() and pairs.available_pairs.eq(1).all()
    np.testing.assert_allclose(pairs.value_b-pairs.value_a, 1.)
    summary = repeat["summary"].query("metric == 'D1'").iloc[0]
    assert summary.subjects == 6 and summary.spearman_rho == pytest.approx(1.)
    assert summary.median_signed_difference == summary.median_absolute_difference == 1.
    assert summary.rho_ci_low == pytest.approx(1.)


def test_synchronous_probe_branches_and_supports_do_not_create_retest():
    rows = [_window(record=f"part1_probe{probe}", pair=f"probe{probe}",
                    native_support_id=f"support{probe}", repeat_id="part1",
                    repeat_kind="same_visit_cross_part", record_order=0) for probe in (1, 2)]
    result = analysis.aggregate_fixed_metrics(pd.DataFrame(rows))
    repeat = analysis.native_record_repeatability(result["native_pairs"], bootstrap_samples=20)
    assert repeat["pairs"].status.eq("unavailable_independent_repeat").all()
    assert repeat["summary"].subjects.eq(0).all()
    assert repeat["summary"].planned_subjects.eq(1).all()


def test_failed_first_repeat_is_not_replaced_by_successful_middle_record():
    rows = [_window(record=name, value=value, repeat_id=name,
                    repeat_kind="native_record", record_order=i)
            for i, (name, value) in enumerate((("first", np.nan), ("middle", 1.), ("last", 2.)))]
    tables = analysis.aggregate_fixed_metrics(pd.DataFrame(rows))
    result = analysis.native_record_repeatability(tables["native_pairs"], bootstrap_samples=20)
    row = _value(result["pairs"])
    assert row.repeat_a == "first" and row.repeat_b == "last"
    assert np.isnan(row.value_a) and np.isnan(row.value_b)
    assert row.planned_pairs == 1 and row.available_pairs == 0


def test_split_half_discards_central_window_and_does_not_reweight_records():
    rows = []
    for record, values in (("long", (1., 1., 999., 3., 3.)), ("short", (3., 3., 1., 1.))):
        rows.extend(_window(record=record, start=i*30, value=value) for i, value in enumerate(values))
    rows.append(_window(record="too_short", start=0., value=99.))
    result = analysis.within_record_consistency(pd.DataFrame(rows), bootstrap_samples=20)
    pair = _value(result["pairs"])
    assert pair.planned_records == 3 and pair.available_records == 2
    assert pair.value_a == pair.value_b == pytest.approx(np.sqrt(5.))
    record = _value(result["records"], record_id="long")
    assert record.windows_per_half == 2 and record.dropped_middle_windows == 1
    assert record.value_a == 1. and record.value_b == 3.
    summary = result["summary"].query("metric == 'D1'").iloc[0]
    assert summary.status == "insufficient_subjects" and np.isnan(summary.rho_ci_low)


def test_empty_input_has_stable_table_schemas():
    frame = pd.DataFrame([_window()]).iloc[:0]
    result = analysis.aggregate_fixed_metrics(frame)
    assert {"subjects", "subject_tasks", "native_records", "native_pairs"} == set(result)
    assert "available_records" in result["subjects"].columns
    assert analysis.paired_task_comparisons(result["subject_tasks"]).empty
    assert analysis.native_record_repeatability(result["native_pairs"])["summary"].empty
    assert analysis.within_record_consistency(frame)["summary"].empty


def test_split_half_keeps_boundaries_and_requires_two_valid_times_per_half():
    rows = [_window(start=i*30, value=value) for i, value in enumerate((1., np.nan, 2., 3.))]
    # Additional simultaneous channels do not count as another time window.
    rows += [_window(start=0., pair="second", value=1.),
             _window(start=60., pair="second", value=2.),
             _window(start=90., pair="second", value=3.)]
    result = analysis.within_record_consistency(pd.DataFrame(rows), bootstrap_samples=20)
    record = _value(result["records"])
    assert record.windows_per_half == 2
    assert record.available_windows_a == 1 and record.available_windows_b == 2
    assert record.status == "insufficient_available_half_windows"
    assert np.isnan(record.value_a) and np.isnan(record.value_b)
    assert _value(result["pairs"]).available_records == 0
