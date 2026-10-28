"""Portable report checks use small, temporary, non-authorizing evidence."""
from __future__ import annotations

import hashlib
import json

import fitz
import numpy as np
import pandas as pd
import pytest

from experiments.scripts import render_tak_mismatch as renderer


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _terminal_view(*, planned, completed, numerical_failure, unavailable_support,
                   execution_failure=0, missing=0, fitted):
    return dict(planned=planned, completed=completed, numerical_failure=numerical_failure,
                unavailable_support=unavailable_support, execution_failure=execution_failure,
                missing=missing, fitted=fitted, rates={
                    "support_rate": dict(numerator=fitted, denominator=planned),
                    "numerical_failure_rate": dict(numerator=numerical_failure,
                                                   denominator=completed+numerical_failure),
                    "completion_rate": dict(numerator=completed, denominator=planned)})


def _run(tmp_path, *, current=False):
    run = tmp_path / "completed_fixture"
    (run / "tasks").mkdir(parents=True)
    (run / "weak").mkdir()
    rows, subjects = [], []
    for dataset in ("fixture_a", "fixture_b", "fixture_c", "fixture_d"):
        for subject in range(3):
            for window in range(2):
                value = .1 + subject*.2 + window*.1
                rows.append(dict(task_id=f"{dataset}_{subject}_{window}", dataset=dataset,
                                 subject=f"s{subject}", record="r0", pair=0, window=window, fold=subject % 2,
                                 parameter_mode="fixed", status="completed", success=True,
                                 hb_only_success=True, eeg_only_success=True, joint_success=True,
                                 selected_candidate="sixth_to_seventh",
                                 D1=value, D4=value/2, D_obs=value*1.5, Delta_spatial=(-1)**subject*value,
                                 time_weight_D4_s=10+window*5, time_weight_Delta_spatial_s=10+window*5,
                                 D_J_E=value/3, D_J_O=value/2, D_J_R=value/2,
                                 time_weight_s=10+window*5))
            for metric in ("D1", "D4", "D_obs", "Delta_spatial"):
                values = [r[metric] for r in rows if r["dataset"] == dataset and r["subject"] == f"s{subject}"]
                subjects.append(dict(dataset=dataset, subject=f"s{subject}", parameter_mode="fixed",
                                     metric=metric, value=np.mean(values), planned=2, available=2))
    # An explicit failed plan item remains in the coverage denominator and has
    # no finite subject score. No penalty or zero is invented by the report.
    rows.append(dict(task_id="failed", dataset="fixture_a", subject="failed_subject", record="r1", pair=0,
                     window=0, fold=0, parameter_mode="fixed", status="failed_integrator", success=False,
                     hb_only_success=False, eeg_only_success=False, joint_success=False,
                     D1=np.nan, D4=np.nan, Delta_spatial=np.nan, time_weight_s=0,
                     time_weight_D4_s=0, time_weight_Delta_spatial_s=0))
    pd.DataFrame(rows).to_csv(run / "metrics.csv", index=False)
    pd.DataFrame(subjects).to_csv(run / "subject_summary.csv", index=False)
    _write(run / "summary.json", dict(status="completed", planned=len(rows), terminal=len(rows),
                                      aggregation="被试内记录等权，记录内窗口时间加权", completed_at="2026-10-28",
                                      missing={"failed_integrator": 1}))
    t = np.arange(60)*.5
    target = np.column_stack([np.sin(t), .2*np.sin(t/4), -.05*np.sin(t/4+.3)])
    mask = np.ones(target.shape, bool)
    mask[10:12, 1:] = False
    sd = np.array([1., .2, .1])
    common = mask[:, 1:].all(axis=1)
    delta = float(np.mean(((target[common, 1:]*.5)/sd[1:])**2-((target[common, 1:]*.25)/sd[1:])**2))
    rows[0].update(Delta_spatial=delta, joint_success=False, joint_status="failed_convergence",
                   status="partial_or_failed_fit", D4=np.nan, D_J_E=np.nan, D_J_O=np.nan, D_J_R=np.nan)
    for item in subjects:
        values = [r.get(item["metric"], np.nan) for r in rows
                  if r["dataset"] == item["dataset"] and r["subject"] == item["subject"]]
        item["value"] = float(np.nanmean(values))
    pd.DataFrame(rows).to_csv(run / "metrics.csv", index=False)
    pd.DataFrame(subjects).to_csv(run / "subject_summary.csv", index=False)
    np.savez_compressed(run / "tasks/fixture.npz", time_s=t, target=target,
                        hb_only_prediction=target*.9, eeg_only_prediction=target*.5,
                        joint_prediction=target*.8, spatial_prediction=target*.75, score_mask=mask, sd=sd,
                        driver_h=np.sin(t)/3, driver_e=np.sin(t)/2, driver_j=np.sin(t)/2.5,
                        spatial_lag_s=np.array([-1., 0., 1.]),
                        spatial_current_loss=np.array([.6, .4, .5]),
                        spatial_alternative_loss=np.array([.4, .3, .6]),
                        observable_standardized_hb_difference=np.full((len(t), 2), .15))
    weak = {}
    for width in (10, 20, 40):
        centers = np.arange(width/2, 100-width/2, 1.)
        prefix = f"pair_0_{width}s_"
        weak.update({prefix+"center_s": centers, prefix+"w": np.sin(centers/4),
                     prefix+"valid": np.ones(len(centers), bool)})
    np.savez_compressed(run / "weak/fixture.npz", **weak)
    example = dict(task_id="fixture_a_0_0", dataset="fixture_a", subject="s0", record="r0", pair=0,
                   window=0, role="median", parameter_mode="fixed", array_path="tasks/fixture.npz",
                   weak_path="weak/fixture.npz", start_s=30., rank=3, selection_denominator=6,
                   hb_unit="合成 Hb 共同坐标", observable_retained_rank=2, observable_numerical_rank=3,
                   hb_only_success=True, eeg_only_success=True, joint_success=False,
                   selected_candidate="sixth_to_seventh", Delta_spatial=delta,
                   spatial_status=dict(success=True, status="completed"),
                   view_statuses={"hb_only": dict(success=True, status="completed"),
                                  "eeg_only": dict(success=True, status="completed"),
                                  "joint": dict(success=False, status="failed_convergence")},
                   observable_scenarios={"primary_ar1_sd0p1": dict(noise_sd=.1, ar1=.9,
                       cross_correlation=.3, retained_rank=2, D_obs=.15)})
    _write(run / "examples.json", dict(selection="D1 中位数／P90／最高有效；尽量不同被试", examples=[example]))
    if current:
        (run / "current_dof/tasks").mkdir(parents=True)
        frame = pd.DataFrame(rows)
        companion = frame[["task_id", "dataset", "subject", "record", "pair", "window", "status", "success"]].copy()
        companion["fixed_task_id"] = companion.task_id
        companion["hb_only_success"] = frame.hb_only_success
        companion["joint_success"] = frame.joint_success
        companion["hb_only_status"] = np.where(frame.hb_only_success, "completed", "failed_convergence")
        companion["joint_status"] = np.where(frame.joint_success, "completed", "failed_integration")
        for name in ("D1", "D4", "D_obs", "Delta_spatial", "D_J_E", "D_J_O", "D_J_R"):
            if name != "Delta_spatial":
                companion[name+"_free"] = frame[name]*.8
        companion["array_path"] = ""
        companion["array_index"] = 0
        companion.loc[companion.task_id == "fixture_a_0_0", "array_path"] = "current_dof/tasks/group.npz"
        companion.to_csv(run / "metrics_current_dof.csv", index=False)
        subject_frame = pd.DataFrame(subjects)
        subject_frame = subject_frame[subject_frame.metric != "Delta_spatial"].copy()
        subject_frame["metric"] += "_free"
        subject_frame["value"] *= .8
        subject_frame.to_csv(run / "subject_summary_current_dof.csv", index=False)
        _write(run / "current_dof/summary.json", dict(status="completed", planned_groups=1, terminal_groups=1,
                                                       planned_windows=len(rows), terminal_windows=len(rows)))
        with np.load(run / "tasks/fixture.npz", allow_pickle=False) as archive:
            group = dict(time_s=archive["time_s"], target=archive["target"][None], sd=archive["sd"],
                         hb_only_prediction=(archive["target"]*.95)[None],
                         joint_prediction=(archive["target"]*.9)[None],
                         hb_only_driver=archive["driver_h"][None], joint_driver=archive["driver_j"][None])
        np.savez_compressed(run / "current_dof/tasks/group.npz", **group)
        summary = json.loads((run / "summary.json").read_text())
        summary["schema"] = "tak_mismatch_summary_v1"
        _write(run / "summary.json", summary)
    _write(run / "weak_verification.json", dict(
        schema="tak_mismatch_weak_independent_verification_v1", status="passed",
        summary=dict(record_pairs=1, width_checks=1, saved_array_comparisons=5, saved_D_bal_exact_matches=1,
                     independent_quadrature_center_checks=3, maximum_independent_quadrature_relative_error=3.4e-13,
                     maximum_reference_D_bal_difference=5e-15, maximum_scale_D_bal_difference=6e-15,
                     valid_cross_segment_kernels=0),
        source_function_exact_match={"weak_balance_kernel": True, "weak_balance_residual": True},
        records=[dict(dataset="fixture_a", record="fixture_a|s0|Probe1", pair="Probe1:CH1", rows=[
            dict(width_s=20, planned_centers=80, valid_centers=77, D_bal=.707)])],
        record_pair_status_audit_20s=dict(width_s=20, by_dataset={"fixture_a": dict(
            planned_record_pairs=7, completed_record_pairs=4,
            status_counts={"completed": 4, "unavailable_data_support": 2, "insufficient_dynamic_excitation": 1},
            noncomplete=[dict(record="fixture_a|s9|Probe1", pair="Probe1:CH2", status="unavailable_data_support",
                              valid_centers=0, planned_centers=20, mean_denominator=None),
                         dict(record="fixture_a|s9|Probe2", pair="Probe2:CH2", status="unavailable_data_support",
                              valid_centers=0, planned_centers=20, mean_denominator=None),
                         dict(record="fixture_a|s8|r0", pair="CH3", status="insufficient_dynamic_excitation",
                              valid_centers=20, planned_centers=20, mean_denominator=0.)])})))
    modes = {"fixed": {"views": {"hb_only": _terminal_view(planned=7, completed=6, numerical_failure=1,
                                                            unavailable_support=0, fitted=7)}}}
    if current:
        modes["current_dof"] = {"views": {"hb_only": _terminal_view(planned=7, completed=6, numerical_failure=1,
                                                                  unavailable_support=0, fitted=7)}}
    _write(run / "terminal_status_audit.json", dict(schema="tak_mismatch_terminal_status_audit_v1",
        status="completed", by_dataset={"fixture_a": modes}))
    return run


def test_ecdf_uses_explicit_weights_preserves_negative_spatial_effects():
    x, y = renderer.weighted_ecdf([3, -2, 1, np.nan], [1, 2, 1, 100])
    assert x.tolist() == [-2, 1, 3]
    np.testing.assert_allclose(y, [.5, .75, 1.])
    with pytest.raises(ValueError, match="nonnegative"):
        renderer.weighted_ecdf([1, 2], [1, -1])
    assert not len(renderer.weighted_ecdf([np.nan], [1])[0])


def test_reader_requires_completed_owning_evidence_and_retains_failures(tmp_path):
    run = _run(tmp_path)
    evidence = renderer.load_evidence(run)
    coverage = renderer._coverage(evidence)
    first = coverage["rows"][0]
    assert first[1:4] == [4, 7, 6]
    assert first[4] == "14.29%"
    assert len(renderer._subject_values(evidence, "fixture_a", "D1")) == 3
    summary = json.loads((run / "summary.json").read_text())
    summary["status"] = "incomplete"
    _write(run / "summary.json", summary)
    with pytest.raises(ValueError, match="completed summary"):
        renderer.load_evidence(run)
    summary["status"] = "completed"
    summary["terminal"] -= 1
    _write(run / "summary.json", summary)
    with pytest.raises(ValueError, match="all planned terminal"):
        renderer.load_evidence(run)


def test_reader_rejects_artifact_escape_and_duplicate_subject_statistic(tmp_path):
    run = _run(tmp_path)
    with pytest.raises(ValueError, match="inside the completed run"):
        renderer._path(run, "../outside.npz")
    frame = pd.read_csv(run / "subject_summary.csv")
    pd.concat([frame, frame.iloc[:1]]).to_csv(run / "subject_summary.csv", index=False)
    with pytest.raises(ValueError, match="one value per subject"):
        renderer.load_evidence(run)


def test_report_is_searchable_bitmap_only_fresh_and_read_only(tmp_path):
    run = _run(tmp_path, current=True)
    originals = {p.relative_to(run): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in run.rglob("*") if p.is_file()}
    output = tmp_path / "export_v1"
    checks = renderer.render_tak_mismatch(run, output)
    audit = checks["exports"]["TECHNICAL_REPORT.pdf"]
    assert audit["image_objects"] >= 9
    assert audit["text_outside_page"] == []
    assert audit["full_page_raster_images"] == 0
    assert audit["selectable_characters"] > 1000
    assert all(item["effective_dpi_x"] >= 249 for item in audit["image_placements"])
    assert checks["exports"]["FIGURE_ATLAS.pdf"]["displayed_images"] == audit["displayed_images"]
    with fitz.open(output / "TECHNICAL_REPORT.pdf") as document:
        text = "\n".join(page.get_text() for page in document)
    assert "Tak 单驱动模型" in text
    assert "14.29%" in text
    assert "fixture_d" in text
    assert "负值表示当前近邻更好" in text
    assert "组级 κ、τ 条件" in text
    assert "近正交" in text and "70% 未解释方差" in text
    assert "全量 20 秒核的状态核验" in text
    assert "数值未通过率" in text and "无支持" in text and "执行异常" in text
    assert "未通过数值检查" in text and "failed_convergence" in text and "failed_integration" in text
    assert "不是时移／lag 扫描" in text
    assert "已核对保留 Δ_spatial" in text
    assert "自比较" in text and "不证明空间映射无误差" in text.replace("\n", "")
    assert json.loads((output / "report_evidence.json").read_text())["terminal_status_audit"].endswith("terminal_status_audit.json")
    assert "3.4e-13" in text
    assert "Probe1:CH1" in text
    assert "6" in text and "排序／候选分母" in text
    assert r"\frac" not in text
    assert len(list((output / "formulas").glob("*.tex"))) == 6
    assert len(list((output / "formulas").glob("*.png"))) == 6
    assert json.loads((output / "report_evidence.json").read_text())["metric_recomputation"] is False
    portable = json.loads((output / "content.json").read_text())
    example_sections = [section for section in portable["sections"] if "Hb 对" in section["title"]]
    assert len(example_sections) == 1
    temporal = example_sections[0]["time_diagnostics"]
    assert temporal["scope"] == "selected_example_retained_arrays_only"
    assert len(temporal["components"]) == 12
    assert temporal["spatial"]["effective_scored_seconds"] == pytest.approx(29.)
    assert all(not item["comparable"] for item in temporal["paired_view_comparisons"])
    assert "fixture_a_0_0" in example_sections[0]["blocks"][0]["text"]
    assert {p.relative_to(run): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in run.rglob("*") if p.is_file()} == originals
    original_pdf = (output / "TECHNICAL_REPORT.pdf").read_bytes()
    with pytest.raises(FileExistsError, match="must be fresh"):
        renderer.render_tak_mismatch(run, output)
    assert (output / "TECHNICAL_REPORT.pdf").read_bytes() == original_pdf


def test_weak_plot_does_not_replace_missing_full_support_with_zero(tmp_path):
    run = _run(tmp_path)
    evidence = renderer.load_evidence(run)
    example = evidence["examples"][0]
    arrays = renderer._array_data(evidence, example)
    assert np.isnan(renderer._masked([1., 2.], [False, True])[0])
    assert renderer._weak_series({}, 0, 40) is None
    series = renderer._weak_series(dict(pair_0_40s_center_s=np.arange(3),
                                        pair_0_40s_w=np.array([1., 0., 2.]),
                                        pair_0_40s_valid=np.array([True, False, True])), 0, 40)
    assert np.isnan(series[1][1])
    arrays["spatial_prediction"][:] = np.nan
    spatial = renderer.spatial_time_curve(arrays, example)
    assert spatial["status"] == "unavailable"
    assert "缺失预测" in spatial["reason"]
    assert np.isnan(spatial["values"]).all()


def test_current_companion_requires_terminal_group_evidence_and_matching_targets(tmp_path):
    run = _run(tmp_path, current=True)
    evidence = renderer.load_evidence(run)
    assert set(evidence["subjects"].parameter_mode) == {"fixed", "current_dof"}
    assert "D1_free" not in set(evidence["subjects"].metric)
    example = evidence["examples"][0]
    arrays, row = renderer._attach_current_example(evidence, example, renderer._array_data(evidence, example))
    assert "free_joint_prediction" in arrays
    assert row["D1"] == pytest.approx(.08)
    changed = dict(arrays)
    changed["target"] = changed["target"]+1
    with pytest.raises(ValueError, match="identical observations"):
        renderer._attach_current_example(evidence, example, changed)
    summary = json.loads((run / "current_dof/summary.json").read_text())
    summary["terminal_windows"] -= 1
    _write(run / "current_dof/summary.json", summary)
    with pytest.raises(ValueError, match="completed current_dof"):
        renderer.load_evidence(run)


def test_weak_centers_follow_retained_native_clock_phase():
    clock = np.arange(10)*.1+.05
    series = renderer._weak_series(dict(continuous_hb_time_s=clock,
        pair_0_10s_center_time_s=np.array([.2, .5]), pair_0_10s_center_index=np.array([2, 5]),
        pair_0_10s_w=np.array([1., 2.]), pair_0_10s_valid=np.array([True, True])), 0, 10)
    np.testing.assert_allclose(series[0], [.25, .55])


def test_time_distribution_uses_metric_specific_weights_and_never_repeats_weak_records(tmp_path, monkeypatch):
    run = _run(tmp_path)
    evidence = renderer.load_evidence(run)
    evidence["rows"]["time_weight_D4_s"] = np.where(np.isfinite(evidence["rows"].D4), 3., 0.)
    calls = []
    original = renderer.weighted_ecdf
    def observe(values, weights=None):
        if weights is not None:
            calls.append(np.asarray(weights))
        return original(values, weights)
    monkeypatch.setattr(renderer, "weighted_ecdf", observe)
    renderer.distribution_figure(evidence, "D4", tmp_path/"weighted.png")
    assert len(calls) == 4
    assert all(set(values) <= {0., 3.} for values in calls)
    calls.clear()
    evidence["rows"]["D_bal_10s"] = .2
    evidence["rows"]["time_weight_D_bal_10s_s"] = 100
    renderer.distribution_figure(evidence, "D_bal_10s", tmp_path/"weak.png")
    assert calls == []


def test_weak_verification_keeps_full_metadata_audit_separate_from_sample_recalculation(tmp_path):
    evidence = renderer.load_evidence(_run(tmp_path))
    blocks = renderer.weak_verification_blocks(evidence)
    text = "\n".join(block.get("text", "") for block in blocks)
    assert "1 个记录内 Hb 对" in text
    assert "3 个中心" in text
    assert "3.4e-13" in text
    tables = [block for block in blocks if block["type"] == "table"]
    assert tables[1]["rows"][0][1:3] == [7, 4]
    assert tables[2]["rows"][0][2] == 2
    assert "零有效支持 2" in tables[2]["rows"][0][3]
    assert "mean_denominator 精确为零 1" in tables[2]["rows"][1][3]
    assert "Probe1:CH2, Probe2:CH2" in tables[2]["rows"][0][3]
    assert all(block["source"] == str(evidence["weak_verification_path"]) for block in tables)


def test_terminal_audit_preserves_group_fitted_count_and_excludes_no_support_from_numerical_failure(tmp_path):
    run = _run(tmp_path)
    audit = dict(schema="tak_mismatch_terminal_status_audit_v1", status="completed", by_dataset={
        "fixture_a": {"current_dof": {"views": {
            "hb_only": _terminal_view(planned=7, completed=2, numerical_failure=1,
                                      unavailable_support=4, fitted=3),
            "joint": _terminal_view(planned=7, completed=0, numerical_failure=0,
                                    unavailable_support=7, fitted=0)}}}})
    _write(run / "terminal_status_audit.json", audit)
    evidence = renderer.load_evidence(run)
    blocks = renderer.terminal_status_blocks(evidence)
    tables = [block for block in blocks if block["type"] == "table"]
    assert tables[0]["rows"][0][3:] == [7, 2, 1, 4, 0, 0]
    assert tables[1]["rows"][0][3:] == ["3/7 (42.86%)", "1/3 (33.33%)", "2/7 (28.57%)"]
    assert tables[1]["rows"][1][4] == "0/0（不可用）"
    assert "未进入拟合的无支持窗口" in blocks[1]["text"]
    assert all(table["source"] == str(run / "terminal_status_audit.json") for table in tables)
    audit["by_dataset"]["fixture_a"]["current_dof"]["views"]["hb_only"]["unavailable_support"] = 5
    _write(run / "terminal_status_audit.json", audit)
    with pytest.raises(ValueError, match="partition the planned denominator"):
        renderer.terminal_status_blocks(renderer.load_evidence(run))


def test_terminal_audit_keeps_execution_and_missing_separate_and_checks_rate_denominators(tmp_path):
    run = _run(tmp_path)
    audit = dict(schema="tak_mismatch_terminal_status_audit_v1", status="completed", by_dataset={
        "fixture_a": {"fixed": {"views": {"hb_only": _terminal_view(planned=9, completed=2,
            numerical_failure=1, unavailable_support=3, execution_failure=2, missing=1, fitted=4)}}}})
    _write(run / "terminal_status_audit.json", audit)
    evidence = renderer.load_evidence(run)
    tables = [block for block in renderer.terminal_status_blocks(evidence) if block["type"] == "table"]
    assert tables[0]["rows"][0][3:] == [9, 2, 1, 3, 2, 1]
    assert tables[1]["rows"][0][4] == "1/3 (33.33%)"
    view = audit["by_dataset"]["fixture_a"]["fixed"]["views"]["hb_only"]
    view["rates"]["numerical_failure_rate"]["denominator"] = 9
    _write(run / "terminal_status_audit.json", audit)
    with pytest.raises(ValueError, match="numerical_failure_rate denominator"):
        renderer.terminal_status_blocks(renderer.load_evidence(run))


def test_spatial_time_curve_preserves_common_mask_and_matches_retained_delta(tmp_path):
    evidence = renderer.load_evidence(_run(tmp_path))
    example = evidence["examples"][0]
    arrays = renderer._array_data(evidence, example)
    result = renderer.spatial_time_curve(arrays, example)
    assert result["status"] == "completed"
    assert result["mean"] == pytest.approx(example["Delta_spatial"], abs=1e-15)
    assert result["count"] == int(arrays["score_mask"][:, 1:].all(1).sum())
    assert np.isnan(result["values"][10:12]).all()
    wrong = dict(example, Delta_spatial=example["Delta_spatial"]+.1)
    with pytest.raises(ValueError, match="disagrees with retained Delta_spatial"):
        renderer.spatial_time_curve(arrays, wrong)


@pytest.mark.parametrize("change,reason", [
    ({"Delta_spatial": None}, "无有效保留"),
    ({"eeg_only_success": False}, "当前近邻"),
    ({"spatial_status": {"success": False}}, "所选空间预测"),
])
def test_spatial_time_curve_does_not_display_failed_finite_predictions(tmp_path, change, reason):
    evidence = renderer.load_evidence(_run(tmp_path))
    example = evidence["examples"][0]
    arrays = renderer._array_data(evidence, example)
    result = renderer.spatial_time_curve(arrays, dict(example, **change))
    assert result["status"] == "unavailable" and reason in result["reason"]
    assert np.isnan(result["values"]).all()


def test_nearest6_spatial_zero_is_same_prediction_self_comparison(tmp_path):
    evidence = renderer.load_evidence(_run(tmp_path))
    example = dict(evidence["examples"][0], selected_candidate="nearest6", Delta_spatial=0.)
    arrays = renderer._array_data(evidence, example)
    arrays["spatial_prediction"] = arrays["eeg_only_prediction"].copy()
    result = renderer.spatial_time_curve(arrays, example)
    assert result["mean"] == 0.
    assert not result["values"][np.isfinite(result["values"])].any()
    arrays["spatial_prediction"][:, 1:] += .1
    with pytest.raises(ValueError, match="spatial time-curve mean|same retained prediction"):
        renderer.spatial_time_curve(arrays, example)


def test_prediction_labels_use_view_validation_not_parent_status_or_finite_arrays():
    record = dict(status="failed_exception", view_statuses={
        "hb_only": dict(success=True, status="completed"),
        "joint": dict(success=False, status="failed_integration")})
    validation = renderer._example_validation(record, {}, dict(
        hb_only_success=False, hb_only_status="failed_convergence",
        joint_success=True, joint_status="completed"))
    assert validation["hb_only_prediction"]["passed"] is True
    assert validation["joint_prediction"]["label"] == "未通过数值检查"
    assert validation["free_hb_only_prediction"]["label"] == "未通过数值检查"
    assert validation["free_joint_prediction"]["passed"] is True
    assert "未通过" in renderer._validation_label("联合", "joint_prediction", validation)
    assert renderer._view_validation({}, "joint")["passed"] is None


def test_spatial_selection_counts_outer_fold_sites_instead_of_repeated_windows(tmp_path):
    evidence = renderer.load_evidence(_run(tmp_path))
    table = renderer.spatial_selection_table(evidence)
    assert len(table["rows"]) == 4
    assert all(row[2:4] == [2, 2] for row in table["rows"])
    assert "不是独立被试" in table["caption"]
    duplicate = evidence["rows"].iloc[[0]].copy()
    duplicate["selected_candidate"] = "nearest6"
    evidence["rows"] = pd.concat([evidence["rows"], duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="conflicting retained spatial candidates"):
        renderer.spatial_selection_table(evidence)


def test_residual_plot_contains_hb_only_deh_and_hb_components_on_shared_masks(tmp_path, monkeypatch):
    arrays = dict(time_s=np.arange(4.), target=np.zeros((4, 3)),
                  hb_only_prediction=-np.tile([1., 2., 3.], (4, 1)),
                  joint_prediction=-np.tile([4., 5., 6.], (4, 1)),
                  sd=np.array([1., 2., 3.]), score_mask=np.ones((4, 3), bool))
    arrays["score_mask"][0, 0] = False
    arrays["score_mask"][1, 1] = False
    captured, styles = [], []
    def inspect_figure(fig, path):
        captured.extend([[line.get_ydata().copy() for line in axis.lines[:2]] for axis in fig.axes])
        styles.extend([[line.get_linestyle() for line in axis.lines[:2]] for axis in fig.axes])
        renderer._plotting().close(fig)
        return str(path)
    monkeypatch.setattr(renderer, "_save", inspect_figure)
    renderer.example_residual_figure(arrays, tmp_path/"residual.png")
    assert np.isnan(captured[0][0][0])
    np.testing.assert_allclose(captured[0][0][1:], [1., 1., 1.])
    np.testing.assert_allclose(captured[0][1][1:], [4., 4., 4.])
    assert np.isnan(captured[1][0][1]) and np.isnan(captured[2][0][1])
    assert captured[1][0][0] == 1. and captured[2][0][0] == 1.
    assert captured[1][1][0] == 2.5 and captured[2][1][0] == 2.
    assert all(style[0] != style[1] for style in styles)


def test_hb_units_read_only_declared_prepared_metadata_without_following_raw_paths(tmp_path):
    run = _run(tmp_path)
    (run / "prepared").mkdir()
    metadata = run / "prepared/unit_sample.json"
    _write(metadata, dict(metadata=dict(hb_provenance=dict(native_unit="mmol/L", output_unit="uM",
        source_files=[dict(path="/sealed/not_opened_raw.mat")], transform="retained_cache_bandpass"))))
    verification = json.loads((run / "weak_verification.json").read_text())
    verification["records"][0]["prepared_metadata_path"] = str(metadata)
    _write(run / "weak_verification.json", verification)
    evidence = renderer.load_evidence(run)
    assert renderer.hb_unit_label(evidence, "fixture_a") == "μM"
    assert evidence["hb_provenance"][0]["metadata_path"] == str(metadata)
    blocks = renderer.hb_unit_blocks(evidence)
    assert "窗口参考后" in blocks[0]["text"] and "Hb 单位 × 秒" in blocks[0]["text"]
    assert blocks[1]["rows"][0][1:3] == ["mmol/L", "uM"]
    assert str(metadata) in blocks[1]["source"]


def test_dataset_hb_unit_explanations_do_not_assign_unknown_absolute_concentrations():
    evidence = dict(hb_provenance=[
        dict(dataset="eeg_fnirs_single_trial", metadata_path="single.json", provenance=dict(native_unit="V", output_unit="relative_Hb_repo_approximation")),
        dict(dataset="refed", metadata_path="refed.json", provenance=dict(native_unit="unreported_LABNIRS_export", output_unit="relative:refed")),
        dict(dataset="visual_cognitive_motivation", metadata_path="visual.json", provenance=dict(native_unit="unreported_ETG7100_export", output_unit="relative:visual_cognitive_motivation")),
        dict(dataset="simultaneous_eeg_nirs", metadata_path="sim.json", provenance=dict(native_unit="mmol/L", output_unit="uM")),
    ])
    assert renderer.hb_unit_label(evidence, "eeg_fnirs_single_trial") == "近似 MBLL 相对 Hb"
    assert renderer.hb_unit_label(evidence, "refed") == "发布相对单位"
    assert renderer.hb_unit_label(evidence, "visual_cognitive_motivation") == "发布相对单位"
    assert renderer.hb_unit_label(evidence, "simultaneous_eeg_nirs") == "μM"
    blocks = renderer.hb_unit_blocks(evidence)
    descriptions = [row[3] for row in blocks[1]["rows"]]
    assert "近似 MBLL" in descriptions[0]
    assert all("绝对单位未知" in text for text in descriptions[1:3])
    assert "缓存已把已知 mmol/L 转为 μM" in descriptions[3]
    assert "不是上述原缓存单位下已知的绝对 Hb 浓度" in blocks[-1]["text"]


def test_temporal_residual_bins_duration_peak_and_thirds_are_reproducible():
    time = np.arange(300)*.1
    mask = np.zeros(300, bool); mask[50:299] = True
    values = np.full(300, np.nan); values[mask] = 1.; values[90] = -4.
    result = renderer._time_series_diagnostics(time, values, mask)
    assert result["scored_bins"] == result["finite_bins"] == 249
    assert result["effective_scored_seconds"] == pytest.approx(24.9)
    assert result["effective_scored_seconds"] != pytest.approx(time[298]-time[50])
    assert result["rms"] == pytest.approx(np.sqrt(264/249))
    assert result["signed_peak"] == -4. and result["peak_time_s"] == pytest.approx(9.)
    assert [third["scored_bins"] for third in result["thirds"]] == [83, 83, 83]
    assert result["dominant_thirds"] == [0]
    assert result["thirds"][0]["standardized_square_share"] == pytest.approx(98/264)
    assert result["thirds"][0]["rms"] == pytest.approx(np.sqrt(98/83))


def test_temporal_zero_and_missing_prediction_do_not_get_false_dominant_intervals():
    time = np.arange(9)*.1
    zero = renderer._time_series_diagnostics(time, np.zeros(9), np.ones(9, bool))
    assert zero["rms"] == 0. and zero["peak_time_s"] is None
    assert zero["dominant_thirds"] == []
    assert all(item["standardized_square_share"] is None for item in zero["thirds"])
    mask = np.array([True, False, False, True, True, True, True, True, True])
    uneven = renderer._time_series_diagnostics(time, np.ones(9), mask)
    assert [item["scored_bins"] for item in uneven["thirds"]] == [1, 3, 3]
    np.testing.assert_allclose([item["standardized_square_share"] for item in uneven["thirds"]], [1/7, 3/7, 3/7])
    assert all(item["rms"] == 1. for item in uneven["thirds"])
    assert uneven["dominant_thirds"] == [1, 2]
    values = np.ones(9); values[4] = np.nan
    missing = renderer._time_series_diagnostics(time, values, mask)
    assert missing["scored_bins"] == 7 and missing["finite_bins"] == 6
    assert missing["rms"] is None and missing["dominant_thirds"] == []
    assert missing["effective_scored_seconds"] == pytest.approx(.7)


def test_temporal_failed_joint_has_diagnostic_numbers_without_valid_improvement_claim():
    arrays = dict(time_s=np.arange(6)*.1, target=np.zeros((6, 3)), sd=np.ones(3),
                  hb_only_prediction=-np.ones((6, 3)), joint_prediction=np.zeros((6, 3)),
                  score_mask=np.ones((6, 3), bool))
    validation = {"hb_only_prediction": dict(passed=True, label="通过数值检查", raw_status="completed"),
                  "joint_prediction": dict(passed=False, label="未通过数值检查", raw_status="failed_convergence")}
    diagnostics = renderer.example_time_diagnostics(arrays, validation, dict(status="unavailable", reason="no spatial support"))
    assert all(not item["comparable"] for item in diagnostics["paired_view_comparisons"])
    joint = [item for item in diagnostics["components"] if item["prediction"] == "joint_prediction"]
    assert all(item["rms"] == 0. and item["validated"] is False for item in joint)
    blocks = renderer.example_time_blocks(diagnostics, "synthetic arrays")
    assert "仅作失败／未认证诊断" in blocks[0]["text"]
    assert "不作联合改善结论" in blocks[0]["text"]
    validation["joint_prediction"] = dict(passed=True, label="通过数值检查", raw_status="completed")
    valid = renderer.example_time_diagnostics(arrays, validation, dict(status="unavailable", reason="no spatial support"))
    assert all(item["comparable"] and item["joint_minus_hb_mean_square"] == -1. for item in valid["paired_view_comparisons"])


def test_spatial_temporal_sign_fractions_use_effective_bins_and_preserve_self_comparison():
    arrays = dict(time_s=np.arange(3)*.1, target=np.zeros((3, 3)), sd=np.ones(3), score_mask=np.ones((3, 3), bool))
    spatial = dict(status="completed", reason="verified", values=np.array([2., -1., 0.]), mean=1/3,
                   Delta_spatial=1/3, selected_candidate="sixth_to_seventh")
    diagnostics = renderer.example_time_diagnostics(arrays, {}, spatial)
    summary = diagnostics["spatial"]
    assert summary["effective_scored_seconds"] == pytest.approx(.3)
    assert [summary[key] for key in ("positive_bins", "negative_bins", "zero_bins")] == [1, 1, 1]
    assert summary["positive_time_fraction"] == summary["negative_time_fraction"] == summary["zero_time_fraction"] == 1/3
    assert summary["maximum_gain"] == dict(value=2., time_s=0.)
    assert summary["maximum_cost"]["value"] == -1. and summary["maximum_cost"]["time_s"] == pytest.approx(.1)
    spatial.update(values=np.zeros(3), mean=0., Delta_spatial=0., selected_candidate="nearest6")
    zero = renderer.example_time_diagnostics(arrays, {}, spatial)
    assert zero["spatial"]["zero_time_fraction"] == 1.
    assert zero["spatial"]["maximum_gain"] is None and zero["spatial"]["maximum_cost"] is None
    blocks = renderer.example_time_blocks(zero, "synthetic")
    assert "同一预测作自比较" in blocks[-1]["text"] and "不证明空间映射无误差" in blocks[-1]["text"]


def test_weak_exclusion_reason_uses_exact_plan_identity_without_changing_verification(tmp_path):
    run = _run(tmp_path)
    original = (run / "weak_verification.json").read_bytes()
    _write(run / "plan.json", dict(summary={}, records=[dict(id="fixture_a|s9|Probe1",
        excluded_reason="registered_fixture_exclusion", reason="wrong_generic_field")]))
    summary = json.loads((run / "summary.json").read_text()); summary["files"] = dict(plan="plan.json")
    _write(run / "summary.json", summary)
    evidence = renderer.load_evidence(run)
    tables = [item for item in renderer.weak_verification_blocks(evidence) if item["type"] == "table"]
    assert "registered_fixture_exclusion (1)" in tables[-1]["rows"][0][3]
    assert "wrong_generic_field" not in tables[-1]["rows"][0][3]
    assert str(run / "plan.json") in tables[-1]["source"]
    assert (run / "weak_verification.json").read_bytes() == original


def test_free_dobs_paired_description_is_window_relaxation_not_group_model_improvement(tmp_path):
    evidence = renderer.load_evidence(_run(tmp_path, current=True))
    evidence['summary']['paired_current_dof'] = {'fixture_a': {'D_obs': dict(
        paired_windows=6, paired_subjects=3, fixed_minus_free_subject_mean=.1,
        subject_bootstrap_95pct=[-.1, .2])}}
    text = renderer._metric_interpretation(evidence, 'D_obs')
    assert '组级解处逐窗放松量' in text
    assert '固定量减逐窗放松量' in text
    assert '不作改善判定' in text
    assert '改善方向仍不确定' not in text
    assert '不解释为组共享模型改善' in text
