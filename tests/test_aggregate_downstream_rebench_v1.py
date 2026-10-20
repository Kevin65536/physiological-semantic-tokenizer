from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from comparative_methods.aggregate_downstream_rebench_v1 import main, summarize, write_summary
from src.metrics.downstream_benchmark import classification_baselines, regression_train_mean_baseline, score_predictions


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def signature(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True, default=str).encode()).hexdigest()


def make_evidence(tmp_path, *, task="motor_imagery", methods=("biot",), seeds=(17, 42)):
    rows = []
    for subject in ("s0", "s1", "s2"):
        for index in range(2):
            rows.append(dict(sample_id=f"{subject}:{index}", subject_id=subject,
                             record_id=f"{subject}:record", task_id=task))
    inventory = dict(tasks={task: dict(classes=[] if task == "refed_regression" else ["a", "b"])}, samples=rows)
    splits = dict(tasks={task: dict(folds=[dict(eval_indices=[0, 1]), dict(eval_indices=[2, 3, 4, 5])])})
    ip, sp = tmp_path / "inventory.json", tmp_path / "splits.json"
    write_json(ip, inventory)
    write_json(sp, splits)
    jobs = []
    for method in methods:
        for fold, indices in enumerate(([0, 1], [2, 3, 4, 5])):
            for seed in seeds:
                job_id = f"{method}-{fold}-{seed}"
                output = tmp_path / "run" / job_id
                job_path = tmp_path / "contracts" / (job_id + ".json")
                job = dict(schema="downstream_rebench_job_v1", job_id=job_id, kind="evaluate", method=method,
                           task=task, track="lp", modality_profile="eeg_only", outer_fold=fold, seed=seed,
                           inventory_path=str(ip), splits_path=str(sp), output_dir=str(output))
                write_json(job_path, job)
                selected = [rows[index] for index in indices]
                if task == "refed_regression":
                    target = np.stack([np.stack((np.arange(20) / 20 + index, np.arange(20) / 10 + index)) for index in indices])
                    prediction = target.copy()
                else:
                    target = np.asarray([index % 2 for index in indices])
                    prediction = target.copy()
                    for index, row in enumerate(selected):
                        if row["subject_id"] == "s2" or (row["subject_id"] == "s1" and seed == 17):
                            prediction[index] = 0
                arrays = dict(target=target, prediction=prediction, mask=np.ones_like(target, dtype=bool),
                              subject_id=np.asarray([row["subject_id"] for row in selected]),
                              record_id=np.asarray([row["record_id"] for row in selected]),
                              sample_id=np.asarray([row["sample_id"] for row in selected]))
                metrics = score_predictions(target, prediction, arrays["mask"], arrays["subject_id"], task, 2, arrays["record_id"])
                baseline = None if task == "refed_regression" else classification_baselines([0, 1, 0, 1], arrays, 2)
                result = dict(schema="downstream_rebench_result_v1", status="completed", job_identity=signature(job),
                              job_id=job_id, method=method, task=task, track="lp", metrics=metrics, baselines=baseline)
                output.mkdir(parents=True)
                np.savez(output / "predictions.npz", **arrays)
                write_json(output / "result.json", result)
                jobs.append(dict(id=job_id, kind="evaluate", command=["python", "worker.py", "--job", str(job_path)],
                                 cwd=str(tmp_path), output_dir=str(output)))
    schedule = dict(schema="downstream_rebench_schedule_v1", run_id="synthetic-non-authorizing", jobs=jobs,
                    unsupported_cells=[dict(method="reve", task="refed_regression", reason="no temporal mask adapter")])
    path = tmp_path / "jobs.json"
    write_json(path, schedule)
    return path, schedule


def amend_result(entry, change):
    path = Path(entry["output_dir"]) / "result.json"
    result = json.loads(path.read_text())
    change(result)
    write_json(path, result)


def amend_predictions(entry, change):
    path = Path(entry["output_dir"]) / "predictions.npz"
    with np.load(path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    change(arrays)
    np.savez(path, **arrays)
    job = json.loads(Path(entry["command"][-1]).read_text())
    metrics = score_predictions(arrays["target"], arrays["prediction"], arrays["mask"],
                                arrays["subject_id"], job["task"], 2, arrays["record_id"])
    amend_result(entry, lambda result: result.update(metrics=metrics))


def test_subject_then_seed_aggregation_bootstrap_and_separate_sd(tmp_path):
    path, _ = make_evidence(tmp_path)
    cell = summarize(path)["cells"][0]
    assert cell["value"] == pytest.approx(2 / 3)
    assert cell["value"] != pytest.approx(0.75)  # fold means would overweight s0
    assert [row["score"] for row in cell["subjects"]] == pytest.approx([1, 2 / 3, 1 / 3])
    assert cell["B0"]["value"] == pytest.approx(5 / 12)
    assert cell["decision"] == "TABLE_READY_WITH_NOTE"
    assert cell["target"]["H"] is None
    assert cell["target"]["source_target"] == "not_applicable"
    assert cell["companion"]["fold_sd"] == pytest.approx(np.std([1, 0.5], ddof=1))
    assert cell["companion"]["seed_sd"] == pytest.approx(np.std([5 / 9, 7 / 9], ddof=1))
    assert cell["uncertainty"]["replicates"] == 2000
    assert cell["uncertainty"]["interval"] == summarize(path)["cells"][0]["uncertainty"]["interval"]


@pytest.mark.parametrize("failure", ("missing", "failed", "interrupted", "shortened_denominator", "mismatched_identity", "wrong_summary"))
def test_failed_planned_unit_never_promotes_success_subset(tmp_path, failure):
    path, schedule = make_evidence(tmp_path)
    entry = schedule["jobs"][0]
    if failure == "missing":
        (Path(entry["output_dir"]) / "result.json").unlink()
    elif failure in {"failed", "interrupted"}:
        amend_result(entry, lambda result: result.update(status=failure, error="synthetic failure"))
    elif failure == "shortened_denominator":
        amend_predictions(entry, lambda arrays: arrays.update({name: values[:1] for name, values in list(arrays.items())}))
    elif failure == "mismatched_identity":
        amend_result(entry, lambda result: result.update(job_identity="unrelated-job"))
    else:
        amend_result(entry, lambda result: result["metrics"].update(primary=0.99))
    summary = summarize(path)
    cell = summary["cells"][0]
    assert cell["planned_job_count"] == 4
    assert cell["completed_job_count"] == 3
    assert cell["value"] is None
    assert cell["table_value"] is None
    assert cell["decision"] == "INVALID_VALUE"
    assert cell["ranking_eligible"] is False
    assert cell["diagnostic_defined_subset_mean"] is not None
    assert cell["uncertainty"]["interval"] is None
    assert summary["failures"]


def test_b0_uses_same_subject_order_and_max_after_aggregation(tmp_path):
    path, schedule = make_evidence(tmp_path)
    for entry in schedule["jobs"]:
        fold = json.loads(Path(entry["command"][-1]).read_text())["outer_fold"]
        def change(result):
            for row in result["baselines"]["subjects"]:
                row.update(majority_macro_f1=0.4 if fold == 0 else 0.1,
                           prior_random_expected_macro_f1=0.2 if fold == 0 else 0.3)
            result["baselines"]["b0"] = 0.99  # never average retained fold-level maxima
        amend_result(entry, change)
    baseline = summarize(path)["cells"][0]["B0"]
    assert baseline["components"]["majority_macro_f1"] == pytest.approx(0.2)
    assert baseline["value"] == pytest.approx(0.8 / 3)


def test_rejected_negative_result_is_retained_without_chance_clipping(tmp_path):
    path, schedule = make_evidence(tmp_path, seeds=(17,))
    for entry in schedule["jobs"]:
        amend_predictions(entry, lambda arrays: arrays.update(prediction=1 - arrays["target"]))
    cell = summarize(path)["cells"][0]
    assert cell["value"] == 0
    assert cell["decision"] == "REJECTED_VALUE"
    assert cell["table_value"] is None
    assert cell["companion"]["seed_sd"] is None
    assert cell["planned_seeds"] == [17]


def test_refed_missing_record_endpoint_stays_undefined_and_mean_baseline_is_missing(tmp_path):
    path, schedule = make_evidence(tmp_path, task="refed_regression", seeds=(17,))
    good = summarize(path)["cells"][0]
    assert good["value"] == pytest.approx(1)
    assert good["B0"]["value"] == 0
    assert good["B0"]["train_mean_errors"]["status"] == "missing"
    assert good["companion"]["mae"] == 0
    amend_predictions(schedule["jobs"][0], lambda arrays: arrays["mask"].__setitem__((slice(None), 1, slice(None)), False))
    bad = summarize(path)["cells"][0]
    assert bad["value"] is None
    assert bad["decision"] == "INVALID_VALUE"
    assert bad["failures"]


def retain_refed_baselines(schedule):
    for entry in schedule["jobs"]:
        with np.load(Path(entry["output_dir"]) / "predictions.npz") as saved:
            arrays = {name: saved[name] for name in saved.files}
        baseline = regression_train_mean_baseline([0.0, 0.0], [100, 100], arrays)
        amend_result(entry, lambda result: result.update(baselines=baseline))


def test_refed_retained_train_mean_baseline_is_subject_equal_and_exported(tmp_path):
    path, schedule = make_evidence(tmp_path, task="refed_regression")
    retain_refed_baselines(schedule)
    # Fold-level summaries must not determine the overall result.
    for entry in schedule["jobs"]:
        amend_result(entry, lambda result: result["baselines"].update(mae=999.0, rmse=999.0))
    summary = summarize(path)
    cell = summary["cells"][0]
    errors = cell["B0"]["train_mean_errors"]
    assert cell["value"] == pytest.approx(1)
    assert errors["status"] == "available"
    assert errors["mae"] == pytest.approx(3.2125)
    assert [row["mae"] for row in errors["subjects"]] == pytest.approx([1.2125, 3.2125, 5.2125])
    assert errors["mae"] != pytest.approx(2.7125)  # unequal fold sizes
    assert errors["rmse"] > errors["mae"]
    output = tmp_path / "summary"
    write_summary(summary, output)
    exported = list(csv.DictReader((output / "cells.csv").open()))[0]
    assert float(exported["train_mean_mae"]) == pytest.approx(3.2125)
    assert exported["train_mean_error_status"] == "available"
    markdown = (output / "summary.md").read_text()
    assert "REFED 原生标签误差" in markdown
    assert "3.2125" in markdown


@pytest.mark.parametrize("tamper", ("error", "mask_count"))
def test_refed_baseline_must_match_retained_native_scoring_support(tmp_path, tamper):
    path, schedule = make_evidence(tmp_path, task="refed_regression", seeds=(17,))
    retain_refed_baselines(schedule)
    def change(result):
        if tamper == "error":
            result["baselines"]["subjects"][0]["mae"] += 100
        else:
            result["baselines"]["subjects"][0]["records"][0]["valid_point_counts"][0] -= 1
    amend_result(schedule["jobs"][0], change)
    cell = summarize(path)["cells"][0]
    assert cell["value"] is None
    assert cell["B0"]["train_mean_errors"]["status"] == "missing"
    assert any("baseline" in failure["reason"] for failure in cell["failures"])


def test_classification_calibration_metadata_does_not_change_sample_axes(tmp_path):
    path, schedule = make_evidence(tmp_path, seeds=(17,))
    for entry in schedule["jobs"]:
        prediction_path = Path(entry["output_dir"]) / "predictions.npz"
        with np.load(prediction_path) as saved:
            arrays = {name: saved[name] for name in saved.files}
        arrays.update(probability=np.tile([0.5, 0.5], (len(arrays["target"]), 1)),
                      class_ids=np.arange(2), score_kind=np.asarray("probability"))
        np.savez(prediction_path, **arrays)
    assert summarize(path)["cells"][0]["value"] is not None


def test_only_explicit_jobs_are_read_and_overlap_track_is_separate(tmp_path, monkeypatch):
    path, schedule = make_evidence(tmp_path, methods=("reve",), seeds=(17,))
    unlisted = tmp_path / "unlisted-legacy-run"
    write_json(unlisted / "result.json", {"must_not_read": True})
    allowed = {str(Path(entry["output_dir"]) / "predictions.npz") for entry in schedule["jobs"]}
    original_load = np.load
    def checked_load(path, **kwargs):
        assert str(path) in allowed
        assert kwargs["allow_pickle"] is False
        return original_load(path, **kwargs)
    monkeypatch.setattr(np, "load", checked_load)
    monkeypatch.setattr(Path, "glob", lambda *args, **kwargs: pytest.fail("directory scanning forbidden"))
    summary = summarize(path)
    assert summary["cells"][0]["reporting_track"] == "open_world_pretrained_with_target_corpus_overlap"
    assert summary["scope"] == "public_development_not_fresh_confirmation"
    assert summary["unsupported_cells"][0]["task"] == "refed_regression"
    assert all(str(unlisted) not in item for item in summary["metadata_inputs"])


def test_cross_model_scoring_support_drift_invalidates_paired_cells(tmp_path):
    path, schedule = make_evidence(tmp_path, methods=("biot", "cbramod", "reve"), seeds=(17,))
    for entry in schedule["jobs"]:
        if entry["id"].startswith("cbramod"):
            amend_predictions(entry, lambda arrays: arrays.update(target=1 - arrays["target"]))
    cells = summarize(path)["cells"]
    assert all(cell["value"] is None for cell in cells)
    assert all(cell["decision"] == "INVALID_VALUE" for cell in cells)


def test_duplicate_planned_fold_seed_is_invalid(tmp_path):
    path, schedule = make_evidence(tmp_path, seeds=(17,))
    schedule["jobs"].append(dict(schedule["jobs"][0], id="duplicate"))
    write_json(path, schedule)
    cell = summarize(path)["cells"][0]
    assert cell["value"] is None
    assert any("grid" in failure["reason"] for failure in cell["failures"])


def test_unreadable_planned_contract_cannot_shrink_the_plan(tmp_path):
    path, schedule = make_evidence(tmp_path, seeds=(17,))
    Path(schedule["jobs"][0]["command"][-1]).unlink()
    summary = summarize(path)
    assert summary["planned_evaluate_job_count"] == 2
    assert summary["cells"][0]["value"] is None
    assert summary["cells"][0]["decision"] == "INVALID_VALUE"
    assert any("complete denominator audit" in row["reason"] for row in summary["failures"])


def test_cli_writes_reviewable_outputs_and_resumes_only_same_schedule(tmp_path):
    path, _ = make_evidence(tmp_path)
    output = tmp_path / "aggregate"
    assert main(["--schedule", str(path), "--output", str(output)]) == 0
    summary = json.loads((output / "summary.json").read_text())
    csv_rows = list(csv.DictReader((output / "cells.csv").open()))
    assert float(csv_rows[0]["value"]) == pytest.approx(summary["cells"][0]["value"])
    assert "不是新的未查看确认集" in (output / "summary.md").read_text()
    assert "2,000" in (output / "summary.md").read_text()
    assert len(summary["source_inputs"]) == 4
    write_summary(summary, output)
    with pytest.raises(ValueError, match="fresh versioned"):
        write_summary({**summary, "schedule_identity": "different-run"}, output)
