"""Data-free checks for measured-report summaries and retained-curve audits."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.scripts import render_model_compatibility as report


def test_cached_coverage_reports_actual_units_and_filtering():
    identity = dict(record_token='one', dataset_id='refed', subject_id='s1', record_id='r1')
    frame = pd.DataFrame([dict(identity, length_s=90, status='ok')])
    meta = dict(identity, execution='completed', sample_rate_hz=10.,
        measurement_provenance=dict(output_unit='relative:refed', transform='cache_bandpass_0.01_0.2'))
    blocks = report.coverage_blocks(frame, [identity], {'one': meta}, 'fixture', cached=True)
    text = str(blocks)
    assert 'relative:refed' in text and 'cache_bandpass_0.01_0.2' in text
    assert '0.01–0.2 Hz' in text and '普查无新增项目时间滤波' not in text


def test_cached_report_never_reads_native_parent_panels(tmp_path, monkeypatch):
    import json
    from experiments.scripts import render_source_diagnostics_report as renderer
    run, output = tmp_path / 'run', tmp_path / 'report'
    run.mkdir()
    (run / 'plan.json').write_text(json.dumps(dict(created_at='2026-10-27')))
    (run / 'summary.json').write_text(json.dumps(dict(execution='completed', schema='model_compatibility_census_cached_v2')))
    frame = pd.DataFrame([dict(status='ok')])
    monkeypatch.setattr(report, '_read_csv', lambda *_: pd.DataFrame())
    monkeypatch.setattr(report, 'read_census', lambda *_: (frame, [], {}))
    monkeypatch.setattr(report, 'prepare_hb_examples', lambda *_: [])
    monkeypatch.setattr(report, 'census_distribution_blocks', lambda *_: ([], {}))
    monkeypatch.setattr(report, 'coverage_blocks', lambda *a, **k: [])
    monkeypatch.setattr(report, 'hb_example_blocks', lambda *a, **k: [])
    monkeypatch.setattr(report, 'read_panels', lambda *_: (_ for _ in ()).throw(AssertionError('historical native panels must not be mixed in')))
    def render(path, destination):
        content = json.loads(path.read_text())
        assert '统一带通' in content['title']
        assert '缓存' in str(content['sections'])
        destination.mkdir()
        return dict(output=str(destination))
    monkeypatch.setattr(renderer, 'render_report', render)
    report.build_report(run, output)
    assert json.loads((output / 'report_generation.json').read_text())['historical_panels_included'] is False


def test_full_distribution_weights_do_not_turn_repeated_pairs_into_subjects():
    rows = [("A", "r1", 0, 0.), ("A", "r1", 0, 2.), ("A", "r1", 1, 4.),
            ("A", "r2", 0, 8.), ("B", "r3", 0, 10.)]
    frame = pd.DataFrame(rows, columns=["subject_id", "record_id", "window_id", "nominal_D"])
    frame["dataset_id"], frame["length_s"] = "refed", 90
    weights = report.subject_equal_weights(frame)
    assert np.average(frame.nominal_D, weights=weights) == pytest.approx(7.625)
    assert weights[frame.subject_id == "A"].sum() == pytest.approx(weights[frame.subject_id == "B"].sum())
    duplicated = pd.concat([frame, frame.loc[(frame.record_id == "r1") & (frame.window_id == 0)]], ignore_index=True)
    original = report.ecdf(frame.nominal_D, weights)
    repeated = report.ecdf(duplicated.nominal_D, report.subject_equal_weights(duplicated))
    np.testing.assert_allclose(original, repeated)
    assert original[0][0] == 0  # Exact zero remains part of the distribution.


def test_exemplars_use_fixed_score_full_denominator_and_preserve_subject_diversity():
    rows = []
    for index in range(30):
        rows.append(dict(dataset_id="refed", subject_id=f"S{index % 3}", record_id=f"R{index:02d}",
                         length_s=90, window_id=0, start_s=0, pair_index=0, status="ok",
                         nominal_D=index, best_D=29-index))
    rows.append(dict(rows[0], nominal_D=1000, status="missing_support"))
    frame = pd.DataFrame(rows)
    chosen = report.select_exemplars(frame)
    assert [row["selection_quantile"] for row in chosen] == [.1, .5, .9]
    assert {row["selection_denominator"] for row in chosen} == {30}
    assert len({row["subject_id"] for row in chosen}) == 3
    assert chosen[0]["nominal_D"] == 3
    assert all(row["status"] == "ok" for row in chosen)
    # Changing the fitted/best score cannot affect a fixed-score example choice.
    frame["best_D"] = np.arange(len(frame))*100
    assert [r["record_id"] for r in report.select_exemplars(frame)] == [r["record_id"] for r in chosen]


def test_census_reader_preserves_explicit_failed_records(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    inventory = [dict(record_token="ok", dataset_id="refed", subject_id="S1", record_id="R1"),
                 dict(record_token="bad", dataset_id="refed", subject_id="S2", record_id="R2")]
    import json
    (tmp_path / "record_inventory.jsonl").write_text("".join(json.dumps(r)+"\n" for r in inventory))
    for row in inventory:
        (records / (row["record_token"]+".meta.json")).write_text(json.dumps(dict(row, execution="completed" if row["record_token"] == "ok" else "failed")))
    pd.DataFrame([dict(inventory[0], length_s=90, status="ok", nominal_D=.5, best_D=.2)]).to_csv(records / "ok.csv.gz", index=False)
    frame, planned, metadata = report.read_census(tmp_path)
    assert len(frame) == 1 and len(planned) == 2
    assert metadata["bad"]["execution"] == "failed"
    blocks = report.coverage_blocks(frame, planned, metadata, str(tmp_path))
    assert any("整记录读取" in str(block) for block in blocks)


def test_window_support_denominator_distinguishes_short_records_from_failures():
    inventory = [dict(dataset_id="refed", record_token=name, subject_id=name, record_id=name)
                 for name in ("long", "short", "failed")]
    metadata = {row["record_token"]: dict(row, execution="failed" if row["record_token"] == "failed" else "completed")
                for row in inventory}
    frame = pd.DataFrame([
        dict(inventory[0], length_s=30, status="ok"),
        dict(inventory[1], length_s=30, status="ok"),
        dict(inventory[0], length_s=90, status="ok"),
        dict(inventory[0], length_s=90, status="excluded_alignment_boundary"),
    ])
    blocks = report.coverage_blocks(frame, inventory, metadata, "fixture")
    support = next(b for b in blocks if b["type"] == "table" and "有整窗记录" in b["columns"])
    refed = [row for row in support["rows"] if row[0] == "REFED"]
    assert refed == [["REFED", 30, 3, 2, 2, 0], ["REFED", 90, 3, 1, 1, 1]]
    assert any("不能归因为窗长" in b.get("text", "") for b in blocks)


def test_native_example_figure_uses_retained_arrays_without_a_reader(tmp_path):
    examples = tmp_path / "examples"
    examples.mkdir()
    t = np.arange(40)*.25
    positions = np.array([[1., 4.], [2., 4.], [3., 4.], [1., 8.], [2., 8.], [3., 8.]])
    np.savez_compressed(examples / "one.npz", time_s=t, observed=np.column_stack((np.sin(t), -np.sin(t))),
                        kernel_positions=positions, nominal_left=np.ones(6), nominal_right=np.ones(6)*.8,
                        nominal_residual=np.ones(6)*.2, best_residual=np.zeros(6),
                        nominal_D=[.4], best_D=[0.], best_tau=[2.], taus=[.5, 2., 8.], profiles=[1., 0., 2.])
    choice = dict(arrays="one.npz", dataset_id="refed", subject_id="S1", record_id="R1", pair="C1",
                  selection_quantile=.5, selection_denominator=10, start_s=0, stop_s=10, pair_index=0, samples=40)
    blocks = report.hb_example_blocks(tmp_path, [choice], tmp_path)
    images = [Path(block["path"]) for block in blocks if block["type"] == "figure"]
    assert len(images) == 1 and images[0].is_file()
    from PIL import Image
    with Image.open(images[0]) as image:
        assert image.width >= 1800 and image.height >= 1700


def test_missing_and_negative_ecdf_weights_are_explicit():
    np.testing.assert_allclose(report.ecdf([0, np.nan, 1], [1, 10, 1]), ([0, 1], [.5, 1]))
    with pytest.raises(ValueError, match="nonnegative"):
        report.ecdf([0, 1], [1, -1])
    assert report.number(np.nan) == "N/A"


def test_displayed_curve_audit_respects_original_training_scale_and_mask():
    target = np.array([[2., 6.], [400., 800.]])
    prediction = np.zeros_like(target)
    mask = np.array([[True, True], [False, False]])
    residual = report.audit_prediction(target, prediction, [2., 3.], mask, 2.5, "fixture")
    np.testing.assert_allclose(residual[0], [1., 2.])
    with pytest.raises(ValueError, match="retained NMSE"):
        report.audit_prediction(target, prediction, [2., 3.], mask, 0., "tampered")
    with pytest.raises(ValueError, match="shape"):
        report.audit_prediction(target, prediction, [2., 3.], mask.astype(int), 2.5, "mask")


def test_local_curve_description_ignores_unscored_boundary_predictions():
    t = np.arange(40)*.25
    residual = np.r_[np.ones(20), np.ones(19)*2, 1e9]
    mask = np.ones(40, bool)
    mask[-1] = False
    peak = report.peak_local_rms(t, residual, mask)
    assert peak == dict(start_s=5., stop_s=10., rms=2., count=19)
