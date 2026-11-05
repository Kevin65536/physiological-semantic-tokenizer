"""Development-only diagnostics use actual task support and train-only fits."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from comparative_methods.performance_analysis import audit_downstream_inputs_v1 as audit
from src.metrics.downstream_benchmark import score_predictions


def config():
    return audit.read_config(Path(__file__).parents[1] / "comparative_methods/downstream_input_audit_v1.yaml")


def test_scope_cannot_include_test(tmp_path):
    import yaml
    cfg = config(); cfg["partitions"].append("test")
    path = tmp_path / "bad.yaml"; path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="train/validation only"):
        audit.read_config(path)


def test_followup_reuses_frozen_split_and_has_a_finite_correction_scope():
    path = Path(__file__).parents[1] / "comparative_methods/downstream_input_followup_v1.yaml"
    cfg = audit.read_config(path)
    assert cfg["partitions"] == ["train", "validation"]
    assert cfg["input_cache"] == config()["input_cache"]
    assert cfg["followup"]["brainfusion"]["HRF_step_s"] == .1
    assert cfg["followup"]["refed"]["actual_sequence_steps"] == 30


def test_fixed_probe_predictions_ignore_validation_targets(tmp_path):
    labels = np.tile([0, 1], 30)
    a = dict(target=labels, mask=np.ones(60, bool), sample_id=np.asarray([str(i) for i in range(60)]),
             subject_id=np.repeat(["train_a", "train_b", "validation"], 20), record_id=np.repeat(["a", "b", "c"], 20),
             partition=np.asarray(["train"] * 40 + ["validation"] * 20))
    x = (labels * 2 - 1).reshape(-1, 1).astype(float)
    first = audit.linear_grid(x, a, "motor_imagery", config(), tmp_path, "first", fixed=True)
    assert first["selected"]["validation"]["primary"] == 1
    changed = deepcopy(a); changed["target"][40:] = 1 - changed["target"][40:]
    second = audit.linear_grid(x, changed, "motor_imagery", config(), tmp_path, "second", fixed=True)
    assert second["selected"]["validation"]["primary"] == 0
    with np.load(tmp_path / "first_predictions.npz") as p, np.load(tmp_path / "second_predictions.npz") as q:
        np.testing.assert_array_equal(p["prediction"], q["prediction"])
        np.testing.assert_array_equal(p["train_prediction"], q["train_prediction"])


def test_crop_respects_event_anchor_and_erp_removes_only_available_prefix():
    native = dict(channel_names=["C3", "C4"], eeg_geometry=[{}, {}], window_start_s=-5., duration_s=30.)
    eeg = np.full((1, 2, 6000), 3., dtype=np.float32)
    eeg[..., 1000:1200] = 3 + np.repeat(np.arange(10), 20)
    a = dict(eeg=eeg, fnirs=np.ones((1, 2, 300)), eeg_mask=np.ones_like(eeg, bool), fnirs_mask=np.ones((1, 2, 300), bool),
             target=np.array([0]), mask=np.array([True]), sample_id=np.array(["one"]), subject_id=np.array(["s"]), record_id=np.array(["r"]))
    scaler = {m: dict(mean=[[0.], [0.]], scale=[[1.], [1.]]) for m in ("eeg", "fnirs")}
    full = audit.ArrayDataset(a, native, native, [0])
    np.testing.assert_allclose(audit.event_voltage_features(full, scaler)[0], np.tile(np.arange(10), 2))
    event = audit.ArrayDataset(a, native, {**native, "window_start_s": 0., "duration_s": 8.}, [0])
    np.testing.assert_array_equal(event[0]["eeg"], eeg[0, :, 1000:2600])
    assert event[0]["fnirs"].shape == (2, 80)
    with pytest.raises(ValueError, match="leaves"):
        audit.ArrayDataset(a, native, {**native, "window_start_s": -6.}, [0])


def test_regression_metrics_preserve_thirty_second_mask_and_record_weighting():
    target = np.stack([np.tile(np.arange(30), (2, 1)), np.tile(np.arange(30), (2, 1))]).astype(float)
    prediction = target + 1
    mask = np.ones_like(target, bool); mask[1, :, 5:] = False
    target[~mask] = np.nan; prediction[~mask] = -99999
    result = score_predictions(target, prediction, mask, ["s", "s"], "refed_regression", 0, ["long", "partial"])
    assert result["coverage_complete"]
    assert result["mae"] == 1 and result["rmse"] == 1
    assert result["subjects"][0]["record_count"] == 2


def test_training_row_coordinates_preserve_l2_linear_fit_without_validation_fit():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(44)
    x = rng.normal(size=(20, 4200)); z = rng.normal(size=(7, 4200)); y = np.arange(20) % 2
    a, b, state = audit.linear_row_coordinates(x, z)
    assert state["kind"] == "training_row_space_equivalent_L2"
    np.testing.assert_allclose(a @ a.T, x @ x.T, rtol=1e-10, atol=1e-9)
    m = LogisticRegression(C=.01, tol=1e-9, max_iter=1000).fit(x, y)
    n = LogisticRegression(C=.01, tol=1e-9, max_iter=1000).fit(a, y)
    np.testing.assert_allclose(n.predict_proba(b), m.predict_proba(z), rtol=1e-6, atol=1e-6)
    a2, _, _ = audit.linear_row_coordinates(x, z * 100)
    np.testing.assert_array_equal(a, a2)


def test_independent_metric_check_uses_subject_weights_and_fixed_missing_classes():
    y = np.r_[np.tile([0, 1], 50), 0]
    p = y.copy(); p[-1] = 1
    subjects = np.array(["large"] * 100 + ["small"])
    mask = np.ones(len(y), bool)
    assert audit.independent_primary(y, p, mask, subjects, subjects, "motor_imagery") == .5
    with pytest.raises(ValueError, match="undefined"):
        audit.independent_primary(np.zeros((1, 2, 30)), np.zeros((1, 2, 30)), np.ones((1, 2, 30), bool),
                                  np.array(["one"]), np.array(["record"]), "refed_regression")
