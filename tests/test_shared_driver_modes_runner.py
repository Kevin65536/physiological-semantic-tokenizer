"""Public metadata and synthetic fixtures for the versioned mode runner."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.scripts import evaluate_shared_driver_modes as runner
from src.inference.t3a_balloon_robust_ssm import BalloonParameters


@pytest.fixture
def cfg():
    return runner.read_config(runner.DEFAULT_CONFIG)


@pytest.mark.parametrize("keys,value", [
    (("source_run",), "comparative_methods/protected"),
    (("split_run",), "experiments/runs/other_split"),
    (("protected_data",), "allowed"),
    (("data_boundary",), "all_public_data"),
    (("tensor", "eeg_features"), 29),
    (("tensor", "feature_order"), "band_major_channel_minor"),
    (("tensor", "temporal_modes"), 25),
    (("measured", "hidden_EEG_channels"), [1, 3, 5]),
    (("measured", "shift_steps"), 47),
    (("calibration", "training_windows"), 19),
    (("calibration", "selection_windows"), 5),
    (("model", "vascular_projection"), "unnormalized_a_plus_gamma_b"),
    (("model", "ar_phi"), 1.),
    (("model", "state_sd"), .1),
    (("model", "initial_weight"), 0.),
    (("solver", "max_nfev"), 0),
])
def test_config_rejects_source_information_mask_and_prior_drift(cfg, tmp_path, keys, value):
    changed = deepcopy(cfg)
    owner = changed
    for key in keys[:-1]:
        owner = owner[key]
    owner[keys[-1]] = value
    path = tmp_path/"bad.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError):
        runner.read_config(path)


def _ref(parent, identity="sample", subject="train", window=0, site="frontal", **extra):
    ref = dict(id=identity, dataset="eeg_fnirs_single_trial", subject=subject,
        record="record_"+subject, key="key_"+subject, window=window, site=site,
        region=site, hb_channel=site+"_Hb", eeg_channels=list(range(6)),
        eeg_start_s=float(window*30), hb_start_s=float(window*30),
        array_path=str(parent/"prepared"/"fixture.npz"), array_index=window,
        subject_fold=0, task="fixture", condition="condition")
    ref.update(extra)
    return ref


@pytest.mark.parametrize("case", ["protected", "excluded", "outside", "traversal", "symlink"])
def test_payload_ref_rejection_occurs_before_np_load(tmp_path, monkeypatch, case):
    parent = tmp_path/"parent"
    (parent/"prepared").mkdir(parents=True)
    ref = _ref(parent)
    if case == "protected":
        ref["dataset"] = "protected_test_split"
    elif case == "excluded":
        ref.update(dataset="visual_cognitive_motivation", subject="S06", record="S06_Part1_Task")
    elif case == "outside":
        ref["array_path"] = str(tmp_path/"outside.npz")
    elif case == "traversal":
        ref["array_path"] = str(parent/"prepared"/".."/".."/"outside.npz")
    else:
        outside = tmp_path/"outside.npz"
        outside.write_bytes(b"not an array")
        link = parent/"prepared"/"escape.npz"
        link.symlink_to(outside)
        ref["array_path"] = str(link)
    reads = []
    def forbidden_load(*args, **kwargs):
        reads.append(args)
        pytest.fail("scope must be rejected before array read")
    monkeypatch.setattr(runner.np, "load", forbidden_load)
    with pytest.raises(ValueError):
        runner.raw_target(ref, parent)
    assert not reads


def test_region_split_maps_native_identity_instead_of_order_and_retains_missing(tmp_path):
    parent = tmp_path/"parent"
    source = [_ref(parent, "source1", window=1), _ref(parent, "source0", window=0)]
    region = [_ref(parent, "region0", window=0, site="motor"),
              _ref(parent, "region1", window=1, site="motor")]
    mapped, missing = runner.region_split_ids(["source1", "source0"], region+source, "motor")
    assert mapped == ["region1", "region0"] and not missing
    mapped, missing = runner.region_split_ids(["source1", "source0"], region[:1]+source, "motor")
    assert mapped == ["region0"] and missing[0]["source_id"] == "source1"
    duplicate = dict(region[1], id="ambiguous")
    mapped, missing = runner.region_split_ids(["source1"], source+region+[duplicate], "motor")
    assert not mapped and missing[0]["reason"] == "regional_identity_not_unique_or_missing"
    misaligned = dict(region[1], hb_start_s=31.)
    mapped, missing = runner.region_split_ids(["source1"], source+[misaligned], "motor")
    assert not mapped and len(missing) == 1


@pytest.mark.parametrize("mode", runner.MODES)
def test_pairing_shift_and_real_controls_have_identical_nonwrap_scoring_support(cfg, mode):
    y = np.arange(120*32, dtype=float).reshape(120, 32)
    real, real_mask, real_endpoint = runner.pairing_target(cfg, y, mode, "real_shift_support")
    shifted, shifted_mask, shifted_endpoint = runner.pairing_target(cfg, y, mode, "nonwrapping_shift_12s")
    np.testing.assert_array_equal(real_mask, shifted_mask)
    np.testing.assert_array_equal(real_endpoint, shifted_endpoint)
    assert not real_endpoint[:48].any()
    assert not shifted_mask[:48, :30].any()
    assert np.isnan(real[:48, :30]).all() and np.isnan(shifted[:48, :30]).all()
    np.testing.assert_array_equal(real[48:, :30], y[48:, :30])
    np.testing.assert_array_equal(shifted[48:, :30], y[:-48, :30])
    np.testing.assert_array_equal(shifted[:, 30:], y[:, 30:])
    original = runner.visible_mask(mode, cfg)
    np.testing.assert_array_equal(shifted_endpoint[48:], runner.endpoint_mask(mode, original)[48:])


def test_hidden_EEG_channel_mask_preserves_channel_band_identity(cfg):
    mask = runner.visible_mask("EEG_channels_hidden", cfg)
    for channel in range(6):
        np.testing.assert_array_equal(mask[:, channel*5:(channel+1)*5], channel not in (0, 2, 4))
    assert mask[:, 30:].all()
    assert runner.endpoint_mask("EEG_channels_hidden", mask).sum() == 120*15
    with pytest.raises(ValueError, match="unregistered"):
        runner.visible_mask("future_mask", cfg)


def test_interpolation_uses_only_visible_values_and_training_template(cfg):
    rng = np.random.default_rng(11)
    y, template = rng.normal(size=(120, 32)), rng.normal(size=(120, 32))
    mask = runner.visible_mask("center_Hb", cfg)
    mask[:, :5] = False
    reference = runner.interpolation_inputs(y, mask, template)
    np.testing.assert_array_equal(reference[:, :5], template[:, :5])
    for poison in (1e30, np.nan, np.inf):
        changed = y.copy()
        changed[~mask] = poison
        np.testing.assert_array_equal(runner.interpolation_inputs(changed, mask, template), reference)
    np.testing.assert_array_equal(reference[mask], y[mask])


def test_score_balances_modalities_preserves_native_scales_and_empty_endpoints():
    target = np.zeros((120, 32))
    sd = np.r_[np.full(30, 3.), 2., 4.]
    prediction = np.broadcast_to(sd*np.r_[np.full(30, 2.), 4., 4.], target.shape).copy()
    score = runner.score_prediction(prediction, target, sd, np.ones_like(target, bool),
        physical=np.zeros_like(target), component=prediction, eeg_factor=2., hb_factor=4.)
    assert score["EEG_nrmse"] == 2. and score["Hb_nrmse"] == 4.
    assert score["total_nrmse"] == pytest.approx(np.sqrt(10.))
    assert score["EEG_rmse_native_feature"] == 3.
    assert score["Hb_rmse_native_feature"] == pytest.approx(np.sqrt((2.**2+4.**2)/2))
    assert score["physical_Hb_nrmse"] == 0.
    assert score["common_Hb_RMS_training_SD"] == 4.
    endpoint = np.zeros_like(target, bool)
    endpoint[50:70, 30] = True
    single = runner.score_prediction(prediction, target, sd, endpoint)
    assert single["EEG_nrmse"] is None and single["HbR_nrmse"] is None
    assert single["total_nrmse"] == 4. and single["scored_HbO_points"] == 20
    empty = runner.score_prediction(prediction, target, sd, np.zeros_like(endpoint))
    assert empty["total_nrmse"] is None


def test_paired_scores_require_both_finite_endpoints_and_keep_failure_denominator():
    import pandas as pd
    rows = []
    for i, (arm_value, control_value, arm_converged) in enumerate([
            (.5, 1., True), (.2, np.nan, True), (None, 1., True), (.2, 1., False), (.25, .75, True)]):
        for arm, value, converged in (("R2_selected", arm_value, arm_converged),
                                       ("R1_broadband", control_value, True)):
            rows.append(dict(id=f"w{i}", subject=f"s{i}", mode="Hb_hidden", arm=arm,
                converged=converged, Hb_nrmse=value))
    compared = runner.paired_rows(pd.DataFrame(rows), controls=["R1_broadband"], groups=["mode"])
    assert len(compared) == 1
    result = compared[0]
    assert result["planned_pairs"] == 5 and result["common_success_pairs"] == 2
    assert result["failed_or_invalid_pairs"] == 3 and result["subjects"] == 2
    assert result["gain"] == pytest.approx(.5)


def test_mode_recovery_centers_by_fixed_reference_but_retains_DC_diagnostics():
    t = np.arange(120.)
    truth = dict(r=np.column_stack((np.sin(t/20), np.cos(t/17))), u_h=np.sin(t/25))
    fitted = dict(r=truth["r"]+np.array([3., -4.]), u_h=truth["u_h"]+5.)
    result = runner.recovery_metrics(fitted, truth, reference_steps=20)
    for mode, offset in (("a", 3.), ("b", -4.), ("u_H", 5.)):
        assert result[mode+"_reference_centered_RMSE"] < 1e-14
        assert result[mode+"_raw_RMSE"] == pytest.approx(abs(offset))
    assert result["a_reference_mean_error"] == pytest.approx(3.)
    assert result["u_H_reference_mean_error"] == pytest.approx(5.)
    scalar = dict(r=fitted["r"][:, :1], u_h=fitted["u_h"])
    assert "b_raw_RMSE" not in runner.recovery_metrics(scalar, truth)
    assert runner.recovery_metrics({}, truth) == {}


@pytest.fixture
def parent_metadata(cfg, tmp_path, monkeypatch):
    parent, split = tmp_path/cfg["source_run"], tmp_path/cfg["split_run"]
    panel = [_ref(parent, f"eval_{s:02d}_{w}", subject=f"E{s:02d}", window=w)
             for s in range(18) for w in range(4)]
    train = [_ref(parent, f"train_{i:02d}", subject="TRAIN", window=i) for i in range(18)]
    selection = [_ref(parent, f"selection_{i}", subject="SELECT", window=i) for i in range(6)]
    key = "eeg_fnirs_single_trial__frontal__outer0"
    meta = {
        parent/"summary.json": dict(execution="completed"),
        split/"summary.json": dict(execution="completed"),
        parent/"cohort.json": dict(refs=panel+train+selection),
        parent/"diagnostic_plan.json": dict(windows=panel),
        split/"measured_plan.json": dict(windows=deepcopy(panel)),
        split/"calibration"/(key+".json"): dict(training_ids=[r["id"] for r in train],
            selection_ids=[r["id"] for r in selection], training_subjects=["TRAIN"], selection_subjects=["SELECT"]),
        parent/"coordinates"/(key+".json"): dict(key=key, training_subjects=["TRAIN"],
            training_windows=18, hb_factor=1., pc=(np.ones(30)/np.sqrt(30)).tolist(), sd=[1., 1., 1.]),
    }
    monkeypatch.setattr(runner, "read_json", lambda path: deepcopy(meta[Path(path)]))
    def forbidden_load(*args, **kwargs):
        pytest.fail("metadata planning must not read payload arrays")
    monkeypatch.setattr(runner.np, "load", forbidden_load)
    return dict(root=tmp_path, parent=parent, split=split, meta=meta, panel=panel,
        train=train, selection=selection, key=key)


def test_parent_plan_preserves_public_panel_and_frozen_subject_splits(cfg, parent_metadata):
    fixture = parent_metadata
    plan = runner.parent_plan(cfg, fixture["root"])
    assert plan["original_windows"] == 72 and plan["expanded_windows"] == 72
    assert [item["ref"]["id"] for item in plan["windows"]] == [r["id"] for r in fixture["panel"]]
    assert len(plan["calibration"]) == 1
    cal = plan["calibration"][0]
    assert cal["training_ids"] == [r["id"] for r in fixture["train"]]
    assert cal["selection_ids"] == [r["id"] for r in fixture["selection"]]
    assert cal["training_subjects"] == ["TRAIN"] and cal["selection_subjects"] == ["SELECT"]
    assert cal["outer_evaluation_subjects"] == [f"E{s:02d}" for s in range(18)]
    assert cal["status"] == "eligible"


@pytest.mark.parametrize("case", ["coordinate_leak", "train_leak", "selection_overlap", "panel_order", "duplicate_id", "nonterminal"])
def test_parent_metadata_rejects_leaks_and_identity_drift_before_payload(cfg, parent_metadata, case):
    fixture = parent_metadata
    meta, parent, split, key = fixture["meta"], fixture["parent"], fixture["split"], fixture["key"]
    if case == "coordinate_leak":
        meta[parent/"coordinates"/(key+".json")]["training_subjects"] = ["E00"]
    elif case == "train_leak":
        fixture["train"][0]["subject"] = "E00"
    elif case == "selection_overlap":
        fixture["selection"][0]["subject"] = "TRAIN"
    elif case == "panel_order":
        meta[split/"measured_plan.json"]["windows"].reverse()
    elif case == "duplicate_id":
        meta[parent/"cohort.json"]["refs"].append(deepcopy(fixture["train"][0]))
    else:
        meta[parent/"summary.json"]["execution"] = "running"
    with pytest.raises(ValueError):
        runner.parent_plan(cfg, fixture["root"])


@pytest.mark.parametrize("retained_training", [17, 0])
def test_parent_plan_keeps_missing_regional_training_support_ineligible(cfg, parent_metadata, retained_training):
    fixture = parent_metadata
    meta, parent = fixture["meta"], fixture["parent"]
    refs = meta[parent/"cohort.json"]["refs"]
    # One exact synchronous evaluation region with partial or entirely absent
    # regional training support; either remains ineligible without admitting
    # subjects outside the frozen partition or blocking unrelated regions.
    for ref in fixture["panel"][:1]+fixture["train"][:retained_training]+fixture["selection"]:
        refs.append(dict(ref, id=ref["id"]+"_motor", site="motor", region="motor", hb_channel="motor_Hb"))
    key = "eeg_fnirs_single_trial__motor__outer0"
    meta[parent/"coordinates"/(key+".json")] = dict(key=key, training_subjects=["TRAIN"],
        training_windows=18, hb_factor=1., pc=(np.ones(30)/np.sqrt(30)).tolist(), sd=[1., 1., 1.])
    plan = runner.parent_plan(cfg, fixture["root"])
    assert plan["expanded_windows"] == 73
    cal = next(c for c in plan["calibration"] if c["site"] == "motor")
    assert cal["status"] == "ineligible_missing_regional_training_support"
    assert len(cal["training_ids"]) == retained_training and len(cal["selection_ids"]) == 6
    assert cal["missing"][0]["source_id"] == f"train_{retained_training:02d}"
    assert len(cal["missing"]) == 18-retained_training
    assert plan["regional_exclusions"] == cal["missing"]


def test_missing_regional_calibration_records_failure_without_payload_reads(cfg, tmp_path, monkeypatch):
    def forbidden_target(*args, **kwargs):
        pytest.fail("ineligible calibration must not read arrays")
    monkeypatch.setattr(runner, "raw_target", forbidden_target)
    spec = dict(key="missing_motor", status="ineligible_missing_regional_training_support")
    result = runner.calibration_worker((cfg, tmp_path, tmp_path/"run", spec, [], tmp_path/"parent"))
    assert result["status"] == "failed_missing_training_support"
    assert result["key"] == "missing_motor" and result["seconds"] == 0.
    retained = runner.read_json(tmp_path/"run"/"calibration"/"missing_motor.json")
    assert retained == result


def test_arm_adapter_preserves_equal_observation_and_prior_contracts(cfg, monkeypatch):
    captured = []
    def fit(eeg, hb, p, dt, **kwargs):
        captured.append((eeg, hb, dt, kwargs))
        return dict(marker=True)
    monkeypatch.setattr(runner, "fit_shared_driver_modes", fit)
    target = np.zeros((120, 32))
    cal = dict(pc=np.ones(30)/np.sqrt(30), sd=np.ones(32), component_sd=np.arange(1., 5.), selected_gamma=-.5)
    mask = runner.visible_mask("center_Hb", cfg)
    for arm in runner.ARMS:
        assert runner.fit_arm(cfg, object(), target, cal, arm, mask)["marker"]
    expected_dimensions = [1, 1, 2, 2, 2]
    for arm, dimensions, (e, h, dt, kwargs) in zip(runner.ARMS, expected_dimensions, captured):
        assert e.shape == (120, 30) and h.shape == (120, 2) and dt == .25
        assert kwargs["loadings"].shape == (30, dimensions)
        np.testing.assert_allclose(np.linalg.norm(kwargs["loadings"], axis=0), 1.)
        assert kwargs["temporal_basis"].shape == (120, 24)
        assert kwargs["state_sd"] == .025 and kwargs["ar_weight"] == .1
        assert kwargs["initial_weight"] == 100.
        assert kwargs["independent"] == (arm == "independent")
        assert kwargs["gamma"] == (-.5 if arm == "R2_selected" else 0.)
        np.testing.assert_array_equal(kwargs["component_sd"], cal["component_sd"])
        np.testing.assert_array_equal(kwargs["visible"], mask)
        assert kwargs["hb_basis"].shape == (120, 2, 4)


def test_synthetic_b_only_intervention_and_runner_fit_use_separate_seed_streams(cfg, monkeypatch):
    monkeypatch.setattr(runner, "parameters", lambda *args, **kwargs: BalloonParameters())
    y, truth, calibration = runner.synthetic_case(cfg, 0, "b_only_gamma0", cfg["synthetic"]["evaluation_seed_stream"])
    repeated, repeated_truth, _ = runner.synthetic_case(cfg, 0, "b_only_gamma0", cfg["synthetic"]["evaluation_seed_stream"])
    training, _, _ = runner.synthetic_case(cfg, 0, "b_only_gamma0", cfg["synthetic"]["calibration_seed_stream"])
    np.testing.assert_array_equal(y, repeated)
    np.testing.assert_array_equal(truth["r"], repeated_truth["r"])
    assert not np.array_equal(y, training)
    np.testing.assert_array_equal(truth["r"][:, 0], 0.)
    np.testing.assert_array_equal(truth["u_h"], 0.)
    assert np.sqrt(np.mean(truth["r"][:, 1]**2)) == pytest.approx(.025)
    np.testing.assert_allclose(truth["physical_prediction"][:, 30:], 0., atol=1e-15)
    fit = runner.fit_arm(cfg, BalloonParameters(), y, calibration, "R2_zero", runner.visible_mask("full", cfg))
    assert fit["converged"] and fit["physical_check"]["valid"]
    assert fit["complexity"]["driving_trajectories"] == 2
    score = runner.fit_score(fit, y, calibration, np.ones_like(y, bool))
    assert score["total_nrmse"] is not None and np.isfinite(score["total_nrmse"])
