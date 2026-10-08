"""Public-boundary and information-isolation checks using temporary fixtures."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.scripts import evaluate_semantic_response_discovery as runner


@pytest.fixture
def cfg():
    return runner.read_config(runner.DEFAULT_CONFIG)


def _ref(parent, *, subject="training", record="record", region="motor", window=0,
         dataset="eeg_fnirs_single_trial", site=None):
    key = f"{dataset}__{subject}__{record}"
    site = region if site is None else site
    return dict(id=f"{key}__{site}__w{window}", key=key, dataset=dataset, subject=subject,
                record=record, region=region, site=site, window=window, array_index=window,
                array_path=str(parent / "prepared" / key / (region + ".npz")),
                eeg_channels=[f"channel{i}" for i in range(6)], hb_channel=region + "_Hb",
                eeg_start_s=window * 30., hb_start_s=window * 30., task="task", condition="condition")


@pytest.mark.parametrize("keys,value", [
    (("source_run",), "experiments/runs/private"),
    (("data_boundary",), "any_public_data"),
    (("protected_data",), "allowed"),
    (("tensor", "eeg_features"), 29),
    (("tensor", "feature_order"), "band_major_channel_minor"),
    (("model", "state_sd"), .1),
    (("model", "initial_weight"), 0.),
    (("model", "initial_prior_sd"), [.1, .3, 0., .3]),
    (("model", "random_seeds"), [1, 1, 2]),
    (("measured", "prefix_steps"), [40, 20]),
    (("measured", "score_start"), 40),
    (("measured", "pairings"), ["real"]),
    (("uncertainty", "minimum_training_sd_fraction"), 0.),
    (("solver", "numerical_backend"), "numpy"),
    (("resources", "max_workers"), 49),
])
def test_config_rejects_boundary_tensor_information_and_prior_drift(cfg, tmp_path, keys, value):
    changed = deepcopy(cfg)
    current = changed
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError):
        runner.read_config(path)


@pytest.mark.parametrize("case", ["protected", "excluded", "outside", "traversal", "symlink", "identity", "channel"])
def test_public_payload_boundary_rejects_before_numpy_load(tmp_path, monkeypatch, case):
    parent = tmp_path / "parent"
    ref = _ref(parent)
    if case == "protected":
        ref["dataset"] = "protected_test_split"
    elif case == "excluded":
        ref = _ref(parent, dataset="visual_cognitive_motivation", subject="S06", record="S06_Part1_Probe1")
    elif case == "outside":
        ref["array_path"] = str(tmp_path / "outside.npz")
    elif case == "traversal":
        ref["array_path"] = str(parent / "prepared" / ".." / ".." / "outside.npz")
    elif case == "symlink":
        outside = tmp_path / "outside.npz"
        outside.write_bytes(b"not an array")
        path = Path(ref["array_path"])
        path.parent.mkdir(parents=True)
        path.symlink_to(outside)
    elif case == "identity":
        ref["id"] = "different_native_window"
    else:
        ref["eeg_channels"] = ["duplicated"] * 6
    calls = []
    def forbidden_load(*args, **kwargs):
        calls.append(args)
        pytest.fail("boundary must reject before payload read")
    monkeypatch.setattr(runner.np, "load", forbidden_load)
    with pytest.raises(ValueError):
        runner.raw_target(ref, parent)
    assert not calls


def test_visual_site_identity_retains_parent_montage_suffix(tmp_path):
    ref = _ref(tmp_path / "parent", dataset="visual_cognitive_motivation", subject="S01",
               record="S01_Part2_Probe1", region="prefrontal", site="prefrontal__montage0")
    assert runner.validate_ref(ref, tmp_path / "parent").name == "prefrontal.npz"
    changed = dict(ref, site="prefrontal__invented")
    with pytest.raises(ValueError, match="region identity"):
        runner.validate_ref(changed, tmp_path / "parent")


def test_public_prepared_shape_support_and_array_identity_are_checked(tmp_path):
    parent = tmp_path / "parent"
    ref = _ref(parent)
    path = Path(ref["array_path"])
    path.parent.mkdir(parents=True)
    rng = np.random.default_rng(4)
    eeg, hb = rng.normal(size=(1, 120, 30)), rng.normal(size=(1, 120, 2))
    np.savez(path, eeg_features=eeg, hb=hb)
    e, h = runner.raw_target(ref, parent)
    np.testing.assert_array_equal(e, eeg[0])
    np.testing.assert_array_equal(h, hb[0])
    runner.prepared_arrays.cache_clear()
    np.savez(path, eeg_features=eeg[:, :, :29], hb=hb)
    with pytest.raises(ValueError, match="shape mismatch"):
        runner.raw_target(ref, parent)
    runner.prepared_arrays.cache_clear()
    hb[0, 119, 0] = np.nan
    np.savez(path, eeg_features=eeg, hb=hb)
    with pytest.raises(ValueError, match="finite support"):
        runner.raw_target(ref, parent)
    runner.prepared_arrays.cache_clear()


def test_training_coordinate_reproduces_common_Hb_rule_without_parent_parameters(cfg):
    rng = np.random.default_rng(14)
    eeg = rng.normal(size=(4, 120, 30)) * np.linspace(.3, 2., 30)
    hb = rng.normal(size=(4, 120, 2)) * [3., 6.] + [1., -2.]
    cal = runner.fit_coordinates(eeg, hb, cfg, {"coordinate": {"common_Hb_training_sd_target": .025}})
    assert cal["hb_factor"] == pytest.approx(.025 / np.sqrt(hb.reshape(-1, 2).var(axis=0).mean()))
    assert np.sqrt(np.mean((eeg * cal["eeg_factor"] @ runner.spectral_loadings()[:, 0]) ** 2)) == pytest.approx(.025)
    assert np.sqrt(np.mean(np.square(cal["sd_hb"]))) == pytest.approx(.025)
    np.testing.assert_allclose(cal["fit_eeg_sd"], cal["sd_eeg"])
    np.testing.assert_allclose(cal["fit_hb_sd"], cal["sd_hb"])
    assert cal["pca_loading"][np.argmax(np.abs(cal["pca_loading"]))] > 0
    assert cal["coordinate_fit"].endswith("no_parent_fold_coordinates")


@pytest.mark.parametrize("prefix", [0, 40])
@pytest.mark.parametrize("poison", [np.nan, np.inf, 1e30])
def test_response_fitter_never_receives_hidden_Hb_values(cfg, monkeypatch, prefix, poison):
    from src.inference import semantic_response_ssm as library
    observed_calls = []
    def fake_fit(hb, driver, p, dt, **kwargs):
        observed_calls.append((hb.copy(), kwargs["visible"].copy(), driver.copy()))
        assert np.isnan(hb[~kwargs["visible"]]).all()
        value = hb[kwargs["visible"]].mean() if kwargs["visible"].any() else 0.
        return dict(prediction=np.full_like(hb, value), success=True, converged=prefix > 0,
                    fit_performed=prefix > 0, status="fit" if prefix else "prior_prediction")
    monkeypatch.setattr(library, "fit_fixed_driver_response", fake_fit)
    hb = np.arange(240.).reshape(120, 2)
    modes = dict(a=np.linspace(0, .03, 120), b=np.zeros(120))
    spec = dict(arm="broad_a", gamma=0., gain=1., tau_n=0., tau_v=0., initial_mode="tied_pv_logprior")
    cal = dict(fit_hb_sd=[.02, .01], component_sd=[.01] * 4)
    first = runner.fit_response(hb, modes, spec, cal, cfg, None, prefix)
    changed = hb.copy()
    changed[prefix:] = poison
    second = runner.fit_response(changed, modes, spec, cal, cfg, None, prefix)
    np.testing.assert_array_equal(first["prediction"], second["prediction"])
    assert len(observed_calls) == 2
    if not prefix:
        assert first["success"] and not first["converged"] and not first["fit_performed"]


@pytest.mark.parametrize("kind", runner.BASELINES[2:])
def test_frozen_linear_inputs_use_only_own_Hb_prefix(kind):
    rng = np.random.default_rng(24)
    eeg, own, spatial = rng.normal(size=(120, 30)), rng.normal(size=(120, 2)), rng.normal(size=(120, 4))
    original = runner.context_features(eeg, own, spatial, 40, kind)
    for poison in (np.nan, np.inf, -1e30):
        changed = own.copy()
        changed[40:] = poison
        np.testing.assert_array_equal(runner.context_features(eeg, changed, spatial, 40, kind), original)
    zero = runner.context_features(eeg, np.full_like(own, np.nan), spatial, 0, kind)
    assert zero.shape[1] == original.shape[1] - 80


def test_spatial_context_reads_other_regions_and_never_target_Hb(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    refs = [_ref(parent, region=region) for region in runner.REGIONS]
    target, reads = refs[1], []
    def fake_raw(ref, parent):
        reads.append(ref["id"])
        return np.zeros((120, 30)), np.ones((120, 2)) * runner.REGIONS.index(ref["region"])
    monkeypatch.setattr(runner, "raw_target", fake_raw)
    values, donors = runner.spatial_target(target, refs, parent, dict(hb_factor=2.))
    assert values.shape == (120, 4) and len(donors) == 2
    assert target["id"] not in reads and set(reads) == {r["id"] for r in donors}
    contaminated = deepcopy(refs)
    contaminated[0]["hb_channel"] = target["hb_channel"]
    with pytest.raises(ValueError, match="target cannot"):
        runner.spatial_target(target, contaminated, parent, dict(hb_factor=2.))


def test_training_pair_permutation_is_bijective_matched_and_subject_disjoint(tmp_path):
    refs = [_ref(tmp_path, subject=f"subject{s}", window=w) for s in range(4) for w in range(2)]
    for i, ref in enumerate(refs):
        ref["condition"] = "left" if i % 2 else "right"
    perm, missing = runner.training_permutation(refs, 123)
    other, _ = runner.training_permutation(list(reversed(refs)), 123)
    assert perm == other and not missing
    assert set(perm) == set(perm.values()) == {r["id"] for r in refs}
    lookup = {r["id"]: r for r in refs}
    for identity, donor in perm.items():
        assert lookup[identity]["subject"] != lookup[donor]["subject"]
        assert lookup[identity]["condition"] == lookup[donor]["condition"]
    impossible = [_ref(tmp_path, subject="same", window=i) for i in range(3)] + [_ref(tmp_path, subject="other")]
    perm, missing = runner.training_permutation(impossible, 123)
    assert not perm and len(missing) == 4
    assert all(r["reason"] == "matched_bijective_cross_subject_permutation_unavailable" for r in missing)


def test_selection_ranking_subject_balances_and_penalizes_failures():
    rows = [dict(subject="many", gamma=gamma, success=True, Hb_nrmse=value)
            for gamma, value in ((0., 0.), (1., .5)) for _ in range(4)]
    rows += [dict(subject="one", gamma=0., success=True, Hb_nrmse=2.),
             dict(subject="one", gamma=1., success=True, Hb_nrmse=.5)]
    ranking = runner.subject_equal_ranking(rows, [0., 1.], "gamma", 10.)
    assert ranking[0]["gamma"] == 1. and ranking[0]["score"] == .5
    failed = [dict(subject="one", gamma=1., success=False, Hb_nrmse=0.)]
    ranking = runner.subject_equal_ranking(failed, [1.], "gamma", 10.)
    assert ranking[0]["score"] == 10.


def test_predictive_sd_selection_residuals_are_subject_balanced_and_training_floored():
    mask = np.zeros((120, 2), bool)
    mask[80:] = True
    records = [(dict(subject="many", success=True), np.full((120, 2), [2., 0.]), mask) for _ in range(4)]
    records += [(dict(subject="one", success=True), np.full((120, 2), [10., 0.]), mask)]
    result = runner.calibrate_predictive_sd(records, np.array([1., 4.]), .05)
    np.testing.assert_allclose(result["predictive_sd"], [np.sqrt(52.), .2])
    assert result["subjects"] == 2 and result["successful_windows"] == 5
    absent = runner.calibrate_predictive_sd([(dict(subject="bad", success=False), np.zeros((120, 2)), mask)], [1., 1.])
    assert absent["status"] == "selection_residuals_unavailable"


def test_proper_scores_preserve_Hb_components_and_80percent_interval():
    target, prediction = np.zeros((120, 2)), np.ones((120, 2))
    mask = np.zeros_like(target, bool)
    mask[80:] = True
    result = runner.prediction_scores(prediction, target, [1., 2.], mask, [1., 1.])
    assert result["HbO_nrmse"] == 1. and result["HbR_nrmse"] == .5
    assert result["Hb_nrmse"] == pytest.approx(np.sqrt(.625))
    assert result["Gaussian_NLL"] == pytest.approx(.5 * np.log(2 * np.pi) + .5)
    assert result["coverage_50"] == 0. and result["coverage_80"] == 1. and result["coverage_95"] == 1.
    assert result["Gaussian_CRPS"] > 0
    assert result["Gaussian_CRPS_normalized"] == pytest.approx(result["Gaussian_CRPS"] * .75)


def test_shift_null_and_real_use_same_nonwrapping_support(cfg):
    eeg = np.arange(3600.).reshape(120, 30)
    real, real_visible, real_support = runner.pairing_eeg(eeg, cfg, "real_shift_support")
    shifted, shifted_visible, shifted_support = runner.pairing_eeg(eeg, cfg, "nonwrapping_shift_12s")
    np.testing.assert_array_equal(real_visible, shifted_visible)
    np.testing.assert_array_equal(real_support, shifted_support)
    assert not real_support[:48].any() and np.isnan(real[:48]).all() and np.isnan(shifted[:48]).all()
    np.testing.assert_array_equal(shifted[48:], eeg[:-48])
    np.testing.assert_array_equal(real[48:], eeg[48:])
    np.testing.assert_array_equal(runner.endpoint_mask(cfg, real_support), runner.endpoint_mask(cfg, shifted_support))


def test_paired_summary_uses_success_not_convergence_and_keeps_missing_denominator():
    rows = [dict(id="shared", subject="s1", arm="candidate", dataset="fixture", prefix_steps=0,
                 success=True, converged=False, Hb_nrmse=.5),
            dict(id="shared", subject="s1", arm="control", dataset="fixture", prefix_steps=0,
                 success=True, converged=False, Hb_nrmse=1.),
            dict(id="failed", subject="s2", arm="candidate", dataset="fixture", prefix_steps=0,
                 success=False, converged=True, Hb_nrmse=.1),
            dict(id="failed", subject="s2", arm="control", dataset="fixture", prefix_steps=0,
                 success=True, converged=True, Hb_nrmse=1.),
            dict(id="missing", subject="s3", arm="control", dataset="fixture", prefix_steps=0,
                 success=True, converged=True, Hb_nrmse=1.)]
    result = runner.paired_rows(pd.DataFrame(rows), controls=["control"], groups=["dataset", "prefix_steps"])[0]
    assert result["planned_pairs"] == 3 and result["common_success_pairs"] == 1
    assert result["failed_or_missing_pairs"] == 2 and result["gain"] == .5


def test_dataset_equal_aggregation_does_not_weight_six_subject_dataset_more():
    rows = []
    for dataset, count, gain in zip(runner.DATASETS, [6, 5, 4], [1., 0., -1.]):
        for i in range(count):
            for arm, value in (("control", 2.), ("candidate", 2. - gain)):
                rows.append(dict(id=f"{dataset}_{i}", subject=str(i), arm=arm, dataset=dataset,
                                 prefix_steps=40, success=True, Hb_nrmse=value))
    result = runner.dataset_equal_pairs(pd.DataFrame(rows), controls=["control"], metric="Hb_nrmse", seed=123)[0]
    assert result["gain"] == 0. and result["subjects"] == 15 and result["datasets"] == 3


def test_measured_plan_has_real_best_stable_and_only_prefix40_initial_ablations(cfg, tmp_path):
    ref = _ref(tmp_path)
    item = dict(ref=ref, split="evaluation", native_id="native", calibration_key="cal", synchronous_region_ids=[])
    cells = runner.measured_cells(cfg, dict(windows=[item]))
    assert len(cells) == 64
    assert sum(c["arm"] == "best_stable" and c["pairing"] == "real" for c in cells) == 2
    ablations = [c for c in cells if "__" in c["arm"]]
    assert len(ablations) == 2 and all(c["prefix_steps"] == 40 for c in ablations)
    assert not runner.measured_cells(cfg, dict(windows=[dict(item, split="train")]))


def test_feature_batches_hide_target_suffix_preserve_score_target_and_native_join(tmp_path):
    items = []
    cal = dict(key="fixture", eeg_factor=1., hb_factor=1., sd_eeg=np.ones(30), sd_hb=np.array([.02, .01]))
    eeg, hb = np.ones((120, 30)), np.arange(240.).reshape(120, 2)
    modes = dict(r=np.ones((120, 2)))
    for split in runner.WINDOW_CAPS:
        for region in runner.REGIONS:
            ref = _ref(tmp_path / "parent", subject=split, region=region)
            runner.export_features(ref, split, eeg, hb, modes, cal, tmp_path)
            items.append(dict(ref=ref, split=split, native_id=split + "_native", calibration_key=cal["key"]))
    retained_v1 = tmp_path / "features" / "evaluation.npz"
    np.savez(retained_v1, retained_identity=np.array(["v1_evidence"]))
    retained_bytes = retained_v1.read_bytes()
    runner.export_batched_features(dict(windows=items), tmp_path)
    assert retained_v1.read_bytes() == retained_bytes
    batch_root = tmp_path / "features" / "batch_v2"
    with np.load(batch_root / "evaluation.npz", allow_pickle=False) as arrays:
        assert arrays["hb_mask"][:, :40].all() and not arrays["hb_mask"][:, 40:].any()
        assert np.isnan(arrays["hb"][:, 40:]).all()
        np.testing.assert_array_equal(arrays["hb_target"], np.broadcast_to(hb, (3, 120, 2)))
        np.testing.assert_array_equal(arrays["sd_hb"], [[.02, .01]] * 3)
        np.testing.assert_array_equal(arrays["hb_factor"], np.ones(3))
        np.testing.assert_array_equal(arrays["eeg_factor"], np.ones(3))
        assert len(set(arrays["native_id"])) == 1 and set(arrays["region"]) == set(runner.REGIONS)
    manifest = runner.read_json(batch_root / "evaluation.json")
    assert manifest["hb_target_role"] == "scoring_only_never_encoder_input"
    with np.load(batch_root / "evaluation__full_spatial_context.npz", allow_pickle=False) as arrays:
        assert arrays["hb_mask"].all() and np.isfinite(arrays["hb"]).all()


def test_failed_worker_evidence_is_terminal_and_keeps_traceback(cfg, tmp_path, monkeypatch):
    def broken(*args):
        raise RuntimeError("deliberate fixture failure")
    monkeypatch.setattr(runner, "calibration_worker", broken)
    row = runner.safe_worker(("calibrate", (cfg, str(tmp_path), str(tmp_path), dict(key="fixture"), [], str(tmp_path))))
    assert not row["success"] and row["status"] == "failed_exception"
    retained = runner.read_json(tmp_path / "calibration" / "fixture.json")
    assert "deliberate fixture failure" in retained["traceback"]


def test_small_synthetic_dryrun_reads_no_public_payload_and_checks_hidden_poison(cfg, tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("synthetic dry-run must not enter public payload reader")
    monkeypatch.setattr(runner, "raw_target", forbidden)
    result = runner.dry_run(cfg, tmp_path)
    assert result["execution"] == "completed" and result["fixture_valid"] and result["public_payloads_read"] == 0
    assert all(row["hidden_suffix_invariant"] for row in result["checks"])
    runner.require_synthetic_fixture(tmp_path)


def test_temporary_calibration_evaluation_resume_and_summary_integrate_all_arms(cfg, tmp_path, monkeypatch):
    """Exercise controller ownership without measured artifacts or costly fitting."""
    parent, run_root = tmp_path / "parent", tmp_path / "run"
    source = tmp_path / cfg["source_config"]
    source.parent.mkdir(parents=True)
    source.write_text((runner.CODE_ROOT / cfg["source_config"]).read_text())
    rng, refs = np.random.default_rng(19), []
    subjects = ["train_a", "train_b", "select_a", "select_b", "evaluate"]
    for subject in subjects:
        for region in runner.REGIONS:
            count = 1 if subject == "evaluate" else 2
            first = _ref(parent, subject=subject, region=region)
            path = Path(first["array_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, eeg_features=rng.normal(0, .1, (count, 120, 30)),
                     hb=rng.normal(0, .01, (count, 120, 2)))
            refs.extend(_ref(parent, subject=subject, region=region, window=i) for i in range(count))
    def simple_fit(hb, modes, spec, cal, cfg, p, prefix):
        visible, mask = runner.visible_hb(hb, prefix)
        prediction = np.full((120, 2), visible[mask].mean() if mask.any() else 0.)
        return dict(prediction=prediction, physical_prediction=prediction.copy(), observation_component=np.zeros_like(hb),
                    driver=runner.driver_from_modes(modes, spec), states=np.ones((120, 6)),
                    initial_state=np.array([0., 1., 1., 1., 1.]), component_coefficients=np.zeros(4),
                    success=True, converged=prefix > 0, fit_performed=prefix > 0,
                    status="completed" if prefix else "prior_prediction")
    monkeypatch.setattr(runner, "fit_response", simple_fit)
    spec = dict(key="eeg_fnirs_single_trial__motor", dataset=runner.DATASETS[0], region="motor",
                train_ids=[r["id"] for r in refs if r["subject"].startswith("train") and r["region"] == "motor"],
                selection_ids=[r["id"] for r in refs if r["subject"].startswith("select") and r["region"] == "motor"],
                evaluation_ids=[r["id"] for r in refs if r["subject"] == "evaluate" and r["region"] == "motor"],
                train_subjects=["train_a", "train_b"], selection_subjects=["select_a", "select_b"], evaluation_subjects=["evaluate"])
    cal = runner.calibration_worker((cfg, str(tmp_path), str(run_root), spec, refs, str(parent)))
    assert cal["status"] == "completed" and cal["selected"]["0"] == cal["selected"]["40"]
    assert set(cal["ridge_models"]["40"]) == set(runner.BASELINES[2:])
    assert set(cal["training_pair_permutation"]["donors"].values()) == set(spec["train_ids"])
    ref = next(r for r in refs if r["subject"] == "evaluate" and r["region"] == "motor")
    item = dict(ref=ref, split="evaluation", native_id="native_eval", calibration_key=spec["key"],
                synchronous_region_ids=[r["id"] for r in refs if r["subject"] == "evaluate" and r["region"] != "motor"])
    payload = (cfg, str(tmp_path), str(run_root), item, refs, str(parent))
    first, resumed = runner.measured_worker(payload), runner.measured_worker(payload)
    assert first == resumed and first["planned_cells"] == 64 and first["successful_cells"] == 64
    runner.write_json(run_root / "measured_plan.json", dict(windows=[item]))
    summary = runner.summarize(cfg, run_root)
    assert summary["measured"]["recorded"] == 64 and summary["execution"] == "incomplete"
    paired_nulls = pd.read_csv(run_root / "paired_nulls.csv")
    assert "best_stable" in set(paired_nulls.arm)
    assert (paired_nulls.planned_pairs == 1).all()


def _primary_null_fixture(cfg):
    rows = []
    for dataset, count, effect in zip(runner.DATASETS, [6, 5, 4], [.1, .2, .3]):
        for subject in range(count):
            for window in range(8):
                for region in runner.REGIONS:
                    for arm in cfg["measured"]["null_arms"]:
                        for pairing in runner.PAIRINGS:
                            rows.append(dict(id=f"{dataset}_{subject}_{window}_{region}", dataset=dataset,
                                             subject=f"s{subject}", prefix_steps=40, region=region, arm=arm,
                                             pairing=pairing, success=True,
                                             Hb_nrmse=1. if pairing in ("real", "real_shift_support") else 1. + effect))
    return pd.DataFrame(rows)


def test_primary_null_family_exact_subject_blocks_dataset_weights_and_Holm_nine(cfg):
    frame = _primary_null_fixture(cfg)
    rows = runner.primary_null_specificity(frame, cfg)
    assert len(rows) == 9
    for row in rows:
        assert row["subjects"] == 15 and row["planned_pairs"] == row["common_success_pairs"] == 360
        assert row["dataset_equal_gain"] == pytest.approx(.2)
        assert row["p_one_sided_sign_flip"] == 1 / 32768
        assert row["p_Holm_nine"] == 9 / 32768
        assert row["pass_eligible"] and row["null_specificity_pass"]
        assert row["sign_patterns"] == 32768
    shifted = [r for r in rows if r["null_pairing"] == "nonwrapping_shift_12s"]
    assert all(r["real_pairing"] == "real_shift_support" for r in shifted)


def test_null_specificity_cannot_pass_by_dropping_failed_pairs(cfg):
    frame = _primary_null_fixture(cfg)
    target = (frame.arm == "gamma_selected") & (frame.pairing == "wrong_training_subject")
    first = frame[target].index[0]
    frame.loc[first, "success"] = False
    rows = runner.primary_null_specificity(frame, cfg)
    affected = next(r for r in rows if r["arm"] == "gamma_selected" and r["null_pairing"] == "wrong_training_subject")
    assert affected["common_success_pairs"] == 359 and affected["failed_or_missing_pairs"] == 1
    assert affected["subjects"] == 15 and affected["p_Holm_nine"] < .05
    assert not affected["pass_eligible"] and not affected["null_specificity_pass"]


def test_counterfactual_summary_reads_repeat_column_and_retained_prediction_pair(cfg, tmp_path):
    cfg = deepcopy(cfg)
    cfg["synthetic"].update(families=["restricted"], scenarios=["input_lowpass"], repeats=1)
    for variant, prediction, target in (("control", 0., 0.), ("intervention", 2., 3.)):
        spec = dict(family="restricted", scenario="input_lowpass", repeat=0, variant=variant,
                    arm="gain_selected", prefix_steps=40)
        path = runner.synthetic_path(tmp_path, spec)
        runner.save_arrays(path.with_suffix(".npz"), prediction=np.full((120, 2), prediction),
                           Hb_truth=np.full((120, 2), target))
        runner.write_json(path, dict(spec, base_id="fixture_base", success=True, converged=True,
                                    fit_performed=True, generated_valid=True, Hb_nrmse=1.,
                                    arrays=str(path.with_suffix(".npz").relative_to(tmp_path))))
    result = runner.summarize_synthetic(cfg, tmp_path)
    assert result["recorded"] == 2 and result["execution"] == "incomplete"
    paired = pd.read_csv(tmp_path / "synthetic_counterfactual_response.csv")
    assert len(paired) == 1 and paired.iloc[0]["repeat"] == 0
    assert paired.iloc[0]["counterfactual_response_change_RMSE"] == 1.
    assert paired.iloc[0]["true_observation_change_RMS"] == 3.
    assert paired.iloc[0]["predicted_observation_change_RMS"] == 2.


def test_overall_execution_waits_for_owning_tokenizer_and_public_probes(tmp_path):
    assert runner.overall_execution_status("completed", tmp_path)["execution"] == "incomplete"
    path = tmp_path / "tokenizer" / "summary.json"
    runner.write_json(path, dict(schema="semantic_response_tokenizer_suite_v1", status="complete", public_probe_status="incomplete"))
    assert runner.overall_execution_status("completed", tmp_path)["execution"] == "incomplete"
    runner.write_json(path, dict(schema="semantic_response_tokenizer_suite_v1", status="complete", public_probe_status="complete"))
    result = runner.overall_execution_status("completed", tmp_path)
    assert result["execution"] == "completed" and result["tokenizer_summary_path"] == "tokenizer/summary.json"
    assert runner.overall_execution_status("incomplete", tmp_path)["execution"] == "incomplete"
