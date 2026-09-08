from __future__ import annotations

import copy
import inspect
import json
import math

import numpy as np
import pytest

import experiments.evaluate_t3c_composite_synthetic_t2 as t2


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    """Non-evidence source contract for unit/smoke checks after forward repairs.

    The registered September 3 experiment still rejects changed source bytes.
    These temporary copies exercise the real hash validator without rewriting
    that experiment's frozen hashes or disabling its validation function.
    """
    config = t2.load_config()
    expected = {}
    for name, (path, _) in t2.EXPECTED_SOURCES.items():
        fixture = tmp_path / (name + '.source')
        fixture.write_bytes((t2.REPO_ROOT / path).read_bytes())
        expected[name] = (str(fixture), t2._sha256(fixture))
        config['sources'][name] = dict(path=str(fixture), sha256=expected[name][1])
    monkeypatch.setattr(t2, 'EXPECTED_SOURCES', expected)
    return config


def _tiny_config() -> dict:
    config = t2._effective_config(t2.load_config(), True)
    config["simulation"]["independent_replicates_per_direction"] = 1
    config["simulation"]["duration_s_per_trial"] = 4.0
    config["simulation"]["heldout_fnirs_mask_from_s"] = 1.0
    config["inference"]["multistarts"] = 2
    config["inference"]["profile_points"] = 3
    config["inference"]["posterior_grid_points"] = 5
    return config


def test_frozen_experiment_keeps_original_source_identity_and_rejects_repaired_core() -> None:
    config = t2.load_config()
    for name, (path, digest) in t2.EXPECTED_SOURCES.items():
        assert config['sources'][name] == dict(path=path, sha256=digest)
    with pytest.raises(ValueError, match='registered source hash mismatch: src/inference/t3a_balloon_robust_ssm.py'):
        t2.validate_config(config)


def test_config_and_composite_axes_preserve_the_fixed_gauge(source_fixture) -> None:
    config = source_fixture
    t2.validate_config(config)
    assert "evaluate_t3_measured_reconstruction_null" not in inspect.getsource(t2)
    for name, (path, digest) in t2.EXPECTED_SOURCES.items():
        assert config["sources"][name] == {"path": path, "sha256": digest}
        assert t2._sha256(t2.REPO_ROOT / path) == digest
    reference = t2.raw_from_relative(0.0, 0.0, config)

    gain = t2.raw_from_relative(0.4, 0.0, config)
    assert gain["beta"] == pytest.approx(math.exp(0.4))
    assert gain["gamma"] == reference["gamma"]
    assert gain["kappa"] == reference["kappa"]

    time = t2.raw_from_relative(0.0, 0.4, config)
    assert time["beta"] == pytest.approx(math.exp(-0.8))
    assert time["gamma"] == pytest.approx(reference["gamma"] * math.exp(-0.8))
    assert time["kappa"] == pytest.approx(reference["kappa"] * math.exp(-0.4))
    for raw in (gain, time):
        assert raw["beta"] / raw["gamma"] == pytest.approx(reference["beta"] / reference["gamma"] if raw is time else math.exp(0.4) * reference["beta"] / reference["gamma"])
        assert raw["kappa"] / (2.0 * math.sqrt(raw["gamma"])) == pytest.approx(reference["kappa"] / (2.0 * math.sqrt(reference["gamma"])))
        assert raw["tau"] * raw["alpha"] == pytest.approx(reference["tau"] * reference["alpha"])
    with pytest.raises(ValueError):
        t2.raw_from_relative(0.0, 1.0, config)
    for mutate in (
        lambda value: value["sources"].pop("model"),
        lambda value: value["sources"]["model"].__setitem__("sha256", "0" * 64),
        lambda value: value["composite"].__setitem__("fixed_gauge", []),
        lambda value: value["composite"]["candidates"]["C1_G"].__setitem__("fixed", {"log_time_relative": 0.1}),
        lambda value: value["inference"].__setitem__("boundary_fraction_of_span", 0.02),
        lambda value: value["output"].__setitem__("root", "elsewhere"),
    ):
        drifted = copy.deepcopy(config)
        mutate(drifted)
        with pytest.raises(ValueError):
            t2.validate_config(drifted)


def test_temporary_source_contract_detects_byte_tampering(source_fixture) -> None:
    from pathlib import Path
    model = Path(source_fixture['sources']['model']['path'])
    model.write_bytes(model.read_bytes()+b'\n# changed after binding\n')
    with pytest.raises(ValueError, match='registered source hash mismatch'):
        t2.validate_config(source_fixture)


def test_truth_is_not_a_fitdataset_field_and_same_observations_fit_identically() -> None:
    config = _tiny_config()
    raw = t2._coordinate_raw("C1_G", 0.2, config)
    generated = t2._truth_trial(raw, 7, config)
    problem = t2.FitDataset("C1_G", 0, (generated["observations"],), 71_000_000, t2._fit_contract(config))
    assert not any("truth" in field for field in problem.__dataclass_fields__)
    first = t2._fit_one(problem)
    second = t2._fit_one(copy.deepcopy(problem))
    assert first["best_estimate"] == pytest.approx(second["best_estimate"], abs=1.0e-12)
    np.testing.assert_allclose(first["profile_nll"], second["profile_nll"], atol=0.0, rtol=0.0)


def test_c2_gate_requires_both_c1_passes() -> None:
    assert t2.c2_gate_state({"C1_G": "PASS", "C1_T": "FAIL"}) == "NOT_RUN_C1_GATE_NOT_MET"
    assert t2.c2_gate_state({"C1_G": "PASS"}) == "NOT_RUN_C1_GATE_NOT_MET"
    assert t2.c2_gate_state({"C1_G": "PASS", "C1_T": "PASS"}) == "ELIGIBLE_FOR_SEPARATE_REGISTERED_RUN"
    assert t2.c2_gate_state({"C1_G": "PASS", "C1_T": "PASS"}, smoke=True) == "NOT_RUN_SMOKE"


def test_formal_row_contract_and_active_joint_sensitivity() -> None:
    config = t2.load_config()
    expected, _ = t2._expected_rows(config, smoke=False)
    assert expected == {
        "truth_parameters.csv": 120,
        "synthetic_inventory.csv": 480,
        "multistart_results.csv": 3840,
        "profile_likelihood.csv": 2520,
        "posterior_grid.csv": 9720,
        "posterior_diagnostics.csv": 240,
        "parameter_recovery.csv": 120,
        "state_confounding.csv": 120,
        "heldout_scores.csv": 120,
        "sensitivity_svd.csv": 120,
        "calibration.csv": 4,
        "gates.csv": 38,
    }
    raw = t2._coordinate_raw("C1_G", 0.1, config)
    trials = [t2._truth_trial(raw, seed, config) for seed in (1, 2, 3)]
    result = t2._sensitivity("C1_G", raw, [trial["driver"] for trial in trials], config)
    assert result["active_effective_rank"] == 1
    assert result["effective_rank"] == 2
    assert result["relative_singular_value_2"] >= config["inference"]["sensitivity_relative_singular_threshold"]


def test_sbc_gate_rejects_pseudoreplicated_small_panels() -> None:
    config = t2.load_config()
    rows = [
        {
            "posterior_method": "exact_1d_grid_quadrature_under_EKF_likelihood",
            "rank_u": (index + 0.5) / 10.0,
            "covered95": index != 0,
            "resolution_pass": True,
        }
        for index in range(10)
    ]
    result = t2._calibration(rows, "exact_1d_grid_quadrature_under_EKF_likelihood", config)
    assert result["independent_replicates"] == 10
    assert result["passed"] is False


def test_heldout_score_masks_every_post_onset_fnirs_value(monkeypatch) -> None:
    config = _tiny_config()
    raw = t2._coordinate_raw("C1_G", 0.1, config)
    generated = t2._truth_trial(raw, 11, config)
    original = t2.smooth_balloon
    observed_masks = []

    def spy(observations, *args, observation_mask=None, **kwargs):
        observed_masks.append(np.asarray(observation_mask).copy())
        return original(observations, *args, observation_mask=observation_mask, **kwargs)

    monkeypatch.setattr(t2, "smooth_balloon", spy)
    score, count = t2._score((generated["observations"],), "C1_G", 0.1, config)
    first_target = int(config["simulation"]["sampling_hz"] * config["simulation"]["heldout_fnirs_mask_from_s"])
    assert np.isfinite(score)
    assert count == (len(generated["observations"]) - first_target) * 2
    assert len(observed_masks) == 1
    assert np.all(observed_masks[0][:first_target])
    assert np.all(observed_masks[0][first_target:, 0])
    assert not np.any(observed_masks[0][first_target:, 1:])


def test_tiny_smoke_writes_complete_hash_bound_artifacts(tmp_path, monkeypatch, source_fixture) -> None:
    config = source_fixture
    effective = _tiny_config()
    effective['sources'] = copy.deepcopy(config['sources'])
    monkeypatch.setattr(t2, "_effective_config", lambda _config, _smoke: copy.deepcopy(effective))
    run_dir = tmp_path / "run"
    summary = t2.run(config, run_dir, smoke=True)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert summary["decision"] == "SMOKE_COMPLETE_NOT_EVIDENCE"
    assert manifest["completion_status"] == "complete"
    assert manifest["boundary"]["truth_passed_to_fitter"] is False
    assert manifest["boundary"]["measured_arrays_opened"] is False
    assert manifest["C2_GT_state"] == "NOT_RUN_SMOKE"
    for name, artifact in manifest["artifacts"].items():
        path = run_dir / name
        assert path.is_file()
        assert t2._sha256(path) == artifact["sha256"]
    with pytest.raises(FileExistsError):
        t2.run(config, run_dir, smoke=True)


def test_failure_manifest_is_fail_closed(tmp_path, monkeypatch, source_fixture) -> None:
    config = source_fixture
    effective = _tiny_config()
    effective['sources'] = copy.deepcopy(config['sources'])
    monkeypatch.setattr(t2, "_effective_config", lambda _config, _smoke: copy.deepcopy(effective))
    monkeypatch.setattr(t2, "_fit_one", lambda _problem: (_ for _ in ()).throw(RuntimeError("injected")))
    run_dir = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected"):
        t2.run(config, run_dir, smoke=True)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete_failed"
    assert manifest["completion_status"] == "incomplete"
    assert manifest["boundary"]["measured_arrays_opened"] is False
    assert not (run_dir / "summary.json").exists()
