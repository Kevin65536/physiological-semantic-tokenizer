"""Synthetic-only contracts for independent continuous semantic encoders."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tokenizers.semantic_response_tokenizer import (
    SemanticResponseTokenizer, TrainingNormalizer, gaussian_crps, gaussian_nll,
)


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_model(**kwargs):
    torch.manual_seed(21)
    return SemanticResponseTokenizer(width=8, observation_dimensions=2, **kwargs)


def runner():
    path = Path(__file__).resolve().parents[1] / "experiments/scripts/train_semantic_response_tokenizer.py"
    spec = importlib.util.spec_from_file_location("semantic_response_training_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_each_modality_has_only_its_own_values_and_gradients():
    model = small_model().eval()
    eeg, hb = torch.randn(2, 120, 30), torch.randn(2, 120, 2)
    original = model(eeg, hb)
    changed = model(eeg * 100, hb)
    for key in original["hb"]:
        assert torch.equal(original["hb"][key], changed["hb"][key])
    original["eeg"]["shape_mean"].square().sum().backward()
    assert any(parameter.grad is not None for parameter in model.eeg_branch.source_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.hb_branch.parameters())


@pytest.mark.parametrize("modality,channels", [("eeg", 30), ("hb", 2)])
def test_hidden_nan_inf_and_large_values_never_reach_encoder(modality, channels):
    model = small_model().eval()
    values = torch.randn(1, 120, channels, requires_grad=True)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[:, 40:] = False
    mask[:, :40, 0] = False
    original = getattr(model, f"encode_{modality}")(values, mask)
    poisoned = values.detach().clone()
    poisoned[~mask] = float("nan")
    poisoned[:, 90:, :] = float("inf")
    changed = getattr(model, f"encode_{modality}")(poisoned, mask)
    for key in original:
        assert torch.equal(original[key], changed[key])
    original["shape_mean"].square().sum().backward()
    assert torch.count_nonzero(values.grad[~mask]) == 0
    assert torch.isfinite(values.grad).all()


def test_visible_nonfinite_rejected_before_arithmetic():
    model = small_model()
    eeg = torch.zeros(1, 120, 30)
    eeg[0, 10, 1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model.encode_eeg(eeg)


def test_reconstruction_stops_at_supervised_source_but_trains_obs_only_capacity():
    eeg = torch.randn(1, 120, 30)
    for observation_only in (False, True):
        model = small_model(observation_only=observation_only)
        output = model.encode_eeg(eeg)
        output["reconstruction"].square().mean().backward()
        semantic_gradients = [parameter.grad for parameter in model.eeg_branch.source_encoder.parameters()]
        assert any(gradient is not None for gradient in semantic_gradients) == observation_only
        assert any(parameter.grad is not None for parameter in model.eeg_branch.observation_encoder.parameters())
        assert all(parameter.grad is None for parameter in model.hb_branch.parameters())


def test_all_arms_match_mean_bottleneck_and_parameter_capacity():
    models = [small_model(distribution=distribution, observation_only=observation_only)
              for distribution, observation_only in ((False, False), (True, False), (False, True))]
    assert len({sum(parameter.numel() for parameter in model.parameters()) for model in models}) == 1
    for model in models:
        output = model.encode_eeg(torch.randn(2, 120, 30))
        assert output["shape_mean"].shape == (2, 120)
        assert output["log_amplitude_mean"].shape == (2,)
        assert output["observation_code"].shape == (2, 120, 2)
        assert output["reconstruction"].shape == (2, 120, 30)


def test_continuous_inference_precedes_supported_patch_export():
    model = small_model(distribution=True)
    output = model.encode_hb(torch.randn(1, 120, 2))
    assert torch.allclose(output["shape_mean"][:, :20].mean(1), torch.zeros(1), atol=1e-7)
    output["shape_mean"] = torch.arange(120, dtype=torch.float32)[None]
    output["shape_log_sd"] = torch.zeros(1, 120)
    output["time_support"][:, 3:10] = False
    output["time_support"][:, 20:30] = False
    tokens = model.export_tokens(output)
    assert tokens["semantic_mean"].shape == (1, 12)
    assert tokens["semantic_mean"][0, 0].item() == 1.0
    assert tokens["support_count"][0, 0].item() == 3
    assert not tokens["support"][0, 2]
    assert tokens["semantic_mean"][0, 2].item() == 0.0
    assert tokens["semantic_sd_upper_bound"][0, 0].item() == 1.0
    assert "not_calibrated" in model.token_metadata()["patch_uncertainty"]


def test_normalization_and_probe_must_fit_training_partition_only():
    values = torch.arange(24, dtype=torch.float32).reshape(2, 6, 2)
    normalizer = TrainingNormalizer(2).fit(values)
    before = copy.deepcopy(normalizer.state_dict())
    normalizer(values + 1e8)
    for key in before:
        assert torch.equal(before[key], normalizer.state_dict()[key])
    with pytest.raises(ValueError, match="training"):
        normalizer.fit(values, partition="evaluation")
    module = runner()
    features, targets = np.arange(20).reshape(5, 4), np.arange(10).reshape(5, 2)
    probe = module.fit_training_probe(features, targets)
    assert probe["partition"] == "train"
    with pytest.raises(ValueError, match="training"):
        module.fit_training_probe(features, targets, partition="selection")


def test_proper_scores_known_normal_and_train_only_calibration_application():
    zero = torch.zeros(4)
    expected_crps = (np.sqrt(2) - 1) / np.sqrt(np.pi)
    assert torch.allclose(gaussian_crps(zero, zero, zero), torch.full_like(zero, expected_crps))
    assert torch.allclose(gaussian_nll(zero, zero, zero), torch.full_like(zero, 0.5 * np.log(2 * np.pi)))
    assert gaussian_crps(zero, zero, torch.ones(4)).mean() > expected_crps
    module = runner()
    values = dict(shape_mean=np.zeros((2, 120)), log_amplitude_mean=np.zeros(2),
                  shape_log_sd=np.zeros((2, 120)), log_amplitude_log_sd=np.zeros(2))
    selection = dict(source_shape=np.ones((2, 120)) * 2, log_amplitude=np.ones(2) * 3,
                     source_mask=np.ones((2, 120), bool))
    calibration = module.calibrate("distribution_randomized_generator", {"eeg": values}, selection, {})
    assert calibration["eeg"]["partition"] == "selection"
    shape_sd, amp_sd = module.uncertainty("distribution_randomized_generator", values, calibration["eeg"])
    np.testing.assert_allclose(shape_sd, 2)
    np.testing.assert_allclose(amp_sd, 3)


def test_exact_alias_changes_truth_without_changing_encoder_output():
    from src.inference.semantic_response_synthetic import generate_response_case
    pair = generate_response_case(101, scenario="exact_joint_alias")
    model = small_model(distribution=True).eval()
    encoded = []
    for variant in ("control", "intervention"):
        record = pair[variant]
        encoded.append(model(torch.as_tensor(record["eeg"], dtype=torch.float32)[None],
                             torch.as_tensor(record["hb"], dtype=torch.float32)[None]))
    assert pair["control"]["log_amplitude"] != pair["intervention"]["log_amplitude"]
    for modality in ("eeg", "hb"):
        for key in encoded[0][modality]:
            assert torch.equal(encoded[0][modality][key], encoded[1][modality][key])


def test_alias_confident_error_checks_both_truths_and_constant_scale_cannot_rank():
    from src.inference.semantic_response_synthetic import generate_response_case
    module = runner()
    pair = generate_response_case(101, scenario="exact_joint_alias")
    records = [pair["control"], pair["intervention"]]
    data = {key: np.asarray([record[key] for record in records]) for key in (
        "eeg", "hb", "eeg_mask", "hb_mask", "source_shape", "source_mask", "log_amplitude")}
    data.update(base_id=np.array(["alias", "alias"]),
        scenario=np.array(["exact_joint_alias", "exact_joint_alias"]),
        variant=np.array(["control", "intervention"]))
    predicted = {}
    for modality, channels in (("eeg", 30), ("hb", 2)):
        predicted[modality] = dict(shape_mean=np.tile(records[1]["source_shape"], (2, 1)),
            log_amplitude_mean=np.full(2, records[1]["log_amplitude"]),
            reconstruction=np.zeros((2, 120, channels)))
    checkpoint = dict(probes={}, calibration={modality: dict(shape=0.1, amplitude=0.1,
        kind="selection_constant_residual_scale_not_network_uncertainty") for modality in ("eeg", "hb")})
    _, pairs, _, risk = module.evaluation_records("point_randomized_generator", predicted,
                                                 data, checkpoint, small_model())
    for pair_row in pairs:
        assert pair_row["alias_control_confident_amplitude_error"]
        assert not pair_row["alias_intervention_confident_amplitude_error"]
        assert pair_row["alias_confident_amplitude_error"]
    for row in risk:
        assert row["retained_records"] == row["total_records"] == 2
        assert row["retained_fraction"] == 1.0


def test_cli_parallel_prepare_binds_contract_seed(tmp_path):
    import json
    import subprocess
    from src.inference.semantic_response_synthetic import generate_response_case
    module = runner()
    root = Path(__file__).resolve().parents[1]
    configuration = tmp_path / "fixture.yaml"
    configuration.write_text("seed: 1337\ntokenizer:\n  seeds: [101]\n  epochs: 1\n  patience: 1\n"
        "  base_counts: {train: 18, selection: 1, evaluation: 1}\n"
        "  train_scenarios: [baseline]\n  evaluation_scenarios: [baseline]\n")
    subprocess.run([str(root / ".venv/bin/python"),
        str(root / "experiments/scripts/train_semantic_response_tokenizer.py"),
        "--stage", "prepare", "--project-root", str(root), "--output", str(tmp_path / "run"),
        "--config", str(configuration), "--workers", "2", "--family", "randomized",
        "--partition", "train"], check=True, timeout=40, capture_output=True)
    prepared = tmp_path / "run/prepared/randomized/train.npz"
    manifest = json.loads(prepared.with_suffix(".json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["completed_records"] == manifest["expected_records"] == 36
    assert manifest["workers"] == 2
    assert manifest["generator_seed"] == 1337
    with np.load(prepared) as arrays:
        expected = generate_response_case(1337, partition="train", base_id=0)
        np.testing.assert_array_equal(arrays["source_shape"][0], expected["control"]["source_shape"].astype(np.float32))


def public_feature_fixture(root):
    import json
    root.mkdir(parents=True)
    for split_index, split in enumerate(("train", "selection", "evaluation")):
        columns = {key: [] for key in ("eeg", "hb_target", "base_id", "dataset", "subject", "region", "native_id", "sd_hb", "hb_factor")}
        records = []
        for subject in range(3):
            rng = np.random.default_rng(8000 + 100 * split_index + subject)
            source = rng.normal(size=120).cumsum() * 0.025
            shared_eeg = source[:, None] * rng.normal(size=(1, 30)) + rng.normal(size=(120, 30)) * 0.01
            for region in range(3):
                identity = f"{split}/{subject}/r{region}"
                native_id = f"{split}/{subject}/window0"
                target = np.column_stack((source, -0.3 * source)) + rng.normal(size=(120, 2)) * 0.001
                row = dict(eeg=shared_eeg, hb_target=target, base_id=identity,
                    dataset="nonprotected_fixture", subject=f"{split}_{subject}", region=f"r{region}",
                    native_id=native_id, sd_hb=np.array([0.025, 0.025]), hb_factor=1.0)
                for key, value in row.items():
                    columns[key].append(value)
                records.append(dict(ref=dict(id=identity), split=split, native_id=native_id,
                                    calibration_key=f"fixture_r{region}"))
        arrays = {key: np.asarray(value) for key, value in columns.items()}
        hb = arrays["hb_target"].copy()
        hb[:, 40:] = np.nan
        hb_mask = np.ones_like(hb, dtype=bool)
        hb_mask[:, 40:] = False
        np.savez_compressed(root / f"{split}.npz", **arrays, hb=hb, hb_mask=hb_mask,
                            eeg_mask=np.ones_like(arrays["eeg"], bool))
        spatial = {key: value for key, value in arrays.items() if key != "hb_target"}
        np.savez_compressed(root / f"{split}__full_spatial_context.npz", **spatial,
            hb=arrays["hb_target"], hb_mask=np.ones_like(hb_mask), eeg_mask=np.ones_like(arrays["eeg"], bool))
        (root / f"{split}.json").write_text(json.dumps(dict(
            schema="semantic_response_discovery_feature_batch_v2", split=split, rows=len(hb), records=records)))


def test_public_features_hide_target_suffix_and_exclude_target_spatial_context(tmp_path):
    module = runner()
    public_feature_fixture(tmp_path / "features")
    batches, spatial = module.load_public_feature_batches(tmp_path / "features")
    data = batches["evaluation"]
    encoding = module.public_encoding_view(data, 40)
    context, donors = module.public_raw_context(data, spatial["evaluation"], 40)
    assert set(encoding) == {"eeg", "eeg_mask", "hb", "hb_mask"}
    assert np.isnan(encoding["hb"][:, 40:]).all()
    assert not encoding["hb_mask"][:, 40:].any()
    assert all(str(identity) not in donor for identity, donor in zip(data["base_id"], donors))
    changed_data = copy.deepcopy(data)
    changed_data["hb_target"][:, 40:] = 1e30
    changed_context, _ = module.public_raw_context(changed_data, spatial["evaluation"], 40)
    np.testing.assert_array_equal(context, changed_context)
    changed_spatial = copy.deepcopy(spatial["evaluation"])
    changed_spatial["hb"][0] = 1e30
    context_with_own_poison, _ = module.public_raw_context(data, changed_spatial, 40)
    np.testing.assert_array_equal(context[0], context_with_own_poison[0])
    model = small_model().eval()
    encoded = module.predictions(model, encoding, torch.device("cpu"), 4, 0, 1)
    zero_prefix_code = module.public_code_features(encoded, 0, include_source=True)
    encoded["hb"]["shape_mean"][:] = 1e30
    encoded["hb"]["observation_code"][:] = 1e30
    np.testing.assert_array_equal(zero_prefix_code, module.public_code_features(encoded, 0, include_source=True))


def test_public_probe_calibration_and_scores_use_training_units_only():
    module = runner()
    rng = np.random.default_rng(101)
    train_x, selection_x = rng.normal(size=(6, 9)), rng.normal(size=(4, 9))
    train_y, selection_y = rng.normal(size=(6, 80)), rng.normal(size=(4, 80))
    model, calibration = module.fit_public_probe(train_x, train_y, selection_x, selection_y,
                                                np.array(["a", "a", "b", "b"]), [0.1, 1.0])
    assert calibration["fitting_partition"] == "train"
    assert calibration["alpha_selection_partition"] == "selection"
    assert calibration["uncertainty_calibration_partition"] == "selection"
    with pytest.raises(ValueError, match="training"):
        module.fit_public_probe(train_x, train_y, selection_x, selection_y,
                                np.array(["a", "a", "b", "b"]), [0.1], partition="evaluation")
    data = dict(base_id=np.array(["x"]), native_id=np.array(["native"]),
        dataset=np.array(["fixture"]), subject=np.array(["subject"]), region=np.array(["r0"]),
        sd_hb=np.array([[0.025, 0.025]]))
    row = module.public_probe_rows(np.zeros((1, 80)), np.ones((1, 80)), data,
        np.array([0]), np.ones(2), 40, "fixture")[0]
    assert row["CRPS_parent_prepared"] == pytest.approx(row["CRPS_training_SD"] * 0.025)
    assert row["Gaussian_NLL_parent_prepared"] == pytest.approx(row["Gaussian_NLL_training_SD"] + np.log(0.025))
    scaled_data = copy.deepcopy(data)
    scaled_data["sd_hb"] *= 10
    scaled_data["hb_factor"] = np.array([10.0])
    restored = module.public_probe_rows(np.zeros((1, 80)), np.ones((1, 80)), scaled_data,
        np.array([0]), np.ones(2), 40, "fixture")[0]
    assert restored["RMSE_parent_prepared"] == pytest.approx(0.025)
    assert restored["CRPS_parent_prepared"] == pytest.approx(row["CRPS_parent_prepared"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA checkpoint restore requires a CUDA device")
def test_cuda_checkpoint_resume_restores_cpu_rng_byte_tensors(tmp_path, monkeypatch):
    module = runner()
    options = dict(width=8, observation_dimensions=2, learning_rate=0.001, epochs=2,
        patience=2, batch_size=16, seeds=[101], reconstruction_weight=1.0, source_weight=1.0,
        amplitude_weight=0.25, ridge_alpha=1.0, reference_steps=20, patches=12,
        train_scenarios=["baseline"], evaluation_scenarios=["baseline"],
        data_counts={"train": 8, "selection": 8, "evaluation": 1})
    identity = "non-authorized-CUDA-resume-fixture"
    module.prepare(tmp_path, options, identity)
    original = module.atomic_checkpoint

    def checkpoint_then_interrupt(path, payload):
        original(path, payload)
        if Path(path).name == "checkpoint.pt":
            raise RuntimeError("synthetic fixture interruption after durable epoch")

    monkeypatch.setattr(module, "atomic_checkpoint", checkpoint_then_interrupt)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        module.fit(tmp_path, "distribution_randomized_generator", 101, options,
                   identity, torch.device("cuda:0"))
    path = tmp_path / "fits/distribution_randomized_generator/seed_101/checkpoint.pt"
    checkpoint = torch.load(path, map_location="cuda:0", weights_only=False)
    assert checkpoint["next_epoch"] == 1
    assert all(state.device.type == "cuda" for state in checkpoint["cuda_rng"])
    monkeypatch.setattr(module, "atomic_checkpoint", original)
    resumed = module.fit(tmp_path, "distribution_randomized_generator", 101, options,
                         identity, torch.device("cuda:0"))
    assert resumed["status"] == "complete"
    assert resumed["completed_epochs"] == 2
    assert [epoch["epoch"] for epoch in resumed["history"]] == [1, 2]


def test_finite_training_evaluation_export_and_resume(tmp_path):
    module = runner()
    options = dict(width=8, observation_dimensions=2, learning_rate=0.001, epochs=1,
        patience=1, batch_size=4, seeds=[101], reconstruction_weight=1.0, source_weight=1.0,
        amplitude_weight=0.25, ridge_alpha=1.0, reference_steps=20, patches=12,
        train_scenarios=["baseline"], evaluation_scenarios=["baseline", "exact_joint_alias"],
        data_counts={"train": 2, "selection": 2, "evaluation": 2})
    identity = "non-authorized-synthetic-fixture"
    module.prepare(tmp_path, options, identity)
    for arm in module.ARMS:
        fit_record = module.fit(tmp_path, arm, 101, options, identity, torch.device("cpu"))
        assert fit_record["status"] == "complete"
        assert fit_record["completed_epochs"] == 1
        same_record = module.fit(tmp_path, arm, 101, options, identity, torch.device("cpu"))
        assert same_record == fit_record
        evaluated = module.evaluate(tmp_path, arm, 101, options, identity, torch.device("cpu"))
        assert evaluated["expected_records"] == evaluated["completed_records"] == 16
        assert evaluated["completed_pairs"] == evaluated["expected_pairs"] == 8
        assert evaluated["finite_records"] == 16
        module.export(tmp_path, arm, 101, options, identity, torch.device("cpu"))
        with np.load(tmp_path / "exports" / arm / "seed_101" / "synthetic_evaluation.npz") as exported:
            assert exported["eeg_semantic_mean"].shape == (8, 12)
            assert exported["hb_observation_code"].shape == (8, 120, 2)
        with pytest.raises(ValueError, match="identity"):
            module.fit(tmp_path, arm, 101, options, identity + "changed", torch.device("cpu"))
    module.summarize(tmp_path, options, identity, require_public_probes=False)
    import json
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["expected_fits"] == 4
    import csv
    with (tmp_path / "scenario_variant_metrics.csv").open(newline="") as handle:
        variants = list(csv.DictReader(handle))
    alias_rows = [row for row in variants if row["scenario"] == "exact_joint_alias"]
    assert len(alias_rows) == 16
    assert {row["variant"] for row in alias_rows} == {"control", "intervention"}
    assert all(row["records"] == row["unique_evaluation_base_ids"] == "2" for row in alias_rows)
    public_feature_fixture(tmp_path / "features")
    contract = dict(measured=dict(prefix_steps=[40, 0], score_start=80, score_stop=120),
                    baselines=dict(ridge_candidates=[0.1, 1.0]))
    public_record = module.probe(tmp_path, "distribution_randomized_generator", 101, options,
        identity, torch.device("cpu"), tmp_path / "features", contract)
    assert public_record["status"] == "complete"
    assert public_record["planned_rows"] == public_record["completed_rows"] == 90
    assert public_record["failed_rows"] == 0
    assert len(public_record["contrasts"]) == 6
    for prefix in (0, 40):
        group = {row["method"]: row for row in public_record["selections"]
                 if row["region"] == "r0" and row["prefix_steps"] == prefix}
        assert group["context_plus_O_plus_S"]["feature_dimensions"] == group[
            "context_plus_capacity_matched_observation_only"]["feature_dimensions"]
    module.summarize(tmp_path, options, identity)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "incomplete"
    assert summary["synthetic_status"] == "complete"
    assert summary["public_probe_status"] == "incomplete"
    assert summary["public_probe_planned_rows"] == summary["public_probe_completed_rows"] == 90
    assert len(summary["missing"]) == 3
    assert all(item["stage"] == "public_probes" for item in summary["missing"])
