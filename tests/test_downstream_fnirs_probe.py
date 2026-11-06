"""Pure-fNIRS adaptation, mask semantics and subject-isolated development reads."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from comparative_methods.MtReNIRS.rebench_v1 import FrozenMtReNIRS, geometry_from_profile, normalize_window
from comparative_methods.rebench_baselines_v1 import fnirs_block_covariances, fnirs_tangent_features
from comparative_methods.performance_analysis import audit_downstream_inputs_v1 as audit


def config():
    return audit.read_config(Path(__file__).parents[1] / "comparative_methods/downstream_fnirs_probe_v1.yaml")


def test_window_normalization_ignores_padding_and_other_samples():
    rng = np.random.default_rng(9)
    x = torch.tensor(rng.normal(size=(3, 4, 20)), dtype=torch.float32)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[0, 1] = False; mask[1, 2, 10:] = False
    first = normalize_window(x, mask)
    changed = x.clone(); changed[~mask] = float("nan"); changed[2] *= 1000
    second = normalize_window(changed, mask)
    torch.testing.assert_close(first[:2], second[:2], rtol=0, atol=0)
    assert torch.isfinite(second).all() and not second[~mask].any()


def test_template_alignment_preserves_known_coordinate_frame_and_chromophores():
    points = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., -1., -1.]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    reference = dict(labels=["A", "B", "C", "D"], positions=points * .09)
    native = points @ rotation * 3. + np.array([2., -5., 1.])
    def entry(name, pos, **kwargs):
        return dict(channel_name=name, position_available=True, x=pos[0], y=pos[1], z=pos[2], **kwargs)
    names = ["pair_HbO", "pair_HbR"]
    profile = dict(fnirs_channel_names=names,
        fnirs_geometry=[entry(n, native[1], component=k, base_channel_name="unknown") for n, k in zip(names, ["HbO", "HbR"])],
        eeg_geometry=[entry(n, p) for n, p in zip(reference["labels"], native)])
    kinds, positions, valid, details = geometry_from_profile(profile, reference)
    np.testing.assert_allclose(positions, np.repeat(reference["positions"][[1]], 2, axis=0), atol=1e-8)
    assert kinds == ["hbo", "hbr"] and valid.all() and details["anchor_rms_m"] < 1e-12
    profile["fnirs_channel_names"] = names[::-1]
    with pytest.raises(ValueError, match="order"):
        geometry_from_profile(profile, reference)


def test_short_input_and_missing_channels_reach_attention_as_declared():
    class Capture(torch.nn.Module):
        def forward(self, signal, coordinates, indices, valid):
            self.signal, self.valid, self.indices = signal.clone(), valid.clone(), indices.clone()
            return signal[..., :1].expand(-1, -1, 128) * valid.unsqueeze(-1)
    model = FrozenMtReNIRS.__new__(FrozenMtReNIRS)
    torch.nn.Module.__init__(model)
    model.expected_samples, model.channels, model.windows = 20, 4, 1
    model.encoder = Capture()
    model.register_buffer("coordinates", torch.zeros(4, 4))
    x = torch.arange(80, dtype=torch.float32).reshape(1, 4, 20)
    mask = torch.ones_like(x, dtype=torch.bool); mask[:, 1] = False
    x[:, 1] = float("nan")
    output, valid = model.tokens(x, mask)
    assert output.shape == (1, 1, 4, 128) and valid.tolist() == [[[True, False, True, True]]]
    assert not model.encoder.signal[..., 20:].any() and not output[:, :, 1].any()
    torch.testing.assert_close(model.encoder.signal[..., :20], normalize_window(x, mask))


def test_refed_planar_geometry_uses_named_neighbours_without_inventing_3d_registration():
    reference = dict(labels=["A", "B", "C"], positions=np.array([[0., 0., .1], [.1, 0., 0.], [0., .1, 0.]]))
    def point(name, x, y, **extra):
        return dict(channel_name=name, x=x, y=y, z=0., position_available=True,
                    coordinate_system="refed_paper_layout_2d_v1", **extra)
    profile = dict(fnirs_channel_names=["CH1_HbO", "CH1_HbR"],
        fnirs_geometry=[point("CH1_"+kind, .91, .05, component=kind, base_channel_name="CH1") for kind in ["HbO", "HbR"]],
        eeg_geometry=[point("A", 0., 0.), point("B", 1., 0.), point("C", 0., 1.)])
    _, positions, valid, details = geometry_from_profile(profile, reference)
    np.testing.assert_allclose(positions, np.repeat(reference["positions"][[1]], 2, axis=0))
    assert valid.all() and details["canonical_labels"] == ["B", "B"]
    assert details["method"] == "refed_registered_2d_nearest_eeg_name_to_model_template"
    assert "anchor_rms_m" not in details
    for g in profile["fnirs_geometry"]: g["coordinate_system"] = "unregistered_plane"
    with pytest.raises(ValueError, match="noncoplanar"):
        geometry_from_profile(profile, reference)


def test_covariance_keeps_missing_values_out_and_is_positive_definite():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 6, 40)); mask = np.ones(x.shape, bool)
    mask[0, 1] = False; mask[1, :, 30:] = False
    cov, coverage = fnirs_block_covariances(x, mask, ["hbo"] * 3 + ["hbr"] * 3)
    changed = x.copy(); changed[~mask] = np.nan
    same, _ = fnirs_block_covariances(changed, mask, ["hbo"] * 3 + ["hbr"] * 3)
    np.testing.assert_array_equal(cov, same)
    assert (np.linalg.eigvalsh(cov) > 0).all() and coverage[0, 1] == 0
    scaled, _ = fnirs_block_covariances(x * 1e-12, mask, ["hbo"] * 3 + ["hbr"] * 3)
    np.testing.assert_allclose(cov, scaled, atol=1e-12)


def test_riemann_reference_and_training_features_ignore_held_covariances():
    rng = np.random.default_rng(22)
    raw = rng.normal(size=(9, 2, 3, 3))
    covariance = raw @ raw.swapaxes(-1, -2) + np.eye(3)
    first, references = fnirs_tangent_features(covariance, np.arange(6))
    changed = covariance.copy(); changed[6:] *= 1e4
    second, other = fnirs_tangent_features(changed, np.arange(6))
    np.testing.assert_array_equal(references, other)
    np.testing.assert_array_equal(first[:6], second[:6])


@pytest.mark.parametrize("alter", ["test", "refed"])
def test_probe_contract_does_not_expand_signal_or_target_scope(tmp_path, alter):
    cfg = config()
    if alter == "test": cfg["source_partitions"].append("test")
    else: cfg["tasks"].append("refed_regression")
    path = tmp_path / "bad.yaml"; path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="development population"):
        audit.read_config(path)


def make_parent(tmp_path, task):
    rng = np.random.default_rng(5)
    classes = len(audit.TASKS[task]["classes"])
    labels = np.tile(np.arange(classes), 12)
    subjects = np.repeat([f"s{i}" for i in range(6)], 2 * classes)
    n = len(labels)
    names = ["L_HbO", "R_HbO", "L_HbR", "R_HbR"]
    profile = dict(fnirs_channel_names=names, fnirs_geometry=[dict(component=kind) for kind in ["HbO"] * 2 + ["HbR"] * 2],
        fnirs_sampling_rate_hz=10., fnirs_samples=300, channel_names=["C3"],
        sampling_rate_hz=200., input_samples=6000, duration_s=30., window_start_s=-5., input_window_schema="downstream_input_v1")
    arrays = dict(target=labels, mask=np.ones(n, bool), subject_id=subjects,
        partition=np.where(np.isin(subjects, ["s4", "s5"]), "validation", "train"),
        sample_id=np.array([str(i) for i in range(n)]), record_id=np.array(["record"] * n),
        fnirs=rng.normal(size=(n, 4, 300)).astype(np.float32), fnirs_mask=np.ones((n, 4, 300), bool))
    arrays["fnirs"][:, 0] += labels[:, None] * 2
    cfg = config(); cfg["parent_run"] = "parent"; cfg["tasks"] = [task]
    cfg["models"] = {task: {"statistics": dict(features="fnirs_statistics", estimator="logistic", C=1.)}}
    root = tmp_path / "parent"; prepare = root / task / "prepare"; prepare.mkdir(parents=True)
    for name, values in arrays.items(): np.save(prepare / (name + ".npy"), values, allow_pickle=False)
    retained = dict(profile=profile, counts={p: int((arrays["partition"] == p).sum()) for p in ["train", "validation"]},
        subjects={p: sorted(set(subjects[arrays["partition"] == p])) for p in ["train", "validation"]})
    audit.write(root / "plan.json", dict(schema=audit.SCHEMA, scope="public_train_validation_diagnostic_not_final_paper_ranking", tasks={task: retained}))
    audit.write(root / "summary.json", dict(status="completed"))
    return cfg, arrays


@pytest.mark.parametrize("task", ["motor_imagery", "nback"])
def test_pure_fnirs_probe_scores_binary_and_multiclass_without_eeg_files(tmp_path, task):
    cfg, arrays = make_parent(tmp_path, task)
    run = tmp_path / "run"
    audit.probe_plan(cfg, tmp_path, run, tmp_path, "fnirs.yaml")
    features = run / task / "features"
    audit.fnirs_probe_extract(cfg, tmp_path, run, task, features, "cpu")
    assert json.loads((features / "features.json").read_text())["signal_arrays_read"] == ["fnirs", "fnirs_mask"]
    for fold in range(3):
        output = run / task / "statistics" / f"fold_{fold}"; output.mkdir(parents=True)
        result = audit.probe_fit(cfg, tmp_path, run, task, "statistics", fold, output, "cpu")
        audit.write(output / "result.json", dict(status="completed", result=result))
        assert set(result["training_subjects"]).isdisjoint(result["evaluation_subjects"])
    assert audit.probe_aggregate(cfg, tmp_path, run)["status"] == "completed"
    verification = audit.probe_verify(cfg, tmp_path, run)
    assert verification["status"] == "passed" and verification["fold_predictions"] == 3
    # Fail before any fNIRS array is opened if the retained identity crosses test.
    path = tmp_path / "parent" / task / "prepare" / "partition.npy"
    changed = arrays["partition"].astype("U20"); changed[0] = "test"; np.save(path, changed)
    with pytest.raises(ValueError, match="original development support"):
        audit.probe_inputs(cfg, tmp_path, task, signals="fnirs")
