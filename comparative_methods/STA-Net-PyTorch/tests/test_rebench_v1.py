from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / "rebench_v1.py"
SPEC = importlib.util.spec_from_file_location("sta_downstream_rebench_v1", SOURCE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.fixture(autouse=True)
def small_synthetic_thread_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _profile(task):
    duration = module.TASKS[task][2]
    return {
        "duration_s": duration, "sampling_rate_hz": 10, "fnirs_sampling_rate_hz": 2,
        "eeg_geometry": [[-1, -1], [-1, 1], [1, -1], [1, 1]],
        "fnirs_geometry": [[-0.8, -1], [-0.8, 1], [0.8, -1], [0.8, 1]],
        "embedding_dim": 8, "attention_heads": 2, "attention_key_dim": 4, "dropout": 0,
    }


@pytest.mark.parametrize("task", module.TASKS)
def test_all_task_shapes_and_no_added_time_support(task):
    model = module.create_model(task, _profile(task), "cpu").eval()
    duration = module.TASKS[task][2]
    eeg, fnirs = torch.randn(2, 4, duration * 10), torch.randn(2, 8, duration * 2)
    with torch.no_grad():
        prediction = model(eeg, fnirs)
    expected = (2, module.TASKS[task][1], 20) if task == "refed_regression" else (2, module.TASKS[task][1])
    assert tuple(prediction.shape) == expected
    assert bool(torch.isfinite(prediction).all())
    assert model.spec["extra_temporal_context_s"] == 0
    assert all(start >= 0 and start + model.segment_samples <= duration * 2 for start in model.segment_starts)
    if task != "refed_regression":
        torch.testing.assert_close(prediction.exp().sum(-1), torch.ones(2))


def test_masked_inputs_never_change_prediction_and_unknown_geometry_is_rejected():
    model = module.create_model("dsr", _profile("dsr"), "cpu", trainable=False)
    eeg, fnirs = torch.randn(2, 4, 20), torch.randn(2, 8, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[:, 0] = False
    first = model(eeg, fnirs, eeg_mask=mask)
    eeg[:, 0] = float("nan")
    second = model(eeg, fnirs, eeg_mask=mask)
    torch.testing.assert_close(first, second)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    profile = _profile("dsr")
    profile.pop("eeg_geometry")
    with pytest.raises(ValueError, match="real EEG geometry"):
        module.create_model("dsr", profile, "cpu")


def test_refed_native_sequence_loss_masks_missing_targets_and_backpropagates():
    torch.manual_seed(17)
    model = module.create_model("refed_regression", _profile("refed_regression"), "cpu")
    eeg, fnirs = torch.randn(2, 4, 200), torch.randn(2, 8, 40)
    target = torch.randn(2, 2, 20)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, :, -3:] = False
    first = model.supervised_loss(eeg, fnirs, target, mask)
    target[:, :, -3:] = float("nan")
    second = model.supervised_loss(eeg, fnirs, target, mask)
    torch.testing.assert_close(first["main"], second["main"])
    second["total"].backward()
    assert bool(torch.isfinite(second["total"]))
    assert model.backbone.eeg_head.weight.grad is not None
    assert model.backbone.block1.fgsa.channel_pooling.weight.grad is not None


def test_common_window_mismatch_is_a_failure_not_silent_crop():
    model = module.create_model("dsr", _profile("dsr"), "cpu")
    with pytest.raises(ValueError, match="common encoding window"):
        model(torch.randn(2, 4, 30), torch.randn(2, 8, 4))


def test_full_hb_channel_geometry_checks_paired_names_coordinates_and_order():
    profile = _profile("dsr")
    coordinates = profile["fnirs_geometry"]
    profile["fnirs_location_names"] = [f"loc{index}" for index in range(4)]
    profile["fnirs_channel_names"] = [f"loc{index}_{role}" for role in ("HbO", "HbR") for index in range(4)]
    profile["fnirs_geometry"] = coordinates + coordinates
    model = module.create_model("dsr", profile, "cpu")
    assert model.fnirs_projection.shape == (256, 4)
    profile["fnirs_geometry"] = list(profile["fnirs_geometry"])
    profile["fnirs_geometry"][-1] = [99, 99]
    with pytest.raises(ValueError, match="geometry must match"):
        module.create_model("dsr", profile, "cpu")
