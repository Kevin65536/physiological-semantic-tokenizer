"""Small synthetic API/gradient checks, independent of local assets and data."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from comparative_methods.NormWear import rebench_v1 as adapter
from comparative_methods.NormWear.adapters.normwear import NormWearFrozenEncoder


class _Patch(nn.Module):
    patch_size = (9, 5)

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 768, self.patch_size, stride=self.patch_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.proj(values).flatten(2).transpose(1, 2)


class _TokenMap(nn.Module):
    def __init__(self, *, fusion: bool = False) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((768,), 0.01))
        self.fusion = fusion

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        context = values.mean(dim=1, keepdim=True) if self.fusion else values.roll(1, dims=1)
        return values + self.scale * torch.tanh(context)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _Patch()
        self.pos_embed = nn.Parameter(torch.zeros(1, 560, 768), requires_grad=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768) * 0.01)
        self.encoder_blocks = nn.ModuleList()
        for layer in range(12):
            block = nn.Module()
            block.curr_layer = layer
            block.fuse_frequency = 2
            block.no_fusion = False
            block.mean_fuse = False
            block.variate_encoder = _TokenMap()
            if layer % 2 == 0:
                block.cls_fusion = _TokenMap(fusion=True)
            self.encoder_blocks.append(block)
        self.norm = nn.LayerNorm(768)

    def pos_adjust(self, size: tuple[int, int], *, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, (size[0] // 9) * (size[1] // 5) + 1, 768, device=device)


def _cwt(values: torch.Tensor, low: float, high: float) -> torch.Tensor:
    assert (low, high) == (0.1, 64)
    first = values[:, 1:] - values[:, :-1]
    second = first[:, 1:] - first[:, :-1]
    planes = torch.stack((values[:, 2:], first[:, 1:], second), dim=1)
    return planes.unsqueeze(-1).expand(-1, -1, -1, 65)


def _profile(**overrides: object) -> dict:
    return {
        "channel_names": ["Cz"],
        "fnirs_location_names": ["L1"],
        "fnirs_channel_names": ["HbO:L1", "HbR:L1"],
        "sampling_rate_hz": 200,
        "fnirs_sampling_rate_hz": 10,
        "duration_s": 2,
        "input_samples": 400,
        "fnirs_samples": 20,
        "channel_chunk_size": 2,
        **overrides,
    }


@pytest.fixture(autouse=True)
def _small_cpu_thread_budget() -> None:
    # Tiny fixtures should not schedule a machine-wide numerical thread pool.
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def _model(*, trainable: bool, profile: dict | None = None) -> adapter.NormWearRebenchEncoder:
    return adapter.NormWearRebenchEncoder(
        _Backbone(),
        SimpleNamespace(cwt_wrap=_cwt),
        task="dsr",
        profile=profile or _profile(),
        trainable=trainable,
        checkpoint_metadata={"artifact_id": "synthetic_fixture_only"},
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    return (
        torch.randn(1, 1, 400, generator=generator),
        torch.randn(1, 2, 20, generator=generator),
    )


def test_frozen_features_match_retained_computation_and_only_head_receives_gradients() -> None:
    model = _model(trainable=False)
    eeg, fnirs = _inputs()
    # The reference uses the retained implementation with exactly the same weights.
    reference = NormWearFrozenEncoder(
        deepcopy(model.backbone), model.upstream_module, channel_chunk_size=2
    )
    expected = reference(
        eeg,
        fnirs[:, :1],
        fnirs[:, 1:],
        eeg_sampling_rate_hz=200,
        fnirs_sampling_rate_hz=10,
        eeg_channel_names=["Cz"],
        fnirs_location_names=["L1"],
    )
    model.train()
    features = model(eeg, fnirs)
    torch.testing.assert_close(features, expected, rtol=0, atol=0)
    assert features.shape == (1, model.feature_dim) == (1, 2304)
    assert not features.requires_grad
    assert not model.backbone.training
    head = nn.Linear(model.feature_dim, 2)
    head(features).square().sum().backward()
    assert head.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_finetuning_reaches_patch_first_last_layer_and_complete_channel_fusion() -> None:
    model = _model(trainable=True)
    eeg, fnirs = _inputs()
    eeg.requires_grad_()
    fnirs.requires_grad_()
    assert model.backbone.training
    features = model(eeg, fnirs)
    head = nn.Linear(model.feature_dim, 2)
    head(features).square().sum().backward()
    gradient_names = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
    assert "backbone.patch_embed.proj.weight" in gradient_names
    assert "backbone.encoder_blocks.0.variate_encoder.scale" in gradient_names
    assert "backbone.encoder_blocks.11.variate_encoder.scale" in gradient_names
    assert "backbone.encoder_blocks.0.cls_fusion.scale" in gradient_names
    assert "backbone.pos_embed" not in gradient_names
    assert eeg.grad is None and fnirs.grad is None
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert bool(torch.isfinite(parameter.grad).all()), name
            assert float(parameter.grad.abs().max()) > 0, name


def test_gradient_checkpointing_preserves_features_and_parameter_gradients() -> None:
    checkpointed = _model(trainable=True)
    ordinary = _model(trainable=True, profile=_profile(gradient_checkpointing=False))
    ordinary.load_state_dict(checkpointed.state_dict())
    eeg, fnirs = _inputs()
    expected = ordinary(eeg, fnirs)
    actual = checkpointed(eeg, fnirs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.linspace(-1, 1, checkpointed.feature_dim).unsqueeze(0)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    for (name, left), (other_name, right) in zip(
        checkpointed.named_parameters(), ordinary.named_parameters(), strict=True
    ):
        assert name == other_name
        if left.requires_grad:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("task", "change", "match"),
    [
        ("refed_regression", {}, "NO_PARTIAL_TIME_MASK"),
        ("dsr", {"sampling_rate_hz": 100}, "EEG 200 Hz"),
        ("dsr", {"duration_s": 8}, "2 seconds"),
        ("dsr", {"input_samples": 1600}, "input_samples"),
        ("dsr", {"fnirs_samples": 80}, "fnirs_samples"),
        ("dsr", {"channel_names": ["Cz", "Cz"]}, "unique"),
        ("dsr", {"fnirs_location_names": []}, "unique"),
        ("dsr", {"fnirs_channel_names": ["HbO:L1"]}, "one HbO"),
        ("dsr", {"eeg_channel_names": ["Fz"]}, "aliases disagree"),
    ],
)
def test_invalid_contract_is_rejected_before_checkpoint_access(
    monkeypatch: pytest.MonkeyPatch, task: str, change: dict, match: str
) -> None:
    def forbidden_loader(**_: object) -> None:
        pytest.fail("invalid or unsupported input must not read source/weights")

    monkeypatch.setattr(adapter, "load_verified_normwear_encoder", forbidden_loader)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        adapter.create_model(task, _profile(**change), "cpu")


def test_wrong_input_shape_missing_modality_and_partial_support_are_rejected() -> None:
    model = _model(trainable=False)
    eeg, fnirs = _inputs()
    with pytest.raises(ValueError, match="paired HbO/HbR"):
        model(eeg, None)
    with pytest.raises(ValueError, match="EEG must have shape"):
        model(eeg[..., :-1], fnirs)
    with pytest.raises(ValueError, match="fNIRS must have shape"):
        model(eeg, fnirs[:, :1])
    with pytest.raises(ValueError, match="missing or padded EEG"):
        model(eeg, fnirs, eeg_sample_valid=torch.zeros(1, 400, dtype=torch.bool))
    with pytest.raises(ValueError, match="missing or padded channel"):
        model(eeg, fnirs, channel_valid=torch.zeros(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="missing or padded fNIRS"):
        model(eeg, fnirs, fnirs_sample_valid=torch.zeros(1, 20, dtype=torch.bool))
    with pytest.raises(TypeError, match="unsupported NormWear forward fields"):
        model(eeg, fnirs, mask=torch.ones(1, 400, dtype=torch.bool))


@pytest.mark.parametrize("trainable", [False, True])
def test_missing_native_channel_retains_windows_and_uses_declared_training_mean_imputation(
    trainable: bool,
) -> None:
    model = _model(trainable=trainable)
    eeg, fnirs = (value.repeat(2, 1, 1) for value in _inputs())
    eeg_mask = torch.ones_like(eeg, dtype=torch.bool)
    fnirs_mask = torch.ones_like(fnirs, dtype=torch.bool)
    # First window lacks one entire native HbO channel; the second is complete.
    fnirs_mask[0, 0, :] = False
    fnirs[~fnirs_mask] = 0
    original_mask = fnirs_mask.clone()
    with torch.no_grad():
        expected = model(eeg, fnirs)
    actual = model(eeg, fnirs, eeg_mask=eeg_mask, fnirs_mask=fnirs_mask)
    assert actual.shape == (2, model.feature_dim)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(fnirs_mask, original_mask)
    assert model.method_identity["input_imputation"]["native_mask_support"] is False
    assert model.method_identity["mask_handling"] == "train_mean_standardized_zero_imputation_preserve_input_masks"
    if trainable:
        nn.Linear(model.feature_dim, 2)(actual).square().sum().backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        assert all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients)
        assert model.backbone.patch_embed.proj.weight.grad.abs().max() > 0
        assert model.backbone.encoder_blocks[-1].variate_encoder.scale.grad.abs().max() > 0


def test_input_masks_validate_standardized_zero_fill_and_cannot_hide_nonzero_or_nan_values() -> None:
    model = _model(trainable=False)
    eeg, fnirs = _inputs()
    mask = torch.ones_like(fnirs, dtype=torch.bool)
    mask[:, 0, :] = False
    for hidden_value in (3.0, float("nan")):
        contaminated = fnirs.clone()
        contaminated[~mask] = hidden_value
        with pytest.raises(ValueError, match="training-mean standardized zero fill"):
            model(eeg, contaminated, fnirs_mask=mask)
    with pytest.raises(ValueError, match="bool.*matching input"):
        model(eeg, fnirs, fnirs_mask=mask.float())
    with pytest.raises(ValueError, match="bool.*matching input"):
        model(eeg, fnirs, fnirs_mask=mask[:, 0])


@pytest.mark.parametrize("task", sorted(adapter.SUPPORTED_TASKS))
def test_common_profile_resolves_both_supported_window_lengths(task: str) -> None:
    duration = 2 if task == "dsr" else 8
    profile = _profile(duration_s=duration, input_samples=duration * 200, fnirs_samples=duration * 10)
    resolved = adapter._profile(task, profile)
    assert resolved["input_samples"] == duration * 200
    assert resolved["fnirs_samples"] == duration * 10


def test_factory_uses_verified_loader_and_serializable_source_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class _Metadata:
        path: Path
        artifact_id: str

    calls = []

    def loader(*, device: str) -> tuple:
        calls.append(device)
        return (
            _Backbone(),
            SimpleNamespace(cwt_wrap=_cwt),
            _Metadata(Path("synthetic-not-read.pth"), "synthetic_fixture_only"),
        )

    monkeypatch.setattr(adapter, "load_verified_normwear_encoder", loader)
    model = adapter.create_model("dsr", _profile(), "cpu", trainable=True)
    assert calls == ["cpu"]
    assert model.trainable
    assert model.method_identity["checkpoint"]["path"] == "synthetic-not-read.pth"


def test_identity_reports_actual_adaptation_and_does_not_add_input_normalization() -> None:
    model = _model(trainable=True)
    identity = model.method_identity
    assert identity["method_id"] == "normwear_eeg_fnirs_adapted"
    assert identity["track"] == "full_finetune"
    assert identity["pretraining_included_fnirs"] is False
    assert identity["stateless_transforms"]["upstream_basic_preproc_applied"] is False
    eeg, fnirs = _inputs()
    with torch.no_grad():
        prepared, names = model.prepare_model_input(
            eeg,
            fnirs[:, :1],
            fnirs[:, 1:],
            eeg_sampling_rate_hz=200,
            fnirs_sampling_rate_hz=10,
            eeg_channel_names=["Cz"],
            fnirs_location_names=["L1"],
        )
        scaled, _ = model.prepare_model_input(
            3 * eeg,
            3 * fnirs[:, :1],
            3 * fnirs[:, 1:],
            eeg_sampling_rate_hz=200,
            fnirs_sampling_rate_hz=10,
            eeg_channel_names=["Cz"],
            fnirs_location_names=["L1"],
        )
    assert names == ("eeg:Cz", "fnirs_hbo:L1", "fnirs_hbr:L1")
    torch.testing.assert_close(scaled, 3 * prepared, rtol=2e-6, atol=2e-6)
