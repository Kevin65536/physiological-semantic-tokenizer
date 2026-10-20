"""Asset-independent regression checks for the versioned EEG FT boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


CHANNELS = tuple(f"C{index}" for index in range(16))
METHODS = (("BIOT", 256), ("CBraMod", 200), ("REVE", 512))


@dataclass
class Identity:
    n_channels: int = 16
    path: Path = Path("/synthetic/non-authorizing-checkpoint")
    sha256: str = "0" * 64


class BIOTFixture(nn.Module):
    n_fft = 200
    hop_length = 100

    def __init__(self):
        super().__init__()
        self.index = nn.Parameter(torch.arange(16), requires_grad=False)
        self.channel_tokens = nn.Embedding(16, 256)
        self.projection = nn.Linear(1, 256)
        self.dropout = nn.Dropout(0.5)

    def forward(self, eeg):
        return self.dropout(self.projection(eeg.mean((1, 2)).unsqueeze(1)) + self.channel_tokens.weight.mean(0))


class CBraModFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Module()
        self.patch_embedding.mask_encoding = nn.Parameter(torch.zeros(200), requires_grad=False)
        self.proj_out = nn.Identity()
        self.projection = nn.Linear(200, 200)
        self.dropout = nn.Dropout(0.5)

    def forward(self, patches):
        return self.dropout(self.projection(patches))


class REVEFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(embed_dim=512, patch_size=200, patch_overlap=20)
        self.final_layer = nn.Identity()
        self.cls_query_token = nn.Parameter(torch.randn(1, 1, 512))
        self.projection = nn.Linear(200, 512)
        self.dropout = nn.Dropout(0.5)
        self.last_positions = None

    def forward(self, eeg, pos):
        self.last_positions = pos.detach().clone()
        return self.dropout(self.projection(eeg.unfold(2, 200, 180)))

    def attention_pooling(self, tokens):
        flattened = tokens.flatten(1, 2)
        weights = torch.softmax(self.cls_query_token @ flattened.transpose(-1, -2) / 512**0.5, -1)
        return (weights @ flattened).squeeze(1)


class PositionFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapping = {name: index for index, name in enumerate(CHANNELS)}
        self.embedding = nn.Parameter(torch.arange(48, dtype=torch.float32).reshape(16, 3) / 1000)


@pytest.fixture(params=METHODS, ids=lambda item: item[0])
def adapter(request, monkeypatch):
    name, dimension = request.param
    module = importlib.import_module(f"comparative_methods.{name}.rebench_v1")
    encoder = {"BIOT": BIOTFixture, "CBraMod": CBraModFixture, "REVE": REVEFixture}[name]()
    encoder.requires_grad_(False).eval()
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        if name == "REVE":
            return deepcopy(encoder), PositionFixture(), Identity()
        return deepcopy(encoder), Identity()

    monkeypatch.setattr(module, {"BIOT": "load_verified_biot_encoder", "CBraMod": "load_verified_cbramod_encoder", "REVE": "load_verified_reve_base"}[name], load)
    return name, dimension, module, calls


def profile(task="dsr"):
    return {"channel_names": CHANNELS, "sampling_rate_hz": 200, "input_samples": 400 if task == "dsr" else 1600}


def test_lp_matches_retained_pooling_and_stays_deterministic_in_train_mode(adapter):
    name, dimension, module, _ = adapter
    model = module.create_model("dsr", profile(), "cpu")
    model.train()
    assert model.encoder.training is False
    assert not any(parameter.requires_grad for parameter in model.parameters())
    eeg = torch.randn(2, 16, 400)
    features = model(eeg)
    assert features.shape == (2, dimension)
    assert features.requires_grad is False
    torch.testing.assert_close(model(eeg), features, rtol=0, atol=0)
    frozen_module = importlib.import_module(f"comparative_methods.{name}.adapters.{name.lower()}")
    wrapper = getattr(frozen_module, f"{name}FrozenEncoder")
    retained = wrapper(model.encoder, model.position_bank) if name == "REVE" else wrapper(model.encoder)
    expected = retained(eeg, sampling_rate_hz=200, channel_names=CHANNELS)
    torch.testing.assert_close(features, expected, rtol=0, atol=0)
    head = nn.Linear(dimension, 2)
    head(features).square().mean().backward()
    assert head.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.encoder.parameters())


def test_ft_propagates_and_updates_real_backbone_parameters(adapter):
    name, dimension, module, _ = adapter
    model = module.create_model("dsr", profile(), "cpu", trainable=True)
    model.train()
    assert model.encoder.training is True
    initial = model.encoder.projection.weight.detach().clone()
    optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.1)
    features = model(torch.randn(2, 16, 400))
    assert features.shape == (2, dimension)
    assert features.requires_grad is True
    features.square().mean().backward()
    assert model.encoder.projection.weight.grad is not None
    assert bool(torch.isfinite(model.encoder.projection.weight.grad).all())
    assert bool(model.encoder.projection.weight.grad.abs().sum() > 0)
    optimizer.step()
    assert not torch.equal(model.encoder.projection.weight, initial)
    if name == "BIOT":
        assert model.encoder.index.requires_grad is False
    if name == "CBraMod":
        assert model.encoder.patch_embedding.mask_encoding.requires_grad is False
    if name == "REVE":
        assert model.encoder.cls_query_token.requires_grad is True
        assert model.encoder.cls_query_token.grad is not None
        assert model.position_bank.training is False
        assert not any(parameter.requires_grad for parameter in model.position_bank.parameters())
        assert all(parameter.grad is None for parameter in model.position_bank.parameters())
        expected = model.position_bank.embedding.detach().unsqueeze(0).expand(2, -1, -1)
        torch.testing.assert_close(model.encoder.last_positions, expected)
    model.eval()
    assert model.encoder.training is False


def test_missing_support_wrong_panel_and_fnirs_are_rejected(adapter):
    _, _, module, _ = adapter
    model = module.create_model("dsr", profile(), "cpu", trainable=True)
    eeg = torch.randn(2, 16, 400)
    full = {"channel_valid": torch.ones(2, 16, dtype=torch.bool), "sample_valid": torch.ones(2, 400, dtype=torch.bool)}
    assert model(eeg, **full).shape[0] == 2
    for key in full:
        broken = {**full, key: full[key].clone()}
        broken[key].view(-1)[-1] = False
        with pytest.raises(ValueError, match="missing or padded"):
            model(eeg, **broken)
    assert model(eeg, eeg_mask=torch.ones_like(eeg, dtype=torch.bool)).shape[0] == 2
    broken_eeg_mask = torch.ones_like(eeg, dtype=torch.bool)
    broken_eeg_mask[0, 0, -1] = False
    with pytest.raises(ValueError, match="eeg_mask: missing or padded"):
        model(eeg, eeg_mask=broken_eeg_mask)
    with pytest.raises(ValueError, match="EEG only"):
        model(eeg, torch.randn(2, 2, 40))
    with pytest.raises(ValueError, match="channel order"):
        model(eeg, channel_names=tuple(reversed(CHANNELS)))
    with pytest.raises(ValueError, match="shape"):
        model(eeg[..., :-1])
    with pytest.raises(ValueError, match="finite"):
        model(eeg * float("nan"))
    with pytest.raises(TypeError, match="unsupported adapter arguments"):
        model(eeg, target_mask=torch.ones(2))


def test_unsupported_task_and_bad_profile_fail_before_weight_loading(adapter):
    _, _, module, calls = adapter
    with pytest.raises(NotImplementedError, match="temporal output"):
        module.create_model("refed_regression", profile(), "cpu")
    with pytest.raises(ValueError, match="16 distinct"):
        module.create_model("dsr", {**profile(), "channel_names": CHANNELS[:8] * 2}, "cpu")
    with pytest.raises(ValueError, match="200 Hz"):
        module.create_model("dsr", {**profile(), "sampling_rate_hz": 100}, "cpu")
    with pytest.raises(ValueError, match="1600"):
        module.create_model("visual", profile(), "cpu")
    assert not calls


def test_method_root_is_explicit_and_checkpoint_metadata_is_json_compatible(adapter, tmp_path):
    name, dimension, module, calls = adapter
    model = module.create_model("visual", {**profile("visual"), "method_root": str(tmp_path)}, "cpu")
    assert calls[0][1]["method_root"] == tmp_path
    assert model.feature_dim == dimension
    assert model.metadata["checkpoint"]["path"] == "/synthetic/non-authorizing-checkpoint"
    assert model.supports_mask is False
    assert model.supports_temporal_output is False
    if name == "REVE":
        assert model.metadata["reporting_track"] == "official_pretrained"
        overlap_model = module.create_model("motor_imagery", profile("motor_imagery"), "cpu")
        assert overlap_model.metadata["reporting_track"] == "open_world_pretrained_with_target_corpus_overlap"
        with pytest.raises(ValueError, match="no coordinates"):
            module.create_model("dsr", {**profile(), "channel_names": (*CHANNELS[:-1], "UNKNOWN")}, "cpu")


@pytest.mark.parametrize("trainable", (False, True))
def test_synthetic_smoke_reports_backbone_gradient_boundary(adapter, trainable):
    _, dimension, module, _ = adapter
    result = module.synthetic_smoke("dsr", profile(), "cpu", trainable=trainable)
    assert result["feature_shape"] == [2, dimension]
    assert result["peak_vram_bytes"] == 0
    assert result["trainable"] is trainable
    assert (result["backbone_grad_norm"] > 0) is trainable
    assert (result["backbone_grad_parameter_count"] > 0) is trainable
