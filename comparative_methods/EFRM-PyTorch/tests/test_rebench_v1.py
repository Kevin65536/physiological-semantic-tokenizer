"""Synthetic EFRM rebenchmark invariants; no retained measured data are read."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

METHOD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_ROOT))

import rebench_v1 as adapter


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def profile(duration: int = 2) -> dict:
    return {
        "channel_names": ["C3", "Cz", "C4"],
        # Deliberately permuted: component positions must follow identities.
        "fnirs_channel_names": ["B_HbR", "A_HbO", "B_HbO", "A_HbR"],
        "fnirs_location_names": ["A", "B"],
        "sampling_rate_hz": 200, "fnirs_sampling_rate_hz": 10,
        "duration_s": duration,
        "efrm_model_config": {
            "embed_dim": 16, "depth": 1, "num_heads": 4,
            "decoder_embed_dim": 16, "decoder_depth": 1,
            "decoder_num_heads": 4, "mlp_ratio": 2,
            "activation_checkpointing": False,
        },
    }


def inputs(duration: int = 2):
    return torch.randn(2, 3, 200 * duration), torch.randn(2, 4, 10 * duration)


def provenance(**updates):
    result = {
        "input_identity": dict(adapter.INPUT_IDENTITY),
        "excluded_target_dataset": "simultaneous_eeg_nirs",
        "source_dataset_ids": ["eeg_fnirs_single_trial", "refed", "visual_cognitive_motivation"],
        "source_membership_identity": "synthetic-source-membership",
        "status": "completed_refit", "stage": "stage_b",
        "initialization": "fresh_random", "epochs_completed": 1, "seed": 17,
    }
    return {**result, **updates}


def test_component_identity_and_native_masks():
    eeg, fnirs = inputs()
    mask = torch.ones_like(fnirs, dtype=torch.bool)
    mask[0, 0, -1] = False  # B/HbR invalid makes the whole B patch invalid.
    batch = adapter.adapt_batch(eeg, fnirs, profile(), fnirs_mask=mask)
    torch.testing.assert_close(batch["fnirs"][1, 0, 0], fnirs[1, 1])
    torch.testing.assert_close(batch["fnirs"][1, 1, 0], fnirs[1, 3])
    assert not batch["fnirs_patch_valid"][0, 1, 0]
    assert batch["fnirs"][0, :, 1, -1].eq(0).all()
    with pytest.raises(ValueError, match="requires exactly one"):
        adapter.adapt_batch(eeg, fnirs, {**profile(), "fnirs_channel_names": ["A_HbO", "A_HbR", "B_HbO", "C_HbR"]})


def test_frozen_features_ignore_invalid_inputs_and_have_no_decoder_gradients():
    params = {**profile(), "synthetic_random_encoder": True}
    model = adapter.create_model("dsr", params, "cpu", trainable=True)
    eeg, fnirs = inputs()
    mask = torch.ones_like(eeg, dtype=torch.bool)
    mask[:, :, -50:] = False
    baseline = model(eeg, fnirs, eeg_mask=mask)
    altered = eeg.clone()
    altered[:, :, -50:] = 1e8
    torch.testing.assert_close(baseline, model(altered, fnirs, eeg_mask=mask))
    assert baseline.shape == (2, model.feature_dim)
    baseline.square().mean().backward()
    assert model.backbone.eeg_model.patch_embed.proj.weight.grad is not None
    assert model.backbone.eeg_model.decoder_pred.weight.grad is None
    assert not model.backbone.eeg_model.decoder_pred.weight.requires_grad


def test_refed_has_real_temporal_features_and_retains_partial_mask():
    model = adapter.create_model("refed_regression", {**profile(20), "synthetic_random_encoder": True}, "cpu")
    eeg, fnirs = inputs(20)
    eeg_mask = torch.ones_like(eeg, dtype=torch.bool)
    fnirs_mask = torch.ones_like(fnirs, dtype=torch.bool)
    eeg_mask[1, :, 200:] = False
    fnirs_mask[1, :, 10:] = False
    features = model(eeg, fnirs, eeg_mask=eeg_mask, fnirs_mask=fnirs_mask)
    support = model.sequence_feature_mask(eeg, fnirs, eeg_mask=eeg_mask, fnirs_mask=fnirs_mask)
    assert features.shape == (2, 20, 16)
    assert features[0].std(0).max() > 0.01
    assert support[1].tolist() == [True] + [False] * 19
    torch.testing.assert_close(features[1, 1:], torch.zeros_like(features[1, 1:]))
    altered = eeg.clone()
    altered[1, :, 200:] = 1e9
    torch.testing.assert_close(features, model(altered, fnirs, eeg_mask=eeg_mask, fnirs_mask=fnirs_mask))
    with pytest.raises(ValueError, match="unrecognized signal masks"):
        model(eeg, fnirs, target_mask=torch.ones(2, 2, 20, dtype=torch.bool))


def test_pretraining_really_updates_both_maes_and_contrastive_branch(tmp_path):
    model = adapter.create_pretraining_model(profile(), "cpu", task="dsr")
    eeg, fnirs = inputs()
    before = model.backbone.fnirs_model.patch_embed.proj.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = adapter.pretrain_step(model, eeg, fnirs, optimizer, microbatch_size=1, amp_dtype=None)
    assert metrics["optimizer_steps"] == 1
    assert metrics["pair_count"] == 2
    assert all(metrics[key] > 0 for key in ("eeg_reconstruction_loss", "fnirs_reconstruction_loss", "clip_alignment_loss"))
    assert not torch.equal(model.backbone.fnirs_model.patch_embed.proj.weight, before)
    output = tmp_path / "completed.pt"
    adapter.save_pretraining_checkpoint(model, output, provenance(), optimizer=optimizer)
    loaded = adapter.create_model("dsr", {**profile(), "pretrained_checkpoint": str(output)}, "cpu")
    base_profile = profile()
    base_profile.pop("efrm_model_config")
    with pytest.raises(ValueError, match="architecture disagrees"):
        adapter.create_model("dsr", {**base_profile, "pretrained_checkpoint": str(output)}, "cpu")
    model.eval()
    torch.testing.assert_close(loaded(eeg, fnirs), model(eeg, fnirs))
    assert not any(parameter.requires_grad for parameter in loaded.parameters())


def test_resume_preserves_stage_membership_rng_and_optimizer(tmp_path):
    model = adapter.create_pretraining_model(profile(), "cpu", task="dsr")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    eeg, fnirs = inputs()
    adapter.pretrain_step(model, eeg, fnirs, optimizer, amp_dtype=None)
    boundary = provenance(stage="stage_a", status="in_progress")
    output = tmp_path / "resume.pt"
    adapter.save_pretraining_checkpoint(model, output, boundary, optimizer=optimizer)
    expected_random = torch.rand(3)
    restored = adapter.create_pretraining_model(profile(), "cpu", task="dsr")
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=2e-3)
    adapter.load_pretraining_checkpoint(restored, output, expected_provenance=boundary, optimizer=restored_optimizer)
    torch.testing.assert_close(torch.rand(3), expected_random)
    assert restored.pretraining_steps == 1
    assert restored_optimizer.param_groups[0]["lr"] == 1e-3
    with pytest.raises(ValueError, match="resume boundary drift: seed"):
        adapter.load_pretraining_checkpoint(restored, output, expected_provenance={**boundary, "seed": 42})
    with pytest.raises(ValueError, match="completed fresh non-target refit"):
        adapter.create_model("dsr", {**profile(), "pretrained_checkpoint": str(output)}, "cpu")


def test_downstream_rejects_random_historical_wrong_domain_or_target_exposure(tmp_path):
    with pytest.raises(ValueError, match="needs a completed"):
        adapter.create_model("dsr", profile(), "cpu")
    old = tmp_path / "old.pt"
    torch.save({"schema": "efrm_sync_pretrain_checkpoint_v1"}, old)
    with pytest.raises(ValueError, match="historical/foreign"):
        adapter.create_model("dsr", {**profile(), "pretrained_checkpoint": old}, "cpu")
    model = adapter.create_pretraining_model(profile(), "cpu", task="dsr")
    with pytest.raises(ValueError, match="untrained"):
        adapter.save_pretraining_checkpoint(model, tmp_path / "random.pt", provenance())
    wrong_input = deepcopy(provenance())
    wrong_input["input_identity"]["cache_schema"] = "physiology_semantic_clean_v1"
    with pytest.raises(ValueError, match="measurement-v5"):
        adapter.save_pretraining_checkpoint(model, tmp_path / "domain.pt", wrong_input)
    with pytest.raises(ValueError, match="all three non-target"):
        adapter.save_pretraining_checkpoint(model, tmp_path / "leaky.pt", provenance(source_dataset_ids=list(adapter.TASK_DATASETS.values())))


def test_plain_pretraining_loss_is_differentiable_and_not_counted_as_update():
    model = adapter.create_pretraining_model(profile(), "cpu")
    eeg, fnirs = inputs()
    loss = adapter.pretraining_loss(model, eeg, fnirs)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.backbone.eeg_model.decoder_pred.weight.grad is not None
    assert model.backbone.fnirs_model.decoder_pred.weight.grad is not None
    assert model.pretraining_steps == 0  # Only an optimizer update establishes training.
