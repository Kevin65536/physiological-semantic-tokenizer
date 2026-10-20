"""Measurement-v5 EFRM retraining and downstream features for rebenchmark v1.

No historical launcher, split, checkpoint or cache is used here.  The caller
owns the target-excluded source membership, train-only scaling and durable
training loop.  This module owns the native EFRM model, masks and objectives.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import torch
from torch import nn

METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from efrm_pytorch.model import EFRMSyncModel, sincos_2d
from efrm_pytorch.training import cached_pretrain_backward


METHOD_IDENTITY = "efrm_measurement_v5_target_excluded_rebench_v1"
OUTPUT_KIND = "features"
CHECKPOINT_SCHEMA = "efrm_measurement_v5_rebench_pretrain_checkpoint_v1"
TASK_DATASETS = {
    "motor_imagery": "eeg_fnirs_single_trial",
    "mental_arithmetic": "eeg_fnirs_single_trial",
    "wg": "simultaneous_eeg_nirs",
    "nback": "simultaneous_eeg_nirs",
    "dsr": "simultaneous_eeg_nirs",
    "visual": "visual_cognitive_motivation",
    "refed_regression": "refed",
}
INPUT_IDENTITY = {
    "cache_schema": "physiology_semantic_clean_v5",
    "preprocessing_id": "physiology_measurement_alignment_v5",
    "coordinate": "measurement",
    "motion_method": "none",
}
DEFAULT_MODEL_CONFIG = {
    "eeg_patch_samples": 50, "fnirs_patch_samples": 20,
    "mask_ratio": 0.5, "embed_dim": 768, "depth": 12, "num_heads": 12,
    "decoder_embed_dim": 512, "decoder_depth": 8, "decoder_num_heads": 16,
    "mlp_ratio": 4.0, "clip_logit_multiplier": 0.1,
    "activation_checkpointing": True,
}
PRETRAINING_BUDGET = {
    "max_epochs": 100, "min_epochs": 20, "patience": 15,
    "effective_batch_size": 32, "microbatch_size": 8,
    "learning_rate": 1e-4, "weight_decay": 0.01,
    "adam_betas": [0.9, 0.95], "gradient_clip_norm": 5.0,
    "scheduler": "cosine", "seeds": [17, 42, 73],
    "objectives": ["eeg_mae", "fnirs_mae", "symmetric_clip"],
    "clip_logit_multiplier": 0.1,
    "stage_a": "non_target_training_with_non_target_validation_epoch_selection",
    "stage_b": "fresh_initialization_refit_all_non_target_for_selected_epochs",
}


def _model_config(profile: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(DEFAULT_MODEL_CONFIG)
    unknown = set(profile.get("efrm_model_config", {})) - set(config)
    if unknown:
        raise ValueError(f"unknown EFRM model settings: {sorted(unknown)}")
    config.update(profile.get("efrm_model_config", {}))
    # Patch duration is part of the named EFRM model, including small fixtures.
    if config["eeg_patch_samples"] != 50 or config["fnirs_patch_samples"] != 20:
        raise ValueError("EFRM requires 0.25 s EEG and 2 s fNIRS patches")
    return config


def _pair_indices(profile: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    names = [str(name) for name in profile["fnirs_channel_names"]]
    if len(names) != len(set(names)):
        raise ValueError("fNIRS channel identities must be unique")
    components: dict[str, dict[str, int]] = {"hbo": {}, "hbr": {}}
    for index, name in enumerate(names):
        match = re.fullmatch(r"(.+)_(hbo|hbr)", name, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"fNIRS channel needs explicit _HbO/_HbR role: {name}")
        components[match[2].lower()][match[1]] = index
    locations = list(profile.get("fnirs_location_names", components["hbo"]))
    if not locations or len(locations) != len(set(locations)):
        raise ValueError("fNIRS location names must be nonempty and unique")
    if set(locations) != set(components["hbo"]) or set(locations) != set(components["hbr"]):
        raise ValueError("every retained fNIRS location requires exactly one HbO and HbR")
    return ([components["hbo"][name] for name in locations],
            [components["hbr"][name] for name in locations])


def _validate_profile(profile: Mapping[str, Any]) -> None:
    if float(profile["sampling_rate_hz"]) != 200 or float(profile["fnirs_sampling_rate_hz"]) != 10:
        raise ValueError("EFRM rebenchmark requires EEG@200 Hz and fNIRS@10 Hz")
    eeg_names = list(profile["channel_names"])
    if not eeg_names or len(eeg_names) != len(set(eeg_names)):
        raise ValueError("EEG channel identities must be nonempty and unique")
    _pair_indices(profile)


def _sample_mask(mask: Any, values: torch.Tensor, name: str) -> torch.Tensor:
    """Resolve [B,T] or [B,C,T] recorded support without assuming validity."""
    if mask is None:
        return torch.isfinite(values)
    valid = torch.as_tensor(mask, device=values.device, dtype=torch.bool)
    if valid.ndim == 2 and valid.shape == (values.shape[0], values.shape[-1]):
        valid = valid[:, None, :].expand_as(values)
    if valid.shape != values.shape:
        raise ValueError(f"{name} must have shape [B,T] or {tuple(values.shape)}")
    return valid & torch.isfinite(values)


def adapt_batch(
    eeg: torch.Tensor,
    fnirs: torch.Tensor,
    profile: Mapping[str, Any],
    *,
    eeg_mask: Any = None,
    fnirs_mask: Any = None,
    eeg_bad_channel_mask: Any = None,
    fnirs_bad_channel_mask: Any = None,
) -> dict[str, torch.Tensor]:
    """Map the common measured panel to EFRM's explicit component axes.

    A patch is valid only if all of its native samples are valid.  Invalid
    input values are zeroed before projection and cannot enter attention or
    reconstruction loss. No per-window normalization is performed.
    """
    _validate_profile(profile)
    if eeg.ndim != 3 or fnirs.ndim != 3 or eeg.shape[0] != fnirs.shape[0]:
        raise ValueError("expected paired EEG/fNIRS tensors [B,C,T]")
    if eeg.shape[1] != len(profile["channel_names"]) or fnirs.shape[1] != len(profile["fnirs_channel_names"]):
        raise ValueError("tensor channels disagree with the frozen panel")
    duration = float(profile["duration_s"])
    if eeg.shape[-1] != round(200 * duration) or fnirs.shape[-1] != round(10 * duration):
        raise ValueError("input lengths disagree with the frozen duration")
    eeg_valid = _sample_mask(eeg_mask, eeg, "eeg_mask")
    fnirs_valid = _sample_mask(fnirs_mask, fnirs, "fnirs_mask")
    for bad, valid, name in ((eeg_bad_channel_mask, eeg_valid, "eeg"),
                             (fnirs_bad_channel_mask, fnirs_valid, "fnirs")):
        if bad is not None:
            bad_tensor = torch.as_tensor(bad, device=valid.device, dtype=torch.bool)
            if bad_tensor.ndim == 1:
                bad_tensor = bad_tensor[None].expand(valid.shape[:2])
            if bad_tensor.shape != valid.shape[:2]:
                raise ValueError(f"{name} bad-channel mask disagrees with channels")
            valid &= ~bad_tensor[..., None]
    hbo, hbr = _pair_indices(profile)
    paired = torch.stack((fnirs[:, hbo], fnirs[:, hbr]), dim=1)
    paired_valid = torch.stack((fnirs_valid[:, hbo], fnirs_valid[:, hbr]), dim=1).all(1)
    paired = paired.masked_fill(~paired_valid[:, None], 0)
    eeg = eeg.masked_fill(~eeg_valid, 0)
    # Fixed window lengths are integral multiples of both physical patch sizes.
    if eeg.shape[-1] % 50 or fnirs.shape[-1] % 20:
        raise ValueError("duration must preserve complete physical EFRM patch grids")
    return {
        "eeg": eeg[:, None], "fnirs": paired,
        "eeg_patch_valid": eeg_valid.reshape(*eeg_valid.shape[:2], -1, 50).all(-1),
        "fnirs_patch_valid": paired_valid.reshape(*paired_valid.shape[:2], -1, 20).all(-1),
    }


def _temporal_tokens(mae: nn.Module, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Unmasked source encoder with channel pooling, retaining physical time."""
    embedded, height, width = mae.patch_embed(values)
    flattened = mae._flatten_valid(valid, height=height, width=width)
    positions = sincos_2d(mae.embed_dim, height, width).to(embedded)
    embedded = (embedded + positions[:, 1:]) * flattened.unsqueeze(-1)
    cls = (mae.cls_token + positions[:, :1]).expand(values.shape[0], -1, -1)
    encoded = mae.norm(mae._run_blocks(
        torch.cat((cls, embedded), dim=1), mae.blocks,
        mae._attention_mask(flattened, dtype=embedded.dtype),
    ))[:, 1:].reshape(values.shape[0], height, width, mae.embed_dim)
    return (encoded * valid[..., None]).sum(1) / valid.sum(1)[..., None].clamp_min(1)


class EFRMRebenchFeatures(nn.Module):
    """EFRM pooled classification or native time-token sequence features."""

    def __init__(self, task: str, profile: Mapping[str, Any], *, trainable: bool = False):
        super().__init__()
        if task not in TASK_DATASETS:
            raise ValueError(f"unknown rebenchmark task: {task}")
        _validate_profile(profile)
        self.task = task
        self.profile = deepcopy(dict(profile))
        self.model_config = _model_config(profile)
        self.backbone = EFRMSyncModel(**self.model_config)
        self.feature_dim = self.backbone.embed_dim
        self.sequence_features = task == "refed_regression"
        self.method_identity = METHOD_IDENTITY
        self.pretraining_provenance: dict[str, Any] | None = None
        self.pretraining_steps = 0
        self.configure_transfer(trainable)

    def configure_transfer(self, trainable: bool) -> None:
        self.backbone.requires_grad_(False)
        if trainable:
            for mae in (self.backbone.eeg_model, self.backbone.fnirs_model):
                mae.patch_embed.requires_grad_(True)
                mae.cls_token.requires_grad_(True)
                mae.blocks.requires_grad_(True)
                mae.norm.requires_grad_(True)

    def _adapt(self, eeg: torch.Tensor, fnirs: torch.Tensor, kwargs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        allowed = {"eeg_mask", "fnirs_mask", "eeg_bad_channel_mask", "fnirs_bad_channel_mask"}
        # Do not accept target masks as signal validity, or silently drop masks.
        unexpected_masks = {key for key in kwargs if "mask" in key and key not in allowed}
        if unexpected_masks:
            raise ValueError(f"unrecognized signal masks: {sorted(unexpected_masks)}")
        return adapt_batch(eeg, fnirs, self.profile, **{key: kwargs[key] for key in allowed if key in kwargs})

    def forward_sequence(self, eeg: torch.Tensor, fnirs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        batch = self._adapt(eeg, fnirs, kwargs)
        eeg_tokens = _temporal_tokens(self.backbone.eeg_model, batch["eeg"], batch["eeg_patch_valid"])
        fnirs_tokens = _temporal_tokens(self.backbone.fnirs_model, batch["fnirs"], batch["fnirs_patch_valid"])
        seconds = int(round(float(self.profile["duration_s"])))
        # EEG: four 0.25-second patches per output second. Hb: one 2-second
        # patch contributes to its two native-support seconds; no global
        # embedding is copied across the trajectory.
        eeg_valid = batch["eeg_patch_valid"].any(1).reshape(eeg.shape[0], seconds, 4)
        eeg_tokens = eeg_tokens.reshape(eeg.shape[0], seconds, 4, self.feature_dim)
        eeg_seconds = (eeg_tokens * eeg_valid[..., None]).sum(2) / eeg_valid.sum(2)[..., None].clamp_min(1)
        fnirs_seconds = fnirs_tokens.repeat_interleave(2, dim=1)
        return eeg_seconds + fnirs_seconds

    def sequence_feature_mask(self, eeg: torch.Tensor, fnirs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        batch = self._adapt(eeg, fnirs, kwargs)
        seconds = int(round(float(self.profile["duration_s"])))
        eeg_valid = batch["eeg_patch_valid"].any(1).reshape(eeg.shape[0], seconds, 4).any(-1)
        fnirs_valid = batch["fnirs_patch_valid"].any(1).repeat_interleave(2, dim=1)
        return eeg_valid | fnirs_valid

    def forward(self, eeg: torch.Tensor, fnirs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if self.task == "refed_regression":
            return self.forward_sequence(eeg, fnirs, **kwargs)
        batch = self._adapt(eeg, fnirs, kwargs)
        eeg_features, fnirs_features = self.backbone.encode(**batch)
        return eeg_features + fnirs_features


def create_pretraining_model(profile: dict[str, Any], device: str = "cpu", *, task: str = "motor_imagery") -> EFRMRebenchFeatures:
    """Fresh source-faithful EFRM; explicitly not a pretrained downstream model."""
    model = EFRMRebenchFeatures(task, profile)
    model.backbone.requires_grad_(True)
    model.method_identity = METHOD_IDENTITY + "__pretraining_in_progress"
    return model.to(device)


def pretraining_loss(
    model: EFRMRebenchFeatures,
    eeg: torch.Tensor,
    fnirs: torch.Tensor,
    *,
    profile: Mapping[str, Any] | None = None,
    **masks: Any,
) -> torch.Tensor:
    """All three source objectives on this exact admitted paired batch.

    The common worker owns optimizer/RNG/loop state. For larger effective
    batches use ``pretrain_step`` to retain the same CLIP negative matrix with
    bounded activation memory instead of ordinary gradient accumulation.
    """
    if eeg.shape[0] < 2:
        raise ValueError("EFRM symmetric contrastive loss needs at least two pairs")
    batch = adapt_batch(eeg, fnirs, model.profile if profile is None else profile, **masks)
    return model.backbone(**batch)["loss"]


def pretrain_step(
    model: EFRMRebenchFeatures,
    eeg: torch.Tensor,
    fnirs: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    profile: Mapping[str, Any] | None = None,
    microbatch_size: int = 8,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    gradient_clip_norm: float = 5.0,
    **masks: Any,
) -> dict[str, float]:
    """One exact full-negative-set update, usable by the durable common worker."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch = adapt_batch(eeg, fnirs, model.profile if profile is None else profile, **masks)
    metrics = cached_pretrain_backward(model.backbone, batch, chunk_size=microbatch_size, amp_dtype=amp_dtype)
    if not math.isfinite(metrics["loss"]):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("nonfinite EFRM pretraining objective")
    grad_norm = nn.utils.clip_grad_norm_(model.backbone.parameters(), gradient_clip_norm, error_if_nonfinite=True)
    optimizer.step()
    model.pretraining_steps += 1
    return {**metrics, "gradient_norm": float(grad_norm), "optimizer_steps": float(model.pretraining_steps)}


def _validate_provenance(provenance: Mapping[str, Any], *, task: str | None = None, terminal: bool = True) -> None:
    if dict(provenance.get("input_identity", {})) != INPUT_IDENTITY:
        raise ValueError("EFRM checkpoint must retain the exact measurement-v5 input identity")
    excluded = str(provenance.get("excluded_target_dataset", ""))
    expected = TASK_DATASETS.get(task) if task else excluded
    if excluded not in set(TASK_DATASETS.values()) or excluded != expected:
        raise ValueError("EFRM checkpoint must exclude the entire requested target dataset")
    source_ids = set(provenance.get("source_dataset_ids", []))
    if source_ids != set(TASK_DATASETS.values()) - {excluded}:
        raise ValueError("EFRM LODO source membership must be all three non-target datasets")
    if not provenance.get("source_membership_identity"):
        raise ValueError("EFRM checkpoint needs an auditable source membership identity")
    if provenance.get("seed") not in PRETRAINING_BUDGET["seeds"]:
        raise ValueError("EFRM rebenchmark pretraining seed must be 17, 42 or 73")
    if provenance.get("initialization") != "fresh_random" or provenance.get("stage") not in {"stage_a", "stage_b"}:
        raise ValueError("old-domain checkpoint continuation is not measurement-v5 retraining")
    if terminal and (provenance.get("status") != "completed_refit"
                     or provenance.get("stage") != "stage_b"
                     or int(provenance.get("epochs_completed", 0)) < 1):
        raise ValueError("downstream EFRM requires a completed fresh non-target refit")


def save_pretraining_checkpoint(
    model: EFRMRebenchFeatures,
    path: str | Path,
    provenance: Mapping[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
) -> None:
    """Atomically retain a selection/refit state with its exact source boundary."""
    _validate_provenance(provenance, terminal=False)
    if model.pretraining_steps < 1:
        raise ValueError("cannot label an untrained EFRM encoder as pretrained")
    payload = {
        "schema": CHECKPOINT_SCHEMA, "method_identity": METHOD_IDENTITY,
        "model_config": model.model_config, "model": model.backbone.state_dict(),
        "pretraining_provenance": dict(provenance), "optimizer_steps": model.pretraining_steps,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_pretraining_checkpoint(
    model: EFRMRebenchFeatures,
    path: str | Path,
    *,
    expected_provenance: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
) -> dict[str, Any]:
    """Resume only the same stage, membership, data identity and model shape.

    NumPy/Python sampler RNG and iteration position remain worker-owned; the
    payload may be retained beside its checkpoint without new controllers.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("method_identity") != METHOD_IDENTITY:
        raise ValueError("cannot resume a historical EFRM checkpoint as measurement-v5")
    observed = payload["pretraining_provenance"]
    _validate_provenance(observed, terminal=False)
    _validate_provenance(expected_provenance, terminal=False)
    for key in ("input_identity", "excluded_target_dataset", "source_dataset_ids",
                "source_membership_identity", "stage", "initialization", "seed"):
        if observed.get(key) != expected_provenance.get(key):
            raise ValueError(f"EFRM resume boundary drift: {key}")
    if payload["model_config"] != model.model_config:
        raise ValueError("EFRM resume architecture drift")
    model.backbone.load_state_dict(payload["model"], strict=True)
    model.pretraining_steps = int(payload["optimizer_steps"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload["cuda_rng_state_all"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    return dict(observed)


def create_model(task: str, profile: dict[str, Any], device: str, trainable: bool = False) -> nn.Module:
    """Load a newly retrained target-excluded checkpoint, never old v1 weights."""
    checkpoint_path = profile.get("pretrained_checkpoint", profile.get("checkpoint_path"))
    if checkpoint_path is None:
        if not profile.get("synthetic_random_encoder", False):
            raise ValueError("EFRM needs a completed measurement-v5 LODO checkpoint; use create_pretraining_model for retraining")
        model = EFRMRebenchFeatures(task, profile, trainable=trainable)
        model.method_identity = METHOD_IDENTITY + "__synthetic_random_encoder_not_pretrained"
    else:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("method_identity") != METHOD_IDENTITY:
            raise ValueError("historical/foreign EFRM checkpoints cannot be relabeled measurement-v5 retraining")
        _validate_provenance(payload["pretraining_provenance"], task=task)
        if int(payload.get("optimizer_steps", 0)) < 1:
            raise ValueError("EFRM checkpoint has no pretraining updates")
        requested_config = _model_config(profile)
        if any(payload["model_config"].get(key) != value for key, value in requested_config.items()
               if key != "activation_checkpointing"):
            raise ValueError("EFRM checkpoint architecture disagrees with the requested profile")
        model_profile = {**profile, "efrm_model_config": payload["model_config"]}
        model = EFRMRebenchFeatures(task, model_profile, trainable=trainable)
        model.backbone.load_state_dict(payload["model"], strict=True)
        model.pretraining_provenance = dict(payload["pretraining_provenance"])
        model.pretraining_steps = int(payload["optimizer_steps"])
    return model.to(device).eval()
