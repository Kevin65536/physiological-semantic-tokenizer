"""Same-window STA-Net adapter for DOWNSTREAM-REBENCH-v1.

Independent supervised PyTorch reimplementation, not paper-number reproduction.
Inputs are real-channel EEG [B,C,T] and paired Hb [B,2F,T]. Geometry projection
is a fixed interpolation of those channels, not additional measured channels.
Only fNIRS segments contained in the common encoding window are constructed.
Classification forward returns log probabilities [B,K], valid as CE logits;
REFED returns actual sequence predictions [B,2,20]. ``return_details=True``
exposes the retained model's branches and source-shaped composite objective.
No loader, training loop, historical checkpoint, or measured-data access lives here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from comparative_methods.rebench_eeg_common_v1 import encoding_duration

METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from sta_net_pytorch.data import STANetSpatialProjector, _generic_grid_coordinates
from sta_net_pytorch.model import STANet, STANetConfig, STANetObjective


OUTPUT_KIND = "predictions"
# Historical optimizer anchor, with a finite neighbourhood fixed before scores.
SEARCH_SOURCE = "configs/train_all_tasks.yaml training lr/weight_decay/grad_clip_norm"
DEFAULT_SEARCH = (
    {"lr": 1e-4, "weight_decay": 1e-4, "grad_clip_norm": 1.0},
    {"lr": 3e-5, "weight_decay": 1e-4, "grad_clip_norm": 1.0},
    {"lr": 3e-4, "weight_decay": 1e-4, "grad_clip_norm": 1.0},
    {"lr": 1e-4, "weight_decay": 1e-3, "grad_clip_norm": 1.0},
)


TASKS = {
    "motor_imagery": ("classification", 2, 8),
    "mental_arithmetic": ("classification", 2, 8),
    "wg": ("classification", 2, 8),
    "nback": ("classification", 3, 8),
    "dsr": ("classification", 2, 2),
    "visual": ("classification", 4, 8),
    "refed_regression": ("regression", 2, 20),
}


def _coordinates(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"STA-Net requires real {name} geometry in profile")
    if len(value) and isinstance(value[0], Mapping):
        rows = value
    else:
        array = np.asarray(value, dtype=float)
        if array.ndim != 2 or array.shape[1] < 2:
            raise ValueError(f"{name} geometry must be finite [C,2] or [C,3]")
        rows = [{"x": row[0], "y": row[1]} for row in array]
    array = np.asarray([[row.get("x"), row.get("y")] for row in rows], dtype=float)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} geometry must contain finite real-channel coordinates")
    if len(np.unique(array, axis=0)) < 3 or np.linalg.matrix_rank(array - array.mean(0)) < 2:
        raise ValueError(f"{name} geometry needs three non-collinear locations")
    return _generic_grid_coordinates(rows)


def _projection(coordinates: np.ndarray) -> Tensor:
    # Reuse the retained geometry interpolation; columns remain real sensors.
    values = np.eye(len(coordinates), dtype=np.float32)
    return torch.from_numpy(STANetSpatialProjector().project(values, coordinates).reshape(256, -1))


def _paired_fnirs_geometry(profile: Mapping[str, Any]) -> Any:
    geometry = profile.get("fnirs_geometry")
    if geometry is None:
        return None
    locations = profile.get("fnirs_location_names")
    names = profile.get("fnirs_channel_names")
    count = len(locations) if locations is not None else (len(names) // 2 if names is not None else len(geometry))
    order = profile.get("fnirs_component_order", "hbo_then_hbr")
    if order == "hbo_then_hbr":
        hbo_indices, hbr_indices = range(count), range(count, 2 * count)
    elif order == "interleaved":
        hbo_indices, hbr_indices = range(0, 2 * count, 2), range(1, 2 * count, 2)
    else:
        raise ValueError("fnirs_component_order must be hbo_then_hbr or interleaved")
    if names is not None:
        if len(names) != 2 * count:
            raise ValueError("fNIRS channel names must contain one HbO/HbR pair per location")
        for index, (hbo_index, hbr_index) in enumerate(zip(hbo_indices, hbr_indices, strict=True)):
            hbo_name, hbr_name = str(names[hbo_index]), str(names[hbr_index])
            if not hbo_name.lower().endswith("hbo") or not hbr_name.lower().endswith("hbr"):
                raise ValueError("fNIRS channel names disagree with HbO/HbR component order")
            hbo_location = re.sub(r"[_\-\s]*hbo$", "", hbo_name, flags=re.IGNORECASE)
            hbr_location = re.sub(r"[_\-\s]*hbr$", "", hbr_name, flags=re.IGNORECASE)
            if hbo_location != hbr_location or (locations is not None and hbo_location != str(locations[index])):
                raise ValueError("fNIRS channel names do not describe matched real locations")
    if len(geometry) == count:
        return geometry
    if len(geometry) != 2 * count:
        raise ValueError("fNIRS geometry must have F locations or 2F paired channel rows")
    if isinstance(geometry[0], Mapping):
        coordinates = np.asarray([[row.get(axis, 0) for axis in ("x", "y", "z")] for row in geometry], dtype=float)
    else:
        coordinates = np.asarray(geometry, dtype=float)
    hbo_indices, hbr_indices = list(hbo_indices), list(hbr_indices)
    if not np.allclose(coordinates[hbo_indices], coordinates[hbr_indices], rtol=0, atol=1e-8):
        raise ValueError("HbO and HbR geometry must match at every real fNIRS location")
    return [geometry[index] for index in hbo_indices]


def _masked_input(value: Tensor, mask: Tensor | None, name: str) -> Tensor:
    if mask is None:
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite recorded input")
        return value
    mask = torch.as_tensor(mask, device=value.device, dtype=torch.bool)
    if mask.ndim == 2:
        if mask.shape == value.shape[:2]:
            mask = mask[:, :, None]
        elif mask.shape == (value.shape[0], value.shape[-1]):
            mask = mask[:, None, :]
    try:
        mask = torch.broadcast_to(mask, value.shape)
    except RuntimeError as error:
        raise ValueError(f"{name} mask does not match canonical input") from error
    if not bool(torch.isfinite(value[mask]).all()):
        raise ValueError(f"{name} contains non-finite recorded input")
    return torch.where(mask, value, torch.zeros_like(value))


class SameWindowSTANet(nn.Module):
    def __init__(self, task: str, profile: Mapping[str, Any]):
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"unsupported STA-Net task: {task}")
        self.task = task
        self.profile = dict(profile)
        task_type, output_dim, _ = TASKS[task]
        duration = encoding_duration(task, profile)
        if task_type == "classification" and int(profile.get("num_classes", output_dim)) != output_dim:
            raise ValueError("STA-Net class count differs from the fixed task vocabulary")
        if float(profile.get("duration_s", duration)) != duration:
            raise ValueError("STA-Net encoding duration differs from the rebenchmark task")
        self.eeg_rate = float(profile.get("sampling_rate_hz", profile.get("eeg_sample_rate_hz", 200)))
        self.fnirs_rate = float(profile.get("fnirs_sampling_rate_hz", profile.get("fnirs_sample_rate_hz", 10)))
        if self.eeg_rate <= 0 or self.fnirs_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.eeg_samples = round(duration * self.eeg_rate)
        self.fnirs_samples = round(duration * self.fnirs_rate)
        eeg_coordinates = _coordinates(profile.get("eeg_geometry"), "EEG")
        hb_coordinates = _coordinates(_paired_fnirs_geometry(profile), "fNIRS location")
        if "channel_names" in profile and len(profile["channel_names"]) != len(eeg_coordinates):
            raise ValueError("EEG geometry and real-channel inventory differ in length")
        if "fnirs_location_names" in profile and len(profile["fnirs_location_names"]) != len(hb_coordinates):
            raise ValueError("fNIRS geometry and real-location inventory differ in length")
        self.register_buffer("eeg_projection", _projection(eeg_coordinates))
        self.register_buffer("fnirs_projection", _projection(hb_coordinates))
        order = profile.get("fnirs_component_order", "hbo_then_hbr")
        count = len(hb_coordinates)
        if order == "hbo_then_hbr":
            hbo_indices, hbr_indices = list(range(count)), list(range(count, 2 * count))
        elif order == "interleaved":
            hbo_indices, hbr_indices = list(range(0, 2 * count, 2)), list(range(1, 2 * count, 2))
        else:
            raise ValueError("fnirs_component_order must be hbo_then_hbr or interleaved")
        self.register_buffer("hbo_indices", torch.tensor(hbo_indices))
        self.register_buffer("hbr_indices", torch.tensor(hbr_indices))
        segment_s = float(profile.get("fnirs_segment_s", min(3, duration)))
        step_s = float(profile.get("fnirs_lag_step_s", 1))
        self.segment_samples = round(segment_s * self.fnirs_rate)
        step_samples = round(step_s * self.fnirs_rate)
        if not 1 <= self.segment_samples <= self.fnirs_samples or step_samples < 1:
            raise ValueError("fNIRS segments must lie inside the common window")
        self.segment_starts = tuple(range(0, self.fnirs_samples - self.segment_samples + 1, step_samples))
        config = STANetConfig(
            task_type=task_type,
            output_dim=output_dim,
            sequence_length=20 if task_type == "regression" else 1,
            dropout=float(profile.get("dropout", 0.5)),
            embedding_dim=int(profile.get("embedding_dim", 256)),
            attention_heads=int(profile.get("attention_heads", 10)),
            attention_key_dim=int(profile.get("attention_key_dim", 256)),
            max_lags=len(self.segment_starts),
        )
        self.backbone = STANet(config)
        self.objective = STANetObjective(
            task_type,
            eeg_aux_weight=float(profile.get("eeg_aux_weight", 1)),
            alignment_weight=float(profile.get("alignment_weight", 1)),
            regression_loss=str(profile.get("regression_loss", "smooth_l1")),
        )
        self.output_dim = output_dim
        self.output_kind = "log_probabilities" if task_type == "classification" else "sequence_regression"
        self.spec = {
            "schema": "sta_net_downstream_rebench_v1",
            "method_identity": "independent_pytorch_sta_net_same_window_supervised",
            "task": task,
            "duration_s": duration,
            "eeg_channel_count": len(eeg_coordinates),
            "fnirs_location_count": count,
            "grid_is_interpolated_measurement": True,
            "grid_size": [16, 16],
            "fnirs_component_order": order,
            "fnirs_segment_samples": self.segment_samples,
            "fnirs_segment_starts": list(self.segment_starts),
            "extra_temporal_context_s": 0,
            "output_kind": self.output_kind,
            "output_shape": ["B", output_dim, 20] if task_type == "regression" else ["B", output_dim],
            "source_deviations": [
                "canonical_real_channel_geometry_interpolated_to_fixed_grid",
                "task_encoding_window_replaces_historical_3s_eeg_13s_hb_context",
                "only_within_window_hb_lags",
                "project_multiclass_and_refed_sequence_heads",
            ],
        }

    def forward(
        self,
        eeg: Tensor,
        fnirs: Tensor,
        *,
        eeg_mask: Tensor | None = None,
        fnirs_mask: Tensor | None = None,
        return_details: bool = False,
    ) -> Tensor | dict[str, Any]:
        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != (self.eeg_projection.shape[1], self.eeg_samples):
            raise ValueError("EEG must match declared real channels and common encoding window")
        if fnirs.ndim != 3 or tuple(fnirs.shape[1:]) != (2 * self.fnirs_projection.shape[1], self.fnirs_samples):
            raise ValueError("fNIRS must match paired real locations and common encoding window")
        if eeg.shape[0] != fnirs.shape[0]:
            raise ValueError("EEG and fNIRS batch sizes differ")
        eeg = _masked_input(eeg, eeg_mask, "EEG")
        fnirs = _masked_input(fnirs, fnirs_mask, "fNIRS")
        eeg_grid = torch.einsum("gc,bct->bgt", self.eeg_projection, eeg).reshape(eeg.shape[0], 1, 16, 16, -1)
        paired = torch.stack((fnirs.index_select(1, self.hbo_indices), fnirs.index_select(1, self.hbr_indices)), dim=1)
        hb_grid = torch.einsum("gc,bkct->bkgt", self.fnirs_projection, paired).reshape(eeg.shape[0], 2, 16, 16, -1)
        segments = torch.stack([hb_grid[..., start : start + self.segment_samples] for start in self.segment_starts], dim=1)
        outputs = self.backbone(eeg_grid, segments)
        if return_details:
            return outputs
        prediction = outputs["prediction"]
        if self.output_kind == "log_probabilities":
            return prediction.clamp_min(1e-8).log()
        return prediction

    def supervised_loss(
        self, eeg: Tensor, fnirs: Tensor, target: Tensor,
        target_valid_mask: Tensor | None = None, **kwargs: Any,
    ) -> dict[str, Tensor]:
        if self.backbone.config.task_type == "regression" and target_valid_mask is not None:
            target = torch.where(target_valid_mask.to(device=target.device, dtype=torch.bool), target, torch.zeros_like(target))
        return self.objective(self(eeg, fnirs, return_details=True, **kwargs), target, target_valid_mask)


def create_model(
    task: str, profile: Mapping[str, Any], device: torch.device | str,
    trainable: bool = True,
) -> SameWindowSTANet:
    model = SameWindowSTANet(task, profile).to(device)
    model.requires_grad_(trainable)
    if not trainable:
        model.eval()
    return model
