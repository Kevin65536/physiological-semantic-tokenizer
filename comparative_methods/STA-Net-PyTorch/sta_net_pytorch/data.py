"""Unified-loader adapters for the PyTorch STA-Net comparison variants."""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.interpolate import griddata
from torch.utils.data import Dataset

from src.data.clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT
from src.data.unified_physiology import (
    REFEDContinuousSequenceDataset,
    UnifiedPhysiologyWindowDataset,
    canonical_label,
    refed_continuous_target_window,
)


OFFICIAL_EEG_GRID = {
    name: coordinate
    for name, coordinate in zip(
        (
            "Fp1", "AFF5h", "AFz", "F1", "FC5", "FC1", "T7", "C3", "Cz", "CP5", "CP1",
            "P7", "P3", "Pz", "POz", "O1", "Fp2", "AFF6h", "F2", "FC2", "FC6", "C4",
            "T8", "CP2", "CP6", "P4", "P8", "O2",
        ),
        (
            (0, 6), (2, 5), (2, 8), (3, 7), (5, 2), (5, 6), (7, 1), (7, 4), (7, 8),
            (9, 2), (9, 6), (11, 2), (11, 5), (11, 8), (13, 8), (14, 6), (0, 10),
            (2, 11), (3, 9), (5, 10), (5, 14), (7, 12), (7, 15), (9, 10), (9, 14),
            (11, 11), (11, 14), (14, 10),
        ),
        strict=True,
    )
}

OFFICIAL_FNIRS_GRID = {
    name: coordinate
    for name, coordinate in zip(
        (
            "AF7", "AFF5", "AFp7", "AF5h", "AFp3", "AFF3h", "AF1", "AFFz", "AFpz", "AF2",
            "AFp4", "FCC3", "C3h", "C5h", "CCP3", "CPP3", "P3h", "P5h", "PPO3", "AFF4h",
            "AF6h", "AFF6", "AFp8", "AF8", "FCC4", "C6h", "C4h", "CCP4", "CPP4", "P6h",
            "P4h", "PPO4", "PPOz", "PO1", "PO2", "POOz",
        ),
        (
            (2, 4), (3, 4), (1, 5), (2, 5), (1, 7), (3, 6), (2, 7), (3, 8), (1, 8),
            (2, 9), (1, 9), (6, 4), (7, 5), (7, 3), (8, 4), (10, 5), (11, 6), (11, 4),
            (12, 5), (3, 10), (2, 11), (3, 12), (1, 11), (2, 12), (6, 12), (7, 13),
            (7, 11), (8, 12), (10, 11), (11, 12), (11, 10), (12, 11), (12, 8), (13, 7),
            (13, 9), (14, 8),
        ),
        strict=True,
    )
}


@dataclass(frozen=True)
class STANetTaskSpec:
    key: str
    dataset_id: str
    namespace: str
    task_type: str
    class_names: tuple[str, ...] = ()
    target_names: tuple[str, ...] = ()
    target_length: int = 1
    eeg_duration_s: float = 3.0
    fnirs_segment_s: float = 3.0
    fnirs_lag_count: int = 11
    fnirs_lag_step_s: float = 1.0
    scientific_scope: str = "project_adapted_comparison_variant"

    @property
    def output_dim(self) -> int:
        return len(self.class_names) if self.task_type == "classification" else len(self.target_names)


TASK_SPECS: dict[str, STANetTaskSpec] = {
    "motor_imagery": STANetTaskSpec(
        key="motor_imagery", dataset_id="eeg_fnirs_single_trial",
        namespace="eeg_fnirs_single_trial:motor_imagery", task_type="classification",
        class_names=("LMI", "RMI"), scientific_scope="source_task_pytorch_reimplementation",
    ),
    "mental_arithmetic": STANetTaskSpec(
        key="mental_arithmetic", dataset_id="eeg_fnirs_single_trial",
        namespace="eeg_fnirs_single_trial:mental_arithmetic", task_type="classification",
        class_names=("MA", "BL"), scientific_scope="source_task_pytorch_reimplementation",
    ),
    "wg": STANetTaskSpec(
        key="wg", dataset_id="simultaneous_eeg_nirs", namespace="simultaneous_eeg_nirs:wg",
        task_type="classification", class_names=("WG", "BL"),
        scientific_scope="source_task_pytorch_reimplementation",
    ),
    "nback": STANetTaskSpec(
        key="nback", dataset_id="simultaneous_eeg_nirs", namespace="simultaneous_eeg_nirs:nback",
        task_type="classification", class_names=("0-back session", "2-back session", "3-back session"),
    ),
    "dsr": STANetTaskSpec(
        key="dsr", dataset_id="simultaneous_eeg_nirs", namespace="simultaneous_eeg_nirs:dsr",
        task_type="classification", class_names=("Go", "No-go"), eeg_duration_s=2.0,
        scientific_scope="eeg_primary_fnirs_context_sensitivity_variant",
    ),
    "visual": STANetTaskSpec(
        key="visual", dataset_id="visual_cognitive_motivation",
        namespace="visual_cognitive_motivation:visual_cognitive_motivation",
        task_type="classification", class_names=("RR", "RF", "FF", "FR"),
    ),
    "refed_regression": STANetTaskSpec(
        key="refed_regression", dataset_id="refed", namespace="refed:emotion_video",
        task_type="regression", target_names=("valence", "arousal"), target_length=20,
        eeg_duration_s=20.0, fnirs_segment_s=3.0, fnirs_lag_count=18,
        scientific_scope="sta_net_regression_adapter",
    ),
}


def get_sta_net_task_spec(key: str) -> STANetTaskSpec:
    try:
        return TASK_SPECS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown STA-Net task {key!r}; expected one of {sorted(TASK_SPECS)}") from exc


def _strip_component(name: str) -> str:
    return re.sub(r"_(?:HbO|HbR)$", "", name, flags=re.IGNORECASE)


def _generic_grid_coordinates(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    coordinates = np.asarray([[row.get("x"), row.get("y")] for row in rows], dtype=np.float64)
    valid = np.isfinite(coordinates).all(axis=1)
    if valid.sum() < 3:
        raise ValueError("STA-Net projection requires at least three finite channel coordinates")
    center = np.nanmean(coordinates[valid], axis=0)
    span = float(np.nanmax(np.ptp(coordinates[valid], axis=0)))
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("Channel coordinates have zero spatial span")
    return (coordinates - center) * (13.0 / span) + 7.5


class STANetSpatialProjector:
    """Project channels to the 16x16 STA-Net grid using cached interpolation matrices."""

    def __init__(self, grid_size: int = 16):
        self.grid_size = int(grid_size)
        grid_x, grid_y = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
        self.query = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float64)
        self._matrix_cache: dict[str, np.ndarray] = {}

    def _matrix(self, coordinates: np.ndarray) -> np.ndarray:
        key = hashlib.sha256(np.round(coordinates, 8).tobytes()).hexdigest()
        if key not in self._matrix_cache:
            basis = np.eye(coordinates.shape[0], dtype=np.float64)
            method = "cubic" if coordinates.shape[0] >= 4 else "linear"
            try:
                weights = griddata(coordinates, basis, self.query, method=method)
            except Exception:
                weights = griddata(coordinates, basis, self.query, method="linear")
            nearest = griddata(coordinates, basis, self.query, method="nearest")
            weights = np.where(np.isfinite(weights), weights, nearest)
            self._matrix_cache[key] = np.asarray(weights, dtype=np.float32)
        return self._matrix_cache[key]

    @staticmethod
    def _deduplicate(coordinates: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rounded = np.round(coordinates, 8)
        unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
        if len(unique) == len(coordinates):
            return coordinates, values
        aggregated = np.zeros((len(unique), values.shape[1]), dtype=np.float32)
        counts = np.zeros(len(unique), dtype=np.float32)
        for index, group in enumerate(inverse):
            aggregated[group] += values[index]
            counts[group] += 1.0
        return unique.astype(np.float64), aggregated / counts[:, None]

    def project(self, values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        coordinates, values = self._deduplicate(coordinates, np.asarray(values, dtype=np.float32))
        projected = self._matrix(coordinates) @ values
        return projected.reshape(self.grid_size, self.grid_size, values.shape[1])


class STANetSampleAdapter:
    """Convert a unified physiology sample into PyTorch STA-Net tensors."""

    def __init__(self, spec: STANetTaskSpec):
        self.spec = spec
        self.projector = STANetSpatialProjector()
        self.class_to_index = {name: index for index, name in enumerate(spec.class_names)}
        self.target_center: np.ndarray | None = None
        self.target_scale: np.ndarray | None = None

    def set_target_scaler(self, center: Sequence[float], scale: Sequence[float]) -> None:
        center_array = np.asarray(center, dtype=np.float32).reshape(-1, 1)
        scale_array = np.asarray(scale, dtype=np.float32).reshape(-1, 1)
        if center_array.shape[0] != self.spec.output_dim or scale_array.shape != center_array.shape:
            raise ValueError("Target scaler dimensions do not match the regression head")
        if not np.isfinite(center_array).all() or not np.isfinite(scale_array).all() or (scale_array <= 0).any():
            raise ValueError("Target scaler must contain finite centers and positive finite scales")
        self.target_center = center_array
        self.target_scale = scale_array

    @staticmethod
    def _take_with_padding(values: np.ndarray, length: int) -> np.ndarray:
        output = np.zeros((values.shape[0], length), dtype=np.float32)
        usable = min(length, values.shape[1])
        output[:, :usable] = values[:, :usable]
        return output

    @staticmethod
    def _time_mask(mask: np.ndarray, length: int) -> np.ndarray:
        output = np.zeros(length, dtype=bool)
        usable = min(length, mask.size)
        output[:usable] = mask[:usable]
        return output

    def _eeg_coordinates(self, sample: Mapping[str, Any], keep: np.ndarray) -> tuple[np.ndarray, str]:
        names = list(sample["channel_names"]["eeg"])
        if all(name in OFFICIAL_EEG_GRID for name in names):
            coordinates = np.asarray([OFFICIAL_EEG_GRID[name] for name in names], dtype=np.float64)
            return coordinates[keep], "official_sta_net_wg_grid"
        coordinates = _generic_grid_coordinates(sample["channel_geometry"]["eeg"])
        return coordinates[keep], "unified_geometry_normalized_grid"

    def _fnirs_coordinates(
        self, sample: Mapping[str, Any], component_indices: np.ndarray, keep: np.ndarray
    ) -> tuple[np.ndarray, str]:
        names = [_strip_component(sample["channel_names"]["fnirs"][index]) for index in component_indices]
        if all(name in OFFICIAL_FNIRS_GRID for name in names):
            coordinates = np.asarray([OFFICIAL_FNIRS_GRID[name] for name in names], dtype=np.float64)
            return coordinates[keep], "official_sta_net_wg_grid"
        rows = [sample["channel_geometry"]["fnirs"][index] for index in component_indices]
        coordinates = _generic_grid_coordinates(rows)
        return coordinates[keep], "unified_geometry_normalized_grid"

    def adapt(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        eeg_rate = float(sample["sample_rate_hz"]["eeg"])
        fnirs_rate = float(sample["sample_rate_hz"]["fnirs"])
        eeg_length = int(round(self.spec.eeg_duration_s * eeg_rate))
        fnirs_segment = int(round(self.spec.fnirs_segment_s * fnirs_rate))
        fnirs_step = int(round(self.spec.fnirs_lag_step_s * fnirs_rate))
        fnirs_required = fnirs_segment + (self.spec.fnirs_lag_count - 1) * fnirs_step

        eeg_values = self._take_with_padding(np.asarray(sample["eeg"], dtype=np.float32), eeg_length)
        # `valid_mask` represents only real record support.  In particular, do not
        # consume the legacy `analysis_valid_mask`: historical caches could encode
        # dataset-specific artifact detections there, while the current unified
        # contract deliberately gives artifact masks no invalidity authority.
        eeg_time_valid = self._time_mask(np.asarray(sample["valid_mask"]["eeg"], dtype=bool), eeg_length)
        eeg_values[:, ~eeg_time_valid] = 0.0
        eeg_keep = ~np.asarray(sample["bad_channel_mask"]["eeg"], dtype=bool)
        eeg_coordinates, eeg_coordinate_mode = self._eeg_coordinates(sample, eeg_keep)
        eeg_grid = self.projector.project(eeg_values[eeg_keep], eeg_coordinates)

        fnirs_values = self._take_with_padding(np.asarray(sample["fnirs"], dtype=np.float32), fnirs_required)
        fnirs_time_valid = self._time_mask(
            np.asarray(sample["valid_mask"]["fnirs"], dtype=bool), fnirs_required
        )
        fnirs_values[:, ~fnirs_time_valid] = 0.0
        roles = np.asarray(sample["component_roles"]["fnirs"], dtype=object)
        bad_fnirs = np.asarray(sample["bad_channel_mask"]["fnirs"], dtype=bool)
        component_grids = []
        fnirs_coordinate_modes = []
        for component in ("HbO", "HbR"):
            indices = np.flatnonzero(roles == component)
            keep = ~bad_fnirs[indices]
            coordinates, mode = self._fnirs_coordinates(sample, indices, keep)
            component_grids.append(self.projector.project(fnirs_values[indices[keep]], coordinates))
            fnirs_coordinate_modes.append(mode)
        fnirs_series = np.stack(component_grids, axis=0)  # [2,H,W,T]
        lagged = np.stack(
            [fnirs_series[..., lag * fnirs_step : lag * fnirs_step + fnirs_segment] for lag in range(self.spec.fnirs_lag_count)],
            axis=0,
        )

        result: dict[str, Any] = {
            "eeg": torch.from_numpy(eeg_grid[None]),
            "fnirs": torch.from_numpy(lagged),
            "sample_id": str(sample.get("sample_id") or self._sample_id(sample)),
            "subject": str(sample["subject"]),
            "record_id": str(sample["record_id"]),
            "join_key": str(sample["join_key"]),
            "adapter_state": {
                "eeg_coordinate_mode": eeg_coordinate_mode,
                "fnirs_coordinate_modes": sorted(set(fnirs_coordinate_modes)),
                "eeg_record_support_fraction": float(eeg_time_valid.mean()),
                "fnirs_record_support_fraction": float(fnirs_time_valid.mean()),
                "artifact_mask_consumed": False,
                "validity_source": "valid_mask_only",
                "eeg_bad_channel_count": int((~eeg_keep).sum()),
                "fnirs_bad_channel_count": int(bad_fnirs.sum()),
            },
        }
        if self.spec.task_type == "classification":
            condition = str(sample["label"]["condition"])
            if condition not in self.class_to_index:
                raise ValueError(f"Unknown label {condition!r} for {self.spec.namespace}")
            result["target"] = torch.tensor(self.class_to_index[condition], dtype=torch.long)
            result["target_valid_mask"] = torch.tensor(True)
        else:
            target = np.asarray(sample["target"], dtype=np.float32).copy()
            target_valid = np.asarray(sample["target_valid_mask"], dtype=bool)
            if self.target_center is not None and self.target_scale is not None:
                target = (target - self.target_center) / self.target_scale
            target[~target_valid] = 0.0
            result["target"] = torch.as_tensor(target)
            result["target_valid_mask"] = torch.as_tensor(target_valid)
        return result

    @staticmethod
    def _sample_id(sample: Mapping[str, Any]) -> str:
        event = sample.get("event", {})
        event_index = event.get("event_index", event.get("onset_ms", "unknown"))
        offset = sample.get("alignment", {}).get("event_relative_window_start_s", 0.0)
        return f"{sample['join_key']}|event={event_index}|offset_s={float(offset):.6f}"

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "sta_net_unified_adapter_v2",
            "task": asdict(self.spec),
            "pytorch_eeg_shape": [1, 16, 16, "eeg_samples"],
            "pytorch_fnirs_shape": [self.spec.fnirs_lag_count, 2, 16, 16, "fnirs_segment_samples"],
            "mask_policy": (
                "artifact masks have no authority; valid_mask only marks real record support; "
                "omit independently defined bad channels before spatial interpolation"
            ),
            "artifact_mask_policy": "disabled_all_false_no_invalid_authority_v1",
            "spatial_projection": "official source grid when channel inventory matches; otherwise normalized unified geometry",
            "target_policy": "explicit class order or native REFED sequence plus per-coordinate validity mask",
            "target_scaler": None if self.target_center is None else {
                "center": self.target_center[:, 0].tolist(),
                "scale": self.target_scale[:, 0].tolist(),
                "fit_scope": "training subjects only",
            },
        }


class STANetUnifiedTaskDataset(Dataset):
    """Task-filtered unified-loader dataset with deterministic STA-Net adaptation."""

    def __init__(
        self,
        spec: STANetTaskSpec,
        cache_root: str = DEFAULT_CLEAN_CACHE_ROOT,
        *,
        adapted_cache_size: int = 0,
    ):
        self.spec = spec
        self.adapter = STANetSampleAdapter(spec)
        self.adapted_cache_size = max(0, int(adapted_cache_size))
        self._adapted_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        if spec.task_type == "regression":
            self.base = REFEDContinuousSequenceDataset(cache_root=cache_root)
            self.indices = list(range(len(self.base)))
            self.excluded_label_counts: dict[str, int] = {}
        else:
            self.base = UnifiedPhysiologyWindowDataset(cache_root=cache_root, dataset_ids=(spec.dataset_id,))
            allowed = set(spec.class_names)
            self.indices = []
            excluded: dict[str, int] = {}
            for index, window in enumerate(self.base.windows):
                label = canonical_label(window.event, window.record.dataset_id)
                if label["namespace"] != spec.namespace:
                    continue
                condition = str(label["condition"])
                if condition not in allowed:
                    excluded[condition] = excluded.get(condition, 0) + 1
                    continue
                self.indices.append(index)
            self.excluded_label_counts = excluded
        if not self.indices:
            raise RuntimeError(f"No admitted samples found for {spec.namespace}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.adapted_cache_size:
            cached = self._adapted_cache.get(int(index))
            if cached is not None:
                self._adapted_cache.move_to_end(int(index))
                return cached
        adapted = self.adapter.adapt(self.base[self.indices[index]])
        if self.adapted_cache_size:
            self._adapted_cache[int(index)] = adapted
            self._adapted_cache.move_to_end(int(index))
            while len(self._adapted_cache) > self.adapted_cache_size:
                self._adapted_cache.popitem(last=False)
        return adapted

    def fit_regression_target_scaler(self, dataset_indices: Sequence[int]) -> dict[str, Any]:
        """Fit per-coordinate standardization from training windows without loading signals."""

        if self.spec.task_type != "regression":
            raise RuntimeError("Target scaling is only defined for regression tasks")
        values_by_coordinate: list[list[np.ndarray]] = [[] for _ in range(self.spec.output_dim)]
        for dataset_index in dataset_indices:
            ref = self.base.windows[self.indices[int(dataset_index)]]
            target = refed_continuous_target_window(
                ref.event,
                window_start_s=ref.window_offset_s,
                window_duration_s=self.base.window_duration_s,
                target_sample_rate_hz=self.base.target_sample_rate_hz,
            )
            values = np.asarray(target["values"], dtype=np.float64)
            valid = np.asarray(target["valid_mask"], dtype=bool) & np.isfinite(values)
            for coordinate in range(self.spec.output_dim):
                values_by_coordinate[coordinate].append(values[coordinate, valid[coordinate]])
        flattened = [np.concatenate(parts) for parts in values_by_coordinate]
        center = np.asarray([np.mean(values) for values in flattened], dtype=np.float32)
        scale = np.asarray([np.std(values) for values in flattened], dtype=np.float32)
        scale = np.where(scale > 1e-6, scale, 1.0)
        self.adapter.set_target_scaler(center, scale)
        return {
            "center": center.tolist(),
            "scale": scale.tolist(),
            "valid_value_count": [int(values.size) for values in flattened],
            "fit_scope": "training subjects only",
        }

    def lightweight_metadata(self, index: int) -> dict[str, Any]:
        window = self.base.windows[self.indices[index]]
        label = canonical_label(window.event, window.record.dataset_id)
        return {
            "subject": str(window.record.canonical_subject_id),
            "record_id": str(window.record.base_record_id),
            "join_key": str(window.record.join_key),
            "condition": str(label["condition"]),
            "class_index": label.get("class_index"),
            "window_offset_s": float(window.window_offset_s),
            "event_index": int(window.event.get("event_index", -1)),
            "trial_group": f"{window.record.join_key}|event={int(window.event.get('event_index', -1))}",
        }

    def contract_summary(self) -> dict[str, Any]:
        subjects = sorted({self.lightweight_metadata(index)["subject"] for index in range(len(self))})
        return {
            "schema": "sta_net_unified_task_dataset_v1",
            "task": asdict(self.spec),
            "sample_count": len(self),
            "subject_count": len(subjects),
            "subjects": subjects,
            "excluded_label_counts": dict(self.excluded_label_counts),
            "source_loader_contract": self.base.contract_summary(),
            "adapter": self.adapter.manifest(),
        }


def collate_sta_net(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty STA-Net batch")
    return {
        "eeg": torch.stack([sample["eeg"] for sample in samples]),
        "fnirs": torch.stack([sample["fnirs"] for sample in samples]),
        "target": torch.stack([sample["target"] for sample in samples]),
        "target_valid_mask": torch.stack([sample["target_valid_mask"] for sample in samples]),
        "sample_id": [str(sample["sample_id"]) for sample in samples],
        "subject": [str(sample["subject"]) for sample in samples],
        "record_id": [str(sample["record_id"]) for sample in samples],
        "join_key": [str(sample["join_key"]) for sample in samples],
        "adapter_state": [dict(sample["adapter_state"]) for sample in samples],
    }


def task_contract_sha256(spec: STANetTaskSpec) -> str:
    payload = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
