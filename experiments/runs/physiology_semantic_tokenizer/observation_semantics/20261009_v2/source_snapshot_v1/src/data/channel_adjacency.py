"""Spatial adjacency helpers for EEG-fNIRS source targets and visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
from scipy.spatial import Delaunay, QhullError
import torch
import torch.nn.functional as F


_FNIRS_SUFFIX_PATTERN = re.compile(r'(?:highWL|lowWL|_O|_R)$', flags=re.IGNORECASE)


# A compact 10-10 adjacency table covering the current Single-Trial montage anchors.
# The labels intentionally target the filtered 30-channel EEG montage used in training.
DEFAULT_1010_NEIGHBORS: Dict[str, Tuple[str, ...]] = {
    'af7': ('F7', 'AFF5h', 'AFp1'),
    'af3': ('F3', 'AFF1h', 'AFF5h', 'AFp1'),
    'afz': ('AFF1h', 'AFF2h', 'AFp1', 'AFp2', 'Cz'),
    'fpz': ('AFp1', 'AFp2'),
    'af4': ('F4', 'AFF2h', 'AFF6h', 'AFp2'),
    'af8': ('F8', 'AFF6h', 'AFp2'),
    'oz': ('POO1', 'POO2', 'PPO1h', 'PPO2h', 'Pz'),
    'poz': ('POO1', 'POO2', 'PPO1h', 'PPO2h', 'Pz'),
    'c5': ('FCC5h', 'CCP5h', 'T7'),
    'fc3': ('FCC3h', 'FCC5h', 'F3', 'Cz'),
    'cp3': ('CCP3h', 'CCP5h', 'P3', 'T7'),
    'c1': ('FCC3h', 'CCP3h', 'Cz', 'Pz'),
    'c2': ('FCC4h', 'CCP4h', 'Cz', 'Pz'),
    'fc4': ('FCC4h', 'FCC6h', 'F4', 'Cz'),
    'cp4': ('CCP4h', 'CCP6h', 'P4', 'T8'),
    'c6': ('FCC6h', 'CCP6h', 'T8'),
    'fp1': ('AFp1', 'AFF1h', 'F7'),
    'fp2': ('AFp2', 'AFF2h', 'F8'),
    'cp5': ('CCP5h', 'P7', 'T7'),
    'o1': ('POO1', 'PPO1h', 'P7'),
    'o2': ('POO2', 'PPO2h', 'P8'),
    'cp6': ('CCP6h', 'P8', 'T8'),
    'fc5': ('FCC5h', 'F7', 'T7'),
    'c3': ('FCC3h', 'CCP3h', 'P3', 'Cz'),
    'fc1': ('FCC3h', 'AFF1h', 'Cz'),
    'cp1': ('CCP3h', 'Pz', 'Cz'),
    'fc2': ('FCC4h', 'AFF2h', 'Cz'),
    'cp2': ('CCP4h', 'Pz', 'Cz'),
    'c4': ('FCC4h', 'CCP4h', 'P4', 'Cz'),
    'fc6': ('FCC6h', 'F8', 'T8'),
}


@dataclass(frozen=True)
class SpatialAdjacencyInfo:
    dataset_id: str
    eeg_channel_names: List[str]
    fnirs_channel_names: List[str]
    adjacency_matrix: np.ndarray
    eeg_positions_2d: np.ndarray
    eeg_positions_3d: np.ndarray
    fnirs_channel_positions_2d: np.ndarray
    fnirs_channel_positions_3d: np.ndarray
    fnirs_source_names: List[str]
    fnirs_detector_names: List[str]
    fnirs_source_positions_2d: np.ndarray
    fnirs_source_positions_3d: np.ndarray
    fnirs_detector_positions_2d: np.ndarray
    fnirs_detector_positions_3d: np.ndarray
    anchor_matches: List[Dict[str, Any]]
    warnings: List[str]
    reference_subject_id: Optional[int] = None

    def to_serializable(self) -> Dict[str, Any]:
        return {
            'dataset_id': self.dataset_id,
            'reference_subject_id': self.reference_subject_id,
            'eeg_channel_names': list(self.eeg_channel_names),
            'fnirs_channel_names': list(self.fnirs_channel_names),
            'adjacency_matrix': self.adjacency_matrix.tolist(),
            'fnirs_source_names': list(self.fnirs_source_names),
            'fnirs_detector_names': list(self.fnirs_detector_names),
            'anchor_matches': list(self.anchor_matches),
            'warnings': list(self.warnings),
        }

    def adjacency_tensor(self, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(self.adjacency_matrix, dtype=dtype, device=device)


@dataclass(frozen=True)
class EEGAdjacencyInfo:
    """Within-EEG topology derived from versioned channel geometry."""

    dataset_id: str
    channel_names: List[str]
    adjacency_matrix: np.ndarray
    edges: List[Tuple[int, int]]
    positions_2d: np.ndarray
    positions_3d: np.ndarray
    method: str
    coordinate_systems: List[str]
    coordinate_status: List[str]

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "schema": "eeg_channel_adjacency_v1",
            "dataset_id": self.dataset_id,
            "channel_names": list(self.channel_names),
            "adjacency_matrix": self.adjacency_matrix.tolist(),
            "edges": [
                {
                    "source_index": source,
                    "target_index": target,
                    "source_channel": self.channel_names[source],
                    "target_channel": self.channel_names[target],
                }
                for source, target in self.edges
            ],
            "positions_2d": self.positions_2d.tolist(),
            "positions_3d": self.positions_3d.tolist(),
            "method": self.method,
            "coordinate_systems": list(self.coordinate_systems),
            "coordinate_status": list(self.coordinate_status),
            "includes_self_loops": False,
            "intended_use": "within_eeg_channel_adjacency_only",
        }

    def adjacency_tensor(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.as_tensor(self.adjacency_matrix, dtype=dtype, device=device)


@dataclass(frozen=True)
class FNIRSChannelAdjacencyInfo:
    """Within-fNIRS topology for channels connected through a shared optode."""

    dataset_id: str
    probe: str
    channel_names: List[str]
    adjacency_matrix: np.ndarray
    edges: List[Tuple[int, int]]
    positions_3d: np.ndarray
    optode_endpoints: List[Tuple[str, str]]
    coordinate_status: List[str]
    method: str = "shared_optode_line_graph_v1"

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "schema": "fnirs_channel_adjacency_v1",
            "dataset_id": self.dataset_id,
            "probe": self.probe,
            "channel_names": list(self.channel_names),
            "adjacency_matrix": self.adjacency_matrix.tolist(),
            "edges": [
                {
                    "source_index": source,
                    "target_index": target,
                    "source_channel": self.channel_names[source],
                    "target_channel": self.channel_names[target],
                }
                for source, target in self.edges
            ],
            "positions_3d": self.positions_3d.tolist(),
            "optode_endpoints": [list(value) for value in self.optode_endpoints],
            "coordinate_status": list(self.coordinate_status),
            "method": self.method,
            "includes_self_loops": False,
            "intended_use": "within_fnirs_channel_adjacency_only",
            "prohibited_interpretation": (
                "participant_digitization_source_detector_distance_or_exact_coregistration"
            ),
        }


def build_fnirs_adjacency_from_shared_optodes(
    dataset_id: str,
    probe: str,
    channel_geometry: Sequence[Mapping[str, Any]],
    channel_names: Sequence[str],
) -> FNIRSChannelAdjacencyInfo:
    """Build the line graph of an fNIRS optode grid from geometry metadata."""

    lookup = {
        canonicalize_channel_label(str(row.get("channel_name", ""))): row
        for row in channel_geometry
        if str(row.get("modality", "fnirs")) == "fnirs"
        and str(row.get("base_record_id", row.get("record_id", ""))) == probe
    }
    rows: List[Mapping[str, Any]] = []
    endpoints: List[Tuple[str, str]] = []
    missing: List[str] = []
    for name in channel_names:
        row = lookup.get(canonicalize_channel_label(name))
        raw_endpoints = row.get("metadata", {}).get("optode_endpoints", []) if row else []
        if (
            row is None
            or any(row.get(axis) is None for axis in ("x", "y", "z"))
            or len(raw_endpoints) != 2
        ):
            missing.append(str(name))
            continue
        rows.append(row)
        endpoints.append((str(raw_endpoints[0]), str(raw_endpoints[1])))
    if missing:
        raise ValueError(
            f"fNIRS shared-optode adjacency requires complete geometry; missing channels: {missing}"
        )

    edge_set: set[Tuple[int, int]] = set()
    for left in range(len(channel_names)):
        for right in range(left + 1, len(channel_names)):
            if set(endpoints[left]) & set(endpoints[right]):
                edge_set.add((left, right))
    edges = sorted(edge_set)
    adjacency = np.zeros((len(channel_names), len(channel_names)), dtype=np.float32)
    for source, target in edges:
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    positions = np.asarray(
        [[float(row[axis]) for axis in ("x", "y", "z")] for row in rows],
        dtype=np.float32,
    )
    return FNIRSChannelAdjacencyInfo(
        dataset_id=str(dataset_id),
        probe=str(probe),
        channel_names=[str(value) for value in channel_names],
        adjacency_matrix=adjacency,
        edges=edges,
        positions_3d=positions,
        optode_endpoints=endpoints,
        coordinate_status=[
            str(row.get("metadata", {}).get("coordinate_status", "unspecified"))
            for row in rows
        ],
    )


def build_eeg_adjacency_from_geometry(
    dataset_id: str,
    channel_geometry: Sequence[Mapping[str, Any]],
    channel_names: Sequence[str],
) -> EEGAdjacencyInfo:
    """Build a deterministic symmetric EEG graph from a top-view Delaunay mesh.

    The graph uses only relative template/measured electrode topology.  It does
    not imply participant-specific digitization or EEG-fNIRS co-registration.
    """

    lookup = {
        canonicalize_channel_label(str(row.get("channel_name", ""))): row
        for row in channel_geometry
        if str(row.get("modality", "eeg")) == "eeg"
    }
    rows: List[Mapping[str, Any]] = []
    missing: List[str] = []
    for name in channel_names:
        row = lookup.get(canonicalize_channel_label(name))
        if row is None or any(row.get(axis) is None for axis in ("x", "y", "z")):
            missing.append(str(name))
        else:
            rows.append(row)
    if missing:
        raise ValueError(f"EEG adjacency requires complete geometry; missing channels: {missing}")
    if len(rows) < 3:
        raise ValueError("EEG adjacency requires at least three positioned channels")

    positions_3d = np.asarray(
        [[float(row[axis]) for axis in ("x", "y", "z")] for row in rows],
        dtype=np.float64,
    )
    positions_2d = positions_3d[:, :2]
    if np.linalg.matrix_rank(positions_2d - positions_2d.mean(axis=0, keepdims=True)) < 2:
        raise ValueError("EEG top-view coordinates are degenerate")
    try:
        triangulation = Delaunay(positions_2d)
    except QhullError as exc:
        raise ValueError("could not triangulate EEG top-view coordinates") from exc

    edge_set: set[Tuple[int, int]] = set()
    for simplex in triangulation.simplices:
        indices = [int(value) for value in simplex]
        for left, right in ((0, 1), (1, 2), (0, 2)):
            edge_set.add(tuple(sorted((indices[left], indices[right]))))
    edges = sorted(edge_set)
    adjacency = np.zeros((len(channel_names), len(channel_names)), dtype=np.float32)
    for source, target in edges:
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0

    return EEGAdjacencyInfo(
        dataset_id=str(dataset_id),
        channel_names=[str(value) for value in channel_names],
        adjacency_matrix=adjacency,
        edges=edges,
        positions_2d=positions_2d.astype(np.float32),
        positions_3d=positions_3d.astype(np.float32),
        method="top_view_xy_delaunay_v1",
        coordinate_systems=[str(row.get("coordinate_system", "unknown")) for row in rows],
        coordinate_status=[str(row.get("coordinate_status", "unspecified")) for row in rows],
    )


def canonicalize_channel_label(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', str(name)).lower()


def strip_fnirs_chromophore_suffix(name: str) -> str:
    return _FNIRS_SUFFIX_PATTERN.sub('', str(name))


def _coerce_name_list(values: Any) -> List[str]:
    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        return [str(array.item())]
    return [str(item) for item in array.tolist()]


def _coerce_positions_2d(x_values: Any, y_values: Any) -> np.ndarray:
    x = np.asarray(x_values, dtype=np.float32).reshape(-1)
    y = np.asarray(y_values, dtype=np.float32).reshape(-1)
    return np.stack([x, y], axis=1)


def _coerce_positions_3d(values: Any) -> np.ndarray:
    pos = np.asarray(values, dtype=np.float32)
    if pos.ndim != 2:
        raise ValueError(f'Expected 2D positions, got shape {pos.shape}')
    if pos.shape[0] == 3:
        pos = pos.T
    return pos


def _resolve_eeg_montage_path(data_root: Path, subject_id: int, *, use_artifact_data: bool) -> Path:
    subject_dir = data_root / 'EEG_01-29' / f'subject {subject_id:02d}'
    if use_artifact_data:
        artifact_path = subject_dir / 'with occular artifact' / 'mnt.mat'
        if artifact_path.exists():
            return artifact_path
        fallback_artifact = subject_dir / 'mnt_artifact.mat'
        if fallback_artifact.exists():
            return fallback_artifact
    standard_path = subject_dir / 'mnt.mat'
    if standard_path.exists():
        return standard_path
    raise FileNotFoundError(f'Could not find EEG montage file under {subject_dir}')


def _resolve_fnirs_montage_path(data_root: Path, subject_id: int) -> Path:
    subject_dir = data_root / 'NIRS_01-29' / f'subject {subject_id:02d}'
    standard_path = subject_dir / 'mnt.mat'
    if standard_path.exists():
        return standard_path
    artifact_path = subject_dir / 'mnt_artifact.mat'
    if artifact_path.exists():
        return artifact_path
    raise FileNotFoundError(f'Could not find fNIRS montage file under {subject_dir}')


def _load_mnt(path: Path):
    data = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    if 'mnt' not in data:
        raise KeyError(f'Montage file {path} does not contain an mnt struct')
    return data['mnt']


def _filter_layout(names: Sequence[str], positions_2d: np.ndarray, positions_3d: np.ndarray, requested_names: Sequence[str]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    name_to_index = {canonicalize_channel_label(name): index for index, name in enumerate(names)}
    resolved_indices: List[int] = []
    for requested_name in requested_names:
        key = canonicalize_channel_label(requested_name)
        if key not in name_to_index:
            raise ValueError(f'Channel {requested_name!r} is not available in montage metadata')
        resolved_indices.append(name_to_index[key])
    return list(requested_names), positions_2d[resolved_indices], positions_3d[resolved_indices]


def _safe_distances(anchor_position: np.ndarray, eeg_positions_3d: np.ndarray) -> np.ndarray:
    diff = eeg_positions_3d - anchor_position.reshape(1, 3)
    return np.sqrt(np.sum(diff * diff, axis=1))


def _anchor_weight_vector(
    *,
    anchor_name: str,
    anchor_position: np.ndarray,
    eeg_channel_names: Sequence[str],
    eeg_positions_3d: np.ndarray,
    warnings: List[str],
    neighbor_fallback_k: int = 2,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    weights = np.zeros(len(eeg_channel_names), dtype=np.float32)
    if not anchor_name or anchor_name == '-':
        return weights, {
            'anchor_name': anchor_name,
            'selection_mode': 'missing',
            'direct_labels': [],
            'neighbor_labels': [],
        }

    distances = _safe_distances(anchor_position, eeg_positions_3d)
    nearest_indices = np.argsort(distances)
    nearest_index = int(nearest_indices[0])
    nearest_label = eeg_channel_names[nearest_index]
    anchor_key = canonicalize_channel_label(anchor_name)

    exact_indices = [
        index
        for index, channel_name in enumerate(eeg_channel_names)
        if canonicalize_channel_label(channel_name) == anchor_key
    ]
    direct_indices: List[int]
    selection_mode = 'geometry'
    if exact_indices:
        exact_index = int(exact_indices[0])
        if exact_index != nearest_index:
            warnings.append(
                f'Anchor {anchor_name} disagrees with nearest 3D EEG electrode '
                f'({eeg_channel_names[exact_index]} vs {nearest_label}); using 3D match.'
            )
            direct_indices = [nearest_index]
        else:
            direct_indices = [exact_index]
            selection_mode = 'name+geometry'
    else:
        direct_indices = [nearest_index]

    for direct_index in direct_indices:
        weights[direct_index] += 1.0

    neighbor_indices: List[int] = []
    for neighbor_label in DEFAULT_1010_NEIGHBORS.get(anchor_key, ()): 
        for index, channel_name in enumerate(eeg_channel_names):
            if canonicalize_channel_label(channel_name) == canonicalize_channel_label(neighbor_label):
                if index not in direct_indices and index not in neighbor_indices:
                    neighbor_indices.append(index)
                break

    if not neighbor_indices:
        for neighbor_index in nearest_indices[1: 1 + max(int(neighbor_fallback_k), 0)]:
            if int(neighbor_index) not in direct_indices and int(neighbor_index) not in neighbor_indices:
                neighbor_indices.append(int(neighbor_index))

    for neighbor_index in neighbor_indices:
        weights[neighbor_index] += 0.5

    return weights, {
        'anchor_name': anchor_name,
        'selection_mode': selection_mode,
        'direct_labels': [eeg_channel_names[index] for index in direct_indices],
        'neighbor_labels': [eeg_channel_names[index] for index in neighbor_indices],
    }


def build_channel_adjacency(
    dataset_id: str,
    data_root: str | Path,
    eeg_channel_names: Sequence[str],
    fnirs_channel_names: Sequence[str],
    *,
    reference_subject_id: int = 1,
    use_artifact_data: bool = True,
) -> SpatialAdjacencyInfo:
    if str(dataset_id) != 'eeg_fnirs_single_trial':
        raise ValueError(f'Spatial adjacency currently supports eeg_fnirs_single_trial only, got {dataset_id!r}')

    root = Path(data_root)
    eeg_mnt = _load_mnt(_resolve_eeg_montage_path(root, int(reference_subject_id), use_artifact_data=use_artifact_data))
    fnirs_mnt = _load_mnt(_resolve_fnirs_montage_path(root, int(reference_subject_id)))

    eeg_names_full = _coerce_name_list(eeg_mnt.clab)
    eeg_pos_2d_full = _coerce_positions_2d(eeg_mnt.x, eeg_mnt.y)
    eeg_pos_3d_full = _coerce_positions_3d(eeg_mnt.pos_3d)
    eeg_names, eeg_pos_2d, eeg_pos_3d = _filter_layout(
        eeg_names_full,
        eeg_pos_2d_full,
        eeg_pos_3d_full,
        eeg_channel_names,
    )

    fnirs_base_names_full = _coerce_name_list(fnirs_mnt.clab)
    fnirs_channel_pos_2d_full = _coerce_positions_2d(fnirs_mnt.x, fnirs_mnt.y)
    fnirs_channel_pos_3d_full = _coerce_positions_3d(fnirs_mnt.pos_3d)
    source_names_full = _coerce_name_list(fnirs_mnt.source.clab)
    source_pos_2d_full = _coerce_positions_2d(fnirs_mnt.source.x, fnirs_mnt.source.y)
    source_pos_3d_full = _coerce_positions_3d(fnirs_mnt.source.pos_3d)
    detector_names_full = _coerce_name_list(fnirs_mnt.detector.clab)
    detector_pos_2d_full = _coerce_positions_2d(fnirs_mnt.detector.x, fnirs_mnt.detector.y)
    detector_pos_3d_full = _coerce_positions_3d(fnirs_mnt.detector.pos_3d)
    sd_pairs = np.asarray(fnirs_mnt.sd, dtype=np.int64)
    if sd_pairs.ndim != 2 or sd_pairs.shape[1] != 2:
        raise ValueError(f'Expected sd matrix with shape [n_channels, 2], got {sd_pairs.shape}')

    fnirs_base_to_index = {
        canonicalize_channel_label(name): index for index, name in enumerate(fnirs_base_names_full)
    }

    adjacency_rows: List[np.ndarray] = []
    fnirs_channel_pos_2d: List[np.ndarray] = []
    fnirs_channel_pos_3d: List[np.ndarray] = []
    fnirs_source_names: List[str] = []
    fnirs_detector_names: List[str] = []
    fnirs_source_pos_2d: List[np.ndarray] = []
    fnirs_source_pos_3d: List[np.ndarray] = []
    fnirs_detector_pos_2d: List[np.ndarray] = []
    fnirs_detector_pos_3d: List[np.ndarray] = []
    anchor_matches: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for requested_name in fnirs_channel_names:
        base_name = strip_fnirs_chromophore_suffix(requested_name)
        base_key = canonicalize_channel_label(base_name)
        if base_key not in fnirs_base_to_index:
            raise ValueError(f'fNIRS channel {requested_name!r} is not available in montage metadata')
        base_index = int(fnirs_base_to_index[base_key])
        source_index = int(sd_pairs[base_index, 0]) - 1
        detector_index = int(sd_pairs[base_index, 1]) - 1

        source_name = source_names_full[source_index]
        detector_name = detector_names_full[detector_index]
        source_position = source_pos_3d_full[source_index]
        detector_position = detector_pos_3d_full[detector_index]

        source_weights, source_match = _anchor_weight_vector(
            anchor_name=source_name,
            anchor_position=source_position,
            eeg_channel_names=eeg_names,
            eeg_positions_3d=eeg_pos_3d,
            warnings=warnings,
        )
        detector_weights, detector_match = _anchor_weight_vector(
            anchor_name=detector_name,
            anchor_position=detector_position,
            eeg_channel_names=eeg_names,
            eeg_positions_3d=eeg_pos_3d,
            warnings=warnings,
        )

        weights = source_weights + detector_weights
        if float(weights.sum()) <= 0.0:
            channel_position = fnirs_channel_pos_3d_full[base_index]
            fallback_weights, fallback_match = _anchor_weight_vector(
                anchor_name=base_name,
                anchor_position=channel_position,
                eeg_channel_names=eeg_names,
                eeg_positions_3d=eeg_pos_3d,
                warnings=warnings,
            )
            weights = fallback_weights
            source_match = fallback_match
            detector_match = {
                'anchor_name': '-',
                'selection_mode': 'fallback_channel_position',
                'direct_labels': [],
                'neighbor_labels': [],
            }
        weights = weights / np.clip(weights.sum(), 1e-8, None)

        adjacency_rows.append(weights.astype(np.float32, copy=False))
        fnirs_channel_pos_2d.append(
            0.5 * (source_pos_2d_full[source_index] + detector_pos_2d_full[detector_index])
        )
        fnirs_channel_pos_3d.append(fnirs_channel_pos_3d_full[base_index])
        fnirs_source_names.append(source_name)
        fnirs_detector_names.append(detector_name)
        fnirs_source_pos_2d.append(source_pos_2d_full[source_index])
        fnirs_source_pos_3d.append(source_pos_3d_full[source_index])
        fnirs_detector_pos_2d.append(detector_pos_2d_full[detector_index])
        fnirs_detector_pos_3d.append(detector_pos_3d_full[detector_index])
        anchor_matches.append(
            {
                'fnirs_channel': requested_name,
                'base_channel': base_name,
                'source_anchor': source_match,
                'detector_anchor': detector_match,
            }
        )

    return SpatialAdjacencyInfo(
        dataset_id=str(dataset_id),
        reference_subject_id=int(reference_subject_id),
        eeg_channel_names=list(eeg_names),
        fnirs_channel_names=list(fnirs_channel_names),
        adjacency_matrix=np.stack(adjacency_rows, axis=0),
        eeg_positions_2d=eeg_pos_2d.astype(np.float32, copy=False),
        eeg_positions_3d=eeg_pos_3d.astype(np.float32, copy=False),
        fnirs_channel_positions_2d=np.stack(fnirs_channel_pos_2d, axis=0).astype(np.float32, copy=False),
        fnirs_channel_positions_3d=np.stack(fnirs_channel_pos_3d, axis=0).astype(np.float32, copy=False),
        fnirs_source_names=fnirs_source_names,
        fnirs_detector_names=fnirs_detector_names,
        fnirs_source_positions_2d=np.stack(fnirs_source_pos_2d, axis=0).astype(np.float32, copy=False),
        fnirs_source_positions_3d=np.stack(fnirs_source_pos_3d, axis=0).astype(np.float32, copy=False),
        fnirs_detector_positions_2d=np.stack(fnirs_detector_pos_2d, axis=0).astype(np.float32, copy=False),
        fnirs_detector_positions_3d=np.stack(fnirs_detector_pos_3d, axis=0).astype(np.float32, copy=False),
        anchor_matches=anchor_matches,
        warnings=warnings,
    )


def _resolve_smoothing_kernel_size(smoothing_samples: int) -> int:
    kernel_size = max(int(smoothing_samples), 1)
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def compute_per_channel_rms_envelope(
    eeg: torch.Tensor,
    *,
    smoothing_samples: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    if eeg.ndim != 3:
        raise ValueError(f'Expected EEG tensor [B, C, T], got shape {tuple(eeg.shape)}')

    power = eeg.pow(2)
    kernel_size = _resolve_smoothing_kernel_size(smoothing_samples)
    if kernel_size > 1:
        window = torch.hann_window(kernel_size, device=eeg.device, dtype=eeg.dtype)
        window = window / window.sum().clamp_min(float(eps))
        batch_size, channels, time_steps = power.shape
        working = power.reshape(batch_size * channels, 1, time_steps)
        pad = kernel_size // 2
        if time_steps > pad:
            working = F.pad(working, (pad, pad), mode='reflect')
        else:
            working = F.pad(working, (pad, pad), mode='replicate')
        power = F.conv1d(working, window.view(1, 1, -1)).reshape(batch_size, channels, time_steps)

    return torch.sqrt(power.clamp_min(0.0) + float(eps))


def downsample_temporal_driver(driver: torch.Tensor, target_length: int) -> torch.Tensor:
    if driver.ndim != 3:
        raise ValueError(f'Expected driver tensor [B, C, T], got shape {tuple(driver.shape)}')
    if driver.shape[-1] == int(target_length):
        return driver
    if driver.shape[-1] % int(target_length) == 0:
        factor = driver.shape[-1] // int(target_length)
        batch_size, channels, _ = driver.shape
        pooled = F.avg_pool1d(driver.reshape(batch_size * channels, 1, driver.shape[-1]), kernel_size=factor, stride=factor)
        return pooled.reshape(batch_size, channels, int(target_length))
    return F.interpolate(driver, size=int(target_length), mode='linear', align_corners=False)


def compute_spatial_fnirs_driver(
    eeg: torch.Tensor,
    adjacency_matrix: torch.Tensor,
    *,
    target_length: int,
) -> torch.Tensor:
    if eeg.ndim != 3:
        raise ValueError(f'Expected EEG tensor [B, C, T], got shape {tuple(eeg.shape)}')
    if adjacency_matrix.ndim != 2:
        raise ValueError(
            f'Expected adjacency matrix [F, E], got shape {tuple(adjacency_matrix.shape)}'
        )
    eeg_power = eeg.pow(2)
    weighted = torch.einsum('fe,bet->bft', adjacency_matrix.to(device=eeg.device, dtype=eeg.dtype), eeg_power)
    return downsample_temporal_driver(weighted, int(target_length))


def compute_cross_modal_correlation_matrix(
    eeg: torch.Tensor,
    fnirs: torch.Tensor,
    *,
    fnirs_target_length: Optional[int] = None,
) -> torch.Tensor:
    if eeg.ndim != 3 or fnirs.ndim != 3:
        raise ValueError(
            f'Expected EEG/fNIRS tensors [B, C, T], got {tuple(eeg.shape)} and {tuple(fnirs.shape)}'
        )
    target_length = int(fnirs.shape[-1] if fnirs_target_length is None else fnirs_target_length)
    eeg_driver = downsample_temporal_driver(eeg.pow(2), target_length)
    eeg_flat = eeg_driver.permute(1, 0, 2).reshape(eeg_driver.shape[1], -1)
    fnirs_flat = fnirs.permute(1, 0, 2).reshape(fnirs.shape[1], -1)
    eeg_norm = (eeg_flat - eeg_flat.mean(dim=1, keepdim=True)) / eeg_flat.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    fnirs_norm = (fnirs_flat - fnirs_flat.mean(dim=1, keepdim=True)) / fnirs_flat.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return torch.matmul(fnirs_norm, eeg_norm.t()) / max(eeg_norm.shape[1], 1)


def _maybe_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def compute_fnirs_midpoint_diagnostics(adjacency: SpatialAdjacencyInfo) -> Dict[str, Any]:
    source_positions = np.asarray(adjacency.fnirs_source_positions_3d, dtype=np.float32)
    detector_positions = np.asarray(adjacency.fnirs_detector_positions_3d, dtype=np.float32)
    channel_positions = np.asarray(adjacency.fnirs_channel_positions_3d, dtype=np.float32)
    direct_midpoints = 0.5 * (source_positions + detector_positions)
    midpoint_norms = np.linalg.norm(direct_midpoints, axis=1, keepdims=True)
    normalized_midpoints = direct_midpoints / np.clip(midpoint_norms, 1e-8, None)

    direct_errors = np.linalg.norm(channel_positions - direct_midpoints, axis=1)
    normalized_errors = np.linalg.norm(channel_positions - normalized_midpoints, axis=1)

    return {
        'direct_midpoints_3d': direct_midpoints,
        'normalized_midpoints_3d': normalized_midpoints,
        'direct_midpoint_error_mean': float(np.mean(direct_errors)),
        'direct_midpoint_error_max': float(np.max(direct_errors)),
        'normalized_midpoint_error_mean': float(np.mean(normalized_errors)),
        'normalized_midpoint_error_max': float(np.max(normalized_errors)),
        'per_channel_direct_midpoint_errors': direct_errors.tolist(),
        'per_channel_normalized_midpoint_errors': normalized_errors.tolist(),
    }


def _set_equal_3d_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1e-3)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def project_points_to_2d(points_3d: np.ndarray, *, method: str = 'orthographic') -> np.ndarray:
    points = np.asarray(points_3d, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'Expected points shaped [N, 3], got {points.shape}')

    if method == 'orthographic':
        return points[:, :2].copy()

    if method == 'stereographic':
        denom = 1.0 + points[:, 2:3]
        return points[:, :2] / np.clip(denom, 1e-6, None)

    raise ValueError(f'Unsupported 2D projection method: {method}')


def plot_channel_layout_3d(adjacency: SpatialAdjacencyInfo, output_path: Path) -> Optional[str]:
    plt = _maybe_import_matplotlib()
    if plt is None:
        return None

    diagnostics = compute_fnirs_midpoint_diagnostics(adjacency)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 8))
    ax_layout = fig.add_subplot(1, 2, 1, projection='3d')
    ax_confirm = fig.add_subplot(1, 2, 2, projection='3d')

    eeg_xyz = np.asarray(adjacency.eeg_positions_3d, dtype=np.float32)
    fnirs_channel_xyz = np.asarray(adjacency.fnirs_channel_positions_3d, dtype=np.float32)
    fnirs_source_xyz = np.asarray(adjacency.fnirs_source_positions_3d, dtype=np.float32)
    fnirs_detector_xyz = np.asarray(adjacency.fnirs_detector_positions_3d, dtype=np.float32)
    direct_midpoints = np.asarray(diagnostics['direct_midpoints_3d'], dtype=np.float32)
    normalized_midpoints = np.asarray(diagnostics['normalized_midpoints_3d'], dtype=np.float32)

    ax_layout.scatter(eeg_xyz[:, 0], eeg_xyz[:, 1], eeg_xyz[:, 2], c='#2E86AB', s=36, label='EEG electrodes', depthshade=False)
    ax_layout.scatter(fnirs_channel_xyz[:, 0], fnirs_channel_xyz[:, 1], fnirs_channel_xyz[:, 2], c='#6C757D', s=28, marker='x', label='fNIRS channels', depthshade=False)
    ax_layout.scatter(fnirs_source_xyz[:, 0], fnirs_source_xyz[:, 1], fnirs_source_xyz[:, 2], c='#F18F01', s=56, marker='^', label='fNIRS sources', depthshade=False)
    ax_layout.scatter(fnirs_detector_xyz[:, 0], fnirs_detector_xyz[:, 1], fnirs_detector_xyz[:, 2], c='#2ECC71', s=56, marker='s', label='fNIRS detectors', depthshade=False)

    for source_point, detector_point in zip(fnirs_source_xyz, fnirs_detector_xyz):
        ax_layout.plot(
            [float(source_point[0]), float(detector_point[0])],
            [float(source_point[1]), float(detector_point[1])],
            [float(source_point[2]), float(detector_point[2])],
            color='#ADB5BD',
            linewidth=0.8,
            alpha=0.35,
        )
    for name, point in zip(adjacency.fnirs_channel_names, fnirs_channel_xyz):
        ax_layout.text(point[0], point[1], point[2], strip_fnirs_chromophore_suffix(name), fontsize=6, color='#495057')

    ax_layout.set_title('3D EEG-fNIRS layout')
    ax_layout.set_xlabel('x')
    ax_layout.set_ylabel('y')
    ax_layout.set_zlabel('z')
    ax_layout.legend(loc='upper left')

    ax_confirm.scatter(fnirs_source_xyz[:, 0], fnirs_source_xyz[:, 1], fnirs_source_xyz[:, 2], c='#F18F01', s=42, marker='^', label='sources', depthshade=False)
    ax_confirm.scatter(fnirs_detector_xyz[:, 0], fnirs_detector_xyz[:, 1], fnirs_detector_xyz[:, 2], c='#2ECC71', s=42, marker='s', label='detectors', depthshade=False)
    ax_confirm.scatter(fnirs_channel_xyz[:, 0], fnirs_channel_xyz[:, 1], fnirs_channel_xyz[:, 2], c='#343A40', s=32, marker='x', label='channel positions', depthshade=False)
    ax_confirm.scatter(direct_midpoints[:, 0], direct_midpoints[:, 1], direct_midpoints[:, 2], c='#D6336C', s=26, marker='o', alpha=0.45, label='3D direct midpoints', depthshade=False)
    ax_confirm.scatter(normalized_midpoints[:, 0], normalized_midpoints[:, 1], normalized_midpoints[:, 2], facecolors='none', edgecolors='#C92A2A', s=72, marker='o', linewidths=1.2, label='3D normalized midpoints', depthshade=False)

    for source_point, detector_point, channel_point, midpoint_point in zip(
        fnirs_source_xyz,
        fnirs_detector_xyz,
        fnirs_channel_xyz,
        normalized_midpoints,
    ):
        ax_confirm.plot(
            [float(source_point[0]), float(detector_point[0])],
            [float(source_point[1]), float(detector_point[1])],
            [float(source_point[2]), float(detector_point[2])],
            color='#ADB5BD',
            linewidth=0.8,
            alpha=0.25,
        )
        ax_confirm.plot(
            [float(channel_point[0]), float(midpoint_point[0])],
            [float(channel_point[1]), float(midpoint_point[1])],
            [float(channel_point[2]), float(midpoint_point[2])],
            color='#C92A2A',
            linewidth=0.8,
            alpha=0.35,
        )

    ax_confirm.set_title(
        'fNIRS channels vs source-detector midpoints\n'
        f"direct mean err={diagnostics['direct_midpoint_error_mean']:.4f}, "
        f"normalized mean err={diagnostics['normalized_midpoint_error_mean']:.4e}"
    )
    ax_confirm.set_xlabel('x')
    ax_confirm.set_ylabel('y')
    ax_confirm.set_zlabel('z')
    ax_confirm.legend(loc='upper left')

    all_points = np.concatenate(
        [eeg_xyz, fnirs_channel_xyz, fnirs_source_xyz, fnirs_detector_xyz, normalized_midpoints],
        axis=0,
    )
    _set_equal_3d_axes(ax_layout, all_points)
    _set_equal_3d_axes(ax_confirm, all_points)
    ax_layout.view_init(elev=18.0, azim=-58.0)
    ax_confirm.view_init(elev=18.0, azim=-58.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return str(output_path)


def plot_channel_layout_projected_2d(
    adjacency: SpatialAdjacencyInfo,
    output_path: Path,
    *,
    projection_method: str = 'orthographic',
) -> Optional[str]:
    plt = _maybe_import_matplotlib()
    if plt is None:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))

    eeg_xy = project_points_to_2d(adjacency.eeg_positions_3d, method=projection_method)
    fnirs_channel_xy = project_points_to_2d(adjacency.fnirs_channel_positions_3d, method=projection_method)
    fnirs_source_xy = project_points_to_2d(adjacency.fnirs_source_positions_3d, method=projection_method)
    fnirs_detector_xy = project_points_to_2d(adjacency.fnirs_detector_positions_3d, method=projection_method)

    ax.scatter(eeg_xy[:, 0], eeg_xy[:, 1], c='#2E86AB', s=48, label='EEG electrodes', zorder=3)
    ax.scatter(fnirs_channel_xy[:, 0], fnirs_channel_xy[:, 1], c='#6C757D', s=36, marker='x', label='fNIRS channels', zorder=2)
    ax.scatter(fnirs_source_xy[:, 0], fnirs_source_xy[:, 1], c='#F18F01', s=72, marker='^', label='fNIRS sources', zorder=4)
    ax.scatter(fnirs_detector_xy[:, 0], fnirs_detector_xy[:, 1], c='#2ECC71', s=72, marker='s', label='fNIRS detectors', zorder=4)

    for source_point, detector_point in zip(fnirs_source_xy, fnirs_detector_xy):
        ax.plot(
            [float(source_point[0]), float(detector_point[0])],
            [float(source_point[1]), float(detector_point[1])],
            color='#ADB5BD',
            linewidth=0.8,
            alpha=0.35,
            zorder=1,
        )

    for name, (x_coord, y_coord) in zip(adjacency.eeg_channel_names, eeg_xy):
        ax.text(x_coord, y_coord, name, fontsize=7, color='#1B4F72', ha='center', va='bottom')
    for name, (x_coord, y_coord) in zip(adjacency.fnirs_channel_names, fnirs_channel_xy):
        ax.text(x_coord, y_coord, strip_fnirs_chromophore_suffix(name), fontsize=6, color='#495057', ha='center', va='top')

    ax.set_title(f'EEG-fNIRS spatial layout (2D projection from 3D: {projection_method})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return str(output_path)


def plot_channel_layout(adjacency: SpatialAdjacencyInfo, output_path: Path) -> Optional[str]:
    plt = _maybe_import_matplotlib()
    if plt is None:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    eeg_xy = adjacency.eeg_positions_2d
    fnirs_channel_xy = adjacency.fnirs_channel_positions_2d
    fnirs_source_xy = adjacency.fnirs_source_positions_2d
    fnirs_detector_xy = adjacency.fnirs_detector_positions_2d

    ax.scatter(eeg_xy[:, 0], eeg_xy[:, 1], c='#2E86AB', s=48, label='EEG electrodes', zorder=3)
    ax.scatter(fnirs_channel_xy[:, 0], fnirs_channel_xy[:, 1], c='#6C757D', s=36, marker='x', label='fNIRS channels', zorder=2)
    ax.scatter(fnirs_source_xy[:, 0], fnirs_source_xy[:, 1], c='#F18F01', s=72, marker='^', label='fNIRS sources', zorder=4)
    ax.scatter(fnirs_detector_xy[:, 0], fnirs_detector_xy[:, 1], c='#2ECC71', s=72, marker='s', label='fNIRS detectors', zorder=4)

    for name, (x_coord, y_coord) in zip(adjacency.eeg_channel_names, eeg_xy):
        ax.text(x_coord, y_coord, name, fontsize=7, color='#1B4F72', ha='center', va='bottom')
    for name, (x_coord, y_coord) in zip(adjacency.fnirs_channel_names, fnirs_channel_xy):
        ax.text(x_coord, y_coord, strip_fnirs_chromophore_suffix(name), fontsize=6, color='#495057', ha='center', va='top')

    ax.set_title('EEG-fNIRS spatial layout (stored 2D; fNIRS channels at source-detector midpoints)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return str(output_path)


def plot_adjacency_heatmap(adjacency: SpatialAdjacencyInfo, output_path: Path) -> Optional[str]:
    plt = _maybe_import_matplotlib()
    if plt is None:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(10, 0.3 * len(adjacency.eeg_channel_names)), max(8, 0.28 * len(adjacency.fnirs_channel_names))))
    image = ax.imshow(adjacency.adjacency_matrix, aspect='auto', interpolation='nearest', cmap='viridis')
    ax.set_title('fNIRS-to-EEG spatial adjacency weights')
    ax.set_xlabel('EEG channels')
    ax.set_ylabel('fNIRS channels')
    ax.set_xticks(np.arange(len(adjacency.eeg_channel_names)))
    ax.set_xticklabels(adjacency.eeg_channel_names, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(adjacency.fnirs_channel_names)))
    ax.set_yticklabels([strip_fnirs_chromophore_suffix(name) for name in adjacency.fnirs_channel_names], fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label='Normalized weight')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return str(output_path)


def plot_cross_modal_correlation_heatmap(
    correlation_matrix: np.ndarray,
    adjacency: SpatialAdjacencyInfo,
    output_path: Path,
) -> Optional[str]:
    plt = _maybe_import_matplotlib()
    if plt is None:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corr = np.asarray(correlation_matrix, dtype=np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    adjacency_image = axes[0].imshow(adjacency.adjacency_matrix, aspect='auto', interpolation='nearest', cmap='viridis')
    corr_image = axes[1].imshow(corr, aspect='auto', interpolation='nearest', cmap='coolwarm', vmin=-1.0, vmax=1.0)
    axes[0].set_title('Spatial adjacency')
    axes[1].set_title('EEG power vs fNIRS signal correlation')
    for axis in axes:
        axis.set_xlabel('EEG channels')
        axis.set_xticks(np.arange(len(adjacency.eeg_channel_names)))
        axis.set_xticklabels(adjacency.eeg_channel_names, rotation=90, fontsize=7)
    axes[0].set_ylabel('fNIRS channels')
    axes[0].set_yticks(np.arange(len(adjacency.fnirs_channel_names)))
    axes[0].set_yticklabels([strip_fnirs_chromophore_suffix(name) for name in adjacency.fnirs_channel_names], fontsize=7)
    fig.colorbar(adjacency_image, ax=axes[0], fraction=0.046, pad=0.04, label='Weight')
    fig.colorbar(corr_image, ax=axes[1], fraction=0.046, pad=0.04, label='Pearson r')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return str(output_path)


def plot_spatial_target_preview(
    *,
    adjacency: SpatialAdjacencyInfo,
    eeg: np.ndarray,
    fnirs: np.ndarray,
    eeg_target: np.ndarray,
    fnirs_target: np.ndarray,
    eeg_sample_rate_hz: float,
    fnirs_sample_rate_hz: float,
    output_path: Path,
    sample_index: int = 0,
    max_channels: int = 4,
) -> Optional[str]:
    plt = _maybe_import_matplotlib()
    if plt is None:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eeg = np.asarray(eeg, dtype=np.float32)
    fnirs = np.asarray(fnirs, dtype=np.float32)
    eeg_target = np.asarray(eeg_target, dtype=np.float32)
    fnirs_target = np.asarray(fnirs_target, dtype=np.float32)
    if eeg.ndim != 3 or fnirs.ndim != 3:
        raise ValueError('Expected batched arrays shaped [B, C, T] for preview plotting')

    eeg_count = max(1, min(int(max_channels), eeg.shape[1]))
    fnirs_count = max(1, min(int(max_channels), fnirs.shape[1]))
    eeg_indices = np.linspace(0, eeg.shape[1] - 1, num=eeg_count, dtype=np.int64)
    fnirs_indices = np.linspace(0, fnirs.shape[1] - 1, num=fnirs_count, dtype=np.int64)

    n_rows = max(eeg_count, fnirs_count)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 3.2 * n_rows), squeeze=False)
    eeg_time = np.arange(eeg.shape[-1], dtype=np.float32) / max(float(eeg_sample_rate_hz), 1e-6)
    fnirs_time = np.arange(fnirs.shape[-1], dtype=np.float32) / max(float(fnirs_sample_rate_hz), 1e-6)

    for row in range(n_rows):
        left_axis = axes[row, 0]
        if row < eeg_count:
            channel_index = int(eeg_indices[row])
            label = adjacency.eeg_channel_names[channel_index]
            left_axis.plot(eeg_time, eeg[sample_index, channel_index], label='Model input', color='#2E86AB', linewidth=1.0)
            left_axis.plot(eeg_time, eeg_target[sample_index, channel_index], label='RMS target', color='#F18F01', linewidth=1.0)
            left_axis.set_title(f'EEG {label}')
            left_axis.set_xlabel('Time (s)')
            left_axis.set_ylabel('Amplitude')
            left_axis.grid(alpha=0.25)
            if row == 0:
                left_axis.legend(loc='upper right')
        else:
            left_axis.axis('off')

        right_axis = axes[row, 1]
        if row < fnirs_count:
            channel_index = int(fnirs_indices[row])
            label = strip_fnirs_chromophore_suffix(adjacency.fnirs_channel_names[channel_index])
            top_driver_indices = np.argsort(adjacency.adjacency_matrix[channel_index])[::-1][:3]
            top_driver_text = ', '.join(
                f"{adjacency.eeg_channel_names[index]} ({adjacency.adjacency_matrix[channel_index, index]:.2f})"
                for index in top_driver_indices
            )
            right_axis.plot(fnirs_time, fnirs[sample_index, channel_index], label='Model input', color='#6C757D', linewidth=1.0)
            right_axis.plot(fnirs_time, fnirs_target[sample_index, channel_index], label='Spatial HRF target', color='#2ECC71', linewidth=1.0)
            right_axis.set_title(f'fNIRS {label} | drivers: {top_driver_text}')
            right_axis.set_xlabel('Time (s)')
            right_axis.set_ylabel('Amplitude')
            right_axis.grid(alpha=0.25)
            if row == 0:
                right_axis.legend(loc='upper right')
        else:
            right_axis.axis('off')

    fig.suptitle(f'Spatial source-target preview (sample {sample_index})', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return str(output_path)


__all__ = [
    'DEFAULT_1010_NEIGHBORS',
    'EEGAdjacencyInfo',
    'SpatialAdjacencyInfo',
    'build_channel_adjacency',
    'build_eeg_adjacency_from_geometry',
    'canonicalize_channel_label',
    'compute_fnirs_midpoint_diagnostics',
    'project_points_to_2d',
    'strip_fnirs_chromophore_suffix',
    'compute_per_channel_rms_envelope',
    'downsample_temporal_driver',
    'compute_spatial_fnirs_driver',
    'compute_cross_modal_correlation_matrix',
    'plot_channel_layout',
    'plot_channel_layout_3d',
    'plot_channel_layout_projected_2d',
    'plot_adjacency_heatmap',
    'plot_cross_modal_correlation_heatmap',
    'plot_spatial_target_preview',
]
