"""Channel geometry normalization for clean multimodal physiology caches."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.io import loadmat

from .clean_physiology_cache import base_record_id, canonical_subject_id


CHANNEL_GEOMETRY_SCHEMA = "physiology_channel_geometry_v1"


@dataclass(frozen=True)
class ChannelGeometryRecord:
    dataset_id: str
    subject: str
    record_id: str
    modality: str
    channel_name: str
    channel_role: str
    coordinate_system: str
    coordinate_units: str
    x: float | None = None
    y: float | None = None
    z: float | None = None
    source_index: int | None = None
    detector_index: int | None = None
    source_name: str | None = None
    detector_name: str | None = None
    source_file: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = CHANNEL_GEOMETRY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["canonical_subject_id"] = canonical_subject_id(self.dataset_id, self.subject)
        payload["base_record_id"] = base_record_id(self.dataset_id, self.record_id)
        payload["geometry_key"] = "|".join(
            (
                self.dataset_id,
                payload["canonical_subject_id"],
                payload["base_record_id"],
                self.modality,
                self.channel_name,
            )
        )
        payload["metadata"] = _jsonable(payload["metadata"])
        return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_list(value: Any) -> list[Any]:
    array = np.asarray(value, dtype=object)
    if array.ndim == 0:
        return [array.item()]
    return array.ravel().tolist()


def _labels(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value)]


def _coords_from_mnt(mnt: Any, index: int) -> tuple[float | None, float | None, float | None]:
    pos = getattr(mnt, "pos_3d", None)
    if pos is not None:
        arr = np.asarray(pos, dtype=np.float64)
        if arr.ndim == 2:
            if arr.shape[0] == 3 and index < arr.shape[1]:
                return float(arr[0, index]), float(arr[1, index]), float(arr[2, index])
            if arr.shape[1] == 3 and index < arr.shape[0]:
                return float(arr[index, 0]), float(arr[index, 1]), float(arr[index, 2])
    x = np.asarray(getattr(mnt, "x", []), dtype=np.float64).reshape(-1)
    y = np.asarray(getattr(mnt, "y", []), dtype=np.float64).reshape(-1)
    return (
        float(x[index]) if index < len(x) else None,
        float(y[index]) if index < len(y) else None,
        None,
    )


def _load_first_matstruct(path: Path) -> Any:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    key = next(name for name in payload if not name.startswith("__"))
    value = payload[key]
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        return value.item()
    return value


def records_from_mnt(
    path: Path,
    *,
    dataset_id: str,
    subject: str,
    record_id: str,
    modality: str,
    channel_role: str,
    coordinate_system: str = "dataset_mnt_pos_3d",
    coordinate_units: str = "normalized_head_unit",
    source_file: str | None = None,
) -> list[ChannelGeometryRecord]:
    mnt = _load_first_matstruct(path)
    labels = _labels(getattr(mnt, "clab", []))
    sd = np.asarray(getattr(mnt, "sd", np.empty((0, 2))), dtype=np.float64)
    rows = []
    for index, label in enumerate(labels):
        x, y, z = _coords_from_mnt(mnt, index)
        source_index = detector_index = None
        if sd.ndim == 2 and index < sd.shape[0] and sd.shape[1] >= 2:
            source_index = int(sd[index, 0])
            detector_index = int(sd[index, 1])
        rows.append(
            ChannelGeometryRecord(
                dataset_id=dataset_id,
                subject=subject,
                record_id=record_id,
                modality=modality,
                channel_name=label,
                channel_role=channel_role,
                coordinate_system=coordinate_system,
                coordinate_units=coordinate_units,
                x=x,
                y=y,
                z=z,
                source_index=source_index,
                detector_index=detector_index,
                source_file=source_file or str(path),
                metadata={"mnt_index": index},
            )
        )
    return rows


def records_from_refed_coordinates(path: Path, *, source_file: str | None = None) -> list[ChannelGeometryRecord]:
    rows: list[ChannelGeometryRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for item in csv.DictReader(handle):
            channel = str(item.get("Channel", "")).strip()
            rows.append(
                ChannelGeometryRecord(
                    dataset_id="refed",
                    subject="all",
                    record_id="global_fnirs_coordinates",
                    modality="fnirs",
                    channel_name=f"CH{int(float(channel))}" if channel else "",
                    channel_role="fnirs_channel_midpoint",
                    coordinate_system="dataset_head_coordinates",
                    coordinate_units="unknown_native",
                    x=_float_or_none(item.get("X")),
                    y=_float_or_none(item.get("Y")),
                    z=_float_or_none(item.get("Z")),
                    source_index=_int_or_none(item.get("Source")),
                    detector_index=_int_or_none(item.get("Detector")),
                    source_file=source_file or str(path),
                    metadata={"raw_channel": channel},
                )
            )
    return rows


def records_from_template_montage_csv(
    path: Path,
    *,
    dataset_id: str,
    subject: str,
    record_id: str,
    template_name: str,
    coordinate_system: str,
    coordinate_units: str,
    source_file: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> list[ChannelGeometryRecord]:
    """Load versioned template EEG coordinates without claiming digitization.

    Template coordinates are suitable for deterministic channel topology and
    visualization.  They remain distinct from participant-specific electrode
    digitization and must not be promoted to measured co-registration.
    """

    rows: list[ChannelGeometryRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, item in enumerate(csv.DictReader(handle)):
            channel_name = str(item.get("channel_name", "")).strip()
            if not channel_name:
                raise ValueError(f"template montage row {index} has no channel_name: {path}")
            coordinates = tuple(_float_or_none(item.get(axis)) for axis in ("x_mm", "y_mm", "z_mm"))
            if any(value is None for value in coordinates):
                raise ValueError(
                    f"template montage channel {channel_name!r} has incomplete coordinates: {path}"
                )
            metadata = {
                "template_name": template_name,
                "coordinate_status": str(item.get("coordinate_status", "template_exact")),
                "source_labels": [
                    value for value in str(item.get("source_labels", channel_name)).split("|") if value
                ],
                "intended_use": "channel_adjacency_and_visualization_only",
                "measured_subject_coordinate": False,
            }
            metadata.update(dict(provenance or {}))
            rows.append(
                ChannelGeometryRecord(
                    dataset_id=dataset_id,
                    subject=subject,
                    record_id=record_id,
                    modality="eeg",
                    channel_name=channel_name,
                    channel_role="template_eeg_electrode",
                    coordinate_system=coordinate_system,
                    coordinate_units=coordinate_units,
                    x=coordinates[0],
                    y=coordinates[1],
                    z=coordinates[2],
                    source_file=source_file or str(path),
                    metadata=metadata,
                )
            )
    return rows


def records_from_visual_ced(path: Path, *, source_file: str | None = None) -> list[ChannelGeometryRecord]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    header = [item.strip() for item in lines[0].split("\t") if item.strip()]
    output: list[ChannelGeometryRecord] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = [item.strip() for item in line.split("\t")]
        row = {header[index]: values[index] if index < len(values) else "" for index in range(len(header))}
        label = row.get("labels", "")
        if not label:
            continue
        output.append(
            ChannelGeometryRecord(
                dataset_id="visual_cognitive_motivation",
                subject="all",
                record_id="global_eeg_location_ced",
                modality="eeg",
                channel_name=label,
                channel_role="eeg_electrode",
                coordinate_system="EEGLAB_CED_cartesian",
                coordinate_units="normalized_head_unit",
                x=_float_or_none(row.get("X")),
                y=_float_or_none(row.get("Y")),
                z=_float_or_none(row.get("Z")),
                source_file=source_file or str(path),
                metadata={key: value for key, value in row.items() if key not in {"labels", "X", "Y", "Z"}},
            )
        )
    return output


def records_from_visual_fnirs_reference(path: Path, *, source_file: str | None = None) -> list[ChannelGeometryRecord]:
    raw_rows = _read_xlsx_matrix(path)
    output: list[ChannelGeometryRecord] = []
    for index, values in enumerate(raw_rows[2:]):
        pairs = [
            ("Probe1", values[0] if len(values) > 0 else "", values[1] if len(values) > 1 else ""),
            ("Probe2", values[3] if len(values) > 3 else "", values[4] if len(values) > 4 else ""),
        ]
        for probe, fnirs_channel, eeg_label in pairs:
            fnirs_channel = str(fnirs_channel).strip()
            if not re.fullmatch(r"CH\d+", fnirs_channel):
                continue
            output.append(
                ChannelGeometryRecord(
                    dataset_id="visual_cognitive_motivation",
                    subject="all",
                    record_id=probe,
                    modality="fnirs",
                    channel_name=fnirs_channel,
                    channel_role="fnirs_channel_to_eeg_reference",
                    coordinate_system="referenced_to_visual_eeg_ced",
                    coordinate_units="label_reference",
                    source_file=source_file or str(path),
                    metadata={"nearest_eeg_label": str(eeg_label).strip(), "row_index": index},
                )
            )
    return output


def records_from_visual_fnirs_graphical_projection(
    reference_path: Path,
    ced_path: Path,
    topology_path: Path,
    *,
    graphical_model_path: Path,
    source_files: Mapping[str, str] | None = None,
) -> list[ChannelGeometryRecord]:
    """Project all Visual fNIRS channels onto the dataset EEG head template.

    The dataset supplies a bilateral optode-layout figure, a partial table of
    fNIRS-channel-to-EEG anchors, and global EEGLAB CED coordinates.  Anchored
    channels inherit the referenced CED position.  Unlabelled channels are
    filled by harmonic interpolation on the 24-channel line graph induced by
    the documented 4x4 optode layout, then projected back to the CED head
    radius.  These are template/reference coordinates for topology and coarse
    alignment, not participant digitization or measured source-detector 3D.
    """

    ced_rows = records_from_visual_ced(ced_path)
    ced_lookup = {row.channel_name.casefold(): row for row in ced_rows}
    topology = _read_visual_topology(topology_path)
    expected_channels = [f"CH{index}" for index in range(1, 25)]
    if list(topology) != expected_channels:
        raise ValueError(
            "Visual 4x4 topology must contain CH1..CH24 in acquisition order: "
            f"{topology_path}"
        )

    reference_matrix = _read_xlsx_matrix(reference_path)
    if len(reference_matrix) < 26:
        raise ValueError(f"Visual fNIRS reference table is incomplete: {reference_path}")
    anchors: dict[str, dict[str, str]] = {"Probe1": {}, "Probe2": {}}
    raw_anchors: dict[str, dict[str, str]] = {"Probe1": {}, "Probe2": {}}
    for row in reference_matrix[2:]:
        for probe, channel_index, label_index in (
            ("Probe1", 0, 1),
            ("Probe2", 3, 4),
        ):
            channel = str(row[channel_index]).strip() if len(row) > channel_index else ""
            label = str(row[label_index]).strip() if len(row) > label_index else ""
            if not re.fullmatch(r"CH\d+", channel):
                continue
            raw_anchors[probe][channel] = label
            if label and label != "-":
                anchors[probe][channel] = label

    # The distributed workbook contains "FP4" for Probe2 CH13, while the
    # bilateral counterpart, CED file, and graphical model all use FC4.
    if anchors["Probe2"].get("CH13", "").casefold() == "fp4" and "fp4" not in ced_lookup:
        anchors["Probe2"]["CH13"] = "FC4"

    graph = _visual_channel_graph(topology)
    source_names = {
        "graphical_model": str(graphical_model_path),
        "channel_reference": str(reference_path),
        "eeg_ced": str(ced_path),
        "topology_asset": str(topology_path),
    }
    source_names.update(dict(source_files or {}))
    output: list[ChannelGeometryRecord] = []
    for probe in ("Probe1", "Probe2"):
        anchor_positions: dict[str, np.ndarray] = {}
        for channel, label in anchors[probe].items():
            ced = ced_lookup.get(label.casefold())
            if ced is None or any(value is None for value in (ced.x, ced.y, ced.z)):
                raise ValueError(
                    f"Visual {probe} {channel} references missing CED label {label!r}"
                )
            anchor_positions[channel] = np.asarray([ced.x, ced.y, ced.z], dtype=np.float64)
        positions = _harmonic_channel_positions(graph, expected_channels, anchor_positions)

        for channel in expected_channels:
            raw_label = raw_anchors[probe].get(channel, "")
            resolved_label = anchors[probe].get(channel, "")
            is_anchor = channel in anchor_positions
            correction = None
            if raw_label.casefold() == "fp4" and resolved_label == "FC4":
                correction = "FP4_to_FC4_bilateral_mirror_and_graphical_model_v1"
            metadata = {
                "coordinate_status": (
                    "graphical_template_eeg_anchor"
                    if is_anchor
                    else "graphical_template_harmonic_interpolation"
                ),
                "raw_nearest_eeg_label": raw_label,
                "nearest_eeg_label": resolved_label,
                "reference_correction": correction,
                "grid_x": topology[channel]["grid_x"],
                "grid_y": topology[channel]["grid_y"],
                "optode_endpoints": [topology[channel]["optode_a"], topology[channel]["optode_b"]],
                "graph_neighbors": sorted(graph[channel], key=_channel_number),
                "interpolation_method": None if is_anchor else "graph_laplacian_harmonic_ced_projection_v1",
                "anchor_channel_count": len(anchor_positions),
                "source_files": source_names,
                "measured_subject_coordinate": False,
                "intended_use": "within_fnirs_adjacency_and_coarse_eeg_fnirs_alignment_only",
                "prohibited_interpretation": (
                    "participant_digitization_source_detector_distance_or_exact_coregistration"
                ),
            }
            position = positions[channel]
            output.append(
                ChannelGeometryRecord(
                    dataset_id="visual_cognitive_motivation",
                    subject="all",
                    record_id=probe,
                    modality="fnirs",
                    channel_name=channel,
                    channel_role="fnirs_channel_midpoint_graphical_template_proxy",
                    coordinate_system="visual_eeg_ced_graphical_template_projection_v1",
                    coordinate_units="normalized_head_unit",
                    x=float(position[0]),
                    y=float(position[1]),
                    z=float(position[2]),
                    source_file=source_names["graphical_model"],
                    metadata=metadata,
                )
            )
    return output


def _read_xlsx_matrix(path: Path) -> list[list[str]]:
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", namespace):
                shared.append("".join(text.text or "" for text in item.findall(".//a:t", namespace)))
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in root.findall(".//a:row", namespace):
            values: dict[int, str] = {}
            for cell in row.findall("a:c", namespace):
                column = _excel_column_index(cell.get("r", ""))
                value_node = cell.find("a:v", namespace)
                value = "" if value_node is None else str(value_node.text or "")
                if cell.get("t") == "s" and value:
                    value = shared[int(value)]
                values[column] = value
            if values:
                rows.append([values.get(index, "") for index in range(max(values) + 1)])
    return rows


def _excel_column_index(cell_reference: str) -> int:
    column = "".join(character for character in cell_reference if character.isalpha())
    value = 0
    for character in column:
        value = value * 26 + (ord(character.upper()) - ord("A") + 1)
    return max(value - 1, 0)


def _read_visual_topology(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            channel = str(row.get("channel_name", "")).strip()
            if not channel:
                continue
            output[channel] = {
                "optode_a": str(row.get("optode_a", "")).strip(),
                "optode_b": str(row.get("optode_b", "")).strip(),
                "grid_x": float(row["grid_x"]),
                "grid_y": float(row["grid_y"]),
            }
    return output


def _visual_channel_graph(
    topology: Mapping[str, Mapping[str, Any]],
) -> dict[str, set[str]]:
    graph = {channel: set() for channel in topology}
    channels = list(topology)
    for left_index, left in enumerate(channels):
        left_optodes = {topology[left]["optode_a"], topology[left]["optode_b"]}
        for right in channels[left_index + 1 :]:
            right_optodes = {topology[right]["optode_a"], topology[right]["optode_b"]}
            if left_optodes & right_optodes:
                graph[left].add(right)
                graph[right].add(left)
    return graph


def _harmonic_channel_positions(
    graph: Mapping[str, set[str]],
    channel_names: list[str],
    anchors: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not anchors:
        raise ValueError("Visual fNIRS geometry interpolation requires at least one CED anchor")
    index = {channel: offset for offset, channel in enumerate(channel_names)}
    adjacency = np.zeros((len(channel_names), len(channel_names)), dtype=np.float64)
    for channel, neighbors in graph.items():
        for neighbor in neighbors:
            adjacency[index[channel], index[neighbor]] = 1.0
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    known = [index[channel] for channel in channel_names if channel in anchors]
    unknown = [index[channel] for channel in channel_names if channel not in anchors]
    values = np.zeros((len(channel_names), 3), dtype=np.float64)
    for channel, position in anchors.items():
        values[index[channel]] = position
    if unknown:
        values[unknown] = np.linalg.solve(
            laplacian[np.ix_(unknown, unknown)],
            -laplacian[np.ix_(unknown, known)] @ values[known],
        )
        target_radius = float(np.median(np.linalg.norm(values[known], axis=1)))
        for row_index in unknown:
            radius = float(np.linalg.norm(values[row_index]))
            if radius <= 0:
                raise ValueError("Visual harmonic projection produced a zero-radius coordinate")
            values[row_index] *= target_radius / radius
    return {channel: values[index[channel]] for channel in channel_names}


def _channel_number(channel_name: str) -> int:
    match = re.search(r"\d+", channel_name)
    return int(match.group()) if match else 0


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    parsed = _float_or_none(value)
    return int(parsed) if parsed is not None else None
