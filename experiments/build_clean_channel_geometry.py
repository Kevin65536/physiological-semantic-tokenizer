#!/usr/bin/env python3
"""Build a normalized channel-geometry sidecar for the clean physiology cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.channel_geometry import (  # noqa: E402
    CHANNEL_GEOMETRY_SCHEMA,
    ChannelGeometryRecord,
    records_from_mnt,
    records_from_refed_coordinates,
    records_from_template_montage_csv,
    records_from_visual_ced,
    records_from_visual_fnirs_graphical_projection,
)
from src.data.channel_adjacency import (  # noqa: E402
    build_eeg_adjacency_from_geometry,
    build_fnirs_adjacency_from_shared_optodes,
)
from src.utils.io import write_json  # noqa: E402


DATA_ROOTS = {
    "eeg_fnirs_single_trial": PROJECT_ROOT / "data/EEG+NIRS Single-Trial",
    "refed": PROJECT_ROOT / "data/REFED-dataset",
    "visual_cognitive_motivation": PROJECT_ROOT / "data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults",
    "simultaneous_eeg_nirs": PROJECT_ROOT / "data/Simultaneous EEG&NIRS",
}
REFED_EEG_TEMPLATE_ASSET = PROJECT_ROOT / "src/data/assets/refed_standard_1005_montage_v1.csv"
VISUAL_FNIRS_TOPOLOGY_ASSET = PROJECT_ROOT / "src/data/assets/visual_fnirs_4x4_topology_v1.csv"
REFED_EEG_TEMPLATE_PROVENANCE = {
    "schema": "refed_standard_montage_geometry_v1",
    "template_name": "fieldtrip_standard_1005_refed_1010_subset",
    "upstream_repository": "https://github.com/fieldtrip/fieldtrip",
    "upstream_commit": "462487e4dd6dd1c4caba626723d5650919b31e86",
    "upstream_file": "template/electrode/standard_1005.elc",
    "upstream_file_sha256": "1ee59197946d62de872db2ac7f2243a596662c231427366f6dc5d84ed237f853",
    "exact_template_channel_count": 62,
    "interpolated_channel_count": 2,
    "interpolated_channels": {
        "CB1": ["PO7", "O1"],
        "CB2": ["PO8", "O2"],
    },
    "interpolation_rule": "arithmetic_3d_midpoint_v1",
    "intended_use": "within_eeg_channel_adjacency_and_visualization_only",
    "prohibited_interpretation": "participant_digitization_or_eeg_fnirs_coregistration",
}
VISUAL_FNIRS_TEMPLATE_PROVENANCE = {
    "schema": "visual_fnirs_graphical_ced_projection_v1",
    "layout": "bilateral Hitachi 4x4 optode grids with 24 channels per probe",
    "channel_numbering_reference": {
        "paper": "Iso et al. 2021, Figure 2",
        "doi": "10.3389/fnhum.2021.603069",
        "scope": "standard Hitachi 4x4 channel index topology only",
        "not_used_as": "dataset coordinate or participant placement source",
    },
    "channel_topology_method": "shared_optode_line_graph_v1",
    "coordinate_projection_method": "graph_laplacian_harmonic_ced_projection_v1",
    "probe1_anchor_channel_count": 14,
    "probe2_anchor_channel_count": 14,
    "interpolated_channel_count_per_probe": 10,
    "workbook_correction": {
        "probe": "Probe2",
        "channel": "CH13",
        "raw_label": "FP4",
        "resolved_label": "FC4",
        "basis": "bilateral FC3/FC4 symmetry, Location.ced, and graphical head model",
    },
    "intended_use": "within_fnirs_adjacency_and_coarse_eeg_fnirs_alignment_only",
    "prohibited_interpretation": (
        "participant_digitization_source_detector_distance_or_exact_coregistration"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATA_ROOTS), choices=list(DATA_ROOTS))
    from src.data.clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT
    parser.add_argument("--output-dir", default=f"{DEFAULT_CLEAN_CACHE_ROOT}/channel_geometry")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def _subject_id_from_name(name: str) -> int:
    return int(name.split()[-1])


def iter_single_trial(root: Path) -> Iterable[ChannelGeometryRecord]:
    for subject_dir in sorted((root / "NIRS_01-29").glob("subject *"), key=lambda item: _subject_id_from_name(item.name)):
        if _subject_id_from_name(subject_dir.name) not in range(1,30):
            continue
        mnt = subject_dir / "mnt.mat"
        if mnt.exists():
            yield from records_from_mnt(
                mnt,
                dataset_id="eeg_fnirs_single_trial",
                subject=subject_dir.name,
                record_id="global_nirs_mnt",
                modality="fnirs",
                channel_role="fnirs_channel_midpoint",
                source_file=_rel(mnt),
            )
    for subject_dir in sorted((root / "EEG_01-29").glob("subject *"), key=lambda item: _subject_id_from_name(item.name)):
        if _subject_id_from_name(subject_dir.name) not in range(1,30):
            continue
        for name in ("mnt.mat", "mnt_artifact.mat"):
            mnt = subject_dir / name
            if mnt.exists():
                yield from records_from_mnt(
                    mnt,
                    dataset_id="eeg_fnirs_single_trial",
                    subject=subject_dir.name,
                    record_id=name.replace(".mat", ""),
                    modality="eeg",
                    channel_role="eeg_electrode",
                    source_file=_rel(mnt),
                )
                break


def iter_simultaneous(root: Path) -> Iterable[ChannelGeometryRecord]:
    for subject_dir in sorted(root.glob("VP*-EEG")):
        subject = subject_dir.name.split("-")[0]
        for mnt in sorted(subject_dir.glob("mnt_*.mat")):
            yield from records_from_mnt(
                mnt,
                dataset_id="simultaneous_eeg_nirs",
                subject=subject,
                record_id=mnt.stem.replace("mnt_", "cnt_"),
                modality="eeg",
                channel_role="eeg_electrode",
                source_file=_rel(mnt),
            )
    for subject_dir in sorted(root.glob("VP*-NIRS")):
        subject = subject_dir.name.split("-")[0]
        for mnt in sorted(subject_dir.glob("mnt_*.mat")):
            yield from records_from_mnt(
                mnt,
                dataset_id="simultaneous_eeg_nirs",
                subject=subject,
                record_id=mnt.stem.replace("mnt_", "cnt_"),
                modality="fnirs",
                channel_role="fnirs_channel_midpoint",
                source_file=_rel(mnt),
            )


def refed_eeg_template_records(root: Path) -> list[ChannelGeometryRecord]:
    channel_file = root / "EEG_channels.csv"
    with channel_file.open("r", encoding="utf-8-sig", newline="") as handle:
        expected_names = [str(row.get("ch_name", "")).strip() for row in csv.DictReader(handle)]
    provenance = {
        **REFED_EEG_TEMPLATE_PROVENANCE,
        "refed_channel_file": _rel(channel_file),
        "refed_channel_file_sha256": _sha256(channel_file),
        "refed_reference_figure": _rel(root / "Figures/Figure_1.png"),
        "refed_reference_figure_sha256": _sha256(root / "Figures/Figure_1.png"),
    }
    rows = records_from_template_montage_csv(
        REFED_EEG_TEMPLATE_ASSET,
        dataset_id="refed",
        subject="all",
        record_id="global_eeg_standard_1005_v1",
        template_name=REFED_EEG_TEMPLATE_PROVENANCE["template_name"],
        coordinate_system="fieldtrip_standard_1005_mni_template",
        coordinate_units="mm",
        source_file=_rel(REFED_EEG_TEMPLATE_ASSET),
        provenance=provenance,
    )
    actual_names = [row.channel_name for row in rows]
    if actual_names != expected_names:
        raise ValueError(
            "REFED standard montage asset must exactly match EEG_channels.csv order: "
            f"expected={expected_names}, actual={actual_names}"
        )
    return rows


def _visual_mode_4x4_files(root: Path) -> list[Path]:
    files = sorted(root.glob("S*/fNIRS/*.csv"))
    invalid: list[str] = []
    for path in files:
        mode = None
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for _, line in zip(range(40), handle):
                if line.startswith("Mode,"):
                    mode = line.strip().split(",", 1)[1]
                    break
        if mode != "4x4":
            invalid.append(f"{_rel(path)}={mode!r}")
    if not files or invalid:
        raise ValueError(
            "Visual fNIRS topology requires every raw export to declare Mode,4x4; "
            f"file_count={len(files)}, invalid={invalid[:5]}"
        )
    return files


def iter_visual(root: Path) -> Iterable[ChannelGeometryRecord]:
    _visual_mode_4x4_files(root)
    ced = root / "Location.ced"
    if ced.exists():
        yield from records_from_visual_ced(ced, source_file=_rel(ced))

    reference = root / "fNIRS_to_EEG_channel_reference.xlsx"
    graphical_model = root / "Graphical_recording_head_model.pdf"
    if reference.exists() and ced.exists() and graphical_model.exists():
        yield from records_from_visual_fnirs_graphical_projection(
            reference,
            ced,
            VISUAL_FNIRS_TOPOLOGY_ASSET,
            graphical_model_path=graphical_model,
            source_files={
                "graphical_model": _rel(graphical_model),
                "channel_reference": _rel(reference),
                "eeg_ced": _rel(ced),
                "topology_asset": _rel(VISUAL_FNIRS_TOPOLOGY_ASSET),
            },
        )


def iter_records(datasets: list[str]) -> Iterable[ChannelGeometryRecord]:
    if "eeg_fnirs_single_trial" in datasets:
        yield from iter_single_trial(DATA_ROOTS["eeg_fnirs_single_trial"])
    if "refed" in datasets:
        yield from refed_eeg_template_records(DATA_ROOTS["refed"])
        path = DATA_ROOTS["refed"] / "fNIRS_coordinates.csv"
        if path.exists():
            yield from records_from_refed_coordinates(path, source_file=_rel(path))
    if "visual_cognitive_motivation" in datasets:
        yield from iter_visual(DATA_ROOTS["visual_cognitive_motivation"])
    if "simultaneous_eeg_nirs" in datasets:
        yield from iter_simultaneous(DATA_ROOTS["simultaneous_eeg_nirs"])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output directory exists; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [record.to_dict() for record in iter_records(args.datasets)]
    channels_path = output_dir / "channels.jsonl"
    count = _write_jsonl(channels_path, rows)
    files = {"channels_jsonl": _rel(channels_path)}
    if "refed" in args.datasets:
        refed_eeg_rows = [
            {
                **row,
                "coordinate_status": row.get("metadata", {}).get("coordinate_status"),
            }
            for row in rows
            if row["dataset_id"] == "refed" and row["modality"] == "eeg"
        ]
        adjacency = build_eeg_adjacency_from_geometry(
            "refed",
            refed_eeg_rows,
            [row["channel_name"] for row in refed_eeg_rows],
        )
        adjacency_path = output_dir / "refed_eeg_adjacency.json"
        write_json(adjacency_path, adjacency.to_serializable(), ensure_ascii=False)
        files["refed_eeg_adjacency"] = _rel(adjacency_path)
    if "visual_cognitive_motivation" in args.datasets:
        visual_probes: dict[str, Any] = {}
        for probe in ("Probe1", "Probe2"):
            probe_rows = [
                row
                for row in rows
                if row["dataset_id"] == "visual_cognitive_motivation"
                and row["modality"] == "fnirs"
                and row["base_record_id"] == probe
            ]
            adjacency = build_fnirs_adjacency_from_shared_optodes(
                "visual_cognitive_motivation",
                probe,
                probe_rows,
                [f"CH{index}" for index in range(1, 25)],
            )
            visual_probes[probe] = adjacency.to_serializable()
        visual_adjacency_path = output_dir / "visual_fnirs_adjacency.json"
        write_json(
            visual_adjacency_path,
            {
                "schema": "visual_fnirs_bilateral_adjacency_v1",
                "dataset_id": "visual_cognitive_motivation",
                "probes": visual_probes,
                "claim_boundary": VISUAL_FNIRS_TEMPLATE_PROVENANCE["prohibited_interpretation"],
            },
            ensure_ascii=False,
        )
        files["visual_fnirs_adjacency"] = _rel(visual_adjacency_path)
    counts: dict[str, int] = {}
    modality_counts: dict[str, int] = {}
    for row in rows:
        counts[row["dataset_id"]] = counts.get(row["dataset_id"], 0) + 1
        key = f'{row["dataset_id"]}:{row["modality"]}'
        modality_counts[key] = modality_counts.get(key, 0) + 1
    manifest = {
        "schema": CHANNEL_GEOMETRY_SCHEMA,
        "code_sha256": {
            "builder": _sha256(Path(__file__).resolve()),
            "channel_geometry": _sha256(PROJECT_ROOT / "src/data/channel_geometry.py"),
            "channel_adjacency": _sha256(PROJECT_ROOT / "src/data/channel_adjacency.py"),
        },
        "files": files,
        "datasets": args.datasets,
        "record_count": count,
        "counts": counts,
        "modality_counts": modality_counts,
        "refed_eeg_template_provenance": {
            **REFED_EEG_TEMPLATE_PROVENANCE,
            "asset": _rel(REFED_EEG_TEMPLATE_ASSET),
            "asset_sha256": _sha256(REFED_EEG_TEMPLATE_ASSET),
            "refed_channel_file_sha256": _sha256(DATA_ROOTS["refed"] / "EEG_channels.csv"),
            "refed_reference_figure_sha256": _sha256(
                DATA_ROOTS["refed"] / "Figures/Figure_1.png"
            ),
            "adjacency_method": "top_view_xy_delaunay_v1",
        } if "refed" in args.datasets else None,
        "visual_fnirs_template_provenance": {
            **VISUAL_FNIRS_TEMPLATE_PROVENANCE,
            "topology_asset": _rel(VISUAL_FNIRS_TOPOLOGY_ASSET),
            "topology_asset_sha256": _sha256(VISUAL_FNIRS_TOPOLOGY_ASSET),
            "graphical_model_sha256": _sha256(
                DATA_ROOTS["visual_cognitive_motivation"] / "Graphical_recording_head_model.pdf"
            ),
            "channel_reference_sha256": _sha256(
                DATA_ROOTS["visual_cognitive_motivation"] / "fNIRS_to_EEG_channel_reference.xlsx"
            ),
            "eeg_ced_sha256": _sha256(
                DATA_ROOTS["visual_cognitive_motivation"] / "Location.ced"
            ),
            "raw_export_mode": "4x4",
            "raw_export_file_count": len(
                _visual_mode_4x4_files(DATA_ROOTS["visual_cognitive_motivation"])
            ),
        } if "visual_cognitive_motivation" in args.datasets else None,
        "notes": [
            "Coordinates preserve each dataset's native coordinate system and units.",
            "Visual fNIRS coordinates are graphical/CED template projections, not measured 3D optode coordinates.",
            "Visual projected coordinates are valid for within-fNIRS adjacency and coarse cross-modal alignment only.",
            "REFED EEG coordinates are a standard-template topology proxy, not participant digitization.",
            "REFED EEG adjacency is valid only as within-EEG neighborhood information.",
        ],
    }
    write_json(output_dir / "geometry_manifest.json", manifest, ensure_ascii=False)
    print(json.dumps({"output_dir": str(output_dir), "record_count": count, "counts": counts}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
