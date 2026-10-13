import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from src.data.channel_geometry import (
    records_from_mnt,
    records_from_refed_coordinates,
    records_from_refed_paper_layout,
    records_from_template_montage_csv,
    records_from_visual_fnirs_reference,
    records_from_visual_fnirs_graphical_projection,
)


def test_records_from_mnt_preserves_channel_coordinates_and_sd(tmp_path):
    path = tmp_path / "mnt.mat"
    savemat(
        path,
        {
            "mnt": {
                "clab": np.array(["AF7Fp1", "AF3Fp1"], dtype=object),
                "pos_3d": np.array([[-0.4, -0.2], [0.9, 0.8], [0.1, 0.2]], dtype=float),
                "sd": np.array([[2, 1], [3, 1]], dtype=np.uint8),
            }
        },
    )

    rows = records_from_mnt(
        path,
        dataset_id="eeg_fnirs_single_trial",
        subject="subject 01",
        record_id="global_nirs_mnt",
        modality="fnirs",
        channel_role="fnirs_channel_midpoint",
    )

    assert [row.channel_name for row in rows] == ["AF7Fp1", "AF3Fp1"]
    assert rows[0].x == -0.4
    assert rows[0].detector_index == 1
    assert rows[1].source_index == 3


def test_records_from_refed_coordinates_preserves_source_detector_indices(tmp_path):
    path = tmp_path / "fNIRS_coordinates.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Channel", "Source", "Detector", "X", "Y", "Z"])
        writer.writeheader()
        writer.writerow({"Channel": "1", "Source": "7", "Detector": "8", "X": "-1.5", "Y": "2.5", "Z": "3.5"})

    rows = records_from_refed_coordinates(path)

    assert len(rows) == 1
    assert rows[0].channel_name == "CH1"
    assert rows[0].source_index == 7
    assert rows[0].detector_index == 8
    assert rows[0].coordinate_system == "dataset_head_coordinates"


def _paper_layout_inputs(tmp_path):
    """Synthetic release metadata; tests never require the ignored dataset."""
    asset = Path(__file__).resolve().parents[1] / "src/data/assets/refed_paper_layout_v1.json"
    layout = json.loads(asset.read_text())
    eeg_path, hb_path = tmp_path / "eeg.csv", tmp_path / "hb.csv"
    with eeg_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ch_id", "ch_name"])
        writer.writerows((i, row["channel_name"]) for i, row in enumerate(layout["eeg"], 1))
    with hb_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Channel", "Source", "Detector", "X", "Y", "Z"])
        writer.writerows((r["channel"], r["source"], r["detector"], 1000, -2000, 3000)
                         for r in layout["channels"])
    return asset, eeg_path, hb_path


def test_refed_paper_layout_is_common_plane_not_native_xyz_or_reference_channel(tmp_path):
    rows = records_from_refed_paper_layout(*_paper_layout_inputs(tmp_path))
    eeg = {row.channel_name: row for row in rows if row.modality == "eeg"}
    hb = {row.channel_name: row for row in rows if row.modality == "fnirs"}
    assert len(eeg) == 64 and len(hb) == 51 and "AFZ" not in eeg
    assert eeg["FP1"].x < 0 < eeg["FP2"].x
    assert eeg["FPZ"].y > eeg["CZ"].y > eeg["OZ"].y
    assert hb["CH1"].x > 0 > hb["CH18"].x
    assert hb["CH48"].y > hb["CH38"].y > 0
    assert hb["CH1"].metadata["native_geometry"]["x"] == 1000
    assert hb["CH1"].metadata["native_geometry"]["coordinate_units"] == "unknown_native"
    assert all(row.z == 0 and abs(row.x) < 1.1 and abs(row.y) < 1.1 for row in rows)
    assert {row.coordinate_system for row in rows} == {"refed_paper_layout_2d_v1"}
    assert {row.coordinate_units for row in rows} == {"normalized_figure_radius"}
    assert all(row.metadata["spatial_dimension"] == 2
               and row.metadata["measured_subject_coordinate"] is False for row in rows)


@pytest.mark.parametrize("mismatch", ["eeg_order", "reference", "link", "duplicate_optode", "nonfinite"])
def test_refed_paper_layout_rejects_inconsistent_identity_or_coordinates(tmp_path, mismatch):
    asset, eeg_path, hb_path = _paper_layout_inputs(tmp_path)
    layout = json.loads(asset.read_text())
    if mismatch == "eeg_order":
        layout["eeg"][0], layout["eeg"][1] = layout["eeg"][1], layout["eeg"][0]
    elif mismatch == "reference":
        layout["reference"]["channel_name"] = "CZ"
    elif mismatch == "link":
        layout["channels"][0]["detector"] = 8
    elif mismatch == "duplicate_optode":
        layout["sources"][1]["index"] = layout["sources"][0]["index"]
    else:
        layout["detectors"][0]["center_page_pt"][0] = float("nan")
    changed = tmp_path / "layout.json"
    changed.write_text(json.dumps(layout))
    with pytest.raises(ValueError):
        records_from_refed_paper_layout(changed, eeg_path, hb_path)


def test_refed_paper_layout_builder_uses_explicit_version_without_replacing_legacy(tmp_path, monkeypatch):
    from experiments import build_clean_channel_geometry as builder
    asset, eeg_path, hb_path = _paper_layout_inputs(tmp_path)
    eeg_path.rename(tmp_path / "EEG_channels.csv")
    hb_path.rename(tmp_path / "fNIRS_coordinates.csv")
    monkeypatch.setitem(builder.DATA_ROOTS, "refed", tmp_path)
    monkeypatch.setattr(builder, "REFED_PAPER_LAYOUT_ASSET", asset)
    rows = list(builder.iter_records(["refed"], refed_geometry="paper_layout_v1"))
    assert len(rows) == 115
    assert all(row.metadata["geometry_version"] == "refed_paper_layout_v1" for row in rows)
    monkeypatch.setattr(builder, "refed_eeg_template_records", lambda _: [])
    legacy = list(builder.iter_records(["refed"]))
    assert len(legacy) == 51 and legacy[0].coordinate_system == "dataset_head_coordinates"


def test_refed_standard_montage_asset_matches_all_loader_channels_with_explicit_provenance():
    root = Path(__file__).resolve().parents[1]
    asset = root / "src/data/assets/refed_standard_1005_montage_v1.csv"
    rows = records_from_template_montage_csv(
        asset,
        dataset_id="refed",
        subject="all",
        record_id="global_eeg_standard_1005_v1",
        template_name="fieldtrip_standard_1005_refed_1010_subset",
        coordinate_system="fieldtrip_standard_1005_mni_template",
        coordinate_units="mm",
    )
    with (root / "data/REFED-dataset/EEG_channels.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        expected = [row["ch_name"] for row in csv.DictReader(handle)]

    assert [row.channel_name for row in rows] == expected
    assert len(rows) == 64
    assert sum(row.metadata["coordinate_status"] == "template_exact" for row in rows) == 62
    assert {
        row.channel_name: row.metadata["source_labels"]
        for row in rows
        if row.metadata["coordinate_status"] != "template_exact"
    } == {"CB1": ["PO7", "O1"], "CB2": ["PO8", "O2"]}
    assert all(row.metadata["measured_subject_coordinate"] is False for row in rows)
    assert all(row.metadata["intended_use"] == "channel_adjacency_and_visualization_only" for row in rows)


def test_visual_graphical_ced_projection_completes_both_24_channel_probes():
    root = Path(__file__).resolve().parents[1]
    data_root = (
        root
        / "data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults"
    )
    reference_rows = records_from_visual_fnirs_reference(
        data_root / "fNIRS_to_EEG_channel_reference.xlsx"
    )
    assert len(reference_rows) == 48
    assert {row.record_id for row in reference_rows} == {"Probe1", "Probe2"}
    rows = records_from_visual_fnirs_graphical_projection(
        data_root / "fNIRS_to_EEG_channel_reference.xlsx",
        data_root / "Location.ced",
        root / "src/data/assets/visual_fnirs_4x4_topology_v1.csv",
        graphical_model_path=data_root / "Graphical_recording_head_model.pdf",
    )

    assert len(rows) == 48
    for probe in ("Probe1", "Probe2"):
        probe_rows = [row for row in rows if row.record_id == probe]
        assert [row.channel_name for row in probe_rows] == [f"CH{index}" for index in range(1, 25)]
        assert sum(
            row.metadata["coordinate_status"] == "graphical_template_eeg_anchor"
            for row in probe_rows
        ) == 14
        assert sum(
            row.metadata["coordinate_status"] == "graphical_template_harmonic_interpolation"
            for row in probe_rows
        ) == 10
        assert np.isfinite([[row.x, row.y, row.z] for row in probe_rows]).all()
        assert all(row.metadata["measured_subject_coordinate"] is False for row in probe_rows)
        assert all(
            row.metadata["intended_use"]
            == "within_fnirs_adjacency_and_coarse_eeg_fnirs_alignment_only"
            for row in probe_rows
        )
    probe2_ch13 = next(
        row for row in rows if row.record_id == "Probe2" and row.channel_name == "CH13"
    )
    assert probe2_ch13.metadata["raw_nearest_eeg_label"] == "FP4"
    assert probe2_ch13.metadata["nearest_eeg_label"] == "FC4"
    assert probe2_ch13.metadata["reference_correction"] is not None
