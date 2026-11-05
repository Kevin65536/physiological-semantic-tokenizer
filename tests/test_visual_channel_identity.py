"""Synthetic publisher table and EDF; no local measured data are required."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xlrd

from src.data import unified_physiology as data


NAMES = ["Fp1", "Fp2", "AF3", "AF4", "F3", "F4", "FC3", "FC4", "C3", "C4",
         "CP3", "CP4", "P3", "P4", "PO3", "PO4", "O1", "O2", "F5", "F6",
         "F7", "F8", "T7", "T8", "P5", "P6", "P7", "P8", "Fz", "Cz", "FCz"]
RAW = [f"A{i}" for i in range(1, 31)] + ["A64"]


@pytest.fixture
def native_root(tmp_path, monkeypatch):
    root = tmp_path / "data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults"
    root.mkdir(parents=True)
    # Number deliberately disagrees with the amplifier ID. Join geometry by name.
    (root / "Location.ced").write_text("Number\tlabels\n" + "".join(
        f"{100-i}\t{name}\n" for i, name in enumerate(NAMES)))
    rows = [["Channel", "Name"], *[[raw, name.upper() if name.startswith("Fp") else name]
                                  for raw, name in zip(RAW, NAMES)], ["DC09", "Trigger"], ["E", "AFz"]]
    sheet = SimpleNamespace(ncols=2, nrows=len(rows), row_values=lambda i: rows[i])
    monkeypatch.setattr(xlrd, "open_workbook", lambda path: SimpleNamespace(sheet_by_index=lambda i: sheet))
    return tmp_path, root, rows


def write_edf(path, labels):
    n = len(labels)
    def field(value, width):
        return str(value).ljust(width).encode("ascii")
    fixed = (field("0", 8) + field("fixture", 80) + field("fixture", 80)
             + field("01.01.01", 8) + field("00.00.00", 8)
             + field(256 + 256*n, 8) + field("", 44) + field(2, 8)
             + field(1, 8) + field(n, 4))
    values = [labels, [""]*n, ["uV"]*n, [-200]*n, [200]*n, [-1000]*n,
              [1000]*n, [""]*n, [4]*n, [""]*n]
    widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    header = b"".join(field(v, width) for row, width in zip(values, widths) for v in row)
    digital = np.stack([np.arange(4) + i*10 for i in range(n)])
    body = np.tile(digital.reshape(-1), 2).astype("<i2").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fixed + header + body)


def test_visual_native_identity_uses_workbook_and_keeps_calibrated_fcz(native_root):
    project, root, _ = native_root
    path = root / "S01/EEG/raw/S01.edf"
    write_edf(path, RAW + ["DC09"])
    record = SimpleNamespace(canonical_subject_id="S01", base_record_id="S01_Probe1", join_key="fixture")
    eeg = data._visual_eeg(project, record)
    assert eeg.channel_names == tuple(NAMES)
    assert eeg.values.shape == (8, 31) and eeg.sample_rate_hz == 4
    assert eeg.channel_identity["schema"] == data.VISUAL_CHANNEL_IDENTITY_SCHEMA
    for channel in ("O1", "F5", "FCz"):
        slot = NAMES.index(channel)
        np.testing.assert_allclose(eeg.values[:4, slot], (np.arange(4) + slot*10)*.2)
    assert "DC09" not in eeg.channel_identity["raw_channels"]


def test_duplicate_or_incomplete_publisher_map_is_rejected(native_root):
    project, _, rows = native_root
    rows[2][0] = "A1"
    with pytest.raises(ValueError, match="Duplicate amplifier"):
        data._visual_eeg_name_map(project)


def test_visual_missing_declared_part_never_falls_back_to_other_record(native_root):
    project, root, _ = native_root
    write_edf(root / "S01/EEG/raw/S01_part1.edf", RAW + ["DC09"])
    record = SimpleNamespace(canonical_subject_id="S01", base_record_id="S01_Part2_Probe1", join_key="fixture")
    with pytest.raises(ValueError, match="Exactly one native EDF"):
        data._visual_eeg(project, record)


def test_historical_visual_cache_fails_before_any_array_read():
    dataset = data.UnifiedPhysiologyWindowDataset.__new__(data.UnifiedPhysiologyWindowDataset)
    dataset._record_cache = {}; dataset.output_coordinate = "measurement"
    dataset.index = SimpleNamespace(load_record_arrays=lambda _: pytest.fail("array opened"))
    record = SimpleNamespace(join_key="fixture", dataset_id="visual_cognitive_motivation", manifest=dict(
        storage="measurement_npy_record_v1", schema="clean_eeg_fnirs_cache_v2",
        processing_schema="physiology_measurement_alignment_v3",
        measurement={"eeg_preprocessing_state": {"signal_branch": "raw_with_ocular_artifact"}}))
    with pytest.raises(ValueError, match="invalid channel identity"):
        dataset._load_canonical_record(record)
