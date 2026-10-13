"""Extract the versioned public REFED Figure 2 layout; never reads signals.

Run from the repository root using .venv/bin/python and the command retained
in the owning geometry audit. Extraction precedes CSV comparison:
the CSV only supplies canonical EEG ordering and independently checked topology.
Every output is exclusive-created to preserve retained evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex

import fitz
import numpy as np

URL = "https://proceedings.neurips.cc/paper_files/paper/2025/file/2bf0ed7c35d9d84128e7f7c72ab76402-Paper-Datasets_and_Benchmarks_Track.pdf"
COLORS = {
    "eeg": (0.0, 0.439, 0.753),
    "reference": (0.773, 0.878, 0.706),
    "source": (1.0, 0.0, 0.0),
    "detector": (0.0, 0.69, 0.314),
    "channel": (1.0, 0.753, 0.0),
}


def center(rect):
    return np.asarray([(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2])


def matches_color(drawing, name):
    fill = drawing.get("fill")
    return fill is not None and np.max(np.abs(np.asarray(fill) - COLORS[name])) < 1e-6


def in_panel(rect, panel):
    x0, x1, y0, y1 = panel
    return x0 < rect[0] and rect[2] < x1 and y0 < rect[1] and rect[3] < y1


def extract_dots(drawings, words, panel, role):
    """Read the circle label using its nearest text bounding-box midpoint."""
    labels = [
        w for w in words
        if in_panel(w[:4], panel)
        and (w[4].isdigit() if role in {"source", "detector"} else not w[4].isdigit())
        and (w[3] - w[1] < 2.1 if panel[0] > 330 else True)
    ]
    dots, distances = [], []
    for index, drawing in enumerate(drawings):
        if not in_panel(drawing["rect"], panel) or not matches_color(drawing, role):
            continue
        assert len(drawing["items"]) == 4 and all(item[0] == "c" for item in drawing["items"])
        xy = center(drawing["rect"])
        label = min(labels, key=lambda w: np.linalg.norm(center(w[:4]) - xy))
        distance = float(np.linalg.norm(center(label[:4]) - xy))
        assert distance < 0.3, (role, index, label, distance)
        distances.append(distance)
        dots.append({
            "index" if role in {"source", "detector"} else "channel_name":
                int(label[4]) if role in {"source", "detector"} else label[4].upper(),
            "center_page_pt": xy.tolist(),
            "drawing_index": index,
        })
    names = [row["index" if role in {"source", "detector"} else "channel_name"] for row in dots]
    assert len(set(names)) == len(dots)
    return dots, max(distances)


def segment_endpoints(drawing):
    """Midpoints of the two short sides of a thin yellow quadrilateral."""
    items = drawing["items"]
    if len(items) == 1 and items[0][0] == "re":
        rect = items[0][1]
        if rect.width > rect.height:
            return np.asarray([[rect.x0, (rect.y0 + rect.y1) / 2], [rect.x1, (rect.y0 + rect.y1) / 2]])
        return np.asarray([[(rect.x0 + rect.x1) / 2, rect.y0], [(rect.x0 + rect.x1) / 2, rect.y1]])
    assert len(items) == 4 and all(item[0] == "l" for item in items)
    vertices = np.asarray([[item[1].x, item[1].y] for item in items])
    lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
    short = np.argsort(lengths)[:2]
    assert (short[0] - short[1]) % 2 == 0
    return np.asarray([(vertices[i] + vertices[(i + 1) % 4]) / 2 for i in short])


def write_exclusive(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--eeg-metadata", type=Path, required=True)
    parser.add_argument("--fnirs-metadata", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    assert not args.asset.exists() and not args.audit.exists()
    document = fitz.open(args.pdf)
    page = document[4]
    drawings, words = page.get_drawings(), page.get_text("words")
    left, inset = (166, 330, 70, 225), (340, 444, 70, 165)
    eeg, eeg_distance = extract_dots(drawings, words, left, "eeg")
    references, reference_distance = extract_dots(drawings, words, left, "reference")
    sources, source_distance = extract_dots(drawings, words, left, "source")
    detectors, detector_distance = extract_dots(drawings, words, left, "detector")
    assert len(eeg) == 64 and len(references) == 1
    assert references[0]["channel_name"] == "AFZ"
    assert len(sources) == len(detectors) == 18
    assert {row["index"] for row in sources} == {row["index"] for row in detectors} == set(range(1, 19))
    heads = [
        (i, drawing) for i, drawing in enumerate(drawings)
        if in_panel(drawing["rect"], left)
        and drawing.get("color") == (0, 0, 0)
        and len(drawing["items"]) == 4
        and all(item[0] == "c" for item in drawing["items"])
        and drawing["rect"].width > 100 and drawing["rect"].height > 100
    ]
    assert len(heads) == 1
    head_index, head = heads[0]
    rect = head["rect"]
    assert abs(rect.width - rect.height) < 0.001
    optodes = []
    for role in ("source", "detector"):
        rows, _ = extract_dots(drawings, words, inset, role)
        assert len(rows) == 18
        optodes.extend([{**row, "role": role} for row in rows])
    segments = [
        (i, segment_endpoints(drawing)) for i, drawing in enumerate(drawings)
        if in_panel(drawing["rect"], inset) and matches_color(drawing, "channel")
    ]
    labels = [w for w in words if in_panel(w[:4], inset) and w[4].isdigit() and w[3] - w[1] > 3]
    assert len(segments) == len(labels) == 51
    assert {int(w[4]) for w in labels} == set(range(1, 52))
    channels, used_segments, label_distances, endpoint_distances = [], set(), [], []
    for label in labels:
        label_center = center(label[:4])
        segment_index, endpoints = min(segments, key=lambda pair: np.linalg.norm(label_center - pair[1].mean(axis=0)))
        assert segment_index not in used_segments
        used_segments.add(segment_index)
        label_distance = float(np.linalg.norm(label_center - endpoints.mean(axis=0)))
        assert label_distance < 1.0
        label_distances.append(label_distance)
        attached = []
        for endpoint in endpoints:
            optode = min(optodes, key=lambda row: np.linalg.norm(np.asarray(row["center_page_pt"]) - endpoint))
            distance = float(np.linalg.norm(np.asarray(optode["center_page_pt"]) - endpoint))
            assert distance < 0.75
            attached.append(optode)
            endpoint_distances.append(distance)
        assert {row["role"] for row in attached} == {"source", "detector"}
        channels.append({
            "channel": int(label[4]),
            "source": next(row["index"] for row in attached if row["role"] == "source"),
            "detector": next(row["index"] for row in attached if row["role"] == "detector"),
            "inset_drawing_index": segment_index,
        })
    channels.sort(key=lambda row: row["channel"])

    # Public metadata is read only after the paper topology has been extracted.
    with args.eeg_metadata.open(encoding="utf-8-sig", newline="") as stream:
        eeg_metadata = list(csv.DictReader(stream))
    with args.fnirs_metadata.open(encoding="utf-8-sig", newline="") as stream:
        fnirs_metadata = list(csv.DictReader(stream))
    by_name = {row["channel_name"]: row for row in eeg}
    eeg_names = [row["ch_name"].upper() for row in eeg_metadata]
    eeg_name_mismatches = len(set(eeg_names) ^ set(by_name))
    expected = {int(row["Channel"]): (int(row["Source"]), int(row["Detector"])) for row in fnirs_metadata}
    topology_mismatches = sum(expected.get(row["channel"]) != (row["source"], row["detector"]) for row in channels)
    missing_channels = sorted(set(expected) ^ {row["channel"] for row in channels})
    assert len(eeg_metadata) == 64 and len(fnirs_metadata) == 51
    assert eeg_name_mismatches == topology_mismatches == 0 and not missing_channels
    asset = {
        "schema": "refed_paper_layout_v1",
        "source": {
            "url": URL,
            "pdf_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
            "page_1based": 5,
            "figure": "2",
            "extraction": "PyMuPDF vector drawing circle bounding-box midpoints and nearest PDF text labels. All spatial positions use the left combined EEG/fNIRS panel. Channel/source/detector incidence was independently recovered from right-inset yellow segment endpoints and black numbered labels before CSV comparison.",
        },
        "normalization": {
            "center_page_pt": center(rect).tolist(),
            "radius_page_pt": (rect.width + rect.height) / 4,
            "x_positive": "figure_right",
            "y_positive": "figure_anterior",
            "head_drawing_index": head_index,
        },
        "eeg": [by_name[name] for name in eeg_names],
        "reference": references[0],
        "sources": sorted(sources, key=lambda row: row["index"]),
        "detectors": sorted(detectors, key=lambda row: row["index"]),
        "channels": channels,
    }
    audit = {
        "schema": "refed_paper_layout_extraction_audit_v1",
        "command": shlex.join([".venv/bin/python", str(Path(__file__).relative_to(Path.cwd())),
            "--pdf", str(args.pdf), "--eeg-metadata", str(args.eeg_metadata),
            "--fnirs-metadata", str(args.fnirs_metadata), "--asset", str(args.asset), "--audit", str(args.audit),
        ]),
        "pymupdf_version": fitz.VersionBind,
        "source": asset["source"],
        "counts": {"eeg": len(eeg), "reference": len(references), "sources": len(sources), "detectors": len(detectors), "channels": len(channels)},
        "eeg_metadata_name_mismatch_count": eeg_name_mismatches,
        "fnirs_metadata_topology_mismatch_count": topology_mismatches,
        "fnirs_metadata_missing_channel_numbers": missing_channels,
        "max_label_to_dot_distance_page_pt": {"eeg": eeg_distance, "reference": reference_distance, "source": source_distance, "detector": detector_distance},
        "inset_channel_label_to_segment_center_max_page_pt": max(label_distances),
        "inset_segment_endpoint_to_optode_center_max_page_pt": max(endpoint_distances),
        "inset_segment_nearest_label_assignment_is_bijective": len(used_segments) == 51,
        "metadata_access": "EEG_channels.csv and fNIRS_coordinates.csv only; no signal arrays accessed",
        "note": "The common 2D coordinate is a paper layout in page points, not an individual 3D digitization. Native fNIRS XYZ metadata is neither fitted nor changed by this extraction. EEG CSV order is applied only after independent named-point extraction.",
    }
    write_exclusive(args.asset, asset)
    write_exclusive(args.audit, audit)
    print(json.dumps({key: value for key, value in audit.items() if key not in {"command", "source"}}, indent=2))


if __name__ == "__main__":
    main()
