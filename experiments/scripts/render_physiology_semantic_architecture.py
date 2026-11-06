#!/usr/bin/env python3
"""Render physiology-semantic architecture specifications as accessible SVG.

The visual language mirrors the Draw.io-owned detailed candidate architecture:
color follows modality/content, while scientific state remains explicit in text
and metadata. Plan overlays do not mutate the canonical specification.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import textwrap
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "docs/physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.json"
DRAWIO_OWNED_OUTPUTS = {
    REPO_ROOT / "docs/physiology_semantic_tokenizer/figures/physiology_semantic_architecture.svg",
    REPO_ROOT
    / "docs/physiology_semantic_tokenizer/figures/physiology_semantic_runtime_overview.svg",
    REPO_ROOT
    / "docs/physiology_semantic_tokenizer/figures/plans/observation_source_exploration_v2.svg",
}

SPEC_SCHEMAS = {"physiology_semantic_architecture_v1", "physiology_semantic_architecture_v2"}
OVERLAY_SCHEMAS = {
    "physiology_semantic_architecture_changes_v1",
    "physiology_semantic_architecture_changes_v2",
}

VISUAL_STYLE = {
    "neutral": ("#FFFFFF", "#98A2B3", "#475467"),
    "shared": ("#EEF8F4", "#2F9C77", "#17664E"),
    "teacher": ("#F5F0FF", "#7B56D8", "#4E2D9A"),
    "source": ("#F2F7FF", "#3B73E8", "#234D9B"),
    "observation": ("#F0FBFA", "#168C83", "#116A63"),
    "coupling": ("#FFF7E8", "#F28B2B", "#A7520B"),
    "selected": ("#EDF9EF", "#25A84A", "#167231"),
    "boundary": ("#FFF7F5", "#E06C5F", "#9A3D32"),
}

# Legacy JSON has no ``visual`` field; this fallback is compatibility, not a style contract.
LEGACY_ROLE_VISUAL = {
    "data": "shared",
    "teacher": "teacher",
    "encoder": "source",
    "latent": "source",
    "quantizer": "source",
    "objective": "coupling",
    "lifecycle": "neutral",
    "interface": "selected",
    "evaluator": "neutral",
}

EDGE_STYLE = {
    "data": ("#3B73E8", "", "solid data flow"),
    "training": ("#7B56D8", "9 6", "training-only supervision"),
    "gradient": ("#F28B2B", "5 4", "gradient path"),
    "control": ("#667085", "6 5", "lifecycle control"),
    "evaluation": ("#98A2B3", "5 5", "frozen evaluation"),
    "guarded": ("#F28B2B", "", "guarded transition"),
    "blocked": ("#98A2B3", "5 5", "blocked transition"),
    "removed": ("#E06C5F", "4 5", "removed relationship"),
}

CHANGE_STYLE = {
    "add": ("#25A84A", "A"),
    "modify": ("#3B73E8", "M"),
    "remove": ("#E06C5F", "R"),
}

VALID_ROLE = set(LEGACY_ROLE_VISUAL)
VALID_VISUAL = set(VISUAL_STYLE)
VALID_SCOPE = {"inference", "training_only", "export", "evaluation", "governance"}
VALID_IMPLEMENTATION = {"implemented", "planned", "removed"}
VALID_EVIDENCE = {"admitted", "guarded", "blocked", "n_a"}
VALID_SIDES = {"left", "right", "top", "bottom"}
REPLACE_FIELDS = {"label", "details", "role", "visual", "scope", "implementation", "evidence"}
LAYOUT_FIELDS = {"x", "y", "width", "height"}
EDGE_REPLACE_FIELDS = {"label", "style", "route", "label_at"}
LEGACY_LAYOUT_SCALE = 1.2
LEGACY_MIN_NODE_HEIGHT = 96

LEGACY_STATUS = {
    "implemented": ("encoder", "inference", "implemented", "n_a"),
    "interface": ("interface", "inference", "implemented", "n_a"),
    "training": ("objective", "training_only", "implemented", "n_a"),
    "guarded": ("lifecycle", "training_only", "implemented", "guarded"),
    "output": ("interface", "export", "implemented", "n_a"),
    "blocked": ("evaluator", "evaluation", "planned", "blocked"),
    "planned": ("lifecycle", "training_only", "planned", "n_a"),
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scale_geometry(item: MutableMapping[str, Any], scale: float, *, fit_text: bool = False) -> None:
    for field in ("x", "y", "width", "height"):
        if field in item:
            item[field] = round(float(item[field]) * scale, 1)
    if fit_text and "height" in item:
        item["height"] = max(float(item["height"]), LEGACY_MIN_NODE_HEIGHT)


def _prepare_inputs(
    spec: Mapping[str, Any], changes: Mapping[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Expand legacy geometry for legibility without changing its source JSON."""

    prepared_spec = copy.deepcopy(dict(spec))
    prepared_changes = copy.deepcopy(dict(changes)) if changes is not None else None
    if prepared_spec.get("schema") != "physiology_semantic_architecture_v1":
        return prepared_spec, prepared_changes

    scale = LEGACY_LAYOUT_SCALE
    prepared_spec["width"] = int(round(float(prepared_spec["width"]) * scale))
    prepared_spec["height"] = int(round(float(prepared_spec["height"]) * scale + 60))
    prepared_spec["footer_y"] = prepared_spec["height"] - 82
    for section in prepared_spec.get("sections", []):
        _scale_geometry(section, scale)
    for node in prepared_spec.get("nodes", []):
        _scale_geometry(node, scale, fit_text=True)

    if prepared_changes is None:
        return prepared_spec, None
    for node in prepared_changes.get("add_nodes", []):
        _scale_geometry(node, scale, fit_text=True)
    for change in prepared_changes.get("changes", []):
        layout = change.get("layout")
        if layout:
            _scale_geometry(layout, scale, fit_text="height" in layout)
    return prepared_spec, prepared_changes


def _normalize_node(node: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(node)
    if "role" not in normalized:
        role, scope, implementation, evidence = LEGACY_STATUS[str(normalized.get("status", "implemented"))]
        normalized.update(role=role, scope=scope, implementation=implementation, evidence=evidence)
    normalized.setdefault("scope", "inference")
    normalized.setdefault("implementation", "implemented")
    normalized.setdefault("evidence", "n_a")
    normalized.setdefault("details", [])
    return normalized


def _edge_id(edge: Mapping[str, Any]) -> str:
    explicit = edge.get("id")
    if explicit:
        return str(explicit)
    return f'{edge["from"]}--{edge["to"]}'


def _validate_node(node: Mapping[str, Any], width: int, content_bottom: int) -> None:
    missing = {"id", "label", "x", "y", "width", "height"}.difference(node)
    if missing:
        raise ValueError(f"Node is missing fields {sorted(missing)}: {node}")
    normalized = _normalize_node(node)
    if normalized["role"] not in VALID_ROLE:
        raise ValueError(f"Unsupported role {normalized['role']!r}")
    if normalized.get("visual", LEGACY_ROLE_VISUAL[normalized["role"]]) not in VALID_VISUAL:
        raise ValueError(f"Unsupported visual style {normalized.get('visual')!r}")
    if normalized["scope"] not in VALID_SCOPE:
        raise ValueError(f"Unsupported scope {normalized['scope']!r}")
    if normalized["implementation"] not in VALID_IMPLEMENTATION:
        raise ValueError(f"Unsupported implementation {normalized['implementation']!r}")
    if normalized["evidence"] not in VALID_EVIDENCE:
        raise ValueError(f"Unsupported evidence {normalized['evidence']!r}")
    x, y = float(node["x"]), float(node["y"])
    node_width, node_height = float(node["width"]), float(node["height"])
    if min(x, y) < 0 or min(node_width, node_height) <= 0:
        raise ValueError(f"Invalid node geometry: {node['id']}")
    if x + node_width > width or y + node_height > content_bottom:
        raise ValueError(f"Node lies outside content area: {node['id']}")
    required_height = 62 + 17 * len(normalized.get("details", []))
    if node_height < required_height:
        raise ValueError(
            f"Node {node['id']!r} is too short for {len(normalized.get('details', []))} detail lines "
            f"({node_height:g} < {required_height:g})"
        )


def _validate_route(route: Mapping[str, Any], edge_id: str) -> None:
    if route.get("kind", "orthogonal") != "orthogonal":
        raise ValueError(f"Only orthogonal routing is supported: {edge_id}")
    for field in ("from_side", "to_side"):
        if field in route and route[field] not in VALID_SIDES:
            raise ValueError(f"Invalid {field} for edge {edge_id}")
    for point in route.get("via", []):
        if not isinstance(point, Sequence) or len(point) != 2:
            raise ValueError(f"Invalid waypoint for edge {edge_id}: {point}")


def _validate_node_overlaps(nodes: Sequence[Mapping[str, Any]]) -> None:
    for index, first in enumerate(nodes):
        if first.get("implementation") == "removed":
            continue
        ax1, ay1 = float(first["x"]), float(first["y"])
        ax2, ay2 = ax1 + float(first["width"]), ay1 + float(first["height"])
        for second in nodes[index + 1 :]:
            if second.get("implementation") == "removed":
                continue
            bx1, by1 = float(second["x"]), float(second["y"])
            bx2, by2 = bx1 + float(second["width"]), by1 + float(second["height"])
            if min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1):
                raise ValueError(f"Architecture nodes overlap: {first['id']} and {second['id']}")


def _validate(spec: Mapping[str, Any], changes: Mapping[str, Any] | None = None) -> None:
    if spec.get("schema") not in SPEC_SCHEMAS:
        raise ValueError("Unsupported architecture specification schema")
    width = int(spec["width"])
    content_bottom = int(spec.get("footer_y", spec["height"]))
    nodes = [_normalize_node(node) for node in spec.get("nodes", [])]
    for node in nodes:
        _validate_node(node, width, content_bottom)
    node_ids = [str(node["id"]) for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Architecture node ids must be unique")
    known = set(node_ids)
    canonical_node_map = {str(node["id"]): node for node in nodes}
    edge_ids: list[str] = []
    for edge in spec.get("edges", []):
        edge_id = _edge_id(edge)
        edge_ids.append(edge_id)
        if edge["from"] not in known or edge["to"] not in known:
            raise ValueError(f"Unknown edge endpoint: {edge}")
        if edge.get("style", "data") not in EDGE_STYLE:
            raise ValueError(f"Unsupported edge style: {edge}")
        _validate_route(edge.get("route", {}), edge_id)
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("Architecture edge ids must be unique; specify explicit ids for parallel edges")
    if changes is None:
        return
    if changes.get("schema") not in OVERLAY_SCHEMAS:
        raise ValueError("Unsupported change-overlay schema")
    add_nodes = [_normalize_node(node) for node in changes.get("add_nodes", [])]
    added_ids = [str(node["id"]) for node in add_nodes]
    if len(added_ids) != len(set(added_ids)):
        raise ValueError("Added node ids must be unique")
    if known.intersection(added_ids):
        raise ValueError("Added node ids must not replace baseline nodes")
    for node in add_nodes:
        _validate_node(node, width, content_bottom)
    known |= set(added_ids)
    for change in changes.get("changes", []):
        if change["node_id"] not in known:
            raise ValueError(f"Change references unknown node {change['node_id']!r}")
        if change["kind"] not in CHANGE_STYLE:
            raise ValueError(f"Unsupported change kind {change['kind']!r}")
        if not str(change.get("note", "")).strip():
            raise ValueError(f"Change note is required for {change['node_id']}")
        unknown_replace = set(change.get("replace", {})).difference(REPLACE_FIELDS)
        unknown_layout = set(change.get("layout", {})).difference(LAYOUT_FIELDS)
        if unknown_replace or unknown_layout:
            raise ValueError(f"Unsupported node replacement fields: {sorted(unknown_replace | unknown_layout)}")
        candidate = dict(canonical_node_map[change["node_id"]])
        candidate.update(change.get("replace", {}))
        candidate.update(change.get("layout", {}))
        if change["kind"] == "remove":
            candidate["implementation"] = "removed"
        _validate_node(candidate, width, content_bottom)
    known_edges = set(edge_ids)
    for change in changes.get("edge_changes", []):
        if change["edge_id"] not in known_edges:
            raise ValueError(f"Change references unknown edge {change['edge_id']!r}")
        if change["kind"] not in CHANGE_STYLE:
            raise ValueError(f"Unsupported edge change kind {change['kind']!r}")
        if not str(change.get("note", "")).strip():
            raise ValueError(f"Edge change note is required for {change['edge_id']}")
        unknown_edge_replace = set(change.get("replace", {})).difference(EDGE_REPLACE_FIELDS)
        if unknown_edge_replace:
            raise ValueError(f"Unsupported edge replacement fields: {sorted(unknown_edge_replace)}")
        if change.get("replace", {}).get("style", "data") not in EDGE_STYLE:
            raise ValueError(f"Unsupported replacement edge style for {change['edge_id']}")
        if "route" in change.get("replace", {}):
            _validate_route(change["replace"]["route"], change["edge_id"])
    added_edge_ids: list[str] = []
    for edge in changes.get("add_edges", []):
        if edge["from"] not in known or edge["to"] not in known:
            raise ValueError(f"Unknown added-edge endpoint: {edge}")
        edge_id = _edge_id(edge)
        added_edge_ids.append(edge_id)
        if edge_id in known_edges:
            raise ValueError(f"Added edge id replaces baseline edge: {edge_id}")
        if edge.get("style", "data") not in EDGE_STYLE:
            raise ValueError(f"Unsupported added-edge style: {edge}")
        _validate_route(edge.get("route", {}), edge_id)
    if len(added_edge_ids) != len(set(added_edge_ids)):
        raise ValueError("Added edge ids must be unique")


def _change_records(changes: Mapping[str, Any]) -> tuple[list[dict[str, str]], dict[tuple[str, str], str]]:
    records: list[dict[str, str]] = []
    references: dict[tuple[str, str], str] = {}
    counts = {kind: 0 for kind in CHANGE_STYLE}

    def add(kind: str, object_type: str, object_id: str, note: str) -> None:
        counts[kind] += 1
        reference = f"{CHANGE_STYLE[kind][1]}{counts[kind]}"
        records.append(
            {"kind": kind, "object_type": object_type, "object_id": object_id, "note": note, "ref": reference}
        )
        references[(object_type, object_id)] = reference

    for change in changes.get("changes", []):
        add(str(change["kind"]), "node", str(change["node_id"]), str(change["note"]))
    for node in changes.get("add_nodes", []):
        add("add", "node", str(node["id"]), str(node.get("note", f"Add {node['label']}")))
    for change in changes.get("edge_changes", []):
        add(str(change["kind"]), "edge", str(change["edge_id"]), str(change["note"]))
    for edge in changes.get("add_edges", []):
        edge_id = _edge_id(edge)
        add("add", "edge", edge_id, str(edge.get("note", f"Add {edge['from']} to {edge['to']} flow")))
    return records, references


def _compose_view(
    spec: Mapping[str, Any], changes: Mapping[str, Any] | None
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    view = copy.deepcopy(dict(spec))
    view["nodes"] = [_normalize_node(node) for node in view["nodes"]]
    view["edges"] = [dict(edge) for edge in view["edges"]]
    if changes is None:
        return view, []

    records, references = _change_records(changes)
    node_map: dict[str, MutableMapping[str, Any]] = {node["id"]: node for node in view["nodes"]}
    for change in changes.get("changes", []):
        node = node_map[change["node_id"]]
        node["_canonical"] = {
            key: node[key] for key in ("label", "role", "scope", "implementation", "evidence")
        }
        node.update(copy.deepcopy(change.get("replace", {})))
        node.update(copy.deepcopy(change.get("layout", {})))
        if change["kind"] == "remove":
            node["implementation"] = "removed"
        node["_change_kind"] = change["kind"]
        node["_change_note"] = change["note"]
        node["_change_ref"] = references[("node", change["node_id"])]

    for raw_node in changes.get("add_nodes", []):
        node = _normalize_node(raw_node)
        node["_change_kind"] = "add"
        node["_change_note"] = raw_node.get("note", f"Add {raw_node['label']}")
        node["_change_ref"] = references[("node", raw_node["id"])]
        view["nodes"].append(node)
        node_map[node["id"]] = node

    edge_map = {_edge_id(edge): edge for edge in view["edges"]}
    for change in changes.get("edge_changes", []):
        edge = edge_map[change["edge_id"]]
        edge.update(copy.deepcopy(change.get("replace", {})))
        edge["_change_kind"] = change["kind"]
        edge["_change_note"] = change["note"]
        edge["_change_ref"] = references[("edge", change["edge_id"])]
        if change["kind"] == "remove":
            edge["style"] = "removed"
    for raw_edge in changes.get("add_edges", []):
        edge = dict(raw_edge)
        edge["_change_kind"] = "add"
        edge["_change_note"] = raw_edge.get("note", f"Add {raw_edge['from']} to {raw_edge['to']} flow")
        edge["_change_ref"] = references[("edge", _edge_id(raw_edge))]
        view["edges"].append(edge)

    view["title"] = changes.get("title", f"Plan · {changes['plan_id'].replace('_', ' ')}")
    view["subtitle"] = changes.get("subtitle", changes.get("summary", "Proposed architecture after-state"))
    if "banner" in changes:
        view["banner"] = changes["banner"]
    view["_plan_id"] = changes["plan_id"]
    view["_plan_summary"] = changes.get("summary", "")
    return view, records


def _node_anchor(node: Mapping[str, Any], side: str, fraction: float = 0.5) -> tuple[float, float]:
    x, y = float(node["x"]), float(node["y"])
    width, height = float(node["width"]), float(node["height"])
    fraction = max(0.05, min(0.95, fraction))
    if side == "left":
        return x, y + height * fraction
    if side == "right":
        return x + width, y + height * fraction
    if side == "top":
        return x + width * fraction, y
    return x + width * fraction, y + height


def _route_points(edge: Mapping[str, Any], node_map: Mapping[str, Mapping[str, Any]]) -> list[tuple[float, float]]:
    source = node_map[edge["from"]]
    target = node_map[edge["to"]]
    route = edge.get("route", {})
    source_x = float(source["x"]) + float(source["width"]) / 2
    target_x = float(target["x"]) + float(target["width"]) / 2
    source_y = float(source["y"]) + float(source["height"]) / 2
    target_y = float(target["y"]) + float(target["height"]) / 2
    if "from_side" in route:
        from_side = route["from_side"]
    else:
        from_side = "right" if abs(target_x - source_x) >= abs(target_y - source_y) and target_x >= source_x else "bottom"
    if "to_side" in route:
        to_side = route["to_side"]
    else:
        to_side = "left" if from_side in {"right", "left"} else "top"
    start = _node_anchor(source, from_side, float(route.get("from_fraction", 0.5)))
    end = _node_anchor(target, to_side, float(route.get("to_fraction", 0.5)))
    via = [(float(point[0]), float(point[1])) for point in route.get("via", [])]
    if via:
        points = [start, *via, end]
    elif from_side in {"left", "right"} and to_side in {"left", "right"}:
        midpoint = (start[0] + end[0]) / 2
        points = [start, (midpoint, start[1]), (midpoint, end[1]), end]
    elif from_side in {"top", "bottom"} and to_side in {"top", "bottom"}:
        midpoint = (start[1] + end[1]) / 2
        points = [start, (start[0], midpoint), (end[0], midpoint), end]
    else:
        points = [start, (end[0], start[1]), end]
    compact: list[tuple[float, float]] = []
    for point in points:
        if compact and point == compact[-1]:
            continue
        if len(compact) >= 2:
            a, b = compact[-2], compact[-1]
            if (a[0] == b[0] == point[0]) or (a[1] == b[1] == point[1]):
                compact[-1] = point
                continue
        compact.append(point)
    return compact


def _label_position(points: Sequence[tuple[float, float]], label_at: Mapping[str, Any] | None) -> tuple[float, float]:
    if label_at:
        segment = max(0, min(int(label_at.get("segment", 0)), len(points) - 2))
        position = float(label_at.get("position", 0.5))
    else:
        lengths = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(points, points[1:])]
        segment = max(range(len(lengths)), key=lengths.__getitem__)
        position = 0.5
    start, end = points[segment], points[segment + 1]
    x = start[0] + (end[0] - start[0]) * position
    y = start[1] + (end[1] - start[1]) * position
    if label_at:
        x += float(label_at.get("dx", 0))
        y += float(label_at.get("dy", -8))
    else:
        y -= 8
    return x, y


def _safe_xml_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


def _render_edge(edge: Mapping[str, Any], node_map: Mapping[str, Mapping[str, Any]]) -> str:
    edge_id = _edge_id(edge)
    safe_id = _safe_xml_id(edge_id)
    style = str(edge.get("style", "data"))
    color, dash_pattern, meaning = EDGE_STYLE[style]
    points = _route_points(edge, node_map)
    path = " ".join(
        [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
        + [f"L {point[0]:.1f} {point[1]:.1f}" for point in points[1:]]
    )
    dash = f' stroke-dasharray="{dash_pattern}"' if dash_pattern else ""
    change_kind = edge.get("_change_kind")
    change_attrs = ""
    halo = ""
    if change_kind:
        change_attrs = f' data-change-kind="{escape(str(change_kind))}" data-change-ref="{escape(str(edge["_change_ref"]))}"'
        change_color = CHANGE_STYLE[str(change_kind)][0]
        halo = f'<path d="{path}" fill="none" stroke="{change_color}" stroke-width="7" opacity="0.14"/>'
    label_text = str(edge.get("label", ""))
    if edge.get("_change_ref"):
        label_text = f'{edge["_change_ref"]} · {label_text}' if label_text else str(edge["_change_ref"])
    label = ""
    if label_text:
        x, y = _label_position(points, edge.get("label_at"))
        label_width = max(36, min(260, 8 + len(label_text) * 6.6))
        label = (
            f'<g class="edge-label" transform="translate({x:.1f},{y:.1f})">'
            f'<rect x="{-label_width / 2:.1f}" y="-13" width="{label_width:.1f}" height="20" rx="7" fill="#FFFFFF" stroke="#CDD5DB" stroke-width="1" opacity=".96"/>'
            f'<text x="0" y="1" text-anchor="middle" fill="#34424E">{escape(label_text)}</text></g>'
        )
    description = str(edge.get("_change_note", edge.get("label", meaning)))
    return (
        f'<g id="edge-{safe_id}" class="edge edge-{escape(style)}" data-edge-style="{escape(style)}"{change_attrs} '
        f'role="group" aria-labelledby="edge-{safe_id}-title edge-{safe_id}-desc">'
        f'<title id="edge-{safe_id}-title">{escape(str(edge["from"]))} to {escape(str(edge["to"]))}</title>'
        f'<desc id="edge-{safe_id}-desc">{escape(description)}</desc>{halo}'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round" '
        f'marker-end="url(#arrow-{escape(style)})"{dash}/>{label}</g>'
    )


def _render_node(node: Mapping[str, Any]) -> str:
    role = str(node["role"])
    visual = str(node.get("visual", LEGACY_ROLE_VISUAL[role]))
    scope = str(node["scope"])
    implementation = str(node["implementation"])
    evidence = str(node["evidence"])
    fill, accent, text_color = VISUAL_STYLE[visual]
    x, y = float(node["x"]), float(node["y"])
    width, height = float(node["width"]), float(node["height"])
    opacity = "0.45" if implementation == "removed" else "1"
    details = list(node.get("details", []))
    detail_lines = "".join(
        f'<tspan x="{x + width / 2:.1f}" dy="17">{escape(str(line))}</tspan>' for line in details
    )
    change_kind = node.get("_change_kind")
    change_overlay = ""
    change_attrs = ""
    if change_kind:
        change_color = CHANGE_STYLE[str(change_kind)][0]
        reference = str(node["_change_ref"])
        change_attrs = (
            f' data-change-kind="{escape(str(change_kind))}" data-change-ref="{escape(reference)}"'
        )
        change_overlay = (
            f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="{width + 10:.1f}" height="{height + 10:.1f}" rx="15" '
            f'fill="none" stroke="{change_color}" stroke-width="3" stroke-dasharray="8 5"/>'
            f'<rect x="{x + width - 45:.1f}" y="{y - 15:.1f}" width="45" height="24" rx="12" fill="{change_color}"/>'
            f'<text x="{x + width - 22.5:.1f}" y="{y + 1:.1f}" class="change-badge">{escape(reference)}</text>'
        )
    if implementation == "removed":
        change_overlay += (
            f'<path d="M {x + 10:.1f} {y + 10:.1f} L {x + width - 10:.1f} {y + height - 10:.1f} '
            f'M {x + width - 10:.1f} {y + 10:.1f} L {x + 10:.1f} {y + height - 10:.1f}" '
            f'stroke="#9F1239" stroke-width="4"/>'
        )
    canonical = node.get("_canonical", {})
    canonical_attrs = ""
    if canonical:
        canonical_attrs = (
            f' data-canonical-status="{escape(str(canonical["implementation"]))}"'
            f' data-canonical-evidence="{escape(str(canonical["evidence"]))}"'
        )
    description_parts = [str(node["label"]), *map(str, details), f"scope {scope}", f"implementation {implementation}"]
    if evidence != "n_a":
        description_parts.append(f"evidence {evidence}")
    if node.get("_change_note"):
        description_parts.append(str(node["_change_note"]))
    node_id = _safe_xml_id(str(node["id"]))
    return (
        f'<g id="node-{node_id}" class="node visual-{escape(visual)}" data-role="{escape(role)}" data-visual="{escape(visual)}" data-scope="{escape(scope)}" '
        f'data-implementation="{escape(implementation)}" data-evidence="{escape(evidence)}" data-status="{escape(evidence if evidence != "n_a" else implementation)}"'
        f'{canonical_attrs}{change_attrs} opacity="{opacity}" role="group" '
        f'aria-labelledby="node-{node_id}-title node-{node_id}-desc">'
        f'<title id="node-{node_id}-title">{escape(str(node["label"]))}</title>'
        f'<desc id="node-{node_id}-desc">{escape("; ".join(description_parts))}</desc>'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="16" fill="{fill}" stroke="{accent}" stroke-width="2"/>'
        f'<text x="{x + width / 2:.1f}" y="{y + 29:.1f}" class="node-title" fill="{text_color}">{escape(str(node["label"]))}</text>'
        f'<text x="{x + width / 2:.1f}" y="{y + 42:.1f}" class="node-detail" fill="{text_color}">{detail_lines}</text>'
        f'{change_overlay}</g>'
    )


def _render_sections(sections: Iterable[Mapping[str, Any]]) -> str:
    rendered = []
    for section in sections:
        description = str(section.get("description", ""))
        rendered.append(
            f'<g id="section-{escape(str(section["id"]))}">'
            f'<rect x="{section["x"]}" y="{section["y"]}" width="{section["width"]}" height="{section["height"]}" rx="18" class="section-box" fill="#FAFBFD" stroke="#667085" stroke-width="2"/>'
            f'<text x="{section["x"] + 18}" y="{section["y"] + 31}" class="section-title">{escape(str(section["label"]))}</text>'
            + (
                f'<text x="{section["x"] + 18}" y="{section["y"] + 52}" class="section-description">{escape(description)}</text>'
                if description
                else ""
            )
            + "</g>"
        )
    return "".join(rendered)


def _wrap_callout(note: str, width: int = 96) -> list[str]:
    return textwrap.wrap(note, width=width, break_long_words=False, break_on_hyphens=False) or [""]


def _render_callouts(records: Sequence[Mapping[str, str]], start_y: int, width: int) -> tuple[str, int]:
    if not records:
        return "", start_y
    columns = 2
    column_width = (width - 120) / columns
    column_x = [40, 80 + column_width]
    per_column = (len(records) + columns - 1) // columns
    rendered = [
        f'<g id="plan-callouts"><rect x="30" y="{start_y}" width="{width - 60}" height="__HEIGHT__" rx="18" class="callout-panel"/>'
        f'<text x="50" y="{start_y + 34}" class="callout-title">Plan delta · proposed after-state</text>'
    ]
    bottoms = []
    for column in range(columns):
        y = start_y + 64
        for record in records[column * per_column : (column + 1) * per_column]:
            lines = _wrap_callout(str(record["note"]), 88)
            row_height = 34 + 17 * len(lines)
            color = CHANGE_STYLE[str(record["kind"])][0]
            callout_id = _safe_xml_id(f'{record["object_type"]}-{record["object_id"]}')
            rendered.append(
                f'<g id="callout-{callout_id}" role="group" aria-labelledby="callout-{callout_id}-title callout-{callout_id}-desc">'
                f'<title id="callout-{callout_id}-title">{escape(str(record["ref"]))}: {escape(str(record["object_id"]))}</title>'
                f'<desc id="callout-{callout_id}-desc">{escape(str(record["note"]))}</desc>'
                f'<rect x="{column_x[column]:.1f}" y="{y:.1f}" width="42" height="24" rx="12" fill="{color}"/>'
                f'<text x="{column_x[column] + 21:.1f}" y="{y + 16:.1f}" class="change-badge">{escape(str(record["ref"]))}</text>'
                f'<text x="{column_x[column] + 54:.1f}" y="{y + 15:.1f}" class="callout-object">{escape(str(record["object_type"]))} · {escape(str(record["object_id"]))}</text>'
                f'<text x="{column_x[column] + 54:.1f}" y="{y + 35:.1f}" class="callout-note">'
                + "".join(
                    f'<tspan x="{column_x[column] + 54:.1f}" dy="{0 if index == 0 else 17}">{escape(line)}</tspan>'
                    for index, line in enumerate(lines)
                )
                + "</text></g>"
            )
            y += row_height
        bottoms.append(y)
    panel_height = max(bottoms) - start_y + 18
    rendered[0] = rendered[0].replace("__HEIGHT__", f"{panel_height:.1f}")
    rendered.append("</g>")
    return "".join(rendered), int(start_y + panel_height)


def render_svg(
    spec: Mapping[str, Any],
    changes: Mapping[str, Any] | None = None,
    *,
    spec_source: str | None = None,
    changes_source: str | None = None,
) -> str:
    """Render a deterministic SVG from a canonical spec and optional overlay."""

    prepared_spec, prepared_changes = _prepare_inputs(spec, changes)
    _validate(prepared_spec, prepared_changes)
    view, records = _compose_view(prepared_spec, prepared_changes)
    width = int(view["width"])
    base_height = int(view["height"])
    content_bottom = int(view.get("footer_y", base_height))
    callout_svg = ""
    canvas_height = base_height
    if records:
        callout_svg, callout_bottom = _render_callouts(records, base_height + 20, width)
        canvas_height = callout_bottom + 24
    callout_block = f"  {callout_svg}\n" if callout_svg else ""
    nodes = [dict(node) for node in view["nodes"]]
    for node in nodes:
        _validate_node(node, width, content_bottom)
    _validate_node_overlaps(nodes)
    node_map = {str(node["id"]): node for node in nodes}
    metadata = {
        "schema": view["schema"],
        "change_plan": view.get("_plan_id"),
        "spec_source": spec_source or str(view.get("source", "in-memory")),
        "changes_source": changes_source,
    }
    section_svg = _render_sections(view["sections"])
    edge_svg = "".join(_render_edge(edge, node_map) for edge in view["edges"])
    node_svg = "".join(_render_node(node) for node in nodes)
    banner = str(view.get("banner", ""))
    banner_svg = ""
    if banner:
        banner_svg = (
            f'<g id="evidence-boundary"><rect x="40" y="96" width="{width - 80}" height="43" rx="16" class="banner-box" fill="#FFF4E5" stroke="#F28B2B" stroke-width="2"/>'
            f'<text x="{width / 2:.1f}" y="123" class="banner-text" fill="#B85B08">{escape(banner)}</text></g>'
        )
    banner_block = f"  {banner_svg}\n" if banner_svg else ""
    marker_defs = "".join(
        f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{style[0]}"/></marker>'
        for name, style in EDGE_STYLE.items()
    )
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{canvas_height}" viewBox="0 0 {width} {canvas_height}" role="img" aria-labelledby="svg-title svg-desc">
  <title id="svg-title">{escape(str(view["title"]))}</title>
  <desc id="svg-desc">{escape(str(view["subtitle"]))}</desc>
  <metadata>{escape(json.dumps(metadata, sort_keys=True))}</metadata>
  <defs>{marker_defs}</defs>
  <style>
    text {{ font-family: Helvetica, "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", Arial, sans-serif; }}
    .page-title {{ font-size: 28px; font-weight: 700; fill: #172033; }}
    .page-subtitle {{ font-size: 13px; fill: #4B5565; }}
    .banner-box {{ fill: #FFF4E5; stroke: #F28B2B; stroke-width: 2; }}
    .banner-text {{ font-size: 12px; font-weight: 700; fill: #B85B08; text-anchor: middle; }}
    .section-box {{ fill: #FAFBFD; stroke: #667085; stroke-width: 2; }}
    .section-title {{ font-size: 20px; font-weight: 700; fill: #303846; }}
    .section-description {{ font-size: 12px; fill: #475467; }}
    .node-title {{ font-size: 13px; font-weight: 700; text-anchor: middle; }}
    .node-detail {{ font-size: 10.5px; text-anchor: middle; }}
    .edge-label rect {{ fill: #FFFFFF; stroke: #CDD5DB; stroke-width: 1; opacity: .96; }}
    .edge-label text {{ font-size: 10.5px; text-anchor: middle; fill: #34424E; }}
    .change-badge {{ font-size: 10px; font-weight: 850; text-anchor: middle; fill: #FFFFFF; }}
    .callout-panel {{ fill: #FAFBFD; stroke: #667085; stroke-width: 1.4; }}
    .callout-title {{ font-size: 17px; font-weight: 750; fill: #2D3A45; }}
    .callout-object {{ font-size: 12px; font-weight: 750; fill: #2D3A45; }}
    .callout-note {{ font-size: 11.5px; fill: #53616D; }}
  </style>
  <rect width="100%" height="100%" fill="#FFFFFF"/>
  <text x="40" y="49" class="page-title">{escape(str(view["title"]))}</text>
  <text x="40" y="77" class="page-subtitle">{escape(str(view["subtitle"]))}</text>
{banner_block}  {section_svg}
  <g id="edges">{edge_svg}</g>
  <g id="nodes">{node_svg}</g>
{callout_block}</svg>
'''


def render_framework_20261029(output: Path, *, layout: str = "detailed") -> list[Path]:
    """Export a dated, read-free presentation of the Tak/voltage runtime.

    This is a communication asset, not an architecture or execution-state owner.
    Numerical traces below are explicitly illustrative. No measured arrays,
    checkpoints, protected evidence, or historical figure specifications are read.
    Equations and editable layout are retained here; PDF figures are PNG objects.

    Content owners checked for this snapshot:
      docs/EXPERIMENT_PLAN.md, sections dated 2026-10-27 through 2026-10-29;
      docs/DATA_CONTRACT.md, independent Tak targets / electrode voltage;
      experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_v2.yaml;
      src/data/dual_component_tokenizer.py;
      src/tokenizers/dual_component_tokenizer.py;
      experiments/scripts/run_dual_component_tokenizer.py, voltage_train;
      research_state/registry.json, completed dual-component / voltage records.
    The selected Hb synthesis is owned by the retained v2 voltage_selection.json.
    """
    import json

    import fitz
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors, font_manager, patches
    from PIL import Image

    if layout not in {"detailed", "flow", "token_first", "paper", "paper_flow"}:
        raise ValueError("Framework layout must be detailed, flow, token_first, paper or paper_flow")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh versioned export directory; retained exports are immutable")
    output.mkdir(parents=True, exist_ok=True)
    publication = layout in {"paper", "paper_flow"}
    font_family = "NotoSansCJK" if publication else "NotoSerifCJK"
    regular = Path(f"/usr/share/fonts/opentype/noto/{font_family}-Regular.ttc")
    bold = regular.with_name(f"{font_family}-Bold.ttc")
    if not regular.exists() or not bold.exists():
        raise FileNotFoundError(f"{font_family} regular and bold fonts are required")
    matplotlib.rcParams.update({
        "mathtext.fontset": "cm", "svg.fonttype": "none", "axes.unicode_minus": False,
        "savefig.facecolor": "white", "font.family": "serif",
    })
    ink, gray, line = "#282828", "#747474", "#A1A1A1"
    blue, orange, purple = "#306FB6", "#D18431", "#9160C0"
    red, teal, fill = "#D95757", "#229B94", "#F0F0F0"
    if layout == "paper":
        ink, gray, line = "#19232F", "#384451", "#536171"
        blue, orange, purple = "#155E9A", "#A74D06", "#65338D"
    elif layout == "paper_flow":
        ink, gray, line = "#111111", "#333333", "#59616A"
        blue, orange, purple = "#2166AC", "#B35C00", "#6F3C91"
    sources = [
        "docs/EXPERIMENT_PLAN.md", "docs/DATA_CONTRACT.md", "docs/ARCHITECTURE.md",
        "experiments/configs/physiology_semantic_tokenizer/dual_component_tokenizer_v1.yaml",
        "experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_v2.yaml",
        "src/data/dual_component_tokenizer.py", "src/tokenizers/dual_component_tokenizer.py",
        "src/tokenizers/ema_vector_quantizer.py", "src/foundation/dual_component_masked.py",
        "src/analysis/dual_component_tokenizer.py",
        "experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_10hz_v2.yaml",
        "experiments/scripts/run_dual_component_tokenizer.py", "research_state/registry.json",
        "experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/"
        "20261029_voltage_v2/voltage_selection.json",
    ]
    metadata = {"Date": "2026-10-29", "Creator": str(Path(__file__).relative_to(REPO_ROOT)),
                "Description": "Dated explanatory projection; all waveforms are schematic. Sources: "
                + "; ".join(sources)}

    class Plate:
        def __init__(self, name, title, subtitle, height, width=2000):
            self.name, self.height, self.title, self.width = name, height, title, width
            unit = 72 if publication else 100
            self.fig = plt.figure(figsize=(width / unit, height / unit), dpi=100)
            self.ax = self.fig.add_axes((0, 0, 1, 1))
            self.ax.set(xlim=(0, width), ylim=(height, 0))
            self.ax.axis("off")
            self.texts = []
            if publication:
                return
            self.t(65, 53, title, 25, boldface=True)
            self.t(65, 96, subtitle, 12.5, color=gray)
            self.t(width-85, 53, "结构示意 · 2026-10-29" if layout=="token_first" else "方法快照 · 2026-10-29",
                   10, color=gray, ha="right")

        def t(self, x, y, value, size=13, *, color=ink, ha="left", boldface=False):
            text = self.ax.text(x, y, value, va="center", ha=ha, fontsize=size,
                                color=color, fontproperties=font_manager.FontProperties(
                                    fname=bold if boldface else regular),
                                linespacing=1.25 if layout == "paper" else 1.6)
            self.texts.append(text)
            return text

        def m(self, x, y, value, size=19, *, color=ink, ha="center"):
            text = self.ax.text(x, y, "$" + value + "$", va="center", ha=ha,
                                fontsize=size, color=color, math_fontfamily="cm")
            self.texts.append(text)
            return text

        def box(self, x, y, w, h, *, face="white", edge=line, dashed=False):
            self.ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=face,
                              edgecolor=edge, linewidth=1.1 if layout == "paper" else 1.05,
                              linestyle=(0, (4, 3)) if dashed else "solid"))

        def block(self, x, y, w, h, heading, detail="", *, dashed=False, face=fill):
            self.box(x, y, w, h, face=face, dashed=dashed)
            self.t(x + w/2, y + (27 if detail else h/2), heading, 14, ha="center")
            if detail:
                self.t(x + w/2, y + h/2 + 15, detail, 11.5, color=gray, ha="center")

        def segment(self, points, *, color=ink, dashed=False, width=1.25):
            x, y = zip(*points)
            self.ax.plot(x, y, color=color, lw=width, solid_capstyle="butt",
                         linestyle=(0, (4, 2.6)) if dashed else "solid")

        def arrow(self, points, *, color=ink, dashed=False, width=1.25):
            if len(points) > 2:
                self.segment(points[:-1], color=color, dashed=dashed, width=width)
            self.ax.add_patch(patches.FancyArrowPatch(points[-2], points[-1],
                              arrowstyle="-|>", mutation_scale=5 if layout == "paper" else 11, linewidth=width,
                              color=color, shrinkA=0, shrinkB=0,
                              linestyle=(0, (4, 2.6)) if dashed else "solid"))

        def rule(self, y, dashed=False):
            self.segment([(65, y), (1915, y)], color="#CBCBCB", dashed=dashed, width=.8)

        def section(self, y, label):
            self.t(65, y, label, 17, boldface=True)

        def wave(self, x, y, w, h, kind="eeg", *, color=None):
            self.box(x, y, w, h, edge="#A7A7A7")
            t = np.linspace(0, 1, 240)
            if kind.startswith("eeg"):
                q = (.47*np.sin(2*np.pi*12*t) + .23*np.sin(2*np.pi*27*t + .3)
                     + .14*np.sin(2*np.pi*4*t)) * (.8 + .2*np.cos(2*np.pi*t))
                if kind == "eeg_residual":
                    q = .65*np.sin(2*np.pi*11*t)*np.exp(-((t-.55)/.25)**2)
                self.ax.plot(x+8+(w-16)*t, y+h/2-q*h*.35, lw=1, color=color or blue)
            elif kind == "hb":
                q = np.exp(-((t-.42)/.19)**2) - .18*np.exp(-((t-.84)/.15)**2)
                self.ax.plot(x+8+(w-16)*t, y+h*.52-q*h*.35, lw=1.3, color=red)
                self.ax.plot(x+8+(w-16)*t, y+h*.52+q*h*.27, lw=1.3, color=teal)
            else:
                q = np.exp(-((t-.38)/.19)**2) - .36*np.exp(-((t-.78)/.18)**2)
                if kind in ("residual", "hb_residual"):
                    q = .45*np.sin(2*np.pi*2*t)+.16*np.sin(2*np.pi*5*t)
                self.ax.plot(x+8+(w-16)*t, y+h*.60-q*h*.4, lw=1.45, color=color or blue)
                if kind in ("hb_component", "hb_residual"):
                    self.ax.plot(x+8+(w-16)*t, y+h*.46+q*h*.28,
                                 lw=1.15, color=color or blue, linestyle=(0,(3,2)))

        def book(self, x, y, w, h, color, levels):
            rgb = np.asarray(colors.to_rgb(color))
            for row in range(4):
                for col in range(3):
                    alpha = [.28, .46, .65, .9][(row+col)%4]
                    self.box(x+col*(w/3), y+row*(h/4), w/3-3, h/4-3,
                             face=tuple(1-alpha*(1-rgb)), edge="white")
            self.box(x+w/3-1, y+h/4-1, w/3-1, h/4-1, face="none", edge=ink)
            self.t(x+w/2, y+h+18, f"{levels} 层 × 128 码字", 10.5, color=gray, ha="center")

        def tokens(self, x, y, color, levels=1):
            rgb = np.asarray(colors.to_rgb(color))
            for layer in range(min(levels, 4)):
                for pos in range(4):
                    alpha = .32 + .16*((pos+layer)%4)
                    self.box(x+pos*31, y+layer*13, 27, 10 if levels>1 else 26,
                             face=tuple(1-alpha*(1-rgb)), edge="white")

        def predictor(self, x, y, modality, component):
            self.ax.add_patch(patches.Polygon([(x,y-39),(x+185,y-27),(x+185,y+27),(x,y+39)],
                              facecolor=fill, edgecolor="#888888", lw=1.1))
            self.m(x+92, y-5, rf"F_{{{modality}}}^{{{component}}}=D_{{{modality}}}^{{{component}}}\circ E_{{{modality}}}^{{{component}}}", 18)
            self.t(x+92, y+22, "编码 → 连续波形预测", 10, color=gray, ha="center")

        def save(self):
            self.fig.canvas.draw()
            renderer = self.fig.canvas.get_renderer()
            outside = [t.get_text() for t in self.texts
                       if not self.fig.bbox.contains(*t.get_window_extent(renderer).get_points()[0])
                       or not self.fig.bbox.contains(*t.get_window_extent(renderer).get_points()[1])]
            if outside:
                raise ValueError(f"Clipped figure labels in {self.name}: {outside}")
            bounds = [(t, t.get_window_extent(renderer)) for t in self.texts]
            overlapping = []
            for i, (first, a) in enumerate(bounds):
                for second, b in bounds[i+1:]:
                    if min(a.x1,b.x1)-max(a.x0,b.x0)>2 and min(a.y1,b.y1)-max(a.y0,b.y0)>2:
                        overlapping.append((first.get_text(),second.get_text()))
            if overlapping:
                raise ValueError(f"Overlapping figure labels in {self.name}: {overlapping}")
            png, svg = output/f"{self.name}.png", output/f"{self.name}.svg"
            self.fig.savefig(png, dpi=600 if layout == "paper_flow" else 300,
                             metadata={"Description": metadata["Description"]})
            # Outlined glyphs keep the editable paper artwork portable across
            # machines without relying on an installed CJK font collection.
            with matplotlib.rc_context({"svg.fonttype": "path" if publication else "none"}):
                self.fig.savefig(svg, metadata=dict(metadata, Title=self.title))
            with Image.open(png) as original:
                preview = original.convert("RGB")
                preview.thumbnail((2200, 1800))
                preview.save(output/f"{self.name}_preview.png")
            plt.close(self.fig)
            return png, svg

    def finish(figures, captions, pdf_name):
        # Complete bitmap figures, with searchable titles/captions outside them.
        pdf = fitz.open()
        for index, ((png, _), caption) in enumerate(zip(figures, captions), 1):
            with Image.open(png) as im:
                width, height = im.size
            page_width = max(1440, width/300*72+48)
            image_height = (page_width-48)*height/width
            page = pdf.new_page(width=page_width, height=image_height+108)
            page.insert_font(fontname="china-s")
            page.insert_text((24,28), f"项目方法框架 · 2026-10-29 · 图 {index}/{len(figures)}",
                             fontname="china-s",fontsize=12)
            page.insert_image(fitz.Rect(24,40,page_width-24,40+image_height),filename=str(png))
            remaining = page.insert_textbox(fitz.Rect(24,52+image_height,page_width-24,100+image_height),
                                           caption,fontname="china-s",fontsize=11)
            if remaining < 0:
                raise ValueError("PDF caption exceeds its allotted searchable area")
        pdf.subset_fonts()
        pdf_path = output/pdf_name
        pdf.save(pdf_path,garbage=4,deflate=True)
        pdf.close()
        with fitz.open(pdf_path) as check:
            if len(check)!=len(figures) or any(len(page.get_images())!=1 or not page.get_text().strip()
                                              for page in check):
                raise ValueError("PDF must have one complete bitmap figure and searchable captions per page")
        index = {"kind":"dated_framework_export", "snapshot_date":"2026-10-29", "layout":layout,
                 "sources":sources,"generator":str(Path(__file__).relative_to(REPO_ROOT)),
                 "waveforms":"illustrative_not_experiment_results", "dpi":300,
                 "figures":[{"png":a.name,"svg":b.name} for a,b in figures],"pdf":pdf_path.name,
                 "pdf_figures":"one_png_image_object_per_page; selectable_titles_and_captions"}
        if layout == "token_first":
            index["architecture_basis"] = (
                "User-requested token-first architecture; hard_student EMA-VQ and frozen-ID reader interfaces; "
                "conditional voltage target adapter retained; not a claim of a completed integrated voltage campaign")
        (output/"figure_index.json").write_text(json.dumps(index,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        return [path for pair in figures for path in pair]+[pdf_path]

    def finish_publication(p, caption, *, width_mm, dpi, design):
        """Export complete bitmap figures at print size and selectable captions."""
        import io
        from fontTools.subset import Subsetter
        from fontTools.ttLib import TTFont

        figure = p.save()
        figure_pdf = output / f"{p.name}.pdf"
        with fitz.open() as pdf:
            page = pdf.new_page(width=p.width, height=p.height)
            page.insert_image(page.rect, filename=str(figure[0]))
            pdf.set_metadata({"title": p.title, "subject": metadata["Description"]})
            pdf.save(figure_pdf, deflate=True)
        atlas = output / f"{p.name}_captioned.pdf"
        # Subset the font's Unicode map before embedding. Full CJK fonts map
        # some shared glyphs back to compatibility characters (e.g. 路 vs 路),
        # which otherwise breaks literal search/copy of the Chinese caption.
        with TTFont(regular, fontNumber=0) as font:
            subsetter = Subsetter()
            subsetter.options.retain_gids = True  # Preserve CFF CID-to-glyph identities in PDF.
            subsetter.populate(text=caption)
            subsetter.subset(font)
            caption_font = io.BytesIO()
            font.save(caption_font)
        with fitz.open() as pdf:
            page = pdf.new_page(width=p.width + 36, height=p.height + 205)
            page.insert_image(fitz.Rect(18, 12, 18 + p.width, 12 + p.height), filename=str(figure[0]))
            page.insert_font(fontname="caption", fontbuffer=caption_font.getvalue())
            remaining = page.insert_textbox(fitz.Rect(18, p.height + 24, 18 + p.width, p.height + 192),
                                            caption, fontname="caption", fontsize=8, lineheight=1.4)
            if remaining < 0:
                raise ValueError("Paper figure caption exceeds the searchable caption area")
            pdf.save(atlas, garbage=4, deflate=True)
        with fitz.open(figure_pdf) as check, fitz.open(atlas) as caption_check:
            if (len(check[0].get_images()) != 1 or check[0].get_drawings()
                    or len(caption_check[0].get_images()) != 1 or not caption_check[0].get_text().strip()):
                raise ValueError("Paper PDFs require complete bitmap figures and a searchable companion caption")
        index = {
            "kind": "dated_framework_export", "snapshot_date": "2026-10-29",
            "export_date": "2026-11-05", "layout": layout, "sources": sources,
            "generator": str(Path(__file__).relative_to(REPO_ROOT)),
            "waveforms": "illustrative_not_experiment_results", "dpi": dpi,
            "figure_size_mm": [width_mm, p.height / 72 * 25.4], **design,
            "figures": [{"png": figure[0].name, "svg": figure[1].name}],
            "pdf": figure_pdf.name, "captioned_pdf": atlas.name,
            "pdf_figures": "one_complete_png_image_object_per_page; companion_caption_selectable",
            "architecture_basis": "Publication redraw of the 2026-10-29 user-requested token-first schematic; "
                                  "not a claim of a completed integrated voltage campaign",
            "caption": caption,
        }
        extra = []
        if dpi == 600:
            tiff = output / f"{p.name}.tiff"
            with Image.open(figure[0]) as im:
                im.convert("RGB").save(tiff, compression="tiff_lzw", dpi=(dpi, dpi))
            index["figures"][0]["tiff"] = tiff.name
            extra.append(tiff)
        (output / "figure_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return [*figure, figure_pdf, atlas, *extra]

    if layout == "paper_flow":
        # The user explicitly retained the latent-VQ architecture for this visual
        # comparison. Do not relabel it as the waveform-codebook voltage v2 run.
        from matplotlib import patheffects
        from matplotlib.path import Path as MplPath

        p = Plate("project_framework_paper_flow", "Tak 教师引导的双分支离散表示：连续数据流版",
                  "", 388, width=170 / 25.4 * 72)
        cjk_family = font_manager.FontProperties(fname=regular).get_name()

        def text(x, y, label, *, ha="center", weight="normal"):
            artist = p.ax.text(x, y, label, fontsize=7, color=ink,
                fontproperties=font_manager.FontProperties(
                    family=["Liberation Sans", cjk_family], weight=weight),
                ha=ha, va="center", linespacing=1.25, zorder=6)
            p.texts.append(artist)
            return artist

        def box(x, y, w, h, *, edge=line, face="white", lw=.7):
            p.ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=face,
                                             edgecolor=edge, linewidth=lw, zorder=2))

        def node(x, y, w, h, labels, *, edge=line, face="white", weight="normal"):
            box(x, y, w, h, edge=edge, face=face)
            for i, label in enumerate(labels):
                text(x + w/2, y + h*(i + .5)/len(labels), label, weight=weight)

        def path(points, *, color=ink, dashed=False, arrow=True, lw=1., zorder=3):
            if len(points) > 2 or not arrow:
                vertices = points[:-1] if arrow else points
                xx, yy = zip(*vertices)
                p.ax.plot(xx, yy, color=color, lw=lw, zorder=zorder,
                          solid_capstyle="butt", linestyle=(0, (3, 2)) if dashed else "solid")
            if arrow:
                p.ax.add_patch(patches.FancyArrowPatch(points[-2], points[-1],
                    arrowstyle="-|>,head_length=.45,head_width=.2", mutation_scale=10,
                    color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=zorder,
                    linestyle=(0, (3, 2)) if dashed else "solid"))

        def section(y, h, label):
            box(3, y, 475, h, edge="#AEB5BC", face="#FCFCFD", lw=.45)
            text(11, y + 9, label, ha="left", weight="bold")

        def id_exit(x, start_y, lane_y, color):
            # A separate ID port leaves the VQ box vertically. Its only crossing
            # is a bridge over the purple supervision trunk, well away from the
            # decoder inputs. Four paths terminate at four named collector ports.
            if lane_y <= 252:
                vertices = [(x, start_y), (x, lane_y), (409.3, lane_y),
                            (409.3, lane_y-3.5), (414.7, lane_y-3.5), (414.7, lane_y),
                            (420, lane_y)]
                codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO,
                         MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO]
            else:
                vertices = [(x, start_y), (x, lane_y), (420, lane_y)]
                codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO]
            curve = patches.PathPatch(MplPath(vertices, codes), facecolor="none",
                                      edgecolor=color, lw=.85, zorder=4)
            curve.set_path_effects([patheffects.Stroke(linewidth=3, foreground="white"),
                                    patheffects.Normal()])
            p.ax.add_patch(curve)
            path([(420, lane_y), (429, lane_y)], color=color, lw=.85, zorder=4)
            text(175, lane_y - 5, "ID")

        def component_wave(x, cy, component, modality, color):
            box(x, cy - 10, 46, 20, edge=color)
            text(x + 23, cy - 5, f"{component} 重建")
            t = np.linspace(0, 1, 150)
            # Both components contain structured mixtures. Neither schematic is
            # a pure sine/noise caricature of physiological signal versus noise.
            phase = .2 if component == "S" else 1.0
            if modality == "E":
                wave = (.46*np.sin(2*np.pi*5*t+phase) + .24*np.sin(2*np.pi*11*t)
                        + .18*np.cos(2*np.pi*2*t+phase))
            else:
                wave = (.65*np.sin(2*np.pi*t+phase) + .22*np.sin(2*np.pi*2.3*t+.3))
            p.ax.plot(x+3+40*t, cy+4.2-wave*3.3, color=color, lw=.8, zorder=3)
            if modality == "F":
                other = .5*np.sin(2*np.pi*t+phase+.9)+.1*np.cos(2*np.pi*3*t)
                p.ax.plot(x+3+40*t, cy+4.2+other*2.6, color=color,
                          lw=.8, linestyle=(0, (2, 1.5)), zorder=3)

        # (a): fitting is a functional module, never a loss in the data path.
        # The two raw-observation bypasses are required to construct voltage S
        # and the fixed complete residuals; teacher outputs alone are insufficient.
        section(1, 87, "(a)  离线物理教师与分量目标构造")
        node(11, 34, 55, 28, ["配对观测", "EEG + fNIRS"])
        node(84, 34, 107, 28, ["Tak 离线物理拟合", "共同驱动 · 血流响应"])
        path([(66, 48), (84, 48)])
        node(219, 28, 119, 20, ["原采样率电压目标构造"])
        node(219, 58, 119, 20, ["HbO/HbR 目标构造"])
        path([(191, 42), (204, 42), (204, 38), (219, 38)])
        path([(191, 54), (204, 54), (204, 68), (219, 68)])
        path([(37, 34), (37, 20), (278, 20), (278, 28)])
        text(205, 14, "原始 EEG")
        path([(37, 62), (37, 83), (278, 83), (278, 78)])
        text(132, 76, "原始 HbO/HbR")
        node(364, 33, 105, 42, ["冻结 S/O 分量目标", "按模态与分支对应"],
             edge=purple, face="#F6F1F9")
        path([(338, 38), (364, 38)])
        path([(338, 68), (352, 68), (352, 62), (364, 62)])

        # (b): four private encoding/quantization/decoding paths. Supervision
        # and ID collection have distinct destinations and routing channels.
        section(94, 184, "(b)  双分支离散表示学习")
        text(397, 103, "编码器、码本与解码器各自独立", ha="right")
        # The target trunk is drawn before ID bridges, with no junction at a crossing.
        path([(412, 75), (412, 252)], color=purple, dashed=True, arrow=False, lw=.9)
        text(419, 84, "监督目标", ha="left")
        for x, title in [(32, "本模态输入"), (86, "编码"), (142, "潜变量量化"),
                         (202, "分支解码"), (260, "分量重建"), (333, "训练监督")]:
            text(x, 112, title, weight="bold")
        path([(10, 192), (401, 192)], lw=.45, color="#B3BAC1", arrow=False)
        box(429, 104, 47, 172, edge=line, face="#F1F4F6")
        text(452.5, 144, "多路 token", weight="bold")
        text(452.5, 155, "编排", weight="bold")
        text(452.5, 224, "保留模态")
        text(452.5, 235, "分支身份")
        text(452.5, 246, "时间支持")

        for modality, name, center, lanes in [
            ("E", "EEG", 150, (119, 185)), ("F", "fNIRS", 234, (199, 269))
        ]:
            labels = [name, "电压", "原采样率"] if modality == "E" else [name, "HbO/HbR"]
            node(10, center-17, 43, 34, labels)
            for component, cy, lane, color, face in [
                ("S", center-18, lanes[0], blue, "#F0F5FB"),
                ("O", center+18, lanes[1], orange, "#FCF4EA")
            ]:
                path([(53, center), (59, center), (59, cy), (65, cy)])
                node(65, cy-10, 42, 20, [f"{component} 编码器"], edge=color, face=face)
                path([(107, cy), (120, cy)])
                node(120, cy-10, 44, 20, ["VQ", "量化向量"], edge=color, face=face)
                path([(164, cy), (181, cy)])
                node(181, cy-10, 42, 20, [f"{component} 解码器"], edge=color, face=face)
                path([(223, cy), (237, cy)])
                component_wave(237, cy, component, modality, color)
                path([(283, cy), (310, cy)])
                node(310, cy-8, 47, 16, ["分量监督"], edge=purple, face="#F6F1F9")
                path([(412, cy), (357, cy)], color=purple, dashed=True, lw=.9)
                # The plus node receives decoded waveforms only, with explicit
                # junction dots. Its output has no edge back to a training loss.
                p.ax.plot(294, cy, "o", color=ink, markersize=2, zorder=4)
                path([(294, cy), (294, center-4 if component == "S" else center+4)])
                id_exit(142, cy-10 if component == "S" else cy+10, lane, color)
                text(451.5, lane, f"{modality}–{component}")
            p.ax.add_patch(patches.Circle((294, center), 4, facecolor="white",
                                          edgecolor=ink, lw=.75, zorder=4))
            text(294, center-.2, "+")
            path([(298, center), (309, center)])
            node(309, center-7, 90, 14, ["电压重建检查" if modality == "E" else "观测重建检查"])

        # (c): one named collector feeds the masked reader. The unmasked label
        # branch starts before masking; no decoder waveform enters this route.
        section(296, 65, "(c)  掩码预训练")
        text(469, 305, "冻结 Tokenizer · 输入仅为整数 token", ha="right")
        path([(452.5, 276), (452.5, 291), (7, 291), (7, 342), (13, 342)])
        text(236, 284, "四路离散码流 · 保留各自时间支持")
        p.ax.plot(7, 316, "o", color=ink, markersize=2.2, zorder=5)
        path([(7, 316), (417, 316), (417, 328)], color=purple, dashed=True, lw=.9)
        text(216, 310, "掩码前原 token 标签")
        node(13, 328, 90, 28, ["按时间支持掩码", "S/O 同步 · 嵌入"])
        path([(103, 342), (119, 342)])
        node(119, 328, 97, 28, ["跨模态 Transformer", "共享上下文"])
        path([(216, 342), (232, 342)])
        node(232, 328, 81, 28, ["四路 token 预测", "各自词表分类"])
        path([(313, 342), (364, 342)])
        text(338, 334, "预测概率")
        node(364, 328, 107, 28, ["掩码监督", "遮挡位置交叉熵"], edge=purple, face="#F6F1F9")

        path([(6, 370), (21, 370)])
        text(25, 370, "前向数据流", ha="left")
        path([(91, 370), (106, 370)], color=purple, dashed=True, lw=.9)
        text(110, 370, "目标监督", ha="left")
        box(175, 367, 8, 6, edge=blue, face="#F0F5FB", lw=.7)
        text(187, 370, "S：物理引导分量", ha="left")
        box(284, 367, 8, 6, edge=orange, face="#FCF4EA", lw=.7)
        text(296, 370, "O：观测残差分量", ha="left")
        text(475, 370, "E：EEG；F：fNIRS", ha="right")
        text(6, 381, "EEG 重建保持原采样率；波形仅为示意。O 可含生理相关信息与测量影响。", ha="left")
        caption = (
            "图 1  Tak 教师引导的双分支离散表示与掩码预训练方案。"
            "(a) 离线教师读取配对 EEG 与 HbO/HbR，拟合共同驱动和血流响应。"
            "EEG 电压目标构造同时接收教师约束与原始 EEG，不由低速功率代理直接逆生成电压；"
            "Hb 目标构造同样保留原观测输入。S 为条件模型目标，固定的 O=观测−S，"
            "不随学生当前的 S 预测变化。教师目标只用于分量监督。"
            "(b) EEG、fNIRS 各有 S/O 两条独立路径。VQ 的量化向量只进入本分支解码器，"
            "独立 ID 出口经四条连续路径进入具名的 token 编排模块；跨线处的桥形标记表示"
            "不相连。分量监督同时接收本分支重建和对应冻结目标。加号节点只合成同模态"
            "重建并作重建检查，不表示无损保证，也不作为额外训练损失。Hb 波形的实线和"
            "虚线分别示意 HbO 与 HbR。"
            "(c) 编排保留模态、分支和时间支持；掩码前的原 token 分叉为标签，"
            "模型读取掩码与嵌入后的离散输入，四个预测头分别在各自词表上分类。"
            "教师目标和解码波形不作为预训练输入；掩码按真实时间支持作用于 S/O。"
            "本版按用户选择沿用 2026-10-29 潜变量 VQ 方案供视觉比较，并非原生电压 v2 的"
            "连续预测后波形词典路线，也不声明已完成该完整组合的预训练实验。"
            "字体为 Noto Sans CJK 与 Liberation Sans；当前机器未安装幼圆及 Arial。"
        )
        return finish_publication(p, caption, width_mm=170, dpi=600, design={
            "minimum_font_pt": 7, "main_stroke_pt": 1., "box_stroke_pt": .7,
            "section_stroke_pt": .45, "supervision_stroke_pt": .9,
            "fonts": [cjk_family, "Liberation Sans"],
            "font_note": "YouYuan and Arial unavailable; this export is a visual comparison with font substitutes.",
            "version_choice": "User explicitly retained the previous latent-VQ schematic for visual comparison.",
        })

    if layout == "paper":
        # A publication-size redraw of the user's 2026-10-29 token-first
        # schematic, not a new model generation or a measured-data report.
        # Geometry is in physical points, so font and stroke sizes survive
        # insertion at 180 mm in a double-column manuscript.
        page_width = 180 / 25.4 * 72
        p = Plate("project_framework_paper", "Tak 教师引导的双分支 Tokenizer 与掩码预训练",
                  "", 450, width=page_width)

        def panel(x, y, w, h, title, *, tint, edge=line):
            p.box(x, y, w, h, edge=edge)
            p.ax.add_patch(patches.Rectangle((x + .6, y + .6), w - 1.2, 21,
                                             facecolor=tint, edgecolor="none"))
            p.t(x + 8, y + 11, title, 9.4, boldface=True)

        def cell(x, y, w, h, lines, *, edge=line, face="white", size=8):
            p.box(x, y, w, h, edge=edge, face=face)
            for j, label in enumerate(lines):
                p.t(x + w / 2, y + h * (j + .5) / len(lines), label,
                    size, ha="center")

        def port(x, y, label, color):
            p.ax.add_patch(patches.Circle((x, y), 5.5, facecolor="white",
                                          edgecolor=color, linewidth=1.2, zorder=5))
            label_artist = p.t(x, y - .1, label, 7.3, color=color,
                               ha="center", boldface=True)
            label_artist.set_zorder(6)

        def trace(x, y, w, h, component, modality):
            color = blue if component == "S" else orange
            p.box(x, y, w, h, edge=color)
            t = np.linspace(0, 1, 120)
            if modality == "E":
                q = np.sin(2*np.pi*5*t) * (.65 + .25*np.cos(2*np.pi*t))
                if component == "O":
                    q *= np.exp(-((t-.5)/.25)**2)
            else:
                q = np.exp(-((t-.38)/.2)**2) - .35*np.exp(-((t-.8)/.17)**2)
                if component == "O":
                    q = .5*np.sin(2*np.pi*2*t)
            xx = x + 3 + (w - 6)*t
            p.ax.plot(xx, y + h*.61 - q*h*.24, color=color, lw=1.1)
            if modality == "F":
                p.ax.plot(xx, y + h*.5 + q*h*.19, color=color,
                          lw=1.1, linestyle=(0, (2, 1.5)))

        panel(3, 3, 148, 277, "(a)  Tak 条件教师", tint="#EEE7F5", edge=purple)
        p.t(77, 35, "离线配对支持：90 s", 7.8, ha="center")
        cell(12, 45, 130, 25, ["EEG 电位 + HbO / HbR"], size=8)
        p.arrow([(77, 70), (77, 79)])
        cell(12, 79, 130, 31, ["EEG 功率 / PCA", "Hb 对齐 / 参考"], size=8)
        p.arrow([(77, 110), (77, 118)])
        p.box(12, 118, 130, 44, face="#F5F0FA", edge=purple)
        p.m(77, 126, r"\mathcal{L}_T=\mathcal{L}_{\rm fit}+\lambda_rR_r", 9.1, color=purple)
        p.m(77, 140, r"+\,\lambda_0R_0+\lambda_fR_f", 9.1, color=purple)
        p.t(77, 154, "拟合驱动与四初态；参数固定", 7.6, ha="center")
        p.arrow([(77, 162), (77, 168)], width=1.1)
        p.box(12, 168, 130, 44, face="#FAF8FC", edge=purple)
        p.t(77, 177, "Tak 动力学 · RK4 积分", 8, ha="center", boldface=True)
        for x, w, expression in [(17, 30, r"r(t)"), (59, 31, r"s,\ f"), (102, 35, r"v,\ q")]:
            p.box(x, 187, w, 19, edge=line)
            p.m(x + w/2, 196.5, expression, 9.5)
        p.arrow([(47, 196.5), (58, 196.5)])
        p.arrow([(90, 196.5), (101, 196.5)])
        p.arrow([(32, 206), (32, 222)])
        p.arrow([(119, 206), (119, 222)])
        cell(12, 222, 61, 26, ["EEG 代理读出", "条件电压修正"], size=7.4)
        cell(81, 222, 61, 26, ["HbO/HbR 读出", "时间箱 / 参考"], size=7.4)
        for x in (42, 112):
            p.arrow([(x, 248), (x, 256)], color=purple)
        p.box(12, 256, 130, 18, edge=purple, face="#F5F0FA")
        p.m(71, 265, r"S_m^*,\quad O_m^*=X_m-S_m^*", 9, color=purple)
        port(138, 255, "T", purple)

        panel(159, 3, 348, 277, "(b)  双分支 Tokenizer", tint="#E8EDF3")
        p.t(167, 35, "编码器、码本与解码器均按模态及分支独立", 7.5)
        p.t(497, 35, "HbO / HbR：实线 / 虚线", 7.3, ha="right")
        for x, label in [(184, "观测"), (237, "编码"), (287, "VQ / token"),
                         (337, "解码"), (386, "分量时序"), (456, "分量损失")]:
            p.t(x, 48, label, 7.5, ha="center", boldface=True)

        for modality, name, sub, off in [("E", "EEG", "电位", 0), ("F", "fNIRS", "HbO/HbR", 96)]:
            if off:
                p.segment([(166, 144), (492, 144)], color=line, width=.8)
            p.box(166, 65 + off, 37, 48)
            p.t(184.5, 76 + off, name, 8.2, ha="center", boldface=True)
            p.t(184.5, 90 + off, sub, 7.3, ha="center")
            p.m(184.5, 103 + off, rf"X_{modality}", 9)
            for component, y, color, face in [("S", 61 + off, blue, "#EDF4FB"),
                                              ("O", 94 + off, orange, "#FCF2E8")]:
                cy = y + 12
                p.arrow([(203, 89 + off), (210, 89 + off), (210, cy), (218, cy)])
                cell(219, y, 36, 24, [f"{component} 编码"], edge=color, face=face, size=7.9)
                p.arrow([(255, cy), (265, cy)])
                p.box(266, y, 42, 24, edge=color, face=face)
                p.t(287, y + 7, "VQ", 7.5, ha="center", boldface=True)
                for j in range(4):
                    p.ax.add_patch(patches.Rectangle((272 + j*8, y + 14), 6, 6,
                                                     facecolor=color, edgecolor=ink, lw=.7))
                p.arrow([(308, cy), (318, cy)])
                cell(319, y, 36, 24, [f"{component} 解码"], edge=color, face=face, size=7.9)
                p.arrow([(355, cy), (365, cy)])
                trace(366, y, 41, 24, component, modality)
                p.arrow([(407, cy), (420, cy)])
                p.box(421, y, 70, 24, edge=purple, face="#F7F3FB")
                p.m(456, cy, rf"\mathcal{{L}}_{{{modality},{component}}}(\hat {component}_{modality},{component}_{modality}^*)",
                    8.8, color=purple)
                p.ax.plot(313, cy, "o", color=ink, markersize=2.6)
            # K identifies only integer IDs; decoded waveforms never feed the reader.
            p.segment([(313, 73 + off), (313, 123 + off)], width=1.05)
            p.arrow([(313, 123 + off), (302, 123 + off)], width=1.05)
            port(196, 129 + off, "K", ink)
            p.m(265, 129 + off, rf"K_{modality}=(k_{{{modality},S}},k_{{{modality},O}})", 8.5)
            p.m(438, 127 + off, rf"\hat X_{modality}=\hat S_{modality}+\hat O_{modality}", 8.4)
            p.t(438, 137 + off, "相加仅作重建诊断", 7.2, ha="center")

        port(499, 48, "T", purple)
        p.segment([(499, 54), (499, 202)], color=purple, dashed=True, width=1.1)
        for y in (73, 106, 169, 202):
            p.arrow([(499, y), (492, y)], color=purple, dashed=True, width=1.1)
        p.t(167, 238, "VQ：EMA 更新码本；梯度直通；编码—码字承诺项", 7.3)
        p.box(167, 246, 331, 27, face="#F7F3FB", edge=purple)
        p.m(267, 255, r"\mathcal{L}_m=\mathcal{L}_{m,S}+\lambda_O\mathcal{L}_{m,O}+\beta\mathcal{L}_{m,\rm VQ}", 9, color=purple)
        p.m(438, 255, r"\mathcal{L}_{\rm tok}=\frac{\mathcal{L}_E+\mathcal{L}_F}{2}", 9, color=purple)
        p.t(332, 267, "分量损失：有效点上的均方误差，按训练观测尺度归一化", 7.3, ha="center")

        panel(3, 288, 504, 136, "(c)  掩码预训练", tint="#E8F0EF")
        p.t(496, 299, "冻结 Tokenizer；输入仅为四路整数 token", 7.5, ha="right")
        p.arrow([(18, 319), (18, 327)])
        p.arrow([(24, 317), (485, 317), (485, 343)], color=purple, dashed=True, width=1.1)
        port(18, 317, "K", ink)
        p.t(442, 330, "原 token 标签", 7.6, color=purple, ha="center")
        p.box(12, 327, 86, 73)
        for row, (name, color) in enumerate([(r"E_S", blue), (r"E_O", orange),
                                             (r"F_S", blue), (r"F_O", orange)]):
            y = 338 + row*13
            p.m(28, y, name, 8.5)
            for j in range(5):
                masked = j in (2, 3)
                p.ax.add_patch(patches.Rectangle((43 + j*10, y - 4), 8, 8,
                    facecolor="#CBD0D5" if masked else color,
                    edgecolor=ink, linewidth=.7, hatch="//" if masked else None))
        p.t(55, 392, "时间块 / 模态块", 7.4, ha="center")
        p.t(55, 412, "S/O 同步遮挡", 7.5, ha="center")
        p.arrow([(98, 365), (108, 365)])
        cell(109, 340, 75, 51, ["Token embedding", "+ 时间位置编码", "+ 模态 / 分支编码"],
             face="#F3F5F7", size=7.7)
        p.arrow([(184, 365), (194, 365)])
        p.box(195, 331, 112, 68, face="#F0F5F4")
        p.t(251, 341, "Transformer × 2", 9, ha="center", boldface=True)
        cell(201, 351, 100, 18, ["LN → 多头注意力 → 残差"], size=7.4)
        p.arrow([(251, 369), (251, 376)])
        cell(201, 376, 100, 18, ["LN → 前馈网络 → 残差"], size=7.4)
        p.segment([(301, 385), (304, 385), (304, 365), (307, 365)], width=1.1)
        p.arrow([(307, 365), (318, 365)])
        cell(319, 343, 63, 42, ["四路预测头", "各自词表概率"], size=8)
        p.arrow([(382, 365), (394, 365)])
        p.box(395, 343, 101, 42, edge=purple, face="#F7F3FB")
        p.m(445.5, 355, r"\mathcal{L}_{\rm mask}", 12, color=purple)
        p.t(445.5, 373, "遮挡位置交叉熵", 7.8, ha="center")
        p.arrow([(307, 365), (312, 365), (312, 412), (318, 412)], color=gray, dashed=True, width=1.05)
        p.box(319, 404, 63, 16, dashed=True, edge=gray)
        p.t(350.5, 412, "任务头 → 输出", 7.3, ha="center")
        p.arrow([(382, 412), (394, 412)], color=gray, dashed=True, width=1.05)
        p.box(395, 404, 101, 16, dashed=True, edge=gray)
        p.t(445.5, 412, r"$\mathcal{L}_{\rm task}$ · 下游接口", 8, ha="center")

        p.arrow([(6, 435), (22, 435)], width=1.2)
        p.t(27, 435, "前向数据流", 7.5)
        p.arrow([(98, 435), (114, 435)], color=purple, dashed=True, width=1.1)
        p.t(119, 435, "冻结目标监督", 7.5, color=purple)
        p.t(207, 435, "S：条件分量", 7.6, color=blue)
        p.t(290, 435, "O：固定残差", 7.6, color=orange)
        p.t(374, 435, "同字母端口相连：T 目标 / K token", 7.3)
        p.t(6, 446, "E = EEG，F = fNIRS；波形仅作结构示意。教师目标是条件估计，保留原测量坐标。", 7.2)
        caption = (
            "图 1  Tak 教师引导的双分支 Tokenizer 与掩码预训练。"
            "(a) 教师在离线 90 秒配对支持上拟合共享驱动 r(t) 与四个初态；Tak 参数固定，"
            "s、f 为血流响应状态，v、q 为血容量与脱氧状态，RK4 为四阶 Runge–Kutta 积分。"
            "Rr、R0、Rf 分别约束驱动平滑、初态先验和血流先验。EEG 经条件电压修正、"
            "Hb 经时间箱及参考变换后形成冻结目标 T。S* 是条件模型估计，O*=X−S*。"
            "(b) m∈{E,F} 分别表示 EEG 与 fNIRS；四个分支均为编码、VQ、时序解码路径，"
            "参数各自独立。VQ 为向量量化，EMA 为指数移动平均；承诺项是编码向量与停止"
            "梯度的选中码字之间的均方差。分量误差先在有效点上按训练观测尺度归一化，"
            "再等权汇总两个模态；分量之和只用于重建诊断。Hb 时序图中实线与虚线分别"
            "示意 HbO 与 HbR。(c) K 汇集四路冻结整数 token；灰色斜线格为遮挡位置，"
            "S/O 同步遮挡。原 token 只作为遮挡交叉熵的标签，教师目标与解码波形不输入"
            "预训练模型。LN 为层归一化，两个 Transformer 层共享跨模态上下文，四路预测头"
            "各自进行词表分类；任务头与任务标签损失仅标为下游接口。本图沿用 2026-10-29"
            "的先量化后解码架构表达，不声明该完整电压组合已经完成实验验证。"
        )
        return finish_publication(p, caption, width_mm=180, dpi=300, design={
            "minimum_font_pt": 7.2, "main_stroke_pt": 1.25, "box_stroke_pt": 1.1,
        })

    if layout == "token_first":
        # This view follows the user's requested architecture, not the experimental
        # continuous-prediction-then-waveform-quantization deployment projection.
        from matplotlib import patheffects
        p = Plate("project_framework_token_first", "Tak 教师引导的双分支 Tokenizer 与预训练模型",
                  "编码 → 量化为 token → 解码为时序；教师监督、VQ 约束与掩码预训练通过实际数据流连接。",
                  1840, width=2800)

        def neural_block(x, y, w, h, title, color, *, decoder=False):
            points = ([(x,y+12),(x+w,y),(x+w,y+h),(x,y+h-12)] if decoder else
                      [(x,y),(x+w,y+12),(x+w,y+h-12),(x,y+h)])
            p.ax.add_patch(patches.Polygon(points,facecolor="#F4F4F4",edgecolor="#7F7F7F",lw=1.15))
            p.t(x+w/2,y+23,title,14,ha="center",boldface=True)
            layers = [(x+35,[y+53,y+76]),(x+w/2,[y+46,y+65,y+84]),(x+w-35,[y+53,y+76])]
            for (a,ys),(b,zs) in zip(layers,layers[1:]):
                for ay in ys:
                    for by in zs:
                        p.segment([(a,ay),(b,by)],color="#999999",width=.65)
            for a,ys in layers:
                for yy in ys:
                    p.ax.add_patch(patches.Circle((a,yy),5,facecolor=color,edgecolor="white",lw=.5,zorder=4))

        # Each modality remains a complete encoder -> hard token -> temporal decoder path.
        for modality,title,offset,is_eeg in [("E","EEG Tokenizer",0,True),("F","fNIRS Tokenizer",580,False)]:
            p.box(800,150+offset,1935,460,edge="#7C8991",dashed=True)
            p.t(830,180+offset,title,20,boldface=True)
            p.t(2705,180+offset,"编码器 · 码本 · 解码器按模态与分支独立",12,color=gray,ha="right")
            p.box(1140,200+offset,330,285,face="#FAFAFA",edge="#7D898F")
            p.t(1305,220+offset,"VQ → token 序列",15.5,ha="center",boldface=True)
            p.t(1305,474+offset,"EMA 码本 · 梯度直通",11,ha="center",color=gray)
            for component,base_y,color in [("S",260,blue),("O",440,orange)]:
                yy=base_y+offset
                neural_block(885,yy-48,190,96,f"{component} 编码器",color)
                p.arrow([(1085,yy),(1129,yy)])
                rgb=np.asarray(colors.to_rgb(color))
                for row in range(3):
                    for col in range(3):
                        alpha=[.32,.55,.82][(row+col)%3]
                        p.box(1165+col*16,yy-24+row*16,14,14,
                              face=tuple(1-alpha*(1-rgb)),edge="white")
                p.box(1180,yy-9,16,16,face="none",edge=ink)
                p.arrow([(1223,yy),(1262,yy)],color=color,width=1)
                for token in range(5):
                    alpha=[.3,.47,.65,.85,.57][token]
                    p.box(1273+token*33,yy-20,28,40,face=tuple(1-alpha*(1-rgb)),edge="white")
                p.arrow([(1480,yy),(1529,yy)])
                neural_block(1540,yy-48,190,96,f"{component} 解码器",color,decoder=True)
                p.arrow([(1740,yy),(1788,yy)])
                kind=("eeg" if component=="S" else "eeg_residual") if is_eeg else (
                    "hb_component" if component=="S" else "hb_residual")
                p.wave(1800,yy-33,205,66,kind,color=color)
                p.t(1800,yy-59,f"{component} 时序",14,color=color)
                p.arrow([(2016,yy),(2103,yy)])
                p.box(2115,yy-38,210,76,edge=purple,face="#FCFAFE")
                p.t(2220,yy-19,"分量重建",12,ha="center",color=gray)
                p.m(2220,yy+13,rf"\mathcal{{L}}_{component}",24,color=purple)

            # Sum reconstruction is shown as an observation-space diagnostic, not
            # silently added to the registered S/O/VQ training objective.
            cy=350+offset
            p.arrow([(1902,293+offset),(1902,332+offset)])
            p.arrow([(1902,407+offset),(1902,368+offset)])
            p.ax.add_patch(patches.Circle((1902,cy),17,facecolor="white",edgecolor=ink,lw=1))
            p.t(1902,cy-1,"+",18,ha="center")
            p.arrow([(1921,cy),(1983,cy)])
            p.wave(1995,cy-26,130,52,"eeg" if is_eeg else "hb")
            p.t(2060,cy-46,"总观测",11.5,ha="center",color=gray)
            p.t(2060,cy+44,"重建诊断",11,ha="center",color=gray)

            p.box(1170,510+offset,270,58,edge=purple,face="#FCFAFE")
            p.m(1229,540+offset,r"\mathcal{L}_{\rm VQ}",21,color=purple)
            p.t(1349,540+offset,"承诺项（S/O）",11,ha="center",color=gray)
            p.arrow([(1305,485+offset),(1305,510+offset)])
            p.box(2435,315+offset,280,115,edge=purple,face="#F6F0FA")
            p.m(2575,336+offset,rf"\mathcal{{L}}_{modality}",22,color=purple)
            p.m(2575,371+offset,r"\mathcal{L}_S+\lambda_O\mathcal{L}_O",20,color=purple)
            p.m(2575,408+offset,r"+\,\beta\mathcal{L}_{\rm VQ}",19,color=purple)
            p.arrow([(2220,298+offset),(2220,350+offset),(2424,350+offset)],color=purple)
            p.arrow([(2220,402+offset),(2220,397+offset),(2424,397+offset)],color=purple)
            p.arrow([(1440,539+offset),(1490,539+offset),(1490,590+offset),
                     (2575,590+offset),(2575,430+offset)],color=purple)

        # Raw observations fan out, retaining the same-modality student boundary.
        for title,cy,kind in [("EEG 电位",350,"eeg"),("fNIRS",1240,"hb")]:
            p.t(170,cy-62,title,18,ha="center",boldface=True)
            p.wave(80,cy-35,180,70,kind)
            if kind=="hb":p.t(170,cy+57,"HbO / HbR",12,ha="center",color=gray)
        p.arrow([(270,350),(840,350)])
        p.arrow([(270,1240),(750,1240),(750,930),(840,930)])
        for cy in (350,930):
            for branch_y in (cy-90,cy+90):
                p.arrow([(840,cy),(840,branch_y),(873,branch_y)])
        p.ax.plot(340,350,"o",color=ink,markersize=3)
        p.ax.plot(720,1240,"o",color=ink,markersize=3)

        # Expanded mechanistic teacher: observations -> conditional fit -> RK4
        # dynamics/readouts -> frozen targets. It contains no invented neural ODE.
        p.box(60,490,650,650,face="#FCFAFE",edge=purple,dashed=True)
        p.t(90,517,"Tak 条件教师",18,boldface=True)
        p.t(680,517,"离线 90 秒配对支持",11.5,color=gray,ha="right")
        p.block(90,550,590,65,"配对观测变换","EEG 功率 / PCA  ·  Hb 对齐 / 参考",face="white")
        p.arrow([(385,615),(385,640)])
        p.box(90,640,590,72,face="#F4EFF8",edge=purple)
        p.m(385,662,r"\mathcal{L}_T=\mathcal{L}_{\rm fit}+\lambda_rR_r+\lambda_0R_0+\lambda_fR_f",19,color=purple)
        p.t(385,697,"观测误差 · 驱动平滑 · 初态先验 · 血流先验",11,ha="center",color=gray)
        p.arrow([(155,712),(155,762)],color=gray,dashed=True)
        p.t(145,738,"拟合",10.5,color=gray,ha="right")
        p.box(245,723,455,151,edge="#8E969D",dashed=True)
        p.t(473,744,"Tak 动力学 · RK4 积分",13,ha="center",boldface=True)
        p.box(90,763,135,65,face="white")
        p.t(157,782,"共享驱动",12,ha="center")
        p.m(157,809,r"r(t)",18)
        p.arrow([(225,796),(261,796)])
        p.box(270,763,170,65,face=fill)
        p.t(355,782,"血流响应",12.5,ha="center")
        p.m(355,810,r"s,\ f",18)
        p.arrow([(440,796),(480,796)])
        p.box(490,763,190,65,face=fill)
        p.t(585,782,"血容量 / 脱氧",12,ha="center")
        p.m(585,810,r"v,\ q",18)
        p.t(355,843,"参数固定",10.5,color=gray,ha="center")
        p.t(355,861,"四初态与驱动共同拟合",10,color=gray,ha="center")
        p.arrow([(157,828),(157,900)])
        p.arrow([(585,828),(585,900)])
        p.box(90,910,230,35)
        p.t(205,928,"EEG 代理读出",12.5,ha="center")
        p.box(90,962,230,35)
        p.t(205,980,"条件电压修正",12.5,ha="center")
        p.arrow([(205,945),(205,962)])
        p.box(445,910,235,35)
        p.t(563,928,"HbO/HbR 读出",12.5,ha="center")
        p.box(445,962,235,35)
        p.t(563,980,"时间箱 / 参考",12.5,ha="center")
        p.arrow([(563,945),(563,962)])
        p.arrow([(90,928),(75,928),(75,677),(90,677)],color=gray,dashed=True,width=1)
        p.arrow([(680,980),(695,980),(695,677),(680,677)],color=gray,dashed=True,width=1)
        p.arrow([(205,997),(205,1020)])
        p.arrow([(563,997),(563,1020)])
        p.box(90,1020,590,80,edge=purple,face="white")
        p.t(385,1040,"冻结分量目标",14,ha="center",boldface=True)
        p.m(385,1073,r"S^*,\qquad O^*=X-S^*",22,color=purple)
        p.t(385,1120,"条件模型目标；保留原测量坐标",11,ha="center",color=gray)

        # One target bundle routes to the matching modality/branch reconstruction loss.
        p.segment([(680,1060),(735,1060),(735,650),(2765,650)],color=purple,dashed=True)
        p.segment([(2765,260),(2765,1020)],color=purple,dashed=True)
        p.t(2410,630,"冻结监督目标（按模态与分支对应）",12,color=purple,ha="center")
        for yy,component in [(260,"S"),(440,"O"),(840,"S"),(1020,"O")]:
            p.arrow([(2765,yy),(2333,yy)],color=purple,dashed=True)
            p.m(2370,yy-20,rf"{component}^*",18,color=purple)
        # Connect both measured modalities to the teacher's observation transform,
        # rather than terminating the Hb input near its frozen output targets.
        p.arrow([(340,350),(340,550)])
        hb_input,=p.ax.plot([720,720],[1240,585],color=ink,lw=1.25,zorder=5)
        hb_input.set_path_effects([patheffects.Stroke(linewidth=4,foreground="white"),patheffects.Normal()])
        p.arrow([(720,585),(680,585)])

        # Pretraining uses four distinct vocabularies and explicit stream identity.
        p.box(65,1350,2670,390,edge="#6E8787",dashed=True)
        p.t(95,1380,"预训练模型",20,boldface=True)
        p.t(1900,1380,"预训练时冻结 Tokenizer；输入仅为整数 token",12,color=gray,ha="right")
        p.box(105,1420,320,260,edge="#9CA4AA",face="white")
        p.t(265,1445,"时间块 / 模态块遮挡",14,ha="center",boldface=True)
        for stream,(name,color) in enumerate([(r"E_S",blue),(r"E_O",orange),(r"F_S",blue),(r"F_O",orange)]):
            yy=1480+stream*43
            p.m(145,yy,name,16)
            for j in range(5):
                p.box(185+j*40,yy-13,33,27,face="#D5D7DA" if j in (2,3) else color,edge="white")
        p.t(265,1662,"S / O 同步遮挡",11.5,ha="center",color=gray)
        p.arrow([(425,1540),(480,1540)])
        p.box(490,1450,300,190,edge=line,face=fill)
        p.t(640,1480,"Token embedding",16,ha="center",boldface=True)
        p.t(640,1530,"＋ 时间位置编码",14,ha="center")
        p.t(640,1575,"＋ 模态 / 分支编码",14,ha="center")
        p.arrow([(790,1540),(853,1540)])
        p.box(865,1410,590,260,edge="#74858B",face="#F5F7F8")
        p.t(1160,1440,"Transformer × 2",18,ha="center",boldface=True)
        p.box(900,1470,520,66,edge=line,face="white")
        p.t(1160,1503,"LayerNorm → 多头注意力 → 残差",14,ha="center")
        p.arrow([(1160,1536),(1160,1570)])
        p.box(900,1570,520,66,edge=line,face="white")
        p.t(1160,1603,"LayerNorm → 前馈网络 → 残差",14,ha="center")
        p.t(1160,1653,"共享跨模态上下文",11.5,color=gray,ha="center")
        p.arrow([(1420,1603),(1438,1603),(1438,1540),(1455,1540)])
        p.segment([(1455,1540),(1484,1540)])
        p.ax.plot(1484,1540,"o",color=ink,markersize=3)
        p.arrow([(1484,1540),(1484,1505),(1528,1505)])
        p.box(1540,1450,270,110,edge=line,face=fill)
        p.t(1675,1476,"四路 token 预测头",15,ha="center",boldface=True)
        p.t(1675,1524,"各自词表分类",13,ha="center",color=gray)
        p.arrow([(1810,1505),(1860,1505)])
        p.box(1870,1450,200,110,edge=line)
        p.t(1970,1470,"token 概率",13,ha="center")
        for j,hh in enumerate([18,45,31,62,23]):
            p.box(1900+j*29,1550-hh,18,hh,edge="white",face=blue)
        p.arrow([(2070,1505),(2168,1505)])
        p.box(2180,1475,330,105,edge=purple,face="#FCFAFE")
        p.m(2345,1503,r"\mathcal{L}_{\rm mask}",25,color=purple)
        p.t(2345,1545,"遮挡 token 交叉熵（S/O 求和）",12,ha="center",color=gray)
        p.box(2180,1400,330,47,edge=purple)
        p.t(2345,1424,"原 token（监督标签）",13,ha="center",color=purple)
        p.arrow([(2345,1447),(2345,1475)],color=purple,dashed=True)

        p.arrow([(1484,1540),(1484,1635),(1528,1635)],color=gray,dashed=True)
        p.box(1540,1600,270,70,edge=gray,dashed=True)
        p.t(1675,1635,"任务头（接口）",14,ha="center")
        p.arrow([(1810,1635),(1882,1635)],color=gray,dashed=True)
        p.box(1895,1600,175,70,edge=gray,dashed=True)
        p.t(1982,1635,"任务输出",13,ha="center")
        p.arrow([(2070,1635),(2168,1635)],color=gray,dashed=True)
        p.box(2180,1610,330,80,edge=gray,dashed=True)
        p.m(2345,1640,r"\mathcal{L}_{\rm task}",23)
        p.t(2345,1672,"监督任务损失（接口）",11.5,ha="center",color=gray)
        p.t(2610,1618,"任务标签",12,color=gray,ha="center")
        p.arrow([(2580,1650),(2510,1650)],color=gray,dashed=True)
        p.t(95,1714,"灰色 token 为被遮挡位置；任务头为下游接口。",11.5,color=gray)

        # The bus carries typed integer streams. Only the four explicit dots are
        # junctions; white under-strokes make unrelated supervision crossings clear.
        bus,=p.ax.plot([1505,1505],[260,1320],color="#405B69",lw=1.55,zorder=5)
        bus.set_path_effects([patheffects.Stroke(linewidth=5,foreground="white"),patheffects.Normal()])
        for yy in (260,440,840,1020):p.ax.plot(1505,yy,"o",color="#405B69",markersize=4,zorder=6)
        p.arrow([(1505,1320),(50,1320),(50,1540),(105,1540)],color="#405B69")
        p.t(1070,1298,"冻结后的四路整数 token",13,color="#405B69",ha="center")
        p.arrow([(1505,1320),(2345,1320),(2345,1400)],color=purple,dashed=True)
        p.t(1950,1298,"遮挡位置的原 token 标签",12,color=purple,ha="center")

        p.arrow([(80,1780),(140,1780)])
        p.t(155,1780,"前向数据流",12)
        p.arrow([(440,1780),(500,1780)],color=purple,dashed=True)
        p.t(515,1780,"冻结目标监督",12,color=purple)
        p.t(890,1780,"S：条件分量",13,color=blue)
        p.t(1210,1780,"O：固定残差",13,color=orange)
        p.m(1970,1780,r"\mathcal{L}_{\rm tok}=(\mathcal{L}_E+\mathcal{L}_F)/2",21,color=purple)
        p.t(80,1819,"VQ 承诺项：编码向量与选中码字的均方差（码字停止梯度）；码本采用 EMA 更新。",11,color=gray)
        p.t(2735,1819,"S/O 在有效时刻按训练观测尺度计算归一化误差；波形仅为结构示意。",11,color=gray,ha="right")
        return finish([p.save()], [
            "按用户指定的先量化、后时序解码路线绘制。教师、Tokenizer 与预训练模型各自优化；S/O 重建、VQ 承诺项和掩码交叉熵分别标出，码本采用 EMA 更新。电压目标衔接与任务头为结构表达，本图不声明该完整组合已完成实验验证。"
        ], "project_framework.pdf")

    if layout == "flow":
        p = Plate("project_framework_flow", "Tak 引导的双分量 Tokenizer",
                  "同模态独立预测与量化，冻结教师目标监督分量学习。", 1180)

        # Both observations fan out to their own student and the shared teacher.
        for title, cy, kind, detail in [
            ("EEG",340,"eeg","原始电位"), ("fNIRS",915,"hb","HbO / HbR")]:
            p.t(170,cy-65,title,20,ha="center",boldface=True)
            p.wave(80,cy-35,180,70,kind)
            p.t(170,cy+62,detail,13,color=gray,ha="center")

        # The two modality enclosures describe modules, not chronological stages.
        for title, offset, is_eeg in [("EEG Tokenizer",0,True),("fNIRS Tokenizer",575,False)]:
            p.box(450,160+offset,1465,330,edge="#78858B",dashed=True)
            p.t(480,185+offset,title,18,boldface=True)
            p.t(1885,185+offset,"S / O 分支独立",12,color=gray,ha="right")
            p.box(855,211+offset,275,245,face="white",edge="#A1A1A1")
            p.t(992,232+offset,"预测分量",14,ha="center")
            p.box(1250,211+offset,600,245,face="#FAFAFA",edge="#7D898F")
            p.t(1550,235+offset,"量化 → token 序列",17,ha="center",boldface=True)
            for component, base_y, color, pale in [
                ("S",275,blue,"#EFF5FB"),("O",405,orange,"#FBF2E8")]:
                yy = base_y+offset
                p.box(535,yy-39,235,78,face=pale,edge=color)
                p.t(652,yy-11,f"{component} 分支",16,color=color,ha="center",boldface=True)
                p.t(652,yy+20,"编码与分量预测",12,ha="center",color=gray)
                p.arrow([(781,yy),(884,yy)])
                kind = ("eeg" if component=="S" else "eeg_residual") if is_eeg else (
                    "hb_component" if component=="S" else "hb_residual")
                p.wave(895,yy-25,200,50,kind,color=color)
                p.arrow([(1106,yy),(1238,yy)])
                p.t(1290,yy,component,17,color=color,ha="center",boldface=True)
                rgb = np.asarray(colors.to_rgb(color))
                for token in range(8):
                    alpha = [.3,.52,.72,.94][token%4]
                    p.box(1350+token*49,yy-24,43,48,
                          face=tuple(1-alpha*(1-rgb)),edge="white")
                p.t(1770,yy,"…",18,color=color,ha="center")
            p.t(1550,443+offset,"词表拟合训练集预测",10.5,color=gray,ha="center")

        # Draw these edges after the module enclosures so their input ports remain visible.
        for cy in (340,915):
            p.arrow([(270,cy),(484,cy)])
            p.ax.plot(305,cy,"o",color=ink,markersize=3)
            for branch_y in (cy-65,cy+65):
                p.arrow([(484,cy),(484,branch_y),(523,branch_y)])
        p.arrow([(305,340),(305,590),(330,590)])
        p.arrow([(305,915),(305,665),(330,665)])

        # Loss reads the paired S/O predictions, before quantization. The targets
        # are frozen, so there is no teacher input or teacher gradient in a student.
        p.box(330,525,350,190,face="#F7F3FA",edge=purple,dashed=True)
        p.t(505,550,"Tak 教师与目标构建",16,ha="center",boldface=True)
        p.t(505,583,"配对拟合 · 冻结目标",12,color=gray,ha="center")
        p.t(505,620,"EEG：电压修正",14,ha="center")
        p.t(505,669,"fNIRS：Hb 预测",14,ha="center")

        for y,title in [(535,"EEG loss"),(635,"fNIRS loss")]:
            p.box(900,y,350,90,face="#FCFAFE",edge=purple)
            p.t(1075,y+19,title,13,ha="center")
            p.m(1075,y+55,r"\frac{\mathcal{L}_S+\mathcal{L}_O}{2}",23,color=purple)
        p.arrow([(995,456),(995,535)])
        p.arrow([(995,786),(995,725)])
        p.arrow([(680,620),(745,620),(745,575),(900,575)],color=purple,dashed=True)
        p.arrow([(680,669),(775,669),(775,675),(900,675)],color=purple,dashed=True)
        p.t(815,553,"EEG 目标",11.5,color=purple,ha="center")
        p.t(835,651,"Hb 目标",11.5,color=purple,ha="center")

        p.t(1550,586,"分量 loss：归一化 MSE",16,ha="center",color=purple)
        p.t(1550,626,"按训练分量方差归一化，S / O 等权",12,ha="center",color=gray)

        p.arrow([(80,1120),(143,1120)])
        p.t(158,1120,"数据流",12)
        p.arrow([(400,1120),(463,1120)],color=purple,dashed=True)
        p.t(478,1120,"冻结目标监督",12,color=purple)
        p.t(940,1120,"S：条件分量",13,color=blue)
        p.t(1300,1120,"O：固定残差",13,color=orange)
        p.t(1915,1158,"示意波形 · 量化后解码省略",10,color=gray,ha="right")
        return finish([p.save()], [
            "统一数据流图。两模态各自预测 S/O 分量并量化为 token；O 目标为测量减冻结 S 目标。教师目标与量化前预测计算归一化 MSE，S/O 等权。量化后的查表与合成省略。"
        ], "project_framework.pdf")

    # Main plate: the deployed predictor/word lookup path, not historical latent VQ.
    p = Plate("01_project_framework", "图组 1A  |  Tak 条件分解与双分量 Tokenizer：完整框架",
              "离线配对观测构建条件目标；各模态独立预测、离散编码与解码；S 为条件分量，O 为固定残差。", 1440)
    p.section(145, "(a)  离线构建并冻结监督目标")
    p.box(65, 185, 235, 165)
    p.wave(82, 202, 91, 51, "eeg")
    p.wave(188, 202, 94, 51, "hb")
    p.t(182, 277, "同步 EEG + fNIRS", 13, ha="center")
    p.t(182, 310, "Single-Trial / Simultaneous", 10.5, color=gray, ha="center")
    p.arrow([(308,265),(344,265)])
    p.block(355,185,320,165,"统一观测坐标",
            "EEG：五频带功率 → 10 Hz 代理\nHbO/HbR：统一缓存 → 10 Hz\nPCA、参考和尺度按合同冻结")
    p.arrow([(684,265),(725,265)])
    p.box(735,185,375,165,face=fill)
    p.t(922,215,"固定参考 Tak 教师",15,ha="center")
    p.m(922,256,r"r(t)\ \longrightarrow\ (s,f,v,q)\ \longrightarrow\ \widetilde{\mathbf{u}}(t)",20)
    p.t(922,296,"90 秒配对支持：拟合驱动与四初态",11.5,ha="center",color=gray)
    p.t(922,326,"保留结果复用；电位版不新增 SSM 拟合",10.5,ha="center",color=gray)
    p.arrow([(1118,265),(1166,265)])
    p.box(1177,179,738,177)
    p.t(1199,205,"20 秒目标  [0,20) s",14,boldface=True)
    p.t(1199,247,"EEG",12.5)
    p.m(1530,247,r"S_E^*=\mathcal{C}(X_E;\widetilde u_E),\qquad O_E^*=X_E-S_E^*",20,color=purple)
    p.t(1199,293,"fNIRS",12.5)
    p.m(1530,293,r"S_F^*=\widetilde{\mathbf{u}}_F,\qquad O_F^*=X_F-S_F^*",20,color=purple)
    p.t(1199,332,r"$\mathcal{C}$：冻结代理引导的电压幅度修正；两类目标均为条件估计。",10.5,color=gray)
    p.rule(386)
    p.section(420,"(b)  单模态独立分词与整数解码")
    p.t(1915,420,"E：EEG；F：fNIRS；四个分支各有独立预测器与词表",11.5,color=gray,ha="right")
    for x,label in [(158,"同模态输入"),(432,"连续分量预测"),(657,"预测波形片段"),
                    (844,"波形空间词表"),(1062,"整数 token 序列"),(1297,"整数查表与合成"),
                    (1525,"分量重建"),(1810,"完整测量重建")]:
        p.t(x,450,label,12,ha="center")

    for modality, center, title, levels, kind, context in [
        ("E",610,"EEG",4,"eeg","200 Hz · 单电极 2 秒上下文"),
        ("F",955,"fNIRS",2,"hb","10 Hz · 6 秒；红 HbO / 绿 HbR")]:
        upper, lower = center-70, center+70
        p.t(77,center-115,title,16,boldface=True)
        p.wave(77,center-42,175,84,kind)
        p.m(164,center+66,rf"X_{{{modality}}}",19)
        p.t(164,center+100,context,10.5,color=gray,ha="center")
        p.t(164,center+126,"另含有效掩码与已知通道身份",10,color=gray,ha="center")
        p.segment([(253,center),(285,center)])
        p.ax.plot(285,center,"o",color=ink,markersize=3)
        for component, yy, color in [("S",upper,blue),("O",lower,orange)]:
            p.arrow([(285,center),(285,yy),(335,yy)])
            p.predictor(343,yy,modality,component)
            p.m(435,yy-58,rf"{component}_{{{modality}}}^*\ \dashrightarrow\ \mathcal{{L}}_{{{modality}}}^{{{component}}}",16,color=purple)
            p.arrow([(435,yy-45),(435,yy-32)],color=purple,dashed=True)
            p.arrow([(537,yy),(580,yy)])
            wave_kind = ("eeg" if component=="S" else "eeg_residual") if modality=="E" else ("hb_component" if component=="S" else "hb_residual")
            p.wave(590,yy-28,135,56,wave_kind,color=color)
            p.m(657,yy-52,rf"\widetilde{{{component}}}_{{{modality},p}}",18,color=color)
            p.arrow([(734,yy),(779,yy)])
            p.book(789,yy-33,115,66,color,levels)
            p.m(846,yy-57,rf"Q_{{{modality}}}^{{{component}}}",18,color=color)
            p.arrow([(917,yy),(983,yy)])
            p.tokens(1001,yy-(26 if levels==4 else 13),color,levels)
            p.m(1062,yy-55,rf"\mathbf{{k}}_{{{modality},p}}^{{{component}}}",18,color=color)
            p.t(1062,yy+53,"列：时间片；行：量化层",9.7,color=gray,ha="center")
            p.arrow([(1134,yy),(1208,yy)])
            p.box(1220,yy-36,155,72,face=fill)
            p.m(1297,yy-13,rf"\sum_\ell c_{{{modality},\ell,k_\ell}}^{{{component}}}",16)
            p.t(1297,yy+24,"固定码字 · 拼接" if modality=="E" else "码字 · 固定合成",10.5,color=gray,ha="center")
            p.arrow([(1386,yy),(1443,yy)])
            p.wave(1453,yy-28,145,56,wave_kind,color=color)
            p.m(1525,yy-54,rf"\widehat{{{component}}}_{{{modality}}}",20,color=color)
            p.arrow([(1607,yy),(1663,yy),(1663,center+(-18 if component=="S" else 18))])
        p.ax.add_patch(patches.Circle((1663,center),17,facecolor="white",edgecolor=ink,lw=1))
        p.t(1663,center-1,"+",19,ha="center")
        p.arrow([(1683,center),(1724,center)])
        p.wave(1734,center-37,174,74,kind)
        p.m(1821,center-62,rf"\widehat X_{{{modality}}}=\widehat S_{{{modality}}}+\widehat O_{{{modality}}}",19)
        p.t(1818,center+66,"对齐原测量坐标",11,color=gray,ha="center")
        if modality=="E":
            p.t(1050,center+155,"100 ms 输出 / 步幅；每分支 4 层；双分支合计 560 bit/s / 电极",11,color=gray,ha="center")
        else:
            p.t(1050,center+149,"2 秒输出 / 步幅；每分支 2 层；双分支合计 14 bit/s / Hb 通道对",11,color=gray,ha="center")
            p.t(1818,center+104,"Hb：逐分支固定 100 ms 高斯合成",10,color=gray,ha="center")
            p.t(1910,813,"Hb 分量图：实线 HbO，虚线 HbR",10.5,color=gray,ha="right")
    p.rule(786,True)
    p.rule(1130)
    p.section(1155,"(c)  训练、验证与后续表示接口")
    p.block(65,1190,575,137,"训练顺序与输入边界",
            "冻结目标 → 独立连续监督 → 训练预测拟合词表\n部署：同模态输入 → 整数 ID → 固定码字\n任务标签不进入教师与 tokenizer")
    p.block(685,1190,575,137,"分开评价分量、总重建与码本",
            "S / O / 总误差 · 同码率直接量化 · 码字占用\nHb 边界跳变与较宽时间跨度分别核验\n先记录内汇总，再按被试等权；保留失败分母")
    p.block(1305,1190,610,137,"冻结码流 → 上下文 / 下游接口",
            "时间块 / 模态块联合遮挡 S、O → token 预测\n功率代理旧版已有掩码先导；本电位版尚未开展\n任务迁移与生理解释分别验证",dashed=True,face="white")
    p.arrow([(80,1362),(142,1362)])
    p.t(157,1362,"部署前向",11)
    p.arrow([(340,1362),(402,1362)],color=purple,dashed=True)
    p.t(417,1362,"冻结目标，仅训练监督",11,color=purple)
    p.box(800,1353,45,18,dashed=True)
    p.t(860,1362,"需区分版本的研究接口",11,color=gray)
    p.t(1915,1362,"蓝：S 条件分量    橙：O 固定残差",11,ha="right")
    p.t(65,1406,"结构与示意波形用于解释实现，不是实验曲线；条件分解与重建保真不等于生理真值或已证实跨模态收益。",11,color=red)
    figures = [p.save()]

    p = Plate("02_teacher_and_targets", "图组 1B  |  从配对观测到冻结目标",
              "固定参考 Tak 动力学约束共享驱动；EEG 电位通过代理引导的条件修正获得目标，Hb 保留完整模型输出。", 1210)
    p.section(145,"(a)  连续观测、Tak 动力学与观测回放")
    p.wave(70,204,180,75,"eeg")
    p.t(160,184,"原生 EEG 电位",12.5,ha="center")
    p.t(160,303,"200 Hz · μV",11,color=gray,ha="center")
    p.arrow([(260,241),(310,241)])
    p.block(322,192,322,108,"区域功率代理",
            "6 电极 × 5 频带 → 100 ms 功率\n对数 / 参考 → 训练 PCA 与尺度")
    p.wave(70,370,180,75,"hb")
    p.t(160,350,"缓存 HbO / HbR",12.5,ha="center")
    p.t(160,469,"原生单位与时钟保留",11,color=gray,ha="center")
    p.arrow([(260,407),(310,407)])
    p.block(322,356,322,108,"统一 Hb 观测",
            "带通缓存 → 10 Hz 时间箱\n5 秒可见参考与共同正尺度")
    p.arrow([(652,241),(679,241),(679,320),(700,320)])
    p.segment([(652,407),(679,407),(679,320)])
    p.box(711,179,620,312,face=fill)
    p.t(1021,208,"Tak 血流动力学核心",15,ha="center")
    p.m(1021,251,r"\dot s=r-\kappa s-\gamma(f-1),\qquad\dot f=s",21)
    p.m(1021,301,r"\dot v=\frac{f-v^{1/\alpha}}{\tau+\tau_v},\qquad f_{\rm out}=v^{1/\alpha}+\tau_v\dot v",20)
    p.m(1021,361,r"\dot q=\frac{fE(f)/E_0-f_{\rm out}q/v}{\tau},\quad E(f)=1-(1-E_0)^{1/f}",20)
    p.t(1021,420,"r：共享驱动；s：流量诱导信号",11.5,ha="center",color=gray)
    p.t(1021,452,"f / v / q：相对静息的血流 / 血容量 / 脱氧 Hb",11.5,ha="center",color=gray)
    p.arrow([(1342,330),(1382,330)])
    p.box(1393,179,522,312)
    p.t(1654,208,"物理读出 → 同坐标观测算子",14,ha="center")
    p.m(1654,260,r"H_O=P_0(v-1)-Q_0(q-1)",21)
    p.m(1654,304,r"H_R=Q_0(q-1)",21)
    p.m(1654,357,r"\widetilde{\mathbf{u}}=\mathcal{A}[\,r,H_O,H_R\,]",23,color=purple)
    p.t(1654,406,r"$P_0$ / $Q_0$：静息总 Hb / 脱氧 Hb 的基准量",11,color=gray,ha="center")
    p.t(1654,437,r"$\mathcal{A}$：时间箱、参考及冻结观测变换",11,color=gray,ha="center")
    p.t(1654,469,"固定参数；拟合完整驱动与四个初态",11,color=gray,ha="center")
    p.t(65,511,r"$\kappa$：衰减；$\gamma$：反馈；$\tau$：转运时间；$\tau_v$：黏弹时间（两者均为秒）；$\alpha$：血管弹性指数；$E_0$：静息氧提取率。",10.5,color=gray)
    p.t(65,540,"时间坐标：教师使用事件 [−30,60) s 的 90 秒支持；目标取 [0,20) s；参考为 [−30,−25) s。",11,color=gray)
    p.rule(556)
    p.section(591,"(b)  两种测量坐标分别构造 S / O")
    p.t(78,633,"EEG：冻结代理引导原始电压幅度修正",14,boldface=True)
    p.box(66,660,1155,243)
    p.block(86,688,300,97,"联合区域约束",
            "三个区域共享唯一电极集合\n最小对数功率改动 → 平滑幅度增益")
    p.arrow([(399,736),(438,736)])
    p.box(449,685,425,104,face=fill)
    p.m(662,714,r"S_E^*=X_E+\sum_b(g_b-1)\,B_bX_E",20,color=blue)
    p.m(662,760,r"O_E^*=X_E-S_E^*",21,color=orange)
    p.arrow([(884,736),(923,736)])
    p.block(934,688,265,97,"实际回算功率代理",
            "重新滤波 / 功率 / 参考 / PCA\n仅接受代理误差下降的修正")
    p.t(88,826,"B：五频带滤波；g：平滑增益；保持原生电压参考和原 5 秒功率参考。",11.5,color=gray)
    p.t(88,861,"100 ms 平滑，目标两端各 2 秒余弦过渡；回退强度固定，失败保留。",11.5,color=gray)
    p.t(1280,633,"fNIRS：完整 Tak 读出与固定差值",14,boldface=True)
    p.box(1265,660,650,243)
    p.m(1590,710,r"S_F^*=\widetilde{\mathbf{u}}_F",25,color=blue)
    p.m(1590,765,r"O_F^*=X_F-S_F^*",25,color=orange)
    p.t(1590,825,"HbO / HbR 在同一观测坐标中逐点相减",12,ha="center",color=gray)
    p.t(1590,864,r"两模态均满足：$S^*+O^*=X$",12,ha="center",color=gray)
    p.rule(938)
    p.section(973,"(c)  条件目标的用途与独立诊断")
    p.block(65,1010,575,116,"教师的用途",
            "目标只监督本模态连续预测器；部署不读取教师\n本版复用保留 Tak 结果，不重新拟合 SSM")
    p.block(685,1010,575,116,"数值与分解核验",
            "物理域 / 收敛 / 积分加密 / 目标闭合\n保留计划分母、数值失败和修正失败")
    p.block(1305,1010,610,116,"另行开展的模型诊断",
            "Hb 可拟合性 · EEG–Hb 驱动一致性\n空间对应 · 联合三观测失配；与分词评价分开")
    p.t(65,1170,"解释边界：电压 S 是条件幅度修正，O 是对应差值；二者不直接命名为“真实神经信号”与“纯噪声”。",11.5,color=red)
    figures.append(p.save())

    p = Plate("03_discrete_tokens_and_evaluation", "图组 1C  |  离散词表、部署接口与评价边界",
              "当前电位版本采用连续分量预测后的波形空间残差量化；训练、词典拟合、冻结推理和研究接口分别展开。", 1320)
    p.section(145,"(a)  先学习分量预测，再拟合训练波形词表")
    p.block(65,195,300,146,"同模态输入",
            "EEG：单电极原生电压\nHb：同一通道对的 HbO/HbR\n附有效掩码、已知通道身份\n被试划分：训练 / 验证 / 测试 = 6/2/2")
    p.arrow([(376,267),(423,267)])
    p.block(435,195,355,146,"S / O 独立连续预测器",
            "各自的 MLP 编码器与波形解码器\n128 隐层 → 64 维潜变量\n仅分支目标误差参与主训练")
    p.arrow([(800,267),(849,267)])
    p.block(860,195,480,146,"冻结训练预测 → 波形空间词典",
            "分别收集训练集 S / O 连续预测\n逐层 MiniBatch K-means 拟合量化残差\n每层 128 个码字；模态、分支、层均独立")
    p.arrow([(1350,267),(1398,267)])
    p.block(1410,195,505,146,"验证集选择 → 冻结部署",
            "固定预测器、词表、尺度与合成方式\n推理只需本模态输入；解码只需整数 ID\n逐片段均值 / 幅度不作为连续旁路")
    p.m(613,395,r"\mathcal{L}_m=\frac{1}{2}\sum_{a\in\{S,O\}}\frac{\operatorname{MSE}_{\Omega}(\widetilde Y_m^a,Y_m^{a*})}{v_{m,a}^{\rm train}}",24,color=purple)
    p.t(1338,381,"Ω：有效目标元素；v：训练分量方差（实现设数值下限）。",11.5,color=gray,ha="center")
    p.t(1338,413,"本页为选中的 EEG 与 Hb blocks 路径；边界 / 重叠损失属于其他候选。",11,color=gray,ha="center")
    p.rule(452)
    p.section(487,"(b)  每个分量独立做逐层残差量化与整数查表")
    p.wave(73,566,177,86,"smooth",color=blue)
    p.m(161,536,r"\rho_0=\widetilde Y_p^a",21,color=blue)
    p.arrow([(262,608),(311,608)])
    for i,x in enumerate([325,570,815,1060]):
        p.book(x,567,118,80,blue,1)
        p.m(x+59,537,rf"Q_{{{i+1}}}^a",21,color=blue)
        p.m(x+59,699,rf"k_{{{i+1}}}^a",20)
        if i<3:
            p.arrow([(x+132,608),(x+229,608)])
            p.t(x+182,581,"减去码字",10,color=gray,ha="center")
    p.arrow([(1192,608),(1256,608)])
    p.box(1268,550,647,132,face=fill)
    p.m(1591,593,r"\widehat Y_p^a=\sum_{\ell=1}^{L_m}c_{m,\ell,k_{p,\ell}^a}^a",20)
    p.t(1591,659,"同一分支的固定码字相加 → 时间拼接 / 固定合成",11.5,color=gray,ha="center")
    p.m(639,760,r"k_\ell=\arg\min_k\|\rho_{\ell-1}-c_{\ell,k}\|_2^2,\qquad\rho_\ell=\rho_{\ell-1}-c_{\ell,k_\ell}",23)
    p.t(639,795,"ρ 表示逐层量化误差；O 表示冻结的测量减条件分量。",10.5,color=gray,ha="center")
    p.t(1591,709,"图示 4 层为 EEG；Hb 使用前 2 层。",11.5,ha="center",color=gray)
    p.t(1591,752,r"Hb 高斯合成：$\sigma=100$ ms，截断于 $\pm4\sigma$。",11.5,ha="center",color=gray)
    p.t(1591,789,"输出支持两端各增 0.4 秒；后续遮挡需计入。",11.5,ha="center",color=gray)
    p.box(65,813,1850,101)
    for x,label,values in [
        (230,"模态 / 原生输入","EEG：200 Hz 单电极\nfNIRS：10 Hz HbO/HbR 通道对"),
        (700,"可见上下文 → 单词输出 / 步幅","2 秒 → 0.1 秒 / 0.1 秒\n6 秒 → 2 秒 / 2 秒"),
        (1145,"每分支层数 / 码字数","4 层 × 128\n2 层 × 128"),
        (1640,"S + O 总名义码率","560 bit/s / 电极\n14 bit/s / Hb 通道对")]:
        p.t(x,837,label,11.5,ha="center",boldface=True)
        p.t(x,882,values,11,ha="center",color=gray)
    p.rule(947)
    p.section(982,"(c)  三类问题使用各自的证据，不能互相替代")
    p.block(65,1019,575,179,"本电位版：分量与波形保真",
            "S / O / 总重建误差；HbO、HbR 分列\n评分：EEG 用训练 RMS，Hb 用训练 SD\n对照：连续预测、同码率直接量化、零修正\n按完整被试汇总；码字健康与波形误差分列")
    p.block(685,1019,575,179,"保留功率代理版：掩码先导",
            "冻结词表 → 联合遮挡 S/O → Transformer\n时间块与模态块；输出 token 概率\n与插值、自身模态、错配和时间移位比较\n电位词表尚未重新开展该先导",face="white",dashed=True)
    p.block(1305,1019,610,179,"下游任务：独立比较轨道",
            "EEG / fNIRS / 融合表征 → 任务读出\n线性探测与微调，沿用各任务划分和标签\n与现有方法在共同支持和预算内比较\n本电位词表的任务迁移收益尚待验证",face="white",dashed=True)
    p.t(65,1240,"训练目标、码本与变换按训练 / 验证边界冻结；现有公开测试结果属于探索性证据。",11.5,color=gray)
    p.t(65,1281,"读图约定：波形与码字色块均为示意；分词重建、上下文补全、任务性能与生理解释是四个不同结论。",11.5,color=red)
    figures.append(p.save())

    captions = [
        "完整框架。主图按 2026-10-29 电位 v2 实现绘制：连续分量预测后进行波形空间量化，S/O 独立查表解码。紫色虚线表示冻结目标的训练监督。",
        "目标构建。Tak 教师基于配对观测形成条件预测；EEG 目标另经代理引导的电压幅度修正。模型目标、观测残差与生理真值保持概念区分。",
        "离散与评价。整数码率计入全部分支和量化层；Hb 固定合成扩展输出支持。保留功率代理版的掩码结果不自动转移到本电位版。",
    ]
    return finish(figures, captions, "project_framework_atlas.pdf")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--changes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Fail when output differs from a fresh render")
    parser.add_argument("--framework-20261029", action="store_true",
                        help="Export the dated Tak/voltage framework PNG, SVG and bitmap PDF atlas to a fresh directory")
    parser.add_argument("--framework-layout", choices=("detailed", "flow", "token_first", "paper", "paper_flow"), default="detailed",
                        help="Use flow for the current projection, token_first for the token-first architecture, paper for the 180 mm panel view, or paper_flow for its 170 mm connected-flow redraw")
    args = parser.parse_args()

    if args.framework_20261029:
        if args.check or args.changes:
            parser.error("The dated framework export does not accept --check or --changes")
        for path in render_framework_20261029(args.output.resolve(), layout=args.framework_layout):
            print(path)
        return 0

    spec = load_json(args.spec)
    changes = load_json(args.changes) if args.changes else None
    output = args.output.resolve()
    if output in {path.resolve() for path in DRAWIO_OWNED_OUTPUTS}:
        raise ValueError("Draw.io-owned SVGs must be exported from their matching .drawio source")
    try:
        spec_source = str(args.spec.resolve().relative_to(REPO_ROOT))
    except ValueError:
        spec_source = str(args.spec.resolve())
    changes_source = None
    if args.changes:
        try:
            changes_source = str(args.changes.resolve().relative_to(REPO_ROOT))
        except ValueError:
            changes_source = str(args.changes.resolve())
    rendered = render_svg(spec, changes, spec_source=spec_source, changes_source=changes_source)
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"Architecture SVG is stale: {output}")
        print(f"Architecture SVG is current: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
