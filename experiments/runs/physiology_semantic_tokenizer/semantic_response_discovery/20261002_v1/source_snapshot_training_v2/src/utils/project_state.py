"""Validate and render the repository-wide research state registry.

The registry deliberately has two status axes only:

* ``execution`` records what happened operationally;
* ``scientific_verdict`` records what the evidence permits scientifically.

Protocol and data-access constraints remain in their owning contracts.  They are not
project-state axes and are not interpreted by this module.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "research_state/registry.json"
DEFAULT_STATUS_DOCUMENT = REPO_ROOT / "docs/PROJECT_STATUS.md"
DEFAULT_README = REPO_ROOT / "README.md"

README_START = "<!-- project-state:begin -->"
README_END = "<!-- project-state:end -->"

SCHEMA = "research_state_registry_v1"
EXECUTION_STATES = {
    "planned",
    "blocked",
    "running",
    "completed",
    "stopped",
    "abandoned",
    "failed",
    "aborted",
}
SCIENTIFIC_VERDICTS = {
    "unreviewed",
    "qualified",
    "rejected",
    "inconclusive",
    "not_applicable",
    "mixed",
}
ANALYSIS_INTENTS = {
    "confirmatory",
    "exploratory",
    "descriptive",
    "engineering",
    "not_applicable",
}
ENTITY_KINDS = {
    "project",
    "experiment",
    "analysis",
    "aggregate",
    "engineering",
}
EVIDENCE_ROLES = {
    "protocol",
    "run_manifest",
    "result_summary",
    "decision",
    "diagnostic",
    "data_contract",
}

EXECUTION_LABELS = {
    "planned": "未开始",
    "blocked": "前置条件未满足",
    "running": "运行中",
    "completed": "已完成",
    "stopped": "已停止（此前已完成）",
    "abandoned": "已废弃（未完成且不再开展）",
    "failed": "技术失败",
    "aborted": "已中止",
}
VERDICT_LABELS = {
    "unreviewed": "尚未判定",
    "qualified": "支持限定主张",
    "rejected": "不支持预定主张",
    "inconclusive": "证据不足",
    "not_applicable": "不适用",
    "mixed": "混合结论",
}
INTENT_LABELS = {
    "confirmatory": "验证性",
    "exploratory": "探索性",
    "descriptive": "描述性",
    "engineering": "工程性",
    "not_applicable": "不适用",
}

_TOP_LEVEL_FIELDS = {"schema", "updated_at", "tracks", "evidence", "records"}
_TRACK_FIELDS = {"id", "label", "order", "headline_entity"}
_EVIDENCE_FIELDS = {"id", "path", "role", "label"}
_RECORD_FIELDS = {
    "state_id",
    "entity",
    "label",
    "track",
    "order",
    "entity_kind",
    "execution",
    "scientific_verdict",
    "analysis_intent",
    "summary",
    "evidence_ids",
    "supersedes",
    "updated_at",
    "progress",
    "outcome_counts",
    "depends_on",
    "next_step",
    "headline",
}
_PROGRESS_FIELDS = {"completed", "total", "unit"}
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_STATE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*@[0-9]{4}-[0-9]{2}-[0-9]{2}(?:\.[a-z0-9_-]+)?$")


class ProjectStateError(ValueError):
    """Raised when the project-state registry or a generated view is invalid."""


def _unknown_fields(value: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ProjectStateError(f"{context}: unknown fields {sorted(unknown)}")


def _require_fields(value: Mapping[str, Any], required: set[str], context: str) -> None:
    missing = required - set(value)
    if missing:
        raise ProjectStateError(f"{context}: missing fields {sorted(missing)}")


def _parse_timestamp(value: Any, context: str) -> datetime:
    """Parse a practical ISO timestamp for display and sorting.

    The registry is a reading aid rather than a long-lived event ledger.  Accept
    date-only values and timestamps without an explicit offset, normalising the
    latter to UTC for comparisons.  Publication-time provenance can still use
    full RFC3339 values when desired.
    """

    if not isinstance(value, str) or not value.strip():
        raise ProjectStateError(f"{context}: expected an ISO date or timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProjectStateError(f"{context}: invalid ISO date/timestamp {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _validate_relative_path(value: Any, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ProjectStateError(f"{context}: path must be a non-empty string")
    if "\\" in value:
        raise ProjectStateError(f"{context}: path must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ProjectStateError(f"{context}: path must be normalized and repo-relative")
    return path


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    """Load a JSON registry without applying semantic validation."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProjectStateError(f"registry does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProjectStateError(f"registry is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ProjectStateError("registry root must be an object")
    return value


def validate_registry(
    registry: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
) -> None:
    """Validate links and status semantics in the compact current-state registry."""

    _unknown_fields(registry, _TOP_LEVEL_FIELDS, "registry")
    _require_fields(registry, _TOP_LEVEL_FIELDS, "registry")
    if registry["schema"] != SCHEMA:
        raise ProjectStateError(f"registry: unsupported schema {registry['schema']!r}")

    _parse_timestamp(registry["updated_at"], "registry.updated_at")

    tracks = registry["tracks"]
    if not isinstance(tracks, list) or not tracks:
        raise ProjectStateError("registry.tracks must be a non-empty list")
    track_by_id: dict[str, Mapping[str, Any]] = {}
    track_orders: set[int] = set()
    for index, track in enumerate(tracks):
        context = f"registry.tracks[{index}]"
        if not isinstance(track, Mapping):
            raise ProjectStateError(f"{context}: track must be an object")
        _unknown_fields(track, _TRACK_FIELDS, context)
        _require_fields(track, _TRACK_FIELDS, context)
        track_id = track["id"]
        if not isinstance(track_id, str) or not _ID_RE.fullmatch(track_id):
            raise ProjectStateError(f"{context}.id: invalid identifier")
        if track_id in track_by_id:
            raise ProjectStateError(f"{context}.id: duplicate track {track_id!r}")
        if not isinstance(track["label"], str) or not track["label"].strip():
            raise ProjectStateError(f"{context}.label: must be non-empty")
        if not isinstance(track["order"], int) or track["order"] < 0:
            raise ProjectStateError(f"{context}.order: must be a non-negative integer")
        if track["order"] in track_orders:
            raise ProjectStateError(f"{context}.order: duplicate order {track['order']}")
        headline_entity = track["headline_entity"]
        if not isinstance(headline_entity, str) or not _ID_RE.fullmatch(headline_entity):
            raise ProjectStateError(f"{context}.headline_entity: invalid identifier")
        track_orders.add(track["order"])
        track_by_id[track_id] = track

    evidence = registry["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ProjectStateError("registry.evidence must be a non-empty list")
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    evidence_paths: set[str] = set()
    for index, source in enumerate(evidence):
        context = f"registry.evidence[{index}]"
        if not isinstance(source, Mapping):
            raise ProjectStateError(f"{context}: evidence must be an object")
        _unknown_fields(source, _EVIDENCE_FIELDS, context)
        _require_fields(source, {"id", "path", "role", "label"}, context)
        evidence_id = source["id"]
        if not isinstance(evidence_id, str) or not _ID_RE.fullmatch(evidence_id):
            raise ProjectStateError(f"{context}.id: invalid identifier")
        if evidence_id in evidence_by_id:
            raise ProjectStateError(f"{context}.id: duplicate evidence ID {evidence_id!r}")
        relative = _validate_relative_path(source["path"], f"{context}.path")
        if relative.as_posix() in evidence_paths:
            raise ProjectStateError(f"{context}.path: duplicate evidence path")
        if source["role"] not in EVIDENCE_ROLES:
            raise ProjectStateError(f"{context}.role: unsupported role {source['role']!r}")
        if not isinstance(source["label"], str) or not source["label"].strip():
            raise ProjectStateError(f"{context}.label: must be non-empty")
        absolute = repo_root / relative
        if not absolute.is_file():
            raise ProjectStateError(f"{context}.path: evidence file is missing: {relative}")
        evidence_paths.add(relative.as_posix())
        evidence_by_id[evidence_id] = source

    records = registry["records"]
    if not isinstance(records, list) or not records:
        raise ProjectStateError("registry.records must be a non-empty list")
    record_by_id: dict[str, Mapping[str, Any]] = {}
    entity_by_record_id: dict[str, str] = {}
    for index, record in enumerate(records):
        context = f"registry.records[{index}]"
        if not isinstance(record, Mapping):
            raise ProjectStateError(f"{context}: record must be an object")
        _unknown_fields(record, _RECORD_FIELDS, context)
        required = {
            "state_id",
            "entity",
            "label",
            "track",
            "order",
            "entity_kind",
            "execution",
            "scientific_verdict",
            "analysis_intent",
            "summary",
            "evidence_ids",
            "updated_at",
            "headline",
        }
        _require_fields(record, required, context)

        state_id = record["state_id"]
        entity = record["entity"]
        if not isinstance(state_id, str) or not _STATE_ID_RE.fullmatch(state_id):
            raise ProjectStateError(f"{context}.state_id: invalid identifier")
        if state_id in record_by_id:
            raise ProjectStateError(f"{context}.state_id: duplicate {state_id!r}")
        if not isinstance(entity, str) or not _ID_RE.fullmatch(entity):
            raise ProjectStateError(f"{context}.entity: invalid identifier")
        if record["track"] not in track_by_id:
            raise ProjectStateError(f"{context}.track: unknown track {record['track']!r}")
        if not isinstance(record["label"], str) or not record["label"].strip():
            raise ProjectStateError(f"{context}.label: must be non-empty")
        if not isinstance(record["summary"], str) or not record["summary"].strip():
            raise ProjectStateError(f"{context}.summary: must be non-empty")
        if not isinstance(record["order"], int) or record["order"] < 0:
            raise ProjectStateError(f"{context}.order: must be a non-negative integer")
        if record["entity_kind"] not in ENTITY_KINDS:
            raise ProjectStateError(f"{context}.entity_kind: unsupported value")
        if record["execution"] not in EXECUTION_STATES:
            raise ProjectStateError(f"{context}.execution: unsupported value")
        if record["scientific_verdict"] not in SCIENTIFIC_VERDICTS:
            raise ProjectStateError(f"{context}.scientific_verdict: unsupported value")
        if record["analysis_intent"] not in ANALYSIS_INTENTS:
            raise ProjectStateError(f"{context}.analysis_intent: unsupported value")
        if not isinstance(record["headline"], bool):
            raise ProjectStateError(f"{context}.headline: must be boolean")

        evidence_ids = record["evidence_ids"]
        if not isinstance(evidence_ids, list) or any(
            not isinstance(item, str) for item in evidence_ids
        ):
            raise ProjectStateError(f"{context}.evidence_ids: must be a string list")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ProjectStateError(f"{context}.evidence_ids: duplicate evidence IDs")
        missing_evidence = set(evidence_ids) - set(evidence_by_id)
        if missing_evidence:
            raise ProjectStateError(
                f"{context}.evidence_ids: unknown IDs {sorted(missing_evidence)}"
            )
        supersedes = record.get("supersedes", [])
        depends_on = record.get("depends_on", [])
        for field_name, value in (("supersedes", supersedes), ("depends_on", depends_on)):
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ProjectStateError(f"{context}.{field_name}: must be a string list")
            if len(value) != len(set(value)):
                raise ProjectStateError(f"{context}.{field_name}: duplicate values")
        if state_id in supersedes:
            raise ProjectStateError(f"{context}.supersedes: cannot supersede itself")
        if entity in depends_on:
            raise ProjectStateError(f"{context}.depends_on: cannot depend on itself")

        _parse_timestamp(record["updated_at"], f"{context}.updated_at")

        execution = record["execution"]
        verdict = record["scientific_verdict"]
        if execution in {"planned", "blocked", "running"} and verdict not in {
            "unreviewed",
            "not_applicable",
        }:
            raise ProjectStateError(
                f"{context}: {execution} work cannot have scientific verdict {verdict}"
            )
        if execution == "abandoned" and verdict not in {
            "unreviewed",
            "not_applicable",
            "inconclusive",
        }:
            raise ProjectStateError(
                f"{context}: abandoned work cannot have scientific verdict {verdict}"
            )
        if execution in {"failed", "aborted"} and verdict != "inconclusive":
            raise ProjectStateError(
                f"{context}: {execution} work must remain scientifically inconclusive"
            )
        if verdict in {"qualified", "rejected", "mixed"} and execution not in {
            "completed",
            "stopped",
        }:
            raise ProjectStateError(
                f"{context}: scientific verdict {verdict} requires completed or stopped execution"
            )
        if verdict == "mixed" and record["entity_kind"] != "aggregate":
            raise ProjectStateError(f"{context}: mixed verdict is aggregate-only")
        if execution in {"completed", "stopped", "failed", "aborted", "abandoned"} and not evidence_ids:
            raise ProjectStateError(f"{context}: terminal execution requires evidence")
        if verdict in {"qualified", "rejected", "mixed"}:
            roles = {evidence_by_id[item]["role"] for item in evidence_ids}
            if not roles & {"decision", "result_summary"}:
                raise ProjectStateError(
                    f"{context}: verdict {verdict} requires decision/result evidence"
                )

        progress = record.get("progress")
        if progress is not None:
            if not isinstance(progress, Mapping):
                raise ProjectStateError(f"{context}.progress: must be an object")
            _unknown_fields(progress, _PROGRESS_FIELDS, f"{context}.progress")
            _require_fields(progress, _PROGRESS_FIELDS, f"{context}.progress")
            completed, total = progress["completed"], progress["total"]
            if (
                not isinstance(completed, int)
                or isinstance(completed, bool)
                or not isinstance(total, int)
                or isinstance(total, bool)
                or completed < 0
                or total <= 0
                or completed > total
            ):
                raise ProjectStateError(f"{context}.progress: invalid completed/total")
            if not isinstance(progress["unit"], str) or not progress["unit"].strip():
                raise ProjectStateError(f"{context}.progress.unit: must be non-empty")
            if execution in {"completed", "stopped"} and completed != total:
                raise ProjectStateError(
                    f"{context}.progress: {execution} execution requires completed == total"
                )
            if execution == "abandoned" and completed == total:
                raise ProjectStateError(
                    f"{context}.progress: abandoned execution cannot be complete"
                )

        outcome_counts = record.get("outcome_counts")
        if outcome_counts is not None:
            if record["entity_kind"] != "aggregate":
                raise ProjectStateError(
                    f"{context}.outcome_counts: only aggregate entities may carry counts"
                )
            if not isinstance(outcome_counts, Mapping) or not outcome_counts:
                raise ProjectStateError(f"{context}.outcome_counts: must be a non-empty object")
            for key, value in outcome_counts.items():
                if not isinstance(key, str) or not _ID_RE.fullmatch(key):
                    raise ProjectStateError(f"{context}.outcome_counts: invalid key {key!r}")
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ProjectStateError(f"{context}.outcome_counts.{key}: invalid count")
        if verdict == "mixed" and (
            not isinstance(outcome_counts, Mapping)
            or sum(value > 0 for value in outcome_counts.values()) < 2
        ):
            raise ProjectStateError(
                f"{context}: mixed verdict requires at least two non-zero outcome counts"
            )

        next_step = record.get("next_step")
        if next_step is not None and (
            not isinstance(next_step, str) or not next_step.strip()
        ):
            raise ProjectStateError(f"{context}.next_step: must be non-empty when present")
        if execution in {"stopped", "abandoned"} and next_step is not None:
            raise ProjectStateError(
                f"{context}: {execution} execution cannot carry next_step"
            )

        record_by_id[state_id] = record
        entity_by_record_id[state_id] = entity

    superseded_ids: set[str] = set()
    for state_id, record in record_by_id.items():
        for target in record.get("supersedes", []):
            if target not in record_by_id:
                raise ProjectStateError(f"record {state_id}: supersedes unknown state {target!r}")
            if entity_by_record_id[target] != record["entity"]:
                raise ProjectStateError(f"record {state_id}: supersedes a different entity")
            if _parse_timestamp(record["updated_at"], f"record {state_id}.updated_at") <= (
                _parse_timestamp(
                    record_by_id[target]["updated_at"],
                    f"record {target}.updated_at",
                )
            ):
                raise ProjectStateError(
                    f"record {state_id}: must be newer than superseded state {target!r}"
                )
            if target in superseded_ids:
                raise ProjectStateError(f"state {target!r} is superseded more than once")
            superseded_ids.add(target)

    active_records = [
        record for state_id, record in record_by_id.items() if state_id not in superseded_ids
    ]
    active_by_entity: dict[str, Mapping[str, Any]] = {}
    for record in active_records:
        entity = record["entity"]
        if entity in active_by_entity:
            raise ProjectStateError(f"entity {entity!r} has more than one current state")
        active_by_entity[entity] = record

    all_entities = {record["entity"] for record in records}
    missing_current = all_entities - set(active_by_entity)
    if missing_current:
        raise ProjectStateError(
            "every entity must have exactly one current state; missing current state for "
            f"{sorted(missing_current)}"
        )

    def visit_supersedes(state_id: str, visiting: set[str], visited: set[str]) -> None:
        if state_id in visited:
            return
        if state_id in visiting:
            raise ProjectStateError(f"supersedes cycle includes state {state_id!r}")
        visiting.add(state_id)
        for target in record_by_id[state_id].get("supersedes", []):
            visit_supersedes(target, visiting, visited)
        visiting.remove(state_id)
        visited.add(state_id)

    visited_states: set[str] = set()
    for state_id in record_by_id:
        visit_supersedes(state_id, set(), visited_states)

    for record in active_records:
        missing_dependencies = set(record.get("depends_on", [])) - all_entities
        if missing_dependencies:
            raise ProjectStateError(
                f"record {record['state_id']}: unknown dependencies {sorted(missing_dependencies)}"
            )

    def visit(entity: str, visiting: set[str], visited: set[str]) -> None:
        if entity in visited:
            return
        if entity in visiting:
            raise ProjectStateError(f"dependency cycle includes entity {entity!r}")
        visiting.add(entity)
        record = active_by_entity.get(entity)
        if record:
            for dependency in record.get("depends_on", []):
                visit(dependency, visiting, visited)
        visiting.remove(entity)
        visited.add(entity)

    visited: set[str] = set()
    for entity in active_by_entity:
        visit(entity, set(), visited)

    for track_id, track in track_by_id.items():
        active_orders = [
            record["order"] for record in active_records if record["track"] == track_id
        ]
        if len(active_orders) != len(set(active_orders)):
            raise ProjectStateError(f"track {track_id!r}: duplicate current record order")
        headline = active_by_entity.get(track["headline_entity"])
        if headline is None:
            raise ProjectStateError(
                f"track {track_id!r}: headline entity {track['headline_entity']!r} is not current"
            )
        if headline["track"] != track_id or headline["headline"] is not True:
            raise ProjectStateError(
                f"track {track_id!r}: headline entity must be a headline in the same track"
            )
        headline_entities = {
            record["entity"]
            for record in active_records
            if record["track"] == track_id and record["headline"] is True
        }
        if headline_entities != {track["headline_entity"]}:
            raise ProjectStateError(
                f"track {track_id!r}: expected exactly one current headline "
                f"{track['headline_entity']!r}, found {sorted(headline_entities)}"
            )


def current_records(registry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the single current state for every entity, ordered for presentation."""

    superseded = {
        target
        for record in registry["records"]
        for target in record.get("supersedes", [])
    }
    tracks = {track["id"]: track["order"] for track in registry["tracks"]}
    return sorted(
        (record for record in registry["records"] if record["state_id"] not in superseded),
        key=lambda record: (tracks[record["track"]], record["order"], record["entity"]),
    )


def _effective_updated_at(registry: Mapping[str, Any]) -> str:
    """Return the newest registry/record timestamp for generated views.

    Updating one current record is the normal daily workflow.  The project-level
    timestamp is retained as a convenient snapshot marker, but it should not make
    a user edit the same date in two places just to refresh a status view.
    """

    latest_value = str(registry["updated_at"])
    latest_timestamp = _parse_timestamp(latest_value, "registry.updated_at")
    for index, record in enumerate(registry.get("records", [])):
        value = record.get("updated_at") if isinstance(record, Mapping) else None
        if value is None:
            continue
        timestamp = _parse_timestamp(value, f"registry.records[{index}].updated_at")
        if timestamp > latest_timestamp:
            latest_timestamp = timestamp
            latest_value = str(value)
    return latest_value


def _markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _progress_text(record: Mapping[str, Any]) -> str:
    progress = record.get("progress")
    if not progress:
        return "—"
    return f"{progress['completed']}/{progress['total']} {progress['unit']}"


def _status_text(record: Mapping[str, Any]) -> str:
    """Return one compact status cell while retaining both scientific axes."""

    status = (
        f"{EXECUTION_LABELS[record['execution']]} / "
        f"{VERDICT_LABELS[record['scientific_verdict']]}"
    )
    progress = _progress_text(record)
    return f"{status}（{progress}）" if progress != "—" else status


def _updated_text(record: Mapping[str, Any]) -> str:
    """Render a short date; the source registry retains the full timestamp."""

    value = str(record.get("updated_at", ""))
    return value[:10] if value else "—"


def _next_text(
    record: Mapping[str, Any],
    records_by_entity: Mapping[str, Mapping[str, Any]],
) -> str:
    """Combine dependencies and the optional next action in one readable cell."""

    if record["execution"] not in {"running", "planned", "blocked"}:
        return "—"

    parts: list[str] = []
    dependencies = []
    for dependency in record.get("depends_on", []):
        dependency_record = records_by_entity.get(dependency)
        dependencies.append(
            dependency_record["label"] if dependency_record else dependency
        )
    if dependencies:
        parts.append("依赖：" + "、".join(dependencies))
    next_step = record.get("next_step")
    if next_step:
        parts.append(str(next_step))
    return "；".join(parts) if parts else "—"


def _evidence_links(
    record: Mapping[str, Any],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    output_parent: Path,
    repo_root: Path,
) -> str:
    links = []
    for evidence_id in record["evidence_ids"]:
        source = evidence_by_id[evidence_id]
        relative = os.path.relpath(repo_root / source["path"], output_parent)
        label = source.get("label", evidence_id)
        links.append(f"[{_markdown_escape(str(label))}]({Path(relative).as_posix()})")
    return ", ".join(links) if links else "—"


def render_status_markdown(
    registry: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
    output_path: Path = DEFAULT_STATUS_DOCUMENT,
) -> str:
    """Render the complete human/agent status view from current records."""

    records = current_records(registry)
    evidence_by_id = {source["id"]: source for source in registry["evidence"]}
    records_by_track: dict[str, list[Mapping[str, Any]]] = {
        track["id"]: [] for track in registry["tracks"]
    }
    for record in records:
        records_by_track[record["track"]].append(record)

    records_by_entity = {record["entity"]: record for record in records}
    lines = [
        "# Project research status",
        "",
        "<!-- AUTO-GENERATED by experiments/scripts/project_state.py; DO NOT EDIT. -->",
        "",
        f"_Registry snapshot: `{_effective_updated_at(registry)}` · "
        "source: [`research_state/registry.json`](../research_state/registry.json)_",
        "",
        "Status combines execution and scientific verdict; a terminal run does not "
        "by itself mean that the hypothesis passed.",
        "",
        "## Current overview",
        "",
        "| Track | Current item | Status | Conclusion | Updated |",
        "| --- | --- | --- | --- | --- |",
    ]

    active_by_entity = {record["entity"]: record for record in records}
    for track in sorted(registry["tracks"], key=lambda item: item["order"]):
        record = active_by_entity[track["headline_entity"]]
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_escape(track["label"]),
                    _markdown_escape(record["label"]),
                    _markdown_escape(_status_text(record)),
                    _markdown_escape(record["summary"]),
                    _updated_text(record),
                )
            )
            + " |"
        )

    for track in sorted(registry["tracks"], key=lambda item: item["order"]):
        lines.extend(
            [
                "",
                f"## {track['label']}",
                "",
                "| Item | Status | Conclusion | Evidence | Next | Updated |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for record in records_by_track[track["id"]]:
            evidence = _evidence_links(
                record, evidence_by_id, output_path.parent, repo_root
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        _markdown_escape(record["label"]),
                        _markdown_escape(_status_text(record)),
                        _markdown_escape(record["summary"]),
                        evidence,
                        _markdown_escape(_next_text(record, records_by_entity)),
                        _updated_text(record),
                    )
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def render_readme_block(registry: Mapping[str, Any]) -> str:
    """Render the compact status block embedded in the repository README."""

    records = current_records(registry)
    active = {record["entity"]: record for record in records}
    lines = [
        README_START,
        "## Current research status",
        "",
        "_Generated from `research_state/registry.json`; do not edit this block._",
        "",
    ]
    for track in sorted(registry["tracks"], key=lambda item: item["order"]):
        record = active[track["headline_entity"]]
        lines.append(
            f"- **{track['label']}**（{record['label']}）— "
            f"{EXECUTION_LABELS[record['execution']]} / "
            f"{VERDICT_LABELS[record['scientific_verdict']]}：{record['summary']}"
        )
    next_priority = {
        "running": 0,
        "planned": 0,
        "blocked": 1,
    }
    next_records = sorted(
        (
            (track, active[track["headline_entity"]])
            for track in registry["tracks"]
            if active[track["headline_entity"]].get("next_step")
            and active[track["headline_entity"]]["execution"] in next_priority
        ),
        key=lambda item: (next_priority[item[1]["execution"]], item[0]["order"]),
    )[:3]
    if next_records:
        lines.extend(["", "### Next steps"])
        for track, record in next_records:
            lines.append(f"- **{track['label']}** — {record['next_step']}")
    lines.extend(
        [
            "",
            "See the [generated project status](docs/PROJECT_STATUS.md) for lifecycle "
            "states and evidence links.",
            README_END,
        ]
    )
    return "\n".join(lines)


def replace_readme_block(text: str, block: str) -> str:
    """Replace exactly one managed README block without touching surrounding prose."""

    start_count = text.count(README_START)
    end_count = text.count(README_END)
    if start_count != 1 or end_count != 1:
        raise ProjectStateError(
            "README must contain exactly one project-state begin/end marker pair"
        )
    start = text.index(README_START)
    end_start = text.find(README_END, start + len(README_START))
    if end_start < 0:
        raise ProjectStateError("README project-state markers are out of order")
    end = end_start + len(README_END)
    return text[:start] + block + text[end:]


def expected_outputs(
    registry: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
    status_path: Path = DEFAULT_STATUS_DOCUMENT,
    readme_path: Path = DEFAULT_README,
) -> dict[Path, str]:
    """Return every generated view without writing to disk."""

    status = render_status_markdown(registry, repo_root=repo_root, output_path=status_path)
    readme = replace_readme_block(
        readme_path.read_text(encoding="utf-8"), render_readme_block(registry)
    )
    return {status_path: status, readme_path: readme}


def check_outputs(
    registry: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
    status_path: Path = DEFAULT_STATUS_DOCUMENT,
    readme_path: Path = DEFAULT_README,
) -> list[Path]:
    """Return generated files that are missing or stale; never write."""

    stale = []
    for path, expected in expected_outputs(
        registry,
        repo_root=repo_root,
        status_path=status_path,
        readme_path=readme_path,
    ).items():
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            stale.append(path)
    return stale


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_outputs(
    registry: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
    status_path: Path = DEFAULT_STATUS_DOCUMENT,
    readme_path: Path = DEFAULT_README,
) -> list[Path]:
    """Atomically update all generated status views."""

    outputs = expected_outputs(
        registry,
        repo_root=repo_root,
        status_path=status_path,
        readme_path=readme_path,
    )
    for path, content in outputs.items():
        _atomic_write(path, content)
    return list(outputs)


def render_agent_summary(registry: Mapping[str, Any]) -> str:
    """Render a compact, deterministic context packet for a local agent."""

    tracks = {track["id"]: track for track in registry["tracks"]}
    lines = [
        f"registry={registry['schema']}",
        f"updated_at={_effective_updated_at(registry)}",
        "status_axes=execution,scientific_verdict",
    ]
    for record in current_records(registry):
        next_step = record.get("next_step", "-")
        lines.append(
            f"{record['entity']} | track={tracks[record['track']]['label']} | "
            f"execution={record['execution']} | "
            f"scientific_verdict={record['scientific_verdict']} | "
            f"next={next_step} | evidence={','.join(record['evidence_ids']) or '-'}"
        )
    return "\n".join(lines) + "\n"


def current_snapshot(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Return a machine-readable current-state projection for CLI consumers."""

    records = list(current_records(registry))
    evidence_ids = {
        evidence_id for record in records for evidence_id in record["evidence_ids"]
    }
    return {
        "schema": "research_state_snapshot_v1",
        "updated_at": _effective_updated_at(registry),
        "status_axes": ["execution", "scientific_verdict"],
        "tracks": registry["tracks"],
        "evidence": [
            source for source in registry["evidence"] if source["id"] in evidence_ids
        ],
        "records": records,
    }
