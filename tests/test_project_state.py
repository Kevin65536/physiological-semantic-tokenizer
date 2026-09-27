import copy
import re

import pytest

from src.utils.project_state import (
    DEFAULT_REGISTRY,
    ProjectStateError,
    current_records,
    current_snapshot,
    load_registry,
    render_agent_summary,
    render_readme_block,
    render_status_markdown,
    validate_registry,
)


def _registry():
    return load_registry(DEFAULT_REGISTRY)


@pytest.fixture
def registry_repo(tmp_path):
    """Exercise link/schema rules without requiring retained campaign files.

    These placeholders only satisfy the validator's file-presence contract;
    they contain no experiment evidence. The project-state CLI validates real
    evidence paths separately in the repository that retains those artifacts.
    """
    for source in _registry()["evidence"]:
        path = tmp_path / source["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Test-only evidence-link placeholder.\n", encoding="utf-8")
    return tmp_path


def _current_by_entity(registry):
    return {record["entity"]: record for record in current_records(registry)}


def test_current_registry_is_lightweight_and_views_are_readable(registry_repo):
    registry = _registry()
    validate_registry(registry, repo_root=registry_repo)

    status = render_status_markdown(registry, repo_root=registry_repo)
    assert "| Item | Status | Conclusion | Evidence | Next | Updated |" in status
    assert "## Evidence registry" not in status
    assert "主方法实验日志" in status
    assert "已停止（此前已完成）" in status
    assert "已废弃（未完成且不再开展）" in status
    terminal_rows = [
        line
        for line in status.splitlines()
        if line.startswith("| ")
        and line.count("|") >= 7
        and ("已停止（此前已完成）" in line or "已废弃（未完成且不再开展）" in line)
    ]
    assert terminal_rows
    assert all(re.search(r"\| — \| \d{4}-\d{2}-\d{2} \|$", line) for line in terminal_rows)

    readme = render_readme_block(registry)
    assert "### Next steps" in readme
    assert "Step5A" in readme
    assert "联合似然" in readme

    snapshot = current_snapshot(registry)
    assert snapshot["status_axes"] == ["execution", "scientific_verdict"]
    assert {source["id"] for source in snapshot["evidence"]} == {
        evidence_id
        for record in snapshot["records"]
        for evidence_id in record["evidence_ids"]
    }


def test_execution_and_scientific_verdict_remain_independent():
    current = _current_by_entity(_registry())

    assert current["main.r1p"]["execution"] == "stopped"
    assert current["main.r1p"]["scientific_verdict"] == "rejected"

    assert current["main.d1b"]["execution"] == "abandoned"
    assert current["main.d1b"]["scientific_verdict"] == "inconclusive"

    assert current["main.future_vq"]["execution"] == "abandoned"
    assert current["main.future_vq"]["scientific_verdict"] == "unreviewed"

    assert current["comparison.campaign"]["execution"] == "stopped"
    assert current["comparison.campaign"]["scientific_verdict"] == "mixed"
    assert current["comparison.efrm"]["execution"] == "stopped"
    assert current["comparison.efrm"]["scientific_verdict"] == "mixed"

    assert current["main.program"]["execution"] == "planned"
    assert current["main.program"]["scientific_verdict"] == "unreviewed"
    assert current["main.step5a"]["execution"] == "completed"
    assert current["main.step5a"]["scientific_verdict"] == "inconclusive"
    # Completed diagnostics can retain follow-up work; stopped and abandoned
    # queues must not become active merely because a new experiment is added.
    assert current["main.program"]["next_step"]
    assert current["main.shared_driver_conditional_optical_gain"]["execution"] == "completed"
    assert current["main.shared_driver_conditional_optical_gain"]["next_step"]
    assert all(
        not record.get("next_step")
        for record in current.values()
        if record["execution"] in {"stopped", "abandoned"}
    )


def test_comparison_method_totals_match_the_campaign_aggregate():
    current = _current_by_entity(_registry())
    method_entities = (
        "comparison.biot",
        "comparison.cbramod",
        "comparison.reve",
        "comparison.efrm",
        "comparison.normwear",
        "comparison.brainfusion",
    )
    campaign = current["comparison.campaign"]

    method_job_total = sum(
        current[entity]["progress"]["total"] for entity in method_entities
    )
    assert method_job_total == campaign["progress"]["total"] == 540
    for outcome, expected in campaign["outcome_counts"].items():
        assert (
            sum(
                current[entity].get("outcome_counts", {}).get(outcome, 0)
                for entity in method_entities
            )
            == expected
        )


def test_registry_does_not_scan_authorization_names_recursively(registry_repo):
    registry = copy.deepcopy(_registry())
    campaign = next(
        item for item in registry["records"] if item["entity"] == "comparison.campaign"
    )
    campaign["outcome_counts"]["authorization_status"] = 1

    validate_registry(registry, repo_root=registry_repo)


def test_registry_rejects_scientific_verdict_before_execution(registry_repo):
    registry = copy.deepcopy(_registry())
    record = next(item for item in registry["records"] if item["entity"] == "atlas.statistical")
    record["scientific_verdict"] = "qualified"

    with pytest.raises(ProjectStateError, match="planned work cannot"):
        validate_registry(registry, repo_root=registry_repo)


def test_registry_rejects_scientific_verdict_for_abandoned_work(registry_repo):
    registry = copy.deepcopy(_registry())
    record = _current_by_entity(registry)["atlas.statistical"]
    record["scientific_verdict"] = "qualified"

    with pytest.raises(ProjectStateError, match="abandoned work cannot"):
        validate_registry(registry, repo_root=registry_repo)


def test_stopped_work_requires_complete_progress_and_no_next_step(registry_repo):
    registry = copy.deepcopy(_registry())
    record = _current_by_entity(registry)["comparison.campaign"]
    record["progress"]["completed"] -= 1

    with pytest.raises(ProjectStateError, match="stopped execution requires completed == total"):
        validate_registry(registry, repo_root=registry_repo)

    record["progress"]["completed"] += 1
    record["next_step"] = "旧队列"
    with pytest.raises(ProjectStateError, match="stopped execution cannot carry next_step"):
        validate_registry(registry, repo_root=registry_repo)


def test_registry_rejects_mixed_verdict_without_mixed_outcomes(registry_repo):
    registry = copy.deepcopy(_registry())
    campaign = next(
        item for item in registry["records"] if item["entity"] == "comparison.campaign"
    )
    campaign["outcome_counts"] = {"table_ready_with_note": 42}

    with pytest.raises(ProjectStateError, match="two non-zero outcome counts"):
        validate_registry(registry, repo_root=registry_repo)


def test_registry_accepts_a_date_only_snapshot_timestamp(registry_repo):
    registry = copy.deepcopy(_registry())
    registry["updated_at"] = "2026-08-16"

    validate_registry(registry, repo_root=registry_repo)


def test_evidence_entries_use_path_only():
    assert all(
        set(source) == {"id", "label", "path", "role"}
        for source in _registry()["evidence"]
    )


def test_registry_rejects_missing_evidence(registry_repo):
    registry = _registry()
    (registry_repo / registry["evidence"][0]["path"]).unlink()

    with pytest.raises(ProjectStateError, match="evidence file is missing"):
        validate_registry(registry, repo_root=registry_repo)


def test_current_record_can_be_updated_in_place_without_supersedes(registry_repo):
    registry = copy.deepcopy(_registry())
    record = _current_by_entity(registry)["main.data_contract"]
    # Optional supersedes applies to a single-version entity. The live data
    # contract now has history; removing that link alone creates two current
    # states, which the validator must reject.
    registry["records"] = [
        row for row in registry["records"]
        if row["entity"] != record["entity"] or row is record
    ]
    record["summary"] = "状态摘要可在当前记录中直接更新。"
    record.pop("supersedes", None)
    record.pop("depends_on", None)

    validate_registry(registry, repo_root=registry_repo)


def test_effective_snapshot_timestamp_follows_a_current_record_update(registry_repo):
    registry = copy.deepcopy(_registry())
    record = _current_by_entity(registry)["atlas.statistical"]
    record["updated_at"] = "2099-09-01"

    validate_registry(registry, repo_root=registry_repo)
    assert current_snapshot(registry)["updated_at"] == "2099-09-01"
    assert "updated_at=2099-09-01" in render_agent_summary(registry)
    assert "_Registry snapshot: `2099-09-01`" in render_status_markdown(
        registry, repo_root=registry_repo
    )


def test_superseding_record_replaces_exactly_one_current_state(registry_repo):
    registry = copy.deepcopy(_registry())
    previous = _current_by_entity(registry)["atlas.statistical"]
    replacement = copy.deepcopy(previous)
    replacement.update(
        {
            "state_id": "atlas.statistical@2026-08-26.reviewed",
            "supersedes": [previous["state_id"]],
            "updated_at": "2026-08-26T09:00:00+08:00",
        }
    )
    registry["records"].append(replacement)
    registry["updated_at"] = replacement["updated_at"]

    validate_registry(registry, repo_root=registry_repo)
    current = _current_by_entity(registry)
    assert current["atlas.statistical"]["state_id"] == replacement["state_id"]


def test_optional_supersedes_cannot_replace_a_newer_snapshot_with_an_older_one(registry_repo):
    registry = copy.deepcopy(_registry())
    previous = _current_by_entity(registry)["atlas.statistical"]
    replacement = copy.deepcopy(previous)
    replacement.update(
        {
            "state_id": "atlas.statistical@2020-01-01.backfill",
            "supersedes": [previous["state_id"]],
            "updated_at": "2020-01-01",
        }
    )
    registry["records"].append(replacement)

    with pytest.raises(ProjectStateError, match="newer than superseded"):
        validate_registry(registry, repo_root=registry_repo)


def test_registry_rejects_supersedes_cycle_and_missing_current_state(registry_repo):
    registry = copy.deepcopy(_registry())
    first = _current_by_entity(registry)["atlas.full"]
    second = copy.deepcopy(first)
    second.update(
        {
            "state_id": "atlas.full@2026-08-17.revision",
            "supersedes": [first["state_id"]],
            "updated_at": "2026-08-17T09:00:00+08:00",
        }
    )
    first["supersedes"] = [second["state_id"]]
    registry["records"].append(second)
    registry["updated_at"] = second["updated_at"]

    with pytest.raises(ProjectStateError, match="newer than superseded|cycle|current state"):
        validate_registry(registry, repo_root=registry_repo)


def test_agent_summary_has_only_the_two_status_axes():
    summary = render_agent_summary(_registry())

    assert "status_axes=execution,scientific_verdict" in summary
    assert "main.r1p" in summary
    assert "authorization" not in summary
    assert "authorized" not in summary
