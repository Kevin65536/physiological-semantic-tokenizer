#!/usr/bin/env python3
"""Add canonical subject, record, branch, and join keys to clean cache metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.clean_physiology_cache import with_canonical_fields  # noqa: E402
from src.utils.io import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default="data/cache/physiology_semantic_clean_v4")
    return parser.parse_args()


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


def _normalize_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(with_canonical_fields(json.loads(line)))
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root)
    if not cache_root.is_absolute():
        cache_root = PROJECT_ROOT / cache_root

    manifest_path = cache_root / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for row in manifest.get("records", []):
        normalized = with_canonical_fields(row)
        records.append(normalized)
        record_manifest = (PROJECT_ROOT / str(normalized["record_npz"])).with_suffix(".manifest.json")
        if record_manifest.exists():
            write_json(record_manifest, _jsonable(normalized), ensure_ascii=False)
    manifest["records"] = records
    manifest["canonical_join_contract"] = {
        "schema": "clean_physiology_cache_index_v1",
        "key_fields": ["dataset_id", "canonical_subject_id", "base_record_id"],
        "join_key": "dataset_id|canonical_subject_id|base_record_id",
        "signal_branch": "separates multiple signal exports for the same canonical record",
    }
    write_json(manifest_path, _jsonable(manifest), ensure_ascii=False)

    event_rows = _normalize_jsonl(cache_root / "event_index" / "events.jsonl")
    report_rows = _normalize_jsonl(cache_root / "event_index" / "alignment_reports.jsonl")
    event_manifest_path = cache_root / "event_index" / "event_manifest.json"
    if event_manifest_path.exists():
        event_manifest = json.loads(event_manifest_path.read_text(encoding="utf-8"))
        event_manifest["canonical_join_contract"] = manifest["canonical_join_contract"]
        write_json(event_manifest_path, _jsonable(event_manifest), ensure_ascii=False)

    print(
        json.dumps(
            {
                "cache_root": str(cache_root),
                "records": len(records),
                "events": event_rows,
                "alignment_reports": report_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
