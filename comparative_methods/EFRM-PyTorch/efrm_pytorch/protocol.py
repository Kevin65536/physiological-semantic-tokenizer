"""Leakage-safe reuse of the shared STA-Net split registry for EFRM pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data.clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT

from .data import EFRMSyncPretrainDataset
from .tasks import EFRMUnifiedTaskDataset, get_task_spec


BOUNDARY_SCHEMA = "efrm_pretraining_boundary_v1"
TRIAL_MIXED_BOUNDARY_SCHEMA = "efrm_trial_mixed_boundary_v1"
PUBLIC_SPLIT_SCHEMAS = {"sta_net_split_registry_v2", "sta_net_subject_split_v1"}
SOURCE_TARGET_COHORT_SCHEMA = "efrm_source_target_cohort_v1"
LODO_PRETRAINING_MANIFEST_SCHEMA = "efrm_lodo_pretraining_manifest_v2"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PublicSplitSubjects:
    task: str
    dataset_id: str
    manifest_path: str
    manifest_sha256: str
    manifest_schema: str
    train_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]
    allowed_subjects: tuple[str, ...]
    metadata_sha256: str | None


def _assert_public_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    if "protected" in {part.lower() for part in path.parts}:
        raise PermissionError(f"refusing to read a protected split manifest: {path}")
    if manifest.get("schema") not in PUBLIC_SPLIT_SCHEMAS:
        raise ValueError(f"unsupported public split schema in {path}: {manifest.get('schema')}")
    if manifest.get("protected_test_opened", manifest.get("reserved_test_opened", False)):
        raise PermissionError(f"split manifest reports an opened protected test: {path}")
    forbidden = {"test_indices", "reserved_test_indices", "protected_indices"}.intersection(manifest)
    if forbidden:
        raise PermissionError(f"public manifest exposes protected indices: {sorted(forbidden)}")


def load_public_split_subjects(
    path: str | Path,
    *,
    task: str | None = None,
    cache_root: str = DEFAULT_CLEAN_CACHE_ROOT,
) -> PublicSplitSubjects:
    """Resolve train/validation subjects without ever reading protected files."""

    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _assert_public_manifest(manifest_path, manifest)
    task_key = str(task or manifest.get("task") or manifest_path.parent.name)
    spec = get_task_spec(task_key)
    dataset = EFRMUnifiedTaskDataset(spec, cache_root=cache_root)

    if manifest["schema"] == "sta_net_split_registry_v2":
        train_indices, validation_indices = dataset.validate_shared_public_split(manifest_path)
        train = {dataset.lightweight_metadata(index)["subject"] for index in train_indices}
        validation = {
            dataset.lightweight_metadata(index)["subject"] for index in validation_indices
        }
        metadata_sha256 = str(manifest.get("metadata_sha256") or "") or None
    else:
        train = {str(value) for value in manifest["train_subjects"]}
        validation = {str(value) for value in manifest["validation_subjects"]}
        reserved = {str(value) for value in manifest.get("reserved_test_subjects", ())}
        if (train | validation) & reserved:
            raise RuntimeError(f"public and reserved subject sets overlap in {manifest_path}")
        available = {
            dataset.lightweight_metadata(index)["subject"] for index in range(len(dataset))
        }
        unknown = (train | validation) - available
        if unknown:
            raise RuntimeError(f"split contains subjects absent from task ordering: {sorted(unknown)}")
        metadata_sha256 = dataset.metadata_fingerprint()

    if not train or not validation or train & validation:
        raise RuntimeError(f"train/validation subjects are empty or overlapping in {manifest_path}")
    return PublicSplitSubjects(
        task=task_key,
        dataset_id=spec.dataset_id,
        manifest_path=str(manifest_path),
        manifest_sha256=sha256_file(manifest_path),
        manifest_schema=str(manifest["schema"]),
        train_subjects=tuple(sorted(train)),
        validation_subjects=tuple(sorted(validation)),
        allowed_subjects=tuple(sorted(train | validation)),
        metadata_sha256=metadata_sha256,
    )


class PretrainingBoundary:
    """Subject admission boundary shared by SSL train and validation loaders.

    If several task manifests use the same dataset, the common checkpoint keeps
    only subjects admitted by every manifest. A validation role in any task wins
    over a training role, preventing a downstream validation subject from
    entering self-supervised training through a second task.
    """

    def __init__(self, sources: Sequence[PublicSplitSubjects], *, mode: str) -> None:
        if not sources:
            raise ValueError("at least one public split manifest is required")
        self.sources = tuple(sources)
        self.mode = str(mode)
        by_dataset: dict[str, list[PublicSplitSubjects]] = {}
        for source in self.sources:
            by_dataset.setdefault(source.dataset_id, []).append(source)

        self.train_subjects: dict[str, tuple[str, ...]] = {}
        self.validation_subjects: dict[str, tuple[str, ...]] = {}
        for dataset_id, rows in by_dataset.items():
            common_allowed = set(rows[0].allowed_subjects)
            for row in rows[1:]:
                common_allowed.intersection_update(row.allowed_subjects)
            validation = common_allowed.intersection(
                set().union(*(set(row.validation_subjects) for row in rows))
            )
            train = common_allowed - validation
            if not train or not validation or train & validation:
                raise RuntimeError(
                    f"strict common boundary is empty or overlapping for {dataset_id}: "
                    f"train={len(train)}, validation={len(validation)}"
                )
            self.train_subjects[dataset_id] = tuple(sorted(train))
            self.validation_subjects[dataset_id] = tuple(sorted(validation))

    @classmethod
    def from_manifests(
        cls,
        paths: Sequence[str | Path],
        *,
        tasks: Sequence[str] | None = None,
        mode: str = "development_public_only",
        cache_root: str = DEFAULT_CLEAN_CACHE_ROOT,
    ) -> "PretrainingBoundary":
        if tasks is not None and len(tasks) != len(paths):
            raise ValueError("tasks must be omitted or have one entry per manifest")
        rows = [
            load_public_split_subjects(
                path,
                task=None if tasks is None else tasks[index],
                cache_root=cache_root,
            )
            for index, path in enumerate(paths)
        ]
        return cls(rows, mode=mode)

    def indices_for(self, dataset: EFRMSyncPretrainDataset, role: str) -> list[int]:
        if role not in {"train", "validation"}:
            raise ValueError("role must be train or validation")
        admitted = self.train_subjects if role == "train" else self.validation_subjects
        indices: list[int] = []
        for index in range(len(dataset)):
            row = dataset.lightweight_metadata(index)
            if row["subject"] in admitted.get(row["dataset_id"], ()):
                indices.append(index)
        if not indices:
            raise RuntimeError(f"no pretraining windows survive the {role} boundary")
        return indices

    def assert_disjoint(self) -> None:
        for dataset_id in sorted(set(self.train_subjects) | set(self.validation_subjects)):
            overlap = set(self.train_subjects.get(dataset_id, ())).intersection(
                self.validation_subjects.get(dataset_id, ())
            )
            if overlap:
                raise RuntimeError(f"pretraining subject leakage in {dataset_id}: {sorted(overlap)}")

    def manifest(self) -> dict[str, Any]:
        self.assert_disjoint()
        result = {
            "schema": BOUNDARY_SCHEMA,
            "mode": self.mode,
            "protected_test_opened": False,
            "combination_rule": (
                "intersection_of_allowed_subjects_per_dataset; "
                "validation_role_wins_over_training_role"
            ),
            "train_subjects_by_dataset": self.train_subjects,
            "validation_subjects_by_dataset": self.validation_subjects,
            "sources": [asdict(source) for source in self.sources],
        }
        result["boundary_sha256"] = _stable_hash(result)
        return result


class SourceTargetBoundary:
    """Frozen source-only pretraining boundary for the dual-protocol track."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        cohort = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if cohort.get("schema") != SOURCE_TARGET_COHORT_SCHEMA:
            raise ValueError(
                f"expected {SOURCE_TARGET_COHORT_SCHEMA} at {self.manifest_path}"
            )
        if cohort.get("protocol_id") != "efrm_resource_bounded_dual_protocol_v1":
            raise ValueError("source/target cohort protocol ID does not match frozen v1")
        if cohort.get("target_opened_during_pretraining") is not False:
            raise PermissionError(
                "source-only pretraining requires target_opened_during_pretraining=false"
            )
        datasets = cohort.get("datasets", {})
        if not datasets:
            raise ValueError("source/target cohort manifest has no datasets")
        self.train_subjects: dict[str, tuple[str, ...]] = {}
        self.validation_subjects: dict[str, tuple[str, ...]] = {}
        self.target_subjects: dict[str, tuple[str, ...]] = {}
        for dataset_id, row in sorted(datasets.items()):
            source = {str(value) for value in row["source_subjects"]}
            train = {str(value) for value in row["source_train_subjects"]}
            validation = {str(value) for value in row["source_validation_subjects"]}
            target = {str(value) for value in row["target_subjects"]}
            if not train or not validation or train & validation:
                raise RuntimeError(
                    f"invalid source train/validation boundary for {dataset_id}"
                )
            if train | validation != source:
                raise RuntimeError(
                    f"source roles do not exactly cover source cohort for {dataset_id}"
                )
            if source & target:
                raise RuntimeError(f"source/target subjects overlap for {dataset_id}")
            self.train_subjects[str(dataset_id)] = tuple(sorted(train))
            self.validation_subjects[str(dataset_id)] = tuple(sorted(validation))
            self.target_subjects[str(dataset_id)] = tuple(sorted(target))
        self.cohort = cohort

    def indices_for(self, dataset: EFRMSyncPretrainDataset, role: str) -> list[int]:
        if role not in {"train", "validation"}:
            raise ValueError("role must be train or validation")
        admitted = self.train_subjects if role == "train" else self.validation_subjects
        indices = [
            index
            for index in range(len(dataset))
            if str(dataset.lightweight_metadata(index)["subject"])
            in admitted.get(str(dataset.lightweight_metadata(index)["dataset_id"]), ())
        ]
        if not indices:
            raise RuntimeError(f"no pretraining windows survive the source {role} boundary")
        return indices

    def manifest(self) -> dict[str, Any]:
        result = {
            "schema": BOUNDARY_SCHEMA,
            "mode": "source_target_source_only_v1",
            "protocol_id": "efrm_resource_bounded_dual_protocol_v1",
            "protected_test_opened": False,
            "target_opened_during_pretraining": False,
            "cohort_manifest_path": str(self.manifest_path),
            "cohort_manifest_sha256": sha256_file(self.manifest_path),
            "train_subjects_by_dataset": self.train_subjects,
            "validation_subjects_by_dataset": self.validation_subjects,
            "target_subjects_by_dataset": self.target_subjects,
            "combination_rule": "frozen dataset-level source cohort; source validation held out",
        }
        result["boundary_sha256"] = _stable_hash(result)
        return result


class LODOPretrainingBoundary:
    """Target-dataset-excluded boundary for the frozen EFRM v2 protocol."""

    def __init__(self, manifest_path: str | Path, *, stage: str) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != LODO_PRETRAINING_MANIFEST_SCHEMA:
            raise ValueError(
                f"expected {LODO_PRETRAINING_MANIFEST_SCHEMA} at "
                f"{self.manifest_path}"
            )
        if manifest.get("protocol_id") != "efrm_lodo_full_target_fivefold_v2":
            raise ValueError("LODO manifest protocol ID does not match frozen v2")
        if manifest.get("target_dataset_exposure") is not False:
            raise PermissionError("LODO pretraining requires target_dataset_exposure=false")
        if stage not in {"selection", "final_refit"}:
            raise ValueError("LODO stage must be selection or final_refit")

        self.stage = stage
        self.target_dataset = str(manifest["excluded_target_dataset"])
        included = tuple(str(value) for value in manifest["included_datasets"])
        if not included or self.target_dataset in included:
            raise RuntimeError("invalid LODO included/excluded dataset declaration")

        datasets = manifest.get("datasets", {})
        if set(datasets) != set(included):
            raise RuntimeError("LODO dataset rows do not exactly match included datasets")
        self.train_subjects: dict[str, tuple[str, ...]] = {}
        self.validation_subjects: dict[str, tuple[str, ...]] = {}
        self.all_subjects: dict[str, tuple[str, ...]] = {}
        for dataset_id, row in sorted(datasets.items()):
            all_subjects = {str(value) for value in row["all_subjects"]}
            selection_train = {
                str(value) for value in row["selection_train_subjects"]
            }
            selection_validation = {
                str(value) for value in row["selection_validation_subjects"]
            }
            if (
                not selection_train
                or not selection_validation
                or selection_train & selection_validation
                or selection_train | selection_validation != all_subjects
            ):
                raise RuntimeError(
                    f"invalid LODO selection split for {dataset_id}"
                )
            self.all_subjects[str(dataset_id)] = tuple(sorted(all_subjects))
            if stage == "selection":
                self.train_subjects[str(dataset_id)] = tuple(
                    sorted(selection_train)
                )
            else:
                self.train_subjects[str(dataset_id)] = tuple(sorted(all_subjects))
            # Stage-B validation is diagnostic only. It deliberately overlaps
            # the full refit train pool and cannot select or stop a checkpoint.
            self.validation_subjects[str(dataset_id)] = tuple(
                sorted(selection_validation)
            )
        self.manifest_data = manifest

    def indices_for(self, dataset: EFRMSyncPretrainDataset, role: str) -> list[int]:
        if role not in {"train", "validation"}:
            raise ValueError("role must be train or validation")
        admitted = self.train_subjects if role == "train" else self.validation_subjects
        indices: list[int] = []
        for index in range(len(dataset)):
            row = dataset.lightweight_metadata(index)
            dataset_id = str(row["dataset_id"])
            if dataset_id == self.target_dataset:
                raise PermissionError(
                    f"active target dataset {self.target_dataset} entered LODO loader"
                )
            if str(row["subject"]) in admitted.get(dataset_id, ()):
                indices.append(index)
        if not indices:
            raise RuntimeError(f"no pretraining windows survive LODO {role} boundary")
        return indices

    def manifest(self) -> dict[str, Any]:
        result = {
            "schema": BOUNDARY_SCHEMA,
            "mode": f"lodo_{self.stage}_v2",
            "protocol_id": "efrm_lodo_full_target_fivefold_v2",
            "protected_test_opened": False,
            "target_dataset_exposure": False,
            "excluded_target_dataset": self.target_dataset,
            "included_datasets": tuple(sorted(self.all_subjects)),
            "lodo_manifest_path": str(self.manifest_path),
            "lodo_manifest_sha256": sha256_file(self.manifest_path),
            "train_subjects_by_dataset": self.train_subjects,
            "validation_subjects_by_dataset": self.validation_subjects,
            "all_included_subjects_by_dataset": self.all_subjects,
            "validation_role": (
                "checkpoint_selection"
                if self.stage == "selection"
                else "diagnostic_only_overlaps_full_refit_train"
            ),
            "checkpoint_selection_allowed": self.stage == "selection",
            "early_stopping_allowed": self.stage == "selection",
        }
        result["boundary_sha256"] = _stable_hash(result)
        return result


class TrialMixedBoundary:
    """Deterministic within-subject trial split over public subjects only.

    Every subject/record/condition stratum contributes trials to both roles.
    This is intentionally a relaxed diagnostic boundary: subjects, records,
    preprocessing state, and task conditions are shared across train and
    validation, while exact trial indices remain disjoint.
    """

    def __init__(
        self,
        dataset: EFRMSyncPretrainDataset,
        source: PublicSplitSubjects,
        *,
        validation_fraction: float = 0.2,
        seed: int = 42,
        mode: str = "within_subject_trial_mixed_public",
    ) -> None:
        if not 0.0 < float(validation_fraction) < 1.0:
            raise ValueError("validation_fraction must lie strictly between zero and one")
        self.dataset = dataset
        self.source = source
        self.validation_fraction = float(validation_fraction)
        self.seed = int(seed)
        self.mode = str(mode)

        allowed_subjects = set(source.allowed_subjects)
        groups: dict[tuple[str, str, str], list[int]] = {}
        excluded_subjects: set[str] = set()
        for index in range(len(dataset)):
            row = dataset.lightweight_metadata(index)
            subject = str(row["subject"])
            if row["dataset_id"] != source.dataset_id or subject not in allowed_subjects:
                excluded_subjects.add(subject)
                continue
            stratum = (
                subject,
                str(row["join_key"]),
                str(row["condition"]),
            )
            groups.setdefault(stratum, []).append(index)
        if not groups:
            raise RuntimeError("no public trials survive the mixed-trial boundary")

        train: list[int] = []
        validation: list[int] = []
        self.stratum_counts: dict[str, dict[str, int]] = {}
        for stratum, indices in sorted(groups.items()):
            ranked = sorted(
                indices,
                key=lambda index: _stable_hash({
                    "seed": self.seed,
                    "index": index,
                    "metadata": dataset.lightweight_metadata(index),
                }),
            )
            validation_count = max(
                1, int(round(len(ranked) * self.validation_fraction))
            )
            if validation_count >= len(ranked):
                raise RuntimeError(
                    "trial-mixed split requires at least two trials per "
                    f"subject/record/condition stratum: {stratum}"
                )
            validation.extend(ranked[:validation_count])
            train.extend(ranked[validation_count:])
            self.stratum_counts["|".join(stratum)] = {
                "total": len(ranked),
                "train": len(ranked) - validation_count,
                "validation": validation_count,
            }

        self.train_indices = tuple(sorted(train))
        self.validation_indices = tuple(sorted(validation))
        self.excluded_subjects = tuple(sorted(excluded_subjects - allowed_subjects))
        self._validate()

    def _validate(self) -> None:
        train = set(self.train_indices)
        validation = set(self.validation_indices)
        if not train or not validation or train & validation:
            raise RuntimeError("mixed-trial boundary is empty or has overlapping trials")
        train_subjects = {
            self.dataset.lightweight_metadata(index)["subject"]
            for index in self.train_indices
        }
        validation_subjects = {
            self.dataset.lightweight_metadata(index)["subject"]
            for index in self.validation_indices
        }
        expected = set(self.source.allowed_subjects)
        if train_subjects != expected or validation_subjects != expected:
            raise RuntimeError(
                "every public subject must appear in both mixed-trial roles"
            )

    def indices_for(
        self, dataset: EFRMSyncPretrainDataset, role: str
    ) -> list[int]:
        if dataset is not self.dataset:
            raise ValueError("mixed-trial boundary must be used with its source dataset")
        if role == "train":
            return list(self.train_indices)
        if role == "validation":
            return list(self.validation_indices)
        raise ValueError("role must be train or validation")

    def manifest(self) -> dict[str, Any]:
        self._validate()
        train_subjects = sorted({
            self.dataset.lightweight_metadata(index)["subject"]
            for index in self.train_indices
        })
        validation_subjects = sorted({
            self.dataset.lightweight_metadata(index)["subject"]
            for index in self.validation_indices
        })
        result = {
            "schema": TRIAL_MIXED_BOUNDARY_SCHEMA,
            "mode": self.mode,
            "strategy": "within_subject_record_condition_stratified_hash_v1",
            "protected_test_opened": False,
            "preprocessing_leakage_boundary": (
                "full-record normalization state is shared across trial roles"
            ),
            "seed": self.seed,
            "validation_fraction": self.validation_fraction,
            "public_subject_source": asdict(self.source),
            "train_indices": self.train_indices,
            "validation_indices": self.validation_indices,
            "train_subjects": train_subjects,
            "validation_subjects": validation_subjects,
            "subject_overlap_count": len(set(train_subjects) & set(validation_subjects)),
            "trial_overlap_count": 0,
            "stratum_counts": self.stratum_counts,
            "excluded_nonpublic_subjects": self.excluded_subjects,
        }
        result["boundary_sha256"] = _stable_hash(result)
        return result


def role_counts(
    dataset: EFRMSyncPretrainDataset,
    indices: Iterable[int],
) -> dict[str, Any]:
    datasets: dict[str, int] = {}
    subjects: dict[str, set[str]] = {}
    records: dict[str, set[str]] = {}
    total = 0
    for index in indices:
        row = dataset.lightweight_metadata(index)
        dataset_id = row["dataset_id"]
        total += 1
        datasets[dataset_id] = datasets.get(dataset_id, 0) + 1
        subjects.setdefault(dataset_id, set()).add(row["subject"])
        records.setdefault(dataset_id, set()).add(row["join_key"])
    return {
        "sample_count": total,
        "sample_count_by_dataset": datasets,
        "subject_count_by_dataset": {key: len(value) for key, value in subjects.items()},
        "record_count_by_dataset": {key: len(value) for key, value in records.items()},
    }
