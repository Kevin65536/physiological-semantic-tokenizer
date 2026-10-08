"""End-to-end Token Physiology Atlas orchestration.

This module turns a versioned physiology-semantic export into auditable,
subject-balanced token phenotype tables.  It intentionally keeps three
concepts separate:

* measurement features, which depend on the input patches and preprocessing;
* checkpoint assignments, which depend on the tokenizer checkpoint; and
* descriptive associations, which do not name a token as a physiological
  state.

The public entry point :func:`build_token_physiology_atlas` consumes train/val
export paths, writes CSV/JSON/NPZ artifacts atomically, and leaves the protected
test split sealed unless its caller explicitly authorizes it.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .physiological_patch_features import (
    DEFAULT_FEATURE_SPEC,
    FrequencyBand,
    PatchFeatureBatch,
    PhysiologicalPatchFeatureSpec,
    extract_physiological_patch_features,
)
from .token_information_ledger import (
    InformationLedgerConfig,
    build_token_representations,
    evaluate_information_ledger,
)
from .token_physiology import (
    TokenPhysiologyConfig,
    TokenPhysiologyResult,
    analyze_token_physiology,
    match_token_signatures,
)
from .token_sequence import (
    analyze_cross_modal_lags,
    markov_log_loss,
    summarize_sequences,
    transition_counts,
)
ATLAS_SCHEMA_VERSION = "token_physiology_atlas_v1"
SUPPORTED_EXPORT_SCHEMAS = {"physiology_semantic_export_v3"}
MODALITIES = ("eeg", "fnirs")


@dataclass
class ModalitySplitAtlas:
    """In-memory result for one split and modality."""

    split: str
    modality: str
    sample_ids: np.ndarray
    subjects_by_sample: np.ndarray
    hard_ids: np.ndarray
    posterior: np.ndarray | None
    valid_mask: np.ndarray
    feature_values: np.ndarray
    feature_names: tuple[str, ...]
    feature_units: tuple[str, ...]
    token_result: TokenPhysiologyResult
    diagnostic_rows: list[dict[str, Any]]
    exemplar_rows: list[dict[str, Any]]
    measurement_cache: Mapping[str, Any]
    codebook_size: int


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _stage_path(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(name)


def _publish(staged: Path, target: Path, *, force: bool) -> None:
    if force:
        os.replace(staged, target)
        return
    try:
        os.link(staged, target)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {target}") from error
    staged.unlink()


def _write_text_atomic(
    target: Path,
    text: str,
    *,
    force: bool,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")
    staged = _stage_path(target)
    try:
        with staged.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _publish(staged, target, force=force)
    finally:
        staged.unlink(missing_ok=True)


def _write_json_atomic(
    target: Path,
    payload: Mapping[str, Any] | Sequence[Any],
    *,
    force: bool,
) -> None:
    serializable = _json_value(payload)
    text = json.dumps(
        serializable,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    _write_text_atomic(target, text, force=force)


def _write_npz_atomic(
    target: Path,
    payload: Mapping[str, np.ndarray],
    *,
    force: bool,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")
    staged = _stage_path(target)
    try:
        with staged.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        _publish(staged, target, force=force)
    finally:
        staged.unlink(missing_ok=True)


def _csv_scalar(value: Any) -> Any:
    value = _json_value(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv_atomic(
    target: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    force: bool,
) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")
    staged = _stage_path(target)
    try:
        with staged.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {key: _csv_scalar(row.get(key)) for key in fieldnames}
                    )
            handle.flush()
            os.fsync(handle.fileno())
        _publish(staged, target, force=force)
    finally:
        staged.unlink(missing_ok=True)


def _manifest_path(export_path: Path) -> Path:
    return export_path.with_suffix(export_path.suffix + ".manifest.json")


def load_token_export(
    path: str | Path,
    *,
    expected_split: str | None = None,
    allow_test: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and validate one v3 export without enabling pickle."""

    export_path = Path(path).resolve()
    manifest_path = _manifest_path(export_path)
    if not export_path.is_file():
        raise FileNotFoundError(export_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Export manifest is required for provenance: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") not in SUPPORTED_EXPORT_SCHEMAS:
        raise ValueError(
            f"Unsupported export schema {manifest.get('schema')!r}; "
            f"expected one of {sorted(SUPPORTED_EXPORT_SCHEMAS)}"
        )
    split = str(manifest.get("split"))
    if expected_split is not None and split != expected_split:
        raise ValueError(
            f"Export split {split!r} does not match requested {expected_split!r}"
        )
    if split == "test" and not allow_test:
        raise ValueError(
            "Protected test export is sealed; pass explicit allow_test=True only "
            "for an authorized final evaluation"
        )
    with np.load(export_path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    expected_npz_hash = manifest.get("npz_sha256")
    if expected_npz_hash is not None and expected_npz_hash != _sha256(export_path):
        raise ValueError("Export NPZ content hash does not match its manifest")
    sample_count = manifest.get("sample_count", manifest.get("count"))
    if sample_count is None:
        raise ValueError("Export manifest is missing sample_count")
    sample_count = int(sample_count)
    if sample_count <= 0:
        raise ValueError("Export sample_count must be positive")
    for modality in MODALITIES:
        required = [
            f"{modality}_hard_ids",
            f"{modality}_semantic_latent",
            f"{modality}_codebook_embedding",
        ]
        if f"{modality}_patches" not in payload:
            measurement_references = manifest.get("measurement_caches", {})
            if modality not in measurement_references:
                required.append(f"{modality}_patches")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(
                f"{split} export is missing Atlas arrays for {modality}: {missing}. "
                "Re-export with --include-patches."
            )
        hard_ids = np.asarray(payload[f"{modality}_hard_ids"])
        semantic = np.asarray(payload[f"{modality}_semantic_latent"])
        embedding = np.asarray(payload[f"{modality}_codebook_embedding"])
        if hard_ids.ndim != 2 or hard_ids.shape[0] != sample_count:
            raise ValueError(
                f"{modality}_hard_ids must have shape [sample_count,tokens]"
            )
        token_shape = hard_ids.shape
        for name, values in (
            (f"{modality}_semantic_latent", semantic),
            (f"{modality}_codebook_embedding", embedding),
        ):
            if values.ndim != 3 or values.shape[:2] != token_shape:
                raise ValueError(f"{name} must align as [sample_count,tokens,D]")
        for optional_name in (
            f"{modality}_posterior",
            f"{modality}_expected_embedding",
            f"{modality}_residual",
        ):
            if optional_name in payload:
                values = np.asarray(payload[optional_name])
                if values.ndim != 3 or values.shape[:2] != token_shape:
                    raise ValueError(
                        f"{optional_name} must align as [sample_count,tokens,D]"
                    )
        mask_name = f"{modality}_token_valid_mask"
        if mask_name in payload and np.asarray(payload[mask_name]).shape != token_shape:
            raise ValueError(f"{mask_name} must align with hard token IDs")
        patch_name = f"{modality}_patches"
        if patch_name in payload:
            patches = np.asarray(payload[patch_name])
            if patches.ndim != 4 or patches.shape[:2] != token_shape:
                raise ValueError(
                    f"{patch_name} must align as [sample_count,tokens,C,P]"
                )
        codebook_name = f"{modality}_codebook"
        if codebook_name in payload:
            codebook = np.asarray(payload[codebook_name])
            if codebook.ndim != 2 or codebook.shape[0] <= 0:
                raise ValueError(f"{codebook_name} must have shape [K,D]")
            if codebook.shape[1] != semantic.shape[-1]:
                raise ValueError(
                    f"{codebook_name} embedding dimension does not match latent"
                )
            if f"{modality}_posterior" in payload and (
                np.asarray(payload[f"{modality}_posterior"]).shape[-1]
                != codebook.shape[0]
            ):
                raise ValueError(
                    f"{modality} posterior width does not match static codebook"
                )
    sample_key = str(manifest.get("sample_key_array", "sample_id"))
    if sample_key not in payload:
        raise ValueError(f"Manifest sample key {sample_key!r} is absent")
    observed_hash = hashlib.sha256(
        "\n".join(np.asarray(payload[sample_key], dtype=np.str_).tolist()).encode(
            "utf-8"
        )
    ).hexdigest()
    if observed_hash != manifest.get("sample_order_sha256"):
        raise ValueError("Export sample order hash does not match its manifest")
    if len(np.asarray(payload[sample_key]).reshape(-1)) != sample_count:
        raise ValueError("Export sample key does not align with sample_count")
    for name in manifest.get("sample_aligned_arrays", ()):
        if name not in payload:
            raise ValueError(f"Manifest sample-aligned array is absent: {name}")
        values = np.asarray(payload[name])
        if values.ndim == 0 or values.shape[0] != sample_count:
            raise ValueError(
                f"Manifest sample-aligned array does not align: {name}"
            )
    return payload, manifest


def _feature_spec(config: Mapping[str, Any]) -> PhysiologicalPatchFeatureSpec:
    feature_cfg = dict(config.get("features", {}))
    eeg_cfg = dict(feature_cfg.get("eeg", {}))
    if not bool(eeg_cfg.get("absolute_band_power", True)) or not bool(
        eeg_cfg.get("relative_band_power", True)
    ):
        raise ValueError(
            "Atlas v1 requires both EEG absolute and relative band-power features"
        )
    fnirs_cfg = dict(feature_cfg.get("fnirs", {}))
    if not bool(fnirs_cfg.get("local_morphology", True)):
        raise ValueError("Atlas v1 requires the standard fNIRS local morphology set")
    if bool(fnirs_cfg.get("band_power", False)):
        raise ValueError("Short-patch fNIRS band power is outside the Atlas contract")
    band_cfg = eeg_cfg.get("bands_hz")
    if band_cfg:
        bands = tuple(
            FrequencyBand(str(name), float(bounds[0]), float(bounds[1]))
            for name, bounds in band_cfg.items()
        )
    else:
        bands = DEFAULT_FEATURE_SPEC.eeg_bands
    reference = tuple(
        float(value)
        for value in eeg_cfg.get(
            "reference_band_hz", DEFAULT_FEATURE_SPEC.eeg_reference_band_hz
        )
    )
    if len(reference) != 2:
        raise ValueError("features.eeg.reference_band_hz must contain two values")
    return PhysiologicalPatchFeatureSpec(
        eeg_bands=bands,
        eeg_reference_band_hz=(reference[0], reference[1]),
        psd_window=str(
            eeg_cfg.get("psd_window", DEFAULT_FEATURE_SPEC.psd_window)
        ),
        psd_detrend=str(
            eeg_cfg.get("psd_detrend", DEFAULT_FEATURE_SPEC.psd_detrend)
        ),
        minimum_valid_fraction=float(
            feature_cfg.get(
                "minimum_valid_fraction",
                DEFAULT_FEATURE_SPEC.minimum_valid_fraction,
            )
        ),
    )


def _sample_ids(
    payload: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> np.ndarray:
    key = str(manifest.get("sample_key_array", "sample_id"))
    return np.asarray(payload[key], dtype=np.str_)


def _subjects(payload: Mapping[str, np.ndarray], sample_count: int) -> np.ndarray:
    for key in ("subject_key", "subject", "subject_id"):
        if key in payload:
            values = np.asarray(payload[key]).reshape(-1)
            if len(values) != sample_count:
                raise ValueError(f"{key} does not align with exported samples")
            return values.astype(np.str_)
    raise ValueError(
        "Atlas subject-equal analysis requires subject_key, subject, or subject_id"
    )


def _token_mask(
    payload: Mapping[str, np.ndarray],
    modality: str,
    shape: tuple[int, int],
) -> np.ndarray:
    key = f"{modality}_token_valid_mask"
    if key not in payload:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(payload[key], dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{key} shape {mask.shape} does not match {shape}")
    return mask


def _full_posterior(
    payload: Mapping[str, np.ndarray],
    modality: str,
    shape: tuple[int, int],
) -> np.ndarray | None:
    key = f"{modality}_posterior"
    if key not in payload:
        # Truncated top-k posterior is deliberately not renormalized and called
        # a full soft assignment; doing so would overstate boundary certainty.
        return None
    values = np.asarray(payload[key], dtype=np.float64)
    if values.shape[:2] != shape or values.ndim != 3:
        raise ValueError(f"{key} must have shape [samples,tokens,K]")
    return values


def _codebook_size(
    hard_ids: np.ndarray,
    posterior: np.ndarray | None,
    codebook: np.ndarray | None = None,
) -> int:
    declared_sizes: list[int] = []
    if posterior is not None:
        declared_sizes.append(int(posterior.shape[-1]))
    if codebook is not None:
        values = np.asarray(codebook)
        if values.ndim != 2 or values.shape[0] <= 0:
            raise ValueError("Static codebook must have shape [K,D]")
        declared_sizes.append(int(values.shape[0]))
    if declared_sizes:
        if len(set(declared_sizes)) != 1:
            raise ValueError(
                "Posterior width and static codebook size do not agree"
            )
        codebook_size = declared_sizes[0]
        valid = hard_ids[np.isfinite(hard_ids) & (hard_ids >= 0)]
        if len(valid) and int(np.max(valid)) >= codebook_size:
            raise ValueError("Hard token ID lies outside the declared codebook")
        return codebook_size
    valid = hard_ids[np.isfinite(hard_ids) & (hard_ids >= 0)]
    if not len(valid):
        raise ValueError("Cannot infer codebook size from an empty hard sequence")
    return int(np.max(valid)) + 1


def _summary_features(
    extracted: PatchFeatureBatch,
    *,
    modality: str,
    patches: np.ndarray,
    token_mask: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    values = np.where(extracted.feature_valid_mask, extracted.values, np.nan)
    if modality == "eeg":
        finite = np.isfinite(values)
        counts = finite.sum(axis=2)
        channel_mean = np.divide(
            np.where(finite, values, 0.0).sum(axis=2),
            counts,
            out=np.full(values.shape[:2] + (values.shape[-1],), np.nan),
            where=counts > 0,
        )
        centered = values - channel_mean[:, :, None, :]
        channel_variance = np.divide(
            np.where(finite, np.square(centered), 0.0).sum(axis=2),
            counts,
            out=np.full_like(channel_mean, np.nan),
            where=counts > 0,
        )
        channel_sd = np.sqrt(channel_variance)
        summary = np.concatenate((channel_mean, channel_sd), axis=-1)
        names = tuple(
            [f"channel_mean/{name}" for name in extracted.feature_names]
            + [f"channel_sd/{name}" for name in extracted.feature_names]
        )
        units = tuple(extracted.feature_units + extracted.feature_units)
    else:
        if values.shape[2] != 2:
            raise ValueError(
                "Standard fNIRS Atlas contract requires two ordered HbO/HbR channels"
            )
        flattened = values.reshape(values.shape[0], values.shape[1], -1)
        names = tuple(
            f"{role}/{name}"
            for role in ("HbO", "HbR")
            for name in extracted.feature_names
        )
        units = tuple(extracted.feature_units * 2)

        hbo = np.asarray(patches[:, :, 0, :], dtype=np.float64)
        hbr = np.asarray(patches[:, :, 1, :], dtype=np.float64)
        finite = np.isfinite(hbo) & np.isfinite(hbr)
        count = finite.sum(axis=-1)
        hbo_mean = np.divide(
            np.where(finite, hbo, 0.0).sum(axis=-1),
            count,
            out=np.full(hbo.shape[:2], np.nan),
            where=count > 1,
        )
        hbr_mean = np.divide(
            np.where(finite, hbr, 0.0).sum(axis=-1),
            count,
            out=np.full(hbr.shape[:2], np.nan),
            where=count > 1,
        )
        hbo_centered = np.where(finite, hbo - hbo_mean[..., None], 0.0)
        hbr_centered = np.where(finite, hbr - hbr_mean[..., None], 0.0)
        covariance = np.sum(hbo_centered * hbr_centered, axis=-1)
        scale = np.sqrt(
            np.sum(np.square(hbo_centered), axis=-1)
            * np.sum(np.square(hbr_centered), axis=-1)
        )
        correlation = np.divide(
            covariance,
            scale,
            out=np.full(hbo.shape[:2], np.nan),
            where=(count > 1) & (scale > 0),
        )
        summary = np.concatenate((flattened, correlation[..., None]), axis=-1)
        names = names + ("HbO_HbR/within_patch_correlation",)
        units = units + ("dimensionless",)
    summary[~token_mask] = np.nan
    return summary.astype(np.float32), names, units


def _measurement_cache_key(
    patches: np.ndarray,
    token_mask: np.ndarray,
    sample_ids: np.ndarray,
    *,
    modality: str,
    sample_rate_hz: float,
    feature_spec_hash: str,
    channel_identity: np.ndarray | None = None,
) -> str:
    identity = json.dumps(
        {
            "cache_schema": "token_physiology_measurement_cache_v2",
            "modality": modality,
            "sample_rate_hz": float(sample_rate_hz),
            "feature_spec_hash": feature_spec_hash,
            "sample_ids": sample_ids.tolist(),
            "channel_identity": (
                np.asarray(channel_identity, dtype=np.str_).tolist()
                if channel_identity is not None
                else None
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(identity)
    digest.update(_array_digest(patches, token_mask).encode("ascii"))
    return digest.hexdigest()


def _load_or_extract_measurements(
    payload: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    *,
    split: str,
    modality: str,
    sample_rate_hz: float,
    spec: PhysiologicalPatchFeatureSpec,
    cache_dir: Path,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], Mapping[str, Any]]:
    hard_ids = np.asarray(payload[f"{modality}_hard_ids"])
    patch_key = f"{modality}_patches"
    if patch_key not in payload:
        reference = manifest.get("measurement_caches", {}).get(modality)
        if not isinstance(reference, Mapping):
            raise ValueError(
                f"{modality} export has no patches or measurement-cache reference"
            )
        cache_path = Path(str(reference.get("path", ""))).resolve()
        sidecar = cache_path.with_suffix(".manifest.json")
        if not cache_path.is_file() or not sidecar.is_file():
            raise FileNotFoundError(
                f"Referenced measurement cache is incomplete: {cache_path}, {sidecar}"
            )
        cached_manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        expected_key = str(reference.get("measurement_cache_key", ""))
        checks = {
            "cache schema": cached_manifest.get("schema")
            == "token_physiology_measurement_cache_v2",
            "reference key": expected_key
            == cached_manifest.get("measurement_cache_key"),
            "reference feature spec": reference.get("feature_spec_hash")
            == cached_manifest.get("feature_spec_hash"),
            "feature spec": spec.spec_hash
            == cached_manifest.get("feature_spec_hash"),
            "reference sample order": reference.get(
                "source_sample_order_sha256"
            )
            == cached_manifest.get("source_sample_order_sha256"),
            "sample order": manifest.get("sample_order_sha256")
            == cached_manifest.get("source_sample_order_sha256"),
            "modality": modality == cached_manifest.get("modality"),
            "sample rate": np.isclose(
                sample_rate_hz,
                float(cached_manifest.get("sample_rate_hz", np.nan)),
            ),
            "cache hash": reference.get("npz_sha256") == _sha256(cache_path),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(
                f"Referenced {modality} measurement cache failed: {failed}"
            )
        with np.load(cache_path, allow_pickle=False) as archive:
            required_arrays = {
                "token_features",
                "feature_names",
                "feature_units",
                "sample_ids",
                "token_valid_mask",
                "channel_feature_values",
                "channel_feature_valid_mask",
                "channel_valid_mask",
                "valid_sample_fraction",
            }
            missing_arrays = sorted(required_arrays - set(archive.files))
            if missing_arrays:
                raise ValueError(
                    f"Referenced {modality} cache is missing arrays: "
                    f"{missing_arrays}"
                )
            summary = archive["token_features"]
            channel_values_shape = archive["channel_feature_values"].shape
            channel_feature_valid_shape = archive[
                "channel_feature_valid_mask"
            ].shape
            channel_valid_shape = archive["channel_valid_mask"].shape
            valid_fraction_shape = archive["valid_sample_fraction"].shape
            names = tuple(archive["feature_names"].astype(str).tolist())
            units = tuple(archive["feature_units"].astype(str).tolist())
            cached_sample_ids = archive["sample_ids"].astype(str)
            cached_token_mask = np.asarray(
                archive["token_valid_mask"], dtype=bool
            )
            if modality == "eeg":
                assignment_has_channels = "selected_eeg_channels" in payload
                cache_has_channels = "selected_eeg_channels" in archive.files
                if assignment_has_channels != cache_has_channels:
                    raise ValueError(
                        "Referenced EEG cache channel identity presence mismatch"
                    )
                if assignment_has_channels and not np.array_equal(
                    np.asarray(
                        archive["selected_eeg_channels"],
                        dtype=np.str_,
                    ),
                    np.asarray(
                        payload["selected_eeg_channels"],
                        dtype=np.str_,
                    ),
                ):
                    raise ValueError(
                        "Referenced EEG cache channel identities do not align"
                    )
        if (
            summary.ndim != 3
            or summary.shape[:2] != hard_ids.shape
            or summary.shape[-1] != len(names)
            or len(names) != len(units)
            or len(channel_values_shape) != 4
            or channel_values_shape[:2] != hard_ids.shape
            or channel_feature_valid_shape != channel_values_shape
            or channel_valid_shape != channel_values_shape[:3]
            or valid_fraction_shape != channel_values_shape[:3]
            or not np.array_equal(
                cached_sample_ids,
                _sample_ids(payload, manifest).astype(str),
            )
            or not np.array_equal(
                cached_token_mask,
                _token_mask(payload, modality, hard_ids.shape),
            )
        ):
            raise ValueError(
                f"Referenced {modality} measurement cache alignment mismatch"
            )
        return summary, names, units, {
            "status": "hit",
            "path": str(cache_path),
            "measurement_cache_key": expected_key,
            "manifest": cached_manifest,
            "source": "assignment_manifest_reference",
        }

    patches = np.asarray(payload[patch_key])
    if patches.ndim != 4 or hard_ids.shape != patches.shape[:2]:
        raise ValueError(
            f"{modality} patches/hard IDs must align as [samples,tokens,...]"
        )
    mask = _token_mask(payload, modality, hard_ids.shape)
    sample_ids = _sample_ids(payload, manifest)
    key = _measurement_cache_key(
        patches,
        mask,
        sample_ids,
        modality=modality,
        sample_rate_hz=sample_rate_hz,
        feature_spec_hash=spec.spec_hash,
        channel_identity=(
            np.asarray(payload["selected_eeg_channels"], dtype=np.str_)
            if modality == "eeg" and "selected_eeg_channels" in payload
            else np.asarray(("HbO", "HbR"), dtype=np.str_)
            if modality == "fnirs"
            else None
        ),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_{modality}_{key[:24]}.npz"
    sidecar = cache_path.with_suffix(".manifest.json")
    if cache_path.is_file() != sidecar.is_file():
        raise RuntimeError(
            "Incomplete measurement cache pair; remove or quarantine the exact "
            f"orphan after audit: {cache_path}, {sidecar}"
        )
    if cache_path.is_file() and sidecar.is_file():
        cached_manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            cached_manifest.get("schema")
            != "token_physiology_measurement_cache_v2"
        ):
            raise ValueError(f"Measurement cache schema mismatch: {cache_path}")
        if cached_manifest.get("measurement_cache_key") != key:
            raise ValueError(f"Measurement cache key mismatch: {cache_path}")
        if cached_manifest.get("npz_sha256") != _sha256(cache_path):
            raise ValueError(f"Measurement cache content hash mismatch: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as archive:
            required_arrays = {
                "token_features",
                "feature_names",
                "feature_units",
                "sample_ids",
                "token_valid_mask",
                "channel_feature_values",
                "channel_feature_valid_mask",
                "channel_valid_mask",
                "valid_sample_fraction",
            }
            missing_arrays = sorted(required_arrays - set(archive.files))
            if missing_arrays:
                raise ValueError(
                    f"Measurement cache is missing arrays: {missing_arrays}"
                )
            summary = archive["token_features"]
            channel_values_shape = archive["channel_feature_values"].shape
            channel_feature_valid_shape = archive[
                "channel_feature_valid_mask"
            ].shape
            channel_valid_shape = archive["channel_valid_mask"].shape
            valid_fraction_shape = archive["valid_sample_fraction"].shape
            names = tuple(archive["feature_names"].astype(str).tolist())
            units = tuple(archive["feature_units"].astype(str).tolist())
            cached_sample_ids = np.asarray(archive["sample_ids"], dtype=np.str_)
            cached_token_mask = np.asarray(
                archive["token_valid_mask"], dtype=bool
            )
        if (
            summary.ndim != 3
            or summary.shape[:2] != hard_ids.shape
            or summary.shape[-1] != len(names)
            or len(names) != len(units)
            or len(channel_values_shape) != 4
            or channel_values_shape[:2] != hard_ids.shape
            or channel_feature_valid_shape != channel_values_shape
            or channel_valid_shape != channel_values_shape[:3]
            or valid_fraction_shape != channel_values_shape[:3]
            or not np.array_equal(cached_sample_ids, sample_ids)
            or not np.array_equal(cached_token_mask, mask)
        ):
            raise ValueError(f"Measurement cache alignment mismatch: {cache_path}")
        return summary, names, units, {
            "status": "hit",
            "path": str(cache_path),
            "measurement_cache_key": key,
            "manifest": cached_manifest,
        }

    channel_names: Sequence[str] | None
    if modality == "fnirs":
        channel_names = ("HbO", "HbR")
    else:
        channel_names = tuple(f"local_eeg_channel_{index}" for index in range(patches.shape[2]))
    extracted = extract_physiological_patch_features(
        patches,
        modality=modality,
        sample_rate_hz=sample_rate_hz,
        valid_mask=mask,
        channel_names=channel_names,
        spec=spec,
    )
    summary, names, units = _summary_features(
        extracted,
        modality=modality,
        patches=patches,
        token_mask=mask,
    )
    cache_payload: dict[str, np.ndarray] = {
        "token_features": summary,
        "feature_names": np.asarray(names, dtype=np.str_),
        "feature_units": np.asarray(units, dtype=np.str_),
        "channel_feature_values": extracted.values.astype(np.float32),
        "channel_feature_valid_mask": extracted.feature_valid_mask,
        "channel_valid_mask": extracted.channel_valid_mask,
        "valid_sample_fraction": extracted.valid_sample_fraction,
        "sample_ids": sample_ids,
        "token_valid_mask": mask,
    }
    if modality == "eeg" and "selected_eeg_channels" in payload:
        cache_payload["selected_eeg_channels"] = np.asarray(
            payload["selected_eeg_channels"], dtype=np.str_
        )
    cache_manifest = {
        "schema": "token_physiology_measurement_cache_v2",
        "measurement_cache_key": key,
        "checkpoint_independent": True,
        "split": split,
        "modality": modality,
        "source_sample_order_sha256": manifest["sample_order_sha256"],
        "sample_count": int(len(sample_ids)),
        "sample_rate_hz": float(sample_rate_hz),
        "feature_spec_hash": spec.spec_hash,
        "feature_extraction": extracted.manifest.to_dict(),
        "summary_feature_names": list(names),
        "summary_feature_units": list(units),
        "raw_patches_stored": False,
        "notes": (
            [
                "EEG channel summaries pool sample-specific local channels; "
                "selected channel identities remain in the cache."
            ]
            if modality == "eeg"
            else [
                "fNIRS features preserve ordered HbO/HbR roles and contain no "
                "short-window band power."
            ]
        ),
    }
    # Content-derived paths make cache replacement unnecessary and unsafe.
    _write_npz_atomic(cache_path, cache_payload, force=False)
    cache_manifest["npz_sha256"] = _sha256(cache_path)
    _write_json_atomic(sidecar, cache_manifest, force=False)
    return summary, names, units, {
        "status": "miss",
        "path": str(cache_path),
        "measurement_cache_key": key,
        "manifest": cache_manifest,
    }


def prepare_measurement_feature_caches(
    export_path: str | Path,
    *,
    config: Mapping[str, Any],
    measurement_cache_dir: str | Path,
    expected_split: str | None = None,
    allow_test: bool = False,
) -> dict[str, dict[str, Any]]:
    """Extract checkpoint-independent features before compacting assignments."""

    payload, manifest = load_token_export(
        export_path,
        expected_split=expected_split,
        allow_test=allow_test,
    )
    split = str(manifest["split"])
    cache_dir = Path(measurement_cache_dir).resolve()
    spec = _feature_spec(config)
    input_cfg = dict(config.get("input", {}))
    rates = dict(input_cfg.get("sample_rate_hz", {}))
    references: dict[str, dict[str, Any]] = {}
    for modality in MODALITIES:
        patch_key = f"{modality}_patches"
        sample_rate_hz = float(
            rates.get(modality, 200.0 if modality == "eeg" else 10.0)
        )
        expected_duration = input_cfg.get("expected_patch_duration_s")
        if patch_key in payload:
            patches = np.asarray(payload[patch_key])
            observed_duration = float(patches.shape[-1] / sample_rate_hz)
            if expected_duration is not None and not np.isclose(
                observed_duration,
                float(expected_duration),
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    f"{modality} patch duration {observed_duration:g}s does not "
                    f"match configured {float(expected_duration):g}s"
                )
        _, _, _, cache = _load_or_extract_measurements(
            payload,
            manifest,
            split=split,
            modality=modality,
            sample_rate_hz=sample_rate_hz,
            spec=spec,
            cache_dir=cache_dir,
        )
        cache_manifest = cache["manifest"]
        cached_duration = float(
            cache_manifest["feature_extraction"]["patch_duration_seconds"]
        )
        if expected_duration is not None and not np.isclose(
            cached_duration,
            float(expected_duration),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                f"{modality} cached patch duration {cached_duration:g}s does "
                f"not match configured {float(expected_duration):g}s"
            )
        references[modality] = {
            "path": str(Path(str(cache["path"])).resolve()),
            "measurement_cache_key": cache["measurement_cache_key"],
            "npz_sha256": cache_manifest["npz_sha256"],
            "feature_spec_hash": cache_manifest["feature_spec_hash"],
            "sample_rate_hz": cache_manifest["sample_rate_hz"],
            "source_sample_order_sha256": cache_manifest[
                "source_sample_order_sha256"
            ],
        }
    return references


def _repeat_sample(values: np.ndarray, token_count: int) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    return np.repeat(array[:, None], token_count, axis=1).reshape(-1)


def _metadata(
    payload: Mapping[str, np.ndarray],
    token_count: int,
    subjects: np.ndarray,
) -> dict[str, np.ndarray]:
    fields = (
        "dataset_id",
        "task_namespace",
        "source_name",
        "source_task",
        "anchor",
        "label",
        "label_name",
        "record_id",
        "dependency_group_id",
        "has_auxiliary_target",
        "crop_start_s",
    )
    result = {
        key: _repeat_sample(np.asarray(payload[key]), token_count)
        for key in fields
        if key in payload
    }
    result["subject_identity"] = _repeat_sample(subjects, token_count)
    if "selected_eeg_channels" in payload:
        selected = np.asarray(payload["selected_eeg_channels"], dtype=np.str_)
        if selected.ndim == 2 and len(selected) == len(subjects):
            result["eeg_channel_set"] = _repeat_sample(
                np.asarray(["|".join(row.tolist()) for row in selected], dtype=np.str_),
                token_count,
            )
    result["token_position"] = np.tile(np.arange(token_count), len(subjects))
    return result


def _diagnostic_rows(
    payload: Mapping[str, np.ndarray],
    *,
    split: str,
    modality: str,
    hard_ids: np.ndarray,
    subjects: np.ndarray,
    valid_mask: np.ndarray,
    codebook_size: int,
) -> list[dict[str, Any]]:
    metrics: dict[str, np.ndarray] = {}
    prefix = f"{modality}_"
    for key, raw in payload.items():
        if not key.startswith(prefix):
            continue
        name = key[len(prefix) :]
        if not (
            name in {
                "posterior_entropy",
                "posterior_top1_top2_margin",
                "latent_code_l2",
            }
            or name.endswith("_reconstruction_mse")
        ):
            continue
        values = np.asarray(raw, dtype=np.float64)
        if values.shape == hard_ids.shape:
            metrics[name] = values
    rows: list[dict[str, Any]] = []
    for token_id in range(codebook_size):
        selected = valid_mask & (hard_ids == token_id)
        subject_means_by_metric: dict[str, list[float]] = {
            name: [] for name in metrics
        }
        for subject in np.unique(subjects):
            subject_selected = selected & (subjects[:, None] == subject)
            for name, values in metrics.items():
                finite_values = values[subject_selected]
                finite_values = finite_values[np.isfinite(finite_values)]
                if len(finite_values):
                    subject_means_by_metric[name].append(
                        float(np.mean(finite_values))
                    )
        for name, subject_means in subject_means_by_metric.items():
            array = np.asarray(subject_means, dtype=np.float64)
            rows.append(
                {
                    "split": split,
                    "modality": modality,
                    "token_id": token_id,
                    "metric": name,
                    "subject_count": int(len(array)),
                    "subject_equal_mean": (
                        float(np.mean(array)) if len(array) else None
                    ),
                    "subject_equal_median": (
                        float(np.median(array)) if len(array) else None
                    ),
                    "statistical_unit": "subject",
                    "measurement_kind": "model_diagnostic",
                }
            )
    return rows


def _exemplars(
    *,
    split: str,
    modality: str,
    sample_ids: np.ndarray,
    hard_ids: np.ndarray,
    posterior: np.ndarray | None,
    valid_mask: np.ndarray,
    features: np.ndarray,
    support_rows: Sequence[Mapping[str, Any]],
    per_token: int = 3,
) -> list[dict[str, Any]]:
    flat_ids = hard_ids.reshape(-1)
    flat_valid = valid_mask.reshape(-1)
    flat_features = features.reshape(-1, features.shape[-1]).astype(np.float64)
    sample_index = np.repeat(np.arange(len(sample_ids)), hard_ids.shape[1])
    position = np.tile(np.arange(hard_ids.shape[1]), len(sample_ids))
    flat_posterior = (
        posterior.reshape(-1, posterior.shape[-1]) if posterior is not None else None
    )
    rows: list[dict[str, Any]] = []
    for support in support_rows:
        token_id = int(support["token_id"])
        if bool(support["insufficient_support"]):
            continue
        indices = np.flatnonzero(flat_valid & (flat_ids == token_id))
        if not len(indices):
            continue
        selected_features = flat_features[indices]
        center = np.nanmedian(selected_features, axis=0)
        scale = np.nanstd(flat_features[flat_valid], axis=0)
        usable = (
            np.isfinite(selected_features)
            & np.isfinite(center)[None, :]
            & (scale[None, :] > 1e-8)
        )
        distance = np.full(len(indices), np.inf)
        for local_index in range(len(indices)):
            feature_mask = usable[local_index]
            if np.any(feature_mask):
                distance[local_index] = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                (
                                    selected_features[local_index, feature_mask]
                                    - center[feature_mask]
                                )
                                / scale[feature_mask]
                            )
                        )
                    )
                )
        choices: list[tuple[str, int]] = [
            ("nearest_phenotype_center", int(indices[np.argmin(distance)]))
        ]
        if flat_posterior is not None:
            confidence = flat_posterior[indices, token_id]
            choices.extend(
                (
                    ("highest_assignment_confidence", int(indices[np.argmax(confidence)])),
                    ("lowest_assignment_confidence", int(indices[np.argmin(confidence)])),
                )
            )
        seen: set[int] = set()
        for role, flat_index in choices:
            if flat_index in seen or len(seen) >= per_token:
                continue
            seen.add(flat_index)
            confidence_value = (
                float(flat_posterior[flat_index, token_id])
                if flat_posterior is not None
                else None
            )
            rows.append(
                {
                    "split": split,
                    "modality": modality,
                    "token_id": token_id,
                    "role": role,
                    "sample_id": str(sample_ids[sample_index[flat_index]]),
                    "patch_position": int(position[flat_index]),
                    "assignment_probability": confidence_value,
                    "raw_patch_copied": False,
                }
            )
    return rows


def _feature_distribution_rows(
    result: ModalitySplitAtlas,
) -> list[dict[str, Any]]:
    """Descriptive hard-token patch distributions, separate from inference."""

    flat_ids = result.hard_ids.reshape(-1)
    flat_mask = result.valid_mask.reshape(-1)
    flat_features = result.feature_values.reshape(
        -1, result.feature_values.shape[-1]
    )
    support = {
        int(row["token_id"]): row for row in result.token_result.support_rows
    }
    rows: list[dict[str, Any]] = []
    for token_id in range(result.codebook_size):
        selected = flat_mask & (flat_ids == token_id)
        for feature_index, (feature_name, feature_unit) in enumerate(
            zip(result.feature_names, result.feature_units, strict=True)
        ):
            values = np.asarray(
                flat_features[selected, feature_index], dtype=np.float64
            )
            values = values[np.isfinite(values)]
            if len(values):
                q05, q25, q50, q75, q95 = np.quantile(
                    values, (0.05, 0.25, 0.50, 0.75, 0.95)
                )
                mean = float(np.mean(values))
                standard_deviation = float(np.std(values))
            else:
                q05 = q25 = q50 = q75 = q95 = np.nan
                mean = standard_deviation = np.nan
            rows.append(
                {
                    "split": result.split,
                    "modality": result.modality,
                    "token_id": token_id,
                    "feature_index": feature_index,
                    "feature_name": feature_name,
                    "feature_unit": feature_unit,
                    "finite_patch_count": int(len(values)),
                    "patch_mean": _json_value(mean),
                    "patch_standard_deviation": _json_value(standard_deviation),
                    "patch_q05": _json_value(q05),
                    "patch_q25": _json_value(q25),
                    "patch_median": _json_value(q50),
                    "patch_q75": _json_value(q75),
                    "patch_q95": _json_value(q95),
                    "insufficient_support": bool(
                        support[token_id]["insufficient_support"]
                    ),
                    "distribution_unit": "token-aligned patch",
                    "inference_unit": (
                        "none_descriptive_distribution; use token_profiles for "
                        "subject-equal estimates"
                    ),
                }
            )
    return rows


def _hard_soft_difference_rows(
    result: ModalitySplitAtlas,
) -> list[dict[str, Any]]:
    profiles = {
        (
            str(row["profile_type"]),
            int(row["token_id"]),
            str(row["feature_name"]),
        ): row
        for row in result.token_result.profile_rows
    }
    rows: list[dict[str, Any]] = []
    for token_id in range(result.codebook_size):
        for feature_name in result.feature_names:
            hard = profiles.get(("hard", token_id, feature_name))
            soft = profiles.get(("soft", token_id, feature_name))
            if hard is None or soft is None:
                continue
            hard_value = hard.get("marginal_standardized_effect")
            soft_value = soft.get("marginal_standardized_effect")
            delta = (
                float(soft_value) - float(hard_value)
                if hard_value is not None and soft_value is not None
                else None
            )
            rows.append(
                {
                    "split": result.split,
                    "modality": result.modality,
                    "token_id": token_id,
                    "feature_name": feature_name,
                    "hard_standardized_effect": hard_value,
                    "soft_standardized_effect": soft_value,
                    "soft_minus_hard_standardized_effect": delta,
                    "absolute_difference": (
                        abs(delta) if delta is not None else None
                    ),
                    "effect_unit": "marginal subject-equal scale",
                    "insufficient_support": bool(
                        hard.get("insufficient_support", True)
                    ),
                    "interpretation": (
                        "assignment-boundary sensitivity; not physiological "
                        "state disagreement"
                    ),
                }
            )
    return rows


def _channel_feature_distribution_rows(
    result: ModalitySplitAtlas,
) -> list[dict[str, Any]]:
    """Describe token features using true channel identities from the cache."""

    cache_path = Path(str(result.measurement_cache["path"]))
    with np.load(cache_path, allow_pickle=False) as archive:
        values = np.asarray(archive["channel_feature_values"], dtype=np.float64)
        if (
            result.modality == "eeg"
            and "selected_eeg_channels" in archive.files
        ):
            channel_identity = np.asarray(
                archive["selected_eeg_channels"], dtype=np.str_
            )
        else:
            channel_names = tuple(
                result.measurement_cache["manifest"]["feature_extraction"][
                    "channel_names"
                ]
            )
            channel_identity = np.tile(
                np.asarray(channel_names, dtype=np.str_),
                (values.shape[0], 1),
            )
    definitions = result.measurement_cache["manifest"]["feature_extraction"][
        "feature_definitions"
    ]
    feature_names = tuple(str(row["name"]) for row in definitions)
    feature_units = tuple(str(row["unit"]) for row in definitions)
    if (
        values.shape[:3]
        != (
            result.hard_ids.shape[0],
            result.hard_ids.shape[1],
            channel_identity.shape[1],
        )
        or values.shape[-1] != len(feature_names)
    ):
        raise ValueError("Channel feature cache does not align with token assignments")

    batch_size, token_count, channel_count, _ = values.shape
    flat_values = values.reshape(-1, values.shape[-1])
    flat_tokens = np.broadcast_to(
        result.hard_ids[:, :, None],
        (batch_size, token_count, channel_count),
    ).reshape(-1)
    flat_valid = np.broadcast_to(
        result.valid_mask[:, :, None],
        (batch_size, token_count, channel_count),
    ).reshape(-1)
    flat_channels = np.broadcast_to(
        channel_identity[:, None, :],
        (batch_size, token_count, channel_count),
    ).reshape(-1)
    flat_subjects = np.broadcast_to(
        result.subjects_by_sample[:, None, None],
        (batch_size, token_count, channel_count),
    ).reshape(-1)
    grouped: dict[tuple[int, str], list[int]] = {}
    for index in np.flatnonzero(flat_valid):
        key = (int(flat_tokens[index]), str(flat_channels[index]))
        grouped.setdefault(key, []).append(int(index))

    thresholds = result.token_result.manifest["support_thresholds"]
    token_support = {
        int(row["token_id"]): bool(row["insufficient_support"])
        for row in result.token_result.support_rows
    }
    rows: list[dict[str, Any]] = []
    for (token_id, channel_name), raw_indices in sorted(grouped.items()):
        indices = np.asarray(raw_indices, dtype=np.int64)
        for feature_index, (feature_name, feature_unit) in enumerate(
            zip(feature_names, feature_units, strict=True)
        ):
            feature_values = flat_values[indices, feature_index]
            finite = np.isfinite(feature_values)
            finite_indices = indices[finite]
            feature_values = feature_values[finite]
            subject_means: list[float] = []
            if len(feature_values):
                for subject in np.unique(flat_subjects[finite_indices]):
                    selected = flat_subjects[finite_indices] == subject
                    subject_means.append(float(np.mean(feature_values[selected])))
                q25, q50, q75 = np.quantile(feature_values, (0.25, 0.5, 0.75))
            else:
                q25 = q50 = q75 = np.nan
            rows.append(
                {
                    "split": result.split,
                    "modality": result.modality,
                    "token_id": token_id,
                    "channel_name": channel_name,
                    "feature_index": feature_index,
                    "feature_name": feature_name,
                    "feature_unit": feature_unit,
                    "finite_patch_count": int(len(feature_values)),
                    "subject_count": len(subject_means),
                    "token_insufficient_support": token_support[token_id],
                    "token_channel_feature_insufficient_support": (
                        len(feature_values) < int(thresholds["min_count"])
                        or len(subject_means) < int(thresholds["min_subjects"])
                    ),
                    "subject_equal_mean": (
                        float(np.mean(subject_means)) if subject_means else None
                    ),
                    "patch_q25": _json_value(q25),
                    "patch_median": _json_value(q50),
                    "patch_q75": _json_value(q75),
                    "channel_identity_policy": (
                        "per-sample selected EEG channel name"
                        if result.modality == "eeg"
                        else "fixed HbO/HbR role"
                    ),
                    "inference_unit": (
                        "descriptive; no channel-level bootstrap in this table"
                    ),
                }
            )
    return rows


def analyze_export_split(
    payload: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    *,
    split: str,
    config: Mapping[str, Any],
    measurement_cache_dir: Path,
    bootstrap_iterations: int | None = None,
) -> dict[str, ModalitySplitAtlas]:
    """Analyze both modalities from an already validated export."""

    profiles_cfg = dict(config.get("profiles", {}))
    associations_cfg = dict(config.get("associations", {}))
    input_cfg = dict(config.get("input", {}))
    rates = dict(input_cfg.get("sample_rate_hz", {}))
    spec = _feature_spec(config)
    sample_ids = _sample_ids(payload, manifest)
    subjects = _subjects(payload, len(sample_ids))
    results: dict[str, ModalitySplitAtlas] = {}
    for modality in MODALITIES:
        hard_ids = np.asarray(payload[f"{modality}_hard_ids"], dtype=np.int64)
        patches = (
            np.asarray(payload[f"{modality}_patches"])
            if f"{modality}_patches" in payload
            else None
        )
        valid_mask = _token_mask(payload, modality, hard_ids.shape)
        posterior = _full_posterior(payload, modality, hard_ids.shape)
        codebook_size = _codebook_size(
            hard_ids,
            posterior,
            payload.get(f"{modality}_codebook"),
        )
        sample_rate_hz = float(
            rates.get(modality, 200.0 if modality == "eeg" else 10.0)
        )
        expected_duration = input_cfg.get("expected_patch_duration_s")
        if patches is not None:
            observed_duration = float(patches.shape[-1] / sample_rate_hz)
            if expected_duration is not None and not np.isclose(
                observed_duration,
                float(expected_duration),
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    f"{modality} patch duration {observed_duration:g}s does not "
                    f"match configured {float(expected_duration):g}s"
                )
        features, names, units, cache = _load_or_extract_measurements(
            payload,
            manifest,
            split=split,
            modality=modality,
            sample_rate_hz=sample_rate_hz,
            spec=spec,
            cache_dir=measurement_cache_dir,
        )
        cached_duration = float(
            cache["manifest"]["feature_extraction"]["patch_duration_seconds"]
        )
        if expected_duration is not None and not np.isclose(
            cached_duration,
            float(expected_duration),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                f"{modality} cached patch duration {cached_duration:g}s does "
                f"not match configured {float(expected_duration):g}s"
            )
        token_count = hard_ids.shape[1]
        metadata = _metadata(payload, token_count, subjects)
        state_fields = tuple(associations_cfg.get("state_fields", ()))
        for field in state_fields:
            if field in metadata or field not in payload:
                continue
            raw_state = np.asarray(payload[field])
            if raw_state.ndim == 1 and len(raw_state) == len(subjects):
                metadata[field] = _repeat_sample(raw_state, token_count)
            elif raw_state.shape == hard_ids.shape:
                metadata[field] = raw_state.reshape(-1)
        states = {
            field: metadata.pop(field)
            for field in state_fields
            if field in metadata
        }
        physiology_config = TokenPhysiologyConfig(
            codebook_size=codebook_size,
            min_count=int(profiles_cfg.get("min_count", 30)),
            min_subjects=int(profiles_cfg.get("min_subjects", 5)),
            rare_count=int(profiles_cfg.get("rare_count", profiles_cfg.get("min_count", 30))),
            bootstrap_iterations=int(
                bootstrap_iterations
                if bootstrap_iterations is not None
                else profiles_cfg.get("bootstrap", {}).get("iterations", 1000)
            ),
            bootstrap_confidence=float(
                profiles_cfg.get("bootstrap", {}).get("confidence", 0.95)
            ),
            seed=int(config.get("analysis", {}).get("seed", 0)),
            max_state_categories=int(
                associations_cfg.get("max_state_categories", 64)
            ),
            max_metadata_categories=int(
                associations_cfg.get("max_metadata_categories", 64)
            ),
        )
        flat_subjects = _repeat_sample(subjects, token_count)
        token_result = analyze_token_physiology(
            features.reshape(-1, features.shape[-1]),
            hard_ids.reshape(-1),
            flat_subjects,
            feature_names=names,
            posterior=(
                posterior.reshape(-1, posterior.shape[-1])
                if posterior is not None
                else None
            ),
            states=states or None,
            metadata=metadata,
            valid_mask=valid_mask.reshape(-1),
            config=physiology_config,
        )
        results[modality] = ModalitySplitAtlas(
            split=split,
            modality=modality,
            sample_ids=sample_ids,
            subjects_by_sample=subjects,
            hard_ids=hard_ids,
            posterior=posterior,
            valid_mask=valid_mask,
            feature_values=features,
            feature_names=names,
            feature_units=units,
            token_result=token_result,
            diagnostic_rows=_diagnostic_rows(
                payload,
                split=split,
                modality=modality,
                hard_ids=hard_ids,
                subjects=subjects,
                valid_mask=valid_mask,
                codebook_size=codebook_size,
            ),
            exemplar_rows=_exemplars(
                split=split,
                modality=modality,
                sample_ids=sample_ids,
                hard_ids=hard_ids,
                posterior=posterior,
                valid_mask=valid_mask,
                features=features,
                support_rows=token_result.support_rows,
            ),
            measurement_cache=cache,
            codebook_size=codebook_size,
        )
    return results


def _tag_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str,
    modality: str,
) -> list[dict[str, Any]]:
    return [
        {"split": split, "modality": modality, **dict(row)}
        for row in rows
    ]


def _sequence_artifacts(
    split_results: Mapping[str, Mapping[str, ModalitySplitAtlas]],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    sequence_cfg = dict(config.get("sequence", {}))
    requested_lags = tuple(int(value) for value in sequence_cfg.get("lags", (-2, -1, 0, 1, 2)))
    permutations = int(sequence_cfg.get("null", {}).get("permutations", 0))
    patch_duration = float(sequence_cfg.get("patch_duration_s", 2.0))
    seed = int(config.get("analysis", {}).get("seed", 0))
    summary: dict[str, Any] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    lag_rows: list[dict[str, Any]] = []
    for split, modalities in split_results.items():
        summary[split] = {}
        arrays[split] = {}
        for modality, result in modalities.items():
            modality_summary = summarize_sequences(
                result.hard_ids,
                result.valid_mask,
                codebook_size=result.codebook_size,
            )
            serialized_summary = _json_value(asdict(modality_summary))
            for row in serialized_summary["token_rows"]:
                for statistic in ("mean", "median", "max"):
                    token_value = row.get(f"run_length_{statistic}_tokens")
                    row[f"dwell_{statistic}_seconds"] = (
                        float(token_value) * patch_duration
                        if token_value is not None
                        else None
                    )
            serialized_summary["patch_duration_seconds"] = patch_duration
            serialized_summary["dwell_definition"] = (
                "contiguous within-window run length multiplied by patch duration"
            )
            summary[split][modality] = serialized_summary
            if result.hard_ids.shape[1] >= 2:
                arrays[split][f"{modality}_transition_counts"] = transition_counts(
                    result.hard_ids,
                    result.valid_mask,
                    codebook_size=result.codebook_size,
                )
            else:
                arrays[split][f"{modality}_transition_counts"] = np.zeros(
                    (result.codebook_size, result.codebook_size), dtype=np.int64
                )
        eeg = modalities["eeg"]
        fnirs = modalities["fnirs"]
        valid_lags = tuple(
            lag for lag in requested_lags if abs(lag) < eeg.hard_ids.shape[1]
        )
        if eeg.hard_ids.shape == fnirs.hard_ids.shape and valid_lags:
            rows, matrices, null = analyze_cross_modal_lags(
                eeg.hard_ids,
                fnirs.hard_ids,
                eeg_valid_mask=eeg.valid_mask,
                fnirs_valid_mask=fnirs.valid_mask,
                lags=valid_lags,
                eeg_codebook_size=eeg.codebook_size,
                fnirs_codebook_size=fnirs.codebook_size,
                permutations=permutations,
                seed=seed,
                patch_duration_s=patch_duration,
            )
            arrays[split]["cross_modal_lag_counts"] = matrices
            arrays[split]["cross_modal_null_nmi"] = null
            arrays[split]["cross_modal_lags"] = np.asarray(valid_lags, dtype=np.int64)
            lag_rows.extend(
                {"split": split, **dict(row)}
                for row in rows
            )
            summary[split]["cross_modal_status"] = "ok"
        else:
            summary[split]["cross_modal_status"] = "skipped"
            summary[split]["cross_modal_skipped_reason"] = (
                "token_grid_mismatch"
                if eeg.hard_ids.shape != fnirs.hard_ids.shape
                else "no_valid_requested_lags"
            )
    if "train" in split_results and "val" in split_results:
        summary["train_to_val_markov"] = {}
        for modality in MODALITIES:
            train = split_results["train"][modality]
            validation = split_results["val"][modality]
            if train.codebook_size != validation.codebook_size:
                summary["train_to_val_markov"][modality] = {
                    "status": "skipped",
                    "skipped_reason": "codebook_size_mismatch",
                }
            else:
                summary["train_to_val_markov"][modality] = {
                    "status": "ok",
                    **_json_value(
                        markov_log_loss(
                            train.hard_ids,
                            validation.hard_ids,
                            train_valid_mask=train.valid_mask,
                            validation_valid_mask=validation.valid_mask,
                            codebook_size=train.codebook_size,
                        )
                    ),
                }
    return summary, arrays, lag_rows


def _information_ledgers(
    split_results: Mapping[str, Mapping[str, ModalitySplitAtlas]],
    payloads: Mapping[str, Mapping[str, np.ndarray]],
    *,
    config: Mapping[str, Any],
    bootstrap_iterations: int | None,
) -> dict[str, Any]:
    if "train" not in split_results or "val" not in split_results:
        return {
            "schema_version": "token_information_ledger_collection_v1",
            "status": "skipped",
            "skipped_reason": "train_and_val_exports_required",
        }
    iterations = int(
        bootstrap_iterations
        if bootstrap_iterations is not None
        else config.get("profiles", {}).get("bootstrap", {}).get("iterations", 1000)
    )
    ledger_config = InformationLedgerConfig(
        bootstrap_iterations=iterations,
        seed=int(config.get("analysis", {}).get("seed", 0)),
    )
    ledgers: dict[str, Any] = {}

    def add_model_representations(
        representations: dict[str, np.ndarray],
        payload: Mapping[str, np.ndarray],
        modality: str,
        valid_mask: np.ndarray,
    ) -> None:
        latent = np.asarray(payload[f"{modality}_semantic_latent"], dtype=np.float64)
        candidates: list[tuple[str, np.ndarray]] = []
        expected_key = f"{modality}_expected_embedding"
        if expected_key in payload:
            candidates.append(
                (
                    "expected_embedding",
                    np.asarray(payload[expected_key], dtype=np.float64),
                )
            )
        residual_key = f"{modality}_residual"
        if residual_key in payload:
            residual = np.asarray(payload[residual_key], dtype=np.float64)
            candidates.extend(
                (
                    ("residual", residual),
                    (
                        "continuous_semantic_plus_residual",
                        np.concatenate((latent, residual), axis=-1),
                    ),
                )
            )
        for name, values in candidates:
            matrix = values.reshape(-1, values.shape[-1]).copy()
            matrix[~valid_mask.reshape(-1)] = np.nan
            representations[name] = matrix

    for modality in MODALITIES:
        train = split_results["train"][modality]
        validation = split_results["val"][modality]
        if train.feature_names != validation.feature_names:
            ledgers[modality] = {
                "status": "skipped",
                "skipped_reason": "train_validation_feature_contract_mismatch",
            }
            continue
        train_payload = payloads["train"]
        validation_payload = payloads["val"]
        train_representations = build_token_representations(
            continuous_latent=train_payload[f"{modality}_semantic_latent"],
            hard_ids=train.hard_ids,
            posterior=train.posterior,
            codebook_embedding=train_payload[f"{modality}_codebook_embedding"],
            codebook_size=train.codebook_size,
            valid_mask=train.valid_mask,
        )
        add_model_representations(
            train_representations,
            train_payload,
            modality,
            train.valid_mask,
        )
        validation_representations = build_token_representations(
            continuous_latent=validation_payload[f"{modality}_semantic_latent"],
            hard_ids=validation.hard_ids,
            posterior=validation.posterior,
            codebook_embedding=validation_payload[f"{modality}_codebook_embedding"],
            codebook_size=validation.codebook_size,
            valid_mask=validation.valid_mask,
        )
        add_model_representations(
            validation_representations,
            validation_payload,
            modality,
            validation.valid_mask,
        )
        ledgers[modality] = evaluate_information_ledger(
            train.feature_values.reshape(-1, train.feature_values.shape[-1]),
            validation.feature_values.reshape(-1, validation.feature_values.shape[-1]),
            _repeat_sample(train.subjects_by_sample, train.hard_ids.shape[1]),
            _repeat_sample(
                validation.subjects_by_sample, validation.hard_ids.shape[1]
            ),
            train_representations,
            validation_representations,
            coordinate_names=train.feature_names,
            config=ledger_config,
        )
    return {
        "schema_version": "token_information_ledger_collection_v1",
        "status": (
            "ok"
            if all(value.get("status") == "ok" for value in ledgers.values())
            else "partial_or_skipped"
        ),
        "modalities": ledgers,
    }


def _profile_matrix(
    result: TokenPhysiologyResult,
    token_ids: Sequence[int],
) -> np.ndarray:
    names = list(result.manifest["feature_names"])
    matrix = np.full((len(token_ids), len(names)), np.nan)
    token_index = {token_id: index for index, token_id in enumerate(token_ids)}
    feature_index = {name: index for index, name in enumerate(names)}
    for row in result.profile_rows:
        if row["profile_type"] != "hard":
            continue
        token_id = int(row["token_id"])
        if token_id not in token_index:
            continue
        value = row["marginal_standardized_effect"]
        if value is not None:
            matrix[
                token_index[token_id], feature_index[str(row["feature_name"])]
            ] = float(value)
    return matrix


def _embedding_by_token(
    payload: Mapping[str, np.ndarray],
    result: ModalitySplitAtlas,
) -> np.ndarray:
    codebook_key = f"{result.modality}_codebook"
    if codebook_key in payload:
        codebook = np.asarray(payload[codebook_key], dtype=np.float64)
        if (
            codebook.ndim == 2
            and codebook.shape[0] == result.codebook_size
            and codebook.shape[1] >= 2
        ):
            return codebook
    gathered = np.asarray(
        payload[f"{result.modality}_codebook_embedding"], dtype=np.float64
    )
    embedding = np.full((result.codebook_size, gathered.shape[-1]), np.nan)
    for token_id in range(result.codebook_size):
        selected = result.valid_mask & (result.hard_ids == token_id)
        if np.any(selected):
            embedding[token_id] = np.mean(gathered[selected], axis=0)
    return embedding


def _prepare_figure_targets(
    stem: Path,
    formats: Sequence[str],
    *,
    force: bool,
) -> None:
    targets = [Path(f"{stem}.{fmt}") for fmt in formats]
    existing = [path for path in targets if path.exists() or path.is_symlink()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing figure artifact(s): "
            + ", ".join(str(path) for path in existing)
        )
    if force:
        for path in existing:
            if path.is_dir():
                raise IsADirectoryError(path)
            path.unlink()


def _write_figures(
    split_results: Mapping[str, Mapping[str, ModalitySplitAtlas]],
    payloads: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: Path,
    *,
    config: Mapping[str, Any],
    formats_override: Sequence[str] | None,
    force: bool,
) -> list[Path]:
    # Keep the tabular analysis API usable without importing Matplotlib.
    import matplotlib.pyplot as plt

    from src.visualization.token_physiology_plots import (
        plot_codebook_embedding_colored,
        plot_token_feature_heatmap,
        plot_token_support,
        save_figure_atomic,
    )

    figures_cfg = dict(config.get("figures", {}))
    formats = tuple(
        str(value).lower().lstrip(".")
        for value in (
            formats_override
            if formats_override is not None
            else figures_cfg.get("formats", ("png",))
        )
    )
    dpi = int(figures_cfg.get("dpi", 180))
    top_tokens = int(figures_cfg.get("top_tokens", 24))
    figure_dir = output_dir / "figures"
    produced: list[Path] = []
    for split, modalities in split_results.items():
        for modality, result in modalities.items():
            support_rows = result.token_result.support_rows
            token_ids = np.asarray(
                [int(row["token_id"]) for row in support_rows], dtype=np.int64
            )
            counts = np.asarray([int(row["count"]) for row in support_rows])
            supported = np.asarray(
                [not bool(row["insufficient_support"]) for row in support_rows]
            )
            support_stem = figure_dir / f"{split}_{modality}_token_support"
            _prepare_figure_targets(support_stem, formats, force=force)
            figure, _ = plot_token_support(
                token_ids,
                counts,
                supported,
                minimum_support=float(
                    result.token_result.manifest["support_thresholds"]["min_count"]
                ),
                title=f"{split} {modality.upper()} token support",
            )
            artifacts = save_figure_atomic(
                figure,
                support_stem,
                formats=formats,
                dpi=dpi,
            )
            plt.close(figure)
            produced.extend(artifacts.figure_paths)

            ranked = sorted(
                range(len(token_ids)),
                key=lambda index: (supported[index], counts[index], -token_ids[index]),
                reverse=True,
            )[: max(top_tokens, 1)]
            selected_ids = token_ids[ranked]
            selected_support = supported[ranked]
            matrix = _profile_matrix(result.token_result, selected_ids.tolist())
            heatmap_stem = figure_dir / f"{split}_{modality}_phenotype_heatmap"
            _prepare_figure_targets(heatmap_stem, formats, force=force)
            figure, _ = plot_token_feature_heatmap(
                matrix,
                selected_ids,
                result.feature_names,
                selected_support,
                title=(
                    f"{split} {modality.upper()} hard-token physiological profiles "
                    "(subject-equal)"
                ),
            )
            artifacts = save_figure_atomic(
                figure,
                heatmap_stem,
                formats=formats,
                dpi=dpi,
            )
            plt.close(figure)
            produced.extend(artifacts.figure_paths)

            feature_candidates = (
                ("channel_mean/log_relative_power_alpha",)
                if modality == "eeg"
                else ("HbO/slope",)
            )
            feature_name = next(
                (name for name in feature_candidates if name in result.feature_names),
                result.feature_names[0],
            )
            feature_column = result.feature_names.index(feature_name)
            full_matrix = _profile_matrix(result.token_result, token_ids.tolist())
            colors = full_matrix[:, feature_column]
            embedding = _embedding_by_token(payloads[split], result)
            if embedding.shape[1] >= 2 and np.sum(np.all(np.isfinite(embedding), axis=1)) >= 2:
                embedding_stem = (
                    figure_dir / f"{split}_{modality}_codebook_{feature_name.replace('/', '_')}"
                )
                _prepare_figure_targets(embedding_stem, formats, force=force)
                figure, _ = plot_codebook_embedding_colored(
                    embedding,
                    colors,
                    token_ids,
                    supported,
                    feature_name=feature_name,
                    units="standardized enrichment",
                    center=0.0,
                    title=(
                        f"{split} {modality.upper()} codebook geometry and "
                        f"{feature_name}"
                    ),
                )
                artifacts = save_figure_atomic(
                    figure,
                    embedding_stem,
                    formats=formats,
                    dpi=dpi,
                )
                plt.close(figure)
                produced.extend(artifacts.figure_paths)
    return produced


def build_token_physiology_atlas(
    exports: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    config: Mapping[str, Any],
    measurement_cache_dir: str | Path,
    allow_test: bool = False,
    force: bool = False,
    bootstrap_iterations: int | None = None,
    coupling_permutations: int | None = None,
    formats: Sequence[str] | None = None,
    plots: bool = True,
    information_ledger: bool = True,
) -> Path:
    """Build all standard Atlas artifacts and return the output directory."""

    started = time.perf_counter()
    output = Path(output_dir).resolve()
    cache_dir = Path(measurement_cache_dir).resolve()
    if config.get("input", {}).get("unit") != "canonical_robust_sd":
        raise ValueError(
            "Atlas requires input.unit=canonical_robust_sd; the tokenizer export "
            "does not restore physical voltage or concentration units"
        )
    if not exports:
        raise ValueError("At least one split export is required")
    invalid_splits = sorted(set(exports) - {"train", "val", "test"})
    if invalid_splits:
        raise ValueError(f"Unsupported split names: {invalid_splits}")
    if "test" in exports and not allow_test:
        raise ValueError(
            "Protected test split requested without explicit allow_test=True"
        )
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(
            f"Output directory is not empty: {output}. Choose a new directory "
            "or pass force=True to replace exact Atlas artifacts."
        )
    output.mkdir(parents=True, exist_ok=True)
    effective_config = json.loads(json.dumps(config))
    if coupling_permutations is not None:
        effective_config.setdefault("sequence", {}).setdefault("null", {})[
            "permutations"
        ] = int(coupling_permutations)
    resolved_exports = {
        split: Path(path).resolve() for split, path in exports.items()
    }
    payloads: dict[str, dict[str, np.ndarray]] = {}
    split_results: dict[str, dict[str, ModalitySplitAtlas]] = {}
    for split, path in resolved_exports.items():
        payload, export_manifest = load_token_export(
            path,
            expected_split=split,
            allow_test=allow_test,
        )
        payloads[split] = payload
        split_results[split] = analyze_export_split(
            payload,
            export_manifest,
            split=split,
            config=effective_config,
            measurement_cache_dir=cache_dir,
            bootstrap_iterations=bootstrap_iterations,
        )

    support_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    hard_soft_rows: list[dict[str, Any]] = []
    channel_distribution_rows: list[dict[str, Any]] = []
    exemplar_rows: list[dict[str, Any]] = []
    for split, modalities in split_results.items():
        for modality, result in modalities.items():
            support_rows.extend(
                _tag_rows(
                    result.token_result.support_rows,
                    split=split,
                    modality=modality,
                )
            )
            profile_rows.extend(
                _tag_rows(
                    [
                        {
                            **row,
                            "feature_unit": result.feature_units[
                                int(row["feature_index"])
                            ],
                            "marginal_standardized_effect_unit": (
                                "marginal subject-equal scale"
                            ),
                        }
                        for row in result.token_result.profile_rows
                    ],
                    split=split,
                    modality=modality,
                )
            )
            state_rows.extend(
                _tag_rows(
                    result.token_result.state_rows,
                    split=split,
                    modality=modality,
                )
            )
            metadata_rows.extend(
                _tag_rows(
                    result.token_result.metadata_rows,
                    split=split,
                    modality=modality,
                )
            )
            diagnostic_rows.extend(result.diagnostic_rows)
            distribution_rows.extend(_feature_distribution_rows(result))
            hard_soft_rows.extend(_hard_soft_difference_rows(result))
            channel_distribution_rows.extend(
                _channel_feature_distribution_rows(result)
            )
            exemplar_rows.extend(result.exemplar_rows)

    table_dir = output / "tables"
    _write_csv_atomic(table_dir / "token_support.csv", support_rows, force=force)
    _write_csv_atomic(table_dir / "token_profiles.csv", profile_rows, force=force)
    _write_csv_atomic(
        table_dir / "state_associations.csv", state_rows, force=force
    )
    _write_csv_atomic(
        table_dir / "metadata_associations.csv", metadata_rows, force=force
    )
    _write_csv_atomic(
        table_dir / "assignment_diagnostics.csv",
        diagnostic_rows,
        force=force,
    )
    _write_csv_atomic(
        table_dir / "token_feature_distributions.csv",
        distribution_rows,
        force=force,
    )
    _write_csv_atomic(
        table_dir / "hard_soft_profile_differences.csv",
        hard_soft_rows,
        force=force,
    )
    _write_csv_atomic(
        table_dir / "token_channel_feature_distributions.csv",
        channel_distribution_rows,
        force=force,
    )
    _write_text_atomic(
        table_dir / "token_exemplars.jsonl",
        "".join(
            json.dumps(_json_value(row), ensure_ascii=False, allow_nan=False)
            + "\n"
            for row in exemplar_rows
        ),
        force=force,
    )
    _write_json_atomic(
        output / "token_analysis_manifests.json",
        {
            "schema": "token_physiology_analysis_manifest_collection_v1",
            "splits": {
                split: {
                    modality: {
                        **result.token_result.manifest,
                        "feature_units": list(result.feature_units),
                    }
                    for modality, result in modalities.items()
                }
                for split, modalities in split_results.items()
            },
        },
        force=force,
    )

    sequence_summary, sequence_arrays, lag_rows = _sequence_artifacts(
        split_results, effective_config
    )
    _write_json_atomic(
        output / "sequence_summary.json", sequence_summary, force=force
    )
    _write_csv_atomic(
        table_dir / "cross_modal_lags.csv", lag_rows, force=force
    )
    array_dir = output / "arrays"
    for split, arrays in sequence_arrays.items():
        _write_npz_atomic(
            array_dir / f"{split}_sequence_counts.npz", arrays, force=force
        )

    stability: dict[str, Any]
    if "train" in split_results and "val" in split_results:
        stability = {
            "schema": "token_physiology_atlas_stability_v1",
            "comparison": "train_to_val",
            "modalities": {
                modality: match_token_signatures(
                    split_results["train"][modality].token_result,
                    split_results["val"][modality].token_result,
                    profile_type=str(
                        effective_config.get("stability", {}).get(
                            "profile_type", "hard"
                        )
                    ),
                    min_feature_overlap=int(
                        effective_config.get("stability", {}).get(
                            "min_feature_overlap", 2
                        )
                    ),
                    bootstrap_iterations=int(
                        bootstrap_iterations
                        if bootstrap_iterations is not None
                        else effective_config.get("profiles", {})
                        .get("bootstrap", {})
                        .get("iterations", 1000)
                    ),
                    seed=int(effective_config.get("analysis", {}).get("seed", 0)),
                )
                for modality in MODALITIES
            },
        }
    else:
        stability = {
            "schema": "token_physiology_atlas_stability_v1",
            "status": "skipped",
            "skipped_reason": "train_and_val_exports_required",
        }
    _write_json_atomic(output / "stability.json", stability, force=force)

    if information_ledger:
        information_ledger_result = _information_ledgers(
            split_results,
            payloads,
            config=effective_config,
            bootstrap_iterations=bootstrap_iterations,
        )
    else:
        information_ledger_result = {
            "schema_version": "token_information_ledger_collection_v1",
            "status": "skipped",
            "skipped_reason": "disabled_by_automation_tier",
        }
    _write_json_atomic(
        output / "information_ledger.json",
        information_ledger_result,
        force=force,
    )

    if plots:
        _write_figures(
            split_results,
            payloads,
            output,
            config=effective_config,
            formats_override=formats,
            force=force,
        )

    modality_summaries = {
        split: {
            modality: {
                "sample_count": int(len(result.sample_ids)),
                "valid_patch_count": int(np.sum(result.valid_mask)),
                "codebook_size": result.codebook_size,
                "active_token_count": sum(
                    not bool(row["inactive"])
                    for row in result.token_result.support_rows
                ),
                "supported_token_count": sum(
                    not bool(row["insufficient_support"])
                    for row in result.token_result.support_rows
                ),
                "feature_count": len(result.feature_names),
                "soft_profile_available": result.posterior is not None,
                "measurement_cache": _json_value(result.measurement_cache),
            }
            for modality, result in modalities.items()
        }
        for split, modalities in split_results.items()
    }
    elapsed = time.perf_counter() - started
    summary = {
        "schema": ATLAS_SCHEMA_VERSION,
        "status": "complete",
        "interpretation": (
            "descriptive token-conditioned measurement phenotypes; token IDs "
            "are nominal and are not physiological-state names"
        ),
        "splits": list(split_results),
        "protected_test_opened": "test" in split_results,
        "modalities": modality_summaries,
        "information_ledger_status": information_ledger_result.get("status"),
        "stability_status": stability.get("status", "ok"),
        "wall_time_seconds": elapsed,
    }
    _write_json_atomic(output / "summary.json", summary, force=force)
    return output


__all__ = [
    "ATLAS_SCHEMA_VERSION",
    "ModalitySplitAtlas",
    "analyze_export_split",
    "build_token_physiology_atlas",
    "load_token_export",
    "prepare_measurement_feature_caches",
]
