"""Quality report for the four original EEG-fNIRS datasets.

The report is generated from :class:`UnifiedPhysiologyWindowDataset`, so every
dataset is inspected after the same unit, component, temporal, label, and
geometry contracts.  Croce caches are described only as derived target
provenance and are never counted as a fifth dataset.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

from .fnirs_standardization import DATASET_FNIRS_CONTRACTS, FNIRSMeasurementContract
from .homer2_preprocessing import DATASET_HOMER2_COMPATIBILITY
from .registry import PROJECT_ROOT, get_dataset_registration
from .signal_visualization import compute_power_spectrum
from .unified_physiology import (
    CANONICAL_EEG_SAMPLE_RATE_HZ,
    CANONICAL_FNIRS_COMPONENTS,
    DEFAULT_UNIFIED_WINDOW_DURATION_S,
    CANONICAL_FNIRS_SAMPLE_RATE_HZ,
    CANONICAL_PREPROCESSING,
    CANONICAL_UNIT,
    RAW_DATASET_IDS,
    UnifiedPhysiologyWindowDataset,
    UNIFIED_PHYSIOLOGY_SCHEMA,
)


IMPLEMENTED_DATASET_IDS = set(RAW_DATASET_IDS)
CONTINUOUS_VIS_DATASET_IDS = set(RAW_DATASET_IDS)
PLANNED_DATASET_IDS: set[str] = set()
DERIVED_TARGET_IDS = ("croce_local_cache",)
STAT_KEYS = ("min", "max", "mean", "std", "median", "p01", "p99", "skew", "kurtosis")

_COLOR_PALETTE = {
    "eeg": "#2E86AB",
    "fnirs": "#A23B72",
    "success": "#2E8B57",
    "warning": "#D97706",
    "failure": "#B91C1C",
    "light": "#64748B",
}


@dataclass
class ChannelAmplitudeStats:
    per_channel: Dict[str, Dict[str, float]] = field(default_factory=dict)
    global_: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"per_channel": dict(self.per_channel), "global": dict(self.global_)}


@dataclass
class DatasetQualitySnapshot:
    dataset_id: str
    display_name: str
    loader_status: str
    sync_strategy: str
    eeg_sample_rate_hz: Optional[float] = None
    fnirs_sample_rate_hz: Optional[float] = None
    eeg_channels: Optional[int] = None
    fnirs_channels: Optional[int] = None
    fnirs_contracts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    native_units: Dict[str, str] = field(default_factory=dict)
    canonical_units: Dict[str, str] = field(default_factory=dict)
    component_roles: Dict[str, List[str]] = field(default_factory=dict)
    eeg_amplitude_stats: Optional[Dict[str, Any]] = None
    fnirs_amplitude_stats: Optional[Dict[str, Any]] = None
    waveform_figures: Dict[str, Path] = field(default_factory=dict)
    amplitude_figure: Optional[Path] = None
    montage_figure: Optional[Path] = None
    artifact_summary: Dict[str, Any] = field(default_factory=dict)
    homer2_compatibility: Optional[Dict[str, Any]] = None
    alignment_report: Optional[Dict[str, Any]] = None
    fnirs_standardization_state: Optional[Dict[str, Any]] = None
    label_contract: Optional[Dict[str, Any]] = None
    geometry_summary: Dict[str, Any] = field(default_factory=dict)
    preprocessing_contract: Dict[str, Any] = field(default_factory=dict)
    contract_checks: Dict[str, bool] = field(default_factory=dict)
    data_files_found: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

    @property
    def contract_passed(self) -> bool:
        return bool(self.contract_checks) and all(self.contract_checks.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "display_name": self.display_name,
            "loader_status": self.loader_status,
            "sync_strategy": self.sync_strategy,
            "eeg_sample_rate_hz": self.eeg_sample_rate_hz,
            "fnirs_sample_rate_hz": self.fnirs_sample_rate_hz,
            "eeg_channels": self.eeg_channels,
            "fnirs_channels": self.fnirs_channels,
            "fnirs_contracts": self.fnirs_contracts,
            "native_units": self.native_units,
            "canonical_units": self.canonical_units,
            "component_roles": self.component_roles,
            "eeg_amplitude_stats": self.eeg_amplitude_stats,
            "fnirs_amplitude_stats": self.fnirs_amplitude_stats,
            "waveform_figures": {key: str(value) for key, value in self.waveform_figures.items()},
            "amplitude_figure": str(self.amplitude_figure) if self.amplitude_figure else None,
            "montage_figure": str(self.montage_figure) if self.montage_figure else None,
            "artifact_summary": self.artifact_summary,
            "homer2_compatibility": self.homer2_compatibility,
            "alignment_report": self.alignment_report,
            "fnirs_standardization_state": self.fnirs_standardization_state,
            "label_contract": self.label_contract,
            "geometry_summary": self.geometry_summary,
            "preprocessing_contract": self.preprocessing_contract,
            "contract_checks": self.contract_checks,
            "contract_passed": self.contract_passed,
            "data_files_found": self.data_files_found,
            "issues": self.issues,
        }


@dataclass
class CrossDatasetComparison:
    unit_family_table: List[Dict[str, Any]] = field(default_factory=list)
    sampling_rate_table: List[Dict[str, Any]] = field(default_factory=list)
    channel_count_table: List[Dict[str, Any]] = field(default_factory=list)
    sync_strategy_table: List[Dict[str, Any]] = field(default_factory=list)
    homer2_overview: List[Dict[str, Any]] = field(default_factory=list)
    loader_status_table: List[Dict[str, Any]] = field(default_factory=list)
    contract_table: List[Dict[str, Any]] = field(default_factory=list)


def _unit_family_label(contract: FNIRSMeasurementContract) -> str:
    return f"{contract.measurement_family.replace('_', ' ').title()} ({contract.native_unit})"


def _status_badge(loader_status: str) -> str:
    implemented = loader_status.startswith("implemented")
    colour = _COLOR_PALETTE["success"] if implemented else _COLOR_PALETTE["warning"]
    text = "Implemented" if implemented else "Planned"
    return f'<span class="badge" style="background:{colour}">{text}</span>'


def _figure_to_base64(fig_path: Path) -> str:
    encoded = base64.b64encode(fig_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _figure_to_base64_from_fig(fig: plt.Figure) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _human_size(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _stats(values: np.ndarray) -> Dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return {key: float("nan") for key in STAT_KEYS}
    standard_deviation = float(np.std(finite))
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": standard_deviation,
        "median": float(np.median(finite)),
        "p01": float(np.quantile(finite, 0.01)),
        "p99": float(np.quantile(finite, 0.99)),
        "skew": 0.0 if standard_deviation == 0.0 else float(scipy_skew(finite)),
        "kurtosis": 0.0 if standard_deviation == 0.0 else float(scipy_kurtosis(finite)),
    }


def _compute_amplitude_stats(signal: np.ndarray, *, channel_first: bool | None = None) -> ChannelAmplitudeStats:
    """Compute amplitude statistics with an explicit orientation when known."""
    array = np.asarray(signal)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D signal, got {array.shape}")
    if channel_first is None:
        channel_first = array.shape[0] <= array.shape[1]
    if not channel_first:
        array = array.T
    per_channel = {f"ch_{index}": _stats(array[index]) for index in range(min(array.shape[0], 256))}
    return ChannelAmplitudeStats(per_channel=per_channel, global_=_stats(array))


def _sample_indices(dataset: UnifiedPhysiologyWindowDataset, limit: int) -> List[int]:
    selected: List[int] = []
    seen: set[tuple[str, str]] = set()
    for index, window in enumerate(dataset.windows):
        key = (window.record.join_key, str(window.event.get("label")))
        if key in seen and len(selected) < max(1, limit // 2):
            continue
        selected.append(index)
        seen.add(key)
        if len(selected) >= limit:
            break
    return selected


def _concat_windows(samples: Sequence[Mapping[str, Any]], modality: str) -> np.ndarray:
    arrays = [np.asarray(sample[modality], dtype=np.float32) for sample in samples]
    channel_count = arrays[0].shape[0]
    compatible = [array for array in arrays if array.shape[0] == channel_count]
    return np.concatenate(compatible, axis=1)


def _geometry_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    available = sum(bool(row.get("position_available")) for row in rows)
    systems = sorted({str(row.get("coordinate_system")) for row in rows})
    return {
        "channel_count": len(rows),
        "position_available_count": available,
        "position_available_fraction": float(available / len(rows)) if rows else 0.0,
        "coordinate_systems": systems,
        "schema_uniform": bool(rows) and all(row.get("schema") == "canonical_channel_geometry_v1" for row in rows),
    }


class DatasetQualityReporter:
    def __init__(
        self,
        output_dir: Path,
        *,
        cache_root: Path | str = PROJECT_ROOT / "data/cache/physiology_semantic_clean_v1",
        embed_images: bool = True,
        max_channels: int = 8,
        samples_per_dataset: int = 4,
        window_duration_s: float = DEFAULT_UNIFIED_WINDOW_DURATION_S,
        eeg_signal_branch: str = "single_trial_eeg_artifact_clean_v4",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.cache_root = Path(cache_root)
        self.embed_images = bool(embed_images)
        self.max_channels = int(max_channels)
        self.samples_per_dataset = max(1, int(samples_per_dataset))
        self.window_duration_s = float(window_duration_s)
        self.eeg_signal_branch = str(eeg_signal_branch)

    def compute_snapshot(self, dataset_id: str, *, subject_id: int = 1) -> DatasetQualitySnapshot:
        del subject_id  # selection comes from the canonical cache index
        if dataset_id not in RAW_DATASET_IDS:
            if dataset_id in DERIVED_TARGET_IDS:
                raise ValueError("Croce cache is a derived target cache, not one of the four datasets")
            raise KeyError(f"unknown raw dataset: {dataset_id}")
        registration = get_dataset_registration(dataset_id)
        dataset = UnifiedPhysiologyWindowDataset(
            self.cache_root,
            dataset_ids=[dataset_id],
            window_duration_s=self.window_duration_s,
            eeg_signal_branch=self.eeg_signal_branch,
            require_paired_timestamps=True,
        )
        indices = _sample_indices(dataset, self.samples_per_dataset)
        if not indices:
            raise RuntimeError(f"no paired EEG-fNIRS windows for {dataset_id}")
        samples = [dataset[index] for index in indices]
        first = samples[0]
        first_join_key = dataset.windows[indices[0]].record.join_key
        first_alignment_cases = sorted({
            str(report.get("alignment_case", ""))
            for report in dataset.index.reports_by_join_key.get(first_join_key, [])
        })
        eeg = _concat_windows(samples, "eeg")
        fnirs = _concat_windows(samples, "fnirs")

        contracts = DATASET_FNIRS_CONTRACTS.get(dataset_id, {})
        snapshot = DatasetQualitySnapshot(
            dataset_id=dataset_id,
            display_name=registration.display_name,
            loader_status="implemented_unified_cache_bridge",
            sync_strategy=registration.sync_strategy,
            eeg_sample_rate_hz=float(first["sample_rate_hz"]["eeg"]),
            fnirs_sample_rate_hz=float(first["sample_rate_hz"]["fnirs"]),
            eeg_channels=eeg.shape[0],
            fnirs_channels=fnirs.shape[0],
            fnirs_contracts={key: value.to_dict() for key, value in contracts.items()},
            native_units={
                "eeg": str(first["preprocessing_state"]["eeg"].get("native_unit", "unknown")),
                "fnirs": str(first["preprocessing_state"]["fnirs"].get("native_contract", {}).get("native_unit", "unknown")),
            },
            canonical_units=dict(first["unit"]),
            component_roles={key: list(value) for key, value in first["component_roles"].items()},
            eeg_amplitude_stats=_compute_amplitude_stats(eeg, channel_first=True).to_dict(),
            fnirs_amplitude_stats=_compute_amplitude_stats(fnirs, channel_first=True).to_dict(),
            homer2_compatibility=DATASET_HOMER2_COMPATIBILITY[dataset_id].to_dict(),
            alignment_report=dict(first["alignment"]),
            fnirs_standardization_state=dict(first["preprocessing_state"]["fnirs"]),
            label_contract=dict(first["label"]),
            preprocessing_contract=dict(first["preprocessing_contract"]),
        )
        eeg_geometry = _geometry_summary(first["channel_geometry"]["eeg"])
        fnirs_geometry = _geometry_summary(first["channel_geometry"]["fnirs"])
        snapshot.geometry_summary = {"eeg": eeg_geometry, "fnirs": fnirs_geometry}
        snapshot.artifact_summary = {
            "audited_window_count": len(samples),
            "available_paired_window_count": len(dataset),
            "window_duration_s": self.window_duration_s,
            "eeg_window_samples": int(first["eeg"].shape[1]),
            "fnirs_window_samples": int(first["fnirs"].shape[1]),
            "eeg_finite_fraction": float(np.isfinite(eeg).mean()),
            "fnirs_finite_fraction": float(np.isfinite(fnirs).mean()),
            "eeg_valid_fraction": float(np.mean([np.mean(sample["valid_mask"]["eeg"]) for sample in samples])),
            "eeg_analysis_valid_fraction": float(np.mean([np.mean(sample["analysis_valid_mask"]["eeg"]) for sample in samples])),
            "eeg_artifact_fraction": float(np.mean([np.mean(sample["artifact_mask"]["eeg"]) for sample in samples])),
            "eeg_bad_channel_fraction": float(np.mean([np.mean(sample["bad_channel_mask"]["eeg"]) for sample in samples])),
            "eeg_signal_branch": str(first["eeg_signal_branch"]),
            "eeg_artifact_schema": str(first["preprocessing_state"]["eeg"].get("artifact_cleaning", {}).get("schema", "unavailable")),
            "eeg_artifact_cache_used": bool(
                first["preprocessing_state"]["eeg"].get("artifact_cache", {}).get("used", False)
            ),
            "eeg_artifact_cache_schema": str(
                first["preprocessing_state"]["eeg"].get("artifact_cache", {}).get("schema", "not_applicable")
            ),
            "fnirs_valid_fraction": float(np.mean([np.mean(sample["valid_mask"]["fnirs"]) for sample in samples])),
            "fnirs_component_set": sorted(set(first["component_roles"]["fnirs"])),
            "separate_modality_clocks_used": bool(first["alignment"]["separate_modality_clocks_used"]),
            "admitted_alignment_cases": first_alignment_cases,
            "excluded_alignment_record_count": len(dataset.excluded_alignment_records),
            "excluded_alignment_records": dict(dataset.excluded_alignment_records),
        }
        snapshot.contract_checks = {
            "canonical_units": set(first["unit"].values()) == {CANONICAL_UNIT},
            "canonical_sample_rates": first["sample_rate_hz"] == {
                "eeg": CANONICAL_EEG_SAMPLE_RATE_HZ,
                "fnirs": CANONICAL_FNIRS_SAMPLE_RATE_HZ,
            },
            "configured_window_length": (
                first["eeg"].shape[1] == int(round(self.window_duration_s * CANONICAL_EEG_SAMPLE_RATE_HZ))
                and first["fnirs"].shape[1] == int(round(self.window_duration_s * CANONICAL_FNIRS_SAMPLE_RATE_HZ))
            ),
            "fnirs_hbo_hbr_components": set(first["component_roles"]["fnirs"]) == set(CANONICAL_FNIRS_COMPONENTS),
            "paired_timestamps": first["event"].get("eeg_time_ms") is not None and first["event"].get("fnirs_time_ms") is not None,
            "alignment_admission_filter": bool(first_alignment_cases) and not set(first_alignment_cases).isdisjoint(dataset.admissible_alignment_cases or set()),
            "separate_modality_clocks": bool(first["alignment"]["separate_modality_clocks_used"]),
            "canonical_label_schema": first["label"].get("schema") == "canonical_task_label_v1",
            "canonical_geometry_schema": eeg_geometry["schema_uniform"] and fnirs_geometry["schema_uniform"],
            "finite_amplitudes": bool(np.isfinite(eeg).all() and np.isfinite(fnirs).all()),
            "full_window_coverage": snapshot.artifact_summary["eeg_valid_fraction"] == 1.0 and snapshot.artifact_summary["fnirs_valid_fraction"] == 1.0,
        }
        if dataset_id == "eeg_fnirs_single_trial" and self.eeg_signal_branch == "single_trial_eeg_artifact_clean_v4":
            snapshot.contract_checks["single_trial_v4_cache_loaded"] = bool(
                snapshot.artifact_summary["eeg_artifact_cache_used"]
                and snapshot.artifact_summary["eeg_artifact_cache_schema"]
                == "single_trial_eeg_artifact_cache_v4"
            )
        for check, passed in snapshot.contract_checks.items():
            if not passed:
                snapshot.issues.append(f"contract check failed: {check}")

        snapshot.data_files_found = [
            str(first["preprocessing_state"]["eeg"].get("source_path", "")),
            str(dataset.windows[indices[0]].record.npz_path),
        ]
        self._build_figures(snapshot, first, eeg, fnirs)
        return snapshot

    def _build_figures(
        self,
        snapshot: DatasetQualitySnapshot,
        sample: Mapping[str, Any],
        eeg: np.ndarray,
        fnirs: np.ndarray,
    ) -> None:
        directory = self.figures_dir / snapshot.dataset_id
        directory.mkdir(parents=True, exist_ok=True)
        waveform = directory / "aligned_waveforms.png"
        _save_waveform_figure(sample, waveform, self.max_channels)
        snapshot.waveform_figures["aligned_waveforms"] = waveform
        psd = directory / "power_spectra.png"
        _save_psd_figure(eeg, fnirs, psd)
        snapshot.waveform_figures["power_spectra"] = psd
        amplitude = directory / "canonical_amplitude_distribution.png"
        _save_amplitude_distribution_figure(eeg, fnirs, amplitude)
        snapshot.amplitude_figure = amplitude
        montage = directory / "canonical_channel_geometry.png"
        if _save_geometry_figure(sample["channel_geometry"], montage):
            snapshot.montage_figure = montage

    def compute_cross_dataset_comparisons(
        self, snapshots: Iterable[DatasetQualitySnapshot]
    ) -> CrossDatasetComparison:
        comparison = CrossDatasetComparison()
        for snapshot in snapshots:
            comparison.unit_family_table.append({
                "dataset_id": snapshot.dataset_id,
                "native_eeg_unit": snapshot.native_units.get("eeg", "unknown"),
                "native_fnirs_unit": snapshot.native_units.get("fnirs", "unknown"),
                "canonical_unit": CANONICAL_UNIT,
            })
            comparison.sampling_rate_table.append({
                "dataset_id": snapshot.dataset_id,
                "eeg_sample_rate_hz": snapshot.eeg_sample_rate_hz,
                "fnirs_sample_rate_hz": snapshot.fnirs_sample_rate_hz,
                "ratio": snapshot.eeg_sample_rate_hz / snapshot.fnirs_sample_rate_hz,
            })
            comparison.channel_count_table.append({
                "dataset_id": snapshot.dataset_id,
                "eeg_channels": snapshot.eeg_channels,
                "fnirs_component_channels": snapshot.fnirs_channels,
            })
            comparison.sync_strategy_table.append({
                "dataset_id": snapshot.dataset_id,
                "native_sync_strategy": snapshot.sync_strategy,
                "window_alignment": "separate EEG/fNIRS event clocks",
            })
            comparison.homer2_overview.append({
                "dataset_id": snapshot.dataset_id,
                "entry_stage": snapshot.homer2_compatibility.get("entry_stage", "") if snapshot.homer2_compatibility else "",
                "completeness": snapshot.homer2_compatibility.get("completeness", "") if snapshot.homer2_compatibility else "",
            })
            comparison.loader_status_table.append({
                "dataset_id": snapshot.dataset_id,
                "loader_status": snapshot.loader_status,
            })
            comparison.contract_table.append({
                "dataset_id": snapshot.dataset_id,
                "passed": snapshot.contract_passed,
                "failed_checks": [key for key, value in snapshot.contract_checks.items() if not value],
            })
        return comparison

    def build_full_report(
        self,
        dataset_ids: Optional[Sequence[str]] = None,
        *,
        subject_id: int = 1,
    ) -> Tuple[Path, Path]:
        del subject_id
        selected = list(dataset_ids or RAW_DATASET_IDS)
        invalid = sorted(set(selected) - set(RAW_DATASET_IDS))
        if invalid:
            raise ValueError(f"quality report accepts exactly the four original datasets; invalid={invalid}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        snapshots: List[DatasetQualitySnapshot] = []
        for dataset_id in selected:
            try:
                snapshots.append(self.compute_snapshot(dataset_id))
            except Exception as exc:
                snapshots.append(self._error_snapshot(dataset_id, str(exc)))
        comparison = self.compute_cross_dataset_comparisons(snapshots)
        self._build_cross_dataset_figures(snapshots)
        html_path = self.build_html_report(snapshots, comparison)
        md_path = self.build_markdown_summary(snapshots, comparison)
        payload = {
            "generated": datetime.now().isoformat(),
            "scope": {
                "raw_dataset_ids": selected,
                "dataset_count": len(selected),
                "croce_cache_role": "derived_source_observation_supervision_only",
                "derived_targets_excluded_from_dataset_count": list(DERIVED_TARGET_IDS),
            },
            "loader_contract": {
                "loader_class": "UnifiedPhysiologyWindowDataset",
                "schema": UNIFIED_PHYSIOLOGY_SCHEMA,
                "window_duration_s": self.window_duration_s,
                "eeg_signal_branch": self.eeg_signal_branch,
            },
            "canonical_preprocessing_contract": CANONICAL_PREPROCESSING.to_dict(),
            "all_contract_checks_passed": all(snapshot.contract_passed for snapshot in snapshots),
            "datasets": [snapshot.to_dict() for snapshot in snapshots],
        }
        (self.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return html_path, md_path

    def _build_cross_dataset_figures(self, snapshots: Sequence[DatasetQualitySnapshot]) -> None:
        directory = self.figures_dir / "cross_dataset"
        directory.mkdir(parents=True, exist_ok=True)
        valid = [snapshot for snapshot in snapshots if snapshot.eeg_amplitude_stats and snapshot.fnirs_amplitude_stats]
        if valid:
            _save_cross_dataset_amplitude_figure(valid, directory / "canonical_amplitude_comparison.png")

    def build_html_report(
        self,
        snapshots: List[DatasetQualitySnapshot],
        comparison: CrossDatasetComparison,
    ) -> Path:
        path = self.output_dir / "quality_report.html"
        rows = []
        for snapshot in snapshots:
            rows.append({
                "dataset": snapshot.display_name,
                "contract": "PASS" if snapshot.contract_passed else "FAIL",
                "EEG": f"{snapshot.eeg_channels} ch @ {snapshot.eeg_sample_rate_hz} Hz",
                "fNIRS": f"{snapshot.fnirs_channels} component-ch @ {snapshot.fnirs_sample_rate_hz} Hz",
                "canonical unit": CANONICAL_UNIT,
            })
        parts = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>Four-dataset physiology quality audit</title>",
            _HTML_CSS,
            "</head><body>",
            "<h1>Four-Dataset Unified Physiology Quality Audit</h1>",
            f"<p>Generated {escape(datetime.now().isoformat(timespec='seconds'))}</p>",
            "<div class='notice'><strong>Scope:</strong> exactly four original datasets. Croce caches are derived EEG/fNIRS source/observation supervision targets and are excluded from the dataset count.</div>",
            f"<div class='notice'><strong>Loader:</strong> UnifiedPhysiologyWindowDataset · {UNIFIED_PHYSIOLOGY_SCHEMA} · {self.window_duration_s:g} s observation context · EEG branch {escape(self.eeg_signal_branch)}.</div>",
            "<h2>Final contract status</h2>",
            _dict_list_to_html_table(rows),
            "<h2>Canonical preprocessing</h2>",
            _dict_list_to_html_table([CANONICAL_PREPROCESSING.to_dict()]),
            "<h2>Cross-dataset comparisons</h2>",
            "<h3>Units</h3>", _dict_list_to_html_table(comparison.unit_family_table),
            "<h3>Sampling</h3>", _dict_list_to_html_table(comparison.sampling_rate_table),
            "<h3>Contract checks</h3>", _dict_list_to_html_table(comparison.contract_table),
        ]
        cross = self.figures_dir / "cross_dataset/canonical_amplitude_comparison.png"
        if cross.exists():
            parts.append(_embed_figure(cross, self.embed_images, self.figures_dir))
        for snapshot in snapshots:
            parts.append(self._build_dataset_section(snapshot))
        parts.append("</body></html>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return path

    def _build_dataset_section(self, snapshot: DatasetQualitySnapshot) -> str:
        checks = [{"check": key, "status": "PASS" if value else "FAIL"} for key, value in snapshot.contract_checks.items()]
        parts = [
            f"<section><h2>{escape(snapshot.display_name)} {_status_badge(snapshot.loader_status)}</h2>",
            f"<p><code>{escape(snapshot.dataset_id)}</code></p>",
            "<h3>Contract checks</h3>", _dict_list_to_html_table(checks),
            "<h3>Native provenance and canonical units</h3>",
            _dict_list_to_html_table([{"native_units": snapshot.native_units, "canonical_units": snapshot.canonical_units, "components": sorted(set(snapshot.component_roles.get('fnirs', [])))}]),
            "<h3>Post-unification amplitude</h3>", _build_amplitude_stats_html(snapshot),
            "<h3>Alignment and labels</h3>", _dict_list_to_html_table([{"alignment": snapshot.alignment_report, "label": snapshot.label_contract}]),
            "<h3>Channel geometry</h3>", _dict_list_to_html_table([snapshot.geometry_summary]),
            "<h3>Artifact and validity masks</h3>", _dict_list_to_html_table([snapshot.artifact_summary]),
        ]
        if snapshot.issues:
            parts.append("<div class='failure'><strong>Issues:</strong><ul>" + "".join(f"<li>{escape(issue)}</li>" for issue in snapshot.issues) + "</ul></div>")
        for figure in [*snapshot.waveform_figures.values(), snapshot.amplitude_figure, snapshot.montage_figure]:
            if figure and figure.exists():
                parts.append(_embed_figure(figure, self.embed_images, self.figures_dir))
        parts.append("</section>")
        return "\n".join(parts)

    def build_markdown_summary(
        self,
        snapshots: List[DatasetQualitySnapshot],
        comparison: CrossDatasetComparison,
    ) -> Path:
        del comparison
        path = self.output_dir / "quality_report.md"
        lines = [
            "# Dataset Quality Audit Report",
            "",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "Scope: the four original EEG-fNIRS datasets only. `croce_local_cache` is a derived Croce-2017 source/observation supervision cache, not a dataset.",
            "",
            f"Loader: `UnifiedPhysiologyWindowDataset` / `{UNIFIED_PHYSIOLOGY_SCHEMA}`; observation context: **{self.window_duration_s:g} s**; EEG branch: `{self.eeg_signal_branch}`.",
            "",
            "## Final status",
            "",
            "| Dataset | Contract | EEG | fNIRS | Canonical unit |",
            "| --- | --- | --- | --- | --- |",
        ]
        for snapshot in snapshots:
            status = "PASS" if snapshot.contract_passed else "FAIL"
            lines.append(f"| {snapshot.display_name} | {status} | {snapshot.eeg_channels} channels @ {snapshot.eeg_sample_rate_hz} Hz | {snapshot.fnirs_channels} component-channels @ {snapshot.fnirs_sample_rate_hz} Hz | {CANONICAL_UNIT} |")
        lines.extend(["", "## Dataset details", ""])
        for snapshot in snapshots:
            lines.extend([
                f"### {snapshot.display_name} (`{snapshot.dataset_id}`)",
                "",
                f"- Contract: {'PASS' if snapshot.contract_passed else 'FAIL'}",
                f"- Native units retained in provenance: `{snapshot.native_units}`",
                f"- Canonical components: `{sorted(set(snapshot.component_roles.get('fnirs', [])))}`",
                f"- Canonical label: `{snapshot.label_contract}`",
                f"- Alignment: `{snapshot.alignment_report}`",
                f"- Geometry: `{snapshot.geometry_summary}`",
                f"- Artifact/validity masks: `{snapshot.artifact_summary}`",
            ])
            for modality, stats in (("EEG", snapshot.eeg_amplitude_stats), ("fNIRS", snapshot.fnirs_amplitude_stats)):
                if stats:
                    global_stats = stats["global"]
                    lines.append(
                        f"- {modality} post-unification amplitude: "
                        f"median={global_stats.get('median', global_stats.get('mean', float('nan'))):.4g}, "
                        f"std={global_stats.get('std', float('nan')):.4g}, "
                        f"p01={global_stats.get('p01', global_stats.get('min', float('nan'))):.4g}, "
                        f"p99={global_stats.get('p99', global_stats.get('max', float('nan'))):.4g}"
                    )
            for issue in snapshot.issues:
                lines.append(f"- Issue: {issue}")
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _error_snapshot(self, dataset_id: str, error_message: str) -> DatasetQualitySnapshot:
        try:
            registration = get_dataset_registration(dataset_id)
            display_name, sync = registration.display_name, registration.sync_strategy
        except Exception:
            display_name, sync = dataset_id, "unknown"
        return DatasetQualitySnapshot(
            dataset_id=dataset_id,
            display_name=display_name,
            loader_status="failed",
            sync_strategy=sync,
            issues=[f"Snapshot computation failed: {error_message}"],
            contract_checks={"snapshot_generated": False},
        )


def _save_waveform_figure(sample: Mapping[str, Any], path: Path, max_channels: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)
    for axis, modality, colour in zip(axes, ("eeg", "fnirs"), (_COLOR_PALETTE["eeg"], _COLOR_PALETTE["fnirs"])):
        signal = np.asarray(sample[modality])
        rate = float(sample["sample_rate_hz"][modality])
        time = np.arange(signal.shape[1]) / rate
        count = min(max_channels, signal.shape[0])
        spacing = max(6.0, float(np.nanquantile(np.abs(signal[:count]), 0.99)) * 2.5)
        for channel in range(count):
            axis.plot(time, signal[channel] + channel * spacing, color=colour, linewidth=0.7)
        axis.set_title(f"{modality.upper()} aligned window — {CANONICAL_UNIT}")
        axis.set_xlabel("Time from modality-specific event onset (s)")
        axis.set_ylabel("Channel + offset")
        axis.grid(alpha=0.2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_psd_figure(eeg: np.ndarray, fnirs: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for axis, signal, rate, modality, colour in (
        (axes[0], eeg, CANONICAL_EEG_SAMPLE_RATE_HZ, "EEG", _COLOR_PALETTE["eeg"]),
        (axes[1], fnirs, CANONICAL_FNIRS_SAMPLE_RATE_HZ, "fNIRS", _COLOR_PALETTE["fnirs"]),
    ):
        frequencies, psd = compute_power_spectrum(signal, rate, channel_first=True)
        for channel in range(min(8, psd.shape[0])):
            axis.semilogy(frequencies, psd[channel], color=colour, alpha=0.45, linewidth=0.8)
        axis.set_title(f"{modality} post-unification PSD")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("PSD")
        axis.grid(alpha=0.2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_amplitude_distribution_figure(eeg: np.ndarray, fnirs: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for row, (signal, modality, colour) in enumerate(((eeg, "EEG", _COLOR_PALETTE["eeg"]), (fnirs, "fNIRS", _COLOR_PALETTE["fnirs"]))):
        finite = signal[np.isfinite(signal)]
        if finite.size > 200_000:
            finite = finite[:: max(1, finite.size // 200_000)]
        axes[row, 0].hist(finite, bins=120, color=colour, alpha=0.75)
        axes[row, 0].set_title(f"{modality} global amplitude")
        axes[row, 0].set_xlabel(CANONICAL_UNIT)
        channel_data = [channel[np.isfinite(channel)] for channel in signal[: min(24, signal.shape[0])]]
        axes[row, 1].boxplot(channel_data, showfliers=False)
        axes[row, 1].set_title(f"{modality} per-channel amplitude")
        axes[row, 1].set_ylabel(CANONICAL_UNIT)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_geometry_figure(geometry: Mapping[str, Sequence[Mapping[str, Any]]], path: Path) -> bool:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    plotted = False
    for axis, modality, colour in zip(axes, ("eeg", "fnirs"), (_COLOR_PALETTE["eeg"], _COLOR_PALETTE["fnirs"])):
        rows = [row for row in geometry[modality] if row.get("x") is not None and row.get("y") is not None]
        if rows:
            plotted = True
            axis.scatter([row["x"] for row in rows], [row["y"] for row in rows], color=colour, s=24)
            for row in rows[:64]:
                axis.annotate(str(row["base_channel_name"]), (row["x"], row["y"]), fontsize=5)
        else:
            axis.text(0.5, 0.5, "Position metadata unavailable", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(f"{modality.upper()} canonical geometry rows")
        axis.set_xlabel("Native/referenced X")
        axis.set_ylabel("Native/referenced Y")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plotted


def _save_cross_dataset_amplitude_figure(snapshots: Sequence[DatasetQualitySnapshot], path: Path) -> None:
    labels = [snapshot.dataset_id for snapshot in snapshots]
    eeg_std = [snapshot.eeg_amplitude_stats["global"]["std"] for snapshot in snapshots]
    fnirs_std = [snapshot.fnirs_amplitude_stats["global"]["std"] for snapshot in snapshots]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(x - 0.18, eeg_std, 0.36, label="EEG", color=_COLOR_PALETTE["eeg"])
    ax.bar(x + 0.18, fnirs_std, 0.36, label="fNIRS", color=_COLOR_PALETTE["fnirs"])
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1, label="robust-SD reference")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel(f"Global standard deviation ({CANONICAL_UNIT})")
    ax.set_title("Post-unification amplitude scale across the four datasets")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _embed_figure(path: Path, embed: bool, figures_root: Path) -> str:
    if not path.exists():
        return ""
    if embed:
        source = _figure_to_base64(path)
    else:
        try:
            source = str(path.relative_to(figures_root.parent))
        except ValueError:
            source = str(path)
    return f'<img src="{escape(source)}" alt="{escape(path.stem)}">'


def _display_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return escape(json.dumps(value, ensure_ascii=False))
    if isinstance(value, float):
        return f"{value:.5g}"
    return escape(str(value))


def _dict_list_to_html_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "<p>No data available.</p>"
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{escape(str(key))}</th>" for key in keys)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        parts.extend(f"<td>{_display_value(row.get(key, '—'))}</td>" for key in keys)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _build_amplitude_stats_html(snapshot: DatasetQualitySnapshot) -> str:
    rows = []
    for modality, stats in (("EEG", snapshot.eeg_amplitude_stats), ("fNIRS", snapshot.fnirs_amplitude_stats)):
        if stats:
            rows.append({"modality": modality, "unit": CANONICAL_UNIT, **stats["global"]})
    return _dict_list_to_html_table(rows)


_HTML_CSS = """
<style>
body{font-family:system-ui,sans-serif;max-width:1200px;margin:auto;padding:2rem;color:#172033;line-height:1.45}
h1,h2,h3{color:#174A68}section{border:1px solid #d8e1e8;border-radius:10px;padding:1.2rem;margin:1.5rem 0}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:.7rem 0}th,td{border:1px solid #d8e1e8;padding:.45rem;vertical-align:top}th{background:#174A68;color:white}
img{max-width:100%;margin:.8rem 0;border:1px solid #e2e8f0}.badge{color:white;padding:.15rem .5rem;border-radius:999px;font-size:.72rem}
.notice{padding:1rem;background:#e8f3f8;border-left:5px solid #2E86AB}.failure{padding:.8rem;background:#fff1f2;border-left:5px solid #B91C1C}code{background:#f1f5f9;padding:.1rem .25rem}
</style>
"""
