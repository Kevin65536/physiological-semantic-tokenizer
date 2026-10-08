"""Versioned, auditable measurement adapters for physiology-semantic inputs.

The adapter removes a record-level baseline and applies a scale learned only
from training records.  It deliberately does not normalize individual crops:
the same samples therefore receive the same canonical values regardless of
where a downstream crop is taken.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np


ADAPTER_SCHEMA = "physiology_measurement_adapter_v1"
PAIRED_ADAPTER_SCHEMA = "physiology_paired_measurement_adapter_v2"
_MAD_TO_STD = 1.482602218505602


def measurement_unit_conversion(
    native_unit: str, *, quantity: str, evidence: str, group: str,
) -> dict[str, Any]:
    """Resolve an evidenced measurement unit; unknown exports stay in their group.

    An optical detector voltage is not an EEG potential. Concentration times
    pathlength is deliberately absent from the concentration conversion table.
    """
    unit = str(native_unit).strip().replace("µ", "u").replace("μ", "u")
    tables = {
        "electric_potential": ("uV", {"V": 1e6, "mV": 1e3, "uV": 1., "nV": 1e-3}),
        "concentration_change": ("uM", {"mol/L": 1e6, "M": 1e6, "mmol/L": 1e3,
                                         "mM": 1e3, "umol/L": 1., "uM": 1.}),
        "optical_density": ("dimensionless", {"dimensionless": 1.}),
        "detector_voltage": ("V", {"V": 1.}),
    }
    if quantity not in tables:
        raise ValueError(f"unsupported measurement quantity: {quantity}")
    canonical, factors = tables[quantity]
    known = bool(evidence) and unit in factors
    factor = factors[unit] if known else 1.
    return dict(native_unit=str(native_unit), quantity=quantity, evidence=str(evidence),
                unit_status="verified" if known else "unknown",
                output_unit=canonical if known else f"relative:{group}",
                factor=factor, inverse_factor=1. / factor,
                measurement_group=quantity + ":" + (canonical if known else str(group)))


def measurement_baseline(values, time_s, valid_mask, *, interval_s, role, evidence,
                         minimum_samples=2):
    """Baseline only a declared interval on real, common channel support.

    The caller supplies times relative to the documented event, or explicitly
    labels a reference interval that is not rest. No task-mean substitution.
    Returned weights define C = I - 1 w^T for both mean and noise propagation.
    """
    values = _as_time_channels(values)
    times = np.asarray(time_s, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim == 1:
        mask = mask[:,None]
    mask = np.broadcast_to(mask,values.shape) & np.isfinite(values)
    start, stop = map(float,interval_s)
    if (times.shape != (len(values),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.isfinite([start,stop]).all() or stop <= start or not role or not evidence
            or minimum_samples < 2):
        raise ValueError('baseline requires a documented role, interval and increasing seconds clock')
    selected = (times >= start) & (times < stop) & mask.all(axis=1)
    count = int(selected.sum())
    state = dict(role=role,evidence=evidence,interval_s=[start,stop],sample_count=count,
                 minimum_samples=minimum_samples,common_channel_support=True,
                 status='available' if count >= minimum_samples else 'insufficient_support',
                 resting_state_inferred=False)
    if count < minimum_samples:
        return None, np.zeros(len(values)), state
    weights = selected.astype(float)/count
    baseline = weights @ np.where(mask,values,0.)
    state['weights'] = weights.tolist()
    return baseline, weights, state


def _as_time_channels(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"measurement values must have shape [time, channels], got {array.shape}")
    return array


def robust_location_scale(values: np.ndarray, *, epsilon: float = 1e-8) -> tuple[float, float]:
    """Return a finite pooled median and robust standard-deviation estimate."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot estimate robust statistics without finite values")
    location = float(np.median(finite))
    scale = float(_MAD_TO_STD * np.median(np.abs(finite - location)))
    if not np.isfinite(scale) or scale < epsilon:
        q25, q75 = np.quantile(finite, [0.25, 0.75])
        scale = float((q75 - q25) / 1.3489795003921634)
    if not np.isfinite(scale) or scale < epsilon:
        scale = float(np.std(finite))
    return location, max(scale, float(epsilon))


@dataclass(frozen=True)
class MeasurementAdapterSpec:
    dataset: str
    modality: str
    original_semantics: str
    original_unit: str
    canonical_semantics: str
    transform: str
    channel_names: tuple[str, ...]
    shared_scale: float
    fit_subjects: tuple[str, ...]
    schema: str = ADAPTER_SCHEMA
    baseline_rule: str = "full_record_channel_median"
    scale_rule: str = "train_only_pooled_mad"
    fit_record_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["channel_names"] = list(self.channel_names)
        payload["fit_subjects"] = list(self.fit_subjects)
        payload["fit_record_ids"] = list(self.fit_record_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MeasurementAdapterSpec":
        values = dict(payload)
        values["channel_names"] = tuple(str(item) for item in values["channel_names"])
        values["fit_subjects"] = tuple(str(item) for item in values["fit_subjects"])
        values['fit_record_ids'] = tuple(str(item) for item in values.get('fit_record_ids', ()))
        return cls(**values)


class PhysiologyMeasurementAdapter:
    """Apply an explicit dataset-specific transform in a shared relative scale."""

    VALID_TRANSFORMS = {"center", "relative_change"}

    def __init__(self, spec: MeasurementAdapterSpec):
        if spec.schema not in (ADAPTER_SCHEMA, PAIRED_ADAPTER_SCHEMA):
            raise ValueError(f"unsupported adapter schema {spec.schema!r}")
        if spec.transform not in self.VALID_TRANSFORMS:
            raise ValueError(f"unsupported measurement transform {spec.transform!r}")
        if not np.isfinite(spec.shared_scale) or spec.shared_scale <= 0:
            raise ValueError("shared_scale must be finite and positive")
        self.spec = spec

    @staticmethod
    def record_baseline(values: np.ndarray) -> np.ndarray:
        array = _as_time_channels(values)
        baseline = np.nanmedian(array, axis=0)
        if not np.all(np.isfinite(baseline)):
            raise ValueError("every channel must contain at least one finite baseline sample")
        return baseline

    @classmethod
    def _relative_values(
        cls,
        values: np.ndarray,
        *,
        baseline: np.ndarray,
        transform: str,
    ) -> np.ndarray:
        array = _as_time_channels(values)
        baseline = np.asarray(baseline, dtype=np.float64).reshape(1, -1)
        if baseline.shape[1] != array.shape[1]:
            raise ValueError("baseline channel count does not match measurement")
        if not np.isfinite(baseline).all():
            raise ValueError('baseline must be finite on declared measurement support')
        centered = array - baseline
        if transform == "center":
            return centered
        if transform == "relative_change":
            finite_baseline = np.abs(baseline[np.isfinite(baseline)])
            reference = float(np.median(finite_baseline)) if finite_baseline.size else 1.0
            floor = max(reference * 1e-3, np.finfo(np.float64).eps)
            return centered / np.maximum(np.abs(baseline), floor)
        raise ValueError(f"unsupported transform {transform!r}")

    @classmethod
    def fit(
        cls,
        records: Iterable[np.ndarray],
        *,
        dataset: str,
        modality: str,
        original_semantics: str,
        original_unit: str,
        canonical_semantics: str,
        transform: str,
        channel_names: Iterable[str],
        fit_subjects: Iterable[str],
        baselines: Iterable[np.ndarray] | None = None,
        valid_masks: Iterable[np.ndarray] | None = None,
        fit_record_ids: Iterable[str] = (),
    ) -> "PhysiologyMeasurementAdapter":
        records = [_as_time_channels(record) for record in records]
        if not records:
            raise ValueError("at least one training record is required")
        if transform not in cls.VALID_TRANSFORMS:
            raise ValueError(f"unsupported measurement transform {transform!r}")
        channel_names = tuple(str(item) for item in channel_names)
        if any(record.shape[1] != len(channel_names) for record in records):
            raise ValueError("all records must match channel_names")
        paired = baselines is not None
        identities = tuple(str(s) for s in fit_record_ids)
        if paired:
            baselines = list(baselines)
            if (len(baselines) != len(records) or len(identities) != len(records)
                    or len(set(identities)) != len(identities)):
                raise ValueError('explicit baseline and unique training identity required per record')
            if modality == 'fnirs':
                pairs = {}
                for name in channel_names:
                    position, _, role = name.rpartition('_')
                    if role not in ('HbO', 'HbR') or role in pairs.setdefault(position, set()):
                        raise ValueError('paired scaling requires unique named HbO/HbR pairs')
                    pairs[position].add(role)
                if any(roles != {'HbO', 'HbR'} for roles in pairs.values()) or transform != 'center':
                    raise ValueError('HbO/HbR require complete pairs and a shared positive scale')
        else:
            baselines = [cls.record_baseline(record) for record in records]
        if valid_masks is not None:
            masks = list(valid_masks)
            if len(masks) != len(records):
                raise ValueError('one true measurement support mask required per training record')
            supported = []
            for record, mask in zip(records, masks):
                mask = np.asarray(mask, dtype=bool)
                if mask.ndim == 1:
                    mask = mask[:, None]
                supported.append(np.where(np.broadcast_to(mask, record.shape), record, np.nan))
            records = supported
        relative = [
            cls._relative_values(
                record,
                baseline=baseline,
                transform=transform,
            )
            for record, baseline in zip(records,baselines)
        ]
        _, scale = robust_location_scale(np.concatenate([item.ravel() for item in relative]))
        spec = MeasurementAdapterSpec(
            dataset=str(dataset),
            modality=str(modality),
            original_semantics=str(original_semantics),
            original_unit=str(original_unit),
            canonical_semantics=str(canonical_semantics),
            transform=str(transform),
            channel_names=channel_names,
            shared_scale=scale,
            fit_subjects=tuple(str(item) for item in fit_subjects),
            fit_record_ids=identities,
            schema=PAIRED_ADAPTER_SCHEMA if paired else ADAPTER_SCHEMA,
            baseline_rule='explicit_declared_support' if paired else 'full_record_channel_median',
        )
        return cls(spec)

    def transform(self, values: np.ndarray, *, baseline: np.ndarray | None = None) -> np.ndarray:
        array = _as_time_channels(values)
        if array.shape[1] != len(self.spec.channel_names):
            raise ValueError("measurement channel count does not match adapter")
        if baseline is None:
            if self.spec.schema == PAIRED_ADAPTER_SCHEMA:
                raise ValueError('paired measurement transform requires an explicit declared baseline')
            baseline = self.record_baseline(array)
        relative = self._relative_values(array, baseline=baseline, transform=self.spec.transform)
        return relative / self.spec.shared_scale

    def inverse_transform(self, canonical: np.ndarray, *, baseline: np.ndarray) -> np.ndarray:
        canonical = _as_time_channels(canonical)
        baseline = np.asarray(baseline, dtype=np.float64).reshape(1, -1)
        relative = canonical * self.spec.shared_scale
        if self.spec.transform == "center":
            return relative + baseline
        finite_baseline = np.abs(baseline[np.isfinite(baseline)])
        reference = float(np.median(finite_baseline)) if finite_baseline.size else 1.0
        floor = max(reference * 1e-3, np.finfo(np.float64).eps)
        return relative * np.maximum(np.abs(baseline), floor) + baseline
