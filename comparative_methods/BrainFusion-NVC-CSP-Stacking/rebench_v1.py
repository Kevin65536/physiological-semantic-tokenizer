"""Subject-OOF, complete-training refit BrainFusion rebenchmark estimator.

Reuses the retained NVC, CSP, feature, and stacking implementations without
changing sealed code. Supervised feature state is fitted separately inside each
subject OOF training partition. Final feature/base/meta state consumes the whole
training argument (the common driver supplies all outer-training samples after
selection); there is no local train/selection split. DSR and REFED remain
unsupported. Real-channel input is EEG [N,C,T], Hb [N,2F,T], HbO then HbR by
default. Chunking controls tensor memory, not sample selection.
Recorded-support masks are preserved: missing points use a channel mean fitted
only on the corresponding OOF training partition or final complete training set.
Constant traces have undefined Pearson correlation. Their NVC feature
contributions are filled with zero, without inventing a waveform or sensor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn.functional as F
from comparative_methods.rebench_eeg_common_v1 import encoding_duration

METHOD_ROOT = Path(__file__).resolve().parent
ADAPTER_ROOT = METHOD_ROOT / "adapters"
if str(ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTER_ROOT))

from brainfusion_gpu.features import BrainFusionFeaturePipeline, CSPConfig, TorchCSP, _validate_fit_identities
from brainfusion_gpu.nvc import NVCConfig, spm_hrf
from brainfusion_gpu.pipeline import BrainFusionFoldPipeline
from brainfusion_gpu.stacking import (
    VIEW_ORDER, FoldLocalStackingClassifier, StackingConfig,
    _clone_candidate, _estimator_candidates, _score_matrix,
)


SUPPORTED_TASKS = ("motor_imagery", "mental_arithmetic", "wg", "nback", "visual")
TASK_CLASS_COUNTS = {"motor_imagery": 2, "mental_arithmetic": 2, "wg": 2, "nback": 3, "visual": 4}
UNSUPPORTED_TASKS = {"dsr": "event-level DSR adapter is not validated", "refed_regression": "continuous sequence regression is not implemented"}
SEARCH_SOURCE = "adapters/brainfusion_gpu/{features.py,stacking.py} retained default anchor and finite neighbourhood"
DEFAULT_SEARCH = (
    {"nvc_pair_count": 32, "components_per_class": 2, "csp_regularization": 0.1, "meta_svm_c": 1.0},
    {"nvc_pair_count": 16, "components_per_class": 2, "csp_regularization": 0.1, "meta_svm_c": 1.0},
    {"nvc_pair_count": 32, "components_per_class": 2, "csp_regularization": 0.3, "meta_svm_c": 1.0},
    {"nvc_pair_count": 32, "components_per_class": 3, "csp_regularization": 0.1, "meta_svm_c": 0.1},
)


def _subject_macro_f1(labels: np.ndarray, predictions: np.ndarray, subjects: np.ndarray, classes: np.ndarray) -> float:
    return float(np.mean([
        f1_score(labels[subjects == subject], predictions[subjects == subject], labels=classes, average="macro", zero_division=0)
        for subject in np.unique(subjects)
    ]))


class _BatchedFeatures(BrainFusionFeaturePipeline):
    """Same fitted feature definitions using bounded sufficient statistics."""

    def __init__(self, *, batch_size: int, device: torch.device | str, **kwargs: Any):
        super().__init__(**kwargs)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.imputation_state_: dict[str, Any] | None = None
        self.nvc_training_degeneracy_: dict[str, int] = {}
        self.last_transform_degeneracy_: dict[str, int] = {}
        if self.batch_size < 1:
            raise ValueError("feature_batch_size must be positive")

    def _chunks(self, count: int):
        for start in range(0, count, self.batch_size):
            yield slice(start, min(count, start + self.batch_size))

    def _device_values(self, eeg: torch.Tensor, hbo: torch.Tensor, hbr: torch.Tensor, rows: slice):
        return tuple(value[rows].to(self.device, dtype=torch.float64) for value in (eeg, hbo, hbr))

    def _nvc(self, eeg, hbo, hbr):
        """Retained NVC on variable traces; undefined pairs are zero features.

        The zero convention applies to a feature used by this adapted model,
        not to the scientific Pearson coefficient, which remains undefined.
        """
        if any(value.ndim != 3 for value in (eeg, hbo, hbr)) or hbo.shape != hbr.shape or eeg.shape[0] != hbo.shape[0]:
            raise ValueError("BrainFusion NVC requires paired [B,C,T] inputs")
        if not all(bool(torch.isfinite(value).all()) for value in (eeg, hbo, hbr)):
            raise ValueError("BrainFusion NVC recorded/imputed inputs contain non-finite values")
        window = self.nvc_config.eeg_window_samples
        if eeg.shape[-1] % window or eeg.shape[-1] // window != hbo.shape[-1]:
            raise ValueError("BrainFusion NVC requires identical complete EEG-summary and Hb grids")

        def minmax(values):
            minimum = values.amin(-1, keepdim=True)
            span = values.amax(-1, keepdim=True) - minimum
            return (values - minimum) / torch.where(span > 0, span, torch.ones_like(span))

        def unit_centered(values):
            values = minmax(values)
            centered = values - values.mean(-1, keepdim=True)
            norm = torch.linalg.vector_norm(centered, dim=-1, keepdim=True)
            return centered / torch.where(norm > 0, norm, torch.ones_like(norm)), (norm > 0).squeeze(-1)

        summary = eeg.reshape(*eeg.shape[:-1], -1, window).mean(-1)
        processed = minmax(summary)
        config = self.nvc_config
        hrf = torch.as_tensor(spm_hrf(config.hrf_tr, oversampling=config.hrf_oversampling, time_length=config.hrf_time_length), dtype=eeg.dtype, device=eeg.device)
        convolved = F.conv1d(F.pad(processed.reshape(-1, 1, processed.shape[-1]), (hrf.numel() - 1, 0)), hrf.flip(0).reshape(1, 1, -1)).reshape_as(processed)
        standardized_eeg, eeg_defined = unit_centered(convolved)
        standardized_hb, hb_defined = unit_centered(torch.cat((hbo, hbr), dim=1))
        undefined_pairs = ~(eeg_defined.unsqueeze(2) & hb_defined.unsqueeze(1))
        self._last_nvc_degeneracy = {
            "window_count": len(eeg),
            "eeg_summary_constant_channel_window_count": int((summary.amax(-1) == summary.amin(-1)).sum()),
            "eeg_hrf_constant_channel_window_count": int((~eeg_defined).sum()),
            "hb_constant_channel_window_count": int((~hb_defined).sum()),
            "undefined_pair_window_count": int(undefined_pairs.sum()),
            "total_pair_window_count": undefined_pairs.numel(),
            "windows_with_undefined_pairs": int(undefined_pairs.flatten(1).any(1).sum()),
        }
        return (standardized_eeg.unsqueeze(2) * standardized_hb.unsqueeze(1)).flatten(1, 2)

    @staticmethod
    def _trial_covariances(values):
        centered = values - values.mean(-1, keepdim=True)
        # Exactly constant filled channels carry no temporal covariance, even
        # when floating-point summation of their mean is not exactly constant.
        variable = values.amax(-1, keepdim=True) > values.amin(-1, keepdim=True)
        centered = torch.where(variable, centered, torch.zeros_like(centered))
        covariance = centered @ centered.transpose(-1, -2)
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        if not bool(torch.isfinite(trace).all()):
            raise ValueError("BrainFusion CSP covariance contains non-finite values")
        # A wholly missing/constant view contributes a zero covariance, retaining
        # its sample in the class denominator and its fixed feature dimensions.
        return covariance / torch.where(trace > 0, trace, torch.ones_like(trace))[:, None, None]

    def _fit_covariances(self, csp: TorchCSP, sums: torch.Tensor, counts: torch.Tensor, classes: torch.Tensor, identity: str) -> None:
        covariances = [csp._regularize(sums[index] / counts[index]) for index in range(len(classes))]
        selected = []
        if len(classes) == 2:
            _, filters = csp._generalized_filters(covariances[0], covariances[1])
            count = min(csp.config.components_per_class, filters.shape[0] // 2)
            selected.extend((filters[:count], filters[-count:]))
        else:
            for index, positive in enumerate(covariances):
                negative = torch.stack([value for other, value in enumerate(covariances) if other != index]).mean(0)
                _, filters = csp._generalized_filters(positive, negative)
                selected.append(filters[-min(csp.config.components_per_class, filters.shape[0]) :])
        csp.filters_ = torch.cat(selected).detach().clone()
        csp.classes_ = classes.detach().clone()
        csp.fit_sample_identity_sha256_ = identity

    @torch.no_grad()
    def fit(self, eeg, hbo, hbr, labels, *, sample_ids):
        identity = _validate_fit_identities(sample_ids, len(labels))
        labels = torch.as_tensor(labels, dtype=torch.long, device="cpu")
        classes, counts = torch.unique(labels, sorted=True, return_counts=True)
        if len(classes) < 2 or bool((counts < 2).any()):
            raise ValueError("BrainFusion requires at least two training trials per class")
        classes = classes.to(self.device)
        counts = counts.to(self.device, dtype=torch.float64)
        sums = None
        squares = None
        self.nvc_training_degeneracy_ = {}
        # ANOVA selection needs only per-class sums/squares of pair correlation.
        for rows in self._chunks(len(labels)):
            values = self._device_values(eeg, hbo, hbr, rows)
            correlations = self._nvc(*values).sum(-1)
            for name, count in self._last_nvc_degeneracy.items():
                self.nvc_training_degeneracy_[name] = self.nvc_training_degeneracy_.get(name, 0) + count
            if sums is None:
                sums = torch.zeros((len(classes), correlations.shape[1]), dtype=torch.float64, device=self.device)
                squares = torch.zeros_like(sums)
            chunk_labels = labels[rows].to(self.device)
            for index, label in enumerate(classes):
                selected = correlations[chunk_labels == label]
                sums[index] += selected.sum(0)
                squares[index] += selected.square().sum(0)
        overall = sums.sum(0) / counts.sum()
        means = sums / counts[:, None]
        between = (counts[:, None] * (means - overall).square()).sum(0)
        within = (squares - sums.square() / counts[:, None]).sum(0).clamp_min(0)
        scores = between / within.clamp_min(self.selector.variance_floor)
        score_array = scores.cpu().numpy()
        indices = np.lexsort((np.arange(len(score_array)), -score_array))[: self.selector.pair_count]
        self.selector.indices_ = torch.as_tensor(indices.copy(), dtype=torch.long, device=self.device)
        self.selector.scores_ = scores[self.selector.indices_].detach().clone()
        self.selector.fit_sample_identity_sha256_ = identity
        covariance_sums: dict[str, torch.Tensor] = {}
        for rows in self._chunks(len(labels)):
            eeg_chunk, hbo_chunk, hbr_chunk = self._device_values(eeg, hbo, hbr, rows)
            nvc = self.selector.transform(self._nvc(eeg_chunk, hbo_chunk, hbr_chunk))
            chunk_labels = labels[rows].to(self.device)
            for view, values in zip(VIEW_ORDER, (eeg_chunk, hbo_chunk, hbr_chunk, nvc), strict=True):
                covariance = self._trial_covariances(values)
                if view not in covariance_sums:
                    covariance_sums[view] = torch.zeros((len(classes), values.shape[1], values.shape[1]), dtype=torch.float64, device=self.device)
                for index, label in enumerate(classes):
                    covariance_sums[view][index] += covariance[chunk_labels == label].sum(0)
        for view in VIEW_ORDER:
            self._fit_covariances(self.csps[view], covariance_sums[view], counts, classes, identity)
        self.fit_sample_identity_sha256_ = identity
        return self

    @torch.no_grad()
    def transform(self, eeg, hbo, hbr):
        blocks: dict[str, list[torch.Tensor]] = {view: [] for view in VIEW_ORDER}
        self.last_transform_degeneracy_ = {}
        for rows in self._chunks(eeg.shape[0]):
            values = self._device_values(eeg, hbo, hbr, rows)
            for view, features in super().transform(*values).items():
                blocks[view].append(features.cpu())
            for name, count in self._last_nvc_degeneracy.items():
                self.last_transform_degeneracy_[name] = self.last_transform_degeneracy_.get(name, 0) + count
        if not blocks["eeg"]:
            raise ValueError("BrainFusion refuses an empty prediction partition")
        return {view: torch.cat(values) for view, values in blocks.items()}

    def state_dict(self):
        state = super().state_dict()
        if self.imputation_state_ is not None:
            state["rebench_imputation"] = self.imputation_state_
        state["rebench_nvc_constant_policy"] = "strict_zero_undefined_feature_fill_v1"
        state["rebench_nvc_training_degeneracy"] = self.nvc_training_degeneracy_
        return state


def _support_mask(values: torch.Tensor, mask: Any, name: str) -> torch.Tensor:
    if mask is None:
        return torch.ones((), dtype=torch.bool).expand(values.shape)
    mask = torch.as_tensor(mask, dtype=torch.bool, device="cpu")
    if mask.ndim == 2:
        if tuple(mask.shape) == tuple(values.shape[:2]):
            mask = mask[:, :, None]
        elif tuple(mask.shape) == (values.shape[0], values.shape[-1]):
            mask = mask[:, None, :]
    try:
        return torch.broadcast_to(mask, values.shape)
    except RuntimeError as error:
        raise ValueError(f"BrainFusion {name} mask differs from canonical input shape") from error


def _fit_channel_mean(values: torch.Tensor, mask: torch.Tensor, batch_size: int) -> dict[str, Any]:
    total = torch.zeros(values.shape[1], dtype=torch.float64)
    count = torch.zeros(values.shape[1], dtype=torch.long)
    partial_window_count = 0
    whole_channel_window_count = 0
    for start in range(0, len(values), batch_size):
        chunk = values[start : start + batch_size].to(dtype=torch.float64)
        valid = mask[start : start + batch_size]
        if not bool(torch.isfinite(chunk[valid]).all()):
            raise ValueError("BrainFusion recorded input contains non-finite values")
        total += torch.where(valid, chunk, 0).sum((0, 2))
        count += valid.sum((0, 2))
        partial_window_count += int((~valid).flatten(1).any(1).sum())
        whole_channel_window_count += int((~valid).all(-1).sum())
    if bool((count == 0).any()):
        raise ValueError("BrainFusion training partition has a channel with no recorded support")
    return {
        "channel_mean": total / count,
        "recorded_point_count": count,
        "masked_point_count": values.shape[0] * values.shape[-1] - count,
        "partial_window_count": partial_window_count,
        "whole_channel_window_count": whole_channel_window_count,
    }


def _impute(values: torch.Tensor, mask: torch.Tensor, state: Mapping[str, Any]) -> torch.Tensor:
    if len(state["channel_mean"]) != values.shape[1]:
        raise ValueError("BrainFusion imputer channel inventory differs from fitted training inputs")
    if not bool(torch.isfinite(values[mask]).all()):
        raise ValueError("BrainFusion recorded input contains non-finite values")
    if bool(mask.all()):
        return values
    means = state["channel_mean"].to(dtype=values.dtype).view(1, -1, 1)
    return torch.where(mask, values, means)


def _imputation_audit(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": state["schema"],
        "fit_sample_identity_sha256": state["fit_sample_identity_sha256"],
        "fit_sample_count": state["fit_sample_count"],
        "fit_scope": "recorded_points_of_supplied_training_samples_only",
        "channel_mean_units": "native_measurement_coordinates",
        **{name: {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in state[name].items()}
           for name in ("eeg", "fnirs")},
    }


class RebenchBrainFusionEstimator:
    def __init__(self, profile: Mapping[str, Any], seed: int = 17, device: torch.device | str = "cpu"):
        self.profile = dict(profile)
        self.task = str(profile.get("task", "motor_imagery"))
        if self.task in UNSUPPORTED_TASKS:
            raise NotImplementedError(f"BrainFusion {self.task}: {UNSUPPORTED_TASKS[self.task]}")
        if self.task not in SUPPORTED_TASKS:
            raise ValueError(f"unknown BrainFusion task: {self.task}")
        self.n_classes = TASK_CLASS_COUNTS[self.task]
        if int(profile.get("num_classes", self.n_classes)) != self.n_classes:
            raise ValueError("BrainFusion class count differs from the fixed task vocabulary")
        self.seed = int(seed)
        self.device = torch.device(device)
        eeg_rate = int(profile.get("sampling_rate_hz", profile.get("eeg_sample_rate_hz", 200)))
        fnirs_rate = int(profile.get("fnirs_sampling_rate_hz", profile.get("fnirs_sample_rate_hz", 10)))
        if eeg_rate <= 0 or fnirs_rate <= 0 or eeg_rate % fnirs_rate:
            raise ValueError("BrainFusion needs an integer EEG/fNIRS sampling-rate ratio")
        self.duration_s = encoding_duration(self.task, profile)
        self.nvc_config = NVCConfig(
            eeg_sampling_rate_hz=eeg_rate, fnirs_sampling_rate_hz=fnirs_rate,
            eeg_window_samples=eeg_rate // fnirs_rate,
        )
        self.csp_config = CSPConfig(
            components_per_class=int(profile.get("components_per_class", 2)),
            regularization=float(profile.get("csp_regularization", 0.1)),
        )
        self.stacking_config = StackingConfig(
            inner_folds=int(profile.get("inner_folds", 3)), seed=self.seed,
            linear_svm_c_values=tuple(profile.get("linear_svm_c_values", (0.1, 1.0))),
            rbf_svm_c_values=tuple(profile.get("rbf_svm_c_values", (1.0,))),
            random_forest_estimators=int(profile.get("random_forest_estimators", 200)),
            random_forest_max_depth=profile.get("random_forest_max_depth"),
            meta_svm_c=float(profile.get("meta_svm_c", 1.0)),
        )
        self.pipeline: BrainFusionFoldPipeline | None = None
        self.fit_subjects_: tuple[str, ...] = ()
        self.fit_sample_ids_: tuple[str, ...] = ()
        self.imputation_state_: dict[str, Any] | None = None
        self.spec = {
            "schema": "brainfusion_downstream_rebench_v1",
            "method_identity": "brainfusion_nvc_csp_stacking_reimplementation",
            "task": self.task, "duration_s": self.duration_s,
            "fit_scope": "all_supplied_outer_training_samples",
            "oof_unit": "subject",
            "oof_feature_fit": "oof_training_subjects_only",
            "missing_input_rule": "native_channel_mean_fitted_inside_each_oof_training_partition",
            "nvc_constant_input_rule": "zero_feature_contribution_for_undefined_correlation",
            "csp_zero_energy_trial_rule": "zero_covariance_with_original_class_denominator",
            "selection_metric": "subject_equal_macro_f1_fixed_training_classes",
            "fnirs_component_order": profile.get("fnirs_component_order", "hbo_then_hbr"),
            "nvc_config": asdict(self.nvc_config),
            "feature_batch_size": int(profile.get("feature_batch_size", 32)),
            "unsupported_tasks": UNSUPPORTED_TASKS,
        }

    def _new_features(self):
        return _BatchedFeatures(
            nvc_config=self.nvc_config, csp_config=self.csp_config,
            nvc_pair_count=int(self.profile.get("nvc_pair_count", 32)),
            batch_size=int(self.profile.get("feature_batch_size", 32)), device=self.device,
        )

    def _inputs(self, eeg, fnirs):
        eeg = torch.as_tensor(eeg, device="cpu")
        fnirs = torch.as_tensor(fnirs, device="cpu")
        if eeg.ndim != 3 or fnirs.ndim != 3 or eeg.shape[0] != fnirs.shape[0] or fnirs.shape[1] % 2:
            raise ValueError("BrainFusion expects EEG [N,C,T] and paired Hb [N,2F,T]")
        if eeg.shape[-1] != round(self.duration_s * self.nvc_config.eeg_sampling_rate_hz) or fnirs.shape[-1] != round(self.duration_s * self.nvc_config.fnirs_sampling_rate_hz):
            raise ValueError("BrainFusion input does not match the common encoding window")
        if eeg.shape[1] < 2 or fnirs.shape[1] < 4:
            raise ValueError("BrainFusion needs at least two real EEG and Hb locations")
        if "channel_names" in self.profile and len(self.profile["channel_names"]) != eeg.shape[1]:
            raise ValueError("EEG input differs from the fixed real-channel inventory")
        if "fnirs_location_names" in self.profile and 2 * len(self.profile["fnirs_location_names"]) != fnirs.shape[1]:
            raise ValueError("Hb input differs from the fixed real-location inventory")
        order = self.profile.get("fnirs_component_order", "hbo_then_hbr")
        if order == "hbo_then_hbr":
            hbo, hbr = fnirs.chunk(2, dim=1)
        elif order == "interleaved":
            hbo, hbr = fnirs[:, ::2], fnirs[:, 1::2]
        else:
            raise ValueError("unknown fnirs_component_order")
        return eeg, hbo, hbr

    def _fit_imputation(self, eeg, fnirs, eeg_mask, fnirs_mask, sample_ids):
        batch_size = int(self.profile.get("feature_batch_size", 32))
        if batch_size < 1:
            raise ValueError("feature_batch_size must be positive")
        return {
            "schema": "brainfusion_masked_channel_mean_imputation_v1",
            "fit_sample_identity_sha256": _validate_fit_identities(sample_ids, len(eeg)),
            "fit_sample_count": len(eeg),
            "eeg": _fit_channel_mean(eeg, eeg_mask, batch_size),
            "fnirs": _fit_channel_mean(fnirs, fnirs_mask, batch_size),
        }

    def _imputed_inputs(self, eeg, fnirs, eeg_mask, fnirs_mask, state):
        return self._inputs(_impute(eeg, eeg_mask, state["eeg"]), _impute(fnirs, fnirs_mask, state["fnirs"]))

    def fit(self, eeg, fnirs, labels, *, subjects: Sequence[str], sample_ids: Sequence[str], eeg_mask=None, fnirs_mask=None, **kwargs):
        if kwargs:
            raise TypeError(f"unknown BrainFusion fit arguments: {sorted(kwargs)}")
        eeg, fnirs = torch.as_tensor(eeg, device="cpu"), torch.as_tensor(fnirs, device="cpu")
        self._inputs(eeg, fnirs)
        eeg_mask, fnirs_mask = _support_mask(eeg, eeg_mask, "EEG"), _support_mask(fnirs, fnirs_mask, "Hb")
        target = np.asarray(torch.as_tensor(labels, device="cpu"), dtype=np.int64).reshape(-1)
        subjects = np.asarray([str(value) for value in subjects])
        identities = tuple(str(value) for value in sample_ids)
        if len(target) != len(eeg) or len(subjects) != len(eeg):
            raise ValueError("BrainFusion labels and subjects must match every sample")
        _validate_fit_identities(identities, len(eeg))
        if any(not subject for subject in subjects):
            raise ValueError("BrainFusion requires nonempty subject identities")
        if len(np.unique(subjects)) < self.stacking_config.inner_folds:
            raise ValueError("too few subjects for the declared stacking OOF folds")
        stacking = FoldLocalStackingClassifier(self.stacking_config)
        stacking.classes_ = np.unique(target)
        if not np.array_equal(stacking.classes_, np.arange(self.n_classes)):
            raise ValueError("BrainFusion training partition lacks the fixed task class vocabulary")
        splits, fold_reports = stacking._splits(target, subjects, identities)
        oof_feature_folds = []
        for (train, validation), report in zip(splits, fold_reports, strict=True):
            train_ids = [identities[index] for index in train]
            imputation = self._fit_imputation(eeg[train], fnirs[train], eeg_mask[train], fnirs_mask[train], train_ids)
            train_inputs = self._imputed_inputs(eeg[train], fnirs[train], eeg_mask[train], fnirs_mask[train], imputation)
            validation_inputs = self._imputed_inputs(eeg[validation], fnirs[validation], eeg_mask[validation], fnirs_mask[validation], imputation)
            features = self._new_features().fit(
                *train_inputs, torch.from_numpy(target[train]), sample_ids=train_ids,
            )
            train_values = BrainFusionFoldPipeline._numpy_features(features.transform(*train_inputs))
            validation_values = BrainFusionFoldPipeline._numpy_features(features.transform(*validation_inputs))
            oof_feature_folds.append((train, validation, train_values, validation_values))
            report["train_subjects"] = sorted(set(subjects[train].tolist()))
            report["validation_subjects"] = sorted(set(subjects[validation].tolist()))
            report["feature_fit_audit"] = features.audit_state()
            report["imputation_fit_audit"] = _imputation_audit(imputation)
            report["nvc_training_degeneracy"] = dict(features.nvc_training_degeneracy_)
            # Save only the small feature matrices; release covariances/GPU state.
            del features, train_inputs, validation_inputs
        imputation = self._fit_imputation(eeg, fnirs, eeg_mask, fnirs_mask, identities)
        final_inputs = self._imputed_inputs(eeg, fnirs, eeg_mask, fnirs_mask, imputation)
        final_features = self._new_features()
        full_values = BrainFusionFoldPipeline._numpy_features(final_features.fit_transform(
            *final_inputs, torch.from_numpy(target), sample_ids=identities,
        ))
        final_features.imputation_state_ = imputation
        selection = {}
        oof_blocks = []
        candidates = [name for name, _ in _estimator_candidates(self.stacking_config)]
        for view in VIEW_ORDER:
            outputs = {}
            scores = {}
            for name in candidates:
                oof = np.full((len(target), len(stacking.classes_)), np.nan)
                for train, validation, train_values, validation_values in oof_feature_folds:
                    scaler = StandardScaler().fit(train_values[view])
                    estimator = _clone_candidate(name, self.stacking_config)
                    estimator.fit(scaler.transform(train_values[view]), target[train])
                    oof[validation] = _score_matrix(estimator, scaler.transform(validation_values[view]), stacking.classes_)
                if not np.isfinite(oof).all():
                    raise RuntimeError("BrainFusion subject OOF predictions are incomplete")
                outputs[name] = oof
                predictions = stacking.classes_[oof.argmax(1)]
                scores[name] = _subject_macro_f1(target, predictions, subjects, stacking.classes_)
            chosen = max(candidates, key=lambda name: (scores[name], -candidates.index(name)))
            oof_blocks.append(outputs[chosen])
            stacking.selected_candidates_[view] = chosen
            selection[view] = {"candidate_subject_macro_f1": scores, "selected_candidate": chosen}
            scaler = StandardScaler().fit(full_values[view])
            estimator = _clone_candidate(chosen, self.stacking_config)
            estimator.fit(scaler.transform(full_values[view]), target)
            stacking.view_scalers_[view] = scaler
            stacking.base_estimators_[view] = estimator
        meta = np.concatenate(oof_blocks, axis=1)
        stacking.meta_scaler_ = StandardScaler().fit(meta)
        stacking.meta_estimator_ = SVC(C=self.stacking_config.meta_svm_c, kernel="linear", class_weight="balanced", decision_function_shape="ovr", random_state=self.seed)
        stacking.meta_estimator_.fit(stacking.meta_scaler_.transform(meta), target)
        stacking.fit_sample_identity_sha256_ = final_features.fit_sample_identity_sha256_
        stacking.audit_ = {
            "schema": "brainfusion_downstream_rebench_v1_stacking_audit",
            "fit_sample_count": len(target),
            "fit_sample_identity_sha256": stacking.fit_sample_identity_sha256_,
            "fit_group_count": len(np.unique(subjects)),
            "classes": stacking.classes_.tolist(), "inner_folds": fold_reports,
            "inner_validation_covers_training_exactly_once": True,
            "oof_feature_fitting_subject_isolated": True,
            "selection": selection,
            "meta_fit_scope": "outer_training_subject_oof_base_scores_only",
            "validation_or_protected_labels_consumed": False,
        }
        pipeline = BrainFusionFoldPipeline(features=final_features, stacking_config=self.stacking_config)
        pipeline.stacking = stacking
        self.pipeline = pipeline
        self.fit_subjects_ = tuple(sorted(set(subjects.tolist())))
        self.fit_sample_ids_ = identities
        self.imputation_state_ = imputation
        self.classes_ = stacking.classes_
        return self

    def _fitted(self) -> BrainFusionFoldPipeline:
        if self.pipeline is None:
            raise RuntimeError("BrainFusion estimator must be fitted before prediction")
        return self.pipeline

    def _prediction_inputs(self, eeg, fnirs, eeg_mask, fnirs_mask):
        self._fitted()
        eeg, fnirs = torch.as_tensor(eeg, device="cpu"), torch.as_tensor(fnirs, device="cpu")
        self._inputs(eeg, fnirs)
        return self._imputed_inputs(eeg, fnirs, _support_mask(eeg, eeg_mask, "EEG"), _support_mask(fnirs, fnirs_mask, "Hb"), self.imputation_state_)

    def predict(self, eeg, fnirs, *, eeg_mask=None, fnirs_mask=None) -> np.ndarray:
        values = self._prediction_inputs(eeg, fnirs, eeg_mask, fnirs_mask)
        prediction = self._fitted().predict(*values)
        self.last_prediction_degeneracy_ = dict(self._fitted().features.last_transform_degeneracy_)
        return prediction

    def decision_function(self, eeg, fnirs, *, eeg_mask=None, fnirs_mask=None) -> np.ndarray:
        values = self._prediction_inputs(eeg, fnirs, eeg_mask, fnirs_mask)
        scores = self._fitted().decision_function(*values)
        self.last_prediction_degeneracy_ = dict(self._fitted().features.last_transform_degeneracy_)
        return scores

    def audit_state(self) -> dict[str, Any]:
        return {"spec": self.spec, "fit_subjects": list(self.fit_subjects_), "pipeline": self._fitted().audit_state(), "imputation": _imputation_audit(self.imputation_state_), "nvc_training_degeneracy": dict(self._fitted().features.nvc_training_degeneracy_)}

    def save(self, directory: str | Path) -> Path:
        return self._fitted().save(directory)

    @classmethod
    def load(
        cls, directory: str | Path, *, profile: Mapping[str, Any],
        seed: int = 17, device: torch.device | str = "cpu",
    ) -> "RebenchBrainFusionEstimator":
        output = cls(profile, seed, device)
        pipeline = BrainFusionFoldPipeline.load(directory, device=device)
        state = torch.load(Path(directory) / "feature_state.pt", map_location="cpu", weights_only=True)
        if state.get("rebench_nvc_constant_policy") != "strict_zero_undefined_feature_fill_v1":
            raise ValueError("BrainFusion checkpoint predates undefined-correlation zero features")
        imputation = state.get("rebench_imputation")
        if not isinstance(imputation, dict) or imputation.get("schema") != "brainfusion_masked_channel_mean_imputation_v1":
            raise ValueError("BrainFusion checkpoint predates recorded-support mean imputation")
        if imputation.get("fit_sample_identity_sha256") != pipeline.features.fit_sample_identity_sha256_:
            raise ValueError("BrainFusion imputation and feature training identities differ")
        for name in ("eeg", "fnirs"):
            if not bool(torch.isfinite(imputation[name]["channel_mean"]).all()) or bool((imputation[name]["recorded_point_count"] <= 0).any()):
                raise ValueError("BrainFusion imputer checkpoint contains invalid training support")
        if pipeline.features.nvc_config != output.nvc_config or pipeline.features.csps["eeg"].config != output.csp_config:
            raise ValueError("BrainFusion reload profile differs from fitted feature configuration")
        if pipeline.stacking.config != output.stacking_config:
            raise ValueError("BrainFusion reload profile differs from fitted stacking configuration")
        # The retained checkpoint reader restores numerical state. Reattach the
        # bounded-memory transform before any prediction on the native panel.
        features = output._new_features()
        features.selector = pipeline.features.selector
        features.csps = pipeline.features.csps
        features.fit_sample_identity_sha256_ = pipeline.features.fit_sample_identity_sha256_
        features.imputation_state_ = imputation
        features.nvc_training_degeneracy_ = dict(state["rebench_nvc_training_degeneracy"])
        pipeline.features = features
        output.pipeline = pipeline
        output.imputation_state_ = imputation
        output.classes_ = pipeline.stacking.classes_
        output.fit_subjects_ = tuple(sorted({
            subject for fold in pipeline.stacking.audit_state()["inner_folds"]
            for subject in fold["train_subjects"] + fold["validation_subjects"]
        }))
        return output


def create_estimator(
    profile: Mapping[str, Any], seed: int = 17, device: torch.device | str = "cpu", **kwargs: Any,
) -> RebenchBrainFusionEstimator:
    return RebenchBrainFusionEstimator({**dict(profile), **kwargs}, seed, device)
