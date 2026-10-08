"""Validation plans for dataset loading and cross-modal synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from .registry import DatasetRegistration, get_dataset_registration, list_registered_datasets


@dataclass(frozen=True)
class ValidationCheck:
    check_id: str
    stage: str
    title: str
    objective: str
    method: str
    pass_criteria: Sequence[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'check_id': self.check_id,
            'stage': self.stage,
            'title': self.title,
            'objective': self.objective,
            'method': self.method,
            'pass_criteria': list(self.pass_criteria),
        }


def _base_checks(registration: DatasetRegistration) -> List[ValidationCheck]:
    return [
        ValidationCheck(
            check_id='root-layout',
            stage='pre_loader',
            title='Root layout and subject discovery',
            objective='Verify that the dataset root and expected subject/task folders exist before any modality parser is used.',
            method='Enumerate dataset root, subject folders, and raw documentation paths from the registry.',
            pass_criteria=(
                'Configured data_root exists.',
                'At least one subject or record is discoverable with the dataset-specific naming scheme.',
                'Referenced original documentation files exist locally.',
            ),
        ),
        ValidationCheck(
            check_id='eeg-parse-smoke',
            stage='loader_smoke',
            title='EEG parse smoke test',
            objective='Confirm that one representative EEG record can be opened and basic metadata can be extracted.',
            method='Load one subject/task record, then read channel labels, sampling rate, and signal shape.',
            pass_criteria=(
                'EEG sample rate is readable.',
                'EEG channel labels are non-empty and match the loaded data dimension.',
                'The EEG array contains finite values and at least one non-zero segment.',
            ),
        ),
        ValidationCheck(
            check_id='fnirs-parse-smoke',
            stage='loader_smoke',
            title='fNIRS parse smoke test',
            objective='Confirm that one representative fNIRS record can be opened and metadata can be extracted.',
            method='Load one subject/task record, then read channel labels, sampling rate, signal types, and signal shape.',
            pass_criteria=(
                'fNIRS sample rate is readable or reconstructable from metadata.',
                'At least one valid fNIRS signal family is present.',
                'Loaded fNIRS arrays contain finite values and consistent channel counts.',
            ),
        ),
        ValidationCheck(
            check_id='window-shape-consistency',
            stage='post_loader',
            title='Window extraction and label consistency',
            objective='Ensure that extracted EEG/fNIRS samples have deterministic shapes and usable labels.',
            method='Build a minimal dataset split and sample several windows from train/val/test partitions.',
            pass_criteria=(
                'Returned tensors match config-driven window duration or sample counts.',
                'Labels are non-empty and drawn from the expected task label vocabulary.',
                'No NaN or Inf appears after preprocessing or normalization.',
            ),
        ),
        ValidationCheck(
            check_id='global-visual-alignment',
            stage='visual_validation',
            title='Global continuous alignment visualization',
            objective='Inspect one continuous record with EEG and fNIRS on the same aligned timeline before trusting model training inputs.',
            method='Run the continuous alignment visualization tool and inspect the global PNG plus JSON timing summary.',
            pass_criteria=(
                'Global figure shows EEG and fNIRS events on a shared axis.',
                'Continuous traces are non-flat and event order is visually consistent.',
                'JSON summary reports acceptable marker count agreement and timing residuals.',
            ),
        ),
        ValidationCheck(
            check_id='local-visual-alignment',
            stage='visual_validation',
            title='Local event zoom visualization',
            objective='Inspect one representative event window in detail to confirm local alignment around the task onset.',
            method='Generate the local zoom figure around a chosen event and compare EEG and fNIRS onsets inside the zoomed window.',
            pass_criteria=(
                'Local figure marks the focus event and task interval clearly.',
                'Selected EEG and fNIRS channels show plausible task-locked responses near the focus event.',
                'Focus event metadata in summary.json matches the rendered local figure.',
            ),
        ),
    ]


def _sync_checks(registration: DatasetRegistration) -> List[ValidationCheck]:
    if registration.sync_strategy == 'shared_parallel_port_markers':
        return [
            ValidationCheck(
                check_id='shared-trigger-alignment',
                stage='sync_validation',
                title='Shared trigger alignment',
                objective='Verify that EEG and fNIRS event streams share the same task structure and event order.',
                method='Compare per-session/task marker counts, class labels, and onset sequences after applying dataset-specific marker mapping.',
                pass_criteria=(
                    'Marker counts match for paired EEG/fNIRS task segments.',
                    'Mapped class labels match for all common events.',
                    'Onset offset is stable within a dataset-specific tolerance and drift is negligible.',
                ),
            ),
            ValidationCheck(
                check_id='interval-drift-check',
                stage='sync_validation',
                title='Inter-event interval drift check',
                objective='Detect whether EEG and fNIRS clocks drift apart over long concatenated recordings.',
                method='Compare successive inter-event intervals and first/last common events after converting timestamps to seconds.',
                pass_criteria=(
                    'Mean interval difference remains near zero after marker mapping.',
                    'Cumulative drift over the full record stays within the chosen tolerance budget.',
                ),
            ),
        ]

    if registration.sync_strategy == 'continuous_annotation_alignment':
        return [
            ValidationCheck(
                check_id='record-duration-alignment',
                stage='sync_validation',
                title='Record duration alignment',
                objective='Verify that EEG, fNIRS, and annotation streams cover the same per-video time span.',
                method='For each sampled video, compare EEG duration, fNIRS duration, and dynamic annotation length after resampling to seconds.',
                pass_criteria=(
                    'Per-video EEG and fNIRS durations agree within tolerance.',
                    'Annotation duration covers the same interval as the paired neural recordings.',
                    'Video IDs are aligned across EEG, fNIRS, and annotation files.',
                ),
            ),
            ValidationCheck(
                check_id='annotation-resample-check',
                stage='sync_validation',
                title='Annotation resampling sanity check',
                objective='Confirm that resampled valence/arousal traces remain monotonic in time and align with chosen window boundaries.',
                method='Resample annotations to the loader timeline, then validate shape, monotonic timestamps, and window coverage.',
                pass_criteria=(
                    'Resampled annotation length matches the target timeline.',
                    'Each model window is fully covered by a valid annotation slice.',
                ),
            ),
        ]

    return [
        ValidationCheck(
            check_id='cross-device-event-reconstruction',
            stage='sync_validation',
            title='Cross-device event reconstruction',
            objective='Reconstruct shared trial timelines when EEG and fNIRS are stored in different raw formats.',
            method='Extract EEG triggers and fNIRS Mark events, rebuild stimulus-onset/stimulus-offset/response triplets, and match trials by order and timing.',
            pass_criteria=(
                'Both modalities yield the same ordered event triplets per record.',
                'Trial counts agree after excluding corrupt or incomplete trials.',
                'Stimulus onset, offset, and response delays are consistent across modalities within tolerance.',
            ),
        ),
        ValidationCheck(
            check_id='label-join-consistency',
            stage='sync_validation',
            title='Behavioral label join consistency',
            objective='Ensure that reconstructed trials can be joined to behavioral labels without ambiguity.',
            method='Join reconstructed event timelines with external label tables or type files and verify one-to-one trial assignment.',
            pass_criteria=(
                'Each retained trial maps to exactly one behavioral label.',
                'No label class is created solely by missing-event fallback logic.',
            ),
        ),
    ]


def build_dataset_validation_plan(dataset_id: str) -> Dict[str, Any]:
    registration = get_dataset_registration(dataset_id)
    checks = [* _base_checks(registration), * _sync_checks(registration)]
    return {
        'dataset_id': registration.dataset_id,
        'display_name': registration.display_name,
        'sync_strategy': registration.sync_strategy,
        'loader_status': registration.loader_status,
        'documentation': [
            {
                'title': ref.title,
                'relative_path': ref.relative_path,
                'kind': ref.kind,
            }
            for ref in registration.documentation
        ],
        'checks': [check.to_dict() for check in checks],
    }


def build_all_validation_plans() -> Dict[str, Dict[str, Any]]:
    return {
        registration.dataset_id: build_dataset_validation_plan(registration.dataset_id)
        for registration in list_registered_datasets()
    }


def render_validation_plan_markdown(dataset_id: str) -> str:
    plan = build_dataset_validation_plan(dataset_id)
    lines = [
        f"# {plan['display_name']} Validation Plan",
        '',
        f"- Dataset id: {plan['dataset_id']}",
        f"- Sync strategy: {plan['sync_strategy']}",
        f"- Loader status: {plan['loader_status']}",
        '',
        '## Documentation',
        '',
    ]
    for doc in plan['documentation']:
        lines.append(f"- {doc['title']}: {doc['relative_path']} ({doc['kind']})")
    lines.extend(['', '## Checks', ''])
    for check in plan['checks']:
        lines.append(f"### {check['title']}")
        lines.append(f"- Stage: {check['stage']}")
        lines.append(f"- Objective: {check['objective']}")
        lines.append(f"- Method: {check['method']}")
        lines.append('- Pass criteria:')
        for criterion in check['pass_criteria']:
            lines.append(f"  - {criterion}")
        lines.append('')
    return '\n'.join(lines).strip() + '\n'
