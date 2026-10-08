"""Dataset registration and shared experiment config normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from .fnirs_standardization import DATASET_FNIRS_CONTRACTS, default_standardization_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / 'experiments' / 'configs'


@dataclass(frozen=True)
class DocumentationReference:
    title: str
    relative_path: str
    kind: str


@dataclass(frozen=True)
class DatasetRegistration:
    dataset_id: str
    display_name: str
    default_root: str
    aliases: Sequence[str]
    supported_modalities: Sequence[str]
    eeg_sample_rate_hz: Optional[float]
    fnirs_sample_rate_hz: Optional[float]
    eeg_channels: Optional[int]
    fnirs_channels: Optional[int]
    default_task: Optional[str]
    sync_strategy: str
    loader_status: str
    documentation: Sequence[DocumentationReference]
    notes: Sequence[str]
    resource_kind: str = 'raw_dataset'
    primary_loader: str = ''
    loader_contract: str = ''
    loader_interfaces: Sequence[str] = ()
    default_eeg_signal_branch: Optional[str] = None
    eeg_artifact_status: Optional[str] = None

    def runtime_metadata(self, data_root: str) -> Dict[str, Any]:
        metadata = {
            'dataset_id': self.dataset_id,
            'display_name': self.display_name,
            'data_root': data_root,
            'default_root': self.default_root,
            'supported_modalities': list(self.supported_modalities),
            'eeg_sample_rate_hz': self.eeg_sample_rate_hz,
            'fnirs_sample_rate_hz': self.fnirs_sample_rate_hz,
            'eeg_channels': self.eeg_channels,
            'fnirs_channels': self.fnirs_channels,
            'default_task': self.default_task,
            'sync_strategy': self.sync_strategy,
            'loader_status': self.loader_status,
            'documentation': [
                {
                    'title': ref.title,
                    'relative_path': ref.relative_path,
                    'kind': ref.kind,
                }
                for ref in self.documentation
            ],
            'notes': list(self.notes),
            'resource_kind': self.resource_kind,
            'primary_loader': self.primary_loader,
            'loader_contract': self.loader_contract,
            'loader_interfaces': list(self.loader_interfaces),
            'default_eeg_signal_branch': self.default_eeg_signal_branch,
            'eeg_artifact_status': self.eeg_artifact_status,
        }
        contracts = DATASET_FNIRS_CONTRACTS.get(self.dataset_id)
        if contracts:
            metadata['fnirs_measurement_contracts'] = {
                key: value.to_dict() for key, value in contracts.items()
            }
        return metadata


REGISTERED_DATASETS: Dict[str, DatasetRegistration] = {
    'croce_local_cache': DatasetRegistration(
        dataset_id='croce_local_cache',
        display_name='Croce local source/observation cache',
        default_root='croce_validation/cache',
        aliases=(
            'croce cache',
            'croce_local',
            'croce source observation cache',
        ),
        supported_modalities=('both',),
        eeg_sample_rate_hz=200.0,
        fnirs_sample_rate_hz=10.0,
        eeg_channels=6,
        fnirs_channels=1,
        default_task='mixed',
        sync_strategy='anchor_event_window_cache',
        loader_status='implemented',
        documentation=(
            DocumentationReference(
                title='Current source/observation architecture',
                relative_path='docs/ARCHITECTURE.md',
                kind='markdown',
            ),
        ),
        notes=(
            'Each sample is one fNIRS spatial anchor with its local six-channel EEG neighbourhood.',
            'This is a derived Croce-2017 supervision cache, not a fifth raw dataset.',
            'It is not the primary input of any newly planned experiment; use is limited to named historical or teacher ablations.',
        ),
        resource_kind='derived_supervision_cache',
        primary_loader='CroceLocalCacheDataset',
        loader_contract='croce_source_observation_cache',
        loader_interfaces=('derived_cache', 'legacy_multimodal'),
    ),
    'eeg_fnirs_single_trial': DatasetRegistration(
        dataset_id='eeg_fnirs_single_trial',
        display_name='EEG+NIRS Single-Trial',
        default_root='data/EEG+NIRS Single-Trial',
        aliases=(
            'single_trial',
            'eeg+fnirs single-trial',
            'eeg+nirs single-trial',
            'tu_berlin_single_trial',
        ),
        supported_modalities=('eeg', 'fnirs', 'both'),
        eeg_sample_rate_hz=200.0,
        fnirs_sample_rate_hz=10.0,
        eeg_channels=30,
        fnirs_channels=36,
        default_task='motor_imagery',
        sync_strategy='shared_parallel_port_markers',
        loader_status='implemented',
        documentation=(
            DocumentationReference(
                title='Original HTML description',
                relative_path='data/EEG+NIRS Single-Trial/Open access dataset for simultaneous EEG and NIRS Brain-Computer Interfaces (BCIs).html',
                kind='html',
            ),
            DocumentationReference(
                title='Project dataset summary',
                relative_path='docs/DATASETS_DESCRIPTION.md',
                kind='markdown',
            ),
        ),
        notes=(
            'BBCI toolbox cell-array structure with six sessions per subject.',
            'EEG and fNIRS triggers are delivered simultaneously through a parallel port.',
            'fNIRS keeps direct lowWL/highWL optical measurements at 760/850 nm in the checked repository files.',
            'Task EEG is released with ocular artifacts; single_trial_eeg_artifact_clean_v4 is the default with 50 Hz removal and no dataset-specific bad-channel mask.',
        ),
        primary_loader='UnifiedPhysiologyWindowDataset',
        loader_contract='unified_physiology_measurement_window_v3',
        loader_interfaces=('unified_physiology', 'legacy_multimodal', 'continuous_visualization'),
        default_eeg_signal_branch='single_trial_eeg_artifact_clean_v4',
        eeg_artifact_status='artifact_clean_v4_line_clean_no_bad_mask',
    ),
    'refed': DatasetRegistration(
        dataset_id='refed',
        display_name='REFED-dataset',
        default_root='data/REFED-dataset',
        aliases=('refed-dataset', 'real-time dynamic labeled', 'emotion_refed'),
        supported_modalities=('eeg', 'fnirs', 'both'),
        eeg_sample_rate_hz=1000.0,
        fnirs_sample_rate_hz=47.62,
        eeg_channels=64,
        fnirs_channels=51,
        default_task='emotion_recognition',
        sync_strategy='continuous_annotation_alignment',
        loader_status='implemented',
        documentation=(
            DocumentationReference(
                title='Original dataset README',
                relative_path='data/REFED-dataset/README.md',
                kind='markdown',
            ),
            DocumentationReference(
                title='Project dataset summary',
                relative_path='docs/DATASETS_DESCRIPTION.md',
                kind='markdown',
            ),
        ),
        notes=(
            'EEG and fNIRS are stored per video rather than as a shared marker stream.',
            'Dynamic valence/arousal annotations are aligned by time instead of discrete trial IDs.',
            'fNIRS includes direct optical-domain Abs 780/805/830 channels in addition to HbO/HbR/HbT.',
        ),
        primary_loader='UnifiedPhysiologyWindowDataset',
        loader_contract='unified_physiology_measurement_window_v3',
        loader_interfaces=('unified_physiology',),
    ),
    'visual_cognitive_motivation': DatasetRegistration(
        dataset_id='visual_cognitive_motivation',
        display_name='Visual Cognitive Motivation Study',
        default_root='data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults',
        aliases=(
            'visual_cognitive_motivation_study',
            'visual cognitive motivation',
            'kyushu_visual_cognitive',
        ),
        supported_modalities=('eeg', 'fnirs', 'both'),
        eeg_sample_rate_hz=500.0,
        fnirs_sample_rate_hz=10.0,
        eeg_channels=30,
        fnirs_channels=24,
        default_task='memory_motivation',
        sync_strategy='cross_device_event_reconstruction',
        loader_status='implemented',
        documentation=(
            DocumentationReference(
                title='Original dataset readme',
                relative_path='data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults/readme.txt',
                kind='text',
            ),
            DocumentationReference(
                title='Project dataset summary',
                relative_path='docs/DATASETS_DESCRIPTION.md',
                kind='markdown',
            ),
        ),
        notes=(
            'EEG is available both as raw EDF and preprocessed epoched MAT files.',
            'fNIRS is stored as per-part/per-probe Oxy/Deoxy CSV files and must be aligned with reconstructed events.',
            'The checked CSV exports do not include direct optical-domain channels, but preserve ETG-7100 695/830 nm metadata in their headers.',
        ),
        primary_loader='UnifiedPhysiologyWindowDataset',
        loader_contract='unified_physiology_measurement_window_v3',
        loader_interfaces=('unified_physiology',),
    ),
    'simultaneous_eeg_nirs': DatasetRegistration(
        dataset_id='simultaneous_eeg_nirs',
        display_name='Simultaneous EEG&NIRS',
        default_root='data/Simultaneous EEG&NIRS',
        aliases=(
            'simultaneous eeg&nirs',
            'simultaneous_eeg&nirs',
            'scientific_data_cognitive',
            'cognitive_tasks_eeg_nirs',
        ),
        supported_modalities=('eeg', 'fnirs', 'both'),
        eeg_sample_rate_hz=200.0,
        fnirs_sample_rate_hz=10.0,
        eeg_channels=30,
        fnirs_channels=36,
        default_task='nback',
        sync_strategy='shared_parallel_port_markers',
        loader_status='implemented',
        documentation=(
            DocumentationReference(
                title='Original MATLAB description',
                relative_path='data/Simultaneous EEG&NIRS/Dataset description_MATLAB.pdf',
                kind='pdf',
            ),
            DocumentationReference(
                title='Original BrainVision/NIRx description',
                relative_path='data/Simultaneous EEG&NIRS/Dataset description_BrainVision and NIRx.pdf',
                kind='pdf',
            ),
            DocumentationReference(
                title='Project dataset summary',
                relative_path='docs/DATASETS_DESCRIPTION.md',
                kind='markdown',
            ),
        ),
        notes=(
            'Task files are stored separately for n-back, DSR, and WG, with three sessions concatenated per task.',
            'EEG uses BBCI-like cnt/mrk structs while fNIRS stores oxy/deoxy under nested fields.',
            'The checked MATLAB export does not include direct optical-domain channels, and the source wavelengths are not exposed in the exported fields.',
            'Training-ready loaders now cover single-modality windows plus task-dependent multimodal segmentation for n-back and WG.',
            'DSR uses EEG-native Go/No-go labels projected from paired block anchors; unstable alignment is excluded.',
        ),
        primary_loader='UnifiedPhysiologyWindowDataset',
        loader_contract='unified_physiology_measurement_window_v3',
        loader_interfaces=('unified_physiology', 'legacy_multimodal', 'continuous_visualization'),
    ),
}


def _normalize_dataset_key(value: str) -> str:
    return value.strip().lower().replace('\\', '/').replace('_', ' ').replace('-', ' ')


DATASET_ALIAS_MAP: Dict[str, str] = {}
for registration in REGISTERED_DATASETS.values():
    keys = {
        registration.dataset_id,
        registration.display_name,
        registration.default_root,
        Path(registration.default_root).name,
        *registration.aliases,
    }
    for key in keys:
        DATASET_ALIAS_MAP[_normalize_dataset_key(str(key))] = registration.dataset_id


def list_registered_datasets() -> List[DatasetRegistration]:
    return list(REGISTERED_DATASETS.values())


def list_raw_datasets() -> List[DatasetRegistration]:
    """Return the four original datasets, excluding derived target caches."""
    return [registration for registration in REGISTERED_DATASETS.values() if registration.resource_kind == 'raw_dataset']


def get_dataset_registration(dataset_id: str) -> DatasetRegistration:
    canonical_id = DATASET_ALIAS_MAP.get(_normalize_dataset_key(dataset_id), dataset_id)
    if canonical_id not in REGISTERED_DATASETS:
        raise KeyError(f'Unknown dataset: {dataset_id}')
    return REGISTERED_DATASETS[canonical_id]


def infer_dataset_id_from_root(data_root: str) -> Optional[str]:
    normalized_root = _normalize_dataset_key(data_root)
    for registration in REGISTERED_DATASETS.values():
        root_name = _normalize_dataset_key(Path(registration.default_root).name)
        default_root = _normalize_dataset_key(registration.default_root)
        if root_name in normalized_root or default_root in normalized_root:
            return registration.dataset_id
    return None


def resolve_dataset_id(data_cfg: Mapping[str, Any]) -> str:
    explicit = data_cfg.get('dataset')
    if isinstance(explicit, str) and explicit.strip() and explicit.strip().upper() != 'TBD':
        return get_dataset_registration(explicit).dataset_id

    sources = data_cfg.get('sources')
    if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)) and len(sources) > 0:
        return 'multi_source'

    data_root = data_cfg.get('data_root') or data_cfg.get('root')
    if isinstance(data_root, str) and data_root.strip():
        inferred = infer_dataset_id_from_root(data_root)
        if inferred is not None:
            return inferred

    raise KeyError('Unable to resolve dataset id from config; set data.dataset explicitly.')


def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def normalize_split_config(data_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(data_cfg.get('split'), Mapping):
        return dict(data_cfg['split'])

    split: Dict[str, Any] = {}
    for key in ('train_subjects', 'val_subjects', 'test_subjects', 'train', 'val', 'test'):
        if key in data_cfg:
            split[key] = data_cfg[key]
    return split


_FNIRS_LEGACY_PREPROCESSING_KEYS = (
    'highpass',
    'lowpass',
    'low_hz',
    'high_hz',
    'target_sampling_rate',
    'resample_rate',
    'sampling_rate',
    'sample_rate',
    'measurement_standardization',
)


def resolve_modality_preprocessing(data_cfg: Mapping[str, Any], modality: str) -> Dict[str, Any]:
    normalized_modality = str(modality).strip().lower()
    if normalized_modality not in ('eeg', 'fnirs'):
        raise ValueError(f'Unsupported modality for preprocessing resolution: {modality!r}')

    explicit_key = f'{normalized_modality}_preprocessing'
    explicit_cfg = data_cfg.get(explicit_key)
    if isinstance(explicit_cfg, Mapping):
        return dict(explicit_cfg)

    legacy_cfg = data_cfg.get('preprocessing')
    if not isinstance(legacy_cfg, Mapping):
        return {}

    if normalized_modality == 'eeg':
        return dict(legacy_cfg)

    return {
        key: value
        for key, value in legacy_cfg.items()
        if key in _FNIRS_LEGACY_PREPROCESSING_KEYS
    }


def normalize_data_config(data_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(data_cfg)
    if 'data_root' not in normalized and 'root' in normalized:
        normalized['data_root'] = normalized['root']

    dataset_id = resolve_dataset_id(normalized)
    if dataset_id == 'multi_source':
        normalized['dataset'] = dataset_id
        normalized.setdefault('dataset_params', {})
        normalized['split'] = normalize_split_config(normalized)
        normalized['eeg_preprocessing'] = resolve_modality_preprocessing(normalized, 'eeg')
        normalized['fnirs_preprocessing'] = resolve_modality_preprocessing(normalized, 'fnirs')
        normalized['dataset_registry'] = {
            'dataset_id': 'multi_source',
            'display_name': 'Multi-source multimodal mix',
            'data_root': normalized.get('data_root'),
            'default_root': None,
            'supported_modalities': ['both'],
            'eeg_sample_rate_hz': None,
            'fnirs_sample_rate_hz': None,
            'eeg_channels': None,
            'fnirs_channels': None,
            'default_task': None,
            'sync_strategy': 'source_defined',
            'loader_status': 'implemented',
            'documentation': [],
            'notes': ['This config combines multiple registered datasets/tasks through data.sources.'],
        }
        return normalized

    registration = get_dataset_registration(dataset_id)
    normalized['dataset'] = dataset_id
    normalized.setdefault('data_root', registration.default_root)
    normalized.setdefault('root', normalized['data_root'])
    normalized.setdefault('task', registration.default_task)
    normalized.setdefault('dataset_params', {})
    normalized['split'] = normalize_split_config(normalized)
    normalized['eeg_preprocessing'] = resolve_modality_preprocessing(normalized, 'eeg')
    normalized['fnirs_preprocessing'] = resolve_modality_preprocessing(normalized, 'fnirs')
    default_fnirs_standardization = default_standardization_config(dataset_id)
    if default_fnirs_standardization is not None:
        normalized['fnirs_preprocessing'].setdefault(
            'measurement_standardization',
            default_fnirs_standardization,
        )
    normalized['dataset_registry'] = registration.runtime_metadata(normalized['data_root'])
    return normalized


def normalize_experiment_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(config)
    data_cfg = normalize_data_config(normalized.get('data', {}))

    top_level_modality = normalized.get('modality')
    if 'modality' not in data_cfg and isinstance(top_level_modality, str):
        data_cfg['modality'] = top_level_modality
    data_cfg.setdefault('modality', 'eeg')
    normalized.setdefault('modality', data_cfg['modality'])
    normalized['data'] = data_cfg
    return normalized


def _resolve_config_path(
    config_path: Path,
    configs_dir: Path,
    current_config_path: Optional[Path] = None,
) -> Path:
    candidates: List[Path] = []
    if config_path.is_absolute():
        candidates.append(config_path)
    else:
        if current_config_path is not None:
            candidates.append(current_config_path.parent / config_path)
        candidates.append(configs_dir / config_path)
        candidates.append(configs_dir / 'downstream' / config_path)
        candidates.append(Path.cwd() / config_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f'Config not found: {config_path}')


def _load_config_recursive(config_path: Path, configs_dir: Path) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}

    if '_base_' not in config:
        return config

    base_reference = Path(config.pop('_base_'))
    base_path = _resolve_config_path(base_reference, configs_dir=configs_dir, current_config_path=config_path)
    base_config = _load_config_recursive(base_path, configs_dir=configs_dir)
    return deep_merge_dicts(base_config, config)


def load_experiment_config(
    config_path: str | Path,
    configs_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    configs_root = Path(configs_dir) if configs_dir is not None else DEFAULT_CONFIGS_DIR
    resolved_path = _resolve_config_path(Path(config_path), configs_dir=configs_root)
    merged_config = _load_config_recursive(resolved_path, configs_dir=configs_root)
    return normalize_experiment_config(merged_config)


def dataset_loader_is_implemented(dataset_id: str, interface: str | None = None) -> bool:
    registration = get_dataset_registration(dataset_id)
    if registration.loader_status != 'implemented':
        return False
    if interface is None:
        return True
    return interface in registration.loader_interfaces


def require_dataset_loader(dataset_id: str, interface: str | None = None) -> None:
    registration = get_dataset_registration(dataset_id)
    if not dataset_loader_is_implemented(dataset_id, interface=interface):
        interface_note = '' if interface is None else f" for interface {interface!r}"
        raise NotImplementedError(
            f"Dataset '{registration.dataset_id}' is registered, but its loader adapter{interface_note} is not implemented."
        )
