"""Shared dataset factories for training and visualization entry points."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .croce_local_cache_dataset import (
    PHYSIOLOGY_SEMANTIC_DATA_CONTRACT,
    CroceLocalCacheDataset,
    CrocePhysiologySemanticDataset,
    validate_physiology_semantic_data_config,
)
from .eeg_fnirs_dataset import EEGfNIRSDataset, MultiModalEEGfNIRSDataset, create_dataloaders as create_single_trial_dataloaders
from .physiology_semantic_local import UnifiedPhysiologyLocalViewDataset
from .clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT
from .registry import normalize_data_config, resolve_dataset_id, resolve_modality_preprocessing
from .simultaneous_eeg_nirs_dataset import (
    SimultaneousContinuousDataset,
    SimultaneousEEGfNIRSDataset,
    SimultaneousMultiModalDataset,
    resolve_fnirs_signal,
    resolve_segmentation_mode,
)


def _resolve_dataloader_kwargs(data_cfg: Dict[str, Any], *, is_train: bool) -> Dict[str, Any]:
    dataloader_cfg = data_cfg.get('dataloader', {})
    num_workers = int(data_cfg.get('num_workers', 0))

    kwargs: Dict[str, Any] = {
        'num_workers': num_workers,
        'pin_memory': bool(dataloader_cfg.get('pin_memory', True)),
    }

    if num_workers > 0 and bool(dataloader_cfg.get('persistent_workers', False)):
        kwargs['persistent_workers'] = True

    prefetch_factor = dataloader_cfg.get('prefetch_factor')
    if num_workers > 0 and prefetch_factor is not None:
        kwargs['prefetch_factor'] = int(prefetch_factor)

    if num_workers > 0 and 'in_order' in dataloader_cfg:
        kwargs['in_order'] = bool(dataloader_cfg['in_order'])

    return kwargs


def _resolve_drop_last(data_cfg: Dict[str, Any], *, is_train: bool) -> bool:
    if not is_train:
        return False
    return bool(data_cfg.get('dataloader', {}).get('drop_last', True))


class CombinedMultiModalDataset(Dataset):
    """Concatenate multiple multimodal datasets into one training view.

    EEG channels can be remapped into a shared union by channel name, while fNIRS
    currently assumes a shared channel count across sources.
    """

    def __init__(
        self,
        datasets: List[Tuple[str, Dict[str, Any], Dataset]],
        *,
        eeg_channel_strategy: str = 'union_pad',
        fnirs_channel_strategy: str = 'identity',
    ):
        if not datasets:
            raise ValueError('CombinedMultiModalDataset requires at least one source dataset.')

        self.sources = []
        self.cumulative_sizes: List[int] = []
        total = 0
        for source_name, source_cfg, dataset in datasets:
            self.sources.append({
                'name': source_name,
                'config': dict(source_cfg),
                'dataset': dataset,
                'length': len(dataset),
            })
            total += len(dataset)
            self.cumulative_sizes.append(total)

        self.eeg_channel_strategy = eeg_channel_strategy
        self.fnirs_channel_strategy = fnirs_channel_strategy
        self.eeg_channel_names, self._eeg_source_index_maps = self._build_channel_layout('eeg', eeg_channel_strategy)
        self.fnirs_channel_names, self._fnirs_source_index_maps = self._build_channel_layout('fnirs', fnirs_channel_strategy)
        self.eeg_sample_rate = float(self.sources[0]['dataset'].get_eeg_sample_rate())
        self.fnirs_sample_rate = float(self.sources[0]['dataset'].get_fnirs_sample_rate())

    def _source_channel_names(self, source: Dict[str, Any], modality: str) -> List[str]:
        dataset = source['dataset']
        if modality == 'eeg':
            return list(dataset.get_eeg_channel_names())
        return list(dataset.get_fnirs_channel_names())

    def _build_channel_layout(self, modality: str, strategy: str) -> Tuple[List[str], List[Optional[List[int]]]]:
        per_source_names = [self._source_channel_names(source, modality) for source in self.sources]

        if strategy == 'identity':
            reference = per_source_names[0]
            reference_count = len(reference)
            for names in per_source_names[1:]:
                if len(names) != reference_count:
                    raise ValueError(
                        f'{modality} channel count mismatch across combined sources with identity strategy: '
                        f'{reference_count} vs {len(names)}'
                    )
            return reference, [None] * len(self.sources)

        if strategy != 'union_pad':
            raise ValueError(f'Unsupported {modality} channel strategy: {strategy}')

        ordered = OrderedDict()
        for names in per_source_names:
            for name in names:
                ordered.setdefault(name, len(ordered))
        target_names = list(ordered.keys())
        source_maps: List[List[int]] = []
        for names in per_source_names:
            name_to_index = {name: index for index, name in enumerate(names)}
            source_maps.append([name_to_index.get(name, -1) for name in target_names])
        return target_names, source_maps

    def _resolve_source_index(self, idx: int) -> Tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        source_index = 0
        while idx >= self.cumulative_sizes[source_index]:
            source_index += 1
        previous = 0 if source_index == 0 else self.cumulative_sizes[source_index - 1]
        return source_index, idx - previous

    def _remap_channels(self, tensor: torch.Tensor, source_index: int, modality: str) -> torch.Tensor:
        if modality == 'eeg':
            strategy = self.eeg_channel_strategy
            target_names = self.eeg_channel_names
            index_maps = self._eeg_source_index_maps
        else:
            strategy = self.fnirs_channel_strategy
            target_names = self.fnirs_channel_names
            index_maps = self._fnirs_source_index_maps

        if strategy == 'identity':
            return tensor

        remapped = torch.zeros((len(target_names), tensor.shape[1]), dtype=tensor.dtype)
        source_map = index_maps[source_index]
        for target_index, source_channel_index in enumerate(source_map):
            if source_channel_index >= 0:
                remapped[target_index] = tensor[source_channel_index]
        return remapped

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        source_index, local_index = self._resolve_source_index(idx)
        source = self.sources[source_index]
        item = dict(source['dataset'][local_index])
        item['eeg'] = self._remap_channels(item['eeg'], source_index, 'eeg')
        item['fnirs'] = self._remap_channels(item['fnirs'], source_index, 'fnirs')
        item['source_name'] = source['name']
        item['source_dataset'] = source['config'].get('dataset')
        item['source_task'] = source['config'].get('task')
        return item

    def get_num_eeg_channels(self) -> int:
        return len(self.eeg_channel_names)

    def get_num_fnirs_channels(self) -> int:
        return len(self.fnirs_channel_names)

    def get_eeg_channel_names(self) -> List[str]:
        return list(self.eeg_channel_names)

    def get_fnirs_channel_names(self) -> List[str]:
        return list(self.fnirs_channel_names)

    def get_eeg_sample_rate(self) -> float:
        return self.eeg_sample_rate

    def get_fnirs_sample_rate(self) -> float:
        return self.fnirs_sample_rate

    def describe_sources(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': source['name'],
                'dataset': source['config'].get('dataset'),
                'task': source['config'].get('task'),
                'length': source['length'],
            }
            for source in self.sources
        ]


def resolve_normalization_config(data_cfg: Dict[str, Any]) -> Tuple[bool, str]:
    norm_cfg = data_cfg.get('normalization', {})
    if isinstance(norm_cfg, dict):
        enabled = bool(norm_cfg.get('enabled', data_cfg.get('normalize', True)))
        mode = norm_cfg.get('mode', 'session' if enabled else 'none')
    else:
        enabled = bool(data_cfg.get('normalize', True))
        mode = 'session' if enabled else 'none'

    if not enabled:
        mode = 'none'
    return enabled, mode


def _dataset_params(data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    params = data_cfg.get('dataset_params', {})
    return dict(params) if isinstance(params, dict) else {}


def _resolve_fnirs_signal_from_config(data_cfg: Dict[str, Any]) -> str:
    params = _dataset_params(data_cfg)
    return resolve_fnirs_signal(
        params.get('fnirs_signal', 'oxy'),
        hbo_only=bool(data_cfg.get('hbo_only', True)),
        hbr_only=bool(data_cfg.get('hbr_only', False)),
    )


def create_unimodal_window_dataset(
    data_cfg: Dict[str, Any],
    subject_ids: List[int],
    modality: str,
    *,
    window_samples: int,
    normalize: bool,
    normalization_mode: str,
):
    dataset_id = resolve_dataset_id(data_cfg)
    params = _dataset_params(data_cfg)

    if dataset_id == 'eeg_fnirs_single_trial':
        return EEGfNIRSDataset(
            data_root=data_cfg['data_root'],
            modality=modality,
            subject_ids=subject_ids,
            task=data_cfg.get('task', 'motor_imagery'),
            window_samples=window_samples,
            window_offset_ms=data_cfg['window'].get('offset_ms', 0),
            normalize=normalize,
            normalization_mode=normalization_mode,
            preprocessing=resolve_modality_preprocessing(data_cfg, modality),
            exclude_eog=data_cfg.get('exclude_eog', False),
            hbo_only=data_cfg.get('hbo_only', False),
            hbr_only=data_cfg.get('hbr_only', False),
        )

    if dataset_id == 'simultaneous_eeg_nirs':
        return SimultaneousEEGfNIRSDataset(
            data_root=data_cfg['data_root'],
            modality=modality,
            subject_ids=subject_ids,
            task=data_cfg.get('task', 'nback'),
            window_samples=window_samples,
            window_offset_ms=float(data_cfg['window'].get('offset_ms', 0)),
            normalize=normalize,
            normalization_mode=normalization_mode,
            preprocessing=resolve_modality_preprocessing(data_cfg, modality),
            exclude_eog=data_cfg.get('exclude_eog', True),
            hbo_only=data_cfg.get('hbo_only', True),
            hbr_only=data_cfg.get('hbr_only', False),
            fnirs_signal=_resolve_fnirs_signal_from_config(data_cfg),
            segmentation_mode=params.get('segmentation_mode', 'auto'),
        )

    raise NotImplementedError(f'Unified unimodal dataset factory does not support dataset {dataset_id!r} yet.')


def create_multimodal_window_dataset(
    data_cfg: Dict[str, Any],
    subject_ids: List[int],
    *,
    window_duration_s: float,
    normalize: bool,
    normalization_mode: str,
):
    dataset_id = resolve_dataset_id(data_cfg)
    params = _dataset_params(data_cfg)

    if dataset_id == 'croce_local_cache':
        crop_cfg = data_cfg.get('crop', {})
        is_target_contract = str(data_cfg.get('contract', '')) == PHYSIOLOGY_SEMANTIC_DATA_CONTRACT
        dataset_cls = CrocePhysiologySemanticDataset if is_target_contract else CroceLocalCacheDataset
        extra_kwargs = (
            {'causal_history_s': float(data_cfg.get('teacher', {}).get('causal_history_s', 10.0))}
            if is_target_contract
            else {}
        )
        return dataset_cls(
            cache_sources=data_cfg.get('cache_sources', [data_cfg.get('data_root', 'croce_validation/cache')]),
            subject_ids=subject_ids,
            split=str(data_cfg.get('_split_name', 'train')),
            crop_duration_s=float(data_cfg.get('window', {}).get('duration_s', window_duration_s)),
            eeg_sample_rate_hz=float(data_cfg.get('eeg_sample_rate_hz', 200.0)),
            fnirs_sample_rate_hz=float(data_cfg.get('fnirs_sample_rate_hz', 10.0)),
            train_random_crop=bool(crop_cfg.get('train_random', True)),
            eval_event_offsets_s=crop_cfg.get('eval_event_offsets_s', [-10.0, 0.0, 20.0]),
            seed=int(data_cfg.get('seed', 42)),
            cache_npz_handles=bool(data_cfg.get('cache_npz_handles', params.get('cache_npz_handles', True))),
            max_npz_cache_size=int(data_cfg.get('max_npz_cache_size', params.get('max_npz_cache_size', 128))),
            normalization=data_cfg.get('normalization', {}),
            entry_filters=data_cfg.get('entry_filters', {}),
            **extra_kwargs,
        )

    if dataset_id == 'eeg_fnirs_single_trial':
        return MultiModalEEGfNIRSDataset(
            data_root=data_cfg['data_root'],
            subject_ids=subject_ids,
            task=data_cfg.get('task', 'motor_imagery'),
            window_duration_s=window_duration_s,
            window_offset_ms=float(data_cfg['window'].get('offset_ms', 0)),
            normalize=normalize,
            normalization_mode=normalization_mode,
            eeg_preprocessing=resolve_modality_preprocessing(data_cfg, 'eeg'),
            fnirs_preprocessing=resolve_modality_preprocessing(data_cfg, 'fnirs'),
            exclude_eog=data_cfg.get('exclude_eog', True),
            hbo_only=data_cfg.get('hbo_only', True),
            hbr_only=data_cfg.get('hbr_only', False),
        )

    if dataset_id == 'simultaneous_eeg_nirs':
        return SimultaneousMultiModalDataset(
            data_root=data_cfg['data_root'],
            subject_ids=subject_ids,
            task=data_cfg.get('task', 'wg'),
            window_duration_s=window_duration_s,
            window_offset_ms=float(data_cfg['window'].get('offset_ms', 0)),
            normalize=normalize,
            normalization_mode=normalization_mode,
            eeg_preprocessing=resolve_modality_preprocessing(data_cfg, 'eeg'),
            fnirs_preprocessing=resolve_modality_preprocessing(data_cfg, 'fnirs'),
            exclude_eog=data_cfg.get('exclude_eog', True),
            hbo_only=data_cfg.get('hbo_only', True),
            hbr_only=data_cfg.get('hbr_only', False),
            fnirs_signal=_resolve_fnirs_signal_from_config(data_cfg),
            segmentation_mode=params.get('segmentation_mode', 'auto'),
        )

    raise NotImplementedError(f'Unified multimodal dataset factory does not support dataset {dataset_id!r} yet.')


def _build_multisource_config(source_cfg: Dict[str, Any], parent_data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(parent_data_cfg)
    merged.pop('sources', None)
    merged.pop('multi_source', None)
    merged = {**merged, **dict(source_cfg)}
    return normalize_data_config(merged)


def _build_multisource_dataset(
    data_cfg: Dict[str, Any],
    split_name: str,
    *,
    normalize: bool,
    normalization_mode: str,
) -> CombinedMultiModalDataset:
    sources_cfg = data_cfg.get('sources', [])
    if not isinstance(sources_cfg, list) or not sources_cfg:
        raise ValueError('data.sources must be a non-empty list for multi-source multimodal loading.')

    built_sources: List[Tuple[str, Dict[str, Any], Dataset]] = []
    for source_index, source_cfg in enumerate(sources_cfg):
        source_data_cfg = _build_multisource_config(source_cfg, data_cfg)
        split_cfg = source_data_cfg.get('split', {})
        subject_key = f'{split_name}_subjects'
        subject_ids = split_cfg.get(subject_key)
        if not isinstance(subject_ids, list) or not subject_ids:
            continue
        dataset = create_multimodal_window_dataset(
            source_data_cfg,
            subject_ids,
            window_duration_s=float(source_data_cfg['window']['duration_s']),
            normalize=normalize,
            normalization_mode=normalization_mode,
        )
        source_name = source_cfg.get('name', f'source_{source_index}')
        built_sources.append((str(source_name), source_data_cfg, dataset))

    if not built_sources:
        raise ValueError(f'No source dataset produced any subjects for split {split_name!r}.')

    multi_source_cfg = data_cfg.get('multi_source', {})
    return CombinedMultiModalDataset(
        built_sources,
        eeg_channel_strategy=str(multi_source_cfg.get('eeg_channel_strategy', 'union_pad')),
        fnirs_channel_strategy=str(multi_source_cfg.get('fnirs_channel_strategy', 'identity')),
    )


def create_configured_dataloader(config: Dict[str, Any], split: str) -> DataLoader:
    data_cfg = config['data']
    normalize, normalization_mode = resolve_normalization_config(data_cfg)

    if split == 'train':
        subjects = data_cfg['split']['train_subjects']
        shuffle = True
    elif split == 'val':
        subjects = data_cfg['split']['val_subjects']
        shuffle = False
    else:
        subjects = data_cfg['split']['test_subjects']
        shuffle = False

    dataset = create_unimodal_window_dataset(
        data_cfg,
        subjects,
        data_cfg['modality'],
        window_samples=int(data_cfg['window']['length']),
        normalize=normalize,
        normalization_mode=normalization_mode,
    )

    return DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        num_workers=data_cfg.get('num_workers', 0),
        pin_memory=True,
        drop_last=split == 'train',
    )


def create_configured_multimodal_dataloaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    data_cfg = config['data']
    normalize, normalization_mode = resolve_normalization_config(data_cfg)
    loader_class = str(data_cfg.get('loader_class', ''))
    dataset_id = (
        'unified_physiology_local'
        if loader_class == 'UnifiedPhysiologyLocalViewDataset'
        else resolve_dataset_id(data_cfg)
    )

    if dataset_id == 'unified_physiology_local':
        dataloaders: Dict[str, DataLoader] = {}
        split_cfg = data_cfg.get('split', {})
        split_subjects = {name: set(split_cfg.get(f'{name}_subject_keys', split_cfg.get(name, [])))
                          for name in ('train', 'val', 'test')}
        if any(split_subjects[a] & split_subjects[b] for a,b in (('train','val'),('train','test'),('val','test'))):
            raise ValueError('Measurement factory requires disjoint subject splits')
        coordinate = str(data_cfg.get('output_coordinate', 'measurement'))
        for split_name in ('train', 'val', 'test'):
            subject_keys = split_cfg.get(f'{split_name}_subject_keys', split_cfg.get(split_name, []))
            target_cfg = data_cfg.get('auxiliary_target', {}) or {}
            target_root = target_cfg.get('root')
            required_splits = set(target_cfg.get('required_splits', ()))
            dataset = UnifiedPhysiologyLocalViewDataset(
                cache_root=data_cfg.get('cache_root', DEFAULT_CLEAN_CACHE_ROOT),
                dataset_ids=tuple(data_cfg.get('dataset_ids', ('eeg_fnirs_single_trial',))),
                subject_keys=subject_keys,
                task_namespaces=data_cfg.get('task_namespaces'),
                window_duration_s=float(data_cfg.get('window', {}).get('duration_s', 20.0)),
                window_offset_s=float(data_cfg.get('window', {}).get('offset_s', 0.0)),
                eeg_signal_branch=str(
                    data_cfg.get('eeg_signal_branch', 'single_trial_eeg_artifact_clean_v4')
                ),
                output_coordinate=coordinate,
                local_eeg_channels=int(data_cfg.get('local_view', {}).get('eeg_channels', 6)),
                reject_unknown_labels=bool(data_cfg.get('reject_unknown_labels', True)),
                allow_cross_coordinate_systems=bool(
                    data_cfg.get('local_view', {}).get('allow_cross_coordinate_systems', False)
                ),
                auxiliary_target_root=target_root,
                auxiliary_target_family=target_cfg.get('family'),
                auxiliary_target_version=target_cfg.get('version'),
                require_auxiliary_target=split_name in required_splits,
            )
            is_train = split_name == 'train'
            loader_kwargs = _resolve_dataloader_kwargs(data_cfg, is_train=is_train)
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=config['training']['batch_size'],
                shuffle=is_train,
                drop_last=_resolve_drop_last(data_cfg, is_train=is_train),
                **loader_kwargs,
            )
        return dataloaders

    if dataset_id == 'croce_local_cache':
        if str(data_cfg.get('contract', '')) == PHYSIOLOGY_SEMANTIC_DATA_CONTRACT:
            validate_physiology_semantic_data_config(data_cfg)
        dataloaders: Dict[str, DataLoader] = {}
        force_deterministic = bool(data_cfg.get('crop', {}).get('force_deterministic_all_splits', False))
        for split_name in ('train', 'val', 'test'):
            split_cfg = data_cfg.get('split', {})
            subjects = split_cfg.get(f'{split_name}_subjects', split_cfg.get(split_name, []))
            split_data_cfg = dict(data_cfg)
            split_data_cfg['_split_name'] = 'audit' if force_deterministic else split_name
            dataset = create_multimodal_window_dataset(
                split_data_cfg,
                subjects,
                window_duration_s=float(data_cfg['window']['duration_s']),
                normalize=normalize,
                normalization_mode=normalization_mode,
            )
            is_train_loader = split_name == 'train' and not force_deterministic
            loader_kwargs = _resolve_dataloader_kwargs(data_cfg, is_train=is_train_loader)
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=config['training']['batch_size'],
                shuffle=is_train_loader,
                drop_last=_resolve_drop_last(data_cfg, is_train=is_train_loader),
                **loader_kwargs,
            )
        return dataloaders

    if isinstance(data_cfg.get('sources'), list) and data_cfg.get('sources'):
        dataloaders: Dict[str, DataLoader] = {}
        for split_name in ('train', 'val', 'test'):
            dataset = _build_multisource_dataset(
                data_cfg,
                split_name,
                normalize=normalize,
                normalization_mode=normalization_mode,
            )
            loader_kwargs = _resolve_dataloader_kwargs(data_cfg, is_train=(split_name == 'train'))
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=config['training']['batch_size'],
                shuffle=(split_name == 'train'),
                drop_last=_resolve_drop_last(data_cfg, is_train=(split_name == 'train')),
                **loader_kwargs,
            )
        return dataloaders

    if dataset_id == 'eeg_fnirs_single_trial':
        return create_single_trial_dataloaders(
            data_root=data_cfg['data_root'],
            modality='both',
            task=data_cfg.get('task', 'motor_imagery'),
            train_subjects=data_cfg['split']['train_subjects'],
            val_subjects=data_cfg['split']['val_subjects'],
            test_subjects=data_cfg['split']['test_subjects'],
            window_duration_s=float(data_cfg['window']['duration_s']),
            batch_size=config['training']['batch_size'],
            num_workers=data_cfg.get('num_workers', 0),
            window_offset_ms=float(data_cfg['window'].get('offset_ms', 0)),
            normalize=normalize,
            normalization_mode=normalization_mode,
            eeg_preprocessing=resolve_modality_preprocessing(data_cfg, 'eeg'),
            fnirs_preprocessing=resolve_modality_preprocessing(data_cfg, 'fnirs'),
            exclude_eog=data_cfg.get('exclude_eog', True),
            hbo_only=data_cfg.get('hbo_only', True),
            hbr_only=data_cfg.get('hbr_only', False),
            dataloader_cfg=data_cfg.get('dataloader', {}),
        )

    splits = {
        'train': data_cfg['split']['train_subjects'],
        'val': data_cfg['split']['val_subjects'],
        'test': data_cfg['split']['test_subjects'],
    }
    dataloaders: Dict[str, DataLoader] = {}
    for split_name, subjects in splits.items():
        dataset = create_multimodal_window_dataset(
            data_cfg,
            subjects,
            window_duration_s=float(data_cfg['window']['duration_s']),
            normalize=normalize,
            normalization_mode=normalization_mode,
        )
        loader_kwargs = _resolve_dataloader_kwargs(data_cfg, is_train=(split_name == 'train'))
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=(split_name == 'train'),
            drop_last=_resolve_drop_last(data_cfg, is_train=(split_name == 'train')),
            **loader_kwargs,
        )
    return dataloaders


def create_continuous_visualization_dataset(
    data_cfg: Dict[str, Any],
    modality: str,
    subject_id: int,
    *,
    normalize: bool,
    normalization_mode: str,
):
    dataset_id = resolve_dataset_id(data_cfg)
    params = _dataset_params(data_cfg)

    if dataset_id == 'eeg_fnirs_single_trial':
        window_length = data_cfg.get('window', {}).get('length', 1)
        return EEGfNIRSDataset(
            data_root=data_cfg['data_root'],
            subject_ids=[subject_id],
            task=data_cfg.get('task', 'motor_imagery'),
            modality=modality,
            window_samples=int(window_length),
            window_offset_ms=float(data_cfg.get('window', {}).get('offset_ms', 0)),
            normalize=normalize,
            normalization_mode=normalization_mode,
            preprocessing=resolve_modality_preprocessing(data_cfg, modality),
            exclude_eog=data_cfg.get('exclude_eog', True),
            hbo_only=data_cfg.get('hbo_only', True),
            hbr_only=data_cfg.get('hbr_only', False),
        )

    if dataset_id == 'simultaneous_eeg_nirs':
        visualization_segmentation_mode = params.get('visualization_segmentation_mode', 'auto')
        if visualization_segmentation_mode == 'auto':
            visualization_segmentation_mode = resolve_segmentation_mode(
                data_cfg.get('task', 'nback'),
                'both',
                params.get('segmentation_mode', 'auto'),
            )
        return SimultaneousContinuousDataset(
            data_root=data_cfg['data_root'],
            task=data_cfg.get('task', 'nback'),
            modality=modality,
            subject_ids=[subject_id],
            normalize=normalize,
            normalization_mode=normalization_mode,
            preprocessing=resolve_modality_preprocessing(data_cfg, modality),
            exclude_eog=data_cfg.get('exclude_eog', True),
            fnirs_signal=_resolve_fnirs_signal_from_config(data_cfg),
            segmentation_mode=visualization_segmentation_mode,
        )

    raise NotImplementedError(f'Continuous visualization dataset factory does not support dataset {dataset_id!r} yet.')
