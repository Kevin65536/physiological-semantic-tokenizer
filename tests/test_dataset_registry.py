import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src.data.factory import create_configured_multimodal_dataloaders
from src.data.registry import (
    dataset_loader_is_implemented,
    get_dataset_registration,
    list_raw_datasets,
    load_experiment_config,
    normalize_data_config,
    require_dataset_loader,
)
from src.data.simultaneous_eeg_nirs_dataset import classify_alignment_pattern, detect_offset_blocks
from src.data.validation import build_dataset_validation_plan


class DatasetRegistryTests(unittest.TestCase):
    def test_raw_dataset_listing_excludes_croce_derived_target_cache(self):
        self.assertEqual(len(list_raw_datasets()), 4)
        self.assertNotIn('croce_local_cache', {item.dataset_id for item in list_raw_datasets()})
        self.assertEqual(get_dataset_registration('croce_local_cache').resource_kind, 'derived_supervision_cache')

    def test_all_four_raw_datasets_use_the_unified_primary_loader(self):
        for registration in list_raw_datasets():
            self.assertEqual(registration.loader_status, 'implemented')
            self.assertEqual(registration.primary_loader, 'UnifiedPhysiologyWindowDataset')
            self.assertEqual(registration.loader_contract, 'unified_physiology_window_v2')
            self.assertTrue(dataset_loader_is_implemented(registration.dataset_id, 'unified_physiology'))

    def test_loader_interfaces_do_not_overstate_legacy_visualization_support(self):
        self.assertFalse(dataset_loader_is_implemented('refed', 'continuous_visualization'))
        with self.assertRaises(NotImplementedError):
            require_dataset_loader('visual_cognitive_motivation', 'continuous_visualization')

    def test_single_trial_registry_uses_v4_line_clean_no_bad_mask_branch(self):
        registration = get_dataset_registration('eeg_fnirs_single_trial')
        self.assertEqual(registration.default_eeg_signal_branch, 'single_trial_eeg_artifact_clean_v4')
        self.assertEqual(registration.eeg_artifact_status, 'artifact_clean_v4_line_clean_no_bad_mask')
        runtime = registration.runtime_metadata(registration.default_root)
        self.assertEqual(runtime['default_eeg_signal_branch'], 'single_trial_eeg_artifact_clean_v4')

    @staticmethod
    def _write_multimodal_config(root: Path) -> Path:
        base = root / 'base.yaml'
        base.write_text(
            """data:
  data_root: data/EEG+NIRS Single-Trial
  eeg_preprocessing:
    bandpass: [0.5, 45]
  fnirs_preprocessing:
    lowpass: 0.2
""",
            encoding='utf-8',
        )
        child = root / 'multimodal.yaml'
        child.write_text('_base_: ./base.yaml\n', encoding='utf-8')
        return child

    def test_infers_single_trial_from_root(self):
        normalized = normalize_data_config({
            'data_root': 'data/EEG+NIRS Single-Trial',
            'modality': 'eeg',
        })
        self.assertEqual(normalized['dataset'], 'eeg_fnirs_single_trial')
        self.assertEqual(normalized['dataset_registry']['sync_strategy'], 'shared_parallel_port_markers')
        standardization = normalized['fnirs_preprocessing']['measurement_standardization']
        self.assertTrue(standardization['enabled'])
        self.assertEqual(standardization['signal_key'], 'wavelength_pair')
        self.assertEqual(
            normalized['dataset_registry']['fnirs_measurement_contracts']['wavelength_pair']['native_unit'],
            'V',
        )

    def test_simultaneous_default_contract_keeps_declared_concentration_unit(self):
        normalized = normalize_data_config({
            'dataset': 'simultaneous_eeg_nirs',
            'data_root': 'data/Simultaneous EEG&NIRS',
        })
        self.assertEqual(
            normalized['dataset_registry']['fnirs_measurement_contracts']['oxy_deoxy']['native_unit'],
            'mmol/L',
        )
        self.assertEqual(
            normalized['fnirs_preprocessing']['measurement_standardization']['signal_key'],
            'oxy_deoxy',
        )

    def test_explicit_standardization_disable_is_respected(self):
        normalized = normalize_data_config({
            'dataset': 'eeg_fnirs_single_trial',
            'fnirs_preprocessing': {
                'measurement_standardization': {'enabled': False},
            },
        })
        self.assertFalse(normalized['fnirs_preprocessing']['measurement_standardization']['enabled'])

    def test_load_experiment_config_normalizes_shared_config(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_experiment_config(self._write_multimodal_config(root), configs_dir=root)
        self.assertEqual(config['data']['dataset'], 'eeg_fnirs_single_trial')
        self.assertEqual(config['data']['data_root'], 'data/EEG+NIRS Single-Trial')
        self.assertIn('dataset_registry', config['data'])

    def test_load_experiment_config_exposes_modality_specific_preprocessing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_experiment_config(self._write_multimodal_config(root), configs_dir=root)
        self.assertEqual(config['data']['eeg_preprocessing']['bandpass'], [0.5, 45])
        self.assertEqual(config['data']['fnirs_preprocessing']['lowpass'], 0.2)

    def test_multimodal_factory_projects_legacy_preprocessing_by_modality(self):
        config = {
            'data': {
                'dataset': 'eeg_fnirs_single_trial',
                'data_root': 'data/EEG+NIRS Single-Trial',
                'task': 'motor_imagery',
                'window': {'duration_s': 10.0, 'offset_ms': 0.0},
                'split': {
                    'train_subjects': [1],
                    'val_subjects': [2],
                    'test_subjects': [3],
                },
                'preprocessing': {
                    'bandpass': [0.5, 45],
                    'lowpass': 0.1,
                    'resample_rate': 200,
                },
                'exclude_eog': True,
                'hbo_only': True,
                'hbr_only': False,
                'num_workers': 0,
            },
            'training': {'batch_size': 2},
        }

        with patch('src.data.factory.create_single_trial_dataloaders', return_value={'train': None, 'val': None, 'test': None}) as mocked:
            create_configured_multimodal_dataloaders(config)

        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs['eeg_preprocessing']['bandpass'], [0.5, 45])
        self.assertEqual(kwargs['fnirs_preprocessing']['lowpass'], 0.1)
        self.assertEqual(kwargs['fnirs_preprocessing']['resample_rate'], 200)
        self.assertNotIn('bandpass', kwargs['fnirs_preprocessing'])

    def test_load_experiment_config_resolves_downstream_base(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            downstream = root / 'downstream'
            downstream.mkdir()
            config_path = self._write_multimodal_config(downstream)
            with config_path.open('a', encoding='utf-8') as handle:
                handle.write('data:\n  task: motor_imagery\n')
            config = load_experiment_config(config_path.name, configs_dir=root)
        self.assertEqual(config['data']['dataset'], 'eeg_fnirs_single_trial')
        self.assertEqual(config['data']['data_root'], 'data/EEG+NIRS Single-Trial')
        self.assertEqual(config['data']['task'], 'motor_imagery')

    def test_normalize_data_config_accepts_multi_source(self):
        normalized = normalize_data_config({
            'modality': 'both',
            'sources': [
                {
                    'dataset': 'eeg_fnirs_single_trial',
                    'data_root': 'data/EEG+NIRS Single-Trial',
                    'task': 'motor_imagery',
                },
                {
                    'dataset': 'simultaneous_eeg_nirs',
                    'data_root': 'data/Simultaneous EEG&NIRS',
                    'task': 'nback',
                },
            ],
        })
        self.assertEqual(normalized['dataset'], 'multi_source')
        self.assertEqual(normalized['dataset_registry']['sync_strategy'], 'source_defined')

    def test_refed_plan_uses_annotation_alignment(self):
        plan = build_dataset_validation_plan('refed')
        self.assertEqual(plan['sync_strategy'], 'continuous_annotation_alignment')
        check_ids = {check['check_id'] for check in plan['checks']}
        self.assertIn('record-duration-alignment', check_ids)
        self.assertIn('annotation-resample-check', check_ids)
        self.assertIn('global-visual-alignment', check_ids)
        self.assertIn('local-visual-alignment', check_ids)

    def test_visual_plan_uses_reconstruction_checks(self):
        plan = build_dataset_validation_plan('visual_cognitive_motivation')
        check_ids = {check['check_id'] for check in plan['checks']}
        self.assertIn('cross-device-event-reconstruction', check_ids)
        self.assertIn('label-join-consistency', check_ids)

    def test_classifies_stable_fixed_offset_pattern(self):
        residual_ms = np.asarray([1000.0, 1015.0, 995.0, 1005.0])
        blocks = detect_offset_blocks(residual_ms, jump_threshold_ms=20_000.0)
        pattern = classify_alignment_pattern(residual_ms, blocks)
        self.assertEqual(pattern['case'], 'stable_fixed_offset')

    def test_classifies_piecewise_constant_offset_pattern(self):
        residual_ms = np.asarray([1000.0, 1010.0, 995.0, 52000.0, 52015.0, 51990.0])
        blocks = detect_offset_blocks(residual_ms, jump_threshold_ms=20_000.0)
        pattern = classify_alignment_pattern(residual_ms, blocks)
        self.assertEqual(pattern['case'], 'piecewise_constant_offset')
        self.assertEqual(pattern['num_blocks'], 2)


if __name__ == '__main__':
    unittest.main()
