"""Cache-only Hb inputs, units, support and real preparation boundaries."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import yaml

from src.data.clean_physiology_cache import CleanCacheRecord, CleanPhysiologyCacheIndex
from src.data import unified_physiology as unified


@pytest.fixture
def cached_record(tmp_path):
    folder = tmp_path / 'cache' / 'record'
    folder.mkdir(parents=True)
    t = np.arange(1000) / 10.
    values = np.column_stack((np.sin(t / 5), .3 * np.cos(t / 7),
                             .2 * np.sin(t / 6), -.1 * np.cos(t / 8))).astype('float64')
    np.save(folder / 'fnirs.npy', values)
    np.save(folder / 'supported.npy', np.ones(4, bool))
    manifest = dict(schema='clean_eeg_fnirs_cache_v2', dataset_id='eeg_fnirs_single_trial',
        processing_schema='physiology_measurement_alignment_v5', motion_method='none',
        storage='measurement_npy_record_v1', arrays=dict(fnirs='fnirs.npy', fnirs_supported_channels='supported.npy'),
        array_shapes=dict(fnirs=list(values.shape)),
        source_files=[dict(path='data/EEG+NIRS Single-Trial/missing_original/cnt.mat', sha256='retained-source-identity')],
        native_contract=dict(native_unit='V'),
        homer2_aligned_contract=dict(
            alignment_state=dict(schema='physiology_measurement_alignment_v5', parameters=dict(motion_method='none')),
            quality=dict(bandpass=dict(status='applied', low_hz=.01, high_hz=.2, order=3.))),
        measurement=dict(fnirs_channel_names=['p0_HbO', 'p0_HbR', 'p1_HbO', 'p1_HbR'],
            fnirs_preprocessing_state=dict(canonical_sample_rate_hz=10., canonical_unit='relative_Hb')))
    record = CleanCacheRecord('eeg_fnirs_single_trial', 'subject 01', 'session_00',
        'subject_01', 'session_00', 'homer2_wavelength_pair',
        'eeg_fnirs_single_trial|subject_01|session_00', 10., folder, manifest)
    return record, values


def forbid_raw(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('raw Hb, MBLL or a second bandpass must not run')
    monkeypatch.setattr(unified, 'load_native_fnirs_record', forbidden)
    monkeypatch.setattr(unified, 'loadmat', forbidden)
    from src.data import homer2_preprocessing as hp
    monkeypatch.setattr(hp, 'modified_beer_lambert', forbidden)
    monkeypatch.setattr(hp, 'bandpass_fnirs', forbidden)


def test_cached_reader_preserves_exact_values_units_and_support(cached_record, tmp_path, monkeypatch):
    record, values = cached_record
    forbid_raw(monkeypatch)
    result = unified.load_cached_fnirs_record(tmp_path, record)
    np.testing.assert_array_equal(result['values'].reshape(values.shape), values)
    np.testing.assert_array_equal(result['time_s'], np.arange(len(values)) / 10.)
    assert result['provenance']['output_unit'] == 'relative_Hb'
    assert result['provenance']['schema'] == unified.CACHED_FNIRS_INPUT_SCHEMA
    assert 'optical_intensity' not in result
    np.save(record.npz_path / 'supported.npy', np.array([True, False, True, True]))
    masked = unified.load_cached_fnirs_record(tmp_path, record)
    assert np.isnan(masked['values'][:, 0, 1]).all()
    assert not masked['valid_mask'][:, 0, 1].any()
    np.testing.assert_array_equal(masked['values'][:, 1], values[:, 2:])


@pytest.mark.parametrize('fault', ['old_schema', 'unfiltered', 'wrong_band', 'wrong_channels', 'wrong_rate'])
def test_cache_contract_fails_before_array_read(cached_record, tmp_path, monkeypatch, fault):
    record, _ = cached_record
    m = record.manifest
    if fault == 'old_schema': m['processing_schema'] = 'physiology_measurement_alignment_v3'
    elif fault == 'unfiltered': m['homer2_aligned_contract']['quality']['bandpass']['status'] = 'skipped_record_too_short'
    elif fault == 'wrong_band': m['homer2_aligned_contract']['quality']['bandpass']['high_hz'] = .5
    elif fault == 'wrong_channels': m['measurement']['fnirs_channel_names'][1] = 'other_HbR'
    else: m['measurement']['fnirs_preprocessing_state']['canonical_sample_rate_hz'] = 47.62
    monkeypatch.setattr(CleanPhysiologyCacheIndex, 'load_record_arrays',
        lambda *_: (_ for _ in ()).throw(AssertionError('invalid provenance must fail before array access')))
    with pytest.raises(ValueError):
        unified.load_cached_fnirs_record(tmp_path, record)


def test_missing_cache_never_falls_back_to_raw(cached_record, tmp_path, monkeypatch):
    record, _ = cached_record
    forbid_raw(monkeypatch)
    with pytest.raises(FileNotFoundError):
        unified.load_cached_fnirs_record(tmp_path, replace(record, npz_path=tmp_path / 'absent'))


def eeg_fixture():
    return SimpleNamespace(values=np.random.default_rng(3).normal(size=(20000, 6)),
        sample_rate_hz=200., channel_names=tuple(f'E{i}' for i in range(6)),
        native_unit='uV', source_path=Path('fixture_EEG'))


def test_output_preparation_uses_cached_target_once(cached_record, tmp_path, monkeypatch):
    from experiments.scripts import evaluate_ssm_output_support as runner
    from src.inference.observation_baselines import disjoint_native_bin_operator
    record, values = cached_record
    forbid_raw(monkeypatch)
    monkeypatch.setattr(unified, 'load_native_eeg_record', lambda *_: eeg_fixture())
    monkeypatch.setattr(runner, 'cache_index', lambda *_: SimpleNamespace(records=[record]))
    cfg = runner.load_config(runner.DEFAULT)
    (tmp_path / 'pilot_summary.json').write_text(json.dumps(dict(software_and_synthetic_pass=True)))
    spec = dict(id='fixture', dataset=record.dataset_id, partition='development', join_key=record.join_key,
        eeg_start_s=0., hb_start_s=.02, anchor=dict(eeg_channels=list(eeg_fixture().channel_names), hb_pair=0))
    result = runner.prepare_record(cfg, tmp_path, spec)
    assert result['fnirs_input'] == unified.CACHED_FNIRS_INPUT_SCHEMA
    assert 'hidden_native_invariance' not in result
    take = (np.arange(len(values)) / 10 >= .02) & (np.arange(len(values)) / 10 < 90.02)
    with np.load(tmp_path / 'prepared' / 'fixture.npz') as a:
        ops = disjoint_native_bin_operator(a['native_time'], steps=360, dt=.25, reference_steps=20)
        np.testing.assert_array_equal(a['hb'], ops['raw'] @ values[take, :2])
    old = yaml.safe_load((runner.CODE_ROOT / 'experiments/configs/physiology_semantic_tokenizer/ssm_output_support_v1.yaml').read_text())
    with pytest.raises(ValueError, match='retired'):
        runner.prepare_record(old, tmp_path, spec)


def test_tokenizer_preparation_needs_no_original_hb_file(cached_record, tmp_path, monkeypatch):
    from src.data import functional_tokenizer as data
    from experiments.scripts import run_functional_tokenizer as runner
    record, values = cached_record
    forbid_raw(monkeypatch)
    monkeypatch.setattr(data, 'load_native_eeg_record', lambda *_: eeg_fixture())
    monkeypatch.setattr(CleanPhysiologyCacheIndex, '__init__', lambda self, *_: setattr(self, 'records', [record]))
    cfg = runner.config(runner.DEFAULT)
    row = dict(join_key=record.join_key, dataset=record.dataset_id, split='train',
        source_identity=record.manifest['source_files'], eeg_start_s=0., hb_start_s=0.,
        anchor=dict(eeg_channels=list(eeg_fixture().channel_names), hb_pair=0), id='fixture', trial_id='trial')
    _, payload, error = next(data.prepare_cached_record(tmp_path, cfg, [row]))
    assert error is None
    assert str(payload['fnirs_input']) == unified.CACHED_FNIRS_INPUT_SCHEMA
    assert json.loads(str(payload['hb_provenance']))['cache_path'] == str(record.npz_path)
    assert payload['hb'].shape == (360, 2)


def test_active_model_preparation_has_no_native_hb_reader():
    """A future call-site addition must not silently reintroduce the bypass."""
    root = Path(__file__).resolve().parents[1]
    for name in ('src/data/functional_tokenizer.py', 'src/data/shared_driver_spatial.py',
                 'experiments/scripts/evaluate_ssm_output_support.py',
                 'experiments/scripts/evaluate_model_compatibility.py'):
        assert 'load_native_fnirs_record' not in (root / name).read_text(), name


def test_native_prepared_arrays_cannot_enter_cached_calibration(tmp_path):
    from src.data.ssm_prepared import require_cached_hb_array
    from experiments.scripts import evaluate_ssm_output_support as runner
    with pytest.raises(ValueError, match='regenerate'):
        require_cached_hb_array(dict(hb=np.zeros((360, 2))))
    require_cached_hb_array(dict(fnirs_input=np.array(unified.CACHED_FNIRS_INPUT_SCHEMA)))
    cfg = runner.load_config(runner.DEFAULT)
    (tmp_path / 'prepared').mkdir()
    (tmp_path / 'tasks' / 'prepare').mkdir(parents=True)
    np.savez(tmp_path / 'prepared' / 'old.npz', eeg=np.ones((360, 30)), hb=np.ones((360, 2)))
    (tmp_path / 'tasks' / 'prepare' / 'old.json').write_text(json.dumps(dict(status='completed')))
    plan = dict(records=[dict(id='old', coordinate='c', partition='development')])
    with pytest.raises(ValueError, match='regenerate'):
        runner.calibrate(cfg, tmp_path, plan)
    assert not (tmp_path / 'coordinates.json').exists()


def test_cached_census_refuses_historical_fit_panels():
    from experiments.scripts import evaluate_model_compatibility as census
    from experiments.scripts import evaluate_model_compatibility_panels as panels
    with pytest.raises(ValueError, match='newly fitted cached parents'):
        panels.panel_config(census.load_config())


def test_refed_cached_contract_preserves_its_dataset_boundary():
    from experiments.scripts import evaluate_ssm_output_support as runner
    cfg = runner.load_config(runner.CODE_ROOT / 'experiments/configs/physiology_semantic_tokenizer/ssm_output_support_refed_cached_v3.yaml')
    assert cfg['data']['datasets'] == ['refed']
    assert cfg['data']['fnirs_input'] == unified.CACHED_FNIRS_INPUT_SCHEMA


def test_dated_output_report_refuses_to_relabel_cached_results(tmp_path):
    from experiments.scripts import evaluate_ssm_output_support as runner
    from experiments.scripts import render_ssm_output_support as report
    (tmp_path / 'resolved_config.yaml').write_text(runner.DEFAULT.read_text())
    with pytest.raises(ValueError, match='dated narrative'):
        report.render(tmp_path, tmp_path / 'new_report.md')
    assert not (tmp_path / 'new_report.md').exists()
