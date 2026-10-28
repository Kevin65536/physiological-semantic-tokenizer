"""Synthetic tests for the new public target boundary and exact decomposition."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.data import dual_component_tokenizer as data


ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT / 'experiments/configs/physiology_semantic_tokenizer/dual_component_tokenizer_v1.yaml').read_text())


def row(identity='train', region='prefrontal', split='train'):
    return dict(id=identity + '__' + region, trial_id=identity,
        dataset='eeg_fnirs_single_trial', subject=identity, record='session_0',
        split=split, coordinate='eeg_fnirs_single_trial__' + region,
        join_key='eeg_fnirs_single_trial|' + identity + '|session_0',
        eeg_start_s=0., hb_start_s=0., source_identity=[],
        anchor=dict(region=region, hb_pair=0, hb_channel='test',
            eeg_channels=['A', 'B', 'C', 'D', 'E', 'F']))


def prepared(r, multiplier=1.):
    rng = np.random.default_rng(13)
    features = rng.normal(size=(900, 30)) * multiplier
    features -= features[:50].mean(axis=0)
    hb = rng.normal(size=(900, 2)) * multiplier
    hb -= hb[:50].mean(axis=0)
    return dict(eeg=rng.normal(size=(4000, 6)).astype('float32') * multiplier,
        features=features, hb=hb, native_time=np.arange(900) / 10,
        time_s=np.arange(900) * .1,
        eeg_proxy_input=np.array(data.EEG_PROXY_INPUT_SCHEMA), eeg_proxy_rate_hz=np.array(10.),
        fnirs_input=np.array(data.CACHED_FNIRS_INPUT_SCHEMA), identity=np.array(r['id']))


def teacher(r, a):
    u = np.column_stack((a['features'][:, 0], a['hb']))
    return dict(u=u, physical=.75 * u, remaining=.25 * u, identity=np.array(r['id']))


def test_scope_rejection_precedes_readers(tmp_path, monkeypatch):
    cfg = config(); cfg['data']['protected_data'] = 'allowed'
    def forbidden(*args, **kwargs):
        pytest.fail('measured reader was reached before scope rejection')
    monkeypatch.setattr(data, 'load_native_eeg_record', forbidden)
    monkeypatch.setattr(data, 'load_cached_fnirs_record', forbidden)
    with pytest.raises(ValueError, match='exact public parent'):
        list(data.prepare_cached_record(tmp_path, cfg, [row()]))


def test_metadata_uses_complete_inventory_not_old_success_subset(tmp_path, monkeypatch):
    cfg = config()
    parent = tmp_path / data.PARENT; parent.mkdir(parents=True)
    retained = dict(schema='functional_tokenizer_inventory_v1', rows=[row(), row('test', split='test')],
        subjects={'eeg_fnirs_single_trial': {'train': ['train'], 'test': ['test']}})
    (parent / 'inventory.json').write_text(json.dumps(retained))
    current = dict(retained, excluded=[], measured_arrays_read=0)
    monkeypatch.setattr(data, 'inventory', lambda *args: current)
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('metadata opened an array'))
    result = data.metadata_inventory(tmp_path, cfg)
    assert result['planned_trials'] == 2
    assert result['rows'] == retained['rows']
    assert not result['parent_teacher_success_filter']
    changed = deepcopy(current); changed['rows'][0]['split'] = 'test'
    monkeypatch.setattr(data, 'inventory', lambda *args: changed)
    with pytest.raises(ValueError, match='differ'):
        data.metadata_inventory(tmp_path, cfg)


@pytest.mark.parametrize('hb_start', [0., 108.80000000000001, 108.84])
def test_prepare_new_10hz_features_and_local_input_support(tmp_path, monkeypatch, hb_start):
    r = row(); names = r['anchor']['eeg_channels']
    r['hb_start_s'] = hb_start
    t = np.arange(18000) / 200
    e = np.column_stack([np.sin(2 * np.pi * (8 + k) * t) for k in range(6)])
    native = SimpleNamespace(values=e, sample_rate_hz=200, native_unit='uV',
        unit_evidence='synthetic fixture', channel_names=names, source_path='synthetic')
    record = SimpleNamespace(manifest=dict(schema='clean_eeg_fnirs_cache_v2',
        processing_schema='physiology_measurement_alignment_v5', motion_method='none',
        measurement=dict(eeg_channel_names=names),
        homer2_aligned_contract=dict(quality=dict(bandpass=dict(status='applied', low_hz=.01, high_hz=.2)))))
    cached = dict(time_s=np.arange(3000) / 10, values=np.sin(np.arange(3000)[:, None, None] / 30) * np.ones((1, 1, 2)),
        channel_names=['test_HbO', 'test_HbR'], provenance={'schema': 'synthetic'})
    monkeypatch.setattr(data, '_registered_record', lambda *args: (tmp_path, record))
    monkeypatch.setattr(data, 'load_native_eeg_record', lambda *args: native)
    monkeypatch.setattr(data, 'load_cached_fnirs_record', lambda *args: cached)
    _, arrays, error = next(data.prepare_cached_record(tmp_path, config(), [r]))
    assert error is None
    assert arrays['features'].shape == (900, 30)
    assert arrays['hb'].shape == (900, 2)
    np.testing.assert_array_equal(arrays['eeg'], e[6000:10000].astype('float32'))
    np.testing.assert_allclose(arrays['features'][:50].mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(arrays['hb'][:50].mean(0), 0, atol=1e-12)
    assert 'operator' not in arrays
    provenance = json.loads(str(arrays['provenance']))
    assert provenance['student_Hb_reference_event_s'] == [-30, -25]
    assert len(arrays['native_time']) == 900
    if hb_start == 108.80000000000001:
        assert arrays['native_time'][0] == 0.
        assert arrays['native_time'][-1] < 90


def test_prepare_real_missing_cached_bin_remains_failure(tmp_path, monkeypatch):
    r = row(); names = r['anchor']['eeg_channels']
    native = SimpleNamespace(values=np.random.default_rng(19).normal(size=(18000, 6)), sample_rate_hz=200,
        native_unit='uV', unit_evidence='synthetic fixture', channel_names=names, source_path='synthetic')
    record = SimpleNamespace(manifest=dict(schema='clean_eeg_fnirs_cache_v2',
        processing_schema='physiology_measurement_alignment_v5', motion_method='none',
        measurement=dict(eeg_channel_names=names),
        homer2_aligned_contract=dict(quality=dict(bandpass=dict(status='applied', low_hz=.01, high_hz=.2)))))
    times = np.delete(np.arange(900) / 10, 100)
    cached = dict(time_s=times, values=np.ones((899, 1, 2)),
        channel_names=['test_HbO', 'test_HbR'], provenance={'schema': 'synthetic'})
    monkeypatch.setattr(data, '_registered_record', lambda *args: (tmp_path, record))
    monkeypatch.setattr(data, 'load_native_eeg_record', lambda *args: native)
    monkeypatch.setattr(data, 'load_cached_fnirs_record', lambda *args: cached)
    _, arrays, error = next(data.prepare_cached_record(tmp_path, config(), [r]))
    assert arrays is None
    assert error == 'every 100 ms bin requires recorded cached Hb support'


def test_training_coordinate_never_reads_validation_or_test():
    rows = [row(), row('test', split='test')]
    calls = []
    def read(r):
        calls.append(r['id'])
        assert r['split'] == 'train'
        return prepared(r)
    coordinates = data.fit_training_coordinates(rows, read)
    assert calls == [rows[0]['id']]
    assert coordinates[rows[0]['coordinate']]['training_subjects'] == ['train']
    assert coordinates[rows[0]['coordinate']]['eeg_proxy_rate_hz'] == 10


def test_pack_common_training_scale_preserves_sum_and_failure_denominator(tmp_path):
    regions = ('prefrontal', 'left_motor', 'right_motor')
    rows = [row(identity, region, split) for identity, split in
        [('train', 'train'), ('test', 'test'), ('failed', 'validation')] for region in regions]
    arrays = {r['id']: prepared(r, 1000. if r['split'] == 'test' else 1.) for r in rows}
    targets = {r['id']: teacher(r, arrays[r['id']]) for r in rows}
    targets[rows[-1]['id']] = None
    coordinates = {r['coordinate']: dict(eeg_mean=0., eeg_scale=1., hb_factor=1.) for r in rows}
    manifest = data.pack_training_cache(config(), rows, coordinates,
        lambda r: arrays[r['id']], lambda r: targets[r['id']], tmp_path)
    assert manifest['planned_trials'] == 3 and manifest['retained_trials'] == 2
    assert manifest['failed_trials'] == ['failed']
    assert manifest['target_units'] == 'training_observation_SD'
    for r in rows[:3]:
        expected = targets[r['id']]['u'][300:500].std(0)
        np.testing.assert_allclose(manifest['observation_training_sd'][r['coordinate']], expected)
    for modality, dimension in [('eeg', 20), ('hb', 40)]:
        values = {k: np.load(tmp_path / manifest['arrays'][modality + '_' + k]) for k in ['u', 's', 'e']}
        assert values['u'].shape == (6, 10, dimension)
        np.testing.assert_allclose(values['u'], values['s'] + values['e'], rtol=1e-6, atol=1e-6)
    assert np.load(tmp_path / 'eeg_input.npy').shape == (6, 10, 2400)
    assert np.load(tmp_path / 'hb_input.npy').shape == (6, 10, 40)


def test_teacher_fixed_tak_stays_in_observation_space(monkeypatch):
    import src.inference.shared_driver_reconstruction as implementation
    r = row(); a = prepared(r)
    co = dict(pc=np.r_[1., np.zeros(29)], eeg_factor=.025, hb_factor=.01,
        sd=np.array([.025, .01, .01]), eeg_proxy_input=data.EEG_PROXY_INPUT_SCHEMA)
    cfg = yaml.safe_load((ROOT / 'experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_10hz_v2.yaml').read_text())
    def fit(observations, parameters, dt, **kwargs):
        assert dt == .1 and kwargs['hemodynamics'] == 'tak'
        assert kwargs['parameter_names'] == []
        assert kwargs['data_weight'] == .4
        assert not kwargs['visible'][0, -1].any()
        return dict(status='converged', converged=True, prediction=np.zeros_like(observations),
            objective=.1,
            driver=np.zeros((1, 900)), initial_state=np.array([[0, 1, 1, 1, 1]]),
            states=np.ones((1, 900, 6)), tau=2., kappa=.64)
    monkeypatch.setattr(implementation, 'fit_nonlinear_shared_parameters', fit)
    monkeypatch.setattr(implementation, 'nonlinear_driver_forward', lambda *args, **kwargs:
        dict(canonical_prediction=np.zeros((900, 3))))
    result, status = data.produce_teacher(a, co, cfg)
    assert status['accepted'] and status['teacher_arm'] == 'tak_fixed'
    np.testing.assert_array_equal(result['physical'], 0)
    np.testing.assert_array_equal(result['remaining'], result['u'])
    assert result['u'].shape == (900, 3)
    invalid = dict(a, eeg_proxy_input=np.array('old_4Hz'))
    with pytest.raises(ValueError, match='versioned'):
        data.produce_teacher(invalid, co, cfg)


def test_alternative_requires_distinct_whole_trajectory_within_objective_budget(monkeypatch):
    r = row(); a = prepared(r)
    reference = dict(teacher_arm=np.array('tak_fixed'), objective=np.array(10.),
        initial_state=np.array([0., 1., 1., 1., 1.]), driver=np.zeros(900),
        physical=np.zeros((900, 3)))
    calls = []
    def produce(*args, warm_start=None, **kwargs):
        calls.append(warm_start)
        assert warm_start['initial_state'][2] == warm_start['initial_state'][3]
        assert warm_start['driver'].shape == (900,)
        candidate = dict(objective=np.array(10.2), physical=np.ones((900, 3)) * .001)
        return candidate, dict(accepted=True, status='completed')
    monkeypatch.setattr(data, 'produce_teacher', produce)
    settings = dict(offsets=[-.25, .25], maximum_objective_fraction=.05)
    payload, status = data.produce_alternative_teacher(a, dict(sd=np.ones(3)), {}, reference, settings)
    assert payload is not None and status['accepted']
    assert len(calls) == 2 and len(status['witnesses']) == 2
    np.testing.assert_allclose(status['objective_fraction'], .02)
    reference['physical'] = payload['physical']
    payload, status = data.produce_alternative_teacher(a, dict(sd=np.ones(3)), {}, reference, settings)
    assert payload is None and status['status'] == 'unavailable'
    assert all(not witness['distinct'] for witness in status['witnesses'])
