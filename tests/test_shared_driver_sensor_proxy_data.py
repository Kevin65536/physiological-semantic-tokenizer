"""Sensor-proxy support and native feature contracts on synthetic arrays only."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.data import shared_driver_spatial as spatial
from src.data.ssm_prepared import (
    fit_feature_coordinate, pc1_band_contributions, ring_band_features,
    spatial_band_features,
)
from src.inference.observation_baselines import eeg_band_power


def geometry():
    common = dict(coordinate_units='normalized_head_unit', coordinate_system='fixture_head')
    eeg = [dict(common, channel_name=f'E{i:02d}', x=i/30., y=1., z=0.) for i in range(30)]
    hb = [dict(common, channel_name='H0_'+role, base_channel_name='H0', component=role,
               x=0., y=1., z=0., source_index=0, detector_index=0) for role in ('HbO', 'HbR')]
    return eeg, hb


def test_metadata_sets_have_nested_ranges_and_matched_disjoint_controls():
    eeg, hb = geometry()
    support = spatial.select_proxy_channel_sets(eeg, hb, 0)
    sets = support['sets']
    assert sets['nearest3'] == list(range(3))
    assert sets['nearest6'] == list(range(6))
    assert sets['nearest12'] == list(range(12))
    assert sets['sixth_to_seventh'] == [0, 1, 2, 3, 4, 6]
    assert sets['outer6'] == list(range(6, 12))
    assert sets['far6'] == list(range(29, 23, -1))
    assert support == spatial.select_proxy_channel_sets(eeg, hb, 0)
    for name in ('nearest6_far6', *(f'nearest6_random6_{i}' for i in range(3))):
        ids = sets[name]
        assert ids[:6] == sets['nearest6']
        assert len(ids) == len(set(ids)) == 12
        assert not set(ids[6:]) & set(sets['nearest6'])
    # Source ordering and ties cannot silently change the named support.
    shuffled = list(reversed(eeg))
    other = spatial.select_proxy_channel_sets(shuffled, hb, 0)
    assert other['set_channel_names'] == support['set_channel_names']


@pytest.mark.parametrize('mutate', [
    lambda e, h: e[0].update(channel_name='E01'),
    lambda e, h: e[0].update(coordinate_system='other'),
    lambda e, h: h[1].update(x=.2),
    lambda e, h: e[0].update(x=np.nan),
])
def test_unknown_or_inconsistent_geometry_is_rejected(mutate):
    eeg, hb = geometry()
    mutate(eeg, hb)
    with pytest.raises(ValueError):
        spatial.select_proxy_channel_sets(eeg, hb, 0)


def test_band_correction_exactly_nests_original_pc1_and_uses_its_training_scale():
    rng = np.random.default_rng(17)
    training = rng.normal(size=(70, 6, 5))
    hb = rng.normal(size=(70, 2))
    coordinate = fit_feature_coordinate([(training.reshape(70, 30), hb)])
    heldout = rng.normal(size=(23, 6, 5))*37
    frozen = deepcopy(coordinate)
    contributions = pc1_band_contributions(heldout, coordinate)
    baseline = heldout.reshape(23, 30) @ coordinate['pc'] * coordinate['eeg_factor']
    np.testing.assert_allclose(contributions @ np.ones(5), baseline, rtol=2e-14, atol=2e-16)
    np.testing.assert_array_equal(coordinate['pc'], frozen['pc'])
    assert coordinate['eeg_factor'] == frozen['eeg_factor']
    assert coordinate['training_samples'] == 70
    np.testing.assert_array_equal(pc1_band_contributions(heldout.reshape(23, 30), coordinate), contributions)
    # Holding the scale fixed preserves learned amplitude rather than
    # independently standardizing every output back to the original SD.
    np.testing.assert_allclose(contributions @ np.full(5, 2.), 2*baseline, rtol=2e-14, atol=4e-16)


def test_ring_endpoints_recover_six_and_twelve_channel_means_without_refitting():
    x = np.random.default_rng(8).normal(size=(2, 19, 30, 5))
    factor = .031
    inner, outer = list(range(6)), list(range(6, 12))
    six = spatial_band_features(x, inner, eeg_factor=factor)
    twelve = spatial_band_features(x, inner+outer, eeg_factor=factor)
    np.testing.assert_array_equal(ring_band_features(x, inner, outer, alpha=0, eeg_factor=factor), six)
    np.testing.assert_allclose(ring_band_features(x, inner, outer, alpha=.5, eeg_factor=factor), twelve,
                               rtol=2e-14, atol=2e-17)
    quarter = ring_band_features(x, inner, outer, alpha=.25, eeg_factor=factor)
    np.testing.assert_allclose(quarter, .75*six+.25*spatial_band_features(x, outer, eeg_factor=factor))
    with pytest.raises(ValueError):
        spatial_band_features(x, [0, 0])
    with pytest.raises(ValueError):
        ring_band_features(x, inner, inner, alpha=.25)


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    eeg_rows, hb_rows = geometry()
    subject = 'subject_01'
    sources = [dict(path=str(Path('data/EEG+NIRS Single-Trial')/group/'subject 01'/'cnt.mat'))
               for group in ('EEG_01-29', 'NIRS_01-29')]
    record = SimpleNamespace(dataset_id='eeg_fnirs_single_trial', subject='subject 01',
        canonical_subject_id=subject, record_id='session_0', base_record_id='session_0',
        sample_rate_hz=10.,
        join_key='eeg_fnirs_single_trial|subject_01|session_0',
        manifest=dict(source_files=sources, array_shapes=dict(eeg=[18000, 30], fnirs=[900, 2]), measurement=dict(
            eeg_channel_names=[r['channel_name'] for r in eeg_rows],
            fnirs_channel_names=[r['channel_name'] for r in hb_rows])))
    row = dict(dataset=record.dataset_id, subject=subject, record=record.base_record_id,
        join_key=record.join_key, split='train', source_identity=deepcopy(sources),
        window_seconds=30., reference_seconds=5., eeg_start_s=25., hb_start_s=25.,
        eeg_filter_start_s=0., eeg_filter_seconds=90., patch=dict(native_target_index=0, patch_id='fixture_patch'),
        id=record.join_key+'|event=0|fixture_patch', event_id=0, task='motor_imagery', condition='LMI',
        geometry_identity=dict(eeg=eeg_rows, fnirs=hb_rows))
    support = spatial.select_proxy_channel_sets(eeg_rows, hb_rows, 0)
    voltage = np.random.default_rng(4).normal(size=(18000, 30))
    native = SimpleNamespace(values=voltage, channel_names=record.manifest['measurement']['eeg_channel_names'],
        sample_rate_hz=200., native_unit='uV', unit_evidence='synthetic fixture unit',
        reference='fixture linked reference')
    time = np.arange(900)/10.
    values = np.column_stack((np.sin(time/8), np.cos(time/6)))
    cached = dict(values=values[:, None, :], time_s=time, sample_rate_hz=10.,
        channel_names=record.manifest['measurement']['fnirs_channel_names'],
        provenance=dict(schema='unified_cached_fnirs_v1', output_unit='fixture_relative_Hb'))
    reads = []
    def read_eeg(*_): reads.append('native_eeg'); return native
    def read_hb(*_): reads.append('cached_hb'); return cached
    monkeypatch.setattr(spatial, 'load_native_eeg_record', read_eeg)
    monkeypatch.setattr(spatial, 'load_cached_fnirs_record', read_hb)
    import src.data.unified_physiology as unified
    def forbidden(*a, **kw): raise AssertionError('raw Hb must not be read')
    monkeypatch.setattr(unified, 'load_native_fnirs_record', forbidden)
    return SimpleNamespace(root=tmp_path, record=record, row=row, support=support,
                           voltage=voltage, values=values, reads=reads, native=native, cached=cached)


def test_preparation_uses_90s_native_filtering_explicit_crop_and_cached_hb(preparation):
    f = preparation
    bundle = spatial.prepare_sensor_proxy_record(f.root, f.record, f.row, f.support)
    assert f.reads == ['native_eeg', 'cached_hb']
    assert bundle['eeg_logpower'].shape == (300, 30, 5)
    assert bundle['hb'].shape == (300, 2)
    power = eeg_band_power(f.voltage, bands=spatial.DEFAULT_BANDS)
    expected = np.log(power[250:550]); expected -= expected[:50].mean(0)
    np.testing.assert_allclose(bundle['eeg_logpower'], expected, rtol=0, atol=1e-15)
    target = f.values[250:550]-f.values[250:300].mean(0)
    np.testing.assert_allclose(bundle['hb'], target, atol=4e-15)
    np.testing.assert_allclose(bundle['hb_operator'] @ f.values[250:550], target, atol=4e-15)
    assert bundle['eeg_operator'].shape == bundle['hb_operator'].shape == (300, 300)
    np.testing.assert_allclose(bundle['eeg_logpower'][:50].mean(0), 0., atol=2e-14)
    assert bundle['metadata']['final_crop_seconds'] == [25., 55.]
    assert bundle['metadata']['eeg_filtering_support_seconds'] == 90.
    np.testing.assert_array_equal(bundle['time_s'], np.arange(300)/10.)


@pytest.mark.parametrize('change', ['identity', 'context', 'target', 'sets'])
def test_frozen_identity_and_support_fail_before_native_reads(preparation, change):
    f = preparation
    row, support = deepcopy(f.row), deepcopy(f.support)
    if change == 'identity': row['subject'] = 'subject_02'
    elif change == 'context': row['eeg_filter_start_s'] = 40.
    elif change == 'target': row['patch']['native_target_index'] = 1
    else: support['sets']['nearest6'][0] = 19
    with pytest.raises(ValueError):
        spatial.prepare_sensor_proxy_record(f.root, f.record, row, support)
    assert not f.reads


def test_record_group_reuses_preloaded_central_reader_objects(preparation):
    f = preparation
    first = spatial.prepare_sensor_proxy_record(f.root, f.record, f.row, f.support,
        eeg_record=f.native, hb_record=f.cached)
    shifted = deepcopy(f.row); shifted.update(eeg_start_s=35., hb_start_s=35.)
    second = spatial.prepare_sensor_proxy_record(f.root, f.record, shifted, f.support,
        eeg_record=f.native, hb_record=f.cached)
    assert not f.reads
    assert first['metadata']['final_crop_seconds'] == [25., 55.]
    assert second['metadata']['final_crop_seconds'] == [35., 65.]
    assert np.max(abs(first['eeg_logpower']-second['eeg_logpower'])) > .1


def test_eeg_only_donor_never_reads_hb_or_produces_hb_fields(preparation):
    f = preparation
    data = spatial.prepare_sensor_proxy_eeg_record(f.root, f.record, f.row, f.support)
    assert f.reads == ['native_eeg']
    assert not {'hb', 'hb_operator', 'fnirs_input'} & data.keys()
    assert 'hb_provenance' not in data['metadata']
    np.testing.assert_allclose(data['eeg_logpower'][:50].mean(0), 0., atol=2e-14)


def test_donor_plan_uses_canonical_events_only_and_actual_disjoint_filter_support(preparation):
    f = preparation
    record = deepcopy(f.record)
    record.manifest['array_shapes'] = dict(eeg=[80000, 30], fnirs=[4000, 2])
    target = deepcopy(f.row); target['channel_sets'] = f.support
    def event(index, time_s, **metadata):
        return dict(join_key=record.join_key, event_index=index, eeg_time_ms=time_s*1000,
            fnirs_time_ms=time_s*1000, metadata=dict(task='motor_imagery', condition_label='LMI', **metadata))
    events = [event(0, 30), event(1, 80), event(2, 200), event(3, 300)]
    events.append(event(4, 370))  # Complete context clamps to the record boundary.
    plan = spatial.metadata_sensor_proxy_donors([target], [target], {record.join_key: record},
        {record.join_key: events, 'outside_parent': [event(99, 400)]})
    assert not f.reads and plan['measured_arrays_read'] == 0
    assert plan['candidate_count'] == 5
    assert len(plan['donor_rows']) == 1
    donor = plan['donor_rows'][0]
    assert donor['event_id'] == 2 and donor['eeg_start_s'] == 195.
    assert donor['eeg_filter_start_s'] == 165.
    assert plan['wrong_eeg_partners'][target['id']] == donor['id']
    assert donor['donor_only'] and donor['split'] == target['split']
    # The earlier event is 50 s from the target but its 90 s filtering
    # contexts overlap; a target-time-only gap check would choose it wrongly.
    assert plan['availability']['train'] == 1


def test_donor_plan_retains_unavailable_without_subject_or_condition_fallback(preparation):
    f = preparation
    target = deepcopy(f.row); target['channel_sets'] = f.support
    event = dict(join_key=f.record.join_key, event_index=10, eeg_time_ms=30000.,
        fnirs_time_ms=30000., metadata=dict(task='motor_imagery', condition_label='RMI'))
    plan = spatial.metadata_sensor_proxy_donors([target], [target], {f.record.join_key: f.record},
        {f.record.join_key: [event]})
    assert not f.reads
    assert plan['wrong_eeg_partners'] == {target['id']: None}
    assert not plan['donor_rows']
