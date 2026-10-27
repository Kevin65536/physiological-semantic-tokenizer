"""Spatial data boundaries exercised only with temporary metadata and arrays."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.data import shared_driver_spatial as spatial
from src.data.clean_physiology_cache import CleanCacheRecord


def geometry():
    common = dict(coordinate_units='normalized_head_unit', coordinate_system='dataset_mnt_pos_3d',
                  source_file='fixture_mnt.mat', position_available=True)
    eeg = [dict(common, channel_name=f'E{i:02d}', base_channel_name=f'E{i:02d}',
                component='eeg', x=.03 * i, y=1. - .02 * i, z=.01 * i)
           for i in range(16)]
    hb = []
    for i in range(5):
        for role in ('HbO', 'HbR'):
            hb.append(dict(common, channel_name=f'H{i}_{role}', base_channel_name=f'H{i}',
                component=role, x=.12 * i, y=1. - .04 * i, z=.02 * i,
                source_index=i + 1, detector_index=i + 1))
    return eeg, hb


@pytest.fixture
def fixture_data(tmp_path, monkeypatch):
    eeg_rows, hb_rows = geometry()
    cfg = dict(seed=23, data=dict(datasets=['eeg_fnirs_single_trial'], protected_data='forbidden',
        cache_root='data/cache/physiology_semantic_clean_v5', fnirs_input='unified_cached_fnirs_v1', region='prefrontal',
        hb_pairs=4, eeg_channels=12, window_seconds=30, reference_seconds=5,
        window_offset_s=-5, split_seed=11, train_fraction=.6, selection_fraction=.2,
        max_records_per_subject=2, max_windows_per_record=3, max_subjects=10),
        tensor=dict(steps=120, dt_s=.25, reference_steps=20,
                    eeg_bands_hz=[list(b) for b in spatial.DEFAULT_BANDS]))
    records, refs = [], []
    for i in range(10):
        subject = f'subject_{i + 1:02d}'
        native_subject = f'subject {i + 1:02d}'
        sources = [dict(path=str(Path('data/EEG+NIRS Single-Trial') / group / native_subject / 'cnt.mat'),
                        sha256=f'fixture_identity_{group}') for group in ('EEG_01-29', 'NIRS_01-29')]
        for session in range(2):
            record_id = f'session_{session}'
            key = f'eeg_fnirs_single_trial|{subject}|{record_id}'
            manifest = dict(source_files=deepcopy(sources),
                array_shapes=dict(eeg=[20000, 16], fnirs=[1000, 10]),
                measurement=dict(eeg_channel_names=[r['channel_name'] for r in eeg_rows],
                                 fnirs_channel_names=[r['channel_name'] for r in hb_rows]))
            record = CleanCacheRecord(dataset_id='eeg_fnirs_single_trial', subject=native_subject,
                record_id=record_id, canonical_subject_id=subject, base_record_id=record_id,
                signal_branch='homer2_wavelength_pair', join_key=key, sample_rate_hz=10.,
                npz_path=tmp_path / 'not_read', manifest=manifest)
            records.append(record)
            for event_id in range(3):
                event = dict(event_index=event_id, eeg_time_ms=5000. + 30000 * event_id,
                             fnirs_time_ms=5000. + 30000 * event_id,
                             metadata=dict(task='motor_imagery', condition_label='LMI'))
                refs.append(SimpleNamespace(record=record, event=event))
    def for_channels(*, record, modality, channel_names):
        rows = eeg_rows if modality == 'eeg' else hb_rows
        lookup = {r['channel_name']: r for r in rows}
        return [deepcopy(lookup[n]) for n in channel_names]
    data = SimpleNamespace(windows=refs, geometry_index=SimpleNamespace(for_channels=for_channels),
                           excluded_alignment_records={}, excluded_alignment_support_windows=0)
    monkeypatch.setattr(spatial, 'UnifiedPhysiologyWindowDataset', lambda *a, **k: data)
    import src.data.clean_physiology_cache as cache
    monkeypatch.setattr(cache, 'CleanPhysiologyCacheIndex', lambda _: SimpleNamespace(records=records))
    native_time = np.arange(1000) / 10.
    rng = np.random.default_rng(41)
    eeg = SimpleNamespace(values=rng.normal(size=(20000, 16)), sample_rate_hz=200.,
        channel_names=tuple(r['channel_name'] for r in eeg_rows), native_unit='uV',
        unit_evidence='fixture explicit unit', reference='fixture native reference')
    intensity = np.exp(.02 * np.sin(native_time[:, None, None] / 2 + np.arange(5)[None, :, None])
                       + np.array([0., .01])[None, None, :])
    native = dict(values=np.log(intensity).copy(), time_s=native_time,
                  sample_rate_hz=10., channel_names=[r['channel_name'] for r in hb_rows],
                  optical_intensity=intensity, provenance=dict(native_unit='optical_detector_V', output_unit='relative_Hb', transform='fixture_bandpassed_cache'))
    reads = []
    def read_eeg(*_):
        reads.append('eeg'); return eeg
    def read_hb(*_):
        reads.append('hb'); return native
    monkeypatch.setattr(spatial, 'load_native_eeg_record', read_eeg)
    monkeypatch.setattr(spatial, 'load_cached_fnirs_record', read_hb)
    return SimpleNamespace(cfg=cfg, root=tmp_path, records=records, refs=refs, data=data,
                           eeg_rows=eeg_rows, hb_rows=hb_rows, eeg=eeg, native=native, reads=reads)


def test_patch_sizes_are_nested_unique_and_target_first():
    e, h = geometry()
    large = spatial.select_spatial_patch(e, h, 0, hb_pairs=4, eeg_channels=12)
    small = spatial.select_spatial_patch(e, h, 0, hb_pairs=2, eeg_channels=6)
    assert small['hb_pair_indices'] == large['hb_pair_indices'][:2]
    assert small['eeg_indices'] == large['eeg_indices'][:6]
    assert large['target_index'] == 0
    assert large['hb_channel_names'][0] == ['H0_HbO', 'H0_HbR']
    assert len(set(large['eeg_channel_names'])) == 12
    assert large['anchor_pair_indices'] == [0, 3]
    weights = np.asarray(large['anchor_eeg_weights'])
    np.testing.assert_allclose(weights.sum(axis=1), 1.)
    assert (weights > 0).sum(axis=1).tolist() == [6, 6]


def test_disjoint_selection_uses_verified_hardware_and_preserves_negative_support():
    e, h = geometry()
    for row in h[2:4]:
        row['source_index'] = 1
    patch = spatial.select_spatial_patch(e, h, 0)
    assert patch['hb_pair_names'] == ['H0', 'H2', 'H3', 'H4']
    assert len(set(patch['source_indices'])) == 4
    assert spatial.select_spatial_patch(e, h, 0, disjoint_optodes=False)['hb_pair_names'][1] == 'H1'
    for row in h[4:6]:
        row['detector_index'] = 1
    with pytest.raises(ValueError, match='insufficient Hb pairs'):
        spatial.select_spatial_patch(e, h, 0)


def test_prefrontal_three_by_three_optode_grid_excludes_target_sharing_only():
    """Four sites remain feasible although three mutually disjoint donors do not."""
    e, original = geometry()
    hb = []
    for source in range(1, 4):
        for detector in range(1, 4):
            base = f'S{source}D{detector}'
            for role, template in zip(('HbO', 'HbR'), original[:2]):
                hb.append(dict(template, channel_name=f'{base}_{role}', base_channel_name=base,
                               component=role, source_index=source, detector_index=detector,
                               x=.1 * (source - 1), y=1., z=.1 * (detector - 1)))
    patch = spatial.select_spatial_patch(e, hb, 0, hb_pairs=4, max_patch_angle_deg=45.)
    assert patch['hb_pair_names'] == ['S1D1', 'S2D2', 'S2D3', 'S3D2']
    assert patch['optode_policy'] == spatial.OPTODE_POLICY
    adjacency = np.asarray(patch['shared_optode_adjacency'])
    assert not adjacency[0].any() and not adjacency[:, 0].any()
    np.testing.assert_array_equal(adjacency[1:, 1:], [[False, True, True],
                                                   [True, False, False],
                                                   [True, False, False]])
    np.testing.assert_array_equal(patch['donor_shared_optode_adjacency'], adjacency[1:, 1:])
    assert patch['donor_shared_optode_support'] == [
        dict(pair_indices=[1, 2], source_index=2, detector_index=None),
        dict(pair_indices=[1, 3], source_index=None, detector_index=2)]
    small = spatial.select_spatial_patch(e, hb, 0, hb_pairs=2, eeg_channels=6)
    assert small['hb_pair_indices'] == patch['hb_pair_indices'][:2]


@pytest.mark.parametrize('fault', ['pair', 'missing_hardware', 'missing_geometry', 'units', 'frame', 'duplicate_eeg'])
def test_incomplete_identity_or_geometry_fails_closed(fault):
    e, h = geometry()
    if fault == 'pair': h.pop()
    elif fault == 'missing_hardware': h[0]['source_index'] = h[1]['source_index'] = None
    elif fault == 'missing_geometry': e[0]['x'] = None
    elif fault == 'units': e[0]['coordinate_units'] = 'unknown'
    elif fault == 'frame': h[0]['coordinate_system'] = h[1]['coordinate_system'] = 'other_frame'
    else: e[1]['channel_name'] = e[0]['channel_name'].lower()
    with pytest.raises(ValueError):
        spatial.select_spatial_patch(e, h, 0)


def test_inventory_is_array_free_deterministic_and_groups_every_subject(fixture_data):
    f = fixture_data
    first = spatial.inventory_spatial(f.cfg, f.root)
    f.data.windows.reverse()
    second = spatial.inventory_spatial(f.cfg, f.root)
    assert first == second
    assert first['measured_arrays_read'] == 0 and f.reads == []
    assert len(first['rows']) == 60
    assert [len(first['subjects'][s]) for s in spatial.SPLITS] == [6, 2, 2]
    assert len({r['patch']['patch_id'] for r in first['rows']}) == 1
    for subject in {r['subject'] for r in first['rows']}:
        assert len({r['split'] for r in first['rows'] if r['subject'] == subject}) == 1


def test_subject_limit_selects_metadata_supported_subjects_and_keeps_exclusions(fixture_data):
    f = fixture_data
    f.cfg['data']['max_subjects'] = 5
    plan = spatial.inventory_spatial(f.cfg, f.root)
    assert len({r['subject'] for r in plan['rows']}) == 5
    assert len(plan['excluded']) == 30
    assert {r['reason'] for r in plan['excluded']} == {'deterministic_metadata_subject_limit'}
    assert f.reads == []


@pytest.mark.parametrize('clock', ['eeg_time_ms', 'fnirs_time_ms'])
def test_either_native_clock_overlap_is_excluded_without_consuming_window_limit(fixture_data, clock):
    f = fixture_data
    f.refs[1].event[clock] = 15000.
    plan = spatial.inventory_spatial(f.cfg, f.root)
    assert len(plan['rows']) == 59
    assert len(plan['excluded']) == 1
    assert plan['excluded'][0]['reason'] == 'overlapping_native_window_support'
    key = f.refs[0].record.join_key
    assert [r['event_id'] for r in plan['rows'] if r['join_key'] == key] == [0, 2]
    assert f.reads == []


def test_engineering_angle_bound_prevents_distant_optode_substitution():
    e, h = geometry()
    for row in h[6:]:
        row.update(x=1., y=0., z=.1)
    with pytest.raises(ValueError, match='insufficient Hb pairs'):
        spatial.select_spatial_patch(e, h, 0, max_patch_angle_deg=45.)
    assert len(spatial.select_spatial_patch(e, h, 0, hb_pairs=2)['hb_pair_names']) == 2


@pytest.mark.parametrize('cause', ['hardware', 'subject_count'])
def test_unavailable_metadata_plan_preserves_exclusions_and_blocks_prepare(fixture_data, cause):
    f = fixture_data
    row = spatial.inventory_spatial(f.cfg, f.root)['rows'][0]
    if cause == 'hardware':
        f.hb_rows[0]['source_index'] = f.hb_rows[1]['source_index'] = None
    else:
        f.data.windows = [r for r in f.refs if r.record.canonical_subject_id in {'subject_01', 'subject_02'}]
    plan = spatial.inventory_spatial(f.cfg, f.root)
    assert plan['status'] == 'unavailable' and plan['rows'] == []
    assert plan['reason'] and plan['excluded']
    assert plan['subjects'] == {s: [] for s in spatial.SPLITS}
    if cause == 'hardware':
        assert len(plan['excluded']) == 60
        assert all('source/detector' in r['reason'] for r in plan['excluded'])
    with pytest.raises(ValueError, match='plan unavailable before array access'):
        spatial.prepare_spatial_record(f.cfg, f.root, row)
    assert f.reads == []


@pytest.mark.parametrize('fault', ['dataset', 'split', 'source', 'subject', 'geometry', 'window'])
def test_frozen_identity_is_checked_before_signal_read(fixture_data, fault):
    f = fixture_data
    row = deepcopy(spatial.inventory_spatial(f.cfg, f.root)['rows'][0])
    if fault == 'dataset': row['dataset'] = 'croce_local_cache'
    elif fault == 'split': row['split'] = 'protected_test'
    elif fault == 'source': row['source_identity'][0]['path'] = 'protected_DO_NOT_OPEN'
    elif fault == 'subject': row['subject'] = 'subject_99'
    elif fault == 'geometry': row['patch']['hb_coordinates'][0][0] += 1.
    else: row['hb_start_s'] += 1.
    with pytest.raises(ValueError, match='before array access'):
        spatial.prepare_spatial_record(f.cfg, f.root, row)
    assert f.reads == []


def test_out_of_scope_source_is_rejected_before_source_or_signal_open(fixture_data, monkeypatch):
    f = fixture_data
    row = deepcopy(spatial.inventory_spatial(f.cfg, f.root)['rows'][0])
    f.records[0].manifest['source_files'][1]['path'] = str(f.root / 'protected_DO_NOT_OPEN')
    def forbidden_open(*a, **k):
        raise AssertionError('source file must never be opened')
    monkeypatch.setattr(Path, 'open', forbidden_open)
    with pytest.raises(ValueError, match='before array access'):
        spatial.prepare_spatial_record(f.cfg, f.root, row)
    assert f.reads == []


def test_prepare_shapes_heldout_pair_mask_and_scoring_reference(fixture_data):
    f = fixture_data
    row = spatial.inventory_spatial(f.cfg, f.root)['rows'][0]
    prepared = spatial.prepare_spatial_record(f.cfg, f.root, row)
    assert prepared['eeg'].shape == (120, 12, 5)
    assert prepared['features'].shape == (120, 60)
    assert prepared['hb'].shape == prepared['input_hb'].shape == (120, 4, 2)
    assert np.isfinite(prepared['hb']).all()
    assert np.isnan(prepared['input_hb'][:, 0]).all()
    assert prepared['hb_observed_mask'][:, 1:].all()
    assert not prepared['hb_observed_mask'][:, 0].any()
    assert prepared['target_mask'][:, 0].all()
    assert not prepared['target_mask'][:, 1:].any()
    assert not prepared['scoring_mask'][:20].any()
    assert prepared['scoring_mask'][20:, 0].all()
    assert not prepared['scoring_mask'][:, 1:].any()
    assert prepared['metadata']['target_reference'] == 'scoring_coordinate_only_never_visible_input'
    assert prepared['metadata']['fnirs_input'] == 'unified_cached_fnirs_v1'
    np.testing.assert_allclose(prepared['hb'][:20].mean(axis=0), 0., atol=1e-10)
    assert f.reads == ['eeg', 'hb']


def test_poisoning_hidden_pair_including_reference_cannot_change_any_visible_input(fixture_data):
    f = fixture_data
    row = spatial.inventory_spatial(f.cfg, f.root)['rows'][0]
    original = spatial.prepare_spatial_record(f.cfg, f.root, row)
    target = row['patch']['hb_pair_indices'][0]
    times = f.native['time_s']
    f.native['values'][:, target, :] += .4 * np.cos(times[:, None] / 3.) + np.array([.1, -.2])
    poisoned = spatial.prepare_spatial_record(f.cfg, f.root, row)
    assert not np.array_equal(original['hb'][:, 0], poisoned['hb'][:, 0])
    np.testing.assert_array_equal(original['eeg'], poisoned['eeg'])
    np.testing.assert_array_equal(original['input_hb'], poisoned['input_hb'])
    np.testing.assert_array_equal(original['hb'][:, 1:], poisoned['hb'][:, 1:])
    # Raw optical data are not a model input and cannot override cached Hb.
    f.native['optical_intensity'][:] = np.nan
    again = spatial.prepare_spatial_record(f.cfg, f.root, row)
    np.testing.assert_array_equal(poisoned['hb'], again['hb'])


def test_missing_cached_support_is_retained_as_failure_without_interpolation(fixture_data):
    f = fixture_data
    row = spatial.inventory_spatial(f.cfg, f.root)['rows'][0]
    f.native['values'][80, 1, 0] = np.nan
    with pytest.raises(ValueError, match='complete finite unified-cache'):
        spatial.prepare_spatial_record(f.cfg, f.root, row)
