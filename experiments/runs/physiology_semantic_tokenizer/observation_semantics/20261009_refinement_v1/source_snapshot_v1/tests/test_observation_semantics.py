"""Synthetic-only tests; no retained evidence or native datasets are required."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.inference.observation_semantics import (
    lagged, event_impulses, coordinate_stability, hardware_donors,
    candidate_inference, block_interval, sign_flip_p, holm, ar1_whiten,
)


@pytest.fixture(scope='module')
def runner():
    path = Path(__file__).resolve().parents[1]/'experiments/scripts/evaluate_observation_semantics.py'
    spec = importlib.util.spec_from_file_location('observation_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nonwrapping_lags_cannot_copy_window_end_into_beginning():
    x = np.arange(30.).reshape(10, 3)
    z = lagged(x, [2, 0, -2])
    assert np.isnan(z[:2, :3]).all()
    assert np.isnan(z[-2:, -3:]).all()
    np.testing.assert_array_equal(z[2:, :3], x[:-2])
    np.testing.assert_array_equal(z[:-2, -3:], x[2:])
    with pytest.raises(ValueError):
        lagged(x, [10])


def test_total_observation_can_be_stable_while_attribution_changes():
    t = np.linspace(0, 1, 120)
    c = np.stack((np.c_[t, -t], np.c_[-t, t]))
    physical = -c
    observed = coordinate_stability(c+physical, np.ones(2))
    component = coordinate_stability(c, np.ones(2))
    assert observed['stable_coordinate'].all()
    assert component['stable_coordinate'][0]
    assert not component['stable_coordinate'][1]
    np.testing.assert_allclose(component['rms_range_SD'][1], np.sqrt(np.mean((2*t)**2)))


def test_one_solution_is_not_identifiability_evidence():
    result = coordinate_stability(np.zeros((1, 120, 2)), np.ones(2))
    assert not result['sufficient_candidates']
    assert not result['stable_coordinate'].any()
    assert not result['pointwise_stable'].any()
    with pytest.raises(ValueError):
        coordinate_stability(np.zeros((2, 120, 2)), [0., 1.])


def test_hardware_matching_uses_ids_not_position_or_missing_ids():
    def row(s, d, x):
        return dict(source_index=s, detector_index=d, x=x, y=0., z=0.)
    rows = [row(1, 1, 0), row(1, 2, 1), row(2, 2, 1.1), row(3, 3, 3), row(None, None, .01)]
    result = hardware_donors(rows, 0)
    assert result['indices'] == dict(shared_optode=1, disjoint_distance_matched=2, disjoint_far=3)
    assert abs(result['distance_mismatch']-.1) < 1e-12
    assert not hardware_donors(rows, 4)['available']


def test_eog_event_detector_preserves_sign_and_uses_frozen_scale():
    x = np.zeros((500, 2))
    x[100, 1], x[300, 1] = 4., -5.
    _, rows = event_impulses(x, [1., 1.], rate=50, height=3.)
    vertical = [r for r in rows if r['kind'] == 0]
    assert [r['signed_height'] for r in vertical] == [4., -5.]
    _, rows2 = event_impulses(x, [2., 1.], rate=50, height=3.)
    assert not rows2


def test_exact_observation_alias_remains_ambiguous_without_anchors():
    obs = np.zeros((120, 2))
    pred = np.stack((obs, obs))
    absent = candidate_inference(obs, pred, .1)
    assert absent['admissible'].all()
    supported = candidate_inference(obs, pred, .1, anchors=np.array([1.]),
        anchor_predictions=np.array([[0.], [1.]]), anchor_sd=.1)
    assert supported['best'] == 1
    assert supported['admissible'].tolist() == [False, True]
    missing_again = candidate_inference(obs, pred, .1)
    np.testing.assert_array_equal(absent['cost'], missing_again['cost'])


def test_wrong_anchor_can_confidently_reverse_attribution():
    obs = np.ones((120, 2))
    pred = np.stack((obs, obs+.001))
    inferred = candidate_inference(obs, pred, 1., anchors=np.array([1.]),
        anchor_predictions=np.array([[0.], [1.]]), anchor_sd=.01)
    assert inferred['best'] == 1
    assert not inferred['admissible'][0]


def test_independent_unit_statistics_and_multiple_comparisons():
    x = np.arange(1., 6.)
    assert sign_flip_p(x) == 1/32
    assert block_interval(x)['n'] == 5
    np.testing.assert_allclose(holm([.01, .04, .02]), [.03, .04, .04])


def test_ar1_whitening_matches_dense_inverse_covariance():
    rho = .95
    values = np.random.default_rng(22).normal(size=(12, 2))
    covariance = rho**abs(np.arange(12)[:, None]-np.arange(12))
    white = ar1_whiten(values, rho)
    np.testing.assert_allclose(np.sum(white**2), np.sum(values*np.linalg.solve(covariance, values)), rtol=1e-12)
    np.testing.assert_array_equal(ar1_whiten(values, 0), values)


def test_geometry_rematching_minimizes_distance_without_accessing_signals():
    def row(s, d, x):
        return dict(source_index=s, detector_index=d, x=x, y=0., z=0.)
    rows = [row(1, 1, 0), row(1, 2, 1), row(1, 3, 2), row(2, 2, 2.01), row(3, 3, 4)]
    first = hardware_donors(rows, 0)
    matched = hardware_donors(rows, 0, matching='minimum_distance_mismatch')
    assert first['indices']['shared_optode'] == 1
    assert matched['indices']['shared_optode'] == 2
    assert abs(matched['distance_mismatch']) < .02


def test_configuration_and_probe_do_not_read_measured_data(runner, monkeypatch):
    cfg = runner.read_config(runner.DEFAULT_CONFIG)
    monkeypatch.setattr(runner, 'make_plan', lambda *args: pytest.fail('unexpected data read'))
    assert runner.software_probe(cfg)['measured_reads'] == 0
    assert runner.phase('subject_02') == 'evaluate'
    assert runner.phase('VP003') == 'train'


def test_window_selection_checks_both_native_clocks(runner):
    refs = [dict(id=str(i), eeg_start_s=e, hb_start_s=h) for i, (e, h) in enumerate(
        [(0, 10), (30, 35), (60, 70), (90, 100)])]
    assert [r['id'] for r in runner.nonoverlap(refs, 3)] == ['0', '2', '3']


def test_mechanism_generator_has_separate_physical_and_measurement_actions(runner):
    cfg = runner.read_config(runner.DEFAULT_CONFIG)
    lib = runner.mechanism_library(cfg, 0, 'train')
    assert len(lib['entries']) == 36
    for e in lib['entries']:
        np.testing.assert_allclose(e['prediction'], e['physical']+e['component'])
        assert np.isfinite(e['prediction']).all()
        assert e['prediction'].shape == (120, 2)
        if e['mechanism'] == 'tau':
            assert not np.any(e['component'])
            assert not np.any(e['anchor'])  # no direct tau/class-label oracle in anchor
        else:
            np.testing.assert_array_equal(e['physical'], lib['baseline'])
    heldout = runner.mechanism_library(cfg, 0, 'evaluate')
    assert not np.array_equal(lib['driver'], heldout['driver'])


def test_synthetic_worker_missing_anchor_is_exactly_absent_and_keeps_all_cells(runner, tmp_path):
    cfg = runner.read_config(runner.DEFAULT_CONFIG)
    cfg = copy.deepcopy(cfg)
    cfg['mechanisms']['training_identities'] = 2
    runner.calibrate_mechanisms(cfg, tmp_path)
    result = runner.mechanism_worker((cfg, str(tmp_path), str(tmp_path), dict(repeat=0)))
    assert result['status'] == 'completed'
    assert len(result['rows']) == 48
    import pandas as pd
    df = pd.DataFrame(result['rows'])
    a = df[df.arm == 'absent'].drop(columns='arm').reset_index(drop=True)
    b = df[df.arm == 'missing_at_inference'].drop(columns='arm').reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_native_missing_pair_does_not_discard_other_channels(runner, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import src.data.clean_physiology_cache as cache
    import src.data.unified_physiology as unified
    cfg = runner.read_config(runner.DEFAULT_CONFIG)
    rng = np.random.default_rng(21)
    record = SimpleNamespace(join_key='eeg_fnirs_single_trial|subject_01|session_00')
    names = [f'CH{i}_{c}' for i in range(36) for c in ('HbO', 'HbR')]
    native_hb = rng.normal(size=(301, 36, 2))
    native_hb[100, 1] = np.nan
    native = dict(values=native_hb, time_s=np.arange(301)/10., channel_names=names,
                  optical_intensity=None, provenance={'unit': 'synthetic_fixture'})
    eeg = SimpleNamespace(values=rng.normal(size=(6000, 6)), sample_rate_hz=200.,
        channel_names=tuple('ABCDEF'), auxiliary_channel_names=('HEOG', 'VEOG'),
        auxiliary_values=rng.normal(size=(6000, 2)), native_unit='uV', unit_evidence='fixture',
        source_path=tmp_path/'synthetic_native')
    monkeypatch.setattr(cache, 'CleanPhysiologyCacheIndex', lambda *a: SimpleNamespace(records=[record]))
    monkeypatch.setattr(unified, 'load_native_eeg_record', lambda *a: eeg)
    monkeypatch.setattr(unified, 'load_native_fnirs_record', lambda *a: native)
    monkeypatch.setattr(unified, 'ChannelGeometryIndex', lambda *a: SimpleNamespace(for_channels=lambda **k: []))
    ref = dict(id='fixture', eeg_start_s=0., hb_start_s=0., eeg_channels=list('ABCDEF'), hb_channel='CH0')
    spec = dict(key='fixture', dataset='eeg_fnirs_single_trial', record='session_00', subject='subject_01',
                split='train', join_key=record.join_key, windows=[ref])
    result = runner.prepare_worker((cfg, str(tmp_path), str(tmp_path), spec))
    assert result['available_windows'] == 1
    assert result['valid_Hb_pairs_per_window'] == [35]
    assert result['valid_target_Hb_windows'] == 1
    with np.load(tmp_path/'prepared/fixture.npz') as arrays:
        assert np.isfinite(arrays['eeg_voltage']).all()
        assert not np.isfinite(arrays['hb'][0, :, 1]).any()
        assert np.isfinite(arrays['hb'][0, :, 0]).all()


def test_event_pipeline_freezes_training_and_preserves_missing_endpoint_denominator(runner, tmp_path, monkeypatch):
    rng = np.random.default_rng(10)
    windows = []
    rate = 50
    for subject in ('subject_01', 'subject_02', 'subject_03', 'subject_04'):
        for i in range(2):
            eog = rng.normal(size=(1500, 2))
            eog[:, 1] += 8*np.exp(-.5*((np.arange(1500)-700)/8)**2)
            voltage = eog@rng.normal(size=(2, 6))+.1*rng.normal(size=(1500, 6))
            hb = rng.normal(size=(120, 1, 2))
            if subject == 'subject_02' and i == 0:
                hb[:] = np.nan
            windows.append(dict(ref=dict(id=subject+str(i), condition='A', window=i),
                subject=subject, split=runner.phase(subject), meta=dict(target_pair=0),
                eeg_voltage=voltage, eog_voltage=eog, eeg_raw=np.repeat(voltage, 4, axis=0),
                hb=hb, od_jump=abs(rng.normal(size=(120, 1)))))
    cfg = copy.deepcopy(runner.read_config(runner.DEFAULT_CONFIG))
    cfg['measured']['datasets'] = ['eeg_fnirs_single_trial']
    monkeypatch.setattr(runner, 'load_prepared', lambda *a: ([], windows))
    result = runner.event_analysis(cfg, tmp_path, {})
    assert result['status'] == 'completed'
    import pandas as pd
    metrics = pd.read_csv(tmp_path/'event_metrics.csv')
    volts = metrics[metrics.endpoint == 'EEG_voltage']
    assert volts.nrmse.notna().all()  # identical support for nonwrapping shifted control
    hb = metrics[(metrics.endpoint == 'HbT') & (metrics.arm == 'EOG_real')]
    assert len(hb) == 4 and hb.nrmse.isna().sum() == 1
    frozen = runner.read_json(tmp_path/'event_models.json')['eeg_fnirs_single_trial']
    assert frozen['training_subjects'] == ['subject_01', 'subject_03']
    assert frozen['evaluation_subjects'] == ['subject_02', 'subject_04']
    expected = runner.fit_standardized_ridge(
        np.concatenate([runner.lagged(w['eog_voltage'], [-2, 0, 2])[runner.score_mask(50, [8, 21])]
                        for w in windows if w['split'] == 'train']),
        np.concatenate([w['eeg_voltage'][runner.score_mask(50, [8, 21])]
                        for w in windows if w['split'] == 'train']), .01)
    np.testing.assert_allclose(frozen['voltage_model']['coefficient'], expected['coefficient'])
