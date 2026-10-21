"""Small non-authorizing fixtures for the independent-information suite."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.inference.crossmodal_information import (
    exact_sign_flip, fit_condition, paired_statistics, perturb_inputs, physical_parameters,
    predict_ridge, score_target,
)
from src.inference.observation_baselines import ridge_fit
from src.inference.shared_driver_spatial import forward_spatial_shared_driver, temporal_mode_basis

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('crossmodal_information_runner', ROOT/'experiments/scripts/evaluate_crossmodal_information.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def records(t=120):
    rng = np.random.default_rng(1)
    return dict(eeg=rng.normal(size=(t, 60)), hb=rng.normal(size=(t, 4, 2)),
                eeg_operator=np.eye(t), hb_operator=np.eye(t))


def test_config_freezes_input_and_artifact_boundaries(tmp_path):
    cfg = runner.read_config()
    assert cfg['model']['fixed_eeg_weight'] == 1
    assert cfg['synthetic']['calibration_seeds'] == cfg['synthetic']['evaluation_seeds'] == 16
    changed = deepcopy(cfg)
    changed['source_run'] = 'data/protected'
    import yaml
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match='boundary'):
        runner.read_config(path)


def test_shift_is_nonwrapping_and_every_arm_has_identical_visible_support():
    record = records()
    record['eeg'] = np.broadcast_to(np.arange(120)[:, None], (120, 60)).copy()
    shifts = dict(shift_m4s=-16, shift_p4s=16, shift_m8s=-32, shift_p8s=32)
    for condition, lag in [('paired', 0), *shifts.items()]:
        e, h, mask = perturb_inputs(record, condition, shift_steps=shifts)
        np.testing.assert_array_equal(e[32:88, 0], np.arange(32, 88)+lag)
        assert mask[:, :60].sum() == 56*60
        assert np.isnan(e[:32]).all() and np.isnan(e[88:]).all()
        assert np.isnan(h[:, 0]).all()
        assert mask[:, 62:].all()
    with pytest.raises(ValueError, match='noncircular'):
        perturb_inputs(record, 'bad', shift_steps={'bad': 80})


def test_wrong_donor_hb_is_never_an_input_and_missing_partner_has_no_fallback():
    record = records()
    wrong = dict(eeg=2*record['eeg'], hb=object())
    eeg, hb, _ = perturb_inputs(record, 'wrong_EEG', wrong=wrong)
    np.testing.assert_equal(eeg[32:88], 2*record['eeg'][32:88])
    np.testing.assert_equal(hb[:, 1:], record['hb'][:, 1:])
    with pytest.raises(ValueError, match='unavailable'):
        perturb_inputs(record, 'wrong_EEG')


def test_hidden_target_poison_cannot_change_features_or_frozen_ridge():
    record = records()
    rng = np.random.default_rng(4)
    donors = record['hb'][:, 1:].reshape(120, -1)
    target = rng.normal(size=(120, 2))
    calibration = dict(baselines={
        'spatial_ridge__Hb_only': ridge_fit(donors, target, .1),
        'joint_ridge__paired': ridge_fit(np.c_[record['eeg'], donors], target, .1)})
    poisoned = deepcopy(record)
    poisoned['hb'][:, 0] = np.nan
    for condition in ('Hb_only', 'paired'):
        before = perturb_inputs(record, condition)
        after = perturb_inputs(poisoned, condition)
        for left, right in zip(before, after):
            np.testing.assert_equal(left, right)
        np.testing.assert_array_equal(predict_ridge(record, calibration, condition),
                                      predict_ridge(poisoned, calibration, condition))


def test_scoring_uses_frozen_training_sd_and_fixed_support():
    target = np.ones((120, 2))
    prediction = np.zeros_like(target)
    prediction[:20] = 999
    score = score_target(prediction, target, np.array([1., 2.]))
    assert score['nmse'] == pytest.approx(.625)
    assert score['scored_values'] == 200
    target[:20] = np.nan
    assert score_target(prediction, target, np.array([1., 2.]))['nmse'] == pytest.approx(.625)


def test_exact_subject_sign_flip_and_paired_direction():
    assert exact_sign_flip(np.ones(6)) == pytest.approx(2/64)
    result = paired_statistics({'a': 1., 'b': 2.}, {'a': 2., 'b': 3.}, repeats=100)
    assert result['delta_reference_minus_candidate'] == 1
    assert result['independent_units'] == 2


def test_failure_penalty_and_common_wrong_denominator():
    cfg = runner.read_config()
    rows = []
    for i in range(3):
        for condition in ('paired', 'wrong_EEG'):
            rows.append(dict(record_id=str(i), unit='subject', arm='K1_true_H4_E12', method='SSM_fixed1',
                condition=condition, success=True, status='completed', nmse=1. if condition == 'paired' else 4.))
    rows[-1].update(success=False, status='unavailable_wrong_EEG')
    result = runner.compare(rows, 'K1_true_H4_E12', 'SSM_fixed1', 'wrong_EEG', cfg)
    assert result['planned_window_pairs'] == 3
    assert result['available_window_pairs'] == 2
    assert result['delta_reference_minus_candidate'] == 1
    rows[3].update(success=False, status='failed_numerical')
    result = runner.compare(rows, 'K1_true_H4_E12', 'SSM_fixed1', 'wrong_EEG', cfg)
    assert result['reference_failed'] == 1
    assert result['mean_unit_delta_nmse_reference_minus_candidate'] == 51


def test_matched_synthetic_fitter_cannot_use_hidden_target_or_zero_weight_eeg():
    t, m, k = 24, 4, 1
    parent = dict(fixed=dict(tau=2., kappa=.64, gamma=.32, alpha=.32, E0=.32, P0=1., Q0=.35,
        neurovascular_gain=1., eeg_loading=1.), tensor=dict(temporal_modes=m, dt_s=.25),
        solver=dict(numerical_backend='numba', substeps=4, validation_substeps=8,
            integration_tolerance_training_sd=.005, max_nfev=50, gradient_tolerance=1e-5,
            driver_curvature_weight=.01, initial_weight=100., flow_prior_weight=1., flow_log_sd=np.log(2.)))
    loading = np.ones((60, k))
    calibration = dict(readout=dict(eeg_loadings=loading), design=dict(hb_loadings=np.ones((4, k))))
    forward = forward_spatial_shared_driver(np.array([[.01], [.005], [-.002], [.001]]), np.zeros((k, 5)),
        physical_parameters(parent), .25, eeg_loadings=loading, hb_loadings=np.ones((4, k)),
        temporal_basis=temporal_mode_basis(t, m), derivative=False)
    record = dict(eeg=forward['eeg_prediction'], hb=forward['hb_prediction'],
        eeg_operator=np.eye(t), hb_operator=np.eye(t))
    coordinate = dict(sd=np.r_[np.full(60, .02), np.tile([.002, .001], 4)])
    fit = fit_condition(record, calibration, coordinate, parent, 'Hb_only', support=(4, 20))
    poisoned = deepcopy(record)
    poisoned['hb'][:, 0] = np.nan
    poisoned['eeg'][:] = np.nan
    repeated = fit_condition(poisoned, calibration, coordinate, parent, 'Hb_only', support=(4, 20))
    assert fit['status'] == repeated['status']
    np.testing.assert_array_equal(fit['coefficients'], repeated['coefficients'])
    np.testing.assert_array_equal(fit['prediction'], repeated['prediction'])


def test_transform_hides_nonfinite_target_before_validation():
    from src.inference.crossmodal_information import scaled_record
    record = records()
    record['eeg'] = record['eeg'].reshape(120, 12, 5)
    coordinate = dict(eeg_scale=np.ones((12, 5)), hb_scale=1., hb_shape=(4, 2))
    original = scaled_record(record, coordinate)
    record['hb'][:, 0] = np.inf
    poisoned = scaled_record(record, coordinate)
    np.testing.assert_equal(original['eeg'], poisoned['eeg'])
    np.testing.assert_equal(original['hb'], poisoned['hb'])
    with pytest.raises(ValueError):
        scaled_record(record, coordinate, hide_target=False)


def test_retained_eeg_only_donor_needs_no_hb_array(tmp_path):
    row = dict(id='donor')
    path = tmp_path/'prepared'/(runner.token(row['id'])+'.npz')
    path.parent.mkdir()
    np.savez(path, eeg=np.ones((120, 12, 5)))
    path.with_suffix('.json').write_text(__import__('json').dumps(dict(status='completed', row=row)))
    donor = runner.load_retained(tmp_path, row, fields=('eeg',))
    assert set(donor) == {'eeg'}


def test_wrong_partner_metadata_fails_closed_before_read():
    from src.inference.crossmodal_information import validate_wrong_partners
    base = dict(dataset='public', subject='s1', task='MI', condition='left', split='evaluation',
        window_seconds=30., record='r1', eeg_start_s=0., hb_start_s=0.,
        patch=dict(eeg_channel_names=['E'], hb_pair_names=['H'], coordinate_system='head',
                   eeg_coordinates=[[1, 0, 0]], hb_coordinates=[[1, 0, 0]]))
    row = dict(base, id='a')
    donor = dict(base, id='b', eeg_start_s=40., hb_start_s=40.)
    assert validate_wrong_partners([row, donor], {'a': 'b'})
    for changed in (dict(donor, split='train'), dict(donor, subject='s2'),
                    dict(donor, eeg_start_s=5.), dict(donor, patch={})):
        with pytest.raises(ValueError):
            validate_wrong_partners([row, changed], {'a': 'b'})
    with pytest.raises(ValueError):
        validate_wrong_partners([row, donor], {'a': 'absent'})


def test_worker_retains_identical_prediction_when_only_scoring_target_is_poisoned(tmp_path, monkeypatch):
    cfg = runner.read_config()
    raw = records()
    raw['eeg'] = raw['eeg'].reshape(120, 12, 5)
    donors = raw['hb'][:, 1:].reshape(120, -1)
    calibration = dict(selected_eeg_weight=0., baselines={
        'spatial_ridge__Hb_only': ridge_fit(donors, raw['hb'][:, 0], .1),
        'joint_ridge__paired': ridge_fit(np.c_[raw['eeg'].reshape(120, 60), donors], raw['hb'][:, 0], .1)})
    source = dict(plan=dict(rows=[dict(id='row')]), path=tmp_path,
        coordinate=dict(eeg_scale=np.ones((12, 5)), hb_scale=1., hb_shape=(4, 2), sd=np.ones(68)),
        calibrations={'K1_true_H4_E12': calibration})
    monkeypatch.setattr(runner, 'retained_source', lambda *a: source)
    monkeypatch.setattr(runner, 'load_retained', lambda *a, **kw: deepcopy(raw))
    spec = dict(id='clean', record_id='row', family='measured', unit='s1', arm='K1_true_H4_E12',
                condition='paired', method='raw_ridge', wrong_eeg_id=None)
    clean = runner.evaluate_cell((cfg, str(tmp_path), str(tmp_path), spec))
    raw['hb'][:, 0] = np.inf
    poisoned = runner.evaluate_cell((cfg, str(tmp_path), str(tmp_path), dict(spec, id='poisoned')))
    assert clean['input_fit_success'] and poisoned['input_fit_success']
    assert clean['status'] == 'completed'
    assert poisoned['status'] == 'failed_scoring'
    with np.load(tmp_path/'cells/clean.npz') as left, np.load(tmp_path/'cells/poisoned.npz') as right:
        np.testing.assert_array_equal(left['target_prediction'], right['target_prediction'])


def test_unused_eeg_poison_never_changes_worker_prediction(tmp_path, monkeypatch):
    cfg = runner.read_config()
    raw = records()
    raw['eeg'] = raw['eeg'].reshape(120, 12, 5)
    donors = raw['hb'][:, 1:].reshape(120, -1)
    calibration = dict(selected_eeg_weight=0., baselines={
        'spatial_ridge__Hb_only': ridge_fit(donors, raw['hb'][:, 0], .1),
        'joint_ridge__paired': ridge_fit(np.c_[raw['eeg'].reshape(120, 60), donors], raw['hb'][:, 0], .1)})
    source = dict(plan=dict(rows=[dict(id='row')]), path=tmp_path,
        coordinate=dict(eeg_scale=np.ones((12, 5)), hb_scale=1., hb_shape=(4, 2), sd=np.ones(68)),
        calibrations={'K1_true_H4_E12': calibration})
    monkeypatch.setattr(runner, 'retained_source', lambda *a: source)
    monkeypatch.setattr(runner, 'load_retained', lambda *a, **kw: deepcopy(raw))
    for condition in ('Hb_only', 'paired', 'shift_p8s'):
        spec = dict(id=condition+'_clean', record_id='row', family='measured', unit='s1', arm='K1_true_H4_E12',
            condition=condition, method='raw_ridge', wrong_eeg_id=None)
        clean = runner.evaluate_cell((cfg, str(tmp_path), str(tmp_path), spec))
        before = raw['eeg'].copy()
        used = np.zeros(120, bool)
        if condition != 'Hb_only':
            lag = cfg['perturbations']['shift_steps'].get(condition, 0)
            used[np.arange(32, 88)+lag] = True
        raw['eeg'][~used] = np.inf
        poisoned = runner.evaluate_cell((cfg, str(tmp_path), str(tmp_path), dict(spec, id=condition+'_poisoned')))
        raw['eeg'] = before
        assert clean['status'] == poisoned['status'] == 'completed'
        with np.load(tmp_path/'cells'/(condition+'_clean.npz')) as left, np.load(tmp_path/'cells'/(condition+'_poisoned.npz')) as right:
            np.testing.assert_array_equal(left['target_prediction'], right['target_prediction'])


def positive_fixture():
    parent = dict(fixed=dict(tau=2., kappa=.64, gamma=.32, alpha=.32, E0=.32, P0=1., Q0=.35,
        neurovascular_gain=1., eeg_loading=1.), tensor=dict(steps=120, temporal_modes=16, dt_s=.25),
        solver=dict(numerical_backend='numba', substeps=4, validation_substeps=8,
            integration_tolerance_training_sd=.005, max_nfev=180, gradient_tolerance=1e-5,
            driver_curvature_weight=.01, initial_weight=100., flow_prior_weight=1., flow_log_sd=np.log(2.)))
    coordinate = dict(sd=np.r_[np.full(60, .025), np.tile([.03, .011], 4)])
    return parent, coordinate


def test_positive_contract_has_independent_full_288_cell_denominator():
    cfg = runner.read_positive_config()
    plan = runner.positive_manifest(cfg)
    assert plan['planned_cells'] == 288
    assert sum(row['partition'] == 'calibration' for row in plan['cells']) == 96
    assert len({row['seed'] for row in plan['cells'] if row['partition'] == 'evaluation'}) == 16
    assert all(row['dose'] == 0 for row in plan['cells'] if row['partition'] == 'calibration')


def test_positive_deleting_private_direction_preserves_mode1_and_noise_exactly():
    from src.inference.crossmodal_information import positive_record
    parent, coordinate = positive_fixture()
    full_off = positive_record(30261021, 0., 'full_donors', .5, parent, coordinate)
    private_off = positive_record(30261021, 0., 'private_donors', .5, parent, coordinate)
    full_on = positive_record(30261021, 1., 'full_donors', .5, parent, coordinate)
    private_on = positive_record(30261021, 1., 'private_donors', .5, parent, coordinate)
    np.testing.assert_allclose(full_off['record']['hb'], private_off['record']['hb'], atol=1e-15)
    np.testing.assert_array_equal(full_on['record']['hb'][:, 0], private_on['record']['hb'][:, 0])
    np.testing.assert_array_equal(private_on['record']['hb'][:, 1:], private_off['record']['hb'][:, 1:])
    for name in ('eeg', 'wrong_eeg', 'hb'):
        np.testing.assert_array_equal(full_off['noise'][name], full_on['noise'][name])
        np.testing.assert_array_equal(full_off['noise'][name], private_on['noise'][name])
    loading = private_on['calibration']['readout']['eeg_loadings']
    assert np.count_nonzero(loading[:30, 1]) == np.count_nonzero(loading[30:, 0]) == 0
    assert np.linalg.matrix_rank(private_on['calibration']['design']['hb_loadings'][1:]) == 1
    assert np.linalg.matrix_rank(full_on['calibration']['design']['hb_loadings'][1:]) == 2


def test_positive_known_forward_private_direction_is_recoverable_by_same_fitter():
    from src.inference.crossmodal_information import positive_record
    parent, coordinate = positive_fixture()
    generated = positive_record(31261021, 1., 'private_donors', .5, parent, coordinate)
    results = {}
    for condition in ('Hb_only', 'paired', 'wrong_EEG'):
        fit = fit_condition(generated['record'], generated['calibration'], coordinate, parent,
                            condition, wrong=generated['wrong'])
        assert fit['success'], fit.get('convergence_reason')
        results[condition] = score_target(fit['hb_prediction'][:, 0], generated['record']['hb'][:, 0], coordinate['sd'][60:62])['nmse']
    # The contract tests an increment with partial EEG and retained priors;
    # it does not prescribe near-zero absolute waveform recovery.
    assert results['paired'] < results['Hb_only']
    assert results['paired'] < results['wrong_EEG']


def test_positive_exception_cells_keep_full_manifest_denominators(tmp_path, monkeypatch):
    cfg = runner.read_positive_config()
    plan = runner.positive_manifest(cfg)
    runner.write_json(tmp_path/'manifest.json', plan)
    runner.write_json(tmp_path/'calibration.json', dict(known_hb_gain=.5))
    for spec in plan['cells']:
        runner.write_json(tmp_path/'cells'/(spec['id']+'.json'),
                          dict(spec, success=False, status='failed_exception', error='fixture'))
    monkeypatch.setattr(runner, 'render_positive', lambda *args: None)
    result = runner.summarize_positive(cfg, tmp_path)
    assert result['execution'] == 'completed'
    assert result['planned'] == result['terminal'] == 288
    assert result['status_counts'] == {'failed_exception': 288}
    for regime in result['regimes'].values():
        assert regime['empirical_zero_threshold_nmse'] is None
        for group in regime['groups'].values():
            assert group['planned_units'] == group['complete_units'] == 16
            assert group['successful_units'] == 0
            assert group['planned_exceedance_rate'] is None
