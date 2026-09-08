import copy

import numpy as np
import pytest

from experiments import evaluate_step5_observation_diagnostic as diagnostic


def test_scope_fails_before_native_reader(monkeypatch, tmp_path):
    cfg, base, measured, metadata = diagnostic.load_config()
    import src.data.unified_physiology as native
    calls = []
    monkeypatch.setattr(native, 'load_native_eeg_record', lambda *args: calls.append(args))
    for subject in ('subject_02', 'subject_19', 'subject_24'):
        with pytest.raises(ValueError, match='boundary'):
            diagnostic.load_training_subject(subject, cfg, base, measured, metadata)
    assert not calls
    cfg['subjects'] = ['subject_24']
    import yaml
    path = tmp_path/'bad.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='preselected'):
        diagnostic.load_config(path)


def test_original_heldout_events_never_enter_training_windows():
    _, base, _, _ = diagnostic.load_config()
    events = [dict(label='MA', event_index=i, eeg_time_ms=i*1000) for i in range(10)]
    first = diagnostic.training_events(events, base)
    events[4]['eeg_time_ms'] = -1e9
    events[9]['eeg_time_ms'] = 1e9
    second = diagnostic.training_events(events, base)
    assert first == second
    assert [i for i, _ in first] == [0, 1, 2, 3, 5, 6, 7, 8]


def test_trace_matches_joint_filter_and_segments_sum_without_resets():
    cfg, base, _, _ = diagnostic.load_config()
    generated = diagnostic.step5.localization.generate_matched(base, 'W', 0., 29)
    y = generated['observations']
    y[30:40, 0] = np.nan
    y[70:80, 1:] = np.nan
    p, c = diagnostic.step5.localization.model(base, 'W', -.5)
    expected = diagnostic.step5.joint.parameter_log_likelihood(y, p, config=c, quadrature_order=7)
    trace = diagnostic.filter_trace(y, base, -.5, 7, True)
    assert trace['increments'].sum() == pytest.approx(expected, abs=1e-10)
    row = diagnostic.curve_job(base, y[None], 'joint', -.5, cfg, 7)
    assert sum(row['chronological_segment_log_likelihood'].values()) == pytest.approx(expected, abs=1e-10)


def test_single_modality_curve_retains_valid_json_and_exact_eeg_w_invariance():
    import json
    cfg, base, _, _ = diagnostic.load_config()
    base['model']['steps'] = 24
    y = diagnostic.step5.localization.generate_matched(base, 'W', 0., 71)['observations']
    rows = [diagnostic.curve_job(base, y[None], 'EEG_only', w, cfg, 7) for w in (0., -.5)]
    for row in rows:
        json.dumps(diagnostic.step5.localization.jsonable(row), allow_nan=False)
    assert rows[0]['parameter_log_likelihood'] == pytest.approx(rows[1]['parameter_log_likelihood'], abs=1e-10)


def test_local_feature_hides_raw_target_before_filter_and_power():
    cfg, _, _, _ = diagnostic.load_config()
    eeg = np.random.default_rng(81).normal(size=(6000, 2))
    changed = eeg.copy()
    changed[2600:3400, 0] = 1e7
    first = diagnostic.local_eeg_feature(eeg, ['F3', 'F4'], cfg, 'center_EEG')
    second = diagnostic.local_eeg_feature(changed, ['F3', 'F4'], cfg, 'center_EEG')
    np.testing.assert_array_equal(first, second)
    assert np.isnan(first[52:68]).all()
    assert not np.allclose(diagnostic.local_eeg_feature(eeg, ['F3', 'F4'], cfg),
                           diagnostic.local_eeg_feature(changed, ['F3', 'F4'], cfg))


def test_baseline_operator_introduces_shared_noise_covariance():
    cfg, _, _, _ = diagnostic.load_config()
    operator = diagnostic.bridge_transform(np.eye(120, 3), 'baseline', cfg)
    np.testing.assert_allclose(operator[:20].mean(axis=0), 0., atol=1e-16)
    rng = np.random.default_rng(45)
    noise = rng.normal(size=(20000, 120))
    corrected = noise-noise[:, :20].mean(axis=1, keepdims=True)
    covariance = np.cov(corrected[:, [50, 90]], rowvar=False)
    assert covariance[0, 1] == pytest.approx(1/20, abs=.02)


def synthetic_trials():
    rng = np.random.default_rng(34)
    trials = []
    for session in range(3):
        for ordinal in range(8):
            eeg = rng.normal(size=(120, 3))
            fnirs = rng.normal(size=(120, 2, 2))
            local = rng.normal(size=120)
            views = {'target': dict(eeg_log_power=eeg, fnirs=fnirs, local_eeg=local)}
            for modality in ('EEG', 'fNIRS'):
                features = copy.deepcopy(views['target'])
                if modality == 'EEG':
                    features['eeg_log_power'][52:68] = np.nan
                    features['local_eeg'][52:68] = np.nan
                else:
                    features['fnirs'][52:68] = np.nan
                views['center_'+modality] = features
            trials.append(dict(views=views, eligible=np.ones(2, dtype=bool), session=str(session), ordinal=ordinal))
    return trials


def test_nested_cv_fits_no_transform_or_hyperparameter_to_outer_validation():
    cfg, base, measured, _ = diagnostic.load_config()
    trials = synthetic_trials()
    first = diagnostic.cv_fold_job(cfg, base, measured, trials, 'broadband_pca', 'EEG', 0)
    changed = copy.deepcopy(trials)
    for i in first['validation']:
        changed[i]['views']['target']['eeg_log_power'] *= 100.
        changed[i]['views']['target']['fnirs'] *= 100.
    second = diagnostic.cv_fold_job(cfg, base, measured, changed, 'broadband_pca', 'EEG', 0)
    assert first['selected_alphas'] == second['selected_alphas']
    for key in ('loading', 'fnirs_pair', 'eeg_factor', 'fnirs_factor', 'local_factor'):
        np.testing.assert_array_equal(first['projection'][key], second['projection'][key])
    for key in first['inner_losses']:
        np.testing.assert_array_equal(first['inner_losses'][key], second['inner_losses'][key])
    assert len(first['validation']) == 6
    assert set(first['validation']).isdisjoint(first['train'])


def test_linear_predictor_can_detect_fixed_lag_and_cannot_see_masked_target():
    cfg, _, _, _ = diagnostic.load_config()
    rng = np.random.default_rng(8)
    x, truth, shuffled = [], [], []
    for _ in range(40):
        eeg = rng.normal(size=120)
        y = np.column_stack((eeg, np.roll(eeg, 8), -.5*np.roll(eeg, 8)))
        masked = y.copy()
        masked[52:68, 1:] = np.nan
        _, joint, hidden, cols = diagnostic.linear_features(masked, np.zeros_like(y), 'fNIRS', cfg)
        x.append(joint)
        truth.append(y[hidden][:, cols])
        _, null, _, _ = diagnostic.linear_features(masked, np.zeros_like(y), 'fNIRS', cfg, np.roll(y, 60, axis=0))
        shuffled.append(null)
    model = diagnostic.ridge_fit(np.concatenate(x[:30]), np.concatenate(truth[:30]), .1)
    error = np.mean((diagnostic.ridge_predict(model, np.concatenate(x[30:]))-np.concatenate(truth[30:]))**2)
    null_error = np.mean((diagnostic.ridge_predict(model, np.concatenate(shuffled[30:]))-np.concatenate(truth[30:]))**2)
    assert error < .03
    assert null_error > .5


def test_review_does_not_turn_flat_or_incomplete_curves_into_parameter_estimates():
    from experiments.scripts import review_step5_observation_diagnostic as review
    cfg, _, _, _ = diagnostic.load_config()
    rows = [dict(w=float(w), parameter_log_likelihood=20.+1e-13*w)
            for w in np.linspace(-.5, .5, 17)]
    summary = dict(curves={'subject_01': {'broadband_pca': {
        'EEG_only': dict(rows=rows), 'fNIRS_only': dict(rows=rows[4:])}}})
    result = review.audited_curves(cfg, summary)['subject_01']['broadband_pca']
    assert result['EEG_only']['interpretation'] == 'flat_no_W_information'
    assert result['fNIRS_only']['interpretation'] == 'incomplete_curve'
    assert all(r['likelihood_maximum_w'] is None for r in result.values())


def test_historical_numerical_reviewer_still_records_legacy_saturation(monkeypatch):
    from experiments.scripts import review_step5_observation_diagnostic as review
    _, base, _, _ = diagnostic.load_config()
    core = diagnostic.step5.core
    # Preserve this regression of the old reviewer with an explicit legacy
    # failure fixture; production now accepts mathematically valid saturation.
    def extraction(f, e0):
        if -np.expm1(np.log1p(-e0)/f) == 1.:
            raise FloatingPointError('oxygen extraction left its strict physical domain')
        raise AssertionError('fixture only covers the retained saturation failure')

    monkeypatch.setattr(core, '_extraction', extraction)

    def saturated_transition(*args):
        core._extraction(.003, .32)

    monkeypatch.setattr(core, 'rk4_transition_with_jacobian', saturated_transition)
    result = review.replay_failure(np.zeros((2, 3)), base, -.5)
    assert result['execution'] == 'failed'
    assert result['event_relative_time_s'] == pytest.approx(-4.75)
    failure = result['diagnostic']['extraction_failure']
    assert failure['computed_E'] == 1.
    assert failure['one_minus_E'] > 0.
    assert core._extraction is extraction
    assert core.rk4_transition_with_jacobian is saturated_transition
