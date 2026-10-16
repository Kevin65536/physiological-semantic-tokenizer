"""Small, non-native fixtures for the train-frozen spatial runner."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.scripts import evaluate_shared_driver_spatial as runner


@pytest.fixture
def cfg():
    return runner.read_config()


@pytest.mark.parametrize('keys,value', [
    (('data', 'datasets'), ['protected_test']),
    (('data', 'protected_data'), 'allowed'),
    (('data', 'cache_root'), 'data/cache/another'),
    (('data', 'geometry_root'), 'unverified_geometry'),
    (('data', 'disjoint_optodes'), False),
    (('data', 'max_patch_angle_deg'), 90.),
    (('tensor', 'eeg_feature_order'), 'band_major'),
    (('evaluation', 'target_pair'), 1),
    (('evaluation', 'center_steps'), [40, 80]),
    (('evaluation', 'wrong_EEG'), 'shuffle_any_subject'),
    (('model', 'modes'), [1, 2, 3]),
    (('model', 'network_edges'), 'complete'),
    (('solver', 'validation_substeps'), 4),
    (('solver', 'max_nfev'), 0),
    (('resources', 'numerical_threads'), 2),
    (('calibration', 'eeg_loading_ridge_definition'), 'absolute'),
])
def test_config_rejects_information_or_model_contract_drift(cfg, tmp_path, keys, value):
    changed = deepcopy(cfg)
    owner = changed
    for key in keys[:-1]:
        owner = owner[key]
    owner[keys[-1]] = value
    path = tmp_path/'changed.yaml'
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError):
        runner.read_config(path)


def test_default_check_reads_no_signal_or_metadata_payload(cfg, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail('check stage must not load data')
    monkeypatch.setattr(runner.np, 'load', forbidden)
    assert runner.main([]) == 0
    assert '"measured_arrays_read": 0' in capsys.readouterr().out
    assert runner.dryrun(cfg)['planned_cells'] == 2*(7*3*3*5+10+4)


def test_seven_registered_arms_and_nested_design_coordinate_prefixes(cfg):
    arms = runner.arm_specs(cfg)
    assert len(arms) == 7
    assert all(a['K'] == 1 and a['geometry'] == 'true' for a in arms[5:])
    complete, small = arms[0], arms[-1]
    a = runner.spatial_design(cfg, runner._synthetic_patch(), complete)
    b = runner.spatial_design(cfg, runner._synthetic_patch(), small)
    np.testing.assert_array_equal(a['spatial_eeg'][:6], b['spatial_eeg'])
    assert np.linalg.norm(b['spatial_eeg']) < np.linalg.norm(a['spatial_eeg'])
    coordinate = dict(eeg_scale=np.arange(60).reshape(12, 5)+1., hb_scale=2.,
        eeg_shape=(12, 5), hb_shape=(4, 2), sd=np.arange(68)+1.)
    sliced = runner.panel_coordinate(coordinate, small)
    np.testing.assert_array_equal(sliced['sd'], np.r_[np.arange(30)+1., np.arange(60, 68)+1.])
    assert sliced['hb_scale'] == coordinate['hb_scale']


def test_fixed_weight_and_tied_cells_have_only_registered_main_panel_support(cfg):
    rows = [_row('train', 'train'), _row('selection', 'selection'), _row('evaluation', 'evaluation')]
    cells = runner.plan_manifest(cfg, dict(rows=rows))['cells']
    fixed = [r for r in cells if r['method'] == 'SSM_fixed1']
    tied = [r for r in cells if r['method'] == 'SSM_tied_fixed1']
    assert len(fixed) == 10 and len(tied) == 4
    assert all(r['arm'].endswith('H4_E12') and r['mode'] in ('heldout_pair', 'center_pair') and
               r['pairing'] == 'paired' for r in fixed+tied)
    assert all(r['arm'].startswith('K2_') for r in tied)


def test_masks_hide_the_target_reference_and_respect_center_context(cfg):
    arm = runner.arm_specs(cfg)[0]
    held = runner.visible_mask(cfg, 'heldout_pair', arm)
    assert not held[:, 60:62].any()
    assert held[:, 62:].all() and held[:, :60].all()
    center = runner.visible_mask(cfg, 'center_pair', arm, pairing='Hb_only')
    assert not center[:, :60].any()
    assert center[:48, 60:62].all() and not center[48:72, 60:62].any()
    assert center[72:, 60:62].all()
    endpoint = runner.endpoint_mask(cfg, 'center_pair')
    assert endpoint.sum() == 24*2


def _record(steps=40):
    rng = np.random.default_rng(172)
    return dict(eeg=rng.normal(size=(steps, 12, 5)), hb=rng.normal(size=(steps, 4, 2)),
        eeg_operator=np.eye(steps), hb_operator=np.eye(steps))


def _coordinate():
    return dict(eeg_scale=np.ones((12, 5)), hb_scale=2., eeg_shape=(12, 5), hb_shape=(4, 2),
        sd=np.ones(68))


@pytest.mark.parametrize('poison', [np.nan, np.inf, 1e30])
def test_raw_hidden_poison_is_removed_before_transform_and_fit(cfg, monkeypatch, poison):
    arm, coordinate = runner.arm_specs(cfg)[0], _coordinate()
    record = _record()
    design = runner.spatial_design(cfg, runner._synthetic_patch(), arm)
    loading = np.repeat(design['spatial_eeg'], 5, axis=0)
    captured = []
    def capture(eeg, hb, *args, **kwargs):
        captured.append((eeg.copy(), hb.copy(), deepcopy(kwargs)))
        return dict(success=False, status='failed_numerical')
    monkeypatch.setattr(runner, 'fit_spatial_shared_driver', capture)
    runner.fit_record(cfg, record, coordinate, arm, design, loading, 'heldout_pair', 'Hb_only', 1.)
    poisoned = deepcopy(record)
    poisoned['hb'][:, 0] = poison
    poisoned['eeg'][:] = poison
    runner.fit_record(cfg, poisoned, coordinate, arm, design, loading, 'heldout_pair', 'Hb_only', 1.)
    for a, b in zip(captured[0][:2], captured[1][:2]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(captured[0][2]['hb_gains'], np.ones(4))
    assert captured[0][2]['eeg_weight'] == 0.


def test_wrong_EEG_never_transforms_or_validates_its_Hb(cfg, monkeypatch):
    arm, coordinate = runner.arm_specs(cfg)[0], _coordinate()
    record, wrong = _record(), _record()
    wrong['hb'][:] = np.inf
    design = runner.spatial_design(cfg, runner._synthetic_patch(), arm)
    def capture(eeg, hb, *args, **kwargs):
        np.testing.assert_array_equal(eeg, wrong['eeg'].reshape(40, 60))
        assert np.isnan(hb[:, 0]).all() and np.isfinite(hb[:, 1:]).all()
        return dict(success=False, status='failed_numerical')
    monkeypatch.setattr(runner, 'fit_spatial_shared_driver', capture)
    runner.fit_record(cfg, record, coordinate, arm, design,
        np.repeat(design['spatial_eeg'], 5, axis=0), 'heldout_pair', 'wrong_EEG', 1., wrong=wrong)


def test_selected_zero_weight_masks_paired_EEG_before_any_transform(cfg, monkeypatch):
    arm, coordinate, record = runner.arm_specs(cfg)[0], _coordinate(), _record()
    record['eeg'][:] = np.inf
    design = runner.spatial_design(cfg, runner._synthetic_patch(), arm)
    def capture(eeg, hb, *args, **kwargs):
        assert np.isnan(eeg).all() and not kwargs['visible'][:, :60].any()
        return dict(success=False, status='failed_numerical')
    monkeypatch.setattr(runner, 'fit_spatial_shared_driver', capture)
    runner.fit_record(cfg, record, coordinate, arm, design,
        np.repeat(design['spatial_eeg'], 5, axis=0), 'heldout_pair', 'paired', 0.)


@pytest.mark.parametrize('poison', [np.nan, np.inf, 1e30])
def test_ridge_target_poison_and_Hb_only_EEG_poison_cannot_change_prediction(cfg, poison):
    rng = np.random.default_rng(11)
    training = [(rng.normal(size=(40, 60)), rng.normal(size=(40, 4, 2))) for _ in range(2)]
    models = runner.fit_baselines(training, cfg, ['train0', 'train1'])
    arm, record, coordinate = runner.arm_specs(cfg)[0], _record(), _coordinate()
    mask = runner.visible_mask(cfg, 'heldout_pair', arm, steps=40, pairing='Hb_only')
    e, h = runner.masked_panel_arrays(record, coordinate, arm, mask)
    original = runner.predict_baseline(models, e, h, 'joint_ridge', 'Hb_only')
    record['hb'][:, 0] = poison
    record['eeg'][:] = poison
    e, h = runner.masked_panel_arrays(record, coordinate, arm, mask)
    np.testing.assert_array_equal(runner.predict_baseline(models, e, h, 'joint_ridge', 'Hb_only'), original)


def _row(identity, split='evaluation', **extra):
    return dict(id=identity, dataset='fixture', subject=split, record=identity, split=split,
        task='same', condition='same', eeg_start_s=0., hb_start_s=0., window_seconds=30.,
        patch=runner._synthetic_patch(), **extra)


def test_wrong_EEG_requires_matched_subject_condition_montage_and_both_clocks_nonoverlap():
    a, b = _row('a'), _row('b')
    assert runner.wrong_eeg_partners([a, b]) == {'a': 'b', 'b': 'a'}
    for field, value in [('subject', 'another'), ('task', 'other'), ('condition', 'other')]:
        changed = dict(b, **{field: value})
        assert runner.wrong_eeg_partners([a, changed])['a'] is None
    changed = deepcopy(b)
    changed['patch']['eeg_channel_names'][0] = 'different'
    assert runner.wrong_eeg_partners([a, changed])['a'] is None
    for es, hs in [(29., 30.), (30., 29.), (29., 29.)]:
        same_record = dict(b, record=a['record'], eeg_start_s=es, hb_start_s=hs)
        assert runner.wrong_eeg_partners([a, same_record])['a'] is None
    separate = dict(b, record=a['record'], eeg_start_s=30., hb_start_s=30.)
    assert runner.wrong_eeg_partners([a, separate])['a'] == 'b'


def test_selection_pools_subjects_before_windows_and_keeps_failures():
    rows = []
    for weight, a, b in [(0., 0., 10.), (.1, 3., 3.), (1., 0., 0.)]:
        rows.extend(dict(subject='many_windows', eeg_weight=weight, nmse=a, success=True, status='completed') for _ in range(9))
        rows.append(dict(subject='one_window', eeg_weight=weight, nmse=b, success=weight != 1.,
                         status='completed' if weight != 1. else 'failed_numerical'))
    selected = runner.choose_eeg_weight(rows, [0., .1, 1.], 100.)
    assert selected['selected_eeg_weight'] == .1
    assert selected['candidates'][0]['subject_mean_nmse'] == 5.
    assert selected['candidates'][2]['subject_mean_nmse'] == 50.
    rows = [dict(subject='s', eeg_weight=w, nmse=1., success=True, status='completed') for w in (0., .1, 1.)]
    assert runner.choose_eeg_weight(rows, [0., .1, 1.], 100.)['selected_eeg_weight'] == 0.


def test_own_context_interpolation_reads_only_visible_target():
    h = np.zeros((10, 4, 2))
    h[:, 0] = np.arange(10)[:, None]
    mask = np.ones((10, 2), bool)
    mask[3:7] = False
    expected = runner.predict_own_context(np.ones((10, 2)), h, mask)
    h[3:7, 0] = np.inf
    np.testing.assert_array_equal(runner.predict_own_context(np.ones((10, 2)), h, mask), expected)
    np.testing.assert_array_equal(runner.predict_own_context(np.ones((10, 2)), h, np.zeros_like(mask)), np.ones((10, 2)))


def test_plan_retains_unavailable_inventory_and_rejects_subject_leak(cfg):
    inventory = dict(status='unavailable', rows=[], excluded=[dict(reason='no_verified_optodes')])
    plan = runner.plan_manifest(cfg, inventory)
    assert plan['status'] == 'unavailable' and plan['planned_cells'] == 0
    assert plan['excluded'] == inventory['excluded']
    rows = [_row('a', 'train'), _row('b', 'selection'), _row('c', 'evaluation')]
    assert runner.plan_manifest(cfg, dict(rows=rows))['status'] == 'available'
    rows[-1]['subject'] = 'train'
    assert runner.plan_manifest(cfg, dict(rows=rows))['status'] == 'unavailable'


def test_primary_success_requires_finer_integration(cfg, monkeypatch):
    arm, coordinate, record = runner.arm_specs(cfg)[0], _coordinate(), _record()
    design = runner.spatial_design(cfg, runner._synthetic_patch(), arm)
    fit = dict(success=True, status='completed', coefficients=np.zeros((6, 1)),
        initial_coordinates=np.zeros((1, 5)), prediction=np.zeros((40, 68)))
    monkeypatch.setattr(runner, 'fit_spatial_shared_driver', lambda *a, **k: deepcopy(fit))
    monkeypatch.setattr(runner, 'forward_spatial_shared_driver',
                        lambda *a, **k: dict(prediction=np.full((40, 68), .006)))
    result = runner.fit_record(cfg, record, coordinate, arm, design,
        np.repeat(design['spatial_eeg'], 5, axis=0), 'heldout_pair', 'paired', 1.)
    assert not result['success'] and result['status'] == 'failed_integration'


def test_tied_real_core_fit_replays_effective_single_core_without_changing_readout(cfg):
    row, record = runner.synthetic_record(cfg, 'identical_sources', 'evaluation', 0, smoke=True)
    arm = next(a for a in runner.arm_specs(cfg) if a['K'] == 2 and a['geometry'] == 'true')
    design = runner.spatial_design(cfg, row['patch'], arm)
    coordinate = _coordinate()
    coordinate['hb_scale'] = 1.
    coordinate['sd'] = np.maximum(np.column_stack((record['eeg'].reshape(40, 60), record['hb'].reshape(40, 8))).std(axis=0), 1e-6)
    result = runner.fit_record(cfg, record, coordinate, arm, design, record['truth_eeg_loadings'],
        'heldout_pair', 'paired', 1., tied=True)
    assert result['success'] and result['status'] == 'completed'
    assert result['coefficients'].shape == (6, 1) and result['states'].shape == (40, 1, 6)
    assert result['regularization_multiplicity'] == 2
    assert result['integration_max_training_sd'] <= cfg['solver']['integration_tolerance_training_sd']


def test_subject_scores_square_root_after_NMSE_pooling_and_bootstrap():
    rows = [dict(subject='s1', status='completed', success=True, nmse=0.),
            dict(subject='s1', status='completed', success=True, nmse=8.),
            dict(subject='s2', status='failed_numerical', success=False)]
    result = runner.subject_endpoint(rows, 100.)
    assert result['subject_nrmse_with_failure_penalty'] == {'s1': 2., 's2': 10.}
    assert result['mean_subject_nrmse_with_failure_penalty'] == 6.
    assert result['success_subset_mean_subject_nrmse'] == 2.
    boot = runner.paired_bootstrap({'s1': 1., 's2': 4.}, {'s1': 4., 's2': 9.}, repeats=30)
    assert boot['mean_nrmse_improvement'] == 1.
    np.testing.assert_array_equal(boot['interval_95'], [1., 1.])


def test_unavailable_controls_do_not_become_failure_penalties_or_fake_benefit(cfg):
    base = dict(dataset='fixture', subject='s', record_id='window', mode='heldout_pair', scenario='mechanism')
    candidate = dict(base, status='completed', success=True, nmse=1.)
    reference = dict(base, status='unavailable_wrong_EEG', success=False)
    result = runner.compare_cells([candidate], [reference], cfg)
    assert result['status'] == 'not_estimable_no_common_available_control'
    assert result['mean_nrmse_improvement'] is None
    assert result['unavailable_pair_cells'] == 1 and result['common_available_cells'] == 0
    endpoint = runner.subject_endpoint([reference], 100.)
    assert endpoint['available'] == 0 and endpoint['subject_nmse_with_failure_penalty'] == {}
    numerical = dict(base, status='failed_numerical', success=False)
    result = runner.compare_cells([candidate], [numerical], cfg)
    assert result['mean_nrmse_improvement'] == 9.


def test_completed_status_and_finite_endpoint_are_required_for_success_count():
    rows = [dict(subject='s', status='failed_exception', success=True, nmse=1.),
            dict(subject='s', status='completed', success=True, nmse=np.inf)]
    result = runner.subject_endpoint(rows, 100.)
    assert result['succeeded'] == 0 and result['subject_nmse_with_failure_penalty']['s'] == 100.


def test_synthetic_twins_have_wrong_EEG_support_and_training_only_identity_coordinates(cfg, tmp_path):
    rows = []
    for split in ('training', 'selection', 'evaluation'):
        for record_index in (0, 1):
            row, arrays = runner.synthetic_record(cfg, 'two_sources', split, 0, smoke=True, record_index=record_index)
            row['split'] = 'train' if split == 'training' else split
            path = runner._prepared_path(tmp_path, row['id'])
            runner.write_npz(path, **arrays)
            runner.write_json(path.with_suffix('.json'), dict(status='completed', row=row))
            rows.append(row)
    partners = runner.wrong_eeg_partners(rows)
    assert all(partners.values())
    assert partners[rows[0]['id']] == rows[1]['id']
    a, b = (runner.load_prepared(tmp_path, row) for row in rows[:2])
    assert not np.array_equal(a['eeg'], b['eeg'])
    coordinate = runner.fit_coordinate(tmp_path, rows, cfg)
    np.testing.assert_array_equal(coordinate['eeg_scale'], np.ones((12, 5)))
    assert coordinate['hb_scale'] == 1.
    expected = np.concatenate([np.column_stack((x['eeg'].reshape(40, 60), x['hb'].reshape(40, 8))) for x in (a, b)]).std(axis=0)
    np.testing.assert_allclose(coordinate['sd'], expected)
    assert coordinate['training_ids'] == [r['id'] for r in rows[:2]]


def test_smoke_coordinate_uses_training_seed_and_rejects_evaluation_record(cfg):
    training_row, training_record = runner.synthetic_record(cfg, 'two_sources', 'training', 0, smoke=True)
    evaluation_row, evaluation_record = runner.synthetic_record(cfg, 'two_sources', 'evaluation', 0, smoke=True)
    coordinate = runner.smoke_coordinate(training_row, training_record)
    expected = np.column_stack((training_record['eeg'].reshape(40, 60), training_record['hb'].reshape(40, 8))).std(axis=0)
    np.testing.assert_array_equal(coordinate['sd'], np.maximum(expected, 1e-6))
    assert coordinate['training_ids'] == [training_row['id']]
    assert '|training|' in coordinate['training_ids'][0]
    assert coordinate['sd_reference_scope'] == 'complete_training_window_no_evaluation_statistics'
    evaluation_record['hb'][:, 0] = np.inf
    np.testing.assert_array_equal(runner.smoke_coordinate(training_row, training_record)['sd'], coordinate['sd'])
    with pytest.raises(ValueError, match='training'):
        runner.smoke_coordinate(evaluation_row, evaluation_record)


def test_smoke_passes_identical_training_SD_to_both_modes_despite_hidden_evaluation_poison(cfg, tmp_path, monkeypatch):
    cfg = deepcopy(cfg)
    cfg['synthetic']['scenarios'] = ['single_source']
    generate = runner.synthetic_record
    calls, fit_sd = [], []
    def poisoned_generator(cfg, scenario, split, repeat, **kwargs):
        calls.append(split)
        row, record = generate(cfg, scenario, split, repeat, **kwargs)
        if split == 'evaluation':
            record['hb'][:, 0] = np.inf
        return row, record
    monkeypatch.setattr(runner, 'synthetic_record', poisoned_generator)
    def capture_fit(cfg, record, coordinate, arm, *args, **kwargs):
        fit_sd.append(np.asarray(coordinate['sd']).copy())
        t, k = len(record['eeg']), arm['K']
        return dict(success=False, status='failed_numerical', prediction=np.zeros((t, 68)),
            drivers=np.zeros((t, k)), states=np.ones((t, k, 6)))
    monkeypatch.setattr(runner, 'fit_record', capture_fit)
    runner.synthetic_smoke(cfg, tmp_path)
    assert calls == ['training', 'evaluation'] and len(fit_sd) == 2
    np.testing.assert_array_equal(fit_sd[0], fit_sd[1])
    training_row, training_record = generate(cfg, 'single_source', 'training', 0, smoke=True)
    expected = runner.smoke_coordinate(training_row, training_record)
    np.testing.assert_array_equal(fit_sd[0], expected['sd'])
    retained = runner.read_json(tmp_path/'scenarios'/'single_source'/'coordinate.json')
    assert retained['training_ids'] == expected['training_ids']


def test_summaries_keep_mechanism_scenarios_separate(cfg, tmp_path):
    cells = []
    for scenario, value in [('one', 1.), ('two', 9.)]:
        spec = dict(id=scenario, dataset='synthetic', subject='same_subject', record_id=scenario,
            arm='K1_true_H4_E12', mode='heldout_pair', pairing='paired', method='SSM', scenario=scenario)
        runner.write_json(tmp_path/'cells'/(scenario+'.json'), dict(spec, status='completed', success=True, nmse=value))
        cells.append(spec)
    runner.write_json(tmp_path/'plan.json', dict(cells=cells))
    result = runner.summarize(cfg, tmp_path)
    assert result['mechanisms_pooled'] is False and len(result['groups']) == 2
    assert sorted(g['mean_subject_nrmse_with_failure_penalty'] for g in result['groups'].values()) == [1., 3.]


def test_truth_diagnostics_separate_driver_DC_and_each_state_coordinate():
    record = dict(eeg=np.zeros((40, 12, 5)), truth_drivers=np.zeros((40, 1)), truth_states=np.ones((40, 1, 6)))
    fit = dict(drivers=np.ones((40, 1)), states=np.ones((40, 1, 6))*2.)
    result = runner.truth_diagnostics(fit, record)
    driver = result['drivers'][0]
    assert driver['raw_rmse'] == 1. and driver['reference_dc_error'] == 1.
    assert driver['reference_centered_rmse'] == 0. and driver['reference_centered_nrmse'] is None
    states = result['states'][0]['coordinates']
    assert list(states) == ['r', 's', 'f', 'v', 'p', 'q']
    assert all(x['raw_rmse'] == 1. and x['reference_centered_nrmse'] is None for x in states.values())


def test_full_synthetic_namespaces_freeze_calibration_within_each_scenario(cfg, tmp_path, monkeypatch):
    cfg = deepcopy(cfg)
    cfg['synthetic']['scenarios'] = ['single_source', 'two_sources']
    for split in ('training', 'selection', 'evaluation'):
        cfg['synthetic'][split+'_repeats'] = 1
    original_generator, arm = runner.synthetic_record, runner.arm_specs(cfg)[0]
    monkeypatch.setattr(runner, 'arm_specs', lambda cfg: [arm])
    monkeypatch.setattr(runner, 'synthetic_record', lambda cfg, scenario, split, repeat, record_index=0:
        original_generator(cfg, scenario, split, repeat, smoke=True, record_index=record_index))
    calls = []
    def capture(worker, payloads, workers, out, stage):
        payloads = list(payloads)
        calls.append((stage, Path(out), payloads))
        if stage == 'calibrate':
            assert all({row['scenario'] for row in p[2]} == {Path(out).name} for p in payloads)
        return dict(stage=stage, terminal=len(payloads), noncompleted=0)
    monkeypatch.setattr(runner, 'parallel_stage', capture)
    def fixture_summary(cfg, out):
        plan = runner.read_json(Path(out)/'plan.json')
        return dict(execution='completed', planned=len(plan['cells']), terminal=len(plan['cells']),
                    status_counts={'completed': len(plan['cells'])})
    monkeypatch.setattr(runner, 'summarize', fixture_summary)
    result = runner.synthetic_stage(cfg, tmp_path)
    assert len(calls) == 4 and len(result['scenarios']) == 2
    assert result['execution'] == 'completed' and result['mechanisms_pooled'] is False
    assert result['calibration_scope'] == 'within_scenario_disjoint_seed_calibration_not_cross_mechanism_transfer'
    for scenario in cfg['synthetic']['scenarios']:
        child = tmp_path/'scenarios'/scenario
        coordinate = runner.read_json(child/'coordinate.json')
        assert all(scenario in identity and '|training|' in identity for identity in coordinate['training_ids'])
    assert result['planned'] == sum(s['planned'] for s in result['scenarios'])
    assert runner.summarize_synthetic(cfg, tmp_path)['planned'] == result['planned']


def test_real_core_training_selection_and_evaluation_on_temporary_fixture(cfg, tmp_path, monkeypatch):
    """Runs the actual nonlinear fitting path; no native data or oracle readout."""
    arm = runner.arm_specs(cfg)[0]
    monkeypatch.setattr(runner, 'arm_specs', lambda cfg: [arm])
    rows = []
    for split in ('training', 'selection', 'evaluation'):
        row, arrays = runner.synthetic_record(cfg, 'single_source', split, 0, smoke=True)
        row['split'] = 'train' if split == 'training' else split
        path = runner._prepared_path(tmp_path, row['id'])
        runner.write_npz(path, **arrays)
        runner.write_json(path.with_suffix('.json'), dict(status='completed', row=row))
        rows.append(row)
    coordinate = runner.fit_coordinate(tmp_path, rows, cfg)
    assert coordinate['training_ids'] == [rows[0]['id']]
    assert coordinate['training_subjects'] == [rows[0]['subject']]
    calibration = runner._calibrate_job((cfg, str(tmp_path), rows, arm))
    assert calibration['status'] == 'completed', calibration.get('error')
    assert len(calibration['selection_fits']) == 3
    assert calibration['evaluation_data_used'] is False
    assert calibration['physical_hb_gain'] == 1.
    plan = runner.plan_manifest(cfg, dict(rows=rows))
    # One registered information condition suffices to exercise both the SSM
    # and every independently trained baseline in this short integration test.
    plan['cells'] = [c for c in plan['cells'] if c['mode'] == 'heldout_pair' and c['pairing'] == 'paired']
    plan['planned_cells'] = len(plan['cells'])
    runner.write_json(tmp_path/'plan.json', plan)
    results = [runner._evaluate_job((cfg, str(tmp_path), rows[-1], c)) for c in plan['cells']]
    assert len(results) == 6
    assert all(r['status'] == 'completed' for r in results)
    assert all(r['status'] in runner.TERMINAL for r in results)
    summary = runner.summarize(cfg, tmp_path)
    assert summary['execution'] == 'completed' and summary['planned'] == summary['terminal'] == 6
    assert summary['scientific_verdict'] == 'not_assigned'
    # Resume reuses a terminal result rather than rerunning a failed fit or
    # silently relearning readouts from evaluation arrays.
    def forbidden(*args, **kwargs):
        pytest.fail('terminal resume must not refit')
    monkeypatch.setattr(runner, 'fit_record', forbidden)
    assert runner._calibrate_job((cfg, str(tmp_path), rows, arm))['status'] == 'completed'
    assert runner._evaluate_job((cfg, str(tmp_path), rows[-1], plan['cells'][0]))['status'] in runner.TERMINAL
    # A failed SSM calibration does not disable independently trained ridge.
    calibration['status'] = 'failed_calibration'
    runner.write_json(tmp_path/'calibration'/(arm['id']+'.json'), calibration)
    spec = dict(next(c for c in plan['cells'] if c['method'] == 'joint_ridge'), id='fresh_ridge')
    assert runner._evaluate_job((cfg, str(tmp_path), rows[-1], spec))['status'] == 'completed'


def test_zero_EEG_readout_does_not_disable_zero_weight_Hb_only_or_baselines(cfg, tmp_path, monkeypatch):
    arm, rows = runner.arm_specs(cfg)[0], []
    monkeypatch.setattr(runner, 'arm_specs', lambda cfg: [arm])
    for split in ('training', 'selection', 'evaluation'):
        row, arrays = runner.synthetic_record(cfg, 'single_source', split, 0, smoke=True)
        row['split'] = 'train' if split == 'training' else split
        arrays['eeg'][:] = 0.
        path = runner._prepared_path(tmp_path, row['id'])
        runner.write_npz(path, **arrays)
        runner.write_json(path.with_suffix('.json'), dict(status='completed', row=row))
        rows.append(row)
    coordinate = runner.fit_coordinate(tmp_path, rows, cfg)
    assert np.asarray(coordinate['sd_floor_applied'])[:60].all()
    calibration = runner._calibrate_job((cfg, str(tmp_path), rows, arm))
    assert calibration['status'] == 'completed' and calibration['eeg_readout_status'] == 'unavailable'
    assert 'eeg_loadings' not in calibration['readout']
    assert calibration['selected_eeg_weight'] == 0.
    assert [r['status'] for r in calibration['selection_fits']] == ['completed', 'failed_calibration', 'failed_calibration']
    plan = runner.plan_manifest(cfg, dict(rows=rows))
    runner.write_json(tmp_path/'plan.json', plan)
    for pairing, method in [('paired', 'SSM'), ('Hb_only', 'SSM'), ('paired', 'joint_ridge'), ('paired', 'SSM_fixed1')]:
        spec = next(c for c in plan['cells'] if c['mode'] == 'heldout_pair' and c['pairing'] == pairing and c['method'] == method)
        result = runner._evaluate_job((cfg, str(tmp_path), rows[-1], spec))
        assert result['status'] == ('failed_calibration' if method == 'SSM_fixed1' else 'completed')
        if method == 'SSM':
            assert result['eeg_ignored'] and result['eeg_readout_status'] == 'unavailable'


def test_expensive_stage_requires_durable_supervisor_before_data_access(cfg, tmp_path, monkeypatch):
    for key in ('SYSTEMD_EXEC_PID', 'INVOCATION_ID', 'SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID'):
        monkeypatch.delenv(key, raising=False)
    out = tmp_path/cfg['artifact_root']/'fixture'
    with pytest.raises(SystemExit):
        runner.main(['--stage', 'prepare', '--project-root', str(tmp_path), '--output', str(out)])
    assert not out.exists()


def test_check_and_dryrun_are_neutral_before_synthetic_family(cfg, tmp_path, monkeypatch):
    out = tmp_path/cfg['artifact_root']/'fixture'
    common = ['--project-root', str(tmp_path), '--output', str(out)]
    assert runner.main(['--stage', 'dryrun', *common]) == 0
    assert runner.read_json(out/'run_identity.json')['family'] == 'neutral'
    monkeypatch.setattr(runner, 'synthetic_smoke', lambda cfg, out: dict(status='completed', measured_arrays_read=0))
    assert runner.main(['--stage', 'synthetic-smoke', *common]) == 0
    assert runner.read_json(out/'run_identity.json')['family'] == 'synthetic-smoke'


def test_source_snapshot_retains_native_cache_and_alignment_read_dependencies(tmp_path):
    runner.snapshot_sources(tmp_path, runner.DEFAULT_CONFIG)
    manifest = {entry['path']: entry for entry in runner.read_json(tmp_path/'source_snapshot'/'files.json')}
    for dependency in ('src/data/clean_physiology_cache.py', 'src/data/event_alignment.py'):
        assert dependency in manifest
        retained = tmp_path/manifest[dependency]['retained']
        assert retained.read_bytes() == (runner.CODE_ROOT/dependency).read_bytes()


def test_execution_history_preserves_each_stage_command_resources_and_supervisor(cfg, tmp_path, monkeypatch):
    out = tmp_path/cfg['artifact_root']/'history_fixture'
    common = ['--project-root', str(tmp_path), '--output', str(out)]
    previous = {}
    monkeypatch.setattr(runner.sys, 'argv', ['-'])
    for index, stage in enumerate(('dryrun', 'check', 'dryrun'), start=1):
        monkeypatch.setattr(runner, 'resource_metadata', lambda index=index: dict(resource_sample=index))
        monkeypatch.setattr(runner, 'supervisor_metadata', lambda index=index: dict(INVOCATION_ID=f'fixture_{index}'))
        argv = ['--stage', stage, '--workers', str(index), *common]
        assert runner.main(argv) == 0
        latest = runner.read_json(out/'execution.json')
        history_path = out/latest['history_path']
        assert runner.read_json(history_path) == latest
        assert latest['status'] == 'completed' and latest['stage'] == stage
        assert latest['command'][-len(argv):] == argv
        assert latest['argv'] == argv and latest['command'][1] != '-'
        assert latest['resources'] == dict(resource_sample=index)
        assert latest['supervisor'] == dict(INVOCATION_ID=f'fixture_{index}')
        assert len(list((out/'executions').glob('*.json'))) == index
        for path, content in previous.items():
            assert path.read_bytes() == content
        previous[history_path] = history_path.read_bytes()


def test_failed_stage_also_retains_its_execution_record(cfg, tmp_path, monkeypatch):
    out = tmp_path/cfg['artifact_root']/'failed_history_fixture'
    def fail(cfg):
        raise ValueError('fixture stage failure')
    monkeypatch.setattr(runner, 'dryrun', fail)
    with pytest.raises(ValueError, match='fixture stage failure'):
        runner.main(['--stage', 'dryrun', '--project-root', str(tmp_path), '--output', str(out)])
    latest = runner.read_json(out/'execution.json')
    assert latest['status'] == 'failed' and latest['error'] == 'fixture stage failure'
    assert runner.read_json(out/latest['history_path']) == latest
