"""Data-independent contracts for complete-inventory Tak scheduling and folds."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.scripts import evaluate_tak_mismatch as runner


@pytest.fixture
def cfg():
    return runner.read_config()


@pytest.mark.parametrize('section,key,value', [
    ('data', 'protected_data', 'allowed'),
    ('data', 'cache_root', 'data/cache/other'),
    ('data', 'folds', 3),
    ('tensor', 'dt_s', .25),
    ('fixed', 'viscoelastic_s', 0.),
    ('model', 'fitted_physiological_parameters', ['tau']),
    ('spatial', 'equal_channels', 12),
    ('coordinate', 'candidate_scale', 'per_window'),
    ('resources', 'numerical_threads', 2),
])
def test_config_rejects_scientific_scope_drift(cfg, tmp_path, section, key, value):
    altered = deepcopy(cfg); altered[section][key] = value
    path = tmp_path/'changed.yaml'; path.write_text(yaml.safe_dump(altered))
    with pytest.raises(ValueError): runner.read_config(path)


def test_default_check_cannot_open_arrays(monkeypatch, capsys):
    monkeypatch.setattr(runner.np, 'load', lambda *a, **k: pytest.fail('array read during check'))
    assert runner.main([]) == 0
    assert '"measured_arrays_read": 0' in capsys.readouterr().out


def _pair(name='pair'):
    return dict(id=name, native_index=0, channel_sets=dict(nearest6=list(range(6)),
        sixth_to_seventh=[0, 1, 2, 3, 4, 6], next6=list(range(6, 12)), random6=[1, 3, 5, 7, 9, 11]))


def _window(identity='w', start=0., context=(0., 90.), **extra):
    return dict(id=identity, start_s=start, eeg_start_s=start, hb_start_s=start,
        eeg_filter_start_s=context[0], eeg_filter_end_s=context[1],
        actual_context_s=context[1]-context[0], condition='same', task='task', **extra)


def _record(subject='s', fold=0, windows=None):
    return dict(id=f'record-{subject}', dataset='eeg_fnirs_single_trial', subject=subject,
        record=f'native-{subject}', fold=fold, windows=windows or [_window()], pairs=[_pair()])


def _arrays(seed=0, scale=1.):
    rng = np.random.default_rng(seed)
    eeg = rng.normal(size=(1, 300, 12, 5)); hb = rng.normal(size=(1, 300, 1, 2))*scale
    eeg -= eeg[:, :50].mean(1, keepdims=True); hb -= hb[:, :50].mean(1, keepdims=True)
    return dict(eeg_logpower=eeg, hb=hb, hb_bin_phase_s=np.array([.037]))


def test_unique_full_population_tasks_include_unavailable_EEG_and_geometry():
    records = [_record(), _record('bad', 1)]
    records[1]['windows'] = [
        dict(id='hb-only', start_s=None, eeg_start_s=None, hb_start_s=30.,
             eeg_filter_start_s=None, eeg_filter_end_s=None, actual_context_s=0.)]
    records[1]['pairs'][0]['channel_sets'] = {}
    tasks = list(runner.task_manifest(dict(records=records)))
    assert len(tasks) == 2 and len({r['id'] for r in tasks}) == 2
    assert tasks[1]['hb_start_s'] == 30. and tasks[1]['wrong_window'] is None
    assert tasks[1]['parameter_mode'] == 'fixed'


def test_wrong_control_matches_actual_context_not_just_target_windows():
    a = _window('a', context=(0., 90.))
    b = _window('b', start=90., context=(89., 179.))
    assert runner.wrong_eeg_partners([a, b]) == {'a': None, 'b': None}
    b['eeg_filter_start_s'] = 90.
    assert runner.wrong_eeg_partners([a, b]) == {'a': 'b', 'b': 'a'}
    b['condition'] = 'different'
    assert runner.wrong_eeg_partners([a, b])['a'] is None
    b['condition'] = a['condition']; b['wrong_pairing_eligible'] = False
    assert runner.wrong_eeg_partners([a, b])['a'] is None


def test_hb_sample_phase_is_in_model_operator_and_never_interpolates_measurements():
    from src.inference.observation_baselines import disjoint_native_bin_operator
    phase = .037
    expected = disjoint_native_bin_operator(np.arange(300)*.1+phase, steps=300, dt=.1, reference_steps=50)
    operator = runner.reference_operator(hb_phase_s=phase).reshape(300, 3, 300, 3)
    np.testing.assert_allclose(operator[:, 1, :, 1], expected['model'], atol=1e-12)
    np.testing.assert_allclose(operator[:, 0, :, 0], expected['eeg'], atol=1e-12)
    assert not operator[:, 0, :, 1].any()


def test_training_statistics_preserve_Hb_ratio_and_share_nearest_PCA_factor(cfg):
    record = _record(); arrays = _arrays()
    stats = runner._statistics(arrays, record)
    coord = runner.fit_coordinate_statistics({k: v[0] for k, v in stats.items()}, cfg)
    assert coord['status'] == 'completed' and coord['common_Hb_scale']
    assert not coord['candidate_output_renormalized']
    sd = np.asarray(coord['sd'])
    assert sd[0] == pytest.approx(.025)
    for pc in coord['pc'].values(): assert np.linalg.norm(pc) == pytest.approx(1.)
    p0 = runner.proxy_for_window(arrays, 0, record['pairs'][0], coord, 'nearest6')
    p1 = runner.proxy_for_window(arrays, 0, record['pairs'][0], coord, 'next6')
    assert np.isfinite(p0).all() and np.isfinite(p1).all()
    assert np.std(p0[:-1]) == pytest.approx(.025, rel=1e-12)
    # Other candidate output scales are not individually refitted to .025.
    assert abs(np.std(p1[:-1])-.025) > 1e-6


def test_outer_holdout_cannot_change_its_training_coordinate(cfg, tmp_path):
    records = [_record(f's{i}', i) for i in range(5)]
    plan = dict(records=records, subject_folds={'eeg_fnirs_single_trial': {f's{i}': i for i in range(5)}})
    def build(out, holdout_scale):
        for i, record in enumerate(records):
            metadata, _, statistics = runner.prepared_paths(out, record['id'])
            runner.write_json(metadata, dict(status='completed'))
            runner.write_npz(statistics, **runner._statistics(_arrays(i, holdout_scale if i == 0 else 1.), record))
        return runner.coordinate_stage(cfg, out, plan)['models']['eeg_fnirs_single_trial|pair']['0']
    first = build(tmp_path/'first', 1.)
    poisoned = build(tmp_path/'second', 1e8)
    assert first == poisoned
    assert 's0' not in first['training_subjects']
    assert set(first['inner_training_subjects']).isdisjoint(first['selection_subjects'])


def test_spatial_selection_counts_failures_and_equal_subjects():
    rows = []
    for candidate, value in zip(runner.CANDIDATES, (1., .8, .5, .7)):
        for subject in ('a', 'b'):
            rows.append(dict(candidate=candidate, subject=subject, planned=4, successful=4,
                failed=0, nmse=value, window_scores={str(i): value for i in range(4)}))
    # An excellent successful subset cannot hide a larger failed denominator.
    next6 = [r for r in rows if r['candidate'] == 'next6'][0]
    next6.update(successful=3, failed=1)
    next6['window_scores']['0'] = None
    result = runner.select_spatial_candidate(rows)
    assert result['selected_candidate'] == 'random6'
    assert result['assessment']['next6']['failed'] == 1


def test_prediction_input_contains_no_target_Hb():
    proxy = np.arange(300.)
    target = runner.independent_eeg_input(proxy)
    assert np.isnan(target[:, 1:]).all() and np.isnan(target[-1]).all()
    np.testing.assert_array_equal(target[:-1, 0], proxy[:-1])


def test_terminal_failure_is_reused_without_array_access(cfg, tmp_path, monkeypatch):
    task = dict(id='failed')
    retained = dict(task, status='failed_exception', success=False, error='retained negative evidence')
    runner.write_json(tmp_path/'tasks/failed.json', retained)
    monkeypatch.setattr(runner, '_load_prepared', lambda *a: pytest.fail('must not refit terminal failure'))
    assert runner._task_job(cfg, tmp_path, task, {}, {}) == retained


def test_unavailable_preparation_retains_its_planned_identity(cfg, tmp_path):
    record = _record(); task = next(runner.task_manifest(dict(records=[record])))
    result = runner._task_job(cfg, tmp_path, task, record, {})
    assert result['status'] == 'unavailable_preparation'
    assert runner.read_json(tmp_path/'tasks'/f'{task["id"]}.json')['pair'] == 'pair'


def test_subject_aggregation_is_record_equal_and_retains_failed_denominator():
    rows = [dict(dataset='d', subject='s', record_id='long', D1=1.) for _ in range(10)]
    rows += [dict(dataset='d', subject='s', record_id='short', D1=3.),
             dict(dataset='d', subject='failed', record_id='bad', D1=None)]
    result = {r['subject']: r for r in runner.subject_aggregate(rows, 'D1')}
    assert result['s']['value'] == pytest.approx(np.sqrt(5.))
    assert result['s']['planned'] == 11 and result['s']['native_records'] == 2
    assert result['failed']['value'] is None and result['failed']['planned'] == 1


def test_examples_lock_true_maximum_and_retain_selection_denominator():
    rows = [dict(id=str(i), dataset='d', subject='same' if i > 3 else f's{i}', D1=float(i),
                 array_path=f'/tmp/{i}.npz') for i in range(10)]
    examples = runner.choose_examples(rows)['examples']
    assert [r['role'] for r in examples] == ['median', 'p90', 'max']
    assert examples[-1]['D1'] == 9. and examples[-1]['rank'] == 10
    assert all(r['candidate_denominator'] == 10 for r in examples)


def test_summary_preserves_all_missing_terminal_denominators(cfg, tmp_path):
    record = _record(); plan = dict(records=[record])
    result = runner.summarize_stage(cfg, tmp_path, plan)
    assert result['status'] == 'incomplete' and result['planned'] == result['missing'] == 1
    assert (tmp_path/'metrics.csv').is_file() and (tmp_path/'subject_summary.csv').is_file()
    assert result['by_dataset'][record['dataset']]['metrics']['D1']['available'] == 0


def test_preparation_weak_kernels_never_cross_native_session_boundary(cfg, tmp_path, monkeypatch):
    from src.data import tak_mismatch
    record = _record(); record['continuous_hb_segments'] = [dict(start_s=0., end_s=45.), dict(start_s=45., end_s=90.)]
    arrays = _arrays(); clock = np.arange(900)*.1
    arrays.update(continuous_hb=np.column_stack((np.sin(clock/5), -.3*np.sin(clock/5)))[:, None],
        continuous_hb_time_s=clock, continuous_hb_valid_mask=np.ones((900, 1, 2), bool), metadata={})
    monkeypatch.setattr(tak_mismatch, 'prepare_tak_mismatch_record', lambda *a, **k: arrays)
    monkeypatch.setattr(runner, '_metadata_index', lambda *a: None)
    result = runner._prepare_record_job((cfg, str(tmp_path), str(tmp_path/'out'), record))
    assert result['status'] == 'completed'
    with np.load(result['weak_path'], allow_pickle=False) as weak:
        centers, valid = weak['pair_0_10s_center_time_s'], weak['pair_0_10s_valid']
        assert not valid[(centers >= 40)&(centers <= 50)].any()
        assert valid[centers == 20].all()


def test_full_synthetic_window_preserves_four_axis_and_control_evidence(cfg, tmp_path):
    from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
    record = _record(); task = next(runner.task_manifest(dict(records=[record])))
    clock = np.arange(300)*.1; driver = .02*np.sin(2*np.pi*clock/15)
    truth = nonlinear_driver_forward(driver, np.r_[0., np.ones(4)], runner.parameters(cfg), .1,
        substeps=8, derivative=False, numerical_backend='numba', hemodynamics='tak', viscoelastic_s=2.)
    target = (runner.reference_operator()@truth['canonical_prediction'].ravel()).reshape(300, 3)
    features = np.zeros((1, 300, 12, 5)); features[0, :, 0, 0] = target[:, 0]
    metadata, array_path, _ = runner.prepared_paths(tmp_path, record['id'])
    runner.write_npz(array_path, eeg_logpower=features, hb=target[None, :, None, 1:], hb_bin_phase_s=np.zeros(1))
    runner.write_json(metadata, dict(status='completed'))
    coord = dict(pc=dict(nearest6=np.r_[1., np.zeros(29)]), eeg_factor=1., hb_factor=1.,
        sd=[.025, .025, .015], template=np.zeros((300, 2)))
    selection = dict(models={'eeg_fnirs_single_trial|pair': {'0': dict(coordinate=coord,
        selected_candidate='nearest6', training_subjects=['train'], selection_subjects=['selection'],
        selection_status='completed')}})
    result = runner._task_job(cfg, tmp_path, task, record, selection)
    assert result['status'] == 'completed', result.get('traceback')
    assert result['hb_only_success'] and result['eeg_only_success'] and result['joint_success']
    assert result['D1'] < .01 and result['D4'] < .01
    assert result['Delta_spatial'] == pytest.approx(0.)
    assert result['wrong_statuses']['wrong'] == 'unavailable_nonoverlap_donor'
    assert result['observable_primary_noise_scenario'] == 'primary_ar1_sd0p1'
    with np.load(result['array_path'], allow_pickle=False) as saved:
        assert saved['hb_only_prediction'].shape == (300, 3)
        assert saved['observable_standardized_hb_difference'].shape == (300, 2)
        assert saved['score_mask'].sum() == 249*3


def test_candidate_selection_rejects_disjoint_success_subsets():
    rows = []
    for subject in ('a', 'b'):
        for candidate in runner.CANDIDATES:
            scores = dict(first=1., second=1.)
            scores['first' if candidate == 'nearest6' else 'second'] = None
            rows.append(dict(subject=subject, record=subject, candidate=candidate, planned=2,
                successful=1, failed=1, nmse=1., window_scores=scores))
    result = runner.select_spatial_candidate(rows)
    assert result['selected_candidate'] is None
    assert all(r['paired_windows'] == 0 for r in result['assessment'].values())


def test_time_weight_counts_the_same_native_window_once_across_available_pairs(cfg):
    rows = [dict(record_id='r', window='w', D1=1., D_EH=.5),
            dict(record_id='r', window='w', D1=2., D_EH=None)]
    runner.add_time_weights(rows, cfg)
    assert sum(r['time_weight_D1_s'] for r in rows) == pytest.approx(24.9)
    assert [r['time_weight_D1_s'] for r in rows] == pytest.approx([12.45, 12.45])
    assert [r['time_weight_D_EH_s'] for r in rows] == pytest.approx([24.9, 0.])


def test_complete_summary_pairs_free_and_fixed_on_identical_windows(cfg, tmp_path):
    records = [_record('a', 0), _record('b', 1)]; plan = dict(records=records)
    fixed = []
    for i, task in enumerate(runner.task_manifest(plan)):
        row = dict(task, status='completed', success=True, D1=float(2*(i+1)), D4=.5)
        runner.write_json(tmp_path/'tasks'/f'{task["id"]}.json', row); fixed.append(row)
    for i, (_, group) in enumerate(runner.current_dof_groups(plan)):
        task = group['tasks'][0]
        row = dict(task, parameter_mode='current_dof', status='completed', success=True,
            D1_free=float(i+1), D4_free=.25, hb_only_parameters=dict(kappa=.64, tau=2.))
        runner.write_json(tmp_path/'current_dof/tasks'/f'{group["id"]}.json',
            dict(status='completed', rows=[row], weak_summary={'10s': dict(D_bal=.2, denominator=2.,
                planned_centers=20, valid_centers=20)}))
    summary = runner.summarize_stage(cfg, tmp_path, plan)
    assert summary['status'] == 'completed'
    paired = summary['paired_current_dof']['eeg_fnirs_single_trial']['D1']
    assert paired['paired_windows'] == paired['paired_subjects'] == 2
    assert paired['fixed_minus_free_subject_mean'] == pytest.approx(1.5)
    free = summary['by_dataset_current_dof']['eeg_fnirs_single_trial']['metrics']['D1_free']
    assert free['subject_equal_mean'] == pytest.approx(np.sqrt(2.5))
    assert (tmp_path/'weak_record_summary_current_dof.csv').exists()


def test_prepare_resource_pilot_round_robins_datasets_without_changing_full_plan(cfg, tmp_path, monkeypatch):
    records = []
    for dataset in cfg['data']['datasets']:
        for i in range(2):
            record = _record(f'{dataset}-{i}', i); record['dataset'] = dataset
            records.append(record)
    plan = dict(records=records); original = deepcopy(plan)
    def prepare(payload):
        record = payload[-1]
        return dict(record_id=record['id'], status='completed', success=True,
            elapsed_seconds=1., max_rss_kib=1024)
    monkeypatch.setattr(runner, '_prepare_record_job', prepare)
    result = runner.prepare_stage(cfg, tmp_path, tmp_path, plan, 1, max_records=4)
    assert plan == original and result['status'] == 'pilot_completed'
    assert len(result['record_ids']) == 4
    assert [next(r['dataset'] for r in records if r['id'] == identity) for identity in result['record_ids']] == cfg['data']['datasets']
    assert (tmp_path/'prepare_pilot_summary.json').exists() and not (tmp_path/'prepare_summary.json').exists()


def test_fixed_throughput_pilot_round_robins_four_datasets(cfg):
    records = []
    for dataset in cfg['data']['datasets']:
        record = _record(dataset, windows=[_window('w0'), _window('w1', start=30.)])
        record['dataset'] = dataset; records.append(record)
    selected = runner.pilot_task_manifest(cfg, dict(records=records), 8)
    assert len(selected) == len({r['id'] for r in selected}) == 8
    assert [r['dataset'] for r in selected[:4]] == cfg['data']['datasets']
    assert all(sum(r['dataset'] == dataset for r in selected) == 2 for dataset in cfg['data']['datasets'])


def test_weak_exception_cannot_mark_preparation_completed(cfg, tmp_path, monkeypatch):
    from src.data import tak_mismatch as data
    from src.inference import tak_mismatch as metrics
    arrays = _arrays(); arrays.update(continuous_hb=np.ones((900, 1, 2)),
        continuous_hb_time_s=np.arange(900)*.1, continuous_hb_valid_mask=np.ones((900, 1, 2), bool), metadata={})
    monkeypatch.setattr(data, 'prepare_tak_mismatch_record', lambda *a, **k: arrays)
    monkeypatch.setattr(runner, '_metadata_index', lambda *a: None)
    def fail(*args, **kwargs): raise RuntimeError('weak numerical failure')
    monkeypatch.setattr(metrics, 'weak_balance_residual', fail)
    record = _record(); result = runner._prepare_record_job((cfg, str(tmp_path), str(tmp_path/'out'), record))
    assert result['status'] == 'failed_preparation' and not result['success']
    assert 'weak numerical failure' in result['error']
    assert runner.prepared_paths(tmp_path/'out', record['id'])[1].exists()


def _audit_fixture(cfg, out):
    windows = [_window(f'w{i}', start=30.*i) for i in range(7)]
    record = _record(windows=windows); plan = dict(records=[record])
    runner.write_json(out/'plan.json', plan); runner.write_json(out/'resolved_config.json', cfg)
    runner.write_json(out/'source_manifest.json', dict(source_git_head='frozen-production-head',
        source_snapshot='source_snapshot', source_files=['frozen/runner.py']))
    runner.write_json(out/'launch.json', dict(supervisor='fixture.service'))
    tasks = list(runner.task_manifest(plan))
    success = dict(success=True, status='completed', solver_status='completed')
    for i, task in enumerate(tasks):
        row = dict(task, status='completed', view_statuses={view: deepcopy(success)
            for view in ('hb_only', 'eeg_only', 'joint')})
        if i == 0: row['D_obs_status'] = 'insufficient_observable_rank'
        if i == 1:
            row['view_statuses']['hb_only'] = dict(success=False, status='failed_numerical', solver_status='failed_line_search')
        elif i == 2:
            row.update(status='unavailable_complete_Hb_pair'); row.pop('view_statuses')
        elif i == 3:
            row['view_statuses'] = dict(hb_only=deepcopy(success))
            row['eeg_input_status'] = 'unavailable_native_EEG_or_geometry'
        elif i in (4, 5):
            row['status'] = 'failed_exception' if i == 4 else 'unavailable_preparation'
            row.pop('view_statuses')
        elif i == 6:
            row['view_statuses']['hb_only'] = dict(success=False, status='failed_numerical', solver_status='unavailable_data_support')
        runner.write_json(out/'tasks'/f'{task["id"]}.json', row)
    _, group = next(runner.current_dof_groups(plan))
    rows = []
    for i, task in enumerate(tasks):
        row = dict(task, parameter_mode='current_dof', status='partial_or_failed_fit',
            hb_only_success=i < 2, joint_success=False)
        if i < 2: row['hb_only_status'] = 'completed'
        elif i == 2: row['hb_only_status'] = 'failed_integration'
        rows.append(row)
    group_result = dict(id=group['id'], status='partial_or_unavailable', rows=rows,
        views=dict(hb_only=dict(planned=7, fitted=3, status='completed'),
                   joint=dict(planned=7, fitted=2, status='failed_iterations')))
    runner.write_json(out/'current_dof/tasks'/f'{group["id"]}.json', group_result)
    return plan, tasks, group, group_result


def test_terminal_audit_separates_optimizer_failures_from_support(cfg, tmp_path, monkeypatch):
    _, _, _, _ = _audit_fixture(cfg, tmp_path)
    monkeypatch.setattr(runner.np, 'load', lambda *a, **k: pytest.fail('status audit must not open arrays'))
    monkeypatch.setattr(runner, '_fit_views', lambda *a, **k: pytest.fail('status audit must not fit'))
    result = runner.terminal_status_audit_stage(cfg, tmp_path)
    fixed = result['by_dataset']['eeg_fnirs_single_trial']['fixed']['views']['hb_only']
    assert {name: fixed[name] for name in runner.AUDIT_CATEGORIES} == dict(
        completed=2, numerical_failure=1, unavailable_support=2, execution_failure=2, missing=0)
    free = result['by_dataset']['eeg_fnirs_single_trial']['current_dof']['views']['hb_only']
    assert free['planned'] == 7 and free['fitted'] == 3
    assert free['completed'] == 2 and free['numerical_failure'] == 1 and free['unavailable_support'] == 4
    assert free['rates']['numerical_failure_rate'] == dict(numerator=1, denominator=3, value=1/3)
    assert free['rates']['support_rate'] == dict(numerator=3, denominator=7, value=3/7)
    assert result['status'] == 'completed' and result['measured_arrays_read'] == result['fits_executed'] == 0
    empty = result['by_dataset']['refed']['fixed']['views']['hb_only']
    assert empty['rates']['numerical_failure_rate']['value'] is None
    assert result['production_sources']['source_git_head'] == 'frozen-production-head'
    assert 'def _audit_fixed_view' in result['audit_source']['implementation']
    assert result['by_dataset']['eeg_fnirs_single_trial']['fixed']['extra_statuses']['D_obs_status']['insufficient_observable_rank'] == 1


def test_terminal_audit_current_group_exception_is_not_no_support(cfg, tmp_path):
    _, _, group, retained = _audit_fixture(cfg, tmp_path)
    retained.update(status='failed_exception', views={})
    runner.write_json(tmp_path/'current_dof/tasks'/f'{group["id"]}.json', retained)
    result = runner.terminal_status_audit_stage(cfg, tmp_path)
    for view in result['by_dataset']['eeg_fnirs_single_trial']['current_dof']['views'].values():
        assert view['execution_failure'] == 7 and view['unavailable_support'] == view['numerical_failure'] == 0
    assert result['all_parameter_modes_terminal']


def test_terminal_audit_preserves_completed_view_before_later_group_exception(cfg, tmp_path):
    _, _, group, retained = _audit_fixture(cfg, tmp_path)
    retained['status'] = 'failed_exception'; retained['views'].pop('joint')
    runner.write_json(tmp_path/'current_dof/tasks'/f'{group["id"]}.json', retained)
    result = runner.terminal_status_audit_stage(cfg, tmp_path)
    section = result['by_dataset']['eeg_fnirs_single_trial']['current_dof']
    assert section['views']['hb_only']['completed'] == 2
    assert section['views']['joint']['execution_failure'] == 7
    assert section['group_status_counts']['failed_exception'] == 1


@pytest.mark.parametrize('missing_mode', ['fixed', 'current_dof'])
def test_terminal_audit_missing_results_remain_missing_without_artifact(cfg, tmp_path, missing_mode):
    _, tasks, group, _ = _audit_fixture(cfg, tmp_path)
    if missing_mode == 'fixed': (tmp_path/'tasks'/f'{tasks[-1]["id"]}.json').unlink()
    else: (tmp_path/'current_dof/tasks'/f'{group["id"]}.json').unlink()
    result = runner.terminal_status_audit_stage(cfg, tmp_path)
    section = result['by_dataset']['eeg_fnirs_single_trial'][missing_mode]
    expected = 1 if missing_mode == 'fixed' else 7
    assert section['views']['hb_only']['missing'] == expected
    assert result['status'] == 'incomplete' and not result['artifact_written']
    assert not (tmp_path/'terminal_status_audit.json').exists()


@pytest.mark.parametrize('mutation', ['too_many_fitted', 'success_above_fitted', 'missing_row'])
def test_terminal_audit_rejects_broken_group_denominators(cfg, tmp_path, mutation):
    _, _, group, retained = _audit_fixture(cfg, tmp_path)
    if mutation == 'too_many_fitted': retained['views']['hb_only']['fitted'] = 8
    elif mutation == 'success_above_fitted': retained['views']['hb_only']['fitted'] = 1
    else: retained['rows'].pop()
    runner.write_json(tmp_path/'current_dof/tasks'/f'{group["id"]}.json', retained)
    with pytest.raises(ValueError): runner.terminal_status_audit_stage(cfg, tmp_path)
    assert not (tmp_path/'terminal_status_audit.json').exists()


def test_terminal_audit_is_exclusive_and_leaves_original_evidence_unchanged(cfg, tmp_path):
    _, tasks, group, _ = _audit_fixture(cfg, tmp_path)
    paths = [tmp_path/'plan.json', tmp_path/'resolved_config.json', tmp_path/'source_manifest.json',
        tmp_path/'launch.json', tmp_path/'current_dof/tasks'/f'{group["id"]}.json',
        *(tmp_path/'tasks'/f'{task["id"]}.json' for task in tasks)]
    original = {path: path.read_bytes() for path in paths}
    runner.terminal_status_audit_stage(cfg, tmp_path)
    audit = (tmp_path/'terminal_status_audit.json').read_bytes()
    with pytest.raises(FileExistsError): runner.terminal_status_audit_stage(cfg, tmp_path)
    assert (tmp_path/'terminal_status_audit.json').read_bytes() == audit
    assert all(path.read_bytes() == content for path, content in original.items())


def test_terminal_audit_checks_frozen_inventory_summary(cfg, tmp_path):
    plan, _, _, _ = _audit_fixture(cfg, tmp_path)
    plan['summary'] = {'eeg_fnirs_single_trial': {'planned_pair_windows': 8}}
    runner.write_json(tmp_path/'plan.json', plan)
    with pytest.raises(ValueError, match='frozen inventory summary'):
        runner.terminal_status_audit_stage(cfg, tmp_path)
    assert not (tmp_path/'terminal_status_audit.json').exists()


def test_terminal_audit_cli_does_not_freeze_or_acquire_production_controller(cfg, tmp_path, monkeypatch, capsys):
    out = tmp_path/runner.ARTIFACT_ROOT/cfg['run_name']; _audit_fixture(cfg, out)
    def forbidden(*args, **kwargs): pytest.fail('read-only audit cannot mutate production controller/config')
    monkeypatch.setattr(runner, 'freeze_config', forbidden); monkeypatch.setattr(runner, 'run_lock', forbidden)
    assert runner.main(['--stage', 'audit-status', '--project-root', str(tmp_path)]) == 0
    assert '"status": "completed"' in capsys.readouterr().out
