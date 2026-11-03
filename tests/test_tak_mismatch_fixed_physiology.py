"""Read-only fixed-result workflow checks using temporary synthetic evidence."""
from copy import deepcopy
import gzip
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.scripts import evaluate_tak_mismatch as runner


def _record(name='session_00', dataset='eeg_fnirs_single_trial'):
    identity = f'{dataset}|s|{name}'
    windows = [dict(id=f'{identity}|w={i}', start_s=None, hb_start_s=i*30., eeg_start_s=None,
        eeg_filter_start_s=None, eeg_filter_end_s=None, actual_context_s=0., status='hb_only',
        task='motor_imagery' if i else None, condition=None, wrong_pairing_eligible=False,
        native_support_id=f'{identity}|support={i}', hb_support_admitted=True,
        segment_index=None, hb_segment_index=i) for i in range(2)]
    return dict(id=identity, join_key=identity, dataset=dataset, subject='s', record=name,
        status='available', fold=0, windows=windows, pairs=[dict(id='p1'), dict(id='p2')],
        continuous_hb_segments=[dict(id=w['native_support_id'], start_s=i*30., end_s=(i+1)*30.)
            for i,w in enumerate(windows)])


def _populate(parent, record, *, missing=()):
    tasks = list(runner.task_manifest(dict(records=[record])))
    for i, task in enumerate(tasks):
        if i in missing: continue
        value = dict(task, status='completed' if i else 'partial_or_failed_fit', success=True,
            hb_only_success=True, eeg_only_success=False, joint_success=False,
            D1=float(i+1), D4=None, zero_NMSE=1., array_path=str(parent/'tasks'/f'{task["id"]}.npz'),
            view_statuses=dict(hb_only=dict(success=True, status='completed', solver_status='completed')))
        runner.write_json(parent/'tasks'/f'{task["id"]}.json', value)
    metadata = dict(status='completed', weak_path=str(parent/'weak.npz'), weak_summary={
        '0': {'20s': dict(D_bal=.25, status='completed')},
        '1': {'20s': dict(D_bal=.75, status='completed')}})
    runner.write_json(runner.prepared_paths(parent, record['id'])[0], metadata)
    return tasks


def test_analysis_contract_is_exact_and_does_not_enable_fitting(tmp_path):
    cfg = runner.read_physiology_config()
    assert cfg['parameter_mode'] == 'fixed'
    assert cfg['scope']['scalar_pair_windows'] == 473088
    assert cfg['scope']['array_pair_windows'] == 38268
    assert cfg['scope']['model_fitting'] == cfg['scope']['parent_writes'] == 'none'
    drift = deepcopy(cfg); drift['scope']['protected_data'] = 'permitted'
    path = tmp_path/'drift.yaml'; path.write_text(yaml.safe_dump(drift))
    with pytest.raises(ValueError, match='contract drift'):
        runner.read_physiology_config(path)


def test_extract_keeps_failed_rows_record_task_fallback_and_original_clock(tmp_path, monkeypatch):
    parent, output = tmp_path/'parent', tmp_path/'analysis'
    record = _record(); tasks = _populate(parent, record, missing=(3,))
    before = {p: p.read_bytes() for p in (parent/'tasks').iterdir()}
    monkeypatch.setattr(runner.np, 'load', lambda *a, **k: pytest.fail('extract must not read arrays'))
    result = runner._physiology_extract_record((parent, output, record, {tasks[0]['id']}))
    frame = pd.read_csv(result['files']['metrics'])
    assert result['rows'] == result['planned'] == 4 and result['missing'] == 1
    assert result['status'] == 'incomplete' and result['measured_arrays_read'] == 0
    assert frame.task_label.eq('motor_imagery').all()
    assert frame.window_start_s.tolist() == [0., 0., 30., 30.]
    assert frame.panel_selected.tolist() == [True, False, False, False]
    assert frame.hb_only_success.tolist() == [True, True, True, False]
    assert frame.D1.notna().sum() == 3 and frame.D4.isna().all()
    assert frame.D_bal_20s.tolist() == [.25, .75, .25, .75]
    assert frame.loc[3, 'status'] == 'missing'
    weak = pd.read_csv(result['files']['weak'])
    assert len(weak) == 6  # Each record/pair/width appears once, not per window.
    assert all(path.read_bytes() == content for path, content in before.items())


def test_incomplete_extract_can_resume_without_fitting(tmp_path):
    record = _record(); parent, output = tmp_path/'parent', tmp_path/'analysis'
    tasks = _populate(parent, record, missing=(3,))
    runner._physiology_extract_record((parent, output, record, set()))
    runner.write_json(parent/'tasks'/f'{tasks[3]["id"]}.json', dict(tasks[3], status='unavailable_support', success=False))
    result = runner._physiology_extract_record((parent, output, record, set()))
    assert result['status'] == 'completed' and result['missing'] == 0
    assert pd.read_csv(result['files']['metrics']).D1.notna().sum() == 3


def test_extract_rejects_changed_fixed_identity(tmp_path):
    record = _record(); parent = tmp_path/'parent'; tasks = _populate(parent, record)
    runner.write_json(parent/'tasks'/f'{tasks[0]["id"]}.json', dict(tasks[0], parameter_mode='current_dof', status='completed'))
    with pytest.raises(ValueError, match='identity changed'):
        runner._physiology_extract_record((parent, tmp_path/'analysis', record, set()))


def test_native_repeat_metadata_keeps_visual_probes_synchronous_and_videos_distinct():
    a = runner._physiology_record_metadata(_record('S01_Part1_Probe1', 'visual_cognitive_motivation'))
    b = runner._physiology_record_metadata(_record('S01_Part1_Probe2', 'visual_cognitive_motivation'))
    c = runner._physiology_record_metadata(_record('S01_Part2_Probe1', 'visual_cognitive_motivation'))
    assert a['repeat_id'] == b['repeat_id'] != c['repeat_id']
    assert a['record_order'] == b['record_order'] == 1 and c['record_order'] == 2
    assert a['repeat_kind'] == 'same_visit_cross_part'
    refed = _record('video_2', 'refed')
    assert runner._physiology_task_label(refed)[0] == 'emotion_video/video_2'
    assert runner._physiology_record_metadata(refed)['repeat_kind'] == 'unavailable_different_video_stimuli'


def test_extraction_stage_pilot_preserves_complete_plan_and_merges_after_resume(tmp_path):
    records = [_record('session_00'), _record('session_01')]
    plan = dict(datasets=['eeg_fnirs_single_trial'], records=records)
    parent, output = tmp_path/'parent', tmp_path/'analysis'
    tasks = [_populate(parent, record) for record in records]
    panel = {tasks[0][0]['id'], tasks[1][0]['id']}
    cfg = dict(scope=dict(scalar_pair_windows=8))
    first = runner.physiology_extract_stage(cfg, parent, output, plan, panel, 1, max_records=1)
    assert first['status'] == 'incomplete' and first['planned_records'] == 2
    assert not (output/'fixed_window_metrics.csv.gz').exists()
    result = runner.physiology_extract_stage(cfg, parent, output, plan, panel, 1)
    assert result['status'] == 'completed' and result['new_records'] == 1
    frame = pd.read_csv(output/result['files']['windows'])
    assert len(frame) == 8 and frame.id.is_unique and frame.panel_selected.sum() == 2
    assert result['statuses'] == {'partial_or_failed_fit': 2, 'completed': 6}
    assert (parent/'plan.json').exists() is False


def test_analysis_directory_is_strictly_below_the_declared_parent(tmp_path):
    cfg = runner.read_physiology_config(); parent = tmp_path/cfg['parent_run']
    assert runner._physiology_directory(cfg, parent, tmp_path) == parent/'fixed_physiology_v1'
    with pytest.raises(ValueError, match='exact retained parent'):
        runner._physiology_directory(cfg, tmp_path/'other', tmp_path)
    drift = dict(cfg, analysis_subdir='..')
    with pytest.raises(ValueError, match='own analysis directory'):
        runner._physiology_directory(drift, parent, tmp_path)


def test_scalar_analysis_writes_owner_tables_and_waits_for_auxiliary_stages(tmp_path):
    record = _record(); parent, output = tmp_path/'parent', tmp_path/'analysis'
    tasks = _populate(parent, record)
    cfg = runner.read_physiology_config(); cfg['scope']['scalar_pair_windows'] = 4
    cfg['statistics'].update(bootstrap_samples=20, permutation_samples=20)
    plan = dict(datasets=[record['dataset']], records=[record])
    runner.write_json(parent/'resolved_config.json', dict(fixed=dict(tau=2.)))
    runner.physiology_extract_stage(cfg, parent, output, plan, {t['id'] for t in tasks}, 1)
    result = runner.physiology_analyze_stage(cfg, parent, output)
    assert result['status'] == 'completed'
    for key in ('subjects', 'subject_tasks', 'native_pairs', 'native_records', 'subject_descriptives',
                'task_comparisons', 'repeat_pairs', 'within_record_pairs', 'examples'):
        assert (output/result['files'][key]).exists()
    subjects = pd.read_csv(output/result['files']['subjects'])
    d1 = subjects[subjects.metric.eq('D1')].iloc[0]
    assert d1.value == pytest.approx(np.sqrt((1+4+9+16)/4))
    weak = subjects[subjects.metric.eq('D_bal_20s')].iloc[0]
    assert weak.value == .5 and weak.planned_windows == 2
    summary = runner.read_json(output/'summary.json')
    assert summary['schema'] == 'tak_mismatch_fixed_physiology_analysis_v1'
    assert summary['status'] == 'incomplete' and summary['stages']['ocular'] == 'pending'
    assert summary['fixed_parameters'] == dict(tau=2.)


def _fake_ocular(record):
    n = len(record['windows'])
    return dict(rms=np.ones((n,2)), peak_abs=np.ones((n,2)), derivative_rms=np.ones((n,2)),
        rms_combined=np.ones(n), rms_10hz=np.ones((n,300,2)), mean_10hz=np.ones((n,300,2)),
        valid_mask=np.ones((n,300,2),bool), complete=np.ones((n,2),bool),
        window_ids=np.asarray([w['id'] for w in record['windows']]), time_s=np.arange(300)/10,
        metadata=dict(status='available', reason=None, measured_signal_reads=1,
            windows=[dict(status='available', reason=None) for w in record['windows']]))


def test_ocular_stage_keeps_only_frozen_panel_windows_and_other_datasets_are_na(tmp_path, monkeypatch):
    from src.data import tak_mismatch_ocular as ocular
    native = _record(); other = _record('video_1', 'refed')
    parent, output = tmp_path/'parent', tmp_path/'analysis'
    runner.write_json(parent/'resolved_config.json', dict(fixed={}))
    parent_cfg = runner.read_config(); cfg = runner.read_physiology_config()
    selected = [next(runner.task_manifest(dict(records=[r]))) for r in (native,other)]
    calls = []
    def prepare(root, record, **kwargs):
        calls.append(record['id']); return _fake_ocular(record)
    monkeypatch.setattr(ocular, 'prepare_tak_ocular_record', prepare)
    monkeypatch.setattr(runner, '_metadata_index', lambda *a: None)
    result = runner.physiology_ocular_stage(parent_cfg, cfg, tmp_path, parent, output,
        dict(records=[native,other]), {t['id'] for t in selected}, 1)
    assert result['status'] == 'completed' and calls == [native['id']]
    frame = pd.read_csv(output/result['files']['ocular_windows'])
    assert len(frame) == 2
    assert frame[frame.dataset.eq('refed')].eog_rms_combined.isna().all()
    aux = runner._physiology_read_aux(output/'ocular_records'/f'{runner.token(native["id"])}.npz')
    assert aux['window_indices'].tolist() == [0]
    assert runner._physiology_aux_window(aux, 0)['rms_10hz'].shape == (300,2)
    assert runner._physiology_aux_window(aux, 1) is None
    assert (output/'ocular_stage_summary.json').exists()


def test_counterfactual_uses_frozen_models_and_never_calls_a_fit(tmp_path, monkeypatch):
    from src.data import tak_mismatch_ocular as ocular
    record = _record(); parent, output = tmp_path/'parent', tmp_path/'analysis'
    tasks = list(runner.task_manifest(dict(records=[record])))
    runner.write_npz(runner.prepared_paths(parent, record['id'])[1], marker=np.array([42]))
    models = {runner._group_key(record,p): {'0':dict(coordinate=dict(marker=p['id']))} for p in record['pairs']}
    runner.write_json(parent/'spatial_selection.json', dict(models=models))
    seen = {}
    def prepare(root, plan, coordinate_models, selected_pairs, *, prepared, **kwargs):
        seen.update(models=coordinate_models, pairs=selected_pairs, marker=prepared['marker'].item())
        return dict(raw_proxy=np.ones((2,2,300)), clean_proxy=np.ones((2,2,300)),
            amplitude_proxy=np.ones((2,2,300)), complete=np.ones((2,2),bool),
            amplitude_available=np.ones((2,2),bool),
            window_ids=np.asarray([w['id'] for w in record['windows']]), pair_ids=np.array(['p1','p2']),
            time_s=np.arange(300)/10, metadata=dict(status='available', measured_signal_reads=1))
    monkeypatch.setattr(ocular, 'prepare_tak_ocular_counterfactual', prepare)
    monkeypatch.setattr(runner, '_metadata_index', lambda *a: None)
    monkeypatch.setattr(runner, '_fit_views', lambda *a,**k: pytest.fail('counterfactual must never fit'))
    result = runner._physiology_counterfactual_record((runner.read_config(), str(tmp_path), parent, output, record, tasks[:2]))
    assert result['status'] == 'completed' and result['available_window_pairs'] == 2
    assert seen == dict(models=dict(p1=dict(marker='p1'), p2=dict(marker='p2')), pairs=['p1','p2'], marker=42)
    arrays = runner._physiology_read_aux(result['files']['arrays'])
    assert arrays['raw_proxy'].shape == (1,2,300)
    assert runner._physiology_aux_window(arrays, 0, 'p1')['raw_proxy'].shape == (300,)


def test_shape_record_uses_exact_retained_arrays_and_auxiliary_slices(tmp_path, monkeypatch):
    from src.analysis import tak_mismatch as analysis
    record = _record(); parent, output = tmp_path/'parent', tmp_path/'analysis'
    tasks = _populate(parent, record)
    selected = tasks[:1]
    runner._physiology_extract_record((parent, output, record, {selected[0]['id']}))
    runner.write_npz(parent/'tasks'/f'{selected[0]["id"]}.npz', marker=np.array([5]))
    seen = []
    def features(arrays, row, *, ocular, counterfactual):
        seen.append((arrays['marker'].item(), row['id'], ocular, counterfactual))
        return dict(feature_status='available_Hb_residual', hb_residual_bias_ms=.5)
    monkeypatch.setattr(analysis, 'fixed_window_physiology_features', features)
    result = runner._physiology_shape_record((parent, output, record, selected))
    assert result['status'] == 'completed' and result['arrays_read'] == 1
    assert seen == [(5,selected[0]['id'],None,None)]
    frame = pd.read_csv(result['files']['features'])
    assert frame.hb_residual_bias_ms.tolist() == [.5]


def test_view_coverage_keeps_support_absence_separate_from_numerical_failure():
    rows = []
    for status, view_status, success, eeg_input in (
        ('completed', 'completed', True, 'completed'),
        ('unavailable_complete_Hb_pair', 'unavailable', False, 'unavailable_native_EEG'),
        ('partial_or_failed_fit', 'failed_convergence', False, 'completed'),
        ('failed_exception', 'unavailable', False, None),
    ):
        row = dict(dataset='d', status=status, eeg_input_status=eeg_input)
        for view in ('hb_only', 'eeg_only', 'joint'):
            row.update({view+'_status':view_status, view+'_success':success, view+'_solver_status':view_status})
        rows.append(row)
    bucket = runner._physiology_view_coverage(pd.DataFrame(rows))['by_dataset']['d']['hb_only']
    assert bucket['planned'] == 4 and bucket['fitted'] == 2
    assert {k:bucket[k] for k in ('completed','unavailable_support','numerical_failure','execution_failure')} == dict(
        completed=1, unavailable_support=1, numerical_failure=1, execution_failure=1)
    assert bucket['rates']['numerical_failure_rate']['denominator'] == 2


def test_complete_frozen_pipeline_does_not_overwrite_ocular_stage_summary(tmp_path, monkeypatch):
    from src.data import tak_mismatch_ocular as ocular
    record = _record(); parent, output = tmp_path/'parent', tmp_path/'analysis'
    tasks = _populate(parent, record); panel = {task['id'] for task in tasks}
    cfg = runner.read_physiology_config()
    cfg['scope'].update(scalar_pair_windows=4, array_pair_windows=4, ocular_native_records=1)
    cfg['statistics'].update(bootstrap_samples=20, permutation_samples=20)
    parent_cfg = runner.read_config(); plan = dict(datasets=[record['dataset']], records=[record])
    runner.write_json(parent/'resolved_config.json', parent_cfg)
    runner.write_json(parent/'spatial_selection.json', dict(models={
        runner._group_key(record,p): {'0':dict(coordinate={})} for p in record['pairs']}))
    runner.write_npz(runner.prepared_paths(parent, record['id'])[1], marker=np.array([1]))
    signal = np.sin(np.arange(300)/10.)
    for i, task in enumerate(tasks):
        target = np.column_stack((signal, np.full(300,i+1.), np.full(300,i+1.)))
        target[-1] = np.nan
        mask = np.zeros((300,3),bool); mask[50:299] = True
        runner.write_npz(parent/'tasks'/f'{task["id"]}.npz', target=target, sd=np.ones(3),
            score_mask=mask, hb_only_prediction=np.zeros((300,3)), time_s=np.arange(300)/10.)
    monkeypatch.setattr(runner, '_metadata_index', lambda *a: None)
    monkeypatch.setattr(ocular, 'prepare_tak_ocular_record', lambda root,record,**kwargs:_fake_ocular(record))
    def counter(*args, **kwargs):
        raw = np.broadcast_to(signal,(2,2,300)).copy()
        return dict(raw_proxy=raw, clean_proxy=raw*.5, amplitude_proxy=raw.copy(),
            complete=np.ones((2,2),bool), amplitude_available=np.ones((2,2),bool),
            window_ids=np.asarray([w['id'] for w in record['windows']]), pair_ids=np.array(['p1','p2']),
            time_s=np.arange(300)/10., metadata=dict(status='available', measured_signal_reads=1))
    monkeypatch.setattr(ocular, 'prepare_tak_ocular_counterfactual', counter)
    runner.physiology_extract_stage(cfg, parent, output, plan, panel, 1)
    runner.physiology_analyze_stage(cfg, parent, output)
    runner.physiology_ocular_stage(parent_cfg, cfg, tmp_path, parent, output, plan, panel, 1)
    runner.physiology_counterfactual_stage(parent_cfg, cfg, tmp_path, parent, output, plan, panel, 1)
    result = runner.physiology_shapes_stage(cfg, parent, output, plan, panel, 1)
    assert result['status'] == 'completed' and result['rows'] == 4 and result['errors'] == 0
    summary = runner.read_json(output/'summary.json')
    assert summary['status'] == 'completed' and set(summary['stages'].values()) == {'completed'}
    assert runner.read_json(output/'ocular_stage_summary.json')['status'] == 'completed'
    assert summary['files']['ocular_summary'] == 'ocular_summary.json'
    assert summary['files']['shape_windows'] == 'shape_window_features.csv.gz'
