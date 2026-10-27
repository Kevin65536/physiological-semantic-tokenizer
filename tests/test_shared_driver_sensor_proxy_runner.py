"""Synthetic contracts for the fixed-Tak sensor proxy suite controller."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from experiments.scripts import evaluate_shared_driver_sensor_proxy as runner


@pytest.fixture
def cfg():
    return runner.read_config()


@pytest.mark.parametrize('key,value', [
    (('data', 'protected_data'), 'allowed'),
    (('data', 'target_hb_pair'), 'another'),
    (('tensor', 'dt_s'), .25),
    (('model', 'fitted_physiological_parameters'), ['tau']),
    (('model', 'eeg_weight'), 0.),
    (('fixed', 'viscoelastic_s'), 0.),
    (('solver', 'validation_substeps'), 4),
    (('resources', 'numerical_threads'), 2),
])
def test_read_config_rejects_contract_drift(cfg, tmp_path, key, value):
    altered = deepcopy(cfg); altered[key[0]][key[1]] = value
    path = tmp_path/'changed.yaml'; path.write_text(yaml.safe_dump(altered))
    with pytest.raises(ValueError): runner.read_config(path)


def test_check_opens_no_arrays(cfg, monkeypatch, capsys):
    def forbidden(*args, **kwargs): pytest.fail('check cannot read measured arrays')
    monkeypatch.setattr(runner.np, 'load', forbidden)
    assert runner.main([]) == 0
    assert '"measured_arrays_read": 0' in capsys.readouterr().out
    assert runner.check(cfg)['planned_logical_cells'] == 420


def _row(i, subject='s', split='selection', start=0.):
    return dict(id=str(i), subject=subject, split=split, task='MI', condition='left',
        record='r', eeg_filter_start_s=start,
        channel_sets=dict(eeg_channel_names=['Cz'], sets=dict(nearest6=list(range(6)),
            outer6=list(range(6, 12)), nearest12=list(range(12)))))


def test_wrong_pairing_uses_full_filter_support_and_preserves_missing():
    a, b = _row('a'), _row('b', start=89.)
    assert runner.wrong_eeg_partners([a, b]) == {'a': None, 'b': None}
    b['eeg_filter_start_s'] = 90.
    assert runner.wrong_eeg_partners([a, b]) == {'a': 'b', 'b': 'a'}
    b['condition'] = 'right'
    assert runner.wrong_eeg_partners([a, b])['a'] is None


def _plan():
    rows = [_row(i, subject=f'd{i//2}', split='selection') for i in range(12)]
    rows += [_row(12+i, subject=f'e{i//6}', split='evaluation') for i in range(36)]
    return dict(rows=rows, wrong_eeg_partners={r['id']: None for r in rows})


def test_logical_budget_and_identity_are_fixed_with_unavailable_wrong_controls():
    plan = _plan(); screen = dict(shortlist=['pc1', 'pc1_band_gain', 'mean12'])
    first = runner.nonlinear_tasks(plan, screen, phase='development')
    second = runner.nonlinear_tasks(plan, screen, phase='evaluation', evaluation_candidates=['pc1', 'mean12'])
    assert len(first) == 204 and len(second) == 216
    assert len({r['id'] for r in first+second}) == 420
    assert sum(r['candidate'] == 'Hb_only' for r in first) == 48
    assert sum(r['candidate'] == 'zero' for r in first) == 48
    assert sum(r['pairing'] == 'wrong_EEG' for r in first+second) == 108
    assert all(r['wrong_eeg_id'] is None for r in first+second)


def _record(n=300):
    rng = np.random.default_rng(42)
    reference = np.eye(n); reference[:, :50] -= .02
    return dict(eeg_logpower=rng.normal(size=(n, 12, 5)), hb=rng.normal(size=(n, 2)),
        eeg_operator=reference, hb_operator=reference, time_s=np.arange(n)*.1, row=_row('x'))


def _coord():
    return dict(pc=np.ones(30)/np.sqrt(30), eeg_factor=.03, hb_factor=2., sd=[.025, .03, .02])


@pytest.mark.parametrize('poison', [np.nan, np.inf, 1e30])
def test_prediction_input_hides_both_Hb_before_assembly(cfg, poison):
    a = _record(); b = deepcopy(a); b['hb'][:] = poison
    clean = runner.assemble_fit_input(cfg, a, _coord(), 'pc1', np.ones(5), condition='prediction')
    changed = runner.assemble_fit_input(cfg, b, _coord(), 'pc1', np.ones(5), condition='prediction')
    np.testing.assert_array_equal(clean[0], changed[0])
    np.testing.assert_array_equal(clean[1], changed[1])
    assert np.isnan(clean[0][:, 1:]).all() and not clean[1][:, 1:].any()
    assert not clean[1][-1].any()


def test_fir_design_has_zero_reference_mean_and_independent_species_coefficients(cfg):
    design = runner.fir_design(cfg, _record(), _coord(), 'pc1', np.ones(5))
    assert design.shape == (300, 2, 12)
    np.testing.assert_allclose(design[:50].mean(axis=0), 0., atol=1e-15)
    assert not design[:, 0, 6:].any() and not design[:, 1, :6].any()
    assert np.isfinite(design).all()


def _scores(candidate, count=6, nmse=1., success=True):
    return [dict(record_id=str(i), subject=f's{i//2}', candidate=candidate, condition='prediction',
                 pairing='paired', split='selection', status='completed' if success else 'failed_numerical',
                 success=success, nmse=nmse) for i in range(count)]


def test_promotion_requires_real_wrong_evidence_and_no_failure_manufactured_gain(cfg):
    baseline = _scores('pc1', nmse=1.)
    candidate = _scores('mean12', nmse=.8)
    wrong = _scores('mean12', nmse=1.1)
    zero = _scores('zero', nmse=1.2)
    assert runner.promotion_assessment(cfg, candidate, baseline, wrong, zero)['eligible']
    failed_wrong = _scores('mean12', nmse=100., success=False)
    result = runner.promotion_assessment(cfg, candidate, baseline, failed_wrong, zero)
    assert not result['eligible'] and result['wrong_paired_windows'] == 0
    scarce = wrong[:2]
    assert not runner.promotion_assessment(cfg, candidate, baseline, scarce, zero)['eligible']


def test_failure_penalty_subject_weighting_does_not_drop_failed_subject():
    rows = _scores('pc1', nmse=1.)
    rows[-1].update(success=False, nmse=.01)
    means = runner._subject_score(rows)
    assert means['s2'] == 50.5
    assert np.mean(list(means.values())) == pytest.approx(17.5)


def test_resume_terminal_failure_never_refits(cfg, tmp_path, monkeypatch):
    task = dict(id='x', record_id='r', subject='s', split='selection', candidate='pc1', condition='prediction', pairing='paired')
    retained = dict(task, status='failed_numerical', success=False, nmse=None)
    runner.write_json(tmp_path/'tasks/x.json', retained)
    monkeypatch.setattr(runner, '_load_prepared', lambda *a: pytest.fail('terminal result must be reused'))
    assert runner._nonlinear_job((cfg, str(tmp_path), task, {}, None, {})) == retained


def test_worker_failure_and_missing_preparation_have_terminal_records(cfg, tmp_path):
    task = dict(id='x', record_id='r', subject='s', split='selection', candidate='pc1', condition='prediction', pairing='wrong_EEG')
    result = runner._nonlinear_job((cfg, str(tmp_path), task, dict(id='r'), None, {}))
    assert result['status'] == 'unavailable_wrong_EEG'
    assert runner.read_json(tmp_path/'tasks/x.json') == result


def test_paired_bootstrap_uses_subjects_and_retains_failed_records(cfg):
    original, alternative = _scores('pc1'), _scores('mean12', nmse=.9)
    alternative[-1].update(success=False, status='failed_numerical')
    result = runner.paired_comparisons(cfg, original+alternative)['selection|mean12_vs_pc1']
    assert result['paired_subjects'] == 3 and result['paired_windows'] == 6
    assert result['candidate_nmse'] > 1.


@pytest.mark.parametrize('donor_only', [False, True])
def test_grouped_preparation_normalizes_serialized_root_before_native_read(cfg, tmp_path, monkeypatch, donor_only):
    from src.data import clean_physiology_cache, shared_driver_spatial, unified_physiology
    source = [{'path': 'registered/native.mat'}]
    record = SimpleNamespace(join_key='record', dataset_id='eeg_fnirs_single_trial',
        canonical_subject_id='subject', base_record_id='record', manifest={'source_files': source})
    row = dict(id='window', join_key='record', dataset=record.dataset_id, subject='subject',
        record='record', split='selection', source_identity=source, donor_only=donor_only,
        geometry_identity={'eeg': [], 'fnirs': []}, patch={'native_target_index': 3}, channel_sets={})
    monkeypatch.setattr(clean_physiology_cache, 'CleanPhysiologyCacheIndex',
        lambda path: SimpleNamespace(records=[record]))
    monkeypatch.setattr(shared_driver_spatial, 'select_proxy_channel_sets', lambda *args, **kwargs: {})
    loaded = []
    def load(root, native_record):
        assert isinstance(root, Path)
        assert root == tmp_path and native_record is record
        loaded.append(root)
        return object()
    monkeypatch.setattr(unified_physiology, 'load_native_eeg_record', load)
    monkeypatch.setattr(unified_physiology, 'load_cached_fnirs_record', load)
    def prepare(payload, **kwargs):
        assert isinstance(payload[2], Path)
        assert kwargs['record'] is record and kwargs['eeg_record'] is not None
        assert (kwargs['hb_record'] is None) == donor_only
        return dict(id=payload[3]['id'], status='completed')
    monkeypatch.setattr(runner, '_prepare_job', prepare)
    result = runner._prepare_record_job((cfg, str(tmp_path/'artifacts'), str(tmp_path), [row]))
    assert result == [dict(id='window', status='completed')]
    assert len(loaded) == (1 if donor_only else 2)


def test_readonly_residual_analysis_uses_one_common_cohort_and_never_reads_or_fits(cfg, tmp_path, monkeypatch):
    from src.data import shared_driver_spatial, unified_physiology
    from src.inference import low_capacity_eeg_proxy, shared_driver_reconstruction
    run = tmp_path/'completed_run'; run.mkdir()
    retained = deepcopy(cfg); retained['tensor']['score_steps'] = [0, 4]
    runner.write_json(run/'resolved_config.json', retained)
    runner.write_json(run/'screen.json', dict(coordinate=dict(sd=[.025, 2., 1.])))
    runner.write_json(run/'evaluation_selection.json', dict(candidates=['pc1', 'mean6']))
    rows = [dict(id=f'w{i}', split='evaluation', subject='subject', record='A' if i < 2 else 'B') for i in range(4)]
    runner.write_json(run/'plan.json', dict(rows=rows))
    tasks = []
    amplitudes = {'pc1': [1., 3., 10., 10.], 'mean6': [2., 4., 1., 1.], 'Hb_only': [.5, .5, .5, .5]}
    for candidate, values in amplitudes.items():
        for i, value in enumerate(values):
            task = dict(id=f'{candidate}_{i}', record_id=f'w{i}', native_record=rows[i]['record'],
                subject='subject', split='evaluation', candidate=candidate, condition='joint', pairing='paired')
            tasks.append(task)
            success = not ((candidate == 'pc1' and i == 0) or (candidate == 'mean6' and i == 3))
            result = dict(task, status='completed' if success else 'failed_numerical', success=success, nmse=value**2,
                residual_diagnostics=dict(lag_s=[.5, 1., 2., 5.], lag_correlation=np.full((4, 2), .4),
                    spectral_bands_hz=[[0., .05], [.05, .15], [.15, .5]], spectral_band_power=np.full((2, 3), .04),
                    spectrum_rule='retained_contiguous_periodogram'))
            runner.write_json(run/'tasks'/f'{task["id"]}.json', result)
            residual = np.full((4, 2), value)*np.array([2., 1.])
            runner.write_npz(run/'tasks'/f'{task["id"]}.npz', residual=residual, target_hb=np.zeros((4, 2)))
    runner.write_json(run/'nonlinear_tasks.json', tasks)
    runner.write_json(run/'summary.json', dict(status='completed', planned=len(tasks), terminal=len(tasks)))
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    def forbidden(*args, **kwargs): pytest.fail('read-only analysis must not load native signals or fit')
    for module, names in ((unified_physiology, ['load_native_eeg_record', 'load_cached_fnirs_record']),
            (shared_driver_spatial, ['prepare_sensor_proxy_record', 'prepare_sensor_proxy_eeg_record']),
            (shared_driver_reconstruction, ['fit_nonlinear_shared_parameters', 'nonlinear_driver_forward']),
            (low_capacity_eeg_proxy, ['build_tak_eeg_conditional_map', 'fit_anchored_ridge'])):
        for name in names: monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(runner, 'read_config', forbidden)
    derivative = run/'residual_analysis_v1'
    assert runner.main(['--stage', 'analyze', '--output', str(run), '--analysis-output', str(derivative)]) == 0
    result = runner.read_json(derivative/'residual_summary.json')
    assert result['planned_windows'] == 4 and result['common_success_windows'] == 2
    assert result['common_success_window_ids'] == ['w1', 'w2']
    assert result['methods']['pc1']['joint_nmse'] == 54.5
    # Candidate-only successful support would give 5.5; the common cohort gives 8.5.
    assert result['methods']['mean6']['joint_nmse'] == 8.5
    assert result['methods']['Hb_only']['joint_nmse'] == .25
    assert result['native_record_counts']['subject']['A']['common_success_windows'] == 1
    assert result['native_reads'] == 0 and result['new_fits'] == 0
    assert {p: (run/p).read_bytes() for p in before} == before
    with pytest.raises(ValueError, match='already exists'):
        runner.analyze_stage(run, derivative)
    with pytest.raises(ValueError, match='within the completed run'):
        runner.analyze_stage(run, tmp_path/'outside_run')
