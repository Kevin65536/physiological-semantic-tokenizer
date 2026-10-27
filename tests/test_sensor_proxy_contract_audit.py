"""Independent runner-boundary audits using temporary synthetic records only."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.inference import shared_driver_reconstruction as inference


_PATH = Path(__file__).resolve().parents[1] / 'experiments/scripts/evaluate_shared_driver_sensor_proxy.py'
_SPEC = importlib.util.spec_from_file_location('sensor_proxy_contract_audit_runner', _PATH)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


@pytest.mark.parametrize('pairing', ['paired', 'wrong_EEG'])
def test_hidden_target_and_wrong_donor_hb_never_reach_nonlinear_fit(tmp_path, monkeypatch, pairing):
    """Changing only held-out Hb may change scores, never the solver request."""
    cfg = runner.read_config()
    n = cfg['tensor']['steps']
    rng = np.random.default_rng(902)
    reference = np.eye(n)
    reference[:, :50] -= 1 / 50
    row = dict(id='target', record='record_target', subject='subject_fixture', split='selection')
    donor_row = dict(id='donor', record='record_donor', subject='subject_fixture', split='selection')
    rows = {}
    for identity, source_row in [('target', row), ('donor', donor_row)]:
        rows[identity] = dict(row=dict(source_row, channel_sets=dict(sets=dict(nearest6=list(range(6))))),
            eeg_logpower=rng.normal(size=(n, 6, 5)) * .01,
            hb=rng.normal(size=(n, 2)) * .01,
            eeg_operator=reference.copy(), hb_operator=reference.copy(), time_s=np.arange(n) / 10)
    coordinate = dict(pc=np.arange(1, 31) / 100, eeg_factor=.02, hb_factor=1., sd=[.025, .03, .02])
    screen = dict(coordinate=coordinate, models=dict(pc1=dict(coefficients=np.ones(5))))
    task = dict(id='synthetic_prediction', record_id='target', subject=row['subject'],
        split='selection', candidate='pc1', condition='prediction', pairing=pairing,
        wrong_eeg_id='donor' if pairing == 'wrong_EEG' else None)
    requests = []

    def fake_forward(driver, initial, parameters, dt, **kwargs):
        states = np.tile(np.r_[0., 0., 1., 1., 1., 1.], (len(driver), 1))
        return dict(canonical_prediction=np.zeros((len(driver), 3)), states=states)

    def capture_fit(observations, parameters, dt, **kwargs):
        requests.append((observations.copy(), deepcopy(kwargs)))
        assert np.isnan(observations[:, :, 1:]).all()
        assert not kwargs['visible'][:, :, 1:].any()
        assert 'starts' not in kwargs  # No target-derived driver or initial-state warm start.
        state = fake_forward(np.zeros(n), None, parameters, dt)['states']
        return dict(status='completed', converged=True, prediction=np.zeros((1, n, 3)),
            driver=np.zeros((1, n)), initial_state=np.array([[0., 1., 1., 1., 1.]]),
            states=state[None], objective=0., objective_components=dict(data=0., driver=0., initial=0., flow=0.))

    monkeypatch.setattr(runner, '_load_prepared', lambda out, source_row: rows[source_row['id']])
    monkeypatch.setattr(inference, 'fit_nonlinear_shared_parameters', capture_fit)
    monkeypatch.setattr(inference, 'nonlinear_driver_forward', fake_forward)
    for name in ('first', 'second'):
        for identity in rows:
            runner.write_json(runner._paths(tmp_path / name, identity)[0], dict(status='completed'))
    donor = donor_row if pairing == 'wrong_EEG' else None
    first = runner._nonlinear_job((cfg, str(tmp_path / 'first'), task, row, donor, screen))
    rows['target']['hb'] += 37.
    rows['donor']['hb'] -= 81.
    second = runner._nonlinear_job((cfg, str(tmp_path / 'second'), task, row, donor, screen))
    assert first['success'] and second['success']
    assert first['nmse'] != second['nmse']  # The intervention really reached scoring.
    np.testing.assert_array_equal(requests[0][0], requests[1][0])
    for field in ('visible', 'sd', 'mean_operator'):
        np.testing.assert_array_equal(requests[0][1][field], requests[1][1][field])
    for field in ('penalty', 'initial_penalty', 'flow_prior_weight', 'initial_state_mode', 'data_weight'):
        assert requests[0][1][field] == requests[1][1][field]


def test_failed_task_is_terminal_on_resume_and_keeps_failure_denominator(tmp_path, monkeypatch):
    """A numerical failure remains evidence and must not be replaced on resume."""
    cfg = runner.read_config()
    task = dict(id='failed_cell', record_id='target', subject='subject_fixture', split='selection',
        candidate='pc1', condition='prediction', pairing='paired', wrong_eeg_id=None)
    result = dict(task, status='failed_numerical', success=False, nmse=.001)
    runner.write_json(tmp_path / 'tasks' / 'failed_cell.json', result)

    def forbidden_read(*args):
        raise AssertionError('terminal failed tasks must not read or refit')

    monkeypatch.setattr(runner, '_load_prepared', forbidden_read)
    resumed = runner._nonlinear_job((cfg, str(tmp_path), task, {}, None, {}))
    assert resumed == result
    successful = dict(result, id='successful_cell', success=True, status='completed', nmse=2.)
    assert runner._subject_score([resumed, successful], penalty=100.) == {'subject_fixture': 51.}


def test_screen_fits_each_coordinate_without_heldout_subjects_or_evaluation(tmp_path, monkeypatch):
    """Audit the screen's real PCA/ridge routing, replacing only physical solves."""
    cfg = runner.read_config()
    rng = np.random.default_rng(911)
    n = 100
    sets = dict(nearest3=list(range(3)), nearest6=list(range(6)), nearest12=list(range(12)),
        outer6=list(range(6, 12)), sixth_to_seventh=[0, 1, 2, 3, 4, 6],
        nearest6_far6=list(range(6)) + list(range(24, 30)))
    for index in range(3):
        sets[f'nearest6_random6_{index}'] = list(range(6)) + list(range(6 + index, 12 + index))
    rows, records = [], {}
    for index in range(8):
        split = 'train' if index < 6 else 'selection' if index == 6 else 'evaluation'
        row = dict(id=f'fixture_{index}', record=f'record_{index}', subject=f'subject_{index}', split=split,
            channel_sets=dict(sets=deepcopy(sets)))
        rows.append(row)
        record = dict(row=row, eeg_logpower=rng.normal(size=(n, 30, 5)) * (index + 1),
            hb=rng.normal(size=(n, 2)) * (index + 1), eeg_operator=np.eye(n),
            hb_operator=np.eye(n), time_s=np.arange(n) / 10)
        records[row['id']] = record
    plan = dict(rows=rows, wrong_eeg_partners={r['id']: None for r in rows})
    coordinate_owner = runner.coordinate
    training_members = []

    def audit_coordinate(config, inputs):
        assert all(r['row']['split'] == 'train' for r in inputs)
        training_members.append({r['row']['subject'] for r in inputs})
        return coordinate_owner(config, inputs)

    class FixedPhysicalMap:
        hb_offset = np.zeros((n, 2))

        def map_features(self, features):
            return np.stack((features * .1, features * -.2), axis=1)

        def predict(self, proxy):
            return dict(prediction=np.column_stack((proxy, .1 * proxy, -.2 * proxy)))

    def load(out, row):
        assert row['split'] != 'evaluation', 'screen must not inspect final evaluation signals'
        return records[row['id']]

    monkeypatch.setattr(runner, '_load_prepared', load)
    monkeypatch.setattr(runner, 'coordinate', audit_coordinate)
    monkeypatch.setattr(runner, 'conditional_map', lambda *args: FixedPhysicalMap())
    result = runner.screen_stage(cfg, tmp_path, plan)
    all_training = {r['subject'] for r in rows if r['split'] == 'train'}
    assert len(training_members) == 4
    for fitted_subjects, heldout_subjects in zip(training_members[:3], result['folds']):
        assert fitted_subjects == all_training - set(heldout_subjects)
        assert not fitted_subjects & set(heldout_subjects)
    assert training_members[-1] == all_training
    assert set(result['training_ids']) == {r['id'] for r in rows if r['split'] == 'train'}


def test_unequal_window_counts_preserve_native_record_equal_aggregation():
    rows = [dict(subject='S', native_record=record, success=True, nmse=value)
        for record, value in [('A', 0.), ('A', 2.), ('B', 9.)]]
    assert runner._subject_score(rows) == {'S': 5.}
    # Repeating all windows of one native record must not give it more weight.
    assert runner._subject_score(rows + rows[:2]) == {'S': 5.}


def test_wrong_pair_promotion_uses_record_equal_success_pairing():
    cfg = runner.read_config()
    candidate, baseline, wrong, zero = [], [], [], []
    for subject in ('S0', 'S1', 'S2'):
        for record, window, true_error, wrong_error in [('A', 0, 0., 5.), ('A', 1, 0., 5.), ('B', 0, 9., 0.)]:
            identity = f'{subject}|{record}|{window}'
            common = dict(subject=subject, record_id=identity, native_record=record,
                success=True, status='completed')
            candidate.append(dict(common, nmse=true_error))
            wrong.append(dict(common, nmse=wrong_error))
            baseline.append(dict(common, nmse=10.))
            zero.append(dict(common, nmse=10.))
    assessment = runner.promotion_assessment(cfg, candidate, baseline, wrong, zero)
    assert assessment['subject_mean_nmse'] == pytest.approx(4.5)
    assert not assessment['true_better_than_wrong']
    assert not assessment['eligible']


def test_bootstrap_point_estimates_match_native_record_then_subject_unit():
    cfg = runner.read_config()
    rows = []
    for subject, candidate_errors, baseline_errors in [('S0', [0., 2., 9.], [10., 10., 10.]),
                                                        ('S1', [2., 4., 1.], [2., 2., 4.])]:
        for index, (candidate_error, baseline_error) in enumerate(zip(candidate_errors, baseline_errors)):
            common = dict(subject=subject, native_record='A' if index < 2 else 'B',
                record_id=f'{subject}|window{index}', success=True, status='completed',
                split='evaluation', condition='prediction', pairing='paired')
            rows.extend([dict(common, candidate='mean12', nmse=candidate_error),
                         dict(common, candidate='pc1', nmse=baseline_error)])
    result = runner.paired_comparisons(cfg, rows)['evaluation|mean12_vs_pc1']
    assert result['candidate_nmse'] == pytest.approx(3.5)
    assert result['baseline_nmse'] == pytest.approx(6.5)
    assert result['delta_nmse_candidate_minus_baseline'] == pytest.approx(-3.)
    assert result['relative_nmse_improvement'] == pytest.approx(1 - 3.5 / 6.5)


def test_failed_wrong_donor_preparation_remains_input_unavailability(tmp_path, monkeypatch):
    cfg = runner.read_config()
    target = dict(id='target', record='A', subject='S', split='selection')
    donor = dict(id='donor', record='B', subject='S', split='selection')
    task = dict(id='wrong_input_failure', record_id='target', native_record='A', subject='S',
        split='selection', candidate='pc1', condition='prediction', pairing='wrong_EEG', wrong_eeg_id='donor')
    runner.write_json(runner._paths(tmp_path, 'target')[0], dict(status='completed'))
    runner.write_json(runner._paths(tmp_path, 'donor')[0], dict(status='failed_preparation'))

    def load(out, row):
        assert row['id'] != 'donor', 'failed donor preparation must be recognized before array access'
        return {}

    monkeypatch.setattr(runner, '_load_prepared', load)
    result = runner._nonlinear_job((cfg, str(tmp_path), task, target, donor, {}))
    assert result['status'] == 'unavailable_preparation'
    assert not result['success']
    assert result['wrong_eeg_id'] == 'donor'
