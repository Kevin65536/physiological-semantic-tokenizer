"""The shared-parameter companion preserves information and native support."""
from copy import deepcopy

import numpy as np

from experiments.scripts import evaluate_tak_mismatch as runner


def _fixture(tmp_path):
    cfg = runner.read_config()
    pair = dict(id='pair', native_index=0, channel_sets={name: list(range(6)) for name in runner.CANDIDATES})
    windows = [dict(id=f'w{i}', start_s=30.*i, hb_start_s=30.*i,
        eeg_filter_start_s=0., eeg_filter_end_s=60., actual_context_s=60.,
        native_support_id='support-a', wrong_pairing_eligible=False,
        condition=None, task=None) for i in range(2)]
    record = dict(id='record', dataset='eeg_fnirs_single_trial', subject='heldout',
        record='native', fold=0, pairs=[pair], windows=windows,
        continuous_hb_segments=[dict(id='support-a', start_s=0., end_s=60.)])
    clock = np.arange(300)*.1
    from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
    curves = []
    for index in range(2):
        r = (.01+.003*index)*np.sin(2*np.pi*clock/15)
        forward = nonlinear_driver_forward(r, np.array([0., 1., 1., 1.]), runner.parameters(cfg), .1,
            derivative=False, numerical_backend='numba', hemodynamics='tak', viscoelastic_s=2.)
        curves.append((runner.reference_operator()@forward['canonical_prediction'].ravel()).reshape(300, 3))
    target = np.array(curves)
    eeg = np.zeros((2, 300, 6, 5)); eeg[:, :, 0, 0] = target[..., 0]
    hb = target[..., 1:][:, :, None, :]
    meta, path, _ = runner.prepared_paths(tmp_path, record['id'])
    runner.write_json(meta, dict(status='completed'))
    runner.write_npz(path, eeg_logpower=eeg, hb=hb, hb_bin_phase_s=np.zeros(2),
        continuous_hb=hb.reshape(600, 1, 2), continuous_hb_time_s=np.arange(600)*.1,
        continuous_hb_valid_mask=np.ones((600, 1, 2), bool),
        continuous_hb_gap_valid_mask=np.ones(600, bool))
    pc = np.zeros(30); pc[0] = 1.
    model = dict(coordinate=dict(hb_factor=1., eeg_factor=1., sd=[.025, .025, .015],
        pc={name: pc for name in runner.CANDIDATES}), training_subjects=['training'])
    runner.write_json(tmp_path/'spatial_selection.json',
        dict(models={runner._group_key(record, pair): {'0': model}}))
    plan = dict(records=[record])
    return cfg, plan, target


def test_groups_never_share_parameters_across_native_support(tmp_path):
    _, plan, _ = _fixture(tmp_path)
    record = plan['records'][0]
    assert len(list(runner.current_dof_groups(plan))) == 1
    record['windows'][1]['native_support_id'] = 'support-b'
    groups = list(runner.current_dof_groups(plan))
    assert len(groups) == 2
    assert {group['native_support_id'] for _, group in groups} == {'support-a', 'support-b'}
    assert all(len(group['tasks']) == 1 for _, group in groups)


def test_current_views_sanitize_hidden_data_and_warm_only_from_same_view(tmp_path, monkeypatch):
    cfg, plan, _ = _fixture(tmp_path)
    record, group = next(runner.current_dof_groups(plan))
    for task in group['tasks']:
        runner.write_npz(tmp_path/'tasks'/f'{task["id"]}.npz',
            driver_h=np.full(300, .001), driver_j=np.full(300, .002),
            hb_only_initial_state=np.array([0., 1., 1., 1., 1.]),
            joint_initial_state=np.array([.01, 1., 1., 1., 1.]))
    calls = []
    def capture(observations, parameters, dt, **kwargs):
        calls.append((observations.copy(), deepcopy(kwargs)))
        return dict(status='failed_domain', converged=False, parameter_values=None)
    monkeypatch.setattr('src.inference.shared_driver_reconstruction.fit_nonlinear_shared_parameters', capture)
    result = runner._current_group_job((cfg, str(tmp_path), record, group))
    assert result['terminal'] == 2 and len(calls) == 2
    hb, joint = calls
    assert np.isnan(hb[0][..., 0]).all()
    assert np.isfinite(joint[0][:, :299, 0]).all()
    for (values, options), level in zip(calls, (.001, .002)):
        assert values.shape == (2, 300, 3)
        np.testing.assert_allclose(options['parameter_prior_precision'], [16., 16.])
        assert len(options['starts']) == 3
        assert options['parameter_names'] == ['kappa', 'tau']
        for start in options['starts']:
            np.testing.assert_array_equal(start['driver'], np.full((2, 300), level))
    stored = runner.read_json(tmp_path/'current_dof'/'tasks'/f'{group["id"]}.json')
    assert len(stored['rows']) == 2
    assert all(row['D1_free'] is None for row in stored['rows'])
    assert all(not row['success'] for row in stored['rows'])


def test_shared_parameter_companion_replays_fitted_parameters_and_resumes(tmp_path, monkeypatch):
    cfg, plan, _ = _fixture(tmp_path)
    record, group = next(runner.current_dof_groups(plan))
    result = runner._current_group_job((cfg, str(tmp_path), record, group))
    stored = runner.read_json(tmp_path/'current_dof'/'tasks'/f'{group["id"]}.json')
    assert result['terminal'] == 2
    assert 'error' not in stored, stored.get('traceback')
    assert stored['status'] == 'completed'
    for view in ('hb_only', 'joint'):
        entry = stored['views'][view]
        assert entry['converged'] and entry['fitted'] == 2
        assert len(entry['starts']) == 3
        assert entry['prior_multiplicity'] == 'once_per_group'
        assert max(entry['integration_difference_training_sd']) < .005
    for row in stored['rows']:
        assert row['D1_free'] is not None and row['D4_free'] is not None
        assert row['D_obs_free'] is None  # No successful fixed EEG fit was supplied.
        with np.load(row['array_path'], allow_pickle=False) as arrays:
            i = row['array_index']; mask = arrays['score_mask'][i, :, 1:].all(1)
            residual = (arrays['target'][i, :, 1:]-arrays['hb_only_prediction'][i, :, 1:])/arrays['sd'][1:]
            np.testing.assert_allclose(row['D1_free'], np.sqrt(np.mean(residual[mask]**2)))
    monkeypatch.setattr(runner, '_load_prepared', lambda *a, **k: (_ for _ in ()).throw(AssertionError('resume read')))
    assert runner._current_group_job((cfg, str(tmp_path), record, group)) == result
