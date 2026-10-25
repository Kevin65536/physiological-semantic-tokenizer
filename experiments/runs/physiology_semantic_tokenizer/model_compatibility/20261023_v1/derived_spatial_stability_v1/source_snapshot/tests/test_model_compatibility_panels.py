"""Synthetic frozen-panel contracts; no local measured artifacts are consumed."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from threadpoolctl import threadpool_limits


SPEC = importlib.util.spec_from_file_location('compatibility_panels',
    Path(__file__).parents[1]/'experiments/scripts/evaluate_model_compatibility_panels.py')
panels = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(panels)


def config():
    return dict(data=dict(protected_data='forbidden'), panels=dict(
        output_support_runs=list(panels.OUTPUT_SUPPORT_RUNS), spatial_run=panels.SPATIAL_RUN,
        lengths_s=[30, 90], linear=dict(tau_grid=list(np.arange(.5, 8.001, .5)),
            kappa_grid=[.2, .4, .64, 1., 1.5], tolerances=[.25, .5, 1.], rcond=1e-10),
        nonlinear=dict(max_nfev=100, substeps=4, validation_substeps=8)))


@pytest.fixture
def frozen_support(tmp_path):
    cfg = config()
    ref = None
    for i, relative in enumerate(panels.OUTPUT_SUPPORT_RUNS):
        p = tmp_path/relative
        p.mkdir(parents=True)
        ds = 'eeg_fnirs_single_trial' if i == 0 else 'refed'
        row = dict(id='record_'+str(i), dataset=ds, subject='eval_subject', duration_s=90,
                   partition='evaluation', coordinate='frozen')
        panels.write_json(p/'summary.json', dict(schema='ssm_output_support_summary_v1', status='completed'))
        panels.write_json(p/'plan.json', dict(schema='ssm_output_support_plan_v1', records=[row],
                                            evaluation_subjects={ds: ['eval_subject']}))
        (p/'resolved_config.yaml').write_text(yaml.safe_dump(dict(
            schema='ssm_output_support_v1' if i == 0 else 'ssm_output_support_refed_v2',
            data=dict(protected_data='forbidden'))))
        panels.write_json(p/'coordinates.json', dict(frozen=dict(
            pc=[1.]+[0.]*29, eeg_factor=2., hb_factor=3., sd=[1., 2., 4.],
            training_subjects=['train_subject'])))
        panels.write_json(p/'tasks'/'prepare'/(row['id']+'.json'), dict(status='completed'))
        operator = np.eye(360)
        operator[:, :20] -= 1/20
        x = np.arange(360, dtype=float)
        eeg = np.zeros((360, 30)); eeg[:, 0] = x
        panels.write_npz(p/'prepared'/(row['id']+'.npz'), eeg=eeg,
                         hb=np.column_stack((x, -x)), eeg_operator=operator, hb_operator=operator)
        if ref is None:
            ref = dict(row, parent=relative)
    return cfg, ref, tmp_path


def test_retained_crop_preserves_training_coordinates_and_references(frozen_support):
    cfg, ref, root = frozen_support
    with threadpool_limits(limits=1):
        record = panels.load_output_panel(cfg, ref, 30, root)
    assert record['target'].shape == (120, 3)
    np.testing.assert_array_equal(record['sd'], [1., 2., 4.])
    # Same physical bins, transformed by the retained PC/gains, then referenced.
    reference = np.arange(120)-np.arange(20).mean()
    np.testing.assert_allclose(record['target'], reference[:, None]*np.array([2., 3., -3.]))
    np.testing.assert_allclose(record['target'][:20].mean(0), 0., atol=1e-12)
    assert record['score_mask'][:-1].all()
    assert not record['score_mask'][-1].any()
    assert record['operator'].shape == (360, 360)


@pytest.mark.parametrize('corruption', ['protected', 'overlap', 'identity', 'preparation', 'symlink'])
def test_every_array_read_fails_closed_on_metadata_or_path(frozen_support, monkeypatch, corruption):
    cfg, ref, root = frozen_support
    parent = root/ref['parent']
    if corruption == 'protected':
        path = parent/'resolved_config.yaml'
        content = yaml.safe_load(path.read_text()); content['data']['protected_data'] = 'allowed'
        path.write_text(yaml.safe_dump(content))
    elif corruption == 'overlap':
        path = parent/'coordinates.json'
        content = json.loads(path.read_text()); content['frozen']['training_subjects'].append(ref['subject'])
        panels.write_json(path, content)
    elif corruption == 'identity':
        ref = dict(ref, duration_s=30)
    elif corruption == 'preparation':
        panels.write_json(parent/'tasks'/'prepare'/(ref['id']+'.json'), dict(status='failed'))
    else:
        path = parent/'prepared'/(ref['id']+'.npz')
        path.unlink()
        path.symlink_to(root/'outside.npz')
    def forbidden_read(*args, **kwargs):
        raise AssertionError('np.load was reached before metadata admission')
    monkeypatch.setattr(panels.np, 'load', forbidden_read)
    with pytest.raises(ValueError):
        panels.load_output_panel(cfg, ref, 30, root)


def test_crop_rejects_scored_halo_but_excludes_last_interpolated_bin():
    y = np.zeros((360, 3))
    operator = np.eye(360)
    operator[:, :20] -= 1/20
    operator[239, 240] = .5
    with threadpool_limits(limits=1):
        panels.crop_output_panel(y, dict(eeg=operator, hb=operator), 30)
        operator[238, 240] = .5
        with pytest.raises(ValueError, match='outside-window'):
            panels.crop_output_panel(y, dict(eeg=operator, hb=operator), 30)


@pytest.mark.parametrize('operator_name', ['eeg_operator', 'hb_operator'])
def test_parent_operator_with_constant_offset_is_unsupported(frozen_support, operator_name):
    cfg, ref, root = frozen_support
    path = root/ref['parent']/'prepared'/(ref['id']+'.npz')
    with np.load(path, allow_pickle=False) as values:
        arrays = {key: values[key].copy() for key in values.files}
    arrays[operator_name][150, 150] += .01
    panels.write_npz(path, **arrays)
    with threadpool_limits(limits=1):
        with pytest.raises(ValueError, match='constants must be annihilated'):
            panels.load_output_panel(cfg, ref, 30, root)


def test_feasible_nodes_intersect_same_theta_without_splicing_profile_minima():
    profiles = [dict(tau=.5, kappa=.2, hb_nmse=.01, cross_increment_nmse=2., joint_nmse=.7,
                     joint_eeg_nmse=2., joint_hbo_nmse=.01, joint_hbr_nmse=.01),
                dict(tau=2., kappa=.64, hb_nmse=2., cross_increment_nmse=.01, joint_nmse=.7,
                     joint_eeg_nmse=.01, joint_hbo_nmse=2., joint_hbr_nmse=.01)]
    result = panels.compatibility_sets(profiles, [.5], [.5, 2.], [.2, .64])
    assert result[0]['n_nodes'] == 0
    # Small averaged Hb/cross/joint residuals cannot hide one failing modality.
    profiles.append(dict(tau=1.5, kappa=1., hb_nmse=.04, cross_increment_nmse=.09, joint_nmse=.1267,
                         joint_eeg_nmse=.01, joint_hbo_nmse=.01, joint_hbr_nmse=.36))
    profiles.append(dict(tau=1., kappa=.4, hb_nmse=.04, cross_increment_nmse=.09, joint_nmse=.06,
                         joint_eeg_nmse=.09, joint_hbo_nmse=.05, joint_hbr_nmse=.04))
    result = panels.compatibility_sets(profiles, [.5], [.5, 1., 1.5, 2.], [.2, .4, .64, 1.])
    assert result[0]['n_nodes'] == 1
    assert result[0]['n_hb_cross_nodes'] == 2
    assert result[0]['nodes'] == [[1., .4]]
    assert result[0]['tau_width'] == 0
    assert not result[0]['hit_boundary']
    assert result[0]['max_normalized_distance_to_boundary'] > 0
    assert not result[0]['all_nodes_on_boundary']


def test_nonlinear_uses_one_selected_theta_and_reports_its_retained_witness(tmp_path, monkeypatch):
    cfg = config()
    task = dict(id='stable_id', ref=dict(id='record', dataset='refed', subject='s', parent=panels.OUTPUT_SUPPORT_RUNS[1]), length_s=30)
    folder = tmp_path/'tasks'/'nonlinear'
    panels.write_json(folder.parent/'linear'/'stable_id.json', dict(
        panels._identity(task), status='completed', best_joint=dict(tau=3., kappa=.4)))
    n = 8
    mask = np.ones((n, 3), bool); mask[-1] = False
    record = dict(target=np.ones((n, 3)), operator=np.eye(3*n), sd=np.array([1., 2., 3.]), score_mask=mask)
    monkeypatch.setattr(panels, 'load_output_panel', lambda *args: record)
    def solve(hb, p, dt, **kwargs):
        assert p.free.tau == 3. and p.free.kappa == .4
        assert dt == .25
        assert len(kwargs['starts']) == 2
        assert kwargs['starts'][0]['coefficients'].shape == (n,)
        assert kwargs['max_nfev'] == 100
        assert not kwargs['score_mask'][-1].any()
        np.testing.assert_array_equal(kwargs['hb_operator'], np.eye(2*n))
        return dict(status='completed', legal_witness=True, optimizer_converged=False,
            weighted_data_sse=4., minimum_legal_weighted_data_sse=1.,
            prediction=np.zeros((n, 2)))
    monkeypatch.setattr(panels, 'fit_vascular_flow_relaxation', solve)
    result = panels._nonlinear_task(cfg, task, folder, tmp_path)
    # The score must audit the saved selected prediction rather than switch to
    # the different minimum-SSE candidate returned only as a solver diagnostic.
    assert result['D_Hb_NL_recon_nmse'] == pytest.approx(4/(2*(n-1)))
    assert result['selected_tau'] == 3.
    assert result['legal_witness'] and not result['optimizer_converged']


def test_fixed_spatial_distance_and_free_rank_bound_are_distinct():
    t = np.arange(24, dtype=float)
    y = np.stack((t[:, None]*np.array([[1., 2., 3., 4.]]),
                  -t[:, None]*np.array([[4., 3., 2., 1.]])), axis=2)
    loading = np.ones((4, 1))
    with threadpool_limits(limits=1):
        result, arrays = panels.spatial_distances(y/2, y, np.eye(24), loading,
            np.array([[1., 2.], [2., 3.], [3., 4.], [4., 5.]]))
    assert result['D_space_fraction_raw'] > .1
    assert result['HbO_free_rank_K_fraction_raw'] < 1e-25
    assert result['HbR_free_rank_K_fraction_raw'] < 1e-25
    assert result['D_rank_joint'] > .01
    assert result['D_rank_joint'] <= result['D_space_fraction_raw']+1e-12
    assert result['weighted_lower_bound_nmse'] <= result['weighted_nmse']+1e-10
    assert result['raw_energy'] == pytest.approx(np.sum(y[20:]**2))
    np.testing.assert_array_equal(arrays['hb_loadings'], loading)
    assert not arrays['score_mask'][:20].any()
    assert arrays['score_mask'][20:].all()


@pytest.mark.parametrize('singular', [False, True])
def test_spatial_raw_projection_avoids_ill_conditioned_temporal_inverse(singular):
    rng = np.random.default_rng(20261023)
    n = 120
    temporal_left = np.linalg.qr(rng.normal(size=(100, 100)))[0]
    seeds = rng.normal(size=(n, 100)); seeds -= seeds.mean(axis=0)
    temporal_right = np.linalg.qr(seeds, mode='reduced')[0]
    spectrum = np.geomspace(1., 1e-12, 100)
    if singular:
        spectrum[-1] = 0.
    operator = np.zeros((n, n))
    operator[20:] = (temporal_left*spectrum)@temporal_right.T
    loading = np.array([[.9, .1], [.6, .4], [.3, .7], [.1, .9]])
    raw = .01*rng.normal(size=(n, 4, 2))
    basis = np.linalg.qr(loading, mode='reduced')[0]
    expected = np.stack([raw[:, :, j]@basis@basis.T for j in range(2)], axis=2)
    with threadpool_limits(limits=1):
        result, arrays = panels.spatial_distances(raw, raw, operator, loading, np.ones((4, 2)))
    np.testing.assert_allclose(arrays['raw_prediction'], expected, atol=1e-15, rtol=1e-13)
    expected_fraction = np.sum((expected[20:]-raw[20:])**2)/np.sum(raw[20:]**2)
    assert result['D_space_fraction_raw'] == pytest.approx(expected_fraction, abs=1e-14)
    assert result['raw_spatial_projection_interpretation'] == 'stable_fixed_spatial_projection_relax_temporal_constraints'
    for j in range(2):
        np.testing.assert_allclose((arrays['raw_prediction'][20:, :, j]-raw[20:, :, j])@loading,
                                   0., atol=1e-15)


def test_spatial_relaxation_rejects_different_channel_time_operators():
    operators = np.broadcast_to(np.eye(24), (4, 24, 24)).copy()
    operators[1, 20, 20] = .5
    with pytest.raises(ValueError, match='common temporal operator'):
        panels.spatial_distances(np.ones((24, 4, 2)), np.ones((24, 4, 2)),
                                operators, np.ones((4, 1)), np.ones((4, 2)))


def test_terminal_failure_is_retained_and_never_silently_retried(tmp_path, monkeypatch):
    task = dict(id='failure', ref=dict(id='record', dataset='refed', subject='s', parent='parent'), length_s=30)
    calls = []
    def fail(*args):
        calls.append(True)
        raise RuntimeError('known synthetic failure')
    monkeypatch.setattr(panels, '_linear_task', fail)
    payload = (config(), task, 'linear', str(tmp_path), str(tmp_path))
    first = panels._execute(payload)
    second = panels._execute(payload)
    assert first == second
    assert first['status'] == 'failed'
    assert len(calls) == 1
    assert 'known synthetic failure' in first['error']


def test_linear_task_retains_one_full_prediction_and_auditable_component_scores(tmp_path, monkeypatch):
    from src.inference.shared_driver_reconstruction import build_shared_driver_design
    cfg = config()
    n = 16
    operator = np.eye(3*n)
    with threadpool_limits(limits=1):
        design = build_shared_driver_design(panels.parameters(cfg), n, .25,
                                             processed_mean_operator=operator)
    coefficients = .001*np.sin(np.arange(n+5))
    target = (design.observation_design@coefficients).reshape(n, 3)+design.offset
    mask = np.ones((n, 3), bool); mask[-1] = False
    record = dict(target=target, operator=operator, sd=np.array([1., 2., 3.]), score_mask=mask,
                  pc=np.ones(30)/np.sqrt(30), eeg_factor=1., hb_factor=1.)
    monkeypatch.setattr(panels, 'load_output_panel', lambda *args: record)
    task = dict(id='linear', ref=dict(id='record', dataset='refed', subject='s',
                parent=panels.OUTPUT_SUPPORT_RUNS[1]), length_s=30)
    folder = tmp_path/'tasks'/'linear'
    with threadpool_limits(limits=1):
        result = panels._linear_task(cfg, task, folder, tmp_path)
    assert len(result['profiles']) == 80
    assert result['best_joint']['joint_nmse'] < 1e-20
    assert result['dynamic_support']
    with np.load(folder/'linear.npz', allow_pickle=False) as values:
        prediction = values['best_joint_joint_prediction']
        for j, key in enumerate(('joint_eeg_nmse', 'joint_hbo_nmse', 'joint_hbr_nmse')):
            nmse = np.mean(((prediction[:-1, j]-target[:-1, j])/record['sd'][j])**2)
            assert nmse == pytest.approx(result['best_joint'][key], abs=1e-20)
        assert values['best_joint_joint_coefficients'].shape == (n+8,)
    assert result['score_count'] == 3*(n-1)
