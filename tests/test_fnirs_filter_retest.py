"""Small fixtures for the post-10.20, no-extra-component filter sensitivity."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest
import yaml

from experiments.scripts import evaluate_shared_driver_reconstruction as run


CONFIG = run.CODE_ROOT / 'experiments/configs/physiology_semantic_tokenizer/shared_driver_filter_retest_v1.yaml'


def test_contract_excludes_extra_components_before_arrays(tmp_path, monkeypatch):
    cfg, base = run.filter_retest_config(CONFIG)
    assert cfg['arms'] == ['original', 'slow', 'no_decay']
    assert cfg['solver']['max_evaluations'] == base['solver']['max_evaluations']
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('unexpected array read'))
    for key, value in [('forcing_modes', 7), ('arms', ['original', 'driven_original']),
                       ('protected_data', 'allowed'), ('raw_source_reads', 'allowed')]:
        changed = deepcopy(cfg); changed[key] = value
        path = tmp_path / 'bad.yaml'; path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError, match='contract changed'):
            run.filter_retest_config(path)


def test_selection_uses_only_old_baseline_and_retains_denominators():
    rows = [dict(id=str(i), dataset='fixture', subject=f's{i}') for i in range(5)]
    scores = {'0': 1., '1': 2., '2': 100., '3': 3.}
    selected, counts = run.filter_retest_select(rows, scores, ['fixture'])
    assert selected[0]['id'] == '1'  # equally close to median; lexical tie break
    assert selected[0]['selection_median'] == 2.5
    assert counts['fixture']['planned_records'] == 5
    assert counts['fixture']['usable_records'] == 4
    assert counts['fixture']['missing_or_failed'] == ['4']


def test_selection_reads_corrected_fit_qualification(tmp_path):
    name = 'case_90_joint_full.json'
    old = tmp_path / 'measured'; old.mkdir()
    (old / name).write_text(json.dumps({'qualified': False, 'reason': 'counterfactual_failure'}))
    assert run.filter_retest_source_task(tmp_path, 'case', 90, 'joint_full')['qualified'] is False
    corrected = tmp_path / 'evidence_v2' / 'measured'; corrected.mkdir(parents=True)
    (corrected / name).write_text(json.dumps({'qualified': True, 'reason': 'fitted_trajectory_valid'}))
    assert run.filter_retest_source_task(tmp_path, 'case', 90, 'joint_full')['qualified'] is True


def test_low_nrmse_does_not_hide_inverted_waveform_and_wrong_amplitude():
    t = np.arange(120) * .25
    y = np.tile(.001 * np.sin(t)[:, None], (1, 3))
    reversed_metrics = run.filter_retest_waveform_metrics(y, -y, np.ones(3))
    assert all(row['nrmse'] < .01 and row['pearson_r'] < -.99 for row in reversed_metrics)
    assert all(row['signed_gain'] == pytest.approx(-1) for row in reversed_metrics)
    scaled = run.filter_retest_waveform_metrics(y, 0.1 * y + .1, np.ones(3),
                                               hb_factor=2., common_sd=np.array([.02, .03]))
    assert scaled[1]['centered_amplitude_ratio'] == pytest.approx(.1)
    assert scaled[1]['pearson_r'] == pytest.approx(1.)
    assert scaled[1]['nrmse_common_cached_training_sd'] == pytest.approx(
        scaled[1]['absolute_rmse_in_cache_unit'] / .02)


def test_all_hidden_fitting_inputs_are_nan_and_forcing_is_absent(tmp_path, monkeypatch):
    from src.inference import shared_driver_reconstruction as fitting
    from src.inference.shared_driver_rk4 import decay_forward
    cfg, base = run.filter_retest_config(CONFIG)
    y, op, sd, _ = run.decay_synthetic(base, 0, 'matched', 30)
    path = tmp_path / 'prepared'; path.mkdir()
    np.savez(path / 'case__unified_cache__30.npz', target=y, operator=op, sd=sd,
             hb_factor=1., common_cached_sd_physical=sd[1:])
    calls = []
    def fit(observed, parameters, dt, **kwargs):
        assert kwargs['forcing_modes'] == 0
        mask = kwargs['visible']
        assert np.isnan(observed[~mask]).all()
        calls.append(kwargs['decay_multiplier'])
        driver = np.zeros(len(y)); initial = np.r_[0., np.ones(4)]
        forward = decay_forward(driver, initial, parameters, dt,
            decay_multiplier=kwargs['decay_multiplier'], forcing_coefficients=np.empty(0), substeps=4)
        prediction = (op @ forward['canonical_prediction'].ravel()).reshape(y.shape)
        return dict(prediction=prediction, states=forward['states'], driver=driver, initial_state=initial,
            converged=True, status='fixture', objective=1., scaled_gradient_inf_norm=0., costs={}, evaluations=1)
    monkeypatch.setattr(fitting, 'fit_decay_shared_driver', fit)
    spec = dict(key='fixture', ref=dict(id='case'), branch='unified_cache',
                length_s=30, condition='joint_blocked')
    result = run.filter_retest_worker((cfg, base, str(tmp_path), str(tmp_path), spec))
    assert result['status'] == 'completed'
    assert calls == [1., .1, 0.]
    saved = json.loads((tmp_path / 'tasks/fixture.json').read_text())
    assert len(saved['rows']) == 3
    assert all(row['forcing_modes'] == 0 and not row['input_present'] for row in saved['rows'])
    assert all(row['integration_max_training_sd'] < 1e-12 for row in saved['rows'])


def test_supervision_checked_before_fitting(tmp_path, monkeypatch):
    cfg, base = run.filter_retest_config(CONFIG)
    monkeypatch.delenv('INVOCATION_ID', raising=False)
    monkeypatch.delenv('SSM_SYSTEMD_UNIT', raising=False)
    with pytest.raises(ValueError, match='durable'):
        run.filter_retest_run(cfg, base, tmp_path, tmp_path, dict(jobs=[]), 1)
