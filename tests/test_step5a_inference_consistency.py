from __future__ import annotations

import copy
import inspect

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, t

import experiments.evaluate_step5a_inference_consistency as step5


def tiny_config():
    cfg = copy.deepcopy(step5.load_config())
    cfg['replicates_per_axis'] = 1
    cfg['model']['steps'] = 12
    cfg['oracle']['steps'] = 24
    cfg['inference'].update(grid_points=5, mask_center_steps=4, bootstrap_repetitions=20)
    cfg['reference'].update(steps=4, particles=[32, 64], grid_points=5)
    return cfg


def test_contract_rejects_wrong_scope_priors_and_reference_budget():
    cfg = step5.load_config()
    step5.validate_config(cfg)
    for section, key, value in (
        (None, 'scope', 'measured'), (None, 'output_root', 'data/test'),
        ('reference', 'particles', [32, 16]), ('reference', 'independent_runs', 1),
        ('inference', 'grid_points', 4), ('inference', 'mask_center_steps', 1000),
    ):
        bad = copy.deepcopy(cfg)
        (bad if section is None else bad[section])[key] = value
        with pytest.raises(ValueError):
            step5.validate_config(bad)
    bad = copy.deepcopy(cfg)
    bad['axes']['Z']['prior'] = 'logistic_uniform'
    with pytest.raises(ValueError):
        step5.validate_config(bad)
    assert 'truth' not in inspect.signature(step5.fit_score_grid).parameters
    assert 'driver' not in inspect.signature(step5.fit_score_grid).parameters


def test_gwz_and_linear_gaussian_specialization():
    cfg = step5.load_config()
    assert step5.parameterization_check(cfg)['passed']
    result = step5.linear_gaussian_check(cfg)
    assert result['passed']
    # Marginal scoring is not the joint likelihood even when state filtering is exact.
    assert abs(result['score_minus_joint']) > 1e-4


def test_matched_generator_has_declared_initial_and_six_state_noise_law():
    cfg = tiny_config()
    cfg['model']['steps'] = 2
    cfg['model']['process_std'] = [.08, .01, .006, .004, .004, .004]
    initial, process_noise_increments = [], []
    raw = step5.raw_parameters(cfg, 'W', .1)
    for seed in range(800):
        trial = step5.generate_matched(cfg, 'W', .1, seed)
        z = trial['transformed_states']
        initial.append(z[0])
        process_noise_increments.append(z[1]-step5.independent_transition(z[0], raw, cfg))
    np.testing.assert_allclose(np.std(initial, axis=0), cfg['model']['initial_state_std'], rtol=.12)
    np.testing.assert_allclose(np.std(process_noise_increments, axis=0), np.array(cfg['model']['process_std'])*np.sqrt(cfg['model']['dt']), rtol=.12)
    np.testing.assert_array_equal(step5.generate_matched(cfg,'W',.1,10)['observations'], step5.generate_matched(cfg,'W',.1,10)['observations'])


def test_particle_joint_density_against_exact_single_eeg_integral():
    cfg = tiny_config()
    observed = .09
    sigma, scale, nu = cfg['model']['initial_state_std'][0], cfg['model']['observation_scale'][0], cfg['model']['student_nu']
    exact, error = quad(lambda r: norm.pdf(r,scale=sigma)*t.pdf((observed-r)/scale,df=nu)/scale, -1,1, epsabs=1e-10)
    assert error < 1e-8
    estimates = [step5.particle_filter(np.array([[observed,np.nan,np.nan]]), cfg,'G',0.,40000,seed)['parameter_log_likelihood'] for seed in range(4)]
    likelihood = np.exp(estimates)
    assert abs(np.log(likelihood.mean())-np.log(exact)) < .015
    missing = step5.particle_filter(np.full((3,3),np.nan),cfg,'G',0.,64,42,paths=True)
    assert missing['parameter_log_likelihood'] == pytest.approx(0.)
    assert missing['mean'].shape == (3,4)
    assert np.all(missing['variance'] >= 0)


def test_quadrature_keeps_uniform_physical_zeta_prior_and_correct_mass():
    cfg = step5.load_config()
    grid = np.linspace(*cfg['axes']['Z']['bounds'],65)
    posterior = step5.posterior_grid(cfg,'Z',grid,np.zeros(len(grid)))
    np.testing.assert_allclose(posterior['cdf'],np.linspace(0,1,len(grid)),atol=1e-14)
    result = step5.posterior_summary(cfg,'Z',grid,np.zeros(len(grid)),.5)
    assert result['boundary_mass'] == pytest.approx(.1)
    assert result['mean'] == pytest.approx(.575)


def test_small_case_has_separate_scores_truth_metrics_and_no_mask_leakage(monkeypatch):
    cfg = tiny_config()
    step5.validate_config(cfg)
    original = step5.fit_score_grid
    masked_calls = []
    def spy(y, *args):
        if np.isnan(y).any():
            masked_calls.append(y.copy())
        return original(y,*args)
    monkeypatch.setattr(step5,'fit_score_grid',spy)
    row, payload = step5.run_case(cfg,'G',0)
    assert row['oracle_r_known']['parameter_log_likelihood'] is not None
    assert row['nonlinear_reference']['parameter_log_likelihood'] is not None
    assert len(masked_calls) == 2
    for y in masked_calls:
        assert np.isnan(y[4:8,1:]).all()
        assert np.isfinite(y[:,0]).all()
    for law in ('matched_model_calibration','misspecification_stress_test'):
        assert row[law]['parameter_log_likelihood'] is None
        state = row[law]['state_recovery']['true_parameters']['full']
        assert set(state) == set(step5.TARGETS)
        for target in state.values():
            assert 0 <= target['coverage95'] <= 1
        assert payload[f'{law}_states'].shape == (12,6)
    summary = step5.summarize(dict(cfg,axes={'G':cfg['axes']['G']}),[row],{})
    assert summary['independent_cases'] == 1
    assert summary['scientific_verdict'] == 'localization_only_no_admission'


def test_oracle_refinement_keeps_cases_threshold_and_original_evidence(tmp_path, monkeypatch):
    import hashlib
    import json
    import yaml
    cfg = tiny_config()
    root = tmp_path/cfg['output_root']
    source = root/'original'
    source.mkdir(parents=True)
    (source/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
    (source/'manifest.json').write_text(json.dumps(dict(execution='completed', scope=cfg['scope'])))
    for axis in cfg['axes']:
        theta = .5 if axis == 'Z' else .1
        seed = 15
        original = dict(truth=theta, seed=seed, oracle_r_known=step5.oracle_case(cfg,axis,theta,seed+1))
        step5.write_json(source/f'case_{axis}_00.json', original)
    before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    monkeypatch.setattr(step5, 'REPO_ROOT', tmp_path)
    destination = root/'refined'
    result = step5.refine_oracles(source,destination)
    assert result['cases'] == 3
    after = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    assert before == after
    resolved = yaml.safe_load((destination/'resolved_config.yaml').read_text())
    assert resolved['checks'] == cfg['checks']
    assert resolved['inference']['grid_points'] == 9
    assert json.loads((destination/'manifest.json').read_text())['additional_independent_cases'] == 0
    with pytest.raises(FileExistsError):
        step5.refine_oracles(source,destination)
