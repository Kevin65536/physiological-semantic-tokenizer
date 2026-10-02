"""Synthetic-only contracts for observation projection and teacher robustness."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.inference import shared_driver_reconstruction as inference
from src.inference.t3a_balloon_robust_ssm import BalloonParameters


def fixture():
    p = BalloonParameters()
    n = 24
    driver = .025*np.sin(np.arange(n)*.3)
    initial = np.r_[0., np.ones(4)]
    y = inference.nonlinear_driver_forward(driver, initial, p, .25, derivative=False)['canonical_prediction']
    basis = inference.waveform_component_basis(p, n, .25, np.eye(3*n), 'volume', modes=2)
    return p, y, basis


def test_profiled_observation_jacobian_and_hidden_value_invariance():
    p, y, basis = fixture()
    n = len(y)
    mask = np.ones_like(y, dtype=bool);mask[8:14, 1:] = False
    scale = np.full_like(y, .025)
    wb = basis[mask.ravel()]/scale[mask, None]
    cs = np.array([.03, .04])
    mapping = -np.linalg.solve(wb.T@wb+np.diag(cs**-2), wb.T)
    component = basis, wb, mapping, cs
    x = np.r_[np.zeros(n), np.zeros(5)]
    def evaluate(x, derivative):
        return inference._nonlinear_trial_objective(x, p, .25, None, y, mask, scale,
            np.zeros((0,n+5)), 10., 4, derivative, observation_component=component)
    fit = evaluate(x, True)
    for j in (0, 10, n+1, n+4):
        delta = np.eye(len(x))[j]*1e-6
        numeric = (evaluate(x+delta, False)['residual']-evaluate(x-delta, False)['residual'])/2e-6
        np.testing.assert_allclose(fit['jacobian'][:,j], numeric, rtol=2e-5, atol=2e-7)
    kwargs = dict(visible=mask, sd=scale, observation_basis=basis, observation_coefficient_sd=cs,
        penalty=.01, initial_penalty=100., max_evaluations=160)
    a = inference.fit_nonlinear_shared_driver(y, p, .25, **kwargs)
    changed = y.copy();changed[~mask] = 1e20
    b = inference.fit_nonlinear_shared_driver(changed, p, .25, **kwargs)
    np.testing.assert_array_equal(a['driver'], b['driver'])
    np.testing.assert_array_equal(a['observation_coefficients'], b['observation_coefficients'])
    np.testing.assert_allclose(a['prediction'], a['physical_prediction']+a['observation_component'])
    np.testing.assert_array_equal(a['observation_component'][:,0], 0.)


def test_component_has_proper_penalty_and_cannot_observe_EEG():
    p, y, basis = fixture()
    bad = basis.copy();bad[0,0] = 1.
    with pytest.raises(ValueError, match='zero EEG'):
        inference.fit_nonlinear_shared_driver(y, p, .25, observation_basis=bad)
    with pytest.raises(ValueError, match='SD'):
        inference.fit_nonlinear_shared_driver(y, p, .25, observation_basis=basis, observation_coefficient_sd=0.)
    visible = np.zeros_like(y, dtype=bool);visible[:,0] = True
    fit = inference.fit_nonlinear_shared_driver(y, p, .25, visible=visible, observation_basis=basis,
        observation_coefficient_sd=.1, max_evaluations=160)
    np.testing.assert_array_equal(fit['observation_coefficients'], 0.)


def test_proper_prior_penalizes_both_constant_and_linear_drivers():
    p, y, _ = fixture()
    for driver in (np.ones(len(y))*.02, np.linspace(-.02,.03,len(y))):
        np.testing.assert_allclose(np.diff(driver,n=2), 0., atol=1e-15)
        result = inference.fit_nonlinear_shared_driver(y, p, .25,
            starts=[dict(driver=driver,initial_state=np.r_[0.,np.ones(4)])],
            driver_amplitude_weight=.1, driver_prior_sd=.025, max_evaluations=1)
        assert result['driver_amplitude_cost'] == pytest.approx(.1*.25*np.sum((driver/.025)**2))
        assert result['driver_amplitude_cost'] > 0


def test_versioned_boundary_and_counterfactual_identity(tmp_path):
    import yaml
    from experiments.scripts import evaluate_shared_driver_reconstruction as runner
    root=Path(__file__).resolve().parents[1]
    cfg=runner.robustness_config(root/'experiments/configs/physiology_semantic_tokenizer/shared_driver_teacher_robustness_v1.yaml')
    base=runner.semantics_config(root/cfg['source_config'])
    a=runner.robustness_synthetic_case(cfg,base,0,'mixed','nonrest','control')
    b=runner.robustness_synthetic_case(cfg,base,0,'mixed','nonrest','common_correlated')
    np.testing.assert_array_equal(a['driver'],b['driver'])
    np.testing.assert_array_equal(a['physical_truth'],b['physical_truth'])
    np.testing.assert_allclose(b['target']-a['target'],b['observation_truth'],atol=1e-16)
    changed=runner.robustness_synthetic_case(cfg,base,0,'mixed','nonrest','driver_amplitude')
    np.testing.assert_allclose(changed['driver'],1.5*a['driver'])
    calibration=runner.robustness_synthetic_case(cfg,base,0,'mixed','nonrest','control',calibration=True)
    assert not np.array_equal(calibration['driver'],a['driver'])
    cfg['source_run']='comparative_methods/protected'
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='boundary'):runner.robustness_config(path)


def test_teacher_missing_or_unconverged_alternative_cannot_give_reliable_mask():
    from experiments.scripts import evaluate_shared_driver_reconstruction as runner
    fit=dict(driver=np.linspace(-1,1,24),converged=True,
        physical_prediction=np.zeros((24,3)),observation_component=np.zeros((24,3)),prediction=np.zeros((24,3)))
    uncertain=dict(fit,converged=False)
    assert not runner.robustness_teacher(fit,[uncertain],1.,.25)['robustness_mask'].any()
    assert not runner.robustness_teacher(fit,[],1.,.25)['robustness_mask'].any()
    stable=runner.robustness_teacher(fit,[dict(fit,driver=fit['driver']+.1)],1.,.25)
    assert stable['robustness_mask'].all()
    varying=runner.robustness_teacher(fit,[dict(fit,driver=fit['driver']+.5)],1.,.25)
    assert not varying['robustness_mask'].any()
    assert varying['centered_robustness_mask'].all()


def test_summary_retains_failed_denominator_and_pairs_only_common_success():
    import pandas as pd
    from experiments.scripts import evaluate_shared_driver_reconstruction as runner
    frame=pd.DataFrame([dict(id=f'w{i}',subject=f's{i}',dataset='fixture',mode='full',arm=arm,
        converged=not (i==2 and arm=='M-prior'),total_nrmse=value)
        for i in range(3) for arm,value in [('M0',1.),('M-prior',.5)]])
    table=runner.robustness_group_summary(frame,['arm'],['total_nrmse']).set_index('arm')
    assert table.loc['M-prior','denominator']==3
    assert table.loc['M-prior','converged']==2
    pairs=runner.robustness_paired(frame,['dataset','mode'])
    assert pairs[0]['denominator']==3 and pairs[0]['common_success']==2
    assert pairs[0]['mean_gain']==pytest.approx(.5)
