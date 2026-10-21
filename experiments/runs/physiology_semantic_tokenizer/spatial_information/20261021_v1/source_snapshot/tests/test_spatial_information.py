"""Synthetic contracts for the independent spatial-information experiment."""
from copy import deepcopy
import json
import numpy as np
import pytest
from src.inference.spatial_information import (
    panel_mask, legacy_objective_weights, weak_direction_lift, generic_equal_rank_readout)
from src.inference.source_observability import weighted_jacobian
from experiments.scripts import evaluate_spatial_information as runner


def test_channel_prefix_masks_keep_original_columns_and_hidden_pair():
    small=panel_mask(8,60,4,observed_eeg=30,observed_hb=2)
    full=panel_mask(8,60,4,observed_eeg=60,observed_hb=4)
    assert np.all(small <= full)
    assert not full[:,60:62].any()
    assert small[:,62:64].all() and not small[:,64:].any()
    assert not small[:,30:60].any()
    with pytest.raises(ValueError):panel_mask(8,60,4,observed_eeg=61,observed_hb=4)


def test_added_channel_information_uses_old_directions_and_no_renormalization():
    old=np.diag([3.,1.,.001]);new=np.vstack((old,[0.,0.,2.]))
    result=weak_direction_lift(old,new,directions=2)
    np.testing.assert_allclose(result['old_direction_energy'],[1.,1e-6])
    np.testing.assert_allclose(result['full_direction_energy'],[1.,4.000001])
    np.testing.assert_allclose(result['energy_gain_ratio'],[1.,4000001.])
    with pytest.raises(ValueError,match='add rows'):weak_direction_lift(new,old,directions=2)


def test_legacy_modality_mean_can_remove_observation_count_gain():
    jac=np.ones((3,6,1));sd=np.ones(6)
    small=panel_mask(3,4,1,observed_eeg=2,observed_hb=1)
    full=panel_mask(3,4,1,observed_eeg=4,observed_hb=1)
    simple=[weighted_jacobian(jac,sd,m) for m in (small,full)]
    objective=[weighted_jacobian(jac,sd,m,legacy_objective_weights(m,4)) for m in (small,full)]
    assert (simple[1].T@simple[1]).item()==2*(simple[0].T@simple[0]).item()
    np.testing.assert_allclose(objective[0].T@objective[0],objective[1].T@objective[1])


def test_generic_equal_rank_preserves_loading_spectra_and_hb_convexity():
    rng=np.random.default_rng(3);eeg=rng.normal(size=(10,2));a=np.array([.1,.3,.7,.9]);hb=np.column_stack((a,1-a))
    original=eeg.copy(),hb.copy();control=generic_equal_rank_readout(eeg,hb,seed=4)
    np.testing.assert_allclose(np.linalg.svd(control['eeg_loadings'],compute_uv=False),np.linalg.svd(eeg,compute_uv=False),rtol=1e-13)
    np.testing.assert_allclose(np.linalg.svd(control['hb_loadings'],compute_uv=False),np.linalg.svd(hb,compute_uv=False),rtol=1e-13)
    np.testing.assert_allclose(control['hb_loadings'].sum(axis=1),1)
    assert (control['hb_loadings']>=0).all()
    assert not np.allclose(control['eeg_loadings'],eeg)
    np.testing.assert_array_equal(eeg,original[0]);np.testing.assert_array_equal(hb,original[1])


def test_paired_statistics_keep_failed_planned_units_and_do_not_count_windows():
    cfg=runner.cfg_read();candidate=[dict(subject='A',record_id='a',status='completed',success=True,nmse=0.),
        dict(subject='A',record_id='b',status='failed_numerical',success=False),
        dict(subject='B',record_id='c',status='completed',success=True,nmse=1.)]
    reference=[dict(subject=r['subject'],record_id=r['record_id'],status='completed',success=True,nmse=1.) for r in candidate]
    out=runner.paired(candidate,reference,cfg,unit='subject',key_fields=['record_id'])
    assert out['available_pairs']==3 and out['independent_units']==2
    assert out['candidate']['failed']==1
    assert out['candidate']['mean_nrmse_with_failure_penalty']==pytest.approx((np.sqrt(50)+1)/2)
    assert out['candidate']['success_subset_mean_nrmse']==.5
    assert out['mean_nrmse_improvement']<0


def test_check_stage_never_reads_retained_measured_artifacts(monkeypatch,capsys):
    monkeypatch.setattr(runner,'source_context',lambda *_:pytest.fail('check must not read measured data'))
    assert runner.main(['--stage','check'])==0
    assert json.loads(capsys.readouterr().out)['measured_reads'] is False


def short_synthetic():
    cfg=deepcopy(runner.cfg_read());cfg['synthetic'].update(steps=24,temporal_modes=4,reference_steps=4,score_steps=[4,24],training_seeds=[100,101])
    source={'fixed':dict(tau=2.,kappa=.64,gamma=.32,alpha=.32,E0=.32,P0=1.,Q0=.35,neurovascular_gain=1.,eeg_loading=1.)}
    return cfg,source


def test_synthetic_controls_share_signal_and_training_scales_exclude_eval():
    cfg,source=short_synthetic();coordinate=runner.synthetic_coordinate(cfg,source)
    assert not coordinate['evaluation_used']
    assert set(coordinate['training_seeds']).isdisjoint(cfg['synthetic']['seeds'])
    one=runner.synthetic_case(cfg,source,'two_sources',0);two=runner.synthetic_case(cfg,source,'two_sources',0)
    np.testing.assert_array_equal(one['eeg'],two['eeg']);np.testing.assert_array_equal(one['hb'],two['hb'])
    task=dict(scenario='two_sources',seed=0,control='sham',panel=dict(hb_pairs=4,eeg_channels=12),model='free_K2')
    result=runner.synthetic_job(cfg,task,coordinate,source)
    assert result['hidden_input_all_nonfinite']
    assert result['fixed_eeg_weight']==1
    assert result['status'] in ('completed','failed_numerical','failed_integration','failed_domain','failed_physical')
    # A failure is terminal evidence, never silently dropped by the runner.
    if result['success']:
        assert result['nmse']>=0 and np.isfinite(result['driver_relative_RMSE'])


def test_information_keeps_common_nuisance_dimension_across_panels():
    cfg,source=short_synthetic();case=runner.synthetic_case(cfg,source,'two_sources',0)
    coordinate=runner.synthetic_coordinate(cfg,source)
    truth=case['truth'];result,_=runner.linear_information(cfg,truth['coefficients'],truth['initial_coordinates'],
        runner.parameters(source),cfg['synthetic']['dt_s'],np.asarray(coordinate['sd']),case['args'])
    assert result['fixed_unknowns']==18 and result['expanded_physiology_unknowns']==20
    for panel in result['panels']:
        for name in cfg['information']['weighting']:
            assert panel[name]['fixed_physiology']['unknowns']==18
            assert len(panel[name]['physiology_profile']['nuisance_columns'])==18
    for lift in result['weak_lifts'].values():
        assert np.all(lift['added_direction_energy'] >= -1e-8)
