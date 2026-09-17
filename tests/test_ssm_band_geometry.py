"""Small non-measured checks for the band/geometry structural diagnostic."""
import copy
import numpy as np
import pytest
from experiments import evaluate_ssm_overnight_diagnostics as run
from src.inference.observation_baselines import (
    eeg_band_power, gaussian_driver_statistic, signed_loading_fit,
    correlated_feature_noise, geometry_neighbours,
)

CONFIG='experiments/configs/physiology_semantic_tokenizer/ssm_band_geometry_v1.yaml'


def test_correlated_sufficient_statistic_preserves_objective_differences_and_gradient():
    rng=np.random.default_rng(4);x=rng.normal(size=(13,18));ell=rng.normal(size=18)
    a=rng.normal(size=(18,18));cov=a@a.T+np.eye(18)*.3
    out=gaussian_driver_statistic(x,ell,cov)
    r1=rng.normal(size=13);r2=rng.normal(size=13)
    def full(r):
        residual=x-r[:,None]*ell
        return np.sum(residual*np.linalg.solve(cov,residual.T).T)/2
    def reduced(r):return np.sum((out['values']-r)**2)/(2*out['variance'])
    assert full(r1)-full(r2)==pytest.approx(reduced(r1)-reduced(r2),abs=1e-10)
    gradient=-((x-r1[:,None]*ell)@np.linalg.solve(cov,ell))
    np.testing.assert_allclose(gradient,(r1-out['values'])/out['variance'],atol=1e-11)
    # Correlated copies do not count as eighteen independent measurements.
    cov=.99*np.ones((18,18))+.01*np.eye(18)
    out=gaussian_driver_statistic(np.zeros((3,18)),np.ones(18),cov)
    assert out['variance']>.99


def test_power_features_preserve_band_axis_and_physical_merge():
    t=np.arange(6000)/200
    x=np.column_stack((np.sin(2*np.pi*10*t),2*np.sin(2*np.pi*20*t)))
    p=eeg_band_power(x)
    assert p.shape==(120,2,3) and p.dtype==np.float64
    assert np.median(p[8:-8,0,0])>.4
    assert np.median(p[8:-8,1,1])>1.6
    assert np.median(p[8:-8,0,1])<.01
    assert not np.allclose(np.log(p.sum(-1)),np.log(np.maximum(p,1e-12)).sum(-1))


def test_signed_geometry_penalty_and_permutation():
    rng=np.random.default_rng(8);r=rng.normal(size=(18,120));w=np.linspace(.2,1,6)
    truth=(w[:,None]*np.array([1.,-.8,.3])).ravel();x=r[:,:,None]*truth+rng.normal(size=(18,120,18))*.2
    free=signed_loading_fit(x,r,bands=3)
    geometric=signed_loading_fit(x,r,w,bands=3,geometry_penalty=1.)
    wrong=signed_loading_fit(x,r,w[::-1],bands=3,geometry_penalty=1.)
    assert np.mean((geometric-truth)**2)<np.mean((wrong-truth)**2)
    assert geometric.reshape(6,3)[:,1].max()<0
    assert free.shape==(18,)


def test_geometry_requires_compatible_coordinates_without_fallback():
    target=dict(modality='fnirs',channel_name='H',x=0.,y=0.,z=0.,coordinate_system='mnt',coordinate_units='u')
    rows=[target]+[dict(target,modality='eeg',channel_name=f'E{i}',x=float(i+1)) for i in range(8)]
    out=geometry_neighbours(rows,[f'E{i}' for i in range(8)],'H')
    assert out['indices']==list(range(6))
    assert np.linalg.norm(out['weights'])==pytest.approx(1.)
    for row in rows[1:]:row['coordinate_system']='other'
    with pytest.raises(ValueError,match='no compatible'):geometry_neighbours(rows,['E0','E1'],'H')


def test_feature_loading_and_noise_do_not_fit_evaluation_values():
    cfg,*_=run.load_config(CONFIG);rng=np.random.default_rng(9)
    features=rng.normal(size=(24,120,18));driver=rng.normal(size=(24,120))
    geom=dict(weights=np.arange(1,7),indices=list(range(6)))
    first=run.structure_fit_features(features,driver,list(range(18)),geom,'geometry',cfg,.2,.1)
    features[18:]*=100;driver[18:]*=-30
    second=run.structure_fit_features(features,driver,list(range(18)),geom,'geometry',cfg,.2,.1)
    for key in ('center','scale','loading','covariance','sufficient_weights'):
        np.testing.assert_array_equal(first[2][key],second[2][key])
    np.testing.assert_array_equal(first[0][:18],second[0][:18])
    cov=correlated_feature_noise(features[:18],floor=.1)['covariance']
    assert np.linalg.eigvalsh(cov).min()>0


def test_frozen_parser_and_complete_task_denominators(tmp_path):
    cfg,*_=run.load_config(CONFIG)
    inventory={s:run.v3_inventory(cfg,s) for s in cfg['subjects']}
    tasks=run.structure_tasks(cfg,inventory)
    assert len({t['id'] for t in tasks})==len(tasks)
    run.persist_tasks(tmp_path,tasks)
    assert run.read_tasks(tmp_path)==tasks
    for candidate in cfg['structure']['candidates']:
        measured=[t for t in tasks if t['kind']=='structure_fit' and t['family']=='M' and t['candidate_name']==candidate]
        assert len(measured)==72
        assert all(len(t['modes'])==13 for t in measured)
        assert all(t['prerequisites'] for t in measured)
    wrong=copy.deepcopy(cfg['structure']);wrong['bands'][0]=[7.,13.]
    with pytest.raises(ValueError):run.structure_validate(wrong)


def test_feature_mask_hides_target_before_time_mixing():
    cfg,*_=run.load_config(CONFIG);rng=np.random.default_rng(42)
    arrays=dict(feature_eeg=rng.normal(size=(24,120)),feature_fnirs=rng.normal(size=(24,300,2)),
                target=np.zeros((24,120,3)),normalizer=np.ones(3))
    info=dict(train=list(range(18)),feature_noise=dict(eeg_sd=.1,fnirs_mixing=np.eye(2)*.1))
    ids=[dict(session='s',training_ordinal=i) for i in range(24)]
    before=run.structure_compile_view(cfg,arrays,info,ids,20,'center_fNIRS')
    arrays['feature_fnirs'][20,130:170]=1e12
    after=run.structure_compile_view(cfg,arrays,info,ids,20,'center_fNIRS')
    np.testing.assert_array_equal(before['input'],after['input'])
    np.testing.assert_array_equal(before['noise_factor'],after['noise_factor'])
    # Whole missing fNIRS has no target baseline or target statistics in inputs.
    whole=run.structure_compile_view(cfg,arrays,info,ids,20,'EEG_only')
    arrays['feature_fnirs'][20]=1e20
    changed=run.structure_compile_view(cfg,arrays,info,ids,20,'EEG_only')
    np.testing.assert_array_equal(whole['input'],changed['input'])


def test_report_primary_risk_requires_the_same_complete_identity_set():
    from experiments.scripts.render_ssm_band_geometry_report import risk
    rows=[dict(sample_id=f't{i}',subject='s',session='one',mode=mode,status='completed',mse=float(i+1))
          for i in range(3) for mode in ('center_fNIRS','EEG_only')]
    assert risk(rows,3)['RF']==2.
    # Same marginal counts but different identities do not form a complete RF.
    rows[-1]['sample_id']='not_the_same_trial'
    result=risk(rows,3)
    assert result['valid_center']==result['valid_whole']==3
    assert result['paired_identities']==2 and result['RF'] is None


def test_whole_modality_training_template_propagates_mean_noise():
    cfg,*_=run.load_config(CONFIG);rng=np.random.default_rng(61)
    arrays=dict(feature_eeg=rng.normal(size=(24,120)),feature_fnirs=rng.normal(size=(24,300,2)),
                target=np.zeros((24,120,3)),normalizer=np.ones(3))
    ids=[dict(session='s',training_ordinal=i) for i in range(24)]
    info=dict(train=list(range(18)),feature_noise=dict(eeg_sd=.2,fnirs_mixing=np.eye(2)*.1))
    single=run.structure_compile_view(cfg,arrays,info,ids,20,'EEG_only')
    template=run.structure_compile_view(cfg,arrays,info,ids,20,'EEG_only_template')
    np.testing.assert_allclose(template['noise_factor'],single['noise_factor']/np.sqrt(18))
    expected=run.v3_native_operators()['eeg']@arrays['feature_eeg'][:18].mean(0)
    np.testing.assert_allclose(template['input'][:,0],expected)
    assert np.all(template['target_noise_factor'][0::3]==0)


def test_report_uses_panel_uncertainty_and_equal_session_weight():
    from experiments.scripts.render_ssm_band_geometry_report import paired_interval, hierarchical
    assert paired_interval([.01]*4)['upper95']==pytest.approx(.01)
    assert paired_interval([.01]*3)['upper95'] is None
    rows=[dict(subject='s',session='one',score=0.)]*9+[dict(subject='s',session='two',score=10.)]
    assert hierarchical(rows,'score')==5.
