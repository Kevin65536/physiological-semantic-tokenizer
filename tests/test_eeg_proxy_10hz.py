"""Current EEG clock, physical-time fitting and native preparation boundaries."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from experiments.scripts import evaluate_shared_driver_reconstruction as runner
from src.data.ssm_prepared import EEG_PROXY_INPUT_SCHEMA, require_10hz_eeg_proxy_array
from src.inference.observation_baselines import eeg_band_power, disjoint_native_bin_operator
from src.inference.shared_driver_reconstruction import (
    _nonlinear_trial_objective, _volume_tied_initial_embedding,
    fit_nonlinear_shared_parameters, nonlinear_driver_forward,
)
from src.inference.t3a_balloon_robust_ssm import BalloonParameters


def test_default_power_recomputes_100ms_bins_from_native_voltage():
    t=np.arange(6000)/200.
    x=((1+.6*np.sin(2*np.pi*3.1*t))*np.sin(2*np.pi*10*t))[:,None]
    power=eeg_band_power(x,bands=[(8,13)])
    filtered=sosfiltfilt(butter(4,[8,13],btype='bandpass',fs=200,output='sos'),x,axis=0)
    expected=(filtered**2).reshape(300,20,1).mean(1)
    np.testing.assert_allclose(power[:,:,0],expected,rtol=1e-14)
    old=eeg_band_power(x,bands=[(8,13)],target_rate=4.)[:,0,0]
    upcast=np.interp(np.arange(300)/10.,np.arange(120)/4.,old)
    assert np.max(abs(power[:,0,0]-upcast))>.01
    with pytest.raises(ValueError,match='integer clock'):
        eeg_band_power(x[:-1])


def test_current_contract_and_synthetic_use_same_10hz_clock(tmp_path):
    import yaml
    cfg=runner.tak_config(runner.DEFAULT)
    target,operator,sd,truth=runner.tak_synthetic(cfg,0,'tak_nonrest')
    assert target.shape==(300,3) and operator.shape==(900,900)
    np.testing.assert_allclose(target[:50].mean(0),0,atol=1e-15)
    np.testing.assert_array_equal(truth['states'][:,3],truth['states'][:,4])
    for section,key,value in [('tensor','dt_s',.25),('tensor','reference_steps',20),
                              ('tensor','eeg_proxy_rate_hz',4.),('solver','observation_reference_dt_s',.1),
                              ('data','raw_reads',True)]:
        bad=deepcopy(cfg);bad[section][key]=value
        path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError):runner.tak_config(path)


def test_10hz_crop_reference_masks_and_scoring_remain_in_seconds():
    time=np.arange(900)/10.
    ops=disjoint_native_bin_operator(time)
    np.testing.assert_array_equal(ops['native_counts'],np.ones(900))
    y=np.column_stack((np.sin(time),np.cos(time/7),np.sin(time/11)))
    target,operator=runner.volume_drive_crop(y,dict(eeg=ops['eeg'],hb=ops['model']),30,dt=.1)
    expected=y[300:600]-y[300:350].mean(0)
    np.testing.assert_allclose(target,expected,atol=2e-15)
    np.testing.assert_allclose((operator@y[300:600].ravel()).reshape(300,3),expected,atol=2e-14)
    np.testing.assert_allclose(target[:50].mean(0),0,atol=1e-15)
    for n in (300,900):
        mask=runner.volume_drive_mask(n,'joint_blocked',dt=.1)
        assert not mask[-1].any() and mask[:-1,0].all()
        hidden=np.flatnonzero(~mask[:-1,1])
        times=hidden/10.%30
        assert np.all(((times>=10)&(times<15))|((times>=20)&(times<25)))
        assert len(hidden)==100*(n//300)
    wrong=target.copy();wrong[100:150,1:]+=2.;wrong[200:250,1:]+=2.
    wrong[-1]=1e15
    score=runner.volume_drive_metrics(target,wrong,np.ones(3),'joint_blocked',dt=.1)
    assert score['score_hb_nmse']==pytest.approx(4.)
    assert score['eeg_nmse']==0.


def test_current_prepared_boundary_rejects_old_or_relabelled_clock():
    arrays=dict(target=np.zeros((300,3)),eeg_proxy_input=np.array(EEG_PROXY_INPUT_SCHEMA),
        eeg_proxy_rate_hz=np.array(10.),time_s=np.arange(300)/10.)
    require_10hz_eeg_proxy_array(arrays,steps=300)
    for key,value in [('eeg_proxy_input',''),('eeg_proxy_rate_hz',4.),
                      ('target',np.zeros((120,3))),('time_s',np.arange(300)/4.)]:
        bad=dict(arrays);bad[key]=value
        with pytest.raises(ValueError,match='regenerate'):
            require_10hz_eeg_proxy_array(bad,steps=300)


@pytest.mark.parametrize('schema',['shared_driver_tak_v1','shared_driver_tak_10hz_v2'])
def test_report_dispatch_recognizes_retained_and_current_clocks(schema,monkeypatch,tmp_path):
    import sys
    from experiments.scripts import render_shared_driver_reconstruction as renderer
    (tmp_path/'resolved_config.json').write_text(json.dumps(dict(schema=schema)))
    seen=[]
    monkeypatch.setattr(renderer,'render_tak',lambda run,out:seen.append((run,out)))
    monkeypatch.setattr(sys,'argv',['renderer','--run',str(tmp_path),'--output',str(tmp_path/'report')])
    renderer.main()
    assert seen==[(tmp_path,tmp_path/'report')]


def test_waveform_peak_difference_uses_declared_seconds():
    time=np.arange(300)/10.
    target=np.tile(np.exp(-(time-12.)**2)[:,None],(1,3))
    prediction=np.tile(np.exp(-(time-13.)**2)[:,None],(1,3))
    scores=runner.filter_retest_waveform_metrics(target,prediction,np.ones(3),dt=.1)
    assert all(s['peak_time_difference_s']==pytest.approx(1.) for s in scores)


def test_new_preparation_uses_native_eeg_and_training_only_coordinates(monkeypatch,tmp_path):
    from src.data import clean_physiology_cache, unified_physiology, ssm_prepared
    cfg=runner.tak_config(runner.DEFAULT);cfg['parents']=['parent']
    parent=tmp_path/'parent';parent.mkdir();out=tmp_path/'run';out.mkdir()
    anchor=dict(eeg_channels=list('ABCDEF'),hb_pair=0,hb_channel='CH1')
    def ref(name,partition):
        return dict(id=name,subject=name,partition=partition,coordinate='local',
            parent='parent',dataset='refed',join_key=name,anchor=anchor,eeg_start_s=1.,hb_start_s=0.)
    train,evaluation=ref('train','development'),ref('eval','evaluation')
    (parent/'plan.json').write_text(json.dumps(dict(records=[train,evaluation])))
    (out/'software_checks.json').write_text('{"status":"passed"}')
    (out/'synthetic_summary.json').write_text('{"status":"completed"}')
    plan=dict(records=[evaluation]);monkeypatch.setattr(runner,'tak_plan',lambda *a:plan)
    records=[SimpleNamespace(join_key=name) for name in ('train','eval')]
    monkeypatch.setattr(clean_physiology_cache,'CleanPhysiologyCacheIndex',lambda *a:SimpleNamespace(records=records))
    voltages={name:np.random.default_rng(i).normal(size=(18200,6)) for i,name in enumerate(('train','eval'))}
    native_reads=[];cache_reads=[]
    def native(root,record):
        native_reads.append(record.join_key)
        return SimpleNamespace(values=voltages[record.join_key],channel_names=list('ABCDEF'),
            sample_rate_hz=200.,native_unit='uV',source_path=tmp_path/(record.join_key+'.mat'))
    time=np.arange(900)/10.
    def cached(root,record):
        cache_reads.append(record.join_key)
        hb=np.column_stack((np.sin(time/7),np.cos(time/9)))
        if record.join_key=='eval':hb*=1000
        return dict(values=hb[:,None,:],time_s=time,channel_names=['CH1_HbO','CH1_HbR'],
            provenance=dict(input_schema='unified_cached_fnirs_v1'))
    monkeypatch.setattr(unified_physiology,'load_native_eeg_record',native)
    monkeypatch.setattr(unified_physiology,'load_cached_fnirs_record',cached)
    def forbidden(*a,**kw):raise AssertionError('raw Hb must not be read')
    monkeypatch.setattr(unified_physiology,'load_native_fnirs_record',forbidden)
    fit_coordinate=ssm_prepared.fit_feature_coordinate;calibration=[]
    def capture(samples):
        calibration.extend(samples)
        return fit_coordinate(samples)
    monkeypatch.setattr(ssm_prepared,'fit_feature_coordinate',capture)
    result=runner.tak_prepare(cfg,tmp_path,out,plan)
    assert result['prepared_records']==1 and not result['failures']
    assert native_reads==cache_reads==['train','eval'] and result['raw_fnirs_reads']==0
    assert result['coordinates']['local']['training_subjects']==['train']
    assert len(calibration)==1 and np.max(abs(calibration[0][1]))<3
    power=eeg_band_power(voltages['train'][200:],bands=cfg['tensor']['eeg_bands_hz'])
    features=np.log(power).reshape(900,30);features-=features[:50].mean(0)
    np.testing.assert_allclose(calibration[0][0],features,atol=2e-14)
    # No parent prepared arrays exist: successfully preparing cannot reuse them.
    for seconds in (30,90):
        with np.load(out/'prepared'/f'eval__{seconds}.npz',allow_pickle=False) as a:
            require_10hz_eeg_proxy_array(a,steps=seconds*10)
            assert a['operator'].shape==(seconds*30,seconds*30)
            np.testing.assert_allclose(a['target'][:50].mean(0),0,atol=2e-12)


def test_data_weight_scales_observation_tangents_but_preserves_priors():
    n=28;dt=.1;p=BalloonParameters();embedding=_volume_tied_initial_embedding(n)
    x=np.r_[.02*np.sin(np.arange(n)*.21),.01,np.log([1.02,.98,1.01])]
    y=np.zeros((n,3));mask=np.ones_like(y,bool);mask[12:16,1:]=False
    reg=np.zeros((n-2,n+5));reg[:,:n]=.1*np.diff(np.eye(n),n=2,axis=0)/dt**1.5
    opts=dict(operator=None,y=y,mask=mask,scale=np.full_like(y,.02),regularizer=reg,
        initial_weight=100.,substeps=4,derivative=True,parameter_name=['tau','kappa'],
        initial_embedding=embedding,flow_prior_weight=1.,flow_prior_log_sd=np.log(2),
        numerical_backend='numba',hemodynamics='tak',viscoelastic_s=2.,
        initial_coordinate_weights=np.array([1,1,1,0,1]))
    full=_nonlinear_trial_objective(x,p,dt,**opts)
    weighted=_nonlinear_trial_objective(x,p,dt,**opts,data_weight=.4)
    count=mask.sum()
    for key in ('residual','jacobian','parameter_jacobian'):
        np.testing.assert_allclose(weighted[key][:count],np.sqrt(.4)*full[key][:count])
        np.testing.assert_array_equal(weighted[key][count:],full[key][count:])
    direction=np.random.default_rng(1).normal(size=x.size)*.03;eps=1e-5
    def residual(v,params=p):
        return _nonlinear_trial_objective(v,params,dt,**dict(opts,derivative=False),data_weight=.4)['residual']
    numeric=(residual(x+eps*direction)-residual(x-eps*direction))/(2*eps)
    np.testing.assert_allclose(weighted['jacobian']@direction,numeric,atol=1e-8,rtol=2e-6)
    numeric_tau=(residual(x,replace(p,free=replace(p.free,tau=p.free.tau+eps)))-
                 residual(x,replace(p,free=replace(p.free,tau=p.free.tau-eps))))/(2*eps)
    np.testing.assert_allclose(weighted['parameter_jacobian'][:,0],numeric_tau,atol=1e-8,rtol=2e-6)


def test_shared_fit_weighted_objective_and_per_second_normalization():
    p=BalloonParameters();n=24;dt=.1
    driver=.02*np.sin(np.arange(n)*.23)
    y=nonlinear_driver_forward(driver,[0,1,1,1],p,dt,numerical_backend='numba')['canonical_prediction'][None]
    sd=np.array([.02,.03,.02]);visible=np.ones_like(y,bool);visible[:,8:12,1:]=False
    fit=fit_nonlinear_shared_parameters(y,p,dt,parameter_names=['tau'],parameter_bounds=[[.5,8]],
        parameter_prior_log_mean=[np.log(2.)],parameter_prior_precision=[16.],data_weight=.4,
        visible=visible,sd=sd,penalty=.01,initial_penalty=100.,flow_prior_weight=1.)
    assert fit['converged'] and fit['data_weight']==.4
    actual=.4*np.sum(((fit['prediction']-y)/sd)[visible]**2)
    assert fit['weighted_data_sse']==pytest.approx(actual)
    assert fit['objective_components']['data']==pytest.approx(actual)
    assert fit['objective']==pytest.approx(sum(fit['objective_components'].values()))
    # Same 30 s constant standardized residual, at either clock, has equal cost.
    costs=[]
    for step in (.25,.1):
        samples=round(30/step);target=np.ones((1,samples,3))
        fixed=fit_nonlinear_shared_parameters(target,p,step,parameter_names=[],parameter_bounds=[],
            sd=np.ones(3),max_iterations=0,data_weight=step/.25)
        costs.append(fixed['weighted_data_sse'])
    np.testing.assert_allclose(costs,[360.,360.])
