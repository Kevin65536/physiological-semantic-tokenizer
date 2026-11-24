"""The diagnostic keeps the held patch out and cannot promote a self control."""
import importlib
from pathlib import Path
import numpy as np


def owner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments/scripts'))
    return importlib.import_module('audit_integer_neighbor_voltage')


def test_projection_preserves_45_and_excludes_50(monkeypatch):
    m=owner(monkeypatch);t=np.arange(400)/200
    wave=np.sin(2*np.pi*45*t);line=np.sin(2*np.pi*50*t)
    x=np.broadcast_to((wave+line)[None,:,None],(15,400,1))
    np.testing.assert_allclose(m.project(x),np.broadcast_to(wave[None,:,None],x.shape),atol=1e-12)


def test_other_patch_fit_cannot_see_held_target_or_partial_patch(monkeypatch):
    m=owner(monkeypatch);rng=np.random.default_rng(17);x=rng.normal(size=(15,400,2));y=x[...,0]-2*x[...,1]
    complete=np.ones(15,bool);complete[3]=False
    a,fit=m.predict_patch(x,y,complete,7)
    changed=y.copy();changed[7]+=1000;changed[3]-=1000
    b,again=m.predict_patch(x,changed,complete,7)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(fit['coefficients'],again['coefficients']);assert fit['fit_patches']==13


def test_self_control_cannot_bypass_integer_neighbor_failure(monkeypatch):
    m=owner(monkeypatch)
    cfg=dict(admission=dict(RMS_gain_interval=[.75,1.25],all_17_error_ratio_to_original_max=1.10,
        integer_neighbor_error_over_target_energy_max=.5,integer_neighbor_error_ratio_to_original_max=.75,
        RMS_peak_delay_absolute_max_s=.25,required_candidate_max_datasets=2))
    records=[dict(index=j,dataset=str(j),selection=['candidate_max'],arms={'original':dict(error_over_target_energy=1.),
        'integer_neighbors':dict(error_over_target_energy=1.1,RMS_gain=.4,RMS_peak_delay_s=0.),
        'integer_all':dict(error_over_target_energy=.01,RMS_gain=1.,RMS_peak_delay_s=0.)}) for j in range(17)]
    assert not m.admission(records,cfg)['admitted']
