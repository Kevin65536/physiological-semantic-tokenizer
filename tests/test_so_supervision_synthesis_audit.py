"""Analytic and independent numerical checks; no local data dependencies."""
import numpy as np
import pytest
from experiments.scripts.audit_so_supervision_synthesis import reference_disagreement,staged_synthesis,validate_source
from src.data.so_supervision_cache import bind_reference
from src.data.dual_component_tokenizer import voltage_power_cap_target,voltage_power_cap_reference


def test_pairwise_gram_matches_explicit_single_position_completion():
    rng=np.random.default_rng(8);a=rng.normal(size=(4,15));u=rng.normal(size=(15,15));cov=u@u.T+np.eye(15)
    mu=rng.normal(size=15);p=rng.normal(size=(70,4));neighbors=[[0,1],[1,2],[0,2],[0,1]]
    values,_,pairs=reference_disagreement(cov,a,p,mu,neighbors)
    ell=[]
    for i in range(4):
        model=bind_reference(dict(covariance=cov),a[i:i+1],70)
        ell.append(mu+(p[:,i:i+1]-a[i]@mu)@model['lift'].T)
    expected=[np.sqrt(np.mean((ell[i][:,c*5:c*5+5]-ell[j][:,c*5:c*5+5])**2)) for i,j,c in pairs]
    np.testing.assert_allclose(values,expected,rtol=1e-12,atol=1e-12)


def test_duplicate_constraint_null_and_inconsistent_same_input():
    a=np.ones((2,10));p=np.zeros((60,2));cov=np.eye(10);mu=np.zeros(10)
    v,identical,_=reference_disagreement(cov,a,p,mu,[[0,1],[0,1]])
    assert not v.any() and not identical.any()
    p[:,1]=2
    v,identical,_=reference_disagreement(cov,a,p,mu,[[0,1],[0,1]])
    np.testing.assert_allclose(v,.2);np.testing.assert_allclose(identical,2/np.sqrt(10))


def test_carrier_separated_synthesis_matches_production_and_closure():
    rng=np.random.default_rng(3);c=3;x=rng.normal(size=(1200,c))*2
    stats=dict(mean=np.zeros(c*5),covariance=np.eye(c*5),baseline_log_power=np.zeros(c*5),correlation=np.tile(np.eye(c),(6,1,1)),complement_power=np.ones(c),sample_rate_hz=200)
    model=bind_reference(stats,rng.normal(size=(3,c*5)),60);p=rng.normal(size=(60,3))*.1
    actual=staged_synthesis(x,model,p);expected=voltage_power_cap_target(x,model,p,frame_batch_size=8)
    np.testing.assert_allclose(actual['s'],expected['s'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(actual['final'],expected['final_reanalysis_ratio'],rtol=1e-10,atol=1e-10)
    assert actual['local'].max()<1+1e-8
    assert actual['final_diagonal'].shape==(61,6,c)


def test_uncompleted_source_rejected_before_array_access(tmp_path):
    import json
    (tmp_path/'cache_manifest.json').write_text(json.dumps(dict(schema='physiology_so_supervision_v2',status='running')))
    (tmp_path/'verification.json').write_text(json.dumps(dict(status='passed',issues=[])))
    with pytest.raises(AssertionError):validate_source(tmp_path)


def test_constant_proxy_intervention_preserves_frame_mean_log_power():
    rng=np.random.default_rng(19);c=3;steps=60
    stats=dict(mean=rng.normal(size=c*5),covariance=np.eye(c*5),baseline_log_power=rng.normal(size=c*5),correlation=np.tile(np.eye(c),(6,1,1)),complement_power=np.ones(c),sample_rate_hz=200)
    model=bind_reference(stats,rng.normal(size=(3,c*5)),steps);p=rng.normal(size=(steps,3))
    flat=np.broadcast_to((p.sum(0)+p[-1])/(steps+1),p.shape).copy();times=np.arange(steps+1)*.1
    before=voltage_power_cap_reference(model,p,times);after=voltage_power_cap_reference(model,flat,times)
    np.testing.assert_allclose(np.log(before).mean(0),np.log(after).mean(0),atol=1e-12,rtol=1e-12)
    again=np.broadcast_to((flat.sum(0)+flat[-1])/(steps+1),flat.shape)
    np.testing.assert_allclose(again,flat,rtol=1e-12,atol=1e-12)


def test_hann_selects_central_energy_without_pointwise_amplification():
    from experiments.scripts.audit_so_supervision_synthesis import normalized_overlap
    blocks=np.tile(np.array([0.,np.sqrt(2),np.sqrt(2),0.])[None,:,None],(8,1,1))
    np.testing.assert_allclose(np.mean(blocks**2,axis=1),1)
    hann,weights=normalized_overlap(blocks,2,np.hanning(5)[:-1]);box,_=normalized_overlap(blocks,2,np.ones(4))
    np.testing.assert_allclose(np.mean(hann[4:-4]**2),1.25)
    np.testing.assert_allclose(np.mean(box[4:-4]**2),.5)
    assert hann.max()<=blocks.max()
    np.testing.assert_allclose(weights[4:-4],1)


def test_consistent_overlapping_estimates_reconstruct_without_gain():
    from experiments.scripts.audit_so_supervision_synthesis import normalized_overlap
    x=np.sin(np.arange(20)*.37)[:,None]
    blocks=np.stack([x[k:k+4] for k in range(0,17,2)])
    y,w=normalized_overlap(blocks,2,np.hanning(5)[:-1])
    np.testing.assert_allclose(y[w>0],x[w>0],rtol=1e-14,atol=1e-14)


def test_band_spectral_identity_and_known_sinusoid_partition():
    from experiments.scripts.audit_so_supervision_synthesis import spectral_metrics
    t=np.arange(18000)/200;s=np.sin(2*np.pi*10*t)[:,None];o=2*np.sin(2*np.pi*20*t)[:,None];x=s+o
    signals=dict(X=x,S_i=s,S_j=s,S_joint=s,O_i=o,O_j=o,O_joint=o)
    r=spectral_metrics(signals,200)
    assert float(r['closure_error'])<1e-12
    np.testing.assert_allclose(r['retention'][2,3],1,atol=1e-12)
    np.testing.assert_allclose(r['retention'][3,3],0,atol=1e-12)
    np.testing.assert_allclose(r['shape'][:,0].sum(),1,atol=1e-12)


def test_binned_jensen_mixture_matches_production_reconstruction():
    from experiments.scripts.audit_so_supervision_synthesis import overlap_analysis
    rng=np.random.default_rng(22);c=2;x=rng.normal(size=(18000,c))*3
    stats=dict(mean=np.zeros(c*5),covariance=np.eye(c*5),baseline_log_power=np.zeros(c*5),correlation=np.tile(np.eye(c),(6,1,1)),complement_power=np.ones(c),sample_rate_hz=200)
    model=bind_reference(stats,rng.normal(size=(3,c*5)),900);proxy=np.zeros((900,3))
    s,r=overlap_analysis(x,model,proxy);expected=voltage_power_cap_target(x,model,proxy,audit_final=False,frame_batch_size=8)
    np.testing.assert_allclose(s,expected['s'],rtol=1e-12,atol=1e-12)
    assert r['jensen_min_eigenvalue'].min()>-1e-9
    assert np.all(r['hann_ratio']<=r['mixture_ratio']+1e-9)
    assert r['disagreement_energy_fraction'].min()>-1e-9
