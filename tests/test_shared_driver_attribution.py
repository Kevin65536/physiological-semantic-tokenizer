"""No measured/ignored fixtures: direction, information and gradient boundaries."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.inference.shared_driver_attribution import (
    equal_capacity_hb_basis, hb_diagnostic_coordinates, project_component,
    subspace_overlap, fit_standardized_ridge, predict_standardized_ridge,
)


def test_equal_capacity_directions_preserve_Hb_algebra():
    n=24;operator=np.eye(3*n)
    fixed=equal_capacity_hb_basis(operator)
    for rho in (.1,.35,.9):
        basis=equal_capacity_hb_basis(operator,rho=rho)
        np.testing.assert_allclose(np.linalg.norm(basis,axis=0),np.linalg.norm(fixed,axis=0))
        np.testing.assert_array_equal(basis[0::3],0.)
        assert np.linalg.matrix_rank(basis)==4
    values=(fixed@np.array([1.,2.,3.,4.])).reshape(n,3)
    coordinates=hb_diagnostic_coordinates(values[:,1:])
    np.testing.assert_allclose(coordinates[:,1],0.,atol=1e-14)
    exchange=equal_capacity_hb_basis(operator,exchange=True)
    np.testing.assert_allclose(exchange[1::3]+exchange[2::3],0.)
    np.testing.assert_allclose(np.linalg.norm(exchange,axis=0),np.linalg.norm(fixed,axis=0))


def test_component_projection_has_no_hidden_or_EEG_information():
    rng=np.random.default_rng(20);y=rng.normal(size=(24,3))
    basis=equal_capacity_hb_basis(np.eye(72));mask=np.ones_like(y,bool);mask[8:16,1:]=False
    a=project_component(y,basis,[1.,2.,3.],.3,mask)
    modified=y.copy();modified[~mask]=1e30;modified[:,0]=1e25
    b=project_component(modified,basis,[1.,2.,3.],.3,mask)
    np.testing.assert_array_equal(a,b)
    only_EEG=np.zeros_like(mask);only_EEG[:,0]=True
    np.testing.assert_array_equal(project_component(y,basis,[1.,2.,3.],.3,only_EEG),0.)


def test_subspace_overlap_distinguishes_identical_and_orthogonal_spaces():
    basis=equal_capacity_hb_basis(np.eye(72))
    overlap=subspace_overlap(basis,basis,[1.,1.,1.])
    np.testing.assert_allclose(overlap['principal_cosines'],1.,atol=1e-12)
    eeg=np.zeros((72,24));eeg[0::3]=np.eye(24)
    np.testing.assert_allclose(subspace_overlap(eeg,basis,[1.,1.,1.])['principal_cosines'],0.,atol=1e-12)


def test_training_ridge_freezes_coordinate_and_predicts_independent_values():
    x=np.arange(20.).reshape(10,2);y=2*x[:,:1]-3
    model=fit_standardized_ridge(x,y,1e-10);saved=deepcopy(model)
    prediction=predict_standardized_ridge(model,x+1.)
    np.testing.assert_allclose(prediction,2*(x[:,:1]+1)-3,atol=1e-7)
    for key in model:np.testing.assert_array_equal(model[key],saved[key])


def test_typed_prototype_preserves_modality_and_reconstruction_gradient_ownership():
    import torch
    from src.tokenizers.typed_component_prototype import TypedComponentPrototype
    torch.manual_seed(1);model=TypedComponentPrototype(steps=24,width=16,driver_patches=6,morphology_modes=2)
    x=torch.randn(3,24,3);a=model(x)
    modified=x.clone();modified[:,:,1:]*=10.;b=model(modified)
    torch.testing.assert_close(a['semantic'][:,:6],b['semantic'][:,:6],rtol=0,atol=0)
    changed=x.clone();changed[:,:,0]+=20.;c=model(changed)
    torch.testing.assert_close(a['semantic'][:,6:],c['semantic'][:,6:],rtol=0,atol=0)
    a['reconstruction'].square().mean().backward()
    assert all(p.grad is None for p in model.neural_encoder.parameters())
    assert all(p.grad is None for p in model.morphology_encoder.parameters())
    assert any(p.grad is not None and torch.any(p.grad!=0) for p in model.hb_observation_encoder.parameters())


def test_new_synthetic_interventions_preserve_declared_truth_and_seed_splits(tmp_path):
    from experiments.scripts import evaluate_shared_driver_reconstruction as runner
    root=Path(__file__).resolve().parents[1]
    cfg=runner.attribution_config(root/'experiments/configs/physiology_semantic_tokenizer/shared_driver_attribution_v1.yaml')
    base=runner.semantics_config(root/cfg['source_config'])
    control=runner.attribution_synthetic_case(cfg,base,0,'mixed','nonrest','control')
    for scenario in ('common_rho070','common_exchange'):
        case=runner.attribution_synthetic_case(cfg,base,0,'mixed','nonrest',scenario)
        np.testing.assert_array_equal(control['driver'],case['driver'])
        np.testing.assert_array_equal(control['physical_truth'],case['physical_truth'])
        np.testing.assert_allclose(case['target']-control['target'],case['observation_truth'],atol=1e-15)
    changed=runner.attribution_synthetic_case(cfg,base,0,'mixed','nonrest','physiology_kappa')
    np.testing.assert_array_equal(control['driver'],changed['driver'])
    assert np.linalg.norm(changed['physical_truth']-control['physical_truth'])>1e-3
    dynamic=runner.attribution_synthetic_case(cfg,base,0,'mixed','nonrest','unmodeled_dynamics')
    np.testing.assert_array_equal(control['driver'],dynamic['driver'])
    np.testing.assert_array_equal(control['physical_truth'][:,0],dynamic['physical_truth'][:,0])
    training=runner.attribution_synthetic_case(cfg,base,0,'mixed','nonrest','control',calibration=True)
    assert not np.array_equal(control['driver'],training['driver'])
    cfg['source_run']='comparative_methods/protected';path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='boundary'):runner.attribution_config(path)


def test_spatial_donor_identity_excludes_target_and_misaligned_support():
    from experiments.scripts import evaluate_shared_driver_reconstruction as runner
    ref=dict(id='target',key='record',window=1,site='frontal',hb_channel='A',eeg_start_s=10.,hb_start_s=11.)
    good=dict(ref,id='donor',site='motor',hb_channel='B')
    copied=dict(ref,id='copied',site='motor')
    shifted=dict(good,id='shifted',hb_start_s=12.)
    assert runner.attribution_donors(ref,[ref,good,copied,shifted])==[good]


def test_spatial_training_donor_uses_target_outer_fold_and_never_target_Hb(tmp_path,monkeypatch):
    import json
    from experiments.scripts import evaluate_shared_driver_reconstruction as runner
    ref=dict(id='target',key='record',window=1,site='frontal',hb_channel='A',eeg_start_s=10.,hb_start_s=11.,
        dataset='fixture',subject='training_subject',subject_fold=1)
    donor=dict(ref,id='donor',site='motor',hb_channel='B')
    coordinate=dict(key='fixture__frontal__outer0',training_subjects=['training_subject'],hb_factor=1.,sd=[1.,1.,1.])
    path=tmp_path/'coordinates';path.mkdir()
    (path/'fixture__motor__outer0.json').write_text(json.dumps(dict(coordinate,hb_factor=2.)))
    read=[]
    def target(r,c):
        read.append(r['id']);assert r['id']=='donor'
        return np.ones((120,3))
    monkeypatch.setattr(runner,'semantics_target',target)
    monkeypatch.setattr(runner,'attribution_fit',lambda *a,**k:dict(converged=True,status='completed',physical_prediction=np.zeros((120,3))))
    cal=dict(selected_rho=.35,direction_scales={a:[.1]*4 for a in runner.ATTRIBUTION_ARMS[1:]},coefficient_multiplier=1.)
    result=runner.attribution_spatial_inputs({'direction':{'modes':4}},None,tmp_path,[ref,donor],ref,coordinate,cal)
    assert result['available'] and read==['donor']
    np.testing.assert_array_equal(result['raw_Hb'],.5)


def test_attribution_paired_scores_keep_failed_denominator():
    import pandas as pd
    from experiments.scripts import evaluate_shared_driver_reconstruction as runner
    frame=pd.DataFrame([dict(id=f'w{i}',subject=f's{i}',mode='hidden_channel',arm=arm,
        converged=not(i==2 and arm=='D-trained'),total_nrmse=value)
        for i in range(3) for arm,value in [('M0',1.),('D-trained',.5)]])
    result=runner.robustness_paired(frame,['mode'],arms=['D-trained'])[0]
    assert result['denominator']==3 and result['common_success']==2
    assert result['mean_gain']==pytest.approx(.5)
