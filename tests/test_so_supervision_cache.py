"""Synthetic long-window cache contracts; no measured inputs."""
from pathlib import Path
import json
import numpy as np
import pytest
from src.data.dual_component_tokenizer import fit_voltage_power_cap_reference, voltage_power_cap_reference, voltage_power_cap_target
from src.data import so_supervision_cache as cache


def fixture_statistics(c=4):
    return dict(mean=np.zeros(c*5),covariance=np.eye(c*5),baseline_log_power=np.zeros(c*5),
        correlation=np.tile(np.eye(c),(6,1,1)),complement_power=np.ones(c),sample_rate_hz=np.array(200))


def test_streamed_reference_matches_original_training_formula():
    rng=np.random.default_rng(64);x=[rng.normal(size=(18000,4)) for _ in range(2)];a=rng.normal(size=(3,20))
    old=fit_voltage_power_cap_reference([dict(x=v,split='train') for v in x],a)
    new=cache.combine_reference_moments([cache.reference_moments(v,200) for v in x])
    for key in ['mean','covariance','baseline_log_power','correlation','complement_power']:
        np.testing.assert_allclose(new[key],old[key],rtol=1e-10,atol=1e-11)


def test_generalized_subject_projection_exactly_completes_short_proxy():
    rng=np.random.default_rng(55);a=rng.normal(size=(7,20));stats=fixture_statistics()
    model=cache.bind_reference(stats,a,600);p=rng.normal(size=(600,7))*.01
    power=voltage_power_cap_reference(model,p,np.arange(600)/10)
    ell=np.log(power[:,:5].transpose(0,2,1).reshape(600,20))
    np.testing.assert_allclose(ell@a.T,p,rtol=1e-10,atol=1e-11)
    with pytest.raises(ValueError,match='Rank-deficient'):cache.bind_reference(stats,np.stack([a[0],a[0]]),600)
    with pytest.raises(ValueError,match='complete finite'):voltage_power_cap_reference(dict(model,proxy_steps=900),p,np.arange(600)/10)


def test_batched_whitening_and_reanalysis_equal_full_operator():
    rng=np.random.default_rng(6);x=rng.normal(size=(1200,4))*4;model=cache.bind_reference(fixture_statistics(),rng.normal(size=(3,20)),60);p=np.zeros((60,3))
    a=voltage_power_cap_target(x,model,p);b=voltage_power_cap_target(x,model,p,frame_batch_size=7)
    for key in ['s','o','reference_power','local_output_ratio','final_reanalysis_ratio']:
        np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(x,b['s']+b['o'],rtol=1e-12,atol=1e-12)
    assert b['local_output_ratio'].max()<=1+1e-8


def test_bad_source_split_rejected_before_native_read(tmp_path,monkeypatch):
    cfg=dict(source_ssm='parent',measurement='measurement');w=dict(eeg_start_s=0,duration_s=90)
    r=dict(cache_windows=[w],split='train')
    monkeypatch.setattr(cache,'parent_plan',lambda _: ({},{'r':r},{}));monkeypatch.setattr(cache,'measurement_index',lambda _:None)
    monkeypatch.setattr(cache,'load_native_eeg_record',lambda *args:pytest.fail('raw read'))
    g=dict(members=[dict(record_key='r',window=w)],split='test')
    with pytest.raises(ValueError,match='before native read'):cache.read_native(tmp_path,cfg,g)


def test_failed_hb_positions_keep_observation_and_false_teacher_mask(tmp_path):
    g=dict(key='d/s/r/w',dataset='d',subject='s',split='train',samples=1200,duration_s=6,EEG_channels=['a','b'])
    path=tmp_path/'windows'/g['key'];rng=np.random.default_rng(1);x=rng.normal(size=(60,2,2));s=x*.6;o=x-s;mask=np.ones_like(x,bool);mask[:,1]=False;s[:,1]=np.nan;o[:,1]=np.nan
    e=rng.normal(size=(1200,2));cache.save_arrays(path.with_suffix('.npz'),eeg_x=e,eeg_s=e,eeg_o=np.zeros_like(e),eeg_teacher_valid=np.ones_like(e,bool),
        fnirs_x=x,fnirs_s=s,fnirs_o=o,fnirs_teacher_valid=mask,source_scoring_mask=np.r_[np.ones(59,bool),False],local_cap_ratio=np.ones((61,6)))
    meta=dict(schema=cache.SCHEMA,source=g,status='completed',EEG_available=True,array_file=str(path.with_suffix('.npz')),source_task_paths=['a','b'],SSM_qualified_pairs=1,SSM_failed_pairs=1,fNIRS_positions=['p1','p2'])
    cache.write_json(path.with_suffix('.json'),meta);r=cache.verify_worker((str(tmp_path),g));assert not r['issues'] and r['SSM_failed_pairs']==1
    cache.write_json(tmp_path/'cache_manifest.json',dict(schema=cache.SCHEMA,status='completed',windows=[dict(split='train',metadata=str(path.with_suffix('.json')))]))
    dataset=cache.SOSupervisionDataset(tmp_path,split='train');assert len(dataset)==1
    assert np.isnan(dataset[0]['arrays']['fnirs_s'][:,1]).all()


def test_three_proxy_selection_uses_only_training_statistics_and_projection():
    rng=np.random.default_rng(47);stats=fixture_statistics();a=rng.normal(size=(12,20))
    selected=cache.select_proxy_rows(stats,a,3)
    assert len(selected)==3 and len(set(selected))==3 and np.all(selected[:-1]<selected[1:])
    model=cache.bind_reference(stats,a[selected],790)
    np.testing.assert_allclose(model['projection']@model['lift'],np.eye(3),atol=1e-12)
    np.testing.assert_array_equal(cache.select_proxy_rows(stats,a,3),selected)
