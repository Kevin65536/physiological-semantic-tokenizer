"""Synthetic contracts for contextual-ID pretraining; no retained arrays needed."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import json

import numpy as np
import pytest
import torch

from src.foundation.dual_component_masked import SpatialTemporalMaskedModel, spatial_temporal_masks, masked_code_loss
from src.data.dual_component_tokenizer import masked_state_plan, masked_state_identity, masked_observation_features

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('masked_state_runner', ROOT/'experiments/scripts/run_dual_component_tokenizer.py')
runner=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(runner)
CONFIG=ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_masked_state_v1.yaml'


def sample(interaction='spatial',geometry=True):
    torch.manual_seed(17);torch.set_num_threads(2)
    model=SpatialTemporalMaskedModel(positions=4,vocab_size=8,width=16,layers=2,heads=2,dropout=0.,interaction=interaction,geometry=geometry)
    ids=torch.randint(8,(2,4,15,2));valid=torch.ones(2,4,15,dtype=torch.bool)
    visible=valid.clone();visible[:,:,5:8]=False
    return model,ids,visible,valid,torch.tensor([0,0,1,1]),torch.arange(4),torch.randn(4,3)


def test_hidden_targets_never_enter_input_and_gradient_is_finite():
    model,ids,visible,valid,*layout=sample()
    a=model(ids,visible,valid,*layout)
    changed=ids.clone();changed[~visible]=99999
    b=model(changed,visible,valid,*layout)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    loss=masked_code_loss(a,ids,valid&~visible,layout[0]);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_bfloat16_autocast_with_float32_residual_stream():
    model,ids,visible,valid,*layout=sample('joint')
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
        logits=model(ids,visible,valid,*layout)
        loss=masked_code_loss(logits,ids,valid&~visible,layout[0])
    loss.backward()
    assert torch.isfinite(logits).all() and torch.isfinite(loss)


@pytest.mark.parametrize('interaction',['temporal','spatial'])
def test_modality_isolation_without_cls_bypass(interaction):
    model,ids,visible,valid,*layout=sample(interaction)
    model.eval();a=model(ids,visible,valid,*layout)
    changed=ids.clone();changed[:,2:]=(changed[:,2:]+1)%8
    b=model(changed,visible,valid,*layout)
    torch.testing.assert_close(a[:,:2],b[:,:2],rtol=0,atol=0)


def test_temporal_model_does_not_mix_positions():
    model,ids,visible,valid,*layout=sample('temporal')
    a=model(ids,visible,valid,*layout);changed=ids.clone();changed[:,1]=(changed[:,1]+1)%8
    torch.testing.assert_close(a[:,0],model(changed,visible,valid,*layout)[:,0],rtol=0,atol=0)


def test_anonymous_model_is_sensor_permutation_equivariant():
    model,ids,visible,valid,m,s,c=sample(geometry=False)
    a=model(ids,visible,valid,m,s,c);perm=torch.tensor([1,0,3,2])
    b=model(ids[:,perm],visible[:,perm],valid[:,perm],m[perm],s[perm],c[perm])
    torch.testing.assert_close(a[:,perm],b,atol=1e-5,rtol=1e-5)


def test_absent_position_and_modality_have_finite_outputs():
    model,ids,visible,valid,*layout=sample('joint')
    valid[:,1:]=False;visible&=valid
    out=model(ids,visible,valid,*layout)
    assert torch.isfinite(out).all()
    masked_code_loss(out,ids,valid&~visible,layout[0]).backward()


def test_masks_equal_exposure_joint_branches_and_invalid_support():
    valid=np.ones((3,9,15),bool);valid[:,8]=False
    counts=[]
    for task in ('random','span','space'):
        v,s=spatial_temporal_masks(valid,task,np.random.default_rng(17))
        assert not np.any(v&s) and not np.any(s&~valid)
        assert np.array_equal(v|s,valid)
        counts.append(s.sum())
    assert len(set(counts))==1


def test_hb_features_use_seconds_and_signed_change():
    t=np.arange(20)/10
    x=np.stack([2+3*t,-4-2*t],-1)[None,None]
    f=masked_observation_features(x,'hb',10)[0,0]
    np.testing.assert_allclose(f[4:6],[3,-2],rtol=1e-6)
    np.testing.assert_allclose(f[6:8],[5.7,-3.8],rtol=1e-6)


def test_heldout_signals_cannot_change_training_prior():
    manifest={'layouts':{'d':{'eeg':{'positions':['a']},'hb':{'positions':['b']}}},
              'samples':[{'dataset':'d','split':'train'},{'dataset':'d','split':'validation'}]}
    arrays=[{'ids':np.zeros((2,15,2),int),'valid':np.ones((2,15),bool)} for _ in range(2)]
    before=runner._masked_priors(manifest,arrays,8)
    arrays[1]['ids'][:]=7
    np.testing.assert_array_equal(before['d'],runner._masked_priors(manifest,arrays,8)['d'])


def test_subject_macro_does_not_weight_long_records_more():
    rows=[dict(dataset='d',subject='s',record='long',value=1.) for _ in range(9)]+[dict(dataset='d',subject='s',record='short',value=3.)]
    result=runner._masked_macro(rows,[],['value'])
    assert result['macro'][0]['value']==2.


def test_forbidden_split_rejected_before_any_file_access(tmp_path):
    cfg=runner.read_config(CONFIG);bad=deepcopy(cfg['data']);bad['allowed_splits'].append('test')
    with pytest.raises(ValueError,match='before reads'):masked_state_plan(tmp_path,bad,tmp_path/'missing')
    with pytest.raises(ValueError,match='before arrays'):
        masked_state_identity(tmp_path,{'split':'test'},{},cfg['data'])


def test_plan_deduplicates_input_support_without_reading_test_metadata(tmp_path):
    cfg=runner.read_config(CONFIG);spec=cfg['data'];root=tmp_path/'cache';root.mkdir();windows=[]
    for split,subject,starts in [('train','s1',[0,30]),('validation','s2',[0])]:
        for start in starts:
            key=f'eeg_fnirs_single_trial/{subject}/record/w{start}';path=root/'windows'/f'{key}.json';path.parent.mkdir(parents=True,exist_ok=True)
            source=dict(key=key,dataset='eeg_fnirs_single_trial',subject=subject,split=split,rate=200,
                        start_sample=start*200,duration_s=90,native_record='record')
            meta=dict(source=source,schema=spec['source_schema'],array_file=str(path.with_suffix('.npz')),
                      **{k:source[k] for k in ('key','dataset','subject','split')})
            path.write_text(json.dumps(meta));windows.append(dict(metadata=str(path),**{k:source[k] for k in ('key','dataset','subject','split')}))
    windows.append(dict(dataset='eeg_fnirs_single_trial',subject='forbidden',split='test',metadata='/must/not/read'))
    (root/'cache_manifest.json').write_text(json.dumps(dict(schema=spec['source_schema'],status='completed',source_ssm=str(tmp_path/'ssm'),windows=windows)))
    (root/'verification.json').write_text('{"status":"passed"}')
    plan=masked_state_plan(root,spec,tmp_path/'ssm')
    assert plan['planned_inputs']==7 and plan['measured_arrays_read']==0
    assert len(plan['overlapping_inputs_skipped'])==2


def test_synthetic_spatial_learning_and_independent_null():
    r=runner.masked_state_synthetic(runner.read_config(CONFIG))
    assert r['status']=='passed' and r['hidden_ID_influence']==0


def test_training_evaluation_and_completed_resume_on_temporary_tokens(tmp_path):
    from types import SimpleNamespace
    cfg=deepcopy(runner.read_config(CONFIG));cfg['model'].update(width=16,layers=1,heads=2,dropout=0.)
    cfg['training'].update(steps=2,batch_size=2,checkpoint_every=1,warmup_steps=1)
    cfg['evaluation']['probes_positions_per_modality']=1
    samples=[];layouts={};offsets={};rng=np.random.default_rng(12)
    for j,d in enumerate(cfg['data']['datasets']):
        layouts[d]={m:dict(positions=[m],coordinates=[[0.,0.,1.]]) for m in ('eeg','hb')};offsets[d]=2*j
        for split in ('train','validation'):
            path=tmp_path/f'{d}_{split}.npz'
            np.savez(path,ids=rng.integers(0,128,(2,15,2),dtype=np.int16),valid=np.ones((2,15),bool),features=rng.normal(size=(2,15,8)).astype('float32'))
            samples.append(dict(path=str(path),dataset=d,subject=split,record='record',parent='parent',start_s=0,
                                absolute_start_sample=0,split=split))
    runner.write_json(tmp_path/'tokens/manifest.json',dict(status='completed',samples=samples,layouts=layouts,sensor_offsets=offsets,total_positions=8))
    args=SimpleNamespace(output=tmp_path,candidate='spatial',seed=17,device='cpu',pilot_output=None)
    r=runner.train_masked_state(cfg,args)
    assert r['status']=='completed' and r['completed_steps']==2
    job=tmp_path/'models/spatial/seed_17';before=(job/'checkpoint.pt').stat().st_mtime_ns
    scored=json.loads((job/'masked_rows.json').read_text())
    assert sum(row['masked_IDs'] for row in scored)==4*3*15*2
    assert {row['task'] for row in scored}=={'random','span','space'}
    resumed=runner.train_masked_state(cfg,args)
    assert resumed['completed_steps']==2 and (job/'checkpoint.pt').stat().st_mtime_ns==before
    assert (job/'physiology_probes.json').is_file()


def test_frozen_reference_json_yaml_identity_and_unseen_sensor_scale(tmp_path,monkeypatch):
    import yaml
    import src.data.dual_component_tokenizer as data
    cfg=deepcopy(runner.read_config(CONFIG));cfg['data']['datasets']=['d']
    source={'evaluation':{None:'null hypothesis'},'training':{'continuation_steps':3000}}
    parent=tmp_path/cfg['tokenizer']['run'];protocol=tmp_path/cfg['tokenizer']['contract'];protocol.parent.mkdir(parents=True)
    protocol.write_text(yaml.safe_dump(source));runner.write_json(parent/'resolved_config.json',dict(config=source))
    records=[];rows=[]
    for mod in ('eeg','hb'):
        spec=cfg['tokenizer'][mod];checkpoint=parent/'models'/mod/(spec['architecture']+'__'+spec['candidate'])/'seed_17/checkpoint.pt'
        checkpoint.parent.mkdir(parents=True);checkpoint.touch()
        records.append(dict(modality=mod,**spec,checkpoint=str(checkpoint),health_pass=True,macro={}))
        rows.append(dict(modality=mod,dataset='d',duration_s=90,record='r',native_start_sample=0,split='train',positions=['a','b'],scale=[[2],[4]],parent='p',subject='s',training_scale_end_s=60))
    runner.write_json(parent/'evaluation_export_v3/comparison.json',dict(status='completed',records=records))
    runner.write_json(tmp_path/cfg['tokenizer']['normalization_run']/'cache/manifest.json',dict(status='completed',allowed_splits=['train'],rows=rows))
    meta=dict(split='train',key='p',EEG_channels=['a','b','c'],fNIRS_positions=['a','b','c'])
    plan={'windows':[{'row':{'dataset':'d'},'metadata':meta}]}
    monkeypatch.setattr(data,'masked_state_geometry',lambda root,spec,meta,mod,names:(np.zeros((len(names),3)),[]))
    r=runner._masked_references(cfg,tmp_path,plan)
    assert r['scales']['hb']['d']['scales']['c']==3
    assert r['scales']['hb']['d']['fallback_positions']==['c']
    source['training']['continuation_steps']=1;protocol.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError,match='contract drift'):runner._masked_references(cfg,tmp_path,plan)
