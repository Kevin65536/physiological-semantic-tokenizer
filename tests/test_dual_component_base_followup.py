"""Explicit dense continuation and G2/G4 controls, independent of local run artifacts."""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from test_dual_component_base_g1 import prepared, runner, ROOT
from src.analysis.dual_component_tokenizer import base_followup_decisions, base_training_metrics

CONFIG=ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g124_v1.yaml'


def test_frozen_matrix_shapes_and_no_replay_check():
    cfg=runner.read_config(CONFIG)
    assert len(cfg['tasks'])==5 and cfg['evaluation']['integer_replay_check']=='disabled_by_user'
    assert 'replay_atol' not in cfg['evaluation']
    assert cfg['architectures']['waveform32']['latent_dim']==32
    assert cfg['architectures']['local_shape']['context_ratio']==.25
    assert cfg['resumption']['eeg__dense_long__vq128']['step']==7000
    for task in cfg['tasks']:
        model=runner._base_training_model(cfg,task)
        assert model.branches['S']['to_latent'].out_features==(32 if task['architecture'] in ('waveform32','dense_long') else 64)


@pytest.mark.parametrize('candidate',['continuous','vq128'])
def test_local_shape_mean_context_bound_and_context_independence(candidate):
    cfg=runner.read_config(CONFIG);model=runner._base_training_model(cfg,dict(modality='hb',architecture='local_shape',candidate=candidate))
    model.eval();z=torch.randn(2,15,64)
    a=model._decode_component_parts('S',z)
    local=a['local'].reshape(2,15,20,2);context=a['context'].reshape_as(local)
    assert torch.max(torch.abs(local.mean(2)))<1e-6
    assert torch.max(torch.abs(context.mean(2)))<1e-6
    assert torch.all(torch.linalg.vector_norm(context,dim=2)<=.25*torch.linalg.vector_norm(local,dim=2)+1e-6)
    assert torch.equal(a['full'],a['level']+a['local']+a['context'])
    changed=z.clone();changed[:,:7]+=3;changed[:,8:]-=2
    b=model._decode_component_parts('S',changed)
    assert torch.equal(a['local'][:,7],b['local'][:,7])
    assert not torch.equal(a['level'][:,7],b['level'][:,7])


def test_zero_prototype_has_zero_context_and_finite_backward():
    cfg=runner.read_config(CONFIG);model=runner._base_training_model(cfg,dict(modality='hb',architecture='local_shape',candidate='continuous'))
    with torch.no_grad():
        model.branches['S']['local_shape'].weight.zero_();model.branches['S']['local_shape'].bias.zero_()
    z=torch.randn(2,15,64,requires_grad=True);parts=model._decode_component_parts('S',z)
    assert torch.count_nonzero(parts['local'])==0 and torch.count_nonzero(parts['context'])==0
    (parts['full']-torch.randn_like(parts['full'])).square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.branches['S']['local_shape'].weight.grad.abs().sum()>0


def test_disabled_comparison_uses_actual_integer_decoder_without_forward(tmp_path,monkeypatch):
    cfg,out=prepared(tmp_path);cfg['evaluation']['integer_replay_check']='disabled_by_user'
    manifest,scales,audit,_=runner._base_training_data(cfg,tmp_path)
    task=dict(modality='hb',architecture='waveform',candidate='vq128');model=runner._base_training_model(cfg,task)
    runner._base_initialize(model,manifest['rows'],audit['hb']['initialization'],scales,'cpu',OrderedDict(),cfg)
    monkeypatch.setattr(model,'forward',lambda *a,**k:pytest.fail('disabled comparison called old forward'))
    monkeypatch.setattr(torch.testing,'assert_close',lambda *a,**k:pytest.fail('integer replay comparison still active'))
    result=runner.evaluate_base_training(cfg,model,manifest['rows'],scales,'hb','cpu',out/'evaluation',OrderedDict(),limit=8)
    assert result['integer_replay_check']=='disabled_by_user' and result['integer_replay_max_abs'] is None
    assert np.isfinite(result['validation']['selection_score'])


def test_versioned_resume_preserves_optimizer_rng_and_old_evidence(tmp_path,monkeypatch):
    old,out=prepared(tmp_path);old_run=tmp_path/'previous';contract=tmp_path/'previous.yaml'
    contract.write_text(yaml.safe_dump(old));runner.freeze_contract(old_run,old,tmp_path)
    task=dict(modality='eeg',architecture='dense_long',candidate='vq128');warm=dict(task,candidate='warmup')
    runner.train_base_tokenizer(old,tmp_path,old_run,warm,'cpu');evaluate=runner.evaluate_base_training
    def interrupt(*args,**kwargs):
        if Path(args[6]).name=='step_00006':raise RuntimeError('saved step 6')
        return evaluate(*args,**kwargs)
    monkeypatch.setattr(runner,'evaluate_base_training',interrupt)
    with pytest.raises(RuntimeError,match='saved step 6'):runner.train_base_tokenizer(old,tmp_path,old_run,task,'cpu')
    monkeypatch.setattr(runner,'evaluate_base_training',evaluate)
    previous=old_run/'models/eeg__dense_long__vq128';before=(previous/'checkpoint.pt').read_bytes();failure=(previous/'failure.json').read_bytes()
    cfg=deepcopy(old);cfg['schema']='dual_component_base_followup_v1';cfg['evaluation']['integer_replay_check']='disabled_by_user'
    cfg['resumption']={'eeg__dense_long__vq128':dict(run='previous',contract='previous.yaml',step=6,reuse_milestones=[4])}
    resumed=runner.train_base_tokenizer(cfg,tmp_path,out,task,'cpu')
    assert resumed['resumed_from_step']==6 and resumed['new_training_steps']==2
    assert resumed['milestones']['6']['integer_replay_check']=='disabled_by_user'
    assert (out/'models/eeg__dense_long__vq128/evaluation/step_00004').is_symlink()
    assert (previous/'checkpoint.pt').read_bytes()==before and (previous/'failure.json').read_bytes()==failure
    reference=tmp_path/'reference';runner.train_base_tokenizer(old,tmp_path,reference,warm,'cpu');runner.train_base_tokenizer(old,tmp_path,reference,task,'cpu')
    a=torch.load(out/'models/eeg__dense_long__vq128/checkpoint.pt',weights_only=False)
    b=torch.load(reference/'models/eeg__dense_long__vq128/checkpoint.pt',weights_only=False)
    for key in a['model_state']:assert torch.equal(a['model_state'][key],b['model_state'][key])
    assert a['numpy_rng']==b['numpy_rng']


def test_scheduler_starts_resume_and_warmups_without_serial_gate(tmp_path,monkeypatch):
    cfg=runner.read_config(CONFIG);cfg['resources']['devices']=['cpu:0','cpu:1']
    args=SimpleNamespace(output=tmp_path,config=CONFIG,project_root=ROOT)
    dense=dict(modality='eeg',architecture='dense_long',candidate='vq128')
    warm=dict(modality='eeg',architecture='waveform32',candidate='warmup')
    child=dict(warm,candidate='continuous');launched=[]
    class Process:
        def __init__(self,command,**kwargs):
            self.key=command[command.index('--candidate')+1];self.pid=100+len(launched);self.calls=0
            if self.key.endswith('continuous'):assert (tmp_path/'models/eeg__waveform32__warmup/summary.json').exists()
            launched.append(self.key)
        def poll(self):
            self.calls+=1
            if 'dense_long' in self.key and self.calls<4:return None
            runner.write_json(tmp_path/'models'/self.key/'summary.json',dict(status='completed',task_id=self.key))
            return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Process);monkeypatch.setattr(runner.time,'sleep',lambda n:None)
    result=runner._base_training_jobs(cfg,args,[dense,warm,child],1,dependencies={'eeg__waveform32__continuous':'eeg__waveform32__warmup'})
    assert len(result['results'])==3 and launched[:2]==['eeg__dense_long__vq128','eeg__waveform32__warmup']


def test_followup_decisions_keep_continuous_and_component_guards():
    cfg=runner.read_config(CONFIG)
    def record(key,value,pattern=1.,status='completed'):
        validation=dict(metrics=dict(nmse=dict(macro=np.full((3,2),value).tolist())),
            single_ID_centered_readout=dict(macro=np.full((4,2),pattern).tolist()),single_ID_unsupported_fraction={'S_s_unsupported_fraction':0.})
        return dict(task_id=key,status=status,milestones={'10000':dict(validation=validation)})
    baseline={m+'__waveform__'+c:record(m+'__waveform__'+c,1.) for m in ('eeg','hb') for c in ('continuous','vq128')}
    models=[record('eeg__waveform32__continuous',1.),record('eeg__waveform32__vq128',.8),
            record('hb__local_shape__continuous',1.),record('hb__local_shape__vq128',1.01,pattern=.8)]
    result=base_followup_decisions(models,baseline,cfg['evaluation']['screen'],10000)
    assert result['G2']['selected_latent_dim']==32 and result['G4']['pattern_screen_pass']
    models[0]['milestones']['10000']['validation']['metrics']['nmse']['macro']=[[1.2,1.2]]*3
    assert base_followup_decisions(models,baseline,cfg['evaluation']['screen'],10000)['G2']['selected_latent_dim']==64
    models[3]['status']='failed'
    assert base_followup_decisions(models,baseline,cfg['evaluation']['screen'],10000)['G4']['status']=='incomplete'


def test_followup_synthetic_never_performs_replay_comparison(monkeypatch):
    monkeypatch.setattr(torch.testing,'assert_close',lambda *a,**k:pytest.fail('cancelled replay check'))
    result=runner.base_training_synthetic(runner.read_config(CONFIG))
    assert result['status']=='passed' and result['integer_replay_check']=='disabled_by_user'


def test_patch_boundary_error_is_prediction_minus_target_jump():
    x=np.zeros((2,15,40));level=np.broadcast_to(np.arange(15)[None,:,None]*.25,x.shape)
    batch=dict(x=x,s=x,o=x,mask=np.ones_like(x,bool),**{k:np.ones_like(x) for k in ('vx','vs','vo')})
    result=base_training_metrics(dict(sum=level,S=level*.5,O=level*.5),batch,2)
    expected=np.broadcast_to(np.array([.25,.125,.125])[None,:,None],(2,3,2))
    np.testing.assert_allclose(result['boundary_jump_bias'],expected)
    np.testing.assert_allclose(result['boundary_jump_nmse'],expected**2)
