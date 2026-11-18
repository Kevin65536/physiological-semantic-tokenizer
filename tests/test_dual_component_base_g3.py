"""G3 tests prove the main decoder blend, fixed VQ controls and matched warm start."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from test_dual_component_base_g1 import prepared,runner,ROOT
from src.analysis.dual_component_tokenizer import base_g3_decision

CONFIG=ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g3_v1.yaml'


def test_schedule_exact_endpoints_absolute_clock_and_no_replay():
    cfg=runner.read_config(CONFIG)
    assert cfg['architectures']['waveform']['latent_dim']==64 and len(cfg['tasks'])==1
    assert cfg['evaluation']['integer_replay_check']=='disabled_by_user'
    assert runner._base_decoder_blend(cfg,1001)==0.
    assert runner._base_decoder_blend(cfg,1500)==pytest.approx(499/999)
    assert runner._base_decoder_blend(cfg,2000)==1.
    assert runner._base_decoder_blend(cfg,10000)==1.
    assert runner._base_learning_rate(cfg,7000)==pytest.approx(.000165)


def test_main_forward_blends_actual_latents_without_changing_commitment_or_ema():
    cfg=runner.read_config(CONFIG);cfg['model'].update(codebook_size=8,kmeans_iters=2)
    cfg['architectures']['waveform'].update(width=16,latent_dim=8,local_width=4)
    torch.manual_seed(17);model=runner._base_training_model(cfg,cfg['tasks'][0]);x=torch.randn(4,15,400)
    model.initialize_codebooks(x);captured={};results={};models={}
    for alpha in (0.,.5,1.):
        m=deepcopy(model);m.train();m.set_decoder_quantization_blend(alpha);seen={}
        hooks=[m.branches[b]['from_latent'].register_forward_pre_hook(lambda module,inputs,b=b:seen.__setitem__(b,inputs[0].detach().clone())) for b in ('S','O')]
        results[alpha]=m(x);captured[alpha]=seen;models[alpha]=m
        for hook in hooks:hook.remove()
    for b in ('S','O'):
        torch.testing.assert_close(captured[0.][b],results[0.]['latent'][b],atol=0,rtol=0)
        torch.testing.assert_close(captured[.5][b],.5*(captured[0.][b]+captured[1.][b]),atol=1e-7,rtol=1e-6)
        assert not torch.equal(captured[0.][b],captured[1.][b])
        assert torch.equal(models[0.].branches[b]['quantizer'].codebook,models[1.].branches[b]['quantizer'].codebook)
        assert float(models[0.].branches[b]['quantizer'].quantization_strength)==1.
    assert torch.equal(results[0.]['commitment'],results[1.]['commitment'])
    loss=results[.5]['sum'].square().mean()+results[.5]['commitment'];loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in models[.5].parameters())


def test_evaluation_always_hard_even_if_training_blend_is_zero():
    cfg=runner.read_config(CONFIG);cfg['model'].update(codebook_size=8,kmeans_iters=2)
    cfg['architectures']['waveform'].update(width=16,latent_dim=8,local_width=4)
    m=runner._base_training_model(cfg,cfg['tasks'][0]);x=torch.randn(4,15,400);m.initialize_codebooks(x);m.eval()
    with torch.no_grad():
        m.set_decoder_quantization_blend(0.);zero=m(x)
        m.set_decoder_quantization_blend(1.);one=m(x)
    assert torch.equal(zero['S'],one['S']) and torch.equal(zero['O'],one['O'])


def prepared_g3(tmp_path):
    old,out=prepared(tmp_path);old_run=tmp_path/'baseline';source_cfg=tmp_path/'baseline.yaml'
    source_cfg.write_text(yaml.safe_dump(old));runner.freeze_contract(old_run,old,tmp_path)
    task=dict(modality='eeg',architecture='waveform',candidate='vq128');warm=dict(task,candidate='warmup')
    runner.train_base_tokenizer(old,tmp_path,old_run,warm,'cpu')
    cfg=deepcopy(old);cfg['schema']='dual_component_base_g3_v1';cfg['tasks']=[task]
    cfg['training'].update(quantization_transition_steps=3,transition_monitor_every=1)
    cfg['evaluation']['integer_replay_check']='disabled_by_user'
    cfg['external_warmups']={'eeg__waveform':dict(run='baseline',contract='baseline.yaml',task_id='eeg__waveform__warmup')}
    return cfg,out,old_run


def test_external_warmup_and_mid_transition_resume_preserve_full_training(tmp_path,monkeypatch):
    cfg,out,old_run=prepared_g3(tmp_path);task=cfg['tasks'][0]
    old_bytes=(old_run/'models/eeg__waveform__warmup/checkpoint.pt').read_bytes()
    original=runner.evaluate_base_training
    def interrupt(*args,**kwargs):
        if Path(args[6]).name=='step_00004':raise RuntimeError('transition interruption')
        return original(*args,**kwargs)
    monkeypatch.setattr(runner,'evaluate_base_training',interrupt)
    with pytest.raises(RuntimeError,match='transition interruption'):runner.train_base_tokenizer(cfg,tmp_path,out,task,'cpu')
    checkpoint=torch.load(out/'models/eeg__waveform__vq128/checkpoint.pt',weights_only=False)
    assert checkpoint['step']==4 and checkpoint['decoder_quantization_blend']==.5
    monkeypatch.setattr(runner,'evaluate_base_training',original)
    resumed=runner.train_base_tokenizer(cfg,tmp_path,out,task,'cpu')
    direct_run=tmp_path/'direct';direct=runner.train_base_tokenizer(cfg,tmp_path,direct_run,task,'cpu')
    a=torch.load(out/'models/eeg__waveform__vq128/checkpoint.pt',weights_only=False)
    b=torch.load(direct_run/'models/eeg__waveform__vq128/checkpoint.pt',weights_only=False)
    for key in a['model_state']:assert torch.equal(a['model_state'][key],b['model_state'][key])
    assert (old_run/'models/eeg__waveform__warmup/checkpoint.pt').read_bytes()==old_bytes
    assert resumed['new_training_steps']==4 and direct['new_training_steps']==6
    history={r['step']:r for r in direct['history']}
    assert history[3]['decoder_quantization_blend']==0 and history[5]['decoder_quantization_blend']==1
    assert 'hard_minus_continuous_nmse' in history[4] and 'fixed_training_probe_usage' in history[4]
    assert (direct_run/'models/eeg__waveform__vq128/checkpoint_step_00005.pt').is_file()
    assert (out/'models/eeg__waveform__vq128/warmup_reuse.json').is_file()


def test_warmup_rejects_unmatched_architecture_and_exposure(tmp_path):
    cfg,_,_=prepared_g3(tmp_path)
    bad=deepcopy(cfg);bad['architectures']['waveform']['latent_dim']=4
    with pytest.raises(ValueError,match='identical model'):runner._base_external_warmup(bad,tmp_path,bad['tasks'][0])
    bad=deepcopy(cfg);bad['training']['batch_size']*=2
    with pytest.raises(ValueError,match='identical model'):runner._base_external_warmup(bad,tmp_path,bad['tasks'][0])


def test_G3_screen_keeps_reconstruction_and_branch_health_guards():
    screen=dict(reconstruction_improvement_min=.1,component_degradation_max=.05,reconstruction_close_relative=.02,
                effective_code_improvement_min=.1,health_branch_degradation_max=.05)
    def record(error,effective):
        return dict(status='completed',milestones={'10000':dict(validation=dict(metrics=dict(nmse=dict(macro=[[error]]*3)),
            code_health=[dict(dataset='macro',branch=b,effective=effective,top1=.1) for b in ('S','O')]))})
    base=record(1.,60)
    assert base_g3_decision(record(.85,60),base,screen,10000)['selected_schedule']=='gradual'
    assert base_g3_decision(record(1.01,70),base,screen,10000)['health_screen_pass']
    assert base_g3_decision(record(1.04,70),base,screen,10000)['selected_schedule']=='direct_hard'
    assert base_g3_decision(dict(status='failed'),base,screen,10000)['status']=='incomplete'


def test_G3_synthetic_exercises_ramp_without_replay_comparison(monkeypatch):
    monkeypatch.setattr(torch.testing,'assert_close',lambda *a,**k:pytest.fail('cancelled replay comparison'))
    result=runner.base_training_synthetic(runner.read_config(CONFIG))
    assert result['status']=='passed' and result['integer_replay_check']=='disabled_by_user'
    assert [r['alpha'] for r in result['cases'][0]['transition_cases']]==[0.,.5,1.]
    assert all(r['commitment_strength']==[1.,1.] for r in result['cases'][0]['transition_cases'])


def test_G3_rejects_wrong_dimension_before_loading_any_model_or_arrays(tmp_path,monkeypatch):
    cfg=runner.read_config(CONFIG);cfg['selection_run']='selection';cfg['completion_review']='review.json'
    runner.write_json(tmp_path/'selection/summary.json',dict(status='completed',completed_models=5,planned_models=5,
        groups={'G2':dict(selected_latent_dim=32)},G1_budget_after_resumption=dict(budget=10000)))
    runner.write_json(tmp_path/'review.json',dict(status='passed',selection_rules_recomputed=True))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('unexpected checkpoint read'))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('unexpected array read'))
    with pytest.raises(ValueError,match='dimension/budget'):runner._base_g3_sources(cfg,tmp_path,'unused')


def test_explicit_gpu1_reaches_worker_without_cuda_renumbering(tmp_path,monkeypatch):
    from types import SimpleNamespace
    cfg=runner.read_config(CONFIG);args=SimpleNamespace(output=tmp_path,config=CONFIG,project_root=ROOT,devices=['cuda:1'])
    devices=[]
    class Process:
        def __init__(self,command,**kwargs):
            devices.append(command[command.index('--device')+1]);self.pid=123
            key=command[command.index('--candidate')+1]
            runner.write_json(tmp_path/'models'/key/'summary.json',dict(status='completed',task_id=key))
        def poll(self):return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    result=runner._base_training_jobs(cfg,args,cfg['tasks'],1)
    assert devices==['cuda:1'] and len(result['results'])==1
    runner.main(['--config',str(CONFIG),'--stage','check','--devices','cuda:1'])
    with pytest.raises(ValueError,match='one declared GPU'):
        runner.main(['--config',str(CONFIG),'--stage','check','--devices','cuda:2'])
