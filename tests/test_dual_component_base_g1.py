"""Five-model G1 contracts, exact continuation and evaluation on small G0 fixtures."""
from collections import OrderedDict
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.analysis.dual_component_tokenizer import base_training_metrics,base_training_aggregate,base_training_budget
from src.data.dual_component_tokenizer import base_tokenizer_batch,base_tokenizer_plan
from test_dual_component_base_g0 import fixture as g0_fixture

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('g1_test_runner',ROOT/'experiments/scripts/run_dual_component_tokenizer.py')
runner=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(runner)
CONFIG=ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v1.yaml'


def prepared(tmp_path):
    parent_cfg,cache,source,regression=g0_fixture(tmp_path)
    plan=base_tokenizer_plan(cache,parent_cfg['data'],source,regression);out=tmp_path/'g0'
    records=[runner._base_prepare_window((parent_cfg,str(out),str(cache),w,False)) for w in plan['windows']]
    runner._base_finalize(parent_cfg,out,plan,records)
    result=runner.verify_base_tokenizer(parent_cfg,out,plan)
    runner.write_json(out/'summary.json',dict(result,status='completed',retained_source_failures=[]))
    runner.write_json(out/'resolved_config.json',dict(config=parent_cfg))
    contract=tmp_path/'parent.yaml';contract.write_text(yaml.safe_dump(parent_cfg))
    cfg=deepcopy(runner.read_config(CONFIG));cfg['data'].update(run='g0',contract='parent.yaml')
    cfg['model'].update(codebook_size=8,heads=2,encoder_layers=1,decoder_layers=1,kmeans_iters=2)
    for a in cfg['architectures'].values():a.update(width=16,latent_dim=8,local_width=4)
    cfg['training'].update(warmup_steps=2,total_steps=8,batch_size=4,monitor_examples=8,checkpoint_every=2,decay_start_step=4)
    cfg['evaluation'].update(steps=[4,6,8],batch_size=8)
    cfg['resources'].update(benchmark_steps=2,benchmark_evaluation_examples=8,mmap_parent_cache=4)
    return cfg,tmp_path/'run'


def test_G1_contract_matrix_schedule_and_forbidden_stage(tmp_path):
    cfg=runner.read_config(CONFIG)
    assert len(cfg['tasks'])==5 and cfg['training']['total_steps']==10000
    assert runner._base_learning_rate(cfg,1000)==pytest.approx(.0003)
    assert runner._base_learning_rate(cfg,4000)==pytest.approx(.0003)
    assert runner._base_learning_rate(cfg,7000)==pytest.approx(.000165)
    assert runner._base_learning_rate(cfg,10000)==pytest.approx(.00003)
    with pytest.raises(ValueError,match='development stages only'):
        runner.main(['--config',str(CONFIG),'--stage','all','--output',str(tmp_path)])


def test_G0_scope_path_and_frozen_scale_rejected_before_arrays(tmp_path,monkeypatch):
    cfg,out=prepared(tmp_path)
    manifest,scales,audit,summary=runner._base_training_data(cfg,tmp_path)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('unexpected numerical read'))
    bad=deepcopy(cfg);bad['data']['allowed_splits'].append('test')
    with pytest.raises(ValueError,match='boundary'):runner._base_training_data(bad,tmp_path)
    manifest['rows'][0]['folder']='/unapproved'
    runner.write_json(tmp_path/'g0/cache/manifest.json',manifest)
    with pytest.raises(ValueError,match='folder identity'):runner._base_training_data(cfg,tmp_path)


def test_mmap_cache_matches_uncached_batches_and_rejects_escape(tmp_path):
    cfg,out=prepared(tmp_path);manifest,scales,audit,_=runner._base_training_data(cfg,tmp_path)
    examples=audit['eeg']['initialization'];rows=manifest['rows'];cache=OrderedDict()
    expected=base_tokenizer_batch(rows,examples,scales)
    actual=base_tokenizer_batch(rows,examples,scales,array_cache=cache,cache_limit=2)
    assert len(cache)<=2
    for k in expected:np.testing.assert_array_equal(actual[k],expected[k])
    row=rows[examples[0][0]];path=Path(row['folder'])/'x.npy';saved=path.with_name('x_original.npy')
    path.rename(saved);path.symlink_to(tmp_path/'external.npy')
    with pytest.raises(ValueError,match='escapes'):base_tokenizer_batch(rows,examples,scales,array_cache=cache)


def test_masked_nan_targets_have_finite_loss_and_gradients(tmp_path):
    cfg,out=prepared(tmp_path);manifest,scales,audit,_=runner._base_training_data(cfg,tmp_path)
    task=dict(modality='hb',architecture='waveform',candidate='continuous')
    batch=runner._base_training_batch(manifest['rows'],audit['hb']['initialization'],scales,'cpu',OrderedDict(),cfg)
    assert (~batch['mask']).any() and torch.isfinite(batch['s']).all()
    model=runner._base_training_model(cfg,task)
    loss,_=runner._state_refinement_loss(model(batch['x']),batch,cfg['architectures']['waveform'])
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_mmap_cache_respects_small_process_file_limit(tmp_path):
    import resource
    cfg,out=prepared(tmp_path);manifest,scales,audit,_=runner._base_training_data(cfg,tmp_path)
    examples=audit['eeg']['initialization'];rows=manifest['rows']
    expected=base_tokenizer_batch(rows,examples,scales);cache=OrderedDict()
    limits=resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE,(96,limits[1]))
        actual=base_tokenizer_batch(rows,examples,scales,array_cache=cache,cache_limit=256)
        assert len(cache)==1
        for k in expected:np.testing.assert_array_equal(actual[k],expected[k])
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE,limits)


def test_metrics_level_shape_identity_and_color_normalization():
    rng=np.random.default_rng(17);x=rng.normal(size=(3,15,40));mask=np.ones_like(x,bool);mask[:,-1,-2:]=False
    batch=dict(x=x,s=x*.6,o=x*.4,mask=mask,**{k:np.full_like(x,2.) for k in ('vx','vs','vo')})
    pred=dict(sum=x+2,S=x*.6+1,O=x*.4+1)
    m=base_training_metrics(pred,batch,2)
    np.testing.assert_allclose(m['nmse'],np.tile([2.,.5,.5],(3,2,1)).transpose(0,2,1))
    np.testing.assert_allclose(m['level_nmse']+m['shape_nmse'],m['nmse'],atol=1e-12)
    np.testing.assert_allclose(m['shape_nmse'],0.,atol=1e-12)
    np.testing.assert_allclose(m['correlation'],1.,atol=1e-12)


def test_aggregation_keeps_subjects_records_windows_and_colors_equal():
    rows=[dict(dataset='d',subject='a',record='r1',parent='p1'),dict(dataset='d',subject='a',record='r2',parent='p2'),
          dict(dataset='d',subject='b',record='r',parent='p3')]
    examples=[(0,0,0),(0,0,1),(0,30,0),(1,0,0),(2,0,0)]
    values=np.array([0.,2.,3.,8.,20.])[:,None,None]*np.ones((5,3,2))
    result=base_training_aggregate(dict(nmse=values),examples,rows)
    # r1: mean(mean(0,2),3)=2; a:mean(2,8)=5; dataset:mean(5,20)=12.5.
    assert result['selection_score']==pytest.approx(12.5)
    assert result['subjects']==2 and result['records']==3 and result['windows']==4


def test_budget_rule_keeps_failed_models_and_checks_component_regressions():
    def model(i):
        return dict(status='completed',task_id=str(i),milestones={str(s):dict(validation=dict(metrics=dict(nmse=dict(macro=np.full((3,1),v).tolist())))) for s,v in [(4000,1.1),(7000,1.01),(10000,1.)]})
    models=[model(i) for i in range(5)]
    result=base_training_budget(models,[4000,7000,10000],.02,.05)
    assert result['budget']==7000
    models[0]['milestones']['7000']['validation']['metrics']['nmse']['macro']=[[.9],[1.06],[1.]]
    assert base_training_budget(models,[4000,7000,10000],.02,.05)['budget']==10000
    models[0]['status']='failed'
    assert base_training_budget(models,[4000,7000,10000],.02,.05)['budget'] is None


def test_matched_warmup_full_training_evaluation_and_completed_resume(tmp_path):
    cfg,out=prepared(tmp_path)
    warm=dict(modality='hb',architecture='waveform',candidate='warmup')
    a=runner.train_base_tokenizer(cfg,tmp_path,out,warm,'cpu')
    assert a['completed_step']==2 and not a['milestones']
    for candidate in ('continuous','vq128'):
        task=dict(warm,candidate=candidate)
        result=runner.train_base_tokenizer(cfg,tmp_path,out,task,'cpu')
        assert result['completed_step']==8 and set(result['milestones'])=={'4','6','8'}
        assert result['selected_step'] in (4,6,8)
        for point in result['milestones'].values():
            assert point['validation']['subjects']==1
            if candidate=='vq128':assert point['integer_replay_max_abs']<2e-6
        assert runner.train_base_tokenizer(cfg,tmp_path,out,task,'cpu')==result


def test_interrupted_milestone_resumes_exact_updates(tmp_path,monkeypatch):
    cfg,out=prepared(tmp_path);warm=dict(modality='hb',architecture='dense_long',candidate='warmup')
    runner.train_base_tokenizer(cfg,tmp_path,out,warm,'cpu')
    task=dict(warm,candidate='vq128');original=runner.evaluate_base_training
    def interrupt(*args,**kwargs):raise RuntimeError('synthetic interruption after checkpoint')
    monkeypatch.setattr(runner,'evaluate_base_training',interrupt)
    with pytest.raises(RuntimeError,match='synthetic interruption'):runner.train_base_tokenizer(cfg,tmp_path,out,task,'cpu')
    saved=torch.load(out/'models/hb__dense_long__vq128/checkpoint.pt',weights_only=False)
    assert saved['step']==4
    monkeypatch.setattr(runner,'evaluate_base_training',original)
    resumed=runner.train_base_tokenizer(cfg,tmp_path,out,task,'cpu')
    other=tmp_path/'uninterrupted'
    runner.train_base_tokenizer(cfg,tmp_path,other,warm,'cpu')
    direct=runner.train_base_tokenizer(cfg,tmp_path,other,task,'cpu')
    aa=torch.load(out/'models/hb__dense_long__vq128/checkpoint.pt',weights_only=False)
    bb=torch.load(other/'models/hb__dense_long__vq128/checkpoint.pt',weights_only=False)
    for key in aa['model_state']:torch.testing.assert_close(aa['model_state'][key],bb['model_state'][key],atol=0,rtol=0)
    assert resumed['milestones']['8']['validation']['selection_score']==direct['milestones']['8']['validation']['selection_score']


def test_all_five_synthetic_integer_and_gradient_interfaces():
    assert runner.base_training_synthetic(runner.read_config(CONFIG))['status']=='passed'


def test_development_subject_subset_removes_all_other_validation_inputs(tmp_path):
    cfg, _ = prepared(tmp_path)
    path = tmp_path/'g0/cache/manifest.json'
    manifest = runner.read_json(path)
    excluded = deepcopy(next(r for r in manifest['rows'] if r['split'] == 'validation'))
    excluded['subject'] = 'reserved_synthetic_subject'
    excluded['parent'] += '_reserved'
    excluded['folder'] = str(path.parent/runner.identity(excluded['parent'])/excluded['modality'])
    manifest['rows'].append(excluded)
    runner.write_json(path, manifest)
    full = runner._base_training_data(cfg, tmp_path)[0]['rows']
    validation = sorted({(r['dataset'], r['subject']) for r in full if r['split'] == 'validation'})
    assert len(validation) > 1
    cfg['data']['validation_subjects'] = [list(validation[0])]
    subset = runner._base_training_data(cfg, tmp_path)[0]['rows']
    for original, restricted in zip(full, subset, strict=True):
        if original['split'] == 'train' or (original['dataset'], original['subject']) == validation[0]:
            assert restricted['inputs'] == original['inputs']
        else:
            assert not any(restricted['inputs'].values())
    cfg['data']['validation_subjects'] = [['not-a-library', 'unregistered']]
    with pytest.raises(ValueError, match='outside the G0 validation partition'):
        runner._base_training_data(cfg, tmp_path)


def test_warm_mapping_uses_open_file_and_revalidates_on_reopen(tmp_path):
    cfg,_ = prepared(tmp_path)
    manifest,scales,audit,_ = runner._base_training_data(cfg,tmp_path)
    rows = manifest['rows']
    first = audit['eeg']['initialization'][:1]
    cache = OrderedDict()
    def read():
        return base_tokenizer_batch(rows,first,scales,array_cache=cache,cache_limit=1)
    original = read()
    path = Path(rows[first[0][0]]['folder'])/'x.npy'
    outside = tmp_path/'relocated.npy'
    path.rename(outside)
    path.symlink_to(outside)
    for key, value in read().items():
        np.testing.assert_array_equal(value,original[key])
    cache.clear()
    with pytest.raises(ValueError,match='escapes'):
        read()
