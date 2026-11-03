"""EEG-O refinement: synthetic and temporary non-authoritative fixtures only."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest
import torch

from src.data.dual_component_tokenizer import load_eeg_residual_development, eeg_residual_band_features, load_retained_eeg_projection
from src.tokenizers.dual_component_tokenizer import BandGainResidualPredictor, ProjectionGainResidualPredictor
from src.analysis.dual_component_tokenizer import eeg_residual_case_metrics, eeg_residual_aggregate

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('eeg_o_runner',ROOT/'experiments/scripts/run_dual_component_tokenizer.py')
runner=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(runner)
CONFIG=ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_eeg_o_v1.yaml'


@pytest.fixture
def residual_cache(tmp_path):
    parent=tmp_path/'parent';cache=parent/'voltage_cache';cache.mkdir(parents=True)
    rows=[];coordinates=[]
    for dataset in ('eeg_fnirs_single_trial','simultaneous_eeg_nirs'):
        coordinates.extend(f'{dataset}|c{i}' for i in range(16))
        for split in ('train','validation'):
            for channel in range(16):
                coordinate=f'{dataset}|c{channel}'
                rows.append(dict(dataset=dataset,subject=split,record='r',split=split,
                    trial_id=f'{dataset}_{split}',id=f'{dataset}_{split}_{channel}',channel=f'c{channel}',
                    coordinate=coordinate,coordinate_index=coordinates.index(coordinate)))
    t=np.arange(4000)/200
    raw=np.asarray([np.sin(2*np.pi*2*t+i/10)+.2*np.sin(2*np.pi*8*t) for i in range(len(rows))],dtype='float32')[:,:,None]
    arrays={'eeg_u':raw,'eeg_s':.9*raw,'eeg_o':.1*raw}
    manifest=dict(schema='dual_component_voltage_cache_v1',protected_data='forbidden',final=False,
        rows={'eeg':rows},coordinates={'eeg':coordinates},arrays={k:k+'.npy' for k in arrays},
        shapes={k:list(v.shape) for k,v in arrays.items()},planned_trials=4,retained_trials=4,
        statistics={k:dict(variance=float(np.var(v)),mean=float(np.mean(v)),
            derivative_mse=float(np.mean(np.diff(v,axis=1)**2))) for k,v in arrays.items()})
    for key,value in arrays.items():np.save(cache/(key+'.npy'),value)
    runner.write_json(cache/'manifest.json',manifest);runner.write_json(parent/'voltage_summary.json',dict(status='completed'))
    return parent


def test_development_reader_uses_only_eeg_arrays(residual_cache):
    manifest,arrays,trials=load_eeg_residual_development(residual_cache)
    assert set(arrays)=={'eeg_u','eeg_s','eeg_o'}
    assert len(trials)==4 and all(len(t['indices'])==16 for t in trials)
    assert set(t['split'] for t in trials)=={'train','validation'}


@pytest.mark.parametrize('violation',['test','protected','subject','path'])
def test_development_boundary_fails_before_array_open(residual_cache,monkeypatch,violation):
    path=residual_cache/'voltage_cache/manifest.json';manifest=runner.read_json(path)
    if violation=='test':manifest['rows']['eeg'][0]['split']='test'
    if violation=='protected':manifest['protected_data']='allowed'
    if violation=='subject':manifest['rows']['eeg'][0]['split']='validation'
    if violation=='path':manifest['arrays']['eeg_u']='../voltage_test_cache/eeg_u.npy'
    runner.write_json(path,manifest)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('array opened before boundary rejection'))
    with pytest.raises(ValueError):load_eeg_residual_development(residual_cache)


def test_observed_only_band_features_have_declared_shapes_and_do_not_mutate_input():
    t=np.arange(4000)/200;x=np.stack([np.sin(2*np.pi*2*t),np.sin(2*np.pi*20*t)]).astype('float32')
    original=x.copy();bands,power=eeg_residual_band_features(x)
    assert bands.shape==(2,4000,5) and power.shape==(2,200,5)
    assert np.array_equal(x,original) and np.isfinite(power).all()
    assert np.mean(bands[0,:,0]**2)>20*np.mean(bands[0,:,3]**2)
    with pytest.raises(TypeError):eeg_residual_band_features(x,target=x)
    with pytest.raises(ValueError):eeg_residual_band_features(np.full_like(x,np.nan))


@pytest.mark.parametrize('joint',[False,True])
def test_gain_model_zero_identity_known_gain_and_gradient(joint):
    torch.manual_seed(17);channels=3
    model=BandGainResidualPredictor(np.zeros((channels,5)),np.ones((channels,5)),
        channels=channels,width=8,context_bins=19,joint=joint)
    carriers=torch.randn(2,channels,400,5);features=torch.randn(2,channels,20,5)
    coordinates=torch.arange(channels)[None].expand(2,-1)
    assert torch.count_nonzero(model(carriers,features,coordinates)['O'])==0
    with torch.no_grad():model.network[-1].bias[::5]=float(np.arctanh(np.log(.8)/.7))
    result=model(carriers,features,coordinates)
    assert torch.allclose(result['O'],.2*carriers[...,0],atol=5e-7)
    result['O'].square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert result['log_gain'].abs().max()<=.7


def test_gain_context_and_single_electrode_separation():
    torch.manual_seed(9)
    model=BandGainResidualPredictor(np.zeros((2,5)),np.ones((2,5)),channels=2,width=8,context_bins=19)
    torch.nn.init.normal_(model.network[-1].weight,std=.2)
    carriers=torch.randn(1,2,2000,5);features=torch.randn(1,2,100,5);coordinates=torch.tensor([[0,1]])
    before=model(carriers,features,coordinates)
    changed=features.clone();changed[:,0,0]+=100
    after=model(carriers,changed,coordinates)
    assert torch.equal(before['O'][:,1],after['O'][:,1])
    assert torch.equal(before['log_gain'][:,:,50],after['log_gain'][:,:,50])
    assert not torch.equal(before['log_gain'][:,0,0],after['log_gain'][:,0,0])


def test_residual_metrics_expose_flat_prediction_and_equal_record_aggregation():
    t=np.arange(200)/20;truth=np.stack([np.sin(t),np.sin(t),3*np.sin(t)])
    flat=eeg_residual_case_metrics(np.zeros_like(truth),truth)
    rows=[dict(dataset='d',split='validation',subject='s',record=r) for r in ('a','a','b')]
    subjects,summary=eeg_residual_aggregate(flat,rows)
    assert summary[0]['zero_prediction_skill']==0 and summary[0]['RMS_ratio']==0
    assert summary[0]['centered_correlation'] is None and summary[0]['undefined_correlations']==3
    assert np.isclose(summary[0]['NMSE'],5*np.mean(np.sin(t)**2))
    good=eeg_residual_case_metrics(.8*truth,truth);_,summary=eeg_residual_aggregate(good,rows)
    assert np.isclose(summary[0]['zero_prediction_skill'],.96) and np.isclose(summary[0]['RMS_ratio'],.8)


def test_selection_cannot_read_test_and_uses_worst_dataset():
    def candidate(name,ratios):
        return dict(status='completed',candidate={'name':name},evaluation={'metrics':{'hard':[
            dict(dataset=str(i),split='validation',NMSE=v,zero_NMSE=1) for i,v in enumerate(ratios)]}})
    choice=runner.select_eeg_o([candidate('one',[.2,1.2]),candidate('both',[.8,.9])])
    assert choice['candidate']=='both' and not choice['test_used']
    bad=candidate('test',[.1]);bad['evaluation']['metrics']['hard'][0]['split']='test'
    with pytest.raises(ValueError,match='validation'):runner.select_eeg_o([bad])


def test_fixed_budget_training_resume_uses_temporary_development_fixture(tmp_path,residual_cache):
    cfg=runner.read_config(CONFIG);cfg['eeg_o'].update(parent_run=str(residual_cache),steps=2,batch_trials=1,
        checkpoint_every=1,width=4,derivative_weight=.01)
    cfg['resources']['torch_threads']=1
    out=tmp_path/'run';out.mkdir();runner.freeze_contract(out,cfg,tmp_path)
    manifest,arrays,trials=load_eeg_residual_development(residual_cache)
    folder=out/'eeg_o_features';folder.mkdir()
    bands,power=eeg_residual_band_features(arrays['eeg_u'][:,:,0]);np.save(folder/'bands.npy',bands);np.save(folder/'log_power.npy',power)
    runner.write_json(folder/'manifest.json',dict(source_parent=str(residual_cache),trials=trials,
        coordinates=manifest['coordinates']['eeg'],feature_mean=np.zeros((32,5)),feature_scale=np.ones((32,5))))
    result=runner.eeg_o_train(cfg,out,'single_short',17,'cpu',export=False)
    assert result['status']=='completed' and result['step']==result['budget']==2 and result['test_used'] is False
    assert runner.eeg_o_train(cfg,out,'single_short',17,'cpu',export=False)==result
    job=out/'eeg_o_jobs/single_short/seed_17';checkpoint=torch.load(job/'checkpoint.pt',weights_only=False)
    assert checkpoint['step']==2 and [r['step'] for r in checkpoint['history']]==[1,2]
    (job/'summary.json').unlink();cfg['eeg_o']['steps']=3
    with pytest.raises(ValueError,match='contract drift'):runner.eeg_o_train(cfg,out,'single_short',17,'cpu',export=False)


def test_synthetic_protocol_and_cli_test_rejection_before_data(monkeypatch):
    cfg=runner.read_config(CONFIG);result=runner.eeg_o_synthetic(cfg)
    assert result['measured_arrays_read']==0 and result['integer_reload_exact']
    monkeypatch.setattr(runner,'_eeg_o_data',lambda *a:pytest.fail('test request reached arrays'))
    with pytest.raises(ValueError,match='forbids test'):
        runner.main(['--config',str(CONFIG),'--stage','eeg-o-one','--evaluation-split','test'])


@pytest.mark.parametrize('architecture',['band_gain','projection_gain'])
def test_temporary_fixture_completes_quantization_validation_and_frozen_S_comparison(tmp_path,residual_cache,architecture):
    cfg=runner.read_config(CONFIG)
    cfg['voltage'].update(steps=2,checkpoint_every=1,width=8,latent_dim=4,codebook_size=8,
                         codebook_fit_patches=128,codebook_iterations=3)
    cfg['voltage']['eeg']['batch_size']=4
    cfg['eeg_o'].update(parent_run=str(residual_cache),steps=2,batch_trials=2,checkpoint_every=1,
                       width=4,codebook_size=8,codebook_levels=2,codebook_fit_patches=128,codebook_iterations=3)
    candidate='single_short'
    if architecture=='projection_gain':
        candidate='proxy_short';cfg['eeg_o'].update(architecture=architecture,candidates=[dict(name=candidate,context_bins=19)])
    cfg['resources']['torch_threads']=1
    runner.voltage_train(cfg,residual_cache,'eeg_voltage',17,'cpu')
    out=tmp_path/'complete';out.mkdir();runner.freeze_contract(out,cfg,tmp_path)
    manifest,arrays,trials=load_eeg_residual_development(residual_cache)
    folder=out/'eeg_o_features';folder.mkdir()
    bands,power=eeg_residual_band_features(arrays['eeg_u'][:,:,0])
    np.save(folder/'bands.npy',bands);np.save(folder/'log_power.npy',power)
    metadata=dict(source_parent=str(residual_cache),trials=trials,
        coordinates=manifest['coordinates']['eeg'],feature_mean=np.zeros((32,5)),feature_scale=np.ones((32,5)),
        raw_support='temporary fixture complete 20s')
    if architecture=='projection_gain':
        mapping=np.zeros((2,80,3));mapping[:,::5,0]=1
        metadata.update(proxy_mean=np.zeros((2,3)),proxy_scale=np.ones((2,3)),gain_map=mapping)
        np.save(folder/'observed_proxy.npy',np.zeros((4,200,3),dtype='float32'))
        np.save(folder/'proxy_dataset_ids.npy',np.asarray([0,0,1,1],dtype='int64'))
    runner.write_json(folder/'manifest.json',metadata)
    result=runner.eeg_o_train(cfg,out,candidate,17,'cpu')
    evaluation=result['evaluation']
    assert evaluation['status']=='completed' and evaluation['validation_cases']==32
    assert evaluation['integer_reload_max_abs']==0 and evaluation['test_used'] is False
    assert len(evaluation['metrics']['hard'])==2
    assert all(np.isfinite(r['frozen_S_plus_O_NMSE']) for r in evaluation['metrics']['hard'])


def test_observed_projection_reader_excludes_neighboring_teacher_and_Hb_fields(residual_cache,monkeypatch):
    manifest=runner.read_json(residual_cache/'voltage_cache/manifest.json');rows=manifest['rows']['eeg'][:16]
    meta=dict(trial_id=rows[0]['trial_id'],eeg_channels=[r['channel'] for r in rows],diagnostic={'native_proxy_replay_max_abs':0.})
    accessed=[]
    class Archive:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __getitem__(self,key):
            accessed.append(key)
            if key=='original_proxy':return np.zeros((200,3))
            if key=='projection':return np.eye(3,80)
            pytest.fail('teacher or Hb field entered observed-input reader')
    monkeypatch.setattr(np,'load',lambda *a,**k:Archive())
    proxy,mapping=load_retained_eeg_projection('fixture.npz',meta,rows)
    assert accessed==['original_proxy','projection'] and proxy.shape==(200,3) and mapping.shape==(80,3)
    meta['diagnostic']['native_proxy_replay_max_abs']=float('nan')
    accessed.clear()
    with pytest.raises(ValueError):load_retained_eeg_projection('fixture.npz',meta,rows)
    assert not accessed


def test_projection_gain_matches_known_subspace_and_has_finite_gradients():
    mapping=np.zeros((1,10,3),dtype='float32');mapping[0,::5,0]=1
    model=ProjectionGainResidualPredictor(np.zeros((1,3)),np.ones((1,3)),mapping,channels=2,width=8,context_bins=19)
    carriers=torch.randn(2,2,800,5);proxy=torch.zeros(2,40,3);ids=torch.zeros(2,dtype=torch.long)
    assert not model(carriers,proxy,ids)['O'].any()
    with torch.no_grad():model.network[-1].bias[0]=2*np.log(.8)
    result=model(carriers,proxy,ids)
    assert torch.allclose(result['O'],.2*carriers[...,0],atol=5e-7)
    result['O'].square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert result['log_gain'].abs().max()<=.7


def test_reference_aware_contract_synthetic():
    cfg=runner.read_config(CONFIG.with_name('dual_component_eeg_o_v2.yaml'))
    assert cfg['eeg_o']['architecture']=='projection_gain' and cfg['eeg_o']['test_access']=='forbidden'
    result=runner.eeg_o_synthetic(cfg)
    assert result['analytical_max_abs_error']<1e-6 and result['measured_arrays_read']==0


def test_final_comparison_pairs_subjects_before_seed_bootstrap(tmp_path):
    from src.analysis.dual_component_tokenizer import summarize_eeg_o_run,_csv
    parent=tmp_path/'parent';run=tmp_path/'run';run.mkdir();jobs=[]
    for seed in [17,29,43]:
        new=[];old=[]
        for subject,scale in [('a',1.),('b',3.)]:
            new.append(dict(dataset='eeg_fnirs_single_trial',split='validation',subject=subject,variant='hard',cases=2,records=1,
                NMSE=.5*scale,zero_NMSE=2*scale,prediction_power=scale,centered_correlation=.8,signed_projection=scale,
                boundary_error_NMSE=.01,derivative_error_NMSE=.01,frozen_S_plus_O_NMSE=.8*scale))
            for component,metric,value in [('O','NMSE',scale),('O','centered_correlation',.1),('sum','NMSE',scale)]:
                old.append(dict(dataset='eeg_fnirs_single_trial',split='validation',subject=subject,component=component,
                    signal='joint',metric=metric,value=value,cases=2))
        _csv(run/'eeg_o_jobs/proxy_full'/f'seed_{seed}/subjects.csv',new)
        _csv(parent/'voltage_jobs/eeg_voltage'/f'seed_{seed}/evaluation/vq4_subjects.csv',old)
        jobs.append(dict(candidate={'name':'proxy_full'},seed=seed,step=2,budget=2))
    runner.write_json(run/'eeg_o_summary.json',dict(status='completed',source_parent=str(parent),selection={'candidate':'proxy_full'},
        selected_results=jobs,planned_trials=4,retained_trials=4))
    runner.write_json(run/'eeg_o_screen_jobs.json',dict(jobs=jobs[:1]));runner.write_json(run/'eeg_o_replication_jobs.json',dict(jobs=jobs[1:]))
    runner.write_json(run/'resolved_config.json',dict(config={'eeg_o':{'raw_input_support':'fixture','architecture':'projection_gain'}}))
    summary=summarize_eeg_o_run(run,tmp_path/'analysis');row=summary['comparisons'][0]
    assert row['subjects']==2 and row['cases']==4
    assert row['NMSE']==1 and row['old_NMSE']==2 and row['relative_error_reduction_percent']==50
    assert row['paired_error']['subjects']==2 and row['paired_error']['mean_difference']==-1
    assert row['paired_error']['CI95']==[-1.5,-.5]
