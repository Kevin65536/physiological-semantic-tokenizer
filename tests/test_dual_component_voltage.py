"""Native-voltage conditional targets and same-rate Hb continuity, synthetic only."""
from pathlib import Path
import importlib.util
import json
import shutil
import subprocess
import sys
import numpy as np
import pytest
import torch
from src.data.dual_component_tokenizer import voltage_proxy_correction,BANDS
from src.inference.observation_baselines import eeg_band_power
from src.tokenizers.dual_component_tokenizer import waveform_windows,overlap_decode,WaveformDictionary,synthesize_component
from src.analysis.dual_component_tokenizer import voltage_case_metrics,voltage_aggregate,boundary_neighborhood_errors

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('voltage_runner',ROOT/'experiments/scripts/run_dual_component_tokenizer.py')
runner=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(runner)


@pytest.fixture(scope='module')
def synthetic_voltage():
    rng=np.random.default_rng(19);t=np.arange(18000)/200
    x=np.column_stack([np.sin(2*np.pi*f*t)*(1+.1*np.cos(.3*t+i))+.1*rng.normal(size=len(t)) for i,f in enumerate([7,11,17])])
    f=np.log(eeg_band_power(x,bands=BANDS,target_rate=10)).reshape(900,-1);f-=f[:50].mean(0)
    a=np.stack((np.linspace(-.2,.8,15),np.cos(np.arange(15))))*.02
    u=f@a.T;target=u+.002*np.sin(np.arange(900)[:,None]*.05)*np.array([1.,-.5])
    return x,a,target


def test_zero_correction_exact_identity_and_nonzero_proxy_replay(synthetic_voltage):
    x,a,target=synthetic_voltage
    zero=voltage_proxy_correction(x,a,target,strength=0)
    assert np.array_equal(zero['corrected'],x) and not zero['remaining'].any()
    result=voltage_proxy_correction(x,a,target)
    assert result['diagnostic']['status']=='corrected'
    assert result['diagnostic']['after'] < result['diagnostic']['before']
    assert np.max(np.abs(result['corrected']+result['remaining']-x))<1e-14
    assert np.array_equal(result['corrected'][:1000],x[:1000])
    assert result['diagnostic']['maximum_log_gain']<=.7


def test_duplicate_roi_constraint_does_not_double_voltage_correction(synthetic_voltage):
    x,a,target=synthetic_voltage
    one=voltage_proxy_correction(x,a[:1],target[:,:1])
    duplicate=voltage_proxy_correction(x,np.repeat(a[:1],2,axis=0),np.repeat(target[:,:1],2,axis=1))
    assert np.allclose(one['corrected'],duplicate['corrected'],rtol=1e-10,atol=1e-10)


def test_proxy_correction_rejects_nonfinite_or_unregistered_support(synthetic_voltage):
    x,a,target=synthetic_voltage
    with pytest.raises(ValueError):voltage_proxy_correction(x[:-1],a,target)
    with pytest.raises(ValueError):voltage_proxy_correction(x,a,np.full_like(target,np.nan))


@pytest.mark.parametrize('span',[20,40,60])
def test_overlap_add_identity_gradient_and_component_independence(span):
    torch.manual_seed(5);x=torch.randn(2,200,2,dtype=torch.float64)
    windows,mask=waveform_windows(x,hop=20,span=span)
    words=windows.clone().requires_grad_();decoded=overlap_decode(words,hop=20,channels=2,length=200)
    assert torch.allclose(decoded,x,atol=1e-12)
    assert np.allclose(overlap_decode(windows.numpy(),hop=20,channels=2,length=200),x.numpy())
    decoded.square().mean().backward();assert torch.isfinite(words.grad).all()
    if span>20:assert not mask[0,0,:((span-20)//2)*2].any()
    unchanged=overlap_decode(words.detach(),hop=20,channels=2,length=200)
    other=windows+100
    assert torch.equal(unchanged,overlap_decode(words.detach(),hop=20,channels=2,length=200))
    assert not torch.equal(unchanged,overlap_decode(other,hop=20,channels=2,length=200))


def test_overlap_reduces_independent_patch_offsets_without_raw_blending():
    rng=np.random.default_rng(17);offset=rng.normal(size=(1,10,1))
    blocks=np.repeat(offset,20,axis=2);extended=np.repeat(offset,40,axis=2)
    first=overlap_decode(blocks,hop=20,channels=1,length=200)
    second=overlap_decode(extended,hop=20,channels=1,length=200)
    seam=np.arange(19,199,20)
    assert np.mean(np.diff(second,axis=1)[:,seam]**2)<.1*np.mean(np.diff(first,axis=1)[:,seam]**2)


def test_boundary_metric_compares_signed_differences_and_records_equal_weights():
    target=np.tile(np.arange(200)[None,:,None]/200,(3,1,1))
    reverse=-target;values,signals=voltage_case_metrics(dict(sum=reverse),dict(S=target,O=target,sum=target),hop=20)
    assert np.allclose(values[:,2,0,2],.01**2)
    rows=[dict(dataset='d',split='validation',subject='s',record=r) for r in ['a','a','b']]
    values[:,2,0,0]=[1,1,5]
    subjects,summary=voltage_aggregate(values,rows,modality='eeg',variant='fixture',signals=signals)
    assert next(r['value'] for r in summary if r['component']=='sum' and r['metric']=='NMSE')==3


def test_waveform_dictionary_reload_exact():
    rng=np.random.default_rng(4);x=rng.normal(size=(100,20)).astype('float32')
    book=WaveformDictionary(size=8,levels=2,max_iter=5).fit(x)
    loaded=WaveformDictionary.from_state(book.state());ids=loaded.encode(x)
    assert np.array_equal(ids,book.encode(x)) and np.array_equal(loaded.decode(ids),book.decode(ids))


def _fixture_cache(path):
    cache=path/'voltage_cache';cache.mkdir();rng=np.random.default_rng(4)
    rows=[dict(dataset='d',subject=f's{i}',record='r',split='train' if i<8 else 'validation',
        coordinate='d|c',coordinate_index=0,id=f'case{i}',trial_id=f'trial{i}') for i in range(10)]
    arrays={};statistics={}
    for m,length,c in [('eeg',4000,1),('hb',200,2)]:
        x=rng.normal(size=(10,length,c)).astype('float32')
        for suffix,values in [('u',x),('s',.8*x),('o',.2*x)]:
            name=m+'_'+suffix;np.save(cache/(name+'.npy'),values);arrays[name]=name+'.npy'
            statistics[name]=dict(variance=1.,derivative_mse=1.,mean=0.)
    runner.write_json(cache/'manifest.json',dict(schema='dual_component_voltage_cache_v1',protected_data='forbidden',
        rows={'eeg':rows,'hb':rows},coordinates={'eeg':['d|c'],'hb':['d|c']},arrays=arrays,statistics=statistics))


def test_native_forward_requires_no_teacher_or_other_modality_arrays(tmp_path):
    _fixture_cache(tmp_path);cfg=runner.read_config(ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_v1.yaml')
    manifest,arrays=runner.load_voltage_cache(tmp_path);candidate=cfg['voltage']['candidates'][0]
    only={'eeg_u':arrays['eeg_u']}
    a,mask,targets,_=runner._voltage_batch(manifest,only,'eeg',[0,1],candidate,cfg,'cpu',positions=[20,30],load_targets=False)
    assert a.shape==(2,1,401) and not targets and mask.all()


@pytest.mark.parametrize('name',['eeg_voltage','hb_overlap'])
def test_fixed_budget_native_training_and_integer_evaluation(tmp_path,name):
    _fixture_cache(tmp_path);cfg=runner.read_config(ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_v1.yaml')
    cfg['voltage'].update(steps=2,checkpoint_every=1,width=8,latent_dim=4,codebook_size=8,codebook_fit_patches=64,codebook_iterations=5)
    cfg['voltage']['eeg'].update(batch_size=4,codebook_levels=[1,2]);cfg['voltage']['hb'].update(batch_size=2,codebook_levels=[1])
    cfg['resources']['torch_threads']=1
    result=runner.voltage_train(cfg,tmp_path,name,17,'cpu')
    assert result['status']=='completed' and result['step']==result['budget']==2
    assert not result['test_used'] and result['evaluation']['cases']==10
    again=runner.voltage_train(cfg,tmp_path,name,17,'cpu');assert again==runner.read_json(tmp_path/f'voltage_jobs/{name}/seed_17/summary.json')
    shutil.copytree(tmp_path/'voltage_cache',tmp_path/'voltage_test_cache')
    manifest=runner.read_json(tmp_path/'voltage_test_cache/manifest.json')
    for rows in manifest['rows'].values():
        for row in rows:row['split']='test'
    runner.write_json(tmp_path/'voltage_test_cache/manifest.json',manifest)
    runner.write_json(tmp_path/'voltage_selection.json',dict(hb_candidate='hb_overlap'))
    final=runner.voltage_final(cfg,tmp_path,name,17,'cpu')
    assert final['status']=='completed' and final['evaluation']['final']


def test_test_evaluation_rejects_unfrozen_choice_before_cache_read(tmp_path,monkeypatch):
    cfg=runner.read_config(ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_v1.yaml')
    monkeypatch.setattr(runner,'load_voltage_cache',lambda *a,**k:pytest.fail('test cache read before choice'))
    with pytest.raises(ValueError,match='validation selection'):runner.voltage_final(cfg,tmp_path,'eeg_voltage',17,'cpu')
    runner.write_json(tmp_path/'voltage_selection.json',dict(hb_candidate='hb_overlap'))
    with pytest.raises(ValueError,match='unselected'):runner.voltage_final(cfg,tmp_path,'hb_boundary',17,'cpu')


def test_cli_check_runs_once_without_signal_access():
    result=subprocess.run([sys.executable,str(ROOT/'experiments/scripts/run_dual_component_tokenizer.py'),
        '--config',str(ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_v1.yaml'),
        '--stage','check'],capture_output=True,text=True,check=True)
    record=json.loads(result.stdout)
    assert record['status']=='voltage_contract_valid' and record['measured_arrays_read']==0


def test_fixed_synthesis_preserves_constant_and_linear_component_sum():
    from scipy.ndimage import gaussian_filter1d
    rng=np.random.default_rng(9);a=rng.normal(size=(2,10,40));b=rng.normal(size=a.shape)
    decode=lambda x:synthesize_component(x,hop=20,channels=2,length=200,smoothing_sigma=1.)
    assert np.allclose(decode(a+b),decode(a)+decode(b))
    assert np.array_equal(decode(np.ones_like(a)),np.ones((2,200,2)))
    expected=gaussian_filter1d(overlap_decode(a,hop=20,channels=2,length=200),sigma=1.,axis=1,mode='nearest',truncate=4.)
    assert np.array_equal(decode(a),expected)


def test_synthesis_selection_uses_validation_and_retains_fidelity_condition():
    cfg={'voltage':{'decoder_synthesis_sigma_samples':1.,'minimum_boundary_improvement':.3}}
    def variant(mse,boundary):
        rows=[]
        for split in ['validation','test']:
            for metric,value in [('NMSE',mse),('boundary_error_NMSE',boundary)]:
                rows.append(dict(dataset='d',component='S',signal='joint',split=split,metric=metric,value=value if split=='validation' else 1e9))
        return dict(metrics=rows,health_pass=True)
    result=dict(candidate={'modality':'hb','name':'hb_blocks'},evaluation={'variants':{'vq2':variant(1,1),'vq2_smoothed':variant(.99,.15)}})
    choice=runner.select_voltage(cfg,[result]);assert choice['hb_variant']=='vq2_smoothed' and choice['boundary_goal_met']
    result['evaluation']['variants']['vq2_smoothed']=variant(1.1,.01)
    choice=runner.select_voltage(cfg,[result]);assert choice['hb_variant']=='vq2' and not choice['boundary_goal_met']


def test_boundary_neighborhood_does_not_mistake_ramp_for_removed_offset():
    from scipy.ndimage import gaussian_filter1d
    x=np.zeros((1,200,1));x[:,100:]=1.;target=np.zeros_like(x)
    smooth=gaussian_filter1d(x,sigma=1.,axis=1,mode='nearest',truncate=4.)
    before=boundary_neighborhood_errors(x,target,hop=100,spans=(1,11))
    after=boundary_neighborhood_errors(smooth,target,hop=100,spans=(1,11))
    assert after[0,0]<.2*before[0,0]
    assert np.isclose(after[0,1],before[0,1])
