"""G0 coverage and read-boundary checks use disposable, unprotected fixtures."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.dual_component_tokenizer import (
    base_tokenizer_plan, base_tokenizer_coordinates, base_tokenizer_scales,
    base_tokenizer_sampling_index, base_tokenizer_sample, base_tokenizer_batch,
    masked_state_window,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('base_g0_runner', ROOT/'experiments/scripts/run_dual_component_tokenizer.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
CONFIG = ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g0_v1.yaml'


def fixture(tmp_path):
    cfg = deepcopy(runner.read_config(CONFIG))
    cfg['data']['datasets'] = ['refed']
    cfg['sampling']['audit_draws'] = 1000
    root = tmp_path/'source'; source = tmp_path/'ssm'
    windows = []
    for subject,split,record,origin in [('a','train','r1',0),('a','train','r1',30),
            ('a','train','r2',0),('b','train','r1',0),('c','validation','r1',0)]:
        key=f'refed/{subject}/{record}/w{origin}'
        path=root/'windows'/key;path.parent.mkdir(parents=True,exist_ok=True)
        src=dict(key=key,dataset='refed',subject=subject,split=split,rate=1000,native_record=record,
                 start_sample=origin*1000,samples=60000,duration_s=60)
        meta=dict(schema='physiology_so_supervision_v2',**{k:src[k] for k in ('key','dataset','subject','split')},
                  source=src,status='completed',EEG_channels=['e1','e2'],fNIRS_positions=['h1','failed'],
                  array_file=str(path.with_suffix('.npz')))
        arrays={}
        for prefix,rate in [('eeg',1000),('fnirs',10)]:
            t=np.arange(60*rate)/rate
            x=np.sin(2*np.pi*(11 if prefix=='eeg' else .1)*t)[:,None]*np.array([1.,2.])[None]
            if prefix=='fnirs':x=np.repeat(x[...,None],2,-1)
            for k,v in dict(x=x,s=x*.7,o=x*.3,observation_valid=np.ones_like(x,bool),teacher_valid=np.ones_like(x,bool)).items():
                arrays[prefix+'_'+k]=v
            arrays[prefix+'_time_s']=t
            arrays['eeg_source_scoring_mask' if prefix=='eeg' else 'source_scoring_mask']=np.r_[np.ones(len(t)-rate//10,bool),np.zeros(rate//10,bool)]
        arrays['fnirs_teacher_valid'][:,1]=False
        arrays['fnirs_s'][:,1]=np.nan;arrays['fnirs_o'][:,1]=np.nan
        np.savez(path.with_suffix('.npz'),**arrays)
        runner.write_json(path.with_suffix('.json'),meta)
        windows.append(dict(key=key,dataset='refed',subject=subject,split=split,metadata=str(path.with_suffix('.json'))))
    windows.append(dict(key='forbidden',dataset='refed',subject='test',split='test',metadata='/must/not/read'))
    runner.write_json(root/'cache_manifest.json',dict(schema='physiology_so_supervision_v2',status='completed',source_ssm=str(source),windows=windows))
    runner.write_json(root/'verification.json',dict(status='passed'))
    regression=[dict(modality='eeg',split='train',dataset='refed',subject='a',record='r1',
                     parent=windows[0]['key'],holdout_starts_s=[30],native_start_sample=0,native_rate_hz=1000)]
    return cfg,root,source,regression


def test_plan_is_array_free_preserves_split_and_excludes_overlapping_regression(tmp_path,monkeypatch):
    cfg,root,source,regression=fixture(tmp_path)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('metadata plan opened an array'))
    plan=base_tokenizer_plan(root,cfg['data'],source,regression)
    assert len(plan['subjects'])==3 and len(plan['windows'])==5
    assert plan['regression_inputs']==1 and plan['regression_crops_excluded']>0
    for w in plan['windows']:
        src=w['metadata']['source']
        if w['row']['subject']=='a' and src['native_record']=='r1':
            for start in w['train_starts_s']+w['score_starts_s']:
                lo=src['start_sample']/1000+start
                assert lo+30<=30 or lo>=60
        if w['row']['split']=='validation':assert not w['train_starts_s']
    bad=deepcopy(cfg['data']);bad['allowed_splits'].append('test')
    with pytest.raises(ValueError,match='boundary'):
        base_tokenizer_plan(root,bad,source,regression)


def test_duplicate_parent_inputs_have_one_owner_and_scoring_is_disjoint(tmp_path):
    cfg,root,source,_=fixture(tmp_path)
    plan=base_tokenizer_plan(root,cfg['data'],source,[])
    assert plan['duplicate_training_crops_removed']>0
    support={};training=set()
    for w in plan['windows']:
        src=w['metadata']['source'];record=(w['row']['subject'],src['native_record'])
        for start in w['train_starts_s']:
            key=(*record,src['start_sample']+start*1000)
            assert key not in training;training.add(key)
        for start in w['score_starts_s']:
            begin=src['start_sample']+start*1000
            assert all(begin+30000<=a or begin>=b for a,b in support.get(record,[]))
            support.setdefault(record,[]).append((begin,begin+30000))


def test_identity_and_test_rejected_before_array_access(tmp_path,monkeypatch):
    cfg,root,source,regression=fixture(tmp_path)
    plan=base_tokenizer_plan(root,cfg['data'],source,regression);chosen=plan['windows'][0]
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('boundary failed'))
    bad=deepcopy(chosen);bad['row']['split']='test'
    with pytest.raises(ValueError,match='forbidden split'):masked_state_window(root,bad,cfg['data'])
    path=Path(chosen['row']['metadata']);meta=json.loads(path.read_text());meta['array_file']='/forbidden.npz';path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='escapes'):masked_state_window(root,chosen,cfg['data'])


def test_cross_partition_subject_rejected_without_arrays(tmp_path,monkeypatch):
    cfg,root,source,regression=fixture(tmp_path)
    path=root/'cache_manifest.json';m=json.loads(path.read_text());m['windows'][-1]['subject']='a';path.write_text(json.dumps(m))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('boundary failed'))
    with pytest.raises(ValueError,match='crosses'):base_tokenizer_plan(root,cfg['data'],source,regression)


def test_native_200hz_one_sample_parent_offset_keeps_exact_regression_support(tmp_path):
    cfg,root,source,regression=fixture(tmp_path)
    cfg['data']['native_rate_hz']['refed']=200
    for row in json.loads((root/'cache_manifest.json').read_text())['windows']:
        if row['split']=='test':continue
        path=Path(row['metadata']);meta=json.loads(path.read_text())
        meta['source'].update(rate=200,samples=12000,start_sample=meta['source']['start_sample']//5)
        if meta['key']=='refed/a/r1/w0':meta['source']['start_sample']=1
        path.write_text(json.dumps(meta))
    regression[0].update(native_start_sample=1,native_rate_hz=200)
    plan=base_tokenizer_plan(root,cfg['data'],source,regression)
    second=next(w for w in plan['windows'] if w['row']['key']=='refed/a/r1/w30')
    assert second['blocked_samples']==[[1,6001]]
    assert second['train_starts_s']==[]  # 60 s parent has no crop starting after 30.005 s.


def test_regression_perturbation_cannot_change_training_arrays_or_statistics(tmp_path):
    cfg,root,source,regression=fixture(tmp_path)
    plan=base_tokenizer_plan(root,cfg['data'],source,regression);chosen=plan['windows'][0]
    data=masked_state_window(root,chosen,cfg['data'])
    r,a=base_tokenizer_coordinates(data['metadata'],data['arrays'],'eeg',chosen)
    changed=deepcopy(data['arrays'])
    for k,f in [('x',1),('s',.7),('o',.3)]:changed['eeg_'+k][30000:]+=1000*f
    r2,a2=base_tokenizer_coordinates(data['metadata'],changed,'eeg',chosen)
    assert r['moments']==r2['moments']
    np.testing.assert_array_equal(a['x'][:6000],a2['x'][:6000])
    assert not a['source_scoring'][6000]


def test_equal_subject_record_moments_not_window_count():
    def row(subject,record,n,mean):
        moments={k:dict(count=[[n]],sum=[[n*mean]],second=[[n*(mean**2+1)]]) for k in ('x','s','o','dx','ds','do')}
        return dict(split='train',modality='eeg',dataset='d',subject=subject,record=record,positions=['p'],moments=moments)
    rows=[row('a','r1',100,0),row('a','r2',1,2),row('b','r1',1,10)]
    spec=dict(target_variance_floor=.01,difference_variance_floor_fraction=.0001)
    scales=base_tokenizer_scales(rows,spec)
    # record-equal within a => mean=1, second=3; b => mean=10, second=101.
    expected=52-5.5**2
    assert scales['positions']['eeg']['d']['p']['scale']==pytest.approx(np.sqrt(expected))
    rows.append(dict(split='validation',moments={}))
    assert base_tokenizer_scales(rows,spec)==scales
    rows[-1]['moments']={'x':{}}
    with pytest.raises(ValueError,match='validation'):base_tokenizer_scales(rows,spec)


def test_sampler_equal_hierarchy_and_initialization_covers_every_record():
    rows=[]
    for d,subjects in [('d1',['a','b']),('d2',['c'])]:
        for s in subjects:
            for r in (['r1','r2','r3'] if s=='a' else ['r1']):
                for parent in range(5 if r=='r1' else 1):
                    rows.append(dict(modality='eeg',split='train',dataset=d,subject=s,record=r,
                                     inputs={'train':[dict(start_s=0,positions=[0,1])]}))
    groups=base_tokenizer_sampling_index(rows,'eeg')
    sample=base_tokenizer_sample(np.random.default_rng(17),groups,60000)
    ds=np.mean([rows[i]['dataset']=='d1' for i,_,_ in sample])
    a=np.mean([rows[i]['subject']=='a' for i,_,_ in sample if rows[i]['dataset']=='d1'])
    r1=np.mean([rows[i]['record']=='r1' for i,_,_ in sample if rows[i]['subject']=='a'])
    assert abs(ds-.5)<.015 and abs(a-.5)<.015 and abs(r1-1/3)<.02
    init=base_tokenizer_sample(np.random.default_rng(17),groups,0,initialization=True)
    assert len({(rows[i]['dataset'],rows[i]['subject'],rows[i]['record']) for i,_,_ in init})==5


def test_prepared_pool_scales_masks_batch_and_resume(tmp_path):
    cfg,root,source,regression=fixture(tmp_path)
    plan=base_tokenizer_plan(root,cfg['data'],source,regression);out=tmp_path/'run'
    records=[runner._base_prepare_window((cfg,str(out),str(root),w,False)) for w in plan['windows']]
    assert all(r['status']=='completed' for r in records)
    rows,scales=runner._base_finalize(cfg,out,plan,records)
    verify=runner.verify_base_tokenizer(cfg,out,plan)
    assert verify['status']=='passed'
    assert verify['counts']['hb/refed/train']['teacher_unavailable']>0
    sample=base_tokenizer_sample(np.random.default_rng(17),base_tokenizer_sampling_index(rows,'eeg'),4)
    batch=base_tokenizer_batch(rows,sample,scales)
    assert batch['x'].shape==(4,15,400)
    assert np.isfinite(batch['x']).all()
    bad=deepcopy(cfg);bad['sampling']['seed']=19
    with pytest.raises(ValueError,match='resume identity'):
        runner._base_prepare_window((bad,str(out),str(root),plan['windows'][0],False))
    changed=deepcopy(scales);changed['positions']['eeg']['refed']['e1']['scale']*=2
    runner.write_json(out/'scales.json',changed)
    with pytest.raises(ValueError,match='contract changed'):runner.verify_base_tokenizer(cfg,out,plan)


def test_g0_synthetic_integer_interface_and_training_stage_rejection(tmp_path):
    cfg=runner.read_config(CONFIG)
    assert runner.base_tokenizer_synthetic(cfg)['status']=='passed'
    with pytest.raises(ValueError,match='preparation stages only'):
        runner.main(['--config',str(CONFIG),'--stage','state-pilot','--output',str(tmp_path)])


def test_retained_source_failure_keeps_hb_and_planned_denominator(tmp_path):
    cfg,root,source,regression=fixture(tmp_path)
    path=root/'windows/refed/a/r2/w0.json'
    meta=json.loads(path.read_text());meta.update(status='failed',EEG_available=False);path.write_text(json.dumps(meta))
    plan=base_tokenizer_plan(root,cfg['data'],source,regression);out=tmp_path/'run'
    records=[runner._base_prepare_window((cfg,str(out),str(root),w,False)) for w in plan['windows']]
    rejected=next(r for r in records if r['key']=='refed/a/r2/w0')
    assert rejected['status']=='completed' and not rejected['failures']
    assert rejected['unavailable'][0]['reason']=='retained_source_EEG_failure'
    assert [r['modality'] for r in rejected['rows']]==['hb']
    runner._base_finalize(cfg,out,plan,records)
    result=runner.verify_base_tokenizer(cfg,out,plan)
    assert result['status']=='passed'
    assert result['planned_counts']['eeg/refed/train']['parents']==4
    assert result['counts']['eeg/refed/train']['parents']==3
