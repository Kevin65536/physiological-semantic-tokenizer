"""Synthetic TK contracts; no ignored, measured or protected artifacts."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data.functional_tokenizer import subject_partitions,label_blind_donors
from src.tokenizers.functional_tokenizer import FunctionalTokenizer,ARMS
from src.analysis.functional_tokenizer import (
    read_hard_tokens,join_export,trial_features,budget_indices,classification_probe,
    align_ids,compare_arms,coupling_features,session_probe_split,
)
from src.teachers.functional_teacher import match_smoothing,apply_smoothing,synthetic_interventions

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'experiments/configs/physiology_semantic_tokenizer/functional_tokenizer_v1.yaml'


@pytest.fixture
def cfg(): return yaml.safe_load(CONFIG.read_text())


@pytest.fixture
def runner():
    spec=importlib.util.spec_from_file_location('functional_runner',ROOT/'experiments/scripts/run_functional_tokenizer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_contract_and_forbidden_scope_fail_closed(cfg,runner,tmp_path):
    assert runner.config(CONFIG)['training']['arms']==list(ARMS)
    for key,value in [('protected_data','allowed'),('labels_as_training_inputs',True)]:
        c=deepcopy(cfg);c['data'][key]=value;p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(c))
        with pytest.raises(ValueError):runner.config(p)
    from src.data.functional_tokenizer import inventory
    c=deepcopy(cfg);c['data']['datasets']=['croce_local_cache']
    with pytest.raises(ValueError):inventory(tmp_path,c)


def test_source_scope_checked_before_even_hash_reads(cfg,tmp_path,monkeypatch):
    from types import SimpleNamespace
    import src.data.clean_physiology_cache as registry
    from src.data.functional_tokenizer import prepare_native_record
    source=[dict(path=str(tmp_path/'protected_DO_NOT_OPEN'),sha256='unused')]
    record=SimpleNamespace(join_key='public',dataset_id='eeg_fnirs_single_trial',manifest=dict(source_files=source))
    monkeypatch.setattr(registry,'CleanPhysiologyCacheIndex',lambda _:SimpleNamespace(records=[record]))
    row=dict(join_key='public',dataset='eeg_fnirs_single_trial',split='train',source_identity=source)
    with pytest.raises(ValueError,match='outside the registered public dataset root'):
        next(prepare_native_record(tmp_path,cfg,[row]))


def test_subject_split_all_variants_stay_together():
    subjects={'a':[f's{i}' for i in range(10)],'b':[f's{i}' for i in range(15)]}
    a=subject_partitions(subjects,12);b=subject_partitions({k:v[::-1] for k,v in subjects.items()},12)
    assert a==b
    assert set(a.values())=={'train','validation','test'}
    assert len(a)==25


def test_shuffle_never_uses_labels_and_never_self_subject():
    rows=[dict(split='train',coordinate='d_roi',subject=f's{i//2}',task='a',condition=str(i%2)) for i in range(12)]
    one=label_blind_donors(rows,10)
    changed=[dict(r,task='POISON',condition='UNKNOWN') for r in rows]
    assert np.array_equal(one,label_blind_donors(changed,10))
    assert all(rows[i]['subject']!=rows[j]['subject'] for i,j in enumerate(one))


@pytest.mark.parametrize('arm',ARMS)
def test_all_arms_shape_capacity_and_finite_gradient(cfg,arm):
    torch.set_num_threads(1);torch.manual_seed(1)
    model=FunctionalTokenizer(cfg['model'],arm)
    e=torch.randn(2,10,2400);h=torch.randn(2,10,16)
    target=dict(eeg=torch.randn(2,10,8),hb=torch.randn(2,10,16))
    output=model(e,h);loss=model.objective(output,e,h,target,cfg['training'])['total']
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert output['eeg']['reconstruction'].shape==e.shape
    assert output['hb']['reconstruction'].shape==h.shape
    reference=FunctionalTokenizer(cfg['model'],'C0')
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in reference.parameters())


def test_stop_has_no_shared_gradient_path(cfg):
    model=FunctionalTokenizer(cfg['model'],'C-stop')
    e=torch.randn(2,10,2400);h=torch.randn(2,10,16)
    o=model(e,h);sum(x['reconstruction'].square().mean() for x in o.values()).backward()
    for branch in (model.eeg,model.hb):
        assert all(p.grad is None for p in branch.source_encoder.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in branch.observation_encoder.parameters())
    model.zero_grad();o=model(e,h);sum(x['teacher'].square().mean() for x in o.values()).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.eeg.source_encoder.parameters())


def test_hard_teacher_really_uses_lookup_and_encoders_are_modality_local(cfg):
    model=FunctionalTokenizer(cfg['model'],'QT');e=torch.randn(2,10,2400);h=torch.randn(2,10,16)
    model(e,h);model.eval()
    with torch.no_grad():
        a=model(e,h);b=model(e,h*500)
        assert torch.equal(a['eeg']['hard_ids'],b['eeg']['hard_ids'])
        for m in ('eeg','hb'):
            lookup=getattr(model,m).quantizer.get_embedding(a[m]['hard_ids'])
            assert torch.allclose(lookup,a[m]['semantic'],atol=1e-6)
            assert torch.allclose(getattr(model,m).teacher_head(lookup),a[m]['teacher'],atol=1e-6)


def test_hidden_nan_removed_before_encoder_and_coupling(cfg):
    model=FunctionalTokenizer(cfg['model'],'CT').eval()
    h=torch.randn(2,10,16);mask=torch.ones_like(h,dtype=torch.bool);mask[:,4:6]=False
    poisoned=h.clone();poisoned[~mask]=float('nan')
    assert torch.equal(model.hb(h,mask)['pre'],model.hb(poisoned,mask)['pre'])
    x=np.random.default_rng(1).normal(size=(2,80,2));p=x.copy();p[:,32:48]=np.nan
    assert np.array_equal(coupling_features(x,None)[0],coupling_features(p,None)[0])


def test_hard_reader_rejects_continuous_side_channel(tmp_path):
    values=dict(ids_eeg=np.zeros((2,10),dtype=int),ids_hb=np.ones((2,10),dtype=int),
        codebook_eeg=np.ones((128,16)),codebook_hb=np.ones((128,16)),identities=np.array(['b','a']))
    path=tmp_path/'hard.npz';np.savez(path,**values)
    ids,readout=read_hard_tokens(path)
    joined=join_export(['a','b'],ids,readout);assert joined['eeg'].shape==(2,10,16)
    np.savez(path,**values,confidence=np.ones(2))
    with pytest.raises(ValueError):read_hard_tokens(path)
    values['ids_hb']=values['ids_hb'].astype(float);np.savez(path,**values)
    with pytest.raises(ValueError):read_hard_tokens(path)


def test_regions_are_not_trials_and_budget_is_nested():
    rows=[dict(trial_id=f't{i}',split='train',anchor=dict(region=r)) for i in range(20) for r in ['left','right']]
    meta,x=trial_features(rows,np.arange(40)[:,None],['left','right'])
    assert x.shape==(20,2)
    small=budget_indices(meta,'train',.1,1);large=budget_indices(meta,'train',.25,1)
    assert len(small)==2 and len(large)==5 and set(small)<=set(large)
    with pytest.raises(ValueError):trial_features(rows[:-1],np.arange(39)[:,None],['left','right'])


def test_prototype_alignment_only_fits_training():
    ids=np.arange(4)[None].repeat(6,axis=0);permutation=np.array([2,0,3,1]);other=permutation[ids]
    training=np.array([True,True,True,False,False,False])
    aligned,mapping=align_ids(ids,other,training)
    assert np.array_equal(aligned,ids)
    changed=other.copy();changed[~training]=0
    assert np.array_equal(align_ids(ids,changed,training)[1],mapping)


def test_smoothing_train_fit_and_coordinate_shape():
    rng=np.random.default_rng(1);x=rng.normal(size=(4,40,3))
    selection=match_smoothing(x,x,[1,2,4])
    assert apply_smoothing(x,selection).shape==x.shape
    assert len(selection['sigmas'])==3


def test_seed_is_not_biological_replicate(cfg):
    rows=[]
    for arm,score in [('QT',.7),('Q0',.6)]:
        for subject in ('s1','s2'):
            for seed in cfg['training']['seeds']:
                rows.append(dict(arm=arm,subject=subject,seed=seed,path='hard',task='motor_imagery',
                    representation='S',modality='both',label_fraction=1.,balanced_accuracy=score))
    effect=next(r for r in compare_arms(rows,cfg) if r['comparison']=='teacher_discrete'
        and r['task']=='motor_imagery' and r['representation']=='S' and r['modality']=='both' and r['label_fraction']==1.)
    assert effect['subjects']==2 and effect['paired_seed_subjects']==6
    assert effect['mean']==pytest.approx(.1)


def test_native_session_split_uses_only_outer_heldout_subjects():
    rows=[dict(dataset='d',subject='s',task='t',split='test',session=str(s),eeg_start_s=s*100,
        trial_id=f'{s}_{i}') for s in range(3) for i in range(2)]
    rows.append(dict(rows[0],split='train',subject='other'))
    meta,indices=session_probe_split(rows)
    assert len(indices)==6 and set(r['split'] for r in meta)=={'train','validation','test'}
    assert all(r['subject']=='s' for r in meta)


def test_synthetic_pairs_share_noise_and_exact_alias(cfg):
    physical=yaml.safe_load((ROOT/cfg['teacher']['physical_config']).read_text())
    result=synthetic_interventions(cfg,physical,17)
    assert set(result)==set(cfg['mechanism']['scenarios'])
    assert np.array_equal(result['base']['eeg'],result['exact_alias']['eeg'])
    assert np.array_equal(result['base']['hb'],result['exact_alias']['hb'])
    assert not np.array_equal(result['base']['truth'],result['exact_alias']['truth'])
    assert result['tau']['tau']==3.
    assert not np.allclose(result['base']['hb'],result['tau']['hb'])


def test_freeze_refuses_silent_resume_config_change(cfg,runner,tmp_path):
    runner.freeze_run(cfg,tmp_path,ROOT)
    changed=deepcopy(cfg);changed['training']['steps']+=1
    with pytest.raises(ValueError):runner.freeze_run(changed,tmp_path,ROOT)


def miniature_cache(runner,path):
    path.mkdir();cache=path/'cache';cache.mkdir()
    rng=np.random.default_rng(3);n=12
    data=dict(eeg=rng.normal(size=(n,10,2400)).astype('float32'),
        hb=rng.normal(size=(n,10,16)).astype('float32'),
        teacher=rng.normal(size=(n,80,3)).astype('float32'),donors=np.arange(n))
    rows=[dict(id=f'id{i}',split=('train' if i<6 else 'validation' if i<9 else 'test'),
        dataset='a' if i%2 else 'b') for i in range(n)]
    for key,value in data.items():np.save(cache/(key+'.npy'),value)
    runner.write(cache/'manifest.json',dict(rows=rows,shapes={k:v.shape for k,v in data.items()},identity='synthetic_fixture'))


def test_training_optimizer_and_rng_resume_matches_uninterrupted(cfg,runner,tmp_path,monkeypatch):
    torch.set_num_threads(1)
    c=deepcopy(cfg);c['training'].update(steps=4,batch_size=2,checkpoint_every=2)
    continuous=tmp_path/'continuous';interrupted=tmp_path/'interrupted'
    miniature_cache(runner,continuous);miniature_cache(runner,interrupted)
    runner.train_one(c,continuous,'QT',17,'cpu')
    export=runner.export_model
    def interrupt(*args,**kwargs): raise RuntimeError('simulated_disconnect_after_atomic_checkpoint')
    monkeypatch.setattr(runner,'export_model',interrupt)
    with pytest.raises(RuntimeError):runner.train_one(c,interrupted,'QT',17,'cpu')
    monkeypatch.setattr(runner,'export_model',export)
    runner.train_one(c,interrupted,'QT',17,'cpu')
    left=torch.load(continuous/'models/QT__seed17/checkpoint.pt',weights_only=False)['model']
    right=torch.load(interrupted/'models/QT__seed17/checkpoint.pt',weights_only=False)['model']
    assert left.keys()==right.keys()
    assert all(torch.equal(left[k],right[k]) for k in left)
    ids,lookup=read_hard_tokens(interrupted/'models/QT__seed17/latest_export/hard.npz')
    assert len(ids)==12 and lookup['eeg'].shape==(12,10,16)


def test_synthetic_training_export_probes_mechanism_and_nulls(cfg,runner,tmp_path):
    """Exercise the complete consumer chain without any local measured cache."""
    c=deepcopy(cfg);c['training'].update(steps=2,batch_size=2,checkpoint_every=2)
    c['data']['tasks']=['motor_imagery']
    c['evaluation'].update(label_fractions=[1.],probe_C=[1.],ridge_alpha=[1.],probe_dimensions=4,kmeans_sizes=[16])
    c['mechanism'].update(train_identities=4,validation_identities=2,test_identities=2)
    out=tmp_path/'suite';cache=out/'cache';cache.mkdir(parents=True)
    rows=[]
    for split in ('train','validation','test'):
        for subject in range(2):
            for session in range(3):
                for event,condition in enumerate(('LMI','RMI')):
                    trial=f'{split}_{subject}_{session}_{event}'
                    for region in c['data']['regions']:
                        rows.append(dict(id=trial+'_'+region,trial_id=trial,split=split,
                            dataset='eeg_fnirs_single_trial',subject=split+str(subject),record=str(session),
                            session=str(session),task='motor_imagery',condition=condition,
                            eeg_start_s=120*event,coordinate='eeg_fnirs_single_trial__'+region,
                            anchor=dict(region=region)))
    n=len(rows);rng=np.random.default_rng(8)
    data=dict(eeg=rng.normal(size=(n,10,2400)).astype('float32'),hb=rng.normal(size=(n,10,16)).astype('float32'),
        teacher=rng.normal(size=(n,80,3)).astype('float32'),bands=rng.normal(size=(n,80,30)).astype('float32'),
        u=rng.normal(size=(n,80,3)).astype('float32'),donors=label_blind_donors(rows,3))
    data.update(smooth=data['teacher'],physical=data['u']*.75,remaining=data['u']*.25)
    for key,value in data.items():np.save(cache/(key+'.npy'),value)
    runner.write(cache/'manifest.json',dict(rows=rows,shapes={k:v.shape for k,v in data.items()},identity='synthetic_fixture',
        target_mean=[0,0,0],target_scale=[1,1,1],planned_trials=n//3,retained_trials=n//3,failed_trials=[]))
    runner.train_one(c,out,'QT',17,'cpu')
    path=out/'models/QT__seed17/latest_export/continuous.npz';original=path.read_bytes();path.write_bytes(b'poison continuous path')
    hard=runner.evaluate_hard_worker((c,str(out),'QT',17))
    assert hard['status']=='completed' and hard['classification_rows']>0
    path.write_bytes(original)
    assert runner.evaluate_model_worker((c,str(out),'QT',17))['status']=='completed'
    assert runner.audit_information(c,out)['status']=='completed'
    runner.mechanism_inputs(c,out)
    assert runner.mechanism_worker((c,str(out),'QT',17))['status']=='completed'
    assert runner.coupling_worker((c,str(out),'QT',17))['status']=='completed'
    runner.atlas_and_seed_stability(c,out)
    coupling=runner.read(out/'models/QT__seed17/coupling.json')
    assert coupling['paired_trials']==n//3 and coupling['common_support']
    assert runner.read(out/'models/QT__seed17/hard_evaluation.json')['continuous_export_read'] is False
