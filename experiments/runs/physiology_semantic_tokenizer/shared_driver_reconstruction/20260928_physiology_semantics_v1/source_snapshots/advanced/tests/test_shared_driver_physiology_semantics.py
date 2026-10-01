"""Public-surface campaign contracts; no ignored or measured input fixtures."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.scripts import evaluate_shared_driver_reconstruction as runner


CONFIG = Path(__file__).resolve().parents[1]/'experiments/configs/physiology_semantic_tokenizer/shared_driver_physiology_semantics_v1.yaml'


def test_versioned_contract_and_truth_grid(tmp_path):
    cfg=runner.semantics_config(CONFIG)
    runner.semantics_parameters(cfg).validate()
    specs=runner.semantics_synthetic_specs(cfg)
    assert len(specs)==288 and len({s['key'] for s in specs})==288
    assert {s['initial'] for s in specs}=={'rest','nonrest'}
    bad=deepcopy(cfg);bad['data']['motion_method']='mne';path=tmp_path/'bad.yaml'
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError):runner.semantics_config(path)


def make_cohort(tmp_path):
    rng=np.random.default_rng(5);records=[]
    for subject in ('S01','S02','S03','S04','S05','S06'):
        for probe in (1,2):
            key=f'{subject}_Probe{probe}';folder=tmp_path/'prepared'/key;folder.mkdir(parents=True)
            windows=[dict(window=i,eeg_start_s=i*30.,hb_start_s=i*30.+2.,
                event_id=i,task='visual',condition='RR',independent_block='continuous') for i in range(6)]
            anchor=dict(region='prefrontal',eeg_channels=[f'c{i}' for i in range(6)],hb_pair=0,hb_channel='CH1',
                windows=windows,array_file='prefrontal.npz')
            spec=dict(key=key,dataset='visual_cognitive_motivation',subject=subject,record=key,
                repeat_group=subject,anchors=[anchor])
            records.append(spec)
            record=dict(prepared=[anchor],qc=dict(channels=[dict(rho=.4,status='supported',support=1,
                jump_fraction=[0,0],flat_step_fraction=[0,0])]))
            (folder/'record.json').write_text(json.dumps(record))
            eeg=rng.normal(size=(6,120,30));eeg-=eeg[:,:20].mean(axis=1,keepdims=True)
            hb=rng.normal(size=(6,120,2));hb-=hb[:,:20].mean(axis=1,keepdims=True)
            np.savez(folder/'prefrontal.npz',eeg_features=eeg,hb=hb)
    (tmp_path/'inventory.json').write_text(json.dumps(dict(records=records)))
    return runner.semantics_cohort(runner.semantics_config(CONFIG),tmp_path)


def test_native_records_without_part_have_disjoint_repeats_and_probe_grouping(tmp_path):
    refs,coordinates=make_cohort(tmp_path)
    assert len(refs)==72 and len(coordinates)==5
    for subject in sorted({r['subject'] for r in refs}):
        own=[r for r in refs if r['subject']==subject]
        for start in {r['eeg_start_s'] for r in own}:
            assert len({r['repeat_fold'] for r in own if r['eeg_start_s']==start})==1
        a=[r for r in own if r['repeat_fold']==0];b=[r for r in own if r['repeat_fold']==1]
        assert max(r['eeg_start_s']+30 for r in a)<=min(r['eeg_start_s'] for r in b)
        assert all(r['repeat_kind']=='within_record_disjoint_blocks' for r in own)
        for r in own:
            key=f'{r["dataset"]}__{r["site"]}__outer{r["subject_fold"]}'
            assert subject not in coordinates[key]['training_subjects']


def test_target_and_coordinate_do_not_fit_heldout_amplitudes(tmp_path):
    refs,coordinates=make_cohort(tmp_path);cfg=runner.semantics_config(CONFIG)
    held=refs[0];key=f'{held["dataset"]}__{held["site"]}__outer{held["subject_fold"]}'
    original=coordinates[key];train=[r for r in refs if r['subject_fold']!=held['subject_fold']]
    path=Path(held['array_path']);e,h=runner.semantics_prepared_arrays(str(path))
    np.savez(path,eeg_features=e*1000,hb=h-1e8);runner.semantics_prepared_arrays.cache_clear()
    revised=runner.semantics_coordinate(train,tmp_path,key+'_recheck',cfg)
    for k in ('pc','eeg_factor','hb_factor','sd'):np.testing.assert_array_equal(original[k],revised[k])
    target=runner.semantics_target(train[0],revised)
    assert target.shape==(120,3) and np.isfinite(target).all()
    np.testing.assert_allclose(target[:20].mean(axis=0),0,atol=1e-15)


def test_processed_generator_has_matching_mean_contract():
    cfg=runner.semantics_config(CONFIG);spec=runner.semantics_synthetic_specs(cfg)[0]
    spec=dict(spec,coordinate='processed');a=runner.semantics_generate(cfg,spec)
    assert a['target'].shape==(24,120,3)
    assert a['states'].shape==(24,120,6)
    np.testing.assert_allclose(a['clean'][:,:20].mean(axis=1),0,atol=1e-13)
    assert np.linalg.matrix_rank(runner.semantics_model_operator())<360


def test_ridge_completion_hidden_value_intervention():
    rng=np.random.default_rng(17);train=rng.normal(size=(18,120,3));target=rng.normal(size=(120,3))
    for mode in ('center_EEG','center_Hb','HbO_hidden','HbR_hidden','EEG_hidden','Hb_hidden'):
        mask=runner.semantics_missing_mask(mode);changed=target.copy();changed[~mask]=1e12
        for cross in (False,True):
            a=runner.semantics_ridge_completion(train,target,mask,cross=cross)
            b=runner.semantics_ridge_completion(train,changed,mask,cross=cross)
            np.testing.assert_array_equal(a,b)
        assert (~mask).sum() in (16,32,120,240)


def test_observation_gauge_and_stress_pairing(tmp_path):
    cfg=runner.semantics_config(CONFIG)
    result=runner.semantics_observation_math(cfg,tmp_path)
    assert result['instantaneous_observation_rank']==3
    assert max(r['prediction_max_difference'] for r in result['gauge_examples'])<1e-13
    assert result['HbT_identity_max_error']<1e-13
    spec=dict(tau=2.,gain=1.,spectrum='slow',repeat=0,initial='nonrest',stress='control')
    control=runner.semantics_stress_generate(cfg,spec)
    perturbed=runner.semantics_stress_generate(cfg,dict(spec,stress='common_colored'))
    np.testing.assert_array_equal(control['driver'],perturbed['driver'])
    np.testing.assert_array_equal(control['states'],perturbed['states'])
    np.testing.assert_array_equal(control['target'][:,:,0],perturbed['target'][:,:,0])
    assert np.max(abs(control['target'][:,:,1:]-perturbed['target'][:,:,1:]))>1e-5


def test_hierarchy_truth_and_measurement_only_counterexample_are_distinct():
    cfg=runner.semantics_config(CONFIG)
    base=runner.semantics_hierarchy_generate(cfg,dict(mechanism='global',repeat=0))
    gain=runner.semantics_hierarchy_generate(cfg,dict(mechanism='hb_gain_only',repeat=0))
    assert base['target'].shape==(4*4*3*6,120,3)
    np.testing.assert_array_equal(base['truth_tau'],gain['truth_tau'])
    np.testing.assert_array_equal(base['driver'],gain['driver'])
    np.testing.assert_array_equal(base['states'],gain['states'])
    np.testing.assert_array_equal(base['target'][:,:,0],gain['target'][:,:,0])
    assert not np.allclose(base['target'][:,:,1:],gain['target'][:,:,1:])


def test_joint_summary_keeps_failed_fits_in_full_denominator(tmp_path):
    cfg=runner.semantics_config(CONFIG);specs=runner.semantics_synthetic_specs(cfg)[:3]
    for i,spec in enumerate(specs):
        path=tmp_path/'synthetic_joint'/spec['key']/'result.json';path.parent.mkdir(parents=True)
        row=dict(converged=i!=2,parameter_values=[spec['tau'],spec['gain']],relative_errors=[0.,0.])
        path.write_text(json.dumps(row))
    summary=runner.semantics_joint_summary(cfg,tmp_path,specs)
    assert summary['terminal']==3 and summary['success_rate']==2/3 and not summary['passed']
    assert (tmp_path/'synthetic_joint_metrics.csv').exists()
