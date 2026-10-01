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
