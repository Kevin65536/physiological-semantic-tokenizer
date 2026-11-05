"""Small public-shaped fixtures exercise the new input contract without local data."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.data import downstream_input_cache as data
from src.data.clean_physiology_cache import CleanCacheRecord, with_canonical_fields
from src.data.functional_tokenizer import subject_partitions
from src.data.registry import REGISTERED_DATASETS
from src.data.unified_physiology import NativeEEGRecord, VISUAL_CHANNEL_IDENTITY_SCHEMA


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    cfg = data.read_config(Path(__file__).parents[1]/"experiments/configs/physiology_semantic_tokenizer/downstream_input_v1.yaml")
    records, events = [], defaultdict(list)
    visual_names = [f"E{i}" for i in range(30)] + ["FCz"]
    mapping = {**{f"A{i+1}": visual_names[i] for i in range(30)}, "A64":"FCz"}
    monkeypatch.setattr(data, "_visual_eeg_name_map", lambda _: mapping)
    for ds in cfg["datasets"]:
        for index in range(1, 6):
            subject = (f"subject_{index:02d}" if ds == "eeg_fnirs_single_trial" else
                       f"VP{index:03d}" if ds == "simultaneous_eeg_nirs" else
                       f"S{index:02d}" if ds == data.VISUAL else str(index))
            names = [f"{subject}_Probe1", f"{subject}_Probe2"] if ds == data.VISUAL else [
                "session_00" if ds == "eeg_fnirs_single_trial" else "cnt_wg" if ds == "simultaneous_eeg_nirs" else "video_1_hbo_hbr"]
            for record_id in names:
                seconds = 65 if ds == "refed" else 70
                folder = tmp_path/"source_arrays"/ds/subject/record_id; folder.mkdir(parents=True)
                hb = np.column_stack((np.sin(np.arange(seconds*10)/30), np.cos(np.arange(seconds*10)/50))).astype("float64")
                np.save(folder/"fnirs.npy",hb); np.save(folder/"fnirs_supported_channels.npy",[True,False])
                source = tmp_path/REGISTERED_DATASETS[ds].default_root/subject/"fixture.mat"
                source.parent.mkdir(parents=True,exist_ok=True); source.write_bytes(b"fixture")
                identity = with_canonical_fields(dict(dataset_id=ds,subject=subject,record_id=record_id))
                old_names = visual_names[:-1] if ds == data.VISUAL else ["E0","E1"]
                m = dict(identity,schema="clean_eeg_fnirs_cache_v2",storage="measurement_npy_record_v1",
                    processing_schema="physiology_measurement_alignment_v5",motion_method="none",
                    sample_rate_hz=10.,record_npz=str(folder),source_files=[dict(path=str(source),sha256="fixture")],
                    native_contract={"native_unit":"fixture"},
                    arrays={k:k+".npy" for k in ("fnirs","fnirs_supported_channels")},
                    array_shapes=dict(eeg=[seconds*200,len(old_names)],fnirs=list(hb.shape),fnirs_supported_channels=[2]),
                    measurement=dict(eeg_channel_names=old_names,fnirs_channel_names=["CH1_HbO","CH1_HbR"],
                        fnirs_preprocessing_state=dict(canonical_unit="relative_fixture",canonical_sample_rate_hz=10.)),
                    homer2_aligned_contract=dict(alignment_state=dict(schema="physiology_measurement_alignment_v5",parameters={"motion_method":"none"}),
                        quality=dict(bandpass=dict(status="applied",low_hz=.01,high_hz=.2))))
                r = CleanCacheRecord(**{k:identity[k] for k in ("dataset_id","subject","record_id","canonical_subject_id","base_record_id","signal_branch","join_key")},sample_rate_hz=10.,npz_path=folder,manifest=m)
                records.append(r)
                task = "motor_imagery" if ds == "eeg_fnirs_single_trial" else "wg" if ds == "simultaneous_eeg_nirs" else "visual_cognitive_motivation"
                condition = "LMI" if ds == "eeg_fnirs_single_trial" else "WG" if ds == "simultaneous_eeg_nirs" else "RR"
                e = dict(event_index=0,event_type="trial",label=condition,label_index=0,eeg_time_ms=20000.,fnirs_time_ms=20000.,
                         metadata=dict(task=task,alignment_support_ms={"eeg":[0,70000],"fnirs":[0,70000]}))
                if ds == "refed":
                    e.update(event_type="video_segment_with_continuous_labels",eeg_time_ms=0.,fnirs_time_ms=0.,duration_ms=65000.,
                        metadata=dict(task="emotion_video",eeg_samples=65000,fnirs_samples=3096,
                            continuous_label_stream=dict(values=np.column_stack((128+np.arange(65),128-np.arange(65))).tolist(),
                                names=["valence","arousal"],sample_count=65,native_sample_rate_hz=1.)))
                events[r.join_key].append(e)
    fake = SimpleNamespace(records=records,events_by_join_key=events,
        reports_by_join_key={r.join_key:[dict(alignment_case="stable_fixed_offset",label_sequence_match=True)] for r in records})
    real_index = data.CleanPhysiologyCacheIndex
    monkeypatch.setattr(data,"CleanPhysiologyCacheIndex",lambda _:fake)
    monkeypatch.setattr(data,"_source_index",lambda _: {r.join_key:r for r in records})
    subjects={ds:sorted({r.canonical_subject_id for r in records if r.dataset_id==ds}) for ds in cfg['datasets'][:2]}
    assigned=subject_partitions(subjects,20261010)
    old={ds:{s:[k[1] for k,v in assigned.items() if k[0]==ds and v==s] for s in ['train','validation','test']} for ds in subjects}
    parent=tmp_path/cfg['split']['retained_two_dataset_inventory'];parent.parent.mkdir(parents=True);parent.write_text(json.dumps({'subjects':old}))
    native_reads=[]
    def native(root, record):
        native_reads.append(record.join_key)
        n=record.manifest['array_shapes']['eeg'][0];names=visual_names if record.dataset_id==data.VISUAL else ['E0','E1']
        values=np.sin(2*np.pi*10*np.arange(n)[:,None]/200)*np.arange(1,len(names)+1)[None,:]
        return NativeEEGRecord(values,200.,tuple(names),'uV',Path(record.manifest['source_files'][0]['path']),
            unit_evidence='synthetic',channel_identity={'schema':VISUAL_CHANNEL_IDENTITY_SCHEMA} if record.dataset_id==data.VISUAL else None)
    monkeypatch.setattr(data,'load_native_eeg_record',native)
    return SimpleNamespace(root=tmp_path,cfg=cfg,records=records,events=events,index=fake,
                           real_index=real_index,native_reads=native_reads)


def test_plan_is_signal_free_preserves_subjects_and_covers_refed_tail(prepared,monkeypatch):
    f=prepared
    monkeypatch.setattr(data,'load_native_eeg_record',lambda *a:pytest.fail('signal opened while planning'))
    plan=data.make_plan(f.root,f.cfg)
    assert len(plan['records'])==25
    assert len(plan['samples'])==35
    refed=[s for s in plan['samples'] if s['dataset']=='refed' and s['subject']=='1']
    assert [s['eeg_start_s'] for s in refed]==[0,30,60]
    assert [s['supported_seconds'] for s in refed]==[30,30,5]
    assert refed[-1]['partial']
    visual=[s for s in plan['samples'] if s['dataset']==data.VISUAL and s['subject']=='S01']
    assert len({s['dependency_id'] for s in visual})==1 and all(s['dependency_views']==2 for s in visual)
    # Dataset order is not part of the random seed.
    cfg=deepcopy(f.cfg);cfg['datasets'].reverse()
    assert data.make_plan(f.root,cfg)['subject_partitions']==plan['subject_partitions']


def test_clock_gaps_and_unknown_labels_stay_in_full_exclusion_record(prepared):
    f=prepared;r=f.records[0];e=f.events[r.join_key][0]
    e['metadata']['alignment_support_ms']['eeg']=[19000,70000]
    other=f.records[-1];f.events[other.join_key][0]['label']='unknown'
    plan=data.make_plan(f.root,f.cfg)
    reasons={r['reason'] for r in plan['exclusions']}
    assert 'crosses_unverified_clock_support' in reasons
    assert 'unknown_or_unsupported_label' in reasons
    assert len(plan['records'])==25


def test_changed_source_fails_before_native_access(prepared,monkeypatch):
    f=prepared;plan=data.make_plan(f.root,f.cfg);entry=plan['records'][0]
    entry['source_signature']['source_files']=[]
    monkeypatch.setattr(data,'load_native_eeg_record',lambda *a:pytest.fail('invalid source opened'))
    with pytest.raises(ValueError,match='source changed'):
        data.prepare_group(f.root,f.cfg,f.root/'out',[entry])


def test_record_cache_roundtrip_masks_labels_and_visual_shared_storage(prepared,monkeypatch):
    f=prepared;plan=data.make_plan(f.root,f.cfg);out=f.root/'out'
    selected=[r for r in plan['records'] if (r['dataset']=='refed' and r['subject']=='1') or
              (r['dataset']==data.VISUAL and r['subject']=='S01')]
    groups=defaultdict(list)
    for r in selected:groups[r['native_group']].append(r)
    generated=[]
    for entries in groups.values():generated.extend(data.prepare_group(f.root,f.cfg,out,entries)['records'])
    assert len(f.native_reads)==2  # Two Visual probes decode one EEG record.
    visual=[r for r in generated if r['dataset_id']==data.VISUAL]
    assert (Path(visual[0]['record_npz'])/'eeg.npy').stat().st_ino==(Path(visual[1]['record_npz'])/'eeg.npy').stat().st_ino
    for r in generated:
        original=next(old for old in f.records if old.join_key==r['join_key'])
        np.testing.assert_array_equal(np.load(Path(r['record_npz'])/'fnirs.npy'),np.load(original.npz_path/'fnirs.npy'))
    keys={r['join_key'] for r in selected}
    plan['records']=selected;plan['samples']=[s for s in plan['samples'] if s['join_key'] in keys]
    data.write_json(out/'input_plan.json',plan)
    data.write_json(out/'cache_manifest.json',dict(schema='clean_eeg_fnirs_cache_v2',execution='completed',
        storage='measurement_npy_record_v1',processing_schema='physiology_measurement_alignment_v5',
        motion_method='none',downstream_input_schema=data.SCHEMA,records=generated))
    data.write_json(out/'event_index/event_manifest.json',{'event_alignment_schema':'physiology_event_alignment_v2'})
    monkeypatch.setattr(data,'CleanPhysiologyCacheIndex',f.real_index)
    monkeypatch.setattr(data,'load_native_eeg_record',lambda *a:pytest.fail('cache consumer reopened native EEG'))
    dataset=data.DownstreamInputDataset(out)
    sample=next(dataset[i] for i,s in enumerate(dataset.samples) if s['dataset']=='refed' and s['partial'])
    assert sample['eeg'].shape==(2,6000) and sample['fnirs'].shape==(2,300)
    assert sample['valid_mask']['eeg'].sum()==2*5*200
    assert sample['valid_mask']['fnirs'].sum()==5*10
    assert sample['label']['values'].shape==(2,30) and sample['label']['valid_mask'].sum()==10
    np.testing.assert_array_equal(sample['label']['values'][0,:5],128+np.arange(60,65))
    assert np.all(sample['eeg'][:,1000:]==0)
    v=next(dataset[i] for i,s in enumerate(dataset.samples) if s['dataset']==data.VISUAL)
    assert v['eeg'].shape==(31,6000) and v['label']['target_event_time_s']==5
    assert v['dependency_views']==2 and v['label']['class_index']==0
    # A consumer may annotate its returned metadata without changing later samples.
    v['channel_geometry']['eeg'][0]['x']=999.
    again=next(dataset[i] for i,s in enumerate(dataset.samples) if s['dataset']==data.VISUAL)
    assert again['channel_geometry']['eeg'][0]['x']!=999.


def test_old_subjects_cannot_silently_move_when_expanding_datasets(prepared):
    f=prepared;cfg=deepcopy(f.cfg)
    cfg['split']['dataset_seeds']['simultaneous_eeg_nirs']+=1
    with pytest.raises(ValueError,match='change a retained subject'):
        data.make_plan(f.root,cfg)
