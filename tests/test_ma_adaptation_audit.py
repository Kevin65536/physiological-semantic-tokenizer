from copy import deepcopy
import importlib

import numpy as np
import pytest
import torch
from torch import nn

from comparative_methods.performance_analysis.audit_ma_adaptation_v1 import resolve_plan
from comparative_methods.rebench_eeg_common_v1 import validate_profile, encoding_duration
from comparative_methods.rebench_baselines_v1 import SignalFeatures


def profile():
    return dict(task="mental_arithmetic",duration_s=30,window_start_s=-5,input_window_schema="ma_30s_diagnostic_v1",
                sampling_rate_hz=200,fnirs_sampling_rate_hz=10,input_samples=6000,fnirs_samples=300,
                channel_names=[f"C{i}" for i in range(16)],fnirs_channel_names=["a_HbO","b_HbO","a_HbR","b_HbR"],
                fnirs_location_names=["a","b"],num_classes=2)


def test_explicit_window_profile_and_rejection_of_silent_v1_relabel():
    p=profile()
    assert validate_profile("mental_arithmetic",p)[1]==6000
    assert encoding_duration("mental_arithmetic",p)==30
    for field,value in [("window_start_s",0),("duration_s",8),("input_window_schema","downstream_rebench_v1")]:
        with pytest.raises(ValueError):validate_profile("mental_arithmetic",{**p,field:value})
    with pytest.raises(ValueError):validate_profile("wg",p)
    old={**p,"input_window_schema":"downstream_rebench_v1","duration_s":8,"input_samples":1600}
    assert validate_profile("mental_arithmetic",old)[1]==1600


def test_same_30s_masked_signal_features_ignore_unobserved_values():
    p={**profile(),"baseline_modality":"multimodal"}
    model=SignalFeatures("mental_arithmetic",p)
    eeg=torch.randn(2,16,6000);hb=torch.randn(2,4,300);mask=torch.ones_like(hb,dtype=torch.bool)
    mask[0,:,-40:]=False
    expected=model(eeg,hb,fnirs_mask=mask)
    hb[~mask]=float("nan")
    torch.testing.assert_close(model(eeg,hb,fnirs_mask=mask),expected)
    assert expected.shape==(2,112)


def test_migration_freezes_subjects_and_new_ids_before_signals(monkeypatch):
    p=profile();eeg={**p,"channel_names":p["channel_names"],"eeg_geometry":[[i,i%3] for i in range(16)],"geometry_provenance":{"eeg":[{}]*16}}
    multi={**deepcopy(eeg),"channel_names":p["channel_names"]+["extra"],"eeg_geometry":[[i,i%3] for i in range(17)]}
    samples=[];events={};records={}
    for subject in ("train","validation","outer"):
        records[subject]=dict(array_shapes={"eeg":[20000,17],"fnirs":[1000,4]})
        for label in (0,1):
            event=dict(eeg_time_ms=10000+30000*label,fnirs_time_ms=10000+30000*label,metadata={})
            events[(subject,label)]=event
            samples.append(dict(task_id="mental_arithmetic",subject_id=subject,dataset_id="eeg_fnirs_single_trial",record_id=subject,
                                event_id=label,target=label,label=["MA","BL"][label],sample_id=f"old_{subject}_{label}",
                                event_refs=[dict(join_key=subject,event_index=label,**{k:v for k,v in event.items() if k!="metadata"})]))
    inv=dict(samples=samples,records=records,tasks={"mental_arithmetic":dict(classes=["MA","BL"],profiles={"eeg_only":eeg,"multimodal":multi})})
    splits=dict(tasks={"mental_arithmetic":dict(folds=[dict(eval_subjects=["outer"],inner_folds=[dict(train_subjects=["train"],validation_subjects=["validation"])])])})
    config=dict(task="mental_arithmetic",outer_fold=0,inner_fold=0,input=dict(duration_s=30,window_start_s=-5))
    before=deepcopy(inv)
    monkeypatch.setattr(np,"load",lambda *a,**k:pytest.fail("metadata migration opened arrays"))
    new,plan,part=resolve_plan(inv,splits,config,events)
    assert inv==before and len(new["samples"])==4 and plan["train_count"]==2
    assert set(part)=={"train","validation"}
    assert new["tasks"]["mental_arithmetic"]["profiles"]["multimodal"]["channel_names"]==p["channel_names"]
    assert all("start_us=-5000000|stop_us=25000000" in r["sample_id"] for r in new["samples"])
    assert all(r["subject_id"]!="outer" for r in new["samples"])
    events[("train",1)]["eeg_time_ms"]=20000
    with pytest.raises(ValueError,match="another task onset"):resolve_plan(inv,splits,config,events)


def test_cbramod_scale_roundtrip_and_ordered_tokens(monkeypatch):
    module=importlib.import_module("comparative_methods.CBraMod.rebench_v1")
    class Encoder(nn.Module):
        def forward(self,x):return x
    class Base(nn.Module):
        def __init__(self):super().__init__();self.encoder=Encoder();self.input_samples=6000
    monkeypatch.setattr(module,"create_model",lambda *a,**k:Base())
    p={**profile(),"audit_training_eeg_scaler":{"mean":[[2.]]*16,"scale":[[4.]]*16},"eeg_unit":"uV","audit_readout":"flatten","audit_scale":"physical_100uv"}
    x=torch.randn(2,16,6000)
    model=module.create_audit_model("mental_arithmetic",p,"cpu")
    torch.testing.assert_close(model(x),((x*4+2)/100).flatten(1))
    with pytest.raises(ValueError,match="complete measured EEG"):model(x,eeg_mask=torch.zeros_like(x,dtype=torch.bool))


def test_brainfusion_30s_requires_matching_tensor_lengths():
    module=importlib.import_module("comparative_methods.BrainFusion-NVC-CSP-Stacking.rebench_v1")
    model=module.create_estimator(profile())
    model._inputs(torch.randn(2,16,6000),torch.randn(2,4,300))
    with pytest.raises(ValueError,match="common encoding window"):model._inputs(torch.randn(2,16,1600),torch.randn(2,4,80))


def test_sta_30s_uses_only_the_28_within_window_segments():
    module=importlib.import_module("comparative_methods.STA-Net-PyTorch.rebench_v1")
    p=dict(duration_s=30,window_start_s=-5,input_window_schema="ma_30s_diagnostic_v1",sampling_rate_hz=10,
           fnirs_sampling_rate_hz=2,eeg_geometry=[[-1,-1],[-1,1],[1,-1],[1,1]],
           fnirs_geometry=[[-.8,-1],[-.8,1],[.8,-1],[.8,1]],embedding_dim=8,attention_heads=2,attention_key_dim=4,dropout=0)
    model=module.create_model("mental_arithmetic",p,"cpu").eval()
    assert len(model.segment_starts)==28
    assert model.segment_starts[-1]+model.segment_samples==60
    with torch.no_grad():out=model(torch.randn(2,4,300),torch.randn(2,8,60))
    assert out.shape==(2,2) and torch.isfinite(out).all()
