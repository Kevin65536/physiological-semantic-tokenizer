import importlib
import numpy as np
import pytest
import torch
from torch import nn

from comparative_methods.performance_analysis.investigate_visual_v1 import target_view,chronological_indices,window_arrays,corrected_storage_panel
from comparative_methods.rebench_eeg_common_v1 import encoding_duration,validate_profile


def test_two_label_dimensions_keep_native_four_class_order():
    y=np.arange(4)  # RR, RF, FF, FR
    np.testing.assert_array_equal(target_view(y,'four_class')[0],[0,1,2,3])
    np.testing.assert_array_equal(target_view(y,'motivation')[0],[0,0,1,1])
    np.testing.assert_array_equal(target_view(y,'recognition')[0],[0,1,1,0])


def test_chronological_control_excludes_development_subjects_and_guards_boundary():
    a=dict(subject_id=np.asarray(['s1']*10+['s2']*10+['heldout']*10),partition=np.asarray(['train']*20+['validation']*10),event_id=np.tile(np.arange(1,11),3))
    tr,va,gaps=chronological_indices(a)
    assert not set(tr)&set(va) and not (set(tr)|set(va))&set(gaps)
    assert max(np.r_[tr,va,gaps])<20
    assert a['event_id'][tr].max()==7 and a['event_id'][va].min()==9
    assert a['event_id'][gaps].tolist()==[8,8]


def test_eight_second_prefix_retains_identities_labels_and_masks():
    a=dict(eeg=np.arange(2*16*2400).reshape(2,16,2400),fnirs=np.zeros((2,96,120)),eeg_mask=np.ones((2,16,2400),bool),fnirs_mask=np.ones((2,96,120),bool),target=np.array([1,3]),sample_id=np.array(['a','b']))
    a['fnirs_mask'][0,:,-50:]=False
    b=window_arrays(a,8)
    assert b['eeg'].shape==(2,16,1600) and b['fnirs'].shape==(2,96,80)
    assert b['sample_id'] is a['sample_id'] and b['target'] is a['target']
    np.testing.assert_array_equal(b['fnirs_mask'],a['fnirs_mask'][...,:80])


def test_visual_window_contract_is_explicit_and_task_specific():
    p=dict(input_window_schema='visual_12s_diagnostic_v1',window_start_s=0,duration_s=12,sampling_rate_hz=200,channel_names=[f'C{i}' for i in range(16)],input_samples=2400)
    assert validate_profile('visual',p)[1]==2400
    with pytest.raises(ValueError):encoding_duration('mental_arithmetic',p)
    with pytest.raises(ValueError):encoding_duration('visual',{**p,'window_start_s':-5})
    with pytest.raises(ValueError):encoding_duration('visual',{**p,'input_window_schema':'downstream_rebench_v1'})


def test_cbramod_diagnostic_head_emits_all_four_classes(monkeypatch):
    m=importlib.import_module('comparative_methods.CBraMod.rebench_v1')
    class Base(nn.Module):
        def __init__(self):super().__init__();self.input_samples=1600;self.encoder=nn.Identity()
    monkeypatch.setattr(m,'create_model',lambda *a,**k:Base())
    p=dict(audit_readout='mean',audit_head='linear',num_classes=4,audit_training_eeg_scaler={'mean':[[0.]]*16,'scale':[[1.]]*16})
    model=m.create_audit_model('visual',p,'cpu',trainable=True)
    prediction=model(torch.randn(8,16,1600));assert prediction.shape==(8,4)
    nn.functional.cross_entropy(prediction,torch.arange(8)%4).backward()
    assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all()


def test_native_electrode_mapping_selects_correct_storage_without_changing_physical_names():
    p=dict(channel_names=['Fp1','CP3','P3','O1'])
    rows=[dict(documented='FP1',raw_channel='A1',current='Fp1',currently_excluded=False),
          dict(documented='CP3',raw_channel='A11',current='P3',currently_excluded=False),
          dict(documented='P3',raw_channel='A13',current='PO3',currently_excluded=False),
          dict(documented='O1',raw_channel='A17',current='F5',currently_excluded=False),
          dict(documented='FCz',raw_channel='A64',current=None,currently_excluded=True)]
    aliases,raw=corrected_storage_panel(p,dict(channel_comparison=rows))
    assert aliases==['Fp1','P3','PO3','F5'] and raw==['A1','A11','A13','A17']
    assert p['channel_names']==['Fp1','CP3','P3','O1']
    with pytest.raises(KeyError):corrected_storage_panel(dict(channel_names=['FCz']),dict(channel_comparison=rows))
