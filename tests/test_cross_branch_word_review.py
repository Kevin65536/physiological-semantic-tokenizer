"""Actual paid-ID readout must preserve person folds and the single-word namespace."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def owner():
    path=Path(__file__).resolve().parents[1]/'experiments/scripts/audit_cross_branch_words.py'
    spec=importlib.util.spec_from_file_location('cross_branch_words_audit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fixture():
    ids=np.broadcast_to(np.array([1,2]*7+[1],np.uint16),(6,15)).copy()
    bank=dict(valid=np.ones((6,15),bool),fold=np.array([0,0,0,1,1,1]),subject=np.arange(6),background=np.zeros((6,15),int),S_attributes=np.repeat(ids[...,None].astype(float),10,axis=2),fixed_band=np.arange(6)%5)
    return bank,ids


def test_actual_word_has_no_hidden_fixed_role_namespace():
    module=owner();bank,ids=fixture()
    held,error,mass,usage,counts=module.read_fold(bank,ids,0,'S',1)
    np.testing.assert_array_equal(held,[0,1,2])
    np.testing.assert_array_equal(error[:,1],0)
    np.testing.assert_array_equal(mass,1)
    assert len(usage['fit'])==5116 and usage['fit'][1:3].tolist()==[24,21]
    assert usage['fit_people'][1:3].tolist()==[3,3]
    assert counts['complete_held_patches']==counts['qualified_held_patches']==45
    # An absent fit word retains the same physical background baseline.
    ids[0]=3
    _,error,_,_,_=module.read_fold(bank,ids,0,'S',1)
    np.testing.assert_array_equal(error[0,0],error[0,1])


def test_actual_word_scope_and_dtype_fail_closed():
    module=owner();bank,ids=fixture()
    with pytest.raises(ValueError,match='uint16'):module.read_fold(bank,ids.astype(float),0,'S',1)
    illegal=ids.copy();illegal[0,0]=5116
    with pytest.raises(ValueError,match='K5116'):module.read_fold(bank,illegal,0,'S',1)
    bank['subject'][3]=bank['subject'][0]
    with pytest.raises(ValueError,match='person overlap'):module.read_fold(bank,ids,0,'S',1)
