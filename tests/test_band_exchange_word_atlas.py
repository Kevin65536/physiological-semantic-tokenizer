import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('atlas',Path(__file__).resolve().parents[1]/'experiments/scripts/export_band_exchange_word_atlas.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def test_paid_words_remain_whole_and_fold_subject_counts_do_not_overflow():
    ids=np.zeros((4,15),np.uint16);ids[:2]=1024;ids[2:]=2047
    valid=np.ones_like(ids,bool);fold=np.array([1,1,0,0]);people=np.array([50,51,58,59],np.int16)
    r=a.occupancy(ids,valid,fold,people,np.array([0,1,2,3]),0)
    assert r['train'][1024]==30 and r['held'][2047]==30
    assert r['train_subjects'][1024]==2 and r['held_subjects'][2047]==2
    assert r['train_datasets'][1024]==2
    assert not r['major'][0] and r['major'][1024]
    assert r['train'][2047]==0
    with pytest.raises(ValueError):a.occupancy(ids.astype(float),valid,fold,people,np.zeros(4),0)


def test_representatives_prefer_people_and_never_score_waveforms():
    ix=np.arange(10);people=np.array([0]*6+[1]*2+[2]*2)
    x=a.representative_indices(ix,people,123,45)
    np.testing.assert_array_equal(x,a.representative_indices(ix,people,123,45))
    assert len(np.unique(people[x]))==3
    assert set(x)<=set(ix)


def test_encoded_full_member_template_quantiles_and_true_examples():
    import base64
    waves=np.arange(20,dtype=np.float32).reshape(5,4)
    p=a.waveform_record(waves,np.arange(5),np.array([3,1]))
    v=np.frombuffer(base64.b64decode(p['rows']),dtype='<f4').reshape(-1,4)
    np.testing.assert_array_equal(v[0],waves.mean(0))
    np.testing.assert_allclose(v[1:3],np.quantile(waves,[.1,.9],axis=0),rtol=1e-6)
    np.testing.assert_array_equal(v[3:],waves[[3,1]])
    assert a.waveform_record(waves,np.array([],int),np.array([],int)) is None
