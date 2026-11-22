import numpy as np
import pytest
from experiments.scripts.unified_so_production import admitted, choose_positions, output_grid


def test_short_refed_and_fractional_visual_keep_acquisition_identity():
    g=output_grid(dict(key='parent',start_sample=38,rate=500,duration_s=60.,samples=30000))
    assert (g['key'],g['native_rate_hz'],g['native_start_sample'])==('parent',500,38)
    assert (g['start_sample'],g['rate'],g['samples'])==(15,200,12000)
    assert g['grid_rounding_s']==pytest.approx(-.001)


def test_match_prior_proxy_then_replace_unavailable_site_only_from_qualified_pool():
    stats={'covariance':np.eye(5)}
    pre=choose_positions(['b','e','a'],list('abcde'),np.ones(5,bool),np.eye(5),stats)
    assert list(pre)==[1,4,0]
    remaining=choose_positions(['b','e','a'],list('abcde'),np.array([1,0,1,0,0],bool),np.eye(5),stats)
    assert list(remaining)==[0,2]


def test_dataset_and_original_public_subject_membership_before_reads():
    b=dict(datasets=['refed'],splits=['train','validation','test'])
    parts={'refed':{'test':['1'], 'train':[], 'validation':[]}}
    admitted(dict(dataset='refed',subject='1',split='test'),b,parts)
    with pytest.raises(ValueError):
        admitted(dict(dataset='refed',subject='1',split='train'),b,parts)
    with pytest.raises(ValueError):
        admitted(dict(dataset='sealed',subject='1',split='test'),b,parts)
