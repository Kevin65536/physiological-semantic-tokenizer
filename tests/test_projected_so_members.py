import numpy as np
import pandas as pd
import pytest

from experiments.scripts.audit_projected_so_members import CFG, attributes, match, select


def test_signed_shape_and_physical_frequency():
    t = np.arange(400) / 200
    x = np.array([np.sin(2*np.pi*f*t) for f in (10,2)])
    unit, (alpha, slow) = attributes(x)
    neg, _ = attributes(-x)
    np.testing.assert_allclose((unit*neg).sum(1), -1)
    assert alpha[0] > .99 and slow[1] > .99 and slow[0] < .001
    with pytest.raises(ValueError):
        attributes(x[:,:200])


def test_selection_excludes_validation_and_matching_obeys_bounds():
    n = CFG['eligibility']['patches']
    ids = np.repeat([10,20,30], n)
    x = np.tile(np.sin(2*np.pi*10*np.arange(400)/200),(len(ids),1))
    unit, _ = attributes(x)
    m = pd.DataFrame(dict(S=ids, role=np.where(ids==30,'validation','train'), alpha=ids, slow=-ids,
                          person=np.arange(len(ids))%8,dataset=np.arange(len(ids))%2))
    out = select(m,unit,'S')
    assert [r[1] for r in out[1:]] == [20,10]
    groups = pd.DataFrame(dict(person=['a','a','b','a','a'],site=['Pz','Pz','Pz','Pz','Fz'],log_rms=[0,.12,.01,1,.02]))
    members = np.array([1,0,0,1,0],dtype=bool)
    np.testing.assert_array_equal(match(groups,members,~members),[[0,1]])
