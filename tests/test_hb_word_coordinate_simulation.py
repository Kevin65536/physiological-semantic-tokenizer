import numpy as np
import pandas as pd
import pytest

from experiments.scripts.run_hb_word_coordinate_simulation import CONFIG, configuration
from src.analysis.hb_word_coordinate_simulation import codebook, evaluate_cell, population, summarize


def test_hidden_linear_slope_and_paired_regimes():
    conditions = {}
    for regime in ('ordinary','level_tails','slope_tails','both_tails'):
        x, y, persons = population(71,0,0,32,16,regime,noise_sd=0)
        inferred = codebook('standard',17).pattern_features(x)[:,2:]
        np.testing.assert_allclose(inferred,y,atol=3e-6)
        conditions[regime] = x,y
    np.testing.assert_array_equal(conditions['ordinary'][1],conditions['level_tails'][1])
    assert not np.array_equal(conditions['ordinary'][1],conditions['slope_tails'][1])
    assert not np.array_equal(conditions['ordinary'][0],population(71,0,1,32,16,'ordinary',noise_sd=0)[0])


@pytest.mark.parametrize('arm,kept', [('rank_levels',slice(2,None)),('rank_slopes',slice(0,2))])
def test_partial_rank_changes_only_chosen_group(arm,kept):
    x,_,_ = population(71,0,0,4,32,'both_tails')
    c = codebook(arm,17)
    with pytest.raises(ValueError):
        c.fit(x,partition='validation')
    c.fit(x)
    f = c.pattern_features(x)
    np.testing.assert_array_equal(c._route_features(x)[:,kept],f[:,kept])
    assert not np.allclose(c._route_features(x),f)
    assert c.rate('prototype')['bits_per_item'] == 4


def test_endpoints_pair_generator_replicates_and_reject_dropped_cells():
    cfg = dict(configuration(CONFIG),replicates=2,regimes=['ordinary','level_tails'],
               training_subjects_per_population=4,training_patches_per_subject=12,
               scoring_subjects=3,scoring_patches_per_subject=8)
    t = pd.DataFrame([r for i in range(2) for regime in cfg['regimes'] for r in evaluate_cell(cfg,i,regime)])
    s = summarize(t,cfg)
    assert s['passed'] == all(s['checks'].values())
    assert s['overall']['means']['standard']['continuous_ols_nmse'] < .03
    a = t.groupby('arm').ari.mean()
    assert s['overall']['paired']['rank_levels']['ari']['difference'] == pytest.approx(a.rank_levels-a.standard)
    with pytest.raises(ValueError):
        summarize(t.iloc[:-1],cfg)
