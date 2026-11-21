import json

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.scripts import run_observation_word_stability as runner
from src.analysis.observation_tokenizer_evaluation import hierarchy_weights
from src.analysis.observation_word_stability import subject_agreement, summarize, training_draw, word_support
from src.tokenizers.rank_observation_words import RankObservationWords


def frame(prefix, n=16):
    return pd.DataFrame([dict(dataset=f'd{d}', subject=f'{prefix}{s}', record='r', parent='p',
                              start_s=0, position='a', patch=j)
                         for d in range(4) for s in range(3) for j in range(n)])


@pytest.mark.parametrize('mod,shape', [('eeg', (1, 400)), ('hb', (2, 20))])
def test_rank_integer_batch_invariance_and_split(mod, shape):
    rng = np.random.default_rng(231)
    train, val = (rng.normal(size=(n,*shape)) for n in (128, 32))
    c = RankObservationWords(mod, seed=17)
    with pytest.raises(ValueError, match='partition=train'):
        c.fit(val, partition='validation')
    c.fit(train)
    expected = c.encode(val, 'prototype')
    assert set(expected) == {'pattern'} and expected['pattern'].dtype.kind in 'iu'
    assert c.rate('prototype')['bits_per_item'] == 4 and expected['pattern'].max() < 16
    np.testing.assert_array_equal(c.encode(np.concatenate([val, 1e6*val]), 'prototype')['pattern'][:32], expected['pattern'])
    with pytest.raises(ValueError, match='expected'):
        c.encode(val[:, :, :-1], 'prototype')


def test_draw_omits_training_dataset_and_bootstraps_people():
    cfg, _ = runner.configuration(runner.CONFIG)
    f = frame('t')
    w = hierarchy_weights(f)
    full = training_draw(f,w,'full',0,cfg)
    np.testing.assert_array_equal(full, np.random.default_rng(93).choice(len(w), 16384, p=w))
    omit = training_draw(f,w,'omit','d1',cfg)
    assert 'd1' not in set(f.iloc[omit].dataset)
    a = training_draw(f,w,'bootstrap',4,cfg)
    np.testing.assert_array_equal(a, training_draw(f,w,'bootstrap',4,cfg))
    assert len(f.iloc[a][['dataset','subject']].drop_duplicates()) < 12
    with pytest.raises(ValueError):
        training_draw(f,w,'omit','other',cfg)


def test_parser_and_matrix(tmp_path):
    cfg, _ = runner.configuration(runner.CONFIG)
    assert json.loads(json.dumps(cfg)) == cfg
    cases = runner.tasks(cfg)
    assert len(set(cases)) == len(cases) == 414
    altered = dict(cfg, splits=['test'])
    path = tmp_path/'new.yaml'
    path.write_text(yaml.safe_dump(altered))
    with pytest.raises(ValueError):
        runner.configuration(path)


def test_permutation_support_and_degeneracy():
    f = frame('v')
    a = np.arange(len(f)) % 4
    scores = subject_agreement(f.groupby(['dataset','subject']).indices,a,3-a)
    assert len(scores) == 12 and all(r['ari'] == 1 and r['refit_words'] == 4 for r in scores)
    support = word_support(f,a)
    assert len(support) == 16 and sum(r['patches'] for r in support) == len(f)
    assert support[5]['subjects'] == 0 and np.isclose(sum(r['mass'] for r in support),1)


def test_paired_cell_through_readouts_and_decision(tmp_path, monkeypatch):
    cfg, _ = runner.configuration(runner.CONFIG)
    cfg = dict(cfg, draw_size=256)
    mod = 'hb'
    t, v = frame('t', 8), frame('v', 8)
    rng = np.random.default_rng(1493)
    xt, xv = (rng.normal(size=(len(f), 2, 20)) for f in (t,v))
    w = hierarchy_weights(t)
    ids = training_draw(t,w,'full',0,cfg)
    base = runner.fit_arm(mod,'standard',xt[ids],17)
    for f,x in [(t,xt),(v,xv)]:
        f['pattern'] = base.encode(x,'prototype')['pattern']
        f['blind'] = rng.integers(16, size=len(f))
    env = dict(train=t,val=v,xt=xt,xv=xv,weights=w,groups=v.groupby(['dataset','subject']).indices,reference={})
    monkeypatch.setattr(runner,'WORK',dict(hb=env,output=tmp_path,config=cfg))
    (tmp_path/'cells').mkdir()
    rows = []
    for kind,idx,seed in [('full',0,17),('full',0,73),('bootstrap',0,73),('omit','d1',17)]:
        case = (mod, kind, idx, seed)
        runner.run_cell(case)
        cell = runner.read(runner.cell_path(tmp_path,case))
        rows.extend(cell['rows'])
        if kind == 'omit':
            assert {r['dataset'] for r in cell['rows']} == {'d1'}
    def tables(suffix):
        return pd.concat([pd.read_csv(tmp_path/f'hb_{arm}_{suffix}.csv') for arm in cfg['arms']])
    s = summarize(pd.DataFrame(rows),tables('attributes'),tables('support'),cfg)[mod]
    assert s['paired_ari']['n_subjects'] == 12
    assert s['passed'] == all(s['checks'].values())
    # Equal partitions cannot pass the predeclared improvement threshold.
    equal = pd.DataFrame(rows).assign(ari=1.)
    s = summarize(equal,tables('attributes'),tables('support'),cfg)[mod]
    assert not s['passed'] and s['paired_ari']['ci95'] == [0,0]
