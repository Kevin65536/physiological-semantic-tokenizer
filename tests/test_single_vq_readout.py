from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/scripts'))
import run_single_vq_readout as reader
from run_dual_component_tokenizer import _base_training_model


@pytest.mark.parametrize('mod,width', [('eeg', 400), ('hb', 40)])
def test_extract_shapes_and_exact_integer_replay(tmp_path, monkeypatch, mod, width):
    import yaml
    cap = reader.capacity
    cfg = yaml.safe_load((cap.CONTRACT.parent / 'dual_component_base_g1_v2.yaml').read_text())
    monkeypatch.setattr(cap, 'configuration', lambda: cfg)
    monkeypatch.setattr(reader, 'OUT', tmp_path)
    spec = reader.PLAN['models'][mod]
    ds = cap.DATASETS[0]
    x = np.random.default_rng(17).normal(size=(2, 15, width)).astype('float32')
    mask = np.ones_like(x, bool)
    mask[1, 0, 0] = False
    batch = dict(x=x, s=x*.6, o=x*.4, mask=mask)
    monkeypatch.setattr(reader, 'base_tokenizer_batch', lambda *a, **kw: batch)
    ex = np.array([[0, 0, 0], [0, 30, 0]])
    monkeypatch.setattr(cap, 'PANEL', dict(rows=[dict(positions=['Cz'])], scales={},
                                           examples={(mod, 'validation', ds): ex}))
    monkeypatch.setitem(reader.SITES, mod, {(ds, 'Cz'): 0})
    models, ids = {}, {}
    for size, seed in [(0, 17), *((spec['size'], s) for s in spec['seeds'])]:
        model = _base_training_model(cap.resolved(mod, spec['dimension'], seed), cap.model_task(mod, size)).eval()
        models[size, seed] = model
        if size:
            for branch in model.branches.values():
                branch['quantizer'].initialized.fill_(True)
            with torch.inference_mode():
                ids[seed] = {b: v.numpy() for b,v in model(torch.from_numpy(x))['ids'].items()}
    monkeypatch.setattr(reader, 'network', lambda m, k, s: models[k,s])
    monkeypatch.setattr(reader, 'retained', lambda m, d, r, s: ids[s])
    task = (mod, ds, 'validation', 0, 2)
    assert reader.extract(task, pilot=True)['patches'] == 29
    reader.extract(task)
    with np.load(reader.destination(task)) as z:
        assert z['S_17_z'].shape == (29, spec['dimension'])
        assert z['O_target'].shape == z['O_continuous_wave'].shape
        np.testing.assert_array_equal(z['S_17_id'], ids[17]['S'][mask.all(-1)])


def test_sparse_shrinkage_and_person_balance():
    from src.analysis.frozen_token_readout import Conditional
    bg = np.zeros(5, int)
    ids = np.array([3,3,3,3,1])
    people = np.array([0,0,0,1,1])
    y = np.array([[0.],[0.],[0.],[8.],[12.]])
    queries = np.array([3,2])
    for balanced, mean, count, prior in [(False,2.,4,4.),(True,4.,2,5.)]:
        table = Conditional(bg,ids,people,y,8,balanced)
        parts = table.read(bg[:2],queries)
        np.testing.assert_array_equal(parts[2],[count,0])
        np.testing.assert_allclose(table.shrink(*parts,0)[:,0],[mean,prior])
        np.testing.assert_allclose(table.shrink(*parts,2)[:,0],[(count*mean+2*prior)/(count+2),prior])


def test_grouped_selection_and_background_adjusted_ridge():
    from score_single_vq_readout import fit
    rng = np.random.default_rng(57)
    person = np.repeat(np.arange(12),40)
    x = rng.normal(size=(len(person),3))
    bg = rng.integers(0,3,len(x))
    y = np.column_stack((x[:,0]+4*bg, x[:,1]-bg))
    data = dict(person=person,bg=bg,id=np.where(x[:,0]>0,3,5),hard_z=x,y=y)
    dataset = {p:p//6 for p in np.unique(person)}
    _,_,ridge,coefs,_,selection = fit(data,8,['hard_z'],{'first':(0,1),'second':(1,2)},dataset)
    assert set(selection['folds'].values()) == {0,1,2}
    np.testing.assert_array_equal(selection['selected']['hard_z'],[.0001,.0001])
    assert np.max(abs(ridge['hard_z'].predict(bg,x,coefs['hard_z'])-y)) < .001
