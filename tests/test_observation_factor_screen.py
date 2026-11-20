"""Small, data-free check of rate selection, split lock, hierarchy and local readouts."""
import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.scripts import run_observation_factorization as run
from src.analysis.observation_tokenizer_evaluation import (
    KEYS, _pairs, aggregate, hierarchy_weights, paired_contrast, pattern_atlas,
    probe_attributes, reconstruction, roll_patterns)


def frame():
    rows=[]
    for ds, ns in [('d1',3), ('d2',1)]:
        for sub in range(ns):
            for rec in range(1+sub%2):
                for p in range(1+(rec==0)):
                    for j in [0,1,3,4,5,6]:
                        rows.append(dict(dataset=ds,subject=str(sub),record=str(rec),parent=rec,start_s=0,position=str(p),patch=j))
    return pd.DataFrame(rows)


def test_scope_shape_and_operating_point(tmp_path):
    import json
    cfg=run.config(run.CONFIG)
    assert json.loads(json.dumps(cfg)) == cfg  # YAML null must never silently become a different JSON key
    cfg['data']['splits'].append('test')
    p=tmp_path/'cfg.yaml';p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):run.config(p)
    t=pd.DataFrame({'level':['macro']*3,'method':['full']*3,'bits_per_patch':[14,64,128],
                    'x_nmse':[.11,.07,.09],'centered_cosine':[.9,.91,.95],'band_30_45_db':[0.,-4.,-2.]})
    assert run.operating_point(t,'eeg',cfg)['bits_per_patch']==128
    assert run.operating_point(t,'hb',cfg)['bits_per_patch']==64
    assert not run.operating_point(t.assign(centered_cosine=.7),'eeg',cfg)['met_screen']


def test_weighted_moments_equal_nested_means_and_contrasts():
    f=frame(); y=np.arange(len(f),dtype=float); w=hierarchy_weights(f)
    t=aggregate(f,{'y':y}); assert t[t.level=='macro'].iloc[0].y==pytest.approx(w@y)
    assert w[f.dataset=='d1'].sum()==pytest.approx(.5)
    a=t.assign(bits_per_patch=14,method='flat'); b=t.assign(y=t.y+.1,bits_per_patch=14,method='full')
    r=paired_contrast(pd.concat([a,b]),'y',('flat',14),('full',14))
    assert r['n_subjects']==4 and r['ci95']==pytest.approx([.1,.1]) and r['n_lower']==0


def test_masks_pairs_do_not_bridge_gaps_or_sites():
    f=frame(); i,j=_pairs(f,1)
    assert np.array_equal(f.iloc[i][KEYS].to_numpy(),f.iloc[j][KEYS].to_numpy())
    assert (f.patch.to_numpy()[j]-f.patch.to_numpy()[i]==1).all()
    roll=roll_patterns(f,np.arange(len(f)));assert np.array_equal(f[KEYS].to_numpy(), f.iloc[roll][KEYS].to_numpy())
    with pytest.raises(ValueError):_pairs(pd.concat([f,f]),1)


@pytest.mark.parametrize('mod,c,t',[('eeg',1,400),('hb',2,20)])
def test_reconstruction_and_held_out_probes(mod,c,t):
    ft=frame();fv=frame().assign(subject=lambda a:'v'+a.subject);rng=np.random.default_rng(23)
    x=rng.normal(size=(len(ft),c,t));z=rng.normal(size=(len(fv),c,t));vx=np.full((len(x),c),2.)
    s=reconstruction(ft,x,x+.2,vx,mod).query("level=='macro'").iloc[0]
    assert s.x_nmse==pytest.approx(.02) and s.centered_cosine==pytest.approx(1.)
    assert s.x_nmse==pytest.approx(s.shape_nmse+s.level_nmse)
    with pytest.raises(ValueError):reconstruction(ft,x,x,vx[:1],mod)
    d={'pattern':np.arange(len(x))%4,'wave_kmeans':np.arange(len(x))%4}
    p=probe_attributes(ft,fv,x,z,d,d,mod,4)
    assert set(p.lag_patches)=={0,1} and np.isfinite(p.pattern_gain).all()
    atlas=pattern_atlas(ft,d['pattern'],x[:,0,:2],['m','n'],5,mod,'train')
    assert atlas[(atlas.feature=='m') & (atlas.pattern_id==4)].iloc[0].n_patches==0
    assert atlas.query("feature=='m'").n_patches.sum()==len(ft)
