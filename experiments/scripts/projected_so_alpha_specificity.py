"""Post hoc specificity of the two previously selected alpha IDs on identical matches."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.scripts.audit_projected_so_members import attributes, CFG, OUT, SOURCE
from src.data.so_supervision_cache import write_json


def run():
    meta = pd.read_csv(SOURCE/'examples.csv',dtype={'subject':str})
    ex, block = np.where(np.load(SOURCE/'mask.npy') & (meta.role!='shift').to_numpy()[:,None])
    m = meta.iloc[ex].reset_index(drop=True)
    x, px = (np.load(SOURCE/(k+'.npy'),mmap_mode='r').reshape(-1,15,400) for k in ['x','projected_x'])
    design = np.empty((len(m),5))
    target = np.empty(len(m))
    for ix in np.array_split(np.arange(len(m)),100):
        y, p = x[ex[ix],block[ix]], px[ex[ix],block[ix]]
        _, (target[ix], _) = attributes(y)
        _, (alpha, _) = attributes(p)
        energy = np.mean(y**2,axis=1)
        ratio = np.mean(p**2,axis=1)/energy
        design[ix] = np.column_stack((np.ones(len(ix)),alpha,ratio,alpha*ratio,.5*np.log(energy)))
    residual = np.empty(len(m))
    for _, g in m.groupby(['dataset','site']):
        train = g[g.role=='train'].index
        assert len(train) >= 5
        beta = np.linalg.lstsq(design[train],target[train],rcond=1e-10)[0]
        residual[g.index] = target[g.index] - design[g.index] @ beta
    val = pd.read_csv(OUT/'validation_members.csv.gz',index_col='row')
    np.testing.assert_array_equal(np.c_[ex[val.index],block[val.index]],val[['example','block']].values)
    np.testing.assert_allclose(target[val.index],val.alpha,atol=1e-10)
    matches = pd.read_csv(OUT/'matches.csv').query("endpoint=='alpha'").copy()
    a, b = matches.member_row, matches.control_row
    matches['person'] = val.loc[a,'person'].to_numpy()
    matches['raw'] = target[a]-target[b]
    matches['residual'] = residual[a]-residual[b]
    grouped = matches.groupby(['branch','token','person']).agg(n=('raw','size'),raw=('raw','mean'),residual=('residual','mean'))
    grouped.to_csv(OUT/'alpha_specificity.csv')
    write_json(OUT/'resolved_config.json',CFG)


if __name__=='__main__':
    run()
