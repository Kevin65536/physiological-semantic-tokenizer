"""Independently bound the number of occupied levels for every complete nonconstant input."""
from pathlib import Path
import sys,json,numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path.cwd()/'experiments/scripts'))
import audit_tokenizer_observation_quality as a

def verify(i):
 w=a.WORK[i];m=w['metadata']
 ntested=0
 with np.load(m['array_file']) as z:
  for key,fs in [('eeg',200),('fnirs',10)]:
   x=z[key+'_x'].reshape(z[key+'_observation_valid'].shape[0],-1)
   mask=z[key+'_observation_valid'].reshape(x.shape)
   for t in w['score_starts_s']:
    sl=slice(t*fs,(t+30)*fs)
    for v,g in zip(x[sl].T,mask[sl].T):
     if not g.all() or not np.isfinite(v).all() or np.ptp(v)==0:continue
     if len(set(v[:16]))<5: assert len(np.unique(v))>=5
     ntested+=1
 return ntested
if __name__=='__main__':
 a.init()
 with ProcessPoolExecutor(12) as pool:n=sum(pool.map(verify,range(len(a.WORK)),chunksize=4))
 print('complete nonconstant traces with at least five distinct levels',n,flush=True)
