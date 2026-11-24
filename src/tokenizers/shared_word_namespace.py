"""Remove deterministic role tags and align the meanings of the remaining paid indices."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .cross_branch_sampling import layouts
from .cross_branch_innovations import PILOT_SIZE, pilot_positions

WIDTH=2047
STRATA=10


def split_words(ids):
    ids=np.asarray(ids)
    if ids.dtype.kind not in 'iu' or np.any((ids<0)|(ids>=15351)):raise ValueError('invalid full word')
    x=ids.astype(np.int32)
    residual=x>=PILOT_SIZE
    band=np.where(residual,(x-PILOT_SIZE)//2047,(x-1)//1023)
    atom=np.where(residual,(x-PILOT_SIZE)%2047,(x-1)%1023)
    return np.where(x==0,-1,band+5*residual),atom


def fit_alignment(ids,background,properties,seed):
    """Assign one distinct common symbol to each original word within a known stratum."""
    group,atom=split_words(ids);valid=group>=0
    group,atom,bg=group[valid],atom[valid],background[valid]
    props=np.log1p(np.column_stack((properties[valid,:2],properties[valid,2:].mean(1))))
    bg_totals=np.zeros((bg.max()+1,3));np.add.at(bg_totals,bg,props)
    residual=props-(bg_totals/np.bincount(bg,minlength=len(bg_totals)).clip(1)[:,None])[bg]
    features=np.column_stack((props,residual))
    low,high=np.quantile(features,[.01,.99],axis=0)
    features=np.clip(features,low,high)
    features=(features-features.mean(0))/features.std(0).clip(1e-12)
    counts=np.zeros((STRATA,WIDTH),int);totals=np.zeros((STRATA,WIDTH,6))
    np.add.at(counts,(group,atom),1);np.add.at(totals,(group,atom),features)
    signatures=totals/(counts[:,:,None]+8)
    pool=signatures[counts>0];rng=np.random.default_rng(seed)
    centers=pool[rng.choice(len(pool),WIDTH,replace=len(pool)<WIDTH)]
    spelling=np.zeros((STRATA,WIDTH),np.uint16)
    for s in range(STRATA):
        capacity=1023 if s<5 else WIDTH
        active=np.flatnonzero(counts[s]>0)
        rows,cols=linear_sum_assignment(cdist(signatures[s,active],centers,'sqeuclidean'))
        spelling[s,active[rows]]=cols+1
        inactive=np.flatnonzero(counts[s,:capacity]==0)
        remaining=np.setdiff1d(np.arange(1,WIDTH+1),cols+1)
        spelling[s,inactive]=remaining[:len(inactive)]
        if len(np.unique(spelling[s,:capacity]))!=capacity:raise ValueError('spelling must be injective in each physical stratum')
    return spelling


def compact(ids,spelling):
    group,atom=split_words(ids);valid=group>=0
    words=np.zeros(group.shape,np.uint16);words[valid]=spelling[group[valid],atom[valid]];return words


def expand_native(sent,spellings):
    """Use only paid missing codes, branch identity and native site order to select the inverse."""
    n=len(sent['S']);seen=np.asarray(sent['S'])[:,0]!=0
    roles=layouts(seen)['staggered'];pilots=pilot_positions(n);result={}
    for bi,b in enumerate('SO'):
        ids=np.asarray(sent[b])
        if ids.dtype.kind not in 'iu' or ids.shape!=(n,15) or np.any((ids<0)|(ids>WIDTH)) or not np.all((ids!=0)==seen[:,None]):
            raise ValueError('invalid compact native IDs')
        recovered=np.zeros_like(ids,dtype=np.uint16)
        for s in range(STRATA):
            use=seen & ((roles[b]+5*~pilots[bi*n:(bi+1)*n])==s)
            if not use.any():continue
            length=1023 if s<5 else WIDTH;inverse=np.full(WIDTH+1,-1)
            inverse[spellings[b][s,:length]]=np.arange(length)
            idx=inverse[ids[use]]
            if np.any(idx<0):raise ValueError('unused compact label for fixed native role')
            recovered[use]=idx+(1+s*1023 if s<5 else PILOT_SIZE+(s-5)*WIDTH)
        result[b]=recovered
    return result
