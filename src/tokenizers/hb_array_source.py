"""Train-only supervised low-rank space/time Hb allocation from visible arrays."""
import numpy as np
from scipy.linalg import svd,solve


LAGS=(-8,-4,-2,-1,0,1,2,4,8)


def lagged_array(values,observed,lags=LAGS):
    """Visible30s only; positive lag reads earlier time, no circular crossing."""
    x=np.asarray(values,float);m=np.asarray(observed,bool)
    if x.ndim!=3 or x.shape[0]!=300 or m.shape!=x.shape[:2] or not np.isfinite(x).all():
        raise ValueError('finite300-time by native-position by feature array and mask required')
    if tuple(lags)!=LAGS:raise ValueError('registered non-circular lag support required')
    parts=[]
    for seconds in lags:
        shift=int(seconds*10);a=np.zeros_like(x);valid=np.zeros_like(m)
        if shift>0:a[shift:]=x[:-shift];valid[shift:]=m[:-shift]
        elif shift<0:a[:shift]=x[-shift:];valid[:shift]=m[-shift:]
        else:a[:]=x;valid[:]=m
        parts.extend((np.where(valid[:,:,None],a,0).reshape(300,-1),valid.astype(float)))
    return np.c_[np.ones(300),*parts].astype(np.float32)


def local_columns(sites,channels,position):
    width=sites*(channels+1);cols=[0]
    for lag in range(len(LAGS)):
        offset=1+lag*width
        cols.extend(range(offset+position*channels,offset+(position+1)*channels));cols.append(offset+sites*channels+position)
    return np.asarray(cols,int)


def fit_transfer(features,target,weight,rank=8,ridge=.01,local=None):
    """Ridge-preconditioned predictive directions, then exact masked weighted ridge.

    Low-rank directions are learned jointly over space and time using fit
    targets. The readout retains each output's original hierarchy/variance mask.
    This is a supervised subspace estimator, not an instantaneous input PCA.
    """
    a=np.asarray(features,float);y=np.asarray(target,float);w=np.asarray(weight,float)
    if a.ndim!=2 or y.ndim!=2 or y.shape!=w.shape or len(a)!=len(y) or not np.isfinite(a).all() or not np.isfinite(y).all() or np.any(w<0):
        raise ValueError('finite matched fit features/targets and nonnegative masked weights required')
    if rank!=8 or ridge!=.01:raise ValueError('single predeclared supervised rank/ridge required')
    mass=w.sum(0);common=w.mean(1);total=common.sum()
    if total<=0:raise ValueError('positive train support required')
    scale=np.sqrt(np.einsum('n,nf,nf->f',common/total,a,a)).clip(1e-8);normalized=a/scale
    if local is None:
        cross=normalized.T@(w*y)/np.sqrt(np.maximum(mass,1))[None]
        gram=normalized.T@((common/total)[:,None]*normalized)
        penalty=ridge*max(float(np.trace(gram)/len(gram)),1e-12)
        predictive=solve(gram+penalty*np.eye(len(gram)),cross,assume_a='pos',check_finite=False)
        u,s,_=svd(predictive,full_matrices=False,check_finite=False);active=min(rank,int(np.sum(s>max(float(s[0])*1e-10,1e-20))))
        projection=u[:,:active]/scale[:,None];z=a@projection;coeff=np.zeros((active,y.shape[1]))
        for c in range(y.shape[1]):
            if mass[c]<=0:continue
            g=z.T@(w[:,c,None]*z)/mass[c];r=z.T@(w[:,c]*y[:,c])/mass[c]
            penalty=ridge*max(float(np.trace(g)/max(active,1)),1e-12);coeff[:,c]=np.linalg.solve(g+penalty*np.eye(active),r)
        return dict(projection=projection,readout=coeff,scale=scale,rank=active,singular_values=s,mode='supervised_space_time',ridge=ridge)
    sites,channels=local
    if y.shape[1]!=sites*2:raise ValueError('paired-color local outputs required')
    coefficient=np.zeros((a.shape[1],y.shape[1]))
    for site in range(sites):
        cols=local_columns(sites,channels,site);z=normalized[:,cols]
        for c in [2*site,2*site+1]:
            if mass[c]<=0:continue
            g=z.T@(w[:,c,None]*z)/mass[c];r=z.T@(w[:,c]*y[:,c])/mass[c]
            penalty=ridge*max(float(np.trace(g)/len(g)),1e-12)
            coefficient[cols,c]=np.linalg.solve(g+penalty*np.eye(len(g)),r)/scale[cols]
    return dict(coefficient=coefficient,scale=scale,rank=None,mode='same_site_space_time',ridge=ridge)


def predict_transfer(features,fitted):
    a=np.asarray(features,float)
    return a@fitted['coefficient'] if fitted['mode']=='same_site_space_time' else (a@fitted['projection'])@fitted['readout']


def allocate(base,delta,quiet):
    d=np.asarray(delta,float).copy();d[np.asarray(quiet,bool)]=0
    result=dict(S=np.asarray(base['S'],float)+d,O=np.asarray(base['O'],float)-d,sum=np.asarray(base['sum']).copy())
    for b in result:result[b][quiet]=base[b][quiet]
    return result
