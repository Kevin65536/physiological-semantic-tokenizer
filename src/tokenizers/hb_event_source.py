"""Train-person nonlinear Hb event allocation at a frozen integer-decoded sum."""
import numpy as np
from scipy.fft import dct,idct
from scipy.linalg import solve
from sklearn.ensemble import HistGradientBoostingRegressor

LAGS=(-8,-4,-2,-1,0,1,2,4,8)


def event_features(x,observed,base,crops,sites,array=False):
    """Only visible X, its observation flags, fixed site, and frozen X->111.

    X is [crop,time,native,2], base is [example,time,4] S/O colors.
    Window means are encoder-side deterministic functions of observed X.
    No target scoring mask enters these features.
    """
    x=np.asarray(x,float);seen=np.asarray(observed,bool);crops=np.asarray(crops,int);sites=np.asarray(sites,int)
    if x.ndim!=4 or x.shape[1]!=300 or x.shape[-1]!=2 or seen.shape!=x.shape[:-1] or base.shape!=(len(crops),300,4) or not np.isfinite(x).all():
        raise ValueError('visible300-point paired-color array and matching native examples required')
    if np.any(crops<0) or np.any(crops>=len(x)) or np.any(sites<0) or np.any(sites>=x.shape[2]):raise ValueError('native index outside fixed layout')
    own=x[crops,:,sites];valid=seen[crops,:,sites]
    own=np.where(valid[:,:,None],own,0);mean=own.sum(1,keepdims=True)/valid.sum(1).clip(1)[:,None,None]
    fields=[own,own-mean,np.asarray(base,float)];flags=[valid]
    if array:
        auxiliary=[];aflag=[]
        for crop,site in zip(crops,sites):
            m=seen[crop].copy();m[:,site]=False;count=m.sum(1);safe=np.where(m[:,:,None],x[crop],0)
            avg=safe.sum(1)/count.clip(1)[:,None];rms=np.sqrt((safe*safe).sum(1)/count.clip(1)[:,None])
            masked=np.where(m[:,:,None],x[crop],np.nan)
            # Missing native sites are natural observations, never hidden targets.
            with np.errstate(all='ignore'):
                import warnings
                with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);median=np.nanmedian(masked,axis=1)
            median=np.nan_to_num(median);z=np.c_[avg,median,rms];auxiliary.append(np.c_[z,z-z.mean(0,keepdims=True)]);aflag.append(count>0)
        fields.append(np.asarray(auxiliary));flags.append(np.asarray(aflag))
    raw=np.concatenate(fields,2);available=np.stack(flags,2);parts=[np.broadcast_to(sites[:,None,None],(len(sites),300,1)).astype(float)]
    time=np.arange(300)
    for seconds in LAGS:
        source=time-int(10*seconds);index=np.clip(source,0,299);inside=(source>=0)&(source<300)
        parts.extend((raw[:,index],available[:,index].astype(float),np.broadcast_to(inside[None,:,None],(len(sites),300,1)).astype(float)))
    result=np.concatenate(parts,2).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('nonfinite allowed input feature')
    return result


def centered(values,mask=None):
    a=np.asarray(values,float)
    if a.ndim!=3 or a.shape[1:]!=(300,2):raise ValueError('300-sample color waves required')
    if mask is None:return a-a.mean(1,keepdims=True)
    m=np.asarray(mask,bool)
    if m.shape!=a.shape:raise ValueError('same scoring support required')
    return a-(np.where(m,a,0).sum(1)/m.sum(1).clip(1))[:,None]


def project_event(delta):
    a=np.asarray(delta,float)
    if a.ndim!=3 or a.shape[1:]!=(300,2) or not np.isfinite(a).all():raise ValueError('finite300-point delta required')
    spectrum=dct(a,axis=1,norm='ortho');spectrum[:,0]=0;spectrum[:,13:]=0
    result=idct(spectrum,axis=1,norm='ortho');return result-result.mean(1,keepdims=True)


def fit_event(features,target,weight,kind,seed=230511):
    a=np.asarray(features,float);y=np.asarray(target,float);w=np.asarray(weight,float)
    if a.ndim!=2 or y.shape!=(len(a),2) or w.shape!=y.shape or not np.isfinite(a).all() or not np.isfinite(y).all() or np.any(w<0):raise ValueError('finite fit-person data and matched masked weights required')
    if kind not in ['linear','nonlinear']:raise ValueError('fixed model kind')
    common=w.sum(1);mass=common.sum()
    if mass<=0 or np.any(w.sum(0)<=0):raise ValueError('both colors need fit-person support')
    scale=np.sqrt(np.einsum('n,nf,nf->f',common/mass,a,a)).clip(1e-8);scale[0]=1
    z=a/scale;models=[];target_scale=np.sqrt((w*y*y).sum(0)/w.sum(0)).clip(1e-8);categories=np.unique(a[:,0]).astype(int)
    for c in range(2):
        good=w[:,c]>0;ww=w[good,c];ww=ww/ww.mean();yy=y[good,c]/target_scale[c]
        if kind=='linear':
            zz=np.c_[np.ones(good.sum()),a[good,0,None]==categories[None],z[good,1:]];gram=zz.T@(ww[:,None]*zz)/ww.sum();right=zz.T@(ww*yy)/ww.sum();penalty=.1*max(np.trace(gram)/len(gram),1e-12)
            models.append(solve(gram+penalty*np.eye(len(gram)),right,assume_a='pos'))
        else:
            model=HistGradientBoostingRegressor(max_iter=120,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=128,l2_regularization=10,early_stopping=False,categorical_features=[0],random_state=seed+c)
            model.fit(z[good],yy,sample_weight=ww);models.append(model)
    return dict(kind=kind,models=models,scale=scale,target_scale=target_scale,categories=categories,fit_rows=len(a),color_rows=(w>0).sum(0))


def predict_event(features,fitted):
    a=np.asarray(features,float);shape=a.shape[:-1];z=a.reshape(-1,a.shape[-1])/fitted['scale'];out=[]
    for c,m in enumerate(fitted['models']):
        v=np.c_[np.ones(len(z)),z[:,0,None]==fitted['categories'][None],z[:,1:]]@m if fitted['kind']=='linear' else m.predict(z)
        out.append(v*fitted['target_scale'][c])
    return np.stack(out,1).reshape(*shape,2)


def allocate_event(base,delta,quiet):
    d=project_event(delta);q=np.asarray(quiet,bool)
    if q.shape!=(len(d),):raise ValueError('one original quiet flag per physical window')
    d[q]=0
    s=np.asarray(base['S'],float).reshape(-1,300,2);o=np.asarray(base['O'],float).reshape(-1,300,2)
    return dict(S=(s+d).reshape(-1,15,40),O=(o-d).reshape(-1,15,40),sum=np.asarray(base['sum']).copy())
