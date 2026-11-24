"""One paired Hb allocation correction at a frozen integer-decoded total."""
from pathlib import Path
import hashlib
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.fft import dct,idct
import torch
from .hb_native_id_readout import waveform_design, smooth_basis, load_model as load_native
from .hb_parent_distillation import validate_ids


def variance_color(bank,key):
    v=np.asarray(bank[key],float)
    return v if v.ndim==2 else v[:,0,:2]


def optimal_delta(prediction,bank):
    """Exact pointwise minimizer of the original S/O target-variance NMSE sum."""
    n=len(prediction['S']);vs=variance_color(bank,'vs')[:,None,:];vo=variance_color(bank,'vo')[:,None,:]
    if np.any(vs<=0) or np.any(vo<=0):raise ValueError('positive original target variances required')
    s,o=[np.asarray(prediction[b],float).reshape(n,300,2) for b in 'SO']
    st,ot=[np.asarray(bank[b],float).reshape(n,300,2) for b in 'so']
    mask=np.asarray(bank['mask'],bool).reshape(n,300,2)
    return np.where(mask,(vs*(ot-o)+vo*(s-st))/(vs+vo),0.)


def design(prediction,family):
    if family not in ('local','window'):raise ValueError('registered allocation feature family required')
    n=len(prediction['S']);features=waveform_design(prediction).reshape(n,15,37)
    if family=='window':
        t=np.arange(300)+.5
        basis=np.cos(np.pi*t[:,None]*np.arange(3)[None]/300)
        basis/=np.sqrt(np.mean(basis*basis,axis=0))
        waves=np.stack([np.asarray(prediction[b]).reshape(n,300,2) for b in 'SO'],axis=2)
        coefficients=np.einsum('ntbc,tk->nbck',waves,basis)/300
        center=np.cos(np.pi*(np.arange(15)*20+10)[:,None]*np.arange(3)[None]/300)
        # k1/k2 need the same sqrt(2) scaling as the full-window basis.
        center[:,1:]*=np.sqrt(2)
        global_fields=(coefficients[:,None]*center[None,:,None,None]).reshape(n,15,12)
        features=np.concatenate((features,global_fields),axis=-1)
    raw=(features[:,:,None,:,None]*smooth_basis()[None,None,:,None,:]).reshape(n,300,-1)
    # The final deterministic smoothing is part of the regressors being fitted,
    # rather than an unscored operation applied after the least-squares solve.
    return sosfiltfilt(butter(4,.4,fs=10,output='sos'),raw,axis=1)


def sufficient_statistics(ids,prediction,bank,window_weights,batch_size=64):
    arrays,quiet=validate_ids(ids);n=len(arrays['S']);mask=np.asarray(bank['mask'],bool).reshape(n,300,2)
    counts=mask.sum(1)
    vs,vo=[variance_color(bank,k) for k in ('vs','vo')]
    inv=1/vs+1/vo
    raw=np.asarray(window_weights)[:,None,None]*mask/np.maximum(counts[:,None],1)*inv[:,None]
    raw[quiet]=0
    normalization=((~quiet).sum()*15)/(raw.sum()/2)
    raw*=normalization
    gram=np.zeros((2,147,147));right=np.zeros((2,147));weight_sum=raw.sum((0,1))
    for start in range(0,n,batch_size):
        take=slice(start,start+batch_size)
        p={b:v[take] for b,v in prediction.items()}
        sub={k:v[take] for k,v in bank.items()}
        a=design(p,'window').reshape(-1,147)
        y=optimal_delta(p,sub).reshape(-1,2)
        w=raw[take].reshape(-1,2)
        for c in range(2):
            gram[c]+=a.T@(w[:,c,None]*a)
            right[c]+=a.T@(w[:,c]*y[:,c])
    scale=np.sqrt((np.diag(gram[0])+np.diag(gram[1]))/weight_sum.sum()).clip(1e-10)
    return dict(gram=gram,right=right,scale=scale,weight_sum=weight_sum,
        normalization=float(normalization),valid_points=counts.sum(0),quiet_windows=int(quiet.sum()),
        exact_fit_objective='all original mask points; hierarchy/count*(1/vs+1/vo); common normalization only')


def fit_from_statistics(stats,family,ridge):
    width={'local':111,'window':147}[family];scale=stats['scale'][:width]
    coefficients=[]
    for c in range(2):
        gram=stats['gram'][c,:width,:width]/scale[:,None]/scale[None]
        right=stats['right'][c,:width]/scale
        coefficients.append(np.linalg.solve(gram+ridge*np.eye(width),right)/scale)
    return dict(family=family,ridge=float(ridge),coefficient=np.stack(coefficients,axis=1),
        scalar_coefficients=width*2,source_total='frozen_111_head',delta_colors=['HbO','HbR'],
        fit_normalization=stats['normalization'])


def fixed_color_gains(fitted,dataset,n):
    if 'dataset_color_gains' in fitted:
        if dataset is None:raise ValueError('fixed dataset metadata required for signed allocation gain')
        labels=np.full(n,dataset) if isinstance(dataset,str) else np.asarray(dataset)
        if labels.shape!=(n,):raise ValueError('one fixed dataset identity per integer window required')
        table=fitted['dataset_color_gains']
        if any(str(label) not in table for label in labels):raise ValueError('unknown fixed dataset identity')
        gains=np.asarray([table[str(label)] for label in labels],float)
    else:gains=np.broadcast_to(np.asarray(fitted.get('global_color_gain',[1.,1.]),float),(n,2))
    if gains.shape!=(n,2) or not np.isfinite(gains).all() or np.any(abs(gains)>1):raise ValueError('bounded signed two-color gain required')
    return gains


def fit_signed_gains(delta,target,bank,window_weights,datasets,scope):
    """Fit bounded scalar corrections under the unchanged joint S/O objective."""
    n=len(delta);d=np.asarray(delta,float).reshape(n,300,2);y=np.asarray(target,float).reshape(n,300,2)
    mask=np.asarray(bank['mask'],bool).reshape(n,300,2)
    inv=1/variance_color(bank,'vs')+1/variance_color(bank,'vo')
    weight=np.asarray(window_weights)[:,None,None]*mask/np.maximum(mask.sum(1)[:,None],1)*inv[:,None]
    numerator=np.sum(weight*d*y,axis=1);denominator=np.sum(weight*d*d,axis=1)
    labels=np.asarray(datasets)
    if labels.shape!=(n,) or scope not in ('global','dataset'):raise ValueError('declared fixed dataset population required')
    result={}
    for name in (['global'] if scope=='global' else sorted(set(labels))):
        use=np.ones(n,bool) if scope=='global' else labels==name
        num,den=numerator[use].sum(0),denominator[use].sum(0)
        raw=np.divide(num,den,out=np.zeros(2),where=den>0)
        result[str(name)]=dict(gain=np.clip(raw,-1,1).tolist(),unconstrained_gain=raw.tolist(),numerator=num.tolist(),denominator=den.tolist())
    return result


SPECTRAL_GROUPS=((0,1),(1,3),(3,13))


def spectral_design(prediction):
    """Three disjoint full30s DCT bands, each with four decoded S/O-color inputs."""
    n=len(prediction['S'])
    waves=np.stack([np.asarray(prediction[b],float).reshape(n,300,2) for b in 'SO'],axis=2).reshape(n,300,4)
    spectrum=dct(waves,type=2,axis=1,norm='ortho')
    result=[]
    for lo,hi in SPECTRAL_GROUPS:
        part=np.zeros_like(spectrum);part[:,lo:hi]=spectrum[:,lo:hi]
        result.append(idct(part,type=2,axis=1,norm='ortho'))
    return np.concatenate(result,axis=2)


def spectral_statistics(ids,prediction,bank,window_weights,datasets,batch_size=256):
    arrays,quiet=validate_ids(ids);n=len(arrays['S']);labels=np.asarray(datasets)
    if labels.shape!=(n,):raise ValueError('one fixed train dataset per spectral fitting window required')
    stats={str(d):dict(gram=np.zeros((2,12,12)),right=np.zeros((2,12)),weight_sum=np.zeros(2),windows=int(np.sum(labels==d))) for d in sorted(set(labels))}
    inverse=1/variance_color(bank,'vs')+1/variance_color(bank,'vo')
    for start in range(0,n,batch_size):
        take=slice(start,start+batch_size);p={b:v[take] for b,v in prediction.items()};sub={k:v[take] for k,v in bank.items()}
        features=spectral_design(p);target=optimal_delta(p,sub);mask=np.asarray(sub['mask'],bool).reshape(-1,300,2)
        weight=np.asarray(window_weights)[take,None,None]*mask/np.maximum(mask.sum(1)[:,None],1)*inverse[take,None]
        weight[quiet[take]]=0
        for dataset in sorted(set(labels[take])):
            chosen=labels[take]==dataset;a=features[chosen].reshape(-1,12);y=target[chosen].reshape(-1,2);w=weight[chosen].reshape(-1,2);s=stats[str(dataset)]
            s['weight_sum']+=w.sum(0)
            for color in range(2):
                s['gram'][color]+=a.T@(w[:,color,None]*a);s['right'][color]+=a.T@(w[:,color]*y[:,color])
    return stats


def fit_spectral(stats,scope,ridge):
    if scope not in ('global','dataset') or ridge<=0:raise ValueError('registered shrunk spectral scope required')
    groups=stats if scope=='dataset' else {'global':{k:sum(v[k] for v in stats.values()) for k in ('gram','right','weight_sum')}}
    fitted={}
    for dataset,s in groups.items():
        if np.any(s['weight_sum']<=0):raise ValueError('both colors need positive fit-only spectral support')
        gram=s['gram']/s['weight_sum'][:,None,None];right=s['right']/s['weight_sum'][:,None]
        scale=np.sqrt((np.diag(gram[0])+np.diag(gram[1]))/2).clip(1e-10)
        columns=[]
        for color in range(2):
            matrix=gram[color]/scale[:,None]/scale[None]
            penalty=ridge*max(float(np.trace(matrix)/12),1e-12)
            columns.append(np.linalg.solve(matrix+penalty*np.eye(12),right[color]/scale)/scale)
        fitted[str(dataset)]=np.stack(columns,axis=1)
    return dict(family='whole_window_spectral',scope=scope,ridge=float(ridge),coefficients=fitted,groups=[list(v) for v in SPECTRAL_GROUPS],
        scalar_coefficients=sum(v.size for v in fitted.values()),normalization='fit-only uncentered feature RMS; color weights retain original relative window/time weighting')


def apply_spectral(ids,prediction,fitted,dataset=None,batch_size=256,groups=None):
    arrays,quiet=validate_ids(ids);n=len(arrays['S']);table=fitted['coefficients']
    if fitted.get('groups')!=[list(v) for v in SPECTRAL_GROUPS]:raise ValueError('declared whole30s spectral bands changed')
    labels=np.full(n,'global') if fitted['scope']=='global' else (np.full(n,dataset) if isinstance(dataset,str) else np.asarray(dataset))
    if labels.shape!=(n,) or any(str(d) not in table for d in labels):raise ValueError('known fixed dataset metadata required for spectral allocation')
    output={b:np.asarray(prediction[b],float).copy() for b in 'SO'}|{'sum':np.asarray(prediction['sum']).copy()}
    active=range(3) if groups is None else groups
    for start in range(0,n,batch_size):
        take=slice(start,start+batch_size);a=spectral_design({b:v[take] for b,v in prediction.items()})
        delta=np.zeros((len(a),300,2))
        for name in sorted(set(labels[take])):
            chosen=labels[take]==name;coefficient=np.asarray(table[str(name)],float).copy()
            if coefficient.shape!=(12,2):raise ValueError('four-input two-color three-band spectral coefficients required')
            for group in range(3):
                if group not in active:coefficient[group*4:(group+1)*4]=0
            delta[chosen]=a[chosen]@coefficient
        delta[quiet[take]]=0;delta=delta.reshape(-1,15,40)
        output['O'][take]+=delta;output['S'][take]-=delta
    for b in output:output[b][quiet]=prediction[b][quiet]
    return output


def spectral_group_delta(ids,prediction,fitted,dataset):
    """Return the three fixed correction directions; quiet remains identically zero."""
    _,quiet=validate_ids(ids);n=len(quiet);labels=np.asarray(dataset)
    if labels.shape!=(n,) or fitted['scope']!='dataset':raise ValueError('dataset spectral directions required')
    features=spectral_design(prediction);delta=np.zeros((n,300,2,3))
    for name in sorted(set(labels)):
        take=labels==name;coef=np.asarray(fitted['coefficients'][str(name)])
        for group in range(3):delta[take,:,:,group]=features[take,:,4*group:4*group+4]@coef[4*group:4*group+4]
    delta[quiet]=0
    return delta


def phase_design(prediction,space='odd'):
    """Non-circular quadrature: DCT-II cosine coefficients into sine companions.

    The physical sample centers are q=.5,...,299.5. Sine companions vanish
    at q=0 and q=300, corresponding to virtual edges at -.05 and29.95s.
    DC has no quadrature. Two groups each share four decoded input directions.
    """
    if space not in ('even','odd','even_plus_odd'):raise ValueError('declared phase space required')
    n=len(prediction['S']);waves=np.stack([np.asarray(prediction[b],float).reshape(n,300,2) for b in 'SO'],axis=2).reshape(n,300,4)
    coefficients=dct(waves,type=2,axis=1,norm='ortho');parts=[]
    for lo,hi in [(1,3),(3,13)]:
        basis=np.sqrt(2/300)*np.sin(np.pi*(np.arange(300)+.5)[:,None]*np.arange(lo,hi)[None]/300)
        parts.append(np.einsum('tk,nkc->ntc',basis,coefficients[:,lo:hi]))
    odd=np.concatenate(parts,axis=2)
    if space=='odd':return odd
    even=spectral_design(prediction)[:,:,4:]
    return even if space=='even' else np.concatenate((even,odd),axis=2)


def phase_oracle(ids,prediction,bank,space,batch_size=128):
    """Target-informed per-window upper bound; coefficients are continuous truth leakage.

    This diagnostic can never be used as an integer-only inference model.
    Every supplied coefficient is fitted from that same window's S/O labels.
    """
    _,quiet=validate_ids(ids);n=len(quiet);width=16 if space=='even_plus_odd' else 8
    output={b:np.asarray(prediction[b],float).copy() for b in ('S','O','sum')};coef=np.zeros((n,2,width));rank=np.zeros((n,2),int)
    for start in range(0,n,batch_size):
        take=slice(start,start+batch_size);p={b:v[take] for b,v in prediction.items()};sub={k:v[take] for k,v in bank.items()}
        a=phase_design(p,space);target=-optimal_delta(p,sub);mask=np.asarray(sub['mask'],bool).reshape(-1,300,2);delta=np.zeros_like(target)
        for row in range(len(a)):
            if quiet[start+row]:continue
            scale=np.sqrt(np.mean(a[row]*a[row],axis=0));live=scale>max(float(scale.max())*1e-10,np.finfo(float).tiny)
            for color in range(2):
                valid=mask[row,:,color]
                if not valid.any() or not live.any():continue
                matrix=a[row,valid][:,live]/scale[live]
                beta,_,r,_=np.linalg.lstsq(matrix,target[row,valid,color],rcond=1e-10)
                coef[start+row,color,live]=beta/scale[live];rank[start+row,color]=r
                delta[row,:,color]=a[row]@coef[start+row,color]
        delta=delta.reshape(-1,15,40);output['S'][take]+=delta;output['O'][take]-=delta
    for b in output:output[b][quiet]=prediction[b][quiet]
    return output,coef,rank


def phase_statistics(ids,prediction,bank,window_weights,datasets,batch_size=128):
    """Fit-only sufficient statistics for the single registered complex transfer."""
    _,quiet=validate_ids(ids);labels=np.asarray(datasets);n=len(quiet)
    if labels.shape!=(n,):raise ValueError('one fixed dataset per phase window required')
    stats={str(d):dict(gram=np.zeros((2,16,16)),right=np.zeros((2,16)),weight_sum=np.zeros(2)) for d in sorted(set(labels))}
    inverse=1/variance_color(bank,'vs')+1/variance_color(bank,'vo')
    for start in range(0,n,batch_size):
        take=slice(start,start+batch_size);p={b:v[take] for b,v in prediction.items()};sub={k:v[take] for k,v in bank.items()}
        features=phase_design(p,'even_plus_odd');target=-optimal_delta(p,sub);mask=np.asarray(sub['mask'],bool).reshape(-1,300,2)
        weight=np.asarray(window_weights)[take,None,None]*mask/np.maximum(mask.sum(1)[:,None],1)*inverse[take,None];weight[quiet[take]]=0
        for dataset in sorted(set(labels[take])):
            chosen=labels[take]==dataset;a=features[chosen].reshape(-1,16);y=target[chosen].reshape(-1,2);w=weight[chosen].reshape(-1,2);s=stats[str(dataset)]
            s['weight_sum']+=w.sum(0)
            for color in range(2):s['gram'][color]+=a.T@(w[:,color,None]*a);s['right'][color]+=a.T@(w[:,color]*y[:,color])
    return stats


def fit_phase(stats,ridge=1.):
    if ridge!=1.:raise ValueError('single predeclared phase shrinkage required')
    table={}
    for dataset,s in stats.items():
        if np.any(s['weight_sum']<=0):raise ValueError('positive phase fit support required')
        gram=s['gram']/s['weight_sum'][:,None,None];right=s['right']/s['weight_sum'][:,None]
        scale=np.sqrt((np.diag(gram[0])+np.diag(gram[1]))/2).clip(1e-10);columns=[]
        for color in range(2):
            matrix=gram[color]/scale[:,None]/scale[None];penalty=ridge*max(float(np.trace(matrix)/16),1e-12)
            columns.append(np.linalg.solve(matrix+penalty*np.eye(16),right[color]/scale)/scale)
        table[dataset]=np.stack(columns,axis=1)
    return dict(family='whole_window_even_odd_phase',scope='dataset',ridge=1.,coefficients=table,scalar_coefficients=sum(v.size for v in table.values()),
        groups=['even_slow','even_dynamic','odd_slow','odd_dynamic'],normalization='fit-only uncentered feature RMS; exact original joint window/time weights')


def phase_group_delta(ids,prediction,fitted,dataset):
    _,quiet=validate_ids(ids);n=len(quiet);labels=np.full(n,dataset) if isinstance(dataset,str) else np.asarray(dataset)
    if labels.shape!=(n,) or fitted['family']!='whole_window_even_odd_phase' or any(str(d) not in fitted['coefficients'] for d in labels):raise ValueError('known dataset phase model required')
    features=phase_design(prediction,'even_plus_odd');delta=np.zeros((n,300,2,4))
    for name in sorted(set(labels)):
        take=labels==name;coef=np.asarray(fitted['coefficients'][str(name)])
        if coef.shape!=(16,2):raise ValueError('four-input two-color even/odd two-band coefficients required')
        for group in range(4):delta[take,:,:,group]=features[take,:,4*group:4*group+4]@coef[4*group:4*group+4]
    delta[quiet]=0
    return delta


def apply_phase(ids,prediction,fitted,dataset,batch_size=128):
    _,quiet=validate_ids(ids);n=len(quiet);labels=np.full(n,dataset) if isinstance(dataset,str) else np.asarray(dataset)
    output={b:np.asarray(prediction[b],float).copy() for b in ('S','O','sum')}
    for start in range(0,n,batch_size):
        take=slice(start,start+batch_size);d=phase_group_delta({b:v[take] for b,v in ids.items()},{b:v[take] for b,v in prediction.items()},fitted,labels[take]).sum(-1).reshape(-1,15,40)
        output['S'][take]+=d;output['O'][take]-=d
    for b in output:output[b][quiet]=prediction[b][quiet]
    return output


def protection_moments(prediction,bank,directions,color,window_weights,finite_weights=None):
    """Exact per-window native-coordinate moments for fixed group directions."""
    n=len(directions);d=np.asarray(directions)[:,:,color,:];mask=np.asarray(bank['mask'],bool).reshape(n,300,2)[:,:,color]
    weights=np.asarray(window_weights,float);weights=weights/weights.sum()
    mean=lambda x,m:np.sum(np.where(m[(...,)+(None,)*(x.ndim-m.ndim)],x,0),axis=1)/np.maximum(m.sum(1),1)[(...,)+(None,)*(x.ndim-m.ndim)]
    def quad(p,y,v,valid):
        return dict(energy=mean(p*p,valid),target_energy=mean(y*y,valid),linear=mean(p[:,:,None]*v,valid),gram=mean(v[:,:,:,None]*v[:,:,None,:],valid))
    branches={};t=(np.arange(20)-9.5)/10;q=t*t-np.mean(t*t)
    for branch,key,sign in [('S','s',-1),('O','o',1)]:
        p=np.asarray(prediction[branch],float).reshape(n,300,2)[:,:,color];y=np.asarray(bank[key],float).reshape(n,300,2)[:,:,color]
        moments=quad(p,y,d,mask);moments['sign']=sign;moments['correlations']={}
        finite=moments['target_energy']>0
        moments['rms_weights']=finite_weights(finite) if finite_weights else weights[finite]/weights[finite].sum()
        error=p-y;variance=variance_color(bank,'v'+key)[:,color]
        moments['error']=mean(error*error,mask)/variance
        moments['error_linear']=mean(error[:,:,None]*d,mask)/variance[:,None]
        moments['error_gram']=moments['gram']/variance[:,None,None]
        full=mask.reshape(n,15,20).all(2)
        for name,basis in [('slope',t/(t@t)),('curvature',2*q/(q@q))]:
            ap=np.einsum('npt,t->np',p.reshape(n,15,20),basis);ay=np.einsum('npt,t->np',y.reshape(n,15,20),basis)
            ad=np.einsum('nptg,t->npg',d.reshape(n,15,20,d.shape[-1]),basis)
            ap=ap-mean(ap,full)[:,None];ay=ay-mean(ay,full)[:,None];ad=ad-mean(ad,full)[:,None]
            cm=quad(ap,ay,ad,full);cm['covariance']=mean(ap*ay,full);cm['target_linear']=mean(ay[:,:,None]*ad,full)
            cm['finite']=(cm['energy']*cm['target_energy']>1e-20)&(full.sum(1)>0)
            f=cm['finite'];cm['weights']=finite_weights(f) if finite_weights and f.any() else weights[f]/max(weights[f].sum(),1e-300)
            moments['correlations'][name]=cm
        branches[branch]=moments
    return dict(branches=branches,weights=weights,windows=n,color=color)


def protection_values(moments,gain):
    """Original hierarchical RMS/correlations and joint loss from sufficient moments."""
    gain=np.asarray(gain,float);weight=moments['weights'];out={};loss=0.
    for branch,m in moments['branches'].items():
        sign=m['sign'];energy=m['energy']+2*sign*(m['linear']@gain)+np.einsum('g,ngh,h->n',gain,m['gram'],gain)
        rms=np.sqrt(np.maximum(energy,0)/np.maximum(m['target_energy'],1e-300));finite=m['target_energy']>0
        out[branch+'_rms']=float(m['rms_weights']@rms[finite])
        loss+=float(weight@(m['error']+2*sign*(m['error_linear']@gain)+np.einsum('g,ngh,h->n',gain,m['error_gram'],gain)))
        for name,cm in m['correlations'].items():
            e=cm['energy']+2*sign*(cm['linear']@gain)+np.einsum('g,ngh,h->n',gain,cm['gram'],gain)
            cov=cm['covariance']+sign*(cm['target_linear']@gain);finite=cm['finite']
            corr=cov/np.sqrt(np.maximum(e*cm['target_energy'],1e-300))
            out[branch+'_'+name]=float(cm['weights']@corr[finite]) if finite.any() else None
    out['joint_loss']=loss
    return out


def fit_protected_gains(moments,maxiter=120,protection_reference=None):
    """One deterministic fit-only constrained solve, with zero as feasible fallback."""
    from scipy.optimize import minimize
    width=moments['branches']['S']['linear'].shape[-1];zero=np.zeros(width)
    baseline=protection_values(moments,zero);reference=baseline if protection_reference is None else protection_reference
    scale=max(baseline['joint_loss'],1e-12);best=[zero.copy(),baseline['joint_loss']]
    bounds={}
    for branch in 'SO':
        rms=reference[branch+'_rms'];lo,hi=(.75,1.25) if .75<=rms<=1.25 else (1-abs(rms-1)-.005,1+abs(rms-1)+.005)
        bounds[branch+'_rms']=(max(0,lo),hi)
        for name in ['slope','curvature']:
            v=reference[branch+'_'+name]
            if v is not None:bounds[branch+'_'+name]=(v-.01,None)
    def constraints(gain):
        values=protection_values(moments,gain);res=[]
        for key,(lo,hi) in bounds.items():
            res.append(values[key]-lo)
            if hi is not None:res.append(hi-values[key])
        return np.asarray(res)
    def objective(gain):return protection_values(moments,gain)['joint_loss']/scale
    def keep(gain):
        value=objective(gain)*scale
        if np.min(constraints(gain))>=-1e-9 and value<best[1]:best[:]=[np.asarray(gain).copy(),value]
    if np.min(constraints(zero)) < -1e-8:raise ValueError('zero phase increment violates original fit-only protection reference')
    result=minimize(objective,zero,method='SLSQP',bounds=[(0.,1.)]*width,constraints=[dict(type='ineq',fun=constraints)],callback=keep,
        options=dict(maxiter=maxiter,ftol=1e-10));keep(result.x)
    gains=best[0]
    return dict(gain=gains.tolist(),baseline=baseline,protection_reference=reference,protected=protection_values(moments,gains),bounds=bounds,
        minimum_constraint=float(np.min(constraints(gains))),optimizer_success=bool(result.success),optimizer_message=str(result.message),iterations=int(result.nit),
        fallback_zero=bool(np.all(gains==0)),selection='minimum objective among feasible solver iterates; deterministic start zero; fit people only')


def apply_pair(ids,prediction,fitted,batch_size=128,dataset=None):
    arrays,quiet=validate_ids(ids);n=len(arrays['S'])
    gains=fixed_color_gains(fitted,dataset,n) if 'dataset_color_gains' in fitted or 'global_color_gain' in fitted else None
    output={b:np.asarray(prediction[b],float).copy() for b in ('S','O')}
    output['sum']=np.asarray(prediction['sum']).copy()
    for start in range(0,n,batch_size):
        stop=min(n,start+batch_size);p={b:v[start:stop] for b,v in prediction.items()}
        delta=(design(p,fitted['family'])@np.asarray(fitted['coefficient'])).reshape(stop-start,15,40)
        if gains is not None:delta=(delta.reshape(stop-start,300,2)*gains[start:stop,None]).reshape(stop-start,15,40)
        delta[quiet[start:stop]]=0
        output['O'][start:stop]=np.asarray(p['O'],float)+delta
        output['S'][start:stop]=np.asarray(p['S'],float)-delta
    for b in output:output[b][quiet]=prediction[b][quiet]
    return output


class HbPairAllocation:
    def __init__(self,source,metadata,fitted):self.source,self.metadata,self.fitted=source,metadata,fitted
    def encode(self,x):return self.source.encode(x)
    def decode_ids(self,ids,dataset=None):return apply_pair(ids,self.source.decode_ids(ids),self.fitted,dataset=dataset)


class HbSpectralAllocation(HbPairAllocation):
    def decode_ids(self,ids,dataset=None):return apply_spectral(ids,self.source.decode_ids(ids),self.fitted,dataset=dataset)


class HbPhaseAllocation(HbPairAllocation):
    def __init__(self,source,metadata,fitted,protected_fitted):
        # Derive counts from frozen arrays; inherited source metadata describes
        # the pre-phase readout and its historical128/256 base alphabets.
        meta=dict(metadata);phase_count=sum(np.asarray(v).size for v in fitted['coefficients'].values())
        protected_count=sum(np.asarray(v).size for v in protected_fitted['coefficients'].values())
        native_count=np.asarray(source.fitted['coefficient']).size if hasattr(source,'fitted') else 0
        meta.update(phase_scalar_coefficients=phase_count,extra_allocation_scalar_coefficients=protected_count+phase_count,
            extra_scalar_coefficients=native_count+protected_count+phase_count,
            bits_per_branch_position_per_second={'S':4.,'O':4.5},fixed_width_bits_per_position_per_second=8.5)
        super().__init__(source,meta,fitted);self.protected_fitted=protected_fitted
    def decode_ids(self,ids,dataset=None):
        p=apply_spectral(ids,self.source.decode_ids(ids),self.protected_fitted,dataset=dataset)
        return apply_phase(ids,p,self.fitted,dataset)


def load_model(path):
    path=Path(path);saved=torch.load(path,map_location='cpu',weights_only=False);meta=saved['metadata']
    if meta['schema'] not in ('hb_pair_allocation_v1','hb_pair_allocation_gain_v1','hb_spectral_allocation_v1','hb_spectral_protected_v1','hb_phase_fit_v1'):raise ValueError('versioned paired allocation model required')
    source=(path.parent/meta['source_file']).resolve()
    if source.parent!=path.parent.resolve() or hashlib.sha256(source.read_bytes()).hexdigest()!=meta['source_sha256']:
        raise ValueError('frozen 111-head source identity changed')
    native=load_native(source)
    if 'source111_fold_override' in saved:
        native.fitted=saved['source111_fold_override']
    if meta['schema']=='hb_phase_fit_v1':return HbPhaseAllocation(native,meta,saved['fitted'],saved['protected_fitted'])
    return (HbSpectralAllocation if meta['schema'] in ('hb_spectral_allocation_v1','hb_spectral_protected_v1') else HbPairAllocation)(native,meta,saved['fitted'])
