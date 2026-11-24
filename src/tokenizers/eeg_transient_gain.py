"""Observed-amplitude conditional, bounded gains of the frozen EEG Wiener source.

This continuous source estimator precedes integer assignment. It is not an
integer decoder and accepts no teacher references or per-window hidden state.
"""
import json
from pathlib import Path
import numpy as np

BANDS=((1,4),(4,8),(8,13),(13,30),(30,45))
FEATURES=('constant','window_vs_training','window_vs_spatial',
          'local_transient','positive_transient','positive_transient_squared',
          'broadband_transient','positive_broadband_squared','common_transient',
          'local_vs_common','coherence_change','envelope_change','background_x_transient')


def band_indices():
    hz=np.fft.rfftfreq(6000,1/200)
    return [np.flatnonzero((hz>=lo)&((hz<=hi) if hi==45 else (hz<hi))) for lo,hi in BANDS]


def prepare_x(x,observed,transfer):
    x=np.asarray(x,float);observed=np.asarray(observed,bool)
    if x.shape!=(6000,len(observed)):raise ValueError('one 30s native 200Hz array required')
    wave=np.where(observed[:,None],x.T,0.)
    if not np.isfinite(wave).all():raise ValueError('finite observed X required')
    spectrum=np.fft.rfft(wave,axis=-1)
    prior={b:np.fft.irfft(spectrum*np.asarray(transfer[b]).reshape(1,3001),n=6000,axis=-1) for b in 'SO'}
    prior['sum']=prior['S']+prior['O']
    amplitudes=[];carriers=[];coherence=[]
    for indices in band_indices():
        analytic_spectrum=np.zeros((len(observed),6000),complex)
        analytic_spectrum[:,indices]=2*spectrum[:,indices]
        z=np.fft.ifft(analytic_spectrum,axis=-1)
        energy=(abs(z)**2).reshape(len(observed),300,20).mean(-1)/2
        amplitudes.append(np.sqrt(energy))
        if observed.any():
            common=z[observed].mean(0)
            numerator=(abs(common)**2).reshape(300,20).mean(-1)
            denominator=(abs(z[observed])**2).reshape(observed.sum(),300,20).mean((0,2))
            coherence.append(np.sqrt(numerator/denominator.clip(1e-16)).clip(0,1))
        else:coherence.append(np.zeros(300))
        selected=np.zeros_like(spectrum);selected[:,indices]=spectrum[:,indices]*np.asarray(transfer['S']).reshape(3001)[indices]
        carriers.append(np.fft.irfft(selected,n=6000,axis=-1))
    amplitude=np.asarray(amplitudes)
    return dict(prior=prior,amplitude=amplitude,carriers=np.asarray(carriers),
                coherence=np.asarray(coherence),window_rms=np.sqrt(np.mean(amplitude**2,axis=-1)),observed=observed)


def signed_log_ratio(a,b):
    return np.clip(np.log2(np.maximum(a,1e-8)/np.maximum(b,1e-8)),-4,4)/4


def feature_field(prepared,background,band,transient=True):
    amp=prepared['amplitude'][band];win=prepared['window_rms'][band];seen=prepared['observed']
    common=np.median(amp[seen],axis=0) if seen.any() else np.ones(300)
    spatial=float(np.median(win[seen])) if seen.any() else 1.
    broad=np.sqrt(np.sum(prepared['amplitude']**2,axis=0))
    broadwin=np.sqrt(np.sum(prepared['window_rms']**2,axis=0))
    phi=np.zeros((*amp.shape,len(FEATURES)));phi[...,0]=1
    phi[...,1]=signed_log_ratio(win,background[band])[:,None]
    phi[...,2]=signed_log_ratio(win,spatial)[:,None]
    if transient:
        own=signed_log_ratio(amp,win[:,None]);positive=np.maximum(own,0)
        bb=signed_log_ratio(broad,broadwin[:,None]);coh=prepared['coherence'][band]
        phi[...,3]=own;phi[...,4]=positive;phi[...,5]=positive**2
        phi[...,6]=bb;phi[...,7]=np.maximum(bb,0)**2
        phi[...,8]=signed_log_ratio(common,spatial)[None]
        phi[...,9]=signed_log_ratio(amp,common[None])
        phi[...,10]=(coh-coh.mean())[None]
        logamp=np.log2(amp.clip(1e-8))
        phi[...,11]=np.clip(abs(np.gradient(logamp,axis=-1))/4,0,1)
        phi[...,12]=phi[...,1]*positive
    return phi


def weighted_ridge_gain(gram,cross,ridge=.1):
    """Quadratic frame-energy surrogate; ridge shrinks toward the unchanged source."""
    beta=np.zeros(cross.shape);beta[...,0]=1
    for b in range(len(gram)):
        for p,matrix in enumerate(gram[b]):
            diag=matrix.diagonal().clip(0)
            if diag[0]<=1e-14:continue
            penalty=ridge*np.maximum(diag,max(diag.mean(),1e-14)*1e-6)
            anchor=np.zeros(len(diag));anchor[0]=1
            beta[b,p]=np.linalg.solve(matrix+np.diag(penalty),cross[b,p]+penalty*anchor)
    if not np.isfinite(beta).all():raise ValueError('nonfinite source coefficients')
    return beta


def fit_background(items):
    shape=items[0]['prepared']['window_rms'].shape
    total=np.zeros(shape);mass=np.zeros(shape)
    for item in items:
        p=item['prepared'];w=item['weight']*p['observed'][None]
        total += w*np.log(p['window_rms'].clip(1e-8));mass += np.broadcast_to(w,shape)
    result=np.exp(total/mass.clip(1e-12))
    for b in range(5):
        valid=mass[b]>0
        result[b,~valid]=np.median(result[b,valid]) if valid.any() else 1.
    return result


def fit_gain(items,background,transient=True,ridge=.1):
    channels=background.shape[1];width=len(FEATURES)
    gram=np.zeros((5,channels,width,width));cross=np.zeros((5,channels,width))
    for item in items:
        for band in range(5):
            phi=feature_field(item['prepared'],background,band,transient)
            a=item['frame_energy'][band]*item['weight']/300
            b=item['frame_cross'][band]*item['weight']/300
            gram[band] += np.einsum('cti,ctj,ct->cij',phi,phi,a,optimize=True)
            cross[band] += np.einsum('cti,ct->ci',phi,b,optimize=True)
    return weighted_ridge_gain(gram,cross,ridge)


def apply_gain(prepared,background,beta,transient=True,bounds=(0.,3.)):
    correction=np.zeros_like(prepared['prior']['S']);gains=[]
    for b,indices in enumerate(band_indices()):
        phi=feature_field(prepared,background,b,transient)
        gain=np.einsum('ctf,cf->ct',phi,beta[b],optimize=True).clip(*bounds)
        delta=prepared['carriers'][b]*np.repeat(gain-1,20,axis=-1)
        spectrum=np.fft.rfft(delta,axis=-1);projected=np.zeros_like(spectrum);projected[:,indices]=spectrum[:,indices]
        correction += np.fft.irfft(projected,n=6000,axis=-1)
        gains.append(gain)
    s=prepared['prior']['S']+correction
    total=prepared['prior']['sum'].copy()
    return dict(S=s,O=total-s,sum=total),np.asarray(gains)


class BoundedTransientSource:
    def __init__(self,transfer,models,metadata):
        self.transfer=transfer;self.models=models;self.metadata=metadata

    def predict_continuous(self,values,dataset,observed):
        values=np.asarray(values,float);seen=np.asarray(observed,bool)
        if values.ndim!=4 or values.shape[1:]!=(len(self.metadata['positions'][dataset]),15,400) or seen.shape!=values.shape[:2]:
            raise ValueError('X-only native-array 30s source input required')
        fitted=self.models[dataset];output={b:[] for b in ['S','O','sum']}
        for x,mask in zip(values,seen):
            prepared=prepare_x(x.reshape(len(x),6000).T,mask,self.transfer)
            pred,_=apply_gain(prepared,fitted['background'],fitted['beta'],self.metadata['transient'],self.metadata['gain_bounds'])
            for b in output:output[b].append(pred[b].reshape(x.shape))
        return {b:np.asarray(v) for b,v in output.items()}

    def save(self,path):
        arrays={b:self.transfer[b] for b in 'SO'}
        for d,model in self.models.items():
            arrays[d+'_background']=model['background'];arrays[d+'_beta']=model['beta']
        np.savez_compressed(path,metadata=json.dumps(self.metadata),**arrays)


def load_model(path):
    with np.load(Path(path)) as saved:
        metadata=json.loads(str(saved['metadata']))
        if metadata['schema']!='eeg_bounded_transient_source_v1':raise ValueError('bounded continuous source checkpoint required')
        models={d:dict(background=saved[d+'_background'].copy(),beta=saved[d+'_beta'].copy()) for d in metadata['positions']}
        transfer={b:saved[b].copy() for b in 'SO'}
    return BoundedTransientSource(transfer,models,metadata)
