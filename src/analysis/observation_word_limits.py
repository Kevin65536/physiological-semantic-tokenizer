"""Construct explicit counterexamples to interpreting a four-bit word as a source.

These are equal-feature *synthetic* two-second inputs, not physiological samples
or estimates of how often a collision occurs. No fitting/evaluation labels.
"""
import numpy as np


def frozen_word_collisions(routers):
    t = np.arange(400)/200.
    # Exact DFT bins in the SAME 8--13 Hz assignment band; their frequencies
    # and shapes still differ. The sign-inverted middle item changes phase by pi.
    a=np.sin(2*np.pi*8.5*t); e=np.stack([a,-a,np.sin(2*np.pi*12*t)])[:,None,:]
    hbt = (np.arange(20)-9.5)/10.
    # Draw residual orthogonal to constant and time. Put the same residual into
    # both colors, then reverse it only in HbR. The two signed OLS slopes and
    # both levels are unchanged; the detrended color correlation reverses.
    base=np.array([[.18],[-.08]])+np.array([[.09],[-.045]])*hbt
    residual=np.cos(2*np.pi*.5*hbt);residual-=residual.mean();residual-=hbt*(hbt@residual)/(hbt@hbt)
    h=np.stack([base + .12*np.stack([residual,residual]), base + .12*np.stack([residual,-residual])])
    result={}
    for m,x in [('eeg',e),('hb',h)]:
        ids=routers[m].encode(x,mode='prototype')['pattern'];f=routers[m].pattern_features(x)
        # Exact assignment is the endpoint; constants above yield roundoff only in features.
        if not np.all(ids==ids[0]) or not np.allclose(f,f[0,None,:],atol=1e-12,rtol=1e-12):
            raise ValueError('equal-property witness failed')
        result[m]=dict(ids=ids.astype(int).tolist(),features=f[0].tolist(),feature_max_diff=float(np.max(abs(f-f[0]))))
    freq=np.fft.rfftfreq(400,1/200.);result['eeg'].update(peak_hz=freq[np.abs(np.fft.rfft(e[:,0],axis=1)).argmax(axis=1)].tolist(),waveform_correlation_vs_first=np.corrcoef(e[:,0])[0].tolist())
    r=h-h.mean(axis=-1,keepdims=True)-np.einsum('nct,t->nc',h,hbt)[:,:,None]*hbt/(hbt@hbt)
    result['hb']['detrended_color_correlation']=[float(np.corrcoef(u)[0,1]) for u in r]
    return dict(schema='fixed_word_identifiability_witness_v1',status='passed',synthetic_only=True,checks=result,
                scope='Exact counterexamples to frequency/phase and detrended Hb color coupling identifiability; not the prevalence of a real artifact or state.')
