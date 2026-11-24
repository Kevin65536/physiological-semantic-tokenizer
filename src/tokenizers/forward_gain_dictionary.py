"""Current-patch amplitude diagnostics for a frozen atomic spatial decoder."""
import numpy as np
from .spatial_waveform_dictionary import band_filter


def projection_gains(source, ids, decoder, dataset, *, nonnegative=True, complete=None):
    """Least squares for this local atom's role-band waveform, not final output."""
    result={}
    for branch in 'SO':
        code=np.asarray(ids[branch]);value=np.asarray(source[branch],float)
        if value.shape!=(*code.shape,400) or code.dtype.kind not in 'iu':
            raise ValueError('native integer shape and full local source required')
        gain=np.ones(code.shape)
        for band in range(5):
            sites=np.flatnonzero(decoder.roles(dataset,branch)==band)
            atom=band_filter(decoder.atoms[branch,band][code[:,sites]],band)
            target=band_filter(value[:,sites],band)
            energy=(atom**2).sum(-1)
            g=np.divide((atom*target).sum(-1),energy,out=np.ones_like(energy),where=energy>1e-20)
            if nonnegative:g=np.maximum(g,0)
            gain[:,sites]=g
        gain[code==0]=1
        if complete is not None:gain[~np.asarray(complete,bool)]=1
        result[branch]=gain
    return result


def decode_scaled_atoms(decoder, ids, gains, dataset):
    """Exact old linear synthesis, with a paid-or-diagnostic local atom scale.

    A continuous gain is a diagnostic input. It is not an integer-only decoder.
    The original gain=1 endpoint follows the same arithmetic as decode_ids.
    """
    if decoder.readout or decoder.metadata.get('joint_spatial_inputs'):
        raise ValueError('this version requires the frozen separable semantic decoder')
    result={};shape=None
    for branch in 'SO':
        codes=np.asarray(ids[branch]);g=np.asarray(gains[branch],float);roles=decoder.roles(dataset,branch)
        if (codes.ndim!=3 or codes.shape[1:]!=(len(roles),15) or codes.dtype.kind not in 'iu'
            or np.any(codes<0) or np.any(codes>=decoder.codebook_size) or g.shape!=codes.shape or not np.isfinite(g).all()):
            raise ValueError('matching finite gains and native atomic integers required')
        if shape is not None and codes.shape!=shape:raise ValueError('same branch shapes required')
        shape=codes.shape;decoded=np.zeros((*codes.shape,400),float)
        for band in range(5):
            positions=np.flatnonzero(roles==band)
            decoded[:,positions]=decoder.atoms[branch,band][codes[:,positions]]*g[:,positions,:,None]
        waveforms=decoded.reshape(len(codes),len(roles),6000).transpose(0,2,1)
        reconstructed=waveforms.copy()
        for band in range(5):
            positions=np.flatnonzero(roles==band)
            baseline=band_filter(waveforms,band,axis=1)
            predicted=baseline[:,:,positions]@decoder.weights[dataset,branch,band]
            available=(codes[:,positions]!=0).any(axis=(1,2))
            reconstructed[available]+=predicted[available]-baseline[available]
        result[branch]=reconstructed.transpose(0,2,1).reshape(*codes.shape,400)
    if decoder.metadata.get('output_projection_hz')!=[1,45]:
        raise ValueError('same frozen 30-second output projection required')
    frequency=np.fft.rfftfreq(6000,1/200);retained=(frequency>=1)&(frequency<45)
    for branch in 'SO':
        v=result[branch];spectrum=np.fft.rfft(v.reshape(*v.shape[:2],6000),axis=-1)
        result[branch]=np.fft.irfft(spectrum*retained[None,None],n=6000,axis=-1).reshape(v.shape)
    result['sum']=result['S']+result['O'];return result
