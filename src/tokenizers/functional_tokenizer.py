"""Eight-arm functional supervision screen with a genuine hard VQ forward path.

All arms have independent S/O paths and identical trainable dimensions. The
decoder reconstructs the selected measured EEG voltage and Hb bin samples.
An integer-only consumer has no access to this encoder or to soft assignments.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .ema_vector_quantizer import EMAVectorQuantizer

ARMS = ('C0','CT','Q0','QT','C-wide','C-smooth','C-shuffle','C-stop')


class PatchEncoder(nn.Module):
    def __init__(self, inputs, width, latent):
        super().__init__()
        self.stem = nn.Linear(inputs*2, width)
        self.context = nn.ModuleList([nn.Conv1d(width,width,3,padding=d,dilation=d) for d in (1,2,4)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in self.context])
        self.head = nn.Linear(width, latent)

    def forward(self, values, mask):
        x = F.gelu(self.stem(torch.cat((values,mask.to(values.dtype)), -1)))
        for layer,norm in zip(self.context,self.norms):
            x = norm(x+F.gelu(layer(x.transpose(1,2)).transpose(1,2)))
        return self.head(x)


class FunctionalBranch(nn.Module):
    def __init__(self, input_dim, target_dim, cfg, quantized, stop):
        super().__init__()
        w,s,o = cfg['width'],cfg['semantic_dim'],cfg['observation_dim']
        self.input_dim,self.target_dim = input_dim,target_dim
        self.quantized,self.stop = quantized,stop
        self.source_encoder = PatchEncoder(input_dim,w,s)
        self.observation_encoder = PatchEncoder(input_dim,w,o)
        self.decoder = nn.Sequential(nn.Linear(s+o,w),nn.GELU(),nn.Linear(w,input_dim))
        self.teacher_head = nn.Sequential(nn.Linear(s,w),nn.GELU(),nn.Linear(w,target_dim))
        # Continuous arms retain the same buffers (no additional trainable VQ parameters).
        self.quantizer = EMAVectorQuantizer(cfg['codebook_size'],s,decay=cfg['ema_decay'],
            commitment_cost=cfg['commitment'],kmeans_init=True,kmeans_iters=10,
            revive_dead_codes=False)

    def forward(self, values, mask=None):
        if mask is None:
            mask = torch.ones_like(values,dtype=torch.bool)
        mask = torch.broadcast_to(mask,values.shape).bool()
        clean = torch.where(mask,values,torch.zeros_like(values))
        if not torch.isfinite(clean).all():
            raise ValueError('nonfinite visible measurement')
        pre = self.source_encoder(clean,mask)
        observation = self.observation_encoder(clean,mask)
        if self.quantized:
            q = self.quantizer(pre,mask.any(-1))
            semantic,soft,ids,commitment = q.quantized,q.expected_embedding,q.hard_ids,q.commitment_loss
            health=q.health
        else:
            semantic,soft,ids,commitment = pre,pre,None,pre.sum()*0
            health={}
        reconstructed = self.decoder(torch.cat((semantic.detach() if self.stop else semantic, observation),-1))
        return dict(pre=pre,soft=soft,hard_ids=ids,semantic=semantic,observation=observation,
            reconstruction=reconstructed,teacher=self.teacher_head(semantic),commitment=commitment,
            support=mask.any(-1),health=health)


class FunctionalTokenizer(nn.Module):
    def __init__(self,cfg,arm):
        super().__init__()
        if arm not in ARMS or not cfg['independent_encoders']:
            raise ValueError('registered arm with independent encoders required')
        self.arm,self.config = arm,dict(cfg)
        self.eeg = FunctionalBranch(400*6,8,cfg,arm in ('Q0','QT'),arm=='C-stop')
        self.hb = FunctionalBranch(8*2,8*2,cfg,arm in ('Q0','QT'),arm=='C-stop')

    def forward(self,eeg,hb,eeg_mask=None,hb_mask=None):
        if eeg.shape[1:]!=(10,2400) or hb.shape[1:]!=(10,16):
            raise ValueError('expected ten synchronized 2s patches')
        return dict(eeg=self.eeg(eeg,eeg_mask),hb=self.hb(hb,hb_mask))

    def objective(self,output,eeg,hb,targets,weights):
        terms = {}
        total = eeg.new_zeros(())
        for name,measurement in (('eeg',eeg),('hb',hb)):
            r = output[name]
            rec = F.mse_loss(r['reconstruction'],measurement)
            teacher = rec*0
            if self.arm not in ('C0','Q0'):
                prediction = r['teacher']
                truth = targets[name]
                if name=='hb' and self.arm!='C-wide':
                    prediction,truth = prediction.reshape(-1,10,8,2)[...,0],truth.reshape(-1,10,8,2)[...,0]
                teacher = F.mse_loss(prediction,truth)
            total = total+weights['reconstruction_weight']*rec+weights['teacher_weight']*teacher+r['commitment']
            terms[name+'_reconstruction']=rec; terms[name+'_teacher']=teacher
            terms[name+'_commitment']=r['commitment']
        terms['total']=total
        return terms


class ConditionalShapeTokenizer:
    """Local conditional waveform tokenizer with explicit scale sidecars.

    One instance serves one modality and every ROI. The ridge and both shape
    codebooks fit training patches only; validation chooses the ridge penalty.
    The residual is the unexplained observation, not a physiological source.
    """
    def __init__(self,codebook_size=16,seed=17,amplitude_floor=1e-6,
                 kmeans_batch_size=1024,kmeans_max_iter=100):
        if codebook_size<2 or amplitude_floor<=0 or kmeans_batch_size<1 or kmeans_max_iter<1:
            raise ValueError('positive fixed shape tokenizer settings required')
        self.codebook_size=int(codebook_size);self.seed=int(seed)
        self.amplitude_floor=float(amplitude_floor)
        self.kmeans_batch_size=int(kmeans_batch_size);self.kmeans_max_iter=int(kmeans_max_iter)

    @staticmethod
    def decompose(values,amplitude_floor=1e-6):
        """Per-channel means and one positive, joint time/channel RMS."""
        values=np.asarray(values,float)
        if values.ndim<2 or values.shape[-2]!=8 or not np.isfinite(values).all() or amplitude_floor<=0:
            raise ValueError('finite eight-sample patches and positive amplitude floor required')
        mu=values.mean(axis=-2)
        centered=values-mu[...,None,:]
        rms=np.sqrt(np.mean(centered**2,axis=(-2,-1)))
        amplitude=np.maximum(rms,float(amplitude_floor))
        shape=centered/amplitude[...,None,None]
        shape=np.where((rms<=amplitude_floor)[...,None,None],0.,shape)
        return mu,amplitude,shape

    def fit(self,observations,conditional_targets,train_indices,validation_indices,ridge_alpha,
            validation_groups=None):
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.linear_model import Ridge
        x,y=np.asarray(observations,float),np.asarray(conditional_targets,float)
        if x.ndim!=3 or x.shape[1]!=8 or y.shape!=x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('aligned finite [patches,8,channels] observations and conditional targets required')
        train=np.asarray(train_indices);validation=np.asarray(validation_indices)
        if train.dtype.kind=='b': train=np.flatnonzero(train)
        if validation.dtype.kind=='b': validation=np.flatnonzero(validation)
        if (train.ndim!=1 or validation.ndim!=1 or train.dtype.kind not in 'iu' or validation.dtype.kind not in 'iu'
                or len(train)<self.codebook_size or not len(validation) or len(set(train.tolist()))!=len(train)
                or len(set(validation.tolist()))!=len(validation) or set(train.tolist())&set(validation.tolist())
                or np.any(train<0) or np.any(validation<0) or np.any(train>=len(x)) or np.any(validation>=len(x))):
            raise ValueError('disjoint nonempty training/validation patches and fixed codebook support required')
        alphas=[float(a) for a in ridge_alpha]
        if not alphas or any(not np.isfinite(a) or a<=0 for a in alphas):
            raise ValueError('positive finite ridge penalties required')
        self.channels=x.shape[-1]
        flat=x.reshape(len(x),-1)
        self.input_mean=flat[train].mean(axis=0)
        self.input_scale=np.maximum(flat[train].std(axis=0),1e-8)
        standardized=(flat-self.input_mean)/self.input_scale
        target_scale=np.maximum(y[train].std(axis=(0,1)),1e-8)
        target_scale_flat=np.tile(target_scale,8)
        target=y.reshape(len(y),-1)/target_scale_flat
        groups=None if validation_groups is None else np.asarray(validation_groups)
        if groups is not None and groups.shape!=(len(x),):
            raise ValueError('one validation group identity per patch required')
        best=None
        for alpha in alphas:
            fitted=Ridge(alpha=alpha).fit(standardized[train],target[train])
            error=np.mean((fitted.predict(standardized[validation])-target[validation])**2,axis=1)
            score=(float(error.mean()) if groups is None else
                float(np.mean([error[groups[validation]==group].mean() for group in np.unique(groups[validation])])))
            if best is None or score<best[0]: best=(score,alpha,fitted)
        self.validation_MSE,self.selected_alpha=best[:2]
        self.ridge_coefficients=best[2].coef_*target_scale_flat[:,None]
        self.ridge_intercept=best[2].intercept_*target_scale_flat
        self.target_training_SD=target_scale
        self.centers={}
        components=self.predict_components(x[train])
        for offset,name in enumerate(('c','r')):
            _,_,shape=self.decompose(components[name],self.amplitude_floor)
            fit=MiniBatchKMeans(n_clusters=self.codebook_size,random_state=self.seed+offset,n_init=3,
                batch_size=self.kmeans_batch_size,max_iter=self.kmeans_max_iter).fit(shape.reshape(len(train),-1))
            self.centers[name]=fit.cluster_centers_.reshape(self.codebook_size,8,self.channels)
        self.training_patches=len(train);self.validation_patches=len(validation)
        return self

    def predict_components(self,observations):
        values=np.asarray(observations,float)
        if (not hasattr(self,'ridge_coefficients') or values.ndim<2 or values.shape[-2:]!=(8,self.channels)
                or not np.isfinite(values).all()):
            raise ValueError('fitted tokenizer and finite modality-local eight-sample patches required')
        flat=values.reshape(-1,8*self.channels)
        conditional=(((flat-self.input_mean)/self.input_scale)@self.ridge_coefficients.T+self.ridge_intercept).reshape(values.shape)
        return dict(c=conditional,r=values-conditional)

    def encode(self,observations):
        packet={}
        for name,values in self.predict_components(observations).items():
            mu,amplitude,shape=self.decompose(values,self.amplitude_floor)
            flat=shape.reshape(-1,8*self.channels);centers=self.centers[name].reshape(self.codebook_size,-1)
            distance=(flat**2).sum(1)[:,None]+(centers**2).sum(1)[None,:]-2*flat@centers.T
            packet[name+'_ids']=distance.argmin(axis=1).reshape(values.shape[:-2]).astype('int16')
            packet[name+'_mu']=mu.astype('float32')
            packet[name+'_amplitude']=amplitude.astype('float32')
        return packet

    def _id_features(self,packet):
        parts=[];shape=None
        for name in ('c','r'):
            ids=np.asarray(packet[name+'_ids'])
            if ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=self.codebook_size):
                raise ValueError('valid integer shape IDs required')
            if shape is not None and ids.shape!=shape: raise ValueError('component ID supports disagree')
            shape=ids.shape
            parts.append(self.centers[name][ids].reshape(ids.shape+(8*self.channels,)))
        return np.concatenate(parts,axis=-1)

    def _sidecar_features(self,packet):
        parts=[];shape=None
        for name in ('c','r'):
            mu=np.asarray(packet[name+'_mu']);amplitude=np.asarray(packet[name+'_amplitude'])
            if (mu.shape!=amplitude.shape+(self.channels,) or not np.isfinite(mu).all()
                    or not np.isfinite(amplitude).all() or np.any(amplitude<=0)):
                raise ValueError('finite per-channel means and positive common RMS sidecars required')
            if shape is not None and amplitude.shape!=shape: raise ValueError('component sidecar supports disagree')
            shape=amplitude.shape;parts.extend((mu,amplitude[...,None]))
        return np.concatenate(parts,axis=-1)

    def shape_features(self,packet,representation,components=None):
        """ID-only never reads sidecars; continuous requires explicit waveforms."""
        if representation=='id_only': return self._id_features(packet)
        if representation=='sidecar_only': return self._sidecar_features(packet)
        if representation=='hybrid':
            # Apply the known scale/shape decoder before a linear downstream
            # probe, so it need not learn the multiplication a * shape.
            return self.shape_features({},'continuous',components=self.decode(packet))
        if representation=='continuous':
            if components is None: raise ValueError('explicit conditional/residual waveforms required')
            waves=[np.asarray(components[name]) for name in ('c','r')]
            if (waves[0].shape!=waves[1].shape or waves[0].shape[-2:]!=(8,self.channels)
                    or not all(np.isfinite(wave).all() for wave in waves)):
                raise ValueError('aligned finite conditional/residual waveforms required')
            return np.concatenate([wave.reshape(wave.shape[:-2]+(8*self.channels,)) for wave in waves],axis=-1)
        raise ValueError('unknown shape representation')

    def decode(self,packet):
        self._id_features(packet);self._sidecar_features(packet)
        return {name:np.asarray(packet[name+'_mu'])[...,None,:]+
            np.asarray(packet[name+'_amplitude'])[...,None,None]*self.centers[name][packet[name+'_ids']]
            for name in ('c','r')}

    def artifact_arrays(self):
        """Numeric replay artifact; no Python/sklearn object serialization."""
        result={name:np.asarray(getattr(self,name)) for name in ('codebook_size','seed','amplitude_floor',
            'kmeans_batch_size','kmeans_max_iter','channels','selected_alpha','validation_MSE',
            'training_patches','validation_patches','input_mean','input_scale','ridge_coefficients',
            'ridge_intercept','target_training_SD')}
        result.update(c_centers=self.centers['c'],r_centers=self.centers['r'])
        return result

    @classmethod
    def from_artifact(cls,arrays):
        model=cls(**{name:np.asarray(arrays[name]).item() for name in ('codebook_size','seed','amplitude_floor',
            'kmeans_batch_size','kmeans_max_iter')})
        for name in ('channels','selected_alpha','validation_MSE','training_patches','validation_patches'):
            setattr(model,name,np.asarray(arrays[name]).item())
        for name in ('input_mean','input_scale','ridge_coefficients','ridge_intercept','target_training_SD'):
            setattr(model,name,np.asarray(arrays[name]).copy())
        model.centers={name:np.asarray(arrays[name+'_centers']).copy() for name in ('c','r')}
        if (model.input_mean.shape!=(8*model.channels,) or model.input_scale.shape!=model.input_mean.shape
                or np.any(model.input_scale<=0) or model.ridge_coefficients.shape!=(8*model.channels,8*model.channels)
                or model.ridge_intercept.shape!=model.input_mean.shape
                or any(center.shape!=(model.codebook_size,8,model.channels) for center in model.centers.values())
                or not all(np.isfinite(value).all() for value in arrays.values())):
            raise ValueError('invalid numeric shape tokenizer artifact')
        return model

    def bitrate(self):
        sidecar_scalars=2*self.channels+2
        return dict(nominal_ID_bits_per_modality_patch=2*np.log2(self.codebook_size),
            serialized_ID_bits_per_modality_patch=32,sidecar_bits_per_modality_patch=sidecar_scalars*32,
            hybrid_nominal_bits_per_modality_patch=2*np.log2(self.codebook_size)+sidecar_scalars*32,
            hybrid_serialized_bits_per_modality_patch=32+sidecar_scalars*32,
            continuous_bits_per_modality_patch=2*8*self.channels*32,
            frozen_dictionary_and_ridge_overhead='not_amortized_into_per_patch_rate',
            scope='shape_IDs_and_continuous_scale_sidecars_are_separately_consumable')
