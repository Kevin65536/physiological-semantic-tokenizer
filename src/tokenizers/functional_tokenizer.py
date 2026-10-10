"""Eight-arm functional supervision screen with a genuine hard VQ forward path.

All arms have independent S/O paths and identical trainable dimensions. The
decoder reconstructs the selected measured EEG voltage and Hb bin samples.
An integer-only consumer has no access to this encoder or to soft assignments.
"""
from __future__ import annotations

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
