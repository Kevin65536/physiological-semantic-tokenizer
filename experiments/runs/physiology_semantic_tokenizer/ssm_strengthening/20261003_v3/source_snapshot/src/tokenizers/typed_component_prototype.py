"""Small continuous, typed-token experiment; not a qualified measured tokenizer.

EEG alone owns driver tokens. Hb alone owns total-Hb morphology tokens.
Observation encoders are separate so reconstruction cannot update semantic
encoders through a shared stem. The Hb decoder uses detached, paired semantic
conditions; its reconstruction is privileged and is labelled as such.
"""
from __future__ import annotations

import torch
from torch import nn


class TypedComponentPrototype(nn.Module):
    def __init__(self, steps=120, width=128, driver_patches=12, morphology_modes=4,
                 observation_dimensions=8, detach_semantic=True):
        super().__init__()
        if steps < 3 or width < 1 or driver_patches < 1 or morphology_modes < 1:
            raise ValueError('positive model dimensions required')
        if observation_dimensions < 2 or observation_dimensions % 2:
            raise ValueError('observation dimensions must split equally across modalities')
        self.steps=steps;self.detach_semantic=detach_semantic
        self.driver_patches=driver_patches;self.morphology_modes=morphology_modes
        def encoder(inputs, outputs):
            return nn.Sequential(nn.Linear(inputs,width),nn.GELU(),
                nn.Linear(width,width//2),nn.GELU(),nn.Linear(width//2,outputs))
        self.neural_encoder=encoder(steps,driver_patches)
        self.morphology_encoder=encoder(2*steps,morphology_modes)
        self.eeg_observation_encoder=encoder(steps,observation_dimensions//2)
        self.hb_observation_encoder=encoder(2*steps,observation_dimensions//2)
        self.eeg_decoder=encoder(driver_patches+observation_dimensions//2,steps)
        self.hb_decoder=encoder(driver_patches+morphology_modes+observation_dimensions//2,2*steps)

    def forward(self, values):
        if values.ndim != 3 or values.shape[1:] != (self.steps,3):
            raise ValueError(f'expected [B,{self.steps},3]')
        eeg=values[:,:,0];hb=values[:,:,1:].flatten(1)
        neural=self.neural_encoder(eeg);morphology=self.morphology_encoder(hb)
        oe=self.eeg_observation_encoder(eeg);oh=self.hb_observation_encoder(hb)
        ns=neural.detach() if self.detach_semantic else neural
        ms=morphology.detach() if self.detach_semantic else morphology
        re=self.eeg_decoder(torch.cat((ns,oe),dim=1))
        rh=self.hb_decoder(torch.cat((ns,ms,oh),dim=1)).reshape(-1,self.steps,2)
        return dict(semantic=torch.cat((neural,morphology),dim=1),
            observation=torch.cat((oe,oh),dim=1),
            reconstruction=torch.cat((re[:,:,None],rh),dim=2))

    def token_metadata(self):
        return dict(schema='typed_component_prototype_v1',continuous=True,
            coordinate_id='synthetic_training_standardized_reference_centered_driver_and_HbT_cosines',
            source_status='known_synthetic_intervention_only_measured_source_unresolved',
            semantic_type=['reference_centered_driver_patch_mean']*self.driver_patches+
                          ['structured_HbT_cosine_coefficient']*self.morphology_modes,
            support_mask=[True]*(self.driver_patches+self.morphology_modes),
            neural_input='EEG_only',morphology_input='Hb_only',
            decoder='paired_privileged_semantic_condition',qualified_for_measured_data=False)
