"""Frozen observation encoder, one atomic latent dictionary per component."""
from pathlib import Path

import numpy as np
import torch

from .dual_component_tokenizer import GlobalLatentComponentTokenizer


class ContextualLatentDictionary:
    def __init__(self, model, metadata):
        self.model = model.eval()
        self.metadata = metadata

    def encode(self, values):
        tensor = torch.as_tensor(values, dtype=torch.float32)
        with torch.inference_mode():
            latents = self.model.encode_continuous(tensor)
            result = {}
            for branch, latent in latents.items():
                quantizer = self.model.branches[branch]['quantizer']
                logits = quantizer._assignment_logits(latent.reshape(-1, latent.shape[-1]))
                result[branch] = logits.argmax(-1).reshape(latent.shape[:2]).numpy()
        return result

    def decode_ids(self, ids):
        tensors = {key: torch.as_tensor(value) for key, value in ids.items()}
        with torch.inference_mode():
            result = self.model.decode_ids(tensors)
        return self.synthesize({key: value.numpy() for key, value in result.items() if key in ('S', 'O', 'sum')})

    def predict_continuous(self, values):
        with torch.inference_mode():
            result = self.model(torch.as_tensor(values, dtype=torch.float32), quantize=False)
        return self.synthesize({key: value.numpy() for key, value in result.items() if key in ('S', 'O', 'sum')})

    def synthesize(self, prediction):
        gain = self.metadata.get('decoder_color_gain')
        cutoff = self.metadata.get('decoder_lowpass_hz')
        transfer = self.metadata.get('decoder_component_transfer_gain')
        if gain is None and cutoff is None and transfer is None:
            return prediction
        channels = 1 if self.metadata['modality'] == 'eeg' else 2
        result = {}
        for branch in ('S', 'O'):
            original = prediction[branch]
            values = original.reshape(len(original), -1, channels)
            if gain is not None:
                factor = np.asarray(gain[branch], dtype=float)
                if factor.shape != (channels,) or not np.isfinite(factor).all():
                    raise ValueError('one finite frozen decoder gain per component and color required')
                values = values * factor
            if cutoff is not None:
                if channels != 2 or not 0 < float(cutoff) < 5:
                    raise ValueError('Hb decoder cutoff must be inside the 10 Hz Nyquist support')
                from scipy.signal import butter, sosfiltfilt
                sections = butter(4, float(cutoff), fs=10, output='sos')
                values = sosfiltfilt(sections, values, axis=1)
            result[branch] = values.reshape(original.shape)
        result['sum'] = result['S'] + result['O']
        if transfer is not None:
            factor = np.asarray(transfer,dtype=float)
            if channels != 2 or factor.shape != (2,) or not np.isfinite(factor).all() or (factor<=0).any():
                raise ValueError('two positive frozen component transfer gains required')
            # Preserve the source X output exactly. Subtraction in float64 leaves
            # only machine rounding in the independently checked S+O closure.
            conserved = prediction['sum'] if gain is None and cutoff is None else result['sum']
            result['O'] = (result['O'].reshape(len(result['O']),-1,2).astype(float)*factor).reshape(prediction['O'].shape)
            result['S'] = np.asarray(conserved,float)-result['O']
            result['sum'] = conserved
        return result

    def __call__(self, values):
        ids = self.encode(values)
        return dict(self.decode_ids(ids), ids=ids)


def load_model(path):
    saved = torch.load(Path(path), map_location='cpu', weights_only=False)
    if saved['metadata']['schema'] == 'hb_semantic_analysis_synthesis_v1':
        from .hb_semantic_analysis_synthesis import load_model as load_search_model
        return load_search_model(path)
    if saved['metadata']['schema'] == 'silent_observation_tokenizer_v1':
        from .silent_observation_tokenizer import load_model as load_silent_model
        return load_silent_model(path)
    if saved['metadata']['schema'] not in ('contextual_latent_dictionary_v1', 'contextual_latent_dictionary_v2'):
        raise ValueError('contextual dictionary schema required')
    model_class = {'unrestricted_hb_context_v1':UnrestrictedHbContextDecoder,
                   'prototype_supervised_hb_v1':PrototypeSupervisedHbTokenizer}.get(
                       saved['metadata'].get('decoder_architecture'),GlobalLatentComponentTokenizer)
    model = model_class(saved['model_config'], codebook_size=saved['metadata']['codebook_size'], hard=True)
    model.load_state_dict(saved['model_state'])
    if saved['metadata']['schema'] == 'contextual_latent_dictionary_v2':
        reference = GlobalLatentComponentTokenizer(saved['model_config'],
            codebook_size=saved['metadata']['codebook_size'], hard=True)
        reference.load_state_dict(saved['reference_state'])
        return SemanticContextualDictionary(model, saved['metadata'], reference, saved['assignment'])
    return ContextualLatentDictionary(model, saved['metadata'])


def hb_semantic_features(waveforms):
    """Paired-color level, within-patch shape, slope and curvature in input units."""
    values = np.asarray(waveforms).reshape(-1, 15, 20, 2)
    level = values.mean(axis=2)
    shape = values - level[:, :, None]
    time = (np.arange(20) - 9.5) / 10.
    centered_square = time ** 2 - np.mean(time ** 2)
    slope = np.einsum('bptc,t->bpc', shape, time) / np.sum(time ** 2)
    curvature = np.einsum('bptc,t->bpc', shape, centered_square) / np.sum(centered_square ** 2)
    return np.concatenate((level, shape.reshape(*level.shape[:2], 40), slope, curvature), axis=-1)


class SemanticContextualDictionary(ContextualLatentDictionary):
    def __init__(self, model, metadata, reference_model, assignment):
        super().__init__(model, metadata)
        self.reference_model = reference_model.eval()
        self.assignment = assignment

    def encode(self, values):
        tensor = torch.as_tensor(values, dtype=torch.float32)
        with torch.inference_mode():
            predicted = self.reference_model(tensor, quantize=False)
        ids = {}
        for branch in ('S', 'O'):
            feature = hb_semantic_features(predicted[branch].numpy())
            feature = (feature - self.assignment[branch + '_mean']) / self.assignment[branch + '_scale']
            feature = (feature * self.assignment[branch + '_weight']).astype(np.float32)
            flat = feature.reshape(-1, feature.shape[-1])
            centers = self.assignment[branch + '_centers']
            distance = (np.sum(flat ** 2, axis=1, keepdims=True) - 2 * flat @ centers.T
                        + np.sum(centers ** 2, axis=1)[None])
            ids[branch] = distance.argmin(axis=1).reshape(feature.shape[:2]).astype(np.int64)
        return ids


class UnrestrictedHbContextDecoder(GlobalLatentComponentTokenizer):
    """Same frozen IDs and local shape with a freely fitted same-branch context term."""
    def _decode_component_parts(self, component, latent):
        if not self.hb_local_shape:
            raise ValueError('unrestricted Hb context requires the paired local-shape architecture')
        branch = self.branches[component]
        context = branch['decoder'](branch['from_latent'](latent) + self.position)
        local = branch['local_shape'](latent).reshape(*latent.shape[:2], 20, 2)
        local = local-local.mean(-2, keepdim=True)
        correction = branch['output'](context).reshape(local.shape)
        correction = correction-correction.mean(-2, keepdim=True)
        level = branch['level'](context).unsqueeze(-2).expand_as(local)
        return dict(full=(level+local+correction).flatten(-2), level=level.flatten(-2),
                    local=local.flatten(-2), context=correction.flatten(-2))


class PrototypeSupervisedHbTokenizer(GlobalLatentComponentTokenizer):
    """Free sequence reconstruction with an auxiliary waveform determined by one ID."""
    def __init__(self, cfg, *, codebook_size=128, hard=True):
        super().__init__(cfg,codebook_size=codebook_size,hard=hard)
        if self.input_dim != 40 or self.hb_local_shape:
            raise ValueError('prototype supervision uses the paired free Hb waveform architecture')
        self.prototype_heads = torch.nn.ModuleDict({key:torch.nn.Linear(int(cfg['latent_dim']),40) for key in ('S','O')})

    def prototype(self, branch, value):
        shape = self.prototype_heads[branch](value).reshape(*value.shape[:2],20,2)
        return (shape-shape.mean(-2,keepdim=True)).flatten(-2)

    def _from_latents(self, latents, values, use_hard):
        result = super()._from_latents(latents,values,use_hard)
        for branch,latent in latents.items():
            value = latent
            if use_hard:
                lookup = self.branches[branch]['quantizer'].get_embedding(result['ids'][branch])
                value = latent+(lookup-latent).detach() if self.training else lookup
            result[branch+'_prototype'] = self.prototype(branch,value)
        return result

    def decode_ids(self, ids):
        result = super().decode_ids(ids)
        for branch in ('S','O'):
            quantizer = self.branches[branch]['quantizer']
            value = quantizer.get_embedding(ids[branch].to(device=quantizer.codebook.device,dtype=torch.long))
            result[branch+'_prototype'] = self.prototype(branch,value)
        return result
