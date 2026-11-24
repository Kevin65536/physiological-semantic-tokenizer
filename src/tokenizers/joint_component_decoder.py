"""Frozen G4 integer encoders with paired cross-branch decoder controls."""
from pathlib import Path

import torch
from torch import nn

from .dual_component_tokenizer import GlobalLatentComponentTokenizer


class JointComponentDecoder(GlobalLatentComponentTokenizer):
    def __init__(self, setting, *, routing, codebook_size=1024):
        super().__init__(setting, codebook_size=codebook_size, hard=True)
        if routing not in ('joint', 'branch_only') or not self.hb_local_shape or self.input_dim != 40:
            raise ValueError('joint or branch-only paired Hb G4 decoder required')
        self.routing = routing
        latent = int(setting['latent_dim'])
        width = int(setting['width'])
        self.register_buffer('embedding_mean', torch.zeros(2, latent))
        self.register_buffer('embedding_scale', torch.ones(2, latent))
        for branch in self.branches.values():
            branch['cross_from_latent'] = nn.Linear(latent, width)
            branch['free_output'] = nn.Linear(width, 40)
            for module in (branch['cross_from_latent'], branch['free_output']):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
            for name in ('input', 'encoder', 'to_latent', 'quantizer'):
                branch[name].requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        for branch in self.branches.values():
            for name in ('input', 'encoder', 'to_latent', 'quantizer'):
                branch[name].eval()
        return self

    @torch.no_grad()
    def encode_ids(self, values):
        latents = self.encode_continuous(values)
        return {name: self.branches[name]['quantizer']._assignment_logits(value.reshape(-1, value.shape[-1]))
                .argmax(-1).reshape(value.shape[:2]) for name, value in latents.items()}

    def _check_ids(self, branch, indices):
        if (not isinstance(indices, torch.Tensor) or indices.dtype not in (torch.int32, torch.int64)
                or indices.ndim != 2 or indices.shape[1] != 15
                or bool(((indices < 0) | (indices >= self.branches[branch]['quantizer'].codebook_size)).any())):
            raise ValueError('one in-range integer per branch and two-second patch required')

    def decode_latents(self, latents):
        result = {}
        for name in ('S', 'O'):
            other = name if self.routing == 'branch_only' else ('O' if name == 'S' else 'S')
            other_index = 0 if other == 'S' else 1
            branch = self.branches[name]
            own = latents[name]
            cross = (latents[other] - self.embedding_mean[other_index]) / self.embedding_scale[other_index]
            hidden = branch['from_latent'](own) + branch['cross_from_latent'](cross) + self.position
            context = branch['decoder'](hidden)
            local = branch['local_shape'](own).reshape(*own.shape[:2], 20, 2)
            local = local - local.mean(-2, keepdim=True)
            correction = branch['output'](context).reshape(local.shape)
            correction = correction - correction.mean(-2, keepdim=True)
            local_norm = torch.linalg.vector_norm(local, dim=-2, keepdim=True)
            correction_norm = torch.linalg.vector_norm(correction, dim=-2, keepdim=True)
            factor = (self.context_ratio * local_norm / correction_norm.clamp_min(self.context_norm_epsilon)).clamp(max=1.)
            level = branch['level'](context).unsqueeze(-2).expand_as(local)
            base = (level + local + correction * factor).flatten(-2)
            result[name] = base + branch['free_output'](context)
        result['sum'] = result['S'] + result['O']
        return result

    def decode_ids(self, ids):
        if set(ids) != {'S', 'O'}:
            raise ValueError('exactly the declared S/O integer sequences required')
        for name in ('S', 'O'):
            self._check_ids(name, ids[name])
        if ids['S'].shape != ids['O'].shape:
            raise ValueError('S/O supports differ')
        latents = {name: self.branches[name]['quantizer'].get_embedding(ids[name].long()) for name in ('S', 'O')}
        return self.decode_latents(latents)

    def decode_branch_ids(self, branch, indices):
        if self.routing != 'branch_only' or branch not in ('S', 'O'):
            raise ValueError('independent branch replay belongs to the branch-only control')
        self._check_ids(branch, indices)
        latent = self.branches[branch]['quantizer'].get_embedding(indices.long())
        # Each control output uses its own real ID in both capacity-matched input slots.
        return self.decode_latents(dict(S=latent, O=latent))[branch]


class JointComponentTokenizer:
    def __init__(self, model, metadata):
        self.model, self.metadata = model.eval(), metadata

    def encode(self, values):
        with torch.inference_mode():
            ids = self.model.encode_ids(torch.as_tensor(values, dtype=torch.float32))
        return {name: value.numpy() for name, value in ids.items()}

    def decode_ids(self, ids):
        tensors = {name: torch.as_tensor(value) for name, value in ids.items()}
        with torch.inference_mode():
            result = self.model.decode_ids(tensors)
        return {name: value.numpy() for name, value in result.items()}

    def decode_branch_ids(self, branch, indices):
        with torch.inference_mode():
            result = self.model.decode_branch_ids(branch, torch.as_tensor(indices))
        return result.numpy()

    def predict_continuous(self, values):
        with torch.inference_mode():
            latents = self.model.encode_continuous(torch.as_tensor(values, dtype=torch.float32))
            result = self.model.decode_latents(latents)
        return {name: value.numpy() for name, value in result.items()}

    def __call__(self, values):
        ids = self.encode(values)
        return dict(self.decode_ids(ids), ids=ids)


def load_model(path):
    saved = torch.load(Path(path), map_location='cpu', weights_only=False)
    metadata = saved['metadata']
    if metadata['schema'] != 'joint_component_decoder_v1':
        raise ValueError('frozen cross-branch decoder schema required')
    model = JointComponentDecoder(saved['model_config'], routing=metadata['routing'],
                                  codebook_size=metadata['codebook_size'])
    model.load_state_dict(saved['model_state'])
    return JointComponentTokenizer(model, metadata)
