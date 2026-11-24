"""Nonlinear full-array EEG encoding and integer-only message-passing synthesis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .fullband_eeg_tokenizer import FullbandEEGTokenizer


class MontageMessage(nn.Module):
    def __init__(self, width, heads, position_width):
        super().__init__()
        self.score = nn.Linear(width, heads)
        self.pool = nn.Linear(heads * width, width)
        self.update = nn.Sequential(nn.Linear(2 * width + position_width, 2 * width), nn.GELU(),
                                    nn.Linear(2 * width, width))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, values, mask, position, cross_position):
        # Both arms use the same learned pooling and nonlinear update. The local
        # arm pools only the visible patches of its own electrode.
        b, n, t, d = values.shape
        if cross_position:
            features = values.transpose(1, 2).reshape(b * t, n, d)
            valid = mask.transpose(1, 2).reshape(b * t, n)
        else:
            features = values.reshape(b * n, t, d)
            valid = mask.reshape(b * n, t)
        logits = self.score(features).masked_fill(~valid[..., None], -1e9)
        weights = torch.softmax(logits, dim=1) * valid[..., None]
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        pooled = torch.einsum('bnh,bnd->bhd', weights, features)
        context = self.pool(pooled.flatten(1))
        if cross_position:
            context = context.reshape(b, t, d)[:, None].expand(b, n, t, d)
        else:
            context = context.reshape(b, n, d)[:, :, None].expand(b, n, t, d)
        position = position[None, :, None].expand(b, n, t, -1)
        return values + self.update(torch.cat((values, context, position), dim=-1))


class NonlinearMontageTokenizer(nn.Module):
    def __init__(self, config, metadata, initialization=None):
        super().__init__()
        self.config = config
        self.metadata = metadata
        self.codebook_size = int(config['vocabulary'])
        if self.codebook_size < 3 or metadata['schema'] != 'nonlinear_montage_tokenizer_v1':
            raise ValueError('versioned montage tokenizer and reserved missing ID required')
        settings = dict(codebook_size=self.codebook_size - 1, latent_dim=config['latent_dim'],
                        hidden_width=config['hidden_width'], ema_decay=config['ema_decay'],
                        commitment_cost=config['commitment_cost'])
        self.local = FullbandEEGTokenizer(settings, initialization)
        self.position = nn.Embedding(sum(len(v) for v in metadata['positions'].values()), config['position_dim'])
        offsets, start = {}, 0
        for dataset, positions in metadata['positions'].items():
            offsets[dataset] = start
            start += len(positions)
        self.position_offsets = offsets
        self.encoder_message = nn.ModuleDict({b: MontageMessage(config['latent_dim'], config['pool_heads'], config['position_dim']) for b in 'SO'})
        self.decoder_message = nn.ModuleDict({b: MontageMessage(config['latent_dim'], config['pool_heads'], config['position_dim']) for b in 'SO'})
        self.attribute = nn.ModuleDict({b: nn.Linear(config['latent_dim'], 3) for b in 'SO'})
        self.register_buffer('attribute_mean', torch.zeros(2, 3))
        self.register_buffer('attribute_scale', torch.ones(2, 3))

    def position_values(self, dataset, device):
        n = len(self.metadata['positions'][dataset])
        index = torch.arange(n, device=device) + self.position_offsets[dataset]
        return self.position(index)

    def encode_continuous(self, x, dataset, observed):
        if (x.ndim != 4 or x.shape[1:] != (len(self.metadata['positions'][dataset]), 15, 400)
                or observed.shape != x.shape[:2] or observed.dtype != torch.bool):
            raise ValueError('30-second complete native montage and observation flags required')
        visible = torch.where(observed[..., None, None], x, 0.)
        if not torch.isfinite(visible).all():
            raise ValueError('finite observed X required')
        b, n = x.shape[:2]
        encoded = self.local.encode_continuous(visible.reshape(b * n, 15, 400))
        mask = observed[..., None].expand(b, n, 15)
        position = self.position_values(dataset, x.device)
        return {key: self.encoder_message[key](value.reshape(b, n, 15, -1), mask, position, self.config['cross_position'])
                for key, value in encoded.items()}

    def decode_latents(self, latents, dataset, mask):
        position = self.position_values(dataset, mask.device)
        b, n, t = mask.shape
        decoded = {key: self.decoder_message[key](value, mask, position, self.config['cross_position']).reshape(b * n, t, -1)
                   for key, value in latents.items()}
        outputs = self.local.decode_latents(decoded)
        return {key: value.reshape(b, n, 15, 400) for key, value in outputs.items()}

    def encode_ids(self, x, dataset, observed):
        latent = self.encode_continuous(x, dataset, observed)
        return {key: torch.where(observed[..., None], self.local.branches[key]['quantizer']._assignment_logits(z).argmax(-1) + 1, 0)
                for key, z in latent.items()}

    def decode_ids(self, ids, dataset):
        if set(ids) != {'S', 'O'}:
            raise ValueError('both S/O integer sequences required')
        latents, masks = {}, {}
        shape = None
        for branch in 'SO':
            code = ids[branch]
            if (code.ndim != 3 or code.shape[1:] != (len(self.metadata['positions'][dataset]), 15)
                    or code.dtype not in (torch.int32, torch.int64)
                    or ((code < 0) | (code >= self.codebook_size)).any()):
                raise ValueError('in-range native-position integer arrays required')
            if shape is not None and code.shape != shape:
                raise ValueError('S/O integer supports differ')
            shape = code.shape
            masks[branch] = code > 0
            embedding = self.local.branches[branch]['quantizer'].get_embedding((code.long() - 1).clamp_min(0))
            latents[branch] = torch.where(masks[branch][..., None], embedding, 0.)
        if not torch.equal(masks['S'], masks['O']):
            raise ValueError('observation absence must be represented in both branches')
        return self.decode_latents(latents, dataset, masks['S'])

    def forward(self, x, dataset, observed, *, quantize=False, valid_mask=None):
        latents = self.encode_continuous(x, dataset, observed)
        mask = observed[..., None].expand(*x.shape[:3])
        commitment = x.new_zeros(())
        codes = {}
        if quantize and not self.training:
            codes = {key: torch.where(mask, self.local.branches[key]['quantizer']._assignment_logits(z).argmax(-1) + 1, 0)
                     for key, z in latents.items()}
            # Evaluation shares exactly the independently callable integer path.
            output = self.decode_ids(codes, dataset)
            decoded = {key: self.local.branches[key]['quantizer'].get_embedding((code.long() - 1).clamp_min(0)) for key, code in codes.items()}
        else:
            decoded = latents
            if quantize:
                decoded = {}
                for key, latent in latents.items():
                    item = self.local.branches[key]['quantizer'](latent, valid_mask)
                    decoded[key] = torch.where(mask[..., None], item.quantized, 0.)
                    codes[key] = torch.where(mask, item.hard_ids + 1, 0)
                    commitment = commitment + item.commitment_loss
            output = self.decode_latents(decoded, dataset, mask)
        attributes = {key: self.attribute[key](value) for key, value in decoded.items()}
        return dict(output, ids=codes, commitment=commitment, attributes=attributes)


class NonlinearMontageInference:
    def __init__(self, model):
        self.model = model.eval()
        self.metadata = model.metadata
        self.codebook_size = model.codebook_size

    def encode(self, x, dataset, observed):
        with torch.inference_mode():
            return {k: v.numpy() for k, v in self.model.encode_ids(torch.as_tensor(x, dtype=torch.float32), dataset,
                                                                 torch.as_tensor(observed, dtype=torch.bool)).items()}

    def decode_ids(self, ids, dataset):
        with torch.inference_mode():
            return {k: v.numpy() for k, v in self.model.decode_ids({key: torch.as_tensor(value) for key, value in ids.items()}, dataset).items()}


def load_model(path):
    saved = torch.load(Path(path), map_location='cpu', weights_only=False)
    model = NonlinearMontageTokenizer(saved['model_config'], saved['metadata'])
    model.load_state_dict(saved['model_state'])
    return NonlinearMontageInference(model)
