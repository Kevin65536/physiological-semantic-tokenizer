"""Single-level EEG tokens with a full-sample decoder and train-only Wiener input."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ema_vector_quantizer import EMAVectorQuantizer


class FullbandEEGTokenizer(nn.Module):
    def __init__(self, setting, initialization=None):
        super().__init__()
        self.setting = setting
        self.codebook_size = int(setting['codebook_size'])
        self.latent_dim = int(setting['latent_dim'])
        self.register_buffer('transfer', torch.zeros(2, 3001, dtype=torch.complex64))
        self.register_buffer('band_floor', torch.ones(2, 5))
        self.register_buffer('rms_floor', torch.ones(2))
        self.branches = nn.ModuleDict()
        for branch in ('S', 'O'):
            encoder = nn.Sequential(nn.Linear(self.latent_dim, setting['hidden_width']), nn.GELU(),
                                    nn.Linear(setting['hidden_width'], self.latent_dim))
            decoder = nn.Sequential(nn.Linear(self.latent_dim, setting['hidden_width']), nn.GELU(),
                                    nn.Linear(setting['hidden_width'], 400))
            context_encoder = nn.Sequential(nn.Conv1d(self.latent_dim, self.latent_dim, 3, padding=1),
                                            nn.GELU(), nn.Conv1d(self.latent_dim, self.latent_dim, 1))
            context_decoder = nn.Sequential(nn.Conv1d(self.latent_dim, self.latent_dim, 3, padding=1),
                                            nn.GELU(), nn.Conv1d(self.latent_dim, 400, 1))
            modules = nn.ModuleDict(dict(project=nn.Linear(400, self.latent_dim), encoder=encoder,
                context_encoder=context_encoder, output=nn.Linear(self.latent_dim, 400), decoder=decoder,
                context_decoder=context_decoder,
                quantizer=EMAVectorQuantizer(codebook_size=self.codebook_size, embedding_dim=self.latent_dim,
                    decay=setting['ema_decay'], commitment_cost=setting['commitment_cost'],
                    revive_dead_codes=True, dead_code_threshold=.1, revival_warmup_steps=64,
                    revival_interval=64, revival_stop_after_steps=512, max_revivals_per_event=8)))
            self.branches[branch] = modules
            for residual in (encoder, decoder, context_encoder, context_decoder):
                nn.init.zeros_(residual[-1].weight)
                nn.init.zeros_(residual[-1].bias)
        if initialization is not None:
            self.initialize(initialization)

    @torch.no_grad()
    def initialize(self, values):
        self.transfer.copy_(torch.as_tensor(values['transfer'], dtype=self.transfer.dtype))
        self.band_floor.copy_(torch.as_tensor(values['band_floor']))
        self.rms_floor.copy_(torch.as_tensor(values['rms_floor']))
        for b in ('S', 'O'):
            branch = self.branches[b]
            basis = torch.as_tensor(values[b+'_basis'], dtype=torch.float32)
            center = torch.as_tensor(values[b+'_mean'], dtype=torch.float32)
            scale = torch.as_tensor(values[b+'_scale'], dtype=torch.float32)
            branch['project'].weight.copy_(basis / scale[:, None])
            branch['project'].bias.copy_(-(basis @ center) / scale)
            branch['output'].weight.copy_(basis.T * scale[None])
            branch['output'].bias.copy_(center)

    @staticmethod
    def checked_values(values):
        if values.ndim != 3 or values.shape[1:] != (15, 400) or not torch.isfinite(values).all():
            raise ValueError('finite observed EEG [batch,15,400] required')
        return values

    def encode_continuous(self, values):
        self.checked_values(values)
        spectrum = torch.fft.rfft(values.reshape(len(values), 6000), dim=-1)
        encoded = {}
        for n, b in enumerate(('S', 'O')):
            predicted = torch.fft.irfft(spectrum*self.transfer[n], n=6000, dim=-1).reshape(values.shape)
            branch = self.branches[b]
            latent = branch['project'](predicted)
            encoded[b] = latent + branch['encoder'](latent) + branch['context_encoder'](latent.transpose(1, 2)).transpose(1, 2)
        return encoded

    def decode_latents(self, latents):
        result = {}
        for b in ('S', 'O'):
            branch, latent = self.branches[b], latents[b]
            result[b] = (branch['output'](latent) + branch['decoder'](latent)
                         + branch['context_decoder'](latent.transpose(1, 2)).transpose(1, 2))
        result['sum'] = result['S'] + result['O']
        return result

    def encode_ids(self, values):
        return {b: self.branches[b]['quantizer']._assignment_logits(latent).argmax(-1)
                for b, latent in self.encode_continuous(values).items()}

    def decode_ids(self, ids):
        if set(ids) != {'S', 'O'}:
            raise ValueError('only S/O integer sequences are accepted')
        latents = {}
        for b in ('S', 'O'):
            index = ids[b]
            if (index.ndim != 2 or index.shape[1] != 15 or index.dtype not in (torch.int32, torch.int64)
                    or bool(((index < 0) | (index >= self.codebook_size)).any())):
                raise ValueError('valid integer [batch,15] IDs required')
            if index.shape != ids['S'].shape:
                raise ValueError('S/O integer supports differ')
            latents[b] = self.branches[b]['quantizer'].get_embedding(index.long())
        return self.decode_latents(latents)

    def forward(self, values, *, quantize=True, valid_mask=None):
        latents = self.encode_continuous(values)
        if not quantize:
            return dict(self.decode_latents(latents), commitment=values.new_zeros(()))
        quantized, codes, penalties = {}, {}, []
        for b, latent in latents.items():
            if self.training:
                item = self.branches[b]['quantizer'](latent, valid_mask)
                quantized[b], codes[b] = item.quantized, item.hard_ids
                penalties.append(item.commitment_loss)
            else:
                codes[b] = self.branches[b]['quantizer']._assignment_logits(latent).argmax(-1)
                quantized[b] = self.branches[b]['quantizer'].get_embedding(codes[b])
        return dict(self.decode_latents(quantized), ids=codes,
                    commitment=sum(penalties) if penalties else values.new_zeros(()))


def fullband_loss(model, prediction, batch, weights):
    """Training-variance waveform error, complete-patch bands, and local RMS."""
    valid = batch['mask']
    complete = valid.all(-1)
    window = torch.hann_window(400, periodic=False, device=valid.device)
    freq = torch.fft.rfftfreq(400, d=1/200).to(valid.device)
    point, spectral, envelope = [], [], []
    terms = {}
    for n, (b, target) in enumerate((('S', 's'), ('O', 'o'), ('sum', 'x'))):
        y = torch.where(valid, batch[target], 0.)
        p = prediction[b]
        variance = batch['v'+target]
        mse = torch.where(valid, (p-y).square()/variance, 0.).sum()/valid.sum().clamp_min(1)
        point.append(mse if b != 'sum' else weights['sum']*mse)
        terms[target+'_nmse'] = float(mse.detach())
        if b == 'sum':
            continue
        yy, pp = y-y.mean(-1, keepdim=True), p-p.mean(-1, keepdim=True)
        yf, pf = torch.fft.rfft(yy*window), torch.fft.rfft(pp*window)
        for j, (lo, hi) in enumerate(((1, 4), (4, 8), (8, 13), (13, 30), (30, 45.001))):
            band = (freq >= lo) & (freq < hi)
            ey, ep = yf[..., band].abs().square().sum(-1), pf[..., band].abs().square().sum(-1)
            floor = model.band_floor[n, j]
            error = (torch.log1p(ep/floor)-torch.log1p(ey/floor)).abs()
            spectral.append(torch.where(complete, error, 0.).sum()/complete.sum().clamp_min(1))
        patch_shape = (*y.shape[:2], 8, 50)
        env_mask = valid.reshape(patch_shape).all(-1)
        ry = torch.sqrt(y.reshape(patch_shape).square().mean(-1)+1e-8)
        rp = torch.sqrt(p.reshape(patch_shape).square().mean(-1)+1e-8)
        loss = (torch.log1p(rp/model.rms_floor[n])-torch.log1p(ry/model.rms_floor[n])).abs()
        envelope.append(torch.where(env_mask, loss, 0.).sum()/env_mask.sum().clamp_min(1))
    spectral_loss = torch.stack(spectral).mean()
    envelope_loss = torch.stack(envelope).mean()
    loss = sum(point) + weights['band']*spectral_loss + weights['envelope']*envelope_loss + prediction['commitment']
    terms.update(band=float(spectral_loss.detach()), envelope=float(envelope_loss.detach()),
                 commitment=float(prediction['commitment'].detach()), total=float(loss.detach()))
    return loss, terms


class FullbandEEGInference:
    def __init__(self, model, metadata):
        self.model, self.metadata = model.eval(), metadata
        self.codebook_size = model.codebook_size

    def encode(self, values):
        with torch.inference_mode():
            return {b: ids.numpy() for b, ids in self.model.encode_ids(torch.as_tensor(values, dtype=torch.float32)).items()}

    def decode_ids(self, ids):
        with torch.inference_mode():
            return {b: value.numpy() for b, value in self.model.decode_ids({b: torch.as_tensor(v) for b, v in ids.items()}).items()}

    def predict_continuous(self, values):
        with torch.inference_mode():
            result = self.model(torch.as_tensor(values, dtype=torch.float32), quantize=False)
        return {b: result[b].numpy() for b in ('S', 'O', 'sum')}

    def __call__(self, values):
        ids = self.encode(values)
        return dict(self.decode_ids(ids), ids=ids)


def load_model(path):
    retained = torch.load(Path(path), map_location='cpu', weights_only=False)
    if retained['metadata']['schema'] != 'fullband_eeg_tokenizer_v1':
        raise ValueError('versioned fullband EEG checkpoint required')
    model = FullbandEEGTokenizer(retained['model_config'])
    model.load_state_dict(retained['model_state'])
    return FullbandEEGInference(model, retained['metadata'])
