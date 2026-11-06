"""Opt-in STA-Net correction against the pinned author's TensorFlow 2.10 code.

The retained model.py and all historical checkpoints keep their original
semantics. This version corrects positional encoding order/initialization,
channels-last flatten order, Keras batch statistics, and Pearson epsilon.
It remains a PyTorch task adaptation, not a bitwise TensorFlow reproduction.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .model import STANet, EgtaLayer, FgsaLayer


def source_pearson(x, y):
    """Author's mean absolute feature-wise correlation, including additive eps."""
    x, y = x.float(), y.float()
    x, y = x - x.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
    denominator = x.std(1, unbiased=False) * y.std(1, unbiased=False) + 1e-6
    return ((x * y).mean(1) / denominator).abs().mean()


class SourceBatchNorm3d(nn.BatchNorm3d):
    """Keras documented population-variance moving update, in float32."""

    def __init__(self, channels):
        super().__init__(channels, eps=1e-3, momentum=.01)

    def forward(self, value):
        x = value.float()
        if self.training:
            variance, mean = torch.var_mean(x, dim=(0, 2, 3, 4), unbiased=False)
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_var.lerp_(variance.detach(), self.momentum)
                self.num_batches_tracked.add_(1)
        else:
            mean, variance = self.running_mean, self.running_var
        shape = (1, -1, 1, 1, 1)
        normalized = (x - mean.reshape(shape)) * torch.rsqrt(variance.reshape(shape) + self.eps)
        return (normalized * self.weight.reshape(shape) + self.bias.reshape(shape)).to(value.dtype)


class SourceFgsa(FgsaLayer):
    def forward(self, eeg_fusion, eeg, fnirs):
        batch, lags, channels, height, width, time = fnirs.shape
        pooled = self.channel_pooling(fnirs.reshape(batch * lags, channels, height, width, time))
        pooled = pooled.reshape(batch, lags, 1, *pooled.shape[-3:])
        attention = torch.sigmoid(pooled.mean(-1, keepdim=True).mean(1))
        weight = torch.sigmoid(self.residual_parameter)
        aligned = eeg_fusion * attention + weight * eeg + (1 - weight) * eeg_fusion
        loss = 1 - source_pearson(eeg.mean((1, 4)).flatten(1), attention.flatten(1))
        return aligned, loss, attention


def source_flatten(value):
    """Convert [..., C, H, W, T] to the author's [..., H, W, T, C]."""
    return value.movedim(-4, -1).flatten(-4)


class SourceEgta(EgtaLayer):
    def __init__(self, flat_dim, embedding_dim, num_heads, key_dim, dropout, max_lags):
        super().__init__(flat_dim, embedding_dim, num_heads, key_dim, dropout, max_lags)
        # Keras HeUniform's fan_in for [1, lag, flat_feature] is lag.
        self.position = nn.Parameter(torch.empty(1, max_lags, flat_dim))
        nn.init.uniform_(self.position, -math.sqrt(6 / max_lags), math.sqrt(6 / max_lags))

    def forward(self, eeg_fusion, fnirs):
        eeg_flat, fnirs_flat = source_flatten(eeg_fusion), source_flatten(fnirs)
        lags = fnirs_flat.shape[1]
        if lags > self.position.shape[1]:
            raise ValueError("fNIRS lags exceed the explicit source-adapted support")
        fusion = self.fusion_projection(eeg_flat)
        query = self.query_projection(eeg_flat).unsqueeze(1)
        keys = self.key_projection(fnirs_flat + self.position[:, :lags])
        weighted, attention = self.attention(query, keys)
        weighted = weighted.mean(1)
        return fusion, weighted, attention.mean((1, 2)), 1 - source_pearson(query.mean(1), weighted)


class SourceAlignedSTANet(STANet):
    def __init__(self, config):
        super().__init__(config)
        for block, channels, kernel in ((self.block1, 16, 5), (self.block2, 32, 3)):
            for branch in (block.eeg, block.fnirs, block.eeg_fusion):
                branch.norm = SourceBatchNorm3d(channels)
            block.fgsa = SourceFgsa(channels, kernel)
        self.egta = SourceEgta(32 * 4 * 4, config.embedding_dim, config.attention_heads,
                               config.attention_key_dim, config.dropout, config.max_lags)
        self._reset_parameters()
        # Keras EinsumDense keeps a 3-D attention kernel when computing fans.
        attention = self.egta.attention
        bound = math.sqrt(6 / (config.embedding_dim * (config.attention_heads + config.attention_key_dim)))
        for layer in (attention.query, attention.key, attention.value):
            nn.init.uniform_(layer.weight, -bound, bound)
        bound = math.sqrt(6 / (config.attention_heads * (config.attention_key_dim + config.embedding_dim)))
        nn.init.uniform_(attention.output.weight, -bound, bound)

    def forward(self, eeg, fnirs):
        fusion1, eeg1, fnirs1, fgsa1_loss, attention1 = self.block1(eeg, eeg, fnirs)
        fusion1, eeg1, fnirs1 = self.dropout(fusion1), self.dropout(eeg1), self.dropout(fnirs1)
        fusion2, eeg2, fnirs2, fgsa2_loss, attention2 = self.block2(fusion1, eeg1, fnirs1)
        fusion2, eeg2, fnirs2 = [self.dropout(x.mean(-1, keepdim=True)) for x in (fusion2, eeg2, fnirs2)]
        fusion, optical, lag_attention, egta_loss = self.egta(fusion2, fnirs2)
        fusion, optical = self.fusion_hidden(fusion), self.fnirs_hidden(optical)
        electrical = self.eeg_hidden(source_flatten(eeg2))
        raw_fusion = self._reshape_prediction(self.fusion_head(fusion))
        raw_optical = self._reshape_prediction(self.fnirs_head(optical))
        raw_eeg = self._reshape_prediction(self.eeg_head(electrical))
        gates = torch.softmax(torch.cat([self.fusion_gate(fusion), self.fnirs_gate(optical)], 1), 1)
        if self.config.task_type != "classification":
            raise ValueError("source-aligned v2 is an explicit classification adaptation")
        fusion_prediction, optical_prediction, eeg_prediction = [torch.softmax(x, -1) for x in (raw_fusion, raw_optical, raw_eeg)]
        return dict(prediction=gates[:, :1] * fusion_prediction + gates[:, 1:] * optical_prediction,
            eeg_prediction=eeg_prediction, fusion_prediction=fusion_prediction, fnirs_prediction=optical_prediction,
            fusion_weights=gates, lag_attention=lag_attention, spatial_attention_block1=attention1,
            spatial_attention_block2=attention2,
            alignment_losses=dict(fgsa1=fgsa1_loss, fgsa2=fgsa2_loss, egta=egta_loss))
