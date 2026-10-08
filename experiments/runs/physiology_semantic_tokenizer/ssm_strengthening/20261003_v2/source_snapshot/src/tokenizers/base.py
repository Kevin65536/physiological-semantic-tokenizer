"""
Base class for neural signal tokenizers.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional


class BaseTokenizer(nn.Module, ABC):
    """
    Abstract base class for signal tokenizers.
    
    A tokenizer consists of:
    1. Encoder: Maps input signal to continuous latent
    2. Quantizer: Discretizes the latent to token indices
    3. Decoder: Reconstructs signal from quantized latent
    """
    
    def __init__(self, input_dim: int, latent_dim: int, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input signal to continuous latent representation.
        
        Args:
            x: Input signal [B, T] or [B, C, T]
            
        Returns:
            z: Continuous latent [B, T', D] or [B, D]
        """
        pass
    
    @abstractmethod
    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Quantize continuous latent to discrete tokens.
        
        Args:
            z: Continuous latent [B, T', D]
            
        Returns:
            z_q: Quantized latent [B, T', D]
            indices: Token indices [B, T']
            info: Dict with quantization info (loss, perplexity, etc.)
        """
        pass
    
    @abstractmethod
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latent back to signal space.
        
        Args:
            z_q: Quantized latent [B, T', D]
            
        Returns:
            x_rec: Reconstructed signal [B, T] or [B, C, T]
        """
        pass
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode -> quantize -> decode.
        
        Args:
            x: Input signal [B, T] or [B, C, T]
            
        Returns:
            Dict containing:
                - x_rec: Reconstructed signal
                - z: Continuous latent
                - z_q: Quantized latent
                - indices: Token indices
                - quantize_info: Dict with perplexity, commitment loss, etc.
        """
        z = self.encode(x)
        z_q, indices, quantize_info = self.quantize(z)
        x_rec = self.decode(z_q)
        
        return {
            'x_rec': x_rec,
            'z': z,
            'z_q': z_q,
            'indices': indices,
            **quantize_info
        }
    
    def get_codebook_size(self) -> int:
        """Return the effective codebook size."""
        raise NotImplementedError
    
    def get_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get embedding for given token indices.
        
        Args:
            indices: Token indices [B, T']
            
        Returns:
            embeddings: [B, T', D]
        """
        raise NotImplementedError


class Conv1dEncoder(nn.Module):
    """
    1D Convolutional encoder for time-series signals.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 128, 256],
        kernel_size: int = 7,
        stride: int = 2,
        output_dim: int = 64
    ):
        super().__init__()
        
        layers = []
        in_channels = input_dim
        
        for i, out_channels in enumerate(hidden_dims):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            ])
            in_channels = out_channels
        
        # Final projection
        layers.append(nn.Conv1d(in_channels, output_dim, 1))
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T] or [B, C, T]
            
        Returns:
            z: [B, T', D] where T' = T / (stride^num_layers)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, T]
        
        z = self.encoder(x)  # [B, D, T']
        z = z.transpose(1, 2)  # [B, T', D]
        return z


class Conv1dDecoder(nn.Module):
    """
    1D Transposed Convolutional decoder for time-series signals.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        kernel_size: int = 7,
        stride: int = 2,
        output_dim: int = 1,
        output_length: Optional[int] = None
    ):
        super().__init__()
        
        self.output_length = output_length
        
        layers = []
        in_channels = input_dim
        
        for i, out_channels in enumerate(hidden_dims):
            layers.extend([
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, output_padding=stride-1),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            ])
            in_channels = out_channels
        
        # Final projection
        layers.append(nn.Conv1d(in_channels, output_dim, 1))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: [B, T', D]
            
        Returns:
            x_rec: [B, T] or [B, C, T]
        """
        z_q = z_q.transpose(1, 2)  # [B, D, T']
        x_rec = self.decoder(z_q)  # [B, C, T_rec]
        
        # Adjust output length if needed
        if self.output_length is not None:
            if x_rec.shape[-1] > self.output_length:
                x_rec = x_rec[..., :self.output_length]
            elif x_rec.shape[-1] < self.output_length:
                x_rec = nn.functional.pad(x_rec, (0, self.output_length - x_rec.shape[-1]))
        
        # Squeeze channel dim if single channel
        if x_rec.shape[1] == 1:
            x_rec = x_rec.squeeze(1)  # [B, T]
        
        return x_rec
