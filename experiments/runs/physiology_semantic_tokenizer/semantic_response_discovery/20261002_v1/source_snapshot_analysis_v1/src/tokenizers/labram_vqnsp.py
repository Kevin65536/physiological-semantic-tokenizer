"""
LaBraM-style VQ-NSP (Vector Quantized Neural Signal Processing) Tokenizer.

This implementation is based on the VQNSP architecture from LaBraM:
"Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI"

Key Design Principles:
1. Simple architecture: Transformer encoder -> Single VQ -> Transformer decoder
2. L2-normalized EMA codebook (NormEMAVectorQuantizer) - more stable training
3. Frequency-domain reconstruction (amplitude + phase separately)
4. No complex multi-branch or multi-layer quantization

This is designed to be a stable, easy-to-train baseline for EEG/fNIRS tokenization.

Reference:
- LaBraM: https://github.com/935963004/LaBraM
- NeuroLM: https://github.com/935963004/NeuroLM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from einops import rearrange
import math

from .base import BaseTokenizer


# =============================================================================
# Utility Functions
# =============================================================================

def l2norm(t: torch.Tensor) -> torch.Tensor:
    """L2 normalize along the last dimension."""
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
    """Exponential moving average update in-place."""
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    """Sample vectors for k-means initialization."""
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def kmeans(
    samples: torch.Tensor,
    num_clusters: int,
    num_iters: int = 10,
    use_cosine_sim: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """K-means clustering for codebook initialization."""
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device
    means = sample_vectors(samples, num_clusters)
    
    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
            buckets = dists.max(dim=-1).indices
        else:
            diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)
            buckets = dists.max(dim=-1).indices
        
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)
        
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, buckets.unsqueeze(-1).expand(-1, dim), samples)
        new_means = new_means / bins_min_clamped[..., None]
        
        if use_cosine_sim:
            new_means = l2norm(new_means)
        
        means = torch.where(zero_mask[..., None], means, new_means)
    
    return means, bins


# =============================================================================
# NormEMAVectorQuantizer - Following LaBraM
# =============================================================================

class NormEMAVectorQuantizer(nn.Module):
    """
    L2-normalized EMA Vector Quantizer from LaBraM.
    
    Key features:
    1. L2 normalization of both input and codebook vectors
    2. EMA-based codebook updates (no gradient through codebook)
    3. K-means initialization for better starting point
    4. Cosine similarity for distance computation
    5. Dead code revival to prevent codebook collapse
    
    This is more stable than standard VQ with gradient-based updates.
    """
    
    def __init__(
        self,
        n_embed: int = 8192,
        embedding_dim: int = 64,
        beta: float = 1.0,
        decay: float = 0.99,
        eps: float = 1e-5,
        kmeans_init: bool = True,
        dead_code_threshold: float = 1.0,
        revive_dead_codes: bool = True,
        learnable_codebook_transform: bool = False,
        codebook_transform_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.num_tokens = n_embed
        self.codebook_dim = embedding_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold
        self.revive_dead_codes = revive_dead_codes
        self.learnable_codebook_transform = bool(learnable_codebook_transform)
        self.codebook_transform_loss_weight = max(float(codebook_transform_loss_weight), 0.0)
        self.quantization_strength = 1.0
        self.ema_updates_enabled = True
        
        # Codebook initialization
        if kmeans_init:
            weight = torch.zeros(n_embed, embedding_dim)
        else:
            weight = torch.randn(n_embed, embedding_dim)
            weight = l2norm(weight)
        
        self.register_buffer('weight', weight)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('initted', torch.tensor([not kmeans_init]))
        self.register_buffer('update_count', torch.tensor([0]))
        if self.learnable_codebook_transform:
            self.codebook_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
            with torch.no_grad():
                self.codebook_transform.weight.copy_(torch.eye(embedding_dim))
        else:
            self.codebook_transform = None
    
    def init_embed_(self, data: torch.Tensor):
        """Initialize codebook with k-means if not already done."""
        if self.initted.item():
            return
        
        print("Performing K-means init for codebook...")
        embed, cluster_size = kmeans(data, self.num_tokens, num_iters=10, use_cosine_sim=True)
        self.weight.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.fill_(1)
    
    def _revive_dead_codes(self, z: torch.Tensor):
        """
        Revive dead codes by reinitializing them with samples from encoder output.
        This prevents codebook collapse.
        """
        if not self.training or not self.revive_dead_codes:
            return
        
        # Only revive every 100 updates to avoid too much disruption
        if self.update_count.item() % 100 != 0:
            return
        
        # Find dead codes
        dead_mask = self.cluster_size < self.dead_code_threshold
        n_dead = dead_mask.sum().item()
        
        if n_dead == 0:
            return
        
        # Sample from encoder outputs to reinitialize dead codes
        n_samples = min(n_dead, z.shape[0])
        if n_samples > 0:
            # Randomly sample from z
            perm = torch.randperm(z.shape[0], device=z.device)[:n_samples]
            samples = z[perm]
            
            # Add small noise for diversity
            noise = torch.randn_like(samples) * 0.01
            samples = l2norm(samples + noise)
            
            # Find dead code indices
            dead_indices = torch.where(dead_mask)[0][:n_samples]
            
            # Reinitialize
            self.weight.data[dead_indices] = samples
            self.cluster_size[dead_indices] = self.dead_code_threshold

    def get_codebook_weight(self) -> torch.Tensor:
        """Return the effective codebook used for assignments and lookups."""
        weight = self.weight
        if self.codebook_transform is not None:
            weight = self.codebook_transform(weight)
            weight = l2norm(weight)
        return weight

    def set_quantization_strength(self, strength: float):
        self.quantization_strength = min(max(float(strength), 0.0), 1.0)

    def get_quantization_strength(self) -> float:
        return float(self.quantization_strength)

    def set_ema_updates_enabled(self, enabled: bool):
        self.ema_updates_enabled = bool(enabled)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Quantize input.
        
        Args:
            z: [B, N, D] or [B*N, D] continuous latent
            
        Returns:
            z_q: Quantized latent (same shape as z)
            indices: Token indices [B*N] or [B, N]
            info: Dict with loss and perplexity
        """
        need_reshape = z.dim() == 3
        if need_reshape:
            B, N, D = z.shape
            z = z.view(B * N, D)
        
        # L2 normalize input
        z = l2norm(z)
        
        # Initialize codebook with k-means on first batch
        self.init_embed_(z.detach())

        codebook_weight = self.get_codebook_weight()
        
        # Compute distances using cosine similarity (since both are L2 normalized)
        # Distance = 2 - 2 * cosine_sim = 2 - 2 * (z @ weight.T)
        # Minimizing distance = Maximizing cosine similarity
        sim = z @ codebook_weight.t()  # [B*N, K]
        indices = sim.argmax(dim=-1)  # [B*N]
        
        # Get quantized vectors
        z_q_lookup = F.embedding(indices, codebook_weight)  # [B*N, D]
        
        # Compute loss (commitment/codebook terms can be progressively ramped in).
        quantization_strength = self.get_quantization_strength()
        commitment_loss = quantization_strength * self.beta * F.mse_loss(z_q_lookup.detach(), z)
        codebook_loss = z.new_tensor(0.0)
        if self.codebook_transform is not None and self.codebook_transform_loss_weight > 0.0:
            codebook_loss = (
                quantization_strength * self.codebook_transform_loss_weight * F.mse_loss(z_q_lookup, z.detach())
            )
        vq_loss = commitment_loss + codebook_loss
        
        # EMA codebook update
        if self.training and self.ema_updates_enabled:
            # One-hot encoding
            encodings = F.one_hot(indices, self.num_tokens).float()  # [B*N, K]
            
            # Update cluster sizes
            cluster_size = encodings.sum(0)
            ema_inplace(self.cluster_size, cluster_size, self.decay)
            
            # Update codebook
            embed_sum = encodings.t() @ z.detach()  # [K, D]
            
            # Laplace smoothing
            n = self.cluster_size.sum()
            cluster_size_smooth = (
                (self.cluster_size + self.eps) / (n + self.num_tokens * self.eps) * n
            )
            
            # Normalize and update
            embed_normalized = embed_sum / cluster_size_smooth.unsqueeze(1).clamp(min=1)
            embed_normalized = l2norm(embed_normalized)
            
            # Only update non-dead codes
            active_mask = self.cluster_size > 0.1
            self.weight.data[active_mask] = embed_normalized[active_mask]
            
            # Revive dead codes periodically
            self._revive_dead_codes(z.detach())
            self.update_count += 1
        
        # Progressive quantization interpolates between continuous latents and hard VQ outputs.
        z_q = z + quantization_strength * (z_q_lookup - z).detach()
        
        # Compute perplexity
        with torch.no_grad():
            encodings = F.one_hot(indices, self.num_tokens).float()
            avg_probs = encodings.mean(0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            
            # Code utilization
            used_codes = (self.cluster_size > 0.1).sum()
            utilization = used_codes.float() / self.num_tokens
        
        if need_reshape:
            z_q = z_q.view(B, N, D)
            indices = indices.view(B, N)
        
        info = {
            'vq_loss': vq_loss,
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'quantization_strength': z.new_tensor(quantization_strength),
            'perplexity': perplexity,
            'utilization': utilization,
        }
        
        return z_q, indices, info
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Get codebook vectors for given indices."""
        return F.embedding(indices, self.get_codebook_weight())


# =============================================================================
# Transformer Components (simplified from LaBraM)
# =============================================================================

class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-norm."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
        # Stochastic depth
        self.drop_path_rate = drop_path
    
    def drop_path(self, x: torch.Tensor) -> torch.Tensor:
        """Drop paths (Stochastic Depth) per sample."""
        if not self.training or self.drop_path_rate == 0:
            return x
        keep_prob = 1 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop_path(attn_out)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class PatchEmbedding(nn.Module):
    """Embed patches into transformer dimension."""
    
    def __init__(
        self,
        patch_size: int = 200,
        embed_dim: int = 256,
        use_frequency: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.use_frequency = use_frequency
        
        if use_frequency:
            # FFT-based input (amplitude + phase)
            self.fft_size = patch_size // 2 + 1
            self.proj = nn.Linear(self.fft_size * 2, embed_dim)  # amp + phase
        else:
            # Time-domain input
            self.proj = nn.Linear(patch_size, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, T] patches (N patches, T samples each)
        Returns:
            embeddings: [B, N, D]
        """
        if self.use_frequency:
            # Compute FFT
            fft = torch.fft.rfft(x, dim=-1)
            amplitude = torch.log(torch.abs(fft) + 1e-8)  # Log amplitude
            phase = torch.angle(fft) / math.pi  # Normalized phase
            
            # Concatenate amplitude and phase
            freq_features = torch.cat([amplitude, phase], dim=-1)  # [B, N, F*2]
            embeddings = self.proj(freq_features)
        else:
            embeddings = self.proj(x)
        
        return embeddings


class MultiChannelPatchEmbedding(nn.Module):
    """Embed multi-channel temporal patches into a transformer space."""

    def __init__(
        self,
        input_channels: int,
        patch_size: int,
        embed_dim: int,
        use_frequency: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.use_frequency = use_frequency
        self.fft_size = patch_size // 2 + 1

        if use_frequency:
            input_dim = input_channels * self.fft_size * 2
        else:
            input_dim = input_channels * patch_size
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: [B, N, C, P]
        Returns:
            [B, N, D]
        """
        if self.use_frequency:
            fft = torch.fft.rfft(patches, dim=-1)
            amplitude = torch.log(torch.abs(fft) + 1e-8)
            phase = torch.angle(fft) / math.pi
            features = torch.cat([amplitude, phase], dim=-1)
        else:
            features = patches

        return self.proj(features.flatten(start_dim=2))


class TransformerEncoder(nn.Module):
    """Transformer encoder for patches."""
    
    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.1,
        max_patches: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] patch embeddings
        Returns:
            features: [B, N, D]
        """
        # Add positional embedding
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder for reconstruction."""
    
    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        max_patches: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] quantized embeddings
        Returns:
            features: [B, N, D]
        """
        # Add positional embedding
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x


# =============================================================================
# Main Tokenizer: LaBraM-style VQNSP
# =============================================================================

class LaBraMVQNSP(BaseTokenizer):
    """
    LaBraM-style VQ-NSP Tokenizer.
    
    Architecture:
    1. Split input into non-overlapping patches
    2. FFT -> (amplitude, phase) as input features
    3. Transformer encoder
    4. Project to quantizer dimension -> NormEMA VQ
    5. Project back -> Transformer decoder
    6. Predict amplitude and phase separately
    7. Optional: iFFT for time-domain reconstruction
    
    This is a simplified, stable version suitable for EEG and fNIRS tokenization.
    """
    
    def __init__(
        self,
        # Patch parameters
        patch_size: int = 200,
        seq_length: int = 800,
        
        # Model parameters
        encoder_embed_dim: int = 256,
        encoder_depth: int = 6,
        encoder_num_heads: int = 8,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 3,
        decoder_num_heads: int = 8,
        
        # Quantizer parameters
        codebook_size: int = 8192,
        codebook_dim: int = 64,
        beta: float = 1.0,
        decay: float = 0.99,
        kmeans_init: bool = True,
        revive_dead_codes: bool = True,
        dead_code_threshold: int = 10,
        
        # Loss weights
        amplitude_weight: float = 1.0,
        phase_weight: float = 1.0,
        time_weight: float = 0.0,  # Optional time-domain loss
        
        # Other
        dropout: float = 0.0,
        drop_path: float = 0.1,
        use_smooth_l1: bool = False,
        **kwargs
    ):
        super().__init__(input_dim=1, latent_dim=codebook_dim)
        
        self.patch_size = patch_size
        self.seq_length = seq_length
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.fft_size = patch_size // 2 + 1
        
        # Loss weights
        self.amplitude_weight = amplitude_weight
        self.phase_weight = phase_weight
        self.time_weight = time_weight
        
        # Loss function
        self.loss_fn = F.smooth_l1_loss if use_smooth_l1 else F.mse_loss
        
        # Calculate number of patches
        assert seq_length % patch_size == 0, \
            f"seq_length ({seq_length}) must be divisible by patch_size ({patch_size})"
        self.n_patches = seq_length // patch_size
        
        # Patch embedding (FFT-based)
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            embed_dim=encoder_embed_dim,
            use_frequency=True,
        )
        
        # Encoder
        self.encoder = TransformerEncoder(
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            dropout=dropout,
            drop_path=drop_path,
            max_patches=self.n_patches,
        )
        
        # Encoder to quantizer projection (following LaBraM)
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_embed_dim, encoder_embed_dim),
            nn.Tanh(),
            nn.Linear(encoder_embed_dim, codebook_dim),
        )
        
        # Quantizer
        self.quantizer = NormEMAVectorQuantizer(
            n_embed=codebook_size,
            embedding_dim=codebook_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
            revive_dead_codes=revive_dead_codes,
            dead_code_threshold=dead_code_threshold,
        )
        
        # Quantizer to decoder projection
        self.decode_input_proj = nn.Linear(codebook_dim, decoder_embed_dim)
        
        # Decoder
        self.decoder = TransformerDecoder(
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            dropout=dropout,
            drop_path=0.0,  # No drop path in decoder
            max_patches=self.n_patches,
        )
        
        # Output heads (separate for amplitude and phase, like LaBraM)
        self.amplitude_head = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            nn.Tanh(),
            nn.Linear(decoder_embed_dim, self.fft_size),
        )
        
        self.phase_head = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            nn.Tanh(),
            nn.Linear(decoder_embed_dim, self.fft_size),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Print summary
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*50}")
        print(f"LaBraM-style VQNSP Tokenizer")
        print(f"{'='*50}")
        print(f"  Patch size: {patch_size} samples")
        print(f"  Sequence length: {seq_length} samples")
        print(f"  Patches per sequence: {self.n_patches}")
        print(f"  Encoder: {encoder_depth} layers, {encoder_embed_dim}D, {encoder_num_heads} heads")
        print(f"  Decoder: {decoder_depth} layers, {decoder_embed_dim}D, {decoder_num_heads} heads")
        print(f"  Codebook: {codebook_size} codes x {codebook_dim}D")
        print(f"  Loss weights: amp={amplitude_weight}, phase={phase_weight}, time={time_weight}")
        print(f"  Total parameters: {total_params:,}")
        print(f"{'='*50}\n")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _split_to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Split input into patches. [B, T] -> [B, N, P]"""
        B = x.shape[0]
        return x.view(B, self.n_patches, self.patch_size)
    
    def _compute_fft_targets(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute FFT amplitude and phase as reconstruction targets.
        
        Args:
            patches: [B, N, P] time-domain patches
        Returns:
            amplitude: [B, N, F] log amplitude
            phase: [B, N, F] normalized phase [-1, 1]
        """
        fft = torch.fft.rfft(patches, dim=-1)
        amplitude = torch.log(torch.abs(fft) + 1e-8)
        phase = torch.angle(fft) / math.pi
        return amplitude, phase
    
    def _reconstruct_time(
        self,
        amplitude: torch.Tensor,
        phase: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct time-domain signal from predicted amplitude and phase.
        
        Args:
            amplitude: [B, N, F] log amplitude
            phase: [B, N, F] normalized phase [-1, 1]
        Returns:
            signal: [B, T] reconstructed signal
        """
        # De-normalize
        amp = torch.exp(amplitude)
        pha = phase * math.pi
        
        # Construct complex FFT
        real = amp * torch.cos(pha)
        imag = amp * torch.sin(pha)
        fft = torch.complex(real, imag)
        
        # iFFT
        patches = torch.fft.irfft(fft, n=self.patch_size, dim=-1)
        
        # Merge patches
        B = patches.shape[0]
        signal = patches.view(B, -1)
        return signal
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to continuous latent.
        
        Args:
            x: [B, T] input signal
        Returns:
            z: [B, N, D] continuous latent
        """
        # Split to patches
        patches = self._split_to_patches(x)  # [B, N, P]
        
        # Embed patches (FFT-based)
        embeddings = self.patch_embed(patches)  # [B, N, embed_dim]
        
        # Transformer encoder
        encoder_out = self.encoder(embeddings)  # [B, N, embed_dim]
        
        # Project to quantizer dimension
        z = self.encode_task_layer(encoder_out)  # [B, N, codebook_dim]
        
        return z
    
    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Quantize continuous latent.
        
        Args:
            z: [B, N, D] continuous latent
        Returns:
            z_q: [B, N, D] quantized latent
            indices: [B, N] token indices
            info: Dict with quantization metrics
        """
        return self.quantizer(z)
    
    def decode(self, z_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode quantized latent to frequency domain predictions.
        
        Args:
            z_q: [B, N, D] quantized latent
        Returns:
            pred_amplitude: [B, N, F] predicted log amplitude
            pred_phase: [B, N, F] predicted normalized phase
        """
        # Project to decoder dimension
        decoder_in = self.decode_input_proj(z_q)  # [B, N, decoder_dim]
        
        # Transformer decoder
        decoder_out = self.decoder(decoder_in)  # [B, N, decoder_dim]
        
        # Predict amplitude and phase
        pred_amplitude = self.amplitude_head(decoder_out)
        pred_phase = self.phase_head(decoder_out)
        
        return pred_amplitude, pred_phase
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            x: [B, T] input signal
        Returns:
            Dict with reconstruction, losses, and metrics
        """
        B = x.shape[0]
        
        # Split and get targets
        patches = self._split_to_patches(x)
        target_amp, target_phase = self._compute_fft_targets(patches)
        
        # Encode
        z = self.encode(x)
        
        # Quantize
        z_q, indices, quant_info = self.quantize(z)
        
        # Decode
        pred_amp, pred_phase = self.decode(z_q)
        
        # Compute losses
        amp_loss = self.loss_fn(pred_amp, target_amp)
        phase_loss = self.loss_fn(pred_phase, target_phase)
        
        # Reconstruction loss
        rec_loss = self.amplitude_weight * amp_loss + self.phase_weight * phase_loss
        
        # Optional time-domain loss
        time_loss = torch.tensor(0.0, device=x.device)
        if self.time_weight > 0:
            x_rec = self._reconstruct_time(pred_amp, pred_phase)
            # Ensure x is squeezed to match x_rec shape
            x_squeezed = x.squeeze(1) if x.dim() == 3 else x
            time_loss = self.loss_fn(x_rec, x_squeezed)
            rec_loss = rec_loss + self.time_weight * time_loss
        
        # Total loss
        vq_loss = quant_info['vq_loss']
        total_loss = rec_loss + vq_loss
        
        # Reconstruct signal for visualization
        with torch.no_grad():
            x_rec = self._reconstruct_time(pred_amp, pred_phase)
        
        return {
            'loss': total_loss,
            'rec_loss': rec_loss,
            'amp_loss': amp_loss,
            'phase_loss': phase_loss,
            'time_loss': time_loss,
            'vq_loss': vq_loss,
            'reconstructed': x_rec,
            'x_rec': x_rec,
            'indices': indices,
            'tokens': indices,
            'z': z,
            'z_q': z_q,
            'utilization': quant_info['utilization'],
            'perplexity': quant_info['perplexity'],
        }
    
    def get_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Get token indices for input signal."""
        z = self.encode(x)
        _, indices, _ = self.quantize(z)
        return indices
    
    def get_codebook_size(self) -> int:
        """Return codebook size."""
        return self.codebook_size
    
    def get_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        """Get embeddings for given token indices."""
        return self.quantizer.get_codebook_entry(indices)
    
    def decode_from_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from token indices to signal."""
        z_q = self.get_embedding(indices)
        pred_amp, pred_phase = self.decode(z_q)
        x_rec = self._reconstruct_time(pred_amp, pred_phase)
        return x_rec


# =============================================================================
# Variants for different modalities
# =============================================================================

class LaBraMVQNSP_EEG(LaBraMVQNSP):
    """LaBraM VQNSP configured for EEG (200Hz, 200 samples/patch = 1s)."""
    
    def __init__(
        self,
        patch_size: int = 200,
        seq_length: int = 800,
        codebook_size: int = 8192,
        codebook_dim: int = 64,
        encoder_depth: int = 6,
        decoder_depth: int = 3,
        **kwargs
    ):
        # EEG-optimized defaults
        defaults = {
            'encoder_embed_dim': 256,
            'encoder_num_heads': 8,
            'decoder_embed_dim': 256,
            'decoder_num_heads': 8,
            'amplitude_weight': 1.0,
            'phase_weight': 1.0,
            'time_weight': 0.5,
            'beta': 1.0,
            'decay': 0.99,
        }
        defaults.update(kwargs)
        
        super().__init__(
            patch_size=patch_size,
            seq_length=seq_length,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            **defaults
        )


class LaBraMVQNSP_fNIRS(LaBraMVQNSP):
    """LaBraM VQNSP configured for fNIRS (10Hz, 40 samples/patch = 4s)."""
    
    def __init__(
        self,
        patch_size: int = 40,
        seq_length: int = 200,
        codebook_size: int = 4096,
        codebook_dim: int = 32,
        encoder_depth: int = 4,
        decoder_depth: int = 2,
        **kwargs
    ):
        # fNIRS-optimized defaults (smaller model for lower sample rate)
        defaults = {
            'encoder_embed_dim': 128,
            'encoder_num_heads': 4,
            'decoder_embed_dim': 128,
            'decoder_num_heads': 4,
            'amplitude_weight': 1.0,
            'phase_weight': 0.5,  # fNIRS phase less important
            'time_weight': 1.0,   # Time-domain reconstruction more important
            'beta': 0.5,
            'decay': 0.99,
        }
        defaults.update(kwargs)
        
        super().__init__(
            patch_size=patch_size,
            seq_length=seq_length,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            **defaults
        )
