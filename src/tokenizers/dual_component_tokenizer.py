"""Independent SSM-component and full-residual vocabularies.

The retained local models do not mix leading patch axes. The explicitly
versioned GlobalLatentComponentTokenizer mixes the declared time-patch axis
within one location and keeps each branch's integer decoder independent.
Teacher components are training targets, never inputs to a student forward.
The component-compression reference has a separate explicit input method.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from .ema_vector_quantizer import EMAVectorQuantizer


VARIANTS = (
    "continuous_student",
    "hard_student",
    "component_compression",
    "smooth_hard",
    "ordinary_hard",
)
COMPONENTS = ("S", "O")


def _mask_for(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Accept an element mask or a patch mask, without implicit axis broadcast."""
    if mask is None:
        return torch.ones_like(values, dtype=torch.bool)
    if mask.dtype != torch.bool:
        raise ValueError("masks must have boolean dtype")
    if tuple(mask.shape) == tuple(values.shape[:-1]):
        mask = mask.unsqueeze(-1).expand_as(values)
    elif tuple(mask.shape) != tuple(values.shape):
        raise ValueError("mask must match all elements or the leading patch dimensions")
    return mask.to(device=values.device)


class TemporalPatchEncoder(nn.Module):
    """Same-modality temporal features on the declared context, without normalization bypasses."""
    def __init__(self, inputs, channels, width, latent, output_steps=20):
        super().__init__()
        if channels < 1 or inputs % channels or inputs // channels < output_steps:
            raise ValueError('explicit channels and temporal context required')
        self.inputs, self.channels, self.steps = inputs, channels, inputs // channels
        self.output_steps = output_steps
        self.stem = nn.Conv1d(channels * 2, width, 5, padding=2)
        self.context = nn.Conv1d(width, width, 5, padding=4, dilation=2)
        self.head = nn.Linear(width * (output_steps + 1), latent)

    def forward(self, values_and_masks):
        shape = values_and_masks.shape[:-1]
        values, masks = values_and_masks.split(self.inputs, dim=-1)
        joined = torch.cat((values.reshape(-1, self.steps, self.channels),
            masks.reshape(-1, self.steps, self.channels)), dim=-1).transpose(1,2)
        hidden = torch.nn.functional.gelu(self.stem(joined))
        hidden = hidden + torch.nn.functional.gelu(self.context(hidden))
        start = (self.steps - self.output_steps) // 2
        features = torch.cat((hidden[:, :, start:start + self.output_steps].flatten(1), hidden.mean(-1)), -1)
        return self.head(features).reshape(*shape, -1)


class LocalComponentBranch(nn.Module):
    """One component's patch encoder, EMA vocabulary, and private decoder."""

    def __init__(self, input_dim: int, target_dim: int, cfg: Mapping, hard: bool):
        super().__init__()
        width = int(cfg.get("width", 128))
        latent = int(cfg.get("latent_dim", 32))
        if width <= 0 or latent <= 0:
            raise ValueError("width and latent_dim must be positive")
        self.input_dim, self.target_dim = input_dim, target_dim
        self.hard = hard
        if cfg.get('encoder_kind', 'mlp') == 'temporal':
            self.encoder = TemporalPatchEncoder(input_dim, int(cfg['input_channels']), width, latent)
        elif cfg.get('encoder_kind', 'mlp') == 'mlp':
            self.encoder = nn.Sequential(nn.Linear(input_dim * 2, width), nn.GELU(), nn.Linear(width, latent))
        else:
            raise ValueError('unknown local encoder kind')
        self.decoder = nn.Sequential(
            nn.Linear(latent, width), nn.GELU(), nn.Linear(width, target_dim)
        )
        self.quantizer = EMAVectorQuantizer(
            codebook_size=int(cfg.get("codebook_size", 128)),
            embedding_dim=latent,
            decay=float(cfg.get("ema_decay", 0.99)),
            commitment_cost=float(cfg.get("commitment_cost", 0.25)),
            kmeans_init=bool(cfg.get("kmeans_init", True)),
            kmeans_iters=int(cfg.get("kmeans_iters", 10)),
            revive_dead_codes=bool(cfg.get("revive_dead_codes", False)),
            dead_code_threshold=float(cfg.get("dead_code_threshold", 0.1)),
            revival_warmup_steps=int(cfg.get("revival_warmup_steps", 100)),
            revival_interval=int(cfg.get("revival_interval", 100)),
            revival_stop_after_steps=cfg.get("revival_stop_after_steps"),
            max_revivals_per_event=int(cfg.get("max_revivals_per_event", 8)),
            revival_strategy=cfg.get("revival_strategy", "top_error"),
            revival_count_prior=cfg.get("revival_count_prior", "threshold"),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        if values.ndim < 2 or values.shape[-1] != self.input_dim:
            raise ValueError(f"expected local patches ending in {self.input_dim} inputs")
        if not values.is_floating_point():
            raise ValueError("patch values must use a floating dtype")
        mask = _mask_for(values, mask)
        clean = torch.where(mask, values, torch.zeros_like(values))
        if not torch.isfinite(clean).all():
            raise ValueError("nonfinite visible measurement")
        valid = mask.any(dim=-1)
        latent = self.encoder(torch.cat((clean, mask.to(values.dtype)), dim=-1))
        commitment = latent.sum() * 0.0
        ids, health = None, {}
        if self.hard and bool(valid.any()):
            if not self.training and not bool(self.quantizer.initialized.item()):
                raise RuntimeError("EMA vocabulary is uninitialized; train before evaluation")
            quantized = self.quantizer(latent, valid)
            decoded_latent = quantized.quantized
            ids = quantized.hard_ids.masked_fill(~valid, -1)
            commitment, health = quantized.commitment_loss, quantized.health
        elif self.hard:
            # An empty support must neither initialize nor age an EMA vocabulary.
            decoded_latent = latent * 0.0
            ids = torch.full(valid.shape, -1, dtype=torch.long, device=values.device)
        else:
            decoded_latent = latent
        waveform = self.decoder(decoded_latent)
        return dict(
            waveform=waveform,
            latent=latent,
            quantized=decoded_latent,
            ids=ids,
            valid=valid,
            commitment=commitment,
            health=health,
        )


class DualComponentTokenizer(nn.Module):
    """Two local branches per modality, with independent hard IDs and decoders.

    ``cfg`` declares ``input_dims`` and ``target_dims`` as modality-to-integer
    mappings. For example, 2-second native EEG inputs can have 400*6 inputs
    while the aligned 10 Hz power-proxy target has 20 outputs. Hb inputs and
    targets can each have 20*2 outputs. The model never subtracts native EEG
    voltage from a power proxy.

    The main objective is S error + lambda_O * O error + beta * commitment.
    Each error is a masked, training-scale-normalized element MSE per modality,
    then an equal mean over modalities with valid targets. ``sum`` is diagnostic
    except for ``ordinary_hard``, whose two code streams are supervised only by
    their additive observation reconstruction and have no component semantics.
    """

    def __init__(self, cfg: Mapping, variant: str = "hard_student"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.variant = variant
        self.input_dims = self._dimensions(cfg.get("input_dims"), "input_dims")
        self.target_dims = self._dimensions(cfg.get("target_dims"), "target_dims")
        if self.input_dims.keys() != self.target_dims.keys():
            raise ValueError("input_dims and target_dims must declare the same modalities")
        self.modalities = tuple(self.input_dims)
        self.hard = variant != "continuous_student"
        self.branches = nn.ModuleDict()
        for modality in self.modalities:
            inputs = (self.target_dims[modality] if variant == "component_compression"
                      else self.input_dims[modality])
            branch_cfg = dict(cfg)
            if cfg.get('input_channels'):
                branch_cfg['input_channels'] = cfg['input_channels'][modality]
            self.branches[modality] = nn.ModuleDict({
                component: LocalComponentBranch(inputs, self.target_dims[modality], branch_cfg, self.hard)
                for component in COMPONENTS
            })

    @staticmethod
    def _dimensions(values, field: str) -> dict[str, int]:
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"{field} must be a nonempty modality-to-dimension mapping")
        if any(not isinstance(k, str) or not k or isinstance(v, bool)
               or not isinstance(v, int) or v <= 0 for k, v in values.items()):
            raise ValueError(f"{field} requires named positive integer dimensions")
        return dict(values)

    def _check_modalities(self, values: Mapping, field: str) -> None:
        if not isinstance(values, Mapping) or set(values) != set(self.modalities):
            raise ValueError(f"{field} must contain exactly {self.modalities}")

    def _forward(self, sources: Mapping, masks: Mapping | None) -> dict:
        self._check_modalities(sources, "inputs")
        if masks is not None:
            self._check_modalities(masks, "input_masks")
        result = {}
        for modality in self.modalities:
            source = sources[modality]
            component_results = {}
            for component in COMPONENTS:
                values = source[component]
                mask = None if masks is None else masks[modality][component]
                component_results[component] = self.branches[modality][component](values, mask)
            source_result, residual_result = (component_results[c] for c in COMPONENTS)
            if source_result["waveform"].shape != residual_result["waveform"].shape:
                raise ValueError("S and O must have identical patch support")
            result[modality] = dict(
                S=source_result["waveform"], O=residual_result["waveform"],
                sum=source_result["waveform"] + residual_result["waveform"],
                ids_S=source_result["ids"], ids_O=residual_result["ids"],
                latent_S=source_result["latent"], latent_O=residual_result["latent"],
                quantized_S=source_result["quantized"], quantized_O=residual_result["quantized"],
                valid_S=source_result["valid"], valid_O=residual_result["valid"],
                commitment=source_result["commitment"] + residual_result["commitment"],
                health={c: component_results[c]["health"] for c in COMPONENTS},
            )
        return result

    def forward(self, inputs: Mapping, input_masks: Mapping | None = None) -> dict:
        """Encode deployable observations only; there is no teacher argument."""
        if self.variant == "component_compression":
            raise ValueError("component_compression requires forward_components")
        self._check_modalities(inputs, "inputs")
        if input_masks is not None:
            self._check_modalities(input_masks, "input_masks")
        return self._forward(
            {m: {c: inputs[m] for c in COMPONENTS} for m in self.modalities},
            None if input_masks is None else {
                m: {c: input_masks[m] for c in COMPONENTS} for m in self.modalities
            },
        )

    def forward_components(self, physical: Mapping, remaining: Mapping,
                           target_masks: Mapping | None = None) -> dict:
        """Compression reference: S reads physical; O reads fixed full residual."""
        if self.variant != "component_compression":
            raise ValueError("teacher components are inputs only in component_compression")
        self._check_modalities(physical, "physical")
        self._check_modalities(remaining, "remaining")
        if target_masks is not None:
            self._check_modalities(target_masks, "target_masks")
        sources, masks = {}, {}
        for modality in self.modalities:
            if physical[modality].shape != remaining[modality].shape:
                raise ValueError("physical and remaining must have identical support")
            common = (_mask_for(physical[modality], None if target_masks is None
                               else target_masks[modality])
                      & torch.isfinite(physical[modality]) & torch.isfinite(remaining[modality]))
            sources[modality] = {"S": physical[modality], "O": remaining[modality]}
            masks[modality] = {c: common for c in COMPONENTS}
        return self._forward(sources, masks)

    def objective(self, output: Mapping, observed: Mapping, physical: Mapping | None = None,
                  target_masks: Mapping | None = None, target_scales: Mapping | None = None,
                  lambda_O: float = 1.0, beta: float = 1.0) -> dict:
        """Score aligned target-space observations; O is always observed-physical.

        Scales must be fixed from training data by the caller. A scalar, one
        value per coordinate, or an explicitly broadcastable per-record scale
        is accepted; scales are never fitted here.
        Nonfinite labels and unsupported predictions are excluded before error
        arithmetic. Targets are detached so O cannot track student S errors.
        """
        self._check_modalities(output, "output")
        self._check_modalities(observed, "observed")
        ordinary = self.variant == "ordinary_hard"
        if not ordinary:
            self._check_modalities(physical, "physical")
        if target_masks is not None:
            self._check_modalities(target_masks, "target_masks")
        if target_scales is not None:
            self._check_modalities(target_scales, "target_scales")
        if not math.isfinite(lambda_O) or not math.isfinite(beta) or min(lambda_O, beta) < 0:
            raise ValueError("objective weights must be finite and nonnegative")
        per_modality, active = {}, []
        reference = output[self.modalities[0]]["sum"]
        zero = reference.reshape(-1)[:0].sum()
        for modality in self.modalities:
            prediction, target = output[modality], observed[modality].detach()
            if target.shape != prediction["sum"].shape or target.shape[-1] != self.target_dims[modality]:
                raise ValueError("observed targets must match decoded target-space patches")
            mask = _mask_for(target, None if target_masks is None else target_masks[modality])
            mask = (mask & torch.isfinite(target)
                    & prediction["valid_S"].unsqueeze(-1)
                    & prediction["valid_O"].unsqueeze(-1))
            physical_target = None
            if not ordinary:
                physical_target = physical[modality].detach()
                if physical_target.shape != target.shape:
                    raise ValueError("physical must match observations in units and tensor support")
                mask = mask & torch.isfinite(physical_target)
            scale = torch.as_tensor(1.0 if target_scales is None else target_scales[modality],
                                    device=target.device, dtype=target.dtype).detach()
            try:
                scale = torch.broadcast_to(scale, target.shape)
            except RuntimeError as exc:
                raise ValueError("training scale must broadcast to one value per target element") from exc
            if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
                raise ValueError("training scale must be positive and finite")
            clean_target = torch.where(mask, target, torch.zeros_like(target))

            def mse(estimate, truth):
                clean_estimate = torch.where(mask, estimate, torch.zeros_like(estimate))
                if not bool(torch.isfinite(clean_estimate).all()):
                    raise ValueError("nonfinite prediction on valid support")
                error = ((clean_estimate - truth) / scale).square()
                return error.sum() / mask.sum().clamp_min(1)

            terms = {"sum": mse(prediction["sum"], clean_target),
                     "vq": prediction["commitment"], "valid_elements": mask.sum()}
            if ordinary:
                terms.update(S=zero, O=zero)
            else:
                clean_physical = torch.where(mask, physical_target, torch.zeros_like(physical_target))
                terms.update(S=mse(prediction["S"], clean_physical),
                             O=mse(prediction["O"], clean_target - clean_physical))
            per_modality[modality] = terms
            if bool(mask.any()):
                active.append(terms)
        terms = {name: (torch.stack([m[name] for m in active]).mean() if active else zero)
                 for name in ("S", "O", "sum", "vq")}
        terms["total"] = ((terms["sum"] if ordinary else terms["S"] + lambda_O * terms["O"])
                          + beta * terms["vq"])
        terms["per_modality"] = per_modality
        return terms

    def _require_hard(self) -> None:
        if not self.hard:
            raise ValueError("continuous_student has no deployed hard vocabulary")
        for branches in self.branches.values():
            for branch in branches.values():
                if not bool(branch.quantizer.initialized.item()):
                    raise RuntimeError("EMA vocabulary is uninitialized")

    def decode_ids(self, ids: Mapping) -> dict:
        """Integer-only consumer: reject sidecars, invalid IDs, and soft values."""
        self._require_hard()
        self._check_modalities(ids, "ids")
        result = {}
        for modality in self.modalities:
            if not isinstance(ids[modality], Mapping) or set(ids[modality]) != set(COMPONENTS):
                raise ValueError("each modality must contain only S and O integer IDs")
            decoded, shape = {}, None
            for component in COMPONENTS:
                value = ids[modality][component]
                if not isinstance(value, torch.Tensor) or value.dtype not in (
                    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
                ):
                    raise ValueError("integer tensor IDs required")
                branch = self.branches[modality][component]
                if bool((value < 0).any()) or bool((value >= branch.quantizer.codebook_size).any()):
                    raise ValueError("IDs must lie inside the fixed vocabulary")
                if shape is not None and shape != value.shape:
                    raise ValueError("S and O ID support must match")
                shape = value.shape
                indices = value.to(device=branch.quantizer.codebook.device, dtype=torch.long)
                decoded[component] = branch.decoder(branch.quantizer.get_embedding(indices))
            result[modality] = {**decoded, "sum": decoded["S"] + decoded["O"]}
        return result

    def decoded_codewords(self) -> dict:
        """Each fixed code's decoded waveform, without any sample-specific input."""
        self._require_hard()
        return {m: {c: self.branches[m][c].decoder(self.branches[m][c].quantizer.codebook)
                    for c in COMPONENTS} for m in self.modalities}

    def expected_waveforms(self, probabilities: Mapping) -> dict:
        """Compute p @ D(codewords), never D(p @ codewords)."""
        self._check_modalities(probabilities, "probabilities")
        dictionaries = self.decoded_codewords()
        result = {}
        for modality in self.modalities:
            if (not isinstance(probabilities[modality], Mapping)
                    or set(probabilities[modality]) != set(COMPONENTS)):
                raise ValueError("each modality needs S and O distributions")
            decoded, shape = {}, None
            for component in COMPONENTS:
                p = probabilities[modality][component]
                words = dictionaries[modality][component]
                if not isinstance(p, torch.Tensor) or not p.is_floating_point():
                    raise ValueError("floating probability tensors required")
                if p.ndim < 1 or p.shape[-1] != words.shape[0]:
                    raise ValueError("probability axis must match vocabulary size")
                if (not bool(torch.isfinite(p).all()) or bool((p < 0).any())
                        or not torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-5, rtol=1e-5)):
                    raise ValueError("finite normalized nonnegative probabilities required")
                if shape is not None and shape != p.shape[:-1]:
                    raise ValueError("S and O probability support must match")
                shape = p.shape[:-1]
                decoded[component] = p.to(device=words.device, dtype=words.dtype) @ words
            result[modality] = {**decoded, "sum": decoded["S"] + decoded["O"]}
        return result

    def nominal_bits_per_patch(self) -> dict[str, float]:
        """Total S+O code capacity per modality, with no continuous sidecars."""
        if not self.hard:
            raise ValueError("continuous_student has no deployed hard vocabulary")
        return {m: sum(math.log2(self.branches[m][c].quantizer.codebook_size)
                       for c in COMPONENTS) for m in self.modalities}


def local_patch_context(values: torch.Tensor, radius: int = 0):
    """Concatenate observed neighboring patches, preserving their exact support.

    Out-of-window positions are masked placeholders, never observed padding.
    A radius of one reads six seconds for a two-second output patch. Downstream
    masking must hide every output whose support intersects a hidden input.
    """
    if values.ndim != 3 or radius < 0:
        raise ValueError("[batch,patch,feature] values and nonnegative radius required")
    pieces, masks = [], []
    for offset in range(-radius, radius + 1):
        value = torch.zeros_like(values)
        mask = torch.zeros_like(values, dtype=torch.bool)
        start, stop = max(0, -offset), min(values.shape[1], values.shape[1] - offset)
        if start < stop:
            source = values[:, start + offset:stop + offset]
            valid = torch.isfinite(source)
            value[:, start:stop] = torch.where(valid, source, 0)
            mask[:, start:stop] = valid
        pieces.append(value); masks.append(mask)
    return torch.cat(pieces, -1), torch.cat(masks, -1)


def waveform_windows(values, *, hop, span):
    """Centered fixed-grid windows and explicit out-of-window validity.

    Values are [batch,time,channel]. Window centres are the centres of the
    nonoverlapping hop cells. The integer grid is shared by inputs and outputs.
    """
    if values.ndim != 3 or hop < 1 or span < hop or (span-hop)%2 or values.shape[1]%hop:
        raise ValueError('aligned [batch,time,channel], positive hop and centred span required')
    n=values.shape[1]//hop;left=(span-hop)//2
    indices=torch.arange(n,device=values.device)[:,None]*hop-left+torch.arange(span,device=values.device)[None]
    valid=(indices>=0)&(indices<values.shape[1]);take=indices.clamp(0,values.shape[1]-1)
    result=values[:,take,:]
    mask=valid[None,:,:,None].expand_as(result)&torch.isfinite(result)
    return torch.where(mask,result,0).flatten(-2),mask.flatten(-2)


def overlap_decode(windows, *, hop, channels, length):
    """Decode only supplied component waveforms with a fixed partition of unity.

    Both numpy and torch values use the same sine-squared synthesis weights.
    There is no measured-signal blending and no continuous per-window sidecar.
    """
    import numpy as np
    is_torch=isinstance(windows,torch.Tensor)
    if windows.shape[-1]%channels or length%hop or windows.shape[-2]!=length//hop:
        raise ValueError('complete aligned component windows required')
    span=windows.shape[-1]//channels
    if span<hop or (span-hop)%2:raise ValueError('centred output span required')
    left=(span-hop)//2
    weights=np.ones(span) if span==hop else np.sin(np.pi*(np.arange(span)+.5)/span)**2
    if is_torch:
        result=windows.new_zeros((*windows.shape[:-2],length,channels));denom=windows.new_zeros(length)
        w=torch.as_tensor(weights,device=windows.device,dtype=windows.dtype)
    else:
        result=np.zeros((*windows.shape[:-2],length,channels),dtype=windows.dtype);denom=np.zeros(length);w=weights
    blocks=windows.reshape(*windows.shape[:-1],span,channels)
    for i in range(windows.shape[-2]):
        start=i*hop-left;lo=max(0,start);hi=min(length,start+span);a=lo-start;b=hi-start
        result[...,lo:hi,:]=result[...,lo:hi,:]+blocks[...,i,a:b,:]*w[a:b,None]
        denom[lo:hi]+=w[a:b]
    return result/denom[...,None]


def synthesize_component(windows, *, hop, channels, length, smoothing_sigma=0.):
    """Fixed same-branch codeword synthesis; never accepts measured boundaries.

    Gaussian synthesis has four-sigma finite support. It is linear in decoded
    codewords, preserves constant signals and introduces no new token or sidecar.
    """
    from scipy.ndimage import gaussian_filter1d
    value=overlap_decode(windows,hop=hop,channels=channels,length=length)
    if smoothing_sigma<0:raise ValueError('nonnegative fixed synthesis width required')
    return gaussian_filter1d(value,sigma=smoothing_sigma,axis=-2,mode='nearest',truncate=4.) if smoothing_sigma else value


class WaveformDictionary:
    """Train-only waveform-space residual VQ; decoding is integer lookup/addition.

    Every level is an explicit K-way ID. There are no per-patch continuous
    amplitude/mean sidecars. A second level doubles the nominal rate. This
    quantizes an independently predicted component, not its teacher label at
    deployment. Direct teacher compression must be labelled an oracle reference.
    """
    def __init__(self, size=128, levels=1, channels=1, derivative_weight=0.0,
                 seed=17, max_fit_patches=32768, max_iter=100):
        if size < 2 or levels < 1 or channels < 1 or derivative_weight < 0:
            raise ValueError("invalid waveform vocabulary settings")
        self.size, self.levels, self.channels = int(size), int(levels), int(channels)
        self.derivative_weight, self.seed = float(derivative_weight), int(seed)
        self.max_fit_patches, self.max_iter = int(max_fit_patches), int(max_iter)

    def _features(self, values):
        import numpy as np
        values = np.asarray(values, dtype=np.float32)
        scaled = values / self.value_scale
        if not self.derivative_weight:
            return scaled
        wave = values.reshape(len(values), -1, self.channels)
        delta = np.diff(wave, axis=1).reshape(len(values), -1)
        return np.concatenate((scaled, np.sqrt(self.derivative_weight) * delta / self.delta_scale), axis=-1)

    def fit(self, training_waveforms):
        import numpy as np
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import pairwise_distances_argmin
        x = np.asarray(training_waveforms, dtype=np.float32)
        if x.ndim != 2 or len(x) < self.size or x.shape[-1] % self.channels or not np.isfinite(x).all():
            raise ValueError("finite training-only flattened waveforms required")
        self.dimension = x.shape[-1]
        # Fixed training scalars preserve amplitude; no sample-specific rescaling.
        self.value_scale = max(float(x.std()), 1e-6)
        delta = np.diff(x.reshape(len(x), -1, self.channels), axis=1)
        self.delta_scale = max(float(np.sqrt(np.mean(delta ** 2))), 1e-6)
        if len(x) > self.max_fit_patches:
            selected = np.random.default_rng(self.seed).choice(len(x), self.max_fit_patches, replace=False)
            fit_values = x[selected].copy()
        else:
            fit_values = x.copy()
        self.codewords = []
        residual = fit_values
        for level in range(self.levels):
            features = self._features(residual)
            fitted = MiniBatchKMeans(n_clusters=self.size, n_init=3, random_state=self.seed + level,
                batch_size=2048, max_iter=self.max_iter, reassignment_ratio=.01).fit(features)
            # The feature map is linear: its first D coordinates retain the
            # complete waveform, and weighted differences are derived features.
            words = (fitted.cluster_centers_[:, :self.dimension] * self.value_scale).astype('float32')
            self.codewords.append(words)
            ids = pairwise_distances_argmin(features, self._features(words))
            residual = residual - words[ids]
        self.codewords = np.stack(self.codewords)
        return self

    def encode(self, waveforms, levels=None):
        import numpy as np
        from sklearn.metrics import pairwise_distances_argmin
        x = np.asarray(waveforms, dtype=np.float32)
        count = self.levels if levels is None else int(levels)
        if x.shape[-1] != self.dimension or not 1 <= count <= self.levels or not np.isfinite(x).all():
            raise ValueError("finite same-coordinate waveforms and available levels required")
        residual = x.reshape(-1, self.dimension).copy(); indices = []
        for words in self.codewords[:count]:
            ids = pairwise_distances_argmin(self._features(residual), self._features(words))
            indices.append(ids); residual -= words[ids]
        return np.stack(indices, -1).reshape(x.shape[:-1] + (count,)).astype('int16')

    def decode(self, ids):
        import numpy as np
        ids = np.asarray(ids)
        if ids.dtype.kind not in 'iu' or ids.ndim < 1 or not 1 <= ids.shape[-1] <= self.levels:
            raise ValueError("one integer ID per declared residual-VQ level required")
        if np.any(ids < 0) or np.any(ids >= self.size):
            raise ValueError("ID outside vocabulary")
        return sum(self.codewords[level][ids[..., level]] for level in range(ids.shape[-1]))

    def state(self):
        return dict(codewords=self.codewords, value_scale=self.value_scale, delta_scale=self.delta_scale,
            channels=self.channels, derivative_weight=self.derivative_weight, dimension=self.dimension,
            nominal_bits_per_component_patch=self.levels * math.log2(self.size))

    @classmethod
    def from_state(cls, state):
        import numpy as np
        words=np.asarray(state['codewords'],dtype='float32')
        if words.ndim!=3 or not np.isfinite(words).all():raise ValueError('finite level/code/waveform dictionary required')
        obj=cls(size=words.shape[1],levels=words.shape[0],channels=int(state['channels']),
            derivative_weight=float(state['derivative_weight']))
        obj.codewords=words;obj.dimension=int(state['dimension'])
        obj.value_scale=float(state['value_scale']);obj.delta_scale=float(state['delta_scale'])
        if words.shape[-1]!=obj.dimension or min(obj.value_scale,obj.delta_scale)<=0:
            raise ValueError('inconsistent waveform dictionary state')
        return obj


class BandGainResidualPredictor(nn.Module):
    """Estimate voltage O from observed bands and slow, bounded gains.

    This is an encoder-side continuous estimator. Its output must be quantized
    by a WaveformDictionary for deployment; the hard decoder never reads the
    observed carriers or gains. The caller declares the filtering support.
    """

    def __init__(self, feature_mean, feature_scale, *, channels=16, width=64,
                 context_bins=19, joint=False, max_log_gain=.7, hop=20):
        super().__init__()
        mean = torch.as_tensor(feature_mean, dtype=torch.float32)
        scale = torch.as_tensor(feature_scale, dtype=torch.float32)
        if (mean.ndim != 2 or mean.shape[-1] != 5 or scale.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(scale).all()
                or not (scale > 0).all() or context_bins < 1 or context_bins % 2 != 1
                or channels < 1 or width < 1 or hop < 1 or max_log_gain <= 0):
            raise ValueError('finite training band scales and odd gain context required')
        self.register_buffer('feature_mean', mean)
        self.register_buffer('feature_scale', scale)
        self.channels, self.hop = int(channels), int(hop)
        self.joint, self.context_bins = bool(joint), int(context_bins)
        self.max_log_gain = float(max_log_gain)
        self.channel_embedding = nn.Embedding(len(mean), 8)
        count = channels if joint else 1
        self.network = nn.Sequential(
            nn.Conv1d(13 * count + 1, width, 1), nn.GELU(),
            nn.Conv1d(width, width, context_bins, padding=context_bins // 2, groups=width),
            nn.GELU(), nn.Conv1d(width, width, 1), nn.GELU(),
            nn.Conv1d(width, 5 * count, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, carriers, log_power, coordinate_ids):
        if (carriers.ndim != 4 or log_power.ndim != 4 or carriers.shape[-1] != 5
                or carriers.shape[:2] != log_power.shape[:2]
                or carriers.shape[1] != self.channels or log_power.shape[-1] != 5
                or carriers.shape[2] != log_power.shape[2] * self.hop
                or coordinate_ids.shape != carriers.shape[:2]
                or coordinate_ids.dtype != torch.long):
            raise ValueError('aligned [batch,electrode,time,band] and integer coordinates required')
        if (not torch.isfinite(carriers).all() or not torch.isfinite(log_power).all()
                or (coordinate_ids < 0).any() or (coordinate_ids >= len(self.feature_mean)).any()):
            raise ValueError('finite observed-only features and registered coordinates required')
        batch, channels, steps, _ = log_power.shape
        standardized = ((log_power - self.feature_mean[coordinate_ids, None, :])
                        / self.feature_scale[coordinate_ids, None, :])
        metadata = self.channel_embedding(coordinate_ids)[:, :, None, :].expand(-1, -1, steps, -1)
        value = torch.cat((standardized, metadata), -1)
        if self.joint:
            value = value.permute(0, 1, 3, 2).reshape(batch, channels * 13, steps)
        else:
            value = value.reshape(batch * channels, steps, 13).transpose(1, 2)
        # An explicit support channel distinguishes padded context from a
        # genuine feature vector at the training mean.
        value = torch.cat((value, value.new_ones((len(value), 1, steps))), 1)
        nodes = self.max_log_gain * torch.tanh(self.network(value))
        nodes = nodes.reshape(batch * channels, 5, steps)
        native = torch.nn.functional.interpolate(nodes, size=carriers.shape[2],
                                                  mode='linear', align_corners=False)
        native = native.reshape(batch, channels, 5, -1).transpose(-1, -2)
        residual = ((1 - native.exp()) * carriers).sum(-1)
        return dict(O=residual, log_gain=nodes.reshape(batch, channels, 5, steps).transpose(-1, -2))


class ProjectionGainResidualPredictor(nn.Module):
    """EEG-reference-aware gain estimation in the frozen three-ROI subspace.

    The observed proxy is an EEG measurement transform, not a teacher output.
    The fixed projection inverse constrains gains to the target constructor's
    shared electrode directions. The hard O consumer remains waveform lookup.
    """
    def __init__(self, proxy_mean, proxy_scale, gain_map, *, channels=16,
                 width=64, context_bins=19, max_log_gain=.7, hop=20):
        super().__init__()
        mean=torch.as_tensor(proxy_mean,dtype=torch.float32)
        scale=torch.as_tensor(proxy_scale,dtype=torch.float32)
        mapping=torch.as_tensor(gain_map,dtype=torch.float32)
        if (mean.ndim!=2 or mean.shape[-1]!=3 or scale.shape!=mean.shape
                or mapping.shape!=(len(mean),channels*5,3) or not torch.isfinite(mean).all()
                or not torch.isfinite(scale).all() or not torch.isfinite(mapping).all()
                or not (scale>0).all() or context_bins<1 or context_bins%2!=1 or hop<1):
            raise ValueError('training EEG proxy scales, fixed gain maps and odd context required')
        self.register_buffer('proxy_mean',mean);self.register_buffer('proxy_scale',scale)
        self.register_buffer('gain_map',mapping)
        kernel=torch.exp(-.5*torch.arange(-4,5,dtype=torch.float32)**2);kernel/=kernel.sum()
        self.register_buffer('gain_kernel',kernel.reshape(1,1,-1))
        self.channels,self.hop,self.max_log_gain=channels,hop,float(max_log_gain)
        self.dataset_embedding=nn.Embedding(len(mean),4)
        self.network=nn.Sequential(nn.Conv1d(9,width,1),nn.GELU(),
            nn.Conv1d(width,width,context_bins,padding=context_bins//2,groups=width),nn.GELU(),
            nn.Conv1d(width,width,1),nn.GELU(),nn.Conv1d(width,3,1))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self, carriers, observed_proxy, dataset_ids):
        if (carriers.ndim!=4 or carriers.shape[1]!=self.channels or carriers.shape[-1]!=5
                or observed_proxy.shape!=(len(carriers),carriers.shape[2]//self.hop,3)
                or carriers.shape[2]%self.hop or dataset_ids.shape!=(len(carriers),)
                or dataset_ids.dtype!=torch.long or (dataset_ids<0).any()
                or (dataset_ids>=len(self.proxy_mean)).any()
                or not torch.isfinite(carriers).all() or not torch.isfinite(observed_proxy).all()):
            raise ValueError('aligned observed EEG carriers, proxy and dataset IDs required')
        batch,steps,_=observed_proxy.shape
        normalized=(observed_proxy-self.proxy_mean[dataset_ids,None])/self.proxy_scale[dataset_ids,None]
        metadata=self.dataset_embedding(dataset_ids)[:,None].expand(-1,steps,-1)
        time=torch.linspace(0,1,steps,device=carriers.device,dtype=carriers.dtype)[None,:,None].expand(batch,-1,-1)
        features=torch.cat((normalized,metadata,time,torch.ones_like(time)),-1).transpose(1,2)
        directions=self.network(features)
        nodes=.5*torch.einsum('bkt,bdk->bdt',directions,self.gain_map[dataset_ids])
        smooth=torch.nn.functional.conv1d(torch.nn.functional.pad(nodes.reshape(-1,1,steps),(4,4),mode='replicate'),self.gain_kernel)
        nodes=smooth.reshape(batch,self.channels*5,steps).clamp(-self.max_log_gain,self.max_log_gain)
        native=torch.nn.functional.interpolate(nodes,size=carriers.shape[2],mode='linear',align_corners=False)
        native=native.reshape(batch,self.channels,5,-1).transpose(-1,-2)
        return dict(O=((1-native.exp())*carriers).sum(-1),
                    log_gain=nodes.reshape(batch,self.channels,5,steps).transpose(-1,-2))


class WaveformPatchInput(nn.Module):
    """Sampled-waveform encoder with exactly fifteen temporal output anchors."""
    def __init__(self, channels, strides, width, local_width, patches):
        super().__init__()
        self.channels, self.patches = channels, patches
        layers=[]; incoming=channels
        for stride in strides:
            layers += [nn.Conv1d(incoming, local_width, 2*stride+1, stride=stride, padding=stride), nn.GELU()]
            incoming=local_width
        self.features=nn.Sequential(*layers)
        self.project=nn.Linear(local_width*4, width)

    def forward(self, x):
        h=self.features(x.reshape(x.shape[0],-1,self.channels).transpose(1,2))
        if h.shape[-1] != self.patches*4:raise ValueError('four internal samples per declared anchor required')
        return self.project(h.transpose(1,2).reshape(x.shape[0],self.patches,-1))


class WaveformPatchOutput(nn.Module):
    """Synthesize the full waveform from same-branch decoded anchor features."""
    def __init__(self, channels, strides, width, local_width, patches, input_dim):
        super().__init__()
        self.patches, self.input_dim, self.local_width=patches,input_dim,local_width
        self.project=nn.Linear(width,local_width*4)
        layers=[]
        for stride in reversed(strides):
            layers += [nn.ConvTranspose1d(local_width,local_width,2*stride,stride=stride,
                         padding=(stride+1)//2,output_padding=stride%2),nn.GELU(),
                       nn.Conv1d(local_width,local_width,5,padding=2),nn.GELU()]
        layers += [nn.Conv1d(local_width,channels,7,padding=3)]
        self.synthesis=nn.Sequential(*layers)

    def forward(self, x):
        h=self.project(x).reshape(x.shape[0],self.patches*4,self.local_width).transpose(1,2)
        return self.synthesis(h).transpose(1,2).reshape(x.shape[0],self.patches,self.input_dim)


class GlobalLatentComponentTokenizer(nn.Module):
    """One latent ID per patch/branch; global time context, no continuous bypass.

    Input is [batch, patches, samples * channels] for one optical/electrode
    location. The two color channels remain jointly represented inside every
    Hb patch. No branch can read another branch's IDs, or another modality.
    Decoder context is the complete same-branch sequence, so an ID does not
    have a context-independent waveform interpretation.
    """

    def __init__(self, cfg: Mapping, *, codebook_size: int = 128, hard: bool = True):
        super().__init__()
        self.input_dim = int(cfg['input_dim'])
        self.patches = int(cfg['patches'])
        self.hard = bool(hard)
        width, latent, heads = (int(cfg[k]) for k in ('width', 'latent_dim', 'heads'))
        if (self.input_dim < 1 or self.patches < 1 or width < 2 or width % 2
                or latent < 1 or heads < 1 or width % heads or codebook_size < 2):
            raise ValueError('positive patch/latent dimensions and compatible attention heads required')
        position = torch.arange(self.patches, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.) / width))
        encoding = torch.zeros(1, self.patches, width)
        encoding[0, :, 0::2], encoding[0, :, 1::2] = torch.sin(position * frequency), torch.cos(position * frequency)
        self.register_buffer('position', encoding)
        self.branches = nn.ModuleDict()
        for component in COMPONENTS:
            def blocks(count):
                layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=width * 2,
                    dropout=float(cfg.get('dropout', 0.)), activation='gelu', batch_first=True, norm_first=True)
                return nn.TransformerEncoder(layer, int(count), norm=nn.LayerNorm(width), enable_nested_tensor=False)
            self.branches[component] = nn.ModuleDict(dict(
                input=nn.Sequential(nn.Linear(self.input_dim, width), nn.GELU()),
                encoder=blocks(cfg['encoder_layers']), to_latent=nn.Linear(width, latent),
                from_latent=nn.Linear(latent, width), decoder=blocks(cfg['decoder_layers']),
                output=nn.Linear(width, self.input_dim),
                quantizer=EMAVectorQuantizer(codebook_size=codebook_size, embedding_dim=latent,
                    decay=float(cfg.get('ema_decay', .99)), commitment_cost=float(cfg.get('commitment_cost', .25)),
                    kmeans_init=True, kmeans_iters=int(cfg.get('kmeans_iters', 20)),
                    revive_dead_codes=True, dead_code_threshold=float(cfg.get('dead_code_threshold', .1)),
                    revival_warmup_steps=int(cfg.get('revival_warmup_steps', 100)),
                    revival_interval=int(cfg.get('revival_interval', 50)),
                    revival_stop_after_steps=int(cfg.get('revival_stop_after_steps', 1500)),
                    max_revivals_per_event=int(cfg.get('max_revivals_per_event', 8)),
                    revival_strategy='diverse_farthest', revival_count_prior='uniform_batch')))
        # The retained dense default keeps identical names and behavior.
        self.waveform_io = cfg.get('waveform_io', False)
        if self.waveform_io:
            channels=int(cfg['channels']); strides=tuple(cfg['sample_strides'])
            if self.input_dim != channels*4*math.prod(strides):
                raise ValueError('waveform strides must preserve the declared sample grid')
            for branch in self.branches.values():
                branch['input']=WaveformPatchInput(channels,strides,width,int(cfg['local_width']),self.patches)
                branch['output']=WaveformPatchOutput(channels,strides,width,int(cfg['local_width']),self.patches,self.input_dim)
        self.input_color_weights = cfg.get('input_color_weights')
        if self.input_color_weights is not None:
            if int(cfg['channels']) != 2 or list(self.input_color_weights) != [1., 0.]:
                raise ValueError('Hb input ablation must retain HbO and zero HbR')
        self.code_shape_head = None
        if cfg.get('code_shape_head', False) and self.hard:
            if self.input_dim != 40 or int(cfg['channels']) != 2:
                raise ValueError('single-ID shape head requires 20 paired Hb samples')
            # The additional head must not perturb the matched codebook RNG.
            with torch.random.fork_rng(devices=[]):
                self.code_shape_head = nn.Linear(latent, self.input_dim)

    def encode_continuous(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        if (values.ndim != 3 or tuple(values.shape[1:]) != (self.patches, self.input_dim)
                or not values.is_floating_point() or not bool(torch.isfinite(values).all())):
            raise ValueError('complete finite [batch, declared patches, patch features] observations required')
        if self.input_color_weights is not None:
            weights = values.new_tensor(self.input_color_weights).repeat(self.input_dim // 2)
            values = values * weights
        return {c: b['to_latent'](b['encoder'](b['input'](values) + self.position))
                for c, b in self.branches.items()}

    def _decode_component(self, component: str, latent: torch.Tensor) -> torch.Tensor:
        branch = self.branches[component]
        return branch['output'](branch['decoder'](branch['from_latent'](latent) + self.position))

    def forward(self, values: torch.Tensor, *, quantize: bool | None = None) -> dict:
        use_hard = self.hard if quantize is None else bool(quantize)
        latents = self.encode_continuous(values)
        result, ids, health = {}, {}, {}
        commitment = values.sum() * 0.
        for component, latent in latents.items():
            if use_hard:
                quantizer = self.branches[component]['quantizer']
                if not self.training and not bool(quantizer.initialized):
                    raise RuntimeError('training-only codebook initialization required')
                output = quantizer(latent)
                # The waveform consumer evaluates the actual dictionary entry.
                # STE subtraction/addition is only needed while training; its
                # float32 cancellation can be amplified by the synthesis stack.
                value = (quantizer.get_embedding(output.hard_ids)
                         if self.waveform_io and not self.training else output.quantized)
                ids[component], health[component] = output.hard_ids, output.health
                commitment = commitment + output.commitment_loss
            else:
                value = latent
            result[component] = self._decode_component(component, value)
            if component == 'S' and self.code_shape_head is not None and use_hard:
                shape = self.code_shape_head(value).reshape(*value.shape[:2], 20, 2)
                result['S_shape'] = (shape - shape.mean(dim=-2, keepdim=True)).flatten(-2)
        return dict(result, sum=result['S'] + result['O'], ids=ids, latent=latents,
                    health=health, commitment=commitment / len(COMPONENTS))

    @torch.no_grad()
    def initialize_codebooks(self, training_values: torch.Tensor) -> None:
        """Caller supplies training observations only; no targets or validation."""
        mode = self.training
        self.eval()
        latents = self.encode_continuous(training_values)
        for component, latent in latents.items():
            q = self.branches[component]['quantizer']
            if bool(q.initialized):
                raise RuntimeError('codebook already initialized; do not refit a retained vocabulary')
            if latent.numel() // latent.shape[-1] < q.codebook_size:
                raise ValueError('at least K training patches required for vocabulary initialization')
            q.train()
            q(latent)
        self.train(mode)

    def decode_ids(self, ids: Mapping[str, torch.Tensor]) -> dict:
        if not isinstance(ids, Mapping) or set(ids) != set(COMPONENTS):
            raise ValueError('only S/O integer sequences are accepted; no sidecars')
        result, shape = {}, None
        for component in COMPONENTS:
            value = ids[component]
            q = self.branches[component]['quantizer']
            if (not isinstance(value, torch.Tensor) or value.dtype not in
                    (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                    or value.ndim != 2 or value.shape[1] != self.patches):
                raise ValueError('one integer ID per declared time patch required')
            if shape is not None and value.shape != shape:
                raise ValueError('S/O sequence support must match')
            if not bool(q.initialized) or bool((value < 0).any()) or bool((value >= q.codebook_size).any()):
                raise ValueError('initialized vocabulary and in-range IDs required')
            shape = value.shape
            result[component] = self._decode_component(component,
                q.get_embedding(value.to(device=q.codebook.device, dtype=torch.long)))
        return dict(result, sum=result['S'] + result['O'])


__all__ = ["DualComponentTokenizer", "GlobalLatentComponentTokenizer", "LocalComponentBranch", "TemporalPatchEncoder", "WaveformDictionary",
           "ProjectionGainResidualPredictor",
           "BandGainResidualPredictor", "local_patch_context", "VARIANTS"]
