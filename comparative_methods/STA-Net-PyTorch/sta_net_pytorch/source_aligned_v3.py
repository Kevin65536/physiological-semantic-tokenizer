"""STA-Net v3: the actually observed TF2.10 fused/unfused BN conventions.

Five-dimensional EEG branches use fused BN's unbiased moving variance; the
six-dimensional fNIRS branch uses population moving variance. V2 is retained.
The optional, separately reported calibration below fits inference statistics
on training inputs with frozen parameters; it is not an upstream training step.
"""
from __future__ import annotations

import torch

from .source_aligned_v2 import SourceAlignedSTANet, SourceBatchNorm3d


class SourceBatchNorm3dV3(SourceBatchNorm3d):
    def __init__(self, channels, *, fused_variance):
        super().__init__(channels)
        self.fused_variance = bool(fused_variance)

    def forward(self, value):
        if not self.training:
            return super().forward(value)
        x = value.float()
        variance, mean = torch.var_mean(x, dim=(0, 2, 3, 4), unbiased=False)
        count = x.numel() // x.shape[1]
        if count <= 1:
            raise ValueError("source BN requires more than one value per feature")
        update_variance = variance * (count / (count - 1)) if self.fused_variance else variance
        with torch.no_grad():
            self.running_mean.lerp_(mean.detach(), self.momentum)
            self.running_var.lerp_(update_variance.detach(), self.momentum)
            self.num_batches_tracked.add_(1)
        shape = (1, -1, 1, 1, 1)
        normalized = (x - mean.reshape(shape)) * torch.rsqrt(variance.reshape(shape) + self.eps)
        return (normalized * self.weight.reshape(shape) + self.bias.reshape(shape)).to(value.dtype)


class SourceAlignedSTANetV3(SourceAlignedSTANet):
    def __init__(self, config):
        super().__init__(config)
        for block in (self.block1, self.block2):
            for name in ("eeg", "eeg_fusion", "fnirs"):
                branch = getattr(block, name)
                branch.norm = SourceBatchNorm3dV3(branch.norm.num_features, fused_variance=name != "fnirs")


def convert_retained_parameters(retained):
    """Express a frozen v1 classifier in source order without fitting any data.

    Full-row-rank key projection lets W p_pre = p_post represent the same lag
    offset. Flatten permutations and BN gamma rescaling preserve inference.
    This is a functional conversion, not an equivalent optimizer trajectory.
    """
    if retained.config.task_type != "classification":
        raise ValueError("functional conversion requires an actual classification head")
    source = SourceAlignedSTANetV3(retained.config).to(next(retained.parameters()).device)
    state = {key:value.detach().clone() for key,value in retained.state_dict().items()}
    device = next(retained.parameters()).device
    order = torch.arange(512,device=device).reshape(32,4,4,1).permute(1,2,3,0).flatten()
    for key in ("egta.fusion_projection.weight","egta.query_projection.weight","egta.key_projection.weight","eeg_hidden.0.weight"):
        state[key] = state[key][:,order]
    key = state["egta.key_projection.weight"].double().cpu()
    offset = state["egta.position"].squeeze(0).double().cpu()
    solution = torch.linalg.lstsq(key,offset.T,driver="gelsd")
    if int(solution.rank) != key.shape[0]:
        raise ValueError("retained key projection does not have full row rank")
    pre = solution.solution
    residual = float((key @ pre - offset.T).abs().max())
    if residual > 1e-9:
        raise ValueError("retained positional offset cannot be represented exactly")
    state["egta.position"] = pre.T.unsqueeze(0).to(state["egta.key_projection.weight"])
    for block_index in (1,2):
        for name in ("eeg","eeg_fusion","fnirs"):
            prefix=f"block{block_index}.{name}.norm"
            old=getattr(getattr(retained,f"block{block_index}"),name).norm
            new=getattr(getattr(source,f"block{block_index}"),name).norm
            variance=state[prefix+".running_var"]
            state[prefix+".weight"] *= torch.sqrt((variance+new.eps)/(variance+old.eps))
    source.load_state_dict(state)
    source.requires_grad_(False).eval()
    return source,dict(key_projection_rank=int(solution.rank),positional_residual_float64=residual,
                      interpretation="same_frozen_inference_function;no_data_or_training_equivalence_claim")


def calibrate_training_batchnorm(model, forward_batches):
    """Two layerwise passes over caller-supplied training batches only.

    Dropout stays off. In each pass, collect count-weighted activation moments
    before the three normalization layers of one block; update that block only
    after the whole training population. The next block sees those fixed stats.
    No labels, validation samples, gradients, affine weights or thresholds enter.
    """
    model.eval()
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("calibration requires explicitly frozen parameters")
    records = {}
    for block_index in (1, 2):
        block = getattr(model.backbone, f"block{block_index}")
        layers = {f"block{block_index}.{name}":getattr(block, name).norm for name in ("eeg", "eeg_fusion", "fnirs")}
        moments = {}
        def capture(name, values):
            x = values[0].detach().float()
            variance, mean = torch.var_mean(x, dim=(0,2,3,4), unbiased=False)
            if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
                raise ValueError("nonfinite training activation during BN calibration")
            n = x.numel() // x.shape[1]
            mean, variance = mean.double().cpu(), variance.double().cpu()
            if name not in moments:
                moments[name] = dict(count=n, mean=mean, m2=variance*n, batches=1)
            else:
                old = moments[name]; total = old["count"] + n; delta = mean - old["mean"]
                old["m2"] += variance*n + delta.square() * (old["count"]*n/total)
                old["mean"] += delta * (n/total); old["count"] = total; old["batches"] += 1
        hooks = [layer.register_forward_pre_hook(lambda module, values, name=name:capture(name, values)) for name,layer in layers.items()]
        try:
            with torch.inference_mode():
                for batch in forward_batches():
                    model(**batch)
        finally:
            for hook in hooks: hook.remove()
        if set(moments) != set(layers):
            raise ValueError("training pass missed a declared normalization layer")
        with torch.no_grad():
            for name, layer in layers.items():
                state = moments[name]
                if state["count"] <= 1: raise ValueError("empty training normalization support")
                fused = getattr(layer, "fused_variance", True)
                variance = state["m2"] / (state["count"] - int(fused))
                layer.running_mean.copy_(state["mean"].to(layer.running_mean))
                layer.running_var.copy_(variance.to(layer.running_var))
                layer.num_batches_tracked.fill_(state["batches"])
                records[name] = dict(count=state["count"], batches=state["batches"],
                    variance_convention="unbiased" if fused else "population", mean=state["mean"].tolist(), variance=variance.tolist())
    return dict(schema="sta_training_inference_statistics_v1", layers=records,
                dropout="disabled",passes=2,parameter_updates=0,selection_or_evaluation_inputs="not_consumed")
