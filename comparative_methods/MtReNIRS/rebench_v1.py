"""Frozen, fNIRS-only MtReNIRS adaptation for the retained development panel.

The official encoder/checkpoint stay unchanged. This adapter owns sample-local
normalization, existing template geometry alignment and recorded-input masks.
It never loads signals or labels, and has no REFED sequence-regression head.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

UPSTREAM_REVISION = "16c3082747b689233f123519d2590b88631afb2d"


def load_upstream(root=None):
    root = Path(root) if root is not None else Path(__file__).parent / "upstream"
    for package in ("models", "alignment"):
        source = root / package / "src"
        if not source.is_dir():
            raise FileNotFoundError(f"Missing pinned MtReNIRS {package} package: {source}")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    from mtrenirs.models import MtReNIRS
    from mtrenirs.alignment import MtSi1005
    return MtReNIRS, MtSi1005, root


def geometry_from_profile(profile, reference):
    """Map only registered geometry; never infer anatomy from signal values.

    Exact fNIRS location names use the canonical template directly. Otherwise,
    named EEG template coordinates establish a similarity transform between
    coordinate frames; no EEG measurement enters the fNIRS representation.
    """
    names = profile["fnirs_channel_names"]
    geometry = profile["fnirs_geometry"]
    if len(names) != len(geometry) or any(g["channel_name"] != n for g, n in zip(geometry, names)):
        raise ValueError("fNIRS signal and geometry channel order disagree")
    kinds = [str(g["component"]).lower() for g in geometry]
    if any(kind not in {"hbo", "hbr"} for kind in kinds):
        raise ValueError("MtReNIRS requires real HbO/HbR identities")
    labels = list(reference["labels"])
    xyz = np.asarray(reference["positions"], float)
    lookup = {name.casefold(): i for i, name in enumerate(labels)}
    direct = [lookup.get(str(g["base_channel_name"]).casefold()) for g in geometry]
    valid = np.asarray([bool(g["position_available"]) for g in geometry])
    source = np.asarray([[g.get(axis, np.nan) for axis in ("x", "y", "z")] for g in geometry], float)
    valid &= np.isfinite(source).all(1)
    if not valid.all():
        raise ValueError("the declared fNIRS panel needs registered geometry for every channel")
    if all(index is not None for index in direct):
        indices = np.asarray(direct, int)
        details = dict(method="registered_exact_standard_1005_names", anchor_count=0)
    elif {g.get("coordinate_system") for g in geometry} == {"refed_paper_layout_2d_v1"}:
        # REFED's owning geometry permits coarse EEG/fNIRS neighbours only.
        # Select a registered name in the drawing, then use that name's model
        # template coordinate. Do not fit/lift the planar drawing into 3-D.
        anchors, template_indices = [], []
        for g in profile.get("eeg_geometry", []):
            index = lookup.get(str(g.get("channel_name", "")).casefold())
            point = np.asarray([g.get("x", np.nan), g.get("y", np.nan)], float)
            if (index is not None and g.get("position_available")
                    and g.get("coordinate_system") == "refed_paper_layout_2d_v1" and np.isfinite(point).all()):
                anchors.append(point); template_indices.append(index)
        anchors = np.asarray(anchors)
        if len(anchors) < 3 or np.linalg.matrix_rank(anchors - anchors.mean(0)) < 2:
            raise ValueError("REFED drawing needs at least three noncollinear named planar anchors")
        distances = np.linalg.norm(source[:, None, :2] - anchors[None], axis=-1)
        nearest = distances.argmin(1)
        indices = np.asarray(template_indices)[nearest]
        details = dict(method="refed_registered_2d_nearest_eeg_name_to_model_template",
            anchor_count=len(anchors), nearest_distance_figure_radius=distances[np.arange(len(nearest)), nearest].tolist(),
            interpretation="coarse_drawing_neighbour_names_only;template_model_positions_not_subject_3d_registration",
            source_deviation="fNIRS_location_quantized_to_nearest_registered_EEG_drawing_label")
    else:
        anchors, targets = [], []
        for g in profile.get("eeg_geometry", []):
            index = lookup.get(str(g.get("channel_name", "")).casefold())
            position = np.asarray([g.get(axis, np.nan) for axis in ("x", "y", "z")], float)
            if index is not None and g.get("position_available") and np.isfinite(position).all():
                anchors.append(position); targets.append(xyz[index])
        x, y = np.asarray(anchors), np.asarray(targets)
        if len(x) < 4 or np.linalg.matrix_rank(x - x.mean(0)) < 3:
            raise ValueError("unlabelled fNIRS geometry requires four noncoplanar registered template anchors")
        xc, yc = x - x.mean(0), y - y.mean(0)
        u, singular, vt = np.linalg.svd(xc.T @ yc)
        rotation = u @ vt
        scale = singular.sum() / np.square(xc).sum()
        mapped = (source - x.mean(0)) @ rotation * scale + y.mean(0)
        distances = np.linalg.norm(mapped[:, None] - xyz[None], axis=-1)
        indices = distances.argmin(1)
        residual = (xc @ rotation * scale + y.mean(0)) - y
        details = dict(method="registered_template_anchor_similarity_then_nearest_1005",
                       anchor_count=len(x), anchor_rms_m=float(np.sqrt(np.square(residual).sum(1).mean())),
                       nearest_distance_m=distances[np.arange(len(indices)), indices].tolist(),
                       frame_determinant=float(np.linalg.det(rotation)),
                       interpretation="template_coordinates_only_not_subject_digitization_or_EEG_signal_input")
    positions = xyz[indices].astype(np.float32)
    paired = {}
    for name, g, kind, index in zip(names, geometry, kinds, indices):
        pair = name.rsplit("_", 1)[0]
        if pair in paired and paired[pair] != int(index):
            raise ValueError("HbO and HbR of one physical pair map to different locations")
        paired[pair] = int(index)
    details.update(channel_names=names, chromophores=kinds,
                   canonical_labels=[labels[i] for i in indices],
                   physical_pairs=len(paired), canonical_locations=len(set(paired.values())),
                   source_coordinate_status=sorted({str(g.get("coordinate_status")) for g in geometry}))
    return kinds, positions, valid, details


def normalize_window(signal, mask):
    """Normalize over actual sample-local support; padding cannot alter scale."""
    if signal.ndim != 3 or mask.dtype != torch.bool or mask.shape != signal.shape:
        raise ValueError("fNIRS requires [B,C,T] values and matching boolean mask")
    if not torch.isfinite(signal[mask]).all():
        raise ValueError("nonfinite recorded fNIRS input")
    count = mask.sum(-1, keepdim=True)
    clean = torch.where(mask, signal, 0.)
    mean = clean.sum(-1, keepdim=True) / count.clamp_min(1)
    delta = torch.where(mask, clean - mean, 0.)
    variance = delta.square().sum(-1, keepdim=True) / count.clamp_min(1)
    scale = torch.where(variance > 0, variance.sqrt(), torch.ones_like(variance))
    return (delta / scale).clamp(-6., 6.)


class FrozenMtReNIRS(nn.Module):
    supports_mask = True
    supports_fnirs = True
    supports_temporal_output = False

    def __init__(self, profile, device="cpu", *, upstream_root=None):
        super().__init__()
        if float(profile["fnirs_sampling_rate_hz"]) != 10.:
            raise ValueError("MtReNIRS requires the common 10 Hz Hb input")
        model_type, alignment, root = load_upstream(upstream_root)
        kinds, positions, valid, geometry = geometry_from_profile(profile, alignment.load_reference())
        self.official = model_type.from_pretrained(root / "models/weights/mtrenirs_10s.pt", device=device,
            ch_types=kinds, ch_pos_1005=positions, ch_pos_1005_valid=valid)
        self.encoder = self.official.encoder.requires_grad_(False)
        self.register_buffer("coordinates", torch.as_tensor(self.official._coordinate_array(
            self.official.ch_types, positions, valid), device=device))
        self.expected_samples = int(profile["fnirs_samples"])
        self.channels = len(kinds)
        self.windows = (self.expected_samples + 99) // 100
        self.feature_dim = self.windows * self.channels * 128
        self.metadata = dict(upstream_revision=UPSTREAM_REVISION, geometry=geometry,
            input="fNIRS_only_HbO_HbR_10Hz", encoder_parameters=sum(p.numel() for p in self.encoder.parameters()),
            normalization="per_channel_valid_samples_in_the_same_declared_window_zscore_clip6",
            source_deviations=["common_0.01_0.2Hz_Hb_instead_of_source_0.01_0.5Hz",
                "sample_local_instead_of_full_run_normalization", "registered_template_geometry",
                "recorded_input_masks_applied_to_official_encoder_tokens"],
            padded_samples=self.windows * 100 - self.expected_samples,
            classification_only=True, pretraining_overlap="allowed_and_reported_by_user_request")

    @torch.no_grad()
    def tokens(self, signal, mask):
        if signal.shape[1:] != (self.channels, self.expected_samples):
            raise ValueError("fNIRS input changed the declared channel/time support")
        x = normalize_window(signal, mask)
        count = len(x)
        padded = x.new_zeros(count, self.channels, self.windows * 100)
        support = mask.new_zeros(padded.shape)
        padded[..., :self.expected_samples], support[..., :self.expected_samples] = x, mask
        tokens = padded.reshape(count, self.channels, self.windows, 100).transpose(1, 2)
        observed = support.reshape(count, self.channels, self.windows, 100).transpose(1, 2).sum(-1)
        valid = observed >= 2
        coordinates = self.coordinates.repeat(self.windows, 1).unsqueeze(0).expand(count, -1, -1)
        indices = torch.arange(self.windows, device=x.device).repeat_interleave(self.channels).expand(count, -1)
        encoded = self.encoder(tokens.reshape(count, -1, 100), coordinates, indices, valid.flatten(1))
        encoded = encoded.reshape(count, self.windows, self.channels, 128)
        if not torch.isfinite(encoded).all():
            raise ValueError("MtReNIRS produced nonfinite embeddings")
        return encoded, valid


def create_model(task, profile, device="cpu", trainable=False):
    if task == "refed_regression" or trainable:
        raise ValueError("this MtReNIRS adapter is frozen classification only")
    return FrozenMtReNIRS(profile, device).to(device)
