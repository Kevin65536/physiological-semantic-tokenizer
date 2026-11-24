"""Supervised atomic Hb children with a fixed map to the original parent decoder."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.tree import DecisionTreeRegressor
import torch

from .contextual_latent_dictionary import ContextualLatentDictionary
from .dual_component_tokenizer import GlobalLatentComponentTokenizer


def load_parent(checkpoint, setting, metadata):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = GlobalLatentComponentTokenizer(setting, codebook_size=128, hard=True).eval()
    model.load_state_dict(saved["model_state"])
    return ContextualLatentDictionary(model, metadata)


def local_features(values, latent):
    values = np.asarray(values, np.float32)
    wave = values.reshape(*values.shape[:2], 20, 2)
    t = (np.arange(20) - 9.5) / 10
    q = t * t - np.mean(t * t)
    mean = wave.mean(2)
    centered = wave - mean[:, :, None]
    slope = np.einsum("nptc,t->npc", wave, t / (t @ t))
    curvature = np.einsum("nptc,t->npc", wave, 2 * q / (q @ q))
    rms = np.sqrt(np.mean(wave * wave, axis=2))
    return np.concatenate(
        (
            np.asarray(latent),
            centered.reshape(values.shape),
            mean,
            slope,
            curvature,
            rms,
        ),
        axis=-1,
    )


def encode_parent(parent, values):
    tensor = torch.as_tensor(values, dtype=torch.float32)
    with torch.inference_mode():
        latent = parent.model.encode_continuous(tensor)
        ids = {
            b: parent.model.branches[b]["quantizer"]
            ._assignment_logits(z.reshape(-1, z.shape[-1]))
            .argmax(-1)
            .reshape(z.shape[:2])
            .numpy()
            for b, z in latent.items()
        }
    return ids, local_features(values, latent["O"].numpy())


def fit_children(parent_ids, features, supervision, residual, subjects, settings, seed):
    width = features.shape[1]
    parent = np.asarray(parent_ids)
    target = np.asarray(supervision, float)
    # The supervision coordinate is fitted anew for every outer training group.
    sigma = target.std(0).clip(1e-8)
    normalized = target / sigma
    normalized[:, :40] /= np.sqrt(40)
    normalized[:, 40:] /= np.sqrt(target.shape[1] - 40)
    hbo_weight = np.sqrt(settings.get("HbO_supervision_weight", 1.0))
    normalized[:, :40:2] *= hbo_weight
    normalized[:, 40] *= hbo_weight
    trees = {}
    offsets = {}
    lookup = []
    corrections = []
    support = []
    parents = []

    def fit_one_parent(code):
        select = np.flatnonzero(parent == code)
        tree = None
        direction = None
        leaf_labels = np.zeros(len(select), int)
        if (
            settings["max_leaves"] > 1
            and len(select) >= 2 * settings["minimum_leaf_complete_training_patches"]
        ):
            candidate = DecisionTreeRegressor(
                max_leaf_nodes=settings["max_leaves"],
                min_samples_leaf=settings["minimum_leaf_complete_training_patches"],
                random_state=seed + code,
            )
            predictors = features[select]
            if settings.get("feature_mode") == "ridge":
                mean = predictors.mean(0)
                scale = predictors.std(0).clip(1e-6)
                standardized = (predictors-mean)/scale
                response = target[select, 40:42]
                response_mean = response.mean(0)
                gram = standardized.T @ standardized / len(select)
                penalty = settings['direction_ridge'] * max(float(np.trace(gram)/len(gram)), 1e-12)
                coefficient = np.linalg.solve(gram + penalty*np.eye(len(gram)), standardized.T @ (response-response_mean)/len(select))
                direction = dict(mean=mean, scale=scale, coefficient=coefficient, intercept=response_mean)
                predictors = np.column_stack([predictors, standardized @ coefficient + response_mean])
            candidate.fit(predictors, normalized[select])
            labels = candidate.apply(predictors)
            leaves = np.unique(labels)
            valid = all(
                len(set(subjects[select[labels == leaf]].tolist()))
                >= settings["minimum_leaf_training_subjects"]
                for leaf in leaves
            )
            impurity = candidate.tree_.impurity
            n = candidate.tree_.weighted_n_node_samples
            reduction = 1 - float(np.sum(impurity[leaves] * n[leaves])) / max(
                1e-12, float(impurity[0] * n[0])
            )
            if (
                valid
                and len(leaves) > 1
                and reduction
                >= settings["minimum_parent_supervised_impurity_reduction"]
            ):
                tree = candidate
                leaf_labels = labels
        return select, tree, leaf_labels, direction

    with ThreadPoolExecutor(max_workers=settings.get("tree_workers", 1)) as pool:
        children = list(pool.map(fit_one_parent, range(128)))
    directions = {}
    for code, (select, tree, leaf_labels, direction) in enumerate(children):
        directions[code] = direction
        start = len(lookup)
        labels = np.unique(leaf_labels) if len(select) else np.array([0])
        mapping = {}
        for leaf in labels:
            use = select[leaf_labels == leaf]
            leaf_id = len(lookup)
            mapping[int(leaf)] = leaf_id
            lookup.append(code)
            delta = residual[use].mean(0) if len(use) else np.zeros(40)
            delta = delta.reshape(20, 2)
            delta -= delta.mean(0, keepdims=True)
            corrections.append(delta.reshape(40))
            support.append(
                dict(
                    ID=leaf_id,
                    parent=code,
                    complete_patches=len(use),
                    subjects=len(set(subjects[use].tolist())),
                )
            )
        offsets[code] = mapping
        trees[code] = tree
        parents.append(
            dict(
                parent=code,
                first_leaf=start,
                children=len(labels),
                split=tree is not None,
            )
        )
    return dict(
        trees=trees,
        directions=directions,
        node_to_ID=offsets,
        parent=np.asarray(lookup, np.int32),
        residual=np.asarray(corrections),
        support=support,
        parents=parents,
        feature_width=width,
        target_SD=sigma,
    )


def assign_leaf(parent_ids, features, fitted):
    shape = np.shape(parent_ids)
    codes = np.asarray(parent_ids).reshape(-1)
    flat = np.asarray(features).reshape(-1, fitted["feature_width"])
    result = np.empty(len(codes), np.int32)
    for code in range(128):
        selected = np.flatnonzero(codes == code)
        if not len(selected):
            continue
        tree = fitted["trees"][code]
        predictors = flat[selected]
        direction = fitted.get('directions', {}).get(code)
        if tree is not None and direction is not None:
            predicted = ((predictors-direction['mean'])/direction['scale']) @ direction['coefficient'] + direction['intercept']
            predictors = np.column_stack([predictors,predicted])
        labels = tree.apply(predictors) if tree is not None else np.zeros(len(selected),int)
        result[selected] = [fitted["node_to_ID"][code][int(label)] for label in labels]
    return result.reshape(shape)


def synthesize_parent(original, leaf, fitted, metadata, quiet):
    """Apply a trained leaf residual while retaining the frozen parent sum."""
    correction = fitted["residual"][leaf].reshape(len(leaf), 300, 2)
    sos = butter(4, metadata["residual_lowpass_hz"], fs=10, output="sos")
    correction = sosfiltfilt(sos, correction, axis=1)
    other = (
        np.asarray(original["O"], float).reshape(len(leaf), 300, 2)
        * np.asarray(metadata["color_gain"])[None, None]
    )
    other += metadata["residual_strength"] * correction
    conserved = original["sum"].copy()
    other = other.reshape(original["O"].shape)
    primary = np.asarray(conserved, float) - other
    if quiet.any():
        level = np.tile(metadata["quiet_level_S"], 20)
        primary[quiet] = level[None, None]
        other[quiet] = -level[None, None]
        conserved[quiet] = 0.0
    return dict(S=primary, O=other, sum=conserved)


def emitted_error_polynomials(original, at_one, targets, mask, variances, quiet):
    """Per-window NMSE A(g-1)^2+B(g-1)+C and the exact parent baseline.

    ``at_one`` is the actual synthesizer output, including its fixed residual
    and reserved quiet pair. A gain changes ordinary O by parent_O and S by
    its opposite. Quiet rows have zero derivative and retain their scored error.
    """
    mask=np.asarray(mask,bool).reshape(len(quiet),300,2)
    parent_O=np.asarray(original['O'],float).reshape(mask.shape)
    derivative=np.where(np.asarray(quiet)[:,None,None],0.,parent_O)
    count=np.maximum(mask.sum(1),1)
    result=np.empty((len(quiet),3,2,4))
    for j,b in enumerate(['sum','S','O']):
        error=np.asarray(at_one[b],float).reshape(mask.shape)-targets[b]
        base=np.asarray(original[b],float).reshape(mask.shape)-targets[b]
        d=derivative*(1 if b=='O' else -1) if b!='sum' else np.zeros_like(derivative)
        for k,value in enumerate([d*d,2*error*d,error*error,base*base]):
            result[:,j,:,k]=np.where(mask,value,0.).sum(1)/count/variances[:,j]
    return result


class HbParentLeaf:
    def __init__(self, parent, metadata, fitted):
        self.parent, self.metadata, self.fitted = parent, metadata, fitted
        self.codebook_size = int(metadata["branch_codebook_sizes"]["O"])
        self.leaf_count = len(fitted["parent"])
        self.quiet_S = 128
        self.quiet_O = self.leaf_count

    def encode(self, values):
        ids, features = encode_parent(self.parent, values)
        leaf = assign_leaf(ids["O"], features, self.fitted)
        prediction = self.parent.decode_ids(ids)
        energy = np.square(np.asarray(values, float)).mean((1, 2))
        low = np.sqrt(energy) <= self.metadata["quiet_threshold"]
        quiet = low & (
            energy
            < np.square(np.asarray(prediction["sum"], float) - values).mean((1, 2))
        )
        result = dict(S=ids["S"].copy(), O=leaf)
        result["S"][quiet] = self.quiet_S
        result["O"][quiet] = self.quiet_O
        return result

    def decode_ids(self, ids):
        a = {b: np.asarray(ids[b]) for b in "SO"}
        shape = a["S"].shape
        if (
            len(shape) != 2
            or shape[1] != 15
            or any(v.shape != shape or v.dtype.kind not in "iu" for v in a.values())
        ):
            raise ValueError(
                "one atomic integer per branch and two-second position required"
            )
        if (
            (a["S"] < 0).any()
            or (a["S"] > self.quiet_S).any()
            or (a["O"] < 0).any()
            or (a["O"] > self.quiet_O).any()
        ):
            raise ValueError("in-range parent S and atomic O leaf IDs required")
        quiet = (a["S"] == self.quiet_S).all(1) & (a["O"] == self.quiet_O).all(1)
        any_reserved = (a["S"] == self.quiet_S).any(1) | (a["O"] == self.quiet_O).any(1)
        if not np.array_equal(quiet, any_reserved):
            raise ValueError("quiet branch IDs require a complete reserved pair")
        safeS = np.where(quiet[:, None], 0, a["S"])
        safeO = np.where(quiet[:, None], 0, a["O"])
        parents = dict(S=safeS, O=self.fitted["parent"][safeO])
        original = self.parent.decode_ids(parents)
        return synthesize_parent(original, safeO, self.fitted, self.metadata, quiet)

    def save(self, path):
        torch.save(dict(metadata=self.metadata, fitted=self.fitted), path)


def load_model(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    m = saved["metadata"]
    if m["schema"] != "hb_parent_leaf_v1":
        raise ValueError("frozen atomic parent-leaf Hb model required")
    p = Path(m["parent_checkpoint"])
    with p.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != m["parent_sha256"]:
            raise ValueError("frozen G1 parent changed")
    parent = load_parent(p, m["parent_model_setting"], m["parent_metadata"])
    return HbParentLeaf(parent, m, saved["fitted"])
