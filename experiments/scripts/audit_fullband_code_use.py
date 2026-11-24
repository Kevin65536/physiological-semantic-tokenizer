#!/usr/bin/env python3
"""Audit actual fullband IDs and latent geometry on fixed train/development panels."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from experiments.scripts.run_dual_component_tokenizer import identity, write_json
from src.data.dual_component_tokenizer import base_tokenizer_training_data
from src.data.dual_component_tokenizer import base_tokenizer_batch
from src.tokenizers.fullband_eeg_tokenizer import load_model


def audit(model, rows, scales, examples, role, output):
    size = model.codebook_size
    counts = {b: np.zeros(size, np.int64) for b in ('S', 'O')}
    complete_counts = {b: np.zeros(size, np.int64) for b in ('S', 'O')}
    subjects = {b: set() for b in ('S', 'O')}
    moments = {b: [] for b in ('S', 'O')}
    cache = OrderedDict()
    for offset in range(0, len(examples), 32):
        take = examples[offset:offset+32]
        batch = base_tokenizer_batch(rows, take, scales, array_cache=cache, cache_limit=64)
        complete = batch['mask'].all(-1)
        with torch.inference_mode():
            latents = model.model.encode_continuous(torch.as_tensor(batch['x'], dtype=torch.float32))
            for b, z in latents.items():
                quantizer = model.model.branches[b]['quantizer']
                ids = quantizer._assignment_logits(z).argmax(-1)
                lookup = quantizer.get_embedding(ids)
                distances = (z-lookup).square().mean(-1)
                norm = z.square().mean(-1)
                centroid_norm = lookup.square().mean(-1)
                code = ids.numpy()
                counts[b] += np.bincount(code.reshape(-1), minlength=size)
                complete_counts[b] += np.bincount(code[complete], minlength=size)
                for row, codes, full in zip(take, code, complete):
                    source = rows[row[0]]
                    subjects[b].update((int(c), source['dataset'], source['subject']) for c in np.unique(codes[full]))
                moments[b].append(np.stack([distances.numpy(), norm.numpy(), centroid_norm.numpy()], axis=-1).reshape(-1, 3))
        write_json(output/'progress.json', dict(role=role, completed=min(offset+32, len(examples)), total=len(examples)))
    result = {}
    for b in ('S', 'O'):
        geometry = np.concatenate(moments[b])
        n_subjects = np.zeros(size, np.int64)
        for code, _, _ in subjects[b]:
            n_subjects[code] += 1
        count = counts[b]
        p = count[count > 0]/count.sum()
        result[b] = dict(inputs=len(examples), total_ids=int(count.sum()), active_codes=int((count > 0).sum()),
            unused_codes=int((count == 0).sum()), codebook_size=size, effective_codes=float(np.exp(-(p*np.log(p)).sum())),
            largest_code_fraction=float(p.max()), top10_code_fraction=float(np.sort(count)[-10:].sum()/count.sum()),
            geometry_fields=['latent_to_centroid_MSE', 'latent_mean_square', 'assigned_centroid_mean_square'],
            geometry_mean=geometry.mean(0).tolist(), geometry_quantiles=np.quantile(geometry, [.1, .5, .9, .99], axis=0).tolist(),
            count=count.tolist(), complete_patch_count=complete_counts[b].tolist(), subject_count=n_subjects.tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--panels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    if (args.output/'summary.json').exists():
        raise FileExistsError('completed code audit is retained')
    torch.set_num_threads(4)
    cfg = yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    record = json.loads((args.panels.parent/'panel.json').read_text())
    if (record['panel_sha256'] != hashlib.sha256(args.panels.read_bytes()).hexdigest()
            or record['cfg_identity'] != manifest['cfg_identity']
            or record['scales_identity'] != manifest['scales_identity']):
        raise ValueError('fixed panel identity drift before arrays')
    model = load_model(args.checkpoint)
    if model.metadata['input_identity'] != identity(manifest) or model.metadata['scales_identity'] != identity(scales):
        raise ValueError('frozen checkpoint data identity drift before arrays')
    rows = manifest['rows']
    with np.load(args.panels, allow_pickle=False) as arrays:
        panels = {role: arrays['eeg_'+key] for role, key in (('train', 'pattern_train'), ('screen', 'screen'))}
    for role, examples in panels.items():
        expected_split = 'train' if role == 'train' else 'validation'
        if any(rows[i]['modality'] != 'eeg' or rows[i]['split'] != expected_split for i, _, _ in examples):
            raise ValueError('fixed role/split drift before arrays')
    began = time.monotonic()
    scores = {role: audit(model, rows, scales, examples, role, args.output) for role, examples in panels.items()}
    for b in ('S', 'O'):
        train, validation = scores['train'][b], scores['screen'][b]
        supported = (np.asarray(train['complete_patch_count']) >= 8) & (np.asarray(train['subject_count']) >= 2)
        val_counts = np.asarray(validation['count'])
        validation['mass_on_8patch_2subject_training_codes'] = float(val_counts[supported].sum()/val_counts.sum())
        quantizer = model.model.branches[b]['quantizer']
        scores['train'][b]['ema_state'] = dict(update_count=int(quantizer.update_count), revival_count=int(quantizer.revival_count),
            counts_below_0p1=int((quantizer.ema_count < .1).sum()),
            centroid_mean_square=float(quantizer.codebook.square().mean()),
            ema_count_quantiles=np.quantile(quantizer.ema_count.numpy(), [0, .1, .5, .9, 1]).tolist())
    write_json(args.output/'summary.json', dict(status='completed', checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        panel_sha256=record['panel_sha256'], scope='registered train pattern panel and development screen; no confirmation',
        code_usage=scores, elapsed_seconds=time.monotonic()-began))


if __name__ == '__main__':
    main()
