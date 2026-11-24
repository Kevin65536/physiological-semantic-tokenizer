#!/usr/bin/env python3
"""Change the atomic time support under a frozen X-only Wiener predictor."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hashlib
import json
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits
import yaml

from experiments.scripts.run_dual_component_tokenizer import identity
from src.data.dual_component_tokenizer import base_tokenizer_training_data
from experiments.scripts.run_joint_waveform_dictionary import preserve_record, write_json
from src.data.dual_component_tokenizer import base_tokenizer_batch
from src.tokenizers.supervised_waveform_dictionary import SupervisedWaveformDictionary, load_model


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='experiments/configs/physiology_semantic_tokenizer/dual_component_joint_temporal_diagnostic_v1.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    if (cfg['schema'] != 'dual_component_joint_temporal_diagnostic_v1' or cfg['model']['context_s'] != 30
            or cfg['model']['modality'] != 'eeg' or cfg['model']['sample_rate_hz'] != 200
            or cfg['model']['quantization_levels'] != 1 or cfg['model']['short_time_units_interface_eligible']):
        raise ValueError('explicit EEG short-time single-layer diagnostic contract required')
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    source_path = ROOT / cfg['source']['checkpoint']
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source = load_model(source_path)
    support = json.loads((ROOT / cfg['source']['training_support']).read_text())
    if support['input_identity'] != identity(manifest) or source.metadata['input_identity'] != identity(manifest):
        raise ValueError('exact matched G0 reference required')
    examples = support['examples']
    rows = manifest['rows']
    if any(rows[i]['split'] != 'train' or rows[i]['modality'] != 'eeg' for i, _, _ in examples):
        raise ValueError('same training-only EEG population required before arrays')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    preserve_record(out / 'resolved_config.json', cfg)
    preserve_record(out / 'training_support.json', support)
    preserve_record(out / 'source_identity.json', dict(checkpoint=str(source_path), sha256=source_digest))
    if (out / 'summary.json').exists():
        retained = json.loads((out / 'summary.json').read_text())
        if (retained['status'] == 'trained_pending_diagnostic_evaluation'
                and len(retained['records']) == len(cfg['model']['time_units_s'])):
            for record in retained['records']:
                model = load_model(record['checkpoint'])
                if (model.metadata['config_identity'] != identity(cfg)
                        or model.metadata['input_identity'] != identity(manifest)
                        or model.metadata['reference_checkpoint'] != cfg['source']['checkpoint']
                        or model.metadata['token_s'] != record['atom_support_s']
                        or model.codebook_size != 1024):
                    raise ValueError('completed temporal diagnostic identity changed')
            return
    cache, parts = OrderedDict(), []
    for start in range(0, len(examples), 128):
        parts.append(base_tokenizer_batch(rows, examples[start:start + 128], scales, array_cache=cache))
    batch = {key: np.concatenate([part[key] for part in parts]) for key in ('x', 's', 'o', 'mask')}
    del parts
    records = []
    with threadpool_limits(limits=cfg['resources']['numerical_threads']):
        for seconds in cfg['model']['time_units_s']:
            dimension = int(round(seconds * 200))
            if dimension < 1 or dimension / 200 != seconds or 6000 % dimension:
                raise ValueError('integer short support must exactly partition 30 seconds')
            destination = out / ('unit_' + str(seconds).replace('.', 'p') + '_k1024.npz')
            if destination.exists():
                model = load_model(destination)
                if (model.metadata['config_identity'] != identity(cfg)
                        or model.metadata['input_identity'] != identity(manifest)
                        or model.metadata.get('source_sha256', source_digest) != source_digest
                        or model.atom_dim != dimension or model.codebook_size != 1024):
                    raise ValueError('retained temporal checkpoint identity changed')
                records.append(dict(checkpoint=str(destination), atom_support_s=seconds, interface_eligible=False,
                    recovered_existing_checkpoint=True, integer_replay='retained',
                    integers_per_branch_context=model.atom_patches, bits_per_branch_position_per_second=10 / seconds))
                continue
            started = time.monotonic()
            valid = batch['mask'].reshape(-1, dimension).all(axis=1)
            means, bases, atoms = {}, {}, {}
            for branch in ('S', 'O'):
                values = batch[branch.lower()].reshape(-1, dimension)[valid].astype(np.float64)
                means[branch] = values.mean(0)
                _, eigenvectors = np.linalg.eigh(np.cov(values, rowvar=False))
                bases[branch] = eigenvectors[:, -min(128, dimension):][:, ::-1].T
                projected = ((values - means[branch]) @ bases[branch].T).astype(np.float32)
                setting = cfg['training']
                clustering = MiniBatchKMeans(n_clusters=1024, random_state=setting['seed'],
                    n_init=setting['kmeans_n_init'], max_iter=setting['kmeans_max_iter'],
                    max_no_improvement=setting['kmeans_max_no_improvement'],
                    batch_size=setting['kmeans_batch_size'], reassignment_ratio=setting['kmeans_reassignment_ratio'],
                    init_size=3072)
                labels = clustering.fit_predict(projected)
                counts = np.bincount(labels, minlength=1024)
                sums = np.zeros((1024, dimension))
                np.add.at(sums, labels, values)
                active = counts > 0
                sums[active] /= counts[active, None]
                sums[~active] = means[branch] + clustering.cluster_centers_[~active] @ bases[branch]
                atoms[branch] = sums
            metadata = dict(source.metadata, atom_dim=dimension, atom_patches=6000 // dimension,
                interface_eligible=False, token_s=seconds, context_s=30,
                config_identity=identity(cfg), reference_checkpoint=cfg['source']['checkpoint'], source_sha256=source_digest,
                integers_per_30s_per_position=2 * (6000 // dimension),
                bits_per_branch_per_position_per_second=10 / seconds,
                fixed_length_bits_per_30s_per_position=20 * (6000 // dimension),
                fitting='same estimator seed and training windows; single layer, changed atom duration',
                complete_training_atoms=int(valid.sum()), validation_arrays_read=0)
            model = SupervisedWaveformDictionary(metadata, source.transfer, means, bases, atoms)
            model.save(destination)
            ids = model.encode(batch['x'][:8])
            for branch in ('S', 'O'):
                np.testing.assert_array_equal(model.decode_ids(ids)[branch], load_model(destination).decode_ids(ids)[branch])
            record = dict(checkpoint=str(destination), seconds=time.monotonic() - started,
                          atom_support_s=seconds, interface_eligible=False, integer_replay='exact',
                          integers_per_branch_context=6000 // dimension,
                          bits_per_branch_position_per_second=10 / seconds)
            records.append(record)
            write_json(out / 'progress.json', dict(status='training', completed=records))
            print(json.dumps(record), flush=True)
    write_json(out / 'summary.json', dict(status='trained_pending_diagnostic_evaluation', records=records,
                                         validation_arrays_read=0, test_arrays_read=0))


if __name__ == '__main__':
    main()
