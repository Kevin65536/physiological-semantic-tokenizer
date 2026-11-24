"""Freeze the G0 development and confirmation examples before candidate scoring."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from run_dual_component_tokenizer import SOURCE
from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_examples


def ranked(key):
    return hashlib.sha256(('|'.join(map(str, key))).encode()).digest()


def prepare(rows):
    subjects = defaultdict(set)
    for row in rows:
        if row['split'] == 'validation':
            subjects[row['dataset']].add(row['subject'])
    confirmation = {d: min(subs, key=lambda s: ranked(('confirm-v1', d, s))) for d, subs in subjects.items()}
    panels = {}
    counts = {}
    for modality in ('eeg', 'hb'):
        roles = defaultdict(list)
        by_record = defaultdict(list)
        train_by_subject = defaultdict(list)
        for example in base_tokenizer_examples(rows, modality, 'validation'):
            i, start, pos = example
            row = rows[i]
            key = row['dataset'], row['subject']
            if key[1] == confirmation[key[0]]:
                roles['confirm'].append(example)
            else:
                roles['development'].append(example)
                by_record[(*key, row['record'])].append(example)
        for record, group in by_record.items():
            roles['screen'].extend(sorted(group, key=lambda e: ranked(('screen-v1', modality, *record, *e)))[:4])
        for example in base_tokenizer_examples(rows, modality, 'train'):
            row = rows[example[0]]
            train_by_subject[row['dataset'], row['subject']].append(example)
        for subject, group in train_by_subject.items():
            roles['pattern_train'].extend(sorted(group, key=lambda e: ranked(('pattern-v1', modality, *subject, *e)))[:256])
        for role, examples in roles.items():
            name = f'{modality}_{role}'
            panels[name] = np.asarray(sorted(examples), dtype=np.int32)
            counts[name] = dict(Counter(rows[i]['dataset'] for i, _, _ in examples))
    return panels, dict(confirmation_subjects=confirmation, counts=counts)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    config = SOURCE / 'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml'
    manifest, _, _, _ = base_tokenizer_training_data(yaml.safe_load(config.read_text()), SOURCE)
    arrays, metadata = prepare(manifest['rows'])
    metadata.update(source=str(config.relative_to(SOURCE)), cfg_identity=manifest['cfg_identity'],
                    scales_identity=manifest['scales_identity'], columns=['manifest_row', 'start_s', 'position'],
                    historical_exposure='All validation subjects contributed to historical G1 evaluation.')
    target = args.output / 'panels.npz'
    with target.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    metadata['panel_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    with (args.output / 'panel.json').open('x') as stream:
        json.dump(metadata, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
