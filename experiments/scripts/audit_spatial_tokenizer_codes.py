#!/usr/bin/env python3
"""Audit spatial integer costs and code support conditional on branch and fixed band."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from src.tokenizers.spatial_waveform_dictionary import BANDS, load_model


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--integers', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    if (args.output / 'code_audit.json').exists():
        raise FileExistsError('retained completed code audit')
    model = load_model(args.run / 'model.npz')
    metadata = json.loads(args.integers.with_name('integer_metadata.json').read_text())
    with np.load(args.panel) as stored:
        examples = stored['eeg_screen']
    selected = defaultdict(set)
    resolved = json.loads((args.run / 'resolved_config.json').read_text())
    manifest_path = ROOT / resolved['config']['data']['run'] / 'cache/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['cfg_identity'] != model.metadata['input_identity']:
        raise ValueError('frozen G0 code audit identity changed')
    rows = manifest['rows']
    for row_index, start, position in examples:
        selected[int(row_index), int(start)].add(rows[row_index]['positions'][position].upper())
    usage = {(branch, band): np.zeros(model.codebook_size, dtype=np.int64)
             for branch in ('S', 'O') for band in range(len(BANDS))}
    subjects = defaultdict(lambda: defaultdict(set))
    datasets = defaultdict(lambda: defaultdict(set))
    costs = []
    with np.load(args.integers, allow_pickle=False) as stored:
        for item in metadata:
            dataset = item['dataset']
            names = model.metadata['positions'][dataset]
            if names != item['native_positions']:
                raise ValueError('fixed native electrode order changed')
            roles = model.roles(dataset)
            scored_names = selected[item['row'], item['start_s']]
            auxiliary = np.array([name.upper() not in scored_names for name in names])
            codes = {branch: stored[f"{branch}_{item['crop']}"] for branch in ('S', 'O')}
            cost = dict(crop=item['crop'], dataset=dataset, subject=item['subject'], row=item['row'], start_s=item['start_s'],
                        native_positions=len(names), selected_scoring_positions=len(scored_names),
                        declared_integer_slots=sum(value.size for value in codes.values()),
                        nonmissing_integer_ids=sum(int(np.count_nonzero(value)) for value in codes.values()),
                        auxiliary_declared_integer_slots=sum(value[:, auxiliary].size for value in codes.values()),
                        auxiliary_nonmissing_integer_ids=sum(int(np.count_nonzero(value[:, auxiliary])) for value in codes.values()))
            costs.append(cost)
            for branch, value in codes.items():
                for band in range(len(BANDS)):
                    ids = value[:, roles == band].reshape(-1)
                    usage[branch, band] += np.bincount(ids, minlength=model.codebook_size)
                    for code in np.unique(ids):
                        if code:
                            subjects[branch, band][int(code)].add((dataset, item['subject']))
                            datasets[branch, band][int(code)].add(dataset)
    result = []
    for (branch, band), counts in usage.items():
        with np.load(args.run / f'{branch}_band{band}.npz') as stored:
            training_counts = np.r_[0, stored['counts']]
        positive = counts[1:]
        probabilities = positive[positive > 0] / max(positive.sum(), 1)
        used = np.flatnonzero(positive > 0) + 1
        result.append(dict(branch=branch, fixed_band_hz=BANDS[band], available_nonmissing_codes=model.codebook_size - 1,
            dictionary_fitting_target_patch_occupied_codes=int(np.count_nonzero(training_counts)),
            dictionary_fitting_target_patch_dead_codes=int(np.count_nonzero(training_counts[1:] == 0)),
            actual_validation_X_encoded_occupied_codes=len(used), validation_nonmissing_ids=int(positive.sum()),
            reserved_missing_ids=int(counts[0]), conditional_entropy_bits=float(-np.sum(probabilities * np.log2(probabilities))),
            actual_validation_codes_without_dictionary_fitting_support=int(np.sum(training_counts[used] == 0)),
            validation_codes_with_at_least_two_subjects=sum(len(subjects[branch, band][int(code)]) >= 2 for code in used),
            validation_codes_with_at_least_two_datasets=sum(len(datasets[branch, band][int(code)]) >= 2 for code in used),
            rare_validation_codes_under_five_members=int(np.sum((positive > 0) & (positive < 5))),
            by_code=[dict(code=int(code), dictionary_target_training_patches=int(training_counts[code]),
                         validation_members=int(counts[code]), validation_subjects=len(subjects[branch, band][int(code)]),
                         validation_datasets=sorted(datasets[branch, band][int(code)])) for code in used]))
    total_ids = sum(item['declared_integer_slots'] for item in costs)
    total_position_seconds = sum(item['native_positions'] * 30 for item in costs)
    scored_position_seconds = sum(item['selected_scoring_positions'] * 30 for item in costs)
    payload = dict(status='completed', code_identity='(branch,fixed_band,integer_ID); same number across bands names different atoms',
        role='frozen_screen', normalization='frozen G0 train scales per dataset and native position',
        integer_count=total_ids, reserved_missing_integer_count=total_ids - sum(item['nonmissing_integer_ids'] for item in costs),
        auxiliary_integer_slots=sum(item['auxiliary_declared_integer_slots'] for item in costs),
        auxiliary_nonmissing_integer_ids=sum(item['auxiliary_nonmissing_integer_ids'] for item in costs),
        unique_joint_crops=len(costs), selected_scoring_position_windows=len(examples),
        entire_joint_context_position_seconds=total_position_seconds,
        nominal_fixed_width_bits_per_ID=int(np.ceil(np.log2(model.codebook_size))),
        nominal_total_bits=total_ids * int(np.ceil(np.log2(model.codebook_size))),
        nominal_bits_per_second_per_position_both_EEG_branches=total_ids * np.log2(model.codebook_size) / total_position_seconds,
        amortized_nominal_bits_per_second_per_selected_scoring_position=total_ids * np.log2(model.codebook_size) / scored_position_seconds,
        saved_npz_bytes=args.integers.stat().st_size,
        saved_npz_bits_per_second_per_joint_position=args.integers.stat().st_size * 8 / total_position_seconds,
        storage='NPZ stores int32 arrays with ZIP compression; nominal13-bit packing is not implemented',
        training_usage_scope='Dictionary fitting labels from true train targets are retained. Actual X-only train encoding occupancy has not been re-audited.',
        positions_used='Scored targets use the full native-layout integer sequences, including unsampled auxiliary positions. All such integer slots enter the numerator.',
        crop_costs=costs, code_support=result)
    (args.output / 'code_audit.json').write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
