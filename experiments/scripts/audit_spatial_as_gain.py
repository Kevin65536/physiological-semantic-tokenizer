#!/usr/bin/env python3
"""Render the fixed AS examples through both train-frozen gain decoders."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.analysis.spatial_tokenizer import write
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_training_data
from src.tokenizers.spatial_waveform_dictionary import load_model
from src.tokenizers.spatial_local_gain import load_decoder

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--local-gain-run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    out = (args.local_gain_run.resolve() if args.local_gain_run else run) / 'visual_audit_v1'
    out.mkdir(exist_ok=False)
    cfg = json.loads((run / 'resolved_config.json').read_text())['config']
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    original = ROOT / cfg['source']['screen_ids']
    parent = original.parent
    examples = json.loads((parent / 'assignment_audit_v1/figures.json').read_text())
    metadata = json.loads(original.with_name('integer_metadata.json').read_text())
    lookup = {(v['row'], v['start_s']): v for v in metadata}
    models = {'original_AS': load_model(ROOT / cfg['source']['decoder']), 'minimum_MSE': load_model(run / 'minimum_MSE/model.npz'), 'energy_constrained': load_model(run / 'energy_constrained/model.npz')}
    if args.local_gain_run:
        models = {'original_AS': models['original_AS'], 'fixed_gain': models['energy_constrained'], 'local_gain': load_decoder(args.local_gain_run.resolve() / 'model.npz')}
    with np.load(original) as stored:
        ids = {key: stored[key].copy() for key in stored.files}
    with np.load(run / 'minimum_MSE/integer_ids.npz') as mse, np.load(run / 'energy_constrained/integer_ids.npz') as energy:
        for key, value in ids.items():
            np.testing.assert_array_equal(value, mse[key])
            np.testing.assert_array_equal(value, energy[key])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cache = OrderedDict()
    records = []
    for example in examples:
        i = example['manifest_row']
        start = example['start_s']
        position = example['position_index']
        row = manifest['rows'][i]
        group = lookup[i, start]
        dataset = row['dataset']
        names = group['native_positions']
        crop = group['crop']
        native = next((k for k, name in enumerate(names) if name.upper() == row['positions'][position].upper()))
        codes = {b: ids[f'{b}_{crop}'] for b in ('S', 'O')}
        decoded = {key: model.decode_ids(codes, dataset) for key, model in models.items()}
        batch = base_tokenizer_batch(manifest['rows'], [(i, start, position)], scales, array_cache=cache, cache_limit=8)
        scale = scales['positions']['eeg'][dataset][row['positions'][position]]['scale']
        mask = batch['mask'][0].reshape(-1)
        time_axis = np.arange(6000) / 200
        fig, axes = plt.subplots(3, 3, figsize=(15, 8), constrained_layout=True)
        for r, (branch, field) in enumerate([('S', 's'), ('O', 'o'), ('sum', 'x')]):
            truth = batch[field][0].reshape(-1) * scale
            waves = {'target': truth, **{key: value[branch][0, native].reshape(-1) * scale for key, value in decoded.items()}}
            for c, limits in enumerate([(0, 30), (14, 16), (13.5, 14.5)]):
                ax = axes[r, c]
                for (key, value), color, style in zip(waves.items(), ['black', '#2b6caa', '#9052a0', '#d95f02'], ['-', ':', '--', '-']):
                    ax.plot(time_axis, value, color=color, ls=style, lw=0.85, label=key)
                ax.set_xlim(*limits)
                selected = (time_axis >= limits[0]) & (time_axis < limits[1])
                visible = np.concatenate([value[selected] for value in waves.values()])
                lo, hi = (np.nanmin(visible), np.nanmax(visible))
                padding = 0.05 * (hi - lo + 1e-15)
                ax.set_ylim(lo - padding, hi + padding)
                for boundary in range(2, 30, 2):
                    ax.axvline(boundary, color='grey', ls=':', lw=0.45)
                transitions = np.diff(np.r_[False, ~mask, False].astype(int))
                for a, b in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
                    ax.axvspan(a / 200, b / 200, color='grey', alpha=0.3)
                ax.set_xlabel('Window time (s)')
                ax.set_ylabel(f'{branch} ({row['source_unit']})')
                if r == 0 and c == 0:
                    ax.legend(fontsize=8)
        fig.suptitle(f'{dataset} / {row['subject']} / {row['record']} / {row['positions'][position]} / start {start}s / {example['selection']}\nIdentical AS integers; train-fitted band gains; token boundaries dotted; unscored support grey', fontsize=10)
        path = out / (dataset + '_' + example['selection'] + '.png')
        fig.savefig(path, dpi=200)
        plt.close(fig)
        records.append(dict(example, figure=str(path), comparison=list(models), image_opened=False))
    write(out / 'figures.json', records)
    write(out / 'integer_pairing_verification.json', dict(status='exact', same_saved_integer_files=True, models=list(models), new_transmitted_bits=0))
if __name__ == '__main__':
    main()
