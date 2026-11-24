#!/usr/bin/env python3
"""Compare saved integer changes and render fixed EEG examples after search."""
from __future__ import annotations
import argparse
from collections import OrderedDict, defaultdict
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import yaml
from src.analysis.spatial_tokenizer import write
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_training_data
from src.tokenizers.spatial_analysis_synthesis import load_encoder
from src.tokenizers.spatial_waveform_dictionary import band_filter, load_model

def quantiles(values):
    values = np.asarray(values, float)
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)), p90=float(np.quantile(values, 0.9)), maximum=float(values.max())) if len(values) else dict(count=0)

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    out = run / 'assignment_audit_v1'
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((run / 'resolved_config.json').read_text())['config']
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    model = load_encoder(run / 'encoder.json')
    source_path = ROOT / cfg['source']['decoder']
    source = load_model(source_path)
    baseline_folder = source_path.parent
    meta = json.loads((run / 'integer_metadata.json').read_text())
    oldmeta = json.loads((baseline_folder / 'integer_metadata.json').read_text())
    if meta != oldmeta:
        raise ValueError('source and searched integer crops differ')
    statistics = defaultdict(lambda: dict(planned=0, changed=0, energy=[], envelope=[]))
    oldcodes, newcodes = ({}, {})
    with np.load(baseline_folder / 'integer_ids.npz') as old, np.load(run / 'integer_ids.npz') as new:
        if set(old.files) != set(new.files):
            raise ValueError('source and searched integer fields differ')
        oldcodes = {key: old[key].copy() for key in old.files}
        newcodes = {key: new[key].copy() for key in new.files}
    features = {}
    for branch in ('S', 'O'):
        for band in range(5):
            waveform = band_filter(source.atoms[branch, band], band)
            features[branch, band] = dict(energy=np.mean(waveform ** 2, axis=1), envelope=np.sqrt(np.mean(waveform.reshape(len(waveform), 8, 50) ** 2, axis=2)))
    for item in meta:
        dataset = item['dataset']
        crop = item['crop']
        roles = model.roles(dataset)
        for branch in ('S', 'O'):
            before = oldcodes[f'{branch}_{crop}']
            after = newcodes[f'{branch}_{crop}']
            if before.shape != after.shape:
                raise ValueError('integer count changed')
            np.testing.assert_array_equal(before == 0, after == 0)
            for band in range(5):
                old = before[:, roles == band].reshape(-1)
                new = after[:, roles == band].reshape(-1)
                changed = (old != new) & (old != 0)
                row = statistics[dataset, branch, band]
                row['planned'] += len(old)
                row['changed'] += int(changed.sum())
                if changed.any():
                    energy = features[branch, band]['energy']
                    envelope = features[branch, band]['envelope']
                    row['energy'].extend((10 * np.log10(np.maximum(energy[new[changed]], 1e-12) / np.maximum(energy[old[changed]], 1e-12))).tolist())
                    drift = np.mean((envelope[new[changed]] - envelope[old[changed]]) ** 2, axis=1) / np.maximum(np.mean(envelope[old[changed]] ** 2, axis=1), 1e-12)
                    row['envelope'].extend(drift.tolist())
    rows = []
    for (dataset, branch, band), row in sorted(statistics.items()):
        rows.append(dict(dataset=dataset, branch=branch, role_band=band, integers=row['planned'], changed=row['changed'], changed_fraction=row['changed'] / row['planned'], assigned_band_energy_dB_change=quantiles(row['energy']), envelope_squared_change_over_baseline_atom_RMS_squared=quantiles(row['envelope'])))
    write(out / 'assignment_changes.json', dict(status='completed', integer_count_unchanged=True, missing_ID_support_unchanged=True, source_decoder_unchanged=True, scope='all native positions including auxiliary inputs', rows=rows, interpretation='frozen atom attribute drift under changed assignments; held-out single-ID semantics require their separate readout audit'))
    with np.load(run / 'spatial_metrics.npz') as metrics:
        examples = metrics['examples'].copy()
        score = np.mean(metrics['nmse'], axis=(1, 2))
    selected = []
    for dataset in sorted({manifest['rows'][i]['dataset'] for i, _, _ in examples}):
        indices = [j for j, (i, _, _) in enumerate(examples) if manifest['rows'][i]['dataset'] == dataset]
        first = min(indices, key=lambda j: tuple(examples[j]))
        worst = max(indices, key=lambda j: (score[j], -j))
        selected.append(('fixed_first', first))
        if worst != first:
            selected.append(('candidate_worst', worst))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cache = OrderedDict()
    records = []
    groups = {(item['row'], item['start_s']): item for item in meta}
    for selection, j in selected:
        i, start, position = map(int, examples[j])
        row = manifest['rows'][i]
        dataset = row['dataset']
        item = groups[i, start]
        crop = item['crop']
        names = item['native_positions']
        native = next((k for k, name in enumerate(names) if name.upper() == row['positions'][position].upper()))
        batch = base_tokenizer_batch(manifest['rows'], [(i, start, position)], scales, array_cache=cache, cache_limit=8)
        scale = scales['positions']['eeg'][dataset][row['positions'][position]]['scale']
        before = source.decode_ids({branch: oldcodes[f'{branch}_{crop}'] for branch in ('S', 'O')}, dataset)
        after = source.decode_ids({branch: newcodes[f'{branch}_{crop}'] for branch in ('S', 'O')}, dataset)
        fig, axes = plt.subplots(3, 3, figsize=(15, 8), constrained_layout=True)
        time_axis = np.arange(6000) / 200
        mask = batch['mask'][0].reshape(-1)
        for r, (branch, field) in enumerate([('S', 's'), ('O', 'o'), ('sum', 'x')]):
            truth = batch[field][0].reshape(-1) * scale
            old = before[branch][0, native].reshape(-1) * scale
            new = after[branch][0, native].reshape(-1) * scale
            for c, limits in enumerate([(0, 30), (14, 16), (13.5, 14.5)]):
                ax = axes[r, c]
                for label, waveform, color in [('target', truth, 'black'), ('frozen source', old, '#2b6caa'), ('integer search', new, '#d95f02')]:
                    ax.plot(time_axis, waveform, color=color, lw=0.8, label=label)
                ax.set_xlim(*limits)
                support = (time_axis >= limits[0]) & (time_axis < limits[1])
                values = np.concatenate([a[support] for a in (truth, old, new)])
                minimum, maximum = (np.nanmin(values), np.nanmax(values))
                padding = 0.05 * (maximum - minimum + 1e-15)
                ax.set_ylim(minimum - padding, maximum + padding)
                for boundary in range(2, 30, 2):
                    ax.axvline(boundary, color='grey', lw=0.5, ls=':')
                invalid = ~mask
                edges = np.diff(np.r_[False, invalid, False].astype(int))
                for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                    ax.axvspan(a / 200, b / 200, color='grey', alpha=0.3)
                ax.set_xlabel('Window time (s)')
                ax.set_ylabel(f'{branch} ({row['source_unit']})')
                if r == 0 and c == 0:
                    ax.legend(fontsize=8)
        title = f'{dataset} / {row['subject']} / {row['record']} / {row['positions'][position]} / start {start}s / {selection}'
        fig.suptitle(title + '\nSame fixed decoder and rate; dotted token boundaries; grey unscored support', fontsize=10)
        filename = f'{dataset}_{selection}.png'
        fig.savefig(out / filename, dpi=200)
        plt.close(fig)
        records.append(dict(dataset=dataset, subject=row['subject'], record=row['record'], parent=row['parent'], position=row['positions'][position], start_s=start, manifest_row=i, position_index=position, selection=selection, candidate_XSO_NMSE=float(score[j]), figure=str(out / filename)))
    write(out / 'figures.json', records)
if __name__ == '__main__':
    main()
