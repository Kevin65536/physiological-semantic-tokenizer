#!/usr/bin/env python3
"""Benchmark train-only coordinate search, freeze it, and audit the fixed screen."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
from threadpoolctl import threadpool_limits
import yaml

from src.analysis.dual_component_tokenizer import joint_reconstruction_summary
from src.analysis.spatial_tokenizer import screen, write
from src.data.dual_component_tokenizer import base_tokenizer_joint_eeg, base_tokenizer_training_data
from src.tokenizers.spatial_analysis_synthesis import AnalysisSynthesisSpatialEncoder, load_encoder
from src.tokenizers.spatial_waveform_dictionary import BANDS, SpatialWaveformDictionary, load_model, spectral_features
from experiments.scripts.run_dual_component_tokenizer import identity, owner_lock


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings(cfg, option):
    result = {key: value for key, value in cfg['search'].items() if key not in ('options', 'proxy_targets', 'proxy_denominator', 'update', 'observation_support')}
    return dict(result, **option, fft_workers=cfg['resources']['fft_workers'])


def fixed_variances(model, scales):
    result = {}
    for dataset, names in model.metadata['positions'].items():
        native = scales['positions']['eeg'][dataset]
        lookup = {name.upper(): value for name, value in native.items()}
        result[dataset] = {branch: [float(lookup[name.upper()]['target_variance'][field][0])
                                     if name.upper() in lookup else 1. for name in names]
                           for branch, field in [('S', 's'), ('O', 'o'), ('sum', 'x')]}
    return result


def train_panel(rows, cfg):
    subjects = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        if row['split'] == 'train' and row['modality'] == 'eeg' and row['inputs']['score']:
            subjects[row['dataset']][row['subject']].append(index)
    result = []
    for dataset, group in sorted(subjects.items()):
        selected = sorted(group, key=lambda subject: hashlib.sha256(f'analysis-synthesis-v1/{dataset}/{subject}'.encode()).hexdigest())[:cfg['benchmark']['subjects_per_dataset']]
        for subject in selected:
            index = min(group[subject], key=lambda i: (rows[i]['record'], rows[i]['parent']))
            item = rows[index]['inputs']['score'][0]
            result.append(dict(row=index, start_s=item['start_s'], positions=item['positions']))
    return result


def synthetic():
    positions = ['p' + str(i) for i in range(5)]
    metadata = dict(schema='spatial_waveform_dictionary_v1', codebook_size=5, positions={'fixture': positions}, output_projection_hz=[1, 45])
    time_axis = np.arange(400) / 200
    transfer = {'S': np.ones(3001), 'O': np.zeros(3001)}
    atoms, centers, weights = {}, {}, {}
    amplitudes = np.asarray([0., .5, 1., 1.4, 2.])
    values = []
    for band, frequency in enumerate([2., 6., 10., 20., 35.]):
        waveform = np.sin(2 * np.pi * frequency * time_axis)
        values.append(np.tile(waveform, (15, 1)))
        for branch in ('S', 'O'):
            atoms[branch, band] = amplitudes[:, None] * waveform[None] if branch == 'S' else np.zeros((5, 400))
            centers[branch, band] = spectral_features(atoms[branch, band][1:], band)
            matrix = np.zeros((1, 5))
            matrix[0, band] = .5 if branch == 'S' else 1.
            weights['fixture', branch, band] = matrix
    source = SpatialWaveformDictionary(metadata, transfer, centers, atoms, weights)
    setting = dict(top_m=4, branches=['S', 'O'], sweeps=1, fft_workers=1, sum_weight=.5,
                   semantic_weight=.25, energy_ratio=[.5, 2.], envelope_nrmse_squared_max=.5,
                   distance_radius_factor=2., relative_band_distance_floor=.5)
    variance = {'fixture': {branch: [1.] * 5 for branch in ('S', 'O', 'sum')}}
    model = AnalysisSynthesisSpatialEncoder(source, setting, variance)
    values = np.asarray(values)[None]
    observed = np.ones((1, 5), dtype=bool)
    before = values.copy()
    codes = model.encode(values, 'fixture', observed)
    record = model.search_history[-1]
    assert record['accepted_updates'] > 0
    assert record['proxy_loss_after'] < record['proxy_loss_before']
    np.testing.assert_array_equal(values, before)
    for branch in ('S', 'O'):
        assert codes[branch].shape == (1, 5, 15) and codes[branch].dtype.kind in 'iu'
    repeated = model.encode(values, 'fixture', observed)
    for branch in ('S', 'O'):
        np.testing.assert_array_equal(codes[branch], repeated[branch])
    return dict(status='passed', actual_proxy_nonincrease=True, accepted_updates=record['accepted_updates'],
                proxy_before=record['proxy_loss_before'], proxy_after=record['proxy_loss_after'],
                integer_count_per_branch_position=15, unchanged_decoder=True, deterministic_encoder=True)


def benchmark(cfg, out, manifest, scales, source):
    rows = manifest['rows']
    panel = train_panel(rows, cfg)
    write(out/'training_benchmark_panel.json', panel)
    variances = fixed_variances(source, scales)
    options = []
    # Book FFTs and envelopes are independent of the chosen search budget.
    cache_spectra, cache_envelopes = {}, {}
    precompute_started = time.monotonic()
    warm = AnalysisSynthesisSpatialEncoder(source, settings(cfg, cfg['search']['options'][-1]), variances)
    for branch in ('S', 'O'):
        for band in range(len(BANDS)):
            warm._book(branch, band)
    cache_spectra, cache_envelopes = warm.atom_spectra, warm.atom_envelopes
    precompute_seconds = time.monotonic() - precompute_started
    for option in cfg['search']['options']:
        model = AnalysisSynthesisSpatialEncoder(source, settings(cfg, option), variances)
        model.atom_spectra, model.atom_envelopes = cache_spectra, cache_envelopes
        cache = OrderedDict()
        for crop in panel:
            row = rows[crop['row']]
            names = source.metadata['positions'][row['dataset']]
            fields, observed, _ = base_tokenizer_joint_eeg(rows, crop, scales, names, cache, allowed_splits=('train',))
            values = fields['x'].T.reshape(1, len(names), 15, 400)
            model.encode(values, row['dataset'], observed[None])
        options.append(dict(settings=model.settings, cases=model.search_history,
                            mean_seconds=float(np.mean([v['seconds'] for v in model.search_history])),
                            mean_proxy_reduction=float(np.mean([v['proxy_relative_reduction'] for v in model.search_history]))))
        write(out/'benchmark_progress.json', dict(completed=options, planned_options=len(cfg['search']['options'])))
    best = max(v['mean_proxy_reduction'] for v in options)
    eligible = [v for v in options if v['mean_proxy_reduction'] >= .9 * best]
    chosen = min(eligible, key=lambda record: record['mean_seconds'])
    panel_path = ROOT/cfg['evaluation']['panel']
    panel_metadata = json.loads(panel_path.with_name('panel.json').read_text())
    if digest(panel_path) != panel_metadata['panel_sha256']:
        raise ValueError('frozen screen panel changed')
    with np.load(panel_path) as panel_arrays:
        screen_crops = len({(int(i), int(start)) for i, start, _ in panel_arrays['eeg_screen']})
    estimated = chosen['mean_seconds'] * screen_crops + precompute_seconds
    if estimated > cfg['benchmark']['maximum_screen_seconds_estimate']:
        raise RuntimeError('measured CPU estimate exceeds frozen screen budget')
    bundle = dict(schema='spatial_analysis_synthesis_encoder_v1', source=str((ROOT/cfg['source']['decoder']).resolve()),
                  source_sha256=digest(ROOT/cfg['source']['decoder']), settings=chosen['settings'], variances=variances,
                  input_identity=manifest['cfg_identity'], scales_identity=manifest['scales_identity'],
                  encoder_inputs=['observed_X_30s', 'observation_availability', 'fixed_dataset_native_order'],
                  decoder_new_side_information=[], training_selection=cfg['benchmark']['budget_selection'])
    write(out/'encoder.json', bundle)
    write(out/'search_selection.json', dict(status='frozen_before_screen', options=options, chosen=chosen['settings'],
        expected_screen_seconds=estimated, screen_crops=screen_crops, precompute_seconds=precompute_seconds,
        calibration_inputs=len(panel), codebook_size=source.codebook_size,
        integer_count_per_branch_position=15, fixed_bits_per_branch_position_second=int(np.ceil(np.log2(source.codebook_size))) / 2,
        FFT_cache_bytes=sum(v.nbytes for v in cache_spectra.values()),
        selection='fastest option attaining 90% of largest mean observation-only proxy reduction on fixed train panel'))


def compare(out, cfg, rows):
    baseline_path = ROOT/cfg['source']['baseline_metrics']
    with np.load(baseline_path) as old, np.load(out/'spatial_metrics.npz') as new:
        np.testing.assert_array_equal(old['examples'], new['examples'])
        examples = new['examples'].copy()
        metrics = {key: new[key].copy() for key in new.files if key != 'examples' and not key.startswith('support_')}
        baseline = {key: old[key].copy() for key in metrics}
        support = {key[8:]: new[key].copy() for key in new.files if key.startswith('support_')}
    candidate_summary = joint_reconstruction_summary(metrics, support, examples.tolist(), rows)
    baseline_summary = joint_reconstruction_summary(baseline, support, examples.tolist(), rows)
    paired = {key: dict(baseline=baseline_summary['metrics'][key]['macro'],
                        candidate=candidate_summary['metrics'][key]['macro'],
                        difference=(np.asarray(candidate_summary['metrics'][key]['macro'])-np.asarray(baseline_summary['metrics'][key]['macro'])).tolist())
              for key in metrics}
    write(out/'paired_comparison.json', dict(status='completed', component_order=['X', 'S', 'O'],
        identical_examples=True, planned_examples=len(examples), metrics=paired,
        normalization='unchanged G0 per-component training variance and hierarchy',
        baseline=str(baseline_path), encoder_bundle=str(out/'encoder.json')))


def run(args, cfg):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if cfg['schema'] != 'dual_component_spatial_analysis_synthesis_v1' or cfg['interface']['new_side_information']:
        raise ValueError('versioned integer-only analysis/synthesis contract required')
    with owner_lock(out):
        if (out/'summary.json').exists():
            if json.loads((out/'resolved_config.json').read_text())['config'] != cfg:
                raise ValueError('completed analysis/synthesis configuration changed')
            load_encoder(out/'encoder.json')
            return
        resolved = dict(config=cfg, config_identity=identity(cfg))
        if (out/'resolved_config.json').exists():
            if json.loads((out/'resolved_config.json').read_text()) != resolved:
                raise ValueError('analysis/synthesis resolved configuration changed')
        else:
            write(out/'resolved_config.json', resolved)
        write(out/'synthetic.json', synthetic())
        if args.stage == 'synthetic':
            return
        source_path = ROOT/cfg['source']['decoder']
        source_hash = digest(source_path)
        source = load_model(source_path)
        manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
        if (source.codebook_size != cfg['interface']['codebook_size']
                or source.metadata['input_identity'] != manifest['cfg_identity']
                or source.metadata['scales_identity'] != manifest['scales_identity']):
            raise ValueError('frozen source and G0 data identities differ')
        write(out/'resources.json', dict(cpu_affinity=sorted(os.sched_getaffinity(0)), physical_cores=psutil.cpu_count(logical=False),
            memory_available_bytes=psutil.virtual_memory().available, numerical_threads=cfg['resources']['numerical_threads'],
            sequential_reason='accepted coordinate updates change the common decoder residual state',
            gpu_devices=[], memory_budget_gib=cfg['resources']['memory_gib'], wall_budget_seconds=cfg['resources']['wall_budget_seconds']))
        started = time.monotonic()
        if not (out/'encoder.json').exists():
            benchmark(cfg, out, manifest, scales, source)
        if args.stage == 'benchmark':
            return
        model = load_encoder(out/'encoder.json')
        summaries = screen(model, source_path, manifest['rows'], scales, cfg, out)
        write(out/'search_trace.json', model.search_history)
        compare(out, cfg, manifest['rows'])
        if digest(source_path) != source_hash:
            raise AssertionError('frozen source decoder changed during encoder search')
        write(out/'summary.json', dict(status='completed', encoder=str(out/'encoder.json'), source_decoder=str(source_path),
            source_decoder_sha256=source_hash, source_decoder_unchanged=True,
            elapsed_seconds=time.monotonic()-started, screen_nmse=summaries['spatial']['selection_score'],
            candidate_scope='EEG-only integer encoder with unchanged decoder',
            read_splits=['train','validation'], confirmation_subjects_read=0, test_arrays_read=0,
            search_calls=len(model.search_history), actual_search_seconds=sum(v['seconds'] for v in model.search_history),
            integer_count_and_rates='identical to source for every native position including auxiliaries'))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_spatial_analysis_synthesis_v1.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['synthetic','benchmark','all'], default='all')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    with threadpool_limits(cfg['resources']['numerical_threads']):
        run(args,cfg)


if __name__ == '__main__':
    main()
