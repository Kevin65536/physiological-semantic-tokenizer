#!/usr/bin/env python3
"""Matched v3 continuations of the retained G4 local-shape architecture."""
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
import torch
from threadpoolctl import threadpool_limits
import yaml

from experiments.scripts.run_dual_component_tokenizer import (
    _base_initialize, _base_training_batch, _base_training_model, _state_refinement_loss, identity, owner_lock,
)
from src.data.dual_component_tokenizer import base_tokenizer_training_data
from experiments.scripts.run_joint_waveform_dictionary import preserve_record, write_json
from src.data.dual_component_tokenizer import base_tokenizer_sample, base_tokenizer_sampling_index


def save(path, payload):
    temporary = path.with_suffix('.pending.pt')
    torch.save(payload, temporary)
    temporary.replace(path)


def source_model(cfg, candidate):
    original_cfg = yaml.safe_load((ROOT / cfg['source']['config']).read_text())
    run = ROOT / cfg['source']['run']
    original_record = json.loads((run / 'resolved_config.json').read_text())['config']
    if original_record != original_cfg:
        raise ValueError('retained G4 source configuration changed')
    folder = run / 'models' / cfg['source']['task']
    summary = json.loads((folder / 'summary.json').read_text())
    saved = torch.load(folder / 'checkpoint_step_10000.pt', map_location='cpu', weights_only=False)
    if (summary['status'] != 'completed' or summary['completed_step'] != 10000
            or saved['step'] != 10000 or saved['signature']['config'] != identity(original_cfg)
            or saved['signature']['task'] != {'modality': 'hb', 'architecture': 'local_shape', 'candidate': 'continuous'}):
        raise ValueError('completed exact G4 continuous source required')
    model = _base_training_model(cfg, dict(modality='hb', architecture='local_shape', candidate=candidate))
    loaded = model.load_state_dict({key: value for key, value in saved['model_state'].items() if '.quantizer.' not in key}, strict=False)
    if loaded.unexpected_keys or any('.quantizer.' not in key for key in loaded.missing_keys):
        raise ValueError('warm-start architecture drift')
    setting = dict(cfg['model'], **cfg['architectures']['local_shape'],
                   input_dim=40, channels=2, sample_strides=[5])
    provenance = dict(checkpoint=str(folder / 'checkpoint_step_10000.pt'),
        source_sha256=hashlib.sha256((folder / 'checkpoint_step_10000.pt').read_bytes()).hexdigest(),
        source_signature=saved['signature'],
        historical_validation_exposure=cfg['source']['historical_validation_exposure'],
        old_selected_step=summary['selected_step'], fixed_transferred_step=10000,
        optimizer='reset', normalization='new G0 v3 fixed training coordinate', source_data='v2', target_data='v3')
    return model, setting, provenance


def run(args, cfg):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest, scales, audit, _ = base_tokenizer_training_data(cfg, ROOT)
    rows = manifest['rows']
    groups = base_tokenizer_sampling_index(rows, 'hb')
    signature = dict(config=identity(cfg), input=identity(manifest), scales=identity(scales))
    preserve_record(out / 'resolved_config.json', cfg)
    if args.mode == 'train' and (out / 'summary.json').exists():
        retained = torch.load(out / 'checkpoint.pt', map_location='cpu', weights_only=False)
        if (retained['signature'] != signature or retained['step'] != cfg['training']['total_steps']
                or retained['metadata']['candidate'] != args.candidate):
            raise ValueError('completed continuation identity changed')
        return
    cache = OrderedDict()
    torch.manual_seed(cfg['training']['seed'])
    torch.set_num_threads(cfg['resources']['torch_threads'])
    model, setting, provenance = source_model(cfg, args.candidate)
    preserve_record(out / 'source_provenance_v2.json', provenance)
    if model.hard:
        _base_initialize(model, rows, audit['hb']['initialization'], scales, 'cpu', cache, cfg)
    examples = base_tokenizer_sample(np.random.default_rng(8031), groups, 64)
    monitor = _base_training_batch(rows, examples, scales, 'cpu', cache, cfg)
    if args.mode == 'benchmark':
        results = []
        for threads in cfg['resources']['benchmark_threads']:
            torch.set_num_threads(threads)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            began = time.monotonic()
            for _ in range(cfg['resources']['benchmark_steps']):
                model.train()
                loss, _ = _state_refinement_loss(model(monitor['x']), monitor, cfg['architectures']['local_shape'])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            elapsed = time.monotonic() - began
            results.append(dict(threads=threads, seconds=elapsed,
                steps_per_second=cfg['resources']['benchmark_steps'] / elapsed))
        write_json(out / 'benchmark.json', dict(status='completed', results=results, candidate=args.candidate))
        return
    training = cfg['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'], weight_decay=training['weight_decay'])
    rng = np.random.default_rng(training['seed'])
    checkpoint = out / 'checkpoint.pt'
    start, history = 0, []
    if checkpoint.exists():
        retained = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if (retained['signature'] != signature or retained['metadata']['candidate'] != args.candidate
                or retained['metadata'].get('source_sha256', provenance['source_sha256']) != provenance['source_sha256']):
            raise ValueError('retained continuation identity changed')
        model.load_state_dict(retained['model_state'])
        optimizer.load_state_dict(retained['optimizer_state'])
        start, history = retained['step'], retained['history']
        rng.bit_generator.state = retained['numpy_rng']
        torch.set_rng_state(retained['torch_rng'])
    metadata = dict(schema='contextual_latent_dictionary_v1', modality='hb', codebook_size=1024,
        cfg_identity=manifest['cfg_identity'], scales_identity=manifest['scales_identity'], signature=signature,
        candidate=args.candidate, input='X_only', context_seconds=30, patch_seconds=2, quantization_levels=1,
        integers_per_branch_position_context=15, bits_per_branch_position_per_second=5,
        source_historical_validation_exposure=cfg['source']['historical_validation_exposure'],
        architecture='G4_local_shape', seed=training['seed'], source_sha256=provenance['source_sha256'])
    began = time.monotonic()
    for step in range(start, training['total_steps']):
        model.train()
        take = base_tokenizer_sample(rng, groups, training['batch_size'])
        batch = _base_training_batch(rows, take, scales, 'cpu', cache, cfg)
        phase = (step + 1) / training['total_steps']
        rate = training['final_learning_rate'] + (training['learning_rate'] - training['final_learning_rate']) * (1 + np.cos(np.pi * phase)) / 2
        for group in optimizer.param_groups:
            group['lr'] = float(rate)
        model.set_decoder_quantization_blend(min(1., (step + 1) / training['quantization_transition_steps']) if model.hard else 1.)
        loss, terms = _state_refinement_loss(model(batch['x']), batch, cfg['architectures']['local_shape'])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'])
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):
            raise FloatingPointError('nonfinite continuation')
        optimizer.step()
        if (step + 1) % training['checkpoint_every'] == 0 or step == start:
            model.eval()
            with torch.inference_mode():
                prediction = model(monitor['x'])
                monitored, values = _state_refinement_loss(prediction, monitor, cfg['architectures']['local_shape'])
                replay_error = None
                if model.hard:
                    replay = model.decode_ids(prediction['ids'])
                    replay_error = max(float((prediction[key] - replay[key]).abs().max()) for key in ('S', 'O'))
                    if replay_error != 0:
                        raise AssertionError('frozen integer replay differs')
            history.append(dict(step=step + 1, loss=float(loss.detach()), monitor_loss=float(monitored),
                monitor={key: float(value) for key, value in values.items()}, replay_max_abs=replay_error,
                elapsed_seconds=time.monotonic() - began))
            payload = dict(signature=signature, metadata=metadata, model_config=setting, model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(), step=step + 1, history=history,
                numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state())
            save(checkpoint, payload)
            if step + 1 in (500, 2000, 4000):
                save(out / ('checkpoint_step_' + str(step + 1).zfill(5) + '.pt'), payload)
            write_json(out / 'progress.json', dict(status='training', candidate=args.candidate, completed_step=step + 1,
                budget=training['total_steps'], history=history, parameter_count=sum(p.numel() for p in model.parameters())))
            print(json.dumps(history[-1]), flush=True)
    write_json(out / 'summary.json', dict(status='trained_pending_independent_evaluation', candidate=args.candidate,
        step=training['total_steps'], seconds=time.monotonic() - began, history=history,
        parameter_count=sum(p.numel() for p in model.parameters()), validation_arrays_read=0, test_arrays_read=0))


def calibrate_synthesis(args, cfg):
    from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics, joint_reconstruction_summary
    from src.tokenizers.contextual_latent_dictionary import load_model
    from src.data.dual_component_tokenizer import base_tokenizer_batch
    from scipy.signal import periodogram

    synthesis = yaml.safe_load((ROOT / args.synthesis_config).read_text())
    if (synthesis['schema'] != 'dual_component_joint_hb_synthesis_v1'
            or synthesis['synthesis']['decoder_lowpass_hz'] != .4
            or synthesis['calibration']['validation_cutoff_selection'] != 'forbidden'):
        raise ValueError('fixed train-diagnostic Hb synthesis contract required')
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    source = ROOT / synthesis['source_run'] / args.candidate / ('checkpoint_step_' + str(synthesis['source_step']).zfill(5) + '.pt')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    signature = dict(source_sha256=digest, source=str(source), synthesis_config=identity(synthesis),
                     input=identity(manifest), scales=identity(scales), candidate=args.candidate)
    preserve_record(out / 'resolved_synthesis.json', dict(config=synthesis, signature=signature))
    if (out / 'summary.json').exists():
        return
    frozen = torch.load(source, map_location='cpu', weights_only=False)
    if frozen['signature']['input'] != identity(manifest) or frozen['signature']['scales'] != identity(scales):
        raise ValueError('matched G0 frozen source required')
    frozen['metadata'] = dict(frozen['metadata'], decoder_lowpass_hz=.4,
        synthesis_source_sha256=digest, synthesis_config_identity=identity(synthesis),
        decoder_filter='Butterworth4_sosfiltfilt_default_odd_padding_10Hz')
    destination = out / 'checkpoint.pt'
    save(destination, frozen)
    before, after = load_model(source), load_model(destination)
    rows = manifest['rows']
    groups = base_tokenizer_sampling_index(rows, 'hb')
    examples = base_tokenizer_sample(np.random.default_rng(8031), groups, 64)
    batch = base_tokenizer_batch(rows, examples, scales)
    scale_values = np.asarray([scales['positions']['hb'][rows[i]['dataset']][rows[i]['positions'][position]]['scale']
                               for i, _, position in examples])
    predictions = {}
    if args.candidate == 'continuous':
        predictions = dict(before=before.predict_continuous(batch['x']), after=after.predict_continuous(batch['x']))
        replay = dict(status='continuous_diagnostic')
    else:
        ids = before.encode(batch['x'])
        for branch in ('S', 'O'):
            np.testing.assert_array_equal(ids[branch], after.encode(batch['x'])[branch])
        predictions = dict(before=before.decode_ids(ids), after=after.decode_ids(ids))
        repeated = load_model(destination).decode_ids(ids)
        for branch in ('S', 'O', 'sum'):
            np.testing.assert_array_equal(predictions['after'][branch], repeated[branch])
        np.savez_compressed(out / 'integer_ids.npz', **ids, examples=examples)
        replay = dict(status='passed', independent_reload=True, maximum_absolute_difference=0.)
    summaries = {}
    for name, prediction in predictions.items():
        audited = joint_reconstruction_metrics(prediction, batch, 2, scale_values)
        summaries[name] = joint_reconstruction_summary(audited['metrics'], audited['support'], examples, rows)
    complete = batch['mask'].all(axis=(1, 2))
    energy = {}
    for branch in ('s', 'o'):
        frequency, power = periodogram(batch[branch][complete].reshape(-1, 300, 2), fs=10, window='hann', axis=1)
        fraction = power[:, frequency > .4].sum(1) / np.maximum(power.sum(1), 1e-30)
        energy[branch] = dict(complete_training_windows=int(complete.sum()),
            mean_fraction_above_0p4Hz_by_color=fraction.mean(0).tolist())
    write_json(out / 'summary.json', dict(status='completed_train_diagnostic_pending_screen', signature=signature,
        checkpoint=str(destination), examples=examples, integer_replay=replay,
        input_target_spectral_support=energy, scores=summaries, validation_arrays_read=0))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='experiments/configs/physiology_semantic_tokenizer/dual_component_joint_hb_shape_v1.yaml')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--candidate', choices=['continuous', 'vq1024'], required=True)
    parser.add_argument('--mode', choices=['train', 'benchmark', 'synthesis'], default='train')
    parser.add_argument('--synthesis-config', default='experiments/configs/physiology_semantic_tokenizer/dual_component_joint_hb_synthesis_v1.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    if (cfg['schema'] != 'dual_component_joint_hb_shape_v1' or cfg['model']['patches'] != 15
            or cfg['model']['codebook_size'] != 1024 or cfg['source']['step'] != 10000
            or not cfg['architectures']['local_shape']['hb_local_shape']
            or cfg['training']['total_steps'] < 1):
        raise ValueError('versioned 30s/2s Hb continuation contract required')
    with owner_lock(args.output), threadpool_limits(limits=cfg['resources']['numerical_threads']):
        torch.set_num_threads(cfg['resources']['torch_threads'])
        if args.mode == 'synthesis':
            calibrate_synthesis(args, cfg)
        else:
            run(args, cfg)


if __name__ == '__main__':
    main()
