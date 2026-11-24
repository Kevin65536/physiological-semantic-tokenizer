"""Audit frozen tokenizer outputs on the registered G0 joint-baseline panels."""
import argparse
from collections import OrderedDict, defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from run_dual_component_tokenizer import SOURCE, _base_forward, _base_training_model, identity
from src.data.dual_component_tokenizer import base_tokenizer_training_data
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics, joint_reconstruction_summary
from src.data.dual_component_tokenizer import base_tokenizer_batch


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    def encode(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, indent=2, default=encode) + '\n')
    temporary.replace(path)


def load_source(args):
    config = yaml.safe_load((SOURCE / 'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(config, SOURCE)
    metadata = read(args.panel.parent / 'panel.json')
    if (metadata['cfg_identity'] != manifest['cfg_identity'] or metadata['scales_identity'] != manifest['scales_identity']
            or metadata['panel_sha256'] != hashlib.sha256(args.panel.read_bytes()).hexdigest()):
        raise ValueError('frozen panel G0 identity mismatch')
    with np.load(args.panel) as arrays:
        examples = arrays[args.modality + '_' + args.role].copy()
    rows = manifest['rows']
    expected_split = 'train' if args.role == 'pattern_train' else 'validation'
    for i, _, _ in examples:
        if rows[i]['modality'] != args.modality or rows[i]['split'] != expected_split:
            raise ValueError('panel modality or split mismatch before arrays')
    return rows, scales, examples, manifest


def load_candidate(args, manifest, scales):
    if args.kind != 'legacy':
        if args.kind in ('dictionary', 'temporal_diagnostic'):
            from src.tokenizers.supervised_waveform_dictionary import load_model
        elif args.kind == 'contextual':
            from src.tokenizers.contextual_latent_dictionary import load_model
        elif args.kind == 'spectral':
            from src.tokenizers.spectral_waveform_dictionary import load_model
        elif args.kind == 'joint_decoder':
            from src.tokenizers.joint_component_decoder import load_model
        else:
            from src.tokenizers.fullband_eeg_tokenizer import load_model
        model = load_model(args.checkpoint)
        metadata = model.metadata
        if metadata['modality'] != args.modality:
            raise ValueError('candidate modality mismatch')
        candidate_input = metadata.get('input_identity', metadata.get('signature', {}).get('input'))
        if candidate_input != identity(manifest):
            raise ValueError('candidate G0 input identity mismatch')
        if args.kind not in ('dictionary', 'temporal_diagnostic') and (
                metadata['cfg_identity'] != manifest['cfg_identity'] or
                metadata['scales_identity'] != manifest['scales_identity']):
            raise ValueError('candidate G0 config or scales identity mismatch')
        if args.kind == 'temporal_diagnostic' and metadata['interface_eligible']:
            raise ValueError('short-atom diagnostic must explicitly declare its interface deviation')
        return model
    config = read(args.legacy_run / 'resolved_config.json')['config']
    task = dict(modality=args.modality, architecture='waveform', candidate='continuous' if args.continuous else 'vq128')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if (saved['signature']['config'] != identity(config) or saved['signature']['task'] != task
            or saved['signature']['input'] != identity(manifest) or saved['signature']['scales'] != identity(scales)):
        raise ValueError('legacy checkpoint config or task identity mismatch')
    model = _base_training_model(config, task)
    model.load_state_dict(saved['model_state'])
    return model.eval()


def predict(model, batch, args, ids=None):
    if args.kind != 'legacy':
        if args.continuous:
            return model.predict_continuous(batch['x']), None
        if ids is None:
            ids = model.encode(batch['x'])
        return model.decode_ids(ids), ids
    with torch.inference_mode():
        if ids is None:
            tensor = {key: torch.as_tensor(np.where(batch['mask'], value, 0.) if key in ('s','o') else value,
                                          dtype=torch.bool if key == 'mask' else torch.float32)
                      for key, value in batch.items()}
            result = _base_forward(model, tensor)
            codes = None if args.continuous else {key: result['ids'][key].numpy() for key in ('S','O')}
            if codes is not None:
                replay = model.decode_ids(result['ids'])
                for branch in ('S','O'):
                    torch.testing.assert_close(result[branch],replay[branch],rtol=1e-5,atol=2e-6)
        else:
            result = model.decode_ids({key: torch.as_tensor(value, dtype=torch.long) for key, value in ids.items()})
            codes = ids
        return {key: result[key].numpy() for key in ('sum','S','O')}, codes


def input_scales(rows, examples, scales):
    return np.array([scales['positions'][rows[i]['modality']][rows[i]['dataset']][rows[i]['positions'][p]]['scale']
                     for i, _, p in examples])


def score_pass(args, model, rows, scales, examples, saved_ids=None):
    metrics, support, codes = defaultdict(list), defaultdict(list), defaultdict(list)
    cache = OrderedDict()
    for start in range(0, len(examples), args.batch_size):
        take = examples[start:start+args.batch_size]
        batch = base_tokenizer_batch(rows, take, scales, array_cache=cache, cache_limit=32)
        ids = None if saved_ids is None else {key: value[start:start+len(take)] for key, value in saved_ids.items()}
        pred, ids = predict(model, batch, args, ids)
        audited = joint_reconstruction_metrics(pred, batch, 1 if args.modality == 'eeg' else 2,
                                                input_scales(rows, take, scales))
        for key, value in audited['metrics'].items():
            metrics[key].append(value)
        for key, value in audited['support'].items():
            support[key].append(value)
        if ids is not None:
            for key, value in ids.items():
                patches = model.metadata['atom_patches'] if args.kind == 'temporal_diagnostic' else 15
                if value.shape != (len(take),patches) or value.dtype.kind not in 'iu':
                    raise ValueError('integer sequence differs from the declared rate')
                codes[key].append(value)
        write(args.output / 'progress.json', dict(role=args.role, pass_name='integer_replay' if saved_ids is not None else 'forward',
            completed=start+len(take), planned=len(examples), updated_unix=time.time()))
    return tuple({key: np.concatenate(value) for key,value in collection.items()} for collection in (metrics,support,codes))


def display_selection(args, metrics, rows, examples):
    if args.reference_metrics is not None:
        with np.load(args.reference_metrics) as ref:
            np.testing.assert_array_equal(ref['examples'], examples)
            reference = ref['nmse'].mean((1,2))
    else:
        reference = metrics['nmse'].mean((1,2))
    current = metrics['nmse'].mean((1,2))
    selected = defaultdict(list)
    for dataset in sorted({rows[i]['dataset'] for i,_,_ in examples}):
        rng = np.random.default_rng(23017)
        group = [n for n,(i,_,_) in enumerate(examples) if rows[i]['dataset'] == dataset]
        group.sort(key=lambda n: tuple(examples[n]))
        selected[group[0]].append('fixed_identity_first')
        selected[group[int(rng.integers(len(group)))]].append('fixed_random_seed23017')
        finite = [n for n in group if np.isfinite(reference[n])]
        if finite:
            p90 = np.quantile(reference[finite],.9)
            selected[min(finite,key=lambda n: (abs(reference[n]-p90),n))].append('reference_nearest_p90')
            selected[max(finite,key=lambda n: (reference[n],-n))].append('reference_max')
        finite_current = [n for n in group if np.isfinite(current[n])]
        if finite_current:
            selected[max(finite_current,key=lambda n: (current[n],-n))].append('candidate_max')
        for n in group:
            if not np.isfinite(current[n]):
                selected[n].append('candidate_failed')
    return selected


def export_examples(args, model, metrics, rows, scales, examples, ids):
    selected = display_selection(args, metrics, rows, examples)
    indices = sorted(selected)
    take = examples[indices]
    batch = base_tokenizer_batch(rows,take,scales)
    prediction, _ = predict(model,batch,args,None if not ids else {key:value[indices] for key,value in ids.items()})
    arrays = {key: batch[key] for key in ('x','s','o','mask')}
    arrays.update({'prediction_'+key:value for key,value in prediction.items()})
    arrays.update({'ids_'+key:value[indices] for key,value in ids.items()})
    destination = args.display_output or args.output
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'waveform_examples.npz').exists():
        raise FileExistsError('retained waveform export already exists')
    np.savez_compressed(destination/'waveform_examples.npz', **arrays)
    metadata = []
    for n in indices:
        i,start,p = examples[n]
        row = rows[i]
        metadata.append(dict(modality=args.modality,dataset=row['dataset'],subject=row['subject'],record=row['record'],
            parent=row['parent'],position_name=row['positions'][p],start_s=int(start),row=int(i),position=int(p),
            scale=float(input_scales(rows, [examples[n]],scales)[0]),source_unit=row['source_unit'],
            selection=selected[n],panel_index=n,nmse=metrics['nmse'][n].tolist()))
        metadata[-1]['prediction_label'] = 'Continuous prediction' if args.continuous else 'Integer decode'
        if args.kind == 'temporal_diagnostic':
            metadata[-1].update(token_seconds=30/model.metadata['atom_patches'],
                                interface='diagnostic_outside_single_ID_2s')
    write(destination/'waveform_examples.json',dict(coordinate='G0_normalized',model=str(args.checkpoint),
        role=args.role,reference_metrics=str(args.reference_metrics),examples=metadata))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--modality',choices=['eeg','hb'],required=True)
    parser.add_argument('--role',choices=['screen','development','confirm','pattern_train'],default='screen')
    parser.add_argument('--kind',choices=['legacy','dictionary','contextual','spectral','temporal_diagnostic','fullband','joint_decoder'],required=True)
    parser.add_argument('--legacy-run',type=Path)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--continuous',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-metrics',type=Path)
    parser.add_argument('--display-output',type=Path)
    parser.add_argument('--reexport-only',action='store_true')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--threads',type=int,default=2)
    args = parser.parse_args()
    if args.kind == 'legacy' and args.legacy_run is None:
        parser.error('--legacy-run required for legacy checkpoint')
    args.output.mkdir(parents=True,exist_ok=True)
    evaluation_lock = (args.output/'evaluation.lock').open('a')
    fcntl.flock(evaluation_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.output/'summary.json').exists() and not args.reexport_only:
        raise FileExistsError('retained completed audit already exists')
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    rows,scales,examples,manifest = load_source(args)
    fingerprint = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    model = load_candidate(args,manifest,scales)
    if args.reexport_only:
        if args.display_output is None:
            parser.error('--display-output required for a versioned reexport')
        with np.load(args.output/'metrics.npz') as stored:
            np.testing.assert_array_equal(stored['examples'],examples)
            metrics = {key:stored[key].copy() for key in stored if key != 'examples' and not key.startswith('support_')}
        ids = {}
        if not args.continuous:
            with np.load(args.output/'integer_ids.npz') as stored:
                np.testing.assert_array_equal(stored['examples'],examples)
                ids = {key:stored[key].copy() for key in ('S','O')}
        export_examples(args,model,metrics,rows,scales,examples,ids)
        return
    metrics,support,ids = score_pass(args,model,rows,scales,examples)
    replay = dict(status='continuous_diagnostic')
    if ids:
        np.savez_compressed(args.output/'integer_ids.npz',examples=examples,**ids)
        with np.load(args.output/'integer_ids.npz') as stored:
            np.testing.assert_array_equal(stored['examples'],examples)
            replay_ids = {key: stored[key].copy() for key in ('S','O')}
        model = load_candidate(args,manifest,scales)
        repeated,repeated_support,_ = score_pass(args,model,rows,scales,examples,replay_ids)
        largest = 0.
        for key in metrics:
            np.testing.assert_allclose(metrics[key],repeated[key],rtol=1e-5,atol=2e-6,equal_nan=True,err_msg=key)
            finite = np.isfinite(metrics[key]) & np.isfinite(repeated[key])
            largest = max(largest,float(np.max(abs(metrics[key][finite]-repeated[key][finite]),initial=0.)))
        for key in support:
            np.testing.assert_array_equal(support[key],repeated_support[key])
        metrics,support = repeated,repeated_support
        replay = dict(status='passed',independent_checkpoint_reload=True,source='integer_ids.npz',
                      metric_max_abs_difference=largest,rtol=1e-5,atol=2e-6)
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != fingerprint:
        raise AssertionError('checkpoint changed during frozen audit')
    np.savez_compressed(args.output/'metrics.npz',examples=examples,**metrics,
                        **{'support_'+key:value for key,value in support.items()})
    summary = joint_reconstruction_summary(metrics,support,examples.tolist(),rows)
    summary.update(status='completed',checkpoint=str(args.checkpoint),checkpoint_sha256=fingerprint,
        modality=args.modality,role=args.role,integer_replay=replay,elapsed_seconds=time.monotonic()-started,
        context_seconds=30,patch_seconds=2,integer_count_per_position_30s=0 if args.continuous else 30,
        interface_eligible=args.kind != 'temporal_diagnostic' and not args.continuous,
        metric_patch_seconds=2,local_rms_block_seconds=.25)
    if args.kind == 'temporal_diagnostic':
        patches = model.metadata['atom_patches']
        summary.update(patch_seconds=30/patches,integer_count_per_position_30s=2*patches,
                       diagnostic='shorter atom duration changes the declared two-second single-ID interface')
    export_examples(args,model,metrics,rows,scales,examples,ids)
    write(args.output/'summary.json',summary)
    print(json.dumps(dict(status=summary['status'],inputs=len(examples),selection_score=summary['selection_score'],
                         integer_replay=replay,elapsed_seconds=summary['elapsed_seconds'])))


if __name__ == '__main__':
    main()
