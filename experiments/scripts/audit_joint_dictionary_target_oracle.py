#!/usr/bin/env python3
"""Exact target-input oracle for the frozen two-second waveform dictionaries."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from evaluate_joint_tokenizer import input_scales, load_candidate, load_source
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics, joint_reconstruction_summary
from src.data.dual_component_tokenizer import base_tokenizer_batch
from src.tokenizers.supervised_waveform_dictionary import load_model, nearest_masked_waveform_ids


def write(path, value):
    def serial(a):
        if isinstance(a, dict):
            return {str(k): serial(v) for k, v in a.items()}
        if isinstance(a, (list, tuple, np.ndarray)):
            return [serial(v) for v in a]
        if isinstance(a, np.generic):
            return serial(a.item())
        if isinstance(a, float) and not np.isfinite(a):
            return None
        return a
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(serial(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    pending.replace(path)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def concatenate(parts):
    return {key: np.concatenate(value) for key, value in parts.items()}


def audit_one(payload):
    run, output, modality, size = payload
    folder = output/modality/f'k{size}'
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = run/'dictionary'/modality/f'k{size}.npz'
    panel = run/'panels.npz'
    args = argparse.Namespace(panel=panel, modality=modality, role='screen',
                              kind='dictionary', checkpoint=checkpoint)
    identity = dict(checkpoint=str(checkpoint), checkpoint_sha256=fingerprint(checkpoint),
                    panel=str(panel), panel_sha256=fingerprint(panel), modality=modality,
                    role='screen', codebook_size=size, input='supervision_target_S_and_O',
                    selector='exhaustive_full_waveform_valid_point_MSE_float64_direct_differences',
                    semantics='oracle for this particular frozen dictionary, using privileged supervised targets')
    completed = folder/'summary.json'
    if completed.exists():
        result = json.loads(completed.read_text())
        if result['oracle_identity'] != identity:
            raise ValueError('retained completed oracle identity changed')
        return result
    started = time.monotonic()
    try:
        rows, scales, examples, manifest = load_source(args)
        confirmation = json.loads((run/'panel.json').read_text())['confirmation_subjects']
        if any(rows[i]['subject'] == confirmation[rows[i]['dataset']] for i, _, _ in examples):
            raise ValueError('confirmation subject entered screening oracle')
        model = load_candidate(args, manifest, scales)
        if model.atom_patches != 15 or model.atom_dim != model.input_dim:
            raise ValueError('exactly one full two-second atom per branch required')
        metrics, support, codes, selected_mse, counts = [defaultdict(list) for _ in range(5)]
        mismatch = {branch: dict(scored=0, full=0, scored_total=0, full_total=0) for branch in ('S', 'O')}
        cache = OrderedDict()
        for start in range(0, len(examples), 32):
            take = examples[start:start+32]
            batch = base_tokenizer_batch(rows, take, scales, array_cache=cache, cache_limit=32)
            valid = batch['mask'].reshape(-1, model.input_dim)
            ids = {}
            for branch in ('S', 'O'):
                target = batch[branch.lower()].reshape(valid.shape)
                chosen, mse, valid_count = nearest_masked_waveform_ids(
                    target, valid, model.atoms[branch], target_block_size=64, atom_block_size=128)
                ids[branch] = chosen.reshape(len(take), 15)
                selected_mse[branch].append(mse.reshape(len(take), 15))
                counts[branch].append(valid_count.reshape(len(take), 15))
                projected = model.nearest_ids(batch[branch.lower()], branch).reshape(-1)
                for label, use in (('scored', valid_count > 0), ('full', valid_count == model.input_dim)):
                    mismatch[branch][label] += int(np.sum((chosen != projected) & use))
                    mismatch[branch][label+'_total'] += int(use.sum())
                squared_error = (model.atoms[branch][chosen]-target)**2
                direct = np.where(valid, squared_error, 0.).sum(1)
                direct = np.divide(direct, valid_count, out=np.full(len(target), np.nan), where=valid_count > 0)
                np.testing.assert_allclose(mse, direct, rtol=1e-13, atol=1e-14, equal_nan=True)
            prediction = model.decode_ids(ids)
            audited = joint_reconstruction_metrics(prediction, batch, model.channels, input_scales(rows, take, scales))
            for key, value in audited['metrics'].items():
                metrics[key].append(value)
            for key, value in audited['support'].items():
                support[key].append(value)
            for key, value in ids.items():
                codes[key].append(value)
            write(folder/'progress.json', dict(status='running', completed_inputs=start+len(take),
                   planned_inputs=len(examples), elapsed_seconds=time.monotonic()-started))
        metrics, support, codes = [concatenate(parts) for parts in (metrics, support, codes)]
        np.savez_compressed(folder/'integer_ids.npz', **codes)
        np.savez_compressed(folder/'selection_audit.npz', examples=examples,
                            **{'minimum_valid_mse_'+key: value for key, value in concatenate(selected_mse).items()},
                            **{'valid_counts_'+key: value for key, value in concatenate(counts).items()})
        del model
        restored = load_model(checkpoint)
        with np.load(folder/'integer_ids.npz') as archive:
            replay_ids = {key: archive[key].copy() for key in ('S', 'O')}
        repeated_metrics, repeated_support = defaultdict(list), defaultdict(list)
        for start in range(0, len(examples), 32):
            take = examples[start:start+32]
            batch = base_tokenizer_batch(rows, take, scales, array_cache=cache, cache_limit=32)
            prediction = restored.decode_ids({key: value[start:start+len(take)] for key, value in replay_ids.items()})
            audited = joint_reconstruction_metrics(prediction, batch, restored.channels, input_scales(rows, take, scales))
            for key, value in audited['metrics'].items():
                repeated_metrics[key].append(value)
            for key, value in audited['support'].items():
                repeated_support[key].append(value)
        for key, value in concatenate(repeated_metrics).items():
            np.testing.assert_allclose(value, metrics[key], rtol=0, atol=0, equal_nan=True, err_msg=key)
        for key, value in concatenate(repeated_support).items():
            np.testing.assert_array_equal(value, support[key], err_msg=key)
        if fingerprint(checkpoint) != identity['checkpoint_sha256'] or fingerprint(panel) != identity['panel_sha256']:
            raise ValueError('frozen dictionary or panel changed during oracle audit')
        np.savez_compressed(folder/'metrics.npz', examples=examples, **metrics,
                            **{'support_'+key: value for key, value in support.items()})
        summary = joint_reconstruction_summary(metrics, support, examples.tolist(), rows)
        summary.update(status='completed', oracle_identity=identity,
            integer_replay=dict(status='passed', independent_reload=True, metric_max_abs_difference=0.),
            projection_assignment_mismatch=mismatch, elapsed_seconds=time.monotonic()-started,
            limitation='Each branch ID minimizes scored pointwise MSE within its own fixed atom set. The pair of independently selected IDs need not minimize X=S+O or spectral/boundary distortion.',
            interface_eligible=False, scientific_role='target_input_diagnostic')
        write(completed, summary)
        write(folder/'progress.json', dict(status='completed', completed_inputs=len(examples),
                                           planned_inputs=len(examples), elapsed_seconds=summary['elapsed_seconds']))
        return summary
    except Exception:
        write(folder/f'failure_{time.time_ns()}.json', dict(status='failed', oracle_identity=identity,
              traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-started))
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    args.run = args.run.resolve(); args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if not 1 <= args.workers <= 6:
        raise ValueError('one to six bounded independent dictionary jobs required')
    lock = (args.output/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = [(args.run, args.output, modality, size) for modality in ('eeg', 'hb') for size in (128, 1024, 8192)]
    write(args.output/'plan.json', dict(schema='joint_dictionary_target_oracle_v1', role='screen',
        tasks=[dict(modality=m, codebook_size=k) for _, _, m, k in tasks],
        integer_selector='minimum valid-point full-waveform MSE over every frozen atom per S/O patch',
        empty_patches='keep ID0 and zero valid-point count; no scoring support is created',
        partial_patches='retain all scored points and all planned windows',
        source_input='privileged supervision targets', interface_eligible=False,
        resources=dict(workers=args.workers, numerical_threads_per_worker=1, memory_block_mib=25),
        command=sys.argv, working_directory=str(Path.cwd())))
    shutil.copy2(__file__, args.output/'source_execution.py')
    (args.output/'selector_source.py').write_text('import numpy as np\n\n'+inspect.getsource(nearest_masked_waveform_ids))
    (args.output/'metric_source.py').write_text(inspect.getsource(joint_reconstruction_metrics)+'\n\n'+inspect.getsource(joint_reconstruction_summary))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(audit_one, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            name = result['oracle_identity']
            print(json.dumps(dict(modality=name['modality'], k=name['codebook_size'],
                                  selection_score=result['selection_score'], elapsed_seconds=result['elapsed_seconds'])), flush=True)
    write(args.output/'summary.json', dict(status='completed', models=results, tasks=len(tasks),
         role='screen_target_input_oracle', interface_eligible=False))


if __name__ == '__main__':
    main()
