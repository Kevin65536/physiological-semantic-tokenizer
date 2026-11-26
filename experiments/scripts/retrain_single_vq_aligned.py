"""Refit the two selected v3 single-level tokenizers with 128-dimensional codebooks."""
import argparse
from collections import OrderedDict

import numpy as np
import torch
import yaml

import run_single_vq_capacity as capacity
import run_single_vq_spectral_location as spectral
from run_dual_component_tokenizer import (
    _base_training_model, identity, now, owner_lock, read_json, train_base_tokenizer, write_json,
)
from score_single_vq_spectral_location import score as score_eeg
from src.analysis.dual_component_tokenizer import base_training_aggregate, base_training_metrics
from src.data.dual_component_tokenizer import base_tokenizer_batch

CONTRACT = capacity.CONTRACT.with_name('single_vq_aligned_v1.yaml')
PLAN = yaml.safe_load(CONTRACT.read_text())
ROOT = capacity.ROOT
OUT = ROOT / PLAN['run']


def settings(mod):
    seed, dim, size = PLAN['seed'], PLAN['dimension'], PLAN[mod]['size']
    cfg = capacity.resolved(mod, dim, seed)
    cfg['experiment_id'] = PLAN['experiment_id']
    if (dim != 128 or seed != 17 or any(cfg['training'][k] != v for k, v in PLAN['budget'].items())
            or PLAN['scope'][mod+'_positions'] != capacity.PLAN['support'][mod+'_positions']
            or PLAN['scope']['test'] != 'forbidden' or PLAN['scope']['confirmation'] != 'forbidden'):
        raise ValueError('aligned refit contract drift')
    cache_contract = yaml.safe_load((ROOT / cfg['data']['contract']).read_text())
    if cache_contract['data']['source_schema'] != PLAN['source']:
        raise ValueError('unexpected teacher source')
    if mod == 'eeg':
        results = read_json(spectral.OUT/'summary.json')
        arm = PLAN['eeg']['arm']
        if (results['status'] != 'completed' or size != spectral.K or arm != 'C'
                or any(results[s]['selected'] != arm for s in ('first_seed', 'repeat_seed'))):
            raise ValueError('EEG configuration differs from replicated selection')
    else:
        picked = read_json(capacity.OUT/'selection.json')['replication']
        if [mod, dim, size, capacity.PLAN['seeds'][1]] not in picked or PLAN['hb']['arm'] != 'waveform':
            raise ValueError('Hb configuration differs from replicated selection')
    return cfg, capacity.model_task(mod, size)


def reference(mod):
    if mod == 'eeg':
        return spectral.folder(PLAN['seed'], 'C')/'evaluation/summary.json'
    name = capacity.candidate(mod, PLAN['dimension'], PLAN[mod]['size'], PLAN['seed'])
    return capacity.OUT/'evaluation'/name/'hb_summary.json'


def objective():
    calibration = read_json(spectral.OUT/'calibration.json')
    obj = dict(location=spectral.PLAN['arms']['C'], weight=spectral.PLAN['weight'], calibration=calibration)
    parent = read_json(reference('eeg').parent.parent/'summary.json')
    if parent['signature']['spectral_objective'] != identity(obj):
        raise ValueError('calibration differs from selected C')
    return obj


@torch.inference_mode()
def score_hb(cfg, task, job, state):
    model = _base_training_model(cfg, task).eval().to('cuda:0')
    model.load_state_dict(state['model_state'])
    capacity.prepare_panel(recorded_root=spectral.PLAN['recorded_root'])
    panel = capacity.PANEL
    examples, scores = [], []
    cache = OrderedDict()
    for ds in capacity.DATASETS:
        rows = panel['examples']['hb', 'validation', ds]
        with np.load(reference('hb').parent/f'hb_{ds}_validation.npz') as previous:
            np.testing.assert_array_equal(previous['examples'], rows)
        examples.append(rows)
        for start in range(0, len(rows), 64):
            raw = base_tokenizer_batch(panel['rows'], rows[start:start+64], panel['scales'], array_cache=cache)
            result = model(torch.as_tensor(raw['x'], dtype=torch.float32, device='cuda:0'))
            decoded = model.decode_ids(result['ids'])
            for key in ('S', 'O', 'sum'):
                torch.testing.assert_close(decoded[key], result[key], atol=0, rtol=0)
            preds = {key: decoded[key].cpu().numpy() for key in ('S', 'O', 'sum')}
            scores.append(base_training_metrics(preds, raw, 2)['nmse'])
    scores = np.concatenate(scores)
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite development NMSE')
    result = base_training_aggregate(dict(nmse=scores), np.concatenate(examples), panel['rows'])
    write_json(job/'evaluation/summary.json', dict(result, status='completed'))


def refit(mod):
    cfg, task = settings(mod)
    base = OUT/mod
    job = base/'models'/'__'.join(task.values())
    with owner_lock(base):
        args = dict(recorded_root=spectral.PLAN['recorded_root'])
        obj = objective() if mod == 'eeg' else None
        write_json(base/'resolved_config.json', dict(contract=PLAN, training=cfg, task=task, spectral_objective=obj))
        train_base_tokenizer(cfg, ROOT, base, dict(task, candidate='warmup'), 'cuda:0', **args)
        fit = train_base_tokenizer(cfg, ROOT, base, task, 'cuda:0', spectral_objective=obj, **args)
        state = torch.load(job/'checkpoint.pt', map_location='cpu', weights_only=False)
        if (state['step'] != PLAN['budget']['total_steps'] or state['signature'] != fit['signature']
                or fit['status'] != 'completed' or state['signature']['config'] != identity(cfg)):
            raise ValueError('terminal checkpoint mismatch')
        shapes = {b: list(state['model_state'][f'branches.{b}.quantizer.codebook'].shape) for b in 'SO'}
        if any(shape != [PLAN[mod]['size'], PLAN['dimension']] for shape in shapes.values()):
            raise ValueError('terminal codebook shape mismatch')
        if mod == 'eeg':
            score_eeg(PLAN['seed'], 'C', job=job, cfg=cfg, calibration_path=spectral.OUT/'calibration.json')
        else:
            score_hb(cfg, task, job, state)
        score = read_json(job/'evaluation/summary.json')
        parent = read_json(reference(mod))
        if (score['subjects'] != PLAN['scope']['development_people']
                or score['inputs'] != PLAN['scope'][mod+'_positions'] or score['inputs'] != parent['inputs']):
            raise ValueError('development denominator drift')
        result = dict(status='completed', step=state['step'], shapes=shapes, subjects=score['subjects'],
                      positions=score['inputs'], integer_replay='exact', macro_nmse=score['selection_score'],
                      baseline_macro_nmse=parent['selection_score'], component_nmse=score['metrics']['nmse']['macro'],
                      component_order=score['component_order'], checkpoint=str((job/'checkpoint.pt').relative_to(ROOT)),
                      baseline=str(reference(mod).relative_to(ROOT)))
        if mod == 'eeg':
            result.update(mean_id_gain=score['mean_id_gain'], baseline_mean_id_gain=parent['mean_id_gain'])
        write_json(base/'result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pilot', 'train', 'summarize'])
    parser.add_argument('--modality', choices=['eeg', 'hb'])
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == 'summarize':
        results = {m: read_json(OUT/m/'result.json') for m in ('eeg', 'hb')}
        write_json(OUT/'summary.json', dict(status='completed', contract=PLAN, results=results, finished=now()))
        return
    mod = args.modality
    if mod is None:
        parser.error('--modality is required for pilot and train')
    if args.action == 'pilot':
        cfg, task = settings(mod)
        result = spectral.synthetic('C' if mod == 'eeg' else 'A', 'cuda:0', 24, cfg['training']['batch_size'],
                                    modality=mod, dimension=PLAN['dimension'], size=PLAN[mod]['size'])
        write_json(OUT/mod/'pilot.json', dict(result, task=task, dimension=PLAN['dimension']))
    else:
        refit(mod)


if __name__ == '__main__':
    main()
