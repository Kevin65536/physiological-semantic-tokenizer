"""Frozen integer readout and paired reconstruction endpoints for EEG-S loss placement."""
from collections import OrderedDict, defaultdict

import numpy as np
import torch

import run_single_vq_spectral_location as suite
from run_dual_component_tokenizer import _base_training_model, identity, read_json, write_json
from src.analysis.dual_component_tokenizer import base_training_aggregate, base_training_metrics, eeg_spectral_attributes
from src.analysis.frozen_token_readout import Conditional, Means
from src.data.dual_component_tokenizer import base_tokenizer_batch, eeg_spectrum_coordinates
from src.utils.io import save_npz_atomic


def supported_ids(ids, people, size, limits):
    counts = np.bincount(ids, minlength=size)
    pairs = np.unique(people.astype(np.int64)*size + ids)
    n_people = np.bincount(pairs % size, minlength=size)
    return (counts >= limits['min_train_patches']) & (n_people >= limits['min_train_people'])


def extract(model, examples, ds, role, rows, scales, calibration, sites, cases, dest):
    device = next(model.parameters()).device
    parts = defaultdict(list)
    cache = OrderedDict()
    case_outputs = {}
    for i in range(0, len(examples), 64):
        ex = examples[i:i+64]
        raw = base_tokenizer_batch(rows, ex, scales, array_cache=cache)
        coords = eeg_spectrum_coordinates(rows, ex, scales, calibration)
        valid = raw['mask'].all(-1)
        scale, mean, sd = (torch.as_tensor(coords[k], dtype=torch.float32, device=device)
                           for k in ('input_scale', 'feature_mean', 'feature_sd'))
        def attributes(wave):
            return ((eeg_spectral_attributes(wave, scale) - mean) / sd).cpu().numpy()
        x = torch.as_tensor(raw['x'], device=device, dtype=torch.float32)
        result = model(x)
        ids = result['ids']['S'].cpu().numpy()
        if ids.shape != valid.shape or ids.min() < 0 or ids.max() >= suite.K:
            raise ValueError('invalid emitted S integers')
        values = attributes(torch.as_tensor(raw['s'], device=device, dtype=torch.float32))
        if not np.isfinite(values[valid]).all():
            raise ValueError('nonfinite complete target attributes')
        location = np.array([sites[ds, rows[r]['positions'][p]] for r, _, p in ex])
        bg = location[:, None]*4 + np.digitize(np.sqrt((raw['x']**2).mean(-1)), [.5, 1, 2])
        parts['id'].append(ids[valid].astype(np.uint16))
        parts['bg'].append(bg[valid].astype(np.int32))
        parts['target'].append(values[valid])
        parts['valid'].append(valid)
        if role == 'validation':
            pred = model.decode_ids(result['ids'])
            for b in ('S', 'O', 'sum'):
                torch.testing.assert_close(pred[b], result[b], rtol=0, atol=0)
            decoded = attributes(pred['S'])
            arrays = {b: pred[b].cpu().numpy() for b in ('S','O','sum')}
            if any(not np.isfinite(a).all() for a in arrays.values()):
                raise ValueError('nonfinite decoded waveform')
            parts['decoded'].append(decoded[valid])
            metrics = base_training_metrics(arrays, raw, 1)
            for name in ('nmse', 'correlation'):
                parts[name].append(metrics[name])
            for j, (r, start, pos) in enumerate(ex):
                key = int(r), int(start), rows[r]['positions'][pos]
                if key in cases:
                    name = cases[key]
                    case_outputs[name] = dict(example=ex[j].tolist(), dataset=ds, subject=rows[r]['subject'],
                        record=rows[r]['record'], position=key[-1], nmse=metrics['nmse'][j,:,0].tolist())
                    save_npz_atomic(dest/f'{name}_waveforms.npz', example=ex[j], mask=raw['mask'][j],
                        input_scale=coords['input_scale'][j], **{b:raw[b][j] for b in ('x', 's', 'o')},
                        **{'reconstructed_'+b:a[j] for b,a in arrays.items()},
                        **{'ids_'+b:v[j].cpu().numpy() for b,v in result['ids'].items()})
    if case_outputs:
        write_json(dest/f'{ds}_fixed_windows.json', case_outputs)
    save_npz_atomic(dest/f'{ds}_{role}.npz', examples=examples, **{k:np.concatenate(v) for k,v in parts.items()})


def join(dest, role, rows, people, panel):
    data = defaultdict(list)
    for ds in suite.capacity.DATASETS:
        with np.load(dest/f'{ds}_{role}.npz') as z:
            np.testing.assert_array_equal(z['examples'], panel['examples']['eeg', role, ds])
            for k in z.files:
                data[k].append(z[k])
            person = np.array([people[rows[r]['dataset'], rows[r]['subject']] for r,_,_ in z['examples']])
            data['person'].append(np.broadcast_to(person[:,None], z['valid'].shape)[z['valid']])
    return {k: np.concatenate(v) for k,v in data.items()}


def patch_means(values, valid):
    array = np.zeros((*valid.shape, values.shape[-1]), dtype=np.float64)
    array[valid] = values
    n = valid.sum(1)[:, None]
    return np.divide(array.sum(1), n, out=np.full((len(n), values.shape[-1]), np.nan), where=n>0)


@torch.inference_mode()
def score(seed, arm):
    dest = suite.folder(seed, arm)/'evaluation'
    dest.mkdir(exist_ok=True)
    suite.capacity.prepare_panel(recorded_root=suite.PLAN['recorded_root'])
    panel = suite.capacity.PANEL
    rows, scales = panel['rows'], panel['scales']
    cfg = suite.training_cfg(seed)
    state = torch.load(dest.parent/'checkpoint.pt', map_location='cpu', weights_only=False)
    training = read_json(dest.parent/'summary.json')
    if (training['status'] != 'completed' or state['signature'] != training['signature']
            or state['step'] != suite.PLAN['training']['total_steps'] or state['signature']['task'] != suite.TASK
            or state['signature']['config'] != identity(cfg)):
        raise ValueError('terminal matched checkpoint required')
    calibration = read_json(suite.OUT/'calibration.json')
    if arm != 'A' and state['signature'].get('spectral_objective') != identity(dict(
            calibration=calibration, location=suite.PLAN['arms'][arm], weight=suite.PLAN['weight'])):
        raise ValueError('spectral objective mismatch')
    model = _base_training_model(cfg, suite.TASK, single_id_spectrum=arm=='C').eval().to('cuda:0')
    model.load_state_dict(state['model_state'])
    sites = {key:i for i,key in enumerate(sorted({(r['dataset'],p) for r in rows
                             if r['modality']=='eeg' for p in r['positions']}))}
    people = {key:i for i,key in enumerate(sorted({(r['dataset'],r['subject']) for r in rows}))}
    fixed = read_json(suite.capacity.OUT/'review_20261124_v1/tables/wave_eeg.json')
    cases = {(c['row'], c['start_s'], c['position']):name for c,name in zip(fixed, ['median','difficult'], strict=True)}
    for ds in suite.capacity.DATASETS:
        for role in ('train','validation'):
            if not (dest/f'{ds}_{role}.npz').exists():
                extract(model, panel['examples']['eeg',role,ds], ds, role, rows, scales, calibration, sites, cases, dest)
    if not all((dest/f'{name}_waveforms.npz').is_file() for name in cases.values()):
        raise ValueError('fixed waveform panel incomplete')
    tr, dev = (join(dest, split, rows, people, panel) for split in ('train','validation'))
    if len(set(tr['person'])) != 60 or len(set(dev['person'])) != 16 or set(tr['person']) & set(dev['person']):
        raise ValueError('training and scoring populations changed')
    # Cast before multiplying categorical keys to avoid uint16 wraparound.
    tid, vid = tr['id'].astype(np.int64), dev['id'].astype(np.int64)
    readout = Conditional(tr['bg'].astype(np.int64), tid, tr['person'], tr['target'], suite.K, balanced=True)
    bg, delta, count = readout.read(dev['bg'].astype(np.int64), vid)
    joint = readout.shrink(bg, delta, count, 4.)
    local = Means(tid, tr['target']).lookup(vid)[0]
    supported = supported_ids(tid, tr['person'], suite.K, suite.PLAN['support'])
    stats = {k:dev[k] for k in ('nmse','correlation')}
    for name, pred in [('background',bg), ('conditional_id',joint), ('id_only',local), ('decoded',dev['decoded'])]:
        stats[name] = patch_means((pred-dev['target'])**2, dev['valid'])
    stats['supported_patch_fraction'] = patch_means(supported[vid,None], dev['valid'])
    result = base_training_aggregate(stats, dev['examples'], rows)
    if len(dev['examples']) != suite.PLAN['scope']['development_eeg_positions'] or result['subjects'] != 16:
        raise ValueError('changed development denominator')
    m = result['metrics']
    gain = 1 - np.asarray(m['conditional_id']['macro'])/np.asarray(m['background']['macro'])
    if not np.isfinite(gain).all() or not np.isfinite(m['nmse']['macro']).all():
        raise ValueError('incomplete primary endpoint')
    groups = Means(vid, dev['target'])
    dev_mean = groups.lookup(vid)[0]
    within = Means(vid, (dev['target']-dev_mean)**2)
    save_npz_atomic(dest/'id_dispersion.npz', ids=groups.keys, counts=groups.count, means=groups.values,
                    within_variance=within.values, supported_on_train=supported[groups.keys])
    result.update(id_gain=gain.tolist(), mean_id_gain=float(gain.mean()), train_people=60,
        train_complete_patches=len(tid), development_complete_patches=len(vid),
        supported_ids=int(supported.sum()), supported_fraction_vocabulary=float(supported.mean()),
        supported_development_tokens=int(supported[vid].sum()), emitted_development_ids=len(groups.keys),
        within_id_variance=np.average(within.values, weights=groups.count, axis=0).tolist(),
        between_id_variance=np.average((groups.values-groups.global_mean)**2, weights=groups.count, axis=0).tolist(),
        gain_order=suite.PLAN['features'], status='completed')
    write_json(dest/'summary.json', result)


def paired(left, right):
    def people(result, name):
        return {(a['dataset'],a['subject']):np.asarray(a['value']) for a in result['metrics'][name]['by_subject']}
    def gains(result):
        bg = people(result,'background')
        return {k:1-v/bg[k] for k,v in people(result,'conditional_id').items()}
    a, b = gains(left), gains(right)
    an, bn = people(left,'nmse'), people(right,'nmse')
    if a.keys() != b.keys() or an.keys() != bn.keys() or len(a) != 16:
        raise ValueError('paired development subjects required')
    return dict(mean_gain_delta=right['mean_id_gain']-left['mean_id_gain'],
                macro_nmse_delta=right['selection_score']-left['selection_score'],
                subjects=[dict(dataset=k[0], subject=k[1], gain_delta=(b[k]-a[k]).tolist(),
                               nmse_delta=(bn[k]-an[k]).ravel().tolist()) for k in sorted(a)])


def compare(root, seed, arms, plan):
    summaries = {a:read_json(root/f'seed_{seed}'/a/'evaluation/summary.json') for a in arms}
    ref = summaries['A']
    limits = plan['guardrails']
    def macro(summary, metric):
        return np.asarray(summary['metrics'][metric]['macro'], dtype=float)
    checks, eligible = {}, []
    for arm in arms[1:]:
        s = summaries[arm]
        tests = dict(positive=s['mean_id_gain']>0, improved=s['mean_id_gain']>ref['mean_id_gain'],
            macro=s['selection_score']<=ref['selection_score']*(1+limits['macro_nmse_relative_increase_max']),
            components=bool((macro(s,'nmse') <= macro(ref,'nmse')*(1+limits['component_nmse_relative_increase_max'])).all()),
            correlations=bool((macro(s,'correlation') >= macro(ref,'correlation')-limits['component_correlation_drop_max']).all()))
        checks[arm] = tests
        if all(tests.values()):
            eligible.append(arm)
    return dict(seed=seed, selected=min(eligible, key=lambda a:(summaries[a]['selection_score'],a), default=None),
                checks=checks, results={a:{k:s[k] for k in ('mean_id_gain','selection_score','id_gain',
                'supported_fraction_vocabulary','supported_development_tokens')} for a,s in summaries.items()},
                contrasts={b+'-minus-'+a:paired(summaries[a],summaries[b]) for i,a in enumerate(arms) for b in arms[i+1:]})


def register_completion(out):
    """Update only the launched study after both its screening and conditional repeat are terminal."""
    from run_dual_component_tokenizer import now
    from src.utils.project_state import validate_registry, write_outputs
    registry_path = suite.ROOT/'research_state/registry.json'
    registry = read_json(registry_path)
    entity = 'main.tk_single_vq_spectral_location_v1'
    old = [r for r in registry['records'] if r['entity'] == entity][-1]
    if old['execution'] == 'completed':
        return
    result = read_json(out/'summary.json')
    if result['status'] != 'completed' or old['execution'] != 'running':
        raise ValueError('complete experiment and running registry entry required')
    timestamp = now()
    eid = 'e.tk_single_vq_spectral_location_results'
    registry['evidence'].append(dict(id=eid, label='频谱约束位置的配对筛查与复现决定',
        path=str((out/'summary.json').relative_to(suite.ROOT)), role='result_summary'))
    chosen = result['first_seed']['selected']
    decision = f'{chosen} 与 A 已用种子 43 完成配对复核。' if chosen else '两种放置方式均未满足补种子门槛。'
    registry['records'].append(dict(old, state_id=entity+'@2026-11-25.development', execution='completed',
        supersedes=[old['state_id']], scientific_verdict='inconclusive', updated_at=timestamp,
        evidence_ids=old['evidence_ids']+[eid], summary='四库普通开发面板三个条件均已评分。'+decision,
        next_step='复核逐人配对、覆盖率与固定困难波形，再解释单 ID 属性和重建的取舍。'))
    registry['updated_at'] = timestamp
    validate_registry(registry, repo_root=suite.ROOT)
    write_json(registry_path, registry)
    write_outputs(registry, repo_root=suite.ROOT, status_path=suite.ROOT/'docs/PROJECT_STATUS.md',
                  readme_path=suite.ROOT/'README.md')
