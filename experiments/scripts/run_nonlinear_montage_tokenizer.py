"""Matched local/full-array nonlinear EEG experiment, with train-subject holdout."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments/scripts'))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import yaml

from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_joint_eeg, base_tokenizer_batch
from src.tokenizers.spatial_waveform_dictionary import load_model as load_source
from src.tokenizers.spectral_waveform_dictionary import raw_features
from src.tokenizers.fullband_eeg_tokenizer import fullband_loss
from src.tokenizers.nonlinear_montage_tokenizer import NonlinearMontageTokenizer
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics, joint_reconstruction_summary
from experiments.scripts.evaluate_joint_tokenizer import input_scales, write


def rank(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.pending.pt')
    torch.save(value, temporary)
    temporary.replace(path)


def validate(cfg):
    if (cfg['schema'] != 'nonlinear_montage_experiment_v1' or cfg['data']['fitting_splits'] != ['train']
            or cfg['data']['evaluation_splits'] != ['train'] or cfg['data']['test_access'] != 'forbidden'
            or cfg['data']['protected_data'] != 'forbidden' or cfg['model']['unencoded_decoder_inputs']
            or cfg['model']['sample_rate_hz'] != 200 or cfg['model']['context_s'] != 30
            or cfg['model']['patch_s'] != 2 or cfg['model']['fixed_frequency_roles']
            or cfg['model']['vocabulary'] not in (128, 256, 512)):
        raise ValueError('frozen native EEG single-integer training-only contract required')


def prepare(cfg, out):
    if (out/'initial.pt').exists():
        retained = torch.load(out/'initial.pt', weights_only=False, map_location='cpu')
        if retained['config_identity'] != rank(cfg):
            raise ValueError('frozen preparation differs')
        return
    g0 = yaml.safe_load((ROOT/cfg['source']['g0_config']).read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(g0, ROOT)
    rows = manifest['rows']
    source = load_source(ROOT/cfg['source']['waveform_dictionary'])
    if (source.metadata['input_identity'] != manifest['cfg_identity']
            or source.metadata['scales_identity'] != manifest['scales_identity']):
        raise ValueError('frozen Wiener data identity mismatch before arrays')
    plan = json.loads((ROOT/cfg['source']['native_train_plan']).read_text())['plan']
    held = {d: sorted(p, key=lambda s: rank([d, s, cfg['training']['seed']]))[:cfg['data']['holdout_subjects_per_dataset']]
            for d, p in sorted(plan.items())}
    cases = []
    for dataset, people in sorted(plan.items()):
        for subject, crops in sorted(people.items()):
            for crop in crops:
                row = rows[crop['row']]
                if row['dataset'] != dataset or row['subject'] != subject or row['split'] != 'train':
                    raise ValueError('train crop/subject identity before arrays')
                cases.append(dict(dataset=dataset, subject=subject, crop=crop, held=subject in held[dataset]))
    if len(cases) != 400 or sum(len(p) for p in plan.values()) != 60:
        raise ValueError('fixed 400 complete native crops / 60 train subjects required')
    write(out/'resolved_config.json', cfg)
    panel = dict(held=held, cases=cases, rule=cfg['data']['holdout_rule'],
        new_fits='52 train subjects only', historical_exposure=cfg['data']['historical_exposure'],
        manifest_identity=rank_manifest(manifest), scales_identity=rank_manifest(scales))
    if (out/'panel.json').exists() and json.loads((out/'panel.json').read_text()) != panel:
        raise ValueError('frozen cache identities or holdout panel changed')
    write(out/'panel.json', panel)
    metadata = dict(schema='nonlinear_montage_tokenizer_v1', positions=source.metadata['positions'],
        input_identity=manifest['cfg_identity'], manifest_identity=rank_manifest(manifest),
        scales_identity=manifest['scales_identity'], input='X_only',
        context_seconds=30, patches=15, reconstruction_coordinate='G0_fixed_train_position_input_SD')
    cache = OrderedDict()
    (out/'cache').mkdir(exist_ok=True)
    grouped = defaultdict(list)
    for i, case in enumerate(cases):
        if time.process_time() > cfg['resources']['preparation_CPU_budget_seconds']:
            raise RuntimeError('preparation CPU budget reached')
        dataset, crop = case['dataset'], case['crop']
        names = metadata['positions'][dataset]
        lookup = {n.upper(): j for j, n in enumerate(names)}
        expected_examples = [(crop['row'], crop['start_s'], p) for p in crop['positions']]
        if (out/'cache'/f'{i:04d}.pt').exists():
            item = torch.load(out/'cache'/f'{i:04d}.pt', weights_only=False)
            if item['x'].shape != (len(names), 15, 400) or not np.array_equal(item['examples'], expected_examples):
                raise ValueError('resumed training cache identity mismatch')
            if not case['held']: grouped[dataset, case['subject']].append(i)
            continue
        fields, observed, _ = base_tokenizer_joint_eeg(rows, dict(crop, positions=[]), scales, names, cache)
        ex = [(crop['row'], crop['start_s'], p) for p in crop['positions']]
        target = base_tokenizer_batch(rows, ex, scales, array_cache=cache, cache_limit=16)
        native = np.asarray([lookup[rows[crop['row']]['positions'][p].upper()] for p in crop['positions']], np.int64)
        item = dict(x=fields['x'].T.reshape(len(names), 15, 400).astype('f4'), observed=observed,
            native=native, examples=np.asarray(ex, np.int32), scales=input_scales(rows, ex, scales))
        for k in ('s', 'o', 'mask', 'vx', 'vs', 'vo'):
            item[k] = np.zeros((len(names), 15, 400), dtype=bool if k == 'mask' else 'f4')
            item[k][native] = target[k]
        # Preserve the original per-sample target scoring mask, including partial patches.
        item['score_x'] = target['x'].astype('f4')
        complete = target['mask'].all(-1)
        for branch in 'so':
            labels = np.zeros((len(names), 15, 3), 'f4')
            if complete.any():
                features = raw_features(target[branch][complete], None, None, 'theta_alpha_rms')
                scored = np.zeros((*complete.shape, 3), 'f4')
                scored[complete] = features
                labels[native] = scored
            item['attr_'+branch] = labels
        save(out/'cache'/f'{i:04d}.pt', item)
        if not case['held']:
            grouped[dataset, case['subject']].append(i)
        if i % 20 == 0:
            write(out/'progress.json', dict(stage='prepare', crops=i+1, CPU_seconds=time.process_time()))
    rng = np.random.default_rng(cfg['training']['seed'])
    selected = []
    for person, ids in sorted(grouped.items()):
        choices = []
        for i in ids:
            item = torch.load(out/'cache'/f'{i:04d}.pt', weights_only=False)
            site, patch = np.nonzero(item['mask'].all(-1) & item['observed'][:, None])
            choices.extend((i, int(n), int(t)) for n, t in zip(site, patch))
        selected.extend(choices[j] for j in rng.choice(len(choices), min(len(choices), cfg['training']['pca_patches_per_fit_subject']), replace=False))
    inputs, targets, attributes = [], {b: [] for b in 'SO'}, {b: [] for b in 'SO'}
    by_case = defaultdict(list)
    for i, site, patch in selected:
        by_case[i].append((site, patch))
    wiener = {b: [] for b in 'SO'}
    for i, pairs in sorted(by_case.items()):
        item = torch.load(out/'cache'/f'{i:04d}.pt', weights_only=False)
        n, t = np.asarray(pairs).T
        unique, inverse = np.unique(n, return_inverse=True)
        pred = source.predict_continuous(item['x'][None, unique])
        for b in 'SO':
            wiener[b].append(pred[b][0, inverse, t])
            targets[b].append(item[b.lower()][n, t])
            attributes[b].append(item['attr_'+b.lower()][n, t])
    init = dict(transfer=np.stack([source.transfer[b].reshape(3001) for b in 'SO']))
    band_floor, rms_floor, mean_attr, scale_attr = [], [], [], []
    frequency = np.fft.rfftfreq(400, 1/200)
    for b in 'SO':
        values = np.concatenate(wiener[b]).astype('f8')
        center = values.mean(0)
        eig, basis = np.linalg.eigh(np.cov(values-center, rowvar=False))
        basis = basis[:, -cfg['model']['latent_dim']:][:, ::-1].T.copy()
        std = np.sqrt(np.maximum(eig[-cfg['model']['latent_dim']:][::-1], 1e-12))
        scale = np.maximum(std, np.median(std)*.1)**cfg['model']['latent_scale_exponent']
        init.update({b+'_basis': basis, b+'_mean': center, b+'_scale': scale})
        y = np.concatenate(targets[b])
        power = abs(np.fft.rfft((y-y.mean(-1, keepdims=True))*np.hanning(400)))**2
        band_floor.append([max(float(np.median(power[:, (frequency >= lo) & (frequency < hi)].sum(-1))), 1e-6)
            for lo, hi in ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45.001))])
        rms_floor.append(max(float(np.median(np.sqrt(np.mean(y*y, -1)))), 1e-4))
        a = np.concatenate(attributes[b])
        mean_attr.append(a.mean(0)); scale_attr.append(np.maximum(a.std(0), .1))
    init.update(band_floor=np.asarray(band_floor), rms_floor=np.asarray(rms_floor))
    torch.manual_seed(cfg['training']['seed'])
    model_config = dict(cfg['model'], cross_position=False)
    model = NonlinearMontageTokenizer(model_config, metadata, init)
    model.attribute_mean.copy_(torch.as_tensor(np.asarray(mean_attr)))
    model.attribute_scale.copy_(torch.as_tensor(np.asarray(scale_attr)))
    save(out/'initial.pt', dict(model_config=model_config, metadata=metadata, model_state=model.state_dict(),
        config_identity=rank(cfg), seed=cfg['training']['seed'], step=0))
    write(out/'preparation.json', dict(status='completed', crops=len(cases), fit_subjects=len(grouped),
        held_subjects=sum(map(len, held.values())), PCA_patches_per_branch=len(selected), scoring_position_windows=sum(len(c['crop']['positions']) for c in cases),
        parameters=sum(p.numel() for p in model.parameters()), CPU_seconds=time.process_time(),
        numerical_threads=torch.get_num_threads(), max_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def rank_manifest(value):
    from experiments.scripts.run_dual_component_tokenizer import identity
    return identity(value)


def load_initial(out, arm):
    saved = torch.load(out/'initial.pt', map_location='cpu', weights_only=False)
    config = dict(saved['model_config'], cross_position=arm == 'field')
    model = NonlinearMontageTokenizer(config, saved['metadata'])
    model.load_state_dict(saved['model_state'])
    return model, saved


def caches(out):
    panel = json.loads((out/'panel.json').read_text())
    return panel, [torch.load(out/'cache'/f'{i:04d}.pt', map_location='cpu', weights_only=False) for i in range(len(panel['cases']))]


def batch_for(items):
    keys = ('x', 's', 'o', 'mask', 'vx', 'vs', 'vo', 'observed', 'attr_s', 'attr_o')
    return {k: torch.as_tensor(np.stack([item[k] for item in items])) for k in keys}


def loss_for(model, pred, batch, cfg):
    flat = {k: v.flatten(0, 1) for k, v in batch.items() if k in ('x', 's', 'o', 'mask', 'vx', 'vs', 'vo')}
    prediction = {k: pred[k].flatten(0, 1) for k in ('sum', 'S', 'O')}
    prediction['commitment'] = pred['commitment']
    flat.update({k: torch.where(flat['mask'], v, torch.ones_like(v)) for k, v in flat.items() if k in ('vx', 'vs', 'vo')})
    loss, terms = fullband_loss(model.local, prediction, flat, cfg['training']['loss_weights'])
    complete = batch['mask'].all(-1) & batch['observed'][..., None]
    auxiliary = []
    for n, b in enumerate('SO'):
        target = (batch['attr_'+b.lower()]-model.attribute_mean[n])/model.attribute_scale[n]
        error = (pred['attributes'][b]-target).square().mean(-1)
        auxiliary.append(torch.where(complete, error, 0.).sum()/complete.sum().clamp_min(1))
    attr = sum(auxiliary)/2
    terms['attribute'] = float(attr.detach())
    terms['waveform_spectral_total'] = terms['total']
    loss = loss+cfg['training']['loss_weights']['attribute']*attr
    terms['total'] = float(loss.detach())
    return loss, terms


def benchmark(cfg, out):
    panel, items = caches(out)
    sizes = {c['dataset']: len(items[i]['observed']) for i, c in enumerate(panel['cases']) if not c['held']}
    dataset = max(sizes, key=sizes.get)
    ids = [i for i, c in enumerate(panel['cases']) if c['dataset'] == dataset and not c['held']][:cfg['training']['batch_crops']]
    batch = batch_for([items[i] for i in ids])
    result = []
    for threads in cfg['resources']['benchmark_threads']:
        torch.set_num_threads(threads)
        with threadpool_limits(threads):
            model, _ = load_initial(out, 'field')
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'])
            started, cpu = time.monotonic(), time.process_time()
            for _ in range(cfg['resources']['benchmark_steps']):
                pred = model(batch['x'], dataset, batch['observed'])
                loss, _ = loss_for(model, pred, batch, cfg)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            result.append(dict(threads=threads, seconds_per_step=(time.monotonic()-started)/cfg['resources']['benchmark_steps'],
                CPU_seconds_per_step=(time.process_time()-cpu)/cfg['resources']['benchmark_steps']))
    write(out/'benchmark.json', dict(timings=result, batch_crops=len(ids), dataset=dataset,
        selected_threads=min(result, key=lambda r: r['seconds_per_step'])['threads'], CPU_seconds=time.process_time()))


def schedule(panel, cfg):
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, case in enumerate(panel['cases']):
        if not case['held']:
            tree[case['dataset']][case['subject']][case['crop']['row']].append(i)
    rng = np.random.default_rng(cfg['training']['seed'])
    result = []
    for _ in range(cfg['training']['continuous_steps']):
        dataset = rng.choice(sorted(tree))
        take = []
        for _ in range(cfg['training']['batch_crops']):
            subject = rng.choice(sorted(tree[dataset])); row = rng.choice(sorted(tree[dataset][subject]))
            take.append(int(rng.choice(tree[dataset][subject][row])))
        result.append((str(dataset), take))
    return result


def train(cfg, out, arm, threads):
    panel, items = caches(out)
    model, initial = load_initial(out, arm)
    job = out/arm
    job.mkdir(exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    order = schedule(panel, cfg)
    write(job/'schedule.json', dict(steps=order, identity=rank(order)))
    history, start = [], 0
    if (job/'checkpoint.pt').exists():
        previous = torch.load(job/'checkpoint.pt', map_location='cpu', weights_only=False)
        if previous['config_identity'] != rank(cfg):
            raise ValueError('continuation config identity mismatch')
        model.load_state_dict(previous['model_state']); optimizer.load_state_dict(previous['optimizer'])
        start, history = previous['step'], previous['history']
    model.train()
    started = time.monotonic()
    previous_CPU = history[-1]['CPU_seconds'] if history else 0.
    CPU_start = time.process_time()
    status = 'completed'
    for step in range(start, cfg['training']['continuous_steps']):
        used = previous_CPU+time.process_time()-CPU_start
        if used > cfg['resources']['training_CPU_budget_seconds_each']:
            status = 'resource_budget_reached'; break
        dataset, take = order[step]
        batch = batch_for([items[i] for i in take])
        lr = cfg['training']['final_learning_rate'] + .5*(cfg['training']['learning_rate']-cfg['training']['final_learning_rate'])*(1+math.cos(math.pi*step/len(order)))
        for group in optimizer.param_groups: group['lr'] = lr
        pred = model(batch['x'], dataset, batch['observed'])
        loss, terms = loss_for(model, pred, batch, cfg)
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss retained without filtering')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip']); optimizer.step()
        history.append(dict(step=step+1, loss=float(loss.detach()), learning_rate=lr, CPU_seconds=previous_CPU+time.process_time()-CPU_start, **terms))
        if (step+1) % cfg['training']['checkpoint_every'] == 0 or step+1 == len(order):
            save(job/'checkpoint.pt', dict(model_config=model.config, metadata=model.metadata, model_state=model.state_dict(), optimizer=optimizer.state_dict(),
                step=step+1, history=history, config_identity=rank(cfg)))
            write(job/'progress.json', dict(stage='continuous_training', arm=arm, step=step+1, planned=len(order), CPU_seconds=history[-1]['CPU_seconds'],
                elapsed_seconds=time.monotonic()-started, last=history[-1]))
    final_step = history[-1]['step'] if history else start
    save(job/'checkpoint.pt', dict(model_config=model.config, metadata=model.metadata, model_state=model.state_dict(), optimizer=optimizer.state_dict(),
        step=final_step, history=history, config_identity=rank(cfg)))
    write(job/'training.json', dict(status=status, step=final_step, planned=len(order), CPU_seconds=history[-1]['CPU_seconds'] if history else 0,
        parameters=sum(p.numel() for p in model.parameters()), numerical_threads=threads, max_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        batch_schedule_identity=rank(order), continuous_no_discrete_rate=True))


def evaluate(cfg, out):
    g0 = yaml.safe_load((ROOT/cfg['source']['g0_config']).read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(g0, ROOT)
    rows = manifest['rows']
    panel, items = caches(out)
    summaries, all_metrics, examples = {}, {}, []
    held = [i for i, c in enumerate(panel['cases']) if c['held']]
    for i in held: examples.extend(items[i]['examples'].tolist())
    scoring_cache = OrderedDict()
    for arm in ('local', 'field'):
        saved = torch.load(out/arm/'checkpoint.pt', map_location='cpu', weights_only=False)
        model = NonlinearMontageTokenizer(saved['model_config'], saved['metadata']); model.load_state_dict(saved['model_state']); model.eval()
        metrics, support = defaultdict(list), defaultdict(list)
        for i in held:
            item, case = items[i], panel['cases'][i]
            batch = batch_for([item])
            with torch.inference_mode(): pred = model(batch['x'], case['dataset'], batch['observed'])
            native = item['native']
            prediction = {k: pred[k][0, native].numpy() for k in ('sum', 'S', 'O')}
            scoring = base_tokenizer_batch(rows, item['examples'], scales, array_cache=scoring_cache, cache_limit=16)
            if (not np.array_equal(scoring['mask'], item['mask'][native])
                    or not np.array_equal(scoring['x'].astype('f4'), item['score_x'])):
                raise ValueError('training cache and original scoring support/coordinate differ')
            audited = joint_reconstruction_metrics(prediction, scoring, 1, item['scales'])
            for k, v in audited['metrics'].items(): metrics[k].append(v)
            for k, v in audited['support'].items(): support[k].append(v)
        metrics = {k: np.concatenate(v) for k, v in metrics.items()}; support = {k: np.concatenate(v) for k, v in support.items()}
        np.savez_compressed(out/arm/'held_metrics.npz', **metrics, **{'support_'+k: v for k, v in support.items()}, examples=np.asarray(examples))
        summaries[arm] = joint_reconstruction_summary(metrics, support, examples, rows)
        all_metrics[arm] = metrics
        write(out/arm/'held_summary.json', summaries[arm])
    local, field = [np.asarray(summaries[a]['metrics']['nmse']['macro']).reshape(3) for a in ('local', 'field')]
    gate = cfg['evaluation']['advancement']
    improvement = 1-field.mean()/local.mean(); s_improvement = 1-field[1]/local[1]
    datasets = summaries['local']['metrics']['nmse']['by_dataset']
    n_datasets = sum(float(np.asarray(summaries['field']['metrics']['nmse']['by_dataset'][d]).reshape(3)[1]) < float(np.asarray(datasets[d]).reshape(3)[1]) for d in datasets)
    subjects = summaries['local']['metrics']['nmse']['by_subject']
    # by_subject is a list of named records in the central owning aggregator.
    if isinstance(subjects, list):
        by = lambda arm: {(v['dataset'], v['subject']): v['value'] for v in summaries[arm]['metrics']['nmse']['by_subject']}
    else:
        by = lambda arm: summaries[arm]['metrics']['nmse']['by_subject']
    la, fi = by('local'), by('field')
    n_subjects = sum(float(np.asarray(fi[s]).reshape(3)[1]) < float(np.asarray(la[s]).reshape(3)[1]) for s in la)
    guards = dict(equal_complete_steps=all(json.loads((out/a/'training.json').read_text())['step'] == cfg['training']['continuous_steps'] for a in ('local', 'field')),
        macro=improvement >= gate['held_macro_NMSE_relative_improvement_min'], S=s_improvement >= gate['held_S_NMSE_relative_improvement_min'],
        datasets=n_datasets >= gate['required_improving_datasets_S'], subjects=n_subjects/len(la) >= gate['minimum_improving_subject_fraction_S'],
        components=bool(np.isfinite(field).all() and np.all(field <= gate['maximum_any_component_NMSE_ratio_to_local']*local)))
    write(out/'summary.json', dict(status='completed', continuous_advance=all(guards.values()), guards=guards,
        local_NMSE_X_S_O=local, field_NMSE_X_S_O=field, macro_relative_improvement=improvement, S_relative_improvement=s_improvement,
        improving_datasets_S=n_datasets, improving_subjects_S=n_subjects, held_subjects=len(la), held_crops=len(held),
        held_position_windows=len(examples), data_scope='original_train_split_fixed_subject_holdout',
        validation_arrays_read=0, hard_training_started=False, maximum_candidate_ID_vocabulary=cfg['model']['vocabulary'],
        declared_hard_rate='15 integers/30s/branch/native-position; 8 bits per integer with K256 including missing0; 8 bit/s for EEG S+O'))


def render(cfg, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g0 = yaml.safe_load((ROOT/cfg['source']['g0_config']).read_text())
    manifest, _, _, _ = base_tokenizer_training_data(g0, ROOT)
    rows = manifest['rows']
    panel, items = caches(out)
    models = {}
    for arm in ('local', 'field'):
        saved = torch.load(out/arm/'checkpoint.pt', map_location='cpu', weights_only=False)
        models[arm] = NonlinearMontageTokenizer(saved['model_config'], saved['metadata'])
        models[arm].load_state_dict(saved['model_state']); models[arm].eval()
    figures = out/'waveforms'; figures.mkdir(exist_ok=True)
    records = []
    for dataset in sorted(panel['held']):
        subject = sorted(panel['held'][dataset])[0]
        candidates = [i for i, c in enumerate(panel['cases']) if c['dataset'] == dataset and c['subject'] == subject]
        index = min(candidates, key=lambda i: rank(panel['cases'][i]['crop']))
        case, item = panel['cases'][index], items[index]
        batch = batch_for([item]); predictions = {}
        with torch.inference_mode():
            for arm, model in models.items():
                predictions[arm] = {k: v[0].numpy() for k, v in model(batch['x'], dataset, batch['observed']).items() if k in ('sum', 'S', 'O')}
        errors = []
        for site in item['native']:
            valid = item['mask'][site]
            errors.append(np.mean(((predictions['local']['S'][site]-item['s'][site])[valid]**2)/item['vs'][site][valid]))
        for kind, pos in [('median', int(np.argsort(errors)[len(errors)//2])), ('hard', int(np.argmax(errors)))]:
            site = item['native'][pos]; scale = item['scales'][pos]
            valid = item['mask'][site].reshape(6000)
            unit = rows[case['crop']['row']].get('source_unit', 'native unit')
            fig, axes = plt.subplots(3, 3, figsize=(17, 8.3))
            for col, (target, prediction) in enumerate((('x', 'sum'), ('s', 'S'), ('o', 'O'))):
                y = item['score_x'][pos] if target == 'x' else item[target][site]
                y = y.reshape(6000)*scale
                p = {arm: predictions[arm][prediction][site].reshape(6000)*scale for arm in models}
                time_grid = np.arange(6000)/200
                for r, (start, end) in enumerate(((0, 30), (14, 16), (13.5, 14.5))):
                    ax = axes[r, col]; take = (time_grid >= start) & (time_grid < end)
                    ax.plot(time_grid[take], y[take], color='black', lw=.65, label='target')
                    ax.plot(time_grid[take], p['local'][take], color='#dc8439', lw=.65, alpha=.85, label='local')
                    ax.plot(time_grid[take], p['field'][take], color='#2075ba', lw=.65, alpha=.85, label='field')
                    for boundary in np.arange(0, 31, 2):
                        if start <= boundary <= end: ax.axvline(boundary, color='gray', lw=.5, alpha=.45)
                    ax.fill_between(time_grid[take], 0, 1, where=~valid[take], transform=ax.get_xaxis_transform(), color='gray', alpha=.25)
                    ax.set_xlim(start, end); ax.set_ylabel(unit); ax.set_xlabel('time (s)'); ax.set_title(prediction)
                    if r == 0: ax.legend(fontsize=8)
            name = models['local'].metadata['positions'][dataset][site]
            fig.suptitle(f'{dataset} / {subject} / row {case["crop"]["row"]} +{case["crop"]["start_s"]}s / {name} / {kind}; target and both models share axes; gray = unscored')
            fig.tight_layout(); path = figures/f'{dataset}_{kind}.png'; fig.savefig(path, dpi=180); plt.close(fig)
            records.append(dict(path=str(path.relative_to(ROOT)), dataset=dataset, subject=subject, crop=case['crop'],
                native_site=name, kind=kind, local_S_NMSE=errors[pos], valid_points=int(valid.sum()), status='rendered_awaiting_actual_visual_review'))
    write(figures/'manifest.json', dict(selection=cfg['evaluation']['visual_selection'], figures=records))


def diagnose(cfg, out):
    """Freeze both networks; remove variable pooled context on fit-only crops."""
    panel = json.loads((out/'panel.json').read_text())
    picked = {d: min((i for i, c in enumerate(panel['cases']) if c['dataset'] == d and not c['held']),
                     key=lambda i: rank(panel['cases'][i]['crop'])) for d in sorted(panel['held'])}
    result = {}
    for arm in ('local', 'field'):
        saved = torch.load(out/arm/'checkpoint.pt', map_location='cpu', weights_only=False)
        model = NonlinearMontageTokenizer(saved['model_config'], saved['metadata']); model.load_state_dict(saved['model_state']); model.eval()
        measurements = []
        for dataset, i in picked.items():
            item = torch.load(out/'cache'/f'{i:04d}.pt', weights_only=False)
            batch = batch_for([item])
            with torch.inference_mode():
                reference = model(batch['x'], dataset, batch['observed'])
                # Keep each learned pool bias; remove only its variable input.
                handles = [m.pool.register_forward_hook(lambda module, args, output: module.bias.expand_as(output))
                           for group in (model.encoder_message, model.decoder_message) for m in group.values()]
                try: ablated = model(batch['x'], dataset, batch['observed'])
                finally:
                    for handle in handles: handle.remove()
                valid = batch['mask']
                change = {b: float(torch.sqrt(torch.mean((reference[b]-ablated[b])[valid].square())/torch.mean(reference[b][valid].square()).clamp_min(1e-12))) for b in ('sum', 'S', 'O')}
            measurements.append(dict(dataset=dataset, subject=panel['cases'][i]['subject'], crop=panel['cases'][i]['crop'],
                pooled_context_ablation_RMS_change_fraction_of_prediction=change))
        result[arm] = measurements
    write(out/'fit_context_diagnostic.json', dict(frozen_no_refit=True, data_scope='one_hash_fixed_fit_subject_crop_per_dataset',
        held_or_validation_arrays_read=0, intervention='replace_encoder_and_decoder_variable_pool_outputs_by_their_unchanged_learned_bias',
        interpretation='output_sensitivity_diagnostic_only_not_a_new_held_candidate_or_gate',
        auxiliary_target='raw_log_local_theta_alpha_mean_quarter_second_RMS_standardized_on_52_fit_subjects; background_not_residualized', measurements=result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', 'benchmark', 'train', 'evaluate', 'render', 'diagnose'), required=True)
    parser.add_argument('--arm', choices=('local', 'field'))
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(); cfg = yaml.safe_load(args.config.read_text()); validate(cfg)
    if not 1 <= args.threads <= 6: raise ValueError('per-process numerical thread bound')
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    out = ROOT/cfg['output']; out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(args.threads):
        if args.mode == 'prepare': prepare(cfg, out)
        elif args.mode == 'benchmark': benchmark(cfg, out)
        elif args.mode == 'train':
            if args.arm is None: raise ValueError('matched arm required')
            train(cfg, out, args.arm, args.threads)
        elif args.mode == 'evaluate': evaluate(cfg, out)
        elif args.mode == 'render': render(cfg, out)
        else: diagnose(cfg, out)


if __name__ == '__main__':
    main()
