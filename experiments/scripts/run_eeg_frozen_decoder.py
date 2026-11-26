"""Matched decoder continuations from the aligned EEG model on frozen integer IDs."""
import argparse
import copy
import os
from collections import OrderedDict, defaultdict
import time

import numpy as np
import torch
import yaml

import audit_aligned_tokenizer as audit
import retrain_single_vq_aligned as aligned
from run_dual_component_tokenizer import (
    _base_training_batch, _base_training_model, _state_refinement_loss, _torch_save,
    identity, now, owner_lock, read_json, write_json,
)
from run_single_vq_dynamic_factorial import check_gpu
from score_single_vq_spectral_location import extract, patch_means
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.data.dual_component_tokenizer import base_tokenizer_sample, base_tokenizer_sampling_index

CONTRACT = aligned.CONTRACT.with_name('eeg_frozen_decoder_v1.yaml')
PLAN = yaml.safe_load(CONTRACT.read_text())
OUT = aligned.ROOT / PLAN['run']
CAP = aligned.capacity
BUDGET = PLAN['training']
ARMS = tuple(PLAN['arms'])


def check():
    iface = PLAN['interface']
    scope = PLAN['scope']
    parent_scope = dict(aligned.spectral.PLAN['scope'])
    parent_scope['development_positions'] = parent_scope.pop('development_eeg_positions')
    if PLAN['parent_contract'] != aligned.CONTRACT.name or scope != parent_scope:
        raise ValueError('development-only parent or population drift')
    if iface != dict(modality='eeg', patch_samples=400, patches=15, codebook_size=8192,
                     dimension=128, width=256, parent_step=10000, parent_seed=17):
        raise ValueError('unexpected frozen parent interface')
    if PLAN['arms'] != dict(continued=2, expanded=4) or BUDGET['steps'] % BUDGET['checkpoint_every']:
        raise ValueError('unexpected decoder comparison or monitoring schedule')


def update(model, batch, spec, optimizer):
    optimizer.zero_grad(set_to_none=True)
    result = model.forward_frozen_ids(batch['x'])
    loss, _ = _state_refinement_loss(result, batch, spec)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), BUDGET['gradient_clip'], error_if_nonfinite=True)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite decoder loss')
    optimizer.step()
    return float(loss.detach()), float(norm)


def optimizer_for(model):
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                             lr=BUDGET['learning_rate'], weight_decay=BUDGET['weight_decay'])


def synthetic(arm, device, steps=12):
    """Use generated observations, targets and initialized dictionaries only."""
    cfg = yaml.safe_load((CONTRACT.parent / 'dual_component_base_g1_v2.yaml').read_text())
    cfg['architectures']['waveform'].update(width=256, latent_dim=128)
    cfg['model']['geometry_each_forward'] = False
    cfg['capacity'] = dict(sizes=[8192], uniform_count_fraction=.1)
    torch.manual_seed(17)
    model = _base_training_model(cfg, CAP.model_task('eeg', 8192), single_id_spectrum=True).to(device).eval()
    for branch in model.branches.values():
        branch['quantizer'].codebook.normal_()
        branch['quantizer'].initialized.fill_(True)
    x = torch.randn(BUDGET['batch_size'], 15, 400, device=device)
    batch = dict(x=x, s=.4*x, o=.6*x, mask=torch.ones_like(x, dtype=torch.bool))
    batch['mask'][0, 0, -1] = False
    for k in 'xso':
        batch['v'+k] = torch.ones_like(x)
        batch['dv'+k] = x.new_ones(len(x), 1)
    with torch.no_grad():
        before = model(x)
        model.prepare_decoder_refit(PLAN['arms'][arm])
        replay = model.forward_frozen_ids(x)
        for k in ('S', 'O', 'sum'):
            torch.testing.assert_close(before[k], replay[k], atol=0, rtol=0)
    opt = optimizer_for(model)
    spec = cfg['architectures']['waveform']
    update(model, batch, spec, opt)
    torch.cuda.synchronize()
    began = time.monotonic()
    for _ in range(steps):
        update(model, batch, spec, opt)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - began
    with torch.no_grad():
        after = model(x)
        for k in 'SO':
            torch.testing.assert_close(after['ids'][k], before['ids'][k], atol=0, rtol=0)
            torch.testing.assert_close(model.decode_ids(after['ids'])[k], after[k], atol=0, rtol=0)
            if torch.equal(after[k], before[k]):
                raise ValueError('decoder received no update')
    return dict(status='passed', arm=arm, steps_per_second=steps/elapsed, batch_size=len(x),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), gpu=os.environ['CUDA_VISIBLE_DEVICES'])


def monitor_panel(panel, groups):
    rng = np.random.default_rng(8017)
    train = base_tokenizer_sample(rng, groups, 256)
    people = defaultdict(list)
    for ds in CAP.DATASETS:
        for example in panel['examples']['eeg', 'validation', ds]:
            row = panel['rows'][example[0]]
            people[ds, row['subject']].append(example)
    dev = [items[i] for _, items in sorted(people.items())
           for i in rng.choice(len(items), min(len(items), 16), replace=False)]
    return dict(train=np.asarray(train), development=np.asarray(dev))


@torch.inference_mode()
def monitor(model, examples, get_batch, spec):
    values = []
    for offset in range(0, len(examples), BUDGET['batch_size']):
        batch = get_batch(examples[offset:offset+BUDGET['batch_size']])
        res = model.forward_frozen_ids(batch['x'])
        _, terms = _state_refinement_loss(res, batch, spec)
        values.extend([[float(terms[k]) for k in ('x', 's', 'o')]] * len(batch['x']))
    return np.mean(values, axis=0).tolist()


@torch.inference_mode()
def score(model, panel, dest, parent):
    dest.mkdir(exist_ok=True)
    rows, scales = panel['rows'], panel['scales']
    keys = sorted({(r['dataset'], p) for r in rows if r['modality'] == 'eeg' for p in r['positions']})
    sites = {key:i for i, key in enumerate(keys)}
    fixed = read_json(CAP.OUT / 'review_20261124_v1/tables/wave_eeg.json')
    cases = {(c['row'], c['start_s'], c['position']): name
             for c, name in zip(fixed, ('median', 'difficult'), strict=True)}
    cal = aligned.objective()['calibration']
    metrics = defaultdict(list)
    examples = []
    continuous = CAP.OUT / 'evaluation/eeg_d128_k0_s17'
    for ds in CAP.DATASETS:
        ex = panel['examples']['eeg', 'validation', ds]
        name = ds + '_validation.npz'
        if not (dest / name).is_file():
            extract(model, ex, ds, 'validation', rows, scales, cal, sites, cases, dest)
        with (np.load(dest / name) as z, np.load(parent / name) as ref,
              np.load(continuous / ('eeg_' + name)) as cont):
            for data in (z, ref, cont):
                np.testing.assert_array_equal(data['examples'], ex)
            for k in ('id', 'valid', 'target', 'bg'):
                np.testing.assert_array_equal(z[k], ref[k])
            for k in ('nmse', 'correlation'):
                metrics[k].append(z[k])
            metrics['decoded'].append(patch_means((z['decoded'] - z['target'])**2, z['valid']))
        examples.append(ex)
    for case in cases.values():
        with np.load(dest / (case+'_waveforms.npz')) as z, np.load(parent / (case+'_waveforms.npz')) as ref:
            for k in ('example', 'ids_S', 'ids_O', 'mask'):
                np.testing.assert_array_equal(z[k], ref[k])
    result = base_training_aggregate({k:np.concatenate(v) for k,v in metrics.items()}, np.concatenate(examples), rows)
    if (result['inputs'] != PLAN['scope']['development_positions'] or result['subjects'] != PLAN['scope']['development_people']
            or result['metrics']['nmse']['nonfinite_entries'] or result['metrics']['decoded']['nonfinite_entries']):
        raise ValueError('incomplete development scoring')
    write_json(dest / 'summary.json', dict(result, status='completed'))


def train(arm):
    cfg, task = aligned.settings('eeg')
    parent_result = read_json(aligned.OUT / 'eeg/result.json')
    path = aligned.ROOT / parent_result['checkpoint']
    signature = dict(contract=identity(PLAN), parent=read_json(path.parent / 'summary.json')['signature'], arm=arm)
    dest = OUT / arm
    if (dest / 'result.json').exists():
        if read_json(dest / 'result.json')['signature'] != signature:
            raise ValueError('completed arm signature changed')
        return
    source = audit.network('eeg', 8192, 17, 'cuda:0')
    model = copy.deepcopy(source)
    model.prepare_decoder_refit(PLAN['arms'][arm])
    opt = optimizer_for(model)
    CAP.prepare_panel(recorded_root=aligned.spectral.PLAN['recorded_root'])
    panel = CAP.PANEL
    groups = base_tokenizer_sampling_index(panel['rows'], 'eeg')
    probes = monitor_panel(panel, groups)
    cache = OrderedDict()
    spec = cfg['architectures']['waveform']
    def batch(ex):
        return _base_training_batch(panel['rows'], ex, panel['scales'], 'cuda:0', cache, cfg)
    params = {k:p for k,p in model.named_parameters() if p.requires_grad}
    rng = np.random.default_rng(BUDGET['seed'])
    start = 0
    history = []
    checkpoint = dest / 'checkpoint.pt'
    write_json(dest / 'resolved_config.json', dict(plan=PLAN, parent_checkpoint=parent_result['checkpoint'],
               parent_config=cfg, task=task, signature=signature, trainable=list(params),
               monitor_examples=probes, trainable_parameters=sum(p.numel() for p in params.values())))
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cuda:0', weights_only=False)
        if saved['signature'] != signature or saved['parameters'].keys() != params.keys():
            raise ValueError('resume signature mismatch')
        with torch.no_grad():
            for k,p in params.items():
                p.copy_(saved['parameters'][k])
        opt.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['rng']
        start, history = saved['step'], saved['history']
    else:
        with torch.inference_mode():
            x = batch(probes['train'][:64])['x']
            baseline = source(x)
            replay = model.forward_frozen_ids(x)
            for k in 'SO':
                torch.testing.assert_close(baseline['ids'][k], replay['ids'][k], rtol=0, atol=0)
                torch.testing.assert_close(baseline[k], replay[k], rtol=0, atol=0)
        history.append(dict(step=0, **{key:monitor(model, ex, batch, spec) for key,ex in probes.items()}))
    began = time.monotonic()
    for step in range(start+1, BUDGET['steps']+1):
        examples = base_tokenizer_sample(rng, groups, BUDGET['batch_size'])
        loss, norm = update(model, batch(examples), spec, opt)
        if step == 1 or step % BUDGET['checkpoint_every'] == 0:
            history.append(dict(step=step, training_loss=loss, gradient_norm=norm,
                                **{k:monitor(model, ex, batch, spec) for k,ex in probes.items()}))
            _torch_save(checkpoint, dict(step=step, parameters=params, optimizer=opt.state_dict(),
                                         rng=rng.bit_generator.state, history=history, signature=signature))
            write_json(dest / 'history.json', history)
            write_json(dest / 'progress.json', dict(status='training', step=step, budget=BUDGET['steps'],
                       steps_per_second=(step-start)/(time.monotonic()-began), updated=now()))
    parent_state = source.state_dict()
    for k,v in model.state_dict().items():
        if k not in params:
            torch.testing.assert_close(v, parent_state[k], rtol=0, atol=0)
    write_json(dest / 'progress.json', dict(status='evaluating', step=BUDGET['steps'], updated=now()))
    score(model, panel, dest / 'evaluation', path.parent / 'evaluation')
    result = read_json(dest / 'evaluation/summary.json')
    write_json(dest / 'result.json', dict(status='completed', signature=signature,
               macro_nmse=result['selection_score'], step=BUDGET['steps'], finished=now()))
    write_json(dest / 'progress.json', dict(status='completed', step=BUDGET['steps'], updated=now()))


def compare_fixed_windows(parent):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    cases = read_json(CAP.OUT / 'review_20261124_v1/tables/wave_eeg.json')
    sources = [parent, *(OUT / arm / 'evaluation' for arm in ARMS)]
    names = ['Original', 'Continued (2 layers)', 'Expanded (4 layers)']
    for name, case in zip(('median', 'difficult'), cases, strict=True):
        fig, axes = plt.subplots(3, 3, sharex=True, sharey='row', figsize=(14, 7))
        for col, (source, decoder) in enumerate(zip(sources, names, strict=True)):
            info = read_json(source / 'simultaneous_eeg_nirs_fixed_windows.json')[name]
            with np.load(source / (name+'_waveforms.npz')) as z:
                valid = z['mask'].reshape(-1)
                time_s = np.arange(len(valid)) / 200
                for row, (truth, recon) in enumerate([('x','sum'), ('s','S'), ('o','O')]):
                    ax = axes[row, col]
                    target = 'Observed X' if truth == 'x' else 'Teacher ' + truth.upper()
                    ax.plot(time_s, np.where(valid, z[truth].reshape(-1), np.nan),
                            lw=.6, color='0.42', label=target)
                    ax.plot(time_s, np.where(valid, z['reconstructed_'+recon].reshape(-1), np.nan),
                            lw=.65, color='#007c91', label='Integer reconstruction')
                    ax.set_title(f"{decoder}   NMSE {info['nmse'][row]:.3f}", fontsize=10)
                    if col == 0:
                        ax.set_ylabel(truth.upper()+' / training input SD')
                        ax.legend(loc='upper right', fontsize=7)
                    if row == 2:
                        ax.set_xlabel('Time within window (s)')
        fig.suptitle(f"Fixed {name} EEG window  |  {case['subject']}  {case['record']}  {case['position']}")
        fig.tight_layout()
        fig.savefig(OUT / (name+'_waveforms.png'), dpi=220)
        plt.close(fig)


def summarize():
    parent = aligned.ROOT / read_json(aligned.OUT / 'eeg/result.json')['checkpoint']
    original = read_json(parent.parent / 'evaluation/summary.json')
    continuous = read_json(CAP.OUT / 'evaluation/eeg_d128_k0_s17/eeg_summary.json')
    records = {a:read_json(OUT / a / 'result.json') for a in ARMS}
    scores = {a:read_json(OUT / a / 'evaluation/summary.json') for a in ARMS}
    if any(r['signature']['contract'] != identity(PLAN) or r['status'] != 'completed' for r in records.values()):
        raise ValueError('both terminal results required')
    comparisons = {}
    for metric in ('nmse', 'correlation', 'decoded'):
        def people(record):
            return {(r['dataset'],r['subject']):np.asarray(r['value'], dtype=float)
                    for r in record['metrics'][metric]['by_subject']}
        left, right = (people(scores[a]) for a in ARMS)
        if left.keys() != right.keys() or len(left) != PLAN['scope']['development_people']:
            raise ValueError('paired development people required')
        comparisons[metric] = [dict(dataset=d, subject=s, expanded_minus_continued=right[d,s]-left[d,s])
                               for d,s in sorted(left)]
    compare_fixed_windows(parent.parent / 'evaluation')
    write_json(OUT / 'summary.json', dict(status='completed', finished=now(), arms=records,
               original_macro_nmse=original['selection_score'], continuous_reference_macro_nmse=continuous['selection_score'],
               frozen_id_mean_gain=original['mean_id_gain'], paired=comparisons,
               expanded_minus_continued_macro_nmse=records['expanded']['macro_nmse']-records['continued']['macro_nmse']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pilot', 'train', 'summarize'])
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    check()
    if args.action == 'summarize':
        summarize()
        return
    if not args.arm:
        parser.error('--arm required')
    check_gpu()
    torch.set_num_threads(1)
    with owner_lock(OUT / args.arm):
        if args.action == 'pilot':
            write_json(OUT / args.arm / 'pilot.json', synthetic(args.arm, 'cuda:0'))
        else:
            pilot = read_json(OUT / args.arm / 'pilot.json')
            if pilot['status'] != 'passed' or pilot['arm'] != args.arm:
                raise ValueError('missing matched synthetic pilot')
            try:
                train(args.arm)
            except Exception as error:
                write_json(OUT / args.arm / ('failure_' + str(time.time_ns()) + '.json'),
                           dict(error=repr(error), updated=now()))
                raise


if __name__ == '__main__':
    main()
