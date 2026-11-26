"""Train and evaluate the three matched EEG input and encoder contrasts."""
import argparse
import os
import time

import torch
import yaml

import retrain_single_vq_aligned as aligned
from run_dual_component_tokenizer import (
    _base_forward, _base_training_model, _state_refinement_loss, identity, owner_lock,
    read_json, train_base_tokenizer, write_json,
)
from run_single_vq_dynamic_factorial import check_gpu
from score_single_vq_spectral_location import score as score_hard

CONTRACT = aligned.CONTRACT.with_name('eeg_encoder_diagnostics_v1.yaml')
PLAN = yaml.safe_load(CONTRACT.read_text())
ROOT = aligned.ROOT
OUT = ROOT / PLAN['run']
ARCHS = PLAN['architectures']
PROBES = ROOT / 'experiments/runs/physiology_semantic_tokenizer/eeg_frozen_decoder/20261126_v1/continued/resolved_config.json'


def settings(arch, *, synthetic=False):
    if arch not in ARCHS:
        raise ValueError('unknown architecture')
    if synthetic:
        cfg = yaml.safe_load((CONTRACT.parent / 'dual_component_base_g1_v2.yaml').read_text())
        cfg['architectures']['waveform'].update(width=256, latent_dim=128)
        cfg['capacity'] = dict(sizes=[1024, 4096, 8192], uniform_count_fraction=.1)
        cfg['model']['geometry_each_forward'] = False
        cfg['training'].update(PLAN['budget'])
    else:
        cfg, _ = aligned.settings('eeg')
    a = ARCHS[arch]
    iface = PLAN['interface']
    spec = cfg['architectures']['waveform']
    if (any(cfg['training'][k] != v for k,v in PLAN['budget'].items())
            or (cfg['model']['patches'], cfg['model']['decoder_layers'], spec['local_width'],
                spec['width'], spec['latent_dim']) != (15, 2, 32, 256, 128)
            or iface != dict(rate_hz=200, patches=15, patch_samples=400, width=256, dimension=128,
                             hard_size=8192, decoder_layers=2, decoder_local_width=32)
            or PLAN['scope'] != dict(train_people=60, development_people=16, eeg_positions=53476,
                                      test='forbidden', confirmation='forbidden', raw_reads=False, teacher_refits=False)
            or cfg['data']['allowed_splits'] != ['train','validation']):
        raise ValueError('aligned interface, training or scope drift')
    cfg['experiment_id'] = PLAN['experiment_id']
    cfg['architecture_diagnostic'] = dict(architecture=arch, contract=identity(PLAN))
    if a['source'] == 'SO':
        spec['target_input'] = True
    if a['input_width'] != spec['local_width']:
        spec['input_local_width'] = a['input_width']
    if a['encoder_layers'] != cfg['model']['encoder_layers']:
        cfg['model']['initial_encoder_layers'] = cfg['model']['encoder_layers']
        cfg['model']['encoder_layers'] = a['encoder_layers']
    return cfg


def task_for(arch, fit):
    if fit != 'warmup' and fit not in ARCHS[arch]['fits']:
        raise ValueError('unregistered architecture/representation pair')
    task = aligned.capacity.model_task('eeg', 8192 if fit == 'hard' else 0)
    return dict(task, candidate='warmup') if fit == 'warmup' else task


def job(arch, fit):
    return OUT / arch / 'models' / '__'.join(task_for(arch, fit).values())


def synthetic(arch, fit, device):
    cfg = settings(arch, synthetic=True)
    task = task_for(arch, fit)
    torch.manual_seed(17)
    model = _base_training_model(cfg, task, single_id_spectrum=fit=='hard').to(device)
    for b in model.branches.values():
        q = b['quantizer']
        q.initialized.fill_(True)
        q.ema_count.fill_(1.)
        q.ema_sum.copy_(q.codebook)
    x = torch.randn(64, 15, 400, device=device)
    s = torch.randn_like(x)*.5
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[0, -1, -1] = False
    batch = dict(x=x, s=s.masked_fill(~mask, 0), o=(x-s).masked_fill(~mask, 0), mask=mask,
                 input_scale=x.new_ones(len(x)), feature_mean=x.new_zeros(len(x), 1, 4),
                 feature_sd=x.new_ones(len(x), 1, 4))
    for k in 'xso':
        batch['v'+k] = torch.ones_like(x)
        batch['dv'+k] = x.new_ones(len(x), 1)
    spec = dict(cfg['architectures']['waveform'])
    if fit == 'hard':
        spec['spectral_objective'] = dict(location='single_id_spectrum', weight=.01)
    opt = torch.optim.AdamW(model.parameters(), lr=.0003)
    for step in range(21):
        if step == 1:
            torch.cuda.synchronize()
            start = time.monotonic()
        out = _base_forward(model, batch)
        loss, _ = _state_refinement_loss(out, batch, spec)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        opt.step()
    torch.cuda.synchronize()
    speed = 20 / (time.monotonic()-start)
    with torch.inference_mode():
        model.eval()
        out = _base_forward(model, batch)
        assert out['sum'].shape == x.shape and torch.isfinite(out['sum']).all()
        if fit == 'hard':
            replay = model.decode_ids(out['ids'])
            for k in 'SO':
                torch.testing.assert_close(replay[k], out[k], rtol=0, atol=0)
        if arch == 'privileged':
            other = _base_forward(model, dict(batch, x=x+1))
            torch.testing.assert_close(other['sum'], out['sum'], rtol=0, atol=0)
    return dict(status='passed', contract=identity(PLAN), arch=arch, fit=fit, steps_per_second=speed,
                batch_size=len(x), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                gpu=os.environ['CUDA_VISIBLE_DEVICES'])


def train(arch, fit):
    cfg = settings(arch)
    probe = read_json(PROBES)['monitor_examples']['development']
    if len(probe) != 256 or len(set(map(tuple,probe))) != 256:
        raise ValueError('fixed development monitor changed')
    base = OUT / arch
    path = base / 'resolved_config.json'
    resolved = dict(contract=PLAN, config=cfg, development_probe=probe)
    if path.exists() and read_json(path) != resolved:
        raise ValueError('resolved training configuration changed')
    write_json(path, resolved)
    obj = aligned.objective() if fit == 'hard' else None
    def run_candidate(candidate):
        result = train_base_tokenizer(cfg, ROOT, base, task_for(arch,candidate), 'cuda:0',
            recorded_root=aligned.spectral.PLAN['recorded_root'], development_probe=probe,
            spectral_objective=obj if candidate=='hard' else None)
        expected = PLAN['budget']['warmup_steps' if candidate=='warmup' else 'total_steps']
        if result['status'] != 'completed' or result['completed_step'] != expected:
            raise ValueError('incomplete training')
        return result
    with owner_lock(base, 'warmup_dispatch', blocking=True):
        run_candidate('warmup')
    return run_candidate(fit)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['pilot','train','score','summarize'])
    p.add_argument('--arch', choices=ARCHS)
    p.add_argument('--fit', choices=['continuous','hard'])
    args = p.parse_args()
    if args.action == 'summarize':
        from score_eeg_encoder_diagnostics import summarize
        return summarize()
    if not args.arch or not args.fit:
        p.error('--arch and --fit are required')
    task_for(args.arch,args.fit)
    check_gpu()
    torch.set_num_threads(1)
    folder = OUT / args.arch
    pilot = folder / (args.fit + '_pilot.json')
    with owner_lock(folder, args.fit + '_' + args.action, blocking=True):
        if args.action == 'pilot':
            write_json(pilot, synthetic(args.arch,args.fit,'cuda:0'))
        elif args.action == 'train':
            if read_json(pilot)['contract'] != identity(PLAN) or read_json(pilot)['status'] != 'passed':
                raise ValueError('matched synthetic pilot required')
            train(args.arch,args.fit)
        elif (job(args.arch,args.fit)/'evaluation/summary.json').exists():
            from score_eeg_encoder_diagnostics import verify_training
            verify_training(args.arch,args.fit)
            summary = read_json(job(args.arch,args.fit)/'evaluation/summary.json')
            if (summary['status'],summary['subjects'],summary['inputs']) != ('completed',16,53476):
                raise ValueError('incomplete existing score')
        elif args.fit == 'hard':
            score_hard(17,'C',job=job(args.arch,args.fit),cfg=settings(args.arch),
                       calibration_path=aligned.spectral.OUT/'calibration.json')
        else:
            from score_eeg_encoder_diagnostics import score_continuous
            score_continuous(args.arch)


if __name__ == '__main__':
    main()
