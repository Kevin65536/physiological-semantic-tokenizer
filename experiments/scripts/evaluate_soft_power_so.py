"""Matched three-target EEG comparison on the accepted unified-grid pilot."""
import argparse
import fcntl
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import yaml

from experiments.scripts import evaluate_unified_eeg_so_pilot as parent
from src.data.dual_component_tokenizer import voltage_power_cap_bank, voltage_power_cap_reference, voltage_power_cap_target
from src.data.so_supervision_cache import bind_reference, write_json
from src.data.soft_power_so import fit_soft_channel
from src.data.window_exclusions import local_spans

CONTRACT = ROOT / 'experiments/configs/physiology_semantic_tokenizer/soft_power_so_pilot_v1.yaml'
CFG = yaml.safe_load(CONTRACT.read_text())
OUT = ROOT / CFG['output']
PARENT = ROOT / CFG['parent']
PRODUCTION = ROOT / CFG['production']


def read(path):
    return json.loads(path.read_text())


def plan():
    p = read(PARENT / 'plan.json')
    assert PARENT == parent.OUT and read(PARENT / 'verification.json')['status'] == 'passed'
    assert read(PRODUCTION / 'verification.json')['status'] == 'passed'
    assert p['train_reference_windows'] == 348 and parent.CFG['seed'] == CFG['seed']
    keys = p['cases']
    assert len(keys) == len(set(tuple(k.split('/')[:2]) for k in keys)) == 8
    cases = []
    for i, key in enumerate(keys):
        ident = read(PARENT / 'inputs' / str(i) / 'identity.json')
        g = ident['group']
        assert g['key'] == key and g['split'] == ident['co']['split'] == 'validation'
        f = PRODUCTION / 'windows' / key
        m = read(f.with_suffix('.json'))
        assert (m['schema'], m['key'], m['split'], m['status']) == (
            'physiology_so_supervision_v3', key, 'validation', 'completed')
        assert m['reference_selected_positions'] == ident['positions']
        src = m['source']
        assert m['EEG_channels'] == g['EEG_channels'] and src['samples'] == 18000
        excluded = local_spans(g['dataset'], 'validation', g['subject'], g['native_record'],
                               'eeg', m['EEG_channels'], src['start_sample'], src['samples'])
        with np.load(f.with_suffix('.npz')) as a, np.load(PARENT/'fits'/str(i)/'unified/baseline.npz') as b:
            np.testing.assert_array_equal(a['eeg_reference_proxy'], b['prediction'][:, :, 0])
            np.testing.assert_array_equal(a['eeg_reference_projection'], np.load(PARENT/'inputs'/str(i)/'projection.npy'))
            np.testing.assert_array_equal(a['eeg_x'], np.load(PARENT/'inputs'/str(i)/'unified.npy'))
            assert a['eeg_observation_valid'].all() and a['eeg_teacher_valid'].all()
            np.testing.assert_array_equal(a['eeg_source_scoring_mask'], np.arange(18000) < 17980)
        valid = [j for j in range(len(m['EEG_channels'])) if j not in {s[0] for s in excluded}]
        cases.append(dict(i=i, key=key, dataset=g['dataset'], subject=g['subject'],
                          source=src, unit=m['EEG_unit'], channels=m['EEG_channels'],
                          valid=valid, excluded=excluded, positions=ident['positions']))
    return cases


def generate(case, variant):
    started = time.monotonic()
    i = case['i']
    folder = OUT/'targets'/str(i)
    folder.mkdir(exist_ok=True, parents=True)
    terminal = folder/(variant+'.json')
    if terminal.is_file() and read(terminal)['status'] == 'completed':
        return read(terminal)
    _, x, rate, de, h, _, _ = parent.known_inputs(i, 'unified', variant)
    assert rate == 200 and x.shape == (18000, len(case['channels']))
    fits = PARENT/'fits'/str(i)/'unified'
    assert read(fits/(variant+'.json'))['qualified']
    with np.load(fits/(variant+'.npz')) as f:
        proxy = f['prediction'][:, :, 0]
        np.testing.assert_allclose(h, f['target'][:, :, 1:], atol=1e-13)
    with np.load(PARENT/'references'/(case['dataset']+'.npz')) as f:
        model = bind_reference(dict(f), np.load(PARENT/'inputs'/str(i)/'projection.npy'), 900)
    tolerance = .5 * float(np.median(np.sqrt(np.diag(model['covariance']))))
    old = voltage_power_cap_target(x, model, proxy, audit_final=False, frame_batch_size=8)['s']
    if variant == 'baseline':
        with np.load((PRODUCTION/'windows'/case['key']).with_suffix('.npz')) as a:
            np.testing.assert_allclose(old, a['eeg_s'], rtol=1e-8, atol=1e-10*np.max(model['voltage_rms']))
    valid = case['valid']
    x = np.ascontiguousarray(x[:, valid])
    bank = voltage_power_cap_bank(x)
    arrays = dict(x=x, delta=de[:, valid], v3=old[:, valid], original_channel_index=np.array(valid))
    diag = {}
    for arm, signal in [('soft_ssm', proxy), ('soft_fixed', None)]:
        refs = voltage_power_cap_reference(model, signal, np.arange(5, 896)/10)
        gains, estimates, info = [], [], []
        for j, index in enumerate(valid):
            s, knots, row = fit_soft_channel(bank[:, :, j], refs[:, :, index], model['voltage_rms'][index],
                                              tolerance, CFG['gain_knot_seconds'], CFG['lambda_power'])
            estimates.append(s)
            gains.append(knots)
            info.append(row)
        arrays[arm] = np.column_stack(estimates)
        arrays[arm+'_gain'] = np.stack(gains)
        diag[arm] = info
    closure = {}
    for arm in CFG['arms']:
        residual = x - arrays[arm]
        closure[arm] = float(np.max(abs(arrays[arm]+residual-x)) / parent.rms(x))
        assert closure[arm] < 1e-10
        if variant == 'baseline':
            arrays[arm+'_o'] = residual
    temp = folder/(variant+'.partial.npz')
    np.savez(temp, **arrays)
    temp.replace(folder/(variant+'.npz'))
    row = dict(status='completed', i=i, variant=variant, delta=tolerance, training_windows=int(model['training_windows']),
               closure_over_x_rms=closure, fit=diag, seconds=time.monotonic()-started)
    write_json(terminal, row)
    return row


def run(workers):
    cases = plan()
    manifest = dict(config=CFG, cases=cases, workers=workers)
    plan_file = OUT/'plan.json'
    if plan_file.exists():
        previous = read(plan_file)
        if previous['cases'] != cases or previous['config'] != CFG:
            raise ValueError('Existing targets belong to another plan')
    else:
        write_json(plan_file, manifest)
    tasks = [(c,v) for c in cases for v in CFG['variants']]
    failed = []
    with ProcessPoolExecutor(max_workers=min(len(tasks), workers)) as pool:
        jobs = {pool.submit(generate, c, v): (c['i'], v) for c,v in tasks}
        for future in as_completed(jobs):
            i,v = jobs[future]
            try:
                row = future.result()
                print(i, v, round(row['seconds'], 2), flush=True)
            except Exception:
                trace = traceback.format_exc()
                failed.append(dict(i=i, variant=v, detail=trace))
                print(i, v, trace, flush=True)
    write_json(OUT/'generation_status.json', dict(planned=len(tasks), failed=failed,
               status='failed' if failed else 'completed'))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'generation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.workers)
