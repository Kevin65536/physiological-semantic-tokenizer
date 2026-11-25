"""Fixed continuous-span projection and matched teacher/ocular/physical diagnostics."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.temporal_projection_so import (
    continuous_projection,
    continuous_indices,
    project_blocks,
    complete_blocks,
)
from src.data.so_supervision_cache import bind_reference, write_json
from src.data.dual_component_tokenizer import base_tokenizer_training_data
from experiments.scripts import evaluate_temporal_projection_so as old
from experiments.scripts.score_soft_power_so import eog_basis, DownstreamInputDataset, parent
from experiments.scripts.train_temporal_projection_vq import BASE

CFG = yaml.safe_load(
    (ROOT / 'experiments/configs/physiology_semantic_tokenizer/continuous_projection_so_v2.yaml').read_text()
)
OUT = ROOT / CFG['output']
PRIOR = ROOT / CFG['parent']
FROZEN = old.read(PRIOR / 'frozen.json')
ORDER = FROZEN['order']


def admissible(row, mask):
    # Connected support comes only from accepted ordinary crops. The five regression
    # holdouts cannot affect neighbouring training labels through a global transform.
    valid = np.zeros(mask.shape, bool)
    for role in ('train', 'score'):
        for g in row['inputs'][role]:
            start = g['start_s'] * 200
            valid[start : start + 6000, g['positions']] = True
    valid &= mask
    for site, lo, hi in row.get('qc_spans', []):
        valid[lo:hi, site] = False
    return valid


def projection(x):
    return continuous_projection(x, np.ones(x.shape, bool), ORDER)


def synthetic():
    t = np.arange(18000) / 200
    ans = []
    for hz in (4.0, 8.0, 10.0, 11.0, 11.25, 11.5, 12.5, 12.95, 35.0):
        x = np.sin(2 * np.pi * hz * t[:, None] + np.arange(8) * np.pi / 8)
        b, sl = complete_blocks(x, 0)
        for arm, y in [('continuous', projection(x)), ('block', project_blocks(b, ORDER[:32]))]:
            for phase, (energy, error) in enumerate(
                zip(np.mean(y * y, 0) / np.mean(x * x, 0), np.mean((x - y) ** 2, 0) / np.mean(x * x, 0))
            ):
                ans.append(
                    dict(hz=hz, phase_rad=phase * np.pi / 8, arm=arm, energy=energy, relative_mse=error)
                )
    return pd.DataFrame(ans)


def train_check():
    manifest, _, _, _ = base_tokenizer_training_data(BASE, ROOT, recorded_root=old.CFG['recorded_root'])
    lookup = {r['parent']: r for r in manifest['rows'] if r['modality'] == 'eeg' and r['split'] == 'train'}
    rows = []
    for c in FROZEN['panel']:
        if c['role'] != 'select':
            continue
        r = lookup[c['parent']]
        sites = c['positions']
        path = Path(r['folder'])
        s = np.load(path / 's.npy', mmap_mode='r')[:, :, 0]
        mask = admissible(r, np.load(path / 'mask.npy', mmap_mode='r')[:, :, 0])[:, sites]
        y = continuous_projection(s[:, sites], mask, ORDER)
        origin = round(r['native_start_sample'] * 200 / r['native_rate_hz'])
        a = c['absolute_start_200'] - origin
        b = a + 400 * c['blocks']
        assert mask[a:b].all()
        v = s[a:b, sites].astype(float)
        rows.append(
            dict(dataset=r['dataset'], subject=r['subject'], **old.describe(y[a:b], v, old.band_bins(v)))
        )
    assert len(rows) == 28
    return pd.DataFrame(rows)


def case(c):
    ident = {k: c[k] for k in ('dataset', 'subject', 'i')}
    i = c['i']
    sites = c['valid']
    with np.load(old.SSM / 'references' / (c['dataset'] + '.npz')) as f:
        model = bind_reference(dict(f), np.load(old.SSM / 'inputs' / str(i) / 'projection.npy'), 900)
    base = {}
    tables = [[], [], []]
    for variant in old.CFG['variants']:
        with np.load(old.PARENT / 'targets' / str(i) / (variant + '.npz')) as f:
            np.testing.assert_array_equal(f['original_channel_index'], sites)
            x, s, d = [f[k][:-20] for k in ('x', 'v3', 'delta')]
        blocks, sl = complete_blocks(s, c['source']['start_sample'])
        full_teacher, full_projected = s, projection(s)
        methods = dict(
            v3=s[sl],
            block=project_blocks(blocks, ORDER[:32]),
            continuous=full_projected[sl],
            direct=projection(x)[sl],
        )
        x = x[sl]
        s = s[sl]
        d = d[sl]
        p = old.band_bins(s)
        with np.load(old.SSM / 'fits' / str(i) / 'unified' / (variant + '.npz')) as f:
            proxy = f['prediction'][:, :, 0]
        for arm, y in methods.items():
            o = x - y
            closure = abs(y + o - x).max() / old.rms(x)
            assert closure < 1e-10
            if variant == 'baseline':
                base[arm] = y
                diff = np.diff(y, axis=0)
                jump = diff[399::400]
                tables[0].append(
                    dict(
                        **ident,
                        arm=arm,
                        **old.describe(y, s, p),
                        closure=closure,
                        boundary_ratio=old.rms(jump) / old.rms(diff),
                        jump_rms=old.rms(jump),
                        all_diff_rms=old.rms(diff)
                    )
                )
            else:
                cut = slice(6000 - sl.start, 10000 - sl.start)
                delta = d[cut]
                dy = (y - base[arm])[cut]
                tables[1].append(
                    dict(
                        **ident,
                        arm=arm,
                        variant=variant,
                        leak=old.rms(dy) / old.rms(delta),
                        relative=old.rms(dy) / old.rms(base[arm][cut]),
                        closure=closure
                    )
                )
            if variant in ('baseline', 'ocular1'):
                cap = old.physics(y, model, proxy, sites, sl.start)
                tables[2].append(
                    dict(
                        **ident,
                        arm=arm,
                        variant=variant,
                        p90=np.quantile(cap, 0.9),
                        above_one=(cap > 1 + 1e-8).mean()
                    )
                )
        if variant == 'baseline':
            # Actual first/last 1 s of the full parent, kept separate from interior seams.
            edges = np.r_[0:200, len(full_teacher) - 200 : len(full_teacher)]
            for row in tables[0]:
                if row['arm'] in ('v3', 'continuous'):
                    delta = (full_teacher if row['arm'] == 'v3' else full_projected) - full_teacher
                    row.update(
                        edge_change_from_v3=old.rms(delta[edges]) / old.rms(full_teacher[edges]),
                        interior_change_from_v3=old.rms(delta[200:-200]) / old.rms(full_teacher[200:-200]),
                    )
            front = [j for j, idx in enumerate(sites) if c['channels'][idx].upper().startswith(('FP', 'AF'))]
            if c['dataset'] in ('eeg_fnirs_single_trial', 'simultaneous_eeg_nirs'):
                view = DownstreamInputDataset(ROOT / parent.CFG['downstream'], split='validation')
                q = eog_basis(view, c)[sl][200:-200]
                q = np.linalg.qr(q - q.mean(0))[0]

                def ocular(v):
                    a = v[200:-200, front]
                    return np.linalg.norm(q.T @ (a - a.mean(0)))

                for r in tables[0]:
                    r['eog_ratio'] = ocular(base[r['arm']]) / ocular(x)
    return tables


def run():
    OUT.mkdir(exist_ok=True, parents=True)
    resolved = OUT / 'resolved.json'
    if resolved.exists() and old.read(resolved) != CFG:
        raise ValueError('Target contract differs from the recorded run')
    if (OUT / 'target_gate.json').exists():
        return old.read(OUT / 'target_gate.json')
    write_json(resolved, CFG)
    synth = synthetic()
    synth.to_csv(OUT / 'synthetic.csv', index=False)
    train = train_check()
    train.to_csv(OUT / 'training_target.csv', index=False)
    cases = old.read(PRIOR / 'plan.json')['cases']
    with ProcessPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(case, cases))
    tables = []
    for i, name in enumerate(['target', 'intervention', 'cap']):
        df = pd.DataFrame([r for part in parts for r in part[i]])
        df.to_csv(OUT / (name + '.csv'), index=False)
        tables.append(df)
    base, inte, cap = tables
    b = base[base.arm == 'continuous']
    checks = dict(
        core=(synth.query('arm=="continuous" and hz<13').energy.min() >= 0.95),
        train=old.semantics_ok(train),
        development=old.semantics_ok(b),
        boundary=b.boundary_ratio.median() <= 1.5
        and b.boundary_ratio.median() < base[base.arm == 'block'].boundary_ratio.median(),
    )
    values = {}
    for frame, k, threshold, ratio in [
        (inte.query('variant=="ocular1"'), 'leak', 0.03, False),
        (inte.query('variant=="ocular1"'), 'relative', 0.15, False),
        (base.dropna(subset=['eog_ratio']), 'eog_ratio', 0.05, False),
        (cap.query('variant=="baseline"'), 'p90', 1.25, True),
        (cap.query('variant=="ocular1"'), 'p90', 1.25, True),
    ]:
        w = frame.pivot(index='i', columns='arm', values=k)
        val = (w.continuous / w.v3 if ratio else w.continuous - w.v3).median()
        key = k if not ratio else frame.variant.iloc[0] + '_cap'
        checks[key] = val <= threshold
        values[key] = float(val)
        if k == 'leak':
            checks['less_than_direct'] = (w.continuous - w.direct).median() < 0
    write_json(
        OUT / 'target_gate.json',
        dict(
            status='completed',
            checks={k: bool(v) for k, v in checks.items()},
            effects=values,
            advance=bool(all(checks.values())),
            subjects=8,
            train_check_people=len(train),
            rank_at_17980=len(continuous_indices(17980, ORDER)),
        ),
    )


if __name__ == '__main__':
    run()
