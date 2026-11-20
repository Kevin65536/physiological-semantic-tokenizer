"""A synthetic completed G0 root; no measured files or teacher arrays are needed."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from src.data import observation_tokenizer_panel as p


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def get(path):
    return json.loads(path.read_text())


@pytest.fixture
def g0(tmp_path, monkeypatch):
    root = tmp_path / 'g0'
    monkeypatch.setattr(p, 'G0_ROOT', root)
    plan = dict(schema='dual_component_base_plan_v1', allowed_splits=['train', 'validation'],
                subjects=[], windows=[], duplicate_training_crops_removed=2,
                regression_inputs=4, regression_crops_excluded=4)
    manifest = dict(schema='dual_component_base_cache_v1', status='completed', cfg_identity=p.G0_CONFIG_ID,
                    allowed_splits=plan['allowed_splits'], rows=[])
    scales = dict(schema='dual_component_base_scales_v1', fitting_split='train', positions={'hb': {}, 'eeg': {}})
    for d in p.DATASETS:
        hz = p.NATIVE_HZ[d]
        for subject, split in [('a', 'train'), ('b', 'train'), ('v', 'validation'), ('w', 'validation')]:
            plan['subjects'].append(dict(dataset=d, subject=subject, split=split))
            for r in (['r1', 'r2'] if subject != 'b' else ['r2']):
                key = f'{d}/{subject}/{r}/win'
                ids = dict(key=key, dataset=d, subject=subject, split=split)
                clock = dict(key='w0000', eeg_start_s=.1, hb_start_s=1.1)
                src = dict(**ids, native_record=r, rate=hz, start_sample=hz//10, samples=90*hz,
                           duration_s=90, members=[dict(record_key=f'{d}/{subject}/{r}', window=clock)])
                reg = [30] if (subject, r) == ('a', 'r1') else []
                starts = [0, 60] if reg else [0, 30, 60]
                meta = dict(**ids, source=src, status='completed', EEG_unit='relative', fNIRS_unit='relative')
                w = dict(row=ids, metadata=meta, score_starts_s=starts, regression_starts_s=reg,
                         blocked_samples=[[30*hz, 60*hz]] if reg else [])
                plan['windows'].append(w)
                for m, rate, c in [('eeg', 200, 1), ('hb', 10, 2)]:
                    names = ['Cz', r] if m == 'eeg' else [r, 'HbC']
                    if subject == 'v': names.reverse()  # never inherit a column number
                    meta['EEG_channels' if m == 'eeg' else 'fNIRS_positions'] = names
                    scale_map = scales['positions'][m].setdefault(d, {})
                    for n in names:
                        scale_map[n] = dict(scale=2., target_variance={'x': [1.5]*c})
                    row = dict(modality=m, rate=rate, channels=c, source_unit='relative', dataset=d,
                               subject=subject, record=r, parent=key, split=split, positions=names,
                               native_start_sample=hz//10, native_rate_hz=hz, source_status='completed',
                               folder=str(root/'cache'/p._identity(key)/m), inputs={
                                   'score': [dict(start_s=s, positions=[]) for s in starts],
                                   'regression': [dict(start_s=s, positions=[0]) for s in reg]})
                    manifest['rows'].append(row)
    manifest['scales_identity'] = p._identity(scales)
    manifest['plan_identity'] = p.G0_PLAN_ID
    monkeypatch.setattr(p, 'G0_SCALES_ID', manifest['scales_identity'])
    for filename, obj in [('scales.json', scales), ('plan.json', plan), ('cache/manifest.json', manifest),
                          ('summary.json', dict(status='completed', retained_source_failures=[])),
                          ('verification.json', dict(status='passed', issues=[]))]:
        put(root/filename, obj)
    return root


def test_array_free_balanced_identity_first_panel(g0, monkeypatch):
    monkeypatch.setattr(np, 'load', lambda *args, **kw: pytest.fail('plan touched an array'))
    limits = p.PanelLimits(2, 2, 3, 1)
    panel = p.build_panel(g0, limits=limits)
    path = g0/'cache/manifest.json'
    manifest = get(path); manifest['rows'].reverse(); put(path, manifest)
    assert panel.rows == p.build_panel(g0, limits=limits).rows
    weights = defaultdict(float)
    for x in panel.rows:
        assert x.position == ('Cz' if x.modality == 'eeg' else 'HbC')
        assert (x.subject, x.record, x.start_s) != ('a', 'r1', 30)
        assert x.native_start_sample == x.native_rate_hz//10 + x.start_s*x.native_rate_hz
        assert x.member_clocks[0][3] == pytest.approx(x.start_s+1.1)
        weights[x.modality, x.split, x.dataset, x.subject] += x.weight
    assert len(weights) == 32
    assert list(weights.values()) == pytest.approx([1/8]*32)
    assert panel.counts['g0']['regression_inputs'] == 4


def test_reads_only_three_maps_one_position_and_never_requires_labels(g0, monkeypatch):
    panel = p.build_panel(g0, datasets=(p.DATASETS[0],), positions={'eeg': ['Cz'], 'hb': ['HbC']})
    i, row = next((i, r) for i, r in enumerate(panel.rows) if r.modality == 'hb' and r.start_s == 0)
    folder = g0/'cache'/p._identity(row.parent)/'hb'; folder.mkdir(parents=True)
    x = np.full((900, 2, 2), 8., np.float32); mask = np.ones_like(x, bool)
    x[:, row.position_index] = [4, 10]; mask[38, row.position_index, 1] = False
    score = np.ones(900, bool); score[1] = False
    for name, a in [('x', x), ('observation_valid', mask), ('source_scoring', score)]:
        np.save(folder/(name+'.npy'), a)
    original = np.load; opened = []
    def checked(path, **kw):
        opened.append(Path(path).stem); assert kw == dict(mmap_mode='r', allow_pickle=False)
        return original(path, **kw)
    monkeypatch.setattr(np, 'load', checked)
    b = panel.read(i)
    assert set(opened) == {'x', 'observation_valid', 'source_scoring'}
    assert b['x'].shape == b['mask'].shape == b['score_mask'].shape == (15, 20, 2)
    np.testing.assert_array_equal(b['x'][0, 1], [2, 5])
    assert b['mask'][0, 1].all() and not b['score_mask'][0, 1].any()
    assert not b['available'] and not b['patch_available'][1] and b['x'][1, 18, 1] == 0
    assert b['patch_scorable'].sum() == 13 and row.vx == (1.5, 1.5)


@pytest.mark.parametrize('corruption', ['split', 'cross', 'overlap', 'regression', 'folder', 'scales', 'subject'])
def test_rejects_corrupt_plan_before_array_reads(g0, monkeypatch, corruption):
    monkeypatch.setattr(np, 'load', lambda *args, **kw: pytest.fail('boundary failed'))
    mp, pp = g0/'cache/manifest.json', g0/'plan.json'; m, plan = get(mp), get(pp)
    row, w = m['rows'][0], plan['windows'][0]
    if corruption == 'split': row['split'] = 'test'
    if corruption == 'cross': plan['subjects'].append(dict(plan['subjects'][0], split='validation'))
    if corruption == 'subject': row['subject'] = 'other'
    if corruption == 'overlap':
        w['score_starts_s'].append(1); row['inputs']['score'].append(dict(start_s=1, positions=[]))
    if corruption == 'regression':
        w['score_starts_s'].append(30); row['inputs']['score'].append(dict(start_s=30, positions=[]))
    if corruption == 'folder': row['folder'] = str(g0/'protected')
    if corruption == 'scales': m['scales_identity'] = 'stale'
    put(mp, m); put(pp, plan)
    with pytest.raises(ValueError): p.build_panel(g0)


def test_path_scope_is_checked_again_before_mmap(g0, monkeypatch):
    panel = p.build_panel(g0); row = panel.rows[0]
    folder = g0/'cache'/p._identity(row.parent)/row.modality
    folder.mkdir(parents=True); (folder/'source_scoring.npy').symlink_to(g0/'forbidden.npy')
    monkeypatch.setattr(np, 'load', lambda *args, **kw: pytest.fail('unexpected file open'))
    with pytest.raises(ValueError): panel.read(0)
    with pytest.raises(ValueError): p.build_panel(g0.parent/'forbidden')
