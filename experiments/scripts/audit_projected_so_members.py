"""Retrieve real assigned EEG patches for the projected-S checkpoint."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import yaml

from experiments.scripts.train_temporal_projection_vq import model, read
from experiments.scripts.score_soft_power_so import DownstreamInputDataset, eog_basis, parent
from src.data.so_supervision_cache import write_json

CONFIG = ROOT / 'experiments/configs/physiology_semantic_tokenizer/projected_so_token_members_v1.yaml'
CFG = yaml.safe_load(CONFIG.read_text())
SOURCE, OUT = (ROOT / CFG[k] for k in ('parent', 'output'))
FREQ = np.fft.rfftfreq(400, 1 / 200)
BRANCHES = ('S', 'O')


def attributes(x):
    if x.ndim != 2 or x.shape[1] != 400:
        raise ValueError('Expected two-second EEG patches at 200 Hz')
    centered = x - x.mean(axis=1, keepdims=True)
    length = np.linalg.norm(centered, axis=1, keepdims=True)
    power = abs(np.fft.rfft(centered * np.hanning(400))) ** 2
    total = power[:, (FREQ >= 1) & (FREQ <= 45)].sum(1)
    if (length <= 0).any() or (total <= 0).any() or not np.isfinite(power).all():
        raise ValueError('Invalid waveform')
    fractions = [power[:, (FREQ >= lo) & (FREQ < hi)].sum(1) / total for lo, hi in [(8, 13), (1, 4)]]
    return centered / length, fractions


def match(meta, members, controls):
    """Match inside person and electrode, allow repeat controls with identical weights."""
    ans = []
    for _, group in meta.groupby(['person', 'site'], sort=False):
        idx = group.index.to_numpy()
        left, right = idx[members[idx]], idx[controls[idx]]
        if not len(left) or not len(right):
            continue
        delta = abs(meta.log_rms.to_numpy()[left, None] - meta.log_rms.to_numpy()[right])
        nearest = delta.argmin(1)
        good = delta[np.arange(len(left)), nearest] <= np.log(1.4)
        ans.extend(zip(left[good], right[nearest[good]]))
    return np.asarray(ans, dtype=int).reshape(-1, 2)


def select(meta, unit, branch):
    train = meta[meta.role == 'train']
    results, templates = [], {}
    for token, g in train.groupby(branch):
        ix = g.index
        if any(stat < CFG['eligibility'][key] for key, stat in
               [('patches', len(ix)), ('people', g.person.nunique()), ('datasets', g.dataset.nunique())]):
            continue
        prototype = unit[ix].mean(0)
        strength = np.linalg.norm(prototype)
        templates[token] = prototype / max(float(strength), 1e-12)
        results.append(dict(token=token, shape=strength, alpha=g.alpha.mean(), slow=g.slow.mean(),
                            train_patches=len(ix), train_people=g.person.nunique()))
    scores = pd.DataFrame(results).set_index('token').sort_index()
    return [(m, int(scores[m].idxmax()), scores.loc[scores[m].idxmax()], templates[scores[m].idxmax()])
            for m in ['shape', 'alpha', 'slow']]


def prepare(device, arm=None, network=None):
    meta = pd.read_csv(SOURCE / 'examples.csv', dtype={'subject': str})
    valid = np.load(SOURCE / 'mask.npy')
    x = np.load(SOURCE / 'x.npy', mmap_mode='r').reshape(-1, 15, 400)
    assert valid.shape == x.shape[:2] == (len(meta), 15) and valid.dtype == bool
    ex, block = np.where(valid & (meta.role != 'shift').to_numpy()[:, None])
    m = meta.iloc[ex].reset_index(drop=True).assign(example=ex, block=block)
    m['person'] = m.dataset + '/' + m.subject
    t = m.role == 'train'
    assert (set(m.role) == {'train', 'validation'} and m[t].person.nunique() == 60
            and m[~t].person.nunique() == 8 and not set(m[t].person) & set(m[~t].person))
    assert not m.duplicated(['parent','site','start','block']).any()
    assert read(SOURCE/'summary.json')['status'] == 'completed'
    assert read(SOURCE.parent/'frozen.json')['pairs'] == 32
    if network is None:
        state = torch.load(SOURCE / (arm or CFG['checkpoint']['arm']) / 'resume.pt', map_location='cpu', weights_only=False)
        assert state['step'] == CFG['checkpoint']['step']
        network = model(True, device)
        network.load_state_dict(state['net'])
    net = network.eval()
    codes = {b: np.full(valid.shape, -1, dtype=int) for b in BRANCHES}
    with torch.inference_mode():
        for batches in np.array_split(np.unique(ex), 120):
            ids = net(torch.as_tensor(x[batches], device=device))['ids']
            for b in BRANCHES:
                assert ids[b].shape == (len(batches), 15)
                codes[b][batches] = ids[b].cpu().numpy()
    for b in BRANCHES:
        m[b] = codes[b][ex, block]
        assert m[b].between(0, 8191).all()
    waves = x[ex, block]
    unit = np.empty_like(waves)
    m['log_rms'] = np.log(np.sqrt(np.mean(waves**2, axis=1)))
    bands = np.empty((len(m), 2))
    for ix in np.array_split(np.arange(len(m)), 80):
        unit[ix], attrs = attributes(waves[ix])
        bands[ix] = np.stack(attrs, axis=1)
    m[['alpha','slow']] = bands
    return m, unit


def eog_marks(m):
    view = DownstreamInputDataset(ROOT / parent.CFG['downstream'], split='validation')
    marks = np.full(len(m), np.nan)
    for c in read(SOURCE.parent / 'plan.json')['cases']:
        q = eog_basis(view, c)
        if q is None:
            continue
        g = m[(m.parent == c['key']) & (m.role == 'validation')]
        offset = (200 * g.start + 400 * g.block).to_numpy()
        unique = np.unique(offset)
        assert len(unique) and (unique + 400 <= len(q)).all()
        energy = np.array([np.mean(q[j:j + 400]**2) for j in unique])
        marks[g.index] = (energy >= np.quantile(energy, .75))[np.searchsorted(unique, offset)]
    return marks


def summarize(m, metric, left, right, pairs, label, rows, persons, records):
    a, b = pairs.T
    person = m.person.to_numpy()
    values = []
    for p in np.unique(person[a]):
        j = person[a] == p
        v, c = metric[a[j]].mean(), metric[b[j]].mean()
        persons.append(dict(**label, person=p, n=int(j.sum()), member=v, control=c, difference=v-c))
        values.append((v, c))
    avg = np.mean(values, axis=0) if values else (np.nan, np.nan)
    rows.append(dict(**label, hits=int(left.sum()), available_controls=int(right.sum()), matched=len(a),
                     distinct_controls=len(set(b)), people=len(values), people_positive=sum(v>c for v,c in values),
                     member=avg[0], control=avg[1], difference=avg[0]-avg[1]))
    records.extend(dict(**label, member_row=int(j), control_row=int(k)) for j,k in pairs)


def run(device, *, output=OUT, arm=None, network=None, config=CFG):
    output.mkdir(exist_ok=True, parents=True)
    write_json(output/'resolved_config.json', config)
    m, unit = prepare(device, arm, network)
    valid = (m.role == 'validation').to_numpy()
    eog = eog_marks(m)
    assert m[np.isfinite(eog)].person.nunique() == 4
    m['high_eog'] = eog
    rows, persons, records, selection, pairs_selection = [], [], [], [], []
    templates = {}
    for branch, other in [('S','O'), ('O','S')]:
        for name, token, train, template in select(m, unit, branch):
            label = dict(branch=branch, endpoint=name, token=token)
            templates[branch+'_'+name] = template
            selection.append(dict(**label, train_n=int(train.train_patches), train_people=int(train.train_people),
                                  training_score=float(train[name])))
            hit = (m[branch] == token).to_numpy()
            left, right = valid & hit, valid & ~hit
            metric = unit @ template if name == 'shape' else m[name].to_numpy()
            summarize(m, metric, left, right, match(m,left,right), label, rows, persons, records)
            if name == 'slow':
                frontal = m.site.str.upper().str.startswith(('FP','AF')).to_numpy()
                support = np.isfinite(eog) & frontal
                l, r = left & support, right & support
                summarize(m, eog, l, r, match(m,l,r), dict(label,endpoint='eye'), rows, persons, records)
        train = m[~valid]
        frequencies = train.groupby([branch,other]).person.agg(n='size', people='nunique').reset_index()
        eligible = frequencies[frequencies.people >= 3]
        anchors = eligible.groupby(branch).size()
        anchor = int(train[train[branch].isin(anchors[anchors >= 2].index)][branch].value_counts().sort_index().idxmax())
        selected = eligible[eligible[branch]==anchor].sort_values(['n',other],ascending=[False,True]).iloc[:2]
        first, second = selected[other].astype(int)
        same = valid & (m[branch].to_numpy()==anchor)
        left, right = (same & (m[other].to_numpy()==k) for k in (first,second))
        label = dict(branch=branch,endpoint='pair_alpha',token=anchor)
        pairs_selection.append(dict(**label,first=first,second=second,train_first=int(selected.n.iloc[0]),
                                    train_second=int(selected.n.iloc[1]),validation_first=int(left.sum()),validation_second=int(right.sum())))
        summarize(m,m.alpha.to_numpy(),left,right,match(m,left,right),label,rows,persons,records)
    for fname, data in [('selection',selection),('pair_selection',pairs_selection),('contrasts',rows),
                        ('people',persons),('matches',records)]:
        pd.DataFrame(data).to_csv(output/(fname+'.csv'),index=False)
    m[valid].to_csv(output/'validation_members.csv.gz',index_label='row')
    np.savez(output/'templates.npz',**templates)
    write_json(output/'summary.json',dict(status='completed',experiment_id=config['experiment_id'],training=int((~valid).sum()),
                                      validation=int(valid.sum()),training_people=60,validation_people=8,eog_people=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','cuda:0'],default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.device)
