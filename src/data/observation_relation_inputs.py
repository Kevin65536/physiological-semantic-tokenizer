"""Replay the frozen observation-word panel; no model fitting or raw-data discovery.

Each physical 2 s word belongs to one native sensor, crop and split. EEG and Hb
originate on separately anchored clocks; matching is relative to the common G0
crop, NEVER on absolute EEG seconds or on matching sensor names or integer IDs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import time
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd

from src.analysis.observation_tokenizer_evaluation import KEYS
from src.data.observation_tokenizer_panel import DATASETS, G0_ROOT, PanelLimits, build_panel


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_routers(parent, release, identity):
    """Only unpickle the four exact, locally produced files; not a model download API."""
    archive = Path(release) / 'observation_baseline.zip'
    if digest(archive) != identity['release_sha256']:
        raise ValueError('release archive mismatch')
    with ZipFile(archive) as z:
        manifest = json.loads(z.read('observation_baseline/manifest.json'))
    routers, blind = {}, {}
    for m in ('eeg', 'hb'):
        router = Path(parent) / f'{m}_codec.joblib'; other = Path(parent) / f'{m}_blind_control.joblib'
        h = digest(router)
        if (h != manifest['sha256'][manifest['modalities'][m]['router']] or h != identity[m+'_router_sha256']
                or digest(other) != identity[m+'_blind_sha256']):
            raise ValueError('not the original released / matched-blind dictionary')
        for slot, path, routing in [(routers, router, 'pattern'), (blind, other, 'waveform')]:
            c = joblib.load(path)
            if (c.routing != routing or c.modality != m or c.n_patterns != 16 or c.seed != 17
                    or c.rate('prototype')['bits_per_item'] != 4):
                raise ValueError('frozen router or payload mismatch')
            slot[m] = c
    return routers, blind


def audit_clock(row):
    """Check the prepared EEG/Hb crop anchors; report offset, do not subtract it again."""
    if row.dataset not in DATASETS or row.split not in ('train','validation') or not row.member_clocks:
        raise ValueError('not the frozen development scope')
    origins, members = set(), []
    for rec, window, e, h in row.member_clocks:
        if (rec.split('/')[:2] != [row.dataset, row.subject] or not window or not np.isfinite([e,h]).all()
                or min(e,h) < 0 or abs(e*row.native_rate_hz-row.native_start_sample) > .500001
                or abs(h*10-round(h*10)) > 1e-6):
            raise ValueError('misregistered member anchor')
        origins.add((row.native_start_sample, round(h*10))); members.append(rec)
    if len(origins) != 1 or len(set(members)) != len(members):
        raise ValueError('ambiguous optical member ticks for this relative-time probe')
    if row.dataset == 'visual_cognitive_motivation' and row.modality == 'hb':
        # Native merged Visual slot is ProbeN:CHk, never a bare CHk pooled over two probes.
        probe, sep, ch = row.position.partition(':')
        if not (sep and probe in ('Probe1','Probe2') and ch.startswith('CH') and
                sum(probe.lower() in r.split('/')[-1].lower() for r in members) == 1):
            raise ValueError('Visual Hb position is not linked to one native probe')
    return tuple(sorted(row.member_clocks)), float(row.member_clocks[0][3]-row.member_clocks[0][2])


def features(frame, x, m, router, blind):
    shapes = {'eeg': (1,400), 'hb': (2,20)}
    if (x.shape != (len(frame), *shapes[m]) or not np.isfinite(x).all()
            or frame.duplicated(KEYS+['patch']).any() or not frame.patch.between(0,14).all()):
        raise ValueError('invalid observation wave/word identity')
    out = frame.copy()
    for name,c in [('pattern', router), ('blind', blind)]:
        payload = c.encode(x, mode='prototype'); ids = np.asarray(payload['pattern'])
        if set(payload) != {'pattern'} or ids.shape != (len(x),) or ids.dtype.kind not in 'iu' or not np.all((ids>=0)&(ids<16)):
            raise ValueError('not a four-bit-only word stream')
        out[name] = ids
    a = x.astype(float); out['rms'] = np.sqrt(np.mean((a-a.mean(-1,keepdims=True))**2, axis=(1,2)))
    props = router.pattern_features(x)
    for col,i in ([('alpha',2),('beta',3)] if m=='eeg' else [('hbo',0),('hbr',1),('so',2),('sr',3)]):
        out[col] = props[:,i]
    return out


def replay(parent, identity, routers, blind, io_workers=4):
    """Verify original ordered scope, masks, source clocks and scales; only then emit cells."""
    p = Path(parent); cfg = json.loads((p/'resolved_config.json').read_text()); d = cfg['data']
    if digest(p/'panel.csv') != identity['parent_panel_sha256'] or (d['splits'],d['test_access'],d['protected_access'],d['raw_reads']) != (['train','validation'],'forbidden','forbidden',False):
        raise ValueError('unrecognized parent / data contract')
    panel = build_panel(G0_ROOT, limits=PanelLimits(**d['panel']), seed=d['panel_seed'])
    saved = pd.read_csv(p/'panel.csv', dtype=str, keep_default_na=False); cols = ['modality','split',*KEYS]
    keys = lambda r: tuple(str(getattr(r,k)) for k in cols)
    marked = {tuple(row[k] for k in cols):row for row in saved.to_dict('records')}
    if len(marked)!=len(saved) or set(marked)!={keys(r) for r in panel.rows} or len(panel.rows)!=len(saved):
        raise ValueError('selected parents or sites changed')
    clocks,info = {}, {}; smoke={}; counter=Counter()
    for i,r in enumerate(panel.rows):
        key=tuple(getattr(r,k) for k in KEYS[:5]); anchors,delta=audit_clock(r)
        if anchors!=clocks.setdefault(key,anchors): raise ValueError('different clocks within same multimodal crop')
        old=marked[keys(r)]
        if (repr(r.member_clocks)!=old['member_clocks'] or r.native_start_sample!=int(old['native_start_sample']) or r.scale!=float(old['scale'])):
            raise ValueError('source identity/scale changed')
        row=info.setdefault((r.split,r.dataset),dict(crops=set(),offset=[],n_members=set(),positions={'eeg':set(),'hb':set()}))
        if key not in row['crops']:
            row['crops'].add(key);row['offset'].append(delta);row['n_members'].add(len(anchors))
        row['positions'][r.modality].add(r.position)
        if old['included']=='True': smoke.setdefault((r.split,r.dataset,r.modality),i)
    def verify(i,s):
        r = panel.rows[i]; before = marked[keys(r)]; mask = s['patch_scorable']; inc = s['available'] and mask.any()
        if (''.join(map(str,mask.astype(int)))!=before['valid_bits'] or bool(inc)!=(before['included']=='True')):
            raise ValueError('complete observation/scoring support drifted')
        return mask if inc else None
    if len(smoke)!=16: raise ValueError('smoke missing split/dataset/modality')
    for i in smoke.values():
        if verify(i,panel.read(i)) is None: raise ValueError('incomplete smoke input')
    rows,xs=defaultdict(list),defaultdict(list);t=time.perf_counter()
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        for i,s in enumerate(pool.map(panel.read,range(len(panel.rows)))):
            r=panel.rows[i];bits=verify(i,s);counter[r.split,r.dataset,r.modality,'planned_window_sites']+=1
            if bits is None: continue
            use=np.flatnonzero(bits); key=(r.modality,r.split); meta={k:getattr(r,k) for k in KEYS}
            xs[key].append(s['x'][use].transpose(0,2,1));rows[key].extend(dict(meta,patch=int(j)) for j in use)
            counter[r.split,r.dataset,r.modality,'complete_patches']+=len(use)
    elapsed=time.perf_counter()-t;frames={};waves={}
    for (m,split),parts in xs.items():
        key=(m,split);waves[key]=np.concatenate(parts);frames[key]=features(pd.DataFrame(rows[key]),waves[key],m,routers[m],blind[m])
    clockrows=[dict(split=k[0],dataset=k[1],crops=len(v['crops']),native_members=sorted(v['n_members']),offset_s_min=min(v['offset']),offset_s_max=max(v['offset']),positions={m:sorted(names) for m,names in v['positions'].items()}) for k,v in sorted(info.items())]
    return frames,waves,dict(smoke='16 complete rows; one per split/dataset/modality',clock=clockrows,io_workers=io_workers,io_seconds=elapsed,window_sites_per_s=len(panel.rows)/elapsed,counts=[dict(split=k[0],dataset=k[1],modality=k[2],metric=k[3],n=n) for k,n in sorted(counter.items())])
