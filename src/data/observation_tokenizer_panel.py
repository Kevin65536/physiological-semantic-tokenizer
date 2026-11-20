"""Bounded observation-only view of the verified G0 v2 development cache.

``build_panel(G0_ROOT)`` plans without opening arrays. ``panel.read(i)`` reads
one native position (EEG *or* HbO/HbR) in the train-scaled, uncentered G0 X
coordinate. The 15 disjoint 2-s slices can be encoded separately; they are not
independent subjects nor isolated raw signals (parent preprocessing is offline).
G0's supervised batch loader would read S/O and the teacher mask as well: this
narrower consumer reads only the three observation-side maps retained by G0.
No downstream/raw/test fallback, label lookup, scale fitting or file writes.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


G0_ROOT = (Path(__file__).resolve().parents[2] / 'experiments/runs/physiology_semantic_tokenizer'
           / 'dual_component_tokenizer/20261117_base_g0_v2')
G0_CONFIG_ID = 'f25b2181fcbf12f23d4db1d1'  # retained G0 resolved contract identity
G0_PLAN_ID = '34ab6d8d97a6f92fe2eb33d9'
G0_SCALES_ID = 'd8b6634e55fa73bbc1243650'
NATIVE_HZ = dict(eeg_fnirs_single_trial=200, refed=1000, simultaneous_eeg_nirs=200,
                 visual_cognitive_motivation=500)
DATASETS = tuple(NATIVE_HZ)
SPLITS = ['train', 'validation']
BRANCH = {'eeg': (200, 1, 'EEG_channels'), 'hb': (10, 2, 'fNIRS_positions')}


def _identity(value):
    # Same JSON identity as the G0 producer; used for folder & frozen scale joins.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def _safe(path):
    if path.resolve() != path or not path.is_relative_to(G0_ROOT):
        raise ValueError('path outside the exact completed G0 identity')
    return path


def _json(path):
    with _safe(path).open() as f:
        return json.load(f)


def _stream(path, array, consume):
    """Visit one item at a time in a large G0 JSON table; retain its header only."""
    decoder = json.JSONDecoder()
    with _safe(path).open() as f:
        buf = ''; i = 0

        def fill():
            nonlocal buf, i
            part = f.read(65536); buf = buf[i:] + part; i = 0
            return bool(part)

        def peek():
            nonlocal i
            while True:
                while i < len(buf) and buf[i].isspace(): i += 1
                if i < len(buf): return buf[i]
                if not fill(): return ''

        def take(ch):
            nonlocal i
            if peek() != ch: raise ValueError('malformed G0 JSON')
            i += 1

        def value():
            nonlocal i
            peek()
            while True:
                try:
                    obj, end = decoder.raw_decode(buf, i)
                except json.JSONDecodeError:
                    if fill(): continue
                    raise
                if end == len(buf) and fill(): continue
                i = end
                return obj

        take('{'); header = {}; seen = set()
        while peek() != '}':
            key = value(); take(':')
            if key in seen: raise ValueError('duplicate metadata field')
            seen.add(key)
            if key != array:
                header[key] = value()
            else:
                take('[')
                while peek() != ']':
                    consume(value())
                    if peek() == ']': break
                    take(',')
                take(']')
            if peek() == '}': break
            take(',')
        take('}')
        if peek() or array not in seen: raise ValueError('missing or trailing G0 metadata')
        return header


@dataclass(frozen=True)
class PanelLimits:
    subjects_per_split_dataset: int = 64
    records_per_subject: int = 3
    windows_per_record: int = 3
    positions_per_window: int = 2

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in asdict(self).values()):
            raise ValueError('all panel caps must be positive integers')


@dataclass(frozen=True, slots=True)
class Row:
    split: str
    dataset: str
    subject: str
    record: str
    parent: str
    start_s: int                     # crop offset within parent, both modalities
    native_start_sample: int         # absolute crop start on native EEG record grid
    native_rate_hz: int
    member_clocks: tuple             # (record_key, window_key, EEG start s, Hb start s)
    modality: str
    position: str
    position_index: int
    source_unit: str
    scale: float
    vx: tuple                       # G0 X target variance in scaled coordinates, per color
    weight: float                   # per split AND modality; not per independent observation
    g0_score_eligible: bool         # original supervised row, not an X validity mask
    num_positions: int
    parent_samples: int             # in prepared modality grid
    native_metadata: tuple          # existing task/event identities only, no invented labels


@dataclass(frozen=True)
class ObservationPanel:
    rows: tuple[Row, ...]
    counts: dict

    def read(self, i):
        """Read one selected crop; incomplete observations stay in the denominator.

        Only unobserved values are zero-filled. `score_mask` further excludes G0
        source-scoring edge support; neither mask depends on S/O/teacher validity.
        A corrupt/missing file raises; it is never silently replaced with another row.
        """
        row = self.rows[i]
        if row.split not in SPLITS or row.dataset not in DATASETS or row.modality not in BRANCH:
            raise ValueError('development observations only')
        rate, channels, _ = BRANCH[row.modality]
        folder = G0_ROOT/'cache'/_identity(row.parent)/row.modality
        keys = ('x', 'observation_valid', 'source_scoring')
        paths = [_safe(folder/(k+'.npy')) for k in keys]  # validate all before first mmap
        x, obs, score = [np.load(path, mmap_mode='r', allow_pickle=False) for path in paths]
        if (x.shape != (row.parent_samples, row.num_positions, channels) or obs.shape != x.shape
                or score.shape != (len(x),) or obs.dtype != bool or score.dtype != bool
                or x.dtype != np.float32):
            raise ValueError('G0 prepared shape or dtype drift')
        sl = slice(row.start_s*rate, (row.start_s+30)*rate)
        shape = (15, 2*rate, channels)
        native = x[sl, row.position_index].reshape(shape)
        mask = obs[sl, row.position_index].reshape(shape) & np.isfinite(native)
        values = np.where(mask, native, 0.) / np.float32(row.scale)
        if not np.isfinite(values).all(): raise ValueError('nonfinite scaled G0 input')
        score_mask = mask & score[sl].reshape(15, 2*rate, 1)
        return dict(row=row, x=values, mask=mask, score_mask=score_mask, available=bool(mask.all()),
                    patch_available=mask.all((1, 2)), patch_scorable=score_mask.all((1, 2)),
                    vx=np.asarray(row.vx))


def _parent(w):
    row = w['row']; meta = w['metadata']; src = meta['source']
    d, s, r, sp = row['dataset'], row['subject'], src['native_record'], row['split']
    key = row['key']; rate = src['rate']; origin = src['start_sample']
    if (sp not in SPLITS or d not in DATASETS or rate != NATIVE_HZ[d]
            or any(row[k] != source[k] for source in (meta, src) for k in ('dataset','subject','split','key'))
            or key.split('/')[:-1] != [d,s,r] or type(origin) is not int or origin < 0):
        raise ValueError('G0 native identity/split mismatch')
    members = []
    for member in src['members']:
        v = member['window']; e, h = v['eeg_start_s'], v['hb_start_s']; rec = member['record_key']
        if (rec.split('/')[:2] != [d,s] or round(e*rate) != origin
                or not math.isclose(10*h, round(10*h), abs_tol=1e-6, rel_tol=0)):
            raise ValueError('G0 member clock or identity mismatch')
        members.append((rec, v['key'], e, h))
    if not members: raise ValueError('missing native record clock')
    score = tuple(w['score_starts_s']); reg = tuple(w['regression_starts_s'])
    if (len(set(score)) != len(score) or (reg and sp != 'train')
            or any(type(i) is not int or i < 0 or (i+30)*rate > src['samples'] for i in score+reg)
            or any(i*rate < hi and (i+30)*rate > lo for i in score for lo,hi in w['blocked_samples'])):
        raise ValueError('score crop overlaps regression or exceeds parent')
    extra = tuple((k, v) for k in ('task','task_id','event_id','condition','session_id')
                  if (v := src.get(k, meta.get(k))) is not None and isinstance(v, (str,int,float)))
    return dict(key=key, group=(sp,d,s,r), rate=rate, origin=origin, duration=src['duration_s'],
                members=tuple(members), score=score, reg=reg, extra=extra, branches={},
                layout={m: tuple(meta.get(field, [])) for m,(_,_,field) in BRANCH.items()})


def _caps(values, n, seed, key):
    return sorted(values, key=lambda x: (_identity((seed, key, x)), x))[:n]


def build_panel(g0_root=None, *, limits=PanelLimits(), seed=17, datasets=DATASETS, positions=None):
    """Plan a deterministic, record-balanced *paired time support*, not paired sensors.

    Optional ``datasets`` and modality→native ``positions`` only narrow access.
    Restricting names does not imply EEG/Hb co-location. Prefer common named
    sensors across training records; never let validation choose a training sensor.
    Within each selected record sample one disjoint window per time stratum.
    Source failures, cap exclusions and missing paired support are not backfilled.
    """
    root = G0_ROOT if g0_root is None else Path(g0_root).absolute()
    datasets = tuple(datasets)
    if (root != G0_ROOT or _safe(root) != root or not datasets or len(set(datasets)) != len(datasets)
            or not set(datasets) <= set(DATASETS) or type(seed) is not int or not isinstance(limits, PanelLimits)):
        raise ValueError('invalid scope for the registered G0 development panel')
    if positions is not None and (set(positions) != set(BRANCH) or any(isinstance(v, str) or not v
            or len(v) != len(set(v)) or not all(isinstance(n,str) for n in v) for v in positions.values())):
        raise ValueError('optional fixed EEG/Hb positions must be distinct named lists per modality')
    summary = _json(root/'summary.json'); verification = _json(root/'verification.json')
    scales = _json(root/'scales.json')
    if (summary['status'] != 'completed' or verification['status'] != 'passed' or verification['issues']
            or scales['schema'] != 'dual_component_base_scales_v1' or scales['fitting_split'] != 'train'):
        raise ValueError('exact completed, verified train-scaled G0 required')
    parents = {}; subjects = {}; bounds = defaultdict(list); blocked = defaultdict(list)
    strata = defaultdict(Counter); common = {}

    def plan_row(w):
        p = _parent(w); key = p['key']; group = p['group']; sp,d,s,r = group
        if key in parents or subjects.setdefault((d,s),sp) != sp: raise ValueError('duplicate or cross-split parent')
        parents[key] = p
        for dst, starts in [(bounds,p['score']), (blocked,p['reg'])]:
            for start in starts:
                tick = p['origin'] + start*p['rate']; dst[group].append((tick,tick+30*p['rate']))
        if d in datasets: strata[sp+'/'+d]['planned_parents'] += 1

    plan = _stream(root/'plan.json', 'windows', plan_row)
    if plan['schema'] != 'dual_component_base_plan_v1' or plan['allowed_splits'] != SPLITS:
        raise ValueError('unapproved G0 plan')
    declared = {}
    for s in plan['subjects']:
        key = s['dataset'], s['subject']; sp = s['split']
        if sp not in SPLITS or key in declared or key[0] not in DATASETS: raise ValueError('invalid subject split registry')
        declared[key] = sp
    if any(declared.get(k) != v for k,v in subjects.items()): raise ValueError('subject plan disagrees with registry')
    for group, intervals in bounds.items():
        ordered = sorted(intervals)
        if (any(a[1] > b[0] for a,b in zip(ordered,ordered[1:]))
                or any(lo < end and hi > begin for lo,hi in ordered for begin,end in blocked[group])):
            raise ValueError('overlapping G0 score/regression support')

    def prepared(row):
        key = row['parent']; p = parents.get(key); m = row['modality']
        group = tuple(row[k] for k in ('split','dataset','subject','record'))
        if (not p or p['group'] != group or m not in BRANCH or m in p['branches']
                or row['native_start_sample'] != p['origin'] or row['native_rate_hz'] != p['rate']
                or (row['rate'],row['channels']) != BRANCH[m][:2]
                or Path(row['folder']) != root/'cache'/_identity(key)/m):
            raise ValueError('prepared branch identity/scope/folder mismatch')
        names = tuple(row['positions']); sp,d,_,_ = group
        if names != p['layout'][m] or len(names) != len(set(names)):
            raise ValueError('prepared native position identity mismatch')
        complete = {}
        for item in row['inputs']['score']:
            i = item['start_s']; cols = item['positions']
            if (i not in p['score'] or i in complete or any(type(j) is not int or not 0<=j<len(names) for j in cols)):
                raise ValueError('crop outside original score support')
            complete[i] = frozenset(names[j] for j in cols)
        if set(complete) != set(p['score']): raise ValueError('prepared score starts differ from G0 plan')
        frozen = scales['positions'][m].get(d,{})
        allowed = tuple(n for n in names if n in frozen and (positions is None or n in positions[m]))
        p['branches'][m] = (names, allowed, complete, row['source_unit'])
        if d in datasets:
            ct = strata[sp+'/'+d]; ct[m+'_prepared_parents'] += 1
            ct[m+'_candidate_position_crops'] += len(names)*len(complete)
            ct[m+'_without_training_scale'] += sum(n not in frozen for n in names)*len(complete)
            ct[m+'_named_scaled_position_crops'] += len(allowed)*len(complete)
        if sp == 'train' and complete:
            common.setdefault((d,m), set(allowed)).intersection_update(allowed)

    manifest = _stream(root/'cache/manifest.json', 'rows', prepared)
    if (manifest['schema'] != 'dual_component_base_cache_v1' or manifest['status'] != 'completed'
            or manifest['cfg_identity'] != G0_CONFIG_ID or manifest['plan_identity'] != G0_PLAN_ID
            or manifest['scales_identity'] != G0_SCALES_ID or G0_SCALES_ID != _identity(scales)
            or manifest['allowed_splits'] != SPLITS):
        raise ValueError('G0 contract/scale identity changed')
    # EEG and optical member clocks are not interchangeable, especially across
    # Visual probes. Guard the recorded Hb ticks too (native EEG checks above).
    hb_reg = defaultdict(list)
    for p in parents.values():
        for start in p['reg']:
            for rec,_,_,h in p['members']:
                lo = round(h*10)+10*start; hb_reg[rec].append((lo,lo+300))
    crops = defaultdict(list)
    for p in parents.values():
        sp,d,s,r = p['group']
        if d not in datasets: continue
        ct = strata[sp+'/'+d]; ct['g0_score_windows'] += len(p['score'])
        for start in p['score']:
            if not all(m in p['branches'] and p['branches'][m][1] for m in BRANCH):
                ct['missing_paired_branch_or_scaled_position'] += 1; continue
            ticks = [(rec, round(h*10)+10*start) for rec,_,_,h in p['members']]
            if any(lo < end and lo+300 > begin for rec,lo in ticks for begin,end in hb_reg[rec]):
                ct['hb_regression_overlap'] += 1; continue
            crops[p['group']].append((p['origin']+start*p['rate'],p['key'],start,ticks))
    tree = defaultdict(lambda: defaultdict(dict))
    for (sp,d,s,r), windows in crops.items():
        ends = {}; ok = []
        for _,key,start,ticks in sorted(windows):
            if any(ends.get(rec,-1) > lo for rec,lo in ticks):
                strata[sp+'/'+d]['hb_score_overlap'] += 1; continue
            ok.append((key,start)); ends.update((rec,lo+300) for rec,lo in ticks)
        tree[sp,d][s][r] = ok
    rows = []
    for sp in SPLITS:
        for d in sorted(datasets):
            groups = tree[sp,d]; ct = strata[sp+'/'+d]
            if not groups: raise ValueError(f'no paired eligible {sp} subject/record support in {d}')
            ss = _caps(groups, limits.subjects_per_split_dataset, seed, (sp,d))
            ct.update(candidate_subjects=len(groups), selected_subjects=len(ss),
                      candidate_records=sum(map(len,groups.values())),
                      candidate_windows=sum(len(v) for rr in groups.values() for v in rr.values()))
            for s in ss:
                rr = _caps(groups[s], limits.records_per_subject, seed, (sp,d,s)); ct['selected_records'] += len(rr)
                for r in rr:
                    ws = groups[s][r]; n = min(limits.windows_per_record,len(ws)); ct['selected_windows'] += n
                    for j in range(n):
                        subset = ws[j*len(ws)//n:(j+1)*len(ws)//n]
                        key,start = _caps(subset, 1, seed, (sp,d,s,r,j))[0]; p = parents[key]
                        clocks = tuple((rec,w,e+start,h+start) for rec,w,e,h in p['members'])
                        for m in BRANCH:
                            names,allowed,complete,unit = p['branches'][m]; ranked = sorted(allowed,
                                key=lambda a: (a not in common.get((d,m),()), a not in complete[start], _identity((seed,d,m,a))))
                            chosen = ranked[:limits.positions_per_window]; ct[m+'_selected_position_crops'] += len(chosen)
                            weight = 1/(len(datasets)*len(ss)*len(rr)*n*len(chosen))
                            for name in chosen:
                                ref = scales['positions'][m][d][name]; scale = float(ref['scale']); vx = tuple(ref['target_variance']['x'])
                                if not math.isfinite(scale) or scale<=0 or len(vx)!=BRANCH[m][1] or not all(math.isfinite(v) and v>0 for v in vx):
                                    raise ValueError('invalid frozen G0 training scale or X variance')
                                rows.append(Row(sp,d,s,r,key,start,p['origin']+start*p['rate'],p['rate'],clocks,
                                    m,name,names.index(name),unit,scale,vx,weight,name in complete[start],len(names),
                                    round(p['duration']*BRANCH[m][0]),p['extra']))
                                ct[m+'_selected_without_g0_supervision'] += int(name not in complete[start])
    counts = dict(g0=dict(root=str(root), config_identity=G0_CONFIG_ID, scales_identity=manifest['scales_identity'],
                          plan_identity=manifest.get('plan_identity'), regression_inputs=plan['regression_inputs'],
                          regression_crops_excluded=plan['regression_crops_excluded'],
                          duplicate_crops_removed=plan['duplicate_training_crops_removed'],
                          retained_source_failures=summary['retained_source_failures'],
                          planned_counts=verification.get('planned_counts',{}), verified_counts=verification.get('counts',{})),
                  selection=dict(seed=seed, limits=asdict(limits), datasets=sorted(datasets),
                                 fixed_positions=None if positions is None else {m:list(v) for m,v in positions.items()},
                                 common_train_positions={d:{m:sorted(common.get((d,m),())) for m in BRANCH} for d in datasets}),
                  strata={k:dict(v) for k,v in sorted(strata.items())})
    return ObservationPanel(tuple(rows), counts)
