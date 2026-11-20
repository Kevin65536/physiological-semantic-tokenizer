"""Frozen observation words vs *external trial labels*, not a classifier.

Single-Trial MA is compared with explicit BL in the same subject and session.
The event owner is the b277a356 task probe: downstream-v2 contributes labels
and both marker clocks; only G0 supplies X. No new reader, codec fitting,
validation-derived word names, or signal export. The caller owns the identity
of the two saved main dictionaries and the persistence of the returned evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from src.analysis import observation_task_probe as task
from src.analysis.observation_task_uncertainty import _interval as _small_subject_interval


_SESSIONS = ('session_01', 'session_03', 'session_05')
_PERIODS = {'all': (0, 4), 'early': (0, 2), 'late': (2, 4)}  # indices in the four-cell owner input
_BANDS = {'posterior_alpha': ('eeg_rel_8_13hz', ('P3', 'P4'), 'decrease'),
          'frontal_theta': ('eeg_rel_4_8hz', ('F3', 'F4'), 'increase')}
_HB_WORD = 'hb_joint_rise_fall_occupancy'
_PRIMARY = _HB_WORD + '/all'
_BOOT_DRAWS = 8192


def _check(ok, message):
    if not ok:
        raise ValueError(message)


def _book(codec, modality):
    """Define words from TRAIN conditional physical means, never routing/slot order."""
    _check(codec.modality == modality and codec.routing == 'pattern' and codec.n_patterns == 16
           and codec.seed == 17, 'the original main pattern-route K=16, seed=17 codecs are required')
    _check(codec.rate('prototype')['bits_per_item'] == 4, 'fitted four-bit prototype required')
    counts, centers = np.asarray(codec.pattern_counts_), np.asarray(codec.pattern_centers_)
    names = list(codec.pattern_feature_names)
    _check(len(set(names)) == len(names) and counts.shape == (16,) and counts.dtype.kind in 'iu'
           and (counts >= 0).all() and (counts > 0).any() and centers.shape == (16, len(names))
           and centers.dtype.kind in 'fiu' and np.isfinite(centers).all(), 'invalid learned dictionary profile')
    active = counts > 0
    groups = {}

    def group(key, mask, **rule):
        take = active & mask  # unoccupied slots have only the codec's fallback mean, not a learned word
        groups[key] = dict(**rule, ids=np.flatnonzero(take).tolist(), n_words=int(take.sum()),
                           training_assignments=int(counts[take].sum()))

    if modality == 'hb':
        o, r = centers[:, names.index('hbo_slope_per_s')], centers[:, names.index('hbr_slope_per_s')]
        group(_HB_WORD, (o > 0) & (r < 0), rule='training centroid: hbo_slope_per_s > 0 AND hbr_slope_per_s < 0')
    else:
        for band, (feature, _, _) in _BANDS.items():
            x = centers[:, names.index(feature)]
            reference = float(np.average(x[active], weights=counts[active]))
            # Fixed tolerance for a numerically tied spectral fraction, not a validation quantile.
            group(band + '_enriched_occupancy', x > reference + 1e-12, feature=feature,
                  training_mean=reference, tie_tolerance=1e-12,
                  rule='centroid band fraction > assignment-count-weighted training mean + tie_tolerance')
    return dict(n_patterns=16, seed=17, pattern_bits=4, feature_names=names,
                training_counts=counts.tolist(), training_physical_centroids=centers.tolist(), groups=groups)


def _metrics():
    defs = {}

    def add(name, sites, source, quantity, unit='fraction', direction='increase'):
        defs[name] = dict(positions=list(sites), source=source, quantity=quantity, unit=unit,
                          expected_MA_minus_BL=direction)

    hb = task.POSITIONS['hb']
    add(_HB_WORD, hb, 'pattern_id', 'occupancy of jointly rising-HbO / falling-HbR slope words')
    add('hb_x_joint_rise_fall_fraction', hb, 'G0_X', 'fraction of cells with original HbO slope > 0 AND HbR slope < 0')
    for source, label in [('pattern_id', 'centroid'), ('G0_X', 'x')]:
        add('hb_' + label + '_signed_opposition', hb, source, 'HbO slope minus HbR slope (not a conjunction)',
            'G0 scaled-X units/s')
    for band, (_, sites, direction) in _BANDS.items():
        add(band + '_enriched_occupancy', sites, 'pattern_id', 'occupancy of training-relative band-enriched words',
            direction=direction)
        for source, label in [('pattern_id', 'centroid'), ('G0_X', 'x')]:
            add(band + '_' + label + '_fraction', sites, source, 'relative band power (0.5–100 Hz denominator)',
                direction=direction)
    return defs


def _cells(codec, wave, book, n):
    m = codec.modality
    shape = (n, len(task.POSITIONS[m]), 4, 1 if m == 'eeg' else 2, 2 * task.RATES[m])
    _check(wave.shape == shape, 'incomplete or misaligned event waveform grid')
    x = wave.reshape(-1, *shape[-2:])
    ids, attrs = [], []
    for start in range(0, len(x), 256):
        batch = x[start:start + 256]
        # The owner rejects non-integer IDs, out-of-range values and sidecars.
        ids.append(task._ids(codec, batch).argmax(axis=1))
        f = np.asarray(codec.pattern_features(batch))  # original-X comparator only
        _check(f.shape == (len(batch), len(book['feature_names'])) and np.isfinite(f).all(), 'invalid raw-X attributes')
        attrs.append(f)
    ids = np.concatenate(ids).reshape(shape[:3])
    _check((np.asarray(book['training_counts'])[ids] > 0).all(), 'encoded ID has no training centroid support')
    return ids, np.concatenate(attrs).reshape(*shape[:3], -1)


def _event_values(kept, waves, codecs, books, definitions):
    """Reduce ROI positions and fixed cells equally; retain IDs for the exact occupancy numerator."""
    keys = [name + '/' + period for name in definitions for period in _PERIODS]
    n = len(kept)
    if not n:
        _check(not waves, 'waveforms without included events')
        return keys, np.empty((0, len(keys)))
    columns = {}
    for m, codec in codecs.items():
        b = books[m]
        ids, x = _cells(codec, waves[m], b, n)
        z = np.asarray(b['training_physical_centroids'])[ids]  # ID alone -> training property lookup
        for event, words in zip(kept, ids, strict=True):
            event.setdefault('pattern_ids', {})[m] = words.tolist()  # [named position, time cell]
        names = b['feature_names']
        cells = {}
        if m == 'hb':
            i, j = names.index('hbo_slope_per_s'), names.index('hbr_slope_per_s')
            cells[_HB_WORD] = np.isin(ids, b['groups'][_HB_WORD]['ids'])
            cells['hb_x_joint_rise_fall_fraction'] = (x[..., i] > 0) & (x[..., j] < 0)
            cells['hb_centroid_signed_opposition'] = z[..., i] - z[..., j]
            cells['hb_x_signed_opposition'] = x[..., i] - x[..., j]
        else:
            for band, (feature, _, _) in _BANDS.items():
                word = band + '_enriched_occupancy'
                cells[word] = np.isin(ids, b['groups'][word]['ids'])
                cells[band + '_centroid_fraction'] = z[..., names.index(feature)]
                cells[band + '_x_fraction'] = x[..., names.index(feature)]
        for name, v in cells.items():
            pos = [task.POSITIONS[m].index(p) for p in definitions[name]['positions']]
            for period, (lo, hi) in _PERIODS.items():
                columns[name + '/' + period] = v[:, pos, lo:hi].mean(axis=(1, 2), dtype=np.float64)
    matrix = np.column_stack([columns[key] for key in keys])
    for e, row in zip(kept, matrix, strict=True):
        e['values'] = dict(zip(keys, row.tolist(), strict=True))
    return keys, matrix


def _ci(deltas, split):
    """Pointwise percentiles of subject means, conditional on the frozen eligible trials."""
    n, p = deltas.shape
    if n < 2:
        return [None] * p, dict(method='not_estimable', draws=0, seed=None)
    if split == 'validation':  # at most six independent subjects; no Monte Carlo error here
        intervals = [_small_subject_interval(column)[0] for column in deltas.T]
        return intervals, dict(method='exact_empirical_subject_percentile', draws=n**n, seed=None)
    # Use the same subject draws for every exploratory arm; no cell/trial resampling.
    rng = np.random.default_rng(task.SEED)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=_BOOT_DRAWS) / n
    bounds = np.quantile(weights @ deltas, [.025, .975], axis=0, method='inverted_cdf')
    return bounds.T.tolist(), dict(method='MC_subject_percentile', draws=_BOOT_DRAWS, seed=task.SEED)


def _sign_flip(differences):
    """Exact two-sided sign symmetry reference; labels were not randomized by this test."""
    n = len(differences)
    _check(0 < n <= 6, 'sign reference restricted to paired validation subjects')
    observed = abs(sum(differences))  # exact rational subject occupancy contrasts, including ties
    exceed = sum(abs(sum(sign * d for sign, d in zip(signs, differences, strict=True))) >= observed
                 for signs in product((-1, 1), repeat=n))
    return dict(n_subjects=n, n_sign_vectors=2**n, at_least_as_extreme=exceed, two_sided_p=exceed / 2**n,
                assumption='Independent subject contrasts symmetric about zero under the null; ties included. '
                           'Not a trial-label randomization test.')


def _counts(rows):
    counts = Counter(e['condition'] for e in rows)
    return {cls: counts[cls] for cls in task.CLASSES}


def _word_support(events, definitions, books):
    """Exact labelled trial and site-cell denominators on the *paired-session* subset."""
    support = {}
    for m, book in books.items():
        for key, group in book['groups'].items():
            positions = [task.POSITIONS[m].index(p) for p in definitions[key]['positions']]
            codes = set(group['ids'])
            by_class = {}
            for cls in task.CLASSES:
                rows = [e for e in events if e['condition'] == cls]
                hits = [sum(code in codes for p in positions for code in e['pattern_ids'][m][p]) for e in rows]
                by_class[cls] = dict(hits=sum(hits), cells=len(rows) * len(positions) * 4,
                                     events_with_hit=sum(v > 0 for v in hits), events=len(rows),
                                     subjects_with_hit=len({e['subject'] for e, v in zip(rows, hits, strict=True) if v}))
            support[key] = by_class
    return support


def _summarize(split, expected, events, index, matrix, keys, definitions, books):
    groups = defaultdict(list)
    for event in events:
        if event['split'] == split:
            groups[event['subject'], event['record']].append(event)
    primary_col = keys.index(_PRIMARY)
    primary_cells = 4 * len(task.POSITIONS['hb'])  # identical denominator in every admitted trial
    paired, people, sessions, ma, bl = [], [], [], [], []
    subject_deltas, primary_deltas = [], []
    for s in expected:
        sma, sbl, session_primary = [], [], []
        for r in _SESSIONS:
            rows = groups[s, r]
            kept = [e for e in rows if e['status'] == 'included']
            counts = _counts(kept)
            both = all(counts.values())
            session = dict(subject=s, record=r, planned_by_condition=_counts(rows), retained_by_condition=counts,
                           paired=both, status_counts=dict(Counter(e['status'] for e in rows)), ma_minus_bl=None)
            for e in rows:
                e['used_in_contrast'] = both and e['status'] == 'included'
            if both:
                paired.extend(kept)
                arms = [matrix[[index[e['id']] for e in kept if e['condition'] == cls]] for cls in task.CLASSES]
                a, b = (arm.mean(axis=0) for arm in arms)
                # Recover integer hits from the 0..12/12 grid before subtracting. Unequal
                # class sizes or cancelling sessions must not turn exact ties into evidence.
                pa, pb = (Fraction(int(np.rint(arm[:, primary_col] * primary_cells).sum()), len(arm) * primary_cells)
                          for arm in arms)
                a[primary_col], b[primary_col] = float(pa), float(pb)
                diff = a - b; diff[primary_col] = float(pa - pb)
                session_primary.append(pa - pb)
                sma.append(a); sbl.append(b)
                session['ma_minus_bl'] = dict(zip(keys, diff.tolist(), strict=True))
            sessions.append(session)
        # A single-class session contributes *nothing*, even if another has the other class.
        delta = None
        if sma:
            ma.append(np.mean(sma, axis=0)); bl.append(np.mean(sbl, axis=0))
            delta = np.mean(np.subtract(sma, sbl), axis=0)
            primary_deltas.append(sum(session_primary) / len(session_primary))
            delta[primary_col] = float(primary_deltas[-1])
            subject_deltas.append(delta)
        people.append(dict(subject=s, n_paired_sessions=len(sma), paired_events_by_condition=_counts(
            [e for e in paired if e['subject'] == s]),
            ma_minus_bl=None if delta is None else dict(zip(keys, delta.tolist(), strict=True))))
    n = len(ma)
    deltas = np.asarray(subject_deltas).reshape(n, len(keys))
    complete = n > 0 and n == len(expected)
    cis, bootstrap = _ci(deltas, split)
    estimates = {}
    for j, (key, ci) in enumerate(zip(keys, cis, strict=True)):
        d = deltas[:, j]
        mean = float(d.mean()) if n else None
        estimates[key] = dict(paired_mean_MA=float(np.mean([a[j] for a in ma])) if n else None,
                              paired_mean_BL=float(np.mean([b[j] for b in bl])) if n else None,
                              paired_mean_MA_minus_BL=mean, paired_subject_ci95=ci,
                              full_cohort_mean_MA_minus_BL=mean if complete else None,
                              full_cohort_ci95=ci if complete else None,
                              subject_direction=dict(positive=int((d > 0).sum()), negative=int((d < 0).sum()),
                                                     tie=int((d == 0).sum()), undefined=len(expected)-n))
    rows = [e for e in events if e['split'] == split]
    retained = [e for e in rows if e['status'] == 'included']
    return dict(n_registered_subjects=len(expected), n_paired_subjects=n, full_cohort_paired=complete,
                n_registered_subject_sessions=len(sessions), n_paired_sessions=sum(v['paired'] for v in sessions),
                planned_events=len(rows), retained_events=len(retained), paired_events=len(paired),
                planned_by_condition=_counts(rows), retained_by_condition=_counts(retained),
                paired_by_condition=_counts(paired), subjects=people, sessions=sessions, estimates=estimates,
                paired_word_support_all=_word_support(paired, definitions, books),
                paired_subject_bootstrap=bootstrap,
                primary_sign_flip=(_sign_flip(primary_deltas) if split == 'validation' and complete else None))


def run_event_relations(*, cache_root, g0_root, eeg_codec, hb_codec) -> dict:
    """Return JSON-safe event evidence. Supply the original saved G0-train pattern codecs.

    The sole primary is the six-subject validation MA−BL contrast in prefrontal
    joint-slope word occupancy over all four cells. Training is descriptive.
    Early/late, original-X comparisons and both EEG families are exploratory;
    no fitting, threshold search, new data access route, or files are written.
    """
    _check(cache_root is not None and g0_root is not None, 'explicit downstream and G0 roots required')
    _check(task.SPLITS == ('train', 'validation') and task.CLASSES == ('MA', 'BL') and task.OFFSET == 2
           and task.SECONDS == 8 and task.DATASET == 'eeg_fnirs_single_trial' and task.TASK == 'mental_arithmetic'
           and task.RATES == {'eeg': 200, 'hb': 10} and task.POSITIONS == {
               'eeg': ('F3', 'F4', 'P3', 'P4'), 'hb': ('AF3Fp1', 'FpzAFz', 'AF4Fp2')}, 'b277a356 event contract drift')
    codecs = {'eeg': eeg_codec, 'hb': hb_codec}
    books = {m: _book(c, m) for m, c in codecs.items()}  # fully specified before any event/validation access
    definitions = _metrics()
    events, expected, exclusions = task._events(cache_root)
    _check(len(expected['validation']) == 6, 'expected the registered six validation subjects before G0 array access')
    waves, scales = task._event_inputs(events, expected, task._panel(g0_root))
    kept = [e for e in events if e['status'] == 'included']
    keys, values = _event_values(kept, waves, codecs, books, definitions)
    index = {e['id']: i for i, e in enumerate(kept)}
    identities = sorted((e['split'], e['subject'], e['record'], e['event_index'], e['condition']) for e in kept)
    digest = hashlib.sha256(json.dumps(identities, separators=(',', ':')).encode()).hexdigest()
    by_split = {sp: _summarize(sp, expected[sp], events, index, values, keys, definitions, books) for sp in task.SPLITS}
    val = by_split['validation']
    group = books['hb']['groups'][_HB_WORD]
    observed = val['paired_word_support_all'][_HB_WORD]
    hits, cells = sum(v['hits'] for v in observed.values()), sum(v['cells'] for v in observed.values())
    support = ('empty_training_word_set' if not group['n_words'] else 'no_paired_events' if not cells else
               'no_validation_hits' if not hits else 'all_validation_cells_qualify' if hits == cells else 'observed')
    return dict(schema='frozen_observation_token_events_v1', dataset=task.DATASET, task=task.TASK,
        status='completed' if val['full_cohort_paired'] else 'not_estimable_for_full_validation_cohort',
        primary=dict(split='validation', key=_PRIMARY, expected_MA_minus_BL='increase', word_support=support),
        downstream_source=str(Path(cache_root).resolve()), g0_source=str(Path(g0_root).resolve()),
        positions={m: list(v) for m, v in task.POSITIONS.items()}, rates_hz=task.RATES.copy(),
        periods_s={p: [task.OFFSET + 2*lo, task.OFFSET + 2*hi] for p, (lo, hi) in _PERIODS.items()},
        metrics=definitions, dictionaries=books, scale_by_modality_position=scales,
        planned_events=len(events), retained_events=len(kept), kept_keys_sha256=digest, downstream_cache_exclusions=exclusions,
        events=events, by_split=by_split,
        contract=dict(events='Publisher MA subtraction vs explicitly labelled BL, not pre-event or intertrial rest. '
                      'The verified downstream-v2 owner supplies event identity and separate EEG/Hb clocks only.',
            input='Same b277a356 disjoint G0 30-s score-crop join, all seven positions and all score/observation ticks '
                  'required in [marker+2,+10). G0 X/frozen position SD only; no downstream arrays, S/O or teacher.',
            variables='Four 2-s cells; equally average declared sites and time cells per trial. EEG fractions are '
                      'codec-native untapered non-DC power, not absolute alpha/theta power. Hb uses two OLS slopes '
                      'in the same scaled-X units/s; neither levels nor an absolute concentration difference.',
            priors='Directional hypotheses about external task labels, not physiology assigned to initial IDs. '
                   'Use only supported original TRAIN physical centroids. Keep sparse/empty word groups and their '
                   'zero occupancies; signed opposition is always a separate comparison, never a replacement primary.',
            pairing='Condition balancing without subsampling: separately mean all retained MA and all BL events '
                    'within the same session, subtract BL from MA, average eligible sessions per subject, then '
                    'subjects equally. Paired events are all trials in qualifying sessions, not one-to-one matches. '
                    'Never pair across sessions. Incomplete subjects remain in denominators; paired-subset estimates '
                    'are distinguished from the full registered-cohort estimand.',
            uncertainty='Pointwise 95% subject-bootstrap percentiles (inverted empirical CDF), conditional on the '
                        'frozen data/words/support; exact n**n ordered draws in validation, fixed-seed MC in train. '
                        'No interval for fewer than two paired subjects. Only the complete validation primary has '
                        'a two-sided sign-symmetry test. Other phases/metrics and all train results are exploratory, '
                        'not multiplicity controlled. A zero-width interval for a constant endpoint is not evidence of equivalence.',
            limits='Offline, previously examined public development cohort; native EEG and parent-referenced, '
                   'subject-scaled Hb (including validation). Possible cue, systemic, eye/muscle, order and selection '
                   'effects. These contrasts are neither independent biological replicates per cell nor proof of '
                   'cortical source, causality, real-time decoding, or protected/test generalization.'))
