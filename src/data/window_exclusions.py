"""Read-time exclusions for the four-dataset V3 development census, in 200 Hz ticks."""
import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

VERSION = 'v3_window_exclusions_v1'


@lru_cache(maxsize=1)
def _by_record():
    index = defaultdict(list)
    path = Path(__file__).with_name('assets') / (VERSION + '.csv')
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[k] for k in ('dataset', 'split', 'subject', 'record'))
            lo, hi, native = (int(row[k]) for k in ('start_200', 'stop_200', 'native_start_200'))
            modality = row['modality']
            reasons = ('line50',) if modality == 'eeg' else ('missing_all', 'constant', 'near_flat')
            if (row['reason'] not in reasons or modality not in ('eeg', 'hb') or hi <= lo
                    or key[1] not in ('train', 'validation') or not all(key)):
                raise ValueError(f'Invalid {VERSION} marker: {row}')
            index[key].append((modality, row['position'], lo, hi, native))
    return dict(index)


def paired_window_excluded(dataset, split, subject, record, eeg_start, hb_start, seconds):
    """Compare the actual native EEG/Hb sample intervals used by the synchronized reader."""
    # Match the native rounding of downstream_input_cache._slice_window (onset in ms).
    starts = {'eeg': round((eeg_start * 1000) / 1000 * 200),
              'hb': 20 * round((hb_start * 1000) / 1000 * 10)}
    for modality, _, lo, hi, native in _by_record().get((dataset, split, subject, record), ()):
        start = starts[modality]
        stride = 1 if modality == 'eeg' else 20
        length = stride * round(seconds * (200 // stride))
        if start < native + hi - lo and start + length > native:
            return True
    return False


def filter_synchronized_samples(samples, native_records):
    # Probe views of one Visual event share a dependency. A REFED dependency
    # can be a whole video, so its individual window start must also match.
    def window_key(row):
        return row['dependency_id'], row['eeg_start_s'], row['duration_s']

    excluded = {window_key(r) for r in samples
                if paired_window_excluded(r['dataset'], r['split'], r['subject'],
                                          native_records[r['join_key']], r['eeg_start_s'],
                                          r['hb_start_s'], r['duration_s'])}
    return [r for r in samples if window_key(r) not in excluded]


def local_spans(dataset, split, subject, record, modality, positions, origin, duration):
    """Return half-open exclusions relative to this parent, retaining its native column order."""
    by_name = {name: i for i, name in enumerate(positions)}
    spans = []
    for kind, name, lo, hi, _ in _by_record().get((dataset, split, subject, record), ()):
        if kind == modality and name in by_name and origin < hi and origin + duration > lo:
            spans.append((by_name[name], max(0, lo-origin), min(duration, hi-origin)))
    return spans


def blocked_positions(spans, start_s):
    lo = start_s * 200
    return {p for p, a, b in spans if lo < b and a < lo + 6000}


def apply_to_so(meta, arrays):
    src = meta['source']
    for modality, prefix, names, stride in (
        ('eeg', 'eeg', meta['EEG_channels'], 1),
        ('hb', 'fnirs', meta['fNIRS_positions'], 20),
    ):
        spans = local_spans(meta['dataset'], meta['split'], meta['subject'], src['native_record'],
                            modality, names, src['start_sample'], src['samples'])
        for position, lo, hi in spans:
            cut = slice(lo // stride, (hi + stride - 1) // stride)
            for k in ('observation_valid', 'teacher_valid'):
                arrays[prefix + '_' + k][cut, position] = False
            for k in ('x', 's', 'o'):
                arrays[prefix + '_' + k][cut, position] = 0
