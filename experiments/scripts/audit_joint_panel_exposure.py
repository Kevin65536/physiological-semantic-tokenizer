"""Verify recorded scoring identities against the frozen development and confirmation panels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def keys(a):
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[1] != 3 or a.dtype.kind not in 'iu':
        raise ValueError('G0 row, start and position identity required')
    return {tuple(item) for item in a.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    panel = args.run / 'panels.npz'
    digest = hashlib.sha256(panel.read_bytes()).hexdigest()
    if digest != json.loads((args.run/'panel.json').read_text())['panel_sha256']:
        raise ValueError('frozen panel identity changed')
    with np.load(panel, allow_pickle=False) as data:
        sets = {m: {role: keys(data[m+'_'+role]) for role in ('screen','development','confirm')}
                for m in ('eeg','hb')}
    confirmation = sets['eeg']['confirm'] | sets['hb']['confirm']
    records = []
    failed = []
    for name in ('metrics.npz','spatial_metrics.npz','quantized_slots_metrics.npz','calibrated_slots_metrics.npz'):
        for path in sorted(args.run.rglob(name)):
            with np.load(path, allow_pickle=False) as data:
                if 'examples' not in data.files:
                    records.append(dict(path=str(path.relative_to(args.run)),scope='unclassified_no_G0_triples'))
                    continue
                raw = data['examples']
            seen = keys(raw)
            overlaps = sorted(seen & confirmation)
            matches = [m+'_'+role for m, roles in sets.items() for role, panel_keys in roles.items() if seen==panel_keys]
            records.append(dict(path=str(path.relative_to(args.run)), recorded_rows=len(raw), unique_rows=len(seen),
                                exact_panel=matches, confirmation_overlap=len(overlaps)))
            if overlaps:
                failed.append(dict(path=str(path.relative_to(args.run)),count=len(overlaps),examples=overlaps[:5]))
    if not records:
        raise ValueError('expected retained score arrays below specified run')
    with args.output.open('x') as f:
        json.dump(dict(status='failed' if failed else 'passed', panel_sha256=digest,
                       scope='retained metric archives in this run with the four named filenames; sample identity review only',
                       recorded_arrays=len(records),failed=failed,records=records),f,indent=2)
        f.write('\n')
    print(len(records),'metric archives;',len(failed),'with confirmation overlap')
    if failed: raise SystemExit(1)


if __name__=='__main__':
    main()
