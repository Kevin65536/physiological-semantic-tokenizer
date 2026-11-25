"""Diagnostic sensitivity to moving parent edges and to the onset of a finite alpha burst."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.scripts.check_continuous_so_targets import (
    OUT,
    PRIOR,
    ORDER,
    old,
    projection,
    complete_blocks,
    project_blocks,
)


def burst():
    t = np.arange(18000) / 200
    rows = []
    for onset in (39.5, 40.0):
        use = (t >= onset) & (t < onset + 1)
        for hz in [10, 11, 11.25, 11.5]:
            x = np.zeros((len(t), 8))
            x[use] = np.hanning(use.sum())[:, None] * np.sin(
                2 * np.pi * hz * (t[use] - onset)[:, None] + np.arange(8) * np.pi / 8
            )
            blocks, _ = complete_blocks(x, 0)
            for name, y in [('block', project_blocks(blocks, ORDER[:32])), ('continuous', projection(x))]:
                energy = np.sum(y * y, 0)
                for phase in range(8):
                    v = y[:, phase] ** 2
                    rows.append(
                        dict(
                            onset_s=onset,
                            hz=hz,
                            phase_rad=phase * np.pi / 8,
                            arm=name,
                            energy_retained=energy[phase] / np.sum(x[:, phase] ** 2),
                            before_onset_fraction=v[t < onset].sum() / energy[phase],
                            after_offset_fraction=v[t >= onset + 1].sum() / energy[phase],
                        )
                    )
    return pd.DataFrame(rows)


def parent_change():
    rows = []
    for c in old.read(PRIOR / 'plan.json')['cases']:
        with np.load(old.PARENT / 'targets' / str(c['i']) / 'baseline.npz') as f:
            s = f['v3'][:-20]
        origin = c['source']['start_sample']
        full = projection(s)
        for begin, end in [(0, 12000), (len(s) - 12000, len(s))]:
            piece = s[begin:end]
            new = projection(piece)
            # Old block targets agree wherever both parent supports contain the full record-anchored block.
            b, sl = complete_blocks(piece, origin + begin)
            other = project_blocks(b, ORDER[:32])
            fb, fs = complete_blocks(s, origin)
            reference = project_blocks(fb, ORDER[:32])
            offset = begin + sl.start - fs.start
            np.testing.assert_allclose(other, reference[offset : offset + len(other)], atol=1e-10)
            rel = np.arange(len(piece))
            dist = (len(piece) - 1 - rel if begin == 0 else rel) / 200
            for label, mask in [('first_2s_from_new_edge', dist < 2), ('interior_at_least_10s', dist >= 10)]:
                diff = (new - full[begin:end])[mask]
                rows.append(
                    dict(
                        i=c['i'],
                        dataset=c['dataset'],
                        subject=c['subject'],
                        begin=begin,
                        end=end,
                        region=label,
                        difference_over_full_S=np.sqrt(
                            np.mean(diff**2) / np.mean(full[begin:end][mask] ** 2)
                        ),
                        n_samples=int(mask.sum()),
                    )
                )
    return pd.DataFrame(rows)


if __name__ == '__main__':
    burst().to_csv(OUT / 'synthetic_burst.csv', index=False)
    parent_change().to_csv(OUT / 'parent_edge_sensitivity.csv', index=False)
