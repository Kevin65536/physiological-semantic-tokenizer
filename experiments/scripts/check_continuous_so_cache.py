"""Census actual span ranks and independently check the generated 200 Hz training targets."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train_continuous_so_formal import CACHE, load_parent
from experiments.scripts.check_continuous_so_targets import OUT, ORDER, admissible
from src.data.temporal_projection_so import continuous_indices, continuous_projection
from src.data.so_supervision_cache import write_json


def lengths(item):
    i, row = item
    mask = np.load(Path(row['folder']) / 'mask.npy', mmap_mode='r')[:, :, 0]
    support = admissible(row, mask)
    counts = Counter()
    for s in support.T:
        edges = np.flatnonzero(np.diff(np.r_[False, s, False]))
        counts.update((edges[1::2] - edges[::2]).tolist())
    return row['split'], row['dataset'], counts


def check():
    manifest, _, _, _ = load_parent()
    ex = [(i, r) for i, r in enumerate(manifest['rows']) if r['modality'] == 'eeg']
    census = Counter()
    with ThreadPoolExecutor(4) as pool:
        for split, ds, counts in pool.map(lengths, ex):
            for n, count in counts.items():
                census[split, ds, n] += count
    rows = []
    for (split, ds, n), count in sorted(census.items()):
        keep = continuous_indices(n, ORDER)
        hz = np.arange(n) * 100 / n
        core = np.flatnonzero((hz >= 4) & (hz < 13))
        missing = len(set(core) - set(keep))
        rows.append(
            dict(
                split=split,
                dataset=ds,
                samples=n,
                spans=count,
                coefficients=len(keep),
                missing_core_bins=missing,
                core_bins=len(core),
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'rank_census.csv', index=False)
    checks = []
    for (split, ds), _ in df.groupby(['split', 'dataset']):
        for i, r in ex:
            if (r['split'], r['dataset']) != (split, ds):
                continue
            src = Path(r['folder'])
            v = admissible(r, np.load(src / 'mask.npy', mmap_mode='r')[:, :, 0])
            if not v.any():
                continue
            x = np.load(src / 'x.npy', mmap_mode='r')[:, :, 0]
            old = np.load(src / 's.npy', mmap_mode='r')[:, :, 0]
            s, o = (np.load(CACHE / str(i) / (k + '.npy'), mmap_mode='r')[:, :, 0] for k in 'so')
            y = continuous_projection(old.astype(float), v, ORDER)
            scale = np.sqrt(np.mean(x[v].astype(float) ** 2))
            diff = np.max(np.abs(y[v] - s[v])) / scale
            close = np.max(np.abs(s[v].astype(float) + o[v] - x[v])) / scale
            assert diff < 1e-4 and close < 1e-5 and not s[~v].any()
            checks.append(
                dict(
                    dataset=ds,
                    split=split,
                    parent=r['parent'],
                    max_error_over_X_RMS=diff,
                    max_closure_over_X_RMS=close,
                )
            )
            break
    assert len(checks) == 8
    write_json(
        OUT / 'cache_checks.json',
        dict(
            status='passed',
            float64_checks=checks,
            effective_real_coefficients_per_400=float(
                400 * (df.spans * df.coefficients).sum() / (df.spans * df.samples).sum()
            ),
            eligible_channel_spans=int(df.spans.sum()),
            spans_missing_core_bins=int(df.loc[df.missing_core_bins > 0, 'spans'].sum()),
            shortest_span=int(df.samples.min()),
            longest_span=int(df.samples.max()),
        ),
    )


if __name__ == '__main__':
    check()
