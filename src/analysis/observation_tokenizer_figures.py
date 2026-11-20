"""Draw two development figures from retained CSV summaries, never from signal arrays."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from matplotlib import colormaps, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator


_INK = '#193349'
_MUTED = '#526579'
_TRAIN, _VAL = '#28658c', '#b54e25'
_DISPLAY = {'eeg': 'EEG · one electrode', 'hb': 'Hb · one HbO/HbR pair'}
_STYLES = {
    'full': ('Pattern + detail', '#136f68', 'o', '-'),
    'flat': ('Flat · same bit budget', '#315c9f', 's', '--'),
    'blind_full': ('Blind waveform route', '#806496', '^', ':'),
    'legacy_hard': ('Legacy dual K128 · 14 bits', '#ae4e26', 'D', 'None'),
}
_STYLE = {'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': _INK,
          'axes.labelcolor': _INK, 'axes.titlecolor': _INK, 'axes.edgecolor': '#aabcce',
          'xtick.color': _MUTED, 'ytick.color': _MUTED, 'axes.spines.right': False,
          'axes.spines.top': False, 'savefig.facecolor': 'white'}


def _read(root, name, columns):
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'expected a regular retained table: {name}')
    frame = pd.read_csv(path)
    if missing := set(columns) - set(frame):
        raise ValueError(f'{name}: missing {sorted(missing)}')
    return frame


def _figure(size):
    figure = Figure(figsize=size, facecolor='white')
    FigureCanvasAgg(figure)
    return figure


def _rd_figure(frame):
    if frame.empty or not frame['split'].eq('validation').all():
        raise ValueError('reconstruction.csv must be validation-only')
    data = frame[(frame.level == 'macro') & frame.method.isin(_STYLES)].copy()
    data[['bits_per_patch', 'x_nmse']] = data[['bits_per_patch', 'x_nmse']].apply(pd.to_numeric)
    if (not np.isfinite(data[['bits_per_patch', 'x_nmse']]).all().all() or (data.x_nmse < 0).any()
            or (data.bits_per_patch <= 0).any() or (data.bits_per_patch % 1 != 0).any()
            or data.duplicated(['modality', 'method', 'bits_per_patch']).any()):
        raise ValueError('invalid or duplicate macro reconstruction endpoint')
    fig = _figure((11.8, 5.2))
    axes = fig.subplots(1, 2, sharey=True)
    fig.subplots_adjust(left=.086, right=.981, bottom=.27, top=.72, wspace=.13)
    fig.text(.06, .94, 'Does a pattern ID improve reconstruction at equal bitrate?', size=17, weight='semibold')
    fig.text(.06, .886, 'Observed waveform reconstruction  /  fixed, subject-held-out development panel', color=_MUTED)
    log_y = (data.x_nmse.min() > 0 and data.x_nmse.max() / data.x_nmse.min() >= 30)
    for ax, (mod, title) in zip(axes, _DISPLAY.items()):
        block = data[data.modality == mod]
        by_method = {name: x.sort_values('bits_per_patch') for name, x in block.groupby('method')}
        rates = set(by_method.get('full', data.iloc[:0]).bits_per_patch)
        if (not rates or set(by_method.get('flat', data.iloc[:0]).bits_per_patch) != rates
                or set(by_method.get('legacy_hard', data.iloc[:0]).bits_per_patch) != {14}
                or ('blind_full' in by_method and set(by_method['blind_full'].bits_per_patch) != rates)):
            raise ValueError(f'{mod}: missing a matched rate or the 14-bit legacy point')
        for name, (label, color, marker, ls) in _STYLES.items():
            if name not in by_method:
                continue
            v = by_method[name]; legacy = name == 'legacy_hard'
            ax.plot(v.bits_per_patch, v.x_nmse, color=color, label=label, marker=marker,
                    linestyle=ls, lw=1.8, ms=9 if legacy else 6,
                    markerfacecolor='none' if legacy else 'white', markeredgewidth=1.6)
        ticks = sorted(rates | {14})
        ax.set_xscale('log', base=2); ax.set_xticks(ticks, [str(int(x)) for x in ticks])
        ax.xaxis.set_minor_locator(NullLocator()); ax.margins(x=.09, y=.13)
        if log_y:
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
        ax.set_axisbelow(True); ax.grid(axis='y', color='#dae2e9', lw=.7)
        ax.set_title(title, loc='left', weight='semibold', size=12, pad=9)
        ax.set_xlabel('Fixed bits per 2 s per site  (log₂ scale)')
    if not log_y:
        axes[0].set_ylim(0, 1.10 * data.x_nmse.max() or 1.)
    axes[0].set_ylabel('Observed X NMSE  ↓' + ('  (log scale)' if log_y else ''))
    handles = [Line2D([], [], color=c, marker=m, ls=l, lw=1.8, mfc='white', label=s)
               for k, (s,c,m,l) in _STYLES.items() if k in set(data.method)]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.055,.863),
               ncol=len(handles), frameon=False, columnspacing=1.8, handlelength=2.6)
    fig.text(.06, .113, 'Each point is a validation macro mean: squared errors use frozen training variances; datasets have equal weight.', color=_MUTED)
    fig.text(.06, .074, 'The new codecs encode a 2-second patch; the frozen legacy codec uses 30-second context. Displayed rate ranges differ by modality.', color=_MUTED)
    fig.text(.06, .035, 'Lines connect measured budgets only. Payload counts all integer IDs and detail; fixed dictionaries and framing are shared.', color=_MUTED)
    return fig


def _atlas_arrays(frame, mod):
    sub = frame[frame.modality == mod].copy()
    if sub.empty or not sub.split.isin(['train', 'validation']).all() or set(sub.split) != {'train','validation'}:
        raise ValueError(f'{mod}: atlas requires both development splits')
    numeric = ['pattern_id', 'n_subjects', 'n_records', 'n_patches', 'n_datasets', 'q25', 'median', 'q75']
    sub[numeric] = sub[numeric].apply(pd.to_numeric)
    counts = sub[numeric[:5]]; qs = sub[['q25','median','q75']]
    if (not np.isfinite(counts).all().all() or (counts < 0).any().any() or (counts % 1 != 0).any().any()
            or sub.duplicated(['split','pattern_id','feature']).any() or sub.feature.isna().any()):
        raise ValueError(f'{mod}: ambiguous atlas cell or invalid support')
    valid = sub.n_patches > 0
    if (not np.isfinite(qs[valid]).all().all() or (qs.q25[valid] > qs['median'][valid]).any()
            or (qs.q75[valid] < qs['median'][valid]).any() or qs[~valid].notna().any().any()
            or (sub[['n_subjects','n_records','n_datasets']].min(axis=1)[valid] <= 0).any()
            or (sub.groupby(['split','pattern_id'])[numeric[1:5]].nunique() > 1).any().any()):
        raise ValueError(f'{mod}: inconsistent support or conditional quantiles')
    train = sub[sub.split == 'train']; features = list(dict.fromkeys(train.feature))
    if set(features) != set(sub.feature):
        raise ValueError(f'{mod}: feature absent from train atlas')
    # A real zero-support ID is retained, sorted last; validation never sets the order.
    ids = sorted(set(sub.pattern_id))
    order = train[train.feature == features[0]].set_index('pattern_id')['median'].reindex(ids)
    ids = order.sort_values(kind='stable', na_position='last').index.to_numpy(dtype=int)
    stats, support = {}, {}
    for split in ['train','validation']:
        rows = sub[sub.split == split]
        stats[split] = {q: rows.pivot(index='pattern_id', columns='feature', values=q)
                        .reindex(index=ids, columns=features).to_numpy(float) for q in ['q25','median','q75']}
        support[split] = rows.drop_duplicates('pattern_id').set_index('pattern_id').n_subjects.reindex(ids).to_numpy(float)
    return features, ids, stats, support


def _feature_label(name):
    match = re.fullmatch(r'(?:eeg_)?(?:rel_|power_|band_)?(\d+(?:p\d+)?)_(\d+(?:p\d+)?)(?:hz)?', name.lower())
    if match:
        return '–'.join(x.replace('p','.') for x in match.groups()) + ' Hz'
    return name.replace('hbo', 'HbO').replace('hbr', 'HbR').replace('_mean',' level').replace('_slope_per_s',' slope').replace('_',' ')


def _n(x):
    return str(int(x)) if np.isfinite(x) else '—'


def _atlas_figure(frame):
    eeg, eids, es, en = _atlas_arrays(frame, 'eeg')
    hb, hids, hs, hn = _atlas_arrays(frame, 'hb')
    # EEG fractions already have the encoder's documented denominator. Do not renormalize
    # marginal medians to sum to 1: they need not all come from the same observed patch.
    for d in es.values():
        for arr in d.values():
            v = arr[np.isfinite(arr)]
            if ((v < -1e-9) | (v > 1+1e-9)).any():
                raise ValueError('expected EEG power fractions, not percent or arbitrary intensities')
    fig = _figure((13.0, 9.7)); left, right = .073, .966
    fig.text(.045,.96, 'What do the same IDs describe in train and validation?', size=17, weight='semibold')
    fig.text(.045,.927, 'Properties of measured 2-second observations assigned to each ID, not reconstructed waveforms or identified sources.', color=_MUTED)
    fig.text(.045,.879, 'A   EEG  /  relative spectrum', size=13, weight='semibold')
    fig.text(.045,.852, f'Median fraction of non-DC power in each band. IDs sorted by training {_feature_label(eeg[0])} fraction; gray = no data.', color=_MUTED)
    grid = fig.add_gridspec(1,3, left=left, right=right, bottom=.635, top=.798, width_ratios=[1,1,.028], wspace=.16)
    cmap = colormaps['viridis'].with_extremes(bad='#dce2e7')
    for j, split in enumerate(['train','validation']):
        ax = fig.add_subplot(grid[0,j]); ax.set_title(split.capitalize(), loc='left', fontsize=10, weight='semibold')
        image = ax.imshow(es[split]['median'].T, origin='upper', aspect='auto', interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        ax.set_yticks(range(len(eeg)), [_feature_label(f) for f in eeg] if j == 0 else [])
        ax.set_xticks(range(len(eids)), [str(i) for i in eids], fontsize=8); ax.tick_params(length=0)
        ax.set_xlabel('Pattern ID  (nominal)', labelpad=2)
        for col, n in enumerate(en[split]):
            ax.text(col, -.33, _n(n), transform=ax.get_xaxis_transform(), ha='center', va='center', fontsize=7.5, color=_MUTED)
        ax.text(-.008,-.33, 'n', transform=ax.transAxes, ha='right', va='center', style='italic', color=_MUTED)
    cb = fig.colorbar(image, cax=fig.add_subplot(grid[0,2]), ticks=[0,.25,.5,.75,1],
                      format=FuncFormatter(lambda x, _: f'{100*x:.0f}%')); cb.ax.tick_params(labelsize=8)
    fig.text(.045,.555, 'B   Hb  /  two-color level and slope', size=13, weight='semibold')
    fig.text(.045,.527, f'Dots are medians; bars span the middle half of real patch values (not confidence intervals). IDs sorted by training {_feature_label(hb[0])}.', color=_MUTED)
    fig.legend(handles=[Line2D([],[],color=col,marker=marker,ms=4,label=split) for split,col,marker in
                        [('Train',_TRAIN,'o'),('Validation',_VAL,'s')]], loc='upper left', bbox_to_anchor=(.041,.512),
               ncol=2, frameon=False, handlelength=2.3, columnspacing=1.5)
    grid = fig.add_gridspec(1,len(hb)+1, left=left, right=right, bottom=.112, top=.444,
                            width_ratios=[1]*len(hb)+[.64], wspace=.16)
    y = np.arange(len(hids)); slope = ['slope' in f.lower() for f in hb]
    # The two levels share an axis range; the two slopes share another, in their own units.
    for col, feature in enumerate(hb):
        ax = fig.add_subplot(grid[0,col]); same_unit = np.asarray(slope) == slope[col]
        ends = np.concatenate([hs[s][q][:,same_unit].ravel() for s in hs for q in ['q25','q75']]); ends = ends[np.isfinite(ends)]
        lo, hi = min(0., ends.min(initial=0)), max(0., ends.max(initial=0)); pad = (hi-lo)*.06 or .1
        ax.set_xlim(lo-pad,hi+pad); ax.axvline(0,color='#acbac8',lw=.8,zorder=0)
        for split, off, color, marker in [('train',-.16,_TRAIN,'o'),('validation',.16,_VAL,'s')]:
            d = hs[split]; yy = y+off; med = d['median'][:,col]; ok = np.isfinite(med)
            ax.hlines(yy[ok], d['q25'][ok,col], d['q75'][ok,col], color=color, linewidth=1.3)
            ax.plot(med[ok],yy[ok],marker=marker,ls='none',color=color,ms=3.4)
        ax.set_ylim(len(y)-.45,-.55); ax.set_axisbelow(True); ax.grid(axis='y', color='#edf1f4', lw=.75)
        ax.set_yticks(y, hids if col == 0 else []); ax.tick_params(axis='y',length=0)
        if col == 0:
            ax.set_ylabel('Pattern ID')
        ax.set_title(_feature_label(feature), loc='left', fontsize=10, weight='semibold')
        ax.set_xlabel('Training X SD / s' if slope[col] else 'Training X SD')
        ax.locator_params(axis='x', nbins=4)
    counts = fig.add_subplot(grid[0,-1]); counts.set_ylim(len(y)-.45,-.55); counts.set_axis_off()
    counts.set_title('Subjects\ntrain  /  val', fontsize=9, loc='center')
    for i, (tr,va) in enumerate(zip(hn['train'],hn['validation'])):
        counts.text(.5,i,f'{_n(tr)}  /  {_n(va)}', va='center', ha='center', color=_MUTED, fontsize=8)
    fig.text(.045,.047, 'n = distinct dataset-qualified subjects with this ID, not independent patch counts. All IDs shown; equal numbers across modalities have no implied relation.', color=_MUTED)
    fig.text(.045,.022, 'Quantiles use the frozen hierarchy weights, conditional on assignment. These profiles describe the properties used for assignment; they are not an independent semantic readout.', color=_MUTED)
    return fig


def render_figures(run_dir):
    """Render ``rate_distortion.png`` and ``pattern_atlas.png``; return their paths.

    Only two named CSV tables are opened. No checkpoint, raw trace, token-array
    replay or fitting is involved. Each table must carry explicit split labels.
    """
    root = Path(run_dir).resolve(strict=True)
    rd = _read(root, 'reconstruction.csv', ['level','split','modality','method','bits_per_patch','x_nmse'])
    atlas = _read(root, 'atlas.csv', ['modality','split','pattern_id','feature','median','q25','q75',
                                     'n_subjects','n_records','n_patches','n_datasets'])
    if set(rd.modality) != set(_DISPLAY) or set(atlas.modality) != set(_DISPLAY):
        raise ValueError('expected exactly the EEG and Hb development panels')
    paths = {key: root/f'{key}.png' for key in ['rate_distortion','pattern_atlas']}
    if any(p.is_symlink() for p in paths.values()):
        raise ValueError('refusing redirected figure output')
    with rc_context(_STYLE):
        figs = [_rd_figure(rd), _atlas_figure(atlas)]
        for fig, path in zip(figs, paths.values()):
            fig.savefig(path, dpi=220, format='png')
    return paths
