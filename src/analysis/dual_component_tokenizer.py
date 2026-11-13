"""Train-only controls, subject-level evidence, and frozen-token mask pilot.

The cache is already divided by observation-training SD without centering. MSE
in this space is NMSE against that common frozen denominator, for S/O/sum alike.
Windows and seeds are never treated as independent biological replicates.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json
import hashlib
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from src.foundation.dual_component_masked import (
    DualComponentMaskedModel, block_masks, own_context, expected_waveforms,
    fit_token_baselines, baseline_probabilities, nonwrapping_source_shift,
)

MODALITIES = ('eeg', 'hb')
BRANCHES = ('S', 'O')


def state_balanced_weights(dataset, parent, selected):
    """Equal datasets, equal parents within dataset, equal selected anchors within parent."""
    dataset, parent, selected = np.asarray(dataset), np.asarray(parent), np.asarray(selected, bool)
    weights = np.zeros(len(dataset), dtype=float)
    groups = np.unique(dataset[selected])
    for d in groups:
        parents = np.unique(parent[selected & (dataset == d)])
        for p in parents:
            mask = selected & (dataset == d) & (parent == p)
            weights[mask] = 1 / (len(groups) * len(parents) * mask.sum())
    return weights


def state_weighted_pca(values, weights):
    """Fit in the original latent metric; zero-weight holdout points never affect the axes."""
    x, w = np.asarray(values, float), np.asarray(weights, float)
    if x.ndim != 2 or w.shape != (len(x),) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('matrix and positive training mass required')
    use = w > 0; train = x[use]; w = w[use] / w[use].sum()
    mean = np.sum(train * w[:, None], axis=0)
    centered = train - mean
    covariance = (centered * w[:, None]).T @ centered
    eigen, axes = np.linalg.eigh(covariance); order = np.argsort(eigen)[::-1]
    eigen, axes = np.maximum(eigen[order], 0), axes[:, order]
    for j in range(axes.shape[1]):
        if axes[np.argmax(np.abs(axes[:, j])), j] < 0:
            axes[:, j] *= -1
    return dict(mean=mean, axes=axes, explained=eigen / max(eigen.sum(), 1e-30),
                participation_rank=float(eigen.sum() ** 2 / max(eigen @ eigen, 1e-30)))


def state_pattern_statistics(patches, weights, sample_rate):
    """Dispersion is across patches; PSD is averaged after computing each patch spectrum."""
    from scipy.signal import periodogram
    x, w = np.asarray(patches, float), np.asarray(weights, float)
    if x.ndim != 3 or w.shape != (len(x),) or not np.isfinite(x).all() or w.sum() <= 0:
        raise ValueError('finite complete patches with positive weights required')
    w = w / w.sum(); mean = np.einsum('n,ntc->tc', w, x)
    variance = np.einsum('n,ntc->tc', w, (x - mean) ** 2)
    frequency, psd = periodogram(x, fs=sample_rate, window='hann', detrend='constant', scaling='density', axis=1)
    power = np.einsum('n,nfc->fc', w, psd)
    energy = float(np.dot(w, np.mean(x * x, axis=(1, 2))))
    return dict(mean=mean, sd=np.sqrt(variance), frequency=frequency, psd=power,
                mean_patch_energy=energy, mean_waveform_energy_fraction=float(np.mean(mean * mean) / max(energy, 1e-30)))


def _state_information(ids, groups, weights, size=128):
    joint = np.zeros((size, int(np.max(groups)) + 1))
    np.add.at(joint, (ids, groups), weights)
    joint /= joint.sum(); expected = joint.sum(1)[:, None] * joint.sum(0)[None]
    take = joint > 0
    return float(np.sum(joint[take] * np.log2(joint[take] / expected[take])))


def render_state_codebook_figures(export, modality, candidate, branch):
    """Pattern, occupancy and frozen-latent geometry for one retained codebook."""
    import os
    import resource
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm
    from sklearn.manifold import TSNE, trustworthiness
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1); torch.set_num_threads(1)
    start_time = time.monotonic(); export = Path(export)
    stem = modality + '_' + candidate; key = stem + '_' + branch
    folder = export / key; folder.mkdir()
    meta = json.loads((export / (stem + '_windows.json')).read_text())
    with np.load(export / (stem + '.npz')) as stored:
        ids = stored['ids_' + branch].ravel()
        latent = stored['latent_' + branch].reshape(-1, stored['latent_' + branch].shape[-1]).astype(float)
        book = stored['codebook_' + branch].astype(float)
        size = len(book)
        pattern_index = stored['pattern_index']
        pattern_x = stored['pattern_x']
        pattern_y = stored['pattern_' + branch.lower()] if candidate != 'q_x' else pattern_x
        pattern_hat = stored['pattern_hat_' + branch.lower()] if candidate != 'q_x' else stored['pattern_hat_s'] + stored['pattern_hat_o']
        x_rms = stored['x_rms'].ravel(); x_mean = stored['x_mean'].ravel()
    names = ['eeg_fnirs_single_trial', 'refed', 'simultaneous_eeg_nirs', 'visual_cognitive_motivation']
    short = ['Single-Trial', 'REFED', 'Simultaneous', 'Visual']
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    ds = np.repeat([names.index(r['dataset']) for r in meta], 15)
    parent = np.repeat([r['parent'] for r in meta], 15)
    role = np.repeat([r['role'] for r in meta], 15)
    slot = np.tile(np.arange(15), len(meta))
    weights = {r: state_balanced_weights(ds, parent, role == r) for r in ('train', 'holdout')}
    probabilities = {r: np.bincount(ids, weights=weights[r], minlength=size) for r in weights}
    top = np.argsort(-probabilities['train'], kind='stable')[:8]
    rows = []; usage = {}
    for r in ('train', 'holdout'):
        for d in [-1, 0, 1, 2, 3]:
            selected = (role == r) & ((ds == d) if d >= 0 else True)
            w = state_balanced_weights(ds, parent, selected)
            count = np.bincount(ids[selected], minlength=size)
            p = np.bincount(ids, weights=w, minlength=size)
            positive = p[p > 0]
            record = dict(role=r, dataset='equal-dataset' if d < 0 else short[d], counts=count.tolist(), probability=p.tolist(),
                active=int(np.count_nonzero(count)), effective=float(np.exp(-(positive * np.log(positive)).sum())),
                pmax=float(p.max()), top10=float(np.sort(p)[-10:].sum()), assignments=int(count.sum()))
            usage[(r, d)] = record; rows.append(record)
    _json(folder / 'usage.json', rows)
    display_branch = branch if candidate != 'q_x' else ('codebook 1' if branch == 'S' else 'codebook 2')
    label = f"{modality.upper()} | {'S/O supervision' if candidate != 'q_x' else 'Raw quantization'} | {display_branch}"
    pages = []
    plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 11.5, 'axes.labelsize': 10.5,
        'legend.fontsize': 8.5, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

    def save(fig, name, title, caption, category):
        path = folder / (name + '.png')
        fig.savefig(path, dpi=220); plt.close(fig)
        pages.append(dict(figure=str(path.relative_to(export)), title=title, caption=caption,
                          category=category, book=key))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10.5), constrained_layout=True, gridspec_kw={'height_ratios': [1, 1, 1.45]})
    for r, col in [('train', '#0072B2'), ('holdout', '#D55E00')]:
        p = probabilities[r] * 100
        axes[0].plot(np.arange(size), p, color=col, lw=1, marker='.', ms=3,
                     label=f"{r}: active {usage[(r,-1)]['active']}/{size}; effective {usage[(r,-1)]['effective']:.1f}")
        order = np.argsort(-probabilities['train'], kind='stable')
        axes[1].plot(np.arange(1, size+1), p[order], color=col, label=r)
    axes[0].axhline(100 / size, color='#555', ls='--', lw=.8, label=f'Uniform 1/{size} reference')
    axes[0].set(xlabel='Code ID (categorical)', ylabel='Assignments (%)', xlim=(-1, size), title='Actual code identity; dataset / parent balanced')
    axes[0].legend(ncol=3)
    axes[1].set(xlabel='Rank fixed by training usage', ylabel='Assignments (%)', title='The holdout curve keeps the training ordering')
    axes[1].legend(); axes[0].grid(alpha=.2); axes[1].grid(alpha=.2)
    matrix = np.stack([usage[(r, d)]['probability'] for d in range(4) for r in ('train', 'holdout')]) * 100
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#e5e7eb')
    im = axes[2].imshow(np.ma.masked_equal(matrix, 0), aspect='auto', interpolation='nearest', cmap=cmap,
                        norm=LogNorm(vmin=.01, vmax=30))
    axes[2].set_yticks(np.arange(8), [f"{short[d]} {r}: eff={usage[(r,d)]['effective']:.1f}" for d in range(4) for r in ('train', 'holdout')])
    axes[2].set(xlabel='Code ID', title='Per-dataset use: grey = zero; common logarithmic color scale')
    fig.colorbar(im, ax=axes[2], label='Assignments (%)', ticks=[.01, .1, 1, 10, 30])
    fig.suptitle(label + ' — code usage', fontsize=16)
    save(fig, 'usage', label + '：码字使用率',
        '全部合法训练滑窗与同被试留出输入；一个赋码来自一个位置、一个30秒上下文中的2秒锚点。先在父窗内归一化，再父窗/数据集等权。虚线为等频参考；色图统一0.01%–30%，零频为灰色。重复滑窗不是独立被试。', 'usage')

    # Pattern views use unique physical anchors from nonoverlapping input windows and complete scoring support.
    pr = role[pattern_index]; pd = ds[pattern_index]; pp = parent[pattern_index]; pid = ids[pattern_index]
    pattern_weights = {r: state_balanced_weights(pd, pp, pr == r) for r in ('train', 'holdout')}
    rate = 200 if modality == 'eeg' else 10
    stats = {}; pattern_summary = []
    for r in ('train', 'holdout'):
        for code in top:
            choose = (pr == r) & (pid == code)
            if not choose.any():
                stats[(r, int(code))] = None; continue
            w = pattern_weights[r][choose]
            values = dict(target=state_pattern_statistics(pattern_y[choose], w, rate),
                          decoded=state_pattern_statistics(pattern_hat[choose], w, rate),
                          observed=state_pattern_statistics(pattern_x[choose], w, rate))
            ix = np.flatnonzero(choose)
            overlays = ix[np.linspace(0, len(ix) - 1, min(24, len(ix))).round().astype(int)]
            values.update(n=len(ix), indices=overlays, dataset_counts={short[d]: int(np.sum(choose & (pd == d))) for d in range(4)})
            stats[(r, int(code))] = values
            pattern_summary.append(dict(role=r, code=int(code), n=len(ix), dataset_counts=values['dataset_counts'],
                mean_waveform_energy_fraction=values['target']['mean_waveform_energy_fraction'],
                observed_mean_waveform_energy_fraction=values['observed']['mean_waveform_energy_fraction']))
    limits = {}
    for kind in ('target', 'observed'):
        values = [v for v in stats.values() if v is not None]
        lo = min(float(np.min(v[kind]['mean'] - v[kind]['sd'])) for v in values)
        hi = max(float(np.max(v[kind]['mean'] + v[kind]['sd'])) for v in values)
        if kind == 'target':
            lo = min(lo, min(float(v['decoded']['mean'].min()) for v in values))
            hi = max(hi, max(float(v['decoded']['mean'].max()) for v in values))
        else:
            lo = min(lo, min(float(pattern_x[v['indices']].min()) for v in values))
            hi = max(hi, max(float(pattern_x[v['indices']].max()) for v in values))
        pad = max((hi - lo) * .06, .01); limits[kind] = (lo - pad, hi + pad)
    if modality == 'eeg':
        powers = [v[k]['psd'][1:] for v in stats.values() if v is not None for k in ('target', 'decoded', 'observed')]
        spectral_limits = (max(min(float(x.min()) for x in powers) * .7, 1e-12), max(float(x.max()) for x in powers) * 1.5)
    else:
        shown = np.concatenate([pattern_y[v['indices']] for v in stats.values() if v is not None])
        lo, hi = float(shown.min()), float(shown.max())
        pad = max((hi - lo) * .05, .01)
        limits['joint'] = (lo - pad, hi + pad)
    for r in ('train', 'holdout'):
        for page_number in range(2):
            codes = top[page_number * 4:(page_number + 1) * 4]
            fig, axes = plt.subplots(4, 3, figsize=(14, 10.6), constrained_layout=True)
            for i, code in enumerate(codes):
                v = stats[(r, int(code))]; a, b, c = axes[i]
                if v is None:
                    for ax in axes[i]:
                        ax.axis('off'); ax.text(.05, .5, f'Code {code}: n=0 in this role', transform=ax.transAxes)
                    continue
                t = np.arange(pattern_y.shape[1]) / rate
                target_label = ('Teacher ' + branch + ' (conditional)') if candidate != 'q_x' else 'Observed X'
                if modality == 'eeg':
                    mean, sd = v['target']['mean'][:, 0], v['target']['sd'][:, 0]
                    a.fill_between(t, mean - sd, mean + sd, color='#0072B2', alpha=.2, label='Patch mean +/- SD')
                    a.plot(t, mean, color='#0072B2', label=target_label)
                    a.plot(t, v['decoded']['mean'][:, 0], color='#D55E00', ls='--', label='Integer decode mean')
                    for kind, color, style in [('target', '#0072B2', '-'), ('decoded', '#D55E00', '--'), ('observed', '#777777', ':')]:
                        f = v[kind]['frequency']; power = v[kind]['psd'][:, 0]; good = f > 0
                        b.semilogy(f[good], np.maximum(power[good], 1e-12), color=color, ls=style, label=kind)
                    b.set(xlabel='Frequency (Hz)', ylabel='Training SD squared / Hz', xlim=(.5, 100),
                          ylim=spectral_limits, title='Mean individual-patch PSD; DC removed')
                    for y in pattern_x[v['indices']]:
                        c.plot(t, y[:, 0], color='#808890', alpha=.20, lw=.6)
                    c.plot(t, v['observed']['mean'][:, 0], color='#b32025', lw=1.5, label='Observed X mean')
                    c.set(title='Observed X: up to 24 actual patch overlays')
                else:
                    for ch, color, name in [(0, '#c23b3e', 'HbO'), (1, '#2771ad', 'HbR')]:
                        mean, sd = v['target']['mean'][:, ch], v['target']['sd'][:, ch]
                        a.fill_between(t, mean - sd, mean + sd, color=color, alpha=.16)
                        a.plot(t, mean, color=color, label=name + ' target')
                        a.plot(t, v['decoded']['mean'][:, ch], color=color, ls='--', label=name + ' decode')
                        for y in pattern_x[v['indices']]:
                            c.plot(t, y[:, ch], color=color, alpha=.12, lw=.6)
                        c.plot(t, v['observed']['mean'][:, ch], color=color, lw=1.5, label=name + ' observed mean')
                    for y in pattern_y[v['indices']]:
                        b.plot(y[:, 0], y[:, 1], color='#888', alpha=.3, lw=.6)
                    path = v['target']['mean']; b.plot(path[:, 0], path[:, 1], color='#333', lw=1.5)
                    b.scatter(path[0, 0], path[0, 1], color='#009E73', s=25, marker='o', label='Start')
                    b.scatter(path[-1, 0], path[-1, 1], color='#D55E00', s=30, marker='^', label='End')
                    b.set(xlabel='HbO / joint training SD', ylabel='HbR / joint training SD',
                          title='Joint target trajectory over 2 s', xlim=limits['joint'], ylim=limits['joint'])
                    c.set(title='Observed paired Hb: actual patch overlays')
                a.set(title=f'Code {code} | n={v["n"]} patches', xlabel='Time from anchor (s)',
                      ylabel='Amplitude / training X SD', xlim=(0, 2), ylim=limits['target'])
                c.set(xlabel='Time from anchor (s)', ylabel='Amplitude / training X SD', xlim=(0, 2), ylim=limits['observed'])
                for ax in (a, b, c):
                    ax.grid(alpha=.2)
                if i == 0:
                    a.legend(ncol=2); b.legend(fontsize=7.5); c.legend(fontsize=7.5)
            fig.suptitle(label + f' — {r} patterns | training ranks {page_number * 4 + 1}–{page_number * 4 + 4}', fontsize=15)
            save(fig, f'patterns_{r}_{page_number + 1}', label + f'：{r} token patterns（{page_number + 1}/2）',
                '码字按训练赋码率选定，留出图复用相同ID。n为不重叠输入窗中具有完整评分支持的2秒片段数，阴影为片段间SD。实线为条件目标/观测的加权均值，虚线为完整15-ID序列解码后同锚点的均值；每个ID由30秒上下文决定。'
                + ('频谱先对每段去均值并计算Hann周期图，再平均，分辨率0.5Hz；不是平均波形的频谱。' if modality == 'eeg' else 'Hb用双色团短程轨迹展示联合变化；圆点/三角为2秒片段起止。')
                + '同一码本各页的目标/观测纵轴分别固定；不同码本不共享幅度轴。', 'patterns')
    _json(folder / 'pattern_statistics.json', pattern_summary)
    np.savez_compressed(folder / 'pattern_means.npz', **{f'{r}_{code}_{kind}_{field}': v[kind][field]
        for (r, code), v in stats.items() if v is not None for kind in ('target', 'decoded', 'observed')
        for field in ('mean', 'sd', 'frequency', 'psd')})

    projection = state_weighted_pca(latent, weights['train'])
    xy = (latent - projection['mean']) @ projection['axes'][:, :2]
    bx = (book - projection['mean']) @ projection['axes'][:, :2]
    rng = np.random.default_rng(17); sample = []
    for r in ('train', 'holdout'):
        for d in range(4):
            parents = np.unique(parent[(role == r) & (ds == d)])
            for par in parents:
                ix = np.flatnonzero((role == r) & (ds == d) & (parent == par))
                sample.extend(rng.choice(ix, min(len(ix), 192 // len(parents)), replace=False).tolist())
    sample = np.array(sorted(sample), dtype=int)
    allxy = np.concatenate([xy, bx]); full_lo, full_hi = allxy.min(0), allxy.max(0)
    margin = np.maximum((full_hi - full_lo) * .05, .01); full_lo -= margin; full_hi += margin
    zoom_lo, zoom_hi = np.quantile(xy, [.01, .99], axis=0)
    code_colors = plt.get_cmap('turbo')(np.linspace(0, 1, size))
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.5), constrained_layout=True)
    for i, r in enumerate(('train', 'holdout')):
        ix = sample[role[sample] == r]
        for d in range(4):
            keep = ix[ds[ix] == d]
            axes[i, 0].scatter(xy[keep, 0], xy[keep, 1], s=5, alpha=.45, color=colors[d], label=short[d])
        axes[i, 0].scatter(bx[:, 0], bx[:, 1], marker='x', s=18, color='#222', linewidths=.6, label='Codebook vectors')
        axes[i, 1].scatter(xy[ix, 0], xy[ix, 1], s=5, alpha=.55, c=code_colors[ids[ix]])
        axes[i, 1].scatter(bx[:, 0], bx[:, 1], marker='*', s=35, c=code_colors, edgecolors='#222', linewidths=.35)
        for code in top[:4]:
            if np.all(bx[code] >= zoom_lo) and np.all(bx[code] <= zoom_hi):
                axes[i, 1].annotate(str(code), bx[code], xytext=(3, 3), textcoords='offset points', fontsize=8)
        sc = axes[i, 2].scatter(xy[ix, 0], xy[ix, 1], c=slot[ix], s=6, alpha=.55, cmap='viridis', vmin=0, vmax=14)
        for j in (0, 2):
            axes[i, j].set(xlim=(full_lo[0], full_hi[0]), ylim=(full_lo[1], full_hi[1]))
        outside = int(np.sum(np.any((xy[ix] < zoom_lo) | (xy[ix] > zoom_hi), axis=1)))
        axes[i, 1].set(xlim=(zoom_lo[0], zoom_hi[0]), ylim=(zoom_lo[1], zoom_hi[1]))
        for j, title in enumerate(('Dataset; full range', f'Code ID; 1–99% axis zoom; outside={outside}', 'Window slot; full range')):
            axes[i, j].set(title=f'{r}: {title}', xlabel=f"PC1 ({projection['explained'][0]*100:.1f}%)",
                           ylabel=f"PC2 ({projection['explained'][1]*100:.1f}%)")
            axes[i, j].grid(alpha=.15)
        axes[i, 0].legend(fontsize=8)
    fig.colorbar(sc, ax=axes[:, 2], label='Slot index in 30 s context (0–14)', ticks=[0, 7, 14])
    fig.suptitle(label + f" — training-fitted PCA | first two PCs {projection['explained'][:2].sum()*100:.1f}%", fontsize=15)
    save(fig, 'pca', label + '：量化前 latent 与码本的 PCA',
        'PCA只在训练latent上拟合，按数据集/父窗等权且不逐维标准化。点为上下文中的2秒锚点；黑叉/星为全部码本向量。训练与留出共用轴；中列放大到坐标1%–99%范围，并列出抽样点中未显示数量。各码本独立降维，ID和颜色仅在当前码本内有意义，坐标不可跨码本直接比较。', 'latent')
    embedding_input = np.concatenate([latent[sample], book])
    embeddings = []; neighborhood = []
    for seed in (17, 29):
        tsne = TSNE(n_components=2, perplexity=30, max_iter=750, learning_rate='auto', init='random', random_state=seed, n_jobs=1)
        embedding = tsne.fit_transform(embedding_input)
        trust = float(trustworthiness(embedding_input, embedding, n_neighbors=15))
        embeddings.append(embedding); neighborhood.append(dict(seed=seed, trustworthiness_15=trust, KL=float(tsne.kl_divergence_)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for j, (embedding, check) in enumerate(zip(embeddings, neighborhood)):
        z, centers = embedding[:-size], embedding[-size:]
        for d in range(4):
            take = ds[sample] == d
            axes[0, j].scatter(z[take, 0], z[take, 1], s=6, alpha=.5, color=colors[d], label=short[d])
        axes[0, j].scatter(centers[:, 0], centers[:, 1], marker='x', color='#222', s=18, linewidths=.7, label='Codebook vectors')
        axes[1, j].scatter(z[:, 0], z[:, 1], s=6, alpha=.6, c=code_colors[ids[sample]])
        axes[1, j].scatter(centers[:, 0], centers[:, 1], marker='*', s=45, c=code_colors, edgecolors='#222', linewidths=.4)
        for code in top[:4]:
            axes[1, j].annotate(str(code), centers[code], xytext=(3, 3), textcoords='offset points', fontsize=8)
        axes[0, j].legend(fontsize=8)
        for i in range(2):
            axes[i, j].set(xlabel='t-SNE 1 (arbitrary scale)', ylabel='t-SNE 2 (arbitrary scale)',
                           title=f"seed {check['seed']}; neighborhood preservation {check['trustworthiness_15']:.3f}")
            axes[i, j].grid(alpha=.12)
    fig.suptitle(label + ' — joint t-SNE: datasets (top), code IDs (bottom)', fontsize=15)
    save(fig, 'tsne', label + '：t-SNE 两个固定种子的分布',
        '同一组分层抽样latent与全部码本向量联合降维，仅用于描述分布：每数据集每角色最多192个锚点，父窗内均衡。perplexity=30、750次迭代，两个种子全部展示。邻域保持率用原latent维度与2维的15近邻计算；图中远近、簇大小和跨种子方向不是原空间的定量距离。', 'latent')
    np.savez_compressed(folder / 'embedding_coordinates.npz', PCA_mean=projection['mean'], PCA_axes=projection['axes'],
        PCA_explained=projection['explained'], PCA_points=xy, PCA_codebook=bx, sampled_index=sample,
        tsne_seed17=embeddings[0], tsne_seed29=embeddings[1], code_ids=ids, dataset=ds, parent=parent, role=role, slot=slot)
    def correlation(a, b, w):
        aa = a - np.dot(w, a); bb = b - np.dot(w, b)
        return float(np.dot(w, aa * bb) / max(np.sqrt(np.dot(w, aa * aa) * np.dot(w, bb * bb)), 1e-30))
    summary = dict(status='completed', key=key, label=label, top_training_codes=top.tolist(), usage=rows,
        PCA_first_two_variance=float(projection['explained'][:2].sum()), latent_participation_rank=projection['participation_rank'],
        PC1_observed_RMS_correlation=correlation(xy[:, 0], x_rms, weights['train']),
        PC1_observed_mean_correlation=correlation(xy[:, 0], x_mean, weights['train']),
        code_dataset_MI_bits=_state_information(ids, ds, weights['train'], size),
        code_slot_MI_bits=_state_information(ids, slot, weights['train'], size),
        tsne=neighborhood, pattern_summary=pattern_summary, pages=pages,
        seconds=time.monotonic() - start_time, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    _json(folder / 'summary.json', summary)
    return summary


def _state_bitmap_pdf(export, pages, filename):
    """Shared bitmap-only figure atlas with searchable captions."""
    import fitz
    export = Path(export)
    doc = fitz.open()
    for index, page_info in enumerate(pages):
        page = doc.new_page(width=1000, height=850)
        for rect, text, size in [(fitz.Rect(30, 18, 970, 74), page_info['title'], 17),
                (fitz.Rect(30, 745, 970, 820), page_info['caption'], 10.5)]:
            if page.insert_textbox(rect, text, fontname='china-s', fontsize=size, lineheight=1.25) < 0:
                raise ValueError('figure atlas caption overflow')
        page.insert_image(fitz.Rect(30, 82, 970, 735), filename=str(export / page_info['figure']), keep_proportion=True)
        page.insert_text((30, 837), f"{index+1}/{len(pages)} | 4 training subjects; within-subject holdout | 200 Hz EEG; 30 s context / 2 s anchor | {page_info['book']}", fontsize=8)
    doc.save(export / filename, deflate=True, garbage=4); doc.close()
    pdf = fitz.open(export / filename)
    checks = [dict(page=i+1, images=len(page.get_images()), vector_paths=len(page.get_drawings()), searchable_characters=len(page.get_text()))
              for i, page in enumerate(pdf)]
    if any(c['images'] != 1 or c['vector_paths'] or c['searchable_characters'] < 40 for c in checks):
        raise AssertionError('atlas bitmap/searchable-caption contract failed')
    pdf.close(); _json(export / 'pdf_verification.json', dict(status='passed', pages=checks))
    return checks


def assemble_state_token_atlas(export, results, analysis):
    """PNG-only figures with searchable PDF captions and a local HTML figure index."""
    import html
    export = Path(export)
    results = sorted(results, key=lambda r: (0 if '_q_so_' in r['key'] else 1, r['key']))
    pages = [p for r in results for p in r['pages']]
    intro = ('本图集分析已完成的200 Hz EEG / 10 Hz Hb tokenizer pilot，主模型四个S/O码本及普通量化四个码本均为K128。'
             '只有四名训练被试；holdout为同被试父窗末段，所有图均未读取测试分区。码字由30秒上下文推断，锚点每2秒更新。'
             '使用率覆盖全部允许滑窗；pattern仅采用不重叠输入中的完整评分片段；降维抽样及训练PCA拟合范围在相邻图注中说明。')
    checks = _state_bitmap_pdf(export, pages, 'TOKEN_PATTERN_ATLAS.pdf')
    blocks = []
    for record in results:
        links = ' · '.join(f'<a href="#{html.escape(p["book"]+"_"+Path(p["figure"]).stem)}">{html.escape(Path(p["figure"]).stem)}</a>' for p in record['pages'])
        blocks.append(f'<section><h2>{html.escape(record["label"])}</h2><p>{links}</p><p>PCA前两维解释{record["PCA_first_two_variance"]*100:.1f}%；latent参与率有效维度{record["latent_participation_rank"]:.2f}。数据集/槽位与码字的互信息分别为{record["code_dataset_MI_bits"]:.3f}/{record["code_slot_MI_bits"]:.3f} bit；它们描述依赖关系，不是生理标签。</p><p><a href="{record["key"]}/summary.json">全部数值</a></p></section>')
        for p in record['pages']:
            ident = p['book'] + '_' + Path(p['figure']).stem
            blocks.append(f'<section id="{html.escape(ident)}"><h3>{html.escape(p["title"])}</h3><a href="{p["figure"]}"><img loading="lazy" src="{p["figure"]}" alt="{html.escape(p["title"])}"></a><p>{html.escape(p["caption"])}</p></section>')
    style = 'body{font:16px/1.7 system-ui,sans-serif;background:#edf1f5;color:#15283d;margin:0}main{max-width:1500px;margin:auto;padding:18px}section{background:white;padding:24px;margin:18px 0;border-radius:8px}img{width:100%;height:auto}a{color:#126da1}h1,h2,h3{line-height:1.4}'
    overview = []
    for r in results:
        by_role = {v['role']: v for v in r.get('usage', []) if v['dataset'] == 'equal-dataset'}
        if len(by_role) != 2:
            continue
        overview.append('<tr><td>'+html.escape(r['label'])+'</td>'+''.join(f'<td>{by_role[role]["active"]}/128；{by_role[role]["effective"]:.1f}</td>' for role in ('train','holdout'))+
                        f'<td>{r["PCA_first_two_variance"]*100:.1f}%</td><td>{r["code_slot_MI_bits"]:.3f}</td></tr>')
    table = '<table><tr><th>码本</th><th>训练：活跃；有效码数</th><th>留出：活跃；有效码数</th><th>PCA前两维</th><th>码字–槽位互信息(bit)</th></tr>'+''.join(overview)+'</table>'
    style += 'table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;border-bottom:1px solid #d5dce4;padding:8px}'
    (export / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Token pattern 与码本健康图集</title><style>'+style+'</style><main><section><h1>Token pattern、码字使用率与潜在表示分布</h1><p>'+intro+'</p><p><a href="TOKEN_PATTERN_ATLAS.pdf">完整 PDF 图集</a> · <a href="analysis_config.json">作图口径</a></p>'+table+'</section>'+''.join(blocks)+'</main></html>', encoding='utf-8')
    _json(export / 'figure_index.json', dict(pages=pages, analysis=analysis))
    return dict(status='completed', books=len(results), figures=len(pages), pdf_pages=len(checks),
                index='index.html', pdf='TOKEN_PATTERN_ATLAS.pdf', results=[{k:r[k] for k in
                ('key','PCA_first_two_variance','latent_participation_rank','PC1_observed_RMS_correlation',
                 'PC1_observed_mean_correlation','code_dataset_MI_bits','code_slot_MI_bits','tsne')} for r in results])


def state_reconstruction_key(row):
    return tuple(row[k] for k in ('modality', 'dataset', 'subject', 'record', 'parent', 'role', 'start_s', 'position'))


def select_state_reconstruction_examples(rows):
    """Fixed within-dataset error quantiles, with declared short-window and outlier supplements."""
    population = [r for r in rows if r['role'] == 'within_subject_holdout']
    valid = [r for r in population if r.get('x_nmse') is not None and np.isfinite(r['x_nmse'])]
    selected = {}; denominators = []
    def add(row, tag):
        key = state_reconstruction_key(row)
        if key not in selected:
            selected[key] = dict(row, selection_tags=[])
        selected[key]['selection_tags'].append(tag)
    for modality, dataset in sorted({(r['modality'], r['dataset']) for r in population}):
        group = sorted([r for r in valid if r['modality'] == modality and r['dataset'] == dataset],
                       key=lambda r: (r['x_nmse'], state_reconstruction_key(r)))
        total = sum(r['modality'] == modality and r['dataset'] == dataset for r in population)
        denominators.append(dict(modality=modality, dataset=dataset, planned=total, available=len(group), failed=total-len(group)))
        if not group:
            continue
        for quantile in (.1, .5, .9):
            add(group[int(np.floor(quantile * (len(group) - 1) + .5))], f'P{round(quantile*100)}')
        if dataset == 'refed':
            short = [r for r in group if r['duration_s'] < 90]
            if short:
                add(short[len(short)//2], 'short-parent')
        if dataset == 'visual_cognitive_motivation' and modality == 'hb':
            named = [r for r in group if r['position'] == 'Probe1:CH23']
            if named:
                add(max(named, key=lambda r:r['x_nmse']), 'retained-high-error-position')
    return dict(denominators=denominators, examples=list(selected.values()),
                rule='P10/P50/P90 of hard Q-SO total-X NMSE within dataset/modality; nearest rank; declared supplements; no replacement of failed rows')


def state_reconstruction_metrics(target, predicted, mask, training_variance):
    """Per-color/sample-supported errors in the frozen coordinates, with undefined correlations retained."""
    y, estimate = np.asarray(target, float), np.asarray(predicted, float)
    mask, variance = np.asarray(mask, bool), np.asarray(training_variance, float)
    if y.ndim != 2 or y.shape != estimate.shape or mask.shape != y.shape or variance.shape != (y.shape[1],):
        raise ValueError('aligned [time, channel] arrays and fixed channel variance required')
    if np.any(variance <= 0) or not np.isfinite(variance).all() or not np.isfinite(y[mask]).all() or not np.isfinite(estimate[mask]).all():
        raise ValueError('finite scored values and positive variance required')
    per_channel = []
    for c in range(y.shape[1]):
        a, b = y[:, c][mask[:, c]], estimate[:, c][mask[:, c]]
        if not len(a):
            raise ValueError('empty scored support')
        nmse = float(np.mean((b-a)**2)/variance[c])
        correlation = float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
        per_channel.append(dict(valid_samples=len(a), NMSE=nmse, NRMSE=float(np.sqrt(nmse)),
            correlation=correlation, bias_in_training_SD=float(np.mean(b-a)/np.sqrt(variance[c]))))
    return dict(channels=per_channel, NMSE=float(np.mean([x['NMSE'] for x in per_channel])),
                NRMSE=float(np.sqrt(np.mean([x['NMSE'] for x in per_channel]))))


def state_curve_limits(*curves):
    """Common limits include every plotted finite value, including older controls."""
    finite = np.concatenate([np.asarray(a)[np.isfinite(a)] for a in curves])
    if not len(finite):
        raise ValueError('no finite plotted values')
    lo, hi = float(finite.min()), float(finite.max())
    pad = max(.06 * (hi - lo), .01)
    return lo - pad, hi + pad


def state_patch_features(values, mask, rate, seconds=2):
    """Masked OLS on the actual second grid, separately for every color/patch."""
    values = np.asarray(values, float); mask = np.asarray(mask, bool)
    n = round(rate * seconds)
    if values.shape != mask.shape or values.ndim != 2 or len(values) % n:
        raise ValueError('complete time-by-color blocks and matching mask required')
    y = values.reshape(-1, n, values.shape[-1]); m = mask.reshape(y.shape)
    if not np.isfinite(y[m]).all():
        raise ValueError('nonfinite scored waveform')
    count = m.sum(1); safe = np.maximum(count, 1)
    t = (np.arange(n) / rate)[None, :, None]
    level = np.where(m, y, 0).sum(1) / safe
    u = t - np.where(m, t, 0).sum(1)[:, None, :] / safe[:, None, :]
    centered = np.where(m, y - level[:, None, :], 0.)
    support = np.where(m, u*u, 0.).sum(1)
    slope = (centered*u).sum(1) / np.maximum(support, 1e-30)
    return dict(level=level, slope=slope, rms=np.sqrt((centered**2).sum(1)/safe),
                centered=centered, count=count, valid=count >= 2)


def state_dynamic_metrics(target, predicted, mask, rate, training_variance, thresholds):
    """Orthogonal level/shape errors and fixed 2/6 s dynamics; no input transform.

    thresholds are frozen from nonoverlapping training targets, by dataset,
    component and color. Near-flat patches remain in all absolute errors.
    """
    target=np.asarray(target,float); predicted=np.asarray(predicted,float);mask=np.asarray(mask,bool)
    truth=state_patch_features(target,mask,rate);estimate=state_patch_features(predicted,mask,rate)
    count=truth['count'];den=np.maximum(mask.sum(0),1);variance=np.asarray(training_variance)
    patch_squared=np.where(mask,(predicted-target)**2,0.).reshape(truth['centered'].shape).sum(1)
    patch_level=count*(estimate['level']-truth['level'])**2
    patch_shape=((estimate['centered']-truth['centered'])**2).sum(1)
    patch_closure=np.abs(patch_squared-patch_level-patch_shape)/np.maximum(patch_squared,1.)
    if np.max(patch_closure)>1e-10:raise AssertionError('per-patch level/shape decomposition did not close')
    squared=patch_squared.sum(0);level=patch_level.sum(0);shape=patch_shape.sum(0)
    closure=np.abs(squared-level-shape)/np.maximum(squared,1.)
    if np.max(closure)>1e-10:raise AssertionError('level/shape orthogonal decomposition did not close')
    channels=[]
    for c in range(target.shape[1]):
        dynamic=truth['valid'][:,c]&(truth['rms'][:,c]>thresholds['rms'][c])
        e=((estimate['centered'][:,:,c]-truth['centered'][:,:,c])**2).sum(1)
        energy=(truth['centered'][:,:,c]**2).sum(1)
        channels.append(dict(NMSE=float(squared[c]/den[c]/variance[c]),
            level_nmse=float(level[c]/den[c]/variance[c]),shape_nmse=float(shape[c]/den[c]/variance[c]),
            closure_relative=float(closure[c]),patches=int(truth['valid'][:,c].sum()),
            dynamic_patches=int(dynamic.sum()),near_zero_patches=int((truth['valid'][:,c]&~dynamic).sum()),
            D_shape=float(1-e[dynamic].sum()/energy[dynamic].sum()) if energy[dynamic].sum()>0 else None,
            shape_error_energy=float(e[dynamic].sum()),target_centered_energy=float(energy[dynamic].sum())))
    slopes={}
    for seconds in (2,6):
        a=truth if seconds==2 else state_patch_features(target,mask,rate,seconds)
        b=estimate if seconds==2 else state_patch_features(predicted,mask,rate,seconds)
        out=[]
        for c in range(target.shape[1]):
            valid=a['valid'][:,c];t=a['slope'][valid,c];p=b['slope'][valid,c]
            threshold=thresholds['slope_'+str(seconds)][c]
            direction=lambda z: np.where(z>threshold,1,np.where(z < -threshold,-1,0))
            actual,guess=direction(t),direction(p);dynamic=actual!=0
            confusion=np.zeros((3,3),int)
            np.add.at(confusion,(actual+1,guess+1),1)
            energy=float(np.sum(t[dynamic]**2))
            out.append(dict(patches=len(t),dynamic_patches=int(dynamic.sum()),near_zero_patches=int((~dynamic).sum()),
                mse=float(np.mean((p-t)**2)),dynamic_error_energy=float(np.sum((p[dynamic]-t[dynamic])**2)),
                target_slope_energy=energy,
                dynamic_nmse=float(np.sum((p[dynamic]-t[dynamic])**2)/energy) if energy>0 else None,
                signed_gain=float(np.sum(p[dynamic]*t[dynamic])/energy) if energy>0 else None,
                amplitude_gain=float(np.sqrt(np.sum(p[dynamic]**2)/energy)) if energy>0 else None,
                direction_accuracy=float(np.mean(guess[dynamic]==actual[dynamic])) if dynamic.any() else None,
                false_dynamic=int(np.sum((actual==0)&(guess!=0))),confusion_down_flat_up=confusion.tolist()))
        slopes[str(seconds)]=out
    pair=mask[1:]&mask[:-1];boundary=(np.arange(1,len(target))%(2*rate)==0)[:,None]&pair
    dt=np.diff(target,axis=0);dp=np.diff(predicted,axis=0);diff=[]
    for c in range(target.shape[1]):
        masks={'within':pair[:,c]&~boundary[:,c],'boundary':boundary[:,c]}
        item={}
        for name,m in masks.items():
            item[name+'_pairs']=int(m.sum())
            item[name+'_mse']=float(np.mean((dp[m,c]-dt[m,c])**2)) if m.any() else None
        for name,d in [('target',dt),('prediction',dp)]:
            energy=np.sum(d[pair[:,c],c]**2)
            item[name+'_boundary_energy_fraction']=float(np.sum(d[boundary[:,c],c]**2)/energy) if energy>0 else None
        diff.append(item)
    return dict(channels=channels,slopes=slopes,differences=diff,
        patch_decomposition=dict(total_energy=patch_squared.tolist(),level_energy=patch_level.tolist(),
            shape_energy=patch_shape.tolist(),relative_closure=patch_closure.tolist(),valid_samples=count.tolist(),
            target_slope_per_s=truth['slope'].tolist(),prediction_slope_per_s=estimate['slope'].tolist(),
            target_centered_rms=truth['rms'].tolist(),prediction_centered_rms=estimate['rms'].tolist()))


def render_state_reconstruction_case(export, key):
    import resource
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    began=time.monotonic();export=Path(export)
    info=json.loads((export/(key+'.json')).read_text())
    with np.load(export/(key+'.npz')) as f:arrays=dict(f)
    rate=info['rate'];t=np.arange(len(arrays['x']))/rate
    short={'eeg_fnirs_single_trial':'Single-Trial','refed':'REFED','simultaneous_eeg_nirs':'Simultaneous','visual_cognitive_motivation':'Visual'}
    name=short[info['dataset']];tags=', '.join(info['selection_tags'])
    plt.rcParams.update({'font.size':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9})
    fig,axes=plt.subplots(3,2,figsize=(14,10.4))
    fig.subplots_adjust(left=.075,right=.985,bottom=.06,top=.84,hspace=.52,wspace=.22)
    fig.suptitle(f"{info['modality'].upper()} | {name} | {info['subject']} | {info['record']} | {info['position']} | {tags}",fontsize=14,y=.985)
    fig.text(.5,.947,f"Observed input: whole 0–30 s | parent interval [{info['start_s']},{info['start_s']+30}) s | "
             f"{rate} Hz | amplitudes use frozen training X scale",ha='center',fontsize=10.5)
    colors=['#343a40','#0072B2','#D55E00'];handles=[]
    for i,component in enumerate(('s','o','x')):
        target=arrays[component];hard=arrays['hard_'+component];continuous=arrays['continuous_'+component]
        title={'s':'S target / reconstruction','o':'O target / reconstruction','x':'X / reconstructed S+O'}[component]
        if info['modality']=='hb':
            limits=state_curve_limits(target,hard,continuous,*([arrays['baseline_'+component]] if 'baseline_'+component in arrays else []))
        for j,ax in enumerate(axes[i]):
            channel=j if info['modality']=='hb' else 0
            begin,end=(14.,16.) if info['modality']=='eeg' and j else (0.,30.)
            take=(t>=begin)&(t<end)
            lines=[]
            linespec=[(target,colors[0],'-',.85,'Target / observed X'),
                    (hard,colors[1],'-',1.0,info.get('hard_label','Q-SO integer-ID decode')),
                    (continuous,colors[2],'--',.95,'Continuous S/O student')]
            if 'baseline_'+component in arrays:
                linespec.append((arrays['baseline_'+component],'#8b58a0',':',1.,info.get('baseline_label','Previous 500-step Q-SO K128')))
            for value,color,style,width,label in linespec:
                shown=np.where(arrays['mask'][take,channel],value[take,channel],np.nan)
                lines.append(ax.plot(t[take],shown,color=color,ls=style,lw=width,label=label)[0])
            if not handles:handles=lines
            invalid=~arrays['mask'][:,channel];edges=np.diff(np.r_[False,invalid,False].astype(int))
            for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                ax.axvspan(a/rate,b/rate,color='#b9bfc6',alpha=.35,zorder=0)
            for boundary in np.arange(2,30,2):
                if begin<boundary<end:ax.axvline(boundary,color='#9da6ad',lw=.45,ls=':',alpha=.5)
            ax.axhline(0,color='#777',lw=.5,alpha=.5)
            h=info['metrics']['hard'][component]['channels'][channel]
            comparator='baseline' if 'baseline' in info['metrics'] else 'continuous'
            c=info['metrics'][comparator][component]['channels'][channel]
            def value(x):return 'NA' if x is None else f'{x:.3f}'
            if info['modality']=='eeg':
                head=title+(' | full 30 s' if j==0 else ' | central 2 s, own y scale')
            else:head=title+' | '+('HbO' if j==0 else 'HbR')
            pair='new/old' if comparator=='baseline' else 'hard/cont'
            ax.set_title(head+f"\n30 s NRMSE {pair}: {h['NRMSE']:.3f}/{c['NRMSE']:.3f}; r: {value(h['correlation'])}/{value(c['correlation'])}",fontsize=10)
            ax.set(xlabel='Time within visible input (s)',ylabel='Amplitude / training X scale',xlim=(begin,end))
            if info['modality']=='hb':ax.set_ylim(*limits)
            ax.grid(alpha=.17)
    handles.append(Patch(facecolor='#b9bfc6',alpha=.35,label='Unscored source / filter support'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.924),ncol=3 if 'baseline' in info['metrics'] else 4,fontsize=9,frameon=False)
    figure=key+'.png';fig.savefig(export/figure,dpi=220);plt.close(fig)
    caption=(f"{name}，{info['subject']}，{info['record']}，{info['position']}；父窗{info['duration_s']:g}秒中的[{info['start_s']},{info['start_s']+30})秒。"
        '黑线为缓存条件S/O目标或原始观测X，蓝线为完整15-ID序列解码，橙虚线为独立训练的连续学生；模型仅见本模态30秒X。'
        'NRMSE使用对应分量/色团的冻结训练方差，r为相关；两者均按完整30秒有效掩码计算。灰区不计分。'
        +('EEG右列固定放大14–16秒并独立设置纵轴；放大图未重新拟合。' if info['modality']=='eeg' else 'HbO/HbR分列，同一分量共用纵轴。'))
    if 'baseline' in info['metrics']:
        caption+='紫点线为图例标注的父版本；标题比较新/旧硬模型，橙线为独立连续参照。'
    page=dict(figure=figure,title=f"{info['modality'].upper()} · {name} · {info['position']} · {','.join(info['selection_tags'])}",
              caption=caption,book=key,category='reconstruction')
    # Long supplement labels remain visible in the figure and case metadata; keep the PDF title compact.
    if any(len(tag)>15 for tag in info['selection_tags']):page['title']=f"{info['modality'].upper()} · {name} · {info['position']} · 补充样例"
    return dict(status='completed',key=key,page=page,seconds=time.monotonic()-began,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def assemble_state_reconstruction_atlas(export, results, plan):
    import html
    export=Path(export);results=sorted(results,key=lambda r:r['key']);pages=plan.get('overview_pages',[])+[r['page'] for r in results]
    checks=_state_bitmap_pdf(export,pages,'SO_RECONSTRUCTION_ATLAS.pdf')
    blocks=[];table=[]
    for r in results:
        info=json.loads((export/(r['key']+'.json')).read_text());page=r['page']
        identity=f"{info['modality'].upper()} / {info['dataset']} / {info['record']} / {info['position']} / {','.join(info['selection_tags'])}"
        models=('hard','baseline','continuous') if plan.get('comparison') else ('hard','continuous')
        table.append('<tr><td><a href="#'+r['key']+'">'+html.escape(identity)+'</a></td>'+''.join(
            '<td>'+' / '.join(f"{info['metrics'][m][c]['NRMSE']:.3f}" for m in models)+'</td>' for c in ('s','o','x'))+'</tr>')
        blocks.append(f'<section id="{r["key"]}"><h2>{html.escape(page["title"])}</h2><a href="{page["figure"]}"><img loading="lazy" src="{page["figure"]}"></a><p>{html.escape(page["caption"])}</p><p><a href="{r["key"]}.json">身份、误差与核验</a> · <a href="{r["key"]}.npz">完整曲线、latent、整数ID与掩码</a></p></section>')
    intro=('本图集直接比较冻结tokenizer的S、O输出及两支之和。蓝线来自K128硬量化模型的整数ID解码；橙虚线来自独立训练、等曝光500步的连续S/O学生。'
           '两者输入均为本模态完整30秒观测，绘图没有重新拟合、平滑或逐窗重新定标。S/O目标是缓存条件估计及残差；原始X采用统一200Hz EEG或10Hz Hb视图。'
           '样例按每库每模态的Q-SO总X误差P10/P50/P90选取，另补充REFED短窗和Visual Probe1:CH23；排序单位是位置×窗口，选例用于解释。'
           'holdout为四名训练被试各父窗的留出时间块；完整候选分母保存在selection.json。')
    intro=plan.get('intro',intro)
    if plan.get('model_comparison'):
        model_rows=[]
        for r in plan['model_comparison']['records']:
            health='连续参照' if r['health_pass'] is None else ('通过' if r['health_pass'] else '未通过')
            model_rows.append('<tr><td>'+html.escape(r['modality']+' / '+r['architecture']+' / '+r['candidate'])+'</td>'+
                ''.join(f'<td>{r["macro"][k]:.4f}</td>' for k in ('s','o','x'))+'<td>'+health+'</td></tr>')
        intro+='<table><tr><th>完整模型矩阵</th><th>S NMSE</th><th>O NMSE</th><th>X NMSE</th><th>健康筛查</th></tr>'+''.join(model_rows)+'</table>'
        for modality,r in plan['model_comparison']['selected'].items():
            intro+=f'<p>{modality.upper()} 综合误差排序首位：{r["architecture"]} / {r["candidate"]}；重建目标'+('通过' if r['reconstruction_goal'] else '未通过')+'；健康筛查'+('通过' if r['health_pass'] else '未通过')+'。</p>'
    labels='新硬/旧硬/新连续' if plan.get('comparison') else '硬/连续'
    table_html='<table><tr><th>样例</th>'+''.join(f'<th>{c} NRMSE：{labels}</th>' for c in ('S','O','X'))+'</tr>'+''.join(table)+'</table>'
    table_html+=''.join(f'<h2>{html.escape(p["title"])}</h2><img src="{p["figure"]}"><p>{html.escape(p["caption"])}</p>' for p in plan.get('overview_pages',[]))
    style='body{font:16px/1.7 system-ui,sans-serif;background:#edf1f5;color:#17283a;margin:0}main{max-width:1500px;margin:auto;padding:20px}section{background:white;margin:18px 0;padding:24px;border-radius:8px}img{width:100%;height:auto}a{color:#126da1}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #d5dce4;text-align:left}'
    (export/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>S/O与总信号重建曲线</title><style>'+style+'</style><main><section><h1>S/O目标与双分支总重建：逐窗对比</h1><p>'+intro+'</p><p><a href="SO_RECONSTRUCTION_ATLAS.pdf">完整PDF图集</a> · <a href="selection.json">选例规则与分母</a></p>'+table_html+'</section>'+''.join(blocks)+'</main></html>',encoding='utf-8')
    _json(export/'figure_index.json',dict(pages=pages,selection=plan['rule']))
    return dict(status='completed',cases=len(results),figures=len(pages),pdf_pages=len(checks),
                pdf='SO_RECONSTRUCTION_ATLAS.pdf',index='index.html',new_training=False)


def render_state_refinement_overview(export, comparison):
    """Paired frozen-scale error ratios; no examples are reselected for this comparison."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    export=Path(export)
    fig,axes=plt.subplots(2,3,figsize=(16,8.5))
    fig.subplots_adjust(left=.065,right=.985,bottom=.13,top=.88,wspace=.27,hspace=.65)
    fig.suptitle('Frozen-panel reconstruction | 30 s context | one ID / branch / 2 s',fontsize=17)
    fig.text(.5,.93,'Bar height: NMSE / previous Q-SO NMSE (lower is better) | numbers above bars: actual NMSE | 4 datasets equally weighted',ha='center',fontsize=10)
    order=[('dense_long','continuous'),('dense_long','vq128'),('dense_long','vq512'),('waveform','continuous'),('waveform','vq128'),('waveform','vq512')]
    labels=['Old\nK128','Dense\nC','Dense\nK128','Dense\nK512','Wave\nC','Wave\nK128','Wave\nK512']
    colors=['#888888','#a8bdd0','#5795c8','#245a8b','#f1ba80','#e99242','#af5e20']
    for i,modality in enumerate(('eeg','hb')):
        rows=[next(r for r in comparison['records'] if r['modality']==modality and r['architecture']==a and r['candidate']==c) for a,c in order]
        for j,component in enumerate(('s','o','x')):
            ax=axes[i,j];baseline=rows[0]['baseline_macro'][component]
            actual=[baseline,*[r['macro'][component] for r in rows]];ratio=np.array(actual)/baseline
            ax.bar(np.arange(7),ratio,color=colors)
            ax.axhline(1,color='#333',ls='--',lw=1)
            ax.set(xticks=np.arange(7),xticklabels=labels,ylabel='NMSE relative to old K128',ylim=(0,max(1.25,float(ratio.max())*1.2)))
            ax.set_title(modality.upper()+' — '+{'s':'S component','o':'O component','x':'S+O versus X'}[component],fontsize=12)
            for k,(v,h) in enumerate(zip(actual,ratio)):ax.text(k,h+.025,f'{v:.3f}',ha='center',fontsize=9)
            ax.tick_params(axis='x',labelsize=9);ax.grid(axis='y',alpha=.2)
    fig.text(.5,.025,'Old: 500 steps; Dense/Wave: 4000 steps | K128: 7 bit/s; K512: 9 bit/s (both branches) | same training subjects, development time blocks',ha='center',fontsize=10)
    verdicts=[]
    for m in ('eeg','hb'):
        r=comparison['selected'][m];change=100*(1-r['relative_error']['x']);status='pass' if r['health_pass'] else 'FAIL'
        verdicts.append(f"{m.upper()} selected: {r['architecture']} {r['candidate']} | X error reduction {change:.1f}% | health {status}")
    fig.text(.5,.49,'\n'.join(verdicts),ha='center',va='center',fontsize=11,color='#333333',linespacing=1.6)
    fig.savefig(export/'reconstruction_overview.png',dpi=220);plt.close(fig)
    return dict(figure='reconstruction_overview.png',title='固定开发面板：S、O与总重建的改良幅度',book='comparison',category='overview',
        caption='每行一个模态、每列一个评分目标；柱高为四库等权NMSE相对上一轮K128之比，虚线1为原误差，越低越好。柱顶为实际NMSE。C为连续参照；Dense为原结构延长训练，Wave为卷积波形路径、增宽及损失改良组合。新模型均4000步，旧模型500步；K128/K512双支码率为7/9bit每秒。全部位置包含于主表，Visual高误差位置保留；本轮为四名训练被试的同被试时间留出开发比较。健康筛查与重建排序分别判断：中间文字保留两模态的通过/失败。')


def separation_wave_metrics(x,s,o,scale,eog=None):
    """Natural waveform diagnostics; no clean truth is assumed."""
    from scipy.signal import butter,sosfiltfilt
    x,s,o=(np.asarray(v,float) for v in (x,s,o));a=np.asarray(scale,float);sl=slice(6000,10000)
    def high(v):return sosfiltfilt(butter(4,[25,70],fs=200,btype='bandpass',output='sos'),v,axis=0)[sl]
    xf,of,sf=high(x),high(o),high(s)
    result=dict(change_NMSE=float(np.mean(((s[sl]-x[sl])/a)**2)),
        S_RMS_ratio=float(np.sqrt(np.mean(s[sl]**2)/np.mean(x[sl]**2))),
        O_RMS_uV=float(np.sqrt(np.mean(o[sl]**2))),O_fine_RMS_uV=float(np.sqrt(np.mean(of**2))),
        O_fine_RMS_ratio=float(np.sqrt(np.mean(of**2)/max(np.mean(xf**2),1e-20))),
        S_fine_RMS_ratio=float(np.sqrt(np.mean(sf**2)/max(np.mean(xf**2),1e-20))),
        O_roughness=float(np.sqrt(np.mean((np.diff(o[sl],axis=0)/a)**2))),
        closure_max_abs=float(np.max(abs(s+o-x))))
    if eog is not None:
        reference=sosfiltfilt(butter(4,[.5,15],fs=200,btype='bandpass',output='sos'),eog,axis=0)[sl]
        b=reference-reference.mean(0)
        for key,value in [('raw',x),('S',s),('O',o)]:
            v=value[sl]-value[sl].mean(0);den=np.sqrt(np.sum(v*v,axis=0)[:,None]*np.sum(b*b,axis=0)[None])
            corr=np.divide(v.T@b,den,out=np.zeros_like(den),where=den>1e-20)
            result[key+'_EOG_correlation']=float(np.median(np.max(abs(corr),axis=1)))
    return result


def separation_change_metrics(base,changed,n,kind,*,base_o=None,changed_o=None,
                              score_slice=slice(6000,10000)):
    """NRMSE is relative to the known change, not natural EEG purity."""
    sl=score_slice;n=np.asarray(n,float)[sl]
    delta=np.asarray(changed,float)[sl]-np.asarray(base,float)[sl]
    expected=n if kind.startswith('neural') else 0.
    energy=float(np.mean(n*n))
    error=float(np.mean((delta-expected)**2)/energy)
    projection=float(np.mean(delta*n)/energy)
    delta_o=n-delta if base_o is None else np.asarray(changed_o,float)[sl]-np.asarray(base_o,float)[sl]
    expected_o=0. if kind.startswith('neural') else n
    return dict(change_NMSE=error,S_signed_gain=projection,
        O_change_NMSE=float(np.mean((delta_o-expected_o)**2)/energy),
        O_signed_gain=float(np.mean(delta_o*n)/energy),
        delta_O_correlation=float(np.corrcoef(delta_o.ravel(),n.ravel())[0,1]) if np.std(delta_o)>1e-12 else 0.,
        injection_RMS_uV=float(np.sqrt(energy)))


def separation_decode_metrics(target,decoded,scale):
    """Separate actual fine texture from reconstruction/boundary errors."""
    from scipy.signal import butter,sosfiltfilt
    target,decoded=np.asarray(target,float),np.asarray(decoded,float);scale=np.asarray(scale,float)
    error=decoded-target
    high=lambda v:sosfiltfilt(butter(4,[25,70],fs=200,btype='bandpass',output='sos'),v,axis=0)
    a,b=high(target),high(decoded);den=max(float(np.mean(a*a)),1e-20)
    boundaries=np.arange(20,len(target),20)
    difference=np.diff(error,axis=0)
    return dict(NMSE=float(np.mean((error/scale)**2)),
        fine_error_NMSE=float(np.mean((a-b)**2)/den),fine_RMS_ratio=float(np.sqrt(np.mean(b*b)/den)),
        fine_correlation=float(np.corrcoef(a.ravel(),b.ravel())[0,1]) if np.std(a)>1e-10 and np.std(b)>1e-10 else 0.,
        boundary_error_RMS=float(np.sqrt(np.mean((difference[boundaries-1]/scale)**2))),
        waveform_correlation=float(np.corrcoef(target.ravel(),decoded.ravel())[0,1]) if np.std(target)>1e-10 else None)



def summarize_voltage_power_cap_examples(entries,fits,*,planned_new_fits=16):
    """Five displayed pairs, with failed cases retained in the denominator."""
    rows=[]
    def read(path):
        with np.load(path,allow_pickle=False) as archive:return dict(archive)
    for entry in entries:
        folder=Path(entry['run']);case=entry['case']
        row=dict(subject=case['subject'],case_id=case['id'],run=str(folder),
            case=case,reused=entry['reused'],status='failed',display_channel=entry.get('display_channel','Fp1'))
        try:
            target_paths=[folder/'targets'/f'physical_refitted_reference__{v}__cap1.npz'
                for v in ('baseline','ocular')]
            diagnostics=[json.loads(p.with_suffix('.json').read_text()) for p in target_paths]
            if any(d['status']!='completed' for d in diagnostics):
                raise ValueError('at least one paired voltage target failed; see target diagnostics')
            inputs=read(folder/'inputs.npz');base,changed=map(read,target_paths)
            n=inputs['n_ocular'][6000:10000];display=entry.get('display_channel','Fp1')
            fp=list(inputs['channels']).index(display);channels=len(inputs['channels'])
            metrics={}
            for view,idx in [(display,[fp]),(f'all{channels}',list(range(channels)))]:
                m=separation_change_metrics(base['s_score'][:,idx],changed['s_score'][:,idx],n[:,idx],
                    'ocular',base_o=base['o_score'][:,idx],changed_o=changed['o_score'][:,idx],
                    score_slice=slice(None))
                m.update(S_error_NRMSE=float(np.sqrt(m['change_NMSE'])),
                    S_change_RMS_uV=float(np.sqrt(np.mean((changed['s_score'][:,idx]-base['s_score'][:,idx])**2))))
                metrics[view]=m
            reference=inputs['x'][6000:10000]
            closure=max(float(np.max(abs(base['s_score']+base['o_score']-reference))),
                float(np.max(abs(changed['s_score']+changed['o_score']-reference-n))))
            physical=json.loads((folder/'physical_fits.json').read_text())
            coordinates=json.loads((folder/'coordinates.json').read_text())
            metric_roi=entry.get('metric_roi',1)
            sd=np.array(coordinates[case['rows'][metric_roi]['coordinate']]['sd'])
            quality={}
            for variant in ('baseline','ocular'):
                fit=next(f for f in physical if f['variant']==variant and f['roi']==metric_roi)
                if fit['status']!='completed':raise ValueError('display-region fit is not accepted')
                a=read(fit['array_path']);err=(a['physical']-a['u'])/sd
                quality[variant]={label:np.sqrt(np.mean(err[sl]**2,axis=0)).tolist()
                    for label,sl in [('score30_50',slice(300,500)),('full899',slice(0,899))]}
            ac=reference-reference.mean(0);s=base['s_score'];ds=(s-reference)-(s-reference).mean(0)
            row.update(status='completed',metrics=metrics,model_NRMSE_order=['EEG_proxy','HbO','HbR'],
                model_NRMSE=quality,display_channel=display,channel_count=channels,metric_roi=metric_roi,
                closure_max_abs_uV=closure,
                baseline_AC_distortion=float(np.linalg.norm(ds)/np.linalg.norm(ac)),
                baseline_AC_RMS_ratio=float(np.linalg.norm(s-s.mean(0))/np.linalg.norm(ac)),
                target_diagnostics=diagnostics,
                O_total_injection_correlation=float(np.corrcoef(changed['o_score'][:,fp],n[:,fp])[0,1]),
                target_paths=list(map(str,target_paths)))
            if metric_roi==1:
                row.update(frontal_NRMSE=quality,frontal_NRMSE_order=['EEG_proxy','HbO','HbR'])
        except Exception as exc:row['error']=str(exc)
        rows.append(row)
    return dict(schema='voltage_power_cap_examples_results_v1',status='completed',
        counts=dict(planned_pairs=len(entries),completed_pairs=sum(r['status']=='completed' for r in rows),
            failed_pairs=sum(r['status']!='completed' for r in rows),independent_subjects=len(entries),
            new_planned_fits=planned_new_fits,new_terminal_fits=len(fits),
            new_accepted_fits=sum(f['status']=='completed' for f in fits),
            new_failed_fits=sum(f['status']!='completed' for f in fits)),
        rows=rows,score_seconds=[30,50],
        metric_definitions=dict(S_error='norm(S_after-S_before)/norm(known addition), lower is better',
            O_error='norm(O_after-O_before-known addition)/norm(known addition), equals S_error by closure',
            O_total='O_after contains the original residual as well as its response to the injection',
            NRMSE='RMSE divided by the original frozen training SD, never the optimizer-weighted SD',
            units='native EEG microvolts; cached Hb converted with its retained coordinate, no extra scaling'),
        interpretation='illustrative paired examples, not independent confirmation after choosing this setting')



def render_voltage_power_cap_examples(out,export):
    """Bitmap gallery; fourth row compares paired O change with the injection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties,fontManager
    from experiments.scripts.render_source_diagnostics_report import render_report
    out,export=Path(out),Path(export)
    if export.exists():raise FileExistsError('use a fresh examples export')
    export.mkdir(parents=True)
    summary=json.loads((out/'summary.json').read_text());plan=json.loads((out/'plan.json').read_text())
    posterior=summary['schema']=='voltage_power_cap_posterior_results_v1'
    number_of_cases=len(summary['rows']);label_start=ord(summary.get('label_start','A'))
    channel_count=summary.get('channels',16)
    display=lambda row:row.get('display_channel','Fp1')
    source=Path(plan['reference_run'])
    original=json.loads((source.parent/'result.json').read_text())
    zoom=original['display']['zoom_seconds'];time_s=np.arange(6000,10000)/200
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':FontProperties(fname=str(font)).get_name(),'font.size':13,
        'axes.labelsize':14,'axes.titlesize':15,'xtick.labelsize':11,'ytick.labelsize':11,
        'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,
        'figure.facecolor':'white','savefig.facecolor':'white','text.color':'#18324d','axes.labelcolor':'#18324d'})
    blue,red,gray,green='#3268b2','#c04c51','#657586','#168374'
    def read(path):
        with np.load(path,allow_pickle=False) as archive:return dict(archive)
    prepared={}
    for row in summary['rows']:
        if row['status']!='completed':continue
        inputs=read(Path(row['run'])/'inputs.npz');fp=list(inputs['channels']).index(display(row))
        before,after=[read(p) for p in row['target_paths']]
        prepared[row['subject']]=dict(x=inputs['x'][6000:10000,fp],n=inputs['n_ocular'][6000:10000,fp],
            s0=before['s_score'][:,fp],s1=after['s_score'][:,fp],
            o0=before['o_score'][:,fp],o1=after['o_score'][:,fp])
    values=[v for a in prepared.values() for v in [a['x'],a['x']+a['n'],a['s0'],a['s1'],a['o0'],a['o1'],a['o1']-a['o0'],a['n']]]
    lo=min((float(v.min()) for v in values),default=-1);hi=max((float(v.max()) for v in values),default=1)
    pad=.08*max(hi-lo,1);limits=(lo-pad,hi+pad)
    provenance=[];figures={}
    for number,row in enumerate(summary['rows']):
        subject=row['subject'];label=chr(label_start+number);electrode=display(row)
        if row['status']!='completed':continue
        a=prepared[subject];m=row['metrics'][electrode]
        fig,axes=plt.subplots(4,2,figsize=(11.2,11.2),sharex='col',sharey=True)
        fig.subplots_adjust(left=.11,right=.975,top=.86,bottom=.075,hspace=.24,wspace=.14)
        fig.text(.055,.97,f'{label} · {subject} · {electrode}：S/O 与新增眼电',fontsize=22,weight='bold',va='top')
        fig.text(.055,.923,f'HbO/HbR ×10，EEG ×0.1；功率容差 1；本组 {number_of_cases} 个样例统一电压刻度。',fontsize=13,color=gray)
        fig.text(.055,.887,f"{electrode} 的 S 变化 RMS {m['S_change_RMS_uV']:.2f} µV；ΔO 误差 {m['S_error_NRMSE']:.4f}；ΔO–注入相关 {m['delta_O_correlation']:.4f}",fontsize=12,color=gray)
        panels=[(a['x'],a['x']+a['n'],'输入 EEG'),(a['s0'],a['s1'],'S 目标'),(a['o0'],a['o1'],'O 残差目标')]
        for r in range(4):
            for c,xlim in enumerate([(30,50),zoom]):
                ax=axes[r,c]
                if r<3:
                    u,v,name=panels[r]
                    ax.plot(time_s,u,color=blue,lw=1.1,label='未额外加眼电')
                    ax.plot(time_s,v,color=red,lw=1.1,ls='--',label='加入眼电')
                else:
                    name='O 变化 / 新增眼电'
                    ax.plot(time_s,a['o1']-a['o0'],color=green,lw=1.5,label='O 的变化 ΔO')
                    ax.plot(time_s,a['n'],color=gray,lw=1.2,ls=':',label='额外加入的眼电')
                ax.set_xlim(*xlim);ax.set_ylim(*limits);ax.grid(alpha=.14)
                ax.axhline(0,color='#bac5ce',lw=.6,zorder=0)
                ax.legend(fontsize=9.5,frameon=False,ncol=2 if r<3 else 1,loc='upper right')
                if c==0:ax.set_ylabel(name+'\n（µV）')
                if r==0:ax.set_title('完整评分段 [30,50) 秒' if c==0 else '相同注入峰附近放大',loc='left',fontsize=14)
                if r==3:ax.set_xlabel('窗口相对时间（秒）')
        suffix='_'+electrode if posterior else ''
        path=export/f'{label}_{subject}{suffix}_voltage.png';fig.savefig(path,dpi=250);plt.close(fig)
        figures[subject]=path.name
        peak=int(np.argmax(abs(a['n'])))
        provenance.append(dict(figure=path.name,case_id=row['case_id'],input=str(Path(row['run'])/'inputs.npz'),
            targets=row['target_paths'],y_limits_uV=list(limits),score_seconds=[30,50],zoom_seconds=zoom,
            fourth_row=['O_after-O_before','known_added_ocular'],no_frame_or_window_renormalization=True,
            known_peak=dict(time_s=float(time_s[peak]),injection_uV=float(a['n'][peak]),
                delta_S_uV=float(a['s1'][peak]-a['s0'][peak]),delta_O_uV=float(a['o1'][peak]-a['o0'][peak]))))
    # Compact comparison uses paired changes, whose desired responses are known.
    fig,axes=plt.subplots(number_of_cases,2,figsize=(11.2,max(6.5,number_of_cases*2.24)),sharex=True,sharey=True,squeeze=False)
    fig.subplots_adjust(left=.12,right=.975,top=.78 if posterior else .88,bottom=.10 if posterior else .07,hspace=.35,wspace=.15)
    fig.text(.055,.97,f'{number_of_cases} 个固定样例：比较 S 的泄漏与 O 的吸收',fontsize=22,weight='bold',va='top')
    fig.text(.055,.87 if posterior else .924,'灰点线：同一已知眼电；左列蓝线：ΔS；右列绿线：ΔO。编号与逐例图一致。',fontsize=13,color=gray)
    delta_values=[v for a in prepared.values() for v in (a['n'],a['s1']-a['s0'],a['o1']-a['o0'])]
    bound=1.13*max((float(np.max(abs(v))) for v in delta_values),default=1.)
    for i,row in enumerate(summary['rows']):
        subject=row['subject'];label=chr(label_start+i)
        for c in range(2):
            ax=axes[i,c];ax.set_xlim(*zoom);ax.set_ylim(-bound,bound);ax.grid(alpha=.14)
            ax.axhline(0,color='#bac5ce',lw=.6)
            if row['status']=='completed':
                a=prepared[subject];delta=a['s1']-a['s0'] if c==0 else a['o1']-a['o0']
                ax.plot(time_s,a['n'],color=gray,lw=1.1,ls=':')
                ax.plot(time_s,delta,color=blue if c==0 else green,lw=1.6)
                metric=row['metrics'][display(row)]['S_error_NRMSE']
                ax.text(.98,.92,f'误差 {metric:.4f}',ha='right',va='top',transform=ax.transAxes,fontsize=11)
            else:ax.text(.5,.5,'拟合/目标失败；保留该样例',ha='center',va='center',transform=ax.transAxes)
            if c==0:ax.set_ylabel(f'{label} · {subject}\n{display(row)}（µV）',fontsize=12)
            if i==0:ax.set_title('S 变化：期望为 0' if c==0 else 'O 变化：期望等于注入',loc='left')
            if i==number_of_cases-1:ax.set_xlabel('窗口相对时间（秒）')
    overview=export/'00_examples_overview.png';fig.savefig(overview,dpi=250);plt.close(fig)
    provenance.append(dict(figure=overview.name,source=str(out/'summary.json'),y_limits_uV=[-bound,bound],
        zoom_seconds=zoom,failed_pairs_visible=True))
    table=[];quality=[]
    for i,row in enumerate(summary['rows']):
        label=f"{chr(label_start+i)} · {row['subject']}"+(' · '+display(row) if posterior else '')
        if row['status']=='completed':
            fp=row['metrics'][display(row)];all16=row['metrics'][f'all{channel_count}']
            table.append([label,f"{all16['S_error_NRMSE']:.4f}",f"{fp['S_change_RMS_uV']:.2f}",
                f"{fp['delta_O_correlation']:.4f}"])
            for key,support in [('score30_50','30–50秒'),('full899','完整窗口')]:
                quality_source=row.get('model_NRMSE',row.get('frontal_NRMSE'))
                e,hbo,hbr=quality_source['ocular'][key]
                quality.append([label,support,f'{hbo:.4f}',f'{hbr:.4f}',f'{e:.4f}'])
        else:
            table.append([label,'失败','—','—']);quality.append([label,'全部','失败','—','—'])
    def paragraph(text):return dict(type='paragraph',text=text)
    def figure(path,caption):return dict(type='figure',path=path,caption=caption,preserve_bitmap=True)
    counts=summary['counts']
    content=dict(title='Hb 优先联合拟合：五个眼电配对样例',subtitle='原样例增加 O 对比；另外四名验证被试各一段',date='2026-11-06',
        summary=[f"计划展示 5 个独立被试的 5 对窗口，其中 {counts['completed_pairs']} 对完成、{counts['failed_pairs']} 对失败；原 VP002 复用已验收结果，其余四例各取原验证面板首窗。全部预定例子均列出，供展示选图，不按结果替换窗口。"],
        provenance=[f'数值事实源：{out}/summary.json；逐例 inputs.npz、physical_fits.json、fit_attempts.json 与 targets/。'],
        sections=[
        dict(title='设置、比较量与读图方式',blocks=[
            paragraph('EEG/HbO/HbR 标准化平方误差权重为 0.1/10/10；固定 Tak 参数、训练坐标、功率参考和容差 1。每次可见完整 90 秒 EEG 与 Hb，重新估计驱动和四初态；这是一项离线条件重建。评分段为 [30,50) 秒。'),
            paragraph('只向 Fp1 加入同一条已知眼电，评分段 RMS 为 36.13 微伏；其他 EEG 通道和 Hb 不变。波形来自 VP005 的另一保留窗口；因此 VP005 样例与供体属于同一被试。图中未加注入的自然背景并非干净 EEG 真值。'),
            paragraph('前 3 行分别比较输入、S、O 的注入前后波形；新增第 4 行只比较 O 的前后差值 ΔO 与已知眼电。完整 O 含原基底残差，在第 3 行保留；ΔO 与注入的差才是新增干扰的吸收误差。'),
            paragraph('S 泄漏误差为 S 配对变化的范数除以已知注入范数；O 变化误差与它相同，因为 O 等于输入减去 S。范数先汇总所列全部时间点/通道，再取比值，不逐窗重新定标。相关系数只反映形状，不代表幅度正确。'),
            dict(type='table',columns=['样例','全16通道S泄漏↓','Fp1的S变化RMS/微伏','Fp1的ΔO相关↑'],widths=[1.25,1.3,1.55,1.25],rows=table)]),
        dict(title='模型误差：局部展示段与完整拟合支持',page_break=True,blocks=[
            paragraph('下表为前额 ROI 眼电注入后的 HbO/HbR/EEG代理 NRMSE。均先计算各模态观测与完整预测之间的 RMSE，再除以原冻结训练 SD；不使用优化权重改变展示分母。'),
            paragraph('30–50 秒对应图中的展示段，含 200 个 10 Hz 代理/Hb 点；完整窗口使用原拟合掩码的 899 个有效点。原始电压图则为 200 Hz、4000 点。EEG 代理误差与 S/O 电压的眼电泄漏是不同指标。'),
            dict(type='table',columns=['样例','评分支持','HbO','HbR','EEG代理'],widths=[1.25,1.25,1,1,1.25],rows=quality),
            paragraph('选择展示图时同时考虑 S 对眼电的响应、ΔO 对已知增量的误差、基底波形和模型拟合状态。较低的局部 NRMSE 不代表完整窗口同样准确，也不证明自然 S 是干净神经真值。'),
            paragraph('可复用设置由 docs/METHOD_RATIONALE.md 的 Hb-prioritized joint fitting 章节说明；具体合同、完整来源与本次选样见本运行 resolved_config.json、source_manifest.json、plan.json。')]),
        dict(title='固定放大区间下的配对变化',page_break=True,blocks=[
            figure(overview.name,'每行是一名被试的一对窗口；左列检查新增眼电进入 S 的程度，右列检查 ΔO 是否复现已知注入。全图统一坐标，曲线颜色与逐例图一致。'),
            paragraph('这组结果用于挑选展示样例。既有神经样变化损失及最终合成后的功率越界仍需保留；当前样例只改变眼电，没有新增神经保留资格证据。')])],
        sources=[])
    for i,row in enumerate(summary['rows']):
        subject=row['subject'];case=row['case'];timing=case['rows'][0]
        blocks=[]
        if row['status']=='completed':
            blocks.append(figure(figures[subject],'前 3 行蓝实线/红虚线为注入前/后；末行绿实线为 ΔO，灰点线为已知眼电。左右显示完整评分段与同一预定峰附近区间。所有逐例图统一电压刻度。'))
            fp=row['metrics'][display(row)];full=row['metrics'][f'all{channel_count}']
            a=prepared[subject];k=int(np.argmax(abs(a['n'])))
            blocks.append(paragraph(f"全 {channel_count} 通道 S 泄漏误差 {full['S_error_NRMSE']:.4f}；{display(row)} 的 ΔO 误差 {fp['S_error_NRMSE']:.4f}，相关 {fp['delta_O_correlation']:.4f}。在已知注入绝对峰 {time_s[k]:.3f} 秒处，新增眼电为 {a['n'][k]:.2f} 微伏，S 变化为 {a['s1'][k]-a['s0'][k]:.2f} 微伏，ΔO 为 {a['o1'][k]-a['o0'][k]:.2f} 微伏。两路变化共同组成该注入；峰值只是读图定位，误差仍按完整评分段计算。"))
        else:blocks.append(paragraph('该计划样例未形成完整合格输出，未补换窗口。原因：'+row.get('error','未知；见逐项记录')))
        blocks.append(paragraph(f"身份：{case['id']}；EEG 起点 {timing['eeg_start_s']:.3f} 秒，Hb 起点 {timing['hb_start_s']:.3f} 秒；各自连续记录坐标，模型窗口均长 90 秒。"))
        content['sections'].append(dict(title=f"{chr(label_start+i)} · {subject} · {display(row)}"+('（原样例重绘）' if row['reused'] else '（新增样例）'),page_break=True,blocks=blocks))
    if posterior:
        content.update(title='Hb 优先联合拟合：枕区 O1/O2 眼电注入',subtitle='相同眼电波形直接加入枕部电极；末行比较 ΔO 与已知注入',
            summary=[f"两个预定枕区样例：VP002 的 O1、VP005 的 O2；{counts['completed_pairs']} 对完成，{counts['failed_pairs']} 对失败。沿用原窗口与眼电波形，位置改变为枕区。这里只用于说明性选图，不按结果换窗。"])
        blocks=content['sections'][0]['blocks']
        blocks[0]=paragraph('EEG/HbO/HbR 标准化平方误差权重仍为 0.1/10/10。17 电极面板包含左运动区、前额和枕区；枕区采用 POOz 的 Hb 与 POz、O1、O2、Pz、P3、P4 的 EEG。原六名训练被试首窗拟合枕区 PCA/尺度及新的 17 通道功率参考，随后全部冻结；容差仍为 1。')
        blocks[1]=paragraph('同一条来自 VP005 另一窗口的眼电分别只加入 O1 或 O2，评分段 RMS 为 36.13 微伏；因此 VP005 组与供体同被试。其他 EEG 与 Hb 不变。这是受控等幅注入，不代表自然眼电在枕区的幅度或空间分布。完整 90 秒均可见，枕区原始和污染输入各自联合拟合驱动与四初态；其余两个 ROI 在逐点核验输入一致后复用已接受解。')
        blocks[-1]['columns']=['样例','全17通道S泄漏↓','注入电极S变化RMS/微伏','注入电极ΔO相关↑']
        content['sections'][1]['blocks'][0]['text']=content['sections'][1]['blocks'][0]['text'].replace('前额 ROI','枕区 ROI')
        remarks=[]
        for i,row in enumerate(summary['rows']):
            if row['status']!='completed':continue
            eeg,hbo,hbr=row['model_NRMSE']['ocular']['full899']
            remarks.append(f"{chr(label_start+i)} 的完整窗口 EEG代理/HbO/HbR NRMSE 为 {eeg:.4f}/{hbo:.4f}/{hbr:.4f}")
        content['sections'][1]['blocks'].append(paragraph('；'.join(remarks)+'。数值收敛与观测重建准确是两件事；F 的模型误差明显较高，应作为高误差样例解释。局部眼电泄漏还须结合全 17 通道指标，以检查未注入电极的响应。'))
        content['sections'][2]['blocks'][-1]=paragraph('本组更换了 ROI 和训练参考的通道集合；与前额图的差异不能只归因于脑区位置。眼电稳定性和神经信息保留仍是不同问题，本次没有新增神经保留对照。')
    path=export/'report_content.json';path.write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n')
    (export/'figure_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    checks=render_report(path,export/'report')
    return dict(status='completed',export=str(export),report_pages=checks['exports']['TECHNICAL_REPORT.pdf']['pages'],
        completed_figures=len(figures),planned_examples=len(summary['rows']))



def voltage_power_cap_intermediate_replay(x,model,proxy,display_index,*,frame_index=417,band_index=0,cap=1.):
    """Replay the frozen voltage computation and retain compact explanatory traces.

    No fitting occurs. The returned per-band overlap-add is independently
    accumulated and checked against the producer's sum-before-overlap-add order.
    """
    from src.data.dual_component_tokenizer import (
        voltage_power_cap_bank,voltage_power_cap_frames,voltage_power_cap_reference,
        project_voltage_second_moment,
    )
    x=np.asarray(x,float);channels=x.shape[1]
    if x.shape!=(18000,channels) or not 0<=display_index<channels:
        raise ValueError('one 90-second native voltage window and a displayed channel required')
    a=np.asarray(model['projection']);bank=voltage_power_cap_bank(x)
    p=np.mean(bank[:,:5].reshape(900,20,5,channels)**2,axis=1).transpose(0,2,1)
    log=np.log(np.maximum(p,1e-20));current_baseline=log[:50].mean(0)
    ell=log-current_baseline
    frames,t=voltage_power_cap_frames(bank)
    reference=voltage_power_cap_reference(model,proxy,t)
    aligned=np.stack([np.interp(t,np.arange(900)/10,proxy[:,j]) for j in range(3)],axis=1)
    completed=model['mean']+(aligned-a@model['mean'])@model['lift'].T
    scale=np.sqrt(reference)
    covariance=scale[..., :,None]*model['correlation'][None]*scale[...,None,:]
    projected,diag=project_voltage_second_moment(frames,covariance,cap)
    h=np.hanning(201)[:-1];acc=np.zeros((18200,6,channels));direct=np.zeros((18200,channels));den=np.zeros(18200)
    for k,block in enumerate(projected):
        start=k*20
        acc[start:start+200]+=block.transpose(2,0,1)*h[:,None,None]
        direct[start:start+200]+=block.sum(0).T*h[:,None]
        den[start:start+200]+=h
    section=slice(100,18100);contribution=acc[section]/den[section,None,None]
    s=contribution.sum(1);o=x-s
    direct=direct[section]/den[section,None]
    selected=frames[frame_index,band_index];cov=covariance[frame_index,band_index]
    d=np.sqrt(np.diag(cov));r=cov/d[:,None]/d[None,:];root=np.linalg.cholesky(r)
    whitened=np.linalg.solve(root,selected/d[:,None])
    left,singular,right=np.linalg.svd(whitened,full_matrices=False)
    limit=np.sqrt(cap*200);clipped=np.minimum(singular,limit)
    gains=np.minimum(1.,limit/np.maximum(singular,np.finfo(float).tiny))
    transform=(d[:,None]*root)@(left*gains[None,:])@left.T@np.linalg.solve(root,np.diag(1/d))
    selected_output=projected[frame_index,band_index]
    final_bank=voltage_power_cap_bank(s);final_frames,_=voltage_power_cap_frames(final_bank)
    # Independent largest-eigenvalue audit, without applying another projection.
    white_final=np.linalg.solve(np.linalg.cholesky(model['correlation'])[None],
        final_frames/scale[..., :,None])
    final_ratio=np.linalg.eigvalsh(white_final@white_final.swapaxes(-1,-2)/200)[...,-1]/cap
    final_power=np.mean(final_bank[:,:5].reshape(900,20,5,channels)**2,axis=1).transpose(0,2,1)
    final_log=np.log(np.maximum(final_power,1e-20));final_log-=final_log[:50].mean(0)
    return dict(time_native=np.arange(18000)/200,time_proxy=np.arange(900)/10,time_frame=t,
        raw_voltage=x[:,display_index],carrier_voltage=bank[:,:,display_index],
        power100ms=p,current_baseline_log_power=current_baseline,log_features=ell,
        observed_proxy=ell.reshape(900,-1)@a.T,aligned_proxy=aligned,
        completed_log_power=completed,reference_power=reference,
        input_frame_RMS=np.sqrt(np.mean(frames[:,:,display_index]**2,axis=-1)),
        projected_frame_RMS=np.sqrt(np.mean(projected[:,:,display_index]**2,axis=-1)),
        local_input_ratio=diag['max_input_eigenvalue']/cap,local_output_ratio=diag['max_output_eigenvalue']/cap,
        minimum_gain=diag['minimum_gain'],clipped_directions=diag['clipped_directions'],
        band_contributions=contribution[:,:,display_index],S=s[:,display_index],O=o[:,display_index],
        S_score_all=s[6000:10000],O_score_all=o[6000:10000],ola_denominator=den[section],
        final_reanalysis_ratio=final_ratio,final_proxy=final_log.reshape(900,-1)@a.T,
        frame_input=selected,frame_projected=selected_output,
        frame_time=(frame_index*20-100+np.arange(200))/200,
        svd_input=singular,svd_output=clipped,svd_gain=gains,
        raw_space_matrix=transform,frame_reference_covariance=cov,
        display_index=np.array(display_index),frame_index=np.array(frame_index),band_index=np.array(band_index),
        max_closure=np.array(np.max(abs(s+o-x))),max_overlap_order_difference=np.array(np.max(abs(s-direct))),
        max_selected_matrix_difference=np.array(np.max(abs(transform@selected-selected_output))),
        max_lift_equality_error=np.array(np.max(abs(completed@a.T-aligned))))


def _voltage_method_replay_case(payload):
    """Read only a named accepted example and validate its deterministic replay."""
    import ast
    from src.data import dual_component_tokenizer as data
    run,variant,display,destination,source_snapshot=payload
    run,destination=Path(run),Path(destination)
    def read(p):
        with np.load(p,allow_pickle=False) as z:return dict(z)
    names=['voltage_power_cap_bank','voltage_power_cap_frames','voltage_power_cap_reference',
        'project_voltage_second_moment','voltage_power_cap_target']
    def definitions(path):
        tree=ast.parse(Path(path).read_text())
        return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    current=definitions(data.__file__);frozen=definitions(Path(source_snapshot)/'src/data/dual_component_tokenizer.py')
    if any(current[n]!=frozen[n] for n in names):
        raise ValueError('numerical owner differs from the retained run; a separate replay migration is required')
    started=time.monotonic();inputs=read(run/'inputs.npz');model=read(run/'reference.npz')
    index=list(inputs['channels']).index(display);n=inputs['n_ocular']
    x=inputs['x'] if variant=='baseline' else inputs['x']+n
    proxies=read(run/'proxies.npz')[variant]
    peak=int(np.argmax(abs(n[6000:10000,index])))+6000
    frame_index=int(round(peak/200/.1))
    traces=voltage_power_cap_intermediate_replay(x,model,proxies,index,frame_index=frame_index,band_index=0)
    retained=read(run/'targets'/f'physical_refitted_reference__{variant}__cap1.npz')
    errors={key:float(np.max(abs(traces[new]-retained[key])))
        for new,key in [('S_score_all','s_score'),('O_score_all','o_score'),
            ('reference_power','reference_power'),('final_proxy','final_proxy'),
            ('local_input_ratio','local_input_ratio'),('local_output_ratio','local_output_ratio')]}
    relative_final=float(np.max(abs(traces['final_reanalysis_ratio']-retained['final_reanalysis_ratio'])/
        (1+abs(retained['final_reanalysis_ratio']))))
    if (max(errors[k] for k in ['s_score','o_score','final_proxy'])>1e-8
            or errors['reference_power']>1e-8
            or relative_final>1e-8 or float(traces['max_lift_equality_error'])>1e-9
            or float(traces['max_selected_matrix_difference'])>1e-8):
        raise AssertionError(dict(replay_errors=errors,final_ratio_relative=relative_final))
    destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,**traces)
    record=dict(status='passed',run=str(run),variant=variant,display_channel=display,
        array_path=str(destination),source_snapshot=str(source_snapshot),
        numerical_function_AST_equal=True,retained_array_max_differences=errors,
        independent_final_ratio_relative_difference=relative_final,
        frame_selection='nearest 100 ms centre to fixed known-injection absolute peak; band 1–4 Hz',
        frame_index=frame_index,frame_centre_s=frame_index/10,
        selected_frame_clipped_modes=int(np.sum(traces['svd_gain']<1)),
        selected_frame_singular_max=float(traces['svd_input'].max()),
        selected_frame_minimum_gain=float(traces['svd_gain'].min()),
        overlap_order_error=float(traces['max_overlap_order_difference']),
        closure_error=float(traces['max_closure']),seconds=time.monotonic()-started)
    destination.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return record


def voltage_normalization_statistics(raw,coordinate,*,prediction=None,weights=(.1,10.,10.)):
    """Separate physical-feature scaling, training-SD residual units and weights."""
    raw=np.asarray(raw,float);sd=np.asarray(coordinate['sd'],float);weights=np.asarray(weights,float)
    scale=np.array([coordinate['eeg_factor'],coordinate['hb_factor'],coordinate['hb_factor']])
    if (raw.ndim!=2 or raw.shape[1]!=3 or len(raw)<2 or sd.shape!=(3,) or weights.shape!=(3,)
            or not all(np.isfinite(v).all() for v in (raw,sd,weights,scale))
            or min(sd.min(),weights.min(),scale.min())<=0):
        raise ValueError('finite three-component observations and positive scales required')
    u=raw*scale;standardized=u/sd
    def distribution(x):
        q=np.quantile(x,[.01,.05,.25,.5,.75,.95,.99],axis=0)
        return dict(mean=x.mean(0).tolist(),sd=x.std(0).tolist(),rms=np.sqrt(np.mean(x*x,0)).tolist(),
            minimum=x.min(0).tolist(),maximum=x.max(0).tolist(),
            **{name:y.tolist() for name,y in zip(('p01','p05','p25','median','p75','p95','p99'),q)})
    std=u.std(0);hb_sd=np.sqrt(np.mean(std[1:]**2))
    if std[0]<=0 or hb_sd<=0:raise ValueError('nondegenerate EEG and Hb required')
    result=dict(samples=len(raw),scale=scale.tolist(),training_sd=sd.tolist(),weights=weights.tolist(),
        distributions={key:distribution(x) for key,x in [('before_scale',raw),('model',u),
            ('training_sd_units',standardized),('objective_units',standardized*np.sqrt(weights))]},
        hb_combined_sd=float(hb_sd),hb_to_eeg_sd_ratio=float(hb_sd/std[0]),
        sd_over_training=(std/sd).tolist(),
        window_only_rescaling_factors=dict(eeg=float(.025/std[0]),common_hb=float(.025/hb_sd)))
    if prediction is not None:
        prediction=np.asarray(prediction,float)
        if prediction.shape!=u.shape or not np.isfinite(prediction).all():raise ValueError('prediction shape/finite mismatch')
        residual=(prediction-u)/sd
        result['prediction']=dict(distribution=distribution(prediction),
            sd_ratio=(prediction.std(0)/std).tolist(),
            correlation=[float(np.corrcoef(u[:,j],prediction[:,j])[0,1]) if prediction[:,j].std()>0 else None for j in range(3)],
            nrmse=np.sqrt(np.mean(residual**2,0)).tolist(),
            data_cost_by_modality=(.4*np.sum(weights*residual**2,0)).tolist())
    return result


def voltage_normalization_variance_partition(chunks):
    """Population variance = average within-window variance + mean-shift variance."""
    chunks=[np.asarray(x,float) for x in chunks]
    if not chunks or any(x.ndim!=2 or x.shape[1]!=3 or not len(x) or not np.isfinite(x).all() for x in chunks):
        raise ValueError('nonempty finite three-component windows required')
    lengths=np.array([len(x) for x in chunks]);weights=lengths/lengths.sum()
    means=np.array([x.mean(0) for x in chunks]);mean=np.sum(means*weights[:,None],axis=0)
    within=np.sum(np.array([x.var(0) for x in chunks])*weights[:,None],axis=0)
    between=np.sum((means-mean)**2*weights[:,None],axis=0);total=np.concatenate(chunks).var(0)
    np.testing.assert_allclose(total,within+between,atol=1e-12,rtol=1e-10)
    return dict(samples=int(lengths.sum()),windows=len(chunks),mean=mean.tolist(),pooled_sd=np.sqrt(total).tolist(),
        within_window_variance=within.tolist(),between_window_mean_variance=between.tolist(),
        between_window_fraction=np.divide(between,total,out=np.zeros(3),where=total>0).tolist())


def audit_voltage_power_cap_normalization(front,posterior,export):
    """Distribution audit of the admitted six training/five validation subjects."""
    import shutil
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties,fontManager
    front,posterior,export=map(lambda p:Path(p).resolve(),(front,posterior,export))
    if export.exists():raise FileExistsError('use a fresh normalization audit export')
    def read(p):return json.loads(Path(p).read_text())
    def arrays(p,keys):
        with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in keys}
    summaries=[read(p/'summary.json') for p in (front,posterior)]
    for root,summary in zip((front,posterior),summaries):
        if summary['status']!='completed' or read(root/'verification.json')['status']!='passed':
            raise ValueError('completed verified retained examples required')
    postplan=read(posterior/'plan.json');parent=Path(postplan['parent']);plan=read(parent/'plan.json')
    for value in (postplan,plan):
        if value['test_access']!='forbidden' or value['protected_data']!='forbidden':
            raise ValueError('retained public development boundary required before array reads')
    training=[r for r in plan['rows'] if r['phase']=='fit']
    if ({r['id'] for r in training}!={r['id'] for r in postplan['training']}
            or any(r['split']!='train' for r in training)
            or {r['subject'] for r in training}!=set(('VP001','VP003','VP006','VP007','VP008','VP009'))):
        raise ValueError('six original first training windows required')
    rows=summaries[0]['rows']+summaries[1]['rows']
    if (len(rows)!=7 or any(r['case']['split']!='validation' for r in rows)
            or {r['subject'] for r in rows}!=set(('VP002','VP005','VP014','VP017','VP019'))):
        raise ValueError('seven retained examples from five validation subjects required')
    coordinates=read(parent/'coordinates.json');coordinates.update(read(posterior/'coordinates.json'))
    records=[];groups={};checks=[]
    def observation(feature,hb,indices,co):
        return np.column_stack((feature[:,indices].reshape(900,30)@np.asarray(co['pc']),hb))
    for region in ('left_motor','prefrontal','right_motor','occipital'):
        key='simultaneous_eeg_nirs__'+region;co=coordinates[key];parts=[]
        ispost=region=='occipital';roi=2 if ispost else ('left_motor','prefrontal','right_motor').index(region)
        for entry in training:
            path=(posterior/'prepared' if ispost else parent/'features')/(entry['id']+'.npz')
            feature_key='features_baseline' if ispost else 'features'
            a=arrays(path,[feature_key,'hb','roi_indices'])
            raw=observation(a[feature_key],a['hb'][:,roi],a['roi_indices'][roi],co);parts.append(raw)
            records.append(dict(label=entry['subject'],subject=entry['subject'],case_id=entry['id'],region=region,
                split='train',variant='baseline',source=str(path),statistics=voltage_normalization_statistics(raw[:899],co)))
        scale=np.array([co['eeg_factor'],co['hb_factor'],co['hb_factor']]);u=[x*scale for x in parts]
        partition=voltage_normalization_variance_partition(u)
        np.testing.assert_allclose(partition['pooled_sd'],co['sd'],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose([partition['pooled_sd'][0],np.sqrt(np.mean(np.array(partition['pooled_sd'][1:])**2))],[.025,.025],rtol=1e-10)
        groups[region]=dict(coordinate=co,variance_partition=partition,
            pooled_training=voltage_normalization_statistics(np.concatenate(parts),co),
            statistical_unit='six retained windows, one per training subject; 900 bins per window')
    for label,row in zip('ABCDEFG',rows):
        if row['status']!='completed':
            raise ValueError('retain failed-case evidence explicitly before expanding this audit')
        run=Path(row['run']);case=row['case'];roi=row.get('metric_roi',1);region=case['rows'][roi]['anchor']['region']
        key=case['rows'][roi]['coordinate'];co=read(run/'coordinates.json')[key]
        for name in ('pc','eeg_factor','hb_factor','sd'):
            np.testing.assert_array_equal(co[name],coordinates[key][name])
        fits=read(run/'physical_fits.json')
        a=arrays(run/'inputs.npz',['features_baseline','features_ocular','hb','roi_indices'])
        for variant in ('baseline','ocular'):
            f=next(f for f in fits if f['variant']==variant and f['roi']==roi)
            if f['status']!='completed' or f['observation_weights']!=[.1,10.,10.]:
                raise ValueError('same accepted weighted fit required')
            v=arrays(f['array_path'],['u','physical','visible']);raw=observation(a['features_'+variant],a['hb'][:,roi],a['roi_indices'][roi],co)
            scaled=raw*np.array([co['eeg_factor'],co['hb_factor'],co['hb_factor']])
            np.testing.assert_allclose(scaled,v['u'],rtol=1e-12,atol=1e-12)
            if not v['visible'][:899].all() or v['visible'][-1].any():raise ValueError('899-bin full-modality support required')
            stats=voltage_normalization_statistics(raw[:899],co,prediction=v['physical'][:899])
            cost_error=abs(sum(stats['prediction']['data_cost_by_modality'])-f['fit']['objective_components']['data'])
            if cost_error>1e-8:raise ValueError('weighted data term mismatch')
            checks.append(dict(label=label,variant=variant,observed_replay_error=float(np.max(abs(scaled-v['u']))),data_cost_error=cost_error))
            records.append(dict(label=label,subject=row['subject'],case_id=case['id'],region=region,
                split='validation',variant=variant,source=f['array_path'],statistics=stats))
    export.mkdir(parents=True)
    result=dict(schema='voltage_normalization_audit_v1',status='completed',records=records,training_groups=groups,
        counts=dict(training_subjects=6,validation_subjects=5,unique_subjects=11,training_region_windows=24,
            validation_region_windows=7,validation_conditions=14),checks=checks,
        scope=dict(front=str(front),posterior=str(posterior),parent=str(parent),raw_signal_reads=0,new_SSM_fits=0,
            access='retained train/validation arrays only'),
        definitions=dict(modalities=['EEG_proxy','HbO','HbR'],before_scale_units=['PC1 relative log-power','uM','uM'],
            support='per-window statistics: bins0..898, 899 points; exact calibration replay: all900 bins',
            reference='first5s already subtracted; no full-window mean subtraction in the actual target',
            sd='population SD ddof0 within stated support; RMS includes the mean',
            stages='before_scale -> positive coordinate gain -> divide by original training SD -> multiply sqrt(objective weight)',
            hb_combined_sd='sqrt((SD_HbO^2+SD_HbR^2)/2), not an amplitude average or across-subject pooling',
            window_rescaling='arithmetic factors to give current-window EEG SD and combined Hb SD0.025; no SSM rerun and not a new calibration proposal',
            uncertainty='descriptive retained windows; no timepoint independence, confidence interval, or causal effect of re-normalization'))
    (export/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    flat=[]
    for r in records:
        for stage,distribution in r['statistics']['distributions'].items():
            for j,modality in enumerate(result['definitions']['modalities']):
                flat.append(dict(**{k:r[k] for k in ('label','subject','case_id','region','split','variant','source')},
                    stage=stage,modality=modality,**{k:v[j] for k,v in distribution.items()}))
    with (export/'distribution_statistics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');fontManager.addfont(font.get_file())
    plt.rcParams.update({'font.family':font.get_name(),'font.size':10,'axes.unicode_minus':False})
    primary=[r for r in records if r['split']=='validation' and r['variant']=='baseline']
    labels=[r['label']+' '+r['subject']+'\n'+('前额' if r['region']=='prefrontal' else '枕区') for r in primary]
    colors=['#2469a0','#c24e40','#24826c'];modalities=['EEG 代理','HbO','HbR']
    fig,axs=plt.subplots(2,3,figsize=(12,7.5),sharex=True,sharey='row')
    for i,stage in enumerate(('model','training_sd_units')):
        for j,ax in enumerate(axs[i]):
            stats=[r['statistics']['distributions'][stage] for r in primary]
            boxes=[dict(label=r['label'],med=s['median'][j],q1=s['p25'][j],q3=s['p75'][j],
                whislo=s['p05'][j],whishi=s['p95'][j],fliers=[]) for r,s in zip(primary,stats)]
            ax.bxp(boxes,showfliers=False,patch_artist=True,boxprops=dict(facecolor=colors[j],alpha=.5))
            ax.axhline(0,color='grey',ls=':',lw=.8);ax.grid(alpha=.15);ax.set_title(modalities[j])
            ax.set_ylabel('模型坐标' if i==0 else '原训练 SD 单位（未减全窗均值）')
            ax.set_xlabel('保留样例 A–G；未注入眼电')
    fig.suptitle('归一化后的实测分布：箱体 P25–P75，中线中位数，须 P5–P95；同一行共用纵轴',fontsize=12)
    fig.tight_layout();fig.savefig(export/'observed_distributions.png',dpi=250);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for ax,stage,title in zip(axs,('model','training_sd_units','objective_units'),
            ('实际模型输入的标准差','标准差 / 原训练 SD','标准差 / 原训练 SD × 权重平方根')):
        values=np.array([r['statistics']['distributions'][stage]['sd'] for r in primary])
        for j in range(3):ax.bar(np.arange(7)+(j-1)*.25,values[:,j],width=.24,label=modalities[j],color=colors[j])
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.legend(ncol=3)
        if stage=='training_sd_units':ax.axhline(1,color='grey',ls=':')
    axs[-1].set_xticks(np.arange(7),labels);axs[0].set_ylabel('模型坐标');axs[1].set_ylabel('无量纲');axs[2].set_ylabel('无量纲')
    fig.tight_layout();fig.savefig(export/'scale_comparison.png',dpi=250);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for ax,region in zip(axs,('prefrontal','occipital')):
        selected=[r for r in records if r['region']==region and r['variant']=='baseline']
        for j,modality in enumerate(modalities):
            vals=[r['statistics']['distributions']['model']['sd'][j] for r in selected]
            ax.plot(range(len(vals)),vals,'o-',color=colors[j],label=modality)
        ax.axvline(5.5,color='grey',ls=':');ax.set_xticks(range(len(selected)),[r['subject'] for r in selected],rotation=60,ha='right')
        ax.set_title(('前额' if region=='prefrontal' else '枕区')+'：左六名训练被试，右侧验证样例')
        ax.set_ylabel('归一化后窗口内 SD / 模型坐标');ax.legend();ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(export/'training_and_validation_sd.png',dpi=250);plt.close(fig)
    for source in (Path(__file__),Path('src/data/ssm_prepared.py'),Path('experiments/scripts/run_dual_component_tokenizer.py')):
        destination=export/'source'/source.name;destination.parent.mkdir(exist_ok=True);shutil.copyfile(source,destination)
    return dict(status='completed',export=str(export),counts=result['counts'],replay_checks=len(checks),new_SSM_fits=0)


def render_voltage_power_cap_method_report(front,posterior,export,*,workers=2):
    """Explain retained development targets; replay voltage only, never refit SSM."""
    import concurrent.futures
    import multiprocessing
    import shutil
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties,fontManager
    from experiments.scripts.render_source_diagnostics_report import render_report
    front,posterior,export=map(lambda p:Path(p).resolve(),(front,posterior,export))
    if export.exists():raise FileExistsError('method export must be fresh')
    if shutil.which('pdflatex') is None:
        raise RuntimeError('include the installed pdflatex directory in PATH before exporting')
    def read(p):return json.loads(Path(p).read_text())
    def arrays(p):
        with np.load(p,allow_pickle=False) as z:return dict(z)
    summaries=[read(p/'summary.json') for p in (front,posterior)]
    for root,summary,schema in zip((front,posterior),summaries,
            ('voltage_power_cap_examples_results_v1','voltage_power_cap_posterior_results_v1')):
        if summary['schema']!=schema or summary['status']!='completed' or read(root/'verification.json')['status']!='passed':
            raise ValueError('accepted retained examples required')
        for row in summary['rows']:
            case=row['case']
            if case['split']!='validation' or case['dataset']!='simultaneous_eeg_nirs' or row['status']!='completed':
                raise ValueError('only named completed development examples are supported')
    chosen={label:(root,dict(next(r for r in s['rows'] if r['subject']==subject)))
        for label,root,s,subject in [('B',front,summaries[0],'VP005'),('G',posterior,summaries[1],'VP005'),
                                    ('F',posterior,summaries[1],'VP002')]}
    for _,row in chosen.values():
        row.setdefault('display_channel','Fp1')
        row.setdefault('channel_count',16)
        plan=read(Path(row['run'])/'plan.json')
        if plan.get('test_access')!='forbidden' or plan.get('protected_data')!='forbidden':
            raise ValueError('retained development boundary not established')
    export.mkdir(parents=True);(export/'intermediates').mkdir();(export/'figures').mkdir()
    jobs=[(row['run'],variant,row['display_channel'],str(export/'intermediates'/f'{label}_{variant}.npz'),str(root/'source'))
        for label,(root,row) in chosen.items() if label!='F' for variant in ('baseline','ocular')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(max(1,workers),2),
            mp_context=multiprocessing.get_context('spawn')) as pool:
        replay=list(pool.map(_voltage_method_replay_case,jobs))
    (export/'replay_verification.json').write_text(json.dumps(replay,ensure_ascii=False,indent=2)+'\n')
    cases={}
    for label,(root,row) in chosen.items():
        run=Path(row['run']);inputs=arrays(run/'inputs.npz');co=read(run/'coordinates.json')
        fits=read(run/'physical_fits.json');roi=row.get('metric_roi',1)
        physical={v:arrays(next(f['array_path'] for f in fits if f['variant']==v and f['roi']==roi))
            for v in ('baseline','ocular')}
        cases[label]=dict(row=row,inputs=inputs,model=arrays(run/'reference.npz'),physical=physical,
            coordinate=co[row['case']['rows'][roi]['coordinate']],roi=roi,
            traces={v:arrays(export/'intermediates'/f'{label}_{v}.npz') for v in ('baseline','ocular')} if label!='F' else {})
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fontManager.addfont(font.get_file());plt.rcParams.update({'font.family':font.get_name(),'font.size':10,'axes.unicode_minus':False})
    blue,red,green,grey='#2469a0','#c24e40','#24826c','#68727e'
    bands=['1–4 Hz','4–8 Hz','8–13 Hz','13–30 Hz','30–45 Hz','互补载波']
    figures={};provenance={};statistics={}
    def save(name,fig,source):
        fig.tight_layout();path=export/'figures'/f'{name}.png';fig.savefig(path,dpi=250,facecolor='white');plt.close(fig)
        figures[name]=path;provenance[name]=source
    def style(ax,xlabel='窗口内时间 / s',ylabel=None):
        ax.grid(alpha=.18);ax.set_xlabel(xlabel)
        if ylabel:ax.set_ylabel(ylabel)
    def pair(ax,t,y0,y1):
        ax.plot(t,y0,color=blue,lw=1.1,label='注入前');ax.plot(t,y1,color=red,lw=1.05,ls='--',label='注入后')
    # Shapes are deliberately visible; this schematic does not claim source localization.
    fig,axs=plt.subplots(3,1,figsize=(10,5.4))
    texts=[('原始 EEG：18000 × C，200 Hz','5 带功率 → log → 当前前 5 秒参考','3 条 ROI 代理：900 × 3，10 Hz'),
           ('每个 ROI 的 EEG代理 + HbO/HbR','独立 Tak 状态空间条件拟合','3 条重建代理 → 训练统计补全：901 × 5C'),
           ('原始 EEG → 6 载波 → 1 秒电压块','参考白化 → 奇异值截断 → 重叠合成','S：18000 × C；O = 原始电压 - S')]
    for ax,row in zip(axs,texts):
        ax.axis('off')
        for k,text in enumerate(row):
            ax.text(.17+.33*k,.5,text,ha='center',va='center',fontsize=10,
                bbox=dict(boxstyle='round,pad=.7',facecolor='#edf3f7',edgecolor=blue))
            if k<2:ax.annotate('',xy=(.34+.33*k,.5),xytext=(.30+.33*k,.5),arrowprops=dict(arrowstyle='->',color=grey))
    save('01_flow',fig,'Data owner functions voltage_power_cap_*; C=16 frontal or17 posterior; training statistics frozen.')
    b=cases['B'];bt=b['traces'];idx=int(bt['baseline']['display_index']);t=bt['baseline']['time_native'];tp=bt['baseline']['time_proxy'];tf=bt['baseline']['time_frame']
    zoom=(t>=40.195)&(t<=43.195)
    fig,axs=plt.subplots(7,1,figsize=(9.5,9),sharex=True)
    for j,ax in enumerate(axs):
        key='raw_voltage' if j==0 else 'carrier_voltage'
        pair(ax,t[zoom],bt['baseline'][key][zoom] if j==0 else bt['baseline'][key][zoom,j-1],
             bt['ocular'][key][zoom] if j==0 else bt['ocular'][key][zoom,j-1])
        ax.set_ylabel(('原始' if j==0 else bands[j-1])+'\nµV');ax.grid(alpha=.18)
    axs[0].legend(ncol=2);axs[-1].set_xlabel('窗口内时间 / s；各行电压刻度独立')
    save('02_carriers',fig,'B traces raw_voltage,carrier_voltage; fixed known-injection peak ±1.5s.')
    fig,axs=plt.subplots(3,2,figsize=(9.5,7.6),sharex=True)
    for col,band in enumerate((0,2)):
        for row,key in enumerate(('power100ms','log_features','log_features')):
            for v,color,ls in [('baseline',blue,'-'),('ocular',red,'--')]:
                y=bt[v][key][:,idx,band]
                if row==1:y=y+bt[v]['current_baseline_log_power'][idx,band]
                axs[row,col].plot(tp,y,color=color,ls=ls,label='注入前' if v=='baseline' else '注入后')
            style(axs[row,col],ylabel=['100 ms 二阶矩 / µV²','自然对数功率','前 5 秒参考后 log-power'][row])
        axs[0,col].set_title('B · Fp1 · '+bands[band]);axs[0,col].legend()
    save('03_power_log',fig,'B traces power100ms, current_baseline_log_power,log_features; no per-plot normalization.')
    fig,axs=plt.subplots(1,2,figsize=(9.5,4.2))
    for ax,label in zip(axs,('B','G')):
        c=cases[label];anchor=c['row']['case']['rows'][c['roi']]['anchor'];pc=np.array(c['coordinate']['pc']).reshape(6,5)
        im=ax.imshow(pc,cmap='coolwarm',vmin=-.4,vmax=.4,aspect='auto');ax.set_yticks(range(6),anchor['eeg_channels'])
        ax.set_xticks(range(5),bands[:5],rotation=40,ha='right');ax.set_title(label+' · '+anchor['hb_channel']+' 邻域 PCA 载荷')
        fig.colorbar(im,ax=ax,label='单位范数载荷，无量纲')
    save('04_pca',fig,'B/G coordinates.json pc (6 channels ×5 bands), frozen six training windows; shared color limits.')
    for label,c in cases.items():
        physical=c['physical'];factor=c['coordinate']['hb_factor'];sd=np.array(c['coordinate']['sd'])
        fig,axs=plt.subplots(3,2,figsize=(9.5,7.5),sharex=True,sharey='row')
        for col,v in enumerate(('baseline','ocular')):
            a=physical[v]
            for j in range(3):
                scale=1. if j==0 else factor
                axs[j,col].plot(a['time_s'][:899],a['u'][:899,j]/scale,color=grey,lw=1,label='观测')
                axs[j,col].plot(a['time_s'][:899],a['physical'][:899,j]/scale,color=blue if col==0 else red,lw=1.1,label='SSM 完整预测')
                err=np.sqrt(np.mean(((a['physical'][:899,j]-a['u'][:899,j])/sd[j])**2))
                axs[j,col].set_title(('注入前' if col==0 else '注入后')+f' · NRMSE={err:.4f}')
                style(axs[j,col],ylabel=['EEG 代理 / 模型尺度','HbO / µM','HbR / µM'][j]);axs[j,col].legend(fontsize=9)
        save(label+'_ssm',fig,f"{c['row']['run']}/physical_fits.json: selected ROI {c['roi']}; physical/u, first899 valid points; Hb divided by frozen hb_factor.")
    fig,axs=plt.subplots(3,1,figsize=(9.5,6.6),sharex=True)
    for j,ax in enumerate(axs):
        pair(ax,tp,bt['baseline']['observed_proxy'][:,j],bt['ocular']['observed_proxy'][:,j])
        ax.plot(tf,bt['baseline']['aligned_proxy'][:,j],color=green,label='SSM 注入前')
        ax.plot(tf,bt['ocular']['aligned_proxy'][:,j],color='#8e62ad',ls=':',label='SSM 注入后')
        style(ax,ylabel=['左运动区','前额','右运动区'][j]+'\n模型尺度')
    axs[0].legend(ncol=4,fontsize=9)
    save('05_three_proxies',fig,'B replay observed_proxy vs aligned_proxy; full90s offline; independent regional fits.')
    fig,axs=plt.subplots(5,1,figsize=(9.5,7.4),sharex=True)
    for ax,j,name in zip(axs,[0,1,2,3,5],['驱动 r','血管信号 a','相对流量 f','相对体积 v','相对脱氧量 q']):
        pair(ax,tp,b['physical']['baseline']['states'][:,j],b['physical']['ocular']['states'][:,j]);style(ax,ylabel=name)
        if j>=2:ax.axhline(1,color=grey,ls=':',lw=.7)
    axs[0].legend(ncol=2)
    save('06_states',fig,'B physical states[:,0,1,2,3,5]; stored pressure column4 equals volume, not a sixth free state.')
    fig,axs=plt.subplots(1,3,figsize=(10,6))
    model=b['model'];lift=model['lift']*.025;lim=np.max(abs(lift))
    for j,ax in enumerate(axs):
        im=ax.imshow(lift[:,j].reshape(-1,5),cmap='coolwarm',vmin=-lim,vmax=lim,aspect='auto')
        ax.set_yticks(range(len(b['inputs']['channels'])),b['inputs']['channels']);ax.set_xticks(range(5),bands[:5],rotation=65,ha='right')
        ax.set_title(['左运动区 +1 SD','前额 +1 SD','右运动区 +1 SD'][j]);fig.colorbar(im,ax=ax,label='补全 log-power 变化')
    save('07_lift',fig,'B reference.npz lift × original EEG proxy trainingSD0.025; common color limits; no spatial interpolation.')
    fig,axs=plt.subplots(5,1,figsize=(9.5,7.5),sharex=True)
    for j,ax in enumerate(axs):
        ax.plot(tp,bt['ocular']['log_features'][:,idx,j],color=grey,label='注入后 EEG 实测特征')
        pair(ax,tf,bt['baseline']['completed_log_power'][:,idx*5+j],bt['ocular']['completed_log_power'][:,idx*5+j])
        style(ax,ylabel=bands[j]+'\nlog-power')
    axs[0].legend(ncol=3,fontsize=9)
    save('08_completed_log',fig,'B Fp1 five features; completed_log_power reference versus observed log_features; reference not a fitted per-electrode waveform.')
    for label in ('B','G'):
        c=cases[label];tr=c['traces'];di=int(tr['baseline']['display_index'])
        fig,axs=plt.subplots(3,2,figsize=(9.5,8),sharex=True)
        for j,ax in enumerate(axs.flat):
            for v,color,ls in [('baseline',blue,'-'),('ocular',red,'--')]:
                ax.plot(tf,np.sqrt(tr[v]['reference_power'][:,j,di]),color=color,ls=ls,label='参考 RMS '+('前' if v=='baseline' else '后'))
            ax.plot(tf,tr['ocular']['input_frame_RMS'][:,j],color=grey,lw=.9,label='注入后输入块 RMS')
            ax.plot(tf,tr['ocular']['projected_frame_RMS'][:,j],color=green,lw=.9,label='投影后块 RMS')
            ax.set_title(label+' · '+c['row']['display_channel']+' · '+bands[j]);style(ax,ylabel='µV（各图独立刻度）')
        axs[0,0].legend(fontsize=8)
        save(label+'_reference_rms',fig,f'{label} traces reference_power, input_frame_RMS, projected_frame_RMS; sqrt(diagΓ) is uncentered RMS scale, not SD.')
    fig,axs=plt.subplots(1,2,figsize=(10,5))
    for ax,mat,title in [(axs[0],model['correlation'][0],'冻结 1–4 Hz 二阶矩相关'),
                          (axs[1],bt['ocular']['raw_space_matrix'],'41.7 秒块的电压线性作用矩阵')]:
        lim=max(1.,float(np.max(abs(mat))));im=ax.imshow(mat,cmap='coolwarm',vmin=-lim,vmax=lim)
        ax.set_xticks(range(16),b['inputs']['channels'],rotation=90,fontsize=7);ax.set_yticks(range(16),b['inputs']['channels'],fontsize=7)
        ax.set_title(title);fig.colorbar(im,ax=ax,label='无量纲；各图独立色标')
    save('09_matrices',fig,'B correlation[0] and ocular raw_space_matrix; rows output channel, columns input channel; not a fixed cross-window denoising filter.')
    fig,axs=plt.subplots(2,2,figsize=(9.5,6.7));z=bt['ocular'];nmode=len(z['svd_input'])
    axs[0,0].plot(np.arange(nmode)+1,z['svd_input'],'o-',color=grey,label='白化输入');axs[0,0].plot(np.arange(nmode)+1,z['svd_output'],'x--',color=green,label='截断后')
    axs[0,0].axhline(np.sqrt(200),color=red,ls=':',label='上限 sqrt(200)');axs[0,0].set_yscale('log');axs[0,0].legend(fontsize=8);style(axs[0,0],'奇异方向编号','奇异值 / 无量纲')
    axs[0,1].bar(np.arange(nmode)+1,z['svd_gain'],color=green);style(axs[0,1],'奇异方向编号','乘法增益');axs[0,1].set_ylim(0,1.1)
    for ax,ch in zip(axs[1],[idx,list(b['inputs']['channels']).index('C3')]):
        ax.plot(z['frame_time'],z['frame_input'][ch],color=grey,label='局部输入');ax.plot(z['frame_time'],z['frame_projected'][ch],color=green,label='局部输出')
        ax.set_title(str(b['inputs']['channels'][ch])+' · 1–4 Hz');style(ax,ylabel='µV');ax.legend()
    save('10_svd',fig,'B ocular frame nearest known-injection absolute peak (417), band0; no best-output selection. Local block before overlap-add.')
    fig,axs=plt.subplots(3,1,figsize=(9.5,6.5),sharex=True)
    for j,ax in enumerate(axs):
        for v,color,ls in [('baseline',blue,'-'),('ocular',red,'--')]:
            key=['minimum_gain','clipped_directions','local_input_ratio'][j];y=bt[v][key][:,0]
            if j==1:y=y/16
            ax.plot(tf,y,color=color,ls=ls,label='注入前' if v=='baseline' else '注入后')
        style(ax,ylabel=['最小奇异增益','被截断方向比例','输入最大广义二阶矩比'][j])
    axs[2].set_yscale('log');axs[0].legend(ncol=2)
    save('11_gains',fig,'B lowband local diagnostics; perframe independent SVD, direction indices need not track identical modes across time.')
    fig,axs=plt.subplots(6,1,figsize=(9.5,8),sharex=True)
    for j,ax in enumerate(axs):
        pair(ax,t[zoom],bt['baseline']['band_contributions'][zoom,j],bt['ocular']['band_contributions'][zoom,j]);style(ax,ylabel=bands[j]+'\nµV')
    axs[0].legend(ncol=2);axs[-1].set_xlabel('窗口内时间 / s；各行独立刻度；六行相加得到 S')
    save('12_ola',fig,'B independently accumulated perband overlap-add contributions; verified vs producer sum-before-OLA order.')
    for label in ('B','G'):
        c=cases[label];tr=c['traces'];roi=c['roi'];fig,axs=plt.subplots(2,1,figsize=(9.5,5.5),sharex=True)
        for ax,v in zip(axs,('baseline','ocular')):
            ax.plot(tp,tr[v]['final_proxy'][:,roi],color=green,label='最终 S 再提取代理')
            ax.plot(tf,tr[v]['aligned_proxy'][:,roi],color=blue if v=='baseline' else red,ls='--',label='SSM 重建代理')
            ax.set_title(label+' · '+('注入前' if v=='baseline' else '注入后'));style(ax,ylabel='模型尺度');ax.legend()
        save(label+'_proxy_consistency',fig,f'{label} final_proxy vs aligned_proxy; final S log powers re-referenced to its own first5s; equality is not imposed.')
    fig,axs=plt.subplots(2,2,figsize=(9.5,6.5),sharex=True,sharey='row')
    for col,label in enumerate(('B','G')):
        tr=cases[label]['traces']
        for row,key in enumerate(('local_output_ratio','final_reanalysis_ratio')):
            for v,color,ls in [('baseline',blue,'-'),('ocular',red,'--')]:
                y=tr[v][key].max(1);axs[row,col].plot(tf,y,color=color,ls=ls,label='注入前' if v=='baseline' else '注入后')
            axs[row,col].axhline(1,color=grey,ls=':',label='约束上限 1');axs[row,col].set_yscale('log')
            axs[row,col].set_ylim((.5,2) if row==0 else (.5,200))
            axs[row,col].set_title(label+' · '+['局部投影后','最终 S 重新滤波后'][row]);style(axs[row,col],ylabel='六载波最大比值 / 对数轴');axs[row,col].legend(fontsize=8)
        statistics[label]={v:dict(max_local=float(tr[v]['local_output_ratio'].max()),max_final=float(tr[v]['final_reanalysis_ratio'].max()),
            proxy_RMSE=float(np.sqrt(np.mean((tr[v]['final_proxy'][:899, cases[label]['roi']]-tr[v]['aligned_proxy'][:899,cases[label]['roi']])**2)))) for v in tr}
    save('13_final_bound',fig,'Perframe maximum over6bands of max eigenvalue of whitened uncentered second moment / cap. Not voltage amplification or confidence interval.')
    for label,filename in [('B','B_VP005_voltage.png'),('G','G_VP005_O2_voltage.png'),('F','F_VP002_O1_voltage.png')]:
        src=chosen[label][0]/'export_v3'/filename;dest=export/'figures'/filename;shutil.copyfile(src,dest)
        figures[label+'_final']=dest;provenance[label+'_final']=str(src)
    content=_voltage_method_report_content(cases,summaries,figures,statistics,replay,front,posterior)
    path=export/'report_content.json';path.write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n')
    (export/'figure_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    (export/'derived_statistics.json').write_text(json.dumps(statistics,indent=2)+'\n')
    checks=render_report(path,export/'report')
    result=dict(status='completed',export=str(export),figures=len(figures),replays=len(replay),checks=checks)
    (export/'completion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return dict(status='completed',export=str(export),figures=len(figures),replays=len(replay))


def _voltage_method_report_content(cases,summaries,figures,statistics,replay,front,posterior):
    """Visible Chinese explanation, tied to retained arrays and actual operators."""
    def p(text):return dict(type='paragraph',text=text)
    def eq(latex,explanation):return dict(type='equation',latex=latex,explanation=explanation)
    def fig(key,caption):return dict(type='figure',path=str(figures[key]),caption=caption,preserve_bitmap=True)
    def table(columns,rows,widths=None):return dict(type='table',columns=columns,rows=rows,widths=widths or [1]*len(columns))
    sections=[]
    def section(title,*blocks):sections.append(dict(title=title,page_break=True,blocks=list(blocks)))
    b=cases['B'];z=b['traces']['ocular'];co=b['coordinate']
    section('1. 监督对象：由原始电压和条件代理共同定义',
        p('目标是给原始 EEG 构造同采样率、同通道的两路连续监督信号。S 是满足局部参考二阶矩上限的电压分量，O 是剩余电压。这里的 S/O 是工程分解标签，尚不能命名为纯神经信号与纯伪迹。'),
        eq(r'S=\mathcal F(x,\widehat{\boldsymbol p};\Theta_{\rm train}),\qquad O=x-S',
           'x 为当前窗口原始 EEG 电压，单位微伏；重建代理为三个 ROI 的低维时间轨迹；训练统计包括 PCA、尺度、log-power 协方差及通道二阶矩相关。'),
        p('仅有 SSM 输出不能唯一还原 200 Hz EEG。平方功率丢失符号和相位：一段电压及其整体反号具有相同功率代理；三个数也不能唯一指定 80 或 85 个通道×频带特征。当前设计用原始 EEG 提供快速波形，用 SSM 输出形成幅度约束，再按训练统计选择一种条件分解。投影仍会改变幅度、混合通道并可能改变局部波形，不能声称严格保持相位。'),
        fig('01_flow','三个计算层次：观测压缩、条件模型、原始电压约束。C 为面板中不重复的 EEG 电极数；前额面板 C=16，枕区面板 C=17。'))
    anchors=[]
    for label in ('B','G'):
        c=cases[label]
        for r in c['row']['case']['rows']:
            a=r['anchor'];anchors.append([label,a['region'],a['hb_channel'],', '.join(a['eeg_channels'])])
    section('2. fNIRS 通道是空间锚点，不是电压源定位',
        p('每个感兴趣区域（ROI）选定一个 Hb 通道，并在归一化头部坐标中选其最近的六个 EEG 电极。六通道×五频带经一个训练 PCA 方向压成一条 EEG 代理，与该 Hb 通道的 HbO/HbR 联合拟合。由此得到的是“以 Hb 通道为锚点的局部 EEG 功率统计”，不是该点的皮层电势，也不是 EEG 逆源定位结果。'),
        table(['面板','ROI','Hb 锚点','六个 EEG 电极'],anchors,[.5,1.2,.7,2.8]),
        p('前额面板有三个 ROI、共 16 个不重复电极；枕区面板保留左运动区与前额，替换右运动区为枕区，共 17 个电极。共享电极在联合电压向量中只出现一次，三个 PCA 投影行可以重叠。没有导联场、皮层源网格或体积传导逆算，因而不能把统计补全解释为从 fNIRS 点向头皮插值电压。'))
    section('3. 数据支持、冻结对象与样例选择',
        table(['阶段','可见数据与用途','本版本处理'],[
            ['训练统计','VP001/003/006/007/008/009 各一段 90 秒','PCA 用 5400 个 100 ms 点；参考统计用各窗 [15,75) 秒，共 3600 点'],
            ['当前条件拟合','验证窗口全部 90 秒 EEG 代理与 Hb','每 ROI 重估 900 点驱动及四初态；Tak 参数、PCA、统计参考冻结'],
            ['电压目标生成','同窗原始 EEG 与三个重建代理','901 个 1 秒块；100 ms 步长；每块六载波联合通道投影'],
            ['眼电评分','注入前后各一套条件解','只在 [30,50) 秒计算已知变化误差；4000 电压点或 200 代理点'],
            ['完整模型评分','模型输出与观测','每模态 899 个有效点；除以原训练 SD，不使用加权优化尺度']],[1,2,2.8]),
        p('主线 B 为 VP005/Fp1，扩展 G 为同一 VP005 窗口的 O2；两者并非独立被试。B/G 用于解释较清楚的计算链，F=VP002/O1 展示高模型误差。全部七组、五名被试都在汇总表中保留。原 A 为 VP002 的事件 26，其余新例使用既定事件 0；各被试原候选分母为 23 窗，本报告没有按中间量重新挑窗。RMS 指均方根，SD 指标准差，NRMSE 指归一化均方根误差；二阶矩开方的 RMS 不等于去均值后的 SD。'),
        p('EEG 原始参考为 TP9、采样率 200 Hz、单位微伏。Hb 来自保留的统一 10 Hz 缓存，单位微摩尔，已按原合同完成转换与 0.01–0.2 Hz 处理，不再次乘 1000 或新增滤波。EEG 与 Hb 使用各自记录时间起点对齐事件；B/G 的 EEG 起点为 0 秒、Hb 为 14.352 秒，报告横轴均为对齐后的窗口时间。'),
        p('这是一项完整窗口可见的离线条件重建。零相位 EEG 滤波、离线 Hb 处理和整窗求解均使用未来信息。七组不是七个独立被试，数千时间点或重叠块也不是独立样本；本报告不做显著性检验或总体置信区间。'))
    section('4. 从原始电压形成五频带相对功率特征',
        eq(r'\begin{aligned}z_b&=\mathcal B_bx\quad(b=1,\ldots,5),&z_6&=x-\sum_{b=1}^{5}z_b,\\P_{kcb}&=\frac1{20}\sum_{n=20k}^{20k+19}z_b[n,c]^2,&\ell_{kcb}&=\log\max(P_{kcb},10^{-20})-\frac1{50}\sum_{j=0}^{49}\log\max(P_{jcb},10^{-20}).\end{aligned}',
           '五个四阶 Butterworth 双向带通为 1–4、4–8、8–13、13–30、30–45 Hz；c 是电极，b 是频带，k 是 100 ms 时间箱。P 是不减均值的二阶矩，单位平方微伏；log-power 差无量纲。'),
        p('先滤波，再逐 20 点平方平均，最后取自然对数。当前窗口前 5 秒的 50 个 log-power 点只定义相对变化零点，并不是训练均值。第六载波是精确电压互补项，包含直流、频带外成分和带通重叠补偿；它不是单独的高频噪声带。五个 Butterworth 载波也不是正交能量分解，不能将六项平方和当作原始总功率。'),
        p('100 ms 功率箱时间戳取左边界 0、0.1、…、89.9 秒。之后的 1 秒电压块使用中心时间 0、0.1、…、90 秒。两个统计的时间支持不同，900 点功率与 901 个局部块不可直接视作同一算子。'))
    section('5. 实际电压：同一眼电如何进入不同载波',
        fig('02_carriers','B：VP005/Fp1。蓝实线为原始自然背景，红虚线为加入同一眼电后；首行为原始电压，其余六行为滤波载波。放大区间固定为已知注入绝对峰附近 ±1.5 秒；各行刻度独立。'),
        p('低频载波呈现慢峰及其滤波延伸，快速载波仍包含背景起伏；互补项并非零。滤波会把有限时段注入影响延伸到邻近时间。图示用于解释信号路径，眼电评分仍使用完整 [30,50) 秒，不能只按峰附近的视觉差异判断性能。'))
    section('6. 中间量：功率、对数与参考零点',
        fig('03_power_log','B/Fp1 的 1–4 Hz 与 8–13 Hz。每列从上到下为 100 ms 二阶矩、自然对数功率、当前前 5 秒参考后的特征。蓝实线/红虚线为注入前/后；每幅纵轴独立。'),
        p('平方平均消除了电压正负号；对数压缩幅度跨度；减去首段均值使每个特征表达相对变化。三步作用不同。代理约束依赖这种相对功率特征，而不是 200 Hz 电压的瞬时极性。参考零点和后面恢复绝对功率所用的训练基线也不同。'))
    section('7. PCA 将六电极、五频带压成一个代理',
        eq(r'p^{\rm obs}_{kj}=g_{Ej}\,\boldsymbol v_j^\top\boldsymbol\ell_{k,\mathcal N_j},\qquad g_{Ej}=\frac{0.025}{\sqrt{\lambda_{1j}}},\qquad \boldsymbol p^{\rm obs}_k=A\boldsymbol\ell_k',
           'j 为 ROI，邻域特征维数为 30；v 为单位范数第一主成分，最大绝对载荷取正号；特征值来自六个训练窗口的中心化协方差。PCA 训练时中心化，但代理投影不再减训练均值；输入已按当前窗口前 5 秒参考。A 的形状为 3×80 或 3×85。'),
        p(f'B 前额的 EEG 系数为 {co["eeg_factor"]:.9f}，Hb 共用系数为 {co["hb_factor"]:.9f}；EEG 原训练 SD 固定为 0.025。HbO/HbR 共用缩放系数按两者训练方差均值确定，但各自原训练 SD 仍不同。模型尺度是数据集/位置冻结坐标，不是绝对生理标定。'),
        fig('04_pca','前额 B 与枕区 G 的训练载荷。行是六个 EEG 电极，列是五个频带，颜色为有符号 PCA 权重；两图共用 -0.4 至 0.4 色标。权重是统计方向，不是解剖贡献或脑源强度。'))
    section('8. SSM：每个 ROI 用一个驱动解释三条观测',
        eq(r'\begin{aligned}\dot a&=r-0.64a-0.32(f-1),&\dot f&=a,\\\dot v&=\frac{f-v^{1/0.32}}{2+2},&F_{\rm out}&=v^{1/0.32}+2\dot v,\\\dot q&=\frac{fE(f)/0.32-F_{\rm out}q/v}{2},&E(f)&=1-(1-0.32)^{1/f},\\h_O&=(v-1)-0.35(q-1),&h_R&=0.35(q-1).\end{aligned}',
           'r 为有效驱动，a 为血管信号；f、v、q 分别为相对流量、体积和脱氧量，静息参考为 1。固定 kappa=0.64、gamma=0.32、alpha=0.32、tau=2 秒、黏弹时间=2 秒、E0=0.32、P0=1、Q0=0.35、神经血管增益=1。四初态为 a/f/v/q；存储压力列与 v 绑定。'),
        p('驱动 r 经过 EEG 参考观测算子后给出重建代理，即减去其前 50 点均值；Hb 状态先生成模型 HbO/HbR，再经原生时间插值、100 ms 聚合及首段参考算子与观测比较。图中的 Hb 曲线除以冻结 Hb 系数后回到缓存单位。三个 ROI 分别拟合，不存在本版本中一个共享的全脑驱动。'),
        p('这些输出是给定参数、观测、正则化和完整时间窗后的条件估计。它们不是隐藏生理状态真值。该固定参数实现借鉴 Tak 血流动力学，但本项目的代理构造与电压功率约束是额外工程设计，不能由生理模型文献直接担保正确性。'))
    section('9. Hb 优先权重改变目标函数，不改变报告尺度',
        eq(r'\begin{aligned}J={}&0.4\!\sum_{k,m\ {\rm valid}}w_m\left(\frac{\widehat u_{km}-u_{km}}{s_m}\right)^2+0.01\left\|\frac{D_2r}{(0.1)^{3/2}}\right\|_2^2\\&+100\{a_0^2+(f_0-1)^2+(v_0-1)^2+(q_0-1)^2\}+\frac{0.1}{(\log2)^2}\sum_k(\log f_k)^2,\\&(w_E,w_O,w_R)=(0.1,10,10).\end{aligned}',
           'u/预测u 为模型尺度的 EEG代理、HbO、HbR；s 为各自原训练 SD，D2 为不除步长的二阶差分。0.4 是 dt=0.1 秒与原观测参考步长 0.25 秒之比。先对每模态标准化平方误差加权，再跨有效时间点求和。'),
        p('等价实现是仅在优化器中使用 s 除以权重平方根。积分误差检查和所有报告 NRMSE 仍使用原 s，不会因为将 EEG 权重降低而人为缩小 EEG 报告误差。逐 ROI 可见三条完整观测的前 899 点，最后一点按原掩码排除。'),
        p('当前求解自由度是 900 个驱动采样和四个血管初态；没有重新估计 PCA、功率协方差或 Tak 参数。RK4 每 0.1 秒四子步，八子步复核；接受条件包括投影缩放梯度不超过 1e-5、积分差不超过 0.005 原训练 SD。数值接受不等于观测准确，更不等于生理状态可识别。'))
    section('10. 前额实测：Hb 拟合与 EEG 代理各自检查',
        fig('B_ssm','B/VP005 前额 ROI：左列注入前、右列注入后。灰线为实际观测，彩线为 SSM 完整预测；各行左右共用纵轴。标题 NRMSE 使用完整 899 点与原训练 SD，向下为好。'),
        p('加入眼电只改变 EEG，Hb 数据逐点不变，但联合拟合的驱动和初态可以随之改变。Hb 权重较大使解更偏向 Hb 重建；这是一种折中，不能推断偏离模型的 EEG 就都是伪迹。图中 EEG 代理的快变与模型轨迹的差异，正是随后功率参考可能压低真实变化的原因之一；这一机制解释仍需结合保留的神经注入对照。'))
    section('11. 三条重建代理共同约束整个电极面板',
        fig('05_three_proxies','B 的三个 ROI。蓝实线/红虚线为实际 EEG 代理注入前/后；绿实线/紫点线为对应 SSM 重建。曲线是 10 Hz 功率统计或其条件估计，不是电压。'),
        p('本次眼电只进入前额邻域；未受影响的左右运动区在输入逐点一致后复用原解。目标生成仍使用三个 ROI 的重建代理，并通过 A 与训练协方差共同决定全 16 通道参考。不能只把 Fp1 所属 ROI 单独逆变换后忽略其他约束。'))
    section('12. 条件状态随时间变化，但不是独立测量',
        fig('06_states','B 前额的 r、a、f、v、q 轨迹。蓝实线/红虚线为注入前/后；f/v/q 的灰点线为参考值 1。各行独立刻度，均为模型内部尺度。'),
        p('r 提供驱动，a/f 传递至血管体积 v 和脱氧量 q，后两者再产生 HbO/HbR。图中轨迹由同一组观测联合估计，不能把它们当作额外独立证据。S/O 构造直接使用经观测算子得到的重建 EEG 代理，而不把所有血管状态拼接成电压。'))
    section('13. 三个代理如何补全 80/85 维 log-power',
        eq(r'\begin{aligned}\widehat{\ell}&=\arg\min_{\ell:\,A\ell=\widehat p}(\ell-\mu)^\top\Sigma^{-1}(\ell-\mu)\\&=\mu+K(\widehat p-A\mu),\qquad K=\Sigma A^\top(A\Sigma A^\top)^{-1},\quad AK=I_3.\end{aligned}',
           'mu 为训练相对 log-power 均值，Sigma 为其收缩协方差；K 为条件补全矩阵，形状 80×3 或 85×3。公式逐时间点应用；代理先线性插值至 901 个块中心，90 秒端点沿用末值。'),
        eq(r'\Sigma=0.9\,\Sigma_{\rm emp}+0.1\,\operatorname{diag}\{\max(\operatorname{diag}\Sigma_{\rm emp},\epsilon)\},\quad\epsilon=\max(10^{-8}\max\operatorname{diag}\Sigma_{\rm emp},10^{-12})',
           '训练均值和总体协方差采用各训练窗 [15,75) 秒相对 log-power，共 3600 点；收缩提高可逆性。这些训练信号未被宣称为干净 EEG。'),
        p('可由拉格朗日乘子直接求得该解：最小化训练 Mahalanobis 距离，并令三个代理投影严格一致。80 维空间剩余 77 维、85 维剩余 82 维由训练统计选择；因此这是一个带先验的补全，不是信息无损逆变换。代理超出训练范围时，补全与指数功率会外推；实现不裁剪这些外推值。'))
    section('14. 补全矩阵展示训练统计如何传递代理变化',
        fig('07_lift','B：K 的三列各乘原 EEG 代理训练 SD=0.025。每图表示仅相应代理增加一个 SD、其余代理固定时，各电极五频带 log-power 的线性变化；行列及共同色标均明确标出。'),
        p('同一代理可影响面板中的多个电极和频带，这是训练协方差关系的结果。正色表示参考 log-power 增加，负色表示减少；既不是能量占比，也不是实际刺激引起的因果效应。该图是固定矩阵的解析作用，不是重新拟合实验。'))
    section('15. 补全波形不要求逐电极贴合实测特征',
        fig('08_completed_log','B/Fp1 五频带：灰线为注入后实际相对 log-power；蓝实线/红虚线为由注入前/后 SSM 代理补全的参考。每行独立纵轴。'),
        p('严格成立的是补全向量经 A 投影等于 SSM 代理，而不是每个电极频带的补全值等于观测。灰线与参考的偏离会传递到局部功率约束，但偏离本身无法辨别眼电、正常神经活动、训练分布变化或 SSM 失配。'))
    section('16. 从相对 log-power 到绝对二阶矩参考',
        eq(r'Q_{kcb}=\exp(\widehat\ell_{kcb}+b^{\rm train}_{cb}),\quad\Gamma_{kb}=D_{kb}R_bD_{kb},\quad D_{kb}=\operatorname{diag}\sqrt{Q_{kb}}',
           'b_train 是六个训练窗口前 5 秒平均 log-power 的逐特征中位数；不是当前窗口的基线。Q 单位平方微伏。第六载波没有 PCA 代理，其 Q 是训练局部二阶矩的逐通道中位数，并随时间固定。'),
        eq(r'R_b=0.9\,\operatorname{mean}_{\rm train}\!\left[\frac{M_{b,cd}}{\sqrt{M_{b,cc}M_{b,dd}}}\right]+0.1I,\qquad M_b=\frac{V_bV_b^\top}{200}',
           'V 是 C×200 的 1 秒载波块；M 不减块均值。相关先按每个训练块标准化再平均，并收缩至单位阵；因此 R 不是通常去均值后的 Pearson 相关。Gamma 是通道间非中心二阶矩参考。'),
        p('使用训练基线意在避免把当前污染抬高的基线直接写回参考；代价是当前幅度若与训练分布不同，也会被限制。每通道根号 Q 是 RMS 尺度而非标准差；约束还包含非对角相关，因此不是逐电极独立的幅度裁剪。'))
    section('17. 前额参考与局部 RMS 的实际关系',
        fig('B_reference_rms','B/Fp1。蓝实线/红虚线为参考 RMS 注入前/后；灰线为注入后 1 秒输入块 RMS；绿线为局部投影后 RMS。六幅纵轴独立；第六载波参考不随代理变化。'),
        p('输入 RMS 超过参考时，对应协方差方向可能被收缩；即使单个通道低于其参考，多通道相关也可能导致某个广义方向越界。绿色是局部块输出，尚未做 Hann 重叠合成，不能直接当作最终 S 的重新提取功率。'))
    section('18. 电压投影：白化、奇异值截断、逆变换',
        eq(r'\begin{aligned}Z^*&=\arg\min_Z\|H^{-1}(Z-V)\|_F^2\quad\mathrm{s.t.}\quad ZZ^\top/L\preceq\lambda\Gamma,\\H&=D\operatorname{chol}(R),\quad W=H^{-1}V=U\operatorname{diag}(\sigma_i)B^\top,\\Z^*&=HU\operatorname{diag}\{\min(\sigma_i,\sqrt{\lambda L})\}B^\top,\quad L=200,\ \lambda=1.\end{aligned}',
           'V/Z 均为 C×200 电压块；H 将模型参考变为单位二阶矩尺度。矩阵半序要求所有空间方向的非中心二阶矩均不超过参考，而不仅是对角线。B 为右奇异向量矩阵，与样例标签 B 无关。'),
        p('白化后约束等价于最大奇异值不超过根号 200；在白化 Frobenius 距离下，逐奇异值截断给出最小改动。小于上限的方向保持，越界方向乘以阈值/奇异值。所有方向已可行时原块逐点保留，包括其均值。增益属于当前奇异方向，不是固定电极增益；不同时间块的方向也未做追踪匹配。'),
        p('这是加权最小改动，不能称为原始电压欧氏空间的最小二乘投影；逆白化可混合通道。局部计算用稳定 SVD，避免先平方形成病态矩阵。此处的 lambda=1 是既定参考上限，未为本报告按样例调优。'))
    section('19. 通道相关和一次实际投影的作用矩阵',
        fig('09_matrices','B 左图为冻结 1–4 Hz 二阶矩相关；右图为注入后 41.7 秒中心块的电压作用矩阵 T，使局部输出= T×输入。行是输出电极，列是输入电极；两幅独立色标。'),
        p('右图非对角项说明一个电极的输出可能包含其他电极的线性组合。T 由当前块的奇异向量和增益决定，不是跨所有窗口固定的滤波器。因此必须同时检查注入电极与全通道误差，不能只看到 Fp1/O2 眼电减小便认定没有空间串扰。'))
    section('20. 眼电峰附近：哪些奇异方向被收缩',
        fig('10_svd','B：中心 41.7 秒、1–4 Hz 的 1 秒块。上左为白化奇异值与阈值，上右为对应增益；下方为 Fp1 和 C3 的局部输入/输出。编号只表示该块的奇异值排序。'),
        p(f'该注入后块的 {len(z["svd_gain"])} 个方向中有 {int(np.sum(z["svd_gain"]<1))} 个被截断，最小增益为 {float(z["svd_gain"].min()):.5g}，最大输入奇异值为 {float(z["svd_input"].max()):.5g}。块中心按已知注入绝对峰的最近 100 ms 节点选定，不依据 S 输出挑选。下方局部波形尚未叠加其他载波或邻块。'))
    section('21. 全窗口收缩强度是随块变化的',
        fig('11_gains','B 的 1–4 Hz 载波：最小奇异增益、被截断方向数/16、输入最大广义二阶矩比。蓝实线/红虚线为注入前/后；底图使用对数轴。'),
        p('较小增益表示某个白化空间方向收缩较强，不等价于该时刻某个电极丢失同样比例的能量。自然背景也可触发收缩，眼电以外的活动并没有自动豁免。图中 901 个高度重叠的块用于描述时间行为，不能作为 901 次独立验证。'))
    section('22. 重叠合成恢复原采样率，残差保证闭合',
        eq(r'S[n,c]=\frac{\sum_{k,b}h[n-s_k]Z^*_{kb}[c,n-s_k]}{\sum_kh[n-s_k]},\qquad O[n,c]=x[n,c]-S[n,c]',
           'h 是长 200 点的周期 Hann 窗，即 np.hanning(201) 去掉末点；步长 20 点。输入两端各反射 100 点，完成后裁回原 18000 点。s_k 是反射扩展序列中的块起点；超出块支持的项为零。'),
        p('投影本身使用未加窗的二阶矩；Hann 只在合成时作用。先把六载波的局部输出求和，再做重叠平均，避免逐块边界跳变。独立按载波累加后相加与生产顺序一致，报告保留数值核验。分母在保留区间始终为正，不逐帧重定标。'),
        p('S/O 与原始 EEG 均为 C 通道、200 Hz、90 秒。S+O=x 是定义保证的代数闭合，不代表两者正交，也不证明一个只含神经、另一个只含噪声。平方能量还存在交叉项，不能把 S/O RMS 比简单相加解释为成分百分比。'))
    section('23. 六载波对最终 S 的可加电压贡献',
        fig('12_ola','B/Fp1：各载波在独立 Hann 重叠合成后的电压贡献；六行相加为 S。蓝实线/红虚线为注入前/后，放大支持与原始载波图相同；各行独立刻度。'),
        p('这些曲线保留正负号，是电压贡献而非功率份额。与输入载波相比，可观察局部投影及邻块平均共同改变了哪些峰谷；不能将合成前后两幅图的独立纵轴高度直接当作增益，增益和误差须使用数组计算。'))
    for label,title in [('B','24. 前额最终 S/O：检查完整分量和配对变化'),('G','28. 枕区最终 S/O：相同注入的另一空间条件')]:
        if label=='G':
            section('26. 枕区 SSM：改变空间面板后的条件拟合',
                fig('G_ssm','G=VP005/O2，对应 POOz Hb 与枕区六电极代理。左右为注入前/后；灰线观测、彩线完整预测；每模态左右共用纵轴。'),
                p('枕区权重仍为 0.1/10/10。只向 O2 加入与 Fp1 相同的眼电；Hb 和其他 EEG 不变。枕区代理及 17 电极参考来自原六名训练被试并冻结。面板、PCA 与参考同时变化，因此与 B 的差异不能只归因于电极位置。'))
            section('27. 枕区中间量：参考和投影后局部 RMS',
                fig('G_reference_rms','G/O2：蓝/红为注入前/后参考 RMS，灰为注入后局部输入 RMS，绿为投影输出 RMS。各图纵轴独立。'),
                p('同样的计算步骤可以作用于新增枕区面板，但并未因此获得原面板外的真实脑源电压。参考是否合理仍依赖训练范围和条件模型误差；局部输出受到的限制也包括原有背景信号。'))
        row=cases[label]['row'];m=row['metrics'][row['display_channel']];allm=row['metrics'][f'all{row["channel_count"]}']
        section(title,fig(label+'_final','前 3 行为原始输入、S、O 的注入前后波形，蓝实线/红虚线；第 4 行仅显示绿实线 ΔO 与灰点线已知注入。左列为完整评分段，右列为固定峰附近放大。'),
            p(f'{label}/{row["display_channel"]} 的 S 变化 RMS 为 {m["S_change_RMS_uV"]:.4f} 微伏，相对于已知注入 RMS {m["injection_RMS_uV"]:.4f} 微伏，局部泄漏误差为 {m["S_error_NRMSE"]:.4f}；全 {row["channel_count"]} 通道误差为 {allm["S_error_NRMSE"]:.4f}。ΔO 形状相关为 {m["delta_O_correlation"]:.4f}，它不是幅度保留率。完整 O 包含原有残差，故末行用配对增量检验新增眼电。'))
        if label=='B':
            section('25. 最终 S 的代理不必等于 SSM 重建代理',
                fig('B_proxy_consistency','B 前额：绿线为最终 S 重新滤波、聚合、按其自身首 5 秒参考后提取的代理；蓝/红虚线为 SSM 重建代理。上下为注入前/后。'),
                p('补全时满足的是线性条件 A×补全 log-power=重建代理。其后指数变换、局部上限投影、载波合成和重新滤波都是额外步骤；上限允许功率低于参考。最终重新提取还使用 S 自己的首段参考，因此不具备代理严格一致性保证。不能把 SSM 代理拟合好直接等同于最终 S 的功率轨迹拟合好。'))
    section('29. 枕区也存在最终代理与模型代理的偏离',
        fig('G_proxy_consistency','G 枕区的同一一致性检查，单位与该区域训练坐标一致。绿线为最终 S 再提取代理；虚线为 SSM 目标代理。'),
        p('该偏离是当前构造流程需要显式报告的性质。若下游监督要求最终 S 重现指定代理，本版本仅有局部上限约束还不充分；本报告没有修改目标定义或反复投影来隐藏偏差。'))
    rows=[]
    for label,row in zip('ABCDEFG',summaries[0]['rows']+summaries[1]['rows']):
        display=row.get('display_channel','Fp1');quality=row.get('model_NRMSE',row.get('frontal_NRMSE'))['ocular']['full899'];m=row['metrics'][display]
        rows.append([label+' '+row['subject']+'/'+display,f'{m["S_change_RMS_uV"]:.2f}',f'{m["S_error_NRMSE"]:.4f}',*[f'{q:.4f}' for q in quality]])
    section('30. 七组完整结果：眼电稳定性与模型误差分开看',
        table(['样例','ΔS RMS /微伏','局部泄漏↓','EEG代理↓','HbO↓','HbR↓'],rows,[1.7,1.1,1,1,1,1]),
        eq(r'e_{\rm eye}=\frac{\|S^{+}-S^0\|_F}{\|n\|_F}=\frac{\|(O^{+}-O^0)-n\|_F}{\|n\|_F},\quad {\rm NRMSE}_m=\frac{\sqrt{\operatorname{mean}_{k\ {\rm valid}}(\widehat u_{km}-u_{km})^2}}{s_m}',
           '眼电分母为同一已知添加 n 的 Frobenius 范数；按所列通道和完整评分段先汇总再取比。三个模型误差用原训练 SD，支持为 899 个点。所有误差向下为好。'),
        p('七组配对全部保留，五名独立被试；新增前额 16 次、枕区 4 次拟合均接受，无失败。原 A 复用既有结果。眼电来自 VP005 另一窗口，B/G 与供体同被试；自然背景没有干净真值。枕区等幅直接注入是工程敏感性检查，不代表自然眼电在枕部的典型幅度。'),
        p('B 与 G 的局部眼电响应较小，适于说明计算路径；F 的 EEG代理和 Hb 误差明显较高，说明数值收敛无法替代模型适配检验。此表没有对重复样例作独立样本检验，也不把展示选择解释为确认性优越结论。'))
    section('31. 反例 F：S 对眼电较稳定，模型仍明显失配',
        fig('F_ssm','F=VP002/O1，枕区 POOz。左右为注入前/后，灰线观测、彩线完整预测；每模态左右同刻度。高 NRMSE 来自完整窗口，不能由局部电压图替代。'),
        p('F 的眼电局部泄漏约 0.2863，但注入后 EEG代理/HbO/HbR NRMSE 为 5.0765/1.2789/1.3909。较小的眼电响应可以与不准确的模型共存；因此不能仅依据眼电被移入 O，证明所保留的 S 具有正确生理含义。'))
    # Read only the previously admitted single-window comparison, not a new campaign.
    original=Path(summaries[0]['rows'][0]['run']);weighted=json.loads((original/'summary.json').read_text())
    equal=json.loads((original.parent/'power_cap_check_v2/summary.json').read_text())
    comparison=[]
    for title,data,method,cap in [('既有空间目标',weighted,'current_target',None),('等权功率目标',equal,'physical_refitted_reference',1.),('Hb优先功率目标',weighted,'physical_refitted_reference',1.)]:
        matches=[next(r for r in data['paired_metrics'] if r['method']==method and r['cap']==cap and r['variant']==v and r['view']=='all16') for v in ('ocular','neural10','neural35')]
        comparison.append([title,*[f'{r["S_error_NRMSE"]:.4f}' for r in matches]])
    section('32. 既有对照揭示眼电抑制与神经保留的代价',
        table(['同一 A 窗口 / 全16通道','眼电泄漏↓','10 Hz 变化保留误差↓','35 Hz 变化保留误差↓'],comparison,[1.7,1,1.25,1.25]),
        p('既有空间目标采用冻结的 12 维空间投影/回投影，并减去一半的双侧邻域预测创新；原始与污染输入经过同一组冻结参数，不重新拟合。等权功率目标与本版使用同一局部二阶矩投影及上限 1，但 SSM 三模态误差权重均为 1；Hb 优先版改为 0.1/10/10，按每个条件重新求驱动与初态。三者在同一 A 窗口、同一注入和评分支持下比较。'),
        eq(r'e_{\rm neural}=\frac{\|(S^+-S^0)-n\|_F}{\|n\|_F},\qquad g_S=\frac{\langle S^+-S^0,n\rangle}{\|n\|_F^2}',
           '眼电希望 S 不变；神经样注入希望 S 的变化等于已知 n。因此两个误差的期望响应不同。gS 是有符号投影增益，不是能量占比。'),
        p('Hb 优先设置把 A 的眼电泄漏从既有空间目标的 0.8375 降至 0.3520，也低于等权功率版本的 0.5047；但 10/35 Hz 已知神经样变化的保留误差约为 0.9782/0.9903，而既有空间目标为 0.4358/0.6206。眼电抑制改善伴随较大的神经样变化损失，不能将新 S/O 认定为已验证的语义分离。'),
        p('这项对照来自原 A 的保留实验，未在本报告追加拟合或扩大数据。10/35 Hz 是受控合成变化，不是实测神经真值；但它提供了比单看眼电图更直接的保留反证。下一步若改进目标，应联合检查干扰移除与已知神经变化保留，而不是继续仅压低眼电响应。'))
    section('33. 局部上限通过，不保证最终合成仍满足上限',
        fig('13_final_bound','每个时间块跨六载波取最大广义二阶矩比；上行为局部投影输出，下行为最终 S 重新滤波后的块。蓝实线/红虚线为注入前/后；点线 1 为上限，纵轴为对数尺度。两列共用刻度：局部行 0.5–2，最终行 0.5–200。'),
        p('局部块各自投影后，重叠平均、载波相加及再次滤波改变了被测算子，特别是不同块参考随时间变化时，约束不自动闭合。此比值衡量相对参考的最不利方向，不是电压被放大多少倍；参考极小时可以产生很大比值。'),
        table(['样例/条件','局部最大比','最终重新分析最大比'],[[label+'/'+v,f'{statistics[label][v]["max_local"]:.6g}',f'{statistics[label][v]["max_final"]:.6g}'] for label in ('B','G') for v in ('baseline','ocular')]))
    section('34. 监督接口、适用范围与结论',
        p('交付的连续目标与输入具有相同原生电极顺序、微伏单位和 200 Hz 时间轴；前额为 18000×16，枕区为 18000×17。S 是条件参考约束下的电压，O 是逐点剩余。原实验保留评分数组、参考功率、局部诊断、最终重新分析及代理；本报告额外保留 B/G 的完整显示通道轨迹和紧凑中间量，不训练新的神经网络或词表。'),
        p('固定对象包括训练 PCA/尺度、补全统计、六载波相关、频带、窗口/步长、Tak 参数与权重；当前窗口重新估计的是驱动与初态，且原始/污染输入分别拟合。学生若仅看 EEG，这种完整 EEG+Hb 条件目标属于额外信息监督，应与学生部署可见输入严格区分。'),
        p('当前证据支持“可复现地产生原电压尺度的两路条件监督”，并在已知眼电注入下较既有空间目标减小 S 的变化。证据不支持“从 fNIRS 位置代理唯一重建原始 EEG”、纯神经/纯噪声解释、实时推断或最终功率一致性。神经样变化损失和高失配样例使其目前更适合作为受限工程候选，而非已经合格的生理语义标签。'),
        p('后续改进应围绕已知失败设计：把神经保留与眼电移除同时列为目标；检查训练基线和代理外推；若需求确实是最终波形满足约束，则在最终分析算子上定义并验证该要求。以上是由本组结果提出的方向，本次只完成保留证据回放与报告，没有新实验结论。'))
    maxerr=max(max(r['retained_array_max_differences'][k] for k in ('s_score','o_score','reference_power','final_proxy')) for r in replay)
    section('35. 回放核验与复现入口',
        p(f'B/G 各自原始与污染两条件共四次确定性回放均通过。与保留 S/O 评分数组、功率参考和最终代理的最大绝对差为 {maxerr:.3g}；各变量单位不同，该数值仅表示逐变量重现误差上界。按载波与先求和两种合成顺序、选定块作用矩阵及独立最终广义特征值检查均在记录容差内。'),
        p('数值 owner 的五个函数与原运行 source 快照按抽象语法树核对一致。没有重新求解 SSM、重新估计训练统计、访问原始数据或测试分区。图的数值来自明确的保留窗口；所有公式保留 LaTeX 源码，图与公式以 PNG 嵌入 PDF，正文和表格可检索。'),
        p('同目录 report_content.json 为可编辑报告内容，figure_provenance.json 逐图说明来源；intermediates/ 包含四组紧凑 NPZ 及核验 JSON，derived_statistics.json 记录本报告计算的诊断量。它们是保留证据的展示导出，不替代项目状态登记或原 summary.json。'))
    return dict(schema='source_diagnostics_report_v1',title='从区域 EEG 功率代理到原始电压 S/O 监督',
        subtitle='Hb 优先固定 Tak 条件模型与多通道二阶矩约束：设计、计算链、实测中间量和失败边界',date='2026-11-06',
        summary=['本报告解释当前脑电（EEG）连续监督目标的实际实现。功能性近红外光谱（fNIRS）提供氧合/脱氧血红蛋白（HbO/HbR）浓度变化。核心路径是：训练主成分分析（PCA）将局部 EEG 相对功率压缩为三个区域代理；联合 EEG/Hb 状态空间模型（SSM）给出条件代理；训练统计补全电极频带参考；原始电压块经白化奇异值上限投影与 Hann 合成得到 S，剩余量定义 O。',
                 'SSM 代理不能独立恢复电压符号、相位和空间自由度。当前方法必须读取原始 EEG；最终 S 也不保证重新提取出的代理或二阶矩严格等于中间参考。报告用 VP005 前额与枕区实测窗口逐步展开，并保留七组完整结果、VP002 高失配反例及原神经注入负结果。'],
        provenance=['证据范围：既有 validation 保留窗口及既有训练统计；本次仅确定性电压回放与展示，未启动新拟合或训练。'],sections=sections,
        sources=[str(front/'summary.json'),str(posterior/'summary.json'),str(original/'summary.json'),str(original.parent/'power_cap_check_v2/summary.json'),
            '实现：src/data/dual_component_tokenizer.py 中 produce_teacher、voltage_power_cap_bank/frames/reference、project_voltage_second_moment、voltage_power_cap_target；src/inference/shared_driver_reconstruction.py 的 Tak 条件求解。',
            '导出：src/analysis/dual_component_tokenizer.py 的 voltage_power_cap_intermediate_replay 与 render_voltage_power_cap_method_report；experiments/scripts/run_dual_component_tokenizer.py 的 power-cap-method-report 阶段。',
            '数据与范围：docs/DATA_CONTRACT.md；可复用权重说明：docs/METHOD_RATIONALE.md；证据索引：experiments/RESULTS_INDEX.md。',
            '数据背景：Shin 等，2018，Simultaneous acquisition of EEG and NIRS during cognitive tasks for an open access dataset. Scientific Data 5:180003. https://www.nature.com/articles/sdata20183',
            '模型背景：Tak 等，2015，Dynamic causal modelling for functional near-infrared spectroscopy. NeuroImage 111:338–349. doi:10.1016/j.neuroimage.2015.02.035. https://www.fil.ion.ucl.ac.uk/~wpenny/publications/tak-penny15.pdf',
            '原 SPM fNIRS 状态方程实现参考：https://github.com/spm/spm12/blob/main/toolbox/dcm_fnirs/spm_fx_fnirs.m 。本报告具体目标定义与数值以项目保留代码及数组为准。'])


def summarize_voltage_power_cap(out,cfg,plan,outcomes,fits):
    """One-window paired changes; known injections are the only true changes."""
    out=Path(out)
    def read(path):
        with np.load(path,allow_pickle=False) as z:return dict(z)
    inputs=read(out/'inputs.npz');variants=['ocular',*cfg['power_cap']['interventions']['neural']]
    caps=[cfg['power_cap']['reference']['cap_primary'],*cfg['power_cap']['reference']['cap_sensitivity']]
    lookup={r['key']:r for r in outcomes};rows=[];natural=[];decomposition=[]
    fp1=list(inputs['channels']).index('Fp1');x=inputs['x'][6000:10000]
    for cap in [None,*caps]:
        methods=['current_target'] if cap is None else cfg['power_cap']['conditions'][1:]
        for method in methods:
            baseline_method='physical_refitted_reference' if method=='physical_frozen_reference' else method
            baseline_key=f'{baseline_method}__baseline'+('' if cap is None else f'__cap{cap:g}')
            path=out/'targets'/(baseline_key+'.npz')
            if not path.exists():continue
            base=read(path)
            for view,indices in [('Fp1',[fp1]),('all16',list(range(16)))]:
                raw=x[:,indices];s=base['s_score'][:,indices];ac=raw-raw.mean(0)
                diff=(s-raw)-(s-raw).mean(0)
                natural.append(dict(method=method,cap=cap,view=view,
                    baseline_AC_distortion=float(np.linalg.norm(diff)/np.linalg.norm(ac)),
                    baseline_AC_RMS_ratio=float(np.linalg.norm(s-s.mean(0))/np.linalg.norm(ac)),
                    baseline_offset_change_RMS_uV=float(np.sqrt(np.mean((s-raw).mean(0)**2)))))
            for variant in variants:
                key=f'{method}__{variant}'+('' if cap is None else f'__cap{cap:g}')
                path=out/'targets'/(key+'.npz')
                if not path.exists():continue
                changed=read(path);n=inputs['n_'+variant][6000:10000]
                for view,indices in [('Fp1',[fp1]),('all16',list(range(16)))]:
                    result=separation_change_metrics(base['s_score'][:,indices],changed['s_score'][:,indices],
                        n[:,indices],variant,base_o=base['o_score'][:,indices],changed_o=changed['o_score'][:,indices],
                        score_slice=slice(None))
                    result.update(S_error_NRMSE=float(np.sqrt(result['change_NMSE'])),
                        O_error_NRMSE=float(np.sqrt(result['O_change_NMSE'])),
                        S_change_RMS_uV=float(np.sqrt(np.mean((changed['s_score'][:,indices]-base['s_score'][:,indices])**2))))
                    if abs(result['change_NMSE']-result['O_change_NMSE'])>1e-9:
                        raise ValueError('paired S/O closure does not match the known-change error identity')
                    rows.append(dict(method=method,cap=cap,variant=variant,view=view,**result))
    for cap in caps:
        keys=[f'physical_refitted_reference__baseline__cap{cap:g}',
              f'physical_frozen_reference__ocular__cap{cap:g}',f'physical_refitted_reference__ocular__cap{cap:g}']
        if not all((out/'targets'/(k+'.npz')).exists() for k in keys):continue
        base,frozen,refitted=[read(out/'targets'/(k+'.npz')) for k in keys]
        first=frozen['s_score']-base['s_score'];second=refitted['s_score']-frozen['s_score']
        total=refitted['s_score']-base['s_score'];n=inputs['n_ocular'][6000:10000]
        rel=refitted['reference_power']/base['reference_power']
        use=(refitted['frame_time_s']>=30)&(refitted['frame_time_s']<50)
        for view,indices in [('Fp1',[fp1]),('all16',list(range(16)))]:
            denominator=float(np.linalg.norm(n[:,indices]))
            decomposition.append(dict(cap=cap,view=view,
                frozen_reference_response_NRMSE=float(np.linalg.norm(first[:,indices])/denominator),
                changed_reference_additional_response_NRMSE=float(np.linalg.norm(second[:,indices])/denominator),
                total_response_NRMSE=float(np.linalg.norm(total[:,indices])/denominator),
                addition_identity_max_abs=float(np.max(abs(total-first-second))),
                reference_power_ratio_quantiles=np.quantile(rel[use,:5][:,:,indices],[0,.5,.9,1]).tolist()))
    for name,values in [('paired_metrics',rows),('baseline_diagnostics',natural),('response_decomposition',decomposition)]:
        with (out/(name+'.csv')).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]) if values else ['status'])
            writer.writeheader();writer.writerows(values)
    return dict(schema='voltage_power_cap_one_window_results_v1',status='completed',
        experiment_id=cfg['experiment_id'],case_id=plan['case']['id'],
        counts=dict(independent_subjects=1,paired_windows=1,training_windows=6,
            planned_fits=len(fits),fit_successes=sum(r['status']=='completed' for r in fits),
            fit_failures=sum(r['status']!='completed' for r in fits),
            planned_target_tasks=len(outcomes),target_successes=sum(r['status']=='completed' for r in outcomes),
            target_failures=sum(r['status']!='completed' for r in outcomes)),
        primary_cap=1.,sensitivity_caps=caps[1:],paired_metrics=rows,baseline_diagnostics=natural,
        observation_weights=cfg['power_cap']['physical'].get('observation_weights',[1.,1.,1.]),
        response_decomposition=decomposition,proxy_nullspace=json.loads((out/'proxy_nullspace.json').read_text()),
        target_diagnostics=outcomes,physical_fits=fits,
        metric_definitions=dict(ocular='norm(delta S)/norm(known ocular addition)',
            neural='norm(delta S-known neural-like addition)/norm(known neural-like addition)',
            O_error='same norm error as S because O=x-S; not independent evidence',
            projection_gain='dot(delta branch,known addition)/dot(known addition,known addition); not energy fraction',
            primary_support='[30,50) s at native 200 Hz, 4000 samples; separate Fp1 and all16 norms',
            baseline_distortion='relative centred AC waveform change; natural baseline is not clean truth',
            final_power='local carrier cap is exact; bound violations after overlap-add/refiltering are retained'),
        claim_limits=cfg['claim_limits'],threshold_selection='none')


def summarize_voltage_separation(out,plan,evaluations,physical,models,books,cfg):
    import pandas as pd
    out=Path(out);good=[r for r in evaluations if r['status']=='completed']
    natural=[r for e in good for r in e['natural']];changes=[r for e in good for r in e['changes']]
    decoded=[r for e in good for r in e['decoded']];hard_changes=[r for e in good for r in e['hard_changes']]
    frames={'natural_windows':natural,'intervention_windows':changes,'decode_windows':decoded,'hard_intervention_windows':hard_changes}
    for name,rows in frames.items():pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    summaries={}
    for name,keys,metrics in [
        ('intervention',['method','kind','amplitude'],['change_NMSE','S_signed_gain','O_signed_gain','delta_O_correlation']),
        ('natural',['method'],['change_NMSE','O_fine_RMS_ratio','S_fine_RMS_ratio','O_roughness','raw_EOG_correlation','S_EOG_correlation','O_EOG_correlation']),
        ('decode',['method','branch','levels'],['NMSE','fine_error_NMSE','fine_RMS_ratio','fine_correlation','boundary_error_RMS']),
        ('hard_intervention',['method','kind','amplitude'],['change_NMSE','O_change_NMSE','S_signed_gain','O_signed_gain','delta_O_correlation'])]:
        frame=pd.DataFrame(frames[name+'_windows'])
        if frame.empty:summaries[name]=[];continue
        subject=frame.groupby(['subject']+keys,as_index=False)[metrics].mean()
        subject.to_csv(out/(name+'_subjects.csv'),index=False)
        population=subject.groupby(keys,as_index=False)[metrics].mean()
        for metric in metrics:
            if metric.endswith('NMSE'):population[metric.removesuffix('NMSE')+'NRMSE']=np.sqrt(np.maximum(population[metric],0))
        population['subjects']=subject.groupby(keys).size().to_numpy()
        population.to_csv(out/(name+'_summary.csv'),index=False);summaries[name]=population.to_dict('records')
    # Paired, capacity-matched physical-versus-empirical contrasts at the middle amplitude.
    frame=pd.DataFrame(changes);sub=frame[frame.amplitude==.25].groupby(['subject','method','kind']).change_NMSE.mean().reset_index()
    paired=[];rng=np.random.default_rng(cfg['separation']['seed'])
    for kind,g in sub.groupby('kind'):
        table=g.pivot(index='subject',columns='method',values='change_NMSE')
        if {'physical_matched','empirical_matched'}<=set(table):
            d=np.sqrt(table.empirical_matched.to_numpy())-np.sqrt(table.physical_matched.to_numpy())
            boot=d[rng.integers(0,len(d),(2000,len(d)))].mean(1)
            paired.append(dict(kind=kind,subjects=len(d),mean_subject_improvement=float(d.mean()),
                percentile_95=list(np.quantile(boot,[.025,.975])),contrast='empirical_minus_physical_same_rank8_eta0.5'))
    pd.DataFrame(paired).to_csv(out/'paired_physical.csv',index=False)
    return dict(schema='voltage_separation_results_v1',status='completed',experiment_id=cfg['experiment_id'],
        counts=dict(planned_windows=len(plan['rows']),validation_windows=len(evaluations),completed_validation_windows=len(good),
            validation_subjects=len({e['entry']['subject'] for e in good}),physical_fits=len(physical),
            physical_successes=sum(r['status']=='completed' for r in physical),physical_failures=sum(r['status']!='completed' for r in physical),
            model_successes=sum(r['status']=='completed' for r in models),model_failures=sum(r['status']!='completed' for r in models),
            codebook_successes=sum(r['status']=='completed' for r in books),codebook_failures=sum(r['status']!='completed' for r in books)),
        selection=[r.get('selected',dict(method=r['method'],status='failed')) for r in models],
        **summaries,paired_physical=paired,claim_limits=cfg['claim_limits'],
        normalization='time/channel ratios within windows, equal windows within subject, equal subjects; RMS roots after squared-error aggregation',
        raw_signal_branch='native_200Hz_16_channels_unchanged_reference',model_deployment_inputs='EEG_only_140ms_multichannel_support_per_100ms_code_cell',
        EOG_usage='validation_donor_and_independent_diagnostic_only',decoder='own_integer_ids_only_no_raw_residual_shortcut')


def verify_voltage_separation_evidence(out):
    """Replay retained arrays, actual hard O changes and the frozen read boundary."""
    from src.data.dual_component_tokenizer import voltage_separation_plan,read_voltage_separation_input
    from src.tokenizers.dual_component_tokenizer import WaveformDictionary
    out=Path(out);resolved=json.loads((out/'resolved_config.json').read_text());cfg=resolved['config']
    plan=voltage_separation_plan(resolved['project_root'],cfg)
    if plan!=json.loads((out/'plan.json').read_text()):raise ValueError('panel identity changed')
    scale=np.asarray(json.loads((out/'scales.json').read_text())['voltage_scale'])
    methods=json.loads((out/'selected_models.json').read_text());records=[];sums=[];max_error=0.;replays=0
    for entry in plan['rows']:
        if entry['phase']!='validation':continue
        folder=out/'evaluation'/entry['id'];report=json.loads((folder/'metrics.json').read_text())
        if report['status']!='completed':continue
        original,_=read_voltage_separation_input(plan,entry,include_eog=True)
        with np.load(folder/'base.npz',allow_pickle=False) as z:base=dict(z)
        for key in ('x','eog','channels','legacy_s'):
            if not np.array_equal(original[key],base[key]):raise AssertionError('raw/reference identity mismatch: '+key)
        for key,ids in base.items():
            if not key.startswith('ids_'):continue
            name=key.removeprefix('ids_');prefix,levels=name[:-1],int(name[-1])
            with np.load(out/'codebooks'/(prefix+'.npz'),allow_pickle=False) as z:book=WaveformDictionary.from_state(dict(z))
            decoded=book.decode(ids).transpose(0,2,1).reshape(4000,16)*scale
            error=float(np.max(abs(decoded-base['hard_'+name])));max_error=max(max_error,error)
            if not np.allclose(decoded,base['hard_'+name],rtol=2e-6,atol=2e-5):raise AssertionError('independent integer decoder mismatch')
            replays+=1
        for name,pred,rate in [('physical_4plus4',base['hard_physical_S4']+base['hard_physical_O4'],560),
                ('physical_4plus8',base['hard_physical_S4']+base['hard_physical_O8'],840),
                ('raw_8',base['hard_raw_raw8'],560)]:
            sums.append(dict(id=entry['id'],subject=entry['subject'],method=name,bits_per_second_per_electrode=rate,
                NMSE=float(np.mean(((np.asarray(pred,float)-base['x'][6000:10000])/scale)**2))))
        for kind in cfg['separation']['measured_perturbations']:
            with np.load(folder/(kind+'.npz'),allow_pickle=False) as z:changed=dict(z)
            n=np.asarray(changed['n'],float);den=float(np.mean(n*n));expected=n if kind.startswith('neural') else 0.
            for row in report['changes']:
                if row['kind']!=kind or row['amplitude']!=.25:continue
                method=row['method'];before=base['x'][6000:10000] if method=='raw' else base[method][6000:10000]
                after=changed['x'] if method=='raw' else changed[method]
                delta=np.asarray(after,float)-np.asarray(before,float)
                computed=float(np.mean((delta-expected)**2)/den)
                if not np.isclose(computed,row['change_NMSE'],rtol=3e-5,atol=3e-6):raise AssertionError('continuous change metric mismatch')
                records.append(dict(id=entry['id'],kind=kind,method=method,error=abs(computed-row['change_NMSE'])))
            for row in report['hard_changes']:
                if row['kind']!=kind or row['amplitude']!=.25:continue
                name=row['method'];method=name.removesuffix('_O8');levels=8 if name.endswith('_O8') else 4
                for branch,key in [('s','change_NMSE'),('o','O_change_NMSE')]:
                    before=base[f'hard_{method}_{branch.upper()}{4 if branch=="s" else levels}']
                    delta=np.asarray(changed[f'hard_{name}_{branch}'],float)-np.asarray(before,float)
                    truth=expected if branch=='s' else n-expected
                    computed=float(np.mean((delta-truth)**2)/den)
                    if not np.isclose(computed,row[key],rtol=4e-5,atol=4e-6):raise AssertionError('actual hard branch change mismatch')
                for branch in ('s','o'):
                    ids=changed[f'ids_{name}_{branch}']
                    with np.load(out/'codebooks'/f'{method}_{branch.upper()}.npz',allow_pickle=False) as z:book=WaveformDictionary.from_state(dict(z))
                    replay=book.decode(ids).transpose(0,2,1).reshape(4000,16)*scale
                    if not np.allclose(replay,changed[f'hard_{name}_{branch}'],rtol=2e-6,atol=2e-5):raise AssertionError('changed integer decode mismatch')
                    replays+=1
    for file in (out/'physical').glob('*.json'):
        status=json.loads(file.read_text())
        if status['status']!='completed':continue
        with np.load(file.with_suffix('.npz'),allow_pickle=False) as z:
            if z['visible'][:,0].any():raise AssertionError('EEG was visible to Hb-only fitting')
    report=dict(status='passed',continuous_metric_recomputations=len(records),integer_decoder_replays=replays,
        max_retained_float32_decode_abs_error_uV=max_error,
        max_continuous_metric_roundoff=max((r['error'] for r in records),default=0.),
        raw_voltage_reference_identity_exact=True,physical_EEG_rows_used=0,
        scope='declared_public_training_and_validation_only',records=records,sum_reconstructions=sums)
    _json(out/'verification.json',report);return report


def render_voltage_separation_report(out,export=None):
    """Auditable static Chinese report and atlas with bitmap figure objects."""
    import html
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scipy.signal import welch
    import pandas as pd
    import fitz
    from fontTools import ttLib,subset
    out=Path(out).resolve();export=Path(export).resolve() if export else out/'export_v1'
    if export.exists():raise ValueError('fresh versioned export required')
    summary=json.loads((out/'summary.json').read_text())
    if summary['status']!='completed':raise ValueError('completed numerical evidence required')
    verification=verify_voltage_separation_evidence(out)
    resolved=json.loads((out/'resolved_config.json').read_text());cfg=resolved['config'];spec=cfg['separation']
    plan=json.loads((out/'plan.json').read_text());models=json.loads((out/'selected_models.json').read_text())
    if 'physical' not in models:raise ValueError('physical candidate unavailable; inspect retained failures before reporting')
    export.mkdir(parents=True);(export/'figures').mkdir();pages=[];gallery=[];formulas={}
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':12,
        'axes.titlesize':13,'axes.labelsize':12,'legend.fontsize':9,'axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    names=dict(raw='原始直通',physical='物理引导＋预测',physical_spatial='物理引导仅空间',empirical='经验 Hb 引导',
        eeg='EEG 慢功率引导',sham='失配状态引导',pca='PCA',random='随机空间',physical_matched='物理（固定容量）',
        empirical_matched='经验（固定容量）',pca_matched='PCA（固定容量）',sham_matched='失配（固定容量）')
    kinds=dict(ocular='眼电形态',fine='细噪声',combined='两类组合',neural10='10 Hz 神经样',neural35='35 Hz 神经样',neural55='55 Hz 神经样')
    colors=dict(raw='#7a8491',physical='#13866f',physical_spatial='#236ab0',empirical='#bc7d16',eeg='#9863b0',sham='#b24d72',pca='#6b6534',random='#666666')
    col_s='#13866f';col_o='#db6b29';col_x='#657080';col_truth='#111111'
    def read(path):
        with np.load(path,allow_pickle=False) as z:return dict(z)
    def save(fig,key):
        name='figures/'+key+'.png';fig.savefig(export/name,dpi=220,bbox_inches='tight',facecolor='white');plt.close(fig);return name
    def add(title,intro,fig,key,caption,evidence,*,atlas=False):
        implementation='source/experiments/scripts/run_dual_component_tokenizer.py'
        (gallery if atlas else pages).append(dict(title=title,intro=intro,figure=save(fig,key),caption=caption,
            evidence=list(evidence)+[implementation]))
    def grid(ax):ax.grid(alpha=.18);ax.axhline(0,color='#aaaaaa',lw=.6,zorder=0)
    def value(method,kind,key='change_NRMSE',family='intervention'):
        return next(r[key] for r in summary[family] if r['method']==method and r['kind']==kind and r['amplitude']==.25)
    def formula_page(title,intro,lines,key,caption,evidence):
        fig,ax=plt.subplots(figsize=(13.5,5.7));ax.axis('off')
        positions=np.linspace(.84,.16,len(lines))
        for y,(formula,description) in zip(positions,lines):
            ax.text(.5,y,formula,ha='center',va='center',fontsize=23)
            ax.text(.5,y-.105,description,ha='center',va='center',fontsize=12)
            formulas[key+str(y)]=formula
        add(title,intro,fig,key,caption,evidence)
    scale=np.asarray(json.loads((out/'scales.json').read_text())['voltage_scale']);rms=float(np.sqrt(np.mean(scale**2)))
    selected=models['physical'];counts=summary['counts']
    main=[m for m in ['raw','physical_spatial','physical','empirical','pca','sham'] if m=='raw' or m in models]
    fig,axes=plt.subplots(1,3,figsize=(13.5,5.3),constrained_layout=True)
    for ax,kind,title in zip(axes,['ocular','fine','neural55'],['主要伪迹应进入 O','细噪声应进入 O','快速神经样变化应留在 S']):
        vals=[value(m,kind) for m in main];ax.bar(np.arange(len(main)),vals,color=[colors[m] for m in main])
        ax.set_xticks(np.arange(len(main)),[names[m] for m in main],rotation=35,ha='right',fontsize=9)
        for i,v in enumerate(vals):ax.text(i,v,f'{v:.3f}',ha='center',va='bottom',fontsize=9)
        ax.set_ylabel('变化误差 / 已知干预 RMS，越低越好');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.margins(y=.17)
    add('O 获得嘈杂细节，但主要眼电仍留在新版 S 中',
        f"完成 {counts['completed_validation_windows']}/{counts['validation_windows']} 个验证窗口，来自 {counts['validation_subjects']} 名被试。"
        f"训练选择的物理方案秩为 {selected['rank']}，预测残差移出比例为 {selected['innovation_fraction']:g}。"
        f"中档干预均方根（RMS）为 {rms*.25:.2f} µV，眼电、细噪声和 55 Hz 保留误差分别为 {value('physical','ocular'):.3f}、{value('physical','fine'):.3f}、{value('physical','neural55'):.3f}。",
        fig,'01_overview','前两图测量污染漏入 S，第三图测量神经样变化漏入 O。三图均越低越好。每种方法可有不同训练选择；相同秩与预测比例的物理增量另页比较。',
        ['summary.json','intervention_summary.csv','selected_models.json'])
    fig,ax=plt.subplots(figsize=(13.5,5.7));ax.axis('off');ax.set_xlim(0,13.5);ax.set_ylim(0,5.7)
    boxes=[(.1,3.9,3.2,1.1,'训练 EEG 的空间协方差\nHb-only 状态引导方向选择'),
        (4.2,3.9,3.5,1.1,'固定空间投影\n提取实际 EEG 电压载波'),(8.8,3.9,4.5,1.1,'固定对称邻域预测\n将部分不可预测变化分配给 O'),
        (1.,1.6,4.7,1.2,'S：投影波形与邻域预测的混合\nO：原始 EEG 减去 S'),
        (7.,1.6,5.7,1.2,'两路各自的 128 项残差波形词表\n各自硬 ID → 各自解码波形')]
    for x,y,w,h,text in boxes:
        ax.add_patch(plt.Rectangle((x,y),w,h,facecolor='#edf3f8',edgecolor='#7a8c9b'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12)
    for a,b in [((3.3,4.45),(4.2,4.45)),((7.7,4.45),(8.8,4.45)),((10.8,3.9),(4.3,2.8)),((5.7,2.2),(7.,2.2))]:
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.8,color='#56738a'))
    ax.text(6.75,.55,'推断输入：16 电极原生 EEG；200 Hz；µV；TP9 参考；最多 20 ms 未来支持',ha='center',fontsize=13)
    add('设计动机：空间选择处理混合，时间预测检验细节分配',
        '目标是让 O 同时承载大尺度污染和输入中已有的细密扰动。空间前端由血红蛋白（Hb）提供的慢状态选择方向，快速电压仍来自脑电（EEG）；邻域预测补充时间结构依据。没有向 O 人工添加纹理，解码器只读本支硬 token。',
        fig,'02_path','连续前端输出本身就是 S/O 监督目标。硬重建还包含有限码率的失真，必须单独评分。时间预测残差可能含正常快速神经活动，其含义由干预与真值对照检验。',
        ['resolved_config.json','selected_models.json','codebook_fits.json'])
    fig,ax=plt.subplots(figsize=(13.5,5.6));ax.axis('off')
    table=[['训练首窗','6 名被试 / 6 窗','空间总协方差、PCA 坐标、预测器、尺度、词表'],
        ['训练末窗','同 6 名被试 / 6 窗','选择同符号、稳定且线性独立的空间方向'],
        ['预定合成扰动','4 个校准种子','选择秩和预测比例；神经保留双倍权重'],
        ['验证窗口','5 名新被试 / 10 窗','天然波形、已知干预、硬重建；不再选择'],
        ['合成真值','6 个拟合种子 / 8 个评价种子','每种场景独立拟合；已知神经与污染分量'],
        ['独立辅助证据','真实 HEOG / VEOG','仅验证诊断与注入供体；不进入模型输入']]
    tb=ax.table(cellText=table,colLabels=['分区 / 对象','观测单元','用途'],cellLoc='left',loc='center',colWidths=[.20,.28,.52]);tb.auto_set_font_size(False);tb.set_fontsize(12);tb.scale(1,2.6)
    add('训练、选择与评价分别使用哪些信息',
        '沿用上一轮已验证的 Simultaneous n-back 原生电压面板：每窗 90 秒，评分为 [30,50) 秒。三个感兴趣区域（ROI）合并为 16 个唯一电极；Hb 保留原缓存与真实采样时刻。所有窗口均为公开训练或验证划分。',
        fig,'03_design','同一被试的两个窗口有关联，不算两个独立被试。PCA 指主成分分析；经验 Hb 引导在训练首窗内留一被试选择 ridge。验证 Hb-only 拟合只供诊断，不参与冻结投影或编码。',
        ['plan.json','coordinates.json','empirical_guidance.json'])
    formula_page('Hb-only 状态提供训练引导，物理模型不合成高频电压',
        '每个 ROI 由六电极、五频带的 log-power 形成一个训练 PCA 代理。代理频率为 10 Hz；Hb-only 拟合隐藏全部 EEG 观测行，仅重估整窗驱动与初态。固定 Tak 参数为 κ=0.64、τ=2 秒、τv=2 秒。',
        [(r'$u_r(k)=a_r\,v_r^{\mathsf{T}}\ell_r(k),\qquad z_r=\widehat{u}_r^{\,\mathrm{Hb\ only}}$',
          'ℓ：30 维、减去首 5 秒参考的自然对数频带功率；v：训练 PCA；a：训练 EEG 尺度'),
         (r'$z_{r,j}=\operatorname{mean}_{k\in j}\{\operatorname{LP}_{0.2\,\mathrm{Hz}}[z_r(k)]\}$',
          '慢引导先作 0.2 Hz 二阶零相位低通，再汇总到 2 秒协方差单元；全窗离线支持')],
        '04_state','HbO/HbR 使用同一个正训练尺度以保留两色相对幅度。物理失败 ROI 用缺失标记屏蔽，不以联合 EEG/Hb 输出补齐。z 是模型条件估计，既非原生 EEG 电压，也非神经真值。',
        ['teacher_config.json','coordinates.json','physical_fits.json'])
    formula_page('固定空间投影：选方向，再回到原电极电压',
        '五个频带为 1–4、4–8、8–13、13–30、30–45 Hz。每 2 秒计算中心化协方差；方向拟合只使用 [15,75) 秒。全带协方差作 10% 对角收缩，电压输出本身不重参考或中心化。',
        [(r'$C_{z,b,r}=\operatorname{mean}_j[z_{r,j}(C_{b,j}-\overline{C}_b)],\quad C_{z,b,r}w=\lambda C_0w$',
          'z 在各窗支持内标准化；正、负关联都允许；选择训练首末窗同符号的方向'),
         (r'$B=C_0W(W^{\mathsf{T}}C_0W)^{-1},\quad P=BW^{\mathsf{T}},\quad s_{\rm spatial}=Px$',
          'W：16×r 滤波权重；B：16×r 回投模式；P：16×16，满足 P²=P')],
        '05_projection','方向先按训练稳定协变强度排序，再在白化坐标中去除线性依赖。总秩限定 4/8/12。空间投影通常不是原电极欧氏正交投影，单电极幅值可能放大；滤波权重也不是脑区活动图。',
        ['resolved_config.json','model_selection.json'])
    formula_page('时间预测：哪些细节从 S 转入 O',
        '每个空间成分用前后各四个采样点预测当前值，间隔 5 ms。预测系数仅由六个训练首窗拟合，ridge 为 0.05；系数之和约束为一，以保留常量成分。两端各四点沿用空间结果。',
        [(r'$y_t=W^{\mathsf{T}}x_t,\quad\widetilde{y}_t=\sum_{l=1}^{4}h_l(y_{t-l}+y_{t+l})/2,\quad\sum_l h_l=1$',
          '每个空间成分具有各自的四个 h；中心采样点不进入预测；参数冻结后作用于验证 EEG'),
         (r'$s_t=B[(1-\eta)y_t+\eta\widetilde{y}_t],\quad o_t=(I-P)x_t+\eta B(y_t-\widetilde{y}_t)$',
          'η：0–1 的预测残差移出比例；O 的两项分别为空间补充和时间预测残差')],
        '06_prediction','η=0 为附件中的纯空间方案，η=1 为完全采用邻域预测。更大的 η 可以压低部分细噪声，也可能损伤快速神经活动；这一取舍用同频段正负干预直接评价。',
        ['resolved_config.json','selected_models.json'])
    model=read(out/'models'/f"physical_r{selected['rank']}.npz");f=np.linspace(0,100,501)
    response=(1-selected['innovation_fraction'])+selected['innovation_fraction']*(model['predictor']@np.cos(2*np.pi*np.arange(1,5)[:,None]*f/200))
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.5),constrained_layout=True)
    im=axes[0].imshow(model['predictor'],aspect='auto',cmap='RdBu_r',vmin=-max(abs(model['predictor']).max(),.1),vmax=max(abs(model['predictor']).max(),.1))
    axes[0].set_xticks(range(4),['±5','±10','±15','±20']);axes[0].set_xlabel('成对邻点时间间隔（ms）');axes[0].set_ylabel('空间成分编号');axes[0].set_title('冻结的四个预测系数');fig.colorbar(im,ax=axes[0],label='无量纲系数')
    for i,row in enumerate(response):axes[1].plot(f,row,lw=1,alpha=.8,label=str(i+1))
    for hz in (10,35,55):axes[1].axvline(hz,color='#555555',ls=':',lw=.8)
    axes[1].axhline(1,color='#111111',ls='--',lw=1);axes[1].set_xlabel('时间频率（Hz）');axes[1].set_ylabel('所选成分的有符号电压增益');axes[1].set_title('实际混合后的频率响应');axes[1].grid(alpha=.2)
    add('实际预测器的频率响应决定了细节损失风险',
        f"物理方案采用秩 {selected['rank']}、η={selected['innovation_fraction']:g}。右图每条曲线对应一个空间成分，只表示时间环节的作用；完整电极响应还包含空间投影。水平虚线是幅度不变，竖虚线为 10、35、55 Hz 对照。",
        fig,'07_response','系数由训练电压的可预测性确定，不是生理频段标签。响应接近零表示该成分在该频率被移入 O；负增益表示符号反转，不能把它当作能量比例。',
        [f"models/physical_r{selected['rank']}.npz",'selected_models.json'])
    formula_page('指标同时惩罚伪迹泄漏和神经样变化损失',
        f"天然背景与 Hb 保持不变，冻结整个前端，重新输入 x+n。三档干预 RMS 为 {rms*.1:.2f}、{rms*.25:.2f}、{rms*.5:.2f} µV；它们是工程扰动范围，并非正常生理范围。",
        [(r'$E_A^2=\operatorname{mean}(\Delta s^2)/\operatorname{mean}(n^2),\quad E_N^2=\operatorname{mean}[(\Delta s-n)^2]/\operatorname{mean}(n^2)$',
          'EA：污染应进入 O 时的 S 泄漏；EN：神经样变化应留在 S 时的损失；均越低越好'),
         (r'$g_O=\langle\Delta o,n\rangle/\langle n,n\rangle,\quad E=\sqrt{\operatorname{mean}_{\rm subject}\operatorname{mean}_{\rm window}(E^2)}$',
          'gO 是有符号投影增益，不是能量占比；先平均平方误差、最后开方')],
        '08_metrics','所有比值先在 [30,50) 秒的完整时间×16 电极数组上计算。天然 EOG 相关与细频段 RMS 只是诊断，不能识别残差中的神经成分。硬解码两支不严格闭合，O 的变化必须直接计算。',
        ['intervention_windows.csv','intervention_subjects.csv','hard_intervention_windows.csv'])
    synthetic=json.loads((out/'synthetic.json').read_text());syntable=pd.DataFrame([r for s in synthetic for r in s['metrics']])
    scene_names=dict(separable='空间可分',overlapping_fine_noise='细噪声共享神经方向',state_correlated_artifact='主要伪迹与状态同步')
    syn_methods=['raw','pca_spatial','physical_spatial','physical_temporal','sham_temporal','known_subspace']
    syn_names=['原始输入','PCA 仅空间','状态引导仅空间','状态引导＋预测','失配引导＋预测','已知神经子空间']
    fig,axes=plt.subplots(1,3,figsize=(13.5,5.5),constrained_layout=True)
    for ax,scenario in zip(axes,spec['synthetic_scenarios']):
        frame=syntable[syntable.scenario==scenario].groupby('method').clean_NMSE.mean()
        v=[np.sqrt(frame[m]) for m in syn_methods];ax.bar(range(len(v)),v,color=['#8a929d','#7b754d','#236ab0',col_s,'#b24d72','#343b44'])
        ax.set_xticks(range(len(v)),syn_names,rotation=35,ha='right',fontsize=9);ax.set_title(scene_names[scenario])
        ax.set_ylabel('误差 / 干净真值 RMS，越低越好');ax.grid(axis='y',alpha=.2);ax.margins(y=.17)
        for i,number in enumerate(v):ax.text(i,number,f'{number:.3f}',ha='center',va='bottom',fontsize=9)
    add('合成真值：结构可分时有效，重叠与混淆仍会失败',
        '每个场景各有 6 个拟合种子、6 个相应选择窗口和 8 个独立评价种子。方法统一秩 8，时间方案 η=0.5。神经、主要伪迹及细噪声的真实电压分别已知；“状态引导”使用生成器状态，并非估计的 Tak 输出。',
        fig,'09_synthetic_summary','柱高是先平均八个种子的归一化平方误差再开方。已知神经子空间具有额外真值信息，仅用于定位结构限制；它不是算法上界。空间重叠场景的实际信号维度不足所选秩时，也会削弱排除污染的能力。',
        ['synthetic.json','resolved_config.json'])
    for scenario in spec['synthetic_scenarios']:
        a=read(out/'synthetic'/scenario/'example.npz');t=np.arange(4000)/200+30;c=0
        centre=t[np.argmax(abs(a['major'][:,c]))];zoom=(t>=max(30.,centre-.35))&(t<=min(50.,centre+.35))
        s=a['physical_temporal'];o=a['x']-s;true_o=a['major']+a['fine']
        fig,axes=plt.subplots(3,2,figsize=(13.5,6.1),constrained_layout=True)
        curves=[[(a['x'][:,c],col_x,'观测'),(a['clean'][:,c],col_truth,'干净真值')],
            [(s[:,c],col_s,'估计 S'),(a['clean'][:,c],col_truth,'干净真值')],
            [(o[:,c],col_o,'估计 O'),(true_o[:,c],col_truth,'真实污染和')]]
        for row,lines in enumerate(curves):
            for col,mask in enumerate([np.ones(len(t),bool),zoom]):
                ax=axes[row,col]
                for wave,color,label in lines:ax.plot(t[mask],wave[mask],color=color,lw=1,ls='--' if color==col_truth else '-',label=label)
                ax.set_ylabel('µV');ax.legend(loc='upper right',fontsize=8);grid(ax)
        axes[0,0].set_title('完整评分窗口');axes[0,1].set_title('真实主要伪迹峰附近放大，独立纵轴')
        for ax in axes[-1]:ax.set_xlabel('窗口内时间（秒）')
        f=syntable[(syntable.scenario==scenario)&(syntable.method=='physical_temporal')]
        numbers={k:np.sqrt(f[k].mean()) for k in ['clean_NMSE','major_S_NMSE','fine_S_NMSE','neural_loss_NMSE']}
        add(scene_names[scenario]+'：已知真值与估计电压直接对照',
            f"固定首个评价种子 {spec['synthetic_eval_seeds'][0]}、电极 {str(a['channels'][c])}，放大位置按真实主要伪迹峰选择。八种子汇总：主要伪迹泄漏 {numbers['major_S_NMSE']:.3f}，细噪声泄漏 {numbers['fine_S_NMSE']:.3f}，神经损失 {numbers['neural_loss_NMSE']:.3f}。",
            fig,'10_synthetic_'+scenario,'灰色观测为真实干净电压、主要伪迹和细噪声之和。黑虚线是生成器真值；绿为 S、橙为 O。右列逐行独立纵轴，用于检查峰形及细节，跨行幅度必须读坐标；定量统计使用全部电极。',
            [f'synthetic/{scenario}/example.npz',f'synthetic/{scenario}/summary.json'])
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.5),constrained_layout=True)
    selections=json.loads((out/'model_selection.json').read_text())
    for ax,method in zip(axes,['physical','empirical']):
        record=next(r for r in selections if r['method']==method);frame=pd.DataFrame(record['grid'])
        table=frame.pivot(index='rank',columns='innovation_fraction',values='selection_score').reindex(index=[4,8,12],columns=[0,.25,.5,.75,1])
        im=ax.imshow(table,aspect='auto',cmap='viridis_r',vmin=0,vmax=max(r['selection_score'] for rec in selections if rec['status']=='completed' for r in rec['grid']))
        for row in range(3):
            for col in range(5):ax.text(col,row,f'{table.iloc[row,col]:.3f}',ha='center',va='center',color='#18212b',fontsize=10)
        chosen=record['selected'];ax.scatter(spec['innovation_fractions'].index(chosen['innovation_fraction']),spec['ranks'].index(chosen['rank']),marker='s',s=650,facecolors='none',edgecolors='red',lw=2)
        ax.set_xticks(range(5),['0','.25','.5','.75','1']);ax.set_yticks(range(3),[4,8,12]);ax.set_xlabel('预测残差移出比例 η');ax.set_ylabel('空间总秩');ax.set_title(names[method]);fig.colorbar(im,ax=ax,label='训练选择目标，越低越好')
    add('参数由训练信息选择，验证图形不参与调参',
        '热图使用四个预定校准种子的眼电、细噪声、10/35/55 Hz 神经样变化。每格为（眼电 NMSE＋细噪声 NMSE＋2×三项神经 NMSE 之和）/8。红框为该方法的最小训练选择目标。',
        fig,'11_selection','图中数值是平方误差目标，不是开方后的 NRMSE。空间方向的稳定性另由训练末窗决定。主方案与各对照按同一规则选参；固定秩 8、η=0.5 的比较用于隔离容量差别。',
        ['model_selection.json','resolved_config.json'])
    fig,axes=plt.subplots(2,3,figsize=(13.5,6.2),constrained_layout=True)
    compare=[m for m in ['raw','physical_spatial','physical','empirical','eeg','pca','sham','random'] if m=='raw' or m in models]
    for ax,kind in zip(axes.flat,spec['measured_perturbations']):
        values=[value(m,kind) for m in compare];ax.bar(range(len(compare)),values,color=[colors[m] for m in compare])
        ax.set_xticks(range(len(compare)),[names[m] for m in compare],rotation=38,ha='right',fontsize=7.8)
        ax.set_title(kinds[kind]);ax.set_ylabel('变化 NRMSE ↓');ax.grid(axis='y',alpha=.2);ax.margins(y=.15)
        for i,v in enumerate(values):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom',fontsize=8)
    add('六类已知变化：污染移出与神经保留必须一起评价',
        f"中档扰动 RMS 为 {rms*.25:.2f} µV；组合扰动中两类成分各自等 RMS 后相加，再统一定标。每柱汇总五名验证被试的两个窗口。前三类期望 ΔS=0，后三类期望 ΔS=已知神经样变化。",
        fig,'12_measured_interventions','细噪声来自独立 25–70 Hz 随机波形；35/55 Hz 神经样正对照落在相同频段，空间表达在预先定义的中央区域。扰动不按已学习的投影方向构造。原始直通对污染误差为 1、对神经变化为 0。',
        ['intervention_summary.csv','intervention_subjects.csv','resolved_config.json'])
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.5),constrained_layout=True)
    matched=['physical_matched','empirical_matched','pca_matched','sham_matched']
    xx=np.arange(6);width=.19
    for j,m in enumerate(matched):
        axes[0].bar(xx+(j-1.5)*width,[value(m,k) for k in spec['measured_perturbations']],width,label=names[m],color=colors[m.removesuffix('_matched')])
    axes[0].set_xticks(xx,[kinds[k] for k in spec['measured_perturbations']],rotation=25,ha='right');axes[0].set_ylabel('变化 NRMSE ↓');axes[0].legend(fontsize=8);axes[0].grid(axis='y',alpha=.2)
    paired=summary['paired_physical'];positions=np.arange(len(paired));means=np.array([r['mean_subject_improvement'] for r in paired]);low=np.array([r['percentile_95'][0] for r in paired]);high=np.array([r['percentile_95'][1] for r in paired])
    axes[1].errorbar(means,positions,xerr=np.stack([means-low,high-means]),fmt='o',color=col_s,capsize=4);axes[1].axvline(0,color='black',ls='--')
    axes[1].set_yticks(positions,[kinds[r['kind']] for r in paired]);axes[1].set_xlabel('经验 NRMSE − 物理 NRMSE；正值有利于物理');axes[1].set_title('被试配对差：均值和 95% 百分位区间');axes[1].grid(alpha=.2)
    add('相同容量下，检查物理引导是否提供额外收益',
        '四个版本固定相同秩 8、η=0.5、协方差与输入时间支持。右图先对每名被试的两个窗口汇总平方误差并开方，再计算经验减物理的配对差，按被试重采样 2,000 次。',
        fig,'13_matched_physics','区间仅描述五名验证被试、冻结训练配置下的不确定性；没有重训，也未作多重比较校正。区间跨零时，不能将均值方向解释为可靠物理优势。不同方法的独立选参结果不替代本页容量匹配证据。',
        ['paired_physical.csv','intervention_subjects.csv'])
    natural=pd.read_csv(out/'natural_subjects.csv');fig,axes=plt.subplots(1,2,figsize=(13.5,5.4),constrained_layout=True)
    for ax,metric,title in zip(axes,['S_EOG_correlation','O_fine_RMS_ratio'],['S 与独立眼电参考的关联','O 保留的细频段幅度']):
        for j,m in enumerate(main):
            rows=natural[natural.method==m];ax.scatter(np.full(len(rows),j),rows[metric],color=colors[m],s=40,alpha=.8)
            ax.plot([j-.25,j+.25],[rows[metric].mean()]*2,color='black',lw=2)
        ax.set_xticks(range(len(main)),[names[m] for m in main],rotation=30,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('电极中位最大 |相关系数|');axes[1].set_ylabel('O 的 25–70 Hz RMS / 原始 EEG 同频段 RMS')
    add('天然数据：眼电关联和细频段幅度提供不同证据',
        '每点是一名被试，横线为五名被试等权均值。左图对每电极计算与 HEOG/VEOG 的最大绝对相关，再取电极中位数和窗均值；参考 EOG 先作 0.5–15 Hz 带通。右图在完整电压上作 25–70 Hz 诊断带通后计算 RMS 比。',
        fig,'14_natural_diagnostics','左图下降说明所定义眼电关联减弱，右图较大说明 O 有更多快速变化；两者都不能证明天然神经活动完整保留。右图是 RMS 比而非能量占比，两支不正交，比例也不保证在 0–1 内。',
        ['natural_windows.csv','natural_subjects.csv'])
    # The following pages retain the previous report's examples, independent of the new outputs.
    examples=json.loads((Path(plan['parent'])/'summary.json').read_text())['examples']
    display=[]
    for example_index,example in enumerate(examples):
        key=example['id'];entry=next(e for e in plan['rows'] if e['id']==key);a=read(out/'evaluation'/key/'base.npz')
        channels=list(a['channels']);t=np.arange(4000)/200+30;x=a['x'][6000:10000];s=a['physical'][6000:10000];o=x-s
        eog=a['eog'][6000:10000];centre=float(t[np.argmax(abs(eog[:,1]-np.mean(eog[:,1])))])
        centre=float(np.clip(centre,30.5,49.5));zoom=(t>=centre-.4)&(t<=centre+.4)
        sites=[channels.index('Fp1'),channels.index('Cz')];display.append(dict(id=key,subject=entry['subject'],zoom_centre_s=centre,
            selection='same_low_middle_high_raw_EOG_examples_from_parent',channels=['Fp1','Cz']))
        local=json.loads((out/'evaluation'/key/'metrics.json').read_text())
        current=next(r for r in local['natural'] if r['method']=='physical')
        for detail in (False,True):
            mask=zoom if detail else np.ones(len(t),bool);fig,axes=plt.subplots(4,2,figsize=(13.5,6.5),constrained_layout=True)
            for col,c in enumerate(sites):
                legacy=a['legacy_s'][6000:10000,c]
                curves=[[(x[:,c],col_x,'原始 EEG','-')],
                    [(legacy,'#735ba6','保留旧 S','-'),(x[:,c]-legacy,'#a88fc1','保留旧 O','--')],
                    [(s[:,c],col_s,'新版 S 目标','-'),(o[:,c],col_o,'新版 O 目标','--')],
                    [(eog[:,0],'#4a69ad','真实 HEOG','-'),(eog[:,1],'#b54e70','真实 VEOG','--')]]
                bound=max(np.max(abs(wave[mask])) for lines in curves[:3] for wave,_,_,_ in lines)*1.06
                for row,lines in enumerate(curves):
                    ax=axes[row,col]
                    for wave,color,label,ls in lines:ax.plot(t[mask],wave[mask],color=color,label=label,ls=ls,lw=.9)
                    if row<3:ax.set_ylim(-bound,bound)
                    ax.set_ylabel('µV');grid(ax);ax.legend(loc='upper right',fontsize=8,ncol=2)
                axes[0,col].set_title(channels[c]+'；本列前三行统一纵轴');axes[-1,col].set_xlabel('窗口内时间（秒）')
            title=entry['subject']+'：'+('放大细节并对照真实眼电' if detail else '原始电压、保留旧目标与新版 S/O')
            add(title,
                f"{entry['record']}，保留第 {entry['slot']+1} 窗；EEG 记录起点 {entry['rows'][0]['eeg_start_s']:.1f} 秒。"
                f"当前窗新版 S 的 EOG 相关诊断为 {current['S_EOG_correlation']:.3f}，O 的细频段 RMS 比为 {current['O_fine_RMS_ratio']:.3f}。"
                +('放大中心由原始 VEOG 峰确定，与模型输出无关。' if detail else '该样例沿用旧报告的原始 EOG 关联分层，不按新版效果挑选。'),
                fig,f'15_example_{example_index}_{"detail" if detail else "full"}',
                '各列前三行相同 µV 刻度，眼电独立刻度。旧目标来自保留的联合物理拟合增益方案，其训练信息与新版不同，仅用于波形参照。天然 EEG 缺少干净真值，不能把移出的全部波形都命名为伪迹。',
                [f'evaluation/{key}/base.npz',f'evaluation/{key}/metrics.json','plan.json'])
        changed=read(out/'evaluation'/key/'combined.npz');c=channels.index('Fp1');sc=changed['physical'];oc=changed['x']-sc
        masks=[np.ones(len(t),bool),(t>=39.6)&(t<40.4)];fig,axes=plt.subplots(4,2,figsize=(13.5,6.4),constrained_layout=True)
        curves=[[(x[:,c],col_x,'原始背景','--'),(changed['x'][:,c],'#282f37','背景＋两类污染','-')],
            [(s[:,c],'#78b5a7','基准 S','--'),(sc[:,c],col_s,'加扰后 S','-')],
            [(o[:,c],'#e6aa7a','基准 O','--'),(oc[:,c],col_o,'加扰后 O','-')],
            [(changed['n'][:,c],col_truth,'已知新增污染','--'),((oc-o)[:,c],col_o,'实际 ΔO','-')]]
        for row,lines in enumerate(curves):
            for col,mask in enumerate(masks):
                ax=axes[row,col]
                for wave,color,label,ls in lines:ax.plot(t[mask],wave[mask],color=color,label=label,ls=ls,lw=1.)
                ax.set_ylabel('µV');grid(ax);ax.legend(fontsize=8,loc='upper right',ncol=2)
        axes[0,0].set_title('Fp1：完整评分窗口');axes[0,1].set_title('固定 39.6–40.4 秒；逐行独立纵轴')
        for ax in axes[-1]:ax.set_xlabel('窗口内时间（秒）')
        result=next(r for r in local['changes'] if r['method']=='physical' and r['kind']=='combined' and r['amplitude']==.25)
        add(entry['subject']+'：加入两类污染后，O 是否同时响应主结构和细节',
            f"已知污染为跨验证被试的眼电形态与独立 25–70 Hz 细噪声之和，RMS={result['injection_RMS_uV']:.2f} µV。"
            f"全 16 电极 S 泄漏 NRMSE={np.sqrt(result['change_NMSE']):.3f}，O 有符号投影增益={result['O_signed_gain']:.3f}，ΔO 与污染相关={result['delta_O_correlation']:.3f}。",
            fig,f'16_combined_{example_index}','下排直接比较已知新增污染和实际 O 增量：主峰、细密振荡、符号与时间应同时对应。上下行和左右列可能独立纵轴，以读数比较幅度；完整误差使用全电极全评分窗口，不使用图形缩放后的数据。',
            [f'evaluation/{key}/combined.npz',f'evaluation/{key}/metrics.json'])
        fig,axes=plt.subplots(3,2,figsize=(13.5,6.1),constrained_layout=True);c=channels.index('Fp1')
        hard_s=a['hard_physical_S4'];hard_o=a['hard_physical_O4'];hard_o8=a['hard_physical_O8']
        lines=[[(s[:,c],col_s,'S 目标','--'),(hard_s[:,c],'#236ab0','S 四级硬重建','-')],
            [(o[:,c],col_o,'O 目标','--'),(hard_o[:,c],'#236ab0','O 四级硬重建','-'),(hard_o8[:,c],'#9b4f9d','O 八级硬重建',':')],
            [((hard_o-o)[:,c],'#236ab0','O 四级误差','-'),((hard_o8-o)[:,c],'#9b4f9d','O 八级误差',':')]]
        for row,curves in enumerate(lines):
            for col,mask in enumerate([np.ones(len(t),bool),zoom]):
                ax=axes[row,col]
                for wave,color,label,ls in curves:ax.plot(t[mask],wave[mask],color=color,label=label,ls=ls,lw=.9)
                ax.set_ylabel('µV');grid(ax);ax.legend(fontsize=8,ncol=2,loc='upper right')
        axes[0,0].set_title('Fp1：完整评分窗口');axes[0,1].set_title('同一原始 VEOG 峰附近，独立纵轴')
        for ax in axes[-1]:ax.set_xlabel('窗口内时间（秒）')
        d4=next(r for r in local['decoded'] if r['method']=='physical' and r['branch']=='O' and r['levels']==4)
        d8=next(r for r in local['decoded'] if r['method']=='physical' and r['branch']=='O' and r['levels']==8)
        add(entry['subject']+'：监督细节经过硬 token 后保留了多少',
            f"O 四级/八级的细频段相关为 {d4['fine_correlation']:.3f}/{d8['fine_correlation']:.3f}，"
            f"细频段误差 NRMSE 为 {np.sqrt(d4['fine_error_NMSE']):.3f}/{np.sqrt(d8['fine_error_NMSE']):.3f}。"
            '这些指标衡量重建是否忠实于 O 目标，不判断目标本身是否纯净。',
            fig,f'17_hard_{example_index}','四级为每支 280 bits/s/电极；八级 O 为 560 bits/s/电极，总码率更高。误差曲线单独一行，避免把量化产生的高频毛刺当成分离出来的噪声；所有硬波形可仅凭本支整数 ID 重放。',
            [f'evaluation/{key}/base.npz','decode_summary.csv','verification.json'])
        if example_index==1:
            from src.data.dual_component_tokenizer import _registered_record
            _,registered=_registered_record(resolved['project_root'],cfg,entry['rows'])
            unit_state=registered.manifest['measurement']['fnirs_preprocessing_state']
            if unit_state['canonical_unit']!='uM':raise ValueError('verified cached micromolar Hb required for this report')
            _json(out/'report_unit_check.json',dict(entry_id=entry['id'],source_identity_verified=True,
                cache_path=str(registered.npz_path),native_unit=registered.manifest['native_contract']['native_unit'],
                preprocessing=unit_state,measured_arrays_read=0))
            co=json.loads((out/'coordinates.json').read_text());features=read(out/'features'/f'{key}.npz');guidance=read(out/'guidance'/f'{key}.npz')
            fig,axes=plt.subplots(3,3,figsize=(13.5,6.2),constrained_layout=True);tt=np.arange(900)*.1
            for roi,row in enumerate(entry['rows']):
                cc=co[row['coordinate']];path=out/'physical'/f'{key}__{roi}';status=json.loads(path.with_suffix('.json').read_text())
                axes[roi,0].plot(tt,guidance['raw_proxy'][:,roi]/cc['sd'][0],color=col_x,lw=1,label='EEG 实测代理')
                if status['status']=='completed':
                    physical=read(path.with_suffix('.npz'))['physical']
                    axes[roi,0].plot(tt,physical[:,0]/cc['sd'][0],color=col_s,lw=1,label='Hb-only 预测代理')
                else:
                    for ax in axes[roi]:ax.text(.5,.87,'Hb-only 未收敛：预测缺失',transform=ax.transAxes,ha='center',fontsize=9,color='#b13e46')
                for j in range(2):
                    axes[roi,j+1].plot(tt,features['hb'][:,roi,j],color=col_x,lw=1,label='保留实测 Hb')
                    if status['status']=='completed':axes[roi,j+1].plot(tt,physical[:,j+1]/cc['hb_factor'],color=col_s,lw=1,label='Hb-only 重建')
                    axes[roi,j+1].set_ylabel('µmol/L')
                roi_name={'left_motor':'左运动区','right_motor':'右运动区','prefrontal':'前额区'}[row['anchor']['region']]
                axes[roi,0].set_ylabel(roi_name+'\n训练代理 SD',fontsize=9)
                for ax in axes[roi]:
                    ax.axvspan(30,50,color='#7895b1',alpha=.12);grid(ax);ax.legend(fontsize=7)
            for ax,title in zip(axes[0],['EEG 慢功率代理','HbO','HbR']):ax.set_title(title)
            for ax in axes[-1]:ax.set_xlabel('窗口内时间（秒）')
            add(entry['subject']+'：物理重建处在慢代理与血红蛋白空间',
                '展示中间 EOG 样例的三个 ROI。灰线为原始 EEG 转换的慢代理或保留 Hb，绿线为同窗 Hb-only 条件估计。蓝灰背景是电压评分支持 [30,50) 秒；物理模型拟合使用完整 Hb 窗口。',
                fig,'18_physical_prediction','Hb 缓存已由原 mmol/L 转为 µmol/L，此处直接显示缓存单位。EEG 代理除以训练标准差（SD），无量纲；电压重建误差另用训练均方根尺度。该页模型曲线不是 200 Hz 电压；验证状态只作解释，不改变冻结编码。',
                [f'features/{key}.npz',f'guidance/{key}.npz','physical_fits.json','coordinates.json','report_unit_check.json'])
    fig,axes=plt.subplots(1,3,figsize=(13.5,5.3),sharey=True,constrained_layout=True);peaks=[]
    for ax,example in zip(axes,examples):
        a=read(out/'evaluation'/example['id']/'base.npz');x=a['x'][6000:10000];s=a['physical'][6000:10000]
        for label,wave,color,ls in [('原始 EEG',x,col_x,'-'),('S 目标',s,col_s,'-'),('O 目标',x-s,col_o,'--')]:
            frequency,power=welch(wave,fs=200,nperseg=400,noverlap=200,detrend='constant',axis=0)
            mean=power.mean(axis=1);ax.semilogy(frequency,mean,color=color,ls=ls,lw=1.2,label=label)
            if label=='O 目标':
                support=(frequency>=25)&(frequency<=70);peaks.append(float(frequency[support][np.argmax(mean[support])]))
        ax.axvspan(25,70,color='#eeeeee',alpha=.6,zorder=0);ax.axvline(50,color='#444444',ls=':',label='50 Hz 参考')
        ax.set_xlim(0,90);ax.set_title(example['subject']+'：全部 16 电极均值');ax.set_xlabel('频率（Hz）');ax.legend(fontsize=9);ax.grid(alpha=.2)
    axes[0].set_ylabel('功率谱密度（µV²/Hz，对数轴）')
    add('三个样例的 O 细频段谱峰均位于 50 Hz',
        '预选样例的 O 在 25–70 Hz 内的谱峰分别为 '+ '、'.join(f'{v:g} Hz' for v in peaks)+'。'
        '灰色背景标出细噪声探测频段；50 Hz 是频率参考线。频谱反映实际分配的波动，不能将整个高频区都解释成噪声。',
        fig,'18b_spectra','Welch 谱使用 2 秒 Hann 分段、50% 重叠，分段去均值只用于谱估计。先求每电极功率谱，再取 16 电极算术平均；三图共用纵轴。高而窄的谱峰不等于宽频段能量占比，也不能证明其余 O 成分的来源。',
        [f'evaluation/{e["id"]}/base.npz' for e in examples])
    example=examples[1];a=read(out/'evaluation'/example['id']/'base.npz');channels=list(a['channels']);t=np.arange(4000)/200+30
    parts=[a[k][6000:10000] for k in ['physical_o_spatial','physical_o_temporal']];total=a['x'][6000:10000]-a['physical'][6000:10000]
    centre=display[1]['zoom_centre_s'];zoom=(t>=centre-.4)&(t<=centre+.4)
    fig,axes=plt.subplots(2,2,figsize=(13.5,5.6),constrained_layout=True)
    for row,name in enumerate(['Fp1','Cz']):
        c=channels.index(name)
        for col,mask in enumerate([np.ones(len(t),bool),zoom]):
            ax=axes[row,col]
            for wave,color,label,ls in [(total,'#111111','完整 O','-'),(parts[0],col_o,'空间补充','--'),(parts[1],'#236ab0','时间预测残差',':')]:
                ax.plot(t[mask],wave[mask,c],color=color,ls=ls,lw=1.,label=label)
            ax.set_ylabel(name+' · µV');ax.legend(fontsize=9,ncol=3);grid(ax);ax.set_xlabel('窗口内时间（秒）')
    axes[0,0].set_title('完整评分窗口');axes[0,1].set_title('同一 VEOG 峰附近，独立纵轴')
    add('O 由空间补充和时间预测残差共同组成',
        f"中间 EOG 样例 {example['subject']}。全部 16 电极评分窗内，空间项 RMS 为 {np.sqrt(np.mean(parts[0]**2)):.2f} µV，时间项为 {np.sqrt(np.mean(parts[1]**2)):.2f} µV；两项逐点相加得到黑色完整 O。",
        fig,'18c_residual_parts','空间项为原始电压减固定投影，时间项为被移出的邻域预测残差。两项可能相关，各自 RMS 不是总 O 的能量百分比。曲线中既有偏移、主要结构，也有快速细节；其物理归属仍需独立证据。',
        [f'evaluation/{example["id"]}/base.npz','selected_models.json'])
    formula_page('硬 tokenizer 的计算路径与码率',
        '冻结前端先从多电极 EEG 生成 S/O，再按每电极 100 ms、20 个采样点切分。各支词表只用训练首窗拟合；残差向量量化依次选择波形原型，最终解码是本支整数查表求和。没有逐窗连续幅值或均值旁路。',
        [(r'$q_{b,j}^{(l)}=\operatorname{nearest}(r_{b,j}^{(l-1)},\mathcal{C}_b^{(l)}),\quad\widehat{x}_{b,j}=a_c\sum_l \mathcal{C}_b^{(l)}[q_{b,j}^{(l)}]$',
          'b 为 S 或 O；每级词表 128 项、每项 20 维波形；ac 是固定训练电极 RMS'),
         (r'$R_b=10L\log_2(128)=70L\quad\mathrm{bits/s/electrode}$',
          '四级 S＋四级 O：560；原始电压八级：560；四级 S＋八级 O：840')],
        '19_codec','词表最近邻距离包含波形误差和权重 0.25 的差分误差；两个尺度均在训练数据确定。前端对每个 100 ms 编码单元最多需要 140 ms EEG 支持。码率为未作熵编码的名义 ID 码率，不含一次性词表存储。',
        ['resolved_config.json','codebook_fits.json','verification.json'])
    fig,axes=plt.subplots(1,3,figsize=(13.5,5.2),constrained_layout=True)
    dec=[next(r for r in summary['decode'] if r['method']=='physical' and r['branch']==b and r['levels']==l) for b,l in [('S',4),('O',4),('O',8)]]
    labels=['S 四级','O 四级','O 八级']
    for ax,metric,title in zip(axes[:2],['fine_error_NRMSE','fine_correlation'],['细频段相对重建误差','细频段波形相关']):
        values=[r[metric] for r in dec];ax.bar(range(3),values,color=[col_s,col_o,'#9b4f9d']);ax.set_xticks(range(3),labels)
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
        for j,v in enumerate(values):ax.text(j,v,f'{v:.3f}',ha='center',va='bottom')
    axes[0].set_ylabel('误差 / 目标 25–70 Hz RMS ↓');axes[1].set_ylabel('相关系数 ↑');axes[1].set_ylim(0,1.12)
    sums=pd.DataFrame(verification['sum_reconstructions']);sum_subject=sums.groupby(['subject','method']).NMSE.mean();sum_mean=np.sqrt(sum_subject.groupby('method').mean())
    order=['raw_8','physical_4plus4','physical_4plus8'];values=[sum_mean[m] for m in order]
    axes[2].bar(range(3),values,color=[col_x,col_s,'#9b4f9d']);axes[2].set_xticks(range(3),['原始八级\n560 bits/s','S4＋O4\n560 bits/s','S4＋O8\n840 bits/s']);axes[2].set_title('总电压重建：相同训练尺度');axes[2].set_ylabel('NRMSE / 每电极训练 RMS ↓');axes[2].grid(axis='y',alpha=.2)
    for j,v in enumerate(values):axes[2].text(j,v,f'{v:.3f}',ha='center',va='bottom')
    add('O 的真实细节能够硬编码，但需要承担有限码率误差',
        f"O 四级/八级细频段相关为 {dec[1]['fine_correlation']:.3f}/{dec[2]['fine_correlation']:.3f}，NRMSE 为 {dec[1]['fine_error_NRMSE']:.3f}/{dec[2]['fine_error_NRMSE']:.3f}。块边界误差为 {dec[1]['boundary_error_RMS']:.3f}/{dec[2]['boundary_error_RMS']:.3f} 个训练电极 RMS。右图只有前两柱码率相同。",
        fig,'20_decode_quality','细频段误差相对目标 25–70 Hz RMS，按窗、被试平均平方误差后开方。边界误差是每 100 ms 边界两侧重建误差之差，除训练电极 RMS 后在窗内求 RMS、再等权平均。相关较高证明压缩忠实度，不证明目标纯净。',
        ['decode_summary.csv','decode_subjects.csv','verification.json'])
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.5),constrained_layout=True)
    for ax,key,title in zip(axes,['change_NRMSE','O_change_NRMSE'],['S 的实际变化误差','O 的实际变化误差']):
        comparison=[('physical','连续目标',col_s),('physical','硬 S4/O4','#236ab0'),('physical_O8','硬 S4/O8','#9b4f9d')]
        for j,(method,label,color) in enumerate(comparison):
            vals=[value('physical',k) if j==0 else value(method,k,key=key,family='hard_intervention') for k in spec['measured_perturbations']]
            ax.bar(xx+(j-1)*.25,vals,.25,label=label,color=color)
        ax.set_xticks(xx,[kinds[k] for k in spec['measured_perturbations']],rotation=28,ha='right');ax.set_ylabel('变化 NRMSE ↓');ax.set_title(title);ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
    add('硬解码层必须直接检验 O，不能借用标签的闭合恒等式',
        '连续目标逐点满足 S+O=原始 EEG，因此两支相对期望变化的误差相等。硬解码分别量化两支，通常不再严格闭合。本页在原始背景和加扰背景上重新编码，再直接相减各自解码结果。',
        fig,'21_hard_interventions','污染条件期望 ΔS=0、ΔO=已知污染；神经样条件期望 ΔS=已知变化、ΔO=0。八级 O 不改变 S 的 ID，因此两种硬 S 的柱应完全一致。若 O 只是量化毛刺，细频段相关与实际变化误差会揭示它。',
        ['hard_intervention_summary.csv','hard_intervention_subjects.csv','verification.json'])
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.3),constrained_layout=True)
    used=['physical','empirical','eeg','pca','sham','random'];fine=np.array([value(m,'fine') for m in used]);maxfine=max(fine.max(),1.)
    markers=['o','s','^','D','P','X']
    for method,marker,number in zip(used,markers,fine):
        xval=value(method,'ocular');yval=max(value(method,k) for k in ['neural10','neural35','neural55'])
        point=axes[0].scatter(xval,yval,c=[number],vmin=0,vmax=maxfine,cmap='viridis',marker=marker,s=100,edgecolors='black')
        axes[0].annotate(names[method],(xval,yval),xytext=(5,5),textcoords='offset points',fontsize=9)
    axes[0].axvline(.5,color='#777777',ls='--');axes[0].axhline(.25,color='#777777',ls='--');axes[0].set_xlabel('主要眼电 S 泄漏 NRMSE ↓');axes[0].set_ylabel('三类神经样变化最差损失 NRMSE ↓');axes[0].grid(alpha=.2);fig.colorbar(point,ax=axes[0],label='细噪声 S 泄漏 NRMSE ↓')
    fixedP=read(out/'models'/'physical_r8.npz')['P'];distances=[]
    for method in ['empirical','eeg','pca','sham','random']:
        p=read(out/'models'/f'{method}_r8.npz')['P'];distances.append(np.linalg.norm(p-fixedP)/np.linalg.norm(fixedP))
    axes[1].bar(range(5),distances,color=[colors[m] for m in ['empirical','eeg','pca','sham','random']]);axes[1].set_xticks(range(5),[names[m] for m in ['empirical','eeg','pca','sham','random']],rotation=25,ha='right');axes[1].set_ylabel('与物理 P 的 Frobenius 差 / 物理 P 范数');axes[1].set_title('固定秩 8：对照确实改变了投影');axes[1].grid(axis='y',alpha=.2)
    add('当前取舍尚未同时满足主要伪迹、细噪声和神经保留',
        '左图每点是一种训练选定的方法，不是独立被试；横轴为眼电泄漏，纵轴为三个神经频率中的最差损失，颜色为细噪声泄漏。虚线是预先写入合同的工程参照 0.5/0.25，细噪声参照为 0.75。',
        fig,'22_tradeoff','工程参照只帮助描述联合目标，不是生理资格或授权门槛。右图用相同秩的矩阵差检查控制是否改变真实方向；相位失配不同于简单翻转状态符号。降低细噪声误差不足以抵消主要伪迹泄漏或神经损伤。',
        ['intervention_summary.csv','selected_models.json','models/physical_r8.npz','resolved_config.json'])
    failures=[r for r in json.loads((out/'physical_fits.json').read_text()) if r['status']!='completed']
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.4),constrained_layout=True)
    axes[0].bar([0,1,2],[counts['physical_successes'],counts['model_successes'],counts['codebook_successes']],color=col_s,label='完成')
    axes[0].bar([0,1,2],[counts['physical_failures'],counts['model_failures'],counts['codebook_failures']],bottom=[counts['physical_successes'],counts['model_successes'],counts['codebook_successes']],color='#b84d55',label='失败保留')
    axes[0].set_xticks([0,1,2],['Hb-only 拟合','空间模型族','分支词表']);axes[0].set_ylabel('任务数，不是独立被试');axes[0].legend();axes[0].grid(axis='y',alpha=.2)
    axes[1].axis('off');text='物理失败明细：\n'+'\n'.join(f"{r.get('subject',r.get('entry_id'))} · 窗 {next(e['slot'] for e in plan['rows'] if e['id']==r['entry_id'])+1} · ROI {r['roi']+1}：{r.get('fit',{}).get('status','异常')}" for r in failures)
    text+=f"\n\n验证窗口：{counts['completed_validation_windows']} / {counts['validation_windows']}\n连续指标重算：{verification['continuous_metric_recomputations']} 组\n整数解码重放：{verification['integer_decoder_replays']} 次\n原始电压与参考身份：精确一致\nHb-only 读取 EEG 观测行：0"
    axes[1].text(0,.96,text,va='top',fontsize=12,linespacing=1.65)
    add('保留失败和完整分母，区分计算完成与科学目标实现',
        f"计划 22 个公开窗口，物理拟合 {counts['physical_fits']} 项，完成 {counts['physical_successes']} 项、失败 {counts['physical_failures']} 项。失败发生于验证诊断，冻结训练投影仍可对全部 10 个验证窗口编码。未用失败窗口的联合拟合代替 Hb-only 结果。",
        fig,'23_integrity','840 组连续指标从保存数组重算；840 次解码只读取对应词表、整数 ID 与固定训练尺度。浮点存储误差单独核对。失败、低质量结果和困难场景均纳入交付；本轮未读取测试或受保护数组。',
        ['physical_fits.json','evaluation_outcomes.json','verification.json','summary.json'])
    fig,ax=plt.subplots(figsize=(13.5,5.5));ax.axis('off')
    rows=[['O 的细密波动','可以编码与重建',f"四级相关 {dec[1]['fine_correlation']:.3f}；八级 {dec[2]['fine_correlation']:.3f}"],
        ['主要眼电移入 O','本版未达到',f"S 泄漏 {value('physical','ocular'):.3f}；原始直通 1.000"],
        ['快速神经活动保留','有明显损伤风险',f"35/55 Hz 损失 {value('physical','neural35'):.3f}/{value('physical','neural55'):.3f}"],
        ['物理引导增量','依端点而异',f"同容量：细噪声、10 Hz 有利；眼电、35/55 Hz 不利"],
        ['当前可支持的结论','细节分配可实现，去噪叙事未成立','先修正空间成分选择与独立神经保留依据']]
    tb=ax.table(cellText=rows,colLabels=['检验目标','结论','证据与用途'],loc='center',cellLoc='left',colWidths=[.20,.28,.52]);tb.auto_set_font_size(False);tb.set_fontsize(12);tb.scale(1,2.7)
    add('结论：细节可重建，当前目标仍不足以支持“干净 S / 噪声 O”',
        '这次实验将“能生成嘈杂 O”与“能正确分离噪声”分开验证。两阶段前端改善细噪声分配，硬码本也能保存相应纹理；但主要眼电仍被 S 保留，并且快速神经样波形发生损失，不能据此直接替换正式监督。',
        fig,'24_conclusion','下一步应优先修正空间选择为何保留眼电，并使用独立天然事件检验神经保留。增加 O 码率可改善压缩忠实度，却无法纠正目标归属。当前合成和公开开发面板均为有限探索，不能外推为全面生理分离结论。',
        ['summary.json','paired_physical.csv','decode_summary.csv','synthetic.json'])
    for entry in plan['rows']:
        if entry['phase']!='validation':continue
        key=entry['id'];a=read(out/'evaluation'/key/'base.npz');channels=list(a['channels']);t=np.arange(4000)/200+30
        for kind in ('base','combined'):
            b=a if kind=='base' else read(out/'evaluation'/key/'combined.npz')
            x=b['x'][6000:10000] if kind=='base' else b['x'];s=b['physical'][6000:10000] if kind=='base' else b['physical']
            fig,axes=plt.subplots(4,4,figsize=(14,8),sharex=True,constrained_layout=True)
            for c,ax in enumerate(axes.flat):
                ax.plot(t,x[:,c],color=col_x,lw=.6,label='观测');ax.plot(t,s[:,c],color=col_s,lw=.65,label='S');ax.plot(t,x[:,c]-s[:,c],color=col_o,lw=.65,ls='--',label='O')
                ax.set_title(channels[c],fontsize=10);ax.set_xlabel('秒',fontsize=9);ax.set_ylabel('µV',fontsize=9);ax.tick_params(labelsize=8);grid(ax)
            axes.flat[0].legend(fontsize=8,ncol=3)
            add(f"{entry['subject']} · 第 {entry['slot']+1} 窗 · {'天然输入' if kind=='base' else '已加入两类已知污染'}",
                f"{entry['record']}；EEG 记录起点 {entry['rows'][0]['eeg_start_s']:.1f} 秒；当前显示窗口内 [30,50) 秒。完整 16 电极，所有预定验证窗口均收录；未按 S/O 效果筛选。",
                fig,'atlas_'+key+'_'+kind,'灰实线：当前观测；绿实线：新版 S 目标；橙虚线：O=观测−S。各电极独立 µV 刻度，跨图幅度必须读取坐标。组合扰动与天然输入背景相同；图集展示监督目标，硬解码见主报告。',
                [f'evaluation/{key}/{kind}.npz','plan.json'],atlas=True)
    _json(export/'example_selection.json',display);_json(export/'figure_formulas.json',formulas)
    report_font=ttLib.TTFont(font,fontNumber=2);options=subset.Options();options.retain_gids=True
    subsetter=subset.Subsetter(options=options);subsetter.populate(text=''.join(p[k] for p in pages+gallery for k in ('title','intro','caption')));subsetter.subset(report_font)
    fontpath=export/'report_font.otf';report_font.save(fontpath)
    def pdf(destination,records):
        doc=fitz.open()
        for i,p in enumerate(records):
            page=doc.new_page(width=1000,height=700);page.insert_font(fontname='ReportNoto',fontfile=str(fontpath))
            for rect,text,size in [(fitz.Rect(34,20,966,73),p['title'],21),
                    (fitz.Rect(34,80,966,156),p['intro'],12),(fitz.Rect(34,615,966,675),p['caption'],11)]:
                remaining=page.insert_textbox(rect,text,fontname='ReportNoto',fontsize=size,lineheight=1.45)
                if remaining<0:raise ValueError(f'PDF text overflow on page {i+1}: {remaining}')
            page.insert_image(fitz.Rect(34,162,966,607),filename=str(export/p['figure']),keep_proportion=True)
            page.insert_text((35,683),'Evidence: '+', '.join(Path(s).name for s in p['evidence']),fontsize=7,color=(.35,.35,.35))
            page.insert_text((35,694),f'{i+1} / {len(records)}  |  TK-EEG-TWO-SCALE-SEPARATION-v1  |  2026-11-05',fontsize=7,color=(.4,.4,.4))
        doc.save(destination,deflate=True,garbage=4);doc.close()
    pdf(export/'EEG_SO_双尺度分离实验报告.pdf',pages);pdf(export/'全部验证窗口与组合干扰图集.pdf',gallery)
    style='body{font:16px/1.7 system-ui,sans-serif;background:#edf1f5;color:#172739;margin:0}main{max-width:1280px;margin:auto}section{background:white;margin:24px 0;padding:28px;border-radius:10px}img{width:100%;height:auto}h1,h2{line-height:1.35}small{color:#536879}a{color:#1769a4}'
    blocks=[]
    for i,p in enumerate(pages+gallery):
        sources=' · '.join(f'<a href="../{html.escape(s)}">{html.escape(s)}</a>' for s in p['evidence'])
        blocks.append(f'<section id="p{i+1}"><h2>{i+1}. {html.escape(p["title"])}</h2><p>{html.escape(p["intro"])}</p><img src="{p["figure"]}" alt="{html.escape(p["title"])}"><p>{html.escape(p["caption"])}</p><small>数值来源：{sources}</small></section>')
    toc=' · '.join(f'<a href="#p{i+1}">{i+1}. {html.escape(p["title"])}</a>' for i,p in enumerate(pages))
    (export/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EEG S/O 双尺度分离实验</title><style>'+style+'</style><main><section><h1>EEG S/O 双尺度分离实验</h1><p>报告汇总保留数值，不替代运行内 summary 与逐窗证据。图中曲线均来自真实计算数组。</p><p><a href="EEG_SO_双尺度分离实验报告.pdf">主报告 PDF</a> · <a href="全部验证窗口与组合干扰图集.pdf">全部验证窗口图集 PDF</a></p><nav>'+toc+'</nav></section>'+''.join(blocks)+'</main></html>',encoding='utf-8')
    _json(export/'pages.json',dict(pages=pages,gallery=gallery))
    report=dict(status='completed',pages=len(pages),atlas_pages=len(gallery),figures='PNG_220dpi',body='searchable_embedded_font',source_summary=str(out/'summary.json'))
    _json(export/'export.json',report);return report


def voltage_teacher_natural_metrics(x, s, eog, training_scale, low_eog):
    """Natural-data diagnostics; raw EEG is a reference, not clean truth."""
    sl = slice(6000, 10000)
    x, s, e = (np.asarray(v, float)[sl] for v in (x, s, eog))
    scale = np.asarray(training_scale, float)
    low = np.asarray(low_eog, bool)[sl]
    def coupling(value):
        a = value-value.mean(0); b = e-e.mean(0)
        denominator = np.sqrt(np.sum(a*a, axis=0)[:,None]*np.sum(b*b, axis=0)[None])
        cor = np.divide(a.T@b, denominator, out=np.zeros_like(denominator), where=denominator>1e-20)
        return float(np.median(np.max(abs(cor), axis=1)))
    return dict(change_NMSE=float(np.mean(((s-x)/scale)**2)),
        low_EOG_change_NMSE=float(np.mean(((s[low]-x[low])/scale)**2)) if low.any() else None,
        low_EOG_samples=int(low.sum()), EOG_correlation=coupling(s),
        raw_EOG_correlation=coupling(x), RMS_ratio=float(np.sqrt(np.mean(s*s)/np.mean(x*x))),
        raw_waveform_correlation=float(np.corrcoef(s.ravel(),x.ravel())[0,1]))


def voltage_teacher_delta_metrics(base_s, changed_s, disturbance, kind):
    """Only the injected change is known on real data; no natural-clean score."""
    sl = slice(6000,10000)
    delta = (np.asarray(changed_s)-np.asarray(base_s))[sl]
    n = np.asarray(disturbance)[sl]
    energy = float(np.mean(n*n))
    if energy <= 0:
        raise ValueError('nonzero intervention required; zero must be scored separately')
    expected = n if kind == 'neural_alpha' else np.zeros_like(n)
    projection = float(np.mean(delta*n)/energy)
    return dict(delta_error_NMSE=float(np.mean((delta-expected)**2)/energy),
        delta_S_RMS_ratio=float(np.sqrt(np.mean(delta*delta)/energy)),
        signed_projection_S=projection, signed_projection_O=1-projection,
        O_delta_error_NMSE=float(np.mean((n-delta-(np.zeros_like(n) if kind=='neural_alpha' else n))**2)/energy),
        injection_RMS_uV=float(np.sqrt(energy)))


def summarize_voltage_teacher(out, panel, evaluations, fits, cfg):
    """One evidence summary; subjects, windows, calibration and failures separate."""
    import pandas as pd
    out=Path(out)
    complete=[e for e in evaluations if e['status']=='completed']
    natural=[r for e in complete for r in e['natural']]
    interventions=[r for e in complete for r in e['perturbations']]
    pd.DataFrame(natural).to_csv(out/'natural_windows.csv',index=False)
    pd.DataFrame(interventions).to_csv(out/'intervention_windows.csv',index=False)
    df=pd.DataFrame(interventions)
    keys=['subject','method','kind','amplitude']
    subjects=df.groupby(keys,as_index=False)[['delta_error_NMSE','signed_projection_S','signed_projection_O']].mean()
    subjects.to_csv(out/'intervention_subjects.csv',index=False)
    aggregate=[]
    for (method,kind,amplitude), group in subjects.groupby(['method','kind','amplitude']):
        aggregate.append(dict(method=method,kind=kind,amplitude=float(amplitude),subjects=len(group),
            delta_error_NRMSE=float(np.sqrt(group['delta_error_NMSE'].mean())),
            signed_projection_S=float(group['signed_projection_S'].mean()),
            signed_projection_O=float(group['signed_projection_O'].mean())))
    paired=[];rng=np.random.default_rng(cfg['voltage_teacher']['seed'])
    for (kind,amplitude),group in subjects.groupby(['kind','amplitude']):
        table=group.pivot(index='subject',columns='method',values='delta_error_NMSE')
        difference=np.sqrt(table['wiener'].to_numpy())-np.sqrt(table['physics_wiener'].to_numpy())
        boot=difference[rng.integers(0,len(difference),(2000,len(difference)))].mean(1)
        paired.append(dict(kind=kind,amplitude=float(amplitude),subjects=len(difference),
            mean_subject_NRMSE_improvement=float(difference.mean()),
            percentile_95=[float(v) for v in np.quantile(boot,[.025,.975])],
            interpretation='positive_favors_physics; exploratory_subject_bootstrap_no_multiplicity_claim'))
    pd.DataFrame(aggregate).to_csv(out/'intervention_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'physics_paired_comparisons.csv',index=False)
    nd=pd.DataFrame(natural); validation=nd[nd.split=='validation']
    ns=validation.groupby(['subject','method'],as_index=False)[['change_NMSE','low_EOG_change_NMSE','EOG_correlation','RMS_ratio']].mean()
    ns.to_csv(out/'natural_subjects.csv',index=False)
    natural_summary=[]
    for method,g in ns.groupby('method'):
        natural_summary.append(dict(method=method,subjects=len(g),
            change_NRMSE=float(np.sqrt(g.change_NMSE.mean())),
            low_EOG_change_NRMSE=float(np.sqrt(g.low_EOG_change_NMSE.mean())),
            EOG_correlation=float(g.EOG_correlation.mean()), RMS_ratio=float(g.RMS_ratio.mean())))
    synthetic=json.loads((out/'synthetic.json').read_text())
    sd=pd.DataFrame(synthetic['metrics']);sd.to_csv(out/'synthetic_metrics.csv',index=False)
    synthetic_summary=[dict(method=m,clean_NRMSE=float(np.sqrt(g.clean_NMSE.mean())),
        mean_correlation=float(g.correlation.mean()),seeds=len(g)) for m,g in sd.groupby('method')]
    # Display only: pick low/median/high natural EOG coupling, distinct subjects.
    candidates=validation[validation.method=='raw'].sort_values(['EOG_correlation','id']).to_dict('records')
    examples=[];used=set()
    for label,q in [('low',.1),('middle',.5),('high',.9)]:
        target=float(np.quantile([r['EOG_correlation'] for r in candidates],q))
        choices=sorted([r for r in candidates if r['subject'] not in used], key=lambda r:abs(r['EOG_correlation']-target))
        chosen=choices[0];used.add(chosen['subject']);examples.append(dict(label=label,**chosen))
    pairchecks=[r for e in complete for r in e['paired_checks']]
    return dict(schema='voltage_teacher_feasibility_v1',status='completed',experiment_id=cfg['experiment_id'],
        counts=dict(planned_windows=panel['planned_windows'],planned_subjects=panel['planned_subjects'],
            prepared_windows=len(evaluations),evaluated_windows=len(complete),evaluation_failures=len(evaluations)-len(complete),
            validation_subjects=int(validation.subject.nunique()),validation_windows=int(validation.id.nunique()),
            physical_fits=len(fits),physical_successes=sum(r['status']=='completed' for r in fits),
            physical_failures=sum(r['status']!='completed' for r in fits),
            joint_physical_failures=sum(r['status']!='completed' and not r['calibration'] for r in fits),
            calibration_failures=sum(r['status']!='completed' and r['calibration'] for r in fits),
            windows_with_physics_abstention=sum(bool(e['physics_failed_rois']) for e in complete)),
        synthetic=synthetic_summary,natural_validation=natural_summary,interventions=aggregate,physics_paired=paired,
        paired_labels=dict(maximum_S_change=max(r['S_change_max_abs'] for r in pairchecks),
            maximum_O_change_error=max(r['O_change_error_max_abs'] for r in pairchecks),
            maximum_closure=max(r['closure_max_abs'] for r in pairchecks),
            meaning='construction_identity_not_a_robustness_result'),
        examples=examples,example_selection='raw_EOG_coupling_near_P10_P50_P90_distinct_validation_subjects_display_only',
        aggregate='time_channel_NMSE_then_equal_window_mean_within_subject_then_equal_subject_mean_then_square_root',
        claim_limits=cfg['claim_limits'],tokenizer_trained=False)


def verify_voltage_teacher_evidence(out):
    """Audit retained labels and re-score saved interventions independently."""
    from src.data.functional_tokenizer import identity_digest
    from src.data.dual_component_tokenizer import voltage_teacher_plan
    out=Path(out);plan=json.loads((out/'plan.json').read_text())
    resolved=json.loads((out/'resolved_config.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    if (summary['status']!='completed' or plan['protected_data']!='forbidden'
            or any(e['split'] not in ('train','validation') for e in plan['rows'])):
        raise ValueError('completed public development evidence required before audit reads')
    cfg=resolved['config'];root=Path(resolved['project_root'])
    parent='experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261028_v3'
    if (cfg['voltage_teacher']['parent_run']!=parent
            or cfg['voltage_teacher']['legacy_voltage_run']!=parent.replace('20261028_v3','20261029_voltage_v1')):
        raise ValueError('exact public baseline lineage required before audit reads')
    if voltage_teacher_plan(cfg,json.loads((root/parent/'inventory.json').read_text()))!=plan:
        raise ValueError('audit plan differs from canonical development identities')
    train={e['id'] for e in plan['rows'] if e['split']=='train'}
    calibration=json.loads((out/'noise_calibration.json').read_text())
    if set(calibration['training_ids']) != train:
        raise ValueError('noise model calibration does not match the frozen train partition')
    for co in json.loads((out/'coordinates.json').read_text()).values():
        permitted={r['id'] for e in plan['rows'] if e['id'] in train for r in e['rows']}
        if not set(co['training_ids']) <= permitted:raise ValueError('PCA used a nontraining identity')
    old=root/cfg['voltage_teacher']['legacy_voltage_run']
    count=0;checks=0;metric_error=0.;closure=0.;exact=0
    for entry in plan['rows']:
        folder=out/'evaluation'/entry['id']
        status=json.loads((folder/'metrics.json').read_text())
        if status['status']!='completed':continue
        with np.load(folder/'base.npz',allow_pickle=False) as ar:a=dict(ar)
        if a['x'].shape!=(18000,16) or not all(np.isfinite(a[k]).all() for k in ('x','physics_wiener','legacy_target')):
            raise ValueError('complete finite native-voltage label support required')
        with np.load(old/'voltage_prepared'/(identity_digest(entry['id'])[:24]+'.npz'),allow_pickle=False) as ar:
            if (not np.array_equal(a['x'][6000:10000].T,ar['eeg_u'])
                    or not np.array_equal(a['legacy_target'][6000:10000].T,ar['eeg_s'])):
                raise ValueError('legacy actual supervision differs from displayed baseline')
        exact+=1;count+=1
        if entry['split']!='validation':continue
        for kind in ('ocular','unreferenced_muscle','neural_alpha'):
            with np.load(folder/f'{kind}.npz',allow_pickle=False) as ar:b=dict(ar)
            n=b['n'].astype(float);denominator=np.mean(n*n)
            if kind!='neural_alpha':
                if not np.array_equal(a['physics_wiener'][6000:10000],b['paired_s']):
                    raise ValueError('paired S changed')
                closure=max(closure,float(np.max(abs(b['paired_s'].astype(float)+b['paired_o']-b['x']))))
            for method in ('raw','legacy_target','eog_regression','wiener','physics_wiener','shifted_physics'):
                key='x' if method=='raw' else method
                delta=b[key].astype(float)-a[key][6000:10000].astype(float)
                expected=n if kind=='neural_alpha' else 0.
                error=float(np.mean((delta-expected)**2)/denominator)
                projection=float(np.mean(delta*n)/denominator)
                row=next(r for r in status['perturbations'] if r['method']==method and r['kind']==kind and r['amplitude']==1.)
                metric_error=max(metric_error,abs(row['delta_error_NMSE']-error),abs(row['signed_projection_S']-projection));checks+=1
    if metric_error>3e-5 or closure>1e-3 or count!=summary['counts']['evaluated_windows']:
        raise ValueError('retained float32 plot/label arrays do not reproduce numerical evidence')
    result=dict(status='passed',completed_windows=count,legacy_supervision_exact_windows=exact,
        independently_rescored_interventions=checks,maximum_metric_difference_float32=metric_error,
        maximum_pair_closure_float32_uV=closure,source_summary=str(out/'summary.json'),
        training_ids=len(train),test_or_protected_signal_reads=0,
        interpretation='float64 numerical metrics; float32 retained waveform export; no additional fits')
    path=out/'verification.json'
    if path.exists() and json.loads(path.read_text())!=result:
        raise ValueError('retained verification changed; use an explicit versioned audit')
    if not path.exists():_json(path,result)
    return result


def render_voltage_teacher_report(out, export=None):
    """Static Chinese evidence report: bitmap figures, searchable PDF body."""
    import html
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scipy.signal import welch
    import pandas as pd
    import fitz
    out=Path(out).resolve(); export=Path(export).resolve() if export else out/'export_v1'
    if export.exists():
        raise ValueError('use a fresh versioned report export; retained reports are immutable')
    summary=json.loads((out/'summary.json').read_text())
    if summary['status']!='completed':raise ValueError('completed numerical evidence required')
    verify_voltage_teacher_evidence(out)
    cfg=json.loads((out/'resolved_config.json').read_text())['config']
    export.mkdir(parents=True);figures=export/'figures';figures.mkdir()
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
        'axes.unicode_minus':False,'font.size':11,'axes.titlesize':12,'axes.labelsize':11,
        'legend.fontsize':9,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    names=dict(raw='原始输入',legacy_target='当前旧目标',eog_regression='EOG 回归',
        wiener='无物理 Wiener',physics_wiener='物理 Wiener',shifted_physics='错位物理',
        physics_oracle='真功率先验（上界）',wrong_physics='错误功率先验')
    colors=dict(raw='#67717d',legacy_target='#d47a1c',eog_regression='#a65d45',
        wiener='#236eb4',physics_wiener='#168368',shifted_physics='#8b5dba',
        physics_oracle='#168368',wrong_physics='#8b5dba')
    pages=[];gallery=[];formulas={}
    def save(fig,name):
        fig.savefig(figures/(name+'.png'),dpi=240,facecolor='white');plt.close(fig)
        return 'figures/'+name+'.png'
    def add(title,intro,fig,name,caption,evidence):
        pages.append(dict(title=title,intro=intro,figure=save(fig,name),caption=caption,evidence=evidence))
    def read_case(key):
        with np.load(out/'evaluation'/key/'base.npz',allow_pickle=False) as a:return dict(a)
    def metric(method,kind,amplitude=1.):
        return next(r for r in summary['interventions'] if r['method']==method and r['kind']==kind and r['amplitude']==amplitude)
    def natural(method):return next(r for r in summary['natural_validation'] if r['method']==method)
    def axgrid(ax):ax.grid(alpha=.18);ax.axhline(0,color='#999999',lw=.6,zorder=0)
    counts=summary['counts'];m=metric('physics_wiener','ocular');w=metric('wiener','ocular')
    with np.load(out/'noise_model.npz',allow_pickle=False) as model_arrays:
        injection_rms=float(np.sqrt(np.mean(model_arrays['voltage_scale']**2)))
        noise_transfer=model_arrays['transfer'].copy()
    methods=['raw','legacy_target','eog_regression','wiener','physics_wiener','shifted_physics']
    fig,axes=plt.subplots(1,3,figsize=(13,5.2),constrained_layout=True)
    for ax,kind,title in zip(axes[1:],['ocular','neural_alpha'],['眼电注入：S 应保持','神经样变化：S 应保留变化']):
        values=[metric(k,kind)['delta_error_NRMSE'] for k in methods]
        ax.bar(range(len(methods)),values,color=[colors[k] for k in methods]);ax.set_xticks(range(len(methods)),[names[k] for k in methods],rotation=38,ha='right')
        ax.set_title(title);ax.set_ylabel('变化误差 / 已知干预 RMS（越低越好）')
        for i,value in enumerate(values):ax.text(i,value,f'{value:.3f}',ha='center',va='bottom',fontsize=9)
        ax.grid(axis='y',alpha=.2)
    syn=summary['synthetic'];axes[0].bar(range(len(syn)),[r['clean_NRMSE'] for r in syn],color=[colors[r['method']] for r in syn])
    axes[0].set_xticks(range(len(syn)),[names[r['method']] for r in syn],rotation=38,ha='right')
    axes[0].set_title('合成真值：电压恢复');axes[0].set_ylabel('误差 / 干净电压 RMS（越低越好）')
    for i,r in enumerate(syn):axes[0].text(i,r['clean_NRMSE'],f"{r['clean_NRMSE']:.3f}",ha='center',va='bottom',fontsize=9)
    axes[0].grid(axis='y',alpha=.2)
    add('EEG S/O 监督目标：先看波形，再判断物理增量',
        f"固定面板：6 名训练、{counts['validation_subjects']} 名验证被试；计划 {counts['planned_windows']} 段，完成 {counts['evaluated_windows']} 段。"
        f"本次测试监督目标，没有训练 tokenizer。眼电注入为 1 倍训练 RMS（{injection_rms:.1f} µV）时，物理版变化误差 {m['delta_error_NRMSE']:.3f}，无物理版 {w['delta_error_NRMSE']:.3f}。",
        fig,'01_overview','左图的真功率先验使用合成干净真值，仅为机制上界；右两图使用真实验证 EEG 背景及已知新增变化。两类图的分母不同，不能横向比较柱高。各组先按窗口、再按被试等权汇总。',
        ['summary.json','synthetic_metrics.csv','intervention_subjects.csv'])

    fig,ax=plt.subplots(figsize=(13,5.6));ax.set_xlim(0,13);ax.set_ylim(0,5.6);ax.axis('off')
    boxes=[(.1,3.8,2.4,1.,'原生 EEG 电压 X\n200 Hz · µV · TP9 参考'),
        (.1,1.8,2.4,1.,'真实 HEOG / VEOG\n训练估计混合 · 当前 EOG 定位'),
        (3.2,3.8,2.6,1.,'EEG 功率减噪声功率\n训练下限 + 新 PCA / 尺度'),
        (3.2,1.8,2.6,1.,'噪声谱协方差 K_N\n信号背景相关 R_S（满秩）'),
        (6.6,3.8,2.5,1.,'固定 Tak 核心 + Hb\n每窗拟合驱动和初态'),
        (6.6,1.8,2.5,1.,'带置信权重的功率融合\n只修正三个代理约束方向'),
        (10.,2.8,2.8,1.3,'全频谱多通道 Wiener\n逆 STFT → S 电压\nO = X − S')]
    for x,y,width,height,label in boxes:
        ax.add_patch(plt.Rectangle((x,y),width,height,facecolor='#eef4fa',edgecolor='#70859c',lw=1.2));ax.text(x+width/2,y+height/2,label,ha='center',va='center',fontsize=11)
    for start,end in [((2.5,4.3),(3.2,4.3)),((2.5,2.3),(3.2,2.3)),((5.8,4.3),(6.6,4.3)),((7.85,3.8),(7.85,2.8)),((5.8,2.3),(6.6,2.3)),((9.1,2.3),(10.,3.2)),((9.1,4.3),(10.,3.7))]:
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#4a6279',lw=1.8))
    formula=r'$S^*=K_S(K_S+K_N)^{-1}X,\quad s^*=\operatorname{ISTFT}(S^*),\quad o^*=x-s^*$'
    ax.text(6.5,.75,formula,ha='center',fontsize=19);formulas['voltage_teacher']=formula
    fig.tight_layout()
    add('物理模型约束慢功率，快速电压来自观测',
        'S/O 最终都保留原通道、200 Hz 时间轴及 µV 单位。Tak 输出的是 10 Hz EEG 功率代理和 Hb 重建，不能单独称为“物理模型生成的 EEG 电压”。本版伪迹条件均值设为零；EOG 回归单列为对照。',
        fig,'02_method','STFT 使用 1 秒 Hann 窗、250 ms 步幅及完整 0–100 Hz 频谱；逆变换恢复原时间轴。EOG 证据限定为 0.5–15 Hz，其覆盖之外的干扰可能被保留。信号背景作 20% 对角收缩，避免三条代理变成低秩电压空间。',
        ['resolved_config.json','noise_calibration.json','coordinates.json'])

    fig,ax=plt.subplots(figsize=(13,5.5));ax.axis('off')
    fusion=r'$\ell^S=\ell^E+\Sigma_E A^\top(A\Sigma_E A^\top+\Sigma_P)^{-1}(\widehat{u}^{\rm phys}-A\ell^E)$'
    uncertainty=r'$\Sigma_E=\operatorname{diag}(0.05+2\rho),\quad \rho=\operatorname{clip}(P_N/P_X,0,1)$'
    ax.text(.5,.83,fusion,ha='center',fontsize=24)
    ax.text(.5,.62,uncertainty,ha='center',fontsize=21)
    ax.text(.03,.40,'每个 100 ms 时点：\n16 电极 × 5 频带 = 80 个 log-power 特征\n3 个 ROI 代理：A 的维度为 3 × 80\n先融合功率，再调整 K_S；不会从 Hb 合成相位',fontsize=13,va='top',linespacing=1.8)
    ax.text(.55,.40,'ρ 越大，EEG 功率估计越不确定\nΣ_P 使用训练 Hb-only 的跨模态预测误差\nlog-power 改变量工程上限为 ±1.4\n频带之外保留原谱统计和信号空间方向',fontsize=13,va='top',linespacing=1.8)
    fig.tight_layout(pad=2);formulas.update(power_fusion=fusion,power_uncertainty=uncertainty)
    add('物理先验怎样进入电压目标：只修正少数功率方向',
        'ℓE 是扣除噪声功率、使用训练下限并减去前五秒参考的自然对数功率；A 是新训练 PCA 与尺度组成的投影。物理输出使用同一新坐标。ΣE 为 log-power 方差权重，ΣP 为代理坐标平方的误差权重。',
        fig,'02b_power_fusion','Tak 血流核心保持固定：κ=0.64、τ=2 秒、τv=2 秒；每个窗口只重估驱动与初态。权重是正则化约定，不表示两份独立真值。若 K_N=0，滤波应恢复原电压，这项性质已通过数值测试。',
        ['resolved_config.json','coordinates.json','physical_variance.json','teacher_config.json'])

    fig,axes=plt.subplots(2,2,figsize=(13,5.8));
    content=[('哪些量只在训练集确定？','EOG → EEG 混合与 ridge\n低 EOG 背景的空间相关\n频带 / 频谱下限、逐通道电压 RMS\n新噪声扣除功率的 PCA 和 EEG / Hb 尺度'),
        ('哪些量在每个 90 秒窗口估计？','EOG 支持的局部噪声协方差\n观测减噪声的功率\nTak 驱动、四个初始状态坐标\n血流核心参数保持固定'),
        ('物理权重如何避免自证？','训练被试首窗：遮蔽全部 EEG 拟合行\n仅用 Hb 拟合，再预测 EEG 代理\n取跨被试预测 MSE 与训练代理方差较大者\n它是保守正则权重，不是后验置信区间'),
        ('干扰诊断固定了什么？','原始背景 EEG、Hb、训练校准\n基准窗口的物理预测\n重算功率、噪声协方差及电压估计\n没有对污染输入重新拟合 Tak')]
    for ax,(title,body) in zip(axes.flat,content):
        ax.axis('off');ax.set_title(title,loc='left',fontweight='bold');ax.text(0,.82,body,va='top',linespacing=1.8,fontsize=12)
    fig.tight_layout(pad=2)
    confidence=json.loads((out/'physical_variance.json').read_text())
    calibration_counts='、'.join(k.rsplit('__',1)[-1]+f" {len(v['calibration_subjects'])}/{v['planned_subjects']}" for k,v in confidence.items())
    diagnostic_rows=[json.loads(path.read_text())['diagnostics'] for path in (out/'evaluation').glob('*/metrics.json')]
    floor_min=min(r['power_floor_fraction'] for r in diagnostic_rows);floor_max=max(r['power_floor_fraction'] for r in diagnostic_rows)
    add('训练校准、窗口拟合和评价各自使用什么信息',
        '所有目标都是完整窗口离线条件估计；Hb 是特权监督。评分 [30,50) 秒，处理支持 90 秒。Hb-only 权重校准只使用成功拟合，其被试覆盖为 '+calibration_counts+'；失败保留，成功子集不能代表完整校准质量。',
        fig,'03_information',f'选窗只使用元数据中的起止次序，不按拟合/噪声筛选。扣噪功率触及训练下限的比例为 {100*floor_min:.1f}%–{100*floor_max:.1f}%（90 秒×电极×频带位置），不表示这些位置全部是伪迹。物理 log-power 修正未触及 ±1.4 上限。',
        ['plan.json','coordinates.json','physical_variance.json','teacher_config.json'])

    fig,axes=plt.subplots(1,2,figsize=(13,5.4));
    formula1=r'$E_{\Delta S}=\sqrt{\frac{\sum_{c,t}(\Delta S_{ct}-d_{ct})^2}{\sum_{c,t}n_{ct}^2}}$'
    formula2=r'$p_S=\frac{\langle\Delta S,n\rangle}{\langle n,n\rangle},\qquad p_O=1-p_S$'
    for ax in axes:ax.axis('off')
    axes[0].text(0,.83,formula1,fontsize=21);axes[0].text(0,.60,'新增伪迹：d = 0\n神经样变化：d = n\n先计算完整 20 秒数组，最后才绘图\n自然改变量另用固定训练逐通道 RMS',fontsize=13,linespacing=1.8,va='top')
    axes[1].text(0,.83,formula2,fontsize=20);axes[1].text(0,.60,'有符号投影：正负和大于 1 都有可能\n它不是来源能量占比；S/O 不要求正交\n窗口先在被试内平均，再等权汇总被试\n区间按整名被试重采样，不把时间点当样本',fontsize=13,linespacing=1.8,va='top')
    fig.tight_layout(pad=2);formulas.update(delta_error=formula1,signed_projection=formula2)
    add('怎样判定干扰去了哪里，而不是只看波形变平',
        '眼电和高频干扰应主要改变 O；神经样 burst 应由 S 保留。变化误差 0 表示完全符合该干预要求。原始直通输入在伪迹诊断中为 1，在神经保留诊断中为 0。自然数据没有干净 EEG 真值。',
        fig,'04_metrics','总体柱高为：每窗时间/通道归一平方误差 → 同被试窗口均值 → 被试等权均值 → 开方。物理配对差另计算每名被试 NRMSE 之差；2000 次整被试 bootstrap 百分位区间仅为探索性，不宣称多重校正显著性。',
        ['intervention_windows.csv','intervention_subjects.csv','physics_paired_comparisons.csv'])

    with np.load(out/'synthetic_example.npz',allow_pickle=False) as archive:synthetic=dict(archive)
    t=np.arange(18000)/200;take=(t>=36)&(t<=44)
    fig,axes=plt.subplots(4,1,figsize=(13,7),sharex=True,constrained_layout=True)
    ylim=max(np.max(abs(synthetic[k][take,0])) for k in ['x','clean','legacy_target','wiener','physics_oracle'])*1.05
    for ax,key in zip(axes,['x','legacy_target','wiener','physics_oracle']):
        ax.plot(t[take],synthetic['clean'][take,0],color='black',lw=1.,label='隐藏干净电压')
        label='原始受污染输入' if key=='x' else names[key]
        ax.plot(t[take],synthetic[key][take,0],color=colors.get(key,colors['raw']),lw=.9,alpha=.8,label=label)
        ax.set_ylim(-ylim,ylim);ax.set_ylabel('µV');ax.legend(loc='upper right',ncol=2);axgrid(ax)
    axes[-1].set_xlabel('窗口内时间（秒）')
    add('合成真值显示：有噪声抑制，但仍有明显电压误差',
        '显示预先固定的评价种子 200、通道 1。黑线是生成器提供的干净神经样电压，仅用于评分与合成 oracle 对照。各方法使用相同原始输入；纵轴完全一致。真实功率先验并不包含快速相位真值。',
        fig,'05_synthetic_waveform','噪声模型只覆盖带通 EOG 证据，合成眼电含更慢的变化；未被该证据覆盖的偏移可能保留。此页展示幅度、相位及偏移，不能把曲线较平滑等同于恢复正确。总体数字来自 12 个独立生成种子。',
        ['synthetic.json','synthetic_example.npz','synthetic_metrics.csv'])

    fig,axes=plt.subplots(1,2,figsize=(13,5.2),constrained_layout=True)
    sl=slice(6000,10000);bias_rows=[];clean=synthetic['clean'][sl]
    for method in ['raw','legacy_target','wiener','physics_oracle']:
        value=synthetic['x' if method=='raw' else method][sl]-clean
        frequencies,power=welch(value,fs=200,nperseg=2000,noverlap=1000,detrend=False,axis=0)
        axes[0].plot(frequencies,10*np.log10(np.maximum(power.mean(1),1e-12)),color=colors[method],label=names[method])
        bias=float(np.mean(value.mean(0)**2)/np.mean(clean**2))
        residual=float(np.mean((value-value.mean(0))**2)/np.mean(clean**2))
        bias_rows.append((method,bias,residual))
    axes[0].set_xscale('symlog',linthresh=.5);axes[0].set_xlim(0,100);axes[0].axvspan(.5,15,color='#dcebdc',alpha=.5)
    axes[0].set_xlabel('频率（Hz；0–0.5 Hz 为线性段）');axes[0].set_ylabel('误差功率谱密度（dB µV²/Hz）');axes[0].set_title('种子 200：S − 干净真值的频谱');axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
    axes[1].bar(range(4),[r[1] for r in bias_rows],color='#a2afb9',label='窗口均值偏差平方')
    axes[1].bar(range(4),[r[2] for r in bias_rows],bottom=[r[1] for r in bias_rows],color='#236eb4',label='去均值后的误差方差')
    axes[1].set_xticks(range(4),[names[r[0]] for r in bias_rows],rotation=25,ha='right');axes[1].set_ylabel('误差平方 / 干净电压均方');axes[1].set_title('误差分解：均值偏差 + 波形误差');axes[1].legend(fontsize=9)
    wr=next(r for r in bias_rows if r[0]=='wiener');fraction=wr[1]/(wr[1]+wr[2])
    add('合成负结果的定位：慢偏移也属于原电压的一部分',
        f"同一预选种子 200 中，无物理 Wiener 的误差有 {100*fraction:.1f}% 来自 20 秒窗口的均值偏差。绿色频率区为 EOG 参考带通的名义范围。这里保留 DC，没有为评价而额外去均值。",
        fig,'05b_synthetic_error_structure','左图用 10 秒 Welch 窗、50% 重叠、禁止去均值；右图是完整 20 秒数组的精确二阶分解。该例提示 0.5–15 Hz 噪声证据不足以处理全部慢伪迹；这是范围诊断，不据此重新调参或改写本组结果。',
        ['synthetic_example.npz','resolved_config.json'])

    fig,axes=plt.subplots(1,3,figsize=(13,5.2),constrained_layout=True)
    nd=pd.read_csv(out/'natural_subjects.csv')
    for ax,column,title,square in zip(axes,['EOG_correlation','low_EOG_change_NMSE','RMS_ratio'],
            ['S 与 EOG 的相关（仅诊断）','低 EOG 区间的波形改变量','S / 原始 EEG 的 RMS 比'],[False,True,False]):
        for j,method in enumerate(methods):
            vals=nd[nd.method==method][column].to_numpy();vals=np.sqrt(vals) if square else vals
            ax.scatter(j+np.linspace(-.1,.1,len(vals)),vals,color=colors[method],s=36)
            ax.plot([j-.2,j+.2],[np.mean(vals)]*2,color=colors[method],lw=2)
        ax.set_xticks(range(len(methods)),[names[k] for k in methods],rotation=35,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('逐通道 max |r(EEG,EOG)| 的中位数')
    axes[1].set_ylabel('改变量 / 固定训练电压 RMS')
    axes[2].set_ylabel('无量纲幅度比');axes[2].axhline(1,color='#888888',ls='--')
    add('天然眼电相关降低，也必须同时检查波形损伤',
        f"每个点代表一名验证被试（n={counts['validation_subjects']}）；短横线为被试均值。低 EOG 阈值只来自训练窗口的 EOG 活动中位数；低 EOG 不等于完全无伪迹，更不等于神经真值。",
        fig,'06_natural_diagnostics','相关下降可能来自去眼电，也可能来自误删与 EOG 相关的脑活动。幅度变小本身不是收益。低 EOG 改变量用于提示损伤风险，不能把保留原始背景的指标解释为绝对去噪精度。',
        ['natural_windows.csv','natural_subjects.csv','noise_calibration.json'])

    fig,axes=plt.subplots(1,3,figsize=(13,5.2),constrained_layout=True)
    for ax,kind,title in zip(axes,['ocular','unreferenced_muscle','neural_alpha'],
            ['有辅助参考的眼电','无辅助参考的高频干扰','应保留的神经样 α burst']):
        for method in methods:
            rows=sorted([r for r in summary['interventions'] if r['method']==method and r['kind']==kind],key=lambda r:r['amplitude'])
            ax.plot([r['amplitude'] for r in rows],[r['delta_error_NRMSE'] for r in rows],marker='o',color=colors[method],label=names[method])
        ax.set_title(title);ax.set_xlabel('注入 RMS / 固定训练 RMS');ax.set_xticks([.5,1,2]);ax.set_ylabel('变化误差 / 已知干预 RMS');axgrid(ax)
    axes[0].legend(loc='best',fontsize=9)
    add('三类干预检验不同能力：去伪迹与保留神经变化',
        f'0.5、1、2 倍训练 RMS 对应 {injection_rms*.5:.2f}、{injection_rms:.2f}、{injection_rms*2:.2f} µV 的全电极/20 秒注入 RMS，是工程压力测试，不是正常生理幅度范围。眼电取未参与训练的供体，混合有 30% 通道偏差；高频干扰为 25–70 Hz，神经样 burst 为 10 Hz。',
        fig,'07_interventions','左、中两图期望 ΔS=0；右图期望 ΔS=神经样变化，均越低越好。EOG 不变时，固定线性回归本就精确保留加性神经变化；它的零误差不证明 EOG 不含脑信号。每点按窗口和被试汇总，九种干预不是独立被试。',
        ['intervention_windows.csv','intervention_subjects.csv','intervention_summary.csv'])

    fig,axes=plt.subplots(1,2,figsize=(13,5.3),constrained_layout=True)
    paired=summary['physics_paired'];labels=[]
    kindnames=dict(ocular='眼电',unreferenced_muscle='高频干扰',neural_alpha='神经样变化')
    for i,r in enumerate(paired):
        lo,hi=r['percentile_95'];center=r['mean_subject_NRMSE_improvement']
        axes[0].errorbar(center,i,xerr=[[max(center-lo,0)],[max(hi-center,0)]],fmt='o',color='#168368',capsize=3)
        labels.append(f"{kindnames[r['kind']]} · {r['amplitude']:g} 倍")
    axes[0].set_yticks(range(len(labels)),labels);axes[0].axvline(0,color='#444444',ls='--');axes[0].set_xlabel('无物理误差 − 物理误差（正值有利于物理）');axes[0].set_title('配对被试差及 95% 探索性区间')
    for i,kind in enumerate(['ocular','unreferenced_muscle','neural_alpha']):
        rows=[metric(method,kind) for method in ['wiener','physics_wiener','shifted_physics']]
        for j,(method,r) in enumerate(zip(['wiener','physics_wiener','shifted_physics'],rows)):
            axes[1].bar(i+(j-1)*.24,r['delta_error_NRMSE'],width=.22,color=colors[method],label=names[method] if i==0 else None)
    axes[1].set_xticks(range(3),['眼电','高频干扰','神经样变化']);axes[1].set_ylabel('变化误差 / 干预 RMS');axes[1].set_title('1 倍强度：正确与错位物理先验');axes[1].legend();axes[1].grid(axis='y',alpha=.2)
    add('物理增量需要同数据、同噪声估计下的配对证据',
        '无物理、有物理、错位物理三个版本共享噪声模型、训练校准与全部输入。错位版本仅将基准物理预测循环移位 45 秒；它是失配控制，不是新的生理解释。区间跨零时不能宣称稳定增益。',
        fig,'08_physics_increment','区间按整名验证被试重采样 2000 次；5 名验证被试只能提供有限的探索性信息。未做多重比较显著性声明。无参考高频干扰上的接近不变，也不能当成物理模型解决了该伪迹。',
        ['physics_paired_comparisons.csv','intervention_subjects.csv'])

    for number,example in enumerate(summary['examples'],1):
        key=example['id'];a=read_case(key);channels=[str(c) for c in a['channels']];subject=example['subject']
        with (out/'prepared'/f'{key}.json').open() as handle:meta=json.load(handle)
        entry=meta['entry'];start=entry['rows'][0]['eeg_start_s'];tag=f'{number:02d}_{subject}'
        case_metrics=json.loads((out/'evaluation'/key/'metrics.json').read_text())
        failed_rois=case_metrics['physics_failed_rois']
        picked=[channels.index(n) for n in ('Fp1','C3','C4') if n in channels]
        while len(picked)<3:picked.append(next(i for i in range(len(channels)) if i not in picked))
        fig,axes=plt.subplots(3,3,figsize=(13,7),sharex=True,constrained_layout=True)
        proxytime=np.arange(900)/10
        roi_labels=[r['anchor']['region'] for r in entry['rows']]
        for i in range(3):
            axes[i,0].plot(proxytime,a['proxy'][:,i],color='#236eb4',lw=.8,label='噪声扣除后的观测代理')
            if i not in failed_rois:
                axes[i,0].plot(proxytime,a['physical'][:,i],color='#168368',lw=1.4,label='Tak 条件重建')
            else:
                axes[i,0].text(.05,.85,'拟合失败：不显示回退值为物理重建',transform=axes[i,0].transAxes,fontsize=9,color='#b64f55')
            axes[i,0].set_ylabel(roi_labels[i]+'\n训练缩放代理')
            for j in (1,2):
                axes[i,j].plot(proxytime,a['hb'][:,i,j-1],color='#67717d',lw=.9,label='缓存实测')
                axes[i,j].plot(proxytime,a['hb_fit'][:,i,j-1],color='#168368',lw=1.4,label='Tak 条件重建')
                axes[i,j].set_ylabel('µM')
            for ax in axes[i]:ax.axvspan(30,50,color='#dbe4eb',alpha=.55,zorder=-1);axgrid(ax)
        for ax,title in zip(axes[0],['EEG 慢功率代理（非电压）','HbO','HbR']):ax.set_title(title);ax.legend(fontsize=8)
        for ax in axes[-1]:ax.set_xlabel('90 秒窗口内时间（秒）')
        good=[i for i in range(3) if i not in failed_rois]
        error=np.sqrt(np.mean(((a['proxy'][300:500,good]-a['physical'][300:500,good])/.025)**2)) if good else float('nan')
        add(f'样例 {number} · {subject}：物理模型实际重建了什么',
            f"{key}；原记录起点 {start:.3f} 秒。每行一个 ROI，灰色区域为评分 [30,50) 秒。成功的 {len(good)}/3 个 ROI 的 EEG 代理 NRMSE 为 {error:.3f}（固定训练 SD=0.025）。Hb 为带通缓存、减五秒参考后的浓度变化。",
            fig,tag+'_physical','新噪声扣除功率与旧含伪迹代理使用不同的训练坐标，不能混用。Tak 在完整窗口联合使用 EEG 代理与 Hb；这是条件重建，不是跨模态预测。生理状态与神经电压真值均未被这张拟合图验证。',
            [f'evaluation/{key}/base.npz',f'prepared/{key}.json','coordinates.json'])

        from src.data.dual_component_tokenizer import voltage_eog_reference
        filtered_eog=voltage_eog_reference(a['eog'],cfg['voltage_teacher'])
        noise_estimate=filtered_eog@noise_transfer
        sl=slice(6000,10000);times=np.arange(4000)/200;c0=picked[0]
        fig,axes=plt.subplots(3,1,figsize=(13,6.4),sharex=True,constrained_layout=True)
        for i in (0,1):
            axes[i].plot(times,a['eog'][sl,i],color='#aab1b9',lw=.8,label='原生辅助电极')
            axes[i].plot(times,filtered_eog[sl,i],color='#236eb4',lw=.9,label='估计器使用的 0.5–15 Hz 参考')
            axes[i].set_ylabel(meta['eog_names'][i]+' · µV');axes[i].legend(ncol=2,loc='upper right')
        axes[2].plot(times,a['x'][sl,c0],color=colors['raw'],lw=.7,label='原始 '+channels[c0])
        axes[2].plot(times,noise_estimate[sl,c0],color=colors['eog_regression'],lw=.8,label='训练混合预测的眼电（协方差依据）')
        axes[2].plot(times,a['x'][sl,c0]-a['physics_wiener'][sl,c0],color=colors['physics_wiener'],ls='--',lw=.9,label='候选 O')
        axes[2].set_ylabel(channels[c0]+' · µV');axes[2].legend(ncol=3,loc='upper right')
        for ax in axes:axgrid(ax);ax.axvspan(9,11,color='#dbe4eb',alpha=.4)
        axes[-1].set_xlabel('评分片段内时间（秒）')
        before=max(abs(np.corrcoef(a['x'][sl,c0],filtered_eog[sl,j])[0,1]) for j in (0,1))
        after=max(abs(np.corrcoef(a['physics_wiener'][sl,c0],filtered_eog[sl,j])[0,1]) for j in (0,1))
        add(f'样例 {number} · {subject}：O 的变化是否有独立眼电证据',
            f'上两行是真实辅助电极，第三行为同一时段的 {channels[c0]}。该电极与两个带通 EOG 的最大绝对相关由 {before:.3f} 变为 {after:.3f}。灰色时段是后续固定两秒放大范围；各行使用独立的 µV 纵轴。',
            fig,tag+'_eog_evidence','EOG 混合只由训练被试估计。棕线是噪声统计模型的电压预测，不是天然伪迹真值；本版 Wiener 没有直接将它作为条件均值扣除。时间上的共同峰值提供眼电证据，但相关下降不能单独证明脑信号无损。',
            [f'evaluation/{key}/base.npz','noise_model.npz','noise_calibration.json'])

        fig,axes=plt.subplots(3,2,figsize=(13,7),sharex=True,constrained_layout=True)
        sl=slice(6000,10000);times=np.arange(4000)/200
        for i,c in enumerate(picked):
            raw=a['x'][sl,c];old=a['legacy_target'][sl,c];new=a['physics_wiener'][sl,c]
            limit=max(np.max(abs(v)) for v in (raw,old,new,raw-old,raw-new))*1.04
            for value,label,color in [(raw,'原始 EEG',colors['raw']),(old,'当前 S 目标',colors['legacy_target']),(new,'新 S 目标',colors['physics_wiener'])]:
                axes[i,0].plot(times,value,color=color,lw=.65,label=label,alpha=.85)
            axes[i,1].plot(times,raw-old,color=colors['legacy_target'],lw=.8,label='当前 O 目标')
            axes[i,1].plot(times,raw-new,color=colors['physics_wiener'],lw=.8,label='新 O 目标')
            for ax in axes[i]:ax.set_ylim(-limit,limit);ax.set_ylabel(channels[c]+' · µV');axgrid(ax)
        axes[0,0].set_title('S 电压与原始观测');axes[0,1].set_title('O = 原始观测 − S（同一电极使用相同纵轴）')
        axes[0,0].legend(ncol=3,loc='upper right');axes[0,1].legend(ncol=2,loc='upper right')
        for ax in axes[-1]:ax.set_xlabel('评分片段内时间（秒；原窗口 30–50 秒）')
        change=np.sqrt(np.mean(((a['physics_wiener'][sl]-a['x'][sl])/a['voltage_scale'])**2))
        c0=picked[0];r0=a['x'][sl,c0];s0=a['physics_wiener'][sl,c0]
        peak0=float(np.max(abs(r0)));peak1=float(np.max(abs(s0)))
        time0=float(np.argmax(abs(r0))/200);time1=float(np.argmax(abs(s0))/200)
        add(f'样例 {number} · {subject}：原始波形与实际 S/O 监督',
            f"新 S 改变量为 {change:.3f} 倍固定训练逐通道 RMS。{channels[c0]} 的绝对峰值由 {peak0:.1f} 变为 {peak1:.1f} µV，发生时点由 {time0:.2f} 变为 {time1:.2f} 秒。当前目标与历史实际监督完全一致；新目标仅为这次候选。",
            fig,tag+'_targets','每行 S 与 O 使用相同 µV 刻度；不同电极为各自刻度，未裁剪峰值。O 可以包含真实脑活动及估计误差，不能直接称为纯噪声。所有通道、完整数组和训练/验证窗口图集另行保留。',
            [f'evaluation/{key}/base.npz',f'prepared/{key}.json','natural_windows.csv'])

        fig,axes=plt.subplots(3,2,figsize=(13,6.3),sharex=True,constrained_layout=True)
        zoom=(times>=9)&(times<=11)
        for i,c in enumerate(picked):
            raw=a['x'][sl,c][zoom];old=a['legacy_target'][sl,c][zoom];new=a['physics_wiener'][sl,c][zoom]
            limit=max(np.max(abs(v)) for v in (raw,old,new,raw-old,raw-new))*1.06
            for value,label,color in [(raw,'原始 EEG',colors['raw']),(old,'当前 S',colors['legacy_target']),(new,'候选 S',colors['physics_wiener'])]:
                axes[i,0].plot(times[zoom],value,color=color,lw=1.,label=label)
            axes[i,1].plot(times[zoom],raw-old,color=colors['legacy_target'],lw=1.,label='当前 O')
            axes[i,1].plot(times[zoom],raw-new,color=colors['physics_wiener'],lw=1.,label='候选 O')
            for ax in axes[i]:ax.set_ylim(-limit,limit);ax.set_ylabel(channels[c]+' · µV');axgrid(ax)
        axes[0,0].set_title('原电压与 S：固定两秒放大');axes[0,1].set_title('O：与同电极 S 图相同刻度')
        axes[0,0].legend(ncol=3);axes[0,1].legend(ncol=2)
        for ax in axes[-1]:ax.set_xlabel('评分片段内时间（秒）')
        add(f'样例 {number} · {subject}：放大看快速波形与残差',
            '所有样例固定放大片段第 9–11 秒，没有按候选方法效果挑选放大范围。原始输入、旧目标与新目标使用相同采样点和电压单位；曲线的峰谷、极性及偏移可直接比较。',
            fig,tag+'_voltage_zoom','本图只放大显示，前页数值仍来自完整 20 秒；未重新滤波、重新拟合或按放大区间归一化。O 中出现有结构的快速波形，不能仅凭其形状判定它是眼电或神经活动。',
            [f'evaluation/{key}/base.npz','natural_windows.csv'])

        with np.load(out/'evaluation'/key/'ocular.npz',allow_pickle=False) as ar:ocular=dict(ar)
        c=picked[0];view=(times>=6)&(times<=14);vtime=times[view]
        raw=a['x'][sl,c];s0=a['physics_wiener'][sl,c];o0=raw-s0
        fig,axes=plt.subplots(3,1,figsize=(13,6.7),sharex=True,constrained_layout=True)
        axes[0].plot(vtime,raw[view],color=colors['raw'],lw=.8,label='原始基础 EEG')
        axes[0].plot(vtime,ocular['x'][view,c],color='#b64f55',lw=.9,label='加入已知眼电后的输入')
        axes[0].plot(vtime,ocular['n'][view,c],color='#935d9c',lw=1.,ls='--',label='已知新增眼电')
        axes[1].plot(vtime,s0[view],color=colors['physics_wiener'],lw=1.2,label='基础 S 标签')
        axes[1].plot(vtime,ocular['paired_s'][view,c],color='#222222',lw=.8,ls='--',label='配对 S 标签（按定义相同）')
        axes[2].plot(vtime,o0[view],color=colors['raw'],lw=.8,label='基础 O 标签')
        axes[2].plot(vtime,ocular['paired_o'][view,c],color='#b64f55',lw=.9,label='配对 O 标签 = 原 O + 新增眼电')
        for ax in axes:ax.set_ylabel(channels[c]+' · µV');ax.legend(ncol=3,loc='upper right');axgrid(ax)
        axes[-1].set_xlabel('评分片段内时间（秒）')
        add(f'样例 {number} · {subject}：加噪训练标签怎样变化',
            '此处展示真正准备交给 S/O 的配对监督规则：S 与基础片段共享同一条电压目标，O 增加已知眼电。各行按自身物理量幅度显示，纵轴仍为 µV。仅展示评分片段第 6–14 秒，指标使用完整 20 秒。',
            fig,tag+'_paired','S 完全不变是标签构造恒等式，不是网络已经学会去噪。基础 S 中若含天然伪迹，配对标签仍会保留它。没有在加噪后重跑 SSM 来修改这份配对 S 标签。',
            [f'evaluation/{key}/ocular.npz',f'evaluation/{key}/metrics.json'])

        fig,axes=plt.subplots(3,2,figsize=(13,6.8),sharex=True,constrained_layout=True)
        n=ocular['n'][view,c]
        for i,method in enumerate(['legacy_target','wiener','physics_wiener']):
            delta=ocular[method][view,c]-a[method][sl,c][view]
            axes[i,0].plot(vtime,n,color='#888888',ls=':',lw=1.,label='新增眼电（尺度参照）')
            axes[i,0].plot(vtime,delta,color=colors[method],lw=1.,label='估计器重算后的 ΔS')
            axes[i,1].plot(vtime,n,color='black',ls='--',lw=1.,label='期望 ΔO = 已知眼电')
            axes[i,1].plot(vtime,n-delta,color=colors[method],lw=1.,label='实际 ΔO')
            lim=max(np.max(abs(n)),np.max(abs(delta)),np.max(abs(n-delta)))*1.05
            for ax in axes[i]:ax.set_ylim(-lim,lim);ax.set_ylabel(names[method]+'\nµV');axgrid(ax)
        axes[0,0].set_title('ΔS：越接近零越抗新增眼电');axes[0,1].set_title('ΔO：应重现已知新增眼电')
        axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
        for ax in axes[-1]:ax.set_xlabel('评分片段内时间（秒）')
        add(f'样例 {number} · {subject}：重算估计器是否真的抗眼电',
            '对污染输入重算功率、噪声统计及电压目标，再减去基础估计。物理预测和训练校准冻结。完整 20 秒、全电极的变化误差（/注入 RMS）：'+
            '；'.join(f"{names[k]} {np.sqrt(next(r['delta_error_NMSE'] for r in case_metrics['perturbations'] if r['method']==k and r['kind']=='ocular' and r['amplitude']==1.)):.3f}" for k in ['legacy_target','wiener','physics_wiener'])+'。',
            fig,tag+'_reestimate','有色曲线贴近左图零线且贴近右图已知眼电，才表示该估计器将新增干扰分给 O。不能用上一页按定义重合的 S 标签替代这项核验。该实验仍不包含污染后重新拟合 Tak 的响应，也不测 tokenizer 泛化。',
            [f'evaluation/{key}/ocular.npz',f'evaluation/{key}/base.npz','intervention_windows.csv'])

    example=summary['examples'][1];key=example['id'];a=read_case(key)
    # The neural intervention has a central cosine topography: its endpoint
    # electrodes receive zero injection. Show a genuinely injected central
    # channel, while the metric continues to score every electrode.
    channels=[str(value) for value in a['channels']]
    c=channels.index('Cz') if 'Cz' in channels else len(channels)//2
    times=np.arange(4000)/200;view=(times>=6)&(times<=14)
    fig,axes=plt.subplots(2,2,figsize=(13,6.3),sharex=True,constrained_layout=True)
    for i,kind in enumerate(['unreferenced_muscle','neural_alpha']):
        with np.load(out/'evaluation'/key/f'{kind}.npz',allow_pickle=False) as ar:b=dict(ar)
        n=b['n'][:,c];expected=n if kind=='neural_alpha' else np.zeros_like(n)
        for j,method in enumerate(['wiener','physics_wiener']):
            delta=b[method][:,c]-a[method][6000:10000,c]
            axes[i,j].plot(times[view],n[view],color='#999999',ls=':',lw=.9,label='已知干预')
            axes[i,j].plot(times[view],expected[view],color='black',ls='--',lw=1.1,label='期望 ΔS')
            axes[i,j].plot(times[view],delta[view],color=colors[method],lw=.9,label='实际 ΔS')
            axes[i,j].set_ylabel(kindnames[kind]+'\nµV');axes[i,j].set_title(names[method]);axes[i,j].legend(fontsize=8);axgrid(axes[i,j])
    for ax in axes[-1]:ax.set_xlabel('评分片段内时间（秒）')
    add('负对照与保留性：没有证据的干扰，不能假装已分离',
        f"显示中位 EOG 样例 {example['subject']} 的 {str(a['channels'][c])} 电极。上排是没有 EOG 参考的高频干扰；下排是与 Hb 保持不变的神经样 α 变化。原始背景和基准物理预测相同。",
        fig,'09_negative_and_neural','上排期望 ΔS=0；若曲线仍复制已知干扰，则噪声模型没有覆盖它。下排期望 ΔS=已知神经样变化；削弱它是损伤风险，不能以“更符合 Hb/SSM”解释为成功。',
        [f'evaluation/{key}/unreferenced_muscle.npz',f'evaluation/{key}/neural_alpha.npz','intervention_windows.csv'])

    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    expected=[66,18];fail=[counts['joint_physical_failures'],counts['calibration_failures']]
    axes[0].bar([0,1],[expected[i]-fail[i] for i in range(2)],label='完成并通过积分/收敛检查',color='#168368')
    axes[0].bar([0,1],fail,bottom=[expected[i]-fail[i] for i in range(2)],label='失败 / 不可用',color='#b64f55')
    axes[0].set_xticks([0,1],['逐窗联合 Tak（22×3）','训练 Hb-only 校准（6×3）']);axes[0].set_ylabel('拟合数（不是独立被试）');axes[0].legend(fontsize=9)
    axes[1].axis('off');p=summary['paired_labels']
    axes[1].text(0,.95,f"配对 S 最大变化：{p['maximum_S_change']:.3g} µV\n"
        f"配对 O 增量误差：{p['maximum_O_change_error']:.3g} µV\n"
        f"配对闭合最大误差：{p['maximum_closure']:.3g} µV\n\n"
        '数值恒等式通过 ≠ 分离真值通过\n物理失败明确使用无先验回退\n失败、原始分母和回退标记均保留\n未训练 tokenizer，也未检验离散解码',va='top',fontsize=13,linespacing=1.9)
    add('完整分母、数值合同和未解决问题一起交付',
        f"计划 {counts['planned_subjects']} 名被试、{counts['planned_windows']} 个窗口；完成电压评价 {counts['evaluated_windows']} 个。物理拟合失败 {counts['physical_failures']} 个，含联合拟合 {counts['joint_physical_failures']}、校准 {counts['calibration_failures']}；{counts['windows_with_physics_abstention']} 个窗口含物理回退。",
        fig,'10_integrity','初次执行在记录 NumPy 标量时中断；新版本只修复记录接口并复用完成输入。闭合数值使用 float64，保存的显示波形为 float32，另行独立重算核验。所有结果为公开开发面板探索；未读测试或受保护数组。',
        ['plan.json','physical_fits.json','summary.json','input_reuse.json'])

    subject_table=pd.read_csv(out/'intervention_subjects.csv')
    table=subject_table[(subject_table.kind=='ocular')&(subject_table.amplitude==1.)].pivot(index='subject',columns='method',values='delta_error_NMSE')
    fig,axes=plt.subplots(1,2,figsize=(13,5.3),constrained_layout=True)
    palette=['#0072b2','#d55e00','#009e73','#cc79a7','#5c6670'];markers=['o','s','^','D','P']
    for ax,baseline,title in zip(axes,['legacy_target','wiener'],['候选目标与当前旧目标','物理项与同构无物理版本']):
        upper=float(np.sqrt(table[[baseline,'physics_wiener']]).to_numpy().max()*1.12)
        ax.plot([0,upper],[0,upper],color='#777777',ls='--',label='两者相同')
        for i,(subject,row) in enumerate(table.iterrows()):
            ax.scatter(np.sqrt(row[baseline]),np.sqrt(row['physics_wiener']),s=85,color=palette[i],marker=markers[i],label=subject)
        ax.set_xlim(0,upper);ax.set_ylim(0,upper);ax.set_aspect('equal');ax.set_xlabel(names[baseline]+'：眼电变化误差');ax.set_ylabel('物理 Wiener：眼电变化误差');ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=9)
    old=metric('legacy_target','ocular');reg=metric('eog_regression','ocular');neural=metric('physics_wiener','neural_alpha');oldneural=metric('legacy_target','neural_alpha')
    improvement=100*(1-m['delta_error_NRMSE']/old['delta_error_NRMSE'])
    add('结论：电压目标有改进，但物理收益尚未得到支持',
        f"1 倍眼电下，候选误差 {m['delta_error_NRMSE']:.4f}，旧目标 {old['delta_error_NRMSE']:.4f}，降低 {improvement:.1f}%；无物理版 {w['delta_error_NRMSE']:.4f}，EOG 回归 {reg['delta_error_NRMSE']:.4f}。神经样变化误差由旧目标 {oldneural['delta_error_NRMSE']:.4f} 降至 {neural['delta_error_NRMSE']:.4f}；无参考高频干扰约为 1，基本未分离。",
        fig,'11_conclusions','每点是一名被试：先在被试内汇总两窗平方误差再开方；点位于左图对角线下方表示候选优于旧目标。下一步应先验证天然伪迹与脑信号损伤、补足噪声证据，再用匹配信息的先验对照测试物理增量；不据此直接替换正式目标。',
        ['summary.json','intervention_subjects.csv','physics_paired_comparisons.csv'])

    # Complete native-channel gallery, with explicit per-electrode display scales.
    entries=json.loads((out/'plan.json').read_text())['rows']
    for entry in entries:
        path=out/'evaluation'/entry['id']/'base.npz'
        if not path.exists():continue
        a=read_case(entry['id']);channels=[str(c) for c in a['channels']];n=len(channels)
        fig,axes=plt.subplots(int(np.ceil(n/4)),4,figsize=(14,9),sharex=True,constrained_layout=True)
        for c,ax in enumerate(axes.flat):
            if c>=n:ax.axis('off');continue
            x=a['x'][6000:10000,c];s=a['physics_wiener'][6000:10000,c]
            ax.plot(times,x,color=colors['raw'],lw=.5,label='原始 EEG')
            ax.plot(times,s,color=colors['physics_wiener'],lw=.6,label='候选 S')
            ax.plot(times,x-s,color=colors['physics_wiener'],ls='--',lw=.6,label='候选 O')
            ax.set_title(channels[c]);ax.set_ylabel('µV');ax.set_xlabel('秒');axgrid(ax)
        axes.flat[0].legend(fontsize=8,ncol=3,loc='upper left')
        filename=save(fig,'gallery_'+entry['id'])
        gallery.append(dict(title=f"{entry['subject']} · {entry['split']} · 窗口 {entry['slot']+1}",
            intro=f"{entry['id']}；评分范围原窗口 30–50 秒；完整 {n} 电极，每个电极独立纵轴，均保留 µV。",
            figure=filename,caption='灰：原始 EEG；绿实线：候选 S；绿虚线：候选 O。独立纵轴仅用于逐电极阅读，跨电极幅度必须读取坐标，不以视觉高度比较。',evidence=[f'evaluation/{entry["id"]}/base.npz']))

    from fontTools import ttLib, subset
    report_font=ttLib.TTFont(font,fontNumber=2)
    options=subset.Options();options.retain_gids=True
    subsetter=subset.Subsetter(options=options)
    subsetter.populate(text=''.join(p[k] for p in pages+gallery for k in ('title','intro','caption')))
    subsetter.subset(report_font);report_font_path=export/'report_font.otf';report_font.save(report_font_path)
    def pdf_export(destination,records):
        doc=fitz.open()
        for i,p in enumerate(records):
            page=doc.new_page(width=1000,height=700)
            page.insert_font(fontname='ReportNoto',fontfile=str(report_font_path))
            for rect,text,size in [(fitz.Rect(34,22,966,73),p['title'],22),
                    (fitz.Rect(34,80,966,154),p['intro'],12),
                    (fitz.Rect(34,616,966,674),p['caption'],11)]:
                remaining=page.insert_textbox(rect,text,fontname='ReportNoto',fontsize=size,lineheight=1.45)
                if remaining<0:raise ValueError(f'PDF text overflow on page {i+1}: {remaining}')
            page.insert_image(fitz.Rect(34,162,966,607),filename=str(export/p['figure']),keep_proportion=True)
            page.insert_text((35,680),'Evidence: '+', '.join(Path(s).name for s in p['evidence']),fontsize=7,color=(.35,.35,.35))
            page.insert_text((35,691),f'{i+1} / {len(records)}  |  TK-EEG-VOLTAGE-TEACHER-v1  |  2026-11-05',fontsize=8,color=(.4,.4,.4))
        doc.save(destination,deflate=True,garbage=4);doc.close()
    pdf_export(export/'EEG_SO_监督目标可行性报告.pdf',pages)
    pdf_export(export/'全部窗口电压图集.pdf',gallery)
    style='body{font:16px/1.7 system-ui,sans-serif;background:#edf1f5;color:#172739;margin:0}main{max-width:1280px;margin:auto}section{background:white;margin:24px 0;padding:28px;border-radius:10px}img{width:100%;height:auto}h1,h2{line-height:1.35}p{max-width:1100px}small{color:#536879}a{color:#1769a4}'
    blocks=[]
    for p in pages+gallery:
        sources=' · '.join(f'<a href="../{html.escape(s)}">{html.escape(s)}</a>' for s in p['evidence'])
        blocks.append(f'<section><h2>{html.escape(p["title"])}</h2><p>{html.escape(p["intro"])}</p><img src="{p["figure"]}" alt="{html.escape(p["title"])}"><p>{html.escape(p["caption"])}</p><small>数值来源：{sources}</small></section>')
    (export/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EEG S/O 监督目标实验</title><style>'+style+'</style><main><section><h1>EEG S/O 监督目标可行性实验</h1><p>静态详细报告与全部窗口图集。以运行目录 summary.json 和逐窗数值为事实源。</p><p><a href="EEG_SO_监督目标可行性报告.pdf">报告 PDF</a> · <a href="全部窗口电压图集.pdf">全部窗口图集 PDF</a></p></section>'+''.join(blocks)+'</main></html>',encoding='utf-8')
    _json(export/'figure_formulas.json',formulas)
    manifest=dict(source_summary=str(out/'summary.json'),report_pages=pages,gallery=gallery,
        bitmap_dpi=240,body_text='searchable_pdf_text',source='retained_arrays_no_new_fits',
        external_method_sources=['https://github.com/exporl/mwf-artifact-removal','https://proceedings.mlr.press/v80/lehtinen18a.html'])
    _json(export/'figure_index.json',manifest)
    return dict(status='completed',export=str(export),report_pages=len(pages),gallery_pages=len(gallery))


def _json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, default=convert, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def _csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(''); return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n'); writer.writeheader(); writer.writerows(rows)


def _cache(run):
    cache = Path(run) / 'cache'
    manifest = json.loads((cache / 'manifest.json').read_text())
    arrays = {k: np.load(cache / manifest.get('arrays', {}).get(k, k + '.npy'), mmap_mode='r')
              for m in MODALITIES for k in (m + '_u', m + '_s', m + '_e')}
    if any(len(x) != len(manifest['rows']) for x in arrays.values()):
        raise ValueError('cache row identity/array shape mismatch')
    if any(not np.allclose(arrays[m + '_s'] + arrays[m + '_e'], arrays[m + '_u'], atol=2e-5)
           for m in MODALITIES):
        raise ValueError('cached teacher decomposition identity failed')
    return cache, manifest, arrays


def _region(row):
    return str(row.get('roi', row.get('region', row.get('anchor', {}).get('region', ''))))


def normalized_roughness(x, channels=1):
    """Per-channel mean squared 100 ms difference / variance; patch-local."""
    x = np.asarray(x, float).reshape(*np.asarray(x).shape[:-1], -1, channels)
    axes = tuple(range(x.ndim - 1))
    return np.mean(np.diff(x, axis=-2) ** 2, axis=axes) / np.maximum(np.var(x, axis=axes), 1e-12)


def fit_matched_smoothing(observed, physical, training, sigmas=(0., 1., 2., 4., 8.), *, channels=1):
    """Choose each channel's smoothing width using training roughness alone."""
    observed, physical = np.asarray(observed), np.asarray(physical)
    training = np.asarray(training, bool)
    if observed.shape != physical.shape or len(training) != len(observed) or not training.any():
        raise ValueError('same-support observations/teacher and training mask required')
    sigmas = tuple(float(s) for s in sigmas)
    if not sigmas or min(sigmas) < 0:
        raise ValueError('nonnegative candidate Gaussian widths required')
    desired = normalized_roughness(physical[training], channels)
    reshaped = observed.reshape(*observed.shape[:-1], -1, channels)
    candidates = [reshaped if s == 0 else gaussian_filter1d(reshaped, s, axis=-2, mode='nearest')
                  for s in sigmas]
    roughness = np.stack([normalized_roughness(x[training].reshape(observed[training].shape), channels)
                          for x in candidates])
    mismatch = np.abs(np.log(np.maximum(roughness, 1e-12)) - np.log(np.maximum(desired, 1e-12)))
    selected = np.argmin(mismatch, axis=0)
    result = np.stack([candidates[selected[c]][..., c] for c in range(channels)], axis=-1)
    report = dict(sigmas_samples=[sigmas[j] for j in selected],
        sigmas_seconds=[.1 * sigmas[j] for j in selected], candidate_sigmas_samples=sigmas,
        teacher_training_roughness=desired, selected_training_roughness=roughness[selected, np.arange(channels)],
        selection='train-only per-channel absolute log normalized-roughness mismatch',
        support='within each 2-second patch; nearest endpoint; no cross-patch smoothing',
        fit_rows=int(training.sum()), eval_used_for_selection=False)
    return result.reshape(observed.shape).astype('float32'), report


def fit_smooth_targets(cfg, cache_manifest, cache_dir):
    """Return a manifest for fixed matched-smoothing S and its full residual."""
    cache = Path(cache_dir); rows = cache_manifest['rows']
    train = np.array([r['split'] == 'train' for r in rows])
    smoothing = cfg.get('smoothing', cfg.get('analysis', {}).get('smoothing', {}))
    sigmas = smoothing.get('sigma_steps', smoothing.get('sigmas_samples', smoothing.get('sigmas', [0., 1., 2., 4., 8.])))
    result = dict(schema='dual_component_matched_smoothing_v1', arrays={}, modalities={})
    for modality in MODALITIES:
        def read(part):
            key = modality + '_' + part
            return np.load(cache / cache_manifest.get('arrays', {}).get(key, key + '.npy'), mmap_mode='r')
        u, s = read('u'), read('s')
        smooth, report = fit_matched_smoothing(u, s, train, sigmas, channels=2 if modality == 'hb' else 1)
        for name, data in (('s', smooth), ('e', np.asarray(u) - smooth)):
            filename = 'smooth_' + modality + '_' + name + '.npy'
            path = cache / filename
            if path.exists():
                existing = np.load(path, mmap_mode='r')
                if existing.shape != data.shape or not np.array_equal(existing, data):
                    raise ValueError('retained smoothing target changed; use a fresh run namespace')
            else:
                np.save(path, data.astype('float32'))
            result['arrays'][modality + '_' + name] = filename
        result['modalities'][modality] = report
    _json(cache / 'smoothing.json', result)
    return result


def reconstruction_rows(predictions, physical, remaining, observed, rows, *, variant, seed, target_family='Tak_teacher'):
    """Window scores; all three denominators are observation-training SD squared."""
    result = []
    for m in MODALITIES:
        targets = (dict(sum=observed[m]) if variant == 'ordinary_hard'
                   else dict(S=physical[m], O=remaining[m], sum=observed[m]))
        for branch, target in targets.items():
            value = np.asarray(predictions[m + '_' + branch], float)
            target = np.asarray(target, float)
            if value.shape != target.shape or len(value) != len(rows):
                raise ValueError('predictions/targets/identities must share complete support')
            selectors = [('EEG_proxy' if m == 'eeg' else 'Hb_joint', slice(None))]
            if m == 'hb': selectors.extend([('HbO', slice(0, None, 2)), ('HbR', slice(1, None, 2))])
            for signal, selector in selectors:
                difference = value[..., selector] - target[..., selector]
                finite = np.isfinite(difference).all(axis=tuple(range(1, value.ndim)))
                errors = np.mean(difference ** 2, axis=tuple(range(1, value.ndim)))
                for i, row in enumerate(rows):
                    result.append(dict(dataset=row.get('dataset', ''), subject=row['subject'],
                        record=row.get('record', row.get('join_key', '')), split=row['split'],
                        variant=variant, seed=int(seed), modality=m, component=branch, signal=signal,
                        target_family='processed_observation' if branch == 'sum' else target_family,
                        NMSE=float(errors[i]) if finite[i] else None,
                        windows=1, failures=int(not finite[i]),
                        branch_semantics=('arbitrary_additive_streams' if variant == 'ordinary_hard'
                                          else 'matched_smoothing' if variant == 'smooth_hard' else 'Tak_teacher')))
    return result


def subject_aggregate(rows, *, value='NMSE'):
    """Windows -> equal records -> subject; retain failures and observed counts."""
    excluded = {'subject', 'record', value, 'windows', 'failures'}
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple((k, str(v)) for k, v in row.items() if k not in excluded)
        groups[(key, str(row['subject']))][str(row.get('record', ''))].append(row)
    result = []
    for (key, subject), records in groups.items():
        means = []
        total, failures = 0, 0
        for record_rows in records.values():
            values = [float(r[value]) for r in record_rows if r.get(value) is not None and np.isfinite(r[value])]
            total += len(record_rows)
            failures += sum(int(r.get('failures', r.get(value) is None)) for r in record_rows)
            if values: means.append(float(np.mean(values)))
        result.append(dict(key, subject=subject, **{value:float(np.mean(means)) if means else None},
            records=len(records), windows=total, failures=failures))
    return result


def paired_subject_bootstrap(first, second, *, value='NMSE', samples=2000, seed=0):
    """Average seeds within subjects, then paired subject bootstrap; first-second."""
    def reduce(rows):
        values = defaultdict(list)
        for row in rows:
            x = row.get(value)
            if x is not None and np.isfinite(x):
                values[(str(row.get('dataset', '')), str(row['subject']))].append(float(x))
        return {key:float(np.mean(x)) for key, x in values.items()}
    a, b = reduce(first), reduce(second)
    common = sorted(a.keys() & b.keys())
    difference = np.array([a[s] - b[s] for s in common])
    result = dict(subjects=len(common), first_subjects=len(a), second_subjects=len(b),
        unmatched_first=len(a.keys() - b.keys()), unmatched_second=len(b.keys() - a.keys()),
        effect='first minus second; lower NMSE favors first',
        mean_difference=float(difference.mean()) if len(difference) else None,
        interval_type='percentile paired-subject bootstrap, seeds averaged within subject',
        bootstrap_samples=int(samples), CI95=None,
        inference='exploratory; no multiplicity correction; windows/seeds are not independent subjects')
    if len(common) >= 2 and samples >= 1:
        rng = np.random.default_rng(seed)
        draws = difference[rng.integers(len(common), size=(samples, len(common)))].mean(axis=1)
        result['CI95'] = np.quantile(draws, [.025, .975]).tolist()
    return result


def latent_codebook_health(ids, codebook, rows, thresholds):
    """Full-population, per-dataset usage; geometry is latent, never waveform."""
    ids, book = np.asarray(ids), np.asarray(codebook, dtype=float)
    if (ids.ndim != 2 or ids.dtype.kind not in 'iu' or len(ids) != len(rows)
            or book.ndim != 2 or not np.isfinite(book).all()
            or np.any(ids < 0) or np.any(ids >= len(book))
            or any(r['split'] not in ('train', 'validation') for r in rows)):
        raise ValueError('finite latent codebook and complete development integer assignments required')
    centered = book - book.mean(0)
    eigen = np.linalg.eigvalsh(centered.T @ centered / max(len(book), 1)).clip(0)
    rank = float(eigen.sum() ** 2 / max(float(np.square(eigen).sum()), 1e-30))
    distances = np.maximum(np.square(book).sum(1)[:, None] + np.square(book).sum(1)[None] - 2 * book @ book.T, 0)
    np.fill_diagonal(distances, np.inf)
    nearest = np.sqrt(distances.min(1))
    scale = max(float(np.sqrt(np.mean(np.square(centered).sum(1)))), 1e-12)
    geometry = dict(codebook_participation_rank=rank, dimension=book.shape[1],
        nearest_distance_median_over_codebook_RMS=float(np.median(nearest) / scale),
        near_duplicate_fraction=float(np.mean(nearest / scale < .01)),
        geometry='centered latent covariance participation rank; distances are not decoded waveform distances')
    summaries, coverage = [], []
    for dataset in ['pooled', *sorted({r['dataset'] for r in rows})]:
        for split in ('train', 'validation'):
            selected = np.array([r['split'] == split and (dataset == 'pooled' or r['dataset'] == dataset) for r in rows])
            if not selected.any():
                continue
            selected_ids = ids[selected]
            counts = np.bincount(selected_ids.ravel(), minlength=len(book))
            probability = counts / counts.sum()
            entropy = -float(np.sum(probability[probability > 0] * np.log(probability[probability > 0])))
            active, effective = int(np.sum(counts > 0)), float(np.exp(entropy))
            active_floor = thresholds['train_active_fraction' if split == 'train' else 'validation_active_fraction']
            checks = dict(active=active / len(book) >= active_floor,
                effective=effective / len(book) >= thresholds['effective_fraction'],
                concentration=float(probability.max()) <= thresholds['maximum_code_probability'])
            summaries.append(dict(dataset=dataset, split=split, patches=int(counts.sum()),
                windows=int(selected.sum()), subjects=len({(r['dataset'], r['subject']) for i, r in enumerate(rows) if selected[i]}),
                active_codes=active, codebook_size=len(book), active_fraction=active / len(book),
                effective_codes=effective, effective_fraction=effective / len(book),
                maximum_code_probability=float(probability.max()), checks=checks, passed=all(checks.values())))
            selected_rows = [r for i, r in enumerate(rows) if selected[i]]
            for k in range(len(book)):
                member = np.flatnonzero(np.any(selected_ids == k, axis=1))
                coverage.append(dict(dataset=dataset, split=split, code=k, assignments=int(counts[k]),
                    probability=float(probability[k]), windows=len(member),
                    subjects=len({(selected_rows[i]['dataset'], selected_rows[i]['subject']) for i in member}),
                    records=len({(selected_rows[i]['dataset'], selected_rows[i]['subject'], selected_rows[i]['record']) for i in member})))
    return dict(passed=bool(summaries) and all(r['passed'] for r in summaries), summaries=summaries,
                geometry=geometry, coverage=coverage)


def latent_hb_fidelity(predictions, targets, rows):
    """Separate Hb colors and branches using the retained observation-SD units."""
    window_metrics = []
    for branch in ('S', 'O', 'sum'):
        truth = np.asarray(targets['u' if branch == 'sum' else branch], dtype=float)
        estimate = np.asarray(predictions[branch], dtype=float)
        if (truth.shape != estimate.shape or truth.shape != (len(rows), 15, 40)
                or not np.isfinite(truth).all() or not np.isfinite(estimate).all()):
            raise ValueError('complete aligned [window,15,40] Hb targets and predictions required')
        a, b = estimate.reshape(-1, 300, 2), truth.reshape(-1, 300, 2)
        for signal, selector in [('HbO', [0]), ('HbR', [1]), ('joint', [0, 1])]:
            aa, bb = a[:, :, selector], b[:, :, selector]
            ac, bc = aa - aa.mean(1, keepdims=True), bb - bb.mean(1, keepdims=True)
            denominator = np.sqrt(np.square(ac).sum((1, 2)) * np.square(bc).sum((1, 2)))
            values = dict(NMSE=np.square(aa - bb).mean((1, 2)),
                zero_NMSE=np.square(bb).mean((1, 2)),
                centered_correlation=np.divide((ac * bc).sum((1, 2)), denominator,
                    out=np.full(len(rows), np.nan), where=denominator > 1e-12))
            for metric, array in values.items():
                for i, row in enumerate(rows):
                    window_metrics.append(dict(dataset=row['dataset'], split=row['split'], subject=row['subject'],
                        record=row['record'], branch=branch, signal=signal, metric=metric,
                        value=float(array[i]) if np.isfinite(array[i]) else None, failures=int(not np.isfinite(array[i]))))
    subjects = subject_aggregate(window_metrics, value='value')
    grouped = defaultdict(list)
    for row in subjects:
        grouped[tuple(row[k] for k in ('dataset', 'split', 'branch', 'signal', 'metric'))].append(row)
    summary = []
    for key, group in grouped.items():
        values = [r['value'] for r in group if r['value'] is not None]
        summary.append(dict(zip(('dataset', 'split', 'branch', 'signal', 'metric'), key),
            value=float(np.mean(values)) if values else None, subjects=len(values),
            undefined_subjects=len(group) - len(values), undefined_windows=sum(r['failures'] for r in group)))
    return dict(metrics=summary, subjects=subjects,
        units='MSE after division by frozen training observation SD; no window normalization',
        aggregation='time/color then equal windows within record then equal records within subject then equal subjects')


def codebook_health(ids, prototypes, rows, *, modality, branch, variant, seed, observed=None):
    """Split-specific occupancy and independent subject/record coverage per code."""
    ids, prototypes = np.asarray(ids), np.asarray(prototypes)
    if ids.ndim != 2 or prototypes.ndim != 2 or len(ids) != len(rows):
        raise ValueError('codebook health requires [window,time] IDs and [code,waveform] prototypes')
    if ids.dtype.kind not in 'iu': raise ValueError('hard integer IDs required')
    result = []
    for split in ('train', 'validation', 'test'):
        selected = np.array([r['split'] == split for r in rows])
        split_ids = ids[selected]
        valid = split_ids[(split_ids >= 0) & (split_ids < len(prototypes))]
        counts = np.bincount(valid, minlength=len(prototypes))
        p = counts / max(int(counts.sum()), 1)
        entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
        observed_rms = np.sqrt(np.mean(np.asarray(observed, float) ** 2, axis=-1)) if observed is not None else None
        for k, prototype in enumerate(prototypes):
            assignment_mask = selected[:, None] & (ids == k)
            amplitudes = observed_rms[assignment_mask] if observed_rms is not None else np.array([])
            assigned = np.flatnonzero(selected & np.any(ids == k, axis=1))
            result.append(dict(variant=variant, seed=seed, modality=modality, component=branch,
                split=split, code=k, assignments=int(counts[k]), probability=float(p[k]),
                assigned_windows=len(assigned), subjects=len({(rows[i].get('dataset', ''), rows[i]['subject']) for i in assigned}),
                records=len({(rows[i].get('dataset', ''), rows[i]['subject'], rows[i].get('record', rows[i].get('join_key', ''))) for i in assigned}),
                assigned_observation_RMS_P10=float(np.quantile(amplitudes, .1)) if len(amplitudes) else None,
                assigned_observation_RMS_P50=float(np.quantile(amplitudes, .5)) if len(amplitudes) else None,
                assigned_observation_RMS_P90=float(np.quantile(amplitudes, .9)) if len(amplitudes) else None,
                prototype_mean=float(prototype.mean()), prototype_RMS=float(np.sqrt(np.mean(prototype ** 2))),
                prototype_min=float(prototype.min()), prototype_max=float(prototype.max()),
                active_codes=int((counts > 0).sum()), perplexity=float(np.exp(entropy)),
                dead_fraction=float(np.mean(counts == 0))))
    return result


def align_codewords(reference, candidate):
    """Training-frozen waveform correspondence, not equality of arbitrary IDs."""
    a, b = np.asarray(reference, float), np.asarray(candidate, float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError('same code count and decoded waveform dimensions required')
    cost = np.mean((a[:, None] - b[None]) ** 2, axis=-1)
    i, j = linear_sum_assignment(cost)
    mapping = np.empty(len(b), dtype=int); mapping[j] = i
    return mapping, dict(mean_prototype_NMSE=float(cost[i, j].mean()),
                        P90_prototype_NMSE=float(np.quantile(cost[i, j], .9)))


def _job(cfg, run, variant, seed):
    return Path(run) / 'jobs' / variant / ('seed_' + str(seed))


def _subject_plot(path, rows):
    if not rows: return
    import matplotlib; matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    selected = [r for r in rows if r['split'] == 'test' and r['component'] == 'sum' and r['NMSE'] is not None and r.get('signal') in ('EEG_proxy', 'Hb_joint')]
    variants = list(dict.fromkeys(r['variant'] for r in selected))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, m in zip(axes, MODALITIES):
        for j, variant in enumerate(variants):
            values = [r['NMSE'] for r in selected if r['variant'] == variant and r['modality'] == m]
            if values: ax.scatter(np.full(len(values), j), values, s=14, alpha=.45)
        ax.set(xticks=np.arange(len(variants)), xticklabels=variants,
               ylabel='Observation MSE / frozen training SD²', title=m + ': subject-equal record means')
        ax.tick_params(axis='x', labelrotation=35); ax.grid(alpha=.2)
    fig.savefig(path, dpi=180); plt.close(fig)


def analyze_suite(cfg, run):
    """Read only this run's explicit completed job exports; write shared tables."""
    run = Path(run); cache, manifest, arrays = _cache(run); rows = manifest['rows']
    settings = cfg.get('analysis', {})
    variants = cfg.get('variants', cfg.get('training', {}).get('variants', ['continuous_student', 'hard_student',
                'component_compression', 'smooth_hard', 'ordinary_hard']))
    seeds = cfg.get('seeds', cfg.get('training', {}).get('seeds', [17, 29, 43]))
    physical = {m:arrays[m + '_s'] for m in MODALITIES}
    remaining = {m:arrays[m + '_e'] for m in MODALITIES}
    observed = {m:arrays[m + '_u'] for m in MODALITIES}
    subjects, health, alignment, missing = [], [], [], []
    reference = {}
    for variant in variants:
        for seed in seeds:
            job = _job(cfg, run, variant, seed)
            record = json.loads((job / 'summary.json').read_text()) if (job / 'summary.json').exists() else {}
            if (record.get('status') != 'completed' or record.get('step') != int(cfg.get('training', {}).get('steps', 3000))
                    or not (job / 'predictions.npz').exists()):
                missing.append(dict(variant=variant, seed=seed, reason='completed_fixed_budget_job_and_prediction_export_required')); continue
            with np.load(job / 'predictions.npz', allow_pickle=False) as data:
                predictions = {key:data[key] for key in data.files}
            component_physical, component_remaining = physical, remaining
            target_family = 'Tak_teacher'
            if variant == 'smooth_hard':
                smoothing = json.loads((cache / 'smoothing.json').read_text())
                component_physical = {m:np.load(cache / smoothing['arrays'][m + '_s'], mmap_mode='r') for m in MODALITIES}
                component_remaining = {m:np.load(cache / smoothing['arrays'][m + '_e'], mmap_mode='r') for m in MODALITIES}
                if any(not np.allclose(component_physical[m] + component_remaining[m], observed[m], atol=2e-5) for m in MODALITIES):
                    raise ValueError('matched smoothing targets must sum to the same processed observation')
                target_family = 'matched_smoothing'
            windows = reconstruction_rows(predictions, component_physical, component_remaining, observed, rows,
                                           variant=variant, seed=seed, target_family=target_family)
            subjects.extend(subject_aggregate(windows))
            if (job / 'codewords.npz').exists():
                with np.load(job / 'codewords.npz', allow_pickle=False) as data:
                    codewords = {key:data[key] for key in data.files}
                for m in MODALITIES:
                    for branch in BRANCHES:
                        key = m + '_' + branch; ids_key = m + '_ids_' + branch
                        if key not in codewords or ids_key not in predictions: continue
                        health.extend(codebook_health(predictions[ids_key], codewords[key], rows,
                            modality=m, branch=branch, variant=variant, seed=seed, observed=observed[m]))
                        rkey = (variant, m, branch)
                        if rkey not in reference:
                            reference[rkey] = (seed, codewords[key], predictions[ids_key])
                        else:
                            ref_seed, ref_book, ref_ids = reference[rkey]
                            mapping, effect = align_codewords(ref_book, codewords[key])
                            current = predictions[ids_key]; valid = (current >= 0) & (ref_ids >= 0)
                            aligned = mapping[np.maximum(current, 0)]
                            for split in ('train', 'validation', 'test'):
                                selected = np.array([r['split'] == split for r in rows])[:, None] & valid
                                alignment.append(dict(variant=variant, modality=m, component=branch,
                                    reference_seed=ref_seed, seed=seed, split=split, **effect,
                                    aligned_agreement=float(np.mean(aligned[selected] == ref_ids[selected])) if selected.any() else None,
                                    matching='Hungarian distance of fixed decoded waveforms; no eval fitting'))
    # Simple fixed-output controls are fit on train only and evaluated everywhere.
    train = np.array([r['split'] == 'train' for r in rows])
    if train.any():
        for baseline in ('training_mean', 'zero'):
            predictions = {}
            for m in MODALITIES:
                for branch, values in (('S', physical[m]), ('O', remaining[m])):
                    center = np.mean(values[train], axis=(0, 1), keepdims=True) if baseline == 'training_mean' else np.zeros((1, 1, values.shape[-1]))
                    predictions[m + '_' + branch] = np.broadcast_to(center, values.shape)
                predictions[m + '_sum'] = predictions[m + '_S'] + predictions[m + '_O']
            subjects.extend(subject_aggregate(reconstruction_rows(predictions, physical, remaining, observed,
                rows, variant=baseline, seed=0)))
    comparisons = []
    for dataset in sorted({r.get('dataset', '') for r in rows}):
        for split in ('validation', 'test'):
            for m, signal in (('eeg', 'EEG_proxy'), ('hb', 'Hb_joint'), ('hb', 'HbO'), ('hb', 'HbR')):
                for branch in ('S', 'O', 'sum'):
                    def pick(variant):
                        return [r for r in subjects if r['variant'] == variant and r['split'] == split
                                and r['modality'] == m and r['component'] == branch and r['signal'] == signal
                                and r.get('dataset', '') == dataset]
                    for control in ('continuous_student', 'component_compression', 'smooth_hard', 'ordinary_hard', 'training_mean'):
                        if control in ('smooth_hard', 'ordinary_hard') and branch != 'sum': continue
                        effect = paired_subject_bootstrap(pick('hard_student'), pick(control),
                            samples=int(settings.get('bootstrap_samples', cfg.get('evaluation', {}).get('bootstrap_repeats', 1000))), seed=int(cfg.get('seed', 0)))
                        comparisons.append(dict(first='hard_student', second=control, dataset=dataset, split=split,
                            modality=m, component=branch, signal=signal,
                            target_family='processed_observation' if branch == 'sum' else 'Tak_teacher', **effect))
    output = run / 'analysis'; output.mkdir(exist_ok=True)
    _csv(output / 'reconstruction_subjects.csv', subjects)
    _csv(output / 'codebook_health.csv', health)
    _csv(output / 'seed_alignment.csv', alignment)
    _csv(output / 'paired_comparisons.csv', comparisons)
    _subject_plot(output / 'reconstruction_subjects.png', subjects)
    sensitivity = analyze_teacher_sensitivity(cfg, run)
    quality = vocabulary_quality_screen(subjects, cfg)
    masked = summarize_masked_pilots(cfg, run)
    summary = dict(schema='dual_component_analysis_v1', status='completed' if not missing else 'partial',
        missing_jobs=missing, reconstruction_subject_rows=len(subjects), codebook_rows=len(health),
        paired_comparisons=comparisons, teacher_sensitivity=sensitivity, quality_screen=quality, masked=masked,
        physical_compatibility=dict(status='not_evaluated',
            reason='No Tak-compatible whole-trajectory witness was evaluated for decoded masked predictions; 20-second patches retain the 90-second teacher reference coordinate'),
        planned_windows=manifest.get('planned_windows', len(rows)), retained_windows=len(rows),
        failed_trials=manifest.get('failed_trials', []),
        metric='mean squared error in shared frozen observation-training-SD coordinates; no per-window scaling',
        aggregation='equal windows within record; equal records within subject; seeds averaged before paired subject bootstrap',
        scope='processed-observation offline completion; full-record Hb preprocessing retained',
        cross_vocabulary_CE_ranking_allowed=False)
    _json(output / 'summary.json', summary)
    return summary


def _checkpoint(path, model, optimizer, step, rng, losses, signature):
    payload = dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
        step=step, numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
        losses=losses, signature=signature, cuda_rng=torch.cuda.get_rng_state(next(model.parameters()).device) if next(model.parameters()).is_cuda else None)
    tmp = path.with_suffix('.tmp'); torch.save(payload, tmp); tmp.replace(path)


def _fit_reader(ids, rows, config, output, *, device, seed, reader):
    """Fixed-exposure train-only fitting, resumable independently per reader."""
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = DualComponentMaskedModel({m:int(config['vocab_size']) for m in MODALITIES},
        patches=ids.shape[1], width=int(config.get('width', 128)), layers=int(config.get('layers', 2)),
        heads=int(config.get('heads', 4)), dropout=float(config.get('dropout', .1))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get('learning_rate', 3e-4)),
                                  weight_decay=float(config.get('weight_decay', 1e-4)))
    rng = np.random.default_rng(seed); start = 0; losses = []
    path = output / 'checkpoint.pt'
    signature = dict(reader=reader, seed=seed, config=config, id_shape=ids.shape,
        training_identities=[r.get('id', str(i)) for i, r in enumerate(rows) if r['split'] == 'train'])
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved.get('signature') != signature or saved['step'] > int(config.get('steps', 1000)):
            raise ValueError('masked checkpoint identity/config/reader/budget mismatch')
        model.load_state_dict(saved['model_state']); optimizer.load_state_dict(saved['optimizer_state'])
        start = int(saved['step']); rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu()); losses = saved.get('losses', [])
        if saved.get('cuda_rng') is not None and str(device).startswith('cuda'):
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), device)
    datasets = sorted({r.get('dataset', '') for r in rows if r['split'] == 'train'})
    groups = [np.array([i for i, r in enumerate(rows) if r['split'] == 'train' and r.get('dataset', '') == d]) for d in datasets]
    if not groups or any(not len(g) for g in groups): raise ValueError('train data missing')
    steps = int(config.get('steps', 1000)); batch_size = int(config.get('batch_size', 64))
    block = int(config.get('block_patches', 2)); checkpoint_every = int(config.get('checkpoint_every', 250))
    began = time.monotonic(); model.train()
    for step in range(start, steps):
        selected = rng.choice(groups[int(rng.integers(len(groups)))], batch_size, replace=True)
        batch = ids[selected]; target_modality = int(rng.integers(len(MODALITIES)))
        task = 'time' if rng.integers(2) == 0 else 'modality'
        visible, target = block_masks(batch >= 0, rng=rng, target_modality=target_modality,
                                      block_patches=block, task=task)
        if reader == 'own': visible = own_context(visible, target_modality)
        truth = torch.as_tensor(batch, dtype=torch.long, device=device)
        target = torch.as_tensor(target, device=device)
        logits = model(truth, torch.as_tensor(visible, device=device))
        branch_losses = []
        for c in range(2):
            keep = target[..., c]
            if keep.any(): branch_losses.append(F.cross_entropy(logits[..., c, :][keep], truth[..., c][keep]))
        if not branch_losses: raise ValueError('empty masked target support')
        loss = torch.stack(branch_losses).sum()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get('gradient_clip', 5.)))
        optimizer.step()
        if (step + 1) % 50 == 0 or step == 0:
            losses.append(dict(step=step + 1, loss=float(loss.detach()), task=task, target_modality=MODALITIES[target_modality]))
        if (step + 1) % checkpoint_every == 0 or step + 1 == steps:
            _checkpoint(path, model, optimizer, step + 1, rng, losses, signature)
            _json(output / 'progress.json', dict(step=step + 1, steps=steps, reader=reader,
                last_loss=float(loss.detach()), resumed_from=start, elapsed_s=time.monotonic() - began))
    _csv(output / 'training_loss.csv', losses)
    model.eval()
    return model


def _predict(model, ids, visible, *, device, batch_size=128):
    result = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            logits = model(torch.as_tensor(ids[start:start + batch_size], device=device, dtype=torch.long),
                           torch.as_tensor(visible[start:start + batch_size], device=device))
            result.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(result)


def matched_donor_tokens(ids, rows, decoded, *, source_modality, training_ids,
                         training_rows, seed=0):
    """Same dataset/ROI/time, training-amplitude-bin donor, different subject.

    Matching uses only source token IDs and fixed decoded vocabulary, never the
    target, teacher, task/condition labels or eval-fitted amplitude thresholds.
    Donors come from the same evaluation partition and preserve patch index.
    """
    rng = np.random.default_rng(seed); m = int(source_modality)
    def amplitude(x):
        waves = sum(decoded[(m, c)][np.maximum(x[:, :, m, c], 0)] for c in range(2))
        return np.sqrt(np.mean(waves ** 2, axis=-1))
    training_amp = amplitude(training_ids)
    cutoffs = np.quantile(training_amp, [1 / 3, 2 / 3])
    amplitude_bins = np.searchsorted(cutoffs, amplitude(ids))
    result = ids.copy(); available = np.zeros(ids.shape[:2], bool)
    pools = defaultdict(list)
    for j, row in enumerate(rows):
        for t in range(ids.shape[1]):
            if np.all(ids[j, t, m] >= 0):
                pools[(str(row.get('dataset', '')), str(row.get('split', '')), _region(row), t, int(amplitude_bins[j, t]))].append(j)
    for i, row in enumerate(rows):
        for t in range(ids.shape[1]):
            key = (str(row.get('dataset', '')), str(row.get('split', '')), _region(row), t, int(amplitude_bins[i, t]))
            pool = pools.get(key, [])
            donor = None
            # Usually succeeds immediately, while remaining bounded and exact.
            for _ in range(min(8, len(pool))):
                candidate = pool[int(rng.integers(len(pool)))]
                if rows[candidate]['subject'] != row['subject']:
                    donor = candidate; break
            if donor is None:
                candidates = [j for j in pool if rows[j]['subject'] != row['subject']]
                if candidates: donor = int(rng.choice(candidates))
            if donor is not None:
                result[i, t, m] = ids[donor, t, m]
                available[i, t] = True
    return result, available, dict(amplitude_cutoffs_training_SD=cutoffs,
        matching='dataset, ROI, patch phase and train-fixed source decoded-amplitude tertile; other subject',
        eligible_source_positions=int(available.sum()), planned_source_positions=int(available.size),
        missing_donor='source tokens hidden; target scores retained', task_labels_used=False)


def _metric_subjects(window_rows):
    """One row per subject and metric, preserving trial/record dependence."""
    return subject_aggregate(window_rows, value='value')


def _score_probabilities(probability, ids, target, rows, decoded, arrays, marginal, metadata,
                         *, route, seed, task, context, target_modality):
    m = target_modality; modality = MODALITIES[m]; result = []
    predicted = {c:expected_waveforms(probability[:, :, m, c], decoded[(m, c)]) for c in range(2)}
    oracle = {c:decoded[(m, c)][np.maximum(ids[:, :, m, c], 0)] for c in range(2)}
    branch_targets = {0:arrays[modality + '_s'], 1:arrays[modality + '_e']}
    def add(values, selected, component, metric):
        for i, row in enumerate(rows):
            keep = selected[i]
            if not keep.any(): continue
            value = values[i][keep]
            result.append(dict(dataset=row.get('dataset', ''), subject=row['subject'],
                record=row.get('record', row.get('join_key', '')), split=row['split'],
                variant=route, seed=seed, task=task, context=context, modality=modality,
                component=component, metric=metric,
                target_family=('processed_observation' if component == 'sum' else 'arbitrary_additive_streams' if route == 'ordinary_hard' else 'matched_smoothing' if route == 'smooth_hard' else 'Tak_teacher'),
                value=float(np.mean(value)) if np.isfinite(value).all() else None,
                failures=int(not np.isfinite(value).all())))
    for c, branch in enumerate(BRANCHES):
        selected = target[:, :, m, c]
        truth = np.maximum(ids[:, :, m, c], 0)[..., None]
        def ce(p): return -np.log(np.maximum(np.take_along_axis(p[:, :, m, c], truth, axis=-1)[..., 0], 1e-12))
        error, marginal_error, metadata_error = ce(probability), ce(marginal), ce(metadata)
        add(error, selected, branch, 'CE_nats')
        entropy = -np.sum(probability[:, :, m, c] * np.log(np.maximum(probability[:, :, m, c], 1e-12)), axis=-1)
        add(entropy, selected, branch, 'predictive_entropy_nats')
        add(marginal_error - error, selected, branch, 'DeltaCE_marginal_nats')
        add(metadata_error - error, selected, branch, 'DeltaCE_metadata_nats')
        add(np.mean((predicted[c] - oracle[c]) ** 2, axis=-1), selected, branch, 'model_to_tokenizer_NMSE')
        if route != 'ordinary_hard':
            add(np.mean((predicted[c] - branch_targets[c]) ** 2, axis=-1), selected, branch, 'NMSE')
            add(np.mean((oracle[c] - branch_targets[c]) ** 2, axis=-1), selected, branch, 'tokenizer_NMSE')
            if modality == 'hb':
                for channel, name in enumerate(('HbO', 'HbR')):
                    add(np.mean((predicted[c][..., channel::2] - branch_targets[c][..., channel::2]) ** 2, axis=-1), selected, branch, 'NMSE_' + name)
    selected = target[:, :, m].all(-1)
    summed = predicted[0] + predicted[1]; oracle_sum = oracle[0] + oracle[1]
    add(np.mean((summed - arrays[modality + '_u']) ** 2, axis=-1), selected, 'sum', 'NMSE')
    if modality == 'hb':
        for channel, name in enumerate(('HbO', 'HbR')):
            add(np.mean((summed[..., channel::2] - arrays[modality + '_u'][..., channel::2]) ** 2, axis=-1), selected, 'sum', 'NMSE_' + name)
    if route != 'ordinary_hard':
        channels = 2 if modality == 'hb' else 1
        assembled = np.where(target[:, :, m, 0, None], predicted[0], oracle[0])
        segments = assembled.reshape(*assembled.shape[:2], -1, channels)
        teacher_segments = branch_targets[0].reshape(segments.shape)
        signed_jumps = segments[:, 1:, 0] - segments[:, :-1, -1]
        teacher_signed_jumps = teacher_segments[:, 1:, 0] - teacher_segments[:, :-1, -1]
        jumps = np.mean(signed_jumps ** 2, axis=-1)
        true_jumps = np.mean(teacher_signed_jumps ** 2, axis=-1)
        jump_error = np.mean((signed_jumps - teacher_signed_jumps) ** 2, axis=-1)
        boundaries = target[:, 1:, m, 0] | target[:, :-1, m, 0]
        add(np.pad(jumps, ((0, 0), (1, 0))), np.pad(boundaries, ((0, 0), (1, 0))), 'S', 'boundary_jump_squared_trainingSD')
        add(np.pad(jumps - true_jumps, ((0, 0), (1, 0))), np.pad(boundaries, ((0, 0), (1, 0))), 'S', 'boundary_jump_excess_over_teacher_trainingSD')
        add(np.pad(jump_error, ((0, 0), (1, 0))), np.pad(boundaries, ((0, 0), (1, 0))), 'S', 'boundary_jump_error_MSE_trainingSD2')
    add(np.mean((oracle_sum - arrays[modality + '_u']) ** 2, axis=-1), selected, 'sum', 'tokenizer_NMSE')
    add(np.mean((summed - oracle_sum) ** 2, axis=-1), selected, 'sum', 'model_to_tokenizer_NMSE')
    return result, predicted


def _trace_plot(path, rows, arrays, oracle, predicted, target, modality, route):
    if not len(rows): return
    import matplotlib; matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    channels = 2 if modality == 'hb' else 1
    shape = (len(rows), -1, channels)
    fig, axes = plt.subplots(3, channels, figsize=(11, 7), squeeze=False, constrained_layout=True)
    time_axis = np.arange(np.asarray(arrays[modality + '_u'])[0].size // channels) * .1
    s = np.asarray(arrays[modality + '_s']).reshape(shape)
    o = np.asarray(arrays[modality + '_e']).reshape(shape)
    u = np.asarray(arrays[modality + '_u']).reshape(shape)
    mask = np.repeat(target[0], arrays[modality + '_u'].shape[-1] // channels)
    for c in range(channels):
        for a, (teacher, dec, pred, title) in enumerate(((s, oracle[0], predicted[0], 'S'),
                (o, oracle[1], predicted[1], 'O'), (u, oracle[0] + oracle[1], predicted[0] + predicted[1], 'sum'))):
            ax = axes[a, c]
            if route != 'ordinary_hard' or a == 2:
                label = ('processed observation' if a == 2 else 'matched smoothing target' if route == 'smooth_hard' else 'fixed teacher target')
                ax.plot(time_axis, teacher[0, :, c], label=label, color='black')
            ax.plot(time_axis, dec.reshape(shape)[0, :, c], label='true-ID frozen decode', color='#d98b23', ls='--')
            ax.plot(time_axis, pred.reshape(shape)[0, :, c], label='masked expected decode', color='#177db0')
            ax.fill_between(time_axis, 0, 1, where=mask, transform=ax.get_xaxis_transform(), color='#aaaaaa', alpha=.2)
            ax.set(title=f'{modality} {title}' + (' (arbitrary additive stream)' if route == 'ordinary_hard' and a != 2 else '') + (f' channel {c + 1}' if channels > 1 else ''),
                   xlabel='event time (s)', ylabel='frozen training SD')
            ax.legend(fontsize=7); ax.grid(alpha=.2)
    row = rows[0]
    fig.suptitle(f"Offline processed-observation completion | first heldout identity {row.get('id', row['subject'])}\nGrey: jointly hidden S/O blocks; teacher shown for scoring only", fontsize=9)
    fig.savefig(path, dpi=200); plt.close(fig)


def run_masked_pilot(cfg, run, *, device='cpu', seed=17, route='hard_student'):
    """Train paired/own readers, then frozen-context controls at identical masks.

    CE is interpreted only relative to this route's train-fitted baselines.
    Cross-route comparison uses the common observation-space decoded NMSE.
    """
    if route not in ('hard_student', 'smooth_hard', 'ordinary_hard'):
        raise ValueError('masked pilot requires a frozen deployable hard-ID vocabulary')
    device = str(device)
    if device.startswith('cuda'):
        budget_mib = float(cfg.get('resources', {}).get('max_new_gpu_memory_mib', 4096))
        properties = torch.cuda.get_device_properties(device)
        torch.cuda.set_per_process_memory_fraction(min(budget_mib * 1024 ** 2 / properties.total_memory, 1.), device)
        torch.cuda.reset_peak_memory_stats(device)
    run = Path(run); cache, manifest, all_arrays = _cache(run); all_rows = manifest['rows']
    config = dict(cfg.get('masked', {})); config['vocab_size'] = int(cfg['model']['codebook_size'])
    job = _job(cfg, run, route, seed)
    job_record = json.loads((job / 'summary.json').read_text()) if (job / 'summary.json').exists() else {}
    if job_record.get('status') != 'completed' or job_record.get('step') != int(cfg.get('training', {}).get('steps', 3000)):
        raise ValueError('masked reader requires a completed fixed-budget frozen tokenizer job')
    with np.load(job / 'predictions.npz', allow_pickle=False) as export:
        ids = np.stack([np.stack([export[m + '_ids_' + c] for c in BRANCHES], axis=-1) for m in MODALITIES], axis=-2)
    if ids.dtype.kind not in 'iu' or ids.shape != (len(all_rows), int(manifest.get('patches', 10)), 2, 2):
        raise ValueError('exact same-ROI [N,10,2,2] hard ID export required')
    with np.load(job / 'codewords.npz', allow_pickle=False) as books:
        decoded = {(m, c):books[modality + '_' + branch] for m, modality in enumerate(MODALITIES) for c, branch in enumerate(BRANCHES)}
    if any(len(w) != config['vocab_size'] for w in decoded.values()):
        raise ValueError('vocabulary/decoder count mismatch')
    if np.any(ids >= config['vocab_size']):
        raise ValueError('token IDs exceed frozen vocabulary')
    digest = hashlib.sha256(np.ascontiguousarray(ids).tobytes())
    for key in sorted(decoded): digest.update(np.ascontiguousarray(decoded[key]).tobytes())
    config['frozen_representation_signature'] = digest.hexdigest()
    # Only metadata keys permitted in the baseline; condition/task are ignored.
    baseline_rows = [dict(row, roi=_region(row)) for row in all_rows]
    baseline = fit_token_baselines(ids, baseline_rows, config['vocab_size'])
    output = run / 'masked' / route / ('seed_' + str(seed)); output.mkdir(parents=True, exist_ok=True)
    models = {name:_fit_reader(ids, all_rows, config, output / name, device=device,
                seed=seed, reader=name) for name in ('paired', 'own')}
    selected = np.array([r['split'] in ('validation', 'test') for r in all_rows])
    eval_ids = ids[selected]; rows = [r for i, r in enumerate(all_rows) if selected[i]]
    if not len(rows): raise ValueError('heldout scoring support missing')
    arrays = {key:np.asarray(value[selected]) for key, value in all_arrays.items()}
    if route == 'smooth_hard':
        smoothing = json.loads((cache / 'smoothing.json').read_text())
        for m in MODALITIES:
            for name in ('s', 'e'):
                arrays[m + '_' + name] = np.load(cache / smoothing['arrays'][m + '_' + name], mmap_mode='r')[selected]
    baseline_eval_rows = [dict(r, roi=_region(r)) for r in rows]
    marginal = baseline_probabilities(baseline, baseline_eval_rows)
    metadata = baseline_probabilities(baseline, baseline_eval_rows, metadata=True)
    train = np.array([r['split'] == 'train' for r in all_rows])
    scores, donor_reports = [], []
    for target_m, modality in enumerate(MODALITIES):
        source_m = 1 - target_m
        for task in ('time', 'modality'):
            visible, target = block_masks(eval_ids >= 0, rng=np.random.default_rng(seed + target_m * 10 + (task == 'modality')),
                target_modality=target_m, block_patches=int(config.get('block_patches', 2)), task=task)
            contexts = [('paired', models['paired'], eval_ids, visible),
                        ('own_reader', models['own'], eval_ids, own_context(visible, target_m)),
                        ('frozen_remove_source_modality', models['paired'], eval_ids, own_context(visible, target_m))]
            for c, branch in enumerate(BRANCHES):
                ablated = visible.copy(); ablated[:, :, source_m, c] = False
                contexts.append(('frozen_remove_source_' + branch, models['paired'], eval_ids, ablated))
            # Retain only source tokens preceding each sample's first target patch.
            past = visible.copy()
            target_time = target[:, :, target_m].any(-1)
            for i in range(len(rows)):
                locations = np.flatnonzero(target_time[i])
                if len(locations): past[i, locations[0]:] = False
            contexts.append(('frozen_past_only', models['paired'], eval_ids, past))
            donor, available, report = matched_donor_tokens(eval_ids, rows, decoded, source_modality=source_m,
                training_ids=ids[train], training_rows=[r for i, r in enumerate(all_rows) if train[i]], seed=seed)
            donor_visible = visible.copy(); donor_visible[:, :, source_m] &= available[:, :, None]
            contexts.append(('frozen_matched_donor', models['paired'], donor, donor_visible))
            donor_reports.append(dict(target_modality=modality, task=task, **report))
            # Nonwrapping time shift: vacated source tokens are hidden, never wrapped.
            shift = min(int(config.get('shift_patches', 3)), ids.shape[1] - 1)
            shifted, shifted_visible = nonwrapping_source_shift(eval_ids, visible, source_m, shift)
            contexts.append(('frozen_nonwrapping_source_shift', models['paired'], shifted, shifted_visible))
            # Explicit train-fit probability controls, with no contextual input.
            controls = [('training_marginal', marginal), ('training_metadata', metadata)]
            for name, model, context_ids, context_visible in contexts:
                probability = _predict(model, context_ids, context_visible, device=device,
                    batch_size=int(config.get('eval_batch_size', 128)))
                window_scores, predicted = _score_probabilities(probability, eval_ids, target, rows, decoded, arrays,
                    marginal, metadata, route=route, seed=seed, task=task, context=name, target_modality=target_m)
                scores.extend(_metric_subjects(window_scores))
                if name == 'paired' and task == 'modality':
                    oracle = {c:decoded[(target_m, c)][np.maximum(eval_ids[:, :, target_m, c], 0)] for c in range(2)}
                    _trace_plot(output / ('trace_' + modality + '.png'), rows, arrays, oracle, predicted,
                                target[:, :, target_m].all(-1), modality, route)
            for name, probability in controls:
                window_scores, _ = _score_probabilities(probability, eval_ids, target, rows, decoded, arrays,
                    marginal, metadata, route=route, seed=seed, task=task, context=name, target_modality=target_m)
                scores.extend(_metric_subjects(window_scores))
            # Waveform baselines see the same visible token interface as readers.
            oracle_sum = sum(decoded[(target_m, c)][np.maximum(eval_ids[:, :, target_m, c], 0)] for c in range(2))
            mean_wave = np.asarray(all_arrays[modality + '_u'][train]).mean(axis=0)
            selected_target = target[:, :, target_m].all(-1)
            for name in ('training_mean_waveform', 'linear_interpolation_visible_decodes'):
                prediction = np.broadcast_to(mean_wave, oracle_sum.shape).copy()
                if name.startswith('linear'):
                    for i in range(len(rows)):
                        known = np.flatnonzero(visible[i, :, target_m].all(-1))
                        if len(known):
                            for d in range(oracle_sum.shape[-1]):
                                prediction[i, :, d] = np.interp(np.arange(ids.shape[1]), known, oracle_sum[i, known, d])
                error = np.mean((prediction - arrays[modality + '_u']) ** 2, axis=-1)
                window_rows = [dict(dataset=r.get('dataset', ''), subject=r['subject'],
                    record=r.get('record', r.get('join_key', '')), split=r['split'], variant=route,
                    seed=seed, task=task, context=name, modality=modality, component='sum', metric='NMSE', target_family='processed_observation',
                    value=float(error[i][selected_target[i]].mean()), failures=0)
                    for i, r in enumerate(rows) if selected_target[i].any()]
                scores.extend(_metric_subjects(window_rows))
        # A same-position opposite branch remains visible: explicitly diagnostic.
        for c, branch in enumerate(BRANCHES):
            visible, target = block_masks(eval_ids >= 0, rng=np.random.default_rng(seed + 100 + target_m),
                target_modality=target_m, block_patches=int(config.get('block_patches', 2)), task='modality', single_branch=c)
            probability = _predict(models['paired'], eval_ids, visible, device=device)
            window_rows, _ = _score_probabilities(probability, eval_ids, target, rows, decoded, arrays,
                marginal, metadata, route=route, seed=seed, task='single_branch_' + branch + '_diagnostic',
                context='paired', target_modality=target_m)
            scores.extend(_metric_subjects(window_rows))
    _csv(output / 'subject_metrics.csv', scores)
    _json(output / 'donor_controls.json', donor_reports)
    summary = dict(schema='dual_component_masked_pilot_v1', status='completed', route=route, seed=seed,
        reader_steps=int(config.get('steps', 1000)), readers=['paired', 'own'],
        train_rows=int(train.sum()), evaluation_rows=len(rows), subject_metric_rows=len(scores),
        teacher_inputs=False, sample_input='hard integer IDs only, one ROI 10-patch context',
        decoded_expectation='probability-weighted individually decoded fixed codewords',
        cross_vocabulary_CE_ranking_allowed=False, main_comparison='same-observation-space sum NMSE',
        ordinary_branch_semantics='arbitrary additive streams' if route == 'ordinary_hard' else None,
        context_controls='frozen reader ablations are dependencies, not causal physiological effects',
        physical_compatibility=dict(status='not_evaluated',
            reason='Boundary jumps describe morphological continuity only; no Tak whole-trajectory compatibility witness was evaluated'),
        continuity_diagnostics='S patch-boundary jump roughness and signed-jump error MSE in training-SD squared; morphological description, not Tak reachability or compatibility',
        donor_reports=donor_reports, scope='offline processed-observation completion',
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else 0,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.startswith('cuda') else 0,
        cuda_memory_budget_mib=cfg.get('resources', {}).get('max_new_gpu_memory_mib', 4096) if device.startswith('cuda') else None,
        inference='descriptive heldout public-development evidence; no protected confirmation')
    _json(output / 'summary.json', summary)
    return summary


def vocabulary_quality_screen(subjects, cfg):
    settings = cfg.get('evaluation', {}).get('quality_screen', {})
    minimum_gain = float(settings.get('minimum_relative_gain_over_train_mean', .1))
    maximum_gap = float(settings.get('maximum_hard_minus_continuous_over_mean', .1))
    checks = []
    for dataset in sorted({r.get('dataset', '') for r in subjects}):
        for modality in MODALITIES:
            for branch in BRANCHES:
                values = {}
                for variant in ('hard_student', 'continuous_student', 'training_mean'):
                    bysubject = defaultdict(list)
                    for row in subjects:
                        if (row.get('dataset', '') == dataset and row['modality'] == modality
                                and row['component'] == branch and row.get('signal') in ('EEG_proxy', 'Hb_joint') and row['split'] == 'validation'
                                and row['variant'] == variant and row.get('NMSE') is not None):
                            bysubject[row['subject']].append(row['NMSE'])
                    values[variant] = float(np.mean([np.mean(x) for x in bysubject.values()])) if bysubject else None
                hard, continuous, mean = (values[k] for k in ('hard_student', 'continuous_student', 'training_mean'))
                available = all(x is not None and np.isfinite(x) for x in (hard, continuous, mean)) and mean > 0
                gain = 1 - hard / mean if available else None
                gap = (hard - continuous) / mean if available else None
                checks.append(dict(dataset=dataset, modality=modality, component=branch, validation_NMSE=values,
                    relative_gain_over_train_mean=gain, hard_minus_continuous_over_mean=gap,
                    passed=bool(available and gain >= minimum_gain and gap <= maximum_gap)))
    passed = bool(checks and all(x['passed'] for x in checks))
    return dict(passed=passed, status='passed' if passed else 'diagnostic_only', checks=checks,
        thresholds=dict(minimum_relative_gain_over_train_mean=minimum_gain,
                        maximum_hard_minus_continuous_over_mean=maximum_gap),
        on_failure='continue fixed small masked pilot diagnostic-only; no scale expansion',
        selection_split='validation only; no test-based training or selection')


def analyze_teacher_sensitivity(cfg, run):
    """Paired equal-budget training-identity sensitivity, never a truth audit."""
    root = Path(run) / 'sensitivity'
    declared = json.loads((root / 'summary.json').read_text()) if (root / 'summary.json').exists() else {}
    manifest_path = root / 'manifest.json'
    seed = int(cfg.get('sensitivity', {}).get('seed', 17))
    dirs = {name:root / name / ('seed_' + str(seed)) for name in ('reference', 'alternative')}
    expected_steps = int(cfg.get('sensitivity', {}).get('paired_steps', 500))
    arm_records = [json.loads((path / 'summary.json').read_text()) if (path / 'summary.json').exists() else {} for path in dirs.values()]
    ready = all(record.get('status') == 'completed' and record.get('step') == expected_steps for record in arm_records)
    if not manifest_path.exists() or not ready or any(not (path / 'codewords.npz').exists() for path in dirs.values()):
        return dict(status='unavailable', reason=declared.get('reason', 'No complete legal near-equivalent whole-trajectory witness and paired equal-budget retraining exports'),
                    finite_search=declared, physiological_truth_claim=False)
    manifest = json.loads(manifest_path.read_text()); rows = manifest['rows']
    exports, codewords = {}, {}
    for name, path in dirs.items():
        with np.load(path / 'predictions.npz', allow_pickle=False) as data:
            exports[name] = {key:data[key] for key in data.files}
        with np.load(path / 'codewords.npz', allow_pickle=False) as data:
            codewords[name] = {key:data[key] for key in data.files}
    results = []
    for modality in MODALITIES:
        for branch in BRANCHES:
            key = modality + '_' + branch
            mapping, effect = align_codewords(codewords['reference'][key], codewords['alternative'][key])
            ref = exports['reference'][modality + '_ids_' + branch]
            alt = exports['alternative'][modality + '_ids_' + branch]
            valid = (ref >= 0) & (alt >= 0)
            delta = np.mean((exports['reference'][key] - exports['alternative'][key]) ** 2, axis=-1)
            for dataset in sorted({r.get('dataset', '') for r in rows}):
                select = np.array([r.get('dataset', '') == dataset for r in rows])
                keep = select[:, None] & valid
                results.append(dict(dataset=dataset, modality=modality, component=branch, **effect,
                    assigned_decoded_waveform_change_NMSE=float(delta[keep].mean()) if keep.any() else None,
                    aligned_ID_agreement=float(np.mean(ref[keep] == mapping[alt[keep]])) if keep.any() else None,
                    windows=int(select.sum()), subjects=len({r['subject'] for i, r in enumerate(rows) if select[i]}),
                    interpretation='paired train-identity teacher sensitivity; no heldout-generalization or physiological-truth claim'))
    _csv(root / 'prototype_sensitivity.csv', results)
    result = dict(status='completed', seed=seed, paired_steps=int(cfg.get('sensitivity', {}).get('paired_steps', 500)),
        rows=results, evidence=declared, evaluation='same selected training identities, equal optimizer budget',
        scientific_limit='finite legal whole-trajectory witness, not posterior or complete feasible set',
        physiological_truth_claim=False)
    _json(root / 'analysis.json', result)
    return result


def summarize_masked_pilots(cfg, run):
    run = Path(run); config = cfg.get('masked', {}); subjects, missing = [], []
    for route in config.get('routes', ('hard_student', 'smooth_hard', 'ordinary_hard')):
        for seed in config.get('seeds', cfg.get('seeds', (17, 29, 43))):
            path = run / 'masked' / route / ('seed_' + str(seed)) / 'subject_metrics.csv'
            if not path.exists():
                missing.append(dict(route=route, seed=seed)); continue
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    row['value'] = float(row['value']) if row['value'] else None
                    row['seed'] = int(row['seed']); subjects.append(row)
    if not subjects: return dict(status='not_run', missing=missing)
    comparisons = []; samples = int(cfg.get('evaluation', {}).get('bootstrap_repeats', 1000))
    grouped = defaultdict(list)
    for row in subjects:
        grouped[(row['variant'], row['dataset'], row['split'], row['task'], row['modality'], row['component'], row['metric'])].append(row)
    for (route, dataset, split, task, modality, component, metric), group in grouped.items():
        if metric not in ('NMSE', 'NMSE_HbO', 'NMSE_HbR', 'CE_nats', 'DeltaCE_metadata_nats'): continue
        first = [r for r in group if r['context'] == 'paired']
        for context in sorted({r['context'] for r in group} - {'paired'}):
            second = [r for r in group if r['context'] == context]
            result = paired_subject_bootstrap(first, second, value='value', samples=samples, seed=int(cfg.get('seed', 0)))
            result['effect'] = 'paired minus control; ' + ('higher favors paired' if metric.startswith('DeltaCE') else 'lower favors paired')
            comparisons.append(dict(route=route, dataset=dataset, split=split, task=task, modality=modality,
                component=component, metric=metric, first='paired', second=context, **result))
    # Across vocabularies compare only the shared observation-space sum error.
    for control in ('smooth_hard', 'ordinary_hard'):
        keys = {(r['dataset'], r['split'], r['task'], r['modality']) for r in subjects
                if r['metric'] in ('NMSE', 'NMSE_HbO', 'NMSE_HbR') and r['component'] == 'sum' and r['context'] == 'paired'}
        for dataset, split, task, modality in keys:
            def pick(route):
                return [r for r in subjects if r['variant'] == route and r['dataset'] == dataset
                    and r['split'] == split and r['task'] == task and r['modality'] == modality
                    and r['metric'] == 'NMSE' and r['component'] == 'sum' and r['context'] == 'paired']
            result = paired_subject_bootstrap(pick('hard_student'), pick(control), value='value', samples=samples, seed=int(cfg.get('seed', 0)))
            comparisons.append(dict(route='cross_vocabulary', dataset=dataset, split=split, task=task,
                modality=modality, component='sum', metric='NMSE', first='hard_student', second=control, **result))
    _csv(run / 'analysis' / 'masked_subject_metrics.csv', subjects)
    _csv(run / 'analysis' / 'masked_paired_comparisons.csv', comparisons)
    result = dict(status='completed' if not missing else 'partial', missing=missing, subject_rows=len(subjects),
        comparisons=comparisons, cross_vocabulary_CE_ranking_allowed=False,
        aggregation='equal record means per subject; seeds averaged within subject before resampling',
        inference='exploratory percentile subject bootstrap; no multiple-comparison significance claims')
    _json(run / 'analysis' / 'masked_summary.json', result)
    return result


def refinement_fidelity(predictions, targets, rows, variant):
    """Training/validation waveform metrics; no test rows or per-window scaling."""
    if any(r['split'] not in ('train', 'validation') for r in rows):
        raise ValueError('refinement selection is training/validation only')
    training = np.array([r['split'] == 'train' for r in rows])
    window_rows = []
    for modality, channels in (('eeg', 1), ('hb', 2)):
        truth = dict(S=targets[modality + '_s'], O=targets[modality + '_e'], sum=targets[modality + '_u'])
        for component, target in truth.items():
            estimate = np.asarray(predictions[modality + '_' + component], float)
            target = np.asarray(target, float)
            if estimate.shape != target.shape or len(target) != len(rows):
                raise ValueError('aligned complete waveform support required')
            a = estimate.reshape(len(rows), -1, channels)
            b = target.reshape(a.shape)
            centered_a = a - a.mean(1, keepdims=True); centered_b = b - b.mean(1, keepdims=True)
            norm_a = np.sqrt(np.sum(centered_a ** 2, axis=(1, 2)))
            norm_b = np.sqrt(np.sum(centered_b ** 2, axis=(1, 2)))
            denominator = norm_a * norm_b
            correlation = np.divide(np.sum(centered_a * centered_b, axis=(1, 2)), denominator,
                out=np.full(len(rows), np.nan), where=denominator > 1e-12)
            delta_a, delta_b = np.diff(a, axis=1), np.diff(b, axis=1)
            # Frozen training RMS of differences, not SD or a window denominator.
            delta_scale = max(float(np.mean(delta_b[training] ** 2)), 1e-8)
            metrics = dict(NMSE=np.mean((a - b) ** 2, axis=(1, 2)),
                derivative_NMSE=np.mean((delta_a - delta_b) ** 2, axis=(1, 2)) / delta_scale,
                centered_correlation=correlation,
                centered_amplitude_ratio=np.divide(norm_a, norm_b, out=np.full(len(rows), np.nan), where=norm_b > 1e-12))
            for metric, values in metrics.items():
                for i, row in enumerate(rows):
                    window_rows.append(dict(dataset=row['dataset'], subject=row['subject'], record=row['record'],
                        split=row['split'], variant=variant, modality=modality, component=component,
                        metric=metric, value=float(values[i]) if np.isfinite(values[i]) else None,
                        failures=int(not np.isfinite(values[i]))))
    subjects = subject_aggregate(window_rows, value='value')
    grouped = defaultdict(list)
    for row in subjects:
        grouped[(row['dataset'], row['split'], row['modality'], row['component'], row['metric'])].append(row)
    summaries = []
    for (dataset, split, modality, component, metric), values in grouped.items():
        valid = [r['value'] for r in values if r['value'] is not None]
        summaries.append(dict(dataset=dataset, split=split, modality=modality, component=component,
            metric=metric, value=float(np.mean(valid)) if valid else None, subjects=len(valid),
            undefined_subjects=len(values) - len(valid), undefined_windows=sum(r['failures'] for r in values)))
    return subjects, summaries


def refinement_code_health(ids, words, rows, modality, component, size):
    """Per-level occupancy and correctly identified subject/record coverage."""
    result = []
    for level in range(ids.shape[-1]):
        for split in ('train', 'validation'):
            take = np.array([r['split'] == split for r in rows])
            selected = ids[take, :, level]
            counts = np.bincount(selected.ravel(), minlength=size)
            probabilities = counts / counts.sum()
            nz = probabilities > 0
            per_code_subjects, per_code_records = [], []
            for code in range(size):
                assigned = np.flatnonzero(take & (ids[:, :, level] == code).any(-1))
                per_code_subjects.append(len({(rows[i]['dataset'], rows[i]['subject']) for i in assigned}))
                per_code_records.append(len({(rows[i]['dataset'], rows[i]['subject'], rows[i]['record']) for i in assigned}))
            prototypes = np.asarray(words[level], float)
            distance = np.mean((prototypes[:, None] - prototypes[None]) ** 2, axis=-1)
            np.fill_diagonal(distance, np.inf)
            result.append(dict(modality=modality, component=component, level=level, split=split,
                size=size, active_codes=int(nz.sum()), active_fraction=float(nz.mean()),
                perplexity=float(np.exp(-(probabilities[nz] * np.log(probabilities[nz])).sum())),
                max_probability=float(probabilities.max()), minimum_assignments=int(counts.min()),
                codes_with_three_subjects_fraction=float(np.mean(np.asarray(per_code_subjects) >= 3)),
                per_code_subjects=per_code_subjects, per_code_records=per_code_records,
                nearest_prototype_RMS_median=float(np.sqrt(np.median(distance.min(-1))))))
    return result


def refinement_decision(summary, baseline, health, settings):
    """Predeclared engineering screen: usage cannot substitute for fidelity."""
    reference = {(r['dataset'], r['modality'], r['component']):r['value'] for r in baseline
        if r['split'] == 'validation' and r['metric'] == 'NMSE'}
    fidelity = []
    for row in summary:
        if row['split'] != 'validation' or row['metric'] != 'NMSE': continue
        key = row['dataset'], row['modality'], row['component']; base = reference[key]
        gain = 1 - row['value'] / base if row['value'] is not None and base > 0 else None
        fidelity.append(dict(dataset=key[0], modality=key[1], component=key[2], baseline_NMSE=base,
            NMSE=row['value'], relative_improvement=gain,
            passed=bool(gain is not None and gain >= settings['minimum_waveform_improvement'])))
    health_checks = []
    for row in health:
        required_active = settings['train_active_fraction'] if row['split'] == 'train' else settings['validation_active_fraction']
        health_checks.append(dict(modality=row['modality'], component=row['component'], level=row['level'], split=row['split'],
            passed=bool(row['active_fraction'] >= required_active and row['perplexity'] / row['size'] >= settings['effective_fraction']
                and row['max_probability'] <= settings['maximum_code_probability']),
            active_fraction=row['active_fraction'], effective_fraction=row['perplexity'] / row['size'],
            maximum_code_probability=row['max_probability']))
    return dict(fidelity_pass=all(r['passed'] for r in fidelity),
        health_pass=bool(health_checks and all(r['passed'] for r in health_checks)),
        fidelity=fidelity, health=health_checks,
        selection='validation only; no posthoc change of success thresholds')


def render_refinement(cfg, run, selected, *, export=None):
    """Fixed-rule validation cases: teacher, previous hard decode, new hard decode."""
    if selected is None: return []
    import matplotlib; matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    run = Path(run); cache = run / 'cache'; manifest = json.loads((cache / 'manifest.json').read_text())
    folder = run / 'refinement' / selected['candidate'] / ('seed_' + str(selected['seed']))
    variant = 'hard' if selected['name'] == selected['candidate'] else 'vq' + selected['name'].rsplit('_vq', 1)[1]
    continuous_file = folder / 'continuous/validation_predictions.npz'
    folder = folder / variant
    with np.load(run / 'baseline_hard/validation_predictions.npz') as data:
        baseline = {k:data[k] for k in data.files}
    with np.load(folder / 'validation_predictions.npz') as data:
        updated = {k:data[k] for k in data.files}
    if not np.array_equal(baseline['row_index'], updated['row_index']):
        raise ValueError('comparison plots require identical validation identities')
    continuous = None
    if continuous_file.exists():
        with np.load(continuous_file) as data: continuous = {k:data[k] for k in data.files}
        if not np.array_equal(baseline['row_index'], continuous['row_index']):
            raise ValueError('continuous/hard comparison requires identical identities')
    indices = baseline['row_index']; rows = [manifest['rows'][i] for i in indices]
    targets = {m + '_' + c:np.load(cache / (m + '_' + suffix + '.npy'), mmap_mode='r')[indices]
        for m in MODALITIES for c, suffix in (('S','s'),('O','e'),('sum','u'))}
    baseline_error = np.mean((baseline['hb_sum'] - targets['hb_sum']) ** 2, axis=(1,2))
    destination = run if export is None else Path(export)
    if export is not None: destination.mkdir(parents=True, exist_ok=False)
    examples = []; files = []; output = destination / 'figures'; output.mkdir(exist_ok=True)
    for dataset in sorted({r['dataset'] for r in rows}):
        group = [i for i, row in enumerate(rows) if row['dataset'] == dataset]
        used = set()
        for label, quantile in [('first_identity', None), ('median_parent_error', .5), ('p90_parent_error', .9)]:
            candidates = [i for i in group if rows[i]['subject'] not in used]
            if not candidates: continue
            if quantile is None: i = candidates[0]
            else:
                target = float(np.quantile(baseline_error[group], quantile))
                i = min(candidates, key=lambda j: (abs(baseline_error[j] - target), rows[j]['id']))
            used.add(rows[i]['subject']); examples.append(dict(row=rows[i], selection=label,
                parent_sum_NMSE=float(baseline_error[i]), candidate_used_for_selection=False))
            for modality, channels in [('eeg',1),('hb',2)]:
                fig, axes = plt.subplots(3, channels, figsize=(12 if channels == 2 else 9, 8),
                    squeeze=False, constrained_layout=True)
                time_axis = np.arange(200) * .1
                for row_index, component in enumerate(('S','O','sum')):
                    key = modality + '_' + component
                    truth = targets[key][i].reshape(200, channels)
                    old = baseline[key][i].reshape(200, channels)
                    new = updated[key][i].reshape(200, channels)
                    for channel in range(channels):
                        ax = axes[row_index, channel]
                        ax.plot(time_axis, truth[:,channel], color='black', lw=1.7,
                            label='Observed target' if component == 'sum' else 'Fixed Tak component target')
                        ax.plot(time_axis, old[:,channel], color='#888888', ls='--', lw=1.5, label='Previous hard tokenizer')
                        if continuous is not None:
                            before_vq = continuous[key][i].reshape(200, channels)[:,channel]
                            ax.plot(time_axis, before_vq, color='#D55E00', ls=':', lw=1.4, label='Refined before quantization')
                        ax.plot(time_axis, new[:,channel], color='#0072B2', lw=1.6, label='Refined hard tokenizer')
                        before = np.mean((old[:,channel] - truth[:,channel]) ** 2)
                        after = np.mean((new[:,channel] - truth[:,channel]) ** 2)
                        signal = ('HbO','HbR')[channel] if channels == 2 else 'EEG power proxy'
                        meaning = dict(S='S: Tak output', O='O: fixed residual', sum='total: observed')[component]
                        ax.set_title(f'{signal} | {meaning} | NMSE {before:.4g} -> {after:.4g}', fontsize=10)
                        ax.set_xlabel('Event time (s)'); ax.set_ylabel('Frozen training SD'); ax.grid(alpha=.2)
                        if row_index == 0: ax.legend(fontsize=7, loc='best')
                candidate = next(c for c in cfg['refinement']['candidates'] if c['name'] == selected['candidate'])
                support = 2 * (1 + 2 * candidate['context_radius'])
                fig.suptitle(f'{selected["name"]} | seed {selected["seed"]} | {support}s processed-observation support | '
                    f'14 -> {selected["bits"]:g} bits / modality / 2s\n'
                    f'Validation: {rows[i]["id"]}\n'
                    f'{label}; complete observations visible; no masked-model predictions', fontsize=10)
                path = output / f'{dataset}_{label}_{modality}.png'
                fig.savefig(path, dpi=250); plt.close(fig); files.append(str(path))
    _json(destination / 'examples.json', examples)
    report = json.loads((folder / 'summary.json').read_text())
    health = [x for x in report['codebook_health'] if x['split'] == 'validation']
    fig, axes = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    labels = [f'{x["modality"]}-{x["component"]}-L{x["level"]+1}' for x in health]
    for ax, values, threshold, title in [
        (axes[0],[x['active_fraction'] for x in health],cfg['refinement']['health']['validation_active_fraction'],'Active codes / 128'),
        (axes[1],[x['perplexity']/x['size'] for x in health],cfg['refinement']['health']['effective_fraction'],'Effective codes / 128')]:
        ax.bar(labels, values, color='#0072B2'); ax.axhline(threshold,color='#D55E00',ls='--',label='Predeclared threshold')
        ax.set_ylim(0,1.05); ax.set_title(title); ax.tick_params(axis='x',rotation=35); ax.legend(fontsize=8)
        for i,value in enumerate(values): ax.text(i,value+.01,f'{value:.2f}',ha='center',fontsize=8)
    fig.suptitle(f'Validation codebook health | {selected["name"]} | seed {selected["seed"]} | {selected["bits"]:g} bits / modality / 2s')
    path = output / 'selected_codebook_health.png'; fig.savefig(path,dpi=250);plt.close(fig);files.append(str(path))
    suite_path = run / 'refinement_summary.json'
    if suite_path.exists():
        suite = json.loads(suite_path.read_text())
        matching = [v for r in suite['candidates'] + suite.get('confirmation', [])
            if r.get('status') == 'completed' and r['candidate']['name'] == selected['candidate']
            for v in r['variants'] if v['name'] == selected['name']]
        if matching:
            datasets = sorted({r['dataset'] for r in report['decision']['fidelity']})
            labels = ['EEG S','EEG O','EEG total','Hb S','Hb O','Hb total']
            keys = [(m,c) for m in ('eeg','hb') for c in ('S','O','sum')]
            fig, axes = plt.subplots(1,len(datasets),figsize=(13,5),squeeze=False,constrained_layout=True)
            for ax,dataset in zip(axes[0],datasets):
                values = np.array([[next(r['relative_improvement']*100 for r in v['decision']['fidelity']
                    if (r['dataset'],r['modality'],r['component']) == (dataset,*key)) for key in keys] for v in matching])
                mean = values.mean(0); spread = np.stack((mean-values.min(0),values.max(0)-mean))
                ax.bar(np.arange(6),mean,color=['#0072B2']*3+['#009E73']*3,yerr=spread,capsize=4)
                for i,value in enumerate(mean): ax.text(i,value+3,f'{value:.1f}%',ha='center',fontsize=10)
                ax.axhline(30,color='#D55E00',ls='--',label='Predeclared 30% improvement')
                ax.axhline(0,color='black',lw=.8)
                ax.set_xticks(np.arange(6),labels,rotation=25)
                ax.set_ylim(min(-5,float(values.min())-10),105)
                ax.set_ylabel('NMSE reduction versus previous hard tokenizer (%)')
                ax.set_title(dataset); ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
            fig.suptitle(f'{selected["name"]}: mean and range of {len(matching)} seeds on the same validation subjects\n'
                f'{selected["bits"]:g} versus 14 bits / modality / 2s; fixed training-SD denominator; subjects weighted equally',fontsize=11)
            path=output/'seed_fidelity_comparison.png';fig.savefig(path,dpi=250);plt.close(fig);files.append(str(path))
    return files


VOLTAGE_METRICS=('NMSE','centered_correlation','boundary_error_NMSE','boundary_jump_RMS','derivative_error_NMSE')


def boundary_neighborhood_errors(prediction,target,*,hop=20,spans=(1,3,5,11)):
    """Separate instantaneous seam jumps from persistent offsets across a seam.

    An odd span k compares samples k intervals apart, centred on each seam.
    The target's signed change is subtracted before squaring. Result [case,span]
    averages boundaries and channels, with no sample-specific normalization.
    """
    a=np.asarray(prediction,float);b=np.asarray(target,float)
    if a.shape!=b.shape or a.ndim!=3:raise ValueError('aligned case/time/channel waveforms required')
    errors=[]
    for span in spans:
        if span<1 or span%2!=1:raise ValueError('positive odd boundary span required')
        right=np.arange(hop,a.shape[1],hop)+(span-1)//2;left=right-span
        take=(left>=0)&(right<a.shape[1]);left,right=left[take],right[take]
        if not len(left):raise ValueError('no complete boundary neighborhoods')
        difference=(a[:,right]-a[:,left])-(b[:,right]-b[:,left])
        errors.append(np.mean(difference*difference,axis=(1,2)))
    return np.stack(errors,axis=-1)


def voltage_case_metrics(predictions,targets,*,hop):
    """Per-electrode/ROI-case metrics in frozen observation units.

    The boundary error compares *signed* adjacent-sample differences at hop
    boundaries. A smooth but incorrect curve therefore cannot pass by erasing
    legitimate slopes. HbO/HbR are also scored separately.
    """
    values=np.asarray(targets['sum'],float)
    signals=[('joint',slice(None))] if values.shape[-1]==1 else [('joint',slice(None)),('HbO',slice(0,1)),('HbR',slice(1,2))]
    result=np.full((len(values),3,len(signals),len(VOLTAGE_METRICS)),np.nan)
    for j,component in enumerate(('S','O','sum')):
        if component not in predictions:continue
        a=np.asarray(predictions[component],float);b=np.asarray(targets[component],float)
        if a.shape!=b.shape or a.shape!=values.shape:raise ValueError('same complete continuous support required')
        for k,(_,select) in enumerate(signals):
            x,y=a[:,:,select],b[:,:,select];dx,dy=np.diff(x,axis=1),np.diff(y,axis=1)
            xc=x-x.mean(1,keepdims=True);yc=y-y.mean(1,keepdims=True)
            denominator=np.sqrt(np.sum(xc*xc,axis=(1,2))*np.sum(yc*yc,axis=(1,2)))
            corr=np.divide(np.sum(xc*yc,axis=(1,2)),denominator,out=np.full(len(x),np.nan),where=denominator>1e-12)
            boundary=dx[:,hop-1::hop]-dy[:,hop-1::hop]
            result[:,j,k,:]=np.stack((np.mean((x-y)**2,axis=(1,2)),corr,
                np.mean(boundary**2,axis=(1,2)),np.sqrt(np.mean(dx[:,hop-1::hop]**2,axis=(1,2))),
                np.mean((dx-dy)**2,axis=(1,2))),axis=-1)
    return result,[x[0] for x in signals]


def voltage_aggregate(values,rows,*,modality,variant,signals):
    """Equal records within subject, then equal subjects; all case counts retained."""
    grouped=defaultdict(lambda:defaultdict(list))
    for i,row in enumerate(rows):grouped[(row['dataset'],row['split'],row['subject'])][row['record']].append(i)
    def mean(x,axis=0):
        n=np.isfinite(x).sum(axis=axis)
        return np.divide(np.nansum(x,axis=axis),n,out=np.full(n.shape,np.nan),where=n>0)
    subjects=[];summaries=defaultdict(list)
    for (dataset,split,subject),records in grouped.items():
        average=mean(np.stack([mean(values[index]) for index in records.values()]))
        indices=[i for ids in records.values() for i in ids]
        for c,component in enumerate(('S','O','sum')):
            for s,signal in enumerate(signals):
                for j,metric in enumerate(VOLTAGE_METRICS):
                    value=float(average[c,s,j]);entry=dict(dataset=dataset,split=split,subject=subject,modality=modality,
                        variant=variant,component=component,signal=signal,metric=metric,value=value if np.isfinite(value) else None,
                        records=len(records),cases=len(indices),undefined_cases=int(np.sum(~np.isfinite(values[indices,c,s,j]))))
                    subjects.append(entry);summaries[(dataset,split,component,signal,metric)].append(entry)
    result=[]
    for (dataset,split,component,signal,metric),group in summaries.items():
        valid=[r['value'] for r in group if r['value'] is not None]
        result.append(dict(dataset=dataset,split=split,modality=modality,component=component,signal=signal,metric=metric,
            value=float(np.mean(valid)) if valid else None,subjects=len(valid),planned_subjects=len(group),
            cases=sum(r['cases'] for r in group),undefined_cases=sum(r['undefined_cases'] for r in group)))
    return subjects,result


def token_tuple_catalog(ids,rows,*,size=128):
    """Count complete residual-VQ tuples without merging shared first-level IDs."""
    ids=np.asarray(ids)
    if ids.ndim!=3 or len(ids)!=len(rows) or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=size):
        raise ValueError('Complete integer [case,patch,level] IDs and aligned identities required')
    if size**ids.shape[-1]>np.iinfo(np.int64).max:raise ValueError('Tuple packing would overflow')
    packed=np.zeros(ids.shape[:2],dtype='int64')
    for level in range(ids.shape[-1]):packed=packed*size+ids[...,level]
    keys,counts=np.unique(packed,return_counts=True)
    subjects=sorted({(r['dataset'],r['subject']) for r in rows});lookup={s:i for i,s in enumerate(subjects)}
    sid=np.array([lookup[(r['dataset'],r['subject'])] for r in rows],dtype='int64')
    if size**ids.shape[-1]*len(subjects)>np.iinfo(np.int64).max:raise ValueError('Subject/tuple packing would overflow')
    pairs=np.unique(packed*len(subjects)+sid[:,None]);subject_keys,subject_counts=np.unique(pairs//len(subjects),return_counts=True)
    if not np.array_equal(keys,subject_keys):raise ValueError('Tuple subject-count identity mismatch')
    return dict(keys=keys,counts=counts,subject_counts=subject_counts)


def select_token_tuple_examples(catalog,book,*,levels,count=3,mode='frequency'):
    """Choose frequent prototypes spanning amplitude/shape, never input fit error."""
    eligible=np.flatnonzero((catalog['counts']>=20)&(catalog['subject_counts']>=2))
    frequency_order=sorted(eligible,key=lambda i:(-int(catalog['counts'][i]),int(catalog['keys'][i])))
    if mode not in ('frequency','shape'):raise ValueError('Unknown token illustration selection')
    order=frequency_order if mode=='shape' else frequency_order[:25]
    if len(order)<count:raise ValueError('Too few repeated multi-subject full tuples for the requested examples')
    keys=catalog['keys'][order].copy();codes=np.zeros((len(keys),levels),dtype='int16')
    for level in range(levels-1,-1,-1):codes[:,level]=keys%book.size;keys//=book.size
    if keys.any():raise ValueError('Tuple level count does not match dictionary')
    waves=book.decode(codes).astype(float)
    if mode=='shape':
        shaped=waves.reshape(len(waves),-1,book.channels);centered=shaped-shaped.mean(axis=1,keepdims=True)
        dynamic=np.sqrt(np.mean(centered**2,axis=(1,2)));top=np.argsort(-dynamic,kind='stable')[:25]
        order=[order[i] for i in top];codes=codes[top];waves=centered.reshape(len(waves),-1)[top]
    selected=[0]
    while len(selected)<count:
        distance=np.min(np.mean((waves[:,None,:]-waves[selected][None,:,:])**2,axis=-1),axis=1)
        distance[selected]=-1;selected.append(int(np.argmax(distance)))
    return [dict(key=int(catalog['keys'][order[i]]),ids=codes[i].tolist(),occurrences=int(catalog['counts'][order[i]]),
        subject_count=int(catalog['subject_counts'][order[i]]),candidate_rank=i+1,
        frequency_rank=frequency_order.index(order[i])+1) for i in selected]


def render_token_correspondence(analysis,out):
    """Dark fixed S prototypes against every assigned raw input patch in pale blue."""
    import os
    import shutil
    from html import escape
    import matplotlib;matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    import fitz
    analysis,out=Path(analysis),Path(out);summary=json.loads((analysis/'summary.json').read_text())
    if summary['status']!='completed' or summary['protected_reads']!=0 or summary['new_fits']!=0:raise ValueError('Completed retained-only token analysis required')
    if out.exists():raise ValueError('Preserve previous figure exports; choose a fresh output')
    out.mkdir(parents=True);(out/'figures').mkdir()
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':12,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    figures=[];pages=[];source=escape(os.path.relpath(analysis,out));dark='#14578a';light='#9ec9ef'
    for modality in ['hb','eeg']:
        result=summary['results'][modality];channels=result['channels'];items=[]
        for item in result['selected']:
            with np.load(analysis/modality/item['file'],allow_pickle=False) as a:raw=a['original'];prototype=a['prototype'];codes=a['ids']
            if len(raw)!=item['occurrences'] or raw.shape[1:]!=prototype.shape or codes.tolist()!=item['ids']:
                raise ValueError('Prototype/cloud identity or complete-member count mismatch')
            items.append((item,raw,prototype))
        fig,axes=plt.subplots(channels,len(items),figsize=(14,7.4 if channels==2 else 4.7),squeeze=False,layout='constrained')
        labels=['HbO','HbR'] if modality=='hb' else ['EEG']
        shape_view=summary.get('selection_view')=='shape';limits=[]
        for channel in range(channels):
            low=min(min(float(raw[:,:,channel].min()),float(prototype[:,channel].min())) for _,raw,prototype in items)
            high=max(max(float(raw[:,:,channel].max()),float(prototype[:,channel].max())) for _,raw,prototype in items)
            margin=max((high-low)*.06,1e-4);limits.append((low-margin,high+margin))
        time_axis=np.arange(result['hop_samples'])/result['sample_rate_hz']*(1000 if modality=='eeg' else 1)
        for col,(item,raw,prototype) in enumerate(items):
            code=' · '.join(map(str,item['ids']))
            for channel in range(channels):
                ax=axes[channel,col];coordinates=np.stack([np.broadcast_to(time_axis,raw[:,:,channel].shape),raw[:,:,channel]],axis=-1)
                ax.add_collection(LineCollection(coordinates,colors=light,linewidths=.65,alpha=.13,rasterized=True,zorder=1))
                ax.plot(time_axis,prototype[:,channel],color=dark,lw=2.7,zorder=3)
                low,high=limits[channel]
                if shape_view:
                    low=min(float(raw[:,:,channel].min()),float(prototype[:,channel].min()));high=max(float(raw[:,:,channel].max()),float(prototype[:,channel].max()))
                    margin=max((high-low)*.06,1e-4);low-=margin;high+=margin
                ax.set_xlim(0,result['hop_samples']/result['sample_rate_hz']*(1000 if modality=='eeg' else 1));ax.set_ylim(low,high);ax.grid(alpha=.12)
                ax.set_xlabel('片段内时间 / 毫秒' if modality=='eeg' else '片段内时间 / 秒')
                ax.set_ylabel(labels[channel]+(' / 冻结训练RMS' if modality=='eeg' else ' / 冻结训练SD'))
                ax.set_title('S-token ['+code+f']\n{item["occurrences"]}个原始片段 · {item["subjects"]}名被试',fontsize=11)
        fig.suptitle(('HbO / HbR' if modality=='hb' else 'EEG')+'：S-token解码波形与全部对应原始片段'+('\n各列纵轴独立缩放，保留所有原始幅度' if shape_view else ''),fontsize=17)
        fig.legend(handles=[Line2D([],[],color=dark,lw=2.7,label='S-token解码波形'),Line2D([],[],color=light,lw=1.5,label='对应的全部原始输入片段')],
            loc='outside lower center',ncol=2,fontsize=12)
        file='figures/'+modality+'_S_token_correspondence.png';fig.savefig(out/file,dpi=260,facecolor='white');plt.close(fig)
        figures.append(dict(modality=modality,file=file,examples=len(items),plotted_lines=[item['occurrences'] for item,_,_ in items],all_members=True))
        caption=('每列是一个完整S码组合，深蓝线由所有层的固定码字相加直接得到；浅蓝细线逐条绘制该组合在已评价公开测试集中的全部原始输入片段，没有抽样或按贴合程度筛除。'
            +('同列HbO/HbR属于同一批2秒片段。这里显示单码原型，未应用依赖邻码的100ms拼接后平滑。' if modality=='hb' else '每条EEG片段为100毫秒；编码器使用的同电极2秒上下文未在横轴展开。')
            +('各列使用独立纵轴范围，完整显示原始极值；跨列比较幅度须读刻度。' if shape_view else '同一信号的三列共用完整纵轴范围，保留所有极值。')
            +'浅蓝是观测输入u，深蓝仅为S分支；两者差异还包含O分支，不能直接当作完整S+O重建误差。')
        pages.append('<h1>'+('HbO/HbR' if modality=='hb' else 'EEG')+'：S-token与原始输入对应</h1><img width="750" src="'+file+'"><p>'+caption+'</p><p class="source">来源：<a href="'+source+'/'+modality+'/summary.json">完整分母与选码规则</a>；同目录example_*_members.csv记录每条浅蓝线的身份。</p>')
    counts={m:summary['results'][m]['total_patches'] for m in ['eeg','hb']}
    intro='<h1>S-token重建与原始信号：蓝色对应图</h1><p>来源：已完成的电极电压版tokenizer v2，固定seed 17。模型、词表、训练尺度及测试被试均不改变；本次只回放编码和整数解码，不训练、不重新拟合SSM。</p>'
    intro+=f'<p>统计范围为此前已评价的公开测试集，两库各6名被试：EEG共{counts["eeg"]:,}个100毫秒片段，Hb共{counts["hb"]:,}个2秒双Hb片段。EEG的完整S-token为4层整数码组合，Hb为2层；不能只用第一层ID把不同完整解码波形混为一类。</p>'
    intro+=('<p>选例先要求同一完整组合至少出现20次、覆盖至少2名被试，再按解码原型的时间变化RMS选前25组。第一例取变化最大，另两例在去时间均值的原型上依次取波形距离较大的组合；这里只在选例时去均值，实际图中的原始幅度保持不变。选择不使用原始片段贴合误差，属于形态展示样例，不是频率代表或随机样本。</p>' if summary.get('selection_view')=='shape' else
        '<p>选例先要求同一完整组合至少出现20次、覆盖至少2名被试，再取频次最高的25组：第一例为最高频，另两例按解码波形RMS距离依次选择差异较大的原型。此规则展示常见码的幅度与形状范围，没有按原始片段贴合程度选图；不代表总体最优或随机样本。</p>')
    if summary.get('selection_view')=='shape':
        diagnostic=escape(os.path.relpath(Path(summary['frequency_report']),out))
        intro+='<p>最高频码的幅度饱和和接近平坦的原型保留于<a href="'+diagnostic+'/index.html">高频码完整诊断图</a>，本组形态展示不替代这些负结果。</p>'
    intro+='<p>每个所选码组合的全部匹配片段都画成浅蓝背景；深蓝为该S-token唯一的独立解码原型。原始输入保持缓存的冻结训练尺度，不逐片段去均值、对齐峰值或归一化。EEG单位为各数据集/电极的训练RMS，Hb为继承的训练观测SD；跨坐标的原生μV或浓度单位不能直接合并成一条固定原型。</p>'
    intro+='<p>S是条件模型分量，不是原始观测u本身。Hb的S来自固定Tak输出监督，电压S来自条件幅度修正；它们均不是已知生理真值。本图说明“一组原始片段对应一个离散表示”的关系，完整观测重建仍需S+O。</p>'
    sections=[intro]+pages;css='body{font-family:sans-serif;font-size:10pt;line-height:1.45;color:#223746}h1{font-size:18pt;margin:0 0 10pt}p{margin:8pt 0}.source{font-size:8pt}img{max-width:100%}'
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>S-token对应图</title><style>'+css+'body{max-width:1150px;margin:30px auto;padding:0 24px}section{margin:35px 0}img{width:100%}</style><body><p><a href="FIGURE_ATLAS.pdf">PDF图册</a></p>'+''.join('<section>'+p+'</section>' for p in sections)+'</body></html>')
    pdf=fitz.open()
    def rectfn(number,filled):
        r=fitz.Rect(0,0,842,595);return r,r+(32,25,-32,-25),None
    for page in sections:
        part=fitz.Story(page,user_css=css,archive=fitz.Archive(str(out))).write_with_links(rectfn)
        if len(part)!=1:raise ValueError('Token-atlas page overflow; split the visible explanation')
        pdf.insert_pdf(part);part.close()
    pdf.save(out/'FIGURE_ATLAS.pdf',garbage=4,deflate=True)
    validation=dict(pages=len(pdf),bitmap_images=sum(len(p.get_images()) for p in pdf),searchable_characters=sum(len(p.get_text()) for p in pdf),png_dpi=260)
    if validation['bitmap_images']!=2 or validation['searchable_characters']<800:raise ValueError('Searchable bitmap atlas verification failed')
    pdf.close();_json(out/'source_manifest.json',dict(owning_analysis=str(analysis),figures=figures,validation=validation,refits=0))
    shutil.copy2(Path(__file__),out/'renderer_source.py');return figures


def render_voltage_examples(out,examples,summary):
    import matplotlib;matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    out=Path(out);folder=out/'voltage_figures';folder.mkdir(exist_ok=True);files=[];index=[]
    for item in examples:
        row=item['row'];modality=item['modality'];channels=item['targets']['sum'].shape[-1]
        show=400 if modality=='eeg' else 200
        time_axis=np.arange(show)/item['sample_rate_hz']
        fig,axes=plt.subplots(3,channels,figsize=(12 if channels==2 else 10,8),squeeze=False,constrained_layout=True)
        for j,component in enumerate(['S','O','sum']):
            truth=item['targets'][component]
            for c in range(channels):
                ax=axes[j,c];ax.plot(time_axis,truth[:show,c],color='black',lw=1.4,label='Conditional target' if component!='sum' else 'Observed voltage/Hb')
                for role,color,style,label in [('baseline','#888888','--',item['baseline']),('selected','#0072B2','-',item['candidate']),('continuous','#D55E00',':','Before quantization')]:
                    ax.plot(time_axis,item['predictions'][role][component][:show,c],color=color,ls=style,lw=1.2,label=label)
                error=np.mean((item['predictions']['selected'][component][:,c]-truth[:,c])**2)
                label=row['channel'] if modality=='eeg' else ['HbO','HbR'][c]
                ax.set_title(f'{label} | {component} | complete-20s MSE {error:.4g}',fontsize=10)
                ax.set_xlabel('Event time (s)');ax.set_ylabel(item['units']);ax.grid(alpha=.2)
                if j==0:ax.legend(fontsize=7)
                if modality=='hb':
                    for t in range(2,20,2):ax.axvline(t,color='#bbbbbb',lw=.6,alpha=.5)
        fig.suptitle(f'{modality}: {item["selected_rate"]} | seed 17 | public test\n{row["id"]}\n'
            f'{item["selection"]}; '+('first 2s shown; metrics use all 20s' if modality=='eeg' else 'vertical lines: original 2s patch boundaries'),fontsize=10)
        path=folder/f'{row["dataset"]}_{item["selection"]}_{modality}.png';fig.savefig(path,dpi=250);plt.close(fig)
        files.append(str(path));index.append(dict(row=row,modality=modality,selection=item['selection']))
    metrics=summary['metrics'];datasets=sorted({r['dataset'] for r in metrics})
    fig,axes=plt.subplots(1,len(datasets),figsize=(12,4.5),squeeze=False,constrained_layout=True)
    for ax,dataset in zip(axes[0],datasets):
        labels=[];before=[];after=[]
        for component in ['S','O','sum']:
            for signal in ['HbO','HbR']:
                lookup=lambda model:next(r['value'] for r in metrics if (r['dataset'],r['modality'],r['component'],r['signal'],r['metric'],r['model'])==(dataset,'hb',component,signal,'boundary_error_NMSE',model))
                labels.append(component+' '+signal);before.append(lookup('hb_blocks'));after.append(lookup('selected'))
        x=np.arange(len(labels));ax.bar(x-.18,before,width=.36,color='#888888',label='Independent 2s blocks')
        label=summary['selection']['hb_candidate']+(' + 100ms synthesis' if summary['selection'].get('hb_variant')=='vq2_smoothed' else '')
        ax.bar(x+.18,after,width=.36,color='#0072B2',label=label)
        ax.set_xticks(x,labels,rotation=30);ax.set_ylabel('Signed boundary-difference error (training SD squared)')
        ax.set_title(dataset);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Public test Hb boundary error | seeds averaged within subject, subjects weighted equally',fontsize=11)
    path=folder/'hb_boundary_comparison.png';fig.savefig(path,dpi=250);plt.close(fig);files.append(str(path))
    _json(out/'voltage_figure_index.json',dict(files=files,examples=index,
        selection='first identity, median and P90 baseline Hb error; distinct subjects; EEG first native-order retained electrode of same trial',
        source='voltage_summary.json and frozen checkpoints; no fitted graphical smoothing'))
    return files


def eeg_residual_case_metrics(prediction, target, *, hop=20):
    """Complete-window O errors in the frozen raw-observation training units."""
    x, y = np.asarray(prediction, float), np.asarray(target, float)
    if x.shape != y.shape or x.ndim != 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('aligned finite [case,time] residual waveforms required')
    xc, yc = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    denominator = np.sqrt(np.sum(xc * xc, axis=1) * np.sum(yc * yc, axis=1))
    correlation = np.divide(np.sum(xc * yc, axis=1), denominator,
                            out=np.full(len(x), np.nan), where=denominator > 1e-12)
    difference = np.diff(x - y, axis=1)
    return dict(NMSE=np.mean((x - y) ** 2, axis=1),
                zero_NMSE=np.mean(y * y, axis=1),
                prediction_power=np.mean(x * x, axis=1),
                centered_correlation=correlation,
                signed_projection=np.mean(x * y, axis=1),
                boundary_error_NMSE=np.mean(difference[:, hop-1::hop] ** 2, axis=1),
                derivative_error_NMSE=np.mean(difference ** 2, axis=1))


def eeg_residual_aggregate(metrics, rows):
    """Equal records per subject, then equal subjects; ratios after averaging.

    Undefined correlations remain counted and are excluded only from that
    metric's mean. Zero-prediction MSE and predicted power are retained so a
    small error in raw-EEG units cannot masquerade as accurate residuals.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        groups[(row['dataset'], row['split'], row['subject'])][row['record']].append(i)
    def finite_mean(values):
        values = np.asarray(values, float)
        valid = np.isfinite(values)
        return float(values[valid].mean()) if valid.any() else None
    def ratios(record):
        denominator = record['zero_NMSE']
        record['zero_prediction_skill'] = 1 - record['NMSE'] / denominator if denominator > 0 else None
        record['RMS_ratio'] = float(np.sqrt(record['prediction_power'] / denominator)) if denominator > 0 else None
        record['signed_projection_ratio'] = record['signed_projection'] / denominator if denominator > 0 else None
        return record
    subjects = []
    for (dataset, split, subject), records in sorted(groups.items()):
        indices = [i for group in records.values() for i in group]
        row = dict(dataset=dataset, split=split, subject=subject, records=len(records), cases=len(indices),
                   undefined_correlations=int(np.sum(~np.isfinite(metrics['centered_correlation'][indices]))))
        row.update({name: finite_mean([finite_mean(value[group]) for group in records.values()])
                    for name, value in metrics.items()})
        subjects.append(ratios(row))
    datasets = []
    for dataset, split in sorted({(r['dataset'], r['split']) for r in subjects}):
        selected = [r for r in subjects if (r['dataset'], r['split']) == (dataset, split)]
        row = dict(dataset=dataset, split=split, subjects=len(selected), cases=sum(r['cases'] for r in selected),
                   undefined_correlations=sum(r['undefined_correlations'] for r in selected))
        row.update({name: finite_mean([r[name] for r in selected]) for name in metrics})
        datasets.append(ratios(row))
    return subjects, datasets


def summarize_eeg_o_run(run, output):
    """Read completed subject tables and export paired, descriptive evidence."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    run, output = Path(run), Path(output)
    result = json.loads((run/'eeg_o_summary.json').read_text())
    if result['status'] != 'completed':raise ValueError('complete EEG-O producer required')
    output.mkdir(parents=True, exist_ok=True)
    parent=Path(result['source_parent']);candidate=result['selection']['candidate']
    sources=[];paired=[];seeds=[]
    for job in result['selected_results']:
        seed=job['seed'];seeds.append(seed)
        new_path=run/'eeg_o_jobs'/candidate/f'seed_{seed}'/'subjects.csv'
        old_path=parent/'voltage_jobs/eeg_voltage'/f'seed_{seed}'/'evaluation/vq4_subjects.csv'
        sources.extend([str(new_path),str(old_path)])
        with new_path.open() as stream:new=list(csv.DictReader(stream))
        with old_path.open() as stream:old=list(csv.DictReader(stream))
        lookup={(r['dataset'],r['subject'],r['component'],r['metric']):r for r in old
                if r['split']=='validation' and r['signal']=='joint'}
        for row in new:
            if row['variant']!='hard':continue
            if row['split']!='validation':raise ValueError('non-validation row in EEG-O report')
            key=(row['dataset'],row['subject']);base=lookup[(*key,'O','NMSE')]
            if int(base['cases'])!=int(row['cases']):raise ValueError('unmatched residual case population')
            record=dict(dataset=key[0],subject=key[1],seed=seed,cases=int(row['cases']),records=int(row['records']))
            record.update({k:float(row[k]) for k in ['NMSE','zero_NMSE','prediction_power','centered_correlation',
                'signed_projection','boundary_error_NMSE','derivative_error_NMSE','frozen_S_plus_O_NMSE']})
            record['old_NMSE']=float(base['value'])
            record['old_correlation']=float(lookup[(*key,'O','centered_correlation')]['value'])
            record['old_sum_NMSE']=float(lookup[(*key,'sum','NMSE')]['value'])
            paired.append(record)
    groups=defaultdict(list)
    for row in paired:groups[(row['dataset'],row['subject'])].append(row)
    subjects=[]
    for (dataset,subject),rows in sorted(groups.items()):
        if {r['seed'] for r in rows}!=set(seeds):raise ValueError('missing paired optimization seed')
        record=dict(dataset=dataset,subject=subject,cases=rows[0]['cases'],records=rows[0]['records'],seeds=len(rows))
        record.update({k:float(np.mean([r[k] for r in rows])) for k in rows[0]
                       if k not in ('dataset','subject','seed','cases','records')})
        subjects.append(record)
    comparisons=[]
    for dataset in sorted({r['dataset'] for r in subjects}):
        rows=[r for r in subjects if r['dataset']==dataset]
        average={k:float(np.mean([r[k] for r in rows])) for k in rows[0]
                 if k not in ('dataset','subject','cases','records','seeds')}
        new=[dict(dataset=dataset,subject=r['subject'],NMSE=r['NMSE']) for r in rows]
        old=[dict(dataset=dataset,subject=r['subject'],NMSE=r['old_NMSE']) for r in rows]
        paired_error=paired_subject_bootstrap(new,old,samples=5000,seed=20261029)
        comparisons.append(dict(dataset=dataset,subjects=len(rows),cases=sum(r['cases'] for r in rows),**average,
            old_to_zero_ratio=average['old_NMSE']/average['zero_NMSE'],
            new_to_zero_ratio=average['NMSE']/average['zero_NMSE'],
            relative_error_reduction_percent=100*(1-average['NMSE']/average['old_NMSE']),
            zero_prediction_skill=1-average['NMSE']/average['zero_NMSE'],
            RMS_ratio=float(np.sqrt(average['prediction_power']/average['zero_NMSE'])),
            signed_projection_ratio=average['signed_projection']/average['zero_NMSE'],paired_error=paired_error))
    _csv(output/'paired_subject_seeds.csv',paired);_csv(output/'paired_subjects.csv',subjects)
    cfg=json.loads((run/'resolved_config.json').read_text())['config']
    summary=dict(source_run=str(run),producer_summary=str(run/'eeg_o_summary.json'),candidate=candidate,seeds=seeds,
        source_tables=sources,comparisons=comparisons,
        training_budget=[dict(candidate=r['candidate']['name'],seed=r['seed'],step=r['step'],budget=r['budget'])
                         for r in json.loads((run/'eeg_o_screen_jobs.json').read_text())['jobs']
                         +json.loads((run/'eeg_o_replication_jobs.json').read_text())['jobs']],
        planned_trials=result['planned_trials'],retained_trials=result['retained_trials'],
        input_support=cfg['eeg_o']['raw_input_support'],old_input_support='one electrode, local 2 s raw voltage',
        bits_per_O_second_per_electrode=280,test_used=False,
        aggregation='equal cases within record, equal records within subject, seeds averaged within subject, equal subjects; ratios calculated after averaging',
        inference='paired whole-subject percentile95% interval; validation-selected candidate, fixed scales and selection, no multiplicity correction or independent confirmation')
    _json(output/'summary.json',summary)
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans'})
    fig,axes=plt.subplots(len(comparisons),3,figsize=(14,7),squeeze=False)
    fig.subplots_adjust(left=.075,right=.985,bottom=.20,top=.82,hspace=.45,wspace=.28)
    titles=['O MSE / zero-prediction MSE (lower)','O centered correlation (higher)','Frozen S + O NMSE (lower)']
    for i,row in enumerate(comparisons):
        names=('old_to_zero_ratio','new_to_zero_ratio'),('old_correlation','centered_correlation'),('old_sum_NMSE','frozen_S_plus_O_NMSE')
        for j,pair in enumerate(names):
            ax=axes[i,j];ax.bar([0,1],[row[k] for k in pair],color=['#888888','#0072B2'],width=.55)
            for seed_index,seed in enumerate(seeds):
                data=[r for r in paired if r['dataset']==row['dataset'] and r['seed']==seed]
                if j==0:vals=[np.mean([r['old_NMSE'] for r in data])/np.mean([r['zero_NMSE'] for r in data]),np.mean([r['NMSE'] for r in data])/np.mean([r['zero_NMSE'] for r in data])]
                else:vals=[np.mean([r[k] for r in data]) for k in pair]
                ax.scatter(np.array([0,1])+(seed_index-1)*.08,vals,color='black',s=20,marker=['o','^','s'][seed_index],zorder=3)
            if j==0:ax.axhline(1,color='#333',ls='--',lw=1,label='Zero prediction')
            ax.set_xticks([0,1],['Original','New']);ax.set_title(titles[j]);ax.grid(axis='y',alpha=.2)
            for x,k in enumerate(pair):ax.annotate(f'{row[k]:.4g}',(x,row[k]),xytext=(0,5),textcoords='offset points',ha='center')
            if j==0:
                short='Single-Trial' if row['dataset']=='eeg_fnirs_single_trial' else 'Simultaneous'
                ax.set_ylabel(f'{short}\n{row["subjects"]} validation subjects')
            ax.margins(y=.20)
            if j==0:ax.set_ylim(0,1.2)
            elif j==1:ax.set_ylim(0,.5)
            else:ax.set_ylim(0,1.2*max(max(r['old_sum_NMSE'],r['frozen_S_plus_O_NMSE']) for r in paired))
    fig.suptitle(f'EEG-O refinement: {candidate} | fixed targets and O bitrate | three optimization seeds',fontsize=15,y=.98)
    fig.text(.075,.90,'Bars: equal-subject means after seed averaging. Circle/triangle/square: seeds 17/29/43 on the same subjects; not confidence intervals.',fontsize=10)
    support='New: retained 90 s EEG reference/filter support; central 20 s observed proxy.' if cfg['eeg_o'].get('architecture')=='projection_gain' else 'New: full 20 s EEG filtering; gain-network context is shorter.'
    fig.text(.075,.115,'Original: single-electrode 2 s input. '+support,fontsize=10)
    fig.text(.075,.075,'O = fixed conditional voltage correction residual. Both hard O decoders use 280 bit/s/electrode; no measured-signal or gain sidecar.',fontsize=10)
    fig.text(.075,.035,'Identical development subjects and targets; input support differs. Paired subject intervals and all source tables are retained in summary.json.',fontsize=10)
    fig.savefig(output/'eeg_o_comparison.png',dpi=250);plt.close(fig)
    return summary


def render_state_hb_pattern_detail(folder, parent_folder, export):
    """Separate chromophores using retained real assignments and paired decoded arrays."""
    import html
    import inspect
    import shlex
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    folder, parent_folder, export = map(lambda p: Path(p).resolve(), (folder, parent_folder, export))
    read = lambda p: json.loads(p.read_text())
    provenance, parent_provenance = [read(p/'provenance.json') for p in (folder, parent_folder)]
    metadata = read(folder/'samples.json')
    if metadata != read(parent_folder/'samples.json'):
        raise ValueError('paired pattern samples must have identical ordered identities')
    for p in (provenance, parent_provenance):
        if p['checkpoint_signature']['modality'] != 'hb' or p['rate'] != 10:
            raise ValueError('retained 10 Hz Hb diagnostics required')
        if p['config']['data']['scope'] != 'exact_parent_four_training_subjects_five_windows':
            raise ValueError('retained four-subject development panel required')
    if export.exists():
        raise ValueError('fresh export required')
    with np.load(folder/'patterns.npz', allow_pickle=False) as a:
        arrays = {k: a[k] for k in a.files}
    with np.load(parent_folder/'patterns.npz', allow_pickle=False) as a:
        parent = {k: a[k] for k in a.files}
    for key in ('x', 's', 'o', 'mask'):
        if not np.array_equal(arrays[key], parent[key]):
            raise ValueError('paired target, scale and mask identity required')
    thresholds = read(folder/'thresholds.json')['values']
    rate = provenance['rate']; hop = 2*rate
    contexts, samples, colors = arrays['x'].shape
    anchors = samples//hop
    if colors != 2 or samples != 30*rate or len(metadata) != contexts:
        raise ValueError('30-second paired-color contexts required')
    roles = np.repeat([r['role'] for r in metadata], anchors)
    datasets = np.repeat([r['dataset'] for r in metadata], anchors)
    valid = arrays['mask'].reshape(-1, hop, colors).all((1, 2))
    train = valid & (roles == 'train')
    size = provenance['codebook_size']
    config = provenance['config']; signature = provenance['checkpoint_signature']
    candidate_label = f"R{config['round']}" if 'round' in config else signature['name']
    shape_weight = config['architectures'][signature['name'].split('__')[0]].get('shape_weight', 0.)
    selections, records, figures, payload = {}, [], [], {}
    time_s = np.arange(hop)/rate
    # All displayed and summarized patches have complete masks; partial patches stay
    # in the owning full-waveform evaluation and are counted in this export.
    def features(x):
        return state_patch_features(x.reshape(-1, colors), np.ones((x.size//colors, colors), bool), rate)

    export.mkdir(parents=True)
    style = {'font.family': 'Noto Sans CJK JP', 'font.size': 11, 'axes.unicode_minus': False}
    dataset_names = {'eeg_fnirs_single_trial': 'Single-Trial', 'refed': 'REFED',
                     'simultaneous_eeg_nirs': 'Simultaneous', 'visual_cognitive_motivation': 'Visual'}
    with plt.rc_context(style):
        for branch in BRANCHES:
            key = branch.lower(); ids = arrays['ids_'+branch].ravel()
            if len(ids) != len(train) or np.any((ids < 0) | (ids >= size)):
                raise ValueError('code IDs outside retained codebook')
            target = arrays[key].reshape(-1, hop, colors).astype(float)
            decoded = arrays['hat_'+key].reshape(-1, hop, colors).astype(float)
            baseline = parent['hat_'+key].reshape(-1, hop, colors).astype(float)
            if not all(np.isfinite(x[valid]).all() for x in (target, decoded, baseline)):
                raise ValueError('nonfinite complete patch')
            f, h, b = map(features, (target, decoded, baseline))
            counts = np.bincount(ids[train], minlength=size)
            medians = np.array([np.median(f['slope'][train & (ids == code)], axis=0)
                                if counts[code] else [0., 0.] for code in range(size)])
            choices = [('高频码', counts), ('HbO 上升', medians[:, 0]),
                       ('HbO 下降', -medians[:, 0]), ('HbR 强趋势', abs(medians[:, 1]))]
            selected = []
            for reason, score in choices:
                pool = [code for code in range(size) if counts[code] >= 30 and code not in selected]
                if not pool:
                    raise ValueError('insufficient supported distinct codes')
                selected.append(max(pool, key=lambda code: score[code]))
            selections[branch] = []
            for code, (reason, _) in zip(selected, choices):
                take = np.flatnonzero(train & (ids == code))
                membership = []
                for ix in take:
                    context, patch = divmod(int(ix), anchors)
                    membership.append(dict(flat_patch_index=int(ix), context_index=context, anchor_index=patch,
                        patch_start_in_parent_s=metadata[context]['start_s']+patch*2, **metadata[context]))
                ds_counts = {d: int(np.sum(datasets[take] == d)) for d in sorted(set(datasets))}
                selections[branch].append(dict(code=code, reason=reason, n=len(take),
                    datasets=ds_counts, median_target_slope=medians[code], patches=membership))
                for c, color in enumerate(('HbO', 'HbR')):
                    th = np.array([thresholds[d][key]['slope_2'][c] for d in datasets[take]])
                    slopes = f['slope'][take, c]; up = int(np.sum(slopes > th)); down = int(np.sum(slopes < -th))
                    item = dict(branch=branch, code=code, color=color, n=len(take),
                        up=up, near_zero=len(take)-up-down, down=down,
                        target_slope_mean=float(slopes.mean()), target_slope_median=float(np.median(slopes)),
                        decoded_slope_mean=float(h['slope'][take, c].mean()),
                        parent_slope_mean=float(b['slope'][take, c].mean()),
                        cancellation=float(abs(slopes.mean())/max(np.mean(abs(slopes)), 1e-30)))
                    for label, values, feat in [('target', target, f), ('decoded', decoded, h), ('parent', baseline, b)]:
                        for suffix, z in [('raw', values[take, :, c]), ('centered', feat['centered'][take, :, c])]:
                            payload[f'{branch}_{code}_{color}_{label}_{suffix}_mean'] = z.mean(0)
                            payload[f'{branch}_{code}_{color}_{label}_{suffix}_p10_p90'] = np.quantile(z, [.1, .9], axis=0)
                        payload[f'{branch}_{code}_{color}_{label}_slope'] = feat['slope'][take, c]
                    records.append(item)
            for kind, title in [('mean', '形态均值与离散范围'), ('centered', '全部片段的形态'),
                                ('raw', '原幅度与水平'), ('slopes', '码内斜率分布')]:
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                fig.subplots_adjust(left=.07, right=.985, top=.77, bottom=.15, hspace=.55, wspace=.30)
                for j, code in enumerate(selected):
                    take = np.flatnonzero(train & (ids == code))
                    info = selections[branch][j]
                    for c, (color, ink) in enumerate([('HbO', '#bd3836'), ('HbR', '#2166ac')]):
                        ax = axes[c, j]
                        stat = next(r for r in records if (r['branch'], r['code'], r['color']) == (branch, code, color))
                        if kind == 'slopes':
                            z, zhat, zparent = f['slope'][take, c], h['slope'][take, c], b['slope'][take, c]
                            lo, hi = min(z.min(), zhat.min(), zparent.min()), max(z.max(), zhat.max(), zparent.max())
                            bins = np.linspace(lo-.02*(hi-lo+1e-12), hi+.02*(hi-lo+1e-12), 22)
                            ax.hist(z, bins=bins, color=ink, alpha=.25)
                            ax.hist(zhat, bins=bins, histtype='step', color=ink, linewidth=1.7)
                            ax.hist(zparent, bins=bins, histtype='step', color='#666666', linestyle='--', linewidth=1.3)
                            ax.axvline(0, color='black', lw=.6)
                            ax.set(xlabel='2 秒 OLS 斜率（训练 X 尺度 / 秒）', ylabel='片段数')
                            note = f"目标：上升 {stat['up']} / 近零 {stat['near_zero']} / 下降 {stat['down']}"
                        else:
                            values = [target, decoded, baseline] if kind == 'raw' else [f['centered'], h['centered'], b['centered']]
                            actual, hat, old = [v[take, :, c] for v in values]
                            if kind != 'mean': ax.plot(time_s, actual.T, color='#888888', alpha=.12, lw=.6)
                            interval = np.quantile(actual, [.1, .9], axis=0)
                            ax.fill_between(time_s, *interval, color=ink, alpha=.14)
                            ax.plot(time_s, actual.mean(0), color='#151515', lw=2.1)
                            ax.plot(time_s, hat.mean(0), color=ink, lw=2.0, ls='--')
                            ax.plot(time_s, old.mean(0), color='#666666', lw=1.5, ls=':')
                            if kind != 'raw': ax.axhline(0, color='black', lw=.5, alpha=.4)
                            # The overview includes all real patches; the mean view shows
                            # complete means and P10-P90 without drawing individual traces.
                            support = interval.ravel() if kind == 'mean' else actual.ravel()
                            limits = np.concatenate([support, actual.mean(0), hat.mean(0), old.mean(0)])
                            low, high = limits.min(), limits.max(); pad = max((high-low)*.07, 1e-6)
                            ax.set(ylim=(low-pad, high+pad), xlim=(0, 2), xlabel='片段内时间（秒）',
                                   ylabel=color+' / 训练 X 尺度')
                            note = f"目标斜率均值 {stat['target_slope_mean']:+.3f} / 秒"
                        ax.set_title(f"Hb-{branch} · ID {code} · {color}\n{info['reason']} · n = {len(take)}", fontsize=12)
                        ax.text(.5, -.32, note, transform=ax.transAxes, ha='center', fontsize=9)
                        ax.grid(alpha=.16)
                if kind == 'slopes':
                    handles = [Patch(color='#777777', alpha=.25, label='目标片段'),
                        Line2D([], [], color='#333333', label=candidate_label+' 整数重建'),
                        Line2D([], [], color='#666666', ls='--', label='父版本整数重建')]
                else:
                    handles = [Line2D([], [], color='#888888', alpha=.45, label='每条灰线：一个真实目标片段'),
                        Patch(color='#777777', alpha=.2, label='目标 P10–P90'),
                        Line2D([], [], color='#151515', lw=2, label='目标条件均值'),
                        Line2D([], [], color='#bd3836', ls='--', label=candidate_label+' 重建均值（HbO 红 / HbR 蓝）'),
                        Line2D([], [], color='#666666', ls=':', label='父版本重建均值')]
                    if kind == 'mean': handles = handles[1:]
                fig.suptitle(f'Hb-{branch} token pattern｜{title}｜{candidate_label} · shape {shape_weight} · K{size} · latent {provenance["latent_dim"]}',
                             x=.07, ha='left', y=.98, fontsize=19, fontweight='bold')
                fig.text(.07, .925, '同列 HbO / HbR 共用一个 ID；分支完整 30 秒、15 个 ID 联合解码。父版本比较使用相同片段，不匹配数字 ID。', fontsize=11)
                fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.063, .895), ncol=3, frameon=False, fontsize=10)
                footer = ('逐片段、逐色团去均值；此页显示均值与 P10–P90，全部片段见形态总览。' if kind == 'mean' else
                          '逐片段、逐色团减去自身时间均值，仅用于此形态图。' if kind == 'centered' else
                          '保留训练尺度下的原幅度与偏置。' if kind == 'raw' else
                          'OLS 使用真实秒数；近零阈值来自各库训练目标绝对斜率的 P10。')
                fig.text(.07, .03, footer+' 各面板独立坐标；统计为码内片段等权，P10–P90 表示片段离散度。\n'
                         '仅训练前缀、完整掩码、非重叠 30 秒上下文；片段/通道共享被试，n 为片段数。选码与各库组成见图册。', fontsize=10)
                name = f'Hb_{branch}_{kind}.png'; fig.savefig(export/name, dpi=220); plt.close(fig)
                figures.append(dict(path=name, branch=branch, view=kind, codes=selected,
                    provenance='provenance.json', selection='selection.json', statistics='statistics.csv'))
    _json(export/'selection.json', selections)
    _csv(export/'statistics.csv', records)
    np.savez_compressed(export/'pattern_statistics.npz', time_s=time_s, **payload)
    _json(export/'figure_index.json', figures)
    manifest = dict(status='completed', source_export=str(folder), parent_export=str(parent_folder),
        candidate=provenance, parent=parent_provenance,
        source_pair_check='ordered sample identities, x/s/o targets and masks exactly equal',
        training_complete_patches=int(train.sum()), training_incomplete_patches=int(((roles == 'train') & ~valid).sum()),
        heldout_patches_not_used_for_selection=int((roles != 'train').sum()),
        selection_rule='sequential distinct codes with >=30 complete training patches: largest count, '
            'largest median HbO slope, smallest median HbO slope, largest absolute median HbR slope; ties lowest ID',
        weighting='equal patches conditional on each code; not dataset balanced or independent-subject statistics',
        curve_sources=['conditional cached S/O target', 'complete candidate integer sequence decode',
                      'complete parent integer sequence decode on identical samples'],
        renderer=str(Path(__file__).resolve()),
        renderer_sha256=hashlib.sha256(inspect.getsource(render_state_hb_pattern_detail).encode()).hexdigest())
    _json(export/'provenance.json', manifest)
    expression = ('from src.analysis.dual_component_tokenizer import render_state_hb_pattern_detail; '
                  f'render_state_hb_pattern_detail({str(folder)!r}, {str(parent_folder)!r}, "FRESH_EXPORT_PATH")')
    _json(export/'reproduce.json', dict(cwd=str(Path(__file__).resolve().parents[2]),
        command='.venv/bin/python -c '+shlex.quote(expression), note='Replace FRESH_EXPORT_PATH with a new export path.'))
    body = [f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>{candidate_label} HbO/HbR token patterns</title>',
        '<style>body{max-width:1500px;margin:35px auto;padding:0 24px;font:18px/1.6 sans-serif;color:#17242e}'
        'img{width:100%}table{border-collapse:collapse}td,th{padding:7px 14px;border-bottom:1px solid #ddd}'
        'a{color:#17659b}</style>'+f'<h1>{candidate_label}：HbO / HbR token pattern</h1>',
        f'<p>选用形态损失权重 {shape_weight} 的终点模型（seed {signature["seed"]}）。每分支 K{size}、latent {provenance["latent_dim"]}，每 2 秒一个 ID；'
        '同一 ID 同时表示 HbO/HbR。灰线是赋到此码的条件 S/O 目标片段；彩色虚线是完整整数序列解码后切出的片段均值。'
        '它描述真实赋码条件下的形态分布，邻近 ID 也参与解码。</p>',
        '<p>选码仅使用训练前缀，要求至少 30 个完整片段，依次选高频码、HbO 斜率中位数最大/最小码、'
        '剩余码中 HbR 斜率中位数绝对值最大码。所有曲线均值及区间使用码内片段等权；'
        '各数据集份额如下，不能把不同码的组成差异当作跨被试规律。两行独立纵轴避免色团幅度差异遮蔽细节。</p>',
        '<p><a href="provenance.json">模型与来源</a> · <a href="selection.json">全部片段身份</a> · '
        '<a href="statistics.csv">斜率统计</a> · <a href="pattern_statistics.npz">图中数值</a> · '
        '<a href="reproduce.json">复现命令</a></p><table><tr><th>分支 / ID</th><th>选择依据</th><th>片段数</th>']
    body.extend(f'<th>{label}</th>' for label in dataset_names.values()); body.append('</tr>')
    for branch, choices in selections.items():
        for item in choices:
            body.append(f'<tr><td>{branch} / {item["code"]}</td><td>{item["reason"]}</td><td>{item["n"]}</td>'+''.join(
                f'<td>{item["datasets"].get(ds, 0)}</td>' for ds in dataset_names)+'</tr>')
    body.append('</table>')
    for figure in figures:
        name = html.escape(figure['path']); body.append(f'<p><a href="{name}">打开原分辨率图片</a></p><img src="{name}" alt="{name}">')
    body.append('</html>'); (export/'index.html').write_text('\n'.join(body), encoding='utf-8')
    return dict(status='completed', export=str(export), figures=len(figures), selected_codes={
        branch: [r['code'] for r in items] for branch, items in selections.items()})


def render_state_dynamic_patterns(folder, arrays, metadata, thresholds, provenance):
    """Lightweight real-assignment atlas; no embedding fit or cross-model ID matching."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    folder=Path(folder);rate=provenance['rate'];channels=arrays['x'].shape[-1];hop=2*rate
    roles=np.repeat([r['role'] for r in metadata],15);datasets=np.repeat([r['dataset'] for r in metadata],15)
    masks=arrays['mask'].reshape(-1,hop,channels);valid=masks.all((1,2));train=(roles=='train')&valid
    figures=[];all_stats=[]
    def features(patches):
        return state_patch_features(patches.reshape(-1,channels),np.ones((len(patches)*hop,channels),bool),rate)
    for branch in ('S','O'):
        key=branch.lower();ids=arrays['ids_'+branch].ravel();target=arrays[key].reshape(-1,hop,channels)
        prediction=arrays['hat_'+key].reshape(-1,hop,channels);f=features(target);h=features(prediction)
        size=provenance['codebook_size'];top=np.argsort(-np.bincount(ids[train],minlength=size),kind='stable')[:4]
        for role in ('train','within_subject_holdout'):
            for ds in ['pooled',*sorted(set(datasets))]:
                select=valid&(roles==role)&((datasets==ds) if ds!='pooled' else True)
                for code in range(size):
                    take=select&(ids==code)
                    for c in range(channels):
                        item=dict(branch=branch,role=role,dataset=ds,code=code,color=c,n=int(take.sum()))
                        if take.any():
                            for label,values in [('target',f),('integer_decode',h)]:
                                for stat in ('level','slope','rms'):
                                    z=values[stat][take,c];item[label+'_'+stat+'_quantiles']=np.quantile(z,[0,.1,.5,.9,1]).tolist()
                            slopes=f['slope'][take,c]
                            th=np.array([thresholds[d][key]['slope_2'][c] for d in datasets[take]])
                            up=slopes>th;down=slopes < -th
                            item.update(up=int(up.sum()),down=int(down.sum()),near_zero=int((~up&~down).sum()),
                                opposite_sign_mixing=float(2*min(up.sum(),down.sum())/max(up.sum()+down.sum(),1)),
                                cancellation=float(abs(slopes.mean())/max(np.mean(abs(slopes)),1e-30)))
                        all_stats.append(item)
        fig,axes=plt.subplots(4,4,figsize=(16,12),constrained_layout=True)
        t=np.arange(hop)/rate
        for j,code in enumerate(top):
            take=np.flatnonzero(train&(ids==code));show=take[np.linspace(0,len(take)-1,min(20,len(take))).round().astype(int)]
            for c,col in enumerate(['#0072B2','#D55E00'][:channels]):
                color_name=('HbO','HbR')[c] if channels==2 else 'Voltage'
                for ix in show:
                    axes[0,j].plot(t,target[ix,:,c],color=col,alpha=.13,lw=.6)
                    axes[1,j].plot(t,f['centered'][ix,:,c],color=col,alpha=.13,lw=.6)
                for row,actual,hat in [(0,target,prediction),(1,f['centered'],h['centered'])]:
                    axes[row,j].plot(t,actual[take,:,c].mean(0),color=col,lw=2,label=color_name+' target')
                    axes[row,j].plot(t,hat[take,:,c].mean(0),color=col,lw=1.5,ls='--',label=color_name+' integer decode')
                    axes[row,j].fill_between(t,np.quantile(actual[take,:,c],.1,axis=0),np.quantile(actual[take,:,c],.9,axis=0),color=col,alpha=.08)
                axes[2,j].hist(f['level'][take,c],bins=20,color=col,alpha=.45,label=color_name)
                axes[3,j].hist(f['slope'][take,c],bins=20,color=col,alpha=.45,label=color_name)
            axes[0,j].set_title(f'{branch} code {code} | n={len(take)} train patches')
            axes[0,j].set(xlabel='Within patch (s)',ylabel='Original amplitude / training X scale')
            axes[1,j].set(xlabel='Within patch (s)',ylabel='Patch-centered amplitude (display only)')
            axes[2,j].set(xlabel='Target patch level',ylabel='Patches')
            axes[3,j].set(xlabel='Target OLS slope (training X scale / s)',ylabel='Patches');axes[3,j].axvline(0,color='black',lw=.6)
            for ax in axes[:,j]:ax.grid(alpha=.15)
        axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
        fig.suptitle(f"{folder.parent.parent.name} {folder.parent.name} {branch} | K={size}, latent={provenance['latent_dim']}\nReal assignments; nonoverlapping training contexts; mean, P10–P90, up to 20 real patches; each panel own y scale",fontsize=12)
        path=folder/f'patterns_{branch}.png';fig.savefig(path,dpi=160);plt.close(fig);figures.append(str(path))
    _json(folder/'code_pattern_statistics.json',all_stats)
    _json(folder/'figure_index.json',[dict(path=p,provenance='provenance.json',samples='samples.json',statistics='code_pattern_statistics.json',curve_sources=['conditional target','full integer sequence decode']) for p in figures])
    return figures


def render_state_diagnostic_overview(export, summary, *, write_html=True):
    import html
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    export=Path(export);records=summary['records']
    fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    for i,modality in enumerate(('eeg','hb')):
        rs=[r for r in records if r['modality']==modality]
        for j,field in enumerate(('NMSE','shape_nmse','slope_2_dynamic_nmse')):
            ax=axes[i,j];x=np.arange(len(rs))
            for off,key,color in [(-.25,'s','#0072B2'),(0,'o','#D55E00'),(.25,'x','#009E73')]:
                vals=[r['metrics']['macro']['within_subject_holdout'][key][field] for r in rs]
                ax.bar(x+off,vals,.25,label=key.upper(),color=color)
            short={'dense_long':'dense long','waveform':'waveform','slow_encoder':'LR x0.1',
                   'fast_ema':'EMA 0.95','low_commitment':'commit 0.05','shape_loss':'shape 0.2','shape_loss_stronger':'shape 0.5'}
            ax.set(xticks=x,xticklabels=[short.get(r['architecture'],r['architecture'])+'\n'+r['candidate'] for r in rs],title=modality.upper()+' | '+field,ylabel='Error; lower is better')
            ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.15);ax.legend()
    fig.suptitle('Frozen development panel | positions, records, then datasets equally | all original positions retained')
    fig.savefig(export/'overview.png',dpi=180);plt.close(fig)
    if not write_html:return
    body=['<html><meta charset="utf-8"><title>Tokenizer dynamic diagnostics</title><style>body{font:16px sans-serif;max-width:1500px;margin:auto}img{max-width:100%}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style>',
        '<h1>固定开发面板：重建、动态与真实赋码</h1><p>四名训练被试、五父窗；此前查看过的时间留出。所有目标、尺度、mask保持。NMSE使用各分量冻结训练方差。D_shape仅在超过训练P10动态阈值的patch计算；完整分母和近零片段见dynamic_rows.json。pattern来自当前checkpoint的真实赋码，预测来自完整本分支整数序列。ID不跨模型对应。</p>',
        '<img src="overview.png"><table><tr><th>Model</th><th>S/O/X NMSE</th><th>S/O/X D_shape</th><th>Health</th><th>Evidence</th></tr>']
    for r in records:
        folder=Path(r['folder']);relative=folder.relative_to(export);m=r['metrics']['macro']['within_subject_holdout']
        body.append('<tr><td>'+html.escape(r['modality']+' '+r['architecture']+' '+r['candidate'])+'</td><td>'+ ' / '.join(f"{m[k]['NMSE']:.4f}" for k in ('s','o','x'))+'</td><td>'+ ' / '.join(f"{m[k]['D_shape']:.3f}" for k in ('s','o','x'))+f'</td><td>{r["code_health_pass"]}</td><td><a href="{relative}/diagnostics.json">metrics</a> · <a href="{relative}/provenance.json">provenance</a></td></tr>')
    body.append('</table>')
    for r in records:
        for figure in r.get('figures',[]):
            body.append('<h2>'+html.escape(r['modality']+' '+r['architecture']+' '+r['candidate'])+'</h2><img loading="lazy" src="'+str(Path(figure).relative_to(export))+'">')
    body.append('</html>');(export/'index.html').write_text('\n'.join(body))


def render_state_hb_dynamics(export,key):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    export=Path(export);info=json.loads((export/(key+'.json')).read_text())
    with np.load(export/(key+'.npz')) as f:a=dict(f)
    fig,axes=plt.subplots(3,3,figsize=(16,10),constrained_layout=True);rate=info['rate'];t=np.arange(len(a['x']))/rate
    for i,component in enumerate(('s','o','x')):
        for label,prefix,col,style in [('Target','', '#343a40','-'),('Candidate','hard_','#0072B2','-'),('Parent','baseline_','#8b58a0',':'),('Continuous','continuous_','#D55E00','--')]:
            for c in range(2):
                y=a[prefix+component];f=state_patch_features(y,a['mask'],rate)
                # Two colors have independent panels in full-amplitude page; here linestyle identifies color.
                line=style if c==0 else (0,(1,2))
                axes[i,0].plot(t,np.where(a['mask'][:,c],f['centered'][:,:,c].ravel(),np.nan),color=col,ls=line,lw=.8,label=label+(' HbO' if c==0 else ' HbR'))
                for j,seconds in enumerate((2,6),1):
                    q=state_patch_features(y,a['mask'],rate,seconds)
                    axes[i,j].plot((np.arange(len(q['slope']))+.5)*seconds,q['slope'][:,c],color=col,ls=line,marker='.' if c==0 else 'x',ms=4,lw=.9)
        for j,ax in enumerate(axes[i]):
            ax.set(xlabel='Time in visible input (s)',ylabel='Training X scale'+(' / s' if j else ''),title=component.upper()+' | '+('2 s patch-centered shape' if j==0 else f'{[0,2,6][j]} s OLS slope'))
            for boundary in np.arange(2,30,2):ax.axvline(boundary,color='#aaa',lw=.35,ls=':')
            ax.axhline(0,color='black',lw=.5);ax.grid(alpha=.15)
    axes[0,0].legend(ncol=2,fontsize=7)
    fig.suptitle(f"{info['dataset']} {info['position']} | {info['hard_label']}\nPatch centering is diagnostic only; HbO solid/line, HbR dotted/x; full amplitude on adjacent page",fontsize=12)
    name=key+'_dynamics.png';fig.savefig(export/name,dpi=170);plt.close(fig)
    return dict(figure=name,title=info['dataset']+' '+info['position']+' 动态',caption='左列为每2秒片段减去该片段有效样本均值，仅用于形态展示；中右列为真实秒数的2秒/6秒OLS斜率。每条曲线来自完整30秒预测，未重拟合。HbO用原线型/点，HbR用点线/叉；所有曲线共用纵轴。')


def render_state_training_history(job,record):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    job=Path(job);h=json.loads((job/'summary.json').read_text())['history'];step=[r['step'] for r in h]
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for k,col in [('s','#0072B2'),('o','#D55E00'),('x','#009E73')]:
        axes[0,0].plot(step,[r['monitor'][k] for r in h],color=col,label=k.upper()+' hard')
        axes[0,0].plot(step,[r['continuous_bypass_diagnostic'][k] for r in h],color=col,ls='--',label=k.upper()+' bypass diagnostic')
    for b,col in [('S','#0072B2'),('O','#D55E00')]:
        axes[0,1].plot(step,[r['monitor_code_usage'][b]['effective'] for r in h],label=b+' effective',color=col)
        axes[1,0].plot(step,[r['monitor_code_usage'][b]['top1'] for r in h],label=b+' top1',color=col)
        axes[1,1].plot(step,[r['revivals'][b] for r in h],label=b+' cumulative revivals',color=col)
    for ax in axes.flat:ax.axvline(1500,color='black',ls=':',label='revival stop');ax.set_xlabel('Quantized continuation updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ax,title in zip(axes.flat,['Fixed training probe NMSE','Effective codes (fixed probe)','Maximum occupancy (fixed probe)','Cumulative revival events']):ax.set_title(title)
    fig.suptitle(record['modality']+' '+record['architecture']+' | final full-panel health is scored separately')
    fig.savefig(job/'training.png',dpi=180);plt.close(fig)


def state_diagnostic_synthetic_check(export):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    export=Path(export);export.mkdir(parents=True,exist_ok=True);rate=10;t=np.arange(300)/rate
    level=np.repeat(np.linspace(-2,2,15),20);ramp=t/5
    stair=np.repeat(ramp.reshape(15,20).mean(1),20)
    u=np.arange(20)/rate-.95;mixed=np.concatenate([u*(-1)**i for i in range(15)])
    cases=[('level-only constants',level,level),('smooth ramp vs flat patches',ramp,stair),('staircase vs smooth ramp',stair,ramp),('same code: opposite slopes',mixed,np.zeros(300))]
    fig,axes=plt.subplots(4,2,figsize=(13,10),constrained_layout=True);metrics=[]
    for i,(name,y,p) in enumerate(cases):
        m=np.ones((300,1),bool);th=dict(rms=[.001],slope_2=[.001],slope_6=[.001])
        result=state_dynamic_metrics(y[:,None],p[:,None],m,rate,[1],th);metrics.append(dict(case=name,metrics=result))
        axes[i,0].plot(t,y,label='Synthetic target');axes[i,0].plot(t,p,'--',label='Diagnostic prediction');axes[i,0].set(title=name,xlabel='s');axes[i,0].legend()
        f=state_patch_features(y[:,None],m,rate)
        for z in f['centered'][:,:,0]:axes[i,1].plot(np.arange(20)/rate,z,alpha=.2,color='#0072B2')
        # The fourth case uses exactly equal signs for its mean demonstration.
        shown=f['centered'][:14,:,0] if i==3 else f['centered'][:,:,0]
        axes[i,1].plot(np.arange(20)/rate,shown.mean(0),color='black',lw=2,label='Equal-sign mean' if i==3 else 'Mean')
        axes[i,1].set(title=f"patch-centered | boundary target energy fraction={result['differences'][0]['target_boundary_energy_fraction']:.3f}",xlabel='s');axes[i,1].legend()
        if np.max(np.abs(shown))<.01:axes[i,1].set_ylim(-.01,.01)
    assert metrics[0]['metrics']['channels'][0]['D_shape'] is None
    assert abs(metrics[1]['metrics']['channels'][0]['D_shape'])<1e-10
    assert abs(metrics[2]['metrics']['differences'][0]['target_boundary_energy_fraction']-1)<1e-12
    assert abs(np.stack([u,-u]).mean(0)).max()==0
    fig.suptitle('Synthetic tool checks only: levels, within-patch slopes, boundary steps, cancellation')
    fig.savefig(export/'synthetic_dynamics.png',dpi=160);plt.close(fig)
    _json(export/'synthetic_dynamics.json',dict(status='passed',measured_data=False,cases=metrics))
    return dict(status='passed',figure='synthetic_dynamics.png',cases=4)


def state_spectral_diagnostics(folder,rate=200):
    """Frozen waveform spectral errors on the same contiguous valid support."""
    folder=Path(folder);metadata=json.loads((folder/'samples.json').read_text())
    with np.load(folder/'patterns.npz') as f:a=dict(f)
    bands=[(0,1),(1,4),(4,8),(8,13),(13,30),(30,45),(45,100.0001)];records=[]
    for i,meta in enumerate(metadata):
        if meta['role']!='within_subject_holdout':continue
        for key in ('s','o','x'):
            mask=a['mask'][i,:,0];edges=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
            longest=int(np.argmax(ends-starts));begin,end=int(starts[longest]),int(ends[longest])
            y=a[key][i,begin:end,0].astype(float);p=a['hat_'+key][i,begin:end,0].astype(float)
            weight=np.hanning(len(y));Y=np.fft.rfft((y-y.mean())*weight);P=np.fft.rfft((p-p.mean())*weight)
            freq=np.fft.rfftfreq(len(y),1/rate);total=float(np.sum(abs(Y)**2));parts=[]
            for low,high in bands:
                take=(freq>=low)&(freq<high);target=float(np.sum(abs(Y[take])**2));prediction=float(np.sum(abs(P[take])**2));error=float(np.sum(abs(P[take]-Y[take])**2))
                parts.append(dict(band_hz=[low,min(high,100)],target_energy=target,prediction_energy=prediction,
                    amplitude_gain=float(np.sqrt(prediction/target)) if target>max(total*1e-8,1e-20) else None,
                    error_over_total_target=error/max(total,1e-30),target_fraction=target/max(total,1e-30)))
            records.append(dict(**meta,component=key,start_sample=begin,end_sample=end,bands=parts))
    _json(folder/'spectral_diagnostics.json',dict(rate_hz=rate,rule='longest common contiguous scored support per position, Hann taper, mean removed for spectral diagnostic only; errors normalized by full target spectral energy',rows=records))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    datasets=sorted({r['dataset'] for r in records});fig,axes=plt.subplots(3,len(datasets),figsize=(16,10),constrained_layout=True,squeeze=False)
    for i,key in enumerate(('s','o','x')):
        for j,ds in enumerate(datasets):
            group=[r for r in records if r['dataset']==ds and r['component']==key]
            record_names=sorted({r['record'] for r in group});values={}
            for name in ('target_energy','prediction_energy'):
                values[name]=np.mean([np.mean([[b[name] for b in r['bands']] for r in group if r['record']==record],axis=0) for record in record_names],axis=0)
            denominator=values['target_energy'].sum();ax=axes[i,j];x=np.arange(len(bands))
            ax.bar(x-.18,values['target_energy']/denominator,.36,label='Target',color='#343a40')
            ax.bar(x+.18,values['prediction_energy']/denominator,.36,label='Prediction',color='#0072B2')
            ax.set(xticks=x,xticklabels=['0–1','1–4','4–8','8–13','13–30','30–45','45–100'],xlabel='Hz',ylabel='Energy / total target energy',title=ds+' '+key.upper())
            ax.tick_params(axis='x',labelsize=7);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
    fig.suptitle(folder.parent.name+' EEG | fixed development support | unnormalized amplitudes preserved; each facet own y scale')
    fig.savefig(folder/'spectrum.png',dpi=160);plt.close(fig)
    return records
