"""Versioned communication export; only reads retained Suite 1 JSON/NPZ.

Preserves all previous figures. Failed profile nodes break curves and show
crosses; scatter/example plots distinguish nonconverged finite iterates.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

RUN = Path('/SSD_2/pid-mcm-implementation/experiments/runs/physiology_semantic_tokenizer/source_observability/20261021_v1')
OUT = RUN/'figures/readable_v3'
if OUT.exists() and len(list(OUT.glob('*.png'))) == 3:
    raise SystemExit('versioned export exists; preserve it')
OUT.mkdir(parents=True, exist_ok=True)
manifest = json.loads((RUN/'manifest.json').read_text())
records = [json.loads((RUN/'tasks'/f"{t['id']}.json").read_text()) for t in manifest['tasks']]
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 10,
                     'ytick.labelsize': 10, 'legend.fontsize': 10})

profile = [r for r in records if r['kind'] == 'profile' and r['tau_truth'] == 1.3
           and r['kappa_truth'] == .45 and r['seed'] == 2026102101]
fig, axes = plt.subplots(2, 2, figsize=(7, 6.8))
for col, parameter in enumerate(('tau', 'kappa')):
    for modality, color in (('Hb_only', '#0072B2'), ('EEG_Hb', '#D55E00')):
        for prior, style in ((0., '--'), (1., '-')):
            group = sorted([r for r in profile if r['profile_parameter'] == parameter
                and r['modality'] == modality and r['prior_multiplier'] == prior], key=lambda r: r['profile_value'])
            x = [r['profile_value'] for r in group]
            values = [[r['objective_components']['data'] for r in group],
                      [r['objective']-r['objective_components']['data'] for r in group]]
            for row in (0, 1):
                # A failed finite iterate is not a valid profiled minimum.
                y = [v if r['success'] else np.nan for r, v in zip(group, values[row])]
                axes[row, col].plot(x, y, marker='o', color=color, ls=style,
                    label=f'{modality}, penalty={prior:g}' if row == 0 else None)
                for r, v in zip(group, values[row]):
                    if not r['success']:
                        axes[row, col].scatter(r['profile_value'], v, marker='x', s=65, color='black', zorder=5)
    ticks = [.6, 1.3, 4.] if parameter == 'tau' else [.2, .45, 1.3]
    for row in (0, 1):
        ax = axes[row, col]
        ax.axvline(1.3 if parameter == 'tau' else .45, color='k', ls=':')
        ax.set_xscale('log')
        ax.set_xticks(ticks, [f'{v:g}' for v in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set(xlabel=parameter, ylabel='Data objective' if row == 0 else 'Separate prior penalty')
        ax.grid(alpha=.2)
handles, labels = axes[0, 0].get_legend_handles_labels()
handles.extend([Line2D([0], [0], ls=':', color='k'), Line2D([0], [0], marker='x', ls='', color='k')])
labels.extend(['Generator truth', 'Failed finite iterate; line broken'])
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10)
fig.suptitle('Nonlinear profiles: free nuisance refit at each node\nSame truth and observations; no interval claim')
fig.tight_layout(rect=(0, .2, 1, .91))
if not (OUT/'nonlinear_profiles.png').exists():
    fig.savefig(OUT/'nonlinear_profiles.png', dpi=240, bbox_inches='tight')
plt.close(fig)

recovery = [r for r in records if r['kind'] == 'recovery' and r['noise_multiplier'] == 1.]
priors = [0., .1, 1., 10.]
cmap = plt.get_cmap('viridis', 4)
fig, axes = plt.subplots(2, 1, figsize=(7, 7))
for modality, marker in (('Hb_only', 'o'), ('EEG_Hb', '^')):
    part = [r for r in recovery if r['modality'] == modality]
    for ax, key, ylabel in zip(axes, ('theta_log_rmse', 'driver_nrmse'),
                               ('Physiology log-error RMS', 'Driver NRMSE (fixed gauge)')):
        points = ax.scatter([r['metrics']['hb_clean_rmse'] for r in part], [r['metrics'][key] for r in part],
            c=[priors.index(r['prior_multiplier']) for r in part], marker=marker, cmap=cmap,
            vmin=-.5, vmax=3.5, s=28, alpha=.75)
        failed = [r for r in part if not r['success']]
        ax.scatter([r['metrics']['hb_clean_rmse'] for r in failed], [r['metrics'][key] for r in failed],
                   marker='x', s=50, linewidths=1., color='red')
        ax.set(xlabel='Clean Hb prediction RMSE (model units)', ylabel=ylabel)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
        ax.grid(alpha=.2)
handles = [Line2D([0], [0], marker='o', ls='', color='gray', label='Hb_only'),
           Line2D([0], [0], marker='^', ls='', color='gray', label='EEG_Hb'),
           Line2D([0], [0], marker='x', ls='', color='red', label='Nonconverged finite fit')]
axes[0].legend(handles=handles, loc='upper right', fontsize=10)
bar = fig.colorbar(points, ax=axes.tolist(), label='Explicit penalty multiplier', pad=.1)
bar.set_ticks([0, 1, 2, 3], labels=['0', '0.1', '1', '10'])
fig.suptitle(f'{len(recovery)} truth × seed × modality × penalty settings\nPaired variants; {sum(not r["success"] for r in recovery)} nonconverged finite fits marked red')
if not (OUT/'fit_versus_recovery.png').exists():
    fig.savefig(OUT/'fit_versus_recovery.png', dpi=240, bbox_inches='tight')
plt.close(fig)

part = sorted([r for r in recovery if r['tau_truth'] == 1.3 and r['kappa_truth'] == .45
               and r['seed'] == 2026102101 and r['modality'] == 'Hb_only'], key=lambda r: r['prior_multiplier'])
fig, axes = plt.subplots(3, 1, figsize=(7, 7.8), sharex=True)
legend, labels = [], []
for index, r in enumerate(part):
    with np.load(RUN/'tasks'/f"{r['id']}.npz", allow_pickle=False) as data:
        t = np.arange(len(data['prediction']))*.25
        if index == 0:
            truth, = axes[0].plot(t, data['truth_prediction'][:, 3], 'k--')
            axes[1].plot(t, data['truth_drivers'][:, 0], 'k--')
            axes[2].plot(t, data['truth_states'][:, 0, 2], 'k--')
            legend.append(truth)
            labels.append('Generator truth: tau=1.3, kappa=0.45')
        style = '-' if r['success'] else ':'
        line, = axes[0].plot(t, data['prediction'][:, 3], color=cmap(index), ls=style)
        axes[1].plot(t, data['drivers'][:, 0], color=cmap(index), ls=style)
        axes[2].plot(t, data['states'][:, 0, 2], color=cmap(index), ls=style)
        legend.append(line)
        labels.append(f"penalty={r['prior_multiplier']:g}; {'converged' if r['success'] else 'FAILED finite iterate'}")
for ax, ylabel in zip(axes, ('HbO (model units)', 'Driver (fixed gauge)', 'Relative flow f')):
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.2)
axes[-1].set_xlabel('Generator time (s)')
fig.legend(legend, labels, loc='lower center', ncol=1, fontsize=10)
fig.suptitle('Predetermined first noisy Hb-only case\nClose Hb waveforms can accompany different drivers')
fig.tight_layout(rect=(0, .18, 1, .94))
if not (OUT/'prediction_driver_state_example.png').exists():
    fig.savefig(OUT/'prediction_driver_state_example.png', dpi=240, bbox_inches='tight')
plt.close(fig)
print(json.dumps(dict(versioned_export='readable_v3', figures=[str(p.relative_to(RUN)) for p in sorted(OUT.glob('*.png'))],
    evidence='retained_arrays_only_no_fits_original_figures_preserved'), indent=2))
