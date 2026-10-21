"""Communication-only scatter export with external legend; no fits or mutations."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUN = Path('/SSD_2/pid-mcm-implementation/experiments/runs/physiology_semantic_tokenizer/source_observability/20261021_v1')
OUT = RUN/'figures/readable_v4'
OUT.mkdir(exist_ok=False)
manifest = json.loads((RUN/'manifest.json').read_text())
records = [json.loads((RUN/'tasks'/f"{t['id']}.json").read_text()) for t in manifest['tasks']]
rows = [r for r in records if r['kind'] == 'recovery' and r['noise_multiplier'] == 1.]
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 10,
                     'ytick.labelsize': 10, 'legend.fontsize': 10})
priors = [0., .1, 1., 10.]
cmap = plt.get_cmap('viridis', 4)
fig, axes = plt.subplots(2, 1, figsize=(7, 7.4))
for modality, marker in (('Hb_only', 'o'), ('EEG_Hb', '^')):
    part = [r for r in rows if r['modality'] == modality]
    for ax, key, ylabel in zip(axes, ('theta_log_rmse', 'driver_nrmse'),
                               ('Physiology log-error RMS', 'Driver NRMSE (fixed gauge)')):
        points = ax.scatter([r['metrics']['hb_clean_rmse'] for r in part],
            [r['metrics'][key] for r in part], c=[priors.index(r['prior_multiplier']) for r in part],
            marker=marker, cmap=cmap, vmin=-.5, vmax=3.5, s=28, alpha=.75)
        failed = [r for r in part if not r['success']]
        ax.scatter([r['metrics']['hb_clean_rmse'] for r in failed],
            [r['metrics'][key] for r in failed], marker='x', s=50, linewidths=1., color='red')
        ax.set(xlabel='Clean Hb prediction RMSE (model units)', ylabel=ylabel)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
        ax.grid(alpha=.2)
handles = [Line2D([0], [0], marker='o', ls='', color='gray', label='Hb_only'),
           Line2D([0], [0], marker='^', ls='', color='gray', label='EEG_Hb'),
           Line2D([0], [0], marker='x', ls='', color='red', label='Nonconverged finite fit')]
fig.subplots_adjust(top=.82, bottom=.1, left=.14, right=.77, hspace=.35)
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .89), ncol=3,
           fontsize=10, columnspacing=1., handletextpad=.4)
color_axis = fig.add_axes([.82, .1, .035, .72])
bar = fig.colorbar(points, cax=color_axis, label='Explicit penalty multiplier')
bar.set_ticks([0, 1, 2, 3], labels=['0', '0.1', '1', '10'])
fig.suptitle(f'{len(rows)} truth × seed × modality × penalty settings\nPaired variants: {sum(not r["success"] for r in rows)} nonconverged finite fits marked red', y=.985)
fig.savefig(OUT/'fit_versus_recovery.png', dpi=240, bbox_inches='tight')
plt.close(fig)
print(OUT/'fit_versus_recovery.png')
