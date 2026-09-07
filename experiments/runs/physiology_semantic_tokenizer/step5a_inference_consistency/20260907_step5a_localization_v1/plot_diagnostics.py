"""Rebuild the short-case reference CDF figure from this run's saved curves."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from experiments.evaluate_step5a_inference_consistency import posterior_grid

cfg = yaml.safe_load((ROOT/'resolved_config.yaml').read_text())
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8), sharey=True, layout='constrained')
for ax, axis in zip(axes, 'GWZ'):
    case = json.loads((ROOT/f'case_{axis}_00.json').read_text())
    ref = case['nonlinear_reference']
    grid = np.asarray(ref['grid'])
    cdf = posterior_grid(cfg, axis, grid, np.array(ref['parameter_log_likelihood']))['cdf']
    score = posterior_grid(cfg, axis, grid, np.array(ref['predictive_score']))['cdf']
    runs = np.array([posterior_grid(cfg, axis, grid, ll)['cdf'] for ll in np.asarray(ref['independent_log_likelihoods'])[1]])
    ax.fill_between(grid, runs.min(axis=0), runs.max(axis=0), color='#167d9a', alpha=.18, label='PF individual-run range')
    ax.plot(grid, cdf, color='#167d9a', lw=2.2, label='Joint particle likelihood')
    ax.plot(grid, score, color='#cb553e', lw=2, ls='--', label='EKF marginal score')
    ax.axvline(case['truth'], color='#6a6a6a', ls=':', lw=1.3, label='Generating parameter')
    resolved = ref['reference_precision_pass']
    ax.set_title(f"{axis}: {'reference checks passed' if resolved else 'reference unresolved'}\nmax CDF difference = {ref['posterior_cdf_difference']:.3f}", fontsize=11)
    if not resolved:
        ax.set_facecolor('#fff8ed')
    ax.set_xlabel({'G':'Relative log gain, g', 'W':'Relative log frequency, w', 'Z':'Physical damping ratio, zeta'}[axis])
    ax.set_ylim(0, 1)
    ax.grid(alpha=.16)
axes[0].set_ylabel('Parameter posterior CDF')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False, fontsize=9)
fig.suptitle('Step5A0 | Same-model short cases (6 s); no calibration or teacher admission claim', fontsize=12)
fig.savefig(ROOT/'reference_cdf.png', dpi=180)
fig.savefig(ROOT/'reference_cdf.pdf')
plt.close(fig)
