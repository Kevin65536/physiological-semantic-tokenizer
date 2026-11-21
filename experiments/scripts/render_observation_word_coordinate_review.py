"""Plot the retained 4-bit word stability and synthetic Hb coordinate endpoints."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def draw_contrasts(ax, labels, estimates, threshold, xlabel):
    positions = np.arange(len(labels))
    means = np.asarray([row[0] for row in estimates])
    bounds = np.asarray([row[1] for row in estimates])
    ax.errorbar(means, positions, xerr=[means-bounds[:, 0], bounds[:, 1]-means],
                fmt='o', color='#205c80', capsize=4)
    ax.axvline(0, color='0.55', lw=0.8)
    ax.axvline(threshold, color='#9e492c', ls='--', label=f'Criterion {threshold:+.2f}')
    ax.set(yticks=positions, yticklabels=labels, xlabel=xlabel, ylim=(len(labels)-.45, -.55))
    ax.legend(loc='best', fontsize=8)
    ax.grid(axis='x', alpha=.16)


def stability(summary, output):
    methods = ['eeg', 'hb']
    values = summary['findings']
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2), layout='constrained')
    draw_contrasts(left, ['EEG', 'Hb'],
                   [(values[m]['paired_ari']['candidate_minus_reference'], values[m]['paired_ari']['ci95'])
                    for m in methods], .03, 'Rank minus standard ARI')
    left.set_title('Partition stability · 20 validation subjects')
    positions = np.arange(2)
    for j, (arm, color) in enumerate([('standard', '#205c80'), ('rank', '#ac5b35')]):
        gain = [100*values[m]['current_attribute_gains'][arm] for m in methods]
        right.bar(positions+(j-.5)*.34, gain, width=.34, color=color, label=arm, hatch='//' if j else None)
    right.set(xticks=positions, xticklabels=['EEG', 'Hb'], ylabel='Gain over background (%)',
              title='Current-attribute readout · allowed loss 5 pp')
    right.legend()
    fig.suptitle('Four-bit observation words on the development panel')
    fig.supxlabel('ARI intervals: paired 95% subject bootstrap within four fixed datasets.', fontsize=9)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def simulation(summary, output):
    results = summary['findings']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    names = [('Overall', results['overall']), *[(label, results['by_regime'][name])
             for label, name in [('Ordinary','ordinary'), ('Level tails','level_tails'),
                                 ('Slope tails','slope_tails'), ('Both tails','both_tails')]]]
    for ax, (key, threshold, title) in zip(axes, [
            ('ari', .03, 'Partition stability · higher is better'),
            ('slope_nmse', .05, 'Known-slope error · lower is better')]):
        contrasts = [r['paired']['rank_levels'][key] for _,r in names]
        draw_contrasts(ax, [name for name,_ in names], [(r['difference'],r['ci95']) for r in contrasts],
                       threshold, 'Level ranks minus standard')
        ax.set_title(title)
    fig.suptitle('Synthetic Hb · rank-transform levels and retain the slope coordinates')
    fig.supxlabel('95% paired intervals across 256 generator replicates. Overall averages four regimes within each replicate.',
                  fontsize=9)
    fig.savefig(output, dpi=240)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = json.loads(args.summary.read_text())
    render = {'TK-OBS-WORD-STABILITY-v1': stability, 'TK-OBS-HB-COORD-SIM-v1': simulation}
    if result['status'] != 'completed':
        parser.error('a completed run summary is required')
    with args.output.open('xb') as dest:
        render[result['experiment_id']](result, dest)
