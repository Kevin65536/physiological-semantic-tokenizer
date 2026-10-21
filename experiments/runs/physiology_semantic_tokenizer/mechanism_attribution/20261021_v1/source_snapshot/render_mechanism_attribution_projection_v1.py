#!/usr/bin/env python3
"""Export report-ready bitmaps from a completed mechanism-attribution run.

This renderer never fits a model or opens a dataset. The numerical run and its
source snapshot remain the evidence owner; figures communicate their results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.inference.observation_semantics import block_interval


ARMS = ['fixed_no_common', 'fixed_common', 'free_no_common', 'free_common',
        'oracle_mismatch', 'oracle_driver_no_common', 'oracle_driver_common', 'oracle_both']
LABELS = ['Fixed\nno common', 'Fixed\ncommon', 'Free\nno common', 'Free\ncommon',
          'Oracle\nmismatch', 'Oracle\ndriver+initial', 'Oracle driver\n+initial/common', 'Oracle\nboth']
ARM_CODES = ['F0', 'FC', 'T0', 'TC', 'OM', 'OD0', 'ODC', 'OB']
COLORS = {'free_no_common': '#2166ac', 'free_common': '#b2182b', 'oracle_mismatch': '#238443'}


def projection_excess_audit(run):
    """Post-hoc semantic clarification using frozen endpoints, without fitting.

    The mismatch oracle's component equals the generating component exactly.
    Subtracting its raw projection removes genuine nuisance/physical overlap.
    """
    output_names = ['component_projection_excess_v1.csv',
                    'component_projection_excess_summary_v1.csv', 'component_projection_excess_v1.json']
    if any((run/name).exists() for name in output_names):
        raise ValueError('retained projection audit exists; use a new version for revisions')
    summary = json.loads((run/'summary.json').read_text())
    if summary['execution'] != 'completed':
        raise ValueError('projection audit requires terminal evidence')
    data = pd.read_csv(run/'endpoints.csv')
    columns = ['key', 'seed', 'noise_multiplier', 'world', 'tau_true_s', 'hb_gain', 'common_present']
    oracle = data[data.arm == 'oracle_mismatch'][columns+['converged', 'tau_change_extra_projection']].rename(
        columns={'converged': 'oracle_converged', 'tau_change_extra_projection': 'true_component_projection'})
    ordinary = data[data.arm.isin(['free_common', 'fixed_common', 'oracle_driver_common'])][
        columns+['arm', 'converged', 'tau_change_extra_projection']]
    table = ordinary.merge(oracle, on=columns, validate='many_to_one', how='left')
    baseline_tau = summary.get('resolved_reference_parameters', {}).get('free', {}).get('tau', 2.)
    table['contrast_defined'] = table.tau_true_s != baseline_tau
    valid = table.converged & table.oracle_converged & table.contrast_defined
    table['extra_error_projection'] = (table.tau_change_extra_projection-
                                      table.true_component_projection).where(valid)
    table.to_csv(run/output_names[0], index=False)
    grouped = []
    for keys, block in table.groupby(['noise_multiplier', 'tau_true_s', 'hb_gain', 'common_present', 'arm']):
        identity = dict(zip(['noise_multiplier', 'tau_true_s', 'hb_gain', 'common_present', 'arm'], keys))
        grouped.append(dict(**identity, planned_seeds=len(block),
            common_success=int((block.converged & block.oracle_converged).sum()),
            contrast_defined=bool(block.contrast_defined.all()),
            **block_interval(block.extra_error_projection.values, seed=73, repeats=2000)))
    pd.DataFrame(grouped).to_csv(run/output_names[1], index=False)
    metadata = dict(schema='component_projection_excess_v1',
        status='posthoc_semantic_clarification_not_new_confirmatory_endpoint',
        input='endpoints.csv', output_files=output_names[:2],
        source='experiments/scripts/render_mechanism_attribution.py::projection_excess_audit',
        baseline_tau_s=baseline_tau,
        formula='raw_component_change_projection_arm - raw_component_change_projection_oracle_mismatch',
        equivalent_numerator='estimated_component_change_minus_true_component_change_projected_on_true_physical_change',
        oracle_reason='oracle_mismatch saves the exact generating common component; its raw projection is the true component projection',
        contrast='world_minus_same_seed_tau2_gain1_common0_baseline',
        denominator='sum_squared_true_physical_Hb_change_in_fixed_training_RMS_coordinates',
        zero_denominator='NaN for unchanged true tau; no epsilon substitution',
        interval='2000_percentile_seed_bootstrap_within_each_noise_and_world_stratum; no_multiplicity_confirmation',
        interpretation='signed_regression_coefficient_not_energy_fraction_or_proof_that_tau_change_was_causally_stolen',
        frozen_evidence='no_refit; endpoints/profiles/summary_and_original_figure_export_preserved')
    (run/output_names[2]).write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def render(run, output):
    summary = json.loads((run/'summary.json').read_text())
    if summary['execution'] != 'completed':
        raise ValueError('figures require a terminal completed run with failures retained')
    data = pd.read_csv(run/'endpoints.csv')
    profiles = pd.read_csv(run/'profiles.csv')
    if output.exists() and any(output.iterdir()):
        raise ValueError('use a fresh figure directory; retained exports are immutable')
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 11, 'legend.fontsize': 10,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    manifest = []

    def save(fig, name, caption, sources):
        fig.savefig(output/name, dpi=240, bbox_inches='tight', pad_inches=.12)
        plt.close(fig)
        manifest.append(dict(path=str((output/name).relative_to(run)), caption=caption,
                             sources=sources, dpi=240, type='bitmap_png'))

    noisy = data[data.noise_multiplier == .2]
    worlds = noisy[['tau_true_s', 'hb_gain', 'common_present', 'world']].drop_duplicates().sort_values(
        ['tau_true_s', 'hb_gain', 'common_present'])
    if len(noisy):
        row_labels = [f'{r.tau_true_s:g}s / {r.hb_gain:g} / {int(r.common_present)}'
                      for r in worlds.itertuples()]
        for letter, metric, title in zip(['a', 'b', 'c'],
                ['tau_absolute_log_error', 'physical_Hb_nrmse', 'driver_nrmse'],
                ['Absolute log tau error', 'Clean physical Hb NRMSE', 'Driver NRMSE']):
            fig, ax = plt.subplots(figsize=(9, 6.5), layout='constrained')
            matrix = noisy[noisy.converged].pivot_table(index='world', columns='arm', values=metric,
                aggfunc='mean').reindex(index=worlds.world, columns=ARMS)
            im = ax.imshow(matrix.values, aspect='auto', cmap='YlOrRd', vmin=0)
            ax.set_title(title+'; success subset')
            ax.set_xticks(range(len(ARMS)), ARM_CODES, fontsize=13)
            ax.set_yticks(range(len(worlds)), row_labels, fontsize=12)
            ax.set_ylabel('True tau / Hb gain / common present')
            for i in range(len(worlds)):
                for j in range(len(ARMS)):
                    value = matrix.iloc[i, j]
                    ax.text(j, i, f'{value:.2f}' if np.isfinite(value) else 'NA',
                            ha='center', va='center', fontsize=12,
                            color='white' if np.isfinite(value) and value > np.nanmax(matrix.values)*.65 else 'black')
            fig.colorbar(im, ax=ax, fraction=.046, label='Mean across independent seeds')
            failures = int((~noisy.converged).sum())
            fig.suptitle(f'Noisy factorial: {noisy.seed.nunique()} seeds; {failures}/{len(noisy)} selected-fit failures', fontsize=14)
            save(fig, f'01{letter}_recovery_matrix.png',
                'Eight intervention worlds and eight fitting/control arms. F0/FC: fixed tau without/with common; '
                'T0/TC: free grid tau without/with common; OM: known gain/common correction, driver+initial refitted; '
                'OD0/ODC: true driver AND five initial values known, without/with fitted common; OB: all those oracle quantities known. '
                'Each number is the mean across successful independent seeds; all selected-fit failures and profile completeness remain in endpoint tables. '
                'Color scale applies to the named metric only. Noise is 0.2 training RMS. Tau is a finite-grid conditional parameter.', ['endpoints.csv'])

        fig, axes = plt.subplots(2, 1, figsize=(9, 9.4), sharex=True, sharey=True, layout='constrained')
        marker = {(1., False): 'o', (1.25, False): 's', (1., True): '^', (1.25, True): 'D'}
        counts = []
        for ax, tau in zip(axes, [2., 3.]):
            for (gain, common), shape in marker.items():
                block = noisy[(noisy.tau_true_s == tau) & (noisy.hb_gain == gain) & (noisy.common_present == common)]
                a = block[block.arm == 'free_common'].set_index('seed')
                b = block[block.arm == 'free_no_common'].set_index('seed')
                pair = a.join(b, lsuffix='_common', rsuffix='_plain')
                pair = pair[pair.converged_common & pair.converged_plain]
                x = pair.observed_nrmse_plain-pair.observed_nrmse_common
                y = pair.physical_Hb_nrmse_plain-pair.physical_Hb_nrmse_common
                ax.scatter(x, y, marker=shape, alpha=.75, s=50,
                           label=f'gain {gain:g}; common {int(common)}')
                counts.append(dict(tau=tau, gain=gain, common=common, paired_success=len(pair),
                                   better_fit_worse_physical=int(((x > 0) & (y < 0)).sum())))
            ax.axhline(0, color='black', linewidth=.8)
            ax.axvline(0, color='black', linewidth=.8)
            ax.set_title(f'True tau {tau:g} s; each point is one seed')
            ax.set_xlabel('Observation NRMSE gain from common term')
            ax.set_ylabel('Clean physical Hb NRMSE gain')
            ax.grid(alpha=.15)
        axes[1].legend(loc='best')
        save(fig, '02_fit_recovery_tradeoff.png',
            'Paired free-tau common minus no-common comparison, shown as control error minus candidate error. '
            'Right of zero means better observation fit; below zero means worse clean physical Hb recovery. '
            'Points share baseline/noise within a seed; only independent seeds count as replications. '
            'Both axes use training-RMS normalization, with all planned and paired-success counts retained.', ['endpoints.csv', 'paired_comparisons.csv'])
        (output/'tradeoff_counts.json').write_text(json.dumps(counts, indent=2)+'\n')

        task = 'seed201_noise0.2'
        world = 'tau3_gain1.25_common1'
        selected = profiles[(profiles.key == task) & (profiles.world == world)]
        fig, axes = plt.subplots(2, 1, figsize=(9, 9.2), layout='constrained')
        for family, label, color in [('plain', 'Unknown driver/initial; no common', '#2166ac'),
                ('common', 'Unknown driver/initial; common', '#b2182b'),
                ('oracle_mismatch', 'Known gain/common; refit driver/initial', '#238443')]:
            frame = selected[selected.family == family].sort_values('tau_candidate_s')
            for ax, field in zip(axes, ['objective', 'weighted_data_sse']):
                values = frame[field].where(frame.converged)
                ax.plot(frame.tau_candidate_s, values, 'o-', label=label, color=color, linewidth=2)
                failed = frame[~frame.converged]
                if len(failed):
                    ax.scatter(failed.tau_candidate_s, np.zeros(len(failed)), color=color, marker='x', s=90)
                ax.axvline(3, color='black', linestyle=':', label='True tau' if family == 'plain' else None)
                ax.set_xlabel('Candidate tau (s)')
                ax.set_xticks([1, 1.5, 2, 3, 4])
                ax.grid(alpha=.2)
        axes[0].set_title('Total penalized objective')
        axes[1].set_title('Data term alone (weighted SSE)')
        axes[0].set_ylabel('Squared residual objective; no likelihood claim')
        axes[1].set_ylabel('Squared residual data term')
        axes[0].legend(fontsize=9)
        fig.suptitle('Declared example: seed 201, true tau 3 s, gain 1.25, common present', fontsize=14)
        save(fig, '03_declared_example_profiles.png',
            'Every point is a separate nonlinear fit with 120 driver values and five initial coordinates refitted from zero/rest. '
            'The common arm profiles four coefficients with the training-selected penalty. Dotted line is generating tau. '
            'An x at zero marks a failed fit and is not a measured zero objective; lines break across failures. '
            'The grid and near-equal objectives are diagnostic, not confidence intervals.', ['profiles.csv', 'calibration.json'])

        example_path = run/'tasks'/task/'trajectories.npz'
        if example_path.exists():
            with np.load(example_path) as arrays:
                wi = 7
                time = np.arange(120)*.25
                fig, axes = plt.subplots(3, 1, figsize=(9, 9.4), sharex=True, layout='constrained')
                for j, (ax, label) in enumerate(zip(axes, ['Driver', 'HbO', 'HbR'])):
                    if j == 0:
                        ax.plot(time, arrays[f'w{wi}_truth_driver'], color='black', linewidth=2.4, label='Generating driver')
                    else:
                        ax.plot(time, arrays[f'w{wi}_truth_observation'][:, j], color='#999999', alpha=.6, label='Noisy gained/common observation')
                        ax.plot(time, arrays[f'w{wi}_truth_physical'][:, j], color='black', linewidth=2.4, label='Clean physical truth')
                    for arm in ['free_no_common', 'free_common', 'oracle_mismatch']:
                        key = f'w{wi}_{arm}_'+('driver' if j == 0 else 'physical')
                        if key in arrays:
                            value = arrays[key] if j == 0 else arrays[key][:, j]
                            ax.plot(time, value, color=COLORS[arm], linewidth=1.5, label=arm.replace('_', ' '))
                    ax.set_ylabel(label+'\nmodel coordinate')
                    ax.grid(alpha=.2)
                    ax.legend(loc='upper right', ncol=2, fontsize=9)
                axes[-1].set_xlabel('Within-window time (s)')
                fig.suptitle('Declared example: physical recovery\nremains distinct from observation fit', fontsize=14)
                save(fig, '04_declared_example_trajectories.png',
                    'The same prespecified seed/world as the profile plot. Colored Hb lines are canonical physical predictions before gain/common; '
                    'grey is the actual gained/noisy/common observation and black is hidden generating truth. '
                    'Oracle mismatch is unavailable-information control, not an empirical estimator. '
                    'Every reconstruction uses the complete window; this is offline fitting, not forecasting.',
                    [f'tasks/{task}/trajectories.npz', 'endpoints.csv'])

    burden = pd.read_csv(run/'retained_EOG_burden.csv')
    burden = burden[burden.control == 'EOG_wrong_subject']
    fig, axes = plt.subplots(2, 1, figsize=(9, 8.5), layout='constrained')
    dataset_labels = {'eeg_fnirs_single_trial': 'Single-Trial', 'simultaneous_eeg_nirs': 'Simultaneous'}
    for ax, endpoint in zip(axes, ['EEG_voltage', 'HbT']):
        frame = burden[burden.endpoint == endpoint].sort_values(['dataset', 'burden'])
        for i, row in enumerate(frame.itertuples()):
            color = '#b2182b' if row.burden == 'high' else '#2166ac'
            ax.plot([row.low, row.high], [i, i], color=color, linewidth=2)
            ax.plot(row.mean, i, 'o', color=color, markersize=7)
        labels = [f'{dataset_labels[r.dataset]} {r.burden}\n{r.paired_windows} windows / {r.independent_units} subjects'
                  for r in frame.itertuples()]
        ax.set_yticks(range(len(frame)), labels)
        ax.axvline(0, color='black', linewidth=.8)
        ax.set_xlabel('Wrong-subject error minus real EOG error')
        ax.set_title(endpoint+'; panel-specific x scale')
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Frozen heldout EOG predictions\nEvent-burden strata defined on training subjects', fontsize=14)
    save(fig, '05_retained_EOG_burden.png',
        'Each dot is an equal-subject mean NRMSE gain of real EOG over the wrong-subject control; lines are descriptive 95% subject bootstrap intervals. '
        'High burden is strictly above the training-window median event count (17 Single-Trial;18 Simultaneous); zero-event windows remain. '
        'The two panels intentionally use different x scales. EOG validates an ocular association, not cardiac/respiratory/tissue source truth. '
        'The same subject may contribute to both strata, so strata are not independent samples.',
        ['retained_EOG_burden.csv', 'retained_event_burden_inventory.csv', 'source_audit.json'])

    (output/'figure_manifest.json').write_text(json.dumps(dict(
        schema='mechanism_attribution_figures_v1', source_run=str(run), figures=manifest,
        report_export='bitmap figures only; text/captions remain searchable in owning report'), indent=2)+'\n')
    shutil.copy2(Path(__file__), output/'render_mechanism_attribution.py')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--projection-audit', action='store_true',
                        help='write a versioned post-hoc projection diagnostic, without plotting or fitting')
    args = parser.parse_args()
    run = args.run.resolve()
    if args.projection_audit:
        print(json.dumps(projection_excess_audit(run), indent=2))
        return
    output = args.output.resolve() if args.output else run/'figures'
    for figure in render(run, output):
        print(figure['path'])


if __name__ == '__main__':
    main()
