"""Resolve the two-seed replacement checks from completed, pair-matched evaluations."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.scripts.check_continuous_so_targets import OUT, PRIOR, CFG, old, write_json

SHORT = ['short_v3', 'short_projected_v3', 'short_projected_x']
FORMAL = [f'formal_{a}_{s}' for s in [17, 43] for a in ['v3', 'continuous']]


def summarize():
    gate = old.read(OUT / 'target_gate.json')
    assert gate['advance']
    retrieval = []
    shift = []
    guard = []
    persons = []
    macro = []
    dsrows = []
    for tag in SHORT + FORMAL:
        dest = OUT / 'members' / tag
        info = old.read(dest / 'summary.json')
        assert (info['training'], info['validation']) == (67710, 12414)
        frame = pd.read_csv(dest / 'retrieval.csv')
        assert len(frame) == 10
        adj = pd.read_csv(dest / 'alpha_specificity.csv').groupby('branch').residual.mean()
        frame['projection_adjusted_difference'] = [
            adj.get(b, np.nan) if k == 'alpha' else np.nan for b, k in zip(frame.branch, frame.endpoint)
        ]
        retrieval.append(frame.assign(model=tag))
        scores = pd.read_csv(dest / 'shift.csv')
        scores['rate'] = scores.matches / scores.pairs
        assert len(scores) == 16
        shift.append(scores.assign(model=tag))
    retrieval = pd.concat(retrieval, ignore_index=True)
    shift = pd.concat(shift, ignore_index=True)
    retrieval.to_csv(OUT / 'member_comparison.csv', index=False)
    shift.to_csv(OUT / 'natural_shift_people.csv', index=False)
    shift_mean = shift.groupby(['model', 'branch', 'dataset']).rate.mean().groupby(['model', 'branch']).mean()
    for seed in [17, 43]:
        value = {}
        for arm in ['v3', 'continuous']:
            tag = f'formal_{arm}_{seed}'
            dest = OUT / 'evaluation' / tag
            assert old.read(dest / 'summary.json')['status'] == 'completed'
            metrics = old.read(dest / 'metrics.json')
            assert (metrics['inputs'], metrics['subjects']) == (53476, 16)
            value[arm] = {}
            for k, item in metrics['metrics'].items():
                if 'macro' in item:
                    v = np.array(item['macro']).ravel()
                    value[arm][k] = v
                    macro.append(dict(seed=seed, arm=arm, metric=k, **dict(zip('XSO', v))))
                for ds, v in item['by_dataset'].items():
                    dsrows.append(
                        dict(seed=seed, arm=arm, metric=k, dataset=ds, **dict(zip('XSO', np.ravel(v))))
                    )
                for row in item['by_subject']:
                    persons.append(
                        dict(
                            seed=seed,
                            arm=arm,
                            metric=k,
                            dataset=row['dataset'],
                            subject=row['subject'],
                            **dict(zip('XSO', np.ravel(row['value']))),
                        )
                    )
            value[arm]['S_shift'] = shift_mean[tag, 'S']
            row = retrieval.query('model==@tag and branch=="S" and endpoint=="alpha"').iloc[0]
            value[arm]['effect'] = row.difference
            value[arm]['coverage'] = row.people
        a, b = value['continuous'], value['v3']
        checks = dict(
            common_X=a['nmse'][0] < b['nmse'][0],
            S_nmse=a['nmse'][1] <= 1.05 * b['nmse'][1],
            O_nmse=a['nmse'][2] <= 1.05 * b['nmse'][2],
            S_corr=a['correlation'][1] >= b['correlation'][1] - 0.01,
            O_corr=a['correlation'][2] >= b['correlation'][2] - 0.01,
            S_shift=a['S_shift'] >= b['S_shift'],
            S_alpha_effect=a['effect'] >= b['effect'],
            S_coverage=a['coverage'] >= b['coverage'],
        )
        guard.extend(dict(seed=seed, gate=k, passed=bool(v)) for k, v in checks.items())
    for name, rows in [
        ('macro', macro),
        ('people', persons),
        ('dataset', dsrows),
        ('migration_checks', guard),
    ]:
        pd.DataFrame(rows).to_csv(OUT / (name + '.csv'), index=False)
    # Exact reuse of retrieval method, with re-emission checked against the published arm.
    previous = pd.read_csv(PRIOR / 'token_members_v1/contrasts.csv')
    replay = pd.read_csv(OUT / 'members/short_projected_v3/contrasts.csv')
    pd.testing.assert_frame_equal(previous, replay, atol=1e-6, rtol=1e-6)
    write_json(
        OUT / 'summary.json',
        dict(
            experiment_id=CFG['experiment_id'],
            status='completed',
            seeds=[17, 43],
            steps=10000,
            train_people=60,
            formal_dev_people=16,
            formal_dev_electrode_crops=53476,
            member_dev_people=8,
            checks=guard,
            eligible_for_migration=all(r['passed'] for r in guard),
            production_baseline='v3',
            short_block_member_replay='verified',
            training_support=old.read(OUT / 'prepared.json'),
        ),
    )


if __name__ == '__main__':
    summarize()
