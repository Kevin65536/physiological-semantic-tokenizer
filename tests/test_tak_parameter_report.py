"""Parameter-report identities and independent repeats use temporary fixtures."""
import json

import numpy as np
import pandas as pd
import pytest

from experiments.scripts.render_shared_driver_reconstruction import (
    _tak_parameter_evidence,
    _tak_parameter_pairs,
)


def _records():
    rows = []
    for subject, groups, values, good in [
        ('shared', ['g000', 'g000'], [1., 1.], [True, True]),
        ('independent', ['g001', 'g002'], [1., 1.2], [True, True]),
        ('failed', ['g003', 'g004'], [1., 2.], [True, False]),
    ]:
        for side in range(2):
            rows.append(dict(id=f'{subject}_{side}', dataset='fixture', subject=subject,
                record=f'record_{groups[side]}', group=groups[side], side=side,
                arm='tak_prior_narrow', log_sd=.125, qualified=good[side],
                kappa=values[side], tau=2*values[side]))
    return pd.DataFrame(rows)


def test_shared_groups_and_failures_never_become_perfect_repeats():
    result = _tak_parameter_pairs(_records()).set_index('subject')
    assert result.loc['shared', 'status'] == 'shared_parameter_group'
    assert result.loc['failed', 'status'] == 'failed_fit'
    assert np.isnan(result.loc['shared', 'kappa_relative_gap_pct'])
    assert np.isnan(result.loc['failed', 'tau_relative_gap_pct'])
    assert result.loc['independent', 'kappa_relative_gap_pct'] == pytest.approx(20.)
    assert result.loc['independent', 'tau_relative_gap_pct'] == pytest.approx(20.)
    fixed = _records().assign(arm='tak_fixed', log_sd=0.)
    fixed_pairs = _tak_parameter_pairs(fixed)
    assert fixed_pairs.status.eq('fixed_reference_parameters').all()
    assert fixed_pairs.kappa_relative_gap_pct.isna().all()


def test_repeat_identity_is_order_invariant_and_requires_both_planned_sides():
    records = _records()
    a = _tak_parameter_pairs(records)
    b = _tak_parameter_pairs(records.sample(frac=1, random_state=41))
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError, match='two planned record identities'):
        _tak_parameter_pairs(records.iloc[1:])


def _fixture(root):
    root.mkdir(); (root/'tasks').mkdir()
    rows = _records()
    refs = []
    for row in rows.to_dict('records'):
        refs.append(dict(id=row['id'],dataset=row['dataset'],subject=row['subject'],
            record=row['record'],side=row['side'],repeat_kind='fixture',
            anchor=dict(hb_channel='fixture_hb'),eeg_start_s=0.,hb_start_s=0.))
    by_id = {ref['id']:ref for ref in refs}
    groups = [dict(id=gid,refs=[by_id[key] for key in part.id]) for gid,part in rows.groupby('group')]
    config = dict(schema='shared_driver_tak_v1',scope='exploratory_public_cached_Hb_conditional_reconstruction',
                  data=dict(protected_data='forbidden',raw_reads=False,expected_records=6))
    (root/'resolved_config.json').write_text(json.dumps(config))
    (root/'plan.json').write_text(json.dumps(dict(records=refs,groups=groups)))
    for name in ('verification','array_verification','measured_summary','sensitivity_summary'):
        (root/(name+'.json')).write_text(json.dumps(dict(status='passed' if 'verification' in name else 'completed')))
    for group in groups:
        for arm,phase,width in [('tak_prior_narrow','sensitivity',.125),('tak_fit','measured',.25),
                                ('tak_prior_wide','sensitivity',.5),('tak_fixed','measured',0.)]:
            task_rows=rows[rows.group.eq(group['id'])].to_dict('records')
            for row in task_rows:
                row.update(arm=arm,phase=phase,length_s=90,condition='joint_full',
                           fit_status='completed' if row['qualified'] else 'failed_convergence',full_hb_nmse=.1)
            first=task_rows[0]
            values=[first['kappa'],first['tau']]
            qualified=all(row['qualified'] for row in task_rows)
            fit=dict(parameter_names=['kappa','tau'] if width else [],parameter_values=values if width else [],
                     selected_start=0,starts=[dict(start=i,converged=qualified,status='completed' if qualified else 'failed',
                        parameter_values=values if width else [],objective=1.) for i in range(3)])
            spec=dict(group=group,arm=dict(name=arm,hemodynamics='tak',initial='volume_tied',
                fit=['kappa','tau'] if width else [],log_sd=[width,width] if width else []),
                phase=phase,length_s=90,condition='joint_full')
            path=root/'tasks'/f'{phase}__{group["id"]}__90__joint_full__{arm}.json'
            path.write_text(json.dumps(dict(spec=spec,fit=fit,rows=task_rows,status='completed')))


def test_task_reader_deduplicates_shared_groups_and_retains_failure_denominator(tmp_path):
    run=tmp_path/'run';_fixture(run)
    _,records,groups,starts,pairs=_tak_parameter_evidence(run)
    assert len(records)==24  # six planned identities, four parameter settings
    assert len(groups)==20  # five parameter groups, not six independent fits
    narrow=groups[groups.arm.eq('tak_prior_narrow')]
    assert narrow.qualified.sum()==4
    failed=narrow[~narrow.qualified].iloc[0]
    assert np.isnan(failed.kappa_start_span_pct)
    assert failed.converged_starts==0 and failed.planned_starts==3
    assert len(pairs[pairs.status.eq('independent_pair')])==3
    assert len(starts)==45


def test_task_reader_rejects_disagreement_in_hard_shared_parameters(tmp_path):
    run=tmp_path/'run';_fixture(run)
    path=run/'tasks/sensitivity__g000__90__joint_full__tak_prior_narrow.json'
    task=json.loads(path.read_text());task['rows'][1]['tau']=7.;path.write_text(json.dumps(task))
    with pytest.raises(ValueError,match='Hard-shared parameter rows disagree'):
        _tak_parameter_evidence(run)


def test_task_reader_rejects_changed_prior_width(tmp_path):
    run=tmp_path/'run';_fixture(run)
    path=run/'tasks/sensitivity__g000__90__joint_full__tak_prior_narrow.json'
    task=json.loads(path.read_text());task['spec']['arm']['log_sd']=[.05,.05];path.write_text(json.dumps(task))
    with pytest.raises(ValueError,match='Retained parameter prior mismatch'):
        _tak_parameter_evidence(run)
