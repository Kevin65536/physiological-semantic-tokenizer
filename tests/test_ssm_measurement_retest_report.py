import json

import numpy as np
import yaml

from experiments.scripts import render_ssm_measurement_retest_report as report


def test_report_keeps_fixed_denominator_and_matches_control_identities(tmp_path):
    run = tmp_path/'run'
    (run/'prepared').mkdir(parents=True)
    (tmp_path/'historical/prepared').mkdir(parents=True)
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    subject = 'fixture_subject'
    signal = np.broadcast_to(np.linspace(0, 1, 120)[None, :, None, None], (24, 120, 1, 2))
    for root in [run, tmp_path/'historical']:
        np.savez(root/'prepared'/f'{subject}.npz', target_fnirs=signal)
    for outer in range(4):
        write(run/'prepared'/f'{subject}_o{outer}_E0.json', dict(validation=[outer],
            projection=dict(fnirs_factor=2., fnirs_pair=0), normalization_sd=[1., 1., 1.]))
    write(run/'manifest.json', dict(project_root=str(tmp_path)))
    (run/'resolved_config.yaml').write_text(yaml.safe_dump(dict(subjects=[subject],
        measurement_retest=dict(historical_run='historical'))))
    tasks = []
    for trial in range(2):
        task = dict(id=f'trial_{trial}', family='B', subject=subject, outer=0,
            candidate=dict(w=0.), mapping='prior_gauge', solver='O0', modes=report.MODES, planned_solves=14)
        tasks.append(task)
        rows = []
        for mode in report.MODES:
            values = {name: dict(nmse=2. if mode.endswith('_own') else 1., nrmse=1.,
                rmse_coordinate=1., bias_coordinate=0.) for name in report.MODS}
            rows.append(dict(subject=subject, session='session', sample_id=str(trial), mode=mode,
                solver='O0', status='failed_domain' if trial == 1 and mode == 'center_EEG_shift' else 'completed',
                metrics=values, center_metrics=values, clean_map_residual=values))
        write(run/'cells'/task['id']/'result.json', dict(status='completed', rows=rows))
    import csv
    with (run/'task_table.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['payload'])
        writer.writeheader()
        writer.writerows(dict(payload=json.dumps(task)) for task in tasks)
    write(run/'cells/retest_adaptation_trigger/result.json', dict(passed=False))
    data = report.analyze(run)
    assert data['panels'][0]['complete_14_mode_trials'] == 1
    assert data['panels'][0]['expected'] == 72
    assert data['panels'][0]['full_B'] is None
    controls = {r['control']: r for r in data['paired_increments'] if r['direction'] == 'EEG'}
    assert controls['own']['common'] == 2
    assert controls['own']['increment'] == 1.
    assert controls['shift']['common'] == 1
    assert not controls['own']['full_panel']
    assert len(data['failures']) == 1
    assert all(r['rms_change_new_training_sd'] == 0 for r in data['selected_input_attribution'])
