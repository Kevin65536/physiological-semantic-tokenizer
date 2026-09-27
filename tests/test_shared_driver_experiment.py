"""Array-free contracts and temporary public-surface fixtures for the screen."""
from copy import deepcopy
import json
from types import SimpleNamespace
from concurrent.futures import Future

import numpy as np
import pytest
import yaml

from experiments.scripts import evaluate_shared_driver_reconstruction as run


@pytest.fixture
def roi_cfg():
    return run.read_fixed_roi_tau_config(run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_tau_v1.yaml')


def test_fixed_roi_tau_contract_check_only_reads_no_arrays(roi_cfg,tmp_path,monkeypatch,capsys):
    def forbidden(*a,**k):raise AssertionError('check-only cannot read arrays')
    monkeypatch.setattr(np,'load',forbidden)
    run.gain_prior_fit_main(SimpleNamespace(config=run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_tau_v1.yaml',
        fixed_roi_tau_fit=True,check_only=True))
    assert json.loads(capsys.readouterr().out)['source_arrays_read']==0
    assert len(run.nonlinear_group_specs(roi_cfg,'synthetic'))==12
    for mode in roi_cfg['modes']:
        assert run.view_operators(roi_cfg,mode)[2].shape==(360,360)
    for section,key,value in [('fixed_roi','pair_index',1),('fixed','neurovascular_gain',2.),
        ('tau_training','gradient_tolerance',.1),('regularization','driver_amplitude_weight',1.)]:
        bad=deepcopy(roi_cfg);bad[section][key]=value;p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError):run.read_fixed_roi_tau_config(p)


def test_fixed_roi_synthetic_taus_have_paired_driver_initial_and_standard_noise(roi_cfg,monkeypatch):
    # Cheap synthetic forward stub preserves a tau-dependent Hb scale and initial states.
    def replay(driver,initial,p,dt,**kw):
        canonical=np.column_stack((driver,p.free.tau*driver,.3*p.free.tau*driver))
        states=np.tile(np.r_[driver[0],initial],(len(driver),1));states[:,0]=driver
        return dict(status='completed',canonical_prediction=canonical,states=states)
    monkeypatch.setattr(run,'replay_nonlinear_driver',replay)
    specs=[s for s in run.nonlinear_group_specs(roi_cfg,'synthetic') if s['spectrum']=='mixed' and s['replicate']==0]
    panels=[run.nonlinear_synthetic_panel(roi_cfg,s)[0] for s in specs]
    for p,s in zip(panels[1:],specs[1:]):
        np.testing.assert_array_equal(p['truth'],panels[0]['truth'])
        np.testing.assert_array_equal(p['truth_states'][:,0,1:],panels[0]['truth_states'][:,0,1:])
        np.testing.assert_allclose(p['feature_fnirs']/s['true_tau'],panels[0]['feature_fnirs'],atol=1e-15)
        np.testing.assert_array_equal(p['feature_eeg'],panels[0]['feature_eeg'])


@pytest.mark.parametrize('method,prior',[('fixed_roi_trained_tau',None),('fixed_roi_prior_tau',2.)])
def test_tau_training_is_train_only_traced_and_checks_selected_tau_integration(tmp_path,roi_cfg,monkeypatch,method,prior):
    from src.inference import shared_driver_reconstruction as library
    spec=dict(kind='measured',group='subject_01_o0',subject='subject_01',outer=0)
    train=[i for i in range(24) if i%4];validation=[i for i in range(24) if not i%4]
    state=np.tile([0.,0.,1.,1.,1.,1.],(120,1));calls=[]
    def fit(y,p,dt,**kw):
        calls.append(y.copy());assert y.shape==(18,120,3) and np.all(y==2.)
        assert p.fixed.neurovascular_gain==1. and kw['parameter_name']=='tau'
        assert kw['parameter_prior_mean']==prior and kw['record_trace'] is True
        assert kw['driver_amplitude_weight']==0. and kw['starts'][0]['parameter_value']==4.
        return dict(status='completed',converged=True,parameter_value=3.,objective=4.,expected_trials=18,completed_trials=18,
            driver=np.zeros((18,120)),initial_state=np.tile(state[0,1:],(18,1)),states=np.tile(state,(18,1,1)),
            prediction=np.zeros((18,120,3)),canonical_prediction=np.zeros((18,120,3)),trace=[dict(objective=5.)])
    def replay(driver,initial,p,dt,**kw):
        assert p.free.tau==3. and p.fixed.neurovascular_gain==1. and kw['substeps']==8
        return dict(status='completed',states=state,canonical_prediction=np.zeros((120,3)))
    monkeypatch.setattr(library,'fit_nonlinear_shared_parameter',fit);monkeypatch.setattr(run,'replay_nonlinear_driver',replay)
    for folder,hidden in [('a',3.),('b',1e9)]:
        out=tmp_path/folder;target=np.full((24,120,3),2.);target[validation]=hidden
        run.write_json(out/'prepared'/f"{spec['group']}.json",dict(status='completed',metadata=dict(train=train,validation=validation),driver_prior_sd=.2))
        np.savez(out/'prepared'/f"{spec['group']}.npz",target=target,normalizer=np.array([.2,.3,.4]))
        r=run.train_gain_prior_start((roi_cfg,out,spec,method,2,4.))
        assert r['status']=='completed' and r['parameter_name']=='tau' and r['start_tau']==4.
        assert len(r['integration_checks'])==18 and r['trace'][0]['objective']==5.
    np.testing.assert_array_equal(*calls)


@pytest.mark.parametrize('training_status',['completed','failed_training'])
def test_roi_validation_freezes_training_tau_and_preserves_failure_denominator(tmp_path,roi_cfg,monkeypatch,training_status):
    from src.inference import shared_driver_reconstruction as library
    spec=dict(kind='measured',group='subject_01_o0',subject='subject_01',outer=0);method='fixed_roi_trained_tau'
    run.write_json(tmp_path/'prepared'/f"{spec['group']}.json",dict(status='completed',
        metadata=dict(train=list(range(18)),validation=list(range(18,24))),selected_tau={m:2. for m in run.FIXED_ROI_TAU_METHODS}))
    run.write_json(run.gain_training_directory(tmp_path,spec,method)/'selection.json',
        dict(status=training_status,parameter_value=3. if training_status=='completed' else None))
    np.savez(tmp_path/'prepared'/f"{spec['group']}.npz",target=np.ones((24,120,3)),normalizer=np.ones(3),
             feature_eeg=np.zeros((24,120)),feature_fnirs=np.zeros((24,300,2)))
    state=np.tile([0.,0.,1.,1.,1.,1.],(120,1));calls=[]
    monkeypatch.setattr(run,'build_shared_driver_design',lambda *a,**k:None)
    monkeypatch.setattr(run,'fit_shared_driver',lambda *a,**k:dict(driver=np.zeros(120),states=state))
    def fit(y,p,dt,**kw):
        calls.append(1);assert p.free.tau==3. and p.fixed.neurovascular_gain==1.
        assert kw.get('driver_amplitude_weight',0.)==0.
        return dict(status='failed_domain',converged=False,starts=[])
    monkeypatch.setattr(library,'fit_nonlinear_shared_driver',fit)
    r=run.nonlinear_cell((roi_cfg,tmp_path,spec,method,'center_EEG',False))
    assert len(r['rows'])==6
    assert len(calls)==(6 if training_status=='completed' else 0)
    assert all(x['status']==('failed_domain' if training_status=='completed' else 'failed_training') for x in r['rows'])


def test_roi_measured_phase_rejects_missing_synthetic_before_arrays(tmp_path,roi_cfg,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('arrays before synthetic evidence')
    monkeypatch.setattr(np,'load',forbidden)
    args=SimpleNamespace(config=run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_tau_v1.yaml',
        fixed_roi_tau_fit=True,check_only=False,pilot=False,phase='measured',workers=1,
        project_root=tmp_path,run_dir=tmp_path/roi_cfg['output_root']/'fixture')
    with pytest.raises(ValueError,match='synthetic terminal'):
        run.gain_prior_fit_main(args)


def test_roi_tau_selection_waits_for_three_starts_and_excludes_failed_lowest(tmp_path):
    spec=dict(kind='synthetic',group='fixture');method='fixed_roi_trained_tau'
    directory=run.gain_training_directory(tmp_path,spec,method)
    for i in range(2):
        run.write_json(directory/f'start_{i}'/'result.json',dict(status='completed',converged=True,
            expected_trials=18,completed_trials=18,objective=3.-i,parameter_value=2.+i))
    with pytest.raises(ValueError,match='all declared'):
        run.select_gain_prior_training(tmp_path,spec,method,[0,1,2])
    run.write_json(directory/'start_2'/'result.json',dict(status='failed_numerical',converged=False,
        expected_trials=18,completed_trials=18,objective=.1,parameter_value=8.))
    result=run.select_gain_prior_training(tmp_path,spec,method,[0,1,2])
    assert result['parameter_value']==3. and result['parameter_name']=='tau'
    assert result['method']==method and result['amplitude_weight']==0.


@pytest.fixture
def cfg():
    return run.read_config(run.DEFAULT)


@pytest.fixture
def prepared(tmp_path, cfg):
    """Synthetic, non-authorizing fixture; no retained repository arrays read."""
    subject, outer = cfg['subjects'][0], 0
    directory = tmp_path/cfg['source_run']/'prepared'
    directory.mkdir(parents=True)
    rng = np.random.default_rng(71)
    eeg = rng.normal(0, .03, (24, 120))
    hb = rng.normal(0, .01, (24, 300, 2))
    target = np.array([run.view(e, h, cfg, 'full')[0] for e, h in zip(eeg, hb)])
    valid = [i for i in range(24) if (i % 8) % 4 == outer]
    train = [i for i in range(24) if i not in valid]
    sd = np.std(np.concatenate(target[train]), axis=0)
    arrays = dict(feature_eeg=eeg, feature_fnirs=hb, target=target, normalizer=sd)
    info = dict(subject=subject, outer=outer, inner=None, train=train, validation=valid,
                normalization_sd=sd.tolist())
    trials = [dict(session=s, original_ma_trial_position=p)
              for s in cfg['sessions'] for p in cfg['original_positions']]
    identity_path = directory/f'{subject}.json'
    info_path = directory/f'{subject}_o0_E0.json'
    array_path = directory/f'{subject}_o0_E0.npz'
    identity_path.write_text(json.dumps(dict(trials=trials)))
    info_path.write_text(json.dumps(info))
    np.savez(array_path, **arrays)
    return dict(root=tmp_path, subject=subject, info=info, arrays=arrays,
                info_path=info_path, identity_path=identity_path, array_path=array_path)


def test_config_and_task_denominators_are_fixed_without_arrays(cfg):
    assert len(run.tasks(cfg, 'unused', 'unused', 'synthetic')) == 12
    assert len(run.tasks(cfg, 'unused', 'unused', 'measured')) == 12
    grid = run.tau_grid(cfg)
    assert np.all(np.diff(grid) > 0)
    assert all(t in grid for t in (1., 2., 4.))
    _, _, operator, mask = run.view_operators(cfg, 'full')
    design = run.build_shared_driver_design(run.parameters(cfg, 2.), 120, .25,
        processed_mean_operator=operator)
    assert design.observation_design.shape == (360, 125)
    assert mask.shape == (120, 3) and mask.all()


@pytest.mark.parametrize('field,value', [
    ('subjects', ['subject_24']), ('outer_folds', [0]),
    ('center_steps', 15), ('sessions', ['session_07']),
    ('source_run', '../outside_retained_scope'),
    ('protected_boundary', 'allow_everything'),
])
def test_changed_scope_contract_rejected_before_array_access(tmp_path, cfg, field, value):
    cfg[field] = value
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        run.read_config(path)


def test_prepared_temporary_fixture_identity_and_18_6_denominator(cfg, prepared):
    arrays, info = run.load_measured(cfg, prepared['root'], prepared['subject'], 0)
    assert len(info['train']) == 18 and len(info['validation']) == 6
    assert set(info['train']).isdisjoint(info['validation'])
    assert arrays['target'].shape == (24, 120, 3)


@pytest.mark.parametrize('fault', ['split', 'identity'])
def test_metadata_fault_rejected_before_npz_read(cfg, prepared, monkeypatch, fault):
    if fault == 'split':
        info = prepared['info']
        info['validation'] = info['validation'][:-1]
        prepared['info_path'].write_text(json.dumps(info))
    else:
        identity = json.loads(prepared['identity_path'].read_text())
        identity['trials'][0]['original_ma_trial_position'] = 99
        prepared['identity_path'].write_text(json.dumps(identity))
    def forbidden(*args, **kwargs):
        raise AssertionError('array accessed before metadata validation')
    monkeypatch.setattr(run.np, 'load', forbidden)
    with pytest.raises(ValueError):
        run.load_measured(cfg, prepared['root'], prepared['subject'], 0)


@pytest.mark.parametrize('fault', ['normalizer_mismatch', 'zero_normalizer', 'shape', 'target_map'])
def test_prepared_normalizer_tensor_and_target_rejected(cfg, prepared, fault):
    arrays, info = prepared['arrays'], prepared['info']
    if fault == 'normalizer_mismatch':
        arrays['normalizer'] *= 2.
    elif fault == 'zero_normalizer':
        arrays['normalizer'][1] = 0.
        info['normalization_sd'][1] = 0.
        prepared['info_path'].write_text(json.dumps(info))
    elif fault == 'shape':
        arrays['feature_fnirs'] = arrays['feature_fnirs'][:, :-1]
    else:
        arrays['target'][0, 20, 1] += 1.
    np.savez(prepared['array_path'], **arrays)
    with pytest.raises(ValueError):
        run.load_measured(cfg, prepared['root'], prepared['subject'], 0)


@pytest.mark.parametrize('mode', ['center_EEG', 'center_fNIRS', 'EEG_only', 'fNIRS_only'])
def test_hidden_native_features_do_not_enter_processed_visible_inputs(cfg, mode):
    rng = np.random.default_rng(12)
    eeg, hb = rng.normal(size=120), rng.normal(size=(300, 2))
    first, operator, mask = run.view(eeg, hb, cfg, mode)
    if mode == 'center_EEG':
        eeg[52:68] += 1e6
    elif mode == 'center_fNIRS':
        hb[130:170] += 1e6
    elif mode == 'EEG_only':
        hb += 1e6
    else:
        eeg += 1e6
    second, second_operator, second_mask = run.view(eeg, hb, cfg, mode)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(operator, second_operator)
    np.testing.assert_array_equal(mask, second_mask)


def test_parameter_grid_selection_uses_only_training_trials(tmp_path, cfg, monkeypatch):
    """Replace numerical solves, retaining the real group selection/control flow."""
    cfg = deepcopy(cfg)
    cfg['modes'] = ['full']
    target = np.ones((24, 120, 3))*2.
    arrays = dict(target=target, feature_eeg=np.zeros((24, 120)),
        feature_fnirs=np.zeros((24, 300, 2)), normalizer=np.ones(3))
    meta = dict(train=list(range(18)), validation=list(range(18, 24)),
        trials=[dict(session='temporary_fixture') for _ in range(24)])
    monkeypatch.setattr(run, 'load_measured', lambda *args: (arrays, meta))
    monkeypatch.setattr(run, 'tau_grid', lambda cfg: np.array([1., 2., 4.]))
    monkeypatch.setattr(run, 'build_shared_driver_design',
        lambda p, *args, **kwargs: SimpleNamespace(tau=p.free.tau))
    def fake_fit(y, design, sd, cfg, **kwargs):
        cost = float((np.nanmean(y)-design.tau)**2)
        return dict(weighted_data_sse=cost, data_sse=cost, penalty=0., roughness=0.,
            initial_penalty=0., initial_state=np.zeros(5), effective_df=1.,
            rank=1, singular_values=np.array([1., 0.]), rcond=kwargs.get('rcond', cfg['svd_rcond']),
            driver=np.zeros(120), prediction=np.zeros((120, 3)),
            canonical_prediction=np.zeros((120, 3)), states=np.zeros((120, 6)),
            physical_validity=dict(valid=True, mathematical_valid=True,
                small_signal_valid=True, max_fractional_excursion=0.))
    monkeypatch.setattr(run, 'fit', fake_fit)
    outputs = []
    for index, (train_value, heldout_value) in enumerate([(2., 4.), (2., 1e6), (4., 1e6)]):
        target[:18] = train_value
        target[18:] = heldout_value
        out = tmp_path/str(index)
        out.mkdir()
        result = run.run_group((cfg, out, tmp_path, 'measured', 'subject_01', 0, None, None))
        assert result['status'] == 'completed', result.get('error')
        outputs.append(result)
    assert outputs[0]['selected_tau'] == outputs[1]['selected_tau']
    assert outputs[0]['profiles'] == outputs[1]['profiles']
    assert outputs[0]['selected_tau']['fitted_tau'] == 2.
    assert outputs[2]['selected_tau']['fitted_tau'] == 4.


def test_aggregate_nrmse_is_root_equal_subject_session_nmse(tmp_path):
    """Unequal trial/session counts must not turn RMS into mean trial RMSE."""
    groups = tmp_path/'groups'
    groups.mkdir()
    rows = []
    # Subject A: session means 0 and 4 -> NMSE 2; subject B: NMSE 9.
    # Equal subject/session aggregation is 5.5, distinct from pooling trials.
    for subject, session, errors in [('a', 's1', [0., 0., 0.]),
                                      ('a', 's2', [4.]), ('b', 's1', [9., 9.])]:
        for error in errors:
            row = dict(kind='synthetic', method='fixed_tau', mode='full',
                subject=subject, session=session, physical_valid=True, small_signal_valid=True)
            for modality in run.MODALITIES:
                row['nmse_'+modality] = error
                row['nrmse_'+modality] = np.sqrt(error)
            rows.append(row)
    (groups/'fixture.json').write_text(json.dumps(dict(status='completed', rows=rows)))
    summary = run.summarize(tmp_path, 1)
    aggregate = summary['aggregates'][0]
    for modality in run.MODALITIES:
        assert aggregate['nmse_'+modality] == pytest.approx(5.5)
        assert aggregate['nrmse_'+modality] == pytest.approx(np.sqrt(5.5))
        assert aggregate['mean_trial_nrmse_'+modality] == pytest.approx(2.)


@pytest.fixture
def replay_cfg():
    return yaml.safe_load((run.DEFAULT.parent/'shared_driver_nonlinear_replay_v1.yaml').read_text())


def test_replay_group_never_refits_or_mutates_parent_and_retains_failures(tmp_path, cfg, replay_cfg, monkeypatch):
    parent, out = tmp_path/'parent', tmp_path/'replay'
    (parent/'groups').mkdir(parents=True)
    (parent/'predictions').mkdir()
    out.mkdir()
    group = 'temporary_synthetic_o0'
    rows, arrays = [], dict(target=np.zeros((6,120,3)), normalizer=np.ones(3), trial_indices=np.arange(6))
    for method in replay_cfg['methods']:
        state = np.ones((6,120,6))
        state[:,:,1] = 0.
        for i in range(6):
            state[i,:,0] = i-1  # first invalid domain, second resolution disagreement
            rows.append(dict(kind='synthetic', subject='temporary', outer=0, trial=i,
                session='temporary', method=method, tau=2., true_tau=2., mode='full',
                **{prefix+m: 0. for prefix in ('nmse_', 'nrmse_') for m in run.MODALITIES}))
        arrays[f'states_{method}_full'] = state
        arrays[f'prediction_{method}_full'] = np.zeros((6,120,3))
    source = parent/'groups'/f'{group}.json'
    source.write_text(json.dumps(dict(group=group,kind='synthetic',status='completed',rows=rows,
        metadata=dict(validation=list(range(6))),selected_tau={})))
    npz = parent/'predictions'/f'{group}.npz'
    np.savez(npz, **arrays)
    before = (source.read_bytes(), npz.read_bytes())
    monkeypatch.setattr(run, 'view_operators', lambda *args: (None, None, np.eye(360), None))
    def forbidden(*args, **kwargs):
        raise AssertionError('replay must not refit or access measured loaders')
    monkeypatch.setattr(run, 'fit', forbidden)
    monkeypatch.setattr(run, 'load_measured', forbidden)
    def fake_replay(driver, initial, parameters, dt, *, substeps):
        failed = driver[0] < 0
        prediction = np.zeros((120,3))
        if driver[0] == 0 and substeps == 8:
            prediction += .1
        return dict(status='failed_domain' if failed else 'completed',
            canonical_prediction=prediction,states=np.tile(np.r_[driver[0], initial],(120,1)),
            checks=dict(all_intermediate_valid=not failed),failure=dict(stage='k2') if failed else None)
    monkeypatch.setattr(run, 'replay_nonlinear_driver', fake_replay)
    result = run.replay_group((replay_cfg,cfg,out,parent,group))
    assert len(result['rows']) == 24
    statuses = [r['status'] for r in result['rows']]
    assert statuses.count('failed_domain') == 4
    assert statuses.count('failed_integration_check') == 4
    assert statuses.count('completed') == 16
    assert all(len(r['checks']) == 2 for r in result['rows'])
    assert before == (source.read_bytes(), npz.read_bytes())


@pytest.mark.parametrize('failed_group', [False, True])
def test_replay_controller_runs_synthetic_first_and_compares_identical_success_subset(
        tmp_path, cfg, replay_cfg, monkeypatch, failed_group):
    parent = tmp_path/replay_cfg['source_run']
    parent.mkdir(parents=True)
    (parent/'manifest.json').write_text(json.dumps(dict(execution='completed')))
    (parent/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
    config_path = tmp_path/'replay.yaml'
    config_path.write_text(yaml.safe_dump(replay_cfg))
    out = tmp_path/'replay'
    order = []
    class ImmediatePool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, payload):
            future = Future()
            try:
                future.set_result(fn(payload))
            except Exception as exc:
                future.set_exception(exc)
            return future
    def fake_group(payload):
        group = payload[-1]
        kind = 'synthetic' if group.startswith('synthetic') else 'measured'
        order.append(kind)
        if failed_group and group == 'synthetic_t1_r0_o0':
            raise ValueError('temporary fixture source identity error')
        rows = []
        for method in replay_cfg['methods']:
            for trial in range(6):
                status = ['failed_domain','failed_integration_check','completed','completed','completed','completed'][trial]
                row = dict(kind=kind, method=method, subject=group, session='temporary', status=status)
                for m in run.MODALITIES:
                    row['nmse_'+m] = float(trial**2)
                    row['linear_nmse_'+m] = 10000. if trial < 2 else 4.
                rows.append(row)
        return dict(group=group,status='completed',seconds=0.,rows=rows)
    monkeypatch.setattr(run,'ProcessPoolExecutor',ImmediatePool)
    monkeypatch.setattr(run,'replay_group',fake_group)
    args=SimpleNamespace(config=config_path,project_root=tmp_path,replay_of=parent,
        check_only=False,run_dir=out,workers=1,pilot=False)
    run.replay_main(args)
    assert order == ['synthetic']*12+['measured']*12
    summary = json.loads((out/'summary.json').read_text())
    assert summary['expected_method_trials'] == 576
    assert summary['completed_method_trials'] == 384-16*int(failed_group)
    assert summary['failed_groups'] == int(failed_group)
    if failed_group:
        failure = json.loads((out/'groups'/'synthetic_t1_r0_o0.json').read_text())
        assert failure['status'] == 'failed'
        assert 'temporary fixture source identity error' in failure['error']
    for row in summary['aggregates']:
        lost = int(failed_group and row['kind'] == 'synthetic')
        assert row['expected'] == 72
        assert row['completed'] == 48-4*lost
        assert row['failure_counts']['failed_domain'] == 12-lost
        assert row['failure_counts']['failed_integration_check'] == 12-lost
        for m in run.MODALITIES:
            assert row['nrmse_'+m] == pytest.approx(np.sqrt(13.5))
            assert row['linear_nrmse_same_subset_'+m] == pytest.approx(2.)


@pytest.fixture
def nonlinear_cfg():
    return run.read_nonlinear_fit_config(run.DEFAULT.parent/'shared_driver_nonlinear_fit_v1.yaml')


def test_nonlinear_scope_and_numerics_reject_before_array_access(tmp_path, nonlinear_cfg, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('array read before contract validation')
    monkeypatch.setattr(run.np, 'load', forbidden)
    changes = [('subjects', ['subject_24']), ('source_run', '../unknown'),
               ('protected_boundary', 'none'), ('reference_tau_s', -2.), ('svd_rcond', 0.)]
    for field, value in changes:
        changed = deepcopy(nonlinear_cfg)
        changed[field] = value
        path = tmp_path/'bad.yaml'
        path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError):
            run.read_nonlinear_fit_config(path)
    changed = deepcopy(nonlinear_cfg)
    changed['nonlinear']['gradient_tolerance'] = float('nan')
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError):
        run.read_nonlinear_fit_config(path)


def test_nonlinear_phases_preserve_failure_denominator_and_resume_before_measured(
        tmp_path, nonlinear_cfg, monkeypatch):
    """Real phase controller with no source arrays or numerical optimization."""
    config = tmp_path/'contract.yaml'
    config.write_text(yaml.safe_dump(nonlinear_cfg))
    out = tmp_path/'run'
    args = SimpleNamespace(config=config, check_only=False, run_dir=out,
        project_root=tmp_path, workers=1, pilot=False, phase='measured')
    events = []
    def serial_pool(worker, payloads, *limits):
        for payload in payloads:
            yield payload, worker(payload), None
    def preparation(payload):
        cfg, directory, project, spec, pilot = payload
        events.append(('prepare', spec['kind']))
        result = dict(status='completed', spec=spec)
        run.write_json(run.Path(directory)/'prepared'/f'{spec["group"]}.json', result)
        return result
    def cell(payload):
        cfg, directory, spec, method, mode, pilot = payload
        events.append(('fit', spec['kind']))
        rows = []
        for trial in range(6):
            rows.append(dict(spec, method=method, mode=mode, trial=trial, session='fixture',
                status='failed_numerical' if trial == 0 else 'completed', converged=trial != 0,
                **{f'nmse_{c}': None if trial == 0 else 4. for c in run.MODALITIES}))
        identifier = run.nonlinear_cell_id(spec, method, mode)
        result = dict(cell=identifier, status='completed', rows=rows, seconds=0.)
        run.write_json(run.Path(directory)/'cells'/identifier/'result.json', result)
        return result
    monkeypatch.setattr(run, 'bounded_nonlinear_work', serial_pool)
    monkeypatch.setattr(run, 'prepare_nonlinear_group', preparation)
    monkeypatch.setattr(run, 'nonlinear_cell', cell)
    monkeypatch.setattr(run, 'load_measured', lambda *a: pytest.fail('retained data must not be read'))
    with pytest.raises(ValueError, match='synthetic'):
        run.nonlinear_fit_main(args)
    assert not (out/'manifest.json').exists() and events == []
    args.phase = 'synthetic'
    run.nonlinear_fit_main(args)
    assert all(kind == 'synthetic' for _, kind in events)
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['execution'] == 'synthetic_terminal'
    first_count = len(events)
    args.phase = 'measured'
    run.nonlinear_fit_main(args)
    assert all(kind == 'measured' for _, kind in events[first_count:])
    summary = json.loads((out/'summary.json').read_text())
    assert summary['expected_cells'] == summary['terminal_cells'] == 288
    assert summary['expected_rows'] == summary['observed_rows'] == 1728
    assert summary['completed_rows'] == 1440
    for aggregate in summary['aggregates']:
        assert aggregate['expected'] == aggregate['observed_rows'] == 72
        assert aggregate['completed'] == 60 and aggregate['failure_counts']['failed_numerical'] == 12
        assert aggregate['nrmse_EEG'] == pytest.approx(2.)
    with pytest.raises(ValueError, match='immutable'):
        run.nonlinear_fit_main(args)


def test_nonlinear_cell_scores_full_map_and_never_promotes_last_valid_failure(
        tmp_path, nonlinear_cfg, monkeypatch):
    from src.inference import shared_driver_reconstruction as library
    spec = dict(kind='synthetic', group='fixture', subject='fixture', outer=0, true_tau=2.)
    metadata = dict(validation=list(range(6)), train=list(range(6,24)))
    run.write_json(tmp_path/'prepared'/'fixture.json', dict(status='completed', metadata=metadata,
        selected_tau={m: 2. for m in run.NONLINEAR_METHODS}))
    arrays = dict(feature_eeg=np.zeros((24,120)), feature_fnirs=np.zeros((24,300,2)),
        target=np.zeros((24,120,3)), normalizer=np.ones(3),
        truth=np.broadcast_to(np.sin(np.arange(120)),(24,120)), truth_states=np.ones((24,120,6)))
    np.savez(tmp_path/'prepared'/'fixture.npz', **arrays)
    state = np.tile([0., 0., 1., 1., 1., 1.], (120,1))
    monkeypatch.setattr(run, 'build_shared_driver_design', lambda *a, **k: None)
    monkeypatch.setattr(run, 'fit_shared_driver', lambda *a, **k: dict(states=state, driver=np.zeros(120)))
    # Full score map differs from the visible fitting map: a sentinel prediction
    # returned by the solver must never be confused with the full target map.
    monkeypatch.setattr(run, 'view_operators', lambda cfg, mode: (None,None,2.*np.eye(360),None))
    mask = np.ones((120,3), bool)
    mask[52:68,1:] = False
    observation = np.zeros((120,3)); observation[~mask] = np.nan
    monkeypatch.setattr(run, 'view', lambda *a: (observation,3.*np.eye(360),mask))
    required = ['finite','positive_fvpq','oxygen_extraction_in_unit_interval',
        'absolute_hb_nonnegative','hbr_not_above_hbt','rest_equilibrium']
    physical = dict(all_intermediate_valid=True, completed_states={key: True for key in required})
    calls = []
    def fake_fit(y, parameters, dt, **kwargs):
        index = len(calls); calls.append(index)
        np.testing.assert_array_equal(kwargs['visible'], mask)
        np.testing.assert_array_equal(kwargs['mean_operator'], 3.*np.eye(360))
        assert len(kwargs['starts']) == 2
        if index == 1:
            return dict(status='failed_domain', converged=False, starts=[dict(status='infeasible_initial')])
        return dict(status='failed_numerical' if index == 0 else 'completed', converged=index!=0,
            starts=[dict(status='fixture_linear'),dict(status='fixture_rest')],
            prediction=np.full((120,3),999.), canonical_prediction=np.ones((120,3)),
            states=state, driver=np.zeros(120), initial_state=state[0,1:], physical_checks=physical,
            objective=1.)
    monkeypatch.setattr(library, 'fit_nonlinear_shared_driver', fake_fit)
    monkeypatch.setattr(run, 'replay_nonlinear_driver', lambda *a, **k: dict(status='completed',
        canonical_prediction=np.ones((120,3)), states=state, failure=None))
    result = run.nonlinear_cell((nonlinear_cfg,tmp_path,spec,'fixed_tau_0p01','center_fNIRS',False))
    assert len(result['rows']) == 6
    assert result['rows'][0]['last_valid_nmse_HbO'] == pytest.approx(4.)
    assert result['rows'][0]['nmse_HbO'] is None
    assert result['rows'][1]['status'] == 'failed_domain'
    for row in result['rows'][2:]:
        assert row['status'] == 'completed' and row['physical_valid']
        assert row['nmse_HbO'] == pytest.approx(4.) and row['nmse_HbR'] == pytest.approx(4.)
        assert row['nmse_EEG'] is None
    assert result['trial_results'][2]['integration_check']['canonical_max_abs_difference'] == 0.
    summary = run.summarize_nonlinear_fit(tmp_path, [(spec,'fixed_tau_0p01','center_fNIRS')], False)
    assert summary['expected_rows'] == summary['observed_rows'] == 6
    assert summary['completed_rows'] == 4
    with np.load(tmp_path/'cells'/'fixture__fixed_tau_0p01__center_fNIRS'/'trajectories.npz') as saved:
        np.testing.assert_array_equal(saved['prediction'][0], 2.*np.ones((120,3)))
        assert np.isnan(saved['prediction'][1]).all()


def test_nonlinear_training_profiles_ignore_validation_values(tmp_path, nonlinear_cfg, monkeypatch):
    target = np.ones((24,120,3))
    arrays = dict(target=target, feature_eeg=np.zeros((24,120)), feature_fnirs=np.zeros((24,300,2)), normalizer=np.ones(3))
    metadata = dict(train=list(range(18)), validation=list(range(18,24)))
    spec = dict(kind='measured',group='subject_01_o0',subject='subject_01',outer=0)
    monkeypatch.setattr(run, 'load_measured', lambda *a: (arrays, metadata))
    monkeypatch.setattr(run, 'tau_grid', lambda cfg: np.array([1.,2.,4.]))
    monkeypatch.setattr(run, 'build_shared_driver_design', lambda p,*a,**k: SimpleNamespace(tau=p.free.tau))
    def fake_fit(y, design, **kwargs):
        assert kwargs['penalty'] == .01 and kwargs['initial_penalty'] == 100.
        return dict(weighted_data_sse=(np.mean(y)-design.tau)**2, penalty=.01,
            roughness=0., initial_penalty=100., initial_state=np.zeros(5))
    monkeypatch.setattr(run, 'fit_shared_driver', fake_fit)
    results = []
    for index, (training, validation) in enumerate([(2.,4.),(2.,1e6),(4.,1e6)]):
        target[:18] = training; target[18:] = validation
        results.append(run.prepare_nonlinear_group((nonlinear_cfg,tmp_path/str(index),tmp_path,spec,False)))
    assert all(r['status'] == 'completed' for r in results)
    assert results[0]['profiles'] == results[1]['profiles']
    assert results[0]['selected_tau'] == results[1]['selected_tau']
    assert results[0]['selected_tau']['linear_trained_tau_0p01'] == 2.
    assert results[2]['selected_tau']['linear_trained_tau_0p01'] == 4.


@pytest.fixture
def gain_cfg():
    return run.read_gain_prior_config(run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_gain_prior_v1.yaml')


def test_gain_prior_contract_has_independent_training_starts_and_fixed_scope(gain_cfg):
    synthetic=run.gain_prior_group_specs(gain_cfg,'synthetic')
    measured=run.gain_prior_group_specs(gain_cfg,'measured')
    assert len(synthetic)==len(measured)==12
    assert {s['true_gain'] for s in synthetic}=={.5,1.,2.}
    assert all(s['true_tau']==2. for s in synthetic)
    assert len(synthetic)*2*len(gain_cfg['gain_training']['start_values'])==72
    assert len(measured)*len(gain_cfg['methods'])*len(gain_cfg['modes'])*6==864


@pytest.mark.parametrize('field,value', [('subjects',['subject_24']),('reference_tau_s',4.),('methods',['fixed_gain_amplitude'])])
def test_gain_prior_rejects_changed_scope_without_array_access(tmp_path,gain_cfg,field,value):
    gain_cfg[field]=value
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(gain_cfg))
    with pytest.raises(ValueError,match='contract mismatch'):run.read_gain_prior_config(path)


def test_gain_synthetic_preparation_preserves_gain_and_training_scale(tmp_path,gain_cfg,monkeypatch):
    spec=run.gain_prior_group_specs(gain_cfg,'synthetic')[-1]
    arrays=dict(target=np.ones((24,120,3))*7,normalizer=np.array([.3,.4,.5]))
    meta=dict(train=list(range(18)),validation=list(range(18,24)))
    def generate(cfg,actual):
        assert cfg['fixed']['neurovascular_gain']==actual['true_gain']==2.
        assert cfg['reference_tau_s']==2.
        return arrays,meta
    monkeypatch.setattr(run,'nonlinear_synthetic_panel',generate)
    result=run.prepare_gain_prior_group((gain_cfg,tmp_path,tmp_path,spec,False))
    assert result['status']=='completed'
    assert result['driver_prior_sd']==.3
    assert result['profiles']==[] and set(result['selected_tau'].values())=={2.}
    with np.load(tmp_path/'prepared'/f"{spec['group']}.npz") as saved:
        np.testing.assert_array_equal(saved['target'],arrays['target'])
        np.testing.assert_array_equal(saved['normalizer'],arrays['normalizer'])
    assert gain_cfg['fixed']['neurovascular_gain']==1.


def test_gain_training_reads_only_train_targets_and_uses_one_prior_per_group(tmp_path,gain_cfg,monkeypatch):
    from src.inference import shared_driver_reconstruction as library
    spec=dict(kind='measured',group='subject_01_o0',subject='subject_01',outer=0)
    metadata=dict(train=list(range(18)),validation=list(range(18,24)))
    state=np.tile([0.,0.,1.,1.,1.,1.],(120,1));calls=[]
    def optimizer(y,p,dt,**kwargs):
        calls.append(y.copy())
        assert y.shape==(18,120,3) and p.free.tau==2.
        assert kwargs['parameter_name']=='neurovascular_gain'
        assert kwargs['parameter_prior_mean']==1.
        assert kwargs['parameter_prior_log_sd']==pytest.approx(np.log(2.))
        assert kwargs['driver_amplitude_weight']==1. and kwargs['driver_prior_sd']==.2
        assert kwargs['max_evaluations']==1800 and kwargs['max_iterations']==60
        assert len(kwargs['starts'])==1 and kwargs['starts'][0]['parameter_value']==.5
        np.testing.assert_array_equal(kwargs['starts'][0]['driver'],np.zeros((18,120)))
        return dict(status='completed',converged=True,parameter_value=float(y.mean()),
            objective=float(y.mean()),expected_trials=18,completed_trials=18,
            driver=np.zeros((18,120)),initial_state=np.tile(state[0,1:],(18,1)),
            states=np.tile(state,(18,1,1)),prediction=np.zeros((18,120,3)),canonical_prediction=np.zeros((18,120,3)))
    monkeypatch.setattr(library,'fit_nonlinear_shared_parameter',optimizer)
    monkeypatch.setattr(run,'replay_nonlinear_driver',lambda *a,**k:dict(status='completed',states=state,canonical_prediction=np.zeros((120,3))))
    results=[]
    for i,(train,test) in enumerate([(2.,3.),(2.,1e6),(4.,1e6)]):
        out=tmp_path/str(i);target=np.full((24,120,3),train);target[18:]=test
        run.write_json(out/'prepared'/f"{spec['group']}.json",dict(status='completed',metadata=metadata,driver_prior_sd=.2))
        np.savez(out/'prepared'/f"{spec['group']}.npz",target=target,normalizer=np.array([.2,.3,.4]))
        results.append(run.train_gain_prior_start((gain_cfg,out,spec,1,0,.5)))
    assert all(r['status']=='completed' for r in results)
    assert results[0]['parameter_value']==results[1]['parameter_value']==2.
    assert results[2]['parameter_value']==4.
    np.testing.assert_array_equal(calls[0],calls[1])
    assert len(results[0]['integration_checks'])==18


def test_gain_selection_waits_for_every_start_and_never_uses_failed_best(tmp_path):
    spec=dict(kind='synthetic',group='fixture')
    directory=run.gain_training_directory(tmp_path,spec,1)
    records=[dict(status='failed_numerical',converged=False,objective=.01,parameter_value=8.),
             dict(status='completed',converged=True,objective=4.,parameter_value=2.),
             dict(status='completed',converged=True,objective=3.,parameter_value=1.5)]
    for index,r in enumerate(records):
        run.write_json(directory/f'start_{index}'/'result.json',dict(r,expected_trials=18,completed_trials=18))
        if index<2:
            with pytest.raises(ValueError,match='all declared'):run.select_gain_prior_training(tmp_path,spec,1,[0,1,2])
    selected=run.select_gain_prior_training(tmp_path,spec,1,[0,1,2])
    assert selected['parameter_value']==1.5 and selected['selected_training_start']==2
    assert selected['successful_starts']==2


def test_all_failed_gain_training_propagates_complete_validation_denominator(tmp_path,gain_cfg,monkeypatch):
    from src.inference import shared_driver_reconstruction as library
    spec=dict(kind='measured',group='subject_01_o0',subject='subject_01',outer=0)
    directory=run.gain_training_directory(tmp_path,spec,1)
    for index in range(3):
        run.write_json(directory/f'start_{index}'/'result.json',dict(status='failed_numerical',converged=False,
            expected_trials=18,completed_trials=18,objective=index,parameter_value=2.))
    result=run.select_gain_prior_training(tmp_path,spec,1,[0,1,2])
    assert result['status']=='failed_training' and result['parameter_value'] is None
    run.write_json(tmp_path/'prepared'/f"{spec['group']}.json",dict(status='completed',
        metadata=dict(validation=[0,4,8,12,16,20]),selected_tau={m:2. for m in run.GAIN_PRIOR_METHODS}))
    np.savez(tmp_path/'prepared'/f"{spec['group']}.npz",target=np.zeros((24,120,3)),normalizer=np.ones(3))
    monkeypatch.setattr(run,'build_shared_driver_design',lambda *a,**k:None)
    def forbidden(*a,**k):raise AssertionError('failed training reached a fit')
    monkeypatch.setattr(library,'fit_nonlinear_shared_driver',forbidden)
    result=run.nonlinear_cell((gain_cfg,tmp_path,spec,'trained_gain_amplitude','full',False))
    assert len(result['rows'])==6 and {r['status'] for r in result['rows']}=={'failed_training'}
    assert all(r['neurovascular_gain'] is None and not r['converged'] for r in result['rows'])
    summary=run.summarize_gain_prior(tmp_path,[(spec,'trained_gain_amplitude','full')],False)
    assert summary['expected_rows']==summary['observed_rows']==6 and summary['completed_rows']==0


def test_gain_measured_phase_rejects_missing_synthetic_before_preparation(tmp_path,gain_cfg,monkeypatch):
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(gain_cfg))
    args=SimpleNamespace(config=config,check_only=False,phase='measured',pilot=False,workers=1,
        run_dir=tmp_path/gain_cfg['output_root']/'fixture',project_root=tmp_path)
    def forbidden(*a,**k):raise AssertionError('measured preparation accessed before synthetic terminal')
    monkeypatch.setattr(run,'prepare_gain_prior_group',forbidden)
    with pytest.raises(ValueError,match='complete synthetic'):run.gain_prior_fit_main(args)


@pytest.mark.parametrize('section,field,value', [
    ('gain_training','parameter_name','tau'),
    ('regularization','driver_prior_sd','fit_all_observations'),
    ('fixed','eeg_loading',2.),('fixed','neurovascular_gain',2.),
    ('nonlinear','maximum_integration_difference_training_sd',float('nan')),
    ('synthetic','fast_fraction',2.),('synthetic','driver_sd',0.),
])
def test_gain_semantic_and_numeric_metadata_cannot_silently_disagree(tmp_path,gain_cfg,section,field,value):
    gain_cfg[section][field]=value
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(gain_cfg))
    with pytest.raises(ValueError):run.read_gain_prior_config(path)


def test_gain_validation_freezes_gain_and_amplitude_scale(tmp_path,gain_cfg,monkeypatch):
    from src.inference import shared_driver_reconstruction as library
    spec=dict(kind='measured',group='subject_01_o0',subject='subject_01',outer=0)
    run.write_json(tmp_path/'prepared'/f"{spec['group']}.json",dict(status='completed',
        metadata=dict(validation=[0,4,8,12,16,20]),selected_tau={m:2. for m in run.GAIN_PRIOR_METHODS}))
    run.write_json(run.gain_training_directory(tmp_path,spec,1)/'selection.json',
        dict(status='completed',parameter_value=2.5))
    np.savez(tmp_path/'prepared'/f"{spec['group']}.npz",target=np.ones((24,120,3))*1e6,
        normalizer=np.array([.2,.3,.4]),feature_eeg=np.zeros((24,120)),feature_fnirs=np.zeros((24,300,2)))
    state=np.tile([0.,0.,1.,1.,1.,1.],(120,1));calls=[]
    monkeypatch.setattr(run,'build_shared_driver_design',lambda *a,**k:None)
    monkeypatch.setattr(run,'fit_shared_driver',lambda *a,**k:dict(driver=np.zeros(120),states=state))
    def fit(y,p,dt,**kwargs):
        calls.append(kwargs)
        assert p.free.tau==2. and p.fixed.neurovascular_gain==2.5
        assert kwargs['driver_amplitude_weight']==1. and kwargs['driver_prior_sd']==.2
        assert kwargs['penalty']==.01 and kwargs['initial_penalty']==100. and kwargs['max_evaluations']==160
        return dict(status='failed_domain',converged=False,starts=[])
    monkeypatch.setattr(library,'fit_nonlinear_shared_driver',fit)
    result=run.nonlinear_cell((gain_cfg,tmp_path,spec,'trained_gain_amplitude','center_EEG',False))
    assert len(calls)==6 and len(result['rows'])==6
    assert all(r['neurovascular_gain']==2.5 and r['status']=='failed_domain' for r in result['rows'])


@pytest.fixture
def budget_cfg():
    return run.read_gain_prior_config(run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_gain_prior_budget_v1.yaml')


@pytest.fixture
def continuation_parent(tmp_path,gain_cfg,budget_cfg):
    """Complete JSON-only parent: no actual retained or protected artifacts."""
    root=tmp_path/budget_cfg['continuation']['parent_run']
    run.write_json(root/'manifest.json',dict(execution='completed',pilot=False,
        experiment_id=gain_cfg['experiment_id'],synthetic_terminal=True,measured_terminal=True,
        project_root=str(tmp_path),source_root=str(root/'source_snapshot')))
    (root/'resolved_config.yaml').write_text(yaml.safe_dump(gain_cfg))
    eligible=0
    for kind in ('synthetic','measured'):
        for spec in run.gain_prior_group_specs(gain_cfg,kind):
            val=[i for i in range(24) if i%4==spec['outer']];train=[i for i in range(24) if i not in val]
            run.write_json(root/'prepared'/f"{spec['group']}.json",dict(spec=spec,status='completed',metadata=dict(train=train,validation=val,subject=spec['subject'],outer=spec['outer'],
                    trials=[dict(session=session,original_ma_trial_position=pos) for session in gain_cfg['sessions'] for pos in gain_cfg['original_positions']])))
            for amp in (0,1):
                folder=run.gain_training_directory(root,spec,amp)
                run.write_json(folder/'selection.json',dict(expected_starts=3,terminal_starts=3))
                for idx,gain in enumerate(gain_cfg['gain_training']['start_values']):
                    restart=kind=='measured' and eligible<35
                    if restart:eligible+=1
                    record=dict(spec=spec,start_index=idx,start_gain=gain,amplitude_weight=amp,
                        training_trials=train,status='failed_numerical' if restart else 'completed',converged=not restart,
                        expected_trials=18,completed_trials=18,objective=10.,parameter_value=.8,
                        physical_checks=[dict(all_intermediate_valid=True)]*18,
                        starts=[dict(convergence_reason='evaluation_budget' if restart else 'projected_scaled_gradient',domain_rejections=0,derivative_rejections=0)])
                    run.write_json(folder/f'start_{idx}'/'result.json',record)
            for method in run.GAIN_PRIOR_METHODS:
                for mode in gain_cfg['modes']:
                    rows=[dict(spec,method=method,mode=mode,trial=i,status='completed',converged=True) for i in val]
                    run.write_json(root/'cells'/run.nonlinear_cell_id(spec,method,mode)/'result.json',
                        dict(spec=spec,method=method,mode=mode,rows=rows))
    return root


def test_continuation_checks_complete_parent_without_reading_arrays(tmp_path,budget_cfg,continuation_parent,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('metadata check must not read arrays')
    monkeypatch.setattr(np,'load',forbidden)
    assert run.validate_gain_continuation(budget_cfg,tmp_path)==continuation_parent


@pytest.mark.parametrize('change',['objective','unfinished','identity','eligible_count'])
def test_continuation_rejects_parent_contract_changes(tmp_path,budget_cfg,continuation_parent,change):
    if change=='objective':budget_cfg['regularization']['driver_curvature_weight']=.02
    if change=='unfinished':
        p=continuation_parent/'manifest.json';d=json.loads(p.read_text());d['execution']='running';run.write_json(p,d)
    if change in ('identity','eligible_count'):
        p=continuation_parent/'training/subject_01_o0__amplitude0/start_0/result.json';d=json.loads(p.read_text())
        if change=='identity':d['training_trials'][0]=0
        else:d['starts'][0]['domain_rejections']=1
        run.write_json(p,d)
    with pytest.raises(ValueError):run.validate_gain_continuation(budget_cfg,tmp_path)


@pytest.mark.parametrize('field,value',[('domain_rejections',1),('derivative_rejections',1),('convergence_reason','iteration_budget')])
def test_continuation_eligibility_is_not_all_numerical_failures(field,value):
    d=dict(status='failed_numerical',converged=False,expected_trials=18,completed_trials=18,
        objective=3.,parameter_value=1.,physical_checks=[dict(all_intermediate_valid=True)]*18,
        starts=[dict(convergence_reason='evaluation_budget',domain_rejections=0,derivative_rejections=0)])
    assert run.gain_restart_eligible(d)
    d['starts'][0][field]=value
    assert not run.gain_restart_eligible(d)


@pytest.mark.parametrize('bad',['target','normalizer','trial_indices'])
def test_continuation_rejects_training_array_identity_mismatch(tmp_path,bad):
    target=np.zeros((18,120,3));sd=np.ones(3);train=list(range(18));a=dict(target=target.copy(),normalizer=sd.copy(),trial_indices=np.array(train),driver=np.ones((18,120)),initial_state=np.ones((18,5)))
    a[bad].flat[0]+=1
    p=tmp_path/'training.npz';np.savez(p,**a)
    with pytest.raises(ValueError,match='target/normalizer/train_indices'):
        run.gain_restart_arrays(dict(parameter_value=2.),p,target,sd,train)


def test_continuation_inherits_failure_without_changing_primary_status(tmp_path):
    source=tmp_path/'parent/result.json';destination=tmp_path/'new/result.json'
    original=dict(status='failed_numerical',converged=False,objective=9.)
    run.write_json(source,original);np.savez(source.parent/'trajectories.npz',driver=np.ones(3))
    before=source.read_bytes();result=run.inherit_gain_evidence(source,destination,action='inherited_training')
    assert result['status']=='failed_numerical' and not result['converged']
    assert result['continuation']['inherited'] and source.read_bytes()==before
    assert (source.parent/'trajectories.npz').read_bytes()==(destination.parent/'trajectories.npz').read_bytes()
    assert not (destination.parent/'trajectories.npz').is_symlink()


def test_continuation_fixed_cells_retain_all_failed_rows(tmp_path,budget_cfg,monkeypatch):
    parent=tmp_path/'parent';out=tmp_path/'new';spec=run.gain_prior_group_specs(budget_cfg,'measured')[0]
    method='fixed_gain_amplitude';mode='full';identifier=run.nonlinear_cell_id(spec,method,mode)
    rows=[dict(spec,method=method,mode=mode,trial=i,status='failed_numerical',converged=False) for i in [0,4,8,12,16,20]]
    run.write_json(parent/'cells'/identifier/'result.json',dict(spec=spec,method=method,mode=mode,rows=rows))
    np.savez(parent/'cells'/identifier/'trajectories.npz',prediction=np.zeros((6,120,3)))
    run.write_json(out/'prepared'/f"{spec['group']}.json",dict(continuation=dict(parent_record=str(parent/'prepared'/f"{spec['group']}.json"))))
    def forbidden(*a,**k):raise AssertionError('fixed gain inheritance must not fit')
    monkeypatch.setattr(run,'build_shared_driver_design',forbidden)
    result=run.nonlinear_cell((budget_cfg,out,spec,method,mode,False))
    assert result['rows']==rows and len(result['rows'])==6
    assert result['continuation']['action']=='inherited_validation'


@pytest.mark.parametrize('continuous',[True,False])
def test_continuation_uses_only_parent_training_last_solution_and_checks_objective(tmp_path,budget_cfg,monkeypatch,continuous):
    from src.inference import shared_driver_reconstruction as library
    parent=tmp_path/'parent';out=tmp_path/'new';spec=run.gain_prior_group_specs(budget_cfg,'measured')[0]
    val=[0,4,8,12,16,20];train=[i for i in range(24) if i not in val]
    target=np.zeros((24,120,3));target[val]=1e9;sd=np.array([.2,.3,.4])
    run.write_json(out/'prepared'/f"{spec['group']}.json",dict(status='completed',metadata=dict(train=train,validation=val),
        driver_prior_sd=.2,continuation=dict(parent_record=str(parent/'prepared'/f"{spec['group']}.json"))))
    np.savez(out/'prepared'/f"{spec['group']}.npz",target=target,normalizer=sd)
    directory=run.gain_training_directory(parent,spec,1)/'start_0'
    d=dict(status='failed_numerical',converged=False,expected_trials=18,completed_trials=18,objective=10.,parameter_value=.8,
        evaluations=1818,optimization_evaluations=1800,physical_checks=[dict(all_intermediate_valid=True)]*18,
        starts=[dict(convergence_reason='evaluation_budget',domain_rejections=0,derivative_rejections=0)])
    run.write_json(directory/'result.json',d)
    driver=np.ones((18,120))*.125;initial=np.tile([0.,1.,1.,1.,1.],(18,1))
    np.savez(directory/'trajectories.npz',target=target[train],normalizer=sd,trial_indices=train,driver=driver,initial_state=initial)
    state=np.tile([.125,0.,1.,1.,1.,1.],(120,1))
    def fit(y,p,dt,**kw):
        np.testing.assert_array_equal(y,target[train]);assert y.max()==0
        np.testing.assert_array_equal(kw['starts'][0]['driver'],driver)
        np.testing.assert_array_equal(kw['starts'][0]['initial_state'],initial)
        assert kw['starts'][0]['parameter_value']==.8 and kw['record_trace'] is True
        assert kw['max_evaluations']==3600 and kw['max_iterations']==90
        return dict(status='completed',converged=True,parameter_value=.9,objective=8.,expected_trials=18,completed_trials=18,
            evaluations=54,optimization_evaluations=36,trace=[dict(objective=10. if continuous else 11.)],
            driver=driver,initial_state=initial,states=np.tile(state,(18,1,1)),prediction=np.zeros((18,120,3)))
    monkeypatch.setattr(library,'fit_nonlinear_shared_parameter',fit)
    monkeypatch.setattr(run,'replay_nonlinear_driver',lambda *a,**k:dict(status='completed',states=state,canonical_prediction=np.zeros((120,3))))
    result=run.train_gain_prior_start((budget_cfg,out,spec,1,0,.5))
    assert result['status']==('completed' if continuous else 'failed_objective_continuity')
    assert result['converged']==continuous
    assert result['continuation']['cumulative_evaluations']==1872
    assert result['continuation']['objective_continuity']==continuous
