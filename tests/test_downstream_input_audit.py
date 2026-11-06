"""Development-only diagnostics use actual task support and train-only fits."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from comparative_methods.performance_analysis import audit_downstream_inputs_v1 as audit
from src.metrics.downstream_benchmark import score_predictions


def test_sta_optimization_contract_rejects_new_test_scope_or_pretrained_finetuning(tmp_path):
    import yaml
    cfg = audit.read_config(Path(__file__).parents[1] / "comparative_methods/downstream_sta_optimization_v1.yaml")
    assert sum(len(spec["tasks"]) for spec in cfg["arms"].values()) == 12
    for field,value in [("source_partitions",["train","validation","test"]),
                        ("initialization","finetune_existing_backbone"),("refed_label","validation_selected_threshold")]:
        path=tmp_path/"bad.yaml";path.write_text(yaml.safe_dump({**cfg,field:value}))
        with pytest.raises(ValueError,match="fixed development"):
            audit.read_config(path)
    root=Path(__file__).parents[1]/"comparative_methods"
    single=audit.read_config(root/"downstream_sta_refed_source_v3.yaml")
    assert single["tasks"]==["refed_regression"]
    assert audit.sta_optimization_arm(single,tmp_path,"refed_regression","selected")[0]=="source_recipe"
    normalization=audit.read_config(root/"downstream_sta_refed_normalization_v1.yaml")
    assert normalization["tasks"]==["refed_regression"] and normalization["initialization"]=="retained_parent_parameters_no_gradient_updates"


def test_sta_seed_aggregate_averages_subject_metrics_and_keeps_screen_selection(tmp_path, monkeypatch):
    import json
    cfg = audit.read_config(Path(__file__).parents[1] / "comparative_methods/downstream_sta_optimization_v1.yaml")
    cfg={**cfg,"tasks":["visual"],"confirmation_tasks":["visual"],"baseline_run":"baseline",
         "arms":{name:{"tasks":["visual"]} for name in ["source_recipe","regularized_balanced"]}}
    target=np.tile(np.arange(4),3);subjects=np.array(["a"]*4+["b"]*8)
    a=dict(target=target,mask=np.ones(12,bool),subject_id=subjects,record_id=np.repeat("r",12),
           sample_id=np.array([str(i) for i in range(12)]),partition=np.repeat("validation",12))
    monkeypatch.setattr(audit,"frozen_classification_inputs",lambda *args,**kwargs:(a,{}))
    baseline=tmp_path/"baseline";baseline.mkdir();run=tmp_path/"run";run.mkdir()
    old=score_predictions(target,np.zeros(12,int),a["mask"],subjects,"visual",4,metric_version="v2")
    audit.write(baseline/"summary.json",dict(cases=[dict(method="sta_net",source_task="visual",metric=old)]))
    predictions={('source_recipe',17):target.copy(),('regularized_balanced',17):np.zeros(12,int),
                 ('selected',42):(target+1)%4,('selected',73):np.r_[target[:4],(target[4:]+1)%4]}
    for (arm,seed),prediction in predictions.items():
        root=run/"visual"/arm/f"seed_{seed}";root.mkdir(parents=True)
        np.savez(root/"predictions.npz",prediction=prediction,**a)
        metric=score_predictions(target,prediction,a["mask"],subjects,"visual",4,metric_version="v2")
        audit.write(root/"result.json",dict(status="completed",result=dict(task="visual",source_task="visual",arm="source_recipe" if arm=="selected" else arm,
            requested_arm=arm,seed=seed,metric=metric,B0=dict(b0=.25,subjects=[dict(majority_accuracy=.25)]))))
    screen=audit.sta_optimization_aggregate(cfg,tmp_path,run)
    assert screen["selected_arms"]=={"visual":"source_recipe"}
    audit.sta_optimization_aggregate(cfg,tmp_path,run,final=True)
    value=json.loads((run/"summary.json").read_text())["comparison"][0]
    assert value["arm"]=="source_recipe" and value["seeds"]==3
    # Subject a is correct on two seeds, b only on one; no ensemble voting.
    np.testing.assert_allclose(value["ACC"],.5)
    np.testing.assert_allclose(value["ACC_STD"],np.std([2/3,1/3],ddof=1))
    np.testing.assert_allclose(value["F1_STD"],np.std([2/3,1/3],ddof=1))
    # A statistics revision must not choose a new winner on the same validation
    # people, even if its former alternative now has a higher validation score.
    import shutil
    calibrated=tmp_path/"calibrated"
    shutil.copytree(run,calibrated)
    cfg.update(schema="downstream_sta_normalization_v1",fit_run="run",normalization="fixture_training_only")
    path=calibrated/"visual/regularized_balanced/seed_17"
    np.savez(path/"predictions.npz",prediction=target,**a)
    doc=json.loads((path/"result.json").read_text())
    doc["result"]["metric"]=score_predictions(target,target,a["mask"],subjects,"visual",4,metric_version="v2")
    audit.write(path/"result.json",doc)
    path=calibrated/"visual/source_recipe/seed_17"
    prediction=np.zeros(12,int)
    np.savez(path/"predictions.npz",prediction=prediction,**a)
    doc=json.loads((path/"result.json").read_text())
    doc["result"]["metric"]=score_predictions(target,prediction,a["mask"],subjects,"visual",4,metric_version="v2")
    audit.write(path/"result.json",doc)
    audit.sta_optimization_aggregate(cfg,tmp_path,calibrated,final=True)
    value=json.loads((calibrated/"summary.json").read_text())
    assert value["selected_arms"]=={"visual":"source_recipe"} and value["parameter_updates"]==0


def test_frozen_classification_rejects_test_or_finetuning_contract(tmp_path):
    import yaml
    cfg = audit.read_config(Path(__file__).parents[1] / "comparative_methods/downstream_frozen_classification_v1.yaml")
    for field, value in [("source_partitions", ["train", "validation", "test"]),
                         ("encoder_fitting", "finetune"), ("refed_label", "validation_median")]:
        path = tmp_path / "wrong.yaml"
        path.write_text(yaml.safe_dump({**cfg, field: value}))
        with pytest.raises(ValueError, match="fixed development"):
            audit.read_config(path)


def test_retained_prediction_join_rejects_missing_or_changed_targets(tmp_path):
    a = dict(target=np.array([0, 1, 0]), mask=np.ones(3, bool),
             sample_id=np.array(["a", "b", "c"]), subject_id=np.array(["s1", "s2", "s2"]),
             record_id=np.array(["r", "r", "r"]))
    path = tmp_path / "pred.npz"
    order = np.array([2, 1, 0])
    np.savez(path, prediction=np.array([1, 0, 0]), **{k: v[order] for k, v in a.items()})
    np.testing.assert_array_equal(audit.frozen_classification_prediction(path, a, np.arange(3)), [0, 0, 1])
    with pytest.raises(ValueError, match="exactly"):
        audit.frozen_classification_prediction(path, a, np.arange(2))
    a["target"][1] = 0
    with pytest.raises(AssertionError):
        audit.frozen_classification_prediction(path, a, np.arange(3))


def test_new_binary_readout_is_training_only_and_frozen(tmp_path, monkeypatch):
    rng = np.random.default_rng(55)
    labels = np.tile([0, 1], 12)
    a = dict(target=labels, mask=np.ones(24, bool), subject_id=np.repeat(["a", "b", "c", "d"], 6),
             sample_id=np.array([str(i) for i in range(24)]), record_id=np.repeat("r", 24),
             partition=np.array(["train"] * 12 + ["validation"] * 12))
    x = rng.normal(0, .01, (24, 3)) + labels[:, None] * 3
    cfg = audit.read_config(Path(__file__).parents[1] / "comparative_methods/downstream_frozen_classification_v1.yaml")
    monkeypatch.setattr(audit, "frozen_classification_inputs", lambda *args, **kwargs: (a, {}))
    monkeypatch.setattr(audit, "frozen_classification_features", lambda *args, **kwargs: (x, None, {"encoder_updates": 0, "track": "fixture"}))
    out = tmp_path / "first"; out.mkdir()
    result = audit.frozen_classification_fit(cfg, tmp_path, tmp_path, "refed_regression", "efrm", out, "cpu")
    assert result["metric"]["primary"] == 1 and result["metric"]["subject_sd"] == 0
    with np.load(out / "readout.npz") as saved: first = {k: saved[k] for k in saved.files}
    a["target"][12:] = 1 - a["target"][12:]
    x[12:] += 1e6
    out2 = tmp_path / "second"; out2.mkdir()
    audit.frozen_classification_fit(cfg, tmp_path, tmp_path, "refed_regression", "efrm", out2, "cpu")
    with np.load(out2 / "readout.npz") as saved:
        for key in first: np.testing.assert_array_equal(saved[key], first[key])


def test_fold_encoder_scaler_and_training_features_ignore_held_inputs_and_targets(tmp_path, monkeypatch):
    import json
    import torch
    from types import SimpleNamespace

    cfg = audit.read_config(Path(__file__).parents[1] / 'comparative_methods/downstream_repair_probe_v11.yaml')
    rng = np.random.default_rng(2)
    a = dict(eeg=rng.normal(size=(6, 2, 6000)), fnirs=rng.normal(size=(6, 2, 300)),
             eeg_mask=np.ones((6, 2, 6000), bool), fnirs_mask=np.ones((6, 2, 300), bool),
             target=np.ones((6, 2, 30)), mask=np.ones((6, 2, 30), bool),
             subject_id=np.repeat(['a', 'b', 'c'], 2), sample_id=np.array([str(i) for i in range(6)]),
             record_id=np.array([str(i) for i in range(6)]))
    a['eeg_mask'][0, :, -1000:] = False
    a['fnirs_mask'][0, :, -50:] = False
    native = dict(channel_names=['C3', 'C4'], eeg_geometry=[{}, {}], window_start_s=0., duration_s=30.)
    monkeypatch.setattr(audit, 'probe_inputs', lambda *args, **kwargs: (a, native))
    monkeypatch.setattr(audit, 'view_profile', lambda *args, **kwargs: (dict(native), [0, 1]))
    monkeypatch.setattr(audit, 'efrm_checkpoint', lambda *args: 'fixture_checkpoint')
    cfg['models']['refed_regression'] = {'ridge': cfg['models']['refed_regression']['efrm_ridge']}
    run = tmp_path / 'run'
    audit.probe_plan(cfg, tmp_path, run, tmp_path / 'source', 'downstream_repair_probe_v11.yaml')
    plan = json.loads((run / 'plan.json').read_text())['tasks']['refed_regression']
    jobs = json.loads((run / 'jobs.json').read_text())['jobs']
    for fold in range(3):
        fitting = next(j for j in jobs if j['id'] == f'refed_regression_ridge_fold{fold}')
        assert f'refed_regression_encoder_fold{fold}' in fitting['dependencies']
    train = np.array([plan['subject_folds'][s] != 0 for s in a['subject_id']])

    class Encoder(torch.nn.Module):
        feature_dim = 2
        pretraining_provenance = {'fixture': True}

        def forward(self, eeg, fnirs, **kwargs):
            return torch.stack([eeg.reshape(-1, 2, 30, 200).mean((1, 3)),
                                fnirs.reshape(-1, 2, 30, 10).mean((1, 3))], -1)

        def sequence_feature_mask(self, eeg, fnirs, **kwargs):
            return kwargs['eeg_mask'].reshape(-1, 2, 30, 200).any(3).any(1)

    original_import = audit.importlib.import_module
    monkeypatch.setattr(audit.importlib, 'import_module', lambda name: SimpleNamespace(create_model=lambda *args, **kwargs: Encoder())
                        if name == 'comparative_methods.EFRM-PyTorch.rebench_v1' else original_import(name))
    first = run / 'first'
    descriptor = audit.probe_fold_encoder(cfg, tmp_path, run, 'refed_regression', first, 'cpu', fold=0)
    assert descriptor['training_subjects'] == sorted(set(a['subject_id'][train]))
    assert not np.load(first / 'efrm_sequence_mask.npy')[0, -5:].any()
    a['eeg'][~train] += 1e5
    a['fnirs'][~train] *= 1e4
    a['target'][:] = np.nan
    second = run / 'second'
    audit.probe_fold_encoder(cfg, tmp_path, run, 'refed_regression', second, 'cpu', fold=0)
    assert json.loads((first / 'input_scaler.json').read_text()) == json.loads((second / 'input_scaler.json').read_text())
    np.testing.assert_array_equal(np.load(first / 'efrm_sequence.npy')[train], np.load(second / 'efrm_sequence.npy')[train])
    a['sample_id'][0] = 'changed'
    with pytest.raises(ValueError, match='population changed'):
        audit.probe_fold_encoder(cfg, tmp_path, run, 'refed_regression', run / 'bad', 'cpu', fold=0)


def config():
    return audit.read_config(Path(__file__).parents[1] / "comparative_methods/downstream_input_audit_v1.yaml")


def test_probe_class_weight_contrast_preserves_subject_mass_and_training_boundary():
    a = dict(subject_id=np.array(['a'] * 6 + ['b'] * 4 + ['held'] * 2),
             record_id=np.array(['part'] * 12),
             target=np.array([0, 0, 0, 0, 1, 2, 0, 1, 2, 3, 0, 0]))
    train = np.arange(10)
    unbalanced = audit.probe_weights(a, train, 'visual', class_balance='none')
    balanced = audit.probe_weights(a, train, 'visual')
    np.testing.assert_allclose([unbalanced[:6].sum(), unbalanced[6:].sum()], [5., 5.])
    np.testing.assert_allclose(np.bincount(a['target'][train], weights=balanced), [2.5] * 4)
    assert not np.allclose(unbalanced, balanced)
    a['target'][10:] = -100
    np.testing.assert_array_equal(unbalanced, audit.probe_weights(a, train, 'visual', class_balance='none'))
    np.testing.assert_array_equal(balanced, audit.probe_weights(a, train, 'visual'))
    with pytest.raises(ValueError, match='only for Visual'):
        audit.probe_weights(a, train, 'visual', class_balance='unknown')


@pytest.mark.parametrize('record_weight,freeze_encoder', [(None, False), (.05, False), (.05, True)])
def test_efrm_ft_updates_both_encoders_without_held_target_access(tmp_path, monkeypatch, record_weight, freeze_encoder):
    import json
    import torch
    from types import SimpleNamespace
    torch.set_num_threads(1)
    cfg = audit.read_config(Path(__file__).parents[1] / 'comparative_methods/downstream_repair_probe_v12.yaml')
    cfg['retained_feature_run'] = 'parent'
    spec = dict(cfg['models']['refed_regression']['efrm_temporal_ft_2'], epochs=1, effective_batch_size=2)
    spec['freeze_encoder'] = freeze_encoder
    if record_weight is not None:
        spec.update(record_ccc_weight=record_weight, effective_record_batch_size=1)
    rng = np.random.default_rng(7)
    a = dict(eeg=rng.normal(size=(6, 2, 6000)), fnirs=rng.normal(size=(6, 2, 300)),
             eeg_mask=np.ones((6, 2, 6000), bool), fnirs_mask=np.ones((6, 2, 300), bool),
             target=rng.uniform(30, 220, (6, 2, 30)), mask=np.ones((6, 2, 30), bool),
             subject_id=np.repeat(['a', 'b', 'c'], 2), sample_id=np.array([str(i) for i in range(6)]),
             record_id=np.array(['video'] * 6))
    native = dict(channel_names=['C3', 'C4'], eeg_geometry=[{}, {}], window_start_s=0., duration_s=30.)
    monkeypatch.setattr(audit, 'view_profile', lambda *args, **kwargs: (dict(native), [0, 1]))
    monkeypatch.setattr(audit, 'efrm_checkpoint', lambda *args: 'fixture_checkpoint')
    parent = tmp_path / 'parent'
    audit.write(parent / 'plan.json', dict(parent_run=cfg['parent_run'], tasks=dict(refed_regression=dict(
        sample_ids=a['sample_id'].tolist(), subject_folds={'a': 1, 'b': 2, 'c': 0}))))
    features = parent / 'refed_regression/encoder/fold_0'
    audit.write(features / 'features.json', dict(training_subjects=['a', 'b']))
    audit.write(features / 'input_scaler.json', {m: dict(mean=[[0.], [0.]], scale=[[1.], [1.]]) for m in ['eeg', 'fnirs']})
    head_root = parent / 'refed_regression/efrm_tcn_5/fold_0'
    audit.write(head_root / 'result.json', dict(status='completed', result=dict(training_subjects=['a', 'b'])))
    torch.manual_seed(2)
    head = audit.probe_temporal_head(4, 8, 0.)
    torch.save(dict(feature_dim=4, config=dict(neural_width=8), specification=dict(dropout=0.), state_dict=head.state_dict()), head_root / 'model.pt')
    np.savez(head_root / 'transform.npz', mean=np.zeros(4), scale=np.ones(4), counts=np.ones(4), block_normalizer=np.ones(4))

    class Encoder(torch.nn.Module):
        feature_dim = 4
        pretraining_provenance = {'fixture': True}

        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            for name in ('eeg', 'fnirs'):
                model = torch.nn.Module(); model.patch_embed = torch.nn.Module()
                model.patch_embed.proj = torch.nn.Linear(1, 4, bias=False)
                setattr(self.backbone, name + '_model', model)

        def forward(self, eeg, fnirs, **kwargs):
            return self.backbone.eeg_model.patch_embed.proj(eeg.reshape(-1, 2, 30, 200).mean((1, 3))[..., None]) + self.backbone.fnirs_model.patch_embed.proj(fnirs.reshape(-1, 2, 30, 10).mean((1, 3))[..., None])

        def sequence_feature_mask(self, eeg, fnirs, **kwargs):
            return kwargs['eeg_mask'].reshape(-1, 2, 30, 200).any(3).any(1)

    original_import = audit.importlib.import_module
    monkeypatch.setattr(audit.importlib, 'import_module', lambda name, *args, **kwargs:
        SimpleNamespace(create_model=lambda *args, **kwargs: Encoder().requires_grad_(kwargs['trainable'])) if name == 'comparative_methods.EFRM-PyTorch.rebench_v1'
        else original_import(name, *args, **kwargs))
    train = np.arange(4); weights = audit.probe_weights(a, train, 'refed_regression')
    first = tmp_path / 'first'; first.mkdir()
    p, fit = audit.probe_efrm_ft(a, native, train, weights, spec, cfg, tmp_path, 'cpu', first, fold=0)
    assert all((delta == 0 if freeze_encoder else delta > 0) for delta in fit['encoder_change_l2_by_modality'].values())
    assert fit['head_change_l2'] > 0
    assert (fit['encoder_trainable_parameters'] == 0) == freeze_encoder
    if record_weight is not None:
        assert fit['training_records'] == 2
    assert fit['updates'] == 2
    a['target'][4:] = np.nan
    second = tmp_path / 'second'; second.mkdir()
    q, _ = audit.probe_efrm_ft(a, native, train, weights, spec, cfg, tmp_path, 'cpu', second, fold=0)
    np.testing.assert_array_equal(p, q)
    with pytest.raises(ValueError, match='training crosses'):
        audit.probe_efrm_ft(a, native, np.arange(5), weights, spec, cfg, tmp_path, 'cpu', tmp_path / 'bad', fold=0)


def test_scope_cannot_include_test(tmp_path):
    import yaml
    cfg = config(); cfg["partitions"].append("test")
    path = tmp_path / "bad.yaml"; path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="train/validation only"):
        audit.read_config(path)


@pytest.mark.parametrize('components,times', [(1, 2400), (200, 12)])
def test_compact_event_head_preserves_fold_scale_and_ignores_held_labels(tmp_path, components, times):
    import torch
    torch.set_num_threads(1)
    cfg = audit.read_config(Path(__file__).parents[1] / 'comparative_methods/downstream_repair_probe_v13.yaml')
    cfg['fit'].update(neural_epochs=2, neural_batch_size=2)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6, components, 16, times)).astype(np.float32)
    support = np.ones(x.shape, bool); train = np.arange(4)
    transformed, state = audit.probe_event_transform(x, support, train)
    changed = x.copy(); changed[4:] += 1000
    same_train, other_state = audit.probe_event_transform(changed, support, train)
    np.testing.assert_array_equal(state['mean'], other_state['mean'])
    np.testing.assert_array_equal(transformed[train], same_train[train])
    a = dict(target=np.array([0, 1, 2, 3, 0, 1]), mask=np.ones(6, bool))
    spec = dict(estimator='event_conv', auxiliary_weight=.5)
    first = tmp_path / 'first'; first.mkdir()
    p, fit = audit.probe_neural(transformed, a, train, np.ones(4), 'visual', spec, cfg, 'cpu', first)
    assert p.shape == (6,) and set(p).issubset(range(4))
    assert fit['trainable_parameters'] < 10000
    a['target'][4:] = [3, 2]
    second = tmp_path / 'second'; second.mkdir()
    q, _ = audit.probe_neural(transformed, a, train, np.ones(4), 'visual', spec, cfg, 'cpu', second)
    np.testing.assert_array_equal(p, q)
    support[0, 0, 0, 0] = False
    with pytest.raises(ValueError, match='complete'):
        audit.probe_event_transform(x, support, train)


def test_frontal_projection_is_window_local_and_removes_known_synthetic_interference():
    t = np.arange(6000) / 200.
    blink_proxy = 20 * np.sin(2 * np.pi * 1.5 * t)
    horizontal_proxy = 8 * np.sin(2 * np.pi * .8 * t)
    brain = np.sin(2 * np.pi * 11 * t)
    x = np.stack([blink_proxy + horizontal_proxy, blink_proxy - horizontal_proxy,
                  brain + 2 * blink_proxy + .7 * horizontal_proxy])[None]
    x = np.concatenate((x, x * 2), axis=0)
    first, ratio = audit.probe_frontal_projection(x, ['Fp1', 'Fp2', 'C3'])
    assert first.shape == x.shape and ratio.shape == (2, 3)
    assert np.sqrt(np.mean((first[0, 2, 500:-500] - brain[500:-500]) ** 2)) < .1
    x[1] *= 1e4
    second, _ = audit.probe_frontal_projection(x, ['Fp1', 'Fp2', 'C3'])
    np.testing.assert_array_equal(first[0], second[0])
    zero, fraction = audit.probe_frontal_projection(np.zeros((1, 3, 6000)), ['Fp1', 'Fp2', 'C3'])
    assert not zero.any() and not fraction.any()
    with pytest.raises(ValueError, match='physical Fp1/Fp2'):
        audit.probe_frontal_projection(x, ['C1', 'C2', 'C3'])


def test_bipolar_dataset_differences_physical_voltage_before_scaling_and_intersects_masks():
    a = dict(eeg=np.tile(np.array([[[2., 4., 6., 8.], [1., 3., 2., 1.]]]), (1, 1, 5)), fnirs=np.ones((1, 2, 1)),
             eeg_mask=np.tile(np.array([[[True, True, False, True], [True, False, True, True]]]), (1, 1, 5)), fnirs_mask=np.ones((1, 2, 1), bool),
             target=np.array([0]), mask=np.ones(1, bool), sample_id=np.array(['one']), subject_id=np.array(['a']), record_id=np.array(['r']))
    native = dict(channel_names=['Fp1', 'F3'], window_start_s=0., duration_s=.1)
    derived = dict(native, channel_names=['Fp1-F3'], eeg_derivations=[['Fp1', 'F3']])
    dataset = audit.ArrayDataset(a, native, derived, [0])
    sample = dataset[0]
    np.testing.assert_array_equal(sample['eeg'], np.tile([[1., 1., 4., 7.]], (1, 5)))
    np.testing.assert_array_equal(sample['channel_valid_mask']['eeg'], np.tile([[True, False, False, True]], (1, 5)))
    scaler = audit.fit_input_scaler(audit.ArrayDataset(a, native, derived, [0]))
    np.testing.assert_allclose(scaler['eeg']['mean'], [[4.]])
    np.testing.assert_allclose(scaler['eeg']['scale'], [[3.]])
    with pytest.raises(ValueError, match='names disagree'):
        audit.ArrayDataset(a, native, dict(derived, channel_names=['F3-Fp1']), [0])


def test_followup_reuses_frozen_split_and_has_a_finite_correction_scope():
    path = Path(__file__).parents[1] / "comparative_methods/downstream_input_followup_v1.yaml"
    cfg = audit.read_config(path)
    assert cfg["partitions"] == ["train", "validation"]
    assert cfg["input_cache"] == config()["input_cache"]
    assert cfg["followup"]["brainfusion"]["HRF_step_s"] == .1
    assert cfg["followup"]["refed"]["actual_sequence_steps"] == 30


def test_fixed_probe_predictions_ignore_validation_targets(tmp_path):
    labels = np.tile([0, 1], 30)
    a = dict(target=labels, mask=np.ones(60, bool), sample_id=np.asarray([str(i) for i in range(60)]),
             subject_id=np.repeat(["train_a", "train_b", "validation"], 20), record_id=np.repeat(["a", "b", "c"], 20),
             partition=np.asarray(["train"] * 40 + ["validation"] * 20))
    x = (labels * 2 - 1).reshape(-1, 1).astype(float)
    first = audit.linear_grid(x, a, "motor_imagery", config(), tmp_path, "first", fixed=True)
    assert first["selected"]["validation"]["primary"] == 1
    changed = deepcopy(a); changed["target"][40:] = 1 - changed["target"][40:]
    second = audit.linear_grid(x, changed, "motor_imagery", config(), tmp_path, "second", fixed=True)
    assert second["selected"]["validation"]["primary"] == 0
    with np.load(tmp_path / "first_predictions.npz") as p, np.load(tmp_path / "second_predictions.npz") as q:
        np.testing.assert_array_equal(p["prediction"], q["prediction"])
        np.testing.assert_array_equal(p["train_prediction"], q["train_prediction"])


def test_crop_respects_event_anchor_and_erp_removes_only_available_prefix():
    native = dict(channel_names=["C3", "C4"], eeg_geometry=[{}, {}], window_start_s=-5., duration_s=30.)
    eeg = np.full((1, 2, 6000), 3., dtype=np.float32)
    eeg[..., 1000:1200] = 3 + np.repeat(np.arange(10), 20)
    a = dict(eeg=eeg, fnirs=np.ones((1, 2, 300)), eeg_mask=np.ones_like(eeg, bool), fnirs_mask=np.ones((1, 2, 300), bool),
             target=np.array([0]), mask=np.array([True]), sample_id=np.array(["one"]), subject_id=np.array(["s"]), record_id=np.array(["r"]))
    scaler = {m: dict(mean=[[0.], [0.]], scale=[[1.], [1.]]) for m in ("eeg", "fnirs")}
    full = audit.ArrayDataset(a, native, native, [0])
    np.testing.assert_allclose(audit.event_voltage_features(full, scaler)[0], np.tile(np.arange(10), 2))
    event = audit.ArrayDataset(a, native, {**native, "window_start_s": 0., "duration_s": 8.}, [0])
    np.testing.assert_array_equal(event[0]["eeg"], eeg[0, :, 1000:2600])
    assert event[0]["fnirs"].shape == (2, 80)
    with pytest.raises(ValueError, match="leaves"):
        audit.ArrayDataset(a, native, {**native, "window_start_s": -6.}, [0])


def test_regression_metrics_preserve_thirty_second_mask_and_record_weighting():
    target = np.stack([np.tile(np.arange(30), (2, 1)), np.tile(np.arange(30), (2, 1))]).astype(float)
    prediction = target + 1
    mask = np.ones_like(target, bool); mask[1, :, 5:] = False
    target[~mask] = np.nan; prediction[~mask] = -99999
    result = score_predictions(target, prediction, mask, ["s", "s"], "refed_regression", 0, ["long", "partial"])
    assert result["coverage_complete"]
    assert result["mae"] == 1 and result["rmse"] == 1
    assert result["subjects"][0]["record_count"] == 2


def test_training_row_coordinates_preserve_l2_linear_fit_without_validation_fit():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(44)
    x = rng.normal(size=(20, 4200)); z = rng.normal(size=(7, 4200)); y = np.arange(20) % 2
    a, b, state = audit.linear_row_coordinates(x, z)
    assert state["kind"] == "training_row_space_equivalent_L2"
    np.testing.assert_allclose(a @ a.T, x @ x.T, rtol=1e-10, atol=1e-9)
    m = LogisticRegression(C=.01, tol=1e-9, max_iter=1000).fit(x, y)
    n = LogisticRegression(C=.01, tol=1e-9, max_iter=1000).fit(a, y)
    np.testing.assert_allclose(n.predict_proba(b), m.predict_proba(z), rtol=1e-6, atol=1e-6)
    a2, _, _ = audit.linear_row_coordinates(x, z * 100)
    np.testing.assert_array_equal(a, a2)


def test_independent_metric_check_uses_subject_weights_and_fixed_missing_classes():
    y = np.r_[np.tile([0, 1], 50), 0]
    p = y.copy(); p[-1] = 1
    subjects = np.array(["large"] * 100 + ["small"])
    mask = np.ones(len(y), bool)
    assert audit.independent_primary(y, p, mask, subjects, subjects, "motor_imagery") == .5
    with pytest.raises(ValueError, match="undefined"):
        audit.independent_primary(np.zeros((1, 2, 30)), np.zeros((1, 2, 30)), np.ones((1, 2, 30), bool),
                                  np.array(["one"]), np.array(["record"]), "refed_regression")


def test_v2_accuracy_and_subject_sd_do_not_weight_long_subjects_more():
    y = np.r_[np.tile([0, 1], 50), 0]
    p = y.copy(); p[-1] = 1
    subjects = np.array(["large"] * 100 + ["small"])
    result = score_predictions(y, p, np.ones(101, bool), subjects,
                               "motor_imagery", 2, metric_version="v2")
    assert result["subject_equal_accuracy"] == .5
    assert result["primary"] == .5
    assert result["pooled_accuracy"] == pytest.approx(100 / 101)
    assert result["accuracy_subject_sd"] == pytest.approx(2 ** -.5)
    assert result["subject_sd"] == pytest.approx(2 ** -.5)


def test_v2_constant_target_is_common_ccc_scope_but_remains_in_error():
    target = np.stack([np.tile(np.arange(30), (2, 1)), np.ones((2, 30))]).astype(float)
    mask = np.ones_like(target, bool)
    old = score_predictions(target, target.copy(), mask, ["s", "s"],
                            "refed_regression", 0, ["dynamic", "constant"])
    assert not old["coverage_complete"]  # Retained v1 endpoint is unchanged.
    result = score_predictions(target, target.copy(), mask, ["s", "s"],
                               "refed_regression", 0, ["dynamic", "constant"], metric_version="v2")
    assert result["coverage_complete"] and result["primary"] == 1
    assert result["constant_target_record_counts"] == [1, 1]
    changed = target.copy(); changed[1] += 4
    result = score_predictions(target, changed, mask, ["s", "s"],
                               "refed_regression", 0, ["dynamic", "constant"], metric_version="v2")
    assert result["primary"] == 1 and result["mae"] == 2
    assert result["mae_by_dimension"] == [2, 2]
    assert result["subjects"][0]["record_count"] == 2


def test_v2_refuses_missing_scored_values_and_empty_classification_subject():
    with pytest.raises(ValueError, match="nonfinite"):
        score_predictions(np.array([0, 1]), np.array([0, np.nan]), np.ones(2, bool),
                          ["a", "b"], "motor_imagery", 2, metric_version="v2")
    with pytest.raises(ValueError, match="every subject"):
        score_predictions(np.array([0, 1]), np.array([0, 1]), np.array([True, False]),
                          ["a", "b"], "motor_imagery", 2, metric_version="v2")


def test_repair_features_keep_time_and_masked_values_out_of_statistics():
    from comparative_methods.rebench_baselines_v1 import repair_probe_features
    rng = np.random.default_rng(31)
    eeg = rng.normal(size=(2, 3, 6000)); hb = rng.normal(size=(2, 4, 300))
    em, hm = np.ones_like(eeg, bool), np.ones_like(hb, bool)
    em[1, 1, 2000:2200] = False; hm[1, 2, 120:130] = False
    first = repair_probe_features(eeg, hb, em, hm, task="refed_regression")
    eeg[~em] = np.nan; hb[~hm] = 1e100
    second = repair_probe_features(eeg, hb, em, hm, task="refed_regression")
    for key in first:
        np.testing.assert_array_equal(first[key][0], second[key][0])
        assert first[key][0].shape[1] == 30 and np.isfinite(first[key][0]).all()
    assert not first["eeg"][1][1, 10, 5:10].any()
    with pytest.raises(ValueError, match="complete EEG"):
        repair_probe_features(eeg, hb, em, hm, task="visual")


def test_visual_covariance_is_scale_invariant_and_erp_keeps_voltage():
    from comparative_methods.rebench_baselines_v1 import repair_probe_features
    rng = np.random.default_rng(38); eeg = rng.normal(size=(2, 3, 6000))
    hb = np.ones((2, 4, 300)); em, hm = np.ones_like(eeg, bool), np.ones_like(hb, bool)
    x = repair_probe_features(eeg, hb, em, hm, task="visual")
    y = repair_probe_features(eeg * 3, hb, em, hm, task="visual")
    np.testing.assert_allclose(x["covariance"][0], y["covariance"][0], atol=1e-6)
    np.testing.assert_allclose(x["event"][0] * 3, y["event"][0], atol=1e-6)


@pytest.mark.parametrize("task", ["visual", "refed_regression"])
def test_reference_probe_uses_only_window_signals_and_removes_common_reference(task):
    from comparative_methods.rebench_baselines_v1 import repair_probe_features
    rng = np.random.default_rng(58)
    eeg = rng.normal(size=(2, 3, 6000)); hb = rng.normal(size=(2, 4, 300))
    em, hm = np.ones_like(eeg, bool), np.ones_like(hb, bool)
    x = repair_probe_features(eeg, hb, em, hm, task=task, version=2)
    y = repair_probe_features(3 * eeg + rng.normal(size=(2, 1, 6000)), 5 * hb, em, hm, task=task, version=2)
    names = ("normalized_event", "normalized_encoding", "baseline_covariance", "relative_band") if task == "visual" else ("relative", "relative_trajectory")
    for name in names:
        np.testing.assert_allclose(x[name][0], y[name][0], atol=2e-6)
    if task == "refed_regression":
        assert x["relative"][0].shape == (2, 30, 23)
        assert x["relative_trajectory"][0].shape == (2, 230)
        hm[1, :, 120:] = False; em[1, :, 2400:] = False
        original = repair_probe_features(eeg, hb, em, hm, task=task, version=2)
        eeg[~em] = np.nan; hb[~hm] = 1e80
        changed = repair_probe_features(eeg, hb, em, hm, task=task, version=2)
        for name in original:
            np.testing.assert_array_equal(original[name][0], changed[name][0])


def test_probe_scaler_uses_only_training_and_keeps_unseen_features_zero():
    x = np.array([[1., 5., 100.], [3., 7., 0.], [50., 60., 70.]])
    m = np.array([[True, True, False], [True, True, False], [True, True, True]])
    first, a = audit.probe_transform(x, m, [0, 1], [2, 1])
    x[2] = 1e6
    _, b = audit.probe_transform(x, m, [0, 1], [2, 1])
    np.testing.assert_array_equal(a["mean"], b["mean"])
    np.testing.assert_array_equal(a["scale"], b["scale"])
    assert np.all(first[:, 2] == 0)


def test_probe_rejects_test_partition_before_target_or_signal_open(tmp_path):
    import json
    cfg = audit.read_config(Path(__file__).parents[1] / "comparative_methods/downstream_repair_probe_v1.yaml")
    cfg["parent_run"] = "parent"; parent = tmp_path / "parent"; parent.mkdir()
    (parent / "plan.json").write_text(json.dumps(dict(schema=audit.SCHEMA,
        scope="public_train_validation_diagnostic_not_final_paper_ranking",
        tasks=dict(visual=dict(counts=dict(train=1, validation=1))))))
    (parent / "summary.json").write_text('{"status":"completed"}')
    folder = parent / "visual/prepare"; folder.mkdir(parents=True)
    for key, values in dict(partition=["train", "test"], sample_id=["a", "b"], subject_id=["a", "b"], record_id=["r", "r"]).items():
        np.save(folder / (key + ".npy"), np.array(values))
    # No target or signal files exist: the boundary must fail before opening them.
    with pytest.raises(ValueError, match="original development support"):
        audit.probe_inputs(cfg, tmp_path, "visual", signals=True)
    with pytest.raises(ValueError, match="original development support"):
        audit.probe_native_signal_extract(cfg, tmp_path, "visual", tmp_path / "output")


@pytest.mark.parametrize("task", ["visual", "refed_regression"])
def test_hb_native_probe_preserves_pairs_and_masks(task):
    from comparative_methods.rebench_baselines_v1 import repair_hb_probe_features
    values = np.broadcast_to(np.arange(300)[None, None, :] / 10., (2, 4, 300)).copy()
    mask = np.ones_like(values, bool); mask[1, 0, 200:] = False
    x, m, _ = repair_hb_probe_features(values, mask, task=task)
    values[~mask] = np.nan
    y, n, _ = repair_hb_probe_features(values, mask, task=task)
    np.testing.assert_array_equal(x, y); np.testing.assert_array_equal(m, n)
    assert np.isfinite(y).all()
    if task == "refed_regression":
        # HbT is the sum of two matching chromophores, not the channel mean.
        assert x.shape == (2, 30, 12) and x[0, 0, 4] == pytest.approx(2.)
        assert not m[1, 20:, :6].any()


def test_native_hb_filter_audit_retains_clock_and_separates_slow_content(monkeypatch):
    from types import SimpleNamespace
    from src.data import unified_physiology as owner
    time = np.arange(3000) / 10.
    signal = 4 + np.sin(2 * np.pi * .003 * time) + np.sin(2 * np.pi * .1 * time)
    values = np.stack((signal, signal * .5), -1)[:, None, :]
    monkeypatch.setattr(owner, "load_native_fnirs_record", lambda *args:dict(values=values, sample_rate_hz=10., provenance={}))
    windows, provenance = audit.probe_native_hb_record((Path('.'), SimpleNamespace(join_key='fixture', dataset_id='refed'), [(0, 90., [0, 1])]))
    _, columns, views, mask = windows[0]
    assert columns == [0, 1] and mask.all()
    assert all(value.shape == (2, 300) for value in views.values())
    assert views['lowpass'][0].mean() - views['bandpass'][0].mean() > 3
    assert provenance['windows'] == 1


def test_native_visual_replay_uses_published_clock_before_filtering(monkeypatch):
    from types import SimpleNamespace
    from src.data import unified_physiology as owner
    from src.data.homer2_preprocessing import bandpass_fnirs
    t = np.arange(3000) / 10.001
    values = np.stack((np.sin(2 * np.pi * .12 * t), np.cos(2 * np.pi * .08 * t)), -1)
    monkeypatch.setattr(owner, "load_native_fnirs_record", lambda *args:dict(values=values[:, None], time_s=t, sample_rate_hz=10., provenance={}))
    record = SimpleNamespace(join_key='fixture', dataset_id='visual_cognitive_motivation')
    windows, _ = audit.probe_native_hb_record((Path('.'), record, [(0, 90., [0, 1])]))
    grid = np.arange(int(np.floor(t[-1] * 10)) + 1) / 10
    aligned = np.column_stack([np.interp(grid, t, channel) for channel in values.T])
    expected, _ = bandpass_fnirs(aligned, sample_rate_hz=10., low_hz=.01, high_hz=.2)
    np.testing.assert_allclose(windows[0][2]['replay'], expected[900:1200].T, rtol=1e-11, atol=1e-11)


def test_record_ccc_loss_keeps_videos_separate_and_excludes_only_constant_coordinates():
    import torch
    target = torch.tensor([[[.1, .1], [.2, .2]], [[.3, .3], [.4, .4]],
                           [[.7, .7], [.8, .8]], [[.9, .9], [1., 1.]]])
    valid = torch.ones_like(target, dtype=torch.bool); groups = [np.array([0, 1]), np.array([2, 3])]
    prediction = torch.cat([target[:2].flip((0, 1)), target[2:].flip((0, 1))]).requires_grad_()
    loss = audit.probe_record_ccc_loss(prediction, target, valid, groups, np.ones((2, 2)))
    # Across-record offsets would disguise the reversed dynamics if pooled.
    torch.testing.assert_close(loss, torch.tensor(2.))
    perfect = target.clone().requires_grad_()
    torch.testing.assert_close(audit.probe_record_ccc_loss(perfect, target, valid, groups, np.ones((2, 2))), torch.tensor(0.))
    target[:2, :, 1] = .2
    mass = np.array([[1., 0.], [1., 1.]])
    p = (target * .8 + .1).requires_grad_()
    valid[3, 1] = False
    baseline = audit.probe_record_ccc_loss(p, target, valid, groups, mass)
    changed = target.clone(); changed[3, 1] = float('nan')
    torch.testing.assert_close(audit.probe_record_ccc_loss(p, changed, valid, groups, mass), baseline)
    baseline.backward(); assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert not p.grad[~valid].any() and not p.grad[:2, :, 1].any()
    with pytest.raises(ValueError, match='nonconstant'):
        audit.probe_record_ccc_loss(p, target, valid, groups, np.ones((2, 2)))


@pytest.mark.parametrize('record_weight', [None, 0., .05])
def test_neural_temporal_probe_never_uses_held_fold_targets(tmp_path, record_weight):
    import torch
    torch.set_num_threads(2)
    cfg = audit.read_config(Path(__file__).parents[1] / "comparative_methods/downstream_repair_probe_v1.yaml")
    cfg["fit"].update(neural_epochs=2, neural_width=8, neural_batch_size=2)
    rng = np.random.default_rng(9); x = rng.normal(size=(6, 30, 4))
    a = dict(target=128 + np.transpose(x[..., :2], (0, 2, 1)), mask=np.ones((6, 2, 30), bool),
             subject_id=np.repeat(['a', 'b', 'c'], 2), record_id=np.array(['video'] * 6))
    spec = {} if record_weight is None else dict(record_ccc_weight=record_weight, record_batch_size=2)
    weights = np.ones((4, 30, 2)); first = tmp_path / "first"; second = tmp_path / "second"
    first.mkdir(); second.mkdir()
    p, fitted = audit.probe_neural(x, a, np.arange(4), weights, "refed_regression", spec, cfg, "cpu", first)
    a["target"][4:] = np.nan
    q, _ = audit.probe_neural(x, a, np.arange(4), weights, "refed_regression", spec, cfg, "cpu", second)
    np.testing.assert_array_equal(p, q)
    assert p.shape == (6, 2, 30) and np.isfinite(p).all()
    if record_weight is not None:
        assert fitted['training_records'] == 2
        with pytest.raises(ValueError, match='complete training records'):
            audit.probe_neural(x, a, np.array([0, 2, 3]), weights[:3], 'refed_regression', spec, cfg, 'cpu', first)


@pytest.mark.parametrize("scaled", [False, True])
def test_event_encoder_ft_never_uses_held_fold_targets(tmp_path, monkeypatch, scaled):
    import torch
    from torch import nn
    from types import SimpleNamespace
    from comparative_methods.CBraMod import rebench_v1 as module
    class Fixture(nn.Module):
        def __init__(self, task, profile, device):
            super().__init__(); self.base = nn.Module(); self.base.encoder = nn.Linear(16, 12)
            self.head = nn.Linear(12, 8); self.metadata = dict(scope="synthetic")
            self.register_buffer('feature_mean', torch.zeros(12)); self.register_buffer('feature_scale', torch.ones(12))
        def event_features(self, x, **kwargs):
            return self.base.encoder(x.mean(-1))
        def forward(self, x, **kwargs):
            return self.head((self.event_features(x) - self.feature_mean) / self.feature_scale)
    monkeypatch.setattr(module, "CBraModEventProbe", Fixture)
    torch.set_num_threads(2)
    cfg = audit.read_config(Path(__file__).parents[1] / "comparative_methods/downstream_repair_probe_v5.yaml")
    spec = dict(cfg["models"]["visual"]["event_ft"], epochs=2, batch_size=2)
    rng = np.random.default_rng(2); names = list(audit.EEG_ONLY_CHANNELS['visual_cognitive_motivation'])
    a = dict(eeg=rng.normal(size=(6, 16, 6000)), eeg_mask=np.ones((6, 16, 6000), bool), target=np.array([0, 1, 2, 3, 0, 1]))
    native = dict(channel_names=names, eeg_geometry=[{} for _ in names], window_start_s=-5., duration_s=30., input_samples=6000, num_classes=4)
    if scaled:
        cfg['retained_feature_run'] = 'retained'; spec['feature_scaling'] = 'training_latent_columns'
        spec['encoder_training_mode'] = 'deterministic'
        a['sample_id'] = np.array([f'sample_{i}' for i in range(6)]); a['subject_id'] = np.array(['training'] * 4 + ['held'] * 2)
        folder = tmp_path / 'retained/visual/encoder'; folder.mkdir(parents=True)
        audit.write(tmp_path / 'retained/plan.json',dict(tasks=dict(visual=dict(sample_ids=a['sample_id'].tolist()))))
        audit.write(folder / 'features.json', dict(input_profile=dict(channel_names=names)))
        torch.manual_seed(cfg['model_seed']); frozen = Fixture('visual',native,'cpu')
        with torch.no_grad():
            features = frozen.event_features(torch.as_tensor(a['eeg'],dtype=torch.float32)).numpy()
        np.save(folder / 'encoder_event.npy',features)
    first = tmp_path / "first"; second = tmp_path / "second"; first.mkdir(); second.mkdir()
    p, fitted = audit.probe_cbramod_ft(a, native, np.arange(4), np.ones(4), spec, cfg, tmp_path, "cpu", first)
    a['target'][4:] = -1000
    if scaled:
        features[4:] = 1e6; np.save(folder / 'encoder_event.npy',features)
    q, _ = audit.probe_cbramod_ft(a, native, np.arange(4), np.ones(4), spec, cfg, tmp_path, "cpu", second)
    np.testing.assert_array_equal(p, q)
    assert fitted['encoder_parameter_change_l2'] > 0


@pytest.mark.parametrize("task", ["visual", "refed_regression"])
def test_record_control_holds_out_entire_records_and_reports_unavailable_subjects(tmp_path, monkeypatch, task):
    cfg = audit.read_config(Path(__file__).parents[1] / "comparative_methods/downstream_repair_probe_v2.yaml")
    subjects = np.array(["complete"] * 12 + ["one_record"] * 4)
    records = np.repeat(["r0", "r1", "r2", "r0"], 4)
    if task == "visual":
        target = np.tile(np.arange(4), 4); features = np.eye(4)[target]
    else:
        features = np.broadcast_to(np.stack([np.sin(np.arange(30)), np.cos(np.arange(30))], -1), (16, 30, 2)).copy()
        target = (128 + 20 * features).transpose(0, 2, 1)
    arrays = dict(target=target, mask=np.ones_like(target, bool), subject_id=subjects, record_id=records,
                  sample_id=np.array([f"sample_{i}" for i in range(16)]))
    monkeypatch.setattr(audit, "probe_inputs", lambda *args, **kwargs: (arrays, {}))
    name = cfg["record_controls"][task]["features"]; destination = tmp_path / task / "features"; destination.mkdir(parents=True)
    np.save(destination / (name + ".npy"), features); np.save(destination / (name + "_mask.npy"), np.ones_like(features, bool))
    audit.write(destination / "features.json", dict(features={name:dict(shape=list(features.shape), blocks=[features.shape[-1]])}))
    output = tmp_path / "control"; output.mkdir()
    result = audit.probe_record_control(cfg, tmp_path, tmp_path, task, output)
    assert result["planned_samples"] == 16 and result["evaluated_samples"] == 12
    assert result["excluded"][0]["subject"] == "one_record"
    assert all(not set(row["train_records"]) & set(row["evaluation_records"]) for row in result["fits"])
    assert result["metric"]["subject_count"] == 1


@pytest.mark.parametrize("task,estimator", [("visual", "logistic"), ("visual", "attribute_logistic"),
    ("refed_regression", "ridge"), ("refed_regression", "trajectory_rbf")])
def test_repair_fold_fit_and_complete_aggregation_preserve_held_subjects(tmp_path, task, estimator):
    import json
    cfg = audit.read_config(Path(__file__).parents[1] / "comparative_methods/downstream_repair_probe_v1.yaml")
    cfg["parent_run"] = "parent"; cfg["tasks"] = [task]
    n = 12; subjects = np.repeat(["s0", "s1", "s2"], 4)
    part = np.array(["train"] * 8 + ["validation"] * 4)
    if task == "visual":
        target = np.tile(np.arange(4), 3); features = np.eye(4)[target]
        spec = dict(features="event", estimator=estimator, C=1.)
    else:
        values = np.stack((np.sin(np.arange(30) / 4), np.cos(np.arange(30) / 5)), -1)
        features = np.broadcast_to(values[None], (n, 30, 2)).copy()
        target = (128 + 20 * features).transpose(0, 2, 1)
        spec = dict(features="eeg", estimator=estimator, alpha=.001, gamma=.5, landmarks=4)
        if estimator == "trajectory_rbf":
            features = features.reshape(n, -1)
    cfg["models"] = {task: {"fixed": spec}}
    parent = tmp_path / "parent"; folder = parent / task / "prepare"; folder.mkdir(parents=True)
    ids = np.array([f"sample{i}" for i in range(n)])
    arrays = dict(target=target, mask=np.ones_like(target, bool), sample_id=ids,
                  subject_id=subjects, partition=part, record_id=np.repeat("record", n))
    for name, array in arrays.items():
        np.save(folder / (name + ".npy"), array)
    audit.write(parent / "summary.json", dict(status="completed"))
    audit.write(parent / "plan.json", dict(schema=audit.SCHEMA, scope="public_train_validation_diagnostic_not_final_paper_ranking",
        tasks={task: dict(counts=dict(train=8, validation=4), subjects=dict(train=["s0", "s1"], validation=["s2"]), profile={})}))
    run = tmp_path / "run"; destination = run / task / "features"; destination.mkdir(parents=True)
    np.save(destination / (spec["features"] + ".npy"), features)
    np.save(destination / (spec["features"] + "_mask.npy"), np.ones_like(features, bool))
    audit.write(destination / "features.json", dict(features={spec["features"]:dict(shape=list(features.shape), blocks=[features.shape[-1]])}))
    audit.write(run / "plan.json", dict(tasks={task:dict(sample_ids=ids.tolist(), subject_folds=dict(s0=0,s1=1,s2=2))}))
    for fold in range(3):
        output = run / task / "fixed" / f"fold_{fold}"; output.mkdir(parents=True)
        value = audit.probe_fit(cfg, tmp_path, run, task, "fixed", fold, output, "cpu")
        assert value["evaluation_subjects"] == [f"s{fold}"]
        assert value["evaluation"]["primary"] > .99
        audit.write(output / "result.json", dict(status="completed", result=value))
    summary = audit.probe_aggregate(cfg, tmp_path, run)
    assert summary["status"] == "completed" and summary["completed_cells"] == 1
    value = json.loads((run / "summary.json").read_text())["cases"][0]
    assert value["metric"]["sample_count"] == n and value["metric"]["subject_count"] == 3
    assert audit.probe_verify(cfg, tmp_path, run)["status"] == "passed"
    cfg["models"][task]["not_run"] = spec
    assert audit.probe_aggregate(cfg, tmp_path, run)["status"] == "incomplete_or_failed"
    verification = audit.probe_verify(cfg, tmp_path, run)
    assert verification["status"] == "passed_completed_outputs" and verification["incomplete_or_failed_cells"] == 1
    cfg["models"][task].pop("not_run")
    with np.load(run / task / "fixed/fold_2/predictions.npz") as saved:
        first = saved["prediction"].copy()
    target = target.copy(); target[8:] = (3 - target[8:]) if task == "visual" else 1.
    np.save(folder / "target.npy", target)
    output = run / task / "changed"; output.mkdir()
    audit.probe_fit(cfg, tmp_path, run, task, "fixed", 2, output, "cpu")
    with np.load(output / "predictions.npz") as saved:
        np.testing.assert_allclose(first, saved["prediction"])
    if task == 'refed_regression' and estimator == 'ridge':
        retained = tmp_path / 'retained'
        audit.write(retained / 'plan.json', dict(parent_run=cfg['parent_run'], tasks={task:dict(sample_ids=ids.tolist())}))
        directory = retained / task / 'encoder/fold_0'; directory.mkdir(parents=True)
        np.save(directory / 'efrm_sequence.npy', features); np.save(directory / 'efrm_sequence_mask.npy', np.ones_like(features, bool))
        descriptor = dict(training_subjects=['s1', 's2'], features=dict(efrm_sequence=dict(shape=list(features.shape), blocks=[2])))
        audit.write(directory / 'features.json', descriptor)
        cfg['retained_feature_run'] = 'retained'; cfg['models'][task]['fixed']['features'] = 'efrm_sequence'
        reused = run / task / 'reused'; reused.mkdir()
        audit.probe_fit(cfg, tmp_path, run, task, 'fixed', 0, reused, 'cpu')
        descriptor['training_subjects'].append('s0'); audit.write(directory / 'features.json', descriptor)
        with pytest.raises(ValueError, match='crosses held fold'):
            audit.probe_fit(cfg, tmp_path, run, task, 'fixed', 0, reused, 'cpu')


@pytest.mark.parametrize("classes,weight", [(2, None), (2, "balanced"), (3, None), (3, "balanced")])
def test_converged_logistic_satisfies_independent_fixed_objective_gradient(classes, weight):
    from sklearn.utils.class_weight import compute_sample_weight
    cfg = audit.read_config(Path(__file__).parents[1] / "comparative_methods/downstream_convergence_probe_v1.yaml")
    rng = np.random.default_rng(93); x = rng.normal(size=(120, 6)); z = rng.normal(size=(20, 6))
    y = (x @ rng.normal(size=(6, classes)) + rng.normal(size=(120, classes)) * .4).argmax(-1)
    model, checks = audit.fit_converged_logistic(x, y, z, dict(C=3., class_weight=weight), cfg)
    assert checks['numerically_closed'] and max(checks['gradients_inf']) < 1e-7
    weights = compute_sample_weight(weight, y)
    optimum, _ = audit.logistic_objective_gradient(model, x, y, weights)
    model.coef_[0, 0] += .1
    changed, gradient = audit.logistic_objective_gradient(model, x, y, weights)
    assert changed > optimum and gradient > 1e-4


def test_stimulus_student_inference_cannot_read_held_video_time_or_target():
    rng = np.random.default_rng(10); x = rng.normal(size=(12, 4)); train = np.arange(8)
    a = dict(target=128 + rng.normal(size=(12, 2, 30)), mask=np.ones((12, 2, 30), bool),
             record_id=np.array(['v1','v2'] * 6), sample_id=np.array([f'r|start={30*(i%2)}|duration=30' for i in range(12)]))
    cfg = dict(model_seed=17,fit=dict(logistic_max_iter=1000,logistic_tol=1e-8))
    spec = dict(gamma=.5,landmarks=4,C=1.); weights = np.ones((8,30,2))
    first, _ = audit.probe_stimulus_student(x,a,train,weights,spec,cfg)
    a['target'][8:] = 1e6; a['record_id'][8:] = 'xx'; a['sample_id'][8:] = 'invalid_no_time_field'
    second, _ = audit.probe_stimulus_student(x,a,train,weights,spec,cfg)
    np.testing.assert_array_equal(first,second)


def test_cpu_probe_plan_does_not_reserve_an_unlisted_gpu(tmp_path, monkeypatch):
    import json
    cfg = audit.read_config(Path(__file__).parents[1] / 'comparative_methods/downstream_repair_probe_v8.yaml')
    a = dict(target=np.zeros((3,2,30)),subject_id=np.array(['a','b','c']),sample_id=np.array(['a','b','c']))
    monkeypatch.setattr(audit,'probe_inputs',lambda *args,**kwargs:(a,{}))
    audit.probe_plan(cfg,tmp_path,tmp_path/'run',tmp_path/'source','downstream_repair_probe_v8.yaml')
    manifest = json.loads((tmp_path/'run/jobs.json').read_text())
    assert manifest['limits']['gpu_ids'] == [] and manifest['limits']['gpu_reserve_mib'] == {}
    assert all(not job['resources']['gpu'] for job in manifest['jobs'])


def test_individual_input_calibration_is_person_local_masked_and_preserves_variation():
    x = np.array([[1.,7.],[3.,9.],[10.,30.],[30.,50.]])
    mask = np.ones_like(x,bool);mask[2,1]=False;subjects=np.array(['a','a','b','b'])
    first, state = audit.probe_subject_calibration(x,mask,subjects)
    x[2:,0] = [100,500];x[2,1]=np.nan
    second, changed = audit.probe_subject_calibration(x,mask,subjects)
    np.testing.assert_array_equal(first[:2],second[:2])
    np.testing.assert_array_equal(state['mean'][0],changed['mean'][0])
    np.testing.assert_allclose(first[:,0],[-1,1,-1,1])
    assert first[2,1] == second[2,1] == 0 and np.isfinite(second).all()


def test_native_voltage_features_preserve_sign_slope_and_mask():
    from comparative_methods.rebench_baselines_v1 import repair_voltage_probe_features
    time=np.arange(6000)/200.;values=np.broadcast_to((2*time)[None,None],(2,2,6000)).copy()
    mask=np.ones_like(values,bool);mask[1,1,1000:1200]=False
    first,support,_=repair_voltage_probe_features(values,mask,task='refed_regression')
    values[~mask]=np.nan;second,_,_=repair_voltage_probe_features(values,mask,task='refed_regression')
    np.testing.assert_array_equal(first,second)
    assert first.shape==(2,30,4) and first[0,0,2]==pytest.approx(2.)
    assert not support[1,5,[1,3]].any()


def test_native_eeg_filter_replay_and_slow_voltage_support(monkeypatch):
    from types import SimpleNamespace
    from src.data import unified_physiology as owner
    time=np.arange(40000)/200.;values=np.stack((np.sin(2*np.pi*.2*time),np.cos(2*np.pi*.2*time)),-1)
    native=owner.NativeEEGRecord(values=values,sample_rate_hz=200.,channel_names=('C3','C4'),native_unit='uV',
        source_path=Path('/synthetic/non-authorizing-eeg'),unit_evidence='synthetic fixture')
    monkeypatch.setattr(owner,'load_native_eeg_record',lambda *args:native)
    windows,metadata=audit.probe_native_eeg_record((Path('.'),SimpleNamespace(join_key='fixture'),[(0,60.,[0,1])],['C3','C4']))
    expected,_,_=owner.preprocess_eeg_record_with_quality(native,output_coordinate='measurement')
    np.testing.assert_allclose(windows[0][2]['replay'],expected[12000:18000].T)
    low=windows[0][2]['hp_0p1'];baseline=windows[0][2]['replay']
    assert np.sqrt(np.mean(low**2))>20*np.sqrt(np.mean(baseline**2)) and windows[0][3].all()
