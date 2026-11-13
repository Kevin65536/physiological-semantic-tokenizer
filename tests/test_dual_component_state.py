"""Training-only state-pilot and 200 Hz contracts using disposable synthetic data."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dual_component_tokenizer import (
    state_eeg_200hz, state_pilot_plan, state_pilot_window, state_pilot_coordinates,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('state_pilot_runner', ROOT / 'experiments/scripts/run_dual_component_tokenizer.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
CONFIG = ROOT / 'experiments/configs/physiology_semantic_tokenizer/dual_component_state_v1.yaml'
REFINEMENT = ROOT / 'experiments/configs/physiology_semantic_tokenizer/dual_component_state_refinement_v1.yaml'


def settings():
    cfg = deepcopy(runner.read_config(CONFIG))
    cfg['model'].update(width=16, latent_dim=8, heads=2, encoder_layers=1, decoder_layers=1,
                        codebook_size=8, kmeans_iters=2)
    cfg['training'].update(warmup_steps=2, continuation_steps=3, batch_size=4, initialization_per_dataset=2,
                           checkpoint_every=1)
    return cfg


def synthetic_values(rate, seconds=90):
    t = np.arange(round(rate * seconds)) / rate
    x = np.sin(2 * np.pi * 25 * t)[:, None, None] * np.array([1., 2.])[None, :, None]
    return dict(x=x, s=x * .7, o=x * .3, observation_valid=np.ones_like(x, bool), teacher_valid=np.ones_like(x, bool))


def fixture_cache(tmp_path, cfg):
    root = tmp_path / 'source'; source = tmp_path / 'ssm'
    manifest = dict(schema='physiology_so_supervision_v2', status='completed', source_ssm=str(source), windows=[])
    for dataset in cfg['data']['datasets']:
        rate = cfg['data']['native_rate_hz'][dataset]
        for duration, record in [(90, 'record_a'), *(([(60, 'record_b')]) if dataset == 'refed' else [])]:
            key = dataset + '/subject/' + record + '/window'
            path = root / 'windows' / (key + '.json'); path.parent.mkdir(parents=True, exist_ok=True)
            src = dict(key=key, dataset=dataset, subject='subject', split='train', rate=rate,
                       native_record=record, start_sample=0, samples=rate * duration, duration_s=duration)
            meta = dict(schema=manifest['schema'], **{k: src[k] for k in ('key', 'dataset', 'subject', 'split')},
                        source=src, status='completed', EEG_channels=['a', 'b'], fNIRS_positions=['p', 'failed'],
                        array_file=str(path.with_suffix('.npz')))
            arrays = {}
            for modality, hz in [('eeg', rate), ('fnirs', 10)]:
                v = synthetic_values(hz, duration)
                for k, a in v.items():
                    arrays[modality + '_' + k] = a[..., 0] if modality == 'eeg' else np.repeat(a, 2, axis=-1)
                arrays[modality + '_time_s'] = np.arange(duration * hz) / hz
            # Valid X at a failed teacher position must survive with a missing supervision mask.
            arrays['fnirs_teacher_valid'][:, 1] = False
            arrays['fnirs_s'][:, 1] = np.nan; arrays['fnirs_o'][:, 1] = np.nan
            arrays['source_scoring_mask'] = np.r_[np.ones(duration * 10 - 1, bool), False]
            arrays['eeg_source_scoring_mask'] = np.repeat(arrays['source_scoring_mask'], rate // 10)
            np.savez(path.with_suffix('.npz'), **arrays)
            path.write_text(json.dumps(meta))
            manifest['windows'].append(dict(key=key, dataset=dataset, subject='subject', split='train', metadata=str(path)))
    manifest['windows'].append(dict(key='test', dataset='refed', subject='hidden', split='test', metadata='/must/not/open.json'))
    runner.write_json(root / 'cache_manifest.json', manifest)
    runner.write_json(root / 'verification.json', dict(status='passed'))
    return root, source


def test_reviewed_200hz_contract_and_disallowed_stage(tmp_path):
    cfg = runner.read_config(CONFIG)
    assert cfg['data']['eeg_rate_hz'] == 200 and cfg['data']['allowed_splits'] == ['train']
    with pytest.raises(ValueError, match='other stages'):
        runner.main(['--config', str(CONFIG), '--stage', 'all', '--output', str(tmp_path)])
    bad = deepcopy(cfg['data']); bad['allowed_splits'].append('test')
    with pytest.raises(ValueError, match='before reads'):
        state_pilot_plan(tmp_path / 'missing', bad, tmp_path / 'ssm')


@pytest.mark.parametrize('rate', [200, 500, 1000])
def test_200hz_identity_antialias_closure_and_block_isolation(rate):
    values = synthetic_values(rate)
    score = np.ones(90 * rate, bool)
    transformed, scored, meta = state_eeg_200hz(values, score, rate, 60 * rate)
    assert transformed['x'].shape == (18000, 2, 1)
    np.testing.assert_allclose(transformed['x'], transformed['s'] + transformed['o'], atol=1e-12)
    if rate == 200:
        assert transformed is values and scored is score
    else:
        original = np.sin(2 * np.pi * 25 * np.arange(18000) / 200)
        np.testing.assert_allclose(transformed['x'][100:11900, 0, 0], original[100:11900], atol=.003)
        assert not scored[0] and not scored[12000] and scored[12100]
        changed = {k: v.copy() for k, v in values.items()}
        for key, factor in [('x', 1.), ('s', .7), ('o', .3)]:
            changed[key][60 * rate:] += 100 * factor
        altered, _, _ = state_eeg_200hz(changed, score, rate, 60 * rate)
        np.testing.assert_array_equal(altered['x'][:12000], transformed['x'][:12000])
        high = synthetic_values(rate)
        high['x'] = np.sin(2 * np.pi * 150 * np.arange(90 * rate) / rate)[:, None, None]
        high['s'] = high['x'] * .7; high['o'] = high['x'] * .3
        high['observation_valid'] = high['teacher_valid'] = np.ones_like(high['x'], bool)
        down, _, _ = state_eeg_200hz(high, score, rate, 60 * rate)
        assert np.sqrt(np.mean(down['x'][100:11900] ** 2)) < .01
        assert meta['target_hz'] == 200


def test_mask_filter_support_is_not_promoted_to_valid_target():
    values = synthetic_values(1000)
    values['teacher_valid'][10000:10100, 0] = False
    values['s'][10000:10100, 0] = np.nan; values['o'][10000:10100, 0] = np.nan
    transformed, _, _ = state_eeg_200hz(values, np.ones(90000, bool), 1000, 60000)
    assert not transformed['teacher_valid'][1999, 0, 0]
    assert not transformed['teacher_valid'][2020, 0, 0]
    assert np.isnan(transformed['s'][2000, 0, 0])
    assert np.isfinite(transformed['x']).all()


def test_metadata_selection_and_source_rejection_before_arrays(tmp_path, monkeypatch):
    cfg = settings(); root, source = fixture_cache(tmp_path, cfg)
    plan = state_pilot_plan(root, cfg['data'], source)
    assert len(plan['windows']) == 5 and plan['measured_arrays_read'] == 0
    chosen = plan['windows'][0]
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('unexpected array read'))
    bad = deepcopy(chosen); bad['row']['split'] = 'test'
    with pytest.raises(ValueError, match='plan changed'):
        state_pilot_window(root, bad, cfg['data'])
    meta_path = Path(chosen['row']['metadata'])
    meta = json.loads(meta_path.read_text()); meta['array_file'] = '/outside.npz'
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='escapes'):
        state_pilot_window(root, chosen, cfg['data'])


def test_paired_coordinates_masks_and_training_only_scales(tmp_path):
    cfg = settings(); root, source = fixture_cache(tmp_path, cfg)
    plan = state_pilot_plan(root, cfg['data'], source)
    selected = next(w for w in plan['windows'] if w['row']['dataset'] == 'refed' and w['role'] == 'first_full')
    record = state_pilot_window(root, selected, cfg['data'])
    info, packed = state_pilot_coordinates(record['metadata'], record['arrays'], 'eeg', .01)
    assert info['rate'] == 200 and packed['x'].shape == (18000, 2, 1)
    assert info['train_starts_s'][-1] + 30 <= info['holdout_starts_s'][0]
    altered = {k: v.copy() for k, v in record['arrays'].items()}
    for key, factor in [('x', 1.), ('s', .7), ('o', .3)]:
        altered['eeg_' + key][60000:] += 100 * factor
    later, values = state_pilot_coordinates(record['metadata'], altered, 'eeg', .01)
    np.testing.assert_array_equal(info['scale'], later['scale'])
    np.testing.assert_array_equal(packed['x'][:12000], values['x'][:12000])
    hb, arrays = state_pilot_coordinates(record['metadata'], record['arrays'], 'hb', .01)
    assert hb['unavailable_positions'] == [1] and np.isnan(arrays['s'][:, 1]).all()
    assert np.isfinite(arrays['x']).all()


def test_raw_quantization_loss_does_not_read_component_targets():
    x = torch.ones(2, 15, 40)
    result = dict(sum=x, S=x * .5, O=x * .5, commitment=torch.tensor(0.))
    batch = dict(x=x, s=x * .9, o=x * .1, mask=torch.ones_like(x, dtype=torch.bool),
                 vx=x, vs=x, vo=x)
    assert runner._state_loss(result, batch, 'q_x') == 0
    assert runner._state_loss(result, batch, 'q_so') > 0
    batch['s'][:] = float('nan'); batch['o'][:] = float('nan')
    assert runner._state_loss(result, batch, 'q_x') == 0


def test_pilot_training_three_controls_and_completed_resume(tmp_path, monkeypatch):
    cfg = settings(); root, source = fixture_cache(tmp_path, cfg)
    plan = state_pilot_plan(root, cfg['data'], source); out = tmp_path / 'run'
    monkeypatch.setattr(runner, '_bounded_jobs', lambda worker, payloads, *a, **k: [worker(p) for p in payloads])
    manifest = runner.prepare_state_pilot(cfg, out, plan)
    assert manifest['EEG_hz'] == 200 and len(manifest['rows']) == 10
    for candidate in ('warm_so', 'warm_x', 'continuous', 'q_so', 'q_x'):
        result = runner.train_state_pilot(cfg, out, 'eeg', candidate, 'cpu')
        assert result['status'] == 'completed'
        if not candidate.startswith('warm_'):
            assert result['total_exposure_steps'] == 5
            assert len(result['evaluation']['by_dataset']) == 4
            assert result['evaluation']['replay_max_abs'] < 2e-6
    result = runner.train_state_pilot(cfg, out, 'eeg', 'q_so', 'cpu')
    assert result['completed_steps'] == 3
    with pytest.raises(ValueError, match='benchmark'):
        runner.train_state_pilot(cfg, out, 'eeg', 'q_so', 'cpu', steps=400)


def test_known_state_and_task_locked_negative_control():
    result = runner.state_pilot_synthetic(settings())
    assert result['status'] == 'passed' and result['known_state_accuracy'] >= .8
    assert result['task_locked_pairing_gain'] == 0 and result['measured_arrays_read'] == 0


def test_visualization_weights_balance_parent_and_dataset_not_window_count():
    from src.analysis.dual_component_tokenizer import state_balanced_weights
    ds = np.array([0, 0, 0, 1, 1, 1]); parent = np.array([0, 0, 1, 2, 2, 2])
    w = state_balanced_weights(ds, parent, np.array([True, True, True, True, True, False]))
    np.testing.assert_allclose(w, [.125, .125, .25, .25, .25, 0])


def test_visualization_pca_axes_are_frozen_before_holdout_projection():
    from src.analysis.dual_component_tokenizer import state_weighted_pca
    x = np.array([[1., 0], [-1, 0], [0, .2], [0, -.2], [100., 200.]])
    w = np.array([.25, .25, .25, .25, 0])
    fit = state_weighted_pca(x, w)
    x[-1] *= 10000
    second = state_weighted_pca(x, w)
    np.testing.assert_array_equal(fit['mean'], second['mean'])
    np.testing.assert_array_equal(fit['axes'], second['axes'])
    np.testing.assert_allclose(fit['explained'], [1 / 1.04, .04 / 1.04])


def test_pattern_psd_preserves_oscillation_when_average_waveform_cancels():
    from src.analysis.dual_component_tokenizer import state_pattern_statistics
    t = np.arange(400) / 200
    signal = np.sin(2 * np.pi * 10 * t)
    patches = np.stack([signal, -signal])[:, :, None]
    result = state_pattern_statistics(patches, np.ones(2), 200)
    np.testing.assert_allclose(result['mean'], 0, atol=1e-15)
    assert result['frequency'][np.argmax(result['psd'][:, 0])] == 10
    assert np.isclose(np.trapezoid(result['psd'][:, 0], result['frequency']), .5)
    assert result['mean_waveform_energy_fraction'] == 0


@pytest.mark.parametrize('stage', ['state-visualize', 'state-reconstruction'])
def test_visualization_rejects_unfinished_parent_before_arrays(tmp_path, monkeypatch, stage):
    cfg = runner.read_config(CONFIG)
    runner.write_json(tmp_path / 'summary.json', dict(status='running'))
    runner.write_json(tmp_path / 'resolved_config.json', dict(config=cfg))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('unexpected measured-array read'))
    with pytest.raises(ValueError, match='completed exact'):
        runner.main(['--config', str(CONFIG), '--stage', stage, '--output', str(tmp_path),
                     '--analysis-output', str(tmp_path / 'fresh')])


@pytest.mark.parametrize('stage', ['state-visualize', 'state-reconstruction'])
def test_visualization_uses_existing_canonical_config_serialization(tmp_path, monkeypatch, stage):
    cfg = runner.read_config(CONFIG)
    runner.write_json(tmp_path / 'summary.json', dict(status='completed'))
    runner.write_json(tmp_path / 'resolved_config.json', dict(config=cfg))
    def reached_boundary(*args):
        raise RuntimeError('fixture cache boundary reached')
    monkeypatch.setattr(runner, '_state_cache', reached_boundary)
    with pytest.raises(RuntimeError, match='fixture cache boundary reached'):
        runner.main(['--config', str(CONFIG), '--stage', stage, '--output', str(tmp_path),
                     '--analysis-output', str(tmp_path / 'fresh')])


def test_atlas_long_control_title_fits_with_bitmap_and_searchable_caption(tmp_path):
    from PIL import Image
    from src.analysis.dual_component_tokenizer import assemble_state_token_atlas
    Image.new('RGB', (300, 200), 'white').save(tmp_path / 'fixture.png')
    page = dict(figure='fixture.png', title='EEG | Raw quantization | codebook 1：holdout token patterns（1/2）',
                caption='固定合成检查：30秒输入，每2秒一个锚点。图像为位图，图注保持可检索；本页不使用实测数据。',
                category='patterns', book='eeg_q_x_S')
    record = dict(key='eeg_q_x_S', label='Synthetic fixture', pages=[page], PCA_first_two_variance=.8,
                  latent_participation_rank=2., PC1_observed_RMS_correlation=0., PC1_observed_mean_correlation=0.,
                  code_dataset_MI_bits=0., code_slot_MI_bits=0., tsne=[])
    result = assemble_state_token_atlas(tmp_path, [record], {})
    assert result['pdf_pages'] == 1
    check = json.loads((tmp_path / 'pdf_verification.json').read_text())['pages'][0]
    assert check['images'] == 1 and check['vector_paths'] == 0 and check['searchable_characters'] > 40


def test_reconstruction_selection_keeps_failure_denominator_and_named_supplements():
    from src.analysis.dual_component_tokenizer import select_state_reconstruction_examples
    base = dict(modality='hb', dataset='refed', subject='s', record='r', parent='p',
                role='within_subject_holdout', start_s=60, duration_s=90)
    rows = [dict(base, position=str(i), x_nmse=float(i)) for i in range(11)]
    rows[3]['duration_s'] = 60
    rows += [dict(base, position='failed', x_nmse=None),
             dict(base, position='ignored', role='train', x_nmse=-10),
             dict(base, dataset='visual_cognitive_motivation', position='Probe1:CH23', x_nmse=80.)]
    result = select_state_reconstruction_examples(rows)
    refed = [r for r in result['examples'] if r['dataset'] == 'refed']
    assert [r['position'] for r in refed] == ['1', '5', '9', '3']
    assert refed[-1]['selection_tags'] == ['short-parent']
    assert result['denominators'][0] == dict(modality='hb', dataset='refed', planned=12, available=11, failed=1)
    visual = result['examples'][-1]
    assert visual['selection_tags'] == ['P10', 'P50', 'P90', 'retained-high-error-position']


def test_reconstruction_metrics_preserve_masks_and_component_training_variance():
    from src.analysis.dual_component_tokenizer import state_reconstruction_metrics
    target = np.array([[0., 0.], [2., 4.], [np.nan, 8.]])
    estimate = np.array([[2., 4.], [4., 4.], [1e9, 4.]])
    mask = np.array([[True, True], [True, True], [False, True]])
    result = state_reconstruction_metrics(target, estimate, mask, np.array([4., 16.]))
    assert result['channels'][0]['valid_samples'] == 2
    assert result['channels'][1]['valid_samples'] == 3
    np.testing.assert_allclose([r['NMSE'] for r in result['channels']], [1., 2/3])
    assert result['NMSE'] == pytest.approx(5/6)
    assert result['NRMSE'] == pytest.approx(np.sqrt(5/6))
    assert result['channels'][0]['correlation'] == pytest.approx(1.)
    assert result['channels'][1]['correlation'] is None
    with pytest.raises(ValueError, match='empty scored support'):
        state_reconstruction_metrics(target, estimate, np.zeros_like(mask), np.ones(2))


@pytest.mark.parametrize('modality,rate,channels', [('eeg', 200, 1), ('hb', 10, 2)])
def test_reconstruction_synthetic_curve_and_bitmap_atlas(tmp_path, modality, rate, channels):
    from src.analysis.dual_component_tokenizer import (
        state_reconstruction_metrics, render_state_reconstruction_case, assemble_state_reconstruction_atlas)
    t = np.arange(30*rate)/rate
    x = np.sin(2*np.pi*.3*t)[:, None] * np.arange(1, channels+1)[None, :]
    mask = np.ones_like(x, bool); mask[:3] = False
    arrays = dict(x=x, s=x*.7, o=x*.3, mask=mask)
    info = dict(modality=modality, rate=rate, dataset='refed', subject='synthetic', record='fixture',
                position='channel', start_s=30, duration_s=60, selection_tags=['short-parent'], metrics={})
    for model, factor in [('hard', .8), ('continuous', .9)]:
        info['metrics'][model] = {}
        for component in ('s', 'o', 'x'):
            arrays[model+'_'+component] = arrays[component]*factor
            info['metrics'][model][component] = state_reconstruction_metrics(
                arrays[component], arrays[model+'_'+component], mask, np.ones(channels))
    runner.write_npz(tmp_path/'case.npz', **arrays); runner.write_json(tmp_path/'case.json', info)
    render = render_state_reconstruction_case(tmp_path, 'case')
    result = assemble_state_reconstruction_atlas(tmp_path, [render], {'rule':'synthetic fixture'})
    assert result['cases'] == result['pdf_pages'] == 1
    assert (tmp_path/'case.png').stat().st_size > 10000
    assert '30秒' in render['page']['caption']


@pytest.mark.parametrize('modality', ['eeg', 'hb'])
def test_waveform_refinement_integer_replay_branch_isolation_and_gradients(modality):
    torch.set_num_threads(2);torch.manual_seed(17)
    cfg=runner.read_config(REFINEMENT)
    cfg['architectures']['waveform'].update(width=16,latent_dim=8,local_width=4)
    model=runner._state_refinement_model(cfg,'waveform',modality,'vq128')
    x=torch.randn(10,15,400 if modality=='eeg' else 40)
    model.initialize_codebooks(x);model.eval()
    before={k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        result=model(x);replay=model.decode_ids(result['ids'])
        changed=dict(result['ids'],O=(result['ids']['O']+1)%128);other=model.decode_ids(changed)
    assert result['sum'].shape==x.shape and result['ids']['S'].shape==(10,15)
    torch.testing.assert_close(replay['S'],result['S'],atol=2e-6,rtol=1e-5)
    assert torch.equal(replay['S'],result['S']) and torch.equal(replay['O'],result['O'])
    with torch.no_grad():
        far=model(x*1e6);direct=model.decode_ids(far['ids'])
    assert torch.equal(far['sum'],direct['sum'])
    assert torch.equal(replay['S'],other['S']) and not torch.equal(replay['O'],other['O'])
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    model.train();prediction=model(x);prediction['sum'].square().mean().backward()
    for b in ('S','O'):
        for module in ('input','output'):
            assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.branches[b][module].parameters())


def test_refinement_delta_loss_uses_adjacent_valid_samples_and_both_colors():
    x=torch.zeros(1,15,40);mask=torch.ones_like(x,dtype=torch.bool)
    batch=dict(x=x,s=x,o=x,mask=mask,vx=torch.ones_like(x),vs=torch.ones_like(x),vo=torch.ones_like(x),
               dvs=torch.ones(1,2),dvo=torch.ones(1,2),dvx=torch.ones(1,2))
    s=x.clone();s[0,0,0]=1000;mask[0,0,0]=False
    result=dict(S=s,O=x,sum=s,commitment=torch.tensor(0.))
    loss,terms=runner._state_refinement_loss(result,batch,dict(sum_weight=.5,derivative_weight=.1))
    assert loss==0 and terms['derivative']==0
    s[0,0,1]=1
    loss,terms=runner._state_refinement_loss(result,batch,dict(sum_weight=.5,derivative_weight=.1))
    assert loss>0 and terms['derivative']>0


def test_refinement_rejects_data_scope_before_array_reads(tmp_path,monkeypatch):
    cfg=runner.read_config(REFINEMENT);cfg['data']['allowed_splits']=['train','test']
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('unexpected array read'))
    with pytest.raises(ValueError,match='before arrays'):runner._state_refinement_cache(cfg,'eeg')
    with pytest.raises(ValueError,match='bounded refinement'):runner.main(['--config',str(REFINEMENT),'--stage','all'])


def test_refinement_fixed_budgets_shared_warmup_and_completed_resume(tmp_path,monkeypatch):
    old=settings();root,source=fixture_cache(tmp_path,old)
    plan=state_pilot_plan(root,old['data'],source);parent=tmp_path/'parent'
    monkeypatch.setattr(runner,'_bounded_jobs',lambda worker,payloads,*a,**k:[worker(p) for p in payloads])
    runner.prepare_state_pilot(old,parent,plan)
    manifest,rows,arrays=runner._state_cache(old,parent,'hb')
    for row in rows:row['difference_variance']=row['target_variance']
    cfg=runner.read_config(REFINEMENT)
    cfg['training'].update(warmup_steps=2,continuation_steps=3,batch_size=4,checkpoint_every=1,initialization_per_dataset=4)
    cfg['architectures']['waveform'].update(width=16,latent_dim=8,local_width=4)
    monkeypatch.setattr(runner,'_state_refinement_cache',lambda *a:(old,manifest,rows,arrays))
    out=tmp_path/'refinement'
    warm=runner.train_state_refinement(cfg,out,'hb','waveform__warmup','cpu')
    fit=runner.train_state_refinement(cfg,out,'hb','waveform__vq128','cpu')
    assert warm['completed_steps']==2 and fit['completed_steps']==3 and fit['total_exposure_steps']==5
    assert fit['evaluation']['replay_max_abs']<2e-6
    assert runner.train_state_refinement(cfg,out,'hb','waveform__vq128','cpu')==fit
    with pytest.raises(ValueError,match='benchmark'):runner.train_state_refinement(cfg,out,'hb','waveform__vq128','cpu',200)
