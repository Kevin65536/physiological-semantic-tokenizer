"""Synthetic refinement checks; no local experiment/data artifacts required."""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest
import torch

from src.tokenizers.dual_component_tokenizer import WaveformDictionary, local_patch_context, DualComponentTokenizer
from src.analysis.dual_component_tokenizer import (
    refinement_fidelity, refinement_code_health, refinement_decision, codebook_health,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('refinement_runner', ROOT / 'experiments/scripts/run_dual_component_tokenizer.py')
runner = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(runner)


def test_extended_support_and_hidden_value_isolation():
    x = torch.arange(20.).reshape(1, 5, 4)
    context, mask = local_patch_context(x, radius=1)
    assert not mask[0, 0, :4].any() and not mask[0, -1, -4:].any()
    assert torch.equal(context[0, 2], x[0, 1:4].flatten())
    poisoned = x.clone(); poisoned[:, 2] += 1e6
    after, _ = local_patch_context(poisoned, radius=1)
    # Hide all derived tokens whose actual input support intersects patch 2.
    assert torch.equal(context[:, [0, 4]], after[:, [0, 4]])
    assert not torch.equal(context[:, 1:4], after[:, 1:4])


def test_waveform_dictionary_decodes_amplitude_and_explicit_level_rate():
    rng = np.random.default_rng(12); t = np.linspace(-1, 1, 20)
    a = rng.normal(size=(600, 2)); waves = (a[:, :1] + a[:, 1:] * t).astype('float32')
    book = WaveformDictionary(16, 2, 1, .1, 17, 500, 30).fit(waves[:500])
    ids = book.encode(waves[500:]); decoded = book.decode(ids)
    assert ids.shape == (100, 2) and ids.dtype.kind == 'i'
    assert np.array_equal(decoded, book.codewords[0][ids[:, 0]] + book.codewords[1][ids[:, 1]])
    single = book.decode(ids[:, :1])
    assert np.mean((decoded - waves[500:]) ** 2) < np.mean((single - waves[500:]) ** 2)
    assert book.state()['nominal_bits_per_component_patch'] == 8
    with pytest.raises(ValueError, match='integer'): book.decode(ids.astype(float))
    before = book.codewords.copy(); book.encode(waves[500:] * 1000)
    assert np.array_equal(book.codewords, before)


def test_record_coverage_keeps_subject_identity():
    rows = [dict(dataset='d', subject='a', record='session0', split='train'),
            dict(dataset='d', subject='b', record='session0', split='train'),
            dict(dataset='d', subject='c', record='session0', split='validation')]
    ids = np.zeros((3, 3, 1), dtype='int16'); words = np.array([[[0., 0.], [1., 1.]]])
    result = refinement_code_health(ids, words, rows, 'hb', 'S', 2)
    assert result[0]['per_code_records'][0] == 2
    legacy = codebook_health(ids[..., 0], words[0], rows, modality='hb', branch='S', variant='hard_student', seed=17)
    assert legacy[0]['records'] == 2


def test_health_cannot_replace_fidelity():
    baseline = [dict(dataset='d', split='validation', modality='hb', component='S', metric='NMSE', value=1.)]
    poor = [dict(baseline[0], value=1.2)]
    healthy = [dict(modality='hb', component='S', level=0, split='train', size=128,
                    active_fraction=1., perplexity=128., max_probability=1/128)]
    cfg = dict(minimum_waveform_improvement=.3, train_active_fraction=.95,
        validation_active_fraction=.9, effective_fraction=.5, maximum_code_probability=.08)
    verdict = refinement_decision(poor, baseline, healthy, cfg)
    assert verdict['health_pass'] and not verdict['fidelity_pass']


def test_fidelity_rejects_test_rows():
    with pytest.raises(ValueError, match='training/validation only'):
        refinement_fidelity({}, {}, [dict(split='test')], 'x')


def small_refinement(tmp_path):
    rng = np.random.default_rng(38); cache = tmp_path / 'cache'; cache.mkdir()
    rows = [dict(id=f'row{i}', trial_id=f'trial{i}', dataset='d', subject=f's{i}', record='r',
                 split='train' if i < 8 else 'validation') for i in range(12)]
    shapes = {}
    for modality, dim in [('eeg', 20), ('hb', 40)]:
        u = rng.normal(size=(12, 10, dim)).astype('float32')
        for suffix, value in [('input', u), ('u', u), ('s', .7*u), ('e', .3*u)]:
            np.save(cache / (modality + '_' + suffix + '.npy'), value)
            shapes[modality + '_' + suffix] = list(value.shape)
    runner.write_json(cache / 'manifest.json', dict(rows=rows, shapes=shapes,
        arrays={key:key+'.npy' for key in shapes}))
    candidate = dict(name='fixture', input_mode='observation', context_radius=1,
        training_mode='continuous_student', initial='fresh', steps=2, learning_rate=.001,
        balance_components=True, derivative_weight=.1)
    cfg = dict(model=dict(input_dims={'eeg':20,'hb':40}, target_dims={'eeg':20,'hb':40},
        width=8, latent_dim=4, codebook_size=4, kmeans_iters=2), resources=dict(torch_threads=1),
        refinement=dict(candidates=[candidate], batch_size=4, checkpoint_every=1,
            evaluation_batch_size=4, codebook_size=4, waveform_vq_levels=[1,2],
            codebook_fit_patches=80, codebook_fit_iterations=5))
    return cfg, rows


def test_refinement_runs_fixed_budget_and_exports_only_validation(tmp_path):
    cfg, rows = small_refinement(tmp_path)
    result = runner.refine_one(cfg, tmp_path, 'fixture', 17, 'cpu')
    assert result['status'] == 'completed' and len(result['variants']) == 2
    job = tmp_path / 'refinement/fixture/seed_17'
    checkpoint = torch.load(job / 'checkpoint.pt', weights_only=False)
    assert checkpoint['step'] == 2
    assert result['training']['completed_steps'] == 2
    with np.load(job / 'vq2/validation_predictions.npz') as export:
        assert export['row_index'].tolist() == [8,9,10,11]
        assert export['hb_sum'].shape == (4,10,40)
    assert result['variants'][0]['bits'] == 4 and result['variants'][1]['bits'] == 8
    rerun = runner.refine_one(cfg, tmp_path, 'fixture', 17, 'cpu')
    assert rerun == json.loads((job / 'summary.json').read_text())


def test_frontend_uses_only_observation_not_teacher(tmp_path):
    cfg, _ = small_refinement(tmp_path); _, arrays = runner.load_cache(tmp_path)
    candidate = cfg['refinement']['candidates'][0]
    before, mask = runner.refinement_inputs(arrays, [0,1], candidate, 'cpu')
    changed = dict(arrays, eeg_s=np.full_like(arrays['eeg_s'], 1e9), hb_e=np.full_like(arrays['hb_e'], -1e9))
    after, new_mask = runner.refinement_inputs(changed, [0,1], candidate, 'cpu')
    for modality in before:
        assert torch.equal(before[modality], after[modality])
        assert torch.equal(mask[modality], new_mask[modality])


def test_temporal_encoder_keeps_modality_independence():
    cfg=dict(input_dims={'eeg':1860,'hb':120},target_dims={'eeg':20,'hb':40},
        input_channels={'eeg':31,'hb':2},encoder_kind='temporal',width=8,latent_dim=4,codebook_size=4)
    model=DualComponentTokenizer(cfg,'continuous_student').eval()
    x={'eeg':torch.randn(2,10,1860),'hb':torch.randn(2,10,120)}
    before=model(x);after=model(dict(x,eeg=x['eeg']+100))
    assert before['eeg']['S'].shape==(2,10,20)
    assert torch.equal(before['hb']['S'],after['hb']['S'])
    assert not torch.equal(before['eeg']['S'],after['eeg']['S'])


def test_spectral_frontend_excludes_other_modality_and_teacher(tmp_path):
    cfg,_=small_refinement(tmp_path);_,arrays=runner.load_cache(tmp_path)
    arrays['eeg_spectral_input']=np.ones((12,10,620),dtype='float32')
    arrays['patch_metadata']=np.ones((12,10,140),dtype='float32')
    candidate=dict(input_mode='spectral',context_radius=1,include_metadata=True)
    before,_=runner.refinement_inputs(arrays,[0,1],candidate,'cpu')
    changed=dict(arrays,hb_u=arrays['hb_u']+1e6,eeg_s=arrays['eeg_s']+1e6,eeg_e=arrays['eeg_e']-1e6)
    after,_=runner.refinement_inputs(changed,[0,1],candidate,'cpu')
    assert before['eeg'].shape==(2,10,2280)
    assert torch.equal(before['eeg'],after['eeg'])
    assert not torch.equal(before['hb'],after['hb'])


def test_prepared_array_reader_can_exclude_other_modality_payload(tmp_path):
    row=dict(id='synthetic')
    runner.write_npz(runner.array_paths(tmp_path,'prepared',row),features=np.ones((900,30)),hb=np.full((900,2),np.nan))
    runner.task_finish(tmp_path,'prepared',row['id'],'completed')
    result=runner.read_arrays(tmp_path,'prepared',row,keys=('features',))
    assert set(result)=={'features'} and np.isfinite(result['features']).all()


def test_spectral_preparation_fits_normalization_on_training_only(tmp_path):
    from src.data.ssm_prepared import EEG_PROXY_INPUT_SCHEMA
    relative='experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261028_v3'
    parent=tmp_path/relative;parent.mkdir(parents=True)
    cfg,rows=small_refinement(parent)
    for row in rows:row['coordinate']='d__region'
    manifest=runner.read_json(parent/'cache/manifest.json')
    manifest.update(rows=rows,schema='dual_component_training_cache_v1',target_units='training_observation_SD',
        identity='synthetic_parent',retained_windows=len(rows),retained_trials=4,failed_trials=[])
    runner.write_json(parent/'cache/manifest.json',manifest)
    runner.write_json(parent/'inventory.json',dict(rows=rows))
    runner.write_json(parent/'controller.json',dict(status='completed'))
    runner.write_json(parent/'resolved_config.json',dict(config=dict(data=dict(protected_data='forbidden'))))
    for i,row in enumerate(rows):
        offset=float(i if row['split']=='train' else 10000+i)
        runner.write_npz(runner.array_paths(parent,'prepared',row),
            features=np.full((900,30),offset,dtype='float32'),identity=np.array(row['id']),
            eeg_proxy_input=np.array(EEG_PROXY_INPUT_SCHEMA),eeg_proxy_rate_hz=np.array(10.))
        runner.task_finish(parent,'prepared',row['id'],'completed')
    cfg['refinement'].update(parent_run=relative,allowed_splits=['train','validation'])
    cfg['refinement']['candidates'][0]['input_mode']='spectral'
    output=tmp_path/'refinement';output.mkdir()
    result=runner.prepare_refinement(cfg,output,tmp_path)
    fitted=result['spectral_frontend']['fitting']['d__region']
    assert np.allclose(fitted['mean'],3.5)
    assert fitted['training_rows']==8
    arrays=runner.load_cache(output)[1]
    assert arrays['eeg_spectral_input'].shape==(12,10,620)
    assert result['independent_subjects']['d']=={'train':8,'validation':4}


def test_failed_seed_replication_cannot_report_suite_complete(tmp_path, monkeypatch):
    cfg,_=small_refinement(tmp_path)
    cfg['refinement'].update(seed=17,confirmation_seeds=[29,43],seed_replication='selected_even_if_diagnostic')
    monkeypatch.setattr(runner,'prepare_refinement',lambda *a:None)
    monkeypatch.setattr(runner,'refinement_baseline',lambda *a:None)
    import src.analysis.dual_component_tokenizer as analysis
    monkeypatch.setattr(analysis,'render_refinement',lambda *a:[])
    variant=dict(name='fixture_vq1',bits=4,deployable_student=True,
        decision=dict(health_pass=True,fidelity_pass=False,fidelity=[dict(NMSE=.9,baseline_NMSE=1.)]))
    def jobs(*args):
        if args[-1]=='screen':
            return [dict(status='completed',candidate={'name':'fixture'},seed=17,variants=[variant])]
        return [dict(status='failed',seed=29),dict(status='completed',seed=43,variants=[variant])]
    monkeypatch.setattr(runner,'refinement_jobs',jobs)
    result=runner.run_refinement(cfg,tmp_path/'config.yaml',tmp_path,tmp_path,['cpu'],1)
    assert result['status']=='failed'
    summary=runner.read_json(tmp_path/'refinement_summary.json')
    assert len(summary['confirmation'])==2 and not summary['requirements_met']
