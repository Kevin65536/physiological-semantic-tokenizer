"""Global latent Hb pilot contracts using synthetic, non-authorizing fixtures."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tokenizers.dual_component_tokenizer import GlobalLatentComponentTokenizer
from src.data.dual_component_tokenizer import latent_health_plan, latent_health_hb_patch
from src.analysis.dual_component_tokenizer import latent_codebook_health, latent_hb_fidelity

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('latent_health_runner', ROOT / 'experiments/scripts/run_dual_component_tokenizer.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
CONFIG = ROOT / 'experiments/configs/physiology_semantic_tokenizer/dual_component_latent_health_v1.yaml'


def settings():
    cfg = runner.read_config(CONFIG)
    cfg['latent_health'].update(width=16, latent_dim=8, heads=2, encoder_layers=1, decoder_layers=1, kmeans_iters=2)
    return cfg


def test_contract_and_disallowed_stage_before_reads(tmp_path, monkeypatch):
    cfg = runner.read_config(CONFIG)
    assert cfg['latent_health']['patches'] == 15
    assert cfg['latent_health']['codebook_sizes'] == [128, 512]
    with pytest.raises(ValueError, match='legacy stages'):
        runner.main(['--config', str(CONFIG), '--stage', 'all', '--output', str(tmp_path)])
    bad = deepcopy(cfg['latent_health']); bad['allowed_splits'].append('test')
    with pytest.raises(ValueError, match='before arrays'):
        latent_health_plan(tmp_path / 'absent', bad)
    touched = []
    monkeypatch.setattr(runner, 'read_arrays', lambda *a, **k: touched.append(True))
    result = runner._latent_health_trial(('absent', dict(trial_id='x', rows=[dict(split='test')]), {}, cfg['latent_health']))
    assert result['status'] == 'failed' and not touched


@pytest.mark.parametrize('dimension', [40, 400])
def test_one_id_per_patch_integer_replay_and_branch_isolation(dimension):
    torch.set_num_threads(2); torch.manual_seed(3)
    spec = dict(settings()['latent_health'], input_dim=dimension)
    model = GlobalLatentComponentTokenizer(spec, codebook_size=8)
    x = torch.randn(3, 15, dimension)
    model.initialize_codebooks(x)
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        result = model(x)
        decoded = model.decode_ids(result['ids'])
        changed = dict(result['ids'], O=(result['ids']['O'] + 1) % 8)
        other = model.decode_ids(changed)
    assert result['ids']['S'].shape == (3, 15)
    assert decoded['S'].shape == x.shape
    assert torch.allclose(decoded['S'], result['S'], atol=2e-5, rtol=0)
    assert torch.equal(other['S'], decoded['S'])
    assert not torch.equal(other['O'], decoded['O'])
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    with pytest.raises(ValueError, match='integer'):
        model.decode_ids({c: result['ids'][c].float() for c in ('S', 'O')})
    with pytest.raises(ValueError, match='sidecars'):
        model.decode_ids(dict(result['ids'], amplitude=torch.ones(3)))


def test_global_input_and_decoder_context_is_explicit():
    torch.manual_seed(23)
    model = GlobalLatentComponentTokenizer(settings()['latent_health'], codebook_size=8)
    x = torch.randn(2, 15, 40)
    model.initialize_codebooks(x); model.eval()
    changed = x.clone(); changed[0, 0] += 10
    with torch.no_grad():
        first = model.encode_continuous(x); second = model.encode_continuous(changed)
    assert not torch.allclose(first['S'][0, 14], second['S'][0, 14])
    assert torch.allclose(first['S'][1], second['S'][1], atol=1e-6)
    ids = {c: torch.zeros(2, 15, dtype=torch.long) for c in ('S', 'O')}
    changed_ids = {c: v.clone() for c, v in ids.items()}; changed_ids['S'][0, 0] = 1
    with torch.no_grad():
        a, b = model.decode_ids(ids), model.decode_ids(changed_ids)
    assert not torch.allclose(a['S'][0, 14], b['S'][0, 14])
    assert torch.equal(a['O'], b['O'])


def test_hb_pair_extraction_preserves_target_and_observation_identity():
    spec = settings()['latent_health']
    row = dict(id='fixture', split='validation', dataset='simultaneous_eeg_nirs')
    u = np.arange(2700, dtype=float).reshape(900, 3) / 100
    a = dict(identity=np.array('fixture'), teacher_arm=np.array('tak_fixed'), u=u, physical=.8 * u, remaining=.2 * u)
    result = latent_health_hb_patch(row, a, [1., 2., 4.], spec)
    expected = (u[250:550, 1:] / [2., 4.]).reshape(15, 40)
    np.testing.assert_allclose(result['u'], expected)
    np.testing.assert_allclose(result['S'] + result['O'], result['u'], rtol=2e-6)
    other = dict(a, physical=.1 * u, remaining=.9 * u)
    np.testing.assert_array_equal(latent_health_hb_patch(row, other, [1., 2., 4.], spec)['u'], result['u'])
    with pytest.raises(ValueError, match='cannot enter'):
        latent_health_hb_patch(dict(row, split='test'), a, [1., 2., 4.], spec)


def test_full_population_health_rejects_dataset_specific_collapse():
    rows = [dict(dataset=ds, split=split, subject=f'{split}_{i}', record='r')
            for ds in ('a', 'b') for split in ('train', 'validation') for i in range(4)]
    ids = np.tile(np.arange(4), (16, 1)).astype('int16')
    ids[8:] = 0
    h = latent_codebook_health(ids, np.eye(4), rows, settings()['latent_health']['health'])
    assert not h['passed']
    assert next(s for s in h['summaries'] if s['dataset'] == 'pooled' and s['split'] == 'train')['active_codes'] == 4
    b = next(s for s in h['summaries'] if s['dataset'] == 'b' and s['split'] == 'validation')
    assert b['active_codes'] == 1 and not b['passed']
    assert max(r['subjects'] for r in h['coverage'] if r['dataset'] == 'b') == 4


def test_fidelity_separates_color_and_subject_denominator():
    rows = [dict(dataset='a', split='validation', subject='s1', record='r')] * 3
    rows += [dict(dataset='a', split='validation', subject='s2', record='r')]
    truth = np.zeros((4, 15, 40), dtype='float32')
    estimate = truth.copy(); estimate[:3, :, 0::2] = 1.; estimate[3, :, 0::2] = 3.
    result = latent_hb_fidelity(dict(S=estimate, O=truth, sum=estimate), dict(S=truth, O=truth, u=truth), rows)
    metric = next(m for m in result['metrics'] if m['branch'] == 'S' and m['signal'] == 'HbO' and m['metric'] == 'NMSE')
    assert metric['subjects'] == 2 and metric['value'] == 5.
    hb_r = next(m for m in result['metrics'] if m['branch'] == 'S' and m['signal'] == 'HbR' and m['metric'] == 'NMSE')
    assert hb_r['value'] == 0.


def test_bounded_training_and_shared_warmup_checkpoint(tmp_path):
    torch.set_num_threads(2)
    cfg = settings(); cfg['latent_health'].update(warmup_steps=2, continuation_steps=3, batch_size=4,
        initialization_rows=16, checkpoint_every=1, evaluation_batch_size=4)
    folder = tmp_path / 'cache'; folder.mkdir()
    rows = [dict(id=f'{ds}_{sp}_{i}', dataset=ds, split=sp, subject=f'{sp}{i}', record='r')
            for ds in ('eeg_fnirs_single_trial', 'simultaneous_eeg_nirs') for sp in ('train', 'validation') for i in range(2)]
    rng = np.random.default_rng(11); u = rng.normal(size=(8, 15, 40)).astype('float32')
    for k, v in dict(u=u, S=.8 * u, O=.2 * u).items():
        np.save(folder / f'{k}.npy', v)
    runner.write_json(folder / 'manifest.json', dict(schema='dual_component_latent_health_cache_v1', rows=rows,
        arrays={k: f'{k}.npy' for k in ('u', 'S', 'O')}, test_arrays_read=0, plan_identity='synthetic_fixture',
        target_variance=dict(S=[1, 1], O=[.1, .1])))
    warmup = runner.train_latent_health(cfg, tmp_path, 'warmup', 17, 'cpu', memory_bytes=2**32)
    result = runner.train_latent_health(cfg, tmp_path, 'vq128', 17, 'cpu', memory_bytes=2**32)
    assert warmup['completed_steps'] == 2
    assert result['completed_steps'] == 3 and result['total_exposure_steps'] == 5
    assert result['evaluation']['integer_decode_max_abs_difference'] < 2e-5
    initialization = runner.read_json(tmp_path / 'latent_health/vq128/seed_17/initialization.json')
    assert all(rows[i]['split'] == 'train' for i in initialization['row_indices'])
    assert runner.train_latent_health(cfg, tmp_path, 'vq128', 17, 'cpu')['completed_steps'] == 3
    with pytest.raises(ValueError, match='resource pilots'):
        runner.train_latent_health(cfg, tmp_path, 'continuous', 17, 'cpu', steps=500)


def test_no_target_or_other_modality_in_synthetic_interface():
    result = runner.synthetic_latent_health(settings())
    assert result['status'] == 'passed' and result['measured_arrays_read'] == 0
    assert [r['ID_shape'] for r in result['cases']] == [[4, 15], [4, 15]]
