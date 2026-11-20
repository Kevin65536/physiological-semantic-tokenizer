"""Synthetic weights/windows only; the observation adapter needs no G0 data tree."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.analysis.observation_legacy_control import (
    legacy_observation_reconstruct, load_legacy_observation_control,
)
from src.tokenizers.dual_component_tokenizer import GlobalLatentComponentTokenizer


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = Path('experiments/configs/physiology_semantic_tokenizer')
RUNS = Path('experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer')
SOURCE = {
    ('eeg', False): ('20261120_input_quantization_b_v1', 'matrix', 'waveform8'),
    ('eeg', True): ('20261119_e1_waveform8_v1', 'e1', 'waveform8'),
    ('hb', False): ('20261117_base_g1_v2', 'g1', 'waveform'),
    ('hb', True): ('20261117_base_g1_v2', 'g1', 'waveform'),
}


def _dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


@pytest.fixture(scope='module', autouse=True)
def _small_cpu_pool():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def _fixture(root, mod='hb', continuous=False, *, create_weights=True):
    run_name, family, arch = SOURCE[mod, continuous]
    path = CONFIGS / f'dual_component_base_{family}_v1.yaml'
    text = (ROOT / path).read_text()  # reviewed software contract, no real run or dataset
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_text(text)
    cfg = yaml.safe_load(text)
    task = dict(modality=mod, architecture=arch, candidate='continuous' if continuous else 'vq128')
    task_id = '__'.join(task.values())
    run = root / RUNS / run_name
    job = run / 'models' / task_id
    sign = dict(config=sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:24],
                input='37319917430f3aa11e3eaf39', scales='d8b6634e55fa73bbc1243650', task=task, benchmark=False)
    _dump(run / 'resolved_config.json', dict(config=cfg, project_root=str(root)))
    _dump(run / 'plan.json', dict(G0_run=cfg['data']['run'], manifest_identity=sign['input'],
                                  scales_identity=sign['scales'], tasks=cfg['tasks'], seed=17))
    _dump(job / 'summary.json', dict(status='completed', completed_step=10000, task_id=task_id, signature=sign))
    pt = job / 'checkpoint_step_10000.pt'
    if not create_weights:
        return run, job, pt, None
    c, t = (1, 400) if mod == 'eeg' else (2, 20)
    setting = dict(cfg['model'], **cfg['architectures'][arch], channels=c, input_dim=t*c)
    setting.setdefault('sample_strides', [5] if mod == 'hb' else [5, 2, 5])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(121)
        model = GlobalLatentComponentTokenizer(setting, codebook_size=128, hard=not continuous).eval()
        if not continuous:
            # Random synthetic dictionaries; do not fit K-means or perform any training.
            for b in model.branches.values():
                b['quantizer'].initialized.fill_(True)
    torch.save(dict(signature=sign, step=10000, model_state=model.state_dict()), pt)
    return run, job, pt, model


@pytest.mark.parametrize('mod,continuous', SOURCE)
def test_frozen_X_only_replay_matches_owner_and_interleaved_color_layout(tmp_path, monkeypatch, mod, continuous):
    _, _, _, owner = _fixture(tmp_path, mod, continuous)
    c, t = (1, 400) if mod == 'eeg' else (2, 20)
    windows = np.random.default_rng(403).normal(size=(3, 15, c, t))
    if mod == 'hb':  # make accidental channel-major flattening easy to detect
        windows[:, :, 1] += np.linspace(10, 15, t)
    windows.setflags(write=False)
    raw = torch.tensor(windows.transpose(0, 1, 3, 2).reshape(3, 15, c*t), dtype=torch.float32)
    with torch.inference_mode():
        expected = owner(raw)
        pred = expected['sum'].reshape(3, 15, t, c).permute(0, 1, 3, 2).numpy()
    monkeypatch.setattr(np, 'load', lambda *_a, **_kw: pytest.fail('no signal/teacher array reads'))
    monkeypatch.setattr(GlobalLatentComponentTokenizer, 'encode_components',
                        lambda *_a, **_kw: pytest.fail('teacher inputs are forbidden'))
    if not continuous:
        monkeypatch.setattr(GlobalLatentComponentTokenizer, 'forward',
                            lambda *_a, **_kw: pytest.fail('hard decoder must be called on integers'))
    rng = torch.get_rng_state().clone()
    control = load_legacy_observation_control(tmp_path, mod, continuous=continuous)
    actual, ids = control.reconstruct(windows, return_ids=True, batch_size=2)
    np.testing.assert_allclose(actual, pred, atol=2e-5, rtol=2e-5)
    assert actual.dtype == np.float32 and actual.shape == windows.shape
    assert not control._model.training and not any(p.requires_grad for p in control._model.parameters())
    assert torch.equal(torch.get_rng_state(), rng)
    if continuous:
        assert ids is None and control.provenance['nominal_bits_per_two_seconds_per_location'] is None
    else:
        assert set(ids) == {'S', 'O'} and control.provenance['nominal_bits_per_two_seconds_per_location'] == 14
        for branch, codes in ids.items():
            np.testing.assert_array_equal(codes, expected['ids'][branch].numpy())
            assert codes.dtype == np.uint8 and codes.shape == (3, 15) and np.max(codes) < 128
            for key, value in owner.branches[branch]['quantizer'].state_dict().items():
                torch.testing.assert_close(value, control._model.branches[branch]['quantizer'].state_dict()[key])
    single = legacy_observation_reconstruct(tmp_path, mod, windows[:1], continuous=continuous)
    np.testing.assert_allclose(single, pred[:1], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('defect', ['contract', 'root', 'target', 'scales', 'failed', 'source_step', 'symlink'])
def test_rejects_untrusted_sources_before_loading_weights(tmp_path, monkeypatch, defect):
    run, job, pt, _ = _fixture(tmp_path, create_weights=False)
    if defect in ('contract', 'root'):
        path = run / 'resolved_config.json'
        obj = json.loads(path.read_text())
        if defect == 'contract':
            obj['config']['architectures']['waveform']['target_input'] = True
        else:
            obj['project_root'] = '/unapproved'
    elif defect == 'scales':
        path = run / 'plan.json'; obj = json.loads(path.read_text()); obj['scales_identity'] = 'other'
    elif defect == 'symlink':
        (tmp_path / 'outside.pt').touch(); pt.symlink_to(tmp_path / 'outside.pt')
    else:
        path = job / 'summary.json'; obj = json.loads(path.read_text())
        if defect == 'target':
            obj['signature']['task']['architecture'] = 'target_input'
        elif defect == 'failed':
            obj['status'] = 'failed'
        else:
            obj['completed_step'] = 7000
    if defect != 'symlink':
        _dump(path, obj)
    monkeypatch.setattr(torch, 'load', lambda *_a, **_kw: pytest.fail('must reject before loading checkpoint'))
    with pytest.raises(ValueError):
        load_legacy_observation_control(tmp_path, 'hb')


@pytest.mark.parametrize('defect', ['signature', 'step', 'shape', 'nonfinite', 'uninitialized'])
def test_rejects_mismatched_or_incomplete_checkpoint(tmp_path, defect):
    _, _, pt, _ = _fixture(tmp_path)
    saved = torch.load(pt, weights_only=True)
    if defect == 'step':
        saved['step'] = 7000
    elif defect == 'signature':
        saved['signature']['task']['candidate'] = 'continuous'
    elif defect == 'shape':
        saved['model_state']['position'] = torch.zeros(1, 8, 128)
    elif defect == 'nonfinite':
        saved['model_state']['position'][0, 0, 0] = float('nan')
    else:
        saved['model_state']['branches.O.quantizer.initialized'].fill_(False)
    torch.save(saved, pt)
    with pytest.raises((ValueError, RuntimeError)):
        load_legacy_observation_control(tmp_path, 'hb')


def test_geometry_mask_nonfinite_and_metadata_bypasses_are_rejected(tmp_path):
    _fixture(tmp_path)
    control = load_legacy_observation_control(tmp_path, 'hb')
    valid = np.ones((1, 15, 2, 20), dtype=np.float32)
    for bad in (valid.transpose(0, 1, 3, 2), valid[:, :-1], valid.astype(int), valid[:0],
                np.ma.masked_array(valid, False), {'x': valid, 'position': 2}, valid * np.nan):
        with pytest.raises(ValueError):
            control.reconstruct(bad)
    with pytest.raises(TypeError):
        control.reconstruct(valid, teacher=valid)
    with pytest.raises(ValueError):
        control.reconstruct(valid, batch_size=0)
    with pytest.raises(ValueError):
        load_legacy_observation_control(tmp_path, 'target_input')
