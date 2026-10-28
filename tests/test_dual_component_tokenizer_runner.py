"""Temporary synthetic fixtures for controller ownership and exact recovery."""
from copy import deepcopy
import io
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('dual_component_runner',
    ROOT / 'experiments/scripts/run_dual_component_tokenizer.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def small_cfg():
    return dict(schema='synthetic_runner_fixture', experiment_id='non_measured_fixture',
        variants=list(runner.VARIANTS), seeds=[17], data=dict(protected_data='forbidden'),
        model=dict(input_dims=dict(eeg=6, hb=4), target_dims=dict(eeg=2, hb=4),
            width=8, latent_dim=3, codebook_size=4, kmeans_iters=2, ema_decay=.9,
            commitment_cost=.25, kmeans_init=True),
        training=dict(steps=4, batch_size=2, learning_rate=.001, weight_decay=.0001,
            checkpoint_every=1, eval_batch_size=2, gradient_clip=5),
        resources=dict(torch_threads=1), smoothing=dict(sigma_steps=[0, 1]))


def fixture_cache(out, cfg):
    rng = np.random.default_rng(917)
    rows = [dict(id=f'fixture_{i}', trial_id=f'trial_{i}', dataset=f'dataset_{i % 2}',
                 subject=f'subject_{i}', record=f'record_{i}', coordinate='fixture',
                 split='train' if i < 4 else 'validation' if i == 4 else 'test') for i in range(6)]
    arrays = {}
    cache = Path(out) / 'cache'; cache.mkdir(parents=True)
    for modality in ('eeg', 'hb'):
        arrays[modality + '_input'] = rng.normal(size=(6, 3, cfg['model']['input_dims'][modality])).astype('float32')
        u = rng.normal(size=(6, 3, cfg['model']['target_dims'][modality])).astype('float32')
        arrays.update({modality + '_u': u, modality + '_s': u * .6, modality + '_e': u * .4})
    for name, values in arrays.items(): np.save(cache / (name + '.npy'), values)
    runner.write_json(cache / 'manifest.json', dict(rows=rows, shapes={k: v.shape for k, v in arrays.items()},
        observation_training_sd=dict(fixture=[1, 1, 1])))
    return rows, arrays


@pytest.fixture
def cuda_branch_on_cpu(monkeypatch):
    """Execute CUDA budget control flow with real CPU model/optimizer work."""
    module_to, make_batch, load = torch.nn.Module.to, runner._batch, torch.load
    for name in ('set_device', 'manual_seed', 'manual_seed_all',
                 'reset_peak_memory_stats', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: None)
    for name in ('max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: 0)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
        lambda device: SimpleNamespace(total_memory=8 * 1024 ** 3))
    fractions = []
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction',
        lambda fraction, device: fractions.append(fraction))
    monkeypatch.setattr(torch.nn.Module, 'to', lambda self, *a, **kw: module_to(self, 'cpu'))
    monkeypatch.setattr(runner, '_batch', lambda arrays, indices, device, physical=None:
        make_batch(arrays, indices, 'cpu', physical))
    monkeypatch.setattr(torch, 'load', lambda path, **kw:
        load(path, **dict(kw, map_location='cpu')))
    return fractions


@pytest.mark.parametrize('variant', ['continuous_student', 'hard_student'])
@pytest.mark.parametrize('steps', [None, 2])
@pytest.mark.parametrize('gpu_memory_bytes', [None, 7])
def test_cuda_memory_never_changes_optimizer_step_budget(tmp_path, small_cfg,
        cuda_branch_on_cpu, monkeypatch, variant, steps, gpu_memory_bytes):
    fixture_cache(tmp_path, small_cfg)
    expected = small_cfg['training']['steps'] if steps is None else steps
    step = torch.optim.AdamW.step
    actual = []
    def bounded_step(optimizer, *args, **kwargs):
        assert len(actual) < expected, 'CUDA memory bytes leaked into the optimizer step budget'
        actual.append(1)
        return step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, 'step', bounded_step)
    result = runner.train_one(small_cfg, tmp_path, variant, 17, 'cuda:0',
        steps=steps, gpu_memory_bytes=gpu_memory_bytes, export=False)
    progress = runner.read_json(tmp_path / 'jobs' / variant / 'seed_17/progress.json')
    assert len(actual) == result['step'] == result['budget'] == progress['budget'] == expected
    assert progress['estimated_remaining_seconds'] == 0
    memory = 4096 * 1024 ** 2 if gpu_memory_bytes is None else gpu_memory_bytes
    assert cuda_branch_on_cpu == [max(memory - 512 * 1024 ** 2, 64 * 1024 ** 2) / (8 * 1024 ** 3)]


def test_cuda_resume_rejects_checkpoint_beyond_requested_steps(tmp_path, small_cfg, cuda_branch_on_cpu):
    fixture_cache(tmp_path, small_cfg)
    runner.train_one(small_cfg, tmp_path, 'hard_student', 17, 'cuda:0',
        steps=2, gpu_memory_bytes=7, export=False)
    with pytest.raises(ValueError, match='beyond requested fixed step budget'):
        runner.train_one(small_cfg, tmp_path, 'hard_student', 17, 'cuda:0',
            steps=1, gpu_memory_bytes=7, export=False)


@pytest.mark.parametrize('actual_steps', [2, 7])
def test_benchmark_requires_exact_requested_step_budget(tmp_path, small_cfg, monkeypatch, actual_steps):
    def spawn(*args, pilot, steps, **kwargs):
        pilot.mkdir(parents=True, exist_ok=True)
        runner.write_json(pilot / 'summary.json', dict(status='completed', step=actual_steps,
            budget=actual_steps, trained_steps=actual_steps, peak_cuda_reserved_bytes=0))
        return SimpleNamespace(wait=lambda: 0), io.StringIO(), pilot
    monkeypatch.setattr(runner, '_spawn_job', spawn)
    result = runner.benchmark(small_cfg, tmp_path / 'fixture.yaml', tmp_path, ROOT, 'cuda:0', steps=2)
    assert result['status'] == ('completed' if actual_steps == 2 else 'failed')
    if actual_steps != 2:
        with pytest.raises(ValueError, match='retained benchmark violates'):
            runner.benchmark(small_cfg, tmp_path / 'fixture.yaml', tmp_path, ROOT, 'cuda:0', steps=2)


def test_contract_freezes_entire_configuration_and_data_root(tmp_path, small_cfg):
    runner.freeze_contract(tmp_path, small_cfg, ROOT)
    runner.freeze_contract(tmp_path, deepcopy(small_cfg), ROOT)
    drift = deepcopy(small_cfg); drift['training']['steps'] += 1
    with pytest.raises(ValueError, match='contract drift'):
        runner.freeze_contract(tmp_path, drift, ROOT)
    with pytest.raises(ValueError, match='contract drift'):
        runner.freeze_contract(tmp_path, small_cfg, tmp_path)


def test_nonblocking_owner_and_interrupted_attempts_are_retained(tmp_path):
    with runner.owner_lock(tmp_path):
        with pytest.raises(RuntimeError, match='another owner'):
            with runner.owner_lock(tmp_path): pass
    assert runner.task_begin(tmp_path, 'teacher', 'a')['attempt'] == 1
    assert runner.task_begin(tmp_path, 'teacher', 'a')['attempt'] == 2
    assert runner.read_json(runner.task_path(tmp_path, 'teacher', 'a') / 'attempt_0001.json')['status'] == 'interrupted'
    runner.task_finish(tmp_path, 'teacher', 'a', 'failed', error='retained negative result')
    assert runner.task_begin(tmp_path, 'teacher', 'a') is None
    assert runner.task_begin(tmp_path, 'teacher', 'a', retry_failed=True)['attempt'] == 3
    assert runner.read_json(runner.task_path(tmp_path, 'teacher', 'a') / 'attempt_0002.json')['error'] == 'retained negative result'


def test_prepare_reads_one_record_and_preserves_each_failure(tmp_path, monkeypatch):
    import src.data.dual_component_tokenizer as data
    rows = [dict(id=f'roi_{i}', join_key='one_native_record') for i in range(3)]
    calls = []
    def producer(root, cfg, group):
        calls.append([r['id'] for r in group])
        yield group[0], dict(eeg=np.ones(2)), None
        yield group[1], None, 'missing physical support'
        # A missing yield must become a failure, not leave the task pending.
    monkeypatch.setattr(data, 'prepare_cached_record', producer)
    result = runner._prepare_record((str(ROOT), {}, str(tmp_path), rows, False))
    assert len(calls) == 1 and result['prepared'] == 1 and result['failed'] == 2
    assert [runner.task_status(tmp_path, 'prepared', r['id'])['status'] for r in rows] == ['completed', 'failed', 'failed']
    runner._prepare_record((str(ROOT), {}, str(tmp_path), rows, False))
    assert len(calls) == 1
    assert runner.read_arrays(tmp_path, 'prepared', rows[0])['eeg'].tolist() == [1, 1]
    with pytest.raises(FileNotFoundError): runner.read_arrays(tmp_path, 'prepared', rows[1])


def test_prepare_write_error_gets_terminal_failure(tmp_path, monkeypatch):
    import src.data.dual_component_tokenizer as data
    row = dict(id='disk_failure', join_key='fixture_record')
    monkeypatch.setattr(data, 'prepare_cached_record', lambda *args: iter([(row, dict(eeg=np.ones(2)), None)]))
    def fail(*args, **kwargs): raise OSError('synthetic disk write failure')
    monkeypatch.setattr(runner, 'write_npz', fail)
    runner._prepare_record((str(ROOT), {}, str(tmp_path), [row], False))
    status = runner.task_status(tmp_path, 'prepared', row['id'])
    assert status['status'] == 'failed' and 'synthetic disk write failure' in status['error']


@pytest.mark.parametrize('variant', ['continuous_student', 'hard_student'])
def test_checkpoint_resume_reproduces_model_optimizer_and_rng(tmp_path, small_cfg, variant):
    fixture_cache(tmp_path, small_cfg)
    continuous = tmp_path / 'full'; resumed = tmp_path / 'resumed'
    runner.train_one(small_cfg, tmp_path, variant, 17, steps=4, job_output=continuous, export=False)
    runner.train_one(small_cfg, tmp_path, variant, 17, steps=2, job_output=resumed, export=False)
    runner.train_one(small_cfg, tmp_path, variant, 17, steps=4, job_output=resumed, export=False)
    a = torch.load(continuous / 'checkpoint.pt', weights_only=False)
    b = torch.load(resumed / 'checkpoint.pt', weights_only=False)
    assert a['step'] == b['step'] == 4
    for key, tensor in a['model_state'].items(): assert torch.equal(tensor, b['model_state'][key]), key
    for key, state in a['optimizer_state']['state'].items():
        for field, value in state.items():
            assert torch.equal(value, b['optimizer_state']['state'][key][field])
    assert torch.equal(a['torch_rng'], b['torch_rng'])
    assert np.array_equal(a['numpy_rng'][1], b['numpy_rng'][1])


@pytest.mark.parametrize('variant', runner.VARIANTS)
def test_all_variants_export_aligned_hard_ids_and_exact_sum(tmp_path, small_cfg, variant):
    rows, arrays = fixture_cache(tmp_path, small_cfg)
    result = runner.train_one(small_cfg, tmp_path, variant, 17, steps=2)
    assert result['status'] == 'completed'
    job = tmp_path / 'jobs' / variant / 'seed_17'
    with np.load(job / 'predictions.npz') as archive:
        assert archive['row_index'].tolist() == list(range(len(rows)))
        for modality in ('eeg', 'hb'):
            assert np.array_equal(archive[modality + '_sum'], archive[modality + '_S'] + archive[modality + '_O'])
            assert archive[modality + '_sum'].shape == arrays[modality + '_u'].shape
            if variant != 'continuous_student':
                assert archive[modality + '_ids_S'].dtype.kind == 'i'
                assert archive[modality + '_ids_O'].shape == (6, 3)
    assert (job / 'codewords.npz').exists() == (variant != 'continuous_student')


def test_cache_rejects_target_algebra_drift(tmp_path, small_cfg):
    _, arrays = fixture_cache(tmp_path, small_cfg)
    np.save(tmp_path / 'cache/eeg_e.npy', arrays['eeg_e'] + 1)
    with pytest.raises(ValueError, match='sum identity'):
        runner.load_cache(tmp_path)


def test_synthetic_model_path_never_reads_measured_arrays(small_cfg, monkeypatch):
    import src.data.dual_component_tokenizer as data
    def forbidden(*args, **kwargs): raise AssertionError('measured reader reached')
    monkeypatch.setattr(data, 'prepare_cached_record', forbidden)
    monkeypatch.setattr(data, 'metadata_inventory', forbidden)
    assert runner.synthetic(small_cfg, numerical_teacher=False)['measured_arrays_read'] == 0
